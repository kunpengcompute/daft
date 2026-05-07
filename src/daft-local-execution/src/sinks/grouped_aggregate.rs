use std::sync::{Arc, Mutex};

use common_daft_config::DaftExecutionConfig;
use common_error::DaftResult;
use common_metrics::ops::NodeType;
use daft_core::prelude::SchemaRef;
use daft_dsl::expr::{
    bound_col,
    bound_expr::{BoundAggExpr, BoundExpr},
};
use daft_micropartition::MicroPartition;
use itertools::Itertools;
use tracing::{Span, instrument};

use super::blocking_sink::{
    BlockingSink, BlockingSinkFinalizeResult, BlockingSinkOutput, BlockingSinkSinkResult,
};
use crate::{
    ExecutionTaskSpawner,
    cardinality::CardinalityProfile,
    pipeline::{InputId, NodeName},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AggStrategy {
    // TODO: This would probably benefit from doing sharded aggs.
    AggThenPartition,
    PartitionThenAgg(usize),
    PartitionOnly,
}

/// Build a `CardinalityProfile` from the row hashes of one or more record
/// batches forming a single morsel's group-key view.
///
/// Hashes are computed at most once per row and forwarded to the profile
/// builder as an `Option<u64>` stream so that null group keys are correctly
/// excluded from the NDV count.
fn profile_record_batches(
    batches: &[daft_recordbatch::RecordBatch],
) -> DaftResult<CardinalityProfile> {
    // Pre-compute the per-batch hashes so we can stream `Option<u64>` items
    // without allocating intermediate vectors per batch.
    let hashes_per_batch = batches
        .iter()
        .map(|t| t.hash_rows())
        .collect::<DaftResult<Vec<_>>>()?;
    let iter = hashes_per_batch.iter().flat_map(|arr| arr.iter());
    Ok(CardinalityProfile::from_hashes(iter))
}

/// Pick the aggregation strategy for a single morsel given its cardinality
/// profile.
///
/// Decision matrix:
/// - `ndv_ratio < high_cardinality_threshold_ratio` → `AggThenPartition`
///   (low cardinality: heavy key reuse pays for partial aggregation).
/// - Otherwise, `top1_ratio > skew_threshold` → `AggThenPartition`
///   (severe skew: a single hot key would overload one partition under the
///   `PartitionThenAgg` path; the partial-aggregate step compresses it
///   before the partition shuffle).
/// - Otherwise → `PartitionThenAgg(partial_agg_threshold)`.
///
/// Setting `skew_threshold` strictly greater than `1.0` disables the skew
/// override and recovers the legacy `ndv_ratio`-only behaviour.
pub(crate) fn decide_agg_strategy(
    profile: &CardinalityProfile,
    high_cardinality_threshold_ratio: f64,
    skew_threshold: f64,
    partial_agg_threshold: usize,
) -> AggStrategy {
    let is_low_cardinality = profile.ndv_ratio < high_cardinality_threshold_ratio;
    let is_skewed = profile.top1_ratio > skew_threshold;
    if is_low_cardinality || is_skewed {
        AggStrategy::AggThenPartition
    } else {
        AggStrategy::PartitionThenAgg(partial_agg_threshold)
    }
}

impl AggStrategy {
    fn execute_strategy(
        &self,
        inner_states: &mut [Option<SinglePartitionAggregateState>],
        input: MicroPartition,
        params: &GroupedAggregateParams,
    ) -> DaftResult<()> {
        match self {
            Self::AggThenPartition => Self::execute_agg_then_partition(inner_states, input, params),
            Self::PartitionThenAgg(threshold) => {
                Self::execute_partition_then_agg(inner_states, input, params, *threshold)
            }
            Self::PartitionOnly => Self::execute_partition_only(inner_states, input, params),
        }
    }

    fn execute_agg_then_partition(
        inner_states: &mut [Option<SinglePartitionAggregateState>],
        input: MicroPartition,
        params: &GroupedAggregateParams,
    ) -> DaftResult<()> {
        let agged = input.agg(
            params.partial_agg_exprs.as_slice(),
            params.group_by.as_slice(),
        )?;
        let partitioned =
            agged.partition_by_hash(params.final_group_by.as_slice(), inner_states.len())?;
        for (p, state) in partitioned.into_iter().zip(inner_states.iter_mut()) {
            let state = state.get_or_insert_default();
            state.partially_aggregated.push(p);
        }
        Ok(())
    }

    fn execute_partition_then_agg(
        inner_states: &mut [Option<SinglePartitionAggregateState>],
        input: MicroPartition,
        params: &GroupedAggregateParams,
        partial_agg_threshold: usize,
    ) -> DaftResult<()> {
        let partitioned =
            input.partition_by_hash(params.group_by.as_slice(), inner_states.len())?;
        for (p, state) in partitioned.into_iter().zip(inner_states.iter_mut()) {
            let state = state.get_or_insert_default();
            if state.unaggregated_size + p.len() >= partial_agg_threshold {
                let mut unaggregated = std::mem::take(&mut state.unaggregated);
                unaggregated.push(p);
                let aggregated = MicroPartition::concat(unaggregated)?.agg(
                    params.partial_agg_exprs.as_slice(),
                    params.group_by.as_slice(),
                )?;
                state.partially_aggregated.push(aggregated);
                state.unaggregated_size = 0;
            } else {
                state.unaggregated_size += p.len();
                state.unaggregated.push(p);
            }
        }
        Ok(())
    }

    fn execute_partition_only(
        inner_states: &mut [Option<SinglePartitionAggregateState>],
        input: MicroPartition,
        params: &GroupedAggregateParams,
    ) -> DaftResult<()> {
        let partitioned =
            input.partition_by_hash(params.group_by.as_slice(), inner_states.len())?;
        for (p, state) in partitioned.into_iter().zip(inner_states.iter_mut()) {
            let state = state.get_or_insert_default();
            state.unaggregated_size += p.len();
            state.unaggregated.push(p);
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct SinglePartitionAggregateState {
    partially_aggregated: Vec<MicroPartition>,
    unaggregated: Vec<MicroPartition>,
    unaggregated_size: usize,
}

pub(crate) enum GroupedAggregateState {
    Accumulating {
        inner_states: Vec<Option<SinglePartitionAggregateState>>,
        strategy: Option<AggStrategy>,
        partial_agg_threshold: usize,
        high_cardinality_threshold_ratio: f64,
        skew_threshold: f64,
    },
    Done,
}

impl GroupedAggregateState {
    fn new(
        num_partitions: usize,
        partial_agg_threshold: usize,
        high_cardinality_threshold_ratio: f64,
        skew_threshold: f64,
    ) -> Self {
        let inner_states = (0..num_partitions).map(|_| None).collect::<Vec<_>>();
        Self::Accumulating {
            inner_states,
            strategy: None,
            partial_agg_threshold,
            high_cardinality_threshold_ratio,
            skew_threshold,
        }
    }

    fn push(
        &mut self,
        input: MicroPartition,
        params: &GroupedAggregateParams,
        global_strategy_lock: &Arc<Mutex<Option<AggStrategy>>>,
    ) -> DaftResult<()> {
        let Self::Accumulating {
            inner_states,
            strategy,
            partial_agg_threshold,
            high_cardinality_threshold_ratio,
            skew_threshold,
        } = self
        else {
            panic!("GroupedAggregateSink should be in Accumulating state");
        };

        // If we have determined a strategy, execute it.
        if let Some(strategy) = strategy {
            strategy.execute_strategy(inner_states, input, params)?;
        } else {
            // Otherwise, determine the strategy and execute
            let decided_strategy = Self::determine_agg_strategy(
                &input,
                params,
                *high_cardinality_threshold_ratio,
                *skew_threshold,
                *partial_agg_threshold,
                strategy,
                global_strategy_lock,
            )?;
            decided_strategy.execute_strategy(inner_states, input, params)?;
        }
        Ok(())
    }

    fn determine_agg_strategy(
        input: &MicroPartition,
        params: &GroupedAggregateParams,
        high_cardinality_threshold_ratio: f64,
        skew_threshold: f64,
        partial_agg_threshold: usize,
        local_strategy_cache: &mut Option<AggStrategy>,
        global_strategy_lock: &Arc<Mutex<Option<AggStrategy>>>,
    ) -> DaftResult<AggStrategy> {
        let mut global_strategy = global_strategy_lock.lock().unwrap();
        // If some other worker has determined a strategy, use that.
        if let Some(global_strat) = global_strategy.as_ref() {
            *local_strategy_cache = Some(global_strat.clone());
            return Ok(global_strat.clone());
        }

        // Else determine the strategy from the cardinality profile of this morsel's
        // group keys. The previous implementation collected all hashes into a
        // HashSet just to count distinct values; the unified `CardinalityProfile`
        // performs the same job and additionally exposes `top1_ratio`, which the
        // skew-aware override below leverages to keep heavy hot-key workloads on
        // the partial-aggregate-friendly `AggThenPartition` path.
        let groupby = input.eval_expression_list(params.group_by.as_slice())?;
        let profile = profile_record_batches(groupby.record_batches())?;
        let decided_strategy = decide_agg_strategy(
            &profile,
            high_cardinality_threshold_ratio,
            skew_threshold,
            partial_agg_threshold,
        );

        *local_strategy_cache = Some(decided_strategy.clone());
        *global_strategy = Some(decided_strategy.clone());
        Ok(decided_strategy)
    }

    fn finalize(&mut self) -> Vec<Option<SinglePartitionAggregateState>> {
        let res = if let Self::Accumulating { inner_states, .. } = self {
            std::mem::take(inner_states)
        } else {
            panic!("GroupedAggregateSink should be in Accumulating state");
        };
        *self = Self::Done;
        res
    }
}

struct GroupedAggregateParams {
    // The original aggregations and group by expressions
    original_aggregations: Vec<BoundAggExpr>,
    group_by: Vec<BoundExpr>,
    // The expressions for to be used for partial aggregation
    partial_agg_exprs: Vec<BoundAggExpr>,
    // The expressions for the final aggregation
    final_agg_exprs: Vec<BoundAggExpr>,
    final_group_by: Vec<BoundExpr>,
    final_projections: Vec<BoundExpr>,
}

pub struct GroupedAggregateSink {
    grouped_aggregate_params: Arc<GroupedAggregateParams>,
    partial_agg_threshold: usize,
    high_cardinality_threshold_ratio: f64,
    skew_threshold: f64,
    global_strategy_lock: Arc<Mutex<Option<AggStrategy>>>,
}

impl GroupedAggregateSink {
    pub fn new(
        aggregations: &[BoundAggExpr],
        group_by: &[BoundExpr],
        input_schema: &SchemaRef,
        cfg: &DaftExecutionConfig,
    ) -> DaftResult<Self> {
        let (partial_agg_exprs, final_agg_exprs, final_projections) =
            daft_local_plan::agg::populate_aggregation_stages_bound(
                aggregations,
                input_schema,
                group_by,
            )?;

        // MapGroups aggregations cannot be decomposed into partial / final stages and
        // must see the full group in a single pass. Detect this case so that we force
        // a partition-only strategy and run the original aggregations during the
        // final aggregation step.
        let has_map_groups = aggregations
            .iter()
            .any(|agg| matches!(agg.as_ref(), daft_dsl::AggExpr::MapGroups { .. }));

        let final_group_by = if !partial_agg_exprs.is_empty() {
            group_by
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let field = e.as_ref().to_field(input_schema)?;
                    Ok(BoundExpr::new_unchecked(bound_col(i, field)))
                })
                .collect::<DaftResult<Vec<_>>>()?
        } else {
            group_by.to_vec()
        };

        let strategy = if has_map_groups {
            // Always use partition-only for MapGroups so that we only hash-partition
            // the data by the group keys and then run the original MapGroups
            // aggregation once per partition in `finalize`.
            Some(AggStrategy::PartitionOnly)
        } else if partial_agg_exprs.is_empty() && !final_agg_exprs.is_empty() {
            Some(AggStrategy::PartitionOnly)
        } else {
            None
        };

        Ok(Self {
            grouped_aggregate_params: Arc::new(GroupedAggregateParams {
                original_aggregations: aggregations.to_vec(),
                group_by: group_by.to_vec(),
                partial_agg_exprs,
                final_agg_exprs,
                final_group_by,
                final_projections,
            }),
            partial_agg_threshold: cfg.partial_aggregation_threshold,
            high_cardinality_threshold_ratio: cfg.high_cardinality_aggregation_threshold,
            skew_threshold: cfg.high_cardinality_skew_threshold,
            global_strategy_lock: Arc::new(Mutex::new(strategy)),
        })
    }

    fn num_partitions(&self) -> usize {
        self.max_concurrency()
    }
}

impl BlockingSink for GroupedAggregateSink {
    type State = GroupedAggregateState;
    #[instrument(skip_all, name = "GroupedAggregateSink::sink")]
    fn sink(
        &self,
        input: MicroPartition,
        mut state: Self::State,
        _runtime_stats: Arc<Self::Stats>,
        spawner: &ExecutionTaskSpawner,
    ) -> BlockingSinkSinkResult<Self> {
        let params = self.grouped_aggregate_params.clone();
        let strategy_lock = self.global_strategy_lock.clone();
        spawner
            .spawn(
                async move {
                    state.push(input, &params, &strategy_lock)?;
                    Ok(state)
                },
                Span::current(),
            )
            .into()
    }

    #[instrument(skip_all, name = "GroupedAggregateSink::finalize")]
    fn finalize(
        &self,
        states: Vec<Self::State>,
        spawner: &ExecutionTaskSpawner,
    ) -> BlockingSinkFinalizeResult {
        let params = self.grouped_aggregate_params.clone();
        let num_partitions = self.num_partitions();
        spawner
            .spawn(
                async move {
                    let mut state_iters = states
                        .into_iter()
                        .map(|mut state| state.finalize().into_iter())
                        .collect::<Vec<_>>();

                    let mut per_partition_finalize_tasks = tokio::task::JoinSet::new();
                    for _ in 0..num_partitions {
                        let per_partition_state = state_iters
                            .iter_mut()
                            .map(|state| {
                                state.next().expect(
                                "GroupedAggregateState should have SinglePartitionAggregateState",
                            )
                            })
                            .collect::<Vec<_>>();
                        let params = params.clone();
                        per_partition_finalize_tasks.spawn(async move {
                            let mut unaggregated = vec![];
                            let mut partially_aggregated = vec![];
                            for state in per_partition_state.into_iter().flatten() {
                                unaggregated.extend(state.unaggregated);
                                partially_aggregated.extend(state.partially_aggregated);
                            }

                            // If we have no partially aggregated partitions, aggregate the unaggregated partitions using the original aggregations
                            if params.partial_agg_exprs.is_empty() && !unaggregated.is_empty() {
                                let concated = MicroPartition::concat(unaggregated)?;
                                let agged = concated
                                    .agg(&params.original_aggregations, &params.group_by)?;
                                Ok(agged)
                            }
                            // If we have no unaggregated partitions, finalize the partially aggregated partitions
                            else if unaggregated.is_empty() {
                                let concated = MicroPartition::concat(partially_aggregated)?;
                                let agged = concated
                                    .agg(&params.final_agg_exprs, &params.final_group_by)?;
                                let projected =
                                    agged.eval_expression_list(&params.final_projections)?;
                                Ok(projected)
                            }
                            // Otherwise, partially aggregate the unaggregated partitions, concatenate them with the partially aggregated partitions, and finalize the result.
                            else {
                                let leftover_partial_agg = MicroPartition::concat(unaggregated)?
                                    .agg(&params.partial_agg_exprs, &params.group_by)?;
                                partially_aggregated.push(leftover_partial_agg);
                                let concated = MicroPartition::concat(partially_aggregated)?;
                                let agged = concated
                                    .agg(&params.final_agg_exprs, &params.final_group_by)?;
                                let projected =
                                    agged.eval_expression_list(&params.final_projections)?;
                                Ok(projected)
                            }
                        });
                    }
                    let results = per_partition_finalize_tasks
                        .join_all()
                        .await
                        .into_iter()
                        .collect::<DaftResult<Vec<_>>>()?;
                    Ok(BlockingSinkOutput::Partitions(results))
                },
                Span::current(),
            )
            .into()
    }

    fn name(&self) -> NodeName {
        "GroupedAggregate".into()
    }

    fn op_type(&self) -> NodeType {
        NodeType::GroupByAgg
    }

    fn multiline_display(&self) -> Vec<String> {
        let mut display = vec![];
        display.push(format!(
            "GroupedAggregate: {}",
            self.grouped_aggregate_params
                .original_aggregations
                .iter()
                .map(|e| e.to_string())
                .join(", ")
        ));
        display.push(format!(
            "Group by: {}",
            self.grouped_aggregate_params
                .group_by
                .iter()
                .map(|e| e.to_string())
                .join(", ")
        ));
        display
    }

    fn make_state(&self, _input_id: InputId) -> DaftResult<Self::State> {
        Ok(GroupedAggregateState::new(
            self.num_partitions(),
            self.partial_agg_threshold,
            self.high_cardinality_threshold_ratio,
            self.skew_threshold,
        ))
    }
}

#[cfg(test)]
mod tests {
    //! Integration tests for the cardinality-aware strategy router.
    //!
    //! These tests pin down the contract that `determine_agg_strategy` used to
    //! enforce with its inline `HashSet` counter:
    //! - high `ndv_ratio` -> `PartitionThenAgg`
    //! - low `ndv_ratio`  -> `AggThenPartition`
    //!
    //! They additionally exercise the threshold boundary, which the previous
    //! implementation only covered indirectly through end-to-end aggregation
    //! tests.

    use super::*;
    use crate::cardinality::CardinalityProfile;

    fn profile_with_ratio(num_rows: usize, num_distinct: usize) -> CardinalityProfile {
        let hashes = (0..num_rows).map(|i| Some((i % num_distinct.max(1)) as u64));
        CardinalityProfile::from_hashes(hashes)
    }

    /// Defaults that effectively disable the skew override so existing
    /// `decide_agg_strategy` callers exercise the original `ndv_ratio`-only
    /// path. `top1_ratio` is bounded by 1.0 so a threshold of 1.0 with strict
    /// `>` never triggers.
    const SKEW_DISABLED: f64 = 1.0;

    #[test]
    fn test_decide_agg_strategy_low_cardinality_picks_agg_then_partition() {
        // 1000 rows, 10 distinct -> ratio 0.01 (well below default 0.8)
        let profile = profile_with_ratio(1000, 10);
        let strategy = decide_agg_strategy(&profile, 0.8, SKEW_DISABLED, 10_000);
        assert_eq!(strategy, AggStrategy::AggThenPartition);
    }

    #[test]
    fn test_decide_agg_strategy_high_cardinality_picks_partition_then_agg() {
        // 1000 rows, 1000 distinct -> ratio 1.0 (>= 0.8)
        let profile = profile_with_ratio(1000, 1000);
        let strategy = decide_agg_strategy(&profile, 0.8, SKEW_DISABLED, 10_000);
        assert_eq!(strategy, AggStrategy::PartitionThenAgg(10_000));
    }

    #[test]
    fn test_decide_agg_strategy_threshold_inclusive() {
        // Build a profile whose ndv_ratio is exactly 0.5 by construction.
        let profile = profile_with_ratio(1000, 500);
        assert!((profile.ndv_ratio - 0.5).abs() < 1e-9);
        // At threshold 0.5 the decision must flip into PartitionThenAgg (>=).
        assert_eq!(
            decide_agg_strategy(&profile, 0.5, SKEW_DISABLED, 42),
            AggStrategy::PartitionThenAgg(42)
        );
        // Just above threshold -> still PartitionThenAgg
        assert_eq!(
            decide_agg_strategy(&profile, 0.49, SKEW_DISABLED, 42),
            AggStrategy::PartitionThenAgg(42)
        );
        // Just below threshold -> AggThenPartition
        assert_eq!(
            decide_agg_strategy(&profile, 0.51, SKEW_DISABLED, 42),
            AggStrategy::AggThenPartition
        );
    }

    #[test]
    fn test_decide_agg_strategy_propagates_partial_agg_threshold() {
        let profile = profile_with_ratio(100, 100); // ratio 1.0
        let strategy = decide_agg_strategy(&profile, 0.8, SKEW_DISABLED, 1234);
        assert_eq!(strategy, AggStrategy::PartitionThenAgg(1234));
    }

    #[test]
    fn test_decide_agg_strategy_empty_input_low_cardinality_default() {
        // An empty profile has ndv_ratio == 0.0; we must NOT classify the
        // empty input as high-cardinality (which would trigger the heavier
        // PartitionThenAgg path on a no-op morsel).
        let profile = CardinalityProfile::default();
        assert_eq!(profile.ndv_ratio, 0.0);
        assert_eq!(
            decide_agg_strategy(&profile, 0.8, SKEW_DISABLED, 10_000),
            AggStrategy::AggThenPartition
        );
    }

    /// Build a `CardinalityProfile` whose `top1_ratio` matches the requested
    /// `top1_share` exactly. The remaining mass is spread across `cold_ndv`
    /// distinct cold keys so we can drive the `ndv_ratio` independently of
    /// the skew. Returns `(profile, ndv_ratio, top1_ratio)` for sanity
    /// checking inside the test.
    fn profile_with_skew(
        num_rows: usize,
        top1_share: f64,
        cold_ndv: usize,
    ) -> CardinalityProfile {
        let hot_count = (num_rows as f64 * top1_share).round() as usize;
        let cold_count = num_rows - hot_count;
        let cold_ndv = cold_ndv.max(1);
        let hashes = (0..hot_count)
            .map(|_| Some(0u64))
            .chain(
                (0..cold_count).map(move |i| Some(1u64 + (i % cold_ndv) as u64)),
            );
        CardinalityProfile::from_hashes(hashes)
    }

    #[test]
    fn test_decide_agg_strategy_high_skew_overrides_partition_then_agg() {
        // 1000 rows, 90% on a single hot key + 100 distinct cold keys.
        // ndv_ratio is high (101 / 1000 = 0.101) so without the skew gate
        // we would pick PartitionThenAgg; the skew override (top1_ratio = 0.9
        // > threshold 0.5) must flip the decision to AggThenPartition.
        let profile = profile_with_skew(1000, 0.9, 100);
        assert!(profile.top1_ratio > 0.5);
        assert!(profile.ndv_ratio > 0.05);
        let strategy = decide_agg_strategy(&profile, 0.05, 0.5, 10_000);
        assert_eq!(strategy, AggStrategy::AggThenPartition);
    }

    #[test]
    fn test_decide_agg_strategy_skew_threshold_strict_greater_than() {
        // top1_ratio sits exactly at the threshold; strict `>` means we do
        // NOT trigger the skew override. Pair it with high ndv_ratio so the
        // decision must be PartitionThenAgg.
        let profile = profile_with_skew(1000, 0.5, 50);
        // The construction above produces top1 = 500 / 1000 = 0.5 exactly.
        assert!((profile.top1_ratio - 0.5).abs() < 1e-9);
        assert!(profile.ndv_ratio > 0.05);
        let strategy = decide_agg_strategy(&profile, 0.05, 0.5, 10_000);
        assert_eq!(strategy, AggStrategy::PartitionThenAgg(10_000));
    }

    #[test]
    fn test_decide_agg_strategy_skew_disabled_above_one() {
        // A skew threshold above the maximum possible `top1_ratio` (1.0) must
        // never trigger the override, even on a fully-skewed input.
        let profile = profile_with_skew(1000, 1.0, 1);
        assert!((profile.top1_ratio - 1.0).abs() < 1e-9);
        // ndv_ratio = 1 / 1000 = 0.001 -> low cardinality regardless.
        // Use a high-card threshold of 0.0 so the ONLY reason to pick
        // AggThenPartition would be the skew gate (which is disabled).
        let strategy = decide_agg_strategy(&profile, 0.0, 1.5, 10_000);
        assert_eq!(strategy, AggStrategy::PartitionThenAgg(10_000));
    }

    #[test]
    fn test_decide_agg_strategy_low_skew_obeys_ndv_only() {
        // 1000 rows uniformly across 100 distinct keys -> top1_ratio = 0.01.
        // ndv_ratio = 0.1 is above the high-card threshold so the decision
        // must be PartitionThenAgg.
        let profile = profile_with_ratio(1000, 100);
        assert!(profile.top1_ratio < 0.5);
        let strategy = decide_agg_strategy(&profile, 0.05, 0.5, 10_000);
        assert_eq!(strategy, AggStrategy::PartitionThenAgg(10_000));
    }

    #[test]
    fn test_decide_agg_strategy_skew_does_not_override_low_cardinality() {
        // When `ndv_ratio` is already low we always take AggThenPartition;
        // the skew gate must agree (or be a no-op) and never escalate the
        // path back to PartitionThenAgg.
        let profile = profile_with_ratio(1000, 5);
        assert!(profile.ndv_ratio < 0.05);
        // Use a deliberately tight skew threshold; the decision must remain
        // AggThenPartition because the low-card branch dominates.
        let strategy = decide_agg_strategy(&profile, 0.05, 0.001, 10_000);
        assert_eq!(strategy, AggStrategy::AggThenPartition);
    }
}
