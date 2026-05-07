use std::sync::Arc;

use common_error::DaftResult;
use daft_core::prelude::{IntoSeries, UInt64Array};
use daft_logical_plan::JoinType;
use daft_micropartition::MicroPartition;
use daft_recordbatch::{ProbeState, RecordBatch};
use futures::{StreamExt, stream};

use crate::join::{
    hash_join::{HashJoinParams, HashJoinProbeState},
    index_bitmap::IndexBitmapBuilder,
    probe_dedup::{walk_probe_exists, walk_probe_matches},
};

pub(crate) fn probe_anti_semi(
    input: &MicroPartition,
    probe_state: &ProbeState,
    params: &HashJoinParams,
) -> DaftResult<MicroPartition> {
    let is_semi = params.join_type == JoinType::Semi;

    let input_tables = input.record_batches();
    let mut input_idxs = vec![vec![]; input_tables.len()];
    for (probe_side_table_idx, table) in input_tables.iter().enumerate() {
        let join_keys = table.eval_expression_list(&params.probe_on)?;
        walk_probe_exists(
            probe_state,
            &join_keys,
            params.probe_dedup,
            |probe_row_idx, matched| {
                // 1. If this is a semi join, we keep the row if it matches.
                // 2. If this is an anti join, we keep the row if it doesn't match.
                if is_semi == matched {
                    input_idxs[probe_side_table_idx].push(probe_row_idx as u64);
                }
                Ok(())
            },
        )?;
    }
    let probe_side_tables = input_idxs
        .into_iter()
        .zip(input_tables.iter())
        .map(|(idxs, table)| {
            let idxs_arr = UInt64Array::from_vec("idxs", idxs);
            table.take(&idxs_arr)
        })
        .collect::<DaftResult<Vec<_>>>()?;
    Ok(MicroPartition::new_loaded(
        probe_side_tables[0].schema.clone(),
        Arc::new(probe_side_tables),
        None,
    ))
}

pub(crate) fn probe_anti_semi_with_bitmap(
    input: &MicroPartition,
    bitmap_builder: &mut IndexBitmapBuilder,
    probe_state: &ProbeState,
    params: &HashJoinParams,
) -> DaftResult<()> {
    for table in input.record_batches() {
        let join_keys = table.eval_expression_list(&params.probe_on)?;
        walk_probe_matches(
            probe_state,
            &join_keys,
            params.probe_dedup,
            |_probe_row_idx, maybe_matches| {
                // Bitmap-based anti/semi join: only the build-side bitmap
                // matters; no probe rows are emitted here. A probe row with
                // no match (the `None` arm) is a no-op.
                if let Some(matches) = maybe_matches {
                    for (build_table_idx, build_row_idx) in matches {
                        bitmap_builder
                            .mark_used(build_table_idx as usize, build_row_idx as usize);
                    }
                }
                Ok(())
            },
        )?;
    }
    Ok(())
}

pub(crate) async fn finalize_anti_semi(
    states: Vec<HashJoinProbeState>,
    is_semi: bool,
) -> DaftResult<Option<MicroPartition>> {
    let mut states_iter = states.into_iter();
    let first_state = states_iter
        .next()
        .expect("At least one state should be present for anti/semi join finalize");
    let first_tables = first_state.probe_state.get_record_batches();
    let first_bitmap = first_state
        .bitmap_builder
        .expect("Bitmap builder should be set for anti/semi join finalize")
        .build();

    let mut merged_bitmap = {
        let bitmaps = stream::once(async move { first_bitmap })
            .chain(stream::iter(states_iter).map(|s| {
                s.bitmap_builder
                    .expect("Bitmap builder should be set for anti/semi join finalize")
                    .build()
            }))
            .collect::<Vec<_>>()
            .await;

        bitmaps.into_iter().fold(None, |acc, x| match acc {
            None => Some(x),
            Some(acc) => Some(acc.merge(&x)),
        })
    }
    .expect("At least one bitmap should be present for anti/semi join finalize");

    // The bitmap marks matched rows as 0, so we need to negate it if we are doing semi join, i.e. the matched rows become 1 so that
    // we can we can keep them in the final result.
    if is_semi {
        merged_bitmap = merged_bitmap.negate();
    }

    let leftovers = merged_bitmap
        .convert_to_boolean_arrays()
        .zip(first_tables.iter())
        .map(|(bitmap, table)| table.mask_filter(&bitmap.into_series()))
        .collect::<DaftResult<Vec<_>>>()?;
    let build_side_table = RecordBatch::concat(&leftovers)?;

    Ok(Some(MicroPartition::new_loaded(
        build_side_table.schema.clone(),
        Arc::new(vec![build_side_table]),
        None,
    )))
}

#[cfg(test)]
mod tests {
    //! Equivalence tests for the anti/semi-join probe dedup paths.
    //!
    //! Anti/semi joins emit only one row per probe input (or zero), so the
    //! invariant we pin is that the *kept* probe row indices are identical
    //! whether dedup is on or off. The bitmap-tracked path additionally
    //! requires the build-side bitmap to converge to the same state.

    use daft_core::prelude::{DataType, Field, Int64Array};
    use daft_recordbatch::make_probeable_builder;

    use super::*;
    use crate::join::probe_dedup::{ProbeDedupTuning, walk_probe_exists, walk_probe_matches};

    fn rb_from_int_col(name: &str, values: Vec<Option<i64>>) -> RecordBatch {
        let arr = Int64Array::from_iter(Field::new(name, DataType::Int64), values);
        RecordBatch::from_nonempty_columns(vec![arr.into_series()]).unwrap()
    }

    fn build_probe_state(build_keys: Vec<Option<i64>>) -> ProbeState {
        let build_table = rb_from_int_col("k", build_keys);
        let mut builder = make_probeable_builder(build_table.schema.clone(), None, true).unwrap();
        builder.add_table(&build_table).unwrap();
        ProbeState::new(builder.build(), vec![build_table])
    }

    fn force_dedup() -> ProbeDedupTuning {
        ProbeDedupTuning {
            enabled: true,
            min_rows: 1,
            unique_ratio_max: 1.0,
        }
    }

    fn force_direct() -> ProbeDedupTuning {
        ProbeDedupTuning::disabled()
    }

    /// Mirrors `probe_anti_semi`'s emit logic against the runner so the
    /// kept-row order is preserved across both dedup tunings.
    fn run_anti_semi_kept_idxs(
        probe_state: &ProbeState,
        join_keys: &RecordBatch,
        is_semi: bool,
        tuning: ProbeDedupTuning,
    ) -> Vec<u64> {
        let mut kept = Vec::new();
        walk_probe_exists(probe_state, join_keys, tuning, |probe_row_idx, matched| {
            if is_semi == matched {
                kept.push(probe_row_idx as u64);
            }
            Ok(())
        })
        .unwrap();
        kept
    }

    fn run_anti_semi_bitmap(
        probe_state: &ProbeState,
        join_keys: &RecordBatch,
        tuning: ProbeDedupTuning,
    ) -> IndexBitmapBuilder {
        let mut bitmap_builder = IndexBitmapBuilder::new(probe_state.get_record_batches());
        walk_probe_matches(
            probe_state,
            join_keys,
            tuning,
            |_probe_row_idx, maybe_matches| {
                if let Some(matches) = maybe_matches {
                    for (build_table_idx, build_row_idx) in matches {
                        bitmap_builder
                            .mark_used(build_table_idx as usize, build_row_idx as usize);
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        bitmap_builder
    }

    fn assert_anti_semi_match(
        probe_state: &ProbeState,
        join_keys: &RecordBatch,
        is_semi: bool,
    ) {
        let direct = run_anti_semi_kept_idxs(probe_state, join_keys, is_semi, force_direct());
        let dedup = run_anti_semi_kept_idxs(probe_state, join_keys, is_semi, force_dedup());
        assert_eq!(
            direct, dedup,
            "kept probe-side indices must match across direct/dedup paths"
        );
    }

    fn assert_anti_semi_bitmap_match(probe_state: &ProbeState, join_keys: &RecordBatch) {
        let bm_d = run_anti_semi_bitmap(probe_state, join_keys, force_direct());
        let bm_p = run_anti_semi_bitmap(probe_state, join_keys, force_dedup());
        let bools_d: Vec<_> = bm_d.build().convert_to_boolean_arrays().collect();
        let bools_p: Vec<_> = bm_p.build().convert_to_boolean_arrays().collect();
        assert_eq!(bools_d.len(), bools_p.len());
        for (a, b) in bools_d.iter().zip(bools_p.iter()) {
            assert_eq!(a.len(), b.len());
            for i in 0..a.len() {
                assert_eq!(a.get(i), b.get(i), "bitmap divergence at row {i}");
            }
        }
    }

    #[test]
    fn test_semi_dedup_repeated_keys() {
        let probe_state = build_probe_state(vec![Some(10), Some(20), Some(30)]);
        let probe_keys =
            rb_from_int_col("k", vec![Some(20), Some(10), Some(20), Some(99), Some(20)]);
        assert_anti_semi_match(&probe_state, &probe_keys, /* is_semi */ true);
    }

    #[test]
    fn test_anti_dedup_repeated_keys() {
        let probe_state = build_probe_state(vec![Some(10), Some(20), Some(30)]);
        let probe_keys =
            rb_from_int_col("k", vec![Some(20), Some(10), Some(20), Some(99), Some(20)]);
        assert_anti_semi_match(&probe_state, &probe_keys, /* is_semi */ false);
    }

    #[test]
    fn test_semi_dedup_with_null_probe_keys() {
        // Null probe keys must not match, so the semi join drops them and
        // the anti join keeps them.
        let probe_state = build_probe_state(vec![Some(1), Some(2)]);
        let probe_keys = rb_from_int_col(
            "k",
            vec![Some(1), None, Some(2), None, Some(99), Some(1)],
        );
        assert_anti_semi_match(&probe_state, &probe_keys, true);
        assert_anti_semi_match(&probe_state, &probe_keys, false);
    }

    #[test]
    fn test_semi_dedup_heavy_skew() {
        let probe_state = build_probe_state(vec![Some(7), Some(99)]);
        let probe_keys = rb_from_int_col("k", (0..1000).map(|_| Some(7i64)).collect());
        assert_anti_semi_match(&probe_state, &probe_keys, true);
    }

    #[test]
    fn test_anti_semi_with_bitmap_repeated_keys() {
        let probe_state = build_probe_state(vec![Some(10), Some(20), Some(30), Some(20)]);
        let probe_keys =
            rb_from_int_col("k", vec![Some(20), Some(10), Some(20), Some(30), Some(20)]);
        assert_anti_semi_bitmap_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_anti_semi_with_bitmap_heavy_skew() {
        let probe_state = build_probe_state(vec![Some(7), Some(7), Some(99)]);
        let probe_keys = rb_from_int_col("k", (0..1000).map(|_| Some(7i64)).collect());
        assert_anti_semi_bitmap_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_anti_semi_with_bitmap_no_matches() {
        let probe_state = build_probe_state(vec![Some(1), Some(2), Some(3)]);
        let probe_keys = rb_from_int_col("k", vec![Some(99); 50]);
        assert_anti_semi_bitmap_match(&probe_state, &probe_keys);
    }
}
