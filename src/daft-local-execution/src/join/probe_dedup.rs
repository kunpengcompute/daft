//! Batch-level deduplication helper for the hash join probe phase.
//!
//! In low-cardinality probe inputs many rows share the same key. Probing the
//! build-side hash table once per *unique* key and then expanding the result
//! back to all original row positions can dramatically reduce CPU cost.
//!
//! This module is intentionally pure: it operates on a slice of optional row
//! hashes and produces a deterministic mapping from original rows to a compact
//! sequence of unique keys. It does not perform key equality verification —
//! that remains the responsibility of the caller (the hash table probe).

use std::collections::HashMap;

use common_daft_config::DaftExecutionConfig;
use common_error::DaftResult;
use daft_core::prelude::UInt64Array;
use daft_recordbatch::{ProbeState, RecordBatch};

/// Runtime tuning for probe-side dedup. Captured from `DaftExecutionConfig`
/// once at operator construction so the hot probe path does not need to chase
/// references into the global config.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeDedupTuning {
    pub enabled: bool,
    pub min_rows: usize,
    pub unique_ratio_max: f64,
}

impl ProbeDedupTuning {
    pub fn from_exec_config(cfg: &DaftExecutionConfig) -> Self {
        Self {
            enabled: cfg.enable_hash_join_probe_dedup,
            min_rows: cfg.low_cardinality_min_rows,
            // The probe batch's `unique_ratio` is exactly its NDV / num_rows;
            // we reuse the unified low-cardinality threshold so groupby and
            // hash join share the same notion of "low cardinality".
            unique_ratio_max: cfg.low_cardinality_ndv_ratio_threshold,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            min_rows: 0,
            unique_ratio_max: 0.0,
        }
    }
}

/// Result of deduplicating a probe batch by row hash.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeBatchDedup {
    /// One entry per original input row.
    /// - `None` means the row's key was null and must be passed through to the
    ///   join's null-handling path (no probe is performed for it).
    /// - `Some(k)` means the row's key matches `unique_hashes[k]`.
    pub row_to_unique: Vec<Option<usize>>,
    /// Unique non-null hashes in first-occurrence order. Stable ordering keeps
    /// the dedup output deterministic and lets callers reuse a fixed indexing
    /// scheme (e.g. when materializing equality-check key columns).
    pub unique_hashes: Vec<u64>,
    /// Indices of the *first* original row that produced each entry of
    /// `unique_hashes`. Callers use these to build a compact key batch for
    /// equality verification against build-side rows.
    pub unique_first_row: Vec<usize>,
}

impl ProbeBatchDedup {
    /// Build a dedup map from an iterator of optional row hashes.
    pub fn build<I>(hashes: I) -> Self
    where
        I: IntoIterator<Item = Option<u64>>,
    {
        let iter = hashes.into_iter();
        let (lower, _) = iter.size_hint();
        let mut row_to_unique = Vec::with_capacity(lower);
        let mut unique_hashes = Vec::with_capacity(lower);
        let mut unique_first_row = Vec::with_capacity(lower);
        let mut seen: HashMap<u64, usize> = HashMap::with_capacity(lower);
        for (row_idx, h) in iter.enumerate() {
            match h {
                None => row_to_unique.push(None),
                Some(h) => {
                    let dedup_idx = match seen.get(&h) {
                        Some(idx) => *idx,
                        None => {
                            let idx = unique_hashes.len();
                            unique_hashes.push(h);
                            unique_first_row.push(row_idx);
                            seen.insert(h, idx);
                            idx
                        }
                    };
                    row_to_unique.push(Some(dedup_idx));
                }
            }
        }
        Self {
            row_to_unique,
            unique_hashes,
            unique_first_row,
        }
    }

    pub fn num_rows(&self) -> usize {
        self.row_to_unique.len()
    }

    pub fn num_unique(&self) -> usize {
        self.unique_hashes.len()
    }

    /// Ratio of unique non-null keys over total rows. A value close to `1.0`
    /// means dedup will not save much; close to `0.0` means high reuse and
    /// dedup is likely beneficial.
    pub fn unique_ratio(&self) -> f64 {
        if self.row_to_unique.is_empty() {
            0.0
        } else {
            self.num_unique() as f64 / self.row_to_unique.len() as f64
        }
    }

    /// Expand a per-unique-key value (e.g. probe result index) back to the
    /// original row order. Rows whose key was null are mapped via
    /// `null_value()`. The closure is invoked at most once per unique key.
    pub fn expand_to_rows<T, F, NF>(&self, mut value_for_unique: F, mut null_value: NF) -> Vec<T>
    where
        F: FnMut(usize) -> T,
        NF: FnMut() -> T,
        T: Clone,
    {
        let mut materialized: Vec<Option<T>> = vec![None; self.unique_hashes.len()];
        let mut out = Vec::with_capacity(self.row_to_unique.len());
        for &slot in &self.row_to_unique {
            match slot {
                None => out.push(null_value()),
                Some(idx) => {
                    if materialized[idx].is_none() {
                        materialized[idx] = Some(value_for_unique(idx));
                    }
                    out.push(materialized[idx].clone().unwrap());
                }
            }
        }
        out
    }
}

/// Threshold-based decision: should we use dedup for this batch?
///
/// `min_rows` guards against tiny batches where the hashmap overhead dominates.
/// `unique_ratio_max` ensures dedup is only used when there is meaningful
/// repetition in the probe input.
pub fn should_use_dedup(
    num_rows: usize,
    unique_ratio: f64,
    min_rows: usize,
    unique_ratio_max: f64,
) -> bool {
    num_rows >= min_rows && unique_ratio <= unique_ratio_max
}

/// Decide whether the probe batch is a good candidate for dedup. Combines the
/// global feature flag with a runtime sample-size + unique-ratio gate so the
/// per-batch hash cost is paid only on inputs that are likely to benefit.
pub(crate) fn try_build_probe_dedup(
    join_keys: &RecordBatch,
    tuning: ProbeDedupTuning,
) -> DaftResult<Option<ProbeBatchDedup>> {
    if !tuning.enabled || join_keys.len() < tuning.min_rows {
        return Ok(None);
    }
    let hashes = join_keys.hash_rows()?;
    let dedup = ProbeBatchDedup::build(hashes.iter());
    if should_use_dedup(
        dedup.num_rows(),
        dedup.unique_ratio(),
        tuning.min_rows,
        tuning.unique_ratio_max,
    ) {
        Ok(Some(dedup))
    } else {
        Ok(None)
    }
}

/// Iterate per-row build-side matches, transparently switching between the
/// direct probe path and the dedup probe path based on `tuning`.
///
/// The single `on_row` callback is invoked exactly once per probe row, in
/// row order. Its second argument is:
///
/// * `Some(iter)` when the row's key produced one or more build matches; the
///   iterator yields `(build_table_idx, build_row_idx)` pairs.
/// * `None` when the row's key had no match (including rows with a null join
///   key).
///
/// We use a single closure (rather than two — one for matches, one for
/// no-match) so callers can mutate join-local state (growable batches, probe
/// index vectors, bitmap builders) inside the callback without falling foul
/// of the borrow checker's "two FnMut closures cannot both capture the same
/// state mutably" rule. Output ordering is identical to the direct path.
pub(crate) fn walk_probe_matches<F>(
    probe_state: &ProbeState,
    join_keys: &RecordBatch,
    tuning: ProbeDedupTuning,
    mut on_row: F,
) -> DaftResult<()>
where
    F: FnMut(usize, Option<&mut dyn Iterator<Item = (u32, u64)>>) -> DaftResult<()>,
{
    if let Some(dedup) = try_build_probe_dedup(join_keys, tuning)? {
        let unique_first_row = dedup
            .unique_first_row
            .iter()
            .map(|&i| i as u64)
            .collect::<Vec<_>>();
        let unique_indices = UInt64Array::from_vec("probe_dedup_unique_idx", unique_first_row);
        let compact_keys = join_keys.take(&unique_indices)?;
        let compact_iter = probe_state.probe_indices(&compact_keys)?;
        let per_unique: Vec<Option<Vec<(u32, u64)>>> = compact_iter
            .map(|maybe_iter| maybe_iter.map(|it| it.collect::<Vec<_>>()))
            .collect();
        for (probe_row_idx, slot) in dedup.row_to_unique.iter().enumerate() {
            match slot.and_then(|unique_idx| per_unique[unique_idx].as_deref()) {
                Some(matches) => {
                    let mut iter = matches.iter().copied();
                    on_row(probe_row_idx, Some(&mut iter))?;
                }
                None => on_row(probe_row_idx, None)?,
            }
        }
    } else {
        let idx_iter = probe_state.probe_indices(join_keys)?;
        for (probe_row_idx, maybe_iter) in idx_iter.enumerate() {
            match maybe_iter {
                Some(mut inner) => on_row(probe_row_idx, Some(&mut inner))?,
                None => on_row(probe_row_idx, None)?,
            }
        }
    }
    Ok(())
}

/// Iterate per-row build-side existence flags. Used by anti/semi joins which
/// only care whether the key matched, not which specific build rows matched.
///
/// The dedup path probes the build hash table once per *unique* probe key and
/// then replays the boolean result back to the original row positions.
pub(crate) fn walk_probe_exists<F>(
    probe_state: &ProbeState,
    join_keys: &RecordBatch,
    tuning: ProbeDedupTuning,
    mut on_row: F,
) -> DaftResult<()>
where
    F: FnMut(usize, bool) -> DaftResult<()>,
{
    if let Some(dedup) = try_build_probe_dedup(join_keys, tuning)? {
        let unique_first_row = dedup
            .unique_first_row
            .iter()
            .map(|&i| i as u64)
            .collect::<Vec<_>>();
        let unique_indices = UInt64Array::from_vec("probe_dedup_unique_idx", unique_first_row);
        let compact_keys = join_keys.take(&unique_indices)?;
        let exists_iter = probe_state.probe_exists(&compact_keys)?;
        let per_unique: Vec<bool> = exists_iter.collect();
        for (probe_row_idx, slot) in dedup.row_to_unique.iter().enumerate() {
            // Null probe keys are excluded from dedup and always match no
            // build row in equality-based hash joins.
            let exists = match slot {
                Some(unique_idx) => per_unique[*unique_idx],
                None => false,
            };
            on_row(probe_row_idx, exists)?;
        }
    } else {
        let exists_iter = probe_state.probe_exists(join_keys)?;
        for (probe_row_idx, exists) in exists_iter.enumerate() {
            on_row(probe_row_idx, exists)?;
        }
    }
    Ok(())
}

/// Test-only helpers shared by the per-join-type equivalence tests below.
#[cfg(test)]
pub(crate) mod test_utils {
    use daft_core::array::ops::DaftCompare;
    use daft_recordbatch::RecordBatch;

    /// Null-safe equality assertion for two `RecordBatch`es.
    ///
    /// `RecordBatch`'s built-in `PartialEq` delegates to `Series::PartialEq`,
    /// which uses SQL-style equality (`null == null` is `null`, treated as
    /// `false`). For our equivalence tests both probe paths must produce
    /// byte-for-byte identical outputs *including* null positions, so we
    /// drive the comparison through the null-safe equality kernel and require
    /// every column to evaluate to `true` at every row.
    pub(crate) fn assert_record_batches_eq_null_safe(left: &RecordBatch, right: &RecordBatch) {
        assert_eq!(
            left.len(),
            right.len(),
            "row counts differ: {} vs {}",
            left.len(),
            right.len()
        );
        assert_eq!(left.schema, right.schema, "schemas differ");
        for (l_col, r_col) in left.columns().iter().zip(right.columns().iter()) {
            let l_series = l_col.as_materialized_series();
            let r_series = r_col.as_materialized_series();
            let arr = l_series
                .eq_null_safe(r_series)
                .expect("eq_null_safe must succeed for matching schemas");
            for i in 0..arr.len() {
                let v = arr.get(i).unwrap_or(false);
                assert!(
                    v,
                    "column {:?} differs at row {} (null-safe equality returned false)",
                    l_col.name(),
                    i,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_batch() {
        let d = ProbeBatchDedup::build(std::iter::empty::<Option<u64>>());
        assert_eq!(d.num_rows(), 0);
        assert_eq!(d.num_unique(), 0);
        assert_eq!(d.unique_ratio(), 0.0);
        assert!(d.row_to_unique.is_empty());
        assert!(d.unique_hashes.is_empty());
        assert!(d.unique_first_row.is_empty());
    }

    #[test]
    fn test_all_unique_keys_no_savings() {
        let hashes: Vec<Option<u64>> = (0..8).map(|i| Some(i as u64)).collect();
        let d = ProbeBatchDedup::build(hashes.iter().copied());
        assert_eq!(d.num_rows(), 8);
        assert_eq!(d.num_unique(), 8);
        assert_eq!(d.unique_ratio(), 1.0);
        let expected: Vec<Option<usize>> = (0..8).map(Some).collect();
        assert_eq!(d.row_to_unique, expected);
        assert_eq!(d.unique_first_row, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn test_all_same_key_full_dedup() {
        let hashes: Vec<Option<u64>> = (0..1000).map(|_| Some(42u64)).collect();
        let d = ProbeBatchDedup::build(hashes.iter().copied());
        assert_eq!(d.num_rows(), 1000);
        assert_eq!(d.num_unique(), 1);
        assert!((d.unique_ratio() - 0.001).abs() < 1e-9);
        assert!(d.row_to_unique.iter().all(|v| v == &Some(0)));
        assert_eq!(d.unique_first_row, vec![0]);
        assert_eq!(d.unique_hashes, vec![42u64]);
    }

    #[test]
    fn test_mixed_keys_first_occurrence_order_preserved() {
        let hashes: Vec<Option<u64>> = vec![
            Some(7),
            Some(3),
            Some(7),
            Some(5),
            Some(3),
            Some(7),
            Some(5),
        ];
        let d = ProbeBatchDedup::build(hashes.iter().copied());
        assert_eq!(d.unique_hashes, vec![7, 3, 5]);
        assert_eq!(d.unique_first_row, vec![0, 1, 3]);
        assert_eq!(
            d.row_to_unique,
            vec![Some(0), Some(1), Some(0), Some(2), Some(1), Some(0), Some(2)]
        );
    }

    #[test]
    fn test_null_keys_passthrough_and_excluded_from_unique() {
        let hashes: Vec<Option<u64>> = vec![None, Some(1), None, Some(1), None, Some(2)];
        let d = ProbeBatchDedup::build(hashes.iter().copied());
        assert_eq!(d.num_rows(), 6);
        assert_eq!(d.num_unique(), 2);
        assert_eq!(
            d.row_to_unique,
            vec![None, Some(0), None, Some(0), None, Some(1)]
        );
        assert_eq!(d.unique_hashes, vec![1, 2]);
        assert_eq!(d.unique_first_row, vec![1, 5]);
    }

    #[test]
    fn test_expand_to_rows_reuses_unique_value() {
        // One unique key repeated 5 times: closure should be called exactly once.
        let hashes: Vec<Option<u64>> = (0..5).map(|_| Some(9u64)).collect();
        let d = ProbeBatchDedup::build(hashes.iter().copied());

        let mut calls = 0usize;
        let expanded = d.expand_to_rows(
            |idx| {
                calls += 1;
                idx as i64 + 100
            },
            || -1,
        );
        assert_eq!(expanded, vec![100, 100, 100, 100, 100]);
        assert_eq!(calls, 1, "value_for_unique must be invoked once per unique");
    }

    #[test]
    fn test_expand_to_rows_handles_nulls() {
        let hashes: Vec<Option<u64>> = vec![None, Some(1), None, Some(2), Some(1)];
        let d = ProbeBatchDedup::build(hashes.iter().copied());

        let expanded = d.expand_to_rows(|idx| 10 + idx as i64, || -1);
        // unique order: [1, 2] -> indices [0, 1]
        assert_eq!(expanded, vec![-1, 10, -1, 11, 10]);
    }

    #[test]
    fn test_expand_to_rows_empty() {
        let d = ProbeBatchDedup::build(std::iter::empty::<Option<u64>>());
        let expanded = d.expand_to_rows(|_| 1i32, || 0);
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_should_use_dedup_below_min_rows() {
        assert!(!should_use_dedup(10, 0.1, 100, 0.8));
    }

    #[test]
    fn test_should_use_dedup_high_unique_ratio_skipped() {
        assert!(!should_use_dedup(10_000, 0.95, 100, 0.8));
    }

    #[test]
    fn test_should_use_dedup_low_unique_ratio_enabled() {
        assert!(should_use_dedup(10_000, 0.05, 100, 0.8));
    }

    #[test]
    fn test_should_use_dedup_boundary_inclusive() {
        // exactly equal to the upper bound -> still enabled (<=)
        assert!(should_use_dedup(100, 0.8, 100, 0.8));
    }

    #[test]
    fn test_should_use_dedup_min_rows_inclusive() {
        // exactly equal to min_rows is allowed (>=)
        assert!(should_use_dedup(100, 0.5, 100, 0.8));
        assert!(!should_use_dedup(99, 0.5, 100, 0.8));
    }

    mod try_build {
        //! Tests for the runtime gate `try_build_probe_dedup`. These tests
        //! exercise the full path of "hash the probe-batch keys + decide
        //! whether dedup should kick in" against real `RecordBatch` inputs.

        use daft_core::prelude::{DataType, Field, Int64Array, IntoSeries};
        use daft_recordbatch::RecordBatch;

        use super::super::*;

        fn rb_from_int_col(name: &str, values: Vec<Option<i64>>) -> RecordBatch {
            let arr = Int64Array::from_iter(Field::new(name, DataType::Int64), values);
            RecordBatch::from_nonempty_columns(vec![arr.into_series()]).unwrap()
        }

        #[test]
        fn test_disabled_returns_none() {
            let join_keys = rb_from_int_col("k", (0..1000).map(|i| Some(i % 4)).collect());
            let tuning = ProbeDedupTuning {
                enabled: false,
                min_rows: 100,
                unique_ratio_max: 0.5,
            };
            assert!(try_build_probe_dedup(&join_keys, tuning).unwrap().is_none());
        }

        #[test]
        fn test_below_min_rows_returns_none() {
            let join_keys = rb_from_int_col("k", vec![Some(1), Some(1), Some(1)]);
            let tuning = ProbeDedupTuning {
                enabled: true,
                min_rows: 100,
                unique_ratio_max: 0.5,
            };
            assert!(try_build_probe_dedup(&join_keys, tuning).unwrap().is_none());
        }

        #[test]
        fn test_high_unique_ratio_returns_none() {
            // 10 rows, all distinct -> unique_ratio = 1.0, above threshold 0.5.
            let join_keys = rb_from_int_col("k", (0..10).map(Some).collect());
            let tuning = ProbeDedupTuning {
                enabled: true,
                min_rows: 5,
                unique_ratio_max: 0.5,
            };
            assert!(try_build_probe_dedup(&join_keys, tuning).unwrap().is_none());
        }

        #[test]
        fn test_low_unique_ratio_returns_some() {
            // 1000 rows, 4 distinct -> unique_ratio = 0.004
            let join_keys = rb_from_int_col("k", (0..1000).map(|i| Some(i % 4)).collect());
            let tuning = ProbeDedupTuning {
                enabled: true,
                min_rows: 100,
                unique_ratio_max: 0.5,
            };
            let dedup = try_build_probe_dedup(&join_keys, tuning)
                .unwrap()
                .expect("should have been activated");
            assert_eq!(dedup.num_rows(), 1000);
            assert_eq!(dedup.num_unique(), 4);
        }
    }
}
