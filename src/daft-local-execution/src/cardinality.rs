//! Shared cardinality profiling for hash-based operators.
//!
//! Used by grouped aggregations and hash joins to make low-cardinality aware
//! decisions such as strategy routing (`AggThenPartition` vs `PartitionThenAgg`)
//! and probe-side batch deduplication.
//!
//! The profile is intentionally cheap to construct from a stream of row hashes
//! so it can be evaluated on a small sample of the input (e.g. the first
//! morsel) without materializing the full hash table.

use std::collections::HashMap;

/// Tunable thresholds for declaring an input to be "low cardinality".
///
/// These are surfaced as fields on `DaftExecutionConfig` in the integration
/// step; the struct is kept self-contained here so the pure logic can be
/// unit-tested in isolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CardinalityConfig {
    /// Maximum NDV / num_rows ratio at which the profile is considered low
    /// cardinality. A value of `0.05` means 1 distinct key per 20 rows.
    pub low_cardinality_ndv_ratio_threshold: f64,
    /// Minimum number of observed rows required before the profile is allowed
    /// to mark the input as low cardinality. Avoids reacting to tiny samples.
    pub low_cardinality_min_rows: usize,
}

impl Default for CardinalityConfig {
    fn default() -> Self {
        Self {
            low_cardinality_ndv_ratio_threshold: 0.05,
            low_cardinality_min_rows: 4096,
        }
    }
}

/// Compact statistics describing the cardinality of a key set.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CardinalityProfile {
    pub num_rows: usize,
    pub estimated_ndv: usize,
    /// `estimated_ndv / num_rows`, or `0.0` for an empty input.
    pub ndv_ratio: f64,
    /// Hottest single key's share of non-null rows. Useful for skew-aware
    /// repartitioning decisions; computed from the same sample as `ndv`.
    pub top1_ratio: f64,
}

impl CardinalityProfile {
    /// Build a profile from an iterator of optional row hashes. `None` is
    /// treated as a null key: it is counted toward `num_rows` but excluded
    /// from `estimated_ndv` and the `top1_ratio` denominator.
    pub fn from_hashes<I>(hashes: I) -> Self
    where
        I: IntoIterator<Item = Option<u64>>,
    {
        let iter = hashes.into_iter();
        let (lower, _) = iter.size_hint();
        let mut counts: HashMap<u64, usize> = HashMap::with_capacity(lower);
        let mut num_rows = 0usize;
        let mut non_null = 0usize;
        for h in iter {
            num_rows += 1;
            if let Some(h) = h {
                non_null += 1;
                *counts.entry(h).or_insert(0) += 1;
            }
        }
        let estimated_ndv = counts.len();
        let ndv_ratio = if num_rows == 0 {
            0.0
        } else {
            estimated_ndv as f64 / num_rows as f64
        };
        let top1 = counts.values().copied().max().unwrap_or(0);
        let top1_ratio = if non_null == 0 {
            0.0
        } else {
            top1 as f64 / non_null as f64
        };
        Self {
            num_rows,
            estimated_ndv,
            ndv_ratio,
            top1_ratio,
        }
    }

    /// Whether this profile crosses both gates (sample size + ndv ratio) for
    /// being treated as low cardinality.
    pub fn is_low_cardinality(&self, cfg: &CardinalityConfig) -> bool {
        self.num_rows >= cfg.low_cardinality_min_rows
            && self.ndv_ratio <= cfg.low_cardinality_ndv_ratio_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CardinalityConfig {
        CardinalityConfig {
            low_cardinality_ndv_ratio_threshold: 0.05,
            low_cardinality_min_rows: 100,
        }
    }

    #[test]
    fn test_empty_input_zero_metrics() {
        let p = CardinalityProfile::from_hashes(std::iter::empty::<Option<u64>>());
        assert_eq!(p.num_rows, 0);
        assert_eq!(p.estimated_ndv, 0);
        assert_eq!(p.ndv_ratio, 0.0);
        assert_eq!(p.top1_ratio, 0.0);
        assert!(!p.is_low_cardinality(&cfg()));
    }

    #[test]
    fn test_all_unique_high_cardinality() {
        let hashes = (0..1000u64).map(Some);
        let p = CardinalityProfile::from_hashes(hashes);
        assert_eq!(p.num_rows, 1000);
        assert_eq!(p.estimated_ndv, 1000);
        assert!((p.ndv_ratio - 1.0).abs() < 1e-9);
        assert!((p.top1_ratio - 0.001).abs() < 1e-9);
        assert!(!p.is_low_cardinality(&cfg()));
    }

    #[test]
    fn test_low_cardinality_detected() {
        // 1000 rows, 10 distinct keys → ndv_ratio = 0.01 < 0.05
        let hashes = (0..1000u64).map(|i| Some(i % 10));
        let p = CardinalityProfile::from_hashes(hashes);
        assert_eq!(p.num_rows, 1000);
        assert_eq!(p.estimated_ndv, 10);
        assert!((p.ndv_ratio - 0.01).abs() < 1e-9);
        // uniform distribution → each key is ~10%
        assert!((p.top1_ratio - 0.1).abs() < 1e-9);
        assert!(p.is_low_cardinality(&cfg()));
    }

    #[test]
    fn test_threshold_boundary_just_above_disabled() {
        // 100 rows, 6 distinct → ratio = 0.06 > 0.05
        let hashes = (0..100u64).map(|i| Some(i % 6));
        let p = CardinalityProfile::from_hashes(hashes);
        assert_eq!(p.estimated_ndv, 6);
        assert!(p.ndv_ratio > 0.05);
        assert!(!p.is_low_cardinality(&cfg()));
    }

    #[test]
    fn test_threshold_boundary_at_or_below_enabled() {
        // 200 rows, 10 distinct → ratio = 0.05 == threshold (inclusive)
        let hashes = (0..200u64).map(|i| Some(i % 10));
        let p = CardinalityProfile::from_hashes(hashes);
        assert_eq!(p.estimated_ndv, 10);
        assert!((p.ndv_ratio - 0.05).abs() < 1e-9);
        assert!(p.is_low_cardinality(&cfg()));
    }

    #[test]
    fn test_min_rows_gate_disables_low_card() {
        // ndv_ratio is low (2 distinct over 50 rows -> 0.04), but rows below
        // the min threshold (100) -> low-cardinality must NOT be reported.
        let hashes = (0..50u64).map(|i| Some(i % 2));
        let p = CardinalityProfile::from_hashes(hashes);
        assert_eq!(p.estimated_ndv, 2);
        assert!(p.ndv_ratio <= 0.05);
        assert!(p.num_rows < 100);
        assert!(!p.is_low_cardinality(&cfg()));
    }

    #[test]
    fn test_top1_ratio_skewed_distribution() {
        // 100 rows, 90 hot key + 10 cold keys spread out → top1_ratio = 0.9
        let mut hashes: Vec<Option<u64>> = (0..90).map(|_| Some(7u64)).collect();
        hashes.extend((0..10).map(|i| Some(100 + i as u64)));
        let p = CardinalityProfile::from_hashes(hashes);
        assert_eq!(p.num_rows, 100);
        assert_eq!(p.estimated_ndv, 11);
        assert!((p.top1_ratio - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_top1_ratio_ignores_nulls() {
        // 100 rows, 50 None, 50 Some(7) → top1 over non_null is 1.0
        let hashes = (0..100u64).map(|i| if i % 2 == 0 { None } else { Some(7) });
        let p = CardinalityProfile::from_hashes(hashes);
        assert_eq!(p.num_rows, 100);
        assert_eq!(p.estimated_ndv, 1);
        assert!((p.top1_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_all_nulls_zero_ndv_and_top1() {
        let hashes = (0..200usize).map(|_| None::<u64>);
        let p = CardinalityProfile::from_hashes(hashes);
        assert_eq!(p.num_rows, 200);
        assert_eq!(p.estimated_ndv, 0);
        assert_eq!(p.ndv_ratio, 0.0);
        assert_eq!(p.top1_ratio, 0.0);
        // ratio is 0 so threshold passes, but spec: NDV must be > 0 to be useful.
        // Current rule keeps it simple: all-null is treated as low cardinality;
        // callers handle the special case via num_rows / estimated_ndv directly.
        assert!(p.is_low_cardinality(&cfg()));
        assert_eq!(p.estimated_ndv, 0);
    }

    #[test]
    fn test_default_config_values() {
        let c = CardinalityConfig::default();
        assert!((c.low_cardinality_ndv_ratio_threshold - 0.05).abs() < 1e-9);
        assert_eq!(c.low_cardinality_min_rows, 4096);
    }

    #[test]
    fn test_profile_is_deterministic_across_orderings() {
        let a = CardinalityProfile::from_hashes((0..1000u64).map(|i| Some(i % 8)));
        let b = CardinalityProfile::from_hashes((0..1000u64).rev().map(|i| Some(i % 8)));
        assert_eq!(a.num_rows, b.num_rows);
        assert_eq!(a.estimated_ndv, b.estimated_ndv);
        assert!((a.ndv_ratio - b.ndv_ratio).abs() < 1e-12);
        assert!((a.top1_ratio - b.top1_ratio).abs() < 1e-12);
    }
}
