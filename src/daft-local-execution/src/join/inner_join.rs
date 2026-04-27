use std::sync::Arc;

use common_error::DaftResult;
use daft_core::prelude::UInt64Array;
use daft_micropartition::MicroPartition;
use daft_recordbatch::{GrowableRecordBatch, ProbeState};

use crate::join::{hash_join::HashJoinParams, probe_dedup::walk_probe_matches};

pub(crate) fn probe_inner(
    input: &MicroPartition,
    probe_state: &ProbeState,
    params: &HashJoinParams,
) -> DaftResult<MicroPartition> {
    let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();
    const DEFAULT_GROWABLE_SIZE: usize = 20;

    let input_tables = input.record_batches();
    let result_tables = input_tables
        .iter()
        .map(|input_table| {
            let mut build_side_growable =
                GrowableRecordBatch::new(&build_side_tables, false, DEFAULT_GROWABLE_SIZE)?;
            let mut probe_side_idxs = Vec::new();

            let join_keys = input_table.eval_expression_list(&params.probe_on)?;
            walk_probe_matches(
                probe_state,
                &join_keys,
                params.probe_dedup,
                |probe_row_idx, maybe_matches| {
                    // Inner join: emit one output row per build-match and
                    // simply drop probe rows with no match.
                    if let Some(matches) = maybe_matches {
                        for (build_rb_idx, build_row_idx) in matches {
                            build_side_growable.extend(
                                build_rb_idx as usize,
                                build_row_idx as usize,
                                1,
                            );
                            probe_side_idxs.push(probe_row_idx as u64);
                        }
                    }
                    Ok(())
                },
            )?;

            let build_side_table = build_side_growable.build()?;
            let probe_side_table = {
                let indices_arr = UInt64Array::from_vec("", probe_side_idxs);
                input_table.take(&indices_arr)?
            };

            let (left_table, right_table) = if params.build_on_left {
                (build_side_table, probe_side_table)
            } else {
                (probe_side_table, build_side_table)
            };

            let common_join_keys: Vec<String> = params.common_join_cols.iter().cloned().collect();
            let left_non_join_columns: Vec<String> = params
                .left_schema
                .field_names()
                .filter(|c| !params.common_join_cols.contains(*c))
                .map(ToString::to_string)
                .collect();
            let right_non_join_columns: Vec<String> = params
                .right_schema
                .field_names()
                .filter(|c| !params.common_join_cols.contains(*c))
                .map(ToString::to_string)
                .collect();

            let join_keys_table =
                daft_recordbatch::get_columns_by_name(&left_table, &common_join_keys)?;
            let left_non_join_columns =
                daft_recordbatch::get_columns_by_name(&left_table, &left_non_join_columns)?;
            let right_non_join_columns =
                daft_recordbatch::get_columns_by_name(&right_table, &right_non_join_columns)?;
            let final_table = join_keys_table
                .union(&left_non_join_columns)?
                .union(&right_non_join_columns)?;
            Ok(final_table)
        })
        .collect::<DaftResult<Vec<_>>>()?;

    Ok(MicroPartition::new_loaded(
        params.output_schema.clone(),
        Arc::new(result_tables),
        None,
    ))
}

#[cfg(test)]
mod tests {
    //! Equivalence tests for the inner-join probe dedup path.
    //!
    //! These tests pin the contract that turning on dedup is purely a
    //! performance optimization: for every build/probe pair the dedup path
    //! MUST produce the identical `(probe_side_idxs, build_side_table)`
    //! tuple as the direct path. We drive `walk_probe_matches` directly with
    //! the inner-join callbacks under both forced-on and forced-off dedup
    //! tunings to compare the two paths.

    use daft_core::prelude::{DataType, Field, Int64Array, IntoSeries};
    use daft_recordbatch::{ProbeState, RecordBatch, make_probeable_builder};

    use super::*;
    use crate::join::probe_dedup::{ProbeDedupTuning, test_utils::assert_record_batches_eq_null_safe};

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
        // Tuning chosen so any non-trivial repeated probe batch activates dedup.
        ProbeDedupTuning {
            enabled: true,
            min_rows: 1,
            unique_ratio_max: 1.0,
        }
    }

    fn force_direct() -> ProbeDedupTuning {
        ProbeDedupTuning::disabled()
    }

    fn run_inner(
        probe_state: &ProbeState,
        join_keys: &RecordBatch,
        tuning: ProbeDedupTuning,
    ) -> (Vec<u64>, RecordBatch) {
        let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();
        let mut growable = GrowableRecordBatch::new(&build_side_tables, false, 8).unwrap();
        let mut idxs = Vec::new();
        walk_probe_matches(
            probe_state,
            join_keys,
            tuning,
            |probe_row_idx, maybe_matches| {
                if let Some(matches) = maybe_matches {
                    for (build_rb_idx, build_row_idx) in matches {
                        growable.extend(build_rb_idx as usize, build_row_idx as usize, 1);
                        idxs.push(probe_row_idx as u64);
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        let table = growable.build().unwrap();
        (idxs, table)
    }

    fn assert_paths_match(probe_state: &ProbeState, join_keys: &RecordBatch) {
        let (idxs_direct, build_direct) = run_inner(probe_state, join_keys, force_direct());
        let (idxs_dedup, build_dedup) = run_inner(probe_state, join_keys, force_dedup());
        assert_eq!(
            idxs_direct, idxs_dedup,
            "probe-side row indices must be identical regardless of path"
        );
        assert_record_batches_eq_null_safe(&build_direct, &build_dedup);
    }

    #[test]
    fn test_dedup_matches_direct_repeated_keys() {
        // Build side: 10, 20, 30, 20 (key 20 appears twice -> two build matches).
        // Probe side: 20, 10, 20, 30, 20 (key 20 dominates the probe input).
        let probe_state = build_probe_state(vec![Some(10), Some(20), Some(30), Some(20)]);
        let probe_keys =
            rb_from_int_col("k", vec![Some(20), Some(10), Some(20), Some(30), Some(20)]);
        assert_paths_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_dedup_matches_direct_with_null_probe_keys() {
        // Probe-side null keys must be passed through both paths in the
        // exact same way (treated as "no match" by the inner join).
        let probe_state = build_probe_state(vec![Some(1), Some(2), Some(3)]);
        let probe_keys = rb_from_int_col(
            "k",
            vec![Some(1), None, Some(2), None, Some(99), Some(2)],
        );
        assert_paths_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_dedup_matches_direct_no_matches_at_all() {
        // No probe key matches any build key.
        let probe_state = build_probe_state(vec![Some(100), Some(200)]);
        let probe_keys = rb_from_int_col("k", vec![Some(1), Some(2), Some(1), Some(3)]);
        assert_paths_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_dedup_matches_direct_all_unique_keys() {
        // Pathological case for dedup: every probe row is unique. Output
        // must still match exactly because dedup degrades to "1 unique per
        // row" rather than producing different matches.
        let probe_state = build_probe_state(vec![Some(1), Some(2), Some(3), Some(4)]);
        let probe_keys = rb_from_int_col("k", vec![Some(1), Some(2), Some(3), Some(4)]);
        assert_paths_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_dedup_matches_direct_single_repeated_key_heavy_skew() {
        // 1000 probe rows, all key=7. Build side has key=7 twice -> 2000
        // output rows. This is the "best case" workload for dedup; we still
        // expect bit-exact equivalence.
        let probe_state = build_probe_state(vec![Some(7), Some(7), Some(99)]);
        let probe_keys = rb_from_int_col("k", (0..1000).map(|_| Some(7i64)).collect());
        assert_paths_match(&probe_state, &probe_keys);
    }
}
