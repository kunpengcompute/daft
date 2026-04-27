use std::sync::Arc;

use common_error::DaftResult;
use daft_core::prelude::{Schema, Series, UInt64Array};
use daft_logical_plan::JoinType;
use daft_micropartition::MicroPartition;
use daft_recordbatch::{GrowableRecordBatch, ProbeState, RecordBatch, get_columns_by_name};

use crate::join::{
    hash_join::{HashJoinParams, HashJoinProbeState},
    index_bitmap::IndexBitmapBuilder,
    outer_join::merge_bitmaps_and_construct_null_table,
    probe_dedup::walk_probe_matches,
};

pub(crate) fn probe_left_right_with_bitmap(
    input: &MicroPartition,
    bitmap_builder: &mut IndexBitmapBuilder,
    probe_state: &ProbeState,
    params: &HashJoinParams,
) -> DaftResult<MicroPartition> {
    let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();

    let final_tables = input
        .record_batches()
        .iter()
        .map(|input_table| {
            let mut build_side_growable = GrowableRecordBatch::new(
                &build_side_tables,
                false,
                build_side_tables.iter().map(|table| table.len()).sum(),
            )?;
            let mut probe_side_idxs = Vec::new();

            let join_keys = input_table.eval_expression_list(&params.probe_on)?;
            walk_probe_matches(
                probe_state,
                &join_keys,
                params.probe_dedup,
                |probe_row_idx, maybe_matches| {
                    // Bitmap-tracking left/right join only emits matched rows
                    // here; unmatched build-side rows are surfaced later by
                    // `finalize_*` so we ignore the no-match case.
                    if let Some(matches) = maybe_matches {
                        for (build_table_idx, build_row_idx) in matches {
                            bitmap_builder
                                .mark_used(build_table_idx as usize, build_row_idx as usize);
                            build_side_growable.extend(
                                build_table_idx as usize,
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

            let final_table = if params.join_type == JoinType::Left {
                let join_table = get_columns_by_name(&build_side_table, &common_join_keys)?;
                let left = get_columns_by_name(&build_side_table, &left_non_join_columns)?;
                let right = get_columns_by_name(&probe_side_table, &right_non_join_columns)?;
                join_table.union(&left)?.union(&right)?
            } else {
                let join_table = get_columns_by_name(&build_side_table, &common_join_keys)?;
                let left = get_columns_by_name(&probe_side_table, &left_non_join_columns)?;
                let right = get_columns_by_name(&build_side_table, &right_non_join_columns)?;
                join_table.union(&left)?.union(&right)?
            };
            Ok(final_table)
        })
        .collect::<DaftResult<Vec<_>>>()?;

    Ok(MicroPartition::new_loaded(
        params.output_schema.clone(),
        Arc::new(final_tables),
        None,
    ))
}

pub(crate) fn probe_left_right(
    input: &MicroPartition,
    probe_state: &ProbeState,
    params: &HashJoinParams,
) -> DaftResult<MicroPartition> {
    let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();

    let final_tables = input
        .record_batches()
        .iter()
        .map(|input_table| {
            let mut build_side_growable = GrowableRecordBatch::new(
                &build_side_tables,
                true,
                build_side_tables.iter().map(|table| table.len()).sum(),
            )?;
            let mut probe_side_idxs = Vec::with_capacity(input_table.len());

            let join_keys = input_table.eval_expression_list(&params.probe_on)?;
            walk_probe_matches(
                probe_state,
                &join_keys,
                params.probe_dedup,
                |probe_row_idx, maybe_matches| {
                    match maybe_matches {
                        Some(matches) => {
                            for (build_table_idx, build_row_idx) in matches {
                                build_side_growable.extend(
                                    build_table_idx as usize,
                                    build_row_idx as usize,
                                    1,
                                );
                                probe_side_idxs.push(probe_row_idx as u64);
                            }
                        }
                        None => {
                            // No match -> still emit the probe row, padded
                            // with nulls on the build side so left/right join
                            // semantics keep every probe row.
                            build_side_growable.add_nulls(1);
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

            let final_table = if params.join_type == JoinType::Left {
                let join_table = get_columns_by_name(&probe_side_table, &common_join_keys)?;
                let left = get_columns_by_name(&probe_side_table, &left_non_join_columns)?;
                let right = get_columns_by_name(&build_side_table, &right_non_join_columns)?;
                join_table.union(&left)?.union(&right)?
            } else {
                let join_table = get_columns_by_name(&probe_side_table, &common_join_keys)?;
                let left = get_columns_by_name(&build_side_table, &left_non_join_columns)?;
                let right = get_columns_by_name(&probe_side_table, &right_non_join_columns)?;
                join_table.union(&left)?.union(&right)?
            };
            Ok(final_table)
        })
        .collect::<DaftResult<Vec<_>>>()?;

    Ok(MicroPartition::new_loaded(
        params.output_schema.clone(),
        Arc::new(final_tables),
        None,
    ))
}

pub(crate) async fn finalize_left(
    states: Vec<HashJoinProbeState>,
    params: &HashJoinParams,
) -> DaftResult<Option<MicroPartition>> {
    let build_side_table = merge_bitmaps_and_construct_null_table(states).await?;

    // If build_side_table is empty, return empty result with correct schema
    if build_side_table.is_empty() {
        return Ok(Some(MicroPartition::empty(Some(
            params.output_schema.clone(),
        ))));
    }

    let common_join_cols: Vec<String> = params.common_join_cols.iter().cloned().collect();
    let left_non_join_columns: Vec<String> = params
        .left_schema
        .field_names()
        .filter(|c| !params.common_join_cols.contains(*c))
        .map(ToString::to_string)
        .collect();
    let right_non_join_schema = Arc::new(Schema::new(
        params
            .right_schema
            .fields()
            .iter()
            .filter(|f| !params.common_join_cols.contains(&*f.name))
            .cloned(),
    ));

    // For left join, we only finalize when build_on_left is true (needs_bitmap check ensures this)
    // So build_side_table has left columns
    let join_table = get_columns_by_name(&build_side_table, &common_join_cols)?;
    let left = get_columns_by_name(&build_side_table, &left_non_join_columns)?;
    let right = {
        let columns = right_non_join_schema
            .fields()
            .iter()
            .map(|field| Series::full_null(&field.name, &field.dtype, left.len()))
            .collect::<Vec<_>>();
        RecordBatch::new_unchecked(right_non_join_schema, columns, left.len())
    };
    let final_table = join_table.union(&left)?.union(&right)?;
    Ok(Some(MicroPartition::new_loaded(
        final_table.schema.clone(),
        Arc::new(vec![final_table]),
        None,
    )))
}

pub(crate) async fn finalize_right(
    states: Vec<HashJoinProbeState>,
    params: &HashJoinParams,
) -> DaftResult<Option<MicroPartition>> {
    let build_side_table = merge_bitmaps_and_construct_null_table(states).await?;

    // If build_side_table is empty, return empty result with correct schema
    if build_side_table.is_empty() {
        return Ok(Some(MicroPartition::empty(Some(
            params.output_schema.clone(),
        ))));
    }

    let common_join_cols: Vec<String> = params.common_join_cols.iter().cloned().collect();
    let right_non_join_columns: Vec<String> = params
        .right_schema
        .field_names()
        .filter(|c| !params.common_join_cols.contains(*c))
        .map(ToString::to_string)
        .collect();
    let left_non_join_schema = Arc::new(Schema::new(
        params
            .left_schema
            .fields()
            .iter()
            .filter(|f| !params.common_join_cols.contains(&*f.name))
            .cloned(),
    ));

    // For right join, we only finalize when build_on_left is false (needs_bitmap check ensures this)
    // So build_side_table has right columns
    let join_table = get_columns_by_name(&build_side_table, &common_join_cols)?;
    let left = {
        let columns = left_non_join_schema
            .fields()
            .iter()
            .map(|field| Series::full_null(&field.name, &field.dtype, build_side_table.len()))
            .collect::<Vec<_>>();
        RecordBatch::new_unchecked(left_non_join_schema, columns, build_side_table.len())
    };
    let right = get_columns_by_name(&build_side_table, &right_non_join_columns)?;
    let final_table = join_table.union(&left)?.union(&right)?;
    Ok(Some(MicroPartition::new_loaded(
        final_table.schema.clone(),
        Arc::new(vec![final_table]),
        None,
    )))
}

#[cfg(test)]
mod tests {
    //! Equivalence tests for the left/right-join probe dedup paths.
    //!
    //! Both `probe_left_right` (no bitmap) and `probe_left_right_with_bitmap`
    //! must produce identical `(probe_idxs, build_table)` pairs whether or not
    //! the dedup runner is engaged. The bitmap path additionally must produce
    //! identical `IndexBitmapBuilder` state, which we exercise by also
    //! comparing the merged bitmap.

    use daft_core::prelude::{DataType, Field, Int64Array, IntoSeries};
    use daft_recordbatch::{GrowableRecordBatch, ProbeState, make_probeable_builder};

    use super::*;
    use crate::join::probe_dedup::{
        ProbeDedupTuning, test_utils::assert_record_batches_eq_null_safe, walk_probe_matches,
    };

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

    /// Replays `probe_left_right`'s emit logic via the runner so we can
    /// directly compare direct-vs-dedup outputs without rebuilding all the
    /// downstream column reshaping.
    fn run_left_right(
        probe_state: &ProbeState,
        join_keys: &RecordBatch,
        tuning: ProbeDedupTuning,
    ) -> (Vec<u64>, RecordBatch) {
        let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();
        let mut growable = GrowableRecordBatch::new(&build_side_tables, true, 8).unwrap();
        let mut idxs = Vec::new();
        walk_probe_matches(
            probe_state,
            join_keys,
            tuning,
            |probe_row_idx, maybe_matches| {
                match maybe_matches {
                    Some(matches) => {
                        for (build_table_idx, build_row_idx) in matches {
                            growable.extend(
                                build_table_idx as usize,
                                build_row_idx as usize,
                                1,
                            );
                            idxs.push(probe_row_idx as u64);
                        }
                    }
                    None => {
                        growable.add_nulls(1);
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

    /// Replays `probe_left_right_with_bitmap`'s emit logic via the runner.
    fn run_left_right_with_bitmap(
        probe_state: &ProbeState,
        join_keys: &RecordBatch,
        tuning: ProbeDedupTuning,
    ) -> (Vec<u64>, RecordBatch, IndexBitmapBuilder) {
        let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();
        let mut growable = GrowableRecordBatch::new(&build_side_tables, false, 8).unwrap();
        let mut idxs = Vec::new();
        let mut bitmap_builder = IndexBitmapBuilder::new(probe_state.get_record_batches());
        walk_probe_matches(
            probe_state,
            join_keys,
            tuning,
            |probe_row_idx, maybe_matches| {
                if let Some(matches) = maybe_matches {
                    for (build_table_idx, build_row_idx) in matches {
                        bitmap_builder
                            .mark_used(build_table_idx as usize, build_row_idx as usize);
                        growable.extend(build_table_idx as usize, build_row_idx as usize, 1);
                        idxs.push(probe_row_idx as u64);
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        let table = growable.build().unwrap();
        (idxs, table, bitmap_builder)
    }

    fn assert_left_right_match(probe_state: &ProbeState, join_keys: &RecordBatch) {
        let (idxs_d, table_d) = run_left_right(probe_state, join_keys, force_direct());
        let (idxs_p, table_p) = run_left_right(probe_state, join_keys, force_dedup());
        assert_eq!(idxs_d, idxs_p);
        assert_record_batches_eq_null_safe(&table_d, &table_p);
    }

    fn assert_left_right_bitmap_match(probe_state: &ProbeState, join_keys: &RecordBatch) {
        let (idxs_d, table_d, bm_d) =
            run_left_right_with_bitmap(probe_state, join_keys, force_direct());
        let (idxs_p, table_p, bm_p) =
            run_left_right_with_bitmap(probe_state, join_keys, force_dedup());
        assert_eq!(idxs_d, idxs_p);
        assert_record_batches_eq_null_safe(&table_d, &table_p);
        // Bitmap state must agree: same set of (table_idx, row_idx) entries
        // must have been marked. We compare via the merged boolean arrays
        // produced by the IndexBitmapBuilder.
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
    fn test_left_right_dedup_repeated_keys() {
        let probe_state = build_probe_state(vec![Some(10), Some(20), Some(30), Some(20)]);
        let probe_keys =
            rb_from_int_col("k", vec![Some(20), Some(10), Some(20), Some(30), Some(20)]);
        assert_left_right_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_left_right_dedup_with_unmatched_and_null_probe_keys() {
        // Mixed null/no-match keys exercise the `on_no_match` callback path.
        let probe_state = build_probe_state(vec![Some(1), Some(2)]);
        let probe_keys = rb_from_int_col(
            "k",
            vec![Some(1), None, Some(99), Some(2), None, Some(1), Some(99)],
        );
        assert_left_right_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_left_right_dedup_all_unmatched() {
        let probe_state = build_probe_state(vec![Some(100)]);
        let probe_keys = rb_from_int_col("k", vec![Some(1), Some(2), Some(1), Some(2)]);
        assert_left_right_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_left_right_with_bitmap_repeated_keys() {
        let probe_state = build_probe_state(vec![Some(10), Some(20), Some(30), Some(20)]);
        let probe_keys =
            rb_from_int_col("k", vec![Some(20), Some(10), Some(20), Some(30), Some(20)]);
        assert_left_right_bitmap_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_left_right_with_bitmap_with_null_probe_keys() {
        let probe_state = build_probe_state(vec![Some(1), Some(2), Some(3)]);
        let probe_keys = rb_from_int_col(
            "k",
            vec![Some(1), None, Some(2), None, Some(99), Some(2)],
        );
        assert_left_right_bitmap_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_left_right_with_bitmap_heavy_skew() {
        // 1000 rows of key=7. Bitmap should mark the build-side rows for key=7
        // identically across both paths.
        let probe_state = build_probe_state(vec![Some(7), Some(7), Some(99)]);
        let probe_keys = rb_from_int_col("k", (0..1000).map(|_| Some(7i64)).collect());
        assert_left_right_bitmap_match(&probe_state, &probe_keys);
    }
}
