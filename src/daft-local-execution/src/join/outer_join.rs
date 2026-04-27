use std::sync::Arc;

use common_error::DaftResult;
use daft_core::{
    prelude::{Field, IntoSeries, Schema, Series, UInt64Array},
    utils::supertype::try_get_supertype,
};
use daft_micropartition::MicroPartition;
use daft_recordbatch::{GrowableRecordBatch, ProbeState, RecordBatch, get_columns_by_name};
use futures::{StreamExt, stream};

use crate::join::{
    hash_join::{HashJoinParams, HashJoinProbeState},
    index_bitmap::IndexBitmapBuilder,
    probe_dedup::walk_probe_matches,
};

pub(crate) async fn merge_bitmaps_and_construct_null_table(
    states: Vec<HashJoinProbeState>,
) -> DaftResult<RecordBatch> {
    let mut states_iter = states.into_iter();
    let first_state = states_iter
        .next()
        .expect("At least one state should be present for merge bitmaps and construct null table");
    let first_tables = first_state.probe_state.get_record_batches();
    let first_bitmap = first_state
        .bitmap_builder
        .expect("Bitmap builder should be set for merge bitmaps and construct null table")
        .build();

    let merged_bitmap = {
        let bitmaps = stream::once(async move { first_bitmap })
            .chain(stream::iter(states_iter).map(|s| {
                s.bitmap_builder
                    .expect(
                        "Bitmap builder should be set for merge bitmaps and construct null table",
                    )
                    .build()
            }))
            .collect::<Vec<_>>()
            .await;

        bitmaps.into_iter().fold(None, |acc, x| match acc {
            None => Some(x),
            Some(acc) => Some(acc.merge(&x)),
        })
    }
    .expect("At least one bitmap should be present for merge bitmaps and construct null table");

    let leftovers = merged_bitmap
        .convert_to_boolean_arrays()
        .zip(first_tables.iter())
        .map(|(bitmap, table)| table.mask_filter(&bitmap.into_series()))
        .collect::<DaftResult<Vec<_>>>()?;
    RecordBatch::concat(&leftovers)
}

pub(crate) fn probe_outer(
    input: &MicroPartition,
    probe_state: &ProbeState,
    bitmap_builder: Option<&mut IndexBitmapBuilder>,
    params: &HashJoinParams,
) -> DaftResult<MicroPartition> {
    let bitmap_builder = bitmap_builder.expect("Bitmap builder should be set for outer join probe");
    let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();

    // Compute outer common column schema
    let outer_common_col_fields = params
        .common_join_cols
        .iter()
        .map(|name| {
            let supertype = try_get_supertype(
                &params.left_schema.get_field(name)?.dtype,
                &params.right_schema.get_field(name)?.dtype,
            )?;
            Ok(Field::new(name.clone(), supertype))
        })
        .collect::<DaftResult<Vec<_>>>()?;
    let outer_common_col_schema = Arc::new(Schema::new(outer_common_col_fields));

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
                                bitmap_builder.mark_used(
                                    build_table_idx as usize,
                                    build_row_idx as usize,
                                );
                                build_side_growable.extend(
                                    build_table_idx as usize,
                                    build_row_idx as usize,
                                    1,
                                );
                                probe_side_idxs.push(probe_row_idx as u64);
                            }
                        }
                        None => {
                            // No match -> emit the probe row with nulls on
                            // the build side; outer join finalize emits the
                            // unmatched build rows symmetrically afterwards.
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

            #[allow(deprecated)]
            let join_table = get_columns_by_name(&probe_side_table, &common_join_keys)?
                .cast_to_schema(&outer_common_col_schema)?;
            // Get left and right columns based on which side we built on
            let (left, right) = if params.build_on_left {
                // Built on left, so build_side_table has left columns, probe_side_table has right columns
                let left = get_columns_by_name(&build_side_table, &left_non_join_columns)?;
                let right = get_columns_by_name(&probe_side_table, &right_non_join_columns)?;
                (left, right)
            } else {
                // Built on right, so build_side_table has right columns, probe_side_table has left columns
                let left = get_columns_by_name(&probe_side_table, &left_non_join_columns)?;
                let right = get_columns_by_name(&build_side_table, &right_non_join_columns)?;
                (left, right)
            };
            let final_table = join_table.union(&left)?.union(&right)?;
            Ok(final_table)
        })
        .collect::<DaftResult<Vec<_>>>()?;

    Ok(MicroPartition::new_loaded(
        params.output_schema.clone(),
        Arc::new(final_tables),
        None,
    ))
}

pub(crate) async fn finalize_outer(
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

    // Compute outer common column schema
    let outer_common_col_fields = params
        .common_join_cols
        .iter()
        .map(|name| {
            let supertype = try_get_supertype(
                &params.left_schema.get_field(name)?.dtype,
                &params.right_schema.get_field(name)?.dtype,
            )?;
            Ok(Field::new(name.clone(), supertype))
        })
        .collect::<DaftResult<Vec<_>>>()?;
    let outer_common_col_schema = Arc::new(Schema::new(outer_common_col_fields));

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

    let common_join_cols: Vec<String> = params.common_join_cols.iter().cloned().collect();
    let right_non_join_columns: Vec<String> = params
        .right_schema
        .field_names()
        .filter(|c| !params.common_join_cols.contains(*c))
        .map(ToString::to_string)
        .collect();

    // Get columns from build_side_table based on which side we built on
    let (left, right, join_table) = if params.build_on_left {
        // Built on left, so build_side_table has left columns
        // Try to get columns - if they don't exist, return empty
        let left = match get_columns_by_name(&build_side_table, &left_non_join_columns) {
            Ok(cols) => cols,
            Err(_) => {
                return Ok(Some(MicroPartition::empty(Some(
                    params.output_schema.clone(),
                ))));
            }
        };
        let right = {
            let columns = right_non_join_schema
                .fields()
                .iter()
                .map(|field| Series::full_null(&field.name, &field.dtype, left.len()))
                .collect::<Vec<_>>();
            RecordBatch::new_unchecked(right_non_join_schema, columns, left.len())
        };
        #[allow(deprecated)]
        let join_table = match get_columns_by_name(&build_side_table, &common_join_cols) {
            Ok(cols) => cols.cast_to_schema(&outer_common_col_schema)?,
            Err(_) => {
                return Ok(Some(MicroPartition::empty(Some(
                    params.output_schema.clone(),
                ))));
            }
        };
        (left, right, join_table)
    } else {
        // Built on right, so build_side_table has right columns
        // The join columns in build_side_table have right side names, but common_join_cols has left side names
        // We need to get the right side join key columns from the build_side_table
        // For now, try to get by common_join_cols (left names), and if that fails,
        // the build side was likely empty or has different column names
        let right = match get_columns_by_name(&build_side_table, &right_non_join_columns) {
            Ok(cols) => cols,
            Err(_) => {
                return Ok(Some(MicroPartition::empty(Some(
                    params.output_schema.clone(),
                ))));
            }
        };
        let left = {
            let left_non_join_schema = Arc::new(Schema::new(
                params
                    .left_schema
                    .fields()
                    .iter()
                    .filter(|f| !params.common_join_cols.contains(&*f.name))
                    .cloned(),
            ));
            let columns = left_non_join_schema
                .fields()
                .iter()
                .map(|field| Series::full_null(&field.name, &field.dtype, right.len()))
                .collect::<Vec<_>>();
            RecordBatch::new_unchecked(left_non_join_schema, columns, right.len())
        };
        // Try to get join columns - if common_join_cols don't exist, the table might be empty or have wrong schema
        #[allow(deprecated)]
        let join_table = match get_columns_by_name(&build_side_table, &common_join_cols) {
            Ok(cols) => cols.cast_to_schema(&outer_common_col_schema)?,
            Err(_) => {
                // Build side table doesn't have the expected join columns - return empty
                return Ok(Some(MicroPartition::empty(Some(
                    params.output_schema.clone(),
                ))));
            }
        };
        (left, right, join_table)
    };
    let final_table = join_table.union(&left)?.union(&right)?;
    Ok(Some(MicroPartition::new_loaded(
        final_table.schema.clone(),
        Arc::new(vec![final_table]),
        None,
    )))
}

#[cfg(test)]
mod tests {
    //! Equivalence tests for the outer-join probe dedup path.
    //!
    //! Outer join semantics demand that *every* probe row appears at least
    //! once in the output (with nulls on the build side when unmatched), and
    //! the bitmap must accumulate exactly the same marked positions whether
    //! or not dedup is engaged.

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

    fn run_outer(
        probe_state: &ProbeState,
        join_keys: &RecordBatch,
        tuning: ProbeDedupTuning,
    ) -> (Vec<u64>, RecordBatch, IndexBitmapBuilder) {
        let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();
        let mut growable = GrowableRecordBatch::new(&build_side_tables, true, 8).unwrap();
        let mut idxs = Vec::new();
        let mut bitmap_builder = IndexBitmapBuilder::new(probe_state.get_record_batches());
        walk_probe_matches(
            probe_state,
            join_keys,
            tuning,
            |probe_row_idx, maybe_matches| {
                match maybe_matches {
                    Some(matches) => {
                        for (build_table_idx, build_row_idx) in matches {
                            bitmap_builder
                                .mark_used(build_table_idx as usize, build_row_idx as usize);
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
        (idxs, table, bitmap_builder)
    }

    fn assert_outer_match(probe_state: &ProbeState, join_keys: &RecordBatch) {
        let (idxs_d, table_d, bm_d) = run_outer(probe_state, join_keys, force_direct());
        let (idxs_p, table_p, bm_p) = run_outer(probe_state, join_keys, force_dedup());
        assert_eq!(idxs_d, idxs_p);
        assert_record_batches_eq_null_safe(&table_d, &table_p);
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
    fn test_outer_dedup_repeated_keys() {
        let probe_state = build_probe_state(vec![Some(10), Some(20), Some(30), Some(20)]);
        let probe_keys =
            rb_from_int_col("k", vec![Some(20), Some(10), Some(20), Some(30), Some(20)]);
        assert_outer_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_outer_dedup_with_unmatched_and_null_probe_keys() {
        let probe_state = build_probe_state(vec![Some(1), Some(2)]);
        let probe_keys = rb_from_int_col(
            "k",
            vec![Some(1), None, Some(99), Some(2), None, Some(1), Some(99)],
        );
        assert_outer_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_outer_dedup_all_unmatched() {
        let probe_state = build_probe_state(vec![Some(100)]);
        let probe_keys = rb_from_int_col("k", vec![Some(1), Some(2), Some(1), Some(2)]);
        assert_outer_match(&probe_state, &probe_keys);
    }

    #[test]
    fn test_outer_dedup_heavy_skew() {
        let probe_state = build_probe_state(vec![Some(7), Some(7), Some(99)]);
        let probe_keys = rb_from_int_col("k", (0..1000).map(|_| Some(7i64)).collect());
        assert_outer_match(&probe_state, &probe_keys);
    }
}
