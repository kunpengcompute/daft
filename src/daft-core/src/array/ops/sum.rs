use common_error::DaftResult;
#[cfg(not(target_arch = "aarch64"))]
use daft_arrow::array::Array;

use super::{DaftSumAggable, as_arrow::AsArrow};
use crate::{array::ops::GroupIndices, datatypes::*};
#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use arrow::array::Array;

// The grouped-sum implementation is split into two arms so that we can
// override the body for aarch64 (Kunpeng 920B) while keeping every other
// target on the original fold-based arrow2 implementation.
//
// The aarch64 path delegates the inner reduction to SVE kernels in
// `crate::array::ops::aarch64::sve_sum`.
macro_rules! impl_daft_numeric_agg {
    ($T:ident, $AggType: ident, $sve_kernel:ident, $sve_kernel_nulls:ident) => {
        impl DaftSumAggable for &DataArray<$T> {
            type Output = DaftResult<DataArray<$T>>;

            fn sum(&self) -> Self::Output {
                let primitive_arr = self.as_arrow2();
                let sum_value = daft_arrow::compute::aggregate::sum_primitive(primitive_arr);
                Ok(DataArray::<$T>::from_iter(
                    self.field.clone(),
                    std::iter::once(sum_value),
                ))
            }

            #[cfg(target_arch = "aarch64")]
            fn grouped_sum(&self, groups: &GroupIndices) -> Self::Output {
                use crate::array::ops::aarch64::sve_sum;

                let arrow_array = self.as_arrow()?;
                let values_ptr = arrow_array.values().as_ptr();
                // Only take the null-aware path when the array actually
                // contains nulls; an array may carry an (empty) null buffer
                // with null_count == 0.
                let nulls = if self.null_count() > 0 {
                    arrow_array.nulls()
                } else {
                    None
                };
                let n_groups = groups.len();

                let mut out: Vec<Option<$AggType>> = Vec::with_capacity(n_groups);

                if let Some(nb) = nulls {
                    for (gi, g) in groups.iter().enumerate() {
                        if gi + 4 < n_groups {
                            // SAFETY: PRFM never faults. The slice ptr is valid.
                            unsafe {
                                sve_sum::prefetch_next_group(&groups[gi + 4]);
                            }
                        }
                        // SAFETY: indices in `g` are produced by Daft's groupers
                        // (see `make_groups` impls) and are always in-bounds for
                        // the value buffer.
                        let v = unsafe { sve_sum::$sve_kernel_nulls(values_ptr, g, nb) };
                        out.push(v);
                    }
                } else {
                    for (gi, g) in groups.iter().enumerate() {
                        if gi + 4 < n_groups {
                            // SAFETY: PRFM never faults.
                            unsafe {
                                sve_sum::prefetch_next_group(&groups[gi + 4]);
                            }
                        }
                        // SAFETY: see above.
                        let v = unsafe { sve_sum::$sve_kernel(values_ptr, g) };
                        out.push(Some(v));
                    }
                }

                Ok(DataArray::<$T>::from_iter(
                    self.field.clone(),
                    out.into_iter(),
                ))
            }

            #[cfg(not(target_arch = "aarch64"))]
            fn grouped_sum(&self, groups: &GroupIndices) -> Self::Output {
                let arrow_array = self.as_arrow2();
                let sum_per_group = if arrow_array.null_count() > 0 {
                    DataArray::<$T>::from_iter(
                        self.field.clone(),
                        groups.iter().map(|g| {
                            g.iter().fold(None, |acc, index| {
                                let idx = *index as usize;
                                match (acc, arrow_array.is_null(idx)) {
                                    (acc, true) => acc,
                                    (None, false) => Some(arrow_array.value(idx)),
                                    (Some(acc), false) => Some(acc + arrow_array.value(idx)),
                                }
                            })
                        }),
                    )
                } else {
                    DataArray::<$T>::from_values_iter(
                        self.field.clone(),
                        groups.iter().map(|g| {
                            g.iter().fold(0 as $AggType, |acc, index| {
                                let idx = *index as usize;
                                acc + unsafe { arrow_array.value_unchecked(idx) }
                            })
                        }),
                    )
                };

                Ok(sum_per_group)
            }
        }
    };
}

impl_daft_numeric_agg!(
    Int64Type,
    i64,
    sve_grouped_sum_i64,
    sve_grouped_sum_i64_with_nulls
);
impl_daft_numeric_agg!(
    UInt64Type,
    u64,
    sve_grouped_sum_u64,
    sve_grouped_sum_u64_with_nulls
);
impl_daft_numeric_agg!(
    Float32Type,
    f32,
    sve_grouped_sum_f32,
    sve_grouped_sum_f32_with_nulls
);
impl_daft_numeric_agg!(
    Float64Type,
    f64,
    sve_grouped_sum_f64,
    sve_grouped_sum_f64_with_nulls
);
impl_daft_numeric_agg!(
    Decimal128Type,
    i128,
    sve_grouped_sum_i128,
    sve_grouped_sum_i128_with_nulls
);
