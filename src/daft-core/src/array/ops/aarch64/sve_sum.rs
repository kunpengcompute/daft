//! SVE-accelerated grouped-sum kernels for Kunpeng 920B.
//!
//! Workload pattern: given a contiguous value buffer `values[..]` and a
//! list of u64 indices `group[..]`, compute `sum values[group[i]]`.
//!
//! This is a "gather + reduce" kernel. On x86 it would map to `vpgatherqq`
//! + horizontal add. On aarch64 + SVE we use `ld1d` (or `ld1w`) with a
//! vector index operand, then either a predicated `add` accumulator
//! (integer types) or a multi-accumulator `fadd` + `faddv` reduction
//! (float types).
//!
//! Vector-length-agnostic: the SVE loop uses `whilelt` + `incd`/`incw`
//! + `b.first` so it works on any SVE VL >= 128. On Kunpeng 920B
//! VL=128, giving 2 lanes (.d) or 4 lanes (.s).
//!
//! Safety: all kernels here are `unsafe fn`. The caller must ensure:
//!  * `values` is a valid pointer to at least `max(group) + 1` elements
//!  * Every `group[i]` is < the length of the underlying value buffer
//!  * The host CPU implements SVE (Kunpeng 920B does)
//!
//! Numerical notes:
//!  * Integer kernels (i64, u64) use modular (`wrapping_add`) arithmetic.
//!    Result is bitwise identical to a scalar `wrapping_add` reduction.
//!  * f64 kernel uses a 4-way independent accumulator + `faddv` reduce.
//!    The reduction order is unspecified ("recursive pairwise") so the
//!    result may differ by a few ULPs from a strict left-to-right scalar
//!    sum. This trade is intentional --see the design discussion in PRs.
//!  * f32 and i128 do not have efficient SVE paths on Kunpeng 920B (no
//!    32-bit gather under .d predication / no 128-bit SVE lanes), so we
//!    fall back to a scalar 4-way unrolled accumulator with PRFM
//!    prefetching. The 4-way split breaks the loop-carried dependency
//!    chain of a single accumulator, recovering most of the ILP that the
//!    SVE paths get from lane parallelism.

use arrow::buffer::NullBuffer;

use crate::array::ops::aarch64::prfm::{prfm_l1_keep, prfm_l1_strm};

// ---------------------------------------------------------------------------
// i64 SVE kernel
// ---------------------------------------------------------------------------

/// Sum `values[group[i]]` for `i in 0..group.len()` using SVE gather.
///
/// # Safety
/// * `values` must point to a buffer of i64 large enough that every
///   `group[i]` is a valid in-bounds element index.
/// * SVE must be available on the host CPU.
#[target_feature(enable = "sve")]
pub unsafe fn sve_grouped_sum_i64(values: *const i64, group: &[u64]) -> i64 {
    let n = group.len();
    if n == 0 {
        return 0;
    }
    let idx_ptr = group.as_ptr();
    let n_u64 = n as u64;
    let pos: u64 = 0;
    let mut sum: i64;
    // SAFETY: see function-level safety contract. The asm block only
    // performs aligned vector loads bounded by the `whilelt` predicate;
    // out-of-range lanes are masked off and never produce loads.
    unsafe {
        core::arch::asm!(
            // z0..z2 are clobbered; p0 is the loop predicate; p1 is all-true
            // for the final horizontal reduction.
            "mov     z2.d, #0",
            "ptrue   p1.d",
            "whilelt p0.d, {pos}, {n}",
            // Loop top.
            "2:",
            // z0 <- indices[pos..]. lsl #3 because each index is 8 bytes.
            "ld1d    {{z0.d}}, p0/z, [{idx}, {pos}, lsl #3]",
            // z1 <- values[z0]. SVE 64-bit gather, scaled by 8.
            "ld1d    {{z1.d}}, p0/z, [{val}, z0.d, lsl #3]",
            // z2 += z1 (under the predicate --masked lanes are no-ops).
            "add     z2.d, p0/m, z2.d, z1.d",
            // Advance cursor by the SVE D-element count (vector-length agnostic).
            "incd    {pos}",
            "whilelt p0.d, {pos}, {n}",
            // Branch back if the first lane is still active.
            "b.first 2b",
            // Final horizontal sum over all lanes (predicate = all-true).
            "uaddv   d3, p1, z2.d",
            "fmov    {sum}, d3",
            pos = inout(reg) pos => _,
            n   = in(reg) n_u64,
            idx = in(reg) idx_ptr,
            val = in(reg) values,
            sum = out(reg) sum,
            // Clobbered SIMD registers. On Kunpeng 920B (SVE VL=128) the
            // z0..z31 SVE registers share storage with v0..v31, so marking
            // v* as clobbered also marks z* clobbered. Rust's stable
            // named-register set does not include SVE predicate registers
            // p0..p15, but they are caller-saved per AAPCS64 SVE PCS and
            // Rust's codegen does not allocate them for normal code, so
            // omitting an explicit p-reg clobber is safe in practice.
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            options(nostack, readonly),
        );
    }
    sum
}

/// Same as [`sve_grouped_sum_i64`] but skips lanes whose underlying
/// Arrow validity bit is zero. The implementation does scalar null checks
/// for simplicity --we only enter this path if the array actually has nulls.
///
/// # Safety
/// Same as [`sve_grouped_sum_i64`]. `nulls` must be a valid Arrow null
/// buffer aligned to the same value buffer.
#[target_feature(enable = "sve")]
pub unsafe fn sve_grouped_sum_i64_with_nulls(
    values: *const i64,
    group: &[u64],
    nulls: &NullBuffer,
) -> Option<i64> {
    // The null path is dominated by branchy bitmap checks. We use a scalar
    // 4-way accumulator here; SVE gather under a dynamic predicate would
    // require materializing the predicate from the bitmap each iteration
    // which costs more than it saves at small/medium group sizes.
    let n = group.len();
    if n == 0 {
        return None;
    }
    let mut any = false;
    let (mut a0, mut a1, mut a2, mut a3) = (0i64, 0i64, 0i64, 0i64);
    let mut i = 0usize;
    while i + 4 <= n {
        // SAFETY: indices are guaranteed in-bounds by the caller.
        unsafe {
            let i0 = *group.get_unchecked(i) as usize;
            let i1 = *group.get_unchecked(i + 1) as usize;
            let i2 = *group.get_unchecked(i + 2) as usize;
            let i3 = *group.get_unchecked(i + 3) as usize;
            if nulls.is_valid(i0) {
                a0 = a0.wrapping_add(*values.add(i0));
                any = true;
            }
            if nulls.is_valid(i1) {
                a1 = a1.wrapping_add(*values.add(i1));
                any = true;
            }
            if nulls.is_valid(i2) {
                a2 = a2.wrapping_add(*values.add(i2));
                any = true;
            }
            if nulls.is_valid(i3) {
                a3 = a3.wrapping_add(*values.add(i3));
                any = true;
            }
        }
        i += 4;
    }
    while i < n {
        // SAFETY: see above.
        unsafe {
            let idx = *group.get_unchecked(i) as usize;
            if nulls.is_valid(idx) {
                a0 = a0.wrapping_add(*values.add(idx));
                any = true;
            }
        }
        i += 1;
    }
    if any {
        Some(a0.wrapping_add(a1).wrapping_add(a2).wrapping_add(a3))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// u64 SVE kernel --identical instruction sequence to i64 (add is sign-agnostic)
// ---------------------------------------------------------------------------

#[target_feature(enable = "sve")]
pub unsafe fn sve_grouped_sum_u64(values: *const u64, group: &[u64]) -> u64 {
    // SAFETY: cast preserves bit-pattern; sum semantics are identical for
    // wrapping integer arithmetic regardless of signedness.
    unsafe { sve_grouped_sum_i64(values as *const i64, group) as u64 }
}

#[target_feature(enable = "sve")]
pub unsafe fn sve_grouped_sum_u64_with_nulls(
    values: *const u64,
    group: &[u64],
    nulls: &NullBuffer,
) -> Option<u64> {
    // SAFETY: see sve_grouped_sum_u64.
    unsafe { sve_grouped_sum_i64_with_nulls(values as *const i64, group, nulls).map(|v| v as u64) }
}

// ---------------------------------------------------------------------------
// f64 SVE kernel --multi-accumulator + faddv
// ---------------------------------------------------------------------------

/// Sum `values[group[i]]` for `i in 0..group.len()` using SVE 64-bit
/// gather with a 4-way independent accumulator chain.
///
/// The kernel runs four `fadd` chains in parallel (z3, z4, z5, z6), each
/// fed by an independent gather. After the loop the four accumulators are
/// tree-reduced and `faddv` produces the scalar result. This breaks the
/// loop-carried dependency chain of single-accumulator FP reduction
/// (LLVM cannot perform this reordering itself because fadd is not
/// associative for IEEE-754).
///
/// Numerical accuracy: the reduction order is unspecified --`faddv`
/// performs a recursive pairwise add. Expect ULP-level differences vs a
/// strict scalar left-to-right sum.
///
/// # Safety
/// See [`sve_grouped_sum_i64`].
#[target_feature(enable = "sve")]
pub unsafe fn sve_grouped_sum_f64(values: *const f64, group: &[u64]) -> f64 {
    let n = group.len();
    if n == 0 {
        return 0.0;
    }
    let idx_ptr = group.as_ptr();
    let n_u64 = n as u64;

    // 4-way unrolled portion: process four SVE vectors per iteration. Each
    // iteration consumes 4xVL_d elements. For VL=128 that is 8 i64 indices.
    // Tail elements are handled by a single-accumulator loop below.
    let pos: u64 = 0;
    let mut sum: f64;

    // SAFETY: see function-level safety contract.
    unsafe {
        core::arch::asm!(
            // Compute the step size: 4 x VL_d (number of 64-bit lanes in 4 SVE regs)
            "cntd    {step}, all, mul #4",
            // Initialise four accumulators to +0.0.
            "mov     z3.d, #0",
            "mov     z4.d, #0",
            "mov     z5.d, #0",
            "mov     z6.d, #0",
            "ptrue   p1.d",
            // Main 4-way unrolled loop. We only enter it while at least
            // `4 x VL_d` elements remain (i.e. all four lanes can be fully
            // populated under an all-true predicate). This avoids having
            // to manage four distinct predicates in the steady-state loop.
            "add     {tmp}, {pos}, {step}",
            "cmp     {tmp}, {n}",
            "b.hi    3f",
            "2:",
            // -- chunk 0 --
            "ld1d    {{z0.d}}, p1/z, [{idx}, {pos}, lsl #3]",
            "ld1d    {{z1.d}}, p1/z, [{val}, z0.d, lsl #3]",
            "fadd    z3.d, p1/m, z3.d, z1.d",
            "incd    {pos}",
            // -- chunk 1 --
            "ld1d    {{z0.d}}, p1/z, [{idx}, {pos}, lsl #3]",
            "ld1d    {{z1.d}}, p1/z, [{val}, z0.d, lsl #3]",
            "fadd    z4.d, p1/m, z4.d, z1.d",
            "incd    {pos}",
            // -- chunk 2 --
            "ld1d    {{z0.d}}, p1/z, [{idx}, {pos}, lsl #3]",
            "ld1d    {{z1.d}}, p1/z, [{val}, z0.d, lsl #3]",
            "fadd    z5.d, p1/m, z5.d, z1.d",
            "incd    {pos}",
            // -- chunk 3 --
            "ld1d    {{z0.d}}, p1/z, [{idx}, {pos}, lsl #3]",
            "ld1d    {{z1.d}}, p1/z, [{val}, z0.d, lsl #3]",
            "fadd    z6.d, p1/m, z6.d, z1.d",
            "incd    {pos}",
            // Continue if another 4x VL_d chunk fits.
            "add     {tmp}, {pos}, {step}",
            "cmp     {tmp}, {n}",
            "b.ls    2b",
            "3:",
            // Tail loop: handle remaining < 4xVL_d elements one SVE vector
            // at a time using a `whilelt` predicate.
            "whilelt p0.d, {pos}, {n}",
            "4:",
            "ld1d    {{z0.d}}, p0/z, [{idx}, {pos}, lsl #3]",
            "ld1d    {{z1.d}}, p0/z, [{val}, z0.d, lsl #3]",
            "fadd    z3.d, p0/m, z3.d, z1.d",
            "incd    {pos}",
            "whilelt p0.d, {pos}, {n}",
            "b.first 4b",
            // Tree-reduce 4 accumulators -> 1 accumulator (all-true predicate).
            "fadd    z3.d, p1/m, z3.d, z4.d",
            "fadd    z5.d, p1/m, z5.d, z6.d",
            "fadd    z3.d, p1/m, z3.d, z5.d",
            // Horizontal reduction across all lanes.
            "faddv   d7, p1, z3.d",
            "fmov    {sum:d}, d7",
            pos  = inout(reg) pos => _,
            n    = in(reg) n_u64,
            idx  = in(reg) idx_ptr,
            val  = in(reg) values,
            sum  = out(vreg) sum,
            step = out(reg) _,
            tmp  = out(reg) _,
            // See the i64 kernel for the SIMD/SVE clobber rationale.
            out("v0") _, out("v1") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _,
            options(nostack, readonly),
        );
    }
    sum
}

#[target_feature(enable = "sve")]
pub unsafe fn sve_grouped_sum_f64_with_nulls(
    values: *const f64,
    group: &[u64],
    nulls: &NullBuffer,
) -> Option<f64> {
    let n = group.len();
    if n == 0 {
        return None;
    }
    let mut any = false;
    // 4-way scalar accumulator; same rationale as the i64 null path.
    let (mut a0, mut a1, mut a2, mut a3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut i = 0usize;
    while i + 4 <= n {
        // SAFETY: indices are guaranteed in-bounds by the caller.
        unsafe {
            let i0 = *group.get_unchecked(i) as usize;
            let i1 = *group.get_unchecked(i + 1) as usize;
            let i2 = *group.get_unchecked(i + 2) as usize;
            let i3 = *group.get_unchecked(i + 3) as usize;
            if nulls.is_valid(i0) {
                a0 += *values.add(i0);
                any = true;
            }
            if nulls.is_valid(i1) {
                a1 += *values.add(i1);
                any = true;
            }
            if nulls.is_valid(i2) {
                a2 += *values.add(i2);
                any = true;
            }
            if nulls.is_valid(i3) {
                a3 += *values.add(i3);
                any = true;
            }
        }
        i += 4;
    }
    while i < n {
        // SAFETY: see above.
        unsafe {
            let idx = *group.get_unchecked(i) as usize;
            if nulls.is_valid(idx) {
                a0 += *values.add(idx);
                any = true;
            }
        }
        i += 1;
    }
    if any { Some((a0 + a1) + (a2 + a3)) } else { None }
}

// ---------------------------------------------------------------------------
// f32 --scalar 4-way accumulator
// ---------------------------------------------------------------------------
//
// f32 with 64-bit indices is awkward on SVE: a .d-lane gather scaled by 4
// would deliver f32 values in the low 32 bits of each .d lane with junk in
// the high bits, requiring an extra `uzp1`/`fcvt` step to pack or convert.
// The throughput recovered does not exceed what a 4-way scalar accumulator
// already achieves on a 128-bit-VL machine (since we can only fit two
// .d lanes anyway). We therefore use scalar 4-way + PRFM here.

/// 4-way unrolled scalar f32 sum with software prefetching.
///
/// # Safety
/// See module docs.
#[inline]
pub unsafe fn sve_grouped_sum_f32(values: *const f32, group: &[u64]) -> f32 {
    let n = group.len();
    if n == 0 {
        return 0.0;
    }
    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut i = 0usize;
    while i + 4 <= n {
        // SAFETY: caller guarantees indices are in-bounds.
        unsafe {
            // Streaming prefetch 16 indices ahead --gather targets do not
            // benefit from cache retention (each element is touched once).
            if i + 16 < n {
                let pf_idx = *group.get_unchecked(i + 16) as usize;
                prfm_l1_strm(values.add(pf_idx));
            }
            let i0 = *group.get_unchecked(i) as usize;
            let i1 = *group.get_unchecked(i + 1) as usize;
            let i2 = *group.get_unchecked(i + 2) as usize;
            let i3 = *group.get_unchecked(i + 3) as usize;
            a0 += *values.add(i0);
            a1 += *values.add(i1);
            a2 += *values.add(i2);
            a3 += *values.add(i3);
        }
        i += 4;
    }
    while i < n {
        // SAFETY: see above.
        unsafe {
            let idx = *group.get_unchecked(i) as usize;
            a0 += *values.add(idx);
        }
        i += 1;
    }
    (a0 + a1) + (a2 + a3)
}

#[inline]
pub unsafe fn sve_grouped_sum_f32_with_nulls(
    values: *const f32,
    group: &[u64],
    nulls: &NullBuffer,
) -> Option<f32> {
    let n = group.len();
    if n == 0 {
        return None;
    }
    let mut any = false;
    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut i = 0usize;
    while i + 4 <= n {
        // SAFETY: caller guarantees indices are in-bounds.
        unsafe {
            let i0 = *group.get_unchecked(i) as usize;
            let i1 = *group.get_unchecked(i + 1) as usize;
            let i2 = *group.get_unchecked(i + 2) as usize;
            let i3 = *group.get_unchecked(i + 3) as usize;
            if nulls.is_valid(i0) {
                a0 += *values.add(i0);
                any = true;
            }
            if nulls.is_valid(i1) {
                a1 += *values.add(i1);
                any = true;
            }
            if nulls.is_valid(i2) {
                a2 += *values.add(i2);
                any = true;
            }
            if nulls.is_valid(i3) {
                a3 += *values.add(i3);
                any = true;
            }
        }
        i += 4;
    }
    while i < n {
        // SAFETY: see above.
        unsafe {
            let idx = *group.get_unchecked(i) as usize;
            if nulls.is_valid(idx) {
                a0 += *values.add(idx);
                any = true;
            }
        }
        i += 1;
    }
    if any { Some((a0 + a1) + (a2 + a3)) } else { None }
}

// ---------------------------------------------------------------------------
// i128 (Decimal128) --scalar 4-way accumulator with aggressive PRFM
// ---------------------------------------------------------------------------
//
// SVE has no 128-bit integer lanes (max element size is .d = 64 bits) so a
// vector path is unavailable. We instead exploit:
//   * 4-way ILP with independent accumulators
//   * `ldp x, x, [..]` pair loads which LLVM emits for *const i128 reads
//   * Aggressive (distance-32) PRFM since each i128 element is 16 bytes
//     and the working set grows fast.

/// 4-way unrolled scalar i128 sum with software prefetching.
///
/// # Safety
/// See module docs.
#[inline]
pub unsafe fn sve_grouped_sum_i128(values: *const i128, group: &[u64]) -> i128 {
    let n = group.len();
    if n == 0 {
        return 0;
    }
    let (mut s0, mut s1, mut s2, mut s3) = (0i128, 0i128, 0i128, 0i128);
    let mut i = 0usize;
    while i + 4 <= n {
        // SAFETY: caller guarantees indices are in-bounds.
        unsafe {
            // i128 = 16B; use a longer prefetch distance to hide DRAM latency.
            if i + 32 < n {
                let pf_idx = *group.get_unchecked(i + 32) as usize;
                prfm_l1_strm(values.add(pf_idx));
            }
            let i0 = *group.get_unchecked(i) as usize;
            let i1 = *group.get_unchecked(i + 1) as usize;
            let i2 = *group.get_unchecked(i + 2) as usize;
            let i3 = *group.get_unchecked(i + 3) as usize;
            s0 = s0.wrapping_add(*values.add(i0));
            s1 = s1.wrapping_add(*values.add(i1));
            s2 = s2.wrapping_add(*values.add(i2));
            s3 = s3.wrapping_add(*values.add(i3));
        }
        i += 4;
    }
    while i < n {
        // SAFETY: see above.
        unsafe {
            let idx = *group.get_unchecked(i) as usize;
            s0 = s0.wrapping_add(*values.add(idx));
        }
        i += 1;
    }
    s0.wrapping_add(s1).wrapping_add(s2).wrapping_add(s3)
}

#[inline]
pub unsafe fn sve_grouped_sum_i128_with_nulls(
    values: *const i128,
    group: &[u64],
    nulls: &NullBuffer,
) -> Option<i128> {
    let n = group.len();
    if n == 0 {
        return None;
    }
    let mut any = false;
    let (mut s0, mut s1, mut s2, mut s3) = (0i128, 0i128, 0i128, 0i128);
    let mut i = 0usize;
    while i + 4 <= n {
        // SAFETY: caller guarantees indices are in-bounds.
        unsafe {
            let i0 = *group.get_unchecked(i) as usize;
            let i1 = *group.get_unchecked(i + 1) as usize;
            let i2 = *group.get_unchecked(i + 2) as usize;
            let i3 = *group.get_unchecked(i + 3) as usize;
            if nulls.is_valid(i0) {
                s0 = s0.wrapping_add(*values.add(i0));
                any = true;
            }
            if nulls.is_valid(i1) {
                s1 = s1.wrapping_add(*values.add(i1));
                any = true;
            }
            if nulls.is_valid(i2) {
                s2 = s2.wrapping_add(*values.add(i2));
                any = true;
            }
            if nulls.is_valid(i3) {
                s3 = s3.wrapping_add(*values.add(i3));
                any = true;
            }
        }
        i += 4;
    }
    while i < n {
        // SAFETY: see above.
        unsafe {
            let idx = *group.get_unchecked(i) as usize;
            if nulls.is_valid(idx) {
                s0 = s0.wrapping_add(*values.add(idx));
                any = true;
            }
        }
        i += 1;
    }
    if any {
        Some(s0.wrapping_add(s1).wrapping_add(s2).wrapping_add(s3))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Outer-loop prefetch helper
// ---------------------------------------------------------------------------

/// Prefetch the head of a group's index array into L1 (temporal).
/// Used by the dispatcher to hide pointer-chasing between groups.
///
/// # Safety
/// PRFM is always safe --invalid pointers are silently ignored.
#[inline(always)]
pub unsafe fn prefetch_next_group(group: &[u64]) {
    if !group.is_empty() {
        // SAFETY: PRFM cannot fault. The pointer is a valid slice ptr.
        unsafe {
            prfm_l1_keep(group.as_ptr());
        }
    }
}
