//! Software prefetch helpers using ARMv8 `PRFM` instruction.
//!
//! These are tiny inline-asm wrappers around the `prfm` instruction. The
//! Rust nightly `_prefetch` intrinsic also exists but it is gated behind
//! `#![feature(stdarch_aarch64_prefetch)]` and exposes only a subset of
//! the available hints. Inline asm gives us direct access to all five
//! `prfm` policy bits.
//!
//! Hint encoding (from ARM ARM):
//!     `<type><target><policy>`
//!   * type   : `pld` (load), `pst` (store), `pli` (instruction)
//!   * target : `l1`, `l2`, `l3`
//!   * policy : `keep` (temporal — keep in cache), `strm` (streaming — do
//!              not pollute cache on reuse)
//!
//! On Kunpeng 920B the L1d is 64 KiB and the private L2 is 1.25 MiB per
//! core. For grouped aggregations the gather target buffer is typically
//! much larger than L1 and is touched only once per group → use
//! `pldl1strm` to avoid polluting hot data (group indices, hash tables).
//! For the *next group's index array* however we expect to re-read it
//! several times during inner-loop processing → use `pldl1keep`.

#[inline(always)]
pub unsafe fn prfm_l1_keep<T>(ptr: *const T) {
    // SAFETY: prefetch never faults — even null/invalid addresses are
    // simply ignored by hardware. Callers do not need to ensure the
    // pointer is dereferenceable.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{p}]",
            p = in(reg) ptr,
            options(nostack, preserves_flags, readonly),
        );
    }
}

#[inline(always)]
pub unsafe fn prfm_l1_strm<T>(ptr: *const T) {
    unsafe {
        core::arch::asm!(
            "prfm pldl1strm, [{p}]",
            p = in(reg) ptr,
            options(nostack, preserves_flags, readonly),
        );
    }
}

#[inline(always)]
pub unsafe fn prfm_l2_keep<T>(ptr: *const T) {
    unsafe {
        core::arch::asm!(
            "prfm pldl2keep, [{p}]",
            p = in(reg) ptr,
            options(nostack, preserves_flags, readonly),
        );
    }
}

#[inline(always)]
pub unsafe fn prfm_l2_strm<T>(ptr: *const T) {
    unsafe {
        core::arch::asm!(
            "prfm pldl2strm, [{p}]",
            p = in(reg) ptr,
            options(nostack, preserves_flags, readonly),
        );
    }
}
