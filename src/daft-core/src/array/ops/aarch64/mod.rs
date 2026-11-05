//! Kunpeng 920B (aarch64 + SVE) optimized aggregation kernels.
//!
//! This module contains SIMD-accelerated kernels for grouped aggregation
//! primitives. It is gated on `target_arch = "aarch64"` and assumes the
//! target supports SVE (Scalable Vector Extension). For the supported
//! deployment target (HiSilicon Kunpeng 920B), the SVE vector length is
//! 128 bits (2 lanes for 64-bit values, 4 lanes for 32-bit values).
//!
//! All public kernels in this module are `unsafe fn` because they perform
//! raw pointer arithmetic and rely on the caller to:
//!   1. Provide a valid value buffer (live for the duration of the call)
//!   2. Provide indices that are all in-bounds for that buffer
//!   3. Run on hardware that supports the SVE feature
//!
//! The kernels are also marked `#[target_feature(enable = "sve")]` so that
//! the integrated assembler will accept SVE instructions even if the
//! crate-level target-feature set is restrictive.
//!
//! Build instructions (for Kunpeng 920B): see
//! `docs/kunpeng-920b-build-guide.md` at the repository root.

#![cfg(target_arch = "aarch64")]

pub mod prfm;
pub mod sve_sum;
