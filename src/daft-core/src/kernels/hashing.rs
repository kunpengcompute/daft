use std::{
    hash::{BuildHasher, Hasher},
    ptr,
};

use daft_arrow::{
    array::{
        Array, BinaryArray, BooleanArray, FixedSizeBinaryArray, NullArray, PrimitiveArray,
        Utf8Array,
    },
    datatypes::{DataType, PhysicalType},
    error::{Error, Result},
    types::{NativeType, Offset},
};
use daft_hash::{HashFunctionKind, MurBuildHasher, Sha1Hasher};
use xxhash_rust::{
    const_xxh3, const_xxh32, const_xxh64, xxh3::xxh3_64_with_seed, xxh32::xxh32, xxh64::xxh64,
};

// XXH3 constants for the len == 8 short-input path.
const XXH3_PRIME_MX2: u64 = 0x9FB2_1C65_1E98_DF25;
// SECRET[8..16] ^ SECRET[16..24] of the default XXH3 secret.
const XXH3_BITFLIP_LEN8_BASE: u64 = 0xC73A_B174_C5EC_D5A2;

/// Specialized XXH3 (64-bit) hash for an exactly 8-byte input.
///
/// On little-endian targets this is bit-identical to
/// `xxh3_64_with_seed(&v.to_le_bytes(), seed)`, but avoids the generic
/// `&[u8]` dispatch.
#[inline(always)]
fn xxh3_64_8b(v: u64, seed: u64) -> u64 {
    // Short-input seed transformation: seed' = seed XOR (bswap32(seed_lo32) << 32)
    let seed_mix = seed ^ (((seed as u32).swap_bytes() as u64) << 32);
    let bitflip = XXH3_BITFLIP_LEN8_BASE.wrapping_sub(seed_mix);
    let keyed = v.rotate_left(32) ^ bitflip;
    let mut h = keyed ^ keyed.rotate_left(49) ^ keyed.rotate_left(24);
    h = h.wrapping_mul(XXH3_PRIME_MX2);
    h ^= (h >> 35).wrapping_add(8);
    h = h.wrapping_mul(XXH3_PRIME_MX2);
    h ^ (h >> 28)
}

/// Hash `len` values stored as contiguous 8-byte words.
///
/// # Safety
///
/// `src` must be 8-byte aligned and valid for `len` reads, `dst` must be valid
/// for `len` writes, and the two ranges must not overlap.
#[inline]
unsafe fn xxh3_64_8b_dense(src: *const u64, dst: *mut u64, len: usize) {
    const UNROLL: usize = 8;
    let chunks = len / UNROLL;
    let tail_start = chunks * UNROLL;

    // SAFETY: the caller guarantees `src`/`dst` validity and non-overlap; all
    // indices below are within `0..len`, so every read/write is in bounds.
    unsafe {
        for i in 0..chunks {
            let b = i * UNROLL;

            #[cfg(target_arch = "aarch64")]
            std::arch::asm!(
                "prfm pldl1keep, [{p}, #256]",
                p = in(reg) src.add(b),
                options(nostack, preserves_flags, readonly),
            );

            let v0 = ptr::read(src.add(b));
            let v1 = ptr::read(src.add(b + 1));
            let v2 = ptr::read(src.add(b + 2));
            let v3 = ptr::read(src.add(b + 3));
            let v4 = ptr::read(src.add(b + 4));
            let v5 = ptr::read(src.add(b + 5));
            let v6 = ptr::read(src.add(b + 6));
            let v7 = ptr::read(src.add(b + 7));

            let h0 = xxh3_64_8b(v0, 0);
            let h1 = xxh3_64_8b(v1, 0);
            let h2 = xxh3_64_8b(v2, 0);
            let h3 = xxh3_64_8b(v3, 0);
            let h4 = xxh3_64_8b(v4, 0);
            let h5 = xxh3_64_8b(v5, 0);
            let h6 = xxh3_64_8b(v6, 0);
            let h7 = xxh3_64_8b(v7, 0);

            ptr::write(dst.add(b), h0);
            ptr::write(dst.add(b + 1), h1);
            ptr::write(dst.add(b + 2), h2);
            ptr::write(dst.add(b + 3), h3);
            ptr::write(dst.add(b + 4), h4);
            ptr::write(dst.add(b + 5), h5);
            ptr::write(dst.add(b + 6), h6);
            ptr::write(dst.add(b + 7), h7);
        }

        for i in tail_start..len {
            let v = ptr::read(src.add(i));
            ptr::write(dst.add(i), xxh3_64_8b(v, 0));
        }
    }
}

/// Vectorized hash loop over a primitive array.
///
/// Reads the array's values through a raw pointer and writes the hashes into
/// `dst`. The no-null, unseeded case is unrolled 4x; nulls hash to `null_hash`
/// and seeded rows use the corresponding per-row seed value, matching the
/// scalar baseline.
///
/// # Safety
///
/// `dst` must be valid for writes of `array.len()` `u64` values and must not
/// overlap the array's value buffer. A `seed` array passed here must have the
/// same length as `array` (enforced by [`hash`]).
#[inline]
unsafe fn xxh_loop<T, F>(
    array: &PrimitiveArray<T>,
    seed: Option<&PrimitiveArray<u64>>,
    dst: *mut u64,
    null_hash: u64,
    f: F,
) where
    T: NativeType,
    F: Fn(&[u8], u64) -> u64,
{
    let len = array.len();
    let values = array.values().as_slice().as_ptr();

    // SAFETY: `values` is valid for `len` reads and `dst` for `len` writes;
    // every index used below is in `0..len`, and `s` (when present) has
    // `len` elements.
    unsafe {
        match (array.null_count() == 0, seed) {
            (true, None) => {
                let chunks = len / 4;
                let tail_start = chunks * 4;
                for i in 0..chunks {
                    let b = i * 4;
                    let v0 = ptr::read(values.add(b));
                    let v1 = ptr::read(values.add(b + 1));
                    let v2 = ptr::read(values.add(b + 2));
                    let v3 = ptr::read(values.add(b + 3));
                    let h0 = f(v0.to_le_bytes().as_ref(), 0);
                    let h1 = f(v1.to_le_bytes().as_ref(), 0);
                    let h2 = f(v2.to_le_bytes().as_ref(), 0);
                    let h3 = f(v3.to_le_bytes().as_ref(), 0);
                    ptr::write(dst.add(b), h0);
                    ptr::write(dst.add(b + 1), h1);
                    ptr::write(dst.add(b + 2), h2);
                    ptr::write(dst.add(b + 3), h3);
                }
                for i in tail_start..len {
                    let v = ptr::read(values.add(i));
                    ptr::write(dst.add(i), f(v.to_le_bytes().as_ref(), 0));
                }
            }
            (true, Some(seeds)) => {
                let s = seeds.values().as_slice().as_ptr();
                for i in 0..len {
                    let v = ptr::read(values.add(i));
                    let sd = *s.add(i);
                    ptr::write(dst.add(i), f(v.to_le_bytes().as_ref(), sd));
                }
            }
            (false, None) => {
                for i in 0..len {
                    let h = if array.is_valid(i) {
                        let v = ptr::read(values.add(i));
                        f(v.to_le_bytes().as_ref(), 0)
                    } else {
                        null_hash
                    };
                    ptr::write(dst.add(i), h);
                }
            }
            (false, Some(seeds)) => {
                let s = seeds.values().as_slice().as_ptr();
                for i in 0..len {
                    let h = if array.is_valid(i) {
                        let v = ptr::read(values.add(i));
                        f(v.to_le_bytes().as_ref(), *s.add(i))
                    } else {
                        null_hash
                    };
                    ptr::write(dst.add(i), h);
                }
            }
        }
    }
}

fn hash_primitive<T: NativeType>(
    array: &PrimitiveArray<T>,
    seed: Option<&PrimitiveArray<u64>>,
    hash_function: HashFunctionKind,
) -> PrimitiveArray<u64> {
    let len = array.len();

    match hash_function {
        HashFunctionKind::MurmurHash3 => {
            let hasher = MurBuildHasher::new(seed.and_then(|s| s.get(0)).unwrap_or(42) as u32);
            let hashes = array
                .iter()
                .map(|v| {
                    let mut hasher = hasher.build_hasher();
                    match v {
                        Some(v) => {
                            hasher.write(v.to_le_bytes().as_ref());
                            hasher.finish()
                        }
                        None => {
                            hasher.write(b"");
                            hasher.finish()
                        }
                    }
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::Sha1 => {
            let hashes = array
                .iter()
                .map(|v| {
                    let mut hasher = Sha1Hasher::default();
                    match v {
                        Some(v) => {
                            hasher.write(v.to_le_bytes().as_ref());
                            hasher.finish()
                        }
                        None => {
                            hasher.write(b"");
                            hasher.finish()
                        }
                    }
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::XxHash32 => {
            const NULL_HASH: u64 = const_xxh32::xxh32(b"", 0) as u64;
            let mut hashes = vec![0u64; len];
            // SAFETY: `hashes` has `len` elements, matching `array`, and its
            // buffer cannot alias the array's immutable value buffer.
            unsafe {
                xxh_loop(array, seed, hashes.as_mut_ptr(), NULL_HASH, |b, s| {
                    xxh32(b, s as u32) as u64
                });
            }
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::XxHash64 => {
            const NULL_HASH: u64 = const_xxh64::xxh64(b"", 0);
            let mut hashes = vec![0u64; len];
            // SAFETY: `hashes` has `len` elements, matching `array`, and its
            // buffer cannot alias the array's immutable value buffer.
            unsafe {
                xxh_loop(array, seed, hashes.as_mut_ptr(), NULL_HASH, xxh64);
            }
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::XxHash3_64 => {
            const NULL_HASH: u64 = const_xxh3::xxh3_64(b"");
            let mut hashes = vec![0u64; len];
            let dst = hashes.as_mut_ptr();
            let values = array.values().as_slice().as_ptr();
            // The dense 8-byte path requires: no nulls, no seed, 8-byte values,
            // and an 8-byte aligned value buffer (raw `u64` reads).
            if std::mem::size_of::<T>() == 8
                && array.null_count() == 0
                && seed.is_none()
                && (values as usize) % std::mem::align_of::<u64>() == 0
            {
                // SAFETY: `values` is valid for `len` reads of 8-byte values,
                // `dst` is valid for `len` writes, and the two allocations are
                // distinct. Alignment was checked above.
                unsafe {
                    xxh3_64_8b_dense(values as *const u64, dst, len);
                }
            } else {
                // SAFETY: `hashes` has `len` elements, matching `array`.
                unsafe {
                    xxh_loop(array, seed, dst, NULL_HASH, xxh3_64_with_seed);
                }
            }
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
    }
}

fn hash_boolean(
    array: &BooleanArray,
    seed: Option<&PrimitiveArray<u64>>,
    hash_function: HashFunctionKind,
) -> PrimitiveArray<u64> {
    fn xxhash<
        const NULL_HASH: u64,
        const TRUE_HASH: u64,
        const FALSE_HASH: u64,
        F: Fn(&[u8], u64) -> u64,
    >(
        array: &BooleanArray,
        seed: Option<&PrimitiveArray<u64>>,
        f: F,
    ) -> PrimitiveArray<u64> {
        let hashes = if let Some(seed) = seed {
            array
                .iter()
                .zip(seed.values_iter())
                .map(|(v, s)| match v {
                    Some(true) => f(b"1", *s),
                    Some(false) => f(b"0", *s),
                    None => NULL_HASH,
                })
                .collect::<Vec<_>>()
        } else {
            array
                .iter()
                .map(|v| match v {
                    Some(true) => TRUE_HASH,
                    Some(false) => FALSE_HASH,
                    None => NULL_HASH,
                })
                .collect::<Vec<_>>()
        };
        PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
    }

    match hash_function {
        HashFunctionKind::MurmurHash3 => {
            let hasher = MurBuildHasher::new(seed.and_then(|s| s.get(0)).unwrap_or(42) as u32);
            let hashes = array
                .iter()
                .map(|v| {
                    let mut hasher = hasher.build_hasher();
                    match v {
                        Some(true) => {
                            hasher.write(b"1");
                            hasher.finish()
                        }
                        Some(false) => {
                            hasher.write(b"0");
                            hasher.finish()
                        }
                        None => {
                            hasher.write(b"");
                            hasher.finish()
                        }
                    }
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::Sha1 => {
            let hashes = array
                .iter()
                .map(|v| {
                    let mut hasher = Sha1Hasher::default();
                    match v {
                        Some(true) => {
                            hasher.write(b"1");
                            hasher.finish()
                        }
                        Some(false) => {
                            hasher.write(b"0");
                            hasher.finish()
                        }
                        None => {
                            hasher.write(b"");
                            hasher.finish()
                        }
                    }
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::XxHash32 => {
            const NULL_HASH: u64 = const_xxh32::xxh32(b"", 0) as u64;
            const FALSE_HASH: u64 = const_xxh32::xxh32(b"0", 0) as u64;
            const TRUE_HASH: u64 = const_xxh32::xxh32(b"1", 0) as u64;
            xxhash::<NULL_HASH, TRUE_HASH, FALSE_HASH, _>(array, seed, |v, s| {
                xxh32(v, s as u32) as u64
            })
        }
        HashFunctionKind::XxHash64 => {
            const NULL_HASH: u64 = const_xxh64::xxh64(b"", 0);
            const FALSE_HASH: u64 = const_xxh64::xxh64(b"0", 0);
            const TRUE_HASH: u64 = const_xxh64::xxh64(b"1", 0);
            xxhash::<NULL_HASH, TRUE_HASH, FALSE_HASH, _>(array, seed, xxh64)
        }
        HashFunctionKind::XxHash3_64 => {
            const NULL_HASH: u64 = const_xxh3::xxh3_64(b"");
            const FALSE_HASH: u64 = const_xxh3::xxh3_64(b"0");
            const TRUE_HASH: u64 = const_xxh3::xxh3_64(b"1");
            xxhash::<NULL_HASH, TRUE_HASH, FALSE_HASH, _>(array, seed, xxh3_64_with_seed)
        }
    }
}

fn hash_null(
    array: &NullArray,
    seed: Option<&PrimitiveArray<u64>>,
    hash_function: HashFunctionKind,
) -> PrimitiveArray<u64> {
    fn xxhash<const NULL_HASH: u64, F: Fn(&[u8], u64) -> u64>(
        len: usize,
        seed: Option<&PrimitiveArray<u64>>,
        f: F,
    ) -> PrimitiveArray<u64> {
        let hashes = if let Some(seed) = seed {
            seed.values_iter().map(|s| f(b"", *s)).collect::<Vec<_>>()
        } else {
            (0..len).map(|_| NULL_HASH).collect::<Vec<_>>()
        };
        PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
    }

    match hash_function {
        HashFunctionKind::MurmurHash3 => {
            let hasher = MurBuildHasher::new(seed.and_then(|s| s.get(0)).unwrap_or(42) as u32);
            let hashes = (0..array.len())
                .map(|_| {
                    let mut hasher = hasher.build_hasher();
                    hasher.write(b"");
                    hasher.finish()
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::Sha1 => {
            let hashes = (0..array.len())
                .map(|_| {
                    let mut hasher = Sha1Hasher::default();
                    hasher.write(b"");
                    hasher.finish()
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::XxHash32 => {
            const NULL_HASH: u64 = const_xxh32::xxh32(b"", 0) as u64;
            xxhash::<NULL_HASH, _>(array.len(), seed, |v, s| xxh32(v, s as u32) as u64)
        }
        HashFunctionKind::XxHash64 => {
            const NULL_HASH: u64 = const_xxh64::xxh64(b"", 0);
            xxhash::<NULL_HASH, _>(array.len(), seed, xxh64)
        }
        HashFunctionKind::XxHash3_64 => {
            const NULL_HASH: u64 = const_xxh3::xxh3_64(b"");
            xxhash::<NULL_HASH, _>(array.len(), seed, xxh3_64_with_seed)
        }
    }
}

fn hash_binary<O: Offset>(
    array: &BinaryArray<O>,
    seed: Option<&PrimitiveArray<u64>>,
    hash_function: HashFunctionKind,
) -> PrimitiveArray<u64> {
    fn xxhash<O: Offset, F: Fn(&[u8], u64) -> u64>(
        array: &BinaryArray<O>,
        seed: Option<&PrimitiveArray<u64>>,
        f: F,
    ) -> PrimitiveArray<u64> {
        let hashes = if let Some(seed) = seed {
            array
                .values_iter()
                .zip(seed.values_iter())
                .map(|(v, s)| f(v, *s))
                .collect::<Vec<_>>()
        } else {
            array.values_iter().map(|v| f(v, 0)).collect::<Vec<_>>()
        };
        PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
    }

    match hash_function {
        HashFunctionKind::XxHash32 => xxhash(array, seed, |v, s| xxh32(v, s as u32) as u64),
        HashFunctionKind::XxHash64 => xxhash(array, seed, xxh64),
        HashFunctionKind::XxHash3_64 => xxhash(array, seed, xxh3_64_with_seed),
        HashFunctionKind::MurmurHash3 => {
            let hasher = MurBuildHasher::new(seed.and_then(|s| s.get(0)).unwrap_or(42) as u32);
            let hashes = array
                .values_iter()
                .map(|v| {
                    let mut hasher = hasher.build_hasher();
                    hasher.write(v);
                    hasher.finish()
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::Sha1 => {
            let hashes = array
                .values_iter()
                .map(|v| {
                    let mut hasher = Sha1Hasher::default();
                    hasher.write(v);
                    hasher.finish()
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
    }
}

fn hash_fixed_size_binary(
    array: &FixedSizeBinaryArray,
    seed: Option<&PrimitiveArray<u64>>,
    hash_function: HashFunctionKind,
) -> PrimitiveArray<u64> {
    fn xxhash<F: Fn(&[u8], u64) -> u64>(
        array: &FixedSizeBinaryArray,
        seed: Option<&PrimitiveArray<u64>>,
        f: F,
    ) -> PrimitiveArray<u64> {
        let hashes = if let Some(seed) = seed {
            array
                .values_iter()
                .zip(seed.values_iter())
                .map(|(v, s)| f(v, *s))
                .collect::<Vec<_>>()
        } else {
            array.values_iter().map(|v| f(v, 0)).collect::<Vec<_>>()
        };
        PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
    }

    match hash_function {
        HashFunctionKind::XxHash32 => xxhash(array, seed, |v, s| xxh32(v, s as u32) as u64),
        HashFunctionKind::XxHash64 => xxhash(array, seed, xxh64),
        HashFunctionKind::XxHash3_64 => xxhash(array, seed, xxh3_64_with_seed),
        HashFunctionKind::MurmurHash3 => {
            let hasher = MurBuildHasher::new(seed.and_then(|s| s.get(0)).unwrap_or(42) as u32);
            let hashes = array
                .values_iter()
                .map(|v| {
                    let mut hasher = hasher.build_hasher();
                    hasher.write(v);
                    hasher.finish()
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::Sha1 => {
            let hashes = array
                .values_iter()
                .map(|v| {
                    let mut hasher = Sha1Hasher::default();
                    hasher.write(v);
                    hasher.finish()
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
    }
}

fn hash_utf8<O: Offset>(
    array: &Utf8Array<O>,
    seed: Option<&PrimitiveArray<u64>>,
    hash_function: HashFunctionKind,
) -> PrimitiveArray<u64> {
    fn xxhash<O: Offset, F: Fn(&[u8], u64) -> u64>(
        array: &Utf8Array<O>,
        seed: Option<&PrimitiveArray<u64>>,
        f: F,
    ) -> PrimitiveArray<u64> {
        let hashes = if let Some(seed) = seed {
            array
                .values_iter()
                .zip(seed.values_iter())
                .map(|(v, s)| f(v.as_bytes(), *s))
                .collect::<Vec<_>>()
        } else {
            array
                .values_iter()
                .map(|v| f(v.as_bytes(), 0))
                .collect::<Vec<_>>()
        };
        PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
    }

    match hash_function {
        HashFunctionKind::XxHash32 => xxhash(array, seed, |v, s| xxh32(v, s as u32) as u64),
        HashFunctionKind::XxHash64 => xxhash(array, seed, xxh64),
        HashFunctionKind::XxHash3_64 => xxhash(array, seed, xxh3_64_with_seed),
        HashFunctionKind::MurmurHash3 => {
            let hasher = MurBuildHasher::new(seed.and_then(|s| s.get(0)).unwrap_or(42) as u32);
            let hashes = array
                .values_iter()
                .map(|v| {
                    let mut hasher = hasher.build_hasher();
                    hasher.write(v.as_bytes());
                    hasher.finish()
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::Sha1 => {
            let hashes = array
                .values_iter()
                .map(|v| {
                    let mut hasher = Sha1Hasher::default();
                    hasher.write(v.as_bytes());
                    hasher.finish()
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
    }
}

fn hash_timestamp_with_timezone(
    array: &PrimitiveArray<i64>,
    timezone: &str,
    seed: Option<&PrimitiveArray<u64>>,
    hash_function: HashFunctionKind,
) -> PrimitiveArray<u64> {
    fn xxhash<const NULL_HASH: u64, F: Fn(&[u8], u64) -> u64>(
        array: &PrimitiveArray<i64>,
        timezone: &str,
        seed: Option<&PrimitiveArray<u64>>,
        f: F,
    ) -> PrimitiveArray<u64> {
        let hashes = if let Some(seed) = seed {
            array
                .iter()
                .zip(seed.values_iter())
                .map(|(v, s)| match v {
                    Some(v) => {
                        // Combine timestamp and timezone for hashing
                        let mut combined = Vec::new();
                        combined.extend_from_slice(&v.to_le_bytes());
                        combined.extend_from_slice(timezone.as_bytes());
                        f(&combined, *s)
                    }
                    None => NULL_HASH,
                })
                .collect::<Vec<_>>()
        } else {
            array
                .iter()
                .map(|v| match v {
                    Some(v) => {
                        // Combine timestamp and timezone for hashing
                        let mut combined = Vec::new();
                        combined.extend_from_slice(&v.to_le_bytes());
                        combined.extend_from_slice(timezone.as_bytes());
                        f(&combined, 0)
                    }
                    None => NULL_HASH,
                })
                .collect::<Vec<_>>()
        };
        PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
    }

    // For timestamps with timezone, we combine the timestamp value with the timezone string
    // to ensure that the same instant in different timezones produces different hashes
    match hash_function {
        HashFunctionKind::XxHash32 => {
            const NULL_HASH: u64 = const_xxh32::xxh32(b"", 0) as u64;
            xxhash::<NULL_HASH, _>(array, timezone, seed, |v, s| xxh32(v, s as u32) as u64)
        }
        HashFunctionKind::XxHash64 => {
            const NULL_HASH: u64 = const_xxh64::xxh64(b"", 0);
            xxhash::<NULL_HASH, _>(array, timezone, seed, xxh64)
        }
        HashFunctionKind::XxHash3_64 => {
            const NULL_HASH: u64 = const_xxh3::xxh3_64(b"");
            xxhash::<NULL_HASH, _>(array, timezone, seed, xxh3_64_with_seed)
        }
        HashFunctionKind::MurmurHash3 => {
            let hasher = MurBuildHasher::new(seed.and_then(|s| s.get(0)).unwrap_or(42) as u32);
            let hashes = array
                .iter()
                .map(|v| {
                    let mut hasher = hasher.build_hasher();
                    match v {
                        Some(v) => {
                            // Combine timestamp and timezone for hashing
                            hasher.write(&v.to_le_bytes());
                            hasher.write(timezone.as_bytes());
                            hasher.finish()
                        }
                        None => {
                            hasher.write(b"");
                            hasher.finish()
                        }
                    }
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::Sha1 => {
            let hashes = array
                .iter()
                .map(|v| {
                    let mut hasher = Sha1Hasher::default();
                    match v {
                        Some(v) => {
                            // Combine timestamp and timezone for hashing
                            hasher.write(&v.to_le_bytes());
                            hasher.write(timezone.as_bytes());
                            hasher.finish()
                        }
                        None => {
                            hasher.write(b"");
                            hasher.finish()
                        }
                    }
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
    }
}

fn hash_decimal(
    array: &PrimitiveArray<i128>,
    seed: Option<&PrimitiveArray<u64>>,
    hash_function: HashFunctionKind,
    scale: usize,
) -> PrimitiveArray<u64> {
    // For decimal hashing, we preserve the exact representation including scale
    // Different scales should produce different hashes (123, 123.0, 123.00 are different)
    // We convert to string representation that preserves the scale information
    fn format_decimal(value: i128, scale: usize) -> Vec<u8> {
        if value == 0 {
            // For zero, return "0.000..." with the appropriate number of decimal places
            let mut result = String::from("0");
            if scale > 0 {
                result.push('.');
                result.push_str(&"0".repeat(scale));
            }
            return result.into_bytes();
        }

        // Handle negative values
        let (is_negative, abs_value) = if value < 0 {
            (true, (-value) as u128)
        } else {
            (false, value as u128)
        };

        // Convert to string with proper decimal placement
        let value_str = abs_value.to_string();
        let mut result = String::new();

        if is_negative {
            result.push('-');
        }

        if scale == 0 {
            // No decimal places
            result.push_str(&value_str);
        } else if value_str.len() <= scale {
            // Value is smaller than scale, so it's 0.00...value
            result.push('0');
            result.push('.');
            // Add leading zeros
            result.push_str(&"0".repeat(scale - value_str.len()));
            result.push_str(&value_str);
        } else {
            // Value has both integer and fractional parts
            let integer_part = &value_str[..value_str.len() - scale];
            let fractional_part = &value_str[value_str.len() - scale..];
            result.push_str(integer_part);
            result.push('.');
            result.push_str(fractional_part);
        }

        result.into_bytes()
    }

    fn xxhash<const NULL_HASH: u64, F: Fn(&[u8], u64) -> u64>(
        array: &PrimitiveArray<i128>,
        seed: Option<&PrimitiveArray<u64>>,
        f: F,
        scale: usize,
    ) -> PrimitiveArray<u64> {
        let hashes = if let Some(seed) = seed {
            array
                .iter()
                .zip(seed.values_iter())
                .map(|(v, s)| match v {
                    Some(v) => {
                        let formatted = format_decimal(*v, scale);
                        f(&formatted, *s)
                    }
                    None => NULL_HASH,
                })
                .collect::<Vec<_>>()
        } else {
            array
                .iter()
                .map(|v| match v {
                    Some(v) => {
                        let formatted = format_decimal(*v, scale);
                        f(&formatted, 0)
                    }
                    None => NULL_HASH,
                })
                .collect::<Vec<_>>()
        };
        PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
    }

    match hash_function {
        HashFunctionKind::XxHash32 => {
            const NULL_HASH: u64 = const_xxh32::xxh32(b"", 0) as u64;
            xxhash::<NULL_HASH, _>(array, seed, |v, s| xxh32(v, s as u32) as u64, scale)
        }
        HashFunctionKind::XxHash64 => {
            const NULL_HASH: u64 = const_xxh64::xxh64(b"", 0);
            xxhash::<NULL_HASH, _>(array, seed, xxh64, scale)
        }
        HashFunctionKind::XxHash3_64 => {
            const NULL_HASH: u64 = const_xxh3::xxh3_64(b"");
            xxhash::<NULL_HASH, _>(array, seed, xxh3_64_with_seed, scale)
        }
        HashFunctionKind::MurmurHash3 => {
            let hasher = MurBuildHasher::new(seed.and_then(|s| s.get(0)).unwrap_or(42) as u32);
            let hashes = array
                .iter()
                .map(|v| {
                    let mut hasher = hasher.build_hasher();
                    match v {
                        Some(v) => {
                            let formatted = format_decimal(*v, scale);
                            hasher.write(&formatted);
                            hasher.finish()
                        }
                        None => {
                            hasher.write(b"");
                            hasher.finish()
                        }
                    }
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
        HashFunctionKind::Sha1 => {
            let hashes = array
                .iter()
                .map(|v| {
                    let mut hasher = Sha1Hasher::default();
                    match v {
                        Some(v) => {
                            let formatted = format_decimal(*v, scale);
                            hasher.write(&formatted);
                            hasher.finish()
                        }
                        None => {
                            hasher.write(b"");
                            hasher.finish()
                        }
                    }
                })
                .collect::<Vec<_>>();
            PrimitiveArray::<u64>::new(DataType::UInt64, hashes.into(), None)
        }
    }
}

macro_rules! with_match_hashing_primitive_type {(
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use daft_arrow::datatypes::PrimitiveType::*;
    use daft_arrow::types::{days_ms, months_days_ns};
    match $key_type {
        Int8 => __with_ty__! { i8 },
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        Int128 => __with_ty__! { i128 },
        DaysMs => __with_ty__! { days_ms },
        MonthDayNano => __with_ty__! { months_days_ns },
        UInt8 => __with_ty__! { u8 },
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
        _ => return Err(Error::NotYetImplemented(format!(
            "Hash not implemented for type {:?}",
            $key_type
        )))
    }
})}

pub fn hash(
    array: &dyn Array,
    seed: Option<&PrimitiveArray<u64>>,
    hash_function: HashFunctionKind,
) -> Result<PrimitiveArray<u64>> {
    if let Some(s) = seed {
        if s.len() != array.len() {
            return Err(Error::InvalidArgumentError(format!(
                "seed length does not match array length: {} vs {}",
                s.len(),
                array.len()
            )));
        }

        if *s.data_type() != DataType::UInt64 {
            return Err(Error::InvalidArgumentError(format!(
                "seed data type expected to be uint64, got {:?}",
                *s.data_type()
            )));
        }
    }

    // Check for Time and Timestamp types first (logical type check)
    match array.data_type() {
        DataType::Time32(_) => {
            let time_array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "Expected Time32 array to be PrimitiveArray<i32>".to_string(),
                    )
                })?;
            return Ok(hash_primitive::<i32>(time_array, seed, hash_function));
        }
        DataType::Time64(_) => {
            let time_array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "Expected Time64 array to be PrimitiveArray<i64>".to_string(),
                    )
                })?;
            return Ok(hash_primitive::<i64>(time_array, seed, hash_function));
        }
        DataType::Timestamp(_, timezone) => {
            // Timestamps are stored as i64 values (microseconds since epoch)
            let timestamp_array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "Expected Timestamp array to be PrimitiveArray<i64>".to_string(),
                    )
                })?;

            // For timestamps with timezone, we need to include the timezone in the hash
            // to ensure that the same instant in different timezones produces different hashes
            if let Some(tz) = timezone {
                return Ok(hash_timestamp_with_timezone(
                    timestamp_array,
                    tz,
                    seed,
                    hash_function,
                ));
            } else {
                // For timestamps without timezone, just hash the timestamp value
                return Ok(hash_primitive::<i64>(timestamp_array, seed, hash_function));
            }
        }
        _ => {}
    }

    Ok(match array.data_type().to_physical_type() {
        PhysicalType::Null => {
            hash_null(array.as_any().downcast_ref().unwrap(), seed, hash_function)
        }
        PhysicalType::Boolean => {
            hash_boolean(array.as_any().downcast_ref().unwrap(), seed, hash_function)
        }
        PhysicalType::Primitive(primitive) => {
            // Check if this is a decimal type
            if let DataType::Decimal(_precision, scale) = array.data_type() {
                // Use special decimal hashing that considers precision and scale
                let decimal_array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i128>>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "Expected decimal array to be PrimitiveArray<i128>".to_string(),
                        )
                    })?;
                hash_decimal(decimal_array, seed, hash_function, *scale)
            } else {
                // Use regular primitive hashing
                with_match_hashing_primitive_type!(primitive, |$T| {
                    hash_primitive::<$T>(array.as_any().downcast_ref().unwrap(), seed, hash_function)
                })
            }
        }
        PhysicalType::Binary => {
            hash_binary::<i32>(array.as_any().downcast_ref().unwrap(), seed, hash_function)
        }
        PhysicalType::LargeBinary => {
            hash_binary::<i64>(array.as_any().downcast_ref().unwrap(), seed, hash_function)
        }
        PhysicalType::FixedSizeBinary => {
            hash_fixed_size_binary(array.as_any().downcast_ref().unwrap(), seed, hash_function)
        }
        PhysicalType::Utf8 => {
            hash_utf8::<i32>(array.as_any().downcast_ref().unwrap(), seed, hash_function)
        }
        PhysicalType::LargeUtf8 => {
            hash_utf8::<i64>(array.as_any().downcast_ref().unwrap(), seed, hash_function)
        }
        t => {
            return Err(Error::NotYetImplemented(format!(
                "Hash not implemented for type {t:?}"
            )));
        }
    })
}

#[cfg(test)]
mod port_tests {
    use super::*;

    /// Deterministic 64-bit LCG (Numerical Recipes constants); no `rand` dependency.
    fn lcg_values(len: usize) -> Vec<i64> {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                state as i64
            })
            .collect()
    }

    #[test]
    fn xxh3_64_dense_matches_reference() {
        let values = lcg_values(1000);
        let array = PrimitiveArray::<i64>::from_vec(values.clone());
        let hashed = hash_primitive(&array, None, HashFunctionKind::XxHash3_64);

        assert_eq!(hashed.len(), values.len());
        assert_eq!(hashed.null_count(), 0);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(
                hashed.value(i),
                const_xxh3::xxh3_64(&v.to_le_bytes()),
                "hash mismatch at index {i}"
            );
        }
    }

    #[test]
    fn xxh3_64_with_nulls_matches_reference() {
        const NULL_HASH: u64 = const_xxh3::xxh3_64(b"");

        let values = lcg_values(1000);
        // Null out every 97th element (11 nulls), forcing the null-aware path.
        let optional: Vec<Option<i64>> = values
            .iter()
            .enumerate()
            .map(|(i, v)| if i % 97 == 0 { None } else { Some(*v) })
            .collect();
        let array = PrimitiveArray::<i64>::from(optional);

        assert!(array.null_count() > 0);
        let hashed = hash_primitive(&array, None, HashFunctionKind::XxHash3_64);
        assert_eq!(hashed.len(), values.len());

        for (i, v) in values.iter().enumerate() {
            let expected = if i % 97 == 0 {
                NULL_HASH
            } else {
                const_xxh3::xxh3_64(&v.to_le_bytes())
            };
            assert_eq!(hashed.value(i), expected, "hash mismatch at index {i}");
        }
    }
}
