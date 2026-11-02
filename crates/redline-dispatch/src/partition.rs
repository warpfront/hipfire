// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Pure CU partition policy types for HIP green-context binding (spec L2).
//!
//! Validation is fail-closed: non-dividing or mismatched policies are
//! configuration errors with no silent fallback to a single full-device
//! partition.

use std::num::NonZeroUsize;

/// How to carve device compute units into worker partitions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PartitionPolicy {
    /// One partition covering the entire device CU budget.
    None,
    /// Split into `n` equal contiguous partitions. Requires
    /// `device_cu_count % n == 0`.
    Equal(NonZeroUsize),
    /// Explicit per-partition CU counts in order; must sum to
    /// `device_cu_count` and contain no zeros.
    Explicit(Vec<u32>),
}

/// One contiguous CU slice after policy validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CuPartition {
    pub index: u32,
    pub cu_offset: u32,
    pub cu_count: u32,
}

/// Failures from [`validate`].
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PartitionError {
    #[error("equal partition count {parts} does not divide device CU count {device_cu_count}")]
    NonDividing { device_cu_count: u32, parts: usize },
    #[error("explicit partition policy is empty")]
    EmptyExplicit,
    #[error("explicit partition at index {index} has zero CU count")]
    ZeroCount { index: usize },
    #[error("explicit partition CU sum {sum} does not match device CU count {device_cu_count}")]
    SumMismatch { sum: u32, device_cu_count: u32 },
}

/// Validate `policy` against `device_cu_count` and materialize contiguous
/// [`CuPartition`] slices.
///
/// - [`PartitionPolicy::None`] → a single full-cover partition.
/// - [`PartitionPolicy::Equal`] → `n` equal parts when `device_cu_count` is
///   divisible by `n`; otherwise [`PartitionError::NonDividing`].
/// - [`PartitionPolicy::Explicit`] → ordered slices when the vector is
///   non-empty, has no zeros, and sums exactly to `device_cu_count`.
pub fn validate(
    policy: &PartitionPolicy,
    device_cu_count: u32,
) -> Result<Vec<CuPartition>, PartitionError> {
    match policy {
        PartitionPolicy::None => Ok(vec![CuPartition {
            index: 0,
            cu_offset: 0,
            cu_count: device_cu_count,
        }]),
        PartitionPolicy::Equal(parts) => {
            let n = parts.get();
            let n_u32 = u32::try_from(n).map_err(|_| PartitionError::NonDividing {
                device_cu_count,
                parts: n,
            })?;
            if n_u32 == 0 || !device_cu_count.is_multiple_of(n_u32) {
                return Err(PartitionError::NonDividing {
                    device_cu_count,
                    parts: n,
                });
            }
            let cu_each = device_cu_count / n_u32;
            Ok((0..n_u32)
                .map(|index| CuPartition {
                    index,
                    cu_offset: index * cu_each,
                    cu_count: cu_each,
                })
                .collect())
        }
        PartitionPolicy::Explicit(counts) => {
            if counts.is_empty() {
                return Err(PartitionError::EmptyExplicit);
            }
            let mut partitions = Vec::with_capacity(counts.len());
            let mut offset: u32 = 0;
            let mut sum: u32 = 0;
            for (i, &count) in counts.iter().enumerate() {
                if count == 0 {
                    return Err(PartitionError::ZeroCount { index: i });
                }
                let index = u32::try_from(i).expect("partition index fits u32");
                partitions.push(CuPartition {
                    index,
                    cu_offset: offset,
                    cu_count: count,
                });
                offset = offset.saturating_add(count);
                sum = sum.saturating_add(count);
            }
            if sum != device_cu_count {
                return Err(PartitionError::SumMismatch {
                    sum,
                    device_cu_count,
                });
            }
            Ok(partitions)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_full_cover() {
        let parts = validate(&PartitionPolicy::None, 64).unwrap();
        assert_eq!(
            parts,
            vec![CuPartition {
                index: 0,
                cu_offset: 0,
                cu_count: 64,
            }]
        );
    }

    #[test]
    fn equal_divide_ok() {
        let n = NonZeroUsize::new(4).unwrap();
        let parts = validate(&PartitionPolicy::Equal(n), 64).unwrap();
        assert_eq!(
            parts,
            vec![
                CuPartition {
                    index: 0,
                    cu_offset: 0,
                    cu_count: 16,
                },
                CuPartition {
                    index: 1,
                    cu_offset: 16,
                    cu_count: 16,
                },
                CuPartition {
                    index: 2,
                    cu_offset: 32,
                    cu_count: 16,
                },
                CuPartition {
                    index: 3,
                    cu_offset: 48,
                    cu_count: 16,
                },
            ]
        );
    }

    #[test]
    fn equal_non_divide_err() {
        let n = NonZeroUsize::new(3).unwrap();
        let err = validate(&PartitionPolicy::Equal(n), 64).unwrap_err();
        assert_eq!(
            err,
            PartitionError::NonDividing {
                device_cu_count: 64,
                parts: 3,
            }
        );
    }

    #[test]
    fn explicit_sum_mismatch_err() {
        let err = validate(&PartitionPolicy::Explicit(vec![16, 16, 16]), 64).unwrap_err();
        assert_eq!(
            err,
            PartitionError::SumMismatch {
                sum: 48,
                device_cu_count: 64,
            }
        );
    }

    #[test]
    fn explicit_zero_count_err() {
        let err = validate(&PartitionPolicy::Explicit(vec![32, 0, 32]), 64).unwrap_err();
        assert_eq!(err, PartitionError::ZeroCount { index: 1 });
    }

    #[test]
    fn explicit_ok_contiguous() {
        let parts = validate(&PartitionPolicy::Explicit(vec![8, 24, 32]), 64).unwrap();
        assert_eq!(
            parts,
            vec![
                CuPartition {
                    index: 0,
                    cu_offset: 0,
                    cu_count: 8,
                },
                CuPartition {
                    index: 1,
                    cu_offset: 8,
                    cu_count: 24,
                },
                CuPartition {
                    index: 2,
                    cu_offset: 32,
                    cu_count: 32,
                },
            ]
        );
    }

    #[test]
    fn explicit_empty_err() {
        let err = validate(&PartitionPolicy::Explicit(vec![]), 64).unwrap_err();
        assert_eq!(err, PartitionError::EmptyExplicit);
    }
}
