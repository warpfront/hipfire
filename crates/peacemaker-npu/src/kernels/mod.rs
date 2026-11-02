// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Hand-encoded AIE2P compute kernels and their DMA/data-layout plans.
pub mod bw_probe;
pub mod experts;
pub mod gemm_array;
pub mod gemm_core;
pub mod gemm_core_g80;
pub mod gemm_g80;
pub mod gemm_g80_n512;
pub mod gemm_i8;
pub mod iu4;
pub mod iu4_fold_model;
pub mod iu4_ief15;
pub mod iu4_ief15_core;
pub mod iu4_ief15_probe;
pub mod ring;
pub mod w8a8_epi;
