// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// Prefix for the in-register producer: first argument is the gate/up GEMM's
// already fused h row. The second (up) and x_rot arguments are unused/null.
// The shared SwiGLU/FWHT body preserves the same AWQ, signs and butterfly.
#define HIPFIRE_FP8_PROD_INREG 1
#define HIPFIRE_SILU_FP8_HIN 1
