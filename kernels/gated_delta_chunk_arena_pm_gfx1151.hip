// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// PM twin source (gated_delta_chunk_arena_pm_gfx1151.hip): the shipped product kernel, byte for byte, compiled under a per-arch symbol.
// The only difference from the incumbent is the kernel symbol (a macro rename before the include),
// so the twin's .text must equal the incumbent hipcc .text; scripts/pm-hip-twin.sh enforces it.
#define NPU_TRUNK_ARENA 1
#define gated_delta_chunk_gate_wmma gated_delta_chunk_gate_wmma_arena_pm_gfx1151
#include "src/gated_delta_chunk_wmma.gfx1151.hip"
