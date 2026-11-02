// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Row gather / scatter for the lane-batched ctx fill and fc of the DFlash
// continuous-batching draft. Pure copies of f32 BITS (u32 load/store): no
// arithmetic, so signed zeros / NaN payloads travel untouched.
//
// One launch moves up to DRAFT_ROW_SPANS_MAX row spans. Span `s` (blockIdx.z)
// covers `rows` rows of `width` f32 each; its rows sit at packed rows
// [packed_row0, packed_row0 + rows) of the packed tensor(s). Span pointers are
// ALREADY advanced to the segment start; a segment's rows are contiguous
// (`width` f32 per row) on the span side as well.
//
//   gather : packed[(packed_row0 + r) * width + c] = src[r * width + c]
//            and, when `positions_out != nullptr` (one thread per row, c == 0):
//            positions_out[packed_row0 + r] = ((const int*)positions)[r]
//   scatter: dst0[r * width + c] = packed0[(packed_row0 + r) * width + c]
//            and, when `packed1 != nullptr`:
//            dst1[r * width + c] = packed1[(packed_row0 + r) * width + c]
//
// Fields a given operation does not use are ignored. Grid
// `[ceil(max_span_rows * width / 256), 1, n_spans]`, block 256. Excess blocks
// and threads return before any memory access.
//
// ABI (kept in lockstep with `DraftRowSpan` in rdna-compute/src/draft_lane_rows.rs):
// 40 bytes, 8-aligned; the table is 32 spans (1280 B), unused slots zero.

#include <hip/hip_runtime.h>

#define DRAFT_ROW_SPANS_MAX 32

struct DraftRowSpan {
    unsigned long long src;
    unsigned long long dst0;
    unsigned long long dst1;
    unsigned long long positions;
    int packed_row0;
    int rows;
};
static_assert(sizeof(DraftRowSpan) == 40, "DraftRowSpan must be 40 bytes");
static_assert(alignof(DraftRowSpan) == 8, "DraftRowSpan must be 8-aligned");

struct DraftRowSpanTable {
    DraftRowSpan span[DRAFT_ROW_SPANS_MAX];
};
static_assert(sizeof(DraftRowSpanTable) == 1280, "DraftRowSpanTable must be 1280 bytes");

extern "C" __global__ void draft_rows_gather_f32(
    unsigned int* __restrict__ packed,
    int* __restrict__ positions_out,
    const DraftRowSpanTable table,
    int width
) {
    const DraftRowSpan sp = table.span[blockIdx.z];
    const int rows = sp.rows;
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (rows <= 0 || idx >= rows * width) return;
    const int r = idx / width;
    const int c = idx - r * width;
    const unsigned int* src = reinterpret_cast<const unsigned int*>(sp.src);
    const long long prow = (long long)sp.packed_row0 + r;
    packed[prow * width + c] = src[(long long)r * width + c];
    if (positions_out != nullptr && c == 0) {
        positions_out[prow] = reinterpret_cast<const int*>(sp.positions)[r];
    }
}

extern "C" __global__ void draft_rows_scatter_f32(
    const unsigned int* __restrict__ packed0,
    const unsigned int* __restrict__ packed1,
    const DraftRowSpanTable table,
    int width
) {
    const DraftRowSpan sp = table.span[blockIdx.z];
    const int rows = sp.rows;
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (rows <= 0 || idx >= rows * width) return;
    const int r = idx / width;
    const int c = idx - r * width;
    const long long prow = (long long)sp.packed_row0 + r;
    const long long poff = prow * width + c;
    const long long doff = (long long)r * width + c;
    reinterpret_cast<unsigned int*>(sp.dst0)[doff] = packed0[poff];
    if (packed1 != nullptr) {
        reinterpret_cast<unsigned int*>(sp.dst1)[doff] = packed1[poff];
    }
}
