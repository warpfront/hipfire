// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Grouped MoE down producer for the gfx1201 A8 route: the grouped gate/up
// activation (`ffn_grouped_activation_pad.hip`: cast_in, SwiGLU/SiTU, cast_out,
// zero pad to phys_n, per-expert AWQ divide, un-scatter), the plain FWHT rotation
// (`mq_rotate_x_i8_gfx12`) and the `block_i8_128` quantizer in ONE launch.
//
// It replaces the three-kernel chain
//     ffn_*_grouped_f32 -> mq_rotate_x_batched -> quantize_int8_mmq_ds128
// whose F32 intermediates (the activation rows `act`, the rotated rows `rot`) cost
// four full passes over `[slots, phys_n]` F32 plus two launches. Here one wave owns one
// (grouped row, 256-wide rotation group): it forms the 256 activation values in
// registers (lane `l` owns columns 8l..8l+7, the register layout of `mq_rotate_x`),
// rotates them with the shared `mq_fwht256_lane`, and emits the group's two
// `block_i8_128` blocks with the shared `emit_i8_sidecar_from_producer8`, at block
// index (2 * group + half) * total_slots + flat_slot (flat_slot = sorted_slot_index of
// the grouped row: the un-scatter).
//
// Bit identity with the chain (byte oracle: tests/ffn_grouped_i8_producer_oracle.rs):
//   * the activation element expression is `ffn_grouped_activation_pad.hip`'s own
//     (`ffn_act<ACT>`, `ffn_bf16_rne`: the file is concatenated ahead of this one), the
//     AWQ divide is `r / scale[col]` after the output cast, a padded column is +0.0f
//     before the divide, an expert outside [0, n_experts) or a null table entry makes
//     the whole row +0.0f WITHOUT a divide;
//   * `mq_rotate_x` forms `x * signs1`, the FWHT, then `v * 0.0625f * signs2`; so does
//     this kernel (an F32 store/reload between the stages is lossless);
//   * the quantizer is the shared register emit, byte-identical to the standalone.
// `x_out` (nullable) additionally stores the rotated F32 row `[flat_slot, phys_n]`: the
// bytes `mq_rotate_x_batched` leaves in its output.
//
// Rows with sorted_slot_index[g] < 0 or >= total_slots (the padding rows of the grouped
// order) write nothing, exactly like the standalone activation. The host admits only
// logical_n % 8 == 0 and phys_n % 256 == 0 with logical_n <= phys_n, so a lane's eight
// columns are all live or all padding and its gate/up reads are 16-byte aligned.
//
// Entries: ffn_swiglu_grouped_rotate_i8_gfx1201, ffn_swiglu_clamp_grouped_rotate_i8_gfx1201
//          (p0 = limit), ffn_situ_grouped_rotate_i8_gfx1201 (p0 = gate_beta, p1 = value_beta).
//
// Grid  (m_total * (phys_n / 256), 1, 1), block 32 (one wave).
//
// Kernarg order:
//   y_grouped*, sorted_slot_index*, expert_tile_ids*, awq_table* (nullable),
//   x_out* (nullable), signs1*, signs2*, x_i8*, m_total, block_m, logical_n,
//   phys_n, n_experts, total_slots, round_inputs, round_output, p0, p1
//
// Concatenation order (kernels.rs): block_i8_128_quant.hip (NO_STANDALONE),
// block_i8_128_emit.hip, mq_fwht256.h, ffn_grouped_activation_pad.hip, this file.

template <int ACT>
static __device__ __forceinline__ void ffn_grouped_rotate_i8_body(
    const float* __restrict__ y_grouped, const int* __restrict__ sorted_slot_index,
    const int* __restrict__ expert_tile_ids, const unsigned long long* __restrict__ awq_table,
    float* __restrict__ x_out, const float* __restrict__ signs1, const float* __restrict__ signs2,
    block_i8_128* __restrict__ x_i8,
    int m_total, int block_m, int logical_n, int phys_n, int n_experts, int total_slots,
    int round_inputs, int round_output, float p0, float p1
) {
    const int lane = threadIdx.x;
    const int groups_total = phys_n / 256;
    const int g_row = blockIdx.x / groups_total;
    const int group = blockIdx.x - g_row * groups_total;
    if (g_row >= m_total) return;
    const int flat = sorted_slot_index[g_row];
    if (flat < 0 || flat >= total_slots) return;

    const int d0 = lane * 8;
    const int base = group * 256 + d0;

    const float* scale = nullptr;
    bool zero_row = false;
    if (awq_table != nullptr) {
        const int e = expert_tile_ids[g_row / block_m];
        if (e >= 0 && e < n_experts)
            scale = reinterpret_cast<const float*>(awq_table[e]);
        zero_row = (scale == nullptr);
    }

    float r[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = 0.0f;
    if (!zero_row) {
        if (base < logical_n) {
            const float* yg = y_grouped + (long long)g_row * 2 * logical_n + base;
            const float4 ga = *reinterpret_cast<const float4*>(yg);
            const float4 gb = *reinterpret_cast<const float4*>(yg + 4);
            const float4 ua = *reinterpret_cast<const float4*>(yg + logical_n);
            const float4 ub = *reinterpret_cast<const float4*>(yg + logical_n + 4);
            const float gv[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w};
            const float uv[8] = {ua.x, ua.y, ua.z, ua.w, ub.x, ub.y, ub.z, ub.w};
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float g = gv[e];
                float u = uv[e];
                if (round_inputs) {
                    g = ffn_bf16_rne(g);
                    u = ffn_bf16_rne(u);
                }
                float v = ffn_act<ACT>(g, u, p0, p1);
                if (round_output) v = ffn_bf16_rne(v);
                r[e] = v;
            }
        }
        if (scale != nullptr) {
#pragma unroll
            for (int e = 0; e < 8; ++e) r[e] = r[e] / scale[base + e];
        }
    }

    // `mq_rotate_x`: (x * signs1) -> FWHT-256 -> v * 0.0625f * signs2.
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = r[e] * signs1[d0 + e];
    mq_fwht256_lane(v, lane);
    float f[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = v[e] * 0.0625f * signs2[d0 + e];

    if (x_out != nullptr) {
        float* o = x_out + (long long)flat * phys_n + base;
        *reinterpret_cast<float4*>(o) = make_float4(f[0], f[1], f[2], f[3]);
        *reinterpret_cast<float4*>(o + 4) = make_float4(f[4], f[5], f[6], f[7]);
    }
    emit_i8_sidecar_from_producer8(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
                                   x_i8, group, flat, total_slots, lane);
}

#define FFN_GROUPED_ROTATE_I8_ENTRY(NAME, ACT)                                           \
extern "C" __global__ __launch_bounds__(32, 16) void NAME(                               \
    const float* __restrict__ y_grouped, const int* __restrict__ sorted_slot_index,      \
    const int* __restrict__ expert_tile_ids,                                             \
    const unsigned long long* __restrict__ awq_table, float* __restrict__ x_out,         \
    const float* __restrict__ signs1, const float* __restrict__ signs2,                  \
    block_i8_128* __restrict__ x_i8,                                                     \
    int m_total, int block_m, int logical_n, int phys_n, int n_experts, int total_slots, \
    int round_inputs, int round_output, float p0, float p1                               \
) {                                                                                      \
    ffn_grouped_rotate_i8_body<ACT>(y_grouped, sorted_slot_index, expert_tile_ids,       \
        awq_table, x_out, signs1, signs2, x_i8, m_total, block_m, logical_n, phys_n,     \
        n_experts, total_slots, round_inputs, round_output, p0, p1);                     \
}

FFN_GROUPED_ROTATE_I8_ENTRY(ffn_swiglu_grouped_rotate_i8_gfx1201, 0)
FFN_GROUPED_ROTATE_I8_ENTRY(ffn_swiglu_clamp_grouped_rotate_i8_gfx1201, 1)
FFN_GROUPED_ROTATE_I8_ENTRY(ffn_situ_grouped_rotate_i8_gfx1201, 2)

// ───────────────────────── dense rows (the shared expert / dense FFN down) ─────────────────────────
//
// The same fusion over `ffn_situ.hip`'s dense-rows activation (`ffn_*_rows_f32` with a vector or
// no AWQ divide): row r = blockIdx.x / (phys_n / 256) of `gate` / `up` (element strides gate_ld /
// up_ld) -> activation, cast, pad, optional `r / awq[col]` -> rotate -> blocks at block index
// (2 * group + half) * rows + r. `ffn_bf16_rne` / `ffn_act` are the ones `ffn_grouped_activation_pad.hip`
// defines above, textually identical to `ffn_situ.hip`'s (the byte oracle covers both sites).
// The host admits logical_n % 8 == 0, phys_n % 256 == 0, gate/up 16-byte aligned with ld % 4 == 0.
//
// Entries: ffn_{swiglu,swiglu_clamp,situ}_rows_rotate_i8_gfx1201
// Grid (rows * (phys_n / 256), 1, 1), block 32.
// Kernarg order:
//   gate*, up*, x_out* (nullable), awq* (nullable F32[phys_n]), signs1*, signs2*, x_i8*,
//   rows, logical_n, phys_n, gate_ld, up_ld, round_inputs, round_output, p0, p1
template <int ACT>
static __device__ __forceinline__ void ffn_rows_rotate_i8_body(
    const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ x_out,
    const float* __restrict__ awq, const float* __restrict__ signs1,
    const float* __restrict__ signs2, block_i8_128* __restrict__ x_i8,
    int rows, int logical_n, int phys_n, int gate_ld, int up_ld,
    int round_inputs, int round_output, float p0, float p1
) {
    const int lane = threadIdx.x;
    const int groups_total = phys_n / 256;
    const int row = blockIdx.x / groups_total;
    const int group = blockIdx.x - row * groups_total;
    if (row >= rows) return;

    const int d0 = lane * 8;
    const int base = group * 256 + d0;

    float r[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = 0.0f;
    if (base < logical_n) {
        const float* gp = gate + (long long)row * gate_ld + base;
        const float* upp = up + (long long)row * up_ld + base;
        const float4 ga = *reinterpret_cast<const float4*>(gp);
        const float4 gb = *reinterpret_cast<const float4*>(gp + 4);
        const float4 ua = *reinterpret_cast<const float4*>(upp);
        const float4 ub = *reinterpret_cast<const float4*>(upp + 4);
        const float gv[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w};
        const float uv[8] = {ua.x, ua.y, ua.z, ua.w, ub.x, ub.y, ub.z, ub.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float g = gv[e];
            float u = uv[e];
            if (round_inputs) {
                g = ffn_bf16_rne(g);
                u = ffn_bf16_rne(u);
            }
            float v = ffn_act<ACT>(g, u, p0, p1);
            if (round_output) v = ffn_bf16_rne(v);
            r[e] = v;
        }
    }
    if (awq != nullptr) {
#pragma unroll
        for (int e = 0; e < 8; ++e) r[e] = r[e] / awq[base + e];
    }

    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = r[e] * signs1[d0 + e];
    mq_fwht256_lane(v, lane);
    float f[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = v[e] * 0.0625f * signs2[d0 + e];

    if (x_out != nullptr) {
        float* o = x_out + (long long)row * phys_n + base;
        *reinterpret_cast<float4*>(o) = make_float4(f[0], f[1], f[2], f[3]);
        *reinterpret_cast<float4*>(o + 4) = make_float4(f[4], f[5], f[6], f[7]);
    }
    emit_i8_sidecar_from_producer8(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
                                   x_i8, group, row, rows, lane);
}

#define FFN_ROWS_ROTATE_I8_ENTRY(NAME, ACT)                                              \
extern "C" __global__ __launch_bounds__(32, 16) void NAME(                               \
    const float* __restrict__ gate, const float* __restrict__ up,                        \
    float* __restrict__ x_out, const float* __restrict__ awq,                            \
    const float* __restrict__ signs1, const float* __restrict__ signs2,                  \
    block_i8_128* __restrict__ x_i8,                                                     \
    int rows, int logical_n, int phys_n, int gate_ld, int up_ld,                         \
    int round_inputs, int round_output, float p0, float p1                               \
) {                                                                                      \
    ffn_rows_rotate_i8_body<ACT>(gate, up, x_out, awq, signs1, signs2, x_i8, rows,       \
        logical_n, phys_n, gate_ld, up_ld, round_inputs, round_output, p0, p1);          \
}

FFN_ROWS_ROTATE_I8_ENTRY(ffn_swiglu_rows_rotate_i8_gfx1201, 0)
FFN_ROWS_ROTATE_I8_ENTRY(ffn_swiglu_clamp_rows_rotate_i8_gfx1201, 1)
FFN_ROWS_ROTATE_I8_ENTRY(ffn_situ_rows_rotate_i8_gfx1201, 2)
