// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// FN dense GDN scan adapter kernels (gfx1201, Qwen3.5 dense geometry:
// 16 q/k heads + 48 v heads, 128 dims, conv channels 10240 = q 2048 | k 2048 |
// v 6144, K=4 convolution with a 3-row history ring, 48 gate heads).  Compiled
// with default flags (no -ffp-contract=off), exactly like tensor_ops, whose
// expressions are copied here; kernels.rs prepends block_i8_128_quant.hip (no
// standalone quantizer), block_i8_128_emit.hip and mq_fwht256.h.
//
// fn_gdn_dense_producer = gated_delta_conv_bf16_f32_batched_k4 (+ its params
//   blocks) + the dense adapter's BF16 Q/K l2 norm (serial 128-channel sums,
//   pipe128 row-parameter expression sequence) + F16 packing + chunk-local
//   inclusive cumsum of the log decay, in one launch. The packed F16 q/k/v
//   rows [rows, n64) are written zero, and the scan's EF plane is zeroed. No
//   DPP/swizzle is used by the norm.
// fn_gdn_dense_gate_bf16in / fn_gdn_dense_gate_rotate_bf16in = gated_delta_gate_bf16_f32_batched[/_rotate]
//   reading the chunk scan's packed BF16 output plane directly (the widening
//   hc_state_bf16_to_f32 and the in-place bf16 rounding of the recurrent
//   plane are exact no-ops on those bits).
// fn_gdn_dense_gate_i8 = fn_gdn_dense_gate_bf16in followed by mq_rotate_x_i8_gfx12 on its
//   output in one launch (the out projection's `block_i8_128` blocks; the F32 gate
//   output never reaches memory).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float bf(float x) {
    return static_cast<float>(static_cast<hip_bfloat16>(x));
}

static __device__ __forceinline__ unsigned short f16bits(float x) {
    return __half_as_ushort(__float2half_rn(x));
}

#define FNGD_CHANNELS 10240
#define FNGD_HEADS 48

// gated_delta_conv_bf16_f32_batched_k4's body for one channel and the tile
// rows [r0, min(r0 + 16, rows)) (r0 < rows); `emit(r, bf16_out)` receives
// each output row.  Only the r0 == 0 tile reads the history and then stores
// next_history (it may alias history), as in the reference kernel.
template <typename Emit>
static __device__ __forceinline__ void fn_gdn_dense_conv_rows(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* next_history,
    int rows,
    int start_cursor,
    int input_bf16,
    int channel,
    int r0,
    Emit emit) {
    const int channels = FNGD_CHANNELS;
    auto in = [&](int j) {
        if (input_bf16) {
            __hip_bfloat16 b;
            b.data = ((const unsigned short*)input)[j * channels + channel];
            return b;
        }
        return bf16_round(input[j * channels + channel]);
    };
    auto x = [&](int j) {
        return j >= 0 ? in(j)
                      : bf16_round(history[((start_cursor + j + 3) % 3) * channels + channel]);
    };
    const float k0 = bf16_to_f32(kernel[channel * 4 + 0]);
    const float k1 = bf16_to_f32(kernel[channel * 4 + 1]);
    const float k2 = bf16_to_f32(kernel[channel * 4 + 2]);
    const float k3 = bf16_to_f32(kernel[channel * 4 + 3]);
    __hip_bfloat16 h0 = x(r0 - 3), h1 = x(r0 - 2), h2 = x(r0 - 1);
    __hip_bfloat16 tail[3];
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s) {
            const int j = rows - 3 + ((s - (start_cursor + rows) % 3) + 6) % 3;
            tail[s] = x(j);
        }
    }
    const int r1 = min(r0 + 16, rows);
    for (int r = r0; r < r1; ++r) {
        const __hip_bfloat16 current_bf16 = in(r);
        float value = k3 * bf16_to_f32(current_bf16);
        value += k0 * bf16_to_f32(h0);
        value += k1 * bf16_to_f32(h1);
        value += k2 * bf16_to_f32(h2);
        const __hip_bfloat16 out = bf16_round(value * (1.0f / (1.0f + expf(-value))));
        emit(r, out);
        h0 = h1;
        h1 = h2;
        h2 = current_bf16;
    }
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s)
            next_history[s * channels + channel] = bf16_to_f32(tail[s]);
    }
}

// gated_delta_params_flat's body; the log decay is returned (the cumsum
// consumes it from LDS, i.e. after the same float store the reference makes).
static __device__ __forceinline__ float fn_gdn_dense_params_flat(
    const float* a,
    const float* b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* beta,
    int flat) {
    const int head = flat % FNGD_HEADS;
    const float al = bf16_to_f32(a_log[head]);
    const float dt = bf16_to_f32(dt_bias[head]);
    const float value = bf16_to_f32(bf16_round(a[flat])) + dt;
    const float softplus = value > 20.0f
        ? value
        : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    const float gate = -expf(al) * softplus;
    beta[flat] = 1.0f / (1.0f + expf(-bf16_to_f32(bf16_round(b[flat]))));
    return gate;
}

// Grid [82, n64 / 32, 1], block 256, no dynamic LDS (32-row tiles).
//   x < 32  : q|k head x (q heads 0-15, k heads 16-31), 128 channels; thread =
//             (row half, channel), 16 rows each; serial norm on 32 threads.
//   x < 80  : v head x - 32, 128 channels, same thread mapping.
//   x == 80 : params + beta + chunk64 cumsum into g for chunk y / 2 (even y).
//   x == 81 : zero `ef` (ef_words u32 words, a multiple of 4), grid-strided
//             over y. ef_words == 0 (live-EF Q8Ef state: `ef` is the owner's
//             persistent residual, which the scan reads on entry) stores
//             nothing; plain Q8 passes the thread-local scratch and its full
//             word count, cleared on every invocation.
// input [rows][10240] F32 (input_bf16 == 0) or BF16 bits; q16/k16 [n64][2048]
// and v16 [n64][6144] F16 bits (q scaled by query_scale; pad rows [rows, n64)
// zeroed); g and beta [rows][48] (pad rows are not written).
extern "C" __global__ __launch_bounds__(256) void fn_gdn_dense_producer(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* next_history,
    unsigned short* q16,
    unsigned short* k16,
    unsigned short* v16,
    float* g,
    float* beta,
    const float* param_a,
    const float* param_b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    unsigned int* ef,
    int rows,
    int n64,
    int start_cursor,
    int input_bf16,
    float query_scale,
    int ef_words) {
    // conv staging [32 rows][129] (q|k) or the chunk's log decay [64][48].
    __shared__ float smem[32 * 129];
    __shared__ float inv[32];
    const int t = (int)threadIdx.x;
    const int x = (int)blockIdx.x;
    const int row0 = (int)blockIdx.y * 32;
    if (x >= 80) {
        if (x == 81) {
            const int stride = (int)gridDim.y * 256;
            uint4* p = reinterpret_cast<uint4*>(ef);
            for (int i = (int)blockIdx.y * 256 + t; i < ef_words / 4; i += stride)
                p[i] = make_uint4(0u, 0u, 0u, 0u);
            return;
        }
        if (blockIdx.y & 1) return;
        const int chunk = (int)blockIdx.y >> 1;
        const int total = rows * FNGD_HEADS;
        for (int i = t; i < 64 * FNGD_HEADS; i += 256) {
            const int flat = chunk * 64 * FNGD_HEADS + i;
            smem[i] = flat < total
                ? fn_gdn_dense_params_flat(param_a, param_b, a_log, dt_bias, beta, flat)
                : 0.0f;
        }
        __syncthreads();
        if (t < FNGD_HEADS) {
            // The dense adapter's chunk-local inclusive cumsum, serial.
            float s = 0.0f;
            for (int r = 0; r < 64; ++r) {
                const int flat = chunk * 64 * FNGD_HEADS + r * FNGD_HEADS + t;
                if (flat < total) {
                    s += smem[r * FNGD_HEADS + t];
                    g[flat] = s;
                }
            }
        }
        return;
    }
    const int c = t & 127;
    const int half = t >> 7;
    const int r0 = row0 + half * 16;
    if (x >= 32) {
        // v: BF16 conv value -> F16.
        const int channel = 4096 + (x - 32) * 128 + c;
        unsigned short* dst = v16 + (x - 32) * 128 + c;
        if (r0 < rows) {
            fn_gdn_dense_conv_rows(input, kernel, history, next_history, rows, start_cursor,
                                   input_bf16, channel, r0,
                                   [&](int r, __hip_bfloat16 out) {
                                       dst[(size_t)r * 6144] = f16bits(bf(bf16_to_f32(out)));
                                   });
        }
        for (int r = r0 > rows ? r0 : rows; r < r0 + 16; ++r) dst[(size_t)r * 6144] = 0;
        return;
    }
    // q|k head x.
    const int channel = x * 128 + c;
    if (r0 < rows) {
        fn_gdn_dense_conv_rows(input, kernel, history, next_history, rows, start_cursor,
                               input_bf16, channel, r0,
                               [&](int r, __hip_bfloat16 out) {
                                   smem[(r - row0) * 129 + c] = bf16_to_f32(out);
                               });
    }
    __syncthreads();
    if (t < 32) {
        // pipe128 row params: serial sum of BF16-rounded squares, BF16 rsqrt.
        const float* h = smem + t * 129;
        float s = 0.0f;
        if (row0 + t < rows) {
            for (int cc = 0; cc < 128; ++cc) {
                const float xv = bf(h[cc]);
                s += bf(xv * xv);
            }
        }
        inv[t] = bf(rsqrtf(bf(s) + bf(1.0e-6f)));
    }
    __syncthreads();
    unsigned short* dst = x < 16 ? q16 + (size_t)x * 128 + c : k16 + (size_t)(x - 16) * 128 + c;
    for (int r = r0; r < r0 + 16; ++r) {
        unsigned short bits = 0;
        if (r < rows) {
            const float n = bf(bf(smem[(r - row0) * 129 + c]) * inv[r - row0]);
            bits = f16bits(x < 16 ? n * query_scale : n);
        }
        dst[(size_t)r * 2048] = bits;
    }
}

// The convolution's SiLU + BF16 rounding of one F32 value, exactly as the v1 kernel computes it.
static __device__ __forceinline__ __hip_bfloat16 fn_gdn_dense_silu_bf16_exact(float value) {
    return bf16_round(value * (1.0f / (1.0f + expf(-value))));
}

// Distance (in F32 ulps of the approximation's low 16 bits) from the BF16 round-to-nearest tie
// point below which the approximation is not trusted. The fast path's error against the exact
// chain is a few ulps (exp2 of a rounded product, hardware rcp), so an approximation whose
// low half is further than this from 0x8000 rounds to the same BF16 as the exact chain.
#define FNGD_SILU_TIE_MARGIN 64

// Cheap SiLU + BF16 rounding for the bulk of the range: hardware exp2/rcp instead of ocml's
// expf and the full-precision division (about 4x fewer instructions), accepted only when the
// approximation provably rounds to the exact chain's BF16: |value| in [2^-20, 16] (so the
// intermediates are normal and NaN/inf/zero/denormals fall to the exact path) and the
// approximation is not within FNGD_SILU_TIE_MARGIN ulps of a BF16 tie. The whole function is
// checked against `fn_gdn_dense_silu_bf16_exact` over all 2^32 F32 inputs by
// tests/fn_gdn_chain_oracle.rs (`silu_fast_path_is_exact_over_all_f32`).
static __device__ __forceinline__ bool fn_gdn_dense_silu_bf16_fast(float value, __hip_bfloat16& out) {
    const float a = fabsf(value);
    const float u = __builtin_amdgcn_exp2f(value * -1.4426950408889634f);
    const float y = value * __builtin_amdgcn_rcpf(1.0f + u);
    const unsigned yb = __float_as_uint(y);
    const int low = (int)(yb & 0xffffu);
    const int off = low - 0x8000;
    const bool ok = a >= 9.5367431640625e-07f && a <= 16.0f &&
                    (off > FNGD_SILU_TIE_MARGIN || off < -FNGD_SILU_TIE_MARGIN);
    out.data = (unsigned short)((yb + 0x7fffu + ((yb >> 16) & 1u)) >> 16);
    return ok;
}

static __device__ __forceinline__ __hip_bfloat16 fn_gdn_dense_silu_bf16(float value) {
    __hip_bfloat16 out;
    if (!fn_gdn_dense_silu_bf16_fast(value, out)) out = fn_gdn_dense_silu_bf16_exact(value);
    return out;
}

// ---- fn_gdn_dense_producer_v2 (HIPFIRE_QWEN4_GDN_CHAIN_FUSE) ----------------------------
//
// The same producer, byte for byte, with the convolution's global loads hoisted: the
// v1 loop interleaves one dependent load per row with the row's math and its store,
// so (the stores may alias `input`) the compiler waits on every load before the
// next one issues (27 VGPRs, `global_load; s_wait_loadcnt 0` per row). v2 loads the
// thread's 3 + 16 input values first, then runs the identical per-row expression
// chain. Specialized on the input dtype: `BF` reads the BF16 bits the exact-INT8
// MQ6 qkv GEMM's BF16 entry stores (RNE of the F32 accumulator, the rounding the F32
// input takes below), else the F32 projection.
template <bool BF, typename Emit>
static __device__ __forceinline__ void fn_gdn_dense_conv_rows_v2(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* next_history,
    int rows,
    int start_cursor,
    int channel,
    int r0,
    Emit emit) {
    const int channels = FNGD_CHANNELS;
    auto in = [&](int j) {
        if (BF) {
            __hip_bfloat16 b;
            b.data = ((const unsigned short*)input)[j * channels + channel];
            return b;
        }
        return bf16_round(input[j * channels + channel]);
    };
    auto x = [&](int j) {
        return j >= 0 ? in(j)
                      : bf16_round(history[((start_cursor + j + 3) % 3) * channels + channel]);
    };
    const float k0 = bf16_to_f32(kernel[channel * 4 + 0]);
    const float k1 = bf16_to_f32(kernel[channel * 4 + 1]);
    const float k2 = bf16_to_f32(kernel[channel * 4 + 2]);
    const float k3 = bf16_to_f32(kernel[channel * 4 + 3]);
    const int r1 = min(r0 + 16, rows);
    // The ring history (only the r0 == 0 tile; wave-uniform) ...
    float hist[3] = {0.0f, 0.0f, 0.0f};
    if (r0 == 0) {
#pragma unroll
        for (int s = 0; s < 3; ++s)
            hist[s] = history[((start_cursor + s) % 3) * channels + channel];
    }
    // ... and the input rows, loaded branch-free (rows outside [0, r1) re-read a
    // valid row and are never used) so all of them are in flight before any is
    // consumed and before any store.
    unsigned short rb[19];
    float rf[19];
#pragma unroll
    for (int i = 0; i < 19; ++i) {
        int j = r0 - 3 + i;
        j = j < 0 ? 0 : (j > r1 - 1 ? r1 - 1 : j);
        if (BF) {
            rb[i] = ((const unsigned short*)input)[j * channels + channel];
            rf[i] = 0.0f;
        } else {
            rf[i] = input[j * channels + channel];
            rb[i] = 0;
        }
    }
    __hip_bfloat16 xs[19];
#pragma unroll
    for (int i = 0; i < 19; ++i) {
        const int j = r0 - 3 + i;
        if (j < 0) {
            xs[i] = bf16_round(hist[i]);
        } else if (BF) {
            xs[i].data = rb[i];
        } else {
            xs[i] = bf16_round(rf[i]);
        }
    }
    __hip_bfloat16 tail[3];
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s) {
            const int j = rows - 3 + ((s - (start_cursor + rows) % 3) + 6) % 3;
            tail[s] = x(j);
        }
    }
    // ... then the per-row chain, identical to the v1 loop.
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int r = r0 + i;
        if (r < r1) {
            const __hip_bfloat16 current_bf16 = xs[i + 3];
            float value = k3 * bf16_to_f32(current_bf16);
            value += k0 * bf16_to_f32(xs[i]);
            value += k1 * bf16_to_f32(xs[i + 1]);
            value += k2 * bf16_to_f32(xs[i + 2]);
            const __hip_bfloat16 out = fn_gdn_dense_silu_bf16(value);
            emit(r, out);
        }
    }
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s)
            next_history[s * channels + channel] = bf16_to_f32(tail[s]);
    }
}

// Grid, block, arguments and every stored byte are fn_gdn_dense_producer's (see above), with
// one proven identity applied where the rounded value only feeds a bit conversion or a square:
// `bf(x)` of a value that is already a BF16 widening (low 16 bits zero: the `out` of the v rows,
// the conv output staged in `smem` read by the norm chain) returns it bit for bit (RNE adds
// < 0x8000 + lsb to a zero low half; inf/NaN with a zero low half skip both fix-ups), so that
// first rounding is dropped. The q/k final `bf(bf(smem) * inv)` keeps both roundings: with a
// NaN in both operands the product's payload follows the operand order the compiler picks. The entry is picked by the input dtype, `input_bf16` is carried only for ABI parity.
template <bool BF>
static __device__ __forceinline__ void fn_gdn_dense_producer_v2_body(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* next_history,
    unsigned short* q16,
    unsigned short* k16,
    unsigned short* v16,
    float* g,
    float* beta,
    const float* param_a,
    const float* param_b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    unsigned int* ef,
    int rows,
    int n64,
    int start_cursor,
    float query_scale,
    int ef_words) {
    __shared__ float smem[32 * 129];
    __shared__ float inv[32];
    const int t = (int)threadIdx.x;
    const int x = (int)blockIdx.x;
    const int row0 = (int)blockIdx.y * 32;
    if (x >= 80) {
        if (x == 81) {
            const int stride = (int)gridDim.y * 256;
            uint4* p = reinterpret_cast<uint4*>(ef);
            for (int i = (int)blockIdx.y * 256 + t; i < ef_words / 4; i += stride)
                p[i] = make_uint4(0u, 0u, 0u, 0u);
            return;
        }
        if (blockIdx.y & 1) return;
        const int chunk = (int)blockIdx.y >> 1;
        const int total = rows * FNGD_HEADS;
        for (int i = t; i < 64 * FNGD_HEADS; i += 256) {
            const int flat = chunk * 64 * FNGD_HEADS + i;
            smem[i] = flat < total
                ? fn_gdn_dense_params_flat(param_a, param_b, a_log, dt_bias, beta, flat)
                : 0.0f;
        }
        __syncthreads();
        if (t < FNGD_HEADS) {
            float s = 0.0f;
            for (int r = 0; r < 64; ++r) {
                const int flat = chunk * 64 * FNGD_HEADS + r * FNGD_HEADS + t;
                if (flat < total) {
                    s += smem[r * FNGD_HEADS + t];
                    g[flat] = s;
                }
            }
        }
        return;
    }
    const int c = t & 127;
    const int half = t >> 7;
    const int r0 = row0 + half * 16;
    if (x >= 32) {
        const int channel = 4096 + (x - 32) * 128 + c;
        unsigned short* dst = v16 + (x - 32) * 128 + c;
        if (r0 < rows) {
            fn_gdn_dense_conv_rows_v2<BF>(input, kernel, history, next_history, rows,
                                          start_cursor, channel, r0,
                                          [&](int r, __hip_bfloat16 out) {
                                              dst[(size_t)r * 6144] = f16bits(bf16_to_f32(out));
                                          });
        }
        for (int r = r0 > rows ? r0 : rows; r < r0 + 16; ++r) dst[(size_t)r * 6144] = 0;
        return;
    }
    const int channel = x * 128 + c;
    if (r0 < rows) {
        fn_gdn_dense_conv_rows_v2<BF>(input, kernel, history, next_history, rows, start_cursor,
                                      channel, r0,
                                      [&](int r, __hip_bfloat16 out) {
                                          smem[(r - row0) * 129 + c] = bf16_to_f32(out);
                                      });
    }
    __syncthreads();
    if (t < 32) {
        const float* h = smem + t * 129;
        float s = 0.0f;
        if (row0 + t < rows) {
            for (int cc = 0; cc < 128; ++cc) {
                const float xv = h[cc];
                s += bf(xv * xv);
            }
        }
        inv[t] = bf(rsqrtf(bf(s) + bf(1.0e-6f)));
    }
    __syncthreads();
    unsigned short* dst = x < 16 ? q16 + (size_t)x * 128 + c : k16 + (size_t)(x - 16) * 128 + c;
    for (int r = r0; r < r0 + 16; ++r) {
        unsigned short bits = 0;
        if (r < rows) {
            const float n = bf(bf(smem[(r - row0) * 129 + c]) * inv[r - row0]);
            bits = f16bits(x < 16 ? n * query_scale : n);
        }
        dst[(size_t)r * 2048] = bits;
    }
}

#define FNGD_PRODUCER_V2_ENTRY(NAME, BF)                                                      \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                   \
        const float* input, const __hip_bfloat16* kernel, const float* history,               \
        float* next_history, unsigned short* q16, unsigned short* k16, unsigned short* v16,   \
        float* g, float* beta, const float* param_a, const float* param_b,                    \
        const __hip_bfloat16* a_log, const __hip_bfloat16* dt_bias, unsigned int* ef,         \
        int rows, int n64, int start_cursor, int input_bf16, float query_scale,               \
        int ef_words) {                                                                       \
        fn_gdn_dense_producer_v2_body<BF>(input, kernel, history, next_history, q16, k16,     \
                                          v16, g, beta, param_a, param_b, a_log, dt_bias, ef, \
                                          rows, n64, start_cursor, query_scale, ef_words);    \
    }
FNGD_PRODUCER_V2_ENTRY(fn_gdn_dense_producer_v2_f32, false)
FNGD_PRODUCER_V2_ENTRY(fn_gdn_dense_producer_v2_bf16, true)

// The gated RMSNorm scale of one 128-wide head from its F32 values in LDS: the
// serial sum of squares in channel order (every lane reads the same values and
// walks the same chain) and gated_delta_gate_head128's rsqrt expression.
static __device__ __forceinline__ float fn_gdn_dense_gate_inv128(const float* rounded) {
    float sum = 0.0f;
    for (int i = 0; i < 128; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(rounded + i);
        sum += v.x * v.x;
        sum += v.y * v.y;
        sum += v.z * v.z;
        sum += v.w * v.w;
    }
    return rsqrtf(sum / 128.0f + 1.0e-6f);
}

// One gated output element: BF16-rounded norm product, BF16-rounded sigmoid gate.
static __device__ __forceinline__ float fn_gdn_dense_gate_elem(
    float in, float inv, float norm_w, float z) {
    const float normalized = bf16_to_f32(bf16_round(in * inv * norm_w));
    const float gate = bf16_to_f32(bf16_round(z));
    return bf16_to_f32(bf16_round(normalized / (1.0f + expf(-gate))));
}

// gated_delta_gate_head128 with `recurrent` the packed BF16 bits plane.
static __device__ __forceinline__ float4 fn_gdn_dense_gate_head128(
    const unsigned short* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    int start,
    int lane,
    float* rounded) {
    const uint2 raw = *reinterpret_cast<const uint2*>(recurrent + start + lane * 4);
    const float4 value = make_float4(
        __uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
        __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u));
    *reinterpret_cast<float4*>(rounded + lane * 4) = value;
    __syncthreads();

    const float inv = fn_gdn_dense_gate_inv128(rounded);
    const float4 zv = *reinterpret_cast<const float4*>(z + start + lane * 4);
    const float in[4] = {value.x, value.y, value.z, value.w};
    const float zs[4] = {zv.x, zv.y, zv.z, zv.w};
    float out[4];
#pragma unroll
    for (int e = 0; e < 4; ++e)
        out[e] = fn_gdn_dense_gate_elem(in[e], inv, bf16_to_f32(norm[lane * 4 + e]), zs[e]);
    const float4 result = make_float4(out[0], out[1], out[2], out[3]);
    *reinterpret_cast<float4*>(output + start + lane * 4) = result;
    return result;
}

// Grid [rows * 48], block 32.
extern "C" __global__ __launch_bounds__(32) void fn_gdn_dense_gate_bf16in(
    const unsigned short* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    int rows,
    int value_heads,
    int value_dim) {
    if ((int)blockIdx.x >= rows * value_heads || value_dim != 128) return;
    __shared__ float rounded[128];
    fn_gdn_dense_gate_head128(recurrent, z, norm, output, (int)blockIdx.x * 128,
                              (int)threadIdx.x, rounded);
}

// Grid [rows * 48 / 2], block 64: the head pair's 256 outputs form one
// rotation group, which wave 0 rotates from LDS.
extern "C" __global__ __launch_bounds__(64) void fn_gdn_dense_gate_rotate_bf16in(
    const unsigned short* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    float* rotated,
    const float* signs1,
    const float* signs2) {
    const int wave = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    __shared__ float rounded[2][128];
    __shared__ float pair[256];
    const float4 result = fn_gdn_dense_gate_head128(
        recurrent, z, norm, output, ((int)blockIdx.x * 2 + wave) * 128, lane, rounded[wave]);
    *reinterpret_cast<float4*>(pair + wave * 128 + lane * 4) = result;
    __syncthreads();
    if (wave != 0) return;
    const int d0 = lane * 8;
    float values[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) values[e] = pair[d0 + e] * signs1[d0 + e];
    mq_fwht256_lane(values, lane);
#pragma unroll
    for (int e = 0; e < 8; ++e)
        rotated[(int)blockIdx.x * 256 + d0 + e] = values[e] * 0.0625f * signs2[d0 + e];
}

// Grid [rows * 12], block 64: gated norm + the out projection's A8 producer in
// one launch, one wave per head pair. Pair g = row * 24 + p (p = 0..23) is one
// 256-wide FWHT rotation group of the MQ6G256V2 out projection (K = 6144 = 24
// pairs); lane L owns pair elements 8L..8L+7, i.e. head L >> 4, channels
// (L & 15) * 8.., which is exactly the register layout of `mq_rotate_x`'s FWHT:
// the gate math (the rotate kernel's, one shared chain per head) feeds the
// rotation (`mq_fwht256_lane`, `mq_rotate_x`'s signs and scale) in registers, and
// the group's two `block_i8_128` blocks go out through the same register emit
// `mq_rotate_x_i8_gfx12` uses (block_i8_128_emit.hip), at block index
// (2 * p + half) * rows + row. The F32 gate output and the F32 rotation never reach
// memory; the blocks are bytewise `mq_rotate_x_i8_gfx12` applied to
// `fn_gdn_dense_gate_bf16in`'s output. The recurrent plane, z, norm and the
// sign tables must be 16-byte aligned (host checked).
extern "C" __global__ __launch_bounds__(64) void fn_gdn_dense_gate_i8(
    const unsigned short* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    block_i8_128* y,
    const float* signs1,
    const float* signs2,
    int rows) {
    const int wave = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    // Per wave: head 0 at [0, 128), head 1 at [132, 260) (4 floats of pad keep
    // the two heads' chain reads off the same banks).
    __shared__ float rounded[2][264];
    const int g = (int)blockIdx.x * 2 + wave;
    const long long start = (long long)g * 256 + lane * 8;
    const int d0 = lane * 8;
    const int c0 = (lane & 15) * 8;

    const uint4 raw = *reinterpret_cast<const uint4*>(recurrent + start);
    const float4 z0 = *reinterpret_cast<const float4*>(z + start);
    const float4 z1 = *reinterpret_cast<const float4*>(z + start + 4);
    float nw[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) nw[e] = bf16_to_f32(norm[c0 + e]);
    const float4 s1a = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s1b = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s2a = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s2b = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    const float in[8] = {
        __uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
        __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u),
        __uint_as_float(raw.z << 16), __uint_as_float(raw.z & 0xffff0000u),
        __uint_as_float(raw.w << 16), __uint_as_float(raw.w & 0xffff0000u)};
    float* head = rounded[wave] + (lane >> 4) * 132;
    *reinterpret_cast<float4*>(head + c0) = make_float4(in[0], in[1], in[2], in[3]);
    *reinterpret_cast<float4*>(head + c0 + 4) = make_float4(in[4], in[5], in[6], in[7]);
    // One wave owns `rounded[wave]`: keep the compiler from moving the chain's
    // reads above the stores (LDS ops of a wave execute in order).
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    __builtin_amdgcn_wave_barrier();

    const float inv = fn_gdn_dense_gate_inv128(head);
    const float zs[8] = {z0.x, z0.y, z0.z, z0.w, z1.x, z1.y, z1.z, z1.w};
    const float sg1[8] = {s1a.x, s1a.y, s1a.z, s1a.w, s1b.x, s1b.y, s1b.z, s1b.w};
    const float sg2[8] = {s2a.x, s2a.y, s2a.z, s2a.w, s2b.x, s2b.y, s2b.z, s2b.w};
    float values[8];
#pragma unroll
    for (int e = 0; e < 8; ++e)
        values[e] = fn_gdn_dense_gate_elem(in[e], inv, nw[e], zs[e]) * sg1[e];
    mq_fwht256_lane(values, lane);
    float f[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = values[e] * 0.0625f * sg2[e];
    const int token = g / 24;
    emit_i8_sidecar_from_producer8(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], y,
                                   g - token * 24, token, rows, lane);
}
