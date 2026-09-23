// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 GDN resident WY/carry/output kernel. K2 supplies the C64 inverse B
// and inclusive gate prefix G. One workgroup owns one (head, BV32 value tile),
// keeps the complete 128x32 state slice in FP32 registers across all chunks,
// emits output from immutable chunk-entry state, and commits Q8+EF once.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) gdn_wr_h8;
typedef float __attribute__((ext_vector_type(8))) gdn_wr_f8;

__device__ __forceinline__ gdn_wr_f8 gdn_wr_mma(
    gdn_wr_h8 a, gdn_wr_h8 b, gdn_wr_f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ gdn_wr_f8 gdn_wr_zero8() {
    return (gdn_wr_f8){0.0f, 0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f, 0.0f};
}

extern "C" __launch_bounds__(256, 2)
__global__ void gdn_wy_carry_output_q8_gfx1201(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ beta,
    const float* __restrict__ B,
    const float* __restrict__ G,
    signed char* __restrict__ sq,
    float* __restrict__ sc,
    _Float16* __restrict__ ef,
    float* __restrict__ out,
    int N,
    int H,
    unsigned long long frame) {
    constexpr int C = 64;
    constexpr int D = 128;
    constexpr int BV = 32;

    const int head = blockIdx.x;
    const int value_tile = blockIdx.y;
    if (head >= H || value_tile >= 4 || N <= 0) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const int value_base = value_tile * BV;
    const unsigned token_stride = (unsigned)H * D;
    const unsigned head_off = (unsigned)head * D;
    const unsigned state_head = (unsigned)head * D * D;

    // 21,504 bytes. D remains FP32 until carry; panel/aux are phase-overlaid.
    struct __attribute__((aligned(16))) ResidentLds {
        float d[C * BV];          // 8,192 B
        _Float16 panel[C * C];    // 8,192 B
        _Float16 aux[C * BV];     // 4,096 B
        float ctrl[256];          // 1,024 B: G, exp(G), beta, spare
    };
    __shared__ ResidentLds lds;

    // wave w owns four value rows; lane owns four adjacent key columns.
    float state[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int vr = value_base + wave * 4 + r;
        const float scale = sc[(unsigned)head * D + (unsigned)vr];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = lane * 4 + j;
            state[r][j] = scale * (float)sq[state_head +
                                               (unsigned)vr * D +
                                               (unsigned)key];
        }
    }

    const int chunks = (N + C - 1) / C;
#pragma clang loop unroll(disable)
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int c0 = chunk * C;
        const int cn = min(C, N - c0);
        const unsigned bh_base = ((unsigned)chunk * (unsigned)H +
                                  (unsigned)head) * C * C;
        const unsigned g_base = ((unsigned)chunk * (unsigned)H +
                                 (unsigned)head) * C;

        if (tid < C) {
            const float gv = G[g_base + (unsigned)tid];
            lds.ctrl[tid] = gv;
            lds.ctrl[64 + tid] = expf(gv);
            lds.ctrl[128 + tid] = tid < cn
                ? beta[(unsigned)(c0 + tid) * (unsigned)H + (unsigned)head]
                : 0.0f;
        }
        __syncthreads();

        // Form D sixteen rows at a time. U remains FP32 in the two consumer
        // waves; each W key half is materialized only long enough for W*H.
#pragma unroll
        for (int rt = 0; rt < 4; ++rt) {
            // B rows [rt*16,rt*16+16), cast once for U.
#pragma clang loop unroll(disable)
            for (int x = tid; x < 16 * C; x += 256) {
                const int rr = x >> 6;
                const int kk = x & 63;
                lds.aux[x] = (_Float16)B[bh_base +
                    (unsigned)(rt * 16 + rr) * C + (unsigned)kk];
            }
            // beta*V rhs, transposed [value, reduction-row].
#pragma clang loop unroll(disable)
            for (int x = tid; x < BV * C; x += 256) {
                const int vc = x >> 6;
                const int rr = x & 63;
                float rhs = 0.0f;
                if (rr < cn) {
                    const unsigned tok = (unsigned)(c0 + rr);
                    rhs = lds.ctrl[128 + rr] *
                          v[tok * token_stride + head_off +
                            (unsigned)(value_base + vc)];
                }
                lds.panel[x] = (_Float16)rhs;
            }
            __syncthreads();

            gdn_wr_f8 uacc = gdn_wr_zero8();
            if (wave < 2) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    gdn_wr_h8 a, b;
                    const int kc = kk * 16 + half_id * 8;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        a[j] = lds.aux[(half * C) + kc + j];
                        b[j] = lds.panel[((wave * 16 + half) * C) + kc + j];
                    }
                    uacc = gdn_wr_mma(a, b, uacc);
                }
            }
            __syncthreads();

            gdn_wr_f8 whacc = gdn_wr_zero8();
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                // U consumed B only logically; restore B after the preceding W
                // half overwrote aux with its short-lived W tile.
                if (kh != 0) {
#pragma clang loop unroll(disable)
                    for (int x = tid; x < 16 * C; x += 256) {
                        const int rr = x >> 6;
                        const int kk = x & 63;
                        lds.aux[x] = (_Float16)B[bh_base +
                            (unsigned)(rt * 16 + rr) * C + (unsigned)kk];
                    }
                }
                // beta*exp(G)*K rhs [key-column, reduction-row].
#pragma clang loop unroll(disable)
                for (int x = tid; x < C * C; x += 256) {
                    const int key = x >> 6;
                    const int rr = x & 63;
                    float rhs = 0.0f;
                    if (rr < cn) {
                        const unsigned tok = (unsigned)(c0 + rr);
                        rhs = lds.ctrl[128 + rr] * lds.ctrl[64 + rr] *
                              k[tok * token_stride + head_off +
                                (unsigned)(kh * 64 + key)];
                    }
                    lds.panel[x] = (_Float16)rhs;
                }
                __syncthreads();

                if (wave < 4) {
                    gdn_wr_f8 wacc = gdn_wr_zero8();
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        gdn_wr_h8 a, b;
                        const int kc = kk * 16 + half_id * 8;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            a[j] = lds.aux[(half * C) + kc + j];
                            b[j] = lds.panel[((wave * 16 + half) * C) + kc + j];
                        }
                        wacc = gdn_wr_mma(a, b, wacc);
                    }
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int rr = half_id * 8 + j;
                        const int key = wave * 16 + half;
                        lds.aux[rr * C + key] = (_Float16)wacc[j];
                    }
                }
                __syncthreads();

                // Transient H operand [value-column,key-in-half]. State itself
                // remains FP32 in registers and is not changed before output.
                {
                    const int first_key = lane * 4;
                    if ((kh == 0 && first_key < 64) ||
                        (kh == 1 && first_key >= 64)) {
#pragma unroll
                        for (int r = 0; r < 4; ++r)
#pragma unroll
                            for (int j = 0; j < 4; ++j)
                                lds.panel[(wave * 4 + r) * C +
                                          first_key + j - kh * 64] =
                                    (_Float16)state[r][j];
                    }
                }
                __syncthreads();

                if (wave < 2) {
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        gdn_wr_h8 a, b;
                        const int kc = kk * 16 + half_id * 8;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            a[j] = lds.aux[(half * C) + kc + j];
                            b[j] = lds.panel[((wave * 16 + half) * C) + kc + j];
                        }
                        whacc = gdn_wr_mma(a, b, whacc);
                    }
                }
                __syncthreads();
            }

            if (wave < 2) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int rr = rt * 16 + half_id * 8 + j;
                    const int vc = wave * 16 + half;
                    lds.d[rr * BV + vc] = uacc[j] - whacc[j];
                }
            }
            __syncthreads();
        }

        // Emit O from the immutable chunk-entry state before the carry update.
#pragma unroll
        for (int rt = 0; rt < 4; ++rt) {
            // Q rows [16,128], cast into aux.
#pragma clang loop unroll(disable)
            for (int x = tid; x < 16 * D; x += 256) {
                const int rr = x >> 7;
                const int key = x & 127;
                const int local_row = rt * 16 + rr;
                float qv = 0.0f;
                if (local_row < cn)
                    qv = q[(unsigned)(c0 + local_row) * token_stride +
                           head_off + (unsigned)key];
                lds.aux[x] = (_Float16)qv;
            }
            __syncthreads();

            gdn_wr_f8 oacc = gdn_wr_zero8();
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                {
                    const int first_key = lane * 4;
                    if ((kh == 0 && first_key < 64) ||
                        (kh == 1 && first_key >= 64)) {
#pragma unroll
                        for (int r = 0; r < 4; ++r)
#pragma unroll
                            for (int j = 0; j < 4; ++j)
                                lds.panel[(wave * 4 + r) * C +
                                          first_key + j - kh * 64] =
                                    (_Float16)state[r][j];
                    }
                }
                __syncthreads();
                if (wave < 2) {
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        gdn_wr_h8 a, b;
                        const int kc = kk * 16 + half_id * 8;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            a[j] = lds.aux[(half * D) + kh * 64 + kc + j];
                            b[j] = lds.panel[((wave * 16 + half) * C) + kc + j];
                        }
                        oacc = gdn_wr_mma(a, b, oacc);
                    }
                }
                __syncthreads();
            }
            if (wave < 2) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int rr = rt * 16 + half_id * 8 + j;
                    oacc[j] *= lds.ctrl[64 + rr];
                }
            }

            // M=causal decay*(Q K^T), accumulated before aux is overwritten.
            gdn_wr_f8 macc = gdn_wr_zero8();
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
#pragma clang loop unroll(disable)
                for (int x = tid; x < C * C; x += 256) {
                    const int tr = x >> 6;
                    const int key_local = x & 63;
                    float kv = 0.0f;
                    if (tr < cn)
                        kv = k[(unsigned)(c0 + tr) * token_stride + head_off +
                               (unsigned)(kh * 64 + key_local)];
                    lds.panel[x] = (_Float16)kv;
                }
                __syncthreads();
                if (wave < 4) {
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        gdn_wr_h8 a, b;
                        const int kc = kk * 16 + half_id * 8;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            a[j] = lds.aux[(half * D) + kh * 64 + kc + j];
                            b[j] = lds.panel[((wave * 16 + half) * C) + kc + j];
                        }
                        macc = gdn_wr_mma(a, b, macc);
                    }
                }
                __syncthreads();
            }
            if (wave < 4) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int rr = rt * 16 + half_id * 8 + j;
                    const int col = wave * 16 + half;
                    float mv = 0.0f;
                    if (rr < cn && col <= rr)
                        mv = macc[j] * expf(lds.ctrl[rr] - lds.ctrl[col]);
                    lds.aux[(half_id * 8 + j) * C + col] = (_Float16)mv;
                }
            }
            __syncthreads();

            // D operand transposed [value,token].
#pragma clang loop unroll(disable)
            for (int x = tid; x < BV * C; x += 256) {
                const int vc = x >> 6;
                const int rr = x & 63;
                lds.panel[x] = (_Float16)lds.d[rr * BV + vc];
            }
            __syncthreads();
            if (wave < 2) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    gdn_wr_h8 a, b;
                    const int kc = kk * 16 + half_id * 8;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        a[j] = lds.aux[(half * C) + kc + j];
                        b[j] = lds.panel[((wave * 16 + half) * C) + kc + j];
                    }
                    oacc = gdn_wr_mma(a, b, oacc);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int rr = rt * 16 + half_id * 8 + j;
                    if (rr < cn)
                        out[(unsigned)(c0 + rr) * token_stride + head_off +
                            (unsigned)(value_base + wave * 16 + half)] = oacc[j];
                }
            }
            __syncthreads();
        }

        // Carry: D^T * diag(exp(Glast-G))*K. aux holds scaled D^T;
        // panel first holds K^T then the FP32 carry result for owner reload.
        const float glast = lds.ctrl[cn - 1];
#pragma clang loop unroll(disable)
        for (int x = tid; x < BV * C; x += 256) {
            const int vc = x >> 6;
            const int rr = x & 63;
            float dv = 0.0f;
            if (rr < cn)
                dv = lds.d[rr * BV + vc] * expf(glast - lds.ctrl[rr]);
            lds.aux[x] = (_Float16)dv;
        }
        __syncthreads();

        const float chunk_decay = expf(glast);
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                state[r][j] *= chunk_decay;

#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
#pragma clang loop unroll(disable)
            for (int x = tid; x < C * C; x += 256) {
                const int key_local = x >> 6;
                const int rr = x & 63;
                float kv = 0.0f;
                if (rr < cn)
                    kv = k[(unsigned)(c0 + rr) * token_stride + head_off +
                           (unsigned)(kh * 64 + key_local)];
                lds.panel[x] = (_Float16)kv;
            }
            __syncthreads();

            const int rb = wave >> 2;
            const int cb = wave & 3;
            gdn_wr_f8 cacc = gdn_wr_zero8();
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                gdn_wr_h8 a, b;
                const int kc = kk * 16 + half_id * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    a[j] = lds.aux[((rb * 16 + half) * C) + kc + j];
                    b[j] = lds.panel[((cb * 16 + half) * C) + kc + j];
                }
                cacc = gdn_wr_mma(a, b, cacc);
            }
            float* carry = reinterpret_cast<float*>(lds.panel);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int vc = rb * 16 + half_id * 8 + j;
                const int key_local = cb * 16 + half;
                carry[vc * C + key_local] = cacc[j];
            }
            __syncthreads();

            const int first_key = lane * 4;
            if ((kh == 0 && first_key < 64) || (kh == 1 && first_key >= 64)) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int vc = wave * 4 + r;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int key_local = first_key + j - kh * 64;
                        state[r][j] += carry[vc * C + key_local];
                    }
                }
            }
            __syncthreads();
        }
    }

    // Single-end deterministic Q8+EF commit. EF is folded exactly once.
    (void)frame;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int vr = value_base + wave * 4 + r;
        float s[4];
        float row_max = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = lane * 4 + j;
            const unsigned off = state_head + (unsigned)vr * D + (unsigned)key;
            s[j] = state[r][j] + (float)ef[off];
            row_max = fmaxf(row_max, fabsf(s[j]));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            row_max = fmaxf(row_max, __shfl_xor(row_max, o));
        const float inv = row_max > 0.0f ? 127.0f / row_max : 0.0f;
        const float scale = row_max > 0.0f ? row_max / 127.0f : 1.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = lane * 4 + j;
            const unsigned off = state_head + (unsigned)vr * D + (unsigned)key;
            const float qf = fminf(fmaxf(rintf(s[j] * inv), -128.0f), 127.0f);
            sq[off] = (signed char)qf;
            ef[off] = (_Float16)(s[j] - qf * scale);
        }
        if (lane == 0)
            sc[(unsigned)head * D + (unsigned)vr] = scale;
    }
}
