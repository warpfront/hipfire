#include <hip/hip_runtime.h>

// Per-routed-expert AWQ + FWHT-256 producer for the MoE gate_up input stage.
//
// One shared activation row `x_in [K]` (the normalized FFN row) feeds the
// `top_k` routed experts of a token. Expert `e` was quantized against its own
// F32 AWQ vector `S_e` (`stored = Q(R . (W (.) S_e))`), so each rank needs its
// own prescaled+rotated row:
//
//     x_out[r, :] = FWHT256( x_in / S[expert_ids[r]] )      r in [0, top_k)
//
// Same arithmetic as `rotate_x_mq_awq` (exact f32 divide BEFORE signs1, same
// butterfly, same `* 0.0625f * signs2`), so with S == 1.0f everywhere every
// output row is bitwise `mq_rotate_x(x_in)`. An id outside `[0, n_exp)` (the
// router's -1 marker) selects no table entry: that row is written as zeros,
// never stale memory.
//
// scale_ptrs : [n_exp] u64 device pointers, each -> that expert's [K] f32 AWQ vector
// expert_ids : [top_k] i32
// x_out      : [top_k, K] f32 row-major, rank order
// grid       : [(K/256) * top_k, 1, 1] folded as block = rank * (K/256) + group
// block      : 32 threads, 8 elements each (as `rotate_x_mq_awq`)

extern "C" __launch_bounds__(32, 16)
__global__ void rotate_x_mq_awq_topk(
    const float* __restrict__ x_in,                          // [K], shared by every rank
    float* __restrict__ x_out,                               // [top_k, K]
    const unsigned long long* __restrict__ scale_ptrs,       // [n_exp] -> [K] f32
    const int* __restrict__ expert_ids,                      // [top_k]
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K,
    int n_exp
) {
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (groups_total <= 0) return;
    const unsigned bx = blockIdx.x;
    const int rank = (int)(bx / (unsigned)groups_total);
    const int group = (int)(bx - (unsigned)rank * (unsigned)groups_total);
    const int expert_id = expert_ids[rank];

    float* x_out_b = x_out + (long long)rank * K;

    const int d0 = tid * 8;
    const int base = group * 256 + d0;

    if (expert_id < 0 || expert_id >= n_exp) {
#pragma unroll
        for (int e = 0; e < 8; ++e) x_out_b[base + e] = 0.0f;
        return;
    }
    const float* __restrict__ awq_scale =
        reinterpret_cast<const float*>(scale_ptrs[expert_id]);

    float v0 = (x_in[base    ] / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (x_in[base + 1] / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (x_in[base + 2] / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (x_in[base + 3] / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (x_in[base + 4] / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (x_in[base + 5] / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (x_in[base + 6] / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (x_in[base + 7] / awq_scale[base + 7]) * signs1[d0 + 7];

    // Local butterfly: strides 1, 2, 4
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle.
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)

    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float s = 0.0625f;
    x_out_b[base    ] = v0 * s * signs2[d0    ];
    x_out_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_out_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_out_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_out_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_out_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_out_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_out_b[base + 7] = v7 * s * signs2[d0 + 7];
}
