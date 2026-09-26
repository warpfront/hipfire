#include <hip/hip_runtime.h>

// QSA attention when every row's selection is its whole causal window (the
// index budget covers every visible block), as causal GQA flash attention on
// gfx11 F16 WMMA.  Replaces indexed_attention_attention_f32_batched_hg4 for
// that case: same scale (1/16), softmax and sigmoid output gate, but Q, K, V
// and the probabilities are rounded to F16 and summed in WMMA / tile order,
// so it is not bit-exact (callers are KLD-gated).
//
// indexed_attention_kv_f16 first writes the cache rows as F16 K and V^T.
// Block: 16 query rows x 4 heads of one KV group, two waves per head; each
// 32-key tile of K and V^T is staged once in LDS for the four heads, every
// operand in fragment-major order (a lane's 16 halves contiguous).  Scores
// are computed transposed (S^T = K Q^T) so a lane owns one query: the online
// softmax is lane-local up to one permlanex16, and the probabilities feed
// O^T += V^T P^T as the B operand without an LDS round trip.  Both waves of a
// head compute the same scores and softmax; each keeps half of O^T (16
// queries x 128 dims) in registers, which halves the accumulators and doubles
// the waves that hide the tile loads.
// Grid [ceil(rows / 16), n_heads / 4], block 256.  head_dim 256,
// (n_heads / n_kv_heads) % 4 == 0, tpad % 32 == 0.

typedef _Float16 __attribute__((ext_vector_type(16))) qsa_h16_t;
typedef float __attribute__((ext_vector_type(8))) qsa_f8_t;

static __device__ __forceinline__ float qsa_other_half(float v) {
    return __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
}

// Cache rows [0, tokens) as F16: K as [token][kv_heads * 256], V transposed
// as [kv_heads * 256][tpad] with tokens [tokens, tpad) zeroed (masked keys
// then multiply a finite 0).  Grid [tpad, kv_heads], block 256.
extern "C" __global__ void indexed_attention_kv_f16(
    const float* __restrict__ k, const float* __restrict__ v, _Float16* __restrict__ k16,
    _Float16* __restrict__ vt16, int tokens, int kv_heads, int tpad) {
    const int tok = blockIdx.x;
    const int c = threadIdx.x + blockIdx.y * 256;
    const int width = kv_heads * 256;
    if (tok >= tpad || c >= width) return;
    if (tok < tokens) {
        k16[(size_t)tok * width + c] = (_Float16)k[(size_t)tok * width + c];
        vt16[(size_t)c * tpad + tok] = (_Float16)v[(size_t)tok * width + c];
    } else {
        vt16[(size_t)c * tpad + tok] = (_Float16)0.0f;
    }
}
extern "C" __global__ __launch_bounds__(256) void indexed_attention_dense_wmma_f16(
    const float* __restrict__ q_with_gate, const _Float16* __restrict__ k16,
    const _Float16* __restrict__ vt16, int tpad, float* __restrict__ output, int rows,
    int position_start, int n_heads, int n_kv_heads) {
    constexpr int HD = 256;
    // Fragment-major: element (row r, dim chunk c) at (c * ROWS + r) * 16.
    __shared__ __align__(16) _Float16 q_lds[4 * 16 * HD];     // [wave][dc][query][16]
    __shared__ __align__(16) _Float16 k_lds[32 * HD];         // [dc][key 0..31][16]
    __shared__ __align__(16) _Float16 vt_lds[2 * HD * 16];    // [ks][dim][16 keys]
    const int tid = threadIdx.x, wave8 = tid >> 5, lane = tid & 31, l16 = lane & 15, half = lane >> 4;
    const int wave = wave8 & 3, dh = wave8 >> 2;  // head in block, dim half
    const int t0 = blockIdx.x * 16;
    const int head = blockIdx.y * 4 + wave;
    const int kv_head = (blockIdx.y * 4) / (n_heads / n_kv_heads);
    const int q_stride = n_heads * 2 * HD, kv_stride = n_kv_heads * HD;
    _Float16* qw = q_lds + wave * 16 * HD;
    for (int e = lane + 32 * dh; e < 16 * HD / 4; e += 64) {
        const int q = e / (HD / 4), d4 = e - q * (HD / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (t0 + q < rows) v = *(const float4*)(q_with_gate + (size_t)(t0 + q) * q_stride + head * 2 * HD + d4 * 4);
        const int d = d4 * 4;
        _Float16* dst = qw + ((d >> 4) * 16 + q) * 16 + (d & 15);
        dst[0] = (_Float16)v.x; dst[1] = (_Float16)v.y; dst[2] = (_Float16)v.z; dst[3] = (_Float16)v.w;
    }
    const int kv_hi = position_start + min(t0 + 16, rows);
    const int my_visible = position_start + t0 + l16 + 1;
    float m_run = -INFINITY, l_run = 0.f;
    qsa_f8_t o[8];
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) o[dt] = qsa_f8_t{0,0,0,0,0,0,0,0};
    // The next key tile's K and V^T pieces are fetched into registers while
    // this tile computes; the barriers order LDS only.
#define AT_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
    uint4 kp[4], vp[4];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e = tid + 256 * i, key = e >> 5, c = e & 31;
            kp[i] = k0 + key < kv_hi
                ? *(const uint4*)(k16 + (size_t)(k0 + key) * kv_stride + kv_head * HD + c * 8)
                : make_uint4(0, 0, 0, 0);
            const int ks = e / (HD * 2), r = e - ks * HD * 2, dim = r >> 1, h = r & 1;
            vp[i] = *(const uint4*)(vt16 + (size_t)(kv_head * HD + dim) * tpad + k0 + ks * 16 + h * 8);
        }
    };
    fetch(0);
    __syncthreads();  // q_lds complete
    for (int k0 = 0; k0 < kv_hi; k0 += 32) {
        AT_BARRIER();  // previous tile's K/V reads are done
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e = tid + 256 * i, key = e >> 5, c = e & 31;
            *(uint4*)(k_lds + ((c >> 1) * 32 + key) * 16 + (c & 1) * 8) = kp[i];
            const int ks = e / (HD * 2), r = e - ks * HD * 2, dim = r >> 1, h = r & 1;
            *(uint4*)(vt_lds + (ks * HD + dim) * 16 + h * 8) = vp[i];
        }
        AT_BARRIER();
        if (k0 + 32 < kv_hi) fetch(k0 + 32);
        qsa_f8_t s0 = qsa_f8_t{0,0,0,0,0,0,0,0}, s1 = s0;
#pragma unroll 8  // full unroll: 243 VGPRs, slower
        for (int dc = 0; dc < HD / 16; ++dc) {
            const qsa_h16_t b = *(const qsa_h16_t*)(qw + (dc * 16 + l16) * 16);
            const qsa_h16_t a0 = *(const qsa_h16_t*)(k_lds + (dc * 32 + l16) * 16);
            const qsa_h16_t a1 = *(const qsa_h16_t*)(k_lds + (dc * 32 + 16 + l16) * 16);
            s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b, s0);
            s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b, s1);
        }
        float p[16];
        float tmax = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int t_a = k0 + 2 * i + half, t_b = t_a + 16;
            p[i] = t_a < my_visible ? s0[i] * 0.0625f : -INFINITY;
            p[8 + i] = t_b < my_visible ? s1[i] * 0.0625f : -INFINITY;
            tmax = fmaxf(tmax, fmaxf(p[i], p[8 + i]));
        }
        tmax = fmaxf(tmax, qsa_other_half(tmax));
        const float m_new = fmaxf(m_run, tmax);
        float alpha = 1.f, tsum = 0.f;
        if (m_new != -INFINITY) {
            alpha = expf(m_run - m_new);
#pragma unroll
            for (int i = 0; i < 16; ++i) { p[i] = expf(p[i] - m_new); tsum += p[i]; }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) p[i] = 0.f;
        }
        tsum += qsa_other_half(tsum);
        l_run = l_run * alpha + tsum;
        m_run = m_new;
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) o[dt] *= alpha;
        qsa_h16_t pb0, pb1;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float x0 = qsa_other_half(p[i]), x1 = qsa_other_half(p[8 + i]);
            pb0[2 * i] = (_Float16)(half ? x0 : p[i]);
            pb0[2 * i + 1] = (_Float16)(half ? p[i] : x0);
            pb1[2 * i] = (_Float16)(half ? x1 : p[8 + i]);
            pb1[2 * i + 1] = (_Float16)(half ? p[8 + i] : x1);
        }
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            const int dg = 8 * dh + dt;
            const qsa_h16_t a0 = *(const qsa_h16_t*)(vt_lds + (dg * 16 + l16) * 16);
            const qsa_h16_t a1 = *(const qsa_h16_t*)(vt_lds + (HD + dg * 16 + l16) * 16);
            o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, pb0, o[dt]);
            o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, pb1, o[dt]);
        }
    }
    // Lane l < 16 holds dims 2i of query l, lane l + 16 dims 2i + 1: one
    // cross-half swap per value gives the low lane dims 0..7 and the high
    // lane dims 8..15 of each 16-dim tile, so the gate loads and output
    // stores are 16-byte runs.
    const int row = t0 + l16;
    const float* gate_row = q_with_gate + (size_t)min(row, rows - 1) * q_stride + head * 2 * HD + HD;
    float* out_row = output + (size_t)row * n_heads * HD + head * HD;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        float r[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float got = qsa_other_half(half ? o[dt][i] : o[dt][4 + i]);
            const float own = half ? o[dt][4 + i] : o[dt][i];
            r[2 * i] = half ? got : own;
            r[2 * i + 1] = half ? own : got;
        }
        if (row >= rows) continue;
        const int d0 = (8 * dh + dt) * 16 + 8 * half;
        const float4 g0 = *(const float4*)(gate_row + d0), g1 = *(const float4*)(gate_row + d0 + 4);
        const float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        float y[8];
#pragma unroll
        for (int e = 0; e < 8; e++)
            y[e] = l_run > 0.f ? r[e] / l_run / (1.0f + expf(-g[e])) : 0.f;
        *(float4*)(out_row + d0) = make_float4(y[0], y[1], y[2], y[3]);
        *(float4*)(out_row + d0 + 4) = make_float4(y[4], y[5], y[6], y[7]);
    }
#undef AT_BARRIER
}
