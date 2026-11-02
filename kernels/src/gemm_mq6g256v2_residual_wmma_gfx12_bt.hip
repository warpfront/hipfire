// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
#define MQ6_DQ6(i,dw,sh) a[i]=sc*(_Float16)(float)(((dw)>>(sh))&63u)+zp
#define MQ6_DQ6_8(d0,d1) do { MQ6_DQ6(0,(d0),0); MQ6_DQ6(1,(d0),6); MQ6_DQ6(2,(d0),12); MQ6_DQ6(3,(d0),18); MQ6_DQ6(4,(d1),0); MQ6_DQ6(5,(d1),6); MQ6_DQ6(6,(d1),12); MQ6_DQ6(7,(d1),18); } while(0)
#define MQ6_HDR_LOAD(gp) const unsigned int hA=*(const unsigned int*)(gp); const unsigned int hB=*(const unsigned int*)(gp+4); const _Float16 sc0=(_Float16)__half2float(__ushort_as_half((unsigned short)(hA&0xFFFFu))); const _Float16 zp0=(_Float16)__half2float(__ushort_as_half((unsigned short)(hA>>16))); const _Float16 sc1=(_Float16)__half2float(__ushort_as_half((unsigned short)(hB&0xFFFFu))); const _Float16 zp1=(_Float16)__half2float(__ushort_as_half((unsigned short)(hB>>16)))
#define MQ6_HDR_SEL _Float16 sc=(kt<8)?sc0:sc1; _Float16 zp=(kt<8)?zp0:zp1;
#define RES_BT_EPI_PLAIN(BV) for(int b=0;b<BV;b++){ const int oc=bs+b*16+ml; if(oc<N) for(int j=0;j<8;j++){ int orow=rs+8*kg+j; if(orow<M) Y[(long long)oc*M+orow]+=acc[b][j]; } }
#ifndef HCW_EPILOGUE_BLOCK
#define HCW_EPILOGUE_BLOCK
// ---------------------------------------------------------------------------
// H4 HC-write epilogue: the producer's own output stage applies
// hyper_write_bf16x2 / hyper_write_f32's expression to the streams in place.
// Stage 1 (per accumulator lane, tokens x 8 channels): the GEMM output as the
// unfused kernel stores it (0.0f + acc) is rounded to BF16 bits and parked in
// LDS (the unfused HC write rounds it to BF16 before use, so nothing is lost);
// an optional F32 mirror gets the unrounded value.  Stage 2 (accumulators
// dead): each item = (token t, channels c0..c0+7) reads the four stream
// columns br*H + c0.. of token t (old values), applies the four gates (the
// paired read's hyper_norm_gate_outputs logits through the same expression as
// hyper_write_bf16x2) and overwrites them; one item owns its columns.
// Requires H % 8 == 0 (the caller's M).  Keep byte-identical across the files
// carrying this block (gfx11 bt, gfx12 bt, lds256).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float hcw_bf16f(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}
static __device__ __forceinline__ unsigned int hcw_bf16bits(float v) {
    return (unsigned int)static_cast<hip_bfloat16>(v).data;
}
// NaN propagation is operand-order dependent (the first NaN source wins) and the
// compiler commutes fadd/fmul freely, so the two products/sums below are pinned to
// the operand order hyper_write_bf16x2 (stream + injection) and hyper_write_f32
// (injection + stream) compile to; gate * mixed is gate-first in both.
static __device__ __forceinline__ float hcw_mul(float a, float b) {
    float r;
    asm("v_mul_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
static __device__ __forceinline__ float hcw_add(float a, float b) {
    float r;
    asm("v_add_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
static __device__ __forceinline__ float hcw_gate(float logit) {
    const float gate_logit = hcw_bf16f(logit);
    const float gate_scaled = hcw_bf16f(gate_logit / 4.0f);
    return hcw_bf16f(2.0f * hcw_bf16f(1.0f / (1.0f + expf(-gate_scaled))));
}
// BF16 bits of (0.0f + r[i]), element i in half (i & 1) of word (i >> 1).
static __device__ __forceinline__ uint4 hcw_pack8(const float (&r)[8]) {
    unsigned int w[4];
#pragma unroll
    for (int p = 0; p < 4; p++)
        w[p] = hcw_bf16bits(0.0f + r[2 * p]) | (hcw_bf16bits(0.0f + r[2 * p + 1]) << 16);
    return make_uint4(w[0], w[1], w[2], w[3]);
}
static __device__ __forceinline__ void hcw_mirror8(const float (&r)[8], int t, int c0, int H,
                                                   float* mirror) {
    float* o = mirror + (long long)t * H + c0;
    *(float4*)o = make_float4(0.0f + r[0], 0.0f + r[1], 0.0f + r[2], 0.0f + r[3]);
    *(float4*)(o + 4) = make_float4(0.0f + r[4], 0.0f + r[5], 0.0f + r[6], 0.0f + r[7]);
}
static __device__ __forceinline__ void hcw_apply8(const uint4 mq, int t, int c0, int H,
                                                  void* streams, const float* gates,
                                                  int bf16_state) {
    const float4 gv = *(const float4*)(gates + (long long)t * 4);
    // All four stream runs are requested before any arithmetic consumes them.
    uint4 sw[4];
    float4 sf[4][2];
    if (bf16_state) {
        const unsigned short* base = (const unsigned short*)streams + (long long)t * 4 * H + c0;
#pragma unroll
        for (int br = 0; br < 4; br++) sw[br] = *(const uint4*)(base + (long long)br * H);
    } else {
        const float* base = (const float*)streams + (long long)t * 4 * H + c0;
#pragma unroll
        for (int br = 0; br < 4; br++) {
            sf[br][0] = *(const float4*)(base + (long long)br * H);
            sf[br][1] = *(const float4*)(base + (long long)br * H + 4);
        }
    }
    const unsigned int mw[4] = {mq.x, mq.y, mq.z, mq.w};
    float mr[8];
#pragma unroll
    for (int p = 0; p < 4; p++) {
        mr[2 * p] = __uint_as_float(mw[p] << 16);
        mr[2 * p + 1] = __uint_as_float(mw[p] & 0xFFFF0000u);
    }
    const float g[4] = {hcw_gate(gv.x), hcw_gate(gv.y), hcw_gate(gv.z), hcw_gate(gv.w)};
    if (bf16_state) {
        unsigned short* base = (unsigned short*)streams + (long long)t * 4 * H + c0;
#pragma unroll
        for (int br = 0; br < 4; br++) {
            const unsigned int in[4] = {sw[br].x, sw[br].y, sw[br].z, sw[br].w};
            unsigned int out[4];
#pragma unroll
            for (int p = 0; p < 4; p++) {
                const float lo = hcw_add(__uint_as_float(in[p] << 16), hcw_bf16f(hcw_mul(g[br], mr[2 * p])));
                const float hi = hcw_add(__uint_as_float(in[p] & 0xFFFF0000u), hcw_bf16f(hcw_mul(g[br], mr[2 * p + 1])));
                out[p] = hcw_bf16bits(lo) | (hcw_bf16bits(hi) << 16);
            }
            *(uint4*)(base + (long long)br * H) = make_uint4(out[0], out[1], out[2], out[3]);
        }
    } else {
        float* base = (float*)streams + (long long)t * 4 * H + c0;
#pragma unroll
        for (int br = 0; br < 4; br++) {
            const float v[8] = {sf[br][0].x, sf[br][0].y, sf[br][0].z, sf[br][0].w,
                                sf[br][1].x, sf[br][1].y, sf[br][1].z, sf[br][1].w};
            float o[8];
#pragma unroll
            for (int i = 0; i < 8; i++)
                o[i] = hcw_bf16f(hcw_add(hcw_bf16f(hcw_mul(g[br], mr[i])), hcw_bf16f(v[i])));
            *(float4*)(base + (long long)br * H) = make_float4(o[0], o[1], o[2], o[3]);
            *(float4*)(base + (long long)br * H + 4) = make_float4(o[4], o[5], o[6], o[7]);
        }
    }
}
#endif  // HCW_EPILOGUE_BLOCK
// HC epilogue glue.  The HCW entries run four independent 16-row waves per block
// (grid.x = ceil(M/64)) so a block owns 64 contiguous channels: stage 1 parks the
// BF16 bits of each valid (token, 8 rows) accumulator lane in LDS [token][64 + 8];
// stage 2 walks (token, 8-channel) items with 8 consecutive lanes per token, i.e.
// 128 B (BF16 streams) / 256 B (F32 streams) contiguous per token and branch.  A
// wave whose 16 rows are past M still runs the K loop on clamped rows and only
// skips stage 1 (no early return before the block barrier).
#define RES_BT_PRO_PLAIN(BV) const int rs=blockIdx.x*16, bs=blockIdx.y*(16*BV), tid=threadIdx.x; if(rs>=M||bs>=N) return;
#define RES_BT_PRO_HCW4(BV) const int hw_wave=threadIdx.x>>5; const int rs=(blockIdx.x*4+hw_wave)*16, bs=blockIdx.y*(16*BV), tid=threadIdx.x&31; if(bs>=N) return;
#define RES_BT_EPI_HCW4(BV) { __shared__ __attribute__((aligned(16))) unsigned short hcw_stage[(BV)*16*72]; \
  _Pragma("unroll") for(int b=0;b<BV;b++){ const int oc=bs+b*16+ml; const int orow=rs+8*kg; if(oc<N && orow<M){ float r[8]; _Pragma("unroll") for(int j=0;j<8;j++) r[j]=acc[b][j]; if(Y) hcw_mirror8(r,oc,orow,M,Y); *(uint4*)(hcw_stage+(b*16+ml)*72+hw_wave*16+8*kg)=hcw_pack8(r); } } \
  __syncthreads(); \
  _Pragma("unroll 2") for(int it=0;it<BV;it++){ const int item=threadIdx.x+it*128; const int tl=item>>3, c8=item&7; const int t=bs+tl, c0=blockIdx.x*64+c8*8; if(t<N && c0<M) hcw_apply8(*(const uint4*)(hcw_stage+tl*72+c8*8),t,c0,M,hc_streams,hc_gates,hc_bf16); } }
#define GEN_RES_BT_X(NAME, BV, XP, EPI, LB, PRO) \
extern "C" __launch_bounds__(LB) __global__ void NAME(const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, int M,int K,int N XP){ \
    PRO(BV) const int ml=tid&15, kg=tid>>4; const int sr=(rs+ml<M)?(rs+ml):(M-1); const int gpr=K/256; const char* rb=A+(long long)sr*gpr*200; int sb[BV]; for(int b=0;b<BV;b++){int t=bs+b*16+ml; sb[b]=(t<N)?t:0;} float8_t acc[BV]; for(int b=0;b<BV;b++) acc[b]=float8_t{0,0,0,0,0,0,0,0}; for(int g=0;g<gpr;g++){const char* gp=rb+g*200; MQ6_HDR_LOAD(gp); const _Float16* xg[BV]; for(int b=0;b<BV;b++) xg[b]=X+(long long)sb[b]*K+g*256; for(int kt=0;kt<16;kt+=4){ unsigned int d0a=*(const unsigned int*)(gp+8+(kt+0)*12+kg*6), d1a=*(const unsigned int*)(gp+8+(kt+0)*12+kg*6+3); unsigned int d0b=*(const unsigned int*)(gp+8+(kt+1)*12+kg*6), d1b=*(const unsigned int*)(gp+8+(kt+1)*12+kg*6+3); unsigned int d0c=*(const unsigned int*)(gp+8+(kt+2)*12+kg*6), d1c=*(const unsigned int*)(gp+8+(kt+2)*12+kg*6+3); unsigned int d0d=*(const unsigned int*)(gp+8+(kt+3)*12+kg*6), d1d=*(const unsigned int*)(gp+8+(kt+3)*12+kg*6+3); _Pragma("unroll") for(int ki=0;ki<4;ki++){ MQ6_HDR_SEL half8_t a; if(ki==0) MQ6_DQ6_8(d0a,d1a); else if(ki==1) MQ6_DQ6_8(d0b,d1b); else if(ki==2) MQ6_DQ6_8(d0c,d1c); else MQ6_DQ6_8(d0d,d1d); _Pragma("unroll") for(int b=0;b<BV;b++){ half8_t bb=*(const half8_t*)(xg[b]+(kt+ki)*16+kg*8); acc[b]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,bb,acc[b]); } } } } EPI(BV) \
}
#define GEN_RES_BT(BV) GEN_RES_BT_X(gemm_mq6g256v2_residual_wmma_gfx12_bt##BV, BV, , RES_BT_EPI_PLAIN, 32 COMMA 2, RES_BT_PRO_PLAIN)
#define COMMA ,
GEN_RES_BT(4)
GEN_RES_BT(8)
GEN_RES_BT(12)
// Y = W·X applied to the HC streams in place (no memset, no residual Y).  Block 128
// (four 16-row waves), grid [ceil(M/64), ceil(N/(16*BV))].
#define GEN_RES_BT_HCW(BV) GEN_RES_BT_X(gemm_mq6g256v2_wmma_gfx12_bt##BV##_hcw, BV, COMMA_HCW, RES_BT_EPI_HCW4, 128 COMMA 1, RES_BT_PRO_HCW4)
#define COMMA_HCW , void* hc_streams, const float* hc_gates, int hc_bf16
GEN_RES_BT_HCW(4)
GEN_RES_BT_HCW(8)
GEN_RES_BT_HCW(12)
