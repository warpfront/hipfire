// SPDX-License-Identifier: MIT
// gfx1201 MQ6 X-LDS port: ascending K tiles and the baseline F16 sc*q+zp.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
#define MQ6_DQ6(i,dw,sh) a[i]=sc*(_Float16)(float)(((dw)>>(sh))&63u)+zp
#define MQ6_DQ6_8(d0,d1) do { MQ6_DQ6(0,(d0),0); MQ6_DQ6(1,(d0),6); MQ6_DQ6(2,(d0),12); MQ6_DQ6(3,(d0),18); MQ6_DQ6(4,(d1),0); MQ6_DQ6(5,(d1),6); MQ6_DQ6(6,(d1),12); MQ6_DQ6(7,(d1),18); } while(0)

template<bool OVERWRITE, bool OUT_BF16=false>
__device__ __forceinline__ void mq6_xlds_gfx12(
 const char* __restrict__ A, const _Float16* __restrict__ X,
 float* __restrict__ Y, int M, int K, int N, int row_block) {
 if(M<=0 || K<=0 || N<=0) return;
 constexpr int BV=8, RW=4, CK=64, STRIDE=72, NT=128, VPT=8, TOK=128;
 __shared__ __attribute__((aligned(16))) _Float16 xs[TOK*STRIDE];
 const int tid=threadIdx.x, lane=tid&31, ml=lane&15, lane_half=lane>>4;
 const int row_start=(row_block*RW+(tid>>5))*16;
 const int batch_start=blockIdx.y*TOK;
 const int safe_row=(row_start+ml<M)?row_start+ml:M-1;
 const int groups_per_row=K/256;
 const __amdgpu_buffer_rsrc_t arsrc=__builtin_amdgcn_make_buffer_rsrc((void*)A,0,0x7FFFFFFF,0x31004000);
 const int row_off=safe_row*groups_per_row*200;
#define XL_FOR8(F) F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7)
#define XL_OFF(I) const int xo##I=((batch_start+I*(NT/VPT)+tid/VPT<N)?batch_start+I*(NT/VPT)+tid/VPT:0)*K+(tid%VPT)*8;
 XL_FOR8(XL_OFF)
#undef XL_OFF
 _Float16* lds0=xs+(tid/VPT)*STRIDE+(tid%VPT)*8;
#define XL_LOAD(I) st##I=*(const uint4*)(X+xo##I+cn*CK);
#define XL_STORE(I) *(uint4*)(lds0+I*(NT/VPT)*STRIDE)=st##I;
 // gfx12 uses separate LDS counters. Workgroup barriers must order LDS accesses.
#define XL_BARRIER() __syncthreads()
 uint4 st0,st1,st2,st3,st4,st5,st6,st7;
 { const int cn=0; XL_FOR8(XL_LOAD) }
 float8_t acc[BV];
#pragma unroll
 for(int b=0;b<BV;b++) acc[b]=float8_t{0,0,0,0,0,0,0,0};
 const int nchunks=K/CK;
 for(int g=0;g<groups_per_row;g++) {
  const int goff=g*200;
  const unsigned int hA=__builtin_amdgcn_raw_buffer_load_b32(arsrc,row_off,goff,0);
  const unsigned int hB=__builtin_amdgcn_raw_buffer_load_b32(arsrc,row_off+4,goff,0);
  const _Float16 sc0=(_Float16)__half2float(__ushort_as_half((unsigned short)hA));
  const _Float16 zp0=(_Float16)__half2float(__ushort_as_half((unsigned short)(hA>>16)));
  const _Float16 sc1=(_Float16)__half2float(__ushort_as_half((unsigned short)hB));
  const _Float16 zp1=(_Float16)__half2float(__ushort_as_half((unsigned short)(hB>>16)));
#pragma unroll
  for(int cq=0;cq<4;cq++) {
   const int c=g*4+cq;
   unsigned int d[12];
#pragma unroll
   for(int u=0;u<6;u++) {
    const auto v=__builtin_amdgcn_raw_buffer_load_b64(arsrc,row_off+8+u*8,goff+cq*48,0);
    d[2*u]=v[0]; d[2*u+1]=v[1];
   }
   XL_FOR8(XL_STORE)
   XL_BARRIER();
   const int cn=(c+1<nchunks)?c+1:c;
   XL_FOR8(XL_LOAD)
#pragma unroll
   for(int j=0;j<4;j++) {
    const int kt=cq*4+j;
    const _Float16 sc=(kt<8)?sc0:sc1, zp=(kt<8)?zp0:zp1;
    const unsigned int lo=lane_half?__builtin_amdgcn_alignbit(d[3*j+2],d[3*j+1],16):d[3*j];
    const unsigned int hi=lane_half?(d[3*j+2]>>16):(d[3*j+1]&0xFFFFu);
    // MQ6_DQ6_8 expects four codes per 24-bit word, like bt12.
    const unsigned int d0=lo, d1=(lo>>24)|(hi<<8);
    half8_t a;
    MQ6_DQ6_8(d0,d1);
#pragma unroll
    for(int b=0;b<BV;b++) {
     half8_t bb=*(const half8_t*)(xs+(b*16+ml)*STRIDE+j*16+lane_half*8);
     acc[b]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,bb,acc[b]);
    }
   }
#pragma unroll
   for(int b=0;b<BV;b++) asm volatile("" : "+v"(acc[b]));
   XL_BARRIER();
  }
 }
#pragma unroll
 for(int b=0;b<BV;b++) {
  const int oc=batch_start+b*16+ml, row0=row_start+8*lane_half;
  if(oc>=N || row0>=M) continue;
  if constexpr (OUT_BF16) {
   unsigned short* out=(unsigned short*)Y+(long long)oc*M+row0;
   unsigned short h[8];
#pragma unroll
   for(int j=0;j<8;j++) {
    const float v=0.0f+acc[b][j];
    const unsigned int u=__float_as_uint(v);
    h[j]=isfinite(v)?(unsigned short)((u+0x7FFFu+((u>>16)&1u))>>16):(unsigned short)(u>>16);
   }
   if(row0+8<=M && M%8==0) *(uint4*)out=__builtin_bit_cast(uint4,h);
   else {
#pragma unroll
    for(int j=0;j<8;j++) if(row0+j<M) out[j]=h[j];
   }
  } else {
   float* out=Y+(long long)oc*M+row0;
   if(row0+8<=M && M%8==0) {
    float4 o0=OVERWRITE?make_float4(0,0,0,0):*(const float4*)out;
    float4 o1=OVERWRITE?make_float4(0,0,0,0):*(const float4*)(out+4);
    o0.x+=acc[b][0];o0.y+=acc[b][1];o0.z+=acc[b][2];o0.w+=acc[b][3];
    o1.x+=acc[b][4];o1.y+=acc[b][5];o1.z+=acc[b][6];o1.w+=acc[b][7];
    *(float4*)out=o0;*(float4*)(out+4)=o1;
   } else {
#pragma unroll
    for(int j=0;j<8;j++) if(row0+j<M) out[j]=(OVERWRITE?0.0f:out[j])+acc[b][j];
   }
  }
 }
#undef XL_LOAD
#undef XL_STORE
#undef XL_BARRIER
#undef XL_FOR8
}

extern "C" __launch_bounds__(128,1) __global__ void gemm_mq6g256v2_residual_wmma_gfx12_bt8_x4(
 const char* __restrict__ A, const _Float16* __restrict__ X,
 float* __restrict__ Y, int M, int K, int N) {
 mq6_xlds_gfx12<false,false>(A,X,Y,M,K,N,blockIdx.x);
}

extern "C" __launch_bounds__(128,1) __global__ void gemm_mq6g256v2_wmma_gfx12_bt8_x4(
 const char* __restrict__ A, const _Float16* __restrict__ X,
 float* __restrict__ Y, int M, int K, int N) {
 mq6_xlds_gfx12<true,false>(A,X,Y,M,K,N,blockIdx.x);
}

extern "C" __launch_bounds__(128,1) __global__ void gemm_mq6g256v2_wmma_gfx12_bt8_x4_bf16out(
 const char* __restrict__ A, const _Float16* __restrict__ X,
 float* __restrict__ Y, int M, int K, int N) {
 mq6_xlds_gfx12<true,true>(A,X,Y,M,K,N,blockIdx.x);
}

// Independent output pitches; zero-row regions contribute no workgroups.
extern "C" __launch_bounds__(128,1) __global__ void
gemm_mq6g256v2_wmma_gfx12_bt8_x4_regions(
 const char* A0, const char* A1, const char* A2, const _Float16* X,
 float* Y0, float* Y1, float* Y2, int M0, int M1, int M2, int K, int N) {
 const int t0=(M0+63)/64, t1=(M1+63)/64;
 const int rb=blockIdx.x;
 if(rb<t0) mq6_xlds_gfx12<true>(A0,X,Y0,M0,K,N,rb);
 else if(rb<t0+t1) mq6_xlds_gfx12<true>(A1,X,Y1,M1,K,N,rb-t0);
 else mq6_xlds_gfx12<true>(A2,X,Y2,M2,K,N,rb-t0-t1);
}
