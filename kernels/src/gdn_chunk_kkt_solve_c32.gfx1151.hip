// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// gfx1151 C32 KKT solve, one value head per 64-thread block.
// Grid (ceil(T/32), 48).  Same Gram/solve/merge arithmetic as the C64 kernel's
// first 32-row pair merge, so the first half of every C64 chunk is identical.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define GK_C 32
#define GK_D 128
#define GK_PITCH 36
#define GK_B 16

typedef _Float16 gk_bfrag __attribute__((ext_vector_type(16)));
typedef float gk_f8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float gk_from_bf(unsigned short x){return __half2float(__ushort_as_half(x));}
__device__ __forceinline__ unsigned short gk_to_bf(float x){return __half_as_ushort(__float2half_rn(x));}
__device__ __forceinline__ gk_f8 gk_zero(){return (gk_f8){0,0,0,0,0,0,0,0};}
__device__ __forceinline__ gk_f8 gk_mma(gk_bfrag a,gk_bfrag b,gk_f8 c){return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c);}
__device__ __forceinline__ int gk_acc_row(int lane,int elem){return 2*elem+(lane>>4);}
__device__ __forceinline__ gk_bfrag gk_frag_shared(const unsigned short*p,int pitch,int row0,int k0,int lane){
 union{unsigned short h[16];gk_bfrag v;}u;const int row=row0+(lane&15);
#pragma unroll
 for(int e=0;e<16;++e)u.h[e]=p[row*pitch+k0+e];
 return u.v;}
__device__ __forceinline__ gk_bfrag gk_frag_transpose(const unsigned short*p,int pitch,int row0,int k0,int lane){
 union{unsigned short h[16];gk_bfrag v;}u;const int row=row0+(lane&15);
#pragma unroll
 for(int e=0;e<16;++e)u.h[e]=p[(k0+e)*pitch+row];
 return u.v;}
__device__ __forceinline__ gk_bfrag gk_frag_global(const unsigned short*p,unsigned row_stride,int row0,int last,int k0,int lane){
 union{unsigned short h[16];gk_bfrag v;}u;int row=row0+(lane&15);if(row>last)row=last;
#pragma unroll
 for(int e=0;e<16;++e)u.h[e]=p[(unsigned)row*row_stride+k0+e];
 return u.v;}

struct __attribute__((aligned(16))) GkLds32{
 unsigned short a[GK_C][GK_PITCH];
 unsigned short tmp[16][36];
 float gate[GK_C];
 float beta[GK_C];
};

extern "C" __launch_bounds__(64)
__global__ void gdn_chunk_kkt_solve_c32(const unsigned short* __restrict__ k,const float* __restrict__ G,const float* __restrict__ beta,
 unsigned short* __restrict__ A,int row0,int T){
 if(T<=0)return;
 const int chunk=(int)blockIdx.x,vh=(int)blockIdx.y,kh=vh/3;
 if(vh>=48||chunk*GK_C>=T)return;
 const int tid=(int)threadIdx.x,wave=tid>>5,lane=tid&31;
 const int local0=chunk*GK_C;
 int rows=T-local0;if(rows>GK_C)rows=GK_C;
 const int parent0=row0+local0;
 const unsigned kstride=16*GK_D;
 const unsigned short* kp=k+(unsigned long long)parent0*kstride+(unsigned)kh*GK_D;
 // Tiles: wave 0 owns (0,0) and (1,1); wave 1 owns (0,1).
 gk_f8 gram[2]={gk_zero(),gk_zero()};
#pragma unroll
 for(int slot=0;slot<2;++slot){
  const int tile=wave+slot*2; // 0,1,2 valid
  if(tile<3){const int rb=tile==2?1:0,cb=tile==0?0:1;
#pragma unroll
   for(int d=0;d<8;++d)gram[slot]=gk_mma(gk_frag_global(kp,kstride,rb*GK_B,rows-1,d*GK_B,lane),gk_frag_global(kp,kstride,cb*GK_B,rows-1,d*GK_B,lane),gram[slot]);}
 }
 __shared__ GkLds32 s;
 for(int x=tid;x<GK_C*GK_PITCH;x+=64)reinterpret_cast<unsigned short*>(s.a)[x]=0;
 if(tid<GK_C){if(tid<rows){const unsigned long long off=(unsigned long long)(parent0+tid)*48+vh;s.gate[tid]=G[off];s.beta[tid]=beta[off];}else{s.gate[tid]=0.0f;s.beta[tid]=0.0f;}}
 __syncthreads();
#pragma unroll
 for(int slot=0;slot<2;++slot){
  const int tile=wave+slot*2;
  if(tile<3){const int rb=tile==2?1:0,cb=tile==0?0:1;const int i=cb*GK_B+(lane&15);
#pragma unroll
   for(int e=0;e<8;++e){const int j=rb*GK_B+gk_acc_row(lane,e);float value=0.0f;
    if(i<rows&&j<i&&j<rows){const float exponent=s.gate[i]-s.gate[j];value=s.beta[i]*expf(exponent)*gram[slot][e];}
    s.a[i][j]=gk_to_bf(value);}}
 }
 __syncthreads();
 if(lane<16){
  const int base=wave*GK_B;float col[GK_B];
#pragma unroll
  for(int i=0;i<GK_B;++i){float value=(i==lane)?1.0f:0.0f;
#pragma unroll
   for(int j=0;j<GK_B;++j)if(j<i)value-=gk_from_bf(s.a[base+i][base+j])*col[j];
   col[i]=value;}
#pragma unroll
  for(int i=0;i<GK_B;++i)s.a[base+i][base+lane]=gk_to_bf(col[i]);
 }
 __syncthreads();
 if(wave==0){
  gk_f8 x=gk_zero();
  x=gk_mma(gk_frag_shared(&s.a[0][0],GK_PITCH,16,0,lane),gk_frag_transpose(&s.a[0][0],GK_PITCH,0,0,lane),x);
  const int xn=lane&15;
#pragma unroll
  for(int e=0;e<8;++e)s.tmp[xn][gk_acc_row(lane,e)]=gk_to_bf(x[e]);
  __builtin_amdgcn_wave_barrier();
  gk_f8 y=gk_zero();
  y=gk_mma(gk_frag_shared(&s.a[0][0],GK_PITCH,16,16,lane),gk_frag_shared(&s.tmp[0][0],36,0,0,lane),y);
  const int yc=lane&15;
#pragma unroll
  for(int e=0;e<8;++e)s.a[16+gk_acc_row(lane,e)][yc]=gk_to_bf(-y[e]);
 }
 __syncthreads();
 const unsigned long long abase=(unsigned long long)local0*48*GK_C+(unsigned long long)vh*GK_C;
 for(int x=tid;x<GK_C*GK_C;x+=64){const int r=x/GK_C,c=x-r*GK_C;A[abase+(unsigned long long)r*48*GK_C+c]=s.a[r][c];}
}
