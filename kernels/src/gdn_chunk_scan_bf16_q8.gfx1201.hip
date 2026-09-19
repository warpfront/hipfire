// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define GS_C 64
#define GS_D 128
#define GS_BV 64
#define GS_KP 132
#define GS_AP 68

typedef _Float16 gs_b8 __attribute__((ext_vector_type(8)));
typedef float gs_f8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float gs_bf(unsigned short x){return __half2float(__ushort_as_half(x));}
__device__ __forceinline__ unsigned short gs_b16(float x){return __half_as_ushort(__float2half_rn(x));}
__device__ __forceinline__ gs_f8 gs_z8(){return (gs_f8){0,0,0,0,0,0,0,0};}
__device__ __forceinline__ gs_f8 gs_mma(gs_b8 a,gs_b8 b,gs_f8 c){return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c);}
__device__ __forceinline__ gs_b8 gs_frag(const unsigned short*p,int pitch,int row0,int k0,int lane){union{unsigned short h[8];gs_b8 v;}u;int r=row0+(lane&15),k=k0+4*(lane>>4);
#pragma unroll
 for(int e=0;e<4;++e){u.h[e]=p[r*pitch+k+e];u.h[e+4]=p[r*pitch+k+8+e];}return u.v;}
__device__ __forceinline__ gs_b8 gs_frag_t(const unsigned short*p,int pitch,int row0,int k0,int lane){union{unsigned short h[8];gs_b8 v;}u;int r=row0+(lane&15),k=k0+4*(lane>>4);
#pragma unroll
 for(int e=0;e<4;++e){u.h[e]=p[(k+e)*pitch+r];u.h[e+4]=p[(k+8+e)*pitch+r];}return u.v;}
__device__ __forceinline__ gs_b8 gs_frag_g(const unsigned short*p,unsigned stride,int row0,int last,int k0,int lane){union{unsigned short h[8];gs_b8 v;}u;int r=row0+(lane&15);if(r>last)r=last;int k=k0+4*(lane>>4);
#pragma unroll
 for(int e=0;e<4;++e){u.h[e]=p[(unsigned)r*stride+k+e];u.h[e+4]=p[(unsigned)r*stride+k+8+e];}return u.v;}
#define GS_BAR() do{__builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");}while(0)

struct __attribute__((aligned(16))) GsLds{
 unsigned short k[GS_C][GS_KP];
 unsigned short score[GS_C][GS_AP];
 unsigned short state[GS_BV][GS_KP];
 unsigned short w[GS_BV][GS_AP];
 unsigned short d[GS_BV][GS_AP];
 float ctrl[4][GS_C];
};

#ifdef GDN_R4D_F32_EDGE
extern "C" __launch_bounds__(256)
__global__ void gdn_chunk_scan_bf16_f32_edge_gfx1201(
 const unsigned short* __restrict__ q,const unsigned short* __restrict__ k,const unsigned short* __restrict__ v,
 const unsigned short* __restrict__ A,const float* __restrict__ G,const float* __restrict__ beta,
 const float* __restrict__ h0,float* __restrict__ ht,float* __restrict__ out,int row0,int T){
#else
extern "C" __launch_bounds__(256)
__global__ void gdn_chunk_scan_bf16_q8_gfx1201(
 const unsigned short* __restrict__ q,const unsigned short* __restrict__ k,const unsigned short* __restrict__ v,
 const unsigned short* __restrict__ A,const float* __restrict__ G,const float* __restrict__ beta,
 signed char* __restrict__ sq,float* __restrict__ sc,_Float16* __restrict__ ef,float* __restrict__ out,int row0,int T){
#endif
 if(T<=0||T>512)return;
 const int half=(int)blockIdx.x,head=(int)blockIdx.y;
 if(half>=2||head>=48)return;
 const int tid=(int)threadIdx.x,wave=tid>>5,lane=tid&31,lo=lane&15,hi=lane>>4;
 const int vbase=half*GS_BV,keyhead=head/3;
 const int kt0=(wave>>1)*2,vt0=(wave&1)*2;
 __shared__ GsLds shared;
 GsLds& sm=shared;

 gs_f8 St[4];
#pragma unroll
 for(int ki=0;ki<2;++ki)for(int vi=0;vi<2;++vi){const int t=ki*2+vi;const int kval=(kt0+ki)*16+8*hi;const int vv=vbase+(vt0+vi)*16+lo;
#pragma unroll
  for(int e=0;e<8;++e){const int key=kval+e;
#ifdef GDN_R4D_F32_EDGE
   St[t][e]=h0[((unsigned long long)head*GS_D+vv)*GS_D+key];
#else
   const unsigned long long off=((unsigned long long)head*GS_D+vv)*GS_D+key;
   St[t][e]=(float)sq[off]*sc[head*GS_D+vv];
#endif
  }}
 auto stage_state=[&](){
#pragma unroll
  for(int ki=0;ki<2;++ki)for(int vi=0;vi<2;++vi){int t=ki*2+vi;int kval=(kt0+ki)*16+8*hi;int vv=(vt0+vi)*16+lo;
   for(int e=0;e<8;++e)sm.state[vv][kval+e]=gs_b16(St[t][e]);}
 };
 stage_state();GS_BAR();
 const int nch=(T+GS_C-1)/GS_C;
 for(int chunk=0;chunk<nch;++chunk){
  const int local0=chunk*GS_C,parent0=row0+local0;int rows=T-local0;if(rows>GS_C)rows=GS_C;const int last=rows-1;
  const float gl=G[(unsigned long long)(parent0+last)*48+head];
  const float mid=.5f*(G[(unsigned long long)parent0*48+head]+gl);
  for(int x=tid;x<GS_C*GS_D;x+=256){int r=x/GS_D,c=x-r*GS_D,rr=r<rows?r:last;sm.k[r][c]=k[((unsigned long long)(parent0+rr)*16+keyhead)*GS_D+c];}
  for(int x=tid;x<GS_C*GS_BV;x+=256){int r=x/GS_BV,c=x-r*GS_BV,rr=r<rows?r:last;sm.d[c][r]=v[((unsigned long long)(parent0+rr)*48+head)*GS_D+vbase+c];}
  if(tid<GS_C){if(tid<rows){const unsigned long long go=(unsigned long long)(parent0+tid)*48+head;const float g=G[go];sm.ctrl[0][tid]=expf(g);sm.ctrl[1][tid]=expf(mid-g);sm.ctrl[2][tid]=expf(g-mid);sm.ctrl[3][tid]=beta[go];}else{sm.ctrl[0][tid]=0;sm.ctrl[1][tid]=0;sm.ctrl[2][tid]=0;sm.ctrl[3][tid]=0;}}
  GS_BAR();

  const int mt=wave>>1,ntb=(wave&1)*2;
  gs_f8 O[2]={gs_z8(),gs_z8()},U[2]={gs_z8(),gs_z8()},P[2]={gs_z8(),gs_z8()};
  const unsigned short*qbase=q+((unsigned long long)parent0*16+keyhead)*GS_D;
#pragma unroll
  for(int kk=0;kk<8;++kk){gs_b8 km=gs_frag(&sm.k[0][0],GS_KP,mt*16,kk*16,lane);gs_b8 qq=gs_frag_g(qbase,16*GS_D,mt*16,last,kk*16,lane);
   for(int j=0;j<2;++j){gs_b8 st=gs_frag(&sm.state[0][0],GS_KP,(ntb+j)*16,kk*16,lane);gs_b8 kn=gs_frag(&sm.k[0][0],GS_KP,(ntb+j)*16,kk*16,lane);U[j]=gs_mma(km,st,U[j]);O[j]=gs_mma(st,qq,O[j]);P[j]=gs_mma(kn,qq,P[j]);}}
  const float oge=sm.ctrl[0][mt*16+lo];
#pragma unroll
  for(int j=0;j<2;++j)for(int e=0;e<8;++e)O[j][e]*=oge;
#pragma unroll
  for(int j=0;j<2;++j){
#pragma unroll
   for(int e=0;e<8;++e){int tok=mt*16+8*hi+e;int vv=(ntb+j)*16+lo;float z=sm.ctrl[3][tok]*(gs_bf(sm.d[vv][tok])-sm.ctrl[0][tok]*U[j][e]);sm.w[vv][tok]=gs_b16(z);}
   const int i=mt*16+lo,j0=(ntb+j)*16+8*hi;
#pragma unroll
   for(int e=0;e<8;++e)sm.score[i][j0+e]=gs_b16((j0+e<=i&&i<rows)?sm.ctrl[2][i]*P[j][e]:0.0f);
  }
  GS_BAR();

  const unsigned short*abase=A+(unsigned long long)local0*48*GS_C+(unsigned long long)head*GS_C;
  gs_f8 Dacc[2]={gs_z8(),gs_z8()};
#pragma unroll
  for(int kk=0;kk<4;++kk){gs_b8 aa=gs_frag_g(abase,48*GS_C,mt*16,last,kk*16,lane);
#pragma unroll
   for(int j=0;j<2;++j)Dacc[j]=gs_mma(aa,gs_frag(&sm.w[0][0],GS_AP,(ntb+j)*16,kk*16,lane),Dacc[j]);}
#pragma unroll
  for(int j=0;j<2;++j)for(int e=0;e<8;++e){int tok=mt*16+8*hi+e;sm.d[(ntb+j)*16+lo][tok]=gs_b16(sm.ctrl[1][tok]*Dacc[j][e]);}
  GS_BAR();

#pragma unroll
  for(int kk=0;kk<4;++kk){gs_b8 score=gs_frag(&sm.score[0][0],GS_AP,mt*16,kk*16,lane);
#pragma unroll
   for(int j=0;j<2;++j)O[j]=gs_mma(gs_frag(&sm.d[0][0],GS_AP,(ntb+j)*16,kk*16,lane),score,O[j]);}
  const int tok=mt*16+lo;
  if(tok<rows){
#pragma unroll
   for(int j=0;j<2;++j)for(int e=0;e<8;++e){int vv=vbase+(ntb+j)*16+8*hi+e;out[((unsigned long long)(parent0+tok)*48+head)*GS_D+vv]=O[j][e];}}

  const float ec=expf(mid),lam=expf(gl-mid);
#pragma unroll
  for(int t=0;t<4;++t)for(int e=0;e<8;++e)St[t][e]*=ec;
#pragma unroll
  for(int kk=0;kk<4;++kk){gs_b8 dv[2]={gs_frag(&sm.d[0][0],GS_AP,(vt0+0)*16,kk*16,lane),gs_frag(&sm.d[0][0],GS_AP,(vt0+1)*16,kk*16,lane)};
#pragma unroll
   for(int ki=0;ki<2;++ki){gs_b8 kt=gs_frag_t(&sm.k[0][0],GS_KP,(kt0+ki)*16,kk*16,lane);
#pragma unroll
    for(int vi=0;vi<2;++vi)St[ki*2+vi]=gs_mma(kt,dv[vi],St[ki*2+vi]);}}
#pragma unroll
  for(int t=0;t<4;++t)for(int e=0;e<8;++e)St[t][e]*=lam;
  stage_state();GS_BAR();
 }

#ifdef GDN_R4D_F32_EDGE
#pragma unroll
 for(int ki=0;ki<2;++ki)for(int vi=0;vi<2;++vi){int t=ki*2+vi,kval=(kt0+ki)*16+8*hi,vv=vbase+(vt0+vi)*16+lo;
#pragma unroll
  for(int e=0;e<8;++e)ht[((unsigned long long)head*GS_D+vv)*GS_D+kval+e]=St[t][e];}
#else
 // Fold the old EF exactly once, reduce a full 128-key absmax per value row,
 // then let every state owner write its disjoint code/residual elements.
 float vmax[2]={0,0};
#pragma unroll
 for(int vi=0;vi<2;++vi){int vv=vbase+(vt0+vi)*16+lo;
#pragma unroll
  for(int ki=0;ki<2;++ki){int t=ki*2+vi,kval=(kt0+ki)*16+8*hi;
#pragma unroll
   for(int e=0;e<8;++e){const unsigned long long off=((unsigned long long)head*GS_D+vv)*GS_D+kval+e;St[t][e]+=(float)ef[off];vmax[vi]=fmaxf(vmax[vi],fabsf(St[t][e]));}}
  vmax[vi]=fmaxf(vmax[vi],__shfl_xor(vmax[vi],16));if(hi==0)sm.ctrl[wave>>1][vv-vbase]=vmax[vi];}
 GS_BAR();
#pragma unroll
 for(int vi=0;vi<2;++vi){int vv=vbase+(vt0+vi)*16+lo;float mx=0;
#pragma unroll
  for(int g=0;g<4;++g)mx=fmaxf(mx,sm.ctrl[g][vv-vbase]);float inv=mx>0?127.0f/mx:0.0f,scale=mx>0?mx/127.0f:1.0f;if(wave<2&&hi==0)sc[head*GS_D+vv]=scale;
#pragma unroll
  for(int ki=0;ki<2;++ki){int t=ki*2+vi,kval=(kt0+ki)*16+8*hi;
#pragma unroll
   for(int e=0;e<8;++e){const unsigned long long off=((unsigned long long)head*GS_D+vv)*GS_D+kval+e;float qf=fminf(fmaxf(rintf(St[t][e]*inv),-128.0f),127.0f);sq[off]=(signed char)qf;ef[off]=(_Float16)(St[t][e]-qf*scale);}}}
#endif
}
