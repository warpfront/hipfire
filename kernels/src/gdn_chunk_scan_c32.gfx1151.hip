// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// gfx1151 C32 V-split GDN chunk scan (grid (2,48), 256 threads).
// Eight waves: token tile mt=w>>2, value tile nt=w&3 for U/O/D; waves with
// nt<2 also form the 32x32 QK score.  State tiles keep the C64 split mapping.
// Requires C32 A (per-chunk 32x32 inverses) and G prefixes reset every 32 rows.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define GS_C 32
#define GS_D 128
#define GS_BV 64
#define GS_KP 132
#define GS_SP 36
#define GS_DP 36

typedef _Float16 gs_bfrag __attribute__((ext_vector_type(16)));
typedef float gs_f8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float gs_bf(unsigned short x){return __half2float(__ushort_as_half(x));}
__device__ __forceinline__ unsigned short gs_b16(float x){return __half_as_ushort(__float2half_rn(x));}
__device__ __forceinline__ gs_f8 gs_z8(){return (gs_f8){0,0,0,0,0,0,0,0};}
__device__ __forceinline__ gs_f8 gs_mma(gs_bfrag a,gs_bfrag b,gs_f8 c){return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c);}
__device__ __forceinline__ int gs_acc_row(int lane,int elem){return 2*elem+(lane>>4);}
__device__ __forceinline__ gs_bfrag gs_frag(const unsigned short*p,int pitch,int row0,int k0,int lane){union{unsigned short h[16];gs_bfrag v;}u;int r=row0+(lane&15);
#pragma unroll
 for(int e=0;e<16;++e)u.h[e]=p[r*pitch+k0+e];
 return u.v;}
__device__ __forceinline__ gs_bfrag gs_frag_t(const unsigned short*p,int pitch,int row0,int k0,int lane){union{unsigned short h[16];gs_bfrag v;}u;int r=row0+(lane&15);
#pragma unroll
 for(int e=0;e<16;++e)u.h[e]=p[(k0+e)*pitch+r];
 return u.v;}
__device__ __forceinline__ gs_bfrag gs_frag_g(const unsigned short*p,unsigned stride,int row0,int last,int k0,int lane){union{unsigned short h[16];gs_bfrag v;}u;int r=row0+(lane&15);if(r>last)r=last;
#pragma unroll
 for(int e=0;e<16;++e)u.h[e]=p[(unsigned)r*stride+k0+e];
 return u.v;}
#define GS_BAR() do{__builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");}while(0)

struct __attribute__((aligned(16))) GsLdsC32{
 unsigned short score[GS_C][GS_SP];
 unsigned short state[GS_BV][GS_KP];
 unsigned short d[GS_BV][GS_DP];
 float ctrl[4][GS_C];
};

extern "C" __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(10,10)))
__global__ void gdn_chunk_scan_c32(
 const unsigned short* __restrict__ q,const unsigned short* __restrict__ k,const unsigned short* __restrict__ v,
 const unsigned short* __restrict__ A,const float* __restrict__ G,const float* __restrict__ beta,
 signed char* __restrict__ sq,float* __restrict__ sc,_Float16* __restrict__ ef,float* __restrict__ out,int row0,int T
#ifdef GS_LAYER
 ,int seg
#endif
 ){
#ifdef GS_LAYER
 if(T<=0||seg<GS_C||seg>512||(seg%GS_C)!=0)return;
#else
 if(T<=0||T>512)return;
 const int seg=T;
#endif
 const int head=(int)blockIdx.y,half=(int)blockIdx.x;
 if(half>=2||head>=48)return;
 const int tid=(int)threadIdx.x,wave=tid>>5,lane=tid&31,lo=lane&15,hi=lane>>4;
 const int vbase=half*GS_BV,keyhead=head/3;
 const int kt0=(wave>>1)*2,vt0=(wave&1)*2;
 const int mt=wave>>2,nt=wave&3;
 __shared__ GsLdsC32 shared;
 GsLdsC32& sm=shared;

 gs_f8 St[4];
#pragma unroll
 for(int ki=0;ki<2;++ki)for(int vi=0;vi<2;++vi){const int t=ki*2+vi;const int vv=vbase+(vt0+vi)*16+lo;
#pragma unroll
  for(int e=0;e<8;++e){const int key=(kt0+ki)*16+gs_acc_row(lane,e);
   const unsigned long long off=((unsigned long long)head*GS_D+vv)*GS_D+key;
   St[t][e]=(float)sq[off]*sc[head*GS_D+vv]+(float)ef[off];
  }}
 float* red=reinterpret_cast<float*>(&sm.state[0][0]);
 for(int s0=0;s0<T;s0+=seg){
 int segT=T-s0;if(segT>seg)segT=seg;
 const int nch=(segT+GS_C-1)/GS_C;
 for(int chunk=0;chunk<nch;++chunk){
  const int local0=chunk*GS_C,parent0=row0+s0+local0;int rows=segT-local0;if(rows>GS_C)rows=GS_C;const int last=rows-1;
  const float gl=G[(unsigned long long)(parent0+last)*48+head];
  const unsigned short*kbase=k+((unsigned long long)parent0*16+keyhead)*GS_D;
  const unsigned short*qbase=q+((unsigned long long)parent0*16+keyhead)*GS_D;
  if(tid<GS_C){if(tid<rows){const unsigned long long go=(unsigned long long)(parent0+tid)*48+head;const float g=G[go];sm.ctrl[0][tid]=expf(g);sm.ctrl[1][tid]=expf(gl-g);sm.ctrl[2][tid]=g;sm.ctrl[3][tid]=beta[go];}else{sm.ctrl[0][tid]=0;sm.ctrl[1][tid]=0;sm.ctrl[2][tid]=0;sm.ctrl[3][tid]=0;}}
#pragma unroll
  for(int ki=0;ki<2;++ki)for(int vi=0;vi<2;++vi){int t=ki*2+vi;int vv=(vt0+vi)*16+lo;
   for(int e=0;e<8;++e)sm.state[vv][(kt0+ki)*16+gs_acc_row(lane,e)]=gs_b16(St[t][e]);}
  for(int x=tid;x<GS_C*GS_BV;x+=256){int r=x/GS_BV,c=x-r*GS_BV,rr=r<rows?r:last;sm.d[c][r]=v[((unsigned long long)(parent0+rr)*48+head)*GS_D+vbase+c];}
  GS_BAR();

  gs_f8 O=gs_z8(),U=gs_z8(),P=gs_z8();
#pragma unroll
  for(int kk=0;kk<8;++kk){gs_bfrag km=gs_frag_g(kbase,16*GS_D,mt*16,last,kk*16,lane);gs_bfrag qq=gs_frag_g(qbase,16*GS_D,mt*16,last,kk*16,lane);
   gs_bfrag st=gs_frag(&sm.state[0][0],GS_KP,nt*16,kk*16,lane);U=gs_mma(km,st,U);O=gs_mma(st,qq,O);
   if(nt<2){gs_bfrag kn=gs_frag_g(kbase,16*GS_D,nt*16,last,kk*16,lane);P=gs_mma(kn,qq,P);}}
  const float oge=sm.ctrl[0][mt*16+lo];
#pragma unroll
  for(int e=0;e<8;++e)O[e]*=oge;
  if(nt<2){const int i=mt*16+lo;
#pragma unroll
   for(int e=0;e<8;++e){const int j0e=nt*16+gs_acc_row(lane,e);sm.score[i][j0e]=gs_b16((j0e<=i&&i<rows)?expf(sm.ctrl[2][i]-sm.ctrl[2][j0e])*P[e]:0.0f);}}
  GS_BAR();
#pragma unroll
  for(int e=0;e<8;++e){int tok=mt*16+gs_acc_row(lane,e);int vv=nt*16+lo;float z=sm.ctrl[3][tok]*(gs_bf(sm.d[vv][tok])-sm.ctrl[0][tok]*U[e]);sm.state[vv][tok]=gs_b16(z);}
  GS_BAR();

  const unsigned short*abase=A+(unsigned long long)(s0+local0)*48*GS_C+(unsigned long long)head*GS_C;
  gs_f8 Dacc=gs_z8();
#pragma unroll
  for(int kk=0;kk<GS_C/16;++kk)Dacc=gs_mma(gs_frag_g(abase,48*GS_C,mt*16,last,kk*16,lane),gs_frag(&sm.state[0][0],GS_KP,nt*16,kk*16,lane),Dacc);
#pragma unroll
  for(int e=0;e<8;++e){int tok=mt*16+gs_acc_row(lane,e);sm.d[nt*16+lo][tok]=gs_b16(Dacc[e]);}
  GS_BAR();

#pragma unroll
  for(int kk=0;kk<GS_C/16;++kk)O=gs_mma(gs_frag(&sm.d[0][0],GS_DP,nt*16,kk*16,lane),gs_frag(&sm.score[0][0],GS_SP,mt*16,kk*16,lane),O);
  {const int tok=mt*16+lo;
   if(tok<rows){
#pragma unroll
    for(int e=0;e<8;++e){int vv=vbase+nt*16+gs_acc_row(lane,e);out[((unsigned long long)(parent0+tok)*48+head)*GS_D+vv]=O[e];}}}
  GS_BAR();
#pragma unroll
  for(int e=0;e<8;++e){int t=mt*16+gs_acc_row(lane,e),vv=nt*16+lo;sm.d[vv][t]=gs_b16(sm.ctrl[1][t]*gs_bf(sm.d[vv][t]));}
  for(int x=tid;x<GS_C*GS_D;x+=256){int r=x/GS_D,c=x-r*GS_D,rr=r<rows?r:last;sm.state[r][c]=kbase[(unsigned)rr*16*GS_D+c];}
  GS_BAR();

  const float ec=expf(gl);
#pragma unroll
  for(int t=0;t<4;++t)for(int e=0;e<8;++e)St[t][e]*=ec;
#pragma unroll
  for(int kk=0;kk<GS_C/16;++kk){gs_bfrag dv[2]={gs_frag(&sm.d[0][0],GS_DP,(vt0+0)*16,kk*16,lane),gs_frag(&sm.d[0][0],GS_DP,(vt0+1)*16,kk*16,lane)};
#pragma unroll
   for(int ki=0;ki<2;++ki){gs_bfrag kt=gs_frag_t(&sm.state[0][0],GS_KP,(kt0+ki)*16,kk*16,lane);
#pragma unroll
    for(int vi=0;vi<2;++vi)St[ki*2+vi]=gs_mma(kt,dv[vi],St[ki*2+vi]);}}
  GS_BAR();
 }

 float vmax[2]={0,0};
#pragma unroll
 for(int vi=0;vi<2;++vi){int vv=vbase+(vt0+vi)*16+lo;
#pragma unroll
  for(int ki=0;ki<2;++ki){int t=ki*2+vi;
#pragma unroll
   for(int e=0;e<8;++e)vmax[vi]=fmaxf(vmax[vi],fabsf(St[t][e]));}
  vmax[vi]=fmaxf(vmax[vi],__shfl_xor(vmax[vi],16));if(hi==0)red[(wave>>1)*GS_BV+vv-vbase]=vmax[vi];}
 GS_BAR();
 const bool final_seg=s0+seg>=T;
#pragma unroll
 for(int vi=0;vi<2;++vi){int vv=vbase+(vt0+vi)*16+lo;float mx=0;
#pragma unroll
  for(int g=0;g<4;++g)mx=fmaxf(mx,red[g*GS_BV+vv-vbase]);float inv=mx>0?127.0f/mx:0.0f,scale=mx>0?mx/127.0f:1.0f;
  if(final_seg){
   if(wave<2&&hi==0)sc[head*GS_D+vv]=scale;
#pragma unroll
   for(int ki=0;ki<2;++ki){int t=ki*2+vi;
#pragma unroll
    for(int e=0;e<8;++e){const unsigned long long off=((unsigned long long)head*GS_D+vv)*GS_D+(kt0+ki)*16+gs_acc_row(lane,e);float qf=fminf(fmaxf(rintf(St[t][e]*inv),-128.0f),127.0f);sq[off]=(signed char)qf;ef[off]=(_Float16)(St[t][e]-qf*scale);}}
  }else{
#pragma unroll
   for(int ki=0;ki<2;++ki){int t=ki*2+vi;
#pragma unroll
    for(int e=0;e<8;++e){float qf=fminf(fmaxf(rintf(St[t][e]*inv),-128.0f),127.0f);_Float16 eh=(_Float16)(St[t][e]-qf*scale);St[t][e]=(float)(signed char)qf*scale+(float)eh;}}
  }}
 GS_BAR();
 }
}
