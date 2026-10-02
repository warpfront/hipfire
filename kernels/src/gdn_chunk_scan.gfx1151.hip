// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Exact-gfx1151 twin of `gdn_chunk_scan` (gdn_chunk_scan.gfx1201.hip, f32
// `out`, one launch per <= 512-row commit segment): `gdn_chunk_scan_gfx1151`.
// Every LDS value, WMMA operand, per-accumulator order and rounding is the
// shipped kernel's, so `out` and the Q8/scale/EF state are byte-identical.
// Scheduling differences from the shipped gfx11 loop:
//  * the gfx1201 pipelined chunk loop: V and K tiles staged with one 16-byte
//    load per 8 halves, the next kk's Q/K fragments and the chunk's A
//    fragments in flight before the WMMAs that precede their use (tiles of the
//    K Q^T score above the causal diagonal are the same +0 without the
//    unobserved product);
//  * K is staged transposed ([key][token], pitch 66, in the dead z tile), so
//    the state update's K^T fragments are row reads instead of 16 u16 loads;
//  * each workgroup owns one BV=64 value half of a head (grid [2, 48], 256
//    threads, 35 KB of LDS; the shipped kernel runs both halves in 512
//    threads), so three workgroups fit a WGP. Both halves compute the
//    (identical) score tiles, two per wave.
// gfx11 WMMA layouts: operands hold halves k0..k0+15 of row lane&15 in both
// lane halves; accumulator element e of lane (lo, hi) is row 2e+hi.

#define GS_C 64
#define GS_D 128
#define GS_BV 64
#define GS_KP 132
#define GS_AP 68
#define GS_KTP 66
#define GS_NT 256

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
// LDS-only workgroup barrier: the "local" fences keep in-flight global
// prefetches in flight. Each wave waits for its own LDS ops (lgkmcnt(0), vmcnt
// and expcnt left open: gfx11 s_waitcnt 0xFC07) before it signals. The fence's
// own wait is a soft s_waitcnt that SIInsertWaitcnts may erase as redundant
// without seeing loop-carried stores (llvm/llvm-project#220357); in WGP mode
// the waves span both CUs, whose LDS queues are not ordered, so a read after
// the barrier could overtake another wave's queued store (the gfx1151
// VerifyAttn race). The explicit wait cannot be erased.
#define GS_WAIT_LGKM0 0xFC07
#define GS_BAR() do{__builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local");__builtin_amdgcn_s_waitcnt(GS_WAIT_LGKM0);__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");}while(0)

struct __attribute__((aligned(16))) GsLds{
 unsigned short score[GS_C][GS_AP];
 unsigned short state[GS_BV][GS_KP];
 unsigned short d[GS_BV][GS_AP];
 float ctrl[4][GS_C];
};
static_assert(sizeof(unsigned short)*GS_D*GS_KTP<=sizeof(unsigned short)*GS_BV*GS_KP,"transposed K fits the z tile");

extern "C" __launch_bounds__(GS_NT,1)
__global__ void gdn_chunk_scan_gfx1151(
 const unsigned short* __restrict__ q,const unsigned short* __restrict__ k,const unsigned short* __restrict__ v,
 const unsigned short* __restrict__ A,const float* __restrict__ G,const float* __restrict__ beta,
 signed char* __restrict__ sq,float* __restrict__ sc,_Float16* __restrict__ ef,float* __restrict__ out,int row0,int T){
 if(T<=0||T>512)return;
 const int head=(int)blockIdx.y;
 const int tid=(int)threadIdx.x,lane=tid&31,lo=lane&15,hi=lane>>4;
 const int wave=__builtin_amdgcn_readfirstlane(tid>>5);
 if((int)blockIdx.x>1||head>=48)return;
 const int half=(int)blockIdx.x,swave=wave;
 const int vbase=half*GS_BV,keyhead=head/3;
 const int kt0=(swave>>1)*2,vt0=(swave&1)*2;
 __shared__ GsLds shared;
 GsLds& sm=shared;

 gs_f8 St[4];
#pragma unroll
 for(int ki=0;ki<2;++ki)for(int vi=0;vi<2;++vi){const int t=ki*2+vi;const int vv=vbase+(vt0+vi)*16+lo;
#pragma unroll
  for(int e=0;e<8;++e){const int key=(kt0+ki)*16+gs_acc_row(lane,e);
   const unsigned long long off=((unsigned long long)head*GS_D+vv)*GS_D+key;
   St[t][e]=(float)sq[off]*sc[head*GS_D+vv]+(float)ef[off];
  }}
 auto stage_state=[&](){
#pragma unroll
  for(int ki=0;ki<2;++ki)for(int vi=0;vi<2;++vi){int t=ki*2+vi;int vv=(vt0+vi)*16+lo;
   for(int e=0;e<8;++e)sm.state[vv][(kt0+ki)*16+gs_acc_row(lane,e)]=gs_b16(St[t][e]);}
 };
 const int nch=(T+GS_C-1)/GS_C;
 typedef const unsigned char* gs_gp;
 const int mt=swave>>1,ntb=(swave&1)*2,hx=swave*32+lane;
 constexpr int NP=2;  // score tiles per wave: K rows ntb and ntb+1
 const int pn[NP]={ntb,ntb+1};
 bool doP[NP];
#pragma unroll
 for(int p=0;p<NP;++p)doP[p]=pn[p]<=mt;
 auto frag=[&](gs_gp b,unsigned rowbytes,int r0,int ls,int k0)->gs_bfrag{int r=r0+lo;if(r>ls)r=ls;
  const unsigned off=(unsigned)r*rowbytes+(unsigned)k0*2u;union{uint4 w[2];gs_bfrag v;}u;
  u.w[0]=*reinterpret_cast<const uint4*>(b+off);u.w[1]=*reinterpret_cast<const uint4*>(b+off+16u);return u.v;};
 unsigned short (*kT)[GS_KTP]=reinterpret_cast<unsigned short(*)[GS_KTP]>(&sm.state[0][0]);
 for(int chunk=0;chunk<nch;++chunk){
  const int local0=chunk*GS_C,parent0=row0+local0;int rows=T-local0;if(rows>GS_C)rows=GS_C;const int last=rows-1;
  const float gl=G[(unsigned long long)(parent0+last)*48+head];
  const gs_gp kb=reinterpret_cast<gs_gp>(k+((unsigned long long)parent0*16+keyhead)*GS_D);
  const gs_gp qb=reinterpret_cast<gs_gp>(q+((unsigned long long)parent0*16+keyhead)*GS_D);
  if(tid<GS_C){if(tid<rows){const unsigned long long go=(unsigned long long)(parent0+tid)*48+head;const float g=G[go];sm.ctrl[0][tid]=expf(g);sm.ctrl[1][tid]=expf(gl-g);sm.ctrl[2][tid]=g;sm.ctrl[3][tid]=beta[go];}else{sm.ctrl[0][tid]=0;sm.ctrl[1][tid]=0;sm.ctrl[2][tid]=0;sm.ctrl[3][tid]=0;}}
  stage_state();
  {const gs_gp vb=reinterpret_cast<gs_gp>(v+((unsigned long long)parent0*48+head)*GS_D+vbase);uint4 vw[2];
#pragma unroll
   for(int it=0;it<2;++it){const int idx=hx+it*256,r=idx>>3,rr=r<rows?r:last;vw[it]=*reinterpret_cast<const uint4*>(vb+(unsigned)rr*(48u*GS_D*2u)+(unsigned)(idx&7)*16u);}
#pragma unroll
   for(int it=0;it<2;++it){const int idx=hx+it*256,r=idx>>3,c=(idx&7)*8;const unsigned u[4]={vw[it].x,vw[it].y,vw[it].z,vw[it].w};
#pragma unroll
    for(int e=0;e<4;++e){sm.d[c+2*e][r]=(unsigned short)(u[e]&0xffffu);sm.d[c+2*e+1][r]=(unsigned short)(u[e]>>16);}}}
  GS_BAR();

  gs_f8 O[2]={gs_z8(),gs_z8()},U[2]={gs_z8(),gs_z8()},Pt[NP];
  gs_bfrag kn[NP];
#pragma unroll
  for(int p=0;p<NP;++p)Pt[p]=gs_z8();
  gs_bfrag km=frag(kb,16u*GS_D*2u,mt*16,last,0),qq=frag(qb,16u*GS_D*2u,mt*16,last,0);
#pragma unroll
  for(int p=0;p<NP;++p){kn[p]=km;if(doP[p])kn[p]=frag(kb,16u*GS_D*2u,pn[p]*16,last,0);}
#pragma unroll
  for(int kk=0;kk<8;++kk){gs_bfrag km1=km,qq1=qq,kn1[NP];
#pragma unroll
   for(int p=0;p<NP;++p)kn1[p]=kn[p];
   if(kk<7){km1=frag(kb,16u*GS_D*2u,mt*16,last,(kk+1)*16);qq1=frag(qb,16u*GS_D*2u,mt*16,last,(kk+1)*16);
#pragma unroll
    for(int p=0;p<NP;++p)if(doP[p])kn1[p]=frag(kb,16u*GS_D*2u,pn[p]*16,last,(kk+1)*16);}
   __builtin_amdgcn_sched_barrier(0);
#pragma unroll
   for(int j=0;j<2;++j){gs_bfrag st=gs_frag(&sm.state[0][0],GS_KP,(ntb+j)*16,kk*16,lane);U[j]=gs_mma(km,st,U[j]);O[j]=gs_mma(st,qq,O[j]);}
#pragma unroll
   for(int p=0;p<NP;++p)if(doP[p])Pt[p]=gs_mma(kn[p],qq,Pt[p]);
   km=km1;qq=qq1;
#pragma unroll
   for(int p=0;p<NP;++p)kn[p]=kn1[p];}
  const float oge=sm.ctrl[0][mt*16+lo];
#pragma unroll
  for(int j=0;j<2;++j)for(int e=0;e<8;++e)O[j][e]*=oge;
  {const int i=mt*16+lo;
#pragma unroll
   for(int p=0;p<NP;++p)
#pragma unroll
    for(int e=0;e<8;++e){const int j0e=pn[p]*16+gs_acc_row(lane,e);sm.score[i][j0e]=gs_b16((j0e<=i&&i<rows)?expf(sm.ctrl[2][i]-sm.ctrl[2][j0e])*Pt[p][e]:0.0f);}}
  GS_BAR();

  const gs_gp ab=reinterpret_cast<gs_gp>(A+(unsigned long long)local0*48*GS_C+(unsigned long long)head*GS_C);
  gs_bfrag aa[4];
#pragma unroll
  for(int kk=0;kk<4;++kk)aa[kk]=frag(ab,48u*GS_C*2u,mt*16,last,kk*16);
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for(int j=0;j<2;++j)
#pragma unroll
   for(int e=0;e<8;++e){int tok=mt*16+gs_acc_row(lane,e);int vv=(ntb+j)*16+lo;float z=sm.ctrl[3][tok]*(gs_bf(sm.d[vv][tok])-sm.ctrl[0][tok]*U[j][e]);sm.state[vv][tok]=gs_b16(z);}
  GS_BAR();

  gs_f8 Dacc[2]={gs_z8(),gs_z8()};
#pragma unroll
  for(int kk=0;kk<4;++kk){
#pragma unroll
   for(int j=0;j<2;++j)Dacc[j]=gs_mma(aa[kk],gs_frag(&sm.state[0][0],GS_KP,(ntb+j)*16,kk*16,lane),Dacc[j]);}
#pragma unroll
  for(int j=0;j<2;++j)for(int e=0;e<8;++e){int tok=mt*16+gs_acc_row(lane,e);sm.d[(ntb+j)*16+lo][tok]=gs_b16(Dacc[j][e]);}
  GS_BAR();

  constexpr int KIT=GS_C*GS_D/8/GS_NT;
  uint4 kw[KIT];
#pragma unroll
  for(int it=0;it<KIT;++it){const int idx=tid+it*GS_NT,r=idx>>4,rr=r<rows?r:last;kw[it]=*reinterpret_cast<const uint4*>(kb+(unsigned)rr*(16u*GS_D*2u)+(unsigned)(idx&15)*16u);}
#pragma unroll
  for(int kk=0;kk<4;++kk){gs_bfrag score=gs_frag(&sm.score[0][0],GS_AP,mt*16,kk*16,lane);
#pragma unroll
   for(int j=0;j<2;++j)O[j]=gs_mma(gs_frag(&sm.d[0][0],GS_AP,(ntb+j)*16,kk*16,lane),score,O[j]);}
  // WMMA needs every lane active. Without this pin, clang sinks the O
  // WMMAs (only used by the store) into `if(tok<rows)`, and on a partial
  // chunk the masked lanes' operand rows come out undefined.
  asm volatile("" : "+v"(O[0]), "+v"(O[1]));
  const int tok=mt*16+lo;
  if(tok<rows){float*ob=out+((unsigned long long)(parent0+tok)*48+head)*GS_D+vbase;
#pragma unroll
   for(int j=0;j<2;++j)
#pragma unroll
    for(int e=0;e<8;++e)ob[(ntb+j)*16+gs_acc_row(lane,e)]=O[j][e];}
  // The z tile in state[] has been dead since the previous barrier: K lands
  // there transposed ([key][token]).
#pragma unroll
  for(int it=0;it<KIT;++it){const int idx=tid+it*GS_NT,r=idx>>4,c=(idx&15)*8;const unsigned u[4]={kw[it].x,kw[it].y,kw[it].z,kw[it].w};
#pragma unroll
   for(int e=0;e<4;++e){kT[c+2*e][r]=(unsigned short)(u[e]&0xffffu);kT[c+2*e+1][r]=(unsigned short)(u[e]>>16);}}
  GS_BAR();
#pragma unroll
  for(int j=0;j<2;++j)for(int e=0;e<8;++e){int t=mt*16+gs_acc_row(lane,e),vv=(ntb+j)*16+lo;sm.d[vv][t]=gs_b16(sm.ctrl[1][t]*gs_bf(sm.d[vv][t]));}
  GS_BAR();

  const float ec=expf(gl);
#pragma unroll
  for(int t=0;t<4;++t)for(int e=0;e<8;++e)St[t][e]*=ec;
#pragma unroll
  for(int kk=0;kk<4;++kk){gs_bfrag dv[2]={gs_frag(&sm.d[0][0],GS_AP,(vt0+0)*16,kk*16,lane),gs_frag(&sm.d[0][0],GS_AP,(vt0+1)*16,kk*16,lane)};
#pragma unroll
   for(int ki=0;ki<2;++ki){gs_bfrag kt=gs_frag(&kT[0][0],GS_KTP,(kt0+ki)*16,kk*16,lane);
#pragma unroll
    for(int vi=0;vi<2;++vi)St[ki*2+vi]=gs_mma(kt,dv[vi],St[ki*2+vi]);}}
  GS_BAR();
 }

 // The old EF was folded into St before the scan so it participates in every
 // output and decay.  The dead state staging tile holds the full-row reductions.
 float vmax[2]={0,0};
#pragma unroll
 for(int vi=0;vi<2;++vi){int vv=vbase+(vt0+vi)*16+lo;
#pragma unroll
  for(int ki=0;ki<2;++ki){int t=ki*2+vi;
#pragma unroll
   for(int e=0;e<8;++e)vmax[vi]=fmaxf(vmax[vi],fabsf(St[t][e]));}
  vmax[vi]=fmaxf(vmax[vi],__shfl_xor(vmax[vi],16));if(hi==0)reinterpret_cast<float*>(&sm.state[0][0])[(swave>>1)*GS_BV+vv-vbase]=vmax[vi];}
 GS_BAR();
#pragma unroll
 for(int vi=0;vi<2;++vi){int vv=vbase+(vt0+vi)*16+lo;float mx=0;
#pragma unroll
  for(int g=0;g<4;++g)mx=fmaxf(mx,reinterpret_cast<float*>(&sm.state[0][0])[g*GS_BV+vv-vbase]);float inv=mx>0?127.0f/mx:0.0f,scale=mx>0?mx/127.0f:1.0f;if(swave<2&&hi==0)sc[head*GS_D+vv]=scale;
#pragma unroll
  for(int ki=0;ki<2;++ki){int t=ki*2+vi;
#pragma unroll
   for(int e=0;e<8;++e){const unsigned long long off=((unsigned long long)head*GS_D+vv)*GS_D+(kt0+ki)*16+gs_acc_row(lane,e);float qf=fminf(fmaxf(rintf(St[t][e]*inv),-128.0f),127.0f);sq[off]=(signed char)qf;ef[off]=(_Float16)(St[t][e]-qf*scale);}}}
}
