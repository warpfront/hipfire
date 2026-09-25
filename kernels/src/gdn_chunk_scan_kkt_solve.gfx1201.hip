// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Fixed Qwen3.5-27B GDN KKT solve: C64, K128, 16 key heads and three
// value heads per key head.  A workgroup forms one Gram matrix and applies
// three independent gate/beta systems.  Only the final FP16 inverses leave it.
#define GK_C 64
#define GK_D 128
#define GK_PITCH 68
#define GK_B 16

#if defined(__gfx1200__) || defined(__gfx1201__)
#define GDN_CHUNK_GFX12 1
typedef _Float16 gk_bfrag __attribute__((ext_vector_type(8)));
#else
#define GDN_CHUNK_GFX12 0
typedef _Float16 gk_bfrag __attribute__((ext_vector_type(16)));
#endif
typedef float gk_f8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float gk_from_bf(unsigned short x) {
    return __half2float(__ushort_as_half(x));
}
__device__ __forceinline__ unsigned short gk_to_bf(float x) {
    return __half_as_ushort(__float2half_rn(x));
}
__device__ __forceinline__ gk_f8 gk_zero() {
    return (gk_f8){0,0,0,0,0,0,0,0};
}
__device__ __forceinline__ gk_f8 gk_mma(gk_bfrag a, gk_bfrag b, gk_f8 c) {
#if GDN_CHUNK_GFX12
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c);
#endif
}
__device__ __forceinline__ int gk_acc_row(int lane, int elem) {
#if GDN_CHUNK_GFX12
    return 8 * (lane >> 4) + elem;
#else
    return 2 * elem + (lane >> 4);
#endif
}
__device__ __forceinline__ gk_bfrag gk_frag_shared(
    const unsigned short* p, int pitch, int row0, int k0, int lane) {
    union { unsigned short h[sizeof(gk_bfrag) / sizeof(unsigned short)]; gk_bfrag v; } u;
    const int row = row0 + (lane & 15);
#if GDN_CHUNK_GFX12
    const int kh = k0 + 4 * (lane >> 4);
#pragma unroll
    for (int e=0;e<4;++e) {
        u.h[e] = p[row*pitch + kh+e];
        u.h[e+4] = p[row*pitch + kh+8+e];
    }
#else
#pragma unroll
    for (int e=0;e<16;++e) u.h[e] = p[row*pitch + k0+e];
#endif
    return u.v;
}
__device__ __forceinline__ gk_bfrag gk_frag_transpose(
    const unsigned short* p, int pitch, int row0, int k0, int lane) {
    union { unsigned short h[sizeof(gk_bfrag) / sizeof(unsigned short)]; gk_bfrag v; } u;
    const int row = row0 + (lane & 15);
#if GDN_CHUNK_GFX12
    const int kh = k0 + 4 * (lane >> 4);
#pragma unroll
    for (int e=0;e<4;++e) {
        u.h[e] = p[(kh+e)*pitch + row];
        u.h[e+4] = p[(kh+8+e)*pitch + row];
    }
#else
#pragma unroll
    for (int e=0;e<16;++e) u.h[e] = p[(k0+e)*pitch + row];
#endif
    return u.v;
}
__device__ __forceinline__ gk_bfrag gk_frag_global(
    const unsigned short* p, unsigned row_stride, int row0, int last, int k0, int lane) {
    union { unsigned short h[sizeof(gk_bfrag) / sizeof(unsigned short)]; gk_bfrag v; } u;
    int row = row0 + (lane & 15);
    if (row > last) row = last;
#if GDN_CHUNK_GFX12
    const int kh = k0 + 4 * (lane >> 4);
#pragma unroll
    for (int e=0;e<4;++e) {
        u.h[e] = p[(unsigned)row*row_stride + kh+e];
        u.h[e+4] = p[(unsigned)row*row_stride + kh+8+e];
    }
#else
#pragma unroll
    for (int e=0;e<16;++e) u.h[e] = p[(unsigned)row*row_stride + k0+e];
#endif
    return u.v;
}
__device__ __forceinline__ void gk_tile(int t, int& rb, int& cb) {
    // Upper-triangular order by row: (0,0..3),(1,1..3),(2,2..3),(3,3).
    if (t < 4) { rb=0; cb=t; }
    else if (t < 7) { rb=1; cb=t-3; }
    else if (t < 9) { rb=2; cb=t-5; }
    else { rb=3; cb=3; }
}

// Exact gfx1201 stages the K chunk once with 16-byte loads, prefetches the
// three gate/beta columns and moves A/zeroing in 16-byte units.  Every WMMA
// operand, solve step and FP16 rounding is unchanged.
#if defined(__gfx1201__)
#define GK_G12_VEC 1
#else
#define GK_G12_VEC 0
#endif
#define GK_KP 132
struct __attribute__((aligned(16))) GkLds {
    unsigned short a[GK_C][GK_PITCH];
    unsigned short tmp[32][36];
    float gate[GK_C];
    float beta[GK_C];
#if GK_G12_VEC
    unsigned short kc[GK_C][GK_KP];
#endif
};

extern "C" __launch_bounds__(128)
__global__ void gdn_chunk_kkt_solve(
    const unsigned short* __restrict__ k,
    const float* __restrict__ G,
    const float* __restrict__ beta,
    unsigned short* __restrict__ A,
    int row0,
    int T) {
    if (T <= 0 || T > 512) return;
    const int chunk = (int)blockIdx.x;
    const int kh = (int)blockIdx.y;
    if (kh >= 16 || chunk*GK_C >= T) return;
    const int tid=(int)threadIdx.x, wave=tid>>5, lane=tid&31;
    const int local0=chunk*GK_C;
    int rows=T-local0; if(rows>GK_C)rows=GK_C;
    const int parent0=row0+local0;
    const unsigned kstride=16*GK_D;
    const unsigned short* kp=k+(unsigned long long)parent0*kstride+(unsigned)kh*GK_D;

    // Ten Gram tiles remain in registers while the three value-head systems
    // are solved.  Each wave owns at most three tiles.
    gk_f8 gram[3]={gk_zero(),gk_zero(),gk_zero()};
#if GK_G12_VEC
    __shared__ GkLds s;
    float pgate[3]={0.0f,0.0f,0.0f},pbeta[3]={0.0f,0.0f,0.0f};
    {uint4 w[8];
#pragma unroll
     for(int it=0;it<8;++it){const int idx=tid+it*128,r=idx>>4,rr=r<rows?r:rows-1;
      w[it]=*reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned char*>(kp)+(unsigned)rr*(kstride*2u)+(unsigned)(idx&15)*16u);}
     if(tid<rows){
#pragma unroll
      for(int sub=0;sub<3;++sub){const unsigned long long off=(unsigned long long)(parent0+tid)*48+kh*3+sub;pgate[sub]=G[off];pbeta[sub]=beta[off];}}
#pragma unroll
     for(int it=0;it<8;++it){const int idx=tid+it*128,r=idx>>4,c=(idx&15)*8;uint2*dst=reinterpret_cast<uint2*>(&s.kc[r][c]);
      dst[0]=make_uint2(w[it].x,w[it].y);dst[1]=make_uint2(w[it].z,w[it].w);}}
    __syncthreads();
#pragma unroll
    for(int slot=0;slot<3;++slot){
        const int tile=wave+slot*4;
        if(tile<10){int rb,cb;gk_tile(tile,rb,cb);
#pragma unroll
            for(int d=0;d<8;++d)
                gram[slot]=gk_mma(
                    gk_frag_shared(&s.kc[0][0],GK_KP,rb*GK_B,d*GK_B,lane),
                    gk_frag_shared(&s.kc[0][0],GK_KP,cb*GK_B,d*GK_B,lane),
                    gram[slot]);
        }
    }
#else
#pragma unroll
    for(int slot=0;slot<3;++slot){
        const int tile=wave+slot*4;
        if(tile<10){int rb,cb;gk_tile(tile,rb,cb);
#pragma unroll
            for(int d=0;d<8;++d)
                gram[slot]=gk_mma(
                    gk_frag_global(kp,kstride,rb*GK_B,rows-1,d*GK_B,lane),
                    gk_frag_global(kp,kstride,cb*GK_B,rows-1,d*GK_B,lane),
                    gram[slot]);
        }
    }

    __shared__ GkLds s;
#endif
#pragma unroll
    for(int sub=0;sub<3;++sub){
        const int vh=kh*3+sub;
#if GK_G12_VEC
        for(int x=tid;x<GK_C*GK_PITCH/8;x+=128)
            reinterpret_cast<uint4*>(s.a)[x]=make_uint4(0u,0u,0u,0u);
        if(tid<GK_C){
            if(tid<rows){s.gate[tid]=pgate[sub]; s.beta[tid]=pbeta[sub];}
            else{s.gate[tid]=0.0f;s.beta[tid]=0.0f;}
        }
#else
        for(int x=tid;x<GK_C*GK_PITCH;x+=128)
            reinterpret_cast<unsigned short*>(s.a)[x]=0;
        if(tid<GK_C){
            if(tid<rows){
                const unsigned long long off=(unsigned long long)(parent0+tid)*48+vh;
                s.gate[tid]=G[off]; s.beta[tid]=beta[off];
            }else{s.gate[tid]=0.0f;s.beta[tid]=0.0f;}
        }
#endif
        __syncthreads();

        // Place the symmetric Gram into strict-lower L.  Masking selects
        // -infinity before exp, so padded rows cannot create 0*infinity.
#if GK_G12_VEC
        // Same element expression; gates/beta come from vector LDS reads and
        // the select replaces the branch (masked values are discarded).
#pragma unroll
        for(int slot=0;slot<3;++slot){
            const int tile=wave+slot*4;
            if(tile<10){int rb,cb;gk_tile(tile,rb,cb);
                const int i=cb*GK_B+(lane&15),j0=rb*GK_B+8*(lane>>4);
                const float gi=s.gate[i],bi=s.beta[i];
                const float4 ga=*reinterpret_cast<const float4*>(&s.gate[j0]),gb=*reinterpret_cast<const float4*>(&s.gate[j0+4]);
                const float gj[8]={ga.x,ga.y,ga.z,ga.w,gb.x,gb.y,gb.z,gb.w};
                unsigned hb[8];
#pragma unroll
                for(int e=0;e<8;++e){const int j=j0+e;
                    const float exponent=gi-gj[e];
                    const float value=bi*expf(exponent)*gram[slot][e];
                    hb[e]=gk_to_bf((i<rows && j<i && j<rows)?value:0.0f);}
                uint2*dst=reinterpret_cast<uint2*>(&s.a[i][j0]);
                dst[0]=make_uint2(hb[0]|(hb[1]<<16),hb[2]|(hb[3]<<16));
                dst[1]=make_uint2(hb[4]|(hb[5]<<16),hb[6]|(hb[7]<<16));
            }
        }
#else
#pragma unroll
        for(int slot=0;slot<3;++slot){
            const int tile=wave+slot*4;
            if(tile<10){int rb,cb;gk_tile(tile,rb,cb);
                const int i=cb*GK_B+(lane&15);
#pragma unroll
                for(int e=0;e<8;++e){const int j=rb*GK_B+gk_acc_row(lane,e);float value=0.0f;
                    if(i<rows && j<i && j<rows){
                        const float exponent=s.gate[i]-s.gate[j];
                        value=s.beta[i]*expf(exponent)*gram[slot][e];
                    }
                    s.a[i][j]=gk_to_bf(value);
                }
            }
        }
#endif
        __syncthreads();

        // Four BC16 diagonal forward solves, one per wave.  Lane 0..15 owns
        // one output column; the second half-wave is intentionally inactive.
        if(lane<16){
            const int base=wave*GK_B;
            float col[GK_B];
#if GK_G12_VEC
            // The strict-lower block is read once as 8-byte rows before the
            // unchanged sequential substitution.
            uint2 rv[GK_B][4];
#pragma unroll
            for(int i=1;i<GK_B;++i)
#pragma unroll
                for(int q=0;q<4;++q) if(4*q<i) rv[i][q]=*reinterpret_cast<const uint2*>(&s.a[base+i][base+4*q]);
#pragma unroll
            for(int i=0;i<GK_B;++i){float value=(i==lane)?1.0f:0.0f;
#pragma unroll
                for(int j=0;j<GK_B;++j) if(j<i){
                    const unsigned w=(j&2)?rv[i][j>>2].y:rv[i][j>>2].x;
                    value-=gk_from_bf((unsigned short)((j&1)?(w>>16):(w&0xffffu)))*col[j];}
                col[i]=value;
            }
#else
#pragma unroll
            for(int i=0;i<GK_B;++i){float value=(i==lane)?1.0f:0.0f;
#pragma unroll
                for(int j=0;j<GK_B;++j) if(j<i)
                    value-=gk_from_bf(s.a[base+i][base+j])*col[j];
                col[i]=value;
            }
#endif
#pragma unroll
            for(int i=0;i<GK_B;++i)s.a[base+i][base+lane]=gk_to_bf(col[i]);
        }
        __syncthreads();

        // Merge the two pairs of diagonal blocks.  X=L10*D0 is staged as
        // X^T, then the same wave forms -D1*X.
        if(wave<2){
            const int origin=wave*32;
            gk_f8 x=gk_zero();
            x=gk_mma(gk_frag_shared(&s.a[0][0],GK_PITCH,origin+16,origin,lane),
                     gk_frag_transpose(&s.a[0][0],GK_PITCH,origin,origin,lane),x);
            const int xn=lane&15;
#pragma unroll
            for(int e=0;e<8;++e)s.tmp[wave*16+xn][gk_acc_row(lane,e)]=gk_to_bf(x[e]);
            __builtin_amdgcn_wave_barrier();
            gk_f8 y=gk_zero();
            y=gk_mma(gk_frag_shared(&s.a[0][0],GK_PITCH,origin+16,origin+16,lane),
                     gk_frag_shared(&s.tmp[0][0],36,wave*16,0,lane),y);
            const int yc=origin+(lane&15);
#pragma unroll
            for(int e=0;e<8;++e)s.a[origin+16+gk_acc_row(lane,e)][yc]=gk_to_bf(-y[e]);
        }
        __syncthreads();

        // Final 64 merge.  First form Y=R*P into Y^T in scratch, then
        // produce -Q*Y into the lower-left quadrant.
        {
            const int tile=wave;
            const int rb=tile>>1, cb=tile&1;
            gk_f8 y=gk_zero();
#pragma unroll
            for(int kk=0;kk<2;++kk)
                y=gk_mma(gk_frag_shared(&s.a[0][0],GK_PITCH,32+rb*16,kk*16,lane),
                         gk_frag_transpose(&s.a[0][0],GK_PITCH,cb*16,kk*16,lane),y);
            const int yn=cb*16+(lane&15);
#pragma unroll
            for(int e=0;e<8;++e)s.tmp[yn][rb*16+gk_acc_row(lane,e)]=gk_to_bf(y[e]);
        }
        __syncthreads();
        {
            const int tile=wave;
            const int rb=tile>>1, cb=tile&1;
            gk_f8 z=gk_zero();
#pragma unroll
            for(int kk=0;kk<2;++kk)
                z=gk_mma(gk_frag_shared(&s.a[0][0],GK_PITCH,32+rb*16,32+kk*16,lane),
                         gk_frag_shared(&s.tmp[0][0],36,cb*16,kk*16,lane),z);
            const int zc=cb*16+(lane&15);
#pragma unroll
            for(int e=0;e<8;++e)s.a[32+rb*16+gk_acc_row(lane,e)][zc]=gk_to_bf(-z[e]);
        }
        __syncthreads();

        const unsigned long long abase=(unsigned long long)local0*48*GK_C+(unsigned long long)vh*GK_C;
#if GK_G12_VEC
        {unsigned char*ab=reinterpret_cast<unsigned char*>(A+abase);
#pragma unroll
         for(int it=0;it<4;++it){const int idx=tid+it*128,r=idx>>3,c=(idx&7)*8;
          const uint2 lo2=*reinterpret_cast<const uint2*>(&s.a[r][c]),hi2=*reinterpret_cast<const uint2*>(&s.a[r][c+4]);
          *reinterpret_cast<uint4*>(ab+(unsigned)r*(48u*GK_C*2u)+(unsigned)c*2u)=make_uint4(lo2.x,lo2.y,hi2.x,hi2.y);}}
#else
        for(int x=tid;x<GK_C*GK_C;x+=128){
            const int r=x/GK_C,c=x-r*GK_C;
            A[abase+(unsigned long long)r*48*GK_C+c]=s.a[r][c];
        }
#endif
        __syncthreads();
    }
}
