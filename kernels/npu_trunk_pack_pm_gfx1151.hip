// SPDX-License-Identifier: Apache-2.0
// Route-A PeaceMaker twin source (scripts/pm-hip-twin.sh) of the Flash-Next A-pack fast producers of
// src/npu_trunk.hip (`npu_trunk_pack_a_bf16`, K=2560, and `npu_trunk_pack_a_bf16_k5120`): the code below is a verbatim copy
// of the incumbent's a_column (lines 6-11) and pack_a_bf16_fixed (lines 51-123); only the two entry symbols carry the `_pm`
// suffix. The sealed `.text` of each twin must equal the incumbent's hipcc `.text` byte for byte (pm-hip-twin.sh
// `judged_text_check`). The scale snap `ceilf` lowers to `v_ceil_f32` (peacemaker-ir isa/*.tbl row, swept over all 2^32 inputs).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// A-stream injecting column of one 8-row group: 128-row block b of the wave, E/O half h ((rt/8)%2).
// G80 N1280 (256-row waves): col = 2b + h. G80 N512 (512-row waves): col = 4(b/2) + 2h + b%2.
static __device__ __forceinline__ int a_column(int t,int wave_rows){
    const int b=(t%wave_rows)/128, h=(t%128/8)%2;
    return wave_rows==256 ? b*2+h : (b>>1)*4+h*2+(b&1);
}

// LDS joins eight rows so uint4 writes fill entire native 8x8 byte microtiles.
// No arithmetic reassociation: the scale/quantizer are the scalar oracle's DAG.
template<int K>
static __device__ __forceinline__ void pack_a_bf16_fixed(
    const __half* __restrict__ x, unsigned char* __restrict__ a,
    unsigned int* __restrict__ d, int rows, int padded_rows, int wave_rows) {
    constexpr int CHUNKS=K/8, PITCH=CHUNKS+1, J=K/256;
    const int tid=threadIdx.x, lane=tid&31, row=tid>>5;
    const int first=blockIdx.x*8, t=first+row;
    __shared__ uint2 codes[8][PITCH];
    __shared__ __align__(16) unsigned short mantissa[8], exponent[8];
    uint4 values[J];
    unsigned int magnitude=0;
    #pragma unroll
    for(int j=0;j<J;j++) {
        uint4 v=make_uint4(0,0,0,0);
        if(t<rows) v=((const uint4*)(x+(size_t)t*K))[j*32+lane];
        values[j]=v;
        const unsigned int words[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int h=0;h<4;h++) {
            magnitude=max(magnitude,words[h]&0x7fffu);
            magnitude=max(magnitude,(words[h]>>16)&0x7fffu);
        }
    }
    #pragma unroll
    for(int s=16;s;s>>=1) magnitude=max(magnitude,__shfl_xor(magnitude,s,32));
    int m=0,e=0; float inv=0;
    if(lane==0) {
        const float am=__half2float(__ushort_as_half((unsigned short)magnitude));
        if(am!=0) {
            const float raw=__fdiv_rn(am,127.0f); int ex;
            const float f=frexpf(raw,&ex); e=ex-15;
            m=(int)ceilf(__fmul_rn(f,32768.0f));
            if(m==32768){m=16384;e++;}
            inv=__fdiv_rn(1.0f,ldexpf((float)m,e));
        }
        mantissa[row]=(unsigned short)m; exponent[row]=(unsigned short)e;
        if(t<rows)d[t]=(unsigned int)m|((unsigned int)(unsigned short)e<<16);
    }
    inv=__shfl(inv,0,32);
    #pragma unroll
    for(int j=0;j<J;j++) {
        const uint4 v=values[j];
        const unsigned int words[4]={v.x,v.y,v.z,v.w};
        unsigned int lo=0,hi=0;
        #pragma unroll
        for(int h=0;h<8;h++) {
            const unsigned short bits=(unsigned short)(words[h/2]>>((h&1)*16));
            int q=__float2int_rn(__fmul_rn(__half2float(__ushort_as_half(bits)),inv));
            q=q>127?127:(q< -127?-127:q);
            if(h<4)lo|=(unsigned int)(unsigned char)(signed char)q<<(h*8);
            else hi|=(unsigned int)(unsigned char)(signed char)q<<((h-4)*8);
        }
        codes[row][j*32+lane]=make_uint2(lo,hi);
    }
    __syncthreads();
    const int waves=padded_rows/wave_rows, kc=K/64, wave=first/wave_rows;
    const int rt=first%128, col=a_column(first,wave_rows);
    for(int i=tid;i<CHUNKS*4;i+=256) {
        const int chunk=i/4, r=(i%4)*2;
        const uint2 lo=codes[r][chunk], hi=codes[r+1][chunk];
        const size_t off=((size_t)col*waves*kc+(size_t)wave*kc+chunk/8)*4608
            +(rt/16)*512+(chunk%8)*64+(rt%8+r)*8;
        *((uint4*)(a+off))=make_uint4(lo.x,lo.y,hi.x,hi.y);
    }
    if(tid<kc*2) {
        const int ch=tid/2,h=tid%2, c=a_column(first-((first/8)%2)*8+h*8,wave_rows);
        const size_t off=((size_t)c*waves*kc+(size_t)wave*kc+ch)*4608+4096+rt*2;
        *((uint4*)(a+off))=*((const uint4*)mantissa);
        *((uint4*)(a+off+256))=*((const uint4*)exponent);
    }
}
extern "C" __global__ __launch_bounds__(256) void npu_trunk_pack_a_bf16_pm(
    const __half* __restrict__ x, unsigned char* __restrict__ a,
    unsigned int* __restrict__ d, int rows, int k, int padded_rows, int wave_rows) {
    pack_a_bf16_fixed<2560>(x,a,d,rows,padded_rows,wave_rows);
}
// qwen3.8-27b dense in-projections (hidden 5120): 80 VGPRs of halves per lane, 41 KiB LDS.
extern "C" __global__ __launch_bounds__(256) void npu_trunk_pack_a_bf16_k5120_pm(
    const __half* __restrict__ x, unsigned char* __restrict__ a,
    unsigned int* __restrict__ d, int rows, int k, int padded_rows, int wave_rows) {
    pack_a_bf16_fixed<5120>(x,a,d,rows,padded_rows,wave_rows);
}
