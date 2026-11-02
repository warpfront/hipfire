// SPDX-License-Identifier: Apache-2.0
// grid15-up producer and exact BF16 epilogue; PM opcode gaps logged in PM-GAPS.md.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// A-stream injecting column of one 8-row group: 128-row block b of the wave, E/O half h ((rt/8)%2).
// G80 N1280 (256-row waves): col = 2b + h. G80 N512 (512-row waves): col = 4(b/2) + 2h + b%2.
static __device__ __forceinline__ int a_column(int t,int wave_rows){
    const int b=(t%wave_rows)/128, h=(t%128/8)%2;
    return wave_rows==256 ? b*2+h : (b>>1)*4+h*2+(b&1);
}

// Retained scalar producer: generic-K fallback and lab byte/timing oracle.
extern "C" __global__ __launch_bounds__(256) void npu_trunk_pack_a_bf16_old(
    const __half* x, unsigned char* a, unsigned int* d, int rows, int k, int padded_rows, int wave_rows) {
    const int t=blockIdx.x, tid=threadIdx.x, kc=k/64, waves=padded_rows/wave_rows;
    __shared__ float red[256];
    float am=0;
    if(t<rows) for(int i=tid;i<k;i+=256) am=fmaxf(am,fabsf(__half2float(x[(size_t)t*k+i])));
    red[tid]=am; __syncthreads();
    for(int s=128;s;s>>=1){ if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]); __syncthreads(); }
    am=red[0]; int m=0,e=0; float inv=0;
    if(am!=0){
        const float raw=__fdiv_rn(am,127.0f); int ex;
        const float f=frexpf(raw,&ex); e=ex-15;
        m=(int)ceilf(__fmul_rn(f,32768.0f));
        if(m==32768){m=16384;e++;}
        inv=__fdiv_rn(1.0f,ldexpf((float)m,e));
    }
    if(tid==0 && t<rows)d[t]=(unsigned int)m|((unsigned int)(unsigned short)e<<16);
    const int wave=t/wave_rows, rt=t%128, col=a_column(t,wave_rows);
    for(int i=tid;i<k;i+=256){
        int q=0; if(t<rows && am!=0)q=__float2int_rn(__fmul_rn(__half2float(x[(size_t)t*k+i]),inv));
        q=q>127?127:(q< -127?-127:q);
        const int ch=i/64, kb=(i%64)/8;
        const size_t off=((size_t)col*waves*kc+(size_t)wave*kc+ch)*4608
            +(rt/16)*512+kb*64+(rt%8)*8+i%8;
        a[off]=(unsigned char)(signed char)q;
    }
    // Each token's scale is repeated in E and O halves for every K64.
    for(int ch=tid;ch<kc;ch+=256)for(int h=0;h<2;h++){
        const int c=a_column(t-((t/8)%2)*8+h*8,wave_rows);
        const size_t off=((size_t)c*waves*kc+(size_t)wave*kc+ch)*4608+4096;
        ((unsigned short*)(a+off))[rt]=(unsigned short)m;
        ((short*)(a+off+256))[rt]=(short)e;
    }
}

// Fixed K: eight wave32 rows per CTA. Load each half once into K/64 packed VGPRs
// per lane, reduce the finite half magnitude bits, then quantize from registers.
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
extern "C" __global__ __launch_bounds__(256) void npu_trunk_pack_a_bf16(
    const __half* __restrict__ x, unsigned char* __restrict__ a,
    unsigned int* __restrict__ d, int rows, int k, int padded_rows, int wave_rows) {
    pack_a_bf16_fixed<2560>(x,a,d,rows,padded_rows,wave_rows);
}
// qwen3.8-27b dense in-projections (hidden 5120): 80 VGPRs of halves per lane, 41 KiB LDS.
extern "C" __global__ __launch_bounds__(256) void npu_trunk_pack_a_bf16_k5120(
    const __half* __restrict__ x, unsigned char* __restrict__ a,
    unsigned int* __restrict__ d, int rows, int k, int padded_rows, int wave_rows) {
    pack_a_bf16_fixed<5120>(x,a,d,rows,padded_rows,wave_rows);
}
static __device__ __forceinline__ unsigned short direct(int isum,int sm,int se,int dm,int de){
    const long long j=((long long)isum*sm)*dm;
    if(j==0)return 0;
    const unsigned int sign=j<0?0x8000u:0u;
    const unsigned long long mag=j<0?(0ull-(unsigned long long)j):(unsigned long long)j;
    const int sh=se+de,l=63-(int)__clzll((long long)mag);
    int q=l+sh-7;if(q< -133)q=-133;
    if(q+133>255)return (unsigned short)(sign|0x7f80u);
    const int rs=q-sh; unsigned long long r;
    if(rs<=0)r=mag<<(-rs);
    else if(rs>=64)r=0;
    else{r=mag>>rs;const unsigned long long rem=mag&((1ull<<rs)-1),half=1ull<<(rs-1);
        if(rem>half||(rem==half&&(r&1)))r++;}
    unsigned int bits=((unsigned int)(q+133)<<7)+(unsigned int)r;
    if(bits>=0x7f80u)bits=0x7f80u;
    return (unsigned short)(sign|bits);
}
extern "C" __global__ __launch_bounds__(256) void npu_trunk_epi_bf16(
    const int* sums,const unsigned int* scales,const unsigned int* d,void* out,
    int rows,int n,int col0,int cols,int bf16){
    const int col=blockIdx.x*256+threadIdx.x,t=blockIdx.y;
    if(col>=cols||t>=rows)return;
    const unsigned int s=scales[col0+col],dt=d[t];
    const unsigned short bits=direct(sums[(size_t)t*n+col0+col],s&65535,(short)(s>>16),dt&65535,(short)(dt>>16));
    const size_t o=(size_t)t*cols+col;
    if(bf16)((unsigned short*)out)[o]=bits;
    else ((unsigned int*)out)[o]=(unsigned int)bits<<16;
}

// One CTA reorders a native 128x80 tile. Consecutive lanes read consecutive uint4s
// from the G80 stream, then LDS bridges to contiguous row-major uint4 stores.
// The 88-element LDS pitch keeps every row 16-byte aligned without the 80-column
// power-of-two bank alias. No floating-point conversion: preserve BF16 NaN bits.
extern "C" __global__ __launch_bounds__(256) void npu_trunk_consume_bf16_fast(
    const unsigned short* __restrict__ c, void* __restrict__ out,
    int padded_rows, int rows, int col0, int cols, int bf16) {
    __shared__ __align__(16) unsigned short tile[128][88];
    const int tid=threadIdx.x, nt=col0/80+blockIdx.x;
    const int panel=nt/16, tn=nt%16, mt=blockIdx.y;
    const int waves=padded_rows/256;
    const size_t base=(size_t)panel*padded_rows*1280
        +((size_t)(tn/2)*waves*4+(mt%2)*waves*2+(mt/2)*2+tn%2)*10240;
    for(int i=tid;i<1280;i+=256) {
        // Native order: [row/8][col/8][row%8][8 BF16].
        const int micro=i/8, r=(micro/10)*8+i%8, j=(micro%10)*8;
        *((uint4*)&tile[r][j])=((const uint4*)(c+base))[i];
    }
    __syncthreads();
    for(int i=tid;i<1280;i+=256) {
        const int r=i/10, j=(i%10)*8, row=mt*128+r;
        const int col=nt*80+j-col0;
        if(row>=rows || col+8<=0 || col>=cols)continue;
        const uint4 v=*((const uint4*)&tile[r][j]);
        const size_t offset=(size_t)row*cols;
        if(col>=0 && col+8<=cols && ((offset+col)&7)==0) {
            if(bf16) ((uint4*)((unsigned short*)out+offset+col))[0]=v;
            else {
                uint4* dst=(uint4*)((unsigned int*)out+offset+col);
                dst[0]=make_uint4(v.x<<16,v.x&0xffff0000u,v.y<<16,v.y&0xffff0000u);
                dst[1]=make_uint4(v.z<<16,v.z&0xffff0000u,v.w<<16,v.w&0xffff0000u);
            }
        } else {
            // Arbitrary projection boundaries keep the same API as the scalar oracle.
            for(int k=0;k<8;k++)if(col+k>=0 && col+k<cols) {
                const unsigned short bits=tile[r][j+k];
                if(bf16)((unsigned short*)out)[offset+col+k]=bits;
                else ((unsigned int*)out)[offset+col+k]=(unsigned int)bits<<16;
            }
        }
    }
}
