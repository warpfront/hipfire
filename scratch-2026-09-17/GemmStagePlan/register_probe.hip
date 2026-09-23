#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
typedef int __attribute__((ext_vector_type(2))) v2i;
typedef float __attribute__((ext_vector_type(8))) v8f;
#define CK(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);}}while(0)
template<bool FOLD> __device__ __forceinline__ void body(float* out,int seed){
 v2i a={(int)(0x30303030u+seed),(int)0x30303030},b={(int)0x38383838,(int)0x38383838};
 v8f P[8],T[8];for(int j=0;j<8;j++){P[j]=v8f{};T[j]=v8f{};}
 float s=0.01f*seed,z=0.03125f*seed,S=2.0f+threadIdx.x;
 for(int it=0;it<1024;it++){
  asm volatile("" : "+v"(a[0]),"+v"(a[1]));
  if constexpr(FOLD){
   #pragma unroll
   for(int j=0;j<8;j++)P[j]=v8f{};
  }
  #pragma unroll
  for(int k=0;k<8;k++){
   #pragma unroll
   for(int j=0;j<8;j++){v2i bj={b[0]+j,b[1]+j*256};P[j]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,bj,P[j]);}
   __builtin_amdgcn_sched_barrier(0);
  }
  if constexpr(FOLD){
   #pragma unroll
   for(int j=0;j<8;j++){
    #pragma unroll
    for(int c=0;c<8;c++){T[j][c]=__fmaf_rn(s,P[j][c],T[j][c]);T[j][c]=__fmaf_rn(z,S,T[j][c]);}
   }
  }
 }
 float sum=0;
 #pragma unroll
 for(int j=0;j<8;j++){
  #pragma unroll
  for(int c=0;c<8;c++)sum+=(FOLD?T[j][c]:P[j][c]);
 }
 out[blockIdx.x*256+threadIdx.x]=sum;
}
extern "C" __global__ __launch_bounds__(256) void pure(float* o,int s){body<false>(o,s);}
extern "C" __global__ __launch_bounds__(256) void fold(float* o,int s){body<true>(o,s);}
int main(int argc,char**argv){int blocks=argc>1?atoi(argv[1]):256;float*o;CK(hipMalloc(&o,blocks*256*4));hipEvent_t a,b;CK(hipEventCreate(&a));CK(hipEventCreate(&b));
 for(int round=0;round<3;round++)for(int mode=0;mode<2;mode++){
  auto f=mode?fold:pure;hipFuncAttributes at;CK(hipFuncGetAttributes(&at,(const void*)f));std::vector<float>ts;
  for(int r=0;r<25;r++){CK(hipEventRecord(a));hipLaunchKernelGGL(f,blocks,256,0,0,o,7);CK(hipEventRecord(b));CK(hipEventSynchronize(b));float ms;CK(hipEventElapsedTime(&ms,a,b));if(r>=4)ts.push_back(ms*1000);}
  std::sort(ts.begin(),ts.end());double ops=(double)blocks*8*1024*64*8192;
  printf("round=%d mode=%s us=%.3f TF=%.3f regs=%d local=%zu shared=%zu\n",round,mode?"fold":"pure",ts[10],ops/(ts[10]*1e6),at.numRegs,at.localSizeBytes,at.sharedSizeBytes);
 }
}
