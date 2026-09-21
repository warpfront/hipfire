#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
typedef int __attribute__((ext_vector_type(2))) i2;
typedef int __attribute__((ext_vector_type(8))) i8;
__global__ __launch_bounds__(256) void peak_iu4(unsigned long long* out,int iters){
  const i2 a={(int)(threadIdx.x|0x11111111u),(int)(threadIdx.x|0x22222222u)};
  const i2 b={(int)(threadIdx.x|0x33333333u),(int)(threadIdx.x|0x44444444u)};
  const i8 z={0,0,0,0,0,0,0,0}; i8 c[8];
#pragma unroll
  for(int q=0;q<8;q++)c[q]=z;
  for(int i=0;i<iters;i++){
#pragma unroll
    for(int q=0;q<8;q++)c[q]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,a,true,b,c[q],false);
  }
  unsigned long long s=0;
#pragma unroll
  for(int q=0;q<8;q++)s+=(unsigned)c[q][threadIdx.x&7];
  if((threadIdx.x&31)==0)atomicAdd(out,s);
}
#define OK(x) do{auto e=(x);if(e){printf("HIP %s\n",hipGetErrorString(e));exit(1);}}while(0)
static void run(const char*name,size_t lds,int cu){int occ=0;OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ,peak_iu4,256,lds));int blocks=cu*occ,iters=10000;unsigned long long*out;OK(hipMalloc(&out,8));OK(hipMemset(out,0,8));peak_iu4<<<blocks,256,lds>>>(out,iters);OK(hipDeviceSynchronize());std::vector<float>t;for(int r=0;r<5;r++){hipEvent_t a,b;OK(hipEventCreate(&a));OK(hipEventCreate(&b));OK(hipEventRecord(a));peak_iu4<<<blocks,256,lds>>>(out,iters);OK(hipEventRecord(b));OK(hipEventSynchronize(b));float ms;OK(hipEventElapsedTime(&ms,a,b));t.push_back(ms);OK(hipEventDestroy(a));OK(hipEventDestroy(b));}std::sort(t.begin(),t.end());double ops=(double)blocks*8.0*iters*8.0*16384.0;printf("PEAK,%s,blocks_per_cu=%d,lds=%zu,ms=%.3f,TOPS=%.3f\n",name,occ,lds,t[2],ops/(t[2]*1e9));OK(hipFree(out));}
int main(){int dev,cu;OK(hipGetDevice(&dev));OK(hipDeviceGetAttribute(&cu,hipDeviceAttributeMultiprocessorCount,dev));run("gemm_occ3",21845,cu);run("max_occupancy",0,cu);}
