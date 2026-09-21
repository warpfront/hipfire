#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define IU4_SYMMETRIC_FOLD 1
#if BENCH_WP
#define IU4_WPRESHUFFLE 1
#endif
#include "../kernels/src/block_i4_128_quant.hip"
#include "../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip"

#define HIP_OK(x) do { hipError_t e=(x); if(e!=hipSuccess){std::fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); std::exit(1);} } while(0)

int main(int argc,char**argv){
  const bool down=argc>1;
  const int B=8192, M=down?5120:34816, K=down?17408:5120;
  const size_t groups=(size_t)K/256;
  const size_t wb=(size_t)M*groups*136;
  const size_t xb=(size_t)(K/128)*B*sizeof(block_i4_128);
  const size_t yb=(size_t)B*M*sizeof(float);
  char *w=nullptr; block_i4_128 *x=nullptr; float *y=nullptr;
  HIP_OK(hipMalloc(&w,wb)); HIP_OK(hipMalloc(&x,xb)); HIP_OK(hipMalloc(&y,yb));
  HIP_OK(hipMemset(w,0,wb)); HIP_OK(hipMemset(x,0,xb)); HIP_OK(hipMemset(y,0,yb));
  dim3 block(256), grid((M+127)/128,(B+127)/128);
  const size_t lds=BENCH_WP?12288:20480;
  for(int i=0;i<5;i++) gemm_mq4g256v2_residual_mmq_iu4_full_set<<<grid,block,lds>>>(w,x,y,M,K,B,0);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> us;
  for(int i=0;i<20;i++){
    hipEvent_t a,b; HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
    HIP_OK(hipEventRecord(a));
    gemm_mq4g256v2_residual_mmq_iu4_full_set<<<grid,block,lds>>>(w,x,y,M,K,B,0);
    HIP_OK(hipEventRecord(b)); HIP_OK(hipEventSynchronize(b));
    float ms=0; HIP_OK(hipEventElapsedTime(&ms,a,b)); us.push_back(ms*1000.0f);
    HIP_OK(hipEventDestroy(a)); HIP_OK(hipEventDestroy(b));
  }
  std::sort(us.begin(),us.end()); double med=(us[9]+us[10])*0.5;
  double mean=0; for(float v:us)mean+=v; mean/=us.size();
  const double tf=(2.0*(double)B*M*K)/(med*1e6);
  int occ=0; HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ,gemm_mq4g256v2_residual_mmq_iu4_full_set,256,lds));
  std::printf("RESULT,%s,%s,B=%d,M=%d,K=%d,median_us=%.3f,mean_us=%.3f,tflops=%.3f,occ=%d,lds=%zu\n",BENCH_WP?"wp":"baseline",down?"down":"gate_up",B,M,K,med,mean,tf,occ,lds);
  HIP_OK(hipFree(w));HIP_OK(hipFree(x));HIP_OK(hipFree(y));
}
