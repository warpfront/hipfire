// Planning-only instruction-shape probe. NOT a correct GDN implementation.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
typedef _Float16 __attribute__((ext_vector_type(8))) h8;
typedef float __attribute__((ext_vector_type(8))) f8;
#define CK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);} }while(0)
__device__ __forceinline__ h8 halfv(f8 x) { return __builtin_convertvector(x,h8); }
__device__ __forceinline__ f8 floatv(h8 x) { return __builtin_convertvector(x,f8); }
__device__ __forceinline__ f8 mma(h8 a,h8 b,f8 c) {return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c);}
template<int SF,bool HALF,bool SOLVE> __global__ __launch_bounds__(256) void probe(float* out,const float* seed,int chunks) {
 extern __shared__ float lds[];
 const int tid=threadIdx.x, lane=tid&31, kg=lane>>4;
 f8 sf[SF]; h8 sh[SF]; h8 k,q;
 #pragma unroll
 for(int j=0;j<8;j++){k[j]=(_Float16)(seed[(tid+j)&255]*0.001f);q[j]=(_Float16)(seed[(tid+j+9)&255]*0.001f);}
 #pragma unroll
 for(int b=0;b<SF;b++) {
  #pragma unroll
  for(int j=0;j<8;j++){if constexpr(HALF)sh[b][j]=(_Float16)(seed[tid]*0.01f+(b+j)*0.0001f);else sf[b][j]=seed[tid]*0.01f+(b+j)*0.0001f;}
 }
 if constexpr(SOLVE) {for(int x=tid;x<4096;x+=256)lds[x]=(x%64<=x/64?0.0f:seed[tid]*0.0001f);__syncthreads();}
 for(int ci=0;ci<chunks;ci++) {
  f8 gram={0,0,0,0,0,0,0,0};
  #pragma unroll
  for(int t=0;t<32;t++)gram=mma(k,q,gram); // KK+QK dense packet quota per wave.
  f8 delta[4];
  #pragma unroll
  for(int t=0;t<4;t++) {
   f8 d=gram+(float)(t+1)*0.00001f; // distinct chains: prohibit compiler CSE
   #pragma unroll
   for(int b=0;b<SF;b++)d=mma(k,HALF?sh[b]:halfv(sf[b]),d);
   delta[t]=d*0.00001f;
  }
  if constexpr(SOLVE) {
   // 64-row unit-lower rank-update solve; delta row distribution is
   // row=t*16+kg*8+j, one value column per lane&15. State stays live.
   #pragma clang loop unroll(disable)
   for(int i=0;i<64;i++) {
    float v=0;
    #pragma unroll
    for(int t=0;t<4;t++) {
     #pragma unroll
     for(int j=0;j<8;j++) if(t*16+kg*8+j==i)v=delta[t][j];
    }
    v+=__shfl_xor(v,16);
    float l0=lds[i*64+lane],l1=lds[i*64+32+lane];
    #pragma unroll
    for(int t=0;t<4;t++) {
     #pragma unroll
     for(int j=0;j<8;j++) {int r=t*16+kg*8+j;float coeff=__shfl(t<2?l0:l1,r&31);delta[t][j]-=coeff*v;}
    }
    asm volatile("" : "+v"(delta[0]), "+v"(delta[1]), "+v"(delta[2]), "+v"(delta[3]) : : "memory");
   }
  }
  float sink=0;
  #pragma unroll
  for(int t=0;t<4;t++) {
   f8 d=gram*(float)(t+1); // distinct output tile chains
   #pragma unroll
   for(int b=0;b<SF;b++)d=mma(q,HALF?sh[b]:halfv(sf[b]),d); // S Q
   #pragma unroll
   for(int b=0;b<4;b++)d=mma(k,halfv(delta[b]),d); // M delta
   #pragma unroll
   for(int j=0;j<8;j++)sink+=d[j];
  }
  #pragma unroll
  for(int b=0;b<SF;b++) {
   f8 c=HALF?floatv(sh[b]):sf[b]; c*=0.99f;
   #pragma unroll
   for(int t=0;t<4;t++)c=mma(k,halfv(delta[t]),c); // carry K^T delta
   if constexpr(HALF)sh[b]=halfv(c);else sf[b]=c;
  }
  q[0]=(_Float16)((sink>1e20f)?0.002f:0.001f); // make output products observable
 }
 float sum=0;
 #pragma unroll
 for(int b=0;b<SF;b++) {
  #pragma unroll
  for(int j=0;j<8;j++)sum+=HALF?(float)sh[b][j]:sf[b][j];
 }
 out[blockIdx.x*256+tid]=sum;
}
template<int SF,bool HALF,bool SOLVE> void run(const char* name,float* out,float* seed,int blocks,int chunks,int ldsbytes,hipEvent_t a,hipEvent_t b) {
 auto f=probe<SF,HALF,SOLVE>;
 hipFuncAttributes attr;CK(hipFuncGetAttributes(&attr,(const void*)f));int occupancy=0;CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,f,256,ldsbytes));
 for(int i=0;i<3;i++)f<<<blocks,256,ldsbytes>>>(out,seed,chunks);CK(hipDeviceSynchronize());
 double us[7];for(int r=0;r<7;r++){CK(hipEventRecord(a));for(int j=0;j<20;j++)f<<<blocks,256,ldsbytes>>>(out,seed,chunks);CK(hipEventRecord(b));CK(hipEventSynchronize(b));float ms;CK(hipEventElapsedTime(&ms,a,b));us[r]=ms*1000/20;}
 std::sort(us,us+7);printf("%s blocks=%d chunks=%d lds=%d regs=%d local=%zu static_shared=%zu activeWG/CU=%d min_us=%.3f median_us=%.3f max_us=%.3f norm_us/head/chunk=%.6f\n",name,blocks,chunks,ldsbytes,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,occupancy,us[0],us[3],us[6],us[3]/blocks/chunks);
}
int main(int argc,char**argv){int chunks=argc>1?atoi(argv[1]):8;hipDeviceProp_t p;CK(hipGetDeviceProperties(&p,0));printf("device=%s arch=%s CUs=%d clock_kHz=%d shared/CU=%zu warp=%d chunks=%d\n",p.name,p.gcnArchName,p.multiProcessorCount,p.clockRate,p.maxSharedMemoryPerMultiProcessor,p.warpSize,chunks);float *d,*s;CK(hipMalloc(&d,256*384*sizeof(float)));CK(hipMalloc(&s,256*sizeof(float)));float seed[256];for(int i=0;i<256;i++)seed[i]=1+i*0.0001f;CK(hipMemcpy(s,seed,sizeof(seed),hipMemcpyHostToDevice));hipEvent_t a,b;CK(hipEventCreate(&a));CK(hipEventCreate(&b));
 for(int lds: {0,28672}) {run<8,false,false>("f32-full",d,s,48,chunks,lds,a,b);run<8,true,false>("f16-full",d,s,48,chunks,lds,a,b);run<4,false,false>("f32-split2",d,s,96,chunks,lds,a,b);run<4,true,false>("f16-split2",d,s,96,chunks,lds,a,b);}
 run<8,false,true>("f32-full-solve",d,s,48,chunks,28672,a,b);run<8,true,true>("f16-full-solve",d,s,48,chunks,28672,a,b);run<4,false,true>("f32-split2-solve",d,s,96,chunks,28672,a,b);
 run<4,true,true>("f16-split2-solve",d,s,96,chunks,28672,a,b);
 run<2,false,true>("f32-split4-solve",d,s,192,chunks,28672,a,b);run<2,true,true>("f16-split4-solve",d,s,192,chunks,28672,a,b);
 CK(hipFree(d));CK(hipFree(s));CK(hipEventDestroy(a));CK(hipEventDestroy(b));}
