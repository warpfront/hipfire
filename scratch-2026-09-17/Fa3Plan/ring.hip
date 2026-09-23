#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
__device__ __forceinline__ void await_epoch(unsigned* flag,unsigned e){
  if((threadIdx.x&31)==0)while(__hip_atomic_load(flag,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_WORKGROUP)!=e) __builtin_amdgcn_s_sleep(1);
  __syncwarp();
}
__device__ __forceinline__ void release_epoch(unsigned* flag,unsigned e){
  __syncwarp();
  if((threadIdx.x&31)==0)__hip_atomic_store(flag,e,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_WORKGROUP);
}
__global__ __launch_bounds__(256,1) void ring(const unsigned* src,unsigned* out,int tiles){
  extern __shared__ unsigned lds[];
  const int wave=threadIdx.x/32,lane=threadIdx.x%32;
  unsigned* ready=lds+10240; unsigned* done=ready+4;
  if(threadIdx.x<16)ready[threadIdx.x]=0;
  __syncthreads();
  const bool producer=(wave&3)==3;
  const int p=wave/4,c=(wave/4)*3+(wave&3);
  unsigned acc=0;
  for(int t=0;t<tiles;t++){
    const int slot=t&1;
    unsigned* payload=lds+slot*4096;
    if(producer){
      if(t>=2)for(int j=0;j<6;j++)await_epoch(done+slot*6+j,t-1);
      for(int j=lane;j<2048;j+=32)payload[p*2048+j]=src[t*4096+p*2048+j];
      release_epoch(ready+slot*2+p,t+1);
    }else{
      for(int j=0;j<2;j++)await_epoch(ready+slot*2+j,t+1);
      for(int j=lane;j<4096;j+=32)acc+=payload[j];
      release_epoch(done+slot*6+c,t+1);
    }
  }
  __syncthreads();
  if(!producer)out[blockIdx.x*192+c*32+lane]=acc;
}
int main(){
  for(int tiles:{1,2,3,7,65,513}){
    std::vector<unsigned> input(tiles*4096),output(64*192),expected(32);
    for(size_t i=0;i<input.size();i++){input[i]=(unsigned)((i*1664525+1013904223u)^((i/4096)*97));expected[i%32]+=input[i];}
    unsigned *s,*d;CK(hipMalloc(&s,input.size()*4));CK(hipMalloc(&d,output.size()*4));CK(hipMemcpy(s,input.data(),input.size()*4,hipMemcpyHostToDevice));
    ring<<<64,256,41216>>>(s,d,tiles);CK(hipGetLastError());CK(hipMemcpy(output.data(),d,output.size()*4,hipMemcpyDeviceToHost));
    int bad=0;for(size_t i=0;i<output.size();i++)bad+=output[i]!=expected[i%32];
    printf("tiles=%d blocks=64 consumers=6 producers=2 LDS=41216 mismatches=%d\n",tiles,bad);if(bad)return 1;
    CK(hipFree(s));CK(hipFree(d));
  }
}
