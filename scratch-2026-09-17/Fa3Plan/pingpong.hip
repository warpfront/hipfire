#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
using I2 = int __attribute__((ext_vector_type(2)));
using F8 = float __attribute__((ext_vector_type(8)));
#define CK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)

// Equal work: every wave runs the same WMMA and softmax-like recurrences.
// SERIAL aligns the phase, PING offsets waves 4..7; both use two WG barriers.
// 64 KiB dynamic LDS reserves the proposed ring's occupancy footprint.
template<int CHAINS> __device__ __forceinline__ void matrix(F8 (&c)[CHAINS], I2 a, I2 b) {
#pragma unroll 1
  for(int i=0;i<32;i++) {
#pragma unroll
    for(int j=0;j<CHAINS;j++) c[j]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c[j]);
  }
}
__device__ __forceinline__ void scalar(F8 &x, F8 (&o)[16]) {
  float mx=-INFINITY;
#pragma unroll
  for(int j=0;j<8;j++) mx=fmaxf(mx,x[j]);
  mx=fmaxf(mx,__shfl_xor(mx,16,32));
  float sum=0;
#pragma unroll
  for(int j=0;j<8;j++){ x[j]=__expf(x[j]-mx); sum+=x[j]; }
  sum+=__shfl_xor(sum,16,32);
  float wm=0;
#pragma unroll
  for(int j=0;j<8;j++){ x[j]*=0.001f*(j+1); wm=fmaxf(wm,x[j]); }
  wm=fmaxf(wm,__shfl_xor(wm,16,32));
  float bn=fmaxf(wm/448.0f,0x1p-64f);
  unsigned p0=0,p1=0;
  p0=__builtin_amdgcn_cvt_pk_fp8_f32(x[0]/bn,x[1]/bn,p0,false);
  p0=__builtin_amdgcn_cvt_pk_fp8_f32(x[2]/bn,x[3]/bn,p0,true);
  p1=__builtin_amdgcn_cvt_pk_fp8_f32(x[4]/bn,x[5]/bn,p1,false);
  p1=__builtin_amdgcn_cvt_pk_fp8_f32(x[6]/bn,x[7]/bn,p1,true);
  const float rho=0.999f+(float)((p0^p1)&255)*0x1p-20f;
#pragma unroll
  for(int d=0;d<16;d++) o[d]*=rho;
#pragma unroll
  for(int j=0;j<8;j++) x[j]=x[j]/sum+0.01f*j;
}
template<int N> __device__ __forceinline__ void phase_barrier(F8 (&c)[N],F8 &x,F8 (&o)[16]) {
#pragma unroll
  for(int j=0;j<N;j++) asm volatile("" : "+v"(c[j]) :: "memory");
  asm volatile("" : "+v"(x) :: "memory");
#pragma unroll
  for(int j=0;j<16;j++) asm volatile("" : "+v"(o[j]) :: "memory");
  __syncthreads();
#pragma unroll
  for(int j=0;j<N;j++) asm volatile("" : "+v"(c[j]) :: "memory");
  asm volatile("" : "+v"(x) :: "memory");
#pragma unroll
  for(int j=0;j<16;j++) asm volatile("" : "+v"(o[j]) :: "memory");
}
template<int MODE,int CHAINS> __global__ __launch_bounds__(256,1) void run(float *out,unsigned *ids,int iters) {
  extern __shared__ unsigned lds[];
  const int tid=threadIdx.x, wave=tid/32, lane=tid%32;
  unsigned hw=0;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
  if(lane==0) ids[blockIdx.x*8+wave]=hw;
  lds[tid]=tid;
  __syncthreads();
  I2 a={(int)0x18181818,(int)0x18181818};
  I2 b={(int)0x10101010,(int)0x10101010};
  F8 c[CHAINS]; F8 x; F8 o[16];
#pragma unroll
  for(int q=0;q<CHAINS;q++) c[q]=(F8){0,0,0,0,0,0,0,0};
#pragma unroll
  for(int j=0;j<8;j++) x[j]=0.1f*j+0.001f*lane;
#pragma unroll
  for(int d=0;d<16;d++)
#pragma unroll
    for(int j=0;j<8;j++) o[d][j]=1.f+0.01f*d+0.001f*j;
  for(int t=0;t<iters;t++) {
    if constexpr(MODE==0) matrix(c,a,b);
    if constexpr(MODE==1) scalar(x,o);
    if constexpr(MODE==2 || MODE==3) {
      if(MODE==2 || wave<4) matrix(c,a,b); else scalar(x,o);
      phase_barrier(c,x,o);
      if(MODE==2 || wave<4) scalar(x,o); else matrix(c,a,b);
      phase_barrier(c,x,o);
    }
    if constexpr(MODE==4) { if(wave<4) matrix(c,a,b); else scalar(x,o); }
    if constexpr(MODE==5) { matrix(c,a,b); scalar(x,o); }
  }
  float z=0;
#pragma unroll
  for(int q=0;q<CHAINS;q++)
#pragma unroll
    for(int j=0;j<8;j++) z+=c[q][j];
#pragma unroll
  for(int d=0;d<16;d++)
#pragma unroll
    for(int j=0;j<8;j++) z+=o[d][j];
#pragma unroll
  for(int j=0;j<8;j++) z+=x[j];
  out[blockIdx.x*256+tid]=z+(float)lds[tid]*0x1p-24f;
}

template<int C> void launch(int mode,int blocks,float* o,unsigned* ids,int iters,int lds) {
#define L(M) case M: run<M,C><<<blocks,256,lds>>>(o,ids,iters); break
  switch(mode){L(0);L(1);L(2);L(3);L(4);L(5);}
#undef L
  CK(hipGetLastError());
}
int main(int argc,char** argv){
  int blocks=argc>1?atoi(argv[1]):64, iters=argc>2?atoi(argv[2]):128;
  int chains=argc>3?atoi(argv[3]):1, lds=argc>4?atoi(argv[4]):65536;
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p,0));
  printf("device=%s CUs=%d blocks=%d threads=256 iters=%d chains=%d LDS=%d\n",p.gcnArchName,p.multiProcessorCount,blocks,iters,chains,lds);
  float *d;unsigned* ids;CK(hipMalloc(&d,blocks*256*sizeof(float)));CK(hipMalloc(&ids,blocks*8*sizeof(unsigned)));
  auto go=[&](int m){if(chains==1)launch<1>(m,blocks,d,ids,iters,lds);else launch<4>(m,blocks,d,ids,iters,lds);};
  for(int w=0;w<4;w++) for(int m=0;m<6;m++)go(m); CK(hipDeviceSynchronize());
  hipEvent_t a,b;CK(hipEventCreate(&a));CK(hipEventCreate(&b));
  std::vector<double> ts[6];
  for(int rep=0;rep<15;rep++)for(int k=0;k<6;k++){
    int m=rep%2?k:5-k;
    CK(hipEventRecord(a));go(m);CK(hipEventRecord(b));CK(hipEventSynchronize(b));float ms;CK(hipEventElapsedTime(&ms,a,b));
    ts[m].push_back(ms*1000);printf("sample,%d,%d,%.6f\n",rep,m,ms*1000);
  }
  const char* names[]={"W","V","SERIAL","PING","MIX_HALF","WV"};
  double med[6]; for(int m=0;m<6;m++){std::sort(ts[m].begin(),ts[m].end());med[m]=ts[m][7];printf("median,%s,%.6f,min=%.6f,max=%.6f\n",names[m],med[m],ts[m][0],ts[m][14]);}
  printf("ratios ping_serial=%.6f ping_wv=%.6f mix_serial_est=%.6f hidden_fraction=%.6f\n",med[3]/med[2],med[3]/med[5],2*med[4]/(med[0]+med[1]),(med[0]+med[1]-2*med[4])/std::min(med[0],med[1]));
  std::vector<float> ser(blocks*256),ping(blocks*256),wv(blocks*256);
  go(2);CK(hipMemcpy(ser.data(),d,ser.size()*4,hipMemcpyDeviceToHost));
  go(3);CK(hipMemcpy(ping.data(),d,ping.size()*4,hipMemcpyDeviceToHost));
  go(5);CK(hipMemcpy(wv.data(),d,wv.size()*4,hipMemcpyDeviceToHost));
  int bad=0;for(size_t i=0;i<ser.size();i++)if(!std::isfinite(ser[i])||memcmp(&ser[i],&ping[i],4)||memcmp(&ser[i],&wv[i],4))bad++;
  std::vector<unsigned> hid(blocks*8);CK(hipMemcpy(hid.data(),ids,hid.size()*4,hipMemcpyDeviceToHost));
  for(int block=0;block<std::min(blocks,8);block++){printf("hwid block=%d",block);for(int w=0;w<8;w++)printf(" w%d=%08x",w,hid[block*8+w]);puts("");}
  printf("exact_serial_ping_wv mismatches=%d checksum=%.9g\n",bad,ser[0]);
  CK(hipFree(d));CK(hipFree(ids));return bad?1:0;
}
