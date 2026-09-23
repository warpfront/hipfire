#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
using I2=int __attribute__((ext_vector_type(2)));
using F8=float __attribute__((ext_vector_type(8)));
#define CK(x) do {auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(2);}}while(0)
template<int MODE> __device__ __forceinline__ float partner(float x){
 if constexpr(MODE==0) return __shfl_xor(x,16,32);
 int b=__builtin_bit_cast(int,x);
 return __builtin_bit_cast(float,__builtin_amdgcn_permlanex16(b,b,0x76543210,0xfedcba98,false,false));
}
// Each iteration is one KT16, D256, one 16-query wave. Register operands are
// deliberately reused: this measures an optimistic math floor, not a KV cache.
// Empty read/write asm constraints prevent CSE/loop hoisting, emit no packets.
// MODE 0 exact recurrence + generic shuffle; 1 exact + fixed partner;
// 2 reciprocal approximation + fixed partner; 3 matrix-only; 4 softmax-only.
template<int MODE> __global__ __launch_bounds__(512,1) void floor_kernel(const unsigned* seed,float* out,int iters){
 extern __shared__ unsigned lds[];
 int tid=threadIdx.x, lane=tid&31, id=blockIdx.x*blockDim.x+tid;
 lds[tid]=tid; __syncthreads();
 I2 q={(int)seed[2*lane],(int)seed[2*lane+1]}, k=q, v=q;
 F8 o[16];
 #pragma unroll
 for(int d=0;d<16;d++){
  #pragma unroll
  for(int j=0;j<8;j++)o[d][j]=0;
  asm volatile("" : "+v"(o[d]));
 }
 float m=-INFINITY,l=0,bprev=1;
 F8 s={0,0,0,0,0,0,0,0};
 for(int t=0;t<iters;t++){
  s=(F8){0,0,0,0,0,0,0,0};
  if constexpr(MODE!=4){
   #pragma unroll
   for(int d=0;d<16;d++){
    asm volatile("" : "+v"(q), "+v"(k), "+v"(s));
    s=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(k,q,s);
   }
  }else{
   #pragma unroll
   for(int j=0;j<8;j++)s[j]=0.03f*j+0.01f*lane+0.0001f*t;
  }
  I2 p=q;
  if constexpr(MODE!=3){
   float mx=-INFINITY;
   #pragma unroll
   for(int j=0;j<8;j++){
    float sk=0.001f*(j+1+((lane>>4)*8));
    s[j]=(s[j]*0.125f)*sk;
    mx=fmaxf(mx,s[j]);
   }
   mx=fmaxf(mx,partner<MODE>(mx));
   float mn=fmaxf(m,mx), alpha=m==-INFINITY?0:__expf(m-mn);m=mn;
   float sum=0,wm=0;
   #pragma unroll
   for(int j=0;j<8;j++){
    float e=mn==-INFINITY?0:__expf(s[j]-mn);
    sum+=e;s[j]=e*(0.01f*(j+1+((lane>>4)*8)));wm=fmaxf(wm,s[j]);
   }
   sum+=partner<MODE>(sum);l=l*alpha+sum;
   wm=fmaxf(wm,partner<MODE>(wm));
   float bn=bprev;if(wm>0)bn=fmaxf(wm/448.f,0x1p-64f);
   float rho;F8 z;
   if constexpr(MODE==2){
    float inv=__builtin_amdgcn_rcpf(bn);rho=(alpha*bprev)*inv;
    #pragma unroll
    for(int j=0;j<8;j++)z[j]=s[j]*inv;
   }else{
    rho=(alpha*bprev)/bn;
    #pragma unroll
    for(int j=0;j<8;j++)z[j]=s[j]/bn;
   }
   unsigned p0=0,p1=0;
   p0=__builtin_amdgcn_cvt_pk_fp8_f32(z[0],z[1],p0,false);
   p0=__builtin_amdgcn_cvt_pk_fp8_f32(z[2],z[3],p0,true);
   p1=__builtin_amdgcn_cvt_pk_fp8_f32(z[4],z[5],p1,false);
   p1=__builtin_amdgcn_cvt_pk_fp8_f32(z[6],z[7],p1,true);
   p=(I2){(int)p0,(int)p1};
   #pragma unroll
   for(int d=0;d<16;d++){asm volatile("" : "+v"(o[d]));o[d]*=rho;}
   bprev=bn;
  }
  if constexpr(MODE!=4){
   #pragma unroll
   for(int d=0;d<16;d++){
    asm volatile("" : "+v"(v), "+v"(o[d]));
    o[d]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(v,p,o[d]);
   }
  }
 }
 // All 128 O components and state observable; no checksum-only equality claim.
 #pragma unroll
 for(int d=0;d<16;d++){
  #pragma unroll
  for(int j=0;j<8;j++)out[id*140+d*8+j]=o[d][j]*(MODE==3?1.f:bprev);
 }
 #pragma unroll
 for(int j=0;j<8;j++)out[id*140+128+j]=s[j];
 out[id*140+136]=m;out[id*140+137]=l;out[id*140+138]=bprev;out[id*140+139]=(float)lds[tid];
}
void launch(int mode,int blocks,int threads,int lds,const unsigned* seed,float* out,int iters){
 #define L(M) case M:floor_kernel<M><<<blocks,threads,lds>>>(seed,out,iters);break
 switch(mode){L(0);L(1);L(2);L(3);L(4);}
 #undef L
 CK(hipGetLastError());
}
int main(int argc,char**argv){
 int blocks=argc>1?atoi(argv[1]):64,threads=argc>2?atoi(argv[2]):128,iters=argc>3?atoi(argv[3]):128,lds=argc>4?atoi(argv[4]):32768;
 hipDeviceProp_t prop;CK(hipGetDeviceProperties(&prop,0));
 printf("device=%s CUs=%d blocks=%d threads=%d KT16iters=%d LDS=%d\n",prop.gcnArchName,prop.multiProcessorCount,blocks,threads,iters,lds);
 int active0,active1,active2;
 CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active0,floor_kernel<0>,threads,lds));
 CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active1,floor_kernel<1>,threads,lds));
 CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active2,floor_kernel<2>,threads,lds));
 printf("occupancy_API_resident_WG_per_reported_MP exact=%d partner=%d rcp=%d\n",active0,active1,active2);
 unsigned h[64];for(int i=0;i<64;i++){h[i]=0;for(int j=0;j<4;j++)h[i]|=(unsigned)(0x20+((i*13+j*3)%24))<<(8*j);}
 unsigned*seed;float*out;size_t n=(size_t)blocks*threads*140;CK(hipMalloc(&seed,sizeof(h)));CK(hipMemcpy(seed,h,sizeof(h),hipMemcpyHostToDevice));CK(hipMalloc(&out,n*4));
 auto go=[&](int mode){launch(mode,blocks,threads,lds,seed,out,iters);};
 for(int w=0;w<4;w++)for(int m=0;m<5;m++)go(m);CK(hipDeviceSynchronize());
 hipEvent_t a,b;CK(hipEventCreate(&a));CK(hipEventCreate(&b));std::vector<double>ts[5];
 for(int rep=0;rep<15;rep++)for(int k=0;k<5;k++){
  int m=rep%2?k:4-k;CK(hipEventRecord(a));go(m);CK(hipEventRecord(b));CK(hipEventSynchronize(b));float ms;CK(hipEventElapsedTime(&ms,a,b));ts[m].push_back(ms*1000);printf("sample,%d,%d,%.6f\n",rep,m,ms*1000);
 }
 const char* names[]={"exact_shfl","exact_permlane","approx_rcp","matrix","softmax"};
 for(int m=0;m<5;m++){std::sort(ts[m].begin(),ts[m].end());printf("median,%s,%.6f,min=%.6f,max=%.6f\n",names[m],ts[m][7],ts[m][0],ts[m][14]);}
 std::vector<float>base(n),x(n);go(0);CK(hipMemcpy(base.data(),out,n*4,hipMemcpyDeviceToHost));
 int err=0;
 for(int mode=1;mode<3;mode++){
  go(mode);CK(hipMemcpy(x.data(),out,n*4,hipMemcpyDeviceToHost));size_t bad=0,nonfinite=0;double maxabs=0,maxrel=0;
  for(size_t i=0;i<n;i++){bad+=memcmp(&base[i],&x[i],4)!=0;nonfinite+=!std::isfinite(base[i])||!std::isfinite(x[i]);maxabs=std::max(maxabs,fabs((double)base[i]-x[i]));maxrel=std::max(maxrel,fabs((double)base[i]-x[i])/std::max(1e-30,fabs((double)base[i])));}
  printf("compare,%s,components=%zu,bitdiff=%zu,nonfinite=%zu,maxabs=%.9g,maxrel=%.9g\n",names[mode],n,bad,nonfinite,maxabs,maxrel);if(mode==1&&(bad||nonfinite))err=1;
 }
 CK(hipFree(seed));CK(hipFree(out));return err;
}
