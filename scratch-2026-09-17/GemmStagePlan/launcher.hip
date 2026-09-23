#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#define CK(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);}}while(0)
#define DGU(n) extern "C" __global__ void n(const char*,const char*,const unsigned char*,const float*,const float*,float*,float*,int,int,int,int);
#define DRE(n) extern "C" __global__ void n(const char*,const unsigned char*,const float*,const float*,float*,int,int,int);
DGU(gu_full) DRE(re_full)
DGU(gu_nofold) DRE(re_nofold)
DGU(gu_noscale) DRE(re_noscale)
DGU(gu_noexpand) DRE(re_noexpand)
DGU(gu_latepref) DRE(re_latepref)
DGU(gu_nostage) DRE(re_nostage)
DGU(gu_nostage_nb) DRE(re_nostage_nb)
DGU(gu_frag) DRE(re_frag)
DGU(gu_nocompute) DRE(re_nocompute)
DGU(gu_nocompute_nb) DRE(re_nocompute_nb)
static void* alloc(size_t n,int val){void*p;CK(hipMalloc(&p,n));CK(hipMemset(p,val,n));return p;}
static uint64_t hash(const float*p,size_t n){std::vector<unsigned char>v(n*4);CK(hipMemcpy(v.data(),p,n*4,hipMemcpyDeviceToHost));uint64_t h=14695981039346656037ull;for(auto b:v){h^=b;h*=1099511628211ull;}return h;}
template<class F> void bench(const char*name,F f,const void* kernel,float* y,size_t size,bool residual){
 hipFuncAttributes at;CK(hipFuncGetAttributes(&at,kernel));hipEvent_t a,b;CK(hipEventCreate(&a));CK(hipEventCreate(&b));std::vector<float>t;
 for(int i=0;i<18;i++){if(residual)CK(hipMemset(y,0,size));CK(hipEventRecord(a));f();CK(hipEventRecord(b));CK(hipEventSynchronize(b));float ms;CK(hipEventElapsedTime(&ms,a,b));if(i>=3)t.push_back(ms*1000);}
 std::sort(t.begin(),t.end());if(residual)CK(hipMemset(y,0,size));f();CK(hipDeviceSynchronize());
 printf("%s median_us=%.3f min_us=%.3f max_us=%.3f regs=%d local=%zu hash=%016lx\n",name,t[7],t.front(),t.back(),at.numRegs,at.localSizeBytes,hash(y,size/4));fflush(stdout);
 CK(hipEventDestroy(a));CK(hipEventDestroy(b));
}
int main(int argc,char**argv){int N=argc>1?atoi(argv[1]):512;int rounds=argc>2?atoi(argv[2]):2;int G=17408,U=17408,KG=5120,M=5120,KR=17408;
 char*Ag=(char*)alloc((size_t)G*(KG/256)*136,0x11),*Au=(char*)alloc((size_t)U*(KG/256)*136,0x11),*Aw=(char*)alloc((size_t)M*(KR/256)*136,0x11);
 auto Xg=(unsigned char*)alloc((size_t)N*KG,0x31),Xr=(unsigned char*)alloc((size_t)N*KR,0x31);
 auto Sg=(float*)alloc((size_t)N*(KG/128)*4,0x3e),Sr=(float*)alloc((size_t)N*(KR/128)*4,0x3e),Rg=(float*)alloc((size_t)N*4,0x3e),Rr=(float*)alloc((size_t)N*4,0x3e);
 auto Yg=(float*)alloc((size_t)N*G*4,0),Yu=(float*)alloc((size_t)N*U*4,0),Yr=(float*)alloc((size_t)N*M*4,0);
 dim3 gg((G+U+127)/128,(N+127)/128),gr((M+127)/128,(N+127)/128);
#define GU(nm) bench(#nm,[&]{hipLaunchKernelGGL(nm,gg,256,19968,0,Ag,Au,Xg,Sg,Rg,Yg,Yu,G,U,KG,N);},(const void*)nm,Yg,(size_t)N*G*4,false)
#define RE(nm) bench(#nm,[&]{hipLaunchKernelGGL(nm,gr,256,19968,0,Aw,Xr,Sr,Rr,Yr,M,KR,N);},(const void*)nm,Yr,(size_t)N*M*4,true)
 for(int round=0;round<rounds;round++){printf("N=%d round=%d\n",N,round);
 GU(gu_full); RE(re_full);
 GU(gu_nofold); RE(re_nofold);
 GU(gu_noscale); RE(re_noscale);
 GU(gu_noexpand); RE(re_noexpand);
 GU(gu_latepref); RE(re_latepref);
 GU(gu_nostage); RE(re_nostage);
 GU(gu_nostage_nb); RE(re_nostage_nb);
 GU(gu_frag); RE(re_frag);
 GU(gu_nocompute); RE(re_nocompute);
 GU(gu_nocompute_nb); RE(re_nocompute_nb);
 }
}
