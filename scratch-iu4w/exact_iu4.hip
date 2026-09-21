#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define IU4_SYMMETRIC_FOLD 1
#if BENCH_WP
#define IU4_WPRESHUFFLE 1
#endif
#include "../kernels/src/block_i4_128_quant.hip"
#include "../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip"
#define HIP_OK(x) do{auto e=(x);if(e){std::fprintf(stderr,"HIP %s\n",hipGetErrorString(e));std::exit(1);}}while(0)
static size_t pd(size_t r,size_t n,size_t k){size_t l=16*((n>>4)&1)+(r&15);return ((((r>>4)*(k>>6)+(n>>6))*32+l)*16)+((n>>5)&1)*8+((n&15)>>1);}
static std::vector<unsigned char> perm(const std::vector<unsigned char>&s,int m,int k){size_t g=k/256,pb=(size_t)m*k/2,rt=m/16;std::vector<unsigned char>o(s.size());for(int r=0;r<m;r++){for(int pc=0;pc<k/2;pc++){int n=pc*2;o[pd(r,n,k)]=s[((size_t)r*g+n/256)*136+8+(n&255)/2];}for(int h=0;h<k/128;h++){size_t q=((size_t)r*g+h/2)*136+(h&1)*4;int pl[4]={1,0,2,3};for(int b=0;b<4;b++)o[pb+(((size_t)h*rt+r/16)*4+pl[b])*16+(r&15)]=s[q+b];}}return o;}
int main(){const int B=128,M=160,K=256;size_t g=1,wb=(size_t)M*136,xb=(size_t)(K/128)*B*sizeof(block_i4_128),yb=(size_t)B*M*4;std::vector<unsigned char>w0(wb);for(int r=0;r<M;r++){for(int h=0;h<2;h++){size_t p=(size_t)r*136+h*4;w0[p]=0;w0[p+1]=0x3c;w0[p+2]=0;w0[p+3]=0;}for(int i=0;i<128;i++)w0[(size_t)r*136+8+i]=(unsigned char)((r*17+i*29+3)&255);}auto wh=BENCH_WP?perm(w0,M,K):w0;std::vector<block_i4_128>xh((K/128)*B);for(size_t i=0;i<xh.size();i++){xh[i].d=0.125f;xh[i].s=0;for(int q=0;q<64;q++){unsigned lo=(i*7+q*3)&15,hi=(i*11+q*5+1)&15;xh[i].qs[q]=(unsigned char)(lo|(hi<<4));xh[i].s+=(lo>=8?(int)lo-16:(int)lo)+(hi>=8?(int)hi-16:(int)hi);}}char*w;block_i4_128*x;float*y;HIP_OK(hipMalloc(&w,wb));HIP_OK(hipMalloc(&x,xb));HIP_OK(hipMalloc(&y,yb));HIP_OK(hipMemcpy(w,wh.data(),wb,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(x,xh.data(),xb,hipMemcpyHostToDevice));HIP_OK(hipMemset(y,0,yb));size_t lds=BENCH_WP?12288:20480;gemm_mq4g256v2_residual_mmq_iu4_full_set<<<dim3(2,1),dim3(256),lds>>>(w,x,y,M,K,B,0);HIP_OK(hipDeviceSynchronize());std::vector<unsigned char>out(yb);HIP_OK(hipMemcpy(out.data(),y,yb,hipMemcpyDeviceToHost));uint64_t h=1469598103934665603ull;for(auto v:out){h^=v;h*=1099511628211ull;}FILE*f=fopen(BENCH_WP?"exact_wp.bin":"exact_baseline.bin","wb");fwrite(out.data(),1,out.size(),f);fclose(f);std::printf("EXACT,%s,fnv=%016llx\n",BENCH_WP?"wp":"baseline",(unsigned long long)h);}
