// Disposable direct launch of unchanged production writer, fwht3-K/q8-V FA2,
// q8/q8 FA2, and segmented 512-row fwht3 FA2. No production host gates.
// Build: hipcc -O3 --offload-arch=gfx1100 -std=c++17 fwht3_g0.hip q8_module.hip -o fwht3_g0
#define HIPFIRE_FA2_KT 32
#define HIPFIRE_FA2_KMODE 3
#include "../kernels/src/kv_cache_write_asym_k_fwht3_batched.hip"
#include "../kernels/src/kv_cache_write_q8_0_batched.hip"
#include "../kernels/src/attention_q8_0_fa2_gqa.gfx11.hip"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" __global__ void attention_q8_0_fa2_gqa_q8_probe(
    const _Float16*, const unsigned char*, const unsigned char*, float*,
    const int*, int, int, int, int, float);
extern "C" __global__ void attention_fa2_q_preconvert_gfx11(
    const float*, _Float16*, const float*, const float*, int, int);

#define HIP(x) do { hipError_t e=(x); if(e!=hipSuccess) { \
  fprintf(stderr,"%s:%d: %s: %s\n",__FILE__,__LINE__,#x,hipGetErrorString(e));exit(2); \
} } while(0)
constexpr int H=24, KV=4, D=256, QDIM=H*D, KVDIM=KV*D, WR=4096;
constexpr size_t GUARD=4096;

template<class T> T* alloc(size_t n) { T* p=nullptr; HIP(hipMalloc(&p,n*sizeof(T))); return p; }
__host__ __device__ static unsigned hash(unsigned x) {
  x ^= x>>16; x*=0x7feb352dU; x ^= x>>15; x*=0x846ca68bU; return x^(x>>16);
}
// A deterministic, nonconstant activation, generated on device to avoid
// transferring 2+ GiB of f32 cache source from the host on every fresh run.
__global__ void fill_f32(float* dst,int rows,int dims,int base,unsigned salt) {
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  size_t n=(size_t)rows*dims;
  if(i<n) {
    unsigned r=(unsigned)(i/dims)+(unsigned)base, d=(unsigned)(i%dims);
    unsigned z=hash(r*0x9e3779b1U ^ d*0x85ebca6bU ^ salt);
    dst[i]=(float(int(z%65537)-32768)/32768.f)*.12f;
  }
}
__global__ void fill_pos(int* p,int n,int base) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) p[i]=base+i;
}
static void check_guard(const void* p,size_t bytes,const char* label) {
  unsigned char x[GUARD]; HIP(hipMemcpy(x,(const char*)p+bytes,GUARD,hipMemcpyDeviceToHost));
  for(unsigned char c:x) if(c!=0xa5) {fprintf(stderr,"GUARD FAILED %s\n",label);exit(4);}
}
struct EventPair {
  hipEvent_t a[16],z[16];
  EventPair(){for(int i=0;i<16;i++){HIP(hipEventCreate(&a[i]));HIP(hipEventCreate(&z[i]));}}
  ~EventPair(){for(int i=0;i<16;i++){HIP(hipEventDestroy(a[i]));HIP(hipEventDestroy(z[i]));}}
};
struct Fixture {
  int start,batch,ctx;
  float *src_k,*src_v,*q,*sign1,*sign2,*out_f,*out_s,*out_q;
  _Float16 *q16f,*q16q;
  unsigned char *kf,*kq,*v;
  int *wpos,*pos;
  Fixture(int s,int b):start(s),batch(b),ctx(s+b),
      src_k(alloc<float>((size_t)WR*KVDIM)),src_v(alloc<float>((size_t)WR*KVDIM)),
      q(alloc<float>((size_t)b*QDIM)),sign1(alloc<float>(D)),sign2(alloc<float>(D)),
      out_f(alloc<float>((size_t)b*QDIM+GUARD/4)),out_s(alloc<float>((size_t)b*QDIM+GUARD/4)),
      out_q(alloc<float>((size_t)b*QDIM+GUARD/4)),
      q16f(alloc<_Float16>((size_t)b*QDIM)),q16q(alloc<_Float16>((size_t)b*QDIM)),
      kf(alloc<unsigned char>((size_t)ctx*400+GUARD)),
      kq(alloc<unsigned char>((size_t)ctx*1088+GUARD)),
      v(alloc<unsigned char>((size_t)ctx*1088+GUARD)),
      wpos(alloc<int>(WR)),pos(alloc<int>(b)) {
    float signs[2][D];
    for(int j=0;j<2;j++) for(int i=0;i<D;i++) signs[j][i]=(hash(unsigned(i*31+j*907+77))&1)?1.f:-1.f;
    HIP(hipMemcpy(sign1,signs[0],sizeof(signs[0]),hipMemcpyHostToDevice));
    HIP(hipMemcpy(sign2,signs[1],sizeof(signs[1]),hipMemcpyHostToDevice));
    HIP(hipMemset(kf,0xa5,(size_t)ctx*400+GUARD));
    HIP(hipMemset(kq,0xa5,(size_t)ctx*1088+GUARD));
    HIP(hipMemset(v,0xa5,(size_t)ctx*1088+GUARD));
    const size_t outbytes=(size_t)batch*QDIM*4;
    HIP(hipMemset(out_f,0xa5,outbytes+GUARD)); HIP(hipMemset(out_s,0xa5,outbytes+GUARD));
    HIP(hipMemset(out_q,0xa5,outbytes+GUARD));
    for(int p=0;p<ctx;p+=WR) {
      int n=std::min(WR,ctx-p);
      size_t count=(size_t)n*KVDIM;
      fill_f32<<<dim3((count+255)/256),dim3(256)>>>(src_k,n,KVDIM,p,0xa8e401U);
      fill_f32<<<dim3((count+255)/256),dim3(256)>>>(src_v,n,KVDIM,p,0x12b113U);
      fill_pos<<<dim3((n+255)/256),dim3(256)>>>(wpos,n,p);
      hipLaunchKernelGGL(kv_cache_write_asym_k_fwht3_batched,dim3(KV,n),dim3(32),1152,0,
                         kf,src_k,wpos,sign1,sign2,KV,D,n);
      hipLaunchKernelGGL(kv_cache_write_q8_0_batched,dim3(32,n),dim3(32),0,0,
                         kq,src_k,wpos,KV,D,n,nullptr,nullptr);
      hipLaunchKernelGGL(kv_cache_write_q8_0_batched,dim3(32,n),dim3(32),0,0,
                         v,src_v,wpos,KV,D,n,nullptr,nullptr);
      HIP(hipGetLastError());
    }
    size_t qcount=(size_t)batch*QDIM;
    fill_f32<<<dim3((qcount+255)/256),dim3(256)>>>(q,batch,QDIM,start,0x61f49U);
    fill_pos<<<dim3((batch+255)/256),dim3(256)>>>(pos,batch,start);
    hipLaunchKernelGGL(attention_fa2_q_preconvert_fwht3_gfx11,dim3((batch*H+3)/4),dim3(128),0,0,
                       q,q16f,sign1,sign2,batch,1);
    hipLaunchKernelGGL(attention_fa2_q_preconvert_gfx11,dim3((batch*H+3)/4),dim3(128),0,0,
                       q,q16q,nullptr,nullptr,batch,0);
    HIP(hipGetLastError());HIP(hipDeviceSynchronize());
    check_guard(kf,(size_t)ctx*400,"fwht3 K");
    check_guard(kq,(size_t)ctx*1088,"q8 K");
    check_guard(v,(size_t)ctx*1088,"q8 V");
    unsigned char ck[100],cv[34];
    HIP(hipMemcpy(ck,kf+(size_t)(ctx-1)*400,100,hipMemcpyDeviceToHost));
    HIP(hipMemcpy(cv,v+(size_t)(ctx-1)*1088,34,hipMemcpyDeviceToHost));
    bool kw=false,vw=false;for(auto x:ck) kw|=(x!=0xa5);for(auto x:cv) vw|=(x!=0xa5);
    if(!kw||!vw){fprintf(stderr,"INCOMPLETE cache at last position\n");exit(3);}
    fprintf(stderr,"FIXTURE ctx=%d B=%d fwht3 K production-writer all positions 0..%d; V and q8 K production q8 writer; Q16=%zu bytes\n",
            ctx,batch,ctx-1,qcount*2);
  }
  ~Fixture(){HIP(hipFree(pos));HIP(hipFree(wpos));HIP(hipFree(v));HIP(hipFree(kq));HIP(hipFree(kf));
    HIP(hipFree(q16q));HIP(hipFree(q16f));HIP(hipFree(out_q));HIP(hipFree(out_s));HIP(hipFree(out_f));
    HIP(hipFree(sign2));HIP(hipFree(sign1));HIP(hipFree(q));HIP(hipFree(src_v));HIP(hipFree(src_k));}
};
static double launch(Fixture& f,EventPair& ev,char mode) {
  int calls=(mode=='S')?(f.batch+511)/512:1;
  float* out=mode=='S'?f.out_s:mode=='Q'?f.out_q:f.out_f;
  const _Float16* q16=mode=='Q'?f.q16q:f.q16f;
  const unsigned char* k=mode=='Q'?f.kq:f.kf;
  for(int i=0;i<calls;i++) {
    int off=(mode=='S')?i*512:0, len=std::min(512,f.batch-off);
    if(mode!='S') len=f.batch;
    HIP(hipEventRecord(ev.a[i],0));
    if(mode=='Q') hipLaunchKernelGGL(attention_q8_0_fa2_gqa_q8_probe,
      dim3((len+7)/8,4,1),dim3(128),32768,0,
      q16+(size_t)off*QDIM,k,f.v,out+(size_t)off*QDIM,f.pos+off,H,KV,D,len,1.f/16.f);
    else hipLaunchKernelGGL(attention_q8_0_fa2_gqa_fwht3k_gfx11,
      dim3((len+7)/8,4,1),dim3(128),32768,0,
      q16+(size_t)off*QDIM,k,f.v,out+(size_t)off*QDIM,f.pos+off,H,KV,D,len,1.f/16.f);
    HIP(hipGetLastError()); HIP(hipEventRecord(ev.z[i],0));
  }
  HIP(hipEventSynchronize(ev.z[calls-1]));
  double us=0;for(int i=0;i<calls;i++){float ms;HIP(hipEventElapsedTime(&ms,ev.a[i],ev.z[i]));us+=ms*1000.;}
  return us;
}
static double median(std::vector<double> x) {std::sort(x.begin(),x.end());return x[x.size()/2];}
static void parity(Fixture& f,EventPair& ev) {
  launch(f,ev,'F');launch(f,ev,'S');
  size_t n=(size_t)f.batch*QDIM;
  std::vector<float> a(n),b(n);
  HIP(hipMemcpy(a.data(),f.out_f,n*4,hipMemcpyDeviceToHost));
  HIP(hipMemcpy(b.data(),f.out_s,n*4,hipMemcpyDeviceToHost));
  size_t wrong=0,nonfinite=0,first=n; double max_abs=0;
  for(size_t i=0;i<n;i++) {
    if(!std::isfinite(a[i])||!std::isfinite(b[i]))nonfinite++;
    if(memcmp(&a[i],&b[i],4)){wrong++;if(first==n)first=i;max_abs=std::max(max_abs,double(std::abs(a[i]-b[i])));}
  }
  check_guard(f.out_f,n*4,"fwht3 wide O");check_guard(f.out_s,n*4,"fwht3 split O");
  check_guard(f.kf,(size_t)f.ctx*400,"fwht3 K");check_guard(f.v,(size_t)f.ctx*1088,"q8 V");
  printf("PARITY start=%d B=%d ctx_end=%d values=%zu mismatches=%zu first=%zu max_abs=%.9g nonfinite=%zu\n",
    f.start,f.batch,f.ctx,n,wrong,first,max_abs,nonfinite);fflush(stdout);
  if(wrong||nonfinite) exit(4);
}
static void timing(Fixture& f,EventPair& ev,char mode) {
  for(int i=0;i<2;i++)launch(f,ev,mode);
  std::vector<double> t;for(int i=0;i<5;i++)t.push_back(launch(f,ev,mode));
  printf("TIME mode=%c start=%d B=%d ctx_end=%d median_us=%.3f samples_us=",mode,f.start,f.batch,f.ctx,median(t));
  for(double x:t)printf("%.3f,",x);puts("");fflush(stdout);
  size_t bytes=(size_t)f.batch*QDIM*4;
  check_guard(mode=='F'?f.out_f:mode=='S'?f.out_s:f.out_q,bytes,"timed output");
}
int main(int argc,char**argv) {
  if(argc!=4){fprintf(stderr,"usage: %s <start> <parity|F|S|Q> <comma-free B:512|2048|8192|all>\n",argv[0]);return 2;}
  HIP(hipSetDevice(0));hipDeviceProp_t p;HIP(hipGetDeviceProperties(&p,0));
  if(std::string(p.gcnArchName)!="gfx1100") {fprintf(stderr,"WRONG DEVICE ordinal 0 %s\n",p.gcnArchName);return 2;}
  int start=atoi(argv[1]);std::string mode=argv[2],bs=argv[3];
  if(start!=0&&start!=32768&&start!=65536&&start!=131072&&start!=245760&&start!=253952)return 2;
  if(mode!="parity"&&mode!="F"&&mode!="S"&&mode!="Q")return 2;
  printf("DEVICE ordinal=0 arch=%s start=%d mode=%s\n",p.gcnArchName,start,mode.c_str());fflush(stdout);
  for(int b:{512,2048,8192}) {
    if(bs!="all"&&bs!=std::to_string(b)) continue;
    Fixture f(start,b);EventPair ev;
    if(mode=="parity")parity(f,ev);
    else { timing(f,ev,mode[0]); if(mode=="F") timing(f,ev,'Q'); }
  }
  return 0;
}
