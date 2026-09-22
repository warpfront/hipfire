#define main fwht3_g0_harness_main
#include "fwht3_g0.hip"
#undef main
extern "C" __global__ void attention_flash_fwht3_tile_batched(const float*,const unsigned char*,const unsigned char*,float*,const int*,const float*,const float*,const float*,int,int,int,int,float,int,int,int,int,int,int);
extern "C" __global__ void attention_flash_asym_reduce_batched(const float*,float*,const int*,int,int,int,int,int,int,int);

int main(int argc,char**argv) {
  if(argc!=2)return 2;
  HIP(hipSetDevice(0));hipDeviceProp_t prop;HIP(hipGetDeviceProperties(&prop,0));
  if(std::string(prop.gcnArchName)!="gfx1100")return 2;
  int start=atoi(argv[1]);if(start<0||start>261632)return 2;
  Fixture f(start,512);EventPair ev;launch(f,ev,'F');
  int row=511, tile=128, max_tiles=(f.ctx+tile-1)/tile;
  size_t count=(size_t)H*max_tiles*(D+2);
  float* partials=alloc<float>(count),*out=alloc<float>(QDIM);
  hipLaunchKernelGGL(attention_flash_fwht3_tile_batched,dim3(H,max_tiles,1),dim3(32),tile*4,0,
    f.q+(size_t)row*QDIM,f.kf,f.v,partials,f.pos,f.sign1,f.sign2,nullptr,
    H,KV,D,f.ctx,1.f/16.f,tile,max_tiles,row,0,0,8);
  hipLaunchKernelGGL(attention_flash_asym_reduce_batched,dim3(H,1,1),dim3(32),0,0,
    partials,out,f.pos,H,D,tile,max_tiles,row,0,0);
  HIP(hipGetLastError());HIP(hipDeviceSynchronize());
  std::vector<float> a(QDIM),b(QDIM);
  HIP(hipMemcpy(a.data(),out,QDIM*4,hipMemcpyDeviceToHost));
  HIP(hipMemcpy(b.data(),f.out_f+(size_t)row*QDIM,QDIM*4,hipMemcpyDeviceToHost));
  double maxabs=0,maxrel=0;size_t nonfinite=0,different=0;
  for(int i=0;i<QDIM;i++) {
    if(!std::isfinite(a[i])||!std::isfinite(b[i]))nonfinite++;
    maxabs=std::max(maxabs,double(std::abs(a[i]-b[i])));
    maxrel=std::max(maxrel,double(std::abs(a[i]-b[i])/(1e-3f+std::abs(a[i]))));
    if(memcmp(&a[i],&b[i],4))different++;
  }
  printf("TILE_ORACLE start=%d pos=%d compared=%d different=%zu max_abs=%.9g max_rel_1e-3=%.9g nonfinite=%zu max_tiles=%d\n",
    start,start+row,QDIM,different,maxabs,maxrel,nonfinite,max_tiles);
  check_guard(f.kf,(size_t)f.ctx*400,"scalar oracle K");
  HIP(hipFree(out));HIP(hipFree(partials));
  return nonfinite||maxabs>0.02?4:0;
}
