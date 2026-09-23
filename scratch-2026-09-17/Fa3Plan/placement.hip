#define main timing_main
#include "pingpong.hip"
#undef main
int main(){
 float* out;unsigned* ids;CK(hipMalloc(&out,64*256*4));CK(hipMalloc(&ids,64*8*4));
 for(int c:{1,4})for(int bytes:{41216,65536}){
  if(c==1)run<3,1><<<64,256,bytes>>>(out,ids,256);else run<3,4><<<64,256,bytes>>>(out,ids,256);
  CK(hipGetLastError());std::vector<unsigned> h(64*8);CK(hipMemcpy(h.data(),ids,h.size()*4,hipMemcpyDeviceToHost));
  int bad=0;for(int b=0;b<64;b++)for(int w=0;w<4;w++)bad+=((h[b*8+w]^h[b*8+w+4])&0xffffff00u)!=0;
  printf("PING placement chains=%d LDS=%d same_upper24_pairs=%d/256 block0",c,bytes,256-bad);
  for(int w=0;w<8;w++)printf(" w%d=%08x",w,h[w]);puts("");if(bad)return 1;
 }
 CK(hipFree(out));CK(hipFree(ids));
}
