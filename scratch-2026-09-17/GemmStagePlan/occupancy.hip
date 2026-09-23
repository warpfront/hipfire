#include <hip/hip_runtime.h>
#include <cstdio>
#define DGU(n) extern "C" __global__ void n(const char*,const char*,const unsigned char*,const float*,const float*,float*,float*,int,int,int,int);
#define DRE(n) extern "C" __global__ void n(const char*,const unsigned char*,const float*,const float*,float*,int,int,int);
DGU(gu_full) DGU(gu_rawpref) DRE(re_full) DRE(re_rawpref)
extern "C" __global__ void qkv_full(const char*,const char*,const char*,const unsigned char*,const float*,const float*,float*,float*,float*,int,int,int,int,int);
extern "C" __global__ void qkvza_full(const char*,const char*,const char*,const char*,const unsigned char*,const float*,const float*,float*,float*,float*,float*,int,int,int,int,int,int);
int main(){hipDeviceProp_t p;hipGetDeviceProperties(&p,0);printf("MP=%d warp=%d maxThreadsMP=%d sharedMP=%zu regsMP=%d arch=%s\n",p.multiProcessorCount,p.warpSize,p.maxThreadsPerMultiProcessor,p.sharedMemPerMultiprocessor,p.regsPerMultiprocessor,p.gcnArchName);
 const void*fns[]={(const void*)gu_full,(const void*)gu_rawpref,(const void*)re_full,(const void*)re_rawpref,(const void*)qkv_full,(const void*)qkvza_full};const char* names[]={"gu_full","gu_rawpref","re_full","re_rawpref","qkv_full","qkvza_full"};
 for(int i=0;i<6;i++)for(size_t lds:{size_t(19968),size_t(38400),size_t(56832)}){int blocks;auto e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fns[i],256,lds);hipFuncAttributes a;hipFuncGetAttributes(&a,fns[i]);printf("%s lds=%zu vgpr=%d blocks=%d error=%d\n",names[i],lds,a.numRegs,blocks,e);}
}
