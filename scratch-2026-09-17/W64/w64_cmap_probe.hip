#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
extern "C" __global__ __launch_bounds__(64) void cmap(const uint32_t* a,const uint32_t* b,int* o){
 int l=threadIdx.x; int32x4_t z={0,0,0,0}; int32x4_t c=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12(false,(int)a[l],true,(int)b[l],z,false); *(int32x4_t*)(o+4*l)=c;
}
static void ck(hipError_t e,const char*w){if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",w,hipGetErrorString(e));exit(2);}}
static uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x;}
int main(){uint32_t *da,*db;int*doo;ck(hipMalloc(&da,256),"ma");ck(hipMalloc(&db,256),"mb");ck(hipMalloc(&doo,1024),"mo");
 int got[5][256], ex[5][16][16];
 for(int t=0;t<5;t++){uint32_t a[64],b[64];
  for(int r=0;r<16;r++)for(int c=0;c<16;c++){ex[t][r][c]=0;for(int k=0;k<32;k++){int av=mix(0x1234567u*t+97*r+13*k+1)&15;int bv=(int)(mix(0x7654321u*t+89*c+17*k+3)&15)-8;ex[t][r][c]+=av*bv;}}
  for(int l=0;l<64;l++){int r=l&15,kg=l>>4;a[l]=b[l]=0;for(int q=0;q<8;q++){int k=kg*8+q;int av=mix(0x1234567u*t+97*r+13*k+1)&15;int bv=(int)(mix(0x7654321u*t+89*r+17*k+3)&15)-8;a[l]|=(uint32_t)av<<(4*q);b[l]|=(uint32_t)(bv&15)<<(4*q);}}
  ck(hipMemcpy(da,a,256,hipMemcpyHostToDevice),"ca");ck(hipMemcpy(db,b,256,hipMemcpyHostToDevice),"cb");hipLaunchKernelGGL(cmap,dim3(1),dim3(64),0,0,da,db,doo);ck(hipMemcpy(got[t],doo,1024,hipMemcpyDeviceToHost),"co");}
 int bad=0;printf("out_lane,out_elem,c_row,c_col\n");
 for(int p=0;p<256;p++){int mr=-1,mc=-1,n=0;for(int r=0;r<16;r++)for(int c=0;c<16;c++){bool eq=true;for(int t=0;t<5;t++)if(got[t][p]!=ex[t][r][c])eq=false;if(eq){mr=r;mc=c;n++;}} if(n!=1){fprintf(stderr,"unmapped p=%d matches=%d sig=%d,%d,%d,%d,%d\n",p,n,got[0][p],got[1][p],got[2][p],got[3][p],got[4][p]);bad++;} else printf("%d,%d,%d,%d\n",p/4,p%4,mr,mc);}
 printf("W64_C_MAP,%s,bad=%d\n",bad?"FAIL":"PASS",bad);return bad?1:0;}
