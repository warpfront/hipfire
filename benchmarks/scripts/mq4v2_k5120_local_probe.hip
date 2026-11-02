// SPDX-License-Identifier: Apache-2.0
// Diagnostic only: load the exact objects archived by the serving replay test.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { auto e_=(x); if(e_!=hipSuccess) { fprintf(stderr,"line %d: %s\n",__LINE__,hipGetErrorString(e_)); exit(1); } } while(0)
constexpr int M=17408,K=5120,SLOTS=16,REPEATS=128;
constexpr size_t BYTES=size_t(M)*(K/256)*136;
__device__ unsigned mix(unsigned x) { x^=x>>16; x*=0x7feb352d; x^=x>>15; x*=0x846ca68b; return x^(x>>16); }
__global__ void initialize(char* w,size_t groups) {
    size_t g=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(g>=groups) return;
    auto p=reinterpret_cast<unsigned*>(w+136*g);
    for(int h=0;h<2;++h) {
        float scale=(mix(g*2+h)%127+1)/512.f;
        float zero=(int(mix(g*7+h+5)%255)-127)/64.f;
        p[h]=unsigned(__half_as_ushort(__float2half(scale))) | (unsigned(__half_as_ushort(__float2half(zero)))<<16);
    }
    for(int j=0;j<32;++j) p[j+2]=mix(g*32+j);
}
void launch(hipFunction_t f,char* w,float* x,float* y,hipStream_t stream) {
    char *up=w+BYTES; float *yu=y+M; int m=M,k=K;
    void* args[]={&w,&up,&x,&y,&yu,&m,&m,&k};
    CHECK(hipModuleLaunchKernel(f,2*M,1,1,32,1,1,0,stream,args,nullptr));
}
int main(int argc,char** argv) {
    if(argc!=3) { fprintf(stderr,"usage: probe generic.hsaco specialized.hsaco\n"); return 1; }
    hipDeviceProp_t p{}; CHECK(hipGetDeviceProperties(&p,0));
    if(strncmp(p.gcnArchName,"gfx1201",7)) return 77;
    char bus[32]; CHECK(hipDeviceGetPCIBusId(bus,sizeof(bus),0));
    printf("device=%s arch=%s pci=%s weight_ring_bytes=%zu\n",p.name,p.gcnArchName,bus,2*BYTES*SLOTS);
    hipModule_t modules[2]; hipFunction_t funcs[2];
    const char* names[]={"fused_gate_up_mq4g256v2","fused_gate_up_mq4g256v2_k5120_gfx1201"};
    for(int a=0;a<2;++a) { CHECK(hipModuleLoad(&modules[a],argv[a+1])); CHECK(hipModuleGetFunction(&funcs[a],modules[a],names[a])); }
    char* w; float *x,*y;
    CHECK(hipMalloc(&w,2*BYTES*SLOTS)); CHECK(hipMalloc(&x,sizeof(float)*K)); CHECK(hipMalloc(&y,sizeof(float)*2*M));
    size_t groups=2*size_t(M)*(K/256)*SLOTS;
    hipLaunchKernelGGL(initialize,dim3((groups+255)/256),dim3(256),0,0,w,groups);
    CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
    hipStream_t stream; CHECK(hipStreamCreate(&stream));
    std::vector<float> hx(K),ref(2*M),got(2*M);
    size_t checked=0;
    for(int seed=0;seed<3;++seed) {
        for(int i=0;i<K;++i) hx[i]=sinf(float(i*17+seed*79))*(1+i%7);
        CHECK(hipMemcpy(x,hx.data(),sizeof(float)*K,hipMemcpyHostToDevice));
        for(int slot=0;slot<SLOTS;++slot) for(int a=0;a<2;++a) {
            CHECK(hipMemsetAsync(y,0xff,sizeof(float)*2*M,stream));
            launch(funcs[a],w+slot*2*BYTES,x,y,stream); CHECK(hipStreamSynchronize(stream));
            CHECK(hipMemcpy(got.data(),y,sizeof(float)*2*M,hipMemcpyDeviceToHost));
            for(float v:got) if(!std::isfinite(v)) return 2;
            if(a==0) ref=got;
            else { if(memcmp(ref.data(),got.data(),sizeof(float)*2*M)) { fprintf(stderr,"mismatch seed=%d slot=%d\n",seed,slot); return 3; } checked+=got.size(); }
        }
    }
    printf("byte_exact_elements=%zu\n",checked);
    hipEvent_t start,end; CHECK(hipEventCreate(&start)); CHECK(hipEventCreate(&end));
    puts("ring,round,position,arm,us_per_call,nominal_weight_GBs");
    for(int ring:{1,SLOTS}) {
        hipGraph_t graphs[2]; hipGraphExec_t execs[2];
        for(int a=0;a<2;++a) {
            CHECK(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));
            for(int i=0;i<REPEATS;++i) launch(funcs[a],w+(i%ring)*2*BYTES,x,y,stream);
            CHECK(hipStreamEndCapture(stream,&graphs[a])); CHECK(hipGraphInstantiate(&execs[a],graphs[a],nullptr,nullptr,0));
        }
        auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(std::chrono::steady_clock::now()<until) { for(auto exec:execs) CHECK(hipGraphLaunch(exec,stream)); CHECK(hipStreamSynchronize(stream)); }
        for(int r=0;r<10;++r) for(int j=0;j<4;++j) {
            int a=(j==1||j==2); if(r%2) a=1-a;
            CHECK(hipEventRecord(start,stream)); CHECK(hipGraphLaunch(execs[a],stream)); CHECK(hipEventRecord(end,stream)); CHECK(hipEventSynchronize(end));
            float ms; CHECK(hipEventElapsedTime(&ms,start,end)); double us=ms*1000./REPEATS;
            printf("%d,%d,%d,%d,%.6f,%.3f\n",ring,r,j,a,us,2*BYTES/(us*1000)); fflush(stdout);
        }
        for(int a=0;a<2;++a) { CHECK(hipGraphExecDestroy(execs[a])); CHECK(hipGraphDestroy(graphs[a])); }
    }
    CHECK(hipEventDestroy(start)); CHECK(hipEventDestroy(end)); CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(w)); CHECK(hipFree(x)); CHECK(hipFree(y));
    for(auto module:modules) CHECK(hipModuleUnload(module));
}
