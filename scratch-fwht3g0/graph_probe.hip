#define main fwht3_g0_harness_main
#include "fwht3_g0.hip"
#undef main

// Capture the same unchanged producer/body with a 512-row scratch, then
// explicitly grow the q16 allocation and recapture at 8192 rows. Compare
// each replay against a direct launch on the identical complete KV bytes.
static void body(Fixture& f,_Float16* scratch,int rows,hipStream_t stream) {
  hipLaunchKernelGGL(attention_fa2_q_preconvert_fwht3_gfx11,dim3((rows*H+3)/4),dim3(128),0,stream,
                     f.q,scratch,f.sign1,f.sign2,rows,1);
  hipLaunchKernelGGL(attention_q8_0_fa2_gqa_fwht3k_gfx11,
    dim3((rows+7)/8,4,1),dim3(128),32768,stream,
    scratch,f.kf,f.v,f.out_f,f.pos,H,KV,D,rows,1.f/16.f);
  HIP(hipGetLastError());
}
static std::vector<float> fetch(Fixture& f,int rows) {
  std::vector<float> out((size_t)rows*QDIM);
  HIP(hipMemcpy(out.data(),f.out_f,out.size()*4,hipMemcpyDeviceToHost));return out;
}
static void run_capture(Fixture& f,_Float16* scratch,int rows) {
  hipStream_t stream;HIP(hipStreamCreate(&stream));
  body(f,scratch,rows,stream);HIP(hipDeviceSynchronize());auto expected=fetch(f,rows);
  hipGraph_t graph;hipGraphExec_t exec;
  HIP(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));
  body(f,scratch,rows,stream);
  HIP(hipStreamEndCapture(stream,&graph));
  HIP(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  for(int n=0;n<2;n++) {
    HIP(hipMemset(f.out_f,0,(size_t)rows*QDIM*4));
    HIP(hipGraphLaunch(exec,stream));HIP(hipDeviceSynchronize());
    auto got=fetch(f,rows);
    if(memcmp(got.data(),expected.data(),got.size()*4)){
      fprintf(stderr,"GRAPH REPLAY MISMATCH B=%d replay=%d\n",rows,n);exit(4);
    }
  }
  check_guard(f.out_f,(size_t)f.batch*QDIM*4,"graph output");
  printf("GRAPH PASS rows=%d replays=2 bits=%zu\n",rows,expected.size());fflush(stdout);
  HIP(hipGraphExecDestroy(exec));HIP(hipGraphDestroy(graph));HIP(hipStreamDestroy(stream));
}
int main() {
  HIP(hipSetDevice(0));hipDeviceProp_t p;HIP(hipGetDeviceProperties(&p,0));
  if(std::string(p.gcnArchName)!="gfx1100")return 2;
  Fixture f(253952,8192);
  _Float16* small=alloc<_Float16>((size_t)512*QDIM);
  run_capture(f,small,512);
  HIP(hipFree(small));
  _Float16* grown=alloc<_Float16>((size_t)8192*QDIM);
  run_capture(f,grown,8192);
  HIP(hipFree(grown));
  return 0;
}
