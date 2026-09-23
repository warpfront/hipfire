#include <hip/hip_runtime.h>
#define K(N) extern "C" __global__ void store_flag_##N(float* p, float x) { \
  auto r=__builtin_amdgcn_make_buffer_rsrc(p,0,0xffffffffu,0x31004000u); \
  __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned,x),r,threadIdx.x*4,0,N); }
K(0) K(1) K(2) K(3) K(4) K(5) K(6) K(7)
