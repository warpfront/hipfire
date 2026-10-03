// boundary_bench's empty kernel: the decode-floor T0 `empty_kernel` body,
// extern "C" so PM4 and HIP resolve one symbol from one code object.
// Build: hipcc --genco --offload-arch=gfx1100 -O3 --no-offload-compress -o empty.co boundary_bench.hip
#include <hip/hip_runtime.h>

extern "C" __global__ void bb_empty() {}
