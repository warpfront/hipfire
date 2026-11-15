// Minimal HIP VMM repro: reserve VA, create physical device memory, map, set access,
// then hipMemset + a kernel write + read back. Run per device.
//
// Why: hipfire's VMM KV backend faulted on the R9700 (gfx1201) on the test box
// (kernel 7.0.1, ROCm 7.2.2), so the hetero split's runs use the legacy KV
// backend (`--kv-backend legacy`). Build and run with a mapping size in bytes:
//   hipcc -O2 -o /tmp/vmm_repro tools/hetero/vmm_sub2mib_repro.hip
//   /tmp/vmm_repro 1048576      # default 2 MiB
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("  %s -> %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)

__global__ void write(unsigned* p, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = i;
}

int test(int dev, size_t bytes) {
    CK(hipSetDevice(dev));
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;
    size_t gran = 0;
    CK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
    bytes = (bytes + gran - 1) / gran * gran;
    void* va = nullptr;
    CK(hipMemAddressReserve(&va, bytes, 0, nullptr, 0));
    hipMemGenericAllocationHandle_t h;
    CK(hipMemCreate(&h, bytes, &prop, 0));
    CK(hipMemMap(va, bytes, 0, h, 0));
    hipMemAccessDesc acc = {};
    acc.location.type = hipMemLocationTypeDevice;
    acc.location.id = dev;
    acc.flags = hipMemAccessFlagsProtReadWrite;
    CK(hipMemSetAccess(va, bytes, &acc, 1));
    printf("  va=%p bytes=%zu gran=%zu\n", va, bytes, gran);
    CK(hipMemset(va, 0, bytes));
    CK(hipDeviceSynchronize());
    hipLaunchKernelGGL(write, 64, 256, 0, 0, (unsigned*)va, bytes / 4);
    CK(hipDeviceSynchronize());
    unsigned last = 0;
    CK(hipMemcpy(&last, (char*)va + bytes - 4, 4, hipMemcpyDeviceToHost));
    printf("  memset + kernel + readback OK (last=%u expect=%zu)\n", last, bytes / 4 - 1);
    CK(hipMemUnmap(va, bytes)); CK(hipMemRelease(h)); CK(hipMemAddressFree(va, bytes));
    return 0;
}

int main(int argc, char** argv) {
    size_t bytes = argc > 1 ? strtoull(argv[1], nullptr, 0) : (2u << 20);
    int n; hipGetDeviceCount(&n);
    for (int d = 0; d < n; d++) {
        hipDeviceProp_t p; hipGetDeviceProperties(&p, d);
        printf("dev%d %s:\n", d, p.gcnArchName);
        if (test(d, bytes)) printf("  FAILED\n");
    }
}
