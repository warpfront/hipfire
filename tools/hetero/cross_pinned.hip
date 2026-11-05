// Proof A for the Qwen4 hetero expert split (tools/hetero/README.md): can the gfx1151
// iGPU read host memory that was pinned through the gfx1201 device, the way hipfire
// pins host-mapped expert tiers (`hipHostMalloc(hipHostMallocMapped)`,
// rdna-compute/src/dispatch.rs alloc_host_mapped_tensor), and how fast?
//
// Every pass verifies a seeded pattern, so a read of the wrong (e.g. zero) page cannot
// pass as bandwidth. Bounded work only, no spins.
//
// Build: hipcc -O2 --offload-arch=gfx1151 --offload-arch=gfx1201 cross_pinned.hip -o cross_pinned
// Run:   ./cross_pinned [bytes=2147483648]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("  %s -> %s\n", #x, hipGetErrorString(e)); return false; } } while (0)

__host__ __device__ inline unsigned pattern(size_t i) { return (unsigned)i * 2654435761u; }

__global__ void verify(const uint4* p, size_t n4, unsigned long long* bad) {
    unsigned long long local = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        size_t w = i * 4;
        local += (v.x != pattern(w)) + (v.y != pattern(w + 1)) + (v.z != pattern(w + 2)) + (v.w != pattern(w + 3));
    }
    if (local) atomicAdd(bad, local);
}

__global__ void fill(uint4* p, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        size_t w = i * 4;
        p[i] = make_uint4(pattern(w), pattern(w + 1), pattern(w + 2), pattern(w + 3));
    }
}

static int find(const char* arch) {
    int n; hipGetDeviceCount(&n);
    for (int d = 0; d < n; d++) {
        hipDeviceProp_t p; hipGetDeviceProperties(&p, d);
        if (std::string(p.gcnArchName).rfind(arch, 0) == 0) return d;
    }
    printf("no %s device\n", arch); exit(1);
}

// Read `dev_ptr` (valid on device `reader`) 5 times, verifying; print median GB/s.
static bool measure(const char* label, int reader, const void* dev_ptr, size_t bytes) {
    CK(hipSetDevice(reader));
    unsigned long long* bad;
    CK(hipHostMalloc((void**)&bad, sizeof(*bad), hipHostMallocMapped));
    hipEvent_t a, b;
    CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    std::vector<float> gbs;
    unsigned long long total_bad = 0;
    for (int pass = 0; pass < 6; pass++) {  // pass 0 = warm-up
        *bad = 0;
        CK(hipEventRecord(a));
        hipLaunchKernelGGL(verify, 1024, 256, 0, 0, (const uint4*)dev_ptr, bytes / 16, bad);
        CK(hipGetLastError());
        CK(hipEventRecord(b));
        CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b));
        total_bad += *bad;
        if (pass) gbs.push_back(bytes / ms / 1e6f);
    }
    std::sort(gbs.begin(), gbs.end());
    printf("%-34s mismatches=%llu  median %.1f GB/s (min %.1f, max %.1f)\n",
           label, total_bad, gbs[2], gbs[0], gbs[4]);
    CK(hipEventDestroy(a)); CK(hipEventDestroy(b)); CK(hipHostFree(bad));
    return true;
}

// Pin through `owner`, fill on the CPU, read on `reader` via its own device pointer.
static bool cross(const char* label, int owner, int reader, unsigned flags, size_t bytes) {
    CK(hipSetDevice(owner));
    void* host = nullptr;
    CK(hipHostMalloc(&host, bytes, flags));
    unsigned* w = (unsigned*)host;
    for (size_t i = 0; i < bytes / 4; i++) w[i] = pattern(i);
    CK(hipSetDevice(reader));
    void* dptr = nullptr;
    hipError_t e = hipHostGetDevicePointer(&dptr, host, 0);
    if (e != hipSuccess) {
        printf("%-34s hipHostGetDevicePointer on reader -> %s\n", label, hipGetErrorString(e));
        CK(hipSetDevice(owner)); CK(hipHostFree(host));
        return true;
    }
    bool ok = measure(label, reader, dptr, bytes);
    CK(hipSetDevice(owner));
    CK(hipHostFree(host));
    return ok;
}

int main(int argc, char** argv) {
    size_t bytes = argc > 1 ? strtoull(argv[1], nullptr, 0) : (2ull << 30);
    bytes &= ~(size_t)15;
    int R = find("gfx1201"), S = find("gfx1151");
    printf("owner gfx1201 dev%d, reader gfx1151 dev%d, %zu MiB\n", R, S, bytes >> 20);
    if (!cross("mapped (hipfire's flags)", R, S, hipHostMallocMapped, bytes)) return 1;
    if (!cross("mapped|portable", R, S, hipHostMallocMapped | hipHostMallocPortable, bytes)) return 1;
    // Control: the same kernel on the iGPU's own allocation.
    if (hipSetDevice(S) != hipSuccess) return 1;
    void* own = nullptr;
    if (hipMalloc(&own, bytes) != hipSuccess) { printf("hipMalloc control failed\n"); return 1; }
    hipLaunchKernelGGL(fill, 1024, 256, 0, 0, (uint4*)own, bytes / 16);
    if (hipDeviceSynchronize() != hipSuccess) return 1;
    if (!measure("hipMalloc control (gfx1151)", S, own, bytes)) return 1;
    hipFree(own);
    return 0;
}
