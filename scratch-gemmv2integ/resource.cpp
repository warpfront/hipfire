// Resource probe for the runtime-compiled V2C HSACO (8 waves, 32,768 B LDS).
// usage: resource <arch> <hsaco> <kernel...>; fails on spills/private, <2 CTAs/MP.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static void check(hipError_t e) { if (e != hipSuccess) { fprintf(stderr, "%s\n", hipGetErrorString(e)); exit(1); } }
int main(int argc, char** argv) {
    if (argc < 4) return 2;
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, 0));
    if (strncmp(prop.gcnArchName, argv[1], strlen(argv[1])) != 0) { fprintf(stderr, "arch %s != %s\n", prop.gcnArchName, argv[1]); return 3; }
    printf("GPU dev 0: %s\n", prop.gcnArchName);
    hipModule_t mod{};
    check(hipModuleLoad(&mod, argv[2]));
    const int threads = 256, lds = 32768;
    int rc = 0;
    for (int a = 3; a < argc; ++a) {
        hipFunction_t fn{};
        check(hipModuleGetFunction(&fn, mod, argv[a]));
        int regs = 0, priv = 0, shared = 0, blocks = 0;
        check(hipFuncGetAttribute(&regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, fn));
        check(hipFuncGetAttribute(&priv, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
        check(hipFuncGetAttribute(&shared, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
        check(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, threads, lds));
        printf("%s: %d VGPR, %d private bytes, %d static LDS, dynamic LDS %d, %d blocks/MP (%d waves)\n",
               argv[a], regs, priv, shared, lds, blocks, blocks * threads / 32);
        if (priv || blocks < 2) rc = 4;
    }
    check(hipModuleUnload(mod));
    return rc;
}
