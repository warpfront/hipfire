#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static void check(hipError_t e) { if (e != hipSuccess) { fprintf(stderr, "%s\n", hipGetErrorString(e)); exit(1); } }
int main(int argc, char** argv) {
    if (argc < 4) return 2;
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, 0));
    if (strcmp(prop.gcnArchName, argv[1]) != 0) { fprintf(stderr, "arch %s != %s\n", prop.gcnArchName, argv[1]); return 3; }
    printf("GPU dev 0: %s\n", prop.gcnArchName);
    hipModule_t mod{};
    check(hipModuleLoad(&mod, argv[2]));
    for (int a = 3; a < argc; ++a) {
        hipFunction_t fn{};
        check(hipModuleGetFunction(&fn, mod, argv[a]));
        int regs = 0, priv = 0, blocks = 0;
        check(hipFuncGetAttribute(&regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, fn));
        check(hipFuncGetAttribute(&priv, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
        check(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, 512, 30720));
        printf("%s: %d VGPR %d private bytes %d blocks/MP (%d waves) LDS=30720\n", argv[a], regs, priv, blocks, blocks * 16);
        if (regs > 96 || priv || blocks < 2) return 4;
    }
    check(hipModuleUnload(mod));
}
