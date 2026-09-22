#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIPFIRE_FA2_KT 32
#include "../kernels/src/attention_q8_0_fa2_gqa.gfx11.hip"
#include "../kernels/src/attention_q8_0_fa2_gqa_wide.gfx11.hip"

static void check(hipError_t e, const char* what)
{
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

static void report(const char* name, const void* fn, int threads, size_t dynamic_lds)
{
    hipFuncAttributes a{};
    check(hipFuncGetAttributes(&a, fn), "hipFuncGetAttributes");
    int blocks = 0;
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks, fn, threads, dynamic_lds),
          "hipOccupancyMaxActiveBlocksPerMultiprocessor");
    const int waves_per_block = (threads + 31) / 32;
    std::printf(
        "RESOURCE kernel=%s api_num_regs=%d api_static_lds=%zu api_dynamic_lds=%zu "
        "api_local_bytes=%zu api_max_threads=%d api_blocks_per_cu=%d "
        "resident_waves_per_cu=%d\n",
        name,
        a.numRegs,
        a.sharedSizeBytes,
        dynamic_lds,
        a.localSizeBytes,
        a.maxThreadsPerBlock,
        blocks,
        blocks * waves_per_block);
}

int main()
{
    int device = 0;
    check(hipGetDevice(&device), "hipGetDevice");
    hipDeviceProp_t p{};
    check(hipGetDeviceProperties(&p, device), "hipGetDeviceProperties");
    const char* expected = std::getenv("EXPECT_ARCH");
    if (expected == nullptr || std::strcmp(expected, p.gcnArchName) != 0) {
        std::fprintf(stderr, "ARCH_ASSERT failed expected=%s actual=%s\n",
                     expected == nullptr ? "<unset>" : expected, p.gcnArchName);
        return 3;
    }
    std::printf("ARCH_ASSERT expected=%s actual=%s\n", expected, p.gcnArchName);
    report("attention_q8_0_fa2_gqa_gfx11",
           (const void*)attention_q8_0_fa2_gqa_gfx11, 128, 32768);
    report("attention_q8_0_fa2_gqa_wide_gfx11",
           (const void*)attention_q8_0_fa2_gqa_wide_gfx11, 512, 52224);
    return 0;
}
