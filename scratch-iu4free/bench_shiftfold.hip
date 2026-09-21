#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define IU4_SYMMETRIC_FOLD 1
#if BENCH_SHIFT
#define IU4_SHIFT_FOLDFREE 1
#endif
#include "../kernels/src/block_i4_128_quant.hip"
#include "../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip"

#define HIP_OK(x) do { hipError_t e=(x); if(e!=hipSuccess){std::fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));std::exit(1);} } while(0)

__global__ void init_pow2_headers(char* w, int m, int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = k / 256;
    if (idx >= m * groups) return;
    const int row = idx / groups;
    const int group = idx - row * groups;
    unsigned short* header = (unsigned short*)(w + (long long)idx * 136);
    const int e0 = -8 + ((row * 13 + group * 7) & 1);
    const int e1 = -8 + ((row * 5 + group * 11 + 1) & 1);
    header[0] = (unsigned short)((e0 + 15) << 10);
    header[1] = 0;
    header[2] = (unsigned short)((e1 + 15) << 10);
    header[3] = 0;
}

__global__ void init_global_a(block_i4_128* x, int blocks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= blocks) return;
    x[i].d = 1.0f;
    x[i].s = 0;
    #pragma unroll
    for (int j = 0; j < 64; ++j) x[i].qs[j] = 0x11;
}

int main(int argc, char** argv) {
    const bool down = argc > 1;
    const int B = 8192, M = down ? 5120 : 34816, K = down ? 17408 : 5120;
    const size_t groups = (size_t)K / 256;
    const size_t wb = (size_t)M * groups * 136;
    const size_t xblocks = (size_t)(K / 128) * B;
    const size_t xb = xblocks * sizeof(block_i4_128);
    const size_t yb = (size_t)B * M * sizeof(float);
    char* w = nullptr; block_i4_128* x = nullptr; float* y = nullptr;
    HIP_OK(hipMalloc(&w, wb)); HIP_OK(hipMalloc(&x, xb)); HIP_OK(hipMalloc(&y, yb));
    HIP_OK(hipMemset(w, 0, wb)); HIP_OK(hipMemset(x, 0, xb)); HIP_OK(hipMemset(y, 0, yb));
    init_pow2_headers<<<((M * groups) + 255) / 256, 256>>>(w, M, K);
    init_global_a<<<(xblocks + 255) / 256, 256>>>(x, (int)xblocks);
    const int threads = 256;
    dim3 block(threads), grid((M + 127) / 128, (B + (BENCH_SHIFT ? 255 : 127)) / (BENCH_SHIFT ? 256 : 128));
    const size_t lds = BENCH_SHIFT ? 30720 : 20480;
    for (int i = 0; i < 5; ++i)
        gemm_mq4g256v2_residual_mmq_iu4_full_set<<<grid, block, lds>>>(w, x, y, M, K, B, 0);
    HIP_OK(hipDeviceSynchronize());
    std::vector<float> us;
    for (int i = 0; i < 20; ++i) {
        hipEvent_t a, b; HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
        HIP_OK(hipEventRecord(a));
        gemm_mq4g256v2_residual_mmq_iu4_full_set<<<grid, block, lds>>>(w, x, y, M, K, B, 0);
        HIP_OK(hipEventRecord(b)); HIP_OK(hipEventSynchronize(b));
        float ms = 0; HIP_OK(hipEventElapsedTime(&ms, a, b)); us.push_back(ms * 1000.0f);
        HIP_OK(hipEventDestroy(a)); HIP_OK(hipEventDestroy(b));
    }
    std::sort(us.begin(), us.end());
    const double med = (us[9] + us[10]) * 0.5;
    double mean = 0; for (float v : us) mean += v; mean /= us.size();
    const double tops = (2.0 * (double)B * M * K) / (med * 1e6);
    int occ = 0;
    HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, gemm_mq4g256v2_residual_mmq_iu4_full_set, threads, lds));
    std::printf("RESULT,%s,%s,B=%d,M=%d,K=%d,median_us=%.3f,mean_us=%.3f,tops=%.3f,occ=%d,lds=%zu\n",
        BENCH_SHIFT ? "shiftfold" : "baseline", down ? "down" : "gate_up",
        B, M, K, med, mean, tops, occ, lds);
    HIP_OK(hipFree(w)); HIP_OK(hipFree(x)); HIP_OK(hipFree(y));
}
