// Diagnostic only: compare rocPRIM's stable descending segmented radix sort
// with the retained-route indexer top-K candidates on one finite-score head.

#include <hip/hip_runtime.h>
#include <rocprim/device/device_segmented_radix_sort.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        const hipError_t error = (expr);                                       \
        if (error != hipSuccess) {                                             \
            std::cerr << #expr << ": " << hipGetErrorString(error) << '\n';   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (false)

struct DeviceBuffers {
    float* keys_in = nullptr;
    float* keys_out = nullptr;
    int* values_in = nullptr;
    int* values_out = nullptr;
    int* top_out = nullptr;
    int* offsets = nullptr;
    void* temporary = nullptr;
    std::size_t temporary_bytes = 0;
};

static std::vector<float> make_scores(int n)
{
    std::vector<float> scores(n);
    for (int i = 0; i < n; ++i) {
        // Deterministic finite values with deliberate ties. Stable descending
        // order must use the original index as the tie-break.
        const int permuted = (i * 977 + 131) % 4093;
        scores[i] = static_cast<float>((permuted % 257) - 128) * 0.03125f;
    }
    return scores;
}

static std::vector<int> cpu_reference(const std::vector<float>& scores, int k)
{
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores[lhs] > scores[rhs];
    });
    order.resize(k);
    return order;
}

static void allocate(DeviceBuffers& buffers, int n, int k)
{
    HIP_CHECK(hipMalloc(&buffers.keys_in, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&buffers.keys_out, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&buffers.values_in, n * sizeof(int)));
    HIP_CHECK(hipMalloc(&buffers.values_out, n * sizeof(int)));
    HIP_CHECK(hipMalloc(&buffers.top_out, k * sizeof(int)));
    HIP_CHECK(hipMalloc(&buffers.offsets, 2 * sizeof(int)));

    HIP_CHECK(rocprim::segmented_radix_sort_pairs_desc(
        nullptr,
        buffers.temporary_bytes,
        buffers.keys_in,
        buffers.keys_out,
        buffers.values_in,
        buffers.values_out,
        static_cast<unsigned int>(n),
        1,
        buffers.offsets,
        buffers.offsets + 1));
    HIP_CHECK(hipMalloc(&buffers.temporary, buffers.temporary_bytes));
}

static void release(DeviceBuffers& buffers)
{
    HIP_CHECK(hipFree(buffers.temporary));
    HIP_CHECK(hipFree(buffers.offsets));
    HIP_CHECK(hipFree(buffers.top_out));
    HIP_CHECK(hipFree(buffers.values_out));
    HIP_CHECK(hipFree(buffers.values_in));
    HIP_CHECK(hipFree(buffers.keys_out));
    HIP_CHECK(hipFree(buffers.keys_in));
}

static void run_case(int n, int k, int iterations)
{
    DeviceBuffers buffers;
    allocate(buffers, n, k);

    const std::vector<float> scores = make_scores(n);
    std::vector<int> values(n);
    std::iota(values.begin(), values.end(), 0);
    const int offsets[2] = {0, n};

    HIP_CHECK(hipMemcpy(
        buffers.keys_in, scores.data(), n * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        buffers.values_in, values.data(), n * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        buffers.offsets, offsets, 2 * sizeof(int), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    auto launch = [&] {
        HIP_CHECK(rocprim::segmented_radix_sort_pairs_desc(
            buffers.temporary,
            buffers.temporary_bytes,
            buffers.keys_in,
            buffers.keys_out,
            buffers.values_in,
            buffers.values_out,
            static_cast<unsigned int>(n),
            1,
            buffers.offsets,
            buffers.offsets + 1,
            0,
            8 * sizeof(float),
            stream));
        HIP_CHECK(hipMemcpyAsync(
            buffers.top_out,
            buffers.values_out,
            k * sizeof(int),
            hipMemcpyDeviceToDevice,
            stream));
    };

    for (int i = 0; i < 20; ++i) launch();
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) launch();
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

    std::vector<int> actual(k);
    HIP_CHECK(hipMemcpy(
        actual.data(), buffers.top_out, k * sizeof(int), hipMemcpyDeviceToHost));
    const std::vector<int> expected = cpu_reference(scores, k);
    const bool exact = actual == expected;

    std::cout << "N=" << n
              << " K=" << k
              << " us=" << std::fixed << std::setprecision(3)
              << (elapsed_ms * 1000.0 / iterations)
              << " temporary_bytes=" << buffers.temporary_bytes
              << " exact=" << (exact ? "yes" : "NO")
              << '\n';

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipStreamDestroy(stream));
    release(buffers);

    if (!exact) std::exit(2);
}

int main(int argc, char** argv)
{
    const int iterations = argc > 1 ? std::stoi(argv[1]) : 1000;
    HIP_CHECK(hipSetDevice(0));
    for (const int n : {513, 640, 1024, 2048}) run_case(n, 512, iterations);
    return 0;
}
