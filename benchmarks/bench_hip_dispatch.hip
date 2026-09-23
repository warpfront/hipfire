// HIP dispatch benchmark — comparable to redline's bench_dispatch.
// Compile: hipcc -O3 --offload-arch=gfx1010 -o bench_hip bench_hip_dispatch.hip
// Run: ./bench_hip [iterations]

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

extern "C" __launch_bounds__(256)
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static long get_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long rss = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &rss);
            break;
        }
    }
    fclose(f);
    return rss;
}

int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(int argc, char** argv) {
    int iterations = argc > 1 ? atoi(argv[1]) : 10000;
    fprintf(stderr, "=== HIP Dispatch Benchmark ===\n");
    fprintf(stderr, "Iterations: %d\n\n", iterations);

    // --- Startup time ---
    double t_start = now_us();
    hipInit(0);
    hipSetDevice(0);
    double startup_init = (now_us() - t_start);

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "GPU: %s\n", prop.name);

    int n = 256;
    int nbytes = n * sizeof(float);

    float* h_a = (float*)malloc(nbytes);
    float* h_b = (float*)malloc(nbytes);
    for (int i = 0; i < n; i++) { h_a[i] = i; h_b[i] = i * 2.0f; }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, nbytes);
    hipMalloc(&d_b, nbytes);
    hipMalloc(&d_c, nbytes);
    hipMemcpy(d_a, h_a, nbytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, nbytes, hipMemcpyHostToDevice);

    double startup_total = (now_us() - t_start);
    fprintf(stderr, "[startup] init: %.2f ms, total: %.2f ms\n",
        startup_init / 1000.0, startup_total / 1000.0);

    // Warm up
    for (int i = 0; i < 10; i++) {
        vector_add<<<1, 256>>>(d_a, d_b, d_c, n);
        hipDeviceSynchronize();
    }

    // --- Per-dispatch latency ---
    double* latencies = (double*)malloc(iterations * sizeof(double));
    for (int i = 0; i < iterations; i++) {
        double t0 = now_us();
        vector_add<<<1, 256>>>(d_a, d_b, d_c, n);
        hipDeviceSynchronize();
        latencies[i] = now_us() - t0;
    }

    qsort(latencies, iterations, sizeof(double), cmp_double);
    double median = latencies[iterations / 2];
    double sum = 0; for (int i = 0; i < iterations; i++) sum += latencies[i];
    double mean = sum / iterations;
    double p99 = latencies[(int)(iterations * 0.99)];
    double mn = latencies[0];
    double mx = latencies[iterations - 1];

    fprintf(stderr, "\n[per-dispatch] %d iterations, vector_add 256 elements:\n", iterations);
    fprintf(stderr, "  median: %.1f µs\n", median);
    fprintf(stderr, "  mean:   %.1f µs\n", mean);
    fprintf(stderr, "  p99:    %.1f µs\n", p99);
    fprintf(stderr, "  min:    %.1f µs\n", mn);
    fprintf(stderr, "  max:    %.1f µs\n", mx);

    // --- Multi-dispatch throughput ---
    int batch = 200;
    double t_batch = now_us();
    for (int i = 0; i < batch; i++) {
        vector_add<<<1, 256>>>(d_a, d_b, d_c, n);
        hipDeviceSynchronize();
    }
    double batch_total = (now_us() - t_batch) / 1000.0; // ms
    double per_kernel = batch_total * 1000.0 / batch; // µs

    fprintf(stderr, "\n[%d-dispatch sequential] total: %.2f ms, per-kernel: %.1f µs\n",
        batch, batch_total, per_kernel);

    // --- Memory ---
    long rss = get_rss_kb();
    fprintf(stderr, "\n[memory] RSS: %ld KB (%.1f MB)\n", rss, rss / 1024.0);

    // --- Verify ---
    float* h_c = (float*)malloc(nbytes);
    hipMemcpy(h_c, d_c, nbytes, hipMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_c[i] - i * 3.0f) > 0.001f) bad++;
    }
    fprintf(stderr, "\n[verify] vector_add: %d/%d correct\n", n - bad, n);

    // --- Machine-readable ---
    printf("BENCH_HIP_MEDIAN_US=%.1f\n", median);
    printf("BENCH_HIP_MEAN_US=%.1f\n", mean);
    printf("BENCH_HIP_P99_US=%.1f\n", p99);
    printf("BENCH_HIP_MIN_US=%.1f\n", mn);
    printf("BENCH_HIP_MAX_US=%.1f\n", mx);
    printf("BENCH_HIP_BATCH_TOTAL_MS=%.2f\n", batch_total);
    printf("BENCH_HIP_BATCH_PER_KERNEL_US=%.1f\n", per_kernel);
    printf("BENCH_HIP_STARTUP_MS=%.2f\n", startup_init / 1000.0);
    printf("BENCH_HIP_RSS_KB=%ld\n", rss);

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c); free(latencies);
    return 0;
}
