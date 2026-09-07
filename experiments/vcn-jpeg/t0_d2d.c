// SPDX-License-Identifier: MIT OR Apache-2.0
// T0 bound: D2D bandwidth on device 0 (bounds GPU preprocess cost).
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
int main(void) {
    void *hip = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_GLOBAL);
    int (*setDev)(int) = dlsym(hip, "hipSetDevice");
    int (*malloc_)(void **, size_t) = dlsym(hip, "hipMalloc");
    int (*memcpy_)(void *, const void *, size_t, int) = dlsym(hip, "hipMemcpy");
    int (*sync)(void) = dlsym(hip, "hipDeviceSynchronize");
    setDev(0);
    const size_t N = 32 << 20;
    void *a = 0, *b = 0;
    malloc_(&a, N);
    malloc_(&b, N);
    memcpy_(a, b, N, 3); // DeviceToDevice=3
    sync();
    double best = 1e18;
    for (int i = 0; i < 20; i++) {
        double t0 = now_ms();
        memcpy_(a, b, N, 3);
        sync();
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    printf("D2D 32MiB best=%.3f ms -> %.1f GB/s\n", best, 32.0 / best * 1000.0 / 1024.0);
    printf("preprocess traffic ~25MB -> bound %.3f ms\n", 25.0 / (32.0 / best * 1000.0 / 1024.0) * 1000.0 / 1024.0 * 1024.0 / 1000.0);
    return 0;
}
