// SPDX-License-Identifier: MIT OR Apache-2.0
// T0 gate-(c) probe: time VCN JPEG decode via rocJPEG HARDWARE backend.
// Throwaway. dlopens librocjpeg + libamdhip64 (no link-time deps).
// Usage: ./t0_vcn_timing <jpeg> [reps]
// Prints: image info + cold ms + warm min/median over reps.
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int RocJpegStatus;
typedef void *RocJpegHandle, *RocJpegStreamHandle;
#define ROCJPEG_MAX_COMPONENT 4
#define ROCJPEG_STATUS_SUCCESS 0
#define ROCJPEG_BACKEND_HARDWARE 0
#define ROCJPEG_OUTPUT_NATIVE 0

typedef struct {
    uint8_t *channel[ROCJPEG_MAX_COMPONENT];
    uint32_t pitch[ROCJPEG_MAX_COMPONENT];
} RocJpegImage;
typedef struct {
    int output_format;
    struct {
        int16_t left, top, right, bottom;
    } crop;
    struct {
        uint32_t width, height;
    } target;
} RocJpegDecodeParams;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <jpeg> [reps]\n", argv[0]);
        return 2;
    }
    int reps = argc > 2 ? atoi(argv[2]) : 10;

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror("fopen");
        return 2;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc(len);
    if (fread(data, 1, len, f) != (size_t)len) {
        perror("fread");
        return 2;
    }
    fclose(f);

    void *rj = dlopen("librocjpeg.so.1", RTLD_NOW);
    void *hip = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_GLOBAL);
    if (!rj || !hip) {
        printf("dlopen FAIL rj=%p hip=%p\n", rj, hip);
        return 2;
    }
    RocJpegStatus (*jCreate)(int, int, RocJpegHandle *) = dlsym(rj, "rocJpegCreate");
    RocJpegStatus (*jDestroy)(RocJpegHandle) = dlsym(rj, "rocJpegDestroy");
    RocJpegStatus (*sCreate)(RocJpegStreamHandle *) = dlsym(rj, "rocJpegStreamCreate");
    RocJpegStatus (*sParse)(const unsigned char *, size_t, RocJpegStreamHandle) =
        dlsym(rj, "rocJpegStreamParse");
    RocJpegStatus (*sDestroy)(RocJpegStreamHandle) = dlsym(rj, "rocJpegStreamDestroy");
    RocJpegStatus (*getInfo)(RocJpegHandle, RocJpegStreamHandle, uint8_t *, int *, uint32_t *,
                             uint32_t *) = dlsym(rj, "rocJpegGetImageInfo");
    RocJpegStatus (*decode)(RocJpegHandle, RocJpegStreamHandle, const RocJpegDecodeParams *,
                            RocJpegImage *) = dlsym(rj, "rocJpegDecode");
    int (*hipMalloc)(void **, size_t) = dlsym(hip, "hipMalloc");
    int (*hipFree)(void *) = dlsym(hip, "hipFree");
    int (*hipSync)(void) = dlsym(hip, "hipDeviceSynchronize");
    int (*hipSetDevice)(int) = dlsym(hip, "hipSetDevice");
    if (!jCreate || !decode || !hipMalloc) {
        printf("dlsym FAIL\n");
        return 2;
    }
    printf("hipSetDevice(0) -> %d\n", hipSetDevice(0));

    RocJpegHandle h = NULL;
    RocJpegStatus st = jCreate(ROCJPEG_BACKEND_HARDWARE, 0, &h);
    printf("rocJpegCreate(HARDWARE,dev0) -> %d\n", st);
    if (st != ROCJPEG_STATUS_SUCCESS) return 1;
    RocJpegStreamHandle s = NULL;
    sCreate(&s);
    st = sParse(data, len, s);
    printf("rocJpegStreamParse(%ld bytes) -> %d\n", len, st);
    if (st != ROCJPEG_STATUS_SUCCESS) return 1;

    uint8_t ncomp = 0;
    int subsampling = -1;
    uint32_t widths[4] = {0}, heights[4] = {0};
    st = getInfo(h, s, &ncomp, &subsampling, widths, heights);
    printf("info: ncomp=%u subsampling=%d w=%u h=%u -> %d\n", ncomp, subsampling, widths[0],
           heights[0], st);
    if (st != ROCJPEG_STATUS_SUCCESS) return 1;
    uint32_t W = widths[0], H = heights[0];

    // NATIVE 420: ch0 = Y (WxH), ch1 = UV interleaved (WxH/2)
    RocJpegImage img;
    memset(&img, 0, sizeof img);
    size_t y_sz = (size_t)W * H, uv_sz = (size_t)W * H / 2;
    int hr = hipMalloc((void **)&img.channel[0], y_sz);
    hr |= hipMalloc((void **)&img.channel[1], uv_sz);
    img.pitch[0] = W;
    img.pitch[1] = W;
    if (hr) {
        printf("hipMalloc FAIL %d\n", hr);
        return 1;
    }
    RocJpegDecodeParams p;
    memset(&p, 0, sizeof p);
    p.output_format = ROCJPEG_OUTPUT_NATIVE;

    double t0 = now_ms();
    st = decode(h, s, &p, &img);
    hipSync();
    double cold = now_ms() - t0;
    printf("decode cold -> %d, %.3f ms\n", st, cold);
    if (st != ROCJPEG_STATUS_SUCCESS) return 1;

    double best = 1e18, sum = 0;
    double *ts = malloc(sizeof(double) * reps);
    for (int i = 0; i < reps; i++) {
        t0 = now_ms();
        st = decode(h, s, &p, &img);
        hipSync();
        ts[i] = now_ms() - t0;
        if (ts[i] < best) best = ts[i];
        sum += ts[i];
        if (st != ROCJPEG_STATUS_SUCCESS) {
            printf("decode iter %d FAIL %d\n", i, st);
            return 1;
        }
    }
    // median
    for (int i = 0; i < reps; i++)
        for (int j = i + 1; j < reps; j++)
            if (ts[j] < ts[i]) {
                double t = ts[i];
                ts[i] = ts[j];
                ts[j] = t;
            }
    printf("decode warm x%d: min=%.3f median=%.3f mean=%.3f ms [%s %ux%u]\n", reps, best,
           ts[reps / 2], sum / reps, argv[1], W, H);

    hipFree(img.channel[0]);
    hipFree(img.channel[1]);
    sDestroy(s);
    jDestroy(h);
    return 0;
}
