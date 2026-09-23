// Standalone gfx11 FA2 (Q8 K/V, H24/KV4/D256) screen: loads each variant's
// code object, runs production-shaped schedules on identical non-periodic
// inputs, times them with HIP events and compares every output bit against
// the first code object (the reference).
//
// usage: fa2_bench <schedule> <reps> <ref.hsaco> [variant.hsaco ...]
//   schedule: halo  = 16 launches B512 at starts 0,512,..,7680 (Halo chunks)
//             xtx   = 1 launch B8192 start 0 (gfx1100 whole-chunk)
//             odd   = (B100,s37) (B77,s300) (B512,s1000) (B64,s0) tails
// Order is interleaved (ref, v1, v2, ..., ref, v1, ...) per rep to spread
// clock/thermal drift; the reported figure is the median per variant.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,        \
                    hipGetErrorString(e_));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static inline uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

struct Launch {
    int batch;
    int start;
};

struct Variant {
    std::string path;
    hipModule_t mod;
    hipFunction_t fn;
    int lds;
    int threads;
};

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s halo|xtx|odd reps ref.hsaco [var.hsaco...]\n", argv[0]);
        return 2;
    }
    const std::string sched = argv[1];
    const int reps = atoi(argv[2]);
    std::vector<Launch> launches;
    if (sched == "halo") {
        for (int s = 0; s < 8192; s += 512)
            launches.push_back({512, s});
    } else if (sched == "xtx") {
        launches.push_back({8192, 0});
    } else if (sched == "odd") {
        launches = {{100, 37}, {77, 300}, {512, 1000}, {64, 0}, {1024, 7168}};
    } else {
        fprintf(stderr, "bad schedule\n");
        return 2;
    }
    int max_pos = 0;
    int max_rows = 0;
    for (auto& l : launches) {
        max_pos = std::max(max_pos, l.start + l.batch);
        max_rows = std::max(max_rows, l.start + l.batch);
    }
    const int NPOS = max_pos;
    const size_t q_elems = (size_t)max_rows * 24 * 256;
    const size_t row_bytes = 4 * 8 * 34;
    const size_t kv_bytes = (size_t)NPOS * row_bytes;

    // Inputs: Q f16 in [-2,2); K/V Q8_0 blocks with log-spread f16 scales
    // (1/512 .. 1/8, a few 4x outliers) and full-range int8 codes.
    std::vector<_Float16> hq(q_elems);
    for (size_t i = 0; i < q_elems; ++i) {
        const uint64_t r = mix64(UINT64_C(0x243f6a8885a308d3) + i);
        const float u = (float)(r & 0xffffff) / 16777216.0f;
        hq[i] = (_Float16)((u - 0.5f) * 4.0f);
    }
    auto fill_kv = [&](std::vector<unsigned char>& buf, uint64_t seed) {
        buf.resize(kv_bytes);
        for (size_t blk = 0; blk < kv_bytes / 34; ++blk) {
            const uint64_t r = mix64(seed + blk * 3);
            const float e = (float)(r & 0xffff) / 65536.0f;          // 0..1
            float sc = exp2f(-9.0f + 6.0f * e);                         // 1/512..1/8
            if (((r >> 16) & 63) == 0)
                sc *= 4.0f;
            const _Float16 hs = (_Float16)sc;
            memcpy(&buf[blk * 34], &hs, 2);
            for (int w = 0; w < 4; ++w) {
                const uint64_t c = mix64(seed + blk * 3 + 1 + (uint64_t)w * 0x9e3779b97f4a7c15ULL);
                for (int b = 0; b < 8; ++b) {
                    int8_t v = (int8_t)(c >> (8 * b));
                    if (v == -128)
                        v = -127;
                    buf[blk * 34 + 2 + w * 8 + b] = (unsigned char)v;
                }
            }
        }
    };
    std::vector<unsigned char> hk, hv;
    fill_kv(hk, UINT64_C(0x13198a2e03707344));
    fill_kv(hv, UINT64_C(0xa4093822299f31d0));
    std::vector<int> hpos(max_rows);
    for (int i = 0; i < max_rows; ++i)
        hpos[i] = i;  // row r at absolute position r (chunk start folds in)

    _Float16* dq;
    unsigned char *dk, *dv;
    float* dout;
    int* dpos;
    CK(hipMalloc(&dq, q_elems * 2));
    CK(hipMalloc(&dk, kv_bytes));
    CK(hipMalloc(&dv, kv_bytes));
    CK(hipMalloc(&dout, q_elems * 4));
    CK(hipMalloc(&dpos, max_rows * 4));
    CK(hipMemcpy(dq, hq.data(), q_elems * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dk, hk.data(), kv_bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(dv, hv.data(), kv_bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(dpos, hpos.data(), max_rows * 4, hipMemcpyHostToDevice));

    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    printf("device: %s (%s) CUs=%d schedule=%s launches=%zu reps=%d\n", prop.name,
           prop.gcnArchName, prop.multiProcessorCount, sched.c_str(), launches.size(), reps);

    std::vector<Variant> vars;
    int* dlds;
    CK(hipMalloc(&dlds, 8));
    for (int a = 3; a < argc; ++a) {
        Variant v;
        v.path = argv[a];
        CK(hipModuleLoad(&v.mod, v.path.c_str()));
        CK(hipModuleGetFunction(&v.fn, v.mod, "attention_q8_0_fa2_gqa_gfx11"));
        hipFunction_t lf;
        v.lds = 32768;
        v.threads = 256;
        if (hipModuleGetFunction(&lf, v.mod, "fa2_var_lds_bytes") == hipSuccess) {
            void* args[] = {&dlds};
            CK(hipModuleLaunchKernel(lf, 1, 1, 1, 1, 1, 1, 0, 0, args, nullptr));
            int meta[2];
            CK(hipMemcpy(meta, dlds, 8, hipMemcpyDeviceToHost));
            v.lds = meta[0];
            v.threads = meta[1];
        } else {
            (void)hipGetLastError();
        }
        vars.push_back(v);
    }

    const float scale = 1.0f / 16.0f;
    auto run = [&](Variant& v) {
        for (auto& l : launches) {
            const _Float16* q = dq + (size_t)l.start * 6144;
            const unsigned char* k = dk;
            const unsigned char* vv = dv;
            float* o = dout + (size_t)l.start * 6144;
            const int* p = dpos + l.start;
            int nh = 24, nkv = 4, hd = 256, bs = l.batch;
            float sc = scale;
            void* args[] = {&q, &k, &vv, &o, &p, &nh, &nkv, &hd, &bs, &sc};
            CK(hipModuleLaunchKernel(v.fn, (l.batch + 15) / 16, 4, 1, v.threads, 1, 1,
                                     v.lds, 0, args, nullptr));
        }
    };

    // Correctness: reference output, then each variant vs reference bits.
    std::vector<uint32_t> ref(q_elems), got(q_elems);
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    std::vector<std::vector<float>> times(vars.size());
    std::vector<long long> mism(vars.size(), 0);
    std::vector<int> nonfinite(vars.size(), 0);
    for (size_t i = 0; i < vars.size(); ++i) {
        CK(hipMemset(dout, 0xff, q_elems * 4));  // poison: NaN everywhere
        run(vars[i]);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(i == 0 ? ref.data() : got.data(), dout, q_elems * 4,
                     hipMemcpyDeviceToHost));
        const std::vector<uint32_t>& g = i == 0 ? ref : got;
        long long m = 0;
        int nf = 0;
        // Only rows covered by launches are defined.
        for (auto& l : launches)
            for (size_t e = (size_t)l.start * 6144; e < (size_t)(l.start + l.batch) * 6144; ++e) {
                float f;
                memcpy(&f, &g[e], 4);
                if (!std::isfinite(f))
                    ++nf;
                if (i > 0 && g[e] != ref[e])
                    ++m;
            }
        mism[i] = m;
        nonfinite[i] = nf;
    }
    // Warmup
    for (auto& v : vars)
        run(v);
    CK(hipDeviceSynchronize());
    for (int r = 0; r < reps; ++r) {
        for (size_t i = 0; i < vars.size(); ++i) {
            CK(hipEventRecord(e0, 0));
            run(vars[i]);
            CK(hipEventRecord(e1, 0));
            CK(hipEventSynchronize(e1));
            float ms;
            CK(hipEventElapsedTime(&ms, e0, e1));
            times[i].push_back(ms);
        }
    }
    double ref_med = 0;
    for (size_t i = 0; i < vars.size(); ++i) {
        auto t = times[i];
        std::sort(t.begin(), t.end());
        const double med = t[t.size() / 2];
        if (i == 0)
            ref_med = med;
        printf("%-34s thr=%3d lds=%6d  median %9.3f ms  min %9.3f  max %9.3f  x%.4f  mismatches=%lld nonfinite=%d\n",
               vars[i].path.c_str(), vars[i].threads, vars[i].lds, med, t.front(), t.back(),
               ref_med / med, mism[i], nonfinite[i]);
    }
    return 0;
}
