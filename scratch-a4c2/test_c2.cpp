// Direct bitwise oracle for the gfx1201 c2 emit: runs qemit (qemit.hip) from
// two code objects (reference = base-tree quant + -DIU4_A4_CANDIDATES=2,
// candidate = new quant + the same flag) on adversarial and random 128-blocks
// and compares every output byte.
// usage: test_c2 <ref.hsaco> <new.hsaco> <n_amax> <seed> [kwalk] [edge_only]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)

struct Block_ { float d; int s; unsigned char qs[64]; };

static float walk_ulps(float v, int k) {
    int32_t b; memcpy(&b, &v, 4);
    int64_t ord = (b >= 0) ? (int64_t)b : -(int64_t)(b & 0x7fffffff);
    ord += k;
    int32_t nb = (ord >= 0) ? (int32_t)ord : (int32_t)((-ord) | 0x80000000LL);
    float r; memcpy(&r, &nb, 4); return r;
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: test_c2 ref.hsaco new.hsaco n_amax seed [kwalk]\n"); return 2; }
    const int n_rand = atoi(argv[3]);
    const unsigned seed = (unsigned)atoi(argv[4]);
    const int KW = argc > 5 ? atoi(argv[5]) : 48;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::uniform_int_distribution<uint32_t> sig(0, (1u << 23) - 1);
    std::uniform_int_distribution<int> ex(-115, 127);

    std::vector<float> amaxes = {0x1p-100f, 0x1.fffffep-101f, 0x1.000002p-100f, 1e-38f, 1e-40f, 1e-45f,
                                 0x1.fffffep+127f, 0x1.fffffep+126f, 1e38f, 1e30f, 1e20f, 1e17f,
                                 1.0f, 7.0f, 3.5f, 0x1.fffffep+0f, 0x1p-126f, 6.0f, 0.1f, 0x1.8p+0f,
                                 12.0f, 49.0f, 0x1.b6db6ep+0f};
    for (int e = -20; e <= 20; ++e) { amaxes.push_back(ldexpf(0x1.fffffep+0f, e)); amaxes.push_back(ldexpf(1.0f, e)); }
    for (int i = 0; i < n_rand; ++i) {
        uint32_t bits = ((uint32_t)(ex(rng) + 127) << 23) | sig(rng);
        float f; memcpy(&f, &bits, 4); amaxes.push_back(f);
    }

    std::vector<float> blocks;  // 128 floats per block
    auto push_block = [&](const float* v) { blocks.insert(blocks.end(), v, v + 128); };
    float v[128];
    // Class A: walks around every half-integer boundary and multiple of d0 or d1.
    for (float amax : amaxes) {
        const float a7 = amax / 7.0f, base = a7 * 0.5f;
        const float ds[2] = {base * 0x1.b6db6ep+0f, base * 2.0f};
        for (int c = 0; c < 2; ++c) {
            const float d = ds[c];
            std::vector<float> vals;
            for (int m = -10; m <= 9; ++m) {
                for (int k = -KW; k <= KW; ++k) vals.push_back(walk_ulps(((float)m + 0.5f) * d, k));
                vals.push_back((float)m * d);
            }
            for (size_t i = 0; i < vals.size(); i += 127) {
                v[0] = (blocks.size() / 128) & 1 ? -amax : amax;
                for (int e = 1; e < 128; ++e) {
                    float x = (i + e - 1 < vals.size()) ? vals[i + e - 1] : 0.0f;
                    if (!(fabsf(x) <= amax)) x = copysignf(amax, x);
                    v[e] = x;
                }
                // Scatter the amax slot so it is not always lane 0.
                std::swap(v[0], v[(i / 127 * 37) % 128]);
                push_block(v);
            }
        }
    }
    const size_t n_adv = blocks.size() / 128;
    // Class B: random blocks, per-block scale over the whole exponent range,
    // occasional outliers and exact zeros.
    for (int b = 0; b < 40000; ++b) {
        const float sc = ldexpf(1.0f, (int)(ud(rng) * 250) - 125);
        for (int e = 0; e < 128; ++e) {
            float x = nd(rng) * sc;
            const float u = ud(rng);
            if (u < 0.01f) x *= 40.0f;
            else if (u < 0.02f) x = 0.0f;
            v[e] = x;
        }
        push_block(v);
    }
    const size_t n_edge0 = blocks.size() / 128;
    // Class D: every value just outside the 2^-18 near-half window of one
    // candidate (distance 2^-18 * (1.02 .. 4)), so whole waves stay on the
    // reciprocal path right at the proof's edge.
    for (size_t ai = 0; ai < amaxes.size(); ai += 1) {
        const float amax = amaxes[ai];
        if (!(amax >= 0x1p-99f && amax <= 0x1p+120f)) continue;
        const float a7 = amax / 7.0f, base = a7 * 0.5f;
        const float ds[2] = {base * 0x1.b6db6ep+0f, base * 2.0f};
        for (int c = 0; c < 2; ++c) {
            v[0] = amax;
            for (int e = 1; e < 128; ++e) {
                const int m = (int)(ud(rng) * 16) - 8;
                const double dl = ldexp(1.0, -18) * (1.02 + 3.0 * ud(rng)) * (ud(rng) < 0.5 ? -1 : 1);
                float x = (float)(((double)m + 0.5 + dl) * (double)ds[c]);
                if (!(fabsf(x) <= amax)) x = copysignf(amax, x);
                v[e] = x;
            }
            std::swap(v[0], v[(ai * 13 + c) % 128]);
            push_block(v);
        }
    }
    const size_t n_edge = blocks.size() / 128 - n_edge0;
    // Class C: specials.
    auto fill = [&](float x) { for (int e = 0; e < 128; ++e) v[e] = x; };
    fill(0.0f); push_block(v);
    fill(-0.0f); push_block(v);
    fill(0.0f); v[5] = NAN; push_block(v);                      // amax 0 with a NaN
    fill(NAN); push_block(v);
    for (int e = 0; e < 128; ++e) v[e] = nd(rng); v[17] = NAN; push_block(v);
    for (int e = 0; e < 128; ++e) v[e] = nd(rng); v[90] = INFINITY; push_block(v);
    for (int e = 0; e < 128; ++e) v[e] = nd(rng); v[3] = -INFINITY; v[4] = NAN; push_block(v);
    for (int e = 0; e < 128; ++e) v[e] = nd(rng) * 1e-40f; push_block(v);   // denormals
    for (int e = 0; e < 128; ++e) v[e] = nd(rng) * 1e-44f; push_block(v);
    for (int e = 0; e < 128; ++e) v[e] = nd(rng) * 1e20f; push_block(v);    // mse >= 1e30
    for (int e = 0; e < 128; ++e) v[e] = nd(rng) * 1e14f; push_block(v);
    for (int e = 0; e < 128; ++e) v[e] = nd(rng) * 3e37f; push_block(v);
    for (int e = 0; e < 128; ++e) v[e] = (e % 2 ? 1.f : -1.f) * 7.0f; push_block(v);
    for (int e = 0; e < 128; ++e) v[e] = (float)((e * 37) % 15 - 7); push_block(v);
    for (int e = 0; e < 128; ++e) v[e] = ldexpf(1.0f, (e % 60) - 30) * (e % 3 ? 1.f : -1.f); push_block(v);
    fill(0.0f); v[64] = 3.5f; push_block(v);
    fill(1.0f); push_block(v);
    fill(-1.0f); v[0] = 1.0f; push_block(v);
    for (int e = 0; e < 128; ++e) v[e] = (e & 1) ? 1.0f : 2.0f; push_block(v);
    // Blocks with few distinct values (more chance of exact mse ties).
    for (int b = 0; b < 4000; ++b) {
        const float a = nd(rng), c = nd(rng) * 3.0f;
        for (int e = 0; e < 128; ++e) v[e] = (ud(rng) < 0.5f) ? a : c;
        push_block(v);
    }
    // Clamp boundaries of d0: +-7.5 d0 and -8.5 d0 walks with amax fixed.
    for (int t = 0; t < 2000; ++t) {
        const float amax = ldexpf(1.0f + ud(rng), (int)(ud(rng) * 60) - 30);
        const float d0 = (amax / 7.0f) * 0.5f * 0x1.b6db6ep+0f;
        v[0] = amax;
        for (int e = 1; e < 128; ++e) {
            const float c = (e % 3 == 0) ? 7.5f : (e % 3 == 1 ? -7.5f : -8.5f);
            float x = walk_ulps(c * d0, (int)(ud(rng) * 64) - 32);
            if (!(fabsf(x) <= amax)) x = copysignf(amax, x);
            v[e] = x;
        }
        push_block(v);
    }
    if (argc > 6 && atoi(argv[6])) {  // edge-only: keep just class D
        blocks = std::vector<float>(blocks.begin() + n_edge0 * 128, blocks.begin() + (n_edge0 + n_edge) * 128);
    }
    if ((blocks.size() / 128) & 1) { for (int e = 0; e < 128; ++e) v[e] = nd(rng); push_block(v); }
    const size_t n_blocks = blocks.size() / 128;
    // Shuffle block order so each wave pairs unrelated blocks in its two rows.
    {
        std::vector<size_t> perm(n_blocks);
        for (size_t i = 0; i < n_blocks; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<float> sh(blocks.size());
        for (size_t i = 0; i < n_blocks; ++i) memcpy(&sh[i * 128], &blocks[perm[i] * 128], 512);
        blocks.swap(sh);
    }
    const int waves = (int)(n_blocks / 2);
    const int N = 7, G = (waves + N - 1) / N;  // odd N: token-major sidecar index
    const size_t in_floats = (size_t)N * G * 256;
    blocks.resize(in_floats, 0.0f);

    hipModule_t mref, mnew; hipFunction_t fref, fnew;
    CK(hipModuleLoad(&mref, argv[1])); CK(hipModuleGetFunction(&fref, mref, "qemit"));
    CK(hipModuleLoad(&mnew, argv[2])); CK(hipModuleGetFunction(&fnew, mnew, "qemit"));
    float* din; CK(hipMalloc(&din, in_floats * 4));
    CK(hipMemcpy(din, blocks.data(), in_floats * 4, hipMemcpyHostToDevice));
    const size_t out_bytes = (size_t)2 * G * N * 72;
    void *oref, *onew;
    CK(hipMalloc(&oref, out_bytes)); CK(hipMalloc(&onew, out_bytes));
    CK(hipMemset(oref, 0xA5, out_bytes)); CK(hipMemset(onew, 0x5A, out_bytes));
    int n = N, g = G;
    void* pr[] = {&din, &oref, &n, &g};
    void* pn[] = {&din, &onew, &n, &g};
    const unsigned grid = (unsigned)(((size_t)N * G * 32 + 255) / 256);
    CK(hipModuleLaunchKernel(fref, grid, 1, 1, 256, 1, 1, 0, nullptr, pr, nullptr));
    CK(hipModuleLaunchKernel(fnew, grid, 1, 1, 256, 1, 1, 0, nullptr, pn, nullptr));
    CK(hipDeviceSynchronize());
    std::vector<unsigned char> a(out_bytes), b(out_bytes);
    CK(hipMemcpy(a.data(), oref, out_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), onew, out_bytes, hipMemcpyDeviceToHost));
    size_t diff = 0, bad_blocks = 0, first = (size_t)-1;
    for (size_t blk = 0; blk < out_bytes / 72; ++blk) {
        size_t bd = 0;
        for (int j = 0; j < 72; ++j) bd += a[blk * 72 + j] != b[blk * 72 + j];
        if (bd) { if (first == (size_t)-1) first = blk; ++bad_blocks; diff += bd; }
    }
    // Content census of the reference: d == 1 blocks, distinct d, sentinels.
    size_t unit = 0, untouched = 0;
    for (size_t blk = 0; blk < out_bytes / 72; ++blk) {
        Block_ r; memcpy(&r, &a[blk * 72], 72);
        unit += r.d == 1.0f;
        bool all = true; for (int j = 0; j < 72; ++j) all &= a[blk * 72 + j] == 0xA5;
        untouched += all;
    }
    printf("edge blocks %zu; ", n_edge);
    printf("blocks %zu (adversarial %zu), waves %d, ref d==1 blocks %zu, untouched %zu\n",
           out_bytes / 72, n_adv, N * G, unit, untouched);
    printf("differing bytes %zu in %zu blocks\n", diff, bad_blocks);
    if (bad_blocks) {
        Block_ r, c; memcpy(&r, &a[first * 72], 72); memcpy(&c, &b[first * 72], 72);
        printf("first block %zu: ref d %a s %d | new d %a s %d\n", first, r.d, r.s, c.d, c.s);
    }
    printf("%s\n", diff ? "QC2_FAIL" : "QC2_PASS");
    return diff ? 1 : 0;
}
