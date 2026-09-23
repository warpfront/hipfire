// Scratch GEMM v2 module host (host-only TU; device code lives in separately
// compiled HSACOs). Modes:
//   resources            print VGPR/private/occupancy of every loaded entry
//   oracle               exact parity corpus vs pinned X5 (+ CPU reference)
//   time <fwd|rev> [shape...]   full-shape timing, median of 7 HIP events
//   diag <fwd|rev> [shape...]   V2 diagnostic arms (not GEMMs) + X5 control
// Env: EXPECTED_ARCH, ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, X5_OBJECT,
//      V2A_OBJECT, V2B_OBJECT, [V2A_DIAG_OBJECTS/V2B_DIAG_OBJECTS: nostage;nofold]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <string>
#include <vector>

#include "oracle_inputs.hpp"

struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};
static_assert(sizeof(block_i4_128) == 72, "ABI");

#define OK(x)                                                                 \
    do {                                                                      \
        hipError_t s_ = (x);                                                  \
        if (s_ != hipSuccess) {                                               \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,         \
                    hipGetErrorString(s_));                                   \
            exit(2);                                                          \
        }                                                                     \
    } while (0)

static double wall() {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// ---------------------------------------------------------------- packing
struct Native {
    std::vector<uint32_t> wd;
    std::vector<uint16_t> ws;
    std::vector<uint32_t> xd;
    std::vector<float> xs;
};

static inline uint32_t rd32(const unsigned char* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static inline uint16_t rd16(const unsigned char* p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}
static inline size_t widx(size_t h, size_t r, size_t s, size_t q, size_t M) {
    return ((((h * (M / 16) + r / 16) * 8 + s) * 16 + (r & 15)) * 2 + q);
}

static Native pack(int M, int K, int N, const std::vector<unsigned char>& w,
                   const std::vector<block_i4_128>& x) {
    Native n;
    const size_t H = K / 128, G = K / 256;
    n.wd.assign((size_t)M * K / 8, 0);
    n.ws.assign(H * M, 0);
    n.xd.assign((size_t)N * K / 8, 0);
    n.xs.assign(H * N, 0);
    for (size_t r = 0; r < (size_t)M; ++r)
        for (size_t g = 0; g < G; ++g) {
            const unsigned char* gp = w.data() + (r * G + g) * 136;
            for (size_t hh = 0; hh < 2; ++hh) {
                const size_t h = 2 * g + hh;
                n.ws[h * M + r] = rd16(gp + 4 * hh);
                for (size_t s = 0; s < 8; ++s)
                    for (size_t q = 0; q < 2; ++q)
                        n.wd[widx(h, r, s, q, M)] =
                            rd32(gp + 8 + 64 * hh + 4 * (2 * s + q)) ^ 0x88888888u;
            }
        }
    for (size_t h = 0; h < H; ++h)
        for (size_t t = 0; t < (size_t)N; ++t) {
            const block_i4_128& b = x[h * N + t];
            n.xs[h * N + t] = b.d;
            for (size_t s = 0; s < 8; ++s)
                for (size_t q = 0; q < 2; ++q)
                    n.xd[widx(h, t, s, q, N)] = rd32(b.qs + 4 * (2 * s + q));
        }
    return n;
}

// Inverse: every original payload nibble, scale bits and X d must round-trip;
// the symmetric contract zp == -8*sc (FP16 bits) must hold for every group.
static bool unpack_check(int M, int K, int N, const std::vector<unsigned char>& w,
                         const std::vector<block_i4_128>& x, const Native& n,
                         size_t* bad_sym) {
    const size_t H = K / 128, G = K / 256;
    size_t sym = 0;
    for (size_t r = 0; r < (size_t)M; ++r)
        for (size_t g = 0; g < G; ++g) {
            const unsigned char* gp = w.data() + (r * G + g) * 136;
            for (size_t hh = 0; hh < 2; ++hh) {
                const size_t h = 2 * g + hh;
                const uint16_t sc = n.ws[h * M + r];
                if (sc != rd16(gp + 4 * hh)) return false;
                const uint16_t zp = rd16(gp + 4 * hh + 2);
                const float fsc = (float)*(const _Float16*)&sc;
                const float fzp = (float)*(const _Float16*)&zp;
                if (!(fzp == -8.0f * fsc && std::signbit(fzp) == !std::signbit(fsc)))
                    ++sym;
                for (size_t kk = 0; kk < 128; ++kk) {
                    const size_t s = kk / 16, q = (kk % 16) / 8, pos = kk & 7;
                    const unsigned nib =
                        ((n.wd[widx(h, r, s, q, M)] ^ 0x88888888u) >> (4 * pos)) & 15;
                    const unsigned char byte = gp[8 + 64 * hh + kk / 2];
                    if (nib != ((kk & 1) ? (byte >> 4) : (byte & 15u))) return false;
                }
            }
        }
    for (size_t h = 0; h < H; ++h)
        for (size_t t = 0; t < (size_t)N; ++t) {
            const block_i4_128& b = x[h * N + t];
            if (memcmp(&n.xs[h * N + t], &b.d, 4)) return false;
            for (size_t kk = 0; kk < 128; ++kk) {
                const size_t s = kk / 16, q = (kk % 16) / 8, pos = kk & 7;
                const unsigned nib = (n.xd[widx(h, t, s, q, N)] >> (4 * pos)) & 15;
                const unsigned char byte = b.qs[kk / 2];
                if (nib != ((kk & 1) ? (byte >> 4) : (byte & 15u))) return false;
            }
        }
    *bad_sym = sym;
    return true;
}

// CPU reference of the compiled X5 symfold DAG:
//   sum = fma(RN(float(sc_h)*d_h), float(C_h), sum), ascending h; ADD old+sum.
static void cpu_ref(int M, int K, int N, const Native& n, bool add,
                    const std::vector<float>& residual, std::vector<float>& out) {
    const size_t H = K / 128;
    out.assign((size_t)M * N, 0.f);
    std::vector<int8_t> wv((size_t)M * K), xv((size_t)N * K);
    auto sx = [](unsigned v) { return (int8_t)(v < 8 ? v : (int)v - 16); };
    for (size_t h = 0; h < H; ++h)
        for (size_t r = 0; r < (size_t)M; ++r)
            for (size_t kk = 0; kk < 128; ++kk)
                wv[r * K + h * 128 + kk] = sx(
                    (n.wd[widx(h, r, kk / 16, (kk % 16) / 8, M)] >> (4 * (kk & 7))) & 15);
    for (size_t h = 0; h < H; ++h)
        for (size_t t = 0; t < (size_t)N; ++t)
            for (size_t kk = 0; kk < 128; ++kk)
                xv[t * K + h * 128 + kk] = sx(
                    (n.xd[widx(h, t, kk / 16, (kk % 16) / 8, N)] >> (4 * (kk & 7))) & 15);
    for (size_t t = 0; t < (size_t)N; ++t)
        for (size_t r = 0; r < (size_t)M; ++r) {
            float sum = 0.0f;
            for (size_t h = 0; h < H; ++h) {
                int c = 0;
                const int8_t* a = &wv[r * K + h * 128];
                const int8_t* b = &xv[t * K + h * 128];
                for (int kk = 0; kk < 128; ++kk) c += a[kk] * b[kk];
                const uint16_t bits = n.ws[h * M + r];
                const float sc = (float)*(const _Float16*)&bits;
                const float t1 = sc * n.xs[h * N + t];
                sum = std::fmaf(t1, (float)c, sum);
            }
            const size_t o = t * M + r;
            out[o] = add ? residual[o] + sum : sum;
        }
}

// ---------------------------------------------------------------- corpus
enum class Corpus { Hashed, Witness, Extrema, OneHot, Scales };

static void recompute_s(block_i4_128& b) {
    b.s = 0;
    for (int j = 0; j < 64; ++j) {
        const int lo = b.qs[j] & 15, hi = b.qs[j] >> 4;
        b.s += (lo < 8 ? lo : lo - 16) + (hi < 8 ? hi : hi - 16);
    }
}

static void make_inputs(Corpus c, int M, int K, int N, std::vector<unsigned char>& w,
                        std::vector<block_i4_128>& x, int onehot_k) {
    xreuse_oracle::fill_inputs(M, K, N, w, x,
                               c == Corpus::Witness ? xreuse_oracle::Pattern::UnequalHalfWitness
                                                    : xreuse_oracle::Pattern::Hashed);
    const size_t G = K / 256;
    if (c == Corpus::Extrema) {
        // Rows cycle through (all -8), (all +7), (alternating -8/+7); tokens
        // through (all -8), (all +7), hashed. Signed products hit +-64/56/49.
        for (size_t r = 0; r < (size_t)M; ++r)
            for (size_t g = 0; g < G; ++g) {
                unsigned char* p = w.data() + (r * G + g) * 136 + 8;
                const unsigned char v = (r % 3 == 0) ? 0x00 : (r % 3 == 1) ? 0xFF : 0xF0;
                memset(p, v, 128);
            }
        for (size_t b = 0; b < x.size(); ++b) {
            const size_t t = b % N;
            if (t % 3 == 0) memset(x[b].qs, 0x88, 64);
            else if (t % 3 == 1) memset(x[b].qs, 0x77, 64);
            recompute_s(x[b]);
        }
    } else if (c == Corpus::OneHot) {
        // All weight codes 8 (rebased 0) except K position onehot_k, whose
        // code is 1 + (r % 15) (rebased -7..+7, never 0 for code 8).
        for (size_t r = 0; r < (size_t)M; ++r)
            for (size_t g = 0; g < G; ++g) memset(w.data() + (r * G + g) * 136 + 8, 0x88, 128);
        for (size_t r = 0; r < (size_t)M; ++r) {
            const size_t g = onehot_k / 256, kk = onehot_k % 256;
            unsigned char* byte = w.data() + (r * G + g) * 136 + 8 + kk / 2;
            unsigned code = (1 + r % 15);
            if (code == 8) code = 15;
            *byte = (kk & 1) ? (unsigned char)((*byte & 0x0F) | (code << 4))
                             : (unsigned char)((*byte & 0xF0) | code);
        }
    } else if (c == Corpus::Scales) {
        // Scale edge values: FP16 zero, subnormal, 8188 (8x = FP16 max), negative;
        // X d zero,
        // negative, tiny. Symmetric zp=-8*sc is kept exact for each.
        const uint16_t specials[] = {0x0000, 0x0001, 0x6FFF, 0xBC00, 0x03FF, 0x3C00};
        for (size_t r = 0; r < (size_t)M; ++r)
            for (size_t g = 0; g < G; ++g)
                for (int hh = 0; hh < 2; ++hh) {
                    if ((r + g + hh) % 4) continue;
                    const uint16_t sc = specials[(r * 7 + g * 3 + hh) % 6];
                    const float z = -8.0f * (float)*(const _Float16*)&sc;
                    const _Float16 zh = (_Float16)z;
                    uint16_t zb;
                    memcpy(&zb, &zh, 2);
                    unsigned char* gp = w.data() + (r * G + g) * 136 + 4 * hh;
                    memcpy(gp, &sc, 2);
                    memcpy(gp + 2, &zb, 2);
                }
        const float dspecial[] = {0.0f, -0.0073f, 1e-30f, -3.5f};
        for (size_t b = 0; b < x.size(); ++b)
            if (b % 5 == 0) x[b].d = dspecial[(b / 5) % 4];
    }
}

// ---------------------------------------------------------------- modules
struct Arm {
    std::string name;
    bool x5 = false;
    bool prod = false;  // production layout (A, Xq, Y, M, K, N): v2c_prod / v2b_prod
    int tn = 0;       // V2 token tile
    int tm = 256;     // V2 row tile
    int waves = 16;   // waves per CTA
    int lds = 0;
    hipFunction_t set = nullptr, add = nullptr;
};

static std::vector<Arm> g_arms;

static hipModule_t load(const char* env) {
    const char* p = getenv(env);
    if (!p) {
        fprintf(stderr, "missing %s\n", env);
        exit(3);
    }
    hipModule_t m;
    OK(hipModuleLoad(&m, p));
    printf("OBJECT %s=%s\n", env, p);
    return m;
}

// Geometry by stem: v2b M256xN256 16 waves 65,536 B; v2a M256xN128 16 waves
// 51,200 B; v2c M128xN128 8 waves 32,768 B; v2d M128xN128 8 waves 16,384 B.
static void add_v2(const char* env, const char* stem, const char* suffix, int /*tn*/,
                   const char* arm_name = nullptr) {
    hipModule_t m = load(env);
    Arm a;
    a.name = arm_name ? arm_name : std::string(stem) + suffix;
    const std::string st(stem);
    const bool small = st == "v2c" || st == "v2d";
    a.tn = st == "v2b" ? 256 : 128;
    a.tm = small ? 128 : 256;
    a.waves = small ? 8 : 16;
    a.lds = st == "v2b" ? 65536 : st == "v2a" ? 51200 : st == "v2c" ? 32768 : 16384;
    std::string s = "iu4_native_" + std::string(stem) + "_set" + suffix + "_gfx11";
    std::string d = "iu4_native_" + std::string(stem) + "_add" + suffix + "_gfx11";
    OK(hipModuleGetFunction(&a.set, m, s.c_str()));
    OK(hipModuleGetFunction(&a.add, m, d.c_str()));
    g_arms.push_back(a);
}

static hipFunction_t g_repack = nullptr;

static void init_modules(bool diag) {
    hipModule_t x5 = load("X5_OBJECT");
    Arm a;
    a.name = "x5";
    a.x5 = true;
    a.lds = 30720;
    OK(hipModuleGetFunction(&a.set, x5, "gemm_mq4g256v2_residual_mmq_iu4_full_set_x5_col_gfx1151_symfold"));
    OK(hipModuleGetFunction(&a.add, x5, "gemm_mq4g256v2_residual_mmq_iu4_full_add_x5_col_gfx1151_symfold"));
    g_arms.push_back(a);
    if (!getenv("SKIP_V2AB")) {
        add_v2("V2A_OBJECT", "v2a", "", 128);
        add_v2("V2B_OBJECT", "v2b", "", 256);
    }
    // Optional exact alternative fold spellings (MAGIC+SCF32), same symbols.
    if (getenv("V2A_MS_OBJECT")) add_v2("V2A_MS_OBJECT", "v2a", "", 128, "v2a_ms");
    if (getenv("V2B_MS_OBJECT")) add_v2("V2B_MS_OBJECT", "v2b", "", 256, "v2b_ms");
    // Further exact variants: EXTRA_ARMS="name=ENVVAR:stem,..." (stem v2a/v2b/v2c
    // gives geometry and entry symbols).
    if (const char* ex = getenv("EXTRA_ARMS")) {
        std::string all(ex);
        size_t pos = 0;
        while (pos < all.size()) {
            size_t end = all.find(',', pos);
            if (end == std::string::npos) end = all.size();
            const std::string item = all.substr(pos, end - pos);
            const size_t eq = item.find('='), colon = item.find(':');
            const std::string name = item.substr(0, eq);
            const std::string envv = item.substr(eq + 1, colon - eq - 1);
            const std::string stem = item.substr(colon + 1);
            add_v2(envv.c_str(), stem.c_str(), "", 0, name.c_str());
            pos = end + 1;
        }
    }
    // Production-layout arms: V2C (M128xN128, 8 waves) and V2B (M256xN256,
    // 16 waves), both with the 6-argument (A, Xq, Y, M, K, N) ABI.
    auto add_prod = [](const char* env, const char* name, int tile, int waves, int lds,
                       const char* set, const char* add) {
        hipModule_t m = load(env);
        Arm p;
        p.name = name;
        p.prod = true;
        p.tn = tile;
        p.tm = tile;
        p.waves = waves;
        p.lds = lds;
        OK(hipModuleGetFunction(&p.set, m, set));
        OK(hipModuleGetFunction(&p.add, m, add));
        g_arms.push_back(p);
    };
    if (getenv("V2CP_OBJECT"))
        add_prod("V2CP_OBJECT", "v2c_prod", 128, 8,
                 getenv("V2CP_LDS") ? atoi(getenv("V2CP_LDS")) : 32768,
                 "gemm_mq4g256v2_residual_iu4_v2c_set_gfx11",
                 "gemm_mq4g256v2_residual_iu4_v2c_add_gfx11");
    if (getenv("V2BP_OBJECT"))
        add_prod("V2BP_OBJECT", "v2b_prod", 256, 16, 65536,
                 "gemm_mq4g256v2_residual_iu4_v2b_set_gfx11",
                 "gemm_mq4g256v2_residual_iu4_v2b_add_gfx11");
    // Scratch variants of the production V2B kernel (same symbols/ABI):
    // V2BP_EXTRA="name=/path/a.hsaco;name2=/path/b.hsaco".
    if (const char* ex = getenv("V2BP_EXTRA")) {
        static std::vector<std::string> keep;
        std::string all(ex);
        size_t pos = 0;
        while (pos < all.size()) {
            size_t end = all.find(';', pos);
            if (end == std::string::npos) end = all.size();
            const std::string item = all.substr(pos, end - pos);
            const size_t eq = item.find('=');
            const std::string name = item.substr(0, eq), path = item.substr(eq + 1);
            const std::string envv = "V2BP_EXTRA_" + name;
            setenv(envv.c_str(), path.c_str(), 1);
            keep.push_back(name);
            keep.push_back(envv);
            add_prod(keep[keep.size() - 1].c_str(), keep[keep.size() - 2].c_str(), 256, 16, 65536,
                     "gemm_mq4g256v2_residual_iu4_v2b_set_gfx11",
                     "gemm_mq4g256v2_residual_iu4_v2b_add_gfx11");
            pos = end + 1;
        }
    }
    {
        hipModule_t m;
        OK(hipModuleLoad(&m, getenv("V2B_OBJECT")));
        OK(hipModuleGetFunction(&g_repack, m, "iu4_native_x_repack_gfx11"));
    }
    if (diag) {
        add_v2("V2A_NOSTAGE_OBJECT", "v2a", "_nostage", 128);
        add_v2("V2B_NOSTAGE_OBJECT", "v2b", "_nostage", 256);
        add_v2("V2A_NOFOLD_OBJECT", "v2a", "_nofold", 128);
        add_v2("V2B_NOFOLD_OBJECT", "v2b", "_nofold", 256);
    }
}

// ---------------------------------------------------------------- device data
struct Dev {
    int M = 0, K = 0, N = 0;
    char* a = nullptr;           // X5 weights (136 B groups)
    block_i4_128* xq = nullptr;  // X5 activations
    uint32_t* wd = nullptr;
    uint16_t* ws = nullptr;
    uint32_t* xd = nullptr;
    float* xs = nullptr;
    float* y = nullptr;
    float* resid = nullptr;
};

template <class T>
static T* upload(const std::vector<T>& v) {
    T* p;
    OK(hipMalloc(&p, std::max<size_t>(v.size() * sizeof(T), 256)));
    OK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

static Dev make_dev(int M, int K, int N, const std::vector<unsigned char>& w,
                    const std::vector<block_i4_128>& x, const Native& n,
                    const std::vector<float>& resid) {
    Dev d;
    d.M = M, d.K = K, d.N = N;
    d.a = (char*)upload(w);
    d.xq = upload(x);
    d.wd = upload(n.wd);
    d.ws = upload(n.ws);
    d.xd = upload(n.xd);
    d.xs = upload(n.xs);
    d.resid = upload(resid);
    OK(hipMalloc(&d.y, (size_t)M * N * 4));
    return d;
}

static void free_dev(Dev& d) {
    for (void* p : {(void*)d.a, (void*)d.xq, (void*)d.wd, (void*)d.ws, (void*)d.xd,
                    (void*)d.xs, (void*)d.y, (void*)d.resid})
        OK(hipFree(p));
}

static void launch(const Arm& a, Dev& d, bool add, hipStream_t st = nullptr) {
    int M = d.M, K = d.K, N = d.N;
    if (a.x5) {
        int av = add;
        void* args[] = {&d.a, &d.xq, &d.y, &M, &K, &N, &av};
        OK(hipModuleLaunchKernel(add ? a.add : a.set, N / 128, M / 128, 1, 32, 16, 1,
                                 a.lds, st, args, nullptr));
    } else if (a.prod) {
        void* args[] = {&d.a, &d.xq, &d.y, &M, &K, &N};
        OK(hipModuleLaunchKernel(add ? a.add : a.set, N / a.tn, M / a.tm, 1, 32, a.waves, 1,
                                 a.lds, st, args, nullptr));
    } else {
        void* args[] = {&d.wd, &d.ws, &d.xd, &d.xs, &d.y, &M, &K, &N};
        OK(hipModuleLaunchKernel(add ? a.add : a.set, N / a.tn, M / a.tm, 1, 32, a.waves, 1,
                                 a.lds, st, args, nullptr));
    }
}

static void reset_y(Dev& d, bool add) {
    if (add)
        OK(hipMemcpy(d.y, d.resid, (size_t)d.M * d.N * 4, hipMemcpyDeviceToDevice));
    else
        OK(hipMemset(d.y, 0x7f, (size_t)d.M * d.N * 4));  // stale NaN-ish fill
}

static std::vector<float> run_once(const Arm& a, Dev& d, bool add) {
    reset_y(d, add);
    launch(a, d, add);
    OK(hipDeviceSynchronize());
    std::vector<float> h((size_t)d.M * d.N);
    OK(hipMemcpy(h.data(), d.y, h.size() * 4, hipMemcpyDeviceToHost));
    return h;
}

// ---------------------------------------------------------------- resources
static void resources() {
    for (auto& a : g_arms)
        for (int add = 0; add < 2; ++add) {
            hipFunction_t f = add ? a.add : a.set;
            int regs = 0, priv = 0, shared = 0, occ = 0;
            OK(hipFuncGetAttribute(&regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, f));
            OK(hipFuncGetAttribute(&priv, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f));
            OK(hipFuncGetAttribute(&shared, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f));
            OK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, 32 * a.waves, a.lds));
            printf("RESOURCE arm=%s add=%d vgpr=%d private=%d static_lds=%d dynamic_lds=%d "
                   "blocks_per_mp=%d waves_per_mp=%d\n",
                   a.name.c_str(), add, regs, priv, shared, a.lds, occ, occ * a.waves);
            if (priv || occ < 1) {
                fprintf(stderr, "REJECT_RESOURCE %s\n", a.name.c_str());
                exit(4);
            }
        }
    fflush(stdout);
}

// ---------------------------------------------------------------- oracle
static const char* corpus_name(Corpus c) {
    switch (c) {
        case Corpus::Hashed: return "hashed";
        case Corpus::Witness: return "witness";
        case Corpus::Extrema: return "extrema";
        case Corpus::OneHot: return "onehot";
        default: return "scales";
    }
}

static size_t g_total_cmp = 0;

static void oracle_case(Corpus c, int M, int K, int N, int onehot_k, bool cpu) {
    std::vector<unsigned char> w;
    std::vector<block_i4_128> x;
    make_inputs(c, M, K, N, w, x, onehot_k);
    Native n = pack(M, K, N, w, x);
    size_t bad_sym = 0;
    if (!unpack_check(M, K, N, w, x, n, &bad_sym) || bad_sym) {
        fprintf(stderr, "PACK_FAIL corpus=%s bad_sym=%zu\n", corpus_name(c), bad_sym);
        exit(5);
    }
    std::vector<float> resid((size_t)M * N);
    for (size_t i = 0; i < resid.size(); ++i)
        resid[i] = c == Corpus::Witness ? 0.5f : xreuse_oracle::residual(i);
    Dev d = make_dev(M, K, N, w, x, n, resid);
    for (int add = 0; add < 2; ++add) {
        std::vector<float> ref = run_once(g_arms[0], d, add);
        std::vector<float> cref;
        if (cpu) cpu_ref(M, K, N, n, add, resid, cref);
        size_t cpu_diff = 0;
        if (cpu)
            for (size_t i = 0; i < ref.size(); ++i) cpu_diff += memcmp(&ref[i], &cref[i], 4) != 0;
        if (c == Corpus::Witness) {
            const float want = add ? K / 2 + 0.5f : (float)(K / 2);
            for (float v : ref)
                if (v != want) {
                    fprintf(stderr, "WITNESS_FAIL x5 %g != %g\n", v, want);
                    exit(5);
                }
        }
        printf("ORACLE arm=x5 corpus=%s M=%d K=%d N=%d add=%d onehot_k=%d compared=%zu "
               "cpu_checked=%d cpu_differing=%zu\n",
               corpus_name(c), M, K, N, add, onehot_k, ref.size(), (int)cpu, cpu_diff);
        if (cpu_diff && c != Corpus::Scales) exit(5);
        for (size_t ai = 1; ai < g_arms.size(); ++ai) {
            const Arm& a = g_arms[ai];
            std::vector<float> out = run_once(a, d, add);
            size_t diff = 0, first = (size_t)-1;
            for (size_t i = 0; i < out.size(); ++i)
                if (memcmp(&out[i], &ref[i], 4)) {
                    if (!diff) first = i;
                    ++diff;
                }
            printf("ORACLE arm=%s corpus=%s M=%d K=%d N=%d add=%d onehot_k=%d compared=%zu "
                   "differing=%zu",
                   a.name.c_str(), corpus_name(c), M, K, N, add, onehot_k, out.size(), diff);
            if (diff)
                printf(" first=%zu (token=%zu row=%zu) got=%.9g want=%.9g", first, first / M,
                       first % M, out[first], ref[first]);
            printf("\n");
            fflush(stdout);
            g_total_cmp += out.size();
            if (diff) {
                fprintf(stderr, "PARITY_FAIL %s\n", a.name.c_str());
                exit(5);
            }
        }
    }
    free_dev(d);
}

static void oracle() {
    // Per-CTA exactness at the five K values, both modes, hashed corpus.
    for (int K : {256, 512, 5120, 6144, 17408}) oracle_case(Corpus::Hashed, 256, K, 256, -1, K <= 6144);
    oracle_case(Corpus::Witness, 256, 256, 256, -1, true);
    oracle_case(Corpus::Witness, 256, 512, 256, -1, true);
    oracle_case(Corpus::Extrema, 256, 512, 256, -1, true);
    oracle_case(Corpus::Scales, 256, 1024, 256, -1, true);
    for (int k : {0, 127, 128, 255, 256, 511}) oracle_case(Corpus::OneHot, 256, 512, 256, k, true);
    oracle_case(Corpus::OneHot, 256, 5120, 256, 5119, true);
    // Inter-CTA indexing: multiple row and token tiles, non-square grid.
    oracle_case(Corpus::Hashed, 512, 1024, 768, -1, true);
    oracle_case(Corpus::Hashed, 768, 1536, 512, -1, false);
    printf("ORACLE_PASS total_v2_outputs_compared=%zu\n", g_total_cmp);
}

// ---------------------------------------------------------------- timing
struct Shape {
    const char* name;
    int M, K;
    bool add;
    int N;
};
// Every SET/ADD shape of the Qwen3.8-27B pp8192 Halo trace (N=8192 single
// chunk) plus the same GEMMs at N=512 (pp512).
static const Shape kShapes[] = {{"set_m17408", 17408, 5120, false, 8192},
                                {"set_m12288", 12288, 5120, false, 8192},
                                {"set_m10240", 10240, 5120, false, 8192},
                                {"set_m6144", 6144, 5120, false, 8192},
                                {"set_m1024", 1024, 5120, false, 8192},
                                {"add_k17408", 5120, 17408, true, 8192},
                                {"add_k6144", 5120, 6144, true, 8192},
                                {"set_m17408_n512", 17408, 5120, false, 512},
                                {"set_m12288_n512", 12288, 5120, false, 512},
                                {"set_m10240_n512", 10240, 5120, false, 512},
                                {"set_m6144_n512", 6144, 5120, false, 512},
                                {"set_m1024_n512", 1024, 5120, false, 512},
                                {"add_k17408_n512", 5120, 17408, true, 512},
                                {"add_k6144_n512", 5120, 6144, true, 512},
                                // Diagnostic only (never in the default list):
                                // the SET entry on the ADD shapes isolates the
                                // residual read of the ADD epilogue.
                                {"setdiag_m5120_k6144", 5120, 6144, false, 8192},
                                {"setdiag_m5120_k17408", 5120, 17408, false, 8192}};

static void timing(bool reverse, const std::vector<std::string>& only, bool diag) {
    // Shape filters may also be ad-hoc "set:M:K:N" / "add:M:K:N" (boundary sweep).
    std::vector<Shape> list(std::begin(kShapes), std::end(kShapes));
    static std::vector<std::string> names;
    names.reserve(only.size());
    for (const std::string& o : only) {
        int M, K, N;
        char kind[8];
        if (sscanf(o.c_str(), "%3[a-z]:%d:%d:%d", kind, &M, &K, &N) == 4) {
            names.push_back(o);
            list.push_back({names.back().c_str(), M, K, !strcmp(kind, "add"), N});
        }
    }
    for (const Shape& sh : list) {
        const int N = sh.N;
        if (!only.empty() && std::find(only.begin(), only.end(), sh.name) == only.end()) continue;
        if (only.empty() && !strncmp(sh.name, "setdiag", 7)) continue;
        std::vector<unsigned char> w;
        std::vector<block_i4_128> x;
        make_inputs(Corpus::Hashed, sh.M, sh.K, N, w, x, -1);
        Native n = pack(sh.M, sh.K, N, w, x);
        std::vector<float> resid((size_t)sh.M * N);
        for (size_t i = 0; i < resid.size(); ++i) resid[i] = xreuse_oracle::residual(i);
        Dev d = make_dev(sh.M, sh.K, N, w, x, n, resid);
        std::vector<size_t> order(g_arms.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = reverse ? order.size() - 1 - i : i;
        hipEvent_t b, e;
        OK(hipEventCreate(&b));
        OK(hipEventCreate(&e));
        const double tops = 2.0 * sh.M * (double)sh.K * N;
        for (size_t oi : order) {
            const Arm& a = g_arms[oi];
            // Warm up this arm for >= 4 launches and >= 0.6 s of wall time so
            // the clock/thermal state is the arm's own steady state.
            {
                const double w0 = wall();
                for (int j = 0; j < 4 || wall() - w0 < 0.6; ++j) {
                    reset_y(d, sh.add);
                    launch(a, d, sh.add);
                    OK(hipDeviceSynchronize());
                }
            }
            std::vector<float> ms;
            const double t0 = wall();
            for (int j = 0; j < 7; ++j) {
                reset_y(d, sh.add);
                OK(hipDeviceSynchronize());
                OK(hipEventRecord(b));
                launch(a, d, sh.add);
                OK(hipEventRecord(e));
                OK(hipEventSynchronize(e));
                float m;
                OK(hipEventElapsedTime(&m, b, e));
                ms.push_back(m);
            }
            const double t1 = wall();
            std::vector<float> sorted = ms;
            std::sort(sorted.begin(), sorted.end());
            const double med = sorted[3];
            printf("RESULT arm=%s shape=%s M=%d K=%d N=%d add=%d ms=%.6f tops=%.6f "
                   "pct_of_105.543=%.4f min=%.6f max=%.6f begin=%.6f end=%.6f all=",
                   a.name.c_str(), sh.name, sh.M, sh.K, N, (int)sh.add, med, tops / (med * 1e9),
                   100.0 * tops / (med * 1e9) / 105.543, sorted[0], sorted[6], t0, t1);
            for (float m : ms) printf("%.4f,", m);
            printf("\n");
            fflush(stdout);
        }
        if (!diag) {
            // X AoS->SoA device repack, the producer-side cost of the native X.
            std::vector<float> ms;
            const long long total = (long long)(sh.K / 128) * N;
            for (int j = 0; j < 9; ++j) {
                int K = sh.K, NN = N;
                void* args[] = {&d.xq, &d.xd, &d.xs, &K, &NN};
                OK(hipEventRecord(b));
                OK(hipModuleLaunchKernel(g_repack, (unsigned)((total + 255) / 256), 1, 1, 256, 1,
                                         1, 0, nullptr, args, nullptr));
                OK(hipEventRecord(e));
                OK(hipEventSynchronize(e));
                float m;
                OK(hipEventElapsedTime(&m, b, e));
                if (j >= 2) ms.push_back(m);
            }
            std::sort(ms.begin(), ms.end());
            std::vector<uint32_t> xd2(n.xd.size());
            std::vector<float> xs2(n.xs.size());
            OK(hipMemcpy(xd2.data(), d.xd, xd2.size() * 4, hipMemcpyDeviceToHost));
            OK(hipMemcpy(xs2.data(), d.xs, xs2.size() * 4, hipMemcpyDeviceToHost));
            const bool same = !memcmp(xd2.data(), n.xd.data(), xd2.size() * 4) &&
                              !memcmp(xs2.data(), n.xs.data(), xs2.size() * 4);
            printf("REPACK shape=%s K=%d N=%d ms=%.6f bytes_in=%lld identical_to_cpu_pack=%d\n",
                   sh.name, sh.K, N, ms[3], total * 72, (int)same);
            fflush(stdout);
            if (!same) exit(5);
        }
        OK(hipEventDestroy(b));
        OK(hipEventDestroy(e));
        free_dev(d);
    }
}

// CPU emulation of the V2 device kernel's data path (same LDS byte layout,
// lane addressing, A-row permutation, metadata offsets and epilogue offsets),
// with WMMA modeled by its ISA definition: A lane m<16 supplies row m, B lane n
// supplies column n, D[m][n] lands in lane n+16*(m&1), VGPR m>>1. Validates the
// index algebra only; GPU parity remains the authority.
static int emulate(int TN, Corpus corpus, int M, int K, int N, int onehot_k) {
    std::vector<unsigned char> w;
    std::vector<block_i4_128> x;
    make_inputs(corpus, M, K, N, w, x, onehot_k);
    Native n = pack(M, K, N, w, x);
    size_t bad_sym = 0;
    if (!unpack_check(M, K, N, w, x, n, &bad_sym) || bad_sym) return 10;
    std::vector<float> resid((size_t)M * N), ref, out((size_t)M * N, NAN);
    for (size_t i = 0; i < resid.size(); ++i) resid[i] = xreuse_oracle::residual(i);
    cpu_ref(M, K, N, n, true, resid, ref);
    const int NA = TN == 256 ? 4 : 2, WAVES_N = TN / 64, WAVE_ROWS = 16 * NA;
    const bool meta_lds = TN == 128;
    const int A_BYTES = 256 * 64, X_BYTES = TN * 64, WS_BYTES = meta_lds ? 512 : 0;
    const int E = K / 128;
    std::vector<unsigned char> slot(A_BYTES + X_BYTES + (meta_lds ? 512 + TN * 4 : 0));
    auto nib = [](const unsigned char* p, int k) {
        uint32_t wv;
        memcpy(&wv, p + 4 * (k / 8), 4);
        const unsigned v = (wv >> (4 * (k & 7))) & 15;
        return v < 8 ? (int)v : (int)v - 16;
    };
    for (int by = 0; by < M / 256; ++by)
        for (int bx = 0; bx < N / TN; ++bx) {
            const int row0 = by * 256, col0 = bx * TN;
            std::vector<float> sum(16 * 32 * NA * 4 * 8, 0.0f);
            for (int e = 0; e < E; ++e) {
                memcpy(slot.data(), (const char*)n.wd.data() + (size_t)e * M * 64 + (size_t)row0 * 64, A_BYTES);
                memcpy(slot.data() + A_BYTES, (const char*)n.xd.data() + (size_t)e * N * 64 + (size_t)col0 * 64, X_BYTES);
                if (meta_lds) {
                    memcpy(slot.data() + A_BYTES + X_BYTES, &n.ws[(size_t)e * M + row0], 512);
                    memcpy(slot.data() + A_BYTES + X_BYTES + 512, &n.xs[(size_t)e * N + col0], TN * 4);
                }
                for (int wave = 0; wave < 16; ++wave) {
                    const int wr = wave / WAVES_N, wc = wave % WAVES_N;
                    auto aoff = [&](int lane) { const int lr = lane & 15; return wr * NA * 1024 + (8 * (lr & 1) + (lr >> 1)) * 8; };
                    auto xoff = [&](int lane) { return A_BYTES + wc * 4 * 1024 + (lane & 15) * 8; };
                    for (int a = 0; a < NA; ++a)
                        for (int c = 0; c < 4; ++c) {
                            int D[16][16] = {};
                            for (int s = 0; s < 8; ++s)
                                for (int m = 0; m < 16; ++m)
                                    for (int nn = 0; nn < 16; ++nn) {
                                        const unsigned char* ap = slot.data() + aoff(m) + a * 1024 + s * 128;
                                        const unsigned char* bp = slot.data() + xoff(nn) + c * 1024 + s * 128;
                                        for (int k = 0; k < 16; ++k) D[m][nn] += nib(ap, k) * nib(bp, k);
                                    }
                            for (int lane = 0; lane < 32; ++lane) {
                                const int lr = lane & 15, hi = lane >> 4;
                                float dc;
                                uint16_t P[8];
                                if (meta_lds) {
                                    memcpy(&dc, slot.data() + A_BYTES + X_BYTES + WS_BYTES + (wc * 64 + lr) * 4 + 64 * c, 4);
                                    memcpy(P, slot.data() + A_BYTES + X_BYTES + (wr * WAVE_ROWS + 8 * hi) * 2 + 32 * a, 16);
                                } else {
                                    dc = n.xs[(size_t)e * N + col0 + wc * 64 + lr + 16 * c];
                                    memcpy(P, &n.ws[(size_t)e * M + row0 + wr * WAVE_ROWS + 8 * hi + 16 * a], 16);
                                }
                                for (int j = 0; j < 8; ++j) {
                                    const int Cv = D[2 * j + hi][lr];
                                    const float sc = (float)*(const _Float16*)&P[j];
                                    float& sm = sum[(((wave * 32 + lane) * NA + a) * 4 + c) * 8 + j];
                                    sm = std::fmaf(sc * dc, (float)Cv, sm);
                                }
                            }
                        }
                }
            }
            for (int wave = 0; wave < 16; ++wave) {
                const int wr = wave / WAVES_N, wc = wave % WAVES_N;
                for (int lane = 0; lane < 32; ++lane)
                    for (int a = 0; a < NA; ++a)
                        for (int c = 0; c < 4; ++c)
                            for (int j = 0; j < 8; ++j) {
                                const size_t token = col0 + wc * 64 + 16 * c + (lane & 15);
                                const size_t row = row0 + wr * WAVE_ROWS + 16 * a + 8 * (lane >> 4) + j;
                                const size_t o = token * M + row;
                                if (!std::isnan(out[o])) return 11;  // double owner
                                out[o] = resid[o] + sum[(((wave * 32 + lane) * NA + a) * 4 + c) * 8 + j];
                            }
            }
        }
    size_t diff = 0;
    for (size_t i = 0; i < out.size(); ++i) diff += memcmp(&out[i], &ref[i], 4) != 0;
    printf("EMULATE tn=%d corpus=%s M=%d K=%d N=%d onehot_k=%d outputs=%zu differing=%zu\n", TN,
           corpus_name(corpus), M, K, N, onehot_k, out.size(), diff);
    fflush(stdout);
    return diff ? 12 : 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && !strcmp(argv[1], "emulate")) {
        int rc = 0;
        for (int tn : {128, 256}) {
            rc |= emulate(tn, Corpus::Hashed, 256, 512, 256, -1);
            rc |= emulate(tn, Corpus::Witness, 256, 256, 256, -1);
            rc |= emulate(tn, Corpus::Scales, 256, 512, 256, -1);
            for (int k : {0, 127, 128, 255, 256, 511}) rc |= emulate(tn, Corpus::OneHot, 256, 512, 256, k);
            rc |= emulate(tn, Corpus::Hashed, 512, 256, 768, -1);
        }
        printf("EMULATE_DONE rc=%d\n", rc);
        return rc;
    }
    const char* rv = getenv("ROCR_VISIBLE_DEVICES");
    const char* hv = getenv("HIP_VISIBLE_DEVICES");
    const char* arch = getenv("EXPECTED_ARCH");
    if (!rv || !hv || !arch || strcmp(hv, "0")) return 3;
    if (!strcmp(arch, "gfx1151") && strcmp(rv, "1")) return 3;
    if (!strcmp(arch, "gfx1100") && strcmp(rv, "0")) return 3;
    hipDeviceProp_t p{};
    OK(hipGetDeviceProperties(&p, 0));
    printf("DEVICE arch=%s name=%s mps=%d ROCR=%s HIP=%s pid=%d\n", p.gcnArchName, p.name,
           p.multiProcessorCount, rv, hv, (int)getpid());
    fflush(stdout);
    if (strncmp(p.gcnArchName, arch, strlen(arch))) return 3;
    const std::string mode = argc > 1 ? argv[1] : "resources";
    const bool diag = mode == "diag";
    init_modules(diag);
    resources();
    if (mode == "oracle") oracle();
    else if (mode == "time" || diag) {
        const bool rev = argc > 2 && !strcmp(argv[2], "rev");
        std::vector<std::string> only;
        for (int i = 3; i < argc; ++i) only.push_back(argv[i]);
        timing(rev, only, diag);
    }
    return 0;
}
