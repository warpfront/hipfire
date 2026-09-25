// SPDX-License-Identifier: Apache-2.0
// DIAGNOSTIC host runner for `peacemaker profile` kernels (never production).
//
//   pmprof f2   <base.hsaco> <profiled.co> <N> <outdir> [rounds]
//   pmprof attn <base.hsaco> <profiled.co> <batch> <start> <outdir> [rounds]
//
// For one real H2 shape it runs the uninstrumented and the profiled kernel on
// identical inputs, compares every output byte (outputs are poisoned before
// each launch), times both interleaved (event timing), and writes the trace
// buffer of a steady-state profiled launch (after the timed loop) to
// <outdir>/trace.bin with meta.json.
// Kernel arguments go through hipModuleLaunchKernel's kernelParams so the
// runtime lays out explicit and hidden arguments from the code-object
// metadata; the profiled kernel's five extension arguments are appended.
//
// f2: gemm_mq4g256v2_fp8_silu_row_b1, gate/up M0 = M1 = 17408, K = 5120,
//     grid [2*M0/256, N/128], block 256, LDS 19456. Synthetic deterministic
//     repacked weights/ratios/activations (timing is data-independent).
// attn: attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201 (KT48), 24 Q / 4 KV
//     heads, d 256, grid [ceil(min(B,512)*6/384), 4, ceil(B/512)], block 768,
//     LDS 49936; inputs generated exactly as fp8-ng/attn/harness/attn_bench.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); exit(1);} } while (0)

static uint32_t salt(uint64_t i, uint32_t s) {
    uint64_t x = i * 0x9e3779b97f4a7c15ull + s;
    x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29;
    return (uint32_t)x;
}

// ---- attention inputs: identical to fp8-ng/attn/harness/attn_bench.cpp ----
static constexpr int NQ = 24, NKV = 4, D = 256;
__device__ __forceinline__ float hrand(uint32_t i, uint32_t s) {
    uint32_t x = (i * 0x9e3779b1u + s) * 0x85ebca6bu;
    x ^= x >> 16; x *= 0xc2b2ae35u; x ^= x >> 16;
    return (float)x * (1.0f / 4294967295.0f) * 2.0f - 1.0f;
}
__device__ __forceinline__ float gauss(uint32_t i, uint32_t s) {
    return 0.5f * (hrand(4 * i, s) + hrand(4 * i + 1, s) + hrand(4 * i + 2, s) + hrand(4 * i + 3, s));
}
__global__ void make_q8(unsigned char* q_codes, float* q_scale, int rows, uint32_t s) {
    const int lane = threadIdx.x & 31;
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const float mag = 0.25f + 2.0f * (0.5f + 0.5f * hrand((uint32_t)r, s ^ 0x777u));
    float o[8];
    for (int k = 0; k < 8; ++k) o[k] = mag * gauss((uint32_t)(r * 256 + lane + 32 * k), s);
    float a = 0.f;
    for (int k = 0; k < 8; ++k) a = fmaxf(a, fabsf(o[k]));
    for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor(a, off, 32));
    const float scale = (a == 0.0f) ? 1.0f : a / 448.0f;
    unsigned char* dst = q_codes + r * 256 + ((lane >> 3) & 1) * 128 + (lane >> 4) * 8 + (lane & 7);
    for (int k = 0; k < 8; ++k) {
        const unsigned code = __builtin_amdgcn_cvt_pk_fp8_f32(o[k] / scale, 0.0f, 0u, false) & 0xffu;
        dst[k * 16] = (unsigned char)code;
    }
    if (lane == 0) q_scale[r] = scale;
}
__global__ void make_kv(unsigned char* cache, int tokens, uint32_t s) {
    const int lane = threadIdx.x & 31;
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= (long long)tokens * 4) return;
    const long long t = r >> 2; const int h = (int)(r & 3);
    float o[8];
    for (int k = 0; k < 8; ++k) o[k] = gauss((uint32_t)(r * 256 + lane + 32 * k), s);
    float a = 0.f;
    for (int k = 0; k < 8; ++k) a = fmaxf(a, fabsf(o[k]));
    for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor(a, off, 32));
    const float scale = (a == 0.0f) ? 1.0f : a / 448.0f;
    const __half hs = __float2half(scale);
    const float sf = __half2float(hs);
    unsigned char* row = cache + t * 1032;
    for (int k = 0; k < 8; ++k) {
        const unsigned code = __builtin_amdgcn_cvt_pk_fp8_f32(o[k] / sf, 0.0f, 0u, false) & 0xffu;
        row[h * 256 + lane + 32 * k] = (unsigned char)code;
    }
    if (lane == 0) *(__half*)(row + 1024 + 2 * h) = hs;
}

struct Arg { std::vector<unsigned char> bytes; };
template <class T> static Arg arg(T v) { Arg a; a.bytes.resize(sizeof(T)); memcpy(a.bytes.data(), &v, sizeof(T)); return a; }

struct Kernel {
    hipModule_t mod; hipFunction_t fn;
    Kernel(const char* path, const std::string& sym) {
        CK(hipModuleLoad(&mod, path));
        CK(hipModuleGetFunction(&fn, mod, sym.c_str()));
    }
};

struct Launch {
    unsigned grid[3], block, lds;
    std::vector<Arg> args;
    void run(const Kernel& k, hipStream_t s, const std::vector<Arg>& extra = {}) const {
        std::vector<Arg> all = args;
        all.insert(all.end(), extra.begin(), extra.end());
        std::vector<void*> ptrs;
        for (auto& a : all) ptrs.push_back((void*)a.bytes.data());
        CK(hipModuleLaunchKernel(k.fn, grid[0], grid[1], grid[2], block, 1, 1, lds, s, ptrs.data(), nullptr));
    }
};

struct Buf { void* p; size_t n; };
static Buf alloc(size_t n) { Buf b{nullptr, n}; CK(hipMalloc(&b.p, n)); return b; }
static Buf upload(const void* host, size_t n) { Buf b = alloc(n); CK(hipMemcpy(b.p, host, n, hipMemcpyHostToDevice)); return b; }

static void write_file(const std::string& path, const void* data, size_t n) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f || fwrite(data, 1, n, f) != n) { fprintf(stderr, "write %s failed\n", path.c_str()); exit(1); }
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: see header\n"); return 2; }
    const std::string mode = argv[1];
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    char uuid[40] = {0};
    for (int i = 0; i < 16; ++i) snprintf(uuid + 2 * i, 3, "%02x", (unsigned char)prop.uuid.bytes[i]);
    fprintf(stderr, "device %s pci %04x:%02x:%02x uuid-bytes %s\n", prop.gcnArchName, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, uuid);

    Launch L{};
    std::string sym, outdir, shape;
    std::vector<Buf> outputs;       // compared byte-for-byte
    unsigned waves_per_wg = 0;
    size_t slot_bytes = 0;
    int rounds = 0;
    size_t periods = 0;     // K128 blocks (f2) or KT48 tiles (attn) per wave, upper bound
    if (mode == "f2") {
        const int N = atoi(argv[4]); outdir = argv[5]; rounds = argc > 6 ? atoi(argv[6]) : 8;
        const int M = 17408, K = 5120, total = 2 * M, padded = (total + 255) / 256 * 256, blocks = K / 128;
        sym = "gemm_mq4g256v2_fp8_silu_row_b1";
        shape = "gate_up M0=M1=17408 K=5120 N=" + std::to_string(N);
        const unsigned char wcodes[16] = {0xd0,0xce,0xcc,0xca,0xc8,0xc4,0xc0,0xb8,0x00,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e};
        const unsigned char xcodes[16] = {0x00,0x01,0x08,0x38,0x40,0x44,0x48,0x4e,0xb8,0xc0,0x80,0xd0,0x50,0x57,0xca,0x4a};
        std::vector<unsigned char> wf((size_t)padded * K), x8((size_t)N * K);
        for (size_t i = 0; i < wf.size(); ++i) wf[i] = wcodes[salt(i, 0x3f1) & 15];
        for (size_t i = 0; i < x8.size(); ++i) x8[i] = xcodes[salt(i, 0xd9fe) & 15];
        std::vector<float> rw((size_t)padded * blocks), ew(padded), d(N);
        for (size_t i = 0; i < rw.size(); ++i) rw[i] = ldexpf(1.0f, (int)(salt(i, 0x77) % 3) - 1);
        for (size_t i = 0; i < ew.size(); ++i) ew[i] = ldexpf(1.0f, -12 - (int)(salt(i, 0x78) % 4));
        for (int i = 0; i < N; ++i) d[i] = ldexpf(1.0f, (int)(salt(i, 0x701a) % 11) - 8);
        Buf Wf = upload(wf.data(), wf.size()), Rw = upload(rw.data(), rw.size() * 4), Ew = upload(ew.data(), ew.size() * 4);
        Buf X = upload(x8.data(), x8.size()), Dd = upload(d.data(), d.size() * 4);
        Buf Y0 = alloc(((size_t)M * N + 64) * 4), Y1 = alloc(((size_t)M * N + 64) * 4);
        outputs = {Y0, Y1};
        void* null = nullptr;
        L.args = {arg(Wf.p), arg(Rw.p), arg(Ew.p), arg(X.p), arg(Dd.p), arg(Y0.p), arg(Y1.p), arg(null), arg(null),
                  arg(M), arg(M), arg(0), arg(0), arg(K), arg(N)};
        L.grid[0] = total / 256; L.grid[1] = (N + 127) / 128; L.grid[2] = 1; L.block = 256; L.lds = 19456;
        waves_per_wg = 8;
        periods = blocks;
    } else if (mode == "attn") {
        const int batch = atoi(argv[4]), start = atoi(argv[5]);
        outdir = argv[6]; rounds = argc > 7 ? atoi(argv[7]) : 8;
        sym = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201";
        shape = "fa2 KT48 batch=" + std::to_string(batch) + " start=" + std::to_string(start);
        const int ctx = start + batch;
        const size_t qcode = (size_t)batch * NQ * D;
        Buf q = alloc(qcode + (size_t)batch * NQ * 4), kc = alloc((size_t)ctx * 1032), vc = alloc((size_t)ctx * 1032);
        Buf out = alloc((size_t)batch * NQ * D * 4), pos = alloc((size_t)batch * 4);
        CK(hipMemset(kc.p, 0, kc.n)); CK(hipMemset(vc.p, 0, vc.n));
        std::vector<int> p(batch);
        for (int i = 0; i < batch; ++i) p[i] = start + i;
        CK(hipMemcpy(pos.p, p.data(), batch * 4, hipMemcpyHostToDevice));
        const uint32_t s = 0xA770u;
        const int qrows = batch * NQ;
        make_q8<<<(qrows + 7) / 8, 256>>>((unsigned char*)q.p, (float*)((unsigned char*)q.p + qcode), qrows, s);
        make_kv<<<(ctx * 4 + 7) / 8, 256>>>((unsigned char*)kc.p, ctx, s ^ 0x4u);
        make_kv<<<(ctx * 4 + 7) / 8, 256>>>((unsigned char*)vc.p, ctx, s ^ 0x5u);
        CK(hipGetLastError()); CK(hipDeviceSynchronize());
        outputs = {out};
        void* qs = (unsigned char*)q.p + qcode;
        L.args = {arg(q.p), arg(qs), arg(kc.p), arg(vc.p), arg(out.p), arg(pos.p),
                  arg(NQ), arg(NKV), arg(D), arg(batch), arg(1.0f / sqrtf((float)D))};
        const int run_rows = batch < 512 ? batch : 512;
        L.grid[0] = (unsigned)((run_rows * 6 + 383) / 384); L.grid[1] = 4; L.grid[2] = (unsigned)((batch + 511) / 512);
        L.block = 768; L.lds = 49936;
        waves_per_wg = 24;
        periods = (size_t)(ctx + 47) / 48 + 2;
    } else { fprintf(stderr, "unknown mode\n"); return 2; }

    // Slot: 32 B header, records (8 B, budget PM_RECORDS_PER_PERIOD per block/tile
    // plus 64 for prologue/epilogue), 16 B exit record, 16 B guard that must stay 0xff.
    const char* rpp = getenv("PM_RECORDS_PER_PERIOD");
    const size_t per_period = rpp ? (size_t)atoi(rpp) : 32;
    slot_bytes = (32 + 8 * (64 + per_period * periods) + 16 + 16 + 255) / 256 * 256;
    Kernel base(argv[2], sym), prof(argv[3], sym + "__pm_profile");
    const unsigned wgs = L.grid[0] * L.grid[1] * L.grid[2];
    const size_t waves = (size_t)wgs * waves_per_wg;
    Buf trace = alloc(waves * slot_bytes);
    std::vector<Arg> ext = {arg(trace.p), arg((uint32_t)slot_bytes), arg(waves_per_wg), arg(L.grid[0]), arg(L.grid[1])};
    hipStream_t st; CK(hipStreamCreate(&st));

    // Byte identity: poison, run, snapshot; three poison patterns per kernel.
    auto snapshot = [&](const Kernel& k, bool profiled, unsigned char poison) {
        for (auto& b : outputs) CK(hipMemsetAsync(b.p, poison, b.n, st));
        if (profiled) CK(hipMemsetAsync(trace.p, 0xff, trace.n, st));
        L.run(k, st, profiled ? ext : std::vector<Arg>{});
        CK(hipStreamSynchronize(st));
        std::vector<std::vector<unsigned char>> host;
        for (auto& b : outputs) { host.emplace_back(b.n); CK(hipMemcpy(host.back().data(), b.p, b.n, hipMemcpyDeviceToHost)); }
        return host;
    };
    const unsigned char pats[3] = {0xA5, 0x00, 0xFF};
    std::vector<std::vector<unsigned char>> ref;
    bool identical = true;
    for (int i = 0; i < 3; ++i) {
        auto b = snapshot(base, false, pats[i]);
        auto p = snapshot(prof, true, pats[i]);
        if (i == 0) ref = b;
        for (size_t o = 0; o < b.size(); ++o) {
            // Whole buffers, guards included: unwritten bytes must hold the same poison.
            if (memcmp(b[o].data(), p[o].data(), b[o].size()) != 0) identical = false;
        }
        fprintf(stderr, "IDENTITY poison=%02x %s\n", pats[i], identical ? "equal" : "DIFFERENT");
    }
    size_t finite = 0, nonzero = 0, n0 = ref[0].size() / 4;
    for (size_t i = 0; i < n0; ++i) { float v; memcpy(&v, &ref[0][i * 4], 4); finite += std::isfinite(v); nonzero += v != 0.0f; }

    // Interleaved timing after a 3 s preheat of the baseline.
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto t_ms = [&](const Kernel& k, bool profiled, int n) {
        CK(hipEventRecord(e0, st));
        for (int i = 0; i < n; ++i) L.run(k, st, profiled ? ext : std::vector<Arg>{});
        CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1)); return ms / n;
    };
    const int per = mode == "f2" && L.grid[1] > 16 ? 4 : 20;
    float heat = 0; for (int i = 0; i < 1000 && heat < 3000.f; ++i) heat += t_ms(base, false, per) * per;
    std::vector<float> tb, tp;
    for (int r = 0; r < rounds; ++r) {
        if (r & 1) { tp.push_back(t_ms(prof, true, per)); tb.push_back(t_ms(base, false, per)); }
        else { tb.push_back(t_ms(base, false, per)); tp.push_back(t_ms(prof, true, per)); }
    }
    // Steady-state trace: right after the timed loop, three back-to-back profiled
    // launches; control flow is deterministic, so every launch writes the same
    // record slots and the buffer holds the last one, which is event-timed too.
    CK(hipMemsetAsync(trace.p, 0xff, trace.n, st));
    L.run(prof, st, ext); L.run(prof, st, ext);
    const float trace_ms = t_ms(prof, true, 1);
    std::vector<unsigned char> tr(trace.n);
    CK(hipMemcpy(tr.data(), trace.p, trace.n, hipMemcpyDeviceToHost));
    std::string cmd = "mkdir -p " + outdir;
    if (system(cmd.c_str()) != 0) return 1;
    write_file(outdir + "/trace.bin", tr.data(), tr.size());
    size_t overflow = 0;
    for (size_t w = 0; w < waves; ++w) {
        const unsigned char* g = tr.data() + (w + 1) * slot_bytes - 16;
        for (int i = 0; i < 16; ++i) if (g[i] != 0xff) { ++overflow; break; }
    }
    if (overflow) { fprintf(stderr, "SLOT OVERFLOW in %zu waves: raise PM_RECORDS_PER_PERIOD\n", overflow); return 4; }
    auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    const float mb = med(tb), mp = med(tp);
    std::string meta = "{\n  \"diagnostic\": \"peacemaker profile run; timing records only\",\n";
    meta += "  \"mode\": \"" + mode + "\", \"shape\": \"" + shape + "\", \"symbol\": \"" + sym + "\",\n";
    meta += "  \"base\": \"" + std::string(argv[2]) + "\", \"profiled\": \"" + std::string(argv[3]) + "\",\n";
    meta += "  \"device_uuid_bytes\": \"" + std::string(uuid) + "\",\n";
    char buf[1024];
    snprintf(buf, sizeof buf, "  \"grid\": [%u, %u, %u], \"block\": %u, \"lds\": %u, \"waves_per_wg\": %u, \"waves\": %zu, \"slot_bytes\": %zu,\n"
        "  \"outputs_identical\": %s, \"output_floats\": %zu, \"finite\": %zu, \"nonzero\": %zu,\n"
        "  \"rounds\": %d, \"launches_per_round\": %d, \"base_ms_median\": %.6f, \"profiled_ms_median\": %.6f, \"overhead\": %.6f, \"trace_launch_ms\": %.6f,\n",
        L.grid[0], L.grid[1], L.grid[2], L.block, L.lds, waves_per_wg, waves, slot_bytes,
        identical ? "true" : "false", n0, finite, nonzero, rounds, per, mb, mp, mp / mb - 1.0, trace_ms);
    meta += buf;
    meta += "  \"base_ms\": [";
    for (size_t i = 0; i < tb.size(); ++i) { snprintf(buf, sizeof buf, "%s%.6f", i ? ", " : "", tb[i]); meta += buf; }
    meta += "],\n  \"profiled_ms\": [";
    for (size_t i = 0; i < tp.size(); ++i) { snprintf(buf, sizeof buf, "%s%.6f", i ? ", " : "", tp[i]); meta += buf; }
    meta += "]\n}\n";
    write_file(outdir + "/meta.json", meta.data(), meta.size());
    printf("%s identical=%d finite=%zu/%zu base=%.3f ms profiled=%.3f ms overhead=%+.2f%%\n",
           shape.c_str(), (int)identical, finite, n0, mb, mp, (mp / mb - 1.0) * 100.0);
    return identical ? 0 : 3;
}
