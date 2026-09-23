// Vfill gate-4 oracle (throwaway, untracked): old-vs-new layout bitwise proof.
// K via token-major writer (unchanged); V_old via the same writer (== old V
// placement); V_new via kv_cache_write_fp8_e4m3_v{,_batched}. Compares:
//  (A) host permutation check of every code/scale byte,
//  (B..G) device old-reader-on-old-layout vs new-reader-on-new-layout, bitwise.
// Compile: hipcc --offload-arch=gfx1201 -O2 -std=c++17 -D__gfx1201__
//   -Wno-macro-redefined -I. oracle_vfill.hip -o oracle_vfill
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

#define HIPFIRE_KV_FP8_E4M3 1
#include "../../../kernels/src/kv_cache_write_q8_0.hip"
#include "../../../kernels/src/kv_cache_write_q8_0_batched.hip"
#include "o_kv.hip"
#include "../../../kernels/src/attention_q8_0_kv.hip"
#include "o_kvb.hip"
#include "../../../kernels/src/attention_q8_0_kv_batched.hip"
#include "o_ft.hip"
#include "../../../kernels/src/attention_flash_q8_0_tile.hip"
#include "o_ftb.hip"
#include "../../../kernels/src/attention_flash_q8_0_tile_batched.hip"
#define HIPFIRE_FA2_KMODE 8
#include "o_fa2q0.hip"
#undef HIPFIRE_FA2_KMODE
#define HIPFIRE_FA2_KMODE 8
#define HIPFIRE_FA2_FP8 1
#include "o_fa2n.hip"
#undef HIPFIRE_FA2_FP8
#undef HIPFIRE_FA2_KMODE
#define HIPFIRE_FA2_KMODE 8
#include "../../../kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip"
#undef HIPFIRE_FA2_KMODE
#define HIPFIRE_FA2_KMODE 8
#define HIPFIRE_FA2_FP8 1
#include "n_fa2n.hip"
#undef HIPFIRE_FA2_FP8
#undef HIPFIRE_FA2_KMODE
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP fail %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    return 1; } } while (0)

static uint64_t lcg_state = 0x12345678u;
static uint64_t lcg_next() {
    lcg_state += 0x9e3779b97f4a7c15ull;
    uint64_t z = lcg_state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static float randn() {
    double u1 = (double)lcg_next() / (double)UINT64_MAX;
    if (u1 < 1e-12) u1 = 1e-12;
    double u2 = (double)lcg_next() / (double)UINT64_MAX;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793 * u2));
}

static int fails = 0;
static void check_bytes(const char* name, const void* a, const void* b, size_t n) {
    if (memcmp(a, b, n) != 0) {
        const uint32_t* x = (const uint32_t*)a;
        const uint32_t* y = (const uint32_t*)b;
        size_t nm = 0;
        for (size_t i = 0; i < n / 4; i++) if (x[i] != y[i]) nm++;
        printf("MISMATCH %s: %zu/%zu words differ\n", name, nm, n / 4);
        fails++;
    } else {
        printf("match    %s (%zu bytes)\n", name, n);
    }
}

int main() {
    CK(hipSetDevice(0));
    const int NH = 24, NKV = 4, HD = 256;
    const size_t ROW = (size_t)NKV * (HD + 2); // 1032
    const size_t CODES = (size_t)NKV * HD;     // 1024
    const int CAP = 128;
    const float scale_attn = 1.0f / sqrtf((float)HD);

    // ---- caches ----
    unsigned char *k_cache, *v_old, *v_new;
    CK(hipMalloc(&k_cache, CAP * ROW));
    CK(hipMalloc(&v_old, CAP * ROW));
    CK(hipMalloc(&v_new, CAP * ROW));
    CK(hipMemset(k_cache, 0x5a, CAP * ROW));
    CK(hipMemset(v_old, 0x5a, CAP * ROW));
    CK(hipMemset(v_new, 0x5a, CAP * ROW));

    // ---- data: decode-scale (B=64) ----
    const int BD = 64;
    std::vector<float> qd(BD * NH * HD), kd(BD * NKV * HD), vd(BD * NKV * HD);
    for (auto& x : qd) x = randn();
    for (auto& x : kd) x = randn() * 1.5f;
    for (auto& x : vd) x = randn() * 0.7f;
    std::vector<int32_t> posd(BD);
    for (int i = 0; i < BD; i++) posd[i] = i;
    float *d_q, *d_k, *d_v;
    int32_t* d_posd;
    int32_t* d_pos1;
    CK(hipMalloc(&d_q, qd.size() * 4));
    CK(hipMalloc(&d_k, kd.size() * 4));
    CK(hipMalloc(&d_v, vd.size() * 4));
    CK(hipMalloc(&d_posd, BD * 4));
    CK(hipMalloc(&d_pos1, 4));
    CK(hipMemcpy(d_q, qd.data(), qd.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_k, kd.data(), kd.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_v, vd.data(), vd.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_posd, posd.data(), BD * 4, hipMemcpyHostToDevice));

    // write K (token-major) + V_old (token-major, == old V) + V_new (grouped)
    {
        dim3 grid(NKV, BD, 1), block(32, 1, 1);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_batched, grid, block, 0, 0,
            k_cache, d_k, d_posd, NKV, HD, BD);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_batched, grid, block, 0, 0,
            v_old, d_v, d_posd, NKV, HD, BD);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_v_batched, grid, block, 0, 0,
            v_new, d_v, d_posd, NKV, HD, BD);
        CK(hipDeviceSynchronize());
    }
    // single-writer cross-check at pos 5 into scratch rows
    unsigned char *v_scratch;
    CK(hipMalloc(&v_scratch, CAP * ROW));
    CK(hipMemcpy(v_scratch, v_new, CAP * ROW, hipMemcpyDeviceToDevice));
    {
        int32_t p5 = 5;
        int32_t* d_p5;
        CK(hipMalloc(&d_p5, 4));
        CK(hipMemcpy(d_p5, &p5, 4, hipMemcpyHostToDevice));
        float* d_v5;
        CK(hipMalloc(&d_v5, (size_t)NKV * HD * 4));
        CK(hipMemcpy(d_v5, vd.data() + (size_t)5 * NKV * HD, (size_t)NKV * HD * 4,
            hipMemcpyHostToDevice));
        dim3 grid(NKV, 1, 1), block(32, 1, 1);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_v, grid, block, 0, 0,
            v_scratch, d_v5, d_p5, NKV, HD);
        CK(hipDeviceSynchronize());
        CK(hipFree(d_p5));
        CK(hipFree(d_v5));
    }

    // ---- (A) host permutation check ----
    {
        std::vector<unsigned char> ho(CAP * ROW), hn(CAP * ROW), hs(CAP * ROW);
        CK(hipMemcpy(ho.data(), v_old, CAP * ROW, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hn.data(), v_new, CAP * ROW, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hs.data(), v_scratch, CAP * ROW, hipMemcpyDeviceToHost));
        int bad = 0;
        for (int t = 0; t < BD; t++)
            for (int h = 0; h < NKV; h++) {
                for (int d = 0; d < HD; d++) {
                    unsigned char a = ho[(size_t)t * ROW + (size_t)h * HD + d];
                    unsigned char b = hn[(size_t)(t >> 3) * ROW * 8
                        + (size_t)h * HD * 8 + (size_t)d * 8 + (size_t)(t & 7)];
                    if (a != b && bad++ < 5)
                        printf("CODE mismatch t=%d h=%d d=%d old=%u new=%u\n",
                            t, h, d, a, b);
                }
                for (int i = 0; i < 2; i++) {
                    unsigned char a = ho[(size_t)t * ROW + CODES + (size_t)h * 2 + i];
                    unsigned char b = hn[(size_t)(t >> 3) * ROW * 8 + CODES * 8
                        + (size_t)(t & 7) * (size_t)NKV * 2 + (size_t)h * 2 + i];
                    if (a != b && bad++ < 5)
                        printf("SCALE mismatch t=%d h=%d i=%d\n", t, h, i);
                }
            }
        if (bad) { printf("MISMATCH placement: %d bytes\n", bad); fails++; }
        else printf("match    placement+scales (80tok x 4hd x 256d)\n");
        check_bytes("single-vs-batched V writer", hn.data(), hs.data(), CAP * ROW);
    }

    // ---- (B) single decode, seq 19 and 64 ----
    for (int seq : {1, 19, 64}) {
        int32_t p = seq - 1;
        CK(hipMemcpy(d_pos1, &p, 4, hipMemcpyHostToDevice));
        size_t on = (size_t)NH * HD;
        float *o_old, *o_new;
        CK(hipMalloc(&o_old, on * 4));
        CK(hipMalloc(&o_new, on * 4));
        dim3 grid(NH, 1, 1), block(256, 1, 1);
        size_t lds = (2048 + 256 + 256) * 4;
        hipLaunchKernelGGL(attention_fp8_e4m3_kv_old, grid, block, lds, 0,
            d_q, k_cache, v_old, o_old, d_pos1, NH, NKV, HD, CAP, scale_attn);
        hipLaunchKernelGGL(attention_fp8_e4m3_kv, grid, block, lds, 0,
            d_q, k_cache, v_new, o_new, d_pos1, NH, NKV, HD, CAP, scale_attn);
        CK(hipDeviceSynchronize());
        std::vector<float> a(on), b(on);
        CK(hipMemcpy(a.data(), o_old, on * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), o_new, on * 4, hipMemcpyDeviceToHost));
        char nm[64];
        snprintf(nm, sizeof nm, "single-decode seq=%d", seq);
        check_bytes(nm, a.data(), b.data(), on * 4);
        if (seq == 1) {
            printf("  seq1 dims0-7 old:");
            for (int dd = 0; dd < 8; dd++) printf(" %a", a[(size_t)0 * HD + dd]);
            printf("\n  seq1 dims0-7 new:");
            for (int dd = 0; dd < 8; dd++) printf(" %a", b[(size_t)0 * HD + dd]);
            printf("\n");
        }
        CK(hipFree(o_old));
        CK(hipFree(o_new));
    }

    // ---- (C) batched decode, B=20 and 64 ----
    for (int B : {20, 64}) {
        float *d_qb;
        int32_t* d_pb;
        CK(hipMalloc(&d_qb, (size_t)B * NH * HD * 4));
        CK(hipMalloc(&d_pb, (size_t)B * 4));
        CK(hipMemcpy(d_qb, qd.data(), (size_t)B * NH * HD * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_pb, posd.data(), (size_t)B * 4, hipMemcpyHostToDevice));
        size_t on = (size_t)B * NH * HD;
        float *o_old, *o_new;
        CK(hipMalloc(&o_old, on * 4));
        CK(hipMalloc(&o_new, on * 4));
        dim3 grid(NH, B, 1), block(256, 1, 1);
        size_t lds = ((size_t)CAP + 256 + 256) * 4;
        int bs0 = 0, bc0 = 0;
        hipLaunchKernelGGL(attention_fp8_e4m3_kv_batched_old, grid, block, lds, 0,
            d_qb, k_cache, v_old, o_old, d_pb, nullptr, NH, NKV, HD, CAP,
            scale_attn, bs0, bc0, nullptr, nullptr);
        hipLaunchKernelGGL(attention_fp8_e4m3_kv_batched, grid, block, lds, 0,
            d_qb, k_cache, v_new, o_new, d_pb, nullptr, NH, NKV, HD, CAP,
            scale_attn, bs0, bc0, nullptr, nullptr);
        CK(hipDeviceSynchronize());
        std::vector<float> a(on), b(on);
        CK(hipMemcpy(a.data(), o_old, on * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), o_new, on * 4, hipMemcpyDeviceToHost));
        char nm[64];
        snprintf(nm, sizeof nm, "batched-decode B=%d", B);
        check_bytes(nm, a.data(), b.data(), on * 4);
        CK(hipFree(d_qb));
        CK(hipFree(d_pb));
        CK(hipFree(o_old));
        CK(hipFree(o_new));
    }

    // ---- (D) flash tile partials, seq 20 and 64 ----
    for (int seq : {20, 64}) {
        int32_t p = seq - 1;
        CK(hipMemcpy(d_pos1, &p, 4, hipMemcpyHostToDevice));
        const int TS = 64, MT = (CAP + TS - 1) / TS;
        size_t pn = (size_t)NH * MT * (2 + HD);
        float *p_old, *p_new;
        CK(hipMalloc(&p_old, pn * 4));
        CK(hipMalloc(&p_new, pn * 4));
        CK(hipMemset(p_old, 0xaa, pn * 4));
        CK(hipMemset(p_new, 0xaa, pn * 4));
        dim3 grid(NH, MT, 1), block(32, 1, 1);
        size_t lds = (TS + HD) * 4;
        int wn = 0, es = 0;
        hipLaunchKernelGGL(attention_flash_fp8_e4m3_tile_old, grid, block, lds, 0,
            d_q, k_cache, v_old, p_old, d_pos1, NH, NKV, HD, CAP, scale_attn,
            TS, wn, es);
        hipLaunchKernelGGL(attention_flash_fp8_e4m3_tile, grid, block, lds, 0,
            d_q, k_cache, v_new, p_new, d_pos1, NH, NKV, HD, CAP, scale_attn,
            TS, wn, es);
        CK(hipDeviceSynchronize());
        std::vector<float> a(pn), b(pn);
        CK(hipMemcpy(a.data(), p_old, pn * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), p_new, pn * 4, hipMemcpyDeviceToHost));
        // only tiles with tile_start < seq are written; compare those
        int bad = 0;
        for (int h = 0; h < NH; h++)
            for (int t = 0; t < MT; t++) {
                if (t * TS >= seq) continue;
                size_t off = ((size_t)h * MT + t) * (2 + HD);
                if (memcmp(a.data() + off, b.data() + off, (2 + HD) * 4)) bad++;
            }
        if (bad) { printf("MISMATCH flash-tile seq=%d: %d tiles\n", seq, bad); fails++; }
        else printf("match    flash-tile seq=%d partials\n", seq);
        CK(hipFree(p_old));
        CK(hipFree(p_new));
    }

    // ---- (E) flash-batched tile partials, B=20 and 64 ----
    for (int B : {20, 64}) {
        float *d_qb;
        int32_t* d_pb;
        CK(hipMalloc(&d_qb, (size_t)B * NH * HD * 4));
        CK(hipMalloc(&d_pb, (size_t)B * 4));
        CK(hipMemcpy(d_qb, qd.data(), (size_t)B * NH * HD * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_pb, posd.data(), (size_t)B * 4, hipMemcpyHostToDevice));
        const int TS = 64, MT = (CAP + TS - 1) / TS;
        size_t pn = (size_t)B * NH * MT * (2 + HD);
        float *p_old, *p_new;
        CK(hipMalloc(&p_old, pn * 4));
        CK(hipMalloc(&p_new, pn * 4));
        CK(hipMemset(p_old, 0xaa, pn * 4));
        CK(hipMemset(p_new, 0xaa, pn * 4));
        dim3 grid(NH, MT, B), block(32, 1, 1);
        size_t lds = TS * 4;
        int vm = 0, wn = 0, bo = 0, bs0 = 0, bc0 = 0;
        hipLaunchKernelGGL(attention_flash_fp8_e4m3_tile_batched_old, grid, block, lds, 0,
            d_qb, k_cache, v_old, p_old, d_pb, d_qb, d_qb, nullptr,
            NH, NKV, HD, CAP, scale_attn, TS, MT, bo, bs0, bc0, vm, wn,
            nullptr, nullptr);
        hipLaunchKernelGGL(attention_flash_fp8_e4m3_tile_batched, grid, block, lds, 0,
            d_qb, k_cache, v_new, p_new, d_pb, d_qb, d_qb, nullptr,
            NH, NKV, HD, CAP, scale_attn, TS, MT, bo, bs0, bc0, vm, wn,
            nullptr, nullptr);
        CK(hipDeviceSynchronize());
        std::vector<float> a(pn), b(pn);
        CK(hipMemcpy(a.data(), p_old, pn * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), p_new, pn * 4, hipMemcpyDeviceToHost));
        int bad = 0;
        for (int r = 0; r < B; r++) {
            int seq = r + 1;
            for (int h = 0; h < NH; h++)
                for (int t = 0; t < MT; t++) {
                    if (t * TS >= seq) continue;
                    size_t off = (((size_t)r * NH + h) * MT + t) * (2 + HD);
                    if (memcmp(a.data() + off, b.data() + off, (2 + HD) * 4)) bad++;
                }
        }
        if (bad) { printf("MISMATCH flash-batched B=%d: %d tiles\n", B, bad); fails++; }
        else printf("match    flash-batched B=%d partials\n", B);
        CK(hipFree(d_qb));
        CK(hipFree(d_pb));
        CK(hipFree(p_old));
        CK(hipFree(p_new));
    }

    // ---- FA2-scale data (B=80) ----
    const int BF = 80;
    std::vector<float> qf(BF * NH * HD), kf(BF * NKV * HD), vf(BF * NKV * HD);
    for (auto& x : qf) x = randn();
    for (auto& x : kf) x = randn() * 1.5f;
    for (auto& x : vf) x = randn() * 0.7f;
    std::vector<int32_t> posf(BF);
    for (int i = 0; i < BF; i++) posf[i] = i;
    float *f_q, *f_k, *f_v;
    int32_t* f_pos;
    unsigned char *f_kc, *f_vo, *f_vn;
    CK(hipMalloc(&f_q, qf.size() * 4));
    CK(hipMalloc(&f_k, kf.size() * 4));
    CK(hipMalloc(&f_v, vf.size() * 4));
    CK(hipMalloc(&f_pos, BF * 4));
    CK(hipMalloc(&f_kc, CAP * ROW));
    CK(hipMalloc(&f_vo, CAP * ROW));
    CK(hipMalloc(&f_vn, CAP * ROW));
    CK(hipMemcpy(f_q, qf.data(), qf.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(f_k, kf.data(), kf.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(f_v, vf.data(), vf.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(f_pos, posf.data(), BF * 4, hipMemcpyHostToDevice));
    {
        dim3 grid(NKV, BF, 1), block(32, 1, 1);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_batched, grid, block, 0, 0,
            f_kc, f_k, f_pos, NKV, HD, BF);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_batched, grid, block, 0, 0,
            f_vo, f_v, f_pos, NKV, HD, BF);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_v_batched, grid, block, 0, 0,
            f_vn, f_v, f_pos, NKV, HD, BF);
        CK(hipDeviceSynchronize());
    }
    for (int B : {79, 80}) {
        size_t q16ne = (size_t)B * NH * HD;
        _Float16 *q16o, *q16n;
        float *o_old, *o_new;
        CK(hipMalloc(&q16o, q16ne * 2));
        CK(hipMalloc(&q16n, q16ne * 2));
        CK(hipMalloc(&o_old, q16ne * 4));
        CK(hipMalloc(&o_new, q16ne * 4));
        {
            dim3 grid((B * NH + 3) / 4, 1, 1), block(128, 1, 1);
            int zero = 0;
            hipLaunchKernelGGL(attention_fp8_e4m3_oldq0_q_preconvert_f16_gfx1201,
                grid, block, 0, 0, f_q, q16o, nullptr, nullptr, B, zero);
            hipLaunchKernelGGL(attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201,
                grid, block, 0, 0, f_q, q16n, nullptr, nullptr, B, zero);
            CK(hipDeviceSynchronize());
        }
        {
            dim3 grid((B + 7) / 8, 4, 1), block(128, 1, 1);
            size_t lds = 65536;
            hipLaunchKernelGGL(attention_fp8_e4m3_oldq0_gqa_f16_gfx1201, grid, block,
                lds, 0, q16o, f_kc, f_vo, o_old, f_pos, NH, NKV, HD, B, scale_attn);
            hipLaunchKernelGGL(attention_fp8_e4m3_fa2_gqa_f16_gfx1201, grid, block,
                lds, 0, q16n, f_kc, f_vn, o_new, f_pos, NH, NKV, HD, B, scale_attn);
            CK(hipDeviceSynchronize());
        }
        size_t on = (size_t)B * NH * HD;
        std::vector<float> a(on), b(on);
        CK(hipMemcpy(a.data(), o_old, on * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), o_new, on * 4, hipMemcpyDeviceToHost));
        char nm[64];
        snprintf(nm, sizeof nm, "fa2-q0-f16 B=%d", B);
        check_bytes(nm, a.data(), b.data(), on * 4);
        {
            int shown = 0, badrows = 0;
            for (int r = 0; r < B; r++) {
                int rowbad = 0;
                for (int h = 0; h < NH; h++)
                    for (int d = 0; d < HD; d++) {
                        float x = a[((size_t)r * NH + h) * HD + d];
                        float y = b[((size_t)r * NH + h) * HD + d];
                        if (x != y) {
                            if (rowbad == 0 && shown < 6)
                                printf("  first diff r=%d h=%d d=%d old=%a new=%a\n",
                                    r, h, d, x, y);
                            rowbad++;
                        }
                    }
                if (rowbad) { badrows++; shown++; }
            }
            printf("  q0 B=%d badrows=%d/%d\n", B, badrows, B);
        }
        CK(hipFree(q16o));
        CK(hipFree(q16n));
        CK(hipFree(o_old));
        CK(hipFree(o_new));
    }

    // ---- (G) FA2 route-N, B=79 and 80 ----
    for (int B : {79, 80}) {
        size_t codesn = (size_t)B * NH * HD;
        size_t sqoff = codesn;
        unsigned char *c_o, *c_n;
        float *o_old, *o_new;
        CK(hipMalloc(&c_o, codesn + (size_t)B * NH * 4));
        CK(hipMalloc(&c_n, codesn + (size_t)B * NH * 4));
        CK(hipMalloc(&o_old, codesn * 4));
        CK(hipMalloc(&o_new, codesn * 4));
        {
            dim3 grid((B * NH + 3) / 4, 1, 1), block(128, 1, 1);
            hipLaunchKernelGGL(attention_fp8_e4m3_oldn_q_preconvert_fp8_gfx1201,
                grid, block, 0, 0, f_q, c_o, (float*)(c_o + sqoff), B);
            hipLaunchKernelGGL(attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201,
                grid, block, 0, 0, f_q, c_n, (float*)(c_n + sqoff), B);
            CK(hipDeviceSynchronize());
        }
        {
            dim3 grid((B + 7) / 8, 4, 1), block(128, 1, 1);
            size_t lds = 32768;
            hipLaunchKernelGGL(attention_fp8_e4m3_oldn_gqa_gfx1201, grid, block,
                lds, 0, c_o, f_kc, f_vo, o_old, f_pos, NH, NKV, HD, B, scale_attn);
            hipLaunchKernelGGL(attention_fp8_e4m3_fa2_gqa_gfx1201, grid, block,
                lds, 0, c_n, f_kc, f_vn, o_new, f_pos, NH, NKV, HD, B, scale_attn);
            CK(hipDeviceSynchronize());
        }
        size_t on = (size_t)B * NH * HD;
        std::vector<float> a(on), b(on);
        CK(hipMemcpy(a.data(), o_old, on * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), o_new, on * 4, hipMemcpyDeviceToHost));
        char nm[64];
        snprintf(nm, sizeof nm, "fa2-routeN B=%d", B);
        check_bytes(nm, a.data(), b.data(), on * 4);
        CK(hipFree(c_o));
        CK(hipFree(c_n));
        CK(hipFree(o_old));
        CK(hipFree(o_new));
    }

    if (fails) printf("ORACLE FAIL: %d checks\n", fails);
    else printf("ORACLE PASS: all bitwise identical\n");
    return fails != 0;
}
