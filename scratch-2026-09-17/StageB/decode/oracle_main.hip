// Per-kernel oracle: scalar fp8 reader + fp8 flash tile, old (software
// decoder, per-element scale) vs new (hw decode, dword loads, folded scale).
// Same seeded inputs; dumps raw outputs for offline comparison.
// Usage: oracle_decode <ksmode: ks1|ksrand> <seed> <seq_len> <out_prefix>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

extern "C" __global__ void attention_fp8_e4m3_kv(
    const float* q, const unsigned char* k_cache, const unsigned char* v_cache,
    float* out, const int* pos_buf, int n_heads, int n_kv_heads, int head_dim,
    int max_seq, float scale_attn);

extern "C" __global__ void attention_flash_fp8_e4m3_tile(
    const float* q, const unsigned char* k_cache, const unsigned char* v_cache,
    float* partials, const int* pos_buf, int n_heads, int n_kv_heads,
    int head_dim, int max_seq, float scale_attn, int tile_size, int window,
    int effective_seq_len);

static void write_bin(const std::string& path, const void* p, size_t n) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); exit(1); }
    fwrite(p, 1, n, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s ks1|ksrand|ksnan seed seq_len out_prefix\n", argv[0]); return 1; }
    const std::string ksmode = argv[1];
    const bool ks1 = ksmode == "ks1" || ksmode == "ksnan";
    const bool force_nan = ksmode == "ksnan";
    const unsigned seed = (unsigned)atoi(argv[2]);
    const int seq_len = atoi(argv[3]);
    const std::string pre = argv[4];

    const int n_heads = 8, n_kv_heads = 2, head_dim = 256;
    const int max_seq = 512, tile_size = 128;
    const size_t row_stride = (size_t)n_kv_heads * (head_dim + 2);
    const size_t codes_len = (size_t)n_kv_heads * head_dim;
    const float scale_attn = 0.0625f;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::normal_distribution<float> qn(0.0f, 0.15f);  // healthy dots, no saturation
    std::normal_distribution<float> kvn(0.0f, 1.0f);
    // Host exact e4m3 values table (finite codes only) for the RNE encoder.
    float e4m3val[256];
    for (int c = 0; c < 256; c++) {
        unsigned u = (unsigned)c, s = (u & 0x80u) << 24;
        unsigned e = (u >> 3) & 0x0fu, m = u & 0x07u;
        if (e == 0) e4m3val[c] = (s ? -1.0f : 1.0f) * (float)m * 1.953125e-3f;
        else if (e == 15) e4m3val[c] = (m == 7) ? 0.0f /*NaN slot*/
            : (s ? -1.0f : 1.0f) * 256.0f * (1.0f + (float)m * 0.125f);
        else {
            uint32_t bits = s | ((e - 7u + 127u) << 23) | (m << 20);
            float f; memcpy(&f, &bits, 4); e4m3val[c] = f;
        }
    }
    auto enc_rne = [&](float x) {
        // Brute-force nearest with ties to even code (writer-faithful).
        int best = 0; double bd = 1e300;
        for (int c = 0; c < 256; c++) {
            if ((c & 0x7f) == 0x7f) continue;  // NaN codes never emitted
            double d = fabs((double)x - (double)e4m3val[c]);
            if (d < bd || (d == bd && (c & 1) == 0)) { bd = d; best = c; }
        }
        return (unsigned char)best;
    };
    struct F16 { static uint16_t from(float f) {
        uint32_t u; memcpy(&u, &f, 4);
        uint32_t s = (u >> 16) & 0x8000u;
        int e = ((u >> 23) & 0xff) - 112;
        uint32_t m = u & 0x7fffffu;
        if (e <= 0) return (uint16_t)s;
        if (e >= 31) return (uint16_t)(s | 0x7bffu);
        uint32_t r = m >> 13;
        if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (r & 1u))) r++;
        if (r == 0x400u) { r = 0; e++; if (e >= 31) return (uint16_t)(s | 0x7bffu); }
        return (uint16_t)(s | ((uint32_t)e << 10) | r);
    } };
    const bool full_bytes = ksmode == "ksfull";

    // Host inputs.
    float* q = new float[n_heads * head_dim];
    for (int i = 0; i < n_heads * head_dim; i++) q[i] = qn(rng);
    unsigned char* kc = new unsigned char[max_seq * row_stride];
    unsigned char* vc = new unsigned char[max_seq * row_stride];
    memset(kc, 0, max_seq * row_stride);
    memset(vc, 0, max_seq * row_stride);
    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < n_kv_heads; h++) {
            float kvf[2][256];
            for (int k = 0; k < 2; k++)
                for (int j = 0; j < head_dim; j++) kvf[k][j] = kvn(rng);
            for (int k = 0; k < 2; k++) {
                unsigned char* row = (k == 0 ? kc : vc) + t * row_stride + h * head_dim;
                if (full_bytes) {
                    // Extreme-coverage arm: uniform bytes, arbitrary scales.
                    for (int j = 0; j < head_dim; j++) {
                        int c = byte_dist(rng);
                        if (c == 0x7f || c == 0xff) c = 0;
                        row[j] = (unsigned char)c;
                    }
                } else {
                    float amax = 0.0f;
                    for (int j = 0; j < head_dim; j++) {
                        float a = fabsf(kvf[k][j]);
                        if (a > amax) amax = a;
                    }
                    float sc = (amax == 0.0f) ? 1.0f : amax / 448.0f;
                    if (ks1) sc = 1.0f;
                    for (int j = 0; j < head_dim; j++)
                        row[j] = enc_rne(kvf[k][j] / sc);
                    uint16_t s16 = F16::from(sc);
                    memcpy((k == 0 ? kc : vc) + t * row_stride + codes_len + h * 2, &s16, 2);
                    continue;
                }
                float sc = ks1 ? 1.0f : expf(2.0f * kvn(rng));
                uint16_t s16 = F16::from(sc);
                memcpy((k == 0 ? kc : vc) + t * row_stride + codes_len + h * 2, &s16, 2);
            }
        }
        if (force_nan) {
            kc[t * row_stride + 0] = 0x7f;  // NaN-poison dim 0 of every K row
            vc[t * row_stride + 1] = 0xff;  // NaN-poison dim 1 of every V row
        }
    }

    // Device buffers.
    float *dq, *dout;
    unsigned char *dkc, *dvc;
    int* dpos;
    hipMalloc(&dq, n_heads * head_dim * 4);
    hipMalloc(&dout, n_heads * head_dim * 4);
    hipMalloc(&dkc, max_seq * row_stride);
    hipMalloc(&dvc, max_seq * row_stride);
    hipMalloc(&dpos, 4);
    hipMemcpy(dq, q, n_heads * head_dim * 4, hipMemcpyHostToDevice);
    hipMemcpy(dkc, kc, max_seq * row_stride, hipMemcpyHostToDevice);
    hipMemcpy(dvc, vc, max_seq * row_stride, hipMemcpyHostToDevice);
    int pos0 = seq_len - 1;
    hipMemcpy(dpos, &pos0, 4, hipMemcpyHostToDevice);

    write_bin(pre + ".inputs_q.bin", q, n_heads * head_dim * 4);
    write_bin(pre + ".inputs_kc.bin", kc, (size_t)seq_len * row_stride);
    write_bin(pre + ".inputs_vc.bin", vc, (size_t)seq_len * row_stride);

    // Scalar reader.
    {
        size_t shared = (2048 + head_dim + 256) * 4;
        hipMemset(dout, 0, n_heads * head_dim * 4);
        hipLaunchKernelGGL(attention_fp8_e4m3_kv, n_heads, 256, shared, 0,
            dq, dkc, dvc, dout, dpos, n_heads, n_kv_heads, head_dim,
            max_seq, scale_attn);
        hipDeviceSynchronize();
        float* out = new float[n_heads * head_dim];
        hipMemcpy(out, dout, n_heads * head_dim * 4, hipMemcpyDeviceToHost);
        write_bin(pre + ".scalar_out.bin", out, n_heads * head_dim * 4);
        delete[] out;
    }

    // Tile kernel partials.
    {
        const int max_tiles = (max_seq + tile_size - 1) / tile_size;
        const int n_tiles = (seq_len + tile_size - 1) / tile_size;
        size_t pn = (size_t)n_heads * max_tiles * (2 + head_dim);
        float *dpar;
        hipMalloc(&dpar, pn * 4);
        hipMemset(dpar, 0, pn * 4);
        size_t shared = (tile_size + head_dim) * 4;
        dim3 grid(n_heads, n_tiles, 1);
        hipLaunchKernelGGL(attention_flash_fp8_e4m3_tile, grid, 32, shared, 0,
            dq, dkc, dvc, dpar, dpos, n_heads, n_kv_heads, head_dim,
            max_seq, scale_attn, tile_size, 0, 0);
        hipDeviceSynchronize();
        float* par = new float[pn];
        hipMemcpy(par, dpar, pn * 4, hipMemcpyDeviceToHost);
        write_bin(pre + ".tile_partials.bin", par, pn * 4);
        delete[] par;
        hipFree(dpar);
    }

    hipFree(dq); hipFree(dout); hipFree(dkc); hipFree(dvc); hipFree(dpos);
    delete[] q; delete[] kc; delete[] vc;
    printf("ORACLE_DONE %s seq=%d\n", pre.c_str(), seq_len);
    return 0;
}
