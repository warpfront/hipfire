// Runtime probe: __builtin_amdgcn_cvt_pk_f32_fp8 exactness vs software decoder.
// Decodes all 256 codes through (dword, sel) for sel in {0,1} and compares
// against the host exact OCP E4M3FN decoder (same math as the kernel's).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

typedef float f32x2 __attribute__((ext_vector_type(2)));

__global__ void probe_decode(const unsigned int* in, float* out) {
    unsigned int w = in[0];
    f32x2 lo = __builtin_amdgcn_cvt_pk_f32_fp8(w, 0);
    f32x2 hi = __builtin_amdgcn_cvt_pk_f32_fp8(w, 1);
    out[0] = lo[0]; out[1] = lo[1]; out[2] = hi[0]; out[3] = hi[1];
}

static float sw_decode(unsigned b) {
    unsigned u = b;
    unsigned s = (u & 0x80u) << 24;
    unsigned e = (u >> 3) & 0x0fu;
    unsigned m = u & 0x07u;
    float f;
    if (e == 0u) {
        f = (float)m * 1.953125e-3f;
        return s ? -f : f;
    }
    if (e == 15u) {
        if (m == 7u) return __builtin_bit_cast(float, 0x7fc00000u);
        f = 256.0f * (1.0f + (float)m * 0.125f);
        return s ? -f : f;
    }
    unsigned bits = s | ((e - 7u + 127u) << 23) | (m << 20);
    return __builtin_bit_cast(float, bits);
}

int main() {
    int mism_sel[2] = {0, 0};
    int nan_mism = 0;
    // For each byte position pattern: pack 4 distinct codes per dword.
    for (int base = 0; base < 256; base += 4) {
        unsigned w = (unsigned)(base) | ((unsigned)(base + 1) << 8) |
                     ((unsigned)(base + 2) << 16) | ((unsigned)(base + 3) << 24);
        unsigned *din; float *dout;
        hipMalloc(&din, 4); hipMalloc(&dout, 16);
        hipMemcpy(din, &w, 4, hipMemcpyHostToDevice);
        probe_decode<<<1, 1>>>(din, dout);
        hipDeviceSynchronize();
        float h[4];
        hipMemcpy(h, dout, 16, hipMemcpyDeviceToHost);
        float ref[4] = {sw_decode(base), sw_decode(base + 1), sw_decode(base + 2), sw_decode(base + 3)};
        for (int i = 0; i < 4; i++) {
            bool ref_nan = ref[i] != ref[i];
            bool got_nan = h[i] != h[i];
            if (ref_nan || got_nan) {
                if (ref_nan != got_nan) {
                    nan_mism++;
                    printf("NAN-MISMATCH code %d (byte %d): ref=%f got=%f\n", base + i, i, ref[i], h[i]);
                }
                continue;
            }
            if (memcmp(&h[i], &ref[i], 4) != 0) {
                int sel = (i < 2) ? 0 : 1;
                mism_sel[sel]++;
                if (mism_sel[0] + mism_sel[1] < 10)
                    printf("MISMATCH code %d (sel %d lane %d): ref=%a got=%a\n",
                           base + i, sel, i & 1, ref[i], h[i]);
            }
        }
        hipFree(din); hipFree(dout);
    }
    printf("sel0 mismatches: %d, sel1 mismatches: %d, nan mismatches: %d\n",
           mism_sel[0], mism_sel[1], nan_mism);
    printf((mism_sel[0] == 0 && mism_sel[1] == 0 && nan_mism == 0) ? "PROBE_PASS\n" : "PROBE_FAIL\n");
    return 0;
}
