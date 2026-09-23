#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
// branchless candidate (host copy of the device logic)
static float cand(unsigned char b) {
    const unsigned int u = b;
    const unsigned int s = (u & 0x80u) << 24;
    const unsigned int e = (u >> 3) & 0x0fu;
    const unsigned int m = u & 0x07u;
    float f_sub = (float)m * 1.953125e-3f;
    uint32_t nb; float f_norm;
    nb = ((e + 120u) << 23) | (m << 20); memcpy(&f_norm, &nb, 4);
    float f_e15 = 256.0f * (1.0f + (float)m * 0.125f);
    float mag = (e == 0u) ? f_sub : f_norm;
    mag = (e == 15u && m != 7u) ? f_e15 : mag;
    uint32_t nan = 0x7fc00000u; float fnan; memcpy(&fnan, &nan, 4);
    mag = (e == 15u && m == 7u) ? fnan : mag;
    uint32_t mb; memcpy(&mb, &mag, 4); mb |= s; memcpy(&mag, &mb, 4);
    return mag;
}
// independent reference: OCP E4M3FN via double
static double ref(int b) {
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    double v;
    if (e == 0) v = m * 0.001953125;
    else if (e == 15) v = (m == 7) ? NAN : 256.0 * (1.0 + m * 0.125);
    else v = pow(2.0, e - 7) * (1.0 + m / 8.0);
    return s ? -v : v;
}
int main() {
    int bad = 0;
    for (int b = 0; b < 256; b++) {
        float c = cand((unsigned char)b);
        double r = ref(b);
        uint32_t cb; memcpy(&cb, &c, 4);
        bool cok = std::isnan(c);
        if (std::isnan(r)) { if (!cok || ((cb & 0x7fffffffu) != 0x7fc00000u)) { printf("NaN mismatch %02x -> %08x\n", b, cb); bad++; } continue; }
        float rf = (float)r; // RNE double->float; E4M3 values all exactly representable? subnormals m*2^-9 exact in f32 yes; normals exact; e15 exact
        uint32_t rb; memcpy(&rb, &rf, 4);
        if (cb != rb) { printf("mismatch %02x: cand %08x ref %08x\n", b, cb, rb); bad++; }
    }
    printf(bad ? "FAIL %d\n" : "ALL 256 EXACT\n", bad);
    return bad != 0;
}
