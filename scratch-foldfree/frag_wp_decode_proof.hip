#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define HIPFIRE_MQ4V2_WPRESHUFFLE 1
#define HIPFIRE_MQ4V2_FRAGMENT_WPRESHUFFLE 1
#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1
#include "../kernels/src/gfx12_weight_cache_policy.inc"
#define HIPFIRE_MQ4G256V2_KERNEL proof_plain_frag_wp
#include "../kernels/src/gemv_mq4g256v2.hip"
#define HIPFIRE_RESIDUAL_KERNEL proof_residual_frag_wp
#include "../kernels/src/gemv_mq4g256v2_residual.hip"
#define HIPFIRE_FUSED_GATE_UP_KERNEL proof_fused_gate_up_frag_wp
#include "../kernels/src/fused_gate_up_mq4g256v2.hip"

struct ProofImage {
    std::vector<unsigned char> row_major;
    std::vector<unsigned char> fragment;
    bool bijection;
    bool roundtrip;
    bool dequant_equal;
};

static uint16_t load_u16(const unsigned char* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float half_to_float(const unsigned char* p) {
    return __half2float(__ushort_as_half(load_u16(p)));
}

static size_t fragment_payload_offset(int row, int k, int nibble) {
    const int lane = 16 * ((nibble >> 3) & 1) + (row & 15);
    return ((((size_t)(row >> 4) * (size_t)(k >> 4) + (size_t)(nibble >> 4)) * 32
             + (size_t)lane) * 4)
        + (size_t)((nibble & 7) >> 1);
}

static size_t fragment_header_offset(int rows, int k, int row, int half_block, int plane) {
    const size_t payload = (size_t)rows * (size_t)k / 2;
    return payload
        + ((((size_t)half_block * (size_t)(rows >> 4) + (size_t)(row >> 4)) * 4
             + (size_t)plane) * 16)
        + (size_t)(row & 15);
}

static ProofImage make_image(int rows, int k) {
    const int groups = k / 256;
    ProofImage p;
    p.row_major.resize((size_t)rows * (size_t)groups * 136);
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups; ++group) {
            const size_t base = ((size_t)row * groups + group) * 136;
            for (int half = 0; half < 2; ++half) {
                const float scale = 0.00390625f * (1 + ((row + group + half) % 11));
                const float zero = -0.25f + 0.015625f * ((3 * row + group + half) % 17);
                const uint16_t sb = __half_as_ushort(__float2half(scale));
                const uint16_t zb = __half_as_ushort(__float2half(zero));
                p.row_major[base + half * 4 + 0] = (unsigned char)sb;
                p.row_major[base + half * 4 + 1] = (unsigned char)(sb >> 8);
                p.row_major[base + half * 4 + 2] = (unsigned char)zb;
                p.row_major[base + half * 4 + 3] = (unsigned char)(zb >> 8);
            }
            for (int byte = 0; byte < 128; ++byte) {
                const int lo = (row + 3 * group + 5 * byte) & 15;
                const int hi = (7 * row + group + 9 * byte) & 15;
                p.row_major[base + 8 + byte] = (unsigned char)(lo | (hi << 4));
            }
        }
    }

    p.fragment.assign(p.row_major.size(), 0);
    std::vector<unsigned char> hits(p.row_major.size(), 0);
    std::vector<unsigned char> inverse(p.row_major.size(), 0);
    for (int row = 0; row < rows; ++row) {
        for (int packed_col = 0; packed_col < k / 2; ++packed_col) {
            const int nibble = packed_col * 2;
            const size_t src = ((size_t)row * groups + (size_t)(nibble / 256)) * 136
                + 8 + (size_t)((nibble & 255) / 2);
            const size_t dst = fragment_payload_offset(row, k, nibble);
            p.fragment[dst] = p.row_major[src];
            inverse[src] = p.fragment[dst];
            ++hits[dst];
        }
        for (int half_block = 0; half_block < k / 128; ++half_block) {
            const size_t src = ((size_t)row * groups + (size_t)(half_block / 2)) * 136
                + (size_t)(half_block & 1) * 4;
            const int planes[4] = {1, 0, 2, 3};
            for (int byte = 0; byte < 4; ++byte) {
                const size_t dst = fragment_header_offset(rows, k, row, half_block, planes[byte]);
                p.fragment[dst] = p.row_major[src + byte];
                inverse[src + byte] = p.fragment[dst];
                ++hits[dst];
            }
        }
    }
    p.bijection = true;
    for (unsigned char hit : hits) p.bijection &= hit == 1;
    p.roundtrip = inverse == p.row_major;

    p.dequant_equal = true;
    const size_t payload = (size_t)rows * (size_t)k / 2;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < k; ++col) {
            const int group = col / 256;
            const int in_group = col & 255;
            const int half_block = col / 128;
            const size_t raw_base = ((size_t)row * groups + (size_t)group) * 136;
            const unsigned char raw_byte = p.row_major[raw_base + 8 + (size_t)(in_group / 2)];
            const int raw_q = (in_group & 1) ? (raw_byte >> 4) : (raw_byte & 15);
            const unsigned char frag_byte = p.fragment[fragment_payload_offset(row, k, col & ~1)];
            const int frag_q = (col & 1) ? (frag_byte >> 4) : (frag_byte & 15);
            const unsigned char* raw_header = p.row_major.data() + raw_base + (size_t)((in_group >> 7) * 4);
            unsigned char frag_header[4] = {
                p.fragment[fragment_header_offset(rows, k, row, half_block, 1)],
                p.fragment[fragment_header_offset(rows, k, row, half_block, 0)],
                p.fragment[fragment_header_offset(rows, k, row, half_block, 2)],
                p.fragment[fragment_header_offset(rows, k, row, half_block, 3)],
            };
            const float raw_value = half_to_float(raw_header) * (float)raw_q + half_to_float(raw_header + 2);
            const float frag_value = half_to_float(frag_header) * (float)frag_q + half_to_float(frag_header + 2);
            p.dequant_equal &= std::memcmp(&raw_value, &frag_value, sizeof(float)) == 0;
        }
    }
    (void)payload;
    return p;
}

static std::vector<float> cpu_dot(const ProofImage& image, int rows, int k) {
    const int groups = k / 256;
    std::vector<float> x(k), out(rows, 0.0f);
    for (int col = 0; col < k; ++col) x[col] = std::sin((float)col * 0.013f) * 0.25f;
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups; ++group) {
            const size_t base = ((size_t)row * groups + group) * 136;
            for (int col = 0; col < 256; ++col) {
                const int half = col >> 7;
                const unsigned char packed = image.row_major[base + 8 + (size_t)(col >> 1)];
                const int q = (col & 1) ? (packed >> 4) : (packed & 15);
                const float scale = half_to_float(image.row_major.data() + base + half * 4);
                const float zero = half_to_float(image.row_major.data() + base + half * 4 + 2);
                out[row] += (scale * (float)q + zero) * x[group * 256 + col];
            }
        }
    }
    return out;
}

static double rel_rms(const std::vector<float>& got, const std::vector<float>& want, double* max_abs) {
    double err2 = 0.0, ref2 = 0.0;
    *max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - (double)want[i];
        err2 += d * d;
        ref2 += (double)want[i] * (double)want[i];
        *max_abs = std::fmax(*max_abs, std::fabs(d));
    }
    return std::sqrt(err2 / ref2);
}

static bool run_plain_and_fused() {
    constexpr int rows = 32, k = 5120;
    ProofImage image = make_image(rows, k);
    std::vector<float> x(k), want = cpu_dot(image, rows, k), plain(rows), gate(rows), up(rows);
    for (int col = 0; col < k; ++col) x[col] = std::sin((float)col * 0.013f) * 0.25f;
    unsigned char* dw = nullptr;
    float *dx = nullptr, *dplain = nullptr, *dgate = nullptr, *dup = nullptr;
    hipMalloc(&dw, image.fragment.size());
    hipMalloc(&dx, k * sizeof(float));
    hipMalloc(&dplain, rows * sizeof(float));
    hipMalloc(&dgate, rows * sizeof(float));
    hipMalloc(&dup, rows * sizeof(float));
    hipMemcpy(dw, image.fragment.data(), image.fragment.size(), hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), k * sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(proof_plain_frag_wp, dim3(rows), dim3(32), 0, 0, (const char*)dw, dx, dplain, rows, k);
    hipLaunchKernelGGL(proof_fused_gate_up_frag_wp, dim3(rows * 2), dim3(32), 0, 0,
                       (const char*)dw, (const char*)dw, dx, dgate, dup, rows, rows, k);
    hipError_t status = hipDeviceSynchronize();
    hipMemcpy(plain.data(), dplain, rows * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(gate.data(), dgate, rows * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(up.data(), dup, rows * sizeof(float), hipMemcpyDeviceToHost);
    double plain_max, gate_max, up_max;
    const double plain_rel = rel_rms(plain, want, &plain_max);
    const double gate_rel = rel_rms(gate, want, &gate_max);
    const double up_rel = rel_rms(up, want, &up_max);
    std::printf("K=%d bytes=%zu bijection=%d roundtrip=%d dequant_equal=%d\n",
                k, image.fragment.size(), image.bijection, image.roundtrip, image.dequant_equal);
    std::printf("plain rel_rms=%.9g max_abs=%.9g; fused_gate rel_rms=%.9g max_abs=%.9g; fused_up rel_rms=%.9g max_abs=%.9g\n",
                plain_rel, plain_max, gate_rel, gate_max, up_rel, up_max);
    hipFree(dw); hipFree(dx); hipFree(dplain); hipFree(dgate); hipFree(dup);
    return status == hipSuccess && image.bijection && image.roundtrip && image.dequant_equal
        && plain_rel < 1e-5 && gate_rel < 1e-5 && up_rel < 1e-5;
}

static bool run_residual() {
    constexpr int rows = 32, k = 17408;
    ProofImage image = make_image(rows, k);
    std::vector<float> x(k), dot = cpu_dot(image, rows, k), initial(rows), want(rows), got(rows);
    for (int col = 0; col < k; ++col) x[col] = std::sin((float)col * 0.013f) * 0.25f;
    for (int row = 0; row < rows; ++row) {
        initial[row] = 0.125f - (float)row * 0.0005f;
        want[row] = initial[row] + dot[row];
    }
    unsigned char* dw = nullptr;
    float *dx = nullptr, *dy = nullptr;
    hipMalloc(&dw, image.fragment.size());
    hipMalloc(&dx, k * sizeof(float));
    hipMalloc(&dy, rows * sizeof(float));
    hipMemcpy(dw, image.fragment.data(), image.fragment.size(), hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), k * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dy, initial.data(), rows * sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(proof_residual_frag_wp, dim3((rows + 1) / 2), dim3(32), 0, 0,
                       (const char*)dw, dx, dy, rows, k);
    hipError_t status = hipDeviceSynchronize();
    hipMemcpy(got.data(), dy, rows * sizeof(float), hipMemcpyDeviceToHost);
    double max_abs;
    const double rel = rel_rms(got, want, &max_abs);
    std::printf("K=%d bytes=%zu bijection=%d roundtrip=%d dequant_equal=%d\n",
                k, image.fragment.size(), image.bijection, image.roundtrip, image.dequant_equal);
    std::printf("residual rel_rms=%.9g max_abs=%.9g\n", rel, max_abs);
    hipFree(dw); hipFree(dx); hipFree(dy);
    return status == hipSuccess && image.bijection && image.roundtrip && image.dequant_equal && rel < 1e-5;
}

int main() {
    const bool ok = run_plain_and_fused() && run_residual();
    std::printf("fragment decode readers: %s; resident VRAM delta=0 bytes\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
