// Standalone real-K1 GDN K2+resident gate probe. Intentionally untracked.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../../kernels/src/gdn_pre_batched.gfx1201.hip"
#include "../../../kernels/src/gdn_kkt_bc16_solve.gfx1201.hip"
#include "../../../kernels/src/gdn_wy_carry_output_q8.gfx1201.hip"
#define HIPFIRE_GDN_PREFETCH 1
#define HIPFIRE_GDN_DPP_REDUCE 1
#include "../../../kernels/src/gated_delta_net_q8_fast.hip"

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t e_ = (call);                                                \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,      \
                         #call, hipGetErrorString(e_));                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

template <typename T> static T* dev_alloc(size_t n) {
    T* p = nullptr;
    HIP_CHECK(hipMalloc(&p, n * sizeof(T)));
    return p;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return 0.5f * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

int main() {
    constexpr int N = 512, H = 48, D = 128, C = 64, J = 8;
    constexpr int KH = 16, ratio = 3, KD = KH * D, VD = H * D;
    constexpr int QKVD = 2 * KD + VD;
    constexpr int warmups = 3, samples = 20;
    constexpr float q_scale = 0.08838834764831845f;
    constexpr float eps = 1.0e-6f;

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int resident_blocks = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks, gdn_wy_carry_output_q8_gfx1201, 256, 0));

    const size_t nh = (size_t)N * H;
    const size_t x_count = (size_t)N * H * D;
    const size_t qkv_count = (size_t)N * QKVD;
    const size_t state_count = (size_t)H * D * D;
    const size_t scale_count = (size_t)H * D;
    const size_t b_count = (size_t)J * H * C * C;
    const size_t g_count = (size_t)J * H * C;
    const size_t raw_count = (size_t)N * KD;
    const size_t conv_count = (size_t)QKVD * 3;

    std::vector<float> hbeta(nh), hgate(nh), hqkv(qkv_count), hcw((size_t)QKVD * 4);
    std::vector<float> hconv(conv_count), hdt(H), halog(H), hsc(scale_count);
    std::vector<signed char> hsq(state_count);
    std::vector<_Float16> hef(state_count);
    for (int h = 0; h < H; ++h) {
        hdt[h] = -4.8f + 0.015f * (float)(h % 7);
        halog[h] = -1.5f + 0.01f * (float)(h % 5);
    }
    for (int t = 0; t < N; ++t) {
        for (int h = 0; h < H; ++h) {
            hbeta[(size_t)t * H + h] = -0.7f + 1.4f * (float)((t * 17 + h * 11) % 101) / 100.0f;
            hgate[(size_t)t * H + h] = -0.4f + 0.8f * (float)((t * 7 + h * 13) % 97) / 96.0f;
        }
        for (int c = 0; c < QKVD; ++c) {
            const float a = std::sin(0.0021f * (float)(1 + t * 37 + c * 3));
            const float b = std::cos(0.0037f * (float)(5 + t * 11 + c));
            hqkv[(size_t)t * QKVD + c] = 0.32f * a + 0.11f * b;
        }
    }
    for (int c = 0; c < QKVD; ++c) {
        hcw[(size_t)c * 4 + 0] = 0.04f;
        hcw[(size_t)c * 4 + 1] = -0.03f;
        hcw[(size_t)c * 4 + 2] = 0.02f;
        hcw[(size_t)c * 4 + 3] = 0.96f;
        hconv[(size_t)c * 3 + 0] = 0.03f * std::sin(0.01f * (float)c);
        hconv[(size_t)c * 3 + 1] = 0.02f * std::cos(0.013f * (float)c);
        hconv[(size_t)c * 3 + 2] = -0.01f * std::sin(0.017f * (float)c);
    }
    for (int h = 0; h < H; ++h) {
        for (int vr = 0; vr < D; ++vr) {
            hsc[(size_t)h * D + vr] = 0.0025f + 0.00001f * (float)((h + vr) % 17);
            for (int key = 0; key < D; ++key) {
                const size_t off = ((size_t)h * D + vr) * D + key;
                hsq[off] = (signed char)(((h * 19 + vr * 7 + key * 3) % 127) - 63);
                hef[off] = (_Float16)(0.00012f * std::sin(0.019f * (float)(1 + off % 10007)));
            }
        }
    }

    float *dbeta = dev_alloc<float>(nh), *dgate = dev_alloc<float>(nh);
    float *ddt = dev_alloc<float>(H), *dalog = dev_alloc<float>(H);
    float *dqkv = dev_alloc<float>(qkv_count), *dcw = dev_alloc<float>((size_t)QKVD * 4);
    float *dconv = dev_alloc<float>(conv_count), *dqraw = dev_alloc<float>(raw_count);
    float *dkraw = dev_alloc<float>(raw_count), *dq = dev_alloc<float>(x_count);
    float *dk = dev_alloc<float>(x_count), *dv = dev_alloc<float>(x_count);
    float *dB = dev_alloc<float>(b_count), *dG = dev_alloc<float>(g_count);
    signed char *dsq0 = dev_alloc<signed char>(state_count);
    float *dsc0 = dev_alloc<float>(scale_count);
    _Float16 *def0 = dev_alloc<_Float16>(state_count);
    signed char *dsqc = dev_alloc<signed char>(state_count), *dsqs = dev_alloc<signed char>(state_count);
    float *dscc = dev_alloc<float>(scale_count), *dscs = dev_alloc<float>(scale_count);
    _Float16 *defc = dev_alloc<_Float16>(state_count), *defs = dev_alloc<_Float16>(state_count);
    float *doutc = dev_alloc<float>(x_count), *douts = dev_alloc<float>(x_count);

    HIP_CHECK(hipMemcpy(dbeta, hbeta.data(), nh * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dgate, hgate.data(), nh * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddt, hdt.data(), H * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dalog, halog.data(), H * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dqkv, hqkv.data(), qkv_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcw, hcw.data(), hcw.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dconv, hconv.data(), conv_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dsq0, hsq.data(), state_count * sizeof(signed char), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dsc0, hsc.data(), scale_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(def0, hef.data(), state_count * sizeof(_Float16), hipMemcpyHostToDevice));

    const int groups = (N + 31) / 32;
    const int pre_blocks = KH * groups + ((VD + 255) / 256) * groups + groups;
    hipLaunchKernelGGL(gdn_pre_batched_gfx1201, dim3(pre_blocks), dim3(256), 0, 0,
                       dbeta, dgate, ddt, dalog, dqkv, dcw, dconv, dqraw, dkraw,
                       dv, dq, dk, H, KH, ratio, KD, VD, QKVD, N, q_scale, eps);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Freeze the actual K1 products for the device/CPU comparisons.
    std::vector<float> hq(x_count), hk(x_count), hv(x_count), pbeta(nh), pgate(nh);
    HIP_CHECK(hipMemcpy(hq.data(), dq, x_count * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hk.data(), dk, x_count * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hv.data(), dv, x_count * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pbeta.data(), dbeta, nh * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pgate.data(), dgate, nh * sizeof(float), hipMemcpyDeviceToHost));

    auto restore = [&](signed char* qs, float* ss, _Float16* es) {
        HIP_CHECK(hipMemcpy(qs, dsq0, state_count * sizeof(signed char), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(ss, dsc0, scale_count * sizeof(float), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(es, def0, state_count * sizeof(_Float16), hipMemcpyDeviceToDevice));
    };
    auto launch_candidate = [&]() {
        hipLaunchKernelGGL(gdn_kkt_bc16_solve_gfx1201, dim3(J, H), dim3(256), 0, 0,
                           dk, dgate, dbeta, dB, dG, N, H);
        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(gdn_wy_carry_output_q8_gfx1201, dim3(H, 4), dim3(256), 0, 0,
                           dq, dk, dv, dbeta, dB, dG, dsqc, dscc, defc, doutc,
                           N, H, (unsigned long long)17);
        HIP_CHECK(hipGetLastError());
    };
    auto launch_serial = [&]() {
        hipLaunchKernelGGL(gated_delta_net_q8_fast, dim3(H, D / 4, 1), dim3(32), 0, 0,
                           dq, dk, dv, dgate, dbeta, dsqs, dscs, douts,
                           N, H, D, 17u, reinterpret_cast<__half*>(defs));
        HIP_CHECK(hipGetLastError());
    };

    // Warm each path, restoring persistent state outside timed intervals.
    for (int i = 0; i < warmups; ++i) {
        restore(dsqc, dscc, defc); launch_candidate();
        restore(dsqs, dscs, defs); launch_serial();
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> candidate_us, serial_us;
    for (int i = 0; i < samples; ++i) {
        restore(dsqc, dscc, defc);
        HIP_CHECK(hipEventRecord(start)); launch_candidate(); HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        candidate_us.push_back(ms * 1000.0f);

        restore(dsqs, dscs, defs);
        HIP_CHECK(hipEventRecord(start)); launch_serial(); HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        serial_us.push_back(ms * 1000.0f);
    }

    // One fresh candidate for the f64 oracle.
    restore(dsqc, dscc, defc); launch_candidate(); HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> hout(x_count), csc(scale_count);
    std::vector<signed char> csq(state_count);
    std::vector<_Float16> cef(state_count);
    HIP_CHECK(hipMemcpy(hout.data(), doutc, x_count * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(csq.data(), dsqc, state_count * sizeof(signed char), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(csc.data(), dscc, scale_count * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(cef.data(), defc, state_count * sizeof(_Float16), hipMemcpyDeviceToHost));

    std::vector<double> state((size_t)D * D), delta(D), refout((size_t)N * D);
    for (int vr = 0; vr < D; ++vr)
        for (int key = 0; key < D; ++key)
            state[(size_t)vr * D + key] = (double)hsc[vr] * (double)hsq[(size_t)vr * D + key];
    for (int t = 0; t < N; ++t) {
        const double a = std::exp((double)pgate[(size_t)t * H]);
        const double b = (double)pbeta[(size_t)t * H];
        for (int vr = 0; vr < D; ++vr) {
            double pred = 0.0;
            for (int key = 0; key < D; ++key)
                pred += (double)hk[(size_t)t * H * D + key] * state[(size_t)vr * D + key];
            delta[vr] = b * ((double)hv[(size_t)t * H * D + vr] - a * pred);
        }
        for (int vr = 0; vr < D; ++vr) {
            double ov = 0.0;
            for (int key = 0; key < D; ++key) {
                double& s = state[(size_t)vr * D + key];
                s = a * s + (double)hk[(size_t)t * H * D + key] * delta[vr];
                ov += (double)hq[(size_t)t * H * D + key] * s;
            }
            refout[(size_t)t * D + vr] = ov;
        }
    }
    double out_sse = 0.0, ref_sse = 0.0, out_max = 0.0, ref_max = 0.0;
    for (int t = 0; t < N; ++t) for (int vr = 0; vr < D; ++vr) {
        const double ref = refout[(size_t)t * D + vr];
        const double got = (double)hout[(size_t)t * H * D + vr];
        const double e = got - ref;
        out_sse += e * e; ref_sse += ref * ref;
        out_max = std::max(out_max, std::abs(e)); ref_max = std::max(ref_max, std::abs(ref));
    }
    const double out_nrmse = std::sqrt(out_sse / std::max(ref_sse, 1.0e-30));
    double state_sse = 0.0, state_ref_sse = 0.0, state_max = 0.0, state_ref_max = 0.0;
    for (int vr = 0; vr < D; ++vr) for (int key = 0; key < D; ++key) {
        const size_t off = (size_t)vr * D + key;
        const double ref = state[off] + (double)hef[off];
        const double got = (double)csc[vr] * (double)csq[off] + (double)cef[off];
        const double e = got - ref;
        state_sse += e * e; state_ref_sse += ref * ref;
        state_max = std::max(state_max, std::abs(e)); state_ref_max = std::max(state_ref_max, std::abs(ref));
    }
    const double state_nrmse = std::sqrt(state_sse / std::max(state_ref_sse, 1.0e-30));

    const float cand_med = median(candidate_us), serial_med = median(serial_us);
    const float saving = serial_med - cand_med;
    const bool numeric_ok = std::isfinite(out_nrmse) && std::isfinite(state_nrmse) &&
        out_nrmse <= 0.002 && state_nrmse <= 0.002 &&
        out_max <= 0.01 * std::max(1.0, ref_max) &&
        state_max <= 0.01 * std::max(1.0, state_ref_max);
    const bool alive = resident_blocks >= 2 && numeric_ok && saving >= 68.257f;

    std::printf("device=%s\n", prop.name);
    std::printf("geometry=N%d H%d resident_grid=(%d,4,1) block=(256,1,1)\n", N, H, H);
    std::printf("runtime_active_blocks_per_mp=%d\n", resident_blocks);
    std::printf("k1_blocks=%d warmups=%d samples=%d\n", pre_blocks, warmups, samples);
    std::printf("candidate_k2_plus_resident_raw_us=");
    for (size_t i = 0; i < candidate_us.size(); ++i) std::printf("%s%.3f", i ? "," : "", candidate_us[i]);
    std::printf("\nserial_raw_us=");
    for (size_t i = 0; i < serial_us.size(); ++i) std::printf("%s%.3f", i ? "," : "", serial_us[i]);
    std::printf("\ncandidate_median_us=%.3f\nserial_median_us=%.3f\nsaving_us=%.3f\n", cand_med, serial_med, saving);
    std::printf("cpu_f64_output_nrmse=%.9g\ncpu_f64_output_max_abs_delta=%.9g\ncpu_f64_output_ref_max_abs=%.9g\n",
                out_nrmse, out_max, ref_max);
    std::printf("cpu_f64_final_state_nrmse=%.9g\ncpu_f64_final_state_max_abs_delta=%.9g\ncpu_f64_final_state_ref_max_abs=%.9g\n",
                state_nrmse, state_max, state_ref_max);
    std::printf("numeric_gate=%s\nperformance_gate=%s\ngate=%s\n",
                numeric_ok ? "ALIVE" : "KILL", saving >= 68.257f ? "ALIVE" : "KILL", alive ? "ALIVE" : "KILL");

    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
    return alive ? 0 : 2;
}
