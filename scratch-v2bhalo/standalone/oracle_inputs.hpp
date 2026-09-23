#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace xreuse_oracle {

enum class Pattern { Hashed, UnequalHalfWitness };

// A permutation of uint64_t: distinct counter words stay distinct. The first
// word therefore makes every 64-byte payload half distinct, not just probable.
inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

inline void put16(unsigned char* p, std::uint16_t x) {
    p[0] = static_cast<unsigned char>(x);
    p[1] = static_cast<unsigned char>(x >> 8);
}

inline void payload(unsigned char* p, std::size_t half, std::uint64_t seed) {
    for (unsigned word = 0; word < 8; ++word) {
        const auto v = mix64(seed + static_cast<std::uint64_t>(half) * 8 + word);
        for (unsigned byte = 0; byte < 8; ++byte)
            p[8 * word + byte] = static_cast<unsigned char>(v >> (8 * byte));
    }
}

// XBlock is the caller's existing 72-byte block_i4_128 {float d; int s;
// unsigned char qs[64];}. No HIP runtime, model, or production dependency.
// Global group/block indices drive generation: never reset counters per row,
// tile, half, or K group. SET and ADD arms receive identical input bytes.
template <class XBlock>
inline void fill_inputs(int M, int K, int N, std::vector<unsigned char>& w,
                        std::vector<XBlock>& x,
                        Pattern pattern = Pattern::Hashed) {
    static_assert(sizeof(XBlock) == 72, "requires the existing Xq ABI");
    if (M <= 0 || N <= 0 || K <= 0 || K % 256 != 0)
        throw std::invalid_argument("oracle requires positive M/N and K%256==0");
    w.resize(static_cast<std::size_t>(M) * (K / 256) * 136);
    x.resize(static_cast<std::size_t>(K / 128) * N);
    const bool witness = pattern == Pattern::UnequalHalfWitness;
    for (std::size_t g = 0; g < w.size() / 136; ++g) {
        for (unsigned h = 0; h < 2; ++h) {
            // Normal positive FP16 scale; z=-8*scale is exactly representable.
            // Distinct adjacent exponent fields guarantee unequal half scales.
            const auto r = mix64(UINT64_C(0x510e527fade682d1) + g * 2 + h);
            const std::uint16_t scale = witness ? 0x3c00 :
                static_cast<std::uint16_t>(0x1800 + h * 0x0400 + (r & 0x03ff));
            const std::uint16_t zero = static_cast<std::uint16_t>(0x8000 | (scale + 0x0c00));
            put16(w.data() + g * 136 + 4 * h, scale);
            put16(w.data() + g * 136 + 4 * h + 2, zero);
            auto* p = w.data() + g * 136 + 8 + 64 * h;
            if (witness) {
                for (unsigned b = 0; b < 64; ++b) p[b] = h ? 0x99 : 0x88;
            } else {
                payload(p, 2 * g + h, UINT64_C(0x243f6a8885a308d3));
            }
        }
    }
    for (std::size_t b = 0; b < x.size(); ++b) {
        const auto r = mix64(UINT64_C(0x13198a2e03707344) + b);
        x[b].d = witness ? 1.0f : 0.0073f * static_cast<float>(1 + r % 17);
        if (witness) {
            for (unsigned j = 0; j < 64; ++j) x[b].qs[j] = 0x11;
        } else {
            payload(x[b].qs, b, UINT64_C(0xa4093822299f31d0));
        }
        x[b].s = 0;
        for (unsigned j = 0; j < 64; ++j) {
            const int lo = x[b].qs[j] & 15;
            const int hi = x[b].qs[j] >> 4;
            x[b].s += (lo < 8 ? lo : lo - 16) + (hi < 8 ? hi : hi - 16);
        }
    }
}

inline float residual(std::size_t output_index) {
    const auto bits = mix64(UINT64_C(0x082efa98ec4e6c89) + output_index);
    return static_cast<float>(static_cast<int>(bits & 2047) - 1024) / 128.0f;
}

// Witness: A half0 rebiases to 0, half1 to +1; X=+1 and scales=1.
// Correct SET is K/2, archived R256 SET is 0. For ADD use residual=0.5f
// everywhere, so at K256 the exact expected outputs are 128 and 128.5.

} // namespace xreuse_oracle
