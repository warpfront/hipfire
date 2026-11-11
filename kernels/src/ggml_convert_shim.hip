// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Derived from ggml (MIT, Copyright (c) 2023-2026 The ggml authors): this file
// #includes the pinned upstream header third_party/ggml/src/ggml-cuda/dequantize.cuh
// (see third_party/ggml/PIN, LICENSE) and calls its `dequantize_{iq4_xs,q4_K,q5_K,
// q6_K,q8_0}` device functions; the kernel bodies below are transcriptions of
// upstream `dequantize_block_*` / `dequantize_block` (convert.cu) and
// `k_get_rows` / `k_get_rows_kq` (getrows.cu), which are non-callable `__global__`
// templates, see "Provenance".

// ggml_convert_shim: GGUF-native block -> f16 / f32 dequantisation, embedding row
// gather and the A8 activation round trip for gfx1201 (RDNA4, wave32).
//
// One `extern "C" __global__` export per (kernel, type, arch) with a fixed kernarg
// layout and a stable symbol (GGML-UMBRELLA-PLAN.md section 3.5). All exports:
// wave32, NO atomics, NO LDS (group segment 0), no hidden kernel arguments (the kernarg
// segment is exactly the explicit arguments listed below). Numeric contract `ggml.convert.<type>`: every
// output element is ONE evaluation of upstream's per-element expression
// (`d * scale * code` / `d * code - m`, f32 operations, no reduction) followed by
// the destination conversion, so results are independent of the launch geometry.
//
// FP contraction policy (explicit; differs from the mmvq shim): `#pragma clang fp contract(off)`
// below, ahead of the upstream include. Every product these kernels form is exact in f32
// (iq4_xs `d * (scale - 32) * kvalue`: 11 + 6 + 7 significant bits; q6_k `d * sc * (q - 32)`:
// 11 + 7 + 6; q8_0 `q * d`: 8 + 11; q4_k / q5_k `dall * sc * q`: 11 + 6 + 5), so contraction
// cannot change an f32 result. It DOES change an f16 one: with contraction on, the backend
// fuses q4_k / q5_k `d1*q - m1` and the f16 conversion into `v_fma_mixlo_f16`, which rounds the
// exact value to f16 ONCE, whereas upstream (and the host reference) round to f32 first and then
// to f16. Measured on this tree before the pragma: 1 ulp f16 differences in about 7e-6 (q4_k) and 9e-6 (q5_k) of the
// elements of a random 5120 x 5120 tensor (K2-GATES.md). With contraction off the
// f16 results are the double rounding upstream defines.
//
// Exports (symbols carry the `_gfx1201` suffix; `{t}` in iq4_xs, q4_k, q5_k, q6_k, q8_0):
//
//   ggml_dequant_{t}_f16_gfx1201   flat tensor -> f16
//   ggml_dequant_{t}_f32_gfx1201   flat tensor -> f32, optionally bf16-rounded
//     kernargs (offsets in bytes):
//       vx         @0   const void*   ggml blocks, row-major whole blocks per row
//                                     (iq4_xs 136 B / 256, q4_k 144 / 256, q5_k 176 / 256,
//                                      q6_k 210 / 256, q8_0 34 / 32)
//       dst        @8   half* / float*  element e of the flat tensor at dst[e] (see colmap)
//       colmap     @16  const int32*  NULL, or an int32 [K/chunk] chunk permutation:
//                                     colmap[g] = DESTINATION chunk of source chunk g
//       nb         @24  int32         number of weight blocks (m*K/qk), nb*qk <= INT32_MAX
//       K          @28  int32         row length in elements, multiple of qk
//       chunk      @32  int32         elements per colmap chunk (128 in use); only read when
//                                     colmap != NULL, then chunk % 128 == 0 and K % chunk == 0
//       bf16_round @36  int32         f32 export only: 1 = round the f32 result to the nearest
//                                     bfloat16 (RNE, NaN quietened, Inf kept; exactly
//                                     ggml_compute_fp32_to_bf16) and store it widened to f32,
//                                     0 = plain f32. The f16 export has no such argument.
//     Flat source element e = row*K + col (row = e / K). colmap == NULL: dst[e]. Otherwise
//     dst[row*K + colmap[col/chunk]*chunk + col%chunk]. colmap must be a permutation of
//     0..K/chunk-1 for the output to be fully written.
//     launch: iq4_xs / q4_k: block (32,1,1), grid (nb,1,1); q5_k / q6_k: block (64,1,1),
//     grid (nb,1,1); q8_0: block (256,1,1), grid (ceil(nb*32/512),1,1). One super-block per
//     workgroup for the k/i-quants, as upstream (convert.cu dequantize_row_*_cuda). No hidden
//     kernel arguments: kernarg segment 36 B (f16) / 40 B (f32).
//     With a colmap each thread's writes stay inside one 128-element window of its block
//     (iq4_xs 32, q4_k 64, q5_k 64, q6_k 128 aligned; q8_0 2), so the permuted store is the
//     unmodified upstream function called with a shifted destination base pointer.
//
//     Alignment of vx: iq4_xs 4 (its `qs` is read as 32-bit words, see dequantize_iq4_xs_lut16), q4_k 4,
//     q5_k 4 (`half2 dm`), q6_k 2, q8_0 2; dst 4 (f16 too: the q8_0 path stores element pairs), colmap 4.
//     The loader's 256-byte tensor alignment satisfies all of them.
//
//   ggml_get_row_{t}_f32_gfx1201    embedding gather, token ids from device memory
//   ggml_get_row1_{t}_f32_gfx1201   embedding gather, ONE token passed by value
//     kernargs: vx @0 (const void*, ggml table rows of K elements);
//               get_row : tok_ids @8 (const int32*), dst @16 (float*), K @24 (int32)
//               get_row1: tok @8 (int32 token id),    dst @16 (float*), K @24 (int32)
//     get_row dequantises table row tok_ids[blockIdx.y] into dst + blockIdx.y*K (f32);
//     get_row1 row `tok` into dst. launch: iq4_xs / q4_k block (32,1,1), q5_k / q6_k block
//     (64,1,1), grid (K/256, n_tokens, 1); q8_0 block (256,1,1), grid (ceil(K/512),
//     n_tokens, 1); get_row1 uses n_tokens = 1. Domain: K % qk == 0, 0 <= token < table
//     rows (not checked on device, as upstream), n_tokens <= 65535.
//
//   ggml_q81_roundtrip_gfx1201      in-place A8 emulation over f32 activations
//     kernargs: x @0 (float*, read and written), ne @8 (int32 elements, multiple of 32)
//     Each 32-element block is quantised exactly like ggml_q8_1_quantize_gfx1201
//     (amax, d = amax/127, q = amax == 0 ? 0 : roundf(x/d)) and written back as
//     (float)(half)d * (float)q, the half rounding being the producer's
//     `make_half2(d, sum)` conversion. launch: block (256,1,1), grid (ceil(ne/256),1,1).
//
// Provenance (pin 2e0d86bebc210947680e8f0d67daa79f3260c106, see third_party/ggml/PIN):
//  * dequantize_block_{iq4_xs,q4_K,q5_K,q6_K} (convert.cu): `dequantize_<t>(vx, ib,
//    yy + ib*QK_K, threadIdx.x)` with dst_t in {half, float, the bf16 proxy below}.
//  * dequantize_block (convert.cu) at qk = 32, qr = 1 for q8_0: the i00 / ib / iqs /
//    iybs index math and the `dequantize_q8_0` call are upstream's, the 2-D / 3-D batch
//    loops (ne01, ne02, ne03) are dropped because the tensor is flat. Upstream's
//    fp16 Pascal-or-newer q8_0 path (`dequantize_block_q8_0_f16`, LDS staged, `__hmul2`
//    of half(q) and the half scale) yields the same bits: q * d is exact in f32 (<= 8 + 11
//    bits), so rounding it to half once equals the half multiply's single rounding.
//  * k_get_rows (q8_0) / k_get_rows_kq (k-quants) (getrows.cu) with dst_t = float, ne11 =
//    ne12 = 1 and the token supplied as block-y index or by value. Upstream's grid-stride
//    loops (over super-blocks for the k-quants, `i00 += gridDim.y*blockDim.x` for q8_0) are
//    single iterations here because the launch grid covers the row (grid x >= K/256, resp.
//    K/512), so a block past the row does nothing; the loop variables are the same
//    `ib` / `i00`. The x/y grid roles are swapped against upstream's (tokens on y), as the
//    task's launch contract states. Group sizes are constants, not blockDim, so no hidden
//    kernel arguments are read (kernarg segment = explicit arguments).
//
// Build (see scripts/build-ggml-convert-shim.sh): hipfire's pinned core flags (`--genco
// --offload-arch=gfx1201 -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6`,
// rdna-compute/src/compiler.rs core_hipcc_flags) plus `-ffp-contract=fast-honor-pragmas
// -DGGML_USE_HIP -Ithird_party/ggml/include -Ithird_party/ggml/src -Ithird_party/ggml/src/
// ggml-cuda`. No fast-math; contraction is switched off by the pragma below.

#pragma clang fp contract(off)

#include "ggml-cuda/dequantize.cuh"
#include "ggml-cuda/vecdotq.cuh"   // get_int_from_table_16 (the iq4_xs lookup), get_int_b4

static_assert(ggml_cuda_get_physical_warp_size() == 32, "gfx1201 is wave32");

// Base alignments the contract (families/ggml_convert.rs) states for the weight operand.
static_assert(alignof(block_iq4_xs) == 2 && sizeof(block_iq4_xs) == 136, "block_iq4_xs layout");
static_assert(alignof(block_q4_K)   == 4 && sizeof(block_q4_K)   == 144, "block_q4_K layout");
static_assert(alignof(block_q5_K)   == 4 && sizeof(block_q5_K)   == 176, "block_q5_K layout");
static_assert(alignof(block_q6_K)   == 2 && sizeof(block_q6_K)   == 210, "block_q6_K layout");
static_assert(alignof(block_q8_0)   == 2 && sizeof(block_q8_0)   == 34,  "block_q8_0 layout");

namespace ggml_convert_shim {

// Work-group sizes are compile-time constants (not blockDim): the kernels then read no hidden
// kernel arguments and the kernarg segment is exactly the explicit arguments.
constexpr int Q8_0_THREADS = 256;

// bf16 destination proxy. Used as `dst_t` of the upstream dequantize functions: their
// `ggml_cuda_cast<dst_t>(f32)` ends in `return float(x)`, which constructs the proxy, so the
// bf16 rounding is applied to the one f32 result of each element and the widened value is
// stored in the same 4-byte slot. Rounding = ggml-impl.h ggml_compute_fp32_to_bf16 verbatim
// (round to nearest even, NaN quietened with bit 6, Inf exact, subnormals not flushed).
struct bf16_widened {
    float v;
    __device__ __forceinline__ bf16_widened(const float x) {
        const uint32_t u = __float_as_uint(x);
        uint32_t bits;
        if ((u & 0x7fffffff) > 0x7f800000) {
            bits = (u >> 16) | 64;
        } else {
            bits = (u + (0x7fff + ((u >> 16) & 1))) >> 16;
        }
        v = __uint_as_float(bits << 16);
    }
};
static_assert(sizeof(bf16_widened) == sizeof(float), "proxy must alias an f32 slot");

// iq4_xs super-block dequantisation: upstream `dequantize_iq4_xs` with the four per-element
// `kvalues_iq4nl[q4[j] & 0xf]` / `[q4[j] >> 4]` memory lookups replaced by ONE call of upstream's
// own register-resident 16-entry table select (`get_int_from_table_16`, vecdotq.cuh: the table
// words are compile-time constants, selected with byte permutes). It returns the same eight
// int8 table entries, so every output is `d * (int)kvalues_iq4nl[idx]` exactly as upstream; the
// scale, indexing and destination expressions are upstream's character for character. The
// dynamic table load would compile to a PC-relative .rodata access, which neither the emulator
// gate nor the lift model can represent, and costs a dependent load per element. Needs
// 4-byte alignment of vx (qs sits at offset 8 of a 136-byte block), see the header.
template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_xs_lut16(
        const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ibs].qs + 16*ib + 4*il;
    const float d = (float)x[ibs].d * ((((x[ibs].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[ibs].scales_h >> 2*ib) & 3) << 4)) - 32);
    const int2 v = get_int_from_table_16(get_int_b4(q4, 0), kvalues_iq4nl);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * (int8_t)(v.x >> (8*j)));
        y[j+16] = ggml_cuda_cast<dst_t>(d * (int8_t)(v.y >> (8*j)));
    }
}

// Destination base the upstream super-block function sees. It writes yy[off] for off in [0,
// QK_K); with a colmap the half h = off >> 7 of the block lands at
// dst[row*K + colmap[col/chunk]*chunk + col%chunk] with col the column of off's half start,
// which is `base + off` for base = dest(half h) - h*128 (every thread's writes lie inside one
// 128-element half, see the header). All index math is 32-bit: nb*QK_K <= INT32_MAX.
template <typename T>
static __device__ __forceinline__ T * kq_block_dst(
        T * dst, const int * colmap, const int64_t ib, const int K, const int chunk, const int h) {
    if (colmap == nullptr) {
        return dst + ib*QK_K;
    }
    const uint32_t e   = (uint32_t)ib * QK_K;
    const uint32_t row = e / (uint32_t)K;
    const uint32_t col = e - row*(uint32_t)K + (uint32_t)h*128;
    const uint32_t g   = col / (uint32_t)chunk;
    const int64_t  at  = (int64_t)row*K + (int64_t)colmap[g]*chunk + (int64_t)(col - g*(uint32_t)chunk);
    return dst + (at - (int64_t)h*128);
}

// q8_0 flat dequantisation (upstream `dequantize_block<QK8_0, QR8_0, dequantize_q8_0>`).
template <typename dst_t>
static __device__ __forceinline__ void dequant_q8_0(
        const void * vx, dst_t * y, const int * colmap, const int nb, const int K, const int chunk) {
    // 32-bit unsigned index math (the domain is nb*QK8_0 <= INT32_MAX): no 64-bit compares.
    const uint32_t i00 = 2u * (Q8_0_THREADS*blockIdx.x + threadIdx.x);
    if (i00 >= (uint32_t)nb * QK8_0) {
        return;
    }
    const int64_t ib       = i00/QK8_0;             // block index
    const int64_t iqs      = (i00%QK8_0)/QR8_0;     // quant index
    const int64_t iybs     = i00 - i00%QK8_0;       // y block start index
    const int64_t y_offset = QR8_0 == 1 ? 1 : QK8_0/2;

    float2 v;
    dequantize_q8_0(vx, ib, iqs, v);

    int64_t iy0 = iybs + iqs;
    if (colmap != nullptr) {
        const uint32_t e   = (uint32_t)iy0;
        const uint32_t row = e / (uint32_t)K;
        const uint32_t col = e - row*(uint32_t)K;
        const uint32_t g   = col / (uint32_t)chunk;
        iy0 = (int64_t)row*K + (int64_t)colmap[g]*chunk + (int64_t)(col - g*(uint32_t)chunk);
    }
    y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
    y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
}

template <typename dst_t>
static __device__ __forceinline__ void dequant_iq4_xs(
        const void * vx, dst_t * y, const int * colmap, const int nb, const int K, const int chunk) {
    const int i = blockIdx.x;
    if (i >= nb) {
        return;
    }
    dequantize_iq4_xs_lut16<dst_t>(vx, i, kq_block_dst(y, colmap, i, K, chunk, (threadIdx.x % 8) / 4), threadIdx.x);
}

template <typename dst_t>
static __device__ __forceinline__ void dequant_q4_K(
        const void * vx, dst_t * y, const int * colmap, const int nb, const int K, const int chunk) {
    const int i = blockIdx.x;
    if (i >= nb) {
        return;
    }
    dequantize_q4_K<dst_t>(vx, i, kq_block_dst(y, colmap, i, K, chunk, threadIdx.x / 16), threadIdx.x);
}

template <typename dst_t>
static __device__ __forceinline__ void dequant_q5_K(
        const void * vx, dst_t * y, const int * colmap, const int nb, const int K, const int chunk) {
    const int i = blockIdx.x;
    if (i >= nb) {
        return;
    }
    dequantize_q5_K<dst_t>(vx, i, kq_block_dst(y, colmap, i, K, chunk, threadIdx.x / 32), threadIdx.x);
}

template <typename dst_t>
static __device__ __forceinline__ void dequant_q6_K(
        const void * vx, dst_t * y, const int * colmap, const int nb, const int K, const int chunk) {
    const int i = blockIdx.x;
    if (i >= nb) {
        return;
    }
    dequantize_q6_K<dst_t>(vx, i, kq_block_dst(y, colmap, i, K, chunk, threadIdx.x / 32), threadIdx.x);
}

// Row bytes of one table row of K elements (64-bit).
template <typename block_t, int qk>
static __device__ __forceinline__ int64_t row_bytes(const int K) {
    return (int64_t)(K / qk) * (int64_t)sizeof(block_t);
}

// Embedding row gather, k-quant flavour (upstream k_get_rows_kq, ne11 = ne12 = 1): grid x
// indexes the super-blocks of the row, one table row per `tok`, output row `out_row`.
template <typename dst_t, dequantize_kq_t<dst_t> dequantize_kq, typename block_t>
static __device__ __forceinline__ void get_row_kq(
        const void * vx, const int tok, dst_t * dst, const int K, const int64_t out_row) {
    const int nsb = K/QK_K;                                          // super-blocks per row
    dst_t * dst_row = dst + out_row*K;
    const void * src0_row = (const char *) vx + (int64_t)tok * row_bytes<block_t, QK_K>(K);
    const int ib = blockIdx.x;
    if (ib < nsb) {
        dequantize_kq(src0_row, ib, dst_row + ib*QK_K, threadIdx.x);
    }
}

// Embedding row gather, q8_0 flavour (upstream k_get_rows<QK8_0, QR8_0, dequantize_q8_0>).
template <typename dst_t>
static __device__ __forceinline__ void get_row_q8_0(
        const void * vx, const int tok, dst_t * dst, const int K, const int64_t out_row) {
    dst_t * dst_row = dst + out_row*K;
    const void * src0_row = (const char *) vx + (int64_t)tok * row_bytes<block_q8_0, QK8_0>(K);
    const uint32_t i00 = 2u*(blockIdx.x*Q8_0_THREADS + threadIdx.x);
    if (i00 < (uint32_t)K) {
        const int64_t ib       = i00/QK8_0;
        const int64_t iqs      = (i00%QK8_0)/QR8_0;
        const int64_t iybs     = i00 - i00%QK8_0;
        const int64_t y_offset = QR8_0 == 1 ? 1 : QK8_0/2;

        float2 v;
        dequantize_q8_0(src0_row, ib, iqs, v);

        dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
        dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
    }
}

} // namespace ggml_convert_shim

// ---- export generators ------------------------------------------------------------------
// f16 and f32 dequant export of one type: tag, launch bound (= block size), device function.
#define GGML_CONVERT_DEQ_F16(tag, threads, fn)                                                   \
    __global__ __launch_bounds__(threads) void ggml_dequant_##tag##_f16_gfx1201(                 \
            const void * vx, half * dst, const int * colmap, const int nb, const int K,          \
            const int chunk) {                                                                    \
        ggml_convert_shim::fn<half>(vx, dst, colmap, nb, K, chunk);                              \
    }
#define GGML_CONVERT_DEQ_F32(tag, threads, fn)                                                   \
    __global__ __launch_bounds__(threads) void ggml_dequant_##tag##_f32_gfx1201(                 \
            const void * vx, float * dst, const int * colmap, const int nb, const int K,         \
            const int chunk, const int bf16_round) {                                              \
        if (bf16_round != 0) {                                                                    \
            ggml_convert_shim::fn<ggml_convert_shim::bf16_widened>(                               \
                vx, (ggml_convert_shim::bf16_widened *) dst, colmap, nb, K, chunk);               \
        } else {                                                                                  \
            ggml_convert_shim::fn<float>(vx, dst, colmap, nb, K, chunk);                          \
        }                                                                                         \
    }

// Evidence builds only: -DGGML_SHIM_ONLY=<0..20> compiles one export (numbering below) so per-export
// lift/emu verdicts can be taken. The shipped object defines none of it and contains all 21.
#ifdef GGML_SHIM_ONLY
#define GGML_CONVERT_WANT(n) (GGML_SHIM_ONLY == (n))
#else
#define GGML_CONVERT_WANT(n) 1
#endif

extern "C" {

// 0..4 iq4_xs, q4_k, q5_k, q6_k, q8_0 f16 dequant
#if GGML_CONVERT_WANT(0)
GGML_CONVERT_DEQ_F16(iq4_xs, 32, dequant_iq4_xs)
#endif
#if GGML_CONVERT_WANT(1)
GGML_CONVERT_DEQ_F16(q4_k, 32, dequant_q4_K)
#endif
#if GGML_CONVERT_WANT(2)
GGML_CONVERT_DEQ_F16(q5_k, 64, dequant_q5_K)
#endif
#if GGML_CONVERT_WANT(3)
GGML_CONVERT_DEQ_F16(q6_k, 64, dequant_q6_K)
#endif
#if GGML_CONVERT_WANT(4)
GGML_CONVERT_DEQ_F16(q8_0, 256, dequant_q8_0)
#endif

// 5..9 f32 / bf16-rounded-f32 dequant
#if GGML_CONVERT_WANT(5)
GGML_CONVERT_DEQ_F32(iq4_xs, 32, dequant_iq4_xs)
#endif
#if GGML_CONVERT_WANT(6)
GGML_CONVERT_DEQ_F32(q4_k, 32, dequant_q4_K)
#endif
#if GGML_CONVERT_WANT(7)
GGML_CONVERT_DEQ_F32(q5_k, 64, dequant_q5_K)
#endif
#if GGML_CONVERT_WANT(8)
GGML_CONVERT_DEQ_F32(q6_k, 64, dequant_q6_K)
#endif
#if GGML_CONVERT_WANT(9)
GGML_CONVERT_DEQ_F32(q8_0, 256, dequant_q8_0)
#endif

// 10..13 k-quant embedding gathers (ids and by-value pair each)
#if GGML_CONVERT_WANT(10)
__global__ __launch_bounds__(32) void ggml_get_row_iq4_xs_f32_gfx1201(
        const void * vx, const int * tok_ids, float * dst, const int K) {
    ggml_convert_shim::get_row_kq<float, ggml_convert_shim::dequantize_iq4_xs_lut16<float>, block_iq4_xs>(
        vx, tok_ids[blockIdx.y], dst, K, (int64_t) blockIdx.y);
}
#endif
#if GGML_CONVERT_WANT(11)
__global__ __launch_bounds__(32) void ggml_get_row1_iq4_xs_f32_gfx1201(
        const void * vx, const int tok, float * dst, const int K) {
    ggml_convert_shim::get_row_kq<float, ggml_convert_shim::dequantize_iq4_xs_lut16<float>, block_iq4_xs>(vx, tok, dst, K, 0);
}
#endif
#if GGML_CONVERT_WANT(12)
__global__ __launch_bounds__(32) void ggml_get_row_q4_k_f32_gfx1201(
        const void * vx, const int * tok_ids, float * dst, const int K) {
    ggml_convert_shim::get_row_kq<float, dequantize_q4_K<float>, block_q4_K>(
        vx, tok_ids[blockIdx.y], dst, K, (int64_t) blockIdx.y);
}
#endif
#if GGML_CONVERT_WANT(13)
__global__ __launch_bounds__(32) void ggml_get_row1_q4_k_f32_gfx1201(
        const void * vx, const int tok, float * dst, const int K) {
    ggml_convert_shim::get_row_kq<float, dequantize_q4_K<float>, block_q4_K>(vx, tok, dst, K, 0);
}
#endif
#if GGML_CONVERT_WANT(14)
__global__ __launch_bounds__(64) void ggml_get_row_q5_k_f32_gfx1201(
        const void * vx, const int * tok_ids, float * dst, const int K) {
    ggml_convert_shim::get_row_kq<float, dequantize_q5_K<float>, block_q5_K>(
        vx, tok_ids[blockIdx.y], dst, K, (int64_t) blockIdx.y);
}
#endif
#if GGML_CONVERT_WANT(15)
__global__ __launch_bounds__(64) void ggml_get_row1_q5_k_f32_gfx1201(
        const void * vx, const int tok, float * dst, const int K) {
    ggml_convert_shim::get_row_kq<float, dequantize_q5_K<float>, block_q5_K>(vx, tok, dst, K, 0);
}
#endif
#if GGML_CONVERT_WANT(16)
__global__ __launch_bounds__(64) void ggml_get_row_q6_k_f32_gfx1201(
        const void * vx, const int * tok_ids, float * dst, const int K) {
    ggml_convert_shim::get_row_kq<float, dequantize_q6_K<float>, block_q6_K>(
        vx, tok_ids[blockIdx.y], dst, K, (int64_t) blockIdx.y);
}
#endif
#if GGML_CONVERT_WANT(17)
__global__ __launch_bounds__(64) void ggml_get_row1_q6_k_f32_gfx1201(
        const void * vx, const int tok, float * dst, const int K) {
    ggml_convert_shim::get_row_kq<float, dequantize_q6_K<float>, block_q6_K>(vx, tok, dst, K, 0);
}
#endif

// 18, 19 q8_0 embedding gathers
#if GGML_CONVERT_WANT(18)
__global__ __launch_bounds__(256) void ggml_get_row_q8_0_f32_gfx1201(
        const void * vx, const int * tok_ids, float * dst, const int K) {
    ggml_convert_shim::get_row_q8_0<float>(vx, tok_ids[blockIdx.y], dst, K, (int64_t) blockIdx.y);
}
#endif
#if GGML_CONVERT_WANT(19)
__global__ __launch_bounds__(256) void ggml_get_row1_q8_0_f32_gfx1201(
        const void * vx, const int tok, float * dst, const int K) {
    ggml_convert_shim::get_row_q8_0<float>(vx, tok, dst, K, 0);
}
#endif

// 20 in-place A8 round trip: the producer's quantiser (ggml_mmvq_shim.hip ggml_q8_1_quantize_gfx1201,
// i.e. upstream quantize_q8_1) followed by the dequantisation of its block: out = half(d) * q.
#if GGML_CONVERT_WANT(20)
__global__ __launch_bounds__(256, 1) void ggml_q81_roundtrip_gfx1201(float * x, const int ne) {
    const int i0 = 256*blockIdx.x + threadIdx.x;
    if (i0 >= ne) {
        return;
    }

    const float xi = x[i0];
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    const half2 ds = make_half2(d, sum);   // the producer's stored scale (sum unused here)
    x[i0] = (float) __low2half(ds) * (float) q;
}
#endif

} // extern "C"
