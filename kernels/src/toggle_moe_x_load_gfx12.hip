// Toggle microbench: isolate X-load cost in gemm_mixed_moe_grouped_wmma_gfx12
// (tag-4 E8 arm, the prefill hot-path for mfp4 on gfx1201).
//
// THREE variants compiled into one source via TOGGLE_VARIANT define:
//   0 = BASELINE  — kernel as-is (X loaded from global/L2 every kt)
//   1 = X_CONST   — X load replaced by register constant (kills X traffic;
//                   all dequant + WMMA still execute, so compute unchanged)
//   2 = X_LDS     — X staged in LDS once per block at the top, then read
//                   from LDS in the kt-loop (the proposed LDS-tile fix)
//
// Geometry matches the production kernel:
//   Grid: [ceil(M/16), m_total/16],  Block: 32 threads
//   k_grp = tid>>4,  m_lane = tid&15
//
// DECISION CRITERION (from the plan):
//   Build the LDS gfx12 variant IFF X_CONST/BASELINE >= 1.5x AND
//   X_LDS recovers >= 60% of the X_CONST-BASELINE gap.
//
// Compile flags added by the Rust bench: -DTOGGLE_VARIANT=<0|1|2>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef TOGGLE_VARIANT
#define TOGGLE_VARIANT 0
#endif

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

// E8 decode — identical to production kernel
__device__ __forceinline__ void e8_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(4 * i)) & 0xFu; sl += e[i]; }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 7); p[i] = coset ? c + 0.5f : c; }
}

// ============================================================
// Variant 2: X_LDS — shared tile declaration
// LDS layout: one 16 m_lane × K f16 X-tile per block.
// Each k_grp reads b = LDS[m_lane][k_off + k_grp*8 .. +8].
// LDS size = 16 * K * 2 bytes. For K=2048 this is 64 KB, which
// just fits (gfx1201 has 128 KB LDS / CU; 32-thread block = 1 wave).
// For K=7168 (down_proj): 16*7168*2 = 229 KB — EXCEEDS LDS.
// So we tile K in chunks of LDS_K_TILE = 2048 per LDS fill.
// We choose LDS_K_TILE = 256 per group (K=groups_per_row*256 always)
// so we can fill one GROUP at a time = 16*256*2 = 8 KB, always fits.
// ============================================================
#if TOGGLE_VARIANT == 2
// Shared tile: 16 lanes × 256 elements × sizeof(f16) = 8 KB per group fill
// This is a per-group X tile reuse (m_lane axis), NOT across row-tiles yet.
// It removes the m_lane L2 re-read for each of ceil(M/16) row-tiles
// within the same slot-tile (both k_grp lanes share one load per m_lane).
// Further row-tile sharing would require a bigger kernel restructure;
// this minimal version stages X once per group per slot-tile block.
#define LDS_GROUP_ELEMS 4096  /* 16 lanes × 256 elems/group */
#endif

__launch_bounds__(32, 2)
#if TOGGLE_VARIANT == 0
extern "C" __global__ void toggle_moe_x_load_baseline(
#elif TOGGLE_VARIANT == 1
extern "C" __global__ void toggle_moe_x_load_x_const(
#else
extern "C" __global__ void toggle_moe_x_load_x_lds(
#endif
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const unsigned char*      __restrict__ dtype_tags,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const _Float16*           __restrict__ X_src,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    // This microbench targets tag=4 (E8) only — the caller must supply all-E8
    // dtype_tags. We read the tag to keep the kernarg layout identical to the
    // production kernel; it is otherwise unused.
    (void)dtype_tags[expert_id];

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;

    // E8 row pointer: 16-byte header + (K/32)*17 bytes data
    const int row_block_bytes = (K / 32) * 17;
    const char* e8_row_ptr = A + (long long)safe_row * (16 + (long long)row_block_bytes);
    const float row_scale_f =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)e8_row_ptr);
    const char* blocks_ptr = e8_row_ptr + 16;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    // -------------------------------------------------------------------
    // Variant 1 (X_CONST): load ONE element from X once, broadcast to all
    // kt iterations. This removes the per-kt global/L2 X fetch entirely
    // while keeping the full E8 decode + WMMA arithmetic unchanged.
    // The WMMA result is garbage (same B for every kt), but we only care
    // about wall-clock, not output correctness.
    // -------------------------------------------------------------------
#if TOGGLE_VARIANT == 1
    // Load a constant b from the first valid X row (or zero_b if no valid row).
    // Same load pattern as production but executed exactly once, not 16*G times.
    half8_t x_const_b = zero_b;
    if (x_base != nullptr) {
        x_const_b = *(const half8_t*)(x_base + k_grp * 8);  // first element, k_grp offset
    }
#endif

    // -------------------------------------------------------------------
    // Variant 2 (X_LDS): stage the 16-lane × 256-elem X group into LDS
    // once per group, then read from LDS in the kt-loop.
    // LDS layout: [m_lane][256 f16] = 16*256*2 = 8 KB per group, always fits.
    // Load: both warps (k_grp 0 and 1) cooperate to fill 16*256=4096 f16.
    //   - 32 threads × 1 f16 per load × 128 iterations = 4096 f16.
    // -------------------------------------------------------------------
#if TOGGLE_VARIANT == 2
    __shared__ _Float16 x_lds[16][256];  // [m_lane][k_within_group]
#endif

    for (int g = 0; g < groups_per_row; g++) {
        const char* group_ptr = blocks_ptr + (long long)g * 136;
        const _Float16* xg = (x_base != nullptr) ? (x_base + (long long)g * 256) : nullptr;

#if TOGGLE_VARIANT == 2
        // Fill LDS: cooperatively load 16 rows × 256 cols of X.
        // 32 threads, each loads 128 f16 values (4096 / 32 = 128).
        // tid maps to: m_lane_fill = tid & 15, k_fill_base = (tid >> 4) * 128.
        // But 4096 / 32 = 128 — each thread loads one f16 per of 128 rounds?
        // Simpler: thread tid loads elements [tid, tid+32, tid+64, ...] (stride-32).
        // LDS[lane][k] = X_src[x_row_for_lane][g*256 + k].
        // We need the x_row for each m_lane — fetch sorted_slot_index per lane.
        // Since m_lane != tid in general (k_grp 0: m_lane=tid, k_grp 1: m_lane=tid-16),
        // both k_grp lanes for the same m_lane write the same x_row data.
        // Use tid & 15 = m_lane, tid >> 4 = k_grp as the natural split:
        //   k_grp 0 (tid 0..15): loads x_lds[m_lane][0..127]
        //   k_grp 1 (tid 16..31): loads x_lds[m_lane][128..255]
        {
            const int fill_lane = tid & 15;
            const int fill_kbase = (tid >> 4) * 128;  // 0 or 128
            // Resolve x_row for fill_lane
            const int fill_slot = slot_start + fill_lane;
            int fill_xrow = -1;
            if (fill_slot < m_total) {
                const int flat = sorted_slot_index[fill_slot];
                if (flat >= 0) fill_xrow = (x_row_div > 1) ? (flat / x_row_div) : flat;
            }
            const _Float16* fill_xg = (fill_xrow >= 0)
                ? (X_src + (long long)fill_xrow * K + (long long)g * 256 + fill_kbase)
                : nullptr;
            #pragma unroll 8
            for (int ki = 0; ki < 128; ki++) {
                x_lds[fill_lane][fill_kbase + ki] =
                    (fill_xg != nullptr) ? fill_xg[ki] : (_Float16)0.0f;
            }
        }
        __syncthreads();
#endif

        for (int kt = 0; kt < 16; kt++) {
            const int k_off = kt * 16;
            const int block_idx = k_off >> 5;            // 0..7
            const char* block_ptr = group_ptr + (long long)block_idx * 17;
            const float ssc = row_scale_f
                * cvt_e4m3_scale_to_f32(*(const unsigned char*)block_ptr) * 0.88f;
            const int cw_base = (k_off & 31) >> 3;        // 0 or 2
            const unsigned int cw =
                *(const unsigned int*)(block_ptr + 1 + (cw_base + k_grp) * 4);
            float p[8];
            e8_decode_index(cw, p);
            half8_t a_reg;
            #pragma unroll
            for (int i = 0; i < 8; i++) a_reg[i] = (_Float16)(p[i] * ssc);

            // B operand: varies by toggle variant
#if TOGGLE_VARIANT == 0
            // BASELINE: load from global/L2 every kt (same as production)
            half8_t b = (xg != nullptr)
                ? *(const half8_t*)(xg + k_off + k_grp * 8) : zero_b;
#elif TOGGLE_VARIANT == 1
            // X_CONST: always use the pre-loaded constant (no global load)
            half8_t b = x_const_b;
            (void)xg;  // suppress unused warning
#else
            // X_LDS: read from LDS (staged once above)
            half8_t b;
            const int lds_k = k_off + k_grp * 8;
            #pragma unroll
            for (int i = 0; i < 8; i++) b[i] = x_lds[m_lane][lds_k + i];
#endif
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b, acc);
        }

#if TOGGLE_VARIANT == 2
        __syncthreads();  // guard before next group's LDS fill
#endif
    }

    // Write output — byte-identical to production kernel's output path.
    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Layout: Y_grouped[out_col * M + out_row].
    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
