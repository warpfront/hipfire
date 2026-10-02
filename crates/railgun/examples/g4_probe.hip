// G4 cache-visibility probe kernels (railgun design §5 G4, §7 M2).
//
// Compiled by `examples/g4_probe.rs` with hipcc for one arch and dispatched
// from one retained PM4 IB per batch of trials. Every kernel takes one
// by-value `P` (64 explicit bytes) and never touches the hidden kernarg suffix
// (no blockDim/gridDim), so the PM4 lowering needs only the kernarg pointer.
//
// Data word i of trial epoch e is `val(e, i)`; e-1 is the "old" value a stale
// cache line would return, `val(e, i) ^ FIRST` is the first writer's value of
// a WAW pair. Each reader classifies every dword it loads into per-trial
// counters (see `Slot`).
#include <hip/hip_runtime.h>

typedef unsigned int u32;
typedef unsigned long long u64;
typedef u32 u32x4 __attribute__((ext_vector_type(4)));
typedef u32 u32x8 __attribute__((ext_vector_type(8)));
#define CONST_AS __attribute__((address_space(4)))

struct P {
    u32* data;     // probe allocation (footprint `n` dwords)
    u32* result;   // this trial's counter row (SLOTS dwords)
    u32* keys;     // this trial's placement keys (3 * nwg dwords) or null
    u32* aux;      // unrelated allocation for the independent node
    u32 n;         // footprint in dwords
    u32 cover;     // dwords each reader workgroup reads (n, or n / nwg)
    u32 epoch;
    u32 rot;       // producer partition rotation (placement)
    u32 flags;
    u32 delay;     // s_sleep rounds per load/store batch (race widening)
    u32 nwg;
    u32 layout;    // 0 = contiguous partitions, 1 = 32-byte interleave
};

enum Flag : u32 { FIRST_WRITER = 1, RECORD_KEYS = 2, CLASSIFY_OLD = 4 };
// Counter slots per trial.
enum Slot : u32 { OK = 0, OLD = 1, FUTURE = 2, FIRSTV = 3, OTHER = 4, CHECKED = 5, SLOTS = 8 };
#define FIRST_XOR 0x5a5a5a5au
#define WG 256

__device__ __forceinline__ u32 val(u32 e, u32 i) { return (e * 0x9E3779B1u) ^ (i * 0x85EBCA77u); }

// Hardware placement key: HW_ID1 without the wave/SIMD fields, i.e.
// (SE, SA, WGP). Read with s_getreg (hwreg id 23, bits [31:0]).
__device__ __forceinline__ u32 hw_key() {
    return __builtin_amdgcn_s_getreg(23 | (31 << 11)) & ~0x3ffu;
}

__device__ __forceinline__ void record_key(const P& p, u32 role, u32 index) {
    if ((p.flags & RECORD_KEYS) && threadIdx.x == 0) p.keys[role * p.nwg + index] = hw_key();
}

__device__ __forceinline__ void spin(u32 rounds) {
    for (u32 r = 0; r < rounds; ++r) __builtin_amdgcn_s_sleep(8);
}

// Classify `v` loaded from index `i` against trial epoch `e`.
// `expect_old`: the reader runs before this trial's writer (WAR reader).
__device__ __forceinline__ void classify(u32 v, u32 i, u32 e, bool expect_old, u32 c[6]) {
    u32 now = val(e, i), old = val(e - 1, i);
    u32 want = expect_old ? old : now;
    if (v == want) c[OK]++;
    else if (!expect_old && v == old) c[OLD]++;
    else if (expect_old && v == now) c[FUTURE]++;
    else if (v == (now ^ FIRST_XOR)) c[FIRSTV]++;
    else c[OTHER]++;
    c[CHECKED]++;
}

// Wave-level reduction and one global atomic per wave and slot. Deliberately
// no __syncthreads(): in WGP mode its workgroup acquire emits an L0
// invalidate (global_inv / buffer_gl0_inv), which would erase the very cache
// lines the probe is trying to observe.
__device__ __forceinline__ u32 wave_sum(u32 x) {
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor(x, o, 32);
    return x;
}

__device__ void flush_counts(const P& p, const u32 c[6]) {
    // Slots OLD..CHECKED (1..5); OK is implied by CHECKED minus the rest.
#pragma unroll
    for (int k = 1; k < 6; ++k) {
        u32 t = wave_sum(c[k]);
        if ((threadIdx.x & 31) == 0 && t) atomicAdd(&p.result[k], t);
    }
}

__device__ __forceinline__ u32 region_base(const P& p, u32 w) { return p.cover == p.n ? 0u : w * p.cover; }

// Index of the k-th dword written by producer workgroup w.
__device__ __forceinline__ u32 produce_index(const P& p, u32 part, u32 j) {
    u32 per = p.n / p.nwg;
    if (p.layout == 0) return part * per + j;
    // 32-byte interleave: chunk c (8 dwords) belongs to partition c % nwg.
    u32 chunk = j / 8, lane = j % 8;
    return (chunk * p.nwg + part) * 8 + lane;
}

extern "C" __global__ __launch_bounds__(WG) void g4_zero(P p) {
    // Zero `n` result dwords (the batch's counter rows); `data` unused.
    for (u32 i = blockIdx.x * WG + threadIdx.x; i < p.n; i += p.nwg * WG) p.result[i] = 0;
}

extern "C" __global__ __launch_bounds__(WG) void g4_init(P p) {
    for (u32 i = blockIdx.x * WG + threadIdx.x; i < p.n; i += p.nwg * WG) p.data[i] = val(p.epoch, i);
}

// Writer. Workgroup w writes partition (w + rot) % nwg; each thread stores 8
// consecutive dwords per step, the store shape of the rotate kernels.
extern "C" __global__ __launch_bounds__(WG) void g4_produce(P p) {
    u32 w = blockIdx.x;
    u32 part = (w + p.rot) % p.nwg;
    record_key(p, 1, part);
    u32 per = p.n / p.nwg;
    u32 x = (p.flags & FIRST_WRITER) ? FIRST_XOR : 0u;
    for (u32 j0 = threadIdx.x * 8; j0 < per; j0 += WG * 8) {
        spin(p.delay);
#pragma unroll
        for (u32 k = 0; k < 8; ++k) {
            u32 i = produce_index(p, part, j0 + k);
            p.data[i] = val(p.epoch, i) ^ x;
        }
    }
}

// VMEM reader: 128-bit loads over the workgroup's region. With CLASSIFY_OLD
// it is a checked WAR/pre-writer reader (expects e-1, counts FUTURE); without
// it, it only warms caches.
extern "C" __global__ __launch_bounds__(WG) void g4_read_v(P p) {
    u32 w = blockIdx.x;
    record_key(p, 0, w);
    u32 base = region_base(p, w);
    u32 c[6] = {0, 0, 0, 0, 0, 0};
    u32 sink = 0;
    const u32x4* d = reinterpret_cast<const u32x4*>(p.data + base);
    for (u32 q = threadIdx.x; q < p.cover / 4; q += WG) {
        spin(p.delay);
        u32x4 v = d[q];
        if (p.flags & CLASSIFY_OLD) {
            u32 i = base + q * 4;
            classify(v.x, i, p.epoch, true, c);
            classify(v.y, i + 1, p.epoch, true, c);
            classify(v.z, i + 2, p.epoch, true, c);
            classify(v.w, i + 3, p.epoch, true, c);
        } else {
            sink ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (p.flags & CLASSIFY_OLD) {
        flush_counts(p, c);
    } else if (sink == 0x13579bdfu && threadIdx.x == 0) {
        p.result[7] = sink;  // keep the loads live
    }
}

// VMEM consumer: expects this trial's value.
extern "C" __global__ __launch_bounds__(WG) void g4_consume_v(P p) {
    u32 w = blockIdx.x;
    record_key(p, 2, w);
    u32 base = region_base(p, w);
    u32 c[6] = {0, 0, 0, 0, 0, 0};
    const u32x4* d = reinterpret_cast<const u32x4*>(p.data + base);
    for (u32 q = threadIdx.x; q < p.cover / 4; q += WG) {
        u32x4 v = d[q];
        u32 i = base + q * 4;
        classify(v.x, i, p.epoch, false, c);
        classify(v.y, i + 1, p.epoch, false, c);
        classify(v.z, i + 2, p.epoch, false, c);
        classify(v.w, i + 3, p.epoch, false, c);
    }
    flush_counts(p, c);
}

// SMEM (scalar cache) reader: each wave s_loads 1/8 of the region in 32-byte
// chunks through a constant-address-space pointer (uniform address).
__device__ __forceinline__ void scalar_pass(const P& p, bool check) {
    u32 w = blockIdx.x;
    u32 wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
    u32 base = region_base(p, w);
    u32 per_wave = p.cover / 8;
    u32 start = __builtin_amdgcn_readfirstlane(base + wave * per_wave);
    const CONST_AS u32x8* d = (const CONST_AS u32x8*)(p.data + start);
    u32 c[6] = {0, 0, 0, 0, 0, 0};
    u32 sink = 0;
    for (u32 q = 0; q < per_wave / 8; ++q) {
        u32x8 v = d[q];
        u32 i = start + q * 8;
        if (check) {
#pragma unroll
            for (u32 k = 0; k < 8; ++k) classify(v[k], i + k, p.epoch, false, c);
        } else {
            sink ^= v[0] ^ v[1] ^ v[2] ^ v[3] ^ v[4] ^ v[5] ^ v[6] ^ v[7];
        }
    }
    if (check) {
        // Scalar counts are wave-uniform: only lane 0 contributes.
        if ((threadIdx.x & 31) != 0)
            for (int k = 0; k < 6; ++k) c[k] = 0;
        flush_counts(p, c);
    } else if (sink == 0x13579bdfu && threadIdx.x == 0) {
        p.result[7] = sink;
    }
}

extern "C" __global__ __launch_bounds__(WG) void g4_read_s(P p) {
    record_key(p, 0, blockIdx.x);
    scalar_pass(p, false);
}

extern "C" __global__ __launch_bounds__(WG) void g4_consume_s(P p) {
    record_key(p, 2, blockIdx.x);
    scalar_pass(p, true);
}

// Independent node (the MQ4R `moe_router` position): touches only `aux`.
extern "C" __global__ __launch_bounds__(WG) void g4_indep(P p) {
    u32 i = blockIdx.x * WG + threadIdx.x;
    if (i < 4096) p.aux[i] = p.aux[i] * 1664525u + 1013904223u + p.epoch;
}
