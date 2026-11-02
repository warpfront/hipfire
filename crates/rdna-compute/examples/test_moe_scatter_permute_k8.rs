//! Byte-exact validation for the SGLang-style MoE scatter pipeline:
//!   Phase 1: moe_scatter_histogram_k8
//!   Phase 2: moe_scatter_offsets_k8
//!   Phase 3: moe_scatter_permute_k8
//!
//! Path 2 stage 1 — produces sorted_slot_index + expert_tile_ids for the
//! grouped-WMMA-GEMM MoE prefill (Stage 2). See memory entry
//! `project_next_session_a3b_path2_stage1` for the algorithm spec.
//!
//! Validates two cases:
//!   1. Toy:   N=4, K_TOP=2, E=4, BLOCK_M=2  (hand-checkable)
//!   2. A3B:   N=256, K_TOP=8, E=256, BLOCK_M=16
//!
//! Compares:
//!   - expert_token_counts (raw and padded)
//!   - expert_offsets (exclusive prefix of padded)
//!   - expert_tile_ids (deterministic by expert)
//!   - sorted_slot_index — bucket SETS (intra-bucket order is non-
//!     deterministic because Phase 3 uses LDS atomic increments)
//!
//! Then the production launcher `Gpu::moe_scatter_fused_k8` (exact gfx1201: the
//! multi-workgroup stable twin `moe_scatter_stable_k8.hip` once the batch has
//! four 1024-slot chunks; the single-workgroup kernel otherwise) against the
//! incumbent single-workgroup kernel launched directly from
//! `kernels::MOE_SCATTER_FUSED_K8_SRC`, on A3B shapes (8192 and 7808 rows,
//! 256 experts, top-8, block_m 16), a skewed routing, ragged sizes and
//! routings with unrouted ids (-1, E, huge). Both are held to the CPU
//! reference: counts, offsets, the full -1-padded tile table, bucket sets,
//! `-1` padding at every bucket tail and after `m_total`, and
//! `inverse_perm` (every routed slot -> its position, unrouted -> -1); the
//! stable twin must additionally equal the CPU input-order scatter exactly
//! (ascending flat slot within each expert). Finally the consumer the
//! within-expert order could touch, `moe_down_combine_grouped_k8`, runs over
//! grouped rows whose content is a function of the flat slot only (what a
//! grouped GEMM row is) and its output must be byte-identical between the
//! incumbent and the production scatter.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_scatter_permute_k8

use hip_bridge::KernargBlob;
use rdna_compute::{kernels, DType, Gpu, GpuTensor};

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc_tensor i32");
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod i32");
    t
}

fn alloc_i32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[n * 4], DType::Raw)
        .expect("alloc_tensor i32");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("memset zero");
    t
}

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, n: usize) -> Vec<i32> {
    let mut data = vec![0i32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("memcpy_dtoh i32");
    data
}

/// Deterministic LCG for topk_indices generation. Glibc constants.
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

fn gen_topk_indices(n: usize, k_top: usize, num_experts: usize, seed: u32) -> Vec<i32> {
    let mut state = seed;
    let mut out = vec![0i32; n * k_top];
    for tok in 0..n {
        // Per-token unique k_top experts: draw without replacement.
        let mut chosen: Vec<i32> = Vec::with_capacity(k_top);
        while chosen.len() < k_top {
            let cand = (lcg(&mut state) as usize % num_experts) as i32;
            if !chosen.contains(&cand) {
                chosen.push(cand);
            }
        }
        for k in 0..k_top {
            out[tok * k_top + k] = chosen[k];
        }
    }
    out
}

/// CPU reference: same algorithm as the GPU pipeline.
struct CpuRef {
    raw_counts: Vec<i32>,
    padded_counts: Vec<i32>,
    offsets: Vec<i32>,
    sorted_slot_index: Vec<i32>, // [m_total]
    expert_tile_ids: Vec<i32>,   // [m_total / block_m]
    m_total: usize,
}

fn cpu_scatter(topk_indices: &[i32], num_experts: usize, block_m: usize) -> CpuRef {
    // Phase 1: raw histogram.
    let mut raw_counts = vec![0i32; num_experts];
    for &e in topk_indices {
        if e >= 0 && (e as usize) < num_experts {
            raw_counts[e as usize] += 1;
        }
    }

    // Phase 2: pad + exclusive prefix sum.
    let padded_counts: Vec<i32> = raw_counts
        .iter()
        .map(|&r| {
            let bm = block_m as i32;
            ((r + bm - 1) / bm) * bm
        })
        .collect();
    let mut offsets = vec![0i32; num_experts + 1];
    {
        let mut acc = 0;
        for i in 0..num_experts {
            offsets[i] = acc;
            acc += padded_counts[i];
        }
        offsets[num_experts] = acc;
    }
    let m_total = offsets[num_experts] as usize;

    // Phase 3: scatter + tile_ids. In-input-order scatter gives a
    // canonical permutation; the GPU's order will differ inside each
    // bucket but the SET equality check below tolerates that.
    let mut sorted_slot_index = vec![-1i32; m_total];
    let mut bucket_fill = vec![0i32; num_experts];
    for (flat, &e) in topk_indices.iter().enumerate() {
        if e >= 0 && (e as usize) < num_experts {
            let eu = e as usize;
            let pos = offsets[eu] + bucket_fill[eu];
            sorted_slot_index[pos as usize] = flat as i32;
            bucket_fill[eu] += 1;
        }
    }

    // Tile ids: walk by expert.
    let num_tiles = m_total / block_m;
    let mut expert_tile_ids = vec![0i32; num_tiles];
    for e in 0..num_experts {
        let s = (offsets[e] as usize) / block_m;
        let en = (offsets[e + 1] as usize) / block_m;
        for t in s..en {
            expert_tile_ids[t] = e as i32;
        }
    }

    CpuRef {
        raw_counts,
        padded_counts,
        offsets,
        sorted_slot_index,
        expert_tile_ids,
        m_total,
    }
}

fn compare_set_per_bucket(
    label: &str,
    gpu_sorted: &[i32],
    cpu_ref: &CpuRef,
    num_experts: usize,
    block_m: usize,
) -> usize {
    let mut fails = 0;
    for e in 0..num_experts {
        let start = cpu_ref.offsets[e] as usize;
        let end = cpu_ref.offsets[e + 1] as usize;
        let raw = cpu_ref.raw_counts[e] as usize;

        // The first `raw` slots in this bucket are real entries; the
        // remaining (padded - raw) are -1 sentinels. The GPU's order
        // inside the real range is non-deterministic, so collect both
        // sides as sets and compare sorted.
        let mut gpu_bucket: Vec<i32> = gpu_sorted[start..end].to_vec();
        let mut cpu_bucket: Vec<i32> = cpu_ref.sorted_slot_index[start..end].to_vec();
        gpu_bucket.sort();
        cpu_bucket.sort();

        if gpu_bucket != cpu_bucket {
            if fails < 3 {
                eprintln!(
                    "  {label}: expert {e} bucket mismatch (raw={raw} pad={} range=[{start},{end}))",
                    cpu_ref.padded_counts[e]
                );
                eprintln!("    cpu={:?}", &cpu_bucket[..cpu_bucket.len().min(8)]);
                eprintln!("    gpu={:?}", &gpu_bucket[..gpu_bucket.len().min(8)]);
            }
            fails += 1;
        }

        // Also verify every non-(-1) entry from the GPU side really
        // maps back to expert e in topk_indices.
        for &flat in &gpu_bucket {
            if flat == -1 {
                continue;
            }
            // flat = tok * K_TOP + k → recovered expert must equal e
            // (cross-checked against the original topk_indices below).
            let _ = flat;
        }

        // Padding count must match.
        let gpu_pads = gpu_sorted[start..end].iter().filter(|&&x| x == -1).count();
        let cpu_pads = end - start - raw;
        if gpu_pads != cpu_pads {
            eprintln!(
                "  {label}: expert {e} padding count mismatch (cpu={cpu_pads} gpu={gpu_pads})"
            );
            fails += 1;
        }

        let _ = block_m; // currently unused; kept for future per-tile checks
    }
    fails
}

fn compare_vec(label: &str, a: &[i32], b: &[i32]) -> usize {
    if a.len() != b.len() {
        eprintln!("{label}: length mismatch (cpu={} gpu={})", a.len(), b.len());
        return 1;
    }
    let mut fails = 0;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            if fails < 3 {
                eprintln!("  {label}[{i}]: cpu={x} gpu={y}");
            }
            fails += 1;
        }
    }
    if fails > 0 {
        eprintln!("{label}: FAIL {}/{}", fails, a.len());
    } else {
        println!("{label}: byte-exact ({} entries)", a.len());
    }
    fails
}

fn validate_back_pointers(
    label: &str,
    sorted_slot_index: &[i32],
    topk_indices: &[i32],
    k_top: usize,
    cpu_ref: &CpuRef,
    num_experts: usize,
) -> usize {
    let mut fails = 0;
    for e in 0..num_experts {
        let s = cpu_ref.offsets[e] as usize;
        let en = cpu_ref.offsets[e + 1] as usize;
        for &flat in &sorted_slot_index[s..en] {
            if flat == -1 {
                continue;
            }
            let recovered = topk_indices[flat as usize];
            if recovered as usize != e {
                if fails < 3 {
                    eprintln!(
                        "  {label}: flat {flat} sits in bucket e={e} but topk_indices[{flat}] = {recovered}"
                    );
                }
                fails += 1;
            }
            // Also: flat must be in range.
            if (flat as usize) >= topk_indices.len() {
                fails += 1;
            }
            let _ = k_top;
        }
    }
    if fails > 0 {
        eprintln!("{label}: back-pointer FAIL ({fails})");
    } else {
        println!("{label}: back-pointers consistent");
    }
    fails
}

fn run_case(
    gpu: &mut Gpu,
    label: &str,
    n: usize,
    k_top: usize,
    num_experts: usize,
    block_m: usize,
    seed: u32,
) -> usize {
    println!("\n=== {label}: N={n} K_TOP={k_top} E={num_experts} BLOCK_M={block_m} ===");

    let topk_indices_host = gen_topk_indices(n, k_top, num_experts, seed);
    let cpu_ref = cpu_scatter(&topk_indices_host, num_experts, block_m);
    let total_slots = n * k_top;
    println!(
        "  total_slots={total_slots} m_total={} num_tiles={}",
        cpu_ref.m_total,
        cpu_ref.m_total / block_m
    );

    let m_total_max = total_slots + num_experts * (block_m - 1);

    // Upload + alloc.
    let d_topk = upload_i32(gpu, &topk_indices_host);
    let d_counts = alloc_i32_zeros(gpu, num_experts);
    let d_offsets = alloc_i32_zeros(gpu, num_experts + 1);
    let d_sorted = alloc_i32_zeros(gpu, m_total_max);
    let d_tile_ids = alloc_i32_zeros(gpu, m_total_max / block_m);
    let d_inverse = alloc_i32_zeros(gpu, total_slots);

    // Phase 1.
    gpu.moe_scatter_histogram_k8(&d_topk, &d_counts, total_slots, num_experts)
        .expect("histogram");
    gpu.hip.device_synchronize().expect("sync after phase 1");
    let gpu_raw_counts = download_i32(gpu, &d_counts, num_experts);
    let mut fails = compare_vec(
        &format!("[{label}] raw_counts"),
        &cpu_ref.raw_counts,
        &gpu_raw_counts,
    );

    // Phase 2.
    gpu.moe_scatter_offsets_k8(&d_counts, &d_offsets, num_experts, block_m)
        .expect("offsets");
    gpu.hip.device_synchronize().expect("sync after phase 2");
    let gpu_padded_counts = download_i32(gpu, &d_counts, num_experts);
    fails += compare_vec(
        &format!("[{label}] padded_counts"),
        &cpu_ref.padded_counts,
        &gpu_padded_counts,
    );
    let gpu_offsets = download_i32(gpu, &d_offsets, num_experts + 1);
    fails += compare_vec(
        &format!("[{label}] expert_offsets"),
        &cpu_ref.offsets,
        &gpu_offsets,
    );

    // Phase 3.
    gpu.moe_scatter_permute_k8(
        &d_topk,
        &d_offsets,
        &d_sorted,
        &d_tile_ids,
        &d_inverse,
        total_slots,
        num_experts,
        cpu_ref.m_total,
        block_m,
    )
    .expect("permute");
    gpu.hip.device_synchronize().expect("sync after phase 3");

    let gpu_sorted = download_i32(gpu, &d_sorted, cpu_ref.m_total);
    let gpu_tile_ids = download_i32(gpu, &d_tile_ids, cpu_ref.m_total / block_m);

    // expert_tile_ids is deterministic — compare byte-exact.
    fails += compare_vec(
        &format!("[{label}] expert_tile_ids"),
        &cpu_ref.expert_tile_ids,
        &gpu_tile_ids,
    );

    // sorted_slot_index — bucket SET equality.
    let bucket_fails = compare_set_per_bucket(
        &format!("[{label}] sorted_slot_index"),
        &gpu_sorted,
        &cpu_ref,
        num_experts,
        block_m,
    );
    if bucket_fails == 0 {
        println!(
            "[{label}] sorted_slot_index: bucket sets match ({} buckets)",
            num_experts
        );
    }
    fails += bucket_fails;

    fails += validate_back_pointers(
        &format!("[{label}] sorted_slot_index back-pointers"),
        &gpu_sorted,
        &topk_indices_host,
        k_top,
        &cpu_ref,
        num_experts,
    );

    println!("[{label}] total fails: {fails}");
    fails
}

// ---------------------------------------------------------------------------
// Production launcher vs the incumbent single-workgroup kernel.
// ---------------------------------------------------------------------------

const POISON: i32 = 0x5a5a_5a5a;

struct ScatterOut {
    counts: Vec<i32>,
    offsets: Vec<i32>,
    sorted: Vec<i32>,
    tiles: Vec<i32>,
    inverse: Vec<i32>,
}

struct ScatterBufs {
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inverse: GpuTensor,
}

fn poisoned_bufs(
    gpu: &mut Gpu,
    e: usize,
    total_slots: usize,
    m_total_max: usize,
    block_m: usize,
) -> ScatterBufs {
    let mut p = |n: usize| upload_i32(gpu, &vec![POISON; n]);
    ScatterBufs {
        counts: p(e),
        offsets: p(e + 1),
        sorted: p(m_total_max),
        tiles: p(m_total_max / block_m),
        inverse: p(total_slots),
    }
}

fn download_out(
    gpu: &Gpu,
    b: &ScatterBufs,
    e: usize,
    total_slots: usize,
    m_total_max: usize,
    block_m: usize,
) -> ScatterOut {
    gpu.hip.device_synchronize().expect("sync scatter");
    ScatterOut {
        counts: download_i32(gpu, &b.counts, e),
        offsets: download_i32(gpu, &b.offsets, e + 1),
        sorted: download_i32(gpu, &b.sorted, m_total_max),
        tiles: download_i32(gpu, &b.tiles, m_total_max / block_m),
        inverse: download_i32(gpu, &b.inverse, total_slots),
    }
}

/// The single-workgroup `moe_scatter_fused_k8` launched exactly as the
/// pre-multi-workgroup launcher did.
#[allow(clippy::too_many_arguments)]
fn incumbent_scatter(
    gpu: &mut Gpu,
    d_topk: &GpuTensor,
    b: &ScatterBufs,
    total_slots: usize,
    e: usize,
    m_total_max: usize,
    block_m: usize,
) {
    gpu.ensure_kernel_public(
        "moe_scatter_fused_k8",
        kernels::MOE_SCATTER_FUSED_K8_SRC,
        "moe_scatter_fused_k8",
    )
    .expect("load incumbent scatter");
    let mut k = KernargBlob::new();
    k.push_ptr(d_topk.buf.as_ptr());
    k.push_ptr(b.counts.buf.as_ptr());
    k.push_ptr(b.offsets.buf.as_ptr());
    k.push_ptr(b.sorted.buf.as_ptr());
    k.push_ptr(b.tiles.buf.as_ptr());
    k.push_ptr(b.inverse.buf.as_ptr());
    k.push_i32(total_slots as i32);
    k.push_i32(e as i32);
    k.push_i32(m_total_max as i32);
    k.push_i32(block_m as i32);
    k.pad_to(16);
    gpu.launch_kernel_blob(
        "moe_scatter_fused_k8",
        [1, 1, 1],
        [256, 1, 1],
        (e * 4) as u32,
        k.as_mut_slice(),
    )
    .expect("launch incumbent scatter");
}

/// Whether the production launcher takes the multi-workgroup stable twin
/// (mirrors `Gpu::moe_scatter_stable_groups`).
fn stable_expected(arch: &str, total_slots: usize, e: usize) -> bool {
    if arch != "gfx1201" || e == 0 || e > 1024 {
        return false;
    }
    let groups = total_slots.div_ceil(1024).min(128);
    groups >= 4 && groups * e <= total_slots
}

/// Structural conformance of one scatter output to the CPU reference.
fn check_scatter_out(
    label: &str,
    out: &ScatterOut,
    topk: &[i32],
    cpu: &CpuRef,
    e: usize,
    block_m: usize,
    m_total_max: usize,
) -> usize {
    let mut fails = 0;
    fails += compare_vec(&format!("{label} counts"), &cpu.padded_counts, &out.counts);
    fails += compare_vec(&format!("{label} offsets"), &cpu.offsets, &out.offsets);
    let mut want_tiles = cpu.expert_tile_ids.clone();
    want_tiles.resize(m_total_max / block_m, -1);
    fails += compare_vec(&format!("{label} tile_ids"), &want_tiles, &out.tiles);
    // Tail after m_total is -1; every bucket is [real entries][-1 padding].
    if out.sorted[cpu.m_total..].iter().any(|&x| x != -1) {
        eprintln!("{label}: sorted tail past m_total is not -1");
        fails += 1;
    }
    for x in 0..e {
        let (s, en) = (cpu.offsets[x] as usize, cpu.offsets[x + 1] as usize);
        let raw = cpu.raw_counts[x] as usize;
        let mut got: Vec<i32> = out.sorted[s..s + raw].to_vec();
        let mut want: Vec<i32> = cpu.sorted_slot_index[s..s + raw].to_vec();
        got.sort_unstable();
        want.sort_unstable();
        if got != want || out.sorted[s + raw..en].iter().any(|&v| v != -1) {
            if fails < 3 {
                eprintln!("  {label}: expert {x} bucket set / padding mismatch");
            }
            fails += 1;
        }
    }
    // inverse_perm: routed -> its sorted position, unrouted -> -1.
    let mut bad = 0usize;
    for (i, &id) in topk.iter().enumerate() {
        let want_inv = if id >= 0 && (id as usize) < e {
            let pos = out.inverse[i];
            if pos < 0 || out.sorted[pos as usize] != i as i32 {
                bad += 1;
            }
            continue;
        } else {
            -1
        };
        if out.inverse[i] != want_inv {
            bad += 1;
        }
    }
    if bad > 0 {
        eprintln!("{label}: inverse_perm FAIL ({bad})");
        fails += 1;
    } else {
        println!("{label}: inverse_perm consistent ({} slots)", topk.len());
    }
    fails
}

/// `moe_down_combine_grouped_k8` over grouped rows that are a pure function of
/// the flat slot (row `p` holds the row of slot `sorted[p]`; padding rows hold
/// a poison the combine must never read). Returns the output bit patterns.
#[allow(clippy::too_many_arguments)]
fn combine_bits(
    gpu: &mut Gpu,
    out: &ScatterOut,
    d_inverse: &GpuTensor,
    weights: &[f32],
    x0: &[f32],
    n: usize,
    k_top: usize,
    dim: usize,
    m_total: usize,
) -> Vec<u32> {
    let mut y = vec![f32::from_bits(0x7fc0_dead); m_total * dim];
    for p in 0..m_total {
        let s = out.sorted[p];
        if s >= 0 {
            for d in 0..dim {
                // Pure function of (slot, column).
                let h = (s as u32).wrapping_mul(2_654_435_761) ^ (d as u32).wrapping_mul(40_503);
                y[p * dim + d] = (h % 2048) as f32 / 256.0 - 4.0;
            }
        }
    }
    let d_y = gpu.upload_f32(&y, &[m_total, dim]).expect("upload y");
    let d_w = gpu.upload_f32(weights, &[n, k_top]).expect("upload w");
    let d_x = gpu.upload_f32(x0, &[n, dim]).expect("upload x");
    gpu.moe_down_combine_grouped_k8(&d_y, d_inverse, &d_w, &d_x, dim, k_top, n)
        .expect("combine");
    gpu.hip.device_synchronize().expect("sync combine");
    let got = gpu.download_f32(&d_x).expect("download combine");
    for t in [d_y, d_w, d_x] {
        gpu.free_tensor(t).expect("free");
    }
    got.iter().map(|v| v.to_bits()).collect()
}

/// `topk` (length `n * k_top`) through the incumbent and the production
/// launcher.
#[allow(clippy::too_many_arguments)]
fn run_fused_case(
    gpu: &mut Gpu,
    label: &str,
    topk: &[i32],
    n: usize,
    k_top: usize,
    e: usize,
    block_m: usize,
    slack: usize,
) -> usize {
    let total_slots = n * k_top;
    let m_total_max = total_slots + e * slack;
    let cpu = cpu_scatter(topk, e, block_m);
    assert!(cpu.m_total <= m_total_max);
    println!(
        "\n=== {label}: N={n} K_TOP={k_top} E={e} BLOCK_M={block_m} slots={total_slots} \
         m_total={} m_total_max={m_total_max} ===",
        cpu.m_total
    );
    let d_topk = upload_i32(gpu, topk);

    let bi = poisoned_bufs(gpu, e, total_slots, m_total_max, block_m);
    incumbent_scatter(gpu, &d_topk, &bi, total_slots, e, m_total_max, block_m);
    let inc = download_out(gpu, &bi, e, total_slots, m_total_max, block_m);

    let bp = poisoned_bufs(gpu, e, total_slots, m_total_max, block_m);
    gpu.moe_scatter_fused_k8(
        &d_topk,
        &bp.counts,
        &bp.offsets,
        &bp.sorted,
        &bp.tiles,
        &bp.inverse,
        total_slots,
        e,
        m_total_max,
        block_m,
    )
    .expect("production scatter");
    let prod = download_out(gpu, &bp, e, total_slots, m_total_max, block_m);

    let stable = stable_expected(&gpu.arch, total_slots, e);
    println!(
        "[{label}] production path: {}",
        if stable {
            "multi-workgroup stable"
        } else {
            "single-workgroup"
        }
    );
    let mut fails = 0;
    fails += check_scatter_out(
        &format!("[{label}] incumbent"),
        &inc,
        topk,
        &cpu,
        e,
        block_m,
        m_total_max,
    );
    fails += check_scatter_out(
        &format!("[{label}] production"),
        &prod,
        topk,
        &cpu,
        e,
        block_m,
        m_total_max,
    );
    // Tables independent of the order are byte-identical to the incumbent's.
    for (name, a, b) in [
        ("counts", &inc.counts, &prod.counts),
        ("offsets", &inc.offsets, &prod.offsets),
        ("tile_ids", &inc.tiles, &prod.tiles),
    ] {
        if a != b {
            eprintln!("[{label}] production {name} differs from the incumbent");
            fails += 1;
        }
    }
    if stable {
        // Ascending flat slot within every expert: the CPU input-order scatter.
        let mut want = cpu.sorted_slot_index.clone();
        want.resize(m_total_max, -1);
        fails += compare_vec(
            &format!("[{label}] production sorted == stable CPU order"),
            &want,
            &prod.sorted,
        );
    }

    // Consumer: the combine over slot-pure grouped rows is byte-identical.
    let dim = 64;
    let mut rng = 0x9e37_79b9u32 ^ (n as u32);
    let mut next = || {
        rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((rng >> 8) & 0xffff) as f32 / 65536.0
    };
    let weights: Vec<f32> = (0..total_slots).map(|_| next()).collect();
    let x0: Vec<f32> = (0..n * dim).map(|_| next() - 0.5).collect();
    let inc_inv = upload_i32(gpu, &inc.inverse);
    let prod_inv = upload_i32(gpu, &prod.inverse);
    let a = combine_bits(
        gpu,
        &inc,
        &inc_inv,
        &weights,
        &x0,
        n,
        k_top,
        dim,
        cpu.m_total,
    );
    let b = combine_bits(
        gpu,
        &prod,
        &prod_inv,
        &weights,
        &x0,
        n,
        k_top,
        dim,
        cpu.m_total,
    );
    if a == b {
        println!(
            "[{label}] combine output byte-identical incumbent vs production ({} words)",
            a.len()
        );
    } else {
        let at = a.iter().zip(&b).position(|(x, y)| x != y).unwrap();
        eprintln!(
            "[{label}] combine output differs at word {at}: {:#010x} vs {:#010x}",
            a[at], b[at]
        );
        fails += 1;
    }
    for t in [
        d_topk, inc_inv, prod_inv, bi.counts, bi.offsets, bi.sorted, bi.tiles, bi.inverse,
        bp.counts, bp.offsets, bp.sorted, bp.tiles, bp.inverse,
    ] {
        gpu.free_tensor(t).expect("free");
    }
    println!("[{label}] fused total fails: {fails}");
    fails
}

/// Routing drawn from a pool of `pool` experts only (skew), optionally with
/// every `bad_every`-th slot replaced by an unrouted id.
fn gen_skewed(
    n: usize,
    k_top: usize,
    pool: usize,
    seed: u32,
    bad_every: usize,
    e: usize,
) -> Vec<i32> {
    let mut t = gen_topk_indices(n, k_top, pool, seed);
    if bad_every > 0 {
        const BAD: [i32; 4] = [-1, i32::MAX, -7, 0];
        for (i, v) in t.iter_mut().enumerate() {
            if i % bad_every == bad_every - 1 {
                let pick = BAD[(i / bad_every) % BAD.len()];
                *v = if pick == 0 { e as i32 } else { pick };
            }
        }
    }
    t
}

fn fused_cases(gpu: &mut Gpu) -> usize {
    let (k, e, bm) = (8usize, 256usize, 16usize);
    let mut fails = 0;
    // Small shapes: single-workgroup fallback.
    fails += run_fused_case(
        gpu,
        "fused-toy",
        &gen_topk_indices(4, 2, 4, 0xc0ffee),
        4,
        2,
        4,
        2,
        2,
    );
    fails += run_fused_case(
        gpu,
        "fused-n100",
        &gen_topk_indices(100, k, e, 7),
        100,
        k,
        e,
        bm,
        bm,
    );
    // Ragged sizes around the four-chunk threshold and a non-multiple-of-256 tail.
    for n in [511usize, 513, 600, 1001, 2049] {
        let t = gen_topk_indices(n, k, e, 0x1000 + n as u32);
        fails += run_fused_case(gpu, &format!("fused-n{n}"), &t, n, k, e, bm, bm - 1);
    }
    // A3B prefill chunks.
    for n in [7808usize, 8192] {
        let t = gen_topk_indices(n, k, e, 0xa3b0 + n as u32);
        fails += run_fused_case(gpu, &format!("fused-a3b-{n}"), &t, n, k, e, bm, bm);
        let t = gen_topk_indices(n, k, e, 0xa3b1 + n as u32);
        fails += run_fused_case(
            gpu,
            &format!("fused-a3b-{n}-slack15"),
            &t,
            n,
            k,
            e,
            bm,
            bm - 1,
        );
    }
    // Skew: every token routed into 12 experts (huge buckets, 244 empty).
    let t = gen_skewed(8192, k, 12, 0x5eed, 0, e);
    fails += run_fused_case(gpu, "fused-a3b-8192-skew12", &t, 8192, k, e, bm, bm);
    // Unrouted ids sprinkled through the batch (-1, huge, negative, E).
    let t = gen_skewed(8192, k, e, 0xbad, 97, e);
    fails += run_fused_case(gpu, "fused-a3b-8192-unrouted", &t, 8192, k, e, bm, bm);
    let t = gen_skewed(4100, k, e, 0xbad2, 5, e);
    fails += run_fused_case(gpu, "fused-a3b-4100-unrouted-dense", &t, 4100, k, e, bm, bm);
    // Other expert counts the shared launcher serves.
    let t = gen_topk_indices(4096, 4, 128, 0x128);
    fails += run_fused_case(gpu, "fused-e128-k4", &t, 4096, 4, 128, bm, bm);
    let t = gen_topk_indices(2048, 10, 512, 0x512);
    fails += run_fused_case(gpu, "fused-e512-k10", &t, 2048, 10, 512, bm, bm);
    fails
}

/// Event-timed median over 50 reps (10 warmups) of the incumbent
/// single-workgroup kernel and the production launcher on one A3B chunk.
fn time_fused(gpu: &mut Gpu, n: usize) {
    let (k_top, e, block_m) = (8usize, 256usize, 16usize);
    let total_slots = n * k_top;
    let m_total_max = total_slots + e * block_m;
    let topk = gen_topk_indices(n, k_top, e, 0x7131 + n as u32);
    let d_topk = upload_i32(gpu, &topk);
    let b = poisoned_bufs(gpu, e, total_slots, m_total_max, block_m);
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    let mut medians = [0.0f64; 2];
    for (arm, median) in medians.iter_mut().enumerate() {
        let mut samples = Vec::new();
        for rep in 0..60 {
            gpu.hip.device_synchronize().expect("sync");
            gpu.hip
                .event_record(&start, gpu.active_stream.as_ref())
                .expect("record start");
            if arm == 0 {
                incumbent_scatter(gpu, &d_topk, &b, total_slots, e, m_total_max, block_m);
            } else {
                gpu.moe_scatter_fused_k8(
                    &d_topk,
                    &b.counts,
                    &b.offsets,
                    &b.sorted,
                    &b.tiles,
                    &b.inverse,
                    total_slots,
                    e,
                    m_total_max,
                    block_m,
                )
                .expect("production scatter");
            }
            gpu.hip
                .event_record(&stop, gpu.active_stream.as_ref())
                .expect("record stop");
            gpu.hip.event_synchronize(&stop).expect("sync stop");
            if rep >= 10 {
                samples.push(
                    f64::from(gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed")) * 1e3,
                );
            }
        }
        samples.sort_by(f64::total_cmp);
        *median = (samples[24] + samples[25]) / 2.0;
    }
    println!(
        "{{\"timing\":true,\"kernel\":\"moe_scatter_fused_k8\",\"rows\":{n},\"slots\":{total_slots},\
         \"median_us\":{{\"incumbent_single_workgroup\":{:.3},\"production\":{:.3}}},\"reps\":50}}",
        medians[0], medians[1]
    );
    gpu.hip.event_destroy(start).expect("destroy");
    gpu.hip.event_destroy(stop).expect("destroy");
    for t in [d_topk, b.counts, b.offsets, b.sorted, b.tiles, b.inverse] {
        gpu.free_tensor(t).expect("free");
    }
}

/// Median event time (us, 50 reps) of each of the three launches of the
/// multi-workgroup scatter, launched directly (same arguments as the
/// production launcher).
fn time_stable_stages(gpu: &mut Gpu, n: usize) {
    let (k_top, e, block_m) = (8usize, 256usize, 16usize);
    let total_slots = n * k_top;
    let m_total_max = total_slots + e * block_m;
    let groups = total_slots.div_ceil(1024).min(128);
    let topk = gen_topk_indices(n, k_top, e, 0x7131 + n as u32);
    let d_topk = upload_i32(gpu, &topk);
    let b = poisoned_bufs(gpu, e, total_slots, m_total_max, block_m);
    for sym in [
        "moe_scatter_stable_hist_k8",
        "moe_scatter_stable_place_k8",
        "moe_scatter_stable_inverse_k8",
    ] {
        gpu.ensure_kernel_public(
            "moe_scatter_stable_k8",
            kernels::MOE_SCATTER_STABLE_K8_SRC,
            sym,
        )
        .expect("load stable scatter");
    }
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    let mut out = [0.0f64; 3];
    for (stage, slot) in out.iter_mut().enumerate() {
        let mut samples = Vec::new();
        for rep in 0..60 {
            let mut k = KernargBlob::new();
            let (name, grid) = match stage {
                0 => {
                    k.push_ptr(d_topk.buf.as_ptr());
                    k.push_ptr(b.inverse.buf.as_ptr());
                    k.push_ptr(b.sorted.buf.as_ptr());
                    k.push_ptr(b.tiles.buf.as_ptr());
                    k.push_i32(total_slots as i32);
                    k.push_i32(e as i32);
                    k.push_i32(m_total_max as i32);
                    k.push_i32(block_m as i32);
                    ("moe_scatter_stable_hist_k8", groups)
                }
                1 => {
                    k.push_ptr(d_topk.buf.as_ptr());
                    k.push_ptr(b.inverse.buf.as_ptr());
                    k.push_ptr(b.counts.buf.as_ptr());
                    k.push_ptr(b.offsets.buf.as_ptr());
                    k.push_ptr(b.sorted.buf.as_ptr());
                    k.push_ptr(b.tiles.buf.as_ptr());
                    k.push_i32(total_slots as i32);
                    k.push_i32(e as i32);
                    k.push_i32(block_m as i32);
                    ("moe_scatter_stable_place_k8", groups)
                }
                _ => {
                    k.push_ptr(d_topk.buf.as_ptr());
                    k.push_ptr(b.sorted.buf.as_ptr());
                    k.push_ptr(b.inverse.buf.as_ptr());
                    k.push_i32(total_slots as i32);
                    k.push_i32(e as i32);
                    k.push_i32(m_total_max as i32);
                    (
                        "moe_scatter_stable_inverse_k8",
                        total_slots.max(m_total_max).div_ceil(4 * 256),
                    )
                }
            };
            k.pad_to(16);
            gpu.hip.device_synchronize().expect("sync");
            gpu.hip
                .event_record(&start, gpu.active_stream.as_ref())
                .expect("record start");
            gpu.launch_kernel_blob(name, [grid as u32, 1, 1], [256, 1, 1], 0, k.as_mut_slice())
                .expect("launch stage");
            gpu.hip
                .event_record(&stop, gpu.active_stream.as_ref())
                .expect("record stop");
            gpu.hip.event_synchronize(&stop).expect("sync stop");
            if rep >= 10 {
                samples.push(
                    f64::from(gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed")) * 1e3,
                );
            }
        }
        samples.sort_by(f64::total_cmp);
        *slot = (samples[24] + samples[25]) / 2.0;
    }
    println!(
        "{{\"timing\":true,\"kernel\":\"moe_scatter_stable_k8 stages\",\"rows\":{n},\"groups\":{groups},\
         \"median_us\":{{\"hist\":{:.3},\"place\":{:.3},\"inverse\":{:.3}}},\"reps\":50}}",
        out[0], out[1], out[2]
    );
    gpu.hip.event_destroy(start).expect("destroy");
    gpu.hip.event_destroy(stop).expect("destroy");
    for t in [d_topk, b.counts, b.offsets, b.sorted, b.tiles, b.inverse] {
        gpu.free_tensor(t).expect("free");
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);

    let mut total_fails = 0usize;

    // Toy case — hand-checkable.
    total_fails += run_case(&mut gpu, "toy", 4, 2, 4, 2, 0xc0ffee);

    // Mid case — sweep both small batch and ragged padding.
    total_fails += run_case(&mut gpu, "mid", 16, 4, 8, 4, 0xdead_beef);

    // A3B-shape — the production target.
    total_fails += run_case(&mut gpu, "a3b", 256, 8, 256, 16, 0x1234_5678);

    // Production launcher vs the incumbent single-workgroup kernel.
    total_fails += fused_cases(&mut gpu);
    for n in [7808usize, 8192] {
        time_fused(&mut gpu, n);
        time_stable_stages(&mut gpu, n);
    }

    if total_fails == 0 {
        println!("\nALL CASES PASS");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL total={total_fails}");
        std::process::exit(1);
    }
}
