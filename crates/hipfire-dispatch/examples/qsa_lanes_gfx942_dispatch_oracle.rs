// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dispatch oracle of the gfx942 batched QSA lanes Core (`pipeline/qsa.rs`,
//! `HIPFIRE_QWEN4_QSA_LANES=1` on exact gfx942) against the per-lane Core it replaces.
//!
//! Both arms are driven through the real dispatch entry points `validate_steps` /
//! `execute_validated_steps` with `DispatchCtx::new(&gpu)`:
//!
//! * reference: one step list per lane holding that lane's `Step::IndexedAttention` Core op (a
//!   group of one never groups, the flag is off): decode prologue, incremental pool, score,
//!   select-from-scores, attention = five launches per lane;
//! * candidate: ONE step list of all N lanes' Core ops with the lanes flag on: the grouper of
//!   `steps.rs` hands them to `execute_indexed_attention_core_lanes`, which builds each lane's
//!   launch arguments as the per-lane interpreter does and runs the five launches ONCE over all
//!   lanes (`qsa_core_lanes_gfx942` + `Gfx942Device::qwen4_qsa_lanes_core`).
//!
//! Each lane owns separate allocations of its state (K / V caches, raw / pooled index keys,
//! persistent selection) and its row scratches; the candidate lanes SHARE one selected scratch
//! (each lane its own rows of it, as the interpreter's group partitions it) while the reference
//! lanes reuse its first rows one after another. Per step every lane gets fresh random
//! projected rows and its own position; the output scratch is sentinel-filled first; after
//! each step every lane tensor of both arms is compared as bytes (a sentinel left behind
//! fails). The launch count of each arm is checked (`hip_bridge::memory_effects`: reference 5N,
//! candidate 5) so a silent per-lane fallback cannot pass, and negative controls MUST be
//! detected: the candidate with two lanes' state tensors swapped and a candidate lane at the
//! wrong position.
//!
//! `cargo run --release -p hipfire-dispatch --example qsa_lanes_gfx942_dispatch_oracle --
//!    [--lanes 2,8,33,64] [--formats f32,fnuz] [--steps 3] [--seed U64]`
//!
//! Prints one line per case / control and a final `PASS|FAIL qsa_lanes_gfx942_dispatch_oracle`.
//! Exit 0 = PASS, 1 = a mismatch / undetected control / wrong launch count, 2 = not exact gfx942.

use hip_bridge::memory_effects;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_validated_steps, validate_steps, IndexedAttentionMode, IndexedAttentionOp,
    IndexedAttentionState, MixerPhase, Step,
};
use rdna_compute::rope::{Rope, RopeScaling, Yarn};
use rdna_compute::tensor_ops::QsaKvFormat;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::Arc;

const NH: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
const IDX_HEADS: usize = 4;
const IDX_DIM: usize = 128;
const COMPRESS: usize = 4;
const BUDGET: usize = 2048;
const CAP: usize = BUDGET + COMPRESS - 1;
const IDXW: usize = (IDX_HEADS + 1) * IDX_DIM;
const QGW: usize = 2 * NH * HD;
const KVW: usize = NKV * HD;
const OUTW: usize = NH * HD;
const HIDDEN: usize = 2560;
const SENT: u32 = 0x7fc0_dead;
const SENT_PERSIST: u32 = 0x7e7e_7e7d;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn sym(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}

fn bf16_exact(x: f32) -> f32 {
    f32::from_bits((bf16_bits(x) as u32) << 16)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn upload_typed(gpu: &mut Gpu, bytes: &[u8], elements: usize, dtype: DType) -> GpuTensor {
    let t = gpu.alloc_tensor(&[elements], dtype).expect("alloc");
    assert_eq!(t.byte_size(), bytes.len(), "upload size");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
    t
}

fn bf16_tensor(gpu: &mut Gpu, rng: &mut Rng, n: usize, amp: f32) -> GpuTensor {
    let bytes: Vec<u8> = (0..n)
        .flat_map(|_| bf16_bits(rng.sym() * amp).to_le_bytes())
        .collect();
    upload_typed(gpu, &bytes, n, DType::BF16)
}

fn f32_tensor(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    gpu.upload_f32(v, &[v.len()]).expect("upload")
}

fn htod(gpu: &Gpu, t: &GpuTensor, v: &[f32]) {
    assert_eq!(t.numel(), v.len());
    gpu.bind_thread().expect("bind");
    gpu.hip.memcpy_htod(&t.buf, &f32_bytes(v)).expect("htod");
}

fn diff_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fmt {
    F32,
    Fnuz,
}

impl Fmt {
    fn kv(self) -> QsaKvFormat {
        match self {
            Fmt::F32 => QsaKvFormat::F32,
            Fmt::Fnuz => QsaKvFormat::Fp8Fnuz,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Fmt::F32 => "f32",
            Fmt::Fnuz => "fnuz",
        }
    }
}

struct Geo {
    fmt: Fmt,
    max_seq: usize,
    pooled_cap: usize,
}

/// One lane's tensors in one arm.
struct LaneBufs {
    full_keys: GpuTensor,
    full_values: GpuTensor,
    raw: GpuTensor,
    pooled: GpuTensor,
    persistent: GpuTensor,
    index: GpuTensor,
    qgate: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    out: GpuTensor,
}

struct HostLane {
    k: Vec<u8>,
    v: Vec<u8>,
    raw: Vec<u8>,
    pooled: Vec<u8>,
}

fn f32_rand(rng: &mut Rng, n: usize, bf16: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        let v = rng.sym();
        out.extend_from_slice(&(if bf16 { bf16_exact(v) } else { v }).to_le_bytes());
    }
    out
}

fn host_lane(g: &Geo, rng: &mut Rng, ctx: usize) -> HostLane {
    let blocks = ctx / COMPRESS;
    let mut raw = f32_rand(rng, ctx * IDX_DIM, true);
    raw.resize(g.max_seq * IDX_DIM * 4, 0);
    let mut pooled = f32_rand(rng, blocks * IDX_DIM, true);
    pooled.resize(g.pooled_cap * IDX_DIM * 4, 0);
    let cache = |rng: &mut Rng| match g.fmt {
        Fmt::F32 => {
            let mut c = f32_rand(rng, ctx * KVW, false);
            c.resize(g.max_seq * KVW * 4, 0);
            c
        }
        Fmt::Fnuz => {
            let row_bytes = g.fmt.kv().kv_row_bytes(NKV, HD);
            let mut c = vec![0u8; g.max_seq * row_bytes];
            for pos in 0..ctx {
                let row = &mut c[pos * row_bytes..(pos + 1) * row_bytes];
                let (codes, scales) = row.split_at_mut(NKV * HD);
                for chunk in codes.chunks_mut(8) {
                    let w = rng.next().to_le_bytes();
                    for (b, r) in chunk.iter_mut().zip(w) {
                        *b = if r == 0x80 { 0 } else { r };
                    }
                }
                for h in 0..NKV {
                    let exp = 8 + (rng.next() % 5) as u16;
                    let mant = (rng.next() & 0x3ff) as u16;
                    scales[h * 2..h * 2 + 2].copy_from_slice(&((exp << 10) | mant).to_le_bytes());
                }
            }
            c
        }
    };
    HostLane {
        k: cache(rng),
        v: cache(rng),
        raw,
        pooled,
    }
}

fn lane_bufs(gpu: &mut Gpu, g: &Geo, h: &HostLane) -> LaneBufs {
    let (kv_elems, dtype) = match g.fmt {
        Fmt::F32 => (g.max_seq * KVW, DType::F32),
        Fmt::Fnuz => (g.max_seq * g.fmt.kv().kv_row_bytes(NKV, HD), DType::Raw),
    };
    let sent = |n: usize| vec![f32::from_bits(SENT); n];
    LaneBufs {
        full_keys: upload_typed(gpu, &h.k, kv_elems, dtype),
        full_values: upload_typed(gpu, &h.v, kv_elems, dtype),
        raw: upload_typed(gpu, &h.raw, g.max_seq * IDX_DIM, DType::F32),
        pooled: upload_typed(gpu, &h.pooled, g.pooled_cap * IDX_DIM, DType::F32),
        persistent: upload_typed(
            gpu,
            &(0..CAP)
                .flat_map(|_| SENT_PERSIST.to_le_bytes())
                .collect::<Vec<_>>(),
            CAP * 4,
            DType::Raw,
        ),
        index: f32_tensor(gpu, &sent(IDXW)),
        qgate: f32_tensor(gpu, &sent(QGW)),
        k: f32_tensor(gpu, &sent(KVW)),
        v: f32_tensor(gpu, &sent(KVW)),
        out: f32_tensor(gpu, &sent(OUTW)),
    }
}

impl LaneBufs {
    fn compared(&self) -> [(&'static str, &GpuTensor); 10] {
        [
            ("full keys", &self.full_keys),
            ("full values", &self.full_values),
            ("raw index keys", &self.raw),
            ("pooled keys", &self.pooled),
            ("persistent selection", &self.persistent),
            ("index scratch", &self.index),
            ("qgate scratch", &self.qgate),
            ("k scratch", &self.k),
            ("v scratch", &self.v),
            ("qsa output", &self.out),
        ]
    }

    fn free(self, gpu: &mut Gpu) {
        for t in [
            self.full_keys,
            self.full_values,
            self.raw,
            self.pooled,
            self.persistent,
            self.index,
            self.qgate,
            self.k,
            self.v,
            self.out,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

struct Shared {
    index_q_norm: GpuTensor,
    index_k_norm: GpuTensor,
    q_norm: GpuTensor,
    k_norm: GpuTensor,
    weight: GpuTensor,
    bf16_scratch: GpuTensor,
    rotation: GpuTensor,
    input: GpuTensor,
    attention_output: GpuTensor,
}

fn weight<'a>(w: &'a GpuTensor, m: usize, k: usize) -> WeightRef<'a> {
    WeightRef {
        buf: w,
        dtype: DType::BF16,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

fn rope() -> Rope {
    Rope {
        theta: 10_000_000.0,
        scaling: RopeScaling::Yarn(Yarn::new(4.0, 8192, None, None, None, None)),
    }
}

#[allow(clippy::too_many_arguments)]
fn core_op<'a>(
    g: &Geo,
    s: &'a Shared,
    l: &'a LaneBufs,
    selected: &'a GpuTensor,
    position: usize,
) -> IndexedAttentionOp<'a> {
    IndexedAttentionOp {
        indexer_qk: weight(&s.weight, IDXW, HIDDEN),
        indexer_q_norm: &s.index_q_norm,
        indexer_k_norm: &s.index_k_norm,
        q: weight(&s.weight, QGW, HIDDEN),
        k: weight(&s.weight, KVW, HIDDEN),
        v: weight(&s.weight, KVW, HIDDEN),
        q_norm: &s.q_norm,
        k_norm: &s.k_norm,
        output: weight(&s.weight, HIDDEN, OUTW),
        state: IndexedAttentionState {
            format: g.fmt.kv(),
            full_keys: &l.full_keys,
            full_values: &l.full_values,
            raw_index_keys: &l.raw,
            pooled_keys: &l.pooled,
            selected_indices: &l.persistent,
            f16_arena: None,
            pooled_bf16: None,
            full_capacity: g.max_seq,
            raw_capacity: g.max_seq,
            pooled_capacity: g.pooled_cap,
            selected_capacity: CAP,
            position_capacity: g.max_seq,
            full_len: position,
            raw_len: position,
            pooled_len: position / COMPRESS,
            selected_len: position.min(CAP),
            position,
        },
        input: &s.input,
        index_scratch: &l.index,
        qgate_scratch: &l.qgate,
        k_scratch: &l.k,
        v_scratch: &l.v,
        qsa_output: &l.out,
        selected_scratch: selected,
        attention_output: &s.attention_output,
        bf16_scratch: &s.bf16_scratch,
        rows: 1,
        index_heads: IDX_HEADS,
        index_kv_heads: 1,
        index_dim: IDX_DIM,
        budget: BUDGET,
        compress: COMPRESS,
        heads: NH,
        kv_heads: NKV,
        head_dim: HD,
        rotary_dim: 64,
        rope: rope(),
        input_width: HIDDEN,
        rotation: &s.rotation,
        mode: IndexedAttentionMode::Full,
        trunk_a4: 0,
        phase: MixerPhase::Core,
        sp_select: None,
    }
}

fn run(gpu: &mut Gpu, ctx: &DispatchCtx, steps: &[Step<'_>]) -> Result<(), String> {
    validate_steps(gpu, steps).map_err(|e| format!("validate: {e:?}"))?;
    execute_validated_steps(gpu, ctx, steps).map_err(|e| format!("execute: {e:?}"))
}

#[derive(Clone, Copy, PartialEq)]
enum Control {
    None,
    SwappedStates,
    WrongPosition,
}

const MIXED: [usize; 12] = [2048, 64, 1000, 65, 255, 2047, 513, 4093, 63, 3, 7, 1999];

fn run_case(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    sh: &Shared,
    fmt: Fmt,
    n: usize,
    steps: usize,
    seed: u64,
) -> bool {
    let ctxs: Vec<usize> = (0..n).map(|l| MIXED[(l * 5) % MIXED.len()]).collect();
    let max_seq = (ctxs.iter().max().unwrap() + steps + 8).next_multiple_of(COMPRESS);
    let g = Geo {
        fmt,
        max_seq,
        pooled_cap: max_seq / COMPRESS,
    };
    let mut ok = true;
    for control in [
        Control::None,
        Control::SwappedStates,
        Control::WrongPosition,
    ] {
        if control != Control::None && n < 2 {
            continue;
        }
        let mut rng = Rng(seed ^ ((n as u64) << 32) ^ ((fmt == Fmt::Fnuz) as u64));
        let host: Vec<HostLane> = ctxs.iter().map(|&c| host_lane(&g, &mut rng, c)).collect();
        let refr: Vec<LaneBufs> = host.iter().map(|h| lane_bufs(gpu, &g, h)).collect();
        let cand: Vec<LaneBufs> = host.iter().map(|h| lane_bufs(gpu, &g, h)).collect();
        drop(host);
        // Selected scratches: the reference lanes reuse one lane's rows; the candidate lanes
        // share one tensor with `n` rows (each lane its own).
        let sel_ref = upload_typed(gpu, &vec![0x7fu8; CAP * 4], CAP * 4, DType::Raw);
        let sel_cand_whole = upload_typed(gpu, &vec![0x7fu8; n * CAP * 4], n * CAP * 4, DType::Raw);
        let mut pos = ctxs.clone();
        let (mut total_diff, mut launch_ok, mut sentinel_left) = (0usize, true, 0usize);
        let mut detail = Vec::new();
        for step in 0..steps {
            for l in 0..n {
                let rows = [
                    (&refr[l].index, &cand[l].index, IDXW),
                    (&refr[l].qgate, &cand[l].qgate, QGW),
                    (&refr[l].k, &cand[l].k, KVW),
                    (&refr[l].v, &cand[l].v, KVW),
                ];
                for (r, c, w) in rows {
                    let data: Vec<f32> = (0..w).map(|_| rng.sym()).collect();
                    htod(gpu, r, &data);
                    htod(gpu, c, &data);
                }
                for arm in [&refr[l], &cand[l]] {
                    htod(gpu, &arm.out, &vec![f32::from_bits(SENT); OUTW]);
                }
            }
            // Reference: one list per lane, flag off.
            Arc::make_mut(&mut gpu.flags).qwen4_qsa_lanes = false;
            let m0 = memory_effects::snapshot();
            for l in 0..n {
                let list = [Step::IndexedAttention(core_op(
                    &g, sh, &refr[l], &sel_ref, pos[l],
                ))];
                run(gpu, ctx, &list).expect("reference lane");
            }
            let ref_launches = memory_effects::snapshot().since(m0).launches;
            // Candidate: one list, flag on.
            Arc::make_mut(&mut gpu.flags).qwen4_qsa_lanes = true;
            let mut cpos = pos.clone();
            if control == Control::WrongPosition {
                cpos[n - 1] += 1;
            }
            let list: Vec<Step<'_>> = (0..n)
                .map(|l| {
                    // Swap control: lane 0 runs on lane 1's state and vice versa.
                    let state_of = match (control, l) {
                        (Control::SwappedStates, 0) => 1,
                        (Control::SwappedStates, 1) => 0,
                        _ => l,
                    };
                    let mut op = core_op(&g, sh, &cand[l], &sel_cand_whole, cpos[l]);
                    let s = &cand[state_of];
                    op.state.full_keys = &s.full_keys;
                    op.state.full_values = &s.full_values;
                    op.state.raw_index_keys = &s.raw;
                    op.state.pooled_keys = &s.pooled;
                    op.state.selected_indices = &s.persistent;
                    Step::IndexedAttention(op)
                })
                .collect();
            let m1 = memory_effects::snapshot();
            run(gpu, ctx, &list).expect("candidate step list");
            let cand_launches = memory_effects::snapshot().since(m1).launches;
            drop(list);
            Arc::make_mut(&mut gpu.flags).qwen4_qsa_lanes = false;
            let launches_good = ref_launches == 5 * n as u64 && cand_launches == 5;
            if control == Control::None {
                launch_ok &= launches_good;
                detail.push(format!(
                    "step {step}: launches reference {ref_launches} (want {}) candidate {cand_launches} (want 5)",
                    5 * n
                ));
            }
            for l in 0..n {
                for ((name, r), (_, c)) in refr[l].compared().into_iter().zip(cand[l].compared()) {
                    let rb = gpu.download_raw_bytes(r).expect("dtoh");
                    let cb = gpu.download_raw_bytes(c).expect("dtoh");
                    let d = diff_bytes(&rb, &cb);
                    if d != 0 && control == Control::None {
                        detail.push(format!("  lane {l} {name}: {d} differing bytes"));
                    }
                    total_diff += d;
                    if name == "qsa output" {
                        sentinel_left += rb
                            .chunks_exact(4)
                            .chain(cb.chunks_exact(4))
                            .filter(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == SENT)
                            .count();
                    }
                    if name == "persistent selection" {
                        sentinel_left += rb
                            .chunks_exact(4)
                            .chain(cb.chunks_exact(4))
                            .filter(|w| {
                                u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == SENT_PERSIST
                            })
                            .count();
                    }
                }
            }
            for p in &mut pos {
                *p += 1;
            }
        }
        match control {
            Control::None => {
                let good = total_diff == 0 && launch_ok && sentinel_left == 0;
                ok &= good;
                println!(
                    "[{} lanes={n}] dispatch lanes route vs per-lane Core, {steps} steps: differing bytes {total_diff}, sentinel words left {sentinel_left}, launch counts {} => {}",
                    fmt.name(),
                    if launch_ok { "as expected" } else { "WRONG" },
                    if good { "BYTE-EQUAL" } else { "MISMATCH" }
                );
                for line in detail.iter().filter(|l| l.starts_with(' ') || !launch_ok) {
                    println!("{line}");
                }
            }
            Control::SwappedStates | Control::WrongPosition => {
                let name = if control == Control::SwappedStates {
                    "two lanes' states swapped"
                } else {
                    "last lane's position off by one"
                };
                let detected = total_diff > 0;
                ok &= detected;
                println!(
                    "[{} lanes={n}] control ({name}): {total_diff} differing bytes (must be detected) => {}",
                    fmt.name(),
                    if detected { "ok" } else { "UNDETECTED" }
                );
            }
        }
        for t in [sel_ref, sel_cand_whole] {
            let _ = gpu.free_tensor(t);
        }
        for l in refr.into_iter().chain(cand) {
            l.free(gpu);
        }
    }
    ok
}

fn main() {
    let mut lanes = vec![2usize, 8, 33, 64];
    let mut formats = vec![Fmt::F32, Fmt::Fnuz];
    let mut steps = 3usize;
    let mut seed = 0x6469_7370_5153_4100u64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--lanes" => lanes = v.split(',').map(|s| s.parse().expect("lanes")).collect(),
            "--formats" => {
                formats = v
                    .split(',')
                    .map(|f| match f {
                        "f32" => Fmt::F32,
                        "fnuz" => Fmt::Fnuz,
                        other => panic!("unknown format {other}"),
                    })
                    .collect()
            }
            "--steps" => steps = v.parse().expect("steps"),
            "--seed" => seed = v.parse().expect("seed"),
            other => panic!("unknown argument {other}"),
        }
    }
    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.try_gfx942().is_none() {
        eprintln!(
            "qsa_lanes_gfx942_dispatch_oracle: needs exact gfx942, this is {}",
            gpu.arch
        );
        std::process::exit(2);
    }
    let ctx = DispatchCtx::new(&gpu);
    let mut rng = Rng(seed);
    let sh = Shared {
        index_q_norm: bf16_tensor(&mut gpu, &mut rng, IDX_DIM, 0.3),
        index_k_norm: bf16_tensor(&mut gpu, &mut rng, IDX_DIM, 0.3),
        q_norm: bf16_tensor(&mut gpu, &mut rng, HD, 0.3),
        k_norm: bf16_tensor(&mut gpu, &mut rng, HD, 0.3),
        weight: bf16_tensor(&mut gpu, &mut rng, 16, 1.0),
        bf16_scratch: bf16_tensor(&mut gpu, &mut rng, OUTW, 1.0),
        rotation: f32_tensor(&mut gpu, &vec![0.0; OUTW]),
        input: f32_tensor(&mut gpu, &vec![0.0; HIDDEN]),
        attention_output: f32_tensor(&mut gpu, &vec![0.0; HIDDEN]),
    };
    let mut all = true;
    for &fmt in &formats {
        for &n in &lanes {
            all &= run_case(&mut gpu, &ctx, &sh, fmt, n, steps, seed);
        }
    }
    println!(
        "{} qsa_lanes_gfx942_dispatch_oracle",
        if all { "PASS" } else { "FAIL" }
    );
    std::process::exit(if all { 0 } else { 1 });
}
