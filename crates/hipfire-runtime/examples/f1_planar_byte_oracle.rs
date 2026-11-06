// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DESIGN F1 on-card byte oracle: drop-zero planar Flash-Next MQ4 routed experts
//! (`DType::MQ4G256V2P` / `MQ4G128V2P`) versus the shipped layout (`MQ4G256V2` qt44 gate/up,
//! `MQ4G128V2` qt53 down), bit for bit.
//!
//! For each layer the REAL shipped expert tensors of the `.mq4-xts` artifact are read, `--experts`
//! of them (evenly spread over the layer's experts) are uploaded twice, exactly as the production
//! weight store does it: one stacked tensor of shipped experts (`[n, rows, k]`, shipped dtype) and
//! one stacked tensor of the host-repacked planar experts (`mq4_planar::repack_tensor`, planar
//! dtype), each with a device `[n] u64` expert pointer table (`base + i * per_expert_bytes`).
//! Every planar-twinned launcher then runs on identical synthetic routing and F32 activations
//! twice, shipped buffers under `Gpu::set_mq4_experts_planar(false)` and planar buffers under
//! `set_mq4_experts_planar(true)`, outputs pre-filled with the `0xa5` sentinel, and every output
//! word (u32, NaN payloads and unwritten sentinel words included) must be identical.
//!
//! Launchers covered (each through its public `Gpu` entry point, the one production calls):
//!   * `gate_up_decode_top10`   `gemv_mq4g256v2_moe_gate_up_top10_indexed_batched`
//!                              (k=2560 rows8 path; y_gate and y_up compared)
//!   * `down_decode_top10`      `gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded`
//!   * `gate_up_grouped`        `gemm_mq4g256v2_moe_grouped_top10` as the production prefill calls
//!                              it: <= 8 rows -> O4xR2 x4, 9..511 -> O4xR8 x4, >= 512 -> F16 WMMA
//!   * `gate_up_grouped_wmma`   `gemm_mq4g256v2_moe_grouped_wmma_k2` (gfx12 F16 WMMA) at every row
//!                              count, so WMMA is covered below the 512-token dispatch threshold
//!   * `down_grouped`           `gemm_mq4g128v2_moe_grouped_top10` (gfx1201 -> O8R16 / planar O8R16)
//!
//! NOT covered (no planar twin: the launcher refuses while the device is planar, by design):
//! `gemv_mq4g256v2_moe_gate_up_k8_indexed_batched`, the legacy F32 SIMT grouped gate/up and the
//! qwen4 IU4 symmetric PM bundles (those need the A4/A8 activation producers of the qwen4 forward).
//!
//! Output: one `PASS|FAIL layer L kernel K rows R bytes N ...` line per case, a final JSON line;
//! exit 1 on any mismatch or launcher error. A device that is not gfx1201 prints `SKIP` per
//! kernel and exits 2 (nothing was compared).
//!
//! Build:
//!   cargo build --release -p hipfire-runtime --example f1_planar_byte_oracle
//! Run (card, shipped artifact):
//!   target/release/examples/f1_planar_byte_oracle --model /path/qwen3.8-flash-next-base.mq4-xts \
//!     [--layers 0,17,47] [--experts 24] [--device N] [--decode-rows 1,2,4,8] \
//!     [--prefill-rows 1,2,8,16,64,257,512] [--seed S]

use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::mq4_planar::{
    planar_bytes, repack_tensor, shipped_bytes, Mq4PlanarKind, FLASH_NEXT_DOWN, FLASH_NEXT_GATE_UP,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;
use std::path::Path;

type Result<T, E = String> = std::result::Result<T, E>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const SCHEMA: &str = "f1-planar-byte-oracle-v1";
const SENTINEL_BYTE: i32 = 0xa5;
const SENTINEL_WORD: u32 = 0xa5a5_a5a5;
const K_TOP: usize = 10;
const TILE: usize = 16;

const GATE_M: usize = FLASH_NEXT_GATE_UP.0;
const GATE_K: usize = FLASH_NEXT_GATE_UP.1;
const DOWN_M: usize = FLASH_NEXT_DOWN.0;
const DOWN_K: usize = FLASH_NEXT_DOWN.1;

const KERNELS: [&str; 5] = [
    "gate_up_decode_top10",
    "down_decode_top10",
    "gate_up_grouped",
    "gate_up_grouped_wmma",
    "down_grouped",
];

// ---------------------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------------------

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

fn name_seed(base: u64, parts: &[&str], nums: &[usize]) -> u64 {
    let mut h = base ^ 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    };
    for p in parts {
        p.bytes().for_each(&mut eat);
        eat(0xff);
    }
    for n in nums {
        n.to_le_bytes().into_iter().for_each(&mut eat);
    }
    h
}

/// Deterministic synthetic activations: roughly normal, with exact `+0.0`, `-0.0` and large
/// magnitudes sprinkled in so the contraction sees the cases a rotated hidden state can produce.
fn gen_activations(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = rng.unit() + rng.unit() + rng.unit() - 1.5;
            match i % 509 {
                7 => 0.0,
                31 => -0.0,
                101 => 64.0 * (rng.unit() - 0.5),
                _ => v,
            }
        })
        .collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Ten distinct local expert indices, skewed toward low indices so some experts collect many
/// slots (several 16-slot tiles) and others one.
fn pick_top10(rng: &mut Rng, n: usize) -> [usize; K_TOP] {
    let mut out = [usize::MAX; K_TOP];
    let mut count = 0;
    let mut tries = 0;
    while count < K_TOP {
        tries += 1;
        let u = rng.unit();
        let idx = if tries > 256 {
            (0..n).find(|i| !out[..count].contains(i)).unwrap()
        } else {
            (((u * u) * n as f32) as usize).min(n - 1)
        };
        if !out[..count].contains(&idx) {
            out[count] = idx;
            count += 1;
        }
    }
    out
}

fn route(rows: usize, n: usize, rng: &mut Rng) -> Vec<usize> {
    (0..rows).flat_map(|_| pick_top10(rng, n)).collect()
}

/// The grouped-prefill layout of a flat `[rows * 10]` routing: per expert (ascending local
/// index) its flat slots in ascending order, padded to 16-slot tiles with `-1`; one expert id
/// per tile. Returns `(expert_tile_ids, sorted_slot_index)`; `grouped_rows = sorted.len()`.
fn group_slots(topk: &[usize], n: usize) -> (Vec<i32>, Vec<i32>) {
    let mut per: Vec<Vec<i32>> = vec![Vec::new(); n];
    for (flat, &e) in topk.iter().enumerate() {
        per[e].push(flat as i32);
    }
    let mut tiles = Vec::new();
    let mut sorted = Vec::new();
    for (e, slots) in per.iter().enumerate() {
        for chunk in slots.chunks(TILE) {
            tiles.push(e as i32);
            sorted.extend_from_slice(chunk);
            sorted.extend(std::iter::repeat(-1).take(TILE - chunk.len()));
        }
    }
    (tiles, sorted)
}

/// `count` expert ids evenly spread over `0..total` (strictly increasing).
fn spread_ids(total: usize, count: usize) -> Vec<usize> {
    (0..count).map(|i| i * total / count).collect()
}

// ---------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------

struct Cfg {
    model: String,
    layers: Vec<usize>,
    experts: usize,
    device: i32,
    decode_rows: Vec<usize>,
    prefill_rows: Vec<usize>,
    seed: u64,
}

const USAGE: &str = "usage: f1_planar_byte_oracle --model PATH [--layers 0,17,47] [--experts 24] \
[--device N] [--decode-rows 1,2,4,8] [--prefill-rows 1,2,8,16,64,257,512] [--seed S]";

fn parse_list(flag: &str, s: &str) -> Result<Vec<usize>> {
    let v: Result<Vec<usize>> = s
        .split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .map_err(|e| format!("{flag} '{p}': {e}"))
        })
        .collect();
    let v = v?;
    if v.is_empty() || v.contains(&0) {
        return Err(format!(
            "{flag}: need a non-empty list of positive integers"
        ));
    }
    Ok(v)
}

fn parse_args(argv: &[String]) -> Result<Cfg> {
    let mut cfg = Cfg {
        model: String::new(),
        layers: vec![0, 17, 47],
        experts: 24,
        device: 0,
        decode_rows: vec![1, 2, 4, 8],
        prefill_rows: vec![1, 2, 8, 16, 64, 257, 512],
        seed: 0xf1_0ac1e,
    };
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let mut val = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--model" => cfg.model = val("--model")?,
            "--layers" => {
                let s = val("--layers")?;
                cfg.layers = s
                    .split(',')
                    .map(|p| {
                        p.trim()
                            .parse::<usize>()
                            .map_err(|e| format!("--layers '{p}': {e}"))
                    })
                    .collect::<Result<_>>()?;
            }
            "--experts" => cfg.experts = val("--experts")?.parse().map_err(err)?,
            "--device" => cfg.device = val("--device")?.parse().map_err(err)?,
            "--decode-rows" => {
                cfg.decode_rows = parse_list("--decode-rows", &val("--decode-rows")?)?
            }
            "--prefill-rows" => {
                cfg.prefill_rows = parse_list("--prefill-rows", &val("--prefill-rows")?)?
            }
            "--seed" => cfg.seed = val("--seed")?.parse().map_err(err)?,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if cfg.model.is_empty() {
        return Err(format!("--model is required\n{USAGE}"));
    }
    if cfg.layers.is_empty() {
        return Err("--layers: empty".to_string());
    }
    if cfg.experts < K_TOP {
        return Err(format!(
            "--experts must be >= {K_TOP} (top-{K_TOP} routing)"
        ));
    }
    Ok(cfg)
}

// ---------------------------------------------------------------------------------------
// Device expert stacks (production layout) and the paired launch harness
// ---------------------------------------------------------------------------------------

/// One stacked expert tensor plus its `[n] u64` pointer table.
struct Stack {
    data: GpuTensor,
    table: GpuTensor,
}

struct Pair {
    shipped: Stack,
    planar: Stack,
}

impl Pair {
    fn table(&self, planar: bool) -> &GpuTensor {
        if planar {
            &self.planar.table
        } else {
            &self.shipped.table
        }
    }
}

/// Upload `bytes` (`n` experts of `per` bytes each, back to back) as a `[n, rows, k]` tensor of
/// `dtype` and build the pointer table the launchers take (`base + i * per`).
fn upload_stack(
    gpu: &mut Gpu,
    bytes: &[u8],
    n: usize,
    per: usize,
    rows: usize,
    k: usize,
    dtype: DType,
) -> Result<Stack> {
    if bytes.len() != n * per {
        return Err(format!("stack of {} bytes != {n} x {per}", bytes.len()));
    }
    let mut data = gpu.upload_raw(bytes, &[n, rows, k]).map_err(err)?;
    data.dtype = dtype;
    let base = data.buf.as_ptr() as usize as u64;
    let table_bytes: Vec<u8> = (0..n)
        .flat_map(|i| (base + (i * per) as u64).to_le_bytes())
        .collect();
    let table = gpu.upload_raw(&table_bytes, &[n]).map_err(err)?;
    Ok(Stack { data, table })
}

fn free_stack(gpu: &mut Gpu, s: Stack) {
    let _ = gpu.free_tensor(s.data);
    let _ = gpu.free_tensor(s.table);
}

struct LayerStacks {
    gate: Pair,
    down: Pair,
    n: usize,
}

/// Read `n_sel` experts of one projection from the real stacked tensor and build both layouts.
fn load_projection(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    layer: usize,
    suffix: &str,
    kind: Mq4PlanarKind,
    (rows, k): (usize, usize),
    ids: &[usize],
) -> Result<Pair> {
    let name = format!("model.language_model.layers.{layer}.mlp.experts.{suffix}");
    let (_, bytes) = hfq
        .tensor_data(&name)
        .ok_or_else(|| format!("tensor {name} not in artifact"))?;
    let per_ship = shipped_bytes(kind, rows, k).map_err(err)?;
    let per_plan = planar_bytes(kind, rows, k).map_err(err)?;
    if bytes.len() % per_ship != 0 {
        return Err(format!(
            "{name}: {} bytes is not a multiple of one shipped expert ({per_ship})",
            bytes.len()
        ));
    }
    let total = bytes.len() / per_ship;
    if let Some(&bad) = ids.iter().find(|&&e| e >= total) {
        return Err(format!("{name}: expert {bad} >= {total} stacked experts"));
    }
    let mut ship = Vec::with_capacity(ids.len() * per_ship);
    let mut plan = vec![0u8; ids.len() * per_plan];
    for (slot, &e) in ids.iter().enumerate() {
        let src = &bytes[e * per_ship..(e + 1) * per_ship];
        ship.extend_from_slice(src);
        repack_tensor(
            kind,
            rows,
            k,
            src,
            &mut plan[slot * per_plan..(slot + 1) * per_plan],
        )
        .map_err(|error| format!("{name}[{e}]: {error}"))?;
    }
    let (ship_dtype, plan_dtype) = match kind {
        Mq4PlanarKind::GateUp => (DType::MQ4G256V2, DType::MQ4G256V2P),
        Mq4PlanarKind::Down => (DType::MQ4G128V2, DType::MQ4G128V2P),
    };
    let shipped = upload_stack(gpu, &ship, ids.len(), per_ship, rows, k, ship_dtype)?;
    let planar = upload_stack(gpu, &plan, ids.len(), per_plan, rows, k, plan_dtype)?;
    Ok(Pair { shipped, planar })
}

struct Outcome {
    pass: bool,
    bytes: usize,
    detail: String,
}

/// Launch closure: `(gpu, planar, outputs)`; the closure picks the matching pointer table.
type Launch<'a> = &'a dyn Fn(&mut Gpu, bool, &[GpuTensor]) -> HipResult<()>;

/// Run `launch` for the shipped layout (flag off) and the planar layout (flag on) into
/// sentinel-filled F32 outputs of `out_words[i]` words, and compare every output word.
fn run_pair(gpu: &mut Gpu, out_words: &[usize], launch: Launch) -> Result<Outcome> {
    let mut results: Vec<Vec<Vec<u8>>> = Vec::new();
    for planar in [false, true] {
        gpu.set_mq4_experts_planar(planar);
        let run = (|| -> Result<Vec<Vec<u8>>> {
            let mut outs = Vec::new();
            for &n in out_words {
                let t = gpu.alloc_tensor(&[n], DType::F32).map_err(err)?;
                gpu.hip.memset(&t.buf, SENTINEL_BYTE, n * 4).map_err(err)?;
                outs.push(t);
            }
            let launched = launch(gpu, planar, &outs).map_err(err);
            let synced = gpu.hip.device_synchronize().map_err(err);
            let mut host = Vec::new();
            let mut read_err = None;
            for (t, &n) in outs.iter().zip(out_words) {
                let mut b = vec![0u8; n * 4];
                match gpu.hip.memcpy_dtoh(&mut b, &t.buf) {
                    Ok(()) => host.push(b),
                    Err(e) => {
                        read_err = Some(err(e));
                        break;
                    }
                }
            }
            for t in outs {
                let _ = gpu.free_tensor(t);
            }
            launched?;
            synced?;
            if let Some(e) = read_err {
                return Err(e);
            }
            Ok(host)
        })();
        gpu.set_mq4_experts_planar(false);
        let layout = if planar { "planar" } else { "shipped" };
        results.push(run.map_err(|e| format!("{layout} launch: {e}"))?);
    }
    let (ship, plan) = (&results[0], &results[1]);
    let bytes: usize = out_words.iter().map(|n| n * 4).sum();
    let mut mismatches = 0usize;
    let mut first: Option<(usize, usize, u32, u32)> = None;
    let mut nonzero = 0usize;
    for (oi, (a, b)) in ship.iter().zip(plan).enumerate() {
        for (wi, (wa, wb)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
            let x = u32::from_le_bytes(wa.try_into().unwrap());
            let y = u32::from_le_bytes(wb.try_into().unwrap());
            if x != 0 && x != SENTINEL_WORD {
                nonzero += 1;
            }
            if x != y {
                mismatches += 1;
                first.get_or_insert((oi, wi, x, y));
            }
        }
    }
    let mut detail = format!("nonzero_words={nonzero}");
    let mut pass = mismatches == 0;
    if let Some((oi, wi, x, y)) = first {
        detail.push_str(&format!(
            " mismatches={mismatches} first=out{oi}[{wi}] shipped=0x{x:08x} planar=0x{y:08x}"
        ));
    }
    if nonzero == 0 {
        pass = false;
        detail.push_str(" VACUOUS(shipped output holds no non-zero word)");
    }
    Ok(Outcome {
        pass,
        bytes,
        detail,
    })
}

// ---------------------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------------------

fn up_i32(gpu: &mut Gpu, v: &[i32]) -> Result<GpuTensor> {
    gpu.upload_raw(&i32_bytes(v), &[v.len()]).map_err(err)
}

fn up_f32(gpu: &mut Gpu, v: &[f32]) -> Result<GpuTensor> {
    gpu.upload_f32(v, &[v.len()]).map_err(err)
}

fn free_all(gpu: &mut Gpu, ts: Vec<GpuTensor>) {
    for t in ts {
        let _ = gpu.free_tensor(t);
    }
}

fn case_gate_up_decode(gpu: &mut Gpu, st: &LayerStacks, rows: usize, seed: u64) -> Result<Outcome> {
    let mut rng = Rng::new(seed);
    let topk: Vec<i32> = route(rows, st.n, &mut rng)
        .iter()
        .map(|&e| e as i32)
        .collect();
    let x = gen_activations(rows * GATE_K, &mut rng);
    let topk_t = up_i32(gpu, &topk)?;
    let x_t = up_f32(gpu, &x)?;
    let half = rows * K_TOP * (GATE_M / 2);
    let out = run_pair(gpu, &[half, half], &|g, planar, o| {
        g.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
            st.gate.table(planar),
            &topk_t,
            &x_t,
            &o[0],
            &o[1],
            GATE_M,
            GATE_K,
            rows,
        )
    });
    free_all(gpu, vec![topk_t, x_t]);
    out
}

fn case_down_decode(gpu: &mut Gpu, st: &LayerStacks, rows: usize, seed: u64) -> Result<Outcome> {
    let mut rng = Rng::new(seed);
    let topk: Vec<i32> = route(rows, st.n, &mut rng)
        .iter()
        .map(|&e| e as i32)
        .collect();
    let hidden = gen_activations(rows * K_TOP * DOWN_K, &mut rng);
    let topk_t = up_i32(gpu, &topk)?;
    let hidden_t = up_f32(gpu, &hidden)?;
    let words = rows * K_TOP * DOWN_M;
    let n = st.n;
    let out = run_pair(gpu, &[words], &|g, planar, o| {
        g.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            st.down.table(planar),
            &topk_t,
            &hidden_t,
            &o[0],
            DOWN_M,
            DOWN_K,
            rows,
            n,
            K_TOP,
        )
    });
    free_all(gpu, vec![topk_t, hidden_t]);
    out
}

#[derive(Clone, Copy)]
enum GateUpGrouped {
    Dispatch,
    Wmma,
}

fn case_gate_up_grouped(
    gpu: &mut Gpu,
    st: &LayerStacks,
    rows: usize,
    seed: u64,
    which: GateUpGrouped,
) -> Result<Outcome> {
    let mut rng = Rng::new(seed);
    let topk = route(rows, st.n, &mut rng);
    let (tiles, sorted) = group_slots(&topk, st.n);
    let grouped_rows = sorted.len();
    let x = gen_activations(rows * GATE_K, &mut rng);
    let tiles_t = up_i32(gpu, &tiles)?;
    let sorted_t = up_i32(gpu, &sorted)?;
    let x_t = up_f32(gpu, &x)?;
    let out = run_pair(gpu, &[grouped_rows * GATE_M], &|g, planar, o| match which {
        GateUpGrouped::Dispatch => g.gemm_mq4g256v2_moe_grouped_top10(
            st.gate.table(planar),
            &tiles_t,
            &sorted_t,
            &x_t,
            &o[0],
            GATE_M,
            GATE_K,
            K_TOP,
            grouped_rows,
            rows,
        ),
        GateUpGrouped::Wmma => g.gemm_mq4g256v2_moe_grouped_wmma_k2(
            st.gate.table(planar),
            &tiles_t,
            &sorted_t,
            &x_t,
            &o[0],
            GATE_M,
            GATE_K,
            K_TOP,
            grouped_rows,
            rows,
        ),
    });
    free_all(gpu, vec![tiles_t, sorted_t, x_t]);
    out
}

fn case_down_grouped(gpu: &mut Gpu, st: &LayerStacks, rows: usize, seed: u64) -> Result<Outcome> {
    let mut rng = Rng::new(seed);
    let topk = route(rows, st.n, &mut rng);
    let (tiles, sorted) = group_slots(&topk, st.n);
    let grouped_rows = sorted.len();
    let total_slots = rows * K_TOP;
    // One rotated activation row per top-10 slot (`x_row_div == 1`).
    let x = gen_activations(total_slots * DOWN_K, &mut rng);
    let tiles_t = up_i32(gpu, &tiles)?;
    let sorted_t = up_i32(gpu, &sorted)?;
    let x_t = up_f32(gpu, &x)?;
    let n = st.n;
    let out = run_pair(gpu, &[grouped_rows * DOWN_M], &|g, planar, o| {
        g.gemm_mq4g128v2_moe_grouped_top10(
            st.down.table(planar),
            &tiles_t,
            &sorted_t,
            &x_t,
            &o[0],
            DOWN_M,
            DOWN_K,
            1,
            grouped_rows,
            total_slots,
            n,
        )
    });
    free_all(gpu, vec![tiles_t, sorted_t, x_t]);
    out
}

// ---------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------

#[derive(Default)]
struct Tally {
    pass: usize,
    fail: usize,
    skip: usize,
}

fn report(
    t: &mut Tally,
    results: &mut Vec<serde_json::Value>,
    layer: usize,
    kernel: &str,
    rows: usize,
    outcome: Result<Outcome>,
) {
    let (verdict, bytes, detail) = match outcome {
        Ok(o) => (if o.pass { "PASS" } else { "FAIL" }, o.bytes, o.detail),
        Err(e) => ("FAIL", 0, format!("error={}", e.replace('\n', " | "))),
    };
    if verdict == "PASS" {
        t.pass += 1;
    } else {
        t.fail += 1;
    }
    println!("{verdict} layer {layer} kernel {kernel} rows {rows} bytes {bytes} {detail}");
    results.push(json!({
        "verdict": verdict, "layer": layer, "kernel": kernel, "rows": rows,
        "bytes": bytes, "detail": detail,
    }));
}

fn run() -> Result<i32> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cfg = parse_args(&argv)?;
    let mut gpu = Gpu::init_with_device(cfg.device).map_err(err)?;
    let arch = gpu.arch.clone();
    let mut tally = Tally::default();
    let mut results: Vec<serde_json::Value> = Vec::new();

    if arch != "gfx1201" {
        for kernel in KERNELS {
            println!(
                "SKIP layer - kernel {kernel} rows - bytes 0 reason=device arch is {arch}, the planar twins are gfx1201 only"
            );
            tally.skip += 1;
        }
        println!(
            "{}",
            json!({
                "schema": SCHEMA, "arch": arch, "model": cfg.model, "verdict": "SKIP",
                "cases": 0, "pass": 0, "fail": 0, "skipped": tally.skip,
            })
        );
        return Ok(2);
    }

    let hfq =
        HfqFile::open(Path::new(&cfg.model)).map_err(|e| format!("open {}: {e}", cfg.model))?;
    let ids = spread_ids(512, cfg.experts.min(512));

    for &layer in &cfg.layers {
        let loaded = (|| -> Result<LayerStacks> {
            let gate = load_projection(
                &mut gpu,
                &hfq,
                layer,
                "gate_up_proj",
                Mq4PlanarKind::GateUp,
                (GATE_M, GATE_K),
                &ids,
            )?;
            let down = load_projection(
                &mut gpu,
                &hfq,
                layer,
                "down_proj",
                Mq4PlanarKind::Down,
                (DOWN_M, DOWN_K),
                &ids,
            )?;
            Ok(LayerStacks {
                gate,
                down,
                n: ids.len(),
            })
        })();
        let st = match loaded {
            Ok(st) => st,
            Err(e) => {
                report(&mut tally, &mut results, layer, "setup", 0, Err(e));
                continue;
            }
        };
        let seed = |kernel: &str, rows: usize| name_seed(cfg.seed, &[kernel], &[layer, rows]);

        for &rows in &cfg.decode_rows {
            let r = case_gate_up_decode(&mut gpu, &st, rows, seed("gate_up_decode_top10", rows));
            report(
                &mut tally,
                &mut results,
                layer,
                "gate_up_decode_top10",
                rows,
                r,
            );
            let r = case_down_decode(&mut gpu, &st, rows, seed("down_decode_top10", rows));
            report(
                &mut tally,
                &mut results,
                layer,
                "down_decode_top10",
                rows,
                r,
            );
        }
        for &rows in &cfg.prefill_rows {
            let r = case_gate_up_grouped(
                &mut gpu,
                &st,
                rows,
                seed("gate_up_grouped", rows),
                GateUpGrouped::Dispatch,
            );
            report(&mut tally, &mut results, layer, "gate_up_grouped", rows, r);
            let r = case_gate_up_grouped(
                &mut gpu,
                &st,
                rows,
                seed("gate_up_grouped_wmma", rows),
                GateUpGrouped::Wmma,
            );
            report(
                &mut tally,
                &mut results,
                layer,
                "gate_up_grouped_wmma",
                rows,
                r,
            );
            let r = case_down_grouped(&mut gpu, &st, rows, seed("down_grouped", rows));
            report(&mut tally, &mut results, layer, "down_grouped", rows, r);
        }

        free_stack(&mut gpu, st.gate.shipped);
        free_stack(&mut gpu, st.gate.planar);
        free_stack(&mut gpu, st.down.shipped);
        free_stack(&mut gpu, st.down.planar);
    }

    let verdict = if tally.fail == 0 && tally.pass > 0 {
        "PASS"
    } else {
        "FAIL"
    };
    println!(
        "{}",
        json!({
            "schema": SCHEMA, "arch": arch, "model": cfg.model, "layers": cfg.layers,
            "experts": ids.len(), "decode_rows": cfg.decode_rows,
            "prefill_rows": cfg.prefill_rows, "seed": cfg.seed,
            "cases": tally.pass + tally.fail, "pass": tally.pass, "fail": tally.fail,
            "skipped": tally.skip, "verdict": verdict,
        })
    );
    Ok(if verdict == "PASS" { 0 } else { 1 })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("f1_planar_byte_oracle: {e}");
            println!(
                "{}",
                json!({"schema": SCHEMA, "verdict": "FAIL", "error": e})
            );
            std::process::exit(1);
        }
    }
}
