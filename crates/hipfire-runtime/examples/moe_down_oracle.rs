// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S3 per-pick routed-DOWN byte oracle (and in-process A/B timer) on the REAL expert tensors of a
//! `.mq4-xts` artifact: the production `gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded`
//! (-> `..._k640_r4`, shipped or F1 planar twin) versus the `dni` kernel selected by
//! `HIPFIRE_QWEN4_MOE_DOWN_DNI` (`Gpu::set_moe_down_dni`), bit for bit.
//!
//! Per layer (the rank's owned experts uploaded in the shipped layout, or repacked planar with
//! `--planar`) and per case (rows x routing mode x seed): the same top-10 local ids (`-1` for picks
//! the rank does not own, exactly as the EP2 decode passes them) and F32 hidden rows run twice, flag
//! off and on, outputs pre-filled with the 0xa5 sentinel; EVERY output word (NaN payloads, the
//! zeros written for expert < 0, sentinel words) must be identical. Routing: `uniform`, `corr`
//! (W1-style correlated, `--corr`), `file` (`--routefile`, K0 compact verify routes). Activations
//! include exact +-0.0 and large magnitudes; `--stress` additionally makes a fraction of every
//! activation row huge/tiny. After the byte checks an interleaved A/B timing (events around one
//! launch, alternating arms) prints median us and GB/s of distinct expert bytes.
//!
//! Build: cargo build --release -p hipfire-runtime --features lab --example moe_down_oracle
//! Run:   moe_down_oracle --model PATH [--layers 0,17,47] [--rows 1,2,3,4,8] [--seeds 3]
//!        [--rank 0] [--ranks 2] [--corr 0.83] [--routefile F] [--planar] [--ab-iters 200]
//!
//! `--reps N` (determinism mode): each selected case is launched N times per arm, interleaved
//! (arms: `r4` = production k640_r4, `dni` = flag on through the production launcher, plus one arm
//! per `--variant PATH[?MACRO=VAL,..]`, a dni-source file loaded straight through
//! `ensure_kernel_public` with its entry names macro-renamed). Every launch gets a fresh 0xa5
//! sentinel fill. Reported per case: each arm against its OWN majority output (self pair: r4~r4,
//! dni~dni ..), each arm's majority against the r4 majority (cross pair), and rep-by-rep against r4
//! (paired), with the differing words (flat pick, rank, token, row) of every minority variant.
//! `--first-seed S`, `--modes uniform,corr` and `--chunk C` (launches buffered per sync) pick cases.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::mq4_planar::{
    planar_bytes, repack_tensor, shipped_bytes, Mq4PlanarKind, FLASH_NEXT_DOWN,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;
use std::path::Path;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const K_TOP: usize = 10;
const DOWN_M: usize = FLASH_NEXT_DOWN.0;
const DOWN_K: usize = FLASH_NEXT_DOWN.1;
const SENTINEL_BYTE: i32 = 0xa5;
const SENTINEL_WORD: u32 = 0xa5a5_a5a5;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn act(&mut self, stress: bool) -> f32 {
        match self.below(64) {
            0 => 0.0,
            1 => -0.0,
            2 => 40.0 * (self.unit() - 0.5),
            3 if stress => 3.0e18 * (self.unit() - 0.5),
            4 if stress => 1.0e-30 * (self.unit() - 0.5),
            _ => (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7,
        }
    }
}

struct Cfg {
    model: String,
    layers: Vec<usize>,
    rows: Vec<usize>,
    seeds: usize,
    rank: usize,
    ranks: usize,
    corr: f64,
    routefile: Option<String>,
    planar: bool,
    stress: bool,
    ab_iters: usize,
    device: i32,
    seed: u64,
    total_experts: usize,
    reps: usize,
    modes: Option<Vec<String>>,
    first_seed: usize,
    variants: Vec<String>,
    chunk: usize,
}

const USAGE: &str = "usage: moe_down_oracle --model PATH [--layers 0,17,47] [--rows 1,2,3,4,8] [--seeds 3] \
[--rank 0] [--ranks 2] [--corr 0.83] [--routefile F] [--planar] [--stress] [--ab-iters 200] [--device N] [--seed S] \
[--reps N] [--modes uniform,corr] [--first-seed S] [--variant PATH[?MACRO=VAL,..]]... [--chunk 64]";

fn parse_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|p| p.trim().parse::<usize>().map_err(err))
        .collect()
}

fn parse_args() -> Result<Cfg> {
    let mut c = Cfg {
        model: String::new(),
        layers: vec![0, 17, 47],
        rows: vec![1, 2, 3, 4, 8],
        seeds: 3,
        rank: 0,
        ranks: 2,
        corr: 0.83,
        routefile: None,
        planar: false,
        stress: false,
        ab_iters: 200,
        device: 0,
        seed: 0xd0,
        total_experts: 512,
        reps: 0,
        modes: None,
        first_seed: 0,
        variants: Vec::new(),
        chunk: 64,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut it = argv.iter();
    while let Some(f) = it.next() {
        let mut v = |n: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{n} needs a value\n{USAGE}"))
        };
        match f.as_str() {
            "--model" => c.model = v("--model")?,
            "--layers" => c.layers = parse_list(&v("--layers")?)?,
            "--rows" => c.rows = parse_list(&v("--rows")?)?,
            "--seeds" => c.seeds = v("--seeds")?.parse().map_err(err)?,
            "--rank" => c.rank = v("--rank")?.parse().map_err(err)?,
            "--ranks" => c.ranks = v("--ranks")?.parse().map_err(err)?,
            "--corr" => c.corr = v("--corr")?.parse().map_err(err)?,
            "--routefile" => c.routefile = Some(v("--routefile")?),
            "--planar" => c.planar = true,
            "--stress" => c.stress = true,
            "--ab-iters" => c.ab_iters = v("--ab-iters")?.parse().map_err(err)?,
            "--device" => c.device = v("--device")?.parse().map_err(err)?,
            "--seed" => c.seed = v("--seed")?.parse().map_err(err)?,
            "--reps" => c.reps = v("--reps")?.parse().map_err(err)?,
            "--modes" => {
                c.modes = Some(
                    v("--modes")?
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect(),
                )
            }
            "--first-seed" => c.first_seed = v("--first-seed")?.parse().map_err(err)?,
            "--variant" => c.variants.push(v("--variant")?),
            "--chunk" => c.chunk = v("--chunk")?.parse().map_err(err)?,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if c.model.is_empty() {
        return Err(format!("--model is required\n{USAGE}"));
    }
    Ok(c)
}

struct Stack {
    data: GpuTensor,
    table: GpuTensor,
    n: usize,
    per: usize,
}

fn load_down(gpu: &mut Gpu, hfq: &HfqFile, layer: usize, cfg: &Cfg) -> Result<Stack> {
    let name = format!("model.language_model.layers.{layer}.mlp.experts.down_proj");
    let (_, bytes) = hfq
        .tensor_data(&name)
        .ok_or_else(|| format!("tensor {name} not in artifact"))?;
    let per_ship = shipped_bytes(Mq4PlanarKind::Down, DOWN_M, DOWN_K).map_err(err)?;
    let per_plan = planar_bytes(Mq4PlanarKind::Down, DOWN_M, DOWN_K).map_err(err)?;
    let total = bytes.len() / per_ship;
    if total != cfg.total_experts {
        return Err(format!("{name}: {total} experts"));
    }
    let owned: Vec<usize> = (0..total).filter(|e| e % cfg.ranks == cfg.rank).collect();
    let per = if cfg.planar { per_plan } else { per_ship };
    let mut buf = vec![0u8; owned.len() * per];
    for (i, &e) in owned.iter().enumerate() {
        let src = &bytes[e * per_ship..(e + 1) * per_ship];
        if cfg.planar {
            repack_tensor(
                Mq4PlanarKind::Down,
                DOWN_M,
                DOWN_K,
                src,
                &mut buf[i * per..(i + 1) * per],
            )
            .map_err(|x| format!("{name}[{e}]: {x}"))?;
        } else {
            buf[i * per..(i + 1) * per].copy_from_slice(src);
        }
    }
    let mut data = gpu
        .upload_raw(&buf, &[owned.len(), DOWN_M, DOWN_K])
        .map_err(err)?;
    data.dtype = if cfg.planar {
        DType::MQ4G128V2P
    } else {
        DType::MQ4G128V2
    };
    let base = data.buf.as_ptr() as usize as u64;
    let tb: Vec<u8> = (0..owned.len())
        .flat_map(|i| (base + (i * per) as u64).to_le_bytes())
        .collect();
    let table = gpu.upload_raw(&tb, &[owned.len()]).map_err(err)?;
    Ok(Stack {
        data,
        table,
        n: owned.len(),
        per,
    })
}

enum Mode {
    Uniform,
    Corr,
    File,
}
impl Mode {
    fn name(&self) -> &'static str {
        match self {
            Mode::Uniform => "uniform",
            Mode::Corr => "corr",
            Mode::File => "file",
        }
    }
}

fn picks(
    mode: &Mode,
    rows: usize,
    rng: &mut Rng,
    cfg: &Cfg,
    file: &[(usize, Vec<i32>)],
) -> Vec<usize> {
    let total = cfg.total_experts;
    let distinct = |rng: &mut Rng, have: &[usize]| loop {
        let e = rng.below(total);
        if !have.contains(&e) {
            return e;
        }
    };
    match mode {
        Mode::Uniform => (0..rows)
            .flat_map(|_| {
                let mut p: Vec<usize> = Vec::new();
                while p.len() < K_TOP {
                    let e = distinct(rng, &p);
                    p.push(e);
                }
                p
            })
            .collect(),
        Mode::Corr => {
            let lanes = 8;
            let bases: Vec<Vec<usize>> = (0..rows.div_ceil(lanes))
                .map(|_| {
                    let mut b: Vec<usize> = Vec::new();
                    while b.len() < K_TOP {
                        let e = distinct(rng, &b);
                        b.push(e);
                    }
                    b
                })
                .collect();
            (0..rows)
                .flat_map(|r| {
                    let mut p: Vec<usize> = Vec::new();
                    for j in 0..K_TOP {
                        let mut e = bases[r / lanes][j];
                        if rng.unit() as f64 >= cfg.corr || p.contains(&e) {
                            e = distinct(rng, &p);
                        }
                        p.push(e);
                    }
                    p
                })
                .collect()
        }
        Mode::File => {
            let cand: Vec<&(usize, Vec<i32>)> = file.iter().filter(|(r, _)| *r <= rows).collect();
            if cand.is_empty() {
                return Vec::new();
            }
            let rec = cand[rng.below(cand.len())];
            rec.1.iter().map(|&e| e as usize).collect()
        }
    }
}

fn read_routefile(path: &str) -> Result<Vec<(usize, Vec<i32>)>> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let mut pos = 4;
    let mut out = Vec::new();
    for _ in 0..n {
        if pos + 4 > b.len() {
            break;
        }
        let rows = u16::from_le_bytes(b[pos + 2..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let cnt = rows * K_TOP;
        if pos + cnt * 4 > b.len() {
            break;
        }
        let ids: Vec<i32> = (0..cnt)
            .map(|i| i32::from_le_bytes(b[pos + i * 4..pos + i * 4 + 4].try_into().unwrap()))
            .collect();
        pos += cnt * 4;
        out.push((rows, ids));
    }
    Ok(out)
}

fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Dev {
    topk: GpuTensor,
    hidden: GpuTensor,
    out: GpuTensor,
    rows: usize,
    distinct: usize,
}

fn launch(gpu: &mut Gpu, st: &Stack, d: &Dev) -> Result<()> {
    gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
        &st.table, &d.topk, &d.hidden, &d.out, DOWN_M, DOWN_K, d.rows, st.n, K_TOP,
    )
    .map_err(err)
}

/// One launchable arm of the `--reps` mode.
enum ArmKind {
    /// production k640_r4 (flag off)
    Prod,
    /// dni through the production launcher (flag on)
    Dni,
    /// a dni source file loaded directly; the macro-renamed entry symbol
    Var(String),
}
struct ArmDef {
    name: String,
    kind: ArmKind,
}

fn build_arms(gpu: &mut Gpu, cfg: &Cfg) -> Result<Vec<ArmDef>> {
    let mut arms = vec![
        ArmDef {
            name: "r4".into(),
            kind: ArmKind::Prod,
        },
        ArmDef {
            name: "dni".into(),
            kind: ArmKind::Dni,
        },
    ];
    for (i, spec) in cfg.variants.iter().enumerate() {
        let (path, defs) = spec.split_once('?').unwrap_or((spec.as_str(), ""));
        let func = format!("dnivar{i}_{}", if cfg.planar { "p" } else { "s" });
        let mut src = format!(
            "#define gemv_mq4g128v2_moe_down_top10_dni_k640 dnivar{i}_s\n#define gemv_mq4g128v2p_moe_down_top10_dni_k640 dnivar{i}_p\n"
        );
        for def in defs.split(',').filter(|x| !x.is_empty()) {
            let (k, v) = def.split_once('=').unwrap_or((def, "1"));
            src.push_str(&format!("#define {k} {v}\n"));
        }
        src.push_str(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?);
        gpu.ensure_kernel_public(&format!("dnivar{i}"), &src, &func)
            .map_err(err)?;
        arms.push(ArmDef {
            name: format!("var{i}"),
            kind: ArmKind::Var(func),
        });
    }
    Ok(arms)
}

fn launch_arm(gpu: &mut Gpu, st: &Stack, d: &Dev, out: &GpuTensor, arm: &ArmDef) -> Result<()> {
    match &arm.kind {
        ArmKind::Prod | ArmKind::Dni => {
            gpu.set_moe_down_dni(matches!(arm.kind, ArmKind::Dni));
            gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
                &st.table, &d.topk, &d.hidden, out, DOWN_M, DOWN_K, d.rows, st.n, K_TOP,
            )
            .map_err(err)
        }
        ArmKind::Var(func) => {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(st.table.buf.as_ptr());
            b.push_ptr(d.topk.buf.as_ptr());
            b.push_ptr(d.hidden.buf.as_ptr());
            b.push_ptr(out.buf.as_ptr());
            b.push_i32(DOWN_M as i32);
            b.push_i32(DOWN_K as i32);
            b.push_i32(d.rows as i32);
            b.push_i32(st.n as i32);
            b.push_i32(K_TOP as i32);
            if b.len() != 52 {
                return Err(format!(
                    "dni kernarg blob is {} B, frozen ABI is 52",
                    b.len()
                ));
            }
            gpu.launch_kernel_blob(
                func,
                [(DOWN_M / 32) as u32, K_TOP as u32, d.rows as u32],
                [128, 1, 1],
                0,
                b.as_mut_slice(),
            )
            .map_err(err)
        }
    }
}

/// Builds one case (routing + activations + output buffer); `None` = empty route (file mode, no record that small).
fn make_case(
    gpu: &mut Gpu,
    cfg: &Cfg,
    mode: &Mode,
    file: &[(usize, Vec<i32>)],
    layer: usize,
    rows: usize,
    s: usize,
) -> Result<Option<Dev>> {
    let mut rng = Rng(cfg.seed ^ (layer as u64) << 40 ^ (rows as u64) << 20 ^ s as u64);
    let p = picks(mode, rows, &mut rng, cfg, file);
    if p.is_empty() {
        return Ok(None);
    }
    let nrows = p.len() / K_TOP;
    // local ids, -1 = not owned by this rank
    let topk: Vec<i32> = p
        .iter()
        .map(|&e| {
            if e % cfg.ranks == cfg.rank {
                (e / cfg.ranks) as i32
            } else {
                -1
            }
        })
        .collect();
    let mut seen = std::collections::BTreeSet::new();
    for &t in &topk {
        if t >= 0 {
            seen.insert(t);
        }
    }
    let hv: Vec<f32> = (0..nrows * K_TOP * DOWN_K)
        .map(|_| rng.act(cfg.stress))
        .collect();
    let topk_t = gpu.upload_raw(&i32b(&topk), &[topk.len()]).map_err(err)?;
    let hidden = gpu.upload_f32(&hv, &[hv.len()]).map_err(err)?;
    let words = nrows * K_TOP * DOWN_M;
    let out = gpu.alloc_tensor(&[words], DType::F32).map_err(err)?;
    Ok(Some(Dev {
        topk: topk_t,
        hidden,
        out,
        rows: nrows,
        distinct: seen.len(),
    }))
}

struct Variant {
    hash: u64,
    bytes: Vec<u8>,
    count: usize,
    first_rep: usize,
}

struct RepsOut {
    ok: bool,
    self_diff: Vec<usize>,
    cross: Vec<usize>,
    paired: Vec<usize>,
}

fn word_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
}

/// Number of differing words between `a` and `b` plus the first `cap` differing indices.
fn diff_words(a: &[u8], b: &[u8], cap: usize) -> (usize, Vec<usize>) {
    let (mut n, mut first) = (0usize, Vec::new());
    for i in 0..a.len() / 4 {
        if word_at(a, i) != word_at(b, i) {
            n += 1;
            if first.len() < cap {
                first.push(i);
            }
        }
    }
    (n, first)
}

fn where_word(i: usize) -> String {
    let (flat, col) = (i / DOWN_M, i % DOWN_M);
    format!(
        "w{i}(token {} rank {} row {col} wave-row0 {} lane-row {})",
        flat / K_TOP,
        flat % K_TOP,
        col / 8 * 8,
        col % 8
    )
}

#[allow(clippy::too_many_arguments)]
fn reps_case(
    gpu: &mut Gpu,
    st: &Stack,
    d: &Dev,
    cfg: &Cfg,
    arms: &[ArmDef],
    pool: &mut std::collections::HashMap<usize, Vec<GpuTensor>>,
    layer: usize,
    mode: &str,
    s: usize,
) -> Result<RepsOut> {
    use std::hash::{Hash, Hasher};
    let words = d.rows * K_TOP * DOWN_M;
    let n = arms.len();
    let chunk = cfg.chunk.clamp(1, cfg.reps);
    if !pool.contains_key(&words) {
        let mut v = Vec::with_capacity(n * chunk);
        for _ in 0..n * chunk {
            v.push(gpu.alloc_tensor(&[words], DType::F32).map_err(err)?);
        }
        pool.insert(words, v);
    }
    let slots = &pool[&words];
    let mut hist: Vec<Vec<Variant>> = (0..n).map(|_| Vec::new()).collect();
    let mut paired = vec![0usize; n];
    let mut paired_words = vec![0usize; n];
    let mut cur: Vec<Vec<u8>> = vec![vec![0u8; words * 4]; n];
    let mut done = 0;
    while done < cfg.reps {
        let b = chunk.min(cfg.reps - done);
        for j in 0..b {
            for a in 0..n {
                gpu.hip
                    .memset(&slots[a * chunk + j].buf, SENTINEL_BYTE, words * 4)
                    .map_err(err)?;
            }
        }
        for j in 0..b {
            for k in 0..n {
                let a = (k + done + j) % n; // rotate which arm goes first
                launch_arm(gpu, st, d, &slots[a * chunk + j], &arms[a])?;
            }
        }
        gpu.hip.device_synchronize().map_err(err)?;
        gpu.set_moe_down_dni(false);
        for j in 0..b {
            for a in 0..n {
                gpu.hip
                    .memcpy_dtoh(&mut cur[a], &slots[a * chunk + j].buf)
                    .map_err(err)?;
                let mut h = std::collections::hash_map::DefaultHasher::new();
                cur[a].hash(&mut h);
                let hash = h.finish();
                match hist[a].iter().position(|v| v.hash == hash) {
                    Some(i) => {
                        hist[a][i].count += 1;
                        if hist[a][i].bytes.is_empty() {
                            hist[a][i].bytes = cur[a].clone();
                        }
                    }
                    None => {
                        // keep the bytes of the first 8 distinct outputs only (a hazard-amplified arm differs every rep)
                        let bytes = if hist[a].len() < 8 {
                            cur[a].clone()
                        } else {
                            Vec::new()
                        };
                        hist[a].push(Variant {
                            hash,
                            bytes,
                            count: 1,
                            first_rep: done + j,
                        });
                    }
                }
            }
            for a in 1..n {
                if cur[a] != cur[0] {
                    paired[a] += 1;
                    paired_words[a] += diff_words(&cur[a], &cur[0], 0).0;
                }
            }
        }
        done += b;
    }
    let maj: Vec<usize> = hist
        .iter()
        .map(|h| {
            h.iter()
                .enumerate()
                .fold((0usize, 0usize), |best, (i, v)| {
                    if v.count > best.1 {
                        (i, v.count)
                    } else {
                        best
                    }
                })
                .0
        })
        .collect();
    let r4m = &hist[0][maj[0]].bytes;
    let nonzero = (0..words)
        .filter(|&i| word_at(r4m, i) != 0 && word_at(r4m, i) != SENTINEL_WORD)
        .count();
    let mut ok = nonzero > 0;
    let (mut self_diff, mut cross) = (vec![0usize; n], vec![0usize; n]);
    let mut line = String::new();
    let mut detail = String::new();
    for a in 0..n {
        let m = &hist[a][maj[a]];
        self_diff[a] = cfg.reps - m.count;
        let mut dw = 0usize;
        for v in hist[a]
            .iter()
            .filter(|v| v.hash != m.hash && !v.bytes.is_empty())
            .take(8)
        {
            let (nw, first) = diff_words(&v.bytes, &m.bytes, 6);
            dw += nw * v.count;
            detail.push_str(&format!(
                "\n   VARIANT {} count {} first_rep {} words_vs_own_majority {nw}:",
                arms[a].name, v.count, v.first_rep
            ));
            for i in first {
                detail.push_str(&format!(
                    " {} maj=0x{:08x} got=0x{:08x};",
                    where_word(i),
                    word_at(&m.bytes, i),
                    word_at(&v.bytes, i)
                ));
            }
        }
        line.push_str(&format!(
            " | {}: self_diff_reps={}/{} words={dw} variants={}",
            arms[a].name,
            self_diff[a],
            cfg.reps,
            hist[a].len()
        ));
        if self_diff[a] != 0 {
            ok = false;
        }
        if a > 0 {
            let (nw, first) = diff_words(&m.bytes, r4m, 6);
            cross[a] = nw;
            line.push_str(&format!(
                " cross_vs_r4_majority_words={nw} paired_vs_r4_diff_reps={} paired_words={}",
                paired[a], paired_words[a]
            ));
            if nw != 0 || paired[a] != 0 {
                ok = false;
            }
            for i in first {
                detail.push_str(&format!(
                    "\n   CROSS {} {} r4=0x{:08x} {}=0x{:08x}",
                    arms[a].name,
                    where_word(i),
                    word_at(r4m, i),
                    arms[a].name,
                    word_at(&m.bytes, i)
                ));
            }
        }
    }
    println!(
        "{} layer {layer} rows {} mode {mode} seed {s} planar {} reps {} distinct {} words {words} nonzero {nonzero}{}{line}{detail}",
        if ok { "REPS_PASS" } else { "REPS_FAIL" },
        d.rows,
        cfg.planar,
        cfg.reps,
        d.distinct,
        if nonzero == 0 { " VACUOUS" } else { "" },
    );
    Ok(RepsOut {
        ok,
        self_diff,
        cross,
        paired,
    })
}

fn run() -> Result<i32> {
    let cfg = parse_args()?;
    let mut gpu = Gpu::init_with_device(cfg.device).map_err(err)?;
    if gpu.arch != "gfx1201" {
        println!(
            "SKIP device arch is {}, the dni down is gfx1201 only",
            gpu.arch
        );
        return Ok(2);
    }
    gpu.set_mq4_experts_planar(cfg.planar);
    let hfq = HfqFile::open(Path::new(&cfg.model)).map_err(|e| format!("open: {e}"))?;
    let file = match &cfg.routefile {
        Some(p) => read_routefile(p)?,
        None => Vec::new(),
    };
    let mut modes = vec![Mode::Uniform, Mode::Corr];
    if !file.is_empty() {
        modes.push(Mode::File);
    }
    if let Some(f) = &cfg.modes {
        modes.retain(|m| f.iter().any(|x| x == m.name()));
    }
    let arms = if cfg.reps > 0 {
        build_arms(&mut gpu, &cfg)?
    } else {
        Vec::new()
    };
    let mut pool: std::collections::HashMap<usize, Vec<GpuTensor>> =
        std::collections::HashMap::new();
    let (mut tot_self, mut tot_cross, mut tot_paired) = (
        vec![0usize; arms.len()],
        vec![0usize; arms.len()],
        vec![0usize; arms.len()],
    );
    let ev0 = gpu.hip.event_create().map_err(err)?;
    let ev1 = gpu.hip.event_create().map_err(err)?;
    let (mut pass, mut fail) = (0usize, 0usize);

    for &layer in &cfg.layers {
        let st = load_down(&mut gpu, &hfq, layer, &cfg)?;
        for &rows in &cfg.rows {
            for mode in &modes {
                let mut ab: Vec<Dev> = Vec::new();
                for s in cfg.first_seed..cfg.first_seed + cfg.seeds {
                    let Some(d) = make_case(&mut gpu, &cfg, mode, &file, layer, rows, s)? else {
                        continue;
                    };
                    if cfg.reps > 0 {
                        let r = reps_case(
                            &mut gpu,
                            &st,
                            &d,
                            &cfg,
                            &arms,
                            &mut pool,
                            layer,
                            mode.name(),
                            s,
                        )?;
                        for a in 0..arms.len() {
                            tot_self[a] += r.self_diff[a];
                            tot_cross[a] += r.cross[a];
                            tot_paired[a] += r.paired[a];
                        }
                        if r.ok {
                            pass += 1;
                        } else {
                            fail += 1;
                        }
                        let _ = gpu.free_tensor(d.topk);
                        let _ = gpu.free_tensor(d.hidden);
                        let _ = gpu.free_tensor(d.out);
                        continue;
                    }
                    let nrows = d.rows;
                    let words = nrows * K_TOP * DOWN_M;
                    let mut outs: Vec<Vec<u8>> = Vec::new();
                    for on in [false, true] {
                        gpu.set_moe_down_dni(on);
                        gpu.hip
                            .memset(&d.out.buf, SENTINEL_BYTE, words * 4)
                            .map_err(err)?;
                        launch(&mut gpu, &st, &d)?;
                        gpu.hip.device_synchronize().map_err(err)?;
                        let mut b = vec![0u8; words * 4];
                        gpu.hip.memcpy_dtoh(&mut b, &d.out.buf).map_err(err)?;
                        outs.push(b);
                    }
                    gpu.set_moe_down_dni(false);
                    let (mut mism, mut nonzero) = (0usize, 0usize);
                    let mut first: Option<(usize, u32, u32)> = None;
                    for (i, (wa, wb)) in outs[0]
                        .chunks_exact(4)
                        .zip(outs[1].chunks_exact(4))
                        .enumerate()
                    {
                        let x0 = u32::from_le_bytes(wa.try_into().unwrap());
                        let y0 = u32::from_le_bytes(wb.try_into().unwrap());
                        if x0 != 0 && x0 != SENTINEL_WORD {
                            nonzero += 1;
                        }
                        if x0 != y0 {
                            mism += 1;
                            first.get_or_insert((i, x0, y0));
                        }
                    }
                    let ok = mism == 0 && nonzero > 0;
                    if ok {
                        pass += 1;
                    } else {
                        fail += 1;
                    }
                    println!(
                        "{} layer {layer} rows {nrows} mode {} seed {s} distinct {} words {words} nonzero {nonzero}{}",
                        if ok { "PASS" } else { "FAIL" },
                        mode.name(),
                        d.distinct,
                        match first {
                            Some((i, x0, y0)) => format!(" mismatches={mism} first=word{i} prod=0x{x0:08x} dni=0x{y0:08x}"),
                            None if nonzero == 0 => " VACUOUS".to_string(),
                            None => String::new(),
                        }
                    );
                    ab.push(d);
                }
                if cfg.ab_iters > 0 && !ab.is_empty() {
                    let (mut t_off, mut t_on): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
                    let mut dsum = 0usize;
                    for i in 0..cfg.ab_iters {
                        let d = &ab[i % ab.len()];
                        dsum += d.distinct;
                        for arm in 0..2 {
                            let on = (i + arm) % 2 == 1;
                            gpu.set_moe_down_dni(on);
                            gpu.hip.event_record(&ev0, None).map_err(err)?;
                            launch(&mut gpu, &st, d)?;
                            gpu.hip.event_record(&ev1, None).map_err(err)?;
                            gpu.hip.event_synchronize(&ev1).map_err(err)?;
                            let us =
                                gpu.hip.event_elapsed_ms(&ev0, &ev1).map_err(err)? as f64 * 1e3;
                            if on {
                                t_on.push(us)
                            } else {
                                t_off.push(us)
                            }
                        }
                    }
                    gpu.set_moe_down_dni(false);
                    let med = |v: &mut Vec<f64>| {
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        v[v.len() / 2]
                    };
                    let (mo, mn) = (med(&mut t_off), med(&mut t_on));
                    let md = dsum as f64 / cfg.ab_iters as f64;
                    let gb = |us: f64| md * st.per as f64 / (us * 1e-6) / 1e9;
                    println!(
                        "AB layer {layer} rows {rows} mode {} : prod {:.1} us ({:.0} GB/s)  dni {:.1} us ({:.0} GB/s)  speedup {:.3}x (mean distinct {:.1})",
                        mode.name(), mo, gb(mo), mn, gb(mn), mo / mn, md
                    );
                }
                for d in ab {
                    let _ = gpu.free_tensor(d.topk);
                    let _ = gpu.free_tensor(d.hidden);
                    let _ = gpu.free_tensor(d.out);
                }
            }
        }
        let _ = gpu.free_tensor(st.data);
        let _ = gpu.free_tensor(st.table);
    }
    for (_, v) in pool.drain() {
        for t in v {
            let _ = gpu.free_tensor(t);
        }
    }
    let verdict = if fail == 0 && pass > 0 {
        "PASS"
    } else {
        "FAIL"
    };
    let names: Vec<&str> = arms.iter().map(|a| a.name.as_str()).collect();
    println!(
        "{}",
        json!({"schema": "moe-down-oracle-v1", "model": cfg.model, "layers": cfg.layers, "rows": cfg.rows,
            "planar": cfg.planar, "cases": pass + fail, "pass": pass, "fail": fail, "verdict": verdict,
            "reps": cfg.reps, "arms": names, "self_diff_reps": tot_self, "cross_words": tot_cross, "paired_diff_reps": tot_paired})
    );
    Ok(if verdict == "PASS" { 0 } else { 1 })
}

fn main() {
    match run() {
        Ok(c) => std::process::exit(c),
        Err(e) => {
            eprintln!("moe_down_oracle: {e}");
            println!(
                "{}",
                json!({"schema": "moe-down-oracle-v1", "verdict": "FAIL", "error": e})
            );
            std::process::exit(1);
        }
    }
}
