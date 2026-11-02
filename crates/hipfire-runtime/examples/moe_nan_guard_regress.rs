// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MoE non-finite router guard: GPU regression harness (producer + consumers).
//!
//! The Qwen4 E512/top-10 router fails a row closed when any of its 512 logits is
//! NaN or +/-Inf (after the optional BF16 round trip): every expert marker `-1`,
//! every weight `0`, and `(layer, row)` stored in the Gpu-lifetime fault word
//! (`rdna_compute::router_fault`). Every consumer treats an id outside
//! `[0, n_experts)` as unrouted. This harness proves, on a real device and
//! against the **old kernels as they are on `origin/beta`**, that
//!
//! * finite inputs produce bitwise the OLD kernels' outputs (producer and
//!   consumers, not "new standard vs new fast");
//! * a non-finite row is failed closed, the fault word names the right
//!   `(layer, row)`, nothing faults, and every other row is bitwise unaffected;
//! * raw consumer ids `-1`, `512`, `INT_MAX` (and `INT_MIN`) do not fault, do not
//!   read/write out of bounds, and leave every valid row bitwise equal to the
//!   OLD kernels' result on the same valid input.
//!
//! # OLD kernels
//!
//! Old sources are read with `git show <old-ref>:kernels/src/<file>` (default
//! the pinned baseline commit `f1a457b`, the origin/beta the guards were cut from)
//! from the checkout the example was built in (or from
//! `--old-dir DIR` holding the same file names, for hosts without history). The
//! bytes are never stored in the tree. Each old symbol `S` is compiled as
//! `S__beta` (`#define S S__beta` ahead of the unmodified source), in its own
//! module, next to the new production kernel of the same name, and launched with
//! the kernarg layout the production launcher had at `origin/beta`. The sha256 of
//! every old source is printed so the receipt shows which bytes were compared.
//!
//! # Run
//!
//! ```sh
//! # build (needs no GPU: hide them so hipcc/rocr never opens a device)
//! HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
//!   cargo build -p hipfire-runtime --release --features lab --example moe_nan_guard_regress
//! # run on the gfx1201 card (fabric P0 hold; the wrapper selects the device)
//! target/release/examples/moe_nan_guard_regress --expect-arch gfx1201
//! # CPU-only source/contract tests (no GPU):
//! HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
//!   cargo test -p hipfire-runtime --release --features lab --example moe_nan_guard_regress
//! ```
//!
//! Default mode is a supervisor: every case runs in its own child process
//! (`--case NAME`), so a device fault names the case that caused it and later
//! cases still run. `--inproc` runs everything in one process. Exit 0 = every
//! case passed or skipped, 1 = a case failed/crashed/timed out.
//!
//! Flags: `--old-ref REF` (default `f1a457b`), `--old-dir DIR`,
//! `--only SUBSTR` (repeatable), `--list`, `--device N`, `--expect-arch ARCH`,
//! `--budget-s N` (total, default 600), `--case-timeout-s N` (default 150),
//! `--out-dir DIR` (default `.`: where `e8p_nan_containment` publishes its JSON and f32le files).
//!
//! `e8p_nan_containment` (gfx1201 + gfx1100 + gfx1151, sibling module `moe_nan_guard_regress/e8p_containment.rs`)
//! compares the old E8P (rc4b; pre-rc5 on gfx1151) (and gfx1100 hot MQ4) K1/K2 images against the new guarded launchers and
//! needs a flat `--old-dir` holding those `.hxaco` images and the pinned beta router sources.
//!
//! Memory (every case but `e8p_nan_containment`): about 150 MiB of device memory per child (16 real
//! Q8_0 experts of 2560x640 = 28 MiB, 16 real MQ4G128V2 experts = 14 MiB, 512-entry pointer
//! tables, activations / grouped rows for <= 300 tokens, router logits); host
//! memory is a few MiB. Wall time on a warm kernel cache is well under 120 s; a
//! cold hipcc cache adds the old-kernel JIT (about 20 small modules).
//!
//! Memory of `e8p_nan_containment` (derived, see the `resources` object of its JSON): gfx1201 peak
//! device 55,224,160 B (52.67 MiB: 16 real E8P experts per projection 20,897,792 B including the
//! two 512-entry pointer tables, plus the 33-row K2 live set 34,318,680 B and the 7,688 B slab union
//! scratch); gfx1100 adds a hot-MQ4 phase (tables freed first) with peak 76,113,760 B (72.59 MiB).
//! Host peak 84,992,000 B (gfx1201) / 98,918,400 B (gfx1100). Flash-Next is never loaded.

#![allow(clippy::too_many_arguments, dead_code)]

#[path = "qwen4_tierc/common.rs"]
mod common;

#[path = "moe_nan_guard_regress/e8p_containment.rs"]
mod e8p_containment;

use common::*;
use hip_bridge::KernargBlob;
use rdna_compute::router_fault::RouterFault;
use rdna_compute::{Gpu, GpuTensor};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

const E: usize = MOE_EXPERTS;
const K: usize = MOE_TOP_K;
const BLOCK_M: usize = 16;
/// Raw ids a corrupt or fail-closed producer can hand a consumer.
const BAD_IDS: [i32; 4] = [-1, 512, i32::MAX, i32::MIN];

type R<T> = Result<T, String>;

// ───────────────────────────────────────────────────────────────────────────────
// Outcome / case table
// ───────────────────────────────────────────────────────────────────────────────
enum Outcome {
    Pass(String),
    Skip(String),
}

struct Case {
    name: &'static str,
    /// What the case exercises (printed by `--list`).
    covers: &'static str,
    run: fn(&mut Ctx) -> R<Outcome>,
}

fn pass(r: R<String>) -> R<Outcome> {
    r.map(Outcome::Pass)
}

const CASES: &[Case] = &[
    Case {
        name: "router_nonfinite",
        covers: "moe_router_softmax_top10_f32 + _fast (new) vs OLD beta incumbent and fast: finite bitwise, \
                 NaN/Inf rows fail closed, fault word (layer,row), isolation",
        run: router_case,
    },
    Case {
        name: "router_old_teeth",
        covers: "negative control: the OLD beta router MUST violate the fail-closed contract on NaN rows",
        run: router_teeth_case,
    },
    Case {
        name: "scatter_stable",
        covers: "qwen4_moe_group_ranks/prefix/scatter (Gpu::moe_scatter_stable_top10) vs OLD kernels + CPU model, raw bad ids",
        run: scatter_stable_case,
    },
    Case {
        name: "scatter_fused",
        covers: "moe_scatter_fused_top10 vs OLD kernel + CPU model, raw bad ids",
        run: scatter_fused_case,
    },
    Case {
        name: "unscatter_silu",
        covers: "moe_gate_up_unscatter_silu_top10 (f32 + bf16in) vs OLD, over the new scatter layouts with unrouted slots",
        run: unscatter_case,
    },
    Case {
        name: "down_indexed",
        covers: "gemv_q8_0 / gemv_mq4g128v2 *_moe_down_top10_indexed_batched_expanded vs OLD, raw bad ids",
        run: down_indexed_case,
    },
    Case {
        name: "down_grouped",
        covers: "gemm_q8_0_moe_grouped_top10 / gemm_mq4g128v2_moe_grouped_top10 vs OLD, bad tile expert ids",
        run: down_grouped_case,
    },
    Case {
        name: "combine_indexed",
        covers: "moe_down_combine_top10_batched vs OLD, failed-closed rows and raw bad ids",
        run: combine_indexed_case,
    },
    Case {
        name: "combine_grouped",
        covers: "moe_down_combine_grouped_top10 and _bf16in (+order kernel) vs OLD, failed-closed rows and raw bad ids",
        run: combine_grouped_case,
    },
    Case {
        name: "chain_prefill_route",
        covers: "router -> stable scatter -> unscatter -> grouped down -> combine, full chain: NaN rows vs OLD on a clean twin",
        run: chain_case,
    },
    Case {
        name: "renorm_producer",
        covers: "moe_topk_renorm_top10_batched (+ moe_router_logits_guard_top10) vs OLD beta renorm: finite bitwise, NaN/Inf rows fail closed with (layer,row)",
        run: renorm_case,
    },
    Case {
        name: "pm_k1_gemv",
        covers: "PM mq_i4_moe_gemv_mq4sym_{r256_gate_up,r128_down}: OLD origin/beta image + OLD planner vs NEW image + NEW \
                 launcher/planner (88 -> 96 B kernarg), raw bad ids in the union, unrouted topk slots",
        run: pm_k1_case,
    },
    Case {
        name: "pm_iu4_sym_grouped",
        covers: "PM grouped IU4 sym gate/up + down images (60 -> 64 B kernarg): OLD image vs NEW launcher, corrupt tile ids",
        run: pm_iu4_case,
    },
    Case {
        name: e8p_containment::CASE_NAME,
        covers: "expert-slab E8P K1/K2 (gfx1201, gfx1100; gfx1100 also hot MQ4): rc4b images vs new guarded launchers bitwise \
                 on finite inputs, raw ids -1/n_experts/INT_MAX/INT_MIN skip with finite canary, shortened-n_experts teeth, \
                 all-NaN/+Inf/-Inf router rows -> consumer chain; publishes JSON + f32le",
        run: e8p_containment::run,
    },
];

// ───────────────────────────────────────────────────────────────────────────────
// Args
// ───────────────────────────────────────────────────────────────────────────────
struct Args {
    device: Option<i32>,
    old_ref: String,
    old_dir: Option<PathBuf>,
    only: Vec<String>,
    list: bool,
    inproc: bool,
    case: Option<String>,
    expect_arch: Option<String>,
    budget_s: u64,
    case_timeout_s: u64,
    /// Checkout holding the `.git` the old ref lives in (default: the one the example was built from).
    repo: Option<PathBuf>,
    /// Required prefix of the old ref's commit (the baseline receipt); `any` disables the check.
    expect_old_sha: String,
    /// Directory `e8p_nan_containment` writes its report and finite output file to.
    out_dir: PathBuf,
}

fn parse_args(argv: impl IntoIterator<Item = String>) -> R<Args> {
    let mut a = Args {
        device: None,
        old_ref: "f1a457b".into(),
        old_dir: None,
        only: Vec::new(),
        list: false,
        inproc: false,
        case: None,
        expect_arch: None,
        budget_s: 600,
        case_timeout_s: 150,
        repo: None,
        expect_old_sha: "f1a457b".into(),
        out_dir: PathBuf::from("."),
    };
    let mut it = argv.into_iter();
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--device" => a.device = Some(val("--device")?.parse().map_err(e)?),
            "--old-ref" => a.old_ref = val("--old-ref")?,
            "--old-dir" => a.old_dir = Some(PathBuf::from(val("--old-dir")?)),
            "--repo" => a.repo = Some(PathBuf::from(val("--repo")?)),
            "--expect-old-sha" => a.expect_old_sha = val("--expect-old-sha")?,
            "--out-dir" => a.out_dir = PathBuf::from(val("--out-dir")?),
            "--only" => a.only.push(val("--only")?),
            "--list" => a.list = true,
            "--inproc" => a.inproc = true,
            "--case" => a.case = Some(val("--case")?),
            "--expect-arch" => a.expect_arch = Some(val("--expect-arch")?),
            "--budget-s" => a.budget_s = val("--budget-s")?.parse().map_err(e)?,
            "--case-timeout-s" => a.case_timeout_s = val("--case-timeout-s")?.parse().map_err(e)?,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(a)
}

// ───────────────────────────────────────────────────────────────────────────────
// Old-kernel loader
// ───────────────────────────────────────────────────────────────────────────────
/// One origin/beta source file and the symbols that are launched from it.
struct OldFile {
    file: &'static str,
    syms: &'static [&'static str],
}

const OLD_ROUTER: OldFile = OldFile {
    file: "moe_router_softmax_top10_f32.hip",
    syms: &["moe_router_softmax_top10_f32"],
};
const OLD_ROUTER_FAST: OldFile = OldFile {
    file: "moe_router_softmax_top10_f32_fast.hip",
    syms: &["moe_router_softmax_top10_f32_fast"],
};
const OLD_STABLE: OldFile = OldFile {
    file: "qwen4_moe_scatter_stable_top10.hip",
    syms: &[
        "qwen4_moe_group_ranks",
        "qwen4_moe_group_prefix",
        "qwen4_moe_group_scatter",
    ],
};
/// `Gpu::moe_scatter_fused_top10` compiles this file behind
/// `#define moe_scatter_fused_k8 moe_scatter_fused_top10`; the exported name
/// does not matter for the old copy, so it is renamed from the source name.
const OLD_FUSED: OldFile = OldFile {
    file: "qwen4_moe_scatter_fused_k8.hip",
    syms: &["moe_scatter_fused_k8"],
};
const OLD_UNSCATTER: OldFile = OldFile {
    file: "moe_gate_up_unscatter_silu_top10.hip",
    syms: &[
        "moe_gate_up_unscatter_silu_top10",
        "moe_gate_up_unscatter_silu_top10_bf16in",
    ],
};
const OLD_COMBINE: OldFile = OldFile {
    file: "moe_down_combine_top10_batched.hip",
    syms: &["moe_down_combine_top10_batched"],
};
const OLD_COMBINE_GROUPED: OldFile = OldFile {
    file: "moe_down_combine_grouped_top10.hip",
    syms: &[
        "moe_down_combine_grouped_top10",
        "moe_combine_order_top10",
        "moe_down_combine_grouped_top10_bf16in",
        "moe_down_combine_grouped_top10_bf16in_zinit",
    ],
};
const OLD_Q8_DOWN: OldFile = OldFile {
    file: "gemv_q8_0_moe_down_top10_indexed_batched_expanded.hip",
    syms: &["gemv_q8_0_moe_down_top10_indexed_batched_expanded"],
};
const OLD_Q8_GROUPED: OldFile = OldFile {
    file: "gemm_q8_0_moe_grouped_top10.hip",
    syms: &["gemm_q8_0_moe_grouped_top10"],
};
const OLD_MQ4_DOWN: OldFile = OldFile {
    file: "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip",
    syms: &["gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded"],
};
const OLD_MQ4_MULTIROW: OldFile = OldFile {
    file: "gemm_mq4g128v2_moe_grouped_top10_multirow.hip",
    syms: &["gemm_mq4g128v2_moe_grouped_top10_multirow"],
};
const OLD_MQ4_O8R16: OldFile = OldFile {
    file: "gemm_mq4g128v2_moe_grouped_top10_o8_r16.gfx1151.hip",
    syms: &["gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151"],
};

/// Pinned baseline commit (origin/beta when the guards were cut); never a moving ref.
const OLD_REF_PIN: &str = "f1a457b";

const OLD_SUFFIX: &str = "__beta";

struct Ctx {
    gpu: Gpu,
    arch: String,
    old_ref: String,
    old_dir: Option<PathBuf>,
    repo: PathBuf,
    out_dir: PathBuf,
    loaded: HashSet<&'static str>,
    /// Notes the case summary appends (provenance, informational counters).
    notes: Vec<String>,
}

impl Ctx {
    fn fast_router_arch(&self) -> bool {
        matches!(self.arch.as_str(), "gfx1151" | "gfx1201")
    }

    fn old_source(&self, file: &str) -> R<String> {
        if let Some(dir) = &self.old_dir {
            let p = dir.join(file);
            return std::fs::read_to_string(&p)
                .map_err(|err| format!("--old-dir {}: {err}", p.display()));
        }
        let spec = format!("{}:kernels/src/{file}", self.old_ref);
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.repo)
            .args(["show", &spec])
            .output()
            .map_err(|err| format!("git show {spec}: {err}"))?;
        if !out.status.success() {
            return Err(format!(
                "git show {spec} failed: {} (pass --old-dir DIR with the origin/beta kernels/src files on a host \
                 without history)",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        String::from_utf8(out.stdout).map_err(e)
    }

    /// Compile every symbol of `old` as `<sym>__beta` from the origin/beta bytes.
    fn load_old(&mut self, old: &'static OldFile) -> R<()> {
        if self.loaded.contains(old.file) {
            return Ok(());
        }
        let text = self.old_source(old.file)?;
        let sha = hex12(&Sha256::digest(text.as_bytes()));
        let src = format!("{}{text}", old_prelude(old.syms));
        let module = format!("nanguard_beta_{}", old.file.replace(['.', '-'], "_"));
        for s in old.syms {
            self.gpu
                .ensure_kernel_public(&module, &src, &format!("{s}{OLD_SUFFIX}"))
                .map_err(|err| format!("compile old {} symbol {s}: {err}", old.file))?;
        }
        self.notes.push(format!(
            "old {} sha256={sha}.. ({})",
            old.file, self.old_ref
        ));
        self.loaded.insert(old.file);
        Ok(())
    }

    fn launch_old(
        &self,
        sym: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        blob: KernargBlob,
    ) -> R<()> {
        let mut bytes = blob.into_vec();
        self.gpu
            .launch_kernel_blob(
                &format!("{sym}{OLD_SUFFIX}"),
                grid,
                block,
                shared,
                &mut bytes,
            )
            .map_err(|err| format!("launch old {sym}: {err}"))
    }
}

/// `#define S S__beta` for every symbol, placed ahead of the unmodified old source: the exported
/// (and every internal) name becomes `S__beta`, so the old copy coexists with the new kernel `S`.
fn old_prelude(syms: &[&str]) -> String {
    syms.iter()
        .map(|s| format!("#define {s} {s}{OLD_SUFFIX}\n"))
        .collect()
}

fn hex12(d: &[u8]) -> String {
    d.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

fn ptr(t: &GpuTensor) -> *const c_void {
    t.buf.as_ptr() as *const c_void
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn up_u64(gpu: &mut Gpu, data: &[u64]) -> R<GpuTensor> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    up_bytes(gpu, &bytes)
}

/// Raw bytes as a tensor whose `byte_size()` is exactly `data.len()`: 4-byte words carried as
/// `DType::F32` bit containers (as `common::up_u32`), so downloads return every byte. A bare
/// `upload_raw` leaves a 1-byte dtype and a download then returns a quarter of the data.
fn up_bytes(gpu: &mut Gpu, data: &[u8]) -> R<GpuTensor> {
    assert_eq!(
        data.len() % 4,
        0,
        "up_bytes: length must be a multiple of 4"
    );
    let mut t = gpu.upload_raw(data, &[data.len() / 4]).map_err(e)?;
    t.dtype = rdna_compute::DType::F32;
    Ok(t)
}

/// Bytes of the grouped IU4 output: BF16 `[grouped_rows, cols]`, `cols` = `m` (down) or `m / 2`
/// (gate/up, SwiGLU halves the rows).
fn iu4_y_bytes(grouped_rows: usize, m: usize, down: bool) -> usize {
    grouped_rows * if down { m } else { m / 2 } * 2
}

/// Bytes of one 16-slot tile of that output.
fn iu4_tile_bytes(grouped_rows: usize, m: usize, down: bool) -> usize {
    BLOCK_M * (iu4_y_bytes(grouped_rows, m, down) / grouped_rows)
}

fn int_fill(gpu: &mut Gpu, n: usize, v: i32) -> R<GpuTensor> {
    up_i32(gpu, &vec![v; n])
}

/// Bit-exact compare of 32-bit words (f32 outputs as bits, ints as u32).
fn exact_bits(label: &str, got: &[u32], want: &[u32]) -> R<String> {
    exact_ints(label, got, want)
}

fn nan_poison() -> f32 {
    f32::from_bits(0x7FC0_DEAD)
}

// ───────────────────────────────────────────────────────────────────────────────
// Fixtures
// ───────────────────────────────────────────────────────────────────────────────
fn valid_id(id: i32) -> bool {
    (0..E as i32).contains(&id)
}

const HOT: [i32; 12] = [7, 100, 101, 255, 256, 257, 300, 333, 400, 500, 510, 511];

#[derive(Clone, Copy, PartialEq)]
enum Spread {
    Uniform,
    Skewed,
    Identical,
}

/// `rows` tokens of ten distinct valid experts each.
fn make_topk(rng: &mut Rng, rows: usize, spread: Spread) -> Vec<i32> {
    let mut out = Vec::with_capacity(rows * K);
    for _ in 0..rows {
        let mut chosen: Vec<i32> = Vec::with_capacity(K);
        if spread == Spread::Identical {
            chosen.extend([3, 17, 64, 65, 200, 255, 256, 300, 480, 511]);
        } else {
            while chosen.len() < K {
                let c = if spread == Spread::Skewed && rng.below(4) != 0 {
                    HOT[rng.below(HOT.len())]
                } else {
                    rng.below(E) as i32
                };
                if !chosen.contains(&c) {
                    chosen.push(c);
                }
            }
        }
        out.extend(chosen);
    }
    out
}

/// What a failed-closed router row looks like to a consumer.
const FAIL_ID: i32 = -1;

/// Deterministic poisoning of a valid `topk`: returns the poisoned ids and the
/// per-slot "unrouted" mask. Row `(r + row_phase) % 4`: 0 untouched, 1 whole
/// row `-1` (a failed-closed router row), 2 four raw bad ids
/// (INT_MAX/512/INT_MIN/-1), 3 one `-1` at slot 5. The final slot of the last
/// row is always `INT_MAX` (chunk-tail case).
fn poison_topk(topk: &[i32], row_phase: usize) -> (Vec<i32>, Vec<bool>) {
    let mut ids = topk.to_vec();
    let rows = ids.len() / K;
    for r in 0..rows {
        match (r + row_phase) % 4 {
            1 => ids[r * K..(r + 1) * K]
                .iter_mut()
                .for_each(|v| *v = FAIL_ID),
            2 => {
                ids[r * K] = i32::MAX;
                ids[r * K + 3] = 512;
                ids[r * K + 6] = i32::MIN;
                ids[r * K + 9] = -1;
            }
            3 => ids[r * K + 5] = -1,
            _ => {}
        }
    }
    if let Some(last) = ids.last_mut() {
        *last = i32::MAX;
    }
    let bad = ids.iter().map(|&v| !valid_id(v)).collect();
    (ids, bad)
}

fn grouped_rows_bound(slots: usize) -> usize {
    (slots + (BLOCK_M - 1) * slots.min(E)).div_ceil(BLOCK_M) * BLOCK_M
}

struct Layout {
    raw: Vec<usize>,
    counts: Vec<i32>,
    offsets: Vec<i32>,
    /// Ascending flat slot inside every expert run (deterministic order).
    sorted: Vec<i32>,
    tiles: Vec<i32>,
    inverse: Vec<i32>,
    m_total_max: usize,
}

/// CPU model of the stable/fused scatter: slots whose id is outside `[0, E)`
/// are unrouted (not counted, no sorted/tile entry, `inverse = -1`).
fn scatter_ref(topk: &[i32]) -> Layout {
    let slots = topk.len();
    let m_total_max = grouped_rows_bound(slots);
    let mut raw = vec![0usize; E];
    for &ex in topk {
        if valid_id(ex) {
            raw[ex as usize] += 1;
        }
    }
    let mut counts = vec![0i32; E];
    let mut offsets = vec![0i32; E + 1];
    let mut acc = 0usize;
    for ex in 0..E {
        let padded = raw[ex].div_ceil(BLOCK_M) * BLOCK_M;
        counts[ex] = padded as i32;
        offsets[ex] = acc as i32;
        acc += padded;
    }
    offsets[E] = acc as i32;
    let mut sorted = vec![-1i32; m_total_max];
    let mut inverse = vec![-1i32; slots];
    let mut cursor = vec![0usize; E];
    for (i, &ex) in topk.iter().enumerate() {
        if !valid_id(ex) {
            continue;
        }
        let ex = ex as usize;
        let pos = offsets[ex] as usize + cursor[ex];
        cursor[ex] += 1;
        sorted[pos] = i as i32;
        inverse[i] = pos as i32;
    }
    let mut tiles = vec![-1i32; m_total_max / BLOCK_M];
    for ex in 0..E {
        for t in (offsets[ex] as usize / BLOCK_M)..(offsets[ex + 1] as usize / BLOCK_M) {
            tiles[t] = ex as i32;
        }
    }
    Layout {
        raw,
        counts,
        offsets,
        sorted,
        tiles,
        inverse,
        m_total_max,
    }
}

struct ScatterOut {
    counts: Vec<i32>,
    offsets: Vec<i32>,
    sorted: Vec<i32>,
    tiles: Vec<i32>,
    inverse: Vec<i32>,
}

// ───────────────────────────────────────────────────────────────────────────────
// Router
// ───────────────────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, Debug)]
enum Bad {
    AllNan,
    OneNan,
    PlusInf,
    MinusInf,
    AllPlusInf,
    NanAndInf,
    /// A finite f32 near `f32::MAX`: finite as given, `+Inf` after the BF16 round trip.
    BigFinite,
}

const BADS: [Bad; 7] = [
    Bad::AllNan,
    Bad::OneNan,
    Bad::PlusInf,
    Bad::MinusInf,
    Bad::AllPlusInf,
    Bad::NanAndInf,
    Bad::BigFinite,
];

fn finite_row(rng: &mut Rng, variant: usize) -> Vec<f32> {
    let mut row = rng.vec(E, -4.0, 4.0);
    match variant % 5 {
        0 => row.iter_mut().for_each(|v| *v = 0.5),
        1 => {
            for &i in &[3usize, 64, 130, 255, 256, 300, 511] {
                row[i] = 3.5;
            }
            for &i in &[9usize, 128, 200, 400] {
                row[i] = 3.0;
            }
        }
        2 => row
            .iter_mut()
            .for_each(|v| *v = (rng.below(4) as f32) - 1.0),
        // Signed zeros (ties across +0/-0) and denormals.
        3 => {
            for (i, v) in row.iter_mut().enumerate().take(96) {
                *v = if i % 2 == 0 { 0.0 } else { -0.0 };
            }
            row[200] = 1e-40;
            row[201] = -1e-40;
        }
        // Extreme finite logits, tied across the two 256-lane halves.
        4 => {
            row[5] = 3.0e38;
            row[300] = 3.0e38;
            row[200] = -3.0e38;
            row[400] = -3.0e38;
        }
        _ => {}
    }
    row
}

fn bad_row(rng: &mut Rng, kind: Bad) -> Vec<f32> {
    let mut row = rng.vec(E, -4.0, 4.0);
    match kind {
        Bad::AllNan => row.iter_mut().for_each(|v| *v = f32::NAN),
        Bad::OneNan => row[300] = f32::NAN,
        Bad::PlusInf => row[17] = f32::INFINITY,
        Bad::MinusInf => row[400] = f32::NEG_INFINITY,
        Bad::AllPlusInf => row.iter_mut().for_each(|v| *v = f32::INFINITY),
        Bad::NanAndInf => {
            row[0] = f32::NAN;
            row[255] = f32::INFINITY;
            row[511] = f32::NEG_INFINITY;
        }
        Bad::BigFinite => row[9] = 3.4028235e38,
    }
    row
}

/// Non-finite after the optional BF16 round trip: the router's fail-closed predicate.
fn row_fails(row: &[f32], round: bool) -> bool {
    row.iter()
        .any(|&v| !(if round { bf16_round(v) } else { v }).is_finite())
}

struct RouterOut {
    ids: Vec<i32>,
    w: Vec<u32>,
    fault: Option<RouterFault>,
}

fn run_router_new(
    cx: &mut Ctx,
    logits: &[f32],
    rows: usize,
    fast: bool,
    norm: bool,
    round: bool,
    layer: usize,
    row_base: usize,
) -> R<RouterOut> {
    Arc::make_mut(&mut cx.gpu.flags).qwen4_router_fast = fast;
    cx.gpu.clear_router_fault();
    let t_l = up_f32(&mut cx.gpu, logits)?;
    let t_i = int_fill(&mut cx.gpu, rows * K, -2)?;
    let t_w = sentinel_f32(&mut cx.gpu, rows * K)?;
    cx.gpu
        .moe_router_softmax_top10_f32(&t_l, &t_i, &t_w, rows, norm, round, layer, row_base)
        .map_err(e)?;
    sync(&cx.gpu)?;
    // The download_* readers fold the fault check in (they would return Err and
    // clear it): read the word first, then clear, then download.
    let fault = cx.gpu.peek_router_fault();
    cx.gpu.clear_router_fault();
    let ids = down_i32(&cx.gpu, &t_i)?;
    let w = bits(&down_f32(&cx.gpu, &t_w)?);
    free_all(&mut cx.gpu, vec![t_l, t_i, t_w])?;
    Ok(RouterOut { ids, w, fault })
}

fn run_router_old(
    cx: &mut Ctx,
    logits: &[f32],
    rows: usize,
    fast: bool,
    norm: bool,
    round: bool,
) -> R<(Vec<i32>, Vec<u32>)> {
    cx.load_old(if fast { &OLD_ROUTER_FAST } else { &OLD_ROUTER })?;
    let t_l = up_f32(&mut cx.gpu, logits)?;
    let t_i = int_fill(&mut cx.gpu, rows * K, -2)?;
    let t_w = sentinel_f32(&mut cx.gpu, rows * K)?;
    let mut b = KernargBlob::new();
    b.push_ptr(ptr(&t_l));
    b.push_ptr(ptr(&t_i));
    b.push_ptr(ptr(&t_w));
    b.push_i32(512);
    b.push_i32(10);
    b.push_i32(i32::from(norm));
    b.push_i32(i32::from(round));
    if fast {
        b.push_i32(rows as i32);
    }
    let (sym, grid) = if fast {
        ("moe_router_softmax_top10_f32_fast", rows.div_ceil(8))
    } else {
        ("moe_router_softmax_top10_f32", rows)
    };
    cx.launch_old(sym, [grid as u32, 1, 1], [256, 1, 1], 0, b)?;
    sync(&cx.gpu)?;
    let ids = down_i32(&cx.gpu, &t_i)?;
    let w = bits(&down_f32(&cx.gpu, &t_w)?);
    free_all(&mut cx.gpu, vec![t_l, t_i, t_w])?;
    Ok((ids, w))
}

/// The fail-closed contract for one launch's output rows.
fn check_router_rows(
    label: &str,
    out: &RouterOut,
    failing: &[bool],
    clean_ids: &[i32],
    clean_w: &[u32],
) -> R<String> {
    let rows = failing.len();
    let mut first = String::new();
    for r in 0..rows {
        let (ids, w) = (&out.ids[r * K..(r + 1) * K], &out.w[r * K..(r + 1) * K]);
        if failing[r] {
            if ids.iter().any(|&v| v != -1) || w.iter().any(|&v| v != 0) {
                first =
                    format!("failing row {r}: ids {ids:?} w_bits {w:x?} (want ids -1, weights +0)");
                break;
            }
        } else if ids != &clean_ids[r * K..(r + 1) * K] || w != &clean_w[r * K..(r + 1) * K] {
            first = format!(
                "finite row {r} differs from the old kernel on the clean twin: ids {ids:?}"
            );
            break;
        }
    }
    if first.is_empty() {
        let n_bad = failing.iter().filter(|f| **f).count();
        Ok(format!(
            "{label} rows ok ({n_bad} failed closed, {} finite bitwise == old)",
            rows - n_bad
        ))
    } else {
        Err(format!("{label}: {first}"))
    }
}

fn router_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E01);
    let mut results: Vec<R<String>> = Vec::new();
    let fast_ok = cx.fast_router_arch();
    let mut launches = 0usize;
    for &rows in &[1usize, 3, 9, 33] {
        // rows == 1: one matrix per bad kind plus a finite control; otherwise
        // (mixed: every third row bad, all rows bad, all bad but one).
        let layouts: Vec<Vec<Option<Bad>>> = if rows == 1 {
            BADS.iter()
                .map(|&b| vec![Some(b)])
                .chain(std::iter::once(vec![None]))
                .collect()
        } else {
            let mut cyc = 0usize;
            let mixed: Vec<Option<Bad>> = (0..rows)
                .map(|r| {
                    if r % 3 == 1 {
                        cyc += 1;
                        Some(BADS[(cyc - 1) % BADS.len()])
                    } else {
                        None
                    }
                })
                .collect();
            let all_bad: Vec<Option<Bad>> = (0..rows).map(|r| Some(BADS[r % BADS.len()])).collect();
            let mut one_good = all_bad.clone();
            one_good[rows / 2] = None;
            vec![mixed, all_bad, one_good]
        };
        for (li, layout) in layouts.iter().enumerate() {
            let mut logits = Vec::with_capacity(rows * E);
            for (r, k) in layout.iter().enumerate() {
                logits.extend(match k {
                    Some(kind) => bad_row(&mut rng, *kind),
                    None => finite_row(&mut rng, r + li),
                });
            }
            for &(norm, round) in &[(true, true), (true, false), (false, false)] {
                let failing: Vec<bool> = (0..rows)
                    .map(|r| row_fails(&logits[r * E..(r + 1) * E], round))
                    .collect();
                // Clean twin: failing rows replaced by finite ones; everything else identical.
                let mut clean = logits.clone();
                for r in 0..rows {
                    if failing[r] {
                        clean[r * E..(r + 1) * E].copy_from_slice(&finite_row(&mut rng, r + 7));
                    }
                }
                let tag = format!("rows={rows} layout={li} norm={norm} round={round}");
                let (old_ids, old_w) = run_router_old(cx, &clean, rows, false, norm, round)?;
                for fast in [false, true] {
                    if fast && (!fast_ok || rows == 1) {
                        continue; // decode (one token), recording and capture keep the incumbent symbol
                    }
                    let (layer, row_base) = (11usize, 4096usize);
                    let new_clean =
                        run_router_new(cx, &clean, rows, fast, norm, round, layer, row_base)?;
                    let new_bad =
                        run_router_new(cx, &logits, rows, fast, norm, round, layer, row_base)?;
                    launches += 2;
                    let ftag = format!("{tag} fast={fast}");
                    // 1. finite input: new == OLD incumbent (and OLD fast), bit for bit; no fault.
                    results.push(exact_ints(
                        &format!("{ftag} clean ids vs old"),
                        &new_clean.ids,
                        &old_ids,
                    ));
                    results.push(exact_ints(
                        &format!("{ftag} clean w vs old"),
                        &new_clean.w,
                        &old_w,
                    ));
                    if fast {
                        let (fids, fw) = run_router_old(cx, &clean, rows, true, norm, round)?;
                        results.push(exact_ints(
                            &format!("{ftag} clean ids vs old FAST"),
                            &new_clean.ids,
                            &fids,
                        ));
                        results.push(exact_ints(
                            &format!("{ftag} clean w vs old FAST"),
                            &new_clean.w,
                            &fw,
                        ));
                    }
                    results.push(match new_clean.fault {
                        None => Ok(format!("{ftag} clean: no fault")),
                        Some(f) => Err(format!("{ftag} clean input recorded a fault {f:?}")),
                    });
                    // 2. poisoned input: fail closed + isolation + fault word.
                    results.push(check_router_rows(
                        &format!("{ftag} poisoned"),
                        &new_bad,
                        &failing,
                        &old_ids,
                        &old_w,
                    ));
                    let first_bad = failing.iter().position(|f| *f);
                    let n_bad = failing.iter().filter(|f| **f).count();
                    results.push(match (first_bad, new_bad.fault) {
                        (None, None) => Ok(format!("{ftag} no failing row, no fault")),
                        (None, Some(f)) => Err(format!("{ftag} no failing row but fault {f:?}")),
                        (Some(_), None) => Err(format!("{ftag} failing rows but the fault word is clear")),
                        (Some(first), Some(f)) => {
                            // One failing row: exact. Several race in one launch, so the word keeps
                            // ONE of the failing global rows (last store wins) with the right layer.
                            let ok_row = if n_bad == 1 {
                                f.row == (row_base + first) as u64
                            } else {
                                failing.get((f.row as usize).wrapping_sub(row_base)).copied().unwrap_or(false)
                            };
                            if f.layer == layer as u32 && ok_row {
                                Ok(format!("{ftag} fault layer={} row={}", f.layer, f.row))
                            } else {
                                Err(format!("{ftag} fault {f:?}, want layer {layer} and a failing row (first bad {})", row_base + first))
                            }
                        }
                    });
                }
            }
        }
    }
    results.push(router_fault_semantics(cx));
    cx.notes.push(format!(
        "router: {launches} new-kernel launches; fast kernel exercised: {fast_ok}"
    ));
    pass(all(results))
}

/// Fault-word semantics through the real launcher: `layer`/`row_base` encoding,
/// earliest-row-wins across launches, and the folded `take_device_wait_error`.
fn router_fault_semantics(cx: &mut Ctx) -> R<String> {
    let mut rng = Rng::new(0x4E41_4E02);
    let rows = 3;
    let mut logits = Vec::new();
    logits.extend(finite_row(&mut rng, 1));
    logits.extend(bad_row(&mut rng, Bad::AllNan));
    logits.extend(finite_row(&mut rng, 2));
    let mut checks: Vec<R<String>> = Vec::new();
    // (layer, row_base) round trips; only row 1 of the launch is bad.
    for &(layer, row_base) in &[(0usize, 0usize), (7, 100), (39, 1 << 20), (1000, 5)] {
        let out = run_router_new(cx, &logits, rows, false, true, true, layer, row_base)?;
        checks.push(match out.fault {
            Some(f) if f.layer == layer as u32 && f.row == (row_base + 1) as u64 => Ok(format!(
                "fault (layer {layer}, row_base {row_base}) -> row {}",
                f.row
            )),
            other => Err(format!(
                "layer {layer} row_base {row_base}: fault {other:?}, want row {}",
                row_base + 1
            )),
        });
    }
    // Earliest row wins across launches, row before layer, until cleared.
    cx.gpu.clear_router_fault();
    let launch = |cx: &mut Ctx, layer: usize, row_base: usize| -> R<()> {
        let t_l = up_f32(&mut cx.gpu, &logits)?;
        let t_i = int_fill(&mut cx.gpu, rows * K, -2)?;
        let t_w = sentinel_f32(&mut cx.gpu, rows * K)?;
        cx.gpu
            .moe_router_softmax_top10_f32(&t_l, &t_i, &t_w, rows, true, true, layer, row_base)
            .map_err(e)?;
        sync(&cx.gpu)?;
        free_all(&mut cx.gpu, vec![t_l, t_i, t_w])
    };
    launch(cx, 9, 500)?; // row 501 layer 9
    launch(cx, 3, 800)?; // row 801 layer 3: later row, must not displace
    checks.push(match cx.gpu.peek_router_fault() {
        Some(f) if f.layer == 9 && f.row == 501 => {
            Ok("later row did not displace the earlier fault".into())
        }
        other => Err(format!(
            "after rows 501 then 801: {other:?}, want layer 9 row 501"
        )),
    });
    launch(cx, 2, 300)?; // row 301: earlier row displaces
    checks.push(match cx.gpu.peek_router_fault() {
        Some(f) if f.layer == 2 && f.row == 301 => {
            Ok("earlier row displaced the later fault".into())
        }
        other => Err(format!("after row 301: {other:?}, want layer 2 row 301")),
    });
    launch(cx, 1, 300)?; // same row, earlier layer wins
    checks.push(match cx.gpu.peek_router_fault() {
        Some(f) if f.layer == 1 && f.row == 301 => Ok("same row: earlier layer won".into()),
        other => Err(format!(
            "same row smaller layer: {other:?}, want layer 1 row 301"
        )),
    });
    // The folded check: take_device_wait_error names layer and row, then clears.
    let err = cx.gpu.take_device_wait_error();
    checks.push(match &err {
        Err(er) if er.to_string().contains("layer 1") && er.to_string().contains("row 301") => {
            Ok("take_device_wait_error names layer 1 row 301".into())
        }
        other => Err(format!(
            "take_device_wait_error: {other:?}, want Err naming layer 1 row 301"
        )),
    });
    checks.push(
        match (cx.gpu.peek_router_fault(), cx.gpu.take_device_wait_error()) {
            (None, Ok(())) => Ok("fault word cleared by the take".into()),
            other => Err(format!("fault word not cleared after take: {other:?}")),
        },
    );
    all(checks)
}

/// Negative control: the OLD beta router, fed the same NaN rows, MUST NOT meet
/// the fail-closed contract — proof the router assertions have teeth.
fn router_teeth_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E03);
    let rows = 3;
    let mut logits = Vec::new();
    logits.extend(finite_row(&mut rng, 1));
    logits.extend(bad_row(&mut rng, Bad::AllNan));
    logits.extend(finite_row(&mut rng, 2));
    let (ids, w) = run_router_old(cx, &logits, rows, false, true, false)?;
    let bad_ids = &ids[K..2 * K];
    let bad_w = &w[K..2 * K];
    let contract_met = bad_ids.iter().all(|&v| v == -1) && bad_w.iter().all(|&v| v == 0);
    if contract_met {
        return Err(
            "old beta router already fails closed: the router assertions would have no teeth"
                .into(),
        );
    }
    let unwritten = bad_w
        .iter()
        .filter(|&&v| v == nan_poison().to_bits())
        .count();
    Ok(Outcome::Pass(format!(
        "old beta router on an all-NaN row: ids {bad_ids:?}, {unwritten}/10 weights left unwritten (sentinel) -> \
         the new-kernel assertions (ids -1, weights +0, fault word) fail on old behavior"
    )))
}

// ───────────────────────────────────────────────────────────────────────────────
// Scatter
// ───────────────────────────────────────────────────────────────────────────────
struct ScatterBufs {
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inverse: GpuTensor,
}

fn scatter_bufs(gpu: &mut Gpu, slots: usize, m_total_max: usize) -> R<ScatterBufs> {
    Ok(ScatterBufs {
        counts: int_fill(gpu, E, -2)?,
        offsets: int_fill(gpu, E + 1, -2)?,
        sorted: int_fill(gpu, m_total_max, -2)?,
        tiles: int_fill(gpu, m_total_max / BLOCK_M, -2)?,
        inverse: int_fill(gpu, slots, -2)?,
    })
}

impl ScatterBufs {
    fn download(&self, gpu: &Gpu) -> R<ScatterOut> {
        Ok(ScatterOut {
            counts: down_i32(gpu, &self.counts)?,
            offsets: down_i32(gpu, &self.offsets)?,
            sorted: down_i32(gpu, &self.sorted)?,
            tiles: down_i32(gpu, &self.tiles)?,
            inverse: down_i32(gpu, &self.inverse)?,
        })
    }
    fn free(self, gpu: &mut Gpu) -> R<()> {
        free_all(
            gpu,
            vec![
                self.counts,
                self.offsets,
                self.sorted,
                self.tiles,
                self.inverse,
            ],
        )
    }
}

fn run_stable_new(cx: &mut Ctx, topk: &[i32]) -> R<ScatterOut> {
    let slots = topk.len();
    let gr = grouped_rows_bound(slots);
    let t_topk = up_i32(&mut cx.gpu, topk)?;
    let b = scatter_bufs(&mut cx.gpu, slots, gr)?;
    cx.gpu
        .moe_scatter_stable_top10(
            &t_topk, &b.counts, &b.offsets, &b.sorted, &b.tiles, &b.inverse, slots, E, gr,
        )
        .map_err(e)?;
    sync(&cx.gpu)?;
    let out = b.download(&cx.gpu)?;
    b.free(&mut cx.gpu)?;
    free_all(&mut cx.gpu, vec![t_topk])?;
    Ok(out)
}

/// The OLD three-kernel stable scatter on caller scratch (valid ids only: the
/// old `qwen4_moe_group_ranks` writes `partial[id * C + c]` unchecked).
fn run_stable_old(cx: &mut Ctx, topk: &[i32]) -> R<ScatterOut> {
    cx.load_old(&OLD_STABLE)?;
    let slots = topk.len();
    let gr = grouped_rows_bound(slots);
    let chunks = slots.div_ceil(256);
    let t_topk = up_i32(&mut cx.gpu, topk)?;
    let b = scatter_bufs(&mut cx.gpu, slots, gr)?;
    let rank = int_fill(&mut cx.gpu, slots, -2)?;
    let partial = int_fill(&mut cx.gpu, E * chunks, -2)?;
    let prefix = int_fill(&mut cx.gpu, E * (chunks + 1), -2)?;
    let mut k1 = KernargBlob::new();
    k1.push_ptr(ptr(&t_topk));
    k1.push_ptr(ptr(&rank));
    k1.push_ptr(ptr(&partial));
    k1.push_i32(slots as i32);
    k1.push_i32(E as i32);
    k1.push_i32(chunks as i32);
    cx.launch_old(
        "qwen4_moe_group_ranks",
        [chunks as u32, 1, 1],
        [256, 1, 1],
        0,
        k1,
    )?;
    let mut k2 = KernargBlob::new();
    k2.push_ptr(ptr(&partial));
    k2.push_ptr(ptr(&prefix));
    k2.push_ptr(ptr(&b.counts));
    k2.push_ptr(ptr(&b.offsets));
    k2.push_i32(E as i32);
    k2.push_i32(chunks as i32);
    cx.launch_old("qwen4_moe_group_prefix", [1, 1, 1], [E as u32, 1, 1], 0, k2)?;
    let mut k3 = KernargBlob::new();
    k3.push_ptr(ptr(&t_topk));
    k3.push_ptr(ptr(&rank));
    k3.push_ptr(ptr(&prefix));
    k3.push_ptr(ptr(&b.offsets));
    k3.push_ptr(ptr(&b.sorted));
    k3.push_ptr(ptr(&b.tiles));
    k3.push_ptr(ptr(&b.inverse));
    k3.push_i32(slots as i32);
    k3.push_i32(E as i32);
    k3.push_i32(gr as i32);
    cx.launch_old(
        "qwen4_moe_group_scatter",
        [gr.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        k3,
    )?;
    sync(&cx.gpu)?;
    let out = b.download(&cx.gpu)?;
    b.free(&mut cx.gpu)?;
    free_all(&mut cx.gpu, vec![t_topk, rank, partial, prefix])?;
    Ok(out)
}

fn cmp_scatter_exact(label: &str, got: &ScatterOut, want: &ScatterOut) -> Vec<R<String>> {
    vec![
        exact_ints(&format!("{label} counts"), &got.counts, &want.counts),
        exact_ints(&format!("{label} offsets"), &got.offsets, &want.offsets),
        exact_ints(&format!("{label} sorted"), &got.sorted, &want.sorted),
        exact_ints(&format!("{label} tiles"), &got.tiles, &want.tiles),
        exact_ints(&format!("{label} inverse"), &got.inverse, &want.inverse),
    ]
}

fn lay_out(l: &Layout) -> ScatterOut {
    ScatterOut {
        counts: l.counts.clone(),
        offsets: l.offsets.clone(),
        sorted: l.sorted.clone(),
        tiles: l.tiles.clone(),
        inverse: l.inverse.clone(),
    }
}

fn scatter_stable_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E10);
    let mut results = Vec::new();
    let mut n = 0usize;
    for &rows in &[1usize, 2, 7, 26, 33, 128, 300] {
        for (si, spread) in [Spread::Uniform, Spread::Skewed, Spread::Identical]
            .into_iter()
            .enumerate()
        {
            let topk = make_topk(&mut rng, rows, spread);
            let tag = format!("rows={rows} spread={si}");
            // Finite/valid input: new == OLD kernels == CPU model, all five outputs.
            let new = run_stable_new(cx, &topk)?;
            let old = run_stable_old(cx, &topk)?;
            let model = lay_out(&scatter_ref(&topk));
            results.extend(cmp_scatter_exact(
                &format!("{tag} valid new-vs-old"),
                &new,
                &old,
            ));
            results.extend(cmp_scatter_exact(
                &format!("{tag} valid new-vs-model"),
                &new,
                &model,
            ));
            // Raw bad ids: no fault, unrouted slots drop out, everything else as the model.
            for phase in 0..4 {
                let (bad_ids, bad) = poison_topk(&topk, phase);
                let new_bad = run_stable_new(cx, &bad_ids)?;
                let model_bad = scatter_ref(&bad_ids);
                results.extend(cmp_scatter_exact(
                    &format!("{tag} phase={phase} bad-ids vs model"),
                    &new_bad,
                    &lay_out(&model_bad),
                ));
                let unrouted: Vec<i32> = new_bad
                    .inverse
                    .iter()
                    .zip(&bad)
                    .filter(|(_, b)| **b)
                    .map(|(&v, _)| v)
                    .collect();
                results.push(exact_ints(
                    &format!("{tag} phase={phase} unrouted inverse == -1"),
                    unrouted.as_slice(),
                    vec![-1; unrouted.len()].as_slice(),
                ));
                n += 1;
            }
        }
    }
    // Every slot unrouted: no tile, no row.
    for &id in &BAD_IDS {
        let topk = vec![id; 33 * K];
        let new = run_stable_new(cx, &topk)?;
        results.extend(cmp_scatter_exact(
            &format!("all-{id} stream vs model"),
            &new,
            &lay_out(&scatter_ref(&topk)),
        ));
    }
    cx.notes.push(format!(
        "scatter_stable: {n} poisoned launches (+ valid old/new/model comparisons)"
    ));
    pass(all(results))
}

fn canonical_sorted(offsets: &[i32], raw: &[usize], sorted: &[i32]) -> R<Vec<i32>> {
    let mut canon = sorted.to_vec();
    for ex in 0..E {
        let (lo, n) = (offsets[ex] as usize, raw[ex]);
        if lo + n > canon.len() {
            return Err(format!("expert {ex} run outside the sorted buffer"));
        }
        canon[lo..lo + n].sort_unstable();
    }
    Ok(canon)
}

fn run_fused(cx: &mut Ctx, topk: &[i32], old: bool) -> R<ScatterOut> {
    let slots = topk.len();
    let gr = grouped_rows_bound(slots);
    let t_topk = up_i32(&mut cx.gpu, topk)?;
    let b = scatter_bufs(&mut cx.gpu, slots, gr)?;
    if old {
        cx.load_old(&OLD_FUSED)?;
        let mut k = KernargBlob::new();
        k.push_ptr(ptr(&t_topk));
        k.push_ptr(ptr(&b.counts));
        k.push_ptr(ptr(&b.offsets));
        k.push_ptr(ptr(&b.sorted));
        k.push_ptr(ptr(&b.tiles));
        k.push_ptr(ptr(&b.inverse));
        k.push_i32(slots as i32);
        k.push_i32(E as i32);
        k.push_i32(gr as i32);
        k.push_i32(BLOCK_M as i32);
        cx.launch_old(
            "moe_scatter_fused_k8",
            [1, 1, 1],
            [256, 1, 1],
            (E * 4) as u32,
            k,
        )?;
    } else {
        cx.gpu
            .moe_scatter_fused_top10(
                &t_topk, &b.counts, &b.offsets, &b.sorted, &b.tiles, &b.inverse, slots, E, gr,
                BLOCK_M,
            )
            .map_err(e)?;
    }
    sync(&cx.gpu)?;
    let out = b.download(&cx.gpu)?;
    b.free(&mut cx.gpu)?;
    free_all(&mut cx.gpu, vec![t_topk])?;
    Ok(out)
}

/// Intra-run order of the fused scatter is LDS-atomic arrival order, so the
/// comparison canonicalises each expert's live run (ascending) and then checks
/// `sorted[inverse[i]] == i` for every routed slot.
fn cmp_fused(label: &str, topk: &[i32], g: &ScatterOut, lay: &Layout) -> Vec<R<String>> {
    let mut r = vec![
        exact_ints(&format!("{label} counts"), &g.counts, &lay.counts),
        exact_ints(&format!("{label} offsets"), &g.offsets, &lay.offsets),
        exact_ints(&format!("{label} tiles"), &g.tiles, &lay.tiles),
    ];
    r.push(
        canonical_sorted(&lay.offsets, &lay.raw, &g.sorted).and_then(|c| {
            exact_ints(
                &format!("{label} sorted (per-run canonical)"),
                &c,
                &lay.sorted,
            )
        }),
    );
    let mut bad = None;
    for (i, &ex) in topk.iter().enumerate() {
        let pos = g.inverse[i];
        if valid_id(ex) {
            let (lo, n) = (lay.offsets[ex as usize], lay.raw[ex as usize] as i32);
            let ok = pos >= lo
                && pos < lo + n
                && (pos as usize) < g.sorted.len()
                && g.sorted[pos as usize] == i as i32;
            if !ok {
                bad = Some(format!(
                    "routed slot {i}: inverse {pos} is not its row in expert {ex}'s run"
                ));
                break;
            }
        } else if pos != -1 {
            bad = Some(format!(
                "unrouted slot {i} (id {ex}): inverse {pos}, want -1"
            ));
            break;
        }
    }
    r.push(match bad {
        None => Ok(format!("{label} inverse consistent")),
        Some(m) => Err(format!("{label}: {m}")),
    });
    r
}

fn scatter_fused_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E11);
    let mut results = Vec::new();
    for &rows in &[1usize, 7, 33, 300] {
        for (si, spread) in [Spread::Uniform, Spread::Skewed, Spread::Identical]
            .into_iter()
            .enumerate()
        {
            let topk = make_topk(&mut rng, rows, spread);
            let tag = format!("rows={rows} spread={si}");
            let lay = scatter_ref(&topk);
            let new = run_fused(cx, &topk, false)?;
            let old = run_fused(cx, &topk, true)?;
            results.extend(cmp_fused(&format!("{tag} valid new"), &topk, &new, &lay));
            results.extend(cmp_fused(&format!("{tag} valid old"), &topk, &old, &lay));
            results.push(exact_ints(
                &format!("{tag} new-vs-old counts"),
                &new.counts,
                &old.counts,
            ));
            results.push(exact_ints(
                &format!("{tag} new-vs-old offsets"),
                &new.offsets,
                &old.offsets,
            ));
            results.push(exact_ints(
                &format!("{tag} new-vs-old tiles"),
                &new.tiles,
                &old.tiles,
            ));
            results.push(
                canonical_sorted(&lay.offsets, &lay.raw, &new.sorted)
                    .and_then(|a| {
                        canonical_sorted(&lay.offsets, &lay.raw, &old.sorted).map(|b| (a, b))
                    })
                    .and_then(|(a, b)| {
                        exact_ints(&format!("{tag} new-vs-old sorted (canonical)"), &a, &b)
                    }),
            );
            for phase in 0..4 {
                let (bad_ids, _) = poison_topk(&topk, phase);
                let lay_bad = scatter_ref(&bad_ids);
                let new_bad = run_fused(cx, &bad_ids, false)?;
                results.extend(cmp_fused(
                    &format!("{tag} phase={phase} bad-ids new"),
                    &bad_ids,
                    &new_bad,
                    &lay_bad,
                ));
                // The old fused kernel already skipped bad ids in histogram/fill (it only left
                // `inverse` stale): everything but the unrouted inverse entries must match.
                let old_bad = run_fused(cx, &bad_ids, true)?;
                results.push(exact_ints(
                    &format!("{tag} phase={phase} bad-ids new-vs-old counts"),
                    &new_bad.counts,
                    &old_bad.counts,
                ));
                results.push(exact_ints(
                    &format!("{tag} phase={phase} bad-ids new-vs-old tiles"),
                    &new_bad.tiles,
                    &old_bad.tiles,
                ));
            }
        }
    }
    pass(all(results))
}

// ───────────────────────────────────────────────────────────────────────────────
// Shared consumer fixtures
// ───────────────────────────────────────────────────────────────────────────────
/// Finite specials every operand mix carries: signed zeros, denormals, large
/// magnitudes, the exp() overflow edge.
const SPECIALS: [f32; 10] = [
    0.0, -0.0, 1e-40, -1e-40, 1e-30, -1e-30, 88.0, -88.0, 1e30, -1e30,
];

fn extreme_vec(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            if rng.below(16) == 0 {
                SPECIALS[rng.below(SPECIALS.len())]
            } else {
                rng.range(lo, hi)
            }
        })
        .collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16_bits_vec(rng: &mut Rng, n: usize) -> Vec<u16> {
    extreme_vec(rng, n, -3.0, 3.0)
        .into_iter()
        .map(f32_to_bf16_bits)
        .collect()
}

const N_REAL: usize = 16;

/// 512 pointer table over `N_REAL` real expert tensors (`expert e` -> real `e % N_REAL`).
struct ExpertTable {
    real: Vec<GpuTensor>,
    table: GpuTensor,
}

impl ExpertTable {
    fn build(gpu: &mut Gpu, real: Vec<GpuTensor>) -> R<Self> {
        let ptrs: Vec<u64> = (0..E)
            .map(|e| real[e % real.len()].buf.as_ptr() as u64)
            .collect();
        let table = up_u64(gpu, &ptrs)?;
        Ok(Self { real, table })
    }
    fn free(self, gpu: &mut Gpu) -> R<()> {
        let mut all_t = self.real;
        all_t.push(self.table);
        free_all(gpu, all_t)
    }
}

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// Q8_0 expert: `m` rows of `k / 32` blocks of `[f16 scale][32 x i8]` (34 B).
fn q8_expert(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 32) * 34);
    for _ in 0..m * (k / 32) {
        out.extend((0x1800u16 + rng.below(0x2000) as u16).to_le_bytes());
        out.extend((0..32).map(|_| rng.below(256) as u8));
    }
    out
}

/// MQ4G128V2 down expert: `m` rows of `k / 128` groups `[sc zp : f16][64 nibble bytes]` (68 B,
/// symmetric header `zp = f16(-8 sc)`).
fn mq4_down_expert(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 128) * 68);
    for _ in 0..m * (k / 128) {
        let sc = 0x1800u16 + rng.below(0x2000) as u16;
        out.extend(sc.to_le_bytes());
        out.extend(f16_bits(-8.0 * half::f16::from_bits(sc).to_f32()).to_le_bytes());
        out.extend((0..64).map(|_| rng.below(256) as u8));
    }
    out
}

/// MQ4G256V2 gate/up expert: `m` rows of `k / 256` groups `[sc0 zp0 sc1 zp1 : f16][128 nibble bytes]`.
fn mq4_gate_expert(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 256) * 136);
    for _ in 0..m * (k / 256) {
        for _ in 0..2 {
            let sc = 0x1800u16 + rng.below(0x2000) as u16;
            out.extend(sc.to_le_bytes());
            out.extend(f16_bits(-8.0 * half::f16::from_bits(sc).to_f32()).to_le_bytes());
        }
        out.extend((0..128).map(|_| rng.below(256) as u8));
    }
    out
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fmt {
    Q8,
    Mq4,
}

fn build_experts(cx: &mut Ctx, rng: &mut Rng, fmt: Fmt, m: usize, k: usize) -> R<ExpertTable> {
    let mut real = Vec::with_capacity(N_REAL);
    for _ in 0..N_REAL {
        let bytes = match fmt {
            Fmt::Q8 => q8_expert(rng, m, k),
            Fmt::Mq4 => mq4_down_expert(rng, m, k),
        };
        real.push(up_bytes(&mut cx.gpu, &bytes)?);
    }
    ExpertTable::build(&mut cx.gpu, real)
}

// ───────────────────────────────────────────────────────────────────────────────
// Unscatter + SiLU
// ───────────────────────────────────────────────────────────────────────────────
fn run_unscatter(
    cx: &mut Ctx,
    y: &[f32],
    y_bf16: Option<&[u16]>,
    sorted: &[i32],
    slots: usize,
    mi: usize,
    rt: bool,
    old: bool,
) -> R<Vec<u32>> {
    let gr = sorted.len();
    let t_sorted = up_i32(&mut cx.gpu, sorted)?;
    let t_act = sentinel_f32(&mut cx.gpu, slots * mi)?;
    let t_y = match y_bf16 {
        Some(b) => up_bytes(&mut cx.gpu, &u16_bytes(b))?,
        None => up_f32(&mut cx.gpu, y)?,
    };
    let sym = if y_bf16.is_some() {
        "moe_gate_up_unscatter_silu_top10_bf16in"
    } else {
        "moe_gate_up_unscatter_silu_top10"
    };
    if old {
        cx.load_old(&OLD_UNSCATTER)?;
        let mut b = KernargBlob::new();
        b.push_ptr(ptr(&t_y));
        b.push_ptr(ptr(&t_sorted));
        b.push_ptr(ptr(&t_act));
        b.push_i32(mi as i32);
        b.push_i32(gr as i32);
        b.push_i32(i32::from(rt));
        cx.launch_old(
            sym,
            [gr as u32, mi.div_ceil(256) as u32, 1],
            [256, 1, 1],
            0,
            b,
        )?;
    } else if y_bf16.is_some() {
        cx.gpu
            .moe_gate_up_unscatter_silu_top10_bf16in(&t_y, &t_sorted, &t_act, mi, gr, rt)
            .map_err(e)?;
    } else {
        cx.gpu
            .moe_gate_up_unscatter_silu_top10(&t_y, &t_sorted, &t_act, mi, gr, rt)
            .map_err(e)?;
    }
    sync(&cx.gpu)?;
    let out = bits(&down_f32(&cx.gpu, &t_act)?);
    free_all(&mut cx.gpu, vec![t_sorted, t_act, t_y])?;
    Ok(out)
}

fn unscatter_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E20);
    let mi = MOE_INTERMEDIATE;
    let mut results = Vec::new();
    for &rows in &[1usize, 7, 33] {
        let topk = make_topk(&mut rng, rows, Spread::Uniform);
        let slots = topk.len();
        for phase in 0..5 {
            // phase 4: valid layout; 0..4: layouts with unrouted slots (sorted has no entry for them).
            let ids = if phase == 4 {
                topk.clone()
            } else {
                poison_topk(&topk, phase).0
            };
            let lay = scatter_ref(&ids);
            let gr = lay.m_total_max;
            let y = extreme_vec(&mut rng, gr * 2 * mi, -4.0, 4.0);
            let yb = bf16_bits_vec(&mut rng, gr * 2 * mi);
            for rt in [false, true] {
                let tag = format!("rows={rows} phase={phase} rt={rt}");
                let new = run_unscatter(cx, &y, None, &lay.sorted, slots, mi, rt, false)?;
                let old = run_unscatter(cx, &y, None, &lay.sorted, slots, mi, rt, true)?;
                results.push(exact_bits(&format!("{tag} f32 act new-vs-old"), &new, &old));
                let newb = run_unscatter(cx, &y, Some(&yb), &lay.sorted, slots, mi, rt, false)?;
                let oldb = run_unscatter(cx, &y, Some(&yb), &lay.sorted, slots, mi, rt, true)?;
                results.push(exact_bits(
                    &format!("{tag} bf16in act new-vs-old"),
                    &newb,
                    &oldb,
                ));
                // Unrouted slots' flat rows are never written (stay the NaN sentinel).
                let sentinel = nan_poison().to_bits();
                let bad = (0..slots)
                    .filter(|&i| lay.inverse[i] < 0)
                    .any(|i| new[i * mi..(i + 1) * mi].iter().any(|&v| v != sentinel));
                results.push(if bad {
                    Err(format!(
                        "{tag}: an unrouted slot's activation row was written"
                    ))
                } else {
                    Ok(format!("{tag}: unrouted rows untouched"))
                });
            }
        }
    }
    pass(all(results))
}

// ───────────────────────────────────────────────────────────────────────────────
// Indexed down gemv (Q8_0 / MQ4G128V2)
// ───────────────────────────────────────────────────────────────────────────────
const DOWN_M: usize = HIDDEN;
const DOWN_K: usize = MOE_INTERMEDIATE;

fn run_down_indexed(
    cx: &mut Ctx,
    fmt: Fmt,
    tab: &ExpertTable,
    ids: &[i32],
    x: &[f32],
    tokens: usize,
    old: bool,
) -> R<Vec<u32>> {
    let t_ids = up_i32(&mut cx.gpu, ids)?;
    let t_x = up_f32(&mut cx.gpu, x)?;
    let t_y = sentinel_f32(&mut cx.gpu, tokens * K * DOWN_M)?;
    if old {
        let (file, sym) = match fmt {
            Fmt::Q8 => (
                &OLD_Q8_DOWN,
                "gemv_q8_0_moe_down_top10_indexed_batched_expanded",
            ),
            Fmt::Mq4 => (
                &OLD_MQ4_DOWN,
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
            ),
        };
        cx.load_old(file)?;
        let mut b = KernargBlob::new();
        b.push_ptr(ptr(&tab.table));
        b.push_ptr(ptr(&t_ids));
        b.push_ptr(ptr(&t_x));
        b.push_ptr(ptr(&t_y));
        b.push_i32(DOWN_M as i32);
        b.push_i32(DOWN_K as i32);
        b.push_i32(tokens as i32);
        b.push_i32(E as i32);
        cx.launch_old(
            sym,
            [DOWN_M as u32, K as u32, tokens as u32],
            [32, 1, 1],
            0,
            b,
        )?;
    } else {
        let launched = match fmt {
            Fmt::Q8 => cx.gpu.gemv_q8_0_moe_down_top10_indexed_batched_expanded(
                &tab.table, &t_ids, &t_x, &t_y, DOWN_M, DOWN_K, tokens, E,
            ),
            Fmt::Mq4 => cx
                .gpu
                .gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
                    &tab.table, &t_ids, &t_x, &t_y, DOWN_M, DOWN_K, tokens, E, 10,
                ),
        };
        launched.map_err(e)?;
    }
    sync(&cx.gpu)?;
    let out = bits(&down_f32(&cx.gpu, &t_y)?);
    free_all(&mut cx.gpu, vec![t_ids, t_x, t_y])?;
    Ok(out)
}

fn down_indexed_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E30);
    let mut results = Vec::new();
    for fmt in [Fmt::Q8, Fmt::Mq4] {
        let tab = build_experts(cx, &mut rng, fmt, DOWN_M, DOWN_K)?;
        for &tokens in &[1usize, 5] {
            let valid = make_topk(&mut rng, tokens, Spread::Uniform);
            let x = extreme_vec(&mut rng, tokens * K * DOWN_K, -2.0, 2.0);
            let tag = format!("{fmt:?} tokens={tokens}");
            let new = run_down_indexed(cx, fmt, &tab, &valid, &x, tokens, false)?;
            let old = run_down_indexed(cx, fmt, &tab, &valid, &x, tokens, true)?;
            results.push(exact_bits(&format!("{tag} valid new-vs-old"), &new, &old));
            for phase in 0..4 {
                let (ids, bad) = poison_topk(&valid, phase);
                let new_bad = run_down_indexed(cx, fmt, &tab, &ids, &x, tokens, false)?;
                // These two kernels already bounds-checked on origin/beta (old is defined on bad ids).
                let old_bad = run_down_indexed(cx, fmt, &tab, &ids, &x, tokens, true)?;
                results.push(exact_bits(
                    &format!("{tag} phase={phase} bad-ids new-vs-old"),
                    &new_bad,
                    &old_bad,
                ));
                let nonzero = bad.iter().enumerate().filter(|(_, b)| **b).any(|(s, _)| {
                    new_bad[s * DOWN_M..(s + 1) * DOWN_M]
                        .iter()
                        .any(|&v| v != 0)
                });
                results.push(if nonzero {
                    Err(format!(
                        "{tag} phase={phase}: an unrouted rank's expanded row is not +0"
                    ))
                } else {
                    Ok(format!("{tag} phase={phase}: unrouted rows are +0"))
                });
            }
        }
        tab.free(&mut cx.gpu)?;
    }
    pass(all(results))
}

// ───────────────────────────────────────────────────────────────────────────────
// Grouped down gemm (Q8_0 / MQ4G128V2)
// ───────────────────────────────────────────────────────────────────────────────
fn gfx11_plus(arch: &str) -> bool {
    arch.starts_with("gfx11") || arch.starts_with("gfx12")
}

fn run_grouped_down(
    cx: &mut Ctx,
    fmt: Fmt,
    tab: &ExpertTable,
    m: usize,
    sorted: &[i32],
    tiles: &[i32],
    x: &[f32],
    slots: usize,
    old: bool,
) -> R<Vec<u32>> {
    let gr = sorted.len();
    let t_tiles = up_i32(&mut cx.gpu, tiles)?;
    let t_sorted = up_i32(&mut cx.gpu, sorted)?;
    let t_x = up_f32(&mut cx.gpu, x)?;
    let t_y = sentinel_f32(&mut cx.gpu, gr * m)?;
    let k = DOWN_K;
    if old {
        let (file, sym, grid_x, block) = match fmt {
            Fmt::Q8 => (&OLD_Q8_GROUPED, "gemm_q8_0_moe_grouped_top10", m, 256u32),
            Fmt::Mq4 if gfx11_plus(&cx.arch) && m == 2560 && k == 640 => (
                &OLD_MQ4_O8R16,
                "gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151",
                m.div_ceil(8),
                32,
            ),
            Fmt::Mq4 => (
                &OLD_MQ4_MULTIROW,
                "gemm_mq4g128v2_moe_grouped_top10_multirow",
                m,
                32,
            ),
        };
        cx.load_old(file)?;
        let mut b = KernargBlob::new();
        b.push_ptr(ptr(&tab.table));
        b.push_ptr(ptr(&t_tiles));
        b.push_ptr(ptr(&t_sorted));
        b.push_ptr(ptr(&t_x));
        b.push_ptr(ptr(&t_y));
        for v in [m, k, 1, gr, slots, E] {
            b.push_i32(v as i32);
        }
        cx.launch_old(
            sym,
            [grid_x as u32, gr.div_ceil(16) as u32, 1],
            [block, 1, 1],
            0,
            b,
        )?;
    } else {
        let launched = match fmt {
            Fmt::Q8 => cx.gpu.gemm_q8_0_moe_grouped_top10(
                &tab.table, &t_tiles, &t_sorted, &t_x, &t_y, m, k, 1, gr, slots, E,
            ),
            Fmt::Mq4 => cx.gpu.gemm_mq4g128v2_moe_grouped_top10(
                &tab.table, &t_tiles, &t_sorted, &t_x, &t_y, m, k, 1, gr, slots, E,
            ),
        };
        launched.map_err(e)?;
    }
    sync(&cx.gpu)?;
    let out = bits(&down_f32(&cx.gpu, &t_y)?);
    free_all(&mut cx.gpu, vec![t_tiles, t_sorted, t_x, t_y])?;
    Ok(out)
}

fn down_grouped_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E40);
    let mut results = Vec::new();
    for fmt in [Fmt::Q8, Fmt::Mq4] {
        for &(m, rows) in &[(DOWN_M, 7usize), (DOWN_M, 33), (1280, 7)] {
            if fmt == Fmt::Q8 && m != DOWN_M {
                continue;
            }
            let tab = build_experts(cx, &mut rng, fmt, m, DOWN_K)?;
            let topk = make_topk(&mut rng, rows, Spread::Skewed);
            let slots = topk.len();
            let x = extreme_vec(&mut rng, slots * DOWN_K, -2.0, 2.0);
            let tag = format!("{fmt:?} m={m} rows={rows}");
            if fmt == Fmt::Mq4 && cx.gpu.qwen4_moe_down_wmma_applies(m, DOWN_K, slots) {
                cx.notes.push(format!("{tag}: F16 WMMA down route applies on this arch/shape; not bitwise-comparable, skipped"));
                tab.free(&mut cx.gpu)?;
                continue;
            }
            // Valid layout and layouts with unrouted slots: new == old, bitwise, whole Y.
            let mut valid_new = Vec::new();
            for phase in 0..5 {
                let ids = if phase == 4 {
                    topk.clone()
                } else {
                    poison_topk(&topk, phase).0
                };
                let lay = scatter_ref(&ids);
                let new =
                    run_grouped_down(cx, fmt, &tab, m, &lay.sorted, &lay.tiles, &x, slots, false)?;
                let old =
                    run_grouped_down(cx, fmt, &tab, m, &lay.sorted, &lay.tiles, &x, slots, true)?;
                results.push(exact_bits(
                    &format!("{tag} phase={phase} new-vs-old"),
                    &new,
                    &old,
                ));
                if phase == 4 {
                    valid_new = new;
                }
            }
            // Corrupt tile expert ids: no fault; untouched tiles bitwise as the valid run; the
            // corrupt tile's rows are not consumed (left as the sentinel or +0, never garbage).
            let lay = scatter_ref(&topk);
            let mut tiles = lay.tiles.clone();
            let live: Vec<usize> = (0..tiles.len()).filter(|&t| tiles[t] >= 0).collect();
            let mut corrupted = Vec::new();
            for (j, &t) in live.iter().enumerate().filter(|(j, _)| j % 5 == 2) {
                tiles[t] = BAD_IDS[(j / 5) % BAD_IDS.len()];
                corrupted.push(t);
            }
            let bad = run_grouped_down(cx, fmt, &tab, m, &lay.sorted, &tiles, &x, slots, false)?;
            let sentinel = nan_poison().to_bits();
            let mut diff = None;
            for t in 0..tiles.len() {
                let rows = &bad[t * BLOCK_M * m..(t + 1) * BLOCK_M * m];
                if corrupted.contains(&t) {
                    if rows.iter().any(|&v| v != sentinel && v != 0) {
                        diff = Some(format!(
                            "corrupt tile {t} (id {}) wrote non-zero data",
                            tiles[t]
                        ));
                        break;
                    }
                } else if rows != &valid_new[t * BLOCK_M * m..(t + 1) * BLOCK_M * m] {
                    diff = Some(format!("untouched tile {t} differs from the valid run"));
                    break;
                }
            }
            results.push(match diff {
                None => Ok(format!(
                    "{tag} corrupt-tile run ok ({} tiles corrupted)",
                    corrupted.len()
                )),
                Some(d) => Err(format!("{tag}: {d}")),
            });
            tab.free(&mut cx.gpu)?;
        }
    }
    pass(all(results))
}

// ───────────────────────────────────────────────────────────────────────────────
// Combines
// ───────────────────────────────────────────────────────────────────────────────
#[derive(Clone)]
struct CombineIn {
    ids: Vec<i32>,
    w: Vec<f32>,
    /// expanded rows `[tokens * 10, H]` (indexed combine)
    eo: Vec<f32>,
    res: Vec<f32>,
}

/// `valid`, the failed-closed `poison` (bad ids, NaN/Inf garbage behind them) and the
/// `reference` the contract defines for it: every unrouted rank is a rank with weight
/// 0 and a zero output row, i.e. contributes nothing.
fn combine_inputs(rng: &mut Rng, tokens: usize, phase: usize) -> (CombineIn, CombineIn, CombineIn) {
    let ids = make_topk(rng, tokens, Spread::Uniform);
    let slots = tokens * K;
    let w: Vec<f32> = extreme_vec(rng, slots, 0.0, 0.4)
        .into_iter()
        .map(|v| {
            if v.is_finite() && v.abs() < 1e3 {
                v.abs()
            } else {
                0.25
            }
        })
        .collect();
    let eo = extreme_vec(rng, slots * HIDDEN, -3.0, 3.0);
    let res: Vec<f32> = rng
        .vec(tokens * HIDDEN, -2.0, 2.0)
        .into_iter()
        .map(|v| if v == 0.0 { 0.5 } else { v })
        .collect();
    let valid = CombineIn {
        ids: ids.clone(),
        w,
        eo,
        res,
    };
    let (pids, bad) = poison_topk(&ids, phase);
    let mut poison = valid.clone();
    let mut reference = valid.clone();
    poison.ids = pids.clone();
    for (s, &is_bad) in bad.iter().enumerate() {
        if !is_bad {
            continue;
        }
        // A failed-closed router row carries weight 0; raw bad ids keep their nonzero weights.
        let r = s / K;
        if pids[r * K..(r + 1) * K].iter().all(|&v| v == FAIL_ID) {
            poison.w[s] = 0.0;
        }
        for c in 0..HIDDEN {
            poison.eo[s * HIDDEN + c] = if (s + c) % 2 == 0 {
                f32::NAN
            } else {
                f32::INFINITY
            };
            reference.eo[s * HIDDEN + c] = 0.0;
        }
        reference.ids[s] = 0;
        reference.w[s] = 0.0;
    }
    (valid, poison, reference)
}

fn run_combine_indexed(cx: &mut Ctx, inp: &CombineIn, tokens: usize, old: bool) -> R<Vec<u32>> {
    let t_eo = up_f32(&mut cx.gpu, &inp.eo)?;
    let t_ids = up_i32(&mut cx.gpu, &inp.ids)?;
    let t_w = up_f32(&mut cx.gpu, &inp.w)?;
    let t_res = up_f32(&mut cx.gpu, &inp.res)?;
    if old {
        cx.load_old(&OLD_COMBINE)?;
        let mut b = KernargBlob::new();
        b.push_ptr(ptr(&t_eo));
        b.push_ptr(ptr(&t_ids));
        b.push_ptr(ptr(&t_w));
        b.push_ptr(ptr(&t_res));
        b.push_i32(HIDDEN as i32);
        b.push_i32(tokens as i32);
        cx.launch_old(
            "moe_down_combine_top10_batched",
            [HIDDEN.div_ceil(64) as u32, tokens as u32, 1],
            [64, 1, 1],
            0,
            b,
        )?;
    } else {
        cx.gpu
            .moe_down_combine_top10_batched(&t_eo, &t_ids, &t_w, &t_res, HIDDEN, tokens, E)
            .map_err(e)?;
    }
    sync(&cx.gpu)?;
    let out = bits(&down_f32(&cx.gpu, &t_res)?);
    free_all(&mut cx.gpu, vec![t_eo, t_ids, t_w, t_res])?;
    Ok(out)
}

fn combine_indexed_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E50);
    let mut results = Vec::new();
    for &tokens in &[1usize, 7, 33] {
        for phase in 0..4 {
            let (valid, poison, reference) = combine_inputs(&mut rng, tokens, phase);
            let tag = format!("tokens={tokens} phase={phase}");
            let new_valid = run_combine_indexed(cx, &valid, tokens, false)?;
            let old_valid = run_combine_indexed(cx, &valid, tokens, true)?;
            results.push(exact_bits(
                &format!("{tag} valid new-vs-old"),
                &new_valid,
                &old_valid,
            ));
            // Poisoned run vs OLD on the contract's reference (unrouted rank == zero contribution).
            let new_poison = run_combine_indexed(cx, &poison, tokens, false)?;
            let old_ref = run_combine_indexed(cx, &reference, tokens, true)?;
            results.push(exact_bits(
                &format!("{tag} poisoned new vs old on reference"),
                &new_poison,
                &old_ref,
            ));
        }
    }
    pass(all(results))
}

fn run_combine_grouped(
    cx: &mut Ctx,
    bf16: bool,
    zinit: bool,
    inp: &CombineIn,
    inverse: &[i32],
    grouped_f32: &[f32],
    grouped_bf16: &[u16],
    gr: usize,
    tokens: usize,
    old: bool,
) -> R<Vec<u32>> {
    let t_ids = up_i32(&mut cx.gpu, &inp.ids)?;
    let t_inv = up_i32(&mut cx.gpu, inverse)?;
    let t_w = up_f32(&mut cx.gpu, &inp.w)?;
    let t_res = up_f32(&mut cx.gpu, &inp.res)?;
    let t_g = if bf16 {
        up_bytes(&mut cx.gpu, &u16_bytes(grouped_bf16))?
    } else {
        up_f32(&mut cx.gpu, grouped_f32)?
    };
    let mut order_buf = Vec::new();
    if bf16 {
        order_buf.push(up_bytes(&mut cx.gpu, &vec![0u8; tokens * K * 8])?);
    }
    if old {
        cx.load_old(&OLD_COMBINE_GROUPED)?;
        if !bf16 {
            let mut b = KernargBlob::new();
            b.push_ptr(ptr(&t_g));
            b.push_ptr(ptr(&t_inv));
            b.push_ptr(ptr(&t_ids));
            b.push_ptr(ptr(&t_w));
            b.push_ptr(ptr(&t_res));
            for v in [HIDDEN, gr, tokens] {
                b.push_i32(v as i32);
            }
            cx.launch_old(
                "moe_down_combine_grouped_top10",
                [HIDDEN.div_ceil(256) as u32, tokens as u32, 1],
                [256, 1, 1],
                0,
                b,
            )?;
        } else {
            let order = &order_buf[0];
            let mut o = KernargBlob::new();
            o.push_ptr(ptr(&t_inv));
            o.push_ptr(ptr(&t_ids));
            o.push_ptr(ptr(&t_w));
            o.push_ptr(ptr(order));
            o.push_i32(gr as i32);
            o.push_i32(tokens as i32);
            cx.launch_old(
                "moe_combine_order_top10",
                [tokens.div_ceil(64) as u32, 1, 1],
                [64, 1, 1],
                0,
                o,
            )?;
            let mut b = KernargBlob::new();
            b.push_ptr(ptr(&t_g));
            b.push_ptr(ptr(order));
            b.push_ptr(ptr(&t_res));
            b.push_i32(HIDDEN as i32);
            b.push_i32(tokens as i32);
            let sym = if zinit {
                "moe_down_combine_grouped_top10_bf16in_zinit"
            } else {
                "moe_down_combine_grouped_top10_bf16in"
            };
            cx.launch_old(
                sym,
                [(HIDDEN / 8).div_ceil(64) as u32, tokens as u32, 1],
                [64, 1, 1],
                0,
                b,
            )?;
        }
    } else if bf16 {
        cx.gpu
            .moe_down_combine_grouped_top10_bf16in(
                &t_g,
                &t_inv,
                &t_ids,
                &t_w,
                &t_res,
                &order_buf[0],
                HIDDEN,
                gr,
                tokens,
                zinit,
                E,
            )
            .map_err(e)?;
    } else {
        cx.gpu
            .moe_down_combine_grouped_top10(
                &t_g, &t_inv, &t_ids, &t_w, &t_res, HIDDEN, gr, tokens, E,
            )
            .map_err(e)?;
    }
    sync(&cx.gpu)?;
    let out = bits(&down_f32(&cx.gpu, &t_res)?);
    let mut free = vec![t_ids, t_inv, t_w, t_res, t_g];
    free.extend(order_buf);
    free_all(&mut cx.gpu, free)?;
    Ok(out)
}

fn combine_grouped_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E60);
    let mut results = Vec::new();
    for &tokens in &[1usize, 7, 33] {
        for phase in 0..4 {
            let (valid, poison, reference) = combine_inputs(&mut rng, tokens, phase);
            let lay_valid = scatter_ref(&valid.ids);
            let lay_poison = scatter_ref(&poison.ids);
            let gr = lay_poison.m_total_max;
            let gf = extreme_vec(&mut rng, gr * HIDDEN, -3.0, 3.0);
            let gb = bf16_bits_vec(&mut rng, gr * HIDDEN);
            for (bf16, zinit) in [(false, false), (true, false), (true, true)] {
                let tag = format!("tokens={tokens} phase={phase} bf16in={bf16} zinit={zinit}");
                let nv = run_combine_grouped(
                    cx,
                    bf16,
                    zinit,
                    &valid,
                    &lay_valid.inverse,
                    &gf,
                    &gb,
                    gr,
                    tokens,
                    false,
                )?;
                let ov = run_combine_grouped(
                    cx,
                    bf16,
                    zinit,
                    &valid,
                    &lay_valid.inverse,
                    &gf,
                    &gb,
                    gr,
                    tokens,
                    true,
                )?;
                results.push(exact_bits(&format!("{tag} valid new-vs-old"), &nv, &ov));
                // Poisoned ids + inverse -1 vs OLD on the zero-contribution reference over the same layout.
                let np = run_combine_grouped(
                    cx,
                    bf16,
                    zinit,
                    &poison,
                    &lay_poison.inverse,
                    &gf,
                    &gb,
                    gr,
                    tokens,
                    false,
                )?;
                let or = run_combine_grouped(
                    cx,
                    bf16,
                    zinit,
                    &reference,
                    &lay_poison.inverse,
                    &gf,
                    &gb,
                    gr,
                    tokens,
                    true,
                )?;
                results.push(exact_bits(
                    &format!("{tag} poisoned new vs old on reference"),
                    &np,
                    &or,
                ));
            }
        }
    }
    pass(all(results))
}

// ───────────────────────────────────────────────────────────────────────────────
// Full chain: router -> stable scatter -> grouped down -> grouped combine
// ───────────────────────────────────────────────────────────────────────────────
struct Chain {
    ids: Vec<i32>,
    out: Vec<u32>,
    fault: Option<RouterFault>,
}

/// One pass of the prefill MoE route over router logits: new = the production
/// launchers fed straight from the router's device tensors; old = the OLD router,
/// stable scatter, Q8 grouped down and grouped combine (their outputs carried
/// through host copies, which is bitwise the same data).
fn run_chain(
    cx: &mut Ctx,
    tab: &ExpertTable,
    logits: &[f32],
    x: &[f32],
    res: &[f32],
    rows: usize,
    fast: bool,
    old: bool,
) -> R<Chain> {
    let slots = rows * K;
    let gr = grouped_rows_bound(slots);
    let (norm, round) = (true, true);
    let (layer, row_base) = (5usize, 64usize);
    let mut fault = None;
    let t_ids;
    let t_w;
    let mut old_layout = None;
    let ids: Vec<i32>;
    if old {
        let (oi, ow) = run_router_old(cx, logits, rows, false, norm, round)?;
        t_ids = up_i32(&mut cx.gpu, &oi)?;
        t_w = up_u32(&mut cx.gpu, &ow)?;
        let lay = run_stable_old(cx, &oi)?;
        old_layout = Some((
            up_i32(&mut cx.gpu, &lay.sorted)?,
            up_i32(&mut cx.gpu, &lay.tiles)?,
            up_i32(&mut cx.gpu, &lay.inverse)?,
        ));
        ids = oi;
    } else {
        Arc::make_mut(&mut cx.gpu.flags).qwen4_router_fast = fast;
        cx.gpu.clear_router_fault();
        let t_l = up_f32(&mut cx.gpu, logits)?;
        t_ids = int_fill(&mut cx.gpu, slots, -2)?;
        t_w = sentinel_f32(&mut cx.gpu, slots)?;
        cx.gpu
            .moe_router_softmax_top10_f32(&t_l, &t_ids, &t_w, rows, norm, round, layer, row_base)
            .map_err(e)?;
        sync(&cx.gpu)?;
        fault = cx.gpu.peek_router_fault();
        cx.gpu.clear_router_fault();
        ids = down_i32(&cx.gpu, &t_ids)?;
        free_all(&mut cx.gpu, vec![t_l])?;
    }
    let b = scatter_bufs(&mut cx.gpu, slots, gr)?;
    if !old {
        cx.gpu
            .moe_scatter_stable_top10(
                &t_ids, &b.counts, &b.offsets, &b.sorted, &b.tiles, &b.inverse, slots, E, gr,
            )
            .map_err(e)?;
    }
    let t_x = up_f32(&mut cx.gpu, x)?;
    let t_y = sentinel_f32(&mut cx.gpu, gr * DOWN_M)?;
    let t_res = up_f32(&mut cx.gpu, res)?;
    {
        let (sorted, tiles, inverse) = match &old_layout {
            Some((s, t, i)) => (s, t, i),
            None => (&b.sorted, &b.tiles, &b.inverse),
        };
        if old {
            cx.load_old(&OLD_Q8_GROUPED)?;
            let mut k = KernargBlob::new();
            k.push_ptr(ptr(&tab.table));
            k.push_ptr(ptr(tiles));
            k.push_ptr(ptr(sorted));
            k.push_ptr(ptr(&t_x));
            k.push_ptr(ptr(&t_y));
            for v in [DOWN_M, DOWN_K, 1, gr, slots, E] {
                k.push_i32(v as i32);
            }
            cx.launch_old(
                "gemm_q8_0_moe_grouped_top10",
                [DOWN_M as u32, gr.div_ceil(16) as u32, 1],
                [256, 1, 1],
                0,
                k,
            )?;
            cx.load_old(&OLD_COMBINE_GROUPED)?;
            let mut c = KernargBlob::new();
            c.push_ptr(ptr(&t_y));
            c.push_ptr(ptr(inverse));
            c.push_ptr(ptr(&t_ids));
            c.push_ptr(ptr(&t_w));
            c.push_ptr(ptr(&t_res));
            for v in [HIDDEN, gr, rows] {
                c.push_i32(v as i32);
            }
            cx.launch_old(
                "moe_down_combine_grouped_top10",
                [HIDDEN.div_ceil(256) as u32, rows as u32, 1],
                [256, 1, 1],
                0,
                c,
            )?;
        } else {
            cx.gpu
                .gemm_q8_0_moe_grouped_top10(
                    &tab.table, tiles, sorted, &t_x, &t_y, DOWN_M, DOWN_K, 1, gr, slots, E,
                )
                .map_err(e)?;
            cx.gpu
                .moe_down_combine_grouped_top10(
                    &t_y, inverse, &t_ids, &t_w, &t_res, HIDDEN, gr, rows, E,
                )
                .map_err(e)?;
        }
    }
    sync(&cx.gpu)?;
    let out = bits(&down_f32(&cx.gpu, &t_res)?);
    let mut free = vec![t_ids, t_w, t_x, t_y, t_res];
    if let Some((s, t, i)) = old_layout {
        free.extend([s, t, i]);
    }
    free_all(&mut cx.gpu, free)?;
    b.free(&mut cx.gpu)?;
    Ok(Chain { ids, out, fault })
}

fn chain_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E70);
    let mut results = Vec::new();
    let tab = build_experts(cx, &mut rng, Fmt::Q8, DOWN_M, DOWN_K)?;
    let rows = 9usize;
    let slots = rows * K;
    let x = extreme_vec(&mut rng, slots * DOWN_K, -2.0, 2.0);
    let res: Vec<f32> = rng
        .vec(rows * HIDDEN, -2.0, 2.0)
        .into_iter()
        .map(|v| if v == 0.0 { 0.5 } else { v })
        .collect();
    for layout_id in 0..3 {
        let layout: Vec<Option<Bad>> = (0..rows)
            .map(|r| match layout_id {
                0 => (r % 3 == 1).then(|| BADS[r % BADS.len()]),
                1 => (r != 4).then(|| BADS[r % BADS.len()]),
                _ => None,
            })
            .collect();
        let mut logits = Vec::new();
        for (r, k) in layout.iter().enumerate() {
            logits.extend(match k {
                Some(kind) => bad_row(&mut rng, *kind),
                None => finite_row(&mut rng, r),
            });
        }
        let failing: Vec<bool> = (0..rows)
            .map(|r| row_fails(&logits[r * E..(r + 1) * E], true))
            .collect();
        let mut clean = logits.clone();
        for r in 0..rows {
            if failing[r] {
                clean[r * E..(r + 1) * E].copy_from_slice(&finite_row(&mut rng, r + 3));
            }
        }
        let old = run_chain(cx, &tab, &clean, &x, &res, rows, false, true)?;
        for fast in [false, true] {
            if fast && !cx.fast_router_arch() {
                continue;
            }
            let tag = format!("layout={layout_id} fast={fast}");
            let new_clean = run_chain(cx, &tab, &clean, &x, &res, rows, fast, false)?;
            results.push(exact_bits(
                &format!("{tag} clean chain output vs OLD chain"),
                &new_clean.out,
                &old.out,
            ));
            results.push(exact_ints(
                &format!("{tag} clean router ids vs OLD"),
                &new_clean.ids,
                &old.ids,
            ));
            let bad = run_chain(cx, &tab, &logits, &x, &res, rows, fast, false)?;
            // Finite rows: bitwise the OLD chain's rows. Failed rows: residual only (bf16 round trip).
            let mut diff = None;
            for r in 0..rows {
                let got = &bad.out[r * HIDDEN..(r + 1) * HIDDEN];
                if failing[r] {
                    let want: Vec<u32> = res[r * HIDDEN..(r + 1) * HIDDEN]
                        .iter()
                        .map(|&v| bf16_round(v).to_bits())
                        .collect();
                    if got != want.as_slice() {
                        diff = Some(format!("failed row {r}: output is not bf16(residual) (a stale/garbage expert row leaked)"));
                        break;
                    }
                } else if got != &old.out[r * HIDDEN..(r + 1) * HIDDEN] {
                    diff = Some(format!(
                        "finite row {r} differs from the OLD chain on the clean twin"
                    ));
                    break;
                }
            }
            results.push(match diff {
                None => Ok(format!(
                    "{tag} poisoned chain ok ({} failed rows)",
                    failing.iter().filter(|f| **f).count()
                )),
                Some(d) => Err(format!("{tag}: {d}")),
            });
            results.push(match (failing.iter().any(|f| *f), bad.fault) {
                (true, Some(f)) if f.layer == 5 => Ok(format!("{tag} fault layer 5 row {}", f.row)),
                (false, None) => Ok(format!("{tag} no fault")),
                (any, f) => Err(format!("{tag}: failing={any} fault={f:?}")),
            });
        }
    }
    tab.free(&mut cx.gpu)?;
    pass(all(results))
}

// ───────────────────────────────────────────────────────────────────────────────
// Second producer: top-10 renorm over pre-softmaxed probabilities (+ logits guard)
// ───────────────────────────────────────────────────────────────────────────────
const OLD_RENORM: OldFile = OldFile {
    file: "moe_topk_renorm_top10_batched.hip",
    syms: &["moe_topk_renorm_top10_batched"],
};

fn prob_row(rng: &mut Rng, variant: usize) -> Vec<f32> {
    let mut row: Vec<f32> = rng.vec(E, 0.0, 1.0);
    match variant % 4 {
        0 => row.iter_mut().for_each(|v| *v = 1.0 / 512.0),
        1 => row
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = if i % 2 == 0 { 0.0 } else { -0.0 }),
        2 => {
            for &i in &[3usize, 64, 130, 255, 256, 300, 511] {
                row[i] = 0.9;
            }
            row[17] = 1e-40;
        }
        _ => {}
    }
    row
}

fn run_renorm_old(cx: &mut Ctx, probs: &[f32], rows: usize, norm: bool) -> R<(Vec<i32>, Vec<u32>)> {
    cx.load_old(&OLD_RENORM)?;
    let t_p = up_f32(&mut cx.gpu, probs)?;
    let t_i = int_fill(&mut cx.gpu, rows * K, -2)?;
    let t_w = sentinel_f32(&mut cx.gpu, rows * K)?;
    let mut b = KernargBlob::new();
    b.push_ptr(ptr(&t_p));
    b.push_ptr(ptr(&t_i));
    b.push_ptr(ptr(&t_w));
    b.push_i32(E as i32);
    b.push_i32(i32::from(norm));
    cx.launch_old(
        "moe_topk_renorm_top10_batched",
        [rows as u32, 1, 1],
        [256, 1, 1],
        0,
        b,
    )?;
    sync(&cx.gpu)?;
    let ids = down_i32(&cx.gpu, &t_i)?;
    let w = bits(&down_f32(&cx.gpu, &t_w)?);
    free_all(&mut cx.gpu, vec![t_p, t_i, t_w])?;
    Ok((ids, w))
}

fn run_renorm_new(
    cx: &mut Ctx,
    probs: &[f32],
    rows: usize,
    norm: bool,
    guard: bool,
    layer: usize,
    row_base: usize,
) -> R<RouterOut> {
    cx.gpu.clear_router_fault();
    let t_p = up_f32(&mut cx.gpu, probs)?;
    let t_i = int_fill(&mut cx.gpu, rows * K, -2)?;
    let t_w = sentinel_f32(&mut cx.gpu, rows * K)?;
    if guard {
        cx.gpu
            .moe_router_logits_guard_top10(&t_p, E, rows, layer, row_base)
            .map_err(e)?;
    }
    cx.gpu
        .moe_topk_renorm_top10_batched(&t_p, &t_i, &t_w, E, norm, rows, layer, row_base)
        .map_err(e)?;
    sync(&cx.gpu)?;
    let fault = cx.gpu.peek_router_fault();
    cx.gpu.clear_router_fault();
    let ids = down_i32(&cx.gpu, &t_i)?;
    let w = bits(&down_f32(&cx.gpu, &t_w)?);
    free_all(&mut cx.gpu, vec![t_p, t_i, t_w])?;
    Ok(RouterOut { ids, w, fault })
}

fn renorm_case(cx: &mut Ctx) -> R<Outcome> {
    let mut rng = Rng::new(0x4E41_4E95);
    let mut results = Vec::new();
    let kinds = [
        Bad::AllNan,
        Bad::OneNan,
        Bad::PlusInf,
        Bad::MinusInf,
        Bad::AllPlusInf,
        Bad::NanAndInf,
    ];
    for &rows in &[1usize, 4, 9] {
        let layout: Vec<Option<Bad>> = (0..rows)
            .map(|r| (rows == 1 || r % 3 == 1).then(|| kinds[(r + rows) % kinds.len()]))
            .collect();
        let mut probs = Vec::new();
        for (r, k) in layout.iter().enumerate() {
            probs.extend(match k {
                Some(kind) => {
                    let mut row = prob_row(&mut rng, r);
                    let bad = bad_row(&mut rng, *kind);
                    for (v, b) in row.iter_mut().zip(&bad) {
                        if !b.is_finite() {
                            *v = *b;
                        }
                    }
                    row
                }
                None => prob_row(&mut rng, r),
            });
        }
        let failing: Vec<bool> = (0..rows)
            .map(|r| probs[r * E..(r + 1) * E].iter().any(|v| !v.is_finite()))
            .collect();
        let mut clean = probs.clone();
        for r in 0..rows {
            if failing[r] {
                clean[r * E..(r + 1) * E].copy_from_slice(&prob_row(&mut rng, r + 1));
            }
        }
        for norm in [true, false] {
            let tag = format!("rows={rows} norm={norm}");
            let (old_ids, old_w) = run_renorm_old(cx, &clean, rows, norm)?;
            let (layer, row_base) = (3usize, 700usize);
            for guard in [false, true] {
                let gtag = format!("{tag} guard={guard}");
                let new_clean = run_renorm_new(cx, &clean, rows, norm, guard, layer, row_base)?;
                results.push(exact_ints(
                    &format!("{gtag} clean ids vs old"),
                    &new_clean.ids,
                    &old_ids,
                ));
                results.push(exact_ints(
                    &format!("{gtag} clean w vs old"),
                    &new_clean.w,
                    &old_w,
                ));
                results.push(match new_clean.fault {
                    None => Ok(format!("{gtag} clean: no fault")),
                    Some(f) => Err(format!("{gtag} clean input recorded a fault {f:?}")),
                });
                // -Inf alone is only caught by the logits guard (a softmax output cannot be -Inf).
                let only_neg_inf = |r: usize| {
                    probs[r * E..(r + 1) * E]
                        .iter()
                        .all(|v| v.is_finite() || *v == f32::NEG_INFINITY)
                };
                let expect_fail: Vec<bool> = (0..rows)
                    .map(|r| failing[r] && (guard || !only_neg_inf(r)))
                    .collect();
                if !guard && (0..rows).any(|r| failing[r] && only_neg_inf(r)) {
                    continue; // unguarded renorm is not required to catch a -Inf-only row
                }
                let bad = run_renorm_new(cx, &probs, rows, norm, guard, layer, row_base)?;
                results.push(check_router_rows(
                    &format!("{gtag} poisoned"),
                    &bad,
                    &expect_fail,
                    &old_ids,
                    &old_w,
                ));
                let first = expect_fail.iter().position(|f| *f);
                results.push(match (first, bad.fault) {
                    (None, None) => Ok(format!("{gtag} no failing row, no fault")),
                    (Some(_), Some(f))
                        if f.layer == layer as u32
                            && expect_fail
                                .get((f.row as usize).wrapping_sub(row_base))
                                .copied()
                                .unwrap_or(false) =>
                    {
                        Ok(format!("{gtag} fault layer={} row={}", f.layer, f.row))
                    }
                    (a, f) => Err(format!("{gtag}: first failing row {a:?}, fault {f:?}")),
                });
            }
        }
    }
    pass(all(results))
}

// ───────────────────────────────────────────────────────────────────────────────
// PM (PeaceMaker) production images: old image vs new image
// ───────────────────────────────────────────────────────────────────────────────
const OLD_SLAB_PLAN: OldFile = OldFile {
    file: "qwen4_slab_union_plan.hip",
    syms: &["qwen4_slab_union_plan"],
};

fn build_experts_with(
    cx: &mut Ctx,
    rng: &mut Rng,
    gen: fn(&mut Rng, usize, usize) -> Vec<u8>,
    m: usize,
    k: usize,
) -> R<ExpertTable> {
    let mut real = Vec::with_capacity(N_REAL);
    for _ in 0..N_REAL {
        let bytes = gen(rng, m, k);
        real.push(up_bytes(&mut cx.gpu, &bytes)?);
    }
    ExpertTable::build(&mut cx.gpu, real)
}

impl Ctx {
    /// Bytes of a tracked file at the old ref (the PM `.hxaco` images), or from `--old-dir`.
    fn old_blob(&self, rel: &str) -> R<Vec<u8>> {
        if let Some(dir) = &self.old_dir {
            let name = std::path::Path::new(rel).file_name().ok_or("bad path")?;
            let p = dir.join(name);
            return std::fs::read(&p).map_err(|err| format!("--old-dir {}: {err}", p.display()));
        }
        let spec = format!("{}:{rel}", self.old_ref);
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.repo)
            .args(["show", &spec])
            .output()
            .map_err(|err| format!("git show {spec}: {err}"))?;
        if !out.status.success() {
            return Err(format!(
                "git show {spec} failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(out.stdout)
    }

    /// Load one OLD PM image (never through `Gpu::functions`: raw module + function handles).
    fn load_old_image(
        &mut self,
        rel: &str,
        syms: &[&str],
    ) -> R<(hip_bridge::Module, Vec<hip_bridge::Function>)> {
        let bytes = self.old_blob(rel)?;
        let sha = hex12(&Sha256::digest(&bytes));
        let module = self
            .gpu
            .hip
            .module_load_data(&bytes)
            .map_err(|err| format!("load old image {rel}: {err}"))?;
        let mut fns = Vec::new();
        for s in syms {
            fns.push(
                self.gpu
                    .hip
                    .module_get_function(&module, s)
                    .map_err(|err| format!("old image {rel}: symbol {s}: {err}"))?,
            );
        }
        self.notes.push(format!(
            "old image {rel} sha256={sha}.. ({}) symbols {syms:?}",
            self.old_ref
        ));
        Ok((module, fns))
    }

    fn launch_raw(
        &self,
        f: &hip_bridge::Function,
        grid: [u32; 3],
        block: [u32; 3],
        blob: KernargBlob,
    ) -> R<()> {
        let mut bytes = blob.into_vec();
        // SAFETY: the blob is the kernel's full kernarg struct built from live tensors.
        unsafe {
            self.gpu
                .hip
                .launch_kernel_blob(f, grid, block, 0, None, &mut bytes)
        }
        .map_err(|err| format!("raw launch: {err}"))
    }
}

struct K1Out {
    y0: Vec<u32>,
    y1: Vec<u32>,
}

fn k1_dims(down: bool, lanes: usize) -> (usize, usize, usize, usize, usize) {
    // (m, k, x_row_div, x_src_rows, y_stride)
    if down {
        (2560, 640, 1, K * lanes, 2560)
    } else {
        (1280, 2560, K, lanes, 640)
    }
}

/// One K1 bucket the harness launches: the union bucket (cold = 0, MQ4 = 1), the cold codec the
/// production launcher is told (`0` = an MQ4-only layer), the `[gate_up, down]` K1 symbols and the
/// f32 bit pattern the output buffers are pre-filled with (the NaN poison here; a finite canary in
/// `e8p_nan_containment`).
#[derive(Clone, Copy)]
struct K1Target {
    bucket: usize,
    cold_codec: u8,
    symbols: [&'static str; 2],
    canary: u32,
}

/// OLD planner (`qwen4_slab_union_plan.hip` on origin/beta, gfx1151/gfx1201 only), uniform `bucket`.
fn run_old_plan(cx: &mut Ctx, topk: &[i32], lanes: usize, bucket: usize) -> R<GpuTensor> {
    cx.load_old(&OLD_SLAB_PLAN)?;
    let s = K * lanes;
    let t_top = up_i32(&mut cx.gpu, topk)?;
    let out = up_u32(&mut cx.gpu, &vec![0xDEAD_BEEFu32; 6 * s + 2])?;
    let mut b = KernargBlob::new();
    b.push_ptr(ptr(&t_top));
    b.push_ptr(std::ptr::null());
    b.push_ptr(ptr(&out));
    b.push_i32(s as i32);
    b.push_i32(bucket as i32);
    cx.launch_old("qwen4_slab_union_plan", [1, 1, 1], [128, 1, 1], 0, b)?;
    sync(&cx.gpu)?;
    free_all(&mut cx.gpu, vec![t_top])?;
    Ok(out)
}

/// The production planner's (`Gpu::qwen4_slab_union_plan`, every slab arch) union of `topk` for the
/// uniform `bucket`, copied out of the slab scratch into its own tensor so a raw launch can read (or
/// have planted into) a stable descriptor. `6 * S + 2` words.
fn prod_union(cx: &mut Ctx, topk: &[i32], lanes: usize, bucket: usize) -> R<GpuTensor> {
    use rdna_compute::slab_moe::SlabBuckets;
    let s = K * lanes;
    let t_top = up_i32(&mut cx.gpu, topk)?;
    cx.gpu
        .qwen4_slab_union_plan(&t_top, SlabBuckets::Uniform(bucket), lanes)
        .map_err(e)?;
    sync(&cx.gpu)?;
    let words = 6 * s + 2;
    let base = cx
        .gpu
        .scratch
        .qwen4_slab_union_scratch
        .as_ref()
        .ok_or("the slab union scratch is not reserved")?;
    let view = GpuTensor {
        buf: base.byte_view(0, words * 4),
        shape: vec![words],
        dtype: rdna_compute::DType::F32,
    };
    let copy = down_u32(&cx.gpu, &view)?;
    free_all(&mut cx.gpu, vec![t_top])?;
    up_u32(&mut cx.gpu, &copy)
}

/// Word offsets `(ids, offsets, slot ids)` of bucket `bucket` inside the planner's `6 * S + 2` words.
fn k1_union_words(s: usize, bucket: usize) -> (usize, usize, usize) {
    (
        bucket * s,
        2 * s + bucket * (s + 1),
        2 * s + 2 * (s + 1) + bucket * s,
    )
}

/// Union pointers of bucket `bucket` inside the planner's output words.
fn k1_union_ptrs(
    union: &GpuTensor,
    s: usize,
    bucket: usize,
) -> (*const c_void, *const c_void, *const c_void) {
    let base = union.buf.as_ptr() as usize;
    let (ids, offs, slots) = k1_union_words(s, bucket);
    (
        (base + ids * 4) as *const c_void,
        (base + offs * 4) as *const c_void,
        (base + slots * 4) as *const c_void,
    )
}

/// `kernarg_extra`: `None` = the OLD 88-byte ABI; `Some(n)` = the NEW 96-byte ABI (`n_experts@88`, zero word@92).
fn k1_blob(
    down: bool,
    lanes: usize,
    bucket: usize,
    union: &GpuTensor,
    wtab: &GpuTensor,
    x: &GpuTensor,
    y0: &GpuTensor,
    y1: Option<&GpuTensor>,
    kernarg_extra: Option<u32>,
) -> KernargBlob {
    let s = K * lanes;
    let (m, k, div, srows, ys) = k1_dims(down, lanes);
    let (ids, offs, slots) = k1_union_ptrs(union, s, bucket);
    let mut b = KernargBlob::new();
    b.push_ptr(ptr(wtab));
    b.push_ptr(ids);
    b.push_ptr(offs);
    b.push_ptr(slots);
    b.push_ptr(ptr(x));
    b.push_ptr(ptr(y0));
    b.push_ptr(y1.map_or(std::ptr::null(), ptr));
    for w in [m, k, s, lanes, div, srows, k, ys] {
        b.push_u32(w as u32);
    }
    if let Some(n) = kernarg_extra {
        b.push_u32(n);
        b.push_u32(0);
    }
    b
}

/// One K1 projection through: the OLD image + OLD planner (`Old`, gfx1151/gfx1201), the OLD image
/// over the production planner's union (`OldImage`, every slab arch), the NEW production launcher +
/// NEW planner (`New`), the NEW launcher fed router ids that stay on the device (`NewDev`), or the
/// NEW image symbol launched raw over a caller union with a declared `n_experts` (`Raw`; the union
/// may carry out-of-range expert ids).
enum K1Path<'a> {
    Old(&'a hip_bridge::Function),
    OldImage(&'a hip_bridge::Function),
    New,
    NewDev(&'a GpuTensor),
    Raw(&'a GpuTensor, u32),
}

fn k1_run(
    cx: &mut Ctx,
    path: K1Path,
    tgt: K1Target,
    down: bool,
    topk: &[i32],
    lanes: usize,
    wtab: &ExpertTable,
    x: &[f32],
) -> R<K1Out> {
    use rdna_compute::slab_moe::SlabBuckets;
    let s = K * lanes;
    let (m, _, _, _, ys) = k1_dims(down, lanes);
    let canary = vec![f32::from_bits(tgt.canary); s * ys];
    let t_x = up_f32(&mut cx.gpu, x)?;
    let y0 = up_f32(&mut cx.gpu, &canary)?;
    let y1 = if down {
        None
    } else {
        Some(up_f32(&mut cx.gpu, &canary)?)
    };
    let mut extra = Vec::new();
    match path {
        K1Path::Old(f) => {
            let union = run_old_plan(cx, topk, lanes, tgt.bucket)?;
            let b = k1_blob(
                down,
                lanes,
                tgt.bucket,
                &union,
                &wtab.table,
                &t_x,
                &y0,
                y1.as_ref(),
                None,
            );
            cx.launch_raw(f, [m as u32, s as u32, 1], [32, 1, 1], b)?;
            extra.push(union);
        }
        K1Path::OldImage(f) => {
            let union = prod_union(cx, topk, lanes, tgt.bucket)?;
            let b = k1_blob(
                down,
                lanes,
                tgt.bucket,
                &union,
                &wtab.table,
                &t_x,
                &y0,
                y1.as_ref(),
                None,
            );
            cx.launch_raw(f, [m as u32, s as u32, 1], [32, 1, 1], b)?;
            extra.push(union);
        }
        K1Path::New => {
            let t_top = up_i32(&mut cx.gpu, topk)?;
            cx.gpu
                .qwen4_slab_union_plan(&t_top, SlabBuckets::Uniform(tgt.bucket), lanes)
                .map_err(e)?;
            cx.gpu
                .mq_i4_moe_gemv_slab(
                    down,
                    SlabBuckets::Uniform(tgt.bucket),
                    &wtab.table,
                    &t_x,
                    &y0,
                    y1.as_ref(),
                    lanes,
                    tgt.cold_codec,
                )
                .map_err(e)?;
            extra.push(t_top);
        }
        K1Path::NewDev(t_top) => {
            cx.gpu
                .qwen4_slab_union_plan(t_top, SlabBuckets::Uniform(tgt.bucket), lanes)
                .map_err(e)?;
            cx.gpu
                .mq_i4_moe_gemv_slab(
                    down,
                    SlabBuckets::Uniform(tgt.bucket),
                    &wtab.table,
                    &t_x,
                    &y0,
                    y1.as_ref(),
                    lanes,
                    tgt.cold_codec,
                )
                .map_err(e)?;
        }
        K1Path::Raw(union, n_experts) => {
            let sym = tgt.symbols[usize::from(down)];
            let blob = k1_blob(
                down,
                lanes,
                tgt.bucket,
                union,
                &wtab.table,
                &t_x,
                &y0,
                y1.as_ref(),
                Some(n_experts),
            );
            let mut bytes = blob.into_vec();
            cx.gpu
                .launch_kernel_blob(sym, [m as u32, s as u32, 1], [32, 1, 1], 0, &mut bytes)
                .map_err(|err| format!("raw launch of new {sym}: {err}"))?;
        }
    }
    sync(&cx.gpu)?;
    let o = K1Out {
        y0: bits(&down_f32(&cx.gpu, &y0)?),
        y1: match &y1 {
            Some(t) => bits(&down_f32(&cx.gpu, t)?),
            None => Vec::new(),
        },
    };
    let mut free = vec![t_x, y0];
    free.extend(y1);
    free.extend(extra);
    free_all(&mut cx.gpu, free)?;
    Ok(o)
}

/// Slot rows of `y` (stride `ys`) that are all the `canary` word.
fn rows_untouched(y: &[u32], ys: usize, slot: usize, canary: u32) -> bool {
    y[slot * ys..(slot + 1) * ys].iter().all(|&v| v == canary)
}

fn pm_k1_case(cx: &mut Ctx) -> R<Outcome> {
    use rdna_compute::slab_admission::SLAB_CODEC_MQ4;
    // The OLD image and planner are beta's, which exist at the pinned ref for gfx1151/gfx1201 only;
    // gfx1100's hot MQ4 K1 (and every E8P K1/K2) is compared against the rc4b images by
    // `e8p_nan_containment`.
    let beta_arch = matches!(cx.arch.as_str(), "gfx1151" | "gfx1201");
    let slab = rdna_compute::slab_moe::slab_arch_kernels(&cx.arch)
        .filter(|_| beta_arch && rdna_compute::qwen4_slab_codec_admitted(&cx.arch, SLAB_CODEC_MQ4));
    let Some(slab) = slab else {
        return Ok(Outcome::Skip(format!(
            "the beta-pinned expert-slab K1 comparison is gfx1151/gfx1201-only, device is {}",
            cx.arch
        )));
    };
    let tgt = K1Target {
        bucket: 1,
        cold_codec: SLAB_CODEC_MQ4,
        symbols: slab.mq4_k1.symbols,
        canary: nan_poison().to_bits(),
    };
    let mut rng = Rng::new(0x4E41_4E80);
    let mut results = Vec::new();
    cx.gpu
        .ensure_qwen4_slab_kernels(SLAB_CODEC_MQ4)
        .map_err(e)?;
    cx.gpu.reserve_qwen4_slab(0).map_err(e)?;
    let rel = format!("kernels/{}.hxaco", slab.mq4_k1.module);
    let (_module, fns) = cx.load_old_image(&rel, &tgt.symbols)?;
    let gate = build_experts_with(cx, &mut rng, mq4_gate_expert, 1280, 2560)?;
    let down = build_experts_with(cx, &mut rng, mq4_down_expert, 2560, 640)?;
    for &lanes in &[1usize, 3, 8] {
        let s = K * lanes;
        let topk = make_topk(&mut rng, lanes, Spread::Uniform);
        for is_down in [false, true] {
            let (_, k, _, srows, ys) = k1_dims(is_down, lanes);
            let wtab = if is_down { &down } else { &gate };
            let f = &fns[usize::from(is_down)];
            let x = extreme_vec(&mut rng, srows * k, -2.0, 2.0);
            let tag = format!("{} lanes={lanes}", if is_down { "down" } else { "gate_up" });
            // 1. finite identity: NEW image + NEW planner vs OLD image + OLD planner.
            let old = k1_run(cx, K1Path::Old(f), tgt, is_down, &topk, lanes, wtab, &x)?;
            let new = k1_run(cx, K1Path::New, tgt, is_down, &topk, lanes, wtab, &x)?;
            results.push(exact_bits(
                &format!("PM K1 {tag} y0 new-vs-old"),
                &new.y0,
                &old.y0,
            ));
            results.push(exact_bits(
                &format!("PM K1 {tag} y1 new-vs-old"),
                &new.y1,
                &old.y1,
            ));
            // 2. unrouted slots through the NEW planner: their rows stay unwritten, routed rows == OLD on the
            //    valid twin (the old planner remapped bad ids to expert 0; those slots are excluded).
            for phase in 0..4 {
                let (bad_ids, bad) = poison_topk(&topk, phase);
                let new_bad = k1_run(cx, K1Path::New, tgt, is_down, &bad_ids, lanes, wtab, &x)?;
                let mut diff = None;
                for slot in 0..s {
                    let (a, b) = (
                        &new_bad.y0[slot * ys..(slot + 1) * ys],
                        &old.y0[slot * ys..(slot + 1) * ys],
                    );
                    if bad[slot] {
                        if !rows_untouched(&new_bad.y0, ys, slot, tgt.canary) {
                            diff = Some(format!("unrouted slot {slot} was written"));
                            break;
                        }
                    } else if a != b {
                        diff = Some(format!("routed slot {slot} differs from the old pipeline"));
                        break;
                    }
                }
                results.push(match diff {
                    None => Ok(format!("PM K1 {tag} phase={phase} bad-id topk: routed rows == old, unrouted untouched")),
                    Some(d) => Err(format!("PM K1 {tag} phase={phase}: {d}")),
                });
            }
            // 3. raw bad ids planted in the union itself (NEW image, 96-byte kernarg): the image must
            //    exit before indexing the expert table. Experts of the first slot get each bad id.
            for &bad_id in &BAD_IDS {
                let union = run_old_plan(cx, &topk, lanes, tgt.bucket)?;
                let mut words = down_u32(&cx.gpu, &union)?;
                // ids[b=1][e=0] (word S + 0) is the first MQ4 expert; its slot run is offs[1][0..1].
                words[s] = bad_id as u32;
                let first_run_end = words[2 * s + (s + 1) + 1] as usize;
                let planted: Vec<usize> = (0..first_run_end)
                    .map(|p| words[2 * s + 2 * (s + 1) + s + p] as usize)
                    .collect();
                let t_union = up_u32(&mut cx.gpu, &words)?;
                let raw = k1_run(
                    cx,
                    K1Path::Raw(&t_union, E as u32),
                    tgt,
                    is_down,
                    &topk,
                    lanes,
                    wtab,
                    &x,
                )?;
                let mut diff = None;
                for slot in 0..s {
                    if planted.contains(&slot) {
                        if !rows_untouched(&raw.y0, ys, slot, tgt.canary) {
                            diff = Some(format!("slot {slot} of the bad-id expert was written"));
                            break;
                        }
                    } else if raw.y0[slot * ys..(slot + 1) * ys]
                        != old.y0[slot * ys..(slot + 1) * ys]
                    {
                        diff = Some(format!("slot {slot} differs from the old pipeline"));
                        break;
                    }
                }
                results.push(match diff {
                    None => Ok(format!("PM K1 {tag} raw union id {bad_id}: no fault, {} slots untouched, others == old", planted.len())),
                    Some(d) => Err(format!("PM K1 {tag} raw union id {bad_id}: {d}")),
                });
                free_all(&mut cx.gpu, vec![union, t_union])?;
            }
        }
    }
    gate.free(&mut cx.gpu)?;
    down.free(&mut cx.gpu)?;
    pass(all(results))
}

/// Grouped IU4 symmetric GEMMs (`gemm_qwen4_moe_gate_up_silu_iu4_sym`, `_down_iu4_sym`, PM images):
/// OLD image (60-byte kernarg) vs the NEW production launcher (64-byte), same sidecar.
fn pm_iu4_case(cx: &mut Ctx) -> R<Outcome> {
    // The case's real shapes: `rows` tokens x top-K routed slots over the E-entry expert table.
    // The old image is loaded once, so the production selector must name the same entries for every
    // row count the case runs.
    const PM_IU4_ROWS: [usize; 2] = [7, 33];
    let symbols_for = |cx: &Ctx, rows: usize| cx.gpu.qwen4_moe_sym_gemm_symbols(false, rows * K, E);
    let Some([gate_sym, down_sym]) = symbols_for(cx, PM_IU4_ROWS[0]) else {
        return Ok(Outcome::Skip(format!(
            "symmetric IU4 MoE route is gfx1151/gfx1201-only, device is {}",
            cx.arch
        )));
    };
    for rows in PM_IU4_ROWS {
        if symbols_for(cx, rows) != Some([gate_sym, down_sym]) {
            return Err(format!(
                "the production selector names different PM IU4 entries at rows={rows} than at \
                 rows={}: the single old-image load would compare the wrong kernels",
                PM_IU4_ROWS[0]
            ));
        }
    }
    let mut rng = Rng::new(0x4E41_4E90);
    let mut results = Vec::new();
    let rel = format!("kernels/qwen4_moe_iu4_sym_pm_{}.hxaco", cx.arch);
    let (_module, fns) = cx.load_old_image(&rel, &[gate_sym, down_sym])?;
    let (gm, gk, dm, dk) = (1280usize, 2560usize, 2560usize, 640usize);
    let gate = build_experts_with(cx, &mut rng, mq4_gate_expert, gm, gk)?;
    let down = build_experts_with(cx, &mut rng, mq4_down_expert, dm, dk)?;
    let sentinel_bytes = |n: usize| vec![0xA5u8; n];
    for rows in PM_IU4_ROWS {
        let topk = make_topk(&mut rng, rows, Spread::Skewed);
        let slots = topk.len();
        let lay = scatter_ref(&topk);
        let gr = lay.m_total_max;
        let t_sorted = up_i32(&mut cx.gpu, &lay.sorted)?;
        for is_down in [false, true] {
            let (m, k, div, src_rows, wtab, f) = if is_down {
                (dm, dk, 1usize, slots, &down, &fns[1])
            } else {
                (gm, gk, K, rows, &gate, &fns[0])
            };
            let y_bytes = iu4_y_bytes(gr, m, is_down);
            // gate/up: F32 activations `[rows, k]` (rotate256 producer). down: the grouped BF16
            // SwiGLU rows `[grouped_rows, k]` the real route hands the rotate128 producer (padding
            // rows carry data too; `sorted == -1` makes the producer skip them).
            let t_x = if is_down {
                let h = bf16_bits_vec(&mut rng, gr * k);
                up_bytes(&mut cx.gpu, &u16_bytes(&h))?
            } else {
                let x: Vec<f32> = extreme_vec(&mut rng, src_rows * k, -2.0, 2.0);
                up_f32(&mut cx.gpu, &x)?
            };
            let tag = format!("{} rows={rows}", if is_down { "down" } else { "gate_up" });
            // Sidecar once; both launches read it.
            let xq = if is_down {
                cx.gpu
                    .qwen4_moe_rotate128_i4(&t_x, &t_sorted, k, gr, slots)
                    .map_err(e)?
            } else {
                cx.gpu.qwen4_moe_rotate256_i4(&t_x, k, rows).map_err(e)?
            };
            let (generation, live) = if is_down {
                cx.gpu.scratch.qwen4_moe_down_i4_live()
            } else {
                cx.gpu.scratch.int4_mmq_live()
            };
            let xp = xq.checked_ptr(generation, live, k, src_rows).map_err(e)?;
            let run = |cx: &mut Ctx, tiles: &[i32], old: bool| -> R<Vec<u8>> {
                let t_tiles = up_i32(&mut cx.gpu, tiles)?;
                let t_y = up_bytes(&mut cx.gpu, &sentinel_bytes(y_bytes))?;
                if old {
                    let mut b = KernargBlob::new();
                    b.push_ptr(ptr(&wtab.table));
                    b.push_ptr(ptr(&t_tiles));
                    b.push_ptr(ptr(&t_sorted));
                    b.push_ptr(xp as *const c_void);
                    b.push_ptr(ptr(&t_y));
                    for v in [m, k, div, gr, src_rows] {
                        b.push_i32(v as i32);
                    }
                    let rr = if is_down {
                        cx.gpu.qwen4_moe_sym_down_rr().map_err(e)?
                    } else {
                        1
                    };
                    cx.launch_raw(
                        f,
                        [(m / (64 * rr)) as u32, gr.div_ceil(16) as u32, 1],
                        [128, 1, 1],
                        b,
                    )?;
                } else if is_down {
                    cx.gpu
                        .gemm_qwen4_moe_down_iu4_sym(
                            &wtab.table,
                            &t_tiles,
                            &t_sorted,
                            &xq,
                            &t_y,
                            m,
                            k,
                            div,
                            gr,
                            src_rows,
                            false,
                        )
                        .map_err(e)?;
                } else {
                    cx.gpu
                        .gemm_qwen4_moe_gate_up_silu_iu4_sym(
                            &wtab.table,
                            &t_tiles,
                            &t_sorted,
                            &xq,
                            &t_y,
                            m,
                            k,
                            div,
                            gr,
                            src_rows,
                            false,
                        )
                        .map_err(e)?;
                }
                sync(&cx.gpu)?;
                let out = down_bytes(&cx.gpu, &t_y)?;
                free_all(&mut cx.gpu, vec![t_tiles, t_y])?;
                Ok(out)
            };
            let new = run(cx, &lay.tiles, false)?;
            let old = run(cx, &lay.tiles, true)?;
            results.push(exact_ints(
                &format!("PM IU4 {tag} new-vs-old bytes"),
                &new,
                &old,
            ));
            // Corrupt tile expert ids (new only): no fault, the tile's rows are never written.
            let mut tiles = lay.tiles.clone();
            let live_tiles: Vec<usize> = (0..tiles.len()).filter(|&t| tiles[t] >= 0).collect();
            let mut corrupted = Vec::new();
            for (j, &t) in live_tiles.iter().enumerate().filter(|(j, _)| j % 4 == 1) {
                tiles[t] = BAD_IDS[(j / 4) % BAD_IDS.len()];
                corrupted.push(t);
            }
            let bad = run(cx, &tiles, false)?;
            let tile_bytes = iu4_tile_bytes(gr, m, is_down);
            assert_eq!(
                (new.len(), bad.len(), tiles.len() * tile_bytes),
                (y_bytes, y_bytes, y_bytes),
                "IU4 output traversal must cover the buffer exactly"
            );
            let mut diff = None;
            for t in 0..tiles.len() {
                let (a, b) = (
                    &bad[t * tile_bytes..(t + 1) * tile_bytes],
                    &new[t * tile_bytes..(t + 1) * tile_bytes],
                );
                if corrupted.contains(&t) {
                    if a.iter().any(|&v| v != 0xA5) {
                        diff = Some(format!("corrupt tile {t} (id {}) was written", tiles[t]));
                        break;
                    }
                } else if a != b {
                    diff = Some(format!("untouched tile {t} differs from the valid run"));
                    break;
                }
            }
            results.push(match diff {
                None => Ok(format!(
                    "PM IU4 {tag} corrupt tiles ({}) left unwritten, rest == valid run",
                    corrupted.len()
                )),
                Some(d) => Err(format!("PM IU4 {tag}: {d}")),
            });
            free_all(&mut cx.gpu, vec![t_x])?;
        }
        free_all(&mut cx.gpu, vec![t_sorted])?;
    }
    gate.free(&mut cx.gpu)?;
    down.free(&mut cx.gpu)?;
    pass(all(results))
}

// ───────────────────────────────────────────────────────────────────────────────
// Entry points: child, in-process, supervisor
// ───────────────────────────────────────────────────────────────────────────────
fn repo_root(args: &Args) -> PathBuf {
    args.repo
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

fn git_rev(repo: &std::path::Path, rev: &str) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", rev])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn make_ctx(args: &Args) -> R<Ctx> {
    let repo = repo_root(args);
    // Receipt: which baseline the OLD kernels are.
    let old_sha = if args.old_dir.is_some() {
        None
    } else {
        git_rev(&repo, &args.old_ref)
    };
    if args.old_dir.is_none() {
        let sha = old_sha.clone().ok_or_else(|| {
            format!(
                "git rev-parse {} failed in {}",
                args.old_ref,
                repo.display()
            )
        })?;
        if args.expect_old_sha != "any" && !sha.starts_with(&args.expect_old_sha) {
            return Err(format!(
                "old ref {} is {sha}, expected a commit starting {} (pass --expect-old-sha any to override)",
                args.old_ref, args.expect_old_sha
            ));
        }
    }
    let gpu = match args.device {
        Some(d) => Gpu::init_with_device(d),
        None => Gpu::init(),
    }
    .map_err(e)?;
    let arch = gpu.arch.clone();
    if let Some(want) = &args.expect_arch {
        if *want != arch {
            return Err(format!("device arch {arch}, expected {want}"));
        }
    }
    println!(
        "DEVICE arch={arch} old_ref={} old_sha={} old_dir={:?} fast_router_arch={}",
        args.old_ref,
        old_sha.as_deref().unwrap_or("-"),
        args.old_dir,
        matches!(arch.as_str(), "gfx1151" | "gfx1201"),
    );
    Ok(Ctx {
        gpu,
        arch,
        old_ref: args.old_ref.clone(),
        old_dir: args.old_dir.clone(),
        repo,
        out_dir: args.out_dir.clone(),
        loaded: HashSet::new(),
        notes: Vec::new(),
    })
}

/// Exit codes of a child: 0 pass, 1 fail, 77 skip.
fn run_cases(cx: &mut Ctx, cases: &[&Case]) -> i32 {
    let mut failed = 0usize;
    for case in cases {
        let start = Instant::now();
        println!("CASE {} START", case.name);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (case.run)(cx)));
        let secs = start.elapsed().as_secs_f32();
        let notes = std::mem::take(&mut cx.notes);
        let verdict = match res {
            Ok(Ok(Outcome::Pass(s))) => format!("PASS ({secs:.1}s): {s}"),
            Ok(Ok(Outcome::Skip(s))) => format!("SKIP ({secs:.1}s): {s}"),
            Ok(Err(s)) => {
                failed += 1;
                format!("FAIL ({secs:.1}s): {s}")
            }
            Err(_) => {
                failed += 1;
                format!("FAIL ({secs:.1}s): panic")
            }
        };
        for n in notes {
            println!("CASE {} NOTE {n}", case.name);
        }
        println!("CASE {} {verdict}", case.name);
    }
    i32::from(failed > 0)
}

fn selected_cases(args: &Args) -> Vec<&'static Case> {
    CASES
        .iter()
        .filter(|c| args.only.is_empty() || args.only.iter().any(|o| c.name.contains(o.as_str())))
        .collect()
}

fn run_child(args: &Args, name: &str) -> i32 {
    let Some(case) = CASES.iter().find(|c| c.name == name) else {
        eprintln!("unknown case {name}");
        return 2;
    };
    let mut cx = match make_ctx(args) {
        Ok(cx) => cx,
        Err(err) => {
            println!("CASE {name} FAIL: init: {err}");
            return 1;
        }
    };
    run_cases(&mut cx, &[case])
}

fn supervise(args: &Args, cases: &[&Case]) -> i32 {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(err) => {
            eprintln!("current_exe: {err}");
            return 2;
        }
    };
    let start = Instant::now();
    let budget = Duration::from_secs(args.budget_s);
    let mut rows: Vec<(String, String)> = Vec::new();
    let mut bad = 0usize;
    for case in cases {
        let left = budget.saturating_sub(start.elapsed());
        if left.is_zero() {
            rows.push((case.name.into(), "NOT RUN (budget exhausted)".into()));
            bad += 1;
            continue;
        }
        let limit = left.min(Duration::from_secs(args.case_timeout_s));
        let mut cmd = Command::new(&exe);
        cmd.args([
            "--case",
            case.name,
            "--old-ref",
            args.old_ref.as_str(),
            "--expect-old-sha",
            args.expect_old_sha.as_str(),
        ])
        .stdin(Stdio::null());
        if let Some(d) = &args.old_dir {
            cmd.arg("--old-dir").arg(d);
        }
        if let Some(r) = &args.repo {
            cmd.arg("--repo").arg(r);
        }
        cmd.arg("--out-dir").arg(&args.out_dir);
        if let Some(d) = args.device {
            cmd.args(["--device", d.to_string().as_str()]);
        }
        if let Some(a) = &args.expect_arch {
            cmd.args(["--expect-arch", a.as_str()]);
        }
        let t0 = Instant::now();
        let status = match cmd.spawn() {
            Ok(mut child) => loop {
                match child.try_wait() {
                    Ok(Some(st)) => break Ok(st),
                    Ok(None) if t0.elapsed() > limit => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break Err(format!("TIMEOUT after {}s", limit.as_secs()));
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(100)),
                    Err(err) => break Err(format!("wait: {err}")),
                }
            },
            Err(err) => Err(format!("spawn: {err}")),
        };
        let verdict = match status {
            Ok(st) if st.success() => "PASS/SKIP (see CASE line)".to_owned(),
            Ok(st) => {
                bad += 1;
                match st.code() {
                    Some(1) => "FAIL".to_owned(),
                    Some(c) => format!("CRASH exit {c}"),
                    None => "CRASH (killed by signal: device fault?)".to_owned(),
                }
            }
            Err(err) => {
                bad += 1;
                err
            }
        };
        rows.push((
            case.name.to_owned(),
            format!("{verdict} [{:.1}s]", t0.elapsed().as_secs_f32()),
        ));
    }
    println!(
        "\n==== moe_nan_guard_regress summary ({:.0}s of {}s budget) ====",
        start.elapsed().as_secs_f32(),
        args.budget_s
    );
    for (n, v) in &rows {
        println!("{n:<22} {v}");
    }
    println!(
        "{}",
        if bad == 0 {
            "RESULT: ALL CASES PASSED"
        } else {
            "RESULT: FAILURES PRESENT"
        }
    );
    i32::from(bad > 0)
}

fn main() {
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(err) => {
            eprintln!("moe_nan_guard_regress: {err}");
            std::process::exit(2);
        }
    };
    if args.list {
        for c in CASES {
            println!("{:<22} {}", c.name, c.covers);
        }
        return;
    }
    let code = if let Some(name) = &args.case {
        run_child(&args, name)
    } else if args.inproc {
        match make_ctx(&args) {
            Ok(mut cx) => run_cases(&mut cx, &selected_cases(&args)),
            Err(err) => {
                eprintln!("moe_nan_guard_regress: init: {err}");
                1
            }
        }
    } else {
        supervise(&args, &selected_cases(&args))
    };
    std::process::exit(code);
}

// ───────────────────────────────────────────────────────────────────────────────
// CPU tests (no GPU): `cargo test -p hipfire-runtime --release --features lab --example moe_nan_guard_regress`
// ───────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn src(file: &str) -> String {
        std::fs::read_to_string(repo().join("kernels/src").join(file))
            .unwrap_or_else(|err| panic!("{file}: {err}"))
    }

    fn old_src(file: &str) -> Option<String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(repo())
            .args(["show", &format!("{OLD_REF_PIN}:kernels/src/{file}")])
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    }

    #[test]
    fn case_table_is_consistent() {
        let mut names: Vec<&str> = CASES.iter().map(|c| c.name).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate case names");
        for c in CASES {
            assert!(!c.covers.is_empty(), "{} has no coverage text", c.name);
        }
        for need in [
            "router_nonfinite",
            "renorm_producer",
            "scatter_stable",
            "combine_indexed",
            "combine_grouped",
            "pm_k1_gemv",
            "pm_iu4_sym_grouped",
            "e8p_nan_containment",
        ] {
            assert!(CASES.iter().any(|c| c.name == need), "missing case {need}");
        }
    }

    #[test]
    fn args_parse_and_reject() {
        let a = parse_args(
            [
                "--only",
                "router",
                "--device",
                "1",
                "--expect-arch",
                "gfx1201",
                "--inproc",
            ]
            .map(String::from),
        )
        .unwrap();
        assert_eq!(
            (
                a.device,
                a.inproc,
                a.old_ref.as_str(),
                a.expect_old_sha.as_str()
            ),
            (Some(1), true, "f1a457b", "f1a457b")
        );
        assert_eq!(a.out_dir, PathBuf::from("."));
        let o = parse_args(["--out-dir", "x/y"].map(String::from)).unwrap();
        assert_eq!(o.out_dir, PathBuf::from("x/y"));
        assert!(parse_args(["--nope"].map(String::from)).is_err());
        assert!(parse_args(["--device"].map(String::from)).is_err());
    }

    #[test]
    fn router_fault_encoding_error_names_layer_and_row() {
        // The device word: ((row + 1) << 24) | (layer + 1), earliest row then earliest layer wins.
        let code = RouterFault::encode(7, 42);
        assert_eq!(code, (43u64 << 24) | 8);
        let f = RouterFault::decode(0, code).unwrap();
        assert_eq!((f.layer, f.row), (7, 42));
        assert!(RouterFault::decode(0, 0).is_none(), "0 is the clear word");
        assert!(
            RouterFault::encode(40, 5) < RouterFault::encode(0, 6),
            "row before layer"
        );
        assert!(
            RouterFault::encode(2, 5) < RouterFault::encode(3, 5),
            "same row: earlier layer"
        );
        // The request-failing error carries layer and row in words, not just a marker.
        let msg = f.to_hip_error().to_string();
        for needle in [
            "layer 7",
            "row 42",
            "NaN or Inf",
            "failed closed",
            "request was aborted",
        ] {
            assert!(msg.contains(needle), "{needle:?} missing from {msg:?}");
        }
        // Round trip over the whole contract range, including the row_base offset.
        for (layer, row_base, local) in [(0u32, 0u64, 0u64), (39, 1 << 20, 17), (1000, 5, 1)] {
            let f = RouterFault::decode(3, RouterFault::encode(layer, row_base + local)).unwrap();
            assert_eq!((f.device_id, f.layer, f.row), (3, layer, row_base + local));
        }
    }

    #[test]
    fn iu4_output_tiles_cover_the_buffer_exactly() {
        let mut rng = Rng::new(21);
        for rows in [7usize, 33] {
            let l = scatter_ref(&make_topk(&mut rng, rows, Spread::Skewed));
            let gr = l.m_total_max;
            for (m, down) in [(1280usize, false), (2560, true)] {
                assert_eq!(
                    l.tiles.len() * iu4_tile_bytes(gr, m, down),
                    iu4_y_bytes(gr, m, down),
                    "rows={rows} down={down}"
                );
            }
        }
        // The sentinel buffer the harness uploads must be representable word-for-word.
        assert_eq!(iu4_y_bytes(1120, 1280, false) % 4, 0);
        assert_eq!(iu4_y_bytes(1120, 2560, true) % 4, 0);
    }

    #[test]
    fn scatter_model_drops_unrouted_slots() {
        let mut ids: Vec<i32> = (0..20).collect();
        ids[3] = -1;
        ids[7] = 512;
        ids[11] = i32::MAX;
        ids[13] = i32::MIN;
        let l = scatter_ref(&ids);
        assert_eq!(l.raw.iter().sum::<usize>(), 16);
        for bad in [3usize, 7, 11, 13] {
            assert_eq!(l.inverse[bad], -1, "slot {bad}");
        }
        let routed = l.sorted.iter().filter(|&&v| v >= 0).count();
        assert_eq!(routed, 16);
        assert!(l.sorted.iter().all(|&v| v < 0 || valid_id(ids[v as usize])));
        // Every live tile carries a valid expert and pad16 offsets.
        assert!(l
            .offsets
            .windows(2)
            .all(|w| w[1] >= w[0] && (w[1] - w[0]) % 16 == 0));
        assert!(l.tiles.iter().all(|&t| t == -1 || valid_id(t)));
        assert_eq!(l.m_total_max % BLOCK_M, 0);
    }

    #[test]
    fn poison_masks_and_row_expectations() {
        let mut rng = Rng::new(7);
        let topk = make_topk(&mut rng, 9, Spread::Uniform);
        for phase in 0..4 {
            let (ids, bad) = poison_topk(&topk, phase);
            assert_eq!(ids.len(), topk.len());
            for (i, (&id, &b)) in ids.iter().zip(&bad).enumerate() {
                assert_eq!(b, !valid_id(id), "slot {i}");
            }
            assert!(*bad.last().unwrap(), "final slot is always unrouted");
            assert!(bad.iter().any(|b| *b) && bad.iter().any(|b| !*b));
        }
        // Router-row predicate: NaN/Inf always fail, BF16 rounding can promote a huge finite logit.
        let mut row = vec![0.5f32; E];
        assert!(!row_fails(&row, true));
        row[9] = 3.4028235e38;
        assert!(!row_fails(&row, false));
        assert!(
            row_fails(&row, true),
            "f32::MAX-class logits overflow the BF16 round trip"
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut r = vec![1.0f32; E];
            r[300] = bad;
            assert!(row_fails(&r, false) && row_fails(&r, true));
        }
    }

    #[test]
    fn combine_reference_zeroes_unrouted_ranks() {
        let mut rng = Rng::new(11);
        for phase in 0..4 {
            let (valid, poison, reference) = combine_inputs(&mut rng, 3, phase);
            for s in 0..poison.ids.len() {
                if valid_id(poison.ids[s]) {
                    assert_eq!(reference.ids[s], valid.ids[s]);
                    assert_eq!(reference.w[s].to_bits(), valid.w[s].to_bits());
                } else {
                    assert_eq!((reference.ids[s], reference.w[s].to_bits()), (0, 0));
                    assert!(poison.eo[s * HIDDEN].is_nan() || poison.eo[s * HIDDEN].is_infinite());
                    assert!(reference.eo[s * HIDDEN..(s + 1) * HIDDEN]
                        .iter()
                        .all(|&v| v == 0.0));
                }
            }
        }
    }

    #[test]
    fn old_symbol_prelude_renames_every_symbol() {
        let p = old_prelude(OLD_STABLE.syms);
        for s in OLD_STABLE.syms {
            assert!(p.contains(&format!("#define {s} {s}{OLD_SUFFIX}\n")), "{s}");
        }
    }

    #[test]
    fn old_table_names_exist_in_the_tree() {
        for old in [
            &OLD_ROUTER,
            &OLD_ROUTER_FAST,
            &OLD_STABLE,
            &OLD_FUSED,
            &OLD_UNSCATTER,
            &OLD_COMBINE,
            &OLD_COMBINE_GROUPED,
            &OLD_Q8_DOWN,
            &OLD_Q8_GROUPED,
            &OLD_MQ4_DOWN,
            &OLD_MQ4_MULTIROW,
            &OLD_MQ4_O8R16,
            &OLD_SLAB_PLAN,
            &OLD_RENORM,
        ] {
            let text = src(old.file);
            for s in old.syms {
                assert!(text.contains(s), "{} does not define {s}", old.file);
            }
        }
    }

    /// Every guarded consumer carries its bound in the working tree and (when the old ref is
    /// reachable) the old ref does not: the harness compares against code that really lacks it.
    #[test]
    fn guard_markers_present_new_absent_old() {
        let table: &[(&str, &[&str])] = &[
            (
                "moe_router_softmax_top10_f32.hip",
                &["router_fail_row", "isfinite(first_logit)"],
            ),
            (
                "moe_router_softmax_top10_f32_fast.hip",
                &["router_fail_row"],
            ),
            ("qwen4_moe_scatter_stable_top10.hip", &["raw < E"]),
            ("qwen4_moe_scatter_fused_k8.hip", &["inverse_perm[i] = -1"]),
            ("moe_down_combine_top10_batched.hip", &["n_experts"]),
            ("moe_down_combine_grouped_top10.hip", &["n_experts"]),
            ("gemm_hfq4g256_moe_grouped_wmma.gfx12.hip", &["n_experts"]),
        ];
        for (file, markers) in table {
            let new = src(file);
            for m in *markers {
                assert!(new.contains(m), "{file}: guard marker {m:?} missing");
            }
            if let Some(old) = old_src(file) {
                for m in *markers {
                    assert!(
                        !old.contains(m),
                        "{file}: marker {m:?} already present at the old ref, no regression delta"
                    );
                }
            }
        }
    }

    /// Family-wide: every grouped/indexed MoE kernel that indexes an expert table takes an expert bound.
    #[test]
    fn every_expert_indexing_moe_kernel_takes_a_bound() {
        let dir = repo().join("kernels/src");
        let mut missing = Vec::new();
        let mut checked = 0usize;
        for ent in std::fs::read_dir(&dir).expect("kernels/src") {
            let path = ent.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if !name.ends_with(".hip") || !name.contains("moe") || name.contains("scatter") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let indexes = text.contains("expert_tile_ids")
                || text.contains("expert_weight_ptrs[")
                || text.contains("expert_ptrs[");
            if !indexes {
                continue;
            }
            checked += 1;
            let lower = text.to_ascii_lowercase();
            if !(lower.contains("n_exp")
                || lower.contains("num_exp")
                || lower.contains("n_experts"))
            {
                missing.push(name);
            }
        }
        assert!(
            checked > 40,
            "scan found only {checked} expert-indexing kernels"
        );
        assert!(
            missing.is_empty(),
            "expert-indexing kernels without an expert bound: {missing:?}"
        );
    }
}
