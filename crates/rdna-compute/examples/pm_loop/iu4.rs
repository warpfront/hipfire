// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `iu4_gemm` workload: the builder-emitted MQ4V2 x `block_i4_128` GEMM for
//! gfx1201 (`peacemaker-isa emit --kernel iu4_gemm`) against the committed
//! production bundles and a CPU reference.
//!
//! # Shape grammar
//! `M=<rows>,N=<tokens>,K=<k>`: all three required, any order, comma
//! separated (e.g. `M=4096,N=512,K=5120`). `K % 256 == 0`, `M % 4 == 0`,
//! `N >= 1`; partial M/N tiles are handled by the kernel. For `silu` /
//! `silu-bf16` `M` is the number of intermediate rows (the gate and the up
//! matrix each have `M` rows, the kernel runs `2M` virtual rows and writes
//! `H[N x M]`), and `M % 128 == 0` as the production launcher requires
//! (`gemm_gate_up_silu_mq4g256v2_iu4_gfx12`). `qkvzagdn` is rejected: its
//! 120-byte fused ABI (conv state, Q/K/V rows, ...) is not reproduced here.
//!
//! # Knobs (candidate; names as `peacemaker-isa emit --kernel iu4_gemm`)
//! `fold` (k128|k256s|k256shared|k256p|k256pow2), `tile`
//! (128x128x8|256x128x16), `cacc` (1|2), `epi` (set|add|silu|silu-bf16),
//! `alayout` (token|slab). Baseline overrides use the `base-` prefix on
//! `fold|tile|cacc|alayout`; with none the baseline is the committed
//! production bundle for the candidate's activation layout, otherwise it is
//! emitted from the knobs with unspecified ones inherited.
//!
//! # `--mma dense|s24x64|s24x32` (2:4 sparse weights)
//! `mma` selects the K-loop MMA of the candidate (`base-mma` of an emitted
//! baseline, which otherwise inherits the candidate's). A sparse kind is
//! `v_swmmac_i32_16x16x64_iu4` (`s24x64`) or `..x32_iu4` (`s24x32`) over
//! 2:4-pruned weights: with any sparse arm the generated weights are pruned
//! deterministically (`sparse-seed`): within every sparse group of 4 K
//! (`sparse24::group_of`, 32 groups per K128 row block) two pseudo-randomly
//! chosen K are kept and the pruned K get code 8 (signed zero). The candidate
//! reads those weights packed by `sparse24::pack_block` (same 136-byte MQ4V2
//! container, same buffer size); the CPU reference and the shipped dense
//! baseline read the PRUNED DENSE weights, so `identical_to_baseline` is
//! expected to be true. The record gains `sparse: {mma, base_mma,
//! groups_per_block, kept_per_group, seed}`.
//!
//! # `--inputs capture:<act_capture.json>[@site]`
//! The activation records (`N` columns of `K/128` blocks) are quantized from the F32 rows of one site of a pinned
//! `act_capture.v1` instead of the synthetic generator: column `c` is capture row `c % rows`. The bin's sha256 is
//! verified, `K` must equal the capture's `k` (else an error), and the record gets
//! `inputs: {kind:"capture", path, id, site}`. Weights stay synthetic.
//!
//! # Generator knobs (`iu4_gemm::Tune`)
//! `ring` (`2`|`3`), `wait` (`after`|`mid`|`before`: where a K-loop block
//! publishes its next slab relative to the slab's WMMAs, see
//! `iu4_gemm::WaitPlace`) and `fold-op` (`fma`|`muladd`, `iu4_gemm::FoldOp`),
//! as `--ring 2 --wait mid --fold-op fma`. `ring` and `wait` are schedule
//! knobs: the candidate must stay `identical_to_baseline`. `ring 3` publishes two
//! slabs ahead (a third A/W LDS slab, a second prefetch set; dense K128 variants,
//! not the fused GDN projection: `Spec::validate` refuses the rest with the
//! numbers); its launch passes the shipped dynamic LDS size
//! (`Spec::lds_launch`) and the object declares the third slab as static
//! LDS (`group_segment_fixed_size`), so the product launcher's constant is
//! right for it too. `fold-op muladd` is a deliberately
//! NUMERIC candidate (non-fused `v_mul_f32` + `v_add_f32` stage-2 accumulate,
//! the hipcc DAG): the record declares `numeric_contract.relation = "numeric"`,
//! reports `identical_to_baseline` honestly and, for set/add, checks the
//! candidate against the `muladd` CPU reference (the baseline against `fma`);
//! for silu there is no CPU reference and the candidate passes when it is
//! finite and NaN-pattern-equal to the baseline (the KLD/state gate decides).
//! The knobs reach the record's `candidate.knobs` only when non-default.
//! Baseline overrides: `base-ring|base-wait|base-fold-op` (the baseline is
//! otherwise the default tune). Not combinable with `--rip`: the script
//! chooses its own tune (its `const RING`, read from the source, sets the
//! candidate's ring depth, i.e. the generator the script is compared with).
//!
//! `idle` (`off`|`skip`|`skiphi`, `iu4_gemm::IdleHalf`), as `--idle skip` (baseline override
//! `--base-idle`): at `N - bs <= 64` the 128x128x8 tile's token-half-1 waves compute zeros whose
//! stores are masked; `skip` branches those waves over their barrier-free compute (fragment loads,
//! WMMA groups, scale loads, fold) and their epilogue, `skiphi` also remaps the waves so the dead
//! ones are 4..7 (the token half is `wave >> 2`). Scheduling only: the candidate must stay
//! `identical_to_baseline`. Dense 128x128x8 ring-2 `wait after|window` only (`Spec::validate`
//! refuses the rest with the reason).
//!
//! # `--base-shipped b1s|v64`
//! Which SHIPPED export is the baseline (default `b1s`, the prefill export). `v64` selects the small-N export the
//! DFlash verify launches for `n <= 128` rows (`iu4_gemm::verify64_tune`: window schedule, PairW fetches, dead-half
//! skip; byte-identical outputs): dense slab 128x128x8 `set|add|silu` only (`--alayout slab`), and it excludes the
//! `--base-<knob>` emitted baselines. The baseline's recorded knobs then name its real schedule. The emitted twin of
//! that export is `--wait window --fetch pairw --idle skiphi` (code sha equal to the shipped export's).
//!
//! # `--pool` / `--flush-l2` / `--flush` (cold timing)
//! Cold-cache timing with a rotating weight pool, see the doc comment of `pm_loop.rs`. The pool holds distinct seeded
//! weight sets (`pool_seed`; gate and up for silu), set 0 being the correctness weights; sparse arms are refused.
//!
//! # Extra knobs
//! `ref-rows`: `full` (default while `M*N*K <= 2e9` MACs) or `sample:<n>`
//! (default `sample:192` beyond that): rows `0..64`, the last 64, and `n`
//! evenly spread rows, every token column. `max_abs_diff` always covers the
//! whole output. `ref-fold`: `fma` or `muladd` (default: each arm's own
//! `fold-op`), see below.
//!
//! # Reference numerics
//! Symmetric fold DAG (production `mq4v2_symmetric`): per K128 block
//! `kb = 2g + h` in ascending order, `dot = sum_t (nib ^ 8 as signed) * q`
//! (exact i32), `t1 = RN(sc_h * d)`, then the accumulate. The builder's fold
//! emitters (`iu4_k1.rs::emit_fold_scale`) issue `v_fmac_f32 acc += t1 * c`
//! with `c = float(C) - magic = float(dot)`: a FUSED `fma(t1, dot, acc)` from
//! `acc = +0.0`. The hipcc `_v3` DAG (`iu4_symmetric_fold_rn_pair`,
//! gemm_mq4g256v2_residual_mmq_iu4_v3.gfx12.hip:~280) is `RN(acc + RN(t1*dot))`.
//! The default follows the builder code (`fma`); `ref-fold=muladd` selects the
//! hipcc DAG. Output `Y[col * M + row]`, f32; ADD is `RN(Y0 + sum)`.

use crate::common::{contains, dev_err, put_i32, put_ptr, sha256_hex};
use crate::workload::{Arm, Knobs, Launch, Object, RefReport, Result, Workload};
use peacemaker_isa::kernels::iu4_gemm::{
    self, sparse24, verify64_tune, ALayout, Epi, FoldOp, Mma, RingDepth, Spec, Tile, Tune,
};
use peacemaker_isa::Arch;
use rayon::prelude::*;
use rdna_compute::Gpu;
use serde_json::{json, Value};
use std::cell::OnceCell;
use std::path::Path;

const GROUP_BYTES: usize = 136;
const REC: usize = 72;
const FULL_REF_MACS: u128 = 2_000_000_000;
/// 2:4 sparse groups per K128 row block, and K kept per group.
const SPARSE_GROUPS: usize = 32;
const SPARSE_KEPT: usize = 2;
const DEFAULT_SPARSE_SEED: u64 = 0x5eed_2404_a5a5_c3c3;
/// Seeds of the gate and up weights of the correctness set (`Host::synth`); pool set `s >= 1`
/// derives its seeds with [`pool_seed`].
const GATE_SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const UP_SEED: u64 = 0xd1b5_4a32_d192_ed03;

/// Seed of weight set `set` of the `--flush` timing pool (set 0 is the correctness set itself).
pub fn pool_seed(base: u64, set: usize) -> u64 {
    base ^ (set as u64).wrapping_mul(0xa076_1d64_78bd_642f)
}

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 23) as f32) - 1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefFold {
    Fma,
    MulAdd,
}

/// Host-side synthetic inputs: everything the reference needs.
pub struct Host {
    /// MQ4G256V2 weights `[M x K/256 x 136]` (gate for silu).
    pub w: Vec<u8>,
    /// Up weights (silu only, empty otherwise).
    pub w_up: Vec<u8>,
    /// `block_i4_128` records, token layout: record `kb * N + col`.
    pub recs: Vec<u8>,
    /// Initial `Y` for ADD `[N x M]`.
    pub y0: Vec<f32>,
}

fn f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

fn f16_val(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

pub fn synth_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let mut rng = Rng::new(seed);
    let mut w = vec![0u8; m * gpr * GROUP_BYTES];
    for blk in w.chunks_exact_mut(GROUP_BYTES) {
        let sc0 = 0.02 + (rng.next() % 16) as f32 * 0.001;
        let sc1 = 0.03 + (rng.next() % 16) as f32 * 0.001;
        for (i, sc) in [sc0, sc1].into_iter().enumerate() {
            let bits = f16_bits(sc);
            blk[4 * i..4 * i + 2].copy_from_slice(&bits.to_le_bytes());
            // zp is ignored by the symmetric fold; keep the asymmetric value.
            let zp = f16_bits(-8.0 * f16_val(bits));
            blk[4 * i + 2..4 * i + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for q in blk[8..].chunks_exact_mut(8) {
            q.copy_from_slice(&rng.next().to_le_bytes());
        }
    }
    w
}

/// One `block_i4_128`: per-block symmetric `d = max|x| / 7`, `q = clamp(round(x/d), -8, 7)`.
fn quantize_block(x: &[f32; 128]) -> [u8; REC] {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let d = amax / 7.0;
    let mut q = [0i32; 128];
    if d > 0.0 {
        for (qi, v) in q.iter_mut().zip(x) {
            *qi = ((v / d).round() as i32).clamp(-8, 7);
        }
    }
    let mut rec = [0u8; REC];
    rec[0..4].copy_from_slice(&d.to_le_bytes());
    rec[4..8].copy_from_slice(&q.iter().sum::<i32>().to_le_bytes());
    for j in 0..64 {
        rec[8 + j] = (q[2 * j] as u8 & 0xf) | ((q[2 * j + 1] as u8 & 0xf) << 4);
    }
    rec
}

impl Host {
    pub fn synth(m: usize, n: usize, k: usize, silu: bool) -> Self {
        let w = synth_weights(m, k, GATE_SEED);
        let w_up = if silu {
            synth_weights(m, k, UP_SEED)
        } else {
            Vec::new()
        };
        let mut rng = Rng::new(0x1234_5678_9abc_def1);
        let mut recs = vec![0u8; k / 128 * n * REC];
        for kb in 0..k / 128 {
            for col in 0..n {
                let amp = 1.0 + (col % 4) as f32;
                let mut x = [0f32; 128];
                for v in &mut x {
                    *v = rng.unit() * amp;
                }
                let o = (kb * n + col) * REC;
                recs[o..o + REC].copy_from_slice(&quantize_block(&x));
            }
        }
        let y0 = (0..n * m).map(|_| 0.25 + rng.unit()).collect();
        Self { w, w_up, recs, y0 }
    }
}

/// The opaque `id` of a pinned `route_hist.v1` / `act_capture.v1` document (its producer's
/// sha256 of the canonical JSON; never recomputed here). Integrity is checked against the bin.
pub fn artefact_id(doc: &Value) -> Result<String> {
    doc["id"]
        .as_str()
        .filter(|s| s.len() == 64)
        .map(str::to_owned)
        .ok_or_else(|| "missing 64-hex top-level `id`".to_owned())
}

/// Resolve `p` (a path stored inside an artefact) against the artefact's directory.
pub fn sibling_path(json_path: &Path, p: &str) -> std::path::PathBuf {
    let p = Path::new(p);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        json_path.parent().unwrap_or(Path::new(".")).join(p)
    }
}

/// `--inputs capture:<act_capture.json>[@site]`: F32 activation rows of one site of a pinned
/// `act_capture.v1` (rows-only bin grouped per site at `offset_bytes`), sha-verified. Row `i` of a workload is
/// capture row `i % rows` (cycled).
#[derive(Clone, Debug, PartialEq)]
pub struct Capture {
    pub path: String,
    pub id: String,
    pub site: String,
    pub k: usize,
    pub nrows: usize,
    rows: Vec<f32>,
}

impl Capture {
    pub fn parse(arg: &str) -> Result<Capture> {
        let spec = arg
            .strip_prefix("capture:")
            .ok_or_else(|| format!("--inputs {arg}: expected capture:<act_capture.json>[@site]"))?;
        let (path, site) = match spec.rsplit_once('@') {
            Some((p, s)) if !s.is_empty() => (p, Some(s)),
            _ => (spec, None),
        };
        Self::load(Path::new(path), site)
    }

    pub fn load(path: &Path, site: Option<&str>) -> Result<Capture> {
        let err = |m: String| format!("--inputs capture:{}: {m}", path.display());
        let text = std::fs::read(path).map_err(|e| err(e.to_string()))?;
        let doc: Value = serde_json::from_slice(&text).map_err(|e| err(e.to_string()))?;
        if doc["schema"] != "act_capture.v1" {
            return Err(err(format!(
                "schema {}: expected act_capture.v1",
                doc["schema"]
            )));
        }
        let id = artefact_id(&doc).map_err(err)?;
        let bin_rel = doc["bin"]["path"]
            .as_str()
            .ok_or_else(|| err("missing bin.path".into()))?;
        let bin_path = sibling_path(path, bin_rel);
        let bin =
            std::fs::read(&bin_path).map_err(|e| err(format!("{}: {e}", bin_path.display())))?;
        let want = doc["bin"]["sha256"].as_str().unwrap_or("");
        let got = sha256_hex(&bin);
        if want != got {
            return Err(err(format!(
                "bin sha256 mismatch ({}): manifest {want}, file {got}",
                bin_path.display()
            )));
        }
        let sites: Vec<&str> = doc["sites"]
            .as_array()
            .map(|a| a.iter().filter_map(|s| s["site"].as_str()).collect())
            .unwrap_or_default();
        let site = match (site, sites.as_slice()) {
            (Some(s), _) if sites.contains(&s) => s,
            (Some(s), _) => return Err(err(format!("site `{s}` not in {sites:?}"))),
            (None, [only]) => only,
            (None, _) => return Err(err(format!("several sites {sites:?}: pick one with @site"))),
        };
        let ent = doc["sites"]
            .as_array()
            .and_then(|a| a.iter().find(|s| s["site"] == site))
            .ok_or_else(|| err(format!("site `{site}` missing")))?;
        let field = |n: &str| {
            ent[n]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| err(format!("site `{site}`: missing {n}")))
        };
        let (k, nrows, off) = (field("k")?, field("rows_total")?, field("offset_bytes")?);
        let end = nrows
            .checked_mul(k)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(off))
            .filter(|&e| e <= bin.len())
            .ok_or_else(|| {
                err(format!(
                    "site `{site}`: rows exceed the bin ({} bytes)",
                    bin.len()
                ))
            })?;
        if nrows == 0 || k == 0 {
            return Err(err(format!("site `{site}`: no rows")));
        }
        let rows = bin[off..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Capture {
            path: path.display().to_string(),
            id,
            site: site.to_owned(),
            k,
            nrows,
            rows,
        })
    }

    /// Capture row `i` (cycled), `k` f32.
    pub fn row(&self, i: usize) -> &[f32] {
        let r = i % self.nrows;
        &self.rows[r * self.k..(r + 1) * self.k]
    }

    /// A workload whose activation width is `k` takes the capture only when the widths agree.
    pub fn require_k(&self, k: usize) -> Result<()> {
        if self.k != k {
            return Err(format!(
                "--inputs capture:{}@{}: capture rows have K={}, the workload needs K={k}",
                self.path, self.site, self.k
            ));
        }
        Ok(())
    }

    pub fn json(&self) -> Value {
        json!({"kind": "capture", "path": self.path, "id": self.id, "site": self.site})
    }
}

/// Token-layout MQ4 activation records (`Host::recs`) from capture rows: column `col` is row
/// `col` of the capture (cycled), K128 block `kb` its elements `kb*128..`.
fn recs_from_capture(c: &Capture, n: usize, k: usize) -> Vec<u8> {
    let mut recs = vec![0u8; k / 128 * n * REC];
    for kb in 0..k / 128 {
        for col in 0..n {
            let mut x = [0f32; 128];
            x.copy_from_slice(&c.row(col)[kb * 128..(kb + 1) * 128]);
            let o = (kb * n + col) * REC;
            recs[o..o + REC].copy_from_slice(&quantize_block(&x));
        }
    }
    recs
}

/// The kept pair (indices into a group's 4 dense K, ascending) of `pick` in `0..6`.
fn kept_pair(pick: u64) -> (usize, usize) {
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)][pick as usize % 6]
}

/// Prune one row's K128 half-block in place: per sparse group keep a
/// pseudo-random pair of its 4 K (stream seeded from `seed`, row, block), set
/// the other codes to 8, and return the kept mask. `codes[k]` natural dense K
/// order; the group of K comes from `sparse24::group_of(mma, k)`.
fn prune_block(codes: &mut [u8; 128], mma: Mma, seed: u64) -> [bool; 128] {
    let mut rng = Rng::new(seed);
    let mut groups = vec![Vec::with_capacity(4); SPARSE_GROUPS];
    for k in 0..128 {
        groups[sparse24::group_of(mma, k)].push(k);
    }
    let mut kept = [false; 128];
    for members in &groups {
        debug_assert_eq!(members.len(), 4);
        let (a, b) = kept_pair(rng.next() >> 33);
        kept[members[a]] = true;
        kept[members[b]] = true;
    }
    for k in 0..128 {
        if !kept[k] {
            codes[k] = 8;
        }
    }
    kept
}

/// Prune the MQ4G256V2 weights `w` (`[m x k/256 x 136]`) to 2:4 in place and
/// return, per requested `pack_for` kind, a copy with every K128 block's 64
/// code bytes replaced by `sparse24::pack_block` (headers unchanged). The
/// pruning numbering uses `prune_mma`; every packer re-validates 2-of-4 under
/// its own numbering.
fn prune_and_pack(
    w: &mut [u8],
    k: usize,
    prune_mma: Mma,
    seed: u64,
    pack_for: &[Mma],
) -> Result<Vec<Vec<u8>>> {
    let row_bytes = k / 256 * GROUP_BYTES;
    let mut packed: Vec<Vec<u8>> = pack_for.iter().map(|_| w.to_vec()).collect();
    let mut iters: Vec<_> = packed.iter_mut().map(|p| p.chunks_mut(row_bytes)).collect();
    let mut rows: Vec<(&mut [u8], Vec<&mut [u8]>)> = Vec::new();
    for dense in w.chunks_mut(row_bytes) {
        let outs = iters
            .iter_mut()
            .map(|it| it.next().expect("packed image has the dense row count"))
            .collect();
        rows.push((dense, outs));
    }
    rows.par_iter_mut()
        .enumerate()
        .try_for_each(|(r, (dense, outs))| -> Result<()> {
            for kb in 0..k / 128 {
                let (g, h) = (kb / 2, kb % 2);
                let at = g * GROUP_BYTES + 8 + 64 * h;
                let mut codes = [0u8; 128];
                for (j, &byte) in dense[at..at + 64].iter().enumerate() {
                    codes[2 * j] = byte & 0xf;
                    codes[2 * j + 1] = byte >> 4;
                }
                let mix = ((r * (k / 128) + kb) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                let kept = prune_block(&mut codes, prune_mma, seed ^ mix);
                for j in 0..64 {
                    dense[at + j] = codes[2 * j] | (codes[2 * j + 1] << 4);
                }
                for (mma, out) in pack_for.iter().zip(outs.iter_mut()) {
                    let bytes = sparse24::pack_block(*mma, &codes, &kept).map_err(|e| {
                        format!("pack_block({}) row {r} block {kb}: {e}", mma.name())
                    })?;
                    out[at..at + 64].copy_from_slice(&bytes);
                }
            }
            Ok(())
        })?;
    Ok(packed)
}

/// Token-layout records to the slab planes `[d:4N][s:4N][qs0..32:32N][qs32..64:32N]` per K128 block.
pub fn to_slab(recs: &[u8], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; recs.len()];
    for (kb, blk) in out.chunks_exact_mut(REC * n).enumerate() {
        for col in 0..n {
            let r = &recs[(kb * n + col) * REC..][..REC];
            blk[4 * col..4 * col + 4].copy_from_slice(&r[0..4]);
            blk[4 * n + 4 * col..4 * n + 4 * col + 4].copy_from_slice(&r[4..8]);
            blk[8 * n + 32 * col..8 * n + 32 * col + 32].copy_from_slice(&r[8..40]);
            blk[40 * n + 32 * col..40 * n + 32 * col + 32].copy_from_slice(&r[40..72]);
        }
    }
    out
}

fn unpack_x(recs: &[u8]) -> (Vec<i8>, Vec<f32>) {
    let mut q = Vec::with_capacity(recs.len() / REC * 128);
    let mut d = Vec::with_capacity(recs.len() / REC);
    for r in recs.chunks_exact(REC) {
        d.push(f32::from_le_bytes(r[0..4].try_into().unwrap()));
        for &b in &r[8..] {
            q.push(((b << 4) as i8) >> 4);
            q.push((b as i8) >> 4);
        }
    }
    (q, d)
}

/// Reference `Y` rows (`rows` x N, row-major in `rows` order) for SET/ADD.
pub fn reference(
    host: &Host,
    m: usize,
    n: usize,
    k: usize,
    add: bool,
    rows: &[usize],
    fold: RefFold,
) -> Vec<f32> {
    let (xq, xd) = unpack_x(&host.recs);
    let (gpr, nkb) = (k / 256, k / 128);
    let mut out = vec![0f32; rows.len() * n];
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let per = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (rs, os) in rows.chunks(per).zip(out.chunks_mut(per * n)) {
            let (xq, xd) = (&xq, &xd);
            s.spawn(move || {
                let mut wq = vec![0i8; k];
                let mut sc = vec![0f32; nkb];
                for (&r, orow) in rs.iter().zip(os.chunks_exact_mut(n)) {
                    for g in 0..gpr {
                        let b = &host.w[(r * gpr + g) * GROUP_BYTES..][..GROUP_BYTES];
                        for h in 0..2 {
                            sc[2 * g + h] = f16_val(u16::from_le_bytes([b[4 * h], b[4 * h + 1]]));
                        }
                        for (j, &byte) in b[8..].iter().enumerate() {
                            // nibble ^ 8 as signed 4-bit (low nibble = even K).
                            wq[256 * g + 2 * j] = ((((byte & 0xf) ^ 8) << 4) as i8) >> 4;
                            wq[256 * g + 2 * j + 1] = (((byte >> 4) ^ 8) << 4) as i8 >> 4;
                        }
                    }
                    for (col, o) in orow.iter_mut().enumerate() {
                        let mut sum = 0f32;
                        for kb in 0..nkb {
                            let x = &xq[(kb * n + col) * 128..][..128];
                            let w = &wq[kb * 128..][..128];
                            let dot: i32 = w
                                .iter()
                                .zip(x)
                                .map(|(&a, &b)| i32::from(a) * i32::from(b))
                                .sum();
                            let t1 = sc[kb] * xd[kb * n + col];
                            sum = match fold {
                                RefFold::Fma => t1.mul_add(dot as f32, sum),
                                RefFold::MulAdd => sum + t1 * dot as f32,
                            };
                        }
                        *o = if add { host.y0[col * m + r] + sum } else { sum };
                    }
                }
            });
        }
    });
    out
}

/// Deterministic sampled rows: first 64, last 64, and `n` evenly spread.
pub fn sample_rows(m: usize, n: usize) -> Vec<usize> {
    let mut s = std::collections::BTreeSet::new();
    s.extend(0..m.min(64));
    s.extend(m.saturating_sub(64)..m);
    for i in 0..n {
        s.insert(i * (m - 1) / n.saturating_sub(1).max(1));
    }
    s.into_iter().collect()
}

fn parse_shape(s: &str) -> Result<(usize, usize, usize)> {
    let (mut m, mut n, mut k) = (None, None, None);
    for part in s.split(',') {
        let (key, v) = part
            .split_once('=')
            .ok_or_else(|| format!("shape `{s}`: expected M=..,N=..,K=.."))?;
        let v: usize = v
            .trim()
            .parse()
            .map_err(|e| format!("shape `{s}`: {key}: {e}"))?;
        let slot = match key.trim() {
            "M" => &mut m,
            "N" => &mut n,
            "K" => &mut k,
            other => return Err(format!("shape `{s}`: unknown dimension `{other}`")),
        };
        if slot.replace(v).is_some() {
            return Err(format!("shape `{s}`: duplicate {key}"));
        }
    }
    match (m, n, k) {
        (Some(m), Some(n), Some(k)) => Ok((m, n, k)),
        _ => Err(format!("shape `{s}`: M, N and K are all required")),
    }
}

/// SET/ADD explicit kernarg: `A Xq Y` at 0/8/16, `M K N add` at 24/28/32/36.
fn pack_set_add(a: u64, xq: u64, y: u64, m: usize, k: usize, n: usize, add: bool) -> Vec<u8> {
    let mut b = Vec::with_capacity(40);
    for p in [a, xq, y] {
        put_ptr(&mut b, p);
    }
    for v in [m, k, n, usize::from(add)] {
        put_i32(&mut b, v as i32);
    }
    b
}

/// Gate/up explicit kernarg: `G U Xq H` at 0/8/16/24, `M K N` at 32/36/40.
fn pack_silu(g: u64, u: u64, xq: u64, h: u64, m: usize, k: usize, n: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(44);
    for p in [g, u, xq, h] {
        put_ptr(&mut b, p);
    }
    for v in [m, k, n] {
        put_i32(&mut b, v as i32);
    }
    b
}

#[derive(Clone)]
struct Strs {
    fold: String,
    tile: String,
    cacc: String,
    epi: String,
    alayout: String,
    mma: String,
    /// Generator knobs (`iu4_gemm::Tune`), the spellings `RingDepth`/`WaitPlace`/`FoldOp` parse.
    ring: String,
    wait: String,
    fold_op: String,
    fetch: String,
    idle: String,
}

impl Strs {
    fn json(&self) -> Value {
        let mut v = json!({"fold": self.fold, "tile": self.tile, "cacc": self.cacc,
               "epi": self.epi, "alayout": self.alayout, "mma": self.mma});
        // Only non-default generator knobs are recorded: every existing record keeps its bytes.
        for (key, val, default) in [
            ("ring", &self.ring, "2"),
            ("wait", &self.wait, "after"),
            ("fold-op", &self.fold_op, "fma"),
            ("fetch", &self.fetch, "split"),
            ("idle", &self.idle, "off"),
        ] {
            if val != default {
                v[key] = json!(val);
            }
        }
        v
    }
    fn tune(&self) -> Result<Tune> {
        Ok(Tune {
            ring: self.ring.parse()?,
            wait: self.wait.parse()?,
            fold_op: self.fold_op.parse()?,
            fetch: self.fetch.parse()?,
            idle: self.idle.parse()?,
            ..Tune::default()
        })
    }
    fn spec(&self) -> Result<Spec> {
        let mut spec = Spec::from_knobs(
            &self.fold,
            &self.tile,
            &self.cacc,
            &self.epi,
            &self.alayout,
            Arch::Gfx1201,
        )?;
        spec.mma = self.mma.parse()?;
        spec.tune = self.tune()?;
        Ok(spec)
    }
}

struct Dev {
    /// Dense weights (pruned when any arm is sparse): the baseline and the reference read these.
    w: hip_bridge::DeviceBuffer,
    /// Per arm ([`arm_index`]): the `pack_block` image a sparse arm reads instead of `w`.
    w_packed: [Option<hip_bridge::DeviceBuffer>; 2],
    w_up: Option<hip_bridge::DeviceBuffer>,
    xq_token: hip_bridge::DeviceBuffer,
    xq_slab: hip_bridge::DeviceBuffer,
    /// Indexed by [`arm_index`].
    y: [hip_bridge::DeviceBuffer; 2],
    /// `--flush` pool: weight sets `1..pool_sets()` (gate, up) next to the correctness set 0.
    pool: Vec<(hip_bridge::DeviceBuffer, Option<hip_bridge::DeviceBuffer>)>,
}

fn arm_index(arm: Arm) -> usize {
    match arm {
        Arm::Baseline => 0,
        Arm::Candidate => 1,
    }
}

pub struct Iu4 {
    m: usize,
    n: usize,
    k: usize,
    cand: Spec,
    cand_strs: Strs,
    /// Spec the baseline arm runs (shipped: tile 128x128x8 with the candidate's epi/act).
    base: Spec,
    base_strs: Strs,
    /// `false`: the shipped bundle; `true`: emitted from `base`.
    base_emitted: bool,
    rows: Vec<usize>,
    rows_desc: String,
    /// `--ref-fold`: overrides every arm's reference fold (default: the arm's own `fold-op`).
    ref_fold: Option<RefFold>,
    host: Option<Host>,
    dev: Option<Dev>,
    /// CPU reference rows per [`RefFold`] (`fma`, `muladd`), computed on first use.
    reference: [OnceCell<Vec<f32>>; 2],
    /// Seed of the 2:4 pruning (`sparse-seed`).
    sparse_seed: u64,
    /// `--rip`: the candidate arm is this source, not `iu4_gemm::emit`.
    rip: Option<RipSource>,
    /// `--pool` / `--flush-l2` / `--flush`: cold-cache timing with the rotating weight pool (`Workload::set_cold`).
    cold: crate::cold::ColdMode,
    /// `--inputs capture:<file>[@site]`: the activation rows come from a pinned capture.
    capture: Option<Capture>,
}

struct RipSource {
    path: String,
    text: String,
}

/// ABI of a kernel in an assembled object: everything the launcher and the
/// kernarg packer depend on (symbol, kernarg layout, workgroup size, LDS,
/// wave mode). Code, VGPR/SGPR counts and the like may differ.
pub fn abi_signature(elf: &[u8], symbol: &str) -> Result<String> {
    use peacemaker_ir::inst::Abi;
    let program = peacemaker_lift::lift_object(
        elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .map_err(dev_err)?
    .program;
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| format!("object has no kernel {symbol}"))?;
    let Abi::Hsa {
        descriptor,
        metadata,
    } = &kernel.abi
    else {
        return Err(format!("{symbol}: not an HSA kernel"));
    };
    let m = &metadata.parsed;
    Ok(format!(
        "symbol={} args={:?} kernarg_bytes={} kernarg_align={} lds_bytes={} max_flat_workgroup_size={} wavefront_size={} workgroup_processor_mode={}",
        m.symbol,
        m.args,
        descriptor.kernarg_size,
        m.kernarg_segment_align,
        descriptor.group_segment_fixed_size,
        m.max_flat_workgroup_size,
        m.wavefront_size,
        m.workgroup_processor_mode,
    ))
}

/// The payload ring depth a `.rip` script builds: its `const RING: usize = 2|3;` (absent: the
/// shipped depth 2).
fn rip_ring(source: &str) -> Result<RingDepth> {
    for line in source.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("const RING: usize =") {
            let v = rest.trim().trim_end_matches(';').trim();
            return v.parse().map_err(|e| format!("`{l}`: {e}"));
        }
    }
    Ok(RingDepth::Two)
}

/// Compile `.rip` source for `spec`, refuse unless its ABI equals that of
/// `iu4_gemm::emit(spec)`, and return the linked ELF, the symbol and the
/// `candidate.rip` record fields (`variant`, `sha256`, `identical_to_generator`).
fn compile_rip(source: &str, spec: Spec) -> Result<(Vec<u8>, String, Value)> {
    let variant = spec.variant_name();
    // The script names the base variant; its own `const RING` picks the depth.
    let base = Spec {
        tune: Tune::default(),
        ..spec
    };
    let compiled = peacemaker_rip::compile(source, spec.arch, &base.variant_name())?;
    let generator = iu4_gemm::emit(spec)?;
    let symbol = spec.symbol();
    let elf = peacemaker_isa::native::assemble(&compiled.s_text, spec.arch)?;
    let gen_elf = peacemaker_isa::native::assemble(&generator.s_text, spec.arch)?;
    let got = abi_signature(&elf, &symbol)
        .map_err(|e| format!("compiled kernel does not expose {symbol}: {e}"))?;
    let want = abi_signature(&gen_elf, &symbol)?;
    if got != want {
        return Err(format!(
            "ABI mismatch with iu4_gemm::emit({variant}):\n  .rip:      {got}\n  generator: {want}"
        ));
    }
    let info = json!({
        "sha256": sha256_hex(source.as_bytes()),
        "variant": variant,
        "identical_to_generator": compiled.s_text == generator.s_text,
    });
    Ok((elf, symbol, info))
}

#[cfg(test)]
pub fn new(shape: &str, knobs: &mut Knobs) -> Result<Box<dyn Workload>> {
    new_with_rip(shape, knobs, None)
}

/// [`new`] with the candidate compiled from a `.rip` file (`--rip`).
pub fn new_with_rip(
    shape: &str,
    knobs: &mut Knobs,
    rip: Option<&Path>,
) -> Result<Box<dyn Workload>> {
    let rip = match rip {
        Some(p) => Some(RipSource {
            path: p.display().to_string(),
            text: std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        }),
        None => None,
    };
    let (m, n, k) = parse_shape(shape)?;
    if m == 0 || n == 0 || k == 0 || k % 256 != 0 || m % 4 != 0 {
        return Err(format!(
            "shape M={m},N={n},K={k}: need M%4==0, K%256==0, N>=1"
        ));
    }
    let cand_strs = Strs {
        fold: knobs.take_or("fold", "k128"),
        tile: knobs.take_or("tile", "128x128x8"),
        cacc: knobs.take_or("cacc", "1"),
        epi: knobs.take_or("epi", "set"),
        alayout: knobs.take_or("alayout", "token"),
        mma: knobs.take_or("mma", "dense"),
        ring: knobs.take_or("ring", "2"),
        wait: knobs.take_or("wait", "after"),
        fold_op: knobs.take_or("fold-op", "fma"),
        fetch: knobs.take_or("fetch", "split"),
        idle: knobs.take_or("idle", "off"),
    };
    let mut cand = cand_strs.spec()?;
    let mut cand_strs = cand_strs;
    if rip.is_some() && !cand.tune.is_default() {
        return Err(
            "--ring/--wait/--fold-op cannot be combined with --rip (the script chooses its tune: `const RING`, `const WAIT`)"
                .into(),
        );
    }
    if let Some(r) = &rip {
        // The script's `const RING` sets the candidate's depth: the generator the
        // script is compared with (`identical_to_generator`).
        cand.tune.ring = rip_ring(&r.text)?;
        cand_strs.ring = cand.tune.ring.name().to_owned();
    }
    if cand.epi == Epi::QkvzaGdn {
        return Err("epi qkvzagdn is not supported by pm_loop (120-byte fused GDN ABI)".into());
    }
    if cand.epi.is_silu() && m % 128 != 0 {
        return Err(format!(
            "epi {}: M={m} must be a multiple of 128 (production gate/up launcher)",
            cand.epi.name()
        ));
    }
    // The baseline is the incumbent: the default tune unless a `base-` generator knob is given.
    let mut base_strs = Strs {
        ring: "2".into(),
        wait: "after".into(),
        fold_op: "fma".into(),
        fetch: "split".into(),
        idle: "off".into(),
        ..cand_strs.clone()
    };
    let mut base_emitted = false;
    for (key, slot) in [
        ("base-fold", &mut base_strs.fold),
        ("base-tile", &mut base_strs.tile),
        ("base-cacc", &mut base_strs.cacc),
        ("base-alayout", &mut base_strs.alayout),
        ("base-mma", &mut base_strs.mma),
        ("base-ring", &mut base_strs.ring),
        ("base-wait", &mut base_strs.wait),
        ("base-fold-op", &mut base_strs.fold_op),
        ("base-fetch", &mut base_strs.fetch),
        ("base-idle", &mut base_strs.idle),
    ] {
        if let Some(v) = knobs.take(key) {
            *slot = v;
            base_emitted = true;
        }
    }
    let mut base = if base_emitted {
        base_strs.spec()?
    } else {
        // The shipped bundle: production tile, K128/cacc1, the candidate's layout and epilogue.
        base_strs = Strs {
            fold: "k128".into(),
            tile: "128x128x8".into(),
            cacc: "1".into(),
            mma: "dense".into(),
            ring: "2".into(),
            wait: "after".into(),
            fold_op: "fma".into(),
            fetch: "split".into(),
            idle: "off".into(),
            ..cand_strs.clone()
        };
        Spec {
            act: cand.act,
            ..Spec::control(cand.epi)
        }
    };
    // `--base-shipped v64`: the baseline is the shipped small-N `_v64` export (`verify64_tune`), the
    // object the 27B DFlash verify launches for n <= 128 rows; `b1s` (default) is the shipped
    // `_b1s` export.
    match knobs.take("base-shipped").as_deref() {
        None | Some("b1s") => {}
        Some(v @ "v64") => {
            if base_emitted {
                return Err(format!(
                    "--base-shipped {v} selects a shipped export; it excludes the emitted-baseline knobs (--base-<knob>)"
                ));
            }
            base = shipped_v64_spec(&cand)?;
            base_strs = Strs {
                wait: base.tune.wait.name().into(),
                fetch: base.tune.fetch.name().into(),
                idle: base.tune.idle.name().into(),
                alayout: "slab".into(),
                ..base_strs
            };
        }
        Some(o) => return Err(format!("--base-shipped {o}: expected b1s|v64")),
    }
    let capture = match knobs.take("inputs") {
        Some(spec) => {
            let c = Capture::parse(&spec)?;
            c.require_k(k)?;
            Some(c)
        }
        None => None,
    };
    let sparse_seed = match knobs.take("sparse-seed") {
        Some(v) => v
            .parse::<u64>()
            .map_err(|e| format!("--sparse-seed {v}: {e}"))?,
        None => DEFAULT_SPARSE_SEED,
    };
    let ref_fold = match knobs.take("ref-fold").as_deref() {
        None => None,
        Some("fma") => Some(RefFold::Fma),
        Some("muladd") => Some(RefFold::MulAdd),
        Some(o) => return Err(format!("--ref-fold {o}: expected fma|muladd")),
    };
    let macs = (m as u128) * (n as u128) * (k as u128);
    let mode = knobs.take_or(
        "ref-rows",
        if macs <= FULL_REF_MACS {
            "full"
        } else {
            "sample:192"
        },
    );
    let (rows, rows_desc) = if mode == "full" {
        ((0..m).collect(), "full".to_owned())
    } else if let Some(c) = mode.strip_prefix("sample:") {
        let c: usize = c.parse().map_err(|e| format!("--ref-rows {mode}: {e}"))?;
        (sample_rows(m, c.max(2)), format!("sample:{c}"))
    } else {
        return Err(format!("--ref-rows {mode}: expected full|sample:<n>"));
    };
    Ok(Box::new(Iu4 {
        m,
        n,
        k,
        cand,
        cand_strs,
        base,
        base_strs,
        base_emitted,
        rows,
        rows_desc,
        ref_fold,
        host: None,
        dev: None,
        reference: [OnceCell::new(), OnceCell::new()],
        sparse_seed,
        rip,
        cold: crate::cold::ColdMode::Hot,
        capture,
    }))
}

/// The shipped small-N export `<symbol>_b1s_v64` that matches the candidate's epilogue: the dense
/// slab 128x128x8 module exports `set`, `add` and the gate/up SiLU in `_v64` form.
fn shipped_v64_spec(cand: &Spec) -> Result<Spec> {
    if cand.act != ALayout::Slab
        || cand.tile != Tile::T128x128x8
        || cand.mma != Mma::Dense
        || !matches!(cand.epi, Epi::Set | Epi::Add | Epi::GateUpSilu)
    {
        return Err(format!(
            "--base-shipped v64: the shipped `_v64` exports are the dense slab 128x128x8 set|add|silu \
             kernels (got alayout {:?}, tile {:?}, mma {}, epi {})",
            cand.act,
            cand.tile,
            cand.mma.name(),
            cand.epi.name()
        ));
    }
    Ok(Spec {
        act: ALayout::Slab,
        tune: verify64_tune(cand.epi, false),
        ..Spec::control(cand.epi)
    })
}

fn emit_elf(spec: Spec) -> Result<(Vec<u8>, String)> {
    let e = iu4_gemm::emit(spec)?;
    Ok((
        peacemaker_isa::native::assemble(&e.s_text, spec.arch)?,
        spec.symbol(),
    ))
}

impl Iu4 {
    fn spec_of(&self, arm: Arm) -> Spec {
        match arm {
            Arm::Baseline => self.base,
            Arm::Candidate => self.cand,
        }
    }

    /// The CPU fold `arm`'s reference follows: `--ref-fold`, else the arm's own `fold-op`.
    fn ref_fold_of(&self, arm: Arm) -> RefFold {
        self.ref_fold
            .unwrap_or(match self.spec_of(arm).tune.fold_op {
                FoldOp::Fma => RefFold::Fma,
                FoldOp::MulAdd => RefFold::MulAdd,
            })
    }

    /// Either arm runs a 2:4 sparse MMA, so the generated weights are pruned.
    fn sparse_active(&self) -> bool {
        self.cand.mma != Mma::Dense || self.base.mma != Mma::Dense
    }

    fn out_bytes(&self) -> usize {
        self.n
            * self.m
            * if self.cand.epi == Epi::GateUpSiluBf16 {
                2
            } else {
                4
            }
    }

    fn rip_object(&self, rip: &RipSource) -> Result<Object> {
        let (elf, symbol, mut info) = compile_rip(&rip.text, self.cand)?;
        info["path"] = json!(rip.path);
        Ok(Object {
            source: json!({
                "source": format!("peacemaker_rip::compile({}) + native::assemble", rip.path),
                "variant": self.cand.variant_name(),
                "sha256": sha256_hex(&elf),
            }),
            desc: json!({"knobs": self.cand_strs.json(), "rip": info}),
            exports: vec![symbol.clone()],
            launch_symbol: symbol,
            bytes: elf,
        })
    }

    fn emit_object(&self, spec: Spec, knobs: Value) -> Result<Object> {
        let (elf, symbol) = emit_elf(spec)?;
        Ok(Object {
            source: json!({
                "source": "peacemaker_isa::kernels::iu4_gemm::emit + native::assemble (peacemaker-isa emit --co)",
                "variant": spec.variant_name(),
                "sha256": sha256_hex(&elf),
            }),
            desc: json!({"knobs": knobs}),
            exports: vec![symbol.clone()],
            launch_symbol: symbol,
            bytes: elf,
        })
    }

    fn shipped(&self) -> Result<Object> {
        let symbol = self.base.symbol();
        let (bytes, file, selection) = match self.base.act {
            ALayout::Token => (
                rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1,
                "kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1.hxaco",
                "crates/rdna-compute/src/gemm.rs:917-926 g12_iu4_b1_image \
                 (kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1); set/add kernel choice \
                 gemm.rs:21707-21726 (gemm_mq4g256v2_mmq_prequant_iu4), silu gemm.rs:36483-36530 \
                 (gemm_gate_up_silu_mq4g256v2_iu4_gfx12)",
            ),
            ALayout::Slab => (
                rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S,
                "kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s.hxaco",
                "crates/rdna-compute/src/gemm.rs:929-934 g12_iu4_b1s_image \
                 (kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S); set/add kernel choice \
                 gemm.rs:21707-21716 (slab arm), silu gemm.rs:36483-36500",
            ),
        };
        if !contains(bytes, symbol.as_bytes()) {
            return Err(format!(
                "symbol {symbol} is not in the shipped bundle {file}"
            ));
        }
        Ok(Object {
            bytes: bytes.to_vec(),
            launch_symbol: symbol.clone(),
            exports: vec![symbol],
            desc: json!({"artifact": file, "knobs": self.base_strs.json()}),
            source: json!({
                "source": selection,
                "artifact": file,
                "sha256": sha256_hex(bytes),
            }),
        })
    }

    /// Lift the candidate and, when the emulator accepts every instruction, run a
    /// SET/ADD GEMM (M=128, N=128, K=256) and compare against the CPU reference.
    fn run_emu(&self) -> Result<Value> {
        use peacemaker_ir::emu::{Emulator, Launch as ELaunch, Memory};
        // The candidate object itself (the generator's, or `--rip`'s).
        let obj = self.candidate()?;
        let (elf, symbol) = (obj.bytes, obj.launch_symbol);
        let lifted = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: peacemaker_ir::inst::Frontend::Builder,
            },
        )
        .map_err(|e| format!("lift: {e:?}"))?;
        // Reports the first opcode without semantics, if any.
        let emu = Emulator::new(&lifted.program, &symbol)?.hw_model();
        if !matches!(self.cand.epi, Epi::Set | Epi::Add) {
            return Err(format!(
                "emulator accepts every {symbol} opcode but pm_loop drives only set/add emulation"
            ));
        }
        let (m, n, k) = (128usize, 128usize, 256usize);
        let add = self.cand.epi == Epi::Add;
        let mut host = Host::synth(m, n, k, false);
        // A sparse candidate reads the packed image; the reference reads the pruned dense weights.
        let w_image = if self.cand.mma == Mma::Dense {
            host.w.clone()
        } else {
            let mma = self.cand.mma;
            prune_and_pack(&mut host.w, k, mma, self.sparse_seed, &[mma])?.remove(0)
        };
        let (kva, ava, xva, yva) = (
            0x1000_0000u64,
            0x2000_0000u64,
            0x3000_0000u64,
            0x4000_0000u64,
        );
        let xq = match self.cand.act {
            ALayout::Token => host.recs.clone(),
            ALayout::Slab => to_slab(&host.recs, n),
        };
        let mut kern = pack_set_add(ava, xva, yva, m, k, n, add);
        kern.resize(self.cand.kernargs().size as usize, 0);
        // ROCm fills the hidden block counts (grid size in workgroups) at launch; the
        // banded raster of the prologue divides by them.
        let tile = self.cand.tile;
        let grid = [
            m.div_ceil(tile.rows() as usize) as u32,
            n.div_ceil(128) as u32,
        ];
        for a in self.cand.kernargs().args {
            let g = match a.name.as_str() {
                "hidden_block_count_x" => grid[0],
                "hidden_block_count_y" => grid[1],
                _ => continue,
            };
            kern[a.offset as usize..a.offset as usize + 4].copy_from_slice(&g.to_le_bytes());
        }
        let y0: Vec<u8> = if add {
            host.y0.iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            vec![0; n * m * 4]
        };
        let mut mem = Memory::default();
        mem.map("kernarg".into(), kva, kern)?;
        for (name, va, bytes) in [("a", ava, w_image), ("xq", xva, xq), ("y", yva, y0)] {
            let len = bytes.len() as u64;
            mem.map(name.into(), va, bytes)?;
            mem.map(format!("{name}_tail"), va + len, vec![0; 4096])?;
        }
        for y in 0..n.div_ceil(128) {
            for x in 0..m.div_ceil(tile.rows() as usize) {
                emu.run_workgroup(
                    &ELaunch {
                        kernarg_ptr: kva,
                        workgroup: [x as u32, y as u32, 0],
                        block: [tile.threads(), 1, 1],
                        dynamic_lds: self.cand.lds_launch(),
                    },
                    &mut mem,
                )?;
            }
        }
        for name in ["a_tail", "xq_tail", "y_tail"] {
            let r = mem.regions.iter().find(|r| r.name == name);
            if r.is_some_and(|r| r.bytes.iter().any(|&b| b != 0)) {
                return Err(format!("kernel wrote past the end of the {name} buffer"));
            }
        }
        let got = mem.read(yva, n * m * 4)?;
        let rows: Vec<usize> = (0..m).collect();
        let want = reference(&host, m, n, k, add, &rows, self.ref_fold_of(Arm::Candidate));
        let mismatches = (0..m)
            .flat_map(|r| (0..n).map(move |c| (r, c)))
            .filter(|&(r, c)| {
                let o = (c * m + r) * 4;
                got[o..o + 4] != want[r * n + c].to_le_bytes()
            })
            .count();
        Ok(json!({
            "ran": true,
            "shape": {"M": m, "N": n, "K": k},
            "symbol": symbol,
            "reference": "cpu_iu4_symfold_rn",
            "mma": self.cand.mma.name(),
            "mismatching_elements": mismatches,
            "bit_exact": mismatches == 0,
        }))
    }
}

pub fn max_diff(a: &[u8], b: &[u8], width: usize) -> f64 {
    let val = |c: &[u8]| -> f32 {
        if width == 2 {
            f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
        } else {
            f32::from_le_bytes(c.try_into().unwrap())
        }
    };
    a.chunks_exact(width)
        .zip(b.chunks_exact(width))
        .map(|(x, y)| elem_diff(val(x), val(y)))
        .fold(0f64, f64::max)
}

fn elem_diff(x: f32, y: f32) -> f64 {
    if x.is_nan() != y.is_nan() {
        f64::INFINITY
    } else if x.is_nan() || x == y {
        0.0
    } else {
        (f64::from(x) - f64::from(y)).abs()
    }
}

impl Iu4 {
    /// Bytes of one weight set a dispatch streams (gate + up for the SiLU epilogue).
    fn set_bytes(&self) -> usize {
        self.m * (self.k / 256) * GROUP_BYTES * if self.cand.epi.is_silu() { 2 } else { 1 }
    }

    /// Weight buffers of pool set `set` (`0`: the correctness set).
    fn set_weights(
        dev: &Dev,
        set: usize,
    ) -> (&hip_bridge::DeviceBuffer, Option<&hip_bridge::DeviceBuffer>) {
        match set {
            0 => (&dev.w, dev.w_up.as_ref()),
            s => {
                let (w, up) = &dev.pool[s - 1];
                (w, up.as_ref())
            }
        }
    }
}

impl Workload for Iu4 {
    fn kernel(&self) -> &'static str {
        "iu4_gemm"
    }
    fn arch(&self) -> &'static str {
        "gfx1201"
    }
    fn shape_json(&self) -> Value {
        let mut v = json!({"M": self.m, "N": self.n, "K": self.k,
               "launch_rows": if self.cand.epi.is_silu() { 2 * self.m } else { self.m }});
        // Cold-cache timing is part of the shape (a `pm-census noise` group never matches across
        // modes); a hot run's shape keeps its bytes.
        crate::cold::annotate_shape(&mut v, self.cold, self.pool_sets(), self.set_bytes());
        v
    }
    fn record_extra(&self) -> Vec<(String, Value)> {
        let mut extra = Vec::new();
        if let Some(c) = &self.capture {
            extra.push(("inputs".to_owned(), c.json()));
        }
        if !self.sparse_active() {
            return extra;
        }
        extra.push((
            "sparse".into(),
            json!({
                "mma": self.cand.mma.name(), "base_mma": self.base.mma.name(),
                "groups_per_block": SPARSE_GROUPS, "kept_per_group": SPARSE_KEPT,
                "seed": self.sparse_seed,
                "weights": "generated MQ4V2 codes pruned 2-of-4 per sparse group, pruned code = 8; \
                    dense arms and the CPU reference read the pruned dense weights, sparse arms the pack_block image",
            }),
        ));
        extra
    }
    fn candidate_knobs(&self) -> Value {
        let mut v = self.cand_strs.json();
        v["variant"] = json!(self.cand.variant_name());
        v
    }

    fn candidate(&self) -> Result<Object> {
        if let Some(rip) = &self.rip {
            return self.rip_object(rip);
        }
        self.emit_object(self.cand, self.cand_strs.json())
    }

    fn baseline(&self) -> Result<Object> {
        if self.base_emitted {
            self.emit_object(self.base, self.base_strs.json())
        } else {
            self.shipped()
        }
    }

    fn setup(&mut self, gpu: &Gpu) -> Result<()> {
        let silu = self.cand.epi.is_silu();
        let mut host = Host::synth(self.m, self.n, self.k, silu);
        if let Some(c) = &self.capture {
            host.recs = recs_from_capture(c, self.n, self.k);
        }
        // Any sparse arm: prune the generated weights to 2:4 (in `host.w`, which
        // the dense baseline and the CPU reference read) and pack one image per
        // sparse arm.
        let sparse: Vec<(usize, Mma)> = [(0, self.base.mma), (1, self.cand.mma)]
            .into_iter()
            .filter(|(_, mma)| *mma != Mma::Dense)
            .collect();
        let mut images = Vec::new();
        if let Some(&(_, prune_mma)) = sparse.first() {
            let kinds: Vec<Mma> = sparse.iter().map(|(_, mma)| *mma).collect();
            images = prune_and_pack(&mut host.w, self.k, prune_mma, self.sparse_seed, &kinds)?;
        }
        let up = |b: &[u8]| -> Result<hip_bridge::DeviceBuffer> {
            let d = gpu.hip.malloc(b.len().max(16)).map_err(dev_err)?;
            if !b.is_empty() {
                gpu.hip.memcpy_htod(&d, b).map_err(dev_err)?;
            }
            Ok(d)
        };
        let out = self.out_bytes().max(16);
        let mut w_packed = [None, None];
        for ((arm, _), image) in sparse.iter().zip(&images) {
            w_packed[*arm] = Some(up(image)?);
        }
        // `--flush`: the timing pool, sets 1.. (set 0 is the correctness set above).
        let mut pool = Vec::new();
        if self.cold.pool() {
            for s in 1..self.pool_sets() {
                let w = up(&synth_weights(self.m, self.k, pool_seed(GATE_SEED, s)))?;
                let w_up = if silu {
                    Some(up(&synth_weights(self.m, self.k, pool_seed(UP_SEED, s)))?)
                } else {
                    None
                };
                pool.push((w, w_up));
            }
        }
        let dev = Dev {
            w: up(&host.w)?,
            w_packed,
            w_up: if silu { Some(up(&host.w_up)?) } else { None },
            xq_token: up(&host.recs)?,
            xq_slab: up(&to_slab(&host.recs, self.n))?,
            y: [
                gpu.hip.malloc(out).map_err(dev_err)?,
                gpu.hip.malloc(out).map_err(dev_err)?,
            ],
            pool,
        };
        self.host = Some(host);
        self.dev = Some(dev);
        for arm in [Arm::Baseline, Arm::Candidate] {
            self.reset(gpu, arm)?;
        }
        Ok(())
    }

    fn launch(&self, arm: Arm) -> Launch {
        self.launch_set(arm, 0)
    }

    fn set_cold(&mut self, mode: crate::cold::ColdMode) -> Result<()> {
        if mode.pool() && self.sparse_active() {
            return Err(format!(
                "{} does not support a 2:4 sparse arm (--mma/--base-mma): the packed image has no weight pool",
                mode.flag()
            ));
        }
        self.cold = mode;
        Ok(())
    }

    fn null_object(&self) -> Result<Option<Object>> {
        let e = iu4_gemm::emit(self.base)?;
        crate::common::null_object(
            &e.s_text,
            self.base.arch,
            &self.base.symbol(),
            "the baseline spec's emit with the body replaced by s_endpgm (descriptor unchanged)",
        )
        .map(Some)
    }

    fn pool_sets(&self) -> usize {
        if self.cold.pool() {
            crate::cold::pool_sets(self.set_bytes())
        } else {
            1
        }
    }

    fn launch_set(&self, arm: Arm, set: usize) -> Launch {
        let dev = self.dev.as_ref().expect("launch before setup");
        let spec = self.spec_of(arm);
        let ptr = |b: &hip_bridge::DeviceBuffer| b.as_ptr() as u64;
        let xq = ptr(match spec.act {
            ALayout::Token => &dev.xq_token,
            ALayout::Slab => &dev.xq_slab,
        });
        let y = ptr(&dev.y[arm_index(arm)]);
        let rows = if spec.epi.is_silu() {
            2 * self.m
        } else {
            self.m
        };
        // A sparse arm reads its packed image, every other arm the (pruned) dense weights.
        let (w_set, up_set) = Self::set_weights(dev, set);
        let w = dev.w_packed[arm_index(arm)].as_ref().unwrap_or(w_set);
        let kernarg = match up_set {
            Some(up) => pack_silu(ptr(w), ptr(up), xq, y, self.m, self.k, self.n),
            None => pack_set_add(ptr(w), xq, y, self.m, self.k, self.n, spec.epi == Epi::Add),
        };
        Launch {
            kernarg,
            grid: [
                rows.div_ceil(spec.tile.rows() as usize) as u32,
                self.n.div_ceil(128) as u32,
                1,
            ],
            block: [spec.tile.threads(), 1, 1],
            lds: spec.lds_launch(),
        }
    }

    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()> {
        let dev = self.dev.as_ref().ok_or("reset before setup")?;
        let y = &dev.y[arm_index(arm)];
        if self.cand.epi == Epi::Add {
            let host = self.host.as_ref().ok_or("reset before setup")?;
            let bytes: Vec<u8> = host.y0.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.hip.memcpy_htod(y, &bytes).map_err(dev_err)
        } else {
            gpu.hip.memset(y, 0, y.size()).map_err(dev_err)
        }
    }

    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>> {
        let dev = self.dev.as_ref().ok_or("output before setup")?;
        // Device buffers are padded to 16 bytes; the payload is the leading `out_bytes`.
        let y = &dev.y[arm_index(arm)];
        let mut v = vec![0u8; y.size()];
        gpu.hip.memcpy_dtoh(&mut v, y).map_err(dev_err)?;
        v.truncate(self.out_bytes());
        Ok(v)
    }

    fn check(&self, out: &[u8]) -> Result<RefReport> {
        self.check_arm(Arm::Candidate, out)
    }

    fn check_arm(&self, arm: Arm, out: &[u8]) -> Result<RefReport> {
        if self.cand.epi.is_silu() {
            return Err(format!(
                "no CPU reference for epi {} (v_exp-approximated SiLU region not reproduced); \
                 correctness is candidate-vs-baseline only",
                self.cand.epi.name()
            ));
        }
        if out.len() != self.out_bytes() {
            return Err(format!(
                "output is {} bytes, expected {}",
                out.len(),
                self.out_bytes()
            ));
        }
        let host = self.host.as_ref().ok_or("check before setup")?;
        let add = self.cand.epi == Epi::Add;
        let fold = self.ref_fold_of(arm);
        let want = self.reference[fold as usize]
            .get_or_init(|| reference(host, self.m, self.n, self.k, add, &self.rows, fold));
        let (mut mismatches, mut worst) = (0usize, 0f64);
        for (ri, &r) in self.rows.iter().enumerate() {
            for col in 0..self.n {
                let o = (col * self.m + r) * 4;
                let got = f32::from_le_bytes(out[o..o + 4].try_into().unwrap());
                let exp = want[ri * self.n + col];
                mismatches += usize::from(got.to_bits() != exp.to_bits());
                worst = worst.max(elem_diff(got, exp));
            }
        }
        Ok(RefReport {
            reference: format!(
                "cpu_iu4_symfold_rn(rows={},fold={}{},mismatching_elements={mismatches})",
                self.rows_desc,
                match fold {
                    RefFold::Fma => "fma",
                    RefFold::MulAdd => "muladd",
                },
                if self.sparse_active() {
                    ",weights=pruned_2of4"
                } else {
                    ""
                }
            ),
            max_abs_err: worst,
            tolerance: 0.0,
            pass: mismatches == 0,
        })
    }

    fn numeric_contract(&self) -> Value {
        if self.cand.tune.fold_op == self.base.tune.fold_op {
            return Value::Null;
        }
        let name = |s: Spec| match s.tune.fold_op {
            FoldOp::Fma => "fma(t, float(C) - magic, acc) (hipcc-K1-exact builder fold)",
            FoldOp::MulAdd => "acc + RN(t * (float(C) - magic)) (non-fused v_mul_f32 + v_add_f32)",
        };
        json!({
            "baseline": name(self.base),
            "candidate": name(self.cand),
            "relation": "numeric",
        })
    }

    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(format!("length mismatch {} vs {}", a.len(), b.len()));
        }
        let width = if self.cand.epi == Epi::GateUpSiluBf16 {
            2
        } else {
            4
        };
        Ok(max_diff(a, b, width))
    }

    fn emu(&self) -> Value {
        match self.run_emu() {
            Ok(v) => v,
            Err(e) => Value::String(format!("refused ({e})")),
        }
    }
}

#[cfg(test)]
mod capture_tests {
    use super::*;
    use std::path::PathBuf;

    /// Write a synthetic `act_capture.v1` (rows-only bin, site `a` rows 3xK, site `b` 2xK).
    fn fixture(tag: &str, k: usize, tamper: bool) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pm_loop_cap_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bin = Vec::new();
        for i in 0..5 * k {
            bin.extend((((i % 97) as f32 - 48.0) * 0.05).to_le_bytes());
        }
        let sha = sha256_hex(&bin);
        if tamper {
            bin[0] ^= 1;
        }
        std::fs::write(dir.join("c.bin"), &bin).unwrap();
        let doc = json!({
            "schema": "act_capture.v1",
            "sites": [
                {"site": "a", "k": k, "rows_total": 3, "records": 1, "offset_bytes": 0},
                {"site": "b", "k": k, "rows_total": 2, "records": 1, "offset_bytes": 3 * k * 4},
            ],
            "bin": {"path": "c.bin", "sha256": sha},
            "id": "ab".repeat(32),
        });
        let p = dir.join("c.json");
        std::fs::write(&p, serde_json::to_vec(&doc).unwrap()).unwrap();
        p
    }

    #[test]
    fn capture_replay_is_deterministic_and_cycles() {
        let p = fixture("det", 256, false);
        let a = Capture::load(&p, Some("a")).unwrap();
        let b = Capture::load(&p, Some("a")).unwrap();
        assert_eq!(a, b);
        assert_eq!((a.nrows, a.k), (3, 256));
        assert_eq!(a.row(3), a.row(0));
        assert_ne!(a.row(0), Capture::load(&p, Some("b")).unwrap().row(0));
        let spec = format!("capture:{}@b", p.display());
        assert_eq!(Capture::parse(&spec).unwrap().site, "b");
        assert_eq!(a.json()["kind"], "capture");
        assert_eq!(a.json()["id"], "ab".repeat(32));
        assert_eq!(recs_from_capture(&a, 4, 256), recs_from_capture(&b, 4, 256));
        assert!(Capture::load(&p, None).is_err(), "two sites need @site");
    }

    #[test]
    fn capture_sha_mismatch_is_refused() {
        let p = fixture("sha", 256, true);
        let e = Capture::load(&p, Some("a")).unwrap_err();
        assert!(e.contains("sha256 mismatch"), "{e}");
    }

    #[test]
    fn capture_k_mismatch_is_refused() {
        let p = fixture("k", 256, false);
        let spec = format!("capture:{}@a", p.display());
        let e = new(
            "M=128,N=8,K=512",
            &mut Knobs([("inputs".to_owned(), spec)].into()),
        )
        .err()
        .unwrap();
        assert!(e.contains("K=256") && e.contains("K=512"), "{e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knobs(kv: &[(&str, &str)]) -> Knobs {
        Knobs(kv.iter().map(|(k, v)| ((*k).into(), (*v).into())).collect())
    }

    #[test]
    fn shape_grammar() {
        assert_eq!(
            parse_shape("M=4096,N=512,K=5120").unwrap(),
            (4096, 512, 5120)
        );
        assert_eq!(parse_shape("K=256,N=1,M=4").unwrap(), (4, 1, 256));
        assert!(parse_shape("M=4,N=1").is_err());
        assert!(parse_shape("M=4,N=1,K=256,K=256").is_err());
        assert!(parse_shape("M=4,N=1,Q=3").is_err());
        assert!(new("M=4,N=1,K=128", &mut knobs(&[])).is_err());
        assert!(new("M=6,N=1,K=256", &mut knobs(&[])).is_err());
        assert!(new("M=64,N=1,K=256", &mut knobs(&[("epi", "silu")])).is_err());
        assert!(new("M=128,N=1,K=256", &mut knobs(&[("epi", "qkvzagdn")])).is_err());
    }

    const TWIN: &str = include_str!("../../../peacemaker-isa/src/kernels/iu4_gemm.rip");

    fn default_spec() -> Spec {
        Spec::from_knobs("k128", "128x128x8", "1", "set", "token", Arch::Gfx1201).unwrap()
    }

    #[test]
    fn rip_twin_is_identical_to_generator() {
        let (_, symbol, info) = compile_rip(TWIN, default_spec()).unwrap();
        assert_eq!(symbol, default_spec().symbol());
        assert_eq!(info["identical_to_generator"], true);
        assert_eq!(info["variant"], default_spec().variant_name());
        assert_eq!(info["sha256"], sha256_hex(TWIN.as_bytes()));
    }

    #[test]
    fn rip_edit_passes_abi_and_reports_difference() {
        let edited = TWIN.replacen(
            "    op(wg.isa(), \"s_setprio 1\", vec![], vec![])?;\n",
            "",
            1,
        );
        assert_ne!(edited, TWIN, "the s_setprio 1 line must exist in the twin");
        let (_, _, info) = compile_rip(&edited, default_spec()).unwrap();
        assert_eq!(info["identical_to_generator"], false);
    }

    #[test]
    fn rip_symbol_change_is_refused() {
        // The twin derives the kernel (and so the symbol) from the variant it
        // parses: make `set` build the `add` kernel.
        let renamed = TWIN.replacen("\"set\" => Epi::Set,", "\"set\" => Epi::Add,", 1);
        assert_ne!(renamed, TWIN, "the epilogue arm must exist in the twin");
        let spec = default_spec();
        let e = compile_rip(&renamed, spec).unwrap_err();
        assert!(
            e.contains("does not expose") || e.contains("ABI mismatch"),
            "{e}"
        );
    }

    #[test]
    fn kernarg_layout() {
        let b = pack_set_add(0x11, 0x22, 0x33, 7, 256, 5, true);
        assert_eq!(b.len(), 40);
        assert_eq!(u64::from_le_bytes(b[0..8].try_into().unwrap()), 0x11);
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 0x22);
        assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 0x33);
        for (off, v) in [(24, 7), (28, 256), (32, 5), (36, 1)] {
            assert_eq!(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()), v);
        }
        let s = pack_silu(1, 2, 3, 4, 128, 256, 9);
        assert_eq!(s.len(), 44);
        assert_eq!(u64::from_le_bytes(s[24..32].try_into().unwrap()), 4);
        for (off, v) in [(32, 128), (36, 256), (40, 9)] {
            assert_eq!(i32::from_le_bytes(s[off..off + 4].try_into().unwrap()), v);
        }
    }

    #[test]
    fn knob_to_spec_and_symbol() {
        let w = new(
            "M=128,N=64,K=256",
            &mut knobs(&[("epi", "add"), ("alayout", "slab")]),
        )
        .unwrap();
        assert_eq!(w.kernel(), "iu4_gemm");
        assert_eq!(
            w.candidate_knobs()["variant"],
            "k128-128x128x8-cacc1-slab-add"
        );
        let spec =
            Spec::from_knobs("k128", "128x128x8", "1", "add", "slab", Arch::Gfx1201).unwrap();
        assert_eq!(
            spec.symbol(),
            "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s"
        );
        let t = Spec::from_knobs("k128", "256x128x16", "1", "set", "token", Arch::Gfx1201).unwrap();
        assert_eq!(
            t.symbol(),
            "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1t256"
        );
        assert_eq!(t.tile.layout().launch, 30720);
        assert_eq!(t.tile.threads(), 512);
    }

    #[test]
    fn closed_axes_surface_the_emit_error() {
        let w = new("M=128,N=64,K=256", &mut knobs(&[("fold", "k256shared")])).unwrap();
        assert!(w.candidate().unwrap_err().contains("closed"));
        let w = new("M=128,N=64,K=256", &mut knobs(&[("cacc", "2")])).unwrap();
        assert!(w.candidate().unwrap_err().contains("Cacc::Two"));
    }

    #[test]
    fn base_knobs_select_emitted_baseline() {
        let mut k = knobs(&[("tile", "256x128x16"), ("base-tile", "128x128x8")]);
        let w = new("M=128,N=64,K=256", &mut k).unwrap();
        assert!(k.0.is_empty());
        assert_eq!(w.candidate_knobs()["tile"], "256x128x16");
    }

    #[test]
    fn sampled_rows_cover_edges() {
        let r = sample_rows(1000, 192);
        assert!(r.starts_with(&[0, 1, 2]));
        assert_eq!(*r.last().unwrap(), 999);
        assert!(r.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(sample_rows(10, 4), (0..10).collect::<Vec<_>>());
    }

    /// Hand-checkable: every weight nibble 9 (signed +1), every x code 1, so
    /// dot = 128 per K128 block; sc = 1.0, d = 0.5 -> t1 = 0.5, term = 64, two blocks.
    #[test]
    fn reference_hand_case() {
        let (m, n, k) = (1usize, 1usize, 256usize);
        let mut w = vec![0u8; GROUP_BYTES];
        for h in 0..2 {
            w[4 * h..4 * h + 2].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        }
        w[8..].fill(0x99);
        let mut recs = Vec::new();
        for _ in 0..2 {
            recs.extend_from_slice(&0.5f32.to_le_bytes());
            recs.extend_from_slice(&128i32.to_le_bytes());
            recs.extend_from_slice(&[0x11; 64]);
        }
        let host = Host {
            w,
            w_up: vec![],
            recs,
            y0: vec![1.5],
        };
        for fold in [RefFold::Fma, RefFold::MulAdd] {
            assert_eq!(reference(&host, m, n, k, false, &[0], fold), vec![128.0]);
            assert_eq!(reference(&host, m, n, k, true, &[0], fold), vec![129.5]);
        }
    }

    #[test]
    fn negative_codes_and_xor_fold() {
        // weight nibble 0 -> signed -8, x code -8 (nibble 0x8): dot = 128 * 64 = 8192 per block, d = 1.
        let mut w = vec![0u8; GROUP_BYTES];
        for h in 0..2 {
            w[4 * h..4 * h + 2].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        }
        let mut recs = Vec::new();
        for _ in 0..2 {
            recs.extend_from_slice(&1.0f32.to_le_bytes());
            recs.extend_from_slice(&(-1024i32).to_le_bytes());
            recs.extend_from_slice(&[0x88; 64]);
        }
        let host = Host {
            w,
            w_up: vec![],
            recs,
            y0: vec![0.0],
        };
        assert_eq!(
            reference(&host, 1, 1, 256, false, &[0], RefFold::Fma),
            vec![2.0 * 8192.0]
        );
    }

    #[test]
    fn slab_planes_hold_the_same_bytes() {
        let n = 3;
        let host = Host::synth(4, n, 256, false);
        let slab = to_slab(&host.recs, n);
        assert_eq!(slab.len(), host.recs.len());
        let (kb, col) = (1, 2);
        let rec = &host.recs[(kb * n + col) * REC..][..REC];
        let base = kb * REC * n;
        assert_eq!(&slab[base + 4 * col..][..4], &rec[0..4]);
        assert_eq!(&slab[base + 4 * n + 4 * col..][..4], &rec[4..8]);
        assert_eq!(&slab[base + 8 * n + 32 * col..][..32], &rec[8..40]);
        assert_eq!(&slab[base + 40 * n + 32 * col..][..32], &rec[40..72]);
    }

    #[test]
    fn synthetic_inputs_are_deterministic_and_diff_is_nan_aware() {
        let a = Host::synth(8, 4, 256, false);
        let b = Host::synth(8, 4, 256, false);
        assert_eq!(a.w, b.w);
        assert_eq!(a.recs, b.recs);
        assert_eq!(
            max_diff(&1.0f32.to_le_bytes(), &f32::NAN.to_le_bytes(), 4),
            f64::INFINITY
        );
        assert_eq!(
            max_diff(&1.0f32.to_le_bytes(), &3.0f32.to_le_bytes(), 4),
            2.0
        );
    }
    const SPARSE: [Mma; 2] = [Mma::S24X64, Mma::S24X32];

    fn nibbles(blk: &[u8]) -> [u8; 128] {
        std::array::from_fn(|k| (blk[k / 2] >> (4 * (k % 2))) & 0xf)
    }

    #[test]
    fn prune_block_keeps_two_per_group_and_zeroes_the_rest() {
        for mma in SPARSE {
            let orig: [u8; 128] = std::array::from_fn(|k| ((k * 7 + 3) % 16) as u8);
            let mut codes = orig;
            let kept = prune_block(&mut codes, mma, 42);
            for g in 0..SPARSE_GROUPS {
                let members: Vec<usize> = (0..128)
                    .filter(|&k| sparse24::group_of(mma, k) == g)
                    .collect();
                assert_eq!(members.len(), 4, "{mma:?} group {g}");
                assert_eq!(
                    members.iter().filter(|&&k| kept[k]).count(),
                    SPARSE_KEPT,
                    "{mma:?} group {g}"
                );
                for &k in &members {
                    assert_eq!(codes[k], if kept[k] { orig[k] } else { 8 }, "{mma:?} k={k}");
                }
            }
            let mut again = orig;
            assert_eq!(prune_block(&mut again, mma, 42), kept, "deterministic");
            assert_eq!(again, codes);
            let mut other = orig;
            assert_ne!(
                prune_block(&mut other, mma, 43),
                kept,
                "seed changes the mask"
            );
        }
    }

    /// Both packers' groups are the same dense-K sets, so one pruning serves both.
    #[test]
    fn x64_and_x32_groups_are_the_same_sets() {
        let part = |mma| {
            let mut sets: Vec<Vec<usize>> = (0..SPARSE_GROUPS)
                .map(|g| {
                    (0..128)
                        .filter(|&k| sparse24::group_of(mma, k) == g)
                        .collect()
                })
                .collect();
            sets.sort();
            sets
        };
        assert_eq!(part(Mma::S24X64), part(Mma::S24X32));
    }

    #[test]
    fn prune_and_pack_prunes_dense_and_replaces_only_code_bytes() {
        let (m, k) = (6usize, 512usize);
        let dense = Host::synth(m, 1, k, false).w;
        let mut pruned = dense.clone();
        let packed = prune_and_pack(&mut pruned, k, Mma::S24X64, 7, &SPARSE).unwrap();
        assert_eq!(packed.len(), 2);
        for image in &packed {
            assert_eq!(image.len(), dense.len());
            for (b, blk) in image.chunks_exact(GROUP_BYTES).enumerate() {
                assert_eq!(&blk[..8], &dense[b * GROUP_BYTES..][..8], "header {b}");
            }
        }
        for (r, row) in pruned.chunks_exact(k / 256 * GROUP_BYTES).enumerate() {
            for (g, blk) in row.chunks_exact(GROUP_BYTES).enumerate() {
                assert_eq!(&blk[..8], &dense[(r * (k / 256) + g) * GROUP_BYTES..][..8]);
                for h in 0..2 {
                    let codes = nibbles(&blk[8 + 64 * h..][..64]);
                    for grp in 0..SPARSE_GROUPS {
                        let zeros = (0..128)
                            .filter(|&kk| sparse24::group_of(Mma::S24X64, kk) == grp)
                            .filter(|&kk| codes[kk] == 8)
                            .count();
                        assert!(
                            zeros >= 2,
                            "row {r} g {g} h {h} group {grp}: pruned dense keeps <=2"
                        );
                    }
                }
            }
        }
        assert_ne!(packed[0], pruned, "x64 image is not the dense layout");
        // Deterministic for a fixed seed, different for another.
        let mut again = dense.clone();
        assert_eq!(
            prune_and_pack(&mut again, k, Mma::S24X64, 7, &SPARSE).unwrap(),
            packed
        );
        assert_eq!(again, pruned);
        let mut other = dense.clone();
        prune_and_pack(&mut other, k, Mma::S24X64, 8, &SPARSE).unwrap();
        assert_ne!(other, pruned);
    }

    #[test]
    fn mma_knobs_reach_the_spec_and_the_record() {
        let w = new("M=128,N=64,K=256", &mut knobs(&[("mma", "s24x64")])).unwrap();
        assert_eq!(w.candidate_knobs()["mma"], "s24x64");
        assert!(w.candidate_knobs()["variant"]
            .as_str()
            .unwrap()
            .ends_with("-s24x64"));
        let extra = w.record_extra();
        assert_eq!(extra.len(), 1);
        let (key, v) = &extra[0];
        assert_eq!(key, "sparse");
        assert_eq!(
            (v["mma"].as_str(), v["base_mma"].as_str()),
            (Some("s24x64"), Some("dense"))
        );
        assert_eq!(
            (v["groups_per_block"].as_u64(), v["kept_per_group"].as_u64()),
            (Some(32), Some(2))
        );
        assert_eq!(v["seed"].as_u64(), Some(DEFAULT_SPARSE_SEED));
        // Dense candidate vs sparse emitted baseline is also a sparse run.
        let mut k = knobs(&[("base-mma", "s24x32"), ("sparse-seed", "9")]);
        let w = new("M=128,N=64,K=256", &mut k).unwrap();
        assert!(k.0.is_empty());
        let v = &w.record_extra()[0].1;
        assert_eq!(
            (
                v["mma"].as_str(),
                v["base_mma"].as_str(),
                v["seed"].as_u64()
            ),
            (Some("dense"), Some("s24x32"), Some(9))
        );
        // Dense runs carry no sparse block.
        assert!(new("M=128,N=64,K=256", &mut knobs(&[]))
            .unwrap()
            .record_extra()
            .is_empty());
        assert!(new("M=128,N=64,K=256", &mut knobs(&[("mma", "bogus")])).is_err());
        assert!(new("M=128,N=64,K=256", &mut knobs(&[("sparse-seed", "x")])).is_err());
    }

    #[test]
    fn sparse_axes_the_generator_refuses_surface_its_error() {
        let w = new(
            "M=128,N=64,K=256",
            &mut knobs(&[("mma", "s24x64"), ("tile", "256x128x16")]),
        )
        .unwrap();
        assert!(w.candidate().unwrap_err().to_lowercase().contains("tile"));
        let w = new(
            "M=128,N=64,K=256",
            &mut knobs(&[("mma", "s24x32"), ("alayout", "slab")]),
        )
        .unwrap();
        assert!(w.candidate().is_err());
    }

    /// The emulator runs the dense and both sparse kernels to completion at
    /// M=128, N=128, K=256 and matches the CPU reference bit for bit (SET and ADD).
    #[test]
    fn emu_runs_dense_and_sparse_bit_exact() {
        for epi in ["set", "add"] {
            for mma in ["dense", "s24x64", "s24x32"] {
                let w = new(
                    "M=128,N=128,K=256",
                    &mut knobs(&[("mma", mma), ("epi", epi)]),
                )
                .unwrap();
                let v = w.emu();
                assert_eq!(v["ran"], true, "{epi} {mma}: {v}");
                assert_eq!(v["bit_exact"], true, "{epi} {mma}: {v}");
                assert_eq!(v["mma"], mma);
            }
        }
    }

    #[test]
    fn generator_knobs_reach_the_spec_and_the_record_only_when_non_default() {
        let shape = "M=128,N=64,K=256";
        let plain = new(shape, &mut knobs(&[])).unwrap();
        let k = plain.candidate_knobs();
        for key in ["ring", "wait", "fold-op"] {
            assert!(
                k.get(key).is_none(),
                "{key} is recorded only when non-default"
            );
        }
        assert!(plain.numeric_contract().is_null());
        let mut kn = knobs(&[("wait", "mid"), ("ring", "2"), ("fold-op", "fma")]);
        let w = new(shape, &mut kn).unwrap();
        assert!(kn.0.is_empty(), "{:?}", kn.0);
        let k = w.candidate_knobs();
        assert_eq!(k["wait"], "mid");
        assert!(k.get("ring").is_none() && k.get("fold-op").is_none());
        assert_eq!(k["variant"], "k128-128x128x8-cacc1-set-wmid");
        // A schedule knob changes the emitted code, not the declared numerics.
        assert!(w.numeric_contract().is_null());
        let (a, b) = (w.candidate().unwrap(), plain.candidate().unwrap());
        assert_ne!(a.bytes, b.bytes);
        assert_eq!(a.launch_symbol, b.launch_symbol);
        // The baseline stays the incumbent unless a `base-` generator knob is given.
        let w = new(
            shape,
            &mut knobs(&[("wait", "before"), ("base-tile", "128x128x8")]),
        )
        .unwrap();
        assert_eq!(w.baseline().unwrap().desc["knobs"].get("wait"), None);
        let w = new(
            shape,
            &mut knobs(&[("base-tile", "128x128x8"), ("base-wait", "mid")]),
        )
        .unwrap();
        assert_eq!(w.baseline().unwrap().desc["knobs"]["wait"], "mid");
        for (key, bad) in [("ring", "4"), ("wait", "late"), ("fold-op", "fused")] {
            assert!(
                new(shape, &mut knobs(&[(key, bad)])).is_err(),
                "{key} {bad}"
            );
        }
    }

    #[test]
    fn ring_three_emits_and_surfaces_the_generators_reason_where_it_has_none() {
        let w = new("M=128,N=64,K=256", &mut knobs(&[("ring", "3")])).unwrap();
        let c = w.candidate().unwrap();
        assert_eq!(
            c.launch_symbol,
            "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1"
        );
        assert_eq!(w.candidate_knobs()["ring"], "3");
        // The depth-3 launch passes the shipped dynamic size; the third slab is static LDS in the descriptor.
        let spec =
            Spec::from_knobs("k128", "128x128x8", "1", "set", "token", Arch::Gfx1201).unwrap();
        assert_eq!(spec.lds_launch(), 20480);
        let r3 = Spec {
            tune: Tune {
                ring: RingDepth::Three,
                ..Tune::default()
            },
            ..spec
        };
        assert_eq!(
            (r3.lds_launch(), r3.lds_fixed(), r3.lds_total()),
            (20480, 8192, 28672)
        );
        // The sparse K-loop blocks have no depth-3 form.
        let w = new(
            "M=128,N=64,K=256",
            &mut knobs(&[("ring", "3"), ("mma", "s24x64")]),
        )
        .unwrap();
        let e = w.candidate().unwrap_err();
        assert!(e.contains("ring depth 3") && e.contains("dense"), "{e}");
    }

    #[test]
    fn a_rip_scripts_ring_constant_sets_the_candidates_depth() {
        assert_eq!(
            rip_ring("const RING: usize = 3;\nfn emit() {}").unwrap(),
            RingDepth::Three
        );
        assert_eq!(
            rip_ring("  const RING: usize = 2;").unwrap(),
            RingDepth::Two
        );
        assert_eq!(rip_ring("fn emit() {}").unwrap(), RingDepth::Two);
        assert!(rip_ring("const RING: usize = 4;").is_err());
    }

    #[test]
    fn generator_knobs_refuse_a_rip_candidate() {
        let dir = std::env::temp_dir().join(format!("iu4-rip-tune-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.rip");
        std::fs::write(&p, "fn emit() {}").unwrap();
        let e = new_with_rip("M=128,N=64,K=256", &mut knobs(&[("wait", "mid")]), Some(&p))
            .err()
            .unwrap();
        assert!(e.contains("--rip"), "{e}");
    }

    #[test]
    fn muladd_is_a_declared_numeric_candidate_checked_against_its_own_fold() {
        let shape = "M=128,N=128,K=256";
        for epi in ["set", "add", "silu"] {
            let w = new(shape, &mut knobs(&[("fold-op", "muladd"), ("epi", epi)])).unwrap();
            let c = w.numeric_contract();
            assert_eq!(c["relation"], "numeric", "{epi}");
            assert!(c["candidate"].as_str().unwrap().contains("v_mul_f32"));
            assert!(c["baseline"].as_str().unwrap().contains("fma"));
            assert_eq!(w.candidate_knobs()["fold-op"], "muladd");
            assert!(!w.requires_baseline_identity());
            // Emitted and linkable: the object is not the incumbent's.
            let plain = new(shape, &mut knobs(&[("epi", epi)])).unwrap();
            assert_ne!(
                w.candidate().unwrap().bytes,
                plain.candidate().unwrap().bytes
            );
        }
        // Two arms with the same fold are an exact comparison again.
        let w = new(
            shape,
            &mut knobs(&[("fold-op", "muladd"), ("base-fold-op", "muladd")]),
        )
        .unwrap();
        assert!(w.numeric_contract().is_null());
        // The emulator reproduces the candidate's own CPU fold bit for bit.
        for (epi, ko) in [("set", "muladd"), ("add", "muladd"), ("set", "fma")] {
            for wait in ["after", "mid", "before"] {
                let w = new(
                    shape,
                    &mut knobs(&[("epi", epi), ("fold-op", ko), ("wait", wait)]),
                )
                .unwrap();
                let v = w.emu();
                assert_eq!(v["ran"], true, "{epi} {ko} {wait}: {v}");
                assert_eq!(v["bit_exact"], true, "{epi} {ko} {wait}: {v}");
            }
        }
    }

    fn sym(w: &dyn Workload) -> String {
        w.baseline().unwrap().launch_symbol
    }

    #[test]
    fn base_shipped_v64_selects_the_shipped_small_n_export() {
        for (epi, want) in [
            ("add", "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s_v64"),
            ("set", "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1s_v64"),
            ("silu", "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s_v64"),
        ] {
            let mut k = knobs(&[("alayout", "slab"), ("epi", epi), ("base-shipped", "v64")]);
            let w = new("M=5120,N=64,K=17408", &mut k).unwrap();
            assert!(k.0.is_empty(), "base-shipped is consumed");
            assert_eq!(sym(&*w), want, "{epi}");
            let b = w.baseline().unwrap();
            assert_eq!(
                b.source["artifact"],
                "kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s.hxaco"
            );
        }
        // Default and explicit `b1s`: the shipped `_b1s` export.
        for kv in [vec![], vec![("base-shipped", "b1s")]] {
            let mut kv = kv;
            kv.extend([("alayout", "slab"), ("epi", "add")]);
            let w = new("M=5120,N=64,K=17408", &mut knobs(&kv)).unwrap();
            assert_eq!(sym(&*w), "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s");
        }
        // The record names the real schedule of the shipped `_v64` export.
        let mut k = knobs(&[("alayout", "slab"), ("epi", "add"), ("base-shipped", "v64")]);
        let w = new("M=5120,N=64,K=17408", &mut k).unwrap();
        assert_eq!(
            w.baseline().unwrap().desc["knobs"]["fetch"],
            "pairw",
            "{:?}",
            w.baseline().unwrap().desc
        );
        assert_eq!(w.baseline().unwrap().desc["knobs"]["idle"], "skiphi");
        assert_eq!(w.baseline().unwrap().desc["knobs"]["wait"], "window");
    }

    #[test]
    fn base_shipped_refusals() {
        let bad = |kv: &[(&str, &str)]| new("M=5120,N=64,K=17408", &mut knobs(kv)).err();
        // Token layout has no `_v64` exports.
        assert!(bad(&[("epi", "add"), ("base-shipped", "v64")])
            .unwrap()
            .contains("dense slab"));
        assert!(
            bad(&[("alayout", "slab"), ("epi", "add"), ("base-shipped", "v32")])
                .unwrap()
                .contains("expected b1s|v64")
        );
        // A shipped export excludes an emitted baseline.
        assert!(bad(&[
            ("alayout", "slab"),
            ("epi", "add"),
            ("base-shipped", "v64"),
            ("base-fold", "k128"),
        ])
        .unwrap()
        .contains("excludes"));
        assert!(bad(&[
            ("alayout", "slab"),
            ("epi", "silu-bf16"),
            ("base-shipped", "v64")
        ])
        .is_some());
    }

    #[test]
    fn flush_pool_sizes_the_verify_shapes_past_the_last_level_cache() {
        let pool = |shape: &str, epi: &str| {
            let mut w = new(
                shape,
                &mut knobs(&[("alayout", "slab"), ("epi", epi), ("base-shipped", "v64")]),
            )
            .unwrap();
            assert_eq!(w.pool_sets(), 1, "hot: no pool");
            w.set_cold(crate::cold::ColdMode::Flush).unwrap();
            (w.pool_sets(), w.shape_json())
        };
        // down (47.3 MB), gate_up (gate + up = 94.7 MB), out-proj (16.7 MB).
        for (shape, epi, set_bytes, sets) in [
            ("M=5120,N=64,K=17408", "add", 5120 * 68 * 136, 5),
            ("M=17408,N=64,K=5120", "silu", 2 * 17408 * 20 * 136, 3),
            ("M=5120,N=64,K=6144", "add", 5120 * 24 * 136, 13),
        ] {
            let (n, s) = pool(shape, epi);
            assert_eq!(n, sets, "{shape}");
            assert_eq!(s["flush"], true);
            assert_eq!(s["timing_pool"]["sets"], sets);
            assert_eq!(s["timing_pool"]["bytes_per_set"], set_bytes);
            assert_eq!(s["timing_pool"]["total_bytes"], sets * set_bytes);
            assert!(sets * set_bytes > crate::cold::POOL_BYTES);
            assert_eq!(sets % 2, 1);
        }
    }

    #[test]
    fn hot_shape_keeps_its_bytes_and_cold_adds_only_flush_keys() {
        let mut w = new("M=5120,N=64,K=17408", &mut knobs(&[("epi", "add")])).unwrap();
        let hot = w.shape_json();
        let mut keys: Vec<_> = hot.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            ["K", "M", "N", "launch_rows"],
            "a pm-census noise group keys on this"
        );
        w.set_cold(crate::cold::ColdMode::Flush).unwrap();
        let cold = w.shape_json();
        for k in ["K", "M", "N", "launch_rows"] {
            assert_eq!(cold[k], hot[k]);
        }
        assert!(cold["flush"] == true && cold["timing_pool"].is_object());
        // Back to hot: the keys go away again.
        w.set_cold(crate::cold::ColdMode::Hot).unwrap();
        assert_eq!(w.shape_json(), hot);
    }

    #[test]
    fn flush_refuses_a_sparse_arm() {
        let mut w = new("M=128,N=64,K=256", &mut knobs(&[("mma", "s24x64")])).unwrap();
        assert!(w
            .set_cold(crate::cold::ColdMode::Flush)
            .unwrap_err()
            .contains("2:4 sparse"));
        let mut w = new("M=128,N=64,K=256", &mut knobs(&[("base-mma", "s24x32")])).unwrap();
        assert!(w.set_cold(crate::cold::ColdMode::Flush).is_err());
        assert!(new("M=128,N=64,K=256", &mut knobs(&[]))
            .unwrap()
            .set_cold(crate::cold::ColdMode::Flush)
            .is_ok());
    }

    #[test]
    fn pool_sets_are_distinct_seeded_weight_sets() {
        use std::collections::BTreeSet;
        assert_eq!(
            pool_seed(GATE_SEED, 0),
            GATE_SEED,
            "set 0 is the correctness set"
        );
        let gate: BTreeSet<u64> = (0..crate::cold::MAX_SETS)
            .map(|s| pool_seed(GATE_SEED, s))
            .collect();
        let up: BTreeSet<u64> = (0..crate::cold::MAX_SETS)
            .map(|s| pool_seed(UP_SEED, s))
            .collect();
        assert_eq!(gate.len(), crate::cold::MAX_SETS);
        assert_eq!(up.len(), crate::cold::MAX_SETS);
        assert!(gate.is_disjoint(&up));
        let w = |s: usize| synth_weights(4, 256, pool_seed(GATE_SEED, s));
        assert_eq!(w(1), w(1), "deterministic");
        assert_ne!(w(0), w(1));
        assert_ne!(w(1), w(2));
        // Set 0 of the pool is `Host::synth`'s weights.
        assert_eq!(w(0), Host::synth(4, 1, 256, false).w);
    }

    #[test]
    fn pool_only_and_l2_flush_modes_pool_like_flush_but_record_their_own_keys() {
        use crate::cold::ColdMode::*;
        let shape = |m| {
            let mut w = new("M=5120,N=64,K=17408", &mut knobs(&[("epi", "add")])).unwrap();
            w.set_cold(m).unwrap();
            (w.pool_sets(), w.shape_json())
        };
        let (np, p) = shape(Pool);
        let (nl, l) = shape(FlushL2);
        let (nf, f) = shape(Flush);
        assert_eq!((np, nl, nf), (5, 5, 5));
        assert_eq!(
            (p["flush"].clone(), l["flush"].clone(), f["flush"].clone()),
            (json!(false), json!(false), json!(true))
        );
        assert!(
            p.get("flush_l2").is_none() && l["flush_l2"] == true && f.get("flush_l2").is_none()
        );
        assert_eq!(p["timing_pool"], f["timing_pool"]);
        // Sparse arms have no pool in any mode.
        for m in [Pool, FlushL2, Flush] {
            let mut w = new("M=128,N=64,K=256", &mut knobs(&[("mma", "s24x64")])).unwrap();
            let e = w.set_cold(m).unwrap_err();
            assert!(e.starts_with(m.flag()) && e.contains("2:4 sparse"), "{e}");
        }
    }

    #[test]
    fn null_object_keeps_the_descriptor_and_drops_the_body() {
        let code = |o: &Object| {
            let p = crate::lift(&o.bytes).unwrap();
            let r = crate::resources(&p, std::slice::from_ref(&o.launch_symbol)).unwrap();
            (
                r[0]["code_bytes"].as_u64().unwrap(),
                r[0]["vgprs"].as_u64().unwrap(),
                r[0]["sgprs"].as_u64().unwrap(),
                r[0]["kernarg_bytes"].as_u64().unwrap(),
                r[0]["max_flat_workgroup_size"].as_u64().unwrap(),
            )
        };
        for (epi, shape) in [
            ("add", "M=5120,N=64,K=17408"),
            ("silu", "M=17408,N=64,K=5120"),
        ] {
            let mut k = knobs(&[("alayout", "slab"), ("epi", epi), ("base-shipped", "v64")]);
            let w = new(shape, &mut k).unwrap();
            let (base, null) = (w.baseline().unwrap(), w.null_object().unwrap().unwrap());
            assert_eq!(null.launch_symbol, base.launch_symbol);
            let (b, n) = (code(&base), code(&null));
            assert!(n.0 <= 8 && b.0 > 1000, "{epi}: code {} -> {}", b.0, n.0);
            assert_eq!(
                (n.1, n.2, n.3, n.4),
                (b.1, b.2, b.3, b.4),
                "{epi}: VGPRs, SGPRs, kernarg, workgroup size stay"
            );
        }
    }
}
