// SPDX-License-Identifier: Apache-2.0
//! gfx1100 W3 (MQ3-I4 planar, qt131/qt132) K1 MoE decode-GEMV **byte oracle** + per-call microbench.
//! Never participates in runtime dispatch.
//!
//! Loads the code object `mq_i4_moe_gemv_mq3only_pm_gfx1100` (`kernels/…hxaco`) and launches the narrow entries
//! `mq_i4_moe_gemv_mq3i4_{r256_gate_up,r128_down}_pm_gfx1100` (grid `(M, U, 1)`, block 32) and the descriptor
//! entries `…_t8` (grid `(ceil(M/8), 80, ceil(C/8))`, block 256) with the frozen 96-byte kernarg ABI
//! (7 pointers `table ids offs slots x y0 y1`, then `m k u c xdiv xsrc xrs ys n_experts` as u32 and a zero u32).
//! Every output word is compared bit for bit (signed zeros and NaN payloads included) with TWO CPU oracles:
//!
//! * `mq_i4_reference::decode_moe_slots` (the engine's F4-tree reference, W3 profiles `MQ3_R256` / `MQ3_R128`,
//!   the one the gfx1201/gfx1151 W3 K1 receipts use), and
//! * an independent twin built on the CPU planar decoder `hipfire_cpu::quant::decode_planar_row`
//!   (`CpuQuant::Mq3I4G128R256` / `R128`, `w = f32(d) * (2c - 7)`) plus the frozen F4 association (lane `l` owns
//!   `g*128 + 4l..+3`, `RN(w0*x0)` then `fma`, chains `acc[g&3]` / one chain, fold `(a0+a1)+(a2+a3)`, snapshot
//!   shuffle-down 16,8,4,2,1). The two oracles must agree with each other first (CPU-only gate), then with the GPU.
//!
//! Cases: synthetic experts (extreme/zero/subnormal/negative f16 scales, all-0/all-7 planes, padded strides,
//! identical/overlapping/unique routes) and REAL experts read from a W3 artifact (`--artifact`, e.g. the
//! `qwen3.8-flash-next-base.mq3-xts` split base: qt131/qt132 experts, SHA-bound per tensor) in VRAM, true
//! mapped host memory and a split placement; plus negative controls (output flip, unwritten word, wrong level law,
//! weight corruption in placement memory, invalid expert ids) that MUST be detected.
//!
//! `--cpu-only` runs the oracle self-agreement (decode_moe_slots vs the CPU decoder twin) on synthetic and artifact
//! experts without any GPU. `--timing` measures per-launch device time of the W3 K1 against the E8P and MQ4 gfx1100
//! K1 images (`--mq4-codeobject`, `--e8p-codeobject`) at C = 1, 4, 8 in VRAM and host-mapped memory.
//!
//!   cargo build -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_gfx1100_w3_g1
//!   mq_i4_moe_gemv_gfx1100_w3_g1 --codeobject kernels/mq_i4_moe_gemv_mq3only_pm_gfx1100.hxaco \
//!     [--artifact ~/.hipfire/models/qwen3.8-flash-next-base.mq3-xts --layers 0,23,47 --real-experts 24] \
//!     [--device N] [--expect-uuid GPU-… --expect-bdf 0000:… --expect-arch gfx1100] [--cpu-only]
//! Prints one machine JSON line on stdout; exit code 1 on any mismatch, control failure or error.
use hip_bridge::{DeviceBuffer, Function, HipRuntime, HIP_HOST_MALLOC_MAPPED};
use hipfire_cpu::quant::{decode_planar_row, CpuQuant};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::mq_i4::MqI4Profile;
use hipfire_runtime::mq_i4_reference::{self, DecodeKind, ExpertTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::rc::Rc;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const SCHEMA: &str = "mq-i4-moe-gemv-gfx1100-w3-g1-v1";
const DEFAULT_UUID: &str = "GPU-43390a851e296ee5";
const DEFAULT_BDF: &str = "0000:67:00.0";
const DEFAULT_ARCH: &str = "gfx1100";
const KERNARG_BYTES: usize = 96;
const N_EXPERTS: usize = 512;
const BAD_IDS: [u32; 4] = [u32::MAX, N_EXPERTS as u32, i32::MAX as u32, i32::MIN as u32];
const GATE_M: usize = 1280;
const GATE_K: usize = 2560;
const DOWN_M: usize = 2560;
const DOWN_K: usize = 640;
const SENTINEL_BITS: u32 = 0xa5a5_a5a5;
const PAD_BITS: u32 = 0x7fc0_0000;
const TILE_WORDS: usize = 482;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    GateUp,
    Down,
}
impl Kind {
    const ALL: [Kind; 2] = [Kind::GateUp, Kind::Down];
    fn m(self) -> usize {
        if self == Kind::GateUp {
            GATE_M
        } else {
            DOWN_M
        }
    }
    fn k(self) -> usize {
        if self == Kind::GateUp {
            GATE_K
        } else {
            DOWN_K
        }
    }
    fn out_rows(self) -> usize {
        if self == Kind::GateUp {
            GATE_M / 2
        } else {
            DOWN_M
        }
    }
    fn x_row_div(self) -> usize {
        if self == Kind::GateUp {
            10
        } else {
            1
        }
    }
    fn tag(self) -> &'static str {
        if self == Kind::GateUp {
            "r256_gate_up"
        } else {
            "r128_down"
        }
    }
    fn profile(self) -> MqI4Profile {
        if self == Kind::GateUp {
            MqI4Profile::MQ3_R256
        } else {
            MqI4Profile::MQ3_R128
        }
    }
    fn decode_kind(self) -> DecodeKind {
        if self == Kind::GateUp {
            DecodeKind::GateUp
        } else {
            DecodeKind::Down
        }
    }
    fn cpu_quant(self) -> CpuQuant {
        if self == Kind::GateUp {
            CpuQuant::Mq3I4G128R256
        } else {
            CpuQuant::Mq3I4G128R128
        }
    }
    /// Bytes of one W3 tensor: codes plane `M*K*3/8` then the f16 scale plane `M*K/128*2`.
    fn w3_bytes(self) -> usize {
        self.m() * self.k() * 3 / 8 + self.m() * (self.k() / 128) * 2
    }
}
fn w3_symbol(kind: Kind, tiled: bool) -> String {
    format!(
        "mq_i4_moe_gemv_mq3i4_{}_pm_gfx1100{}",
        kind.tag(),
        if tiled { "_t8" } else { "" }
    )
}

// ---------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}
fn name_seed(name: &str) -> u64 {
    let d = Sha256::digest(name.as_bytes());
    u64::from_le_bytes(d[..8].try_into().unwrap())
}
fn sha(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}
fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64_bytes(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        0.0
    } else {
        s[s.len() / 2]
    }
}
fn bits_hex(v: u32) -> String {
    format!("{v:#010x}")
}

// ---------------------------------------------------------------------------------------
// W3 expert tensors
// ---------------------------------------------------------------------------------------

/// Valid-domain f16 scale bits (the production writer's import validation: finite, non-negative; a `+0` scale only
/// over the midpoint code 4): the production band, the extremes, `+0` and positive subnormals. Negative scales and
/// `-0` are outside the engine reference's domain; the interpreter byte tests cover them against the model.
fn scale_bits(rng: &mut Rng, i: usize) -> u16 {
    match i % 13 {
        0 => 0x0400,
        1 => 0x7bff,
        2 => 0x0000,
        3 => 0x0001,
        4 => 0x03ff,
        5 => 0x0001 + rng.below(0x3ff) as u16,
        _ => {
            let e = 8 + rng.below(13) as u16;
            (e << 10) | (rng.next() as u16 & 0x3ff)
        }
    }
}

/// One synthetic W3 tensor `[m, k]` in the artifact wire (codes plane then f16 scale plane).
fn gen_w3(rng: &mut Rng, m: usize, k: usize, special: bool) -> Vec<u8> {
    let (groups, row_bytes) = (k / 128, k * 3 / 8);
    let mut out = vec![0u8; m * row_bytes + 2 * m * groups];
    for r in 0..m {
        for g in 0..groups {
            let base = r * row_bytes + g * 48;
            let scale = scale_bits(rng, r * groups + g);
            // Rows 0 / 1 of group 0 are the all-0 / all-7 planes; a +0 scale holds the midpoint code 4 everywhere.
            let fixed = if scale == 0 {
                Some(4u32)
            } else {
                (special && g == 0 && r < 2).then_some(if r == 0 { 0u32 } else { 7 })
            };
            for j in 0..128 {
                let code = fixed.unwrap_or_else(|| rng.below(8) as u32);
                for b in 0..3 {
                    if (code >> b) & 1 == 1 {
                        out[base + (3 * j + b) / 8] |= 1 << ((3 * j + b) % 8);
                    }
                }
            }
            let off = m * row_bytes + 2 * (r * groups + g);
            out[off..off + 2].copy_from_slice(&scale.to_le_bytes());
        }
    }
    out
}

struct Expert {
    gate: Vec<u8>,
    down: Vec<u8>,
    gate_sha: [u8; 32],
    down_sha: [u8; 32],
}
impl Expert {
    fn new(gate: Vec<u8>, down: Vec<u8>) -> Self {
        Expert {
            gate_sha: sha(&gate),
            down_sha: sha(&down),
            gate,
            down,
        }
    }
    fn proj(&self, kind: Kind) -> &[u8] {
        if kind == Kind::GateUp {
            &self.gate
        } else {
            &self.down
        }
    }
}
fn gen_expert(id: u32, salt: u64) -> Expert {
    let mut rng = Rng::new(0x3_0000_0000 ^ u64::from(id) ^ (salt << 20));
    let special = id % 5 == 0;
    Expert::new(
        gen_w3(&mut rng, GATE_M, GATE_K, special),
        gen_w3(&mut rng, DOWN_M, DOWN_K, special),
    )
}

// ---------------------------------------------------------------------------------------
// Real artifact experts
// ---------------------------------------------------------------------------------------

fn expert_entry(layer: usize, expert: usize, proj: &str) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}")
}

struct Real {
    experts: BTreeMap<u32, Rc<Expert>>,
    meta: Value,
}

/// `want` W3 (qt131/qt132) experts of `layer`, spread over the layer's W3 population.
fn load_real(hfq: &HfqFile, path: &str, layer: usize, want: usize) -> Result<Real> {
    let mut avail = Vec::new();
    let (mut mq4, mut other) = (0usize, 0usize);
    for e in 0..N_EXPERTS {
        let (gn, dn) = (
            expert_entry(layer, e, "gate_up_proj"),
            expert_entry(layer, e, "down_proj"),
        );
        let (Some(g), Some(d)) = (hfq.find_tensor_info(&gn), hfq.find_tensor_info(&dn)) else {
            continue;
        };
        match (g.quant_type, d.quant_type) {
            (131, 132) => {
                if g.shape != [GATE_M as u32, GATE_K as u32]
                    || d.shape != [DOWN_M as u32, DOWN_K as u32]
                    || g.data_size != Kind::GateUp.w3_bytes()
                    || d.data_size != Kind::Down.w3_bytes()
                {
                    return Err(format!(
                        "layer {layer} expert {e}: unexpected W3 shape/size {:?}/{} {:?}/{}",
                        g.shape, g.data_size, d.shape, d.data_size
                    ));
                }
                avail.push(e as u32);
            }
            (44, 53) => mq4 += 1,
            _ => other += 1,
        }
    }
    if avail.len() < 10 {
        return Err(format!(
            "layer {layer} of {path}: only {} W3 experts (mq4={mq4}, other={other}); a top-10 lane needs 10",
            avail.len()
        ));
    }
    let take = want.max(10).min(avail.len());
    let sel: Vec<u32> = (0..take).map(|i| avail[i * avail.len() / take]).collect();
    let mut experts = BTreeMap::new();
    let mut recs = Vec::new();
    for &id in &sel {
        let (gn, dn) = (
            expert_entry(layer, id as usize, "gate_up_proj"),
            expert_entry(layer, id as usize, "down_proj"),
        );
        let (gi, gate) = hfq
            .tensor_data_vec(&gn)
            .ok_or_else(|| format!("short read of {gn}"))?;
        let (goff, gq) = (gi.data_offset, gi.quant_type);
        let (di, down) = hfq
            .tensor_data_vec(&dn)
            .ok_or_else(|| format!("short read of {dn}"))?;
        let (doff, dq) = (di.data_offset, di.quant_type);
        let e = Expert::new(gate, down);
        recs.push(json!({"expert": id, "gate_offset": goff, "gate_quant_type": gq, "gate_bytes": e.gate.len(),
            "gate_sha256": hex(&e.gate_sha), "down_offset": doff, "down_quant_type": dq, "down_bytes": e.down.len(),
            "down_sha256": hex(&e.down_sha)}));
        experts.insert(id, Rc::new(e));
    }
    let meta = json!({"path": path, "layer": layer, "layer_w3_experts": avail.len(), "layer_mq4_experts": mq4,
        "layer_other_experts": other, "selected": sel.len(), "experts": recs});
    Ok(Real { experts, meta })
}

// ---------------------------------------------------------------------------------------
// Cases and CPU oracles
// ---------------------------------------------------------------------------------------

struct Case {
    name: String,
    origin: &'static str,
    c: usize,
    route: Vec<u32>,
    experts: BTreeMap<u32, Rc<Expert>>,
    pad: bool,
    x_style: usize,
}

fn ids_of(route: &[u32]) -> Vec<u32> {
    let mut ids = Vec::new();
    for &e in route {
        if !ids.contains(&e) {
            ids.push(e);
        }
    }
    ids
}

struct Bucket {
    ids: Vec<u32>,
    offs: Vec<u32>,
    slots: Vec<u32>,
}
/// First-occurrence experts over slots `range`, ascending absolute slots.
fn bucket(route: &[u32], range: std::ops::Range<usize>) -> Bucket {
    let ids = ids_of(&route[range.clone()]);
    let (mut offs, mut slots) = (Vec::new(), Vec::new());
    for &id in &ids {
        offs.push(slots.len() as u32);
        slots.extend(range.clone().filter(|&s| route[s] == id).map(|s| s as u32));
    }
    offs.push(slots.len() as u32);
    Bucket { ids, offs, slots }
}

fn x_rows(case: &Case, kind: Kind) -> (Vec<f32>, usize, usize, usize) {
    let rows = if kind == Kind::GateUp {
        case.c
    } else {
        10 * case.c
    };
    let (k, out) = (kind.k(), kind.out_rows());
    let (xrs, ys) = if case.pad {
        (k + 32, out + 16)
    } else {
        (k, out)
    };
    let mut rng = Rng::new(name_seed(&case.name) ^ (kind.m() as u64));
    let mut x = Vec::with_capacity(rows * xrs);
    for _ in 0..rows {
        for i in 0..xrs {
            x.push(if i >= k {
                f32::from_bits(PAD_BITS)
            } else {
                let (s, e, m) = if case.x_style == 0 {
                    // wide exponent band, the stress style
                    (
                        (rng.next() & 1) as u32,
                        118 + rng.below(19) as u32,
                        (rng.next() & 0x7f_ffff) as u32,
                    )
                } else {
                    // activation-like magnitudes, |x| in [2^-7, 2)
                    (
                        (rng.next() & 1) as u32,
                        120 + rng.below(8) as u32,
                        (rng.next() & 0x7f_ffff) as u32,
                    )
                };
                f32::from_bits((s << 31) | (e << 23) | m)
            });
        }
    }
    (x, xrs, ys, rows)
}

struct Expected {
    y0: Vec<f32>,
    y1: Option<Vec<f32>>,
}

/// Engine F4 reference on one launch's bucket (`decode_moe_slots` with the W3 profiles).
#[allow(clippy::too_many_arguments)]
fn reference_on(
    experts: &BTreeMap<u32, Rc<Expert>>,
    kind: Kind,
    b: &Bucket,
    c: usize,
    x: &[f32],
    xrs: usize,
    ys: usize,
    total_slots: usize,
) -> Expected {
    let mut table: Vec<Option<ExpertTensor<'_>>> = (0..N_EXPERTS).map(|_| None).collect();
    for (&id, e) in experts {
        table[id as usize] = Some(ExpertTensor {
            profile: kind.profile(),
            bytes: e.proj(kind),
        });
    }
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let mut y0 = vec![sentinel; total_slots * ys];
    let mut y1 = (kind == Kind::GateUp).then(|| vec![sentinel; total_slots * ys]);
    mq_i4_reference::decode_moe_slots(
        kind.decode_kind(),
        kind.m(),
        kind.k(),
        &table,
        &b.ids,
        &b.offs,
        &b.slots,
        c,
        x,
        xrs,
        ys,
        &mut y0,
        y1.as_deref_mut(),
    );
    Expected { y0, y1 }
}

/// The frozen F4 tree of one row from decoded weights.
fn tree_row(gate: bool, k: usize, x: &[f32], w: &[f32]) -> f32 {
    let mut lane = [0f32; 32];
    for (l, out) in lane.iter_mut().enumerate() {
        let mut acc = [0f32; 4];
        for g in 0..k / 128 {
            let base = g * 128 + 4 * l;
            let mut p = w[base] * x[base];
            for j in 1..4 {
                p = w[base + j].mul_add(x[base + j], p);
            }
            acc[if gate { g & 3 } else { 0 }] += p;
        }
        *out = if gate {
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        } else {
            acc[0]
        };
    }
    for off in [16usize, 8, 4, 2, 1] {
        let snap = lane;
        for l in 0..32 {
            if l + off < 32 {
                lane[l] = snap[l] + snap[l + off];
            }
        }
    }
    lane[0]
}

/// One row's f32 weights through the CPU planar decoder (`hipfire_cpu::quant::decode_planar_row`).
fn cpu_row_weights(kind: Kind, tensor: &[u8], row: usize, out: &mut [f32]) {
    let (m, k) = (kind.m(), kind.k());
    let (row_bytes, groups) = (k * 3 / 8, k / 128);
    let codes = &tensor[row * row_bytes..(row + 1) * row_bytes];
    let scale_at = m * row_bytes + 2 * row * groups;
    decode_planar_row(
        kind.cpu_quant(),
        codes,
        &tensor[scale_at..scale_at + 2 * groups],
        k,
        out,
    );
}

/// Cross-check of `decode_moe_slots` against the CPU-decoder twin: every row of the first two slots plus every
/// `stride`th row of the others, bit for bit. `law` selects the level law: `0` = the real `2c-7` through the CPU
/// decoder, else a deliberately wrong law (negative control): weights scaled by `(1 + 1/law)`.
#[allow(clippy::too_many_arguments)]
fn twin_check(
    case: &Case,
    kind: Kind,
    b: &Bucket,
    x: &[f32],
    xrs: usize,
    ys: usize,
    got0: &[f32],
    got1: Option<&[f32]>,
    law: u32,
) -> (u64, usize) {
    let (k, out) = (kind.k(), kind.out_rows());
    let mut w = vec![0f32; k];
    let (mut bad, mut checked) = (0u64, 0usize);
    for (u, &id) in b.ids.iter().enumerate() {
        let e = &case.experts[&id];
        for (si, &slot) in b.slots[b.offs[u] as usize..b.offs[u + 1] as usize]
            .iter()
            .enumerate()
        {
            let slot = slot as usize;
            let xr = slot / kind.x_row_div();
            let xs = &x[xr * xrs..xr * xrs + k];
            let stride = if u == 0 && si < 2 { 1 } else { 97 };
            for row in (0..kind.m()).step_by(stride) {
                cpu_row_weights(kind, e.proj(kind), row, &mut w);
                if law != 0 {
                    for v in &mut w {
                        *v *= 1.0 + 1.0 / law as f32;
                    }
                }
                let want = tree_row(kind == Kind::GateUp, k, xs, &w);
                let (buf, r) = if row < out {
                    (got0, row)
                } else {
                    (got1.expect("gate has y1"), row - out)
                };
                checked += 1;
                if buf[slot * ys + r].to_bits() != want.to_bits() {
                    bad += 1;
                }
            }
        }
    }
    (bad, checked)
}

// ---------------------------------------------------------------------------------------
// Synthetic and real case construction
// ---------------------------------------------------------------------------------------

fn pick_ids(rng: &mut Rng, n: usize) -> Vec<u32> {
    let mut all: Vec<u32> = (0..N_EXPERTS as u32).collect();
    rng.shuffle(&mut all);
    let mut v: Vec<u32> = all.into_iter().take(n).collect();
    // Edge ids: the first and the last table entries are always present.
    v[0] = 0;
    if n > 1 {
        v[1] = N_EXPERTS as u32 - 1;
    }
    v
}

fn experts_for(ids: &[u32], salt: u64) -> BTreeMap<u32, Rc<Expert>> {
    ids.iter()
        .map(|&id| (id, Rc::new(gen_expert(id, salt))))
        .collect()
}

/// `c` lanes of ten distinct experts each, drawn from `pool`.
fn lane_routes(rng: &mut Rng, pool: &[u32], c: usize) -> Vec<u32> {
    let mut route = Vec::with_capacity(10 * c);
    for _ in 0..c {
        let mut p = pool.to_vec();
        rng.shuffle(&mut p);
        route.extend_from_slice(&p[..10]);
    }
    route
}

fn synthetic(name: &str) -> Case {
    let mut rng = Rng::new(name_seed(name));
    let (c, pool, pad, style) = match name {
        "c1_u10" => (1, 10, false, 0),
        "c1_padded" => (1, 10, true, 1),
        "c2_overlap14" => (2, 14, false, 0),
        "c4_overlap20_padded" => (4, 20, true, 0),
        "c8_union33" => (8, 33, false, 0),
        "c8_unique80" => (8, 80, false, 1),
        "c8_identical" => (8, 10, false, 0),
        other => panic!("unknown synthetic case {other}"),
    };
    let ids = pick_ids(&mut rng, pool);
    let experts = experts_for(&ids, 1);
    let route = if name == "c8_identical" {
        let mut lane = ids.clone();
        rng.shuffle(&mut lane);
        (0..c).flat_map(|_| lane.clone()).collect()
    } else if name == "c8_unique80" {
        let mut r = ids.clone();
        rng.shuffle(&mut r);
        r
    } else {
        lane_routes(&mut rng, &ids, c)
    };
    Case {
        name: name.into(),
        origin: "synthetic",
        c,
        route,
        experts,
        pad,
        x_style: style,
    }
}

const SYNTH_NARROW: [&str; 7] = [
    "c1_u10",
    "c1_padded",
    "c2_overlap14",
    "c4_overlap20_padded",
    "c8_union33",
    "c8_unique80",
    "c8_identical",
];
const SYNTH_HOST: [&str; 3] = ["c1_u10", "c4_overlap20_padded", "c8_union33"];
const SYNTH_SPLIT: [&str; 2] = ["c2_overlap14", "c8_unique80"];
const SYNTH_TILED: [(&str, usize); 4] = [
    ("c1_u10", 1),
    ("c8_union33", 8),
    ("c8_union33", 9),
    ("c8_unique80", 8),
];

fn tiled_case(base: &str, c: usize) -> Case {
    let mut case = synthetic(base);
    if case.c != c {
        // More lanes than the base case: re-route from the same expert set (every expert still present).
        let mut rng = Rng::new(name_seed(&format!("{base}@{c}")));
        let pool: Vec<u32> = case.experts.keys().copied().collect();
        case.route = lane_routes(&mut rng, &pool, c);
        case.c = c;
    }
    case.name = format!("{base}_t8_c{c}");
    case
}

fn real_cases(real: &Real, layer: usize) -> Vec<Case> {
    let ids: Vec<u32> = real.experts.keys().copied().collect();
    let mut rng = Rng::new(0x7ea1_0000 + layer as u64);
    let mk = |name: String, c: usize, route: Vec<u32>, style: usize| {
        let used = ids_of(&route);
        let experts = used
            .iter()
            .map(|id| (*id, real.experts[id].clone()))
            .collect();
        Case {
            name,
            origin: "artifact",
            c,
            route,
            experts,
            pad: false,
            x_style: style,
        }
    };
    let mut v = Vec::new();
    v.push(mk(
        format!("real_L{layer}_c1"),
        1,
        lane_routes(&mut rng, &ids, 1),
        1,
    ));
    v.push(mk(
        format!("real_L{layer}_c8_union"),
        8,
        lane_routes(&mut rng, &ids, 8),
        1,
    ));
    v.push(mk(
        format!("real_L{layer}_c9_tiled"),
        9,
        lane_routes(&mut rng, &ids, 9),
        1,
    ));
    v
}

// ---------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Placement {
    Vram,
    Host,
    Split,
}
impl Placement {
    fn name(self) -> &'static str {
        match self {
            Placement::Vram => "vram",
            Placement::Host => "host_mapped",
            Placement::Split => "split_vram_host",
        }
    }
    fn host_for(self, i: usize) -> bool {
        match self {
            Placement::Vram => false,
            Placement::Host => true,
            Placement::Split => i % 2 == 1,
        }
    }
}

#[derive(Clone, Copy)]
struct Buf {
    idx: usize,
    ptr: u64,
}
struct Dev<'a> {
    hip: &'a HipRuntime,
    bufs: Vec<DeviceBuffer>,
    hosts: Vec<*mut c_void>,
}
impl<'a> Dev<'a> {
    fn new(hip: &'a HipRuntime) -> Self {
        Dev {
            hip,
            bufs: Vec::new(),
            hosts: Vec::new(),
        }
    }
    fn alloc(&mut self, n: usize) -> Result<Buf> {
        let b = self.hip.malloc(n.max(4)).map_err(err)?;
        let ptr = b.as_ptr() as u64;
        self.bufs.push(b);
        Ok(Buf {
            idx: self.bufs.len() - 1,
            ptr,
        })
    }
    fn upload(&mut self, bytes: &[u8]) -> Result<Buf> {
        let b = self.alloc(bytes.len())?;
        self.hip
            .memcpy_htod(&self.bufs[b.idx], bytes)
            .map_err(err)?;
        Ok(b)
    }
    fn download(&self, b: Buf, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; n];
        self.hip
            .memcpy_dtoh(&mut out, &self.bufs[b.idx])
            .map_err(err)?;
        Ok(out)
    }
    /// True `hipHostMalloc(hipHostMallocMapped)` + `hipHostGetDevicePointer`: `(host ptr, device-visible ptr)`.
    fn alloc_host(&mut self, n: usize) -> Result<(*mut u8, u64)> {
        let host = self
            .hip
            .host_malloc(n.max(4096), HIP_HOST_MALLOC_MAPPED)
            .map_err(err)?;
        self.hosts.push(host);
        let dev = self.hip.host_get_device_pointer(host, 0).map_err(err)?;
        Ok((host as *mut u8, dev as u64))
    }
    fn release(self) {
        let hip = self.hip;
        for b in self.bufs {
            let _ = hip.free(b);
        }
        for h in self.hosts {
            let _ = hip.host_free(h);
        }
    }
}

enum Loc {
    Vram(usize),
    Host(*mut u8),
}
struct Slab {
    gate_ptr: u64,
    down_ptr: u64,
    len: usize,
    loc: Loc,
}
struct Slabs {
    by_id: BTreeMap<u32, Slab>,
    host: usize,
    vram: usize,
}

/// One slab per expert (gate tensor then down tensor) in shuffled order; VRAM slabs sit between random fillers.
fn upload_experts(
    dev: &mut Dev,
    experts: &BTreeMap<u32, Rc<Expert>>,
    seed: u64,
    placement: Placement,
) -> Result<Slabs> {
    let mut rng = Rng::new(seed ^ 0xa110c);
    let mut order: Vec<u32> = experts.keys().copied().collect();
    rng.shuffle(&mut order);
    let mut slabs = Slabs {
        by_id: BTreeMap::new(),
        host: 0,
        vram: 0,
    };
    for (i, id) in order.into_iter().enumerate() {
        let e = &experts[&id];
        let len = e.gate.len() + e.down.len();
        let mut slab = Vec::with_capacity(len);
        slab.extend_from_slice(&e.gate);
        slab.extend_from_slice(&e.down);
        if placement.host_for(i) {
            let (hp, dp) = dev.alloc_host(len)?;
            // SAFETY: `hp` is a live mapped host allocation of at least `len` bytes.
            unsafe { std::ptr::copy_nonoverlapping(slab.as_ptr(), hp, len) };
            slabs.host += 1;
            slabs.by_id.insert(
                id,
                Slab {
                    gate_ptr: dp,
                    down_ptr: dp + e.gate.len() as u64,
                    len,
                    loc: Loc::Host(hp),
                },
            );
        } else {
            if rng.below(3) != 0 {
                dev.alloc(256 * (1 + rng.below(300)))?;
            }
            let b = dev.upload(&slab)?;
            slabs.vram += 1;
            slabs.by_id.insert(
                id,
                Slab {
                    gate_ptr: b.ptr,
                    down_ptr: b.ptr + e.gate.len() as u64,
                    len,
                    loc: Loc::Vram(b.idx),
                },
            );
        }
    }
    Ok(slabs)
}

fn overwrite_slab(dev: &Dev, slabs: &Slabs, id: u32, e: &Expert) -> Result<()> {
    let s = slabs
        .by_id
        .get(&id)
        .ok_or_else(|| format!("no slab for expert {id}"))?;
    let mut bytes = Vec::with_capacity(s.len);
    bytes.extend_from_slice(&e.gate);
    bytes.extend_from_slice(&e.down);
    if bytes.len() != s.len {
        return Err(format!("expert {id}: slab length changed"));
    }
    match s.loc {
        Loc::Vram(idx) => dev.hip.memcpy_htod(&dev.bufs[idx], &bytes).map_err(err),
        Loc::Host(p) => {
            // SAFETY: live mapped host allocation of `len` bytes; no kernel is in flight (every launch syncs).
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn kernarg(
    ptrs: [u64; 7],
    m: usize,
    k: usize,
    u: usize,
    c: usize,
    xdiv: usize,
    xsrc: usize,
    xrs: usize,
    ys: usize,
) -> [u8; KERNARG_BYTES] {
    let mut a = [0u8; KERNARG_BYTES];
    for (i, p) in ptrs.iter().enumerate() {
        a[i * 8..i * 8 + 8].copy_from_slice(&p.to_le_bytes());
    }
    for (off, v) in [
        (56, m),
        (60, k),
        (64, u),
        (68, c),
        (72, xdiv),
        (76, xsrc),
        (80, xrs),
        (84, ys),
        (88, N_EXPERTS),
    ] {
        a[off..off + 4].copy_from_slice(&(v as u32).to_le_bytes());
    }
    a
}

struct Ctx {
    hip: HipRuntime,
    funcs: Vec<(Kind, bool, Function)>,
}
impl Ctx {
    fn func(&self, kind: Kind, tiled: bool) -> &Function {
        &self
            .funcs
            .iter()
            .find(|(k, t, _)| *k == kind && *t == tiled)
            .expect("all symbols loaded")
            .2
    }
}

/// One prepared projection launch.
struct Prep {
    kind: Kind,
    tiled: bool,
    slots: usize,
    ys: usize,
    n_words: usize,
    y0: Buf,
    y1: Option<Buf>,
    args: [u8; KERNARG_BYTES],
    grid: [u32; 3],
    block: u32,
    bucket: Bucket,
    x: Vec<f32>,
    xrs: usize,
    c: usize,
}

fn prepare(
    dev: &mut Dev,
    case: &Case,
    kind: Kind,
    slabs: &Slabs,
    tiled: bool,
    bad_id: Option<(usize, u32)>,
) -> Result<Prep> {
    let slots = 10 * case.c;
    let (m, k) = (kind.m(), kind.k());
    let (x, xrs, ys, xsrc) = x_rows(case, kind);
    let xb = dev.upload(&f32_bytes(&x))?;
    let mut table = vec![0u64; N_EXPERTS];
    for (&id, s) in &slabs.by_id {
        table[id as usize] = if kind == Kind::GateUp {
            s.gate_ptr
        } else {
            s.down_ptr
        };
    }
    let tb = dev.upload(&u64_bytes(&table))?;
    let n_words = slots * ys;
    let y0 = dev.alloc(n_words * 4)?;
    let y1 = if kind == Kind::GateUp {
        Some(dev.alloc(n_words * 4)?)
    } else {
        None
    };
    let b = bucket(&case.route, 0..slots);
    let (args, grid, block) = if !tiled {
        let mut ids = b.ids.clone();
        if let Some((pos, id)) = bad_id {
            ids[pos] = id;
        }
        let ib = dev.upload(&u32_bytes(&ids))?;
        let ob = dev.upload(&u32_bytes(&b.offs))?;
        let sb = dev.upload(&u32_bytes(&b.slots))?;
        (
            kernarg(
                [
                    tb.ptr,
                    ib.ptr,
                    ob.ptr,
                    sb.ptr,
                    xb.ptr,
                    y0.ptr,
                    y1.map_or(0, |b| b.ptr),
                ],
                m,
                k,
                b.ids.len(),
                case.c,
                kind.x_row_div(),
                xsrc,
                xrs,
                ys,
            ),
            [m as u32, b.ids.len() as u32, 1],
            32u32,
        )
    } else {
        // Fixed-eight-row tiles: 482-word descriptors (ids[0..80], offsets[160..241], slots[322..402]; the second
        // bucket stays 0xffffffff), absolute slots, one descriptor per 80 slots.
        let tiles = case.c.div_ceil(8);
        let mut words = Vec::new();
        for t in 0..tiles {
            let tb_ = bucket(&case.route, 80 * t..(80 * t + 80).min(slots));
            let mut d = vec![u32::MAX; TILE_WORDS];
            d[..tb_.ids.len()].copy_from_slice(&tb_.ids);
            d[160..160 + tb_.offs.len()].copy_from_slice(&tb_.offs);
            let tail = *tb_.offs.last().unwrap();
            for w in &mut d[160 + tb_.offs.len()..160 + 81] {
                *w = tail;
            }
            d[322..322 + tb_.slots.len()].copy_from_slice(&tb_.slots);
            words.extend(d);
        }
        let db = dev.upload(&u32_bytes(&words))?;
        (
            kernarg(
                [
                    tb.ptr,
                    db.ptr,
                    db.ptr + 160 * 4,
                    db.ptr + 322 * 4,
                    xb.ptr,
                    y0.ptr,
                    y1.map_or(0, |b| b.ptr),
                ],
                m,
                k,
                80,
                case.c,
                kind.x_row_div(),
                xsrc,
                xrs,
                ys,
            ),
            [m.div_ceil(8) as u32, 80, tiles as u32],
            256u32,
        )
    };
    Ok(Prep {
        kind,
        tiled,
        slots,
        ys,
        n_words,
        y0,
        y1,
        args,
        grid,
        block,
        bucket: b,
        x,
        xrs,
        c: case.c,
    })
}

fn launch(ctx: &Ctx, dev: &Dev, p: &Prep) -> Result<()> {
    dev.hip
        .memset(&dev.bufs[p.y0.idx], 0xa5, p.n_words * 4)
        .map_err(err)?;
    if let Some(y1) = p.y1 {
        dev.hip
            .memset(&dev.bufs[y1.idx], 0xa5, p.n_words * 4)
            .map_err(err)?;
    }
    let mut a = p.args;
    // SAFETY: exact 96-byte ABI; every pointer is a live allocation of `dev` until release.
    unsafe {
        ctx.hip.launch_kernel_blob(
            ctx.func(p.kind, p.tiled),
            p.grid,
            [p.block, 1, 1],
            0,
            None,
            &mut a,
        )
    }
    .map_err(err)?;
    ctx.hip.device_synchronize().map_err(err)
}

fn read_words(dev: &Dev, b: Buf, n: usize) -> Result<Vec<u32>> {
    let bytes = dev.download(b, n * 4)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[derive(Default)]
struct Cmp {
    count: u64,
    first: Vec<Value>,
}
fn compare_words(label: &str, got: &[u32], want: &[f32], cmp: &mut Cmp) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if *g != w.to_bits() {
            cmp.count += 1;
            if cmp.first.len() < 8 {
                cmp.first.push(json!({"buffer": label, "word": i, "gpu": bits_hex(*g), "cpu": bits_hex(w.to_bits())}));
            }
        }
    }
}

struct RunOut {
    cmp: Cmp,
    twin_bad: u64,
    twin_checked: usize,
}

/// Prepare, launch and compare one (case, kind, placement, path) against both CPU oracles.
fn run_one(
    ctx: &Ctx,
    dev: &mut Dev,
    case: &Case,
    kind: Kind,
    slabs: &Slabs,
    tiled: bool,
) -> Result<RunOut> {
    let p = prepare(dev, case, kind, slabs, tiled, None)?;
    let exp = reference_on(
        &case.experts,
        kind,
        &p.bucket,
        p.c,
        &p.x,
        p.xrs,
        p.ys,
        p.slots,
    );
    launch(ctx, dev, &p)?;
    let g0 = read_words(dev, p.y0, p.n_words)?;
    let g1 = match p.y1 {
        Some(b) => Some(read_words(dev, b, p.n_words)?),
        None => None,
    };
    let mut cmp = Cmp::default();
    compare_words("y0", &g0, &exp.y0, &mut cmp);
    if let (Some(g), Some(e)) = (&g1, &exp.y1) {
        compare_words("y1", g, e, &mut cmp);
    }
    // The twin is checked against the GPU words (which equal the engine reference when cmp.count == 0).
    let (twin_bad, twin_checked) = twin_check(
        case,
        kind,
        &p.bucket,
        &p.x,
        p.xrs,
        p.ys,
        &g0.iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>(),
        g1.as_ref()
            .map(|g| g.iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>())
            .as_deref(),
        0,
    );
    Ok(RunOut {
        cmp,
        twin_bad,
        twin_checked,
    })
}

// ---------------------------------------------------------------------------------------
// Controls (all REQUIRED to be detected)
// ---------------------------------------------------------------------------------------

fn controls(
    ctx: &Ctx,
    dev: &mut Dev,
    case: &Case,
    placement: Placement,
    failures: &mut Vec<String>,
) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let slabs = upload_experts(dev, &case.experts, name_seed(&case.name), placement)?;
    for kind in Kind::ALL {
        let p = prepare(dev, case, kind, &slabs, false, None)?;
        let exp = reference_on(
            &case.experts,
            kind,
            &p.bucket,
            p.c,
            &p.x,
            p.xrs,
            p.ys,
            p.slots,
        );
        launch(ctx, dev, &p)?;
        let g0 = read_words(dev, p.y0, p.n_words)?;
        let g1 = match p.y1 {
            Some(b) => Some(read_words(dev, b, p.n_words)?),
            None => None,
        };
        let count = |g0: &[u32], g1: Option<&[u32]>| -> u64 {
            let mut c = Cmp::default();
            compare_words("y0", g0, &exp.y0, &mut c);
            if let (Some(g), Some(e)) = (g1, &exp.y1) {
                compare_words("y1", g, e, &mut c);
            }
            c.count
        };
        let base = count(&g0, g1.as_deref());
        let mut record = |name: &str, detected: bool, detail: Value| {
            if !detected {
                failures.push(format!("control {name} {kind:?} NOT detected"));
            }
            out.push(json!({"control": name, "profile": kind.tag(), "placement": placement.name(), "detected": detected, "detail": detail}));
        };
        record("baseline_equal", base == 0, json!({"mismatches": base}));
        // output_bit_flip: exactly one word.
        let mut f = g0.clone();
        f[3 * p.ys + 1] ^= 1 << 7;
        record("output_bit_flip", count(&f, g1.as_deref()) == 1, json!({}));
        // output_unwritten_word: a payload word replaced by the sentinel.
        let mut f = g0.clone();
        f[2 * p.ys] = SENTINEL_BITS;
        record(
            "output_unwritten_word",
            count(&f, g1.as_deref()) == 1,
            json!({}),
        );
        // wrong_law: a `2c-5`-like law (weights scaled by 8/7) must differ from the GPU on the twin sample.
        let (bad7, checked) = twin_check(
            case,
            kind,
            &p.bucket,
            &p.x,
            p.xrs,
            p.ys,
            &g0.iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>(),
            g1.as_ref()
                .map(|g| g.iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>())
                .as_deref(),
            7,
        );
        record(
            "wrong_level_law",
            bad7 > 0,
            json!({"twin_mismatches": bad7, "checked": checked}),
        );
        // weight_corruption: flip a code MSB in one expert's slab in placement memory.
        let victim = *case.route.first().unwrap();
        let mut corrupt = Expert::new(
            case.experts[&victim].gate.clone(),
            case.experts[&victim].down.clone(),
        );
        {
            let t = if kind == Kind::GateUp {
                &mut corrupt.gate
            } else {
                &mut corrupt.down
            };
            let at = 2 * (kind.k() * 3 / 8) + 48 + 15; // row 2, group 1, MSB of code 40
            t[at] ^= 0x04;
        }
        let corrupt = Rc::new(corrupt);
        overwrite_slab(dev, &slabs, victim, &corrupt)?;
        launch(ctx, dev, &p)?;
        let c0 = read_words(dev, p.y0, p.n_words)?;
        let c1 = match p.y1 {
            Some(b) => Some(read_words(dev, b, p.n_words)?),
            None => None,
        };
        let mut cexperts = case.experts.clone();
        cexperts.insert(victim, corrupt);
        let cexp = reference_on(&cexperts, kind, &p.bucket, p.c, &p.x, p.xrs, p.ys, p.slots);
        let mut cc = Cmp::default();
        compare_words("y0", &c0, &cexp.y0, &mut cc);
        if let (Some(g), Some(e)) = (&c1, &cexp.y1) {
            compare_words("y1", g, e, &mut cc);
        }
        let differs = c0 != g0 || c1 != g1;
        record(
            "weight_corruption_follows_placement",
            cc.count == 0 && differs,
            json!({"vs_corrupted_oracle": cc.count, "differs_from_original": differs}),
        );
        overwrite_slab(dev, &slabs, victim, &case.experts[&victim])?;
        launch(ctx, dev, &p)?;
        let r0 = read_words(dev, p.y0, p.n_words)?;
        let r1 = match p.y1 {
            Some(b) => Some(read_words(dev, b, p.n_words)?),
            None => None,
        };
        record(
            "weight_restore_equals_original",
            r0 == g0 && r1 == g1,
            json!({}),
        );
        // bad_ids: ids >= n_experts (and negative) exit with no load/store; the expert's slots stay sentinel.
        for bad in BAD_IDS {
            let pb = prepare(dev, case, kind, &slabs, false, Some((1, bad)))?;
            launch(ctx, dev, &pb)?;
            let b0 = read_words(dev, pb.y0, pb.n_words)?;
            let b1 = match pb.y1 {
                Some(b) => Some(read_words(dev, b, pb.n_words)?),
                None => None,
            };
            // Expected: the reference with the second expert's slots left unwritten.
            let mut e0 = exp.y0.clone();
            let mut e1 = exp.y1.clone();
            let skipped =
                p.bucket.slots[p.bucket.offs[1] as usize..p.bucket.offs[2] as usize].to_vec();
            for &s in &skipped {
                for r in 0..kind.out_rows() {
                    e0[s as usize * p.ys + r] = f32::from_bits(SENTINEL_BITS);
                    if let Some(e1) = e1.as_mut() {
                        e1[s as usize * p.ys + r] = f32::from_bits(SENTINEL_BITS);
                    }
                }
            }
            let mut bc = Cmp::default();
            compare_words("y0", &b0, &e0, &mut bc);
            if let (Some(g), Some(e)) = (&b1, &e1) {
                compare_words("y1", g, e, &mut bc);
            }
            record(
                "bad_expert_id_exits_untouched",
                bc.count == 0 && !skipped.is_empty(),
                json!({"id": bits_hex(bad), "mismatches": bc.count}),
            );
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TCodec {
    W3,
    E8p,
    Mq4,
}
impl TCodec {
    fn name(self) -> &'static str {
        match self {
            TCodec::W3 => "w3_mq3i4",
            TCodec::E8p => "e8p_i4",
            TCodec::Mq4 => "mq4sym",
        }
    }
    fn symbol(self, kind: Kind) -> String {
        match self {
            TCodec::W3 => w3_symbol(kind, false),
            TCodec::E8p => format!("e8p_i4_moe_gemv_{}_pm_gfx1100", kind.tag()),
            TCodec::Mq4 => format!("mq_i4_moe_gemv_mq4sym_{}_pm_gfx1100", kind.tag()),
        }
    }
    /// Bytes of one expert projection tensor.
    fn tensor_bytes(self, kind: Kind) -> usize {
        let (m, k) = (kind.m(), kind.k());
        match self {
            TCodec::W3 => kind.w3_bytes(),
            TCodec::E8p => m * k / 4 + m * (k / 128) * 2,
            TCodec::Mq4 => {
                if kind == Kind::GateUp {
                    m * (k / 256) * 136
                } else {
                    m * (k / 128) * 68
                }
            }
        }
    }
    fn gen(self, kind: Kind, rng: &mut Rng) -> Vec<u8> {
        let (m, k) = (kind.m(), kind.k());
        let scale =
            |rng: &mut Rng| (((8 + rng.below(13)) as u16) << 10) | (rng.next() as u16 & 0x3ff);
        match self {
            TCodec::W3 => gen_w3(rng, m, k, false),
            TCodec::E8p => {
                let mut out = vec![0u8; self.tensor_bytes(kind)];
                let words = m * k / 8;
                for i in 0..words {
                    out[2 * i..2 * i + 2].copy_from_slice(&(rng.next() as u16).to_le_bytes());
                }
                for g in 0..m * (k / 128) {
                    let at = 2 * words + 2 * g;
                    out[at..at + 2].copy_from_slice(&scale(rng).to_le_bytes());
                }
                out
            }
            TCodec::Mq4 => {
                let (stride, halves, payload_at, payload) = if kind == Kind::GateUp {
                    (136usize, 2usize, 8usize, 128usize)
                } else {
                    (68, 1, 4, 64)
                };
                let mut out = vec![0u8; self.tensor_bytes(kind)];
                for g in out.chunks_exact_mut(stride) {
                    for h in 0..halves {
                        let sc = scale(rng);
                        g[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
                        g[4 * h + 2..4 * h + 4]
                            .copy_from_slice(&(0x8000 | (sc + (3 << 10))).to_le_bytes());
                    }
                    for b in &mut g[payload_at..payload_at + payload] {
                        *b = rng.next() as u8;
                    }
                }
                out
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn time_codec(
    hip: &HipRuntime,
    funcs: &[(Kind, Function)],
    codec: TCodec,
    placement: Placement,
    c: usize,
    reps: usize,
    warmup: usize,
    launches: usize,
) -> Result<Vec<Value>> {
    let mut rows = Vec::new();
    // Routes: one lane for C=1 (U = 10), a union of `3C+7` experts otherwise.
    let pool_n = if c == 1 {
        10
    } else if c == 8 {
        80
    } else {
        3 * c + 7
    };
    let mut rng = Rng::new(0x71_0000 + c as u64);
    let ids: Vec<u32> = (0..pool_n as u32)
        .map(|i| (i * 5 + 3) % N_EXPERTS as u32)
        .collect();
    let route: Vec<u32> = if c == 8 {
        ids.clone()
    } else {
        lane_routes(&mut rng, &ids, c)
    };
    let slots = 10 * c;
    let b = bucket(&route, 0..slots);
    for kind in Kind::ALL {
        let mut dev = Dev::new(hip);
        let mut tensors: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
        for &id in &b.ids {
            tensors.insert(id, codec.gen(kind, &mut Rng::new(u64::from(id) + 99)));
        }
        let mut table = vec![0u64; N_EXPERTS];
        let mut i = 0usize;
        for (&id, t) in &tensors {
            table[id as usize] = if placement.host_for(i) {
                let (hp, dp) = dev.alloc_host(t.len())?;
                // SAFETY: live mapped host allocation of >= t.len() bytes.
                unsafe { std::ptr::copy_nonoverlapping(t.as_ptr(), hp, t.len()) };
                dp
            } else {
                dev.upload(t)?.ptr
            };
            i += 1;
        }
        let tb = dev.upload(&u64_bytes(&table))?;
        let (k, m) = (kind.k(), kind.m());
        let xsrc = if kind == Kind::GateUp { c } else { slots };
        let mut xr = Rng::new(7);
        let x: Vec<f32> = (0..xsrc * k)
            .map(|_| {
                f32::from_bits(
                    ((xr.next() & 1) as u32) << 31
                        | (120 + xr.below(8) as u32) << 23
                        | (xr.next() & 0x7f_ffff) as u32,
                )
            })
            .collect();
        let xb = dev.upload(&f32_bytes(&x))?;
        let ys = kind.out_rows();
        let y0 = dev.alloc(slots * ys * 4)?;
        let y1 = if kind == Kind::GateUp {
            Some(dev.alloc(slots * ys * 4)?)
        } else {
            None
        };
        let ib = dev.upload(&u32_bytes(&b.ids))?;
        let ob = dev.upload(&u32_bytes(&b.offs))?;
        let sb = dev.upload(&u32_bytes(&b.slots))?;
        let args = kernarg(
            [
                tb.ptr,
                ib.ptr,
                ob.ptr,
                sb.ptr,
                xb.ptr,
                y0.ptr,
                y1.map_or(0, |b| b.ptr),
            ],
            m,
            k,
            b.ids.len(),
            c,
            kind.x_row_div(),
            xsrc,
            k,
            ys,
        );
        let f = &funcs.iter().find(|(kk, _)| *kk == kind).unwrap().1;
        let grid = [m as u32, b.ids.len() as u32, 1];
        let start = hip.event_create().map_err(err)?;
        let stop = hip.event_create().map_err(err)?;
        let mut samples = Vec::with_capacity(reps);
        for i in 0..warmup + reps {
            hip.event_record(&start, None).map_err(err)?;
            for _ in 0..launches {
                let mut a = args;
                // SAFETY: as in `launch`.
                unsafe { hip.launch_kernel_blob(f, grid, [32, 1, 1], 0, None, &mut a) }
                    .map_err(err)?;
            }
            hip.event_record(&stop, None).map_err(err)?;
            hip.event_synchronize(&stop).map_err(err)?;
            let ms = hip.event_elapsed_ms(&start, &stop).map_err(err)?;
            if i >= warmup {
                samples.push(ms as f64 * 1000.0 / launches as f64);
            }
        }
        let _ = hip.event_destroy(start);
        let _ = hip.event_destroy(stop);
        let med = median(&samples);
        let bytes = codec.tensor_bytes(kind) * b.ids.len();
        rows.push(json!({"codec": codec.name(), "symbol": codec.symbol(kind), "profile": kind.tag(), "placement": placement.name(),
            "c": c, "slots": slots, "unique_experts": b.ids.len(), "tensor_bytes_per_expert": codec.tensor_bytes(kind),
            "weight_bytes_read": bytes, "median_us_per_launch": med, "min_us_per_launch": samples.iter().cloned().fold(f64::INFINITY, f64::min),
            "effective_gb_s": bytes as f64 / (med * 1e-6) / 1e9, "samples_us": samples, "launches_per_sample": launches}));
        dev.release();
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------

struct Cfg {
    codeobject: Option<String>,
    mq4_codeobject: Option<String>,
    e8p_codeobject: Option<String>,
    artifact: Option<String>,
    layers: Vec<usize>,
    real_experts: usize,
    device: Option<i32>,
    expect_uuid: String,
    expect_bdf: String,
    expect_arch: String,
    cpu_only: bool,
    timing: bool,
    reps: usize,
    warmup: usize,
    launches: usize,
    only: Vec<String>,
}

const USAGE: &str = "usage: mq_i4_moe_gemv_gfx1100_w3_g1 --codeobject PATH [--artifact HFQ --layers 0,23,47 --real-experts N] \
[--cpu-only] [--timing --mq4-codeobject PATH --e8p-codeobject PATH --reps N --warmup N --launches N] [--device N] \
[--expect-uuid U --expect-bdf B --expect-arch A] [--only SUBSTR]...";

fn parse_args(argv: &[String]) -> Result<Cfg> {
    let mut cfg = Cfg {
        codeobject: None,
        mq4_codeobject: None,
        e8p_codeobject: None,
        artifact: None,
        layers: vec![0, 23, 47],
        real_experts: 24,
        device: None,
        expect_uuid: DEFAULT_UUID.into(),
        expect_bdf: DEFAULT_BDF.into(),
        expect_arch: DEFAULT_ARCH.into(),
        cpu_only: false,
        timing: false,
        reps: 20,
        warmup: 3,
        launches: 8,
        only: Vec::new(),
    };
    let mut i = 0;
    let value = |i: &mut usize| -> Result<String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| USAGE.to_string())
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--codeobject" => cfg.codeobject = Some(value(&mut i)?),
            "--mq4-codeobject" => cfg.mq4_codeobject = Some(value(&mut i)?),
            "--e8p-codeobject" => cfg.e8p_codeobject = Some(value(&mut i)?),
            "--artifact" => cfg.artifact = Some(value(&mut i)?),
            "--layers" => {
                cfg.layers = value(&mut i)?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>().map_err(err))
                    .collect::<Result<_>>()?
            }
            "--real-experts" => cfg.real_experts = value(&mut i)?.parse().map_err(err)?,
            "--device" => cfg.device = Some(value(&mut i)?.parse().map_err(err)?),
            "--expect-uuid" => cfg.expect_uuid = value(&mut i)?,
            "--expect-bdf" => cfg.expect_bdf = value(&mut i)?.to_ascii_lowercase(),
            "--expect-arch" => cfg.expect_arch = value(&mut i)?,
            "--cpu-only" => cfg.cpu_only = true,
            "--timing" => cfg.timing = true,
            "--reps" => cfg.reps = value(&mut i)?.parse().map_err(err)?,
            "--warmup" => cfg.warmup = value(&mut i)?.parse().map_err(err)?,
            "--launches" => cfg.launches = value(&mut i)?.parse().map_err(err)?,
            "--only" => cfg.only.push(value(&mut i)?),
            // The fabric wrapper passes these through; they carry no semantics here.
            "--card" | "--fabric-owner" | "--fabric-wrapper" | "--fabric-card" => {
                value(&mut i)?;
            }
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
        i += 1;
    }
    if !cfg.cpu_only && cfg.codeobject.is_none() {
        return Err(USAGE.into());
    }
    if cfg.timing && (cfg.mq4_codeobject.is_none() || cfg.e8p_codeobject.is_none()) {
        return Err("--timing needs --mq4-codeobject and --e8p-codeobject".into());
    }
    Ok(cfg)
}

fn selected(cfg: &Cfg, name: &str) -> bool {
    cfg.only.is_empty() || cfg.only.iter().any(|s| name.contains(s.as_str()))
}

/// CPU-only oracle agreement: `decode_moe_slots` vs the CPU-decoder twin on every case of the run.
fn cpu_agree(case: &Case) -> Vec<Value> {
    let mut rows = Vec::new();
    for kind in Kind::ALL {
        let (x, xrs, ys, _) = x_rows(case, kind);
        let b = bucket(&case.route, 0..10 * case.c);
        let exp = reference_on(&case.experts, kind, &b, case.c, &x, xrs, ys, 10 * case.c);
        let (bad, checked) = twin_check(case, kind, &b, &x, xrs, ys, &exp.y0, exp.y1.as_deref(), 0);
        rows.push(json!({"case": case.name, "origin": case.origin, "profile": kind.tag(), "twin_rows_checked": checked, "twin_mismatches": bad}));
    }
    rows
}

fn load_image_funcs(
    hip: &HipRuntime,
    path: &str,
    symbols: &[(Kind, String)],
) -> Result<(Vec<(Kind, Function)>, String)> {
    let image = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let sha = hex(&sha(&image));
    let module = hip.module_load_data(&image).map_err(err)?;
    let mut funcs = Vec::new();
    for (kind, name) in symbols {
        funcs.push((
            *kind,
            hip.module_get_function(&module, name)
                .map_err(|e| format!("symbol {name}: {e}"))?,
        ));
    }
    Ok((funcs, sha))
}

fn run() -> Result<(Value, bool)> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cfg = parse_args(&argv)?;
    let mut failures: Vec<String> = Vec::new();

    // Artifact experts (read-only).
    let mut reals: Vec<(usize, Real)> = Vec::new();
    let mut artifact_meta = Value::Null;
    if let Some(path) = &cfg.artifact {
        let hfq = HfqFile::open_at_offset(std::path::Path::new(path), 0).map_err(err)?;
        for &layer in &cfg.layers {
            reals.push((layer, load_real(&hfq, path, layer, cfg.real_experts)?));
        }
        artifact_meta = json!({"path": path, "file_len": hfq.file_len(), "arch_id": hfq.arch_id,
            "index_sha256": hex(&hfq.index_sha256().map_err(err)?),
            "layers": reals.iter().map(|(l, r)| json!({"layer": l, "meta": r.meta})).collect::<Vec<_>>()});
    }

    // Case lists.
    let mut narrow: Vec<(Case, Vec<Placement>)> = Vec::new();
    for name in SYNTH_NARROW {
        if !selected(&cfg, name) {
            continue;
        }
        let mut pl = vec![Placement::Vram];
        if SYNTH_HOST.contains(&name) {
            pl.push(Placement::Host);
        }
        if SYNTH_SPLIT.contains(&name) {
            pl.push(Placement::Split);
        }
        narrow.push((synthetic(name), pl));
    }
    let mut tiled: Vec<(Case, Vec<Placement>)> = Vec::new();
    for (base, c) in SYNTH_TILED {
        let case = tiled_case(base, c);
        if selected(&cfg, &case.name) {
            let pl = if base == "c8_union33" && c == 9 {
                vec![Placement::Vram, Placement::Host]
            } else {
                vec![Placement::Vram]
            };
            tiled.push((case, pl));
        }
    }
    for (layer, real) in &reals {
        for case in real_cases(real, *layer) {
            if !selected(&cfg, &case.name) {
                continue;
            }
            if case.name.ends_with("_tiled") {
                tiled.push((case, vec![Placement::Vram, Placement::Host]));
            } else {
                narrow.push((
                    case,
                    vec![Placement::Vram, Placement::Host, Placement::Split],
                ));
            }
        }
    }

    // CPU-only gate: the two CPU oracles agree on every case (always run; it is the precondition of the GPU run).
    let mut cpu_rows = Vec::new();
    for (case, _) in narrow.iter().chain(tiled.iter()) {
        for row in cpu_agree(case) {
            if row["twin_mismatches"] != json!(0) {
                failures.push(format!(
                    "oracle disagreement on {} {}",
                    row["case"], row["profile"]
                ));
            }
            cpu_rows.push(row);
        }
    }
    let command = json!({"argv": argv, "build_command": "cargo build -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_gfx1100_w3_g1"});
    if cfg.cpu_only {
        let ok = failures.is_empty();
        return Ok((
            json!({"schema": SCHEMA, "mode": "cpu_only", "command": command, "artifact": artifact_meta,
            "oracle_agreement": cpu_rows, "failures": failures, "pass": ok, "rc": if ok { 0 } else { 1 }}),
            ok,
        ));
    }

    // Device.
    hip_bridge::keep_host_memory_out_of_reclaim("gfx1100 W3 G1 host-mapped weights");
    let hip = HipRuntime::load().map_err(err)?;
    if let Some(d) = cfg.device {
        hip.set_device(d).map_err(err)?;
    }
    let id = hip.current_device().map_err(err)?;
    let uuid = hip.device_uuid(id).map_err(err)?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let identity_match =
        uuid == cfg.expect_uuid && bdf == cfg.expect_bdf && arch == cfg.expect_arch;
    let device = json!({"index": id, "uuid": uuid, "bdf": bdf, "arch": arch, "expected_uuid": cfg.expect_uuid,
        "expected_bdf": cfg.expect_bdf, "expected_arch": cfg.expect_arch, "identity_match": identity_match});
    if !identity_match {
        return Ok((
            json!({"schema": SCHEMA, "command": command, "device": device, "error": "device identity mismatch",
            "byte_exact": false, "pass": false, "rc": 1}),
            false,
        ));
    }

    if cfg.timing {
        let w3_syms: Vec<(Kind, String)> = Kind::ALL
            .iter()
            .map(|k| (*k, w3_symbol(*k, false)))
            .collect();
        let (w3f, w3sha) = load_image_funcs(&hip, cfg.codeobject.as_ref().unwrap(), &w3_syms)?;
        let mq4_syms: Vec<(Kind, String)> = Kind::ALL
            .iter()
            .map(|k| (*k, TCodec::Mq4.symbol(*k)))
            .collect();
        let (mq4f, mq4sha) =
            load_image_funcs(&hip, cfg.mq4_codeobject.as_ref().unwrap(), &mq4_syms)?;
        let e8p_syms: Vec<(Kind, String)> = Kind::ALL
            .iter()
            .map(|k| (*k, TCodec::E8p.symbol(*k)))
            .collect();
        let (e8pf, e8psha) =
            load_image_funcs(&hip, cfg.e8p_codeobject.as_ref().unwrap(), &e8p_syms)?;
        let mut rows = Vec::new();
        for placement in [Placement::Vram, Placement::Host] {
            for c in [1usize, 4, 8] {
                for (codec, funcs) in [
                    (TCodec::W3, &w3f),
                    (TCodec::E8p, &e8pf),
                    (TCodec::Mq4, &mq4f),
                ] {
                    rows.extend(time_codec(
                        &hip,
                        funcs,
                        codec,
                        placement,
                        c,
                        cfg.reps,
                        cfg.warmup,
                        cfg.launches,
                    )?);
                }
            }
        }
        let loadavg = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        return Ok((
            json!({"schema": SCHEMA, "mode": "timing", "command": command, "device": device,
            "images": {"w3": w3sha, "mq4": mq4sha, "e8p": e8psha}, "loadavg": loadavg.trim(),
            "gpu_max_hw_queues": std::env::var("GPU_MAX_HW_QUEUES").ok(), "rows": rows, "pass": true, "rc": 0}),
            true,
        ));
    }

    let w3_syms: Vec<(Kind, String)> = Kind::ALL
        .iter()
        .map(|k| (*k, w3_symbol(*k, false)))
        .collect();
    let t8_syms: Vec<(Kind, String)> = Kind::ALL
        .iter()
        .map(|k| (*k, w3_symbol(*k, true)))
        .collect();
    let image = std::fs::read(cfg.codeobject.as_ref().unwrap()).map_err(err)?;
    let codeobject_sha = hex(&sha(&image));
    let module = hip.module_load_data(&image).map_err(err)?;
    let mut funcs = Vec::new();
    let mut loads = Vec::new();
    for (list, tiled_) in [(&w3_syms, false), (&t8_syms, true)] {
        for (kind, name) in list {
            let f = hip
                .module_get_function(&module, name)
                .map_err(|e| format!("symbol {name}: {e}"))?;
            loads.push(name.clone());
            funcs.push((*kind, tiled_, f));
        }
    }
    let ctx = Ctx { hip, funcs };

    let mut results: Vec<Value> = Vec::new();
    let (mut words, mut twin_rows, mut cases_run) = (0u64, 0usize, 0usize);
    let mut all: Vec<(&Case, &Vec<Placement>, bool)> = Vec::new();
    for (c, p) in &narrow {
        all.push((c, p, false));
    }
    for (c, p) in &tiled {
        all.push((c, p, true));
    }
    for (case, placements, is_tiled) in all {
        for &placement in placements {
            let mut dev = Dev::new(&ctx.hip);
            let slabs = upload_experts(&mut dev, &case.experts, name_seed(&case.name), placement)?;
            for kind in Kind::ALL {
                let r = run_one(&ctx, &mut dev, case, kind, &slabs, is_tiled)?;
                let n_words = (10
                    * case.c
                    * if case.pad {
                        kind.out_rows() + 16
                    } else {
                        kind.out_rows()
                    }) as u64
                    * if kind == Kind::GateUp { 2 } else { 1 };
                words += n_words;
                twin_rows += r.twin_checked;
                cases_run += 1;
                if r.cmp.count != 0 || r.twin_bad != 0 {
                    failures.push(format!(
                        "{} {} {} {}: {} word mismatches, {} twin mismatches",
                        case.name,
                        kind.tag(),
                        placement.name(),
                        if is_tiled { "t8" } else { "narrow" },
                        r.cmp.count,
                        r.twin_bad
                    ));
                }
                results.push(json!({"case": case.name, "origin": case.origin, "c": case.c, "path": if is_tiled { "t8" } else { "narrow" },
                    "placement": placement.name(), "profile": kind.tag(), "symbol": w3_symbol(kind, is_tiled),
                    "words_compared": n_words, "mismatch_count": r.cmp.count, "first_mismatches": r.cmp.first,
                    "twin_rows_checked": r.twin_checked, "twin_mismatches": r.twin_bad,
                    "slabs": {"host": slabs.host, "vram": slabs.vram}}));
            }
            dev.release();
        }
    }

    // Negative controls on a synthetic case and (when available) a real one.
    let mut control_rows = Vec::new();
    let ctl_synth = synthetic("c8_union33");
    for placement in [Placement::Vram, Placement::Host] {
        let mut dev = Dev::new(&ctx.hip);
        control_rows.extend(controls(
            &ctx,
            &mut dev,
            &ctl_synth,
            placement,
            &mut failures,
        )?);
        dev.release();
    }
    if let Some((layer, real)) = reals.first() {
        let case = real_cases(real, *layer).remove(1);
        let mut dev = Dev::new(&ctx.hip);
        control_rows.extend(controls(
            &ctx,
            &mut dev,
            &case,
            Placement::Split,
            &mut failures,
        )?);
        dev.release();
    }

    let ok = failures.is_empty() && cases_run > 0;
    Ok((
        json!({"schema": SCHEMA, "mode": "g1", "command": command, "device": device, "codeobject_sha256": codeobject_sha,
            "symbols_loaded": loads, "artifact": artifact_meta,
            "gpu_max_hw_queues": std::env::var("GPU_MAX_HW_QUEUES").ok(),
            "loadavg": std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
            "totals": {"cases_run": cases_run, "output_words_compared": words, "cpu_twin_rows_checked": twin_rows,
                "controls": control_rows.len(), "controls_detected": control_rows.iter().filter(|r| r["detected"] == json!(true)).count()},
            "oracle_agreement": cpu_rows, "results": results, "controls": control_rows,
            "failures": failures, "byte_exact": ok, "pass": ok, "rc": if ok { 0 } else { 1 }}),
        ok,
    ))
}

fn main() {
    match run() {
        Ok((json, ok)) => {
            println!("{json}");
            std::process::exit(if ok { 0 } else { 1 });
        }
        Err(e) => {
            println!(
                "{}",
                json!({"schema": SCHEMA, "error": e, "pass": false, "rc": 1})
            );
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
