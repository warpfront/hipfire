// SPDX-License-Identifier: Apache-2.0
//! K2 G1 oracle for the MQ3-I4 NT1 gate/up grouped IU4 GEMM (raw PM module launch; never runtime dispatch).
//! Build: cargo build -p hipfire-runtime --release --features lab --example qwen4_moe_sym_w3_g1_oracle
//! Run (on a locked card):
//!   qwen4_moe_sym_w3_g1_oracle --w3 kernels/qwen4_moe_iu4_sym_w3_pm_gfx1201.hxaco \
//!       --mq4 kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco [--seed N] [--expect-bdf 0000:..]
//!
//! Per scenario (rows 1/2/7/8/15/16/17, pad slots, sentinel tile, mixed experts, x_row_div 1 and 10):
//!  (1) the `_acc` entry's F32 accumulators == `mq_i4_reference::iu4_gemm_f32_parts`, bit for bit
//!      (pad slots +0), every output including unwritten-poison checks;
//!  (2) the shipped entry's BF16 H bytes == the unchanged MQ4 NT1 entry run on the bit-equivalent QT44
//!      tensors (signed nibble `2c-7` stored as `2c+1`, same f16 scale): the certified MQ4 epilogue is the
//!      byte anchor for identical F32 inputs.
//! Experts sit in one slab buffer at 256-byte offsets with 0xFF sentinel bytes between and after them; the
//! `sentinel_*` scenarios put 0xFF/0x00 low bytes into the scale plane's first scale (the bytes a last-row
//! chunk's unaligned dword read spills into).
use hip_bridge::{Function, HipRuntime};
use hipfire_runtime::{
    mq_i4::{self, MqI4Codec as Codec, MqI4Profile as Profile, MqI4Rotation as Rotation},
    mq_i4_reference::{self, A4Block},
};
use serde_json::{json, Value};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

static DOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn is_down() -> bool {
    DOWN.load(std::sync::atomic::Ordering::Relaxed)
}
/// FN geometry: gate/up `[1280, 2560]` (H = `[P][640]`), down `[2560, 640]` (Y = `[P][2560]`).
fn dm() -> usize {
    if is_down() {
        2560
    } else {
        1280
    }
}
fn dk() -> usize {
    if is_down() {
        640
    } else {
        2560
    }
}
fn dh() -> usize {
    if is_down() {
        2560
    } else {
        640
    }
}
fn dg() -> usize {
    dk() / 128
}
fn dthreads() -> u32 {
    if is_down() {
        128
    } else {
        64
    }
}
const BLK: usize = 72;
const SLAB_ALIGN: usize = 256;

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

struct Expert {
    codes: Vec<u8>,
    scales: Vec<u16>,
    w3: Vec<u8>,
    qt44: Vec<u8>,
}

fn profile() -> Profile {
    Profile {
        codec: Codec::W3,
        rotation: if is_down() {
            Rotation::R128
        } else {
            Rotation::R256
        },
    }
}

/// Anchor tensor holding the same signed values (stored nibble `2c-7+8 = 2c+1`, low nibble first):
/// gate/up QT44 (136 B per K256: two `[f16 sc, f16 zp]` headers, then two 64-byte nibble halves) or
/// down QT53 (68 B per K128: `[f16 sc, f16 zp]` then 64 nibble bytes).
fn qt44(codes: &[u8], scales: &[u16]) -> Vec<u8> {
    let gpb = if is_down() { 1 } else { 2 }; // K128 groups per anchor block
    let mut out = Vec::new();
    for r in 0..dm() {
        for g2 in 0..dg() / gpb {
            let mut block = vec![0u8; 4 * gpb + 64 * gpb];
            for h in 0..gpb {
                let g = gpb * g2 + h;
                let sc = scales[r * dg() + g];
                let zp: u16 = if (sc >> 10) & 0x1f != 0 && ((sc >> 10) & 0x1f) <= 27 {
                    (sc + (3 << 10)) | 0x8000
                } else {
                    0
                };
                block[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
                block[4 * h + 2..4 * h + 4].copy_from_slice(&zp.to_le_bytes());
                for j in 0..128 {
                    let u = 2 * codes[r * dk() + g * 128 + j] + 1;
                    block[4 * gpb + 64 * h + j / 2] |= u << (4 * (j % 2));
                }
            }
            out.extend_from_slice(&block);
        }
    }
    out
}

#[derive(Clone, Copy)]
enum Kind {
    Random,
    Extreme,
    EveryCode,
    Sentinel(u16),
}
fn expert(kind: Kind, seed: u32) -> Result<Expert> {
    let mut rng = Rng(seed | 1);
    let mid = profile().codec.midpoint();
    let mut codes: Vec<u8> = (0..dm() * dk()).map(|_| rng.below(8) as u8).collect();
    let pool = [0x3000u16, 0x3800, 0x3c00, 0x4000, 0x2c00, 0x3555];
    let mut scales: Vec<u16> = (0..dm() * dg())
        .map(|_| pool[rng.below(pool.len() as u32) as usize])
        .collect();
    match kind {
        Kind::Random => {}
        Kind::Extreme => {
            for (i, c) in codes.iter_mut().enumerate() {
                *c = if (i / 3) % 2 == 0 { 0 } else { 7 };
            }
            for (i, s) in scales.iter_mut().enumerate() {
                *s = [0x7bff, 0x0001, 0x0400, 0x3c00][i % 4];
            }
            // zero scale groups hold only the midpoint code
            for r in (0..dm()).step_by(97) {
                for g in 0..dg() {
                    if g % 5 == 0 {
                        scales[r * dg() + g] = 0;
                        for j in 0..128 {
                            codes[r * dk() + g * 128 + j] = mid;
                        }
                    }
                }
            }
        }
        Kind::EveryCode => {
            for (i, c) in codes.iter_mut().enumerate() {
                *c = ((i + i / dk()) % 8) as u8;
            }
        }
        Kind::Sentinel(first) => {
            scales[0] = first;
        }
    }
    let w3 = mq_i4::encode_tensor(profile(), dm(), dk(), &codes, &scales).map_err(err)?;
    let qt = qt44(&codes, &scales);
    Ok(Expert {
        codes,
        scales,
        w3,
        qt44: qt,
    })
}

/// 72-byte A4 blocks for `rows` rows, epoch-major (`[g][row]`); also the token-major copy for the twin.
fn make_x(rows: usize, seed: u32) -> (Vec<u8>, Vec<A4Block>) {
    let mut rng = Rng(seed | 1);
    let ds = [1.0f32, 0.5, 1.0 / 64.0, 1.0e-3, 3.0, 0.0, 65536.0, 1.0e-20];
    let mut tok: Vec<A4Block> = Vec::new();
    for r in 0..rows {
        for g in 0..dg() {
            let mut qs = [0u8; 64];
            let mut s = 0i32;
            for j in 0..128 {
                let v: i8 = match (r + g) % 9 {
                    0 => -8,
                    1 => 7,
                    2 => {
                        if j % 2 == 0 {
                            -8
                        } else {
                            7
                        }
                    }
                    _ => rng.below(16) as i8 - 8,
                };
                qs[j / 2] |= ((v as u8) & 0xf) << (4 * (j % 2));
                s += i32::from(v);
            }
            tok.push(A4Block {
                d: ds[(r * 3 + g) % ds.len()],
                s,
                qs,
            });
        }
    }
    let mut epoch_major = vec![0u8; rows * dg() * BLK];
    for g in 0..dg() {
        for r in 0..rows {
            let b = &tok[r * dg() + g];
            let o = (g * rows + r) * BLK;
            epoch_major[o..o + 4].copy_from_slice(&b.d.to_le_bytes());
            epoch_major[o + 4..o + 8].copy_from_slice(&b.s.to_le_bytes());
            epoch_major[o + 8..o + BLK].copy_from_slice(&b.qs);
        }
    }
    (epoch_major, tok)
}

struct Scenario {
    name: String,
    div: usize,
    rows: usize,
    tiles: Vec<(i32, [i32; 16])>,
    experts: Vec<Expert>,
}

fn tile(expert: i32, slots: &[i32]) -> (i32, [i32; 16]) {
    let mut t = [-1i32; 16];
    t[..slots.len()].copy_from_slice(slots);
    (expert, t)
}

/// `n` slots of one expert, in tiles of 16 (pad -1).
fn tiles_for(expert: i32, slots: Vec<i32>) -> Vec<(i32, [i32; 16])> {
    slots.chunks(16).map(|c| tile(expert, c)).collect()
}

fn scenarios(seed: u32, neg: u8) -> Result<Vec<Scenario>> {
    let mk = |kinds: &[Kind]| -> Result<Vec<Expert>> {
        kinds
            .iter()
            .enumerate()
            .map(|(i, k)| {
                expert(
                    *k,
                    seed.wrapping_mul(2654435761).wrapping_add(i as u32 * 7919),
                )
            })
            .collect()
    };
    let mut out = Vec::new();
    for (div, rows_per) in [(1usize, 20usize), (10, 4)] {
        for n in [1usize, 2, 7, 8, 15, 16, 17, 33] {
            let mut slots: Vec<i32> = (0..n)
                .map(|i| (((i * 7 + 3) % (rows_per * div)) as i32))
                .collect();
            slots.dedup();
            let experts = mk(&[Kind::Random, Kind::EveryCode, Kind::Extreme])?;
            let e = (n % 3) as i32;
            let mut tiles = tiles_for(e, slots);
            tiles.push((-1, [0; 16])); // sentinel tile: must store nothing
            out.push(Scenario {
                name: format!("rows{n}_div{div}"),
                div,
                rows: rows_per,
                tiles,
                experts,
            });
        }
    }
    // mixed experts and a non-monotone slot order
    let experts = mk(&[Kind::Random, Kind::Extreme, Kind::EveryCode])?;
    let mut tiles = tiles_for(2, vec![5, 3, 19, 0, 1, 2, 11, 12]);
    tiles.extend(tiles_for(0, (0..17).map(|i| (i * 3 % 20) as i32).collect()));
    tiles.extend(tiles_for(1, vec![7]));
    tiles.push(tile(0, &[-1, 4, -1, 9]));
    out.push(Scenario {
        name: "mixed_experts_unsorted_pad".into(),
        div: 1,
        rows: 20,
        tiles,
        experts,
    });
    for (tag, first) in [("sentinel_ff", 0x3cffu16), ("sentinel_00", 0x3c00u16)] {
        let experts = mk(&[Kind::Sentinel(first), Kind::Sentinel(first), Kind::Random])?;
        let tiles = tiles_for(1, (0..16).collect());
        out.push(Scenario {
            name: format!("{tag}_last_row_chunk"),
            div: 1,
            rows: 16,
            tiles,
            experts,
        });
    }
    // Negative controls (must fail): 1 flips a twin-side code, 2 flips an anchor-side (QT44) nibble of expert 0.
    for sc in &mut out {
        match neg {
            1 => sc.experts[0].codes[12345] ^= 1,
            2 => sc.experts[0].qt44[1000] ^= 0x10,
            _ => {}
        }
    }
    Ok(out)
}

fn align(n: usize) -> usize {
    n.div_ceil(SLAB_ALIGN) * SLAB_ALIGN
}

/// One slab buffer: each expert's tensor at a 256-byte aligned offset, 0xFF sentinels between and after.
fn slab(tensors: &[&[u8]]) -> (Vec<u8>, Vec<usize>) {
    let mut buf = Vec::new();
    let mut offs = Vec::new();
    for t in tensors {
        offs.push(buf.len());
        buf.extend_from_slice(t);
        buf.resize(align(buf.len()) + SLAB_ALIGN, 0xff);
    }
    (buf, offs)
}

fn run_scenario(
    hip: &HipRuntime,
    f_acc: &Function,
    f_w3: &Function,
    f_mq4: &Function,
    sc: &Scenario,
    seed: u32,
) -> Result<Value> {
    let rows = sc.rows;
    let (xq, x_tok) = make_x(rows, seed ^ 0x51ed);
    let m_total = sc.tiles.len() * 16;
    let tile_ids: Vec<i32> = sc.tiles.iter().map(|t| t.0).collect();
    let sorted: Vec<i32> = sc.tiles.iter().flat_map(|t| t.1).collect();
    let (w3_slab, w3_offs) = slab(
        &sc.experts
            .iter()
            .map(|e| e.w3.as_slice())
            .collect::<Vec<_>>(),
    );
    let (qt_slab, qt_offs) = slab(
        &sc.experts
            .iter()
            .map(|e| e.qt44.as_slice())
            .collect::<Vec<_>>(),
    );
    let up = |bytes: &[u8]| -> Result<hip_bridge::DeviceBuffer> {
        let b = hip.malloc(bytes.len().max(8)).map_err(err)?;
        hip.memcpy_htod(&b, bytes).map_err(err)?;
        Ok(b)
    };
    let w3_dev = up(&w3_slab)?;
    let qt_dev = up(&qt_slab)?;
    let ptrs = |base: u64, offs: &[usize]| -> Vec<u8> {
        offs.iter()
            .flat_map(|o| (base + *o as u64).to_le_bytes())
            .collect()
    };
    let w3_ptrs = up(&ptrs(w3_dev.as_ptr() as u64, &w3_offs))?;
    let qt_ptrs = up(&ptrs(qt_dev.as_ptr() as u64, &qt_offs))?;
    let tiles_dev = up(&tile_ids
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>())?;
    let sorted_dev = up(&sorted
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>())?;
    let x_dev = up(&xq)?;
    let acc_bytes = m_total * dm() * 4;
    let h_bytes = m_total * dh() * 2;
    let y_acc = hip.malloc(acc_bytes).map_err(err)?;
    let y_w3 = hip.malloc(h_bytes).map_err(err)?;
    let y_mq4 = hip.malloc(h_bytes).map_err(err)?;
    let launch = |f: &Function,
                  ptrs: &hip_bridge::DeviceBuffer,
                  y: &hip_bridge::DeviceBuffer|
     -> Result<()> {
        let mut a = [0u8; 64];
        for (off, p) in [
            (0usize, hip_ptr(ptrs)),
            (8, hip_ptr(&tiles_dev)),
            (16, hip_ptr(&sorted_dev)),
            (24, hip_ptr(&x_dev)),
            (32, hip_ptr(y)),
        ] {
            a[off..off + 8].copy_from_slice(&p.to_le_bytes());
        }
        for (off, v) in [
            (40usize, dm() as u32),
            (44, dk() as u32),
            (48, sc.div as u32),
            (52, m_total as u32),
            (56, rows as u32),
            (60, (ptrs.size() / 8) as u32),
        ] {
            a[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // SAFETY: exact 64-byte ABI (60 bytes + trailing n_experts); every buffer outlives the synchronize.
        unsafe {
            hip.launch_kernel_blob(
                f,
                [(dm() / 64) as u32, sc.tiles.len() as u32, 1],
                [dthreads(), 1, 1],
                0,
                None,
                &mut a,
            )
        }
        .map_err(err)?;
        hip.device_synchronize().map_err(err)
    };
    hip.memset(&y_acc, 0xa5, acc_bytes).map_err(err)?;
    hip.memset(&y_w3, 0xa5, h_bytes).map_err(err)?;
    hip.memset(&y_mq4, 0xa5, h_bytes).map_err(err)?;
    launch(f_acc, &w3_ptrs, &y_acc)?;
    launch(f_w3, &w3_ptrs, &y_w3)?;
    launch(f_mq4, &qt_ptrs, &y_mq4)?;
    let read = |b: &hip_bridge::DeviceBuffer, n: usize| -> Result<Vec<u8>> {
        let mut v = vec![0u8; n];
        hip.memcpy_dtoh(&mut v, b).map_err(err)?;
        Ok(v)
    };
    let (got_acc, got_w3, got_mq4) = (
        read(&y_acc, acc_bytes)?,
        read(&y_w3, h_bytes)?,
        read(&y_mq4, h_bytes)?,
    );
    // Twin F32 accumulators per used expert over every X row.
    let mut dense: Vec<Option<Vec<f32>>> = vec![None; sc.experts.len()];
    for t in &sc.tiles {
        if t.0 >= 0 && dense[t.0 as usize].is_none() {
            let e = &sc.experts[t.0 as usize];
            dense[t.0 as usize] = Some(mq_i4_reference::iu4_gemm_f32_parts(
                profile(),
                dm(),
                dk(),
                &e.codes,
                &e.scales,
                &x_tok,
            ));
        }
    }
    let (mut acc_bad, mut first_acc, mut checked) = (0usize, None, 0usize);
    for (ti, t) in sc.tiles.iter().enumerate() {
        for lane in 0..16 {
            let p = ti * 16 + lane;
            let live = t.0 >= 0 && t.1[lane] >= 0;
            for r in 0..dm() {
                let bits = u32::from_le_bytes(
                    got_acc[(p * dm() + r) * 4..(p * dm() + r) * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                let want = if t.0 < 0 {
                    0xa5a5_a5a5
                } else if live {
                    dense[t.0 as usize].as_ref().unwrap()[(t.1[lane] as usize / sc.div) * dm() + r]
                        .to_bits()
                } else {
                    0
                };
                checked += 1;
                if bits != want {
                    acc_bad += 1;
                    if first_acc.is_none() {
                        first_acc = Some(
                            json!({"p": p, "row": r, "gpu": format!("{bits:#010x}"), "twin": format!("{want:#010x}")}),
                        );
                    }
                }
            }
        }
    }
    // BF16 H: W3 entry vs the unchanged MQ4 entry on the equivalent QT44 tensors; sentinel tile untouched in both.
    let (mut h_bad, mut first_h) = (0usize, None);
    for i in 0..h_bytes / 2 {
        let (a, b) = (&got_w3[2 * i..2 * i + 2], &got_mq4[2 * i..2 * i + 2]);
        if a != b {
            h_bad += 1;
            if first_h.is_none() {
                first_h = Some(
                    json!({"index": i, "w3": format!("{:#06x}", u16::from_le_bytes([a[0], a[1]])), "mq4": format!("{:#06x}", u16::from_le_bytes([b[0], b[1]]))}),
                );
            }
        }
    }
    Ok(
        json!({"scenario": sc.name, "tiles": sc.tiles.len(), "acc_outputs_checked": checked, "acc_mismatches": acc_bad, "acc_first": first_acc, "h_mismatches": h_bad, "h_first": first_h}),
    )
}

fn hip_ptr(b: &hip_bridge::DeviceBuffer) -> u64 {
    b.as_ptr() as u64
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[derive(Default)]
struct TimeArgs {
    trace: String,
    layer_artifact: String,
    card: String,
    owner: String,
    out: String,
    reps: usize,
    warmup: usize,
}
impl TimeArgs {
    fn set(&mut self, flag: &str, v: String) -> Result<()> {
        match flag {
            "--trace" => self.trace = v,
            "--layer-artifact" => self.layer_artifact = v,
            "--card" => self.card = v,
            "--owner" => self.owner = v,
            "--time-out" => self.out = v,
            "--reps" => self.reps = v.parse().map_err(err)?,
            "--warmup" => self.warmup = v.parse().map_err(err)?,
            _ => return Err(format!("unknown timing flag {flag}")),
        }
        Ok(())
    }
}

/// Prefill NT1 layer-pass timing (`gp_bench.py` JsonTimingDriver input): per codec, `reps` HIP-event samples of ONE
/// launch covering every padded 16-slot tile of the trace's routed rows (ascending experts), after `warmup` untimed
/// launches. Weights are synthetic per-codec experts of the real FN geometry (kernel time is data independent).
/// Run under `fabric-run.sh --class timing`.
fn time_layer(
    hip: &HipRuntime,
    t: &TimeArgs,
    ids: (&String, &String, &String),
    variants: &[(&str, &str, &Vec<u8>, &Function)],
) -> Result<()> {
    let trace: Value =
        serde_json::from_slice(&std::fs::read(&t.trace).map_err(err)?).map_err(err)?;
    let layer: Value =
        serde_json::from_slice(&std::fs::read(&t.layer_artifact).map_err(err)?).map_err(err)?;
    let rows: Vec<usize> = trace["rows_per_expert"]
        .as_array()
        .ok_or("trace.rows_per_expert")?
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let tokens = trace
        .get("token_count")
        .or(trace.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .ok_or("trace token_count")? as usize;
    let tile_rows = layer["tile_rows"].as_u64().ok_or("layer.tile_rows")? as usize;
    let k_epochs = layer["k_epochs_per_tile"]
        .as_u64()
        .ok_or("layer.k_epochs_per_tile")? as usize;
    if tile_rows != 16 || layer["nt"].as_u64() != Some(1) {
        return Err("NT1 timing needs tile_rows 16 and nt 1".into());
    }
    let (reps, warmup) = (t.reps.max(20), t.warmup.max(3));
    let experts = rows.len();
    let routed: usize = rows.iter().sum();
    let (mut tile_ids, mut sorted) = (Vec::new(), Vec::new());
    let mut next = 0usize;
    let slots = tokens * 10;
    for (e, &n) in rows.iter().enumerate() {
        for chunk in (0..n).collect::<Vec<_>>().chunks(16) {
            tile_ids.push(e as i32);
            let mut t16 = [-1i32; 16];
            for (i, _) in chunk.iter().enumerate() {
                t16[i] = ((next * 7919) % slots) as i32;
                next += 1;
            }
            sorted.extend_from_slice(&t16);
        }
    }
    let (tiles, m_total) = (tile_ids.len(), tile_ids.len() * 16);
    let (xq, _) = {
        let (x, tok) = make_x(1, 7);
        (x, tok)
    };
    // activations: the one random row's blocks replicated across `tokens` rows, epoch-major
    let mut x_all = vec![0u8; dg() * tokens * BLK];
    for g in 0..dg() {
        for r in 0..tokens {
            x_all[(g * tokens + r) * BLK..(g * tokens + r + 1) * BLK]
                .copy_from_slice(&xq[g * BLK..(g + 1) * BLK]);
        }
    }
    let up = |bytes: &[u8]| -> Result<hip_bridge::DeviceBuffer> {
        let b = hip.malloc(bytes.len().max(8)).map_err(err)?;
        hip.memcpy_htod(&b, bytes).map_err(err)?;
        Ok(b)
    };
    let x_dev = up(&x_all)?;
    let tiles_dev = up(&tile_ids
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>())?;
    let sorted_dev = up(&sorted
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>())?;
    let y = hip.malloc(m_total * dh() * 2).map_err(err)?;
    let (ex_w3, ex_qt) = {
        let e = expert(Kind::Random, 11)?;
        (e.w3, e.qt44)
    };
    let mut runs = Vec::new();
    for (codec, symbol, module_bytes, f) in variants {
        let one = if *codec == "mq3i4" { &ex_w3 } else { &ex_qt };
        let mut slab = Vec::with_capacity(one.len() * experts);
        for _ in 0..experts {
            slab.extend_from_slice(one);
        }
        let dev = up(&slab)?;
        let ptrs = up(&(0..experts)
            .flat_map(|e| (dev.as_ptr() as u64 + (e * one.len()) as u64).to_le_bytes())
            .collect::<Vec<_>>())?;
        let launch = || -> Result<()> {
            let mut a = [0u8; 64];
            for (off, p) in [
                (0usize, hip_ptr(&ptrs)),
                (8, hip_ptr(&tiles_dev)),
                (16, hip_ptr(&sorted_dev)),
                (24, hip_ptr(&x_dev)),
                (32, hip_ptr(&y)),
            ] {
                a[off..off + 8].copy_from_slice(&p.to_le_bytes());
            }
            for (off, v) in [
                (40usize, dm() as u32),
                (44, dk() as u32),
                (48, 10),
                (52, m_total as u32),
                (56, tokens as u32),
                (60, experts as u32),
            ] {
                a[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
            // SAFETY: exact 64-byte ABI (60 bytes + trailing n_experts); buffers outlive the event synchronization below.
            unsafe {
                hip.launch_kernel_blob(
                    f,
                    [(dm() / 64) as u32, tiles as u32, 1],
                    [dthreads(), 1, 1],
                    0,
                    None,
                    &mut a,
                )
            }
            .map_err(err)
        };
        for _ in 0..warmup {
            launch()?;
        }
        hip.device_synchronize().map_err(err)?;
        let (e0, e1) = (
            hip.event_create().map_err(err)?,
            hip.event_create().map_err(err)?,
        );
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            hip.event_record(&e0, None).map_err(err)?;
            launch()?;
            hip.event_record(&e1, None).map_err(err)?;
            hip.event_synchronize(&e1).map_err(err)?;
            samples.push(f64::from(hip.event_elapsed_ms(&e0, &e1).map_err(err)?) / 1000.0);
        }
        runs.push(json!({"codec": codec, "symbol": symbol, "code_object_sha": sha256_hex(module_bytes), "timing_source": "device",
            "warmup_runs": warmup, "samples_s": samples, "launches_per_sample": 1, "routed_rows": routed, "tile_epochs": tiles * k_epochs,
            "padded_rows": m_total, "tiles": tiles, "weights": "synthetic per-codec experts, real FN geometry",
            "fabric": {"class": "timing", "card": t.card, "owner": t.owner, "wrapper": "/home/kaden/qcal/locks/fabric-sched/fabric-run.sh"},
            "device": {"arch": ids.2, "card": t.card, "uuid": ids.0, "bdf": ids.1}}));
    }
    let doc = json!({"runs": runs});
    if t.out.is_empty() {
        println!("{doc}");
    } else {
        std::fs::write(&t.out, serde_json::to_vec_pretty(&doc).map_err(err)?).map_err(err)?;
    }
    Ok(())
}

fn run() -> Result<()> {
    let (mut w3, mut mq4, mut seed, mut expect_bdf, mut neg, mut time_args) =
        (None, None, 1u32, None, 0u8, None::<TimeArgs>);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next().ok_or("flag needs a value")?;
        match a.as_str() {
            "--w3" => w3 = Some(v),
            "--mq4" => mq4 = Some(v),
            "--seed" => seed = v.parse().map_err(err)?,
            "--expect-bdf" => expect_bdf = Some(v.to_ascii_lowercase()),
            "--negative-control" => neg = v.parse().map_err(err)?,
            "--trace" | "--layer-artifact" | "--card" | "--owner" | "--time-out" | "--reps"
            | "--warmup" => {
                time_args.get_or_insert_with(TimeArgs::default).set(&a, v)?;
            }
            _ => return Err(format!("unknown flag {a}")),
        }
    }
    let (w3, mq4) = (w3.ok_or("--w3 required")?, mq4.ok_or("--mq4 required")?);
    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let (uuid, bdf, arch) = (
        hip.device_uuid(id).map_err(err)?,
        hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase(),
        hip.get_arch(id).map_err(err)?,
    );
    let device = json!({"uuid": uuid, "bdf": bdf, "arch": arch});
    if arch != "gfx1201" {
        return Err(format!("expected gfx1201, got {arch}"));
    }
    if let Some(b) = &expect_bdf {
        if *b != bdf {
            return Err(format!("expected BDF {b}, got {bdf}"));
        }
    }
    let (w3_bytes, mq4_bytes) = (
        std::fs::read(&w3).map_err(err)?,
        std::fs::read(&mq4).map_err(err)?,
    );
    let (m_w3, m_mq4) = (
        hip.module_load_data(&w3_bytes).map_err(err)?,
        hip.module_load_data(&mq4_bytes).map_err(err)?,
    );
    let sym = "qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1201";
    let f_acc = hip
        .module_get_function(&m_w3, &format!("{sym}_acc"))
        .map_err(err)?;
    let f_w3 = hip.module_get_function(&m_w3, sym).map_err(err)?;
    let f_mq4 = hip
        .module_get_function(&m_mq4, "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201")
        .map_err(err)?;
    if let Some(t) = time_args {
        let ids = (&uuid, &bdf, &arch);
        return time_layer(
            &hip,
            &t,
            ids,
            &[
                ("mq3i4", sym, &w3_bytes, &f_w3),
                (
                    "xts_mq4",
                    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201",
                    &mq4_bytes,
                    &f_mq4,
                ),
            ],
        );
    }
    let (mut rows, mut bad) = (Vec::new(), 0usize);
    let mut symbols = Vec::new();
    for down in [false, true] {
        DOWN.store(down, std::sync::atomic::Ordering::Relaxed);
        let (w3_sym, mq4_sym) = if down {
            (
                "qwen4_moe_down_iu4_sym_w3_pm_gfx1201",
                "qwen4_moe_down_iu4_sym_pm_gfx1201",
            )
        } else {
            (sym, "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201")
        };
        symbols.push(json!({"w3": w3_sym, "mq4_anchor": mq4_sym}));
        let f_acc = hip
            .module_get_function(&m_w3, &format!("{w3_sym}_acc"))
            .map_err(err)?;
        let f_w3 = hip.module_get_function(&m_w3, w3_sym).map_err(err)?;
        let f_mq4 = hip.module_get_function(&m_mq4, mq4_sym).map_err(err)?;
        for sc in scenarios(seed, neg)? {
            let mut r = run_scenario(&hip, &f_acc, &f_w3, &f_mq4, &sc, seed)?;
            r["kind"] = json!(if down { "down" } else { "gate_up" });
            bad += r["acc_mismatches"].as_u64().unwrap() as usize
                + r["h_mismatches"].as_u64().unwrap() as usize;
            rows.push(r);
        }
    }
    println!(
        "{}",
        json!({"schema": "qwen4-moe-sym-w3-g1-oracle-v1", "device": device, "w3_module": w3, "w3_module_sha256": sha256_hex(&w3_bytes),
        "mq4_module": mq4, "mq4_module_sha256": sha256_hex(&mq4_bytes), "symbols": symbols, "seed": seed, "negative_control": neg, "scenarios": rows, "mismatch_total": bad, "pass": bad == 0})
    );
    if bad != 0 {
        return Err("K2 W3 G1 mismatch".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
