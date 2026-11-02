// SPDX-License-Identifier: Apache-2.0
//! Byte-exact GPU check of the MQ3-I4 NT1 grouped IU4 MoE launchers
//! (`Gpu::gemm_qwen4_moe_w3_{gate_up_silu,down}_iu4_sym`, the embedded
//! `qwen4_moe_iu4_sym_w3_pm_gfx1201` module) on synthetic experts. Source: the raw-module oracle
//! `qwen4_moe_sym_w3_g1_oracle`; same scenarios (rows 1/2/7/8/15/16/17/33, pad slots, sentinel
//! tile, mixed experts, x_row_div 1 and 10, sentinel scale bytes), but every launch goes through the
//! runtime launchers and the production scratch sidecar handles.
//!
//! Build: cargo build -p hipfire-runtime --release --example qwen4_moe_sym_w3_launch_oracle
//! Run (gfx1201, on a locked card):
//!   cargo run -p hipfire-runtime --release --example qwen4_moe_sym_w3_launch_oracle -- \
//!       [--seed N] [--expect-bdf 0000:..] [--negative-control 1|2]
//!
//! Expected BF16 bytes per scenario, over the whole `[grouped_rows, H]` output:
//!  - down (`H = 2560`): `bf16_rne(F32 sum)` of `mq_i4_reference::iu4_gemm_f32_parts` (the epilogue
//!    is a pure round, so the CPU twin is a byte oracle);
//!  - gate/up (`H = 640`): the SwiGLU epilogue uses device exp2/rcp, so live rows are the bytes of the
//!    unchanged MQ4 launcher (`gemm_qwen4_moe_gate_up_silu_iu4_sym`) on the bit-equivalent QT44
//!    tensors (signed nibble `2c-7` stored as `2c+1`, same f16 scale);
//!  - both: padding slots +0, sentinel tile (`tile id -1`) untouched (poison `0xa5a5`), and the
//!    launcher output equals the MQ4 anchor launcher (QT44 gate/up, QT53 down) everywhere.
//! `--negative-control 1` flips a twin-side code (the down CPU twin must disagree), `2` flips an
//! anchor-side nibble (gate/up and down must disagree); either makes the run exit 1.
use hipfire_runtime::{
    mq_i4::{self, MqI4Codec as Codec, MqI4Profile as Profile, MqI4Rotation as Rotation},
    mq_i4_reference::{self, A4Block},
};
use rdna_compute::{Gpu, GpuTensor, Int4MmqPrepared};
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

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Uploads the shared A4 sidecar (`xq`, epoch-major `[k/128, rows]`) through the production scratch
/// reservation of the kind under test and seals it into the handle the launchers take.
fn upload_sidecar(gpu: &mut Gpu, xq: &[u8], rows: usize) -> Result<Int4MmqPrepared> {
    if is_down() {
        let res = gpu
            .scratch
            .reserve_qwen4_moe_down_i4(&gpu.hip, dk(), rows)
            .map_err(err)?;
        let buf = gpu
            .scratch
            .qwen4_moe_down_i4_scratch
            .as_ref()
            .ok_or("no down sidecar")?;
        gpu.hip.memcpy_htod(buf, xq).map_err(err)?;
        Ok(Int4MmqPrepared::from_reservation(res))
    } else {
        let res = gpu.reserve_int4_mmq(dk(), rows).map_err(err)?;
        let buf = gpu
            .scratch
            .int4_mmq_x_scratch
            .as_ref()
            .ok_or("no gate/up sidecar")?;
        gpu.hip.memcpy_htod(buf, xq).map_err(err)?;
        Ok(Int4MmqPrepared::from_reservation(res))
    }
}

fn run_scenario(gpu: &mut Gpu, sc: &Scenario, seed: u32) -> Result<Value> {
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
    let h_bytes = m_total * dh() * 2;
    let w3_dev = gpu.upload_raw(&w3_slab, &[w3_slab.len()]).map_err(err)?;
    let qt_dev = gpu.upload_raw(&qt_slab, &[qt_slab.len()]).map_err(err)?;
    let ptr_table = |base: u64, offs: &[usize]| -> Vec<u8> {
        offs.iter()
            .flat_map(|o| (base + *o as u64).to_le_bytes())
            .collect()
    };
    let w3_pt = ptr_table(w3_dev.buf.as_ptr() as u64, &w3_offs);
    let qt_pt = ptr_table(qt_dev.buf.as_ptr() as u64, &qt_offs);
    let w3_ptrs = gpu.upload_raw(&w3_pt, &[w3_pt.len()]).map_err(err)?;
    let qt_ptrs = gpu.upload_raw(&qt_pt, &[qt_pt.len()]).map_err(err)?;
    let tb = i32_bytes(&tile_ids);
    let sb = i32_bytes(&sorted);
    let tiles_dev = gpu.upload_raw(&tb, &[tb.len()]).map_err(err)?;
    let sorted_dev = gpu.upload_raw(&sb, &[sb.len()]).map_err(err)?;
    let poison = vec![0xa5u8; h_bytes];
    let y_w3 = gpu.upload_raw(&poison, &[h_bytes]).map_err(err)?;
    let y_mq4 = gpu.upload_raw(&poison, &[h_bytes]).map_err(err)?;
    let prepared = upload_sidecar(gpu, &xq, rows)?;
    if is_down() {
        gpu.gemm_qwen4_moe_w3_down_iu4_sym(
            &w3_ptrs,
            &tiles_dev,
            &sorted_dev,
            &prepared,
            &y_w3,
            dm(),
            dk(),
            sc.div,
            m_total,
            rows,
        )
        .map_err(err)?;
        gpu.gemm_qwen4_moe_down_iu4_sym(
            &qt_ptrs,
            &tiles_dev,
            &sorted_dev,
            &prepared,
            &y_mq4,
            dm(),
            dk(),
            sc.div,
            m_total,
            rows,
            false,
        )
        .map_err(err)?;
    } else {
        gpu.gemm_qwen4_moe_w3_gate_up_silu_iu4_sym(
            &w3_ptrs,
            &tiles_dev,
            &sorted_dev,
            &prepared,
            &y_w3,
            dm(),
            dk(),
            sc.div,
            m_total,
            rows,
        )
        .map_err(err)?;
        gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
            &qt_ptrs,
            &tiles_dev,
            &sorted_dev,
            &prepared,
            &y_mq4,
            dm(),
            dk(),
            sc.div,
            m_total,
            rows,
            false,
        )
        .map_err(err)?;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let got_w3 = gpu.download_raw_bytes(&y_w3).map_err(err)?;
    let got_mq4 = gpu.download_raw_bytes(&y_mq4).map_err(err)?;
    // Twin F32 sums per used expert over every X row (down CPU expectation only).
    let mut dense: Vec<Option<Vec<f32>>> = vec![None; sc.experts.len()];
    if is_down() {
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
    }
    // Expected bytes, whole output: sentinel tile 0xa5a5, padding +0, live rows per kind.
    let mut want = vec![0u8; h_bytes];
    for (ti, t) in sc.tiles.iter().enumerate() {
        for lane in 0..16 {
            let p = ti * 16 + lane;
            for c in 0..dh() {
                let at = (p * dh() + c) * 2;
                let bits: [u8; 2] = if t.0 < 0 {
                    [0xa5, 0xa5]
                } else if t.1[lane] < 0 {
                    [0, 0]
                } else if is_down() {
                    let sum = dense[t.0 as usize].as_ref().unwrap()
                        [(t.1[lane] as usize / sc.div) * dm() + c];
                    ((mq_i4_reference::bf16_rne_finite_passthrough_bits(sum.to_bits()) >> 16)
                        as u16)
                        .to_le_bytes()
                } else {
                    [got_mq4[at], got_mq4[at + 1]]
                };
                want[at..at + 2].copy_from_slice(&bits);
            }
        }
    }
    let diff = |a: &[u8], b: &[u8]| -> (usize, Option<Value>) {
        let (mut bad, mut first) = (0usize, None);
        for i in 0..a.len() / 2 {
            let (x, y) = (&a[2 * i..2 * i + 2], &b[2 * i..2 * i + 2]);
            if x != y {
                bad += 1;
                if first.is_none() {
                    first = Some(
                        json!({"index": i, "got": format!("{:#06x}", u16::from_le_bytes([x[0], x[1]])), "want": format!("{:#06x}", u16::from_le_bytes([y[0], y[1]]))}),
                    );
                }
            }
        }
        (bad, first)
    };
    let (exp_bad, exp_first) = diff(&got_w3, &want);
    let (anchor_bad, anchor_first) = diff(&got_w3, &got_mq4);
    let tensors: Vec<GpuTensor> = vec![
        w3_dev, qt_dev, w3_ptrs, qt_ptrs, tiles_dev, sorted_dev, y_w3, y_mq4,
    ];
    for t in tensors {
        gpu.free_tensor(t).map_err(err)?;
    }
    Ok(
        json!({"scenario": sc.name, "tiles": sc.tiles.len(), "h_values_checked": h_bytes / 2,
        "expected_mismatches": exp_bad, "expected_first": exp_first, "anchor_mismatches": anchor_bad, "anchor_first": anchor_first}),
    )
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn run() -> Result<()> {
    let (mut seed, mut expect_bdf, mut neg) = (1u32, None, 0u8);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next().ok_or("flag needs a value")?;
        match a.as_str() {
            "--seed" => seed = v.parse().map_err(err)?,
            "--expect-bdf" => expect_bdf = Some(v.to_ascii_lowercase()),
            "--negative-control" => neg = v.parse().map_err(err)?,
            _ => return Err(format!("unknown flag {a}")),
        }
    }
    let mut gpu = Gpu::init().map_err(err)?;
    let id = gpu.hip.current_device().map_err(err)?;
    let (uuid, bdf, arch) = (
        gpu.hip.device_uuid(id).map_err(err)?,
        gpu.hip
            .device_pci_bus_id(id)
            .map_err(err)?
            .to_ascii_lowercase(),
        gpu.hip.get_arch(id).map_err(err)?,
    );
    if arch != "gfx1201" || gpu.arch != "gfx1201" {
        return Err(format!("expected gfx1201, got {arch} / {}", gpu.arch));
    }
    if let Some(b) = &expect_bdf {
        if *b != bdf {
            return Err(format!("expected BDF {b}, got {bdf}"));
        }
    }
    let device = json!({"uuid": uuid, "bdf": bdf, "arch": arch});
    let symbols = gpu
        .qwen4_moe_sym_w3_gemm_symbols()
        .ok_or("no W3 symbols on this device")?;
    let (mut rows, mut bad) = (Vec::new(), 0usize);
    for down in [false, true] {
        DOWN.store(down, std::sync::atomic::Ordering::Relaxed);
        for sc in scenarios(seed, neg)? {
            let mut r = run_scenario(&mut gpu, &sc, seed)?;
            r["kind"] = json!(if down { "down" } else { "gate_up" });
            bad += r["expected_mismatches"].as_u64().unwrap() as usize
                + r["anchor_mismatches"].as_u64().unwrap() as usize;
            rows.push(r);
        }
    }
    println!(
        "{}",
        json!({"schema": "qwen4-moe-sym-w3-launch-oracle-v1", "device": device, "symbols": symbols,
        "w3_module_sha256": sha256_hex(rdna_compute::kernels::QWEN4_MOE_IU4_SYM_W3_PM_GFX1201), "seed": seed, "negative_control": neg,
        "scenarios": rows, "mismatch_total": bad, "pass": bad == 0})
    );
    if bad != 0 {
        return Err("W3 launcher mismatch".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
