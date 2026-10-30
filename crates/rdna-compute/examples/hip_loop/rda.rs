// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `rda_serial`: `gated_delta_net_vector_serial_f32`, the generic gated delta rule kernel the Raon RDA
//! layers launch (state tile in registers for the whole sequence, one wave32 workgroup per (head, 8 value
//! rows), xor-butterfly reductions). The incumbent is the shipped source exactly as the runtime compiles
//! it: module `gated_delta_net_vector_serial` (`Gpu::ensure_kernel` in `gated_delta_vector.rs`), source
//! `kernels/src/gated_delta_net_vector_serial.hip` (the file behind
//! `rdna_compute::gated_delta_vector::GATED_DELTA_NET_VECTOR_SERIAL_SRC`, which is `deltanet`-gated, so
//! the example includes the same bytes by path and builds under `--features lab` alone).
//!
//! Production launch (`Gpu::gated_delta_net_vector_serial_f32`): kernargs `(query, key, value, log_decay,
//! beta, state, output, slot_idx, n_dev` (pointers; the last two nullable), `n_tokens, n_heads, qk_group,
//! decay_per_key, round_out_bf16, slot, q_row_stride, k_row_stride, v_row_stride, decay_row_stride,
//! beta_row_stride, out_row_stride, state_head_stride, state_row_stride` (i32), `state_slot_stride`
//! (u64)`)`, 136 B; grid `[n_heads, 128 / 8, 1]`, block 32 (one wave32), no LDS. Raon's geometry
//! (`hipfire-arch-raon` `rda.rs` with `GatedDeltaStrides::contiguous`): H = 40, qk_group 1, K = V = 128,
//! per-key decay, bf16-rounded output, q/k/v/decay/out row stride `H * 128`, beta row stride `H`, state
//! head stride `128 * 128`, state row stride 128, state slot stride `H * 128 * 128`, slot 0, null
//! `slot_idx`/`n_dev`.
//!
//! Shapes: presets `pp512` (default, T = 512), `pp2048` (T = 2048), `tg` (T = 1), `emu` (T = 7, H = 3,
//! slot 1 of 2 slots, so slot addressing is exercised); keys `t h qk_group per_key(0|1) round(0|1) slot
//! slots dev(0|1)` and `arch=gfx1100|gfx1201` (the card the node is judged on; default gfx1100, the 7900
//! XTX; `gfx1201` is the R9700). `dev=1` passes device `slot_idx`/`n_dev` buffers holding `slot` and `t`
//! (the graph-capture form).
//!
//! Candidate header (`// hip_loop: ...`): `rows_per_wg` (value rows per workgroup, default 8; grid y =
//! `128 / rows_per_wg`, must divide 128), `block` (threads, default 32, multiple of 32, <= 1024),
//! `lds_dynamic` (bytes, default 0). Unknown keys are refused.
//!
//! Inputs are seeded and realistic: q, k unit-L2 per head (N(0,1)/sqrt(128), normalized), value N(0,1),
//! log_decay uniform in [-1, -1e-3] per key, beta uniform in (0, 1), state N(0, 0.1). The mutable
//! outputs are `state` (in place) and `output`; `reset` restores both from pristine copies and
//! `read_output` returns the output bytes followed by the touched slot's state bytes. The emulator gate
//! packs the output, the state (all slots) and a poison pad after each into ONE region (`out`): the
//! output element `(t, h, v)` is owned by workgroup `(h, v / rows_per_wg, 0)`, a state element `(h, v, k)`
//! of the touched slot likewise, and untouched slots and pads are `PAD_OWNER` (a store there fails).
//!
//! Included with `#[path = "hip_loop/rda.rs"] mod rda;`.
#![allow(dead_code)]

use crate::lane_common::{poison_words, word_bytes};
use crate::node::*;
use serde_json::{json, Value};

pub const MODULE: &str = "gated_delta_net_vector_serial";
pub const SYMBOL: &str = "gated_delta_net_vector_serial_f32";
pub const SOURCE: &str = include_str!("../../../../kernels/src/gated_delta_net_vector_serial.hip");
pub const GFX1201: &str = "gfx1201";
/// Head dim (K == V) the kernel is specialized for.
const D: usize = 128;
/// `GDV_TV`: value rows per production workgroup.
const ROWS_PER_WG: usize = 8;
const BLOCK: usize = 32;
/// Explicit kernarg bytes: 9 pointers, 14 i32, one u64.
pub const KERNARG_BYTES: usize = 136;
const SENTINEL: f32 = -1234.5;
const PRODUCTION_H: usize = 40;
const PRESETS: [(&str, usize); 3] = [("pp512", 512), ("pp2048", 2048), ("tg", 1)];
/// The `emu` preset: T, H, slot, slots.
const EMU: (usize, usize, usize, usize) = (7, 3, 1, 2);

/// `(offset, size, class)` of the explicit kernel arguments, the layout the AMDGPU metadata of the
/// shipped object reports.
pub fn expected_args() -> Vec<(u64, u64, &'static str)> {
    let mut v = Vec::new();
    for i in 0..9u64 {
        v.push((i * 8, 8, "pointer"));
    }
    for i in 0..14u64 {
        v.push((72 + i * 4, 4, "value"));
    }
    v.push((128, 8, "value"));
    v
}

struct State {
    q: Buf,
    k: Buf,
    v: Buf,
    decay: Buf,
    beta: Buf,
    state: Buf,
    state0: Buf,
    out: Buf,
    out0: Buf,
    slot_idx: Option<Buf>,
    n_dev: Option<Buf>,
}

pub struct Rda {
    label: String,
    t: usize,
    h: usize,
    qk_group: usize,
    per_key: bool,
    round: bool,
    slot: usize,
    slots: usize,
    dev: bool,
    arch: &'static str,
    st: Option<State>,
}

/// The seeded host buffers of one shape (little-endian f32 bytes).
pub struct Inputs {
    pub q: Vec<u8>,
    pub k: Vec<u8>,
    pub v: Vec<u8>,
    pub decay: Vec<u8>,
    pub beta: Vec<u8>,
    pub state0: Vec<u8>,
    pub out0: Vec<u8>,
}

fn flag(s: &mut ShapeSpec, key: &str, default: usize) -> Result<bool> {
    match s.take_usize(key, Some(default))? {
        0 => Ok(false),
        1 => Ok(true),
        v => Err(format!("rda_serial: {key}={v} must be 0 or 1")),
    }
}

/// Uniform in `[0, 1)`.
fn u01(r: &mut Xs64) -> f32 {
    (r.next_u64() >> 40) as f32 / 16_777_216.0
}

fn push_f32(out: &mut Vec<u8>, x: f32) {
    out.extend_from_slice(&x.to_le_bytes());
}

/// `rows` rows of 128 unit-L2 values (each row N(0,1)/sqrt(128), then normalized).
fn unit_rows(r: &mut Xs64, rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * D * 4);
    let mut row = [0f32; D];
    let amp = 1.0 / (D as f32).sqrt();
    for _ in 0..rows {
        let mut ss = 0f32;
        for x in row.iter_mut() {
            *x = amp * r.normal();
            ss += *x * *x;
        }
        let inv = 1.0 / ss.sqrt();
        for x in row {
            push_f32(&mut out, x * inv);
        }
    }
    out
}

impl Rda {
    pub fn new(shape: &ShapeSpec) -> Result<Self> {
        let mut s = shape.clone();
        let preset = s.preset.clone().unwrap_or_else(|| "pp512".to_owned());
        let emu = preset == "emu";
        let (pt, ph, pslot, pslots) = if emu {
            EMU
        } else {
            match PRESETS.iter().find(|(n, _)| *n == preset) {
                Some((_, t)) => (*t, PRODUCTION_H, 0, 1),
                None => {
                    return Err(format!(
                        "rda_serial: unknown shape preset `{preset}` (have pp512, pp2048, tg, emu)"
                    ));
                }
            }
        };
        let mut label = preset.clone();
        for (k, v) in &s.kv {
            label.push_str(&format!("-{k}{v}"));
        }
        let t = s.take_usize("t", Some(pt))?;
        let h = s.take_usize("h", Some(ph))?;
        let qk_group = s.take_usize("qk_group", Some(1))?;
        let per_key = flag(&mut s, "per_key", 1)?;
        let round = flag(&mut s, "round", 1)?;
        let slot = s.take_usize("slot", Some(pslot))?;
        let slots = s.take_usize("slots", Some(pslots))?;
        let dev = flag(&mut s, "dev", 0)?;
        let arch = match s.kv.remove("arch").as_deref() {
            None | Some(ARCH) => ARCH,
            Some(GFX1201) => GFX1201,
            Some(a) => return Err(format!("rda_serial: arch={a} must be gfx1100 or gfx1201")),
        };
        s.no_leftover()?;
        if t == 0 || t > i32::MAX as usize / 2 {
            return Err(format!("rda_serial: t={t} out of range"));
        }
        if h == 0 || h > 4096 || qk_group == 0 || h % qk_group != 0 {
            return Err(format!(
                "rda_serial: h={h} must be positive and divisible by qk_group={qk_group}"
            ));
        }
        if slots == 0 || slot >= slots {
            return Err(format!(
                "rda_serial: slot={slot} must be below slots={slots}"
            ));
        }
        Ok(Self {
            label,
            t,
            h,
            qk_group,
            per_key,
            round,
            slot,
            slots,
            dev,
            arch,
            st: None,
        })
    }

    fn st(&self) -> Result<&State> {
        self.st
            .as_ref()
            .ok_or_else(|| "rda_serial: setup has not run".to_owned())
    }

    fn hq(&self) -> usize {
        self.h / self.qk_group
    }
    fn q_stride(&self) -> usize {
        self.hq() * D
    }
    fn v_stride(&self) -> usize {
        self.h * D
    }
    fn decay_stride(&self) -> usize {
        if self.per_key {
            self.h * D
        } else {
            self.h
        }
    }
    fn beta_stride(&self) -> usize {
        self.h
    }
    /// Words of one head's state (`128 x 128`) and of one slot.
    fn slot_words(&self) -> usize {
        self.h * D * D
    }
    /// `(output words, word offset of the state segment, state words)` of the packed emulator region.
    fn arena(&self) -> (usize, usize, usize) {
        let outw = self.t * self.v_stride();
        (outw, outw + PAD_WORDS, self.slots * self.slot_words())
    }

    /// Seeded host buffers; deterministic in `seed`.
    pub fn host_inputs(&self, seed: u64) -> Inputs {
        let t = self.t;
        let mut rq = Xs64::new(seed ^ 0x5251);
        let mut rk = Xs64::new(seed ^ 0x4B4B);
        let mut rv = Xs64::new(seed ^ 0x5656);
        let mut rd = Xs64::new(seed ^ 0x4C44);
        let mut rb = Xs64::new(seed ^ 0x4254);
        let mut rs = Xs64::new(seed ^ 0x5354);
        let q = unit_rows(&mut rq, t * self.hq());
        let k = unit_rows(&mut rk, t * self.hq());
        let mut v = Vec::with_capacity(t * self.v_stride() * 4);
        for _ in 0..t * self.v_stride() {
            push_f32(&mut v, rv.normal());
        }
        let mut decay = Vec::with_capacity(t * self.decay_stride() * 4);
        for _ in 0..t * self.decay_stride() {
            push_f32(&mut decay, -(1e-3 + u01(&mut rd) * (1.0 - 1e-3)));
        }
        let mut beta = Vec::with_capacity(t * self.beta_stride() * 4);
        for _ in 0..t * self.beta_stride() {
            push_f32(
                &mut beta,
                ((rb.next_u64() >> 40) as f32 + 0.5) / 16_777_216.0,
            );
        }
        let mut state0 = Vec::with_capacity(self.slots * self.slot_words() * 4);
        for _ in 0..self.slots * self.slot_words() {
            push_f32(&mut state0, 0.1 * rs.normal());
        }
        let mut out0 = Vec::with_capacity(t * self.v_stride() * 4);
        for _ in 0..t * self.v_stride() {
            push_f32(&mut out0, SENTINEL);
        }
        Inputs {
            q,
            k,
            v,
            decay,
            beta,
            state0,
            out0,
        }
    }

    /// The production kernarg blob over the nine pointer values `p` (query, key, value, log_decay,
    /// beta, state, output, slot_idx, n_dev).
    pub fn kernarg_for(&self, p: [u64; 9]) -> Vec<u8> {
        let mut a = Args::new();
        for x in p {
            a.raw_ptr(x);
        }
        for v in self.scalar_args() {
            a.i32(v);
        }
        a.raw_ptr((self.slot_words()) as u64);
        let b = a.finish();
        debug_assert_eq!(b.len(), KERNARG_BYTES);
        b
    }

    /// The 14 i32 arguments after the pointers.
    fn scalar_args(&self) -> [i32; 14] {
        [
            self.t as i32,
            self.h as i32,
            self.qk_group as i32,
            i32::from(self.per_key),
            i32::from(self.round),
            self.slot as i32,
            self.q_stride() as i32,
            self.q_stride() as i32,
            self.v_stride() as i32,
            self.decay_stride() as i32,
            self.beta_stride() as i32,
            self.v_stride() as i32,
            (D * D) as i32,
            D as i32,
        ]
    }

    /// `rows_per_wg` of a candidate header (no header: 8), validated.
    fn rows_per_wg(&self, hdr: &Header) -> Result<usize> {
        let g = self.geometry_with(&mut hdr.clone())?;
        Ok(D / g.grid[1] as usize)
    }
}

impl Node for Rda {
    fn name(&self) -> &'static str {
        "rda_serial"
    }
    fn arch(&self) -> &'static str {
        self.arch
    }
    fn shape_label(&self) -> String {
        self.label.clone()
    }
    fn shape_json(&self) -> Value {
        json!({
            "t": self.t, "h": self.h, "qk_group": self.qk_group, "head_dim": D,
            "per_key": self.per_key, "round_out_bf16": self.round,
            "slot": self.slot, "slots": self.slots, "dev": self.dev, "arch": self.arch,
            "strides": {"q": self.q_stride(), "k": self.q_stride(), "v": self.v_stride(),
                "decay": self.decay_stride(), "beta": self.beta_stride(), "out": self.v_stride(),
                "state_head": D * D, "state_row": D, "state_slot": self.slot_words()},
            "inputs": "q,k unit-L2 per head; value N(0,1); log_decay U[-1,-1e-3]; beta U(0,1); state N(0,0.1)",
            "outputs": "output f32 [T][H][V] (sentinel-filled), state f32 [slots][H][V][K] (in place)",
        })
    }
    fn incumbent(&self) -> Result<Incumbent> {
        Ok(Incumbent {
            module: MODULE.to_owned(),
            source: SOURCE.to_owned(),
            symbol: SYMBOL.to_owned(),
            extra_flags: String::new(),
            selection: "crates/rdna-compute/src/gated_delta_vector.rs \
                        Gpu::gated_delta_net_vector_serial_f32 (ensure_kernel \
                        \"gated_delta_net_vector_serial\", GATED_DELTA_NET_VECTOR_SERIAL_SRC)"
                .to_owned(),
        })
    }
    fn geometry_with(&self, hdr: &mut Header) -> Result<Geometry> {
        let rows = hdr.take_usize("rows_per_wg", ROWS_PER_WG)?;
        let block = hdr.take_usize("block", BLOCK)?;
        let lds = hdr.take_usize("lds_dynamic", 0)?;
        hdr.no_leftover()?;
        if rows == 0 || D % rows != 0 {
            return Err(format!(
                "hip_loop header: rows_per_wg={rows} must divide {D}"
            ));
        }
        if block == 0 || block % 32 != 0 || block > 1024 {
            return Err(format!(
                "hip_loop header: block={block} must be a multiple of 32 in 32..=1024"
            ));
        }
        if lds > 64 * 1024 {
            return Err(format!(
                "hip_loop header: lds_dynamic={lds} exceeds the 64 KiB workgroup LDS"
            ));
        }
        Ok(Geometry {
            grid: [self.h as u32, (D / rows) as u32, 1],
            block: [block as u32, 1, 1],
            lds: lds as u32,
        })
    }
    fn work(&self) -> Work {
        let (t, h, d) = (self.t as f64, self.h as f64, D as f64);
        Work {
            flops: t * h * d * d * 8.0,
            bytes: 2.0 * h * d * d * 4.0 + t * (3.0 * h * d + h * d + h) * 4.0,
        }
    }
    fn record_extra(&self) -> Value {
        json!({"t": self.t, "h": self.h, "slot": self.slot, "slots": self.slots, "dev": self.dev})
    }
    fn static_hazards(&self) -> bool {
        true
    }
    fn extent_shapes(&self) -> Vec<String> {
        let arch = if self.arch == ARCH {
            String::new()
        } else {
            format!(",arch={}", self.arch)
        };
        [
            "pp512",
            "pp512,dev=1",
            "t=64,slot=1,slots=2,per_key=0,round=0",
        ]
        .iter()
        .map(|s| format!("{s}{arch}"))
        .collect()
    }
    fn out_owner<'a>(&'a self, hdr: &Header) -> Result<Box<dyn Fn(usize) -> [u32; 3] + 'a>> {
        let rows = self.rows_per_wg(hdr)?;
        let (outw, off_s, sw) = self.arena();
        let row = self.v_stride();
        let slot_words = self.slot_words();
        let touched = self.slot;
        Ok(Box::new(move |e| {
            let (head, v) = if e < outw {
                let r = e % row;
                (r / D, r % D)
            } else if e >= off_s && e < off_s + sw {
                let s = e - off_s;
                if s / slot_words != touched {
                    return PAD_OWNER;
                }
                let r = s % slot_words;
                (r / (D * D), r % (D * D) / D)
            } else {
                return PAD_OWNER;
            };
            [head as u32, (v / rows) as u32, 0]
        }))
    }
    fn emu_case(&self, seed: u64) -> Result<EmuCase> {
        let inp = self.host_inputs(seed);
        let (outw, _, sw) = self.arena();
        let mut out = inp.out0;
        out.extend_from_slice(&word_bytes(&poison_words(PAD_WORDS)));
        out.extend_from_slice(&inp.state0);
        out.extend_from_slice(&word_bytes(&poison_words(PAD_WORDS)));
        debug_assert_eq!(out.len(), (outw + PAD_WORDS + sw + PAD_WORDS) * 4);
        let mut bufs = vec![
            EmuBuf {
                name: "q",
                bytes: inp.q,
            },
            EmuBuf {
                name: "k",
                bytes: inp.k,
            },
            EmuBuf {
                name: "v",
                bytes: inp.v,
            },
            EmuBuf {
                name: "decay",
                bytes: inp.decay,
            },
            EmuBuf {
                name: "beta",
                bytes: inp.beta,
            },
        ];
        let (mut slot_idx, mut n_dev) = (EmuArg::Null, EmuArg::Null);
        if self.dev {
            slot_idx = EmuArg::Buf(bufs.len());
            bufs.push(EmuBuf {
                name: "slot_idx",
                bytes: (self.slot as i32).to_le_bytes().to_vec(),
            });
            n_dev = EmuArg::Buf(bufs.len());
            bufs.push(EmuBuf {
                name: "n_dev",
                bytes: (self.t as i32).to_le_bytes().to_vec(),
            });
        }
        let oi = bufs.len();
        bufs.push(EmuBuf {
            name: "out",
            bytes: out,
        });
        let mut args = vec![
            EmuArg::Buf(0),
            EmuArg::Buf(1),
            EmuArg::Buf(2),
            EmuArg::Buf(3),
            EmuArg::Buf(4),
            EmuArg::BufOff(oi, ((outw + PAD_WORDS) * 4) as u64),
            EmuArg::BufOff(oi, 0),
            slot_idx,
            n_dev,
        ];
        args.extend(self.scalar_args().map(EmuArg::I32));
        args.push(EmuArg::U64(self.slot_words() as u64));
        Ok(EmuCase {
            bufs,
            args,
            out: oi,
        })
    }

    fn setup(&mut self, gpu: &rdna_compute::Gpu, seed: u64) -> Result<()> {
        let i = self.host_inputs(seed);
        let (slot_idx, n_dev) = if self.dev {
            (
                Some(Buf::from_bytes(gpu, &(self.slot as i32).to_le_bytes())?),
                Some(Buf::from_bytes(gpu, &(self.t as i32).to_le_bytes())?),
            )
        } else {
            (None, None)
        };
        self.st = Some(State {
            q: Buf::from_bytes(gpu, &i.q)?,
            k: Buf::from_bytes(gpu, &i.k)?,
            v: Buf::from_bytes(gpu, &i.v)?,
            decay: Buf::from_bytes(gpu, &i.decay)?,
            beta: Buf::from_bytes(gpu, &i.beta)?,
            state: Buf::from_bytes(gpu, &i.state0)?,
            state0: Buf::from_bytes(gpu, &i.state0)?,
            out: Buf::from_bytes(gpu, &i.out0)?,
            out0: Buf::from_bytes(gpu, &i.out0)?,
            slot_idx,
            n_dev,
        });
        Ok(())
    }
    fn reset(&mut self, gpu: &rdna_compute::Gpu) -> Result<()> {
        let s = self.st()?;
        s.state.copy_from(gpu, &s.state0)?;
        s.out.copy_from(gpu, &s.out0)
    }
    fn kernarg(&self) -> Result<Vec<u8>> {
        let s = self.st()?;
        let p = |b: &Buf| b.ptr() as u64;
        let o = |b: &Option<Buf>| b.as_ref().map_or(0, |b| b.ptr() as u64);
        Ok(self.kernarg_for([
            p(&s.q),
            p(&s.k),
            p(&s.v),
            p(&s.decay),
            p(&s.beta),
            p(&s.state),
            p(&s.out),
            o(&s.slot_idx),
            o(&s.n_dev),
        ]))
    }
    fn read_output(&self, gpu: &rdna_compute::Gpu) -> Result<Vec<u8>> {
        let s = self.st()?;
        let mut out = s.out.download(gpu)?;
        let st = s.state.download(gpu)?;
        let sb = self.slot_words() * 4;
        out.extend_from_slice(&st[self.slot * sb..(self.slot + 1) * sb]);
        Ok(out)
    }
    fn teardown(&mut self, gpu: &rdna_compute::Gpu) {
        if let Some(s) = self.st.take() {
            for b in [
                s.q, s.k, s.v, s.decay, s.beta, s.state, s.state0, s.out, s.out0,
            ] {
                b.free(gpu);
            }
            for b in [s.slot_idx, s.n_dev].into_iter().flatten() {
                b.free(gpu);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(shape: &str) -> Rda {
        Rda::new(&ShapeSpec::parse(Some(shape)).unwrap()).unwrap()
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn shape_grammar_presets_and_keys() {
        let n = Rda::new(&ShapeSpec::parse(None).unwrap()).unwrap();
        assert_eq!((n.t, n.h, n.qk_group, n.slot, n.slots), (512, 40, 1, 0, 1));
        assert!(n.per_key && n.round && !n.dev);
        assert_eq!(n.arch(), "gfx1100");
        assert_eq!(node("pp2048").t, 2048);
        assert_eq!(node("tg").t, 1);
        let e = node("emu");
        assert_eq!((e.t, e.h, e.slot, e.slots), (7, 3, 1, 2));
        let n =
            node("pp512,t=9,h=6,qk_group=2,per_key=0,round=0,slot=1,slots=3,dev=1,arch=gfx1201");
        assert_eq!((n.t, n.h, n.qk_group, n.slot, n.slots), (9, 6, 2, 1, 3));
        assert!(!n.per_key && !n.round && n.dev);
        assert_eq!(n.arch(), "gfx1201");
        assert_eq!(n.decay_stride(), 6);
        assert_eq!(n.q_stride(), 3 * 128);
        for bad in [
            "nope",
            "pp512,zz=1",
            "pp512,per_key=2",
            "pp512,dev=3",
            "pp512,h=7,qk_group=2",
            "pp512,slot=1",
            "pp512,slot=2,slots=2",
            "pp512,t=0",
            "pp512,arch=gfx900",
            "pp512,h=0",
        ] {
            assert!(
                Rda::new(&ShapeSpec::parse(Some(bad)).unwrap()).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn production_geometry_and_header_geometry() {
        let n = node("pp512");
        let g = n.geometry();
        assert_eq!((g.grid, g.block, g.lds), ([40, 16, 1], [32, 1, 1], 0));
        assert_eq!(g.threads(), [1280, 16, 1]);
        let mut h = Header::parse("// hip_loop: rows_per_wg=16 block=64 lds_dynamic=256").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!((g.grid, g.block, g.lds), ([40, 8, 1], [64, 1, 1], 256));
        for bad in [
            "rows_per_wg=3",
            "rows_per_wg=0",
            "rows_per_wg=256",
            "block=48",
            "block=0",
            "block=2048",
            "lds_dynamic=70000",
            "waves=4",
        ] {
            let mut h = Header::parse_list(bad).unwrap();
            assert!(n.geometry_with(&mut h).is_err(), "{bad}");
        }
    }

    #[test]
    fn kernarg_layout_is_the_production_blob() {
        let n = node("pp512");
        let blob = n.kernarg_for([1, 2, 3, 4, 5, 6, 7, 0, 0]);
        assert_eq!(blob.len(), KERNARG_BYTES);
        let w = |o: usize| i32::from_le_bytes(blob[o..o + 4].try_into().unwrap());
        let q = |o: usize| u64::from_le_bytes(blob[o..o + 8].try_into().unwrap());
        for (i, v) in [1u64, 2, 3, 4, 5, 6, 7, 0, 0].iter().enumerate() {
            assert_eq!(q(i * 8), *v);
        }
        // n_tokens n_heads qk_group per_key round slot, then q k v decay beta out strides, then
        // state head/row strides, then the u64 slot stride.
        let want = [
            512, 40, 1, 1, 1, 0, 5120, 5120, 5120, 5120, 40, 5120, 16384, 128,
        ];
        for (i, v) in want.iter().enumerate() {
            assert_eq!(w(72 + i * 4), *v, "arg {i}");
        }
        assert_eq!(q(128), 40 * 128 * 128);
        let ex = expected_args();
        assert_eq!(ex.len(), 24);
        assert_eq!(*ex.last().unwrap(), (128, 8, "value"));
    }

    #[test]
    fn inputs_are_seeded_and_realistic() {
        let n = node("emu");
        let a = n.host_inputs(3);
        let b = n.host_inputs(3);
        assert_eq!(a.q, b.q);
        assert_eq!(a.state0, b.state0);
        assert_ne!(a.q, n.host_inputs(4).q);
        for rows in f32s(&a.q).chunks(128).chain(f32s(&a.k).chunks(128)) {
            let l2: f32 = rows.iter().map(|x| x * x).sum();
            assert!((l2 - 1.0).abs() < 1e-5, "{l2}");
        }
        assert!(f32s(&a.decay).iter().all(|x| (-1.0..=-1e-3).contains(x)));
        assert!(f32s(&a.beta).iter().all(|x| *x > 0.0 && *x < 1.0));
        assert!(f32s(&a.out0).iter().all(|x| *x == SENTINEL));
        assert_eq!(a.state0.len(), 2 * 3 * 128 * 128 * 4);
        assert_eq!(a.v.len(), 7 * 3 * 128 * 4);
    }

    #[test]
    fn emu_case_and_owner_map() {
        let n = node("emu");
        let c = n.emu_case(1).unwrap();
        assert_eq!(c.args.len(), 24);
        assert_eq!(c.args[7], EmuArg::Null);
        assert_eq!(c.args[23], EmuArg::U64(3 * 128 * 128));
        let (outw, off_s, sw) = n.arena();
        assert_eq!(c.bufs[c.out].bytes.len(), (off_s + sw + PAD_WORDS) * 4);
        let own = n.out_owner(&Header::default()).unwrap();
        // output element (t=2, h=1, v=77): workgroup (1, 77/8, 0)
        assert_eq!(own(2 * 384 + 128 + 77), [1, 9, 0]);
        // pads and the untouched slot 0 are PAD_OWNER; the touched slot 1: (h=2, v=5, k=9)
        assert_eq!(own(outw), PAD_OWNER);
        assert_eq!(own(off_s), PAD_OWNER);
        assert_eq!(own(off_s + sw), PAD_OWNER);
        let st = off_s + 3 * 128 * 128;
        assert_eq!(own(st + 2 * 16384 + 5 * 128 + 9), [2, 0, 0]);
        let mut h = Header::parse_list("rows_per_wg=32").unwrap();
        let own = n.out_owner(&h).unwrap();
        assert_eq!(own(2 * 384 + 128 + 77), [1, 2, 0]);
        h.kv.insert("bogus".into(), "1".into());
        assert!(n.out_owner(&h).is_err());
        let d = node("emu,dev=1").emu_case(1).unwrap();
        assert_eq!(d.bufs.len(), 8);
        assert_eq!(d.args[7], EmuArg::Buf(5));
        assert_eq!(d.args[8], EmuArg::Buf(6));
        assert_eq!(d.bufs[5].bytes, 1i32.to_le_bytes());
        assert_eq!(d.bufs[6].bytes, 7i32.to_le_bytes());
    }

    #[test]
    fn extent_shapes_build_and_carry_the_arch() {
        let n = node("emu");
        let s = n.extent_shapes();
        assert_eq!(
            s,
            [
                "pp512",
                "pp512,dev=1",
                "t=64,slot=1,slots=2,per_key=0,round=0"
            ]
        );
        for x in &s {
            Rda::new(&ShapeSpec::parse(Some(x)).unwrap()).unwrap();
        }
        let s = node("pp512,arch=gfx1201").extent_shapes();
        assert_eq!(s[1], "pp512,dev=1,arch=gfx1201");
        for x in &s {
            assert_eq!(node(x).arch(), "gfx1201");
        }
        assert!(n.static_hazards());
    }

    #[test]
    fn work_counts() {
        let w = node("pp512").work();
        assert_eq!(w.flops, 512.0 * 40.0 * 16384.0 * 8.0);
        assert_eq!(
            w.bytes,
            2.0 * 40.0 * 16384.0 * 4.0 + 512.0 * (4.0 * 40.0 * 128.0 + 40.0) * 4.0
        );
    }

    // ---- tests that compile (hipcc) and/or run the emulator; GPU-free ----

    /// sha256 of the runtime-compiled incumbent OBJECTS (`KernelCompiler`, module
    /// `gated_delta_net_vector_serial`) and of the kernel `.text`, per arch.
    const OBJ_SHA_GFX1100: &str =
        "93c1fdefae1636a090479e2e19fddf192729168f2ef151cd86e940f7c0853709";
    const OBJ_SHA_GFX1201: &str =
        "03bd53b1ebf18f633699cdfd828eff91b5ddd9316d2a56170bae43ec0d212b4b";
    const TEXT_SHA_GFX1100: &str =
        "d3526da88204b1c484755b6dc38a137ab7c3a349e9eb493a131bcb3db6d52eed";
    const TEXT_SHA_GFX1201: &str =
        "3b38fe30b526bf3f922e3d26d65fbc8ce0db8f639e8ff309a57389118afbf6aa";

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hip_loop-rda-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn run_opts(shape: &str, source: &str, tag: &str, emu: bool) -> (Value, crate::Outcome) {
        let dir = tmp(tag);
        let cand = dir.join("cand.hip");
        std::fs::write(&cand, source).unwrap();
        crate::run(&crate::Opts {
            node: "rda_serial".into(),
            shape: Some(shape.into()),
            cand: Some(cand),
            offline: true,
            emu,
            no_extent: emu,
            out: Some(dir.join("out")),
            ..crate::Opts::default()
        })
    }

    #[test]
    fn incumbent_is_the_shipped_source_and_module() {
        let i = node("pp512").incumbent().unwrap();
        assert_eq!(i.module, "gated_delta_net_vector_serial");
        assert_eq!(i.symbol, "gated_delta_net_vector_serial_f32");
        assert!(i.extra_flags.is_empty());
        assert!(i.source.contains("gated_delta_net_vector_serial_f32("));
        assert_eq!(node("pp512,arch=gfx1201").arch(), "gfx1201");
        assert!(crate::build_node("rda_serial", None).is_ok());
    }

    #[test]
    fn incumbent_objects_and_kernarg_layout_match_the_metadata() {
        if !crate::jit::hipcc_available() {
            eprintln!("skip: no hipcc");
            return;
        }
        let dir = tmp("objects");
        for (arch, obj, text) in [
            ("gfx1100", OBJ_SHA_GFX1100, TEXT_SHA_GFX1100),
            ("gfx1201", OBJ_SHA_GFX1201, TEXT_SHA_GFX1201),
        ] {
            let n = node(&format!("pp512,arch={arch}"));
            let inc = n.incumbent().unwrap();
            let b = crate::jit::compile_for(n.arch(), &dir, arch, &inc, &inc.source).unwrap();
            assert_eq!(b.image_sha256, obj, "{arch}: object");
            let k = b.kernel(&inc.symbol).unwrap();
            assert_eq!(k.code_sha256, text, "{arch}: .text");
            assert_eq!(k.wavefront_size, 32);
            assert_eq!(k.kernarg_bytes as usize, KERNARG_BYTES + 0, "{arch}");
            let want: Vec<(u64, u64, String)> = expected_args()
                .into_iter()
                .map(|(o, s, c)| (o, s, c.to_owned()))
                .collect();
            assert_eq!(k.explicit_args, want, "{arch}: explicit kernarg layout");
        }
    }

    #[test]
    fn offline_copy_reproduces_the_incumbent_text_and_a_mutation_differs() {
        if !crate::jit::hipcc_available() {
            eprintln!("skip: no hipcc");
            return;
        }
        let src = node("pp512").incumbent().unwrap().source;
        for (arch, text) in [("gfx1100", TEXT_SHA_GFX1100), ("gfx1201", TEXT_SHA_GFX1201)] {
            let (r, o) = run_opts(
                &format!("pp512,arch={arch}"),
                &src,
                &format!("off-{arch}"),
                false,
            );
            assert_eq!(o, crate::Outcome::Ok, "{}", r["error"]);
            assert_eq!(r["arch"], arch);
            assert_eq!(r["identical_code"], true);
            assert_eq!(r["baseline"]["code_sha256"], text);
            assert_eq!(r["candidate"]["code_sha256"], text);
            assert_eq!(
                r["baseline"]["geometry"]["grid_workgroups"],
                json!([40, 16, 1])
            );
        }
        let hdr = format!("// hip_loop: rows_per_wg=16 block=64\n{src}");
        let (r, _) = run_opts("pp512", &hdr, "off-hdr", false);
        assert_eq!(
            r["candidate"]["geometry"]["grid_workgroups"],
            json!([40, 8, 1])
        );
        assert_eq!(r["same_geometry"], false);
        let bad = src
            .replace("o > 0; o >>= 1", "o < 32; o <<= 1")
            .replace("int o = 16", "int o = 1");
        assert_ne!(bad, src);
        let (r, _) = run_opts("pp512", &bad, "off-mut", false);
        assert_eq!(r["identical_code"], false);
    }

    #[test]
    fn emu_small_gate_passes_the_copy_and_fails_wrong_candidates() {
        if !crate::jit::hipcc_available() {
            eprintln!("skip: no hipcc");
            return;
        }
        let src = node("pp512").incumbent().unwrap().source;
        let (r, _) = run_opts("emu", &src, "emu-copy", true);
        let g = &r["emu_gate"];
        assert_eq!(g["small_pass"], true, "{g}");
        assert_eq!(g["identical_to_baseline"], true);
        assert_eq!(g["workgroups"]["baseline"], 3 * 16);
        // butterfly order changed: the f32 sums (hence the state) differ in bits
        let order = src.replace("int o = 16; o > 0; o >>= 1", "int o = 1; o < 32; o <<= 1");
        assert_ne!(order, src);
        let (r, o) = run_opts("emu", &order, "emu-order", true);
        assert_eq!(o, crate::Outcome::IdentityFailed);
        assert_eq!(r["emu_gate"]["small_pass"], false, "{}", r["emu_gate"]);
        // a store past the output (into the pad after it)
        let store = src.replace(
            "if (lane == 0) out_row[r] = round_out_bf16 ? gdv_bf16_rn(y) : y;",
            "if (lane == 0) out_row[r] = round_out_bf16 ? gdv_bf16_rn(y) : y;\n            if (lane == 0 && t == n - 1) out_row[(long long)out_row_stride + r] = y;",
        );
        assert_ne!(store, src);
        let (r, o) = run_opts("emu", &store, "emu-store", true);
        assert_eq!(o, crate::Outcome::IdentityFailed);
        assert_eq!(r["emu_gate"]["small_pass"], false, "{}", r["emu_gate"]);
        // the dev=1 (slot_idx / n_dev) form passes too
        let (r, _) = run_opts("emu,dev=1", &src, "emu-dev", true);
        assert_eq!(r["emu_gate"]["small_pass"], true, "{}", r["emu_gate"]);
    }
}
