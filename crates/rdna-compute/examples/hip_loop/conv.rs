// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `conv_rows`: `causal_conv_depthwise_silu_rows_f32` (`kernels/src/causal_conv_depthwise_rows.hip`, runtime
//! module `causal_conv_depthwise_rows`, `rdna_compute::gated_delta_vector::CAUSAL_CONV_DEPTHWISE_ROWS_SRC`),
//! the Raon depthwise causal short conv over N rows with a raw-input history.
//!
//! Production launch (`gated_delta_vector.rs` `causal_conv_depthwise_rows`): kernargs
//! `(out, in, weight, hist_in, hist_out, n_rows, n_channels, ksize, rows_per_chunk, in_row_stride,
//! out_row_stride, w_ch_stride, w_tap_stride, hist_tap_stride, apply_silu, round_bf16)`: 5 pointers + 11 i32;
//! block 128; grid `[ceil(C/128), chunks, 1]`, `rows_per_chunk = N` when `hist_in == hist_out` (in place,
//! one chunk: Raon's `rda.rs` launch) else 64. Raon: C = 5120, ksize 4, strides 5120 / 5120, `w_ch_stride` 4,
//! `w_tap_stride` 1, `hist_tap_stride` 5120, silu 1, round_bf16 1, history in place.
//!
//! Shapes: presets `pp512` (default, N=512), `pp2048`, `tg` (N=1), `emu` (N=37, C=200 ragged vs 128,
//! in place); keys `n`, `c`, `k`, `inplace` (0/1), `silu`, `round`, `arch` (`gfx1100` default | `gfx1201`).
//! Seeded: `in` ~ N(0,1), `weight` ~ N(0,0.5), history ~ N(0,1). Mutable outputs: `out` and the history
//! buffer (`hist_out`, the same buffer as `hist_in` when in place); `read_output` = out bytes then history bytes.
//!
//! Candidate header: `block` (128; multiple of 32, <= 1024), `ch_per_wg` (128; grid X = ceil(C/ch_per_wg)),
//! `rows_per_wg` (production `rows_per_chunk`: N in place, else 64; grid Y = ceil(N/rows_per_wg)),
//! `hist_slice` (`last` default = the incumbent: the last row slice writes `hist_out`; `first`: row slice 0, the
//! only reader of `hist_in`, writes it after its own reads), `lds_dynamic` (0). An in-place history split over
//! several row slices is a cross-workgroup hazard the sequential emulator sees only through the gate's
//! reversed-workgroup-order pass (`emu_gate.wg_order_reversed`): a candidate whose slices communicate through
//! the history stores different bytes in one of the two orders and fails. The kernarg `rows_per_chunk` stays
//! the production value; the candidate reads its own tiling from its #defines and the identity gate decides.
//! The `--emu` output-owner map is: out element `(r, c)` -> WG `(c / ch_per_wg, r / rows_per_wg)`, history
//! element -> the `hist_slice` row slice of its channel tile.
//! The emulator output buffer is ONE packed region `[out N*C][PAD][hist (k-1)*C][PAD]`, so a race on the
//! history (or a write outside it) is judged in the same gate as the output.
//!
//! Included with `#[path = "hip_loop/conv.rs"] mod conv;`.
#![allow(dead_code)]

use crate::node::*;
use serde_json::{json, Value};

/// The shipped source, byte for byte what `gated_delta_vector::CAUSAL_CONV_DEPTHWISE_ROWS_SRC` includes (that
/// module is behind the `deltanet` feature; the judge builds with `lab` only).
const SOURCE: &str = include_str!("../../../../kernels/src/causal_conv_depthwise_rows.hip");

pub const SYMBOL: &str = "causal_conv_depthwise_silu_rows_f32";
const MODULE: &str = "causal_conv_depthwise_rows";
const CH_PER_WG: usize = 128;
const BLOCK: usize = 128;
const CHUNK_ROWS: usize = 64;
const SENTINEL: f32 = -1234.5;
const POISON: u32 = 0xDEAD_BEEF;
/// `(name, N, C, K)`.
pub const PRESETS: [(&str, usize, usize, usize); 4] = [
    ("pp512", 512, 5120, 4),
    ("pp2048", 2048, 5120, 4),
    ("tg", 1, 5120, 4),
    ("emu", 37, 200, 4),
];

struct State {
    inp: Buf,
    w: Buf,
    hin: Buf,
    out: Buf,
    /// Separate `hist_out` (not in place only).
    hout: Option<Buf>,
    out0: Buf,
    hist0: Buf,
    hout0: Option<Buf>,
}

pub struct ConvRows {
    label: String,
    n: usize,
    c: usize,
    k: usize,
    inplace: bool,
    silu: bool,
    round: bool,
    arch: &'static str,
    st: Option<State>,
}

/// `hist_slice` header value: `true` = row slice 0 writes the history (`first`), `false` = the last row slice
/// (`last`, the incumbent; also when absent).
fn hist_slice(v: Option<&str>) -> Result<bool> {
    match v {
        None | Some("last") => Ok(false),
        Some("first") => Ok(true),
        Some(o) => Err(format!(
            "hip_loop header: hist_slice={o} must be first|last"
        )),
    }
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl ConvRows {
    pub fn new(shape: &ShapeSpec) -> Result<Self> {
        let mut s = shape.clone();
        let preset = s.preset.clone().unwrap_or_else(|| "pp512".to_owned());
        let Some(&(label, pn, pc, pk)) = PRESETS.iter().find(|p| p.0 == preset) else {
            return Err(format!(
                "conv_rows: unknown shape preset `{preset}` (have pp512, pp2048, tg, emu)"
            ));
        };
        let n = s.take_usize("n", Some(pn))?;
        let c = s.take_usize("c", Some(pc))?;
        let k = s.take_usize("k", Some(pk))?;
        let inplace = s.take_usize("inplace", Some(1))?;
        let silu = s.take_usize("silu", Some(1))?;
        let round = s.take_usize("round", Some(1))?;
        let arch = match s.kv.remove("arch").as_deref() {
            None | Some("gfx1100") => "gfx1100",
            Some("gfx1201") => "gfx1201",
            Some(o) => return Err(format!("conv_rows: arch={o}: gfx1100 or gfx1201")),
        };
        s.no_leftover()?;
        if n == 0 || c == 0 || !(2..=8).contains(&k) {
            return Err(format!(
                "conv_rows: n={n} c={c} k={k}: need n,c > 0 and 2 <= k <= 8"
            ));
        }
        for (name, v) in [("inplace", inplace), ("silu", silu), ("round", round)] {
            if v > 1 {
                return Err(format!("conv_rows: {name}={v}: 0 or 1"));
            }
        }
        if n.checked_mul(c).is_none_or(|v| v > i32::MAX as usize) {
            return Err(format!(
                "conv_rows: n*c = {n}*{c} overflows the i32 row offsets"
            ));
        }
        let label = if s_is_default(shape) {
            label.to_owned()
        } else {
            let mut l = label.to_owned();
            for (k, v) in &shape.kv {
                l.push_str(&format!("-{k}{v}"));
            }
            l
        };
        Ok(Self {
            label,
            n,
            c,
            k,
            inplace: inplace == 1,
            silu: silu == 1,
            round: round == 1,
            arch,
            st: None,
        })
    }

    fn st(&self) -> Result<&State> {
        self.st
            .as_ref()
            .ok_or_else(|| "conv_rows: setup has not run".to_owned())
    }

    /// Production `rows_per_chunk`.
    fn rows_per_chunk(&self) -> usize {
        if self.inplace {
            self.n
        } else {
            CHUNK_ROWS
        }
    }

    fn hist_words(&self) -> usize {
        (self.k - 1) * self.c
    }

    /// Seeded host buffers `(in, weight, history)`.
    pub fn host_inputs(&self, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut r = Xs64::new(seed);
        let x: Vec<f32> = (0..self.n * self.c).map(|_| r.normal()).collect();
        let w: Vec<f32> = (0..self.c * self.k).map(|_| 0.5 * r.normal()).collect();
        let h: Vec<f32> = (0..self.hist_words()).map(|_| r.normal()).collect();
        (f32s(&x), f32s(&w), f32s(&h))
    }

    fn sentinel(words: usize) -> Vec<u8> {
        f32s(&vec![SENTINEL; words])
    }

    /// Word offset of the history segment in the packed emulator output region.
    fn hist_off(&self) -> usize {
        self.n * self.c + PAD_WORDS
    }

    fn karg_ints(&self) -> [i32; 11] {
        [
            self.n as i32,
            self.c as i32,
            self.k as i32,
            self.rows_per_chunk() as i32,
            self.c as i32,
            self.c as i32,
            self.k as i32,
            1,
            self.c as i32,
            self.silu as i32,
            self.round as i32,
        ]
    }
}

fn s_is_default(shape: &ShapeSpec) -> bool {
    shape.kv.is_empty()
}

impl Node for ConvRows {
    fn name(&self) -> &'static str {
        "conv_rows"
    }
    fn arch(&self) -> &'static str {
        self.arch
    }
    fn shape_label(&self) -> String {
        self.label.clone()
    }
    fn shape_json(&self) -> Value {
        json!({"n": self.n, "c": self.c, "k": self.k, "inplace": self.inplace, "silu": self.silu,
            "round_bf16": self.round, "rows_per_chunk": self.rows_per_chunk(), "arch": self.arch,
            "in": "f32 [N][C]", "out": "f32 [N][C] (overwritten)",
            "weight": "f32 [C][K] (w_ch_stride K, w_tap_stride 1)",
            "history": "f32 [K-1][C] (hist_tap_stride C, in place = hist_in == hist_out)"})
    }
    fn incumbent(&self) -> Result<Incumbent> {
        Ok(Incumbent {
            module: MODULE.to_owned(),
            source: SOURCE.to_owned(),
            symbol: SYMBOL.to_owned(),
            extra_flags: String::new(),
            selection: "crates/rdna-compute/src/gated_delta_vector.rs causal_conv_depthwise_rows \
                        (Raon: crates/hipfire-arch-raon/src/rda.rs)"
                .to_owned(),
        })
    }
    fn geometry_with(&self, hdr: &mut Header) -> Result<Geometry> {
        let block = hdr.take_usize("block", BLOCK)?;
        let chw = hdr.take_usize("ch_per_wg", CH_PER_WG)?;
        let rw = hdr.take_usize("rows_per_wg", self.rows_per_chunk())?;
        hist_slice(hdr.kv.remove("hist_slice").as_deref())?;
        let lds = hdr.take_usize("lds_dynamic", 0)?;
        hdr.no_leftover()?;
        if chw == 0 || rw == 0 {
            return Err(format!(
                "hip_loop header: ch_per_wg={chw} rows_per_wg={rw} must be positive"
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
        let g = |v: usize| u32::try_from(v).map_err(|_| format!("grid dimension {v} exceeds u32"));
        Ok(Geometry {
            grid: [g(self.c.div_ceil(chw))?, g(self.n.div_ceil(rw))?, 1],
            block: [block as u32, 1, 1],
            lds: lds as u32,
        })
    }
    fn work(&self) -> Work {
        let (n, c, k) = (self.n as f64, self.c as f64, self.k as f64);
        Work {
            flops: n * c * (2.0 * k + 4.0),
            bytes: n * c * 8.0 + c * k * 4.0,
        }
    }
    fn static_hazards(&self) -> bool {
        true
    }
    fn extent_shapes(&self) -> Vec<String> {
        let sfx = if self.arch == "gfx1100" {
            String::new()
        } else {
            format!(",arch={}", self.arch)
        };
        ["pp512", "n=4096,c=200", "pp512,inplace=0"]
            .iter()
            .map(|s| format!("{s}{sfx}"))
            .collect()
    }
    fn out_owner<'a>(&'a self, hdr: &Header) -> Result<Box<dyn Fn(usize) -> [u32; 3] + 'a>> {
        let get = |k: &str, d: usize| -> Result<usize> {
            match hdr.kv.get(k) {
                Some(v) => v
                    .parse::<usize>()
                    .map_err(|e| format!("hip_loop header {k}={v}: {e}")),
                None => Ok(d),
            }
        };
        let chw = get("ch_per_wg", CH_PER_WG)?;
        let rw = get("rows_per_wg", self.rows_per_chunk())?;
        if chw == 0 || rw == 0 {
            return Err("hip_loop header: ch_per_wg/rows_per_wg must be positive".into());
        }
        let (n, c) = (self.n, self.c);
        let last = if hist_slice(hdr.kv.get("hist_slice").map(String::as_str))? {
            0
        } else {
            (n.div_ceil(rw) - 1) as u32
        };
        let (hoff, hend) = (self.hist_off(), self.hist_off() + self.hist_words());
        Ok(Box::new(move |e| {
            if e < n * c {
                [((e % c) / chw) as u32, ((e / c) / rw) as u32, 0]
            } else if e >= hoff && e < hend {
                [(((e - hoff) % c) / chw) as u32, last, 0]
            } else {
                PAD_OWNER
            }
        }))
    }
    fn emu_case(&self, seed: u64) -> Result<EmuCase> {
        let (x, w, h) = self.host_inputs(seed);
        let poison = || POISON.to_le_bytes().repeat(PAD_WORDS);
        let mut packed = Self::sentinel(self.n * self.c);
        packed.extend(poison());
        packed.extend(if self.inplace {
            h.clone()
        } else {
            Self::sentinel(self.hist_words())
        });
        packed.extend(poison());
        let mut bufs = vec![
            EmuBuf {
                name: "out_hist",
                bytes: packed,
            },
            EmuBuf {
                name: "in",
                bytes: x,
            },
            EmuBuf {
                name: "weight",
                bytes: w,
            },
        ];
        let hoff = (self.hist_off() * 4) as u64;
        let hin = if self.inplace {
            EmuArg::BufOff(0, hoff)
        } else {
            bufs.push(EmuBuf {
                name: "hist_in",
                bytes: h,
            });
            EmuArg::Buf(3)
        };
        let mut args = vec![
            EmuArg::BufOff(0, 0),
            EmuArg::Buf(1),
            EmuArg::Buf(2),
            hin,
            EmuArg::BufOff(0, hoff),
        ];
        args.extend(self.karg_ints().map(EmuArg::I32));
        Ok(EmuCase { bufs, args, out: 0 })
    }

    fn setup(&mut self, gpu: &Gpu, seed: u64) -> Result<()> {
        let (x, w, h) = self.host_inputs(seed);
        let sent = Self::sentinel(self.n * self.c);
        let (hout, hout0) = if self.inplace {
            (None, None)
        } else {
            let hs = Self::sentinel(self.hist_words());
            (
                Some(Buf::from_bytes(gpu, &hs)?),
                Some(Buf::from_bytes(gpu, &hs)?),
            )
        };
        self.st = Some(State {
            inp: Buf::from_bytes(gpu, &x)?,
            w: Buf::from_bytes(gpu, &w)?,
            hin: Buf::from_bytes(gpu, &h)?,
            out: Buf::from_bytes(gpu, &sent)?,
            hout,
            out0: Buf::from_bytes(gpu, &sent)?,
            hist0: Buf::from_bytes(gpu, &h)?,
            hout0,
        });
        Ok(())
    }
    fn reset(&mut self, gpu: &Gpu) -> Result<()> {
        let s = self.st()?;
        s.out.copy_from(gpu, &s.out0)?;
        match (&s.hout, &s.hout0) {
            (Some(o), Some(o0)) => o.copy_from(gpu, o0),
            _ => s.hin.copy_from(gpu, &s.hist0),
        }
    }
    fn kernarg(&self) -> Result<Vec<u8>> {
        let s = self.st()?;
        let mut a = Args::new();
        a.ptr(&s.out)
            .ptr(&s.inp)
            .ptr(&s.w)
            .ptr(&s.hin)
            .ptr(s.hout.as_ref().unwrap_or(&s.hin));
        for v in self.karg_ints() {
            a.i32(v);
        }
        Ok(a.finish())
    }
    fn read_output(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        let s = self.st()?;
        let mut v = s.out.download(gpu)?;
        v.extend(s.hout.as_ref().unwrap_or(&s.hin).download(gpu)?);
        Ok(v)
    }
    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(s) = self.st.take() {
            for b in [s.inp, s.w, s.hin, s.out, s.out0, s.hist0] {
                b.free(gpu);
            }
            for b in [s.hout, s.hout0].into_iter().flatten() {
                b.free(gpu);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(shape: &str) -> ConvRows {
        ConvRows::new(&ShapeSpec::parse(Some(shape)).unwrap()).unwrap()
    }

    #[test]
    fn production_geometry_and_presets() {
        let n = ConvRows::new(&ShapeSpec::parse(None).unwrap()).unwrap();
        assert_eq!(n.label, "pp512");
        let g = n.geometry();
        assert_eq!((g.grid, g.block, g.lds), ([40, 1, 1], [128, 1, 1], 0));
        assert_eq!(node("pp2048").geometry().grid, [40, 1, 1]);
        assert_eq!(node("tg").geometry().grid, [40, 1, 1]);
        assert_eq!(node("emu").geometry().grid, [2, 1, 1]);
        // not in place: 64-row chunks
        assert_eq!(node("pp512,inplace=0").geometry().grid, [40, 8, 1]);
        assert_eq!(node("pp512").rows_per_chunk(), 512);
        assert_eq!(node("emu").arch(), "gfx1100");
        assert_eq!(node("arch=gfx1201").arch(), "gfx1201");
    }

    #[test]
    fn shape_grammar_refusals() {
        for s in [
            "nope",
            "pp512,q=1",
            "k=1",
            "k=9",
            "n=0",
            "inplace=2",
            "arch=gfx900",
            "pp512,tg",
        ] {
            let r = ShapeSpec::parse(Some(s)).and_then(|sp| ConvRows::new(&sp));
            assert!(r.is_err(), "{s}");
        }
        let n = node("n=64,c=300,k=3,silu=0,round=0");
        assert_eq!((n.n, n.c, n.k, n.silu, n.round), (64, 300, 3, false, false));
    }

    #[test]
    fn header_geometry_and_refusals() {
        let n = node("pp512,inplace=0");
        let mut h = Header::parse("// hip_loop: block=256 ch_per_wg=256 rows_per_wg=64").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!((g.grid, g.block), ([20, 8, 1], [256, 1, 1]));
        for bad in [
            "waves=4",
            "grid_x=4",
            "block=100",
            "block=2048",
            "ch_per_wg=0",
            "rows_per_wg=0",
            "lds_dynamic=70000",
        ] {
            let mut h = Header::parse(&format!("// hip_loop: {bad}")).unwrap();
            assert!(n.geometry_with(&mut h).is_err(), "{bad}");
        }
        // In place, a row split is admitted (the emu gate's reversed-order pass judges it); `hist_slice` picks
        // the history writer for the owner map; full rows is the production launch.
        let ip = node("pp512");
        let mut h = Header::parse("// hip_loop: rows_per_wg=64 hist_slice=first").unwrap();
        assert_eq!(ip.geometry_with(&mut h).unwrap().grid, [40, 8, 1]);
        let mut h = Header::parse("// hip_loop: hist_slice=middle").unwrap();
        assert!(ip.geometry_with(&mut h).is_err());
        let mut h = Header::parse("// hip_loop: rows_per_wg=512 ch_per_wg=64").unwrap();
        assert_eq!(ip.geometry_with(&mut h).unwrap().grid, [80, 1, 1]);
    }

    #[test]
    fn work_counts() {
        let w = node("pp512").work();
        assert_eq!(w.flops, 512.0 * 5120.0 * 12.0);
        assert_eq!(w.bytes, 512.0 * 5120.0 * 8.0 + 5120.0 * 16.0);
    }

    #[test]
    fn kernarg_offsets_follow_the_incumbent_signature() {
        // 5 pointers (0..40) then 11 i32 (40..84): the order of the incumbent's declaration.
        let n = node("emu");
        let src = n.incumbent().unwrap().source;
        let sig = &src[src.find(SYMBOL).unwrap()..];
        let sig = &sig[..sig.find(") {").unwrap()];
        let names: Vec<&str> = sig
            .split(',')
            .map(|p| {
                p.trim()
                    .rsplit([' ', '*'])
                    .next()
                    .unwrap()
                    .trim_end_matches('(')
            })
            .collect();
        let want = [
            "out",
            "in",
            "weight",
            "hist_in",
            "hist_out",
            "n_rows",
            "n_channels",
            "ksize",
            "rows_per_chunk",
            "in_row_stride",
            "out_row_stride",
            "w_ch_stride",
            "w_tap_stride",
            "hist_tap_stride",
            "apply_silu",
            "round_bf16",
        ];
        assert_eq!(names.len(), want.len(), "{names:?}");
        for (a, b) in names.iter().zip(want) {
            assert!(a.ends_with(b), "{a} vs {b}");
        }
        let c = n.emu_case(1).unwrap();
        assert_eq!(c.args.len(), 16);
        let ints: Vec<i32> = c.args[5..]
            .iter()
            .map(|a| match a {
                EmuArg::I32(v) => *v,
                o => panic!("{o:?}"),
            })
            .collect();
        assert_eq!(ints, vec![37, 200, 4, 37, 200, 200, 4, 1, 200, 1, 1]);
        // in place: hist_in and hist_out are the same packed-region pointer
        assert_eq!(c.args[3], c.args[4]);
        assert_eq!(c.args[0], EmuArg::BufOff(0, 0));
        let n0 = node("emu,inplace=0").emu_case(1).unwrap();
        assert_eq!(n0.args[3], EmuArg::Buf(3));
        assert_ne!(n0.args[3], n0.args[4]);
    }

    #[test]
    fn owner_map_covers_out_and_history_and_pads() {
        let n = node("emu");
        let own = n.out_owner(&Header::default()).unwrap();
        let (nc, hoff) = (37 * 200, 37 * 200 + PAD_WORDS);
        assert_eq!(own(0), [0, 0, 0]);
        assert_eq!(own(36 * 200 + 199), [1, 0, 0]);
        assert_eq!(own(nc), PAD_OWNER);
        assert_eq!(own(hoff + 2 * 200 + 130), [1, 0, 0]);
        assert_eq!(own(hoff + 3 * 200), PAD_OWNER);
        let n = node("pp512,inplace=0");
        let h = Header::parse("// hip_loop: ch_per_wg=256 rows_per_wg=128").unwrap();
        let own = n.out_owner(&h).unwrap();
        assert_eq!(own(130 * 5120 + 300), [1, 1, 0]);
        // history belongs to the last row slice
        assert_eq!(own(n.hist_off() + 5120 + 5119), [19, 3, 0]);
        let c = n.emu_case(1).unwrap();
        assert_eq!(
            c.bufs[0].bytes.len(),
            (512 * 5120 + PAD_WORDS + 3 * 5120 + PAD_WORDS) * 4
        );
    }

    #[test]
    fn extents_carry_the_arch() {
        assert_eq!(
            node("emu").extent_shapes(),
            ["pp512", "n=4096,c=200", "pp512,inplace=0"]
        );
        assert!(node("emu,arch=gfx1201")
            .extent_shapes()
            .iter()
            .all(|s| s.ends_with(",arch=gfx1201")));
    }
}
