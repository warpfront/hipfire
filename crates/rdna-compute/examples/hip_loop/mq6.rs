// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `mq6_res`: the gfx1100 MQ6 trunk residual GEMM
//! `gemm_mq6g256v2_residual_wmma_gfx11_mw{4,8}_lds` of the Flash-Next prefill
//! window (1,770 ms, 21% of the window, PF-1).
//!
//! Production launch (`gemm.rs` `Gpu::gemm_mqv2_residual_wmma_gfx11_mw_lds`,
//! reached from `try_gemm_mqv2_residual_gfx11_reuse` when `mqv2_mw_waves`
//! admits MW: gfx1100, bits 5|6, residual, N >= 464 -> MW8, 416..=463 -> MW4):
//! module `gemm_mqv2_wmma_gfx1100_mw_lds`, source
//! `kernels::GEMM_MQV2_WMMA_GFX11_MW_LDS_SRC`; kernargs `(A, X_f16, Y_f32, M, K,
//! N)`; grid `[ceil(M/16), ceil(N/(16*NW)), 1]` workgroups, block `32*NW`,
//! dynamic LDS 0 (static 8 KiB tile). `X` is the f16 copy of the activations
//! (`ensure_fp16_x`, `[N][K]`), `Y` is the f32 residual `[N][M]` that the
//! kernel accumulates into, `A` is MQ6G256V2 (`MQ6G256V2_GROUP_BYTES` = 200 B
//! per 256 weights: two f16 (scale, zero) pairs, then 192 B of LSB-first 6-bit codes).
//!
//! Shapes (N = 768 tokens): `qkv` M=10240 K=2560, `z` M=6144 K=2560, `out`
//! M=2560 K=6144, `qg` M=12288 K=2560, `ba` M=48 K=2560. `--shape
//! qkv,N=1536` or `M=..,K=..,N=..` override; `waves=4|8` forces MW.
//!
//! Candidate header (`// hip_loop: ...`): `tile_m` (rows per WG, 16), `tile_n`
//! (tokens per WG, `16*NW`), `block` (`32*NW`), `lds_dynamic` (0).
//!
//! Included with `#[path = "hip_loop/mq6.rs"] mod mq6;`.
#![allow(dead_code)]

use crate::node::*;
use serde_json::{json, Value};

/// Presets: `(name, M, K)`.
pub const PRESETS: [(&str, usize, usize); 5] = [
    ("qkv", 10240, 2560),
    ("z", 6144, 2560),
    ("out", 2560, 6144),
    ("qg", 12288, 2560),
    ("ba", 48, 2560),
];
pub const DEFAULT_N: usize = 768;
/// The pre-card emulator gate's shape (`--emu`, preset `emu`): M=144 (rows tile 16..=128 all leave a
/// ragged last tile: 144 = 128 + 16), K=512 (two MQ6 groups), N=40 (ragged under every token tile,
/// 16*NW .. 256), forced MW8 (production routes N < 416 off this kernel).
pub const EMU: (usize, usize, usize) = (144, 512, 40);
const MODULE: &str = "gemm_mqv2_wmma_gfx1100_mw_lds";
const GROUP_BYTES: usize = rdna_compute::MQ6G256V2_GROUP_BYTES;

/// `mqv2_mw_waves(gfx1100, 6, Residual, n)`.
pub fn production_waves(n: usize) -> Option<usize> {
    match n {
        464.. => Some(8),
        416..=463 => Some(4),
        _ => None,
    }
}

struct State {
    w: Buf,
    x: Buf,
    y: Buf,
    y0: Buf,
}

pub struct Mq6Res {
    label: String,
    m: usize,
    k: usize,
    n: usize,
    waves: usize,
    st: Option<State>,
}

impl Mq6Res {
    pub fn new(shape: &ShapeSpec) -> Result<Self> {
        let mut s = shape.clone();
        let emu = s.preset.as_deref() == Some("emu");
        let (label, pm, pk) = match &s.preset {
            Some(_) if emu => ("emu".to_owned(), Some(EMU.0), Some(EMU.1)),
            Some(p) => {
                let (_, m, k) = PRESETS.iter().find(|(n, _, _)| n == p).ok_or_else(|| {
                    let names: Vec<&str> = PRESETS.iter().map(|p| p.0).collect();
                    format!(
                        "mq6_res: unknown shape preset `{p}` (have {}, emu)",
                        names.join(", ")
                    )
                })?;
                (p.clone(), Some(*m), Some(*k))
            }
            None => ("custom".to_owned(), None, None),
        };
        let m = s.take_usize("M", pm)?;
        let k = s.take_usize("K", pk)?;
        let n = s.take_usize("N", Some(if emu { EMU.2 } else { DEFAULT_N }))?;
        let waves = match s.kv.remove("waves") {
            Some(w) => match w.as_str() {
                "4" => 4,
                "8" => 8,
                _ => return Err(format!("mq6_res: waves={w}: must be 4 or 8")),
            },
            None if emu => 8,
            None => production_waves(n).ok_or_else(|| {
                format!("mq6_res: N={n}: production does not route MW below N=416 (pass waves=4|8 to force)")
            })?,
        };
        s.no_leftover()?;
        if m == 0 || n == 0 || k == 0 || k % 256 != 0 {
            return Err(format!(
                "mq6_res: M={m} N={n} K={k}: need positive dims and K % 256 == 0"
            ));
        }
        Ok(Self {
            label,
            m,
            k,
            n,
            waves,
            st: None,
        })
    }

    fn st(&self) -> Result<&State> {
        self.st
            .as_ref()
            .ok_or_else(|| "mq6_res: setup has not run".to_owned())
    }

    /// `m x (k/256)` MQ6G256V2 groups: finite small f16 scales, zero points
    /// near `-32 * scale` (centred codes), random 6-bit codes.
    pub fn synth_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut r = Xs64::new(seed);
        let mut w = vec![0u8; m * (k / 256) * GROUP_BYTES];
        for g in w.chunks_exact_mut(GROUP_BYTES) {
            for h in 0..2 {
                let s = half::f16::from_f32(1e-4 + r.normal().abs() * 1e-3);
                let sf = s.to_f32();
                g[4 * h..4 * h + 2].copy_from_slice(&s.to_bits().to_le_bytes());
                let z = f16_bits(-32.0 * sf + r.normal() * sf);
                g[4 * h + 2..4 * h + 4].copy_from_slice(&z.to_le_bytes());
            }
            for c in g[8..].chunks_mut(8) {
                let v = r.next_u64().to_le_bytes();
                c.copy_from_slice(&v[..c.len()]);
            }
        }
        w
    }

    /// Seeded host buffers `(W, X, Y0)` the GPU `setup` uploads and the emulator gate maps.
    pub fn host_inputs(&self, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let w = Self::synth_weights(self.m, self.k, seed);
        let mut r = Xs64::new(seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut x = Vec::with_capacity(self.n * self.k * 2);
        for _ in 0..self.n * self.k {
            x.extend_from_slice(&f16_bits(r.normal()).to_le_bytes());
        }
        let mut y = Vec::with_capacity(self.n * self.m * 4);
        for _ in 0..self.n * self.m {
            y.extend_from_slice(&(0.5 * r.unit()).to_le_bytes());
        }
        (w, x, y)
    }
}

impl Node for Mq6Res {
    fn name(&self) -> &'static str {
        "mq6_res"
    }
    fn shape_label(&self) -> String {
        self.label.clone()
    }
    fn shape_json(&self) -> Value {
        json!({"M": self.m, "K": self.k, "N": self.n, "waves": self.waves,
            "weights": "MQ6G256V2", "group_bytes": GROUP_BYTES,
            "x": "f16 [N][K]", "y": "f32 [N][M] residual (accumulated)"})
    }
    fn incumbent(&self) -> Result<Incumbent> {
        Ok(Incumbent {
            module: MODULE.to_owned(),
            source: rdna_compute::kernels::GEMM_MQV2_WMMA_GFX11_MW_LDS_SRC.to_owned(),
            symbol: format!("gemm_mq6g256v2_residual_wmma_gfx11_mw{}_lds", self.waves),
            extra_flags: String::new(),
            selection: "crates/rdna-compute/src/gemm.rs Gpu::gemm_mqv2_residual_wmma_gfx11_mw_lds \
                        (try_gemm_mqv2_residual_gfx11_reuse, mqv2_mw_waves)"
                .to_owned(),
        })
    }
    fn geometry_with(&self, hdr: &mut Header) -> Result<Geometry> {
        tiled_geometry(hdr, self.m, self.n, 16, 16 * self.waves, 32 * self.waves)
    }
    fn work(&self) -> Work {
        let (m, k, n) = (self.m as f64, self.k as f64, self.n as f64);
        Work {
            flops: 2.0 * m * k * n,
            bytes: m * (k / 256.0) * GROUP_BYTES as f64 + n * k * 2.0 + n * m * 4.0 * 2.0,
        }
    }

    fn extent_shapes(&self) -> Vec<String> {
        PRESETS.iter().map(|p| p.0.to_owned()).collect()
    }
    fn out_owner<'a>(&'a self, hdr: &Header) -> Result<Box<dyn Fn(usize) -> [u32; 3] + 'a>> {
        // Y is f32 [N][M]; the workgroup is (row tile, token tile).
        let (tm, tn) = header_tiles(hdr, 16, 16 * self.waves)?;
        let m = self.m;
        Ok(Box::new(move |e| {
            [(e % m / tm) as u32, (e / m / tn) as u32, 0]
        }))
    }
    fn emu_case(&self, seed: u64) -> Result<EmuCase> {
        let (w, x, y) = self.host_inputs(seed);
        Ok(EmuCase {
            bufs: vec![
                EmuBuf {
                    name: "a",
                    bytes: w,
                },
                EmuBuf {
                    name: "x",
                    bytes: x,
                },
                EmuBuf {
                    name: "y",
                    bytes: y,
                },
            ],
            args: vec![
                EmuArg::Buf(0),
                EmuArg::Buf(1),
                EmuArg::Buf(2),
                EmuArg::I32(self.m as i32),
                EmuArg::I32(self.k as i32),
                EmuArg::I32(self.n as i32),
            ],
            out: 2,
        })
    }

    fn setup(&mut self, gpu: &rdna_compute::Gpu, seed: u64) -> Result<()> {
        let (w, x, y) = self.host_inputs(seed);
        let ybuf = Buf::from_bytes(gpu, &y)?;
        let y0 = Buf::from_bytes(gpu, &y)?;
        self.st = Some(State {
            w: Buf::from_bytes(gpu, &w)?,
            x: Buf::from_bytes(gpu, &x)?,
            y: ybuf,
            y0,
        });
        Ok(())
    }
    fn reset(&mut self, gpu: &rdna_compute::Gpu) -> Result<()> {
        let s = self.st()?;
        s.y.copy_from(gpu, &s.y0)
    }
    fn kernarg(&self) -> Result<Vec<u8>> {
        let s = self.st()?;
        let mut a = Args::new();
        a.ptr(&s.w)
            .ptr(&s.x)
            .ptr(&s.y)
            .i32(self.m as i32)
            .i32(self.k as i32)
            .i32(self.n as i32);
        Ok(a.finish())
    }
    fn read_output(&self, gpu: &rdna_compute::Gpu) -> Result<Vec<u8>> {
        self.st()?.y.download(gpu)
    }
    fn teardown(&mut self, gpu: &rdna_compute::Gpu) {
        if let Some(s) = self.st.take() {
            s.w.free(gpu);
            s.x.free(gpu);
            s.y.free(gpu);
            s.y0.free(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PF-1 (`xtx-gap/screens/PF/BUCKETS.md`) profiler grids (`grid_size`, work-items).
    const PF1: [(&str, [u64; 3]); 5] = [
        ("qkv", [163840, 6, 1]),
        ("z", [98304, 6, 1]),
        ("out", [40960, 6, 1]),
        ("qg", [196608, 6, 1]),
        ("ba", [768, 6, 1]),
    ];

    fn node(shape: &str) -> Mq6Res {
        Mq6Res::new(&ShapeSpec::parse(Some(shape)).unwrap()).unwrap()
    }

    #[test]
    fn production_grids_match_pf1() {
        for (name, grid) in PF1 {
            let n = node(name);
            let g = n.geometry();
            assert_eq!(g.block, [256, 1, 1], "{name}");
            assert_eq!(g.lds, 0, "{name}");
            // The profiler's grid_size is workgroups x block on the x axis only
            // (rocprof reports y/z in work-items too, and block y/z are 1).
            assert_eq!(g.threads(), grid, "{name}");
            assert_eq!(
                n.incumbent().unwrap().symbol,
                "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds"
            );
        }
    }

    #[test]
    fn waves_follow_the_production_policy() {
        assert_eq!(
            node("qkv,N=416")
                .incumbent()
                .unwrap()
                .symbol
                .contains("mw4"),
            true
        );
        assert_eq!(node("qkv,N=463").waves, 4);
        assert_eq!(node("qkv,N=464").waves, 8);
        assert!(Mq6Res::new(&ShapeSpec::parse(Some("qkv,N=128")).unwrap()).is_err());
        assert_eq!(node("qkv,N=128,waves=8").waves, 8);
    }

    #[test]
    fn header_overrides_geometry_and_refuses_bad_values() {
        let n = node("qkv");
        let mut h = Header::parse("// hip_loop: tile_m=32 tile_n=128 block=256\n").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!(g.grid, [320, 6, 1]);
        let mut h =
            Header::parse("// hip_loop: tile_m=16 tile_n=256 block=512 lds_dynamic=1024").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!((g.grid, g.block, g.lds), ([640, 3, 1], [512, 1, 1], 1024));
        for bad in [
            "tile_m=0",
            "block=100",
            "bogus=1",
            "lds_dynamic=999999",
            "tile_n=abc",
        ] {
            let mut h = Header::parse(&format!("// hip_loop: {bad}")).unwrap();
            assert!(n.geometry_with(&mut h).is_err(), "{bad}");
        }
    }

    #[test]
    fn synthetic_weights_are_valid_and_deterministic() {
        let w = Mq6Res::synth_weights(48, 2560, 7);
        assert_eq!(w.len(), 48 * 10 * 200);
        assert_eq!(w, Mq6Res::synth_weights(48, 2560, 7));
        assert_ne!(w, Mq6Res::synth_weights(48, 2560, 8));
        for g in w.chunks_exact(200) {
            for h in 0..2 {
                let s = half::f16::from_le_bytes([g[4 * h], g[4 * h + 1]]).to_f32();
                let z = half::f16::from_le_bytes([g[4 * h + 2], g[4 * h + 3]]).to_f32();
                assert!(
                    s.is_finite() && s > 0.0 && s < 0.02 && z.is_finite(),
                    "{s} {z}"
                );
            }
        }
    }

    #[test]
    fn work_counts_match_production_bytes() {
        let w = node("qkv").work();
        assert_eq!(w.flops, 2.0 * 10240.0 * 2560.0 * 768.0);
        assert_eq!(
            w.bytes,
            (10240 * 10 * 200 + 768 * 2560 * 2 + 768 * 10240 * 8) as f64
        );
    }
}
