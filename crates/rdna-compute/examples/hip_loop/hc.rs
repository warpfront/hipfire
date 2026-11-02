// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `hc_mix`: `gemm_wmma_lds_64_64_32_64_k64_p` at the Flash-Next
//! hyper-connection mix shape (HC bucket of the PF-1 window, 385 ms for this
//! kernel; grid `(320, 12, 1)` in profiler work-items).
//!
//! Production launch: the HC `input_mix_down` projection (M = 320, K = 10240,
//! N = 768 tokens) goes `Gpu::gemm_bf16_xf16_f16_wmma` (`gemm.rs`; on gfx1100
//! `hc_down_tile_selected` is false, so `m < 512` takes `LdsTile::new(64, 64,
//! 32, 64, 64, false).pipelined()`) -> `gemm_f16_x_f16_wmma_lds_tiled_ld`, which
//! launches `gemm_wmma_lds_64_64_32_64_k64_p` from module
//! `qwen4_gemm_wmma_lds256` (`lds256_tile_source_selected`: the 64x64 k64
//! tiles on a non-gfx1151 arch come from
//! `kernels::QWEN4_GEMM_F16_X_F16_WMMA_LDS256_SRC`) with kernargs `(A_f16, X_f16,
//! Y_f32, Bias = null, M, K, N, has_bias = 0, lda = K, ldx = K)`; grid
//! `[ceil(M/64), ceil(N/64), 1]` workgroups, block `(64/32) * (64/64) * 32 =
//! 64`, dynamic LDS 0. `A` is the f16 shadow of the BF16 HC weight `[M][K]`,
//! `X` the f16 activations `[N][K]`, `Y` f32 `[N][M]` (overwritten, EPI = 0).
//!
//! Shapes: `mix` (default) M=320 K=10240; `sel` M=1 K=2560 (the shared-expert
//! selector logit, same entry, grid X 64); `M=..,K=..,N=..` override
//! (`K % 64 == 0`, N default 768).
//!
//! Candidate header (`// hip_loop: ...`): `tile_m` (rows per WG, 64), `tile_n`
//! (tokens per WG, 64), `block` (64), `lds_dynamic` (0).
//!
//! Included with `#[path = "hip_loop/hc.rs"] mod hc;`.
#![allow(dead_code)]

use crate::node::*;
use serde_json::{json, Value};

pub const PRESETS: [(&str, usize, usize); 2] = [("mix", 320, 10240), ("sel", 1, 2560)];
pub const DEFAULT_N: usize = 768;
/// The pre-card emulator gate's shape (`--emu`, preset `emu`): `(M, K, N)` = (144, 256, 24).
/// M = 144 leaves a ragged last tile under the 64 production tile (64+64+16) and the 80-row
/// candidate tiles (80+64); N = 24 is ragged under every 32/64-token tile; K = 256 is four k64 steps.
pub const EMU: (usize, usize, usize) = (144, 256, 24);
pub const SYMBOL: &str = "gemm_wmma_lds_64_64_32_64_k64_p";
const MODULE: &str = "qwen4_gemm_wmma_lds256";
const TILE: usize = 64;
/// `(64/32) * (64/64)` waves of 32 lanes.
const BLOCK: usize = 64;

struct State {
    a: Buf,
    x: Buf,
    y: Buf,
    /// Pristine sentinel-filled output.
    y0: Buf,
}

pub struct HcMix {
    label: String,
    m: usize,
    k: usize,
    n: usize,
    st: Option<State>,
}

const SENTINEL: f32 = -1234.5;

impl HcMix {
    pub fn new(shape: &ShapeSpec) -> Result<Self> {
        let mut s = shape.clone();
        let preset = s.preset.clone().unwrap_or_else(|| "mix".to_owned());
        let emu = preset == "emu";
        let (label, pm, pk) = match PRESETS.iter().find(|(n, _, _)| *n == preset) {
            Some((n, m, k)) => ((*n).to_owned(), Some(*m), Some(*k)),
            None if emu => ("emu".to_owned(), Some(EMU.0), Some(EMU.1)),
            None => {
                return Err(format!(
                    "hc_mix: unknown shape preset `{preset}` (have mix, sel, emu)"
                ));
            }
        };
        let m = s.take_usize("M", pm)?;
        let k = s.take_usize("K", pk)?;
        let n = s.take_usize("N", Some(if emu { EMU.2 } else { DEFAULT_N }))?;
        s.no_leftover()?;
        if m == 0 || n == 0 || k == 0 || k % 64 != 0 {
            return Err(format!(
                "hc_mix: M={m} N={n} K={k}: need positive dims and K % 64 == 0"
            ));
        }
        Ok(Self {
            label,
            m,
            k,
            n,
            st: None,
        })
    }

    fn st(&self) -> Result<&State> {
        self.st
            .as_ref()
            .ok_or_else(|| "hc_mix: setup has not run".to_owned())
    }

    /// Seeded host buffers `(A, X, Y0)` the GPU `setup` uploads and the emulator gate maps.
    pub fn host_inputs(&self, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut r = Xs64::new(seed);
        let amp = 1.0 / (self.k as f32).sqrt();
        let mut a = Vec::with_capacity(self.m * self.k * 2);
        for _ in 0..self.m * self.k {
            a.extend_from_slice(&f16_bits(amp * r.normal()).to_le_bytes());
        }
        let mut x = Vec::with_capacity(self.n * self.k * 2);
        for _ in 0..self.n * self.k {
            x.extend_from_slice(&f16_bits(r.normal()).to_le_bytes());
        }
        let y: Vec<u8> = (0..self.n * self.m)
            .flat_map(|_| SENTINEL.to_le_bytes())
            .collect();
        (a, x, y)
    }
}

impl Node for HcMix {
    fn name(&self) -> &'static str {
        "hc_mix"
    }
    fn shape_label(&self) -> String {
        self.label.clone()
    }
    fn shape_json(&self) -> Value {
        json!({"M": self.m, "K": self.k, "N": self.n, "tile": "64x64 wave 32x64 k64 pipelined",
            "a": "f16 [M][K]", "x": "f16 [N][K]", "y": "f32 [N][M] (overwritten)"})
    }
    fn incumbent(&self) -> Result<Incumbent> {
        Ok(Incumbent {
            module: MODULE.to_owned(),
            source: rdna_compute::kernels::QWEN4_GEMM_F16_X_F16_WMMA_LDS256_SRC.to_owned(),
            symbol: SYMBOL.to_owned(),
            extra_flags: String::new(),
            selection: "crates/rdna-compute/src/gemm.rs Gpu::gemm_bf16_xf16_f16_wmma -> \
                        gemm_f16_x_f16_wmma_lds_tiled_ld (lds256_tile_source_selected)"
                .to_owned(),
        })
    }
    fn geometry_with(&self, hdr: &mut Header) -> Result<Geometry> {
        tiled_geometry(hdr, self.m, self.n, TILE, TILE, BLOCK)
    }
    fn work(&self) -> Work {
        let (m, k, n) = (self.m as f64, self.k as f64, self.n as f64);
        Work {
            flops: 2.0 * m * k * n,
            bytes: (m * k + n * k) * 2.0 + n * m * 4.0,
        }
    }

    fn extent_shapes(&self) -> Vec<String> {
        PRESETS.iter().map(|p| p.0.to_owned()).collect()
    }
    fn out_owner<'a>(&'a self, hdr: &Header) -> Result<Box<dyn Fn(usize) -> [u32; 3] + 'a>> {
        // Y is f32 [N][M]; the workgroup is (row tile, token tile).
        let (tm, tn) = header_tiles(hdr, TILE, TILE)?;
        let m = self.m;
        Ok(Box::new(move |e| {
            [(e % m / tm) as u32, (e / m / tn) as u32, 0]
        }))
    }
    fn emu_case(&self, seed: u64) -> Result<EmuCase> {
        let (a, x, y) = self.host_inputs(seed);
        Ok(EmuCase {
            bufs: vec![
                EmuBuf {
                    name: "a",
                    bytes: a,
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
                EmuArg::Null,
                EmuArg::I32(self.m as i32),
                EmuArg::I32(self.k as i32),
                EmuArg::I32(self.n as i32),
                EmuArg::I32(0),
                EmuArg::I32(self.k as i32),
                EmuArg::I32(self.k as i32),
            ],
            out: 2,
        })
    }

    fn setup(&mut self, gpu: &rdna_compute::Gpu, seed: u64) -> Result<()> {
        let (a, x, y) = self.host_inputs(seed);
        self.st = Some(State {
            a: Buf::from_bytes(gpu, &a)?,
            x: Buf::from_bytes(gpu, &x)?,
            y: Buf::from_bytes(gpu, &y)?,
            y0: Buf::from_bytes(gpu, &y)?,
        });
        Ok(())
    }
    fn reset(&mut self, gpu: &rdna_compute::Gpu) -> Result<()> {
        // EPI = 0 overwrites every element; the sentinel makes "the kernel
        // wrote the output" observable and stale output from the other arm visible.
        let s = self.st()?;
        s.y.copy_from(gpu, &s.y0)
    }
    fn kernarg(&self) -> Result<Vec<u8>> {
        let s = self.st()?;
        let mut a = Args::new();
        a.ptr(&s.a)
            .ptr(&s.x)
            .ptr(&s.y)
            .raw_ptr(0)
            .i32(self.m as i32)
            .i32(self.k as i32)
            .i32(self.n as i32)
            .i32(0)
            .i32(self.k as i32)
            .i32(self.k as i32);
        Ok(a.finish())
    }
    fn read_output(&self, gpu: &rdna_compute::Gpu) -> Result<Vec<u8>> {
        self.st()?.y.download(gpu)
    }
    fn teardown(&mut self, gpu: &rdna_compute::Gpu) {
        if let Some(s) = self.st.take() {
            s.a.free(gpu);
            s.x.free(gpu);
            s.y.free(gpu);
            s.y0.free(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(shape: &str) -> HcMix {
        HcMix::new(&ShapeSpec::parse(Some(shape)).unwrap()).unwrap()
    }

    #[test]
    fn production_grids_match_pf1() {
        // HC input_mix_down: PF-1 grid (320, 12, 1); selector logit: grid X 64.
        let g = node("mix").geometry();
        assert_eq!(g.block, [64, 1, 1]);
        assert_eq!(g.lds, 0);
        assert_eq!(g.threads(), [320, 12, 1]);
        assert_eq!(node("sel").geometry().threads(), [64, 12, 1]);
        assert_eq!(
            HcMix::new(&ShapeSpec::parse(None).unwrap()).unwrap().label,
            "mix"
        );
    }

    #[test]
    fn kernarg_layout_is_the_production_blob() {
        // (A, X, Y, Bias, M, K, B, has_bias, lda, ldx): 4 pointers + 6 i32.
        let mut a = Args::new();
        a.raw_ptr(1).raw_ptr(2).raw_ptr(3).raw_ptr(0);
        for v in [320, 10240, 768, 0, 10240, 10240] {
            a.i32(v);
        }
        assert_eq!(a.finish().len(), 56);
    }

    #[test]
    fn header_geometry() {
        let n = node("mix");
        let mut h = Header::parse("// hip_loop: tile_m=128 tile_n=64 block=128").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!((g.grid, g.block), ([3, 12, 1], [128, 1, 1]));
        let mut h = Header::parse("// hip_loop: waves=4").unwrap();
        assert!(n.geometry_with(&mut h).is_err());
    }

    #[test]
    fn rejects_bad_shapes() {
        for s in ["mix,K=100", "nope", "mix,Q=3"] {
            assert!(
                HcMix::new(&ShapeSpec::parse(Some(s)).unwrap()).is_err(),
                "{s}"
            );
        }
    }
}
