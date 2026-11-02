// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LAB ONLY: gfx1201 byte oracle for the runtime-head-generalized production
//! GDN chunk route (`gdn_chunk_prep`, `gdn_chunk_prep_fixup`,
//! `gdn_chunk_kkt_solve_batched`, legacy per-segment KKT,
//! `gdn_chunk_scan_layer_mseg`, `gdn_chunk_scan_segment`).
//!
//! Arms (each against the shape-parameterised lab kernels in `lab_kernels/`):
//!   * production at n_key_heads=16, n_value_heads=48  vs  lab built with defaults;
//!   * production at n_key_heads=16, n_value_heads=32  vs  lab built with the A3B defines
//!     `-DGDN_LAB_HK=16 -DGDN_LAB_HV=32 -DGDN_LAB_RATIO=2 -DGDN_LAB_XDIM=8192 -DGDN_LAB_UNITS=64`.
//!
//! Every case (hv, rows in {64,512,4096}, zero / nonzero initial conv+Q8+scale
//! state) runs six stages on both arms from identical deterministic inputs and
//! compares EVERY plane byte-for-byte after each stage: q, k, v, A, g/alpha, beta,
//! conv state, Q8 state, scales, EF, out. Stages:
//!   prep                         gdn_chunk_prep
//!   prep_fixup                   prep, restore state, gdn_chunk_prep_fixup over prep's planes
//!   kkt_batched                  prep + gdn_chunk_kkt_solve_batched
//!   scan_mseg                    prep + batched KKT + gdn_chunk_scan_layer_mseg
//!   scan_segments_batched_kkt    prep + batched KKT + 512-row gdn_chunk_scan_segment (kkt=false, A row view)
//!   scan_segments_legacy_kkt     prep + 512-row gdn_chunk_scan_segment (kkt=true, legacy KKT per segment)
//! A stage on which a lab plane that must be nonzero is all zero is an error
//! (vacuous comparison), not a pass. Production 16/32 is then timed (10 warmups,
//! 50 synchronized reps, hipEvent) for prep -> batched KKT -> mseg scan.
//!
//! The lab kernels are compiled with the production module flags
//! (-ffp-contract=off; -mcumode for scan), mirroring `KernelCompiler::module_flags_for`.
//!
//! Build: cargo build --release -j 8 -p rdna-compute --features lab,deltanet \
//!          --example gdn_chunk_generalized_oracle
//! Run (GPU hold): HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=<gfx1201 device> \
//!          <target-dir>/release/examples/gdn_chunk_generalized_oracle > gdn-chunk-generalized.jsonl
//! Output: one JSON line per case (`"record":"case"`), then one `"record":"summary"`
//! line. Exit status 0 only if every plane of every stage of every case is byte identical.

use hip_bridge::{Function, KernargBlob, Module};
use rdna_compute::norm::GdnScanOut;
use rdna_compute::{DType, Gpu, GpuTensor, KernelCompiler};
use serde_json::{json, Value};

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

const HD: usize = 128;
const CONV_TAPS: usize = 4;
const SEGMENT_ROWS: usize = 512;
const WARMUPS: usize = 10;
const REPS: usize = 50;
const EPS: f32 = 1e-6;
const A3B_DEFINES: &str =
    "-DGDN_LAB_HK=16 -DGDN_LAB_HV=32 -DGDN_LAB_RATIO=2 -DGDN_LAB_XDIM=8192 -DGDN_LAB_UNITS=64";

const LAB_PREP: &str = include_str!("lab_kernels/gdn_chunk_scan_prep.gfx1201.hip");
const LAB_FIXUP: &str = include_str!("lab_kernels/gdn_chunk_prep_fixup.gfx1201.hip");
const LAB_KKT: &str = include_str!("lab_kernels/gdn_chunk_scan_kkt_solve.gfx1201.hip");
const LAB_SCAN: &str = include_str!("lab_kernels/gdn_chunk_scan.gfx1201.hip");

fn q_scale() -> f32 {
    1.0 / (HD as f32).sqrt()
}

#[derive(Clone, Copy)]
struct Shape {
    hk: usize,
    hv: usize,
}

impl Shape {
    fn new(hk: usize, hv: usize) -> Result<Self> {
        if !matches!((hk, hv), (16, 48) | (16, 32)) {
            return Err(format!("unsupported oracle shape hk={hk} hv={hv}").into());
        }
        Ok(Self { hk, hv })
    }
    fn units(&self) -> usize {
        2 * self.hk + self.hv
    }
    fn xdim(&self) -> usize {
        self.units() * HD
    }
    /// Lab compile defines: lab defaults reproduce 16/48; A3B defines for 16/32.
    fn lab_defines(&self) -> &'static str {
        if self.hv == 48 {
            ""
        } else {
            A3B_DEFINES
        }
    }
}

struct Kernel {
    _module: Module,
    function: Function,
}

fn compile(
    gpu: &Gpu,
    name: &str,
    source: &str,
    symbol: &str,
    flags: &str,
    module_flags: &str,
) -> Result<Kernel> {
    let flags = format!("{} {module_flags} {flags}", gpu.flags.hipcc_extra_flags);
    let mut compiler = KernelCompiler::new(&gpu.arch, flags)?;
    let path = compiler
        .compile_for_symbol(name, source, symbol)?
        .to_path_buf();
    let module = gpu
        .hip
        .module_load(path.to_str().ok_or("non-UTF8 kernel path")?)?;
    let function = gpu.hip.module_get_function(&module, symbol)?;
    Ok(Kernel {
        _module: module,
        function,
    })
}

fn launch(
    gpu: &Gpu,
    kernel: &Kernel,
    grid: [u32; 3],
    block: [u32; 3],
    mut args: KernargBlob,
) -> Result<()> {
    // SAFETY: every caller below reproduces the production kernel's complete ABI
    // with live tensors, matching dtype, validated shape, and no graph capture.
    unsafe {
        gpu.hip
            .launch_kernel_blob(&kernel.function, grid, block, 0, None, args.as_mut_slice())?;
    }
    Ok(())
}

fn raw(gpu: &mut Gpu, bytes: usize) -> Result<GpuTensor> {
    Ok(gpu.zeros(&[bytes], DType::Raw)?)
}

fn upload(gpu: &mut Gpu, data: &[f32]) -> Result<GpuTensor> {
    Ok(gpu.upload_f32(data, &[data.len()])?)
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<u8>> {
    let mut data = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut data, &t.buf)?;
    Ok(data)
}

// ---------------------------------------------------------------------------
// Deterministic inputs (same generator and ranges as lab_kernels/gdn_incumbent.rs).
// ---------------------------------------------------------------------------

const SQRT3: f32 = 1.732_050_8;

fn lcg(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 40) as u32) as f32 / (1u32 << 24) as f32;
            lo + (hi - lo) * u
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

struct Inputs {
    shape: Shape,
    rows: usize,
    qkv: GpuTensor,
    beta_raw: GpuTensor,
    alpha_raw: GpuTensor,
    conv_w: GpuTensor,
    dt_bias: GpuTensor,
    a_log: GpuTensor,
    conv0: Vec<u8>,
    q80: Vec<u8>,
    scales0: Vec<u8>,
    ef0: Vec<u8>,
}

impl Inputs {
    fn new(gpu: &mut Gpu, shape: Shape, rows: usize, nonzero: bool) -> Result<Self> {
        let hv = shape.hv;
        let xdim = shape.xdim();
        let qkv = upload(gpu, &lcg(1, rows * xdim, -SQRT3, SQRT3))?;
        let beta_raw = upload(gpu, &lcg(3, rows * hv, -2.0, 2.0))?;
        let alpha_raw = upload(gpu, &lcg(4, rows * hv, -SQRT3, SQRT3))?;
        let conv_w = upload(gpu, &lcg(5, CONV_TAPS * xdim, -0.5, 0.5))?;
        let dt_bias = upload(gpu, &lcg(6, hv, -4.0, -2.0))?;
        let a_log = upload(gpu, &lcg(7, hv, 0.0, 2.0))?;
        let conv0 = if nonzero {
            f32_bytes(&lcg(9, 3 * xdim, -0.5, 0.5))
        } else {
            vec![0u8; 3 * xdim * 4]
        };
        let q80 = if nonzero {
            lcg(10, hv * HD * HD, -64.0, 64.0)
                .iter()
                .map(|x| *x as i8 as u8)
                .collect()
        } else {
            vec![0u8; hv * HD * HD]
        };
        let scales0 = if nonzero {
            f32_bytes(&lcg(11, hv * HD, 0.005, 0.02))
        } else {
            vec![0u8; hv * HD * 4]
        };
        let ef0 = vec![0u8; hv * HD * HD * 2];
        Ok(Self {
            shape,
            rows,
            qkv,
            beta_raw,
            alpha_raw,
            conv_w,
            dt_bias,
            a_log,
            conv0,
            q80,
            scales0,
            ef0,
        })
    }

    fn free(self, gpu: &mut Gpu) -> Result<()> {
        for t in [
            self.qkv,
            self.beta_raw,
            self.alpha_raw,
            self.conv_w,
            self.dt_bias,
            self.a_log,
        ] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

/// Mutable state advanced by the route: conv ring, Q8 state, scales, EF, and the
/// in-place gated alpha(g)/beta work copies of the raw gates.
struct State {
    conv_state: GpuTensor,
    s_q8: GpuTensor,
    s_scales: GpuTensor,
    ef: GpuTensor,
    beta: GpuTensor,
    alpha: GpuTensor,
}

impl State {
    fn new(gpu: &mut Gpu, inp: &Inputs) -> Result<Self> {
        let (hv, rows, xdim) = (inp.shape.hv, inp.rows, inp.shape.xdim());
        let st = Self {
            conv_state: upload(gpu, &vec![0.0f32; 3 * xdim])?,
            s_q8: raw(gpu, hv * HD * HD)?,
            s_scales: upload(gpu, &vec![0.0f32; hv * HD])?,
            ef: raw(gpu, hv * HD * HD * 2)?,
            beta: upload(gpu, &vec![0.0f32; rows * hv])?,
            alpha: upload(gpu, &vec![0.0f32; rows * hv])?,
        };
        st.reset(gpu, inp)?;
        Ok(st)
    }

    /// Restore initial state + raw gates. Called outside every timed region.
    fn reset(&self, gpu: &Gpu, inp: &Inputs) -> Result<()> {
        let gate_bytes = inp.rows * inp.shape.hv * 4;
        gpu.memcpy_htod_auto(&self.conv_state.buf, &inp.conv0)?;
        gpu.memcpy_htod_auto(&self.s_q8.buf, &inp.q80)?;
        gpu.memcpy_htod_auto(&self.s_scales.buf, &inp.scales0)?;
        gpu.memcpy_htod_auto(&self.ef.buf, &inp.ef0)?;
        gpu.memcpy_dtod_auto(&self.beta.buf, &inp.beta_raw.buf, gate_bytes)?;
        gpu.memcpy_dtod_auto(&self.alpha.buf, &inp.alpha_raw.buf, gate_bytes)?;
        Ok(())
    }

    fn free(self, gpu: &mut Gpu) -> Result<()> {
        for t in [
            self.conv_state,
            self.s_q8,
            self.s_scales,
            self.ef,
            self.beta,
            self.alpha,
        ] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

/// One route instance's state plus its compact q/k/v planes, A blocks and bf16 out.
struct Arm {
    state: State,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    a: GpuTensor,
    out: GpuTensor,
}

impl Arm {
    fn new(gpu: &mut Gpu, inp: &Inputs) -> Result<Self> {
        let (hk, hv, rows) = (inp.shape.hk, inp.shape.hv, inp.rows);
        Ok(Self {
            state: State::new(gpu, inp)?,
            q: raw(gpu, rows * hk * HD * 2)?,
            k: raw(gpu, rows * hk * HD * 2)?,
            v: raw(gpu, rows * hv * HD * 2)?,
            a: raw(gpu, rows.div_ceil(64) * 64 * hv * 64 * 2)?,
            out: raw(gpu, rows * hv * HD * 2)?,
        })
    }

    /// Zero the compact planes, A and out (state/gates are restored by `State::reset`).
    fn clear_planes(&self, gpu: &Gpu) -> Result<()> {
        for t in [&self.q, &self.k, &self.v, &self.a, &self.out] {
            gpu.memcpy_htod_auto(&t.buf, &vec![0u8; t.byte_size()])?;
        }
        Ok(())
    }

    fn snapshot(&self, gpu: &Gpu) -> Result<Vec<(&'static str, Vec<u8>)>> {
        let mut result = Vec::new();
        for (name, t) in [
            ("q", &self.q),
            ("k", &self.k),
            ("v", &self.v),
            ("A", &self.a),
            ("alpha", &self.state.alpha),
            ("beta", &self.state.beta),
            ("conv_state", &self.state.conv_state),
            ("state_q8", &self.state.s_q8),
            ("state_scales", &self.state.s_scales),
            ("EF", &self.state.ef),
            ("out", &self.out),
        ] {
            result.push((name, download(gpu, t)?));
        }
        Ok(result)
    }

    fn free(self, gpu: &mut Gpu) -> Result<()> {
        self.state.free(gpu)?;
        for t in [self.q, self.k, self.v, self.a, self.out] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Lab kernels (shape-parameterised copies) and the production route behind one
// interface so both arms run the same stage sequence.
// ---------------------------------------------------------------------------

struct Lab {
    prep: Kernel,
    fixup: Kernel,
    kkt_batched: Kernel,
    kkt_legacy: Kernel,
    scan_mseg: Kernel,
    scan_segment: Kernel,
}

impl Lab {
    fn new(gpu: &Gpu, shape: Shape) -> Result<Self> {
        let d = shape.lab_defines();
        let tag = format!("gdn_gen_oracle_lab_hk{}_hv{}", shape.hk, shape.hv);
        // Production module flags (compiler.rs module_flags_for) for the gdn_chunk_* modules.
        let fp = "-ffp-contract=off";
        let fp_scan = "-ffp-contract=off -mcumode";
        Ok(Self {
            prep: compile(gpu, &format!("{tag}_prep"), LAB_PREP, "gdn_chunk_prep", d, fp)?,
            fixup: compile(
                gpu,
                &format!("{tag}_fixup"),
                LAB_FIXUP,
                "gdn_chunk_prep_fixup",
                d,
                fp,
            )?,
            kkt_batched: compile(
                gpu,
                &format!("{tag}_kkt_batched"),
                LAB_KKT,
                "gdn_chunk_kkt_solve_batched",
                &format!("{d} -DGDN_KKT_BATCHED=1"),
                fp,
            )?,
            kkt_legacy: compile(
                gpu,
                &format!("{tag}_kkt_legacy"),
                LAB_KKT,
                "gdn_chunk_kkt_solve",
                d,
                fp,
            )?,
            scan_mseg: compile(
                gpu,
                &format!("{tag}_scan_mseg"),
                LAB_SCAN,
                "gdn_chunk_scan_bf16_mseg",
                &format!(
                    "{d} -DGDN_SCAN_OUT_BF16=1 -DGDN_SCAN_MULTISEG=1 -DGDN_CHUNK_SCAN_KERNEL=gdn_chunk_scan_bf16_mseg"
                ),
                fp_scan,
            )?,
            scan_segment: compile(
                gpu,
                &format!("{tag}_scan_segment"),
                LAB_SCAN,
                "gdn_chunk_scan_bf16",
                &format!("{d} -DGDN_SCAN_OUT_BF16=1 -DGDN_CHUNK_SCAN_KERNEL=gdn_chunk_scan_bf16"),
                fp_scan,
            )?,
        })
    }
}

#[derive(Clone, Copy)]
enum Impl<'a> {
    Lab(&'a Lab),
    Production,
}

fn prep_args(inp: &Inputs, arm: &Arm) -> KernargBlob {
    let mut b = KernargBlob::new();
    for t in [
        &inp.qkv,
        &inp.conv_w,
        &arm.state.conv_state,
        &arm.state.alpha,
        &arm.state.beta,
        &inp.dt_bias,
        &inp.a_log,
        &arm.q,
        &arm.k,
        &arm.v,
    ] {
        b.push_ptr(t.buf.as_ptr());
    }
    b.push_i32(inp.rows as i32);
    b.push_f32(q_scale());
    b.push_f32(EPS);
    b
}

fn scan_args(arm: &Arm, a_view: &GpuTensor, row0: usize, rows: usize) -> KernargBlob {
    let mut b = KernargBlob::new();
    for p in [
        arm.q.buf.as_ptr(),
        arm.k.buf.as_ptr(),
        arm.v.buf.as_ptr(),
        a_view.buf.as_ptr(),
        arm.state.alpha.buf.as_ptr(),
        arm.state.beta.buf.as_ptr(),
        arm.state.s_q8.buf.as_ptr(),
        arm.state.s_scales.buf.as_ptr(),
        arm.state.ef.buf.as_ptr(),
        arm.out.buf.as_ptr(),
    ] {
        b.push_ptr(p);
    }
    b.push_i32(row0 as i32);
    b.push_i32(rows as i32);
    b
}

fn kkt_args(arm: &Arm, a_view: &GpuTensor, row0: usize, rows: usize) -> KernargBlob {
    let mut b = KernargBlob::new();
    for p in [
        arm.k.buf.as_ptr(),
        arm.state.alpha.buf.as_ptr(),
        arm.state.beta.buf.as_ptr(),
        a_view.buf.as_ptr(),
    ] {
        b.push_ptr(p);
    }
    b.push_i32(row0 as i32);
    b.push_i32(rows as i32);
    b
}

impl Impl<'_> {
    fn prep(&self, gpu: &mut Gpu, inp: &Inputs, arm: &Arm) -> Result<()> {
        let sh = inp.shape;
        match self {
            Impl::Lab(l) => launch(
                gpu,
                &l.prep,
                [
                    inp.rows.div_ceil(64) as u32,
                    sh.units().div_ceil(8) as u32,
                    1,
                ],
                [256, 1, 1],
                prep_args(inp, arm),
            ),
            Impl::Production => Ok(gpu.gdn_chunk_prep(
                &inp.qkv,
                &inp.conv_w,
                &arm.state.conv_state,
                &arm.state.alpha,
                &arm.state.beta,
                &inp.dt_bias,
                &inp.a_log,
                &arm.q,
                &arm.k,
                &arm.v,
                inp.rows,
                q_scale(),
                EPS,
                sh.hk,
                sh.hv,
            )?),
        }
    }

    fn fixup(&self, gpu: &mut Gpu, inp: &Inputs, arm: &Arm) -> Result<()> {
        let sh = inp.shape;
        match self {
            Impl::Lab(l) => launch(
                gpu,
                &l.fixup,
                [
                    inp.rows.div_ceil(64).max(inp.rows.div_ceil(128)) as u32,
                    (sh.units().div_ceil(8) + 1) as u32,
                    1,
                ],
                [256, 1, 1],
                prep_args(inp, arm),
            ),
            Impl::Production => Ok(gpu.gdn_chunk_prep_fixup(
                &inp.qkv,
                &inp.conv_w,
                &arm.state.conv_state,
                &arm.state.alpha,
                &arm.state.beta,
                &inp.dt_bias,
                &inp.a_log,
                &arm.q,
                &arm.k,
                &arm.v,
                inp.rows,
                q_scale(),
                EPS,
                sh.hk,
                sh.hv,
            )?),
        }
    }

    fn kkt_batched(&self, gpu: &mut Gpu, inp: &Inputs, arm: &Arm) -> Result<()> {
        let sh = inp.shape;
        match self {
            Impl::Lab(l) => launch(
                gpu,
                &l.kkt_batched,
                [sh.hk as u32, inp.rows.div_ceil(64) as u32, 1],
                [128, 1, 1],
                kkt_args(arm, &arm.a, 0, inp.rows),
            ),
            Impl::Production => Ok(gpu.gdn_chunk_kkt_solve_batched(
                &arm.k,
                &arm.state.alpha,
                &arm.state.beta,
                &arm.a,
                inp.rows,
                sh.hk,
                sh.hv,
            )?),
        }
    }

    fn scan_mseg(&self, gpu: &mut Gpu, inp: &Inputs, arm: &Arm) -> Result<()> {
        let sh = inp.shape;
        match self {
            Impl::Lab(l) => launch(
                gpu,
                &l.scan_mseg,
                [1, sh.hv as u32, 1],
                [512, 1, 1],
                scan_args(arm, &arm.a, 0, inp.rows),
            ),
            Impl::Production => Ok(gpu.gdn_chunk_scan_layer_mseg(
                &arm.q,
                &arm.k,
                &arm.v,
                &arm.a,
                &arm.state.alpha,
                &arm.state.beta,
                &arm.state.s_q8,
                &arm.state.s_scales,
                &arm.state.ef,
                &arm.out,
                inp.rows,
                sh.hk,
                sh.hv,
            )?),
        }
    }

    /// One 512-row commit segment. `a_view` is the segment's A blocks (batched KKT:
    /// a row view of the layer A; legacy: the whole A buffer, rewritten per segment).
    fn scan_segment(
        &self,
        gpu: &mut Gpu,
        inp: &Inputs,
        arm: &Arm,
        a_view: &GpuTensor,
        row0: usize,
        rows: usize,
        kkt: bool,
    ) -> Result<()> {
        let sh = inp.shape;
        match self {
            Impl::Lab(l) => {
                if kkt {
                    launch(
                        gpu,
                        &l.kkt_legacy,
                        [rows.div_ceil(64) as u32, sh.hk as u32, 1],
                        [128, 1, 1],
                        kkt_args(arm, a_view, row0, rows),
                    )?;
                }
                launch(
                    gpu,
                    &l.scan_segment,
                    [1, sh.hv as u32, 1],
                    [512, 1, 1],
                    scan_args(arm, a_view, row0, rows),
                )
            }
            Impl::Production => Ok(gpu.gdn_chunk_scan_segment(
                &arm.q,
                &arm.k,
                &arm.v,
                a_view,
                &arm.state.alpha,
                &arm.state.beta,
                &arm.state.s_q8,
                &arm.state.s_scales,
                &arm.state.ef,
                &arm.out,
                GdnScanOut::Bf16,
                row0,
                rows,
                kkt,
                sh.hk,
                sh.hv,
            )?),
        }
    }

    fn scan_segments(
        &self,
        gpu: &mut Gpu,
        inp: &Inputs,
        arm: &Arm,
        kkt_batched: bool,
    ) -> Result<()> {
        let hv = inp.shape.hv;
        for row0 in (0..inp.rows).step_by(SEGMENT_ROWS) {
            let rows = (inp.rows - row0).min(SEGMENT_ROWS);
            if kkt_batched {
                // Same view as prefill.rs: this segment's A blocks start at its layer row.
                let view = arm
                    .a
                    .sub_offset(row0 * hv * 64 * 2, rows.div_ceil(64) * 64 * hv * 64 * 2);
                self.scan_segment(gpu, inp, arm, &view, row0, rows, false)?;
            } else {
                self.scan_segment(gpu, inp, arm, &arm.a, row0, rows, true)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Stage {
    Prep,
    PrepFixup,
    KktBatched,
    ScanMseg,
    ScanSegmentsBatchedKkt,
    ScanSegmentsLegacyKkt,
}

impl Stage {
    const ALL: [Stage; 6] = [
        Stage::Prep,
        Stage::PrepFixup,
        Stage::KktBatched,
        Stage::ScanMseg,
        Stage::ScanSegmentsBatchedKkt,
        Stage::ScanSegmentsLegacyKkt,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Prep => "prep",
            Stage::PrepFixup => "prep_fixup",
            Stage::KktBatched => "kkt_batched",
            Stage::ScanMseg => "scan_mseg",
            Stage::ScanSegmentsBatchedKkt => "scan_segments_batched_kkt",
            Stage::ScanSegmentsLegacyKkt => "scan_segments_legacy_kkt",
        }
    }

    /// Lab planes that must be nonzero after the stage; an all-zero plane would make
    /// byte identity vacuous.
    fn required_nonzero(self) -> &'static [&'static str] {
        match self {
            Stage::Prep | Stage::PrepFixup => &["q", "k", "v", "alpha", "beta", "conv_state"],
            Stage::KktBatched => &["q", "k", "v", "A", "alpha", "beta", "conv_state"],
            Stage::ScanMseg | Stage::ScanSegmentsBatchedKkt | Stage::ScanSegmentsLegacyKkt => &[
                "q",
                "k",
                "v",
                "A",
                "alpha",
                "beta",
                "conv_state",
                "state_q8",
                "state_scales",
                "out",
            ],
        }
    }

    fn run(self, imp: &Impl, gpu: &mut Gpu, inp: &Inputs, arm: &Arm) -> Result<()> {
        arm.clear_planes(gpu)?;
        arm.state.reset(gpu, inp)?;
        match self {
            Stage::Prep => imp.prep(gpu, inp, arm)?,
            Stage::PrepFixup => {
                // Fixup completes planes a fused GEMM would have left: run prep to populate
                // q/k/v/gates/ring, restore the raw gates + ring, then run fixup over them.
                imp.prep(gpu, inp, arm)?;
                gpu.hip.device_synchronize()?;
                arm.state.reset(gpu, inp)?;
                imp.fixup(gpu, inp, arm)?;
            }
            Stage::KktBatched => {
                imp.prep(gpu, inp, arm)?;
                imp.kkt_batched(gpu, inp, arm)?;
            }
            Stage::ScanMseg => {
                imp.prep(gpu, inp, arm)?;
                imp.kkt_batched(gpu, inp, arm)?;
                imp.scan_mseg(gpu, inp, arm)?;
            }
            Stage::ScanSegmentsBatchedKkt => {
                imp.prep(gpu, inp, arm)?;
                imp.kkt_batched(gpu, inp, arm)?;
                imp.scan_segments(gpu, inp, arm, true)?;
            }
            Stage::ScanSegmentsLegacyKkt => {
                imp.prep(gpu, inp, arm)?;
                imp.scan_segments(gpu, inp, arm, false)?;
            }
        }
        gpu.hip.device_synchronize()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Comparison and timing
// ---------------------------------------------------------------------------

/// Compare every plane; returns (plane records, all identical).
fn compare(
    stage: Stage,
    lab: &[(&'static str, Vec<u8>)],
    production: &[(&'static str, Vec<u8>)],
) -> Result<(Vec<Value>, bool)> {
    if lab.len() != production.len() {
        return Err("plane snapshot layout mismatch".into());
    }
    let required = stage.required_nonzero();
    let mut records = Vec::new();
    let mut identical = true;
    for ((name, a), (other, b)) in lab.iter().zip(production) {
        if name != other || a.len() != b.len() {
            return Err(format!("plane {name}/{other} layout mismatch").into());
        }
        let nonzero_bytes = a.iter().filter(|x| **x != 0).count();
        if nonzero_bytes == 0 && required.contains(name) {
            return Err(format!(
                "vacuous comparison: lab plane {name} is all zero after stage {}",
                stage.name()
            )
            .into());
        }
        let first = a.iter().zip(b).position(|(x, y)| x != y);
        let mismatching = a.iter().zip(b).filter(|(x, y)| x != y).count();
        identical &= first.is_none();
        records.push(json!({
            "plane": name,
            "bytes": a.len(),
            "lab_nonzero_bytes": nonzero_bytes,
            "byte_identical": first.is_none(),
            "mismatching_bytes": mismatching,
            "first_mismatching_byte": first,
        }));
    }
    Ok((records, identical))
}

/// Nearest-rank percentile over sorted samples (index = round(p * (n - 1))).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[(p * (sorted.len() - 1) as f64).round() as usize]
}

/// Production chain prep -> batched KKT -> mseg scan. State restore and plane clearing
/// are outside the events; every rep is event-synchronized.
fn time_production(gpu: &mut Gpu, inp: &Inputs, arm: &Arm) -> Result<Value> {
    let imp = Impl::Production;
    let start = gpu.hip.event_create()?;
    let stop = gpu.hip.event_create()?;
    let mut samples = Vec::with_capacity(REPS);
    for rep in 0..WARMUPS + REPS {
        arm.state.reset(gpu, inp)?;
        gpu.hip.device_synchronize()?;
        gpu.hip.event_record(&start, None)?;
        imp.prep(gpu, inp, arm)?;
        imp.kkt_batched(gpu, inp, arm)?;
        imp.scan_mseg(gpu, inp, arm)?;
        gpu.hip.event_record(&stop, None)?;
        gpu.hip.event_synchronize(&stop)?;
        let ms = gpu.hip.event_elapsed_ms(&start, &stop)? as f64;
        if rep >= WARMUPS {
            samples.push(ms);
        }
    }
    gpu.hip.event_destroy(start)?;
    gpu.hip.event_destroy(stop)?;
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = (sorted[(n - 1) / 2] + sorted[n / 2]) * 0.5;
    Ok(json!({
        "path": "production gdn_chunk_prep -> gdn_chunk_kkt_solve_batched -> gdn_chunk_scan_layer_mseg",
        "timer": "hipEvent, device-synchronized before and event-synchronized after every rep",
        "warmups": WARMUPS,
        "reps": REPS,
        "median_ms": median,
        "p10_ms": percentile(&sorted, 0.10),
        "p90_ms": percentile(&sorted, 0.90),
        "min_ms": sorted[0],
        "max_ms": sorted[n - 1],
        "samples_ms": samples,
    }))
}

fn run() -> Result<()> {
    if std::env::args().len() > 1 {
        return Err("usage: gdn_chunk_generalized_oracle (no arguments)".into());
    }
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("gfx1201 required, got {}", gpu.arch).into());
    }
    if hipfire_config::developer_var("HIPFIRE_GDN_SCAN_OUT_EMU").is_ok() {
        return Err("HIPFIRE_GDN_SCAN_OUT_EMU changes the production scan module; unset it".into());
    }

    let mut case_count = 0usize;
    let mut stage_count = 0usize;
    let mut plane_count = 0usize;
    let mut mismatched: Vec<String> = Vec::new();
    let mut timings = Vec::new();

    for hv in [48usize, 32] {
        let shape = Shape::new(16, hv)?;
        let lab = Lab::new(&gpu, shape)?;
        for rows in [64usize, 512, 4096] {
            for nonzero in [false, true] {
                let input = Inputs::new(&mut gpu, shape, rows, nonzero)?;
                let lab_arm = Arm::new(&mut gpu, &input)?;
                let prod_arm = Arm::new(&mut gpu, &input)?;
                let mut stages = Vec::new();
                let mut case_identical = true;
                for stage in Stage::ALL {
                    stage.run(&Impl::Lab(&lab), &mut gpu, &input, &lab_arm)?;
                    let lab_planes = lab_arm.snapshot(&gpu)?;
                    stage.run(&Impl::Production, &mut gpu, &input, &prod_arm)?;
                    let prod_planes = prod_arm.snapshot(&gpu)?;
                    let (planes, identical) = compare(stage, &lab_planes, &prod_planes)?;
                    stage_count += 1;
                    plane_count += planes.len();
                    if !identical {
                        for p in planes.iter().filter(|p| p["byte_identical"] == false) {
                            mismatched.push(format!(
                                "hv{hv}/rows{rows}/{}/{}/{}",
                                if nonzero { "nonzero" } else { "zero" },
                                stage.name(),
                                p["plane"].as_str().unwrap_or("?")
                            ));
                        }
                    }
                    case_identical &= identical;
                    stages.push(json!({
                        "stage": stage.name(),
                        "byte_identical": identical,
                        "planes": planes,
                    }));
                }
                let mut case = json!({
                    "record": "case",
                    "oracle": "gdn_chunk_generalized_oracle",
                    "n_key_heads": 16,
                    "n_value_heads": hv,
                    "lab_defines": if hv == 48 { "<lab defaults>" } else { A3B_DEFINES },
                    "rows": rows,
                    "initial_state": if nonzero { "nonzero" } else { "zero" },
                    "byte_identical": case_identical,
                    "stages": stages,
                });
                if hv == 32 {
                    let t = time_production(&mut gpu, &input, &prod_arm)?;
                    timings.push(json!({
                        "rows": rows,
                        "initial_state": if nonzero { "nonzero" } else { "zero" },
                        "median_ms": t["median_ms"],
                        "p10_ms": t["p10_ms"],
                        "p90_ms": t["p90_ms"],
                    }));
                    case["production_timing"] = t;
                }
                println!("{}", serde_json::to_string(&case)?);
                case_count += 1;
                lab_arm.free(&mut gpu)?;
                prod_arm.free(&mut gpu)?;
                input.free(&mut gpu)?;
            }
        }
    }

    let all_identical = mismatched.is_empty();
    println!(
        "{}",
        serde_json::to_string(&json!({
            "record": "summary",
            "oracle": "gdn_chunk_generalized_oracle",
            "arch": gpu.arch,
            "cases": case_count,
            "stages_compared": stage_count,
            "planes_compared": plane_count,
            "all_byte_identical": all_identical,
            "mismatched": mismatched,
            "production_timing_16x32": {
                "warmups": WARMUPS,
                "reps": REPS,
                "per_case": timings,
            },
        }))?
    );
    if !all_identical {
        return Err("generalized production chunk route differs from the lab oracle".into());
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gdn_chunk_generalized_oracle: {e}");
        std::process::exit(1);
    }
}
