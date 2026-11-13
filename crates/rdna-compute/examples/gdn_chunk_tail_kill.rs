// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LAB ONLY: short-tail kill experiment for the production GDN chunk-scan route
//! (`gdn_chunk_prep` -> KKT -> `gdn_chunk_scan_layer_mseg` / `gdn_chunk_scan_segment`).
//!
//! Question: may a final commit segment of 1..63 rows (and any `n % 512`) run
//! inside the scan with masked padding instead of forcing the whole outer chunk
//! onto the sequential recurrence? Two kills per tail `t`:
//!
//!   * poison: the combined `512 + t` call run with every scratch region past
//!     the last valid row filled with NaN (0xFF bytes) or huge (0x77 bytes)
//!     patterns -- prep input, raw gates, q/k/v planes, A blocks, out plane --
//!     must be byte-identical (out rows, Q8 state, scales, EF, conv ring) to
//!     the same call over zero padding, and must not write the padded out rows
//!     or the padded q/k/v rows.
//!   * split: the combined `512 + t` call must be byte-identical to a 512-row
//!     call followed by a separate `t`-row call (the schedule a smaller outer
//!     chunk executes), on out rows, Q8 state, scales, EF and conv ring.
//!
//! An input-bit negative control (sign bits of the last valid row's first v head) must
//! change the result, so the byte comparisons cannot be blind.
//!
//! A standalone `t`-row call (t < 64) is also compared against the combined call's
//! tail rows. The route mirrors `hipfire_dispatch::pipeline::batched_deltanet::
//! gdn_chunk_scan_layer` exactly (segment rows = commit stride or n; batched KKT
//! only when `n > segment_rows`; mseg only on gfx1201 bf16 with 512-row segments).
//!
//! Build: cargo build --release -p rdna-compute --features lab,deltanet --example gdn_chunk_tail_kill
//! Run:   HIPFIRE_GRAPH=0 gdn_chunk_tail_kill --json <receipt.json> [--hv 48|32] [--tails csv]
//! Exit status 0 only if every cell is byte identical and non-vacuous.

use rdna_compute::norm::GdnScanOut;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

const HD: usize = 128;
const HK: usize = 16;
const EPS: f32 = 1e-6;
const PAD_ROWS: usize = 64;
const SEG: usize = 512;

fn q_scale() -> f32 {
    1.0 / (HD as f32).sqrt()
}

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

fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return (sign << 15) | 0x7bff;
    }
    if e <= 0 {
        return sign << 15;
    }
    (sign << 15) | ((e as u16) << 10) | (mant >> 13) as u16
}

fn fnv(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<u8>> {
    let mut data = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut data, &t.buf)?;
    Ok(data)
}

fn raw(gpu: &mut Gpu, bytes: usize) -> Result<GpuTensor> {
    Ok(gpu.zeros(&[bytes], DType::Raw)?)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pad {
    Zero,
    Nan,
    Huge,
}

impl Pad {
    fn byte(self) -> u8 {
        match self {
            Pad::Zero => 0x00,
            Pad::Nan => 0xff,
            Pad::Huge => 0x77,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Pad::Zero => "zero",
            Pad::Nan => "nan",
            Pad::Huge => "huge",
        }
    }
}

struct Fixture {
    hv: usize,
    xdim: usize,
    qkv: Vec<f32>,
    beta_raw: Vec<f32>,
    alpha_raw: Vec<f32>,
    conv_w: GpuTensor,
    dt_bias: GpuTensor,
    a_log: GpuTensor,
    conv0: Vec<u8>,
    q80: Vec<u8>,
    scales0: Vec<u8>,
    ef0: Vec<u8>,
}

impl Fixture {
    fn new(gpu: &mut Gpu, hv: usize, total_rows: usize) -> Result<Self> {
        let xdim = (2 * HK + hv) * HD;
        let up = |gpu: &mut Gpu, d: Vec<f32>| -> Result<GpuTensor> {
            Ok(gpu.upload_f32(&d, &[d.len()])?)
        };
        let conv_w = up(gpu, lcg(5, 4 * xdim, -0.5, 0.5))?;
        let dt_bias = up(gpu, lcg(6, hv, -4.0, -2.0))?;
        let a_log = up(gpu, lcg(7, hv, 0.0, 2.0))?;
        Ok(Self {
            hv,
            xdim,
            qkv: lcg(1, total_rows * xdim, -1.732_050_8, 1.732_050_8),
            beta_raw: lcg(3, total_rows * hv, -2.0, 2.0),
            alpha_raw: lcg(4, total_rows * hv, -1.732_050_8, 1.732_050_8),
            conv_w,
            dt_bias,
            a_log,
            conv0: f32_bytes(&lcg(9, 3 * xdim, -0.5, 0.5)),
            q80: lcg(10, hv * HD * HD, -64.0, 64.0)
                .iter()
                .map(|x| *x as i8 as u8)
                .collect(),
            scales0: f32_bytes(&lcg(11, hv * HD, 0.005, 0.02)),
            ef0: lcg(12, hv * HD * HD, -0.004, 0.004)
                .iter()
                .flat_map(|x| f16_bits(*x).to_ne_bytes())
                .collect(),
        })
    }
}

/// Persistent recurrent state of one layer.
struct State {
    conv: GpuTensor,
    s_q8: GpuTensor,
    s_scales: GpuTensor,
    ef: GpuTensor,
}

impl State {
    fn new(gpu: &mut Gpu, fx: &Fixture) -> Result<Self> {
        let st = Self {
            conv: raw(gpu, 3 * fx.xdim * 4)?,
            s_q8: raw(gpu, fx.hv * HD * HD)?,
            s_scales: raw(gpu, fx.hv * HD * 4)?,
            ef: raw(gpu, fx.hv * HD * HD * 2)?,
        };
        st.reset(gpu, fx)?;
        Ok(st)
    }
    fn reset(&self, gpu: &Gpu, fx: &Fixture) -> Result<()> {
        gpu.memcpy_htod_auto(&self.conv.buf, &fx.conv0)?;
        gpu.memcpy_htod_auto(&self.s_q8.buf, &fx.q80)?;
        gpu.memcpy_htod_auto(&self.s_scales.buf, &fx.scales0)?;
        gpu.memcpy_htod_auto(&self.ef.buf, &fx.ef0)?;
        Ok(())
    }
    fn snapshot(&self, gpu: &Gpu) -> Result<StateBytes> {
        Ok(StateBytes {
            conv: download(gpu, &self.conv)?,
            q8: download(gpu, &self.s_q8)?,
            scales: download(gpu, &self.s_scales)?,
            ef: download(gpu, &self.ef)?,
        })
    }
    fn free(self, gpu: &mut Gpu) -> Result<()> {
        for t in [self.conv, self.s_q8, self.s_scales, self.ef] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct StateBytes {
    conv: Vec<u8>,
    q8: Vec<u8>,
    scales: Vec<u8>,
    ef: Vec<u8>,
}

/// Scratch of one call over `rows` valid rows with `PAD_ROWS` rows of slack and the A / out planes
/// rounded to whole 64-row blocks. Everything past the valid rows is filled with `pad`.
struct Call {
    rows: usize,
    out_fmt: GdnScanOut,
    x: GpuTensor,
    alpha: GpuTensor,
    beta: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    a: GpuTensor,
    out: GpuTensor,
    alloc_rows: usize,
    out_elem: usize,
    hv: usize,
}

impl Call {
    fn new(
        gpu: &mut Gpu,
        fx: &Fixture,
        row0: usize,
        rows: usize,
        out_fmt: GdnScanOut,
        pad: Pad,
        lsb_control: bool,
    ) -> Result<Self> {
        let hv = fx.hv;
        let alloc_rows = rows.div_ceil(64) * 64 + PAD_ROWS;
        let out_elem = if out_fmt == GdnScanOut::Bf16 { 2 } else { 4 };
        let fill = |gpu: &mut Gpu, valid: &[u8], total: usize| -> Result<GpuTensor> {
            let mut data = vec![pad.byte(); total];
            data[..valid.len()].copy_from_slice(valid);
            let t = raw(gpu, total)?;
            gpu.memcpy_htod_auto(&t.buf, &data)?;
            Ok(t)
        };
        let mut xb = f32_bytes(&fx.qkv[row0 * fx.xdim..(row0 + rows) * fx.xdim]);
        if lsb_control {
            // Flip a whole value head, rather than one component that can round away
            // during F16 preparation. No threshold is changed by this sensitivity control.
            for channel in 0..HD {
                let off = ((rows - 1) * fx.xdim + 2 * HK * HD + channel) * 4;
                xb[off + 3] ^= 0x80;
            }
        }
        let ab = f32_bytes(&fx.alpha_raw[row0 * hv..(row0 + rows) * hv]);
        let bb = f32_bytes(&fx.beta_raw[row0 * hv..(row0 + rows) * hv]);
        Ok(Self {
            rows,
            out_fmt,
            x: fill(gpu, &xb, alloc_rows * fx.xdim * 4)?,
            alpha: fill(gpu, &ab, alloc_rows * hv * 4)?,
            beta: fill(gpu, &bb, alloc_rows * hv * 4)?,
            q: fill(gpu, &[], alloc_rows * HK * HD * 2)?,
            k: fill(gpu, &[], alloc_rows * HK * HD * 2)?,
            v: fill(gpu, &[], alloc_rows * hv * HD * 2)?,
            a: fill(gpu, &[], alloc_rows * hv * 64 * 2)?,
            out: fill(gpu, &[], alloc_rows * hv * HD * out_elem)?,
            alloc_rows,
            out_elem,
            hv,
        })
    }

    fn free(self, gpu: &mut Gpu) -> Result<()> {
        for t in [
            self.x, self.alpha, self.beta, self.q, self.k, self.v, self.a, self.out,
        ] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }

    fn out_row_bytes(&self) -> usize {
        self.hv * HD * self.out_elem
    }
}

/// Production route, `commit_stride` = Some(512) when `stride`.
fn run_route(gpu: &mut Gpu, fx: &Fixture, st: &State, c: &Call, stride: bool) -> Result<Value> {
    let n = c.rows;
    let hv = fx.hv;
    gpu.gdn_chunk_prepare(c.out_fmt)?;
    let segment_rows = if stride { SEG } else { n };
    let kkt_batched = gpu.gdn_kkt_batched_enabled() && segment_rows % 64 == 0 && n > segment_rows;
    gpu.gdn_chunk_prep(
        &c.x,
        &fx.conv_w,
        &st.conv,
        &c.alpha,
        &c.beta,
        &fx.dt_bias,
        &fx.a_log,
        &c.q,
        &c.k,
        &c.v,
        n,
        q_scale(),
        EPS,
        HK,
        hv,
    )?;
    if kkt_batched {
        gpu.gdn_chunk_kkt_solve_batched(&c.k, &c.alpha, &c.beta, &c.a, n, HK, hv)?;
    }
    let mseg = kkt_batched && gpu.gdn_scan_mseg_enabled(c.out_fmt, segment_rows);
    if mseg {
        gpu.gdn_chunk_scan_layer_mseg(
            &c.q,
            &c.k,
            &c.v,
            &c.a,
            &c.alpha,
            &c.beta,
            &st.s_q8,
            &st.s_scales,
            &st.ef,
            &c.out,
            n,
            HK,
            hv,
        )?;
    } else {
        for row0 in (0..n).step_by(segment_rows) {
            let rows = (n - row0).min(segment_rows);
            let a_seg = if kkt_batched {
                c.a.sub_offset(row0 * hv * 64 * 2, rows.div_ceil(64) * 64 * hv * 64 * 2)
            } else {
                c.a.shallow_clone()
            };
            gpu.gdn_chunk_scan_segment(
                &c.q,
                &c.k,
                &c.v,
                &a_seg,
                &c.alpha,
                &c.beta,
                &st.s_q8,
                &st.s_scales,
                &st.ef,
                &c.out,
                c.out_fmt,
                row0,
                rows,
                !kkt_batched,
                HK,
                hv,
            )?;
        }
    }
    gpu.hip.device_synchronize()?;
    Ok(json!({
        "rows": n,
        "segment_rows": segment_rows,
        "kkt_batched": kkt_batched,
        "mseg": mseg,
        "segments": n.div_ceil(segment_rows),
    }))
}

struct Result_ {
    route: Value,
    out_valid: Vec<u8>,
    out_pad: Vec<u8>,
    q_pad: Vec<u8>,
    k_pad: Vec<u8>,
    v_pad: Vec<u8>,
    q_valid: Vec<u8>,
    k_valid: Vec<u8>,
    v_valid: Vec<u8>,
    state: StateBytes,
}

fn execute(
    gpu: &mut Gpu,
    fx: &Fixture,
    st: &State,
    row0: usize,
    rows: usize,
    out_fmt: GdnScanOut,
    pad: Pad,
    stride: bool,
    lsb_control: bool,
) -> Result<Result_> {
    let c = Call::new(gpu, fx, row0, rows, out_fmt, pad, lsb_control)?;
    let route = run_route(gpu, fx, st, &c, stride)?;
    let out = download(gpu, &c.out)?;
    let (q, k, v) = (
        download(gpu, &c.q)?,
        download(gpu, &c.k)?,
        download(gpu, &c.v)?,
    );
    let ob = c.out_row_bytes();
    let (qb, vb) = (HK * HD * 2, fx.hv * HD * 2);
    let res = Result_ {
        route,
        out_valid: out[..rows * ob].to_vec(),
        out_pad: out[rows * ob..].to_vec(),
        q_valid: q[..rows * qb].to_vec(),
        k_valid: k[..rows * qb].to_vec(),
        v_valid: v[..rows * vb].to_vec(),
        q_pad: q[rows * qb..].to_vec(),
        k_pad: k[rows * qb..].to_vec(),
        v_pad: v[rows * vb..].to_vec(),
        state: st.snapshot(gpu)?,
    };
    let _ = c.alloc_rows;
    c.free(gpu)?;
    Ok(res)
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn nonfinite_f32(b: &[u8]) -> usize {
    b.chunks_exact(4)
        .filter(|c| !f32::from_ne_bytes([c[0], c[1], c[2], c[3]]).is_finite())
        .count()
}

fn nonfinite_out(b: &[u8], fmt: GdnScanOut) -> usize {
    match fmt {
        GdnScanOut::F32 => nonfinite_f32(b),
        // bf16 bits: exponent all ones.
        GdnScanOut::Bf16 => b
            .chunks_exact(2)
            .filter(|c| u16::from_ne_bytes([c[0], c[1]]) & 0x7f80 == 0x7f80)
            .count(),
    }
}

fn cmp(name: &str, a: &[u8], b: &[u8], cell: &mut Vec<Value>, bad: &mut Vec<String>, ctx: &str) {
    let d = first_diff(a, b);
    if d.is_some() {
        bad.push(format!("{ctx}/{name}"));
    }
    cell.push(json!({
        "plane": name,
        "bytes": a.len(),
        "byte_identical": d.is_none(),
        "first_diff": d,
        "hash_a": fnv(a),
        "hash_b": fnv(b),
    }));
}

fn run() -> Result<()> {
    let mut json_path: Option<String> = None;
    let mut hv = 48usize;
    let mut tails: Vec<usize> = vec![1, 2, 3, 5, 17, 25, 31, 32, 33, 62, 63, 64, 65, 100, 511];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--json" => json_path = args.next(),
            "--hv" => hv = args.next().ok_or("--hv value")?.parse()?,
            "--tails" => {
                tails = args
                    .next()
                    .ok_or("--tails csv")?
                    .split(',')
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()?
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if !matches!(hv, 48 | 32) {
        return Err("--hv must be 48 or 32".into());
    }
    let mut gpu = Gpu::init()?;
    if hv != 48 && gpu.arch != "gfx1201" {
        return Err("hv 32 only runs on gfx1201".into());
    }
    if hipfire_config::developer_var("HIPFIRE_GDN_SCAN_OUT_EMU").is_ok() {
        return Err("unset HIPFIRE_GDN_SCAN_OUT_EMU".into());
    }
    let out_fmt = if gpu.arch == "gfx1201" {
        GdnScanOut::Bf16
    } else {
        GdnScanOut::F32
    };

    let mut cells = Vec::new();
    let mut bad: Vec<String> = Vec::new();
    for &t in &tails {
        let total = SEG + t;
        let fx = Fixture::new(&mut gpu, hv, total)?;
        let st = State::new(&mut gpu, &fx)?;
        let ctx = format!("hv{hv}/tail{t}");
        let mut planes: Vec<Value> = Vec::new();

        // Combined 512+t call, zero padding (reference), then NaN / huge poisoned padding.
        st.reset(&gpu, &fx)?;
        let refc = execute(
            &mut gpu,
            &fx,
            &st,
            0,
            total,
            out_fmt,
            Pad::Zero,
            true,
            false,
        )?;
        let mut poisoned = Vec::new();
        for pad in [Pad::Nan, Pad::Huge] {
            st.reset(&gpu, &fx)?;
            let r = execute(&mut gpu, &fx, &st, 0, total, out_fmt, pad, true, false)?;
            let c = format!("{ctx}/poison_{}", pad.name());
            let mut p = Vec::new();
            cmp(
                "out_valid",
                &refc.out_valid,
                &r.out_valid,
                &mut p,
                &mut bad,
                &c,
            );
            cmp(
                "state_q8",
                &refc.state.q8,
                &r.state.q8,
                &mut p,
                &mut bad,
                &c,
            );
            cmp(
                "state_scales",
                &refc.state.scales,
                &r.state.scales,
                &mut p,
                &mut bad,
                &c,
            );
            cmp(
                "state_ef",
                &refc.state.ef,
                &r.state.ef,
                &mut p,
                &mut bad,
                &c,
            );
            cmp(
                "conv",
                &refc.state.conv,
                &r.state.conv,
                &mut p,
                &mut bad,
                &c,
            );
            cmp("q_valid", &refc.q_valid, &r.q_valid, &mut p, &mut bad, &c);
            cmp("k_valid", &refc.k_valid, &r.k_valid, &mut p, &mut bad, &c);
            cmp("v_valid", &refc.v_valid, &r.v_valid, &mut p, &mut bad, &c);
            // Padded out / q / k / v rows must still hold the poison pattern (never written).
            for (name, bytes) in [
                ("out_pad_untouched", &r.out_pad),
                ("q_pad_untouched", &r.q_pad),
                ("k_pad_untouched", &r.k_pad),
                ("v_pad_untouched", &r.v_pad),
            ] {
                let ok = bytes.iter().all(|b| *b == pad.byte());
                if !ok {
                    bad.push(format!("{c}/{name}"));
                }
                p.push(json!({"plane": name, "bytes": bytes.len(), "byte_identical": ok}));
            }
            poisoned.push(json!({"pad": pad.name(), "planes": p, "route": r.route}));
        }

        // Split: 512 rows, then the t-row tail as its own call with the carried state.
        st.reset(&gpu, &fx)?;
        let head = execute(&mut gpu, &fx, &st, 0, SEG, out_fmt, Pad::Zero, false, false)?;
        let tail = execute(&mut gpu, &fx, &st, SEG, t, out_fmt, Pad::Nan, false, false)?;
        let c = format!("{ctx}/split");
        let ob = refc.out_valid.len() / total;
        cmp(
            "out_head",
            &refc.out_valid[..SEG * ob],
            &head.out_valid,
            &mut planes,
            &mut bad,
            &c,
        );
        cmp(
            "out_tail",
            &refc.out_valid[SEG * ob..],
            &tail.out_valid,
            &mut planes,
            &mut bad,
            &c,
        );
        cmp(
            "state_q8",
            &refc.state.q8,
            &tail.state.q8,
            &mut planes,
            &mut bad,
            &c,
        );
        cmp(
            "state_scales",
            &refc.state.scales,
            &tail.state.scales,
            &mut planes,
            &mut bad,
            &c,
        );
        cmp(
            "state_ef",
            &refc.state.ef,
            &tail.state.ef,
            &mut planes,
            &mut bad,
            &c,
        );
        cmp(
            "conv",
            &refc.state.conv,
            &tail.state.conv,
            &mut planes,
            &mut bad,
            &c,
        );

        // Input-bit negative control: sign-flipping a real value head must change
        // the out rows or persisted state, otherwise the comparisons above are blind.
        st.reset(&gpu, &fx)?;
        let ctl = execute(&mut gpu, &fx, &st, 0, total, out_fmt, Pad::Zero, true, true)?;
        let control_detected = ctl.out_valid != refc.out_valid
            || ctl.state.q8 != refc.state.q8
            || ctl.state.scales != refc.state.scales
            || ctl.state.ef != refc.state.ef;
        if !control_detected {
            bad.push(format!("{ctx}/input_control_not_detected"));
        }

        // Vacuity / finiteness.
        let finite_bad = nonfinite_out(&refc.out_valid, out_fmt)
            + nonfinite_f32(&refc.state.scales)
            + nonfinite_f32(&refc.state.conv)
            + refc
                .state
                .ef
                .chunks_exact(2)
                .filter(|c| u16::from_ne_bytes([c[0], c[1]]) & 0x7c00 == 0x7c00)
                .count();
        let out_nonzero = refc.out_valid.iter().any(|b| *b != 0);
        let state_moved = refc.state.q8 != fx.q80 && refc.state.ef != fx.ef0;
        let tail_nonzero = refc.out_valid[SEG * ob..].iter().any(|b| *b != 0);
        if finite_bad != 0 {
            bad.push(format!("{ctx}/nonfinite({finite_bad})"));
        }
        if !(out_nonzero && state_moved && tail_nonzero) {
            bad.push(format!("{ctx}/vacuous"));
        }

        cells.push(json!({
            "tail": t,
            "rows": total,
            "reference_route": refc.route,
            "split_head_route": head.route,
            "split_tail_route": tail.route,
            "nonfinite_count": finite_bad,
            "out_nonzero": out_nonzero,
            "tail_out_nonzero": tail_nonzero,
            "state_moved": state_moved,
            "input_control_detected": control_detected,
            "poison": poisoned,
            "split": planes,
        }));
        st.free(&mut gpu)?;
    }

    let receipt = json!({
        "oracle": "gdn_chunk_tail_kill",
        "arch": gpu.arch,
        "n_key_heads": HK,
        "n_value_heads": hv,
        "scan_out": format!("{out_fmt:?}"),
        "tails": tails,
        "all_pass": bad.is_empty(),
        "failures": bad,
        "cells": cells,
    });
    let text = serde_json::to_string_pretty(&receipt)?;
    if let Some(p) = json_path {
        std::fs::write(p, &text)?;
    }
    println!(
        "{}",
        serde_json::to_string(&json!({
            "oracle": "gdn_chunk_tail_kill",
            "arch": gpu.arch,
            "hv": hv,
            "all_pass": receipt["all_pass"],
            "failures": receipt["failures"],
        }))?
    );
    if !bad.is_empty() {
        return Err(format!("{} failing comparison(s)", bad.len()).into());
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gdn_chunk_tail_kill: {e}");
        std::process::exit(1);
    }
}
