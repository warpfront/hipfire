// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab gate: the A3B DeltaNet rows-batched twins against the REAL singleton
//! launchers the product calls (`rdna_compute::attention::rows_batched`).
//!
//! A3B exact-gfx1201 shapes: 32 value heads, 16 QK heads, head_dim 128
//! (k_dim 2048, v_dim 4096, qkv 8192, K = 4096), Q8 state + error feedback.
//!
//! Twins covered, each for N in {1, 2, 3, 8} lanes with distinct per-lane
//! data and state, several steps in sequence so state carry-over is tested
//! (every mutated buffer is compared byte-for-byte after every step):
//!   * `fused_sigmoid_alpha_gate_f32_batched` vs N x `fused_sigmoid_alpha_gate_f32`;
//!   * `conv1d_silu_split_qknorm_b256_rows` vs N x `conv1d_silu_split_qknorm`;
//!   * `gated_delta_net_q8_compact2_b2_rows` vs N x `gated_delta_net_q8_compact`
//!     (qk_head_div 2); lanes alternate with / without error feedback, so the
//!     per-row requant-frame binding (stochastic rounding without EF) is
//!     exercised; the global frame counter is replayed identically in both arms;
//!   * `gated_norm_mq_rotate_gfx1201_rows` vs N x `gated_norm_rotate_mq_gfx1100`
//!     (32 heads, no AWQ).
//! Inputs are random plus adversarial (huge / tiny / zero / +-30 gate and
//! activation values). Negative controls: one flipped input byte changes
//! exactly that lane's output; a wrong-frame GDN launch changes exactly the
//! no-EF lanes; staging refuses aliasing, wrong tags and null pointers.
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --features lab,deltanet --test a3b_dn_rows \
//!       -- --ignored --nocapture
#![cfg(all(feature = "lab", feature = "deltanet"))]

use hip_bridge::HipResult;
use rdna_compute::attention::rows_batched::{ConvRow, GatedNormRow, GdnRow};
use rdna_compute::{Gpu, GpuTensor};

const LANES: [usize; 4] = [1, 2, 3, 8];
const STEPS: usize = 4;
const NV: usize = 32;
const NK: usize = 16;
const HD: usize = 128;
const K_DIM: usize = NK * HD; // 2048
const V_DIM: usize = NV * HD; // 4096
const QKV: usize = 2 * K_DIM + V_DIM; // 8192
const EPS: f32 = 1e-6;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32, bias: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * scale + bias).collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.f() * 120.0) as i8 as u8).collect()
    }
    fn halves(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|_| half_bits(self.f() * 0.01).to_le_bytes())
            .collect()
    }
    /// Adversarial activations: mostly random, with huge, tiny, zero and
    /// exactly-representable ties sprinkled in.
    fn adversarial(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| match i % 17 {
                3 => 0.0,
                5 => scale * 1.0e3,
                7 => -scale * 1.0e3,
                11 => 1.0e-30,
                13 => scale,
                _ => self.f() * scale,
            })
            .collect()
    }
}

fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

struct Checks {
    pass: usize,
    fails: Vec<String>,
}
impl Checks {
    fn check(&mut self, ok: bool, what: &str) {
        println!("[{}] {what}", if ok { "PASS" } else { "FAIL" });
        if ok {
            self.pass += 1;
        } else {
            self.fails.push(what.into());
        }
    }
}

fn up_f(gpu: &mut Gpu, v: &[f32]) -> HipResult<GpuTensor> {
    gpu.upload_f32(v, &[v.len()])
}
fn up_raw(gpu: &Gpu, v: &[u8]) -> HipResult<GpuTensor> {
    gpu.upload_raw(v, &[v.len()])
}
fn bytes(gpu: &Gpu, t: &GpuTensor) -> HipResult<Vec<u8>> {
    gpu.download_raw_bytes(t)
}
fn p(t: &GpuTensor) -> u64 {
    t.buf.as_ptr() as u64
}
fn staging(gpu: &Gpu, n: usize) -> HipResult<GpuTensor> {
    gpu.upload_raw(&vec![0u8; n], &[n])
}
fn tags(k: usize) -> Vec<u64> {
    (0..k as u64).map(|i| 1000 + i).collect()
}
fn free_all(gpu: &mut Gpu, ts: Vec<GpuTensor>) -> HipResult<()> {
    ts.into_iter().try_for_each(|t| gpu.free_tensor(t))
}
fn sync(gpu: &Gpu) -> HipResult<()> {
    gpu.hip.device_synchronize()
}
fn overwrite_f(gpu: &mut Gpu, t: &GpuTensor, v: &[f32]) -> HipResult<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &b)
}

// ── sigmoid / alpha gate ───────────────────────────────────────────────────

fn sigmoid_gate(gpu: &mut Gpu, c: &mut Checks) -> HipResult<()> {
    println!("=== fused_sigmoid_alpha_gate_f32 batched (A3B n=32) ===");
    let mut r = Lcg(0x51A);
    let dt_bias = up_f(gpu, &r.vec(NV, 1.0, 0.0))?;
    let a_log = up_f(gpu, &r.vec(NV, 1.0, 0.0))?;
    for n in LANES {
        // Adversarial gate pre-activations: beyond the softplus +-20 guards.
        let mut beta = r.vec(n * NV, 6.0, 0.0);
        let mut alpha = r.vec(n * NV, 12.0, 0.0);
        for (i, v) in alpha.iter_mut().enumerate() {
            match i % 11 {
                2 => *v = 40.0,
                4 => *v = -40.0,
                6 => *v = 20.0,
                8 => *v = -20.0,
                _ => {}
            }
        }
        beta[0] = 90.0;
        beta[n * NV - 1] = -90.0;
        let (rb, ra) = (up_f(gpu, &beta)?, up_f(gpu, &alpha)?);
        let (tb, ta) = (up_f(gpu, &beta)?, up_f(gpu, &alpha)?);
        for row in 0..n {
            gpu.fused_sigmoid_alpha_gate_f32(
                &rb.sub_offset(row * NV, NV),
                &ra.sub_offset(row * NV, NV),
                &dt_bias,
                &a_log,
                NV,
            )?;
        }
        gpu.fused_sigmoid_alpha_gate_f32_batched(&tb, &ta, &dt_bias, &a_log, NV, n)?;
        sync(gpu)?;
        c.check(
            bytes(gpu, &rb)? == bytes(gpu, &tb)? && bytes(gpu, &ra)? == bytes(gpu, &ta)?,
            &format!("sigmoid/alpha gate n={n}: batched bitwise == per-row singleton"),
        );
        c.check(
            bytes(gpu, &rb)?
                != beta
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>(),
            &format!("sigmoid/alpha gate n={n}: not vacuous (input changed)"),
        );
        // Negative control: one flipped input byte in the last row changes
        // exactly that row's outputs.
        let mut b2 = beta.clone();
        b2[(n - 1) * NV + 5] = f32::from_bits(b2[(n - 1) * NV + 5].to_bits() ^ 0x0040_0000);
        let (cb, ca) = (up_f(gpu, &b2)?, up_f(gpu, &alpha)?);
        gpu.fused_sigmoid_alpha_gate_f32_batched(&cb, &ca, &dt_bias, &a_log, NV, n)?;
        sync(gpu)?;
        let (want, got) = (bytes(gpu, &rb)?, bytes(gpu, &cb)?);
        let row_bytes = NV * 4;
        let differs: Vec<bool> = (0..n)
            .map(|row| {
                want[row * row_bytes..(row + 1) * row_bytes]
                    != got[row * row_bytes..(row + 1) * row_bytes]
            })
            .collect();
        c.check(
            differs
                .iter()
                .enumerate()
                .all(|(row, d)| *d == (row == n - 1)),
            &format!(
                "sigmoid/alpha gate n={n}: negative control changes exactly the perturbed row"
            ),
        );
        free_all(gpu, vec![rb, ra, tb, ta, cb, ca])?;
    }
    free_all(gpu, vec![dt_bias, a_log])
}

// ── conv1d + SiLU + split + QK norm ────────────────────────────────────────

struct ConvLane {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    input: GpuTensor,
    state: GpuTensor,
}
impl ConvLane {
    fn new(gpu: &mut Gpu, r: &mut Lcg) -> HipResult<Self> {
        Ok(Self {
            q: up_f(gpu, &vec![0.0; K_DIM])?,
            k: up_f(gpu, &vec![0.0; K_DIM])?,
            v: up_f(gpu, &vec![0.0; V_DIM])?,
            input: up_f(gpu, &r.vec(QKV, 1.0, 0.0))?,
            state: up_f(gpu, &r.vec(QKV * 3, 1.0, 0.0))?,
        })
    }
    fn dup(&self, gpu: &mut Gpu) -> HipResult<Self> {
        let d = |gpu: &mut Gpu, t: &GpuTensor| -> HipResult<GpuTensor> {
            let b = bytes(gpu, t)?;
            up_raw(gpu, &b)
        };
        Ok(Self {
            q: d(gpu, &self.q)?,
            k: d(gpu, &self.k)?,
            v: d(gpu, &self.v)?,
            input: d(gpu, &self.input)?,
            state: d(gpu, &self.state)?,
        })
    }
    fn row(&self) -> ConvRow {
        ConvRow {
            q_out: p(&self.q),
            k_out: p(&self.k),
            v_out: p(&self.v),
            input: p(&self.input),
            state: p(&self.state),
        }
    }
    fn out_bytes(&self, gpu: &Gpu) -> HipResult<Vec<u8>> {
        let mut o = bytes(gpu, &self.q)?;
        o.extend(bytes(gpu, &self.k)?);
        o.extend(bytes(gpu, &self.v)?);
        o.extend(bytes(gpu, &self.state)?);
        Ok(o)
    }
    fn free(self, gpu: &mut Gpu) -> HipResult<()> {
        free_all(gpu, vec![self.q, self.k, self.v, self.input, self.state])
    }
}

fn conv_single(gpu: &mut Gpu, l: &ConvLane, w: &GpuTensor) -> HipResult<()> {
    gpu.conv1d_silu_split_qknorm(
        &l.q,
        &l.k,
        &l.v,
        &l.input,
        w,
        &l.state,
        K_DIM,
        V_DIM,
        NK,
        HD,
        1.0 / (HD as f32).sqrt(),
        EPS,
    )
}

fn conv(gpu: &mut Gpu, c: &mut Checks) -> HipResult<()> {
    println!("=== conv1d_silu_split_qknorm_b256 rows twin (A3B 2048/4096/16x128) ===");
    let mut r = Lcg(0xC0B3);
    let w = up_f(gpu, &r.vec(QKV * 4, 0.5, 0.0))?;
    for n in LANES {
        let refs: Vec<ConvLane> = (0..n)
            .map(|_| ConvLane::new(gpu, &mut r))
            .collect::<HipResult<_>>()?;
        let twin: Vec<ConvLane> = refs.iter().map(|l| l.dup(gpu)).collect::<HipResult<_>>()?;
        let table = staging(gpu, n * 40)?;
        let rows: Vec<ConvRow> = twin.iter().map(ConvLane::row).collect();
        gpu.stage_conv1d_silu_split_qknorm_b256_rows(
            &table,
            &rows,
            &tags(n),
            &tags(n),
            K_DIM,
            V_DIM,
        )?;
        let mut all = true;
        let mut live = false;
        for step in 0..STEPS {
            // Fresh per-lane inputs each step (adversarial on odd steps); the
            // conv ring carries between steps.
            for i in 0..n {
                let x = if step % 2 == 1 {
                    r.adversarial(QKV, 2.0)
                } else {
                    r.vec(QKV, 1.0, 0.0)
                };
                overwrite_f(gpu, &refs[i].input, &x)?;
                overwrite_f(gpu, &twin[i].input, &x)?;
            }
            for l in &refs {
                conv_single(gpu, l, &w)?;
            }
            gpu.conv1d_silu_split_qknorm_b256_rows(
                &table,
                n,
                &w,
                K_DIM,
                V_DIM,
                NK,
                HD,
                1.0 / (HD as f32).sqrt(),
                EPS,
            )?;
            sync(gpu)?;
            for i in 0..n {
                let same = refs[i].out_bytes(gpu)? == twin[i].out_bytes(gpu)?;
                if !same {
                    println!("    n={n} step {step} lane {i} differs");
                }
                all &= same;
            }
            live |= bytes(gpu, &refs[0].q)?.iter().any(|&b| b != 0);
        }
        c.check(
            all,
            &format!("conv n={n}: {STEPS} steps, q/k/v + conv ring bitwise == per-lane singleton"),
        );
        c.check(
            live,
            &format!("conv n={n}: singleton q_out non-zero (not vacuous)"),
        );
        // Negative control: flip one input byte of the last lane, run both a
        // reference lane copy and the twin: only that lane differs.
        let victim = n - 1;
        let base_in = bytes(gpu, &twin[victim].input)?;
        let ctrl: Vec<ConvLane> = twin.iter().map(|l| l.dup(gpu)).collect::<HipResult<_>>()?;
        let pert: Vec<ConvLane> = twin.iter().map(|l| l.dup(gpu)).collect::<HipResult<_>>()?;
        let mut flipped = base_in.clone();
        flipped[4 * 100] ^= 0x01;
        gpu.hip.memcpy_htod(&pert[victim].input.buf, &flipped)?;
        for set in [&ctrl, &pert] {
            let rows: Vec<ConvRow> = set.iter().map(ConvLane::row).collect();
            gpu.stage_conv1d_silu_split_qknorm_b256_rows(
                &table,
                &rows,
                &tags(n),
                &tags(n),
                K_DIM,
                V_DIM,
            )?;
            gpu.conv1d_silu_split_qknorm_b256_rows(
                &table,
                n,
                &w,
                K_DIM,
                V_DIM,
                NK,
                HD,
                1.0 / (HD as f32).sqrt(),
                EPS,
            )?;
        }
        sync(gpu)?;
        let mut ok = true;
        for i in 0..n {
            let differs = ctrl[i].out_bytes(gpu)? != pert[i].out_bytes(gpu)?;
            ok &= differs == (i == victim);
        }
        c.check(
            ok,
            &format!(
                "conv n={n}: negative control — one flipped input byte changes exactly that lane"
            ),
        );
        if n >= 2 {
            let mut alias = rows.clone();
            alias[1].state = alias[0].state;
            c.check(
                gpu.stage_conv1d_silu_split_qknorm_b256_rows(
                    &table,
                    &alias,
                    &tags(n),
                    &tags(n),
                    K_DIM,
                    V_DIM,
                )
                .is_err(),
                &format!("conv n={n}: shared conv state refused"),
            );
            let mut wrong = tags(n);
            wrong[n - 1] += 77;
            c.check(
                gpu.stage_conv1d_silu_split_qknorm_b256_rows(
                    &table,
                    &rows,
                    &wrong,
                    &tags(n),
                    K_DIM,
                    V_DIM,
                )
                .is_err(),
                &format!("conv n={n}: wrong-row tag refused"),
            );
        }
        gpu.free_tensor(table)?;
        for l in refs.into_iter().chain(twin).chain(ctrl).chain(pert) {
            l.free(gpu)?;
        }
    }
    gpu.free_tensor(w)
}

// ── gated delta net compact2 (Q8 state + EF) ───────────────────────────────

struct GdnIo {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
}
impl GdnIo {
    fn new(gpu: &mut Gpu, r: &mut Lcg, adversarial: bool) -> HipResult<Self> {
        let (gate, beta) = if adversarial {
            // alpha pre-activations (<= 0 after the gate) down to -40, betas
            // saturated at 0 / 1.
            let mut g = r.vec(NV, 20.0, -20.0);
            g[0] = 0.0;
            g[1] = -80.0;
            let mut b = r.vec(NV, 0.5, 0.5);
            b[2] = 0.0;
            b[3] = 1.0;
            (g, b)
        } else {
            (r.vec(NV, 0.4, -0.5), r.vec(NV, 0.4, 0.5))
        };
        Ok(Self {
            q: up_f(gpu, &r.vec(K_DIM, 0.09, 0.0))?,
            k: up_f(gpu, &r.vec(K_DIM, 0.09, 0.0))?,
            v: up_f(
                gpu,
                &r.vec(V_DIM, if adversarial { 30.0 } else { 1.0 }, 0.0),
            )?,
            gate: up_f(gpu, &gate)?,
            beta: up_f(gpu, &beta)?,
        })
    }
    fn free(self, gpu: &mut Gpu) -> HipResult<()> {
        free_all(gpu, vec![self.q, self.k, self.v, self.gate, self.beta])
    }
}

struct GdnState {
    sq: GpuTensor,
    sc: GpuTensor,
    ef: Option<GpuTensor>,
    out: GpuTensor,
}
impl GdnState {
    fn new(gpu: &mut Gpu, r: &mut Lcg, ef: bool) -> HipResult<Self> {
        Ok(Self {
            sq: up_raw(gpu, &r.bytes(NV * HD * HD))?,
            sc: up_f(gpu, &r.vec(NV * HD, 0.004, 0.006))?,
            ef: if ef {
                Some(up_raw(gpu, &r.halves(NV * HD * HD))?)
            } else {
                None
            },
            out: up_f(gpu, &vec![0.0; V_DIM])?,
        })
    }
    fn dup(&self, gpu: &mut Gpu) -> HipResult<Self> {
        let d = |gpu: &mut Gpu, t: &GpuTensor| -> HipResult<GpuTensor> {
            let b = bytes(gpu, t)?;
            up_raw(gpu, &b)
        };
        Ok(Self {
            sq: d(gpu, &self.sq)?,
            sc: d(gpu, &self.sc)?,
            ef: match &self.ef {
                Some(e) => Some(d(gpu, e)?),
                None => None,
            },
            out: d(gpu, &self.out)?,
        })
    }
    fn all_bytes(&self, gpu: &Gpu) -> HipResult<Vec<u8>> {
        let mut b = bytes(gpu, &self.sq)?;
        b.extend(bytes(gpu, &self.sc)?);
        if let Some(e) = &self.ef {
            b.extend(bytes(gpu, e)?);
        }
        b.extend(bytes(gpu, &self.out)?);
        Ok(b)
    }
    fn row(&self, io: &GdnIo) -> GdnRow {
        GdnRow {
            q: p(&io.q),
            k: p(&io.k),
            v: p(&io.v),
            gate: p(&io.gate),
            beta: p(&io.beta),
            s_q8: p(&self.sq),
            s_scales: p(&self.sc),
            output: p(&self.out),
            s_ef_residual: self.ef.as_ref().map(p).unwrap_or(0),
        }
    }
    fn free(self, gpu: &mut Gpu) -> HipResult<()> {
        let mut v = vec![self.sq, self.sc, self.out];
        v.extend(self.ef);
        free_all(gpu, v)
    }
}

fn gdn_single(gpu: &mut Gpu, io: &GdnIo, st: &GdnState) -> HipResult<()> {
    gpu.gated_delta_net_q8_compact(
        &io.q,
        &io.k,
        &io.v,
        &io.gate,
        &io.beta,
        &st.sq,
        &st.sc,
        &st.out,
        1,
        NV,
        HD,
        2,
        st.ef.as_ref(),
    )
}

fn gdn(gpu: &mut Gpu, c: &mut Checks) -> HipResult<()> {
    println!("=== gated_delta_net_q8_compact2_b2 rows twin (A3B 32 value / 16 QK heads) ===");
    let mut r = Lcg(0x6D12);
    for n in LANES {
        // Lane i has EF iff i is even except lane 0 of n==1 etc.: alternate so
        // both the EF path and the frame-bound stochastic-rounding path run.
        // A3B proper has EF on every lane; the all-EF case is checked too.
        for all_ef in [true, false] {
            let has_ef = |i: usize| all_ef || i % 2 == 1;
            let refs: Vec<GdnState> = (0..n)
                .map(|i| GdnState::new(gpu, &mut r, has_ef(i)))
                .collect::<HipResult<_>>()?;
            let twin: Vec<GdnState> = refs.iter().map(|s| s.dup(gpu)).collect::<HipResult<_>>()?;
            let table = staging(gpu, n * 72)?;
            let mut all = true;
            let mut live = false;
            // Both arms consume the identical frame sequence.
            let frame0 = rdna_compute::norm::gdn_requant_frame_checkpoint();
            for step in 0..STEPS {
                let ios: Vec<GdnIo> = (0..n)
                    .map(|_| GdnIo::new(gpu, &mut r, step % 2 == 1))
                    .collect::<HipResult<_>>()?;
                let f = frame0 + (step * n) as u32;
                rdna_compute::norm::restore_gdn_requant_frame_checkpoint(f);
                for (io, st) in ios.iter().zip(&refs) {
                    gdn_single(gpu, io, st)?;
                }
                assert_eq!(
                    rdna_compute::norm::gdn_requant_frame_checkpoint(),
                    f + n as u32
                );
                rdna_compute::norm::restore_gdn_requant_frame_checkpoint(f);
                let rows: Vec<GdnRow> = ios.iter().zip(&twin).map(|(io, s)| s.row(io)).collect();
                gpu.stage_gated_delta_net_q8_compact2_b2_rows(
                    &table,
                    &rows,
                    &tags(n),
                    &tags(n),
                    1,
                    NV,
                )?;
                gpu.gated_delta_net_q8_compact2_b2_rows(&table, n, 1, NV, HD)?;
                assert_eq!(
                    rdna_compute::norm::gdn_requant_frame_checkpoint(),
                    f + n as u32,
                    "twin must reserve exactly n frames"
                );
                sync(gpu)?;
                for i in 0..n {
                    let same = refs[i].all_bytes(gpu)? == twin[i].all_bytes(gpu)?;
                    if !same {
                        println!("    n={n} all_ef={all_ef} step {step} lane {i} differs");
                    }
                    all &= same;
                }
                live |= bytes(gpu, &refs[0].out)?.iter().any(|&b| b != 0);
                // Keep the per-step inputs of the last step for the controls.
                if step + 1 == STEPS {
                    // Negative control 1: one flipped bit in lane n-1's V input.
                    let victim = n - 1;
                    let ctrl: Vec<GdnState> =
                        twin.iter().map(|s| s.dup(gpu)).collect::<HipResult<_>>()?;
                    let pert: Vec<GdnState> =
                        twin.iter().map(|s| s.dup(gpu)).collect::<HipResult<_>>()?;
                    let v_clone = up_raw(gpu, &bytes(gpu, &ios[victim].v)?)?;
                    let mut vb = bytes(gpu, &v_clone)?;
                    vb[4 * 7 + 1] ^= 0x04;
                    gpu.hip.memcpy_htod(&v_clone.buf, &vb)?;
                    let run = |gpu: &mut Gpu,
                               set: &[GdnState],
                               vv: Option<&GpuTensor>|
                     -> HipResult<()> {
                        let rows: Vec<GdnRow> = (0..n)
                            .map(|i| {
                                let mut row = set[i].row(&ios[i]);
                                if let (Some(vv), true) = (vv, i == victim) {
                                    row.v = p(vv);
                                }
                                row
                            })
                            .collect();
                        gpu.stage_gated_delta_net_q8_compact2_b2_rows(
                            &table,
                            &rows,
                            &tags(n),
                            &tags(n),
                            1,
                            NV,
                        )?;
                        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(f + 100);
                        gpu.gated_delta_net_q8_compact2_b2_rows(&table, n, 1, NV, HD)
                    };
                    run(gpu, &ctrl, None)?;
                    run(gpu, &pert, Some(&v_clone))?;
                    sync(gpu)?;
                    let mut ok = true;
                    for i in 0..n {
                        let differs = ctrl[i].all_bytes(gpu)? != pert[i].all_bytes(gpu)?;
                        ok &= differs == (i == victim);
                    }
                    c.check(
                        ok,
                        &format!("gdn n={n} all_ef={all_ef}: negative control — one flipped V bit changes exactly that lane"),
                    );
                    // Negative control 2: wrong frame. Exactly the no-EF lanes
                    // (stochastic rounding) change; EF lanes round
                    // deterministically.
                    let shifted: Vec<GdnState> =
                        twin.iter().map(|s| s.dup(gpu)).collect::<HipResult<_>>()?;
                    let ctrl2: Vec<GdnState> =
                        twin.iter().map(|s| s.dup(gpu)).collect::<HipResult<_>>()?;
                    for (set, fr) in [(&ctrl2, f + 100), (&shifted, f + 101)] {
                        let rows: Vec<GdnRow> = (0..n).map(|i| set[i].row(&ios[i])).collect();
                        gpu.stage_gated_delta_net_q8_compact2_b2_rows(
                            &table,
                            &rows,
                            &tags(n),
                            &tags(n),
                            1,
                            NV,
                        )?;
                        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(fr);
                        gpu.gated_delta_net_q8_compact2_b2_rows(&table, n, 1, NV, HD)?;
                    }
                    sync(gpu)?;
                    let mut frame_ok = true;
                    for i in 0..n {
                        let changed = ctrl2[i].all_bytes(gpu)? != shifted[i].all_bytes(gpu)?;
                        frame_ok &= changed != refs[i].ef.is_some();
                    }
                    // A shifted frame moves lane i from frame f+i to f+i+1,
                    // i.e. onto lane i+1's frame: still a different frame for
                    // every lane, so exactly the stochastic lanes change.
                    c.check(
                        frame_ok,
                        &format!("gdn n={n} all_ef={all_ef}: negative control — frame +1 changes exactly the no-EF lanes (frame binding exercised)"),
                    );
                    for s in ctrl.into_iter().chain(pert).chain(ctrl2).chain(shifted) {
                        s.free(gpu)?;
                    }
                    gpu.free_tensor(v_clone)?;
                }
                for io in ios {
                    io.free(gpu)?;
                }
            }
            c.check(
                all,
                &format!("gdn n={n} all_ef={all_ef}: {STEPS} steps, output + S_q8 + scales + EF bitwise == per-lane singleton"),
            );
            c.check(
                live,
                &format!("gdn n={n} all_ef={all_ef}: singleton output non-zero (not vacuous)"),
            );
            if n >= 2 {
                let io = GdnIo::new(gpu, &mut r, false)?;
                let rows: Vec<GdnRow> = (0..n).map(|_| twin[0].row(&io)).collect();
                c.check(
                    gpu.stage_gated_delta_net_q8_compact2_b2_rows(
                        &table,
                        &rows,
                        &tags(n),
                        &tags(n),
                        1,
                        NV,
                    )
                    .is_err(),
                    &format!("gdn n={n}: every row on one state refused"),
                );
                let mut rows: Vec<GdnRow> = (0..n).map(|i| twin[i].row(&io)).collect();
                rows[1].output = rows[0].output + 4;
                c.check(
                    gpu.stage_gated_delta_net_q8_compact2_b2_rows(
                        &table,
                        &rows,
                        &tags(n),
                        &tags(n),
                        1,
                        NV,
                    )
                    .is_err(),
                    &format!("gdn n={n}: overlapping output refused"),
                );
                let mut rows: Vec<GdnRow> = (0..n).map(|i| twin[i].row(&io)).collect();
                let mut wrong = tags(n);
                wrong.swap(0, 1);
                c.check(
                    gpu.stage_gated_delta_net_q8_compact2_b2_rows(
                        &table,
                        &rows,
                        &wrong,
                        &tags(n),
                        1,
                        NV,
                    )
                    .is_err(),
                    &format!("gdn n={n}: wrong-row tags refused"),
                );
                rows[n - 1].v = 0;
                c.check(
                    gpu.stage_gated_delta_net_q8_compact2_b2_rows(
                        &table,
                        &rows,
                        &tags(n),
                        &tags(n),
                        1,
                        NV,
                    )
                    .is_err(),
                    &format!("gdn n={n}: null input refused"),
                );
                io.free(gpu)?;
            }
            // The compact2 twin must not admit a non-compact2 head count.
            c.check(
                gpu.gated_delta_net_q8_compact2_b2_rows(&table, n, 1, 33, HD)
                    .is_err(),
                &format!("gdn n={n}: odd head count refused"),
            );
            gpu.free_tensor(table)?;
            for s in refs.into_iter().chain(twin) {
                s.free(gpu)?;
            }
        }
    }
    Ok(())
}

// ── gated norm + MQ rotate (32 heads, no AWQ) ──────────────────────────────

fn gated_norm(gpu: &mut Gpu, c: &mut Checks) -> HipResult<()> {
    println!("=== gated_norm_mq_rotate_gfx1201 rows twin (A3B 32 x 128, no AWQ) ===");
    let mut r = Lcg(0x6A7F);
    let w = up_f(gpu, &r.vec(HD, 0.3, 1.0))?;
    for n in LANES {
        // Rows: random, then adversarial (huge / zero / tiny x, +-30 z), in
        // one batch: lane i uses its own data.
        let xs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                if i % 3 == 2 {
                    r.adversarial(V_DIM, 2.0)
                } else {
                    r.vec(V_DIM, 2.0, 0.0)
                }
            })
            .collect();
        let zs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut z = r.vec(V_DIM, 3.0, 0.0);
                if i % 3 == 2 {
                    z[0] = 30.0;
                    z[1] = -30.0;
                    z[2] = -100.0;
                    z[3] = 100.0;
                }
                z
            })
            .collect();
        let xt: Vec<GpuTensor> = xs.iter().map(|x| up_f(gpu, x)).collect::<HipResult<_>>()?;
        let zt: Vec<GpuTensor> = zs.iter().map(|z| up_f(gpu, z)).collect::<HipResult<_>>()?;
        let mk = |gpu: &mut Gpu| -> HipResult<Vec<GpuTensor>> {
            (0..n).map(|_| up_f(gpu, &vec![0.0; V_DIM])).collect()
        };
        let (refs, twin) = (mk(gpu)?, mk(gpu)?);
        for i in 0..n {
            gpu.gated_norm_rotate_mq_gfx1100(&xt[i], &zt[i], &w, None, &refs[i], NV, HD, EPS)?;
        }
        let table = staging(gpu, n * 24)?;
        let rows: Vec<GatedNormRow> = (0..n)
            .map(|i| GatedNormRow {
                x: p(&xt[i]),
                z: p(&zt[i]),
                x_rot: p(&twin[i]),
            })
            .collect();
        gpu.stage_gated_norm_mq_rotate_k4096_rows(&table, &rows, &tags(n), &tags(n))?;
        gpu.gated_norm_mq_rotate_k4096_rows(&table, n, &w, EPS)?;
        sync(gpu)?;
        let mut all = true;
        for i in 0..n {
            all &= bytes(gpu, &refs[i])? == bytes(gpu, &twin[i])?;
        }
        c.check(
            all,
            &format!("gated norm n={n}: every lane bitwise == per-lane singleton"),
        );
        c.check(
            bytes(gpu, &refs[0])?.iter().any(|&b| b != 0),
            &format!("gated norm n={n}: singleton x_rot non-zero (not vacuous)"),
        );
        // Several steps with state-free kernel: repeated launch into the same
        // outputs stays equal (stale-output / scratch independence).
        for step in 0..STEPS {
            let victim_x = r.vec(V_DIM, 2.0, 0.0);
            for i in 0..n {
                let x = if (i + step) % 2 == 0 {
                    victim_x.clone()
                } else {
                    r.vec(V_DIM, 2.0, 0.0)
                };
                overwrite_f(gpu, &xt[i], &x)?;
                gpu.gated_norm_rotate_mq_gfx1100(&xt[i], &zt[i], &w, None, &refs[i], NV, HD, EPS)?;
            }
            gpu.gated_norm_mq_rotate_k4096_rows(&table, n, &w, EPS)?;
            sync(gpu)?;
            let mut ok = true;
            for i in 0..n {
                ok &= bytes(gpu, &refs[i])? == bytes(gpu, &twin[i])?;
            }
            c.check(
                ok,
                &format!("gated norm n={n} step {step}: bitwise == singleton"),
            );
        }
        // Negative control: one flipped byte of the last lane's z.
        let victim = n - 1;
        let ctrl: Vec<GpuTensor> = mk(gpu)?;
        let pert: Vec<GpuTensor> = mk(gpu)?;
        let mut zb = bytes(gpu, &zt[victim])?;
        zb[4 * 9 + 2] ^= 0x40;
        let zp = up_raw(gpu, &zb)?;
        for (set, zv) in [(&ctrl, None), (&pert, Some(&zp))] {
            let rows: Vec<GatedNormRow> = (0..n)
                .map(|i| GatedNormRow {
                    x: p(&xt[i]),
                    z: if i == victim {
                        zv.map(p).unwrap_or(p(&zt[i]))
                    } else {
                        p(&zt[i])
                    },
                    x_rot: p(&set[i]),
                })
                .collect();
            gpu.stage_gated_norm_mq_rotate_k4096_rows(&table, &rows, &tags(n), &tags(n))?;
            gpu.gated_norm_mq_rotate_k4096_rows(&table, n, &w, EPS)?;
        }
        sync(gpu)?;
        let mut ok = true;
        for i in 0..n {
            ok &= (bytes(gpu, &ctrl[i])? != bytes(gpu, &pert[i])?) == (i == victim);
        }
        c.check(
            ok,
            &format!(
                "gated norm n={n}: negative control — one flipped z byte changes exactly that lane"
            ),
        );
        if n >= 2 {
            let mut alias = rows.clone();
            alias[1].x_rot = alias[0].x_rot;
            c.check(
                gpu.stage_gated_norm_mq_rotate_k4096_rows(&table, &alias, &tags(n), &tags(n))
                    .is_err(),
                &format!("gated norm n={n}: shared output refused"),
            );
            let mut wrong = tags(n);
            wrong[0] = 1;
            c.check(
                gpu.stage_gated_norm_mq_rotate_k4096_rows(&table, &rows, &wrong, &tags(n))
                    .is_err(),
                &format!("gated norm n={n}: wrong-row tag refused"),
            );
        }
        let mut all_t = vec![table, zp];
        all_t.extend(xt);
        all_t.extend(zt);
        all_t.extend(refs);
        all_t.extend(twin);
        all_t.extend(ctrl);
        all_t.extend(pert);
        free_all(gpu, all_t)?;
    }
    gpu.free_tensor(w)
}

#[test]
#[ignore]
fn a3b_dn_rows_twins_match_singleton() {
    let mut gpu = Gpu::init().expect("GPU required");
    println!("arch={}", gpu.arch);
    assert!(
        gpu.arch_caps.is_gfx1201(),
        "the A3B DeltaNet rows twins are exact-gfx1201 only"
    );
    let mut c = Checks {
        pass: 0,
        fails: vec![],
    };
    type Gate = fn(&mut Gpu, &mut Checks) -> HipResult<()>;
    let gates: [(&str, Gate); 4] = [
        ("sigmoid_gate", sigmoid_gate),
        ("conv", conv),
        ("gdn", gdn),
        ("gated_norm", gated_norm),
    ];
    for (name, f) in gates {
        if let Err(e) = f(&mut gpu, &mut c) {
            c.check(false, &format!("{name} aborted: {e}"));
        }
    }
    println!("A3B DN ROWS: {} PASS, {} FAIL", c.pass, c.fails.len());
    assert!(c.fails.is_empty(), "failed checks: {:#?}", c.fails);
}
