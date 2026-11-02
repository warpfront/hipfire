// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Gate for the rows-batched twins (rdna_compute::attention::rows_batched):
// for each twin and k = 1, 2, 4, 8, every row's outputs AND updated state are
// bitwise equal to the singleton kernel launched once per row on an identical
// copy of that row's (warmed, real-layout) state. Plus negative controls:
// wrong row tag, a row aliasing another row's state/output, null pointers,
// undersized staging; and a row-permuted table (stateless/conv twins) to
// prove row routing comes from the table. Qwen3.6-27B gfx1201 shapes.
//
// Build: cargo build -p rdna-compute --release --features deltanet --example cb_rows_twins_gate

#[cfg(feature = "deltanet")]
mod gate {
    use hip_bridge::HipResult;
    use rdna_compute::attention::rows_batched::{ConvRow, FaPrepRow, GatedNormRow, GdnRow};
    use rdna_compute::{DType, Gpu, GpuTensor};

    const KS: [usize; 4] = [1, 2, 4, 8];

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
    }

    fn half_bits(x: f32) -> u16 {
        // Small finite values only: truncating f32 -> f16 conversion.
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        if exp <= 0 {
            return sign;
        }
        sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
    }

    pub struct Checks {
        pub pass: usize,
        pub fails: Vec<String>,
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
    fn dup(gpu: &mut Gpu, t: &GpuTensor) -> HipResult<GpuTensor> {
        let bytes = gpu.download_raw_bytes(t)?;
        gpu.upload_raw(&bytes, &[bytes.len()])
    }
    fn bytes(gpu: &Gpu, t: &GpuTensor) -> HipResult<Vec<u8>> {
        gpu.download_raw_bytes(t)
    }
    fn p(t: &GpuTensor) -> u64 {
        t.buf.as_ptr() as u64
    }
    fn free_all(gpu: &mut Gpu, ts: Vec<GpuTensor>) -> HipResult<()> {
        for t in ts {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
    fn staging(gpu: &Gpu, bytes: usize) -> HipResult<GpuTensor> {
        gpu.upload_raw(&vec![0u8; bytes], &[bytes])
    }
    fn tags(k: usize) -> Vec<u64> {
        (0..k as u64).map(|i| 1000 + i).collect()
    }

    // ── GDN compact3 ───────────────────────────────────────────────────────
    const GH: usize = 48;
    const GQK: usize = 16;
    const HD: usize = 128;

    struct GdnIo {
        q: GpuTensor,
        k: GpuTensor,
        v: GpuTensor,
        gate: GpuTensor,
        beta: GpuTensor,
    }
    struct GdnState {
        sq: GpuTensor,
        sc: GpuTensor,
        ef: Option<GpuTensor>,
        out: GpuTensor,
    }

    fn gdn_io(gpu: &mut Gpu, r: &mut Lcg) -> HipResult<GdnIo> {
        Ok(GdnIo {
            q: up_f(gpu, &r.vec(GQK * HD, 0.09, 0.0))?,
            k: up_f(gpu, &r.vec(GQK * HD, 0.09, 0.0))?,
            v: up_f(gpu, &r.vec(GH * HD, 1.0, 0.0))?,
            gate: up_f(gpu, &r.vec(GH, 0.4, -0.5))?,
            beta: up_f(gpu, &r.vec(GH, 0.4, 0.5))?,
        })
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
            GH,
            HD,
            3,
            st.ef.as_ref(),
        )
    }

    fn gdn_state_bytes(gpu: &Gpu, s: &GdnState) -> HipResult<Vec<u8>> {
        let mut b = bytes(gpu, &s.sq)?;
        b.extend(bytes(gpu, &s.sc)?);
        if let Some(e) = &s.ef {
            b.extend(bytes(gpu, e)?);
        }
        b.extend(bytes(gpu, &s.out)?);
        Ok(b)
    }

    fn gdn_dup(gpu: &mut Gpu, s: &GdnState) -> HipResult<GdnState> {
        Ok(GdnState {
            sq: dup(gpu, &s.sq)?,
            sc: dup(gpu, &s.sc)?,
            ef: match &s.ef {
                Some(e) => Some(dup(gpu, e)?),
                None => None,
            },
            out: dup(gpu, &s.out)?,
        })
    }

    fn gdn_row(io: &GdnIo, s: &GdnState) -> GdnRow {
        GdnRow {
            q: p(&io.q),
            k: p(&io.k),
            v: p(&io.v),
            gate: p(&io.gate),
            beta: p(&io.beta),
            s_q8: p(&s.sq),
            s_scales: p(&s.sc),
            output: p(&s.out),
            s_ef_residual: s.ef.as_ref().map(p).unwrap_or(0),
        }
    }

    fn free_gdn_state(gpu: &mut Gpu, s: GdnState) -> HipResult<()> {
        let mut v = vec![s.sq, s.sc, s.out];
        v.extend(s.ef);
        free_all(gpu, v)
    }

    pub fn gdn(gpu: &mut Gpu, c: &mut Checks) -> HipResult<()> {
        println!("=== gated_delta_net_q8_compact3_b2 rows twin ===");
        let mut r = Lcg(0x6D11);
        for k in KS {
            // Per-row real state: random Q8 state, warmed by 3 singleton steps.
            let mut ios = vec![];
            let mut states = vec![];
            for row in 0..k {
                let st = GdnState {
                    sq: up_raw(gpu, &r.bytes(GH * HD * HD))?,
                    sc: up_f(gpu, &r.vec(GH * HD, 0.004, 0.006))?,
                    ef: if row % 2 == 1 {
                        Some(up_raw(gpu, &r.halves(GH * HD * HD))?)
                    } else {
                        None
                    },
                    out: up_f(gpu, &vec![0.0; GH * HD])?,
                };
                for _ in 0..3 {
                    let warm = gdn_io(gpu, &mut r)?;
                    gdn_single(gpu, &warm, &st)?;
                    free_all(gpu, vec![warm.q, warm.k, warm.v, warm.gate, warm.beta])?;
                }
                ios.push(gdn_io(gpu, &mut r)?);
                states.push(st);
            }
            let twin: Vec<GdnState> = states
                .iter()
                .map(|s| gdn_dup(gpu, s))
                .collect::<HipResult<_>>()?;
            let shifted: Vec<GdnState> = states
                .iter()
                .map(|s| gdn_dup(gpu, s))
                .collect::<HipResult<_>>()?;
            // Reference: per-row singleton launches in row order, frames F, F+1, ...
            let frame0 = rdna_compute::norm::gdn_requant_frame_checkpoint();
            for (io, st) in ios.iter().zip(&states) {
                gdn_single(gpu, io, st)?;
            }
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame0);
            let rows: Vec<GdnRow> = ios
                .iter()
                .zip(&twin)
                .map(|(io, s)| gdn_row(io, s))
                .collect();
            let table = staging(gpu, rows.len() * 72)?;
            gpu.stage_gated_delta_net_q8_compact3_b2_rows(
                &table,
                &rows,
                &tags(k),
                &tags(k),
                1,
                GH,
            )?;
            gpu.gated_delta_net_q8_compact3_b2_rows(&table, k, 1, GH, HD)?;
            gpu.hip.device_synchronize()?;
            let mut all = true;
            for row in 0..k {
                let same = gdn_state_bytes(gpu, &states[row])? == gdn_state_bytes(gpu, &twin[row])?;
                if !same {
                    println!("    k={k} row {row} differs");
                }
                all &= same;
            }
            c.check(
                all,
                &format!(
                    "gdn k={k}: every row's output+S_q8+scales+EF bitwise == per-row singleton"
                ),
            );
            let live = bytes(gpu, &states[0].out)?.iter().any(|&b| b != 0);
            c.check(
                live,
                &format!("gdn k={k}: singleton output is non-zero (comparison not vacuous)"),
            );
            // Wrong-frame control: the same twin launched one frame late.
            // Stochastic-rounding rows (no EF, even rows) must change, so the
            // per-row frame binding is really tested; EF rows round
            // deterministically and must not.
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame0 + 1);
            let srows: Vec<GdnRow> = ios
                .iter()
                .zip(&shifted)
                .map(|(io, s)| gdn_row(io, s))
                .collect();
            gpu.stage_gated_delta_net_q8_compact3_b2_rows(
                &table,
                &srows,
                &tags(k),
                &tags(k),
                1,
                GH,
            )?;
            gpu.gated_delta_net_q8_compact3_b2_rows(&table, k, 1, GH, HD)?;
            gpu.hip.device_synchronize()?;
            let mut frame_ok = true;
            for row in 0..k {
                let changed =
                    gdn_state_bytes(gpu, &states[row])? != gdn_state_bytes(gpu, &shifted[row])?;
                frame_ok &= changed == states[row].ef.is_none();
            }
            c.check(
                frame_ok,
                &format!("gdn k={k}: frame shifted by 1 changes exactly the stochastic-rounding rows (frame binding exercised)"),
            );
            for s in shifted {
                free_gdn_state(gpu, s)?;
            }
            if k >= 2 {
                let mut alias = rows.clone();
                alias[1].s_q8 = alias[0].s_q8;
                c.check(
                    gpu.stage_gated_delta_net_q8_compact3_b2_rows(
                        &table,
                        &alias,
                        &tags(k),
                        &tags(k),
                        1,
                        GH,
                    )
                    .is_err(),
                    &format!("gdn k={k}: row 1 pointed at row 0's state refused"),
                );
                let mut alias = rows.clone();
                alias[1].output = alias[0].output + 4;
                c.check(
                    gpu.stage_gated_delta_net_q8_compact3_b2_rows(
                        &table,
                        &alias,
                        &tags(k),
                        &tags(k),
                        1,
                        GH,
                    )
                    .is_err(),
                    &format!("gdn k={k}: overlapping output refused"),
                );
                let mut wrong = tags(k);
                wrong.swap(0, 1);
                c.check(
                    gpu.stage_gated_delta_net_q8_compact3_b2_rows(
                        &table,
                        &rows,
                        &wrong,
                        &tags(k),
                        1,
                        GH,
                    )
                    .is_err(),
                    &format!("gdn k={k}: wrong-row tags refused"),
                );
                let mut null = rows.clone();
                null[k - 1].v = 0;
                c.check(
                    gpu.stage_gated_delta_net_q8_compact3_b2_rows(
                        &table,
                        &null,
                        &tags(k),
                        &tags(k),
                        1,
                        GH,
                    )
                    .is_err(),
                    &format!("gdn k={k}: null input refused"),
                );
                let small = staging(gpu, (k - 1) * 72)?;
                c.check(
                    gpu.stage_gated_delta_net_q8_compact3_b2_rows(
                        &small,
                        &rows,
                        &tags(k),
                        &tags(k),
                        1,
                        GH,
                    )
                    .is_err(),
                    &format!("gdn k={k}: undersized row-table staging refused"),
                );
                gpu.free_tensor(small)?;
            }
            gpu.free_tensor(table)?;
            for io in ios {
                free_all(gpu, vec![io.q, io.k, io.v, io.gate, io.beta])?;
            }
            for s in states.into_iter().chain(twin) {
                free_gdn_state(gpu, s)?;
            }
        }
        Ok(())
    }

    // ── conv1d + SiLU + split + QK norm (b256) ─────────────────────────────
    const CK: usize = GQK * HD; // 2048
    const CV: usize = GH * HD; // 6144
    const CC: usize = 2 * CK + CV;

    struct ConvBufs {
        q: GpuTensor,
        k: GpuTensor,
        v: GpuTensor,
        input: GpuTensor,
        state: GpuTensor,
    }

    fn conv_single(gpu: &mut Gpu, b: &ConvBufs, w: &GpuTensor) -> HipResult<()> {
        gpu.conv1d_silu_split_qknorm(
            &b.q,
            &b.k,
            &b.v,
            &b.input,
            w,
            &b.state,
            CK,
            CV,
            GQK,
            HD,
            1.0 / (HD as f32).sqrt(),
            1e-6,
        )
    }
    fn conv_bytes(gpu: &Gpu, b: &ConvBufs) -> HipResult<Vec<u8>> {
        let mut o = bytes(gpu, &b.q)?;
        o.extend(bytes(gpu, &b.k)?);
        o.extend(bytes(gpu, &b.v)?);
        o.extend(bytes(gpu, &b.state)?);
        Ok(o)
    }
    fn conv_dup(gpu: &mut Gpu, b: &ConvBufs) -> HipResult<ConvBufs> {
        Ok(ConvBufs {
            q: dup(gpu, &b.q)?,
            k: dup(gpu, &b.k)?,
            v: dup(gpu, &b.v)?,
            input: dup(gpu, &b.input)?,
            state: dup(gpu, &b.state)?,
        })
    }
    fn conv_row(b: &ConvBufs) -> ConvRow {
        ConvRow {
            q_out: p(&b.q),
            k_out: p(&b.k),
            v_out: p(&b.v),
            input: p(&b.input),
            state: p(&b.state),
        }
    }

    pub fn conv(gpu: &mut Gpu, c: &mut Checks) -> HipResult<()> {
        println!("=== conv1d_silu_split_qknorm_b256 rows twin ===");
        let mut r = Lcg(0xC0B);
        let w = up_f(gpu, &r.vec(CC * 4, 0.5, 0.0))?;
        for k in KS {
            let mut refs = vec![];
            for _ in 0..k {
                let b = ConvBufs {
                    q: up_f(gpu, &vec![0.0; CK])?,
                    k: up_f(gpu, &vec![0.0; CK])?,
                    v: up_f(gpu, &vec![0.0; CV])?,
                    input: up_f(gpu, &r.vec(CC, 1.0, 0.0))?,
                    state: up_f(gpu, &r.vec(CC * 3, 1.0, 0.0))?,
                };
                // Warm the conv ring with two real steps.
                for _ in 0..2 {
                    conv_single(gpu, &b, &w)?;
                    let fresh = up_f(gpu, &r.vec(CC, 1.0, 0.0))?;
                    gpu.hip.memcpy_dtod(&b.input.buf, &fresh.buf, CC * 4)?;
                    gpu.free_tensor(fresh)?;
                }
                refs.push(b);
            }
            let twin: Vec<ConvBufs> = refs
                .iter()
                .map(|b| conv_dup(gpu, b))
                .collect::<HipResult<_>>()?;
            let perm: Vec<ConvBufs> = refs
                .iter()
                .map(|b| conv_dup(gpu, b))
                .collect::<HipResult<_>>()?;
            for b in &refs {
                conv_single(gpu, b, &w)?;
            }
            let table = staging(gpu, k * 40)?;
            let rows: Vec<ConvRow> = twin.iter().map(conv_row).collect();
            gpu.stage_conv1d_silu_split_qknorm_b256_rows(
                &table,
                &rows,
                &tags(k),
                &tags(k),
                CK,
                CV,
            )?;
            gpu.conv1d_silu_split_qknorm_b256_rows(
                &table,
                k,
                &w,
                CK,
                CV,
                GQK,
                HD,
                1.0 / (HD as f32).sqrt(),
                1e-6,
            )?;
            let prow: Vec<ConvRow> = perm.iter().rev().map(conv_row).collect();
            gpu.stage_conv1d_silu_split_qknorm_b256_rows(
                &table,
                &prow,
                &tags(k),
                &tags(k),
                CK,
                CV,
            )?;
            gpu.conv1d_silu_split_qknorm_b256_rows(
                &table,
                k,
                &w,
                CK,
                CV,
                GQK,
                HD,
                1.0 / (HD as f32).sqrt(),
                1e-6,
            )?;
            gpu.hip.device_synchronize()?;
            let (mut all, mut all_p) = (true, true);
            for row in 0..k {
                let want = conv_bytes(gpu, &refs[row])?;
                all &= want == conv_bytes(gpu, &twin[row])?;
                all_p &= want == conv_bytes(gpu, &perm[row])?;
            }
            c.check(
                all,
                &format!("conv k={k}: every row's q/k/v + conv state bitwise == per-row singleton"),
            );
            let live = bytes(gpu, &refs[0].q)?.iter().any(|&b| b != 0);
            c.check(
                live,
                &format!("conv k={k}: singleton q_out is non-zero (comparison not vacuous)"),
            );
            c.check(
                all_p,
                &format!("conv k={k}: reversed row table still bitwise per row"),
            );
            if k >= 2 {
                let mut alias = rows.clone();
                alias[1].state = alias[0].state;
                c.check(
                    gpu.stage_conv1d_silu_split_qknorm_b256_rows(
                        &table,
                        &alias,
                        &tags(k),
                        &tags(k),
                        CK,
                        CV,
                    )
                    .is_err(),
                    &format!("conv k={k}: shared conv state refused"),
                );
                let mut wrong = tags(k);
                wrong[k - 1] += 77;
                c.check(
                    gpu.stage_conv1d_silu_split_qknorm_b256_rows(
                        &table,
                        &rows,
                        &wrong,
                        &tags(k),
                        CK,
                        CV,
                    )
                    .is_err(),
                    &format!("conv k={k}: wrong-row tag refused"),
                );
            }
            gpu.free_tensor(table)?;
            for b in refs.into_iter().chain(twin).chain(perm) {
                free_all(gpu, vec![b.q, b.k, b.v, b.input, b.state])?;
            }
        }
        gpu.free_tensor(w)?;
        Ok(())
    }

    // ── gated norm + MQ rotate AWQ k6144 ───────────────────────────────────
    pub fn gated_norm(gpu: &mut Gpu, c: &mut Checks) -> HipResult<()> {
        println!("=== gated_norm_mq_rotate_awq_k6144_gfx1201 rows twin ===");
        let mut r = Lcg(0x6A7E);
        let n = GH * HD;
        let w = up_f(gpu, &r.vec(HD, 0.3, 1.0))?;
        let awq = up_f(gpu, &r.vec(n, 0.4, 1.0))?;
        for k in KS {
            let xs: Vec<GpuTensor> = (0..k)
                .map(|_| up_f(gpu, &r.vec(n, 2.0, 0.0)))
                .collect::<HipResult<_>>()?;
            let zs: Vec<GpuTensor> = (0..k)
                .map(|_| up_f(gpu, &r.vec(n, 3.0, 0.0)))
                .collect::<HipResult<_>>()?;
            let mk = |gpu: &mut Gpu| -> HipResult<Vec<GpuTensor>> {
                (0..k).map(|_| up_f(gpu, &vec![0.0; n])).collect()
            };
            let (refs, twin, perm) = (mk(gpu)?, mk(gpu)?, mk(gpu)?);
            for i in 0..k {
                gpu.gated_norm_rotate_mq_gfx1100(
                    &xs[i],
                    &zs[i],
                    &w,
                    Some(&awq),
                    &refs[i],
                    GH,
                    HD,
                    1e-6,
                )?;
            }
            let table = staging(gpu, k * 24)?;
            let rows: Vec<GatedNormRow> = (0..k)
                .map(|i| GatedNormRow {
                    x: p(&xs[i]),
                    z: p(&zs[i]),
                    x_rot: p(&twin[i]),
                })
                .collect();
            gpu.stage_gated_norm_mq_rotate_awq_k6144_rows(&table, &rows, &tags(k), &tags(k))?;
            gpu.gated_norm_mq_rotate_awq_k6144_rows(&table, k, &w, &awq, 1e-6)?;
            let prow: Vec<GatedNormRow> = (0..k)
                .rev()
                .map(|i| GatedNormRow {
                    x: p(&xs[i]),
                    z: p(&zs[i]),
                    x_rot: p(&perm[i]),
                })
                .collect();
            gpu.stage_gated_norm_mq_rotate_awq_k6144_rows(&table, &prow, &tags(k), &tags(k))?;
            gpu.gated_norm_mq_rotate_awq_k6144_rows(&table, k, &w, &awq, 1e-6)?;
            gpu.hip.device_synchronize()?;
            let (mut all, mut all_p) = (true, true);
            for i in 0..k {
                let want = bytes(gpu, &refs[i])?;
                all &= want == bytes(gpu, &twin[i])?;
                all_p &= want == bytes(gpu, &perm[i])?;
            }
            c.check(
                all,
                &format!("gated norm k={k}: every row bitwise == per-row singleton"),
            );
            let live = bytes(gpu, &refs[0])?.iter().any(|&b| b != 0);
            c.check(
                live,
                &format!("gated norm k={k}: singleton x_rot is non-zero (comparison not vacuous)"),
            );
            c.check(
                all_p,
                &format!("gated norm k={k}: reversed row table still bitwise per row"),
            );
            if k >= 2 {
                let mut alias = rows.clone();
                alias[1].x_rot = alias[0].x_rot;
                c.check(
                    gpu.stage_gated_norm_mq_rotate_awq_k6144_rows(
                        &table,
                        &alias,
                        &tags(k),
                        &tags(k),
                    )
                    .is_err(),
                    &format!("gated norm k={k}: shared output refused"),
                );
                let mut wrong = tags(k);
                wrong[0] = 1;
                c.check(
                    gpu.stage_gated_norm_mq_rotate_awq_k6144_rows(&table, &rows, &wrong, &tags(k))
                        .is_err(),
                    &format!("gated norm k={k}: wrong-row tag refused"),
                );
            }
            let mut all_t = vec![table];
            all_t.extend(xs);
            all_t.extend(zs);
            all_t.extend(refs);
            all_t.extend(twin);
            all_t.extend(perm);
            free_all(gpu, all_t)?;
        }
        free_all(gpu, vec![w, awq])
    }

    // ── Qwen3.6-27B FA prep ────────────────────────────────────────────────
    pub fn fa_prep(gpu: &mut Gpu, c: &mut Checks) -> HipResult<()> {
        println!("=== qwen36_27b_fa_prep_gfx1201 rows twin ===");
        let mut r = Lcg(0xFA9);
        let qw = up_f(gpu, &r.vec(256, 0.3, 1.0))?;
        let kw = up_f(gpu, &r.vec(256, 0.3, 1.0))?;
        let (eps, theta) = (1e-6f32, 10_000_000.0f32);
        for k in KS {
            let mut qi = vec![];
            let mut pos = vec![];
            let mut k_init = vec![];
            for i in 0..k {
                qi.push(up_f(gpu, &r.vec(24 * 512, 2.0, 0.0))?);
                let pv = (7 + 9_973 * i as i32 * (i as i32 + 1)).to_le_bytes();
                pos.push(up_raw(gpu, &pv)?);
                k_init.push(r.vec(4 * 256, 2.0, 0.0));
            }
            struct Out {
                q: GpuTensor,
                g: GpuTensor,
                k: GpuTensor,
            }
            let mut mk = |gpu: &mut Gpu| -> HipResult<Vec<Out>> {
                (0..k)
                    .map(|i| {
                        Ok(Out {
                            q: up_f(gpu, &vec![0.0; 24 * 256])?,
                            g: up_f(gpu, &vec![0.0; 24 * 256])?,
                            k: up_f(gpu, &k_init[i])?,
                        })
                    })
                    .collect()
            };
            let (refs, twin, perm) = (mk(gpu)?, mk(gpu)?, mk(gpu)?);
            for i in 0..k {
                gpu.qwen35_fa_prep_gfx1100(
                    &qi[i],
                    &refs[i].q,
                    &refs[i].g,
                    &refs[i].k,
                    &qw,
                    &kw,
                    &pos[i].buf,
                    eps,
                    theta,
                    24,
                    4,
                )?;
            }
            let row = |i: usize, o: &Out| FaPrepRow {
                q_interleaved: p(&qi[i]),
                q: p(&o.q),
                gate: p(&o.g),
                k: p(&o.k),
                pos: p(&pos[i]),
            };
            let table = staging(gpu, k * 40)?;
            let rows: Vec<FaPrepRow> = (0..k).map(|i| row(i, &twin[i])).collect();
            gpu.stage_qwen36_27b_fa_prep_rows(&table, &rows, &tags(k), &tags(k))?;
            gpu.qwen36_27b_fa_prep_rows(&table, k, &qw, &kw, eps, theta)?;
            let prow: Vec<FaPrepRow> = (0..k).rev().map(|i| row(i, &perm[i])).collect();
            gpu.stage_qwen36_27b_fa_prep_rows(&table, &prow, &tags(k), &tags(k))?;
            gpu.qwen36_27b_fa_prep_rows(&table, k, &qw, &kw, eps, theta)?;
            gpu.hip.device_synchronize()?;
            let ob = |gpu: &Gpu, o: &Out| -> HipResult<Vec<u8>> {
                let mut b = bytes(gpu, &o.q)?;
                b.extend(bytes(gpu, &o.g)?);
                b.extend(bytes(gpu, &o.k)?);
                Ok(b)
            };
            let (mut all, mut all_p) = (true, true);
            for i in 0..k {
                let want = ob(gpu, &refs[i])?;
                all &= want == ob(gpu, &twin[i])?;
                all_p &= want == ob(gpu, &perm[i])?;
            }
            c.check(all, &format!("fa prep k={k}: every row's q/gate/k (distinct positions) bitwise == per-row singleton"));
            let live = bytes(gpu, &refs[0].q)?.iter().any(|&b| b != 0);
            c.check(
                live,
                &format!("fa prep k={k}: singleton q is non-zero (comparison not vacuous)"),
            );
            c.check(
                all_p,
                &format!("fa prep k={k}: reversed row table still bitwise per row"),
            );
            if k >= 2 {
                let mut alias = rows.clone();
                alias[1].k = alias[0].k;
                c.check(
                    gpu.stage_qwen36_27b_fa_prep_rows(&table, &alias, &tags(k), &tags(k))
                        .is_err(),
                    &format!("fa prep k={k}: shared in-place K refused"),
                );
                let mut wrong = tags(k);
                wrong.reverse();
                c.check(
                    gpu.stage_qwen36_27b_fa_prep_rows(&table, &rows, &wrong, &tags(k))
                        .is_err(),
                    &format!("fa prep k={k}: wrong-row tags refused"),
                );
            }
            let mut all_t = vec![table];
            all_t.extend(qi);
            all_t.extend(pos);
            for o in refs.into_iter().chain(twin).chain(perm) {
                all_t.extend([o.q, o.g, o.k]);
            }
            free_all(gpu, all_t)?;
        }
        free_all(gpu, vec![qw, kw])
    }

    pub fn run() -> i32 {
        let mut gpu = Gpu::init().expect("gpu init");
        println!("arch={}", gpu.arch);
        let mut c = Checks {
            pass: 0,
            fails: vec![],
        };
        let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
        type Gate = fn(&mut Gpu, &mut Checks) -> HipResult<()>;
        let gates: [(&str, Gate); 4] = [
            ("gdn", gdn),
            ("conv", conv),
            ("gated_norm", gated_norm),
            ("fa_prep", fa_prep),
        ];
        for (name, f) in gates {
            if which != "all" && which != name {
                continue;
            }
            if let Err(e) = f(&mut gpu, &mut c) {
                c.check(false, &format!("{name} aborted: {e}"));
            }
        }
        println!("ROWS TWINS: {} PASS, {} FAIL", c.pass, c.fails.len());
        if c.fails.is_empty() {
            println!("ROWS TWINS RESULT: PASS");
            0
        } else {
            println!("ROWS TWINS RESULT: FAIL");
            1
        }
    }
}

fn main() {
    #[cfg(feature = "deltanet")]
    std::process::exit(gate::run());
    #[cfg(not(feature = "deltanet"))]
    {
        eprintln!("build with --features deltanet");
        std::process::exit(2);
    }
}
