// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! True cross-implementation check for the qt44 MoE kernels, device-resident
//! AND host-mapped.
//!
//! Every parity harness in the tree compares the device against the host —
//! dev-vs-host diff 0 passes even when both sides are equally wrong. This
//! example closes that hole for both halves of the routed expert FFN:
//! one real qt44 expert's bytes driven through the production gate_up +
//! down kernels, compared elementwise against `hipfire_cpu` over the same
//! bytes — the transcription proven bit-exact against the canonical decoder
//! on 24k real tensors. Each half runs twice: device-resident, then
//! host-mapped with the pointer table built from `sub_offset(slot*stride)`
//! views exactly as the loader builds it (the one combination nothing else
//! in the tree performs: independent reference AND real host placement).
//!
//! Usage:
//!     cargo run --release -p hipfire-arch-qwen35 --features lab --example mq4v2_moe_down_parity -- MODEL.hfq [FILTER]
//!
//! Requires `--features lab` (Gpu::init) and hipfire-cpu as a dev-dependency
//! (examples build with dev-dependencies).

use hipfire_arch_qwen35::qwen35::load::{load_weight_tensor, qwen35_tensor_name_candidates};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_backend::MemoryTarget;
use rdna_compute::{DType, Gpu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        std::env::var("HOME")
            .map(|h| format!("{h}/.hipfire/models/ornith-1.5-35b-a3b.mq4"))
            .unwrap_or_else(|_| "ornith-1.5-35b-a3b.mq4".into())
    });
    let filter: Option<String> = std::env::args().nth(2);
    println!("model: {path}");

    let hfq = HfqFile::open(std::path::Path::new(&path))?;
    let mut gpu = Gpu::init()?;

    // One real qt44 expert PAIR (gate_up donor + down): gate_up feeds the
    // gate_up kernel, whose silu+rotate output feeds the down kernel — the full
    // routed half this check covers. Both from expert 0 of layer 0.
    let want = filter.as_deref().unwrap_or(".mlp.experts.0.");
    let find = |frag: &str| {
        hfq.tensor_infos()
            .iter()
            .find(|t| {
                t.quant_type == 44
                    && t.shape.len() == 2
                    && t.name.contains(want)
                    && t.name.contains(frag)
            })
            .or_else(|| {
                hfq.tensor_infos().iter().find(|t| {
                    t.quant_type == 44
                        && t.name.contains(".mlp.experts.0.")
                        && t.name.contains(frag)
                })
            })
            .unwrap_or_else(|| panic!("no qt44 {frag} tensor matching {want}"))
            .name
            .clone()
    };
    let gu_name = find("gate_up_proj.weight");
    let dn_name = find("down_proj.weight");
    let gu_info = hfq.find_tensor_info(&gu_name).expect("gate_up info");
    let dn_info = hfq.find_tensor_info(&dn_name).expect("down info");
    let (gu_m, gu_k) = (gu_info.shape[0] as usize, gu_info.shape[1] as usize);
    let (m, k) = (dn_info.shape[0] as usize, dn_info.shape[1] as usize);
    println!("picked {gu_name} gate_up {gu_m}x{gu_k}; {dn_name} down {m}x{k}");
    assert_eq!(gu_m, 2 * k, "fused gate_up rows must be 2*mi");
    assert_eq!(gu_k, m, "gate_up k (dim) must equal down m (dim)");

    let load = |gpu: &mut Gpu, name: &str, m: usize, k: usize, target: MemoryTarget| {
        load_weight_tensor(&hfq, gpu, name, m, k, qwen35_tensor_name_candidates, target)
    };

    // Plain normalized activation (rotated once for gate_up, matching the
    // launcher's InputBasis stage).
    let mut rng: u32 = 0x1234_5678;
    let x_raw: Vec<f32> = (0..m)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let mut x_rot = x_raw.clone();
    hipfire_cpu::quant::rotate_x(&mut x_rot);

    // CPU reference for the FULL expert: gate_up GEMV + silu + hidden-rotate
    // + down GEMV (mirrors `run_experts` for one expert, weight 1.0).
    let cpu_expert = |gu_bytes: &[u8], dn_bytes: &[u8]| -> Vec<f32> {
        let q = hipfire_cpu::quant::CpuQuant::Mq4G256V2;
        let mi = k;
        let mut gate_up_out = vec![0.0f32; 2 * mi];
        let mut hidden = vec![0.0f32; mi];
        let mut down_out = vec![0.0f32; m];
        hipfire_cpu::gemv::gemv(q, gu_bytes, 2 * mi, m, &x_rot, &mut gate_up_out);
        hipfire_cpu::epilogue::silu_mul(&gate_up_out, &mut hidden);
        hipfire_cpu::quant::rotate_x(&mut hidden);
        hipfire_cpu::gemv::gemv(q, dn_bytes, m, mi, &hidden, &mut down_out);
        down_out
    };

    for target in [MemoryTarget::Device, MemoryTarget::HostMapped] {
        let label = match target {
            MemoryTarget::Device => "device",
            MemoryTarget::HostMapped => "host",
        };
        let gu = load(&mut gpu, &gu_name, gu_m, gu_k, target)?;
        let dn = load(&mut gpu, &dn_name, m, k, target)?;
        assert_eq!(gu.gpu_dtype, DType::MQ4G256V2);
        assert_eq!(dn.gpu_dtype, DType::MQ4G256V2);
        assert_eq!(
            gpu.host_located(&gu.buf),
            matches!(target, MemoryTarget::HostMapped),
            "{label} gate_up locality wrong"
        );
        // Loader-visible bytes first: on the host arm these are the
        // device-visible host bytes, so the CPU reference below is the
        // independent-reference AND real-host-placement pairing.
        let (gu_bytes, dn_bytes) = match target {
            MemoryTarget::Device => (
                gpu.download_raw_bytes(&gu.buf)?,
                gpu.download_raw_bytes(&dn.buf)?,
            ),
            MemoryTarget::HostMapped => (
                gpu.host_bytes(&gu.buf)
                    .expect("host gate_up bytes")
                    .to_vec(),
                gpu.host_bytes(&dn.buf).expect("host down bytes").to_vec(),
            ),
        };
        // Pointer tables from `sub_offset(0, stride)` views exactly as the
        // loader builds them (blob + slot*stride with slot 0).
        let gu_view = gu.buf.sub_offset(0, gu.buf.byte_size());
        let dn_view = dn.buf.sub_offset(0, dn.buf.byte_size());
        let gu_ptrs = gpu.upload_raw(&(gu_view.buf.as_ptr() as u64).to_ne_bytes(), &[8])?;
        let dn_ptrs = gpu.upload_raw(&(dn_view.buf.as_ptr() as u64).to_ne_bytes(), &[8])?;
        // k8-specialized kernel: grid.y is always 8 ranks. topk must hold 8
        // entries and y_gate/y_up [8 x mi] each; rank 0 is the comparison.
        // (A 1-entry topk + [mi] outputs overflows rank 1..7 into adjacent
        // pooled memory — the earlier rel-1.65 artifact.)
        let topk = gpu.upload_raw(&[0u8; 32], &[32])?;
        let x = gpu.upload_f32(&x_rot, &[m])?;

        // CPU reference gate_up rows (also feeds the CPU-derived rot below).
        let mut cpu_gu = vec![0.0f32; gu_m];
        hipfire_cpu::gemv::gemv(
            hipfire_cpu::quant::CpuQuant::Mq4G256V2,
            &gu_bytes,
            gu_m,
            m,
            &x_rot,
            &mut cpu_gu,
        );
        let (cpu_g, cpu_u) = cpu_gu.split_at(k);

        // gate_up half. Kernel writes y_gate/y_up as [k_top x mi] each
        // (`y_gate[krank*mi + row]`): [mi] here, NOT [2*mi] each.
        let yg = gpu.zeros(&[8 * k], DType::F32)?;
        let yu = gpu.zeros(&[8 * k], DType::F32)?;
        gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(&gu_ptrs, &topk, &x, &yg, &yu, gu_m, m)?;
        let (g_full, u_full) = (gpu.download_f32(&yg)?, gpu.download_f32(&yu)?);
        let (g, u) = (g_full[..k].to_vec(), u_full[..k].to_vec());
        let gs = cpu_gu.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        let wgg = g
            .iter()
            .zip(cpu_g.iter())
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        let wgu = g
            .iter()
            .zip(cpu_u.iter())
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        let wug = u
            .iter()
            .zip(cpu_g.iter())
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        let wuu = u
            .iter()
            .zip(cpu_u.iter())
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        // Pass requires the CORRECT pairing on both halves; cross terms are
        // diagnostics only (a swapped split must FAIL, not min-pass).
        let wg = wgg.max(wuu);
        eprintln!(
            "  [{label}] gate/up split: g-vs-g {wgg:.3e} g-vs-u {wgu:.3e} u-vs-g {wug:.3e} u-vs-u {wuu:.3e}"
        );

        // down half from a CPU-derived rot (silu+rotate on the CPU
        // reference), so a gate_up mismatch cannot leak into down.
        let mut cpu_hidden = vec![0.0f32; k];
        hipfire_cpu::epilogue::silu_mul(&cpu_gu, &mut cpu_hidden);
        hipfire_cpu::quant::rotate_x(&mut cpu_hidden);
        let rot = gpu.upload_f32(&cpu_hidden, &[k])?;
        let out = gpu.zeros(&[m], DType::F32)?;
        gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            &dn_ptrs, &topk, &rot, &out, m, k, 1, 1,
        )?;
        let kernel = gpu.download_f32(&out)?;
        let expect = cpu_expert(&gu_bytes, &dn_bytes);
        let scale = expect.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        let worst = kernel
            .iter()
            .zip(&expect)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        println!(
            "qt44 MoE [{label}] gate_up max|diff|={wg:.6e} (rel {:.3e}) down max|diff|={worst:.6e} (rel {:.3e})",
            wg / gs,
            worst / scale
        );
        assert!(
            wg <= gs * 5e-3,
            "qt44 MoE gate_up kernel [{label}] disagrees with CPU: {wg:.6e} rel {:.3e}",
            wg / gs
        );
        assert!(
            worst <= scale * 5e-3,
            "qt44 MoE down kernel [{label}] disagrees with CPU: {worst:.6e} rel {:.3e}",
            worst / scale
        );
        println!(
            "MQ4V2_MOE_KERNEL_PARITY PASS (qt44 [{label}], gate_up {gu_m}x{m} + down {m}x{k})"
        );
    }
    Ok(())
}
