// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device-vs-host parity smoke for offloaded projection weights.
//!
//! Partial GPU offload loads an offloaded layer's quantized codes into
//! host-mapped memory (`hipHostMalloc`) instead of VRAM. The promise is that this
//! changes *where*
//! the bytes live and nothing else — same bytes, same dtype, same shape, so the
//! GEMV numerics are unchanged. This example proves that promise on real data
//! rather than by inspection: it loads the same tensor through one reader at both
//! `MemoryTarget`s and compares the code blobs bit-for-bit.
//!
//! It exists because one quant-type match sits behind a target parameter, and a
//! partial swap of that parameter would be invisible until it OOMs or silently
//! spills to VRAM. One comparison covers every arm.
//!
//! Usage:
//!     cargo run --release -p hipfire-arch-qwen35 --example host_offload_smoke -- MODEL.hfq
//!
//! With no argument it defaults to `~/.hipfire/models/qwen3.5-9b.mq4`.

use hipfire_arch_qwen35::qwen35::load::{load_weight_tensor, qwen35_tensor_name_candidates};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_backend::MemoryTarget;
use rdna_compute::{DType, Gpu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        std::env::var("HOME")
            .map(|h| format!("{h}/.hipfire/models/qwen3.5-9b.mq4"))
            .unwrap_or_else(|_| "qwen3.5-9b.mq4".into())
    });
    println!("model: {path}");
    // Optional second argument: a substring the tensor name must contain, so the
    // harness can be pointed at the class the offload path actually spills.
    let filter: Option<String> = std::env::args().nth(2);

    let mut hfq = HfqFile::open(std::path::Path::new(&path))?;
    // Keep the mmap alive: the reader takes the zero-copy mmap path first.
    let mut gpu = Gpu::init()?;

    // Select from the file's own index rather than hardcoding a name or shape:
    // qwen3.5 checkpoints ship under `model.` or `model.language_model.`, VL
    // builds interleave linear- and full-attention layers, and layer dims differ
    // per model. A guess here just yields "tensor not found" or, worse, silently
    // reads the wrong tensor. RAW_CODE_QT lists quant types whose arms upload
    // opaque code blobs (the ones that actually get offloaded).
    const RAW_CODE_QT: &[u8] = &[44, 13, 17, 15, 14, 8, 7, 6];
    // Own the name so the index borrow ends before the reader is called.
    let (name, m, k, qt) = {
        // Prefer a transformer-layer weight: that is what actually gets offloaded.
        // lm_head / embed_tokens are always resident, and lm_head is the largest
        // tensor in the file, so taking the first match would allocate hundreds of
        // MB of host memory to test something that never offloads.
        // Require a realistically-sized weight: the first `layers.` match can be
        // a degenerate 32-row linear-attention projection, which would exercise
        // the plumbing over 68 KB and prove nothing about a real offload target.
        // `shaped` carries the class filter, and EVERY selection arm below must
        // honour it -- otherwise a filter that matches nothing silently falls
        // through to an unrelated tensor (which is exactly what an earlier
        // version of this did, reporting a linear-attention projection while
        // claiming to test a shared expert).
        let shaped = |t: &hipfire_runtime::hfq::HfqTensorInfo| {
            RAW_CODE_QT.contains(&t.quant_type)
                && t.shape.len() == 2
                && filter.as_deref().is_none_or(|f| t.name.contains(f))
        };
        // The size gate exists so an unfiltered run tests something worth
        // offloading; an explicit filter may name a small tensor (a router is
        // [num_experts, hidden]) and must not be overridden by it.
        let large = |t: &hipfire_runtime::hfq::HfqTensorInfo| {
            filter.is_none() && t.shape[0] >= 1024 && t.shape[1] >= 1024
        };
        let big = |t: &hipfire_runtime::hfq::HfqTensorInfo| shaped(t) && large(t);
        let info = RAW_CODE_QT
            .iter()
            .find_map(|qt| {
                hfq.tensor_infos()
                    .iter()
                    .find(|t| big(t) && t.quant_type == *qt && t.name.contains("layers."))
            })
            .or_else(|| hfq.tensor_infos().iter().find(|t| big(t)))
            .or_else(|| hfq.tensor_infos().iter().find(|t| shaped(t)))
            .unwrap_or_else(|| {
                panic!("no 2-D raw-code tensor in {path} matching filter {filter:?}")
            });
        (
            info.name.clone(),
            info.shape[0] as usize,
            info.shape[1] as usize,
            info.quant_type,
        )
    };
    let name = name.as_str();
    println!(
        "picked {name} qt={qt} shape={:?}",
        hfq.find_tensor_info(name).map(|i| &i.shape)
    );

    let device = load_weight_tensor(
        &hfq,
        &mut gpu,
        name,
        m,
        k,
        qwen35_tensor_name_candidates,
        MemoryTarget::Device,
    )?;
    let host = load_weight_tensor(
        &hfq,
        &mut gpu,
        name,
        m,
        k,
        qwen35_tensor_name_candidates,
        MemoryTarget::HostMapped,
    )?;

    assert_eq!(device.gpu_dtype, host.gpu_dtype, "dtype diverged");
    assert_eq!(device.m, host.m, "m diverged");
    assert_eq!(device.k, host.k, "k diverged");
    assert_eq!(device.row_stride, host.row_stride, "row_stride diverged");
    println!(
        "dtype={:?} m={} k={} bytes={} name={name}",
        device.gpu_dtype,
        device.m,
        device.k,
        device.buf.byte_size()
    );

    // Locality is the load-bearing assertion. Byte parity alone would still pass
    // if `MemoryTarget::HostMapped` silently fell back to the device path, so the
    // example would report PASS without having offloaded anything.
    assert!(
        gpu.host_located(&host.buf),
        "host reader did not produce a host-located tensor - offload did not happen"
    );
    assert!(
        !gpu.host_located(&device.buf),
        "device reader unexpectedly produced a host-located tensor"
    );
    let (_, source) = hfq
        .tensor_data(name)
        .expect("selected from the index above");
    assert_eq!(
        device.buf.byte_size(),
        source.len(),
        "device blob size != on-disk tensor size"
    );
    assert_eq!(
        host.buf.byte_size(),
        source.len(),
        "host blob size != on-disk tensor size"
    );
    println!("locality: device=VRAM host=host-mapped (confirmed via Gpu::host_located)");

    let dev_bytes = gpu.download_raw_bytes(&device.buf)?;
    let host_bytes = gpu.download_raw_bytes(&host.buf)?;
    assert_eq!(
        dev_bytes.len(),
        host_bytes.len(),
        "length diverged: device {} vs host {}",
        dev_bytes.len(),
        host_bytes.len()
    );
    assert!(
        dev_bytes == host_bytes,
        "code blobs differ: first mismatch at {:?}",
        dev_bytes.iter().zip(&host_bytes).position(|(a, b)| a != b)
    );

    println!(
        "HOST_OFFLOAD_PARITY PASS ({} bytes identical, host-located confirmed)",
        dev_bytes.len()
    );

    // The MoE decode arm for a packed expert is the *indexed* kernel, not the
    // dense GEMV. Call it directly with a one-expert pointer table built from the
    // device blob and again from the host blob, and compare — the direct test of
    // whether a host-mapped expert reads correctly through the MoE arm.
    let x_data: Vec<f32> = {
        let mut s: u32 = 0x1234_5678;
        (0..k)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    };
    let x = gpu.upload_f32(&x_data, &[k])?;
    let topk = gpu.upload_raw(&0i32.to_ne_bytes(), &[4])?;
    let mut results = Vec::new();
    for (label, w) in [("device", &device), ("host", &host)] {
        let ptrs = gpu.upload_raw(&(w.buf.buf.as_ptr() as u64).to_ne_bytes(), &[8])?;
        let yg = gpu.zeros(&[m], DType::F32)?;
        let yu = gpu.zeros(&[m], DType::F32)?;
        gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(&ptrs, &topk, &x, &yg, &yu, m, k)?;
        let (g, u) = (gpu.download_f32(&yg)?, gpu.download_f32(&yu)?);
        let g1: f32 = g.iter().map(|v| v.abs()).sum();
        let u1: f32 = u.iter().map(|v| v.abs()).sum();
        println!(
            "indexed gate_up [{label}]: |gate|_1={g1:.6e} |up|_1={u1:.6e} y[0]={:.6e} y[last]={:.6e}",
            g[0],
            g[m - 1]
        );
        results.push((g, u));
    }
    let (dg, du) = &results[0];
    let (hg, hu) = &results[1];
    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .fold(0.0f32, |acc, (p, q)| acc.max((p - q).abs()))
    };
    let scale = dg.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
    let (wg, wu) = (worst(dg, hg), worst(du, hu));
    println!(
        "indexed: max|dev-host| gate={wg:.6e} up={wu:.6e} (gate scale {scale:.6e}, rel {:.3e})",
        wg / scale
    );
    assert!(
        wg <= scale * 1e-4 && wu <= 1e-4 * hu.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6),
        "the INDEXED MoE gate_up kernel reads a host-mapped {:?} expert differently from its \
         device twin: gate max|diff|={wg:.6e} (rel {:.3e}), up max|diff|={wu:.6e}",
        device.gpu_dtype,
        wg / scale
    );
    println!(
        "INDEXED_MOE_HOST_READ_PARITY PASS (dtype {:?})",
        device.gpu_dtype
    );

    // The packed path's real shape: ONE blob holding many experts, with the
    // pointer table holding *view* addresses (`blob + slot * stride`). This
    // kernel is k8-specialised (8 top-k slots), so use eight experts and eight
    // slots — the configuration the decode path actually runs.
    const EXPERTS: usize = 8;
    let stride = source.len();
    let mut blob = Vec::with_capacity(stride * EXPERTS);
    for _ in 0..EXPERTS {
        blob.extend_from_slice(&source);
    }
    let dev_blob = gpu.upload_raw(&blob, &[blob.len()])?;
    let host_blob = gpu.upload_raw_host_mapped(&blob, &[blob.len()])?;
    let topk8 = gpu.upload_raw(
        &(0..EXPERTS as i32)
            .flat_map(|i| i.to_ne_bytes())
            .collect::<Vec<u8>>(),
        &[4 * EXPERTS],
    )?;
    let mut outs = Vec::new();
    for (label, owner) in [("device", &dev_blob), ("host", &host_blob)] {
        let mut table = Vec::new();
        for slot in 0..EXPERTS {
            let view = owner.sub_offset(slot * stride, stride);
            table.extend_from_slice(&(view.buf.as_ptr() as u64).to_ne_bytes());
        }
        let ptrs = gpu.upload_raw(&table, &[table.len()])?;
        let yg = gpu.zeros(&[EXPERTS * m], DType::F32)?;
        let yu = gpu.zeros(&[EXPERTS * m], DType::F32)?;
        gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(&ptrs, &topk8, &x, &yg, &yu, m, k)?;
        let (g, u) = (gpu.download_f32(&yg)?, gpu.download_f32(&yu)?);
        println!(
            "packed blob [{label}]: k_top=8 y[0]={:.6e} y[m]={:.6e} |gate|_1={:.6e}",
            g[0],
            g[m],
            g.iter().map(|v| v.abs()).sum::<f32>()
        );
        outs.push((g, u));
    }
    let wg = worst(&outs[0].0, &outs[1].0);
    let wu = worst(&outs[0].1, &outs[1].1);
    let s2 = outs[0]
        .0
        .iter()
        .fold(0.0f32, |a, v| a.max(v.abs()))
        .max(1e-6);
    println!(
        "packed views: max|dev-host| gate={wg:.6e} up={wu:.6e} (scale {s2:.6e}, rel {:.3e})",
        wg / s2
    );
    assert!(
        wg <= s2 * 1e-4 && wu <= 1e-4 * s2,
        "the indexed MoE kernel reads a host-mapped *packed view* (blob + slot*stride) differently \
         from its device twin: gate max|diff|={wg:.6e} (rel {:.3e}), up max|diff|={wu:.6e}",
        wg / s2
    );
    println!("INDEXED_PACKED_VIEW_HOST_READ_PARITY PASS");
    Ok(())
}
