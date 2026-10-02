// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device-vs-host parity smoke for offloaded projection weights.
//!
//! Partial GPU offload loads an offloaded layer's quantized codes into
//! host-mapped memory (`hipHostMalloc`) instead of VRAM. The promise is that this
//! changes *where*
//! the bytes live and nothing else — same bytes, same dtype, same shape, so the
//! GEMV numerics are unchanged. This example proves that promise on real data
//! rather than by inspection: it loads the same tensor through both readers and
//! compares the code blobs bit-for-bit.
//!
//! It exists because the two readers share one quant-type match behind an
//! injected uploader, and a partial swap would be invisible until it OOMs or
//! silently spills to VRAM. One comparison covers every arm.
//!
//! Usage:
//!     cargo run --release -p hipfire-arch-qwen35 --example host_offload_smoke -- MODEL.hfq
//!
//! With no argument it defaults to `~/.hipfire/models/qwen3.5-9b.mq4`.

use hipfire_arch_qwen35::qwen35::load::{
    load_weight_tensor, load_weight_tensor_host, qwen35_tensor_name_candidates,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        std::env::var("HOME")
            .map(|h| format!("{h}/.hipfire/models/qwen3.5-9b.mq4"))
            .unwrap_or_else(|_| "qwen3.5-9b.mq4".into())
    });
    println!("model: {path}");

    let mut hfq = HfqFile::open(std::path::Path::new(&path))?;
    // Keep the mmap alive: both readers take the zero-copy mmap path first.
    let mut gpu = Gpu::init()?;

    // Select from the file's own index rather than hardcoding a name or shape:
    // qwen3.5 checkpoints ship under `model.` or `model.language_model.`, VL
    // builds interleave linear- and full-attention layers, and layer dims differ
    // per model. A guess here just yields "tensor not found" or, worse, silently
    // reads the wrong tensor. RAW_CODE_QT lists quant types whose arms upload
    // opaque code blobs (the ones that actually get offloaded).
    const RAW_CODE_QT: &[u8] = &[44, 13, 17, 15, 14, 8, 7, 6];
    // Own the name so the index borrow ends before the host reader takes `&mut hfq`.
    let (name, m, k, qt) = {
        // Prefer a transformer-layer weight: that is what actually gets offloaded.
        // lm_head / embed_tokens are always resident, and lm_head is the largest
        // tensor in the file, so taking the first match would allocate hundreds of
        // MB of host memory to test something that never offloads.
        // Require a realistically-sized weight: the first `layers.` match can be
        // a degenerate 32-row linear-attention projection, which would exercise
        // the plumbing over 68 KB and prove nothing about a real offload target.
        let big = |t: &hipfire_runtime::hfq::HfqTensorInfo| {
            RAW_CODE_QT.contains(&t.quant_type)
                && t.shape.len() == 2
                && t.shape[0] >= 1024
                && t.shape[1] >= 1024
        };
        let info = hfq
            .tensor_infos()
            .iter()
            .find(|t| big(t) && t.name.contains("layers."))
            .or_else(|| hfq.tensor_infos().iter().find(|t| big(t)))
            .or_else(|| {
                hfq.tensor_infos()
                    .iter()
                    .find(|t| RAW_CODE_QT.contains(&t.quant_type) && t.shape.len() == 2)
            })
            .unwrap_or_else(|| panic!("no 2-D raw-code tensor in {path}"));
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

    let device = load_weight_tensor(&hfq, &gpu, name, m, k, qwen35_tensor_name_candidates)?;
    let host = load_weight_tensor_host(
        &mut hfq,
        &mut gpu,
        name,
        m,
        k,
        qwen35_tensor_name_candidates,
    )?;

    assert_eq!(device.gpu_dtype, host.gpu_dtype, "dtype diverged");
    assert_eq!(device.m, host.m, "m diverged");
    assert_eq!(device.k, host.k, "k diverged");
    assert_eq!(device.row_stride, host.row_stride, "row_stride diverged");
    println!(
        "dtype={:?} m={} k={} bytes={}",
        device.gpu_dtype,
        device.m,
        device.k,
        device.buf.byte_size()
    );

    // Locality is the load-bearing assertion. Byte parity alone would still pass
    // if `upload_raw_host` silently fell back to the device path, so the example
    // would report PASS without having offloaded anything.
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
    Ok(())
}
