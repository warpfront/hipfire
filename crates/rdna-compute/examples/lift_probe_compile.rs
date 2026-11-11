// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Offline production compile of the five gfx1100 hot-prefill hipcc modules, for the
//! lift-and-repass probe (PM-KRSI-1100 LiftProbe). No device. Each module goes through
//! `KernelCompiler::compile_to_paths` (the runtime JIT's argv: `core_hipcc_flags`,
//! ROCm flags, module flags, scheduler profile, per-source flag comments).
//!
//! Usage: lift_probe_compile OUT_DIR
//! Writes `OUT_DIR/<tag>.hsaco`, `OUT_DIR/<tag>.hip` (source as compiled), prints one
//! JSON line per module: tag, module, symbol, argv, sha256 of the object.

use rdna_compute::KernelCompiler;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const QSA_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);

fn main() {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: lift_probe_compile OUT_DIR"),
    );
    std::fs::create_dir_all(&out).unwrap();
    let work = out.join("kernel-cache");
    std::fs::create_dir_all(&work).unwrap();
    std::env::set_var("HIPFIRE_KERNEL_CACHE", &work);
    std::env::set_var("HIPFIRE_HOME", out.join("home"));
    // (tag, module name the runtime passes, symbol, source)
    let modules: [(&str, &str, &str, &str); 5] = [
        (
            "mq6_res_mw8",
            "gemm_mqv2_wmma_gfx1100_mw_lds",
            "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds",
            rdna_compute::kernels::GEMM_MQV2_WMMA_GFX11_MW_LDS_SRC,
        ),
        (
            "qsa_hg4",
            "tensor_ops",
            "indexed_attention_attention_q8_batched_hg4",
            QSA_SRC,
        ),
        (
            "moe_k2_silu",
            "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out",
            "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out",
            rdna_compute::kernels::QWEN4_GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC,
        ),
        (
            "moe_g128_down",
            "gemm_mq4g128v2_moe_grouped_wmma_gfx1151",
            "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out",
            rdna_compute::kernels::GEMM_MQ4G128V2_MOE_GROUPED_WMMA_GFX1151_SRC,
        ),
        (
            "hc_wmma_64",
            "qwen4_gemm_wmma_lds256",
            "gemm_wmma_lds_64_64_32_64_k64_p",
            rdna_compute::kernels::QWEN4_GEMM_F16_X_F16_WMMA_LDS256_SRC,
        ),
    ];
    for (tag, module, symbol, source) in modules {
        let kc = KernelCompiler::new("gfx1100", String::new()).expect("compiler");
        let src = out.join(format!("{tag}.hip"));
        let obj = out.join(format!("{tag}.hsaco"));
        let _ = std::fs::remove_file(&obj);
        kc.compile_to_paths(module, source, &src, &obj)
            .unwrap_or_else(|e| panic!("{tag}: {e}"));
        let argv = kc.jit_argv(module, source, &src, &obj);
        let bytes = std::fs::read(&obj).unwrap();
        println!(
            "{}",
            json!({
                "tag": tag,
                "module": module,
                "symbol": symbol,
                "argv": argv,
                "object": obj.display().to_string(),
                "bytes": bytes.len(),
                "sha256": format!("{:x}", Sha256::digest(&bytes)),
            })
        );
    }
}
