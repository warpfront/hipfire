// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-quantize: Quantize raw FP16/BF16/FP32 model weights to Q4_F16 format.
//!
//! Usage: hipfire-quantize --input <model_dir-or-gguf> --output <output.hfq> [--format mq4]
//!
//! Reads safetensors files from a HuggingFace model directory OR a single
//! `.gguf` file and produces a `.hfq` (HipFire Quantized) file with
//! RDNA-native quantized weights.

mod calibration;
mod cli;
mod dequant;
#[cfg(test)]
mod diagnostics;
mod e8;
mod e8_gptq;
mod gguf_input;
mod hfq;
mod maple;
mod model_filter;
mod pipeline;
mod pipeline_deepseek;
mod pipeline_flux;
mod pipeline_gguf;
mod pipeline_maple;
mod ple_hot_pack;
mod ple_hot_rank;
mod quant_e8;
mod quant_fwht;
mod quant_hfp4;
mod quant_mq;
mod quant_mq4v2_lloyd;
mod quant_q4;
mod qwen4;
mod raon;
mod reap_overlay;
mod trunk_overlay;

fn main() {
    // `ple-hot-rank`, `ple-hot-pack` and `trunk-overlay` are standalone offline tools with their
    // own flags; they never enter the quantize pipeline's argument surface.
    let mut argv = std::env::args().skip(1);
    match argv.next().as_deref() {
        Some("ple-hot-rank") => std::process::exit(ple_hot_rank::run(argv.collect())),
        Some("ple-hot-pack") => std::process::exit(ple_hot_pack::run(argv.collect())),
        Some("trunk-overlay") => std::process::exit(trunk_overlay::run(argv.collect())),
        _ => {}
    }
    pipeline::run();
}
