// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// Emit cumulative DeepSeek-V4 MFP3+GPTQ P1 plans for quality attribution.
//
// Usage:
//   deepseek4_mfp3_p1_plan <stage> <output.json> [layer]
//
// Stages, in cumulative order:
//   wq-b
//   wq-b-wo-a
//   wq-b-wo-a-wo-b
//   attn-all
//   attn-plus-shared-gate-up
//   p1-full

use std::env;
use std::fs::File;
use std::io::{self, Write};

const LAYERS: usize = 43;

fn emit_entry(
    output: &mut File,
    first: &mut bool,
    layer: usize,
    role: &str,
    suffixes: &[&str],
) -> io::Result<()> {
    if suffixes.is_empty() {
        return Ok(());
    }
    if !*first {
        writeln!(output, ",")?;
    }
    *first = false;
    writeln!(
        output,
        "    {{\n      \"layer\": {layer},\n      \"role\": \"{role}\",\n      \"tensors\": ["
    )?;
    for (index, suffix) in suffixes.iter().enumerate() {
        let separator = if index + 1 == suffixes.len() { "" } else { "," };
        writeln!(output, "        \"layers.{layer}.{suffix}\"{separator}")?;
    }
    write!(
        output,
        "      ],\n      \"tier\": \"mfp3g32e8-gptq\"\n    }}"
    )
}

fn stage_tensors(stage: &str) -> Option<(&'static [&'static str], &'static [&'static str])> {
    let attention = match stage {
        "wq-b" => &["attn.wq_b.weight"][..],
        "wq-b-wo-a" => &["attn.wq_b.weight", "attn.wo_a.weight"],
        "wq-b-wo-a-wo-b" => &["attn.wq_b.weight", "attn.wo_a.weight", "attn.wo_b.weight"],
        "attn-all" | "attn-plus-shared-gate-up" | "p1-full" => &[
            "attn.wq_a.weight",
            "attn.wq_b.weight",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
        ],
        _ => return None,
    };
    let shared = match stage {
        "attn-plus-shared-gate-up" => &[
            "ffn.shared_experts.w1.weight",
            "ffn.shared_experts.w3.weight",
        ][..],
        "p1-full" => &[
            "ffn.shared_experts.w1.weight",
            "ffn.shared_experts.w2.weight",
            "ffn.shared_experts.w3.weight",
        ],
        _ => &[],
    };
    Some((attention, shared))
}

fn usage(program: &str) -> ! {
    eprintln!(
        "usage: {program} \
         <wq-b|wq-b-wo-a|wq-b-wo-a-wo-b|attn-all|attn-plus-shared-gate-up|p1-full> \
         <output.json> [layer 0..42]"
    );
    std::process::exit(2);
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if !(args.len() == 3 || args.len() == 4) {
        usage(&args[0]);
    }
    let stage = args[1].as_str();
    let Some((attention, shared)) = stage_tensors(stage) else {
        usage(&args[0]);
    };
    let selected_layer = if let Some(value) = args.get(3) {
        let layer: usize = value.parse().unwrap_or_else(|_| usage(&args[0]));
        if layer >= LAYERS {
            usage(&args[0]);
        }
        Some(layer)
    } else {
        None
    };

    let mut output = File::create(&args[2])?;
    writeln!(
        output,
        "{{\n  \"model_arch\": \"deepseek4\",\n  \"num_layers\": {LAYERS},\n  \
         \"original_experts\": 256,\n  \"recipe\": \"deepseek4-mq2r-mfp3-p1-gptq-v1\",\n  \
         \"stage\": \"{stage}\",\n  \"quant_overrides\": ["
    )?;
    let mut first = true;
    for layer in 0..LAYERS {
        if selected_layer.is_some_and(|selected| selected != layer) {
            continue;
        }
        emit_entry(&mut output, &mut first, layer, "attention", attention)?;
        emit_entry(&mut output, &mut first, layer, "shared_expert", shared)?;
    }
    writeln!(output, "\n  ]\n}}")?;
    output.sync_all()
}
