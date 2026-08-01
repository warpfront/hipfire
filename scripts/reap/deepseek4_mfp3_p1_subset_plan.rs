// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// Emit an exact-name DeepSeek-V4 MFP3 P1 GPTQ attribution plan.
//
// Usage:
//   deepseek4_mfp3_p1_subset_plan <output.json> <stage-label> <tensor-name>...

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{self, Write};

const LAYERS: usize = 43;

fn usage(program: &str) -> ! {
    eprintln!("usage: {program} <output.json> <stage-label> <tensor-name>...");
    std::process::exit(2);
}

fn classify(name: &str) -> Option<(usize, &'static str)> {
    let rest = name.strip_prefix("layers.")?;
    let (layer, suffix) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if layer >= LAYERS {
        return None;
    }
    let role = match suffix {
        "attn.wq_a.weight"
        | "attn.wq_b.weight"
        | "attn.wo_a.weight"
        | "attn.wo_b.weight" => "attention",
        "ffn.shared_experts.w1.weight"
        | "ffn.shared_experts.w2.weight"
        | "ffn.shared_experts.w3.weight" => "shared_expert",
        _ => return None,
    };
    Some((layer, role))
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        usage(&args[0]);
    }
    let output_path = &args[1];
    let stage = &args[2];
    if stage.is_empty()
        || !stage
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsafe stage label {stage:?}"),
        ));
    }

    let mut seen = BTreeSet::new();
    let mut groups = BTreeMap::<(usize, &'static str), Vec<&str>>::new();
    for name in &args[3..] {
        if !seen.insert(name.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("duplicate tensor {name:?}"),
            ));
        }
        let (layer, role) = classify(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("tensor is outside the DeepSeek-V4 P1 allowlist: {name:?}"),
            )
        })?;
        groups.entry((layer, role)).or_default().push(name);
    }

    let mut output = File::create(output_path)?;
    writeln!(
        output,
        "{{\n  \"model_arch\": \"deepseek4\",\n  \"num_layers\": {LAYERS},\n  \
         \"original_experts\": 256,\n  \
         \"recipe\": \"deepseek4-mq2r-mfp3-p1-gptq-v1\",\n  \
         \"stage\": \"{stage}\",\n  \"quant_overrides\": ["
    )?;
    for (group_index, ((layer, role), names)) in groups.iter().enumerate() {
        if group_index != 0 {
            writeln!(output, ",")?;
        }
        writeln!(
            output,
            "    {{\n      \"layer\": {layer},\n      \"role\": \"{role}\",\n      \"tensors\": ["
        )?;
        for (name_index, name) in names.iter().enumerate() {
            let separator = if name_index + 1 == names.len() {
                ""
            } else {
                ","
            };
            writeln!(output, "        \"{name}\"{separator}")?;
        }
        write!(
            output,
            "      ],\n      \"tier\": \"mfp3g32e8-gptq\"\n    }}"
        )?;
    }
    writeln!(output, "\n  ]\n}}")?;
    output.sync_all()
}
