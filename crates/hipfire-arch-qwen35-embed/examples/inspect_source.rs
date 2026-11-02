// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Source diagnostics for arch 17: print how a tensor name resolves through
//! each read path (`tensor_data` borrowed, `tensor_data_pread` / `_vec`
//! owned) for a dir or an `.hfq`.
//!
//! Usage: cargo run --release -p hipfire-arch-qwen35-embed \
//!          --example inspect_source --features lab -- <path> [tensor_name]

use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::model_source::ModelSource as _;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: inspect_source <path> [tensor]")?;
    let want = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "language_model.embed_tokens.weight".into());

    let src = ModelSource::from_path(path)?;
    println!("source: {}", src.describe());
    println!("is_dir={} arch_id={:?}", src.is_dir(), src.arch_id());
    let st: &dyn hipfire_runtime::model_source::ModelSource = match &src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(d) => d,
    };
    let names = st.tensor_names();
    println!("tensor_names: {}", names.len());
    println!("contains {want:?}: {}", names.iter().any(|n| *n == want));
    println!("tensor_info: {}", st.tensor_info(&want).is_some());
    println!(
        "tensor_data (borrowed): {}",
        st.tensor_data(&want).is_some()
    );

    if let ModelSource::Hfq(hfq) = &src {
        println!(
            "hfq.tensor_data_pread: {}",
            hfq.tensor_data_pread(&want).is_some()
        );
        println!(
            "hfq.tensor_data_vec:   {}",
            hfq.tensor_data_vec(&want).is_some()
        );
    }
    // First few names, to eyeball spelling.
    for n in names.iter().take(4) {
        println!("  name[..]: {n}");
    }
    Ok(())
}
