// SPDX-License-Identifier: Apache-2.0
// Throwaway probe (experiment/vcn-jpeg): parse the 5 committed fixtures for
// VA submission without touching the GPU.
// Run: cargo run -p va-bridge --example parse_fixtures
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/vision/images");
    for name in [
        "general_qa.jpg",
        "barney_cigar.jpg",
        "scene_1.jpg",
        "scene_2.jpg",
        "doge.jpeg",
    ] {
        let bytes = std::fs::read(dir.join(name)).expect("fixture");
        match va_bridge::parse_for_va(&bytes) {
            Ok(p) => println!(
                "{name}: {}x{} comps={} maxH={} maxV={} mcus={} entropy={}B qmask={:?} hmask={:?} dri={}",
                p.width,
                p.height,
                p.pic.num_components,
                p.max_h,
                p.max_v,
                p.slice.num_mcus,
                p.entropy.len(),
                p.iq.load_quantiser_table,
                p.huff.load_huffman_table,
                p.slice.restart_interval,
            ),
            Err(e) => println!("{name}: PARSE FAIL: {e}"),
        }
    }
}
