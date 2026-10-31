// SPDX-License-Identifier: Apache-2.0
// Tiny HFQ tensor lister: `hfq_inspect <path.hfq> [substring]`.
use hipfire_runtime::hfq::HfqFile;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: hfq_inspect <path.hfq> [substring]");
    let filter = std::env::args().nth(2);
    let hfq = HfqFile::open(std::path::Path::new(&path)).expect("open hfq");
    println!("arch_id={}  tensors={}", hfq.arch_id, hfq.tensors().len());
    let mut shown = 0;
    for t in hfq.tensors() {
        if let Some(f) = &filter {
            if !t.name.contains(f) {
                continue;
            }
        }
        println!("qt={:2} sz={:>9} {} {:?}", t.quant_type, t.data_size, t.name, t.shape);
        shown += 1;
    }
    if let Some(f) = &filter {
        println!("({shown} tensors matching '{f}')");
    }
}
