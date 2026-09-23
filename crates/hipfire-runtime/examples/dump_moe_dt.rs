use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn qtname(q: u8) -> &'static str {
    match q {
        1 => "F16", 0 => "F32", 8 => "Q8_0",
        13 => "MQ4G256", 15 => "MQ6G256", 31 => "MQ5G256",
        24 => "MFP4G32", 34 => "MFP4G32E8", 35 => "MFP4G32E8SOA",
        _ => "?",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: <path.hfq>")?;
    let hfq = HfqFile::open(Path::new(&path))?;
    // Pick first MoE layer (find a layer with experts)
    for t in hfq.tensors() {
        let nm = &t.name;
        let is_l0 = nm.contains("layers.") && (
            nm.contains(".mlp.shared_expert") ||
            nm.contains(".mlp.gate.") || nm.contains(".mlp.router") ||
            nm.contains("experts.0.") );
        // only first MoE layer: pick the lowest layer index that has experts.0
        if is_l0 && (nm.contains("layers.1.") || nm.contains("layers.0.")) {
            println!("qt={:>2} ({:<12}) {}", t.quant_type, qtname(t.quant_type), nm);
        }
    }
    Ok(())
}
