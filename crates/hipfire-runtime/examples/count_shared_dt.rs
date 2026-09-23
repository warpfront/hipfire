use hipfire_runtime::hfq::HfqFile;
use std::collections::BTreeMap;
use std::path::Path;

fn qtname(q: u8) -> &'static str {
    match q { 3 => "Q8", 34 => "E8", 13 => "MQ4", 15 => "MQ6", _ => "?" }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: <path.hfq>")?;
    let hfq = HfqFile::open(Path::new(&path))?;
    // Per layer: collect shared_expert gate/up/down + expert.0 gate_up/down qt.
    let mut sgate: BTreeMap<usize, u8> = BTreeMap::new();
    let mut sdown: BTreeMap<usize, u8> = BTreeMap::new();
    let mut egu: BTreeMap<usize, u8> = BTreeMap::new();
    for t in hfq.tensors() {
        let nm = &t.name;
        if let Some(pos) = nm.find("layers.") {
            let rest = &nm[pos + 7..];
            if let Some(dot) = rest.find('.') {
                if let Ok(li) = rest[..dot].parse::<usize>() {
                    if nm.contains(".mlp.shared_expert.gate_proj") { sgate.insert(li, t.quant_type); }
                    if nm.contains(".mlp.shared_expert.down_proj") { sdown.insert(li, t.quant_type); }
                    if nm.contains(".mlp.experts.0.gate_up_proj") { egu.insert(li, t.quant_type); }
                }
            }
        }
    }
    let mut counts: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut moe_layers = 0;
    for (&li, &sg) in &sgate {
        let sd = *sdown.get(&li).unwrap_or(&0);
        moe_layers += 1;
        *counts.entry((qtname(sg), qtname(sd))).or_insert(0) += 1;
        println!("layer {:2}: shared.gate={} shared.down={} expert.gate_up={}",
            li, qtname(sg), qtname(sd), qtname(*egu.get(&li).unwrap_or(&0)));
    }
    println!("\n== shared-expert (gate,down) dtype distribution over {moe_layers} MoE layers ==");
    for ((g, d), c) in &counts { println!("  gate={g} down={d}: {c} layers"); }
    Ok(())
}
