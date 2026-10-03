//! Offline R4 MoE schedule inventory: emit, assemble, lift and record resources.
//! Run with --features toolchain and an evidence directory argument.
use hipfire_isa::{Arch, kernels::qwen4_moe_sym::{self, Kind, Spec, R4Spec}, toolchain::{Toolchain, build}};
use serde_json::json;
use std::{collections::BTreeMap, fs, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::args().nth(1).ok_or("evidence directory required")?);
    fs::create_dir_all(&root)?;
    let tools = Toolchain::oracle();
    let route_file = PathBuf::from(std::env::args().nth(2).ok_or("captured top10 i32 route required")?);
    let mut counts = vec![0u32; 512];
    let route_bytes = fs::read(route_file)?;
    if route_bytes.len() % 4 != 0 { return Err("truncated routing record".into()); }
    for raw in route_bytes.chunks_exact(4) {
        let id = i32::from_le_bytes(raw.try_into()?);
        let count = counts.get_mut(usize::try_from(id)?).ok_or("expert index outside route")?;
        *count += 1;
    }
    let mut rows = Vec::new();
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        for nt in [2, 4, 8] {
            for chains in [1, 2, 4, 8].into_iter().filter(|&c| c <= nt) {
                for prefetch in [true, false] {
                    for grouped in [false, true] {
                    for kind in Kind::ALL {
                        let name = format!("{}_{}_nt{nt}_c{chains}_p{}_g{}", arch.name(), kind.tag(), u8::from(prefetch), u8::from(grouped));
                        let mut row = json!({"name":name,"arch":arch.name(),"kind":kind.tag(),"nt":nt,"chains":chains,"prefetch":prefetch,"grouped":grouped,"ring_depth":0,"padding":"pad16"});
                        match qwen4_moe_sym::emit_r4(R4Spec {base:Spec {arch,kind,nt,rr:1}, chains,prefetch,grouped}) {
                            Err(e) => {
                                if !e.starts_with("qwen4_moe_sym: tile width ") && !e.contains(" VGPR exceeds ") {
                                    return Err(format!("{name}: emission failed: {e}").into());
                                }
                                row["rejected"] = json!(e);
                            }
                            Ok(e) => {
                                let source = root.join(format!("{name}.s"));
                                fs::write(&source, &e.s_text)?;
                                fs::write(root.join(format!("{name}.proof.json")), serde_json::to_vec_pretty(&e.proof)?)?;
                                row["shape"] = serde_json::to_value(&e.shape)?;
                                let build = build(&tools, &source, &root.join(format!("{name}.hxaco")), arch.name())?;
                                let bytes = fs::read(&build.elf)?;
                                let lifted = peacemaker_lift::lift_object(&bytes, peacemaker_lift::Options {frontend:peacemaker_ir::inst::Frontend::Builder})?;
                                if peacemaker_lift::emit::module(&lifted.program)? != bytes { return Err(format!("{name}: lift identity differs").into()); }
                                row["lift"] = json!("byte-exact");
                                let symbol = &lifted.program.kernels[0].symbol.0;
                                let routes = BTreeMap::from([(symbol.clone(), hipfire_isa::cost_lint::Route {
                                    rows: counts.clone(), tile_rows: 16, k128_epochs: Some(2), epoch_starts: BTreeMap::new(),
                                })]);
                                row["cost"] = serde_json::to_value(hipfire_isa::cost_lint::analyze(&lifted.program, &routes)?)?;
                                if nt == 4 && prefetch && grouped && matches!(chains, 2 | 4) {
                                    row["certification"] = hipfire_isa::pm_check::m7(&build.elf, arch.name(), symbol)?;
                                }
                                row["elf"] = json!(build.elf);
                                fs::write(root.join(format!("{name}.dis")), build.disassembly)?;
                            }
                        }
                        println!("{}", row);
                        rows.push(row);
                        fs::write(root.join("screen.json"),serde_json::to_vec_pretty(&rows)?)?;
                    }
                    }
                }
            }
        }
    }
    Ok(())
}
