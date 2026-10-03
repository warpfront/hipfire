//! Replay exact captured device addresses; no pointer relocation or GPU dependency.
use std::{fs,io::Read,path::{Path,PathBuf}};
use peacemaker_ir::{emu::{Launch,Memory},inst::Frontend};
use sha2::{Digest,Sha256};
use serde_json::Value;
type Result<T,E=String> = std::result::Result<T,E>;
fn file(dir:&Path,name:&str)->Result<Vec<u8>>{fs::read(dir.join(name)).map_err(|e|format!("{}: {e}",dir.join(name).display()))}
fn text<'a>(v:&'a Value,key:&str)->Result<&'a str>{v[key].as_str().ok_or_else(||format!("missing string {key}"))}
fn uint(v:&Value,key:&str)->Result<u32>{v[key].as_u64().and_then(|n|n.try_into().ok()).ok_or_else(||format!("missing u32 {key}"))}
fn triple(v:&Value,key:&str)->Result<[u32;3]>{let a=v[key].as_array().ok_or_else(||format!("missing {key}"))?;if a.len()!=3{return Err(format!("{key} must have 3 axes"));}let mut r=[0;3];for n in 0..3 {r[n]=a[n].as_u64().and_then(|x|x.try_into().ok()).ok_or("invalid axis")?;}Ok(r)}
fn parse_wg(s:&str)->Result<[u32;3]>{
    let mut axes=s.split(',');
    let mut next=||axes.next().ok_or("--wg requires x,y,z")?.parse::<u32>().map_err(|e|e.to_string());
    let wg=[next()?,next()?,next()?];
    if axes.next().is_some(){return Err("--wg requires x,y,z".into());}
    Ok(wg)
}
fn replay(dir:&Path,selection:&[[u32;3]])->Result<u64>{
    let manifest:Value=serde_json::from_slice(&file(dir,"manifest.json")?).map_err(|e|e.to_string())?;
    let code=file(dir,text(&manifest,"code_object")?)?;
    let digest=format!("{:x}",Sha256::digest(&code));
    if digest!=text(&manifest,"code_object_sha256")?{return Err("code object SHA256 differs from capture".into());}
    let program=peacemaker_lift::lift_object(&code,peacemaker_lift::Options{frontend:Frontend::Builder}).map_err(|e|e.to_string())?.program;
    let arch=format!("{:?}",program.target.arch).to_lowercase();if arch!=text(&manifest,"arch")?{return Err("manifest target differs from ELF".into());}
    let symbol=text(&manifest,"symbol")?;let grid=triple(&manifest,"grid_wg")?;let block=triple(&manifest,"block")?;let dynamic_lds=uint(&manifest,"dynamic_lds")?;
    let workgroups=if selection.is_empty(){grid.iter().map(|&n|u128::from(n)).product::<u128>()}else{selection.len()as u128};
    println!("{}\t{}\tcoverage={}\tworkgroups={}\tgrid={grid:?}",dir.display(),symbol,if selection.is_empty(){"full-grid/all-buffer-bytes"}else{"coverage-limited/selected-written-bytes"},workgroups);
    let mut memory=Memory::default();
    let buffers=manifest["buffers"].as_array().ok_or("missing buffers")?;
    for b in buffers {
        let before=file(dir,text(b,"before")?)?;
        let size=b["bytes"].as_u64().ok_or("missing buffer size")?;
        let after_size=fs::metadata(dir.join(text(b,"after")?)).map_err(|e|e.to_string())?.len();
        if before.len()as u64!=size||after_size!=size{return Err("capture buffer lengths disagree".into());}
        let ptr=u64::from_str_radix(text(b,"device_ptr")?.trim_start_matches("0x"),16).map_err(|e|e.to_string())?;
        memory.map(text(b,"name")?.into(),ptr,before)?;
    }
    let kernarg=file(dir,text(&manifest,"kernarg")?)?;
    // A synthetic host-only kernarg address is never copied into the captured
    // arguments. Select an unmapped address; captured device pointers stay exact.
    let kernarg_ptr=memory.regions.iter().map(|r|r.base+r.bytes.len()as u64).max().unwrap_or(0x10000).checked_add(4096).ok_or("kernarg address overflow")?;
    memory.map("__kernarg".into(),kernarg_ptr,kernarg)?;
    let emulator=peacemaker_ir::emu::Emulator::new(&program,symbol)?;
    let run=|wg:[u32;3],mem:&mut Memory|->Result<()> {
        if (0..3).any(|n|wg[n]>=grid[n]){return Err(format!("selected workgroup {wg:?} outside grid {grid:?}"));}
        emulator.run_workgroup(&Launch{kernarg_ptr,workgroup:wg,block,dynamic_lds},mem)
    };
    if selection.is_empty(){for z in 0..grid[2]{for y in 0..grid[1]{for x in 0..grid[0]{run([x,y,z],&mut memory)?;}}}}
    else{for &wg in selection{run(wg,&mut memory)?;}}
    let mut total=0;
    for (n,r) in memory.regions[..buffers.len()].iter().enumerate() {
        let mut before=fs::File::open(dir.join(text(&buffers[n],"before")?)).map_err(|e|e.to_string())?;
        let mut after=fs::File::open(dir.join(text(&buffers[n],"after")?)).map_err(|e|e.to_string())?;
        let mut old=[0u8;65536];let mut gpu=[0u8;65536];
        let mut differing=0u64;let mut changed_outside_write=0u64;let mut emu_changed=0u64;let mut gpu_changed=0u64;
        for start in (0..r.bytes.len()).step_by(old.len()) {
            let count=old.len().min(r.bytes.len()-start);
            before.read_exact(&mut old[..count]).map_err(|e|e.to_string())?;
            after.read_exact(&mut gpu[..count]).map_err(|e|e.to_string())?;
            for offset in 0..count {
                let byte=start+offset;let actual=r.bytes[byte];let written=r.written.contains(byte);
                differing+=u64::from((selection.is_empty()||written)&&actual!=gpu[offset]);
                changed_outside_write+=u64::from(!written&&actual!=old[offset]);
                emu_changed+=u64::from(actual!=old[offset]);gpu_changed+=u64::from(gpu[offset]!=old[offset]);
            }
        }
        // Capture-only manifests have no role field: infer from GPU changes.
        // An explicit role (new captures) independently checks input preservation
        // on both the GPU and CPU, including for selected-WG replays.
        let explicit=buffers[n]["output"].as_bool();
        let output=explicit.unwrap_or(gpu_changed!=0);
        let outside_output=if output{0}else{emu_changed};
        let capture_input_changed=if output{0}else{gpu_changed};
        println!("{}\t{}\t{}\tdiffering_bytes={}\twritten_bytes={}\tuntouched_changed={}\toutside_output_changed={}\tcapture_input_changed={}\toutput_role={}",dir.display(),symbol,r.name,differing,r.written.count(),changed_outside_write,outside_output,capture_input_changed,if explicit.is_some(){"manifest"}else{"inferred"});
        total+=differing+changed_outside_write+outside_output+capture_input_changed;
    }
    let args=memory.regions.last().unwrap();if args.written.count()!=0{return Err("kernel wrote kernarg storage".into());}
    Ok(total)
}
fn main(){if let Err(e)=run(){eprintln!("pm-emu: {e}");std::process::exit(1);}}
fn run()->Result<()> {let mut args=std::env::args().skip(1);let dir=PathBuf::from(args.next().ok_or("usage: pm-emu SNAPSHOT [--wg x,y,z ...]")?);let mut selected=Vec::new();while let Some(arg)=args.next(){if arg!="--wg"{return Err(format!("unknown option {arg}"));}selected.push(parse_wg(&args.next().ok_or("missing --wg value")?)?);}let n=replay(&dir,&selected)?;if n!=0{return Err(format!("{n} differing or forbidden bytes"));}Ok(())}
#[cfg(test)]mod tests {
    use super::*;
    fn dirs(p:&Path,out:&mut Vec<PathBuf>)->Result<()> {if p.join("manifest.json").exists(){out.push(p.into());return Ok(());}for e in fs::read_dir(p).map_err(|e|e.to_string())?{let p=e.map_err(|e|e.to_string())?.path();if p.is_dir(){dirs(&p,out)?;}}Ok(())}
    #[test]fn captured_snapshots(){
        let Some(root)=std::env::var_os("PM_R2_SNAP")else{eprintln!("PM_R2_SNAP absent: GPU snapshot replay skipped (CPU semantic tests still run)");return;};
        let mut snapshots=Vec::new();dirs(Path::new(&root),&mut snapshots).unwrap();
        assert!(!snapshots.is_empty(),"PM_R2_SNAP root contains no manifests");
        for p in snapshots {
            let manifest:Value=serde_json::from_slice(&fs::read(p.join("manifest.json")).unwrap()).unwrap();
            let selected=manifest.get("recommended_workgroups").map(|v|v.as_array().expect("workgroup list").iter().map(|v|{
                let v=v.as_array().expect("workgroup coordinates");assert_eq!(v.len(),3);
                [0,1,2].map(|i|u32::try_from(v[i].as_u64().expect("unsigned coordinate")).expect("u32 coordinate"))
            }).collect::<Vec<_>>()).unwrap_or_default();
            assert_eq!(replay(&p,&selected).unwrap(),0,"snapshot {}",p.display());
        }
    }
    #[test]fn gated_modules_have_semantic_rows(){
        let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
        for module in ["qsa_gather_pm_gfx1151.hxaco","qsa_gather_pm_gfx1201.hxaco","qwen4_moe_iu4_sym_pm_gfx1151.hxaco","qwen4_moe_iu4_sym_pm_gfx1201.hxaco"] {
            let bytes=fs::read(root.join(module)).unwrap();
            let p=peacemaker_lift::lift_object(&bytes,peacemaker_lift::Options{frontend:Frontend::Builder}).unwrap().program;
            for k in &p.kernels {for id in &k.body.layout {
                let i=k.body.insts.get(*id).unwrap();
                let row=peacemaker_ir::isa::lookup(p.target.arch,i.op,i.form).unwrap();
                assert!(peacemaker_ir::emu::semantic_class(row.name).is_some(),"{module}: {} {:?}",row.name,i.form);
                if let peacemaker_ir::FormFields::Vopd{y_op,..}=i.fields {let name=y_op.name(p.target.arch).unwrap();assert!(peacemaker_ir::emu::semantic_class(name).is_some(),"{module}: {name}");}
            }}
        }
    }
}
