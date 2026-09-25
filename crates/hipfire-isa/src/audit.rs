// SPDX-License-Identifier: Apache-2.0
//! Offline, evidence-backed comparison of authored HIP intent with emitted ISA.
use crate::{arch::Arch, toolchain::{disassemble_code_object, Toolchain}, vopd::{self, Operand, VopdF32, VopdOp}};
use radiowave::{Inspector, SchedulerProfile};
use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, BTreeSet}, fs, path::PathBuf, process::Command};

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Intent { #[serde(default)] pub kernels: BTreeMap<String, KernelIntent> }

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelIntent {
    pub wmma: Option<u32>, pub swmmac: Option<u32>, pub vopd_pairs_possible: Option<u32>,
    pub max_valu: Option<u32>, pub max_s_wait_alu: Option<u32>, pub max_vgpr: Option<u32>,
    #[serde(default, alias = "forbidden")] pub forbidden_patterns: Vec<String>,
    #[serde(default)] pub separate_mul_add: bool,
    #[serde(default)] pub distinct_matrix_operands: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Evidence { pub start_line: usize, pub end_line: usize, pub start_pc: String, pub end_pc: String }
#[derive(Debug, Serialize)]
pub struct Finding { pub kind: String, pub detail: String, pub evidence: Evidence }
#[derive(Debug, Default, Clone, Serialize)]
pub struct Counts {
    pub wmma: u32, pub swmmac: u32, pub matrix_forms: BTreeMap<String,u32>,
    pub valu_slots: u32, pub vopd_packets: u32, pub fold_vopd_packets: u32,
    pub pairable_valu_pairs: u32, pub vopd_candidate_ops: u32,
    pub waits: BTreeMap<String,u32>, pub s_wait_alu: u32,
    pub ds: u32, pub vmem: u32, pub barriers: u32,
}
#[derive(Debug, Serialize)]
pub struct LoopReport { pub evidence: Evidence, pub back_edge: Evidence, pub counts: Counts, pub findings: Vec<Finding> }
#[derive(Debug, Serialize)]
pub struct KernelAudit { pub symbol: String, pub vgpr_count: u32, pub sgpr_count: u32,
    pub private_segment_fixed_size: u32, pub vgpr_spill_count: u32, pub sgpr_spill_count: u32,
    pub loops: Vec<LoopReport>, pub hot_loop: Option<usize>, pub findings: Vec<Finding> }
#[derive(Debug, Serialize)]
pub struct ProfileReport { pub profile: String, pub object: String, pub disassembly: String, pub kernels: Vec<KernelAudit> }
#[derive(Debug, Serialize)]
pub struct AuditReport { pub input: String, pub arch: String, pub profiles: Vec<ProfileReport> }

#[derive(Debug, Default)]
pub struct Options { pub input: PathBuf, pub arch: String, pub prepend: Vec<PathBuf>,
    pub defines: Vec<String>, pub flags: Vec<String>, pub intent: Option<PathBuf>,
    pub json: Option<PathBuf>, pub markdown: Option<PathBuf>, pub sweep_profiles: bool,
}

#[derive(Clone)]
struct Insn { line: usize, pc: u64, text: String, mnemonic: String, operands: Vec<String>, target: Option<u64> }
impl Insn {
    fn regs(&self) -> BTreeSet<String> { self.operands.iter().flat_map(|x| registers(x)).collect() }
    fn defs(&self) -> BTreeSet<String> {
        if self.mnemonic.starts_with("v_") || self.mnemonic.starts_with("s_") && !self.mnemonic.starts_with("s_wait")
            && !self.mnemonic.starts_with("s_cbranch") && !self.mnemonic.starts_with("s_branch")
            && !self.mnemonic.starts_with("s_cmp") && !self.mnemonic.starts_with("s_barrier")
            && !self.mnemonic.starts_with("s_store") {
            if self.mnemonic.starts_with("v_") && self.mnemonic.contains("store") { return BTreeSet::new(); }
            let mut defs: BTreeSet<String>=self.operands.first().map(|x| registers(x).into_iter().collect()).unwrap_or_default();
            if self.mnemonic.starts_with("v_dual_") {
                for part in self.text.split(" :: ").skip(1) {
                    if let Some(operand)=part.split_whitespace().nth(1) {
                        defs.extend(registers(operand));
                    }
                }
            }
            return defs;
        }
        BTreeSet::new()
    }
    fn uses(&self) -> BTreeSet<String> {
        let mut uses=BTreeSet::new();
        for part in self.text.split(" :: ") {
            let mnemonic=part.split_whitespace().next().unwrap_or("");
            for operand in part[mnemonic.len()..].split(',').skip(1) {
                uses.extend(registers(operand));
            }
            if mnemonic.starts_with("v_fmac") || mnemonic.starts_with("v_dual_fmac") {
                if let Some(dst)=part[mnemonic.len()..].split(',').next() {
                    uses.extend(registers(dst));
                }
            }
        }
        uses
    }
}

fn registers(s: &str) -> Vec<String> {
    let mut out=Vec::new(); let bytes=s.as_bytes(); let mut i=0;
    while i<bytes.len() {
        if (bytes[i]==b'v' || bytes[i]==b's') && (i==0 || !bytes[i-1].is_ascii_alphanumeric() && bytes[i-1]!=b'_') {
            let prefix=bytes[i] as char; let mut j=i+1;
            if j<bytes.len() && bytes[j]==b'[' { j+=1; }
            let start=j; while j<bytes.len() && bytes[j].is_ascii_digit() { j+=1; }
            if j>start {
                if let Ok(lo)=s[start..j].parse::<u16>() {
                    let mut hi=lo;
                    if bytes.get(j)==Some(&b':') { j+=1; let begin=j; while j<bytes.len() && bytes[j].is_ascii_digit() { j+=1; }
                        hi=s[begin..j].parse().unwrap_or(lo);
                    }
                    if hi>=lo && hi-lo<256 { for n in lo..=hi { out.push(format!("{prefix}{n}")); } }
                }
                i=j; continue;
            }
        }
        i+=1;
    }
    out
}
fn parse_disassembly(s: &str) -> BTreeMap<String, Vec<Insn>> {
    let mut symbols=BTreeMap::new(); let mut current=None::<String>;
    for (line_number,line) in s.lines().enumerate() {
        let trimmed=line.trim();
        if let Some((address,name))=trimmed.split_once(" <") {
            if let Some(name)=name.strip_suffix(">:") {
                if u64::from_str_radix(address,16).is_ok() {
                    current=Some(name.to_owned()); symbols.entry(name.to_owned()).or_insert_with(Vec::new); continue;
                }
            }
        }
        let Some(symbol)=current.as_ref() else { continue };
        let Some((code,comment))=line.split_once("//") else { continue };
        let Some((pc,_))=comment.trim().split_once(':') else { continue };
        let Ok(pc)=u64::from_str_radix(pc.trim(),16) else { continue };
        let text=code.trim(); let Some(mnemonic)=text.split_whitespace().next() else { continue };
        if !["v_","s_","ds_","buffer_","global_","flat_","scratch_"].iter().any(|p|mnemonic.starts_with(p)) { continue; }
        let operands=text[mnemonic.len()..].trim().split(" :: ").flat_map(|part| part.split(',').map(str::trim)).map(str::to_owned).collect();
        let target=comment.rsplit_once('<').and_then(|(_,tail)|tail.strip_suffix('>'))
            .and_then(|t| if t==symbol {Some(0)}
                else {t.rsplit_once("+0x").filter(|(name,_)|*name==symbol)
                    .and_then(|(_,off)|u64::from_str_radix(off,16).ok())})
            .and_then(|off|symbols.get(symbol).and_then(|v:&Vec<Insn>|v.first().map(|x|x.pc+off)));
        symbols.get_mut(symbol).unwrap().push(Insn{line:line_number+1,pc,text:text.to_owned(),mnemonic:mnemonic.to_owned(),operands,target});
    }
    symbols
}
fn evidence(first:&Insn,last:&Insn)->Evidence { Evidence{start_line:first.line,end_line:last.line,start_pc:format!("0x{:x}",first.pc),end_pc:format!("0x{:x}",last.pc)} }
fn finding(kind:&str,detail:String,first:&Insn,last:&Insn)->Finding { Finding{kind:kind.into(),detail,evidence:evidence(first,last)} }

fn candidate(insn: &Insn, part: &str) -> Option<VopdOp> {
    let mut words=part.split_whitespace(); let mnemonic=words.next()?;
    let op=if mnemonic.starts_with("v_dual_add_f32")||mnemonic.starts_with("v_add_f32") {VopdF32::Add}
        else if mnemonic.starts_with("v_dual_subrev_f32")||mnemonic.starts_with("v_subrev_f32") {VopdF32::Subrev}
        else if mnemonic.starts_with("v_dual_sub_f32")||mnemonic.starts_with("v_sub_f32") {VopdF32::Sub}
        else if mnemonic.starts_with("v_dual_mul_f32")||mnemonic.starts_with("v_mul_f32") {VopdF32::Mul}
        else if mnemonic.starts_with("v_dual_fmac_f32")||mnemonic.starts_with("v_fmac_f32") {VopdF32::Fmac}
        else {return None};
    let operands=part[mnemonic.len()..].split(',').map(str::trim).collect::<Vec<_>>();
    if operands.len()<3 {return None}
    let dst=operands[0].strip_prefix('v')?.parse::<u8>().ok()?;
    let src0=if let Some(n)=operands[1].strip_prefix('v').and_then(|v|v.parse().ok()) {Operand::V(n)}
        else if let Some(n)=operands[1].strip_prefix('s').and_then(|v|v.parse().ok()) {Operand::S(n)}
        else if let Some(n)=operands[1].strip_prefix("0x").and_then(|v|u32::from_str_radix(v,16).ok()) {Operand::Lit(n)}
        else {Operand::Inline(operands[1].parse().ok()?)};
    let src1=operands[2].strip_prefix('v')?.split_whitespace().next()?.parse::<u8>().ok()?;
    let _=insn; Some(VopdOp{op,dst,src0,src1})
}
fn independent(a:VopdOp,b:VopdOp)->bool {
    let ra=[a.src0,Operand::V(a.src1)]; let rb=[b.src0,Operand::V(b.src1)];
    !ra.contains(&Operand::V(b.dst)) && !rb.contains(&Operand::V(a.dst)) && a.dst!=b.dst
}
// Maximum matching on physical parity/bank-compatible candidate pairs. Existing
// packets are included, not counted twice; candidate/2 remains the semantic ceiling.
fn pairable(ops:&[VopdOp],arch:Arch)->u32 {
    let left:Vec<_>=ops.iter().copied().filter(|x|x.dst%2==0).collect();
    let right:Vec<_>=ops.iter().copied().filter(|x|x.dst%2==1).collect();
    let mut matched=vec![None;right.len()];
    fn augment(i:usize,left:&[VopdOp],right:&[VopdOp],matched:&mut [Option<usize>],seen:&mut [bool],arch:Arch)->bool {
        for (j,r) in right.iter().enumerate() {
            if seen[j] || !independent(left[i],*r) || vopd::validate_pair(arch,left[i],*r).is_err() {continue}
            seen[j]=true;
            if matched[j].is_none_or(|other|augment(other,left,right,matched,seen,arch)) {matched[j]=Some(i);return true}
        }
        false
    }
    let mut n=0; for i in 0..left.len() {if augment(i,&left,&right,&mut matched,&mut vec![false;right.len()],arch){n+=1}}
    n
}
fn count_loop(body:&[Insn],arch:Arch)->Counts {
    let mut counts=Counts::default(); let mut candidates=Vec::new();
    for i in body {
        let m=i.mnemonic.as_str();
        if m.starts_with("v_wmma_") {counts.wmma+=1;*counts.matrix_forms.entry(m.into()).or_default()+=1}
        else if m.starts_with("v_swmmac_") {counts.swmmac+=1;*counts.matrix_forms.entry(m.into()).or_default()+=1}
        else if m.starts_with("v_") {counts.valu_slots+=1;
            if m.starts_with("v_dual_") {counts.vopd_packets+=1;
                let parts:Vec<_>=i.text.split(" :: ").collect();
                if parts.len()==2 && parts.iter().all(|p|candidate(i,p).is_some()) {counts.fold_vopd_packets+=1}
                for part in parts {if let Some(op)=candidate(i,part) {candidates.push(op)} }
            }else if let Some(op)=candidate(i,&i.text){candidates.push(op)}
        }
        if m.starts_with("s_wait") { *counts.waits.entry(m.into()).or_default()+=1; if m=="s_wait_alu" {counts.s_wait_alu+=1} }
        if m.starts_with("ds_"){counts.ds+=1}
        if ["buffer_","global_","flat_","scratch_"].iter().any(|p|m.starts_with(p)){counts.vmem+=1}
        if m.starts_with("s_barrier"){counts.barriers+=1}
    }
    counts.vopd_candidate_ops=candidates.len() as u32;
    counts.pairable_valu_pairs=pairable(&candidates,arch);
    counts
}
fn loop_findings(body:&[Insn], counts:&Counts, intent:&KernelIntent, source_contract_off:bool)->Vec<Finding> {
    let mut findings=Vec::new(); let first=&body[0]; let last=body.last().unwrap();
    for (name,actual,expected) in [("wmma",counts.wmma,intent.wmma),("swmmac",counts.swmmac,intent.swmmac)] {
        if let Some(min)=expected.filter(|min|actual<*min) {findings.push(finding("matrix_collapse",format!("{name}: {actual} emitted, {min} intended"),first,last))}
    }
    if intent.distinct_matrix_operands {
        let mut seen=BTreeMap::<(String,String,String),&Insn>::new();
        for i in body.iter().filter(|x|x.mnemonic.starts_with("v_wmma_")||x.mnemonic.starts_with("v_swmmac_")) {
            if i.operands.len()<3 {continue}
            let key=(i.mnemonic.clone(),i.operands[1].clone(),i.operands[2].clone());
            if let Some(prior)=seen.get(&key) { if prior.operands.first()!=i.operands.first() {
                findings.push(finding("matrix_operand_reuse",format!("identical matrix A/B operands in distinct accumulator chains: {}, {}",key.1,key.2),prior,i));
            }}else{seen.insert(key,i);}
        }
    }
    if intent.separate_mul_add || source_contract_off {
        for i in body {
            for part in i.text.split(" :: ") {
                let mnemonic=part.split_whitespace().next().unwrap_or("");
                if mnemonic.starts_with("v_fma") || mnemonic.starts_with("v_dual_fmac") || mnemonic.starts_with("v_fmac") {
                    findings.push(finding("contraction",format!("{mnemonic} despite separate mul+add source intent"),i,i));
                }
            }
        }
    }
    if let Some(expected)=intent.vopd_pairs_possible.filter(|x|counts.fold_vopd_packets<*x) {
        findings.push(finding("vopd_pairing",format!("{} eligible-class packets emitted vs {expected} intended; {} pairable under physical bank rules; {} candidate operations",counts.fold_vopd_packets,counts.pairable_valu_pairs,counts.vopd_candidate_ops),first,last));
    }
    for (kind,actual,maximum) in [("valu_slots",counts.valu_slots,intent.max_valu),("s_wait_alu",counts.s_wait_alu,intent.max_s_wait_alu)] {
        if let Some(max)=maximum.filter(|max|actual>*max){findings.push(finding("excess_issue",format!("{kind}: {actual} > {max}"),first,last))}
    }
    let definitions: BTreeSet<String>=body.iter().flat_map(Insn::defs).collect();
    let mut varying=BTreeSet::new();
    for i in body {let defs=i.defs(); if !defs.is_disjoint(&i.regs()) && !defs.is_disjoint(&i.uses()) {varying.extend(defs)} }
    loop {let old=varying.len();for i in body {if !i.uses().is_disjoint(&varying) {varying.extend(i.defs());}}if varying.len()==old{break}}
    for i in body {
        if (i.mnemonic.starts_with("v_") && !i.mnemonic.starts_with("v_wmma") && !i.mnemonic.starts_with("v_swmmac") || i.mnemonic.starts_with("s_") && !i.mnemonic.starts_with("s_wait"))
            && !i.mnemonic.contains("store") && !i.mnemonic.contains("load") && !i.mnemonic.contains("branch") && !i.mnemonic.starts_with("s_cmp")
            && !i.defs().is_empty() && i.uses().is_disjoint(&definitions) && i.uses().is_disjoint(&varying) {
            findings.push(finding("loop_invariant_rematerialization",format!("{} reads no register defined in this loop",i.mnemonic),i,i));
        }
    }
    findings
}
fn audit_kernel(symbol:&str, insns:&[Insn], metadata:&radiowave::KernelReport, intent:&KernelIntent,arch:Arch, source_contract_off:bool)->KernelAudit {
    let mut findings=Vec::new();
    if let (Some(first),Some(last))=(insns.first(),insns.last()) {
        if metadata.private_segment_fixed_size>0 || metadata.vgpr_spill_count>0 || metadata.sgpr_spill_count>0 {
            findings.push(finding("spills",format!("private segment {} B/thread, VGPR spills {}, SGPR spills {}",metadata.private_segment_fixed_size,metadata.vgpr_spill_count,metadata.sgpr_spill_count),first,last));
        }
        if let Some(max)=intent.max_vgpr.filter(|max|metadata.vgpr_count>*max) {findings.push(finding("vgpr",format!("{} VGPR > {max}",metadata.vgpr_count),first,last));}
    }
    for i in insns {
        if i.mnemonic.starts_with("scratch_"){findings.push(finding("scratch_instruction",i.text.clone(),i,i))}
        for pattern in &intent.forbidden_patterns {
            let prefix=pattern.strip_suffix('*');
            if i.mnemonic==*pattern || prefix.is_some_and(|p|i.mnemonic.starts_with(p)) || i.text.contains(pattern) {
                findings.push(finding("forbidden_pattern",format!("{pattern}: {}",i.text),i,i));
            }
        }
    }
    let by_pc:BTreeMap<u64,usize>=insns.iter().enumerate().map(|(j,i)|(i.pc,j)).collect();
    let mut spans=BTreeMap::<usize,usize>::new();
    for (end,i) in insns.iter().enumerate(){if !i.mnemonic.starts_with("s_branch") && !i.mnemonic.starts_with("s_cbranch"){continue}
        if let Some(start)=i.target.and_then(|pc|by_pc.get(&pc)).copied().filter(|start|*start<=end) {
            spans.entry(start).and_modify(|old|*old=(*old).max(end)).or_insert(end);
        }
    }
    let mut loops=Vec::new();
    let mut regions=Vec::new();
    for (start,end) in spans {
        let body=&insns[start..=end];
        let counts=count_loop(body,arch);
        regions.push((start,end));
        loops.push(LoopReport{evidence:evidence(&insns[start],&insns[end]),
            back_edge:evidence(&insns[end],&insns[end]),counts,findings:Vec::new()});
    }
    // The smallest back-edge span containing the most matrix operations is
    // the steady-state compute loop, rather than a surrounding output loop.
    let hot_loop=loops.iter().enumerate().max_by_key(|(_,l)|
        (l.counts.wmma+l.counts.swmmac,usize::MAX-(l.evidence.end_line-l.evidence.start_line))).map(|(n,_)|n);
    if let Some(index)=hot_loop {
        let (start,end)=regions[index];
        loops[index].findings=loop_findings(&insns[start..=end],&loops[index].counts,intent,source_contract_off);
    }
    if hot_loop.is_none() && (intent.wmma.is_some() || intent.swmmac.is_some() || intent.vopd_pairs_possible.is_some()) {
        if let (Some(first),Some(last))=(insns.first(),insns.last()) {
            findings.push(finding("missing_hot_loop","no back-edge found for a kernel with expected loop ISA".into(),first,last));
        }
    }
    KernelAudit{symbol:symbol.into(),vgpr_count:metadata.vgpr_count,sgpr_count:metadata.sgpr_count,
        private_segment_fixed_size:metadata.private_segment_fixed_size,vgpr_spill_count:metadata.vgpr_spill_count,
        sgpr_spill_count:metadata.sgpr_spill_count,loops,hot_loop,findings}
}
fn run_command(command:&mut Command)->Result<(),String>{let output=command.output().map_err(|e|format!("{command:?}: {e}"))?;
    if !output.status.success(){return Err(format!("{command:?}: {}",String::from_utf8_lossy(&output.stderr)))}Ok(())}
fn render_markdown(report:&AuditReport)->String {
    let mut out=format!("# Peacemaker ISA audit\n\nInput: `{}`; arch: `{}`. ISA evidence lines refer to the disassembly named in each profile.\n\n",report.input,report.arch);
    for profile in &report.profiles {
        out.push_str(&format!("## {} (`{}`; [ISA]({}))\n\n| Kernel | VGPR | Private B/thread | Hot loop WMMA/SWMMAC | VALU slots | VOPD fold/all/pairable | s_wait_alu | Findings |\n|---|---:|---:|---:|---:|---:|---:|---|\n",profile.profile,profile.object,profile.disassembly));
        for k in &profile.kernels {let h=k.hot_loop.and_then(|n|k.loops.get(n));let c=h.map(|l|&l.counts);
            out.push_str(&format!("| `{}` | {} | {} | {}/{} | {} | {}/{}/{} | {} | {} |\n",k.symbol,k.vgpr_count,k.private_segment_fixed_size,
                c.map_or(0,|c|c.wmma),c.map_or(0,|c|c.swmmac),c.map_or(0,|c|c.valu_slots),c.map_or(0,|c|c.fold_vopd_packets),c.map_or(0,|c|c.vopd_packets),c.map_or(0,|c|c.pairable_valu_pairs),c.map_or(0,|c|c.s_wait_alu),k.findings.len()+k.loops.iter().map(|l|l.findings.len()).sum::<usize>()));
        }
        out.push('\n');for k in &profile.kernels {
            out.push_str(&format!("### `{}`\n\n",k.symbol));
            for (j,l) in k.loops.iter().enumerate() {out.push_str(&format!("- Loop {j} ISA lines {}–{} (PC {}–{}): WMMA {}, SWMMAC {}, VALU slots {}, VOPD {} fold/{} all/{} bank-pairable, waits {:?}, DS {}, VMEM {}, barriers {}.\n",l.evidence.start_line,l.evidence.end_line,l.evidence.start_pc,l.evidence.end_pc,l.counts.wmma,l.counts.swmmac,l.counts.valu_slots,l.counts.fold_vopd_packets,l.counts.vopd_packets,l.counts.pairable_valu_pairs,l.counts.waits,l.counts.ds,l.counts.vmem,l.counts.barriers)); }
            for f in k.findings.iter().chain(k.loops.iter().flat_map(|l|&l.findings)) {out.push_str(&format!("- **{}** ISA lines {}–{} (PC {}–{}): {}.\n",f.kind,f.evidence.start_line,f.evidence.end_line,f.evidence.start_pc,f.evidence.end_pc,f.detail));}
            out.push('\n');
        }
    }
    out
}
pub fn run(options:Options)->Result<AuditReport,String> {
    if options.arch.is_empty(){return Err("audit requires --arch (e.g. gfx1201)".into())}
    if options.sweep_profiles && options.input.extension().is_some_and(|x|x=="hsaco") {return Err("--sweep-profiles requires HIP source".into())}
    let arch:Arch=options.arch.parse()?;
    let intent:Intent=options.intent.as_ref().map(|p|fs::read(p).map_err(|e|format!("{}: {e}",p.display())).and_then(|bytes|serde_json::from_slice(&bytes).map_err(|e|format!("{}: {e}",p.display())))).transpose()?.unwrap_or_default();
    let toolchain=Toolchain::default();let hip_source=options.input.extension().is_some_and(|x|x=="hip");
    let json=options.json.clone().unwrap_or_else(||options.input.with_extension("audit.json"));
    let markdown=options.markdown.clone().unwrap_or_else(||options.input.with_extension("audit.md"));
    for path in [&json,&markdown] {if let Some(parent)=path.parent(){fs::create_dir_all(parent).map_err(|e|e.to_string())?}}
    let stem=json.file_stem().and_then(|s|s.to_str()).unwrap_or("audit");
    let work=std::env::temp_dir().join(format!("peacemaker-audit-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e|e.to_string())?.as_nanos()));
    fs::create_dir_all(&work).map_err(|e|e.to_string())?;
    let result=(|| {
        let source=if hip_source && !options.prepend.is_empty() {
            let joined=work.join("source.hip");let mut bytes=Vec::new();
            for path in options.prepend.iter().chain(std::iter::once(&options.input)) {bytes.extend(fs::read(path).map_err(|e|format!("{}: {e}",path.display()))?);bytes.push(b'\n');}
            fs::write(&joined,bytes).map_err(|e|e.to_string())?;joined
        }else{options.input.clone()};
        let source_text=if hip_source {Some(fs::read_to_string(&source).map_err(|e|e.to_string())?)}else{None};
        let contract_off=source_text.as_deref().is_some_and(|s|s.contains("__fmul_rn")&&s.contains("__fadd_rn") || s.contains("contract(off)"));
        let profiles:Vec<_>=if hip_source && options.sweep_profiles {
            SchedulerProfile::ALL.iter().map(|p|(p.as_str().to_owned(),p.llvm_args().iter().map(|s|s.to_string()).collect::<Vec<_>>())).chain(std::iter::once(("amdgpu_iterative_ilp".into(),vec!["-mllvm".into(),"-amdgpu-sched-strategy=iterative-ilp".into()]))).collect()
        }else{vec![("default".into(),vec![])]};
        let mut results=Vec::new();
        for (profile,flags) in profiles {
            let object=if hip_source {let output=json.with_file_name(format!("{stem}.{profile}.hsaco"));let mut command=Command::new(&toolchain.hipcc);
                command.args(["--genco",&format!("--offload-arch={}",options.arch),"-O3","--no-offload-compress"]);
                for define in &options.defines {command.arg(format!("-D{define}"));}
                command.args(&options.flags).args(&flags).arg(&source).arg("-o").arg(&output);
                run_command(&mut command)?;output
            }else{options.input.clone()};
            let disassembly=disassemble_code_object(&toolchain,&object,&options.arch)?;
            let isa_path=json.with_file_name(format!("{stem}.{profile}.isa.txt"));
            fs::write(&isa_path,&disassembly).map_err(|e|format!("{}: {e}",isa_path.display()))?;
            let inspector=Inspector::from_hipcc(&toolchain.hipcc);
            let inspection_input=if fs::read(&object).map_err(|e|e.to_string())?.starts_with(b"\x7fELF") {
                let wrapped=work.join(format!("{profile}-inspection.hsaco"));
                let mut command=Command::new(&toolchain.bundler);
                command.args(["-type=o","-bundle-align=4096"])
                    .arg(format!("-targets={},hipv4-amdgcn-amd-amdhsa--{}",toolchain.host_target,options.arch))
                    .arg("-input=/dev/null").arg(format!("-input={}",object.display()))
                    .arg(format!("-output={}",wrapped.display()));
                run_command(&mut command)?;wrapped
            }else{object.clone()};
            let inspected=inspector.inspect(&inspection_input,&options.arch).map_err(|e|e.to_string())?;
            for key in intent.kernels.keys().filter(|key|key.as_str()!="*") {
                if !inspected.kernels.iter().any(|kernel|kernel.name==*key) {
                    return Err(format!("intent symbol {key} absent from {}",object.display()));
                }
            }
            let symbols=parse_disassembly(&disassembly);
            let default_intent=KernelIntent::default();
            let mut kernels=Vec::new();
            for meta in &inspected.kernels {let Some(insns)=symbols.get(&meta.name) else {return Err(format!("kernel {} absent from disassembly",meta.name))};
                let expected=intent.kernels.get(&meta.name).or_else(||intent.kernels.get("*")).unwrap_or(&default_intent);
                kernels.push(audit_kernel(&meta.name,insns,meta,expected,arch,contract_off));
            }
            if kernels.is_empty(){return Err(format!("no kernels in {}",object.display()))}
            results.push(ProfileReport{profile,object:object.display().to_string(),disassembly:isa_path.display().to_string(),kernels});
        }
        let report=AuditReport{input:options.input.display().to_string(),arch:options.arch.clone(),profiles:results};
        fs::write(&json,serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|format!("{}: {e}",json.display()))?;
        fs::write(&markdown,render_markdown(&report)).map_err(|e|format!("{}: {e}",markdown.display()))?;
        Ok(report)
    })();let _=fs::remove_dir_all(work);result
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn distinguishes_back_edge_and_counts_dual_slots() {
        let text="0000000000000100 <kernel>:\n v_add_f32 v0, v1, v2 // 000000000100: AA\n v_dual_add_f32 v4, v6, v8 :: v_dual_mul_f32 v5, v7, v9 // 000000000104: BB\n v_wmma_i32_16x16x32_iu4 v[16:23], v[0:1], v[2:3], v[16:23] // 00000000010c: CC\n s_cbranch_scc0 65530 // 000000000114: DD <kernel+0x0>\n s_endpgm // 000000000118: EE\n";
        let insns=parse_disassembly(text);let i=&insns["kernel"];
        let r=audit_kernel("kernel",i,&radiowave::KernelReport::default(),&KernelIntent{wmma:Some(2),..Default::default()},Arch::Gfx1201,false);
        assert_eq!(r.loops.len(),1);assert_eq!(r.loops[0].counts.valu_slots,2);assert_eq!(r.loops[0].counts.vopd_packets,1);
        assert!(r.loops[0].findings.iter().any(|f|f.kind=="matrix_collapse"));
    }
    #[test] fn catches_second_half_contraction_without_mislabeling_recurrence() {
        let text="0000000000000100 <kernel>:\n v_add_f32 v0, v0, v2 // 000000000100: AA\n v_dual_mul_f32 v4, v6, v8 :: v_dual_fmac_f32 v5, v7, v9 // 000000000104: BB\n s_cbranch_scc0 65530 // 00000000010c: CC <kernel+0x0>\n";
        let parsed=parse_disassembly(text);
        let report=audit_kernel("kernel",&parsed["kernel"],&radiowave::KernelReport::default(),
            &KernelIntent{separate_mul_add:true,..Default::default()},Arch::Gfx1201,false);
        let findings=&report.loops[0].findings;
        assert!(findings.iter().any(|f|f.kind=="contraction" && f.detail.contains("v_dual_fmac")));
        assert!(!findings.iter().any(|f|f.kind=="loop_invariant_rematerialization"
            && f.evidence.start_pc=="0x100"));
    }
}
