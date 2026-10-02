// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Developer-only QSA evidence. No model or product dispatch changes.
//! `bench_qsa_indexed fixtures OUT` or `snapshot SNAPSHOT.json OUT [oracle]`.
//! `shape ROWS OUT` is the pinned 3072-start fixture for fresh-process timing.
//! CPU-only `analyze SNAPSHOT.json OUT` preserves f64 oracles; `freeze DUMP_DIR
//! MANIFEST.json` refuses incomplete pp8192 data and never overwrites a ceiling.
//! `isolate SNAPSHOT.json OUT` swaps KV, Q, P casts or online order one at a
//! time; all f64 values are retained, and the casts are explicitly nonadditive.
//! Floored relative/ULP reporting uses |GPU reference|>=1e-5 only as a display
//! subset. Raw all-element ceilings and admission thresholds remain unchanged.
//! `self-check` exercises metric/quantized-oracle semantics without a GPU.
//! Snapshots retain little-endian source codes/scales, not re-quantized teachers.
//! The real-model tap also includes this file as a module so metric definitions
//! and the dump schema have exactly one implementation.

use rdna_compute::tensor_ops::{indexed_attention_attention_batch, indexed_attention_attention_batch_exact, IndexedAttentionAttentionBatch, QsaKvFormat};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

pub type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Debug) -> String { format!("{e:?}") }

#[derive(Clone, Debug)]
pub struct Geometry {
    pub rows: usize, pub position_start: usize, pub heads: usize, pub kv_heads: usize,
    pub dim: usize, pub budget_blocks: usize, pub compress: usize,
    pub capacity: usize, pub full_capacity: usize, pub fp8: bool,
}
impl Geometry {
    pub fn json(&self) -> Value {
        json!({"rows":self.rows,"position_start":self.position_start,"heads":self.heads,
            "kv_heads":self.kv_heads,"dim":self.dim,"budget_blocks":self.budget_blocks,
            "compress":self.compress,"capacity":self.capacity,"full_capacity":self.full_capacity,
            "format":if self.fp8 {"fp8"} else {"f32"}})
    }
    fn parse(v: &Value) -> Result<Self> {
        let n = |k: &str| v[k].as_u64().map(|n| n as usize).ok_or_else(|| format!("missing geometry {k}"));
        Ok(Self {rows:n("rows")?,position_start:n("position_start")?,heads:n("heads")?,
            kv_heads:n("kv_heads")?,dim:n("dim")?,budget_blocks:n("budget_blocks")?,
            compress:n("compress")?,capacity:n("capacity")?,full_capacity:n("full_capacity")?,fp8:v["format"]=="fp8"})
    }
    pub fn selected_len(&self, row: usize) -> usize {
        let visible = self.position_start + row + 1;
        (self.budget_blocks.min(visible / self.compress) * self.compress + visible % self.compress).min(self.capacity)
    }
    fn row_bytes(&self) -> usize { self.kv_heads * if self.fp8 {self.dim + 2} else {self.dim * 4} }
}

pub fn sha256(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
pub fn f32_bytes(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|x| x.to_le_bytes()).collect() }
pub fn f32_values(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 { return Err("F32 bytes not divisible by four".into()); }
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
}
pub fn i32_values(bytes: &[u8]) -> Result<Vec<i32>> {
    if bytes.len() % 4 != 0 { return Err("I32 bytes not divisible by four".into()); }
    Ok(bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect())
}
pub fn number(x: f64) -> Value {
    if x.is_nan() { json!("NaN") } else if x == f64::INFINITY { json!("Infinity") }
    else if x == f64::NEG_INFINITY { json!("-Infinity") } else { json!(x) }
}
fn ordered(x: f32) -> u32 { let b=x.to_bits(); if b & 0x8000_0000 != 0 {!b} else {b | 0x8000_0000} }
pub fn ulp(a: f32, b: f32) -> u32 { if a == b {0} else {ordered(a).abs_diff(ordered(b))} }
fn element(r: f32, c: f32) -> (f64,f64,u32) {
    if !r.is_finite() || !c.is_finite() {
        return if r.to_bits()==c.to_bits() {(0.0,0.0,0)} else {(f64::INFINITY,f64::INFINITY,u32::MAX)};
    }
    let a=(c as f64-r as f64).abs();
    (a,if r==0.0 {if c==0.0 {0.0} else {f64::INFINITY}} else {a/(r as f64).abs()},ulp(r,c))
}

/// Per-element abs/raw-relative/ULP and bit identity; no epsilon denominator.
/// Optional records: abs:f64, rel:f64, ULP:u32, equal-bits:u8 (LE, 21 bytes).
pub fn metrics(reference: &[f32], candidate: &[f32], heads: usize, dim: usize, records: Option<&Path>) -> Result<Value> {
    if reference.len()!=candidate.len() || dim==0 || heads==0 || reference.len()%(heads*dim)!=0 {return Err("metric shape mismatch".into());}
    let mut file=records.map(std::fs::File::create).transpose().map_err(err)?.map(std::io::BufWriter::new);
    let (mut ma,mut mr,mut mu)=(0.0f64,0.0f64,0u32);
    let (mut ia,mut ir,mut iu,mut equal,mut rn,mut cn,mut mismatch)=(0usize,0usize,0usize,0usize,0usize,0usize,0usize);
    let mut row_heads=Vec::new(); let mut cancellations=Vec::new(); let mut max_a=0.0f64;
    const REPORT_ABS_FLOOR:f64=1e-5;
    let (mut floor_rel,mut floor_ulp,mut floor_n,mut floor_ir,mut floor_iu)=(0.0f64,0u32,0usize,0usize,0usize);
    for (rh,(rr,cc)) in reference.chunks_exact(dim).zip(candidate.chunks_exact(dim)).enumerate() {
        let ref_max=rr.iter().filter(|x| x.is_finite()).fold(0.0f64,|m,x| m.max((*x as f64).abs()));
        let mut row_abs=0.0f64; let mut row_rel=0.0f64; let mut row_ulp=0u32;
        let mut near=0; let mut near_changed=0;
        for (d,(&r,&c)) in rr.iter().zip(cc).enumerate() {
            let i=rh*dim+d; let (a,rel,u)=element(r,c);
            if a>ma {ma=a;ia=i;} if rel>mr {mr=rel;ir=i;} if u>mu {mu=u;iu=i;}
            row_abs=row_abs.max(a); row_rel=row_rel.max(rel); row_ulp=row_ulp.max(u);
            equal+=usize::from(r.to_bits()==c.to_bits()); rn+=usize::from(!r.is_finite()); cn+=usize::from(!c.is_finite());
            mismatch+=usize::from((!r.is_finite() || !c.is_finite()) && r.to_bits()!=c.to_bits());
            if r.is_finite() && (r as f64).abs()>=REPORT_ABS_FLOOR {
                floor_n+=1;
                if rel>floor_rel {floor_rel=rel;floor_ir=i;}if u>floor_ulp {floor_ulp=u;floor_iu=i;}
            }
            if r.is_finite() && (r as f64).abs()<=ref_max.max(1e-30)*1e-6 {near+=1;near_changed+=usize::from(a!=0.0);}
            if let Some(f)=file.as_mut() {f.write_all(&a.to_le_bytes()).map_err(err)?;f.write_all(&rel.to_le_bytes()).map_err(err)?;f.write_all(&u.to_le_bytes()).map_err(err)?;f.write_all(&[u8::from(r.to_bits()==c.to_bits())]).map_err(err)?;}
        }
        let a=row_abs/ref_max.max(1e-30); max_a=max_a.max(a);
        row_heads.push(json!({"row":rh/heads,"head":rh%heads,"max_abs":number(row_abs),"raw_max_rel":number(row_rel),"max_ulp":row_ulp,"ref_max":number(ref_max),"A":number(a)}));
        if near_changed>0 {cancellations.push(json!({"row":rh/heads,"head":rh%heads,"near_zero_elements":near,"changed_near_zero_elements":near_changed,"threshold":ref_max.max(1e-30)*1e-6}));}
    }
    if let Some(f)=file.as_mut() {f.flush().map_err(err)?;}
    let worst=|i:usize| if reference.is_empty() {Value::Null} else {json!({"flat_index":i,"row":i/(heads*dim),"head":i/dim%heads,"channel":i%dim,"reference":number(reference[i] as f64),"candidate":number(candidate[i] as f64),"reference_bits":reference[i].to_bits(),"candidate_bits":candidate[i].to_bits()})};
    Ok(json!({"elements":reference.len(),"max_abs":number(ma),"raw_max_rel":number(mr),"max_ulp":mu,"max_A":number(max_a),
        "bit_identical_elements":equal,"differing_bits":reference.len()-equal,"reference_nonfinite":rn,"candidate_nonfinite":cn,"nonfinite_mismatch":mismatch,
        "floored_reporting":{"abs_reference_floor":REPORT_ABS_FLOOR,"included_elements":floor_n,
            "max_rel":number(floor_rel),"max_ulp":floor_ulp,
            "worst_rel":if floor_n==0 {Value::Null} else {worst(floor_ir)},
            "worst_ulp":if floor_n==0 {Value::Null} else {worst(floor_iu)},
            "note":"reporting subset only; raw frozen gate and all-element max-abs/ULP/A are unchanged"},
        "worst_abs":worst(ia),"worst_raw_rel":worst(ir),"worst_ulp":worst(iu),"row_heads":row_heads,"near_zero_cancellation_rows":cancellations}))
}

pub fn half(bits: u16) -> f32 {
    let sign=if bits&0x8000!=0 {-1.0} else {1.0}; let exp=(bits>>10)&31; let man=bits&1023;
    match exp {0=>sign*man as f32*2f32.powi(-24),31=>if man==0 {sign*f32::INFINITY} else {f32::NAN},_=>sign*(1.0+man as f32/1024.0)*2f32.powi(exp as i32-15)}
}
/// IEEE binary16 round-to-nearest/ties-even, including gradual underflow.
/// kv_slots::half_from_f32 intentionally truncates small fixture scales and is
/// not the `_Float16` operand conversion used by these gathered kernels.
fn half_rne(x: f32) -> u16 {
    let bits=x.to_bits();let sign=((bits>>16)&0x8000) as u16;
    let exponent=((bits>>23)&255) as i32;let mantissa=bits&0x7f_ffff;
    if exponent==255 {return sign|0x7c00|if mantissa==0 {0} else {0x200|((mantissa>>13) as u16)};}
    let exp=exponent-127;
    if exp>15 {return sign|0x7c00;}if exp< -25 {return sign;}
    let round=|value:u32,shift:u32| {
        let high=value>>shift;let remainder=value&((1<<shift)-1);let midpoint=1<<(shift-1);
        high+u32::from(remainder>midpoint || (remainder==midpoint && high&1!=0))
    };
    if exp< -14 {return sign|round(mantissa|0x80_0000,(-exp-1) as u32) as u16;}
    sign|(((exp+15) as u16)<<10).wrapping_add(round(mantissa,13) as u16)
}
fn cast16(x: f64) -> f64 {half(half_rne(x as f32)) as f64}
fn fp8(code: u8) -> f32 {
    let sign=if code&128!=0 {-1.0} else {1.0}; let e=(code>>3)&15; let m=code&7;
    if e==15 && m==7 {f32::NAN} else if e==0 {sign*m as f32*2f32.powi(-9)} else {sign*(1.0+m as f32/8.0)*2f32.powi(e as i32-7)}
}
pub fn decode_cache(bytes: &[u8], g: &Geometry) -> Result<Vec<f32>> {
    if bytes.len()!=g.full_capacity*g.row_bytes() {return Err("source cache byte count mismatch".into());}
    if !g.fp8 {return f32_values(bytes);}
    let mut out=Vec::with_capacity(g.full_capacity*g.kv_heads*g.dim);
    for row in bytes.chunks_exact(g.row_bytes()) {
        let width=g.kv_heads*g.dim;
        for h in 0..g.kv_heads {
            let scale=half(u16::from_le_bytes([row[width+2*h],row[width+2*h+1]]));
            out.extend(row[h*g.dim..(h+1)*g.dim].iter().map(|&c| fp8(c)*scale));
        }
    }
    Ok(out)
}

pub fn dump_snapshot(dir: &Path, identity: Value, g: &Geometry, blobs: &[(&str,&[u8])], stats: Value) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).map_err(err)?;
    let mut files=serde_json::Map::new();
    for &(name,bytes) in blobs {std::fs::write(dir.join(name),bytes).map_err(err)?;files.insert(name.into(),json!({"bytes":bytes.len(),"sha256":sha256(bytes)}));}
    let path=dir.join("snapshot.json");
    std::fs::write(&path,serde_json::to_vec_pretty(&json!({"schema":"qsa-source-v1","identity":identity,"geometry":g.json(),"files":files,"eager_vs_exact":stats})).map_err(err)?).map_err(err)?;
    Ok(path)
}

/// Mathematical f64 from the exact source cache, with separate operand-only
/// and operand+online-F16-probability casts. Association residual includes F32
/// arithmetic/transcendental implementation, not just WMMA reduction order.
pub fn oracle(q: &[f32], k: &[f32], v: &[f32], selected: &[i32], g: &Geometry) -> Result<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f32>)> {
    if q.len()!=g.rows*g.heads*2*g.dim || k.len()!=g.full_capacity*g.kv_heads*g.dim || v.len()!=k.len() || selected.len()!=g.rows*g.capacity {return Err("oracle shape mismatch".into());}
    let qh: Vec<f64> = q.iter().map(|&x| cast16(x as f64)).collect();
    let kh: Vec<f64> = k.iter().map(|&x| cast16(x as f64)).collect();
    let vh: Vec<f64> = v.iter().map(|&x| cast16(x as f64)).collect();
    let workers=std::thread::available_parallelism().map_or(1,usize::from).min(32).min(g.rows.max(1));
    let chunk=g.rows.div_ceil(workers);
    let parts=std::thread::scope(|scope| {
        let mut jobs=Vec::new();
        for first in (0..g.rows).step_by(chunk) {
            let end=(first+chunk).min(g.rows);
            let (qh, kh, vh) = (&qh, &kh, &vh);
            jobs.push(scope.spawn(move || {
                let mut exact=Vec::new();let mut operands=Vec::new();let mut probability=Vec::new();let mut exp_inputs=Vec::new();
                for row in first..end {
                    let len=g.selected_len(row); let visible=g.position_start+row+1;
                    let sel=&selected[row*g.capacity..row*g.capacity+len];
                    for head in 0..g.heads {
                        let kvh=head/(g.heads/g.kv_heads);let qb=(row*g.heads+head)*2*g.dim;
                        let mut s=vec![f64::NEG_INFINITY;len];let mut sh=s.clone();
                        for (slot,&token) in sel.iter().enumerate() {
                            if token<0 || token as usize>=visible || token as usize>=g.full_capacity {continue;}
                            let kb=(token as usize*g.kv_heads+kvh)*g.dim;
                            let (mut dot,mut dh)=(0.0,0.0);
                            for d in 0..g.dim {dot+=q[qb+d] as f64*k[kb+d] as f64;dh+=qh[qb+d]*kh[kb+d];}
                            s[slot]=dot/(g.dim as f64).sqrt();sh[slot]=dh/(g.dim as f64).sqrt();
                        }
                        let m=s.iter().copied().fold(f64::NEG_INFINITY,f64::max);let mh=sh.iter().copied().fold(f64::NEG_INFINITY,f64::max);
                        let mut a=vec![0.0f64;g.dim];let mut b=a.clone();let mut c=a.clone();let (mut den,mut denh,mut lrun,mut mrun)=(0.0,0.0,0.0,f64::NEG_INFINITY);
                        for (slot,&token) in sel.iter().enumerate() {
                            if s[slot]==f64::NEG_INFINITY {continue;}
                            let p=(s[slot]-m).exp();let ph=(sh[slot]-mh).exp();den+=p;denh+=ph;
                            let vb=(token as usize*g.kv_heads+kvh)*g.dim;
                            for d in 0..g.dim {a[d]+=p*v[vb+d] as f64;b[d]+=ph*vh[vb+d];}
                        }
                        for base in (0..len).step_by(128) {
                            let stop=(base+128).min(len);let mt=sh[base..stop].iter().copied().fold(f64::NEG_INFINITY,f64::max);let mn=mrun.max(mt);
                            if mn==f64::NEG_INFINITY {continue;}
                            let alpha=(mrun-mn).exp();lrun*=alpha;for x in &mut c {*x*=alpha;}
                            for slot in base..stop {
                                let x=sh[slot]-mn;exp_inputs.push(x as f32);
                                if sh[slot]==f64::NEG_INFINITY {continue;}
                                let p=x.exp();lrun+=p;let ph=cast16(p);let vb=(sel[slot] as usize*g.kv_heads+kvh)*g.dim;
                                for d in 0..g.dim {c[d]+=ph*vh[vb+d];}
                            }
                            mrun=mn;
                        }
                        for d in 0..g.dim {
                            let gate=1.0+(-(q[qb+g.dim+d] as f64)).exp();
                            exact.push(if den==0.0 {0.0} else {a[d]/den/gate});
                            operands.push(if denh==0.0 {0.0} else {b[d]/denh/gate});
                            probability.push(if lrun==0.0 {0.0} else {c[d]/lrun/gate});
                        }
                    }
                }
                (exact,operands,probability,exp_inputs)
            }));
        }
        jobs.into_iter().map(|j| j.join().map_err(|_| "oracle worker panicked".to_string())).collect::<Result<Vec<_>>>()
    })?;
    let (mut a,mut b,mut c,mut scores)=(Vec::new(),Vec::new(),Vec::new(),Vec::new());
    for (aa,bb,cc,ss) in parts {a.extend(aa);b.extend(bb);c.extend(cc);scores.extend(ss);}
    Ok((a,b,c,scores))
}

struct Inputs {q:GpuTensor,k:GpuTensor,v:GpuTensor,selected:GpuTensor,out:GpuTensor,exact:GpuTensor}
impl Inputs {
    fn upload(gpu:&mut Gpu,g:&Geometry,q:&[f32],k:&[u8],v:&[u8],s:&[i32])->Result<Self> {
        let cache=|gpu:&mut Gpu,b:&[u8]| if g.fp8 {gpu.upload_raw(b,&[b.len()]).map_err(err)} else {gpu.upload_f32(&f32_values(b)?,&[b.len()/4]).map_err(err)};
        let sb:Vec<u8>=s.iter().flat_map(|x|x.to_le_bytes()).collect();
        Ok(Self {q:gpu.upload_f32(q,&[q.len()]).map_err(err)?,k:cache(gpu,k)?,v:cache(gpu,v)?,selected:gpu.upload_raw(&sb,&[sb.len()]).map_err(err)?,out:gpu.zeros(&[g.rows*g.heads*g.dim],DType::F32).map_err(err)?,exact:gpu.zeros(&[g.rows*g.heads*g.dim],DType::F32).map_err(err)?})
    }
    fn params<'a>(&'a self,g:&Geometry,exact:bool)->IndexedAttentionAttentionBatch<'a> {
        IndexedAttentionAttentionBatch {q_with_gate:&self.q,full_keys:&self.k,full_values:&self.v,selected:&self.selected,output:if exact {&self.exact} else {&self.out},rows:g.rows,position_start:g.position_start,n_heads:g.heads,n_kv_heads:g.kv_heads,head_dim:g.dim,budget_blocks:g.budget_blocks,compress:g.compress,capacity:g.capacity,full_capacity:g.full_capacity,format:if g.fp8 {QsaKvFormat::Fp8} else {QsaKvFormat::F32},shape_selected:g.capacity}
    }
    fn free(self,gpu:&mut Gpu)->Result<()> {for t in [self.q,self.k,self.v,self.selected,self.out,self.exact] {gpu.free_tensor(t).map_err(err)?;}Ok(())}
}

fn timed(gpu:&mut Gpu,mut launch:impl FnMut(&mut Gpu)->Result<()>,cold:Option<&GpuTensor>)->Result<Value> {
    for _ in 0..3 {launch(gpu)?;}gpu.hip.device_synchronize().map_err(err)?;
    let a=gpu.hip.event_create().map_err(err)?;let b=gpu.hip.event_create().map_err(err)?;let mut times=Vec::new();
    for _ in 0..20 {
        if let Some(t)=cold {gpu.hip.memset(&t.buf,0,t.byte_size()).map_err(err)?;gpu.hip.device_synchronize().map_err(err)?;}
        gpu.hip.event_record(&a,None).map_err(err)?;launch(gpu)?;gpu.hip.event_record(&b,None).map_err(err)?;
        gpu.hip.event_synchronize(&b).map_err(err)?;times.push(gpu.hip.event_elapsed_ms(&a,&b).map_err(err)?);
    }
    gpu.hip.event_destroy(a).map_err(err)?;gpu.hip.event_destroy(b).map_err(err)?;
    let mut sorted=times.clone();sorted.sort_by(f32::total_cmp);
    Ok(json!({"ms":times,"median_ms":(sorted[9]+sorted[10])*0.5,"min_ms":sorted[0],"warmups":3}))
}

const HALO:&str=include_str!("../../../kernels/src/indexed_attention_gathered_wmma.gfx1151.hip");
const GFX12:&str=include_str!("../../../kernels/src/indexed_attention_gathered_wmma.gfx1201.hip");

struct Gather {k:GpuTensor,v:GpuTensor,symbol:String,src:String,module:String}
impl Gather {
    fn new(gpu:&mut Gpu,g:&Geometry,variant:&str)->Result<Self> {
        let src=if g.fp8 {GFX12} else {HALO};let original=if g.fp8 {"indexed_attention_gathered_wmma_f16_gfx1201"} else {"indexed_attention_gathered_wmma_f16"};
        let symbol=format!("qsa_evidence_{variant}");let mut src=src.replace(original,&symbol);
        if variant=="score" || variant=="score_dump" {
            if variant=="score" {
                let at=src.find("    float m_run = -INFINITY, l_run = 0.f;").ok_or("missing score accumulator anchor")?;
                src.insert_str(at,"    float diagnostic_sum=0.f;\n");
            }
            let first=src.find("        lmax = fmaxf(lmax, qg_other_half(lmax));").ok_or("missing score clone start")?;
            let last=src.find("    if (l16 >= group) return;").ok_or("missing score clone end")?;
            let sink=if variant=="score_dump" {
                "        for (int i=0;i<8;++i) if (kb+QG_CROW(i,half)<((capacity+127)/128*128)) output[(((size_t)row*n_kv_heads+kvh)*16+l16)*((capacity+127)/128*128)+kb+QG_CROW(i,half)]=sv[i];\n    }\n    return;\n"
            } else {"        diagnostic_sum += sv[tid & 7] == -INFINITY ? 0.f : sv[tid & 7];\n    }\n    output[((size_t)row*n_kv_heads+kvh)*256+tid]=diagnostic_sum;\n    return;\n"};
            src.replace_range(first..last,sink);
        } else if variant=="no_pv" {
            let first=src.find("        for (int kc = 0; kc < 8; ++kc) {").ok_or("missing PV clone start")?;
            let last=src.find("    if (l16 >= group) return;").ok_or("missing PV clone end")?;
            src.replace_range(first..last,"        for (int i = 0; i < 8; ++i) o[0][i] += (float)p_lds[par][wave][l16 * 16 + i];\n    }\n");
        } else if variant=="softmax" || variant=="pv" {
            let first=src.find("        if (kb < selected_len) {").ok_or("missing score block")?;
            let last=src.find("        lmax = fmaxf(lmax, qg_other_half(lmax));").ok_or("missing score end")?;
            src.replace_range(first..last,
                "        for (int i=0;i<8;++i) { sv[i]=((const float*)k16)[(((size_t)row*n_kv_heads+kvh)*16+l16)*((capacity+127)/128*128)+kb+QG_CROW(i,half)]; lmax=fmaxf(lmax,sv[i]); }\n");
            if variant=="softmax" {
                let first=src.find("        for (int kc = 0; kc < 8; ++kc) {").ok_or("missing PV block")?;
                let last=src.find("    if (l16 >= group) return;").ok_or("missing PV end")?;
                src.replace_range(first..last,"        for (int i=0;i<8;++i) o[0][i]+=(float)p_lds[par][wave][l16*16+QG_CROW(i,half)];\n    }\n");
            } else {
                let first=src.find("        for (int i=0;i<8;++i) { sv[i]=").ok_or("missing replacement score block")?;
                let last=src.find("        for (int kc = 0; kc < 8; ++kc) {").ok_or("missing PV block")?;
                src.replace_range(first..last,
                    "        for (int i=0;i<8;++i) p_lds[par][wave][l16*16+QG_CROW(i,half)]=(_Float16)((const float*)k16)[(((size_t)row*n_kv_heads+kvh)*16+l16)*((capacity+127)/128*128)+kb+QG_CROW(i,half)];\n        l_run=1.f; m_run=0.f;\n        __syncthreads();\n");
            }
        } else if variant=="no_exp" {
            src=src.replace("alpha = expf(m_run - m_new);","alpha = 1.f;").replace("const float p = expf(sv[i] - m_new);","const float p = 1.f / (1.f + fabsf(sv[i] - m_new));");
        } else if variant=="contiguous" {
            src=src.replace("tokens[i] = (token >= 0 && token < full_capacity && token < visible) ? token : -1;",
                "tokens[i] = (token >= 0 && token < full_capacity && token < visible) ? i : -1;");
            src=src.replace("const _Float16* kbase = k16 + kvh * HD;",
                "const _Float16* kbase = k16 + (size_t)row * ((capacity + 3) / 4 * 4) * width + kvh * HD;\n    vb16 += (size_t)row * ((capacity + 3) / 4 * 4) * width;");
        } else if variant!="full" {return Err(format!("unknown gathered diagnostic {variant}"));}
        let module=format!("qsa_evidence_{}_{}",if g.fp8 {"gfx1201"} else {"gfx1151"},variant);
        gpu.ensure_kernel_public(&module,&src,&symbol).map_err(err)?;
        let tokens=if variant=="contiguous" {g.rows*g.capacity.div_ceil(4)*4} else {g.position_start+g.rows};let width=g.kv_heads*g.dim;
        Ok(Self {k:gpu.zeros(&[tokens*width],DType::F16).map_err(err)?,v:gpu.zeros(&[tokens.div_ceil(4)*4*width],DType::F16).map_err(err)?,symbol,src,module})
    }
    fn convert(&self,gpu:&mut Gpu,g:&Geometry,x:&Inputs)->Result<()> {
        let symbol=if g.fp8 {"indexed_attention_kv_f16vb_fp8_gfx1201"} else {"indexed_attention_kv_f16vb"};
        gpu.ensure_kernel_public(&self.module,&self.src,symbol).map_err(err)?;
        let mut args=hip_bridge::KernargBlob::new();
        for p in [&x.k,&x.v,&self.k,&self.v] {args.push_ptr(p.buf.as_ptr());}
        args.push_i32((g.position_start+g.rows) as i32);args.push_i32(g.kv_heads as i32);args.pad_to(16);
        gpu.launch_kernel_blob(symbol,[(g.position_start+g.rows) as u32,g.kv_heads as u32,1],[256,1,1],0,args.as_mut_slice()).map_err(err)
    }
    fn launch(&self,gpu:&mut Gpu,g:&Geometry,x:&Inputs)->Result<()> {
        self.launch_overrides(gpu,g,x,&self.k,&x.out)
    }
    fn launch_overrides(&self,gpu:&mut Gpu,g:&Geometry,x:&Inputs,k:&GpuTensor,out:&GpuTensor)->Result<()> {
        let mut args=hip_bridge::KernargBlob::new();for p in [&x.q,k,&self.v,&x.selected,out] {args.push_ptr(p.buf.as_ptr());}
        for n in [g.rows,g.position_start,g.heads,g.kv_heads,g.budget_blocks,g.compress,g.capacity,g.full_capacity] {args.push_i32(n as i32);}args.pad_to(16);
        // A compress tail is non-monotone across rows: the last row may end
        // on a full block while earlier rows need up to compress-1 more slots.
        let max_selected=(g.position_start+g.rows).min(g.capacity).min(g.budget_blocks*g.compress+g.compress-1);
        gpu.launch_kernel_blob(&self.symbol,[g.rows as u32,g.kv_heads as u32,1],[256,1,1],(max_selected.div_ceil(128)*128*4) as u32,args.as_mut_slice()).map_err(err)
    }
    fn free(self,gpu:&mut Gpu)->Result<()> {gpu.free_tensor(self.k).map_err(err)?;gpu.free_tensor(self.v).map_err(err)}
    fn pack_contiguous(&self,gpu:&mut Gpu,g:&Geometry,source:&Gather,s:&[i32])->Result<()> {
        // Rearrange the already-converted GPU bits: this clone must measure
        // addresses alone, not introduce a second host-side F16 conversion.
        let mut k=vec![0u8;source.k.byte_size()];let mut v=vec![0u8;source.v.byte_size()];
        gpu.hip.memcpy_dtoh(&mut k,&source.k.buf).map_err(err)?;
        gpu.hip.memcpy_dtoh(&mut v,&source.v.buf).map_err(err)?;
        let stride=g.capacity.div_ceil(4)*4;let width=g.kv_heads*g.dim;
        let mut kk=vec![0u8;g.rows*stride*width*2];let mut vv=vec![0u8;kk.len()];
        for row in 0..g.rows {for slot in 0..g.selected_len(row) {
            let token=s[row*g.capacity+slot];
            if token<0 || token as usize>=g.position_start+row+1 || token as usize>=g.full_capacity {continue;}
            let token=token as usize;
            for d in 0..width {
                let a=((row*stride+slot)*width+d)*2;let b=(token*width+d)*2;
                kk[a..a+2].copy_from_slice(&k[b..b+2]);
                let a=(row*stride*width+(slot/4*width+d)*4+slot%4)*2;
                let b=((token/4*width+d)*4+token%4)*2;
                vv[a..a+2].copy_from_slice(&v[b..b+2]);
            }
        }}
        gpu.hip.memcpy_htod(&self.k.buf,&kk).map_err(err)?;gpu.hip.memcpy_htod(&self.v.buf,&vv).map_err(err)?;
        Ok(())
    }
}

fn exercise(gpu:&mut Gpu,g:&Geometry,q:&[f32],k:&[u8],v:&[u8],s:&[i32],out:&Path,with_oracle:bool)->Result<Value> {
    std::fs::create_dir_all(out).map_err(err)?;
    let x=Inputs::upload(gpu,g,q,k,v,s)?;
    indexed_attention_attention_batch_exact(gpu,&x.params(g,true)).map_err(err)?;
    let reference=gpu.download_f32(&x.exact).map_err(err)?;
    let grouped=timed(gpu,|gpu| indexed_attention_attention_batch(gpu,&x.params(g,false)).map_err(err),None)?;
    let hg=gpu.download_f32(&x.out).map_err(err)?;
    let gather=Gather::new(gpu,g,"full")?;
    let converted=timed(gpu,|gpu| {gather.convert(gpu,g,&x)?;gather.launch(gpu,g,&x)},None)?;
    let warm=timed(gpu,|gpu| gather.launch(gpu,g,&x),None)?;
    let candidate=gpu.download_f32(&x.out).map_err(err)?;
    let cold=gpu.zeros(&[512*1024*1024],DType::Raw).map_err(err)?;
    let cold_time=timed(gpu,|gpu| gather.launch(gpu,g,&x),Some(&cold))?;
    gpu.free_tensor(cold).map_err(err)?;
    let mut diagnostics=serde_json::Map::new();
    for name in ["score","no_pv","no_exp"] {
        let clone=Gather::new(gpu,g,name)?;clone.convert(gpu,g,&x)?;
        diagnostics.insert(name.into(),timed(gpu,|gpu| clone.launch(gpu,g,&x),None)?);
        std::fs::write(out.join(format!("diagnostic-{name}.hip")),&clone.src).map_err(err)?;clone.free(gpu)?;
    }
    let contiguous=Gather::new(gpu,g,"contiguous")?;
    contiguous.pack_contiguous(gpu,g,&gather,s)?;
    let contiguous_time=timed(gpu,|gpu| contiguous.launch(gpu,g,&x),None)?;
    let packed=gpu.download_f32(&x.out).map_err(err)?;
    let contiguous_parity=metrics(&candidate,&packed,g.heads,g.dim,None)?;
    std::fs::write(out.join("contiguous-parity.json"),serde_json::to_vec_pretty(&contiguous_parity).map_err(err)?).map_err(err)?;
    std::fs::write(out.join("diagnostic-contiguous.hip"),&contiguous.src).map_err(err)?;
    if contiguous_parity["differing_bits"]!=0 {return Err(format!("contiguous diagnostic changed same-value gathered output: {contiguous_parity}"));}
    contiguous.free(gpu)?;
    let score_dump=Gather::new(gpu,g,"score_dump")?;score_dump.convert(gpu,g,&x)?;
    let stride=g.capacity.div_ceil(128)*128;
    let scores=gpu.zeros(&[g.rows*g.kv_heads*16*stride],DType::F32).map_err(err)?;
    score_dump.launch_overrides(gpu,g,&x,&score_dump.k,&scores)?;
    let mut probabilities=gpu.download_f32(&scores).map_err(err)?;
    let mut exp_inputs=Vec::new();
    for (rh,row) in probabilities.chunks_exact_mut(stride).enumerate() {
        let query_row=rh/(g.kv_heads*16);let head=rh%16;let len=g.selected_len(query_row);
        let mut running=f32::NEG_INFINITY;
        for tile in row[..len].chunks_mut(128) {
            let maximum=tile.iter().copied().fold(running,f32::max);
            for value in tile {
                let delta=*value-maximum;
                if head<g.heads/g.kv_heads {exp_inputs.push(if maximum==f32::NEG_INFINITY {f32::NEG_INFINITY} else {delta});}
                *value=if maximum==f32::NEG_INFINITY {0.0} else {(delta as f64).exp() as f32};
            }
            running=maximum;
        }
        row[len..].fill(0.0);
    }
    let p_gpu=gpu.upload_f32(&probabilities,&[probabilities.len()]).map_err(err)?;
    for (name,input) in [("softmax",&scores),("pv",&p_gpu)] {
        let clone=Gather::new(gpu,g,name)?;clone.convert(gpu,g,&x)?;
        diagnostics.insert(name.into(),timed(gpu,|gpu| clone.launch_overrides(gpu,g,&x,input,&x.out),None)?);
        std::fs::write(out.join(format!("diagnostic-{name}.hip")),&clone.src).map_err(err)?;clone.free(gpu)?;
    }
    std::fs::write(out.join("exp-inputs-gpu.f32"),f32_bytes(&exp_inputs)).map_err(err)?;
    let exp_timing=exp_throughput(gpu,&exp_inputs)?;
    gpu.free_tensor(scores).map_err(err)?;gpu.free_tensor(p_gpu).map_err(err)?;score_dump.free(gpu)?;
    // Poison scratch and reconstruct conversion; never restore model state.
    gpu.hip.memset(&gather.k.buf,0x7f,gather.k.byte_size()).map_err(err)?;
    gpu.hip.memset(&gather.v.buf,0x7f,gather.v.byte_size()).map_err(err)?;
    gather.convert(gpu,g,&x)?;gather.launch(gpu,g,&x)?;
    let restored=gpu.download_f32(&x.out).map_err(err)?;
    let scratch_reset=metrics(&candidate,&restored,g.heads,g.dim,None)?;
    let mut rollback_geometry=g.clone();rollback_geometry.position_start=g.position_start.saturating_sub(4);
    gather.convert(gpu,&rollback_geometry,&x)?;gather.launch(gpu,&rollback_geometry,&x)?;
    let rollback_old=gpu.download_f32(&x.out).map_err(err)?;
    let reloaded=Gather::new(gpu,&rollback_geometry,"full")?;
    reloaded.convert(gpu,&rollback_geometry,&x)?;reloaded.launch(gpu,&rollback_geometry,&x)?;
    let rollback_new=gpu.download_f32(&x.out).map_err(err)?;
    let scratch_rollback_reload=metrics(&rollback_old,&rollback_new,g.heads,g.dim,None)?;
    reloaded.free(gpu)?;
    if scratch_reset["differing_bits"]!=0 || scratch_rollback_reload["differing_bits"]!=0 {
        return Err("stale gathered scratch after reset/rollback/reload".into());
    }
    let stats=metrics(&reference,&candidate,g.heads,g.dim,Some(&out.join("G-vs-R.elements")))?;
    let selected_bytes:Vec<u8>=s.iter().flat_map(|x|x.to_le_bytes()).collect();
    dump_snapshot(out,json!({"arch":gpu.arch,"developer_fixture":true}),g,&[("qgate.f32",&f32_bytes(q)),("keys.source",k),("values.source",v),("selected.i32",&selected_bytes),("eager.f32",&f32_bytes(&candidate)),("exact.f32",&f32_bytes(&reference))],stats.clone())?;
    let mut result=json!({"arch":gpu.arch,"geometry":g.json(),"hg4":grouped,"hg4_vs_exact":metrics(&reference,&hg,g.heads,g.dim,None)?,"gather_conversion":converted,"gather_cached":warm,"gather_cold_512MiB_flush":cold_time,"diagnostics_nonadditive":diagnostics,"scratch_reset":scratch_reset,"G_vs_R":stats});
    result["contiguous_same_values"]=json!({"timing":contiguous_time,"vs_original":contiguous_parity,"includes":"per-query duplicate source values; no packing cost timed; working set is larger than original"});
    result["expf_real_gpu_scores"]=exp_timing;
    result["scratch_rollback_reload"]=scratch_rollback_reload;
    if with_oracle {
        let (r,o,p,exp)=oracle(q,&decode_cache(k,g)?,&decode_cache(v,g)?,s,g)?;
        for (name,data) in [("oracle.f64",&r),("operand-casts.f64",&o),("probability-casts.f64",&p)] {std::fs::write(out.join(name),f64_bytes(data)).map_err(err)?;}
        std::fs::write(out.join("exp-inputs.f32"),f32_bytes(&exp)).map_err(err)?;
        result["exact_vs_f64"]=metrics_f64(&r,&reference,g.heads,g.dim)?;
        result["operand_rounding"]=metrics_f64_pair(&r,&o,g.heads,g.dim)?;
        result["probability_rounding"]=metrics_f64_pair(&o,&p,g.heads,g.dim)?;
        result["association_and_f32_transcendentals"]=metrics_f64(&p,&candidate,g.heads,g.dim)?;
    }
    std::fs::write(out.join("manifest.json"),serde_json::to_vec_pretty(&result).map_err(err)?).map_err(err)?;
    gather.free(gpu)?;x.free(gpu)?;Ok(result)
}

fn fixture(g:&Geometry,kind:&str)->(Vec<f32>,Vec<u8>,Vec<u8>,Vec<i32>) {
    let lcg=|seed:usize,n:usize| (0..n).map(|i| ((i.wrapping_mul(2_654_435_761).wrapping_add(seed)%2003) as f32-1001.0)/997.0).collect::<Vec<_>>();
    let mut q=lcg(1,g.rows*g.heads*2*g.dim);let k=lcg(7,g.full_capacity*g.kv_heads*g.dim);let mut v=lcg(13,k.len());
    if kind=="sharp" {for row in q.chunks_exact_mut(g.heads*2*g.dim) {for h in 0..g.heads {for x in &mut row[h*2*g.dim..h*2*g.dim+g.dim] {*x*=8.0;}}}}
    if kind=="flat-cancel" {for row in q.chunks_exact_mut(g.heads*2*g.dim) {for h in 0..g.heads {row[h*2*g.dim..h*2*g.dim+g.dim].fill(0.0);}}
        for (t,row) in v.chunks_exact_mut(g.kv_heads*g.dim).enumerate() {for (d,x) in row.iter_mut().enumerate() {*x=if t%2==0 {1.0+d as f32/1024.0} else {-1.0-d as f32/1024.0};}}}
    if kind=="gate" {for row in q.chunks_exact_mut(g.heads*2*g.dim) {for h in 0..g.heads {for (d,x) in row[h*2*g.dim+g.dim..(h+1)*2*g.dim].iter_mut().enumerate() {*x=if d%2==0 {100.0} else {-100.0};}}}}
    let mut s=vec![-1;g.rows*g.capacity];
    for row in 0..g.rows {let visible=g.position_start+row+1;let blocks=visible/g.compress;let chosen=g.budget_blocks.min(blocks);
        for slot in 0..chosen {let b=blocks-1-(slot*7+row)%blocks;for d in 0..g.compress {let i=slot*g.compress+d;if i<g.capacity {s[row*g.capacity+i]=(b*g.compress+d) as i32;}}}
        for token in blocks*g.compress..visible {let i=chosen*g.compress+token-blocks*g.compress;if i<g.capacity {s[row*g.capacity+i]=token as i32;}}
        match kind {"invalid"=>s[row*g.capacity..(row+1)*g.capacity].fill(-1),"oow"=>{s[row*g.capacity]=g.full_capacity as i32;if g.capacity>1 {s[row*g.capacity+1]=visible as i32;}},"duplicate"=>{if g.capacity>1 {s[row*g.capacity+1]=s[row*g.capacity];}},"arbitrary"=>s[row*g.capacity..row*g.capacity+g.selected_len(row)].reverse(),"flat-cancel"=>{},_=>{if g.selected_len(row)>3 {s[row*g.capacity+3]=-1;}}}
    }
    if g.fp8 {
        let encode=|seed:usize| {
            let mut b=vec![0u8;g.full_capacity*g.row_bytes()];
            for (t,row) in b.chunks_exact_mut(g.row_bytes()).enumerate() {for h in 0..g.kv_heads {
                let scales=if kind=="scales" {[0x0001u16,0x7bff,0x3c00,0x3800]} else {[0x2000u16,0x2400,0x2800,0x2c00]};
                let scale=if kind=="flat-cancel" {0x3c00u16} else {scales[(t+h)%4]};
                let width=g.kv_heads*g.dim;row[width+2*h..width+2*h+2].copy_from_slice(&scale.to_le_bytes());
                for d in 0..g.dim {
                    let codes=[0u8,0x80,1,7,8,0x77,0x78,0x7e,0xfe,0x88];
                    row[h*g.dim+d]=if kind=="fp8-boundaries" || kind=="scales" {codes[(t*7+h*3+d+seed)%codes.len()]}
                        else if kind=="flat-cancel" && seed==13 {0x38 | if t%2==0 {0} else {0x80}}
                        else {((t*31+h*17+d*13+seed)%0x77) as u8 | if (t+d)%2==0 {0x80} else {0}};
                }
            }}
            b
        };
        (q,encode(7),encode(13),s)
    } else {(q,f32_bytes(&k),f32_bytes(&v),s)}
}

fn self_check()->Result<()> {
    for (input,bits) in [(1.00048828125f32,0x3c00u16),(1.00146484375,0x3c02),
        (2f32.powi(-24),1),(2f32.powi(-25),0),(65504.0,0x7bff),(65520.0,0x7c00),(-0.0,0x8000)] {
        if half_rne(input)!=bits {return Err(format!("F16 RNE boundary failed for {input}"));}
    }
    let r=[0.0,-0.0,1.0,-1.0,f32::INFINITY];let c=[-0.0,1.0,f32::from_bits(1.0f32.to_bits()+1),-1.0,f32::INFINITY];
    let m=metrics(&r,&c,1,5,None)?;
    if m["raw_max_rel"]!="Infinity" || m["differing_bits"]!=3 || ulp(0.0,-0.0)!=0 || ulp(1.0,c[2])!=1 || m["nonfinite_mismatch"]!=0 {return Err(format!("metric semantics failed {m}"));}
    if m["floored_reporting"]["included_elements"]!=2 || m["floored_reporting"]["max_ulp"]!=1 {return Err("floored reporting included a zero/nonfinite reference".into());}
    let g=Geometry{rows:1,position_start:0,heads:1,kv_heads:1,dim:2,budget_blocks:1,compress:4,capacity:3,full_capacity:1,fp8:false};
    let (a,_,_,_)=oracle(&[0.0,0.0,0.0,0.0],&[1.0,1.0],&[2.0,-2.0],&[0,-1,-1],&g)?;
    if a!=[1.0,-1.0] {return Err(format!("same-source oracle failed {a:?}"));}
    let two=Geometry{rows:1,position_start:1,heads:1,kv_heads:1,dim:1,budget_blocks:1,compress:4,capacity:3,full_capacity:2,fp8:false};
    let isolated=isolated_oracles(&[1.0,0.0],&[0.0,-1.0],&[2.0,-2.0],&[0,1,-1],&two)?;
    let p=(-1.0f64).exp();let expected=(1.0-cast16(p))/(1.0+p);
    if (isolated[3][0]-expected).abs()>1e-15 || isolated[3]==isolated[0] || isolated[1]!=isolated[0] || isolated[2]!=isolated[0] {
        return Err("P-only isolation changed the denominator or cast an unrelated operand".into());
    }
    println!("metric signed-zero/raw-rel/ULP/nonfinite and same-source oracle checks passed");Ok(())
}

fn snapshot_inputs(path:&Path)->Result<(Value,Geometry,Vec<f32>,Vec<u8>,Vec<u8>,Vec<i32>,Vec<f32>,Vec<f32>)> {
    let root=path.parent().ok_or("snapshot has no parent")?;
    let header:Value=serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)?;
    if header["schema"]!="qsa-source-v1" {return Err("unsupported snapshot schema".into());}
    let get=|name:&str|->Result<Vec<u8>> {let b=std::fs::read(root.join(name)).map_err(err)?;
        if header["files"][name]["sha256"]!=sha256(&b) {return Err(format!("snapshot hash mismatch {name}"));}Ok(b)};
    let g=Geometry::parse(&header["geometry"])?;let q=f32_values(&get("qgate.f32")?)?;
    let k=get("keys.source")?;let v=get("values.source")?;let s=i32_values(&get("selected.i32")?)?;
    let r=f32_values(&get("exact.f32")?)?;let c=f32_values(&get("eager.f32")?)?;
    Ok((header,g,q,k,v,s,r,c))
}

fn f64_bytes(v:&[f64])->Vec<u8> {v.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn metrics_f64(r:&[f64],c:&[f32],heads:usize,dim:usize)->Result<Value> {
    metrics_f64_pair(r,&c.iter().map(|&x|x as f64).collect::<Vec<_>>(),heads,dim)
}
fn metrics_f64_pair(r:&[f64],c:&[f64],heads:usize,dim:usize)->Result<Value> {
    if r.len()!=c.len() || r.len()%(heads*dim)!=0 {return Err("f64 metric shape mismatch".into());}
    let rf:Vec<f32>=r.iter().map(|&x|x as f32).collect();let cf:Vec<f32>=c.iter().map(|&x|x as f32).collect();
    let mut rounded=metrics(&rf,&cf,heads,dim,None)?;
    let (mut ma,mut mr,mut max_a,mut ia,mut ir)=(0.0f64,0.0f64,0.0f64,0usize,0usize);let mut row_heads=Vec::new();
    for (rh,(rr,cc)) in r.chunks_exact(dim).zip(c.chunks_exact(dim)).enumerate() {
        let ref_max=rr.iter().filter(|x|x.is_finite()).fold(0.0f64,|m,x|m.max(x.abs()));let mut ra=0.0f64;let mut rel=0.0f64;
        for (d,(&a,&b)) in rr.iter().zip(cc).enumerate() {
            let abs=if a==b || a.to_bits()==b.to_bits() {0.0} else if a.is_finite() && b.is_finite() {(a-b).abs()} else {f64::INFINITY};
            let raw=if a==0.0 {if b==0.0 {0.0} else {f64::INFINITY}} else {abs/a.abs()};
            if abs>ma {ma=abs;ia=rh*dim+d;}if raw>mr {mr=raw;ir=rh*dim+d;}ra=ra.max(abs);rel=rel.max(raw);
        }
        let a=ra/ref_max.max(1e-30);max_a=max_a.max(a);row_heads.push(json!({"row":rh/heads,"head":rh%heads,"max_abs":number(ra),"raw_max_rel":number(rel),"A":number(a)}));
    }
    let worst=|i:usize| json!({"flat_index":i,"row":i/(heads*dim),"head":i/dim%heads,"channel":i%dim,"reference_f64":number(r[i]),"candidate_f64":number(c[i])});
    rounded["f64_distances"]=json!({"max_abs":number(ma),"raw_max_rel":number(mr),"max_A":number(max_a),"worst_abs":worst(ia),"worst_raw_rel":worst(ir),"row_heads":row_heads});
    rounded["ULP_note"]=json!("ULP and bit identity compare each f64 output rounded once to F32; f64_distances are unrounded");
    Ok(rounded)
}

/// Controlled, one-change-at-a-time mathematical isolation. Arithmetic stays
/// f64 except the named casts; this is not GPU WMMA or device-expf emulation.
fn isolated_oracles(q:&[f32],k:&[f32],v:&[f32],s:&[i32],g:&Geometry)->Result<[Vec<f64>;5]> {
    let kh:Vec<f64>=k.iter().map(|&x|cast16(x as f64)).collect();
    let vh:Vec<f64>=v.iter().map(|&x|cast16(x as f64)).collect();
    let workers=std::thread::available_parallelism().map_or(1,|n|n.get()).min(32);
    let count=g.rows.div_ceil(workers).max(1);
    let parts=std::thread::scope(|scope| {
        let mut jobs=Vec::new();
        for first in (0..g.rows).step_by(count) {
            let last=(first+count).min(g.rows);let (kh,vh)=(&kh,&vh);
            jobs.push(scope.spawn(move || {
                let mut outputs:[Vec<f64>;5]=std::array::from_fn(|_|Vec::with_capacity((last-first)*g.heads*g.dim));
                for row in first..last {for head in 0..g.heads {
                    let kvh=head/(g.heads/g.kv_heads);let qb=(row*g.heads+head)*g.dim*2;
                    let len=g.selected_len(row);let selected=&s[row*g.capacity..row*g.capacity+len];
                    let qh:Vec<f64>=q[qb..qb+g.dim].iter().map(|&x|cast16(x as f64)).collect();
                    let mut scores:[Vec<f64>;3]=std::array::from_fn(|_|vec![f64::NEG_INFINITY;len]);
                    for (slot,&token) in selected.iter().enumerate() {
                        if token<0 || token as usize>=g.position_start+row+1 || token as usize>=g.full_capacity {continue;}
                        let kb=(token as usize*g.kv_heads+kvh)*g.dim;let mut dot=[0.0f64;3];
                        for d in 0..g.dim {
                            dot[0]+=q[qb+d] as f64*k[kb+d] as f64;
                            dot[1]+=q[qb+d] as f64*kh[kb+d];dot[2]+=qh[d]*k[kb+d] as f64;
                        }
                        for stage in 0..3 {scores[stage][slot]=dot[stage]/(g.dim as f64).sqrt();}
                    }
                    let maxima:[f64;3]=std::array::from_fn(|i|scores[i].iter().copied().fold(f64::NEG_INFINITY,f64::max));
                    let mut denominator=[0.0f64;5];let mut values:[Vec<f64>;5]=std::array::from_fn(|_|vec![0.0;g.dim]);
                    for (slot,&token) in selected.iter().enumerate() {
                        if scores[0][slot]==f64::NEG_INFINITY {continue;}
                        let vb=(token as usize*g.kv_heads+kvh)*g.dim;
                        let p:[f64;3]=std::array::from_fn(|i|(scores[i][slot]-maxima[i]).exp());
                        for stage in 0..3 {denominator[stage]+=p[stage];}denominator[3]+=p[0];
                        let p16=cast16(p[0]);
                        for d in 0..g.dim {
                            values[0][d]+=p[0]*v[vb+d] as f64;values[1][d]+=p[1]*vh[vb+d];
                            values[2][d]+=p[2]*v[vb+d] as f64;values[3][d]+=p16*v[vb+d] as f64;
                        }
                    }
                    let mut running=f64::NEG_INFINITY;
                    for base in (0..len).step_by(128) {
                        let end=(base+128).min(len);let maximum=scores[0][base..end].iter().copied().fold(running,f64::max);
                        let alpha=if maximum==f64::NEG_INFINITY {0.0} else {(running-maximum).exp()};
                        denominator[4]*=alpha;for value in &mut values[4] {*value*=alpha;}
                        for slot in base..end {
                            if scores[0][slot]==f64::NEG_INFINITY {continue;}
                            let p=(scores[0][slot]-maximum).exp();denominator[4]+=p;
                            let vb=(selected[slot] as usize*g.kv_heads+kvh)*g.dim;
                            for d in 0..g.dim {values[4][d]+=p*v[vb+d] as f64;}
                        }
                        running=maximum;
                    }
                    for d in 0..g.dim {
                        let gate=1.0/(1.0+(-(q[qb+g.dim+d] as f64)).exp());
                        for stage in 0..5 {outputs[stage].push(if denominator[stage]==0.0 {0.0} else {values[stage][d]/denominator[stage]*gate});}
                    }
                }}
                outputs
            }));
        }
        jobs.into_iter().map(|job|job.join().map_err(|_|"isolation worker panicked".to_string())).collect::<Result<Vec<_>>>()
    })?;
    let mut output:[Vec<f64>;5]=std::array::from_fn(|_|Vec::with_capacity(g.rows*g.heads*g.dim));
    for part in parts {for (target,source) in output.iter_mut().zip(part) {target.extend(source);}}
    Ok(output)
}

fn isolate_snapshot(path:&Path,out:&Path)->Result<()> {
    let (header,g,q,k,v,s,r,c)=snapshot_inputs(path)?;std::fs::create_dir_all(out).map_err(err)?;
    let values=isolated_oracles(&q,&decode_cache(&k,&g)?,&decode_cache(&v,&g)?,&s,&g)?;
    let labels=["source-quantized-f64","kv-f16-only","q-f16-only","p-f16-only","online-order-f64-only"];
    let mut stages=serde_json::Map::new();
    let gpu_reference:Vec<f64>=r.iter().map(|&x|x as f64).collect();
    for (stage,label) in labels.iter().enumerate() {
        std::fs::write(out.join(format!("{label}.f64")),f64_bytes(&values[stage])).map_err(err)?;
        stages.insert((*label).into(),json!({"vs_mathematical_source":metrics_f64_pair(&values[0],&values[stage],g.heads,g.dim)?,
            "vs_exact_gpu_R":metrics_f64_pair(&gpu_reference,&values[stage],g.heads,g.dim)?}));
    }
    let report=json!({"identity":header["identity"],"snapshot":path,"snapshot_sha256":sha256(&std::fs::read(path).map_err(err)?),
        "G_vs_R":metrics(&r,&c,g.heads,g.dim,None)?,"stages":stages,
        "contract":"one swap at a time from same-source f64 mathematical reference; nonadditive; online-only retains f64 to isolate order, not GPU expf or WMMA association"});
    std::fs::write(out.join("isolation-manifest.json"),serde_json::to_vec_pretty(&report).map_err(err)?).map_err(err)?;
    println!("isolated {}",path.display());Ok(())
}

fn analyze_snapshot(path:&Path,out:&Path)->Result<()> {
    let (header,g,q,k,v,s,r,c)=snapshot_inputs(path)?;
    std::fs::create_dir_all(out).map_err(err)?;
    let (oracle,operands,probabilities,scores)=oracle(&q,&decode_cache(&k,&g)?,&decode_cache(&v,&g)?,&s,&g)?;
    let report=json!({"identity":header["identity"],"snapshot":path,"snapshot_sha256":sha256(&std::fs::read(path).map_err(err)?),
        "G_vs_R":metrics(&r,&c,g.heads,g.dim,Some(&out.join("G-vs-R.elements")))?,
        "exact_vs_quantized_f64":metrics_f64(&oracle,&r,g.heads,g.dim)?,
        "operand_rounding":metrics_f64_pair(&oracle,&operands,g.heads,g.dim)?,
        "probability_rounding":metrics_f64_pair(&operands,&probabilities,g.heads,g.dim)?,
        "association_and_f32_transcendentals":metrics_f64(&probabilities,&c,g.heads,g.dim)?,
        "oracle_note":"unrounded f64 outputs retained; F32-distance metrics round oracle once; all unnormalized F16 probability casts retain original 128-slot tiles and masking",
        "exp_distribution_note":"same-source F16-cast f64 QK scores minus each running tile maximum, not synthetic uniform scores"});
    for (name,data) in [("oracle.f64",&oracle),("operand-casts.f64",&operands),("probability-casts.f64",&probabilities)] {std::fs::write(out.join(name),f64_bytes(data)).map_err(err)?;}
    std::fs::write(out.join("exp-inputs.f32"),f32_bytes(&scores)).map_err(err)?;
    std::fs::write(out.join("oracle-manifest.json"),serde_json::to_vec_pretty(&report).map_err(err)?).map_err(err)?;
    println!("analyzed {}",path.display());Ok(())
}

fn freeze(root:&Path,out:&Path)->Result<()> {
    let mut layers:std::collections::BTreeMap<u64,Vec<Value>>=std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(root).map_err(err)? {
        let path=entry.map_err(err)?.path().join("snapshot.json");if !path.is_file() {continue;}
        let header:Value=serde_json::from_slice(&std::fs::read(&path).map_err(err)?).map_err(err)?;
        if header["identity"]["phase"]!="all_prefill" || header["identity"]["ctx"]!=8192 {continue;}
        let layer=header["identity"]["layer"].as_u64().ok_or("missing layer identity")?;
        layers.entry(layer).or_default().push(json!({"path":path,"header":header}));
    }
    if layers.len()!=12 {return Err(format!("full pp8192 requires 12 layers, found {}",layers.len()));}
    let mut layer_metrics=Vec::new();let (mut abs,mut rel,mut a,mut ulp)=(0.0f64,0.0f64,0.0f64,0u64);let mut mismatch=0u64;let mut snapshots=Vec::new();
    let numeric=|v:&Value| if v=="Infinity" {f64::INFINITY} else {v.as_f64().unwrap_or(f64::INFINITY)};
    for (layer,mut chunks) in layers {
        chunks.sort_by_key(|c| c["header"]["geometry"]["position_start"].as_u64());
        let mut position=0u64;let (mut la,mut lr,mut ln,mut lu)=(0.0f64,0.0f64,0.0f64,0u64);
        for chunk in chunks {
            let h=&chunk["header"];let g=&h["geometry"];
            if g["position_start"].as_u64()!=Some(position) {return Err(format!("noncontiguous pp8192 layer {layer}"));}
            position+=g["rows"].as_u64().ok_or("missing rows")?;
            let m=&h["eager_vs_exact"];la=la.max(numeric(&m["max_abs"]));lr=lr.max(numeric(&m["raw_max_rel"]));ln=ln.max(numeric(&m["max_A"]));lu=lu.max(m["max_ulp"].as_u64().ok_or("missing measured ULP")?);mismatch+=m["nonfinite_mismatch"].as_u64().ok_or("missing nonfinite count")?;
            snapshots.push(json!({"path":chunk["path"],"identity":h["identity"],"geometry":g,"files":h["files"],"worst_abs":m["worst_abs"],"worst_raw_rel":m["worst_raw_rel"],"worst_ulp":m["worst_ulp"]}));
        }
        if position!=8192 {return Err(format!("layer {layer} has {position} rows, need 8192"));}
        abs=abs.max(la);rel=rel.max(lr);a=a.max(ln);ulp=ulp.max(lu);
        layer_metrics.push(json!({"layer":layer,"max_abs":number(la),"raw_max_rel":number(lr),"max_ulp":lu,"max_A":number(ln)}));
    }
    let manifest=json!({"schema":"qsa-frozen-G-v1","metric_contract":"abs=|C-R|; raw rel=abs/|R|, 0/0=0, nonzero/0=infinity; ordered F32 ULP, equal signed zeros distance0 with separate bit identity",
        "reference":"indexed_attention_attention_batch_impl(...,allow_fast=false), exact same source cache codes/scales",
        "ceilings":{"max_abs":number(abs),"raw_max_rel":number(rel),"max_ulp":ulp,"max_A":number(a)},"per_layer":layer_metrics,"snapshots":snapshots,
        "error_backstop":a<=1e-3 && mismatch==0,"nonfinite_mismatch":mismatch,
        "numerical_tuning":if a<=1e-3 && mismatch==0 && rel.is_finite() {"requires remaining timing/state/edge gates"} else {"BLOCKED; existing G fails backstop or has an infinite relative ceiling"},
        "performance_credit":false});
    let mut f=std::fs::OpenOptions::new().write(true).create_new(true).open(out).map_err(err)?;
    f.write_all(&serde_json::to_vec_pretty(&manifest).map_err(err)?).map_err(err)?;f.sync_all().map_err(err)?;
    println!("{}",manifest["ceilings"]);Ok(())
}

fn exp_throughput(gpu:&mut Gpu,inputs:&[f32])->Result<Value> {
    if inputs.is_empty() {return Err("empty real-score distribution".into());}
    const SOURCE:&str=r#"#include <hip/hip_runtime.h>
extern "C" __global__ void qsa_evidence_expf(const float* x, float* out, int n) {
    int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;
    float sum=0.f;
    #pragma unroll 1
    for(int j=0;j<64;++j) sum += expf(x[(i+j*257)%n]);
    out[i]=sum;
}"#;
    gpu.ensure_kernel_public("qsa_evidence_expf",SOURCE,"qsa_evidence_expf").map_err(err)?;
    let x=gpu.upload_f32(inputs,&[inputs.len()]).map_err(err)?;let out=gpu.zeros(&[inputs.len()],DType::F32).map_err(err)?;
    let timing=timed(gpu,|gpu| {let mut a=hip_bridge::KernargBlob::new();a.push_ptr(x.buf.as_ptr());a.push_ptr(out.buf.as_ptr());a.push_i32(inputs.len() as i32);a.pad_to(16);
        gpu.launch_kernel_blob("qsa_evidence_expf",[inputs.len().div_ceil(256) as u32,1,1],[256,1,1],0,a.as_mut_slice()).map_err(err)},None)?;
    let ms=timing["median_ms"].as_f64().ok_or("missing exp timing")?;
    let result=json!({"timing":timing,"exp_calls":inputs.len()*64,"Gexp_per_s":inputs.len() as f64*64.0/(ms*1e6),"distribution_sha256":sha256(&f32_bytes(inputs)),
        "includes":"global cached score loads, integer indexing, F32 additions and output sink; not a transcendental-only peak","source_sha256":sha256(SOURCE.as_bytes())});
    gpu.free_tensor(x).map_err(err)?;gpu.free_tensor(out).map_err(err)?;Ok(result)
}


fn rewrite_once(source:&mut String,from:&str,to:&str)->Result<()> {
    let at=source.find(from).ok_or_else(||format!("missing compensated source anchor {from}"))?;
    if source[at+from.len()..].contains(from) {return Err(format!("ambiguous compensated source anchor {from}"));}
    source.replace_range(at..at+from.len(),to);Ok(())
}

fn compensated_source(g:&Geometry,terms:usize)->Result<(String,String)> {
    let original=if g.fp8 {"indexed_attention_gathered_wmma_f16_gfx1201"} else {"indexed_attention_gathered_wmma_f16"};
    let input=if g.fp8 {GFX12} else {HALO};
    let end=input.find("#undef QG_CROW").ok_or("missing original gather guard end")?+"#undef QG_CROW".len();
    let mut src=input[..end].to_string();src.reserve(8192);
    let symbol=format!("qsa_evidence_compensated{terms}");
    rewrite_once(&mut src,&format!("void {original}("),&format!("void {symbol}("))?;
    rewrite_once(&mut src,"const _Float16* __restrict__ vb16, const int* __restrict__ selected_indices,",
        "const _Float16* __restrict__ vb16, const _Float16* __restrict__ k_low, const _Float16* __restrict__ v_low, const int* __restrict__ selected_indices,")?;
    let vector=if g.fp8 {"qg_h8_t"} else {"qg_h16_t"};
    let suffix=if g.fp8 {" + 8 * half"} else {""};
    let intrinsic=if g.fp8 {"__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12"} else {"__builtin_amdgcn_wmma_f32_16x16x16_f16_w32"};
    if terms==3 {
        let qdecl="__shared__ __align__(16) _Float16 q_lds[16 * 16 * 16];";
        rewrite_once(&mut src,qdecl,&format!("{qdecl}\n    __shared__ __align__(16) _Float16 q_low[16 * 16 * 16];"))?;
        let pdecl="__shared__ __align__(16) _Float16 p_lds[2][8][16 * 16];";
        rewrite_once(&mut src,pdecl,&format!("{pdecl}\n    __shared__ __align__(16) _Float16 p_low[2][8][16 * 16];"))?;
        let stores="dst[0] = (_Float16)v.x; dst[1] = (_Float16)v.y; dst[2] = (_Float16)v.z; dst[3] = (_Float16)v.w;";
        rewrite_once(&mut src,stores,&format!("{stores}\n        q_low[dst-q_lds]=(_Float16)(v.x-(float)dst[0]); q_low[dst-q_lds+1]=(_Float16)(v.y-(float)dst[1]); q_low[dst-q_lds+2]=(_Float16)(v.z-(float)dst[2]); q_low[dst-q_lds+3]=(_Float16)(v.w-(float)dst[3]);"))?;
        let pstore="pw[QG_CROW(i, half)] = (_Float16)p;";
        rewrite_once(&mut src,pstore,&format!("{pstore}\n                p_low[par][wave][l16*16+QG_CROW(i,half)]=(_Float16)(p-(float)pw[QG_CROW(i,half)]);"))?;
        rewrite_once(&mut src,"for (int i = 0; i < 8; ++i) pw[QG_CROW(i, half)] = (_Float16)0.0f;",
            "for (int i = 0; i < 8; ++i) { pw[QG_CROW(i, half)] = (_Float16)0.0f; p_low[par][wave][l16*16+QG_CROW(i,half)]=(_Float16)0.f; }")?;
    }
    let qk=format!("s = {intrinsic}(a, b, s);");
    let mut qk_new=format!("{qk}\n                const {vector} al=*(const {vector}*)(k_low+(kp-k16)+dc*16);\n                s={intrinsic}(al,b,s);");
    if terms==3 {qk_new.push_str(&format!("\n                const {vector} bl=*(const {vector}*)(q_low+(dc*16+l16)*16{suffix});\n                s={intrinsic}(a,bl,s);"));}
    rewrite_once(&mut src,&qk,&qk_new)?;
    rewrite_once(&mut src,&format!("{vector} a[2];"),&format!("{vector} a[2], al[2];"))?;
    rewrite_once(&mut src,"qg_h4_t v4;","qg_h4_t v4, vl4;")?;
    let vblock="v4 = *(const qg_h4_t*)(vb16 + ((size_t)(t0 >> 2) * width + c) * 4);";
    rewrite_once(&mut src,vblock,&format!("{vblock}\n                        vl4=*(const qg_h4_t*)(v_low+((size_t)(t0>>2)*width+c)*4);"))?;
    let scalar="v4[e] = tt[e] >= 0\n                                ? vb16[((size_t)(tt[e] >> 2) * width + c) * 4 + (tt[e] & 3)]\n                                : (_Float16)0.0f;";
    rewrite_once(&mut src,scalar,&format!("{scalar}\n                            vl4[e]=tt[e]>=0?v_low[((size_t)(tt[e]>>2)*width+c)*4+(tt[e]&3)]:(_Float16)0.f;"))?;
    rewrite_once(&mut src,"for (int e = 0; e < 4; ++e) a[dt][4 * g + e] = v4[e];",
        "for (int e = 0; e < 4; ++e) { a[dt][4*g+e]=v4[e]; al[dt][4*g+e]=vl4[e]; }")?;
    let pv=format!("o[0] = {intrinsic}(a[0], b, o[0]);\n            o[1] = {intrinsic}(a[1], b, o[1]);");
    let mut pv_new=format!("{pv}\n            o[0]={intrinsic}(al[0],b,o[0]); o[1]={intrinsic}(al[1],b,o[1]);");
    if terms==3 {pv_new.push_str(&format!("\n            const {vector} bl=*(const {vector}*)(p_low[par][kc]+l16*16{suffix});\n            o[0]={intrinsic}(a[0],bl,o[0]); o[1]={intrinsic}(a[1],bl,o[1]);"));}
    rewrite_once(&mut src,&pv,&pv_new)?;Ok((symbol,src))
}

fn compensated_snapshot(gpu:&mut Gpu,path:&Path,out:&Path)->Result<()> {
    let (header,g,q,k,v,s,r,eager)=snapshot_inputs(path)?;
    if g.fp8!=gpu.arch_caps.is_gfx1201() {return Err("compensated probe requires native architecture/source format".into());}
    std::fs::create_dir_all(out).map_err(err)?;
    let x=Inputs::upload(gpu,&g,&q,&k,&v,&s)?;
    indexed_attention_attention_batch_exact(gpu,&x.params(&g,true)).map_err(err)?;
    let replay=gpu.download_f32(&x.exact).map_err(err)?;
    if metrics(&r,&replay,g.heads,g.dim,None)?["differing_bits"]!=0 {return Err("exact source replay changed frozen R bits".into());}
    let hg_time=timed(gpu,|gpu|indexed_attention_attention_batch(gpu,&x.params(&g,false)).map_err(err),None)?;
    let hg=gpu.download_f32(&x.out).map_err(err)?;
    let base=Gather::new(gpu,&g,"full")?;base.convert(gpu,&g,&x)?;
    let base_time=timed(gpu,|gpu|base.launch(gpu,&g,&x),None)?;
    let base_values=gpu.download_f32(&x.out).map_err(err)?;
    let mut khi=vec![0u8;base.k.byte_size()];let mut vhi=vec![0u8;base.v.byte_size()];
    gpu.hip.memcpy_dtoh(&mut khi,&base.k.buf).map_err(err)?;gpu.hip.memcpy_dtoh(&mut vhi,&base.v.buf).map_err(err)?;
    let kd=decode_cache(&k,&g)?;let vd=decode_cache(&v,&g)?;
    let mut kb=vec![0u8;khi.len()];let mut vb=vec![0u8;vhi.len()];let width=g.kv_heads*g.dim;
    for token in 0..g.position_start+g.rows {for d in 0..width {
        let i=token*width+d;let j=(token/4*width+d)*4+token%4;
        let kh=half(u16::from_le_bytes([khi[2*i],khi[2*i+1]]));let vh=half(u16::from_le_bytes([vhi[2*j],vhi[2*j+1]]));
        kb[2*i..2*i+2].copy_from_slice(&half_rne(kd[i]-kh).to_le_bytes());
        vb[2*j..2*j+2].copy_from_slice(&half_rne(vd[i]-vh).to_le_bytes());
    }}
    let kl=gpu.zeros(&[base.k.numel()],DType::F16).map_err(err)?;let vl=gpu.zeros(&[base.v.numel()],DType::F16).map_err(err)?;
    gpu.hip.memcpy_htod(&kl.buf,&kb).map_err(err)?;gpu.hip.memcpy_htod(&vl.buf,&vb).map_err(err)?;
    let mut variants=serde_json::Map::new();
    let hg_stats=metrics(&r,&hg,g.heads,g.dim,None)?;
    let frozen_stats=metrics(&r,&eager,g.heads,g.dim,None)?;
    for terms in [2,3] {
        let (symbol,src)=compensated_source(&g,terms)?;
        let module=format!("qsa_compensated{}_{}",terms,gpu.arch);
        gpu.ensure_kernel_public(&module,&src,&symbol).map_err(err)?;
        let timing=timed(gpu,|gpu| {
            let mut args=hip_bridge::KernargBlob::new();
            for p in [&x.q,&base.k,&base.v,&kl,&vl,&x.selected,&x.out] {args.push_ptr(p.buf.as_ptr());}
            for n in [g.rows,g.position_start,g.heads,g.kv_heads,g.budget_blocks,g.compress,g.capacity,g.full_capacity] {args.push_i32(n as i32);}args.pad_to(16);
            let bound=(g.position_start+g.rows).min(g.capacity).min(g.budget_blocks*g.compress+g.compress-1);
            gpu.launch_kernel_blob(&symbol,[g.rows as u32,g.kv_heads as u32,1],[256,1,1],(bound.div_ceil(128)*128*4) as u32,args.as_mut_slice()).map_err(err)
        },None)?;
        let values=gpu.download_f32(&x.out).map_err(err)?;let stats=metrics(&r,&values,g.heads,g.dim,Some(&out.join(format!("compensated{terms}.elements"))))?;
        let (mut violations,mut zero_mismatch,mut excess)=(0usize,0usize,0.0f64);
        for ((&r,&g),&c) in r.iter().zip(&eager).zip(&values) {
            if !r.is_finite() || !g.is_finite() || !c.is_finite() {
                violations+=usize::from(r.to_bits()!=c.to_bits());
                continue;
            }
            let allowed=(g as f64-r as f64).abs().max(8.0*(f32::from_bits(r.abs().to_bits()+1) as f64-r.abs() as f64));
            let error=(c as f64-r as f64).abs();if error>allowed {violations+=1;excess=excess.max(error-allowed);}
            zero_mismatch+=usize::from((r==0.0)!=(c==0.0));
        }
        std::fs::write(out.join(format!("compensated{terms}.hip")),&src).map_err(err)?;
        std::fs::write(out.join(format!("compensated{terms}.f32")),f32_bytes(&values)).map_err(err)?;
        variants.insert(terms.to_string(),json!({"kernel_only":timing,"vs_R":stats,
            "max_abs_goal_1e-3":stats["max_abs"].as_f64().is_some_and(|x|x<=1e-3),
            "within_hg4_max_abs":stats["max_abs"].as_f64().zip(hg_stats["max_abs"].as_f64()).is_some_and(|(c,h)|c<=h),
            "frozen_element_envelope_violations":violations,"largest_excess":excess,"zero_semantic_mismatches":zero_mismatch,
            "frozen_error_envelope_admitted":violations==0 && zero_mismatch==0 && stats["nonfinite_mismatch"]==0
                && stats["max_A"].as_f64().is_some_and(|a|a<=1e-3)
                && stats["raw_max_rel"].as_f64().zip(frozen_stats["raw_max_rel"].as_f64()).is_some_and(|(c,g)|c<=g)
                && stats["max_ulp"].as_u64().zip(frozen_stats["max_ulp"].as_u64()).is_some_and(|(c,g)|c<=g),
            "extra_static_lds_bytes":if terms==3 {16384} else {0},"KV_scratch_bytes":khi.len()+vhi.len()+kb.len()+vb.len(),
            "wmma_terms_per_product":terms,"source_sha256":sha256(src.as_bytes()),
            "design":"hi=RNE16(x), lo=RNE16(x-f32(hi)); two terms correct K/V, third corrects Q/P; low*low omitted; f32 WMMA accumulation"}));
    }
    let report=json!({"identity":header["identity"],"geometry":g.json(),"variants":variants,"hg4":hg_time,
        "hg4_vs_R":hg_stats,"G_cached":base_time,"G_probe_vs_saved_eager":metrics(&eager,&base_values,g.heads,g.dim,None)?,
        "performance_credit":false,"scope":"developer probe/ISA sizing only; host low-part preparation/upload excluded; not an end-to-end performance measurement; no production dispatch change"});
    std::fs::write(out.join("compensated-manifest.json"),serde_json::to_vec_pretty(&report).map_err(err)?).map_err(err)?;
    gpu.free_tensor(kl).map_err(err)?;gpu.free_tensor(vl).map_err(err)?;base.free(gpu)?;x.free(gpu)?;
    println!("compensated probe {}",path.display());Ok(())
}

fn report_floors(root:&Path,out:&Path)->Result<()> {
    let mut chunks=Vec::new();let (mut max_rel,mut max_ulp,mut all_ulp,mut elements)=(0.0f64,0u64,0u64,0u64);
    for entry in std::fs::read_dir(root).map_err(err)? {
        let path=entry.map_err(err)?.path().join("snapshot.json");if !path.is_file() {continue;}
        let header:Value=serde_json::from_slice(&std::fs::read(&path).map_err(err)?).map_err(err)?;
        if header["identity"]["phase"]!="all_prefill" || header["identity"]["ctx"]!=8192 {continue;}
        let g=Geometry::parse(&header["geometry"])?;
        let get=|name:&str|->Result<Vec<f32>> {
            let bytes=std::fs::read(path.parent().unwrap().join(name)).map_err(err)?;
            if header["files"][name]["sha256"]!=sha256(&bytes) {return Err(format!("metric source hash mismatch {name}"));}
            f32_values(&bytes)
        };
        let m=metrics(&get("exact.f32")?,&get("eager.f32")?,g.heads,g.dim,None)?;let f=&m["floored_reporting"];
        max_rel=max_rel.max(f["max_rel"].as_f64().ok_or("nonfinite floored relative distance")?);
        max_ulp=max_ulp.max(f["max_ulp"].as_u64().ok_or("missing floored ULP")?);
        all_ulp=all_ulp.max(m["max_ulp"].as_u64().ok_or("missing all-element ULP")?);
        elements+=f["included_elements"].as_u64().ok_or("missing included count")?;
        chunks.push(json!({"identity":header["identity"],"snapshot":path,"floored_reporting":f,"max_ulp_all":m["max_ulp"]}));
    }
    if chunks.len()!=72 {return Err(format!("incomplete pp8192 floor report: {} chunks",chunks.len()));}
    let report=json!({"abs_reference_floor":1e-5,"max_rel":max_rel,"max_ulp":max_ulp,"max_ulp_all":all_ulp,
        "included_elements":elements,"chunks":chunks,"gate":"display subset only; original all-element error envelope and backstop unchanged"});
    std::fs::write(out,serde_json::to_vec_pretty(&report).map_err(err)?).map_err(err)?;
    println!("{}",json!({"abs_reference_floor":1e-5,"max_rel":max_rel,"max_ulp":max_ulp,"max_ulp_all":all_ulp,"included_elements":elements}));Ok(())
}

fn main()->Result<()> {
    let args:Vec<String>=std::env::args().collect();
    if args.get(1).is_some_and(|s| s=="self-check") {return self_check();}
    if args.get(1).is_some_and(|s| s=="analyze" || s=="freeze" || s=="isolate" || s=="report-floors") {
        let mode=&args[1];let input=Path::new(args.get(2).ok_or("input required")?);let out=Path::new(args.get(3).ok_or("output required")?);
        return match mode.as_str() {"analyze"=>analyze_snapshot(input,out),"isolate"=>isolate_snapshot(input,out),"report-floors"=>report_floors(input,out),_=>freeze(input,out)};
    }
    if std::env::var_os("HIPFIRE_LOCK_DIR").is_none() {return Err("private HIPFIRE_LOCK_DIR required".into());}
    let mode=args.get(1).ok_or("usage: fixtures OUT | snapshot SNAPSHOT.json OUT [oracle] | self-check")?;
    // One process is the OFF baseline. The existing gathered source is launched
    // directly above, never selected by a production environment override.
    std::env::set_var("HIPFIRE_QWEN4_QSA_WMMA_GATHER","0");
    std::env::set_var("HIPFIRE_QWEN4_F16_WMMA","0");
    let mut gpu=Gpu::init().map_err(err)?;
    gpu.dpm_warmup(10.0).map_err(err)?;
    if mode=="compensated" {
        return compensated_snapshot(&mut gpu,Path::new(args.get(2).ok_or("snapshot required")?),Path::new(args.get(3).ok_or("output required")?));
    }
    if mode=="exp" {
        let input=Path::new(args.get(2).ok_or("exp-inputs.f32 required")?);let out=Path::new(args.get(3).ok_or("output JSON required")?);
        let result=exp_throughput(&mut gpu,&f32_values(&std::fs::read(input).map_err(err)?)?)?;
        std::fs::write(out,serde_json::to_vec_pretty(&result).map_err(err)?).map_err(err)?;println!("{result}");return Ok(());
    }
    if mode=="snapshot" {
        let path=Path::new(args.get(2).ok_or("snapshot path required")?);let root=path.parent().unwrap();let out=Path::new(args.get(3).ok_or("output required")?);
        let header:Value=serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)?;let g=Geometry::parse(&header["geometry"])?;
        let get=|name:&str|->Result<Vec<u8>> {let b=std::fs::read(root.join(name)).map_err(err)?;if header["files"][name]["sha256"]!=sha256(&b) {return Err(format!("snapshot hash mismatch {name}"));}Ok(b)};
        let q=f32_values(&get("qgate.f32")?)?;let k=get("keys.source")?;let v=get("values.source")?;let s=i32_values(&get("selected.i32")?)?;
        let result=exercise(&mut gpu,&g,&q,&k,&v,&s,out,args.get(4).is_some_and(|s|s=="oracle"))?;println!("{}",json!({"manifest":out.join("manifest.json"),"max_abs":result["G_vs_R"]["max_abs"],"raw_max_rel":result["G_vs_R"]["raw_max_rel"],"max_ulp":result["G_vs_R"]["max_ulp"]}));
    } else if mode=="shape" {
        let rows:usize=args.get(2).ok_or("rows required")?.parse().map_err(err)?;
        if rows==0 || rows>2048 {return Err("shape rows must be 1..2048".into());}
        let out=Path::new(args.get(3).ok_or("output required")?);
        let g=Geometry{rows,position_start:3072,heads:24,kv_heads:2,dim:256,budget_blocks:512,compress:4,capacity:2051,full_capacity:3072+rows,fp8:gpu.arch_caps.is_gfx1201()};
        let (q,k,v,s)=fixture(&g,"normal");
        let result=exercise(&mut gpu,&g,&q,&k,&v,&s,out,false)?;
        println!("{}",json!({"manifest":out.join("manifest.json"),"hg4_ms":result["hg4"]["median_ms"],"gather_conversion_ms":result["gather_conversion"]["median_ms"],"G_vs_R":result["G_vs_R"]["max_abs"]}));
    } else if mode=="fixtures" {
        let out=Path::new(args.get(2).ok_or("output required")?);let mut manifests=Vec::new();let fp8=gpu.arch_caps.is_gfx1201();
        let empty_geometry=Geometry{rows:1,position_start:0,heads:24,kv_heads:2,dim:256,budget_blocks:512,compress:4,capacity:2051,full_capacity:1,fp8};
        let (q,k,v,s)=fixture(&empty_geometry,"normal");let empty=Inputs::upload(&mut gpu,&empty_geometry,&q,&k,&v,&s)?;
        let mut p=empty.params(&empty_geometry,true);p.rows=0;
        let rejected_exact=indexed_attention_attention_batch_exact(&mut gpu,&p).err().ok_or("zero rows unexpectedly admitted by exact host entry")?;
        let rejected_fast=indexed_attention_attention_batch(&mut gpu,&p).err().ok_or("zero rows unexpectedly admitted by fast host entry")?;
        empty.free(&mut gpu)?;
        let visible_zero=json!({"zero_rows_exact":format!("{rejected_exact:?}"),"zero_rows_fast":format!("{rejected_fast:?}"),"zero_visible_query":"not representable; all-invalid fixtures exercise empty selection"});
        let geometries=[1,8,511,512,1535,1536,1537,2048];
        for rows in geometries {let g=Geometry{rows,position_start:3072,heads:24,kv_heads:2,dim:256,budget_blocks:512,compress:4,capacity:2051,full_capacity:3072+rows,fp8};let (q,k,v,s)=fixture(&g,"normal");let name=format!("rows-{rows}");let result=exercise(&mut gpu,&g,&q,&k,&v,&s,&out.join(&name),false)?;manifests.push(json!({"fixture":name,"G_vs_R":result["G_vs_R"],"geometry":g.json()}));}
        for visible in [1,3,4,2047,2048,2049,2050,2051,2052] {for kind in ["normal","invalid","oow","duplicate","arbitrary","sharp","flat-cancel","gate","scales","fp8-boundaries"] {
            let g=Geometry{rows:1,position_start:visible-1,heads:24,kv_heads:2,dim:256,budget_blocks:512,compress:4,capacity:if kind=="arbitrary" {3} else {2051},full_capacity:visible,fp8};let (q,k,v,s)=fixture(&g,kind);let name=format!("visible-{visible}-{kind}");let result=exercise(&mut gpu,&g,&q,&k,&v,&s,&out.join(&name),true)?;manifests.push(json!({"fixture":name,"G_vs_R":result["G_vs_R"],"geometry":g.json()}));
        }}
        std::fs::write(out.join("fixtures.json"),serde_json::to_vec_pretty(&json!({"arch":gpu.arch,"fixtures":manifests,"visible_zero":visible_zero,"numerical_tuning":"BLOCKED until full real pp8192 G ceilings and backstop are frozen"})).map_err(err)?).map_err(err)?;
    } else {return Err(format!("unknown mode {mode}"));}
    Ok(())
}
