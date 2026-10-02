use crate::{plan::KernelSpec,reg::RegPlan,insn::Program};

// Imported regions can mention scalar operands outside the authored RegPlan.
// Scan the final operands (not labels/opcodes); explicit pairs/ranges include
// their high register. Lane masks are single SGPRs in this wave32 builder.
fn scalar_high_water(text: &str) -> u16 {
 let Some((_, operands)) = text.split_once(' ') else { return 0 };
 let bytes=operands.as_bytes(); let mut i=0; let mut high=0;
 while i<bytes.len() {
  if bytes[i]!=b's'||i>0&&(bytes[i-1].is_ascii_alphanumeric()||bytes[i-1]==b'_') {i+=1;continue}
  i+=1;
  let bracket=bytes.get(i)==Some(&b'[');if bracket{i+=1}
  let begin=i;while bytes.get(i).is_some_and(u8::is_ascii_digit){i+=1}
  let Some(mut last)=operands[begin..i].parse::<u16>().ok() else {continue};
  if bracket&&bytes.get(i)==Some(&b':') {
   i+=1;let begin=i;while bytes.get(i).is_some_and(u8::is_ascii_digit){i+=1}
   if let Ok(end)=operands[begin..i].parse::<u16>(){last=end}
  }
  if last<128 {high=high.max(last+1)}
 }
 high
}

pub fn assembly(spec:&KernelSpec,regs:&RegPlan,program:&Program)->Result<String,String> {
 spec.kernargs.validate()?;if spec.arch!=program.arch {return Err("program architecture differs from kernel spec".into())}if !spec.wave32 {return Err("builder requires wave32".into())}
 let arch=spec.arch;let symbol=&spec.symbol;let nv=regs.next_free_vgpr();let ns=program.instructions.iter().map(|i|scalar_high_water(&i.text)).max().unwrap_or(0).max(regs.next_free_sgpr());
 // Kernels using VCC reserve it and include its two SGPRs in metadata,
 // matching hipcc accounting on gfx11 and gfx12.
 let vcc=program.instructions.iter().any(|i|i.text.contains("vcc"));let meta_sgprs=ns+if vcc{2}else{0};
 let mut s=format!(".amdgcn_target \"amdgcn-amd-amdhsa--{}\"\n.amdhsa_code_object_version 6\n.text\n.protected {symbol}\n.globl {symbol}\n.p2align 8\n.type {symbol},@function\n{symbol}:\n",arch.name());
 for i in &program.instructions {s.push_str("\t");s.push_str(&i.text);s.push('\n')}
 s.push_str(&format!(".L{symbol}_end:\n.size {symbol}, .L{symbol}_end-{symbol}\n.section .rodata,\"a\",@progbits\n.p2align 6\n.amdhsa_kernel {symbol}\n"));
 for (field,val) in [("group_segment_fixed_size",spec.group_segment_fixed_size),("private_segment_fixed_size",0),("kernarg_size",spec.kernargs.size),("user_sgpr_count",spec.user_sgpr_count as u32),("user_sgpr_dispatch_ptr",0),("user_sgpr_queue_ptr",0),("user_sgpr_kernarg_segment_ptr",1),("user_sgpr_dispatch_id",0),("user_sgpr_private_segment_size",0),("wavefront_size32",1),("uses_dynamic_stack",0),("enable_private_segment",0),("system_sgpr_workgroup_id_x",1),("system_sgpr_workgroup_id_y",spec.system_sgpr_workgroup_id_y as u32),("system_sgpr_workgroup_id_z",0),("system_sgpr_workgroup_info",0),("system_vgpr_workitem_id",0),("next_free_vgpr",nv as u32),("next_free_sgpr",ns as u32),("reserve_vcc",u32::from(vcc)),("float_round_mode_32",0),("float_round_mode_16_64",0),("float_denorm_mode_32",3),("float_denorm_mode_16_64",3),("fp16_overflow",0),("workgroup_processor_mode",u32::from(!spec.cu_mode)),("memory_ordered",1),("forward_progress",1)] {s.push_str(&format!("\t.amdhsa_{field} {val}\n"))}
 let mask=if arch.gfx12(){4080}else{1008};s.push_str(&format!("\t.amdhsa_inst_pref_size ((instprefsize(.L{symbol}_end-{symbol})<<4)&{mask})>>4\n"));
 if arch.gfx12(){s.push_str("\t.amdhsa_round_robin_scheduling 0\n")}else{s.push_str("\t.amdhsa_dx10_clamp 1\n\t.amdhsa_ieee_mode 1\n\t.amdhsa_shared_vgpr_count 0\n")}
 for field in ["fp_ieee_invalid_op","fp_denorm_src","fp_ieee_div_zero","fp_ieee_overflow","fp_ieee_underflow","fp_ieee_inexact","int_div_zero"] {s.push_str(&format!("\t.amdhsa_exception_{field} 0\n"))}
 s.push_str(".end_amdhsa_kernel\n.text\n.amdgpu_metadata\n---\namdhsa.kernels:\n  - .args:\n");
 for a in &spec.kernargs.args {if let Some(space)=&a.address_space {s.push_str(&format!("      - .address_space: {space}\n"))} else {s.push_str("      - ")}
 let indent=if a.address_space.is_some(){"        "}else{"        "};if a.address_space.is_none(){s.push('\n')}
 s.push_str(&format!("{indent}.name: {}\n{indent}.offset: {}\n{indent}.size: {}\n{indent}.value_kind: {}\n",a.name,a.offset,a.size,a.value_kind));}
 s.push_str(&format!("    .group_segment_fixed_size: {}\n    .kernarg_segment_align: 8\n    .kernarg_segment_size: {}\n    .max_flat_workgroup_size: {}\n    .name: {symbol}\n    .private_segment_fixed_size: 0\n    .sgpr_count: {meta_sgprs}\n    .sgpr_spill_count: 0\n    .symbol: {symbol}.kd\n    .uniform_work_group_size: 1\n    .uses_dynamic_stack: false\n    .vgpr_count: {nv}\n    .vgpr_spill_count: 0\n    .wavefront_size: 32\n    .workgroup_processor_mode: {}\namdhsa.target: amdgcn-amd-amdhsa--{}\namdhsa.version: [1, 2]\n...\n.end_amdgpu_metadata\n",spec.group_segment_fixed_size,spec.kernargs.size,spec.workgroup_size,u8::from(!spec.cu_mode),arch.name()));Ok(s)
}

#[cfg(test)]
mod tests {
 use super::scalar_high_water;
 #[test]
 fn scalar_operand_widths_and_region_registers() {
  assert_eq!(scalar_high_water("v_cmp_gt_f32_e64 s88, v0, v1"),89);
  assert_eq!(scalar_high_water("s_mov_b64 s[86:89], s[0:3]"),90);
  assert_eq!(scalar_high_water("buffer_load_b32 v0, v1, s[84:87], s88"),89);
  assert_eq!(scalar_high_water("s_wait_alu depctr_sa_sdst(0)"),0);
  assert_eq!(scalar_high_water("s_branch .Ls99"),0);
  assert_eq!(scalar_high_water(".Ls99:"),0);
 }
}
