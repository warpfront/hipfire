// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use peacemaker_isa::{
    insn::{Instruction, MemoryClass},
    ledger::Counter,
    reg::Live,
    vopd::{Operand, VopdF32, VopdOp},
};
use peacemaker_isa::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
use sha2::{Digest, Sha256};
use std::{env, fs};

fn probe(arch: Arch) -> Result<Emitted, String> {
    let mut regs = RegPlan::new(8, 8)?;
    let mut v = Vec::new();
    for (name, n) in [("lane", 0), ("c", 1), ("w", 2), ("scale", 3), ("result", 4)] {
        v.push(regs.v::<1>(name, n, Live::Whole)?)
    }
    let mut s = Vec::new();
    for (name, n) in [("arg0", 0), ("arg1", 2), ("arg2", 4), ("arg3", 6)] {
        s.push(regs.s::<2>(name, n, Live::Whole)?)
    }
    let spec = KernelSpec {
        kernel_id: "fold_magic".into(),
        variant: "probe".into(),
        arch,
        symbol: "fold_magic".into(),
        kernargs: KernargLayout::new(32)
            .pointer("out", 0)
            .pointer("weight", 8)
            .pointer("scale", 16)
            .pointer("c", 24),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 1024,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, regs);
    b.push(
        Instruction::new(
            "s_load_b256 s[0:7], s[0:1], 0x0",
            s.iter().map(|r| r.reg()).collect(),
            vec![s[0].reg()],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v0, 2, v0",
        vec![v[0].reg()],
        vec![v[0].reg()],
    ))?;
    b.wait(Counter::Km, 0)?;
    b.clause(|b| {
        for (dst, ptr) in [(1, 2), (2, 0), (3, 1), (4, 3)] {
            b.push(
                Instruction::new(
                    format!("global_load_b32 v{dst}, v0, s[{}:{}]", ptr * 2, ptr * 2 + 1),
                    vec![v[dst].reg()],
                    vec![v[0].reg(), s[ptr].reg()],
                )
                .memory(MemoryClass::VmemLoad),
            )?
        }
        Ok(())
    })?;
    b.vopd(
        VopdOp {
            op: VopdF32::Add,
            dst: 1,
            src0: Operand::Lit(0xcb400000),
            src1: 1,
        },
        VopdOp {
            op: VopdF32::Mul,
            dst: 2,
            src0: Operand::V(2),
            src1: 3,
        },
    )?;
    if !arch.gfx12() {
        return Err("probe fold_magic uses a gfx12-only shared VMEM schedule; gfx11 is table-checked separately".into());
    }
    b.wait(Counter::Load, 0)?;
    b.push(Instruction::new(
        "s_delay_alu instid0(VALU_DEP_1)",
        vec![],
        vec![],
    ))?;
    b.push(Instruction::new(
        "v_fmac_f32_e32 v4, v2, v1",
        vec![v[4].reg()],
        vec![v[4].reg(), v[2].reg(), v[1].reg()],
    ))?;
    b.push(
        Instruction::new(
            "global_store_b32 v0, v4, s[6:7]",
            vec![],
            vec![v[0].reg(), v[4].reg(), s[3].reg()],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    b.finish()
}
const USAGE: &str = "usage: peacemaker-isa emit --kernel fold_magic --arch gfx1201 --out FILE --proof FILE
       peacemaker-isa emit --kernel iu4_v2c [--epi set|add|silu|all] --arch gfx1100 --out FILE --proof FILE
       peacemaker-isa emit --kernel iu4_v2b --epi set|add|silu|all --arch gfx1151 --out FILE --proof FILE
       peacemaker-isa emit --kernel iu4_v2b_a4 --epi m512|a4|all --arch gfx1151 --out FILE --proof FILE
       peacemaker-isa emit --kernel iu4_gemm --fold k128 --tile 128x128x8|256x128x16 --cacc 1 --epi set|add|silu|silu-bf16|qkvzagdn|all [--alayout token|slab] [--mma dense|s24x64|s24x32] --arch gfx1201 --out FILE --proof FILE
       peacemaker-isa emit --kernel fp8_gemm --scale row|k128|both --epi set|add|silu|qkv|qkvza|all --arch gfx1201 --out FILE --proof FILE
       peacemaker-isa emit --kernel gdn_scan --arch gfx1201 --out FILE --proof FILE
       peacemaker-isa emit --kernel qsa_select --epi score|score32|score-bf16|score-bf16-rows32|select|all|bf16 --arch gfx1151 --out FILE --proof FILE
       peacemaker-isa emit --kernel spec_sampling_rows_pm|spec_accept_pm --arch gfx1151|gfx1201 --out FILE --proof FILE
       peacemaker-isa emit --kernel qwen4_ep_owner_assign_pm --arch gfx1201 --out FILE --proof FILE
       peacemaker-isa emit --kernel qwen4_mq6_x4 [--epi w4|w8|all] --arch gfx1201 --out FILE --proof FILE
       peacemaker-isa emit --kernel qwen4_mq6_x4_gfx11 [--epi w4|w8|w4_bf16out|w8_bf16out|w8_regions|w8_hcw|all] --arch gfx1151 --out FILE --proof FILE
       peacemaker-isa emit --kernel qwen4_mq4_x4_gfx11 [--epi w4|w8|w4_bf16out|w8_bf16out|w8_regions|w8_hcw|all] --arch gfx1151 --out FILE --proof FILE
       peacemaker-isa emit --kernel mq6_i8_gemm [--epi set|set_bf16|all] [--tile 128x128|256x128|128x256] --arch gfx1201 --out FILE --proof FILE
       peacemaker-isa region-import --disassembly OBJDUMP.txt [--symbol gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3]";
const NATIVE_USAGE: &str = "       emit options: [--co FILE] [--bundle FILE [--host-target TRIPLE]]
         (native assembly/link/bundle needs no ROCm tools; --bundle also writes FILE.sha256, bare lowercase SHA256)
       peacemaker-isa emit --kernel pm_decode_<module> --arch gfx1201 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape FILE.shape.json] [--m7 FILE.m7.json]
         (<module>: fused_gate_up_hfq4g256_mq4v2|gemv_hfq4g256_residual_mq4v2|fused_qkvza_hfq4g256_mq4v2|gemv_hfq4g256_multirow_default_mq4v2|fused_qkv_hfq4g256_mq4v2|gemv_hfq4g256_residual_xbatch_mq4v2|gemv_hfq4g256_xbatch_mq4v2)
       peacemaker-isa emit --kernel mq4_verify [--epi <qkvza|qkv|gate_up|residual>_bt<4|8>w<4|8>|all] --arch gfx1201 --out FILE.s --proof FILE.proof.json [--co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]]
       peacemaker-isa emit --kernel mq4_fp8_verify [--epi <qkvza|qkv|gate_up|residual|set>_bt<1|2|4|8>w<1|4|8>|all] --arch gfx1201 --out FILE.s --proof FILE.proof.json [--co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]]
         (with --co, M7 runs on every selected export; --shape writes DIR/<kernel>.<arch>.<entry>.contract.json)
       peacemaker-isa emit --kernel qwen4_lane_rows --arch gfx1151 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--m7 FILE.m7.json]
         (qwen4_lane_rows_pm: four unchanged narrow exports plus seven t8 exports, exactly SYMBOLS; no --shape)
       peacemaker-isa emit --kernel qwen4_lane_rows [--epi all] --arch gfx1100 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]
         (qwen4_lane_rows_pm_gfx1100: two actual gfx1100 MQ6 tiles then five actual gfx1100 Q8 tiles;
          source builders must provide every frozen export; unsupported singleton recipes refuse emission, never retag Halo;
          Q8 k2560 single accepts K>1536 divisible32, x2 only K2560; k320 and small-wide use the actual four-chain singleton)
       peacemaker-isa emit --kernel lane_rows_mq6_gfx1201|lane_rows_bf16_gfx1201 [--epi all] --arch gfx1201 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]
         (lane_rows_mq6_pm_gfx1201 / lane_rows_bf16_pm_gfx1201: ten unchanged narrow exports then two t8 exports)
       peacemaker-isa emit --kernel mq4_rows_x4 [--epi entry|t8|all] --arch gfx1151|gfx1201|gfx1100 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]
         (mq4_rows_x4_pm_<arch>: unchanged narrow plus t8 on gfx1151/gfx1201; gfx1100 has only the actual scalar-DAG t8 export, no narrow entry; no r16/r32 exports)
       peacemaker-isa emit --kernel qwen4_bf16_rows_wide [--epi t8|k4_t8|all] --arch gfx1151|gfx1100 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]
         (bf16_rows_wide_pm_<arch>: ordinary x4 ABI96 and HC k4 logical44/segment48 t8 exports only;
          gfx1100 uses its independent actual-object builder, never the Halo body with a different target)
       peacemaker-isa emit --kernel mq_i4_moe_gemv [--epi <codec>_<kind>[_t8]|all] --arch gfx1201 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]
         (mq_i4_moe_gemv_pm_gfx1201: four narrow plus four t8 exports, no c16/c32)
         (--arch gfx1151 retains the old common diagnostic image; product Halo rowtiles use mq_i4_moe_gemv_cn, not that image)
       peacemaker-isa emit --kernel mq_i4_moe_gemv_mq4only [--epi mq4sym_<kind>[_t8]|all] --arch gfx1100 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]
         (artifact stem mq_i4_moe_gemv_mq4only_pm_gfx1100, internal module mq_i4_moe_gemv_pm_gfx1100:
          two genuine MQ4 narrow exports then their two t8 exports; the retired mixed W2 image is not emitted)
       peacemaker-isa emit --kernel mq_i4_moe_gemv_cn [--epi <codec>_<kind>[_t8]|all] --arch gfx1151 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]
         (mq_i4_moe_gemv_cn_pm_gfx1151: four unchanged narrow plus four t8 exports)
       peacemaker-isa emit --kernel e8p_i4_moe_gemv [--epi r256_gate_up[_t8]|r128_down[_t8]|all] --arch gfx1151|gfx1201|gfx1100 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]
         (e8p_i4_moe_gemv_pm_<arch>: two unchanged narrow plus two t8 exports; emission is not codec admission)
       peacemaker-isa emit --kernel qwen4_slab_union_plan [--epi t8|all] --arch gfx1151|gfx1201|gfx1100 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]
         (qwen4_slab_union_plan_t8_pm_<arch>: one export, logical36B/segment40B, grid ceil(rows/8), block128, LDS1280)
       peacemaker-isa emit --kernel legacy_mq4_rowtiles --epi all --arch ARCH --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]
         (legacy_mq4_rowtiles_pm_<arch>: original aggregate indexed gate/up and unweighted down ABIs;
          two physical-eight-token exports, original weight buffers without repacking; unsupported AR recipes refuse emission)
         (row images require --features toolchain and --co; M7 runs on EVERY linked export even without --m7;
          --shape writes DIR/<kernel>.<arch>.<proof-variant>.contract.json with exact emitted ABI/resources and tile geometry;
          t8 means physical eight-row tiles, total logical rows 1..64, never old r8 geometry or an admission claim)
       peacemaker-isa emit --kernel mq_i4_rows [--epi w3_r256|w3_r128|w2_r256|w2_r128|all] --arch gfx1201 --out FILE.s --proof FILE.proof.json [--co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]]
       peacemaker-isa emit --kernel mtp_dense_det_gfx11 [--epi w8|all] --arch gfx1151 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--m7 FILE.m7.json]
         (experimental DET dense F16 WMMA GEMM; no --shape; M7 requires --features toolchain)";
const FA_PREFILL_USAGE:&str="       peacemaker-isa emit --kernel fa_prefill_pm --arch gfx1201 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]\n         (H16/KV2/D256 Q8_0 FA2-GQA twin: one module, exports attention_fa2_q_preconvert_pm_gfx1201 then attention_q8_0_fa2_gqa_pm_gfx1201; --shape/--m7 require --co and a --features toolchain build; --shape writes DIR/fa_prefill_pm.gfx1201.{preconvert,direct}.contract.json, --m7 a symbol-keyed receipt map)";
const FA_PREFILL_FP8_USAGE:&str="       peacemaker-isa emit --kernel fa_prefill_fp8_pm --arch gfx1201 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]\n         (H16/KV2/D256 native-fp8 E4M3 FA2-GQA twin: one module, ONE export attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201 (the Q preconvert/qprep stay in the Q8 and F5 modules); --shape/--m7 require --co and a --features toolchain build; --shape writes DIR/fa_prefill_fp8_pm.gfx1201.direct.contract.json, --m7 a symbol-keyed receipt map)";
const FA_QPREP_USAGE:&str="       peacemaker-isa emit --kernel fa_qprep_pm --arch gfx1201 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]\n         (H16/KV2/D256 rotary-64 Q prep: one module, one export attention_fa2_qprep_pm_gfx1201; --shape/--m7 require --co and a --features toolchain build; --shape writes DIR/fa_qprep_pm.gfx1201.contract.json, --m7 a symbol-keyed receipt map)";
const IU4_C8_USAGE:&str="       peacemaker-isa emit --kernel iu4_c8_producer_pm --arch gfx1201 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]\n         (exact C8 IU4 producers of the A3B F3 route: one module, four exports iu4_c8_rotate_pm_gfx1201, iu4_c8_rmsnorm_rotate_pm_gfx1201, iu4_c8_sigmoid_rotate_pm_gfx1201 and iu4_c8_rmsnorm_out_rotate_pm_gfx1201; --shape/--m7 require --co and a --features toolchain build; --shape writes DIR/iu4_c8_producer_pm.gfx1201.{rotate,norm,sigmoid_rotate,norm_out}.contract.json, --m7 a symbol-keyed receipt map)";
const GDN_CHUNK_PREP_USAGE:&str="       peacemaker-isa emit --kernel gdn_chunk_prep_pm --arch gfx1201 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]\n         (HK16/HV32/HD128 GDN chunk prep: one module, one export gdn_chunk_prep_pm_gfx1201; --shape/--m7 require --co and a --features toolchain build; --shape writes DIR/gdn_chunk_prep_pm.gfx1201.contract.json, --m7 a symbol-keyed receipt map)";
/// `--epi all` emits the three epilogue symbols as one module (the product
/// code object the oracle loads); a single epilogue emits one symbol.
fn iu4_gemm(
    fold: &str,
    tile: &str,
    cacc: &str,
    epi: &str,
    act: &str,
    mma: &str,
    arch: Arch,
) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::iu4_gemm::{self, Spec};
    let mma: iu4_gemm::Mma = mma.parse()?;
    if epi == "all" {
        if mma != iu4_gemm::Mma::Dense {
            return Err("--epi all emits the dense module: --mma needs a single --epi".into());
        }
        let (fold, tile, cacc, act) = (fold.parse()?, tile.parse()?, cacc.parse()?, act.parse()?);
        let (_, text, proof) = iu4_gemm::emit_module(fold, tile, cacc, act, arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let mut spec = Spec::from_knobs(fold, tile, cacc, epi, act, arch)?;
    spec.mma = mma;
    let emitted = iu4_gemm::emit(spec)?;
    Ok((
        emitted.s_text,
        serde_json::to_vec_pretty(&emitted.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1100 V2C-equivalent GEMM: one epilogue symbol, or `--epi all` for the
/// product module (every epilogue symbol in one code object).
fn iu4_v2c(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::iu4_v2c::{self, Epi, Spec};
    if epi == "all" {
        let (_, text, proof) = iu4_v2c::module(arch, &Epi::ALL)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let e = iu4_v2c::emit(Spec {
        arch,
        epi: epi.parse()?,
    })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// `--epi all` emits the SET, ADD and SiLU entries as one module.
fn iu4_v2b(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::iu4_v2b::{self, Spec};
    if epi == "all" {
        let (_, text, proof) = iu4_v2b::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let e = iu4_v2b::emit(Spec {
        arch,
        epi: epi.parse()?,
    })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// `--epi all` emits the M512 gate/up entries (`m512` h twin, `a4` fused A4
/// epilogue) as one module: the separate A4-fusion code object.
fn iu4_v2b_a4(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::iu4_v2b_a4::{self, Spec};
    if epi == "all" {
        let (_, text, proof) = iu4_v2b_a4::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let e = iu4_v2b_a4::emit(Spec {
        arch,
        epi: epi.parse()?,
    })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// `--epi all` emits every entry of the arch as one module; `gate_up`/`down`
/// one 16-slot entry, `gate_up_ntN`/`down_ntN` one expert-run entry,
/// `down_nt4rR` one expert-run down entry with R contiguous 64-row blocks per CTA.
fn qwen4_moe_sym(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qwen4_moe_sym::{self, Spec};
    if epi == "gate_up_stream8" {
        let e = qwen4_moe_sym::emit_gate_up_stream8(arch)?;
        return Ok((
            e.s_text,
            serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
        ));
    }
    if epi == "all" {
        let (_, text, proof) = qwen4_moe_sym::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let e = qwen4_moe_sym::emit(Spec::from_epi(epi, arch)?)?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// MQ3-I4 (`WeightCodec::Mq3I4`) grouped IU4 MoE entries: `--epi all` (default) is the module, `gate_up`/`down` one NT1 entry.
fn qwen4_moe_sym_w3(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qwen4_moe_sym::{self, Kind, Spec, WeightCodec};
    if epi == "all" {
        let (_, text, proof) = qwen4_moe_sym::emit_module_codec(arch, WeightCodec::Mq3I4)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let kind = match epi {
        "gate_up" => Kind::GateUp,
        "down" => Kind::Down,
        _ => return Err(format!("qwen4_moe_sym_w3 --epi {epi} (gate_up|down|all)")),
    };
    let e = qwen4_moe_sym::emit(Spec {
        arch,
        kind,
        nt: 1,
        rr: 1,
        codec: WeightCodec::Mq3I4,
    })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// MQ3-I4 NT4 expert-run entries (`qwen4_moe_sym::w3::nt4`, module `qwen4_moe_iu4_sym_w3_nt4_pm_<arch>`, gfx1151 only): `--epi all` (default)
/// is the six-entry module (gate/up, down, down nt4r2, down nt4r4, then the F32 `_acc` twins of gate/up and down), `gate_up`/`down` one shipped `rr == 1` entry.
fn qwen4_moe_sym_w3_nt4(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qwen4_moe_sym::w3::nt4;
    if epi == "all" {
        let (_, text, proof) = nt4::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let want = match epi {
        "gate_up" | "down" => epi,
        _ => {
            return Err(format!(
                "qwen4_moe_sym_w3_nt4 --epi {epi} (gate_up|down|all)"
            ))
        }
    };
    let spec = nt4::module_specs(arch)
        .into_iter()
        .find(|s| s.kind.tag() == want)
        .ok_or_else(|| {
            format!(
                "qwen4_moe_sym_w3_nt4 has no {want} entry on {}",
                arch.name()
            )
        })?;
    let e = nt4::emit(spec)?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// E8P-I4 (`WeightCodec::E8pI4`) grouped IU4 MoE entries, gfx1201 NT1 only: `--epi all` (default) is the
/// `qwen4_moe_iu4_sym_e8p_pm_gfx1201` module (both entries plus their `_acc` qualification entries);
/// `gate_up`/`down` one shipped-shape entry, `gate_up_acc`/`down_acc` one F32 qualification entry.
fn qwen4_moe_sym_e8p(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qwen4_moe_sym::{self, Kind, Spec, WeightCodec};
    if epi == "all" {
        let (_, text, proof) = qwen4_moe_sym::emit_module_codec(arch, WeightCodec::E8pI4)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let (name, acc) = match epi.strip_suffix("_acc") {
        Some(n) => (n, true),
        None => (epi, false),
    };
    let kind = match name {
        "gate_up" => Kind::GateUp,
        "down" => Kind::Down,
        _ => {
            return Err(format!(
                "qwen4_moe_sym_e8p --epi {epi} (gate_up|down|gate_up_acc|down_acc|all)"
            ))
        }
    };
    let spec = Spec {
        arch,
        kind,
        nt: 1,
        rr: 1,
        codec: WeightCodec::E8pI4,
    };
    let e = if acc {
        qwen4_moe_sym::e8p::emit_acc(spec)?
    } else {
        qwen4_moe_sym::emit(spec)?
    };
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx11 NT1 grouped E8P-I4 K2 entries (`kernels::qwen4_moe_sym_gfx11`, gfx1100 and gfx1151; one instruction path, arch-forced deltas
/// only): `--epi all` (default) is the `qwen4_moe_iu4_sym_e8p_pm_<arch>` module (BF16 gate/up, BF16 down, then the `_acc` pair);
/// `gate_up`/`down` one shipped-shape entry, `gate_up_acc`/`down_acc` one F32 qualification entry.
fn qwen4_moe_sym_gfx11_e8p(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::{
        iu4_gemm,
        qwen4_moe_sym_gfx11::{self as k2, Codec, Kind, Negative, Spec},
    };
    let spec = |kind, acc| Spec {
        arch,
        codec: Codec::E8p,
        kind,
        acc,
        negative: Negative::None,
    };
    if epi == "all" {
        let mut entries = Vec::new();
        for acc in [false, true] {
            for kind in Kind::ALL {
                entries.push(k2::emit(spec(kind, acc))?)
            }
        }
        let (text, proof) = iu4_gemm::module(&entries, &Codec::E8p.module(arch))?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let (name, acc) = match epi.strip_suffix("_acc") {
        Some(n) => (n, true),
        None => (epi, false),
    };
    let kind = match name {
        "gate_up" => Kind::GateUp,
        "down" => Kind::Down,
        _ => {
            return Err(format!(
                "qwen4_moe_sym_gfx11_e8p --epi {epi} (gate_up|down|gate_up_acc|down_acc|all)"
            ))
        }
    };
    let e = k2::emit(spec(kind, acc))?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// `--epi all` (default) emits the convert and attention symbols as one module; `convert`/`attend` one symbol.
fn qsa_gather(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qsa_gather::{self, Kind, Spec};
    if epi == "all" {
        let (_, text, proof) = qsa_gather::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let kind = match epi {
        "convert" => Kind::Convert,
        "attend" => Kind::Attend,
        _ => return Err(format!("qsa_gather --epi {epi} (convert|attend|all)")),
    };
    let e = qsa_gather::emit(Spec { arch, kind })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// The PM canonical parallel target-row sampler (`spec_cpu_ar_rows_parallel_f32_pm_<arch>`), one symbol per module.
fn spec_sampling_rows(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) = peacemaker_isa::kernels::spec_sampling_rows_pm::emit_module(arch)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// The PM ordered sampled-prefix acceptance kernel (`spec_accept_ordered_u32_pm_<arch>`), one symbol per module.
fn spec_accept(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) = peacemaker_isa::kernels::spec_accept_pm::emit_module(arch)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// The PM balanced-ownership assignment kernel of the Flash-Next EP exchange (`qwen4_ep_owner_assign_pm_gfx1201`), one symbol per module.
fn qwen4_ep_owner_assign(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) = peacemaker_isa::kernels::qwen4_ep_owner_assign_pm::emit_module(arch)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// The PM tile-ownership mask of the owner-routed EP chunk MoE (`ep_tile_mask_pm_gfx1100` / `ep_tile_mask_pm_gfx1201`), one symbol per module.
fn ep_tile_mask(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) = peacemaker_isa::kernels::ep_tile_mask::emit_module(arch)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// The PM tile pull of the owner-routed EP chunk MoE, one symbol per module: `ep_gather_grouped_rows_by_tile_owner_pm_<arch>`
/// (`Variant::Ep2`, the 48-byte two-rank kernel) or `ep_gather_grouped_rows_by_tile_owner_n8_pm_<arch>` (`Variant::N8`, the 104-byte rank-generic kernel, 2..=8 ranks); `<arch>` is gfx1100 or gfx1201.
fn ep_tile_pull(
    variant: peacemaker_isa::kernels::ep_gather_grouped_rows_by_tile_owner::Variant,
    arch: Arch,
) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) =
        peacemaker_isa::kernels::ep_gather_grouped_rows_by_tile_owner::emit_module(variant, arch)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// The ordered BF16 grouped expert combine twin (gfx1201 only).
fn moe_slot_ops(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) = peacemaker_isa::kernels::moe_slot_ops::emit_module(arch)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1201 exact-INT8 MQ6G256V2 prefill GEMM twin of the hipcc `gemm_mq6g256v2_mmq_i8` route:
/// `--epi set|set_bf16` one entry, `all` (default) the four-entry module (both epilogues on 128x128 and 128x256); `--tile
/// 128x128|256x128|128x256` (default the shipped 128x128; a single entry only).
fn mq6_i8_gemm(epi: &str, tile: Option<&str>, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::mq6_i8_gemm::{self, Epi, Raster, Spec, Tile};
    let tile: Tile = tile.unwrap_or("128x128").parse()?;
    if epi == "all" {
        if tile != Tile::T128x128 {
            return Err(
                "mq6_i8_gemm --tile: the product module holds both tiles; name one --epi".into(),
            );
        }
        let (_, text, proof) = mq6_i8_gemm::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let epi: Epi = epi
        .parse()
        .map_err(|_| format!("mq6_i8_gemm --epi {epi} (set|set_bf16|all)"))?;
    let e = mq6_i8_gemm::emit(Spec {
        epi,
        raster: Raster::Banded,
        tile,
        arch,
    })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// Grouped-position down plane (gfx1201): the grouped producer plus the down NT4/NT8 twins as one module.
fn qwen4_moe_down_gp(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) = peacemaker_isa::kernels::qwen4_moe_down_gp::emit_module(arch)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// The eight-columns-per-lane ordered BF16 grouped expert combine (gfx1201 only), a separate module from v1.
fn moe_slot_ops_v2(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) = peacemaker_isa::kernels::moe_slot_ops::emit_module_v2(arch)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// `all` (default) emits F32 rows16 + rows32 score and select; `bf16` emits the BF16-pooled rows16 + rows32 scores.
fn qsa_select(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qsa_select::{self, Kind, Spec};
    if epi == "all" {
        let (_, text, proof) = qsa_select::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    if epi == "bf16" {
        let (_, text, proof) = qsa_select::emit_module_bf16(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let kind = match epi {
        "score" => Kind::Score,
        "score32" => Kind::Score32,
        "score-bf16" => Kind::ScoreBf16,
        "score-bf16-rows32" => Kind::ScoreBf16Rows32,
        "select" => Kind::Select,
        _ => {
            return Err(format!(
            "qsa_select --epi {epi} (score|score32|score-bf16|score-bf16-rows32|select|all|bf16)"
        ))
        }
    };
    let e = qsa_select::emit(Spec { arch, kind })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1201 sampled-DFlash device accept (`dflash_sampled_accept_pm_gfx1201`, one symbol).
fn dflash_accept(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    if epi != "all" {
        return Err(format!("dflash_accept --epi {epi} (all)"));
    }
    let (_, text, proof) = peacemaker_isa::kernels::dflash_accept::emit_module(arch)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1201 MQ6 trunk GEMM twin: `--epi w4|w8` one entry, `all` the module.
fn qwen4_mq6_x4(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qwen4_mq6_x4::{self, Spec};
    if epi == "all" {
        let (_, text, proof) = qwen4_mq6_x4::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let wr = match epi {
        "w4" => 4,
        "w8" => 8,
        _ => return Err(format!("qwen4_mq6_x4 --epi {epi} (w4|w8|all)")),
    };
    let e = qwen4_mq6_x4::emit(Spec { arch, wr })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1151 MQ6 trunk GEMM twin of hipcc's U3 BV8/RW8 entries: `--epi <wr>[_<kind>]`
/// (`w8`, `w4`, `w8_bf16out`, `w4_bf16out`, `w8_regions`, `w8_hcw`) one entry, `all` the module.
fn qwen4_mq6_x4_gfx11(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qwen4_mq6_x4_gfx11::{self, Kind, Spec};
    if epi == "all" {
        let (_, text, proof) = qwen4_mq6_x4_gfx11::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let (wr, kind) = match epi {
        "w4" => (4, Kind::Plain),
        "w8" => (8, Kind::Plain),
        "w4_bf16out" => (4, Kind::Bf16),
        "w8_bf16out" => (8, Kind::Bf16),
        "w8_regions" => (8, Kind::Regions),
        "w8_hcw" => (8, Kind::Hcw),
        _ => {
            return Err(format!(
            "qwen4_mq6_x4_gfx11 --epi {epi} (w4|w8|w4_bf16out|w8_bf16out|w8_regions|w8_hcw|all)"
        ))
        }
    };
    let e = qwen4_mq6_x4_gfx11::emit(Spec { arch, wr, kind })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1151 Flash-Next lane-step row kernels: the whole `qwen4_lane_rows_pm` module (`--epi all` only).
fn qwen4_lane_rows(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qwen4_lane_rows;
    if epi != "all" {
        return Err(format!("qwen4_lane_rows --epi {epi} (all)"));
    }
    if arch != Arch::Gfx1151 {
        return Err("qwen4_lane_rows is exact-gfx1151 only".into());
    }
    let (_, text, proof) = qwen4_lane_rows::emit_module()?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1151 MQ4G256V2 trunk GEMM twin of the hipcc MQ4 route (the `qwen4_mq6_x4_gfx11` builder over
/// the 136-byte MQ4 group): `--epi <wr>[_<kind>]` as there, `all` the module.
fn qwen4_mq4_x4_gfx11(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qwen4_mq4_x4_gfx11::{self, Kind, Spec};
    if epi == "all" {
        let (_, text, proof) = qwen4_mq4_x4_gfx11::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let (wr, kind) = match epi {
        "w4" => (4, Kind::Plain),
        "w8" => (8, Kind::Plain),
        "w4_bf16out" => (4, Kind::Bf16),
        "w8_bf16out" => (8, Kind::Bf16),
        "w8_regions" => (8, Kind::Regions),
        "w8_hcw" => (8, Kind::Hcw),
        _ => {
            return Err(format!(
            "qwen4_mq4_x4_gfx11 --epi {epi} (w4|w8|w4_bf16out|w8_bf16out|w8_regions|w8_hcw|all)"
        ))
        }
    };
    let e = qwen4_mq4_x4_gfx11::emit(Spec { arch, wr, kind })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// Production row images retain their narrow exports and add only fixed-eight
/// physical tiles. Diagnostic cap16/cap32 builders are not production exports.
struct RowImage {
    module: String,
    entries: Vec<Emitted>,
    contracts: Vec<serde_json::Value>,
}

fn row_export(e: &Emitted) -> Result<&str, String> {
    e.s_text
        .lines()
        .find_map(|line| line.trim().strip_prefix(".amdhsa_kernel "))
        .map(str::trim)
        .ok_or_else(|| format!("{}: missing exported descriptor", e.proof.variant))
}

impl RowImage {
    fn new(module: String) -> Self {
        Self {
            module,
            entries: Vec::new(),
            contracts: Vec::new(),
        }
    }

    fn push(
        &mut self,
        e: Emitted,
        grid: [&str; 3],
        row_domain: serde_json::Value,
        mut contract: serde_json::Value,
    ) -> Result<(), String> {
        let symbol = row_export(&e)?;
        let metadata = e
            .s_text
            .split_once(".amdgpu_metadata\n")
            .ok_or_else(|| format!("{symbol}: missing metadata"))?
            .1;
        let field = |name: &str| -> Result<u32, String> {
            metadata
                .lines()
                .find_map(|line| line.trim().strip_prefix(name))
                .ok_or_else(|| format!("{symbol}: missing metadata field {name}"))?
                .trim()
                .parse::<u32>()
                .map_err(|err| format!("{symbol}: {name}: {err}"))
        };
        if !contract.is_object() {
            return Err(format!("{symbol}: shape contract is not an object"));
        }
        // Bind resources and ABI to this emitted entry, never an old wide
        // geometry or a guessed register/LDS ceiling. The metadata preserves
        // every kernarg's name, byte offset, size and access mode verbatim.
        contract["symbol"] = symbol.into();
        contract["module"] = self.module.clone().into();
        contract["arch"] = e.proof.arch.name().into();
        contract["kernarg_segment_bytes"] = field(".kernarg_segment_size:")?.into();
        contract["group_segment_fixed_bytes"] = field(".group_segment_fixed_size:")?.into();
        contract["launch_dynamic_lds_bytes"] = 0.into();
        contract["vgpr_max"] = e.shape.next_free_vgpr.into();
        contract["sgpr_max"] = e.shape.next_free_sgpr.into();
        contract["require_wave32"] = true.into();
        contract["require_zero_spills"] = true.into();
        contract["require_zero_private"] = true.into();
        contract["launch"] = serde_json::json!({
            "grid": grid, "block": [field(".max_flat_workgroup_size:")?, 1, 1],
            "row_domain": row_domain,
        });
        contract["amdgpu_metadata"] = metadata.into();
        contract["builder_proof"] =
            serde_json::to_value(&e.proof).map_err(|err| err.to_string())?;
        contract["isa_shape"] = serde_json::to_value(&e.shape).map_err(|err| err.to_string())?;
        self.contracts.push(contract);
        self.entries.push(e);
        Ok(())
    }
}

fn row_image(kernel: &str, epi: &str, arch: Arch) -> Result<Option<RowImage>, String> {
    use peacemaker_isa::kernels;
    use serde_json::json;
    let empty = || json!({});
    let rows = |max| json!({"logical_rows": {"min": 1, "max": max}});
    let slots = |tiled| {
        if tiled {
            json!({"C": "total logical rows 1..64", "U": 80,
            "descriptor_words": 482, "descriptor_stride_bytes": 1928,
            "slot_ids": "absolute; X/Y remain full-window bases"})
        } else {
            json!({"C": "1..8", "U": "distinct experts", "slot_ids": "absolute"})
        }
    };
    let selected = |tag: &str, tiled: bool| {
        epi == "all"
            || if tiled {
                epi.strip_suffix("_t8") == Some(tag)
            } else {
                epi == tag
            }
    };
    let mut image = match kernel {
        "qwen4_lane_rows" if arch == Arch::Gfx1100 => {
            if epi != "all" {
                return Err("qwen4_lane_rows --arch gfx1100 requires --epi all".into());
            }
            RowImage::new("qwen4_lane_rows_pm_gfx1100".into())
        }
        "lane_rows_mq6_gfx1201" | "lane_rows_bf16_gfx1201" => {
            if arch != Arch::Gfx1201 || epi != "all" {
                return Err(format!("{kernel} requires --arch gfx1201 --epi all"));
            }
            RowImage::new(if kernel == "lane_rows_mq6_gfx1201" {
                kernels::lane_rows_mq6_gfx1201::MODULE.into()
            } else {
                kernels::lane_rows_bf16_gfx1201::MODULE.into()
            })
        }
        "mq4_rows_x4" => RowImage::new(kernels::mq4_rows_x4::Spec::module(arch)),
        "qwen4_bf16_rows_wide" => RowImage::new(kernels::qwen4_bf16_rows_wide::module(arch)),
        "legacy_mq4_rowtiles" => {
            if epi != "all" {
                return Err("legacy_mq4_rowtiles requires --epi all".into());
            }
            RowImage::new(format!("legacy_mq4_rowtiles_pm_{}", arch.name()))
        }
        "mq_i4_moe_gemv" if arch == Arch::Gfx1100 => {
            return Err("gfx1100 production uses --kernel mq_i4_moe_gemv_mq4only; the retired mixed image is not emitted".into());
        }
        "mq_i4_moe_gemv_mq4only" => {
            if arch != Arch::Gfx1100 {
                return Err("mq_i4_moe_gemv_mq4only is gfx1100 only".into());
            }
            // Preserve the landing object's internal module ID; its dedicated
            // filename and MQ4-only export set identify the production image.
            RowImage::new(kernels::mq_i4_moe_gemv_gfx1100::Spec::module(arch))
        }
        "mq_i4_moe_gemv" => RowImage::new(kernels::mq_i4_moe_gemv::Spec::module(arch)),
        "mq_i4_moe_gemv_cn" => {
            if arch != Arch::Gfx1151 {
                return Err("mq_i4_moe_gemv_cn is gfx1151 only".into());
            }
            RowImage::new(kernels::mq_i4_moe_gemv_cn::Spec::module())
        }
        "e8p_i4_moe_gemv" => RowImage::new(kernels::e8p_i4_moe_gemv::Spec::module(arch)),
        "qwen4_slab_union_plan" => RowImage::new(kernels::qwen4_slab_union_plan::symbol(arch)),
        _ => return Ok(None),
    };
    match kernel {
        "qwen4_lane_rows" => {
            let mut entries = kernels::qwen4_lane_rows_mq6::build_rowtiles_gfx1100()?;
            entries.extend(kernels::qwen4_lane_rows_q8_gfx1100::build_rowtiles_gfx1100()?);
            let exports = [
                ("gemv_mq6g256v2_rows_t8_pm_gfx1100", "ceil(M/8)"),
                (
                    "gemv_mq6g256v2_rows_x4_t8_pm_gfx1100",
                    "sum ceil(max(Mi,0)/8)",
                ),
                ("gemv_q8_0_k8_rows_t8_pm_gfx1100", "ceil(M/2)"),
                ("gemv_q8_0_k320_rows_t8_pm_gfx1100", "ceil(M/8)"),
                ("gemv_q8_0_k2560_rows_t8_pm_gfx1100", "ceil(M/8)"),
                (
                    "gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100",
                    "ceil(M0/8)+ceil(M1/8)",
                ),
                ("gemv_q8_0_wide_rows_t8_pm_gfx1100", "ceil(M/8)"),
            ];
            if entries.len() != exports.len() {
                return Err(format!(
                    "{}: expected {} exports, got {}",
                    image.module,
                    exports.len(),
                    entries.len()
                ));
            }
            for (e, (symbol, grid)) in entries.into_iter().zip(exports) {
                if e.proof.arch != arch || row_export(&e)? != symbol {
                    return Err(format!(
                        "{}: expected exact gfx1100 export {symbol}, got {} ({:?})",
                        image.module,
                        row_export(&e)?,
                        e.proof.arch
                    ));
                }
                let contract = match symbol {
                    "gemv_q8_0_k8_rows_t8_pm_gfx1100" => json!({
                        "shape_domain": {"K": "K>0 && K%256==0"},
                        "arithmetic_recipe": "gfx1100 actual HC singleton, eight fixed K partitions",
                    }),
                    "gemv_q8_0_k320_rows_t8_pm_gfx1100" => json!({
                        "shape_domain": {"K": "K==320"},
                        "arithmetic_recipe": "gfx1100 actual small-K four-chain singleton; not Halo K320",
                    }),
                    "gemv_q8_0_k2560_rows_t8_pm_gfx1100" => json!({
                        "shape_domain": {"K": "K>1536 && K%32==0"},
                        "arithmetic_recipe": "gfx1100 actual plain one-chain singleton",
                        "symbol_note": "k2560 is historical naming, not a literal K restriction",
                    }),
                    "gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100" => json!({
                        "shape_domain": {"K": "K==2560"},
                        "arithmetic_recipe": "gfx1100 actual plain one-chain singleton",
                    }),
                    "gemv_q8_0_wide_rows_t8_pm_gfx1100" => json!({
                        "shape_domain": {"K": "K>0 && K<=1536 && K%32==0"},
                        "arithmetic_recipe": "gfx1100 actual small-K four-chain singleton",
                    }),
                    _ => empty(),
                };
                image.push(e, [grid, "ceil(R/8)", "1"], rows(64), contract)?;
            }
        }
        "lane_rows_mq6_gfx1201" => {
            use kernels::lane_rows_mq6_gfx1201 as k;
            for (spec, grid) in k::SPECS.into_iter().zip(k::GRID_RULE) {
                image.push(k::emit(spec)?, [grid, "1", "1"], rows(spec.cap), empty())?;
            }
            for x4 in [false, true] {
                image.push(
                    k::emit_rowtiles(k::Spec {
                        x4,
                        cap: 8,
                        wg8: true,
                    })?,
                    [
                        if x4 {
                            "sum ceil(max(Mi,0)/8)"
                        } else {
                            "ceil(M/8)"
                        },
                        "ceil(R/8)",
                        "1",
                    ],
                    rows(64),
                    empty(),
                )?;
            }
        }
        "lane_rows_bf16_gfx1201" => {
            use kernels::lane_rows_bf16_gfx1201 as k;
            for (entry, grid) in k::ENTRIES.into_iter().zip(k::GRID_RULE) {
                image.push(k::build(entry)?, [grid, "1", "1"], rows(entry.cap), empty())?;
            }
            for hc in [false, true] {
                image.push(
                    k::build_rowtiles(k::Entry {
                        hc,
                        wg8: false,
                        cap: 8,
                    })?,
                    [if hc { "M" } else { "sum max(Mi,0)" }, "ceil(R/8)", "1"],
                    rows(64),
                    empty(),
                )?;
            }
        }
        "mq4_rows_x4" => {
            use kernels::mq4_rows_x4 as k;
            if epi != "all" && epi != "entry" && epi != "t8" {
                return Err(format!("{kernel} --epi {epi} (entry|t8|all)"));
            }
            if arch == Arch::Gfx1100 && epi == "entry" {
                return Err("mq4_rows_x4 --arch gfx1100 supports only --epi t8|all; no narrow export exists".into());
            }
            // The emitter owns composition: narrow+t8, or gfx1100 t8-only.
            let (entries, _, _) = k::emit_module(arch)?;
            for (i, e) in entries.into_iter().enumerate() {
                let tiled = arch == Arch::Gfx1100 || i == 1;
                if epi == "all" || (epi == "t8") == tiled {
                    image.push(
                        e,
                        [
                            "sum ceil(max(Mi,0)/2)",
                            if tiled { "ceil(R/8)" } else { "1" },
                            "1",
                        ],
                        rows(if tiled { 64 } else { 8 }),
                        empty(),
                    )?;
                }
            }
        }
        "qwen4_bf16_rows_wide" => {
            use kernels::qwen4_bf16_rows_wide as k;
            if !matches!(epi, "all" | "t8" | "k4_t8") {
                return Err(format!("{kernel} --epi {epi} (t8|k4_t8|all)"));
            }
            for hc in [false, true] {
                if epi == "all" || (epi == "k4_t8") == hc {
                    let e = if arch == Arch::Gfx1100 {
                        kernels::qwen4_bf16_rows_gfx1100::emit_rowtiles(hc)?
                    } else {
                        k::emit_rowtiles(arch, hc)?
                    };
                    image.push(
                        e,
                        [if hc { "M" } else { "sum max(Mi,0)" }, "ceil(R/8)", "1"],
                        rows(64),
                        empty(),
                    )?;
                }
            }
        }
        "mq_i4_moe_gemv_mq4only" => {
            use kernels::mq_i4_moe_gemv_gfx1100 as k;
            for tiled in [false, true] {
                for spec in k::module_specs(arch)
                    .into_iter()
                    .filter(|s| s.codec == k::Codec::Mq4Sym)
                {
                    if selected(&spec.tag(), tiled) {
                        let e = if tiled {
                            k::emit_rowtiles(spec)?
                        } else {
                            k::emit(spec)?
                        };
                        let contract = k::contract(spec, &e);
                        image.push(
                            e,
                            if tiled {
                                ["ceil(M/8)", "80", "ceil(C/8)"]
                            } else {
                                ["M", "U", "1"]
                            },
                            slots(tiled),
                            contract,
                        )?;
                    }
                }
            }
        }
        "mq_i4_moe_gemv" if arch == Arch::Gfx1151 => {
            // This common Halo image remains a legacy diagnostic/reference
            // object. Product rowtiles belong to mq_i4_moe_gemv_cn instead.
            use kernels::mq_i4_moe_gemv as k;
            for spec in k::module_specs(arch) {
                if selected(&spec.tag(), false) {
                    let e = k::emit(spec)?;
                    let contract = k::contract(spec, &e);
                    image.push(e, [if spec.cap == k::Cap::C8 { "M" } else { "ceil(M/8)" }, "U", "1"],
                        json!({"C_max": spec.cap.slots(), "U": "distinct experts", "diagnostic": true}),
                        contract)?;
                }
            }
        }
        "mq_i4_moe_gemv" => {
            use kernels::mq_i4_moe_gemv as k;
            for tiled in [false, true] {
                for spec in k::narrow_specs(arch) {
                    if selected(&spec.tag(), tiled) {
                        let e = if tiled {
                            k::emit_rowtiles(spec)?
                        } else {
                            k::emit(spec)?
                        };
                        let contract = if tiled {
                            k::contract_rowtiles(spec, &e)
                        } else {
                            k::contract(spec, &e)
                        };
                        image.push(
                            e,
                            if tiled {
                                ["ceil(M/8)", "80", "ceil(C/8)"]
                            } else {
                                ["M", "U", "1"]
                            },
                            slots(tiled),
                            contract,
                        )?;
                    }
                }
            }
        }
        "mq_i4_moe_gemv_cn" => {
            use kernels::mq_i4_moe_gemv_cn as k;
            for tiled in [false, true] {
                for spec in k::module_specs() {
                    if selected(&spec.tag(), tiled) {
                        let e = if tiled {
                            k::emit_rowtiles(spec)?
                        } else {
                            k::emit(spec)?
                        };
                        let contract = if tiled {
                            k::contract_rowtiles(spec, &e)
                        } else {
                            k::contract(spec, &e)
                        };
                        image.push(
                            e,
                            if tiled {
                                ["ceil(M/8)", "80", "ceil(C/8)"]
                            } else {
                                ["M/8", "U", "1"]
                            },
                            slots(tiled),
                            contract,
                        )?;
                    }
                }
            }
        }
        "e8p_i4_moe_gemv" => {
            use kernels::e8p_i4_moe_gemv as k;
            for tiled in [false, true] {
                for spec in k::module_specs(arch) {
                    if selected(&spec.tag(), tiled) {
                        let e = if tiled {
                            k::emit_rowtiles(spec)?
                        } else {
                            k::emit(spec)?
                        };
                        let contract = k::contract(spec, &e);
                        image.push(
                            e,
                            if tiled {
                                ["ceil(M/8)", "80", "ceil(C/8)"]
                            } else {
                                ["M", "U", "1"]
                            },
                            slots(tiled),
                            contract,
                        )?;
                    }
                }
            }
        }
        "legacy_mq4_rowtiles" => {
            let stages = [
                ("gate_up", ["M/8", "10", "ceil(R/8)"]),
                ("down", ["M", "top_k", "ceil(R/8)"]),
            ];
            for ((stage, grid), e) in stages
                .into_iter()
                .zip(kernels::legacy_mq4_rowtiles::build_rowtiles(arch)?)
            {
                image.push(
                    e,
                    grid,
                    json!({
                        "logical_rows": {"min": 1, "max": 64},
                        "physical_token_rows": 8,
                    }),
                    json!({
                        "projection": stage,
                        "representation": "legacy aggregate qt44 gate/up and qt53 down",
                        "reference": "original indexed singleton AR recipe",
                        "weight_storage": "original aggregate buffers; no repacking",
                    }),
                )?;
            }
        }
        "qwen4_slab_union_plan" => {
            if epi != "all" && epi != "t8" {
                return Err(format!("{kernel} --epi {epi} (t8|all)"));
            }
            image.push(kernels::qwen4_slab_union_plan::emit(arch)?, ["ceil(rows/8)", "1", "1"],
                json!({"argument": "rows", "min": 1, "max": 64,
                    "descriptor_words": 482, "descriptor_stride_bytes": 1928,
                    "slot_capacity": 80, "expert_order": "first occurrence", "slot_ids": "absolute"}),
                empty())?;
        }
        _ => unreachable!(),
    }
    if image.entries.is_empty() {
        return Err(format!(
            "{kernel}: unknown --epi {epi}; use all or an incumbent tag with optional _t8"
        ));
    }
    Ok(Some(image))
}
/// gfx1201 planar MQ3-I4 / MQ2-I4 one-row decode GEMVs (`kernels::mq_i4_rows`):
/// `--epi <w3|w2>_<r256|r128>` one entry, `all` (default) the four-entry module.
fn mq_i4_rows_specs(
    epi: &str,
    arch: Arch,
) -> Result<Vec<peacemaker_isa::kernels::mq_i4_rows::Spec>, String> {
    use peacemaker_isa::kernels::mq_i4_rows;
    let all = mq_i4_rows::module_specs(arch);
    if epi == "all" {
        return Ok(all);
    }
    all.into_iter()
        .find(|s| s.tag() == epi)
        .map(|s| vec![s])
        .ok_or_else(|| format!("mq_i4_rows --epi {epi} (w3_r256|w3_r128|w2_r256|w2_r128|all)"))
}
fn mq_i4_rows(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::mq_i4_rows;
    if epi == "all" {
        let (_, text, proof) = mq_i4_rows::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let [spec] = mq_i4_rows_specs(epi, arch)?[..] else {
        return Err("mq_i4_rows: one entry expected".into());
    };
    let e = mq_i4_rows::emit(spec)?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1201 row-local R128 / K640 A4 producer (`kernels::mq_i4_a4`, symbol `mq_i4_a4_r128_k640`, module
/// `mq_i4_a4_pm_gfx1201`): `--epi r128_k640|all` (default `all`), both the one entry.
/// Usage: `peacemaker-isa emit --kernel mq_i4_a4 [--epi r128_k640|all] --arch gfx1201 --out FILE.s --proof FILE.proof.json [--co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]]`
/// (with `--co`, M7 runs on the linked export; `--shape` writes `DIR/mq_i4_a4.gfx1201.r128_k640.contract.json`).
fn mq_i4_a4(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::mq_i4_a4;
    if epi == "all" {
        let (_, text, proof) = mq_i4_a4::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let [spec] = mq_i4_a4::module_specs(arch)[..] else {
        return Err("mq_i4_a4: one entry expected".into());
    };
    if epi != spec.tag() {
        return Err(format!("mq_i4_a4 --epi {epi} ({}|all)", spec.tag()));
    }
    let e = mq_i4_a4::emit(spec)?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1201 runtime-K R128 A4 producer (`kernels::moe_rotate_i4_pm`, symbol and module `moe_rotate_i4_pm_gfx1201`): the one
/// entry, `--epi r128|all` (default `all`).
/// Usage: `peacemaker-isa emit --kernel moe_rotate_i4_pm --arch gfx1201 --out FILE.s --proof FILE.proof.json [--co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]]`
/// (with `--co`, M7 runs on the linked export; `--shape` writes `DIR/moe_rotate_i4_pm.gfx1201.contract.json`).
fn moe_rotate_i4_pm(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::moe_rotate_i4_pm;
    if epi == "all" {
        let (_, text, proof) = moe_rotate_i4_pm::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let [spec] = moe_rotate_i4_pm::module_specs(arch)[..] else {
        return Err("moe_rotate_i4_pm: one entry expected".into());
    };
    if epi != spec.tag() {
        return Err(format!("moe_rotate_i4_pm --epi {epi} ({}|all)", spec.tag()));
    }
    let e = moe_rotate_i4_pm::emit(spec)?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1151 K1-C1 one-lane slab decode GEMVs (`kernels::mq_i4_moe_gemv_c1`, both codecs per launch):
/// `--epi <r256_gate_up|r128_down>` one entry, `all` (default) the two-entry module.
fn mq_i4_moe_gemv_c1_specs(
    epi: &str,
    arch: Arch,
) -> Result<Vec<peacemaker_isa::kernels::mq_i4_moe_gemv_c1::Spec>, String> {
    use peacemaker_isa::kernels::mq_i4_moe_gemv_c1;
    if arch != Arch::Gfx1151 {
        return Err("mq_i4_moe_gemv_c1 is gfx1151 only".into());
    }
    let all = mq_i4_moe_gemv_c1::module_specs();
    if epi == "all" {
        return Ok(all);
    }
    all.into_iter()
        .find(|s| s.tag() == epi)
        .map(|s| vec![s])
        .ok_or_else(|| format!("mq_i4_moe_gemv_c1 --epi {epi} (r256_gate_up|r128_down|all)"))
}
fn mq_i4_moe_gemv_c1(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::mq_i4_moe_gemv_c1;
    if arch != Arch::Gfx1151 {
        return Err("mq_i4_moe_gemv_c1 is gfx1151 only".into());
    }
    if epi == "all" {
        let (_, text, proof) = mq_i4_moe_gemv_c1::emit_module()?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let [spec] = mq_i4_moe_gemv_c1_specs(epi, arch)?[..] else {
        return Err("mq_i4_moe_gemv_c1: one entry expected".into());
    };
    let e = mq_i4_moe_gemv_c1::emit(spec)?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// `--epi w8|w4` one entry, `all` the module.
fn qwen4_shared_gu_gfx11(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::qwen4_shared_gu_gfx11::{self, Spec};
    if epi == "all" {
        let (_, text, proof) = qwen4_shared_gu_gfx11::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let wr = match epi {
        "w4" => 4,
        "w8" => 8,
        _ => return Err(format!("qwen4_shared_gu_gfx11 --epi {epi} (w4|w8|all)")),
    };
    let e = qwen4_shared_gu_gfx11::emit(Spec { arch, wr })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1151 MTP append DET dense F16 WMMA GEMM (`mtp_dense_det_pm_gfx1151_w8`; module `mtp_dense_det_pm_gfx1151`):
/// `--epi w8|all`, both the one entry.
fn mtp_dense_det_gfx11(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::mtp_dense_det_gfx11::{self, Spec};
    if epi == "all" {
        let (_, text, proof) = mtp_dense_det_gfx11::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    if epi != "w8" {
        return Err(format!("mtp_dense_det_gfx11 --epi {epi} (w8|all)"));
    }
    let e = mtp_dense_det_gfx11::emit(Spec { arch })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1151 MTP append DET Q8_0 -> F16 weight converter (`mtp_q8_0_to_f16_pm_gfx1151`; module `mtp_q8_det_pm_gfx1151`):
/// `--epi convert|all`, both the one entry.
fn mtp_q8_det_gfx11(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::mtp_q8_det_gfx11::{self, Spec};
    if epi == "all" {
        let (_, text, proof) = mtp_q8_det_gfx11::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    if epi != "convert" {
        return Err(format!("mtp_q8_det_gfx11 --epi {epi} (convert|all)"));
    }
    let e = mtp_q8_det_gfx11::emit(Spec { arch })?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1201 MQ4G256V2 wide exact-verify twins of hipcc's `_vt{4,8}w{4,8}_k32`
/// entries: `--epi <op>_bt<4|8>w<4|8>` one entry, `all` (default) the module.
fn mq4_verify_specs(
    epi: &str,
    arch: Arch,
) -> Result<Vec<peacemaker_isa::kernels::mq4_verify::Spec>, String> {
    use peacemaker_isa::kernels::mq4_verify::Spec;
    if epi == "all" {
        return Ok(Spec::all(arch));
    }
    let bad = || format!("mq4_verify --epi {epi} (<qkvza|qkv|gate_up|residual>_bt<4|8>w<4|8>|all)");
    let (op, tile) = epi.rsplit_once("_bt").ok_or_else(bad)?;
    let (bt, waves) = tile.split_once('w').ok_or_else(bad)?;
    Ok(vec![Spec {
        arch,
        bt: bt.parse().map_err(|_| bad())?,
        waves: waves.parse().map_err(|_| bad())?,
        op: op.parse()?,
    }])
}
fn mq4_verify(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::mq4_verify;
    if epi == "all" {
        let (_, text, proof) = mq4_verify::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let [spec] = mq4_verify_specs(epi, arch)?[..] else {
        return Err("mq4_verify: one entry expected".into());
    };
    let e = mq4_verify::emit(spec)?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1201 MQ4G256V2 symmetric FP8 verify GEMMs (1 <= N <= 128): `--epi <op>_bt<b>w<w>`
/// (`qkvza|qkv|gate_up|residual|set`, tiles `bt1w1`, `bt{2,4,8}w{4,8}`) one entry, `all` (default) the module.
fn mq4_fp8_verify_specs(
    epi: &str,
    arch: Arch,
) -> Result<Vec<peacemaker_isa::kernels::mq4_fp8_verify::Spec>, String> {
    use peacemaker_isa::kernels::mq4_fp8_verify::Spec;
    if epi == "all" {
        return Ok(Spec::all(arch));
    }
    let bad = || {
        format!(
            "mq4_fp8_verify --epi {epi} (<qkvza|qkv|gate_up|residual|set>_bt<1|2|4|8>w<1|4|8>|all)"
        )
    };
    let (op, tile) = epi.rsplit_once("_bt").ok_or_else(bad)?;
    let (bt, waves) = tile.split_once('w').ok_or_else(bad)?;
    Ok(vec![Spec {
        arch,
        bt: bt.parse().map_err(|_| bad())?,
        waves: waves.parse().map_err(|_| bad())?,
        op: op.parse()?,
    }])
}
fn mq4_fp8_verify(epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::mq4_fp8_verify;
    if epi == "all" {
        let (_, text, proof) = mq4_fp8_verify::emit_module(arch)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let [spec] = mq4_fp8_verify_specs(epi, arch)?[..] else {
        return Err("mq4_fp8_verify: one entry expected".into());
    };
    let e = mq4_fp8_verify::emit(spec)?;
    Ok((
        e.s_text,
        serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
    ))
}
fn fp8_gemm(scale: &str, epi: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    use peacemaker_isa::kernels::fp8_gemm::{self, ActScale, Epi, Spec};
    if scale == "both" {
        if epi != "all" {
            return Err("--scale both requires --epi all".into());
        }
        let epis = [Epi::Set, Epi::Add, Epi::GateUpSilu, Epi::Qkv, Epi::Qkvza];
        let mut emitted = Vec::with_capacity(10);
        for act_scale in [ActScale::Row, ActScale::K128] {
            for &epi in &epis {
                emitted.push(fp8_gemm::emit(Spec {
                    arch,
                    act_scale,
                    epi,
                })?);
            }
        }
        let (text, proof) = fp8_gemm::module(&emitted)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let scale: ActScale = scale.parse()?;
    if matches!(epi, "all" | "initial") {
        let epis: &[Epi] = if epi == "initial" {
            &[Epi::Set, Epi::Add]
        } else {
            &[Epi::Set, Epi::Add, Epi::GateUpSilu, Epi::Qkv, Epi::Qkvza]
        };
        let (_, text, proof) = fp8_gemm::emit_module(arch, scale, epis)?;
        return Ok((
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        ));
    }
    let emitted = fp8_gemm::emit(Spec {
        arch,
        act_scale: scale,
        epi: epi.parse()?,
    })?;
    Ok((
        emitted.s_text,
        serde_json::to_vec_pretty(&emitted.proof).map_err(|e| e.to_string())?,
    ))
}
/// A PM decode twin module (`kernels::pm_decode`): every frozen export of the
/// HIP module combined into one code object source and proof.
fn pm_decode(module: &str, arch: Arch) -> Result<(String, Vec<u8>), String> {
    if arch != Arch::Gfx1201 {
        return Err(format!("pm_decode_{module} is exact-gfx1201 only"));
    }
    let (_, text, proof) = peacemaker_isa::kernels::pm_decode::emit_module(module)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// Re-slice the SiLU region from a hipcc disassembly and require it to equal
/// the committed golden the gate/up epilogue instantiates.
fn region_import(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    use peacemaker_isa::kernels::iu4_gemm::region;
    let mut dis = None;
    let mut symbol = None;
    let mut gdn = None;
    let mut write = false;
    let mut provenance = String::new();
    let mut arch = Arch::Gfx1201;
    while let Some(flag) = args.next() {
        if flag == "--write" {
            write = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--disassembly" => dis = Some(value),
            "--symbol" => symbol = Some(value),
            "--gdn-object" => gdn = Some(value),
            "--provenance" => provenance = value,
            "--arch" => arch = value.parse()?,
            _ => return Err(format!("unknown flag {flag}\n{USAGE}")),
        }
    }
    if let Some(object) = gdn {
        return gdn_region_import(&object, write, &provenance);
    }
    let text =
        fs::read_to_string(dis.ok_or("missing --disassembly")?).map_err(|e| e.to_string())?;
    if !arch.gfx12() {
        // gfx1100: hipcc's V2C gate/up object (`--write` regenerates the golden, prefixing `--provenance`).
        let symbol = symbol.unwrap_or_else(|| "gemm_mq4g256v2_gate_up_silu_iu4_v2c_gfx11".into());
        let slice = region::slice_silu_gfx11(&text, &symbol)?;
        region::Region::parse(&slice)?;
        let path = format!(
            "{}/kernels/iu4_v2c.gfx1100.silu.region.s",
            env!("CARGO_MANIFEST_DIR")
        );
        if write {
            let head: String = provenance
                .split("\\n")
                .map(|l| format!("; {l}\n"))
                .collect();
            fs::write(&path, format!("{head}{slice}")).map_err(|e| e.to_string())?;
            eprintln!("region-import: wrote {path}");
            return Ok(());
        }
        print!("{slice}");
        if slice != region::body_of(region::SILU_GOLDEN_GFX1100) {
            return Err(format!("sliced region differs from {path}"));
        }
        eprintln!("region-import: {symbol} gfx1100 SiLU slice matches the committed golden");
        return Ok(());
    }
    let symbol = symbol.unwrap_or_else(|| "gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3".into());
    let slice = region::slice_silu(&text, &symbol)?;
    print!("{slice}");
    if slice != region::golden_body() {
        return Err("sliced region differs from kernels/iu4_gemm.silu.region.s".into());
    }
    region::Region::parse(&slice)?;
    eprintln!("region-import: {symbol} SiLU slice matches the committed golden");
    Ok(())
}
/// Lift `gdn_chunk_prep`'s hipcc code object with peacemaker (byte-exact
/// round trip), slice its per-token regions and require them to equal the
/// committed goldens (`--write` regenerates them, prefixing `--provenance`).
fn gdn_region_import(object: &str, write: bool, provenance: &str) -> Result<(), String> {
    use peacemaker_isa::kernels::fp8_gemm::gdn_region::{self, golden_body};
    use sha2::{Digest, Sha256};
    let bytes = fs::read(object).map_err(|e| format!("{object}: {e}"))?;
    let lifted = peacemaker_lift::lift_object(
        &bytes,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Hipcc,
        },
    )
    .map_err(|e| e.to_string())?;
    let kernel = lifted
        .program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == "gdn_chunk_prep")
        .ok_or("object has no gdn_chunk_prep kernel")?;
    let lines = peacemaker_lift::emit::text(kernel, peacemaker_ir::inst::Arch::Gfx1201)
        .map_err(|e| e.to_string())?;
    let g = gdn_region::slice_gdn(&lines)?;
    let sha = format!("{:x}", Sha256::digest(&bytes));
    for (name, text, golden) in [
        ("conv_silu", &g.conv_silu, gdn_region::CONV_SILU_GOLDEN),
        ("norm_q", &g.norm_q, gdn_region::NORM_Q_GOLDEN),
        ("norm_k", &g.norm_k, gdn_region::NORM_K_GOLDEN),
        ("cvt_v", &g.cvt_v, gdn_region::CVT_V_GOLDEN),
    ] {
        let path = format!(
            "{}/kernels/fp8_gemm.gdn.{name}.region.s",
            env!("CARGO_MANIFEST_DIR")
        );
        if write {
            let head: String = provenance
                .split("\\n")
                .map(|l| format!("; {l}\n"))
                .collect();
            fs::write(&path,format!("{head}; Lifted with peacemaker-lift from {object} (sha256 {sha}); region `{name}`.\n; Regenerate/compare: `peacemaker-isa region-import --gdn-object <object>` (feature `lift`).\n{text}")).map_err(|e|e.to_string())?;
            eprintln!("region-import: wrote {path}");
        } else if text != &golden_body(golden) {
            return Err(format!("sliced {name} differs from {path}"));
        }
    }
    if !write {
        eprintln!("region-import: gdn_chunk_prep ({sha}) regions match the committed goldens")
    }
    Ok(())
}
/// The one supported A3B FA2-GQA geometry (H16 / KV2 / D256, Q8_0 K/V). Anything else is refused by the emitter itself.
fn fa_prefill_spec() -> peacemaker_isa::kernels::fa_prefill::FaPrefillSpec {
    use peacemaker_isa::kernels::fa_prefill::{FaKvCodec, FaPrefillSpec};
    FaPrefillSpec {
        heads: 16,
        kv_heads: 2,
        head_dim: 256,
        kv: FaKvCodec::Q8_0,
    }
}
/// gfx1201 causal Q8_0 FA2-GQA PeaceMaker twin: one module of two exports (Q preconvert, then direct attention).
/// The module source and proof cover both; `--co`/`--bundle` link the module, `--shape DIR` writes both contracts
/// and `--m7 FILE` the symbol-keyed receipt map.
fn fa_prefill_pm(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) =
        peacemaker_isa::kernels::fa_prefill::emit_module(arch, &fa_prefill_spec())?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// The native-fp8 twin's geometry: the same H16 / KV2 / D256, K/V codec native E4M3 with f16 scales.
fn fa_prefill_fp8_spec() -> peacemaker_isa::kernels::fa_prefill::FaPrefillSpec {
    use peacemaker_isa::kernels::fa_prefill::{FaKvCodec, FaPrefillSpec};
    FaPrefillSpec {
        heads: 16,
        kv_heads: 2,
        head_dim: 256,
        kv: FaKvCodec::Fp8E4M3,
    }
}
/// gfx1201 causal native-fp8 E4M3 FA2-GQA PeaceMaker twin: one module of ONE direct export (the Q preconvert is the
/// `fa_prefill_pm` module's). `--co`/`--bundle` link it, `--shape DIR` writes the contract, `--m7 FILE` the receipt map.
fn fa_prefill_fp8_pm(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) =
        peacemaker_isa::kernels::fa_prefill::emit_fp8_module(arch, &fa_prefill_fp8_spec())?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1201 A3B FA2-GQA Q prep (one-wave Q RMSNorm + gate passthrough + captured-angle half-split RoPE + f16 RNE
/// Q16): its own single-export module, separate from the `fa_prefill_pm` bundle. `--co`/`--bundle` link it,
/// `--shape DIR` writes the contract and `--m7 FILE` the symbol-keyed receipt map.
fn fa_qprep_pm(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) =
        peacemaker_isa::kernels::fa_qprep::emit_module(arch, &fa_prefill_spec())?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1201 exact C8 IU4 producers (rotate; RMSNorm + rotate) of the A3B F3 prefill route: one module, two exports.
/// `--co`/`--bundle` link it, `--shape DIR` writes both contracts and `--m7 FILE` the symbol-keyed receipt map.
fn iu4_c8_producer_pm(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) = peacemaker_isa::kernels::iu4_c8_producer::emit_module(arch)?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
/// gfx1201 A3B GDN chunk preparation (HK16/HV32/HD128 gates + conv/SiLU + q/k norm + f16 RNE): one export in its
/// own module. `--co`/`--bundle` link it, `--shape DIR` writes the contract and `--m7 FILE` the symbol-keyed
/// receipt map.
fn gdn_chunk_prep_pm(arch: Arch) -> Result<(String, Vec<u8>), String> {
    let (_, text, proof) = peacemaker_isa::kernels::gdn_chunk_prep::emit_module(
        arch,
        &peacemaker_isa::kernels::gdn_chunk_prep::GdnChunkPrepSpec::A3B,
    )?;
    Ok((
        text,
        serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
    ))
}
fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next();
    if command.as_deref() == Some("region-import") {
        return region_import(args);
    }
    if command.as_deref() != Some("emit") {
        return Err(format!("{USAGE}\n{NATIVE_USAGE}\n{FA_PREFILL_USAGE}\n{FA_PREFILL_FP8_USAGE}\n{FA_QPREP_USAGE}\n{IU4_C8_USAGE}\n{GDN_CHUNK_PREP_USAGE}"));
    }
    let mut kernel = None;
    let mut arch = None;
    let mut out = None;
    let mut proof = None;
    let mut variant = None;
    let (mut fold, mut tile, mut cacc, mut epi, mut scale, mut alayout) =
        (None, None, None, None, None, None);
    let mut mma = None::<String>;
    let (mut co, mut bundle, mut host) = (
        None,
        None,
        peacemaker_isa::native::DEFAULT_HOST_TARGET.to_owned(),
    );
    let (mut shape, mut m7) = (None::<String>, None::<String>);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--kernel" => kernel = Some(value),
            "--arch" => arch = Some(value.parse::<Arch>()?),
            "--out" => out = Some(value),
            "--proof" => proof = Some(value),
            "--variant" => variant = Some(value),
            "--fold" => fold = Some(value),
            "--tile" => tile = Some(value),
            "--cacc" => cacc = Some(value),
            "--epi" => epi = Some(value),
            "--scale" => scale = Some(value),
            "--alayout" => alayout = Some(value),
            "--mma" => mma = Some(value),
            "--co" => co = Some(value),
            "--bundle" => bundle = Some(value),
            "--host-target" => host = value,
            "--shape" => shape = Some(value),
            "--m7" => m7 = Some(value),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    let kernel = kernel.ok_or("missing --kernel")?;
    let arch = arch.ok_or("missing --arch")?;
    let row_image = row_image(&kernel, epi.as_deref().unwrap_or("all"), arch)?;
    let (text, proof_json) = if let Some(image) = &row_image {
        let (text, proof) =
            peacemaker_isa::kernels::iu4_gemm::module(&image.entries, &image.module)?;
        (
            text,
            serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
        )
    } else {
        match kernel.as_str() {
            "fold_magic" => {
                if let Some(var) = variant {
                    if var != "probe" {
                        return Err("fold_magic supports only variant probe".into());
                    }
                }
                let emitted = probe(arch)?;
                (
                    emitted.s_text,
                    serde_json::to_vec_pretty(&emitted.proof).map_err(|e| e.to_string())?,
                )
            }
            "iu4_gemm" => iu4_gemm(
                fold.as_deref().unwrap_or("k128"),
                tile.as_deref().ok_or("missing --tile")?,
                cacc.as_deref().unwrap_or("1"),
                epi.as_deref().ok_or("missing --epi")?,
                alayout.as_deref().unwrap_or("token"),
                mma.as_deref().unwrap_or("dense"),
                arch,
            )?,
            "gdn_scan" => {
                let e = peacemaker_isa::kernels::gdn_scan::emit(arch)?;
                (
                    e.s_text,
                    serde_json::to_vec_pretty(&e.proof).map_err(|e| e.to_string())?,
                )
            }
            "iu4_v2c" => iu4_v2c(epi.as_deref().unwrap_or("set"), arch)?,
            "iu4_v2b" => iu4_v2b(epi.as_deref().ok_or("missing --epi")?, arch)?,
            "iu4_v2b_a4" => iu4_v2b_a4(epi.as_deref().ok_or("missing --epi")?, arch)?,
            "qwen4_moe_sym" => qwen4_moe_sym(epi.as_deref().ok_or("missing --epi")?, arch)?,
            "qwen4_moe_sym_w3" => qwen4_moe_sym_w3(epi.as_deref().unwrap_or("all"), arch)?,
            "qwen4_moe_sym_w3_nt4" => qwen4_moe_sym_w3_nt4(epi.as_deref().unwrap_or("all"), arch)?,
            "qwen4_moe_sym_e8p" => qwen4_moe_sym_e8p(epi.as_deref().unwrap_or("all"), arch)?,
            "qwen4_moe_sym_gfx11_e8p" => {
                qwen4_moe_sym_gfx11_e8p(epi.as_deref().unwrap_or("all"), arch)?
            }
            "qsa_gather" => qsa_gather(epi.as_deref().unwrap_or("all"), arch)?,
            "qsa_select" => qsa_select(epi.as_deref().unwrap_or("all"), arch)?,
            "dflash_accept" => dflash_accept(epi.as_deref().unwrap_or("all"), arch)?,
            "spec_sampling_rows_pm" => spec_sampling_rows(arch)?,
            "spec_accept_pm" => spec_accept(arch)?,
            "qwen4_ep_owner_assign_pm" => qwen4_ep_owner_assign(arch)?,
            "ep_tile_mask_pm" => ep_tile_mask(arch)?,
            "moe_slot_ops_pm" => moe_slot_ops(arch)?,
            "qwen4_moe_down_gp_pm" => qwen4_moe_down_gp(arch)?,
            "moe_slot_ops_v2_pm" => moe_slot_ops_v2(arch)?,
            "ep_gather_grouped_rows_by_tile_owner_pm" => ep_tile_pull(
                peacemaker_isa::kernels::ep_gather_grouped_rows_by_tile_owner::Variant::Ep2,
                arch,
            )?,
            "ep_gather_grouped_rows_by_tile_owner_n8_pm" => ep_tile_pull(
                peacemaker_isa::kernels::ep_gather_grouped_rows_by_tile_owner::Variant::N8,
                arch,
            )?,
            "qwen4_mq6_x4" => qwen4_mq6_x4(epi.as_deref().unwrap_or("all"), arch)?,
            "qwen4_mq6_x4_gfx11" => qwen4_mq6_x4_gfx11(epi.as_deref().unwrap_or("all"), arch)?,
            "npu_trunk" => {
                if arch != Arch::Gfx1151 {
                    return Err("npu_trunk is gfx1151 only".into());
                }
                let (_, text, proof) = peacemaker_isa::npu_trunk::emit_module()?;
                (
                    text,
                    serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
                )
            }
            "npu_trunk_arena" => {
                if arch != Arch::Gfx1151 {
                    return Err("npu_trunk_arena is gfx1151 only".into());
                }
                let (_, text, proof) = peacemaker_isa::npu_trunk_arena::emit_module()?;
                (
                    text,
                    serde_json::to_vec_pretty(&proof).map_err(|e| e.to_string())?,
                )
            }
            "qwen4_lane_rows" => qwen4_lane_rows(epi.as_deref().unwrap_or("all"), arch)?,
            "qwen4_mq4_x4_gfx11" => qwen4_mq4_x4_gfx11(epi.as_deref().unwrap_or("all"), arch)?,
            "mq6_i8_gemm" => mq6_i8_gemm(epi.as_deref().unwrap_or("all"), tile.as_deref(), arch)?,
            "mq_i4_rows" => mq_i4_rows(epi.as_deref().unwrap_or("all"), arch)?,
            "mq_i4_a4" => mq_i4_a4(epi.as_deref().unwrap_or("all"), arch)?,
            "moe_rotate_i4_pm" => moe_rotate_i4_pm(epi.as_deref().unwrap_or("all"), arch)?,
            "mq_i4_moe_gemv_c1" => mq_i4_moe_gemv_c1(epi.as_deref().unwrap_or("all"), arch)?,
            "qwen4_shared_gu_gfx11" => {
                qwen4_shared_gu_gfx11(epi.as_deref().unwrap_or("all"), arch)?
            }
            "mtp_dense_det_gfx11" => mtp_dense_det_gfx11(epi.as_deref().unwrap_or("all"), arch)?,
            "mtp_q8_det_gfx11" => mtp_q8_det_gfx11(epi.as_deref().unwrap_or("all"), arch)?,
            "mq4_verify" => mq4_verify(epi.as_deref().unwrap_or("all"), arch)?,
            "fa_prefill_pm" => fa_prefill_pm(arch)?,
            "fa_prefill_fp8_pm" => fa_prefill_fp8_pm(arch)?,
            "fa_qprep_pm" => fa_qprep_pm(arch)?,
            "iu4_c8_producer_pm" => iu4_c8_producer_pm(arch)?,
            "gdn_chunk_prep_pm" => gdn_chunk_prep_pm(arch)?,
            "mq4_fp8_verify" => mq4_fp8_verify(epi.as_deref().unwrap_or("all"), arch)?,
            "fp8_gemm" => fp8_gemm(
                scale.as_deref().ok_or("missing --scale")?,
                epi.as_deref().ok_or("missing --epi")?,
                arch,
            )?,
            k if k.starts_with("pm_decode_") => pm_decode(&k["pm_decode_".len()..], arch)?,
            _ => return Err(format!("kernel {kernel} is not authored\n{USAGE}")),
        }
    };
    let pm_module = kernel.strip_prefix("pm_decode_");
    let mq4v = kernel == "mq4_verify";
    let fapm = kernel == "fa_prefill_pm";
    let fapm8 = kernel == "fa_prefill_fp8_pm";
    let fqp = kernel == "fa_qprep_pm";
    let iu4c8 = kernel == "iu4_c8_producer_pm";
    let gcp = kernel == "gdn_chunk_prep_pm";
    let mq4f8 = kernel == "mq4_fp8_verify";
    let mqi4 = kernel == "mq_i4_rows" || kernel == "mq_i4_moe_gemv_c1";
    let mqia4 = kernel == "mq_i4_a4";
    let mrot = kernel == "moe_rotate_i4_pm";
    let lane_rows = (kernel == "qwen4_lane_rows" && row_image.is_none())
        || kernel == "npu_trunk"
        || kernel == "npu_trunk_arena";
    let spec_pm = kernel == "spec_sampling_rows_pm"
        || kernel == "spec_accept_pm"
        || kernel == "qwen4_ep_owner_assign_pm"
        || kernel == "ep_tile_mask_pm"
        || kernel == "ep_gather_grouped_rows_by_tile_owner_pm"
        || kernel == "ep_gather_grouped_rows_by_tile_owner_n8_pm"
        || kernel == "moe_slot_ops_pm"
        || kernel == "moe_slot_ops_v2_pm";
    let mtp_q8 = kernel == "mtp_q8_det_gfx11";
    let mtp_det = kernel == "mtp_dense_det_gfx11" || mtp_q8;
    if pm_module.is_none()
        && !mq4v
        && !fapm
        && !fapm8
        && !fqp
        && !iu4c8
        && !gcp
        && !mq4f8
        && row_image.is_none()
        && !mqi4
        && !mqia4
        && !mrot
        && !lane_rows
        && !spec_pm
        && !mtp_det
        && (shape.is_some() || m7.is_some())
    {
        return Err("--shape/--m7 apply only to pm_decode_<module>, mq4_verify, fa_prefill_pm, fa_prefill_fp8_pm, fa_qprep_pm, iu4_c8_producer_pm, gdn_chunk_prep_pm, mq4_fp8_verify, mq_i4_rows, mq_i4_a4, moe_rotate_i4_pm, qwen4_lane_rows, spec_sampling_rows_pm, spec_accept_pm, qwen4_ep_owner_assign_pm, ep_tile_mask_pm, ep_gather_grouped_rows_by_tile_owner_pm, ep_gather_grouped_rows_by_tile_owner_n8_pm, moe_slot_ops_pm, moe_slot_ops_v2_pm, mtp_dense_det_gfx11 and mtp_q8_det_gfx11 (--m7 only)".into());
    }
    if spec_pm && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err("spec_*_pm --shape/--m7 require --co (M7 certifies the linked object)".into());
    }
    if lane_rows && shape.is_some() {
        return Err("qwen4_lane_rows has no --shape contract (use --m7)".into());
    }
    if lane_rows && co.is_none() {
        return Err(
            "qwen4_lane_rows emission requires --co (M7 certifies the linked object)".into(),
        );
    }
    if mtp_det && shape.is_some() {
        return Err(format!(
            "{kernel} has no --shape (its contract is committed in kernels/; use --m7)"
        ));
    }
    if mtp_det && co.is_none() {
        return Err(format!(
            "{kernel} emission requires --co (M7 certifies the linked object)"
        ));
    }
    if pm_module.is_some() && co.is_none() {
        return Err("pm_decode emission requires --co (M7 certifies the linked object)".into());
    }
    if mq4v && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err("mq4_verify --shape/--m7 require --co (M7 certifies the linked object)".into());
    }
    if fapm && epi.is_some() {
        return Err("fa_prefill_pm has one module and no --epi variants".into());
    }
    if fapm && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err(
            "fa_prefill_pm --shape/--m7 require --co (M7 certifies the linked object)".into(),
        );
    }
    if fapm8 && epi.is_some() {
        return Err("fa_prefill_fp8_pm has one module and no --epi variants".into());
    }
    if fapm8 && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err(
            "fa_prefill_fp8_pm --shape/--m7 require --co (M7 certifies the linked object)".into(),
        );
    }
    if iu4c8 && epi.is_some() {
        return Err("iu4_c8_producer_pm has one module and no --epi variants".into());
    }
    if iu4c8 && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err(
            "iu4_c8_producer_pm --shape/--m7 require --co (M7 certifies the linked object)".into(),
        );
    }
    if fqp && epi.is_some() {
        return Err("fa_qprep_pm has one module and no --epi variants".into());
    }
    if fqp && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err(
            "fa_qprep_pm --shape/--m7 require --co (M7 certifies the linked object)".into(),
        );
    }
    if gcp && epi.is_some() {
        return Err("gdn_chunk_prep_pm has one module and no --epi variants".into());
    }
    if gcp && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err(
            "gdn_chunk_prep_pm --shape/--m7 require --co (M7 certifies the linked object)".into(),
        );
    }
    if mq4f8 && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err(
            "mq4_fp8_verify --shape/--m7 require --co (M7 certifies the linked object)".into(),
        );
    }
    if mqi4 && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err(format!(
            "{kernel} --shape/--m7 require --co (M7 certifies the linked object)"
        ));
    }
    if mqia4 && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err("mq_i4_a4 --shape/--m7 require --co (M7 certifies the linked object)".into());
    }
    if mrot && co.is_none() && (shape.is_some() || m7.is_some()) {
        return Err(
            "moe_rotate_i4_pm --shape/--m7 require --co (M7 certifies the linked object)".into(),
        );
    }
    if row_image.is_some() && co.is_none() {
        return Err(format!(
            "{kernel} emission requires --co (M7 certifies every linked export)"
        ));
    }
    #[cfg(not(feature = "toolchain"))]
    if row_image.is_some() {
        return Err(format!(
            "{kernel} emission certifies with M7: build peacemaker-isa with --features toolchain"
        ));
    }
    // Native emission: the code object `llvm-mc` + `ld.lld -shared` would link, and its bundle.
    if co.is_some() || bundle.is_some() {
        let elf = peacemaker_isa::native::assemble(&text, arch)?;
        if let Some(path) = &co {
            fs::write(path, &elf).map_err(|e| format!("{path}: {e}"))?
        }
        if let Some(path) = &bundle {
            let bytes = peacemaker_isa::native::bundle(&elf, arch, &host);
            fs::write(path, &bytes).map_err(|e| format!("{path}: {e}"))?;
            let hash_path = format!("{path}.sha256");
            fs::write(&hash_path, format!("{:x}", Sha256::digest(&bytes)))
                .map_err(|e| format!("{hash_path}: {e}"))?;
        }
    }
    #[cfg(feature = "toolchain")]
    if let Some(image) = &row_image {
        let path = co.as_deref().ok_or("missing --co")?;
        let mut receipts = serde_json::Map::new();
        if let Some(dir) = &shape {
            fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
        }
        for (e, contract) in image.entries.iter().zip(&image.contracts) {
            let symbol = row_export(e)?;
            receipts.insert(
                symbol.to_owned(),
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), symbol)?,
            );
            if let Some(dir) = &shape {
                let file = format!(
                    "{dir}/{kernel}.{}.{}.contract.json",
                    arch.name(),
                    e.proof.variant
                );
                fs::write(
                    &file,
                    serde_json::to_vec_pretty(contract).map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("{file}: {e}"))?;
            }
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?;
        }
    }
    // PM decode twins: M7 (lift byte-exact, obligations {}, no ambiguous delays) on every export of the linked object.
    #[cfg(not(feature = "toolchain"))]
    if pm_module.is_some() {
        return Err(
            "pm_decode emission certifies with M7: build peacemaker-isa with --features toolchain"
                .into(),
        );
    }
    #[cfg(feature = "toolchain")]
    if let Some(module) = pm_module {
        let path = co.as_deref().ok_or("missing --co")?;
        let mut receipts = serde_json::Map::new();
        for &symbol in peacemaker_isa::kernels::pm_decode::module(module)?.symbols {
            receipts.insert(
                symbol.to_owned(),
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), symbol)?,
            );
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
        if let Some(file) = &shape {
            fs::write(
                file,
                serde_json::to_vec_pretty(&peacemaker_isa::kernels::pm_decode::shape_contract(
                    module,
                )?)
                .map_err(|e| e.to_string())?,
            )
            .map_err(|e| format!("{file}: {e}"))?
        }
    }
    // MTP append DET dense GEMM: M7 on the single export of the linked object.
    #[cfg(not(feature = "toolchain"))]
    if mtp_det {
        return Err(format!(
            "{kernel} emission certifies with M7: build peacemaker-isa with --features toolchain"
        ));
    }
    #[cfg(feature = "toolchain")]
    if mtp_det {
        let path = co.as_deref().ok_or("missing --co")?;
        let symbol = if mtp_q8 {
            peacemaker_isa::kernels::mtp_q8_det_gfx11::Spec { arch }.symbol()
        } else {
            peacemaker_isa::kernels::mtp_dense_det_gfx11::Spec { arch }.symbol()
        };
        let mut receipts = serde_json::Map::new();
        receipts.insert(
            symbol.clone(),
            peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?,
        );
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // Lane-step row kernels: M7 on every export of the linked object.
    #[cfg(not(feature = "toolchain"))]
    if lane_rows {
        return Err("qwen4_lane_rows emission certifies with M7: build peacemaker-isa with --features toolchain".into());
    }
    #[cfg(feature = "toolchain")]
    if lane_rows {
        let path = co.as_deref().ok_or("missing --co")?;
        let mut receipts = serde_json::Map::new();
        let symbols: &[&str] = if kernel == "npu_trunk" {
            &peacemaker_isa::npu_trunk::SYMBOLS
        } else if kernel == "npu_trunk_arena" {
            &peacemaker_isa::npu_trunk_arena::SYMBOLS
        } else {
            &peacemaker_isa::kernels::qwen4_lane_rows::SYMBOLS
        };
        for &symbol in symbols {
            receipts.insert(
                symbol.to_owned(),
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), symbol)?,
            );
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // PM sampling kernels (`spec_sampling_rows_pm`, `spec_accept_pm`): with --co, M7 on the symbol of the linked object
    // (toolchain builds), plus its shape contract (--shape FILE) and receipt (--m7 FILE).
    #[cfg(not(feature = "toolchain"))]
    if spec_pm && (shape.is_some() || m7.is_some()) {
        return Err("spec_*_pm --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain".into());
    }
    #[cfg(feature = "toolchain")]
    if spec_pm && co.is_some() {
        let path = co.as_deref().ok_or("missing --co")?;
        let (symbol, contract) = if kernel == "spec_sampling_rows_pm" {
            let e = peacemaker_isa::kernels::spec_sampling_rows_pm::emit(arch)?;
            (
                peacemaker_isa::kernels::spec_sampling_rows_pm::symbol(arch),
                peacemaker_isa::kernels::spec_sampling_rows_pm::contract(arch, &e),
            )
        } else if kernel == "qwen4_ep_owner_assign_pm" {
            let e = peacemaker_isa::kernels::qwen4_ep_owner_assign_pm::emit(arch)?;
            (
                peacemaker_isa::kernels::qwen4_ep_owner_assign_pm::symbol(arch),
                peacemaker_isa::kernels::qwen4_ep_owner_assign_pm::contract(arch, &e),
            )
        } else if kernel == "ep_tile_mask_pm" {
            let e = peacemaker_isa::kernels::ep_tile_mask::emit(arch)?;
            (
                peacemaker_isa::kernels::ep_tile_mask::symbol(arch),
                peacemaker_isa::kernels::ep_tile_mask::contract(arch, &e),
            )
        } else if kernel == "moe_slot_ops_pm" {
            let e = peacemaker_isa::kernels::moe_slot_ops::emit(arch)?;
            (
                peacemaker_isa::kernels::moe_slot_ops::symbol(arch),
                peacemaker_isa::kernels::moe_slot_ops::contract(arch, &e),
            )
        } else if kernel == "moe_slot_ops_v2_pm" {
            let e = peacemaker_isa::kernels::moe_slot_ops::emit_v2(arch)?;
            (
                peacemaker_isa::kernels::moe_slot_ops::symbol_v2(arch),
                peacemaker_isa::kernels::moe_slot_ops::contract_v2(arch, &e),
            )
        } else if kernel == "ep_gather_grouped_rows_by_tile_owner_pm"
            || kernel == "ep_gather_grouped_rows_by_tile_owner_n8_pm"
        {
            use peacemaker_isa::kernels::ep_gather_grouped_rows_by_tile_owner as pull;
            let variant = if kernel == "ep_gather_grouped_rows_by_tile_owner_pm" {
                pull::Variant::Ep2
            } else {
                pull::Variant::N8
            };
            let e = pull::emit(variant, arch)?;
            (
                pull::symbol(variant, arch),
                pull::contract(variant, arch, &e),
            )
        } else {
            let e = peacemaker_isa::kernels::spec_accept_pm::emit(arch)?;
            (
                peacemaker_isa::kernels::spec_accept_pm::symbol(arch),
                peacemaker_isa::kernels::spec_accept_pm::contract(arch, &e),
            )
        };
        let receipt =
            peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
        let mut receipts = serde_json::Map::new();
        receipts.insert(symbol, receipt);
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
        if let Some(file) = &shape {
            fs::write(
                file,
                serde_json::to_vec_pretty(&contract).map_err(|e| e.to_string())?,
            )
            .map_err(|e| format!("{file}: {e}"))?
        }
    }
    // mq4_verify twins: with --co, M7 on every export of the linked object
    // (toolchain builds), plus each export's shape contract (--shape DIR).
    #[cfg(not(feature = "toolchain"))]
    if mq4v && (shape.is_some() || m7.is_some()) {
        return Err("mq4_verify --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain".into());
    }
    #[cfg(not(feature = "toolchain"))]
    if fapm && (shape.is_some() || m7.is_some()) {
        return Err("fa_prefill_pm --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain".into());
    }
    #[cfg(not(feature = "toolchain"))]
    if fapm8 && (shape.is_some() || m7.is_some()) {
        return Err("fa_prefill_fp8_pm --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain".into());
    }
    #[cfg(not(feature = "toolchain"))]
    if iu4c8 && (shape.is_some() || m7.is_some()) {
        return Err("iu4_c8_producer_pm --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain".into());
    }
    #[cfg(not(feature = "toolchain"))]
    if fqp && (shape.is_some() || m7.is_some()) {
        return Err("fa_qprep_pm --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain".into());
    }
    #[cfg(not(feature = "toolchain"))]
    if gcp && (shape.is_some() || m7.is_some()) {
        return Err("gdn_chunk_prep_pm --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain".into());
    }
    #[cfg(not(feature = "toolchain"))]
    if mq4f8 && (shape.is_some() || m7.is_some()) {
        return Err("mq4_fp8_verify --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain".into());
    }
    #[cfg(not(feature = "toolchain"))]
    if mqi4 && (shape.is_some() || m7.is_some()) {
        return Err(format!(
            "{kernel} --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain"
        ));
    }
    #[cfg(not(feature = "toolchain"))]
    if mqia4 && (shape.is_some() || m7.is_some()) {
        return Err(
            "mq_i4_a4 --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain"
                .into(),
        );
    }
    #[cfg(not(feature = "toolchain"))]
    if mrot && (shape.is_some() || m7.is_some()) {
        return Err(
            "moe_rotate_i4_pm --shape/--m7 certify with M7: build peacemaker-isa with --features toolchain"
                .into(),
        );
    }
    #[cfg(feature = "toolchain")]
    if mq4v && co.is_some() {
        let path = co.as_deref().ok_or("missing --co")?;
        let mut receipts = serde_json::Map::new();
        for spec in mq4_verify_specs(epi.as_deref().unwrap_or("all"), arch)? {
            let symbol = spec.symbol();
            let m7 =
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
            if let Some(dir) = &shape {
                let e = peacemaker_isa::kernels::mq4_verify::emit(spec)?;
                let file = format!(
                    "{dir}/mq4_verify.{}.{}.contract.json",
                    arch.name(),
                    spec.tag()
                );
                fs::write(
                    &file,
                    serde_json::to_vec_pretty(&peacemaker_isa::kernels::mq4_verify::contract(
                        spec, &e,
                    ))
                    .map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("{file}: {e}"))?;
            }
            receipts.insert(symbol, m7);
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // mq_i4_rows: with --co, M7 on every selected export of the linked object, plus each
    // export's shape contract (--shape DIR).
    #[cfg(feature = "toolchain")]
    if kernel == "mq_i4_rows" && co.is_some() {
        let path = co.as_deref().ok_or("missing --co")?;
        let mut receipts = serde_json::Map::new();
        for spec in mq_i4_rows_specs(epi.as_deref().unwrap_or("all"), arch)? {
            let symbol = spec.symbol();
            let m7 =
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
            if let Some(dir) = &shape {
                let e = peacemaker_isa::kernels::mq_i4_rows::emit(spec)?;
                let file = format!(
                    "{dir}/mq_i4_rows.{}.{}.contract.json",
                    arch.name(),
                    spec.tag()
                );
                fs::write(
                    &file,
                    serde_json::to_vec_pretty(&peacemaker_isa::kernels::mq_i4_rows::contract(
                        spec, &e,
                    ))
                    .map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("{file}: {e}"))?;
            }
            receipts.insert(symbol, m7);
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // mq_i4_a4: with --co, M7 on the linked object's export, plus its shape contract (--shape DIR).
    #[cfg(feature = "toolchain")]
    if mqia4 && co.is_some() {
        let path = co.as_deref().ok_or("missing --co")?;
        let mut receipts = serde_json::Map::new();
        for spec in peacemaker_isa::kernels::mq_i4_a4::module_specs(arch) {
            let symbol = spec.symbol();
            let m7 =
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
            if let Some(dir) = &shape {
                let e = peacemaker_isa::kernels::mq_i4_a4::emit(spec)?;
                let file = format!(
                    "{dir}/mq_i4_a4.{}.{}.contract.json",
                    arch.name(),
                    spec.tag()
                );
                fs::write(
                    &file,
                    serde_json::to_vec_pretty(&peacemaker_isa::kernels::mq_i4_a4::contract(
                        spec, &e,
                    ))
                    .map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("{file}: {e}"))?;
            }
            receipts.insert(symbol, m7);
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // moe_rotate_i4_pm: with --co, M7 on the linked object's export, plus its shape contract (--shape DIR).
    #[cfg(feature = "toolchain")]
    if mrot && co.is_some() {
        let path = co.as_deref().ok_or("missing --co")?;
        let mut receipts = serde_json::Map::new();
        for spec in peacemaker_isa::kernels::moe_rotate_i4_pm::module_specs(arch) {
            let symbol = spec.symbol();
            let m7 =
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
            if let Some(dir) = &shape {
                let e = peacemaker_isa::kernels::moe_rotate_i4_pm::emit(spec)?;
                let file = format!("{dir}/moe_rotate_i4_pm.{}.contract.json", arch.name());
                fs::write(
                    &file,
                    serde_json::to_vec_pretty(
                        &peacemaker_isa::kernels::moe_rotate_i4_pm::contract(spec, &e),
                    )
                    .map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("{file}: {e}"))?;
            }
            receipts.insert(symbol, m7);
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // mq_i4_moe_gemv_c1: M7 on the linked object's two exports, plus their shape contracts (--shape DIR).
    #[cfg(feature = "toolchain")]
    if kernel == "mq_i4_moe_gemv_c1" && co.is_some() {
        let path = co.as_deref().ok_or("missing --co")?;
        let mut receipts = serde_json::Map::new();
        for spec in mq_i4_moe_gemv_c1_specs(epi.as_deref().unwrap_or("all"), arch)? {
            let symbol = spec.symbol();
            let m7 =
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
            if let Some(dir) = &shape {
                let e = peacemaker_isa::kernels::mq_i4_moe_gemv_c1::emit(spec)?;
                let file = format!(
                    "{dir}/mq_i4_moe_gemv_c1.{}.{}.contract.json",
                    arch.name(),
                    spec.tag()
                );
                fs::write(
                    &file,
                    serde_json::to_vec_pretty(
                        &peacemaker_isa::kernels::mq_i4_moe_gemv_c1::contract(spec, &e),
                    )
                    .map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("{file}: {e}"))?;
            }
            receipts.insert(symbol, m7);
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // fa_prefill_pm: M7 on both exports of the linked module (preconvert, direct), plus both contracts (--shape DIR).
    #[cfg(feature = "toolchain")]
    if fapm && co.is_some() {
        use peacemaker_isa::kernels::fa_prefill;
        let path = co.as_deref().ok_or("missing --co")?;
        let (emitted, _, _) = fa_prefill::emit_module(arch, &fa_prefill_spec())?;
        let [pre, direct] = &emitted[..] else {
            return Err(
                "fa_prefill_pm: module must hold exactly the preconvert then the direct export"
                    .into(),
            );
        };
        let mut receipts = serde_json::Map::new();
        for (name, contract) in [
            ("preconvert", fa_prefill::preconvert_contract(arch, pre)),
            ("direct", fa_prefill::contract(arch, direct)),
        ] {
            let symbol = contract["symbol"]
                .as_str()
                .ok_or("fa_prefill_pm: contract without a symbol")?
                .to_owned();
            let m7 =
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
            if let Some(dir) = &shape {
                let file = format!("{dir}/fa_prefill_pm.{}.{name}.contract.json", arch.name());
                fs::write(
                    &file,
                    serde_json::to_vec_pretty(&contract).map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("{file}: {e}"))?;
            }
            receipts.insert(symbol, m7);
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // fa_qprep_pm: M7 on the single export of the linked module, plus its contract (--shape DIR).
    #[cfg(feature = "toolchain")]
    if fqp && co.is_some() {
        use peacemaker_isa::kernels::fa_qprep;
        let path = co.as_deref().ok_or("missing --co")?;
        let (emitted, _, _) = fa_qprep::emit_module(arch, &fa_prefill_spec())?;
        let [qprep] = &emitted[..] else {
            return Err("fa_qprep_pm: module must hold exactly the one qprep export".into());
        };
        let contract = fa_qprep::contract(arch, qprep);
        let symbol = contract["symbol"]
            .as_str()
            .ok_or("fa_qprep_pm: contract without a symbol")?
            .to_owned();
        let m7r = peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
        if let Some(dir) = &shape {
            let file = format!("{dir}/fa_qprep_pm.{}.contract.json", arch.name());
            fs::write(
                &file,
                serde_json::to_vec_pretty(&contract).map_err(|e| e.to_string())?,
            )
            .map_err(|e| format!("{file}: {e}"))?;
        }
        let mut receipts = serde_json::Map::new();
        receipts.insert(symbol, m7r);
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // iu4_c8_producer_pm: M7 on both exports of the linked module (rotate, norm), plus both contracts (--shape DIR).
    #[cfg(feature = "toolchain")]
    if iu4c8 && co.is_some() {
        use peacemaker_isa::kernels::iu4_c8_producer;
        let path = co.as_deref().ok_or("missing --co")?;
        let (emitted, _, _) = iu4_c8_producer::emit_module(arch)?;
        let mut receipts = serde_json::Map::new();
        for (export, e) in iu4_c8_producer::module_exports().into_iter().zip(&emitted) {
            let contract = iu4_c8_producer::contract(arch, e, export);
            let symbol = contract["symbol"]
                .as_str()
                .ok_or("iu4_c8_producer_pm: contract without a symbol")?
                .to_owned();
            let m7r =
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
            if let Some(dir) = &shape {
                let file = format!(
                    "{dir}/iu4_c8_producer_pm.{}.{}.contract.json",
                    arch.name(),
                    export.tag()
                );
                fs::write(
                    &file,
                    serde_json::to_vec_pretty(&contract).map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("{file}: {e}"))?;
            }
            receipts.insert(symbol, m7r);
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // fa_prefill_fp8_pm: M7 on the single direct export of the linked module, plus its contract (--shape DIR).
    #[cfg(feature = "toolchain")]
    if fapm8 && co.is_some() {
        use peacemaker_isa::kernels::fa_prefill;
        let path = co.as_deref().ok_or("missing --co")?;
        let (emitted, _, _) = fa_prefill::emit_fp8_module(arch, &fa_prefill_fp8_spec())?;
        let [direct] = &emitted[..] else {
            return Err("fa_prefill_fp8_pm: module must hold exactly the one direct export".into());
        };
        let contract = fa_prefill::fp8_contract(arch, direct);
        let symbol = contract["symbol"]
            .as_str()
            .ok_or("fa_prefill_fp8_pm: contract without a symbol")?
            .to_owned();
        let m7r = peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
        if let Some(dir) = &shape {
            let file = format!(
                "{dir}/fa_prefill_fp8_pm.{}.direct.contract.json",
                arch.name()
            );
            fs::write(
                &file,
                serde_json::to_vec_pretty(&contract).map_err(|e| e.to_string())?,
            )
            .map_err(|e| format!("{file}: {e}"))?;
        }
        let mut receipts = serde_json::Map::new();
        receipts.insert(symbol, m7r);
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // gdn_chunk_prep_pm: M7 on the single export of the linked module, plus its contract (--shape DIR).
    #[cfg(feature = "toolchain")]
    if gcp && co.is_some() {
        use peacemaker_isa::kernels::gdn_chunk_prep;
        let path = co.as_deref().ok_or("missing --co")?;
        let (emitted, _, _) =
            gdn_chunk_prep::emit_module(arch, &gdn_chunk_prep::GdnChunkPrepSpec::A3B)?;
        let [prep] = &emitted[..] else {
            return Err(
                "gdn_chunk_prep_pm: module must hold exactly the one chunk-prep export".into(),
            );
        };
        let contract = gdn_chunk_prep::contract(arch, prep);
        let symbol = contract["symbol"]
            .as_str()
            .ok_or("gdn_chunk_prep_pm: contract without a symbol")?
            .to_owned();
        let m7r = peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
        if let Some(dir) = &shape {
            let file = format!("{dir}/gdn_chunk_prep_pm.{}.contract.json", arch.name());
            fs::write(
                &file,
                serde_json::to_vec_pretty(&contract).map_err(|e| e.to_string())?,
            )
            .map_err(|e| format!("{file}: {e}"))?;
        }
        let mut receipts = serde_json::Map::new();
        receipts.insert(symbol, m7r);
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    // mq4_fp8_verify: the same, for the FP8 verify entries.
    #[cfg(feature = "toolchain")]
    if mq4f8 && co.is_some() {
        let path = co.as_deref().ok_or("missing --co")?;
        let mut receipts = serde_json::Map::new();
        for spec in mq4_fp8_verify_specs(epi.as_deref().unwrap_or("all"), arch)? {
            let symbol = spec.symbol();
            let m7 =
                peacemaker_isa::pm_check::m7(std::path::Path::new(path), arch.name(), &symbol)?;
            if let Some(dir) = &shape {
                let e = peacemaker_isa::kernels::mq4_fp8_verify::emit(spec)?;
                let file = format!(
                    "{dir}/mq4_fp8_verify.{}.{}.contract.json",
                    arch.name(),
                    spec.tag()
                );
                fs::write(
                    &file,
                    serde_json::to_vec_pretty(&peacemaker_isa::kernels::mq4_fp8_verify::contract(
                        spec, &e,
                    ))
                    .map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("{file}: {e}"))?;
            }
            receipts.insert(symbol, m7);
        }
        let receipts = serde_json::to_vec_pretty(&receipts).map_err(|e| e.to_string())?;
        println!("{}", String::from_utf8_lossy(&receipts));
        if let Some(file) = &m7 {
            fs::write(file, &receipts).map_err(|e| format!("{file}: {e}"))?
        }
    }
    fs::write(out.ok_or("missing --out")?, text).map_err(|e| e.to_string())?;
    fs::write(proof.ok_or("missing --proof")?, proof_json).map_err(|e| e.to_string())?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("peacemaker-isa: {e}");
        std::process::exit(1)
    }
}
