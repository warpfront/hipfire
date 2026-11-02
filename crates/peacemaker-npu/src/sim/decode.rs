// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Timing and instruction fields derive from Xilinx/llvm-aie TableGen, licensed
// Apache-2.0 WITH LLVM-exception.
//! Prepare generated AIE2P bundles and itinerary operand cycles once. Execution
//! performs no allocation or TableGen scans, and checks public program memory
//! against the prepared bytes. Unsupported operations remain explicit errors.
use crate::isa::{
    decode::{self, DecodedInst},
    sched, Reg,
};
use crate::sim::{
    AccSource, Compare, DecodedOps, Decoder, Error, Op, Operations, Result, ScalarAlu, Special,
    SrsAddressing, TimedOp, UnpackSource, VectorAccess, VectorAlu, V512,
};
struct PreparedBundle {
    pc: usize,
    len: usize,
    raw: [u8; 16],
    start: usize,
    count: usize,
}
pub struct PreparedDecoder {
    bundles: Vec<PreparedBundle>,
    operations: Vec<TimedOp>,
}
impl PreparedDecoder {
    pub fn new(bytes: &[u8]) -> Result<Self> {
        let mut bundles = Vec::new();
        let mut operations = Vec::new();
        let mut pc = 0;
        while pc < bytes.len() {
            let bundle = decode::decode(&bytes[pc..], pc as u64).map_err(|_| Error::Decode(pc))?;
            let start = operations.len();
            for instruction in &bundle.instructions {
                operations.push(timed(instruction)?);
            }
            let mut raw = [0; 16];
            raw[..bundle.len].copy_from_slice(&bytes[pc..pc + bundle.len]);
            bundles.push(PreparedBundle {
                pc,
                len: bundle.len,
                raw,
                start,
                count: operations.len() - start,
            });
            pc += bundle.len;
        }
        Ok(Self {
            bundles,
            operations,
        })
    }
    fn prepared(&self, bytes: &[u8], pc: usize) -> Result<(&PreparedBundle, &[TimedOp])> {
        let index = self
            .bundles
            .binary_search_by_key(&pc, |b| b.pc)
            .map_err(|_| Error::Decode(pc))?;
        let bundle = &self.bundles[index];
        if bytes.get(pc..pc + bundle.len) != Some(&bundle.raw[..bundle.len]) {
            return Err(Error::Decode(pc));
        }
        Ok((
            bundle,
            &self.operations[bundle.start..bundle.start + bundle.count],
        ))
    }
}
impl Decoder for PreparedDecoder {
    fn decode(&self, bytes: &[u8], pc: usize) -> Result<(Op, usize)> {
        let (bundle, ops) = self.prepared(bytes, pc)?;
        if ops.len() != 1 {
            return Err(Error::Unsupported);
        }
        Ok((ops[0].op, bundle.len))
    }
    fn bundle(&self, bytes: &[u8], pc: usize) -> Result<DecodedOps<'_>> {
        let (bundle, operations) = self.prepared(bytes, pc)?;
        Ok(DecodedOps {
            operations: Operations::Prepared(operations),
            len: bundle.len,
        })
    }
}
fn value(i: &DecodedInst, name: &str) -> Result<i64> {
    i.operands
        .iter()
        .find(|o| o.name == name)
        .map(|o| o.value)
        .ok_or(Error::Unsupported)
}
fn index(i: &DecodedInst, name: &str) -> Result<u8> {
    value(i, name)?.try_into().map_err(|_| Error::Unsupported)
}
fn scalar(i: &DecodedInst, name: &str) -> Result<Reg> {
    let v = value(i, name)? as u64;
    let description = i
        .encoding
        .operands
        .iter()
        .find(|o| o.name == name)
        .ok_or(Error::Unsupported)?;
    let reg = decode::resolve_register(description, v)
        .ok_or(Error::Unsupported)?
        .name;
    if reg == "sp" {
        return Ok(Reg::Sp);
    }
    if reg == "lr" {
        return Ok(Reg::Lr);
    }
    for (prefix, constructor) in [
        ("dn", Reg::Dn as fn(u8) -> Reg),
        ("dj", Reg::Dj),
        ("dc", Reg::Dc),
        ("r", Reg::R),
        ("p", Reg::P),
        ("m", Reg::M),
    ] {
        if let Some(n) = reg.strip_prefix(prefix).and_then(|s| s.parse::<u8>().ok()) {
            return Ok(constructor(n));
        }
    }
    Err(Error::Unsupported)
}
/// `scalar` restricted to r registers (including the `eRS16` mask registers r16..r31).
fn r_index(i: &DecodedInst, name: &str) -> Result<u8> {
    match scalar(i, name)? {
        Reg::R(n) => Ok(n),
        _ => Err(Error::Unsupported),
    }
}
/// dm operand (`eDM`: raw = dm number) or, for `cm` operands (`OP_mCMs`/`OP_mCMm`), `cml_k` = 2k / `cmh_k` = 2k+1
/// (gen.rs `REGS_6`): half `raw%2` of dm `raw/2`.
fn acc_source(i: &DecodedInst, name: &str, cm: bool) -> Result<AccSource> {
    let raw = index(i, name)?;
    Ok(if cm {
        AccSource::Cm(raw / 2, raw % 2)
    } else {
        AccSource::Dm(raw)
    })
}
/// `VMOV_alu_mv_mv_x` operand: `x<n>`, or bm quarter `bm{ll,lh,hl,hh}<k>` = 4k + (0,1,2,3).
fn v512(i: &DecodedInst, name: &str) -> Result<V512> {
    let v = value(i, name)? as u64;
    let description = i
        .encoding
        .operands
        .iter()
        .find(|o| o.name == name)
        .ok_or(Error::Unsupported)?;
    let reg = decode::resolve_register(description, v)
        .ok_or(Error::Unsupported)?
        .name;
    if let Some(n) = reg.strip_prefix('x').and_then(|s| s.parse::<u8>().ok()) {
        return Ok(V512::X(n));
    }
    for (prefix, q) in [("bmll", 0), ("bmlh", 1), ("bmhl", 2), ("bmhh", 3)] {
        if let Some(k) = reg.strip_prefix(prefix).and_then(|s| s.parse::<u8>().ok()) {
            return Ok(V512::Bm(4 * k + q));
        }
    }
    Err(Error::Unsupported)
}
fn shift_register(name: &str) -> Option<Special> {
    match name {
        "s0" => Some(Special::S(0)),
        "s1" => Some(Special::S(1)),
        "s2" => Some(Special::S(2)),
        "s3" => Some(Special::S(3)),
        _ => None,
    }
}
fn operation(i: &DecodedInst) -> Result<Op> {
    let imm = |name| value(i, name).map(|v| v as i32);
    Ok(match i.encoding.name {
        "NOP" | "NOPA" | "NOPB" | "NOPM" | "NOPS" | "NOPX" | "NOPV" | "NOPXM" => Op::Nop,
        "DONE" => Op::Done,
        "MOVXM" | "MOVA" | "MOV_alu_mv_mv_mv_cg" => {
            let dst = decode::resolve_register(&i.encoding.operands[0], value(i, "dst")? as u64)
                .ok_or(Error::Unsupported)?
                .name;
            let v = value(i, "i")? as i32 as u32;
            if let Some(n) = ["ls", "le", "lc"].iter().position(|&name| name == dst) {
                Op::LoopRegister(n as u8, v)
            } else if let Some(sp) = shift_register(dst) {
                Op::MoveSpecial(sp, v)
            } else {
                Op::Move(scalar(i, "dst")?, v)
            }
        }
        "MOV_alu_mv_mv_mv_scl" => {
            let dst = decode::resolve_register(&i.encoding.operands[0], value(i, "dst")? as u64)
                .ok_or(Error::Unsupported)?
                .name;
            match (shift_register(dst), scalar(i, "src")?) {
                (Some(sp), Reg::R(r)) => Op::MoveSpecialRegister(sp, r),
                _ => return Err(Error::Unsupported),
            }
        }
        "MOVX_mvx_cr_imm" | "MOVX_mvx_cr_r" => {
            let dst = decode::resolve_register(&i.encoding.operands[0], value(i, "dst")? as u64)
                .ok_or(Error::Unsupported)?
                .name;
            let sp = match dst {
                "crSat" => Special::CrSat,
                "crRnd" => Special::CrRnd,
                "crSRSMode" => Special::CrSrsMode,
                "crUPSMode" => Special::CrUpsMode,
                "crUnpackSize" => Special::CrUnpackSize,
                _ => return Err(Error::Unsupported),
            };
            if i.encoding.name == "MOVX_mvx_cr_imm" {
                Op::MoveSpecial(sp, value(i, "src")? as u32)
            } else {
                Op::MoveSpecialRegister(sp, index(i, "src")?)
            }
        }
        name if (name.starts_with("VST_SRS_4x_dmx_sts_srs_dm_")
            || name.starts_with("VST_SRS_2x_dm_sts_srs_cm_"))
            && (name.ends_with("_srsSign0") || name.ends_with("_srsSign1")) =>
        {
            let cm = name.starts_with("VST_SRS_2x");
            let prefix = if cm {
                "VST_SRS_2x_dm_sts_srs_cm_"
            } else {
                "VST_SRS_4x_dmx_sts_srs_dm_"
            };
            let addressing = match &name[prefix.len()..name.len() - "_srsSign0".len()] {
                "pstm_nrm_imm" => SrsAddressing::PostImmediate(imm("imm")?),
                "idx_imm" => SrsAddressing::Immediate(imm("imm")?),
                "pstm_nrm" => SrsAddressing::PostModifier(index(i, "mod")?),
                _ => return Err(Error::Unsupported),
            };
            Op::ShiftRoundSaturateStore {
                src: acc_source(i, "src", cm)?,
                shift: index(i, "su")?,
                ptr: index(i, "ptr")?,
                addressing,
                signed: name.ends_with("srsSign1"),
            }
        }
        "VSRS_4x_mv_x_srs_dm_srsSign1" | "VSRS_2x_mv_x_srs_cm_srsSign1" => Op::ShiftRoundSaturate {
            dst: index(i, "dst")?,
            src: acc_source(i, "src", i.encoding.name.starts_with("VSRS_2x"))?,
            shift: index(i, "su")?,
        },
        "VUPS_4x_mv_ups_x2d_upsSign0"
        | "VUPS_4x_mv_ups_x2d_upsSign1"
        | "VUPS_2x_mv_ups_x2c_upsSign0"
        | "VUPS_2x_mv_ups_x2c_upsSign1" => Op::Ups {
            dst: acc_source(i, "dst", i.encoding.name.starts_with("VUPS_2x"))?,
            src: index(i, "src")?,
            shift: index(i, "su")?,
            signed: i.encoding.name.ends_with("Sign1"),
        },
        "VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign0"
        | "VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign1" => Op::Unpack {
            dst: index(i, "dst")?,
            src: UnpackSource::Memory {
                ptr: index(i, "ptr")?,
                step: imm("imm")?,
            },
            signed: i.encoding.name.ends_with("Sign1"),
        },
        "VUNPACK_mv_unpack_x_unpackSign0" | "VUNPACK_mv_unpack_x_unpackSign1" => Op::Unpack {
            dst: index(i, "dst")?,
            src: UnpackSource::Register(index(i, "src")?),
            signed: i.encoding.name.ends_with("Sign1"),
        },
        "VST_dmx_sts_x_pstm_nrm_imm" => Op::VectorMemory {
            access: VectorAccess::StoreX,
            reg: index(i, "src")?,
            ptr: index(i, "ptr")?,
            addressing: SrsAddressing::PostImmediate(imm("imm")?),
        },
        "VST_dmx_sts_x_pstm_nrm" => Op::VectorMemory {
            access: VectorAccess::StoreX,
            reg: index(i, "src")?,
            ptr: index(i, "ptr")?,
            addressing: SrsAddressing::PostModifier(index(i, "mod")?),
        },
        "VST_dmx_sts_x_idx_imm" => Op::VectorMemory {
            access: VectorAccess::StoreX,
            reg: index(i, "src")?,
            ptr: index(i, "ptr")?,
            addressing: SrsAddressing::Immediate(imm("imm")?),
        },
        "VST_dmx_sts_bm_pstm_nrm" => Op::VectorMemory {
            access: VectorAccess::StoreBm,
            reg: index(i, "src")?,
            ptr: index(i, "ptr")?,
            addressing: SrsAddressing::PostModifier(index(i, "mod")?),
        },
        "VST_dmx_sts_bm_idx_imm" => Op::VectorMemory {
            access: VectorAccess::StoreBm,
            reg: index(i, "src")?,
            ptr: index(i, "ptr")?,
            addressing: SrsAddressing::Immediate(imm("imm")?),
        },
        "VLDA_dmx_lda_x_idx_imm" | "VLDB_dmx_ldb_x_idx_imm" => Op::VectorMemory {
            access: VectorAccess::LoadX,
            reg: index(i, "dst")?,
            ptr: index(i, "ptr")?,
            addressing: SrsAddressing::Immediate(imm("imm")?),
        },
        "VLDA_dmx_lda_x_pstm_nrm" => Op::VectorMemory {
            access: VectorAccess::LoadX,
            reg: index(i, "dst")?,
            ptr: index(i, "ptr")?,
            addressing: SrsAddressing::PostModifier(index(i, "mod")?),
        },
        "VLDA_dmx_lda_bm_idx_imm" => Op::VectorMemory {
            access: VectorAccess::LoadBm,
            reg: index(i, "dst")?,
            ptr: index(i, "ptr")?,
            addressing: SrsAddressing::Immediate(imm("imm")?),
        },
        "VLDA_dmx_lda_bm_pstm_nrm" => Op::VectorMemory {
            access: VectorAccess::LoadBm,
            reg: index(i, "dst")?,
            ptr: index(i, "ptr")?,
            addressing: SrsAddressing::PostModifier(index(i, "mod")?),
        },
        "VNEG" => Op::Negate {
            dst: index(i, "dst")?,
            src: index(i, "acc1")?,
            config: index(i, "acc")?,
        },
        "VSHUFFLE_vec_shuffle_x" => Op::Shuffle {
            dst: index(i, "dst")?,
            a: index(i, "s1")?,
            b: index(i, "s2")?,
            mode: index(i, "mod")?,
        },
        "VADD_16" => Op::VectorAlu {
            kind: VectorAlu::Add16,
            dst: index(i, "d")?,
            a: index(i, "s1")?,
            b: index(i, "s2")?,
        },
        "VADD_32" => Op::VectorAlu {
            kind: VectorAlu::Add32,
            dst: index(i, "d")?,
            a: index(i, "s1")?,
            b: index(i, "s2")?,
        },
        "VSUB_16" => Op::VectorAlu {
            kind: VectorAlu::Sub16,
            dst: index(i, "d")?,
            a: index(i, "s1")?,
            b: index(i, "s2")?,
        },
        "VSUB_32" => Op::VectorAlu {
            kind: VectorAlu::Sub32,
            dst: index(i, "d")?,
            a: index(i, "s1")?,
            b: index(i, "s2")?,
        },
        "VBAND" => Op::VectorAlu {
            kind: VectorAlu::And,
            dst: index(i, "d")?,
            a: index(i, "s1")?,
            b: index(i, "s2")?,
        },
        "VBOR" => Op::VectorAlu {
            kind: VectorAlu::Or,
            dst: index(i, "d")?,
            a: index(i, "s1")?,
            b: index(i, "s2")?,
        },
        "VEQZ_16" | "VEQZ_32" => Op::Compare {
            kind: Compare::Eqz,
            wide: i.encoding.name.ends_with("32"),
            signed: false,
            mask: r_index(i, "cmp")?,
            a: 0,
            b: index(i, "s2")?,
        },
        "VLT_16_vaddSign0" | "VLT_16_vaddSign1" | "VLT_32_vaddSign0" | "VLT_32_vaddSign1"
        | "VGE_16_vaddSign0" | "VGE_16_vaddSign1" | "VGE_32_vaddSign0" | "VGE_32_vaddSign1" => {
            let name = i.encoding.name;
            Op::Compare {
                kind: if name.starts_with("VLT") {
                    Compare::Lt
                } else {
                    Compare::Ge
                },
                wide: name.contains("_32_"),
                signed: name.ends_with("Sign1"),
                mask: r_index(i, "cmp")?,
                a: index(i, "s1")?,
                b: index(i, "s2")?,
            }
        }
        "VMAX_LT_16_vaddSign0"
        | "VMAX_LT_16_vaddSign1"
        | "VMAX_LT_32_vaddSign0"
        | "VMAX_LT_32_vaddSign1"
        | "VMIN_GE_16_vaddSign0"
        | "VMIN_GE_16_vaddSign1"
        | "VMIN_GE_32_vaddSign0"
        | "VMIN_GE_32_vaddSign1" => {
            let name = i.encoding.name;
            // the implicit `cmp` operand is r16 (`mR16_vcompare`)
            if r_index(i, "cmp")? != 16 {
                return Err(Error::Unsupported);
            }
            Op::MinMax {
                max: name.starts_with("VMAX"),
                wide: name.contains("_32_"),
                signed: name.ends_with("Sign1"),
                dst: index(i, "d")?,
                a: index(i, "s1")?,
                b: index(i, "s2")?,
            }
        }
        "VSEL_16" | "VSEL_32" => Op::Select {
            wide: i.encoding.name.ends_with("32"),
            dst: index(i, "d")?,
            a: index(i, "s1")?,
            b: index(i, "s2")?,
            mask: r_index(i, "sel")?,
        },
        "VBCST_16" | "VBCST_32" => Op::Broadcast {
            wide: i.encoding.name.ends_with("32"),
            dst: index(i, "dst")?,
            src: index(i, "src")?,
        },
        "VMOV_alu_mv_mv_x" => Op::VectorMove {
            dst: v512(i, "dst")?,
            src: v512(i, "src")?,
        },
        "AND" | "OR" | "ADD_alu_r_rr" | "LSHL" => Op::Scalar {
            kind: match i.encoding.name {
                "AND" => ScalarAlu::And,
                "OR" => ScalarAlu::Or,
                "ADD_alu_r_rr" => ScalarAlu::Add,
                _ => ScalarAlu::Lshl,
            },
            dst: index(i, "d0")?,
            a: index(i, "s0")?,
            b: index(i, "s1")?,
        },
        "MOV_OR" => Op::Scalar {
            kind: ScalarAlu::Or,
            dst: index(i, "d0")?,
            a: index(i, "s0")?,
            b: index(i, "s0")?,
        },
        "EQZ" | "NEZ" => Op::Scalar {
            kind: if i.encoding.name == "EQZ" {
                ScalarAlu::Eqz
            } else {
                ScalarAlu::Nez
            },
            dst: index(i, "d0")?,
            a: index(i, "s0")?,
            b: 0,
        },
        "ADD_add_r_ri" => Op::Add(index(i, "d0")?, index(i, "s0")?, imm("imm")?),
        "ADD_NC_mv_add_ri" => {
            let dst = decode::resolve_register(&i.encoding.operands[0], value(i, "dst")? as u64)
                .ok_or(Error::Unsupported)?
                .name;
            let n = ["ls", "le", "lc"]
                .iter()
                .position(|&name| name == dst)
                .ok_or(Error::Unsupported)?;
            Op::LoopAdd(n as u8, index(i, "s0")?, imm("imm")?)
        }
        "XOR" => Op::Xor(index(i, "d0")?, index(i, "s0")?, index(i, "s1")?),
        "LDA_dms_lda_idx_imm" => Op::Load(scalar(i, "dst")?, index(i, "ptr")?, imm("imm")?),
        "ST_dms_sts_idx_imm" | "ST_dms_sts_pstm_nrm_imm" => Op::Store(
            scalar(i, "src")?,
            index(i, "ptr")?,
            imm("imm")?,
            i.encoding.name == "ST_dms_sts_pstm_nrm_imm",
        ),
        "ACQ_mLockId_imm" | "REL_mLockId_imm" => Op::Lock(
            index(i, "id")?,
            index(i, "s1")?,
            i.encoding.name == "ACQ_mLockId_imm",
        ),
        "J_lng" | "JL_lng" | "JNZ" => Op::Jump(
            value(i, "i")? as usize,
            if i.encoding.name == "JNZ" {
                Some(index(i, "s0")?)
            } else {
                None
            },
            i.encoding.name == "JL_lng",
        ),
        "JZ" => Op::JumpZero(value(i, "i")? as usize, index(i, "s0")?),
        "VLDA_dmx_lda_x_pstm_nrm_imm" | "VLDB_dmx_ldb_x_pstm_nrm_imm" => {
            Op::VectorLoad(index(i, "dst")?, index(i, "ptr")?, imm("imm")?)
        }
        "VLDB_dmx_ldb_x_pstm_nrm" => {
            Op::VectorLoadModifier(index(i, "dst")?, index(i, "ptr")?, index(i, "mod")?)
        }
        "VLDA_dmx_lda_bm_pstm_nrm_imm" => {
            Op::AccumulatorLoad(index(i, "dst")?, index(i, "ptr")?, imm("imm")?)
        }
        "VST_dmx_sts_bm_pstm_nrm_imm" => {
            Op::VectorStore(index(i, "src")?, index(i, "ptr")?, imm("imm")?)
        }
        "PADDA_pstm_nrm" | "PADDB_pstm_nrm" | "PADDS_pstm_nrm" => {
            Op::PointerAdd(index(i, "ptr")?, index(i, "mod")?)
        }
        "PADDS_pstm_nrm_imm" => Op::PointerAddImmediate(index(i, "ptr")?, imm("imm")?),
        "VMUL_vmul_cm_core_X_X"
        | "VMAC_vmul_cm_core_X_X"
        | "VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_X" => {
            let name = i.encoding.name;
            Op::Multiply {
                dst: index(i, "dst")?,
                acc: if name.starts_with("VMUL") {
                    7
                } else {
                    index(i, "acc1")?
                },
                acc2: if name.starts_with("VADDMAC") {
                    index(i, "acc2")?
                } else {
                    7
                },
                a: index(i, "s1")?,
                b: index(i, "s2")?,
                config: index(i, "acc")?,
            }
        }
        _ => return Err(Error::Unsupported),
    })
}
fn timed(i: &DecodedInst) -> Result<TimedOp> {
    let op = operation(i)?;
    if i.encoding.itinerary == "NoItinerary" {
        return Ok(TimedOp::functional(op));
    }
    // Pseudo aliases (`MOV_OR` = `or d0, s0, s0`) have no opcode entry of their own: use their itinerary class's opcode.
    let info = sched::opcode_info(i.encoding.name)
        .or_else(|| sched::opcode_info(i.encoding.itinerary.trim_start_matches("II_")))
        .ok_or(Error::Unsupported)?;
    let mut canonical = [None; 32];
    if i.operands.len() > canonical.len() {
        return Err(Error::Unsupported);
    }
    for (n, (value, operand)) in i.operands.iter().zip(i.encoding.operands).enumerate() {
        canonical[n] = decode::resolve_register(operand, value.value as u64).map(|r| r.name);
    }
    let itinerary = info.select(&canonical[..i.operands.len()]);
    let timing = |names: &[&str]| {
        names.iter().find_map(|name| {
            itinerary
                .operand_index(name, 0)
                .map(|n| itinerary.operands[n])
        })
    };
    let cycle = |names: &[&str]| timing(names).map_or(1, |t| t.cycle);
    let mut result = TimedOp::functional(op);
    result.dst = cycle(&["dst", "d0", "d", "cmp"]);
    result.pointer_write = cycle(&["ptr_out"]);
    result.source = cycle(&["src", "s0", "s1"]);
    result.pointer = cycle(&["ptr"]);
    result.modifier = cycle(&["mod"]);
    let scalar_alu = matches!(op, Op::Xor(..) | Op::Scalar { .. });
    result.a = cycle(&[if scalar_alu { "s0" } else { "s1" }]);
    result.b = cycle(&[if scalar_alu { "s1" } else { "s2" }]);
    result.a_bypass = timing(&["s1"]).map_or(0, |t| t.bypass);
    result.b_bypass = timing(&["s2"]).map_or(0, |t| t.bypass);
    result.acc2 = cycle(&["acc2"]);
    result.acc2_forwarding = timing(&["acc2"]).map_or(0, |t| t.bypass);
    result.control = cycle(&["crUPSMode", "crUnpackSize"]);
    result.cmp = cycle(&["cmp"]);
    result.accumulator = cycle(&["acc1"]);
    result.configuration = cycle(&["acc"]);
    result.forwarding = timing(&["dst", "d0", "d"]).map_or(0, |t| t.bypass);
    result.acc_forwarding = timing(&["acc1"]).map_or(0, |t| t.bypass);
    result.memory = if matches!(op, Op::Lock(..)) {
        itinerary.retire_cycle().max(1)
    } else {
        itinerary.memory_cycles.last().copied().unwrap_or(1)
    };
    Ok(result)
}
