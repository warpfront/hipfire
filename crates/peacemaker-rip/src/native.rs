//! Native vocabulary of the evaluator: the generic builder objects
//! (`KernelSpec`, `KernargLayout`, `RegPlan`, `Builder`, `Instruction`), the
//! shared kernel helpers of `peacemaker_isa::kernels::common`, the `Chain`
//! emission builtin, and the typed-core surface (`Workgroup` / `Wave`
//! methods, `Ring`, `ready` / `retire` / ...) lowered onto the sealed
//! runtime driver. Everything else a script calls is evaluated script code.

use crate::driver::DynDriver;
use crate::interp::{Ctl, Cx, Interp, R};
use crate::value::*;
use peacemaker_author::runtime::{
    Arrived, Cond, Exit, Phase, Place, RegionId, RingId, Transition, WgCond,
};
use peacemaker_isa::insn::{Instruction, MemoryClass, Swmmac, SwmmacMods, Wmma};
use peacemaker_isa::kernels::common;
use peacemaker_isa::ledger::Counter;
use peacemaker_isa::plan::{Access, MemoryScope};
use peacemaker_isa::reg::{Kind as RegKind, Live, RegRef, V};
use peacemaker_isa::vopd::{Operand, VopdF32, VopdOp};
use peacemaker_isa::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
use std::cell::RefCell;
use std::rc::Rc;

#[path = "native_fa.rs"]
mod native_fa;
#[path = "native_fp8.rs"]
mod native_fp8;
/// The production generators (`iu4_gemm`, `qwen4_moe_sym`) as natives.
#[path = "native_gen.rs"]
mod native_gen;
#[path = "native_mq6.rs"]
mod native_mq6;
#[path = "native_prod.rs"]
mod native_prod;
#[path = "native_qmoe.rs"]
mod native_qmoe;

#[derive(Clone)]
pub struct Chain {
    pub dst: u8,
    pub width: u8,
    pub kind: Rc<str>,
}

/// Opaque native values. Handles of the driver (regions, rings, conditions,
/// exits) are `Copy`-like tokens: all state lives in the sealed driver.
#[derive(Clone)]
pub enum Nat {
    Arch(Arch),
    Reg(RegRef),
    Insn(Rc<Instruction>),
    Mem(MemoryClass),
    Counter(Counter),
    Live(Live),
    Kernargs(KernargLayout),
    Plan(Rc<RefCell<RegPlan>>),
    Builder(Rc<RefCell<Option<Builder>>>),
    Emitted(Rc<Emitted>),
    Chain(Chain),
    /// The workgroup scope handle (`wg` / `w`): the current driver.
    Wg,
    /// `wg.isa()`: the builder behind the current driver.
    Isa,
    /// The `f` of `forward(|w, f| ..)`.
    Fwd,
    /// The `brk` of `loop_until(.., |w, brk| ..)`.
    Brk,
    Region(RegionId),
    Ring(RingId),
    Pend(Place),
    Drained(Place),
    Trans(Transition),
    Cond(Cond),
    WgCond(WgCond),
    Exit(Exit),
    Arrived(Arrived),
    Phase(Phase),
    /// A `Swmmac::…` SWMMAC kind.
    Swmmac(Swmmac),
    /// An `iu4_gemm` generator (`Iu4Gen::new(spec)`): register assignment and
    /// geometry of one valid spec, read by the `iu4::*` blocks.
    Iu4Gen(Rc<peacemaker_isa::kernels::iu4_gemm::Gen>),
    /// An `mq6_i8_gemm` generator (`Mq6Gen::new(spec)`): register assignment and geometry.
    Mq6Gen(Rc<peacemaker_isa::kernels::mq6_i8_gemm::Gen>),
    /// An `fp8_gemm` generator (`Fp8Gen::new(spec)`): the validated spec every `fp8::*` block reads.
    Fp8Gen(Rc<peacemaker_isa::kernels::fp8_gemm::blocks::Gen>),
    /// A `qwen4_moe_sym` NT4 / NT8 entry generator (`MoeNtGen::new(kind, nt, gp, arch)`): registers, labels, launch contract.
    MoeNtGen(Rc<peacemaker_isa::kernels::qwen4_moe_sym::blocks::Gen>),
    /// A `qwen4_moe_sym` NT1 entry generator (`MoeNt1Gen::new(kind, arch)`).
    MoeNt1Gen(Rc<peacemaker_isa::kernels::qwen4_moe_sym::nt1_blocks::Gen>),
    /// A `fa_prefill` generator (`FaGen::new(arch, spec)`): the validated spec every `fa::*` block reads.
    FaGen(Rc<peacemaker_isa::kernels::fa_prefill::blocks::Gen>),
    /// A PM producer generator (`C8Gen::new`, `RotGen::new`, `GdnGen::new`).
    ProdGen(Rc<native_prod::ProdGen>),
}

impl Nat {
    pub fn type_name(&self) -> &'static str {
        match self {
            Nat::Arch(_) => "Arch",
            Nat::Reg(_) => "RegRef",
            Nat::Insn(_) => "Instruction",
            Nat::Mem(_) => "MemoryClass",
            Nat::Counter(_) => "Counter",
            Nat::Live(_) => "Live",
            Nat::Kernargs(_) => "KernargLayout",
            Nat::Plan(_) => "RegPlan",
            Nat::Builder(_) => "Builder",
            Nat::Emitted(_) => "Emitted",
            Nat::Chain(_) => "Chain",
            Nat::Wg => "Workgroup",
            Nat::Isa => "Isa",
            Nat::Fwd => "Forward",
            Nat::Brk => "Breaks",
            Nat::Region(_) => "LdsRegion",
            Nat::Ring(_) => "Ring",
            Nat::Pend(_) => "Pending",
            Nat::Drained(_) => "Drained",
            Nat::Trans(_) => "Transition",
            Nat::Cond(_) => "Cond",
            Nat::WgCond(_) => "WgCond",
            Nat::Exit(_) => "Exit",
            Nat::Arrived(_) => "Arrived",
            Nat::Phase(_) => "Phase",
            Nat::Swmmac(_) => "Swmmac",
            Nat::Iu4Gen(_) => "Iu4Gen",
            Nat::Mq6Gen(_) => "Mq6Gen",
            Nat::Fp8Gen(_) => "Fp8Gen",
            Nat::MoeNtGen(_) => "MoeNtGen",
            Nat::MoeNt1Gen(_) => "MoeNt1Gen",
            Nat::FaGen(_) => "FaGen",
            Nat::ProdGen(_) => "ProdGen",
        }
    }
    pub fn equals(&self, o: &Nat) -> bool {
        match (self, o) {
            (Nat::Arch(a), Nat::Arch(b)) => a == b,
            (Nat::Reg(a), Nat::Reg(b)) => a == b,
            (Nat::Mem(a), Nat::Mem(b)) => a == b,
            (Nat::Counter(a), Nat::Counter(b)) => a == b,
            (Nat::Live(a), Nat::Live(b)) => a == b,
            (Nat::Region(a), Nat::Region(b)) => a == b,
            (Nat::Ring(a), Nat::Ring(b)) => a == b,
            (Nat::Phase(a), Nat::Phase(b)) => a == b,
            (Nat::Swmmac(a), Nat::Swmmac(b)) => a == b,
            (Nat::Wg, Nat::Wg) | (Nat::Isa, Nat::Isa) => true,
            _ => false,
        }
    }
    pub fn display(&self) -> Option<String> {
        match self {
            Nat::Arch(a) => Some(a.name().to_string()),
            Nat::Reg(r) => Some(r.to_string()),
            _ => None,
        }
    }
    pub fn debug(&self) -> String {
        match self {
            Nat::Arch(a) => format!("{a:?}"),
            Nat::Reg(r) => format!("{r}"),
            Nat::Mem(m) => format!("{m:?}"),
            Nat::Counter(c) => format!("{c:?}"),
            Nat::Phase(p) => format!("{p:?}"),
            Nat::Live(l) => format!("{l:?}"),
            Nat::Region(r) => format!("{r:?}"),
            Nat::Ring(r) => format!("{r:?}"),
            Nat::Swmmac(k) => format!("{k:?}"),
            other => other.type_name().to_string(),
        }
    }
}

// ---- names ----------------------------------------------------------------------

pub(crate) const COMMON_FNS: &[&str] = &[
    "v",
    "vr",
    "s",
    "sr",
    "lit",
    "op",
    "mem",
    "sop",
    "smem",
    "srd_tail",
    "add64",
    "add64_imm",
    "smem_off",
    "s_add_u32",
    "s_addc_u32",
    "s_add_i32",
    "bload",
    "bstore_b128",
    "gather_offset",
    "workgroup_id",
    "ready",
    "retire",
    "rotate",
    "prime",
    "retire_cur",
    "advance",
    "join",
];
pub(crate) const ASSOC_FNS: &[&str] = &[
    "Builder::new",
    "RegPlan::new",
    "KernargLayout::new",
    "Instruction::new",
    "Chain::new",
    "Live::Between",
    "Ring::new",
    "Workgroup::new",
    "String::new",
    "String::from",
    "Vec::new",
    "Vec::with_capacity",
    "cmp::max",
    "cmp::min",
    "mem::swap",
    "Wmma::iu4",
    "Wmma::fp8",
    "Wmma::swmmac",
];
/// Method names `wg_method` implements on `Workgroup` / `Wave`.
pub(crate) const WORKGROUP_METHODS: &[&str] = &[
    "isa",
    "position",
    "lds",
    "relayout",
    "join",
    "split",
    "ring",
    "ring_steady",
    "begin_write",
    "ds_store",
    "ds_store2",
    "ds_load",
    "ds_load_cur",
    "wait",
    "wait_all",
    "barrier",
    "signal",
    "wait_arrived",
    "scmp",
    "scmp_wg_uniform",
    "label",
    "skip_if",
    "skip_unless",
    "wg_skip_if",
    "exec_if",
    "forward",
    "if_else",
    "wg_if_else",
    "loop_until",
    "break_if",
    "loop_carried",
    "exit",
    "exit_if",
    "wave_exit_unless",
    "wg_exit_unless",
    "end",
    "end_with",
    "handoff",
];

pub fn is_fn(name: &str) -> bool {
    if COMMON_FNS.contains(&name)
        || ASSOC_FNS.contains(&name)
        || native_gen::is_fn(name)
        || native_mq6::is_fn(name)
        || native_fp8::is_fn(name)
        || native_qmoe::is_fn(name)
        || native_fa::is_fn(name)
        || native_prod::is_fn(name)
    {
        return true;
    }
    matches!(name.split_once("::"), Some((t, "from" | "try_from" | "max" | "min")) if IntTy::parse(t).is_some())
}

pub fn is_type(n: &str) -> bool {
    matches!(
        n,
        "Arch"
            | "MemoryClass"
            | "Counter"
            | "Live"
            | "KernelSpec"
            | "KernargLayout"
            | "RegPlan"
            | "Builder"
            | "Instruction"
            | "Chain"
            | "Wmma"
            | "VopdOp"
            | "Swmmac"
            | "Ring"
            | "Workgroup"
            | "Option"
            | "Result"
            | "String"
            | "Vec"
            | "Phase"
            | "f32"
            | "f64"
            | "bool"
            | "char"
            | "str"
    ) || IntTy::parse(n).is_some()
        || native_gen::is_type(n)
        || native_mq6::is_type(n)
        || native_fp8::is_type(n)
        || native_qmoe::is_type(n)
        || native_fa::is_type(n)
        || native_prod::is_type(n)
}

pub fn is_struct(n: &str) -> bool {
    matches!(n, "KernelSpec" | "GatherTemps" | "VopdOp" | "SwmmacMods")
        || native_gen::is_struct(n)
        || native_mq6::is_struct(n)
        || native_fp8::is_struct(n)
        || native_fa::is_struct(n)
}

pub fn global_const<'a>(n: &str) -> Option<Value<'a>> {
    Some(match n {
        "SRD_WORD3" => Value::u32(common::SRD_WORD3),
        "GATHER_OOB" => Value::u32(common::GATHER_OOB),
        "GATHER_MAX_INDEX" => Value::u32(common::GATHER_MAX_INDEX),
        _ => return None,
    })
}

pub fn assoc_const<'a>(ty: &str, item: &str) -> Option<Value<'a>> {
    if let Some(v) = native_gen::assoc_const(ty, item) {
        return Some(v);
    }
    if let Some(v) = native_mq6::assoc_const(ty, item) {
        return Some(v);
    }
    if let Some(v) = native_fp8::assoc_const(ty, item) {
        return Some(v);
    }
    if let Some(v) = native_fa::assoc_const(ty, item) {
        return Some(v);
    }
    let nat = |n: Nat| Some(Value::Nat(n));
    match (ty, item) {
        ("Arch", "Gfx1151") => nat(Nat::Arch(Arch::Gfx1151)),
        ("Arch", "Gfx1201") => nat(Nat::Arch(Arch::Gfx1201)),
        ("Arch", "Gfx1100") => nat(Nat::Arch(Arch::Gfx1100)),
        ("MemoryClass", "VmemLoad") => nat(Nat::Mem(MemoryClass::VmemLoad)),
        ("MemoryClass", "VmemStore") => nat(Nat::Mem(MemoryClass::VmemStore)),
        ("MemoryClass", "DsLoad") => nat(Nat::Mem(MemoryClass::DsLoad)),
        ("MemoryClass", "DsStore") => nat(Nat::Mem(MemoryClass::DsStore)),
        ("MemoryClass", "SmemLoad") => nat(Nat::Mem(MemoryClass::SmemLoad)),
        ("MemoryClass", "Export") => nat(Nat::Mem(MemoryClass::Export)),
        ("Counter", "Load") => nat(Nat::Counter(Counter::Load)),
        ("Counter", "Store") => nat(Nat::Counter(Counter::Store)),
        ("Counter", "Ds") => nat(Nat::Counter(Counter::Ds)),
        ("Counter", "Km") => nat(Nat::Counter(Counter::Km)),
        ("Counter", "Vm") => nat(Nat::Counter(Counter::Vm)),
        ("Counter", "Vs") => nat(Nat::Counter(Counter::Vs)),
        ("Counter", "Lgkm") => nat(Nat::Counter(Counter::Lgkm)),
        ("Counter", "Exp") => nat(Nat::Counter(Counter::Exp)),
        ("Live", "Whole") => nat(Nat::Live(Live::Whole)),
        ("Phase", "Free") => nat(Nat::Phase(Phase::RtFree)),
        ("Phase", "Writing") => nat(Nat::Phase(Phase::RtWriting)),
        ("Phase", "Published") => nat(Nat::Phase(Phase::RtPublished)),
        ("Swmmac", kind) => Swmmac::ALL
            .iter()
            .find(|k| format!("{k:?}") == kind)
            .map(|k| Value::Nat(Nat::Swmmac(*k))),
        (t, "MAX") if IntTy::parse(t).is_some() => {
            IntTy::parse(t).map(|t| Value::Int(t.max(), Some(t)))
        }
        (t, "MIN") if IntTy::parse(t).is_some() => {
            IntTy::parse(t).map(|t| Value::Int(t.min(), Some(t)))
        }
        (t, "BITS") if IntTy::parse(t).is_some() => {
            IntTy::parse(t).map(|t| Value::Int(i128::from(t.bits()), Some(IntTy::U32)))
        }
        _ => None,
    }
}

// ---- argument conversion --------------------------------------------------------

type Res<T> = Result<T, String>;

fn int<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<i128> {
    match it.deref(v.clone()) {
        Value::Int(i, _) => Ok(i),
        o => Err(format!(
            "{what}: expected an integer, found {}",
            o.type_name()
        )),
    }
}
fn u8_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<u8> {
    u8::try_from(int(it, v, what)?).map_err(|_| format!("{what}: value does not fit u8"))
}
fn u16_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<u16> {
    u16::try_from(int(it, v, what)?).map_err(|_| format!("{what}: value does not fit u16"))
}
fn u32_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<u32> {
    u32::try_from(int(it, v, what)?).map_err(|_| format!("{what}: value does not fit u32"))
}
fn usize_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<usize> {
    usize::try_from(int(it, v, what)?).map_err(|_| format!("{what}: value does not fit usize"))
}
fn str_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<String> {
    match it.deref(v.clone()) {
        Value::Str(s) => Ok(s.to_string()),
        o => Err(format!(
            "{what}: expected a string, found {}",
            o.type_name()
        )),
    }
}
fn nat_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Option<Nat> {
    match it.deref(v.clone()) {
        Value::Nat(n) => Some(n),
        _ => None,
    }
}
fn arch_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Arch> {
    match nat_of(it, v) {
        Some(Nat::Arch(a)) => Ok(a),
        _ => Err("expected an Arch".into()),
    }
}
fn regs_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<Vec<RegRef>> {
    match it.deref(v.clone()) {
        Value::Array(xs) => xs.iter().map(|x| reg_of(it, x, what)).collect(),
        Value::Nat(Nat::Reg(r)) => Ok(vec![r]),
        o => Err(format!(
            "{what}: expected an array of registers, found {}",
            o.type_name()
        )),
    }
}
fn reg_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<RegRef> {
    match nat_of(it, v) {
        Some(Nat::Reg(r)) => Ok(r),
        _ => Err(format!(
            "{what}: expected a register (v(..), s(..), vr(..), sr(..))"
        )),
    }
}
fn u8s_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<Vec<u8>> {
    match it.deref(v.clone()) {
        Value::Array(xs) => xs.iter().map(|x| u8_of(it, x, what)).collect(),
        o => Err(format!(
            "{what}: expected an array of register numbers, found {}",
            o.type_name()
        )),
    }
}
fn insn_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Instruction> {
    match nat_of(it, v) {
        Some(Nat::Insn(i)) => Ok((*i).clone()),
        _ => Err("expected an Instruction".into()),
    }
}

fn vgpr_of<'a, const N: u8>(it: &Interp<'a>, v: &Value<'a>) -> Res<V<N>> {
    let r = reg_of(it, v, "WMMA operand")?;
    if r.kind != RegKind::V || r.len != N {
        return Err(format!("WMMA operand requires {N} VGPRs, found {r}"));
    }
    Ok(V::<N>(r.base))
}

fn wmma_acc_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Option<V<8>>> {
    match it.deref(v.clone()) {
        Value::Enum(t, var, p) if &*t == "Option" => match (&*var, p.as_slice()) {
            ("Some", [v]) => Ok(Some(vgpr_of(it, v)?)),
            ("None", []) => Ok(None),
            _ => Err("WMMA accumulator requires Some(vr(base, 8)) or None".into()),
        },
        _ => Err("WMMA accumulator requires Some(vr(base, 8)) or None".into()),
    }
}

fn vopd_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<VopdOp> {
    let Value::Struct(ty, fields) = it.deref(v.clone()) else {
        return Err("vopd requires VopdOp { op, dst, src0, src1 }".into());
    };
    if &*ty != "VopdOp" {
        return Err("vopd requires VopdOp { op, dst, src0, src1 }".into());
    }
    let get = |name: &str| {
        fields
            .iter()
            .find(|(n, _)| &**n == name)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| format!("VopdOp is missing `{name}`"))
    };
    let op = str_of(it, &get("op")?, "VopdOp.op")?;
    let op = match op.as_str() {
        "add" => VopdF32::Add,
        "sub" => VopdF32::Sub,
        "subrev" => VopdF32::Subrev,
        "mul" => VopdF32::Mul,
        "fmac" => VopdF32::Fmac,
        _ => {
            return Err(format!(
                "VopdOp: unknown operation `{op}` (expected add, sub, subrev, mul, fmac)"
            ))
        }
    };
    let src0 = match it.deref(get("src0")?) {
        Value::Nat(Nat::Reg(r)) if r.len == 1 => match r.kind {
            RegKind::V => Operand::V(r.base),
            RegKind::S => Operand::S(r.base),
        },
        Value::Tuple(xs) if xs.len() == 2 => {
            match str_of(it, &xs[0], "VopdOp.src0 kind")?.as_str() {
                "lit" => Operand::Lit(u32_of(it, &xs[1], "VopdOp literal")?),
                "inline" => Operand::Inline(
                    i8::try_from(int(it, &xs[1], "VopdOp inline")?)
                        .map_err(|_| "VopdOp inline: value does not fit i8".to_string())?,
                ),
                _ => {
                    return Err(
                        "VopdOp.src0 requires v(n), s(n), (\"lit\", u32) or (\"inline\", i8)"
                            .into(),
                    )
                }
            }
        }
        _ => {
            return Err(
                "VopdOp.src0 requires v(n), s(n), (\"lit\", u32) or (\"inline\", i8)".into(),
            )
        }
    };
    Ok(VopdOp {
        op,
        dst: u8_of(it, &get("dst")?, "VopdOp.dst")?,
        src0,
        src1: u8_of(it, &get("src1")?, "VopdOp.src1")?,
    })
}
fn swmmac_kind_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Swmmac> {
    match nat_of(it, v) {
        Some(Nat::Swmmac(k)) => Ok(k),
        _ => Err("Wmma::swmmac: the kind is a Swmmac, e.g. Swmmac::F32F16".into()),
    }
}

fn swmmac_mods_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<SwmmacMods> {
    const SHAPE: &str = "Wmma::swmmac requires SwmmacMods { index_key, neg_lo, neg_hi, clamp }";
    let Value::Struct(ty, fields) = it.deref(v.clone()) else {
        return Err(SHAPE.into());
    };
    if &*ty != "SwmmacMods" {
        return Err(SHAPE.into());
    }
    let get = |name: &str| {
        fields
            .iter()
            .find(|(n, _)| &**n == name)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| format!("SwmmacMods is missing `{name}`"))
    };
    let clamp = match it.deref(get("clamp")?) {
        Value::Bool(b) => b,
        o => {
            return Err(format!(
                "SwmmacMods.clamp: expected a bool, found {}",
                o.type_name()
            ))
        }
    };
    Ok(SwmmacMods {
        index_key: u8_of(it, &get("index_key")?, "SwmmacMods.index_key")?,
        neg_lo: u8_of(it, &get("neg_lo")?, "SwmmacMods.neg_lo")?,
        neg_hi: u8_of(it, &get("neg_hi")?, "SwmmacMods.neg_hi")?,
        clamp,
    })
}

/// An `Instruction` consumed by the active driver, validated against the
/// driver's arch before the driver sees it.
fn checked_insn<'a>(it: &Interp<'a>, v: &Value<'a>, cx: &mut Cx<'_>) -> R<'a, Instruction> {
    let insn = insn_of(it, v)?;
    let arch = driver(cx)?.isa().spec.arch;
    crate::validate::instruction(arch, &insn.text, &insn.defs, &insn.uses)?;
    Ok(insn)
}
fn opt_u8<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<Option<u8>> {
    match it.deref(v.clone()) {
        Value::Enum(t, var, p) if &*t == "Option" => {
            if &*var == "Some" {
                Ok(Some(u8_of(it, &p[0], what)?))
            } else {
                Ok(None)
            }
        }
        o => Err(format!(
            "{what}: expected Option<u8>, found {}",
            o.type_name()
        )),
    }
}

fn unit<'a>() -> Value<'a> {
    Value::Unit
}
fn done<'a>(r: Res<()>) -> Value<'a> {
    match r {
        Ok(()) => Value::ok(unit()),
        Err(e) => Value::err(e),
    }
}
fn done_with<'a, T>(r: Res<T>, f: impl FnOnce(T) -> Value<'a>) -> Value<'a> {
    match r {
        Ok(t) => Value::ok(f(t)),
        Err(e) => Value::err(e),
    }
}
fn nat<'a>(n: Nat) -> Value<'a> {
    Value::Nat(n)
}

fn check_args(name: &str, args: &[Value], n: usize) -> Res<()> {
    if args.len() == n {
        Ok(())
    } else {
        Err(format!("`{name}` takes {n} arguments, got {}", args.len()))
    }
}

/// Run `f` on the builder a value names: a `Builder::new` result, or the
/// current driver's backend (`wg.isa()`).
fn with_b<'a, T>(
    it: &Interp<'a>,
    v: &Value<'a>,
    cx: &mut Cx<'_>,
    f: impl FnOnce(&mut Builder) -> T,
) -> R<'a, T> {
    match it.deref(v.clone()) {
        Value::Nat(Nat::Builder(cell)) => {
            let mut g = cell.borrow_mut();
            let b = g
                .as_mut()
                .ok_or("the builder is owned by the active Workgroup scope: use wg.isa()")?;
            Ok(f(b))
        }
        Value::Nat(Nat::Isa) => {
            let d =
                cx.d.as_deref_mut()
                    .ok_or("wg.isa() outside the Workgroup scope")?;
            Ok(f(d.isa()))
        }
        o => Err(format!(
            "expected a builder (`Builder::new(..)` or `wg.isa()`), found {}",
            o.type_name()
        )
        .into()),
    }
}

/// `workgroup_id(b, axis, dst)`: copy the work-group id of `axis` (0 = x, 1 = y)
/// into SGPR `dst`, from the architecture's launch state. Axis 2 (z) is always an
/// error: `KernelSpec` has no z flag and the emitted descriptor always sets
/// `system_sgpr_workgroup_id_z 0`, so the hardware never fills z.
///
/// gfx12 reads the architected TTMPs (`ttmp9` = x, `ttmp7[15:0]` = y); y needs the
/// descriptor's Y work-group id enabled (`system_sgpr_workgroup_id_y`). gfx11 reads
/// the system SGPRs that follow the user SGPRs (x at `s{user_sgpr_count}`, y at the
/// next one when the Y flag is set). The gfx11 source SGPR is a use, so the
/// register plan must own it.
fn workgroup_id(b: &mut Builder, axis: i128, dst: u8) -> Res<()> {
    use peacemaker_isa::arch::WgIdSource;
    let arch = b.spec.arch;
    let y_flag = b.spec.system_sgpr_workgroup_id_y;
    let name = arch.name();
    let needs_y = || {
        format!("workgroup_id: axis 1 (y) needs the Y work-group id enabled on {name}: set `system_sgpr_workgroup_id_y: true` in the KernelSpec")
    };
    match axis {
        0..=1 => {}
        2 => {
            return Err(
                "workgroup id z is not enabled: KernelSpec has no z flag; fold z into x or y"
                    .into(),
            )
        }
        _ => {
            return Err(format!(
                "workgroup_id: axis must be 0 (x), 1 (y) or 2 (z), got {axis}"
            ))
        }
    }
    if arch.gfx12() {
        let text = match axis {
            0 => format!("s_mov_b32 s{dst}, ttmp9"),
            _ if y_flag => format!("s_and_b32 s{dst}, ttmp7, 0xffff"),
            _ => return Err(needs_y()),
        };
        return common::sop(b, text, &[dst], &[]);
    }
    let ids = arch
        .workgroup_ids(b.spec.user_sgpr_count)
        .map_err(|e| format!("workgroup_id: {e}"))?;
    let src = match (axis, ids.x, ids.y) {
        (0, WgIdSource::Sgpr(n), _) => n,
        (_, _, WgIdSource::Sgpr(n)) if y_flag => n,
        _ => return Err(needs_y()),
    };
    common::sop(b, format!("s_mov_b32 s{dst}, s{src}"), &[dst], &[src])
}

fn driver<'a, 'c, 'x>(cx: &'x mut Cx<'c>) -> R<'a, &'x mut (dyn DynDriver + 'c)> {
    cx.d.as_deref_mut().ok_or_else(|| Ctl::Err("no active Workgroup here: `let mut wg = Workgroup::new(&mut b, arch)?;` opens the scope that the rest of its block runs in".to_string()))
}

// ---- KernelSpec ---------------------------------------------------------------

fn spec_value<'a>(s: &KernelSpec) -> Value<'a> {
    let f = |k: &str, v: Value<'a>| (Rc::<str>::from(k), v);
    Value::Struct(
        "KernelSpec".into(),
        vec![
            f("kernel_id", Value::str(&s.kernel_id)),
            f("variant", Value::str(&s.variant)),
            f("arch", nat(Nat::Arch(s.arch))),
            f("symbol", Value::str(&s.symbol)),
            f("kernargs", nat(Nat::Kernargs(s.kernargs.clone()))),
            f("user_sgpr_count", Value::u8(s.user_sgpr_count)),
            f(
                "system_sgpr_workgroup_id_y",
                Value::Bool(s.system_sgpr_workgroup_id_y),
            ),
            f(
                "workgroup_size",
                Value::int(i128::from(s.workgroup_size), IntTy::U16),
            ),
            f(
                "group_segment_fixed_size",
                Value::u32(s.group_segment_fixed_size),
            ),
            f("wave32", Value::Bool(s.wave32)),
            f("cu_mode", Value::Bool(s.cu_mode)),
        ],
    )
}

fn spec_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<KernelSpec> {
    let Value::Struct(n, fs) = it.deref(v.clone()) else {
        return Err("expected a KernelSpec { .. }".into());
    };
    if &*n != "KernelSpec" {
        return Err(format!("expected a KernelSpec, found {n}"));
    }
    let get = |k: &str| {
        fs.iter()
            .find(|(f, _)| &**f == k)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| format!("KernelSpec is missing `{k}`"))
    };
    let boolean = |k: &str| match get(k)? {
        Value::Bool(b) => Ok(b),
        _ => Err(format!("KernelSpec.{k} must be bool")),
    };
    let kernargs = match nat_of(it, &get("kernargs")?) {
        Some(Nat::Kernargs(k)) => k,
        _ => return Err("KernelSpec.kernargs must be a KernargLayout".into()),
    };
    Ok(KernelSpec {
        kernel_id: str_of(it, &get("kernel_id")?, "kernel_id")?,
        variant: str_of(it, &get("variant")?, "variant")?,
        arch: arch_of(it, &get("arch")?)?,
        symbol: str_of(it, &get("symbol")?, "symbol")?,
        kernargs,
        user_sgpr_count: u8_of(it, &get("user_sgpr_count")?, "user_sgpr_count")?,
        system_sgpr_workgroup_id_y: boolean("system_sgpr_workgroup_id_y")?,
        workgroup_size: u16_of(it, &get("workgroup_size")?, "workgroup_size")?,
        group_segment_fixed_size: u32_of(
            it,
            &get("group_segment_fixed_size")?,
            "group_segment_fixed_size",
        )?,
        wave32: boolean("wave32")?,
        cu_mode: boolean("cu_mode")?,
    })
}

pub fn field<'a>(
    it: &Interp<'a>,
    n: &Nat,
    name: &str,
    cx: &mut Cx<'_>,
) -> R<'a, Option<Value<'a>>> {
    Ok(match (n, name) {
        (Nat::Builder(_) | Nat::Isa, "spec") => {
            Some(with_b(it, &Value::Nat(n.clone()), cx, |b| {
                spec_value(&b.spec)
            })?)
        }
        (Nat::Builder(_) | Nat::Isa, "regs") => {
            Some(with_b(it, &Value::Nat(n.clone()), cx, |b| {
                nat(Nat::Plan(Rc::new(RefCell::new(b.regs.clone()))))
            })?)
        }
        (Nat::Emitted(e), "s_text") => Some(Value::str(&e.s_text)),
        (Nat::Kernargs(k), "size") => Some(Value::u32(k.size)),
        _ => None,
    })
}

// ---- chains ---------------------------------------------------------------------

fn vtxt(base: u8, n: u8) -> String {
    if n == 1 {
        format!("v{base}")
    } else {
        format!("v[{base}:{}]", u16::from(base) + u16::from(n) - 1)
    }
}

fn chain_step(b: &mut Builder, c: &Chain, a: u8, aw: u8, bm: u8, bw: u8) -> Res<()> {
    let mn = match &*c.kind {
        "f16" => "v_wmma_f32_16x16x16_f16",
        "bf16" => "v_wmma_f32_16x16x16_bf16",
        k => {
            return Err(format!(
                "Chain: unsupported matrix kind `{k}` (supported: f16, bf16)"
            ))
        }
    };
    if c.width != 8 {
        return Err(format!(
            "Chain: the accumulator is 8 VGPRs, not {}",
            c.width
        ));
    }
    if !matches!(aw, 4 | 8) || !matches!(bw, 4 | 8) {
        return Err(format!(
            "Chain: operand widths must be 4 or 8 VGPRs, got {aw} and {bw}"
        ));
    }
    let acc = vtxt(c.dst, c.width);
    common::op(
        b,
        format!("{mn} {acc}, {}, {}, {acc}", vtxt(a, aw), vtxt(bm, bw)),
        &[common::vr(c.dst, c.width)],
        &[
            common::vr(a, aw),
            common::vr(bm, bw),
            common::vr(c.dst, c.width),
        ],
    )
}

// ---- native functions -----------------------------------------------------------

fn handles<'a>(it: &Interp<'a>, v: &Value<'a>) -> Vec<Value<'a>> {
    match it.deref(v.clone()) {
        Value::Tuple(xs) | Value::Array(xs) => xs,
        Value::Unit => vec![],
        o => vec![o],
    }
}

fn place_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Place> {
    match it.deref(v.clone()) {
        Value::Nat(Nat::Region(r)) => Ok(Place::Region(r)),
        Value::Nat(Nat::Ring(g)) => Ok(Place::Ring(g)),
        Value::Nat(Nat::Pend(p)) | Value::Nat(Nat::Drained(p)) => Ok(p),
        Value::Tuple(xs) if !xs.is_empty() => place_of(it, &xs[0]),
        o => Err(format!(
            "expected an LDS region or ring, found {}",
            o.type_name()
        )),
    }
}
fn place_val<'a>(p: Place) -> Value<'a> {
    match p {
        Place::Region(r) => nat(Nat::Region(r)),
        Place::Ring(g) => nat(Nat::Ring(g)),
    }
}
fn region_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<RegionId> {
    match place_of(it, v)? {
        Place::Region(r) => Ok(r),
        Place::Ring(_) => Err("expected an LDS region, found a ring".into()),
    }
}
fn ring_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<RingId> {
    match place_of(it, v)? {
        Place::Ring(g) => Ok(g),
        Place::Region(_) => Err("expected a ring, found an LDS region".into()),
    }
}
fn trans_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Vec<Transition>> {
    handles(it, v)
        .iter()
        .map(|t| match it.deref(t.clone()) {
            Value::Nat(Nat::Trans(t)) => Ok(t),
            o => Err(format!("barrier carries ready(..)/retire(..)/rotate(..)/prime(..)/retire_cur(..), found {}", o.type_name())),
        })
        .collect()
}

/// Raw builders name backend slots, not the driver's opaque region/ring ids.
fn slot_trans_of<'a>(
    it: &Interp<'a>,
    v: &Value<'a>,
) -> Res<Vec<peacemaker_author::SlotTransition>> {
    handles(it, v)
        .iter()
        .map(|t| match it.deref(t.clone()) {
            Value::Enum(ty, variant, fields) if &*ty == "SlotTransition" && fields.len() == 1 => {
                let slot = usize_of(it, &fields[0], "LDS slot")?;
                match &*variant {
                    "Ready" => Ok(peacemaker_author::SlotTransition::Ready(slot)),
                    "Retire" => Ok(peacemaker_author::SlotTransition::Retire(slot)),
                    _ => Err("unknown raw LDS slot transition".into()),
                }
            }
            Value::Nat(Nat::Trans(_)) => Err("raw Builder barriers require ready(slot)/retire(slot) with integer LDS slot ids; driver region/ring transitions must use Workgroup barriers".into()),
            o => Err(format!("raw Builder barrier carries ready(slot)/retire(slot), found {}", o.type_name())),
        })
        .collect()
}
fn pend<'a>(p: Place) -> Value<'a> {
    nat(Nat::Pend(p))
}

pub fn call_fn<'a>(it: &Interp<'a>, name: &str, args: Vec<Value<'a>>, cx: &mut Cx<'_>) -> R<'a> {
    let a = &args;
    let need = |n: usize| check_args(name, a, n);
    match name {
        "v" => {
            need(1)?;
            Ok(nat(Nat::Reg(common::v(u8_of(it, &a[0], "v")?))))
        }
        "s" => {
            need(1)?;
            Ok(nat(Nat::Reg(common::s(u8_of(it, &a[0], "s")?))))
        }
        "vr" => {
            need(2)?;
            Ok(nat(Nat::Reg(common::vr(u8_of(it, &a[0], "vr")?, u8_of(it, &a[1], "vr")?))))
        }
        "sr" => {
            need(2)?;
            Ok(nat(Nat::Reg(common::sr(u8_of(it, &a[0], "sr")?, u8_of(it, &a[1], "sr")?))))
        }
        "lit" => {
            need(1)?;
            Ok(Value::str(common::lit(u32_of(it, &a[0], "lit")?)))
        }
        "s_add_u32" | "s_addc_u32" | "s_add_i32" => {
            need(1)?;
            let arch = arch_of(it, &a[0])?;
            Ok(Value::str(match name {
                "s_add_u32" => common::s_add_u32(arch),
                "s_addc_u32" => common::s_addc_u32(arch),
                _ => common::s_add_i32(arch),
            }))
        }
        "smem_off" => {
            need(2)?;
            Ok(Value::str(common::smem_off(arch_of(it, &a[0])?, u32_of(it, &a[1], "smem_off")?)))
        }
        "op" => {
            need(4)?;
            let (text, defs, uses) = (str_of(it, &a[1], "op text")?, regs_of(it, &a[2], "op defs")?, regs_of(it, &a[3], "op uses")?);
            Ok(done(
                with_b(it, &a[0], cx, |b| -> Res<Res<()>> {
                    crate::validate::instruction(b.spec.arch, &text, &defs, &uses)?;
                    Ok(common::op(b, text, &defs, &uses))
                })??,
            ))
        }
        "mem" => {
            need(5)?;
            let (text, defs, uses) = (str_of(it, &a[1], "mem text")?, regs_of(it, &a[2], "mem defs")?, regs_of(it, &a[3], "mem uses")?);
            let Some(Nat::Mem(class)) = nat_of(it, &a[4]) else { return Err("mem: the last argument is a MemoryClass".into()) };
            Ok(done(
                with_b(it, &a[0], cx, |b| -> Res<Res<()>> {
                    crate::validate::instruction(b.spec.arch, &text, &defs, &uses)?;
                    Ok(common::mem(b, text, &defs, &uses, class))
                })??,
            ))
        }
        "sop" => {
            need(4)?;
            let (text, defs, uses) = (str_of(it, &a[1], "sop text")?, u8s_of(it, &a[2], "sop defs")?, u8s_of(it, &a[3], "sop uses")?);
            Ok(done(
                with_b(it, &a[0], cx, |b| -> Res<Res<()>> {
                    let (vd, vu): (Vec<RegRef>, Vec<RegRef>) = (defs.iter().map(|&n| common::s(n)).collect(), uses.iter().map(|&n| common::s(n)).collect());
                    crate::validate::instruction(b.spec.arch, &text, &vd, &vu)?;
                    Ok(common::sop(b, text, &defs, &uses))
                })??,
            ))
        }
        "smem" => {
            need(5)?;
            let (d, l, base, off) = (u8_of(it, &a[1], "smem")?, u8_of(it, &a[2], "smem")?, u8_of(it, &a[3], "smem")?, u32_of(it, &a[4], "smem")?);
            Ok(done(with_b(it, &a[0], cx, |b| common::smem(b, d, l, base, off))?))
        }
        "srd_tail" => {
            need(3)?;
            let (srd, rec) = (u8_of(it, &a[1], "srd_tail")?, opt_u8(it, &a[2], "srd_tail")?);
            Ok(done(with_b(it, &a[0], cx, |b| common::srd_tail(b, srd, rec))?))
        }
        "add64" => {
            need(4)?;
            let (d, s, x) = (u8_of(it, &a[1], "add64")?, u8_of(it, &a[2], "add64")?, u8_of(it, &a[3], "add64")?);
            Ok(done(with_b(it, &a[0], cx, |b| common::add64(b, d, s, x))?))
        }
        "add64_imm" => {
            need(3)?;
            let (r, imm) = (u8_of(it, &a[1], "add64_imm")?, u32_of(it, &a[2], "add64_imm")?);
            Ok(done(with_b(it, &a[0], cx, |b| common::add64_imm(b, r, imm))?))
        }
        "bload" => {
            need(6)?;
            let v: Vec<u8> = (1..=4).map(|i| u8_of(it, &a[i], "bload")).collect::<Res<_>>()?;
            let off = u32_of(it, &a[5], "bload")?;
            Ok(done(with_b(it, &a[0], cx, |b| common::bload(b, v[0], v[1], v[2], v[3], off))?))
        }
        "bstore_b128" => {
            need(5)?;
            let v: Vec<u8> = (1..=3).map(|i| u8_of(it, &a[i], "bstore_b128")).collect::<Res<_>>()?;
            let off = u32_of(it, &a[4], "bstore_b128")?;
            Ok(done(with_b(it, &a[0], cx, |b| common::bstore_b128(b, v[0], v[1], v[2], off))?))
        }
        "workgroup_id" => {
            need(3)?;
            let (axis, dst) = (int(it, &a[1], "workgroup_id axis")?, u8_of(it, &a[2], "workgroup_id dst")?);
            Ok(done(with_b(it, &a[0], cx, |b| workgroup_id(b, axis, dst))?))
        }
        "gather_offset" => {
            need(7)?;
            let (dst, index) = (u8_of(it, &a[1], "gather_offset")?, u8_of(it, &a[2], "gather_offset")?);
            let div = opt_u8(it, &a[3], "gather_offset")?;
            let (row, live) = (u32_of(it, &a[4], "gather_offset")?, u8_of(it, &a[5], "gather_offset")?);
            let Value::Struct(n, fs) = it.deref(a[6].clone()) else { return Err("gather_offset: GatherTemps { v, mask } expected".into()) };
            if &*n != "GatherTemps" {
                return Err("gather_offset: GatherTemps { v, mask } expected".into());
            }
            let get = |k: &str| fs.iter().find(|(f, _)| &**f == k).map(|(_, v)| v.clone()).ok_or_else(|| format!("GatherTemps is missing `{k}`"));
            let vs = u8s_of(it, &get("v")?, "GatherTemps.v")?;
            let vs: [u8; 4] = vs.try_into().map_err(|_| "GatherTemps.v has four registers".to_string())?;
            let t = common::GatherTemps { v: vs, mask: u8_of(it, &get("mask")?, "GatherTemps.mask")? };
            Ok(done(with_b(it, &a[0], cx, |b| common::gather_offset(b, dst, index, div, row, live, t))?))
        }
        "Instruction::new" => {
            need(3)?;
            let i = Instruction::new(str_of(it, &a[0], "Instruction::new text")?, regs_of(it, &a[1], "Instruction::new defs")?, regs_of(it, &a[2], "Instruction::new uses")?);
            Ok(nat(Nat::Insn(Rc::new(i))))
        }
        "Wmma::iu4" | "Wmma::fp8" => {
            need(5)?;
            let arch = arch_of(it, &a[0])?;
            let dst = vgpr_of(it, &a[1])?;
            let x = vgpr_of(it, &a[2])?;
            let insn = |i| nat(Nat::Insn(Rc::new(i)));
            if name == "Wmma::iu4" {
                Ok(insn(Wmma::iu4(arch, dst, x, vgpr_of(it, &a[3])?, wmma_acc_of(it, &a[4])?)))
            } else {
                Ok(done_with(Wmma::fp8(arch, dst, x, vgpr_of(it, &a[3])?, wmma_acc_of(it, &a[4])?), insn))
            }
        }
        "Wmma::swmmac" => {
            need(7)?;
            let arch = arch_of(it, &a[0])?;
            let kind = swmmac_kind_of(it, &a[1])?;
            let dst = reg_of(it, &a[2], "Wmma::swmmac D")?;
            let x = reg_of(it, &a[3], "Wmma::swmmac A")?;
            let y = reg_of(it, &a[4], "Wmma::swmmac B")?;
            let index = vgpr_of::<1>(it, &a[5])?;
            let mods = swmmac_mods_of(it, &a[6])?;
            Ok(done_with(Wmma::swmmac(arch, kind, dst, x, y, index, mods), |i| nat(Nat::Insn(Rc::new(i)))))
        }
        "KernargLayout::new" => {
            need(1)?;
            Ok(nat(Nat::Kernargs(KernargLayout::new(u32_of(it, &a[0], "KernargLayout::new")?))))
        }
        "RegPlan::new" => {
            need(2)?;
            Ok(done_with(RegPlan::new(u16_of(it, &a[0], "RegPlan::new")?, u16_of(it, &a[1], "RegPlan::new")?), |p| nat(Nat::Plan(Rc::new(RefCell::new(p))))))
        }
        "Builder::new" => {
            need(2)?;
            let spec = spec_of(it, &a[0])?;
            let Some(Nat::Plan(p)) = nat_of(it, &a[1]) else { return Err("Builder::new: the second argument is a RegPlan".into()) };
            let plan = p.borrow().clone();
            Ok(nat(Nat::Builder(Rc::new(RefCell::new(Some(Builder::new(spec, plan)))))))
        }
        "Live::Between" => {
            need(2)?;
            Ok(nat(Nat::Live(Live::Between(str_of(it, &a[0], "Live::Between")?, str_of(it, &a[1], "Live::Between")?))))
        }
        "Chain::new" => {
            need(3)?;
            let (dst, width, kind) = (u8_of(it, &a[0], "Chain::new")?, u8_of(it, &a[1], "Chain::new")?, str_of(it, &a[2], "Chain::new")?);
            if !matches!(&*kind, "f16" | "bf16") {
                return Err(format!("Chain::new: unsupported matrix kind `{kind}` (supported: f16, bf16)").into());
            }
            if width != 8 {
                return Err(format!("Chain::new: the accumulator is 8 VGPRs, not {width}").into());
            }
            Ok(nat(Nat::Chain(Chain { dst, width, kind: Rc::from(kind.as_str()) })))
        }
        "Workgroup::new" => Err("`Workgroup::new(&mut b, arch)?` must be the initializer of a `let` statement: the rest of that block runs inside the driver scope".into()),
        "Ring::new" => {
            if !(2..=4).contains(&a.len()) {
                return Err(format!("`Ring::new` takes 2 to 4 LDS regions (a depth-2 to depth-4 ring), found {}", a.len()).into());
            }
            let regions: Vec<RegionId> = a.iter().map(|v| region_of(it, v)).collect::<Res<_>>()?;
            let d = driver(cx)?;
            let n = format!("ring@{}", d.position());
            Ok(Value::Nat(Nat::Ring(d.ring_n(&n, &regions)?)))
        }
        "join" => {
            let parts: Vec<RegionId> = if a.len() == 1 { handles(it, &a[0]).iter().map(|v| region_of(it, v)).collect::<Res<_>>()? } else { a.iter().map(|v| region_of(it, v)).collect::<Res<_>>()? };
            let d = driver(cx)?;
            Ok(done_with(d.join(&parts), |r| nat(Nat::Region(r))))
        }
        "ready" | "retire" => {
            if a.is_empty() || a.len() > 2 {
                return Err(format!("`{name}` takes a region (and its Drained token)").into());
            }
            if matches!(it.deref(a[0].clone()), Value::Int(_, _)) {
                need(1)?;
                let slot = usize_of(it, &a[0], "LDS slot")?;
                return Ok(Value::Enum(
                    "SlotTransition".into(),
                    if name == "ready" { "Ready" } else { "Retire" }.into(),
                    vec![Value::usize(slot)],
                ));
            }
            let r = region_of(it, &a[0])?;
            Ok(nat(Nat::Trans(if name == "ready" { Transition::Ready(r) } else { Transition::Retire(r) })))
        }
        "rotate" | "prime" | "retire_cur" | "advance" => {
            if a.is_empty() || a.len() > 2 {
                return Err(format!("`{name}` takes a ring (and its Drained token)").into());
            }
            if matches!(it.deref(a[0].clone()), Value::Int(_, _)) {
                return Err(format!("{name} is a driver-only ring transition; raw Builder barriers accept only ready(slot)/retire(slot) with integer LDS slot ids").into());
            }
            let g = ring_of(it, &a[0])?;
            Ok(nat(Nat::Trans(match name {
                "rotate" => Transition::Rotate(g),
                "prime" => Transition::Prime(g),
                "advance" => Transition::Advance(g),
                _ => Transition::RetireCur(g),
            })))
        }
        "String::new" => Ok(Value::str("")),
        "String::from" => {
            need(1)?;
            Ok(it.deref(a[0].clone()))
        }
        "Vec::new" | "Vec::with_capacity" => Ok(Value::Array(vec![])),
        "cmp::max" | "cmp::min" => {
            need(2)?;
            let (x, y) = (it.deref(a[0].clone()), it.deref(a[1].clone()));
            let o = crate::ops::compare(&x, &y)?;
            Ok(if (name == "cmp::max") == o.is_ge() { x } else { y })
        }
        n => {
            if let Some(r) = native_gen::call_fn(it, n, a, cx) {
                return r;
            }
            if let Some(r) = native_mq6::call_fn(it, n, a, cx) {
                return r;
            }
            if let Some(r) = native_fp8::call_fn(it, n, a, cx) {
                return r;
            }
            if let Some(r) = native_qmoe::call_fn(it, n, a, cx) {
                return r;
            }
            if let Some(r) = native_fa::call_fn(it, n, a, cx) {
                return r;
            }
            if let Some(r) = native_prod::call_fn(it, n, a, cx) {
                return r;
            }
            if let Some((t, f)) = n.split_once("::") {
                if let Some(ty) = IntTy::parse(t) {
                    need(1)?;
                    let v = it.deref(a[0].clone());
                    return match (f, v) {
                        ("from", Value::Int(i, _)) => Ok(Value::Int(i, Some(ty))),
                        ("from", Value::Bool(b)) => Ok(Value::Int(i128::from(b), Some(ty))),
                        ("from", Value::Char(c)) => Ok(Value::Int(c as i128, Some(ty))),
                        ("try_from", Value::Int(i, _)) => Ok(if ty.fits(i) { Value::ok(Value::Int(i, Some(ty))) } else { Value::err(format!("out of range integral type conversion attempted ({i} as {})", ty.name())) }),
                        (f, v) => Err(format!("{t}::{f}: unsupported argument {}", v.type_name()).into()),
                    };
                }
            }
            Err(format!("native function `{n}` is not implemented").into())
        }
    }
}

// ---- native methods ------------------------------------------------------------

/// Evaluate a script callback inside a driver callback. Hard faults are
/// parked in `fail` (the driver is told the lowering failed) and resurface
/// after the driver call returns.
struct Cb<'a> {
    fail: Option<Ctl<'a>>,
}

fn run_cb<'a>(
    it: &Interp<'a>,
    f: &Value<'a>,
    args: Vec<Value<'a>>,
    d: &mut dyn DynDriver,
    fwd: Option<&mut peacemaker_author::runtime::Forward<Builder>>,
    brk: Option<&peacemaker_author::runtime::Breaks<Builder>>,
    cb: &mut Cb<'a>,
) -> Res<Value<'a>> {
    let mut cx = Cx {
        d: Some(d),
        fwd,
        brk,
    };
    let r = it.call_value(f, args, &mut cx);
    let v = match r {
        Ok(v) => v,
        Err(Ctl::Return(v)) => v,
        Err(other) => {
            cb.fail = Some(other);
            return Err("script fault".into());
        }
    };
    match it.deref(v) {
        Value::Enum(t, var, mut p) if &*t == "Result" => {
            if &*var == "Ok" {
                Ok(p.pop().unwrap_or(Value::Unit))
            } else {
                Err(crate::fmt::display(&p.pop().unwrap_or(Value::Unit)).unwrap_or_else(|e| e))
            }
        }
        v => Ok(v),
    }
}

/// The result of a driver call that ran callbacks.
fn finish_cb<'a>(cb: Cb<'a>, r: Res<()>, v: Value<'a>) -> R<'a> {
    if let Some(f) = cb.fail {
        return Err(f);
    }
    Ok(done_with(r, |()| v))
}

/// Bridge a script closure onto a builder draft without aliasing its mutable borrow.
fn run_builder_cb<'a>(
    it: &Interp<'a>,
    f: &Value<'a>,
    draft: &mut Builder,
    cb: &mut Cb<'a>,
) -> Res<()> {
    let cell = Rc::new(RefCell::new(Some(draft.clone())));
    let mut cx = Cx {
        d: None,
        fwd: None,
        brk: None,
    };
    let v = match it.call_value(f, vec![nat(Nat::Builder(cell.clone()))], &mut cx) {
        Ok(v) | Err(Ctl::Return(v)) => v,
        Err(other) => {
            cb.fail = Some(other);
            return Err("script fault".into());
        }
    };
    if let Value::Enum(t, var, mut p) = it.deref(v) {
        if &*t == "Result" && &*var != "Ok" {
            return Err(crate::fmt::display(&p.pop().unwrap_or(Value::Unit)).unwrap_or_else(|e| e));
        }
    }
    *draft = cell
        .borrow_mut()
        .take()
        .ok_or("the callback builder was already finished")?;
    Ok(())
}

pub fn method<'a>(
    it: &Interp<'a>,
    n: &Nat,
    name: &str,
    targs: &[String],
    args: Vec<Value<'a>>,
    cx: &mut Cx<'_>,
) -> R<'a, Option<Value<'a>>> {
    let a = &args;
    let need = |k: usize| check_args(name, a, k);
    Ok(match n {
        Nat::Iu4Gen(g) => native_gen::method(it, g, name, a)?,
        Nat::Mq6Gen(g) => native_mq6::method(it, g, name, a)?,
        Nat::Fp8Gen(g) => native_fp8::method(it, g, name, a)?,
        Nat::MoeNtGen(g) => native_qmoe::nt_method(it, g, name, a)?,
        Nat::MoeNt1Gen(g) => native_qmoe::nt1_method(it, g, name, a)?,
        Nat::FaGen(g) => native_fa::method(it, g, name, a)?,
        Nat::ProdGen(g) => native_prod::method(it, g, name, a)?,
        Nat::Arch(arch) => match name {
            "gfx12" => Some(Value::Bool(arch.gfx12())),
            "name" => Some(Value::str(arch.name())),
            "buffer_offset_max" => Some(Value::u32(arch.buffer_offset_max())),
            _ => None,
        },
        Nat::Insn(i) => match name {
            "memory" => {
                need(1)?;
                let Some(Nat::Mem(c)) = nat_of(it, &a[0]) else {
                    return Err("memory(..) takes a MemoryClass".into());
                };
                Some(nat(Nat::Insn(Rc::new((**i).clone().memory(c)))))
            }
            _ => None,
        },
        Nat::Kernargs(k) => match name {
            "pointer" => {
                need(2)?;
                Some(nat(Nat::Kernargs(k.clone().pointer(
                    &str_of(it, &a[0], "pointer")?,
                    u32_of(it, &a[1], "pointer")?,
                ))))
            }
            "pointer_access" => {
                need(3)?;
                let access = str_of(it, &a[2], "pointer_access")?;
                let access = match access.as_str() {
                    "read_only" => Access::ReadOnly,
                    "write_only" => Access::WriteOnly,
                    "read_write" => Access::ReadWrite,
                    _ => return Err(format!("pointer_access: unknown access `{access}` (expected read_only, write_only, read_write)").into()),
                };
                Some(nat(Nat::Kernargs(k.clone().pointer_access(
                    &str_of(it, &a[0], "pointer_access")?,
                    u32_of(it, &a[1], "pointer_access")?,
                    access,
                ))))
            }
            "hidden" => {
                need(4)?;
                Some(nat(Nat::Kernargs(k.clone().hidden(
                    &str_of(it, &a[0], "hidden")?,
                    u32_of(it, &a[1], "hidden")?,
                    u32_of(it, &a[2], "hidden")?,
                    &str_of(it, &a[3], "hidden")?,
                ))))
            }
            "validate" => Some(done(k.validate())),
            _ => None,
        },
        Nat::Plan(p) => match name {
            "add_range" => {
                need(5)?;
                let kind = str_of(it, &a[1], "add_range kind")?;
                let kind = match kind.as_str() {
                    "v" => RegKind::V,
                    "s" => RegKind::S,
                    _ => {
                        return Err(format!(
                            "add_range: unknown register kind `{kind}` (expected v, s)"
                        )
                        .into())
                    }
                };
                let Some(Nat::Live(live)) = nat_of(it, &a[4]) else {
                    return Err("the last argument is a Live range".into());
                };
                Some(done(p.borrow_mut().add_range(
                    &str_of(it, &a[0], "register name")?,
                    kind,
                    u8_of(it, &a[2], "register base")?,
                    u8_of(it, &a[3], "register length")?,
                    live,
                )))
            }
            "s" | "v" => {
                need(3)?;
                let width = match targs.first() {
                    Some(t) => t
                        .parse::<u8>()
                        .ok()
                        .or_else(|| it.const_int(t).and_then(|c| u8::try_from(c).ok()))
                        .ok_or_else(|| format!("`{name}::<{t}>`: the width is not a constant"))?,
                    None => return Err(format!("`{name}` needs a width: `{name}::<N>(..)`").into()),
                };
                let (nm, base) = (
                    str_of(it, &a[0], "register name")?,
                    u8_of(it, &a[1], "register base")?,
                );
                let Some(Nat::Live(live)) = nat_of(it, &a[2]) else {
                    return Err("the last argument is a Live range".into());
                };
                let kind = if name == "s" { RegKind::S } else { RegKind::V };
                let r = p.borrow_mut().add_range(&nm, kind, base, width, live);
                Some(done_with(r, |()| {
                    nat(Nat::Reg(RegRef {
                        kind,
                        base,
                        len: width,
                    }))
                }))
            }
            "next_free_vgpr" => Some(Value::int(
                i128::from(p.borrow().next_free_vgpr()),
                IntTy::U16,
            )),
            "next_free_sgpr" => Some(Value::int(
                i128::from(p.borrow().next_free_sgpr()),
                IntTy::U16,
            )),
            "scratch_pool" => {
                need(2)?;
                Some(done(p.borrow_mut().scratch_pool(
                    u8_of(it, &a[0], "scratch_pool")?,
                    u8_of(it, &a[1], "scratch_pool")?,
                )))
            }
            _ => None,
        },
        Nat::Builder(_) | Nat::Isa => match name {
            "release_store_sources" | "lds_relayout" | "barrier_wait" => {
                need(0)?;
                Some(done(with_b(
                    it,
                    &Value::Nat(n.clone()),
                    cx,
                    |b| match name {
                        "release_store_sources" => b.release_store_sources(),
                        "lds_relayout" => b.lds_relayout(),
                        _ => b.barrier_wait(),
                    },
                )?))
            }
            "push" | "ds_crosslane" => {
                need(1)?;
                let insn = insn_of(it, &a[0])?;
                Some(done(with_b(it, &Value::Nat(n.clone()), cx, |b| {
                    if name == "push" {
                        insn.check_raw()?;
                    }
                    crate::validate::instruction(b.spec.arch, &insn.text, &insn.defs, &insn.uses)?;
                    if name == "push" {
                        b.push(insn)
                    } else {
                        b.ds_crosslane(insn)
                    }
                })?))
            }
            "control" => {
                need(1)?;
                let insn = insn_of(it, &a[0])?;
                // Symbolic branch labels cannot round-trip through the frontend validator.
                Some(done(with_b(it, &Value::Nat(n.clone()), cx, |b| {
                    b.control(insn)
                })?))
            }
            "lds_slot" => {
                need(3)?;
                let name = str_of(it, &a[0], "LDS slot name")?;
                let base = u32_of(it, &a[1], "LDS slot base")?;
                let len = u32_of(it, &a[2], "LDS slot length")?;
                Some(done_with(
                    with_b(it, &Value::Nat(n.clone()), cx, |b| {
                        b.lds_slot(&name, base, len)
                    })?,
                    Value::usize,
                ))
            }
            "ds_store" | "ds_load" => {
                need(2)?;
                let slot = usize_of(it, &a[0], "LDS slot")?;
                let insn = insn_of(it, &a[1])?;
                Some(done(with_b(it, &Value::Nat(n.clone()), cx, |b| {
                    crate::validate::instruction(b.spec.arch, &insn.text, &insn.defs, &insn.uses)?;
                    if name == "ds_store" {
                        b.ds_store(slot, insn)
                    } else {
                        b.ds_load(slot, insn)
                    }
                })?))
            }
            "barrier_signal" | "barrier_with_scope" => {
                need(if name == "barrier_signal" { 1 } else { 2 })?;
                let transitions = slot_trans_of(it, &a[0])?;
                let scope = if name == "barrier_with_scope" {
                    let scope = str_of(it, &a[1], "barrier scope")?;
                    match scope.as_str() {
                        "lds_only" => MemoryScope::LdsOnly,
                        "workgroup" => MemoryScope::Workgroup,
                        _ => return Err(format!("barrier_with_scope: unknown scope `{scope}` (expected lds_only, workgroup)").into()),
                    }
                } else {
                    MemoryScope::LdsOnly
                };
                Some(done(with_b(it, &Value::Nat(n.clone()), cx, |b| {
                    if name == "barrier_signal" {
                        b.barrier_signal(&transitions)
                    } else {
                        b.barrier_with_scope(&transitions, scope)
                    }
                })?))
            }
            "loop_" => {
                need(2)?;
                let head = str_of(it, &a[0], "loop head")?;
                let cb = RefCell::new(Cb { fail: None });
                let r = with_b(it, &Value::Nat(n.clone()), cx, |b| {
                    b.loop_(&head, |draft| {
                        run_builder_cb(it, &a[1], draft, &mut cb.borrow_mut())
                    })
                })?;
                return finish_cb(cb.into_inner(), r, Value::Unit).map(Some);
            }
            "vopd" => {
                need(2)?;
                let x = vopd_of(it, &a[0])?;
                let y = vopd_of(it, &a[1])?;
                Some(done(with_b(it, &Value::Nat(n.clone()), cx, |b| {
                    b.vopd(x, y)
                })?))
            }
            "enable_delay_alu" => {
                with_b(it, &Value::Nat(n.clone()), cx, |b| b.enable_delay_alu())?;
                Some(unit())
            }
            "clause" => {
                need(1)?;
                let f = a[0].clone();
                let mut cb = Cb { fail: None };
                let r = with_b(it, &Value::Nat(n.clone()), cx, |b| {
                    b.clause(|draft| {
                        let cell = Rc::new(RefCell::new(Some(draft.clone())));
                        let mut cx = Cx {
                            d: None,
                            fwd: None,
                            brk: None,
                        };
                        let r = it.call_value(&f, vec![nat(Nat::Builder(cell.clone()))], &mut cx);
                        let v = match r {
                            Ok(v) | Err(Ctl::Return(v)) => v,
                            Err(other) => {
                                cb.fail = Some(other);
                                return Err("script fault".into());
                            }
                        };
                        if let Value::Enum(t, var, mut p) = it.deref(v) {
                            if &*t == "Result" && &*var != "Ok" {
                                return Err(crate::fmt::display(&p.pop().unwrap_or(Value::Unit))
                                    .unwrap_or_else(|e| e));
                            }
                        }
                        *draft = cell
                            .borrow_mut()
                            .take()
                            .ok_or("the clause builder was already finished")?;
                        Ok(())
                    })
                })?;
                return finish_cb(cb, r, Value::Unit).map(Some);
            }
            "wait" => {
                need(2)?;
                let Some(Nat::Counter(c)) = nat_of(it, &a[0]) else {
                    return Err("wait(Counter, n)".into());
                };
                let k = u8_of(it, &a[1], "wait count")?;
                Some(done(with_b(it, &Value::Nat(n.clone()), cx, |b| {
                    b.wait(c, k)
                })?))
            }
            "finish" => {
                let Nat::Builder(cell) = n else {
                    return Err("finish() is the builder's, not wg.isa()'s".into());
                };
                let b = cell
                    .borrow_mut()
                    .take()
                    .ok_or("the builder was already finished")?;
                Some(done_with(b.finish(), |e| nat(Nat::Emitted(Rc::new(e)))))
            }
            "position" => Some(Value::usize(with_b(it, &Value::Nat(n.clone()), cx, |b| {
                b.program().instructions.len()
            })?)),
            _ => None,
        },
        Nat::Chain(c) => match name {
            "step" => {
                need(5)?;
                let (x, aw, bm, bw) = (
                    u8_of(it, &a[1], "step")?,
                    u8_of(it, &a[2], "step")?,
                    u8_of(it, &a[3], "step")?,
                    u8_of(it, &a[4], "step")?,
                );
                Some(done(with_b(it, &a[0], cx, |b| {
                    chain_step(b, c, x, aw, bm, bw)
                })?))
            }
            _ => None,
        },
        Nat::Ring(g) => match name {
            "into_steady" => {
                let d = driver(cx)?;
                d.ring_steady(*g).map_err(Ctl::Err)?;
                Some(nat(Nat::Ring(*g)))
            }
            "cur_index" => Some(Value::usize(driver(cx)?.cur_index(*g)?)),
            "next_index" => Some(Value::usize(driver(cx)?.next_index(*g)?)),
            "depth" => Some(Value::usize(driver(cx)?.ring_depth(*g)?)),
            "starting_at" => {
                if a.len() != 1 {
                    return Err("`ring.starting_at(k)` takes the index of the buffer that takes the first store".into());
                }
                let k = usize_of(it, &a[0], "ring.starting_at")?;
                driver(cx)?.ring_start_at(*g, k).map_err(Ctl::Err)?;
                Some(nat(Nat::Ring(*g)))
            }
            "phases" => {
                let (c, nx) = driver(cx)?.ring_phases(*g)?;
                Some(Value::Tuple(vec![nat(Nat::Phase(c)), nat(Nat::Phase(nx))]))
            }
            _ => None,
        },
        Nat::Region(r) => match name {
            "phase" => Some(nat(Nat::Phase(driver(cx)?.region_phase(*r)?))),
            _ => None,
        },
        Nat::Fwd => {
            let label_arg = |k: usize| str_of(it, &a[k], "label");
            match name {
                "branch_if" | "branch_unless" => {
                    need(3)?;
                    let Some(Nat::Cond(c)) = nat_of(it, &a[1]) else {
                        return Err(format!("`{name}` takes a Cond from scmp(..)").into());
                    };
                    let l = label_arg(2)?;
                    Some(fwd_op(cx, |d, f| {
                        if name == "branch_if" {
                            d.fwd_branch_if(f, c, &l)
                        } else {
                            d.fwd_branch_unless(f, c, &l)
                        }
                    })?)
                }
                "goto" | "place" => {
                    need(2)?;
                    let l = label_arg(1)?;
                    Some(fwd_op(cx, |d, f| {
                        if name == "goto" {
                            d.fwd_goto(f, &l)
                        } else {
                            d.fwd_place(f, &l)
                        }
                    })?)
                }
                _ => None,
            }
        }
        Nat::Wg => return wg_method(it, name, args, cx),
        _ => None,
    })
}

fn fwd_op<'a>(
    cx: &mut Cx<'_>,
    f: impl FnOnce(&mut dyn DynDriver, &mut peacemaker_author::runtime::Forward<Builder>) -> Res<()>,
) -> R<'a> {
    let Cx { d, fwd, .. } = cx;
    let d = d
        .as_deref_mut()
        .ok_or("forward handle used outside the Workgroup scope")?;
    let fwd = fwd
        .as_deref_mut()
        .ok_or("forward handle used outside its `forward(|w, f| ..)` block")?;
    Ok(done(f(d, fwd)))
}

/// Splits optional state from the callback in `skip_if(cond, label, [state,] f)`.
fn state_and_fn<'a>(a: &[Value<'a>], from: usize) -> Res<(Option<Value<'a>>, Value<'a>)> {
    match a.len() - from {
        1 => Ok((None, a[from].clone())),
        2 => Ok((Some(a[from].clone()), a[from + 1].clone())),
        _ => Err("expected `(.., [state,] |w[, st]| ..)`".into()),
    }
}

fn wg_method<'a>(
    it: &Interp<'a>,
    name: &str,
    args: Vec<Value<'a>>,
    cx: &mut Cx<'_>,
) -> R<'a, Option<Value<'a>>> {
    let a = &args;
    macro_rules! need {
        ($n:expr) => {
            check_args(name, a, $n)?
        };
    }
    let wgv = || nat(Nat::Wg);
    let cond_of = |v: &Value<'a>| -> Res<Cond> {
        match nat_of(it, v) {
            Some(Nat::Cond(c)) => Ok(c),
            _ => Err(format!("`{name}` takes a wave-uniform Cond from scmp(..)")),
        }
    };
    let wgcond_of = |v: &Value<'a>| -> Res<WgCond> {
        match nat_of(it, v) {
            Some(Nat::WgCond(c)) => Ok(c),
            _ => Err(format!("`{name}` takes a WgCond from scmp_wg_uniform(..)")),
        }
    };
    let exit_of = |v: &Value<'a>| -> Res<Exit> {
        match nat_of(it, v) {
            Some(Nat::Exit(e)) => Ok(e),
            _ => Err(format!("`{name}` takes an Exit from exit(..)")),
        }
    };
    Ok(Some(match name {
        "isa" => nat(Nat::Isa),
        "position" => Value::usize(driver(cx)?.position()),
        "lds" => {
            need!(3);
            let (nm, base, len) = (
                str_of(it, &a[0], "lds name")?,
                u32_of(it, &a[1], "lds base")?,
                u32_of(it, &a[2], "lds len")?,
            );
            done_with(driver(cx)?.lds(&nm, base, len), |r| nat(Nat::Region(r)))
        }
        "relayout" => done(driver(cx)?.relayout()),
        "join" => {
            let parts: Vec<RegionId> = if a.len() == 1 {
                handles(it, &a[0])
                    .iter()
                    .map(|v| region_of(it, v))
                    .collect::<Res<_>>()?
            } else {
                a.iter().map(|v| region_of(it, v)).collect::<Res<_>>()?
            };
            done_with(driver(cx)?.join(&parts), |r| nat(Nat::Region(r)))
        }
        "split" => {
            need!(2);
            let (r, first) = (region_of(it, &a[0])?, usize_of(it, &a[1], "split")?);
            done_with(driver(cx)?.split(r, first), |(x, y)| {
                Value::Tuple(vec![nat(Nat::Region(x)), nat(Nat::Region(y))])
            })
        }
        "ring" => {
            need!(3);
            let (nm, x, y) = (
                str_of(it, &a[0], "ring name")?,
                region_of(it, &a[1])?,
                region_of(it, &a[2])?,
            );
            done_with(driver(cx)?.ring(&nm, x, y), |g| nat(Nat::Ring(g)))
        }
        "ring_steady" => {
            need!(1);
            let g = ring_of(it, &a[0])?;
            done_with(driver(cx)?.ring_steady(g), |()| nat(Nat::Ring(g)))
        }
        "begin_write" => {
            need!(1);
            let p = place_of(it, &a[0])?;
            driver(cx)?.begin_write(p).map_err(Ctl::Err)?;
            Value::Tuple(vec![place_val(p), pend(p)])
        }
        "ds_store" => {
            need!(2);
            let (p, insn) = (place_of(it, &a[0])?, checked_insn(it, &a[1], cx)?);
            done_with(driver(cx)?.ds_store(p, insn), |()| {
                Value::Tuple(vec![place_val(p), pend(p)])
            })
        }
        "ds_store2" => {
            need!(3);
            let (x, y, insn) = (
                place_of(it, &a[0])?,
                place_of(it, &a[1])?,
                checked_insn(it, &a[2], cx)?,
            );
            done_with(driver(cx)?.ds_store2(x, y, insn), |()| {
                Value::Tuple(vec![
                    Value::Tuple(vec![place_val(x), pend(x)]),
                    Value::Tuple(vec![place_val(y), pend(y)]),
                ])
            })
        }
        "ds_load" => {
            need!(2);
            let (r, insn) = (region_of(it, &a[0])?, checked_insn(it, &a[1], cx)?);
            done(driver(cx)?.ds_load(r, insn))
        }
        "ds_load_cur" => {
            need!(2);
            let (g, insn) = (ring_of(it, &a[0])?, checked_insn(it, &a[1], cx)?);
            done(driver(cx)?.ds_load_cur(g, insn))
        }
        "wait" => {
            need!(1);
            let p = place_of(it, &a[0])?;
            done_with(driver(cx)?.wait(p), |()| nat(Nat::Drained(p)))
        }
        "wait_all" => {
            need!(1);
            let ps: Vec<Place> = handles(it, &a[0])
                .iter()
                .map(|v| place_of(it, v))
                .collect::<Res<_>>()?;
            done_with(driver(cx)?.wait_all(&ps), |()| {
                Value::Tuple(ps.iter().map(|p| nat(Nat::Drained(*p))).collect())
            })
        }
        "barrier" => {
            need!(1);
            let ts = trans_of(it, &a[0])?;
            done_with(driver(cx)?.barrier(&ts), |()| {
                Value::Tuple(ts.iter().map(transition_result).collect())
            })
        }
        "signal" => {
            need!(1);
            let ts = trans_of(it, &a[0])?;
            done_with(driver(cx)?.signal(&ts), |arr| nat(Nat::Arrived(arr)))
        }
        "wait_arrived" => {
            need!(1);
            let Some(Nat::Arrived(arr)) = nat_of(it, &a[0]) else {
                return Err("wait_arrived takes the Arrived from signal(..)".into());
            };
            done(driver(cx)?.wait_arrived(arr))
        }
        "scmp" => {
            need!(1);
            let insn = checked_insn(it, &a[0], cx)?;
            done_with(driver(cx)?.scmp(insn), |c| nat(Nat::Cond(c)))
        }
        "scmp_wg_uniform" => {
            need!(1);
            let insn = checked_insn(it, &a[0], cx)?;
            done_with(driver(cx)?.scmp_wg_uniform(insn), |c| nat(Nat::WgCond(c)))
        }
        "label" => {
            need!(1);
            done(driver(cx)?.label(&str_of(it, &a[0], "label")?))
        }
        "skip_if" | "skip_unless" | "wg_skip_if" => {
            if a.len() < 3 {
                return Err(format!("`{name}(cond, label, [state,] |w[, st]| ..)`").into());
            }
            let target = str_of(it, &a[1], "skip target")?;
            let (state, f) = state_and_fn(a, 2)?;
            let mut cb = Cb { fail: None };
            let mut out = Value::Unit;
            let d = driver(cx)?;
            let mut body = |d: &mut dyn DynDriver| -> Res<()> {
                let mut cargs = vec![wgv()];
                cargs.extend(state.clone());
                let v = run_cb(it, &f, cargs, d, None, None, &mut cb)?;
                out = v;
                Ok(())
            };
            let r = match name {
                "skip_if" => d.skip_if(cond_of(&a[0])?, &target, &mut body),
                "skip_unless" => d.skip_unless(cond_of(&a[0])?, &target, &mut body),
                _ => d.wg_skip_if(wgcond_of(&a[0])?, &target, &mut body),
            };
            let keep = if state.is_some() { out } else { Value::Unit };
            return finish_cb(cb, r, keep).map(Some);
        }
        "exec_if" => {
            need!(2);
            let c = cond_of(&a[0])?;
            let f = a[1].clone();
            let mut cb = Cb { fail: None };
            let mut out = Value::Unit;
            let r = driver(cx)?.exec_if(c, &mut |d| {
                out = run_cb(it, &f, vec![wgv()], d, None, None, &mut cb)?;
                Ok(())
            });
            return finish_cb(cb, r, out).map(Some);
        }
        "forward" => {
            need!(1);
            let f = a[0].clone();
            let mut cb = Cb { fail: None };
            let mut out = Value::Unit;
            let r = driver(cx)?.forward(&mut |d, fw| {
                out = run_cb(
                    it,
                    &f,
                    vec![wgv(), nat(Nat::Fwd)],
                    d,
                    Some(fw),
                    None,
                    &mut cb,
                )?;
                Ok(())
            });
            return finish_cb(cb, r, out).map(Some);
        }
        "if_else" => {
            need!(5);
            let (c, el, jn) = (
                cond_of(&a[0])?,
                str_of(it, &a[1], "else label")?,
                str_of(it, &a[2], "join label")?,
            );
            let (t, e) = (a[3].clone(), a[4].clone());
            let mut cb = Cb { fail: None };
            let cb_cell = RefCell::new(&mut cb);
            let r = driver(cx)?.if_else(
                c,
                &el,
                &jn,
                &mut |d| {
                    run_cb(
                        it,
                        &t,
                        vec![wgv()],
                        d,
                        None,
                        None,
                        &mut cb_cell.borrow_mut(),
                    )
                    .map(|_| ())
                },
                &mut |d| {
                    run_cb(
                        it,
                        &e,
                        vec![wgv()],
                        d,
                        None,
                        None,
                        &mut cb_cell.borrow_mut(),
                    )
                    .map(|_| ())
                },
            );
            drop(cb_cell);
            return finish_cb(cb, r, Value::Unit).map(Some);
        }
        "wg_if_else" => {
            if !(5..=6).contains(&a.len()) {
                return Err(
                    "`wg_if_else(cond, else_label, join, [state,] |w[, st]| then, |w[, st]| els)`"
                        .into(),
                );
            }
            let (c, el, jn) = (
                wgcond_of(&a[0])?,
                str_of(it, &a[1], "else label")?,
                str_of(it, &a[2], "join label")?,
            );
            let state = (a.len() == 6).then(|| a[3].clone());
            let (t, e) = (a[a.len() - 2].clone(), a[a.len() - 1].clone());
            let mut cb = Cb { fail: None };
            let cb_cell = RefCell::new(&mut cb);
            let out = RefCell::new(Value::Unit);
            let args_of = || {
                let mut v = vec![wgv()];
                v.extend(state.clone());
                v
            };
            let r = driver(cx)?.wg_if_else(
                c,
                &el,
                &jn,
                &mut |d| {
                    run_cb(it, &t, args_of(), d, None, None, &mut cb_cell.borrow_mut()).map(|_| ())
                },
                &mut |d| {
                    let v = run_cb(it, &e, args_of(), d, None, None, &mut cb_cell.borrow_mut())?;
                    *out.borrow_mut() = v;
                    Ok(())
                },
            );
            drop(cb_cell);
            let keep = if state.is_some() {
                out.into_inner()
            } else {
                Value::Unit
            };
            return finish_cb(cb, r, keep).map(Some);
        }
        "loop_until" => {
            need!(3);
            let (head, exit, f) = (
                str_of(it, &a[0], "loop head")?,
                str_of(it, &a[1], "loop exit")?,
                a[2].clone(),
            );
            let cb = RefCell::new(Cb { fail: None });
            let r = driver(cx)?.loop_until(&head, &exit, &|d, brk| {
                run_cb(
                    it,
                    &f,
                    vec![wgv(), nat(Nat::Brk)],
                    d,
                    None,
                    Some(brk),
                    &mut cb.borrow_mut(),
                )
                .map(|_| ())
            });
            return finish_cb(cb.into_inner(), r, Value::Unit).map(Some);
        }
        "break_if" => {
            need!(2);
            let c = cond_of(&a[0])?;
            let Cx { d, brk, .. } = cx;
            let d = d
                .as_deref_mut()
                .ok_or("break_if outside the Workgroup scope")?;
            let brk = brk.ok_or("break_if outside a `loop_until(.., |w, brk| ..)` body")?;
            done(d.break_if(c, brk))
        }
        "loop_carried" => {
            if a.len() < 2 {
                return Err("`loop_carried(head, [state,] |w[, st]| ..)`".into());
            }
            let head = str_of(it, &a[0], "loop head")?;
            let (state, f) = state_and_fn(a, 1)?;
            let cb = RefCell::new(Cb { fail: None });
            let last: RefCell<Option<Value<'a>>> = RefCell::new(None);
            let r = driver(cx)?.loop_carried(&head, &|d| {
                let mut cargs = vec![wgv()];
                cargs.extend(state.clone());
                let v = run_cb(it, &f, cargs, d, None, None, &mut cb.borrow_mut())?;
                let (st, cond) = match v {
                    Value::Nat(Nat::WgCond(c)) => (None, c),
                    Value::Tuple(mut xs) if matches!(xs.last(), Some(Value::Nat(Nat::WgCond(_)))) => {
                        let Some(Value::Nat(Nat::WgCond(c))) = xs.pop() else { unreachable!() };
                        (Some(if xs.len() == 1 { xs.remove(0) } else { Value::Tuple(xs) }), c)
                    }
                    _ => return Err("a loop_carried body returns Ok(more) or Ok((state, more)) with `more` from scmp_wg_uniform(..)".into()),
                };
                *last.borrow_mut() = st;
                Ok(cond)
            });
            let st = last.into_inner().or(state).unwrap_or(Value::Unit);
            return finish_cb(cb.into_inner(), r, st).map(Some);
        }
        "exit" => {
            need!(1);
            done_with(driver(cx)?.exit(&str_of(it, &a[0], "exit label")?), |e| {
                nat(Nat::Exit(e))
            })
        }
        "exit_if" => {
            need!(2);
            let (c, e) = (wgcond_of(&a[0])?, exit_of(&a[1])?);
            done(driver(cx)?.exit_if(c, &e))
        }
        "wave_exit_unless" => {
            need!(2);
            let (c, e) = (cond_of(&a[0])?, exit_of(&a[1])?);
            done(driver(cx)?.wave_exit_unless(c, &e))
        }
        "wg_exit_unless" => {
            need!(4);
            let (c, t, e, f) = (
                wgcond_of(&a[0])?,
                str_of(it, &a[1], "target")?,
                exit_of(&a[2])?,
                a[3].clone(),
            );
            let mut cb = Cb { fail: None };
            let r = driver(cx)?.wg_exit_unless(c, &t, &e, &mut |d| {
                run_cb(it, &f, vec![wgv()], d, None, None, &mut cb).map(|_| ())
            });
            return finish_cb(cb, r, Value::Unit).map(Some);
        }
        "end" => {
            need!(1);
            done(driver(cx)?.end(exit_of(&a[0])?))
        }
        "end_with" => {
            need!(2);
            let (e, f) = (exit_of(&a[0])?, a[1].clone());
            let mut cb = Cb { fail: None };
            let r = driver(cx)?.end_with(e, &mut |d| {
                run_cb(it, &f, vec![wgv()], d, None, None, &mut cb).map(|_| ())
            });
            return finish_cb(cb, r, Value::Unit).map(Some);
        }
        "handoff" => {
            need!(6);
            let (c, lbl, e, r, w, rd) = (
                cond_of(&a[0])?,
                str_of(it, &a[1], "reader label")?,
                exit_of(&a[2])?,
                region_of(it, &a[3])?,
                a[4].clone(),
                a[5].clone(),
            );
            let cb = RefCell::new(Cb { fail: None });
            let mut out = Value::Unit;
            let res = driver(cx)?.handoff(
                c,
                &lbl,
                e,
                r,
                &mut |d, r| {
                    run_cb(
                        it,
                        &w,
                        vec![wgv(), nat(Nat::Region(r))],
                        d,
                        None,
                        None,
                        &mut cb.borrow_mut(),
                    )
                    .map(|_| ())
                },
                &mut |d, r, ex| {
                    out = run_cb(
                        it,
                        &rd,
                        vec![wgv(), nat(Nat::Region(r)), nat(Nat::Exit(ex.clone()))],
                        d,
                        None,
                        None,
                        &mut cb.borrow_mut(),
                    )?;
                    Ok(())
                },
            );
            return finish_cb(cb.into_inner(), res, out).map(Some);
        }
        _ => return Ok(None),
    }))
}

fn transition_result<'a>(t: &Transition) -> Value<'a> {
    match *t {
        Transition::Ready(r) | Transition::Retire(r) => nat(Nat::Region(r)),
        Transition::Rotate(g)
        | Transition::Prime(g)
        | Transition::RetireCur(g)
        | Transition::Advance(g) => nat(Nat::Ring(g)),
    }
}
