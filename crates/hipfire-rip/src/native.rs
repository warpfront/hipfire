//! Native vocabulary of the evaluator: the generic builder objects
//! (`KernelSpec`, `KernargLayout`, `RegPlan`, `Builder`, `Instruction`), the
//! shared kernel helpers of `hipfire_isa::kernels::common`, the `Chain`
//! emission builtin, and the typed-core surface (`Workgroup` / `Wave`
//! methods, `Ring`, `ready` / `retire` / ...) lowered onto the sealed
//! runtime driver. Everything else a script calls is evaluated script code.

use crate::driver::DynDriver;
use crate::interp::{Cx, Ctl, Interp, R};
use crate::value::*;
use hipfire_isa::insn::{Instruction, MemoryClass};
use hipfire_isa::kernels::common;
use hipfire_isa::ledger::Counter;
use hipfire_isa::reg::{Kind as RegKind, Live, RegRef};
use hipfire_isa::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
use peacemaker_author::runtime::{Arrived, Cond, Exit, Phase, Place, RegionId, RingId, Transition, WgCond};
use std::cell::RefCell;
use std::rc::Rc;

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
            other => other.type_name().to_string(),
        }
    }
}

// ---- names ----------------------------------------------------------------------

const COMMON_FNS: &[&str] = &[
    "v", "vr", "s", "sr", "lit", "op", "mem", "sop", "smem", "srd_tail", "add64", "add64_imm", "smem_off", "s_add_u32", "s_addc_u32", "s_add_i32", "bload", "bstore_b128", "gather_offset",
    "ready", "retire", "rotate", "prime", "retire_cur", "join",
];
const ASSOC_FNS: &[&str] = &[
    "Builder::new", "RegPlan::new", "KernargLayout::new", "Instruction::new", "Chain::new", "Live::Between", "Ring::new", "Workgroup::new", "String::new", "String::from", "Vec::new", "Vec::with_capacity",
    "cmp::max", "cmp::min", "mem::swap",
];

pub fn is_fn(name: &str) -> bool {
    if COMMON_FNS.contains(&name) || ASSOC_FNS.contains(&name) {
        return true;
    }
    matches!(name.split_once("::"), Some((t, "from" | "try_from" | "max" | "min")) if IntTy::parse(t).is_some())
}

pub fn is_type(n: &str) -> bool {
    matches!(n, "Arch" | "MemoryClass" | "Counter" | "Live" | "KernelSpec" | "KernargLayout" | "RegPlan" | "Builder" | "Instruction" | "Chain" | "Ring" | "Workgroup" | "Option" | "Result" | "String" | "Vec" | "Phase" | "f32" | "f64" | "bool" | "char" | "str")
        || IntTy::parse(n).is_some()
}

pub fn is_struct(n: &str) -> bool {
    matches!(n, "KernelSpec" | "GatherTemps")
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
        (t, "MAX") if IntTy::parse(t).is_some() => IntTy::parse(t).map(|t| Value::Int(t.max(), Some(t))),
        (t, "MIN") if IntTy::parse(t).is_some() => IntTy::parse(t).map(|t| Value::Int(t.min(), Some(t))),
        (t, "BITS") if IntTy::parse(t).is_some() => IntTy::parse(t).map(|t| Value::Int(i128::from(t.bits()), Some(IntTy::U32))),
        _ => None,
    }
}

// ---- argument conversion --------------------------------------------------------

type Res<T> = Result<T, String>;

fn int<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<i128> {
    match it.deref(v.clone()) {
        Value::Int(i, _) => Ok(i),
        o => Err(format!("{what}: expected an integer, found {}", o.type_name())),
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
        o => Err(format!("{what}: expected a string, found {}", o.type_name())),
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
        o => Err(format!("{what}: expected an array of registers, found {}", o.type_name())),
    }
}
fn reg_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<RegRef> {
    match nat_of(it, v) {
        Some(Nat::Reg(r)) => Ok(r),
        _ => Err(format!("{what}: expected a register (v(..), s(..), vr(..), sr(..))")),
    }
}
fn u8s_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<Vec<u8>> {
    match it.deref(v.clone()) {
        Value::Array(xs) => xs.iter().map(|x| u8_of(it, x, what)).collect(),
        o => Err(format!("{what}: expected an array of register numbers, found {}", o.type_name())),
    }
}
fn insn_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Instruction> {
    match nat_of(it, v) {
        Some(Nat::Insn(i)) => Ok((*i).clone()),
        _ => Err("expected an Instruction".into()),
    }
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
        o => Err(format!("{what}: expected Option<u8>, found {}", o.type_name())),
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
fn with_b<'a, T>(it: &Interp<'a>, v: &Value<'a>, cx: &mut Cx<'_>, f: impl FnOnce(&mut Builder) -> T) -> R<'a, T> {
    match it.deref(v.clone()) {
        Value::Nat(Nat::Builder(cell)) => {
            let mut g = cell.borrow_mut();
            let b = g.as_mut().ok_or("the builder is owned by the active Workgroup scope: use wg.isa()")?;
            Ok(f(b))
        }
        Value::Nat(Nat::Isa) => {
            let d = cx.d.as_deref_mut().ok_or("wg.isa() outside the Workgroup scope")?;
            Ok(f(d.isa()))
        }
        o => Err(format!("expected a builder (`Builder::new(..)` or `wg.isa()`), found {}", o.type_name()).into()),
    }
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
            f("system_sgpr_workgroup_id_y", Value::Bool(s.system_sgpr_workgroup_id_y)),
            f("workgroup_size", Value::int(i128::from(s.workgroup_size), IntTy::U16)),
            f("group_segment_fixed_size", Value::u32(s.group_segment_fixed_size)),
            f("wave32", Value::Bool(s.wave32)),
            f("cu_mode", Value::Bool(s.cu_mode)),
        ],
    )
}

fn spec_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<KernelSpec> {
    let Value::Struct(n, fs) = it.deref(v.clone()) else { return Err("expected a KernelSpec { .. }".into()) };
    if &*n != "KernelSpec" {
        return Err(format!("expected a KernelSpec, found {n}"));
    }
    let get = |k: &str| fs.iter().find(|(f, _)| &**f == k).map(|(_, v)| v.clone()).ok_or_else(|| format!("KernelSpec is missing `{k}`"));
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
        group_segment_fixed_size: u32_of(it, &get("group_segment_fixed_size")?, "group_segment_fixed_size")?,
        wave32: boolean("wave32")?,
        cu_mode: boolean("cu_mode")?,
    })
}

pub fn field<'a>(it: &Interp<'a>, n: &Nat, name: &str, cx: &mut Cx<'_>) -> R<'a, Option<Value<'a>>> {
    Ok(match (n, name) {
        (Nat::Builder(_) | Nat::Isa, "spec") => Some(with_b(it, &Value::Nat(n.clone()), cx, |b| spec_value(&b.spec))?),
        (Nat::Builder(_) | Nat::Isa, "regs") => Some(with_b(it, &Value::Nat(n.clone()), cx, |b| nat(Nat::Plan(Rc::new(RefCell::new(b.regs.clone())))))?),
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
        k => return Err(format!("Chain: unsupported matrix kind `{k}` (supported: f16, bf16)")),
    };
    if c.width != 8 {
        return Err(format!("Chain: the accumulator is 8 VGPRs, not {}", c.width));
    }
    if !matches!(aw, 4 | 8) || !matches!(bw, 4 | 8) {
        return Err(format!("Chain: operand widths must be 4 or 8 VGPRs, got {aw} and {bw}"));
    }
    let acc = vtxt(c.dst, c.width);
    common::op(b, format!("{mn} {acc}, {}, {}, {acc}", vtxt(a, aw), vtxt(bm, bw)), &[common::vr(c.dst, c.width)], &[common::vr(a, aw), common::vr(bm, bw), common::vr(c.dst, c.width)])
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
        o => Err(format!("expected an LDS region or ring, found {}", o.type_name())),
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
            Ok(done(with_b(it, &a[0], cx, |b| common::op(b, text, &defs, &uses))?))
        }
        "mem" => {
            need(5)?;
            let (text, defs, uses) = (str_of(it, &a[1], "mem text")?, regs_of(it, &a[2], "mem defs")?, regs_of(it, &a[3], "mem uses")?);
            let Some(Nat::Mem(class)) = nat_of(it, &a[4]) else { return Err("mem: the last argument is a MemoryClass".into()) };
            Ok(done(with_b(it, &a[0], cx, |b| common::mem(b, text, &defs, &uses, class))?))
        }
        "sop" => {
            need(4)?;
            let (text, defs, uses) = (str_of(it, &a[1], "sop text")?, u8s_of(it, &a[2], "sop defs")?, u8s_of(it, &a[3], "sop uses")?);
            Ok(done(with_b(it, &a[0], cx, |b| common::sop(b, text, &defs, &uses))?))
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
            need(2)?;
            let (x, y) = (region_of(it, &a[0])?, region_of(it, &a[1])?);
            let d = driver(cx)?;
            let n = format!("ring@{}", d.position());
            Ok(Value::Nat(Nat::Ring(d.ring(&n, x, y)?)))
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
            let r = region_of(it, &a[0])?;
            Ok(nat(Nat::Trans(if name == "ready" { Transition::Ready(r) } else { Transition::Retire(r) })))
        }
        "rotate" | "prime" | "retire_cur" => {
            if a.is_empty() || a.len() > 2 {
                return Err(format!("`{name}` takes a ring (and its Drained token)").into());
            }
            let g = ring_of(it, &a[0])?;
            Ok(nat(Nat::Trans(match name {
                "rotate" => Transition::Rotate(g),
                "prime" => Transition::Prime(g),
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

fn run_cb<'a>(it: &Interp<'a>, f: &Value<'a>, args: Vec<Value<'a>>, d: &mut dyn DynDriver, fwd: Option<&mut peacemaker_author::runtime::Forward<Builder>>, brk: Option<&peacemaker_author::runtime::Breaks<Builder>>, cb: &mut Cb<'a>) -> Res<Value<'a>> {
    let mut cx = Cx { d: Some(d), fwd, brk };
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

pub fn method<'a>(it: &Interp<'a>, n: &Nat, name: &str, targs: &[String], args: Vec<Value<'a>>, cx: &mut Cx<'_>) -> R<'a, Option<Value<'a>>> {
    let a = &args;
    let need = |k: usize| check_args(name, a, k);
    Ok(match n {
        Nat::Arch(arch) => match name {
            "gfx12" => Some(Value::Bool(arch.gfx12())),
            "name" => Some(Value::str(arch.name())),
            "buffer_offset_max" => Some(Value::u32(arch.buffer_offset_max())),
            _ => None,
        },
        Nat::Insn(i) => match name {
            "memory" => {
                need(1)?;
                let Some(Nat::Mem(c)) = nat_of(it, &a[0]) else { return Err("memory(..) takes a MemoryClass".into()) };
                Some(nat(Nat::Insn(Rc::new((**i).clone().memory(c)))))
            }
            _ => None,
        },
        Nat::Kernargs(k) => match name {
            "pointer" => {
                need(2)?;
                Some(nat(Nat::Kernargs(k.clone().pointer(&str_of(it, &a[0], "pointer")?, u32_of(it, &a[1], "pointer")?))))
            }
            "hidden" => {
                need(4)?;
                Some(nat(Nat::Kernargs(k.clone().hidden(&str_of(it, &a[0], "hidden")?, u32_of(it, &a[1], "hidden")?, u32_of(it, &a[2], "hidden")?, &str_of(it, &a[3], "hidden")?))))
            }
            "validate" => Some(done(k.validate())),
            _ => None,
        },
        Nat::Plan(p) => match name {
            "s" | "v" => {
                need(3)?;
                let width = match targs.first() {
                    Some(t) => t.parse::<u8>().ok().or_else(|| it.const_int(t).and_then(|c| u8::try_from(c).ok())).ok_or_else(|| format!("`{name}::<{t}>`: the width is not a constant"))?,
                    None => return Err(format!("`{name}` needs a width: `{name}::<N>(..)`").into()),
                };
                let (nm, base) = (str_of(it, &a[0], "register name")?, u8_of(it, &a[1], "register base")?);
                let Some(Nat::Live(live)) = nat_of(it, &a[2]) else { return Err("the last argument is a Live range".into()) };
                let kind = if name == "s" { RegKind::S } else { RegKind::V };
                let r = p.borrow_mut().add_range(&nm, kind, base, width, live);
                Some(done_with(r, |()| nat(Nat::Reg(RegRef { kind, base, len: width }))))
            }
            "next_free_vgpr" => Some(Value::int(i128::from(p.borrow().next_free_vgpr()), IntTy::U16)),
            "next_free_sgpr" => Some(Value::int(i128::from(p.borrow().next_free_sgpr()), IntTy::U16)),
            "scratch_pool" => {
                need(2)?;
                Some(done(p.borrow_mut().scratch_pool(u8_of(it, &a[0], "scratch_pool")?, u8_of(it, &a[1], "scratch_pool")?)))
            }
            _ => None,
        },
        Nat::Builder(_) | Nat::Isa => match name {
            "enable_delay_alu" => {
                with_b(it, &Value::Nat(n.clone()), cx, |b| b.enable_delay_alu())?;
                Some(unit())
            }
            "wait" => {
                need(2)?;
                let Some(Nat::Counter(c)) = nat_of(it, &a[0]) else { return Err("wait(Counter, n)".into()) };
                let k = u8_of(it, &a[1], "wait count")?;
                Some(done(with_b(it, &Value::Nat(n.clone()), cx, |b| b.wait(c, k))?))
            }
            "finish" => {
                let Nat::Builder(cell) = n else { return Err("finish() is the builder's, not wg.isa()'s".into()) };
                let b = cell.borrow_mut().take().ok_or("the builder was already finished")?;
                Some(done_with(b.finish(), |e| nat(Nat::Emitted(Rc::new(e)))))
            }
            "position" => Some(Value::usize(with_b(it, &Value::Nat(n.clone()), cx, |b| b.program().instructions.len())?)),
            _ => None,
        },
        Nat::Chain(c) => match name {
            "step" => {
                need(5)?;
                let (x, aw, bm, bw) = (u8_of(it, &a[1], "step")?, u8_of(it, &a[2], "step")?, u8_of(it, &a[3], "step")?, u8_of(it, &a[4], "step")?);
                Some(done(with_b(it, &a[0], cx, |b| chain_step(b, c, x, aw, bm, bw))?))
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
                    let Some(Nat::Cond(c)) = nat_of(it, &a[1]) else { return Err(format!("`{name}` takes a Cond from scmp(..)").into()) };
                    let l = label_arg(2)?;
                    Some(fwd_op(cx, |d, f| if name == "branch_if" { d.fwd_branch_if(f, c, &l) } else { d.fwd_branch_unless(f, c, &l) })?)
                }
                "goto" | "place" => {
                    need(2)?;
                    let l = label_arg(1)?;
                    Some(fwd_op(cx, |d, f| if name == "goto" { d.fwd_goto(f, &l) } else { d.fwd_place(f, &l) })?)
                }
                _ => None,
            }
        }
        Nat::Wg => return wg_method(it, name, args, cx),
        _ => None,
    })
}

fn fwd_op<'a>(cx: &mut Cx<'_>, f: impl FnOnce(&mut dyn DynDriver, &mut peacemaker_author::runtime::Forward<Builder>) -> Res<()>) -> R<'a> {
    let Cx { d, fwd, .. } = cx;
    let d = d.as_deref_mut().ok_or("forward handle used outside the Workgroup scope")?;
    let fwd = fwd.as_deref_mut().ok_or("forward handle used outside its `forward(|w, f| ..)` block")?;
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

fn wg_method<'a>(it: &Interp<'a>, name: &str, args: Vec<Value<'a>>, cx: &mut Cx<'_>) -> R<'a, Option<Value<'a>>> {
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
            let (nm, base, len) = (str_of(it, &a[0], "lds name")?, u32_of(it, &a[1], "lds base")?, u32_of(it, &a[2], "lds len")?);
            done_with(driver(cx)?.lds(&nm, base, len), |r| nat(Nat::Region(r)))
        }
        "relayout" => done(driver(cx)?.relayout()),
        "join" => {
            let parts: Vec<RegionId> = if a.len() == 1 { handles(it, &a[0]).iter().map(|v| region_of(it, v)).collect::<Res<_>>()? } else { a.iter().map(|v| region_of(it, v)).collect::<Res<_>>()? };
            done_with(driver(cx)?.join(&parts), |r| nat(Nat::Region(r)))
        }
        "split" => {
            need!(2);
            let (r, first) = (region_of(it, &a[0])?, usize_of(it, &a[1], "split")?);
            done_with(driver(cx)?.split(r, first), |(x, y)| Value::Tuple(vec![nat(Nat::Region(x)), nat(Nat::Region(y))]))
        }
        "ring" => {
            need!(3);
            let (nm, x, y) = (str_of(it, &a[0], "ring name")?, region_of(it, &a[1])?, region_of(it, &a[2])?);
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
            let (p, insn) = (place_of(it, &a[0])?, insn_of(it, &a[1])?);
            done_with(driver(cx)?.ds_store(p, insn), |()| Value::Tuple(vec![place_val(p), pend(p)]))
        }
        "ds_store2" => {
            need!(3);
            let (x, y, insn) = (place_of(it, &a[0])?, place_of(it, &a[1])?, insn_of(it, &a[2])?);
            done_with(driver(cx)?.ds_store2(x, y, insn), |()| Value::Tuple(vec![Value::Tuple(vec![place_val(x), pend(x)]), Value::Tuple(vec![place_val(y), pend(y)])]))
        }
        "ds_load" => {
            need!(2);
            let (r, insn) = (region_of(it, &a[0])?, insn_of(it, &a[1])?);
            done(driver(cx)?.ds_load(r, insn))
        }
        "ds_load_cur" => {
            need!(2);
            let (g, insn) = (ring_of(it, &a[0])?, insn_of(it, &a[1])?);
            done(driver(cx)?.ds_load_cur(g, insn))
        }
        "wait" => {
            need!(1);
            let p = place_of(it, &a[0])?;
            done_with(driver(cx)?.wait(p), |()| nat(Nat::Drained(p)))
        }
        "wait_all" => {
            need!(1);
            let ps: Vec<Place> = handles(it, &a[0]).iter().map(|v| place_of(it, v)).collect::<Res<_>>()?;
            done_with(driver(cx)?.wait_all(&ps), |()| Value::Tuple(ps.iter().map(|p| nat(Nat::Drained(*p))).collect()))
        }
        "barrier" => {
            need!(1);
            let ts = trans_of(it, &a[0])?;
            done_with(driver(cx)?.barrier(&ts), |()| Value::Tuple(ts.iter().map(transition_result).collect()))
        }
        "signal" => {
            need!(1);
            let ts = trans_of(it, &a[0])?;
            done_with(driver(cx)?.signal(&ts), |arr| nat(Nat::Arrived(arr)))
        }
        "wait_arrived" => {
            need!(1);
            let Some(Nat::Arrived(arr)) = nat_of(it, &a[0]) else { return Err("wait_arrived takes the Arrived from signal(..)".into()) };
            done(driver(cx)?.wait_arrived(arr))
        }
        "scmp" => {
            need!(1);
            let insn = insn_of(it, &a[0])?;
            done_with(driver(cx)?.scmp(insn), |c| nat(Nat::Cond(c)))
        }
        "scmp_wg_uniform" => {
            need!(1);
            let insn = insn_of(it, &a[0])?;
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
                out = run_cb(it, &f, vec![wgv(), nat(Nat::Fwd)], d, Some(fw), None, &mut cb)?;
                Ok(())
            });
            return finish_cb(cb, r, out).map(Some);
        }
        "if_else" => {
            need!(5);
            let (c, el, jn) = (cond_of(&a[0])?, str_of(it, &a[1], "else label")?, str_of(it, &a[2], "join label")?);
            let (t, e) = (a[3].clone(), a[4].clone());
            let mut cb = Cb { fail: None };
            let cb_cell = RefCell::new(&mut cb);
            let r = driver(cx)?.if_else(
                c,
                &el,
                &jn,
                &mut |d| run_cb(it, &t, vec![wgv()], d, None, None, &mut cb_cell.borrow_mut()).map(|_| ()),
                &mut |d| run_cb(it, &e, vec![wgv()], d, None, None, &mut cb_cell.borrow_mut()).map(|_| ()),
            );
            drop(cb_cell);
            return finish_cb(cb, r, Value::Unit).map(Some);
        }
        "loop_until" => {
            need!(3);
            let (head, exit, f) = (str_of(it, &a[0], "loop head")?, str_of(it, &a[1], "loop exit")?, a[2].clone());
            let cb = RefCell::new(Cb { fail: None });
            let r = driver(cx)?.loop_until(&head, &exit, &|d, brk| run_cb(it, &f, vec![wgv(), nat(Nat::Brk)], d, None, Some(brk), &mut cb.borrow_mut()).map(|_| ()));
            return finish_cb(cb.into_inner(), r, Value::Unit).map(Some);
        }
        "break_if" => {
            need!(2);
            let c = cond_of(&a[0])?;
            let Cx { d, brk, .. } = cx;
            let d = d.as_deref_mut().ok_or("break_if outside the Workgroup scope")?;
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
            done_with(driver(cx)?.exit(&str_of(it, &a[0], "exit label")?), |e| nat(Nat::Exit(e)))
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
            let (c, t, e, f) = (wgcond_of(&a[0])?, str_of(it, &a[1], "target")?, exit_of(&a[2])?, a[3].clone());
            let mut cb = Cb { fail: None };
            let r = driver(cx)?.wg_exit_unless(c, &t, &e, &mut |d| run_cb(it, &f, vec![wgv()], d, None, None, &mut cb).map(|_| ()));
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
            let r = driver(cx)?.end_with(e, &mut |d| run_cb(it, &f, vec![wgv()], d, None, None, &mut cb).map(|_| ()));
            return finish_cb(cb, r, Value::Unit).map(Some);
        }
        "handoff" => {
            need!(6);
            let (c, lbl, e, r, w, rd) = (cond_of(&a[0])?, str_of(it, &a[1], "reader label")?, exit_of(&a[2])?, region_of(it, &a[3])?, a[4].clone(), a[5].clone());
            let cb = RefCell::new(Cb { fail: None });
            let mut out = Value::Unit;
            let res = driver(cx)?.handoff(
                c,
                &lbl,
                e,
                r,
                &mut |d, r| run_cb(it, &w, vec![wgv(), nat(Nat::Region(r))], d, None, None, &mut cb.borrow_mut()).map(|_| ()),
                &mut |d, r, ex| {
                    out = run_cb(it, &rd, vec![wgv(), nat(Nat::Region(r)), nat(Nat::Exit(ex.clone()))], d, None, None, &mut cb.borrow_mut())?;
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
        Transition::Rotate(g) | Transition::Prime(g) | Transition::RetireCur(g) => nat(Nat::Ring(g)),
    }
}
