//! The `fp8_gemm` generator as natives (a child module of [`super`]).
//!
//! Two layers over `peacemaker_isa::kernels::fp8_gemm`, adding no typestate surface:
//!
//! - Coarse: `Fp8Gemm::emit(Fp8Spec { scale, epi, arch })` and
//!   `Fp8Gemm::emit_module(arch, scale, [epi, ..])` return what the Rust functions return;
//!   `Spec::validate` stays the gate. `Fp8Scale::{Row, K128}` (the Rust `ActScale`) and
//!   `Fp8Epi::{Set, Add, GateUpSilu, GateUpSiluBf16, Qkv, Qkvza, QkvzaGdn}` are enum values.
//! - Blocks: `Fp8Gen::new(spec)` and the `fp8::*` natives of `fp8_gemm::blocks`. The shipped
//!   generator is an untyped-`Builder` kernel (LDS slots, barriers, branches and the loop are
//!   `Builder` calls, the epilogue a jump network), so a twin script is too: it builds with
//!   `Builder::new` (no `Workgroup`), declares the slots with `b.lds_slot`, and issues every LDS
//!   store/load, barrier, branch and the loop itself as `Builder` statements, checked by the
//!   builder's slot machine, wait ledger and loop fixpoint. The natives that touch LDS **return
//!   `Instruction`s** (`fp8::publish_a_insns`, `fp8::fragment_insns`, ...) for `b.ds_store` /
//!   `b.ds_load`; every other native (fetch, address set-up, WMMA, fold arithmetic, the whole
//!   epilogue) emits straight into the builder passed as its first argument (`b`, or `wg.isa()`).

use super::{
    arch_of, check_args, done, done_with, nat, nat_of, spec_value, str_of, u32_of, usize_of,
    with_b, Nat,
};
use crate::interp::{Cx, Interp, R};
use crate::value::Value;
use peacemaker_isa::insn::Instruction;
use peacemaker_isa::kernels::fp8_gemm::{self as fp8, blocks, ActScale, Epi, Spec};
use peacemaker_isa::Emitted;
use std::cell::RefCell;
use std::rc::Rc;

type Res<T> = Result<T, String>;

const FNS: &[&str] = &[
    "Fp8Gemm::emit",
    "Fp8Gemm::emit_module",
    "Fp8Gen::new",
    "fp8::label",
    "fp8::wait_all",
    "fp8::prologue_front",
    "fp8::prologue_lanes",
    "fp8::fetch_a",
    "fp8::publish_a_insns",
    "fp8::weight_base",
    "fp8::fetch_w",
    "fp8::fetch_next_ratio",
    "fp8::ratio_store_insn",
    "fp8::d_prepare",
    "fp8::d_store_insn",
    "fp8::d_restore",
    "fp8::fragment_insns",
    "fp8::mma_insn",
    "fp8::mma",
    "fp8::fold_row_group_load_insns",
    "fp8::fold_row_group_scale",
    "fp8::fold_k128_token_load_insns",
    "fp8::fold_k128_pair_load_insns",
    "fp8::fold_k128_pair_scale",
    "fp8::advance_block",
    "fp8::trips_setup",
    "fp8::trip_end",
    "fp8::epilogue",
];

pub(super) fn is_fn(name: &str) -> bool {
    FNS.contains(&name)
}

pub(super) fn is_type(n: &str) -> bool {
    matches!(n, "Fp8Scale" | "Fp8Epi" | "Fp8Spec" | "Fp8Gemm" | "Fp8Gen")
}

pub(super) fn is_struct(n: &str) -> bool {
    n == "Fp8Spec"
}

const ENUMS: &[(&str, &[&str])] = &[
    ("Fp8Scale", &["Row", "K128"]),
    (
        "Fp8Epi",
        &[
            "Set",
            "Add",
            "GateUpSilu",
            "GateUpSiluBf16",
            "Qkv",
            "Qkvza",
            "QkvzaGdn",
        ],
    ),
];

pub(super) fn assoc_const<'a>(ty: &str, item: &str) -> Option<Value<'a>> {
    if let Some((t, vs)) = ENUMS.iter().find(|(t, _)| *t == ty) {
        let v = vs.iter().find(|v| **v == item)?;
        return Some(Value::Enum(Rc::from(*t), Rc::from(*v), vec![]));
    }
    if ty == "fp8" {
        return Some(Value::str(match item {
            "BEGIN" => fp8::BEGIN,
            "LOOP" => fp8::LOOP,
            "LOOP_END" => fp8::LOOP_END,
            "EPILOGUE" => fp8::EPILOGUE,
            "END" => fp8::END,
            _ => return None,
        }));
    }
    None
}

// ---- Spec conversion ----------------------------------------------------------

fn variant_of<'a>(it: &Interp<'a>, v: &Value<'a>, ty: &str) -> Res<String> {
    match it.deref(v.clone()) {
        Value::Enum(t, var, p) if &*t == ty && p.is_empty() => Ok(var.to_string()),
        o => Err(format!("expected a {ty}, found {}", o.type_name())),
    }
}

fn scale_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<ActScale> {
    match variant_of(it, v, "Fp8Scale")?.as_str() {
        "Row" => Ok(ActScale::Row),
        "K128" => Ok(ActScale::K128),
        o => Err(format!("`Fp8Scale::{o}` is not a variant of Fp8Scale")),
    }
}

fn epi_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Epi> {
    match variant_of(it, v, "Fp8Epi")?.as_str() {
        "Set" => Ok(Epi::Set),
        "Add" => Ok(Epi::Add),
        "GateUpSilu" => Ok(Epi::GateUpSilu),
        "GateUpSiluBf16" => Ok(Epi::GateUpSiluBf16),
        "Qkv" => Ok(Epi::Qkv),
        "Qkvza" => Ok(Epi::Qkvza),
        "QkvzaGdn" => Ok(Epi::QkvzaGdn),
        o => Err(format!("`Fp8Epi::{o}` is not a variant of Fp8Epi")),
    }
}

fn enum_val<'a>(ty: &str, var: &str) -> Value<'a> {
    Value::Enum(Rc::from(ty), Rc::from(var), vec![])
}

type Fields<'a> = Vec<(Rc<str>, Value<'a>)>;

fn fields_of<'a>(it: &Interp<'a>, v: &Value<'a>, name: &str) -> Res<Fields<'a>> {
    match it.deref(v.clone()) {
        Value::Struct(n, fs) if &*n == name => Ok(fs),
        o => Err(format!(
            "expected a `{name} {{ .. }}`, found {}",
            o.type_name()
        )),
    }
}
fn field<'a>(fs: &mut Fields<'a>, name: &str, k: &str) -> Res<Value<'a>> {
    let i = fs
        .iter()
        .position(|(f, _)| &**f == k)
        .ok_or_else(|| format!("`{name}` is missing the field `{k}` (every field is required)"))?;
    Ok(fs.remove(i).1)
}

fn spec_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Spec> {
    let mut fs = fields_of(it, v, "Fp8Spec")?;
    let n = "Fp8Spec";
    let spec = Spec {
        arch: arch_of(it, &field(&mut fs, n, "arch")?)?,
        act_scale: scale_of(it, &field(&mut fs, n, "scale")?)?,
        epi: epi_of(it, &field(&mut fs, n, "epi")?)?,
    };
    if let Some((k, _)) = fs.first() {
        return Err(format!("`{n}` has no field `{k}`"));
    }
    Ok(spec)
}

fn spec_val<'a>(s: Spec) -> Value<'a> {
    let f = |k: &str, v: Value<'a>| (Rc::<str>::from(k), v);
    Value::Struct(
        "Fp8Spec".into(),
        vec![
            f("arch", nat(Nat::Arch(s.arch))),
            f("scale", enum_val("Fp8Scale", &format!("{:?}", s.act_scale))),
            f("epi", enum_val("Fp8Epi", &format!("{:?}", s.epi))),
        ],
    )
}

fn emitted_value<'a>(e: Emitted) -> Value<'a> {
    nat(Nat::Emitted(Rc::new(e)))
}

// ---- argument conversion ------------------------------------------------------

fn gen_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Rc<blocks::Gen>> {
    match nat_of(it, v) {
        Some(Nat::Fp8Gen(g)) => Ok(g),
        _ => Err("expected an Fp8Gen (`Fp8Gen::new(Fp8Spec { .. })?`)".into()),
    }
}
fn bool_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<bool> {
    match it.deref(v.clone()) {
        Value::Bool(b) => Ok(b),
        o => Err(format!("{what}: expected a bool, found {}", o.type_name())),
    }
}
fn insn_val<'a>(i: Instruction) -> Value<'a> {
    nat(Nat::Insn(Rc::new(i)))
}
fn insns_val<'a>(is: impl IntoIterator<Item = Instruction>) -> Value<'a> {
    Value::Array(is.into_iter().map(insn_val).collect())
}

// ---- natives --------------------------------------------------------------------

pub(super) fn call_fn<'a>(
    it: &Interp<'a>,
    name: &str,
    a: &[Value<'a>],
    cx: &mut Cx<'_>,
) -> Option<R<'a>> {
    is_fn(name).then(|| call(it, name, a, cx))
}

fn call<'a>(it: &Interp<'a>, name: &str, a: &[Value<'a>], cx: &mut Cx<'_>) -> R<'a> {
    let need = |n: usize| check_args(name, a, n);
    // `fn(b, g, ..)` natives that emit into the builder `b` (a `Builder` or `wg.isa()`).
    macro_rules! emit {
        ($n:expr, |$b:ident, $g:ident| $body:expr) => {{
            need($n)?;
            let $g = gen_of(it, &a[1])?;
            Ok(done(with_b(it, &a[0], cx, |$b| $body)?))
        }};
    }
    match name {
        "Fp8Gemm::emit" => {
            need(1)?;
            Ok(done_with(fp8::emit(spec_of(it, &a[0])?), emitted_value))
        }
        "Fp8Gemm::emit_module" => {
            need(3)?;
            let (arch, scale) = (arch_of(it, &a[0])?, scale_of(it, &a[1])?);
            let epis = match it.deref(a[2].clone()) {
                Value::Array(xs) => xs.iter().map(|x| epi_of(it, x)).collect::<Res<Vec<_>>>()?,
                o => {
                    return Err(format!(
                        "emit_module: the epilogues are an array of Fp8Epi, found {}",
                        o.type_name()
                    )
                    .into())
                }
            };
            let m = fp8::emit_module(arch, scale, &epis).and_then(|(emitted, text, proof)| {
                let proof =
                    serde_json::to_string(&proof).map_err(|e| format!("module proof: {e}"))?;
                Ok(Value::Tuple(vec![
                    Value::Array(emitted.into_iter().map(emitted_value).collect()),
                    Value::str(text),
                    Value::str(proof),
                ]))
            });
            Ok(done_with(m, |v| v))
        }
        "Fp8Gen::new" => {
            need(1)?;
            let g = blocks::Gen::new(spec_of(it, &a[0])?);
            Ok(done_with(g, |g| nat(Nat::Fp8Gen(Rc::new(g)))))
        }
        "fp8::label" => {
            need(2)?;
            let label = str_of(it, &a[1], "fp8::label")?;
            Ok(done(with_b(it, &a[0], cx, |b| b.label(&label))?))
        }
        "fp8::wait_all" => {
            need(1)?;
            Ok(done(with_b(it, &a[0], cx, blocks::wait_all)?))
        }
        "fp8::prologue_front" => emit!(2, |b, g| blocks::prologue_front(b, &g)),
        "fp8::prologue_lanes" => emit!(2, |b, g| blocks::prologue_lanes(b, &g)),
        "fp8::weight_base" => emit!(2, |b, g| blocks::weight_base(b, &g)),
        "fp8::advance_block" => emit!(2, |b, g| blocks::advance_block(b, &g)),
        "fp8::trips_setup" => emit!(2, |b, g| blocks::trips_setup(b, &g)),
        "fp8::trip_end" => emit!(2, |b, g| blocks::trip_end(b, &g)),
        "fp8::epilogue" => emit!(2, |b, g| blocks::epilogue(b, &g)),
        "fp8::d_restore" => emit!(2, |b, g| blocks::d_restore(b, &g)),
        "fp8::fetch_a" => {
            need(4)?;
            let g = gen_of(it, &a[1])?;
            let (kb, slab) = (
                u32_of(it, &a[2], "K128 offset")?,
                u32_of(it, &a[3], "slab")?,
            );
            Ok(done(with_b(it, &a[0], cx, |b| {
                blocks::fetch_a(b, &g, kb, slab)
            })?))
        }
        "fp8::fetch_w" => {
            need(3)?;
            let (g, unit) = (gen_of(it, &a[1])?, usize_of(it, &a[2], "weight unit")?);
            Ok(done(with_b(it, &a[0], cx, |b| {
                blocks::fetch_w(b, &g, unit)
            })?))
        }
        "fp8::fetch_next_ratio" => {
            need(3)?;
            let (g, kb) = (gen_of(it, &a[1])?, u32_of(it, &a[2], "next K128 offset")?);
            Ok(done(with_b(it, &a[0], cx, |b| {
                blocks::fetch_next_ratio(b, &g, kb)
            })?))
        }
        "fp8::d_prepare" => {
            need(3)?;
            let (g, kb) = (gen_of(it, &a[1])?, u32_of(it, &a[2], "next K128 offset")?);
            Ok(done(with_b(it, &a[0], cx, |b| {
                blocks::d_prepare(b, &g, kb)
            })?))
        }
        "fp8::mma" => {
            need(6)?;
            let g = gen_of(it, &a[1])?;
            let (unit, st, first, i) = (
                usize_of(it, &a[2], "unit")?,
                usize_of(it, &a[3], "K16 step")?,
                bool_of(it, &a[4], "first")?,
                usize_of(it, &a[5], "wmma index")?,
            );
            Ok(done(with_b(it, &a[0], cx, |b| {
                blocks::mma(b, &g, unit, st, first, i)
            })?))
        }
        "fp8::fold_row_group_scale" => {
            need(3)?;
            let (g, i) = (gen_of(it, &a[1])?, usize_of(it, &a[2], "row group")?);
            Ok(done(with_b(it, &a[0], cx, |b| {
                blocks::fold_row_group_scale(b, &g, i)
            })?))
        }
        "fp8::fold_k128_pair_scale" => {
            need(4)?;
            let g = gen_of(it, &a[1])?;
            let (i, e) = (usize_of(it, &a[2], "group")?, usize_of(it, &a[3], "pair")?);
            Ok(done(with_b(it, &a[0], cx, |b| {
                blocks::fold_k128_pair_scale(b, &g, i, e)
            })?))
        }
        "fp8::publish_a_insns" => {
            need(2)?;
            let (g, slot) = (gen_of(it, &a[0])?, usize_of(it, &a[1], "slot")?);
            Ok(done_with(blocks::publish_a_insns(&g, slot), insns_val))
        }
        "fp8::fragment_insns" => {
            need(3)?;
            let g = gen_of(it, &a[0])?;
            let (slot, step) = (usize_of(it, &a[1], "slot")?, usize_of(it, &a[2], "step")?);
            Ok(done_with(blocks::fragment_insns(&g, slot, step), insns_val))
        }
        "fp8::ratio_store_insn" => {
            need(2)?;
            let (g, plane) = (gen_of(it, &a[0])?, usize_of(it, &a[1], "plane")?);
            Ok(done_with(blocks::ratio_store_insn(&g, plane), insn_val))
        }
        "fp8::d_store_insn" => {
            need(2)?;
            let (g, plane) = (gen_of(it, &a[0])?, usize_of(it, &a[1], "plane")?);
            Ok(done_with(blocks::d_store_insn(&g, plane), insn_val))
        }
        "fp8::mma_insn" => {
            need(5)?;
            let g = gen_of(it, &a[0])?;
            let (unit, st, first, i) = (
                usize_of(it, &a[1], "unit")?,
                usize_of(it, &a[2], "K16 step")?,
                bool_of(it, &a[3], "first")?,
                usize_of(it, &a[4], "wmma index")?,
            );
            Ok(done_with(
                blocks::mma_insn(&g, unit, st, first, i),
                insn_val,
            ))
        }
        "fp8::fold_row_group_load_insns" => {
            need(3)?;
            let g = gen_of(it, &a[0])?;
            let (plane, i) = (usize_of(it, &a[1], "plane")?, usize_of(it, &a[2], "group")?);
            Ok(done_with(
                blocks::fold_row_group_load_insns(&g, plane, i),
                insns_val,
            ))
        }
        "fp8::fold_k128_token_load_insns" => {
            need(2)?;
            let (g, plane) = (gen_of(it, &a[0])?, usize_of(it, &a[1], "plane")?);
            Ok(done_with(
                blocks::fold_k128_token_load_insns(&g, plane),
                insns_val,
            ))
        }
        "fp8::fold_k128_pair_load_insns" => {
            need(4)?;
            let g = gen_of(it, &a[0])?;
            let (plane, i, e) = (
                usize_of(it, &a[1], "plane")?,
                usize_of(it, &a[2], "group")?,
                usize_of(it, &a[3], "pair")?,
            );
            Ok(done_with(
                blocks::fold_k128_pair_load_insns(&g, plane, i, e),
                insns_val,
            ))
        }
        n => Err(format!("native function `{n}` is not implemented").into()),
    }
}

/// Methods of an `Fp8Gen`: `plan()`, `kernel_spec()`, `spec()`, `slot(name)`.
pub(super) fn method<'a>(
    it: &Interp<'a>,
    g: &Rc<blocks::Gen>,
    name: &str,
    a: &[Value<'a>],
) -> R<'a, Option<Value<'a>>> {
    let need = |n: usize| check_args(name, a, n);
    Ok(match name {
        "plan" => {
            need(0)?;
            Some(done_with(g.plan(), |p| {
                nat(Nat::Plan(Rc::new(RefCell::new(p))))
            }))
        }
        "kernel_spec" => {
            need(0)?;
            Some(spec_value(&g.kernel_spec()))
        }
        "spec" => {
            need(0)?;
            Some(spec_val(g.spec()))
        }
        "slot" => {
            need(1)?;
            Some(Value::option(g.slot(&str_of(it, &a[0], "slot")?).map(
                |(base, bytes)| Value::Tuple(vec![Value::u32(base), Value::u32(bytes)]),
            )))
        }
        _ => None,
    })
}
