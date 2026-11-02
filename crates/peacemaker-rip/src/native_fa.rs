//! The `fa_prefill` generators as natives (a child module of [`super`]).
//!
//! Two layers over `peacemaker_isa::kernels::fa_prefill`, adding no typestate surface:
//!
//! - Coarse: `FaPrefill::emit(arch, spec)`, `FaPrefill::emit_preconvert(arch, spec)`,
//!   `FaPrefill::emit_module(arch, spec)` and `FaPrefill::emit_fp8_module(arch, spec)` return what
//!   the Rust functions return; the Rust gate (gfx1201, exactly 16/2/256) stays in force.
//!   `FaKv::{Q8_0, Fp8E4M3}` is an enum value and `FaSpec { heads, kv_heads, head_dim, kv }` the
//!   struct literal.
//! - Blocks: `FaGen::new(arch, spec)` and the `fa::*` natives of `fa_prefill::blocks`. The shipped
//!   generators are untyped-`Builder` kernels (LDS slots, barriers, branches and the tile loop are
//!   `Builder` calls), so a twin script is too: it builds with `Builder::new` (no `Workgroup`),
//!   declares the slots with `b.lds_slot` and issues every barrier, branch label and the loop itself
//!   as `Builder` statements, checked by the builder's slot machine, wait ledger and loop fixpoint.
//!   Every `fa::*` native emits straight into the builder passed as its first argument.

use super::{
    arch_of, check_args, done, done_with, nat, nat_of, spec_value, str_of, u32_of, with_b, Nat,
};
use crate::interp::{Cx, Interp, R};
use crate::value::Value;
use peacemaker_isa::kernels::fa_prefill::{self as fa, blocks, FaKvCodec, FaPrefillSpec};
use peacemaker_isa::Emitted;
use std::cell::RefCell;
use std::rc::Rc;

type Res<T> = Result<T, String>;

const FNS: &[&str] = &[
    "FaPrefill::emit",
    "FaPrefill::emit_preconvert",
    "FaPrefill::emit_module",
    "FaPrefill::emit_fp8_module",
    "FaGen::new",
    "fa::label",
    "fa::prologue",
    "fa::tile_setup",
    "fa::fill",
    "fa::fill_k",
    "fa::fill_v",
    "fa::qk",
    "fa::softmax_sub",
    "fa::skip_if_helper",
    "fa::next_tile",
    "fa::epilogue",
    "fa::preconvert",
];

pub(super) fn is_fn(name: &str) -> bool {
    FNS.contains(&name)
}

pub(super) fn is_type(n: &str) -> bool {
    matches!(n, "FaKv" | "FaSpec" | "FaPrefill" | "FaGen")
}

pub(super) fn is_struct(n: &str) -> bool {
    n == "FaSpec"
}

pub(super) fn assoc_const<'a>(ty: &str, item: &str) -> Option<Value<'a>> {
    if ty == "FaKv" {
        let v = ["Q8_0", "Fp8E4M3"].into_iter().find(|v| *v == item)?;
        return Some(Value::Enum(Rc::from("FaKv"), Rc::from(v), vec![]));
    }
    if ty == "fa" {
        return Some(Value::str(match item {
            "LOOP" => blocks::LOOP,
            "FILL_DONE" => blocks::FILL_DONE,
            "KFILL_DONE" => blocks::KFILL_DONE,
            "KREADY" => blocks::KREADY,
            "QK" => blocks::QK,
            "QK_DONE" => blocks::QK_DONE,
            "VFILL_DONE" => blocks::VFILL_DONE,
            "VREADY" => blocks::VREADY,
            "TURN" => blocks::TURN,
            "EPI" => blocks::EPI,
            "END" => blocks::END,
            "PRECONVERT_END" => blocks::PRECONVERT_END,
            _ => return None,
        }));
    }
    None
}

// ---- Spec conversion ----------------------------------------------------------

fn kv_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<FaKvCodec> {
    match it.deref(v.clone()) {
        Value::Enum(t, var, p) if &*t == "FaKv" && p.is_empty() => match &*var {
            "Q8_0" => Ok(FaKvCodec::Q8_0),
            "Fp8E4M3" => Ok(FaKvCodec::Fp8E4M3),
            o => Err(format!("`FaKv::{o}` is not a variant of FaKv")),
        },
        o => Err(format!("expected a FaKv, found {}", o.type_name())),
    }
}

fn spec_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<FaPrefillSpec> {
    let mut fs = match it.deref(v.clone()) {
        Value::Struct(n, fs) if &*n == "FaSpec" => fs,
        o => {
            return Err(format!(
                "expected a `FaSpec {{ .. }}`, found {}",
                o.type_name()
            ))
        }
    };
    let mut take = |k: &str| -> Res<Value<'a>> {
        let i = fs.iter().position(|(f, _)| &**f == k).ok_or_else(|| {
            format!("`FaSpec` is missing the field `{k}` (every field is required)")
        })?;
        Ok(fs.remove(i).1)
    };
    let spec = FaPrefillSpec {
        heads: u32_of(it, &take("heads")?, "heads")?,
        kv_heads: u32_of(it, &take("kv_heads")?, "kv_heads")?,
        head_dim: u32_of(it, &take("head_dim")?, "head_dim")?,
        kv: kv_of(it, &take("kv")?)?,
    };
    if let Some((k, _)) = fs.first() {
        return Err(format!("`FaSpec` has no field `{k}`"));
    }
    Ok(spec)
}

fn spec_val<'a>(s: FaPrefillSpec) -> Value<'a> {
    let f = |k: &str, v: Value<'a>| (Rc::<str>::from(k), v);
    Value::Struct(
        "FaSpec".into(),
        vec![
            f("heads", Value::u32(s.heads)),
            f("kv_heads", Value::u32(s.kv_heads)),
            f("head_dim", Value::u32(s.head_dim)),
            f(
                "kv",
                Value::Enum(Rc::from("FaKv"), Rc::from(format!("{:?}", s.kv)), vec![]),
            ),
        ],
    )
}

fn emitted_value<'a>(e: Emitted) -> Value<'a> {
    nat(Nat::Emitted(Rc::new(e)))
}

fn gen_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Rc<blocks::Gen>> {
    match nat_of(it, v) {
        Some(Nat::FaGen(g)) => Ok(g),
        _ => Err("expected an FaGen (`FaGen::new(arch, FaSpec { .. })?`)".into()),
    }
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

fn module_value<'a>(
    m: Result<
        (
            Vec<Emitted>,
            String,
            peacemaker_isa::kernels::iu4_gemm::ModuleProof,
        ),
        String,
    >,
) -> Result<Value<'a>, String> {
    let (emitted, text, proof) = m?;
    let proof = serde_json::to_string(&proof).map_err(|e| format!("module proof: {e}"))?;
    Ok(Value::Tuple(vec![
        Value::Array(emitted.into_iter().map(emitted_value).collect()),
        Value::str(text),
        Value::str(proof),
    ]))
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
        "FaPrefill::emit" => {
            need(2)?;
            let (arch, spec) = (arch_of(it, &a[0])?, spec_of(it, &a[1])?);
            Ok(done_with(fa::emit(arch, &spec), emitted_value))
        }
        "FaPrefill::emit_preconvert" => {
            need(2)?;
            let (arch, spec) = (arch_of(it, &a[0])?, spec_of(it, &a[1])?);
            Ok(done_with(fa::emit_preconvert(arch, &spec), emitted_value))
        }
        "FaPrefill::emit_module" | "FaPrefill::emit_fp8_module" => {
            need(2)?;
            let (arch, spec) = (arch_of(it, &a[0])?, spec_of(it, &a[1])?);
            let m = if name == "FaPrefill::emit_module" {
                fa::emit_module(arch, &spec)
            } else {
                fa::emit_fp8_module(arch, &spec)
            };
            Ok(done_with(module_value(m), |v| v))
        }
        "FaGen::new" => {
            need(2)?;
            let (arch, spec) = (arch_of(it, &a[0])?, spec_of(it, &a[1])?);
            let g = blocks::Gen::new(arch, spec);
            Ok(done_with(g, |g| nat(Nat::FaGen(Rc::new(g)))))
        }
        "fa::label" => {
            need(2)?;
            let label = str_of(it, &a[1], "fa::label")?;
            Ok(done(with_b(it, &a[0], cx, |b| b.label(&label))?))
        }
        "fa::prologue" => emit!(2, |b, g| blocks::prologue(b, &g)),
        "fa::tile_setup" => emit!(2, |b, g| blocks::tile_setup(b, &g)),
        "fa::fill" => emit!(2, |b, g| blocks::fill(b, &g)),
        "fa::fill_k" => emit!(2, |b, g| blocks::fill_k(b, &g)),
        "fa::fill_v" => emit!(2, |b, g| blocks::fill_v(b, &g)),
        "fa::qk" => emit!(2, |b, g| blocks::qk(b, &g)),
        "fa::epilogue" => emit!(2, |b, g| blocks::epilogue(b, &g)),
        "fa::preconvert" => emit!(2, |b, g| blocks::preconvert(b, &g)),
        "fa::softmax_sub" => {
            need(3)?;
            let g = gen_of(it, &a[1])?;
            let sub = u32_of(it, &a[2], "subtile")?;
            let sub = u8::try_from(sub).map_err(|_| format!("softmax_sub: subtile {sub}"))?;
            Ok(done(with_b(it, &a[0], cx, |b| {
                blocks::softmax_sub(b, &g, sub)
            })?))
        }
        "fa::skip_if_helper" => {
            need(2)?;
            let label = str_of(it, &a[1], "fa::skip_if_helper")?;
            Ok(done(with_b(it, &a[0], cx, |b| {
                blocks::skip_if_helper(b, &label)
            })?))
        }
        "fa::next_tile" => {
            need(1)?;
            Ok(done(with_b(it, &a[0], cx, blocks::next_tile)?))
        }
        n => Err(format!("native function `{n}` is not implemented").into()),
    }
}

/// Methods of an `FaGen`: `plan()`, `kernel_spec()`, `preconvert_plan()`, `preconvert_kernel_spec()`,
/// `spec()`, `lds_slots()`.
pub(super) fn method<'a>(
    _it: &Interp<'a>,
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
        "preconvert_plan" => {
            need(0)?;
            Some(done_with(g.preconvert_plan(), |p| {
                nat(Nat::Plan(Rc::new(RefCell::new(p))))
            }))
        }
        "kernel_spec" => {
            need(0)?;
            Some(spec_value(&g.kernel_spec()))
        }
        "preconvert_kernel_spec" => {
            need(0)?;
            Some(spec_value(&g.preconvert_kernel_spec()))
        }
        "spec" => {
            need(0)?;
            Some(spec_val(g.spec()))
        }
        "lds_slots" => {
            need(0)?;
            Some(Value::Array(
                g.lds_slots()
                    .into_iter()
                    .map(|(n, base, bytes)| {
                        Value::Tuple(vec![Value::str(n), Value::u32(base), Value::u32(bytes)])
                    })
                    .collect(),
            ))
        }
        _ => None,
    })
}
