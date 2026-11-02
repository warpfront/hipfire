//! The PM producers as natives (a child module of [`super`]): `iu4_c8_producer_pm` (all four exports),
//! `moe_rotate_i4_pm` (flat-slot and grouped-position) and `gdn_chunk_prep_pm`.
//!
//! All three are untyped-`Builder` kernels, so their twin scripts are too (the pattern of the `fp8_gemm`
//! twin): `Builder::new` with the generator's `kernel_spec()` and `plan()?`, no `Workgroup`. A generator is
//! an opaque handle (`C8Gen::new(arch, tag)`, `RotGen::new(arch, gp)`, `GdnGen::new(arch)`) and the natives
//! are the public `blocks` of `iu4_c8_producer`, `moe_rotate_i4_pm` and `gdn_chunk_prep`: each emits straight
//! into the builder passed as its first argument. The norm's LDS store, barrier and loads are statements of
//! the script (`c8::norm_store_insn` and `c8::norm_load_insns` return the instructions for `b.ds_store` /
//! `b.ds_load`), and the gate and conv loops of `gdn_chunk_prep` are the script's `b.loop_` statements over
//! `gdn::gate_body` / `gdn::row_body`.

use super::{
    arch_of, check_args, done, done_with, nat, nat_of, spec_value, str_of, u8s_of, with_b, Nat,
};
use crate::interp::{Cx, Interp, R};
use crate::value::Value;
use peacemaker_isa::insn::Instruction;
use peacemaker_isa::kernels::gdn_chunk_prep::{self as gdn, GdnChunkPrepSpec};
use peacemaker_isa::kernels::iu4_c8_producer::blocks as c8;
use peacemaker_isa::kernels::moe_rotate_i4_pm::blocks as rot;
use peacemaker_isa::kernels::moe_rotate_i4_pm::Spec as RotSpec;
use std::cell::RefCell;
use std::rc::Rc;

type Res<T> = Result<T, String>;

/// One producer generator.
pub enum ProdGen {
    C8(c8::Gen),
    Rot(rot::Gen),
    Gdn(gdn::blocks::Gen),
}

const FNS: &[&str] = &[
    "C8Gen::new",
    "RotGen::new",
    "GdnGen::new",
    "prod::label",
    "prod::wait_all",
    "c8::prologue",
    "c8::rotate_front",
    "c8::sigmoid_rotate_front",
    "c8::norm_row_loads",
    "c8::norm_prefetch",
    "c8::norm_sumsq",
    "c8::norm_reduce_pre",
    "c8::norm_store_insn",
    "c8::norm_reduce_addr",
    "c8::norm_load_insns",
    "c8::norm_reduce_post",
    "c8::norm_rms",
    "c8::norm_apply",
    "c8::fwht_scale_store",
    "c8::quantize",
    "c8::store_record",
    "rot::prologue",
    "rot::rotate",
    "rot::quantize",
    "rot::store",
    "gdn::label",
    "gdn::prologue",
    "gdn::gate_guard",
    "gdn::gate_setup",
    "gdn::gate_body",
    "gdn::conv_setup",
    "gdn::conv_to_v",
    "gdn::conv_to_k",
    "gdn::region_setup",
    "gdn::row_body",
    "gdn::to_epi",
    "gdn::epilogue",
];

pub(super) fn is_fn(name: &str) -> bool {
    FNS.contains(&name)
}

pub(super) fn is_type(n: &str) -> bool {
    matches!(n, "C8Gen" | "RotGen" | "GdnGen")
}

fn handle_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Rc<ProdGen>> {
    match nat_of(it, v) {
        Some(Nat::ProdGen(g)) => Ok(g),
        _ => Err("expected a producer generator (`C8Gen::new(arch, tag)?`, `RotGen::new(arch, gp)?` or `GdnGen::new(arch)?`)".into()),
    }
}
fn c8_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Rc<ProdGen>> {
    let g = handle_of(it, v)?;
    match *g {
        ProdGen::C8(_) => Ok(g),
        _ => Err("expected a C8Gen".into()),
    }
}
fn rot_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Rc<ProdGen>> {
    let g = handle_of(it, v)?;
    match *g {
        ProdGen::Rot(_) => Ok(g),
        _ => Err("expected a RotGen".into()),
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
    // `fn(b)` natives: no generator argument.
    macro_rules! emit0 {
        ($f:expr) => {{
            need(1)?;
            Ok(done(with_b(it, &a[0], cx, $f)?))
        }};
    }
    // `fn(b, g)` natives of a `C8Gen` / `RotGen`.
    macro_rules! emit_c8 {
        (|$b:ident, $g:ident| $body:expr) => {{
            need(2)?;
            let h = c8_of(it, &a[1])?;
            let ProdGen::C8(ref $g) = *h else {
                unreachable!()
            };
            Ok(done(with_b(it, &a[0], cx, |$b| $body)?))
        }};
    }
    macro_rules! emit_rot {
        (|$b:ident, $g:ident| $body:expr) => {{
            need(2)?;
            let h = rot_of(it, &a[1])?;
            let ProdGen::Rot(ref $g) = *h else {
                unreachable!()
            };
            Ok(done(with_b(it, &a[0], cx, |$b| $body)?))
        }};
    }
    match name {
        "C8Gen::new" => {
            need(2)?;
            let (arch, tag) = (arch_of(it, &a[0])?, str_of(it, &a[1], "export tag")?);
            Ok(done_with(c8::Gen::by_tag(arch, &tag), |g| {
                nat(Nat::ProdGen(Rc::new(ProdGen::C8(g))))
            }))
        }
        "RotGen::new" => {
            need(2)?;
            let (arch, gp) = (
                arch_of(it, &a[0])?,
                bool_of(it, &a[1], "grouped-position flag")?,
            );
            Ok(done_with(rot::Gen::new(RotSpec { arch }, gp), |g| {
                nat(Nat::ProdGen(Rc::new(ProdGen::Rot(g))))
            }))
        }
        "GdnGen::new" => {
            need(1)?;
            let arch = arch_of(it, &a[0])?;
            Ok(done_with(
                gdn::blocks::Gen::new(arch, &GdnChunkPrepSpec::A3B),
                |g| nat(Nat::ProdGen(Rc::new(ProdGen::Gdn(g)))),
            ))
        }
        "prod::label" => {
            need(2)?;
            let label = str_of(it, &a[1], "prod::label")?;
            Ok(done(with_b(it, &a[0], cx, |b| b.label(&label))?))
        }
        "prod::wait_all" => emit0!(|b| b.wait_all()),
        "c8::prologue" => emit_c8!(|b, g| c8::prologue(b, g)),
        "c8::rotate_front" => emit0!(c8::rotate_front),
        "c8::sigmoid_rotate_front" => emit0!(c8::sigmoid_rotate_front),
        "c8::norm_row_loads" => emit0!(c8::norm_row_loads),
        "c8::norm_prefetch" => emit0!(c8::norm_prefetch),
        "c8::norm_sumsq" => emit0!(c8::norm_sumsq),
        "c8::norm_reduce_pre" => emit_c8!(|b, g| c8::norm_reduce_pre(b, g)),
        "c8::norm_store_insn" => {
            need(0)?;
            Ok(insn_val(c8::norm_store_insn()))
        }
        "c8::norm_reduce_addr" => emit_c8!(|b, g| c8::norm_reduce_addr(b, g)),
        "c8::norm_load_insns" => {
            need(0)?;
            Ok(Value::Array(
                c8::norm_load_insns().into_iter().map(insn_val).collect(),
            ))
        }
        "c8::norm_reduce_post" => emit_c8!(|b, g| c8::norm_reduce_post(b, g)),
        "c8::norm_rms" => emit0!(c8::norm_rms),
        "c8::norm_apply" => emit_c8!(|b, g| c8::norm_apply(b, g)),
        "c8::fwht_scale_store" => emit0!(c8::fwht_scale_store),
        "c8::quantize" => emit0!(c8::quantize),
        "c8::store_record" => emit_c8!(|b, g| c8::store_record(b, g)),
        "rot::prologue" => emit_rot!(|b, g| rot::prologue(b, g)),
        "rot::rotate" => {
            need(1)?;
            let e = with_b(it, &a[0], cx, rot::rotate)?;
            Ok(done_with(e, |e| {
                Value::Array(e.into_iter().map(Value::u8).collect())
            }))
        }
        "rot::quantize" => {
            need(2)?;
            let e = u8s_of(it, &a[1], "rotated registers")?;
            let e: [u8; 4] = e.try_into().map_err(|_| {
                "rot::quantize: the rotated registers are the four `rot::rotate` returned"
            })?;
            Ok(done(with_b(it, &a[0], cx, |b| rot::quantize(b, e))?))
        }
        "rot::store" => emit0!(rot::store),
        "gdn::label" => {
            need(1)?;
            Ok(done_with(
                gdn::blocks::label(&str_of(it, &a[0], "gdn::label")?),
                Value::str,
            ))
        }
        "gdn::prologue" => emit0!(gdn::blocks::prologue),
        "gdn::gate_guard" => emit0!(gdn::blocks::gate_guard),
        "gdn::gate_setup" => emit0!(gdn::blocks::gate_setup),
        "gdn::gate_body" => emit0!(gdn::blocks::gate_body),
        "gdn::conv_setup" => emit0!(gdn::blocks::conv_setup),
        "gdn::conv_to_v" => emit0!(gdn::blocks::conv_to_v),
        "gdn::conv_to_k" => emit0!(gdn::blocks::conv_to_k),
        "gdn::to_epi" => emit0!(gdn::blocks::to_epi),
        "gdn::epilogue" => emit0!(gdn::blocks::epilogue),
        "gdn::region_setup" => {
            need(2)?;
            let r = gdn::blocks::region(&str_of(it, &a[1], "region")?)?;
            Ok(done(with_b(it, &a[0], cx, |b| {
                gdn::blocks::region_setup(b, r)
            })?))
        }
        "gdn::row_body" => {
            need(3)?;
            let r = gdn::blocks::region(&str_of(it, &a[1], "region")?)?;
            let head = str_of(it, &a[2], "loop head")?;
            Ok(done(with_b(it, &a[0], cx, |b| {
                gdn::blocks::row_body(b, r, &head)
            })?))
        }
        n => Err(format!("native function `{n}` is not implemented").into()),
    }
}

/// Methods of a producer generator: `plan()`, `kernel_spec()`; C8: `tag()`, `lds_bytes()`; Rot: `gp()`.
pub(super) fn method<'a>(
    _it: &Interp<'a>,
    g: &Rc<ProdGen>,
    name: &str,
    a: &[Value<'a>],
) -> R<'a, Option<Value<'a>>> {
    let need = |n: usize| check_args(name, a, n);
    Ok(match (name, &**g) {
        ("plan", _) => {
            need(0)?;
            let p = match &**g {
                ProdGen::C8(x) => x.plan(),
                ProdGen::Rot(x) => x.plan(),
                ProdGen::Gdn(x) => x.plan(),
            };
            Some(done_with(p, |p| nat(Nat::Plan(Rc::new(RefCell::new(p))))))
        }
        ("kernel_spec", _) => {
            need(0)?;
            let k = match &**g {
                ProdGen::C8(x) => x.kernel_spec(),
                ProdGen::Rot(x) => x.kernel_spec(),
                ProdGen::Gdn(x) => x.kernel_spec(),
            };
            Some(spec_value(&k))
        }
        ("tag", ProdGen::C8(x)) => {
            need(0)?;
            Some(Value::str(x.export().tag()))
        }
        ("lds_bytes", ProdGen::C8(x)) => {
            need(0)?;
            Some(Value::u32(x.lds_bytes()))
        }
        ("gp", ProdGen::Rot(x)) => {
            need(0)?;
            Some(Value::Bool(x.gp()))
        }
        _ => None,
    })
}
