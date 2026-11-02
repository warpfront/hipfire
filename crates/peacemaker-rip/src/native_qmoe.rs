//! The `qwen4_moe_sym` MQ4 entries as natives (a child module of [`super`]).
//!
//! Two layers over `peacemaker_isa::kernels::qwen4_moe_sym`, adding no typestate surface:
//!
//! - Coarse: `MoeSym::emit` / `MoeSym::emit_module` live in `native_gen` (the Rust functions'
//!   values).
//! - Blocks: `MoeNtGen::new(kind, nt, gp, arch)` with the `moent::*` natives of
//!   `qwen4_moe_sym::blocks` (the gfx1201 NT4 / NT8 entries of `gate_up` and `down`, flat-slot and
//!   grouped-position), and `MoeNt1Gen::new(kind, arch)` with the `moe1::*` natives of
//!   `qwen4_moe_sym::nt1_blocks` (the NT1 entries). Like the `mq6::*` blocks these own no ring and
//!   never enter the typed core (no `Driver::typed`, no `Lend`): every block is a straight run of
//!   instructions emitted into `wg.isa()`, every compare is a native returning an `Instruction`
//!   (`moent::tile_live_insn`, `moent::outside_insn`, ...) which the script turns into a condition
//!   with `wg.scmp` / `wg.scmp_wg_uniform`, and the `gate_up` hand-off's LDS accesses are natives
//!   returning `Instruction`s (`moent::publish_insns`, `moent::gate_load_insns`) which the script
//!   issues through `wg.ds_store` / `wg.ds_load` on the region it declared itself. Exits, the run
//!   walk, per-tile branches, waits, the K-loop, the tail and the hand-off are script statements.

use super::native_gen::wg_arg;
use super::{
    arch_of, check_args, done, done_with, driver, nat, nat_of, spec_value, str_of, u8_of, usize_of,
    Nat,
};
use crate::interp::{Cx, Interp, R};
use crate::value::Value;
use peacemaker_isa::insn::Instruction;
use peacemaker_isa::kernels::qwen4_moe_sym::{
    blocks as nt, nt1_blocks as nt1, Kind, Spec, WeightCodec,
};
use std::cell::RefCell;
use std::rc::Rc;

type Res<T> = Result<T, String>;

const FNS: &[&str] = &[
    "MoeNtGen::new",
    "MoeNt1Gen::new",
    // NT4 / NT8: prologue
    "moent::pro_front",
    "moent::outside_insn",
    "moent::pro_expert",
    "moent::sentinel_insn",
    "moent::pro_run",
    "moent::walk_head",
    "moent::walk_hit_insn",
    "moent::walk_step",
    "moent::pro_leader",
    "moent::follower_insn",
    "moent::pro_cnt",
    "moent::pro_slot_load",
    "moent::pro_desc",
    "moent::pro_tile",
    "moent::pro_init",
    // NT4 / NT8: loads, fold, K-loop pieces
    "moent::load_w",
    "moent::load_x",
    "moent::compute_head",
    "moent::wait_tile",
    "moent::tile_live_insn",
    "moent::fold_tile",
    "moent::advance",
    "moent::trips_zero_insn",
    "moent::trips_more_insn",
    "moent::trips_dec",
    "moent::odd_insn",
    // NT4 / NT8: epilogue
    "moent::epilogue_label",
    "moent::y_descriptor",
    "moent::y_step",
    "moent::store_rows",
    "moent::bf16_roundtrip",
    "moent::up_insn",
    "moent::publish_insns",
    "moent::gate_load_insns",
    "moent::silu_tile",
    // NT1
    "moe1::pro_front",
    "moe1::outside_insn",
    "moe1::pro_expert",
    "moe1::sentinel_insn",
    "moe1::pro_setup",
    "moe1::pro_init",
    "moe1::load_set",
    "moe1::compute",
    "moe1::advance",
    "moe1::trips_zero_insn",
    "moe1::trips_more_insn",
    "moe1::trips_dec",
    "moe1::odd_insn",
    "moe1::tail_pair",
    "moe1::epilogue_label",
    "moe1::swiglu",
    "moe1::store",
];

pub(super) fn is_fn(name: &str) -> bool {
    FNS.contains(&name)
}

pub(super) fn is_type(n: &str) -> bool {
    matches!(n, "MoeNtGen" | "MoeNt1Gen")
}

fn nt_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Rc<nt::Gen>> {
    match nat_of(it, v) {
        Some(Nat::MoeNtGen(g)) => Ok(g),
        _ => Err("expected a MoeNtGen (`MoeNtGen::new(kind, nt, gp, arch)?`)".into()),
    }
}

fn nt1_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Rc<nt1::Gen>> {
    match nat_of(it, v) {
        Some(Nat::MoeNt1Gen(g)) => Ok(g),
        _ => Err("expected a MoeNt1Gen (`MoeNt1Gen::new(kind, arch)?`)".into()),
    }
}

fn kind_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Kind> {
    match str_of(it, v, "kind")?.as_str() {
        "gate_up" => Ok(Kind::GateUp),
        "down" => Ok(Kind::Down),
        k => Err(format!("kind `{k}`: expected gate_up or down")),
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
    // `fn(wg, g, ..)` natives that emit into the builder: the generator is argument 1.
    macro_rules! emit_nt {
        ($n:expr, |$b:ident, $g:ident| $body:expr) => {{
            need($n)?;
            wg_arg(it, &a[0])?;
            let $g = nt_of(it, &a[1])?;
            let $b = driver(cx)?.isa();
            Ok(done($body))
        }};
    }
    macro_rules! emit_nt1 {
        ($n:expr, |$b:ident, $g:ident| $body:expr) => {{
            need($n)?;
            wg_arg(it, &a[0])?;
            let $g = nt1_of(it, &a[1])?;
            let $b = driver(cx)?.isa();
            Ok(done($body))
        }};
    }
    // `fn(wg, ..)` natives that need no generator.
    macro_rules! emit_wg {
        ($b:ident, $body:expr) => {{
            need(1)?;
            wg_arg(it, &a[0])?;
            let $b = driver(cx)?.isa();
            Ok(done($body))
        }};
    }
    match name {
        "MoeNtGen::new" => {
            need(4)?;
            let kind = kind_of(it, &a[0])?;
            let tiles = u8_of(it, &a[1], "tile width nt")?;
            let gp = bool_of(it, &a[2], "grouped-position flag")?;
            let arch = arch_of(it, &a[3])?;
            let spec = Spec {
                arch,
                kind,
                nt: tiles,
                rr: 1,
                codec: WeightCodec::Mq4Sym,
            };
            Ok(done_with(nt::Gen::new(spec, gp), |g| {
                nat(Nat::MoeNtGen(Rc::new(g)))
            }))
        }
        "MoeNt1Gen::new" => {
            need(2)?;
            let kind = kind_of(it, &a[0])?;
            let arch = arch_of(it, &a[1])?;
            let spec = Spec {
                arch,
                kind,
                nt: 1,
                rr: 1,
                codec: WeightCodec::Mq4Sym,
            };
            Ok(done_with(nt1::Gen::new(spec), |g| {
                nat(Nat::MoeNt1Gen(Rc::new(g)))
            }))
        }

        // ---- NT4 / NT8 ----
        "moent::pro_front" => emit_nt!(2, |b, g| nt::pro_front(b, &g)),
        "moent::outside_insn" => {
            need(1)?;
            let g = nt_of(it, &a[0])?;
            Ok(insn_val(nt::outside_insn(&g)))
        }
        "moent::pro_expert" => emit_nt!(2, |b, g| nt::pro_expert(b, &g)),
        "moent::sentinel_insn" => {
            need(1)?;
            let g = nt_of(it, &a[0])?;
            Ok(insn_val(nt::sentinel_insn(&g)))
        }
        "moent::pro_run" => emit_nt!(2, |b, g| nt::pro_run(b, &g)),
        "moent::walk_head" => emit_nt!(2, |b, g| nt::walk_head(b, &g)),
        "moent::walk_hit_insn" => {
            need(1)?;
            let g = nt_of(it, &a[0])?;
            Ok(insn_val(nt::walk_hit_insn(&g)))
        }
        "moent::walk_step" => emit_nt!(2, |b, g| nt::walk_step(b, &g)),
        "moent::pro_leader" => emit_nt!(2, |b, g| nt::pro_leader(b, &g)),
        "moent::follower_insn" => {
            need(1)?;
            let g = nt_of(it, &a[0])?;
            Ok(insn_val(nt::follower_insn(&g)))
        }
        "moent::pro_cnt" => emit_nt!(2, |b, g| nt::pro_cnt(b, &g)),
        "moent::pro_slot_load" => emit_nt!(3, |b, g| nt::pro_slot_load(
            b,
            &g,
            u8_of(it, &a[2], "tile index")?
        )),
        "moent::pro_desc" => emit_nt!(2, |b, g| nt::pro_desc(b, &g)),
        "moent::pro_tile" => emit_nt!(3, |b, g| nt::pro_tile(
            b,
            &g,
            u8_of(it, &a[2], "tile index")?
        )),
        "moent::pro_init" => emit_nt!(2, |b, g| nt::pro_init(b, &g)),
        "moent::load_w" => emit_nt!(3, |b, g| nt::load_w(
            b,
            &g,
            usize_of(it, &a[2], "epoch parity")?
        )),
        "moent::load_x" => emit_nt!(4, |b, g| nt::load_x(
            b,
            &g,
            u8_of(it, &a[2], "tile index")?,
            usize_of(it, &a[3], "epoch parity")?
        )),
        "moent::compute_head" => emit_nt!(3, |b, g| nt::compute_head(
            b,
            &g,
            usize_of(it, &a[2], "epoch parity")?
        )),
        "moent::wait_tile" => emit_nt!(3, |b, g| nt::wait_tile(
            b,
            &g,
            u8_of(it, &a[2], "tile index")?
        )),
        "moent::tile_live_insn" => {
            need(2)?;
            let g = nt_of(it, &a[0])?;
            Ok(done_with(
                nt::tile_live_insn(&g, u8_of(it, &a[1], "tile index")?),
                insn_val,
            ))
        }
        "moent::fold_tile" => emit_nt!(4, |b, g| nt::fold_tile(
            b,
            &g,
            usize_of(it, &a[2], "epoch parity")?,
            u8_of(it, &a[3], "tile index")?
        )),
        "moent::advance" => emit_wg!(b, nt::advance(b)),
        "moent::trips_zero_insn" => {
            need(0)?;
            Ok(insn_val(nt::trips_zero_insn()))
        }
        "moent::trips_more_insn" => {
            need(0)?;
            Ok(insn_val(nt::trips_more_insn()))
        }
        "moent::trips_dec" => emit_wg!(b, nt::trips_dec(b)),
        "moent::odd_insn" => {
            need(0)?;
            Ok(insn_val(nt::odd_insn()))
        }
        "moent::epilogue_label" => emit_nt!(2, |b, g| nt::epilogue_label(b, &g)),
        "moent::y_descriptor" => emit_nt!(2, |b, g| nt::y_descriptor(b, &g)),
        "moent::y_step" => emit_wg!(b, nt::y_step(b)),
        "moent::store_rows" => emit_nt!(4, |b, g| nt::store_rows(
            b,
            &g,
            u8_of(it, &a[2], "tile index")?,
            bool_of(it, &a[3], "gate")?
        )),
        "moent::bf16_roundtrip" => emit_nt!(2, |b, g| nt::bf16_roundtrip(b, &g)),
        "moent::up_insn" => {
            need(0)?;
            Ok(insn_val(nt::up_insn()))
        }
        "moent::publish_insns" => {
            need(2)?;
            let g = nt_of(it, &a[0])?;
            Ok(done_with(
                nt::publish_insns(&g, u8_of(it, &a[1], "tile index")?),
                insns_val,
            ))
        }
        "moent::gate_load_insns" => {
            need(2)?;
            let g = nt_of(it, &a[0])?;
            Ok(done_with(
                nt::gate_load_insns(&g, u8_of(it, &a[1], "tile index")?),
                insns_val,
            ))
        }
        "moent::silu_tile" => emit_nt!(3, |b, g| nt::silu_tile(
            b,
            &g,
            u8_of(it, &a[2], "tile index")?
        )),

        // ---- NT1 ----
        "moe1::pro_front" => emit_nt1!(2, |b, g| nt1::pro_front(b, &g)),
        "moe1::outside_insn" => {
            need(0)?;
            Ok(insn_val(nt1::outside_insn()))
        }
        "moe1::pro_expert" => emit_wg!(b, nt1::pro_expert(b)),
        "moe1::sentinel_insn" => {
            need(0)?;
            Ok(insn_val(nt1::sentinel_insn()))
        }
        "moe1::pro_setup" => emit_nt1!(2, |b, g| nt1::pro_setup(b, &g)),
        "moe1::pro_init" => emit_nt1!(2, |b, g| nt1::pro_init(b, &g)),
        "moe1::load_set" => emit_nt1!(3, |b, g| nt1::load_set(
            b,
            &g,
            usize_of(it, &a[2], "set parity")?
        )),
        "moe1::compute" => emit_nt1!(3, |b, g| nt1::compute(
            b,
            &g,
            usize_of(it, &a[2], "set parity")?
        )),
        "moe1::advance" => emit_nt1!(2, |b, g| nt1::advance(b, &g)),
        "moe1::trips_zero_insn" => {
            need(0)?;
            Ok(insn_val(nt1::trips_zero_insn()))
        }
        "moe1::trips_more_insn" => {
            need(0)?;
            Ok(insn_val(nt1::trips_more_insn()))
        }
        "moe1::trips_dec" => emit_wg!(b, nt1::trips_dec(b)),
        "moe1::odd_insn" => {
            need(0)?;
            Ok(insn_val(nt1::odd_insn()))
        }
        "moe1::tail_pair" => emit_nt1!(2, |b, g| nt1::tail_pair(b, &g)),
        "moe1::epilogue_label" => emit_nt1!(2, |b, g| nt1::epilogue_label(b, &g)),
        "moe1::swiglu" => emit_nt1!(2, |b, g| nt1::swiglu(b, &g)),
        "moe1::store" => emit_nt1!(2, |b, g| nt1::store(b, &g)),
        n => Err(format!("native function `{n}` is not implemented").into()),
    }
}

/// Methods of a `MoeNtGen`: `plan()`, `kernel_spec()`, `nt()`, `gp()`, `gate_up()`, `lds_bytes()`,
/// `label(name)`, `exit_label()`.
pub(super) fn nt_method<'a>(
    it: &Interp<'a>,
    g: &Rc<nt::Gen>,
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
        "nt" => {
            need(0)?;
            Some(Value::u8(g.nt()))
        }
        "gp" => {
            need(0)?;
            Some(Value::Bool(g.gp()))
        }
        "gate_up" => {
            need(0)?;
            Some(Value::Bool(g.gate_up()))
        }
        "lds_bytes" => {
            need(0)?;
            Some(Value::u32(g.lds_bytes()))
        }
        "label" => {
            need(1)?;
            Some(Value::str(g.label(&str_of(it, &a[0], "label")?)))
        }
        "exit_label" => {
            need(0)?;
            Some(Value::str(g.exit_label()))
        }
        _ => None,
    })
}

/// Methods of a `MoeNt1Gen`: `plan()`, `kernel_spec()`, `gate_up()`, `label(name)`, `exit_label()`.
pub(super) fn nt1_method<'a>(
    it: &Interp<'a>,
    g: &Rc<nt1::Gen>,
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
        "gate_up" => {
            need(0)?;
            Some(Value::Bool(g.spec().kind == Kind::GateUp))
        }
        "label" => {
            need(1)?;
            Some(Value::str(g.label(&str_of(it, &a[0], "label")?)))
        }
        "exit_label" => {
            need(0)?;
            Some(Value::str(g.exit_label()))
        }
        _ => None,
    })
}
