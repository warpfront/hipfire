//! The `mq6_i8_gemm` generator as natives (a child module of [`super`]).
//!
//! Two layers over `peacemaker_isa::kernels::mq6_i8_gemm`, adding no typestate surface:
//!
//! - Coarse: `Mq6Gemm::emit(Mq6Spec { epi, raster, tile, arch })` and `Mq6Gemm::emit_module(arch)` return
//!   what the Rust functions return; `Spec::validate` stays the gate. `Mq6Epi::{Set, SetBf16}`,
//!   `Raster::{Banded, Linear}` and `Mq6Tile::{T128x128, T256x128, T128x256}` are enum values.
//! - Blocks: `Mq6Gen::new(spec)` and the `mq6::*` natives of `mq6_i8_gemm::blocks`. Unlike the
//!   `iu4::*` blocks these own no ring and never enter the typed core (no `Driver::typed`, no
//!   `Lend`): the natives that touch LDS **return `Instruction`s** (`mq6::store_w_insns`,
//!   `mq6::step_load_insns`, ...) which the script issues through the checked driver
//!   (`wg.ds_store`, `wg.ds_load_cur`) on rings it made itself (`wg.lds` + `Ring::new`), and every
//!   other native (fetch, decode, WMMA step, fold, seed fill, prologue set-up, epilogue) emits
//!   straight into `wg.isa()`. Waits, barriers, ring transitions and loops are therefore script
//!   statements.

use super::{
    arch_of, check_args, done, done_with, driver, nat, nat_of, spec_value, str_of, u32_of, u8_of,
    usize_of, Nat,
};
use crate::interp::{Cx, Interp, R};
use crate::value::Value;
use peacemaker_isa::insn::Instruction;
use peacemaker_isa::kernels::mq6_i8_gemm::{
    self as mq6, blocks, Epi, Gen, Raster, RingDepth, Spec, Tile, Tune,
};
use peacemaker_isa::Emitted;
use std::cell::RefCell;
use std::rc::Rc;

type Res<T> = Result<T, String>;

const FNS: &[&str] = &[
    "Mq6Gemm::emit",
    "Mq6Gemm::emit_module",
    "Mq6Gen::new",
    "mq6::label",
    "mq6::tail_label",
    "mq6::trips_cmp_insn",
    "mq6::step_descriptor",
    "mq6::trip_end",
    "mq6::declare_block_regs",
    "mq6::fetch_a",
    "mq6::fetch_w",
    "mq6::fetch_meta",
    "mq6::wait_all",
    "mq6::decode_insns",
    "mq6::seed_fill",
    "mq6::mma",
    "mq6::step_load_insns",
    "mq6::store_w_insns",
    "mq6::store_a_insns",
    "mq6::meta_valu",
    "mq6::meta_store_insns",
    "mq6::fold_addr",
    "mq6::fold_token_load_insn",
    "mq6::fold_sz_load_insn",
    "mq6::emit_msn",
    "mq6::fold_stage",
    "mq6::fold_fragment",
    "mq6::prologue_front",
    "mq6::tile_outside_insn",
    "mq6::prologue_scalars",
    "mq6::prologue_lanes",
    "mq6::prologue_fetch",
    "mq6::zero_acc",
    "mq6::epilogue",
    "mq6::group_end",
    "mq6::trips_dec",
    "mq6::ring3_split",
    "mq6::ring3_cmp_insn",
];

pub(super) fn is_fn(name: &str) -> bool {
    FNS.contains(&name)
}

pub(super) fn is_type(n: &str) -> bool {
    matches!(
        n,
        "Mq6Epi" | "Raster" | "Mq6Tile" | "Mq6Spec" | "Mq6Gemm" | "Mq6Gen"
    )
}

pub(super) fn is_struct(n: &str) -> bool {
    n == "Mq6Spec"
}

const ENUMS: &[(&str, &[&str])] = &[
    ("Mq6Epi", &["Set", "SetBf16"]),
    ("Raster", &["Banded", "Linear"]),
    ("Mq6Tile", &["T128x128", "T256x128", "T128x256"]),
];

pub(super) fn assoc_const<'a>(ty: &str, item: &str) -> Option<Value<'a>> {
    if let Some((t, vs)) = ENUMS.iter().find(|(t, _)| *t == ty) {
        let v = vs.iter().find(|v| **v == item)?;
        return Some(Value::Enum(Rc::from(*t), Rc::from(*v), vec![]));
    }
    if ty == "mq6" {
        if item == "GROUP_BYTES" {
            return Some(Value::u32(mq6::spec::GROUP_BYTES));
        }
        return Some(Value::str(match item {
            "ENTRY" => mq6::ENTRY,
            "K_BEGIN" => mq6::K_BEGIN,
            "K_LOOP" => mq6::K_LOOP,
            "K_LOOP_END" => mq6::K_LOOP_END,
            "EPI" => mq6::EPI,
            "END" => mq6::END,
            // The depth-3 arm selector's labels (`kloop::ring3`).
            "K_ARMS_END" => mq6::kloop::ring3::K_ARMS_END,
            "ARM12" => mq6::kloop::ring3::ARM12,
            "ARM12_END" => mq6::kloop::ring3::ARM12_END,
            "ARM2" => mq6::kloop::ring3::ARM2,
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

fn epi_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Epi> {
    match variant_of(it, v, "Mq6Epi")?.as_str() {
        "Set" => Ok(Epi::Set),
        "SetBf16" => Ok(Epi::SetBf16),
        o => Err(format!("`Mq6Epi::{o}` is not a variant of Mq6Epi")),
    }
}

fn raster_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Raster> {
    match variant_of(it, v, "Raster")?.as_str() {
        "Banded" => Ok(Raster::Banded),
        "Linear" => Ok(Raster::Linear),
        o => Err(format!("`Raster::{o}` is not a variant of Raster")),
    }
}

fn tile_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Tile> {
    match variant_of(it, v, "Mq6Tile")?.as_str() {
        "T128x128" => Ok(Tile::T128x128),
        "T256x128" => Ok(Tile::T256x128),
        "T128x256" => Ok(Tile::T128x256),
        o => Err(format!("`Mq6Tile::{o}` is not a variant of Mq6Tile")),
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
    let mut fs = fields_of(it, v, "Mq6Spec")?;
    let n = "Mq6Spec";
    let mut spec = Spec {
        epi: epi_of(it, &field(&mut fs, n, "epi")?)?,
        raster: raster_of(it, &field(&mut fs, n, "raster")?)?,
        tile: tile_of(it, &field(&mut fs, n, "tile")?)?,
        arch: arch_of(it, &field(&mut fs, n, "arch")?)?,
        tune: Tune::default(),
    };
    // `ring` (the A/W ring depth, 2 or 3) is the one optional field: absent means the shipped 2.
    if let Some(i) = fs.iter().position(|(f, _)| &**f == "ring") {
        let ring = u32_of(it, &fs.remove(i).1, "Mq6Spec ring")?;
        spec.tune.ring = ring
            .to_string()
            .parse::<RingDepth>()
            .map_err(|e| format!("`Mq6Spec` ring: {e}"))?;
    }
    if let Some((k, _)) = fs.first() {
        return Err(format!("`{n}` has no field `{k}`"));
    }
    Ok(spec)
}

fn spec_val<'a>(s: Spec) -> Value<'a> {
    let f = |k: &str, v: Value<'a>| (Rc::<str>::from(k), v);
    Value::Struct(
        "Mq6Spec".into(),
        vec![
            f("epi", enum_val("Mq6Epi", &format!("{:?}", s.epi))),
            f("raster", enum_val("Raster", &format!("{:?}", s.raster))),
            f("tile", enum_val("Mq6Tile", &format!("{:?}", s.tile))),
            f("arch", nat(Nat::Arch(s.arch))),
            f("ring", Value::u32(s.tune.ring.name().parse().unwrap_or(2))),
        ],
    )
}

fn emitted_value<'a>(e: Emitted) -> Value<'a> {
    nat(Nat::Emitted(Rc::new(e)))
}

// ---- argument conversion ------------------------------------------------------

fn gen_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Rc<Gen>> {
    match nat_of(it, v) {
        Some(Nat::Mq6Gen(g)) => Ok(g),
        _ => Err("expected an Mq6Gen (`Mq6Gen::new(Mq6Spec { .. })?`)".into()),
    }
}
fn wg_arg<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<()> {
    match nat_of(it, v) {
        Some(Nat::Wg) => Ok(()),
        _ => Err("the first argument is the workgroup scope `wg`".into()),
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
    // `fn(wg, g, ..)` natives that emit into the builder: the generator is argument 1.
    macro_rules! emit {
        ($n:expr, |$b:ident, $g:ident| $body:expr) => {{
            need($n)?;
            wg_arg(it, &a[0])?;
            let $g = gen_of(it, &a[1])?;
            let $b = driver(cx)?.isa();
            Ok(done($body))
        }};
    }
    match name {
        "Mq6Gemm::emit" => {
            need(1)?;
            Ok(done_with(mq6::emit(spec_of(it, &a[0])?), emitted_value))
        }
        "Mq6Gemm::emit_module" => {
            need(1)?;
            let m = mq6::emit_module(arch_of(it, &a[0])?).and_then(|(emitted, text, proof)| {
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
        "Mq6Gen::new" => {
            need(1)?;
            let g = Gen::new(spec_of(it, &a[0])?);
            Ok(done_with(g, |g| nat(Nat::Mq6Gen(Rc::new(g)))))
        }
        "mq6::label" => {
            need(1)?;
            Ok(Value::str(blocks::label(&str_of(it, &a[0], "mq6::label")?)))
        }
        "mq6::tail_label" => {
            need(1)?;
            Ok(Value::str(blocks::tail_label(usize_of(
                it,
                &a[0],
                "mq6::tail_label",
            )?)))
        }
        "mq6::trips_cmp_insn" => {
            need(2)?;
            let (g, cmp) = (gen_of(it, &a[0])?, str_of(it, &a[1], "trips_cmp")?);
            Ok(done_with(blocks::trips_cmp_insn(&g, &cmp), insn_val))
        }
        "mq6::decode_insns" => {
            need(2)?;
            let (g, r) = (gen_of(it, &a[0])?, usize_of(it, &a[1], "decode round")?);
            Ok(done_with(blocks::decode_insns(&g, r), insns_val))
        }
        "mq6::step_load_insns" => {
            need(4)?;
            let g = gen_of(it, &a[0])?;
            let (slot, step, set) = (
                usize_of(it, &a[1], "slot")?,
                u32_of(it, &a[2], "step")?,
                usize_of(it, &a[3], "set")?,
            );
            Ok(done_with(
                blocks::step_load_insns(&g, slot, step, set),
                insns_val,
            ))
        }
        "mq6::store_w_insns" => {
            need(3)?;
            let g = gen_of(it, &a[0])?;
            let (slot, r) = (usize_of(it, &a[1], "slot")?, usize_of(it, &a[2], "round")?);
            Ok(done_with(blocks::store_w_insns(&g, slot, r), insns_val))
        }
        "mq6::store_a_insns" => {
            need(3)?;
            let g = gen_of(it, &a[0])?;
            let (slot, sb) = (usize_of(it, &a[1], "slot")?, u32_of(it, &a[2], "half")?);
            Ok(done_with(blocks::store_a_insns(&g, slot, sb), insns_val))
        }
        "mq6::meta_store_insns" => {
            need(2)?;
            let (g, plane) = (gen_of(it, &a[0])?, usize_of(it, &a[1], "plane")?);
            Ok(done_with(blocks::meta_store_insns(&g, plane), insns_val))
        }
        "mq6::fold_token_load_insn" => {
            need(2)?;
            let (g, nb) = (gen_of(it, &a[0])?, usize_of(it, &a[1], "token block")?);
            Ok(done_with(blocks::fold_token_load_insn(&g, nb), insn_val))
        }
        "mq6::fold_sz_load_insn" => {
            need(4)?;
            let g = gen_of(it, &a[0])?;
            let (h, dst, rel) = (
                usize_of(it, &a[1], "plane")?,
                u8_of(it, &a[2], "destination register")?,
                u32_of(it, &a[3], "plane byte offset")?,
            );
            Ok(done_with(
                blocks::fold_sz_load_insn(&g, h, dst, rel),
                insn_val,
            ))
        }
        "mq6::tile_outside_insn" => {
            need(2)?;
            let (g, which) = (gen_of(it, &a[0])?, usize_of(it, &a[1], "which")?);
            Ok(done_with(blocks::tile_outside_insn(&g, which), insn_val))
        }
        "mq6::step_descriptor" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, which) = (gen_of(it, &a[1])?, usize_of(it, &a[2], "descriptor")?);
            Ok(done(blocks::step_descriptor(driver(cx)?.isa(), &g, which)))
        }
        "mq6::trip_end" => emit!(2, |b, g| blocks::trip_end(b, &g)),
        "mq6::declare_block_regs" => {
            need(5)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let (entry, fold, end) = (
                str_of(it, &a[2], "entry label")?,
                str_of(it, &a[3], "fold label")?,
                str_of(it, &a[4], "end label")?,
            );
            Ok(done(blocks::declare_block_regs(
                driver(cx)?.isa(),
                &g,
                &entry,
                &fold,
                &end,
            )))
        }
        "mq6::fetch_a" => {
            need(5)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let (slab, sb, srd) = (
                u32_of(it, &a[2], "slab")?,
                u32_of(it, &a[3], "half")?,
                u8_of(it, &a[4], "descriptor register")?,
            );
            Ok(done(blocks::fetch_a(driver(cx)?.isa(), &g, slab, sb, srd)))
        }
        "mq6::fetch_w" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, base) = (gen_of(it, &a[1])?, u32_of(it, &a[2], "weight offset")?);
            Ok(done(blocks::fetch_w(driver(cx)?.isa(), &g, base)))
        }
        "mq6::fetch_meta" => {
            need(4)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let (h, hdr) = (
                usize_of(it, &a[2], "plane")?,
                u32_of(it, &a[3], "header offset")?,
            );
            Ok(done(blocks::fetch_meta(driver(cx)?.isa(), &g, h, hdr)))
        }
        "mq6::wait_all" => {
            need(1)?;
            wg_arg(it, &a[0])?;
            Ok(done(blocks::wait_all(driver(cx)?.isa())))
        }
        "mq6::seed_fill" => emit!(2, |b, g| blocks::seed_fill(b, &g)),
        "mq6::meta_valu" => emit!(2, |b, g| blocks::meta_valu(b, &g)),
        "mq6::prologue_front" => emit!(2, |b, g| blocks::prologue_front(b, &g)),
        "mq6::prologue_scalars" => emit!(2, |b, g| blocks::prologue_scalars(b, &g)),
        "mq6::prologue_lanes" => emit!(2, |b, g| blocks::prologue_lanes(b, &g)),
        "mq6::prologue_fetch" => emit!(2, |b, g| blocks::prologue_fetch(b, &g)),
        "mq6::zero_acc" => emit!(2, |b, g| blocks::zero_acc(b, &g)),
        "mq6::epilogue" => emit!(2, |b, g| blocks::epilogue(b, &g)),
        "mq6::mma" => {
            need(6)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let (step, set, first, i) = (
                u32_of(it, &a[2], "step")?,
                usize_of(it, &a[3], "fragment set")?,
                bool_of(it, &a[4], "first")?,
                usize_of(it, &a[5], "wmma index")?,
            );
            Ok(done(blocks::mma(
                driver(cx)?.isa(),
                &g,
                step,
                set,
                first,
                i,
            )))
        }
        "mq6::fold_addr" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, h) = (gen_of(it, &a[1])?, usize_of(it, &a[2], "plane")?);
            Ok(done(blocks::fold_addr(driver(cx)?.isa(), &g, h)))
        }
        "mq6::emit_msn" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, nb) = (gen_of(it, &a[1])?, usize_of(it, &a[2], "token block")?);
            Ok(done(blocks::emit_msn(driver(cx)?.isa(), &g, nb)))
        }
        "mq6::fold_stage" => {
            need(6)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let (k, stage, sc, zp) = (
                usize_of(it, &a[2], "fragment")?,
                usize_of(it, &a[3], "stage")?,
                u8_of(it, &a[4], "sc octet")?,
                u8_of(it, &a[5], "z' octet")?,
            );
            Ok(done(blocks::fold_stage(
                driver(cx)?.isa(),
                &g,
                k,
                stage,
                sc,
                zp,
            )))
        }
        "mq6::fold_fragment" => {
            need(5)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let (k, sc, zp) = (
                usize_of(it, &a[2], "fragment")?,
                u8_of(it, &a[3], "sc octet")?,
                u8_of(it, &a[4], "z' octet")?,
            );
            Ok(done(blocks::fold_fragment(
                driver(cx)?.isa(),
                &g,
                k,
                sc,
                zp,
            )))
        }
        "mq6::group_end" => emit!(2, |b, g| blocks::group_end(b, &g)),
        "mq6::trips_dec" => emit!(2, |b, g| blocks::trips_dec(b, &g)),
        "mq6::ring3_split" => emit!(2, |b, g| blocks::ring3_split(b, &g)),
        "mq6::ring3_cmp_insn" => {
            need(3)?;
            let (g, cmp, imm) = (
                gen_of(it, &a[0])?,
                str_of(it, &a[1], "ring3_cmp")?,
                u32_of(it, &a[2], "immediate")?,
            );
            Ok(done_with(blocks::ring3_cmp_insn(&g, &cmp, imm), insn_val))
        }
        n => Err(format!("native function `{n}` is not implemented").into()),
    }
}

/// Methods of an `Mq6Gen`: `plan()`, `kernel_spec()`, `spec()`, `reg(name)`, `slot(name)`,
/// `a_rounds()`, `w_rounds()` (staging rounds per slab of the tile's activations / weights).
pub(super) fn method<'a>(
    it: &Interp<'a>,
    g: &Rc<Gen>,
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
        "reg" => {
            need(1)?;
            Some(Value::option(
                g.reg(&str_of(it, &a[0], "reg")?).map(Value::u8),
            ))
        }
        "a_rounds" => {
            need(0)?;
            Some(Value::usize(g.spec().tile.a_rounds() as usize))
        }
        "w_rounds" => {
            need(0)?;
            Some(Value::usize(g.spec().tile.w_rounds() as usize))
        }
        "depth" => {
            need(0)?;
            Some(Value::usize(g.depth()))
        }
        "pf_set" => {
            need(1)?;
            let set = usize_of(it, &a[0], "prefetch set")?;
            Some(done_with(g.pf_set(set), |g| nat(Nat::Mq6Gen(Rc::new(g)))))
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
