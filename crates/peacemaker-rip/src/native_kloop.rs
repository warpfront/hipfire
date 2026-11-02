//! The fine steps of the `iu4_gemm` K128 block as natives (a child module of
//! [`super`], which is [`native_gen`](super)): `iu4::block` split into typed
//! pieces a script can place one by one (cookbook 17).
//!
//! Every native here is the public Rust step of `iu4_gemm::blocks` that
//! `kloop::block_with` itself is composed of, so the generator's block and a
//! script that spells it out with these natives emit the same bytes. The LDS
//! rings stay the typed core's: a native that touches a ring takes it from the
//! driver (`Lend::take`, or `Lend::take_writing` for a ring whose next buffer
//! is `Writing` and still carries its `Pending` token) in the state the Rust
//! step is typed for, runs the step on the typed `Workgroup`, and gives it back
//! in the state the step left it (`give_back`, `give_writing`). A ring in
//! another state, of another kind (`PayA`/`PayW`/`MetaD`/`MetaS` tags), held by
//! a split barrier, or already lent is refused before anything is emitted:
//!
//! - a publish wants a `Free` next buffer and leaves it `Writing`, pending;
//! - a fragment/scale read wants a `Published` current buffer (the next one
//!   may be `Free` or `Writing`: reads are not ordered against the stores);
//! - the wait (`iu4::wait`, or `wg.wait_all`) turns pending into drained, and
//!   the driver's barrier refuses a `rotate` of an undrained buffer.
//!
//! The natives that run inside a split barrier's window (the fetches, the
//! fold, the WMMA groups, the descriptor step) touch no ring through `typed`
//! (which the driver refuses there): they emit through the builder, and the
//! WMMA group is gated on the payload rings' phases. `iu4::fragments_body` is
//! the exception: it replays a script-built body of DS loads through the same
//! lend/give protocol instead of calling a Rust step.

use super::{bool_of, gen_of, lds_of, lds_val, typed, wg_arg, Res};
use super::{check_args, done, done_with, driver, nat, str_of, usize_of, Nat};
use super::{field, insn_of, no_more, payload_role_of};
use crate::interp::{Cx, Interp, R};
use crate::value::Value;
use peacemaker_author::runtime::{Lend, Phase, Place, RingId, RingView};
use peacemaker_author::{Free, Gfx12Waits, Lendable, Published, Ring, State, Writing};
use peacemaker_isa::insn::{Instruction, MemoryClass};
use peacemaker_isa::kernels::iu4_gemm::{blocks, Gen, MetaD, MetaS, PayA, PayW, PayloadRole, Wg};

pub(super) const FNS: &[&str] = &[
    "iu4::block_begin",
    "iu4::fetch_ahead",
    "iu4::fetch_slab",
    "iu4::publish_slab_set",
    "iu4::fetch_slab1",
    "iu4::fetch_slab1_next_group",
    "iu4::fetch_next",
    "iu4::fetch_next_meta",
    "iu4::publish_slab",
    "iu4::publish_meta",
    "iu4::wait",
    "iu4::fragments_body",
    "iu4::d_loads",
    "iu4::scale_loads",
    "iu4::scale_tail",
    "iu4::wmma",
    "iu4::fold_scale",
    "iu4::sparse_block_begin",
    "iu4::sparse_fragments",
    "iu4::sparse_wmma",
    "iu4::sparse_scale_loads",
    "iu4::sparse_scale_tail",
];

pub(super) fn is_fn(name: &str) -> bool {
    FNS.contains(&name)
}

// ---- dense and sparse steps -----------------------------------------------------

/// The dense steps (`iu4::block_begin`, `fragments_body`, `wmma`, `scale_loads`,
/// `scale_tail`) are for a dense spec and their `iu4::sparse_*` twins for a
/// sparse one: the other pairing is refused before anything is emitted, so a
/// WMMA never lands in a SWMMAC kernel and the register layouts of the two
/// K-loops (and their scale-load placement) are never mixed.
fn step_spec(what: &str, g: &Gen, sparse: bool) -> Res<()> {
    let mma = g.spec().mma;
    // The dense fragment loads are a script body over `iu4::fragments_body`.
    let twin = |w: &str| match w {
        "iu4::sparse_fragments" => "iu4::fragments_body".to_string(),
        "iu4::fragments_body" => "iu4::sparse_fragments".to_string(),
        _ if sparse => w.replace("sparse_", ""),
        _ => w.replace("iu4::", "iu4::sparse_"),
    };
    match (sparse, mma.is_sparse()) {
        (true, false) => Err(format!(
            "{what}: the spec is dense: a sparse step needs an `s24x64` / `s24x32` spec (the dense step is `{}`)",
            twin(what)
        )),
        (false, true) => Err(format!(
            "{what}: the spec is {}: a dense step cannot run in the sparse K-loop (use `{}`)",
            mma.name(),
            twin(what)
        )),
        _ => Ok(()),
    }
}

/// A sparse or dense row-scale load of a block: `head` is the `_s0` group
/// (`scale_loads`), otherwise the `_s1` tail (`scale_tail`).
fn scale<NS: State>(
    wg: &mut Wg<'_>,
    g: &Gen,
    sparse: bool,
    head: bool,
    x0: &str,
    x1: &str,
    rs: &Ring<MetaS, Published, NS>,
) -> Res<()> {
    match (sparse, head) {
        (false, true) => blocks::scale_loads(wg, g, x0, rs),
        (false, false) => blocks::scale_tail(wg, g, x1, rs),
        (true, true) => blocks::sparse_scale_loads(wg, g, x0, rs),
        (true, false) => blocks::sparse_scale_tail(wg, g, x1, rs),
    }
}

// ---- ring access ----------------------------------------------------------------

/// A read of a ring's current buffer needs a published one.
fn need_published(what: &str, v: &RingView) -> Res<()> {
    if v.cur != Phase::RtPublished || !v.cur_live {
        return Err(format!(
            "{what}: ring {} current buffer is {:?}, not published: a read needs a buffer a barrier has published (publish, wait, then rotate it)",
            v.name, v.cur
        ));
    }
    Ok(())
}

/// Run a read-only step over one ring's current buffer. The ring is lent in
/// the state it is in (`Free` or `Writing` next buffer) and handed back as it
/// was, token and all.
fn read_cur<T: 'static>(
    lend: &mut Lend<'_>,
    wg: &mut Wg<'_>,
    what: &str,
    id: RingId,
    free: impl FnOnce(&mut Wg<'_>, &Ring<T, Published, Free>) -> Res<()>,
    writing: impl FnOnce(&mut Wg<'_>, &Ring<T, Published, Writing>) -> Res<()>,
) -> Res<()> {
    let v = lend.view::<T>(id).map_err(|e| format!("{what}: {e}"))?;
    need_published(what, &v)?;
    match v.next {
        Phase::RtFree => {
            let r = lend
                .take::<T, Published, Free, 2>(id)
                .map_err(|e| format!("{what}: {e}"))?;
            let out = free(wg, &r);
            lend.give_back(id, r)?;
            out
        }
        Phase::RtWriting => {
            let t = lend
                .take_writing::<T, Gfx12Waits, Published, 2>(id)
                .map_err(|e| format!("{what}: {e}"))?;
            let (r, p) = t;
            let out = writing(wg, &r);
            lend.give_writing(id, (r, p))?;
            out
        }
        Phase::RtPublished => Err(format!(
            "{what}: ring {} next buffer is Published: a ring's next buffer is Free or Writing",
            v.name
        )),
    }
}

/// The payload rings' depth (2 or 3): both must agree.
fn pay_depth(what: &str, va: &RingView, vw: &RingView) -> Res<usize> {
    if va.depth != vw.depth {
        return Err(format!(
            "{what}: rings {} and {} have depths {} and {}: a slab's A and W rings are as deep as each other",
            va.name, vw.name, va.depth, vw.depth
        ));
    }
    Ok(va.depth)
}

/// Run a depth-`D` body for the payload rings' depth, `D` a const name the
/// body may use.
macro_rules! by_depth {
    ($what:expr, $depth:expr, $d:ident => $body:expr) => {
        match $depth {
            2 => {
                const $d: usize = 2;
                $body
            }
            3 => {
                const $d: usize = 3;
                $body
            }
            n => Err(format!(
                "{}: the payload rings are {n} deep; the fine natives take 2 or 3",
                $what
            )),
        }
    };
}

/// Publish prefetch set `set` into the payload pair's `Free` next buffers; the
/// current buffers are `C` (`Published` in steady state, `Free` while a fresh
/// ring fills).
fn publish_pay<C: Lendable, const D: usize>(
    lend: &mut Lend<'_>,
    wg: &mut Wg<'_>,
    g: &Gen,
    set: usize,
    ids: [RingId; 4],
) -> Res<()> {
    let ra = lend.take::<PayA, C, Free, D>(ids[0])?;
    let rw = lend.take::<PayW, C, Free, D>(ids[1])?;
    let (sa, sw) = blocks::publish_slab_set(wg, g, set, ra, rw)?;
    lend.give_writing(ids[0], sa)?;
    lend.give_writing(ids[1], sw)?;
    Ok(())
}

/// [`publish_pay`] for the metadata pair (DS, SZ), always two deep.
fn publish_meta_pair<C: Lendable>(
    lend: &mut Lend<'_>,
    wg: &mut Wg<'_>,
    g: &Gen,
    ids: [RingId; 4],
) -> Res<()> {
    let rd = lend.take::<MetaD, C, Free, 2>(ids[2])?;
    let rs = lend.take::<MetaS, C, Free, 2>(ids[3])?;
    let (sd, ss) = blocks::publish_meta(wg, g, rd, rs)?;
    lend.give_writing(ids[2], sd)?;
    lend.give_writing(ids[3], ss)?;
    Ok(())
}

/// A read of the payload pair's current buffers, generic over the pair's
/// next-buffer state `N` and depth `D` (the lent rings' types).
trait PayRead {
    fn read<N: State, const D: usize>(
        self,
        wg: &mut Wg<'_>,
        ra: &Ring<PayA, Published, N, D>,
        rw: &Ring<PayW, Published, N, D>,
    ) -> Res<()>;
}

/// `iu4::sparse_fragments`: the Rust step.
struct SparseFragments<'g>(&'g Gen);

impl PayRead for SparseFragments<'_> {
    fn read<N: State, const D: usize>(
        self,
        wg: &mut Wg<'_>,
        ra: &Ring<PayA, Published, N, D>,
        rw: &Ring<PayW, Published, N, D>,
    ) -> Res<()> {
        blocks::sparse_fragments(wg, self.0, ra, rw)
    }
}

/// One load of a script-native body (`NativeLoad { role, insn }`): the
/// payload ring it reads, named by role (the native maps the role to the
/// sealed `PayA` / `PayW` tag), and the instruction the script built.
struct NativeLoad {
    role: PayloadRole,
    insn: Instruction,
}

/// `iu4::fragments_body`: the script's loads, in its order, each through the
/// typed `ds_load_cur` of the lent ring its role names.
struct BodyLoads(Vec<NativeLoad>);

impl PayRead for BodyLoads {
    fn read<N: State, const D: usize>(
        self,
        wg: &mut Wg<'_>,
        ra: &Ring<PayA, Published, N, D>,
        rw: &Ring<PayW, Published, N, D>,
    ) -> Res<()> {
        for l in self.0 {
            match l.role {
                PayloadRole::A => wg.ds_load_cur(ra, l.insn)?,
                PayloadRole::W => wg.ds_load_cur(rw, l.insn)?,
            }
        }
        Ok(())
    }
}

/// The loads a body passes to `iu4::fragments_body`: a `Vec` of
/// `NativeLoad { role: PayloadRole::A | PayloadRole::W, insn }`. The body's
/// capability is a read of the payload rings, so every instruction must be an
/// LDS load (`MemoryClass::DsLoad`, a `ds_load*` mnemonic) whose declared defs
/// and uses cover its decoded operands (R0603, as for every instruction a
/// script hands the driver); anything else is refused before the rings are
/// lent or anything is emitted.
fn native_loads<'a>(
    it: &Interp<'a>,
    what: &str,
    arch: peacemaker_isa::Arch,
    v: &Value<'a>,
) -> Res<Vec<NativeLoad>> {
    let Value::Array(xs) = it.deref(v.clone()) else {
        return Err(format!(
            "{what}: expected the body's loads, a `Vec<NativeLoad>` (`NativeLoad {{ role, insn }}`)"
        ));
    };
    xs.iter()
        .enumerate()
        .map(|(i, x)| {
            let n = "NativeLoad";
            let mut fs = match it.deref(x.clone()) {
                Value::Struct(s, fs) if &*s == n => fs,
                o => {
                    return Err(format!(
                        "{what}: load {i}: expected a `NativeLoad {{ role, insn }}`, found {}",
                        o.type_name()
                    ))
                }
            };
            let role = payload_role_of(it, &field(&mut fs, n, "role")?)
                .map_err(|e| format!("{what}: load {i}: {e}"))?;
            let insn = insn_of(it, &field(&mut fs, n, "insn")?)
                .map_err(|e| format!("{what}: load {i}: {e}"))?;
            no_more(&fs, n)?;
            if insn.memory != Some(MemoryClass::DsLoad) || !insn.mnemonic().starts_with("ds_load") {
                return Err(format!(
                    "{what}: load {i} `{}` is not an LDS load: a fragments body only reads the payload rings (`ds_load*` with `MemoryClass::DsLoad`)",
                    insn.text
                ));
            }
            crate::validate::instruction(arch, &insn.text, &insn.defs, &insn.uses)
                .map_err(|e| format!("{what}: load {i}: {e}"))?;
            Ok(NativeLoad { role, insn })
        })
        .collect()
}

/// The fragment loads over the payload pair (A, W), which must be published
/// and in the same state (`Free` or `Writing` next buffers): both rings are
/// lent in that state, `body` reads them, and both are given back as they
/// were, token and all.
fn fragments_pay(
    lend: &mut Lend<'_>,
    wg: &mut Wg<'_>,
    what: &str,
    ids: [RingId; 4],
    body: impl PayRead,
) -> Res<()> {
    let (va, vw) = (
        lend.view::<PayA>(ids[0])
            .map_err(|e| format!("{what}: {e}"))?,
        lend.view::<PayW>(ids[1])
            .map_err(|e| format!("{what}: {e}"))?,
    );
    need_published(what, &va)?;
    need_published(what, &vw)?;
    let depth = pay_depth(what, &va, &vw)?;
    by_depth!(what, depth, D => match (va.next, vw.next) {
        (Phase::RtFree, Phase::RtFree) => {
            let ra = lend.take::<PayA, Published, Free, D>(ids[0])
                .map_err(|e| format!("{what}: {e}"))?;
            let rw = lend.take::<PayW, Published, Free, D>(ids[1])
                .map_err(|e| format!("{what}: {e}"))?;
            let out = body.read(wg, &ra, &rw);
            lend.give_back(ids[0], ra)?;
            lend.give_back(ids[1], rw)?;
            out
        }
        (Phase::RtWriting, Phase::RtWriting) => {
            let a = lend.take_writing::<PayA, Gfx12Waits, Published, D>(ids[0])
                .map_err(|e| format!("{what}: {e}"))?;
            let w = lend.take_writing::<PayW, Gfx12Waits, Published, D>(ids[1])
                .map_err(|e| format!("{what}: {e}"))?;
            let out = body.read(wg, &a.0, &w.0);
            lend.give_writing(ids[0], a)?;
            lend.give_writing(ids[1], w)?;
            out
        }
        (a, w) => Err(format!(
            "{what}: rings {} and {} are in different states (next {a:?} and {w:?}): a slab's A and W buffers are published and waited together",
            va.name, vw.name
        )),
    })
}

/// A publish wants the pair's next buffers `Free` and the current ones
/// `Published` (steady state) or `Free` (a fresh ring, before its first `prime`).
fn need_free_next(what: &str, v: &RingView) -> Res<()> {
    if !matches!(v.cur, Phase::RtPublished | Phase::RtFree) {
        return Err(format!(
            "{what}: ring {} is [cur {:?}, next {:?}]: a publish wants its current buffer Published (steady state) or Free (a ring still filling)",
            v.name, v.cur, v.next
        ));
    }
    match v.next {
        Phase::RtFree => Ok(()),
        Phase::RtPublished => Err(format!(
            "{what}: cannot store into ring {}'s next buffer: it is Published and other waves may still be reading it until a barrier retires it",
            v.name
        )),
        Phase::RtWriting => Err(format!(
            "{what}: ring {} next buffer is Writing{}: a publish writes a Free buffer, so wait for the slab in flight and rotate it first",
            v.name,
            if v.pending { " (its stores are still pending)" } else { "" }
        )),
    }
}

// ---- the natives ----------------------------------------------------------------

pub(super) fn call<'a>(it: &Interp<'a>, name: &str, a: &[Value<'a>], cx: &mut Cx<'_>) -> R<'a> {
    let need = |n: usize| check_args(name, a, n);
    let what = name;
    match name {
        "iu4::block_begin" | "iu4::sparse_block_begin" => {
            need(7)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let sparse = name == "iu4::sparse_block_begin";
            if let Err(e) = step_spec(what, &g, sparse) {
                return Ok(done(Err(e)));
            }
            let tag = str_of(it, &a[2], "block tag")?;
            let h = usize_of(it, &a[3], "block plane")?;
            let end_label = str_of(it, &a[4], "end_label")?;
            let entry_label = bool_of(it, &a[5], "entry_label")?;
            let ids = lds_of(it, &a[6])?;
            let ids = typed(cx, |wg, lend| {
                let va = lend
                    .view::<PayA>(ids[0])
                    .map_err(|e| format!("{what}: {e}"))?;
                let vd = lend
                    .view::<MetaD>(ids[2])
                    .map_err(|e| format!("{what}: {e}"))?;
                let begin = if sparse {
                    blocks::sparse_block_begin
                } else {
                    blocks::block_begin
                };
                begin(
                    wg,
                    &g,
                    &tag,
                    h,
                    &end_label,
                    entry_label,
                    va.cur_index,
                    vd.cur_index,
                )?;
                Ok(ids)
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::fetch_slab1" | "iu4::fetch_next" | "iu4::fetch_next_meta" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let h = usize_of(it, &a[2], "block plane")?;
            let b = driver(cx)?.isa();
            Ok(done(match name {
                "iu4::fetch_slab1" => blocks::fetch_slab1(b, &g, h),
                "iu4::fetch_next" => blocks::fetch_next(b, &g, h),
                _ => blocks::fetch_next_meta(b, &g, h),
            }))
        }
        "iu4::fetch_slab1_next_group" | "iu4::fold_scale" => {
            need(2)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let b = driver(cx)?.isa();
            Ok(done(if name == "iu4::fold_scale" {
                blocks::fold_scale(b, &g)
            } else {
                blocks::fetch_slab1_next_group(b, &g)
            }))
        }
        "iu4::publish_slab" | "iu4::publish_slab_set" => {
            let set_arg = name == "iu4::publish_slab_set";
            need(if set_arg { 4 } else { 3 })?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let (set, last) = if set_arg {
                (usize_of(it, &a[2], "prefetch set")?, 3)
            } else {
                (0, 2)
            };
            let ids = lds_of(it, &a[last])?;
            let ids = typed(cx, |wg, lend| {
                let va = lend
                    .view::<PayA>(ids[0])
                    .map_err(|e| format!("{what}: {e}"))?;
                let vw = lend
                    .view::<PayW>(ids[1])
                    .map_err(|e| format!("{what}: {e}"))?;
                need_free_next(what, &va)?;
                need_free_next(what, &vw)?;
                let depth = pay_depth(what, &va, &vw)?;
                if va.cur != vw.cur {
                    return Err(format!(
                        "{what}: rings {} and {} are in different states (cur {:?} and {:?})",
                        va.name, vw.name, va.cur, vw.cur
                    ));
                }
                by_depth!(what, depth, D => {
                    if va.cur == Phase::RtFree {
                        publish_pay::<Free, D>(lend, wg, &g, set, ids)?;
                    } else {
                        publish_pay::<Published, D>(lend, wg, &g, set, ids)?;
                    }
                    Ok(ids)
                })
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::fetch_ahead" => {
            need(4)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let h = usize_of(it, &a[2], "block plane")?;
            let slab = usize_of(it, &a[3], "slab")? as u32;
            let b = driver(cx)?.isa();
            Ok(done(blocks::fetch_ahead(b, &g, h, slab)))
        }
        "iu4::fetch_slab" => {
            need(6)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let set = usize_of(it, &a[2], "prefetch set")?;
            let slab = usize_of(it, &a[3], "slab")? as u32;
            let desc = usize_of(it, &a[4], "descriptor")?;
            let woff = usize_of(it, &a[5], "weight offset")? as u32;
            let b = driver(cx)?.isa();
            Ok(done(blocks::fetch_slab(b, &g, set, slab, desc, woff)))
        }
        "iu4::publish_meta" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let ids = lds_of(it, &a[2])?;
            let ids = typed(cx, |wg, lend| {
                let vd = lend
                    .view::<MetaD>(ids[2])
                    .map_err(|e| format!("{what}: {e}"))?;
                let vs = lend
                    .view::<MetaS>(ids[3])
                    .map_err(|e| format!("{what}: {e}"))?;
                need_free_next(what, &vd)?;
                need_free_next(what, &vs)?;
                if vd.cur != vs.cur {
                    return Err(format!(
                        "{what}: rings {} and {} are in different states (cur {:?} and {:?})",
                        vd.name, vs.name, vd.cur, vs.cur
                    ));
                }
                if vd.cur == Phase::RtFree {
                    publish_meta_pair::<Free>(lend, wg, &g, ids)?;
                } else {
                    publish_meta_pair::<Published>(lend, wg, &g, ids)?;
                }
                Ok(ids)
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::wait" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let ids = lds_of(it, &a[1])?;
            let which = str_of(it, &a[2], "ring set")?;
            let set: &[usize] = match which.as_str() {
                "slab" => &[0, 1],
                "meta" => &[2, 3],
                "all" => &[0, 1, 2, 3],
                o => {
                    return Err(format!(
                        "{what}: `{o}` is not a ring set (\"slab\" = A and W, \"meta\" = DS and SZ, \"all\")"
                    )
                    .into())
                }
            };
            let places: Vec<Place> = set.iter().map(|&i| Place::Ring(ids[i])).collect();
            let r = driver(cx)?.wait_all(&places);
            Ok(done_with(
                r.map_err(|e| format!("{what}(\"{which}\"): {e}")),
                |()| Value::Tuple(places.iter().map(|p| nat(Nat::Drained(*p))).collect()),
            ))
        }
        "iu4::sparse_fragments" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            if let Err(e) = step_spec(what, &g, true) {
                return Ok(done(Err(e)));
            }
            let ids = lds_of(it, &a[2])?;
            let ids = typed(cx, |wg, lend| {
                fragments_pay(lend, wg, what, ids, SparseFragments(&g))?;
                Ok(ids)
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::fragments_body" => {
            need(4)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            if let Err(e) = step_spec(what, &g, false) {
                return Ok(done(Err(e)));
            }
            let ids = lds_of(it, &a[2])?;
            // The loads are values the script built before this call: no
            // evaluation runs while the rings are lent.
            let loads = match native_loads(it, what, g.spec().arch, &a[3]) {
                Ok(l) => l,
                Err(e) => return Ok(done(Err(e))),
            };
            let ids = typed(cx, |wg, lend| {
                fragments_pay(lend, wg, what, ids, BodyLoads(loads))?;
                Ok(ids)
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::d_loads" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let ids = lds_of(it, &a[2])?;
            let ids = typed(cx, |wg, lend| {
                read_cur::<MetaD>(
                    lend,
                    wg,
                    what,
                    ids[2],
                    |wg, rd| blocks::d_loads(wg, &g, rd),
                    |wg, rd| blocks::d_loads(wg, &g, rd),
                )?;
                Ok(ids)
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::scale_loads"
        | "iu4::scale_tail"
        | "iu4::sparse_scale_loads"
        | "iu4::sparse_scale_tail" => {
            need(4)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let sparse = name.starts_with("iu4::sparse_");
            if let Err(e) = step_spec(what, &g, sparse) {
                return Ok(done(Err(e)));
            }
            let tag = str_of(it, &a[2], "block tag")?;
            let ids = lds_of(it, &a[3])?;
            let (_, x0, x1) = blocks::sub_labels(&tag);
            let head = name.ends_with("scale_loads");
            let ids = typed(cx, |wg, lend| {
                read_cur::<MetaS>(
                    lend,
                    wg,
                    what,
                    ids[3],
                    |wg, rs| scale(wg, &g, sparse, head, &x0, &x1, rs),
                    |wg, rs| scale(wg, &g, sparse, head, &x0, &x1, rs),
                )?;
                Ok(ids)
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::wmma" | "iu4::sparse_wmma" => {
            need(6)?;
            wg_arg(it, &a[0])?;
            let g: std::rc::Rc<Gen> = gen_of(it, &a[1])?;
            let sparse = name == "iu4::sparse_wmma";
            if let Err(e) = step_spec(what, &g, sparse) {
                return Ok(done(Err(e)));
            }
            let sb = usize_of(it, &a[2], "sub-block")?;
            let first = bool_of(it, &a[3], "first")?;
            let fold = bool_of(it, &a[4], "fold")?;
            let ids = lds_of(it, &a[5])?;
            // The group consumes the fragments of the payload rings' current
            // slot: refuse it when nothing is published there. (`typed` is
            // refused inside a split barrier's window, the driver's phases
            // are not.)
            for (i, ring) in [("A", ids[0]), ("W", ids[1])] {
                let gate = driver(cx)?.ring_phases(ring);
                let err = match gate {
                    Ok((Phase::RtPublished, _)) => continue,
                    Ok((cur, _)) => format!(
                        "{what}: payload ring {i}'s current buffer is {cur:?}, not published: a WMMA group consumes fragments of a published slot"
                    ),
                    Err(e) => format!("{what}: {e}"),
                };
                return Ok(done(Err(err)));
            }
            let b = driver(cx)?.isa();
            let r = if sparse {
                blocks::sparse_wmma_group(b, &g, sb, first, fold)
            } else {
                blocks::wmma_group(b, &g, sb, first, fold)
            };
            Ok(done_with(r.map(|()| ids), lds_val))
        }
        n => Err(format!("native function `{n}` is not implemented").into()),
    }
}
