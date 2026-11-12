//! Compare every launch site's real params array and kernarg blob against the
//! independently declared ABI, and account for every function that touches a
//! changed kernel.

use crate::decls::*;
use crate::hip::Decl;
use crate::layout::*;
use crate::lex::*;
use crate::scan::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Byte layout the blob builder produces, replayed through the real
/// `hip_bridge::KernargBlob` so padding rules are the production ones.
pub struct BlobArg {
    pub offset: u32,
    pub size: u32,
    pub kind: char,
    pub expr: Vec<Tt>,
}

pub fn blob_layout(
    world: &World,
    ctx: &Ctx,
    pushes: &[Push],
    sloc: &[usize],
) -> Result<(Vec<BlobArg>, u32), String> {
    let mut blob = hip_bridge::KernargBlob::new();
    let mut out = Vec::new();
    for p in pushes {
        let (size, kind) = match p.kind {
            PushKind::Ptr => (8u32, 'p'),
            PushKind::U32 | PushKind::I32 => (4, 'i'),
            PushKind::F32 => (4, 'f'),
            PushKind::U64 => {
                // A u64 word carries a pointer only when its value provably is one
                // (`.as_ptr() as u64`, a raw-pointer cast, or a local initialised so).
                let ptr = is_ptr_flow(&p.arg)
                    || arg_ident(&p.arg)
                        .filter(|_| p.arg.len() == 1)
                        .and_then(|id| world.local_kind(ctx, &id, sloc, 0))
                        .map_or(false, |k| k == (8, 'p'));
                (8, if ptr { 'p' } else { 'i' })
            }
            PushKind::Val => {
                let id = arg_ident(&p.arg).ok_or_else(|| {
                    format!(
                        "launch_params_blob! value `{}` is not a plain local",
                        text(&p.arg)
                    )
                })?;
                let id = id.trim_end_matches("[]").to_string();
                world.local_kind(ctx, &id, sloc, 0).ok_or_else(|| {
                    format!("launch_params_blob! value type of `{id}` is unresolved")
                })?
            }
        };
        let before = blob.len() as u32;
        let offset = (before + size - 1) / size * size;
        match (p.kind, size) {
            (PushKind::Ptr, _) => blob.push_ptr(std::ptr::null()),
            (PushKind::U32, _) => blob.push_u32(0),
            (PushKind::I32, _) => blob.push_i32(0),
            (PushKind::F32, _) => blob.push_f32(0.0),
            (PushKind::U64, _) => blob.push_u64(0),
            (PushKind::Val, 8) if kind == 'p' => blob.push_ptr(std::ptr::null()),
            (PushKind::Val, 8) => blob.push_u64(0),
            (PushKind::Val, _) if kind == 'f' => blob.push_f32(0.0),
            (PushKind::Val, _) => blob.push_u32(0),
        }
        if blob.len() as u32 != offset + size {
            return Err(format!(
                "internal: replayed blob end {} != computed {}",
                blob.len(),
                offset + size
            ));
        }
        out.push(BlobArg {
            offset,
            size,
            kind,
            expr: p.arg.clone(),
        });
    }
    Ok((out, blob.len() as u32))
}

/// Problems of one (variant, declaration) pair; empty = consistent.
pub fn compare(world: &World, ctx: &Ctx, v: &Variant, sloc: &[usize], decl: &Decl) -> Vec<String> {
    if let Some(bl) = &v.bytes {
        return compare_bytes(bl, decl);
    }
    let (lay, end) = match blob_layout(world, ctx, &v.pushes, sloc) {
        Ok(x) => x,
        Err(e) => return vec![format!("ERR {e}")],
    };
    let mut probs = compare_blob(&lay, end, decl);
    let da = &decl.args;
    if v.has_params {
        match params_aggregate_refusal(da) {
            None => compare_params(world, ctx, v, sloc, &lay, da, &mut probs),
            Some(why) => probs.push(why),
        }
    }
    probs
}

/// A params array hands the runtime one pointer per declared argument, a by-value aggregate
/// included as a single whole-aggregate entry; the blob builder pushes its scalar leaves one at
/// a time. Those two shapes cannot be tied entry to entry, so a params launch of a declared
/// aggregate argument is refused rather than flattened.
fn params_aggregate_refusal(da: &[crate::hip::Arg]) -> Option<String> {
    let aggs: Vec<usize> = da
        .iter()
        .enumerate()
        .filter(|(_, d)| !d.fields.is_empty())
        .map(|(i, _)| i)
        .collect();
    (!aggs.is_empty()).then(|| {
        format!(
            "kernelParams launch of by-value aggregate argument(s) {aggs:?} cannot be proven \
             against per-leaf blob pushes (a params entry must point at the whole aggregate)"
        )
    })
}

/// One expected scalar slot of the declared kernarg layout.
struct Leaf {
    arg: usize,
    /// (field index, element index) inside an aggregate argument.
    within: Option<(usize, u32)>,
    offset: u32,
    size: u32,
    kind: char,
}

/// Declared scalar slots in blob order, produced lazily: an aggregate contributes one slot per
/// array element of each source scalar field, so a large array is never materialised. Every
/// aggregate must already have passed `check_aggregate`, which bounds each leaf end by the
/// aggregate end, itself within `u32`, so the offset arithmetic below cannot overflow.
fn leaves<'a>(da: &'a [crate::hip::Arg]) -> impl Iterator<Item = Leaf> + 'a {
    da.iter().enumerate().flat_map(|(i, d)| {
        let scalar = d.fields.is_empty().then(|| Leaf {
            arg: i,
            within: None,
            offset: d.offset,
            size: d.size,
            kind: d.kind,
        });
        let fields = d.fields.iter().enumerate().flat_map(move |(fi, f)| {
            (0..f.count).map(move |j| Leaf {
                arg: i,
                within: Some((fi, j)),
                offset: f.offset + j * f.size,
                size: f.size,
                kind: f.kind,
            })
        });
        scalar.into_iter().chain(fields)
    })
}

/// A source-proven aggregate must describe its bytes exactly: scalar leaves of a known class in
/// increasing order, the first at the aggregate's start, no overlap, no hole wider than natural
/// alignment padding, and no tail beyond alignment padding or past the aggregate's size.
fn check_aggregate(i: usize, d: &crate::hip::Arg) -> Result<(), String> {
    let bad = |why: String| -> Result<(), String> {
        Err(format!("arg {i}: malformed aggregate declaration: {why}"))
    };
    if d.kind != 'v' {
        return bad(format!("aggregate has kind `{}`, not `v`", d.kind));
    }
    let agg_end = u64::from(d.offset) + u64::from(d.size);
    if agg_end > u64::from(u32::MAX) {
        return bad(format!(
            "aggregate at offset {} of {} bytes ends at {agg_end}, past the 32-bit kernarg offset range",
            d.offset, d.size
        ));
    }
    let mut max_size = 0u32;
    for f in &d.fields {
        if f.count == 0 || f.size == 0 || !matches!(f.kind, 'p' | 'i' | 'f') {
            return bad(format!(
                "field at offset {} has count {}, size {}, kind `{}`",
                f.offset, f.count, f.size, f.kind
            ));
        }
        max_size = max_size.max(f.size);
    }
    let mut cursor = u64::from(d.offset);
    for (fi, f) in d.fields.iter().enumerate() {
        let start = u64::from(f.offset);
        let stop = start + u64::from(f.size) * u64::from(f.count);
        if fi == 0 {
            if start != cursor {
                return bad(format!(
                    "first field at offset {start}, aggregate starts at {cursor}"
                ));
            }
        } else if start < cursor {
            return bad(format!(
                "field {fi} at offset {start} overlaps bytes before {cursor}"
            ));
        } else if start - cursor >= u64::from(max_size) {
            return bad(format!(
                "field {fi} at offset {start} leaves a {}-byte hole after {cursor}",
                start - cursor
            ));
        }
        cursor = stop;
    }
    if cursor > agg_end {
        return bad(format!(
            "fields end at {cursor}, past aggregate end {agg_end}"
        ));
    }
    if agg_end - cursor >= u64::from(max_size) {
        return bad(format!(
            "fields end at {cursor}, {} bytes short of aggregate end {agg_end}",
            agg_end - cursor
        ));
    }
    Ok(())
}

/// Compare the replayed raw blob against the declared kernarg layout. Scalar arguments match
/// one push each; a by-value aggregate with source-proven scalar leaves (`fields`) matches
/// one push per leaf at the leaf's exact absolute offset, size and class. An aggregate without
/// `fields` (PeaceMaker/HSA metadata) keeps one-push-per-argument handling.
fn compare_blob(lay: &[BlobArg], end: u32, decl: &Decl) -> Vec<String> {
    let mut probs = Vec::new();
    let da = &decl.args;
    for (i, d) in da.iter().enumerate() {
        if !d.fields.is_empty() {
            if let Err(e) = check_aggregate(i, d) {
                probs.push(e);
            }
        }
    }
    if !probs.is_empty() {
        return probs;
    }
    let expected: u64 = da
        .iter()
        .map(|d| {
            if d.fields.is_empty() {
                1
            } else {
                d.fields.iter().map(|f| u64::from(f.count)).sum()
            }
        })
        .sum();
    if lay.len() as u64 != expected {
        probs.push(format!(
            "raw blob has {} arguments, declared {}",
            lay.len(),
            if expected == da.len() as u64 {
                format!("{expected}")
            } else {
                format!(
                    "{expected} ({} arguments, aggregate fields counted per scalar leaf)",
                    da.len()
                )
            }
        ));
        return probs;
    }
    const LEAF_REPORT_CAP: usize = 8;
    /// Why one slot disagrees; the text is only built once a mismatch is actually reported.
    #[derive(Clone, Copy)]
    enum Why {
        Geometry,
        LeafClass,
        PointerScalar,
        FloatInt,
    }
    let mut leaf_probs = 0usize;
    for (b, l) in lay.iter().zip(leaves(da)) {
        let why = if b.size != l.size || b.offset != l.offset {
            Some(Why::Geometry)
        } else if l.within.is_some() {
            (b.kind != l.kind).then_some(Why::LeafClass)
        } else if (l.kind == 'p') != (b.kind == 'p') && l.kind != 'v' {
            Some(Why::PointerScalar)
        } else if (l.kind == 'f' && b.kind == 'i') || (l.kind == 'i' && b.kind == 'f') {
            Some(Why::FloatInt)
        } else {
            None
        };
        let Some(why) = why else { continue };
        if l.within.is_some() {
            leaf_probs += 1;
            if leaf_probs > LEAF_REPORT_CAP {
                continue;
            }
        }
        let i = l.arg;
        let name = match l.within {
            None => format!("arg {i}"),
            Some((fi, j)) => format!("arg {i} field {fi} element {j}"),
        };
        probs.push(match why {
            Why::Geometry => format!(
                "{name}: blob (offset {}, {} bytes) vs declared (offset {}, {} bytes)",
                b.offset, b.size, l.offset, l.size
            ),
            Why::LeafClass => format!(
                "{name}: aggregate leaf class mismatch (blob {} vs declared {})",
                b.kind, l.kind
            ),
            Why::PointerScalar => format!(
                "{name}: pointer/scalar class mismatch (blob {} vs declared {})",
                b.kind, l.kind
            ),
            Why::FloatInt => format!(
                "{name}: float/integer class mismatch (blob {} vs declared {})",
                b.kind, l.kind
            ),
        });
    }
    if leaf_probs > LEAF_REPORT_CAP {
        probs.push(format!(
            "{} further aggregate leaf mismatches not listed",
            leaf_probs - LEAF_REPORT_CAP
        ));
    }
    // The blob may stop at the last scalar leaf of a trailing aggregate: the bytes after it are
    // the aggregate's own tail padding, which no kernel load reads.
    let tail_leaf_end = da.last().and_then(|d| {
        let f = d.fields.last()?;
        u32::try_from(u64::from(f.offset) + u64::from(f.size) * u64::from(f.count)).ok()
    });
    if end != decl.end && Some(end) != tail_leaf_end {
        probs.push(format!(
            "raw blob ends at byte {end}, declared kernarg bytes end at {}",
            decl.end
        ));
    }
    probs
}

fn compare_params(
    world: &World,
    ctx: &Ctx,
    v: &Variant,
    sloc: &[usize],
    lay: &[BlobArg],
    da: &[crate::hip::Arg],
    probs: &mut Vec<String>,
) {
    let pi = &v.param_idents;
    if pi.len() != da.len() {
        probs.push(format!(
            "kernelParams array has {} entries, declared {}",
            pi.len(),
            da.len()
        ));
    } else if v.pushes.len() == pi.len() && lay.len() == pi.len() {
        for (i, (a, b)) in pi.iter().zip(lay.iter()).enumerate() {
            if let Some((value, loc)) = a.as_ref().and_then(|name| {
                mapped_param_value(ctx, name, sloc)
                    .or_else(|| repeat_param_value(world, ctx, name, sloc))
            }) {
                match world.infer_kind(&value, ctx, &loc, 0) {
                    Some(kind) if kind == (b.size, b.kind) => {}
                    kind => probs.push(format!(
                        "arg {i}: mapped kernelParams value `{}` has type {kind:?}, blob has ({}, {})",
                        text(&value), b.size, b.kind
                    )),
                }
                if arg_ident(&b.expr).as_ref() != a.as_ref() && text(&value) != text(&b.expr) {
                    probs.push(format!(
                        "arg {i}: mapped kernelParams reads `{}` but the blob pushes `{}`",
                        text(&value),
                        text(&b.expr)
                    ));
                }
                continue;
            }
            let bi = arg_ident(&b.expr);
            if let (Some(a), Some(bi)) = (a, &bi) {
                let a_arr = a.ends_with("[]");
                let b_arr = bi.ends_with("[]");
                if !(a_arr && !b_arr) && a.trim_end_matches("[]") != bi.trim_end_matches("[]") {
                    probs.push(format!(
                        "arg {i}: kernelParams points at `{a}` but the blob pushes `{bi}`"
                    ));
                }
            }
            if let Some(a) = a {
                if let Some((sz, _)) = world.local_kind(ctx, a.trim_end_matches("[]"), sloc, 0) {
                    if sz != b.size {
                        probs.push(format!("arg {i}: kernelParams local `{a}` is {sz} bytes but the blob pushes {} bytes", b.size));
                    }
                }
            }
        }
    }
}

/// A declared argument the kernel ignores by design (a `reserved` pad word).
fn is_reserved(a: &crate::hip::Arg) -> bool {
    let name = a.text.split(':').next().unwrap_or("");
    name.rsplit(|c: char| c.is_whitespace() || c == '*' || c == '&')
        .next()
        .map_or(false, |w| w.starts_with("reserved"))
}

/// Hand-written kernarg bytes: every write must land exactly on one declared scalar argument
/// or one scalar leaf of a source-proven aggregate, every non-reserved argument and every leaf
/// must be written, and the byte array must cover the declared kernarg size. A single write
/// covering a whole aggregate is an opaque write whose contents cannot be checked per leaf, so
/// it is refused; aggregates with empty `fields` (metadata) keep exact whole-argument matching.
fn compare_bytes(bl: &ByteLayout, decl: &Decl) -> Vec<String> {
    let mut probs = Vec::new();
    // Aligned in u64: `decl.end` may legitimately be `u32::MAX` (a huge source array).
    let max_len = (u64::from(decl.end) + 7) / 8 * 8;
    if bl.len < decl.end || u64::from(bl.len) > max_len {
        probs.push(format!(
            "kernarg byte array is {} bytes, declared kernarg bytes end at {}",
            bl.len, decl.end
        ));
    }
    for (i, a) in decl.args.iter().enumerate() {
        if !a.fields.is_empty() {
            if let Err(e) = check_aggregate(i, a) {
                probs.push(e);
            }
        }
    }
    if probs.iter().any(|p| p.contains("malformed aggregate")) {
        return probs;
    }
    let writes: BTreeSet<(u32, u32)> = bl.writes.iter().copied().collect();
    let in_field = |f: &crate::hip::Field, off: u32, w: u32| {
        w == f.size
            && off >= f.offset
            && (off - f.offset) % f.size == 0
            && (off - f.offset) / f.size < f.count
    };
    for &(off, w) in &writes {
        let declared = decl.args.iter().any(|a| {
            if a.fields.is_empty() {
                a.offset == off && a.size == w
            } else {
                a.fields.iter().any(|f| in_field(f, off, w))
            }
        });
        if declared {
            continue;
        }
        match decl
            .args
            .iter()
            .position(|a| !a.fields.is_empty() && a.offset == off && a.size == w)
        {
            Some(i) => probs.push(format!(
                "kernarg write of {w} bytes at offset {off} covers aggregate arg {i} as one \
                 opaque write; its scalar leaves must be written individually"
            )),
            None => probs.push(format!(
                "kernarg write of {w} bytes at offset {off} matches no declared argument"
            )),
        }
    }
    for (i, a) in decl.args.iter().enumerate() {
        if a.fields.is_empty() {
            if !writes.contains(&(a.offset, a.size)) && !is_reserved(a) {
                probs.push(format!(
                    "arg {i} (offset {}, {} bytes) is never written into the kernarg bytes",
                    a.offset, a.size
                ));
            }
            continue;
        }
        // Walk the leaves lazily; at most `writes.len()` of them can be written, so a leaf
        // count above that is a shortfall without visiting every element.
        let total: u64 = a.fields.iter().map(|f| u64::from(f.count)).sum();
        if total > writes.len() as u64 {
            probs.push(format!(
                "aggregate arg {i} has {total} scalar leaves but only {} distinct kernarg writes exist",
                writes.len()
            ));
            continue;
        }
        let mut missing = 0usize;
        for (fi, f) in a.fields.iter().enumerate() {
            for j in 0..f.count {
                let off = f.offset + j * f.size;
                if !writes.contains(&(off, f.size)) {
                    if missing < 8 {
                        probs.push(format!(
                            "arg {i} field {fi} element {j} (offset {off}, {} bytes) is never written into the kernarg bytes",
                            f.size
                        ));
                    }
                    missing += 1;
                }
            }
        }
        if missing > 8 {
            probs.push(format!(
                "aggregate arg {i}: {} further leaves never written",
                missing - 8
            ));
        }
    }
    probs
}

/// A launch whose kernarg image comes from a validated `KernelAbiV1` contract
/// (`kernel_params(&mut v, &mut kparams)` + `kernel_abi::blob(v.explicit())`) instead
/// of hand-written pushes. Its layout is the contract's field table of each candidate
/// symbol; `kernel_abi_drift` proves that table against the generator and the object
/// metadata, and `compare_bytes` here proves it against the declared HIP/PM arguments,
/// so a symbol with no contract export fails closed.
fn contract_site_variants(site: &Site, cands: &[String]) -> Option<Result<Vec<Variant>, String>> {
    let squash = |t: &[Tt]| {
        text(t)
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
    };
    if site.kind != 'm'
        || squash(&site.params_ex) != "&mutkparams[..nk]"
        || !squash(&site.closure_ex).contains("crate::kernel_abi::blob(v.explicit())")
    {
        return None;
    }
    let mut out: Vec<Variant> = Vec::new();
    for c in cands {
        let mut layouts: Vec<ByteLayout> = Vec::new();
        for arch in hipfire_kernel_abi::Arch::ALL {
            if let Some((abi, _)) = hipfire_kernel_abi::families::find_export(arch, c) {
                let bl = ByteLayout {
                    len: abi.explicit_bytes,
                    writes: abi.fields.iter().map(|f| (f.offset, f.width)).collect(),
                };
                if !layouts
                    .iter()
                    .any(|l| l.len == bl.len && l.writes == bl.writes)
                {
                    layouts.push(bl);
                }
            }
        }
        if layouts.is_empty() {
            return Some(Err(format!(
                "contract launch of `{c}`: no KernelAbiV1 export carries this symbol"
            )));
        }
        for bl in layouts {
            out.push(Variant {
                assign: BTreeMap::new(),
                param_idents: Vec::new(),
                pushes: Vec::new(),
                has_params: false,
                bytes: Some(bl),
                only_for: Some(vec![c.clone()]),
            });
        }
    }
    Some(Ok(out))
}

/// Outcome for one function.
#[derive(Default, Debug)]
pub struct FnReport {
    pub file: String,
    pub func: String,
    pub line: u32,
    /// (site index, candidate symbols) of launch sites that resolve to changed kernels.
    pub sites: Vec<(usize, Vec<String>)>,
    pub problems: Vec<String>,
    /// Hard failures: unsupported launch shapes, unresolved types, unknown primitives.
    pub errors: Vec<String>,
    pub inert: bool,
    pub is_test: bool,
    /// Changed symbols the function mentions directly (literals or constants).
    pub mentions: BTreeSet<String>,
    pub mentions_changed_source: bool,
    pub forwards_to_param_helper: Vec<String>,
    /// A probe that mentions changed kernels, launches nothing itself, and calls a launching
    /// helper of its own file.
    pub forwards_local: bool,
    /// Cleared: every launching helper it forwards to is checked and covers its mentions.
    pub cleared: bool,
}

pub struct Analysis {
    pub reports: Vec<FnReport>,
    pub covered: BTreeSet<String>,
    pub checked_pairs: usize,
    pub checked_sites: usize,
    pub old_used: BTreeSet<String>,
    pub probe_used: BTreeSet<String>,
    pub policy_errors: Vec<String>,
}

/// Cheap token scan: can this function launch or mention a changed kernel at all?
/// (a launch primitive, a changed symbol literal, or a constant of this file / kernels.rs).
fn quick_relevant(
    ts: &[Tt],
    names: &BTreeSet<String>,
    local: &std::collections::HashMap<String, Vec<ConstRec>>,
    kernels: &std::collections::HashMap<String, Vec<ConstRec>>,
) -> bool {
    ts.iter().any(|t| match t {
        Tt::Id(s, _) => {
            s.starts_with("launch_") || local.contains_key(s) || kernels.contains_key(s)
        }
        Tt::S(s, _) => names.contains(s),
        Tt::G(_, c, _) => quick_relevant(c, names, local, kernels),
        _ => false,
    })
}

/// Project a locally returned descriptor field in its constructing function's
/// context. The file-wide field resolver cannot bind `tile` in `tile.symbols`:
/// it deliberately resolves field initializers in an empty context.
fn descriptor_field_names(
    world: &World,
    ctx: &Ctx,
    ex: &[Tt],
    sloc: &[usize],
    field: Option<&str>,
    depth: usize,
) -> Option<Res> {
    if depth > 24 {
        let mut r = Res::default();
        r.unres.insert("<depth>".into());
        return Some(r);
    }
    if let Some(name) = ex.first().and_then(Tt::id).filter(|_| ex.len() == 1) {
        if !ctx.idx.assigns.contains_key(name) {
            if let Some(b) = ctx.idx.visible(name, sloc) {
                // Tuple bindings still need the existing component resolver.
                if b.let_pos.is_none() && b.for_pos.is_none() {
                    return descriptor_field_names(world, ctx, &b.init, &b.loc, field, depth + 1);
                }
            }
        }
    }
    if field.is_none()
        && ex.len() >= 3
        && ex[0].id().is_some()
        && ex[1].is_p('.')
        && ex[2].id().is_some()
        && (ex.len() == 3 || ex[3].is_g('['))
    {
        if let Some(r) = descriptor_field_names(world, ctx, &ex[..1], sloc, ex[2].id(), depth + 1) {
            return Some(r);
        }
    }
    let Some(field) = field else {
        return Some(world.resolve(ex, ctx, sloc, 0, true));
    };
    let mut result = Res::default();
    let mut found = false;
    // Only inspect the returned value, not descriptors discarded by preceding
    // statements. Keep original indices so constructor-local bindings stay scoped.
    let start = ex.iter().rposition(|t| t.is_p(';')).map_or(0, |i| i + 1);
    // Follow actual calls, never origin mentions or policy-supplied candidates.
    for (i, token) in ex.iter().enumerate().skip(start) {
        if let Some(name) = token.id().filter(|_| {
            ex.get(i + 1).map_or(false, |t| t.is_g('(')) && (i == 0 || !ex[i - 1].is_p('.'))
        }) {
            let defs: Vec<_> = world
                .files
                .iter()
                .flat_map(|f| f.fns.iter().map(move |d| (f, d)))
                .filter(|(_, d)| d.name == name)
                .collect();
            if defs.len() == 1 {
                let (file, func) = defs[0];
                if !func
                    .attrs
                    .iter()
                    .any(|a| attr_is_cfg_any_empty(a) || attr_is_test(a))
                {
                    let idx = BindIndex::build(&func.body);
                    let callee = Ctx {
                        file: &file.path,
                        body: &func.body,
                        params: fn_params(func),
                        idx: &idx,
                    };
                    if let Some(r) = descriptor_field_names(
                        world,
                        &callee,
                        &func.body,
                        &[usize::MAX],
                        Some(field),
                        depth + 1,
                    ) {
                        result.merge(&r);
                        found = true;
                    }
                }
            }
        }
        if let Tt::G(_, body, _) = token {
            // A named struct literal's requested field, including shorthand.
            if token.is_g('{') && i > 0 && ex[i - 1].id().is_some() {
                for part in split_top(body, ',') {
                    if part.first().map_or(false, |t| t.is_id(field))
                        && (part.len() == 1 || part.get(1).map_or(false, |t| t.is_p(':')))
                    {
                        let value = if part.len() == 1 {
                            &part[..]
                        } else {
                            &part[2..]
                        };
                        let mut loc = sloc[..sloc.len().saturating_sub(1)].to_vec();
                        loc.push(i);
                        loc.push(usize::MAX);
                        if let Some(r) =
                            descriptor_field_names(world, ctx, value, &loc, None, depth + 1)
                        {
                            result.merge(&r);
                            found = true;
                        }
                    }
                }
            }
            let mut loc = sloc[..sloc.len().saturating_sub(1)].to_vec();
            loc.push(i);
            loc.push(usize::MAX);
            if let Some(r) = descriptor_field_names(world, ctx, body, &loc, Some(field), depth + 1)
            {
                result.merge(&r);
                found = true;
            }
        }
    }
    found.then_some(result)
}

/// One call of a launch helper: its blob layouts against the kernels that call names.
/// Returns (ABI problems, unsupported-shape errors, pairs checked).
fn check_group(
    world: &World,
    ctx: &Ctx,
    sloc: &[usize],
    kernels: &[String],
    variants: &[Variant],
    syms: &Symbols,
) -> (Vec<String>, Vec<String>, usize) {
    let mut probs = Vec::new();
    let mut errs = Vec::new();
    let mut pairs = 0usize;
    let mut decls: Vec<(String, String, Decl)> = Vec::new();
    for k in kernels {
        for (o, d) in syms.decls(k) {
            decls.push((k.clone(), o, d));
        }
    }
    if decls.is_empty() {
        errs.push(format!("no declaration found for {kernels:?}"));
        return (probs, errs, pairs);
    }
    let differ = decls
        .iter()
        .any(|(_, _, d)| !crate::hip::same_layout(&decls[0].2, d));
    if kernels.len() > 1 && differ {
        errs.push(format!(
            "the call names several kernels {kernels:?} with different ABIs, so its blob cannot be tied to one of them"
        ));
        return (probs, errs, pairs);
    }
    for (c, o, d) in &decls {
        pairs += 1;
        let mut best: Option<Vec<String>> = None;
        for v in variants {
            let p = compare(world, ctx, v, sloc, d);
            if p.is_empty() {
                best = Some(Vec::new());
                break;
            }
            if best.as_ref().map_or(true, |b| p.len() < b.len()) {
                best = Some(p);
            }
        }
        if let Some(p) = best {
            if !p.is_empty() {
                probs.push(format!("`{c}` [{o}]: {}", p.join("; ")));
            }
        }
    }
    for v in variants {
        if !decls
            .iter()
            .any(|(_, _, d)| compare(world, ctx, v, sloc, d).is_empty())
        {
            probs.push(format!(
                "a blob layout of the call matches none of {kernels:?}"
            ));
        }
    }
    (probs, errs, pairs)
}

fn has_launch_ident(ts: &[Tt]) -> bool {
    ts.iter().any(|t| match t {
        Tt::Id(s, _) => PRIMITIVES.contains(&s.as_str()),
        Tt::G(_, c, _) => has_launch_ident(c),
        _ => false,
    })
}

pub(super) fn call_names(ts: &[Tt], out: &mut BTreeSet<String>) {
    for k in 0..ts.len() {
        if let Tt::Id(s, _) = &ts[k] {
            if ts.get(k + 1).map_or(false, |t| t.is_g('(')) {
                out.insert(s.clone());
            }
        }
        if let Tt::G(_, c, _) = &ts[k] {
            call_names(c, out);
        }
    }
}

const PRIMITIVES: [&str; 8] = [
    "launch_maybe_blob",
    "launch_maybe_blob_declared",
    "launch_maybe_blob_position_grid",
    "launch_maybe_blob_bound",
    "launch_params_blob",
    "launch_kernel_blob",
    "launch_blob_recorded",
    "launch_kernel",
];

pub fn analyze(world: &World, syms: &Symbols, policy: &Policy) -> Analysis {
    let empty: std::collections::HashMap<String, Vec<ConstRec>> = std::collections::HashMap::new();
    let kernel_consts = world.file(KERNELS_RS).map(|f| &f.consts).unwrap_or(&empty);
    let names = syms.all_names();
    world.set_names(names.clone());
    // Forwarders extend their launcher's candidate set with the symbols they mention.
    let mut extra: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (k, (to, _)) in &policy.forwarders {
        if let Some((path, fname)) = k.rsplit_once("::") {
            if let Some(file) = world.file(path) {
                for func in file.fns.iter().filter(|f| f.name == fname) {
                    let idx = BindIndex::build(&func.body);
                    let ctx = Ctx {
                        file: &file.path,
                        body: &func.body,
                        params: fn_params(func),
                        idx: &idx,
                    };
                    let m = world.resolve(&func.body, &ctx, &[usize::MAX], 0, false);
                    let mut lits: Vec<String> = Vec::new();
                    strings_in(&func.body, &mut lits);
                    for l in lits.iter().chain(m.lits.iter()) {
                        if names.contains(l) {
                            extra.entry(to.clone()).or_default().insert(l.clone());
                        }
                    }
                }
            }
        }
    }
    let mut reports = Vec::new();
    let mut covered: BTreeSet<String> = BTreeSet::new();
    let mut checked_pairs = 0usize;
    let mut checked_sites = 0usize;
    // Functions that launch a kernel named by one of their own parameters.
    let mut param_helpers: BTreeSet<String> = BTreeSet::new();
    let mut helper_defs: Vec<(String, String)> = Vec::new();
    let mut old_used: BTreeSet<String> = BTreeSet::new();
    let mut probe_used: BTreeSet<String> = BTreeSet::new();

    // OLD-only launch helpers: verified, then not treated as launchers a caller forwards to.
    let mut policy_errors: Vec<String> = Vec::new();
    let mut old_helper_fns: BTreeSet<(String, String)> = BTreeSet::new();
    for key in policy.old_helpers.keys() {
        let Some((path, fname)) = key.rsplit_once("::") else {
            policy_errors.push(format!("old_helpers entry {key} is not `file::fn`"));
            continue;
        };
        let fns: Vec<&FnRec> = world
            .file(path)
            .map(|f| f.fns.iter().filter(|x| x.name == fname).collect())
            .unwrap_or_default();
        if fns.is_empty() {
            policy_errors.push(format!("stale old_helpers entry {key}: no such function"));
            continue;
        }
        for func in fns {
            match find_sites(&func.body) {
                Err(e) => policy_errors.push(format!("old_helpers {key}: {e}")),
                Ok(sites) => {
                    if sites.is_empty() {
                        policy_errors.push(format!(
                            "stale old_helpers entry {key}: the function has no launch site"
                        ));
                    }
                    let idx = BindIndex::build(&func.body);
                    let params = fn_params(func);
                    if let Err(e) = old_suffix_is_beta(world, path) {
                        policy_errors.push(format!("old_helpers {key}: {e}"));
                    }
                    for s in &sites {
                        if !name_ends_with_const(&s.name_ex, "OLD_SUFFIX", &idx, &params, 0) {
                            policy_errors.push(format!(
                                "old_helpers {key}: launch at line {} does not build its kernel name as `<sym>{{OLD_SUFFIX}}` (named override or shadowing of OLD_SUFFIX is refused)",
                                s.line
                            ));
                        }
                    }
                }
            }
        }
        old_helper_fns.insert((path.to_string(), fname.to_string()));
    }
    // Audited OLD-baseline launch chains. The exemption covers ONLY the listed helper, and
    // only while every function of the chain (the loader that reads the baseline bytes, the
    // helper, and each of its callers) still has the audited token fingerprint and the
    // helper has no caller outside the audited list. Anything else is a hard failure and
    // the helper is then checked like any other launch.
    let mut audited: HashSet<(usize, String)> = HashSet::new();
    for (key, au) in &policy.old_audited {
        let Some((path, hname)) = key.rsplit_once("::") else {
            policy_errors.push(format!("old_audited entry {key} is not `file::fn`"));
            continue;
        };
        let Some(&fi) = world.file_index.get(path) else {
            policy_errors.push(format!("stale old_audited entry {key}: file not scanned"));
            continue;
        };
        let file = &world.files[fi];
        let mut ok = true;
        for fname in au
            .fingerprints
            .keys()
            .chain(au.callers.iter())
            .chain(std::iter::once(&hname.to_string()))
        {
            if !au.fingerprints.contains_key(fname) {
                policy_errors.push(format!(
                    "old_audited {key}: `{fname}` has no pinned fingerprint"
                ));
                ok = false;
            }
        }
        for (fname, pinned) in &au.fingerprints {
            let defs: Vec<&FnRec> = file.fns.iter().filter(|f| &f.name == fname).collect();
            if defs.len() != 1 {
                policy_errors.push(format!(
                    "old_audited {key}: `{fname}` must be defined exactly once in {path} (found {})",
                    defs.len()
                ));
                ok = false;
                continue;
            }
            let actual = fingerprint(defs[0]);
            if &actual != pinned {
                policy_errors.push(format!(
                    "old_audited {key}: audited OLD chain function `{fname}` changed (token fingerprint {actual}, pinned {pinned}, baseline {}; audited because: {}): re-audit the OLD loader/caller chain, then update the pin",
                    au.baseline, au.reason
                ));
                ok = false;
            }
        }
        let actual_callers: BTreeSet<String> = world
            .call_srcs(fi, hname, 0)
            .iter()
            .map(|(ck, _, _)| file.fns[*ck].name.clone())
            .collect();
        if actual_callers.is_empty() || !actual_callers.iter().all(|c| au.callers.contains(c)) {
            policy_errors.push(format!(
                "old_audited {key}: callers {actual_callers:?} are not exactly within the audited callers {:?}",
                au.callers
            ));
            ok = false;
        }
        // Every occurrence of the helper's name in the file must be its definition or a direct
        // call inside an audited caller: a function-pointer alias, a macro argument or a call
        // from anywhere else could reach it without being audited.
        let mut direct_calls = 0usize;
        for func in file.fns.iter().filter(|f| au.callers.contains(&f.name)) {
            let mut occ: Vec<Loc> = Vec::new();
            occurrences(&func.body, hname, &mut Vec::new(), &mut occ);
            for o in occ {
                if let Some((last, parent)) = o.split_last() {
                    if tokens_at(&func.body, parent)
                        .map_or(false, |ts| ts.get(last + 1).map_or(false, |t| t.is_g('(')))
                    {
                        direct_calls += 1;
                    }
                }
            }
        }
        let defs = file.fns.iter().filter(|f| f.name == hname).count();
        let total = file.id_counts.get(hname).copied().unwrap_or(0);
        if total != defs + direct_calls {
            policy_errors.push(format!(
                "old_audited {key}: `{hname}` is mentioned {} time(s) beyond its definition and the direct calls of the audited callers (alias, function pointer, macro or another caller)",
                total.saturating_sub(defs + direct_calls)
            ));
            ok = false;
        }
        // Further pinned scopes: other files (or other roles in the same file) whose functions
        // create the OLD handles or call the helper.
        let mut pinned_fns: HashSet<(usize, String)> =
            au.fingerprints.keys().map(|f| (fi, f.clone())).collect();
        let mut scope_files: HashSet<usize> = HashSet::new();
        for sc in &au.scopes {
            let Some(&sfi) = world.file_index.get(&sc.file) else {
                policy_errors.push(format!(
                    "stale old_audited {key}: scope `{}` file {} not scanned",
                    sc.name, sc.file
                ));
                ok = false;
                continue;
            };
            scope_files.insert(sfi);
            let sfile = &world.files[sfi];
            for (fname, pinned) in &sc.fingerprints {
                pinned_fns.insert((sfi, fname.clone()));
                let defs: Vec<&FnRec> = sfile.fns.iter().filter(|f| &f.name == fname).collect();
                if defs.len() != 1 {
                    policy_errors.push(format!(
                        "old_audited {key}: scope `{}`: `{fname}` must be defined exactly once in {} (found {})",
                        sc.name,
                        sc.file,
                        defs.len()
                    ));
                    ok = false;
                    continue;
                }
                let actual = fingerprint(defs[0]);
                if &actual != pinned {
                    policy_errors.push(format!(
                        "old_audited {key}: scope `{}`: audited OLD function `{fname}` in {} changed (token fingerprint {actual}, pinned {pinned}, baseline {}; scope reason: {}): re-audit the source origin of its OLD handles, then update the pin",
                        sc.name, sc.file, au.baseline, sc.reason
                    ));
                    ok = false;
                }
            }
            if sfi != fi {
                // Direct calls of the helper in this file: only inside the scope's callers.
                let mut direct = 0usize;
                for func in sfile.fns.iter().filter(|f| sc.callers.contains(&f.name)) {
                    let mut occ: Vec<Loc> = Vec::new();
                    occurrences(&func.body, hname, &mut Vec::new(), &mut occ);
                    for o in occ {
                        if let Some((last, parent)) = o.split_last() {
                            if tokens_at(&func.body, parent)
                                .map_or(false, |ts| ts.get(last + 1).map_or(false, |t| t.is_g('(')))
                            {
                                direct += 1;
                            }
                        }
                    }
                }
                for c in &sc.callers {
                    if !sc.fingerprints.contains_key(c) {
                        policy_errors.push(format!(
                            "old_audited {key}: scope `{}`: caller `{c}` has no pinned fingerprint",
                            sc.name
                        ));
                        ok = false;
                    }
                }
                let total = sfile.id_counts.get(hname).copied().unwrap_or(0);
                let defs = sfile.fns.iter().filter(|f| f.name == hname).count();
                if total != defs + direct {
                    policy_errors.push(format!(
                        "old_audited {key}: scope `{}`: `{hname}` is mentioned {} time(s) in {} beyond the direct calls of the scope's audited callers {:?} (alias, function pointer, macro or another caller)",
                        sc.name,
                        total.saturating_sub(defs + direct),
                        sc.file,
                        sc.callers
                    ));
                    ok = false;
                }
            }
        }
        // No other scanned file may reach the helper at all.
        for (ofi, ofile) in world.files.iter().enumerate() {
            if ofi == fi || scope_files.contains(&ofi) {
                continue;
            }
            if ofile.id_counts.get(hname).copied().unwrap_or(0) > 0 {
                policy_errors.push(format!(
                    "old_audited {key}: `{hname}` is mentioned in {}, which no audit scope covers",
                    ofile.path
                ));
                ok = false;
            }
        }
        // Whatever creates or selects an OLD handle must be a pinned function.
        for gi in &au.guard_idents {
            for (ofi, ofile) in world.files.iter().enumerate() {
                if ofile.id_counts.get(gi).copied().unwrap_or(0) == 0 {
                    continue;
                }
                for func in &ofile.fns {
                    if &func.name == gi || pinned_fns.contains(&(ofi, func.name.clone())) {
                        continue;
                    }
                    let mut occ: Vec<Loc> = Vec::new();
                    occurrences(&func.body, gi, &mut Vec::new(), &mut occ);
                    if !occ.is_empty() {
                        policy_errors.push(format!(
                            "old_audited {key}: `{}` in {} mentions `{gi}` (an OLD-handle source) but is not a pinned function of any scope",
                            func.name, ofile.path
                        ));
                        ok = false;
                    }
                }
            }
        }
        if ok {
            audited.insert((fi, hname.to_string()));
        }
    }
    // Functions that contain a launch primitive, per file: probe helpers a caller can forward to.
    let mut local_launchers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (fi, file) in world.files.iter().enumerate() {
        for func in &file.fns {
            if has_launch_ident(&func.body)
                && !old_helper_fns.contains(&(file.path.clone(), func.name.clone()))
                && !audited.contains(&(fi, func.name.clone()))
            {
                local_launchers
                    .entry(file.path.clone())
                    .or_default()
                    .insert(func.name.clone());
            }
        }
    }
    // Functions that mention a changed kernel or changed source, and who calls whom. A
    // node is (file, function name); a call resolves to the same file's function of that
    // name when there is one (a local definition shadows everything else), otherwise to
    // every scanned function of that name. A launch helper is only relevant when a
    // mentioning function can reach it.
    type Node = (usize, String);
    let mut mention_res: HashMap<(usize, usize), Res> = HashMap::new();
    let mut callers_of: HashMap<Node, HashSet<Node>> = HashMap::new();
    let mut tainted: HashSet<Node> = HashSet::new();
    for (fi, file) in world.files.iter().enumerate() {
        if file.path == KERNELS_RS {
            continue;
        }
        for (fk, func) in file.fns.iter().enumerate() {
            let me: Node = (fi, func.name.clone());
            let mut cs = BTreeSet::new();
            call_names(&func.body, &mut cs);
            for c in cs {
                let local = file.fns.iter().any(|f| f.name == c);
                if local {
                    callers_of.entry((fi, c)).or_default().insert(me.clone());
                } else if let Some(defs) = world.fn_index.get(&c) {
                    for (fj, _) in defs {
                        callers_of
                            .entry((*fj, c.clone()))
                            .or_default()
                            .insert(me.clone());
                    }
                }
            }
            if !quick_relevant(&func.body, &names, &file.consts, &kernel_consts) {
                continue;
            }
            let idx = BindIndex::build(&func.body);
            let ctx = Ctx {
                file: &file.path,
                body: &func.body,
                params: fn_params(func),
                idx: &idx,
            };
            let m = world.resolve(&func.body, &ctx, &[usize::MAX], 0, false);
            let mut lits: Vec<String> = Vec::new();
            strings_in(&func.body, &mut lits);
            if lits.iter().chain(m.lits.iter()).any(|l| names.contains(l))
                || m.consts.iter().any(|(n, f)| syms.is_changed_const(n, f))
            {
                tainted.insert(me);
            }
            // The analysis loop below needs the same resolution: do it once.
            mention_res.insert((fi, fk), m);
        }
    }
    let reaches_taint = |start: Node| -> bool {
        let mut seen: HashSet<Node> = HashSet::new();
        let mut queue: Vec<Node> = vec![start];
        while let Some(f) = queue.pop() {
            if !seen.insert(f.clone()) {
                continue;
            }
            if tainted.contains(&f) {
                return true;
            }
            if let Some(cs) = callers_of.get(&f) {
                queue.extend(cs.iter().cloned());
            }
        }
        false
    };
    for (fi, file) in world.files.iter().enumerate() {
        if file.path == KERNELS_RS {
            continue;
        }
        let is_probe = file.path.contains("/examples/")
            || file.path.contains("/tests/")
            || file.path.contains("/benches/");
        for (fk, func) in file.fns.iter().enumerate() {
            let mut rep = FnReport {
                file: file.path.clone(),
                func: func.name.clone(),
                line: func.line,
                inert: func.attrs.iter().any(|a| attr_is_cfg_any_empty(a)),
                is_test: func.attrs.iter().any(|a| attr_is_test(a)),
                ..Default::default()
            };
            if !quick_relevant(&func.body, &names, &file.consts, &kernel_consts) {
                continue;
            }
            let idx = BindIndex::build(&func.body);
            let ctx = Ctx {
                file: &file.path,
                body: &func.body,
                params: fn_params(func),
                idx: &idx,
            };
            let str_params = fn_str_params(func);
            // Mentions: direct literals plus constants (no callee expansion).
            let m = match mention_res.remove(&(fi, fk)) {
                Some(m) => m,
                None => world.resolve(&func.body, &ctx, &[usize::MAX], 0, false),
            };
            let mut lits: Vec<String> = Vec::new();
            strings_in(&func.body, &mut lits);
            for l in lits.iter().chain(m.lits.iter()) {
                if names.contains(l) {
                    rep.mentions.insert(l.clone());
                }
            }
            rep.mentions_changed_source = m.consts.iter().any(|(n, f)| syms.is_changed_const(n, f));
            let launches = has_launch_ident(&func.body);
            let relevant_mention = !rep.mentions.is_empty() || rep.mentions_changed_source;
            if !launches {
                // A probe that mentions a changed kernel without launching is only safe to
                // ignore when it does not hand the work to a launching helper of its own file.
                if is_probe && relevant_mention {
                    let mut calls = BTreeSet::new();
                    call_names(&func.body, &mut calls);
                    let empty_set = BTreeSet::new();
                    let local = local_launchers.get(&file.path).unwrap_or(&empty_set);
                    rep.forwards_local = calls.iter().any(|c| local.contains(c) && *c != func.name);
                }
                let mut escaped = false;
                if is_probe && relevant_mention {
                    // A handle derived from a changed kernel name handed to a function nobody
                    // defined here (and that is not a known no-op) could launch it unseen.
                    let mut handles: HashSet<String> = HashSet::new();
                    for (bn, bs) in &idx.map {
                        for b in bs {
                            // Only a loaded kernel FUNCTION is a launchable handle (a kernel-name
                            // string handed to a loader is not).
                            let mut ids = Vec::new();
                            idents_in(&b.init, &mut ids);
                            if !ids.iter().any(|i| {
                                matches!(i.as_str(), "get_function" | "module_get_function")
                            }) {
                                continue;
                            }
                            let r = world.resolve(&b.init, &ctx, &b.loc, 0, false);
                            if r.lits.iter().any(|l| names.contains(l)) {
                                handles.insert(bn.clone());
                            }
                        }
                    }
                    if !handles.is_empty() {
                        let mut found: Vec<(String, String)> = Vec::new();
                        handle_escapes(world, &func.body, &handles, &mut found);
                        for (h, c) in found {
                            escaped = true;
                            rep.errors.push(format!(
                                "the kernel handle `{h}` escapes to `{c}`, which this gate cannot see into (unresolved dispatch)"
                            ));
                        }
                    }
                }
                if (relevant_mention && (!is_probe || rep.forwards_local)) || escaped {
                    reports.push(rep);
                }
                continue;
            }
            let sites = match find_sites(&func.body) {
                Ok(s) => s,
                Err(e) => {
                    if relevant_mention {
                        rep.errors.push(e);
                        reports.push(rep);
                    }
                    continue;
                }
            };
            let mut any_site = false;
            for (si, site) in sites.iter().enumerate() {
                // A launch helper that takes the kernel name as a parameter is checked at
                // every call, pairing that call's kernel with that call's blob.
                // A helper whose whole OLD loader/caller chain is pinned by fingerprints.
                if (site.kind == 'b' || site.kind == 'k')
                    && audited.contains(&(fi, func.name.clone()))
                {
                    continue;
                }
                if (site.kind == 'b' || site.kind == 'k') && !rep.inert {
                    if let Some(res) = world.helper_calls(fi, fk, site, &names) {
                        if !reaches_taint((fi, func.name.clone())) {
                            // No function that mentions a changed kernel (or compiles a
                            // changed source) can reach this helper by name: it cannot
                            // launch one.
                            continue;
                        }
                        match res {
                            Err(e) => {
                                any_site = true;
                                rep.errors
                                    .push(format!("site {si} (line {}): {e}", site.line));
                            }
                            Ok(calls) => {
                                for (ci, call) in calls.iter().enumerate() {
                                    any_site = true;
                                    checked_sites += 1;
                                    rep.sites.push((si, call.kernels.clone()));
                                    for k in &call.kernels {
                                        covered.insert(k.clone());
                                    }
                                    let (probs, errs, pairs) = check_group(
                                        world,
                                        &ctx,
                                        &site.loc,
                                        &call.kernels,
                                        &call.variants,
                                        syms,
                                    );
                                    checked_pairs += pairs;
                                    for p in probs {
                                        rep.problems.push(format!(
                                            "site {si} (line {}), call #{ci} (caller line {}): {p}",
                                            site.line, call.line
                                        ));
                                    }
                                    for e in errs {
                                        rep.errors.push(format!(
                                            "site {si} (line {}), call #{ci} (caller line {}): {e}",
                                            site.line, call.line
                                        ));
                                    }
                                }
                            }
                        }
                        continue;
                    }
                }
                let mut strict_decls = true;
                let mut r = world.resolve(&site.name_ex, &ctx, &site.loc, 0, true);
                if !r.lits.iter().any(|l| names.contains(l))
                    && policy
                        .checked_delegations
                        .values()
                        .any(|d| d.returns && d.chain.last() == Some(&key(&file.path, &func.name)))
                {
                    if let Some(projected) =
                        descriptor_field_names(world, &ctx, &site.name_ex, &site.loc, None, 0)
                    {
                        r.merge(&projected);
                    }
                }
                if r.unres.contains("<depth>") && !rep.inert {
                    rep.errors.push(format!(
                        "site {si} (line {}): the kernel-name resolution was truncated by its depth budget; cannot prove which kernels this launch names",
                        site.line
                    ));
                    continue;
                }
                let mut cands: Vec<String> = r
                    .lits
                    .iter()
                    .filter(|l| names.contains(*l))
                    .cloned()
                    .collect();
                if let Some(ex) = extra.get(&func.name) {
                    for e in ex {
                        if !cands.contains(e) {
                            cands.push(e.clone());
                        }
                    }
                }
                if (site.kind == 'm' || site.kind == 'M')
                    && r.unres.iter().any(|u| {
                        u.strip_prefix("param:")
                            .map_or(false, |x| str_params.contains(x))
                    })
                {
                    param_helpers.insert(func.name.clone());
                    helper_defs.push((file.path.clone(), func.name.clone()));
                }
                if (site.kind == 'b' || site.kind == 'k') && cands.is_empty() {
                    // Hand-handle launches carry no name: use the symbols the probe mentions,
                    // or the inventory's explicit image list.
                    cands = rep.mentions.iter().cloned().collect();
                    strict_decls = false;
                    if cands.is_empty() {
                        let pk = key(&file.path, &func.name);
                        if let Some(images) = policy.probe_candidates.get(&pk) {
                            for img in images {
                                if let Some(v) = syms.origin_symbols.get(&format!("image:{img}")) {
                                    for s in v {
                                        if !cands.contains(s) {
                                            cands.push(s.clone());
                                        }
                                    }
                                }
                            }
                            probe_used.insert(pk);
                        }
                    }
                }
                if cands.is_empty() {
                    continue;
                }
                any_site = true;
                rep.sites.push((si, cands.clone()));
                if rep.inert {
                    continue;
                }
                let probe_site = site.kind == 'b' || site.kind == 'k';
                if site.kind != 'm' && site.kind != 'M' && !probe_site {
                    rep.errors.push(format!(
                        "site {si} (line {}): `{}` launches {:?} but this primitive cannot be replayed",
                        site.line, site.method, cands
                    ));
                    continue;
                }
                let variants_res = if let Some(r) = contract_site_variants(site, &cands) {
                    r
                } else if probe_site {
                    world.probe_variants(fi, fk, site, &names)
                } else {
                    world.site_layouts(&ctx, site)
                };
                let variants = match variants_res {
                    Ok(v) => v,
                    Err(e) => {
                        rep.errors
                            .push(format!("site {si} (line {}): {e}", site.line));
                        continue;
                    }
                };
                // Deliberately historical (OLD ABI) launch variants are exempt, one entry each.
                let mut kept: Vec<Variant> = Vec::new();
                for v in variants {
                    let mut exempt = false;
                    for (a, bv) in &v.assign {
                        let k = format!("{}::{}#{}={}", file.path, func.name, a, bv);
                        if policy.old_variants.contains_key(&k) {
                            old_used.insert(k);
                            exempt = true;
                        }
                    }
                    if !exempt {
                        kept.push(v);
                    }
                }
                let variants = kept;
                checked_sites += 1;
                let prefer: BTreeSet<String> = r
                    .consts
                    .iter()
                    .chain(m.consts.iter())
                    .map(|(n, f)| const_origin(n, f))
                    .collect();
                let mut cand_decls: Vec<(String, String, Decl)> = Vec::new();
                for c in &cands {
                    let all = syms.decls(c);
                    let narrowed: Vec<(String, Decl)> = all
                        .iter()
                        .filter(|(o, _)| prefer.contains(o))
                        .cloned()
                        .collect();
                    let use_list = if narrowed.is_empty() { all } else { narrowed };
                    for (o, d) in use_list {
                        cand_decls.push((c.clone(), o, d));
                    }
                    covered.insert(c.clone());
                }
                let layouts_differ = cand_decls
                    .iter()
                    .any(|(_, _, d)| !crate::hip::same_layout(&cand_decls[0].2, d));
                let assign_text = |v: &Variant| -> String {
                    v.assign
                        .iter()
                        .map(|(k, b)| format!("{k}={b}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                if !strict_decls {
                    // The launch names no kernel: a layout variant cannot be tied to one
                    // candidate, so the candidates must be one identical ABI and every
                    // variant must match all of them.
                    if layouts_differ {
                        rep.errors.push(format!(
                            "site {si} (line {}): the launch carries no kernel name and its candidates {:?} declare different ABIs; a layout variant cannot be tied to one kernel",
                            site.line, cands
                        ));
                        continue;
                    }
                    for (c, o, d) in &cand_decls {
                        checked_pairs += 1;
                        for v in &variants {
                            let p = compare(world, &ctx, v, &site.loc, d);
                            if !p.is_empty() {
                                rep.problems.push(format!(
                                    "site {si} (line {}): `{c}` [{o}] variant [{}]: {}",
                                    site.line,
                                    assign_text(v),
                                    p.join("; ")
                                ));
                            }
                        }
                    }
                    continue;
                }
                if layouts_differ && variants.len() > 1 {
                    // Different kernels with different ABIs behind branching layouts:
                    // each branch must be checked against the kernel it actually selects.
                    // Layouts the inventory pinned to kernels carry that selection; the
                    // rest are tied to kernels through the name expression's branches.
                    let all_pinned = variants.iter().all(|v| v.only_for.is_some());
                    let corr: Vec<(String, Conds)> = if all_pinned {
                        Vec::new()
                    } else {
                        match world
                            .name_conds(&ctx, site, &names)
                            .filter(|c| cands.iter().all(|k| c.iter().any(|(l, _)| l == k)))
                        {
                            Some(c) => c,
                            None => {
                                rep.errors.push(format!(
                                    "site {si} (line {}): layout variants differ and candidates {:?} declare different ABIs, but the kernel selection cannot be correlated with the layout branches",
                                    site.line, cands
                                ));
                                continue;
                            }
                        }
                    };
                    let allowed_of = |v: &Variant| -> BTreeSet<String> {
                        match &v.only_for {
                            Some(of) => cands.iter().filter(|c| of.contains(c)).cloned().collect(),
                            None => corr
                                .iter()
                                .filter(|(_, cd)| compatible(&v.assign, cd))
                                .map(|(l, _)| l.clone())
                                .collect(),
                        }
                    };
                    for v in &variants {
                        let allowed = allowed_of(v);
                        if allowed.is_empty() {
                            rep.errors.push(format!(
                                "site {si} (line {}): layout variant [{}] is selected by no kernel name",
                                site.line,
                                assign_text(v)
                            ));
                            continue;
                        }
                        for (c, o, d) in cand_decls.iter().filter(|(c, _, _)| allowed.contains(c)) {
                            checked_pairs += 1;
                            let p = compare(world, &ctx, v, &site.loc, d);
                            if !p.is_empty() {
                                rep.problems.push(format!(
                                    "site {si} (line {}): variant [{}] launches `{c}` [{o}]: {}",
                                    site.line,
                                    assign_text(v),
                                    p.join("; ")
                                ));
                            }
                        }
                    }
                    for (c, o, d) in &cand_decls {
                        let ok = variants.iter().any(|v| {
                            allowed_of(v).contains(c)
                                && compare(world, &ctx, v, &site.loc, d).is_empty()
                        });
                        if !ok {
                            rep.problems.push(format!(
                                "site {si} (line {}): no layout variant that selects `{c}` matches its declaration [{o}]",
                                site.line
                            ));
                        }
                    }
                    continue;
                }
                // Every candidate declaration must be produced by some variant.
                for (c, o, d) in &cand_decls {
                    checked_pairs += 1;
                    let mut best: Option<Vec<String>> = None;
                    for v in &variants {
                        let p = compare(world, &ctx, v, &site.loc, d);
                        if p.is_empty() {
                            best = Some(Vec::new());
                            break;
                        }
                        if best.as_ref().map_or(true, |b| p.len() < b.len()) {
                            best = Some(p);
                        }
                    }
                    if let Some(p) = best {
                        if !p.is_empty() {
                            rep.problems.push(format!(
                                "site {si} (line {}): `{c}` [{o}]: {}",
                                site.line,
                                p.join("; ")
                            ));
                        }
                    }
                }
                // Every layout the launcher can build must match some candidate.
                for v in &variants {
                    let ok = cand_decls
                        .iter()
                        .any(|(_, _, d)| compare(world, &ctx, v, &site.loc, d).is_empty());
                    if !ok {
                        rep.problems.push(format!(
                            "site {si} (line {}): launch variant [{}] matches none of {:?}",
                            site.line,
                            assign_text(v),
                            cands
                        ));
                    }
                }
            }
            if any_site || relevant_mention {
                reports.push(rep);
            }
        }
    }
    // A probe that only forwards to launching helpers of its own file is cleared when the
    // ACTUAL callee launch sites are all checked and together cover every kernel the
    // caller mentions; a helper with no checked site (or fewer kernels) leaves it open.
    let mut covered_by: HashMap<(String, String), BTreeSet<String>> = HashMap::new();
    for r in &reports {
        if !r.sites.is_empty() {
            let e = covered_by
                .entry((r.file.clone(), r.func.clone()))
                .or_default();
            for (_, c) in &r.sites {
                e.extend(c.iter().cloned());
            }
        }
    }
    for rep in reports.iter_mut() {
        if !(rep.forwards_local && rep.sites.is_empty()) {
            continue;
        }
        let Some(file) = world.file(&rep.file) else {
            continue;
        };
        let Some(func) = file
            .fns
            .iter()
            .find(|f| f.name == rep.func && f.line == rep.line)
        else {
            continue;
        };
        let mut calls = BTreeSet::new();
        call_names(&func.body, &mut calls);
        let empty_set = BTreeSet::new();
        let local = local_launchers.get(&rep.file).unwrap_or(&empty_set);
        let helpers: Vec<&String> = calls
            .iter()
            .filter(|c| local.contains(*c) && **c != rep.func)
            .collect();
        let mut cover: BTreeSet<String> = BTreeSet::new();
        let mut all_checked = !helpers.is_empty();
        for h in helpers {
            match covered_by.get(&(rep.file.clone(), h.clone())) {
                Some(c) => cover.extend(c.iter().cloned()),
                None => all_checked = false,
            }
        }
        // Every kernel the function names must be launched by a checked helper call. (The other
        // kernels of a source it compiles cannot be launched without being named.)
        if all_checked && rep.mentions.is_subset(&cover) {
            rep.cleared = true;
        }
    }

    // Unlike parameter-name forwarders, these rows add no candidates or coverage:
    // the real terminal launch must already check the complete origin symbol set.
    for (origin, delegation) in &policy.checked_delegations {
        let checked = (|| -> Result<usize, String> {
            if delegation.chain.len() < 2 || delegation.chain.first() != Some(origin) {
                return Err(
                    "chain must start at its inventory key and contain a terminal launcher".into(),
                );
            }
            let mut functions = Vec::new();
            let mut distinct = BTreeSet::new();
            for entry in &delegation.chain {
                if !distinct.insert(entry) {
                    return Err(format!("repeated chain function {entry}"));
                }
                let (path, name) = entry
                    .rsplit_once("::")
                    .ok_or_else(|| format!("invalid function key {entry}"))?;
                let file = world
                    .file(path)
                    .ok_or_else(|| format!("unscanned file {path}"))?;
                let defs: Vec<_> = file.fns.iter().filter(|f| f.name == name).collect();
                if defs.len() != 1 {
                    return Err(format!("{entry}: expected exactly one definition"));
                }
                if defs[0]
                    .attrs
                    .iter()
                    .any(|a| attr_is_cfg_any_empty(a) || attr_is_test(a))
                {
                    return Err(format!(
                        "{entry}: chain function is not live production code"
                    ));
                }
                functions.push(defs[0]);
            }
            for pair in functions.windows(2) {
                let (caller, callee) = if delegation.returns {
                    (pair[1], pair[0])
                } else {
                    (pair[0], pair[1])
                };
                if world
                    .files
                    .iter()
                    .flat_map(|f| &f.fns)
                    .filter(|f| {
                        f.name == callee.name
                            && !f
                                .attrs
                                .iter()
                                .any(|a| attr_is_cfg_any_empty(a) || attr_is_test(a))
                    })
                    .count()
                    != 1
                {
                    return Err(format!("{}: ambiguous call target", callee.name));
                }
                let mut calls = BTreeSet::new();
                call_names(&caller.body, &mut calls);
                if !calls.contains(&callee.name) {
                    return Err(format!("{} no longer calls {}", caller.name, callee.name));
                }
            }
            let find_report = |entry: &str| {
                reports
                    .iter()
                    .position(|r| key(&r.file, &r.func) == entry && !r.inert && !r.is_test)
            };
            let start = find_report(origin).ok_or("origin has no live changed-kernel report")?;
            let end = find_report(delegation.chain.last().unwrap())
                .ok_or("terminal has no live checked report")?;
            let terminal = &reports[end];
            let symbols: BTreeSet<_> = terminal
                .sites
                .iter()
                .flat_map(|(_, c)| c.iter().cloned())
                .collect();
            if terminal.sites.is_empty()
                || !terminal.problems.is_empty()
                || !terminal.errors.is_empty()
            {
                return Err("terminal launch has no successful ABI check".into());
            }
            if reports[start].mentions.is_empty() || !reports[start].mentions.is_subset(&symbols) {
                return Err(format!(
                    "terminal does not check every origin symbol {:?}",
                    reports[start].mentions
                ));
            }
            Ok(start)
        })();
        match checked {
            Ok(start) => reports[start].cleared = true,
            Err(e) => policy_errors.push(format!("checked forwarder {origin}: {e}")),
        }
    }

    // A launch helper that takes the kernel name as a parameter, called from a
    // function that mentions changed kernels, is not resolvable here: fail closed.
    let helper_names: BTreeSet<String> = param_helpers
        .into_iter()
        .filter(|n| !PRIMITIVES.contains(&n.as_str()))
        .collect();
    if !helper_names.is_empty() {
        let names_set: HashSet<&String> = helper_names.iter().collect();
        for rep in reports.iter_mut() {
            if rep.is_test {
                continue;
            }
            let file = world.file(&rep.file).unwrap();
            if let Some(func) = file
                .fns
                .iter()
                .find(|f| f.name == rep.func && f.line == rep.line)
            {
                let mut calls = BTreeSet::new();
                call_names(&func.body, &mut calls);
                for c in calls {
                    if names_set.contains(&c) && c != rep.func {
                        rep.forwards_to_param_helper.push(c);
                    }
                }
            }
        }
    }
    let _ = helper_defs;
    Analysis {
        reports,
        covered,
        checked_pairs,
        checked_sites,
        old_used,
        probe_used,
        policy_errors,
    }
}

/// A source-checked call chain or returned-descriptor consumer chain. The final
/// function must already have ABI-checked sites covering every origin symbol.
pub struct CheckedDelegation {
    pub chain: Vec<String>,
    pub returns: bool,
}

/// Inventory-driven policy.
#[derive(Default)]
pub struct Policy {
    /// "file::fn" -> reason, production functions that mention changed kernels but launch none.
    pub non_launching: BTreeMap<String, String>,
    /// "file::fn" -> reason, `#[cfg(any())]` functions.
    pub inert: BTreeMap<String, String>,
    /// "file::fn" -> (launcher fn name, reason): functions that hand changed kernel
    /// names to a launcher through a `&str` parameter. The launcher's candidate set
    /// is extended with the symbols the forwarder mentions.
    pub forwarders: BTreeMap<String, (String, String)>,
    /// Typed `forwarders` rows for wrappers and descriptor-returning selectors.
    pub checked_delegations: BTreeMap<String, CheckedDelegation>,
    /// "file::fn#condition=bool" -> reason: launch variants that deliberately build the
    /// OLD (pre-guard) kernarg ABI. Exempt one by one; every other variant is checked.
    pub old_variants: BTreeMap<String, String>,
    /// "file::fn" -> image paths: hand-handle probes whose launch carries no symbol name
    /// and whose source does not mention one.
    pub probe_candidates: BTreeMap<String, Vec<String>>,
    /// "file::fn" -> reason: probe launch helpers that only launch OLD fixture kernels
    /// (their kernel name carries `OLD_SUFFIX`). Verified from the helper's launch sites.
    pub old_helpers: BTreeMap<String, String>,
    /// "file::helper_fn" -> an audited OLD-baseline launch chain (see `OldAudit`).
    pub old_audited: BTreeMap<String, OldAudit>,
}

pub fn key(file: &str, func: &str) -> String {
    format!("{file}::{func}")
}

/// Violations of the closed-coverage rules (unclassified mentions, errors, ABI problems).
pub fn violations(a: &Analysis, p: &Policy) -> (Vec<String>, Vec<String>) {
    let mut abi: Vec<String> = Vec::new();
    let mut coverage: Vec<String> = Vec::new();
    let mut used_nl: BTreeSet<String> = BTreeSet::new();
    let mut used_inert: BTreeSet<String> = BTreeSet::new();
    let mut used_fw: BTreeSet<String> = BTreeSet::new();
    for r in &a.reports {
        let k = key(&r.file, &r.func);
        for pr in &r.problems {
            abi.push(format!("{}:{} {}: {pr}", r.file, r.line, r.func));
        }
        for e in &r.errors {
            coverage.push(format!("{}:{} {}: UNSUPPORTED {e}", r.file, r.line, r.func));
        }
        for h in &r.forwards_to_param_helper {
            if p.forwarders.get(&k).map_or(false, |(to, _)| to == h) {
                used_fw.insert(k.clone());
            } else {
                coverage.push(format!(
                    "{}:{} {}: forwards a changed-kernel name to parameter-named launch helper `{h}` without a forwarders entry",
                    r.file, r.line, r.func
                ));
            }
        }
        if r.cleared {
            continue;
        }
        if r.inert {
            if p.inert.contains_key(&k) {
                used_inert.insert(k);
            } else {
                coverage.push(format!("{}:{} {}: `#[cfg(any())]` function touches changed kernels but is not listed as inert", r.file, r.line, r.func));
            }
            continue;
        }
        let launches = !r.sites.is_empty();
        if !launches {
            if r.is_test && !r.forwards_local {
                continue;
            }
            if p.forwarders.contains_key(&k) {
                // Classified by its forwarder entry (checked below).
            } else if p.non_launching.contains_key(&k) {
                used_nl.insert(k);
            } else {
                coverage.push(format!(
                    "{}:{} {}: mentions changed kernels {:?} but has no checked launch site and is not listed as non-launching",
                    r.file, r.line, r.func, r.mentions
                ));
            }
        } else if p.non_launching.contains_key(&k) {
            coverage.push(format!(
                "{}:{} {}: listed as non-launching but launches {:?}",
                r.file, r.line, r.func, r.sites
            ));
            used_nl.insert(k);
        }
    }
    for k in p.non_launching.keys() {
        if !used_nl.contains(k) {
            coverage.push(format!(
                "stale non_launching entry {k}: the function no longer mentions changed kernels"
            ));
        }
    }
    for k in p.forwarders.keys() {
        if !used_fw.contains(k) {
            coverage.push(format!(
                "stale forwarder entry {k}: the function no longer forwards names to its launcher"
            ));
        }
    }
    for k in p.inert.keys() {
        if !used_inert.contains(k) {
            coverage.push(format!(
                "stale inert entry {k}: no such cfg(any()) function touches changed kernels"
            ));
        }
    }
    for k in p.old_variants.keys() {
        if !a.old_used.contains(k) {
            coverage.push(format!("stale old_variants entry {k}: no launch variant of that call carries this condition"));
        }
    }
    for k in p.probe_candidates.keys() {
        if !a.probe_used.contains(k) {
            coverage.push(format!("stale probe_candidates entry {k}: the probe now resolves its own symbols or no longer launches"));
        }
    }
    coverage.extend(a.policy_errors.iter().cloned());
    (abi, coverage)
}

/// An audited chain of OLD-baseline launches (`file::helper` in `Policy::old_audited`): the
/// functions that read the baseline bytes, build the handles and call the helper were read
/// and found to launch only pre-guard kernels. Each is pinned by the token fingerprint of its
/// signature and body (whitespace and comments do not matter); any change fails the gate.
#[derive(Clone, Debug)]
pub struct OldAudit {
    pub reason: String,
    /// Baseline commit the OLD kernels come from.
    pub baseline: String,
    /// The only functions allowed to call the helper.
    pub callers: Vec<String>,
    /// Function name -> fingerprint, for the helper, its callers and the loader chain.
    pub fingerprints: BTreeMap<String, String>,
    /// Identifiers that create or select an OLD handle (`load_old_image`, `K1Path::OldImage`):
    /// any function of any scanned file that mentions one must be pinned by this audit.
    pub guard_idents: Vec<String>,
    /// Further pinned scopes (other files, or other provenance such as a separately
    /// loaded historical image): each proves where ITS callers get their OLD handles.
    pub scopes: Vec<OldScope>,
}

/// One provenance scope of an `OldAudit`.
#[derive(Clone, Debug)]
pub struct OldScope {
    pub name: String,
    pub reason: String,
    pub file: String,
    /// Functions of `file` that call the helper directly (none when the file only pins
    /// handle-creating functions that reach the helper through the main audit).
    pub callers: Vec<String>,
    pub fingerprints: BTreeMap<String, String>,
}

/// FNV-1a over the normalised tokens of a function's signature and body.
pub fn fingerprint(func: &FnRec) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text(&func.sig)
        .bytes()
        .chain(std::iter::once(1u8))
        .chain(text(&func.body).bytes())
    {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Calls `c(.. handle ..)` / `x.c(.. handle ..)` where `c` is defined nowhere in the scanned
/// sources and is not a known no-op: (handle, callee).
fn handle_escapes(
    world: &World,
    ts: &[Tt],
    handles: &HashSet<String>,
    out: &mut Vec<(String, String)>,
) {
    const NO_OP: [&str; 24] = [
        "drop",
        "Some",
        "Ok",
        "Err",
        "Box",
        "Vec",
        "String",
        "format",
        "println",
        "eprintln",
        "assert",
        "assert_eq",
        "assert_ne",
        "debug_assert",
        "panic",
        "unwrap",
        "expect",
        "clone",
        "to_string",
        "len",
        "is_empty",
        "map_err",
        "ok_or",
        "as_ref",
    ];
    for k in 0..ts.len() {
        if let Tt::Id(c, _) = &ts[k] {
            if let Some(g) = ts
                .get(k + 1)
                .filter(|t| t.is_g('('))
                .and_then(|t| t.group())
            {
                if !world.fn_index.contains_key(c) && !NO_OP.contains(&c.as_str()) {
                    let mut ids = Vec::new();
                    idents_in(g, &mut ids);
                    if let Some(h) = ids.iter().find(|i| handles.contains(*i)) {
                        out.push((h.clone(), c.clone()));
                    }
                }
            }
        }
        if let Tt::G(_, c, _) = &ts[k] {
            handle_escapes(world, c, handles, out);
        }
    }
}

#[cfg(test)]
mod aggregate_blob_tests {
    use super::*;
    use crate::hip::{Arg, Field};

    /// Replay `(size, kind)` pushes through the production blob exactly as `blob_layout` does.
    fn replay(pushes: &[(u32, char)]) -> (Vec<BlobArg>, u32) {
        let mut blob = hip_bridge::KernargBlob::new();
        let mut out = Vec::new();
        for &(size, kind) in pushes {
            let offset = (blob.len() as u32 + size - 1) / size * size;
            match (size, kind) {
                (8, 'p') => blob.push_ptr(std::ptr::null()),
                (8, _) => blob.push_u64(0),
                (4, 'f') => blob.push_f32(0.0),
                (4, _) => blob.push_i32(0),
                other => panic!("unsupported test push {other:?}"),
            }
            out.push(BlobArg {
                offset,
                size,
                kind,
                expr: Vec::new(),
            });
        }
        (out, blob.len() as u32)
    }

    fn arg(offset: u32, size: u32, kind: char) -> Arg {
        Arg {
            offset,
            size,
            kind,
            text: String::new(),
            fields: Vec::new(),
        }
    }

    /// By-value aggregate: `(offset, element size, kind, element count)` per source field.
    fn agg(offset: u32, size: u32, fields: &[(u32, u32, char, u32)]) -> Arg {
        Arg {
            offset,
            size,
            kind: 'v',
            text: String::new(),
            fields: fields
                .iter()
                .map(|&(offset, size, kind, count)| Field {
                    offset,
                    size,
                    kind,
                    count,
                })
                .collect(),
        }
    }

    fn run(args: Vec<Arg>, end: u32, pushes: &[(u32, char)]) -> Vec<String> {
        let (lay, blob_end) = replay(pushes);
        compare_blob(&lay, blob_end, &Decl { args, end })
    }

    fn has(probs: &[String], needle: &str) -> bool {
        probs.iter().any(|p| p.contains(needle))
    }

    /// ptr, i32, Yarn (4 x f32, align 4, 16 bytes), lane table (256 x i32, 1024 bytes).
    fn qsa_args() -> (Vec<Arg>, u32) {
        (
            vec![
                arg(0, 8, 'p'),
                arg(8, 4, 'i'),
                agg(12, 16, &[(12, 4, 'f', 4)]),
                agg(28, 1024, &[(28, 4, 'i', 256)]),
            ],
            1052,
        )
    }

    fn qsa_pushes() -> Vec<(u32, char)> {
        let mut p = vec![(8, 'p'), (4, 'i')];
        p.extend([(4, 'f'); 4]);
        p.extend([(4, 'i'); 256]);
        p
    }

    #[test]
    fn yarn_then_lane_table_pushed_per_leaf_match() {
        let (args, end) = qsa_args();
        assert_eq!(run(args, end, &qsa_pushes()), Vec::<String>::new());
    }

    #[test]
    fn natural_alignment_padding_between_leaves() {
        // struct { int a; long long b; int c; } -> b at 8, c at 16, size 24; the blob stops at
        // the last leaf (end 20), inside the aggregate's tail padding.
        let args = vec![agg(
            0,
            24,
            &[(0, 4, 'i', 1), (8, 8, 'i', 1), (16, 4, 'i', 1)],
        )];
        let p = run(args.clone(), 24, &[(4, 'i'), (8, 'i'), (4, 'i')]);
        assert_eq!(p, Vec::<String>::new());
        // After an 8-byte argument the aggregate starts on its own 8-byte boundary.
        let mut shifted = vec![arg(0, 8, 'p')];
        shifted.push(agg(
            8,
            24,
            &[(8, 4, 'i', 1), (16, 8, 'i', 1), (24, 4, 'i', 1)],
        ));
        let p = run(
            shifted.clone(),
            32,
            &[(8, 'p'), (4, 'i'), (8, 'i'), (4, 'i')],
        );
        assert_eq!(p, Vec::<String>::new());
        // Behind a 4-byte argument the same declaration is out of reach of per-leaf pushes.
        shifted[0] = arg(0, 4, 'i');
        let p = run(shifted, 32, &[(4, 'i'), (4, 'i'), (8, 'i'), (4, 'i')]);
        assert!(has(
            &p,
            "arg 1 field 0 element 0: blob (offset 4, 4 bytes) vs declared (offset 8"
        ));
    }

    #[test]
    fn array_field_then_wider_scalar_with_tail() {
        // struct { float a[3]; long long b; } -> b at 16, size 24.
        let args = vec![agg(0, 24, &[(0, 4, 'f', 3), (16, 8, 'i', 1)])];
        let p = run(args, 24, &[(4, 'f'), (4, 'f'), (4, 'f'), (8, 'i')]);
        assert_eq!(p, Vec::<String>::new());
    }

    #[test]
    fn tail_padding_only_accepted_for_the_trailing_aggregate() {
        // The aggregate is not last: the next argument must land after the full padded size.
        let args = vec![
            agg(0, 16, &[(0, 8, 'i', 1), (8, 4, 'i', 1)]),
            arg(16, 4, 'i'),
        ];
        assert!(has(
            &run(args.clone(), 20, &[(8, 'i'), (4, 'i'), (4, 'i')]),
            "arg 1: blob (offset 12, 4 bytes) vs declared (offset 16, 4 bytes)"
        ));
        // Blob stopping short of the last leaf is not tail padding.
        let args = vec![agg(0, 16, &[(0, 8, 'i', 1), (8, 4, 'i', 1)])];
        assert!(!run(args, 16, &[(8, 'i')]).is_empty());
    }

    #[test]
    fn wrong_leaf_class_is_refused() {
        let (args, end) = qsa_args();
        let mut p = qsa_pushes();
        p[3] = (4, 'i'); // a yarn float pushed as an integer
        assert!(has(
            &run(args.clone(), end, &p),
            "aggregate leaf class mismatch"
        ));
        let mut p = qsa_pushes();
        p[10] = (4, 'f'); // a lane integer pushed as a float
        assert!(has(&run(args, end, &p), "aggregate leaf class mismatch"));
    }

    #[test]
    fn wrong_order_is_refused() {
        let (args, end) = qsa_args();
        let mut p = vec![(8, 'p'), (4, 'i')];
        p.extend([(4, 'i'); 256]);
        p.extend([(4, 'f'); 4]);
        assert!(!run(args, end, &p).is_empty());
    }

    #[test]
    fn wrong_leaf_count_is_refused() {
        let (args, end) = qsa_args();
        for drop_idx in [3usize, 100] {
            let mut p = qsa_pushes();
            p.remove(drop_idx);
            assert!(has(
                &run(args.clone(), end, &p),
                "raw blob has 261 arguments"
            ));
        }
        let mut p = qsa_pushes();
        p.push((4, 'i'));
        assert!(has(
            &run(args.clone(), end, &p),
            "raw blob has 263 arguments"
        ));
        let mut p = qsa_pushes();
        p.insert(2, (4, 'f')); // extraneous leaf in front of the yarn
        assert!(has(&run(args, end, &p), "raw blob has 263 arguments"));
    }

    #[test]
    fn pointer_or_wide_word_in_leaf_slot_is_refused() {
        let (args, end) = qsa_args();
        let mut p = qsa_pushes();
        p[2] = (8, 'p');
        assert!(!run(args.clone(), end, &p).is_empty());
        let mut p = qsa_pushes();
        p[2] = (8, 'i');
        assert!(!run(args.clone(), end, &p).is_empty());
        // A pointer-class leaf must be fed a pointer, not an integer word.
        let ptr_leaf = vec![agg(0, 8, &[(0, 8, 'p', 1)])];
        assert!(has(
            &run(ptr_leaf.clone(), 8, &[(8, 'i')]),
            "aggregate leaf class mismatch"
        ));
        assert_eq!(run(ptr_leaf, 8, &[(8, 'p')]), Vec::<String>::new());
    }

    #[test]
    fn shifted_declared_offset_is_refused() {
        let (mut args, _) = qsa_args();
        args[3] = agg(32, 1024, &[(32, 4, 'i', 256)]);
        let p = run(args, 1056, &qsa_pushes());
        assert!(has(
            &p,
            "arg 3 field 0 element 0: blob (offset 28, 4 bytes) vs declared (offset 32"
        ));
    }

    #[test]
    fn many_mismatches_are_capped() {
        let (args, end) = qsa_args();
        let mut p = vec![(8, 'p'), (4, 'i')];
        p.extend([(4, 'f'); 4]);
        p.extend([(4, 'f'); 256]);
        let probs = run(args, end, &p);
        assert!(probs.len() <= 10, "{} problems", probs.len());
        assert!(has(&probs, "further aggregate leaf mismatches"));
    }

    #[test]
    fn metadata_byvalue_aggregate_without_fields_gets_no_fragment_acceptance() {
        let (args, end) = (vec![arg(0, 8, 'p'), arg(8, 16, 'v')], 24);
        // Four f32 fragments of a 16-byte PM by-value argument.
        let mut frag = vec![(8, 'p')];
        frag.extend([(4, 'f'); 4]);
        assert!(has(
            &run(args.clone(), end, &frag),
            "raw blob has 5 arguments, declared 2"
        ));
        // Two u64 fragments.
        assert!(has(
            &run(args.clone(), end, &[(8, 'p'), (8, 'i'), (8, 'i')]),
            "raw blob has 3 arguments"
        ));
        // A single push of the wrong width.
        assert!(has(
            &run(args, end, &[(8, 'p'), (8, 'i')]),
            "vs declared (offset 8, 16 bytes)"
        ));
        // 1024-byte opaque by-value argument fed 256 integer fragments.
        let big = vec![arg(0, 1024, 'v')];
        assert!(has(
            &run(big, 1024, &[(4, 'i'); 256]),
            "raw blob has 256 arguments, declared 1"
        ));
        // Scalar by-value argument keeps its existing one-push handling.
        assert_eq!(
            run(vec![arg(0, 4, 'v')], 4, &[(4, 'i')]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn malformed_aggregate_declarations_are_refused() {
        let m = |a: Arg, end: u32| run(vec![a], end, &[(4, 'i')]);
        // Leaf of unknown class, zero count, zero size.
        assert!(has(
            &m(agg(0, 4, &[(0, 4, 'v', 1)]), 4),
            "malformed aggregate"
        ));
        assert!(has(
            &m(agg(0, 4, &[(0, 4, 'i', 0)]), 4),
            "malformed aggregate"
        ));
        assert!(has(
            &m(agg(0, 4, &[(0, 0, 'i', 1)]), 4),
            "malformed aggregate"
        ));
        // First field not at the aggregate start; overlap; hole; past end; short tail.
        assert!(has(&m(agg(0, 8, &[(4, 4, 'i', 1)]), 8), "first field"));
        assert!(has(
            &m(agg(0, 8, &[(0, 4, 'i', 2), (4, 4, 'i', 1)]), 8),
            "overlaps"
        ));
        assert!(has(
            &m(agg(0, 24, &[(0, 4, 'i', 1), (16, 4, 'i', 1)]), 24),
            "-byte hole"
        ));
        assert!(has(
            &m(agg(0, 4, &[(0, 4, 'i', 2)]), 4),
            "past aggregate end"
        ));
        assert!(has(
            &m(agg(0, 16, &[(0, 4, 'i', 1)]), 16),
            "short of aggregate end"
        ));
        // Fields on a non-aggregate kind.
        let mut bad = agg(0, 4, &[(0, 4, 'i', 1)]);
        bad.kind = 'i';
        assert!(has(&m(bad, 4), "not `v`"));
    }

    #[test]
    fn huge_array_count_is_never_materialised() {
        let args = vec![agg(0, 1 << 30, &[(0, 4, 'i', 1 << 28)])];
        let p = run(args, 1 << 30, &[(4, 'i'); 3]);
        assert!(has(&p, "raw blob has 3 arguments, declared 268435456"));
    }

    fn bytes(args: Vec<Arg>, end: u32, len: u32, writes: &[(u32, u32)]) -> Vec<String> {
        compare_bytes(
            &ByteLayout {
                len,
                writes: writes.to_vec(),
            },
            &Decl { args, end },
        )
    }

    fn yarn_decl() -> Vec<Arg> {
        vec![arg(0, 8, 'p'), agg(8, 16, &[(8, 4, 'f', 4)])]
    }

    #[test]
    fn byte_array_per_leaf_writes_match() {
        let w = [(0, 8), (8, 4), (12, 4), (16, 4), (20, 4)];
        assert_eq!(bytes(yarn_decl(), 24, 24, &w), Vec::<String>::new());
        // Two array rows (128 each) of a 1024-byte lane table.
        let table = vec![agg(0, 1024, &[(0, 4, 'i', 128), (512, 4, 'i', 128)])];
        let w: Vec<(u32, u32)> = (0..256).map(|k| (k * 4, 4)).collect();
        assert_eq!(bytes(table, 1024, 1024, &w), Vec::<String>::new());
    }

    #[test]
    fn byte_array_opaque_whole_aggregate_write_is_refused() {
        let p = bytes(yarn_decl(), 24, 24, &[(0, 8), (8, 16)]);
        assert!(has(&p, "covers aggregate arg 1 as one opaque write"));
        assert!(has(
            &p,
            "aggregate arg 1 has 4 scalar leaves but only 2 distinct"
        ));
    }

    #[test]
    fn byte_array_wrong_leaf_writes_are_refused() {
        // A missing leaf.
        let p = bytes(yarn_decl(), 24, 24, &[(0, 8), (8, 4), (12, 4), (20, 4)]);
        assert!(has(
            &p,
            "arg 1 field 0 element 2 (offset 16, 4 bytes) is never written"
        ));
        // A 8-byte write across two leaves.
        let p = bytes(yarn_decl(), 24, 24, &[(0, 8), (8, 8), (16, 4), (20, 4)]);
        assert!(has(
            &p,
            "write of 8 bytes at offset 8 matches no declared argument"
        ));
        // A write past the last element, and one off the element stride.
        let w = [(0, 8), (8, 4), (12, 4), (16, 4), (20, 4), (24, 4)];
        assert!(has(
            &bytes(yarn_decl(), 24, 24, &w),
            "offset 24 matches no declared"
        ));
        let w = [(0, 8), (8, 4), (12, 4), (16, 4), (22, 4)];
        let p = bytes(yarn_decl(), 24, 24, &w);
        assert!(has(&p, "offset 22 matches no declared"));
        assert!(has(&p, "element 3"));
        // Far fewer writes than leaves: a shortfall without walking every element.
        let table = vec![agg(0, 1024, &[(0, 4, 'i', 128), (512, 4, 'i', 128)])];
        let p = bytes(table, 1024, 1024, &[(0, 4), (4, 4), (8, 4)]);
        assert!(has(
            &p,
            "has 256 scalar leaves but only 3 distinct kernarg writes"
        ));
    }

    #[test]
    fn byte_array_metadata_byvalue_keeps_whole_argument_matching_only() {
        let a = vec![arg(0, 16, 'v')];
        assert_eq!(bytes(a.clone(), 16, 16, &[(0, 16)]), Vec::<String>::new());
        let frag = [(0, 4), (4, 4), (8, 4), (12, 4)];
        assert!(has(
            &bytes(a, 16, 16, &frag),
            "matches no declared argument"
        ));
    }

    #[test]
    fn byte_array_malformed_aggregate_is_refused() {
        let p = bytes(vec![agg(0, 16, &[(0, 4, 'i', 1)])], 16, 16, &[(0, 4)]);
        assert!(has(&p, "malformed aggregate"));
    }

    #[test]
    fn aggregate_end_past_u32_is_refused_before_leaf_iteration() {
        // Offset + size = 2^32 + 16: refused up front on both the blob and the byte path, so
        // leaf offsets are never computed.
        let huge = || vec![agg(0xFFFF_FFF0, 32, &[(0xFFFF_FFF0, 4, 'i', 8)])];
        let p = run(huge(), u32::MAX, &[(4, 'i'); 8]);
        assert!(has(&p, "past the 32-bit kernarg offset range"), "{p:?}");
        let p = bytes(huge(), u32::MAX, u32::MAX, &[(0xFFFF_FFF0, 4)]);
        assert!(has(&p, "past the 32-bit kernarg offset range"), "{p:?}");
    }

    #[test]
    fn byte_array_bounds_and_huge_array_do_not_overflow() {
        // One byte-sized field of u32::MAX elements: declared end u32::MAX, aligned bound 2^32.
        let huge = || vec![agg(0, u32::MAX, &[(0, 1, 'i', u32::MAX)])];
        let p = bytes(huge(), u32::MAX, u32::MAX, &[(0, 1)]);
        assert!(!has(&p, "kernarg byte array is"), "{p:?}");
        assert!(
            has(&p, "has 4294967295 scalar leaves but only 1 distinct"),
            "{p:?}"
        );
        // A byte array far past the declared end is still refused, with no wraparound.
        let p = bytes(vec![arg(0, 8, 'p')], 8, u32::MAX, &[(0, 8)]);
        assert!(has(&p, "kernarg byte array is 4294967295 bytes"), "{p:?}");
        // The raw blob path sees the same declaration without allocating elements.
        let p = run(huge(), u32::MAX, &[(4, 'i'); 3]);
        assert!(
            has(&p, "raw blob has 3 arguments, declared 4294967295"),
            "{p:?}"
        );
    }

    #[test]
    fn params_launch_of_aggregate_is_refused() {
        let why = params_aggregate_refusal(&yarn_decl()).expect("aggregate must be refused");
        assert!(
            why.contains("[1]") && why.contains("whole aggregate"),
            "{why}"
        );
        // Scalar, pointer and metadata by-value arguments (empty fields) are not refused.
        let plain = vec![arg(0, 8, 'p'), arg(8, 4, 'i'), arg(16, 16, 'v')];
        assert_eq!(params_aggregate_refusal(&plain), None);
    }
}
