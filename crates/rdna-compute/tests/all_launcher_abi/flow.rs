//! Blob-only launches (`launch_kernel_blob`, `launch_blob_recorded`, `launch_kernel`):
//! the layout of the EXACT operand handed to the launch, traced backwards through
//! the source from that launch site.
//!
//! * Only code that lexically precedes the launch counts (a push after the launch
//!   does not reach it).
//! * The operand is followed through aliases (`into_vec`, `clone`, `&mut`, `[..]`),
//!   block initialisers and calls of same-file (or workspace-unique) functions,
//!   binding the callee's formals to the call's actual arguments so `Option`/`bool`
//!   parameters select the real branch.
//! * A `KernargBlob` builder may only be touched by push methods; a hand-written
//!   `[0u8; N]` array only by indexed `copy_from_slice` writes (so its untouched
//!   bytes are proven zero). Any other use fails closed.
//! * Parameters are followed to every call of the function in the same file, struct
//!   fields to every initialiser/assignment of the field, collections to their
//!   literal elements, `push` calls and `map(..).collect()` closures, tuple loop
//!   variables to the picked component. Nothing in the inventory can stand in for
//!   this: a flow that cannot be followed is an error, never a guess.

use crate::layout::*;
use crate::lex::*;
use crate::scan::*;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

pub enum FlowErr {
    /// The operand cannot be followed statically (an inventory entry may close it).
    Unresolved(String),
    /// A construct that makes any answer unsound.
    Fail(String),
}

type FResult<T> = Result<T, FlowErr>;

fn fail<T>(s: impl Into<String>) -> FResult<T> {
    Err(FlowErr::Fail(s.into()))
}

fn unres<T>(s: impl Into<String>) -> FResult<T> {
    Err(FlowErr::Unresolved(s.into()))
}

#[derive(Clone, Copy)]
struct Sc<'a> {
    fi: usize,
    func: &'a FnRec,
    /// The tokens locations refer to (the fn body, or a block inside it).
    body: &'a [Tt],
}

/// Strip everything that does not change which bytes are launched.
fn peel(ex: &[Tt]) -> Vec<Tt> {
    let mut t: Vec<Tt> = ex.to_vec();
    loop {
        // An argument cut out of a call group keeps its trailing comma (`push(expr,)`).
        while t.last().map_or(false, |x| x.is_p(',')) {
            t.pop();
        }
        while t
            .first()
            .map_or(false, |x| x.is_p('&') || x.is_id("mut") || x.is_p('*'))
        {
            t.remove(0);
        }
        // Parentheses only group; a tuple keeps its group (its components are picked later).
        if t.len() == 1 && t[0].is_g('(') && split_top(t[0].group().unwrap(), ',').len() <= 1 {
            t = t[0].group().unwrap().to_vec();
            continue;
        }
        let n = t.len();
        if n >= 3
            && t[n - 1].is_g('(')
            && t[n - 1].group().unwrap().is_empty()
            && t[n - 3].is_p('.')
            && matches!(
                t[n - 2].id(),
                Some(
                    "as_mut_slice"
                        | "as_slice"
                        | "as_mut"
                        | "as_ref"
                        | "into_vec"
                        | "clone"
                        | "to_vec"
                        | "as_bytes"
                        | "iter"
                        | "iter_mut"
                        | "into_iter"
                        | "cloned"
                        | "copied"
                )
            )
        {
            t.truncate(n - 3);
            continue;
        }
        if n >= 2 && t[n - 1].is_g('[') {
            let g = t[n - 1].group().unwrap();
            // `[..]` (whole slice) and `[i]` (one element of a collection of blobs) do not
            // change which blobs may be launched; a partial range does and is not peeled.
            let full = g.len() == 2 && g[0].is_p('.') && g[1].is_p('.');
            let single = !g.is_empty() && !g.iter().any(|x| x.is_p('.'));
            if full || single {
                t.truncate(n - 1);
                continue;
            }
        }
        return t;
    }
}

fn tail_expr(g: &[Tt]) -> Vec<Tt> {
    split_stmts(g).1
}

/// The inside of `[0u8; LEN]` / `vec![0u8; LEN]`.
fn zero_array_group(init: &[Tt]) -> Option<&[Tt]> {
    let g = match init {
        [a] if a.is_g('[') => a.group()?,
        [v, b, a] if v.is_id("vec") && b.is_p('!') && a.is_g('[') => a.group()?,
        _ => return None,
    };
    (g.len() > 2
        && g[1].is_p(';')
        && matches!(g.first(), Some(Tt::N(s, _)) if s == "0u8" || s == "0"))
    .then_some(g)
}

fn is_zero_array(init: &[Tt]) -> bool {
    zero_array_group(init).is_some()
}

/// `Some`/`None` actuals and boolean literals decide an atom of a callee.
fn atom_truth(atom: &str, actuals: &HashMap<String, Vec<Tt>>) -> Option<bool> {
    if let Some(rest) = atom.strip_prefix("let Some (") {
        let (_, tail) = rest.split_once(") = ")?;
        let a = text(actuals.get(tail.trim())?);
        if a == "None" {
            return Some(false);
        }
        if a.starts_with("Some (") {
            return Some(true);
        }
        return None;
    }
    let a = text(actuals.get(atom.trim())?);
    match a.as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn consistent(v: &Variant, actuals: &HashMap<String, Vec<Tt>>) -> bool {
    v.assign
        .iter()
        .all(|(a, val)| atom_truth(a, actuals).map_or(true, |t| t == *val))
}

/// A builder may only be used through `b.method(..)` (validated by the push
/// collector) or returned as the block's tail.
fn check_builder_uses(ts: &[Tt], name: &str, top: bool) -> Result<(), String> {
    let n = ts.len();
    for k in 0..n {
        match &ts[k] {
            Tt::Id(s, _) if s == name => {
                let method = ts.get(k + 1).map_or(false, |t| t.is_p('.'));
                let tail = top && k + 1 == n;
                if !(method || tail) {
                    return Err(format!("blob builder `{name}` is used other than through its methods before the launch"));
                }
            }
            Tt::G(_, c, _) => check_builder_uses(c, name, false)?,
            _ => {}
        }
    }
    Ok(())
}

/// Only indexed `copy_from_slice` writes (and length reads) may touch the array, plus
/// returning it as the function's tail: that is what keeps unwritten bytes zero.
fn check_array_uses(ts: &[Tt], arr: &str, top: bool) -> Result<(), String> {
    let n = ts.len();
    for k in 0..n {
        match &ts[k] {
            Tt::Id(s, _) if s == arr => {
                let copy_write = ts.get(k + 1).map_or(false, |t| t.is_g('['))
                    && ts.get(k + 2).map_or(false, |t| t.is_p('.'))
                    && ts.get(k + 3).map_or(false, |t| t.is_id("copy_from_slice"));
                let len_read = ts.get(k + 1).map_or(false, |t| t.is_p('.'))
                    && ts.get(k + 2).map_or(false, |t| t.is_id("len"));
                let tail = top && k + 1 == n;
                if !(copy_write || len_read || tail) {
                    return Err(format!(
                        "kernarg array `{arr}` has a use before the launch the zero-init proof cannot follow (near `{}`)",
                        text(&ts[k.saturating_sub(2)..(k + 4).min(n)])
                    ));
                }
            }
            Tt::G(_, c, _) => check_array_uses(c, arr, false)?,
            _ => {}
        }
    }
    Ok(())
}

/// What the blob-operand tracer needs besides the scope: the changed kernel names and
/// the launch's kernel-name variable (to pair tuple-loop blobs with their kernels).
/// Cycle/memo bookkeeping of one blob-operand trace (see `World::guarded`).
#[derive(Default)]
struct FlowState {
    active: HashSet<String>,
    /// Open contexts a finished expansion was cut against.
    cuts: Vec<String>,
}

struct Cx<'a> {
    names: &'a BTreeSet<String>,
    name_var: Option<String>,
    st: RefCell<FlowState>,
}

/// A call actual that is a plain local bound to a literal `None` / `Some(..)` / bool is
/// replaced by that literal (its ORIGIN), so the callee's branches are selected by the
/// value the caller really passes, and the memo key sees that value, not just the name.
fn canon_actual(a: &[Tt], idx: &BindIndex, body: &[Tt], limit: Option<&[usize]>) -> Vec<Tt> {
    // An array literal of locals (`[pp, ip, ..]`): each element that is a plain local
    // whose unchanged value is a pointer expression is replaced by that expression, so the
    // callee sees (and the memo key records) where every element REALLY comes from.
    let pa = peel(a);
    if pa.len() == 1 && pa[0].is_g('[') {
        if let Some(l) = limit {
            let mut out: Vec<Tt> = Vec::new();
            for (k, e) in split_top(pa[0].group().unwrap(), ',')
                .into_iter()
                .enumerate()
            {
                if k > 0 {
                    out.push(Tt::P(',', 0));
                }
                let el = inline_ptr_locals(&e, idx, body, l, 0);
                out.extend(el);
            }
            return vec![Tt::G('[', out, 0)];
        }
    }
    if a.len() == 1 {
        if let (Some(id), Some(l)) = (a[0].id(), limit) {
            if let Some(b) = idx.visible(id, l) {
                let init = peel(&b.init);
                let lit = (init.len() == 1
                    && (init[0].is_id("None") || init[0].is_id("true") || init[0].is_id("false")))
                    || (init.len() == 2 && init[0].is_id("Some") && init[1].is_g('('));
                // The literal only stands for the value at the call when nothing between
                // the binding and the call can have changed it.
                if lit && b.for_pos.is_none() && !changed_between(body, idx, id, &b.loc, l) {
                    return init;
                }
            }
        }
    }
    a.to_vec()
}

/// Can the local `name` (bound at `bloc`) have changed value before `limit`? True for any
/// reassignment (`name = ..`, compound assignment), `&mut name`, mutating method call, any
/// use the scan cannot classify (passed on, moved), or a reassignment recorded by the index.
fn changed_between(
    body: &[Tt],
    idx: &BindIndex,
    name: &str,
    bloc: &[usize],
    limit: &[usize],
) -> bool {
    if idx.assigns.get(name).map_or(false, |v| {
        v.iter()
            .any(|(l, _)| l.as_slice() > bloc && l.as_slice() < limit)
    }) {
        return true;
    }
    let mut occ: Vec<Loc> = Vec::new();
    occurrences(body, name, &mut Vec::new(), &mut occ);
    for o in occ {
        if !(o.as_slice() > bloc && o.as_slice() < limit) {
            continue;
        }
        let Some((last, parent)) = o.split_last() else {
            continue;
        };
        let Some(ts) = tokens_at(body, parent) else {
            return true;
        };
        let i = *last;
        let decl = (i >= 1 && ts[i - 1].is_id("let"))
            || (i >= 2 && ts[i - 1].is_id("mut") && ts[i - 2].is_id("let"));
        if decl {
            continue;
        }
        if i >= 2 && ts[i - 1].is_id("mut") && ts[i - 2].is_p('&') {
            return true;
        }
        match ts.get(i + 1) {
            Some(t) if t.is_p('=') => {
                if !ts.get(i + 2).map_or(false, |x| x.is_p('=')) {
                    return true;
                }
            }
            Some(t) if matches!(t, Tt::P(c, _) if "+-*/|&^%".contains(*c)) => {
                if ts.get(i + 2).map_or(false, |x| x.is_p('=')) {
                    return true;
                }
            }
            Some(t) if t.is_p('.') => {
                let m = ts.get(i + 2).and_then(|x| x.id()).unwrap_or("");
                if !matches!(
                    m,
                    "clone"
                        | "is_some"
                        | "is_none"
                        | "unwrap"
                        | "copied"
                        | "cloned"
                        | "as_ref"
                        | "len"
                ) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Where one call actual comes from: a local's binding location and initialiser (so two
/// callers that pass the same NAME bound to different values never share a memo entry), a
/// parameter, or the expression text itself.
fn actual_origin(a: &[Tt], idx: &BindIndex, limit: Option<&[usize]>) -> String {
    if a.len() == 1 {
        if let Some(id) = a[0].id() {
            if let Some(l) = limit {
                if let Some(b) = idx.visible(id, l) {
                    return format!("{id}@{:?}={}", b.loc, text(&b.init));
                }
            }
            return format!("{id}@param/unbound");
        }
    }
    text(a)
}

/// Read-only views (and `push`, whose argument is traced). The `&mut` views `iter_mut` and
/// `as_mut_slice` are NOT here: they hand out aliases, see `collection_uses`.
const COLLECTION_METHODS: [&str; 8] = [
    "push",
    "iter",
    "into_iter",
    "len",
    "is_empty",
    "clone",
    "as_slice",
    "first",
];

/// Struct-literal field initialisers and `.field = expr;` assignments of `field` in
/// the function bodies of one file: (function, location, expression).
fn field_sources_in(ts: &[Tt], field: &str, loc: &mut Loc, out: &mut Vec<(Loc, Vec<Tt>)>) {
    let n = ts.len();
    for i in 0..n {
        loc.push(i);
        if let Tt::Id(s, _) = &ts[i] {
            let upper = s == "Self" || s.chars().next().map_or(false, |c| c.is_uppercase());
            if upper && ts.get(i + 1).map_or(false, |t| t.is_g('{')) {
                for part in split_top(ts[i + 1].group().unwrap(), ',') {
                    if part.first().map_or(false, |t| t.is_id(field)) {
                        if part.len() == 1 {
                            out.push((loc.clone(), vec![Tt::Id(field.to_string(), 0)]));
                        } else if part[1].is_p(':') && !part.get(2).map_or(false, |t| t.is_p(':')) {
                            out.push((loc.clone(), part[2..].to_vec()));
                        }
                    }
                }
            }
        }
        if ts[i].is_p('.')
            && ts.get(i + 1).map_or(false, |t| t.is_id(field))
            && ts.get(i + 2).map_or(false, |t| t.is_p('='))
            && !ts.get(i + 3).map_or(false, |t| t.is_p('='))
        {
            let mut rhs = Vec::new();
            let mut k = i + 3;
            while k < n && !ts[k].is_p(';') {
                rhs.push(ts[k].clone());
                k += 1;
            }
            out.push((loc.clone(), rhs));
        }
        if let Tt::G(_, c, _) = &ts[i] {
            field_sources_in(c, field, loc, out);
        }
        loc.pop();
    }
}

/// Calls `fname(args)` / `.fname(args)`: (location, actual at `argpos`).
pub(crate) fn call_sources_in(
    ts: &[Tt],
    fname: &str,
    argpos: usize,
    loc: &mut Loc,
    out: &mut Vec<(Loc, Vec<Tt>)>,
) {
    let n = ts.len();
    for i in 0..n {
        loc.push(i);
        if ts[i].is_id(fname)
            && ts.get(i + 1).map_or(false, |t| t.is_g('('))
            && !(i > 0 && ts[i - 1].is_id("fn"))
        {
            let args = split_top(ts[i + 1].group().unwrap(), ',');
            if let Some(a) = args.get(argpos) {
                out.push((loc.clone(), a.clone()));
            }
        }
        if let Tt::G(_, c, _) = &ts[i] {
            call_sources_in(c, fname, argpos, loc, out);
        }
        loc.pop();
    }
}

/// `x.push(EXPR)` calls, plus every other use of the collection `x` before the launch.
/// Only read-only methods, shared borrows (`&x`), the declaration itself and (for a
/// function's returned tail) the final expression are understood: a `&mut` alias, a
/// reassignment, an index write, passing `x` on, moving it, or any mutating method is
/// reported in `bad` (an alias could mutate the blob without naming `x`).
#[allow(clippy::too_many_arguments)]
fn collection_uses(
    ts: &[Tt],
    x: &str,
    bloc: &[usize],
    allow_tail: bool,
    top: bool,
    callee: Option<&str>,
    rows_ok: bool,
    in_struct: bool,
    reader: &dyn Fn(&str, usize, &[usize]) -> bool,
    shadow: &[Loc],
    loc: &mut Loc,
    pushes: &mut Vec<(Loc, Vec<Tt>)>,
    bad: &mut Vec<String>,
) {
    let n = ts.len();
    let plen = bloc.len().saturating_sub(1);
    // `for (a, b) in [(.., &mut x), ..] { launch(.., b) }`: borrowing `x` into the rows of a
    // loop whose bound variable is only ever handed to launches reads it.
    let mut approved: HashSet<usize> = HashSet::new();
    // `for b in x.iter_mut() { .. b only handed to launches .. }`: the index of `x` in the header.
    let mut iter_ok: HashSet<usize> = HashSet::new();
    for f in 0..n {
        if !ts[f].is_id("for") {
            continue;
        }
        let Ok((pat, it, j)) = for_parts(ts, f) else {
            continue;
        };
        if it.len() == 4
            && it[0].is_id(x)
            && it[1].is_p('.')
            && it[2].is_id("iter_mut")
            && it[3].is_g('(')
        {
            if let (Some(var), Some(kin)) = (
                pat.iter().find_map(|t| t.id().filter(|s| *s != "mut")),
                (f + 1..n).find(|k| ts[*k].is_id("in")),
            ) {
                let mut at = loc.clone();
                at.push(f);
                let rd = |c: &str, a: usize| reader(c, a, at.as_slice());
                if ts[j]
                    .group()
                    .map_or(false, |g| uses_are_launch_reads(g, var, &rd))
                {
                    iter_ok.insert(kin + 1);
                }
            }
            continue;
        }
        if !(it.len() == 1 && it[0].is_g('[') && pat.len() == 1 && pat[0].is_g('(')) {
            continue;
        }
        let Some(kin) = (f + 1..n).find(|k| ts[*k].is_id("in")) else {
            continue;
        };
        let comps = split_top(pat[0].group().unwrap(), ',');
        let rows = split_top(it[0].group().unwrap(), ',');
        let mut any = false;
        let mut ok = true;
        for (c, part) in comps.iter().enumerate() {
            let holds = rows.iter().any(|r| {
                r.len() == 1
                    && r[0].is_g('(')
                    && split_top(r[0].group().unwrap(), ',')
                        .get(c)
                        .map_or(false, |t| {
                            t.len() == 3 && t[0].is_p('&') && t[1].is_id("mut") && t[2].is_id(x)
                        })
            });
            if holds {
                any = true;
                let var = part.iter().find_map(|t| t.id().filter(|s| *s != "mut"));
                let mut at = loc.clone();
                at.push(f);
                let rd = |c: &str, a: usize| reader(c, a, at.as_slice());
                let body_ok = ts[j].group().map_or(false, |g| {
                    var.map_or(false, |v| uses_are_launch_reads(g, v, &rd))
                });
                ok &= body_ok;
            }
        }
        if any && ok {
            approved.insert(kin + 1);
        }
    }
    for i in 0..n {
        loc.push(i);
        // Only occurrences in this binding's own scope and after it (another variable of
        // the same name elsewhere in the function is a different value).
        let relevant = loc.len() > plen && loc[..plen] == bloc[..plen] && loc.as_slice() > bloc;
        // `.x` (a field or method of that name) and `x:` (a parameter, field or type
        // annotation) are not uses of the local `x`.
        let named_other = (i >= 1 && ts[i - 1].is_p('.'))
            || (ts.get(i + 1).map_or(false, |t| t.is_p(':'))
                && !ts.get(i + 2).map_or(false, |t| t.is_p(':')));
        if ts[i].is_id(x) && relevant && !named_other {
            let method = ts.get(i + 1).map_or(false, |t| t.is_p('.'))
                && ts.get(i + 2).and_then(|t| t.id()).is_some();
            let decl = (i >= 1 && ts[i - 1].is_id("let"))
                || (i >= 2 && ts[i - 1].is_id("mut") && ts[i - 2].is_id("let"));
            let shared = i >= 1 && ts[i - 1].is_p('&') && !(i >= 2 && ts[i - 2].is_p('&'));
            let mut_ref = i >= 2 && ts[i - 1].is_id("mut") && ts[i - 2].is_p('&');
            let tail = top && allow_tail && i + 1 == n;
            if method {
                let m = ts[i + 2].id().unwrap();
                if m == "push" {
                    if let Some(g) = ts.get(i + 3).and_then(|t| t.group()) {
                        pushes.push((loc.clone(), g.to_vec()));
                    }
                } else if m == "iter_mut" {
                    if !iter_ok.contains(&i) {
                        bad.push("an `iter_mut()` that is not a loop handing each element only to launches".to_string());
                    }
                } else if m == "as_mut_slice" {
                    // Only as a whole launch argument: `launch(.., x.as_mut_slice())`.
                    let direct = (i == 0 || ts[i - 1].is_p(','))
                        && ts.get(i + 4).map_or(true, |t| t.is_p(','))
                        && callee.map_or(false, |c| {
                            LAUNCH_NAMES.contains(&c)
                                || reader(
                                    c,
                                    ts[..i].iter().filter(|t| t.is_p(',')).count(),
                                    loc.as_slice(),
                                )
                        });
                    if !direct {
                        bad.push(
                            "an `as_mut_slice()` view that is not a launch argument".to_string(),
                        );
                    }
                } else if !COLLECTION_METHODS.contains(&m) {
                    bad.push(format!("a `.{m}()` call"));
                }
            } else if mut_ref
                && (rows_ok
                    || callee.map_or(false, |c| {
                        // An earlier launch of the same blob, or a call of a helper that only
                        // hands this very argument to a launch, reads it (`&mut` is the signature).
                        LAUNCH_NAMES.contains(&c)
                            || reader(
                                c,
                                ts[..i].iter().filter(|t| t.is_p(',')).count(),
                                loc.as_slice(),
                            )
                    }))
            {
            } else if mut_ref {
                bad.push("a `&mut` borrow outside a launch call (mutable alias)".to_string());
            } else if !(decl
                || shared
                || tail
                // `Case { .., x, .. }`: moved into a struct field (its later uses are the field's).
                || (in_struct && (i == 0 || ts[i - 1].is_p(',')) && (i + 1 == n || ts[i + 1].is_p(','))))
            {
                bad.push(
                    "a use the tracer cannot follow (alias, move, reassignment or index write)"
                        .to_string(),
                );
            }
        }
        if let Tt::G(_, c, _) = &ts[i] {
            // The body of a closure with its own parameter `x` talks about THAT `x`.
            if !shadow.iter().any(|s| s.as_slice() == loc.as_slice()) {
                let call = if ts[i].is_g('(') && i >= 1 {
                    ts[i - 1].id()
                } else {
                    None
                };
                let lit = ts[i].is_g('{')
                    && i >= 1
                    && ts[i - 1].id().map_or(false, |s| {
                        s == "Self" || s.chars().next().map_or(false, |ch| ch.is_uppercase())
                    });
                collection_uses(
                    c,
                    x,
                    bloc,
                    allow_tail,
                    false,
                    call,
                    rows_ok || approved.contains(&i),
                    lit,
                    reader,
                    shadow,
                    loc,
                    pushes,
                    bad,
                );
            }
        }
        loc.pop();
    }
}

const LAUNCH_NAMES: [&str; 8] = [
    "launch_maybe_blob",
    "launch_maybe_blob_declared",
    "launch_maybe_blob_position_grid",
    "launch_maybe_blob_bound",
    "launch_params_blob",
    "launch_kernel_blob",
    "launch_blob_recorded",
    "launch_kernel",
];

/// Body of a closure value `|..| expr` / `|..| -> T { .. }` / `move || ..`.
fn closure_body(init: &[Tt]) -> Option<Vec<Tt>> {
    let mut k = 0usize;
    if init.get(k).map_or(false, |t| t.is_id("move")) {
        k += 1;
    }
    if !init.get(k).map_or(false, |t| t.is_p('|')) {
        return None;
    }
    k += 1;
    while k < init.len() && !init[k].is_p('|') {
        k += 1;
    }
    let rest = init.get(k + 1..)?;
    // A block body is its brace group; an explicitly typed closure `|..| -> T { .. }` is too.
    // Any other expression (`match x { .. }`, `if c { .. } else { .. }`, a call) is kept whole:
    // its trailing brace group is only one arm or branch of it.
    if rest.len() == 1 && rest[0].is_g('{') {
        Some(vec![rest[0].clone()])
    } else if rest.len() >= 3
        && rest[0].is_p('-')
        && rest[1].is_p('>')
        && rest.last().map_or(false, |t| t.is_g('{'))
    {
        Some(vec![rest.last().unwrap().clone()])
    } else {
        Some(rest.to_vec())
    }
}

impl World {
    /// Layout variants of the blob operand of one launch site, from the operand's
    /// actual origin in the source. Whatever cannot be traced fails closed.
    pub fn probe_variants(
        &self,
        fi: usize,
        fk: usize,
        site: &Site,
        names: &BTreeSet<String>,
    ) -> Result<Vec<Variant>, String> {
        let func = &self.files[fi].fns[fk];
        let operand = site_blob(site).ok_or("launch without arguments")?;
        let sc = Sc {
            fi,
            func,
            body: &func.body,
        };
        let nt = peel(&site.name_ex);
        let name_var = if nt.len() == 1 {
            nt[0].id().map(|s| s.to_string())
        } else {
            None
        };
        let cx = Cx {
            names,
            name_var,
            st: RefCell::new(FlowState::default()),
        };
        let mut out = Vec::new();
        match self.operand(
            operand,
            &sc,
            Some(&site.loc),
            &HashMap::new(),
            None,
            &cx,
            0,
            &mut out,
        ) {
            Ok(()) if out.is_empty() => Err(format!(
                "blob operand `{}` produced no layout",
                text(operand)
            )),
            Ok(()) => Ok(out),
            Err(FlowErr::Fail(e)) | Err(FlowErr::Unresolved(e)) => Err(format!(
                "blob operand `{}` cannot be traced to a builder ({e})",
                text(operand)
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn operand(
        &self,
        ex: &[Tt],
        sc: &Sc,
        limit: Option<&[usize]>,
        actuals: &HashMap<String, Vec<Tt>>,
        comp: Option<usize>,
        cx: &Cx,
        depth: u32,
        out: &mut Vec<Variant>,
    ) -> FResult<()> {
        if depth > 10 {
            return fail("blob operand flow is too deep");
        }
        let t = peel(ex);
        let n = t.len();
        if n == 0 {
            return unres("empty operand");
        }
        // A tuple place: pick the component the consumer binds.
        if let Some(k) = comp {
            if n == 1 && t[0].is_g('(') {
                let parts = split_top(t[0].group().unwrap(), ',');
                return match parts.get(k) {
                    Some(p) => self.operand(p, sc, limit, actuals, None, cx, depth + 1, out),
                    None => fail("tuple component out of range"),
                };
            }
        }
        if n == 1 && t[0].is_g('{') {
            let g = t[0].group().unwrap();
            let tail = tail_expr(g);
            return self.operand(
                &tail,
                &Sc { body: g, ..*sc },
                None,
                actuals,
                comp,
                cx,
                depth + 1,
                out,
            );
        }
        if t[0].is_id("if") {
            let mut bodies: Vec<(Vec<Tt>, Conds)> = Vec::new();
            if_chain(&t, 0, &Vec::new(), |b, c| {
                bodies.push((b.to_vec(), c.clone()));
                Ok(())
            })
            .map_err(FlowErr::Fail)?;
            if bodies.is_empty() {
                return fail("empty if expression");
            }
            for (b, conds) in bodies {
                let tail = tail_expr(&b);
                // Branch bodies are blocks: names are resolved inside them.
                let bsc_tokens = b.clone();
                let bsc = Sc {
                    body: &bsc_tokens,
                    ..*sc
                };
                let mut sub = Vec::new();
                self.operand(&tail, &bsc, None, actuals, comp, cx, depth + 1, &mut sub)?;
                // A layout produced by a branch only exists when its condition holds.
                for v in sub.iter_mut() {
                    for (a, val) in &conds {
                        v.assign.insert(a.clone(), *val);
                    }
                }
                out.extend(sub);
            }
            return Ok(());
        }
        // Collection forms: contributions are the elements.
        if n == 3 && t[0].is_id("vec") && t[1].is_p('!') && t[2].is_g('[') {
            return self.elements(
                t[2].group().unwrap(),
                sc,
                limit,
                actuals,
                comp,
                cx,
                depth,
                out,
            );
        }
        if n == 1 && t[0].is_g('[') {
            let g = t[0].group().unwrap();
            if let Some(semi) = g.iter().position(|x| x.is_p(';')) {
                if is_zero_array(&t) {
                    return unres("zero array used directly as an operand");
                }
                return self.operand(&g[..semi], sc, limit, actuals, comp, cx, depth + 1, out);
            }
            return self.elements(g, sc, limit, actuals, comp, cx, depth, out);
        }
        if n >= 5 && t[n - 1].is_g('(') && t[n - 2].is_id("collect") && t[n - 3].is_p('.') {
            return self.collect_chain(&t[..n - 3], sc, limit, actuals, comp, cx, depth, out);
        }
        // `Vec::new()` / `Vec::with_capacity(n)`: an empty collection (only pushes fill it).
        if n >= 5
            && t[0].is_id("Vec")
            && t[1].is_p(':')
            && t[2].is_p(':')
            && matches!(t[3].id(), Some("new" | "with_capacity" | "default"))
            && t[n - 1].is_g('(')
        {
            return Ok(());
        }
        if n >= 2 && t[n - 1].is_g('(') {
            if let Some(name) = t[n - 2].id() {
                // `Type::assoc(..)` (e.g. `Foo::new()`) is not a same-named free function.
                let qual = if n >= 5 && t[n - 3].is_p(':') && t[n - 4].is_p(':') {
                    t[n - 5].id()
                } else {
                    None
                };
                if qual.map_or(false, |q| {
                    q != "Self" && q.chars().next().map_or(false, |c| c.is_uppercase())
                }) {
                    return unres(format!("associated function `{}::{name}`", qual.unwrap()));
                }
                let method = n >= 3 && t[n - 3].is_p('.');
                let recv_ty = if method {
                    self.recv_type(&t[..n - 3], sc, limit, 0)
                } else {
                    None
                };
                return self.operand_call(
                    name,
                    &t[n - 1],
                    sc,
                    comp,
                    method,
                    recv_ty,
                    limit,
                    cx,
                    depth,
                    out,
                );
            }
        }
        // A field path `a.b.f`: every initialiser/assignment of `f` in this file.
        if n >= 3
            && t.iter().enumerate().all(|(k, x)| {
                if k % 2 == 0 {
                    x.id().is_some()
                } else {
                    x.is_p('.')
                }
            })
        {
            let field = t[n - 1].id().unwrap().to_string();
            // `root.field` where `root` is a local built by a struct literal (directly or
            // by a function whose tail is one): that literal's own field, nothing else.
            if n == 3 {
                if self.project_field(
                    t[0].id().unwrap(),
                    &field,
                    sc,
                    limit,
                    comp,
                    cx,
                    depth,
                    out,
                )? {
                    return Ok(());
                }
            }
            let srcs = self.field_srcs(sc.fi, &field);
            if srcs.is_empty() {
                return unres(format!("field `{field}` is never initialised in this file"));
            }
            let key = format!("field|{}|{field}|{comp:?}|{:?}", sc.fi, cx.name_var);
            let sc_fi = sc.fi;
            return self.guarded(key, cx, out, |sub| {
                for (fk, loc, expr) in srcs.iter() {
                    let f = &self.files[sc_fi].fns[*fk];
                    self.operand(
                        expr,
                        &Sc {
                            fi: sc_fi,
                            func: f,
                            body: &f.body,
                        },
                        Some(loc.as_slice()),
                        &HashMap::new(),
                        comp,
                        cx,
                        depth + 1,
                        sub,
                    )?;
                }
                Ok(())
            });
        }
        if n == 1 {
            if let Some(x) = t[0].id() {
                return self.operand_ident(x, sc, limit, actuals, comp, cx, depth, out);
            }
        }
        unres(format!("operand `{}`", text(&t)))
    }

    /// Each element of an array/`vec!` literal.
    #[allow(clippy::too_many_arguments)]
    fn elements(
        &self,
        g: &[Tt],
        sc: &Sc,
        limit: Option<&[usize]>,
        actuals: &HashMap<String, Vec<Tt>>,
        comp: Option<usize>,
        cx: &Cx,
        depth: u32,
        out: &mut Vec<Variant>,
    ) -> FResult<()> {
        for e in split_top(g, ',') {
            self.operand(&e, sc, limit, actuals, comp, cx, depth + 1, out)?;
        }
        Ok(())
    }

    /// `RECV.map(|..| BODY).collect()` yields BODY's value; without a `map` the
    /// elements are the receiver's own.
    #[allow(clippy::too_many_arguments)]
    fn collect_chain(
        &self,
        chain: &[Tt],
        sc: &Sc,
        limit: Option<&[usize]>,
        actuals: &HashMap<String, Vec<Tt>>,
        comp: Option<usize>,
        cx: &Cx,
        depth: u32,
        out: &mut Vec<Variant>,
    ) -> FResult<()> {
        let n = chain.len();
        for k in (1..n).rev() {
            if chain[k].is_g('(') && chain[k - 1].is_id("map") && k >= 2 && chain[k - 2].is_p('.') {
                let arg = chain[k].group().unwrap();
                let body = closure_body(arg)
                    .ok_or_else(|| FlowErr::Unresolved("map argument is not a closure".into()))?;
                if body.len() == 1 && body[0].is_g('{') {
                    let g = body[0].group().unwrap();
                    let tail = tail_expr(g);
                    return self.operand(
                        &tail,
                        &Sc { body: g, ..*sc },
                        None,
                        actuals,
                        comp,
                        cx,
                        depth + 1,
                        out,
                    );
                }
                return self.operand(&body, sc, limit, actuals, comp, cx, depth + 1, out);
            }
        }
        // No `map`: the receiver's elements (adapters like `.iter().cloned()` peel off).
        let mut r = chain.to_vec();
        while r.len() >= 3
            && r[r.len() - 1].is_g('(')
            && r[r.len() - 3].is_p('.')
            && matches!(
                r[r.len() - 2].id(),
                Some("iter" | "into_iter" | "cloned" | "copied" | "iter_mut")
            )
        {
            let m = r.len() - 3;
            r.truncate(m);
        }
        self.operand(&r, sc, limit, actuals, comp, cx, depth + 1, out)
    }

    fn operand_call(
        &self,
        name: &str,
        args_g: &Tt,
        sc: &Sc,
        comp: Option<usize>,
        method: bool,
        recv_ty: Option<String>,
        limit: Option<&[usize]>,
        cx: &Cx,
        depth: u32,
        out: &mut Vec<Variant>,
    ) -> FResult<()> {
        let file = &self.files[sc.fi].path;
        // A closure bound to a local shadows any function of that name:
        // `let blob = || { .. }; .. blob()`.
        let caller_idx = self.bind_index(sc);
        {
            let idx = &caller_idx;
            if let Some(b) = idx.map.get(name).and_then(|v| v.last()) {
                if let Some(body) = closure_body(&b.init) {
                    if body.len() == 1 && body[0].is_g('{') {
                        let g = body[0].group().unwrap();
                        let tail = tail_expr(g);
                        return self.operand(
                            &tail,
                            &Sc { body: g, ..*sc },
                            None,
                            &HashMap::new(),
                            comp,
                            cx,
                            depth + 1,
                            out,
                        );
                    }
                    return self.operand(
                        &body,
                        sc,
                        None,
                        &HashMap::new(),
                        comp,
                        cx,
                        depth + 1,
                        out,
                    );
                }
            }
        }
        // A method of a known receiver type is that type's `impl` method of the name, wherever
        // the impl lives; otherwise the same-file functions of the name.
        let mut cands: Vec<(usize, usize)> = match (&recv_ty, method) {
            (Some(ty), true) => self
                .fn_index
                .get(name)
                .map(|l| {
                    l.iter()
                        .copied()
                        .filter(|(fi, fk)| {
                            self.files[*fi].fns[*fk].owner.as_deref() == Some(ty.as_str())
                        })
                        .collect()
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        if cands.is_empty() {
            cands = self.callee_fns(name, file);
        }
        if cands.is_empty() && !method {
            cands = self.imported_fns(name, file);
        }
        if cands.is_empty() && method {
            // `recv.method(..)`: the receiver's type is unknown, so a definition elsewhere
            // counts only when it is the single one of that name in the workspace.
            if let Some(l) = self.fn_index.get(name) {
                if l.len() == 1 {
                    cands = l.clone();
                }
            }
        }
        if cands.is_empty() {
            return unres(format!("call `{name}(..)` has no same-file definition"));
        }
        let raw_args = split_top(args_g.group().unwrap(), ',');
        let origins: Vec<String> = raw_args
            .iter()
            .map(|a| actual_origin(a, &caller_idx, limit))
            .collect();
        let call_args: Vec<Vec<Tt>> = raw_args
            .iter()
            .map(|a| canon_actual(a, &caller_idx, sc.body, limit))
            .collect();
        for (cfi, cfk) in cands {
            let cf = &self.files[cfi].fns[cfk];
            let formals = fn_formals(cf);
            if formals.len() != call_args.len() {
                return fail(format!(
                    "call `{name}` passes {} arguments, `{}` declares {}",
                    call_args.len(),
                    cf.name,
                    formals.len()
                ));
            }
            let fnames = formals.clone();
            let mut map: HashMap<String, Vec<Tt>> =
                formals.into_iter().zip(call_args.iter().cloned()).collect();
            // The complete symbolic context of this expansion: caller function, callee, tuple
            // component, kernel-name variable, and for every actual both the (canonical)
            // value the callee's branches see and the ORIGIN it comes from.
            let mut actual_text: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{k}={}", text(v)))
                .collect();
            actual_text.sort();
            let key = format!(
                "call|{}|{}|{cfi}|{cfk}|{comp:?}|{:?}|{}|{}",
                sc.fi,
                sc.func.line,
                cx.name_var,
                actual_text.join(";"),
                origins.join("&")
            );
            // A formal that sizes a kernarg array (`vec![0u8; len.max(64)]`) takes the one integer
            // the caller's actual can be, when that is provable (see `int_set`).
            for (fname_f, raw) in fnames.iter().zip(raw_args.iter()) {
                if sizes_array(&cf.body, fname_f) {
                    let fsc = Sc {
                        fi: sc.fi,
                        func: sc.func,
                        body: &sc.func.body,
                    };
                    self.why.borrow_mut().clear();
                    if let Some(set) = self.int_set(raw, &fsc, None, 0) {
                        if set.len() == 1 {
                            let v = *set.iter().next().unwrap();
                            map.insert(fname_f.clone(), vec![Tt::N(v.to_string(), 0)]);
                        }
                    }
                }
            }
            // A formal that receives an array literal (`ptrs: [u64; 5]` called with
            // `[a.as_ptr() as u64, ..]`) is replaced by it inside the callee, so loops over
            // it expand to the real elements and keep their pointer provenance.
            let arrays: HashMap<String, Vec<Tt>> = map
                .iter()
                .filter_map(|(k, v)| {
                    let p = peel(v);
                    if p.len() == 1 && p[0].is_g('[') {
                        Some((k.clone(), p))
                    } else {
                        None
                    }
                })
                .collect();
            let subst_body: Vec<Tt> = if arrays.is_empty() {
                Vec::new()
            } else {
                substitute(&cf.body, &arrays)
            };
            let cbody: &[Tt] = if arrays.is_empty() {
                &cf.body
            } else {
                &subst_body
            };
            self.guarded(key, cx, out, |sub| {
                let tail = tail_expr(cbody);
                self.operand(
                    &tail,
                    &Sc {
                        fi: cfi,
                        func: cf,
                        body: cbody,
                    },
                    None,
                    &map,
                    comp,
                    cx,
                    depth + 1,
                    sub,
                )?;
                sub.retain(|v| consistent(v, &map));
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Run one origin expansion once per complete context. Re-entering a context that
    /// is still being expanded adds nothing new (the layouts are a union, so the cycle
    /// is a fixed point); a finished expansion is reused, but only when it did not
    /// depend on an enclosing, still-open context (a cut against another open context
    /// would make the result incomplete).
    fn guarded<F>(&self, key: String, cx: &Cx, out: &mut Vec<Variant>, f: F) -> FResult<()>
    where
        F: FnOnce(&mut Vec<Variant>) -> FResult<()>,
    {
        let start;
        {
            if let Some(m) = self.flow_memo.borrow().get(&key) {
                out.extend(m.iter().cloned());
                return Ok(());
            }
            let mut st = cx.st.borrow_mut();
            if st.active.contains(&key) {
                st.cuts.push(key);
                return Ok(());
            }
            st.active.insert(key.clone());
            start = st.cuts.len();
        }
        let mut sub = Vec::new();
        let r = f(&mut sub);
        let mut st = cx.st.borrow_mut();
        st.active.remove(&key);
        r?;
        // Cuts against this very context are resolved by its own result.
        let tail: Vec<String> = st.cuts.split_off(start);
        let foreign: Vec<String> = tail.into_iter().filter(|k| *k != key).collect();
        let complete = foreign.is_empty();
        st.cuts.extend(foreign);
        drop(st);
        if complete {
            self.flow_memo.borrow_mut().insert(key, sub.clone());
        }
        out.extend(sub);
        Ok(())
    }

    /// `BindIndex` of a scope: built once per function body (blocks inside a body, which
    /// are temporary token slices, are indexed fresh).
    fn bind_index(&self, sc: &Sc) -> Rc<BindIndex> {
        if sc.body.as_ptr() == sc.func.body.as_ptr() && sc.body.len() == sc.func.body.len() {
            let key = (sc.fi, sc.func.body.as_ptr() as usize);
            if let Some(i) = self.idx_cache.borrow().get(&key) {
                return i.clone();
            }
            let i = Rc::new(BindIndex::build(sc.body));
            self.idx_cache.borrow_mut().insert(key, i.clone());
            return i;
        }
        Rc::new(BindIndex::build(sc.body))
    }

    /// Every initialiser/assignment of struct field `field` in file `fi`, scanned once.
    fn field_srcs(&self, fi: usize, field: &str) -> Rc<Vec<(usize, Loc, Vec<Tt>)>> {
        let key = format!("field|{fi}|{field}");
        if let Some(s) = self.flow_srcs.borrow().get(&key) {
            return s.clone();
        }
        let mut srcs: Vec<(usize, Loc, Vec<Tt>)> = Vec::new();
        for (fk, f) in self.files[fi].fns.iter().enumerate() {
            let mut found = Vec::new();
            field_sources_in(&f.body, field, &mut Vec::new(), &mut found);
            srcs.extend(found.into_iter().map(|(l, e)| (fk, l, e)));
        }
        let rc = Rc::new(srcs);
        self.flow_srcs.borrow_mut().insert(key, rc.clone());
        rc
    }

    /// The actual argument at `pos` of every call of `fname` in file `fi`, scanned once.
    pub(crate) fn call_srcs(
        &self,
        fi: usize,
        fname: &str,
        pos: usize,
    ) -> Rc<Vec<(usize, Loc, Vec<Tt>)>> {
        let key = format!("call|{fi}|{fname}|{pos}");
        if let Some(s) = self.flow_srcs.borrow().get(&key) {
            return s.clone();
        }
        let mut srcs: Vec<(usize, Loc, Vec<Tt>)> = Vec::new();
        for (fk, f) in self.files[fi].fns.iter().enumerate() {
            let mut found = Vec::new();
            call_sources_in(&f.body, fname, pos, &mut Vec::new(), &mut found);
            srcs.extend(found.into_iter().map(|(l, e)| (fk, l, e)));
        }
        let rc = Rc::new(srcs);
        self.flow_srcs.borrow_mut().insert(key, rc.clone());
        rc
    }

    #[allow(clippy::too_many_arguments)]
    fn operand_ident(
        &self,
        x: &str,
        sc: &Sc,
        limit: Option<&[usize]>,
        actuals: &HashMap<String, Vec<Tt>>,
        comp: Option<usize>,
        cx: &Cx,
        depth: u32,
        out: &mut Vec<Variant>,
    ) -> FResult<()> {
        let idx = self.bind_index(sc);
        // Inside a closure that declares a parameter named `x`, `x` IS that parameter (it
        // shadows any outer local): its blob is whatever each call of the closure passes.
        if let Some(l) = limit {
            if let Some((cname, pos)) = closure_param_origin(sc.body, &idx, x, l) {
                let mut found: Vec<(Loc, Vec<Tt>)> = Vec::new();
                call_sources_in(sc.body, &cname, pos, &mut Vec::new(), &mut found);
                if found.is_empty() {
                    return unres(format!(
                        "parameter `{x}` of closure `{cname}` is never supplied by a call"
                    ));
                }
                let key = format!(
                    "cparam|{}|{}|{cname}|{pos}|{comp:?}|{:?}",
                    sc.fi, sc.func.line, cx.name_var
                );
                return self.guarded(key, cx, out, |sub| {
                    for (loc, actual) in found.iter() {
                        self.operand(
                            actual,
                            sc,
                            Some(loc.as_slice()),
                            &HashMap::new(),
                            comp,
                            cx,
                            depth + 1,
                            sub,
                        )?;
                    }
                    Ok(())
                });
            }
        }
        let b = match limit {
            Some(l) => idx.visible(x, l),
            None => idx.map.get(x).and_then(|v| v.last()),
        };
        let Some(b) = b else {
            // A parameter: every call of this function in the file supplies it.
            let formals = fn_formals(sc.func);
            let Some(pos) = formals.iter().position(|f| f == x) else {
                return unres(format!("`{x}` has no visible binding"));
            };
            let srcs = self.call_srcs(sc.fi, &sc.func.name, pos);
            if srcs.is_empty() {
                return unres(format!(
                    "parameter `{x}` of `{}` is never passed in this file",
                    sc.func.name
                ));
            }
            let key = format!(
                "param|{}|{}|{}|{pos}|{comp:?}|{:?}",
                sc.fi, sc.func.name, sc.func.line, cx.name_var
            );
            let sc_fi = sc.fi;
            return self.guarded(key, cx, out, |sub| {
                for (fk, loc, actual) in srcs.iter() {
                    let f = &self.files[sc_fi].fns[*fk];
                    self.operand(
                        actual,
                        &Sc {
                            fi: sc_fi,
                            func: f,
                            body: &f.body,
                        },
                        Some(loc.as_slice()),
                        &HashMap::new(),
                        comp,
                        cx,
                        depth + 1,
                        sub,
                    )?;
                }
                Ok(())
            });
        };
        let bloc = b.loc.clone();
        let init = peel(&b.init);
        let file = &self.files[sc.fi].path;
        // The binding IS the builder only when `KernargBlob` is its own constructor; a
        // block that merely builds one inside (and returns bytes) is followed instead.
        if comp.is_none() && b.for_pos.is_none() && init.iter().any(|t| t.is_id("KernargBlob")) {
            let lim_body = match limit {
                Some(l) => cut_before(sc.body, l),
                None => sc.body.to_vec(),
            };
            let after = scope_after(&lim_body, &bloc);
            check_builder_uses(&after, x, true).map_err(FlowErr::Fail)?;
            let ctx = Ctx {
                file,
                body: sc.body,
                params: fn_params(sc.func),
                idx: &idx,
            };
            let mut pushes = Vec::new();
            self.collect_pushes(&after, x, &ctx, &Vec::new(), &mut pushes)
                .map_err(FlowErr::Fail)?;
            if pushes.is_empty() {
                return fail(format!(
                    "blob builder `{x}` has no pushes before the launch"
                ));
            }
            let vs = build_variants(&[], &pushes, false).map_err(FlowErr::Fail)?;
            out.extend(vs);
            return Ok(());
        }
        if comp.is_none() && b.for_pos.is_none() && is_zero_array(&init) {
            let g = zero_array_group(&init).unwrap();
            let semi = g.iter().position(|t| t.is_p(';')).unwrap();
            let len_ex = &g[semi + 1..];
            // `x.max(N)` with `x` a provable single integer (a formal given that value by the
            // caller, or a local): exactly that many bytes. Anything unproven fails.
            let len = match self.eval_const_expr_in(len_ex, &HashMap::new(), file) {
                Some(v) => v,
                None => match len_ex {
                    _ if len_ex.len() >= 4
                        && len_ex[len_ex.len() - 1].is_g('(')
                        && len_ex[len_ex.len() - 2].is_id("max")
                        && len_ex[len_ex.len() - 3].is_p('.') =>
                    {
                        let a = &len_ex[len_ex.len() - 1];
                        let recv = &len_ex[..len_ex.len() - 3];
                        let nmin = self
                            .eval_const_expr_in(a.group().unwrap(), &HashMap::new(), file)
                            .ok_or_else(|| {
                                FlowErr::Fail(format!(
                                    "cannot evaluate kernarg array minimum `{}`",
                                    text(a.group().unwrap())
                                ))
                            })?;
                        let given: Option<i64> = if recv.len() == 1 {
                            recv[0].id().and_then(|rid| actuals.get(rid)).and_then(|v| {
                                let p = peel(v);
                                match p.as_slice() {
                                    [Tt::N(s, _)] => s.parse::<i64>().ok(),
                                    _ => None,
                                }
                            })
                        } else {
                            None
                        };
                        let values: Option<BTreeSet<i64>> = match given {
                            Some(v) => Some(BTreeSet::from([v])),
                            None => self.int_set(recv, sc, Some(&bloc), 0),
                        };
                        match values {
                            Some(vs) if vs.len() == 1 => (*vs.iter().next().unwrap()).max(nmin),
                            _ => {
                                return fail(format!(
                                "the kernarg array length `{}` is not one provable size (why: {})",
                                text(len_ex),
                                self.why_text()
                            ))
                            }
                        }
                    }
                    _ => {
                        return fail(format!(
                            "cannot evaluate kernarg array length `{}`",
                            text(len_ex)
                        ))
                    }
                },
            };
            let lim_body = match limit {
                Some(l) => cut_before(sc.body, l),
                None => sc.body.to_vec(),
            };
            let after = scope_after(&lim_body, &bloc);
            check_array_uses(&after, x, true).map_err(FlowErr::Fail)?;
            let mut writes: Vec<(u32, u32, Conds)> = Vec::new();
            self.bytes_walk(&after, x, sc, &HashMap::new(), &Vec::new(), &mut writes)
                .map_err(FlowErr::Fail)?;
            let ec: Vec<&Conds> = writes.iter().map(|w| &w.2).collect();
            let atoms = atoms_of(&ec);
            if atoms.len() > 10 {
                return fail("too many independent kernarg write conditions");
            }
            for bits in 0..(1u32 << atoms.len()) {
                let mut assign = BTreeMap::new();
                for (k, a) in atoms.iter().enumerate() {
                    assign.insert(a.clone(), bits & (1 << k) != 0);
                }
                let ws: Vec<(u32, u32)> = writes
                    .iter()
                    .filter(|w| active(&w.2, &assign))
                    .map(|w| (w.0, w.1))
                    .collect();
                out.push(Variant {
                    assign,
                    param_idents: Vec::new(),
                    pushes: Vec::new(),
                    has_params: false,
                    bytes: Some(ByteLayout {
                        len: len as u32,
                        writes: ws,
                    }),
                    only_for: None,
                });
            }
            return Ok(());
        }
        // A `for` variable bound to a tuple loop: pair each element's blob with the
        // kernel its sibling name variable takes from the same element.
        if let (Some(Some(k)), Some(nv)) = (b.for_pos, cx.name_var.as_deref()) {
            // The name variable of the SAME loop (same binding site), not just the latest
            // binding of that name in the function.
            let sibling = idx.map.get(nv).and_then(|v| {
                v.iter()
                    .find(|nb| nb.loc == b.loc && matches!(nb.for_pos, Some(Some(_))))
            });
            if let Some(nb) = sibling {
                let j = nb.for_pos.unwrap().unwrap();
                let coll = peel(&b.init);
                let lit: Option<Vec<Tt>> = if coll.len() == 1 && coll[0].is_g('[') {
                    Some(coll[0].group().unwrap().to_vec())
                } else if coll.len() == 3 && coll[0].is_id("vec") && coll[2].is_g('[') {
                    Some(coll[2].group().unwrap().to_vec())
                } else {
                    None
                };
                if let Some(g) = lit {
                    let ctx = Ctx {
                        file,
                        body: sc.body,
                        params: fn_params(sc.func),
                        idx: &idx,
                    };
                    for e in split_top(&g, ',') {
                        if !(e.len() == 1 && e[0].is_g('(')) {
                            return fail("tuple loop element is not a tuple literal");
                        }
                        let parts = split_top(e[0].group().unwrap(), ',');
                        let (Some(blob_part), Some(name_part)) = (parts.get(k), parts.get(j))
                        else {
                            return fail("tuple loop element has too few components");
                        };
                        let r = self.resolve(name_part, &ctx, &bloc, 0, true);
                        let kernels: Vec<String> = r
                            .lits
                            .iter()
                            .filter(|l| cx.names.contains(*l))
                            .cloned()
                            .collect();
                        let mut sub = Vec::new();
                        self.operand(
                            blob_part,
                            sc,
                            Some(&bloc),
                            actuals,
                            None,
                            cx,
                            depth + 1,
                            &mut sub,
                        )?;
                        for mut v in sub {
                            if !kernels.is_empty() {
                                v.only_for = Some(kernels.clone());
                            }
                            out.push(v);
                        }
                    }
                    return Ok(());
                }
            }
        }
        // Otherwise the binding's value (or, for a loop variable, its iterable's
        // elements) plus everything pushed into it before the launch.
        let next_comp = match b.for_pos {
            Some(p) => p,
            None => comp,
        };
        let mut sub = Vec::new();
        self.operand(
            &binding_init(b),
            sc,
            Some(&bloc),
            actuals,
            next_comp,
            cx,
            depth + 1,
            &mut sub,
        )?;
        out.extend(sub);
        // Pushes into a collection place (`x.push(e)`) are contributions too; any other
        // mutating use cannot be followed.
        // (A loop variable's iterable was followed above, pushes included.)
        let place = if b.for_pos.is_some() {
            None
        } else {
            Some(x.to_string())
        };
        if let Some(pn) = place {
            let lim_body = match limit {
                Some(l) => cut_before(sc.body, l),
                None => sc.body.to_vec(),
            };
            let mut pushes = Vec::new();
            let mut bad = Vec::new();
            let reader =
                |n: &str, p: usize, at: &[usize]| self.blob_reader_scope(sc, &idx, n, p, at, 0);
            let shadow = closure_shadow_bodies(sc.body, &idx, &pn);
            collection_uses(
                &lim_body,
                &pn,
                &bloc,
                limit.is_none(),
                true,
                None,
                false,
                false,
                &reader,
                &shadow,
                &mut Vec::new(),
                &mut pushes,
                &mut bad,
            );
            if let Some(m) = bad.first() {
                return fail(format!(
                    "collection `{pn}` is changed before the launch by {m}"
                ));
            }
            for (ploc, expr) in pushes {
                self.operand(&expr, sc, Some(&ploc), actuals, comp, cx, depth + 1, out)?;
            }
        }
        Ok(())
    }

    fn array_len_in(&self, sc: &Sc, name: &str) -> Option<usize> {
        let file = &self.files[sc.fi].path;
        for t in &sc.func.sig {
            if let Tt::G('(', c, _) = t {
                for part in split_top(c, ',') {
                    let Some(k) = part.iter().position(|x| x.is_p(':')) else {
                        continue;
                    };
                    if part[..k].iter().any(|x| x.is_id(name)) {
                        for ty in &part[k + 1..] {
                            if let Tt::G('[', g, _) = ty {
                                let semi = g.iter().position(|x| x.is_p(';'))?;
                                return self
                                    .eval_const_expr_in(&g[semi + 1..], &HashMap::new(), file)
                                    .map(|v| v as usize);
                            }
                        }
                    }
                }
            }
        }
        let idx = self.bind_index(sc);
        let b = idx.map.get(name)?.last()?;
        let init = peel(&b.init);
        if init.len() == 1 && init[0].is_g('[') {
            let g = init[0].group().unwrap();
            if let Some(semi) = g.iter().position(|x| x.is_p(';')) {
                return self
                    .eval_const_expr_in(&g[semi + 1..], &HashMap::new(), file)
                    .map(|v| v as usize);
            }
            return Some(split_top(g, ',').len());
        }
        None
    }

    fn for_values(
        &self,
        pat: &[Tt],
        it: &[Tt],
        sc: &Sc,
    ) -> Result<Vec<HashMap<String, i64>>, String> {
        let file = &self.files[sc.fi].path;
        let mut names: Vec<String> = Vec::new();
        if pat.len() == 1 && pat[0].is_g('(') {
            for part in split_top(pat[0].group().unwrap(), ',') {
                names.push(
                    part.iter()
                        .find_map(|t| t.id().filter(|s| *s != "mut" && *s != "ref"))
                        .unwrap_or("_")
                        .to_string(),
                );
            }
        } else if pat.len() == 1 && pat[0].id().is_some() {
            names.push(pat[0].id().unwrap().to_string());
        } else {
            return Err(format!("unsupported loop pattern `{}`", text(pat)));
        }
        let mut out = Vec::new();
        if it.len() == 1 && it[0].is_g('[') {
            for elem in split_top(it[0].group().unwrap(), ',') {
                let fields: Vec<Vec<Tt>> = if elem.len() == 1 && elem[0].is_g('(') {
                    split_top(elem[0].group().unwrap(), ',')
                } else {
                    vec![elem.clone()]
                };
                let mut m = HashMap::new();
                for (k, nm) in names.iter().enumerate() {
                    if let Some(f) = fields.get(k) {
                        if let Some(v) = self.eval_const_expr_in(f, &HashMap::new(), file) {
                            m.insert(nm.clone(), v);
                        }
                    }
                }
                out.push(m);
            }
            return Ok(out);
        }
        // `[a, b, c].iter().enumerate()`: one iteration per literal element.
        if it.len() == 7
            && it[0].is_g('[')
            && it[1].is_p('.')
            && it[2].is_id("iter")
            && it[3].is_g('(')
            && it[4].is_p('.')
            && it[5].is_id("enumerate")
            && it[6].is_g('(')
        {
            let n = split_top(it[0].group().unwrap(), ',').len();
            for k in 0..n {
                let mut m = HashMap::new();
                m.insert(names[0].clone(), k as i64);
                out.push(m);
            }
            return Ok(out);
        }
        if it.len() == 7
            && it[0].id().is_some()
            && it[1].is_p('.')
            && it[2].is_id("iter")
            && it[3].is_g('(')
            && it[4].is_p('.')
            && it[5].is_id("enumerate")
            && it[6].is_g('(')
        {
            let n = self.array_len_in(sc, it[0].id().unwrap()).ok_or_else(|| {
                format!("cannot determine the length of `{}`", it[0].id().unwrap())
            })?;
            for k in 0..n {
                let mut m = HashMap::new();
                m.insert(names[0].clone(), k as i64);
                out.push(m);
            }
            return Ok(out);
        }
        Err(format!(
            "unsupported byte-array loop iterable `{}`",
            text(it)
        ))
    }

    fn bytes_walk(
        &self,
        ts: &[Tt],
        arr: &str,
        sc: &Sc,
        env: &HashMap<String, i64>,
        conds: &Conds,
        writes: &mut Vec<(u32, u32, Conds)>,
    ) -> Result<(), String> {
        let file = &self.files[sc.fi].path;
        let n = ts.len();
        let mut i = 0usize;
        while i < n {
            let t = &ts[i];
            if t.is_id(arr)
                && i + 3 < n
                && ts[i + 1].is_g('[')
                && ts[i + 2].is_p('.')
                && ts[i + 3].is_id("copy_from_slice")
            {
                let rng = ts[i + 1].group().unwrap();
                let d = rng
                    .iter()
                    .position(|x| x.is_p('.'))
                    .ok_or("kernarg write without a range")?;
                let lo = self
                    .eval_const_expr_in(&rng[..d], env, file)
                    .ok_or_else(|| format!("cannot evaluate offset `{}`", text(&rng[..d])))?;
                let hi = self
                    .eval_const_expr_in(&rng[d + 2..], env, file)
                    .ok_or_else(|| format!("cannot evaluate end `{}`", text(&rng[d + 2..])))?;
                if hi <= lo || lo < 0 {
                    return Err("empty or negative kernarg write".into());
                }
                writes.push((lo as u32, (hi - lo) as u32, conds.clone()));
                i += 4;
                continue;
            }
            if t.is_id("if") && !(i > 0 && ts[i - 1].is_id("else")) {
                let end = if_chain(ts, i, conds, |body, bc| {
                    self.bytes_walk(body, arr, sc, env, bc, writes)
                })?;
                i = end;
                continue;
            }
            if t.is_id("match") || t.is_id("while") || t.is_id("loop") {
                let mut j = i + 1;
                while j < n && !ts[j].is_g('{') {
                    j += 1;
                }
                if j < n && contains_id(ts[j].group().unwrap(), arr) {
                    return Err(format!(
                        "kernarg writes inside `{}` cannot be followed",
                        t.id().unwrap()
                    ));
                }
                i = j + 1;
                continue;
            }
            if t.is_id("for") {
                let (pat, it, j) = for_parts(ts, i)?;
                let body = ts[j].group().unwrap();
                if contains_id(body, arr) {
                    for asg in self.for_values(&pat, &it, sc)? {
                        let mut e2 = env.clone();
                        e2.extend(asg);
                        self.bytes_walk(body, arr, sc, &e2, conds, writes)?;
                    }
                }
                i = j + 1;
                continue;
            }
            if let Tt::G(_, c, _) = t {
                self.bytes_walk(c, arr, sc, env, conds, writes)?;
            }
            i += 1;
        }
        Ok(())
    }
}

/// Parameter names of a closure value `|a: T, mut b| ..` (after an optional `move`).
fn closure_params(init: &[Tt]) -> Option<Vec<String>> {
    let mut k = 0usize;
    if init.get(k).map_or(false, |t| t.is_id("move")) {
        k += 1;
    }
    if !init.get(k).map_or(false, |t| t.is_p('|')) {
        return None;
    }
    k += 1;
    let start = k;
    while k < init.len() && !init[k].is_p('|') {
        k += 1;
    }
    if k >= init.len() {
        return None;
    }
    let mut out = Vec::new();
    for part in split_top(&init[start..k], ',') {
        let pat: Vec<&Tt> = part.iter().take_while(|t| !t.is_p(':')).collect();
        out.push(
            pat.iter()
                .find_map(|t| t.id().filter(|s| *s != "mut" && *s != "ref"))
                .unwrap_or("_")
                .to_string(),
        );
    }
    Some(out)
}

/// One call of a launch helper: the kernels its name argument can be and the layouts of
/// the blob argument it passes, both evaluated in the CALLER.
pub struct HelperCall {
    pub line: u32,
    pub kernels: Vec<String>,
    pub variants: Vec<Variant>,
}

impl World {
    /// A blob-only launch whose kernel name is a parameter of the enclosing function (or of
    /// a closure inside it) is checked at every call of that helper, pairing the actual
    /// name with the actual blob of the SAME call. `None` = the launch names its kernel
    /// itself (the ordinary path).
    pub fn helper_calls(
        &self,
        fi: usize,
        fk: usize,
        site: &Site,
        names: &BTreeSet<String>,
    ) -> Option<Result<Vec<HelperCall>, String>> {
        let func = &self.files[fi].fns[fk];
        let sc = Sc {
            fi,
            func,
            body: &func.body,
        };
        let idx = self.bind_index(&sc);
        let nt = peel(&site.name_ex);
        if nt.len() != 1 {
            return None;
        }
        let nv = nt[0].id()?.to_string();
        if idx.visible(&nv, &site.loc).is_some() {
            return None;
        }
        if self.files[fi].consts.contains_key(&nv) || self.const_files.contains_key(&nv) {
            return None;
        }
        // Where is `nv` a parameter: a closure of this function, or the function itself?
        let formals = fn_formals(func);
        let mut closure: Option<(String, Vec<String>)> = None;
        let mut names_sorted: Vec<&String> = idx.map.keys().collect();
        names_sorted.sort();
        for cname in names_sorted {
            for b in &idx.map[cname] {
                if let Some(ps) = closure_params(&b.init) {
                    if ps.contains(&nv) {
                        closure = Some((cname.clone(), ps));
                    }
                }
            }
        }
        let (callee, params, in_fn_only) = match closure {
            Some((c, ps)) => (c, ps, true),
            None if formals.contains(&nv) => (func.name.clone(), formals.clone(), false),
            None => return None,
        };
        let pos_n = params.iter().position(|p| *p == nv)?;
        let ot = peel(site_blob(site)?);
        let root = ot.first().and_then(|t| t.id()).map(|s| s.to_string());
        let pos_b = root
            .as_ref()
            .and_then(|r| params.iter().position(|p| p == r));
        let rest: Vec<Tt> = if pos_b.is_some() {
            ot[1..].to_vec()
        } else {
            Vec::new()
        };

        // The calls: (function index, location) -> actual arguments by position.
        type Src = Vec<(usize, Loc, Vec<Tt>)>;
        let gather = |pos: usize| -> Src {
            if in_fn_only {
                let mut found = Vec::new();
                call_sources_in(&func.body, &callee, pos, &mut Vec::new(), &mut found);
                found.into_iter().map(|(l, e)| (fk, l, e)).collect()
            } else {
                (*self.call_srcs(fi, &callee, pos)).clone()
            }
        };
        let all_srcs: Vec<Src> = (0..params.len()).map(|p| gather(p)).collect();
        let name_srcs = all_srcs[pos_n].clone();
        if name_srcs.is_empty() {
            return Some(Err(format!("launch helper `{callee}` takes the kernel name `{nv}` as a parameter but is never called in this file")));
        }
        // The helper's own blob (when the blob is not a parameter) is traced per call, below;
        // calls without array arguments share one trace.
        let mut plain: Option<Vec<Variant>> = None;
        let mut calls = Vec::new();
        for (ck, loc, _) in &name_srcs {
            let cf = &self.files[fi].fns[*ck];
            let csc = Sc {
                fi,
                func: cf,
                body: &cf.body,
            };
            let cidx = self.bind_index(&csc);
            // Every actual argument of THIS call, then one row per tuple-loop row the
            // call sits in (`for (gl, fname, tag) in [(true, GL_FN, ..), (false, L_FN, ..)]`
            // pairs a kernel with the flag/blob that goes with it).
            let actuals: Vec<Vec<Tt>> = all_srcs
                .iter()
                .map(|s| {
                    s.iter()
                        .find(|(k2, l2, _)| k2 == ck && l2 == loc)
                        .map(|x| x.2.clone())
                        .unwrap_or_default()
                })
                .collect();
            let rows = match tuple_rows(&cidx, &actuals, loc) {
                Ok(r) => r,
                Err(e) => {
                    return Some(Err(format!(
                        "a `{callee}` call sits in a tuple loop the tracer cannot expand ({e})"
                    )))
                }
            };
            for row in rows {
                let name_actual = &row[pos_n];
                let kernels: Vec<String> = match self.names_of_actual(fi, *ck, loc, name_actual, names, 0) {
                    Ok(ks) => ks.into_iter().collect(),
                    Err(e) => {
                        return Some(Err(format!(
                            "the kernel-name argument `{}` of a `{callee}` call cannot be resolved ({e})",
                            text(name_actual)
                        )))
                    }
                };
                if kernels.is_empty() {
                    // Every name this call can pass was resolved and none is a changed kernel.
                    continue;
                }
                let map: HashMap<String, Vec<Tt>> = params
                    .iter()
                    .zip(row.iter())
                    .map(|(p, a)| {
                        (
                            p.clone(),
                            canon_actual(a, &cidx, &cf.body, Some(loc.as_slice())),
                        )
                    })
                    .collect();
                let variants = match (pos_b.is_none(), pos_b) {
                    (true, _) => {
                        // The helper builds its own blob (possibly from array parameters).
                        let arrays: HashMap<String, Vec<Tt>> = map
                            .iter()
                            .filter_map(|(k, v)| {
                                let p = peel(v);
                                if p.len() == 1 && p[0].is_g('[') {
                                    Some((k.clone(), p))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        let mut traced: Vec<Variant> = if arrays.is_empty() && plain.is_some() {
                            plain.clone().unwrap()
                        } else {
                            let subst: Vec<Tt> = if arrays.is_empty() {
                                Vec::new()
                            } else {
                                substitute(&func.body, &arrays)
                            };
                            let hbody: &[Tt] = if arrays.is_empty() {
                                &func.body
                            } else {
                                &subst
                            };
                            let hsc = Sc {
                                fi,
                                func,
                                body: hbody,
                            };
                            let cx = Cx {
                                names,
                                name_var: None,
                                st: RefCell::new(FlowState::default()),
                            };
                            let mut out = Vec::new();
                            match self.operand(
                                site_blob(site)?,
                                &hsc,
                                Some(&site.loc),
                                &HashMap::new(),
                                None,
                                &cx,
                                0,
                                &mut out,
                            ) {
                                Ok(()) if !out.is_empty() => {}
                                Ok(()) => {
                                    return Some(Err("the helper's blob produced no layout".into()))
                                }
                                Err(FlowErr::Fail(e)) | Err(FlowErr::Unresolved(e)) => {
                                    return Some(Err(format!(
                                        "the helper's blob cannot be traced to a builder ({e})"
                                    )))
                                }
                            }
                            if arrays.is_empty() {
                                plain = Some(out.clone());
                            }
                            out
                        };
                        traced.retain(|x| consistent(x, &map));
                        if traced.is_empty() {
                            return Some(Err(format!("no blob layout of `{callee}` is consistent with the arguments of one of its calls")));
                        }
                        traced
                    }
                    (false, Some(pb)) => {
                        let blob_actual = &row[pb];
                        let mut expr = peel(blob_actual);
                        expr.extend(rest.iter().cloned());
                        let cx = Cx {
                            names,
                            name_var: None,
                            st: RefCell::new(FlowState::default()),
                        };
                        let mut out = Vec::new();
                        match self.operand(&expr, &csc, Some(loc.as_slice()), &HashMap::new(), None, &cx, 0, &mut out) {
                            Ok(()) if !out.is_empty() => out,
                            Ok(()) => return Some(Err(format!("the blob passed to `{callee}` produced no layout"))),
                            Err(FlowErr::Fail(e)) | Err(FlowErr::Unresolved(e)) => {
                                return Some(Err(format!(
                                    "the blob `{}` passed to `{callee}` cannot be traced to a builder ({e})",
                                    text(blob_actual)
                                )))
                            }
                        }
                    }
                    (false, None) => {
                        return Some(Err("internal: helper blob source missing".into()))
                    }
                };
                let call_line = loc
                    .split_last()
                    .and_then(|(last, parent)| {
                        tokens_at(&cf.body, parent).and_then(|ts| ts.get(*last))
                    })
                    .map_or(0, |t| t.line());
                calls.push(HelperCall {
                    line: call_line,
                    kernels,
                    variants,
                });
            }
        }
        Some(Ok(calls))
    }
}

/// The expression a struct literal `Type { .., field: EXPR, .. }` (or shorthand `field`)
/// gives `field`.
fn struct_literal_field(init: &[Tt], field: &str) -> Option<Vec<Tt>> {
    let upper = init.len() == 2
        && init[0].id().map_or(false, |s| {
            s == "Self" || s.chars().next().map_or(false, |c| c.is_uppercase())
        })
        && init[1].is_g('{');
    if !upper {
        return None;
    }
    for part in split_top(init[1].group().unwrap(), ',') {
        if part.first().map_or(false, |t| t.is_id(field)) {
            if part.len() == 1 {
                return Some(vec![Tt::Id(field.to_string(), 0)]);
            }
            if part.len() > 2 && part[1].is_p(':') {
                return Some(part[2..].to_vec());
            }
        }
    }
    None
}

impl World {
    #[allow(clippy::too_many_arguments)]
    fn project_field(
        &self,
        root: &str,
        field: &str,
        sc: &Sc,
        limit: Option<&[usize]>,
        comp: Option<usize>,
        cx: &Cx,
        depth: u32,
        out: &mut Vec<Variant>,
    ) -> FResult<bool> {
        let idx = self.bind_index(sc);
        let Some(l) = limit else { return Ok(false) };
        let Some(b) = idx.visible(root, l) else {
            return Ok(false);
        };
        if b.for_pos.is_some() {
            return Ok(false);
        }
        let init = peel(&b.init);
        let bloc = b.loc.clone();
        // Between the constructor and the launch the field (and the whole object) may only be
        // read; any mutation or alias of THAT field, or an escape of the whole object, changes
        // what is launched. Other fields of the object are free to change.
        {
            let lim_body = cut_before(sc.body, l);
            let reader =
                |n: &str, p: usize, at: &[usize]| self.blob_reader_scope(sc, &idx, n, p, at, 0);
            let ro_method = |m: &str| self.ro_method_in(sc.fi, m);
            let obj_reader = |n: &str, p: usize| self.object_field_reader(sc.fi, n, p, field, 0);
            let mut bad: Vec<String> = Vec::new();
            root_field_uses(
                &lim_body,
                root,
                field,
                &bloc,
                None,
                &reader,
                &obj_reader,
                &ro_method,
                &mut Vec::new(),
                &mut bad,
            );
            if let Some(m) = bad.first() {
                return fail(format!(
                    "`{root}.{field}` is changed between the constructor and the launch by {m}"
                ));
            }
        }
        if let Some(e) = struct_literal_field(&init, field) {
            self.operand(
                &e,
                sc,
                Some(&bloc),
                &HashMap::new(),
                comp,
                cx,
                depth + 1,
                out,
            )?;
            return Ok(true);
        }
        let n = init.len();
        if n >= 2 && init[n - 1].is_g('(') {
            if let Some(name) = init[n - 2].id() {
                let cands = self.callee_fns(name, &self.files[sc.fi].path);
                if cands.len() == 1 {
                    let (cfi, cfk) = cands[0];
                    let cf = &self.files[cfi].fns[cfk];
                    let tail = peel(&tail_expr(&cf.body));
                    if let Some(e) = struct_literal_field(&tail, field) {
                        let call_args: Vec<Vec<Tt>> = split_top(init[n - 1].group().unwrap(), ',')
                            .into_iter()
                            .map(|a| canon_actual(&a, &idx, sc.body, Some(bloc.as_slice())))
                            .collect();
                        let formals = fn_formals(cf);
                        if formals.len() == call_args.len() {
                            let map: HashMap<String, Vec<Tt>> =
                                formals.into_iter().zip(call_args.into_iter()).collect();
                            let mut sub = Vec::new();
                            self.operand(
                                &e,
                                &Sc {
                                    fi: cfi,
                                    func: cf,
                                    body: &cf.body,
                                },
                                None,
                                &map,
                                comp,
                                cx,
                                depth + 1,
                                &mut sub,
                            )?;
                            // Only the layouts this call's own arguments select.
                            sub.retain(|v| consistent(v, &map));
                            if sub.is_empty() {
                                return fail(format!("no layout of `{name}`'s `{field}` is consistent with the arguments of this call"));
                            }
                            out.extend(sub);
                            return Ok(true);
                        }
                    }
                }
            }
        }
        // The object's origin is a local, but not a struct literal the tracer can read:
        // another object's same-named field must not stand in for it.
        unres(format!(
            "object `{root}` is not built by a struct literal or a function ending in one"
        ))
    }
}

impl World {
    /// The changed kernels a call's name argument can be, followed through parameters of
    /// the calling functions (every call of them in the file supplies it). `Err` when a
    /// name cannot be resolved: nothing is assumed about it.
    fn names_of_actual(
        &self,
        fi: usize,
        ck: usize,
        loc: &[usize],
        expr: &[Tt],
        names: &BTreeSet<String>,
        depth: u32,
    ) -> Result<BTreeSet<String>, String> {
        let cf = &self.files[fi].fns[ck];
        let csc = Sc {
            fi,
            func: cf,
            body: &cf.body,
        };
        let cidx = self.bind_index(&csc);
        let cctx = Ctx {
            file: &self.files[fi].path,
            body: &cf.body,
            params: fn_params(cf),
            idx: &cidx,
        };
        let r = self.resolve(expr, &cctx, loc, 0, true);
        if r.unres.contains("<depth>") {
            return Err("resolution was truncated by its depth budget".into());
        }
        let ks: BTreeSet<String> = r
            .lits
            .iter()
            .filter(|l| names.contains(*l))
            .cloned()
            .collect();
        if !ks.is_empty() || r.unres.is_empty() {
            return Ok(ks);
        }
        let mut out = BTreeSet::new();
        for u in &r.unres {
            let Some(p) = u.strip_prefix("param:") else {
                return Err(format!("`{u}` has no resolvable origin"));
            };
            if depth >= 4 {
                return Err(format!("parameter `{p}` is passed through too many calls"));
            }
            let formals = fn_formals(cf);
            let Some(pos) = formals.iter().position(|f| f == p) else {
                return Err(format!("`{p}` is not a parameter of `{}`", cf.name));
            };
            let srcs = self.call_srcs(fi, &cf.name, pos);
            if srcs.is_empty() {
                return Err(format!(
                    "parameter `{p}` of `{}` is not supplied by any call in this file",
                    cf.name
                ));
            }
            for (k2, l2, a2) in srcs.iter() {
                out.extend(self.names_of_actual(fi, *k2, l2, a2, names, depth + 1)?);
            }
        }
        Ok(out)
    }
}

/// The kernarg-bytes argument of a blob launch: the last one, except for
/// `launch_blob_recorded(name, grid, block, shared, kernargs, bindings)`.
fn site_blob(site: &Site) -> Option<&Vec<Tt>> {
    if site.method == "launch_blob_recorded" {
        site.args.get(4)
    } else {
        site.args.last()
    }
}

/// The rows of the tuple `for` loops an actual argument comes from. An actual that is a
/// plain loop variable of `for (a, b, ..) in [(..), (..)]` is replaced, row by row, by the
/// matching tuple component, so every other actual of the same call keeps its row. Calls
/// outside such loops yield their actuals unchanged.
fn tuple_rows(
    idx: &BindIndex,
    actuals: &[Vec<Tt>],
    loc: &[usize],
) -> Result<Vec<Vec<Vec<Tt>>>, String> {
    // (actual index, loop location, component)
    let mut uses: Vec<(usize, Loc, usize)> = Vec::new();
    let mut loops: Vec<(Loc, Vec<Tt>)> = Vec::new();
    for (a, act) in actuals.iter().enumerate() {
        let p = peel(act);
        let Some(id) = (if p.len() == 1 { p[0].id() } else { None }) else {
            continue;
        };
        let Some(b) = idx.visible(id, loc) else {
            continue;
        };
        let Some(Some(k)) = b.for_pos else { continue };
        if !loops.iter().any(|(l, _)| *l == b.loc) {
            let it = peel(&b.init);
            let lit: Vec<Tt> = if it.len() == 1 && it[0].is_g('[') {
                it[0].group().unwrap().to_vec()
            } else if it.len() == 3 && it[0].is_id("vec") && it[2].is_g('[') {
                it[2].group().unwrap().to_vec()
            } else {
                return Err(format!(
                    "loop iterable `{}` is not a literal array of tuples",
                    text(&it)
                ));
            };
            let mut rows: Vec<Tt> = Vec::new();
            for e in split_top(&lit, ',') {
                if !(e.len() == 1 && e[0].is_g('(')) {
                    return Err("a loop element is not a tuple literal".into());
                }
                rows.push(e[0].clone());
            }
            loops.push((b.loc.clone(), rows));
        }
        uses.push((a, b.loc.clone(), k));
    }
    if uses.is_empty() {
        return Ok(vec![actuals.to_vec()]);
    }
    // Product over the distinct loops.
    let mut combos: Vec<Vec<usize>> = vec![Vec::new()];
    for (_, rows) in &loops {
        let mut next = Vec::new();
        for c in &combos {
            for r in 0..rows.len() {
                let mut c2 = c.clone();
                c2.push(r);
                next.push(c2);
            }
        }
        combos = next;
        if combos.len() > 64 {
            return Err("too many tuple-loop row combinations".into());
        }
    }
    let mut out = Vec::new();
    for combo in combos {
        let mut row: Vec<Vec<Tt>> = actuals.to_vec();
        for (a, lloc, k) in &uses {
            let li = loops.iter().position(|(l, _)| l == lloc).unwrap();
            let tuple = &loops[li].1[combo[li]];
            let parts = split_top(tuple.group().unwrap(), ',');
            row[*a] = parts
                .get(*k)
                .cloned()
                .ok_or("tuple component out of range")?;
        }
        out.push(row);
    }
    Ok(out)
}

/// Is `src` (the iterator a loop leaf comes from) exactly the collection `var`, borrowed or
/// iterated without adapters?
fn is_collection_iter(src: &[Tt], var: &str) -> bool {
    match src {
        [v] => v.is_id(var),
        [a, v] if a.is_p('&') => v.is_id(var),
        [a, m, v] if a.is_p('&') && m.is_id("mut") => v.is_id(var),
        [v, d, m, g] => {
            v.is_id(var)
                && d.is_p('.')
                && matches!(m.id(), Some("iter" | "iter_mut" | "into_iter"))
                && g.is_g('(')
                && g.group().map_or(false, |x| x.is_empty())
        }
        _ => false,
    }
}

/// `for PAT in IT { body }` statements of `ts` (any depth) where every leaf of PAT that is an
/// element of `var` (through `zip` / `enumerate`, strictly) is only handed to launches in
/// `body`: (location of the statement's token list, header token range) of each.
fn approved_for_headers(
    ts: &[Tt],
    var: &str,
    reader: &dyn Fn(&str, usize) -> bool,
    loc: &mut Loc,
    out: &mut Vec<(Loc, usize, usize)>,
) {
    for i in 0..ts.len() {
        if ts[i].is_id("for") {
            if let Ok((pat, it, j)) = for_parts(ts, i) {
                if let Some(body) = ts.get(j).and_then(|t| t.group()) {
                    let mut leaves: Vec<(String, Vec<usize>)> = Vec::new();
                    if pat.len() == 1 && pat[0].is_g('(') {
                        pat_leaves(&pat[0], &mut Vec::new(), &mut leaves);
                    } else if let Some(id) = pat.iter().find_map(|t| t.id().filter(|x| *x != "mut"))
                    {
                        leaves.push((id.to_string(), Vec::new()));
                    }
                    let (mut any, mut all_ok) = (false, true);
                    for (nm, path) in leaves {
                        if nm == "_" {
                            continue;
                        }
                        let Some(src) = project_iter(&it, &path) else {
                            continue;
                        };
                        if !is_collection_iter(&peel_val(&src), var) {
                            continue;
                        }
                        any = true;
                        all_ok &= uses_are_launch_reads(body, &nm, reader);
                    }
                    if any && all_ok {
                        out.push((loc.clone(), i, j));
                    }
                }
            }
        }
        if let Tt::G(_, c, _) = &ts[i] {
            loc.push(i);
            approved_for_headers(c, var, reader, loc, out);
            loc.pop();
        }
    }
}

/// Is every use of `var` in `body` an argument handed straight to a launch (or to a helper
/// `reader` vouches for)? Allowed forms: `var`, `&mut var`, `var.as_mut_slice()`.
fn uses_are_launch_reads(body: &[Tt], var: &str, reader: &dyn Fn(&str, usize) -> bool) -> bool {
    let mut occ: Vec<Loc> = Vec::new();
    occurrences(body, var, &mut Vec::new(), &mut occ);
    // `for (.., leaf) in .. var.iter_mut() ..`: the header's use of `var` is approved when every
    // leaf that is an element of `var` is itself only handed to launches in the loop body.
    let mut approved: Vec<(Loc, usize, usize)> = Vec::new();
    approved_for_headers(body, var, reader, &mut Vec::new(), &mut approved);
    for o in occ {
        if approved.iter().any(|(base, f, j)| {
            o.len() > base.len() && o[..base.len()] == base[..] && (*f..*j).contains(&o[base.len()])
        }) {
            continue;
        }
        let Some((last, parent)) = o.split_last() else {
            return false;
        };
        let Some(ts) = tokens_at(body, parent) else {
            return false;
        };
        let i = *last;
        // The argument must sit directly in a call group.
        let Some((glast, gparent)) = parent.split_last() else {
            return false;
        };
        let Some(gts) = tokens_at(body, gparent) else {
            return false;
        };
        let callee = if *glast >= 1 && gts[*glast].is_g('(') {
            gts[*glast - 1].id()
        } else {
            None
        };
        let Some(c) = callee else { return false };
        let argidx = ts[..i].iter().filter(|t| t.is_p(',')).count();
        let ok_call = LAUNCH_NAMES.contains(&c) || reader(c, argidx);
        let mut k = i;
        while k > 0 && (ts[k - 1].is_p('&') || ts[k - 1].is_id("mut")) {
            k -= 1;
        }
        let arg_start = ts[..k]
            .iter()
            .rposition(|t| t.is_p(','))
            .map_or(0, |q| q + 1);
        let method_ok = match (ts.get(i + 1), ts.get(i + 2)) {
            (Some(d), Some(m)) if d.is_p('.') => {
                matches!(m.id(), Some("as_mut_slice" | "as_slice" | "as_ref" | "len"))
                    // `obj.field` handed on as the argument (a field, not a method call)
                    || (m.id().is_some() && ts.get(i + 3).map_or(true, |t| t.is_p(',')))
            }
            (Some(d), _) if d.is_p('.') => false,
            // `blobs[vi]`: one element of the collection, launched as that element.
            (Some(d), nx) if d.is_g('[') => nx.map_or(true, |t| t.is_p(',')),
            (Some(d), _) => d.is_p(','),
            (None, _) => true,
        };
        if !(ok_call && arg_start == k && method_ok) {
            return false;
        }
    }
    true
}

impl World {
    /// Does every definition of `fname` in file `fi` only hand its parameter `pos` to a
    /// launch (directly, or through further such helpers)? Then passing `&mut blob` to it
    /// cannot change the blob: it is an earlier launch of the same bytes. Padding it
    /// (`pad_to`), pushing to it or any other use is not.
    fn blob_reader(&self, fi: usize, fname: &str, pos: usize, depth: u32) -> bool {
        if depth > 3 {
            return false;
        }
        let defs: Vec<(usize, usize)> = self
            .callee_fns(fname, &self.files[fi].path)
            .into_iter()
            .filter(|(f, _)| *f == fi)
            .collect();
        if defs.is_empty() {
            return false;
        }
        for (cfi, cfk) in defs {
            let cf = &self.files[cfi].fns[cfk];
            let Some(p) = fn_formals(cf).get(pos).cloned() else {
                return false;
            };
            let rd = |c: &str, a: usize| self.blob_reader(cfi, c, a, depth + 1);
            if !uses_are_launch_reads(&cf.body, &p, &rd) {
                return false;
            }
        }
        true
    }
}

/// Replace every plain local whose unchanged value is a pointer expression by that
/// expression (parenthesised), so `ip + off` becomes `(x.as_ptr() as u64) + off` and keeps
/// its origin; integer locals stay as they are.
fn inline_ptr_locals(ts: &[Tt], idx: &BindIndex, body: &[Tt], l: &[usize], depth: u32) -> Vec<Tt> {
    let mut out: Vec<Tt> = Vec::with_capacity(ts.len());
    for (k, t) in ts.iter().enumerate() {
        match t {
            Tt::Id(id, _)
                if depth < 3
                    && !(k > 0 && (ts[k - 1].is_p('.') || ts[k - 1].is_p(':')))
                    && !ts
                        .get(k + 1)
                        .map_or(false, |n| n.is_g('(') || n.is_p(':') || n.is_p('!')) =>
            {
                let mut replaced = false;
                if let Some(b) = idx.visible(id, l) {
                    let init = peel(&b.init);
                    if b.for_pos.is_none()
                        && is_ptr_flow(&init)
                        && !changed_between(body, idx, id, &b.loc, l)
                    {
                        out.push(Tt::G(
                            '(',
                            inline_ptr_locals(&init, idx, body, &b.loc, depth + 1),
                            0,
                        ));
                        replaced = true;
                    }
                }
                if !replaced {
                    out.push(t.clone());
                }
            }
            Tt::G(d, c, ln) => out.push(Tt::G(*d, inline_ptr_locals(c, idx, body, l, depth), *ln)),
            other => out.push(other.clone()),
        }
    }
    out
}

/// Location of the `{ .. }` body group of the closure bound by `b` (`let name = |..| { .. }`).
fn closure_body_loc(body: &[Tt], b: &Binding) -> Option<Loc> {
    let (last, parent) = b.loc.split_last()?;
    let ts = tokens_at(body, parent)?;
    let mut pipes = 0usize;
    let mut k = *last + 1;
    while k < ts.len() && !ts[k].is_p(';') {
        if ts[k].is_p('|') {
            pipes += 1;
        }
        if pipes >= 2 && ts[k].is_g('{') {
            let mut p = parent.to_vec();
            p.push(k);
            return Some(p);
        }
        k += 1;
    }
    None
}

/// Bodies of the closures that declare a parameter named `x`.
fn closure_shadow_bodies(body: &[Tt], idx: &BindIndex, x: &str) -> Vec<Loc> {
    let mut out = Vec::new();
    for bs in idx.map.values() {
        for b in bs {
            if closure_params(&b.init).map_or(false, |ps| ps.iter().any(|p| p == x)) {
                if let Some(l) = closure_body_loc(body, b) {
                    out.push(l);
                }
            }
        }
    }
    out
}

/// If location `l` lies inside the body of a closure that declares a parameter `x`: that
/// closure's name and the parameter's position.
fn closure_param_origin(
    body: &[Tt],
    idx: &BindIndex,
    x: &str,
    l: &[usize],
) -> Option<(String, usize)> {
    let mut names: Vec<&String> = idx.map.keys().collect();
    names.sort();
    for cname in names {
        for b in &idx.map[cname] {
            let Some(ps) = closure_params(&b.init) else {
                continue;
            };
            let Some(pos) = ps.iter().position(|p| p == x) else {
                continue;
            };
            let Some(bl) = closure_body_loc(body, b) else {
                continue;
            };
            if l.len() > bl.len() && l[..bl.len()] == bl[..] {
                return Some((cname.clone(), pos));
            }
        }
    }
    None
}

impl World {
    /// Like `blob_reader`, for a callee that is a closure bound in the current function
    /// (`let bench = |gpu, name, bytes: &mut [u8], n| { .. launch(.., bytes) .. };`).
    fn blob_reader_scope(
        &self,
        sc: &Sc,
        idx: &BindIndex,
        name: &str,
        pos: usize,
        at: &[usize],
        depth: u32,
    ) -> bool {
        if depth > 3 {
            return false;
        }
        // The binding of `name` that is in scope AT the call (a later or nested rebinding of the
        // same name is a different closure).
        if let Some(b) = idx.visible(name, at) {
            let Some(ps) = closure_params(&b.init) else {
                return false;
            };
            let (Some(param), Some(body)) = (ps.get(pos), closure_body(&b.init)) else {
                return false;
            };
            let def = b.loc.clone();
            let rd = |c: &str, a: usize| self.blob_reader_scope(sc, idx, c, a, &def, depth + 1);
            return uses_are_launch_reads(&body, param, &rd);
        }
        self.blob_reader(sc.fi, name, pos, 0)
    }
}

/// Uses of `root` (an object holding the blob in `field`) after its binding at `bloc`:
/// the field may be read, shared-borrowed, cloned, indexed for reading, or `&mut`-borrowed
/// straight into a launch; any other use of the field (mutating method, assignment, index
/// write, alias, move) or of the whole object (`&mut root` elsewhere, assignment, a
/// non-read-only method, a move) is reported. Other fields are unconstrained.
#[allow(clippy::too_many_arguments)]
fn root_field_uses(
    ts: &[Tt],
    root: &str,
    field: &str,
    bloc: &[usize],
    callee: Option<&str>,
    reader: &dyn Fn(&str, usize, &[usize]) -> bool,
    obj_reader: &dyn Fn(&str, usize) -> bool,
    ro_method: &dyn Fn(&str) -> bool,
    loc: &mut Loc,
    bad: &mut Vec<String>,
) {
    let n = ts.len();
    let plen = bloc.len().saturating_sub(1);
    for i in 0..n {
        loc.push(i);
        let relevant = loc.len() > plen && loc[..plen] == bloc[..plen] && loc.as_slice() > bloc;
        let decl = (i >= 1 && ts[i - 1].is_id("let"))
            || (i >= 2 && ts[i - 1].is_id("mut") && ts[i - 2].is_id("let"));
        let member = i >= 1 && ts[i - 1].is_p('.');
        let labelled = ts.get(i + 1).map_or(false, |t| t.is_p(':'))
            && !ts.get(i + 2).map_or(false, |t| t.is_p(':'));
        if ts[i].is_id(root) && relevant && !decl && !member && !labelled {
            let mut_ref = i >= 2 && ts[i - 1].is_id("mut") && ts[i - 2].is_p('&');
            let shared = i >= 1 && ts[i - 1].is_p('&');
            let in_read = callee.map_or(false, |c| {
                LAUNCH_NAMES.contains(&c)
                    || reader(
                        c,
                        ts[..i].iter().filter(|t| t.is_p(',')).count(),
                        loc.as_slice(),
                    )
            });
            // The whole object handed to a function in this file that provably never mutates,
            // aliases or moves the field (every use of its parameter is checked the same way).
            let in_obj = callee.map_or(false, |c| {
                obj_reader(c, ts[..i].iter().filter(|t| t.is_p(',')).count())
            });
            if ts.get(i + 1).map_or(false, |t| t.is_p('.')) {
                let member_name = ts.get(i + 2).and_then(|t| t.id());
                let after = ts.get(i + 3);
                let assigned = |k: usize| {
                    ts.get(k).map_or(false, |t| t.is_p('='))
                        && !ts.get(k + 1).map_or(false, |t| t.is_p('='))
                        || ts.get(k).map_or(
                            false,
                            |t| matches!(t, Tt::P(c, _) if "+-*/|&^%".contains(*c)),
                        ) && ts.get(k + 1).map_or(false, |t| t.is_p('='))
                };
                match member_name {
                    Some(f) if f == field => {
                        if after.map_or(false, |t| t.is_p('.')) {
                            let m = ts.get(i + 4).and_then(|t| t.id()).unwrap_or("");
                            // Only methods that cannot hand out a `&mut` view of the elements
                            // (`as_mut_slice`, `iter_mut`, `drain`, ..).
                            let reads = matches!(
                                m,
                                "len"
                                    | "is_empty"
                                    | "clone"
                                    | "to_vec"
                                    | "iter"
                                    | "as_slice"
                                    | "as_ptr"
                                    | "capacity"
                                    | "first"
                                    | "last"
                                    | "get"
                                    | "contains"
                            );
                            if !(reads && ts.get(i + 5).map_or(false, |t| t.is_g('('))) {
                                bad.push(format!("`.{m}()` on the field"));
                            }
                        } else if after.map_or(false, |t| t.is_g('[')) {
                            if assigned(i + 4) {
                                bad.push("an index write into the field".to_string());
                            } else if mut_ref && !in_read {
                                bad.push(
                                    "a `&mut` borrow of an element outside a launch".to_string(),
                                );
                            }
                        } else if assigned(i + 3) {
                            bad.push("an assignment to the field".to_string());
                        } else if mut_ref && !in_read {
                            bad.push("a `&mut` alias of the field outside a launch".to_string());
                        } else if !(shared || in_read) {
                            bad.push(
                                "a move or copy of the field the tracer cannot follow".to_string(),
                            );
                        }
                    }
                    Some(_) => {
                        // Another field is free; a method call on the object is not, unless it only reads.
                        if after.map_or(false, |t| t.is_g('(')) {
                            let m = member_name.unwrap_or("");
                            if !ro_method(m) {
                                bad.push(format!("the method `{m}()` on the whole object"));
                            }
                        }
                    }
                    None => {}
                }
            } else if assigned_root(ts, i) {
                bad.push("an assignment to the whole object".to_string());
            } else if mut_ref && !in_obj {
                bad.push("`&mut` of the whole object passed to something that is not proven to leave the field alone".to_string());
            } else if !shared && !mut_ref && !in_obj {
                bad.push("a move of the whole object".to_string());
            }
        }
        if let Tt::G(_, c, _) = &ts[i] {
            let call = if ts[i].is_g('(') && i >= 1 {
                ts[i - 1].id()
            } else {
                None
            };
            root_field_uses(
                c, root, field, bloc, call, reader, obj_reader, ro_method, loc, bad,
            );
        }
        loc.pop();
    }
}

fn assigned_root(ts: &[Tt], i: usize) -> bool {
    ts.get(i + 1).map_or(false, |t| t.is_p('=')) && !ts.get(i + 2).map_or(false, |t| t.is_p('='))
}

impl World {
    /// The type a receiver expression evaluates to, when it can be read off the source:
    /// `Type { .. }`, `Type::ctor(..)`, `f(..)` for a same-file `fn f(..) -> Type`, `self`,
    /// or a local whose annotation or initialiser names one of those.
    fn recv_type(
        &self,
        recv: &[Tt],
        sc: &Sc,
        limit: Option<&[usize]>,
        depth: u32,
    ) -> Option<String> {
        if depth > 3 {
            return None;
        }
        let r = peel(recv);
        let n = r.len();
        let self_ty = || sc.func.owner.clone();
        let named = |q: &str| {
            if q == "Self" {
                self_ty()
            } else {
                Some(q.to_string())
            }
        };
        if n == 1 {
            let id = r[0].id()?;
            if id == "self" {
                return self_ty();
            }
            let idx = self.bind_index(sc);
            let b = match limit {
                Some(l) => idx.visible(id, l),
                None => idx.map.get(id).and_then(|v| v.last()),
            }?;
            if let Some(t) =
                b.ty.iter()
                    .find_map(|t| t.id().filter(|s| *s != "mut" && *s != "dyn"))
            {
                return named(t);
            }
            return self.recv_type(&b.init, sc, Some(b.loc.as_slice()), depth + 1);
        }
        if n >= 2 && r[n - 1].is_g('{') {
            let q = r[n - 2].id()?;
            return q
                .chars()
                .next()
                .filter(|c| c.is_uppercase())
                .and_then(|_| named(q));
        }
        if n >= 2 && r[n - 1].is_g('(') {
            let f = r[n - 2].id()?;
            if n >= 5 && r[n - 3].is_p(':') && r[n - 4].is_p(':') {
                let q = r[n - 5].id()?;
                if q.chars().next().map_or(false, |c| c.is_uppercase()) {
                    return named(q);
                }
            }
            if n == 2 {
                let fs = self.callee_fns(f, &self.files[sc.fi].path);
                if fs.len() == 1 {
                    let (fi, fk) = fs[0];
                    let cf = &self.files[fi].fns[fk];
                    let arrow = cf.sig.iter().position(|t| t.is_p('-'))?;
                    let ty = cf.sig[arrow..].iter().find_map(|t| t.id())?;
                    return if ty == "Self" {
                        cf.owner.clone()
                    } else {
                        Some(ty.to_string())
                    };
                }
            }
        }
        None
    }
}

impl World {
    /// Definitions of the free function `name` that `file` imports with a `use` item: a scanned
    /// file next to it whose module name appears in an item that names `name` or globs
    /// (`use super::common::*;`).
    fn imported_fns(&self, name: &str, file: &str) -> Vec<(usize, usize)> {
        let Some(&fi) = self.file_index.get(file) else {
            return Vec::new();
        };
        let dir = file.rsplit_once('/').map_or("", |(d, _)| d);
        let mut out: Vec<(usize, usize)> = Vec::new();
        for u in &self.files[fi].uses {
            if !(u.iter().any(|x| x == name) || u.iter().any(|x| x == "*")) {
                continue;
            }
            for (dfi, dfk) in self.fn_index.get(name).into_iter().flatten().copied() {
                let df = &self.files[dfi];
                if dfi == fi || df.fns[dfk].owner.is_some() {
                    continue;
                }
                let (ddir, stem) = match df.path.rsplit_once('/') {
                    Some((d, f)) => (d, f.strip_suffix(".rs").unwrap_or(f)),
                    None => ("", df.path.as_str()),
                };
                if ddir == dir && u.iter().any(|x| x == stem) && !out.contains(&(dfi, dfk)) {
                    out.push((dfi, dfk));
                }
            }
        }
        out
    }
}

impl World {
    /// Is `m` a method of this file that only borrows its receiver (`&self`)?
    fn ro_method_in(&self, fi: usize, m: &str) -> bool {
        let defs = self.callee_fns(m, &self.files[fi].path);
        !defs.is_empty()
            && defs.iter().all(|(dfi, dfk)| {
                let sig = &self.files[*dfi].fns[*dfk].sig;
                sig.iter().find_map(|t| t.group()).map_or(false, |g| {
                    g.first().map_or(false, |t| t.is_p('&'))
                        && g.get(1).map_or(false, |t| t.is_id("self"))
                })
            })
    }

    /// Does every definition of `fname` in file `fi` leave the field `field` of its parameter
    /// `pos` alone: no mutation, alias, move or escape of that field or of the whole object
    /// (the other fields are free)? Then passing `&mut obj` to it cannot change the field.
    fn object_field_reader(
        &self,
        fi: usize,
        fname: &str,
        pos: usize,
        field: &str,
        depth: u32,
    ) -> bool {
        if depth > 3 {
            return false;
        }
        let defs: Vec<(usize, usize)> = self
            .callee_fns(fname, &self.files[fi].path)
            .into_iter()
            .filter(|(f, _)| *f == fi)
            .collect();
        if defs.is_empty() {
            return false;
        }
        for (cfi, cfk) in defs {
            let cf = &self.files[cfi].fns[cfk];
            let Some(p) = fn_formals(cf).get(pos).cloned() else {
                return false;
            };
            let rd = |c: &str, a: usize, _at: &[usize]| self.blob_reader(cfi, c, a, depth + 1);
            let od = |c: &str, a: usize| self.object_field_reader(cfi, c, a, field, depth + 1);
            let ro = |m: &str| self.ro_method_in(cfi, m);
            let mut bad: Vec<String> = Vec::new();
            root_field_uses(
                &cf.body,
                &p,
                field,
                &[],
                None,
                &rd,
                &od,
                &ro,
                &mut Vec::new(),
                &mut bad,
            );
            if !bad.is_empty() {
                return false;
            }
        }
        true
    }
}

// ---- exact integer values (kernarg segment sizes) --------------------------------------------

/// How deep the integer / origin proofs may nest (each hop of a value to its source).
const MAX_PROOF_DEPTH: u32 = 48;

/// A value expression with only references, parentheses and commas removed. Unlike `peel`
/// (which treats `.clone()`, `.as_ref()`, `[i]`, .. as the same blob) nothing that could run
/// user code is assumed to return its receiver unchanged.
fn peel_val(ex: &[Tt]) -> Vec<Tt> {
    let mut t: Vec<Tt> = ex.to_vec();
    loop {
        while t.last().map_or(false, |x| x.is_p(',')) {
            t.pop();
        }
        while t
            .first()
            .map_or(false, |x| x.is_p('&') || x.is_id("mut") || x.is_p('*'))
        {
            t.remove(0);
        }
        if t.len() == 1 && t[0].is_g('(') && split_top(t[0].group().unwrap(), ',').len() <= 1 {
            t = t[0].group().unwrap().to_vec();
            continue;
        }
        return t;
    }
}

/// A struct literal `Type { .. }` a value is constructed by, and where it is.
struct Lit {
    fi: usize,
    fk: usize,
    /// The struct type the literal builds.
    ty: String,
    tokens: Vec<Tt>,
}

/// The bindings of `id` that apply at `loc` (the visible one), or every binding of the name
/// when no location is given.
fn bindings_at<'i>(idx: &'i BindIndex, id: &str, loc: Option<&[usize]>) -> Vec<&'i Binding> {
    match loc {
        Some(l) => idx.visible(id, l).into_iter().collect(),
        None => idx
            .map
            .get(id)
            .map(|v| v.iter().collect())
            .unwrap_or_default(),
    }
}

impl World {
    /// Record why an integer / origin proof gave up (the first record is the root) and give up.
    fn miss<T>(&self, cause: &'static str) -> Option<T> {
        self.note(MissRec {
            cause,
            fi: u32::MAX,
            fk: u32::MAX,
            pos: u32::MAX,
            expr: None,
        });
        None
    }

    /// A miss at parameter `pos` of function `fk` of file `fi` (it is never passed).
    fn miss_param<T>(&self, cause: &'static str, sc: &Sc, pos: usize) -> Option<T> {
        self.note(MissRec {
            cause,
            fi: sc.fi as u32,
            fk: self.fn_index_of(sc).map_or(u32::MAX, |k| k as u32),
            pos: pos as u32,
            expr: None,
        });
        None
    }

    fn note(&self, r: MissRec) {
        let mut w = self.why.borrow_mut();
        if w.len() < 8
            && !w
                .iter()
                .any(|x| x.cause == r.cause && x.fi == r.fi && x.fk == r.fk && x.pos == r.pos)
        {
            w.push(r);
        }
    }

    /// A proof step that returned `None` without naming a cause: name the expression, once.
    fn miss_unless_recorded(&self, before: usize, cause: &'static str, sc: &Sc, ex: &[Tt]) {
        if self.why.borrow().len() == before {
            self.note(MissRec {
                cause,
                fi: sc.fi as u32,
                fk: self.fn_index_of(sc).map_or(u32::MAX, |k| k as u32),
                pos: u32::MAX,
                expr: Some(text(ex).into_boxed_str()),
            });
        }
    }

    /// The causes recorded since the last clear, naming file, function and parameter.
    fn why_text(&self) -> String {
        self.why
            .borrow()
            .iter()
            .map(|r| {
                let mut out = r.cause.to_string();
                if let Some(f) = self.files.get(r.fi as usize) {
                    out.push_str(&format!(" [{}", f.path));
                    if let Some(func) = f.fns.get(r.fk as usize) {
                        out.push_str(&format!("::{}", func.name));
                        if r.pos != u32::MAX {
                            let formals = fn_formals(func);
                            out.push_str(&format!(
                                " param #{} `{}`",
                                r.pos,
                                formals.get(r.pos as usize).map_or("?", |s| s.as_str())
                            ));
                        }
                    }
                    out.push(']');
                }
                if let Some(e) = &r.expr {
                    out.push_str(&format!(" at `{e}`"));
                }
                out
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    fn fn_sc(&self, fi: usize, fk: usize) -> Sc<'_> {
        let f = &self.files[fi].fns[fk];
        Sc {
            fi,
            func: f,
            body: &f.body,
        }
    }

    fn fn_index_of(&self, sc: &Sc) -> Option<usize> {
        self.files[sc.fi]
            .fns
            .iter()
            .position(|f| std::ptr::eq(f, sc.func))
    }

    /// Every field name assigned anywhere in the workspace (`x.f = ..`, `x.f[i] = ..`,
    /// `x.f += ..`): such a field has no single construction-time value.
    fn assigned_fields(&self) -> Rc<TypedFields> {
        if let Some(t) = self.typed_fields.borrow().as_ref() {
            return t.clone();
        }
        fn walk(ts: &[Tt], out: &mut TypedFields) {
            for i in 0..ts.len() {
                if ts[i].is_p('.') && ts.get(i + 1).and_then(|t| t.id()).is_some() {
                    let mut k = i + 2;
                    while ts.get(k).map_or(false, |t| t.is_g('[')) {
                        k += 1;
                    }
                    let plain = ts.get(k).map_or(false, |t| t.is_p('='))
                        && !ts.get(k + 1).map_or(false, |t| t.is_p('='));
                    let compound = ts.get(k).map_or(
                        false,
                        |t| matches!(t, Tt::P(c, _) if "+-*/|&^%".contains(*c)),
                    ) && ts.get(k + 1).map_or(false, |t| t.is_p('='));
                    if plain || compound {
                        out.assigned.insert(ts[i + 1].id().unwrap().to_string());
                    }
                }
                if let Tt::G(_, c, _) = &ts[i] {
                    walk(c, out);
                }
            }
        }
        let mut out = TypedFields::default();
        for f in &self.files {
            for func in &f.fns {
                walk(&func.body, &mut out);
            }
        }
        let rc = Rc::new(out);
        *self.typed_fields.borrow_mut() = Some(rc.clone());
        rc
    }

    /// The exact set of integers `ex` can evaluate to in the function scope `sc` (at `loc`; every
    /// binding of a name when `None`), following locals, parameters (through every call of the
    /// function in its file), struct fields of an ACTUAL traced receiver (see `origins`),
    /// closures and functions (their successful `Ok(..)`/tail values; `Err(..)` is no value).
    /// `None` = not provable.
    fn int_set(
        &self,
        ex: &[Tt],
        sc: &Sc,
        loc: Option<&[usize]>,
        depth: u32,
    ) -> Option<BTreeSet<i64>> {
        let before = self.why.borrow().len();
        let r = self.int_set_inner(ex, sc, loc, depth);
        if r.is_none() {
            self.miss_unless_recorded(
                before,
                "an integer expression the tracer does not read",
                sc,
                ex,
            );
        }
        r
    }

    fn int_set_inner(
        &self,
        ex: &[Tt],
        sc: &Sc,
        loc: Option<&[usize]>,
        depth: u32,
    ) -> Option<BTreeSet<i64>> {
        if depth > MAX_PROOF_DEPTH {
            return self.miss("integer proof nests too deeply");
        }
        let file = &self.files[sc.fi].path;
        let mut t: Vec<Tt> = peel_val(ex);
        loop {
            if t.last().map_or(false, |x| x.is_p('?')) {
                t.pop();
                continue;
            }
            if let Some(p) = t.iter().rposition(|x| x.is_id("as")) {
                if p > 0 && t.len() - p <= 2 {
                    t.truncate(p);
                    continue;
                }
            }
            break;
        }
        let n = t.len();
        if n == 0 {
            return self.miss("empty integer expression");
        }
        if let Some(v) = self.eval_const_expr_in(&t, &HashMap::new(), file) {
            return Some(BTreeSet::from([v]));
        }
        if n == 2 && t[1].is_g('(') {
            match t[0].id() {
                Some("Ok" | "Some") => {
                    return self.int_set(t[1].group().unwrap(), sc, loc, depth + 1)
                }
                Some("Err") => return Some(BTreeSet::new()),
                _ => {}
            }
        }
        if n == 1 && t[0].is_g('[') {
            let g = t[0].group().unwrap();
            if let Some(semi) = g.iter().position(|x| x.is_p(';')) {
                return self.int_set(&g[..semi], sc, loc, depth + 1);
            }
            let mut out = BTreeSet::new();
            for e in split_top(g, ',') {
                if !e.is_empty() {
                    out.extend(self.int_set(&e, sc, loc, depth + 1)?);
                }
            }
            return Some(out);
        }
        if let Some(bodies) = branch_tails(&t) {
            let mut out = BTreeSet::new();
            for b in bodies? {
                out.extend(self.int_set(&b, sc, loc, depth + 1)?);
            }
            return Some(out);
        }
        if n >= 2 && t[n - 1].is_g('[') {
            // An element of the collection: any of its elements.
            return self.int_set(&t[..n - 1], sc, loc, depth + 1);
        }
        if n >= 4
            && t[n - 1].is_g('(')
            && t[n - 3].is_p('.')
            && matches!(t[n - 2].id(), Some("max" | "min"))
        {
            let recv = self.int_set(&t[..n - 3], sc, loc, depth + 1)?;
            let arg = self.int_set(t[n - 1].group().unwrap(), sc, loc, depth + 1)?;
            let is_max = t[n - 2].is_id("max");
            let mut out = BTreeSet::new();
            for r in &recv {
                for a in &arg {
                    out.insert(if is_max { *r.max(a) } else { *r.min(a) });
                }
            }
            return Some(out);
        }
        if n == 1 {
            return self.int_ident(t[0].id()?, sc, loc, depth);
        }
        if n == 2 && t[1].is_g('(') {
            return self.int_call(t[0].id()?, sc, loc, depth);
        }
        if n >= 3
            && t.iter().enumerate().all(|(k, x)| {
                if k % 2 == 0 {
                    x.id().is_some()
                } else {
                    x.is_p('.')
                }
            })
        {
            return self.int_field(&t, sc, loc, depth);
        }
        None
    }

    fn int_ident(
        &self,
        id: &str,
        sc: &Sc,
        loc: Option<&[usize]>,
        depth: u32,
    ) -> Option<BTreeSet<i64>> {
        let idx = self.bind_index(sc);
        let bs = bindings_at(&idx, id, loc);
        if !bs.is_empty() {
            // Reassigned anywhere: not one value.
            if idx.assigns.contains_key(id) {
                return self.miss("an integer local is reassigned");
            }
            let mut out = BTreeSet::new();
            for b in bs {
                if b.for_pos.is_some() || closure_params(&b.init).is_some() {
                    return self.miss("an integer local is a loop variable or closure");
                }
                out.extend(self.int_set(&binding_init(b), sc, Some(&b.loc), depth + 1)?);
            }
            return Some(out);
        }
        let Some(pos) = fn_formals(sc.func).iter().position(|f| f == id) else {
            return self.miss("an integer name is neither a local nor a parameter");
        };
        let srcs = self.call_srcs(sc.fi, &sc.func.name, pos);
        if srcs.is_empty() {
            return self.miss_param("an integer parameter is never passed in this file", sc, pos);
        }
        let mut out = BTreeSet::new();
        for (fk, loc, actual) in srcs.iter() {
            let csc = self.fn_sc(sc.fi, *fk);
            out.extend(self.int_set(actual, &csc, Some(loc.as_slice()), depth + 1)?);
        }
        Some(out)
    }

    /// The successful values of a closure or function call: its tail, `Ok(..)` arms and
    /// branches; `Err(..)` produces none. A body with an explicit `return` is not followed.
    fn int_call(
        &self,
        name: &str,
        sc: &Sc,
        loc: Option<&[usize]>,
        depth: u32,
    ) -> Option<BTreeSet<i64>> {
        let idx = self.bind_index(sc);
        let bs = bindings_at(&idx, name, loc);
        if !bs.is_empty() {
            let mut out = BTreeSet::new();
            for b in bs {
                let Some(body) = closure_body(&b.init) else {
                    return self.miss("a called local is not a closure");
                };
                if contains_id(&body, "return") {
                    return self.miss("a closure value has an explicit return");
                }
                out.extend(self.int_set(&body, sc, Some(&b.loc), depth + 1)?);
            }
            return Some(out);
        }
        let mut cands = self.callee_fns(name, &self.files[sc.fi].path);
        if cands.is_empty() {
            cands = self.imported_fns(name, &self.files[sc.fi].path);
        }
        if cands.is_empty() {
            return self.miss("a called function has no same-file or imported definition");
        }
        let mut out = BTreeSet::new();
        for (cfi, cfk) in cands {
            let cf = &self.files[cfi].fns[cfk];
            if contains_id(&cf.body, "return") {
                return self.miss("a called function has an explicit return");
            }
            let csc = self.fn_sc(cfi, cfk);
            out.extend(self.int_set(&tail_expr(&cf.body), &csc, None, depth + 1)?);
        }
        Some(out)
    }

    /// `recv.field`: the field's value in every struct literal the receiver can have been
    /// constructed by (traced from the receiver's actual origins, never from a type name
    /// alone). A field assigned anywhere in the workspace, an unknown constructor, a
    /// parameter nobody passes, or a collection mutated in an unfollowable way is not one
    /// value.
    fn int_field(
        &self,
        t: &[Tt],
        sc: &Sc,
        loc: Option<&[usize]>,
        depth: u32,
    ) -> Option<BTreeSet<i64>> {
        let n = t.len();
        let field = t[n - 1].id()?;
        if self.assigned_fields().assigned.contains(field) {
            return self.miss("the field is assigned somewhere after construction");
        }
        let lits = self.origins(&t[..n - 2], sc, loc, 0)?;
        if lits.is_empty() {
            return self.miss("the receiver has no struct-literal origin");
        }
        let mut out = BTreeSet::new();
        for lit in lits {
            let g = lit.tokens.get(1)?.group()?;
            let part = split_top(g, ',')
                .into_iter()
                .find(|p| p.first().map_or(false, |x| x.is_id(field)))?;
            let e: Vec<Tt> = if part.len() == 1 {
                part
            } else if part.len() > 2 && part[1].is_p(':') {
                part[2..].to_vec()
            } else {
                return self.miss("a struct literal gives the field no value");
            };
            let fsc = self.fn_sc(lit.fi, lit.fk);
            out.extend(self.int_set(&e, &fsc, None, depth + 1)?);
        }
        Some(out)
    }

    /// The struct literals a value can come from: a literal itself, the successful exit of a
    /// function it is the result of, a local's initialiser, a parameter's actuals at every
    /// call, or an element of a collection (its initialiser's elements and every `push`).
    fn origins(&self, ex: &[Tt], sc: &Sc, loc: Option<&[usize]>, depth: u32) -> Option<Vec<Lit>> {
        let before = self.why.borrow().len();
        let r = self.origins_inner(ex, sc, loc, depth);
        if r.is_none() {
            self.miss_unless_recorded(
                before,
                "a value expression the origin tracer does not read",
                sc,
                ex,
            );
        }
        r
    }

    fn origins_inner(
        &self,
        ex: &[Tt],
        sc: &Sc,
        loc: Option<&[usize]>,
        depth: u32,
    ) -> Option<Vec<Lit>> {
        if depth > MAX_PROOF_DEPTH {
            return self.miss("origin proof nests too deeply");
        }
        let mut t: Vec<Tt> = peel_val(ex);
        let mut cloned = false;
        loop {
            if t.last().map_or(false, |x| x.is_p('?')) {
                t.pop();
                continue;
            }
            let n = t.len();
            if n >= 3
                && t[n - 1].is_g('(')
                && t[n - 1].group().unwrap().is_empty()
                && t[n - 2].is_id("clone")
                && t[n - 3].is_p('.')
            {
                t.truncate(n - 3);
                cloned = true;
                continue;
            }
            break;
        }
        if cloned {
            // `x.clone()` is `x` only for a type whose `Clone` is the derived field-wise copy.
            let lits = self.origins(&t, sc, loc, depth + 1)?;
            return self.clone_is_identity(&lits).then_some(lits);
        }
        let n = t.len();
        if n == 0 {
            return self.miss("empty origin expression");
        }
        if n == 2 && t[1].is_g('(') && matches!(t[0].id(), Some("Ok" | "Some")) {
            return self.origins(t[1].group().unwrap(), sc, loc, depth + 1);
        }
        if n == 2 && t[1].is_g('{') {
            let q = t[0].id()?;
            if !(q == "Self" || q.chars().next().map_or(false, |c| c.is_uppercase())) {
                return self.miss("a struct literal names a non-type");
            }
            // Struct update syntax takes fields from another value: not a closed literal.
            if t[1]
                .group()
                .unwrap()
                .windows(2)
                .any(|w| w[0].is_p('.') && w[1].is_p('.'))
            {
                return self.miss("a struct literal uses `..base`");
            }
            let ty = if q == "Self" {
                sc.func.owner.clone()?
            } else {
                q.to_string()
            };
            return Some(vec![Lit {
                fi: sc.fi,
                fk: self.fn_index_of(sc)?,
                ty,
                tokens: t,
            }]);
        }
        if let Some(bodies) = branch_tails(&t) {
            let mut out = Vec::new();
            for b in bodies? {
                out.extend(self.origins(&b, sc, loc, depth + 1)?);
            }
            return Some(out);
        }
        if n >= 2 && t[n - 1].is_g('[') {
            return self.elements_origins(&t[..n - 1], sc, loc, depth + 1);
        }
        if n == 1 {
            let id = t[0].id()?;
            let idx = self.bind_index(sc);
            let bs = bindings_at(&idx, id, loc);
            if !bs.is_empty() {
                if idx.assigns.contains_key(id) {
                    return self.miss("the receiver local is reassigned");
                }
                let mut out = Vec::new();
                for b in bs {
                    if closure_params(&b.init).is_some() {
                        return self.miss("the receiver local is a closure");
                    }
                    if let Some(pos) = b.for_pos {
                        let iter = match pos {
                            Some(k) => match project_iter_unordered(&b.init, &[k]) {
                                Some(c) => c,
                                None => return self.miss(
                                    "the receiver loop pattern is not a followable zip/enumerate",
                                ),
                            },
                            None => b.init.clone(),
                        };
                        out.extend(self.elements_origins(&iter, sc, Some(&b.loc), depth + 1)?);
                    } else {
                        out.extend(self.origins(&binding_init(b), sc, Some(&b.loc), depth + 1)?);
                    }
                }
                return Some(out);
            }
            let Some(pos) = fn_formals(sc.func).iter().position(|f| f == id) else {
                return self.miss("the receiver is neither a local nor a parameter");
            };
            let srcs = self.call_srcs(sc.fi, &sc.func.name, pos);
            if srcs.is_empty() {
                return self.miss_param(
                    "the receiver parameter is never passed in this file",
                    sc,
                    pos,
                );
            }
            let mut out = Vec::new();
            for (fk, l, actual) in srcs.iter() {
                let csc = self.fn_sc(sc.fi, *fk);
                out.extend(self.origins(actual, &csc, Some(l.as_slice()), depth + 1)?);
            }
            return Some(out);
        }
        if n >= 2 && t[n - 1].is_g('(') {
            let name = t[n - 2].id()?;
            let file = &self.files[sc.fi].path;
            let cands: Vec<(usize, usize)> = if n >= 3 && t[n - 3].is_p('.') {
                return self.miss("the receiver is the result of a method call");
            } else if n >= 5 && t[n - 3].is_p(':') && t[n - 4].is_p(':') {
                let q = t[n - 5].id()?;
                let upper = q == "Self" || q.chars().next().map_or(false, |c| c.is_uppercase());
                let qty = if q == "Self" {
                    sc.func.owner.clone()?
                } else {
                    q.to_string()
                };
                self.fn_index
                    .get(name)?
                    .iter()
                    .copied()
                    .filter(|(fi, fk)| {
                        let f = &self.files[*fi];
                        if upper {
                            f.fns[*fk].owner.as_deref() == Some(qty.as_str())
                        } else {
                            // `module::f(..)`: a free fn of the scanned file named after the module.
                            f.fns[*fk].owner.is_none()
                                && f.path
                                    .rsplit('/')
                                    .next()
                                    .map(|s| s.strip_suffix(".rs").unwrap_or(s))
                                    == Some(q)
                        }
                    })
                    .collect()
            } else {
                let mut c = self.callee_fns(name, file);
                if c.is_empty() {
                    c = self.imported_fns(name, file);
                }
                c
            };
            if cands.is_empty() {
                return self.miss("the receiver's constructor function is not in a scanned file");
            }
            let mut out = Vec::new();
            for (cfi, cfk) in cands {
                let cf = &self.files[cfi].fns[cfk];
                if contains_id(&cf.body, "return") {
                    return self.miss("the receiver's constructor has an explicit return");
                }
                let csc = self.fn_sc(cfi, cfk);
                out.extend(self.origins(&tail_expr(&cf.body), &csc, None, depth + 1)?);
            }
            return Some(out);
        }
        None
    }

    /// Is cloning a value built by each of these literals the same value? Only for a struct
    /// whose `Clone` is derived: no hand-written `clone` method of that type exists and its
    /// definition carries `derive(.. Clone ..)`.
    fn clone_is_identity(&self, lits: &[Lit]) -> bool {
        lits.iter().all(|l| {
            let custom = self.fn_index.get("clone").map_or(false, |v| {
                v.iter().any(|(fi, fk)| {
                    self.files[*fi].fns[*fk].owner.as_deref() == Some(l.ty.as_str())
                })
            });
            let derived = self.files.iter().any(|f| {
                f.structs.iter().any(|(name, attrs)| {
                    name == &l.ty
                        && attrs.iter().any(|a| {
                            a.contains("derive")
                                && a.split(|c: char| !c.is_alphanumeric() && c != '_')
                                    .any(|w| w == "Clone")
                        })
                })
            });
            !custom && derived
        })
    }

    /// The literals the ELEMENTS of a collection expression can be constructed by.
    fn elements_origins(
        &self,
        coll: &[Tt],
        sc: &Sc,
        loc: Option<&[usize]>,
        depth: u32,
    ) -> Option<Vec<Lit>> {
        let before = self.why.borrow().len();
        let r = self.elements_origins_inner(coll, sc, loc, depth);
        if r.is_none() {
            self.miss_unless_recorded(
                before,
                "a collection expression the origin tracer does not read",
                sc,
                coll,
            );
        }
        r
    }

    fn elements_origins_inner(
        &self,
        coll: &[Tt],
        sc: &Sc,
        loc: Option<&[usize]>,
        depth: u32,
    ) -> Option<Vec<Lit>> {
        if depth > MAX_PROOF_DEPTH {
            return self.miss("element-origin proof nests too deeply");
        }
        let mut t = peel_val(coll);
        let mut cloned = false;
        while t.len() >= 4
            && t[t.len() - 1].is_g('(')
            && t[t.len() - 3].is_p('.')
            && matches!(
                t[t.len() - 2].id(),
                Some(
                    "iter"
                        | "iter_mut"
                        | "into_iter"
                        | "cloned"
                        | "copied"
                        | "rev"
                        | "filter"
                        | "skip"
                        | "take"
                        | "step_by"
                )
            )
        {
            cloned |= matches!(t[t.len() - 2].id(), Some("cloned" | "copied"));
            let m = t.len() - 3;
            t.truncate(m);
        }
        if cloned {
            let lits = self.elements_origins(&t, sc, loc, depth + 1)?;
            return self.clone_is_identity(&lits).then_some(lits);
        }
        let n = t.len();
        if let Some(g) = literal_elements(&t) {
            let mut out = Vec::new();
            for e in g {
                out.extend(self.origins(&e, sc, loc, depth + 1)?);
            }
            return Some(out);
        }
        if n >= 5
            && t[0].is_id("Vec")
            && t[1].is_p(':')
            && t[2].is_p(':')
            && matches!(t[3].id(), Some("new" | "with_capacity" | "default"))
            && t[n - 1].is_g('(')
        {
            return Some(Vec::new());
        }
        if n != 1 {
            return self.miss("the collection is not a local, literal or Vec::new()");
        }
        let id = t[0].id()?;
        let idx = self.bind_index(sc);
        let bs = bindings_at(&idx, id, loc);
        if !bs.is_empty() {
            let mut out = Vec::new();
            for b in bs {
                if b.for_pos.is_some() || closure_params(&b.init).is_some() {
                    return self.miss("a collection element source is a loop variable or closure");
                }
                out.extend(self.elements_origins(&b.init, sc, Some(&b.loc), depth + 1)?);
                let mut pushes: Vec<(Loc, Vec<Tt>)> = Vec::new();
                let mut ok = true;
                let plen = b.loc.len().saturating_sub(1);
                collection_pushes(
                    &sc.func.body,
                    id,
                    &b.loc,
                    plen,
                    &|m| self.ro_method_in(sc.fi, m),
                    &mut Vec::new(),
                    &mut pushes,
                    &mut ok,
                );
                if !ok {
                    return self.miss(
                        "the collection is mutated or moved in a way the tracer cannot follow",
                    );
                }
                for (pl, arg) in pushes {
                    out.extend(self.origins(&arg, sc, Some(pl.as_slice()), depth + 1)?);
                }
            }
            return Some(out);
        }
        let Some(pos) = fn_formals(sc.func).iter().position(|f| f == id) else {
            return self.miss("the collection is neither a local nor a parameter");
        };
        let srcs = self.call_srcs(sc.fi, &sc.func.name, pos);
        if srcs.is_empty() {
            return self.miss_param(
                "a collection parameter is never passed in this file",
                sc,
                pos,
            );
        }
        let mut out = Vec::new();
        for (fk, l, actual) in srcs.iter() {
            let csc = self.fn_sc(sc.fi, *fk);
            out.extend(self.elements_origins(actual, &csc, Some(l.as_slice()), depth + 1)?);
        }
        Some(out)
    }
}

/// Branch values of an `if`/`else` chain, `match`, or block: `Some(Some(tails))` when `t` is
/// one, `Some(None)` when it is one the tracer cannot read, `None` when it is neither.
fn branch_tails(t: &[Tt]) -> Option<Option<Vec<Vec<Tt>>>> {
    let n = t.len();
    if n == 1 && t[0].is_g('{') {
        return Some(Some(vec![tail_expr(t[0].group().unwrap())]));
    }
    if t.first().map_or(false, |x| x.is_id("if")) {
        let mut bodies: Vec<Vec<Tt>> = Vec::new();
        let end = if_chain(t, 0, &Vec::new(), |b, _| {
            bodies.push(b.to_vec());
            Ok(())
        });
        return Some(match end {
            Ok(e) if e == n => Some(bodies.iter().map(|b| tail_expr(b)).collect()),
            _ => None,
        });
    }
    if t.first().map_or(false, |x| x.is_id("match")) && n >= 2 && t[n - 1].is_g('{') {
        return Some(match_arm_values(t[n - 1].group().unwrap()));
    }
    None
}

/// Elements of `[a, b]` / `vec![a, b]` (not the `[x; n]` form).
fn literal_elements(t: &[Tt]) -> Option<Vec<Vec<Tt>>> {
    let g = match t {
        [a] if a.is_g('[') => a.group()?,
        [v, b, a] if v.is_id("vec") && b.is_p('!') && a.is_g('[') => a.group()?,
        _ => return None,
    };
    if g.iter().any(|x| x.is_p(';')) {
        return None;
    }
    Some(
        split_top(g, ',')
            .into_iter()
            .filter(|e| !e.is_empty())
            .collect(),
    )
}

/// Uses of the local collection `name` after its binding `bloc`: `push(expr)` arguments are
/// collected; reads (`len`, `iter`, indexing for reading, `&name`, a `for` iterable, `&self`
/// methods of an element) are fine; anything else (a `&mut` borrow, a move, a mutating method,
/// an element assignment) sets `ok = false`.
#[allow(clippy::too_many_arguments)]
fn collection_pushes(
    ts: &[Tt],
    name: &str,
    bloc: &[usize],
    plen: usize,
    ro: &dyn Fn(&str) -> bool,
    loc: &mut Loc,
    out: &mut Vec<(Loc, Vec<Tt>)>,
    ok: &mut bool,
) {
    for i in 0..ts.len() {
        loc.push(i);
        let relevant = loc.len() > plen && loc[..plen] == bloc[..plen] && loc.as_slice() > bloc;
        let decl = (i >= 1 && ts[i - 1].is_id("let"))
            || (i >= 2 && ts[i - 1].is_id("mut") && ts[i - 2].is_id("let"));
        let member = i >= 1 && ts[i - 1].is_p('.');
        let labelled = ts.get(i + 1).map_or(false, |t| t.is_p(':'))
            && !ts.get(i + 2).map_or(false, |t| t.is_p(':'));
        if ts[i].is_id(name) && relevant && !decl && !member && !labelled {
            let mut_ref = i >= 2 && ts[i - 1].is_id("mut") && ts[i - 2].is_p('&');
            let shared = i >= 1 && ts[i - 1].is_p('&');
            let for_iter = i >= 1 && ts[i - 1].is_id("in");
            match ts.get(i + 1) {
                Some(d) if d.is_p('.') => {
                    let m = ts.get(i + 2).and_then(|t| t.id()).unwrap_or("");
                    let call = ts.get(i + 3).and_then(|t| t.group());
                    match (m, call) {
                        ("push", Some(g)) => out.push((loc.clone(), g.to_vec())),
                        (
                            "len" | "is_empty" | "iter" | "into_iter" | "get" | "first" | "last"
                            | "clone" | "contains",
                            Some(_),
                        ) => {}
                        _ => *ok = false,
                    }
                }
                Some(b) if b.is_g('[') => {
                    let after = ts.get(i + 2);
                    let assigns = after.map_or(false, |t| t.is_p('='))
                        && !ts.get(i + 3).map_or(false, |t| t.is_p('='))
                        || after.map_or(
                            false,
                            |t| matches!(t, Tt::P(c, _) if "+-*/|&^%".contains(*c)),
                        ) && ts.get(i + 3).map_or(false, |t| t.is_p('='));
                    let method = after.map_or(false, |t| t.is_p('.'))
                        && ts.get(i + 4).map_or(false, |t| t.is_g('('))
                        && !ts.get(i + 3).and_then(|t| t.id()).map_or(false, ro);
                    if assigns || method || mut_ref {
                        *ok = false;
                    }
                }
                _ => {
                    if mut_ref || !(shared || for_iter) {
                        *ok = false;
                    }
                }
            }
        }
        if let Tt::G(_, c, _) = &ts[i] {
            collection_pushes(c, name, bloc, plen, ro, loc, out, ok);
        }
        loc.pop();
    }
}

/// The value expressions of the arms of a `match` body (`pat => expr, pat => { .. }`).
fn match_arm_values(g: &[Tt]) -> Option<Vec<Vec<Tt>>> {
    let n = g.len();
    let mut out: Vec<Vec<Tt>> = Vec::new();
    let mut i = 0usize;
    while i < n {
        // pattern up to `=>`
        let mut j = i;
        while j + 1 < n && !(g[j].is_p('=') && g[j + 1].is_p('>')) {
            j += 1;
        }
        if j + 1 >= n {
            return if i >= n { Some(out) } else { None };
        }
        let start = j + 2;
        let mut k = start;
        if k < n && g[k].is_g('{') {
            out.push(vec![g[k].clone()]);
            k += 1;
        } else {
            while k < n && !g[k].is_p(',') {
                k += 1;
            }
            out.push(g[start..k].to_vec());
        }
        if k < n && g[k].is_p(',') {
            k += 1;
        }
        i = k;
    }
    Some(out)
}

/// Does `body` size an array with `name.max(..)`?
fn sizes_array(body: &[Tt], name: &str) -> bool {
    for (i, t) in body.iter().enumerate() {
        if t.is_id(name)
            && body.get(i + 1).map_or(false, |x| x.is_p('.'))
            && body.get(i + 2).map_or(false, |x| x.is_id("max"))
        {
            return true;
        }
        if let Tt::G(_, c, _) = t {
            if sizes_array(c, name) {
                return true;
            }
        }
    }
    false
}
