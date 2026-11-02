//! The argument layouts a launch site actually builds: the `void**` params
//! array, the raw `KernargBlob` push sequence, and (for probes that hand-write
//! their kernarg) the byte ranges written into a `[0u8; N]` array.

use crate::lex::*;
use crate::scan::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PushKind {
    Ptr,
    U32,
    I32,
    F32,
    U64,
    /// `launch_params_blob!` value: encoded by its Rust type through `KernargValue`.
    Val,
}

pub type Conds = Vec<(String, bool)>;

#[derive(Clone, Debug)]
pub struct Push {
    pub kind: PushKind,
    pub arg: Vec<Tt>,
    pub conds: Conds,
}

/// Byte ranges written into a zero-initialised kernarg byte array.
#[derive(Clone, Debug)]
pub struct ByteLayout {
    pub len: u32,
    pub writes: Vec<(u32, u32)>,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub assign: BTreeMap<String, bool>,
    /// Identifier each `void**` params entry points at (None = not a plain local).
    pub param_idents: Vec<Option<String>>,
    pub pushes: Vec<Push>,
    /// False for probes that only hand the runtime a blob (no params array).
    pub has_params: bool,
    pub bytes: Option<ByteLayout>,
    /// Kernels this layout belongs to (probe blob sources the inventory pins to
    /// specific kernels); `None` = correlated by the launch's name selection.
    pub only_for: Option<Vec<String>>,
}

fn push_kind(m: &str) -> Option<PushKind> {
    match m {
        "push_ptr" => Some(PushKind::Ptr),
        "push_u32" => Some(PushKind::U32),
        "push_i32" => Some(PushKind::I32),
        "push_f32" => Some(PushKind::F32),
        "push_u64" => Some(PushKind::U64),
        _ => None,
    }
}

fn cond_atom(ts: &[Tt]) -> (String, bool) {
    if ts.first().map_or(false, |t| t.is_p('!')) {
        (text(&ts[1..]), false)
    } else {
        (text(ts), true)
    }
}

/// Walk an `if / else if / else` chain starting at `ts[i]` (`if`). Every branch
/// (including `if let` pattern conditions) gets a condition atom; `body_cb`
/// receives the branch body and the accumulated conditions. Returns the index
/// after the chain.
pub fn if_chain<F: FnMut(&[Tt], &Conds) -> Result<(), String>>(
    ts: &[Tt],
    i: usize,
    conds: &Conds,
    mut body_cb: F,
) -> Result<usize, String> {
    let n = ts.len();
    let mut j = i + 1;
    let mut prior: Conds = Vec::new();
    loop {
        let mut c: Vec<Tt> = Vec::new();
        while j < n && !ts[j].is_g('{') {
            c.push(ts[j].clone());
            j += 1;
        }
        if j >= n {
            return Err("malformed if chain".into());
        }
        let body = ts[j].group().unwrap();
        j += 1;
        let (atom, pos) = cond_atom(&c);
        let mut bc = conds.clone();
        bc.extend(prior.iter().cloned());
        bc.push((atom.clone(), pos));
        body_cb(body, &bc)?;
        prior.push((atom, !pos));
        if j < n && ts[j].is_id("else") {
            j += 1;
            if j < n && ts[j].is_id("if") {
                j += 1;
                continue;
            }
            if j < n && ts[j].is_g('{') {
                let mut bc = conds.clone();
                bc.extend(prior.iter().cloned());
                body_cb(ts[j].group().unwrap(), &bc)?;
                j += 1;
            }
        }
        return Ok(j);
    }
}

/// Split `for PAT in ITER { BODY }` starting at `ts[i]` (`for`): (pattern,
/// iterable, index of the body group). The iterable may itself be an `if` chain.
pub fn for_parts(ts: &[Tt], i: usize) -> Result<(Vec<Tt>, Vec<Tt>, usize), String> {
    let n = ts.len();
    let mut j = i + 1;
    let mut pat: Vec<Tt> = Vec::new();
    while j < n && !ts[j].is_id("in") {
        pat.push(ts[j].clone());
        j += 1;
    }
    j += 1;
    let start = j;
    if j < n && ts[j].is_id("if") {
        // Skip `if cond {..} (else if cond {..})* else {..}`.
        loop {
            j += 1;
            while j < n && !ts[j].is_g('{') {
                j += 1;
            }
            j += 1; // the branch group
            if j < n && ts[j].is_id("else") {
                j += 1;
                if j < n && ts[j].is_id("if") {
                    continue;
                }
                j += 1; // the else group
            }
            break;
        }
    } else {
        while j < n && !ts[j].is_g('{') {
            j += 1;
        }
    }
    if j >= n || !ts[j].is_g('{') {
        return Err("malformed for loop".into());
    }
    Ok((pat, ts[start..j].to_vec(), j))
}

pub(crate) fn contains_id(ts: &[Tt], name: &str) -> bool {
    ts.iter().any(|t| match t {
        Tt::Id(s, _) => s == name,
        Tt::G(_, c, _) => contains_id(c, name),
        _ => false,
    })
}

fn parse_int(s: &str) -> Option<i64> {
    if let Some(h) = s.strip_prefix("0x") {
        let hex: String = h.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        return i64::from_str_radix(&hex, 16).ok();
    }
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<i64>().ok()
    }
}

// ---- constant expression evaluation (byte offsets) ------------------------------

struct Ev<'a> {
    w: &'a World,
    env: &'a HashMap<String, i64>,
    ts: &'a [Tt],
    p: usize,
    file: &'a str,
    depth: u32,
}

impl<'a> Ev<'a> {
    fn sum(&mut self) -> Option<i64> {
        let mut v = self.term()?;
        loop {
            if self.ts.get(self.p).map_or(false, |t| t.is_p('+')) {
                self.p += 1;
                v += self.term()?;
            } else if self.ts.get(self.p).map_or(false, |t| t.is_p('-')) {
                self.p += 1;
                v -= self.term()?;
            } else {
                return Some(v);
            }
        }
    }
    fn term(&mut self) -> Option<i64> {
        let mut v = self.factor()?;
        while self.ts.get(self.p).map_or(false, |t| t.is_p('*')) {
            self.p += 1;
            v *= self.factor()?;
        }
        Some(v)
    }
    fn factor(&mut self) -> Option<i64> {
        let v = match self.ts.get(self.p)? {
            Tt::N(s, _) => {
                self.p += 1;
                parse_int(s)?
            }
            Tt::Id(name, _) => {
                self.p += 1;
                match self.env.get(name) {
                    Some(v) => *v,
                    None => self.w.const_value_in(name, self.file, self.depth)?,
                }
            }
            Tt::G('(', c, _) => {
                self.p += 1;
                let mut sub = Ev {
                    w: self.w,
                    env: self.env,
                    ts: c,
                    p: 0,
                    file: self.file,
                    depth: self.depth,
                };
                let v = sub.sum()?;
                if sub.p != c.len() {
                    return None;
                }
                v
            }
            _ => return None,
        };
        while self.ts.get(self.p).map_or(false, |t| t.is_id("as")) {
            self.p += 2;
        }
        Some(v)
    }
}

impl World {
    /// Integer value of a `const NAME: T = <integer expression>;`: the definition in
    /// `file` when it has one, otherwise a workspace-unique one.
    pub fn const_value_in(&self, name: &str, file: &str, depth: u32) -> Option<i64> {
        if depth > 4 {
            return None;
        }
        let mut init: Option<(Vec<Tt>, String)> = None;
        if let Some(f) = self.file(file) {
            if let Some(defs) = f.consts.get(name) {
                if defs.len() != 1 {
                    return None;
                }
                init = Some((defs[0].init(), f.path.clone()));
            }
        }
        if init.is_none() {
            let files = self.const_files.get(name)?;
            let mut count = 0usize;
            for fi in files {
                if let Some(defs) = self.files[*fi].consts.get(name) {
                    for d in defs {
                        count += 1;
                        init = Some((d.init(), self.files[*fi].path.clone()));
                    }
                }
            }
            if count != 1 {
                return None;
            }
        }
        let (init, path) = init?;
        let env = HashMap::new();
        let mut e = Ev {
            w: self,
            env: &env,
            ts: &init,
            p: 0,
            file: &path,
            depth: depth + 1,
        };
        let v = e.sum()?;
        if e.p != init.len() {
            return None;
        }
        Some(v)
    }

    pub fn eval_const_expr_in(
        &self,
        ts: &[Tt],
        env: &HashMap<String, i64>,
        file: &str,
    ) -> Option<i64> {
        let mut e = Ev {
            w: self,
            env,
            ts,
            p: 0,
            file,
            depth: 0,
        };
        let v = e.sum()?;
        if e.p != ts.len() {
            return None;
        }
        Some(v)
    }

    // ---- blob pushes ----------------------------------------------------------

    pub(crate) fn collect_pushes(
        &self,
        ts: &[Tt],
        bname: &str,
        ctx: &Ctx,
        conds: &Conds,
        out: &mut Vec<Push>,
    ) -> Result<(), String> {
        let n = ts.len();
        let mut i = 0usize;
        while i < n {
            let t = &ts[i];
            if t.is_id(bname)
                && i + 3 < n
                && ts[i + 1].is_p('.')
                && ts[i + 2].id().is_some()
                && ts[i + 3].is_g('(')
            {
                let m = ts[i + 2].id().unwrap();
                if let Some(kind) = push_kind(m) {
                    out.push(Push {
                        kind,
                        arg: ts[i + 3].group().unwrap().to_vec(),
                        conds: conds.clone(),
                    });
                } else if !matches!(
                    m,
                    "pad_to"
                        | "len"
                        | "as_bytes"
                        | "as_mut_slice"
                        | "into_vec"
                        | "is_empty"
                        | "clone"
                ) {
                    return Err(format!("unsupported KernargBlob method `{m}`"));
                }
                i += 4;
                continue;
            }
            if t.is_id("if") && !(i > 0 && ts[i - 1].is_id("else")) {
                let end = if_chain(ts, i, conds, |body, bc| {
                    self.collect_pushes(body, bname, ctx, bc, out)
                })?;
                i = end;
                continue;
            }
            if t.is_id("match") || t.is_id("while") || t.is_id("loop") {
                let mut j = i + 1;
                while j < n && !ts[j].is_g('{') {
                    j += 1;
                }
                if j < n {
                    let mut tmp = Vec::new();
                    self.collect_pushes(ts[j].group().unwrap(), bname, ctx, conds, &mut tmp)?;
                    if !tmp.is_empty() {
                        return Err(format!(
                            "KernargBlob pushes inside `{}` are not supported",
                            t.id().unwrap()
                        ));
                    }
                }
                i = j + 1;
                continue;
            }
            if t.is_id("for") {
                let (pat, it, j) = for_parts(ts, i)?;
                let mut added: Vec<Push> = Vec::new();
                self.collect_pushes(ts[j].group().unwrap(), bname, ctx, &Vec::new(), &mut added)?;
                if !added.is_empty() {
                    let mut branches: Vec<(Conds, Vec<Option<Vec<Tt>>>)> = Vec::new();
                    if it.first().map_or(false, |x| x.is_id("if")) {
                        if_chain(&it, 0, &Vec::new(), |body, bc| {
                            if body.len() == 1 && body[0].is_g('[') {
                                branches.push((bc.clone(), array_elems(body[0].group().unwrap())?));
                                Ok(())
                            } else {
                                Err("a conditional for-loop iterable must be array literals"
                                    .to_string())
                            }
                        })?;
                    } else {
                        branches.push((Vec::new(), self.loop_elems(&it, ctx)?));
                    }
                    let var = if pat.len() == 1 {
                        pat[0].id().map(|s| s.to_string())
                    } else {
                        None
                    };
                    // `for v in NAME`: iteration k reads NAME[k], so keep that provenance.
                    let iter_name = if it.len() == 1 {
                        it[0].id().map(|s| s.to_string())
                    } else {
                        None
                    };
                    for (bc, elems) in &branches {
                        for (k, e) in elems.iter().enumerate() {
                            for p in &added {
                                let mut q = p.clone();
                                if let Some(v) = &var {
                                    if !p.arg.is_empty()
                                        && p.arg[0].is_id(v)
                                        && (p.arg.len() == 1 || p.arg[1].is_id("as"))
                                    {
                                        if let Some(el) = e {
                                            if p.arg.len() == 1 {
                                                q.arg = el.clone();
                                            }
                                        } else if let Some(name) = &iter_name {
                                            let mut a = vec![
                                                Tt::Id(name.clone(), 0),
                                                Tt::G('[', vec![Tt::N(k.to_string(), 0)], 0),
                                            ];
                                            a.extend(p.arg[1..].iter().cloned());
                                            q.arg = a;
                                        }
                                    }
                                }
                                let mut cc = conds.clone();
                                cc.extend(bc.iter().cloned());
                                cc.extend(p.conds.iter().cloned());
                                q.conds = cc;
                                out.push(q);
                            }
                        }
                    }
                }
                i = j + 1;
                continue;
            }
            if let Tt::G(_, c, _) = t {
                self.collect_pushes(c, bname, ctx, conds, out)?;
            }
            i += 1;
        }
        Ok(())
    }
}

// ---- params array -----------------------------------------------------------------

/// The one-to-one kernelParams builder used by the few-row launchers.
/// Only immutable literal arrays, an unadapted `.iter()`, and this exact
/// reference-to-void-pointer cast are accepted. Keep each array index as an
/// origin; do not infer a layout from the closure's cast alone.
fn mapped_param_elems(
    expr: &[Tt],
    ctx: &Ctx,
    loc: &[usize],
    sloc: &[usize],
    collect: bool,
) -> Result<Vec<Vec<Tt>>, String> {
    let expected = if collect { 10 } else { 7 };
    if expr.len() != expected
        || expr[0].id().is_none()
        || !expr[1].is_p('.')
        || !expr[2].is_id("iter")
        || !expr[3].is_g('(')
        || !expr[3].group().unwrap().is_empty()
        || !expr[4].is_p('.')
        || !expr[5].is_id("map")
        || !expr[6].is_g('(')
        || (collect
            && (!expr[7].is_p('.')
                || !expr[8].is_id("collect")
                || !expr[9].is_g('(')
                || !expr[9].group().unwrap().is_empty()))
    {
        return Err(
            "params mapping must be NAME.iter().map(canonical cast), without iterator adapters"
                .into(),
        );
    }
    let closure = expr[6].group().unwrap();
    if closure.len() != 12
        || !closure[0].is_p('|')
        || closure[1].id().is_none()
        || !closure[2].is_p('|')
        || closure[3].id() != closure[1].id()
        || !closure[4].is_id("as")
        || !closure[5].is_p('*')
        || !closure[6].is_id("const")
        || !closure[7].is_id("_")
        || !closure[8].is_id("as")
        || !closure[9].is_p('*')
        || !closure[10].is_id("mut")
        || !closure[11].is_id("c_void")
    {
        return Err("params mapping closure must be |v| v as *const _ as *mut c_void".into());
    }
    let name = expr[0].id().unwrap();
    let binding = ctx
        .idx
        .visible(name, loc)
        .ok_or_else(|| format!("params mapping source `{name}` has no visible binding"))?;
    if ctx.idx.visible(name, sloc).map(|b| &b.loc) != Some(&binding.loc) {
        return Err(format!(
            "params mapping source `{name}` is shadowed before the launch"
        ));
    }
    let (start, parent) = binding
        .loc
        .split_last()
        .ok_or("params mapping source has no location")?;
    let tokens = tokens_at(ctx.body, parent).ok_or("params mapping source location is invalid")?;
    // No mutable source, alias, call result, conditional, or repeated array.
    // These arrays contain the immutable pointer/scalar values kernelParams
    // references, so later element mutation cannot be hidden by this reader.
    if !tokens.get(*start).map_or(false, |t| t.is_id("let"))
        || !tokens.get(*start + 1).map_or(false, |t| t.is_id(name))
        || !tokens.get(*start + 2).map_or(false, |t| t.is_p('='))
        || binding.init.len() != 1
        || !binding.init[0].is_g('[')
        || binding.init[0].group().unwrap().iter().any(|t| t.is_p(';'))
    {
        return Err(format!(
            "params mapping source `{name}` must be an immutable literal array"
        ));
    }
    let source = split_top(binding.init[0].group().unwrap(), ',');
    if source.is_empty() || source.iter().any(Vec::is_empty) {
        return Err("params mapping source cannot be empty".into());
    }
    (0..source.len())
        .map(|index| lex(&format!("&{name}[{index}] as *const _ as *mut c_void")))
        .collect()
}

/// Resolve an index emitted by mapped_param_elems to the actual value the
/// immutable array entry stores. The ABI is the value, not the outer void cast.
/// Mapping rejects later shadowing before producing these identifiers.
pub fn mapped_param_value(ctx: &Ctx, name: &str, sloc: &[usize]) -> Option<(Vec<Tt>, Loc)> {
    let ts = lex(name).ok()?;
    if ts.len() != 2 || ts[0].id().is_none() || !ts[1].is_g('[') {
        return None;
    }
    let index = ts[1].group()?;
    let [Tt::N(number, _)] = index else {
        return None;
    };
    let index = usize::try_from(parse_int(number)?).ok()?;
    let binding = ctx.idx.visible(ts[0].id()?, sloc)?;
    if binding.init.len() != 1
        || !binding.init[0].is_g('[')
        || binding.init[0].group()?.iter().any(|t| t.is_p(';'))
    {
        return None;
    }
    let entries = split_top(binding.init[0].group()?, ',');
    Some((entries.get(index)?.clone(), binding.loc.clone()))
}

fn collect_param_pushes(
    ctx: &Ctx,
    ts: &[Tt],
    loc: &mut Loc,
    pname: &str,
    bloc: &[usize],
    sloc: &[usize],
    safe: &[Loc],
    conds: &Conds,
    out: &mut Vec<(Vec<Tt>, Conds)>,
) -> Result<(), String> {
    let n = ts.len();
    let mut i = 0usize;
    while i < n {
        let t = &ts[i];
        let mut here = loc.clone();
        here.push(i);
        if t.is_id(pname)
            && i + 3 < n
            && ts[i + 1].is_p('.')
            && ts[i + 2].id().is_some()
            && ts[i + 3].is_g('(')
        {
            let m = ts[i + 2].id().unwrap();
            let between = here.as_slice() > bloc && here.as_slice() < sloc;
            if m == "push" {
                if between {
                    let args = split_top(ts[i + 3].group().unwrap(), ',');
                    if args.len() != 1 {
                        return Err("params.push with unexpected arity".into());
                    }
                    out.push((args[0].clone(), conds.clone()));
                }
            } else if between && m == "extend" {
                let args = split_top(ts[i + 3].group().unwrap(), ',');
                if args.len() != 1 {
                    return Err("params.extend with unexpected arity".into());
                }
                for entry in mapped_param_elems(&args[0], ctx, &here, sloc, false)? {
                    out.push((entry, conds.clone()));
                }
            } else if between && m != "len" {
                // Only push and the exact mapped append above are understood;
                // reverse, sort, swap and all other mutations remain refused.
                return Err(format!("params array `{pname}` is used through `{m}` between its declaration and the launch"));
            }
            i += 4;
            continue;
        }
        if t.is_id(pname)
            && here.as_slice() > bloc
            && here.as_slice() < sloc
            && !safe
                .iter()
                .any(|g| here.len() > g.len() && here[..g.len()] == g[..])
        {
            let declares = (i >= 1 && ts[i - 1].is_id("let"))
                || (i >= 2 && ts[i - 1].is_id("mut") && ts[i - 2].is_id("let"));
            if !declares {
                return Err(format!("params array `{pname}` is used between its declaration and the launch in a way this reader cannot follow"));
            }
        }
        if t.is_id("if") && !(i > 0 && ts[i - 1].is_id("else")) {
            // Walk the chain manually to keep branch locations.
            let mut j = i + 1;
            let mut prior: Conds = Vec::new();
            loop {
                let mut c: Vec<Tt> = Vec::new();
                while j < n && !ts[j].is_g('{') {
                    c.push(ts[j].clone());
                    j += 1;
                }
                if j >= n {
                    return Err("malformed if chain".into());
                }
                let mut bl = loc.clone();
                bl.push(j);
                let (atom, pos) = cond_atom(&c);
                let mut bc = conds.clone();
                bc.extend(prior.iter().cloned());
                bc.push((atom.clone(), pos));
                collect_param_pushes(
                    ctx,
                    ts[j].group().unwrap(),
                    &mut bl,
                    pname,
                    bloc,
                    sloc,
                    safe,
                    &bc,
                    out,
                )?;
                prior.push((atom, !pos));
                j += 1;
                if j < n && ts[j].is_id("else") {
                    j += 1;
                    if j < n && ts[j].is_id("if") {
                        j += 1;
                        continue;
                    }
                    if j < n && ts[j].is_g('{') {
                        let mut bc = conds.clone();
                        bc.extend(prior.iter().cloned());
                        let mut bl = loc.clone();
                        bl.push(j);
                        collect_param_pushes(
                            ctx,
                            ts[j].group().unwrap(),
                            &mut bl,
                            pname,
                            bloc,
                            sloc,
                            safe,
                            &bc,
                            out,
                        )?;
                        j += 1;
                    }
                }
                break;
            }
            i = j;
            continue;
        }
        if let Tt::G(_, c, _) = t {
            let mut bl = here.clone();
            collect_param_pushes(ctx, c, &mut bl, pname, bloc, sloc, safe, conds, out)?;
        }
        i += 1;
    }
    Ok(())
}

fn unwrap_parens(mut e: &[Tt]) -> &[Tt] {
    while e.len() == 1 && e[0].is_g('(') {
        e = e[0].group().unwrap();
    }
    e
}

/// `a`, `a.b.c` and an optional trailing index (`[3]` literal, `[]` for any other
/// bracket): the identity of the value an entry or push argument refers to, and the
/// number of tokens it spans.
fn path_ident(e: &[Tt]) -> Option<(String, usize)> {
    let mut s = String::new();
    let mut k = 0usize;
    loop {
        s.push_str(e.get(k)?.id()?);
        k += 1;
        if e.get(k).map_or(false, |t| t.is_p('.'))
            && e.get(k + 1).and_then(|t| t.id()).is_some()
            && !e.get(k + 2).map_or(false, |t| t.is_g('('))
        {
            s.push('.');
            k += 1;
        } else {
            break;
        }
    }
    if let Some(Tt::G('[', g, _)) = e.get(k) {
        match g.as_slice() {
            [Tt::N(n, _)] => s.push_str(&format!("[{n}]")),
            _ => s.push_str("[]"),
        }
        k += 1;
    }
    Some((s, k))
}

fn strip_amp_mut(e: &[Tt]) -> &[Tt] {
    let mut k = 0usize;
    while e.get(k).map_or(false, |t| t.is_p('&') || t.is_id("mut")) {
        k += 1;
    }
    &e[k..]
}

/// The value a `kernelParams` entry points at: `&x as *const _ as *mut c_void` (any
/// number of enclosing parentheses), `x as ..`, `&x[..]`, field paths and
/// literal-indexed elements included. `None` = an entry shape this reader does not
/// understand (the caller refuses it).
pub fn elem_ident(e: &[Tt]) -> Option<String> {
    let e = strip_amp_mut(unwrap_parens(e));
    let (id, k) = path_ident(e)?;
    if e.get(k).map_or(false, |t| t.is_id("as")) || (k == e.len() && id.ends_with("[]")) {
        Some(id)
    } else {
        None
    }
}

/// The value a blob push argument names: a path/index, optionally cast with `as`.
pub fn arg_ident(a: &[Tt]) -> Option<String> {
    let a = strip_amp_mut(unwrap_parens(a));
    let (id, k) = path_ident(a)?;
    if k == a.len() || a[k].is_id("as") {
        Some(id)
    } else {
        None
    }
}

pub(crate) fn atoms_of(lists: &[&Conds]) -> Vec<String> {
    let mut s: BTreeSet<String> = BTreeSet::new();
    for l in lists {
        for (a, _) in l.iter() {
            s.insert(a.clone());
        }
    }
    s.into_iter().collect()
}

pub(crate) fn active(conds: &Conds, assign: &BTreeMap<String, bool>) -> bool {
    conds
        .iter()
        .all(|(a, pos)| assign.get(a).copied() == Some(*pos))
}

/// Every condition-assignment variant of one params array + blob.
pub fn build_variants(
    elems: &[(Vec<Tt>, Conds)],
    pushes: &[Push],
    has_params: bool,
) -> Result<Vec<Variant>, String> {
    let ec: Vec<&Conds> = elems
        .iter()
        .map(|(_, c)| c)
        .chain(pushes.iter().map(|p| &p.conds))
        .collect();
    let atoms = atoms_of(&ec);
    if atoms.len() > 10 {
        return Err("too many independent launch conditions".into());
    }
    let mut variants = Vec::new();
    for bits in 0..(1u32 << atoms.len()) {
        let mut assign = BTreeMap::new();
        for (k, a) in atoms.iter().enumerate() {
            assign.insert(a.clone(), bits & (1 << k) != 0);
        }
        for (e, _) in elems {
            if elem_ident(e).is_none() {
                return Err(format!("unrecognised kernelParams entry `{}`", text(e)));
            }
        }
        let param_idents: Vec<Option<String>> = elems
            .iter()
            .filter(|(_, c)| active(c, &assign))
            .map(|(e, _)| elem_ident(e))
            .collect();
        let ps: Vec<Push> = pushes
            .iter()
            .filter(|p| active(&p.conds, &assign))
            .cloned()
            .collect();
        variants.push(Variant {
            assign,
            param_idents,
            pushes: ps,
            has_params,
            bytes: None,
            only_for: None,
        });
    }
    Ok(variants)
}

// ---- statements ---------------------------------------------------------------------

/// Split a block into statements; `for`/`if`/`while`/`match` statements end at their
/// closing brace. The returned tail is the block's trailing expression.
pub(crate) fn split_stmts(ts: &[Tt]) -> (Vec<Vec<Tt>>, Vec<Tt>) {
    let mut stmts: Vec<Vec<Tt>> = Vec::new();
    let mut cur: Vec<Tt> = Vec::new();
    for (i, t) in ts.iter().enumerate() {
        if t.is_p(';') {
            stmts.push(std::mem::take(&mut cur));
            continue;
        }
        cur.push(t.clone());
        if t.is_g('{')
            && cur.first().map_or(false, |f| {
                f.is_id("for") || f.is_id("if") || f.is_id("while") || f.is_id("match")
            })
            && !ts.get(i + 1).map_or(false, |x| x.is_id("else"))
        {
            stmts.push(std::mem::take(&mut cur));
        }
    }
    (stmts, cur)
}

fn parse_range(rng: &[Tt]) -> Result<(usize, usize), String> {
    let d = rng
        .iter()
        .position(|t| t.is_p('.'))
        .ok_or("params slice range without `..`")?;
    let lo = if d == 0 {
        0
    } else if let Some(Tt::N(s, _)) = rng.get(0) {
        parse_int(s).ok_or("bad slice bound")? as usize
    } else {
        return Err("non-literal params slice bound".into());
    };
    match rng.get(d + 2) {
        Some(Tt::N(s, _)) => Ok((lo, parse_int(s).ok_or("bad slice bound")? as usize)),
        _ => Err("non-literal params slice bound".into()),
    }
}

impl World {
    /// `&mut params[..used]` where `used` is chosen by an `if`/`else` block that
    /// also fills `params` (the EP tile pull launcher): one entry list per branch.
    fn sliced_params(
        &self,
        ctx: &Ctx,
        b: &Binding,
        pname: &str,
        range: &[Tt],
        sloc: &[usize],
    ) -> Result<Vec<(Vec<Tt>, Conds)>, String> {
        if !(b.init.len() == 1
            && b.init[0].is_g('[')
            && b.init[0].group().unwrap().iter().any(|t| t.is_p(';')))
        {
            return Err("a sliced params array must be initialised as `[init; N]`".into());
        }
        if !(range.len() == 3 && range[0].is_p('.') && range[1].is_p('.')) {
            return Err("params slice must be `..used`".into());
        }
        let uname = range[2]
            .id()
            .ok_or("params slice length is not an identifier")?;
        let ub = ctx
            .idx
            .visible(uname, sloc)
            .ok_or_else(|| format!("no visible binding for `{uname}`"))?;
        let init = ub.init.clone();
        // Between the declaration and the launch the array may only be touched by the
        // `used` block that fills it (anything else could reorder or overwrite it).
        {
            let plen = ub.loc.len() - 1;
            let parent = &ub.loc[..plen];
            let ub_idx = ub.loc[plen];
            let ptoks = tokens_at(ctx.body, parent).ok_or("cannot locate the `used` statement")?;
            let mut semi = ub_idx;
            while semi < ptoks.len() && !ptoks[semi].is_p(';') {
                semi += 1;
            }
            let mut occ: Vec<Loc> = Vec::new();
            occurrences(ctx.body, pname, &mut Vec::new(), &mut occ);
            let blen = b.loc.len() - 1;
            let prior = prior_launch_groups(ctx.body, sloc);
            for o in occ {
                if !(o.as_slice() > b.loc.as_slice() && o.as_slice() < sloc) {
                    continue;
                }
                let declares = o.len() == blen + 1
                    && o[..blen] == b.loc[..blen]
                    && (o[blen] == b.loc[blen] + 1 || o[blen] == b.loc[blen] + 2);
                let in_used =
                    o.len() > plen && o[..plen] == *parent && o[plen] >= ub_idx && o[plen] <= semi;
                let in_prior = prior
                    .iter()
                    .any(|g| o.len() > g.len() && o[..g.len()] == g[..]);
                if !declares && !in_used && !in_prior {
                    return Err(format!("params array `{pname}` is used between its declaration and the launch outside the block that fills it"));
                }
            }
        }
        if !init.first().map_or(false, |t| t.is_id("if")) {
            return Err(format!(
                "`{uname}` is not an if/else block that selects the params layout"
            ));
        }
        let mut branches: Vec<(Conds, Vec<Tt>)> = Vec::new();
        if_chain(&init, 0, &Vec::new(), |body, bc| {
            branches.push((bc.clone(), body.to_vec()));
            Ok(())
        })?;
        let mut out: Vec<(Vec<Tt>, Conds)> = Vec::new();
        for (bc, body) in branches {
            let (stmts, tail) = split_stmts(&body);
            let used = match tail.as_slice() {
                [Tt::N(s, _)] => parse_int(s).ok_or("bad used count")? as usize,
                _ => {
                    return Err(format!(
                        "branch of `{uname}` does not end in a literal count"
                    ))
                }
            };
            let mut pos: BTreeMap<usize, Vec<Tt>> = BTreeMap::new();
            for st in &stmts {
                if !contains_id(st, pname) {
                    continue;
                }
                self.params_stmt(st, pname, &mut pos)?;
            }
            for i in 0..used {
                let e = pos.get(&i).ok_or_else(|| {
                    format!("params entry {i} is not set on a branch that uses {used}")
                })?;
                out.push((e.clone(), bc.clone()));
            }
        }
        Ok(out)
    }

    fn params_stmt(
        &self,
        st: &[Tt],
        pname: &str,
        pos: &mut BTreeMap<usize, Vec<Tt>>,
    ) -> Result<(), String> {
        if st[0].is_id(pname) && st.len() >= 2 && st[1].is_g('[') {
            let rng = st[1].group().unwrap();
            if st.len() >= 5 && st[2].is_p('.') && st[3].is_id("copy_from_slice") && st[4].is_g('(')
            {
                let (a, b) = parse_range(rng)?;
                let args = st[4].group().unwrap();
                if !(args.len() == 2 && args[0].is_p('&') && args[1].is_g('[')) {
                    return Err("copy_from_slice argument is not `&[..]`".into());
                }
                let elems = split_top(args[1].group().unwrap(), ',');
                if elems.len() != b.saturating_sub(a) {
                    return Err("copy_from_slice length differs from the slice range".into());
                }
                for (i, e) in elems.into_iter().enumerate() {
                    pos.insert(a + i, e);
                }
                return Ok(());
            }
            if st.len() >= 4 && st[2].is_p('=') && rng.len() == 1 {
                if let Tt::N(s, _) = &rng[0] {
                    let i = parse_int(s).ok_or("bad params index")? as usize;
                    pos.insert(i, st[3..].to_vec());
                    return Ok(());
                }
            }
            return Err(format!("unsupported params statement `{}`", text(st)));
        }
        if st[0].is_id("for") {
            let (pat, it, j) = for_parts(st, 0)?;
            let shape_ok = it.len() >= 5
                && it[0].is_id(pname)
                && it[1].is_g('[')
                && it[2].is_p('.')
                && it[3].is_id("iter_mut")
                && it[4].is_g('(');
            if !shape_ok {
                return Err(format!("unsupported params loop `{}`", text(&it)));
            }
            let (a, b) = parse_range(it[1].group().unwrap())?;
            // `.zip(SRC.iter())` is the only adapter: slot a+k is paired with SRC[k] in
            // order. Any other adapter (`rev`, `skip`, `step_by`, ..) would reorder or
            // drop sources, so it is refused.
            let mut zip_src: Option<String> = None;
            if it.len() > 5 {
                let zip_ok =
                    it.len() == 8 && it[5].is_p('.') && it[6].is_id("zip") && it[7].is_g('(') && {
                        let g = it[7].group().unwrap();
                        g.len() == 4
                            && g[0].id().is_some()
                            && g[1].is_p('.')
                            && g[2].is_id("iter")
                            && g[3].is_g('(')
                    };
                if !zip_ok {
                    return Err(format!(
                        "unsupported params loop adapter `{}`",
                        text(&it[5..])
                    ));
                }
                zip_src = it[7].group().unwrap()[0].id().map(|s| s.to_string());
            }
            let (slot, item): (String, Option<String>) = if zip_src.is_some() {
                let g = pat
                    .first()
                    .and_then(|t| t.group())
                    .ok_or("params zip loop pattern is not `(slot, item)`")?;
                let parts = split_top(g, ',');
                let name = |p: &Vec<Tt>| {
                    p.iter()
                        .find_map(|t| t.id().filter(|s| *s != "mut" && *s != "ref"))
                        .map(|s| s.to_string())
                };
                if parts.len() != 2 {
                    return Err("params zip loop pattern is not `(slot, item)`".into());
                }
                (
                    name(&parts[0]).ok_or("params loop slot name")?,
                    name(&parts[1]),
                )
            } else {
                (
                    pat.iter()
                        .find_map(|t| t.id().filter(|s| *s != "mut"))
                        .ok_or("params loop slot name")?
                        .to_string(),
                    None,
                )
            };
            let body = st[j].group().unwrap();
            let mut expr: Option<Vec<Tt>> = None;
            for k in 0..body.len() {
                if body[k].is_p('*')
                    && body.get(k + 1).map_or(false, |t| t.is_id(&slot))
                    && body.get(k + 2).map_or(false, |t| t.is_p('='))
                {
                    let mut e = Vec::new();
                    let mut m = k + 3;
                    while m < body.len() && !body[m].is_p(';') {
                        e.push(body[m].clone());
                        m += 1;
                    }
                    expr = Some(e);
                    break;
                }
            }
            let expr = expr.ok_or("params loop does not assign `*slot`")?;
            for i in a..b {
                let mut e = expr.clone();
                if let (Some(src), Some(it_name)) = (&zip_src, &item) {
                    // Entry i is source element i: keep that provenance in the expression.
                    let pos_tok = Tt::G('[', vec![Tt::N((i - a).to_string(), 0)], 0);
                    let mut sub: Vec<Tt> = Vec::new();
                    for t in &e {
                        if t.is_id(it_name) {
                            sub.push(Tt::Id(src.clone(), 0));
                            sub.push(pos_tok.clone());
                        } else {
                            sub.push(t.clone());
                        }
                    }
                    e = sub;
                }
                pos.insert(i, e);
            }
            return Ok(());
        }
        Err(format!("unsupported params statement `{}`", text(st)))
    }

    /// Build every (condition-assignment) variant of the params array and blob.
    pub fn site_layouts(&self, ctx: &Ctx, site: &Site) -> Result<Vec<Variant>, String> {
        let body_loc = &site.loc;
        let pname: String;
        let mut used_range: Option<Vec<Tt>> = None;
        let mut elems: Vec<(Vec<Tt>, Conds)> = Vec::new();
        let pushes: Vec<Push>;
        let pex = &site.params_ex;
        if site.kind == 'M' {
            if pex.len() != 1 || pex[0].id().is_none() {
                return Err(format!(
                    "params argument `{}` is not a plain identifier",
                    text(pex)
                ));
            }
            pname = pex[0].id().unwrap().to_string();
        } else if site.kind == 'm' {
            if pex.len() == 3 && pex[0].is_p('&') && pex[1].is_id("mut") && pex[2].id().is_some() {
                pname = pex[2].id().unwrap().to_string();
            } else if pex.len() == 4
                && pex[0].is_p('&')
                && pex[1].is_id("mut")
                && pex[2].id().is_some()
                && pex[3].is_g('[')
            {
                pname = pex[2].id().unwrap().to_string();
                used_range = Some(pex[3].group().unwrap().to_vec());
            } else {
                return Err(format!(
                    "params argument `{}` is not `&mut ident` or `&mut ident[..n]`",
                    text(pex)
                ));
            }
        } else {
            return Err(format!(
                "launch primitive `{}` cannot be replayed",
                site.method
            ));
        }
        let b = ctx
            .idx
            .visible(&pname, body_loc)
            .ok_or_else(|| format!("no visible binding for params `{pname}`"))?;
        if let Some(range) = &used_range {
            elems = self.sliced_params(ctx, b, &pname, range, body_loc)?;
        } else {
            let init = &b.init;
            if init.len() == 3 && init[0].is_id("vec") && init[1].is_p('!') && init[2].is_g('[') {
                for e in split_top(init[2].group().unwrap(), ',') {
                    elems.push((e, Vec::new()));
                }
            } else if init.len() == 1 && init[0].is_g('[') {
                if init[0].group().unwrap().iter().any(|t| t.is_p(';')) {
                    return Err(
                        "a `[init; N]` params array is only supported with a `..used` slice".into(),
                    );
                }
                for e in split_top(init[0].group().unwrap(), ',') {
                    elems.push((e, Vec::new()));
                }
            } else if init.len() >= 5
                && init[0].is_id("Vec")
                && (init[3].is_id("with_capacity") || init[3].is_id("new"))
            {
                // Entries come from later `push` calls.
            } else {
                for entry in mapped_param_elems(init, ctx, &b.loc, body_loc, true)? {
                    elems.push((entry, Vec::new()));
                }
            }
            let mut pp: Vec<(Vec<Tt>, Conds)> = Vec::new();
            let mut l: Loc = Vec::new();
            let safe = prior_launch_groups(ctx.body, body_loc);
            collect_param_pushes(
                ctx,
                ctx.body,
                &mut l,
                &pname,
                &b.loc,
                body_loc,
                &safe,
                &Vec::new(),
                &mut pp,
            )?;
            elems.extend(pp);
        }

        if site.kind == 'M' {
            pushes = site
                .values
                .iter()
                .map(|v| Push {
                    kind: PushKind::Val,
                    arg: v.clone(),
                    conds: Vec::new(),
                })
                .collect();
        } else {
            let mut clos = site.closure_ex.clone();
            if clos.len() == 1 && clos[0].id().is_some() {
                let cb = ctx
                    .idx
                    .visible(clos[0].id().unwrap(), body_loc)
                    .ok_or("closure binding not found")?;
                clos = cb.init.clone();
            }
            if clos.first().map_or(false, |t| t.is_id("move")) {
                clos.remove(0);
            }
            if !(clos.len() >= 3 && clos[0].is_p('|') && clos[1].is_p('|')) {
                return Err(format!(
                    "blob builder `{}` is not a zero-argument closure",
                    text(&clos)
                ));
            }
            let mut cbody: Vec<Tt> = clos[2..].to_vec();
            if cbody.len() == 1 && cbody[0].is_g('{') {
                cbody = cbody[0].group().unwrap().to_vec();
            }
            let mut bname: Option<String> = None;
            for i in 0..cbody.len() {
                if cbody[i].is_id("let") {
                    let mut j = i + 1;
                    if cbody.get(j).map_or(false, |t| t.is_id("mut")) {
                        j += 1;
                    }
                    let mut k = j + 1;
                    let mut has_blob = false;
                    while k < cbody.len() && !cbody[k].is_p(';') {
                        if cbody[k].is_id("KernargBlob") {
                            has_blob = true;
                        }
                        k += 1;
                    }
                    if has_blob {
                        bname = cbody.get(j).and_then(|t| t.id()).map(|s| s.to_string());
                        break;
                    }
                }
            }
            let bname = bname.ok_or("blob builder does not create a KernargBlob")?;
            let mut v = Vec::new();
            self.collect_pushes(&cbody, &bname, ctx, &Vec::new(), &mut v)?;
            pushes = v;
        }
        build_variants(&elems, &pushes, true)
    }
}

// ---- type inference of locals ---------------------------------------------------

pub fn kind_of_type(n: &str) -> Option<(u32, char)> {
    match n {
        "i32" | "u32" => Some((4, 'i')),
        "f32" => Some((4, 'f')),
        "u64" | "usize" | "i64" | "isize" => Some((8, 'i')),
        _ => None,
    }
}

impl World {
    pub fn local_kind(
        &self,
        ctx: &Ctx,
        name: &str,
        loc: &[usize],
        depth: u32,
    ) -> Option<(u32, char)> {
        let b = ctx.idx.visible(name, loc)?;
        if b.ty.len() == 1 {
            if let Some(n) = b.ty[0].id() {
                if let Some(k) = kind_of_type(n) {
                    return Some(k);
                }
            }
        }
        let init = b.init.clone();
        let bloc = b.loc.clone();
        self.infer_kind(&init, ctx, &bloc, depth + 1)
    }

    pub fn infer_kind(
        &self,
        ex: &[Tt],
        ctx: &Ctx,
        loc: &[usize],
        depth: u32,
    ) -> Option<(u32, char)> {
        if depth > 4 || ex.is_empty() {
            return None;
        }
        if ex.len() == 1 {
            if let Some(n) = ex[0].id() {
                return if depth < 3 {
                    self.local_kind(ctx, n, loc, depth)
                } else {
                    None
                };
            }
        }
        let n = ex.len();
        if n >= 2 && ex[n - 2].is_id("as") {
            if let Some(t) = ex[n - 1].id() {
                // `ptr as u64` / `ptr as usize` only re-types a pointer's bits.
                if matches!(t, "u64" | "usize") && is_ptr_flow(&ex[..n - 2]) {
                    return Some((8, 'p'));
                }
                return kind_of_type(t);
            }
        }
        for i in 0..n {
            if ex[i].is_id("as") && ex.get(i + 1).map_or(false, |t| t.is_p('*')) {
                return Some((8, 'p'));
            }
        }
        if n >= 3
            && ex[n - 1].is_g('(')
            && matches!(
                ex[n - 2].id(),
                Some("as_ptr") | Some("ptr") | Some("as_mut_ptr")
            )
        {
            return Some((8, 'p'));
        }
        if n == 1 {
            if let Tt::N(s, _) = &ex[0] {
                for suffix in ["i32", "u32", "f32", "u64", "usize", "i64"] {
                    if s.ends_with(suffix) {
                        return kind_of_type(suffix);
                    }
                }
                if let Some((a, b)) = s.split_once('.') {
                    if !a.is_empty()
                        && a.chars().all(|c| c.is_ascii_digit())
                        && b.chars().all(|c| c.is_ascii_digit())
                    {
                        return Some((4, 'f'));
                    }
                }
                return None;
            }
        }
        let ex2: &[Tt] = if ex[n - 1].is_p('?') {
            &ex[..n - 1]
        } else {
            ex
        };
        let m = ex2.len();
        if m >= 2 && ex2[m - 1].is_g('(') {
            if let Some(callee) = ex2[m - 2].id() {
                let mut kinds: BTreeSet<(u32, char)> = BTreeSet::new();
                if let Some(list) = self.fn_index.get(callee) {
                    for (fi, fk) in list {
                        let sig = &self.files[*fi].fns[*fk].sig;
                        for q in 0..sig.len() {
                            if sig[q].is_p('-') && sig.get(q + 1).map_or(false, |t| t.is_p('>')) {
                                let ret = text(&sig[q + 2..]);
                                if ret.contains('*') {
                                    kinds.insert((8, 'p'));
                                } else {
                                    for w in
                                        ret.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                                    {
                                        if let Some(k) = kind_of_type(w) {
                                            kinds.insert(k);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if kinds.len() == 1 {
                    return kinds.into_iter().next();
                }
            }
        }
        if n >= 3 && ex[1].is_p(':') {
            if let Some(t) = ex[0].id() {
                return kind_of_type(t);
            }
        }
        None
    }
}

// ---- kernel selection: which symbol does each layout branch launch? ---------------

/// Definitions of `target` inside `ts` (`let` patterns containing it, and plain
/// `target = expr;` assignments) with the `if` branch conditions they sit under.
fn def_conds(ts: &[Tt], target: &str, conds: &Conds, out: &mut Vec<(Vec<Tt>, Conds)>) {
    let n = ts.len();
    let mut i = 0usize;
    while i < n {
        if ts[i].is_id("if") {
            if let Ok(j) = if_chain(ts, i, conds, |b, c| {
                def_conds(b, target, c, out);
                Ok(())
            }) {
                i = j;
                continue;
            }
        }
        if ts[i].is_id("let") {
            let mut j = i + 1;
            let mut pat: Vec<Tt> = Vec::new();
            while j < n && !ts[j].is_p('=') && !ts[j].is_p(';') {
                pat.push(ts[j].clone());
                j += 1;
            }
            if j < n && ts[j].is_p('=') && contains_id(&pat, target) {
                let mut rhs: Vec<Tt> = Vec::new();
                let mut k = j + 1;
                let is_if = ts
                    .get(j + 1)
                    .map_or(false, |t| t.is_id("if") || t.is_id("match"));
                while k < n && !ts[k].is_p(';') && !(ts[k].is_id("else") && !is_if) {
                    rhs.push(ts[k].clone());
                    k += 1;
                }
                out.push((rhs, conds.clone()));
            }
        } else if ts[i].is_id(target)
            && ts.get(i + 1).map_or(false, |t| t.is_p('='))
            && !ts.get(i + 2).map_or(false, |t| t.is_p('=') || t.is_p('>'))
            && !(i > 0
                && (ts[i - 1].is_p('.')
                    || ts[i - 1].is_p('!')
                    || ts[i - 1].is_p('<')
                    || ts[i - 1].is_p('>')))
        {
            let mut rhs: Vec<Tt> = Vec::new();
            let mut k = i + 2;
            while k < n && !ts[k].is_p(';') {
                rhs.push(ts[k].clone());
                k += 1;
            }
            out.push((rhs, conds.clone()));
        }
        if let Tt::G(_, c, _) = &ts[i] {
            def_conds(c, target, conds, out);
        }
        i += 1;
    }
}

/// Does a layout variant's branch assignment agree with these conditions?
/// (Atoms the variant never branched on are unconstrained.)
pub fn compatible(assign: &BTreeMap<String, bool>, conds: &Conds) -> bool {
    conds
        .iter()
        .all(|(a, v)| assign.get(a).map_or(true, |x| x == v))
}

impl World {
    /// Every changed symbol the launch's name expression can produce, with the
    /// branch conditions (`if`/`else`, `match` on boolean tuple elements) it is
    /// selected under. `None` when the selection cannot be followed.
    pub fn name_conds(
        &self,
        ctx: &Ctx,
        site: &Site,
        names: &BTreeSet<String>,
    ) -> Option<Vec<(String, Conds)>> {
        let mut out = Vec::new();
        self.expand_name(
            &site.name_ex,
            &Vec::new(),
            ctx.body,
            ctx,
            0,
            names,
            &mut out,
        )?;
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_name(
        &self,
        ts: &[Tt],
        conds: &Conds,
        scope: &[Tt],
        ctx: &Ctx,
        depth: u32,
        names: &BTreeSet<String>,
        out: &mut Vec<(String, Conds)>,
    ) -> Option<()> {
        if depth > 6 {
            return None;
        }
        let n = ts.len();
        let mut i = 0usize;
        while i < n {
            match &ts[i] {
                Tt::S(s, _) => {
                    if names.contains(s) {
                        out.push((s.clone(), conds.clone()));
                    }
                }
                Tt::Id(id, _) if id == "if" => {
                    let mut ok = true;
                    let j = if_chain(ts, i, conds, |body, c| {
                        if self
                            .expand_name(body, c, scope, ctx, depth + 1, names, out)
                            .is_none()
                        {
                            ok = false;
                        }
                        Ok(())
                    })
                    .ok()?;
                    if !ok {
                        return None;
                    }
                    i = j;
                    continue;
                }
                Tt::Id(id, _) if id == "match" => {
                    i = self.expand_match(ts, i, conds, scope, ctx, depth, names, out)?;
                    continue;
                }
                Tt::Id(id, _) => {
                    if ts.get(i + 1).map_or(false, |t| t.is_g('(')) {
                        let call_args = split_top(ts[i + 1].group().unwrap(), ',');
                        for (fi, fk) in self.callee_fns(id, ctx.file) {
                            let f = &self.files[fi].fns[fk];
                            // Bind the callee's formals to the ACTUAL arguments: a selector fed
                            // a different expression no longer matches the caller's conditions.
                            let formals = fn_formals(f);
                            if formals.len() != call_args.len() {
                                return None;
                            }
                            let map: HashMap<String, Vec<Tt>> =
                                formals.into_iter().zip(call_args.iter().cloned()).collect();
                            let body = substitute(&f.body, &map);
                            self.expand_name(&body, conds, &body, ctx, depth + 1, names, out)?;
                        }
                    } else if !(i > 0 && ts[i - 1].is_p('.')) {
                        let mut defs = Vec::new();
                        def_conds(scope, id, &Vec::new(), &mut defs);
                        if defs.is_empty() {
                            let r = self.resolve(
                                &[Tt::Id(id.clone(), 0)],
                                ctx,
                                &[usize::MAX],
                                depth + 1,
                                false,
                            );
                            for l in r.lits {
                                if names.contains(&l) {
                                    out.push((l, conds.clone()));
                                }
                            }
                        } else {
                            for (init, c) in defs {
                                let mut cc = conds.clone();
                                cc.extend(c);
                                self.expand_name(&init, &cc, scope, ctx, depth + 1, names, out)?;
                            }
                        }
                    }
                }
                Tt::G(_, c, _) => {
                    self.expand_name(c, conds, scope, ctx, depth + 1, names, out)?;
                }
                _ => {}
            }
            i += 1;
        }
        Some(())
    }

    /// `match (a, cond) { (.., true) => body, .. }`: boolean pattern elements
    /// become `text(scrutinee element) = value` conditions.
    #[allow(clippy::too_many_arguments)]
    fn expand_match(
        &self,
        ts: &[Tt],
        i: usize,
        conds: &Conds,
        scope: &[Tt],
        ctx: &Ctx,
        depth: u32,
        names: &BTreeSet<String>,
        out: &mut Vec<(String, Conds)>,
    ) -> Option<usize> {
        let n = ts.len();
        let mut j = i + 1;
        let mut scrut: Vec<Tt> = Vec::new();
        while j < n && !ts[j].is_g('{') {
            scrut.push(ts[j].clone());
            j += 1;
        }
        if j >= n {
            return None;
        }
        let arms = ts[j].group()?;
        let tuple_scrut = scrut.len() == 1 && scrut[0].is_g('(');
        let scrut_parts: Vec<Vec<Tt>> = if tuple_scrut {
            split_top(scrut[0].group()?, ',')
        } else {
            vec![scrut.clone()]
        };
        let mut k = 0usize;
        while k < arms.len() {
            let mut pat: Vec<Tt> = Vec::new();
            while k + 1 < arms.len() && !(arms[k].is_p('=') && arms[k + 1].is_p('>')) {
                pat.push(arms[k].clone());
                k += 1;
            }
            if k + 1 >= arms.len() {
                break;
            }
            k += 2;
            let mut body: Vec<Tt> = Vec::new();
            if k < arms.len() && arms[k].is_g('{') {
                body.push(arms[k].clone());
                k += 1;
                if k < arms.len() && arms[k].is_p(',') {
                    k += 1;
                }
            } else {
                while k < arms.len() && !arms[k].is_p(',') {
                    body.push(arms[k].clone());
                    k += 1;
                }
                if k < arms.len() {
                    k += 1;
                }
            }
            let pat_parts: Vec<Vec<Tt>> = if tuple_scrut && pat.len() == 1 && pat[0].is_g('(') {
                split_top(pat[0].group()?, ',')
            } else if !tuple_scrut {
                vec![pat.clone()]
            } else {
                Vec::new()
            };
            let mut cc = conds.clone();
            for (pp, sp) in pat_parts.iter().zip(scrut_parts.iter()) {
                if pp.len() == 1 {
                    if let Some(b) = pp[0].id() {
                        if b == "true" || b == "false" {
                            cc.push((text(sp), b == "true"));
                        }
                    }
                }
            }
            self.expand_name(&body, &cc, scope, ctx, depth + 1, names, out)?;
        }
        Some(j + 1)
    }
}

/// Parameter names of `f` in order, `self` excluded.
pub fn fn_formals(f: &FnRec) -> Vec<String> {
    let mut out = Vec::new();
    for t in &f.sig {
        if let Tt::G('(', c, _) = t {
            for part in split_top(c, ',') {
                if let Some(k) = part.iter().position(|x| x.is_p(':')) {
                    if let Some(n) = part[..k]
                        .iter()
                        .filter_map(|x| x.id())
                        .find(|n| *n != "mut" && *n != "ref")
                    {
                        out.push(n.to_string());
                    }
                }
            }
            break;
        }
    }
    out
}

/// Replace every use of a formal by the actual argument tokens (parenthesised when
/// it is more than one token) so conditions inside the callee mention the caller's
/// real expression. Field/path names after `.`/`::` are left alone.
pub fn substitute(ts: &[Tt], map: &HashMap<String, Vec<Tt>>) -> Vec<Tt> {
    let mut out: Vec<Tt> = Vec::with_capacity(ts.len());
    for (k, t) in ts.iter().enumerate() {
        match t {
            Tt::Id(n, _)
                if map.contains_key(n)
                    && !(k > 0 && (ts[k - 1].is_p('.') || ts[k - 1].is_p(':'))) =>
            {
                let a = &map[n];
                if a.len() == 1 {
                    out.push(a[0].clone());
                } else {
                    out.push(Tt::G('(', a.clone(), 0));
                }
            }
            Tt::G(d, c, l) => out.push(Tt::G(*d, substitute(c, map), *l)),
            other => out.push(other.clone()),
        }
    }
    out
}

/// Does this expression carry exactly ONE pointer through integer arithmetic (`p as u64`,
/// `p + offset`, `base + (i * stride) as u64`)? The pointer must be a `.as_ptr()` /
/// `.as_mut_ptr()` result or a raw-pointer cast. Two pointers (a difference) or no pointer
/// at all (a plain integer) are not a pointer.
pub fn is_ptr_flow(ex: &[Tt]) -> bool {
    fn roots(ex: &[Tt], calls: &mut usize, casts: &mut usize) {
        for k in 0..ex.len() {
            if let Tt::Id(s, _) = &ex[k] {
                if (s == "as_ptr" || s == "as_mut_ptr")
                    && k > 0
                    && ex[k - 1].is_p('.')
                    && ex.get(k + 1).map_or(false, |t| t.is_g('('))
                {
                    *calls += 1;
                }
                if s == "as" && ex.get(k + 1).map_or(false, |t| t.is_p('*')) {
                    *casts += 1;
                }
            }
            if let Tt::G(_, c, _) = &ex[k] {
                roots(c, calls, casts);
            }
        }
    }
    let (mut calls, mut casts) = (0usize, 0usize);
    roots(ex, &mut calls, &mut casts);
    // A `.as_ptr() as *const T as u64` chain is one pointer: the casts only re-type it.
    (if calls > 0 { calls } else { casts }) == 1
}

/// The tokens of the group reached by following `path` (indices into nested groups);
/// the empty path is `ts` itself.
pub fn tokens_at<'a>(ts: &'a [Tt], path: &[usize]) -> Option<&'a [Tt]> {
    let mut cur = ts;
    for i in path {
        cur = cur.get(*i)?.group()?;
    }
    Some(cur)
}

/// Locations of every `Id(name)` token under `ts`.
pub fn occurrences(ts: &[Tt], name: &str, loc: &mut Loc, out: &mut Vec<Loc>) {
    for (i, t) in ts.iter().enumerate() {
        loc.push(i);
        if t.is_id(name) {
            out.push(loc.clone());
        }
        if let Tt::G(_, c, _) = t {
            occurrences(c, name, loc, out);
        }
        loc.pop();
    }
}

/// Everything of `ts` that lexically precedes `loc` (the path of a token): earlier
/// siblings in full, and the enclosing groups truncated at the path.
pub fn cut_before(ts: &[Tt], loc: &[usize]) -> Vec<Tt> {
    let Some((first, rest)) = loc.split_first() else {
        return Vec::new();
    };
    let i = (*first).min(ts.len());
    let mut out = ts[..i].to_vec();
    if !rest.is_empty() {
        if let Some(Tt::G(d, c, l)) = ts.get(i) {
            out.push(Tt::G(*d, cut_before(c, rest), *l));
        }
    }
    out
}

/// The tokens after the one at `loc`, within its own group (later siblings only).
pub fn scope_after(ts: &[Tt], loc: &[usize]) -> Vec<Tt> {
    let Some((last, parent)) = loc.split_last() else {
        return Vec::new();
    };
    match tokens_at(ts, parent) {
        Some(g) => {
            // Skip the rest of the binding statement itself (up to and including `;`).
            let mut k = *last + 1;
            while k < g.len() && !g[k].is_p(';') {
                k += 1;
            }
            g.get(k + 1..).unwrap_or(&[]).to_vec()
        }
        None => Vec::new(),
    }
}

/// Argument groups of the `launch_maybe_blob*` / `launch_params_blob!` calls that
/// precede `sloc`: a function may reuse its params array across launches, and the
/// launch primitives only read it (`&mut` is just their signature).
pub fn prior_launch_groups(body: &[Tt], sloc: &[usize]) -> Vec<Loc> {
    let mut out = Vec::new();
    if let Ok(sites) = find_sites(body) {
        for s in sites {
            if s.loc.as_slice() < sloc && (s.kind == 'm' || s.kind == 'M') {
                let mut g = s.loc.clone();
                let last = g.len() - 1;
                g[last] += if s.kind == 'M' { 2 } else { 1 };
                out.push(g);
            }
        }
    }
    out
}

/// The actual argument at `pos` of every call of `fname` in the function bodies of file `fi`:
/// (function index, location, expression).
pub fn calls_in_fns(
    world: &World,
    fi: usize,
    fname: &str,
    pos: usize,
) -> Rc<Vec<(usize, Loc, Vec<Tt>)>> {
    world.call_srcs(fi, fname, pos)
}
