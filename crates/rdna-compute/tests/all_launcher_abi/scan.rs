//! Source scanning: functions, constants, binding resolution, launch sites and
//! the kernel-argument layouts a launch site actually builds.
//!
//! Everything is read from the real `.rs` text (no hand-copied counts): the
//! `void**` params array and the raw `KernargBlob` push sequence of every
//! launch site are extracted and replayed against a declared ABI.

use crate::layout::Variant;
use crate::lex::*;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

pub type Loc = Vec<usize>;

pub struct SrcFile {
    pub path: String,
    pub fns: Vec<FnRec>,
    pub consts: HashMap<String, Vec<ConstRec>>,
    /// How often each identifier occurs anywhere in the file (items, macros, bodies).
    pub id_counts: HashMap<String, usize>,
    /// The identifiers of each `use` item (`*` for a glob).
    pub uses: Vec<Vec<String>>,
    /// Each `struct Name` with its attributes.
    pub structs: Vec<(String, Vec<String>)>,
}

pub struct World {
    pub root: PathBuf,
    pub files: Vec<SrcFile>,
    pub file_index: HashMap<String, usize>,
    pub fn_index: HashMap<String, Vec<(usize, usize)>>,
    pub const_files: HashMap<String, Vec<usize>>,
    texts: RefCell<HashMap<(String, String), Option<String>>>,
    res_cache: RefCell<HashMap<String, Res>>,
    /// Known changed kernel symbols: lets `format!` templates resolve to real names.
    names: RefCell<BTreeSet<String>>,
    /// Blob-operand expansions finished with a complete context (see `flow.rs`).
    pub(crate) flow_memo: RefCell<HashMap<String, Vec<Variant>>>,
    /// Field/call origin indices (field or callee name -> sources), built once per file.
    pub(crate) flow_srcs: RefCell<HashMap<String, Rc<Vec<(usize, Loc, Vec<Tt>)>>>>,
    /// `BindIndex` of each function body, built once.
    pub(crate) idx_cache: RefCell<HashMap<(usize, usize), Rc<BindIndex>>>,
    /// Local bindings/assignments currently being expanded by the name resolver (cycle guard).
    bind_active: RefCell<HashSet<String>>,
    /// Per file: field name -> every expression a struct literal or assignment gives it.
    field_maps: RefCell<HashMap<usize, Rc<HashMap<String, Vec<(Option<usize>, Vec<Tt>)>>>>>,
    /// The field names assigned anywhere (see `flow.rs::assigned_fields`).
    pub(crate) typed_fields: RefCell<Option<Rc<TypedFields>>>,
    /// Dead ends of the integer / origin proofs, for failure messages.
    pub(crate) why: RefCell<Vec<MissRec>>,
}

/// One recorded dead end of an integer / origin proof: a stable cause plus where it happened
/// (file, function, parameter index; ids are `u32::MAX` when not applicable) and, for an
/// expression shape the tracer does not read, its text.
pub(crate) struct MissRec {
    pub cause: &'static str,
    pub fi: u32,
    pub fk: u32,
    pub pos: u32,
    pub expr: Option<Box<str>>,
}

/// Field names assigned after construction anywhere in the workspace.
#[derive(Default)]
pub struct TypedFields {
    pub assigned: HashSet<String>,
}

pub const KERNELS_RS: &str = "crates/rdna-compute/src/kernels.rs";

impl World {
    pub fn from_sources(root: &Path, sources: Vec<(String, String)>) -> Result<World, String> {
        let mut files = Vec::new();
        for (path, text) in sources {
            let ts = lex(&text).map_err(|e| format!("{path}: {e}"))?;
            let items = walk_items(&ts);
            let mut consts: HashMap<String, Vec<ConstRec>> = HashMap::new();
            for (n, c) in items.consts {
                consts.entry(n).or_default().push(c);
            }
            let mut id_counts: HashMap<String, usize> = HashMap::new();
            count_ids(&ts, &mut id_counts);
            files.push(SrcFile {
                path,
                fns: items.fns,
                consts,
                id_counts,
                uses: items.uses,
                structs: items.structs,
            });
        }
        let mut fn_index: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        let mut const_files: HashMap<String, Vec<usize>> = HashMap::new();
        for (fi, f) in files.iter().enumerate() {
            for (k, func) in f.fns.iter().enumerate() {
                fn_index.entry(func.name.clone()).or_default().push((fi, k));
            }
            for n in f.consts.keys() {
                const_files.entry(n.clone()).or_default().push(fi);
            }
        }
        let mut file_index: HashMap<String, usize> = HashMap::new();
        for (fi, f) in files.iter().enumerate() {
            file_index.insert(f.path.clone(), fi);
        }
        Ok(World {
            root: root.to_path_buf(),
            files,
            file_index,
            fn_index,
            const_files,
            texts: RefCell::new(HashMap::new()),
            res_cache: RefCell::new(HashMap::new()),
            names: RefCell::new(BTreeSet::new()),
            flow_memo: RefCell::new(HashMap::new()),
            flow_srcs: RefCell::new(HashMap::new()),
            idx_cache: RefCell::new(HashMap::new()),
            bind_active: RefCell::new(HashSet::new()),
            field_maps: RefCell::new(HashMap::new()),
            typed_fields: RefCell::new(None),
            why: RefCell::new(Vec::new()),
        })
    }

    pub fn load(root: &Path, rels: &[String]) -> Result<World, String> {
        let mut sources = Vec::new();
        for r in rels {
            let text = std::fs::read_to_string(root.join(r)).map_err(|e| format!("{r}: {e}"))?;
            sources.push((r.clone(), text));
        }
        World::from_sources(root, sources)
    }

    pub fn file(&self, path: &str) -> Option<&SrcFile> {
        self.file_index.get(path).map(|i| &self.files[*i])
    }

    // ---- constant text evaluation -----------------------------------------

    pub fn const_text(&self, name: &str, file: &str) -> Option<String> {
        let key = (name.to_string(), file.to_string());
        if let Some(v) = self.texts.borrow().get(&key) {
            return v.clone();
        }
        self.texts.borrow_mut().insert(key.clone(), None);
        let v = (|| {
            let f = self.file(file)?;
            let defs = f.consts.get(name)?;
            if defs.len() != 1 {
                return None;
            }
            self.eval_text(&defs[0].init(), file)
        })();
        self.texts.borrow_mut().insert(key, v.clone());
        v
    }

    pub fn eval_text(&self, ts: &[Tt], file: &str) -> Option<String> {
        let mut ts: Vec<Tt> = ts.to_vec();
        while ts.first().map_or(false, |t| t.is_p('&')) {
            ts.remove(0);
        }
        if ts.len() == 1 {
            if let Tt::S(s, _) = &ts[0] {
                return unescape(s);
            }
        }
        if ts.len() == 3 && ts[0].is_id("include_str") && ts[1].is_p('!') && ts[2].group().is_some()
        {
            let a = ts[2].group().unwrap();
            if a.len() == 1 {
                if let Tt::S(rel, _) = &a[0] {
                    let dir = Path::new(file).parent()?;
                    let p = self.root.join(dir).join(unescape(rel)?);
                    return std::fs::read_to_string(p).ok();
                }
            }
            return None;
        }
        if ts.len() == 3 && ts[0].is_id("concat") && ts[1].is_p('!') && ts[2].group().is_some() {
            let mut out = String::new();
            for part in split_top(ts[2].group().unwrap(), ',') {
                out.push_str(&self.eval_text(&part, file)?);
            }
            return Some(out);
        }
        let ids: Vec<&str> = ts.iter().filter_map(|t| t.id()).collect();
        if !ids.is_empty() && ts.iter().all(|t| t.id().is_some() || t.is_p(':')) {
            let name = ids[ids.len() - 1];
            if ids.len() >= 2 && ids[ids.len() - 2] == "kernels" {
                return self.const_text(name, KERNELS_RS);
            }
            if ids.len() == 1 {
                return self.const_text(name, file);
            }
        }
        None
    }

    // ---- resolution of names to string literals ----------------------------

    pub(crate) fn callee_fns(&self, name: &str, file: &str) -> Vec<(usize, usize)> {
        self.fn_index
            .get(name)
            .map(|l| {
                l.iter()
                    .copied()
                    .filter(|(fi, _)| {
                        self.files[*fi].path == file || self.files[*fi].path == KERNELS_RS
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Install the known changed kernel symbols (lets `format!` templates resolve).
    pub fn set_names(&self, names: BTreeSet<String>) {
        *self.names.borrow_mut() = names;
        self.res_cache.borrow_mut().clear();
        self.flow_memo.borrow_mut().clear();
    }

    /// Symbols a `format!` template can produce: literal segments are fixed, a
    /// placeholder takes the literals its identifier resolves to (any text when it
    /// does not resolve). Only known kernel symbols are produced.
    fn format_names(
        &self,
        tpl: &str,
        ctx: &Ctx,
        sloc: &[usize],
        depth: u32,
        callees: bool,
        res: &mut Res,
    ) {
        // A template none of whose literal pieces can sit in any known kernel name (a log line,
        // a `#define ..` prelude) produces no kernel name whatever its placeholders print.
        if !template_can_match(tpl, &self.names.borrow()) {
            return;
        }
        let mut segs: Vec<Seg> = Vec::new();
        let mut lit = String::new();
        let c: Vec<char> = tpl.chars().collect();
        let mut i = 0usize;
        while i < c.len() {
            if c[i] == '{' && c.get(i + 1) == Some(&'{') {
                lit.push('{');
                i += 2;
            } else if c[i] == '}' && c.get(i + 1) == Some(&'}') {
                lit.push('}');
                i += 2;
            } else if c[i] == '{' {
                let mut j = i + 1;
                let mut inner = String::new();
                while j < c.len() && c[j] != '}' {
                    inner.push(c[j]);
                    j += 1;
                }
                if !lit.is_empty() {
                    segs.push(Seg::Lit(std::mem::take(&mut lit)));
                }
                let name = inner.split(':').next().unwrap_or("").trim().to_string();
                let is_ident = !name.is_empty()
                    && !name.chars().next().unwrap().is_ascii_digit()
                    && name.chars().all(|ch| ch == '_' || ch.is_alphanumeric());
                if is_ident {
                    let r = self.resolve(&[Tt::Id(name, 0)], ctx, sloc, depth + 1, callees);
                    segs.push(Seg::Ph(if r.lits.is_empty() {
                        None
                    } else {
                        Some(r.lits)
                    }));
                } else {
                    segs.push(Seg::Ph(None));
                }
                i = j + 1;
            } else {
                lit.push(c[i]);
                i += 1;
            }
        }
        if !lit.is_empty() {
            segs.push(Seg::Lit(lit));
        }
        let anchored = segs.iter().any(|s| match s {
            Seg::Lit(l) => l.len() >= 3,
            Seg::Ph(Some(_)) => true,
            Seg::Ph(None) => false,
        });
        if !anchored {
            return;
        }
        for nm in self.names.borrow().iter() {
            if seg_match(&segs, nm) {
                res.lits.insert(nm.clone());
            }
        }
    }

    pub fn resolve(&self, ex: &[Tt], ctx: &Ctx, sloc: &[usize], depth: u32, callees: bool) -> Res {
        let mut res = Res::default();
        if depth > 8 {
            res.unres.insert("<depth>".into());
            return res;
        }
        self.resolve_go(ex, ctx, sloc, depth, callees, &mut res);
        res
    }

    fn resolve_go(
        &self,
        ts: &[Tt],
        ctx: &Ctx,
        sloc: &[usize],
        depth: u32,
        callees: bool,
        res: &mut Res,
    ) {
        // When resolving what a launch names, the value of `if c { a } else { b }` is `a` or `b`:
        // the condition (and a `match` scrutinee) only selects, it is never the name.
        let mut skip_to = 0usize;
        for k in 0..ts.len() {
            if k < skip_to {
                continue;
            }
            // (`if let PAT = x` and a `match` scrutinee also BIND names the branches use: kept.)
            if callees && ts[k].is_id("if") && !ts.get(k + 1).map_or(false, |x| x.is_id("let")) {
                if let Some(j) = (k + 1..ts.len()).find(|j| ts[*j].is_g('{')) {
                    skip_to = j;
                    continue;
                }
            }
            if callees && ts[k].is_id("match") {
                if let Some(j) = (k + 1..ts.len()).find(|j| ts[*j].is_g('{')) {
                    let arms = ts[j].group().unwrap();
                    let nonempty = || arms.split(|t| t.is_p(',')).filter(|a| !a.is_empty());
                    // Only discard selectors/patterns when they bind no values. Matches
                    // such as `Some(entry) => entry` retain the conservative origin walk.
                    if !arms.is_empty() && nonempty().all(|a| match_arm_value(a).is_some()) {
                        for arm in nonempty() {
                            self.resolve_go(
                                match_arm_value(arm).unwrap(),
                                ctx,
                                sloc,
                                depth,
                                callees,
                                res,
                            );
                        }
                        skip_to = j + 1;
                        continue;
                    }
                }
            }
            match &ts[k] {
                Tt::S(s, _) => {
                    res.lits.insert(s.clone());
                }
                Tt::G(_, c, _) => self.resolve_go(c, ctx, sloc, depth, callees, res),
                Tt::Id(name, _)
                    if name == "format"
                        && ts.get(k + 1).map_or(false, |x| x.is_p('!'))
                        && ts.get(k + 2).map_or(false, |x| x.group().is_some()) =>
                {
                    let parts = split_top(ts[k + 2].group().unwrap(), ',');
                    if let Some(first) = parts.first() {
                        if first.len() == 1 {
                            if let Tt::S(tpl, _) = &first[0] {
                                self.format_names(tpl, ctx, sloc, depth, callees, res);
                            }
                        }
                    }
                }
                Tt::Id(name, _) if !is_skip_word(name) => {
                    let prev = if k > 0 { ts.get(k - 1) } else { None };
                    let nxt = ts.get(k + 1);
                    if nxt.map_or(false, |x| x.is_p(':'))
                        && !ts.get(k + 2).map_or(false, |x| x.is_p(':'))
                    {
                        continue;
                    }
                    let nxt_call = nxt.map_or(false, |x| x.is_g('('));
                    // `root.field` when launching: the value is the field's own sources, not
                    // everything the root object (a static table row, a whole struct) holds.
                    if callees
                        && !prev.map_or(false, |x| x.is_p('.'))
                        && nxt.map_or(false, |x| x.is_p('.'))
                    {
                        if let Some(f) = ts.get(k + 2).and_then(|t| t.id()) {
                            let call = ts.get(k + 3).map_or(false, |t| t.is_g('('));
                            if !call && !self.field_literals(ctx.file, f).is_empty() {
                                continue;
                            }
                        }
                    }
                    if prev.map_or(false, |x| x.is_p('.')) {
                        if nxt_call && callees {
                            for (fi, fk) in self.callee_fns(name, ctx.file) {
                                let r = self.resolve_fn_body(fi, fk, depth + 1);
                                res.merge(&r);
                            }
                        } else if !nxt_call && callees {
                            // (Only when resolving what a LAUNCH names: a field that happens to hold a
                            // kernel name does not make every reader of it a mention.)
                            // `x.field`: the literals every struct literal / assignment of that
                            // field in this file gives it (a table of kernel names in a const).
                            let gkey = format!("fld|{}|{name}", ctx.file);
                            let cached = self.res_cache.borrow().get(&gkey).cloned();
                            if let Some(r) = cached {
                                res.merge(&r);
                            } else if self.bind_active.borrow_mut().insert(gkey.clone()) {
                                let acc = self.resolve_field(ctx.file, name);
                                self.bind_active.borrow_mut().remove(&gkey);
                                self.res_cache.borrow_mut().insert(gkey, acc.clone());
                                res.merge(&acc);
                            }
                        }
                        continue;
                    }
                    if nxt_call {
                        // `Vec::new()` / `KernargBlob::new()` are other types' associated functions,
                        // not the same-named free function or method of this file.
                        let other_type = prev.map_or(false, |x| x.is_p(':'))
                            && k >= 3
                            && ts[k - 2].is_p(':')
                            && ts[k - 3].id().map_or(false, |q| {
                                q != "Self" && q.chars().next().map_or(false, |c| c.is_uppercase())
                            });
                        if callees && !other_type {
                            for (fi, fk) in self.callee_fns(name, ctx.file) {
                                let r = self.resolve_fn_body(fi, fk, depth + 1);
                                res.merge(&r);
                            }
                        }
                        continue;
                    }
                    if prev.map_or(false, |x| x.is_p(':')) && k >= 2 && ts[k - 2].is_p(':') {
                        let q = if k >= 3 { ts[k - 3].id() } else { None };
                        if q == Some("kernels") {
                            if self
                                .file(KERNELS_RS)
                                .map_or(false, |f| f.consts.contains_key(name))
                            {
                                res.consts.insert((name.clone(), KERNELS_RS.to_string()));
                                let r = self.resolve_const(name, KERNELS_RS, depth + 1, callees);
                                res.merge(&r);
                            }
                        } else if let Some(r) = self.resolve_global_const(name, depth + 1, callees)
                        {
                            res.merge(&r);
                        }
                        continue;
                    }
                    if let Some(list) = ctx.idx.assigns.get(name) {
                        for (aloc, ainit) in list {
                            if aloc.as_slice() == sloc || mentions_name(ainit, name) {
                                continue;
                            }
                            let gkey =
                                format!("a|{}|{name}|{:?}", ctx.body.as_ptr() as usize, aloc);
                            if !self.bind_active.borrow_mut().insert(gkey.clone()) {
                                continue;
                            }
                            let r = self.resolve(ainit, ctx, aloc, depth + 1, callees);
                            self.bind_active.borrow_mut().remove(&gkey);
                            res.merge(&r);
                        }
                    }
                    // Bindings of this name in the function that are not visible at `sloc`
                    // (never the one being initialised, never one that mentions itself).
                    let fb: Vec<&Binding> = if ctx.idx.visible(name, sloc).is_none() {
                        ctx.idx
                            .map
                            .get(name)
                            .map(|bs| {
                                bs.iter()
                                    .filter(|b| {
                                        b.loc.as_slice() != sloc && !mentions_name(&b.init, name)
                                    })
                                    .collect()
                            })
                            .unwrap_or_default()
                    } else {
                        Vec::new()
                    };
                    if let Some(b) = ctx.idx.visible(name, sloc) {
                        // A `for (a, b, ..) in [(..), ..]` variable is one component of each
                        // row, not the whole iterable (its tag strings are not kernel names).
                        let projected = match b.for_pos {
                            Some(Some(k)) => {
                                self.resolve_param_tuple(ctx, &b.init, k, depth, callees)
                            }
                            _ => None,
                        };
                        let component = match (b.let_pos, callees) {
                            (Some(kc), true)
                                if projected.is_none() && project_let(&b.init, kc).is_none() =>
                            {
                                self.field_path_component(ctx.file, &b.init, kc)
                            }
                            _ => None,
                        };
                        if let Some(r) = projected.or(component) {
                            res.merge(&r);
                        } else {
                            let init = binding_init(b);
                            let bloc = b.loc.clone();
                            let r = self.resolve(&init, ctx, &bloc, depth + 1, callees);
                            res.merge(&r);
                        }
                    } else if !fb.is_empty() {
                        // Bound in this function but not visible at `sloc`: every local
                        // binding of that name, never a same-named const of another function.
                        for b in fb {
                            let gkey = format!("{}|{name}|{:?}", ctx.body.as_ptr() as usize, b.loc);
                            if !self.bind_active.borrow_mut().insert(gkey.clone()) {
                                continue;
                            }
                            let init = b.init.clone();
                            let bloc = b.loc.clone();
                            let r = self.resolve(&init, ctx, &bloc, depth + 1, callees);
                            self.bind_active.borrow_mut().remove(&gkey);
                            res.merge(&r);
                        }
                    } else if self
                        .file(ctx.file)
                        .map_or(false, |f| f.consts.contains_key(name))
                    {
                        res.consts.insert((name.clone(), ctx.file.to_string()));
                        let r = self.resolve_const(name, ctx.file, depth + 1, callees);
                        res.merge(&r);
                    } else if ctx.params.contains(name) {
                        res.unres.insert(format!("param:{name}"));
                    } else {
                        match self.resolve_global_const(name, depth + 1, callees) {
                            Some(r) if !r.lits.is_empty() || !r.consts.is_empty() => res.merge(&r),
                            _ => {
                                res.unres.insert(name.clone());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn resolve_const(&self, name: &str, file: &str, _depth: u32, callees: bool) -> Res {
        let key = format!("c|{name}|{file}|{callees}");
        if let Some(r) = self.res_cache.borrow().get(&key) {
            return r.clone();
        }
        self.res_cache
            .borrow_mut()
            .insert(key.clone(), Res::default());
        let mut out = Res::default();
        if let Some(f) = self.file(file) {
            if let Some(defs) = f.consts.get(name) {
                let idx = BindIndex::default();
                let ctx = Ctx {
                    file: &f.path,
                    body: &[],
                    params: HashSet::new(),
                    idx: &idx,
                };
                for d in defs {
                    let r = self.resolve(&d.init(), &ctx, &[usize::MAX], 0, callees);
                    out.merge(&r);
                }
            }
        }
        if out.unres.contains("<depth>") {
            self.res_cache.borrow_mut().remove(&key);
        } else {
            self.res_cache.borrow_mut().insert(key, out.clone());
        }
        out
    }

    fn resolve_global_const(&self, name: &str, depth: u32, callees: bool) -> Option<Res> {
        let files = self.const_files.get(name)?;
        let mut total = 0usize;
        for fi in files {
            total += self.files[*fi].consts.get(name).map_or(0, |v| v.len());
        }
        if total != 1 {
            return Some(Res::default());
        }
        let path = self.files[files[0]].path.clone();
        Some(self.resolve_const(name, &path, depth, callees))
    }

    fn resolve_fn_body(&self, fi: usize, fk: usize, _depth: u32) -> Res {
        let key = format!("f|{fi}|{fk}");
        if let Some(r) = self.res_cache.borrow().get(&key) {
            return r.clone();
        }
        self.res_cache
            .borrow_mut()
            .insert(key.clone(), Res::default());
        let func = &self.files[fi].fns[fk];
        let idx = BindIndex::build(&func.body);
        let ctx = Ctx {
            file: &self.files[fi].path,
            body: &func.body,
            params: fn_params(func),
            idx: &idx,
        };
        // A cached result must not depend on the depth it was first asked at: resolve
        // from a fresh budget, and never cache (or hide) a truncated one.
        let r = self.resolve(&func.body, &ctx, &[usize::MAX], 0, true);
        let truncated = r.unres.contains("<depth>");
        let mut unres = BTreeSet::new();
        if truncated {
            unres.insert("<depth>".to_string());
        }
        let out = Res {
            lits: r.lits,
            unres,
            consts: r.consts,
        };
        if truncated {
            self.res_cache.borrow_mut().remove(&key);
        } else {
            self.res_cache.borrow_mut().insert(key, out.clone());
        }
        out
    }

    // ---- loop iterables (array literals and constant arrays) ----------------

    pub(crate) fn loop_elems(&self, it: &[Tt], ctx: &Ctx) -> Result<Vec<Option<Vec<Tt>>>, String> {
        let mut it: Vec<Tt> = it.to_vec();
        while it.first().map_or(false, |t| t.is_p('&')) {
            it.remove(0);
        }
        if it.len() == 1 {
            if let Some(g) = it[0].group() {
                if it[0].is_g('[') {
                    return array_elems(g);
                }
            }
        }
        // a..b with literal bounds
        if it.len() == 4 && it[1].is_p('.') && it[2].is_p('.') {
            if let (Tt::N(a, _), Tt::N(b, _)) = (&it[0], &it[3]) {
                if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    return Ok(vec![None; b.saturating_sub(a)]);
                }
            }
        }
        if it.len() == 1 {
            if let Some(name) = it[0].id() {
                if let Some(bs) = ctx.idx.map.get(name) {
                    for b in bs.iter().rev() {
                        if b.ty.len() == 1 && b.ty[0].is_g('[') {
                            if let Some(g) = b.ty[0].group() {
                                if let Some(semi) = g.iter().position(|t| t.is_p(';')) {
                                    if let Some(n) = self.eval_const_expr_in(
                                        &g[semi + 1..],
                                        &HashMap::new(),
                                        ctx.file,
                                    ) {
                                        return Ok(vec![None; n as usize]);
                                    }
                                }
                            }
                        }
                        let mut init = b.init.clone();
                        while init.first().map_or(false, |t| t.is_p('&')) {
                            init.remove(0);
                        }
                        if init.len() == 1 && init[0].is_g('[') {
                            return array_elems(init[0].group().unwrap());
                        }
                        let ids: Vec<&str> = init.iter().filter_map(|t| t.id()).collect();
                        if !ids.is_empty() && init.iter().all(|t| t.id().is_some() || t.is_p(':')) {
                            let cname = ids[ids.len() - 1];
                            if let Some(files) = self.const_files.get(cname) {
                                if files.len() == 1 {
                                    if let Some(defs) = self.files[files[0]].consts.get(cname) {
                                        if defs.len() == 1 {
                                            let r = defs[0].init();
                                            if r.len() == 1 && r[0].is_g('[') {
                                                return array_elems(r[0].group().unwrap());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // The constant of the launcher's own file wins over a same-named one elsewhere.
                let mut defs_ref: Option<&Vec<ConstRec>> =
                    self.file(ctx.file).and_then(|f| f.consts.get(name));
                if defs_ref.is_none() {
                    if let Some(files) = self.const_files.get(name) {
                        if files.len() == 1 {
                            defs_ref = self.files[files[0]].consts.get(name);
                        }
                    }
                }
                if let Some(defs) = defs_ref {
                    if defs.len() == 1 {
                        let toks = &defs[0].tokens;
                        let eq = toks.iter().position(|t| t.is_p('=')).unwrap_or(toks.len());
                        for t in &toks[..eq] {
                            if t.is_g('[') {
                                let g = t.group().unwrap();
                                if let Some(semi) = g.iter().position(|x| x.is_p(';')) {
                                    if let Some(n) = self.eval_const_expr_in(
                                        &g[semi + 1..],
                                        &HashMap::new(),
                                        ctx.file,
                                    ) {
                                        return Ok(vec![None; n as usize]);
                                    }
                                }
                            }
                        }
                        let r = defs[0].init();
                        if r.len() == 1 && r[0].is_g('[') {
                            return array_elems(r[0].group().unwrap());
                        }
                    }
                }
            }
        }
        Err(format!(
            "loop iterable `{}` is not a literal array, constant array or literal range",
            text(&it)
        ))
    }
}

pub(crate) fn array_elems(g: &[Tt]) -> Result<Vec<Option<Vec<Tt>>>, String> {
    if let Some(semi) = g.iter().position(|t| t.is_p(';')) {
        if let Some(Tt::N(n, _)) = g.get(semi + 1) {
            let count = n
                .parse::<usize>()
                .map_err(|_| format!("array repeat count {n}"))?;
            return Ok(vec![None; count]);
        }
        return Err("array repeat with non-literal count".into());
    }
    Ok(split_top(g, ',').into_iter().map(Some).collect())
}

pub fn unescape(s: &str) -> Option<String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0usize;
    while i < c.len() {
        if c[i] == '\\' {
            let n = *c.get(i + 1)?;
            match n {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                '0' => out.push('\0'),
                '"' | '\\' | '\'' => out.push(n),
                '\n' => {
                    i += 2;
                    while i < c.len() && c[i].is_whitespace() {
                        i += 1;
                    }
                    continue;
                }
                _ => return None,
            }
            i += 2;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    Some(out)
}

// A binding-free pattern selects an arm, but cannot supply its kernel-name value.
// Unrecognised patterns/guards fall back to the original conservative resolver.
fn binding_free_pattern(pat: &[Tt]) -> bool {
    if pat.len() == 1 {
        return match &pat[0] {
            Tt::Id(s, _) => matches!(s.as_str(), "_" | "true" | "false"),
            Tt::N(_, _) | Tt::S(_, _) => true,
            Tt::G('(', inner, _) => inner
                .split(|t| t.is_p(','))
                .filter(|p| !p.is_empty())
                .all(binding_free_pattern),
            _ => false,
        };
    }
    // Qualified unit variants/constants, e.g. `DType::BF16`, introduce no binding.
    pat.len() >= 4
        && (pat.len() - 1) % 3 == 0
        && pat[0].id().is_some()
        && pat[1..]
            .chunks_exact(3)
            .all(|p| p[0].is_p(':') && p[1].is_p(':') && p[2].id().is_some())
}

fn match_arm_value(arm: &[Tt]) -> Option<&[Tt]> {
    let arrow = arm
        .windows(2)
        .position(|p| p[0].is_p('=') && p[1].is_p('>'))?;
    let value = &arm[arrow + 2..];
    (binding_free_pattern(&arm[..arrow]) && !value.is_empty()).then_some(value)
}

fn is_skip_word(w: &str) -> bool {
    matches!(
        w,
        "if" | "else"
            | "match"
            | "as"
            | "mut"
            | "str"
            | "static"
            | "crate"
            | "self"
            | "Self"
            | "Some"
            | "None"
            | "true"
            | "false"
            | "return"
            | "ref"
            | "_"
            | "let"
            | "in"
            | "for"
            | "while"
            | "loop"
            | "break"
            | "continue"
            | "const"
            | "fn"
            | "pub"
            | "mod"
            | "use"
            | "impl"
            | "where"
            | "super"
            | "unsafe"
            | "move"
            | "dyn"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "i32"
            | "i64"
            | "usize"
            | "f32"
            | "bool"
            | "char"
            | "void"
            | "kernels"
            | "Ok"
            | "Err"
            | "Option"
            | "Result"
            | "std"
            | "HipResult"
            | "c_void"
            | "hip_bridge"
            | "HipError"
    )
}

#[derive(Clone, Default, Debug)]
pub struct Res {
    pub lits: BTreeSet<String>,
    pub unres: BTreeSet<String>,
    /// (const name, file) pairs referenced while resolving.
    pub consts: BTreeSet<(String, String)>,
}

impl Res {
    pub fn merge(&mut self, o: &Res) {
        self.lits.extend(o.lits.iter().cloned());
        self.unres.extend(o.unres.iter().cloned());
        self.consts.extend(o.consts.iter().cloned());
    }
}

pub fn fn_params(f: &FnRec) -> HashSet<String> {
    let mut names = HashSet::new();
    for t in &f.sig {
        if let Tt::G('(', c, _) = t {
            for part in split_top(c, ',') {
                if let Some(k) = part.iter().position(|x| x.is_p(':')) {
                    let mut stack: Vec<&Tt> = part[..k].iter().collect();
                    while let Some(x) = stack.pop() {
                        match x {
                            Tt::Id(n, _) if n != "mut" && n != "ref" => {
                                names.insert(n.clone());
                            }
                            Tt::G(_, cc, _) => stack.extend(cc.iter()),
                            _ => {}
                        }
                    }
                }
            }
            break;
        }
    }
    names
}

/// Parameters whose declared type mentions `str` (candidate kernel-name carriers).
pub fn fn_str_params(f: &FnRec) -> HashSet<String> {
    let mut names = HashSet::new();
    for t in &f.sig {
        if let Tt::G('(', c, _) = t {
            for part in split_top(c, ',') {
                if let Some(k) = part.iter().position(|x| x.is_p(':')) {
                    if text(&part[k + 1..])
                        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
                        .any(|w| w == "str")
                    {
                        let mut stack: Vec<&Tt> = part[..k].iter().collect();
                        while let Some(x) = stack.pop() {
                            match x {
                                Tt::Id(n, _) if n != "mut" && n != "ref" => {
                                    names.insert(n.clone());
                                }
                                Tt::G(_, cc, _) => stack.extend(cc.iter()),
                                _ => {}
                            }
                        }
                    }
                }
            }
            break;
        }
    }
    names
}

// ---- bindings ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Binding {
    pub loc: Loc,
    pub ty: Vec<Tt>,
    pub init: Vec<Tt>,
    /// `Some(pos)` for a `for` loop variable: the iterable is `init`, `pos` the tuple
    /// component the variable binds (None for a plain pattern).
    pub for_pos: Option<Option<usize>>,
    /// For a `for` variable: index of the loop body group (its only scope).
    pub body_idx: Option<usize>,
    /// For `let (a, b, ..) = init`: the tuple position this name takes.
    pub let_pos: Option<usize>,
}

#[derive(Default)]
pub struct BindIndex {
    pub map: HashMap<String, Vec<Binding>>,
    /// Reassignments (`name = expr;`): (location, expression), scope-insensitive.
    pub assigns: HashMap<String, Vec<(Loc, Vec<Tt>)>>,
}

pub struct Ctx<'a> {
    pub file: &'a str,
    /// Body of the function being analysed (empty for constants).
    pub body: &'a [Tt],
    pub params: HashSet<String>,
    pub idx: &'a BindIndex,
}

pub fn loc_visible(b: &[usize], s: &[usize]) -> bool {
    let d = b.len();
    if d == 0 || s.len() < d {
        return false;
    }
    b[..d - 1] == s[..d - 1] && b[d - 1] < s[d - 1]
}

impl BindIndex {
    pub fn build(body: &[Tt]) -> BindIndex {
        let mut idx = BindIndex::default();
        let mut loc: Loc = Vec::new();
        idx.walk(body, &mut loc);
        idx
    }

    fn walk(&mut self, ts: &[Tt], loc: &mut Loc) {
        let n = ts.len();
        for i in 0..n {
            loc.push(i);
            let t = &ts[i];
            if t.is_id("let") || t.is_id("const") || t.is_id("static") {
                let is_let = t.is_id("let");
                let mut j = i + 1;
                if j < n && ts[j].is_id("mut") {
                    j += 1;
                }
                let mut names: Vec<String> = Vec::new();
                let mut simple = false;
                // `let (a, b, ..) = init`: position of each simple name in the tuple pattern.
                let mut pos_of: HashMap<String, usize> = HashMap::new();
                if j < n && (ts[j].is_g('(') || ts[j].is_g('[')) && is_let {
                    let mut ids = Vec::new();
                    idents_in(ts[j].group().unwrap(), &mut ids);
                    names = ids
                        .into_iter()
                        .filter(|x| x != "mut" && x != "ref" && x != "_")
                        .collect();
                    if ts[j].is_g('(') {
                        for (q, part) in split_top(ts[j].group().unwrap(), ',').iter().enumerate() {
                            let ids: Vec<&Tt> = part
                                .iter()
                                .filter(|t| !t.is_id("mut") && !t.is_id("ref"))
                                .collect();
                            if ids.len() == 1 {
                                if let Some(nm) = ids[0].id().filter(|s| *s != "_") {
                                    pos_of.insert(nm.to_string(), q);
                                }
                            }
                        }
                    }
                } else if j + 1 < n && is_let && ts[j].id().is_some() && ts[j + 1].is_g('(') {
                    // Tuple-struct pattern such as `let Some((a, b)) = ... else { .. }`.
                    let mut ids = Vec::new();
                    idents_in(ts[j + 1].group().unwrap(), &mut ids);
                    names = ids
                        .into_iter()
                        .filter(|x| x != "mut" && x != "ref" && x != "_")
                        .collect();
                    // `Some((a, b, c))`: each name is one component of the wrapped tuple.
                    if ts[j].is_id("Some") {
                        let inner = ts[j + 1].group().unwrap();
                        if inner.len() == 1 && inner[0].is_g('(') {
                            for (q, part) in
                                split_top(inner[0].group().unwrap(), ',').iter().enumerate()
                            {
                                let ids: Vec<&Tt> = part
                                    .iter()
                                    .filter(|t| !t.is_id("mut") && !t.is_id("ref"))
                                    .collect();
                                if ids.len() == 1 {
                                    if let Some(nm) = ids[0].id().filter(|s| *s != "_") {
                                        pos_of.insert(nm.to_string(), q);
                                    }
                                }
                            }
                        }
                    }
                } else if j < n && ts[j].id().is_some() && !(j + 1 < n && ts[j + 1].is_g('(')) {
                    names.push(ts[j].id().unwrap().to_string());
                    simple = true;
                }
                if !names.is_empty() {
                    let mut k = j + 1;
                    let mut ty = Vec::new();
                    if simple && k < n && ts[k].is_p(':') {
                        k += 1;
                        while k < n && !ts[k].is_p('=') && !ts[k].is_p(';') {
                            ty.push(ts[k].clone());
                            k += 1;
                        }
                    }
                    while k < n && !ts[k].is_p('=') && !ts[k].is_p(';') {
                        k += 1;
                    }
                    if k < n && ts[k].is_p('=') {
                        k += 1;
                        let mut init = Vec::new();
                        while k < n && !ts[k].is_p(';') {
                            init.push(ts[k].clone());
                            k += 1;
                        }
                        for nm in names {
                            let let_pos = pos_of.get(&nm).copied();
                            self.map.entry(nm).or_default().push(Binding {
                                loc: loc.clone(),
                                ty: ty.clone(),
                                init: init.clone(),
                                for_pos: None,
                                body_idx: None,
                                let_pos,
                            });
                        }
                    }
                }
            }
            // Plain reassignment `name = expr;` / `(a, b) = expr;` at statement start.
            if i + 1 < n
                && ts[i + 1].is_p('=')
                && !ts.get(i + 2).map_or(false, |x| x.is_p('='))
                && (i == 0 || ts[i - 1].is_p(';') || ts[i - 1].is_g('{'))
            {
                let mut names: Vec<String> = Vec::new();
                if let Some(nm) = t.id() {
                    if nm != "let" && nm != "mut" {
                        names.push(nm.to_string());
                    }
                } else if t.is_g('(') {
                    let mut ids = Vec::new();
                    idents_in(t.group().unwrap(), &mut ids);
                    names = ids.into_iter().filter(|x| x != "mut" && x != "_").collect();
                }
                if !names.is_empty() {
                    let mut k = i + 2;
                    let mut init = Vec::new();
                    while k < n && !ts[k].is_p(';') {
                        init.push(ts[k].clone());
                        k += 1;
                    }
                    for nm in names {
                        self.assigns
                            .entry(nm)
                            .or_default()
                            .push((loc.clone(), init.clone()));
                    }
                }
            }
            if t.is_id("for") {
                let mut j = i + 1;
                let mut pat: Vec<Tt> = Vec::new();
                while j < n && !ts[j].is_id("in") {
                    pat.push(ts[j].clone());
                    j += 1;
                }
                j += 1;
                let mut it: Vec<Tt> = Vec::new();
                while j < n && !ts[j].is_g('{') {
                    it.push(ts[j].clone());
                    j += 1;
                }
                // (name, path in the pattern): `for (a, (b, c)) in X` binds a=[0], b=[1, 0], c=[1, 1].
                let mut pat_ids: Vec<(String, Vec<usize>)> = Vec::new();
                if pat.len() == 1 && pat[0].is_g('(') {
                    pat_leaves(&pat[0], &mut Vec::new(), &mut pat_ids);
                } else {
                    let mut flat: Vec<String> = Vec::new();
                    idents_in(&pat, &mut flat);
                    pat_ids.extend(flat.into_iter().map(|s| (s, Vec::new())));
                }
                let body_idx = if j < n && ts[j].is_g('{') {
                    Some(j)
                } else {
                    None
                };
                for (nm, path) in pat_ids
                    .into_iter()
                    .filter(|(x, _)| x != "mut" && x != "ref" && x != "_")
                {
                    // One level of tuple: the component of each row. Deeper: the iterator
                    // the component comes from (through `enumerate` / `zip`).
                    let (init, pos) = match path.len() {
                        0 => (it.clone(), None),
                        1 => (it.clone(), Some(path[0])),
                        _ => (project_iter(&it, &path).unwrap_or_else(|| it.clone()), None),
                    };
                    self.map.entry(nm).or_default().push(Binding {
                        loc: loc.clone(),
                        ty: Vec::new(),
                        init,
                        for_pos: Some(pos),
                        body_idx,
                        let_pos: None,
                    });
                }
            }
            if let Tt::G(_, c, _) = t {
                self.walk(c, loc);
            }
            loc.pop();
        }
    }

    pub fn visible(&self, name: &str, sloc: &[usize]) -> Option<&Binding> {
        self.map
            .get(name)?
            .iter()
            .filter(|b| {
                loc_visible(&b.loc, sloc)
                    // A `for` variable only lives inside the loop body.
                    && b.body_idx.map_or(true, |bi| sloc.get(b.loc.len() - 1) == Some(&bi))
            })
            .last()
    }
}

// ---- launch sites -------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Site {
    pub loc: Loc,
    pub method: String,
    /// `m` = launch_maybe_blob*, `M` = launch_params_blob!, `b`/`k` = primitives this gate cannot replay.
    pub kind: char,
    pub name_ex: Vec<Tt>,
    pub params_ex: Vec<Tt>,
    pub values: Vec<Vec<Tt>>,
    pub closure_ex: Vec<Tt>,
    /// Every call argument (blob-only primitives: the last one is the launched blob).
    pub args: Vec<Vec<Tt>>,
    pub line: u32,
}

fn launch_kind(name: &str) -> Option<char> {
    match name {
        "launch_maybe_blob" | "launch_maybe_blob_declared" | "launch_maybe_blob_position_grid" => {
            Some('m')
        }
        "launch_params_blob" => Some('M'),
        "launch_kernel_blob" | "launch_blob_recorded" => Some('b'),
        "launch_kernel" => Some('k'),
        _ => None,
    }
}

pub fn find_sites(body: &[Tt]) -> Result<Vec<Site>, String> {
    let mut out = Vec::new();
    let mut loc: Loc = Vec::new();
    sites_go(body, &mut loc, &mut out)?;
    Ok(out)
}

fn sites_go(ts: &[Tt], loc: &mut Loc, out: &mut Vec<Site>) -> Result<(), String> {
    let n = ts.len();
    let mut i = 0usize;
    while i < n {
        let t = &ts[i];
        if let Some(name) = t.id() {
            if let Some(kind) = launch_kind(name) {
                if i > 0 && ts[i - 1].is_id("fn") {
                    i += 1;
                    continue;
                }
                let mut here = loc.clone();
                here.push(i);
                if kind == 'M' && i + 2 < n && ts[i + 1].is_p('!') && ts[i + 2].group().is_some() {
                    let inner = ts[i + 2].group().unwrap();
                    let semis: Vec<usize> = inner
                        .iter()
                        .enumerate()
                        .filter(|(_, x)| x.is_p(';'))
                        .map(|(k, _)| k)
                        .collect();
                    if semis.len() != 1 {
                        return Err(format!(
                            "line {}: launch_params_blob! must contain exactly one `;`",
                            t.line()
                        ));
                    }
                    let head = split_top(&inner[..semis[0]], ',');
                    let vals = split_top(&inner[semis[0] + 1..], ',');
                    if head.len() != 6 {
                        return Err(format!(
                            "line {}: launch_params_blob! head has {} arguments, expected 6",
                            t.line(),
                            head.len()
                        ));
                    }
                    out.push(Site {
                        loc: here,
                        method: name.to_string(),
                        kind,
                        name_ex: head[1].clone(),
                        params_ex: head[5].clone(),
                        values: vals,
                        closure_ex: Vec::new(),
                        args: Vec::new(),
                        line: t.line(),
                    });
                    i += 3;
                    continue;
                }
                if kind != 'M' && i + 1 < n && ts[i + 1].is_g('(') {
                    let args = split_top(ts[i + 1].group().unwrap(), ',');
                    if args.is_empty() {
                        return Err(format!("line {}: {name} without arguments", t.line()));
                    }
                    let (params_ex, closure_ex) = if kind == 'm' {
                        if args.len() < 6 {
                            return Err(format!(
                                "line {}: {name} has {} arguments, expected at least 6",
                                t.line(),
                                args.len()
                            ));
                        }
                        (args[4].clone(), args[args.len() - 1].clone())
                    } else {
                        (Vec::new(), Vec::new())
                    };
                    out.push(Site {
                        loc: here,
                        method: name.to_string(),
                        kind,
                        name_ex: args[0].clone(),
                        params_ex,
                        values: Vec::new(),
                        closure_ex,
                        args: args.clone(),
                        line: t.line(),
                    });
                    i += 2;
                    continue;
                }
            }
        }
        if let Tt::G(_, c, _) = t {
            loc.push(i);
            sites_go(c, loc, out)?;
            loc.pop();
        }
        i += 1;
    }
    Ok(())
}

enum Seg {
    Lit(String),
    /// A `{placeholder}`: the literals it resolves to, or a number.
    Ph(Option<BTreeSet<String>>),
}

fn seg_match(segs: &[Seg], s: &str) -> bool {
    match segs.first() {
        None => s.is_empty(),
        Some(Seg::Lit(l)) => s
            .strip_prefix(l.as_str())
            .map_or(false, |r| seg_match(&segs[1..], r)),
        Some(Seg::Ph(Some(set))) => set.iter().any(|c| {
            s.strip_prefix(c.as_str())
                .map_or(false, |r| seg_match(&segs[1..], r))
        }),
        // An unresolved placeholder is only assumed to print a number (a tile/lane count
        // such as `_nt{nt}`); it never stands for arbitrary kernel-name text.
        Some(Seg::Ph(None)) => {
            let digits = s.chars().take_while(|c| c.is_ascii_digit()).count();
            (1..=digits).any(|k| seg_match(&segs[1..], &s[k..]))
        }
    }
}

/// Does an expression read `name`: as an identifier, or as a `{name}` / `{name:..}`
/// placeholder inside a string literal (format arguments captured by name)?
pub fn mentions_name(ts: &[Tt], name: &str) -> bool {
    ts.iter().any(|t| match t {
        Tt::Id(s, _) => s == name,
        Tt::S(s, _) => s.contains(&format!("{{{name}}}")) || s.contains(&format!("{{{name}:")),
        Tt::G(_, c, _) => mentions_name(c, name),
        _ => false,
    })
}

/// Component `k` of every row of a literal array of tuples (`[(a, b), (c, d)]` with `k = 1`
/// gives `b, d`); anything else is returned unchanged.
fn tuple_component(init: &[Tt], k: usize) -> Vec<Tt> {
    let mut t: Vec<Tt> = init.to_vec();
    while t.first().map_or(false, |x| x.is_p('&') || x.is_id("mut")) {
        t.remove(0);
    }
    let lit: Option<&[Tt]> = if t.len() == 1 && t[0].is_g('[') {
        t[0].group()
    } else {
        None
    };
    let Some(lit) = lit else { return init.to_vec() };
    let mut out: Vec<Tt> = Vec::new();
    for e in split_top(lit, ',') {
        if !(e.len() == 1 && e[0].is_g('(')) {
            return init.to_vec();
        }
        let parts = split_top(e[0].group().unwrap(), ',');
        let Some(p) = parts.get(k) else {
            return init.to_vec();
        };
        if !out.is_empty() {
            out.push(Tt::P(',', 0));
        }
        out.extend(p.iter().cloned());
    }
    out
}

impl World {
    /// `for (a, b, ..) in rows` where `rows` is a PARAMETER of the function being resolved:
    /// component `k` of every row of the literal each call of that function passes.
    /// `None` when the iterable is not such a parameter.
    fn resolve_param_tuple(
        &self,
        ctx: &Ctx,
        init: &[Tt],
        k: usize,
        depth: u32,
        callees: bool,
    ) -> Option<Res> {
        let mut t: Vec<Tt> = init.to_vec();
        while t.first().map_or(false, |x| x.is_p('&') || x.is_id("mut")) {
            t.remove(0);
        }
        if t.len() != 1 {
            return None;
        }
        let x = t[0].id()?;
        if ctx.idx.map.contains_key(x) || !ctx.params.contains(x) {
            return None;
        }
        let fi = *self.file_index.get(ctx.file)?;
        let func = self.files[fi]
            .fns
            .iter()
            .find(|f| f.body.as_ptr() == ctx.body.as_ptr() && f.body.len() == ctx.body.len())?;
        let pos = crate::layout::fn_formals(func)
            .iter()
            .position(|p| p == x)?;
        let srcs = self.call_srcs(fi, &func.name, pos);
        let mut out = Res::default();
        if srcs.is_empty() {
            out.unres.insert(format!("param:{x}"));
            return Some(out);
        }
        for (ck, loc, actual) in srcs.iter() {
            let caller = &self.files[fi].fns[*ck];
            let idx = BindIndex::build(&caller.body);
            let cctx = Ctx {
                file: &self.files[fi].path,
                body: &caller.body,
                params: fn_params(caller),
                idx: &idx,
            };
            let comp = tuple_component(actual, k);
            out.merge(&self.resolve(&comp, &cctx, loc, depth + 1, callees));
        }
        Some(out)
    }
}

fn count_ids(ts: &[Tt], acc: &mut HashMap<String, usize>) {
    for t in ts {
        match t {
            Tt::Id(s, _) => *acc.entry(s.clone()).or_insert(0) += 1,
            Tt::G(_, c, _) => count_ids(c, acc),
            _ => {}
        }
    }
}

/// Every `Type { field: EXPR, .. }` / shorthand `field` and `.field = EXPR;` in `ts`, with
/// the index of the enclosing function (`None` in constants).
fn collect_fields(
    ts: &[Tt],
    fk: Option<usize>,
    out: &mut HashMap<String, Vec<(Option<usize>, Vec<Tt>)>>,
) {
    let n = ts.len();
    for i in 0..n {
        if let Tt::Id(s, _) = &ts[i] {
            let upper = s == "Self" || s.chars().next().map_or(false, |c| c.is_uppercase());
            if upper && ts.get(i + 1).map_or(false, |t| t.is_g('{')) {
                for part in split_top(ts[i + 1].group().unwrap(), ',') {
                    let Some(f) = part.first().and_then(|t| t.id()) else {
                        continue;
                    };
                    if part.len() == 1 {
                        out.entry(f.to_string())
                            .or_default()
                            .push((fk, part.clone()));
                    } else if part.len() > 2 && part[1].is_p(':') && !part[2].is_p(':') {
                        out.entry(f.to_string())
                            .or_default()
                            .push((fk, part[2..].to_vec()));
                    }
                }
            }
        }
        if ts[i].is_p('.')
            && ts.get(i + 1).map_or(false, |t| t.id().is_some())
            && ts.get(i + 2).map_or(false, |t| t.is_p('='))
            && !ts.get(i + 3).map_or(false, |t| t.is_p('='))
        {
            let mut rhs = Vec::new();
            let mut k = i + 3;
            while k < n && !ts[k].is_p(';') {
                rhs.push(ts[k].clone());
                k += 1;
            }
            out.entry(ts[i + 1].id().unwrap().to_string())
                .or_default()
                .push((fk, rhs));
        }
        if let Tt::G(_, c, _) = &ts[i] {
            collect_fields(c, fk, out);
        }
    }
}

impl World {
    /// `let Some((a, b, ..)) = path.field else { .. }`: component `k` of every value the field
    /// `field` is given in `file` (`None` entries give nothing). `None` when `init` is not a
    /// plain field path.
    fn field_path_component(&self, file: &str, init: &[Tt], k: usize) -> Option<Res> {
        let mut t: Vec<Tt> = init.to_vec();
        if t.len() >= 2 && t[t.len() - 2].is_id("else") && t[t.len() - 1].is_g('{') {
            t.truncate(t.len() - 2);
        }
        while t.first().map_or(false, |x| x.is_p('&')) {
            t.remove(0);
        }
        let n = t.len();
        if n < 3 || !t[n - 2].is_p('.') || t.iter().any(|x| x.is_id("if") || x.is_g('(')) {
            return None;
        }
        let field = t[n - 1].id()?.to_string();
        let sources = self.field_literals(file, &field);
        if sources.is_empty() {
            return None;
        }
        let empty = BindIndex::default();
        let ectx = Ctx {
            file,
            body: &[],
            params: HashSet::new(),
            idx: &empty,
        };
        let mut acc = Res::default();
        for (_, e) in sources.iter() {
            let mut v: Vec<Tt> = e.clone();
            while v.first().map_or(false, |x| x.is_p('&')) {
                v.remove(0);
            }
            if v.len() == 1 && v[0].is_id("None") {
                continue;
            }
            let part = project_let(&v, k).unwrap_or_else(|| e.clone());
            let r = self.resolve(&part, &ectx, &[usize::MAX], 0, false);
            acc.lits.extend(r.lits);
            acc.consts.extend(r.consts);
        }
        Some(acc)
    }

    fn field_literals(&self, file: &str, field: &str) -> Rc<Vec<(Option<usize>, Vec<Tt>)>> {
        let Some(&fi) = self.file_index.get(file) else {
            return Rc::new(Vec::new());
        };
        if !self.field_maps.borrow().contains_key(&fi) {
            let mut m: HashMap<String, Vec<(Option<usize>, Vec<Tt>)>> = HashMap::new();
            for (fk, f) in self.files[fi].fns.iter().enumerate() {
                collect_fields(&f.body, Some(fk), &mut m);
            }
            for defs in self.files[fi].consts.values() {
                for d in defs {
                    collect_fields(&d.tokens, None, &mut m);
                }
            }
            self.field_maps.borrow_mut().insert(fi, Rc::new(m));
        }
        let maps = self.field_maps.borrow();
        Rc::new(maps[&fi].get(field).cloned().unwrap_or_default())
    }

    /// Literals and constants a struct field can hold in `file`: what struct literals and
    /// assignments give it. A value that is a parameter of the constructing function (a
    /// shorthand `field`, or `field: param`) comes from every call of that function: in
    /// function bodies (resolved in the caller) and in constants (const tables).
    fn resolve_field(&self, file: &str, field: &str) -> Res {
        let mut acc = Res::default();
        let Some(&fi) = self.file_index.get(file) else {
            return acc;
        };
        let empty = BindIndex::default();
        let ectx = Ctx {
            file,
            body: &[],
            params: HashSet::new(),
            idx: &empty,
        };
        for (fk, e) in self.field_literals(file, field).iter() {
            let param = match (fk, e.len()) {
                (Some(fk), 1) => e[0].id().and_then(|id| {
                    let f = &self.files[fi].fns[*fk];
                    crate::layout::fn_formals(f)
                        .iter()
                        .position(|p| p == id)
                        .map(|pos| (*fk, pos))
                }),
                _ => None,
            };
            let Some((cfk, pos)) = param else {
                let r = self.resolve(e, &ectx, &[usize::MAX], 0, false);
                acc.lits.extend(r.lits);
                acc.consts.extend(r.consts);
                continue;
            };
            let fname = self.files[fi].fns[cfk].name.clone();
            // Calls in function bodies, resolved in their caller.
            for (ck, loc, actual) in crate::layout::calls_in_fns(self, fi, &fname, pos).iter() {
                let caller = &self.files[fi].fns[*ck];
                let idx = BindIndex::build(&caller.body);
                let cctx = Ctx {
                    file,
                    body: &caller.body,
                    params: fn_params(caller),
                    idx: &idx,
                };
                let r = self.resolve(actual, &cctx, loc, 0, false);
                acc.lits.extend(r.lits);
                acc.consts.extend(r.consts);
            }
            // Calls in constant tables.
            for defs in self.files[fi].consts.values() {
                for d in defs {
                    let mut found: Vec<(Loc, Vec<Tt>)> = Vec::new();
                    crate::flow::call_sources_in(
                        &d.tokens,
                        &fname,
                        pos,
                        &mut Vec::new(),
                        &mut found,
                    );
                    for (_, actual) in found {
                        let r = self.resolve(&actual, &ectx, &[usize::MAX], 0, false);
                        acc.lits.extend(r.lits);
                        acc.consts.extend(r.consts);
                    }
                }
            }
        }
        acc
    }
}

/// Could `tpl` (a `format!` template) produce any of `names`, with placeholders printing
/// anything? The literal pieces must occur in order, the first at the start when the
/// template starts with literal text, the last at the end when it ends with literal text.
fn template_can_match(tpl: &str, names: &BTreeSet<String>) -> bool {
    let c: Vec<char> = tpl.chars().collect();
    let mut pieces: Vec<String> = Vec::new();
    let mut cur = String::new();
    let (mut starts_lit, mut ends_lit) = (false, false);
    let mut i = 0usize;
    while i < c.len() {
        if c[i] == '{' && c.get(i + 1) == Some(&'{') {
            cur.push('{');
            i += 2;
        } else if c[i] == '}' && c.get(i + 1) == Some(&'}') {
            cur.push('}');
            i += 2;
        } else if c[i] == '{' {
            if i == 0 {
                starts_lit = false;
            }
            if !cur.is_empty() || !pieces.is_empty() {
                pieces.push(std::mem::take(&mut cur));
            } else {
                pieces.push(String::new());
            }
            while i < c.len() && c[i] != '}' {
                i += 1;
            }
            i += 1;
            ends_lit = false;
        } else {
            if i == 0 {
                starts_lit = true;
            }
            cur.push(c[i]);
            i += 1;
            ends_lit = true;
        }
    }
    pieces.push(cur);
    names.iter().any(|nm| {
        let mut pos = 0usize;
        let last = pieces.len() - 1;
        for (k, piece) in pieces.iter().enumerate() {
            if piece.is_empty() {
                continue;
            }
            if k == 0 && starts_lit {
                if !nm.starts_with(piece.as_str()) {
                    return false;
                }
                pos = piece.len();
            } else if k == last && ends_lit {
                return nm.len() >= pos + piece.len() && nm.ends_with(piece.as_str());
            } else {
                match nm[pos..].find(piece.as_str()) {
                    Some(f) => pos += f + piece.len(),
                    None => return false,
                }
            }
        }
        true
    })
}

/// What a binding's name stands for: its initialiser, or, for a tuple pattern
/// (`for (a, b) in rows` / `let (a, b) = if c { (x, y) } else { (z, w) }`), only the
/// component it takes from each row / branch.
pub fn binding_init(b: &Binding) -> Vec<Tt> {
    match (b.for_pos, b.let_pos) {
        (Some(Some(k)), _) => tuple_component(&b.init, k),
        (_, Some(k)) => project_let(&b.init, k).unwrap_or_else(|| b.init.clone()),
        _ => b.init.clone(),
    }
}

/// Component `k` of a tuple expression, following `if`/`else` branches to their tails.
/// `None` when the expression is not a tuple or a chain of them.
fn project_let(init: &[Tt], k: usize) -> Option<Vec<Tt>> {
    let mut t: Vec<Tt> = init.to_vec();
    while t.first().map_or(false, |x| x.is_p('&')) {
        t.remove(0);
    }
    if t.len() == 2 && t[0].is_id("Some") && t[1].is_g('(') {
        return project_let(t[1].group().unwrap(), k);
    }
    if t.len() == 1 && t[0].is_g('(') {
        return split_top(t[0].group().unwrap(), ',').get(k).cloned();
    }
    if t.len() == 1 && t[0].is_g('{') {
        let tail = crate::layout::split_stmts(t[0].group().unwrap()).1;
        return project_let(&tail, k);
    }
    if t.first().map_or(false, |x| x.is_id("if")) {
        let mut bodies: Vec<Vec<Tt>> = Vec::new();
        crate::layout::if_chain(&t, 0, &Vec::new(), |b, _| {
            bodies.push(b.to_vec());
            Ok(())
        })
        .ok()?;
        let mut out: Vec<Tt> = Vec::new();
        for b in bodies {
            let tail = crate::layout::split_stmts(&b).1;
            let comp = project_let(&tail, k)?;
            if !out.is_empty() {
                out.push(Tt::P(',', 0));
            }
            out.extend(comp);
        }
        return Some(out);
    }
    None
}

/// Leaves of a `for` pattern with their positions: `(i, (l, blob))` gives
/// `i` at `[0]`, `l` at `[1, 0]`, `blob` at `[1, 1]`. A leaf is the first plain identifier
/// of its part (`mut x`, `&x`), so a flat pattern keeps its previous meaning.
pub(crate) fn pat_leaves(group: &Tt, path: &mut Vec<usize>, out: &mut Vec<(String, Vec<usize>)>) {
    let Some(g) = group.group() else { return };
    for (k, part) in split_top(g, ',').iter().enumerate() {
        path.push(k);
        if part.len() == 1 && part[0].is_g('(') {
            pat_leaves(&part[0], path, out);
        } else if let Some(id) = part
            .iter()
            .find_map(|t| t.id().filter(|x| *x != "mut" && *x != "ref"))
        {
            out.push((id.to_string(), path.clone()));
        }
        path.pop();
    }
}

/// The iterator an item component comes from. `A.zip(B)` items are `(A item, B item)` and
/// `X.enumerate()` items are `(index, X item)`; `None` when the chain is neither (or the
/// component is the index). Strict: any other adapter may reorder or drop rows, which would
/// break the pairing of a name with its blob.
pub(crate) fn project_iter(chain: &[Tt], path: &[usize]) -> Option<Vec<Tt>> {
    project_iter_with(chain, path, false)
}

/// Like `project_iter` for proofs about WHICH values an element can be (an origin or a size),
/// where order and row subsets do not matter: adapters that keep the item type are looked
/// through.
pub(crate) fn project_iter_unordered(chain: &[Tt], path: &[usize]) -> Option<Vec<Tt>> {
    project_iter_with(chain, path, true)
}

fn project_iter_with(chain: &[Tt], path: &[usize], through_adapters: bool) -> Option<Vec<Tt>> {
    let Some((&first, rest)) = path.split_first() else {
        return Some(chain.to_vec());
    };
    let n = chain.len();
    if n < 4 || !chain[n - 1].is_g('(') || !chain[n - 3].is_p('.') {
        return None;
    }
    let inner = &chain[..n - 3];
    match chain[n - 2].id()? {
        "rev" | "filter" | "skip" | "take" | "step_by" | "skip_while" | "take_while"
        | "inspect" | "peekable"
            if through_adapters =>
        {
            project_iter_with(inner, path, true)
        }
        "enumerate" if first == 1 => project_iter_with(inner, rest, through_adapters),
        "zip" => {
            if first == 0 {
                project_iter_with(inner, rest, through_adapters)
            } else if first == 1 {
                project_iter_with(chain[n - 1].group()?, rest, through_adapters)
            } else {
                None
            }
        }
        _ => None,
    }
}
