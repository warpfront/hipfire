//! The evaluator: items, expressions, patterns and calls over `syn` ASTs.
//!
//! This is a Rust-shaped compile-time evaluator, not a compiler: it executes
//! the script's functions to drive the generic builder. Everything a script
//! executes must be supported; anything else is a located error, never a
//! silent no-op. Items the run never reaches are not examined.

use crate::driver::DynDriver;
use crate::native::{self, Nat};
use crate::ops;
use crate::value::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use syn::spanned::Spanned;
use syn::{BinOp, Block, Expr, Item, Lit, Member, Pat, Stmt, UnOp};

/// Non-local control flow, and failure.
pub enum Ctl<'a> {
    /// A fault not yet given a source location.
    Err(String),
    /// A located fault: evaluation stops.
    Fatal(String),
    Return(Value<'a>),
    Break(Option<String>, Value<'a>),
    Continue(Option<String>),
}
pub type R<'a, T = Value<'a>> = Result<T, Ctl<'a>>;

impl<'a> From<String> for Ctl<'a> {
    fn from(s: String) -> Self {
        Ctl::Err(s)
    }
}
impl<'a> From<&str> for Ctl<'a> {
    fn from(s: &str) -> Self {
        Ctl::Err(s.to_string())
    }
}

/// The driver context of the code being evaluated: the sealed driver of the
/// current scope (`None` outside any `Workgroup` scope) and the
/// forward / break handles of the enclosing control-flow callbacks.
pub struct Cx<'c> {
    pub d: Option<&'c mut (dyn DynDriver + 'c)>,
    pub fwd: Option<&'c mut peacemaker_author::runtime::Forward<hipfire_isa::Builder>>,
    pub brk: Option<&'c peacemaker_author::runtime::Breaks<hipfire_isa::Builder>>,
}
impl<'c> Cx<'c> {
    pub fn root() -> Self {
        Cx { d: None, fwd: None, brk: None }
    }
}

pub struct StructDef {
    pub fields: Vec<(String, Ty)>,
    pub tuple: bool,
}
pub struct EnumDef {
    /// Variant, tuple arity, discriminant.
    pub variants: Vec<(String, usize, Option<i128>)>,
}
struct ConstDef<'a> {
    expr: &'a Expr,
    ty: Ty,
}

pub struct Interp<'a> {
    pub name: String,
    fns: HashMap<String, Rc<FnDef<'a>>>,
    methods: HashMap<String, Rc<FnDef<'a>>>,
    consts: HashMap<String, ConstDef<'a>>,
    const_cache: RefCell<HashMap<String, Option<Value<'a>>>>,
    structs: HashMap<String, StructDef>,
    enums: HashMap<String, EnumDef>,
    depth: Cell<usize>,
}

const MAX_DEPTH: usize = 160;

fn type_key(t: &syn::Type) -> Option<String> {
    match t {
        syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        syn::Type::Reference(r) => type_key(&r.elem),
        syn::Type::Paren(p) => type_key(&p.elem),
        syn::Type::Group(g) => type_key(&g.elem),
        _ => None,
    }
}

fn make_fn<'a>(name: &str, sig: &'a syn::Signature, block: &'a Block, self_ty: Option<Rc<str>>) -> FnDef<'a> {
    let mut params = Vec::new();
    let mut recv = 0;
    for a in &sig.inputs {
        match a {
            syn::FnArg::Receiver(r) => recv = if r.reference.is_some() { if r.mutability.is_some() { 3 } else { 2 } } else { 1 },
            syn::FnArg::Typed(pt) => params.push((&*pt.pat, Ty::from_syn(&pt.ty))),
        }
    }
    let ret = match &sig.output {
        syn::ReturnType::Type(_, t) => Ty::from_syn(t),
        syn::ReturnType::Default => Ty::Any,
    };
    FnDef { name: name.to_string(), sig, block, self_ty, params, ret, recv }
}

impl<'a> Interp<'a> {
    pub fn load(file: &'a syn::File, name: &str) -> Interp<'a> {
        let mut it = Interp {
            name: name.to_string(),
            fns: HashMap::new(),
            methods: HashMap::new(),
            consts: HashMap::new(),
            const_cache: RefCell::new(HashMap::new()),
            structs: HashMap::new(),
            enums: HashMap::new(),
            depth: Cell::new(0),
        };
        let mut traits: HashMap<String, Vec<(String, &'a syn::TraitItemFn)>> = HashMap::new();
        it.load_items(&file.items, &mut traits);
        it
    }

    fn load_items(&mut self, items: &'a [Item], traits: &mut HashMap<String, Vec<(String, &'a syn::TraitItemFn)>>) {
        for item in items {
            match item {
                Item::Fn(f) => {
                    let n = f.sig.ident.to_string();
                    self.fns.insert(n.clone(), Rc::new(make_fn(&n, &f.sig, &f.block, None)));
                }
                Item::Const(c) => {
                    self.consts.insert(c.ident.to_string(), ConstDef { expr: &c.expr, ty: Ty::from_syn(&c.ty) });
                }
                Item::Static(c) => {
                    self.consts.insert(c.ident.to_string(), ConstDef { expr: &c.expr, ty: Ty::from_syn(&c.ty) });
                }
                Item::Struct(s) => {
                    let (fields, tuple) = match &s.fields {
                        syn::Fields::Named(n) => (n.named.iter().map(|f| (f.ident.as_ref().map(|i| i.to_string()).unwrap_or_default(), Ty::from_syn(&f.ty))).collect(), false),
                        syn::Fields::Unnamed(u) => (u.unnamed.iter().enumerate().map(|(i, f)| (i.to_string(), Ty::from_syn(&f.ty))).collect(), true),
                        syn::Fields::Unit => (Vec::new(), false),
                    };
                    self.structs.insert(s.ident.to_string(), StructDef { fields, tuple });
                }
                Item::Enum(e) => {
                    let mut next = 0i128;
                    let mut variants = Vec::new();
                    for v in &e.variants {
                        let disc = match &v.discriminant {
                            Some((_, Expr::Lit(l))) => match &l.lit {
                                Lit::Int(i) => i.base10_parse::<i128>().ok(),
                                _ => None,
                            },
                            _ => None,
                        };
                        let d = disc.unwrap_or(next);
                        next = d + 1;
                        let arity = match &v.fields {
                            syn::Fields::Unit => 0,
                            syn::Fields::Unnamed(u) => u.unnamed.len(),
                            syn::Fields::Named(n) => n.named.len() + 1_000_000,
                        };
                        variants.push((v.ident.to_string(), arity, Some(d)));
                    }
                    self.enums.insert(e.ident.to_string(), EnumDef { variants });
                }
                Item::Trait(t) => {
                    let mut v = Vec::new();
                    for ti in &t.items {
                        if let syn::TraitItem::Fn(f) = ti {
                            if f.default.is_some() {
                                v.push((f.sig.ident.to_string(), f));
                            }
                        }
                    }
                    traits.insert(t.ident.to_string(), v);
                }
                Item::Impl(imp) => {
                    let Some(ty) = type_key(&imp.self_ty) else { continue };
                    let self_ty: Rc<str> = Rc::from(ty.as_str());
                    let mut defined: Vec<String> = Vec::new();
                    for ii in &imp.items {
                        match ii {
                            syn::ImplItem::Fn(f) => {
                                let n = f.sig.ident.to_string();
                                defined.push(n.clone());
                                self.methods.insert(format!("{ty}::{n}"), Rc::new(make_fn(&n, &f.sig, &f.block, Some(self_ty.clone()))));
                            }
                            syn::ImplItem::Const(c) => {
                                self.consts.insert(format!("{ty}::{}", c.ident), ConstDef { expr: &c.expr, ty: Ty::from_syn(&c.ty) });
                            }
                            _ => {}
                        }
                    }
                    if let Some((_, tp, _)) = &imp.trait_ {
                        if let Some(tn) = tp.segments.last().map(|s| s.ident.to_string()) {
                            if let Some(defaults) = traits.get(&tn) {
                                for (n, f) in defaults {
                                    if !defined.contains(n) {
                                        let block = f.default.as_ref().expect("default body");
                                        self.methods.insert(format!("{ty}::{n}"), Rc::new(make_fn(n, &f.sig, block, Some(self_ty.clone()))));
                                    }
                                }
                            }
                        }
                    }
                }
                Item::Mod(m) => {
                    if let Some((_, items)) = &m.content {
                        self.load_items(items, traits);
                    }
                }
                _ => {}
            }
        }
    }

    // ---- driver ownership ------------------------------------------------


    // ---- entry -------------------------------------------------------------

    pub fn has_fn(&self, name: &str) -> bool {
        self.fns.contains_key(name)
    }
    pub fn enum_has_variant(&self, ty: &str, var: &str) -> Option<String> {
        self.enums.get(ty).and_then(|e| e.variants.iter().find(|(v, _, _)| v.eq_ignore_ascii_case(var)).map(|(v, _, _)| v.clone()))
    }
    pub fn enum_variants(&self, ty: &str) -> Option<Vec<String>> {
        self.enums.get(ty).map(|e| e.variants.iter().map(|(v, _, _)| v.clone()).collect())
    }
    pub fn fn_arity(&self, name: &str) -> Option<usize> {
        self.fns.get(name).map(|f| f.params.len())
    }

    /// Call the script function `name`. Hard faults come back as
    /// `Err(located message)`; the script's own `Err(..)` is a normal value.
    pub fn call_entry(&self, name: &str, args: Vec<Value<'a>>) -> Result<Value<'a>, String> {
        let f = self.fns.get(name).cloned().ok_or_else(|| format!("{}: no `fn {name}` entry point", self.name))?;
        let mut cx = Cx::root();
        let r = self.call_fn(&f, args, &mut cx);
        match r {
            Ok(v) => Ok(v),
            Err(Ctl::Return(v)) => Ok(v),
            Err(Ctl::Fatal(m)) => Err(m),
            Err(Ctl::Err(m)) => Err(format!("{}: {m}", self.name)),
            Err(Ctl::Break(..)) | Err(Ctl::Continue(_)) => Err(format!("{}: `break`/`continue` outside a loop", self.name)),
        }
    }

    // ---- values ------------------------------------------------------------

    pub fn deref(&self, v: Value<'a>) -> Value<'a> {
        match v {
            Value::Ref(r) => {
                let inner = self.read_place(&r);
                self.deref(inner)
            }
            v => v,
        }
    }

    fn walk<'v>(v: &'v Value<'a>, p: &Proj) -> Option<&'v Value<'a>> {
        match (v, p) {
            (Value::Struct(_, fs), Proj::Field(n)) => fs.iter().find(|(k, _)| k == n).map(|(_, x)| x),
            (Value::Tuple(xs), Proj::Field(n)) | (Value::Enum(_, _, xs), Proj::Field(n)) => n.parse::<usize>().ok().and_then(|i| xs.get(i)),
            (Value::Array(xs), Proj::Index(i)) | (Value::Tuple(xs), Proj::Index(i)) => xs.get(*i),
            (Value::Ref(r), _) => {
                // A reference stored in the path: continue through it (read-only).
                let _ = r;
                None
            }
            _ => None,
        }
    }
    fn walk_mut<'v>(v: &'v mut Value<'a>, p: &Proj) -> Option<&'v mut Value<'a>> {
        match (v, p) {
            (Value::Struct(_, fs), Proj::Field(n)) => fs.iter_mut().find(|(k, _)| k == n).map(|(_, x)| x),
            (Value::Tuple(xs), Proj::Field(n)) | (Value::Enum(_, _, xs), Proj::Field(n)) => n.parse::<usize>().ok().and_then(|i| xs.get_mut(i)),
            (Value::Array(xs), Proj::Index(i)) | (Value::Tuple(xs), Proj::Index(i)) => xs.get_mut(*i),
            _ => None,
        }
    }

    pub fn read_place(&self, r: &PlaceRef<'a>) -> Value<'a> {
        let cell = r.cell.borrow();
        let mut cur: &Value<'a> = &cell;
        for p in &r.path {
            match Self::walk(cur, p) {
                Some(n) => cur = n,
                None => return Value::Unit,
            }
        }
        cur.clone()
    }

    pub fn write_place(&self, r: &PlaceRef<'a>, v: Value<'a>) -> Result<(), String> {
        let mut cell = r.cell.borrow_mut();
        let mut cur: &mut Value<'a> = &mut cell;
        for p in &r.path {
            cur = Self::walk_mut(cur, p).ok_or("assignment through an invalid place")?;
        }
        *cur = adopt(cur, v);
        Ok(())
    }

    /// Run `f` on the value at a place, mutably.
    pub fn with_place_mut<T>(&self, r: &PlaceRef<'a>, f: impl FnOnce(&mut Value<'a>) -> Result<T, String>) -> Result<T, String> {
        let mut cell = r.cell.borrow_mut();
        let mut cur: &mut Value<'a> = &mut cell;
        for p in &r.path {
            cur = Self::walk_mut(cur, p).ok_or("mutation through an invalid place")?;
        }
        f(cur)
    }

    /// A place expression as a reference: variables, fields and indexes of
    /// variables, and `*r` of a reference value.
    pub fn place(&self, e: &'a Expr, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a, PlaceRef<'a>> {
        match e {
            Expr::Paren(p) => self.place(&p.expr, env, cx),
            Expr::Path(p) if p.path.segments.len() == 1 && p.qself.is_none() => {
                let n = p.path.segments[0].ident.to_string();
                let cell = lookup(env, &n).ok_or_else(|| Ctl::Err(format!("cannot find variable `{n}`")))?.clone();
                Ok(PlaceRef { cell, path: vec![] })
            }
            Expr::Field(f) => {
                let mut base = self.place(&f.base, env, cx)?;
                self.autoderef(&mut base);
                base.path.push(Proj::Field(match &f.member {
                    Member::Named(i) => Rc::from(i.to_string().as_str()),
                    Member::Unnamed(i) => Rc::from(i.index.to_string().as_str()),
                }));
                Ok(base)
            }
            Expr::Index(ix) => {
                let mut base = self.place(&ix.expr, env, cx)?;
                self.autoderef(&mut base);
                let i = self.eval(&ix.index, env, cx)?;
                let Value::Int(i, _) = self.deref(i) else { return Err("index must be an integer".into()) };
                base.path.push(Proj::Index(usize::try_from(i).map_err(|_| "negative index")?));
                Ok(base)
            }
            Expr::Unary(u) if matches!(u.op, UnOp::Deref(_)) => {
                let v = self.eval(&u.expr, env, cx)?;
                match v {
                    Value::Ref(r) => {
                        let mut p = PlaceRef { cell: r.cell.clone(), path: r.path.clone() };
                        self.autoderef(&mut p);
                        Ok(p)
                    }
                    _ => self.place(&u.expr, env, cx),
                }
            }
            _ => Err("expression is not an assignable place".into()),
        }
    }

    /// Follow references stored at the place until it is the data itself.
    fn autoderef(&self, p: &mut PlaceRef<'a>) {
        loop {
            let v = self.read_place(p);
            match v {
                Value::Ref(r) => {
                    let mut path = r.path.clone();
                    path.extend(std::mem::take(&mut p.path).into_iter().take(0));
                    *p = PlaceRef { cell: r.cell.clone(), path };
                }
                _ => return,
            }
        }
    }

    // ---- constants and paths -------------------------------------------------

    fn const_value(&self, key: &str) -> R<'a, Option<Value<'a>>> {
        if let Some(c) = self.const_cache.borrow().get(key) {
            return match c {
                Some(v) => Ok(Some(v.clone())),
                None => Err(format!("constant `{key}` depends on itself").into()),
            };
        }
        let Some(def) = self.consts.get(key) else { return Ok(None) };
        self.const_cache.borrow_mut().insert(key.to_string(), None);
        let mut env: Env<'a> = None;
        if let Some((t, _)) = key.split_once("::") {
            env = bind(&env, "Self", Value::str(t));
        }
        let mut cx = Cx::root();
        let v = self.eval(def.expr, &mut env, &mut cx)?;
        let v = ops::coerce(self.deref(v), &def.ty)?;
        self.const_cache.borrow_mut().insert(key.to_string(), Some(v.clone()));
        Ok(Some(v))
    }

    pub fn const_int(&self, name: &str) -> Option<i128> {
        match self.const_value(name) {
            Ok(Some(Value::Int(i, _))) => Some(i),
            _ => None,
        }
    }

    fn self_ty(&self, env: &Env<'a>) -> Option<String> {
        lookup(env, "Self").and_then(|c| match &*c.borrow() {
            Value::Str(s) => Some(s.to_string()),
            _ => None,
        })
    }

    fn path_segments(&self, p: &syn::Path, env: &Env<'a>) -> Vec<String> {
        let mut segs: Vec<String> = p.segments.iter().map(|s| s.ident.to_string()).collect();
        if segs.first().is_some_and(|s| s == "Self") {
            if let Some(t) = self.self_ty(env) {
                segs[0] = t;
            }
        }
        // Drop module qualifiers that carry no type information.
        while segs.len() > 1 && matches!(segs[0].as_str(), "crate" | "self" | "super" | "std" | "core" | "alloc") {
            segs.remove(0);
        }
        segs
    }

    /// Resolve a path used as a value.
    fn eval_path(&self, path: &syn::Path, env: &Env<'a>) -> R<'a> {
        let segs = self.path_segments(path, env);
        if segs.len() == 1 {
            let n = segs[0].as_str();
            if let Some(c) = lookup(env, n) {
                return Ok(c.borrow().clone());
            }
            if let Some(v) = self.const_value(n)? {
                return Ok(v);
            }
            if let Some(f) = self.fns.get(n) {
                return Ok(Value::Func(Rc::new(Func::User(f.clone()))));
            }
            if let Some(d) = self.structs.get(n) {
                return Ok(if d.tuple { Value::Func(Rc::new(Func::TupleStruct(Rc::from(n)))) } else { Value::Struct(Rc::from(n), vec![]) });
            }
            match n {
                "None" => return Ok(Value::none()),
                "Some" => return Ok(Value::Func(Rc::new(Func::Variant("Option".into(), "Some".into())))),
                "Ok" => return Ok(Value::Func(Rc::new(Func::Variant("Result".into(), "Ok".into())))),
                "Err" => return Ok(Value::Func(Rc::new(Func::Variant("Result".into(), "Err".into())))),
                _ => {}
            }
            if let Some(v) = native::global_const(n) {
                return Ok(v);
            }
            if native::is_fn(n) {
                return Ok(Value::Func(Rc::new(Func::Native(Rc::from(n)))));
            }
            return Err(format!("cannot find `{n}` in this scope").into());
        }
        let item = segs[segs.len() - 1].as_str();
        let ty = segs[segs.len() - 2].as_str();
        let key = format!("{ty}::{item}");
        if let Some(e) = self.enums.get(ty) {
            if let Some((_, arity, _)) = e.variants.iter().find(|(v, _, _)| v == item) {
                if *arity >= 1_000_000 {
                    return Err(format!("struct-like enum variant `{key}` is not supported").into());
                }
                return Ok(if *arity == 0 { Value::Enum(Rc::from(ty), Rc::from(item), vec![]) } else { Value::Func(Rc::new(Func::Variant(Rc::from(ty), Rc::from(item)))) });
            }
        }
        if let Some(v) = self.const_value(&key)? {
            return Ok(v);
        }
        if let Some(f) = self.methods.get(&key) {
            return Ok(Value::Func(Rc::new(Func::User(f.clone()))));
        }
        match (ty, item) {
            ("Option", "None") => return Ok(Value::none()),
            ("Option", "Some") => return Ok(Value::Func(Rc::new(Func::Variant("Option".into(), "Some".into())))),
            ("Result", "Ok") => return Ok(Value::Func(Rc::new(Func::Variant("Result".into(), "Ok".into())))),
            ("Result", "Err") => return Ok(Value::Func(Rc::new(Func::Variant("Result".into(), "Err".into())))),
            _ => {}
        }
        if let Some(v) = native::assoc_const(ty, item) {
            return Ok(v);
        }
        if native::is_fn(&key) {
            return Ok(Value::Func(Rc::new(Func::Native(Rc::from(key.as_str())))));
        }
        // A module path (`common::op`): the last segment names the item.
        if !self.is_type(ty) && ty.chars().next().is_some_and(|c| c.is_lowercase()) {
            let one = syn::Path::from(path.segments.last().expect("segment").ident.clone());
            return self.eval_path(&one, env);
        }
        Err(format!("cannot find `{key}`").into())
    }

    fn is_type(&self, n: &str) -> bool {
        self.structs.contains_key(n) || self.enums.contains_key(n) || native::is_type(n)
    }

    pub fn enum_discr(&self, ty: &str, var: &str) -> Option<i128> {
        self.enums.get(ty)?.variants.iter().find(|(v, _, _)| v == var).and_then(|(_, _, d)| *d)
    }

    // ---- expressions --------------------------------------------------------

    fn loc(&self, e: &Expr) -> String {
        let s = e.span().start();
        format!("{}:{}:{}", self.name, s.line, s.column + 1)
    }

    pub fn eval(&self, e: &'a Expr, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a> {
        match self.eval_inner(e, env, cx) {
            Err(Ctl::Err(m)) => Err(Ctl::Fatal(format!("{}: {m}", self.loc(e)))),
            r => r,
        }
    }

    fn lit(&self, l: &Lit) -> R<'a> {
        Ok(match l {
            Lit::Int(i) => {
                let v = i.base10_parse::<i128>().map_err(|e| e.to_string())?;
                match i.suffix() {
                    "" => Value::Int(v, None),
                    "f32" | "f64" => Value::Float(v as f64),
                    s => {
                        let t = IntTy::parse(s).ok_or_else(|| format!("unsupported literal suffix `{s}`"))?;
                        if !t.fits(v) {
                            return Err(format!("literal {v} out of range for `{s}`").into());
                        }
                        Value::Int(v, Some(t))
                    }
                }
            }
            Lit::Float(f) => Value::Float(f.base10_parse::<f64>().map_err(|e| e.to_string())?),
            Lit::Str(s) => Value::str(s.value()),
            Lit::Char(c) => Value::Char(c.value()),
            Lit::Bool(b) => Value::Bool(b.value),
            Lit::Byte(b) => Value::Int(i128::from(b.value()), Some(IntTy::U8)),
            _ => return Err("unsupported literal".into()),
        })
    }

    fn eval_inner(&self, e: &'a Expr, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a> {
        match e {
            Expr::Lit(l) => self.lit(&l.lit),
            Expr::Path(p) => self.eval_path(&p.path, env),
            Expr::Paren(p) => self.eval(&p.expr, env, cx),
            Expr::Group(g) => self.eval(&g.expr, env, cx),
            Expr::Unary(u) => {
                let v = self.eval(&u.expr, env, cx)?;
                match u.op {
                    UnOp::Deref(_) => Ok(self.deref(v)),
                    UnOp::Not(_) => Ok(ops::not(&self.deref(v))?),
                    UnOp::Neg(_) => Ok(ops::neg(&self.deref(v))?),
                    _ => Err("unsupported unary operator".into()),
                }
            }
            Expr::Binary(b) => self.binary(b, env, cx),
            Expr::Assign(a) => {
                let v = self.eval(&a.right, env, cx)?;
                let v = self.deref(v);
                self.assign(&a.left, v, env, cx)?;
                Ok(Value::Unit)
            }
            Expr::Cast(c) => {
                let v = self.eval(&c.expr, env, cx)?;
                let v = self.deref(v);
                let name = type_key(&c.ty).ok_or("unsupported cast target")?;
                Ok(ops::cast(&v, &name, |t, v| self.enum_discr(t, v))?)
            }
            Expr::Call(c) => self.call(c, env, cx),
            Expr::MethodCall(m) => self.method_call(m, env, cx),
            Expr::Field(f) => {
                let b = self.eval(&f.base, env, cx)?;
                let b = self.deref(b);
                self.field(&b, &f.member, cx)
            }
            Expr::Index(ix) => {
                let b = self.eval(&ix.expr, env, cx)?;
                let b = self.deref(b);
                let i = self.eval(&ix.index, env, cx)?;
                let i = self.deref(i);
                Ok(index_value(&b, &i)?)
            }
            Expr::Tuple(t) => {
                if t.elems.is_empty() {
                    return Ok(Value::Unit);
                }
                Ok(Value::Tuple(t.elems.iter().map(|x| self.eval(x, env, cx)).collect::<R<Vec<_>>>()?))
            }
            Expr::Array(a) => Ok(Value::Array(a.elems.iter().map(|x| self.eval(x, env, cx)).collect::<R<Vec<_>>>()?)),
            Expr::Repeat(r) => {
                let v = self.eval(&r.expr, env, cx)?;
                let n = self.eval(&r.len, env, cx)?;
                let Value::Int(n, _) = self.deref(n) else { return Err("array length must be an integer".into()) };
                Ok(Value::Array(vec![self.deref(v); usize::try_from(n).map_err(|_| "bad array length")?]))
            }
            Expr::Struct(s) => self.struct_lit(s, env, cx),
            Expr::Block(b) => {
                let label = b.label.as_ref().map(|l| l.name.ident.to_string());
                match self.eval_block(&b.block, env, cx) {
                    Err(Ctl::Break(l, v)) if label.is_some() && l == label => Ok(v),
                    r => r,
                }
            }
            Expr::Unsafe(u) => self.eval_block(&u.block, env, cx),
            Expr::If(i) => {
                let mut scope = env.clone();
                if self.cond(&i.cond, &mut scope, cx)? {
                    self.eval_block(&i.then_branch, &mut scope, cx)
                } else if let Some((_, els)) = &i.else_branch {
                    self.eval(els, env, cx)
                } else {
                    Ok(Value::Unit)
                }
            }
            Expr::Match(m) => {
                let v = self.eval(&m.expr, env, cx)?;
                let v = self.deref(v);
                for arm in &m.arms {
                    let mut scope = env.clone();
                    if self.bind_pat(&arm.pat, &v, &mut scope, cx)? {
                        if let Some((_, g)) = &arm.guard {
                            if !self.cond(g, &mut scope, cx)? {
                                continue;
                            }
                        }
                        return self.eval(&arm.body, &mut scope, cx);
                    }
                }
                Err(format!("non-exhaustive match: no arm matches {}", crate::fmt::debug(&v, false)).into())
            }
            Expr::While(w) => {
                let label = w.label.as_ref().map(|l| l.name.ident.to_string());
                loop {
                    let mut scope = env.clone();
                    if !self.cond(&w.cond, &mut scope, cx)? {
                        return Ok(Value::Unit);
                    }
                    match self.eval_block(&w.body, &mut scope, cx) {
                        Ok(_) => {}
                        Err(Ctl::Break(l, _)) if l.is_none() || l == label => return Ok(Value::Unit),
                        Err(Ctl::Continue(l)) if l.is_none() || l == label => {}
                        Err(c) => return Err(c),
                    }
                }
            }
            Expr::Loop(l) => {
                let label = l.label.as_ref().map(|l| l.name.ident.to_string());
                loop {
                    let mut scope = env.clone();
                    match self.eval_block(&l.body, &mut scope, cx) {
                        Ok(_) => {}
                        Err(Ctl::Break(l, v)) if l.is_none() || l == label => return Ok(v),
                        Err(Ctl::Continue(l)) if l.is_none() || l == label => {}
                        Err(c) => return Err(c),
                    }
                }
            }
            Expr::ForLoop(f) => self.for_loop(f, env, cx),
            Expr::Break(b) => {
                let v = match &b.expr {
                    Some(x) => self.eval(x, env, cx)?,
                    None => Value::Unit,
                };
                Err(Ctl::Break(b.label.as_ref().map(|l| l.ident.to_string()), v))
            }
            Expr::Continue(c) => Err(Ctl::Continue(c.label.as_ref().map(|l| l.ident.to_string()))),
            Expr::Return(r) => {
                let v = match &r.expr {
                    Some(x) => self.eval(x, env, cx)?,
                    None => Value::Unit,
                };
                Err(Ctl::Return(v))
            }
            Expr::Range(r) => {
                let s = match &r.start {
                    Some(x) => Some(self.eval(x, env, cx)?),
                    None => None,
                };
                let en = match &r.end {
                    Some(x) => Some(self.eval(x, env, cx)?),
                    None => None,
                };
                let int = |v: Option<Value<'a>>| -> R<'a, Option<(i128, Option<IntTy>)>> {
                    match v.map(|v| self.deref(v)) {
                        None => Ok(None),
                        Some(Value::Int(i, t)) => Ok(Some((i, t))),
                        Some(o) => Err(format!("range bounds must be integers, found {}", o.type_name()).into()),
                    }
                };
                let (s, en) = (int(s)?, int(en)?);
                let ty = match (s.and_then(|x| x.1), en.and_then(|x| x.1)) {
                    (Some(a), Some(b)) if a != b => return Err(format!("mismatched range bound types: {} and {}", a.name(), b.name()).into()),
                    (Some(a), _) | (_, Some(a)) => Some(a),
                    _ => None,
                };
                Ok(Value::Range(Box::new(RangeV { start: s.map(|x| x.0), end: en.map(|x| x.0), inclusive: matches!(r.limits, syn::RangeLimits::Closed(_)), ty })))
            }
            Expr::Closure(c) => {
                let params = c.inputs.iter().map(|p| match p {
                    Pat::Type(pt) => (&*pt.pat, Ty::from_syn(&pt.ty)),
                    p => (p, Ty::Any),
                });
                let ret = match &c.output {
                    syn::ReturnType::Type(_, t) => Ty::from_syn(t),
                    _ => Ty::Any,
                };
                Ok(Value::Closure(Rc::new(Closure { params: params.collect(), ret, body: &c.body, env: env.clone() })))
            }
            Expr::Reference(r) => {
                if r.mutability.is_some() {
                    match self.place(&r.expr, env, cx) {
                        Ok(mut p) => {
                            self.autoderef(&mut p);
                            if matches!(self.read_place(&p), Value::Nat(_)) {
                                // Natives carry identity: the handle itself is the reference.
                                return Ok(self.read_place(&p));
                            }
                            Ok(Value::Ref(Rc::new(p)))
                        }
                        Err(_) => self.eval(&r.expr, env, cx),
                    }
                } else {
                    self.eval(&r.expr, env, cx)
                }
            }
            Expr::Try(t) => {
                let v = self.eval(&t.expr, env, cx)?;
                match self.deref(v) {
                    Value::Enum(ty, var, mut p) if &*ty == "Result" => {
                        if &*var == "Ok" {
                            Ok(p.pop().unwrap_or(Value::Unit))
                        } else {
                            Err(Ctl::Return(Value::Enum(ty, var, p)))
                        }
                    }
                    Value::Enum(ty, var, mut p) if &*ty == "Option" => {
                        if &*var == "Some" {
                            Ok(p.pop().unwrap_or(Value::Unit))
                        } else {
                            Err(Ctl::Return(Value::none()))
                        }
                    }
                    o => Err(format!("`?` needs a Result or Option, found {}", o.type_name()).into()),
                }
            }
            Expr::Let(_) => Err("`let` is only supported as an `if`/`while` condition".into()),
            Expr::Macro(m) => Err(format!("macro `{}!` is not supported by the .rip evaluator", m.mac.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default()).into()),
            other => Err(format!("unsupported expression ({})", expr_kind(other)).into()),
        }
    }

    /// An `if` / `while` / guard condition, with `let` patterns and `&&` chains.
    fn cond(&self, e: &'a Expr, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a, bool> {
        match e {
            Expr::Let(l) => {
                let v = self.eval(&l.expr, env, cx)?;
                let v = self.deref(v);
                let mut scope = env.clone();
                if self.bind_pat(&l.pat, &v, &mut scope, cx)? {
                    *env = scope;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            Expr::Binary(b) if matches!(b.op, BinOp::And(_)) => Ok(self.cond(&b.left, env, cx)? && self.cond(&b.right, env, cx)?),
            Expr::Paren(p) => self.cond(&p.expr, env, cx),
            _ => match self.eval(e, env, cx)? {
                Value::Bool(b) => Ok(b),
                o => match self.deref(o) {
                    Value::Bool(b) => Ok(b),
                    o => Err(format!("condition must be bool, found {}", o.type_name()).into()),
                },
            },
        }
    }

    fn binary(&self, b: &'a syn::ExprBinary, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a> {
        use BinOp::*;
        match &b.op {
            And(_) => {
                let l = self.eval(&b.left, env, cx)?;
                let Value::Bool(l) = self.deref(l) else { return Err("`&&` needs bool operands".into()) };
                if !l {
                    return Ok(Value::Bool(false));
                }
                let r = self.eval(&b.right, env, cx)?;
                match self.deref(r) {
                    Value::Bool(r) => Ok(Value::Bool(r)),
                    _ => Err("`&&` needs bool operands".into()),
                }
            }
            Or(_) => {
                let l = self.eval(&b.left, env, cx)?;
                let Value::Bool(l) = self.deref(l) else { return Err("`||` needs bool operands".into()) };
                if l {
                    return Ok(Value::Bool(true));
                }
                let r = self.eval(&b.right, env, cx)?;
                match self.deref(r) {
                    Value::Bool(r) => Ok(Value::Bool(r)),
                    _ => Err("`||` needs bool operands".into()),
                }
            }
            AddAssign(_) | SubAssign(_) | MulAssign(_) | DivAssign(_) | RemAssign(_) | BitXorAssign(_) | BitAndAssign(_) | BitOrAssign(_) | ShlAssign(_) | ShrAssign(_) => {
                let cur = self.eval(&b.left, env, cx)?;
                let cur = self.deref(cur);
                let r = self.eval(&b.right, env, cx)?;
                let r = self.deref(r);
                let base: BinOp = match &b.op {
                    AddAssign(_) => Add(Default::default()),
                    SubAssign(_) => Sub(Default::default()),
                    MulAssign(_) => Mul(Default::default()),
                    DivAssign(_) => Div(Default::default()),
                    RemAssign(_) => Rem(Default::default()),
                    BitXorAssign(_) => BitXor(Default::default()),
                    BitAndAssign(_) => BitAnd(Default::default()),
                    BitOrAssign(_) => BitOr(Default::default()),
                    ShlAssign(_) => Shl(Default::default()),
                    ShrAssign(_) => Shr(Default::default()),
                    _ => unreachable!(),
                };
                let v = ops::binop(&base, &cur, &r)?;
                self.assign(&b.left, v, env, cx)?;
                Ok(Value::Unit)
            }
            op => {
                let l = self.eval(&b.left, env, cx)?;
                let l = self.deref(l);
                let r = self.eval(&b.right, env, cx)?;
                let r = self.deref(r);
                Ok(ops::binop(op, &l, &r)?)
            }
        }
    }

    fn assign(&self, lhs: &'a Expr, v: Value<'a>, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a, ()> {
        match lhs {
            Expr::Tuple(t) => {
                let Value::Tuple(xs) = v else { return Err("destructuring assignment needs a tuple".into()) };
                if xs.len() != t.elems.len() {
                    return Err("destructuring assignment arity mismatch".into());
                }
                for (l, x) in t.elems.iter().zip(xs) {
                    self.assign(l, x, env, cx)?;
                }
                Ok(())
            }
            Expr::Path(p) if p.path.is_ident("_") => Ok(()),
            Expr::Infer(_) => Ok(()),
            Expr::Paren(p) => self.assign(&p.expr, v, env, cx),
            _ => {
                let mut p = self.place(lhs, env, cx)?;
                if !matches!(lhs, Expr::Path(_)) {
                    // Field / index writes go through references held by the base.
                } else if !matches!(lhs, Expr::Unary(_)) {
                    // `x = v` rebinds the variable itself, never what a reference in it points to.
                    p.path.clear();
                }
                self.write_place(&p, v).map_err(Ctl::Err)
            }
        }
    }

    fn field(&self, b: &Value<'a>, m: &Member, cx: &mut Cx<'_>) -> R<'a> {
        match (b, m) {
            (Value::Struct(n, fs), Member::Named(i)) => {
                let k = i.to_string();
                fs.iter().find(|(f, _)| **f == *k).map(|(_, v)| v.clone()).ok_or_else(|| format!("`{n}` has no field `{k}`").into())
            }
            (Value::Struct(_, fs), Member::Unnamed(i)) => fs.iter().find(|(f, _)| **f == *i.index.to_string()).map(|(_, v)| v.clone()).ok_or_else(|| "no such tuple field".into()),
            (Value::Tuple(xs), Member::Unnamed(i)) => xs.get(i.index as usize).cloned().ok_or_else(|| "tuple index out of range".into()),
            (Value::Nat(n), Member::Named(i)) => native::field(self, n, &i.to_string(), cx)?.ok_or_else(|| format!("`{}` has no field `{i}`", n.type_name()).into()),
            (o, Member::Named(i)) => Err(format!("`{}` has no field `{i}`", o.type_name()).into()),
            (o, _) => Err(format!("`{}` has no tuple fields", o.type_name()).into()),
        }
    }

    fn struct_lit(&self, s: &'a syn::ExprStruct, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a> {
        let segs = self.path_segments(&s.path, env);
        let name = segs.last().cloned().unwrap_or_default();
        if segs.len() >= 2 && self.enums.contains_key(&segs[segs.len() - 2]) {
            return Err(format!("struct-like enum variant `{}` is not supported", segs.join("::")).into());
        }
        let mut given: Vec<(Rc<str>, Value<'a>)> = Vec::new();
        for f in &s.fields {
            let key = match &f.member {
                Member::Named(i) => i.to_string(),
                Member::Unnamed(i) => i.index.to_string(),
            };
            let v = self.eval(&f.expr, env, cx)?;
            given.push((Rc::from(key.as_str()), self.deref(v)));
        }
        let base = match &s.rest {
            Some(r) => {
                let v = self.eval(r, env, cx)?;
                match self.deref(v) {
                    Value::Struct(_, fs) => fs,
                    _ => return Err("struct update base is not a struct".into()),
                }
            }
            None => vec![],
        };
        let Some(def) = self.structs.get(&name) else {
            if native::is_struct(&name) {
                return Ok(Value::Struct(Rc::from(name.as_str()), given));
            }
            return Err(format!("unknown struct `{name}`").into());
        };
        let mut fields = Vec::with_capacity(def.fields.len());
        for (fname, ty) in &def.fields {
            let v = given.iter().position(|(k, _)| **k == **fname).map(|i| given.swap_remove(i).1).or_else(|| base.iter().find(|(k, _)| **k == **fname).map(|(_, v)| v.clone()));
            let v = v.ok_or_else(|| format!("missing field `{fname}` in `{name}`"))?;
            fields.push((Rc::from(fname.as_str()), ops::coerce(v, ty).map_err(|e| format!("field `{name}.{fname}`: {e}"))?));
        }
        if let Some((k, _)) = given.first() {
            return Err(format!("`{name}` has no field `{k}`").into());
        }
        Ok(Value::Struct(Rc::from(name.as_str()), fields))
    }

    fn for_loop(&self, f: &'a syn::ExprForLoop, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a> {
        let label = f.label.as_ref().map(|l| l.name.ident.to_string());
        let it = self.eval(&f.expr, env, cx)?;
        let it = self.deref(it);
        let run = |item: Value<'a>, cx: &mut Cx<'_>| -> R<'a, bool> {
            let mut scope = env.clone();
            if !self.bind_pat(&f.pat, &item, &mut scope, cx)? {
                return Err("refutable pattern in `for`".into());
            }
            match self.eval_block(&f.body, &mut scope, cx) {
                Ok(_) => Ok(true),
                Err(Ctl::Break(l, _)) if l.is_none() || l == label => Ok(false),
                Err(Ctl::Continue(l)) if l.is_none() || l == label => Ok(true),
                Err(c) => Err(c),
            }
        };
        match it {
            Value::Range(r) => {
                let Some(start) = r.start else { return Err("`for` over a range needs a start".into()) };
                let mut i = start;
                loop {
                    match r.end {
                        Some(e) if (r.inclusive && i > e) || (!r.inclusive && i >= e) => break,
                        _ => {}
                    }
                    if !run(Value::Int(i, r.ty), cx)? {
                        break;
                    }
                    i += 1;
                }
            }
            other => {
                for item in crate::builtin::items(&other)? {
                    if !run(item, cx)? {
                        break;
                    }
                }
            }
        }
        Ok(Value::Unit)
    }

    // ---- blocks and statements -----------------------------------------------

    pub fn eval_block(&self, b: &'a Block, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a> {
        let saved = env.clone();
        let r = self.exec_stmts(&b.stmts, env, cx);
        *env = saved;
        r
    }

    /// The statements of a block, in order. `let wg = Workgroup::new(&mut b, arch)?;`
    /// opens the sealed driver over `b` and runs the following statements
    /// with it, up to (not including) the first one that mentions `b` again.
    fn exec_stmts(&self, stmts: &'a [Stmt], env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a> {
        let mut last = Value::Unit;
        let n = stmts.len();
        let mut i = 0;
        while i < n {
            if let Some(ws) = workgroup_let(&stmts[i]) {
                let end = (i + 1..n).find(|&k| mentions(&stmts[k], &ws.builder)).unwrap_or(n);
                let v = self.workgroup_scope(&ws, &stmts[i + 1..end], env, cx)?;
                last = if end == n { v } else { Value::Unit };
                i = end;
                continue;
            }
            last = Value::Unit;
            match &stmts[i] {
                Stmt::Local(l) => {
                    let (pat, ty) = match &l.pat {
                        Pat::Type(pt) => (&*pt.pat, Ty::from_syn(&pt.ty)),
                        p => (p, Ty::Any),
                    };
                    let Some(init) = &l.init else {
                        // `let x;` — declared, assigned later.
                        if let Pat::Ident(pi) = pat {
                            *env = bind(env, &pi.ident.to_string(), Value::Unit);
                            i += 1;
                            continue;
                        }
                        return Err("`let` without initializer needs a simple name".into());
                    };
                    let v = self.eval(&init.expr, env, cx)?;
                    let v = ops::coerce(v, &ty).map_err(|e| Ctl::Fatal(format!("{}: {e}", self.loc(&init.expr))))?;
                    let mut scope = env.clone();
                    if self.bind_pat(pat, &self.deref_for_pat(pat, v), &mut scope, cx)? {
                        *env = scope;
                    } else if let Some((_, els)) = &init.diverge {
                        return match self.eval(els, env, cx) {
                            Ok(_) => Err(Ctl::Fatal(format!("{}: `let ... else` block did not diverge", self.loc(els)))),
                            Err(e) => Err(e),
                        };
                    } else {
                        return Err(Ctl::Fatal(format!("{}: refutable pattern in `let`", self.loc(&init.expr))));
                    }
                }
                Stmt::Item(item) => match item {
                    Item::Fn(f) => {
                        let n = f.sig.ident.to_string();
                        let def = Rc::new(make_fn(&n, &f.sig, &f.block, None));
                        *env = bind(env, &n, Value::Func(Rc::new(Func::User(def))));
                    }
                    Item::Const(c) => {
                        let mut e2 = env.clone();
                        let v = self.eval(&c.expr, &mut e2, cx)?;
                        let v = ops::coerce(self.deref(v), &Ty::from_syn(&c.ty))?;
                        *env = bind(env, &c.ident.to_string(), v);
                    }
                    _ => {}
                },
                Stmt::Expr(e, semi) => {
                    let v = self.eval(e, env, cx)?;
                    if i == n - 1 && semi.is_none() {
                        last = v;
                    }
                }
                Stmt::Macro(_) => return Err("unelaborated macro statement".into()),
            }
            i += 1;
        }
        Ok(last)
    }

    fn workgroup_scope(&self, ws: &WgLet<'a>, body: &'a [Stmt], env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a> {
        if cx.d.is_some() {
            return Err(Ctl::Fatal(format!("{}: a Workgroup scope is already open here", self.loc(ws.call))));
        }
        let arch = self.eval(ws.arch, env, cx)?;
        let Value::Nat(Nat::Arch(arch)) = self.deref(arch) else { return Err(Ctl::Fatal(format!("{}: Workgroup::new takes an Arch", self.loc(ws.arch)))) };
        let bv = lookup(env, &ws.builder).map(|c| c.borrow().clone()).ok_or_else(|| Ctl::Fatal(format!("{}: cannot find `{}`", self.loc(ws.call), ws.builder)))?;
        let Value::Nat(Nat::Builder(cell)) = self.deref(bv) else { return Err(Ctl::Fatal(format!("{}: Workgroup::new borrows a Builder (`&mut {}`)", self.loc(ws.call), ws.builder))) };
        let Some(mut bld) = cell.borrow_mut().take() else { return Err(Ctl::Fatal(format!("{}: the builder already belongs to a Workgroup", self.loc(ws.call)))) };
        let mut scope = env.clone();
        if !self.bind_pat(ws.pat, &Value::Nat(Nat::Wg), &mut scope, cx)? {
            *cell.borrow_mut() = Some(bld);
            return Err(Ctl::Fatal(format!("{}: the Workgroup binding must be a plain name", self.loc(ws.call))));
        }
        *env = scope;
        let r = crate::driver::drive(arch, &mut bld, &mut |cx2| self.exec_stmts(body, env, cx2));
        *cell.borrow_mut() = Some(bld);
        match r {
            Ok(r) => r,
            // `Workgroup::new(..)?`: the driver refused the backend.
            Err(e) => Err(Ctl::Return(Value::err(e))),
        }
    }

    /// `let (a, b) = pair;` binds through references; a plain `let r = x;`
    /// keeps a reference value as it is.
    fn deref_for_pat(&self, pat: &Pat, v: Value<'a>) -> Value<'a> {
        match pat {
            Pat::Ident(_) | Pat::Wild(_) => v,
            _ => self.deref(v),
        }
    }

    // ---- patterns -----------------------------------------------------------------

    pub fn bind_pat(&self, p: &'a Pat, v: &Value<'a>, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a, bool> {
        match p {
            Pat::Wild(_) => Ok(true),
            Pat::Ident(pi) => {
                let name = pi.ident.to_string();
                if pi.subpat.is_none() && pi.by_ref.is_none() {
                    if name == "None" {
                        return Ok(matches!(self.deref(v.clone()), Value::Enum(t, var, _) if &*t == "Option" && &*var == "None"));
                    }
                    if name.chars().next().is_some_and(|c| c.is_uppercase()) && self.consts.contains_key(&name) {
                        let c = self.const_value(&name)?.ok_or("constant vanished")?;
                        return Ok(values_eq(&self.deref(v.clone()), &c));
                    }
                }
                if let Some((_, sub)) = &pi.subpat {
                    if !self.bind_pat(sub, v, env, cx)? {
                        return Ok(false);
                    }
                }
                *env = bind(env, &name, v.clone());
                Ok(true)
            }
            Pat::Lit(l) => {
                let lit = self.lit(&l.lit)?;
                Ok(values_eq(&self.deref(v.clone()), &lit))
            }
            Pat::Range(r) => {
                let v = self.deref(v.clone());
                let mut env2 = env.clone();
                let lo = match &r.start {
                    Some(e) => Some(self.eval(e, &mut env2, cx)?),
                    None => None,
                };
                let hi = match &r.end {
                    Some(e) => Some(self.eval(e, &mut env2, cx)?),
                    None => None,
                };
                if let Some(lo) = lo {
                    if ops::compare(&v, &lo)?.is_lt() {
                        return Ok(false);
                    }
                }
                if let Some(hi) = hi {
                    let o = ops::compare(&v, &hi)?;
                    if o.is_gt() || (o.is_eq() && matches!(r.limits, syn::RangeLimits::HalfOpen(_))) {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Pat::Path(pp) => {
                let c = self.eval_path(&pp.path, env)?;
                Ok(values_eq(&self.deref(v.clone()), &c))
            }
            Pat::Tuple(t) => {
                let v = self.deref(v.clone());
                let xs = match &v {
                    Value::Tuple(xs) => xs.clone(),
                    Value::Unit => vec![],
                    _ => return Ok(false),
                };
                self.bind_seq(t.elems.iter().collect(), &xs, env, cx)
            }
            Pat::Slice(s) => {
                let v = self.deref(v.clone());
                let Value::Array(xs) = &v else { return Ok(false) };
                self.bind_seq(s.elems.iter().collect(), xs, env, cx)
            }
            Pat::TupleStruct(ts) => {
                let v = self.deref(v.clone());
                let segs = self.path_segments(&ts.path, env);
                let name = segs.last().cloned().unwrap_or_default();
                let elems: Vec<&'a Pat> = ts.elems.iter().collect();
                match &v {
                    Value::Enum(ty, var, payload) => {
                        if **var != *name || (segs.len() >= 2 && **ty != *segs[segs.len() - 2]) {
                            return Ok(false);
                        }
                        self.bind_seq(elems, payload, env, cx)
                    }
                    Value::Struct(n, fs) if **n == *name => {
                        let xs: Vec<Value<'a>> = fs.iter().map(|(_, x)| x.clone()).collect();
                        self.bind_seq(elems, &xs, env, cx)
                    }
                    _ => Ok(false),
                }
            }
            Pat::Struct(sp) => {
                let v = self.deref(v.clone());
                let segs = self.path_segments(&sp.path, env);
                let name = segs.last().cloned().unwrap_or_default();
                let Value::Struct(n, fs) = &v else { return Ok(false) };
                if **n != *name {
                    return Ok(false);
                }
                for fp in &sp.fields {
                    let key = match &fp.member {
                        Member::Named(i) => i.to_string(),
                        Member::Unnamed(i) => i.index.to_string(),
                    };
                    let Some((_, x)) = fs.iter().find(|(k, _)| **k == *key) else { return Ok(false) };
                    if !self.bind_pat(&fp.pat, x, env, cx)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Pat::Or(o) => {
                for c in &o.cases {
                    let mut scope = env.clone();
                    if self.bind_pat(c, v, &mut scope, cx)? {
                        *env = scope;
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Pat::Reference(r) => self.bind_pat(&r.pat, v, env, cx),
            Pat::Paren(pp) => self.bind_pat(&pp.pat, v, env, cx),
            Pat::Type(pt) => self.bind_pat(&pt.pat, v, env, cx),
            other => Err(format!("unsupported pattern ({})", pat_kind(other)).into()),
        }
    }

    fn bind_seq(&self, pats: Vec<&'a Pat>, xs: &[Value<'a>], env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a, bool> {
        let rest = pats.iter().position(|p| matches!(p, Pat::Rest(_)));
        match rest {
            None => {
                if pats.len() != xs.len() {
                    return Ok(false);
                }
                for (p, x) in pats.iter().zip(xs) {
                    if !self.bind_pat(p, x, env, cx)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Some(r) => {
                let (head, tail) = (&pats[..r], &pats[r + 1..]);
                if head.len() + tail.len() > xs.len() {
                    return Ok(false);
                }
                for (p, x) in head.iter().zip(xs) {
                    if !self.bind_pat(p, x, env, cx)? {
                        return Ok(false);
                    }
                }
                for (p, x) in tail.iter().zip(&xs[xs.len() - tail.len()..]) {
                    if !self.bind_pat(p, x, env, cx)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    // ---- calls -------------------------------------------------------------------

    fn eval_args(&self, args: &'a syn::punctuated::Punctuated<Expr, syn::Token![,]>, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a, Vec<Value<'a>>> {
        args.iter().map(|a| self.eval(a, env, cx)).collect()
    }

    fn call(&self, c: &'a syn::ExprCall, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a> {
        if let Expr::Path(p) = &*c.func {
            if p.path.segments.len() == 1 {
                let name = p.path.segments[0].ident.to_string();
                if name.starts_with("__") {
                    if let Some(r) = self.internal_call(&name, c, env, cx) {
                        return r;
                    }
                }
            }
        }
        let f = self.eval(&c.func, env, cx)?;
        let args = self.eval_args(&c.args, env, cx)?;
        self.call_value(&f, args, cx)
    }

    pub fn call_value(&self, f: &Value<'a>, args: Vec<Value<'a>>, cx: &mut Cx<'_>) -> R<'a> {
        match self.deref(f.clone()) {
            Value::Closure(c) => self.call_closure(&c, args, cx),
            Value::Func(func) => match &*func {
                Func::User(def) => self.call_fn(def, args, cx),
                Func::Native(n) => native::call_fn(self, n, args, cx),
                Func::Variant(t, v) => Ok(Value::Enum(t.clone(), v.clone(), args.into_iter().map(|a| self.deref(a)).collect())),
                Func::TupleStruct(n) => {
                    let def = self.structs.get(&**n).ok_or("unknown tuple struct")?;
                    if def.fields.len() != args.len() {
                        return Err(format!("`{n}` takes {} fields", def.fields.len()).into());
                    }
                    let mut fs = Vec::new();
                    for ((fname, ty), a) in def.fields.iter().zip(args) {
                        fs.push((Rc::from(fname.as_str()), ops::coerce(self.deref(a), ty)?));
                    }
                    Ok(Value::Struct(n.clone(), fs))
                }
            },
            o => Err(format!("`{}` is not callable", o.type_name()).into()),
        }
    }

    pub fn call_closure(&self, c: &Rc<Closure<'a>>, args: Vec<Value<'a>>, cx: &mut Cx<'_>) -> R<'a> {
        if args.len() != c.params.len() {
            return Err(format!("closure takes {} arguments, got {}", c.params.len(), args.len()).into());
        }
        let mut env = c.env.clone();
        for ((p, ty), a) in c.params.iter().zip(args) {
            let a = ops::coerce(a, ty)?;
            let a = self.deref_for_pat(p, a);
            if !self.bind_pat(p, &a, &mut env, cx)? {
                return Err("closure argument does not match its pattern".into());
            }
        }
        let r = match self.eval(c.body, &mut env, cx) {
            Ok(v) => v,
            Err(Ctl::Return(v)) => v,
            Err(e) => return Err(e),
        };
        Ok(ops::coerce(r, &c.ret)?)
    }

    pub fn call_fn(&self, f: &Rc<FnDef<'a>>, args: Vec<Value<'a>>, cx: &mut Cx<'_>) -> R<'a> {
        if self.depth.get() >= MAX_DEPTH {
            return Err(format!("recursion limit reached in `{}`", f.name).into());
        }
        let want = f.params.len() + usize::from(f.recv != 0);
        if args.len() != want {
            return Err(format!("`{}` takes {want} arguments, got {}", f.name, args.len()).into());
        }
        let mut env: Env<'a> = None;
        if let Some(t) = &f.self_ty {
            env = bind(&env, "Self", Value::Str(t.clone()));
        }
        let mut it = args.into_iter();
        if f.recv != 0 {
            let s = it.next().expect("receiver");
            env = bind(&env, "self", if f.recv == 3 { s } else { self.deref(s) });
        }
        for ((p, ty), a) in f.params.iter().zip(it) {
            let a = ops::coerce(a, ty).map_err(|e| format!("argument of `{}`: {e}", f.name))?;
            let a = self.deref_for_pat(p, a);
            if !self.bind_pat(p, &a, &mut env, cx)? {
                return Err(format!("argument of `{}` does not match its pattern", f.name).into());
            }
        }
        self.depth.set(self.depth.get() + 1);
        let r = self.eval_block(f.block, &mut env, cx);
        self.depth.set(self.depth.get() - 1);
        let v = match r {
            Ok(v) => v,
            Err(Ctl::Return(v)) => v,
            Err(e) => return Err(e),
        };
        Ok(ops::coerce(v, &f.ret).map_err(|e| format!("result of `{}`: {e}", f.name))?)
    }

    fn method_key(&self, v: &Value<'a>) -> Vec<String> {
        match v {
            Value::Struct(n, _) => vec![n.to_string()],
            Value::Enum(t, _, _) => vec![t.to_string()],
            Value::Nat(n) => vec![n.type_name().to_string()],
            Value::Int(_, Some(t)) => vec![t.name().to_string()],
            Value::Int(_, None) => ["i32", "u32", "u8", "u64", "usize"].iter().map(|s| s.to_string()).collect(),
            Value::Str(_) => vec!["String".into(), "str".into()],
            Value::Array(_) => vec!["Vec".into()],
            Value::Bool(_) => vec!["bool".into()],
            Value::Float(_) => vec!["f64".into(), "f32".into()],
            _ => vec![],
        }
    }

    fn method_call(&self, m: &'a syn::ExprMethodCall, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a> {
        let name = m.method.to_string();
        let recv = self.eval(&m.receiver, env, cx)?;
        let d = self.deref(recv.clone());
        let targs = turbofish(m.turbofish.as_ref());
        // Script methods (impls and traits) take precedence.
        for ty in self.method_key(&d) {
            if let Some(f) = self.methods.get(&format!("{ty}::{name}")) {
                let f = f.clone();
                let selfv = if f.recv == 3 {
                    match recv {
                        Value::Ref(_) => recv,
                        _ => match d {
                            Value::Nat(_) => d,
                            _ => Value::Ref(Rc::new(self.place(&m.receiver, env, cx)?)),
                        },
                    }
                } else {
                    d
                };
                let mut args = vec![selfv];
                args.extend(self.eval_args(&m.args, env, cx)?);
                return self.call_fn(&f, args, cx);
            }
        }
        let args = self.eval_args(&m.args, env, cx)?;
        // Mutating methods of std containers go through the place.
        if crate::builtin::is_mutator(&d, &name) {
            let p = match recv {
                Value::Ref(r) => PlaceRef { cell: r.cell.clone(), path: r.path.clone() },
                _ => self.place(&m.receiver, env, cx)?,
            };
            let mut p = p;
            self.autoderef(&mut p);
            let args: Vec<Value<'a>> = args.into_iter().map(|a| self.deref(a)).collect();
            let r = self.with_place_mut(&p, |v| crate::builtin::call_mut(self, v, &name, args));
            return r.map_err(Ctl::Err);
        }
        if let Value::Nat(n) = &d {
            if let Some(r) = native::method(self, n, &name, &targs, args.clone(), cx)? {
                return Ok(r);
            }
        }
        if let Some(r) = crate::builtin::call(self, &d, &name, &targs, args, cx)? {
            return Ok(r);
        }
        Err(format!("no method `{name}` on `{}`", d.type_name()).into())
    }

    // ---- internal calls (elaborated macros) ----------------------------------------

    fn internal_call(&self, name: &str, c: &'a syn::ExprCall, env: &mut Env<'a>, cx: &mut Cx<'_>) -> Option<R<'a>> {
        Some(match name {
            "__fmt" => self.format_call(c, env, cx).map(Value::str),
            "__println" | "__eprintln" | "__print" | "__eprint" => {
                let s = if c.args.is_empty() { Ok(String::new()) } else { self.format_call(c, env, cx) };
                s.map(|s| {
                    if name.contains("err") || name == "__eprintln" {
                        eprintln!("{s}");
                    } else if name.ends_with("ln") {
                        println!("{s}");
                    } else {
                        print!("{s}");
                    }
                    Value::Unit
                })
            }
            "__panic" => {
                let m = self.eval_args(&c.args, env, cx).map(|a| a.first().map(|v| crate::fmt::display(&self.deref(v.clone())).unwrap_or_default()).unwrap_or_default());
                m.and_then(|m| Err(Ctl::Err(format!("panic: {m}"))))
            }
            "__assert" => (|| {
                let a = self.eval_args(&c.args, env, cx)?;
                match self.deref(a[0].clone()) {
                    Value::Bool(true) => Ok(Value::Unit),
                    Value::Bool(false) => Err(Ctl::Err(format!("assertion failed{}", a.get(1).map(|m| format!(": {}", crate::fmt::display(m).unwrap_or_default())).unwrap_or_default()))),
                    _ => Err(Ctl::Err("assert! needs a bool".into())),
                }
            })(),
            "__assert_eq" | "__assert_ne" => (|| {
                let a = self.eval_args(&c.args, env, cx)?;
                let (l, r) = (self.deref(a[0].clone()), self.deref(a[1].clone()));
                if values_eq(&l, &r) == (name == "__assert_eq") {
                    Ok(Value::Unit)
                } else {
                    Err(Ctl::Err(format!("assertion failed: `{}` vs `{}`", crate::fmt::debug(&l, false), crate::fmt::debug(&r, false))))
                }
            })(),
            "__unsupported" => {
                let msg = match c.args.first() {
                    Some(Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. })) => s.value(),
                    _ => "unsupported construct".into(),
                };
                Err(Ctl::Err(msg))
            }
            _ => return None,
        })
    }

    fn format_call(&self, c: &'a syn::ExprCall, env: &mut Env<'a>, cx: &mut Cx<'_>) -> R<'a, String> {
        let mut it = c.args.iter();
        let Some(Expr::Lit(syn::ExprLit { lit: Lit::Str(fmt), .. })) = it.next() else { return Err("format! needs a string literal".into()) };
        let pieces = crate::fmt::parse(&fmt.value())?;
        let mut positional: Vec<Value<'a>> = Vec::new();
        let mut named: Vec<(String, Value<'a>)> = Vec::new();
        for a in it {
            match a {
                Expr::Assign(asg) if matches!(&*asg.left, Expr::Path(p) if p.path.get_ident().is_some()) => {
                    let Expr::Path(p) = &*asg.left else { unreachable!() };
                    let v = self.eval(&asg.right, env, cx)?;
                    named.push((p.path.get_ident().expect("ident").to_string(), self.deref(v)));
                }
                a => {
                    let v = self.eval(a, env, cx)?;
                    positional.push(self.deref(v));
                }
            }
        }
        let mut out = String::new();
        let mut next = 0;
        for p in pieces {
            match p {
                crate::fmt::Piece::Lit(s) => out.push_str(&s),
                crate::fmt::Piece::Arg(target, spec) => {
                    let v = match target {
                        crate::fmt::ArgRef::Next => {
                            let v = positional.get(next).cloned().ok_or("format string has more `{}` than arguments")?;
                            next += 1;
                            v
                        }
                        crate::fmt::ArgRef::Index(i) => positional.get(i).cloned().ok_or_else(|| format!("format argument {i} is missing"))?,
                        crate::fmt::ArgRef::Name(n) => match named.iter().find(|(k, _)| *k == n) {
                            Some((_, v)) => v.clone(),
                            None => {
                                if let Some(cell) = lookup(env, &n) {
                                    let v = cell.borrow().clone();
                                    self.deref(v)
                                } else if let Some(v) = self.const_value(&n)? {
                                    v
                                } else if let Some(v) = native::global_const(&n) {
                                    v
                                } else {
                                    return Err(format!("cannot find `{n}` for format!").into());
                                }
                            }
                        },
                    };
                    out.push_str(&crate::fmt::render(&v, &spec)?);
                }
            }
        }
        if next < positional.len() && !positional.is_empty() {
            // Rust rejects unused arguments at compile time.
            let used_by_index = false;
            if !used_by_index && !crate::fmt::parse(&fmt.value())?.iter().any(|p| matches!(p, crate::fmt::Piece::Arg(crate::fmt::ArgRef::Index(_), _))) {
                return Err("format! has arguments no placeholder uses".into());
            }
        }
        Ok(out)
    }
}

/// Assigning an untyped integer into a typed slot adopts the slot's type.
fn adopt<'a>(old: &Value<'a>, new: Value<'a>) -> Value<'a> {
    match (old, new) {
        (Value::Int(_, Some(t)), Value::Int(v, None)) => Value::Int(v, Some(*t)),
        (_, n) => n,
    }
}

fn index_value<'a>(b: &Value<'a>, i: &Value<'a>) -> Result<Value<'a>, String> {
    match (b, i) {
        (Value::Array(xs), Value::Int(i, _)) => usize::try_from(*i).ok().and_then(|i| xs.get(i)).cloned().ok_or_else(|| format!("index {i} out of bounds (len {})", xs.len())),
        (Value::Array(xs), Value::Range(r)) => {
            let s = r.start.unwrap_or(0) as usize;
            let e = r.end.map_or(xs.len(), |e| e as usize + usize::from(r.inclusive));
            xs.get(s..e).map(|x| Value::Array(x.to_vec())).ok_or_else(|| format!("range {s}..{e} out of bounds (len {})", xs.len()))
        }
        (Value::Str(s), Value::Range(r)) => {
            let a = r.start.unwrap_or(0) as usize;
            let e = r.end.map_or(s.len(), |e| e as usize + usize::from(r.inclusive));
            s.get(a..e).map(Value::str).ok_or_else(|| "string slice out of bounds".to_string())
        }
        (b, _) => Err(format!("cannot index {}", b.type_name())),
    }
}

pub fn turbofish(t: Option<&syn::AngleBracketedGenericArguments>) -> Vec<String> {
    let Some(t) = t else { return vec![] };
    t.args
        .iter()
        .filter_map(|a| match a {
            syn::GenericArgument::Type(ty) => type_key(ty).or_else(|| Some(String::new())),
            syn::GenericArgument::Const(Expr::Lit(l)) => match &l.lit {
                Lit::Int(i) => Some(i.base10_digits().to_string()),
                _ => None,
            },
            syn::GenericArgument::Const(Expr::Block(b)) => Some(format!("{{{}}}", b.block.stmts.len())),
            _ => None,
        })
        .collect()
}

fn expr_kind(e: &Expr) -> &'static str {
    match e {
        Expr::Async(_) => "async block",
        Expr::Await(_) => "await",
        Expr::Const(_) => "const block",
        Expr::Infer(_) => "`_`",
        Expr::RawAddr(_) => "raw address",
        Expr::TryBlock(_) => "try block",
        Expr::Verbatim(_) => "unparsed syntax",
        Expr::Yield(_) => "yield",
        _ => "expression",
    }
}

fn pat_kind(p: &Pat) -> &'static str {
    match p {
        Pat::Const(_) => "const pattern",
        Pat::Macro(_) => "macro pattern",
        Pat::Verbatim(_) => "unparsed pattern",
        Pat::Rest(_) => "`..` outside a tuple or slice",
        _ => "pattern",
    }
}

/// `let wg = Workgroup::new(&mut b, arch)?;`
struct WgLet<'a> {
    pat: &'a Pat,
    builder: String,
    arch: &'a Expr,
    call: &'a Expr,
}

fn workgroup_let(s: &Stmt) -> Option<WgLet<'_>> {
    let Stmt::Local(l) = s else { return None };
    let init = l.init.as_ref()?;
    let Expr::Try(t) = &*init.expr else { return None };
    let Expr::Call(c) = &*t.expr else { return None };
    let Expr::Path(p) = &*c.func else { return None };
    let segs: Vec<String> = p.path.segments.iter().map(|s| s.ident.to_string()).collect();
    if segs.len() < 2 || segs[segs.len() - 2] != "Workgroup" || segs[segs.len() - 1] != "new" || c.args.len() != 2 {
        return None;
    }
    let Expr::Reference(r) = &c.args[0] else { return None };
    let Expr::Path(bp) = &*r.expr else { return None };
    let builder = bp.path.get_ident()?.to_string();
    let pat = match &l.pat {
        Pat::Type(pt) => &*pt.pat,
        p => p,
    };
    Some(WgLet { pat, builder, arch: &c.args[1], call: &t.expr })
}

/// Does the statement mention the identifier (a token scan, so shadowing
/// can only end a Workgroup scope early, never keep it open too long)?
fn mentions(s: &Stmt, name: &str) -> bool {
    use quote::ToTokens;
    fn walk(ts: proc_macro2::TokenStream, name: &str) -> bool {
        ts.into_iter().any(|t| match t {
            proc_macro2::TokenTree::Ident(i) => i == name,
            proc_macro2::TokenTree::Group(g) => walk(g.stream(), name),
            _ => false,
        })
    }
    walk(s.to_token_stream(), name)
}
