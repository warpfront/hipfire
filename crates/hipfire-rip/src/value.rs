//! Runtime values of the `.rip` evaluator.
//!
//! Plain data (`Int`, `Str`, `Tuple`, `Struct`, `Enum`, ...) has Rust's value
//! semantics: a copy is a deep copy. Mutation through `&mut` goes through a
//! [`PlaceRef`] (a variable cell plus a projection path). Natives that carry
//! identity (the register plan, the builder) are `Rc<RefCell<..>>`.

use crate::native::Nat;
use std::cell::RefCell;
use std::rc::Rc;
use syn::{Block, Expr, Pat, Signature};

/// Integer types of the evaluator. Unsuffixed literals are untyped (`None`)
/// and adopt the type of the other operand, like rustc's inference does for
/// the straight-line arithmetic of a kernel script.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntTy {
    U8,
    U16,
    U32,
    U64,
    Usize,
    I8,
    I16,
    I32,
    I64,
    Isize,
}

impl IntTy {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            "usize" => Self::Usize,
            "i8" => Self::I8,
            "i16" => Self::I16,
            "i32" => Self::I32,
            "i64" => Self::I64,
            "isize" => Self::Isize,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Usize => "usize",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Isize => "isize",
        }
    }
    pub fn bits(self) -> u32 {
        match self {
            Self::U8 | Self::I8 => 8,
            Self::U16 | Self::I16 => 16,
            Self::U32 | Self::I32 => 32,
            Self::U64 | Self::I64 | Self::Usize | Self::Isize => 64,
        }
    }
    pub fn signed(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Isize)
    }
    pub fn min(self) -> i128 {
        if self.signed() {
            -(1i128 << (self.bits() - 1))
        } else {
            0
        }
    }
    pub fn max(self) -> i128 {
        if self.signed() {
            (1i128 << (self.bits() - 1)) - 1
        } else {
            (1i128 << self.bits()) - 1
        }
    }
    pub fn fits(self, v: i128) -> bool {
        v >= self.min() && v <= self.max()
    }
    /// Two's complement truncation (`as` between integer types).
    pub fn wrap(self, v: i128) -> i128 {
        let m = 1i128 << self.bits();
        let mut w = v.rem_euclid(m);
        if self.signed() && w > self.max() {
            w -= m;
        }
        w
    }
}

/// Declared types the evaluator acts on (integers for typing and range
/// checks, and the containers that hold them). Every other type is `Any`.
#[derive(Clone, Debug)]
pub enum Ty {
    Int(IntTy),
    Array(Box<Ty>),
    Tuple(Vec<Ty>),
    Option(Box<Ty>),
    Result(Box<Ty>),
    Any,
}

impl Ty {
    pub fn from_syn(t: &syn::Type) -> Ty {
        match t {
            syn::Type::Path(p) if p.qself.is_none() => {
                let Some(seg) = p.path.segments.last() else { return Ty::Any };
                let name = seg.ident.to_string();
                if let Some(i) = IntTy::parse(&name) {
                    return Ty::Int(i);
                }
                let first_arg = || match &seg.arguments {
                    syn::PathArguments::AngleBracketed(a) => a.args.iter().find_map(|g| match g {
                        syn::GenericArgument::Type(t) => Some(Ty::from_syn(t)),
                        _ => None,
                    }),
                    _ => None,
                };
                match name.as_str() {
                    "Vec" => Ty::Array(Box::new(first_arg().unwrap_or(Ty::Any))),
                    "Option" => Ty::Option(Box::new(first_arg().unwrap_or(Ty::Any))),
                    "Result" => Ty::Result(Box::new(first_arg().unwrap_or(Ty::Any))),
                    _ => Ty::Any,
                }
            }
            syn::Type::Array(a) => Ty::Array(Box::new(Ty::from_syn(&a.elem))),
            syn::Type::Slice(a) => Ty::Array(Box::new(Ty::from_syn(&a.elem))),
            syn::Type::Tuple(t) => Ty::Tuple(t.elems.iter().map(Ty::from_syn).collect()),
            syn::Type::Reference(r) => Ty::from_syn(&r.elem),
            syn::Type::Paren(p) => Ty::from_syn(&p.elem),
            syn::Type::Group(g) => Ty::from_syn(&g.elem),
            _ => Ty::Any,
        }
    }
}

/// A function item: a free function, an associated function or a method.
pub struct FnDef<'a> {
    pub name: String,
    pub sig: &'a Signature,
    pub block: &'a Block,
    pub self_ty: Option<Rc<str>>,
    pub params: Vec<(&'a Pat, Ty)>,
    pub ret: Ty,
    /// 0 = no receiver, 1 = `self`, 2 = `&self`, 3 = `&mut self`.
    pub recv: u8,
}

/// A script closure with its lexical environment.
pub struct Closure<'a> {
    pub params: Vec<(&'a Pat, Ty)>,
    pub ret: Ty,
    pub body: &'a Expr,
    pub env: Env<'a>,
}

/// What a call expression can name.
pub enum Func<'a> {
    User(Rc<FnDef<'a>>),
    /// A native function, by its path (`op`, `Builder::new`, ...).
    Native(Rc<str>),
    /// `Enum::Variant(..)` (or `Some`, `Ok`, `Err`).
    Variant(Rc<str>, Rc<str>),
    /// A tuple struct constructor.
    TupleStruct(Rc<str>),
}

#[derive(Clone)]
pub enum Proj {
    Field(Rc<str>),
    Index(usize),
}

pub struct PlaceRef<'a> {
    pub cell: Rc<RefCell<Value<'a>>>,
    pub path: Vec<Proj>,
}

#[derive(Clone)]
pub struct RangeV {
    pub start: Option<i128>,
    pub end: Option<i128>,
    pub inclusive: bool,
    pub ty: Option<IntTy>,
}

#[derive(Clone)]
pub enum Value<'a> {
    Unit,
    Bool(bool),
    Int(i128, Option<IntTy>),
    Float(f64),
    Char(char),
    Str(Rc<str>),
    Tuple(Vec<Value<'a>>),
    /// Arrays, slices, `Vec`s and (materialized) iterators.
    Array(Vec<Value<'a>>),
    Range(Box<RangeV>),
    Struct(Rc<str>, Vec<(Rc<str>, Value<'a>)>),
    /// Type, variant, tuple payload.
    Enum(Rc<str>, Rc<str>, Vec<Value<'a>>),
    Closure(Rc<Closure<'a>>),
    Func(Rc<Func<'a>>),
    Nat(Nat),
    Ref(Rc<PlaceRef<'a>>),
}

impl<'a> Value<'a> {
    pub fn int(v: i128, t: IntTy) -> Self {
        Value::Int(v, Some(t))
    }
    pub fn u8(v: u8) -> Self {
        Value::Int(i128::from(v), Some(IntTy::U8))
    }
    pub fn u32(v: u32) -> Self {
        Value::Int(i128::from(v), Some(IntTy::U32))
    }
    pub fn usize(v: usize) -> Self {
        Value::Int(v as i128, Some(IntTy::Usize))
    }
    pub fn str(s: impl AsRef<str>) -> Self {
        Value::Str(Rc::from(s.as_ref()))
    }
    pub fn ok(v: Value<'a>) -> Self {
        Value::Enum("Result".into(), "Ok".into(), vec![v])
    }
    pub fn err(e: impl AsRef<str>) -> Self {
        Value::Enum("Result".into(), "Err".into(), vec![Value::str(e)])
    }
    pub fn some(v: Value<'a>) -> Self {
        Value::Enum("Option".into(), "Some".into(), vec![v])
    }
    pub fn none() -> Self {
        Value::Enum("Option".into(), "None".into(), vec![])
    }
    pub fn option(v: Option<Value<'a>>) -> Self {
        v.map_or_else(Self::none, Self::some)
    }
    pub fn type_name(&self) -> String {
        match self {
            Value::Unit => "()".into(),
            Value::Bool(_) => "bool".into(),
            Value::Int(_, Some(t)) => t.name().into(),
            Value::Int(_, None) => "{integer}".into(),
            Value::Float(_) => "f64".into(),
            Value::Char(_) => "char".into(),
            Value::Str(_) => "String".into(),
            Value::Tuple(_) => "tuple".into(),
            Value::Array(_) => "array".into(),
            Value::Range(_) => "range".into(),
            Value::Struct(n, _) => n.to_string(),
            Value::Enum(t, _, _) => t.to_string(),
            Value::Closure(_) => "closure".into(),
            Value::Func(_) => "fn".into(),
            Value::Nat(n) => n.type_name().into(),
            Value::Ref(_) => "reference".into(),
        }
    }
}

/// A lexical environment: a persistent chain of bindings, so a closure
/// captures exactly the bindings visible where it was written (a later `let`
/// of the same name does not reach it) while the cells stay shared.
pub struct Frame<'a> {
    pub name: Rc<str>,
    pub cell: Rc<RefCell<Value<'a>>>,
    pub parent: Env<'a>,
}
pub type Env<'a> = Option<Rc<Frame<'a>>>;

pub fn bind<'a>(env: &Env<'a>, name: &str, v: Value<'a>) -> Env<'a> {
    Some(Rc::new(Frame { name: Rc::from(name), cell: Rc::new(RefCell::new(v)), parent: env.clone() }))
}

pub fn lookup<'a, 'e>(env: &'e Env<'a>, name: &str) -> Option<&'e Rc<RefCell<Value<'a>>>> {
    let mut cur = env.as_ref();
    while let Some(f) = cur {
        if &*f.name == name {
            return Some(&f.cell);
        }
        cur = f.parent.as_ref();
    }
    None
}

/// Structural equality (`==` on derived `PartialEq` data).
pub fn values_eq(a: &Value, b: &Value) -> bool {
    use Value::*;
    match (a, b) {
        (Unit, Unit) => true,
        (Bool(x), Bool(y)) => x == y,
        (Int(x, _), Int(y, _)) => x == y,
        (Float(x), Float(y)) => x == y,
        (Char(x), Char(y)) => x == y,
        (Str(x), Str(y)) => x == y,
        (Tuple(x), Tuple(y)) | (Array(x), Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| values_eq(p, q)),
        (Struct(n, x), Struct(m, y)) => n == m && x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.0 == q.0 && values_eq(&p.1, &q.1)),
        (Enum(t, v, x), Enum(u, w, y)) => t == u && v == w && x.len() == y.len() && x.iter().zip(y).all(|(p, q)| values_eq(p, q)),
        (Nat(x), Nat(y)) => x.equals(y),
        (Range(x), Range(y)) => x.start == y.start && x.end == y.end && x.inclusive == y.inclusive,
        _ => false,
    }
}

impl std::fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&crate::fmt::debug(self, false))
    }
}
