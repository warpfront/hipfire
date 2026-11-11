//! Declared search axes of a `.rip` script (`rip-search`).
//!
//! A script declares its search space with one reserved const beside `VARIANTS`
//! and `ARCHES`:
//!
//! ```text
//! const SEARCH_AXES: &[(&str, &str, &[&str])] = &[
//!     ("ROWS",  "schedule", &["1", "2", "4"]),
//!     ("SPLIT", "schedule", &["false", "true"]),
//! ];
//! const ROWS: u8 = 4;        // each axis const: ONE line, `const NAME: TY = LIT;`
//! const SPLIT: bool = false;
//! ```
//!
//! Each tuple is (axis const name, exactness class, values). The axis const's own
//! declared value is the axis' reference point and must be one of the values.
//! The axis const must exist exactly once as a single-line `const NAME: TY = LIT;`
//! where `TY` is an integer type, `bool` or `&str`.
//!
//! [`with_axes`] is a byte-level splice: only the literal of each named axis const
//! changes, so every other byte (and every diagnostic position) is unchanged.
//! [`points`] is the canonical enumeration: axes in declared order, values in
//! declared order, the last axis fastest.
//!
//! Errors are `name:line:col: message`.

use proc_macro2::Span;
use std::fmt::Display;
use syn::spanned::Spanned;
use syn::{Expr, ExprLit, Item, ItemConst, Lit};

/// One declared search axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Axis {
    /// The axis const's name.
    pub name: String,
    /// The exactness class the script claims (`schedule` in phase 1).
    pub class: String,
    /// Canonical values in declared order: integers in decimal, `true`/`false`, or the
    /// unquoted contents of a `&str`.
    pub values: Vec<String>,
    /// The axis const's declared value (canonical), one of `values`.
    pub reference: String,
    /// One-based line of the axis const.
    pub line: usize,
    /// The const's type: an integer type, `bool` or `&str`.
    pub ty: String,
}

const INT_TYPES: [&str; 12] = [
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize",
];

fn at(name: &str, span: Span, msg: impl Display) -> String {
    let p = span.start();
    format!("{name}:{}:{}: {msg}", p.line, p.column + 1)
}

fn peel(e: &Expr) -> &Expr {
    match e {
        Expr::Reference(r) => peel(&r.expr),
        Expr::Paren(p) => peel(&p.expr),
        Expr::Group(g) => peel(&g.expr),
        _ => e,
    }
}

fn str_lit(e: &Expr) -> Option<String> {
    match peel(e) {
        Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }) => Some(s.value()),
        _ => None,
    }
}

fn collect_consts<'a>(items: &'a [Item], out: &mut Vec<&'a ItemConst>) {
    for item in items {
        match item {
            Item::Const(c) => out.push(c),
            Item::Mod(m) => {
                if let Some((_, inner)) = &m.content {
                    collect_consts(inner, out);
                }
            }
            _ => {}
        }
    }
}

/// Canonical text of `text` as a value of `ty`; `Err` names the problem.
fn canon(ty: &str, text: &str) -> Result<String, String> {
    macro_rules! int {
        ($t:ty) => {
            text.parse::<$t>()
                .map(|n| n.to_string())
                .map_err(|e| format!("`{text}` is not a valid {ty} value: {e}"))
        };
    }
    match ty {
        "bool" => match text {
            "true" | "false" => Ok(text.to_string()),
            _ => Err(format!(
                "`{text}` is not a valid bool value (true or false)"
            )),
        },
        "&str" => Ok(text.to_string()),
        "u8" => int!(u8),
        "u16" => int!(u16),
        "u32" => int!(u32),
        "u64" => int!(u64),
        "u128" => int!(u128),
        "usize" => int!(usize),
        "i8" => int!(i8),
        "i16" => int!(i16),
        "i32" => int!(i32),
        "i64" => int!(i64),
        "i128" => int!(i128),
        "isize" => int!(isize),
        _ => Err(format!(
            "unsupported axis type `{ty}` (an integer type, bool or &str)"
        )),
    }
}

/// The literal text of a value rendered by type: a quoted Rust string literal for
/// `&str`, `true`/`false` for bool, a range-checked decimal for integers.
pub fn render(ty: &str, value: &str) -> Result<String, String> {
    let c = canon(ty, value)?;
    Ok(if ty == "&str" { format!("{c:?}") } else { c })
}

/// The canonical value of the source literal `lit` of type `ty`.
fn literal_value(ty: &str, lit: &str) -> Result<String, String> {
    match ty {
        "bool" => canon(ty, lit),
        "&str" => {
            if !lit.starts_with('"') {
                return Err(format!("`{lit}` is not a plain string literal"));
            }
            let s = syn::parse_str::<syn::LitStr>(lit)
                .map_err(|e| format!("`{lit}` is not a string literal: {e}"))?;
            Ok(s.value())
        }
        t if INT_TYPES.contains(&t) => {
            let (neg, body) = match lit.strip_prefix('-') {
                Some(rest) => (true, rest.trim_start()),
                None => (false, lit),
            };
            let n = syn::parse_str::<syn::LitInt>(body)
                .map_err(|e| format!("`{lit}` is not an integer literal: {e}"))?;
            if !n.suffix().is_empty() && n.suffix() != ty {
                return Err(format!(
                    "`{lit}` has suffix `{}` but the const is {ty}",
                    n.suffix()
                ));
            }
            let mag = n
                .base10_parse::<u128>()
                .map_err(|e| format!("`{lit}`: {e}"))?;
            canon(ty, &format!("{}{mag}", if neg { "-" } else { "" }))
        }
        _ => Err(format!(
            "unsupported axis type `{ty}` (an integer type, bool or &str)"
        )),
    }
}

/// Parse a single-line `const NAME: TY = LIT;` (optional `pub`, optional trailing
/// `//` comment) and return (normalized type, literal byte range within `line`).
fn scan_line(line: &str, name: &str) -> Result<(String, usize, usize), String> {
    let b = line.as_bytes();
    let ws = |p: &mut usize| {
        while *p < b.len() && (b[*p] == b' ' || b[*p] == b'\t' || b[*p] == b'\r') {
            *p += 1;
        }
    };
    let ident_char = |c: u8| c == b'_' || c.is_ascii_alphanumeric();
    let mut p = 0;
    ws(&mut p);
    if line[p..].starts_with("pub") && b.get(p + 3).is_some_and(|c| *c == b' ' || *c == b'\t') {
        p += 3;
        ws(&mut p);
    }
    let shape = || {
        format!("axis const `{name}` must be one line `const {name}: TY = LITERAL;` (an integer type, bool or &str)")
    };
    if !line[p..].starts_with("const") || !b.get(p + 5).is_some_and(|c| *c == b' ' || *c == b'\t') {
        return Err(shape());
    }
    p += 5;
    ws(&mut p);
    if !line[p..].starts_with(name) || b.get(p + name.len()).is_some_and(|c| ident_char(*c)) {
        return Err(shape());
    }
    p += name.len();
    ws(&mut p);
    if b.get(p) != Some(&b':') {
        return Err(shape());
    }
    p += 1;
    let eq = line[p..].find('=').map(|i| p + i).ok_or_else(shape)?;
    let ty: String = line[p..eq].split_whitespace().collect();
    let ty = if ty == "&'staticstr" {
        "&str".to_string()
    } else {
        ty
    };
    if ty != "bool" && ty != "&str" && !INT_TYPES.contains(&ty.as_str()) {
        return Err(format!(
            "axis const `{name}` has unsupported type `{ty}` (an integer type, bool or &str)"
        ));
    }
    p = eq + 1;
    ws(&mut p);
    let lit_start = p;
    if b.get(p) == Some(&b'"') {
        p += 1;
        loop {
            match b.get(p) {
                None => return Err(shape()),
                Some(b'\\') => p += 2,
                Some(b'"') => {
                    p += 1;
                    break;
                }
                Some(_) => p += 1,
            }
        }
    } else {
        while p < b.len() && !matches!(b[p], b' ' | b'\t' | b'\r' | b';' | b'/') {
            p += 1;
        }
    }
    let lit_end = p;
    if lit_end == lit_start {
        return Err(shape());
    }
    ws(&mut p);
    if b.get(p) != Some(&b';') {
        return Err(shape());
    }
    p += 1;
    ws(&mut p);
    if p < b.len() && !line[p..].starts_with("//") {
        return Err(shape());
    }
    Ok((ty, lit_start, lit_end))
}

struct Scanned {
    axis: Axis,
    /// Byte range of the axis const's literal in the whole source.
    start: usize,
    end: usize,
}

fn scan(source: &str, name: &str) -> Result<Vec<Scanned>, String> {
    let file = crate::parse(source, name)?;
    let mut consts = Vec::new();
    collect_consts(&file.items, &mut consts);
    let decls: Vec<&ItemConst> = consts
        .iter()
        .copied()
        .filter(|c| c.ident == "SEARCH_AXES")
        .collect();
    if decls.len() > 1 {
        return Err(at(
            name,
            decls[1].const_token.span,
            "`SEARCH_AXES` is declared more than once",
        ));
    }
    let Some(decl) = decls.first() else {
        return Ok(Vec::new());
    };
    let Expr::Array(list) = peel(&decl.expr) else {
        return Err(at(
            name,
            decl.expr.span(),
            "`SEARCH_AXES` must be `&[(&str, &str, &[&str])]` written as an array literal",
        ));
    };
    // Byte offset of the start of every line.
    let mut line_starts = vec![0usize];
    for (i, c) in source.bytes().enumerate() {
        if c == b'\n' {
            line_starts.push(i + 1);
        }
    }
    let mut out: Vec<Scanned> = Vec::new();
    for elem in &list.elems {
        let bad = |msg: &str| at(name, elem.span(), msg);
        let Expr::Tuple(t) = peel(elem) else {
            return Err(bad(
                "each `SEARCH_AXES` entry must be a (name, class, values) tuple",
            ));
        };
        if t.elems.len() != 3 {
            return Err(bad(
                "each `SEARCH_AXES` entry must be a (name, class, values) tuple",
            ));
        }
        let axis_name =
            str_lit(&t.elems[0]).ok_or_else(|| bad("the axis name must be a string literal"))?;
        let class = str_lit(&t.elems[1])
            .ok_or_else(|| bad("the exactness class must be a string literal"))?;
        let Expr::Array(vals) = peel(&t.elems[2]) else {
            return Err(bad("the axis values must be an array of string literals"));
        };
        let raw: Vec<String> = vals
            .elems
            .iter()
            .map(|v| {
                str_lit(v)
                    .ok_or_else(|| at(name, v.span(), "an axis value must be a string literal"))
            })
            .collect::<Result<_, _>>()?;
        if raw.is_empty() {
            return Err(bad(&format!("axis `{axis_name}` has no values")));
        }
        if out.iter().any(|s| s.axis.name == axis_name) {
            return Err(bad(&format!("axis `{axis_name}` is declared twice")));
        }
        let matching: Vec<&&ItemConst> = consts.iter().filter(|c| c.ident == axis_name).collect();
        let c = match matching.as_slice() {
            [] => {
                return Err(bad(&format!(
                    "axis const `{axis_name}` is not declared (expected a one-line `const {axis_name}: TY = LITERAL;`)"
                )))
            }
            [c] => c,
            [_, second, ..] => {
                return Err(at(
                    name,
                    second.const_token.span,
                    format!("axis const `{axis_name}` is declared more than once"),
                ))
            }
        };
        let line = c.const_token.span.start().line;
        let loc = c.const_token.span;
        if line == 0 || line > line_starts.len() {
            return Err(at(
                name,
                loc,
                format!("axis const `{axis_name}`: position outside the source"),
            ));
        }
        let text =
            &source[line_starts[line - 1]..line_starts.get(line).map_or(source.len(), |e| e - 1)];
        let (ty, ls, le) = scan_line(text, &axis_name).map_err(|m| at(name, loc, m))?;
        let reference = literal_value(&ty, &text[ls..le]).map_err(|m| at(name, loc, m))?;
        let mut values: Vec<String> = Vec::new();
        for r in &raw {
            let v = canon(&ty, r).map_err(|m| bad(&format!("axis `{axis_name}`: {m}")))?;
            if values.contains(&v) {
                return Err(bad(&format!("axis `{axis_name}` lists value `{v}` twice")));
            }
            values.push(v);
        }
        if !values.contains(&reference) {
            return Err(at(
                name,
                loc,
                format!(
                    "axis const `{axis_name}` is declared as `{reference}`, which is not in its search values [{}]",
                    values.join(", ")
                ),
            ));
        }
        out.push(Scanned {
            axis: Axis {
                name: axis_name,
                class,
                values,
                reference,
                line,
                ty,
            },
            start: line_starts[line - 1] + ls,
            end: line_starts[line - 1] + le,
        });
    }
    Ok(out)
}

/// The axes declared by `const SEARCH_AXES`, empty when the script declares none.
/// `name` is the source's name for error positions.
pub fn search_axes(source: &str, name: &str) -> Result<Vec<Axis>, String> {
    Ok(scan(source, name)?.into_iter().map(|s| s.axis).collect())
}

/// `source` with each named axis const's literal replaced by `value` rendered by
/// type; every other byte is unchanged. Axes absent from `point` keep their
/// declared value.
pub fn with_axes(source: &str, name: &str, point: &[(String, String)]) -> Result<String, String> {
    let scanned = scan(source, name)?;
    let mut splices: Vec<(usize, usize, String)> = Vec::new();
    for (i, (axis, value)) in point.iter().enumerate() {
        if point[..i].iter().any(|(a, _)| a == axis) {
            return Err(format!("{name}: axis `{axis}` is set twice"));
        }
        let s = scanned
            .iter()
            .find(|s| &s.axis.name == axis)
            .ok_or_else(|| format!("{name}: `{axis}` is not a declared search axis"))?;
        let text = render(&s.axis.ty, value)
            .map_err(|m| format!("{name}:{}: axis `{axis}`: {m}", s.axis.line))?;
        splices.push((s.start, s.end, text));
    }
    splices.sort_by_key(|s| s.0);
    let mut out = String::with_capacity(source.len() + 16 * splices.len());
    let mut pos = 0;
    for (start, end, text) in splices {
        out.push_str(&source[pos..start]);
        out.push_str(&text);
        pos = end;
    }
    out.push_str(&source[pos..]);
    Ok(out)
}

/// The reference point: every axis at its declared value, in declared order.
pub fn reference_point(axes: &[Axis]) -> Vec<(String, String)> {
    axes.iter()
        .map(|a| (a.name.clone(), a.reference.clone()))
        .collect()
}

/// Number of points of the product of `axes` (1 for no axes); `None` on overflow.
pub fn point_count(axes: &[Axis]) -> Option<usize> {
    axes.iter()
        .try_fold(1usize, |n, a| n.checked_mul(a.values.len()))
}

/// Canonical enumeration: axes in declared order, values in declared order, the
/// last axis fastest (odometer). No axes yields the single empty point.
pub fn points(axes: &[Axis]) -> Vec<Vec<(String, String)>> {
    let mut out = Vec::new();
    let mut idx = vec![0usize; axes.len()];
    loop {
        out.push(
            axes.iter()
                .zip(&idx)
                .map(|(a, &i)| (a.name.clone(), a.values[i].clone()))
                .collect(),
        );
        let mut k = axes.len();
        loop {
            if k == 0 {
                return out;
            }
            k -= 1;
            idx[k] += 1;
            if idx[k] < axes[k].values.len() {
                break;
            }
            idx[k] = 0;
        }
    }
}
