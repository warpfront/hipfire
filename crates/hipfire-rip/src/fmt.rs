//! `format!` strings: parsing of `{}` / `{0}` / `{name}` / `{:#x}` pieces and
//! `Display` / `Debug` rendering of values.

use crate::value::{IntTy, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum ArgRef {
    Next,
    Index(usize),
    Name(String),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Spec {
    pub fill: Option<char>,
    pub align: Option<char>,
    pub plus: bool,
    pub alt: bool,
    pub zero: bool,
    pub width: Option<usize>,
    pub prec: Option<usize>,
    /// "", "?", "x", "X", "b", "o", "e".
    pub ty: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Lit(String),
    Arg(ArgRef, Spec),
}

pub fn parse(fmt: &str) -> Result<Vec<Piece>, String> {
    let mut out = Vec::new();
    let mut lit = String::new();
    let mut it = fmt.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '{' if it.peek() == Some(&'{') => {
                it.next();
                lit.push('{');
            }
            '}' if it.peek() == Some(&'}') => {
                it.next();
                lit.push('}');
            }
            '}' => return Err("format string: unmatched `}`".into()),
            '{' => {
                let mut body = String::new();
                loop {
                    match it.next() {
                        Some('}') => break,
                        Some(c) => body.push(c),
                        None => return Err("format string: unmatched `{`".into()),
                    }
                }
                if !lit.is_empty() {
                    out.push(Piece::Lit(std::mem::take(&mut lit)));
                }
                let (name, spec) = match body.split_once(':') {
                    Some((n, s)) => (n.trim(), s),
                    None => (body.trim(), ""),
                };
                let target = if name.is_empty() {
                    ArgRef::Next
                } else if let Ok(i) = name.parse::<usize>() {
                    ArgRef::Index(i)
                } else {
                    ArgRef::Name(name.to_string())
                };
                out.push(Piece::Arg(target, parse_spec(spec)?));
            }
            c => lit.push(c),
        }
    }
    if !lit.is_empty() {
        out.push(Piece::Lit(lit));
    }
    Ok(out)
}

fn parse_spec(s: &str) -> Result<Spec, String> {
    let ch: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut sp = Spec::default();
    if ch.len() >= 2 && matches!(ch[1], '<' | '>' | '^') {
        sp.fill = Some(ch[0]);
        sp.align = Some(ch[1]);
        i = 2;
    } else if !ch.is_empty() && matches!(ch[0], '<' | '>' | '^') {
        sp.align = Some(ch[0]);
        i = 1;
    }
    if i < ch.len() && ch[i] == '+' {
        sp.plus = true;
        i += 1;
    }
    if i < ch.len() && ch[i] == '#' {
        sp.alt = true;
        i += 1;
    }
    if i < ch.len() && ch[i] == '0' && ch.get(i + 1).is_some_and(|c| c.is_ascii_digit()) {
        sp.zero = true;
        i += 1;
    }
    let mut w = String::new();
    while i < ch.len() && ch[i].is_ascii_digit() {
        w.push(ch[i]);
        i += 1;
    }
    if !w.is_empty() {
        sp.width = Some(w.parse().map_err(|_| "format string: bad width")?);
    }
    if i < ch.len() && ch[i] == '$' {
        return Err("format string: `$` width/precision arguments are not supported".into());
    }
    if i < ch.len() && ch[i] == '.' {
        i += 1;
        let mut p = String::new();
        while i < ch.len() && ch[i].is_ascii_digit() {
            p.push(ch[i]);
            i += 1;
        }
        sp.prec = Some(p.parse().map_err(|_| "format string: bad precision")?);
    }
    sp.ty = ch[i..].iter().collect();
    if !matches!(sp.ty.as_str(), "" | "?" | "x" | "X" | "b" | "o" | "e" | "#?") {
        return Err(format!("format string: unsupported spec `{s}`"));
    }
    Ok(sp)
}

pub fn display(v: &Value) -> Result<String, String> {
    Ok(match v {
        Value::Unit => "()".into(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i, _) => i.to_string(),
        Value::Float(f) => float_display(*f),
        Value::Char(c) => c.to_string(),
        Value::Str(s) => s.to_string(),
        Value::Ref(_) => return Err("internal: unresolved reference in format".into()),
        Value::Nat(n) => n.display().ok_or_else(|| format!("`{}` does not implement Display", n.type_name()))?,
        other => return Err(format!("`{}` does not implement Display (use {{:?}})", other.type_name())),
    })
}

fn float_display(f: f64) -> String {
    if f.fract() == 0.0 && f.is_finite() && f.abs() < 1e16 {
        format!("{}", f as i64)
    } else {
        format!("{f}")
    }
}

pub fn debug(v: &Value, pretty: bool) -> String {
    match v {
        Value::Str(s) => format!("{s:?}"),
        Value::Char(c) => format!("{c:?}"),
        Value::Tuple(xs) => {
            let inner: Vec<String> = xs.iter().map(|x| debug(x, pretty)).collect();
            if xs.len() == 1 {
                format!("({},)", inner[0])
            } else {
                format!("({})", inner.join(", "))
            }
        }
        Value::Array(xs) => format!("[{}]", xs.iter().map(|x| debug(x, pretty)).collect::<Vec<_>>().join(", ")),
        Value::Struct(n, fs) => {
            if fs.is_empty() {
                n.to_string()
            } else {
                format!("{n} {{ {} }}", fs.iter().map(|(k, x)| format!("{k}: {}", debug(x, pretty))).collect::<Vec<_>>().join(", "))
            }
        }
        Value::Enum(_, var, xs) => {
            if xs.is_empty() {
                var.to_string()
            } else {
                format!("{var}({})", xs.iter().map(|x| debug(x, pretty)).collect::<Vec<_>>().join(", "))
            }
        }
        Value::Range(r) => format!("{}..{}{}", r.start.map(|s| s.to_string()).unwrap_or_default(), if r.inclusive { "=" } else { "" }, r.end.map(|s| s.to_string()).unwrap_or_default()),
        Value::Nat(n) => n.debug(),
        Value::Closure(_) => "<closure>".into(),
        Value::Func(_) => "<fn>".into(),
        other => display(other).unwrap_or_else(|e| format!("<{e}>")),
    }
}

fn int_radix(i: i128, t: Option<IntTy>, radix: u32, upper: bool) -> String {
    let bits = t.map_or(32, IntTy::bits);
    let u: u128 = if i < 0 { (i + (1i128 << bits)) as u128 } else { i as u128 };
    match (radix, upper) {
        (16, false) => format!("{u:x}"),
        (16, true) => format!("{u:X}"),
        (2, _) => format!("{u:b}"),
        _ => format!("{u:o}"),
    }
}

pub fn render(v: &Value, sp: &Spec) -> Result<String, String> {
    let mut numeric = matches!(v, Value::Int(..) | Value::Float(_));
    let mut body = match sp.ty.as_str() {
        "" => match (v, sp.prec) {
            (Value::Float(f), Some(p)) => format!("{f:.p$}"),
            (Value::Str(s), Some(p)) => s.chars().take(p).collect(),
            _ => display(v)?,
        },
        "?" | "#?" => debug(v, sp.alt || sp.ty == "#?"),
        "x" | "X" | "b" | "o" => {
            let Value::Int(i, t) = v else { return Err(format!("`{}` has no radix formatting", v.type_name())) };
            let (radix, upper) = match sp.ty.as_str() {
                "x" => (16, false),
                "X" => (16, true),
                "b" => (2, false),
                _ => (8, false),
            };
            let digits = int_radix(*i, *t, radix, upper);
            if sp.alt {
                let prefix = match radix {
                    16 => "0x",
                    2 => "0b",
                    _ => "0o",
                };
                format!("{prefix}{digits}")
            } else {
                digits
            }
        }
        "e" => match v {
            Value::Float(f) => format!("{f:e}"),
            Value::Int(i, _) => format!("{:e}", *i as f64),
            _ => return Err("`{:e}` needs a number".into()),
        },
        t => return Err(format!("unsupported format type `{t}`")),
    };
    if sp.plus && numeric && !body.starts_with('-') {
        body.insert(0, '+');
    }
    let n = body.chars().count();
    if let Some(w) = sp.width {
        if n < w {
            let pad = w - n;
            if sp.zero && numeric {
                // Zero padding goes after the sign / radix prefix.
                let split = if body.starts_with("0x") || body.starts_with("0b") || body.starts_with("0o") {
                    2
                } else if body.starts_with('-') || body.starts_with('+') {
                    1
                } else {
                    0
                };
                body.insert_str(split, &"0".repeat(pad));
            } else {
                let fill = sp.fill.unwrap_or(' ');
                let align = sp.align.unwrap_or(if numeric { '>' } else { '<' });
                let (l, r) = match align {
                    '<' => (0, pad),
                    '>' => (pad, 0),
                    _ => (pad / 2, pad - pad / 2),
                };
                body = format!("{}{body}{}", fill.to_string().repeat(l), fill.to_string().repeat(r));
            }
        }
    }
    numeric = false;
    let _ = numeric;
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_cover_positional_named_and_escapes() {
        let p = parse("a{{b}} {} {0} {x} {:#x}").unwrap();
        assert_eq!(p[0], Piece::Lit("a{b} ".into()));
        assert_eq!(p[1], Piece::Arg(ArgRef::Next, Spec::default()));
        assert_eq!(p[3], Piece::Arg(ArgRef::Index(0), Spec::default()));
        assert_eq!(p[5], Piece::Arg(ArgRef::Name("x".into()), Spec::default()));
        assert!(matches!(&p[7], Piece::Arg(ArgRef::Next, s) if s.alt && s.ty == "x"));
        assert!(parse("{").is_err() && parse("}").is_err());
    }

    #[test]
    fn radix_and_padding_follow_rust() {
        let v = Value::Int(255, Some(IntTy::U32));
        assert_eq!(render(&v, &parse_spec("#x").unwrap()).unwrap(), "0xff");
        assert_eq!(render(&v, &parse_spec("#010x").unwrap()).unwrap(), "0x000000ff");
        assert_eq!(render(&v, &parse_spec("04").unwrap()).unwrap(), "0255");
        assert_eq!(render(&Value::Int(-1, Some(IntTy::I32)), &parse_spec("x").unwrap()).unwrap(), "ffffffff");
        assert_eq!(render(&Value::str("ab"), &parse_spec("*^6").unwrap()).unwrap(), "**ab**");
    }
}
