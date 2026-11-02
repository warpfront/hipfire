//! Independent declared-ABI reader for HIP kernel sources.
//!
//! A small C preprocessor (object-like macros, conditionals, parameter macros)
//! feeds a `__global__` signature parser. It evaluates every kernel under each
//! target architecture's predefined macros and refuses (fail closed) any
//! symbol whose argument layout differs between architectures, so the layout
//! it reports is the one every compile of that source produces.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arg {
    pub offset: u32,
    pub size: u32,
    /// `p` pointer, `i` integer, `f` floating point, `v` by-value of unknown class.
    pub kind: char,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decl {
    pub args: Vec<Arg>,
    /// End offset of the last explicit argument (unpadded).
    pub end: u32,
}

impl Decl {
    pub fn from_scalars(args: &[(u32, char)]) -> Decl {
        let mut out = Vec::new();
        let mut off = 0u32;
        for (size, kind) in args {
            off = (off + size - 1) / size * size;
            out.push(Arg {
                offset: off,
                size: *size,
                kind: *kind,
                text: String::new(),
            });
            off += size;
        }
        Decl {
            args: out,
            end: off,
        }
    }
}

#[derive(Clone, Debug)]
enum Def {
    Obj(String),
    Func,
}

pub const ARCHES: [(&str, [&str; 2]); 6] = [
    ("gfx906", ["__gfx906__", "__GFX9__"]),
    ("gfx1030", ["__gfx1030__", "__GFX10__"]),
    ("gfx1100", ["__gfx1100__", "__GFX11__"]),
    ("gfx1151", ["__gfx1151__", "__GFX11__"]),
    ("gfx1201", ["__gfx1201__", "__GFX12__"]),
    ("gfx942", ["__gfx942__", "__GFX9__"]),
];

pub fn strip_comments(t: &str) -> String {
    let c: Vec<char> = t.chars().collect();
    let n = c.len();
    let mut out = String::with_capacity(t.len());
    let mut i = 0usize;
    while i < n {
        if c[i] == '/' && i + 1 < n && c[i + 1] == '/' {
            while i < n && c[i] != '\n' {
                i += 1;
            }
        } else if c[i] == '/' && i + 1 < n && c[i + 1] == '*' {
            i += 2;
            while i + 1 < n && !(c[i] == '*' && c[i + 1] == '/') {
                if c[i] == '\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i += 2;
        } else if c[i] == '"' {
            out.push('"');
            i += 1;
            while i < n && c[i] != '"' {
                if c[i] == '\\' && i + 1 < n {
                    out.push(c[i]);
                    i += 1;
                }
                out.push(c[i]);
                i += 1;
            }
            if i < n {
                out.push('"');
                i += 1;
            }
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    out
}

fn is_ident_char(c: char) -> bool {
    c == '_' || c.is_ascii_alphanumeric()
}

fn expand_obj(s: &str, defs: &BTreeMap<String, Def>, depth: u32) -> Result<String, String> {
    if depth > 24 {
        return Err("macro expansion too deep".into());
    }
    let c: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < c.len() {
        if c[i] == '_' || c[i].is_ascii_alphabetic() {
            let mut j = i;
            while j < c.len() && is_ident_char(c[j]) {
                j += 1;
            }
            let w: String = c[i..j].iter().collect();
            match defs.get(&w) {
                Some(Def::Obj(v)) => out.push_str(&expand_obj(v, defs, depth + 1)?),
                _ => out.push_str(&w),
            }
            i = j;
        } else if c[i].is_ascii_digit() {
            // Keep numeric literals (including suffix letters) intact.
            let mut j = i;
            while j < c.len() && is_ident_char(c[j]) {
                j += 1;
            }
            out.extend(c[i..j].iter());
            i = j;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    Ok(out)
}

// ---- `#if` expression evaluator -------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum ET {
    Num(i64),
    Id(String),
    Op(&'static str),
}

fn etokenize(s: &str) -> Result<Vec<ET>, String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch.is_ascii_digit() {
            let mut j = i;
            while j < c.len() && is_ident_char(c[j]) {
                j += 1;
            }
            let lit: String = c[i..j].iter().collect();
            let digits: String = lit
                .trim_end_matches(|x| matches!(x, 'u' | 'U' | 'l' | 'L'))
                .to_string();
            let v = if let Some(h) = digits.strip_prefix("0x") {
                i64::from_str_radix(h, 16)
            } else {
                digits.parse::<i64>()
            };
            out.push(ET::Num(v.map_err(|_| format!("bad number {lit}"))?));
            i = j;
        } else if ch == '_' || ch.is_ascii_alphabetic() {
            let mut j = i;
            while j < c.len() && is_ident_char(c[j]) {
                j += 1;
            }
            out.push(ET::Id(c[i..j].iter().collect()));
            i = j;
        } else {
            let two: String = c[i..(i + 2).min(c.len())].iter().collect();
            let op: Option<&'static str> = match two.as_str() {
                "&&" => Some("&&"),
                "||" => Some("||"),
                "==" => Some("=="),
                "!=" => Some("!="),
                "<=" => Some("<="),
                ">=" => Some(">="),
                _ => None,
            };
            if let Some(o) = op {
                out.push(ET::Op(o));
                i += 2;
                continue;
            }
            let one: Option<&'static str> = match ch {
                '!' => Some("!"),
                '<' => Some("<"),
                '>' => Some(">"),
                '+' => Some("+"),
                '-' => Some("-"),
                '*' => Some("*"),
                '%' => Some("%"),
                '(' => Some("("),
                ')' => Some(")"),
                _ => None,
            };
            match one {
                Some(o) => out.push(ET::Op(o)),
                None => return Err(format!("unsupported #if character {ch:?} in {s:?}")),
            }
            i += 1;
        }
    }
    Ok(out)
}

struct Eval<'a> {
    t: Vec<ET>,
    p: usize,
    defs: &'a BTreeMap<String, Def>,
    depth: u32,
}

impl<'a> Eval<'a> {
    fn peek_op(&self, o: &str) -> bool {
        matches!(self.t.get(self.p), Some(ET::Op(x)) if *x == o)
    }
    fn or(&mut self) -> Result<i64, String> {
        let mut v = self.and()?;
        while self.peek_op("||") {
            self.p += 1;
            let r = self.and()?;
            v = i64::from(v != 0 || r != 0);
        }
        Ok(v)
    }
    fn and(&mut self) -> Result<i64, String> {
        let mut v = self.eq()?;
        while self.peek_op("&&") {
            self.p += 1;
            let r = self.eq()?;
            v = i64::from(v != 0 && r != 0);
        }
        Ok(v)
    }
    fn eq(&mut self) -> Result<i64, String> {
        let mut v = self.rel()?;
        loop {
            if self.peek_op("==") {
                self.p += 1;
                let r = self.rel()?;
                v = i64::from(v == r);
            } else if self.peek_op("!=") {
                self.p += 1;
                let r = self.rel()?;
                v = i64::from(v != r);
            } else {
                return Ok(v);
            }
        }
    }
    fn rel(&mut self) -> Result<i64, String> {
        let mut v = self.add()?;
        loop {
            let op = ["<=", ">=", "<", ">"]
                .iter()
                .find(|o| self.peek_op(o))
                .copied();
            match op {
                None => return Ok(v),
                Some(o) => {
                    self.p += 1;
                    let r = self.add()?;
                    v = i64::from(match o {
                        "<=" => v <= r,
                        ">=" => v >= r,
                        "<" => v < r,
                        _ => v > r,
                    });
                }
            }
        }
    }
    fn add(&mut self) -> Result<i64, String> {
        let mut v = self.mul()?;
        loop {
            if self.peek_op("+") {
                self.p += 1;
                v += self.mul()?;
            } else if self.peek_op("-") {
                self.p += 1;
                v -= self.mul()?;
            } else {
                return Ok(v);
            }
        }
    }
    fn mul(&mut self) -> Result<i64, String> {
        let mut v = self.unary()?;
        loop {
            if self.peek_op("*") {
                self.p += 1;
                v *= self.unary()?;
            } else if self.peek_op("%") {
                self.p += 1;
                let r = self.unary()?;
                if r == 0 {
                    return Err("modulo by zero in #if".into());
                }
                v %= r;
            } else {
                return Ok(v);
            }
        }
    }
    fn unary(&mut self) -> Result<i64, String> {
        if self.peek_op("!") {
            self.p += 1;
            return Ok(i64::from(self.unary()? == 0));
        }
        if self.peek_op("-") {
            self.p += 1;
            return Ok(-self.unary()?);
        }
        self.primary()
    }
    fn primary(&mut self) -> Result<i64, String> {
        let t = self
            .t
            .get(self.p)
            .cloned()
            .ok_or("truncated #if expression")?;
        self.p += 1;
        match t {
            ET::Num(v) => Ok(v),
            ET::Op("(") => {
                let v = self.or()?;
                if !self.peek_op(")") {
                    return Err("missing ) in #if".into());
                }
                self.p += 1;
                Ok(v)
            }
            ET::Id(name) if name == "defined" => {
                let paren = self.peek_op("(");
                if paren {
                    self.p += 1;
                }
                let id = match self.t.get(self.p) {
                    Some(ET::Id(x)) => x.clone(),
                    _ => return Err("defined without identifier".into()),
                };
                self.p += 1;
                if paren {
                    if !self.peek_op(")") {
                        return Err("defined( without )".into());
                    }
                    self.p += 1;
                }
                Ok(i64::from(self.defs.contains_key(&id)))
            }
            ET::Id(name) => {
                let defs = self.defs;
                match defs.get(&name) {
                    None => Ok(0),
                    Some(Def::Func) => Err(format!("function-like macro {name} in #if")),
                    Some(Def::Obj(v)) => {
                        if v.trim().is_empty() {
                            return Ok(0);
                        }
                        // Textual expansion: the replacement tokens take part in the
                        // surrounding expression's precedence (`#define N 1+2` makes
                        // `N*3` equal 7, not 9).
                        if self.depth > 256 {
                            return Err(format!("macro {name} expands too deeply in #if"));
                        }
                        let toks = etokenize(v)?;
                        self.depth += 1;
                        self.p -= 1;
                        self.t.splice(self.p..self.p + 1, toks);
                        self.primary()
                    }
                }
            }
            other => Err(format!("unexpected {other:?} in #if")),
        }
    }
}

fn eval_expr(expr: &str, defs: &BTreeMap<String, Def>, depth: u32) -> Result<i64, String> {
    let t = etokenize(expr)?;
    let mut e = Eval {
        t,
        p: 0,
        defs,
        depth,
    };
    let v = e.or()?;
    if e.p != e.t.len() {
        return Err(format!("trailing tokens in #if {expr:?}"));
    }
    Ok(v)
}

fn directive_ident(rest: &str) -> String {
    rest.chars().take_while(|c| is_ident_char(*c)).collect()
}

/// Returns the active text with object-like macros expanded on every active line.
pub fn preprocess(text: &str, predef: &BTreeMap<String, String>) -> Result<String, String> {
    let stripped = strip_comments(text);
    let joined = stripped.replace("\\\n", "  ");
    let mut defs: BTreeMap<String, Def> = BTreeMap::new();
    for (k, v) in predef {
        defs.insert(k.clone(), Def::Obj(v.clone()));
    }
    let mut out = String::with_capacity(joined.len());
    let mut stack: Vec<(bool, bool)> = Vec::new();
    let mut active = true;
    for line in joined.split('\n') {
        let s = line.trim();
        if let Some(after) = s.strip_prefix('#') {
            let after = after.trim_start();
            let d: String = directive_ident(after);
            let rest = after[d.len()..].trim();
            match d.as_str() {
                "if" | "ifdef" | "ifndef" => {
                    let v = if !active {
                        false
                    } else if d == "if" {
                        eval_expr(rest, &defs, 0)? != 0
                    } else if d == "ifdef" {
                        defs.contains_key(&directive_ident(rest))
                    } else {
                        !defs.contains_key(&directive_ident(rest))
                    };
                    stack.push((active, v));
                    active = active && v;
                }
                "elif" => {
                    let (pa, taken) = stack.pop().ok_or("#elif without #if")?;
                    if pa && !taken {
                        let v = eval_expr(rest, &defs, 0)? != 0;
                        stack.push((pa, v));
                        active = v;
                    } else {
                        stack.push((pa, true));
                        active = false;
                    }
                }
                "else" => {
                    let (pa, taken) = stack.pop().ok_or("#else without #if")?;
                    stack.push((pa, true));
                    active = pa && !taken;
                }
                "endif" => {
                    let (pa, _) = stack.pop().ok_or("#endif without #if")?;
                    active = pa;
                }
                _ if !active => {}
                "define" => {
                    let name = directive_ident(rest);
                    if name.is_empty() {
                        return Err(format!("bad #define {rest:?}"));
                    }
                    let after_name = &rest[name.len()..];
                    if after_name.starts_with('(') {
                        defs.insert(name, Def::Func);
                    } else {
                        defs.insert(name, Def::Obj(after_name.trim().to_string()));
                    }
                }
                "undef" => {
                    defs.remove(&directive_ident(rest));
                }
                "include" | "pragma" | "error" | "warning" | "line" => {}
                other => return Err(format!("unsupported directive #{other}")),
            }
            out.push('\n');
        } else {
            if active {
                out.push_str(&expand_obj(line, &defs, 0)?);
            }
            out.push('\n');
        }
    }
    if !stack.is_empty() {
        return Err("unterminated #if".into());
    }
    Ok(out)
}

fn type_table(ty: &str) -> Option<(u32, char)> {
    Some(match ty {
        "int" | "int32_t" | "unsigned" | "unsigned int" | "uint32_t" => (4, 'i'),
        "float" => (4, 'f'),
        "long long" | "unsigned long long" | "uint64_t" | "int64_t" | "size_t" | "long"
        | "unsigned long" => (8, 'i'),
        "double" => (8, 'f'),
        "_Float16" | "__half" | "half" | "__hip_bfloat16" => (2, 'f'),
        "unsigned short" | "uint16_t" | "short" => (2, 'i'),
        "char" | "unsigned char" | "uint8_t" | "bool" | "int8_t" => (1, 'i'),
        _ => return None,
    })
}

fn parse_arg(a: &str) -> Result<Arg, String> {
    // Only plain `type [qualifiers] [*] name [array]` declarators are understood:
    // anything else (function pointers, parenthesised names, defaults, references)
    // could change the real parameter size, so it fails closed.
    if a.contains('(') || a.contains(')') || a.contains('=') || a.contains('&') {
        return Err(format!(
            "unsupported kernel parameter declarator {:?}",
            a.trim()
        ));
    }
    let words: Vec<&str> = a
        .split_whitespace()
        .filter(|w| !matches!(*w, "__restrict__" | "const" | "volatile"))
        .collect();
    if words.is_empty() {
        return Err("empty kernel argument".into());
    }
    let joined = words.join(" ");
    // Pointers, and arrays (which adjust to pointers as parameters), are 8 bytes.
    if joined.contains('*') || joined.contains('[') {
        return Ok(Arg {
            offset: 0,
            size: 8,
            kind: 'p',
            text: a.trim().to_string(),
        });
    }
    // Last word is the parameter name unless the whole text is a bare type.
    let (ty, _) = if type_table(&joined).is_some() {
        (joined.clone(), String::new())
    } else {
        let (head, name) = words.split_at(words.len() - 1);
        (head.join(" "), name[0].to_string())
    };
    let (size, kind) =
        type_table(&ty).ok_or_else(|| format!("unknown kernel argument type {ty:?} in {a:?}"))?;
    Ok(Arg {
        offset: 0,
        size,
        kind,
        text: a.trim().to_string(),
    })
}

fn skip_balanced(c: &[char], mut i: usize) -> Result<usize, String> {
    // c[i] == '('; return index after the matching ')'.
    let mut depth = 0i32;
    while i < c.len() {
        if c[i] == '(' {
            depth += 1;
        } else if c[i] == ')' {
            depth -= 1;
            if depth == 0 {
                return Ok(i + 1);
            }
        }
        i += 1;
    }
    Err("unbalanced parentheses in kernel declaration".into())
}

fn skip_ws(c: &[char], mut i: usize) -> usize {
    while i < c.len() && c[i].is_whitespace() {
        i += 1;
    }
    i
}

fn starts_with_word(c: &[char], i: usize, w: &str) -> bool {
    let wc: Vec<char> = w.chars().collect();
    if i + wc.len() > c.len() || c[i..i + wc.len()] != wc[..] {
        return false;
    }
    !(i + wc.len() < c.len() && is_ident_char(c[i + wc.len()]))
}

/// `__global__` entry points of already-preprocessed text: symbol -> arguments.
pub fn parse_globals(active: &str) -> Result<Vec<(String, Decl)>, String> {
    let c: Vec<char> = active.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < c.len() {
        if starts_with_word(&c, i, "__global__") && (i == 0 || !is_ident_char(c[i - 1])) {
            let mut j = skip_ws(&c, i + "__global__".len());
            loop {
                if starts_with_word(&c, j, "__launch_bounds__")
                    || starts_with_word(&c, j, "__attribute__")
                {
                    while j < c.len() && c[j] != '(' {
                        j += 1;
                    }
                    j = skip_ws(&c, skip_balanced(&c, j)?);
                } else if starts_with_word(&c, j, "extern") {
                    j = skip_ws(&c, j + "extern".len());
                    if j < c.len() && c[j] == '"' {
                        j += 1;
                        while j < c.len() && c[j] != '"' {
                            j += 1;
                        }
                        j = skip_ws(&c, j + 1);
                    }
                } else if starts_with_word(&c, j, "__global__") {
                    j = skip_ws(&c, j + "__global__".len());
                } else {
                    break;
                }
            }
            if !starts_with_word(&c, j, "void") {
                let ctx: String = c[j..(j + 40).min(c.len())].iter().collect();
                return Err(format!("__global__ without void return near {ctx:?}"));
            }
            j = skip_ws(&c, j + 4);
            let ns = j;
            while j < c.len() && is_ident_char(c[j]) {
                j += 1;
            }
            let name: String = c[ns..j].iter().collect();
            j = skip_ws(&c, j);
            if name.is_empty() || j >= c.len() || c[j] != '(' {
                return Err(format!("malformed __global__ declaration for {name:?}"));
            }
            let end = skip_balanced(&c, j)?;
            let params: String = c[j + 1..end - 1].iter().collect();
            let mut args = Vec::new();
            for p in params.split(',') {
                if p.trim().is_empty() {
                    continue;
                }
                args.push(parse_arg(p)?);
            }
            let mut off = 0u32;
            for a in args.iter_mut() {
                off = (off + a.size - 1) / a.size * a.size;
                a.offset = off;
                off += a.size;
            }
            out.push((name, Decl { args, end: off }));
            i = end;
        } else {
            i += 1;
        }
    }
    Ok(out)
}

pub fn same_layout(a: &Decl, b: &Decl) -> bool {
    a.end == b.end
        && a.args.len() == b.args.len()
        && a.args.iter().zip(&b.args).all(|(x, y)| {
            x.offset == y.offset && x.size == y.size && kinds_compatible(x.kind, y.kind)
        })
}

/// All kernels of `text`, evaluated under every architecture. A symbol whose
/// layout is not identical wherever it is defined is an error.
pub fn decls_of(text: &str) -> Result<BTreeMap<String, Decl>, String> {
    let mut out: BTreeMap<String, Decl> = BTreeMap::new();
    for (arch, macros) in ARCHES.iter() {
        let mut predef: BTreeMap<String, String> = BTreeMap::new();
        predef.insert("__HIP_DEVICE_COMPILE__".into(), "1".into());
        for m in macros {
            predef.insert((*m).to_string(), "1".into());
        }
        let active = preprocess(text, &predef).map_err(|e| format!("{arch}: {e}"))?;
        for (name, decl) in parse_globals(&active).map_err(|e| format!("{arch}: {e}"))? {
            match out.get(&name) {
                Some(prev) if !same_layout(prev, &decl) => {
                    return Err(format!(
                        "{name}: argument layout depends on the target ({arch})"
                    ))
                }
                Some(_) => {}
                None => {
                    out.insert(name, decl);
                }
            }
        }
    }
    Ok(out)
}

/// Same argument class: a PM image only says "by value" (`v`) for a scalar, which is the
/// same bytes as a HIP `int`/`float`. A pointer never matches a scalar.
fn kinds_compatible(a: char, b: char) -> bool {
    a == b || (a == 'v' && b != 'p') || (b == 'v' && a != 'p')
}
