//! Independent declared-ABI reader for HIP kernel sources.
//!
//! A small C preprocessor (object-like macros, conditionals, parameter macros)
//! feeds a `__global__` signature parser. It evaluates every kernel under each
//! target architecture's predefined macros and refuses (fail closed) any
//! symbol whose argument layout differs between architectures, so the layout
//! it reports is the one every compile of that source produces.

use std::collections::BTreeMap;

/// One scalar member, or a fixed-size array of scalar members, of a source-proven struct.
/// Array elements have stride `size`; offsets become absolute when the argument is placed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub offset: u32,
    pub size: u32,
    pub kind: char,
    pub count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arg {
    pub offset: u32,
    pub size: u32,
    /// `p` pointer, `i` integer, `f` floating point, `v` by-value of unknown class.
    pub kind: char,
    pub text: String,
    /// Source-proven scalar fields. Empty for scalar/pointer and opaque metadata arguments.
    pub fields: Vec<Field>,
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
                fields: Vec::new(),
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
    /// Function-like macro: parameter names and replacement text. Only macros whose
    /// replacement defines a `__global__` entry are ever expanded (see `expand_obj`).
    Func {
        params: Vec<String>,
        body: String,
    },
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
            let mut next = j;
            match defs.get(&w) {
                Some(Def::Obj(v)) => out.push_str(&expand_obj(v, defs, depth + 1)?),
                // A function-like macro whose replacement defines a `__global__` entry
                // really declares kernels at every invocation: expand it, or those
                // kernels would vanish from the declared-ABI index. Any other
                // function-like macro is left as written (it cannot declare an entry).
                Some(Def::Func { params, body }) if body.contains("__global__") => {
                    let (expanded, after) =
                        expand_kernel_macro(&w, params, body, &c, j, defs, depth)?;
                    out.push_str(&expanded);
                    next = after;
                }
                _ => out.push_str(&w),
            }
            i = next;
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

/// Parses the `(PARAMS) BODY` text after the name of a function-like `#define`.
fn parse_func_macro(after_name: &str) -> Result<Def, String> {
    let close = after_name
        .find(')')
        .ok_or_else(|| format!("unterminated function-like macro parameter list {after_name:?}"))?;
    let params: Vec<String> = after_name[1..close]
        .split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    Ok(Def::Func {
        params,
        body: after_name[close + 1..].trim().to_string(),
    })
}

/// Error text of a kernel-macro invocation whose balanced `)` is not in the text given.
/// `preprocess` answers it by appending the next source line and expanding again.
const OPEN_INVOCATION: &str = "the invocation is not closed before the end of the active text";

/// Expands one invocation of a kernel-defining function-like macro starting at the
/// identifier that ends at `c[after_name]`: returns the expansion and the index after
/// the invocation's closing parenthesis. Fail closed on everything this reader does
/// not model: variadics, stringification, non-identifier/non-decimal token pastes,
/// string/char invocation arguments, unclosed arguments (a multi-line invocation is joined by
/// `preprocess` first), and argument-count mismatches.
fn expand_kernel_macro(
    name: &str,
    params: &[String],
    body: &str,
    c: &[char],
    after_name: usize,
    defs: &BTreeMap<String, Def>,
    depth: u32,
) -> Result<(String, usize), String> {
    if params.iter().any(|p| p.contains("...")) {
        return Err(format!(
            "kernel-defining macro {name} uses variadics, which this reader does not expand"
        ));
    }
    let mut k = after_name;
    while k < c.len() && c[k].is_whitespace() {
        k += 1;
    }
    if k >= c.len() {
        // The argument list may start on a later line of the same invocation.
        return Err(format!("kernel-defining macro {name}: {OPEN_INVOCATION}"));
    }
    if c[k] != '(' {
        return Err(format!(
            "kernel-defining function-like macro {name} is used without an argument list"
        ));
    }
    let mut nest = 0i32;
    let mut args: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut end: Option<usize> = None;
    let mut q = k;
    while q < c.len() {
        let ch = c[q];
        match ch {
            '"' | '\'' => {
                return Err(format!(
                    "kernel-defining macro {name}: string or character literal in an invocation argument"
                ))
            }
            '(' | '[' | '{' => {
                nest += 1;
                if nest > 1 {
                    cur.push(ch);
                }
            }
            ')' | ']' | '}' => {
                nest -= 1;
                if nest == 0 {
                    end = Some(q + 1);
                    break;
                }
                cur.push(ch);
            }
            ',' if nest == 1 => args.push(std::mem::take(&mut cur)),
            _ => cur.push(ch),
        }
        q += 1;
    }
    let end = end.ok_or_else(|| format!("kernel-defining macro {name}: {OPEN_INVOCATION}"))?;
    args.push(cur);
    if params.is_empty() && args.len() == 1 && args[0].trim().is_empty() {
        args.clear();
    }
    if args.len() != params.len() {
        return Err(format!(
            "kernel-defining macro {name} takes {} arguments, invoked with {}",
            params.len(),
            args.len()
        ));
    }
    let mut actual: BTreeMap<&str, String> = BTreeMap::new();
    for (p, a) in params.iter().zip(args.iter()) {
        actual.insert(p.as_str(), expand_obj(a.trim(), defs, depth + 1)?);
    }
    let b: Vec<char> = body.chars().collect();
    let raw: BTreeMap<&str, &str> = params
        .iter()
        .zip(&args)
        .map(|(p, a)| (p.as_str(), a.trim()))
        .collect();
    let mut sub = String::with_capacity(body.len());
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == '_' || b[i].is_ascii_alphabetic() {
            let mut j = i;
            while j < b.len() && is_ident_char(b[j]) {
                j += 1;
            }
            let w: String = b[i..j].iter().collect();
            let next = b[j..]
                .iter()
                .position(|c| !c.is_whitespace())
                .map(|k| j + k);
            let pasted = next.is_some_and(|k| b[k] == '#' && b.get(k + 1) == Some(&'#'));
            if pasted {
                sub.push_str(raw.get(w.as_str()).copied().unwrap_or(&w));
            } else {
                match actual.get(w.as_str()) {
                    Some(a) => sub.push_str(a),
                    None => sub.push_str(&w),
                }
            }
            i = j;
        } else if b[i].is_ascii_digit() {
            let mut j = i;
            while j < b.len() && is_ident_char(b[j]) {
                j += 1;
            }
            sub.extend(b[i..j].iter());
            i = j;
        } else if b[i] == '#' {
            if b.get(i + 1) != Some(&'#') {
                return Err(format!(
                    "kernel-defining macro {name}: stringification is unsupported"
                ));
            }
            while sub.ends_with(char::is_whitespace) {
                sub.pop();
            }
            let left = sub
                .char_indices()
                .rev()
                .take_while(|(_, c)| is_ident_char(*c))
                .last()
                .map(|(k, _)| k)
                .ok_or_else(|| format!("kernel-defining macro {name}: missing paste operand"))?;
            // The raw operand must be exactly one supported preprocessing token.
            if left > 0
                && sub[..left]
                    .chars()
                    .last()
                    .is_some_and(|c| !c.is_whitespace() && c != '(' && c != ',' && c != '*')
            {
                return Err(format!(
                    "kernel-defining macro {name}: ambiguous paste operand"
                ));
            }
            let mut start = i + 2;
            while b.get(start).is_some_and(|c| c.is_whitespace()) {
                start += 1;
            }
            let mut end = start;
            while b.get(end).is_some_and(|c| is_ident_char(*c)) {
                end += 1;
            }
            let word: String = b[start..end].iter().collect();
            let right = raw.get(word.as_str()).copied().unwrap_or(&word);
            let token = |s: &str| {
                !s.is_empty()
                    && s.chars().all(is_ident_char)
                    && (s.starts_with(|c: char| c == '_' || c.is_ascii_alphabetic())
                        || s.chars().all(|c| c.is_ascii_digit()))
            };
            if !token(&sub[left..]) || !token(right) {
                return Err(format!(
                    "kernel-defining macro {name}: unsupported paste operand"
                ));
            }
            let joined = format!("{}{right}", &sub[left..]);
            if !token(&joined) {
                return Err(format!(
                    "kernel-defining macro {name}: unsupported pasted token"
                ));
            }
            sub.truncate(left);
            sub.push_str(&joined);
            i = end;
        } else {
            sub.push(b[i]);
            i += 1;
        }
    }
    Ok((expand_obj(&sub, defs, depth + 1)?, end))
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
                    Some(Def::Func { .. }) => Err(format!("function-like macro {name} in #if")),
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
    let lines: Vec<&str> = joined.split('\n').collect();
    let mut idx = 0usize;
    while idx < lines.len() {
        let line = lines[idx];
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
                        defs.insert(name, parse_func_macro(after_name)?);
                    } else {
                        defs.insert(name, Def::Obj(after_name.trim().to_string()));
                    }
                }
                "undef" => {
                    defs.remove(&directive_ident(rest));
                }
                "pragma" if active && rest.trim_start().starts_with("pack") => {
                    return Err("unsupported #pragma pack changes aggregate alignment".into());
                }
                "include" | "pragma" | "error" | "warning" | "line" => {}
                other => return Err(format!("unsupported directive #{other}")),
            }
            out.push('\n');
        } else {
            if active {
                // A kernel-defining macro invocation may span several source lines: extend
                // the text with the following plain lines until its parentheses balance.
                // A directive line, or the end of the file, inside the invocation fails closed.
                let mut text = line.to_string();
                let mut extra = 0usize;
                loop {
                    match expand_obj(&text, &defs, 0) {
                        Ok(expanded) => {
                            out.push_str(&expanded);
                            break;
                        }
                        Err(e) if e.contains(OPEN_INVOCATION) => {
                            let next = lines
                                .get(idx + extra + 1)
                                .filter(|l| !l.trim().starts_with('#'));
                            match next {
                                Some(l) => {
                                    text.push(' ');
                                    text.push_str(l);
                                    extra += 1;
                                }
                                None => return Err(e),
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
                // Keep one output line per source line.
                for _ in 0..extra {
                    out.push('\n');
                }
                idx += extra;
            }
            out.push('\n');
        }
        idx += 1;
    }
    if !stack.is_empty() {
        return Err("unterminated #if".into());
    }
    Ok(out)
}

/// Natural C layout of a named, plain struct. Only scalar fields and fixed-size scalar
/// arrays are accepted; unsupported declarations never acquire a guessed layout.
fn struct_layout(active: &str, ty: &str) -> Result<(u32, u32, Vec<Field>), String> {
    let name = ty.strip_prefix("struct ").unwrap_or(ty);
    if name.is_empty() || !name.chars().all(is_ident_char) {
        return Err(format!("unsupported aggregate type {ty:?}"));
    }
    let c: Vec<char> = active.chars().collect();
    let mut definitions = Vec::new();
    let mut i = 0;
    let mut depth = 0u32;
    while i < c.len() {
        if c[i] == '"' || c[i] == '\'' {
            let quote = c[i];
            i += 1;
            while i < c.len() && c[i] != quote {
                if c[i] == '\\' {
                    i += 1;
                }
                i += 1;
            }
            if i == c.len() {
                return Err("unterminated literal while reading aggregate".into());
            }
            i += 1;
            continue;
        }
        if starts_with_word(&c, i, "struct") && (i == 0 || !is_ident_char(c[i - 1])) {
            let start = skip_ws(&c, i + 6);
            let mut end = start;
            while end < c.len() && is_ident_char(c[end]) {
                end += 1;
            }
            if c[start..end].iter().collect::<String>() == name {
                let body = skip_ws(&c, end);
                if body < c.len() && c[body] == '{' {
                    let previous = c[..i].iter().rev().find(|ch| !ch.is_whitespace());
                    if depth != 0 || previous.is_some_and(|ch| !matches!(ch, ';' | '{' | '}')) {
                        return Err(format!(
                            "struct {name}: unsupported scope or declaration prefix"
                        ));
                    }
                    let close = c[body + 1..]
                        .iter()
                        .position(|&ch| ch == '}')
                        .map(|n| body + 1 + n)
                        .ok_or_else(|| format!("unterminated struct {name}"))?;
                    if c[body + 1..close].contains(&'{')
                        || c.get(skip_ws(&c, close + 1)) != Some(&';')
                    {
                        return Err(format!("unsupported struct definition {name}"));
                    }
                    definitions.push(c[body + 1..close].iter().collect::<String>());
                } else if body < c.len() && c[body] != ';' {
                    // Attributes, inheritance and other definition modifiers are not plain C.
                    // Ordinary references (`struct T *p`) do not define a type.
                    let next = c[body..].iter().position(|&ch| ch == ';').map(|n| body + n);
                    if next.is_some_and(|end| c[body..end].contains(&'{')) {
                        return Err(format!("unsupported struct definition {name}"));
                    }
                }
            }
        }
        match c[i] {
            '{' => depth = depth.checked_add(1).ok_or("aggregate scope overflow")?,
            '}' => depth = depth.checked_sub(1).ok_or("unbalanced aggregate scope")?,
            _ => {}
        }
        i += 1;
    }
    if definitions.len() != 1 {
        return Err(format!(
            "aggregate {name}: expected one plain struct definition, got {}",
            definitions.len()
        ));
    }
    let fields = &definitions[0];
    if !fields.trim_end().ends_with(';') {
        return Err(format!("struct {name}: missing field terminator"));
    }
    let mut offset = 0u32;
    let mut alignment = 1u32;
    let mut count = 0usize;
    let mut names = std::collections::BTreeSet::new();
    let mut layout = Vec::new();
    for field in fields.split(';').filter(|s| !s.trim().is_empty()) {
        let (declaration, extent) = if let Some((head, rest)) = field.split_once('[') {
            let number = rest
                .strip_suffix(']')
                .ok_or("unsupported aggregate array")?
                .trim();
            if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit()) {
                return Err(format!("struct {name}: nonliteral array bound {number:?}"));
            }
            let extent = number
                .parse::<u32>()
                .map_err(|_| "aggregate array bound overflow")?;
            if extent == 0 {
                return Err(format!("struct {name}: zero array bound"));
            }
            (head.trim(), extent)
        } else {
            (field.trim(), 1)
        };
        let (ty, field_name) = declaration
            .rsplit_once(char::is_whitespace)
            .ok_or_else(|| format!("struct {name}: unsupported field {field:?}"))?;
        if field_name.is_empty()
            || !field_name.chars().all(is_ident_char)
            || !field_name
                .chars()
                .next()
                .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
            || type_table(field_name).is_some()
            || !names.insert(field_name)
        {
            return Err(format!("struct {name}: unsupported field {field:?}"));
        }
        let (size, kind) = type_table(ty.trim())
            .ok_or_else(|| format!("struct {name}: unsupported field type {ty:?}"))?;
        alignment = alignment.max(size);
        offset = offset
            .checked_add(size - 1)
            .ok_or("aggregate alignment overflow")?
            / size
            * size;
        layout.push(Field {
            offset,
            size,
            kind,
            count: extent,
        });
        offset = offset
            .checked_add(
                size.checked_mul(extent)
                    .ok_or("aggregate array size overflow")?,
            )
            .ok_or("aggregate size overflow")?;
        count += 1;
    }
    if count == 0 {
        return Err(format!("struct {name}: empty aggregate"));
    }
    let size = offset
        .checked_add(alignment - 1)
        .ok_or("aggregate tail alignment overflow")?
        / alignment
        * alignment;
    Ok((size, alignment, layout))
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

fn parse_arg(a: &str, active: &str) -> Result<(Arg, u32), String> {
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
        return Ok((
            Arg {
                offset: 0,
                size: 8,
                kind: 'p',
                text: a.trim().to_string(),
                fields: Vec::new(),
            },
            8,
        ));
    }
    // Last word is the parameter name unless the whole text is a bare type.
    let (ty, _) = if type_table(&joined).is_some() {
        (joined.clone(), String::new())
    } else {
        let (head, name) = words.split_at(words.len() - 1);
        (head.join(" "), name[0].to_string())
    };
    let (size, kind, alignment, fields) = match type_table(&ty) {
        Some((size, kind)) => (size, kind, size, Vec::new()),
        None => {
            let (size, alignment, fields) = struct_layout(active, &ty)
                .map_err(|e| format!("unknown kernel argument type {ty:?} in {a:?}: {e}"))?;
            (size, 'v', alignment, fields)
        }
    };
    Ok((
        Arg {
            offset: 0,
            size,
            kind,
            text: a.trim().to_string(),
            fields,
        },
        alignment,
    ))
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
                args.push(parse_arg(p, active)?);
            }
            let mut off = 0u32;
            for (a, alignment) in args.iter_mut() {
                off = off
                    .checked_add(*alignment - 1)
                    .ok_or("kernel argument alignment overflow")?
                    / *alignment
                    * *alignment;
                a.offset = off;
                for field in &mut a.fields {
                    field.offset = field
                        .offset
                        .checked_add(off)
                        .ok_or("aggregate field offset overflow")?;
                }
                off = off
                    .checked_add(a.size)
                    .ok_or("kernel argument size overflow")?;
            }
            out.push((
                name,
                Decl {
                    args: args.into_iter().map(|(a, _)| a).collect(),
                    end: off,
                },
            ));
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
                Some(prev)
                    if !same_layout(prev, &decl)
                        || prev
                            .args
                            .iter()
                            .zip(&decl.args)
                            .any(|(a, b)| a.fields != b.fields) =>
                {
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

#[cfg(test)]
mod aggregate_tests {
    use super::*;

    #[test]
    fn plain_struct_size_and_alignment_are_independent() {
        let source = "struct Yarn { float scale; float low; float high; float magnitude; };
            struct Lanes { int positions[128]; int rows[128]; };
            struct Padded { char first; double second; char last; };
            __global__ void k(char prefix, Yarn yarn, Lanes lanes, int suffix, Padded padded) {}";
        let declarations = decls_of(source).expect("plain aggregates");
        let args = &declarations["k"].args;
        assert_eq!(
            args.iter()
                .map(|a| (a.offset, a.size, a.kind))
                .collect::<Vec<_>>(),
            vec![
                (0, 1, 'i'),
                (4, 16, 'v'),
                (20, 1024, 'v'),
                (1044, 4, 'i'),
                (1048, 24, 'v')
            ]
        );
        assert_eq!(declarations["k"].end, 1072);
        assert_eq!(
            args[1]
                .fields
                .iter()
                .map(|f| (f.offset, f.size, f.kind, f.count))
                .collect::<Vec<_>>(),
            vec![
                (4, 4, 'f', 1),
                (8, 4, 'f', 1),
                (12, 4, 'f', 1),
                (16, 4, 'f', 1)
            ]
        );
        assert_eq!(
            args[2]
                .fields
                .iter()
                .map(|f| (f.offset, f.size, f.kind, f.count))
                .collect::<Vec<_>>(),
            vec![(20, 4, 'i', 128), (532, 4, 'i', 128)]
        );
    }

    #[test]
    fn unsupported_aggregate_declarations_refuse_instead_of_guessing() {
        for definition in [
            "struct T {};",
            "struct T { float a; float a; };",
            "struct T { float *p; };",
            "struct T { float a[0]; };",
            "struct T { float a[]; };",
            "struct T { float a[1 + 1]; };",
            "struct T { float a[2][2]; };",
            "struct T { float a[4294967295]; };",
            "struct T { float a:1; };",
            "struct T { float a = 1; };",
            "struct T { static float a; };",
            "struct T { float a, b; };",
            "struct T { float a };",
            "struct T { struct Nested { float a; } b; };",
            "struct T { float a; }; struct T { float b; };",
            "struct __attribute__((packed)) T { float a; };",
            "__attribute__((packed)) struct T { float a; };",
            "struct T { float a; } __attribute__((packed));",
            "struct alignas(16) T { float a; };",
            "struct T : Base { float a; };",
            "union T { float a; int b; };",
            "void helper() { struct T { float a; }; }",
            "namespace hidden { struct T { float a; }; }",
            "#pragma pack(push, 1)\nstruct T { float a; };",
        ] {
            let source = format!("{definition}\n__global__ void k(T value) {{}}");
            assert!(decls_of(&source).is_err(), "accepted {definition}");
        }
    }

    #[test]
    fn aggregate_architecture_drift_and_parameter_overflow_refuse() {
        let source = "struct T {
            #if defined(__gfx942__)
            double value;
            #else
            float value;
            #endif
            };
            __global__ void k(T value) {}";
        assert!(decls_of(source).is_err());
        let overflow = "struct T { char bytes[4294967295]; };
            __global__ void k(T value, int next) {}";
        assert!(decls_of(overflow).is_err());
        let macros = "#define N 128\nstruct T { int first[N]; int second[N]; };
            __global__ void k(struct T value) {}";
        assert_eq!(decls_of(macros).unwrap()["k"].args[0].size, 1024);
        let same_size_drift = "struct T {
            #if defined(__gfx942__)
            int first; float second;
            #else
            float first; int second;
            #endif
            };
            __global__ void k(T value) {}";
        assert!(decls_of(same_size_drift).is_err());
        let huge = "struct T { char bytes[4294967295]; }; __global__ void k(T value) {}";
        let declaration = decls_of(huge).expect("symbolic array does not allocate per element");
        assert_eq!(declaration["k"].args[0].fields.len(), 1);
        assert_eq!(declaration["k"].args[0].fields[0].count, u32::MAX);
    }
}

#[cfg(test)]
mod multiline_macro_tests {
    use super::*;

    const DEFS: &str = "#define KERNEL(NAME, BOUNDS, T)            \\
    extern \"C\" __global__ BOUNDS void NAME(const T* x, int n) {}
";

    fn names(source: &str) -> Result<Vec<(String, u32)>, String> {
        let d = decls_of(source)?;
        Ok(d.into_iter().map(|(k, v)| (k, v.end)).collect())
    }

    #[test]
    fn invocation_split_across_lines_expands_like_the_single_line_form() {
        let one = format!("{DEFS}KERNEL(k_one, __launch_bounds__(256), float)\n");
        let many = format!("{DEFS}KERNEL(k_one,\n        __launch_bounds__(256),\n   float\n)\n");
        assert_eq!(names(&one).unwrap(), vec![("k_one".to_string(), 12)]);
        assert_eq!(names(&many).unwrap(), names(&one).unwrap());
        // The name alone on a line with the argument list on the next lines.
        let split_name = format!("{DEFS}KERNEL\n(k_one, __launch_bounds__(256),\nfloat)\n");
        assert_eq!(names(&split_name).unwrap(), names(&one).unwrap());
    }

    #[test]
    fn several_multiline_invocations_and_following_code_are_kept() {
        let source = format!(
            "{DEFS}KERNEL(a, __launch_bounds__(64),\n       float)\nKERNEL(b, ,\n       double)\n__global__ void c(int z) {{}}\n"
        );
        let got = names(&source).unwrap();
        assert_eq!(
            got,
            vec![
                ("a".to_string(), 12),
                ("b".to_string(), 12),
                ("c".to_string(), 4)
            ]
        );
    }

    #[test]
    fn comments_inside_a_multiline_invocation_do_not_count() {
        let source = format!(
            "{DEFS}KERNEL(k, // closing ) in a line comment\n  /* ) and ( in a\n     block comment */ __launch_bounds__(128),\n  float)\n"
        );
        assert_eq!(names(&source).unwrap(), vec![("k".to_string(), 12)]);
    }

    #[test]
    fn nested_brackets_across_lines_balance() {
        let source = format!("{DEFS}KERNEL(k,\n  __launch_bounds__((128 +\n  (64))),\n  float)\n");
        assert_eq!(names(&source).unwrap(), vec![("k".to_string(), 12)]);
    }

    #[test]
    fn unbalanced_or_interrupted_invocations_fail_closed() {
        // End of file before the closing parenthesis.
        let eof = format!("{DEFS}KERNEL(k, __launch_bounds__(64),\n  float\n");
        assert!(names(&eof).unwrap_err().contains("not closed"));
        // A preprocessor directive inside the argument list is not modelled.
        let directive =
            format!("{DEFS}KERNEL(k,\n#if 1\n  __launch_bounds__(64),\n#endif\n float)\n");
        assert!(names(&directive).unwrap_err().contains("not closed"));
        // Name with no argument list at all.
        let bare = format!("{DEFS}KERNEL\n");
        assert!(names(&bare).is_err());
        // Argument-count and literal checks still apply to the joined invocation.
        let count = format!("{DEFS}KERNEL(k,\n  float)\n");
        assert!(names(&count).unwrap_err().contains("takes 3 arguments"));
        let literal = format!("{DEFS}KERNEL(k,\n  \"a)\",\n float)\n");
        assert!(names(&literal).unwrap_err().contains("literal"));
    }

    #[test]
    fn multiline_calls_of_non_kernel_macros_are_untouched() {
        let source = format!(
            "{DEFS}#define HELPER(A, B) A + B\nint v = HELPER(1,\n  2);\n__global__ void k(int z) {{}}\n"
        );
        assert_eq!(names(&source).unwrap(), vec![("k".to_string(), 4)]);
    }
}
