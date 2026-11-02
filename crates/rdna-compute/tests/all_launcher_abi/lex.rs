//! Std-only Rust lexer, token trees and a module/impl item walker.
//!
//! The launcher scan needs token-level structure (call sites, `let` bindings,
//! array literals, closures), never types or name resolution, so a small
//! lexer is enough and keeps the gate free of new dependencies.

#[derive(Clone, Debug)]
pub enum Tt {
    /// Identifier or keyword.
    Id(String, u32),
    /// Single punctuation character.
    P(char, u32),
    /// String literal (raw source text between the quotes, escapes kept).
    S(String, u32),
    /// Number literal, char literal or lifetime.
    N(String, u32),
    /// Delimited group: `(`, `[` or `{` plus its children.
    G(char, Vec<Tt>, u32),
}

impl Tt {
    pub fn is_p(&self, c: char) -> bool {
        matches!(self, Tt::P(x, _) if *x == c)
    }
    pub fn is_id(&self, s: &str) -> bool {
        matches!(self, Tt::Id(x, _) if x == s)
    }
    pub fn id(&self) -> Option<&str> {
        match self {
            Tt::Id(x, _) => Some(x.as_str()),
            _ => None,
        }
    }
    pub fn is_g(&self, d: char) -> bool {
        matches!(self, Tt::G(x, _, _) if *x == d)
    }
    pub fn group(&self) -> Option<&[Tt]> {
        match self {
            Tt::G(_, c, _) => Some(c.as_slice()),
            _ => None,
        }
    }
    pub fn line(&self) -> u32 {
        match self {
            Tt::Id(_, l) | Tt::P(_, l) | Tt::S(_, l) | Tt::N(_, l) | Tt::G(_, _, l) => *l,
        }
    }
}

/// Token text for diagnostics and for the few places that match shapes by text.
pub fn text(ts: &[Tt]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for t in ts {
        parts.push(match t {
            Tt::Id(s, _) | Tt::N(s, _) => s.clone(),
            Tt::P(c, _) => c.to_string(),
            Tt::S(s, _) => format!("\"{s}\""),
            Tt::G(d, c, _) => {
                let (o, e) = match d {
                    '(' => ('(', ')'),
                    '[' => ('[', ']'),
                    _ => ('{', '}'),
                };
                format!("{o}{}{e}", text(c))
            }
        });
    }
    parts.join(" ")
}

/// Split a token list at top-level occurrences of `sep`. A trailing separator
/// does not produce an empty tail.
pub fn split_top(ts: &[Tt], sep: char) -> Vec<Vec<Tt>> {
    let mut parts: Vec<Vec<Tt>> = vec![Vec::new()];
    for t in ts {
        if t.is_p(sep) {
            parts.push(Vec::new());
        } else {
            parts.last_mut().unwrap().push(t.clone());
        }
    }
    if parts.last().map_or(false, |p| p.is_empty()) {
        parts.pop();
    }
    parts
}

pub fn strings_in(ts: &[Tt], acc: &mut Vec<String>) {
    for t in ts {
        match t {
            Tt::S(s, _) => acc.push(s.clone()),
            Tt::G(_, c, _) => strings_in(c, acc),
            _ => {}
        }
    }
}

pub fn idents_in(ts: &[Tt], acc: &mut Vec<String>) {
    for t in ts {
        match t {
            Tt::Id(s, _) => acc.push(s.clone()),
            Tt::G(_, c, _) => idents_in(c, acc),
            _ => {}
        }
    }
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}
fn is_ident_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

pub fn lex(src: &str) -> Result<Vec<Tt>, String> {
    let c: Vec<char> = src.chars().collect();
    let n = c.len();
    let mut i = 0usize;
    let mut line = 1u32;
    let mut stack: Vec<Vec<Tt>> = vec![Vec::new()];
    let mut dstack: Vec<(char, u32)> = Vec::new();
    while i < n {
        let ch = c[i];
        if ch == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '/' && i + 1 < n && c[i + 1] == '/' {
            while i < n && c[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if ch == '/' && i + 1 < n && c[i + 1] == '*' {
            let mut depth = 1;
            i += 2;
            while i < n && depth > 0 {
                if c[i] == '/' && i + 1 < n && c[i + 1] == '*' {
                    depth += 1;
                    i += 2;
                } else if c[i] == '*' && i + 1 < n && c[i + 1] == '/' {
                    depth -= 1;
                    i += 2;
                } else {
                    if c[i] == '\n' {
                        line += 1;
                    }
                    i += 1;
                }
            }
            continue;
        }
        // Raw strings: r"..", r#".."#, br"..".
        if ch == 'r' || (ch == 'b' && i + 1 < n && c[i + 1] == 'r') {
            let mut j = if ch == 'b' { i + 2 } else { i + 1 };
            let mut hashes = 0usize;
            while j < n && c[j] == '#' {
                hashes += 1;
                j += 1;
            }
            if j < n && c[j] == '"' {
                let start = j + 1;
                let mut k = start;
                let mut end = None;
                while k < n {
                    if c[k] == '"' {
                        let mut h = 0usize;
                        while h < hashes && k + 1 + h < n && c[k + 1 + h] == '#' {
                            h += 1;
                        }
                        if h == hashes {
                            end = Some(k);
                            break;
                        }
                    }
                    k += 1;
                }
                let end = end.ok_or_else(|| format!("line {line}: unterminated raw string"))?;
                let s: String = c[start..end].iter().collect();
                stack.last_mut().unwrap().push(Tt::S(s.clone(), line));
                line += s.chars().filter(|x| *x == '\n').count() as u32;
                i = end + 1 + hashes;
                continue;
            }
        }
        if ch == '"' || (ch == 'b' && i + 1 < n && c[i + 1] == '"') {
            let mut j = if ch == 'b' { i + 2 } else { i + 1 };
            let start = j;
            while j < n && c[j] != '"' {
                if c[j] == '\\' {
                    j += 1;
                }
                j += 1;
            }
            if j >= n {
                return Err(format!("line {line}: unterminated string"));
            }
            let s: String = c[start..j].iter().collect();
            stack.last_mut().unwrap().push(Tt::S(s.clone(), line));
            line += s.chars().filter(|x| *x == '\n').count() as u32;
            i = j + 1;
            continue;
        }
        if ch == '\'' || (ch == 'b' && i + 1 < n && c[i + 1] == '\'') {
            let q = if ch == 'b' { i + 1 } else { i };
            let j = q + 1;
            // Char literal?
            if j < n && c[j] == '\\' {
                let mut k = j + 2;
                let lim = (j + 14).min(n);
                while k < lim && c[k] != '\'' {
                    k += 1;
                }
                if k < lim && c[k] == '\'' {
                    let s: String = c[i..=k].iter().collect();
                    stack.last_mut().unwrap().push(Tt::N(s, line));
                    i = k + 1;
                    continue;
                }
            } else if j + 1 < n && c[j] != '\'' && c[j + 1] == '\'' {
                let s: String = c[i..=j + 1].iter().collect();
                stack.last_mut().unwrap().push(Tt::N(s, line));
                i = j + 2;
                continue;
            }
            // Lifetime.
            let mut k = j;
            while k < n && is_ident_char(c[k]) {
                k += 1;
            }
            let s: String = c[i..k].iter().collect();
            stack.last_mut().unwrap().push(Tt::N(s, line));
            i = k.max(i + 1);
            continue;
        }
        if is_ident_start(ch) {
            let mut j = i;
            while j < n && is_ident_char(c[j]) {
                j += 1;
            }
            let s: String = c[i..j].iter().collect();
            stack.last_mut().unwrap().push(Tt::Id(s, line));
            i = j;
            continue;
        }
        if ch.is_ascii_digit() {
            let mut j = i;
            while j < n && is_ident_char(c[j]) {
                j += 1;
            }
            if j + 1 < n && c[j] == '.' && c[j + 1].is_ascii_digit() {
                j += 1;
                while j < n && is_ident_char(c[j]) {
                    j += 1;
                }
            }
            let s: String = c[i..j].iter().collect();
            stack.last_mut().unwrap().push(Tt::N(s, line));
            i = j;
            continue;
        }
        if ch == '(' || ch == '[' || ch == '{' {
            stack.push(Vec::new());
            dstack.push((ch, line));
            i += 1;
            continue;
        }
        if ch == ')' || ch == ']' || ch == '}' {
            let children = stack.pop().ok_or("unbalanced delimiter")?;
            let (d, l) = dstack
                .pop()
                .ok_or_else(|| format!("line {line}: unbalanced {ch}"))?;
            stack
                .last_mut()
                .ok_or("unbalanced delimiter")?
                .push(Tt::G(d, children, l));
            i += 1;
            continue;
        }
        stack.last_mut().unwrap().push(Tt::P(ch, line));
        i += 1;
    }
    if stack.len() != 1 {
        return Err("unbalanced delimiters at end of file".into());
    }
    Ok(stack.pop().unwrap())
}

/// A function item (free, inherent, trait default or nested).
#[derive(Clone, Debug)]
pub struct FnRec {
    pub name: String,
    /// Attribute texts of the fn and of every enclosing impl/mod/trait.
    pub attrs: Vec<String>,
    pub sig: Vec<Tt>,
    pub body: Vec<Tt>,
    pub line: u32,
    /// The type of the enclosing `impl [Trait for] Type` block (a method's owner).
    pub owner: Option<String>,
}

/// A `const` or `static` item at module or impl level.
#[derive(Clone, Debug)]
pub struct ConstRec {
    pub tokens: Vec<Tt>,
}

impl ConstRec {
    /// Initializer expression tokens (after `=`, without the trailing `;`).
    pub fn init(&self) -> Vec<Tt> {
        after_eq(&self.tokens)
    }
}

pub fn after_eq(tokens: &[Tt]) -> Vec<Tt> {
    for (k, t) in tokens.iter().enumerate() {
        if t.is_p('=') {
            let mut r: Vec<Tt> = tokens[k + 1..].to_vec();
            if r.last().map_or(false, |x| x.is_p(';')) {
                r.pop();
            }
            return r;
        }
    }
    Vec::new()
}

/// `#[cfg(any())]` is the one provably-false configuration predicate.
pub fn attr_is_cfg_any_empty(a: &str) -> bool {
    let compact: String = a.chars().filter(|c| !c.is_whitespace()).collect();
    compact == "cfg(any())"
}

pub fn attr_is_test(a: &str) -> bool {
    let compact: String = a.chars().filter(|c| !c.is_whitespace()).collect();
    compact == "test"
        || compact == "cfg(test)"
        || compact.starts_with("test(")
        || compact.ends_with("::test")
}

pub struct FileItems {
    pub fns: Vec<FnRec>,
    pub consts: Vec<(String, ConstRec)>,
    /// Every `use` item: the identifiers of its path tree, with `*` for a glob.
    pub uses: Vec<Vec<String>>,
    /// Every `struct Name`: its attributes (`derive(..)` among them).
    pub structs: Vec<(String, Vec<String>)>,
}

pub fn walk_items(ts: &[Tt]) -> FileItems {
    let mut out = FileItems {
        fns: Vec::new(),
        consts: Vec::new(),
        uses: Vec::new(),
        structs: Vec::new(),
    };
    go(ts, &[], None, &mut out);
    out
}

fn go(ts: &[Tt], attrs_inh: &[String], owner: Option<&str>, out: &mut FileItems) {
    let n = ts.len();
    let mut i = 0usize;
    let mut attrs: Vec<String> = Vec::new();
    while i < n {
        let t = &ts[i];
        if t.is_p('#') {
            let mut j = i + 1;
            if j < n && ts[j].is_p('!') {
                j += 1;
            }
            if j < n && ts[j].is_g('[') {
                attrs.push(text(ts[j].group().unwrap()));
                i = j + 1;
                continue;
            }
        }
        if t.is_id("struct") && i + 1 < n {
            if let Some(name) = ts[i + 1].id() {
                let mut all = attrs_inh.to_vec();
                all.extend(attrs.iter().cloned());
                out.structs.push((name.to_string(), all));
            }
        }
        if t.is_id("use") {
            let mut j = i + 1;
            let mut names: Vec<String> = Vec::new();
            fn flat(ts: &[Tt], names: &mut Vec<String>) {
                for t in ts {
                    if let Some(id) = t.id() {
                        names.push(id.to_string());
                    } else if t.is_p('*') {
                        names.push("*".to_string());
                    } else if let Some(g) = t.group() {
                        flat(g, names);
                    }
                }
            }
            while j < n && !ts[j].is_p(';') {
                flat(std::slice::from_ref(&ts[j]), &mut names);
                j += 1;
            }
            out.uses.push(names);
            attrs.clear();
            i = j + 1;
            continue;
        }
        if t.is_id("fn") && i + 1 < n {
            let name = ts[i + 1].id().unwrap_or("").to_string();
            let mut j = i + 2;
            while j < n && !ts[j].is_g('{') && !ts[j].is_p(';') {
                j += 1;
            }
            if j < n && ts[j].is_g('{') {
                let mut all = attrs_inh.to_vec();
                all.extend(attrs.iter().cloned());
                let body = ts[j].group().unwrap().to_vec();
                out.fns.push(FnRec {
                    name,
                    attrs: all,
                    sig: ts[i + 2..j].to_vec(),
                    body,
                    line: t.line(),
                    owner: owner.map(str::to_string),
                });
            }
            attrs.clear();
            i = j + 1;
            continue;
        }
        if (t.is_id("impl") || t.is_id("mod") || t.is_id("trait")) && i + 1 < n {
            let mut j = i + 1;
            while j < n && !ts[j].is_g('{') && !ts[j].is_p(';') {
                j += 1;
            }
            if j < n && ts[j].is_g('{') {
                let mut all = attrs_inh.to_vec();
                all.extend(attrs.iter().cloned());
                let inner_owner = if t.is_id("impl") {
                    impl_owner(&ts[i + 1..j])
                } else {
                    None
                };
                go(ts[j].group().unwrap(), &all, inner_owner.as_deref(), out);
            }
            attrs.clear();
            i = j + 1;
            continue;
        }
        if (t.is_id("const") || t.is_id("static")) && i + 1 < n && ts[i + 1].id().is_some() {
            let mut k = i + 1;
            if ts[k].is_id("mut") && k + 1 < n {
                k += 1;
            }
            let kw = ts[k].id().unwrap_or("");
            if kw == "unsafe" || kw == "fn" || kw == "extern" || kw == "async" {
                i += 1;
                continue;
            }
            let name = kw.to_string();
            let mut j = i;
            while j < n && !ts[j].is_p(';') {
                j += 1;
            }
            let end = (j + 1).min(n);
            out.consts.push((
                name,
                ConstRec {
                    tokens: ts[i..end].to_vec(),
                },
            ));
            attrs.clear();
            i = end;
            continue;
        }
        if t.is_g('{') {
            let mut all = attrs_inh.to_vec();
            all.extend(attrs.iter().cloned());
            go(t.group().unwrap(), &all, owner, out);
            attrs.clear();
            i += 1;
            continue;
        }
        if t.is_p(';') {
            attrs.clear();
        }
        i += 1;
    }
}

/// The implemented type of `impl<..> [Trait for] Type<..>` (header tokens after `impl`):
/// the last path segment of the type after `for`, or of the first type when there is no `for`.
fn impl_owner(h: &[Tt]) -> Option<String> {
    let mut i = 0usize;
    if h.first().map_or(false, |t| t.is_p('<')) {
        let mut depth = 0i32;
        while i < h.len() {
            if h[i].is_p('<') {
                depth += 1;
            } else if h[i].is_p('>') {
                depth -= 1;
                if depth == 0 {
                    i += 1;
                    break;
                }
            }
            i += 1;
        }
    }
    let rest = &h[i..];
    let start = rest
        .iter()
        .position(|t| t.is_id("for"))
        .map_or(0, |p| p + 1);
    let mut last: Option<String> = None;
    for t in &rest[start..] {
        if t.is_p('<') || t.is_id("where") {
            break;
        }
        if let Some(id) = t.id() {
            last = Some(id.to_string());
        }
    }
    last
}
