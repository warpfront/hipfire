// SPDX-License-Identifier: Apache-2.0
//! Static coverage gate: every HIP module the runtime can compile or load must
//! be present in the admitted kernel registry for the architectures its name
//! claims, or be explicitly allowlisted with a reason.
//!
//! The gate is pure static analysis over `crates/*/src/**/*.rs` (no GPU, no
//! compiler process, std only):
//!
//! * [`scanner`] lexes Rust (comments, strings, raw strings, chars, lifetimes,
//!   delimiters), drops `#[cfg(test)]`/`#[test]` items and test-only `mod x;`
//!   files, records function spans, and finds every call of a runtime HIP
//!   module sink (`ensure_kernel`, `ensure_kernel_public`,
//!   `compile_and_load_kernel`, `load_kernel_module`,
//!   `module_load_or_recompile`, and `compile`/`compile_for_symbol`/
//!   `recompile`/`compile_batch`/`compile_batch_for_symbols`/`begin_batch` on
//!   a receiver named `compiler`). Native images (`ensure_embedded_kernel`) are
//!   not sinks.
//! * The module argument of each call is resolved by a small value evaluator:
//!   string literals, `const`/`static` items (innermost enclosing function
//!   first, then the file, then a unique workspace match for qualified paths),
//!   `let`/`for`/`if let`/`match`-arm bindings (unambiguous only), tuple and
//!   array/table literals (the module position is projected, so source and
//!   symbol strings are never mistaken for modules), struct literals and
//!   fields, enum struct-variant patterns (`Enum::Hip { module, source } =>`
//!   binds the union of that exact variant's constructor literals across the
//!   workspace, following typed parameters into struct tables, so sibling
//!   native-image variants never leak in), user functions (`*_for_arch`
//!   selectors, `lds256_tile_module`, ...) by their return expressions with
//!   argument substitution, and local
//!   closures. A `format!`, unresolved call, reassigned binding or unsupported
//!   pattern stays *dynamic* and is keyed `crates/<crate>/src/<file>.rs::<fn>`.
//! * A function that forwards its own parameter (or a tuple field of it, e.g.
//!   `(module, src, symbol)`) into a sink is a *wrapper*; its callers are
//!   resolved transitively to a fixpoint.
//!
//! Assertions (module `M` found statically at `file:line`):
//!
//! 1. `M` is in the union of `kernel_registry::entries(arch, "")` over
//!    `SUPPORTED_ARCHES`, unless allowlisted (`module` rows).
//! 2. Arch suffix rule: every `_`-separated token of `M` after the first word
//!    that names an arch is a requirement, wherever it sits (`..._gfx12` or
//!    `..._gfx12_ldsstage`): `gfx1201`/`gfx12` => registered for `gfx1201`;
//!    `gfx1100`, `gfx1151`, `gfx942`, `gfx906` => registered for exactly that
//!    arch; `gfx11` (multi-arch RDNA3) => registered for at least one of
//!    `gfx1100` or `gfx1151`. Several tokens are all required.
//! 3. Every dynamic site key must be allowlisted (`dynamic` rows) and every
//!    allowlist row must still be needed: a `module` row goes stale when the
//!    module is no longer called or when the registry now satisfies it; a
//!    `dynamic` row goes stale when its site vanishes.
//!
//! Fixture `tests/fixtures/lazy-kernel-allowlist.tsv`: header-less or with
//! the header `kind<TAB>name<TAB>category<TAB>reason`, `#` comments and blank
//! lines ignored. Exactly four columns; `kind` is `module` (name = module) or
//! `dynamic` (name = `crates/..../file.rs::fn`); `category` is one of
//! [`CATEGORIES`] (`module` rows may not use the `dynamic-*` categories);
//! `reason` is non-empty and cites a `file:line`; `(kind, name)` is unique.
//!
//! Set `KERNEL_REGISTRY_COVERAGE_DUMP=<path>` (or `-` for stderr) to write every
//! discovered sink site as TSV
//! (`file line fn call mods native dynamic forwards reasons args`).

// FRJ-SCANNER-BEGIN
pub mod scanner {
    #![allow(dead_code)]
    //! Standalone (std-only) static scanner; see the crate-level docs of the
    //! test for the extraction model.

    use std::cell::{Cell, OnceCell, RefCell};
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
    use std::path::Path;

    const NONE: usize = usize::MAX;
    const MAX_DEPTH: usize = 40;
    /// Most same-named user functions a call may union; more is a generic
    /// factory (`new`, `default`, ...), left dynamic.
    const MAX_CALL_CANDS: usize = 6;
    /// Most constructor literals of one struct / variant that are unioned; a
    /// type with more is a data type, not a descriptor table, and stays dynamic.
    const MAX_CTOR_FANOUT: usize = 128;

    // ------------------------------------------------------------------ lexer

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Kind {
        Id,
        Str,
        Chr,
        Num,
        Lt,
        P,
    }

    #[derive(Clone, Copy)]
    struct Tok {
        kind: Kind,
        s: u32,
        e: u32,
        line: u32,
    }

    const P2: [&str; 18] = [
        "::", "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "^=",
        "|=", "&=", "..",
    ];

    fn is_id_start(c: u8) -> bool {
        c.is_ascii_alphabetic() || c == b'_' || c >= 0x80
    }

    fn is_id_cont(c: u8) -> bool {
        c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80
    }

    /// `r"`, `r#"`, `br"`, `cr#"` ... : (index of the opening quote, hashes).
    fn raw_string_open(b: &[u8], i: usize) -> Option<(usize, usize)> {
        let n = b.len();
        let mut j = i;
        if j < n && (b[j] == b'b' || b[j] == b'c') {
            j += 1;
        }
        if j < n && b[j] == b'r' {
            j += 1;
        } else {
            return None;
        }
        let mut h = 0usize;
        while j < n && b[j] == b'#' {
            h += 1;
            j += 1;
        }
        if j < n && b[j] == b'"' {
            Some((j, h))
        } else {
            None
        }
    }

    /// End (exclusive) of the char literal whose opening `'` is at `q`.
    fn char_lit_end(b: &[u8], q: usize) -> Option<usize> {
        let n = b.len();
        if q + 1 >= n {
            return None;
        }
        if b[q + 1] == b'\\' {
            let mut j = q + 3;
            while j < n && b[j] != b'\'' && j < q + 14 {
                j += 1;
            }
            if j < n && b[j] == b'\'' {
                return Some(j + 1);
            }
            return None;
        }
        if b[q + 1] == b'\'' || b[q + 1] == b'\n' {
            return None;
        }
        let c = b[q + 1];
        let len = if c < 0x80 {
            1
        } else if c >= 0xF0 {
            4
        } else if c >= 0xE0 {
            3
        } else {
            2
        };
        if q + 1 + len < n && b[q + 1 + len] == b'\'' {
            Some(q + len + 2)
        } else {
            None
        }
    }

    fn lex(src: &str) -> Vec<Tok> {
        let b = src.as_bytes();
        let n = b.len();
        let mut toks: Vec<Tok> = Vec::new();
        let mut i = 0usize;
        let mut line = 1u32;
        while i < n {
            let c = b[i];
            if c == b'\n' {
                line += 1;
                i += 1;
                continue;
            }
            if c == b' ' || c == b'\t' || c == b'\r' || c == 0x0b || c == 0x0c {
                i += 1;
                continue;
            }
            if c == b'/' && i + 1 < n && b[i + 1] == b'/' {
                while i < n && b[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if c == b'/' && i + 1 < n && b[i + 1] == b'*' {
                let mut depth = 1usize;
                i += 2;
                while i < n && depth > 0 {
                    if b[i] == b'/' && i + 1 < n && b[i + 1] == b'*' {
                        depth += 1;
                        i += 2;
                    } else if b[i] == b'*' && i + 1 < n && b[i + 1] == b'/' {
                        depth -= 1;
                        i += 2;
                    } else {
                        if b[i] == b'\n' {
                            line += 1;
                        }
                        i += 1;
                    }
                }
                continue;
            }
            let start = i;
            let start_line = line;
            // Raw strings: r"..", r#".."#, br"..", cr"..".
            if c == b'r' || c == b'b' || c == b'c' {
                if let Some((q, h)) = raw_string_open(b, i) {
                    let mut j = q + 1;
                    while j < n {
                        if b[j] == b'\n' {
                            line += 1;
                        }
                        if b[j] == b'"' {
                            let mut k = 0usize;
                            while k < h && j + 1 + k < n && b[j + 1 + k] == b'#' {
                                k += 1;
                            }
                            if k == h {
                                j += 1 + h;
                                break;
                            }
                        }
                        j += 1;
                    }
                    let e = j.min(n);
                    toks.push(Tok {
                        kind: Kind::Str,
                        s: start as u32,
                        e: e as u32,
                        line: start_line,
                    });
                    i = e;
                    continue;
                }
            }
            // Ordinary / byte / c strings.
            let str_open = if c == b'"' {
                Some(i)
            } else if (c == b'b' || c == b'c') && i + 1 < n && b[i + 1] == b'"' {
                Some(i + 1)
            } else {
                None
            };
            if let Some(q) = str_open {
                let mut j = q + 1;
                while j < n && b[j] != b'"' {
                    if b[j] == b'\\' && j + 1 < n {
                        if b[j + 1] == b'\n' {
                            line += 1;
                        }
                        j += 2;
                        continue;
                    }
                    if b[j] == b'\n' {
                        line += 1;
                    }
                    j += 1;
                }
                let e = (j + 1).min(n);
                toks.push(Tok {
                    kind: Kind::Str,
                    s: start as u32,
                    e: e as u32,
                    line: start_line,
                });
                i = e;
                continue;
            }
            // Byte char b'x'.
            if c == b'b' && i + 1 < n && b[i + 1] == b'\'' {
                if let Some(e) = char_lit_end(b, i + 1) {
                    toks.push(Tok {
                        kind: Kind::Chr,
                        s: start as u32,
                        e: e as u32,
                        line: start_line,
                    });
                    i = e;
                    continue;
                }
            }
            // Char literal or lifetime.
            if c == b'\'' {
                if let Some(e) = char_lit_end(b, i) {
                    toks.push(Tok {
                        kind: Kind::Chr,
                        s: start as u32,
                        e: e as u32,
                        line: start_line,
                    });
                    i = e;
                } else if i + 1 < n && is_id_start(b[i + 1]) {
                    let mut j = i + 2;
                    while j < n && is_id_cont(b[j]) {
                        j += 1;
                    }
                    toks.push(Tok {
                        kind: Kind::Lt,
                        s: start as u32,
                        e: j as u32,
                        line: start_line,
                    });
                    i = j;
                } else {
                    toks.push(Tok {
                        kind: Kind::P,
                        s: start as u32,
                        e: (i + 1) as u32,
                        line: start_line,
                    });
                    i += 1;
                }
                continue;
            }
            if is_id_start(c) {
                let mut j = i + 1;
                if c == b'r' && i + 2 < n && b[i + 1] == b'#' && is_id_start(b[i + 2]) {
                    j = i + 3;
                }
                while j < n && is_id_cont(b[j]) {
                    j += 1;
                }
                toks.push(Tok {
                    kind: Kind::Id,
                    s: start as u32,
                    e: j as u32,
                    line: start_line,
                });
                i = j;
                continue;
            }
            if c.is_ascii_digit() {
                let mut j = i + 1;
                while j < n && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                    j += 1;
                }
                let prev_dot = match toks.last() {
                    Some(t) => t.kind == Kind::P && &src[t.s as usize..t.e as usize] == ".",
                    None => false,
                };
                if !prev_dot && j + 1 < n && b[j] == b'.' && b[j + 1].is_ascii_digit() {
                    j += 1;
                    while j < n && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                        j += 1;
                    }
                }
                toks.push(Tok {
                    kind: Kind::Num,
                    s: start as u32,
                    e: j as u32,
                    line: start_line,
                });
                i = j;
                continue;
            }
            if i + 1 < n {
                let two = &b[i..i + 2];
                if P2.iter().any(|p| p.as_bytes() == two) {
                    toks.push(Tok {
                        kind: Kind::P,
                        s: start as u32,
                        e: (i + 2) as u32,
                        line: start_line,
                    });
                    i += 2;
                    continue;
                }
            }
            toks.push(Tok {
                kind: Kind::P,
                s: start as u32,
                e: (i + 1) as u32,
                line: start_line,
            });
            i += 1;
        }
        toks
    }

    /// Longest string-literal token whose value is kept; module names and
    /// symbols are short, inline HIP sources and messages are not.
    const STR_KEEP_MAX: usize = 200;

    /// Abstract value of a string literal. Only C identifiers (the only valid
    /// module / symbol names, as `KernelCompiler::valid_identifier`) and the
    /// empty-source sentinel are retained; any other string (inline HIP
    /// source, message, path) becomes an unresolved marker WITHOUT copying its
    /// contents, so a source that is ever fed to a module slot stays dynamic.
    fn str_v(s: String) -> V {
        let keep = s.is_empty() || s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
        if keep && s.len() <= STR_KEEP_MAX {
            let mut v = V::default();
            v.strs.insert(s);
            v
        } else {
            unknown("non-identifier string literal")
        }
    }

    /// Decoded value of a string-literal token's source text.
    fn str_value(raw: &str) -> String {
        let b = raw.as_bytes();
        let mut i = 0usize;
        while i < b.len() && b[i] != b'"' && b[i] != b'r' {
            i += 1;
        }
        if i >= b.len() {
            return String::new();
        }
        if b[i] == b'r' {
            let mut j = i + 1;
            let mut h = 0usize;
            while j < b.len() && b[j] == b'#' {
                h += 1;
                j += 1;
            }
            let end = b.len().saturating_sub(1 + h);
            let lo = (j + 1).min(end);
            return raw.get(lo..end).unwrap_or("").to_string();
        }
        let body = raw.get(i + 1..raw.len().saturating_sub(1)).unwrap_or("");
        let mut out = String::new();
        let mut chars = body.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch != '\\' {
                out.push(ch);
                continue;
            }
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('0') => out.push('\0'),
                Some('\n') => {
                    while matches!(
                        chars.peek(),
                        Some(' ') | Some('\t') | Some('\r') | Some('\n')
                    ) {
                        chars.next();
                    }
                }
                Some(other) => out.push(other),
                None => {}
            }
        }
        out
    }

    fn match_delims(src: &str, toks: &[Tok]) -> Vec<usize> {
        let mut close = vec![NONE; toks.len()];
        let mut stack: Vec<usize> = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            if t.kind != Kind::P {
                continue;
            }
            match &src[t.s as usize..t.e as usize] {
                "(" | "[" | "{" => stack.push(i),
                ")" | "]" | "}" => {
                    if let Some(o) = stack.pop() {
                        close[o] = i;
                    }
                }
                _ => {}
            }
        }
        close
    }

    // ------------------------------------------------------------- file model

    struct FnDef {
        name: String,
        line: u32,
        params: Vec<Option<String>>,
        /// CamelCase type name of each parameter, when it has one.
        param_types: Vec<Option<String>>,
        has_self: bool,
        open: usize,
        close: usize,
        parent: Option<usize>,
    }

    struct ConstDef {
        name: String,
        fn_id: Option<usize>,
        lo: usize,
        hi: usize,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum BKind {
        Let,
        For,
        Match,
    }

    struct Binder {
        pos: usize,
        vlo: usize,
        vhi: usize,
        kind: BKind,
        plo: usize,
        phi: usize,
        ilo: Option<usize>,
        ihi: usize,
        fn_id: Option<usize>,
    }

    struct Binders {
        list: Vec<Binder>,
        by_name: HashMap<String, Vec<usize>>,
    }

    struct File {
        rel: String,
        src: String,
        toks: Vec<Tok>,
        close: Vec<usize>,
        excl_mods: Vec<String>,
        whole: bool,
        fns: Vec<FnDef>,
        consts: Vec<ConstDef>,
        const_by_name: HashMap<String, Vec<usize>>,
        calls: HashMap<String, Vec<usize>>,
        binders: OnceCell<Binders>,
    }

    impl File {
        fn n(&self) -> usize {
            self.toks.len()
        }
        fn t(&self, i: usize) -> &str {
            match self.toks.get(i) {
                Some(t) => &self.src[t.s as usize..t.e as usize],
                None => "",
            }
        }
        fn prev(&self, i: usize) -> &str {
            if i == 0 {
                ""
            } else {
                self.t(i - 1)
            }
        }
        fn k(&self, i: usize) -> Kind {
            self.toks.get(i).map_or(Kind::P, |t| t.kind)
        }
        fn line(&self, i: usize) -> u32 {
            self.toks.get(i).map_or(0, |t| t.line)
        }
        fn cl(&self, i: usize) -> usize {
            self.close.get(i).copied().unwrap_or(NONE)
        }
        fn is_id(&self, i: usize) -> bool {
            self.k(i) == Kind::Id
        }
        fn is_open(&self, i: usize) -> bool {
            matches!(self.t(i), "(" | "[" | "{")
        }
        fn is_closer(&self, i: usize) -> bool {
            matches!(self.t(i), ")" | "]" | "}")
        }
        /// Index after the delimiter group opening at `i` (or `i + 1` if unmatched).
        fn skip(&self, i: usize) -> usize {
            let c = self.cl(i);
            if c == NONE {
                i + 1
            } else {
                c + 1
            }
        }
        fn text(&self, lo: usize, hi: usize) -> String {
            const CAP: usize = 240;
            let mut out = String::new();
            for i in lo..hi.min(self.n()) {
                if out.len() >= CAP {
                    break;
                }
                if !out.is_empty() {
                    out.push(' ');
                }
                let t = self.t(i);
                if t.len() > CAP {
                    out.extend(t.chars().take(CAP / 2));
                    out.push('…');
                } else {
                    out.push_str(t);
                }
            }
            out
        }
    }

    /// Split `[lo, hi)` at depth-0 commas. `types` tracks `<`/`>` nesting
    /// (parameter lists); expression mode skips turbofish `::<..>` groups.
    fn split_commas(f: &File, lo: usize, hi: usize, types: bool) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut start = lo;
        let mut i = lo;
        let mut ang = 0usize;
        while i < hi {
            let x = f.t(i);
            if f.is_open(i) {
                let c = f.cl(i);
                i = if c == NONE { hi } else { c } + 1;
                continue;
            }
            if types {
                if x == "<" {
                    ang += 1;
                } else if x == ">" {
                    ang = ang.saturating_sub(1);
                }
            } else if x == "::" && f.t(i + 1) == "<" {
                let mut d = 0i32;
                i += 1;
                while i < hi {
                    if f.t(i) == "<" {
                        d += 1;
                    } else if f.t(i) == ">" {
                        d -= 1;
                        if d == 0 {
                            break;
                        }
                    }
                    i += 1;
                }
            }
            if x == "," && ang == 0 {
                if i > start {
                    out.push((start, i));
                }
                start = i + 1;
            }
            i += 1;
        }
        if hi > start {
            out.push((start, hi));
        }
        out
    }

    // --------------------------------------------------------- test stripping

    fn cfg_test_only(f: &File, lo: usize, hi: usize) -> bool {
        if lo >= hi {
            return false;
        }
        if hi - lo == 1 && f.t(lo) == "test" {
            return true;
        }
        if (f.t(lo) == "all" || f.t(lo) == "any") && f.t(lo + 1) == "(" {
            let c = f.cl(lo + 1);
            if c == NONE {
                return false;
            }
            let res: Vec<bool> = split_commas(f, lo + 2, c, false)
                .into_iter()
                .map(|(a, b)| cfg_test_only(f, a, b))
                .collect();
            if res.is_empty() {
                return false;
            }
            return if f.t(lo) == "all" {
                res.iter().any(|&x| x)
            } else {
                res.iter().all(|&x| x)
            };
        }
        false
    }

    /// `#[test]`, `#[tokio::test]`, `#[cfg(test)]`, `#[cfg(all(test, ..))]`;
    /// not `cfg(not(test))` and not `cfg(any(test, ..))`.
    fn attr_is_test(f: &File, lo: usize, hi: usize) -> bool {
        if lo >= hi {
            return false;
        }
        if f.t(hi - 1) == "test" && (hi - lo == 1 || f.t(hi - 2) == "::") {
            return true;
        }
        if f.t(lo) == "cfg" && f.t(lo + 1) == "(" {
            let c = f.cl(lo + 1);
            if c == NONE {
                return false;
            }
            return cfg_test_only(f, lo + 2, c);
        }
        false
    }

    /// Exclusive end of the item or statement starting at token `j`.
    fn item_end(f: &File, j: usize) -> usize {
        let n = f.n();
        let mut q = j;
        while q < n {
            let x = f.t(q);
            if matches!(x, "pub" | "unsafe" | "async" | "extern" | "default") {
                q += 1;
                if f.k(q) == Kind::Str {
                    q += 1;
                }
                continue;
            }
            if x == "(" && q > j && f.t(q - 1) == "pub" {
                q = f.skip(q);
                continue;
            }
            break;
        }
        let word = f.t(q);
        let stmt_like = matches!(word, "let" | "static" | "use" | "type")
            || (word == "const" && f.t(q + 1) != "fn");
        let mut i = q;
        while i < n {
            if f.t(i) == "(" || f.t(i) == "[" {
                i = f.skip(i);
                continue;
            }
            if f.t(i) == "{" {
                let e = if f.cl(i) == NONE { n } else { f.cl(i) + 1 };
                if stmt_like {
                    i = e;
                    continue;
                }
                return e;
            }
            if f.t(i) == ";" {
                return i + 1;
            }
            if f.is_closer(i) {
                return i;
            }
            i += 1;
        }
        n
    }

    /// (removed token ranges, test-only `mod x;` names, whole file is test-only)
    fn strip_tests(f: &File) -> (Vec<(usize, usize)>, Vec<String>, bool) {
        let n = f.n();
        let mut removed = Vec::new();
        let mut mods = Vec::new();
        let mut whole = false;
        let mut i = 0usize;
        while i < n {
            if f.t(i) == "#" {
                let mut j = i + 1;
                if f.t(j) == "!" {
                    j += 1;
                }
                if f.t(j) == "[" {
                    let start = i;
                    let mut is_test = false;
                    loop {
                        let a = i;
                        let mut b = a + 1;
                        let mut inner = false;
                        if f.t(b) == "!" {
                            inner = true;
                            b += 1;
                        }
                        if !(f.t(b) == "[" && f.t(a) == "#") {
                            break;
                        }
                        let e = f.cl(b);
                        if e == NONE {
                            break;
                        }
                        if attr_is_test(f, b + 1, e) {
                            if inner {
                                whole = true;
                            } else {
                                is_test = true;
                            }
                        }
                        i = e + 1;
                        if inner {
                            break;
                        }
                    }
                    if i == start {
                        i += 1;
                        continue;
                    }
                    if is_test {
                        let end = item_end(f, i);
                        removed.push((start, end));
                        let mut q = i;
                        while q + 2 < end {
                            if f.t(q) == "mod" && f.is_id(q + 1) && f.t(q + 2) == ";" {
                                mods.push(f.t(q + 1).to_string());
                            }
                            q += 1;
                        }
                        i = i.max(end);
                    }
                    continue;
                }
            }
            i += 1;
        }
        (removed, mods, whole)
    }

    // ------------------------------------------------------------ file build

    fn inner_fn(f: &File, idx: usize) -> Option<usize> {
        let c = f.fns.partition_point(|x| x.open < idx);
        if c == 0 {
            return None;
        }
        let mut j = Some(c - 1);
        while let Some(jj) = j {
            if f.fns[jj].close > idx {
                return Some(jj);
            }
            j = f.fns[jj].parent;
        }
        None
    }

    fn find_fns(f: &File) -> Vec<FnDef> {
        let n = f.n();
        let mut fns: Vec<FnDef> = Vec::new();
        let mut i = 0usize;
        while i + 2 < n {
            if f.t(i) == "fn" && f.is_id(i) && f.is_id(i + 1) {
                let mut j = i + 2;
                if f.t(j) == "<" {
                    let mut d = 0i32;
                    while j < n {
                        if f.t(j) == "<" {
                            d += 1;
                        } else if f.t(j) == ">" {
                            d -= 1;
                            if d == 0 {
                                break;
                            }
                        }
                        j += 1;
                    }
                    j += 1;
                }
                if f.t(j) == "(" && f.cl(j) != NONE {
                    let pc = f.cl(j);
                    let mut params: Vec<Option<String>> = Vec::new();
                    let mut param_types: Vec<Option<String>> = Vec::new();
                    let mut has_self = false;
                    for (mut a, b) in split_commas(f, j + 1, pc, true) {
                        while a < b && f.t(a) == "#" {
                            let c = f.cl(a + 1);
                            if c == NONE {
                                break;
                            }
                            a = c + 1;
                        }
                        if a >= b {
                            continue;
                        }
                        let mut is_self = false;
                        for q in a..b {
                            if f.t(q) == ":" {
                                break;
                            }
                            if f.t(q) == "self" && f.is_id(q) {
                                is_self = true;
                                break;
                            }
                        }
                        if is_self {
                            has_self = true;
                            continue;
                        }
                        let mut q = a;
                        while q < b && (f.t(q) == "mut" || f.t(q) == "ref") {
                            q += 1;
                        }
                        if q + 1 < b && f.is_id(q) && f.t(q + 1) == ":" {
                            params.push(Some(f.t(q).to_string()));
                            // First CamelCase identifier of the type, past `&`/lifetimes/wrappers.
                            let ty = (q + 2..b)
                                .find(|&x| {
                                    f.is_id(x)
                                        && is_camel(f.t(x))
                                        && !matches!(
                                            f.t(x),
                                            "Option"
                                                | "Result"
                                                | "Vec"
                                                | "Box"
                                                | "Arc"
                                                | "Rc"
                                                | "Cow"
                                                | "Self"
                                        )
                                })
                                .map(|x| f.t(x).to_string());
                            param_types.push(ty);
                        } else {
                            params.push(None);
                            param_types.push(None);
                        }
                    }
                    let mut q = pc + 1;
                    let mut body = None;
                    while q < n {
                        if f.t(q) == "(" || f.t(q) == "[" {
                            q = f.skip(q);
                            continue;
                        }
                        if f.t(q) == "{" {
                            body = Some(q);
                            break;
                        }
                        if f.t(q) == ";" || f.is_closer(q) {
                            break;
                        }
                        q += 1;
                    }
                    if let Some(open) = body {
                        let close = f.cl(open);
                        if close != NONE {
                            fns.push(FnDef {
                                name: f.t(i + 1).to_string(),
                                line: f.line(i),
                                params,
                                param_types,
                                has_self,
                                open,
                                close,
                                parent: None,
                            });
                        }
                    }
                }
            }
            i += 1;
        }
        // Enclosing function of each function.
        let mut stack: Vec<usize> = Vec::new();
        for idx in 0..fns.len() {
            while let Some(&top) = stack.last() {
                if fns[top].close < fns[idx].open {
                    stack.pop();
                } else {
                    break;
                }
            }
            fns[idx].parent = stack.last().copied();
            stack.push(idx);
        }
        fns
    }

    fn find_consts(f: &File) -> Vec<ConstDef> {
        let n = f.n();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i + 3 < n {
            if f.is_id(i) && (f.t(i) == "const" || f.t(i) == "static") {
                let mut j = i + 1;
                if f.t(j) == "mut" {
                    j += 1;
                }
                let bad = !f.is_id(j)
                    || matches!(f.t(j), "fn" | "unsafe" | "async" | "extern")
                    || f.t(j + 1) != ":"
                    || (i > 0 && matches!(f.t(i - 1), "<" | ","));
                if !bad {
                    let mut q = j + 2;
                    let mut eq = None;
                    while q < n {
                        if f.is_open(q) {
                            q = f.skip(q);
                            continue;
                        }
                        if f.t(q) == "=" {
                            eq = Some(q);
                            break;
                        }
                        if f.t(q) == ";" || f.is_closer(q) {
                            break;
                        }
                        q += 1;
                    }
                    if let Some(eq) = eq {
                        let mut q = eq + 1;
                        while q < n {
                            if f.is_open(q) {
                                q = f.skip(q);
                                continue;
                            }
                            if f.t(q) == ";" || f.is_closer(q) {
                                break;
                            }
                            q += 1;
                        }
                        out.push(ConstDef {
                            name: f.t(j).to_string(),
                            fn_id: inner_fn(f, i),
                            lo: eq + 1,
                            hi: q,
                        });
                    }
                }
            }
            i += 1;
        }
        out
    }

    fn build_file(rel: &str, src: String) -> File {
        let toks = lex(&src);
        let close = match_delims(&src, &toks);
        let mut f = File {
            rel: rel.to_string(),
            src,
            toks,
            close,
            excl_mods: Vec::new(),
            whole: false,
            fns: Vec::new(),
            consts: Vec::new(),
            const_by_name: HashMap::new(),
            calls: HashMap::new(),
            binders: OnceCell::new(),
        };
        let (removed, mods, whole) = strip_tests(&f);
        f.excl_mods = mods;
        f.whole = whole;
        if whole {
            f.toks = Vec::new();
            f.close = Vec::new();
            return f;
        }
        if !removed.is_empty() {
            let mut kept: Vec<Tok> = Vec::with_capacity(f.toks.len());
            let mut last = 0usize;
            for &(lo, hi) in &removed {
                if lo > last {
                    kept.extend_from_slice(&f.toks[last..lo]);
                }
                last = last.max(hi);
            }
            if last < f.toks.len() {
                kept.extend_from_slice(&f.toks[last..]);
            }
            f.close = match_delims(&f.src, &kept);
            f.toks = kept;
        }
        f.fns = find_fns(&f);
        f.consts = find_consts(&f);
        let mut by_name: HashMap<String, Vec<usize>> = HashMap::new();
        for (ci, c) in f.consts.iter().enumerate() {
            by_name.entry(c.name.clone()).or_default().push(ci);
        }
        f.const_by_name = by_name;
        let mut calls: HashMap<String, Vec<usize>> = HashMap::new();
        for i in 0..f.n() {
            if f.is_id(i) && f.t(i + 1) == "(" && f.prev(i) != "fn" {
                calls.entry(f.t(i).to_string()).or_default().push(i);
            }
        }
        f.calls = calls;
        f
    }

    // --------------------------------------------------------------- binders

    struct Arm {
        plo: usize,
        phi: usize,
        blo: usize,
        bhi: usize,
    }

    fn parse_arms(f: &File, open: usize) -> Vec<Arm> {
        let mut out = Vec::new();
        let hi = f.cl(open);
        if hi == NONE {
            return out;
        }
        let mut pos = open + 1;
        while pos < hi {
            if f.t(pos) == "," {
                pos += 1;
                continue;
            }
            let mut plo = pos;
            let mut arrow = None;
            let mut guard = None;
            let mut q = pos;
            while q < hi {
                if f.is_open(q) {
                    q = f.skip(q);
                    continue;
                }
                if f.t(q) == "if" && guard.is_none() {
                    guard = Some(q);
                }
                if f.t(q) == "=>" {
                    arrow = Some(q);
                    break;
                }
                q += 1;
            }
            let arrow = match arrow {
                Some(a) => a,
                None => break,
            };
            let phi = guard.unwrap_or(arrow);
            while plo < phi && f.t(plo) == "|" {
                plo += 1;
            }
            let blo = arrow + 1;
            let bhi;
            if blo < hi && f.t(blo) == "{" && f.cl(blo) != NONE {
                let mut e = f.cl(blo) + 1;
                if e < hi && matches!(f.t(e), "." | "?") {
                    let mut q = blo;
                    while q < hi {
                        if f.is_open(q) {
                            q = f.skip(q);
                            continue;
                        }
                        if f.t(q) == "," {
                            break;
                        }
                        q += 1;
                    }
                    e = q;
                }
                bhi = e;
            } else {
                let mut q = blo;
                while q < hi {
                    if f.is_open(q) {
                        q = f.skip(q);
                        continue;
                    }
                    if f.t(q) == "," {
                        break;
                    }
                    q += 1;
                }
                bhi = q;
            }
            out.push(Arm { plo, phi, blo, bhi });
            pos = bhi.max(pos + 1);
        }
        out
    }

    fn build_binders(f: &File) -> Binders {
        let n = f.n();
        let mut list: Vec<Binder> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        for i in 0..n {
            if f.is_open(i) {
                stack.push(i);
                continue;
            }
            if f.is_closer(i) {
                stack.pop();
                continue;
            }
            if !f.is_id(i) {
                continue;
            }
            let enc_hi = match stack.last() {
                Some(&o) if f.cl(o) != NONE => f.cl(o),
                _ => n,
            };
            match f.t(i) {
                "let" => {
                    let is_cond = matches!(f.prev(i), "if" | "while" | "&&" | "||");
                    let plo = i + 1;
                    let mut j = plo;
                    let mut phi: Option<usize> = None;
                    while j < n {
                        if f.is_open(j) {
                            j = f.skip(j);
                            continue;
                        }
                        let x = f.t(j);
                        if x == ":" || x == "=" || x == ";" {
                            if phi.is_none() {
                                phi = Some(j);
                            }
                            if x == ":" {
                                j += 1;
                                continue;
                            }
                            break;
                        }
                        if f.is_closer(j) {
                            break;
                        }
                        j += 1;
                    }
                    let phi = phi.unwrap_or(j);
                    if j < n && f.t(j) == "=" {
                        let ilo = j + 1;
                        let mut q = ilo;
                        if is_cond {
                            while q < n {
                                if f.t(q) == "(" || f.t(q) == "[" {
                                    q = f.skip(q);
                                    continue;
                                }
                                if f.t(q) == "{" || f.is_closer(q) || f.t(q) == ";" {
                                    break;
                                }
                                q += 1;
                            }
                            if q < n && f.t(q) == "{" && f.cl(q) != NONE {
                                list.push(Binder {
                                    pos: i,
                                    vlo: q,
                                    vhi: f.cl(q),
                                    kind: BKind::Let,
                                    plo,
                                    phi,
                                    ilo: Some(ilo),
                                    ihi: q,
                                    fn_id: None,
                                });
                            }
                            continue;
                        }
                        let mut else_at: Option<usize> = None;
                        while q < n {
                            if f.is_open(q) {
                                q = f.skip(q);
                                continue;
                            }
                            if f.t(q) == "else" && f.prev(q) != "}" {
                                if else_at.is_none() {
                                    else_at = Some(q);
                                }
                                q += 1;
                                continue;
                            }
                            if f.t(q) == ";" || f.is_closer(q) {
                                break;
                            }
                            q += 1;
                        }
                        list.push(Binder {
                            pos: i,
                            vlo: q,
                            vhi: enc_hi,
                            kind: BKind::Let,
                            plo,
                            phi,
                            ilo: Some(ilo),
                            ihi: else_at.unwrap_or(q),
                            fn_id: None,
                        });
                    } else {
                        list.push(Binder {
                            pos: i,
                            vlo: j,
                            vhi: enc_hi,
                            kind: BKind::Let,
                            plo,
                            phi,
                            ilo: None,
                            ihi: j,
                            fn_id: None,
                        });
                    }
                }
                "for" => {
                    if f.t(i + 1) == "<" {
                        continue;
                    }
                    let mut j = i + 1;
                    let mut inpos = None;
                    while j < n {
                        if f.t(j) == "(" || f.t(j) == "[" {
                            j = f.skip(j);
                            continue;
                        }
                        if f.t(j) == "in" {
                            inpos = Some(j);
                            break;
                        }
                        if f.t(j) == "{" || f.t(j) == ";" || f.is_closer(j) {
                            break;
                        }
                        j += 1;
                    }
                    let inpos = match inpos {
                        Some(p) => p,
                        None => continue,
                    };
                    let mut q = inpos + 1;
                    while q < n {
                        if f.t(q) == "(" || f.t(q) == "[" {
                            q = f.skip(q);
                            continue;
                        }
                        if f.t(q) == "{" || f.is_closer(q) || f.t(q) == ";" {
                            break;
                        }
                        q += 1;
                    }
                    if q < n && f.t(q) == "{" && f.cl(q) != NONE {
                        list.push(Binder {
                            pos: i,
                            vlo: q,
                            vhi: f.cl(q),
                            kind: BKind::For,
                            plo: i + 1,
                            phi: inpos,
                            ilo: Some(inpos + 1),
                            ihi: q,
                            fn_id: None,
                        });
                    }
                }
                "match" => {
                    if f.prev(i) == "." {
                        continue;
                    }
                    let mut q = i + 1;
                    while q < n {
                        if f.t(q) == "(" || f.t(q) == "[" {
                            q = f.skip(q);
                            continue;
                        }
                        if f.t(q) == "{" || f.is_closer(q) || f.t(q) == ";" {
                            break;
                        }
                        q += 1;
                    }
                    if q < n && f.t(q) == "{" && f.cl(q) != NONE {
                        for arm in parse_arms(f, q) {
                            list.push(Binder {
                                pos: arm.blo,
                                vlo: arm.blo,
                                vhi: arm.bhi,
                                kind: BKind::Match,
                                plo: arm.plo,
                                phi: arm.phi,
                                ilo: Some(i + 1),
                                ihi: q,
                                fn_id: None,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        let mut by_name: HashMap<String, Vec<usize>> = HashMap::new();
        for (bi, b) in list.iter_mut().enumerate() {
            let mut seen: HashSet<&str> = HashSet::new();
            for q in b.plo..b.phi.min(n) {
                if f.is_id(q) {
                    let w = f.t(q);
                    if !matches!(w, "mut" | "ref" | "box") && seen.insert(w) {
                        by_name.entry(w.to_string()).or_default().push(bi);
                    }
                }
            }
            b.fn_id = inner_fn(f, b.pos);
        }
        Binders { list, by_name }
    }

    // -------------------------------------------------------------- patterns

    enum Pat {
        Id(String),
        Wild,
        Tup(Vec<Pat>),
        Tr(Box<Pat>),
        /// `Enum::Variant { field: pat, .. }` struct-variant pattern.
        Variant(String, String, Vec<(String, Pat)>),
        Other(BTreeSet<String>),
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    enum Step {
        Idx(usize),
        /// Field of an exact enum struct variant: (enum, variant, field).
        Var(String, String, String),
    }

    enum BindRes {
        NotBound,
        Path(Vec<Step>),
        Bad,
    }

    fn idents_in(f: &File, lo: usize, hi: usize) -> BTreeSet<String> {
        let mut s = BTreeSet::new();
        for i in lo..hi.min(f.n()) {
            if f.is_id(i) {
                s.insert(f.t(i).to_string());
            }
        }
        s
    }

    fn parse_pat(f: &File, mut lo: usize, hi: usize) -> Pat {
        while lo < hi && matches!(f.t(lo), "&" | "&&" | "mut" | "ref" | "box") {
            lo += 1;
        }
        if lo >= hi {
            return Pat::Other(BTreeSet::new());
        }
        if hi - lo == 1 && f.is_id(lo) {
            let w = f.t(lo);
            if w == "_" {
                return Pat::Wild;
            }
            let c = w.chars().next().unwrap_or('A');
            if c.is_lowercase() || c == '_' {
                return Pat::Id(w.to_string());
            }
        }
        if f.t(lo) == "(" && f.cl(lo) == hi - 1 {
            let elems = split_commas(f, lo + 1, hi - 1, false);
            if elems.len() == 1 && !(hi >= 2 && f.t(hi - 2) == ",") {
                return parse_pat(f, elems[0].0, elems[0].1);
            }
            let mut out = Vec::new();
            let mut rest = false;
            for (a, b) in elems {
                if b - a == 1 && f.t(a) == ".." {
                    rest = true;
                    continue;
                }
                if rest {
                    out.push(Pat::Other(idents_in(f, a, b)));
                } else {
                    out.push(parse_pat(f, a, b));
                }
            }
            return Pat::Tup(out);
        }
        // `Enum::Variant { field, other: pat, .. }`
        if f.is_id(lo) {
            let mut j = lo;
            let mut segs: Vec<&str> = Vec::new();
            while j < hi && f.is_id(j) {
                segs.push(f.t(j));
                j += 1;
                if j < hi && f.t(j) == "::" {
                    j += 1;
                    continue;
                }
                break;
            }
            if segs.len() >= 2
                && j < hi
                && f.t(j) == "{"
                && f.cl(j) == hi - 1
                && is_camel(segs[segs.len() - 1])
                && is_camel(segs[segs.len() - 2])
            {
                let mut fields: Vec<(String, Pat)> = Vec::new();
                for (a, b) in split_commas(f, j + 1, hi - 1, false) {
                    if f.t(a) == ".." {
                        continue;
                    }
                    let mut a = a;
                    while a < b && matches!(f.t(a), "ref" | "mut") {
                        a += 1;
                    }
                    if a + 2 < b && f.is_id(a) && f.t(a + 1) == ":" {
                        fields.push((f.t(a).to_string(), parse_pat(f, a + 2, b)));
                    } else if b - a == 1 && f.is_id(a) {
                        fields.push((f.t(a).to_string(), Pat::Id(f.t(a).to_string())));
                    }
                }
                return Pat::Variant(
                    segs[segs.len() - 2].to_string(),
                    segs[segs.len() - 1].to_string(),
                    fields,
                );
            }
        }
        if matches!(f.t(lo), "Some" | "Ok") && f.t(lo + 1) == "(" && f.cl(lo + 1) == hi - 1 {
            return Pat::Tr(Box::new(parse_pat(f, lo + 2, hi - 1)));
        }
        Pat::Other(idents_in(f, lo, hi))
    }

    fn bind_path(pat: &Pat, name: &str) -> BindRes {
        match pat {
            Pat::Id(n) => {
                if n == name {
                    BindRes::Path(Vec::new())
                } else {
                    BindRes::NotBound
                }
            }
            Pat::Wild => BindRes::NotBound,
            Pat::Tr(p) => bind_path(p, name),
            Pat::Other(s) => {
                if s.contains(name) {
                    BindRes::Bad
                } else {
                    BindRes::NotBound
                }
            }
            Pat::Variant(en, var, fields) => {
                for (fname, sub) in fields {
                    match bind_path(sub, name) {
                        BindRes::NotBound => continue,
                        BindRes::Bad => return BindRes::Bad,
                        BindRes::Path(mut p) => {
                            p.insert(0, Step::Var(en.clone(), var.clone(), fname.clone()));
                            return BindRes::Path(p);
                        }
                    }
                }
                BindRes::NotBound
            }
            Pat::Tup(items) => {
                for (i, sub) in items.iter().enumerate() {
                    match bind_path(sub, name) {
                        BindRes::NotBound => continue,
                        BindRes::Bad => return BindRes::Bad,
                        BindRes::Path(mut p) => {
                            p.insert(0, Step::Idx(i));
                            return BindRes::Path(p);
                        }
                    }
                }
                BindRes::NotBound
            }
        }
    }

    // ---------------------------------------------------------------- values

    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    enum Proj {
        Idx(usize),
        Field(String),
    }

    const TAG_OK: u8 = 1;
    const TAG_ERR: u8 = 2;
    const TAG_NONE: u8 = 4;

    /// Abstract value of an expression: HIP strings, native-image strings, a
    /// tuple (elementwise union over branches) or struct, references to the
    /// enclosing function's parameters (with a projection path) and the reasons
    /// it is not fully known.
    #[derive(Clone, Default)]
    struct V {
        strs: BTreeSet<String>,
        nstrs: BTreeSet<String>,
        tag: u8,
        tup: Option<Vec<V>>,
        fields: Option<BTreeMap<String, V>>,
        params: BTreeSet<(usize, Vec<Proj>)>,
        unk: Vec<String>,
    }

    fn unknown(reason: impl Into<String>) -> V {
        V {
            unk: vec![reason.into()],
            ..V::default()
        }
    }

    fn union(vs: Vec<V>) -> V {
        if vs.len() <= 1 {
            return vs.into_iter().next().unwrap_or_default();
        }
        let mut out = V::default();
        for v in vs {
            out.strs.extend(v.strs);
            out.nstrs.extend(v.nstrs);
            out.tag |= v.tag;
            out.params.extend(v.params);
            for u in v.unk {
                if !out.unk.contains(&u) {
                    out.unk.push(u);
                }
            }
            if let Some(t) = v.tup {
                match out.tup.take() {
                    None => out.tup = Some(t),
                    Some(cur) => {
                        if cur.len() == t.len() {
                            out.tup = Some(
                                cur.into_iter()
                                    .zip(t)
                                    .map(|(a, b)| union(vec![a, b]))
                                    .collect(),
                            );
                        } else {
                            out.tup = Some(cur);
                            let msg = "tuple arity mismatch".to_string();
                            if !out.unk.contains(&msg) {
                                out.unk.push(msg);
                            }
                        }
                    }
                }
            }
            if let Some(fs) = v.fields {
                let cur = out.fields.get_or_insert_with(BTreeMap::new);
                for (k, vv) in fs {
                    let merged = match cur.remove(&k) {
                        Some(old) => union(vec![old, vv]),
                        None => vv,
                    };
                    cur.insert(k, merged);
                }
            }
        }
        out
    }

    fn project(v: &V, key: &Proj) -> V {
        let mut parts: Vec<V> = Vec::new();
        match key {
            Proj::Idx(i) => {
                if let Some(t) = &v.tup {
                    parts.push(
                        t.get(*i)
                            .cloned()
                            .unwrap_or_else(|| unknown("tuple index out of range")),
                    );
                } else if !v.strs.is_empty() || v.fields.is_some() {
                    parts.push(unknown("projection of non-tuple"));
                }
            }
            Proj::Field(name) => {
                if let Some(fs) = &v.fields {
                    parts.push(
                        fs.get(name)
                            .cloned()
                            .unwrap_or_else(|| unknown(format!("no field {name}"))),
                    );
                } else if !v.strs.is_empty() || v.tup.is_some() {
                    parts.push(unknown(format!("field of non-struct {name}")));
                }
            }
        }
        let mut extra = V::default();
        for (p, path) in &v.params {
            let mut np = path.clone();
            np.push(key.clone());
            extra.params.insert((*p, np));
        }
        extra.unk = v.unk.clone();
        extra.nstrs = v.nstrs.clone();
        parts.push(extra);
        union(parts)
    }

    fn proj_path(v: &V, path: &[Proj]) -> V {
        let mut cur = v.clone();
        for key in path {
            cur = project(&cur, key);
        }
        cur
    }

    fn no_params(v: &V) -> V {
        V {
            params: BTreeSet::new(),
            ..v.clone()
        }
    }

    /// A tuple is a native-image record, whose other elements' strings (its
    /// module, symbol) are native and never HIP modules, when
    ///
    /// * it holds an `Err(..)` element (`(module, Err(image), kernel)`), or
    /// * it is a `(module, source, name, ..)` descriptor (arity >= 3) whose
    ///   source slot (index 1) is exactly the `None` literal or exactly the
    ///   empty string `""`: a HIP source can be neither absent nor empty, and
    ///   the consumer routes that arm to `ensure_embedded_kernel`.
    ///
    /// A source slot that is only *sometimes* absent (`if c { Some(S) } else
    /// { None }`) is not a sentinel, so the module stays a HIP candidate.
    fn mk_tuple(elems: Vec<V>) -> V {
        let has_err = elems.iter().any(|e| e.tag & TAG_ERR != 0);
        let no_source = elems.len() >= 3 && {
            let s = &elems[1];
            let none = s.tag & TAG_NONE != 0
                && s.strs.is_empty()
                && s.tup.is_none()
                && s.fields.is_none()
                && s.params.is_empty()
                && s.unk.is_empty();
            let empty =
                s.strs.len() == 1 && s.strs.contains("") && s.params.is_empty() && s.unk.is_empty();
            none || empty
        };
        let elems = if has_err || no_source {
            elems
                .into_iter()
                .map(|e| {
                    if e.tag & TAG_ERR != 0 {
                        e
                    } else {
                        let mut ns = e.nstrs.clone();
                        ns.extend(e.strs.iter().cloned());
                        V {
                            strs: BTreeSet::new(),
                            nstrs: ns,
                            ..e
                        }
                    }
                })
                .collect()
        } else {
            elems
        };
        V {
            tup: Some(elems),
            ..V::default()
        }
    }

    // -------------------------------------------------------------- analyzer

    const ADAPT: &[&str] = &[
        "as_str",
        "to_string",
        "to_owned",
        "as_ref",
        "clone",
        "into",
        "as_deref",
        "unwrap",
        "expect",
        "copied",
        "cloned",
        "iter",
        "into_iter",
        "to_str",
        "borrow",
        "as_slice",
        "unwrap_or_default",
        "to_vec",
        "collect",
    ];
    const ADAPT_ARGS: &[&str] = &[
        "skip",
        "take",
        "step_by",
        "rev",
        "filter",
        "ok_or",
        "ok_or_else",
        "map_err",
    ];
    const DIVERGE_MACROS: &[&str] = &["panic", "unreachable", "todo", "unimplemented"];

    struct Env {
        fi: usize,
        lo: usize,
        hi: usize,
        vals: HashMap<String, V>,
    }

    type Arg = (usize, usize, usize);

    /// Constructor literals by exact variant / struct name:
    /// `variants[variant] = [(enum, file, open brace)]`, `structs[name] = [(file, open brace)]`.
    #[derive(Default)]
    struct CtorIndex {
        variants: HashMap<String, Vec<(String, usize, usize)>>,
        structs: HashMap<String, Vec<(usize, usize)>>,
    }

    fn is_camel(w: &str) -> bool {
        let mut cs = w.chars();
        cs.next().map_or(false, char::is_uppercase) && w.chars().any(char::is_lowercase)
    }

    /// Whether a group closing at `close` is in pattern position (`=>`, `|`,
    /// `=` or a guard `if` follows).
    fn pattern_follows(f: &File, close: usize) -> bool {
        matches!(f.t(close + 1), "=>" | "|" | "=" | "if")
    }

    fn build_ctor_index(files: &[File]) -> CtorIndex {
        const NOT_CTOR_PREV: &[&str] = &[
            "struct", "enum", "impl", "for", "->", ">", "union", "trait", "==", "!=", "if",
            "match", "while", "in", "mod", "type", "else", "as",
        ];
        let mut idx = CtorIndex::default();
        for (fi, f) in files.iter().enumerate() {
            let n = f.n();
            let mut stack: Vec<usize> = Vec::new();
            // (closing brace, implemented type) of the enclosing `impl` blocks.
            let mut impls: Vec<(usize, String)> = Vec::new();
            for i in 0..n {
                while impls.last().map_or(false, |&(c, _)| c < i) {
                    impls.pop();
                }
                if f.is_closer(i) {
                    stack.pop();
                    continue;
                }
                if !f.is_open(i) {
                    continue;
                }
                if f.t(i) == "{"
                    && i >= 1
                    && f.cl(i) != NONE
                    && (f.is_id(i - 1) || f.t(i - 1) == ">")
                {
                    if let Some(ty) = impl_target(f, i) {
                        impls.push((f.cl(i), ty));
                    }
                }
                let candidate = f.t(i) == "{"
                    && i >= 1
                    && f.is_id(i - 1)
                    && f.cl(i) != NONE
                    && is_camel(f.t(i - 1));
                if candidate {
                    let name = f.t(i - 1);
                    let close = f.cl(i);
                    let entries = split_commas(f, i + 1, close, false);
                    let body_ok = !entries.is_empty()
                        && entries.iter().all(|&(a, b)| {
                            f.t(a) == ".." && b - a > 1
                                || (b - a >= 3 && f.is_id(a) && f.t(a + 1) == ":")
                                || (b - a == 1 && f.is_id(a))
                        })
                        && !(entries.len() == 1
                            && entries[0].1 - entries[0].0 == 1
                            && f.t(entries[0].0) == "..");
                    let in_pattern = pattern_follows(f, close)
                        || stack.iter().any(|&o| {
                            matches!(f.t(o), "(" | "[")
                                && f.cl(o) != NONE
                                && pattern_follows(f, f.cl(o))
                        });
                    if body_ok && !in_pattern {
                        // `Self` is the enclosing impl's type; outside an impl it is unknown.
                        let resolve = |w: &str| -> Option<String> {
                            if w == "Self" {
                                impls.last().map(|(_, t)| t.clone())
                            } else {
                                Some(w.to_string())
                            }
                        };
                        let enum_path = i >= 3 && f.t(i - 2) == "::" && f.is_id(i - 3);
                        if enum_path && is_camel(f.t(i - 3)) && is_camel(name) {
                            if let Some(en) = resolve(f.t(i - 3)) {
                                idx.variants
                                    .entry(name.to_string())
                                    .or_default()
                                    .push((en, fi, i));
                            }
                        } else if is_camel(name)
                            && (enum_path || i < 2 || !NOT_CTOR_PREV.contains(&f.t(i - 2)))
                        {
                            if let Some(ty) = resolve(name) {
                                idx.structs.entry(ty).or_default().push((fi, i));
                            }
                        }
                    }
                }
                stack.push(i);
            }
        }
        idx
    }

    /// Type implemented by the `impl` block whose `{` is at `open`, if it is one.
    fn impl_target(f: &File, open: usize) -> Option<String> {
        let lo = open.saturating_sub(48);
        let mut j = open;
        let mut at = None;
        while j > lo {
            j -= 1;
            match f.t(j) {
                "impl" => {
                    at = Some(j);
                    break;
                }
                ";" | "{" | "}" => return None,
                _ => {}
            }
        }
        let at = at?;
        let mut depth = 0i32;
        let mut ty: Option<String> = None;
        for k in at + 1..open {
            match f.t(k) {
                "<" => depth += 1,
                ">" => depth -= 1,
                "where" if depth == 0 => break,
                w if depth == 0 && f.is_id(k) && is_camel(w) => ty = Some(w.to_string()),
                _ => {}
            }
        }
        ty
    }

    struct An<'a> {
        files: &'a [File],
        fns_by_name: HashMap<String, Vec<(usize, usize)>>,
        consts_by_name: HashMap<String, Vec<(usize, usize)>>,
        ret_cache: RefCell<HashMap<(usize, usize), V>>,
        ret_stack: RefCell<HashSet<(usize, usize)>>,
        const_cache: RefCell<HashMap<(usize, usize), V>>,
        const_stack: RefCell<HashSet<(usize, usize)>>,
        depth: Cell<usize>,
        env: RefCell<Vec<Env>>,
        ctors: OnceCell<CtorIndex>,
        variant_cache: RefCell<HashMap<(String, String, String), V>>,
        variant_stack: RefCell<HashSet<(String, String, String)>>,
    }

    fn open_of(f: &File, closing: usize, lo: usize) -> Option<usize> {
        let mut depth = 0i32;
        let mut j = closing;
        loop {
            if f.is_closer(j) {
                depth += 1;
            } else if f.is_open(j) {
                depth -= 1;
                if depth == 0 {
                    return Some(j);
                }
            }
            if j == lo {
                return None;
            }
            j -= 1;
        }
    }

    impl<'a> An<'a> {
        fn new(files: &'a [File]) -> An<'a> {
            let mut fns_by_name: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
            let mut consts_by_name: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
            for (fi, f) in files.iter().enumerate() {
                for (i, d) in f.fns.iter().enumerate() {
                    fns_by_name.entry(d.name.clone()).or_default().push((fi, i));
                }
                for (i, c) in f.consts.iter().enumerate() {
                    consts_by_name
                        .entry(c.name.clone())
                        .or_default()
                        .push((fi, i));
                }
            }
            An {
                files,
                fns_by_name,
                consts_by_name,
                ret_cache: RefCell::new(HashMap::new()),
                ret_stack: RefCell::new(HashSet::new()),
                const_cache: RefCell::new(HashMap::new()),
                const_stack: RefCell::new(HashSet::new()),
                depth: Cell::new(0),
                env: RefCell::new(Vec::new()),
                ctors: OnceCell::new(),
                variant_cache: RefCell::new(HashMap::new()),
                variant_stack: RefCell::new(HashSet::new()),
            }
        }

        fn ev(&self, fi: usize, lo: usize, hi: usize) -> V {
            let d = self.depth.get();
            if d > MAX_DEPTH {
                return unknown("depth");
            }
            self.depth.set(d + 1);
            let r = self.ev_inner(fi, lo, hi);
            self.depth.set(d);
            r
        }

        fn ev_inner(&self, fi: usize, mut lo: usize, hi: usize) -> V {
            let f = &self.files[fi];
            while lo < hi && matches!(f.t(lo), "&" | "&&" | "*") {
                lo += 1;
            }
            while lo < hi && f.t(lo) == "mut" {
                lo += 1;
            }
            if lo >= hi {
                return V::default();
            }
            let first = f.t(lo);
            if matches!(first, "return" | "break" | "continue") {
                return V::default();
            }
            if DIVERGE_MACROS.contains(&first) && f.t(lo + 1) == "!" {
                return V::default();
            }
            // Parenthesised expression or tuple.
            if first == "(" && f.cl(lo) == hi - 1 {
                let elems = split_commas(f, lo + 1, hi - 1, false);
                let trailing = hi >= 2 && hi - 2 > lo && f.t(hi - 2) == ",";
                if elems.is_empty() {
                    return V::default();
                }
                if elems.len() == 1 && !trailing {
                    return self.ev(fi, elems[0].0, elems[0].1);
                }
                let vals: Vec<V> = elems.iter().map(|&(a, b)| self.ev(fi, a, b)).collect();
                return mk_tuple(vals);
            }
            // Postfix peeling from the right.
            let last = f.t(hi - 1);
            if last == "?" {
                return self.ev(fi, lo, hi - 1);
            }
            if last == ")" {
                if let Some(o) = open_of(f, hi - 1, lo) {
                    if o > lo + 1 && f.is_id(o - 1) && f.t(o - 2) == "." {
                        let m = f.t(o - 1);
                        if ADAPT.contains(&m) || ADAPT_ARGS.contains(&m) {
                            return self.ev(fi, lo, o - 2);
                        }
                        let args: Vec<Arg> = split_commas(f, o + 1, hi - 1, false)
                            .into_iter()
                            .map(|(a, b)| (fi, a, b))
                            .collect();
                        return self.call(fi, m, &args, true);
                    }
                }
            }
            if last == "]" {
                if let Some(o) = open_of(f, hi - 1, lo) {
                    if o > lo {
                        let p = f.t(o - 1);
                        let operator = f.k(o - 1) == Kind::P
                            && matches!(
                                p,
                                "=" | "," | "(" | "[" | "{" | "&" | "!" | "=>" | ";" | ":"
                            );
                        if !operator && p != "vec" {
                            return self.ev(fi, lo, o);
                        }
                    }
                }
            }
            if (f.k(hi - 1) == Kind::Id || f.k(hi - 1) == Kind::Num)
                && hi >= lo + 2
                && f.t(hi - 2) == "."
            {
                let base = self.ev(fi, lo, hi - 2);
                let txt = f.t(hi - 1);
                let key = match txt.parse::<usize>() {
                    Ok(i) if f.k(hi - 1) == Kind::Num => Proj::Idx(i),
                    _ => Proj::Field(txt.to_string()),
                };
                return project(&base, &key);
            }
            // Primary expressions.
            if f.k(lo) == Kind::Str && hi - lo == 1 {
                if first.len() > STR_KEEP_MAX {
                    return unknown("non-identifier string literal");
                }
                return str_v(str_value(first));
            }
            if f.is_id(lo) {
                if first == "if" {
                    return self.ev_if(fi, lo, hi);
                }
                if first == "match" {
                    let mut q = lo + 1;
                    while q < hi {
                        if f.t(q) == "(" || f.t(q) == "[" {
                            q = f.skip(q);
                            continue;
                        }
                        if f.t(q) == "{" {
                            break;
                        }
                        q += 1;
                    }
                    if q < hi && f.cl(q) == hi - 1 {
                        let vals: Vec<V> = parse_arms(f, q)
                            .iter()
                            .map(|arm| self.ev_arm(fi, arm.blo, arm.bhi))
                            .collect();
                        return union(vals);
                    }
                    return unknown("match");
                }
                if first == "unsafe" && f.t(lo + 1) == "{" {
                    return self.ev_block(fi, lo + 1);
                }
                if first == "move" {
                    return unknown("closure");
                }
            }
            if first == "{" && f.cl(lo) == hi - 1 {
                return self.ev_block(fi, lo);
            }
            if first == "[" && f.cl(lo) == hi - 1 {
                return self.ev_array(fi, lo + 1, hi - 1);
            }
            if first == "|" || first == "||" {
                return unknown("closure");
            }
            if f.is_id(lo)
                && f.t(lo + 1) == "!"
                && lo + 2 < hi
                && f.is_open(lo + 2)
                && f.cl(lo + 2) == hi - 1
            {
                let name = first;
                if name == "vec" {
                    return self.ev_array(fi, lo + 3, hi - 1);
                }
                if name == "concat" {
                    let mut s = String::new();
                    for (a, b) in split_commas(f, lo + 3, hi - 1, false) {
                        if b - a == 1 && f.k(a) == Kind::Str && f.t(a).len() <= STR_KEEP_MAX {
                            s.push_str(&str_value(f.t(a)));
                        } else {
                            return unknown("concat!");
                        }
                    }
                    return str_v(s);
                }
                return unknown(format!("{name}!"));
            }
            if f.k(lo) == Kind::Lt {
                return self.ev(fi, lo + 1, hi);
            }
            if !f.is_id(lo) {
                return unknown(format!("expr:{}", f.text(lo, hi.min(lo + 12))));
            }
            // Path expression.
            let mut j = lo;
            let mut segs: Vec<&str> = Vec::new();
            while j < hi && f.is_id(j) {
                segs.push(f.t(j));
                j += 1;
                if j < hi && f.t(j) == "::" {
                    j += 1;
                    if j < hi && f.t(j) == "<" {
                        let mut d = 0i32;
                        while j < hi {
                            if f.t(j) == "<" {
                                d += 1;
                            } else if f.t(j) == ">" {
                                d -= 1;
                                if d == 0 {
                                    j += 1;
                                    break;
                                }
                            }
                            j += 1;
                        }
                        if j < hi && f.t(j) == "::" {
                            j += 1;
                        }
                    }
                    continue;
                }
                break;
            }
            if segs.is_empty() {
                return unknown(format!("expr:{}", f.text(lo, hi.min(lo + 12))));
            }
            let name = segs[segs.len() - 1];
            let c0 = name.chars().next().unwrap_or('A');
            if j == hi {
                if segs.len() == 1 && (c0.is_lowercase() || c0 == '_') && name != "self" {
                    return self.ev_local(fi, lo, name);
                }
                if name == "None" {
                    return V {
                        tag: TAG_NONE,
                        ..V::default()
                    };
                }
                if name == "self" {
                    return unknown("self");
                }
                if c0.is_uppercase() {
                    return self.ev_const(fi, lo, name, segs.len() > 1);
                }
                return unknown(format!("path {}", segs.join("::")));
            }
            if f.t(j) == "(" && f.cl(j) == hi - 1 {
                let args: Vec<Arg> = split_commas(f, j + 1, hi - 1, false)
                    .into_iter()
                    .map(|(a, b)| (fi, a, b))
                    .collect();
                if name == "Some" && args.len() == 1 {
                    return self.ev(args[0].0, args[0].1, args[0].2);
                }
                if name == "Ok" && args.len() == 1 {
                    let v = self.ev(args[0].0, args[0].1, args[0].2);
                    return V {
                        tag: v.tag | TAG_OK,
                        ..v
                    };
                }
                if name == "Err" && args.len() == 1 {
                    let v = self.ev(args[0].0, args[0].1, args[0].2);
                    return V {
                        tag: TAG_ERR,
                        unk: Vec::new(),
                        ..v
                    };
                }
                if segs.len() == 1 {
                    if let Some(bi) = self.closure_of(fi, lo, name) {
                        return self.call_closure(fi, bi, &args);
                    }
                }
                return self.call(fi, name, &args, false);
            }
            if f.t(j) == "{" && f.cl(j) == hi - 1 && c0.is_uppercase() {
                let mut fields: BTreeMap<String, V> = BTreeMap::new();
                for (a, b) in split_commas(f, j + 1, hi - 1, false) {
                    if f.t(a) == ".." {
                        continue;
                    }
                    if b - a >= 3 && f.is_id(a) && f.t(a + 1) == ":" {
                        fields.insert(f.t(a).to_string(), self.ev(fi, a + 2, b));
                    } else if b - a == 1 && f.is_id(a) {
                        fields.insert(f.t(a).to_string(), self.ev_local(fi, a, f.t(a)));
                    }
                }
                return V {
                    fields: Some(fields),
                    ..V::default()
                };
            }
            unknown(format!("expr:{}", f.text(lo, hi.min(lo + 12))))
        }

        fn ev_arm(&self, fi: usize, blo: usize, bhi: usize) -> V {
            let f = &self.files[fi];
            if bhi >= blo + 2 && f.t(blo) == "{" && f.cl(blo) == bhi - 1 {
                return self.ev_block(fi, blo);
            }
            self.ev(fi, blo, bhi)
        }

        fn ev_array(&self, fi: usize, lo: usize, hi: usize) -> V {
            let f = &self.files[fi];
            let mut vals = Vec::new();
            for (a, b) in split_commas(f, lo, hi, false) {
                let mut q = a;
                let mut semi = None;
                while q < b {
                    if f.is_open(q) {
                        q = f.skip(q);
                        continue;
                    }
                    if f.t(q) == ";" {
                        semi = Some(q);
                        break;
                    }
                    q += 1;
                }
                vals.push(self.ev(fi, a, semi.unwrap_or(b)));
            }
            union(vals)
        }

        fn ev_if(&self, fi: usize, lo: usize, hi: usize) -> V {
            let f = &self.files[fi];
            let mut vals = Vec::new();
            let mut q = lo;
            while q < hi {
                if f.t(q) != "if" {
                    return unknown("if");
                }
                q += 1;
                while q < hi {
                    if f.t(q) == "(" || f.t(q) == "[" {
                        q = f.skip(q);
                        continue;
                    }
                    if f.t(q) == "{" {
                        break;
                    }
                    q += 1;
                }
                if q >= hi || f.cl(q) == NONE {
                    return unknown("if");
                }
                vals.push(self.ev_block(fi, q));
                q = f.cl(q) + 1;
                if q < hi && f.t(q) == "else" {
                    q += 1;
                    if q < hi && f.t(q) == "{" {
                        if f.cl(q) == NONE {
                            return unknown("if");
                        }
                        vals.push(self.ev_block(fi, q));
                        q = f.cl(q) + 1;
                    }
                    continue;
                }
                break;
            }
            if q != hi {
                return unknown("if-postfix");
            }
            union(vals)
        }

        /// Value of the block `{ .. }` opening at `open`: its tail expression.
        fn ev_block(&self, fi: usize, open: usize) -> V {
            let f = &self.files[fi];
            let hi = f.cl(open);
            if hi == NONE {
                return unknown("block");
            }
            let mut q = open + 1;
            let mut start = q;
            while q < hi {
                if f.t(q) == "(" || f.t(q) == "[" {
                    q = f.skip(q);
                    continue;
                }
                if f.t(q) == "{" {
                    let c = if f.cl(q) == NONE { hi } else { f.cl(q) };
                    let nxt = f.t(c + 1);
                    let block_stmt = matches!(
                        f.t(start),
                        "if" | "match" | "for" | "while" | "loop" | "unsafe"
                    ) || q == start;
                    if block_stmt && c + 1 < hi && !matches!(nxt, "else" | "." | "?") {
                        start = c + 1;
                    }
                    q = c + 1;
                    continue;
                }
                if f.t(q) == ";" {
                    start = q + 1;
                }
                q += 1;
            }
            self.ev(fi, start, hi)
        }

        // ---- locals

        fn find_binder(&self, fi: usize, q: usize, name: &str) -> Option<usize> {
            let f = &self.files[fi];
            let fid = inner_fn(f, q);
            let bs = f.binders.get_or_init(|| build_binders(f));
            let mut best: Option<usize> = None;
            if let Some(list) = bs.by_name.get(name) {
                for &bi in list {
                    let b = &bs.list[bi];
                    if b.fn_id == fid && b.vlo <= q && q < b.vhi {
                        if best.map_or(true, |c| b.pos > bs.list[c].pos) {
                            best = Some(bi);
                        }
                    }
                }
            }
            best
        }

        fn closure_of(&self, fi: usize, q: usize, name: &str) -> Option<usize> {
            let f = &self.files[fi];
            let bi = self.find_binder(fi, q, name)?;
            let bs = f.binders.get()?;
            let b = &bs.list[bi];
            if b.kind != BKind::Let {
                return None;
            }
            let ilo = b.ilo?;
            let starts_closure = matches!(f.t(ilo), "|" | "||")
                || (f.t(ilo) == "move" && matches!(f.t(ilo + 1), "|" | "||"));
            if !starts_closure {
                return None;
            }
            match parse_pat(f, b.plo, b.phi) {
                Pat::Id(n) if n == name => Some(bi),
                _ => None,
            }
        }

        fn call_closure(&self, fi: usize, bi: usize, args: &[Arg]) -> V {
            let f = &self.files[fi];
            let b = &f.binders.get().expect("binders built").list[bi];
            let ilo = b.ilo.unwrap_or(b.ihi);
            let mut q = ilo;
            if f.t(q) == "move" {
                q += 1;
            }
            let mut names: Vec<Option<String>> = Vec::new();
            if f.t(q) == "|" {
                let mut e = q + 1;
                while e < b.ihi && f.t(e) != "|" {
                    e += 1;
                }
                for (mut a, bb) in split_commas(f, q + 1, e, true) {
                    while a < bb && matches!(f.t(a), "mut" | "ref" | "&") {
                        a += 1;
                    }
                    names.push(if a < bb && f.is_id(a) {
                        Some(f.t(a).to_string())
                    } else {
                        None
                    });
                }
                q = e + 1;
            } else {
                q += 1;
            }
            if q < b.ihi && f.t(q) == "->" {
                while q < b.ihi && f.t(q) != "{" {
                    q += 1;
                }
            }
            if names.len() != args.len() {
                return unknown("closure arity");
            }
            let mut vals: HashMap<String, V> = HashMap::new();
            for (nm, a) in names.into_iter().zip(args.iter()) {
                if let Some(nm) = nm {
                    vals.insert(nm, self.ev(a.0, a.1, a.2));
                }
            }
            let ihi = b.ihi;
            self.env.borrow_mut().push(Env {
                fi,
                lo: q,
                hi: ihi,
                vals,
            });
            let r = self.ev(fi, q, ihi);
            self.env.borrow_mut().pop();
            r
        }

        fn ev_local(&self, fi: usize, q: usize, name: &str) -> V {
            {
                let env = self.env.borrow();
                for e in env.iter().rev() {
                    if e.fi == fi && e.lo <= q && q < e.hi {
                        if let Some(v) = e.vals.get(name) {
                            return v.clone();
                        }
                    }
                }
            }
            let f = &self.files[fi];
            if let Some(bi) = self.find_binder(fi, q, name) {
                return self.eval_binder(fi, bi, name);
            }
            if let Some(fid) = inner_fn(f, q) {
                if let Some(p) = f.fns[fid]
                    .params
                    .iter()
                    .position(|p| p.as_deref() == Some(name))
                {
                    let mut v = V::default();
                    v.params.insert((p, Vec::new()));
                    return v;
                }
            }
            unknown(format!("unbound {name}"))
        }

        fn mutated(&self, fi: usize, b: &Binder, name: &str) -> bool {
            let f = &self.files[fi];
            for i in b.vlo..b.vhi.min(f.n()) {
                if f.is_id(i) && f.t(i) == name && i > 0 {
                    let p = f.t(i - 1);
                    if matches!(p, "let" | "ref" | "." | "::") {
                        continue;
                    }
                    if p == "mut" {
                        if i >= 2 && f.t(i - 2) == "&" {
                            return true;
                        }
                        continue;
                    }
                    if matches!(
                        f.t(i + 1),
                        "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "^=" | "|=" | "&="
                    ) {
                        return true;
                    }
                }
            }
            false
        }

        fn eval_binder(&self, fi: usize, bi: usize, name: &str) -> V {
            let f = &self.files[fi];
            let b = &f.binders.get().expect("binders built").list[bi];
            if self.mutated(fi, b, name) {
                return unknown(format!("reassigned {name}"));
            }
            let ilo = match b.ilo {
                Some(x) => x,
                None => return unknown(format!("deferred init {name}")),
            };
            let pat = parse_pat(f, b.plo, b.phi);
            let path = match bind_path(&pat, name) {
                BindRes::Path(p) => p,
                _ => return unknown(format!("pattern {name}")),
            };
            // A struct-variant step ignores the scrutinee's value, so do not
            // evaluate it (the constructor union replaces it).
            let mut v = if matches!(path.first(), Some(Step::Var(..))) {
                V::default()
            } else if b.kind == BKind::For {
                self.ev_iter(fi, ilo, b.ihi)
            } else {
                self.ev(fi, ilo, b.ihi)
            };
            for step in path {
                v = match step {
                    Step::Idx(i) => project(&v, &Proj::Idx(i)),
                    Step::Var(en, var, field) => self.variant_field(&en, &var, &field),
                };
            }
            v
        }

        // ---- enum struct-variant patterns

        /// Lazily built index of struct-variant (`Enum::Variant { .. }`) and
        /// plain struct (`Name { .. }`) constructors across the workspace.
        fn ctors(&self) -> &CtorIndex {
            self.ctors.get_or_init(|| build_ctor_index(self.files))
        }

        /// Value of `field` of one constructor literal whose `{` is at `open`.
        /// Parameters of the constructing function cannot be followed (their
        /// indices belong to that function), so a field read off a typed
        /// parameter becomes the union of that struct type's literals.
        ///
        /// `None`: the literal has no such field and no `..base` update, so it
        /// cannot be a constructor of the type/variant the field was bound
        /// from (a variant has every one of its fields) and is skipped.
        fn ctor_field(&self, fi: usize, open: usize, field: &str) -> Option<V> {
            let f = &self.files[fi];
            let close = f.cl(open);
            if close == NONE {
                return Some(unknown("constructor"));
            }
            let mut has_update = false;
            for (a, b) in split_commas(f, open + 1, close, false) {
                if f.t(a) == ".." {
                    has_update = true;
                    continue;
                }
                if b - a >= 3 && f.is_id(a) && f.t(a) == field && f.t(a + 1) == ":" {
                    let v = self.ev(fi, a + 2, b);
                    return Some(self.resolve_ctor_params(fi, a, v));
                }
                if b - a == 1 && f.is_id(a) && f.t(a) == field {
                    let v = self.ev_local(fi, a, field);
                    return Some(self.resolve_ctor_params(fi, a, v));
                }
            }
            if has_update {
                Some(unknown(format!(
                    "constructor at {}:{} takes field {field} from a `..` base",
                    f.rel,
                    f.line(open)
                )))
            } else {
                None
            }
        }

        fn resolve_ctor_params(&self, fi: usize, at: usize, v: V) -> V {
            if v.params.is_empty() {
                return v;
            }
            let f = &self.files[fi];
            let d = self.depth.get();
            if d > MAX_DEPTH {
                return unknown("depth");
            }
            self.depth.set(d + 1);
            let fid = inner_fn(f, at);
            let mut parts = vec![no_params(&v)];
            for (p, path) in &v.params {
                let ty = fid.and_then(|x| f.fns[x].param_types.get(*p).cloned().flatten());
                match (path.first(), ty) {
                    (Some(Proj::Field(fld)), Some(ty))
                        if self.ctors().structs.contains_key(&ty) =>
                    {
                        let sub = self.struct_field(&ty, fld);
                        parts.push(proj_path(&sub, &path[1..]));
                    }
                    _ => parts.push(unknown(format!(
                        "constructor field depends on parameter {p} of `{}`",
                        fid.map_or("?", |x| f.fns[x].name.as_str())
                    ))),
                }
            }
            self.depth.set(d);
            union(parts)
        }

        /// Union of `field` over every literal of struct `ty`.
        fn struct_field(&self, ty: &str, field: &str) -> V {
            let key = (String::new(), ty.to_string(), field.to_string());
            if let Some(v) = self.variant_cache.borrow().get(&key) {
                return v.clone();
            }
            if !self.variant_stack.borrow_mut().insert(key.clone()) {
                return unknown(format!("recursive struct {ty}"));
            }
            let mut vals: Vec<V> = match self.ctors().structs.get(ty) {
                Some(list) if list.len() > MAX_CTOR_FANOUT => vec![unknown(format!(
                    "{} literals of struct {ty} exceed the fan-out bound",
                    list.len()
                ))],
                Some(list) => list
                    .iter()
                    .filter_map(|&(fi, open)| self.ctor_field(fi, open, field))
                    .collect(),
                None => vec![unknown(format!("no literal of struct {ty}"))],
            };
            if vals.is_empty() {
                vals.push(unknown(format!(
                    "no literal of struct {ty} has field {field}"
                )));
            }
            self.variant_stack.borrow_mut().remove(&key);
            let r = union(vals);
            self.variant_cache.borrow_mut().insert(key, r.clone());
            r
        }

        /// Union of `field` over every literal of exactly the struct variant
        /// `enum_name::variant` (`Self` on either side matches any enum), so a
        /// sibling variant (e.g. a native-image one) never contaminates it.
        fn variant_field(&self, enum_name: &str, variant: &str, field: &str) -> V {
            let key = (
                format!("{enum_name}::{variant}"),
                String::new(),
                field.to_string(),
            );
            if let Some(v) = self.variant_cache.borrow().get(&key) {
                return v.clone();
            }
            if !self.variant_stack.borrow_mut().insert(key.clone()) {
                return unknown(format!("recursive variant {enum_name}::{variant}"));
            }
            let mut vals: Vec<V> = Vec::new();
            if let Some(list) = self.ctors().variants.get(variant) {
                if list.len() > MAX_CTOR_FANOUT {
                    vals.push(unknown(format!(
                        "{} literals of {enum_name}::{variant} exceed the fan-out bound",
                        list.len()
                    )));
                } else {
                    for (en, fi, open) in list {
                        if en == enum_name || en == "Self" || enum_name == "Self" {
                            if let Some(v) = self.ctor_field(*fi, *open, field) {
                                vals.push(v);
                            }
                        }
                    }
                }
            }
            if vals.is_empty() {
                vals.push(unknown(format!("no literal of {enum_name}::{variant}")));
            }
            self.variant_stack.borrow_mut().remove(&key);
            let r = union(vals);
            self.variant_cache.borrow_mut().insert(key, r.clone());
            r
        }

        fn ev_iter(&self, fi: usize, lo: usize, hi: usize) -> V {
            let f = &self.files[fi];
            if hi >= lo + 5
                && f.t(hi - 1) == ")"
                && f.t(hi - 3) == "enumerate"
                && f.t(hi - 4) == "."
            {
                let inner = self.ev_iter(fi, lo, hi - 4);
                return V {
                    tup: Some(vec![unknown("index"), inner]),
                    ..V::default()
                };
            }
            for q in lo..hi {
                if f.t(q) == ".." {
                    return unknown("range");
                }
            }
            self.ev(fi, lo, hi)
        }

        // ---- consts

        fn ev_const(&self, fi: usize, q: usize, name: &str, is_path: bool) -> V {
            let f = &self.files[fi];
            let mut cands: Vec<(usize, usize)> = Vec::new();
            if let Some(list) = f.const_by_name.get(name) {
                if let Some(fid) = inner_fn(f, q) {
                    for &ci in list {
                        if f.consts[ci].fn_id == Some(fid) {
                            cands.push((fi, ci));
                        }
                    }
                }
                if cands.is_empty() {
                    for &ci in list {
                        if f.consts[ci].fn_id.is_none() {
                            cands.push((fi, ci));
                        }
                    }
                }
                if cands.is_empty() {
                    for &ci in list {
                        cands.push((fi, ci));
                    }
                }
            }
            if cands.is_empty() {
                if let Some(all) = self.consts_by_name.get(name) {
                    if (is_path && !all.is_empty()) || all.len() == 1 {
                        cands = all.clone();
                    }
                }
            }
            if cands.is_empty() {
                return unknown(format!("unresolved const {name}"));
            }
            let mut vals = Vec::new();
            for (cf, ci) in cands {
                let key = (cf, ci);
                let cached = self.const_cache.borrow().get(&key).cloned();
                if let Some(v) = cached {
                    vals.push(v);
                    continue;
                }
                if self.const_stack.borrow().contains(&key) {
                    vals.push(unknown(format!("recursive const {name}")));
                    continue;
                }
                self.const_stack.borrow_mut().insert(key);
                let c = &self.files[cf].consts[ci];
                let v = self.ev(cf, c.lo, c.hi);
                self.const_stack.borrow_mut().remove(&key);
                self.const_cache.borrow_mut().insert(key, v.clone());
                vals.push(v);
            }
            union(vals)
        }

        // ---- calls of user functions

        fn call(&self, fi: usize, name: &str, args: &[Arg], method: bool) -> V {
            let mut cands: Vec<(usize, usize, usize)> = Vec::new();
            if let Some(list) = self.fns_by_name.get(name) {
                for &(cf, ci) in list {
                    let d = &self.files[cf].fns[ci];
                    if method {
                        if d.has_self && d.params.len() == args.len() {
                            cands.push((cf, ci, 0));
                        }
                    } else if d.has_self && d.params.len() + 1 == args.len() {
                        cands.push((cf, ci, 1));
                    } else if !d.has_self && d.params.len() == args.len() {
                        cands.push((cf, ci, 0));
                    }
                }
            }
            if cands.is_empty() {
                return unknown(format!("call {name}"));
            }
            // Same-file definitions shadow the rest; a name that is still
            // defined many times (`new`, `default`, `from`, ...) is a generic
            // factory, not a module selector, and stays unresolved instead of
            // unioning every body in the workspace.
            if cands.len() > 1 {
                let same: Vec<_> = cands.iter().copied().filter(|c| c.0 == fi).collect();
                if !same.is_empty() {
                    cands = same;
                }
            }
            if cands.len() > MAX_CALL_CANDS {
                return unknown(format!(
                    "ambiguous call {name} ({} candidates)",
                    cands.len()
                ));
            }
            let mut vals = Vec::new();
            for (cf, ci, off) in cands {
                let r = self.fn_returns(cf, ci);
                for (p, path) in &r.params {
                    let ai = p + off;
                    if ai < args.len() {
                        let a = args[ai];
                        vals.push(proj_path(&self.ev(a.0, a.1, a.2), path));
                    } else {
                        vals.push(unknown("arg missing"));
                    }
                }
                vals.push(no_params(&r));
            }
            union(vals)
        }

        /// Union of a function's tail expression and `return` expressions;
        /// parameters stay symbolic in `params`.
        fn fn_returns(&self, cf: usize, ci: usize) -> V {
            let key = (cf, ci);
            let cached = self.ret_cache.borrow().get(&key).cloned();
            if let Some(v) = cached {
                return v;
            }
            let f = &self.files[cf];
            let d = &f.fns[ci];
            if self.ret_stack.borrow().contains(&key) {
                return unknown(format!("recursive {}", d.name));
            }
            self.ret_stack.borrow_mut().insert(key);
            let mut vals = vec![self.ev_block(cf, d.open)];
            for i in d.open + 1..d.close {
                if f.is_id(i) && f.t(i) == "return" && inner_fn(f, i) == Some(ci) {
                    let mut q = i + 1;
                    while q < d.close {
                        if f.is_open(q) {
                            q = f.skip(q);
                            continue;
                        }
                        if f.t(q) == ";" || f.t(q) == "," || f.is_closer(q) {
                            break;
                        }
                        q += 1;
                    }
                    vals.push(self.ev(cf, i + 1, q));
                }
            }
            let r = union(vals);
            self.ret_stack.borrow_mut().remove(&key);
            self.ret_cache.borrow_mut().insert(key, r.clone());
            r
        }
    }

    // ------------------------------------------------------------- sink scan

    /// One module-argument position of a sink or wrapper.
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct Spec {
        arity: Option<usize>,
        arg: usize,
        path: Vec<Proj>,
        has_self: bool,
        recv: Option<String>,
    }

    fn roots() -> Vec<(String, Spec)> {
        fn s(
            name: &str,
            arity: usize,
            arg: usize,
            has_self: bool,
            recv: Option<&str>,
            path: Vec<Proj>,
        ) -> (String, Spec) {
            (
                name.to_string(),
                Spec {
                    arity: Some(arity),
                    arg,
                    path,
                    has_self,
                    recv: recv.map(str::to_string),
                },
            )
        }
        vec![
            s("ensure_kernel", 3, 0, true, None, vec![]),
            s("ensure_kernel_public", 3, 0, true, None, vec![]),
            s("compile_and_load_kernel", 9, 6, false, None, vec![]),
            s("load_kernel_module", 8, 5, false, None, vec![]),
            s("module_load_or_recompile", 6, 2, false, None, vec![]),
            s("compile", 2, 0, true, Some("compiler"), vec![]),
            s("compile_for_symbol", 3, 0, true, Some("compiler"), vec![]),
            s("recompile", 3, 0, true, Some("compiler"), vec![]),
            s(
                "compile_batch",
                1,
                0,
                true,
                Some("compiler"),
                vec![Proj::Idx(0)],
            ),
            s(
                "compile_batch_for_symbols",
                1,
                0,
                true,
                Some("compiler"),
                vec![Proj::Idx(0)],
            ),
            s(
                "begin_batch",
                2,
                0,
                true,
                Some("compiler"),
                vec![Proj::Field("module".to_string())],
            ),
        ]
    }

    /// A call of a runtime module sink (or of a wrapper that forwards to one).
    #[derive(Clone, Debug)]
    pub struct Site {
        /// Workspace-relative path, e.g. `crates/rdna-compute/src/gemm.rs`.
        pub file: String,
        pub line: u32,
        pub fn_name: String,
        pub call: String,
        /// Source text of the module argument.
        pub expr: String,
        /// Source text of every argument (module, source, symbol, ...).
        pub args: Vec<String>,
        /// Statically resolved HIP module names.
        pub mods: Vec<String>,
        /// Module names that only feed native images (never HIP sinks).
        pub native: Vec<String>,
        /// Why part of the value could not be resolved (empty = fully static).
        pub dynamic: Vec<String>,
        /// The module is (also) a forwarded parameter of the enclosing wrapper.
        pub forwards: bool,
        pub(crate) tok: usize,
    }

    impl Site {
        /// `crates/.../file.rs::fn` key of the enclosing function.
        pub fn key(&self) -> String {
            format!("{}::{}", self.file, self.fn_name)
        }
    }

    pub struct Scan {
        pub sites: Vec<Site>,
        /// Human-readable list of discovered wrapper functions.
        pub wrappers: Vec<String>,
        pub files: usize,
        pub tokens: usize,
    }

    impl Scan {
        /// module -> sites that resolve it statically.
        pub fn static_modules(&self) -> BTreeMap<String, Vec<&Site>> {
            let mut out: BTreeMap<String, Vec<&Site>> = BTreeMap::new();
            for s in &self.sites {
                for m in &s.mods {
                    out.entry(m.clone()).or_default().push(s);
                }
            }
            out
        }

        /// `crates/..::fn` -> sites with an unresolved module expression.
        pub fn dynamic_sites(&self) -> BTreeMap<String, Vec<&Site>> {
            let mut out: BTreeMap<String, Vec<&Site>> = BTreeMap::new();
            for s in &self.sites {
                if !s.dynamic.is_empty() {
                    out.entry(s.key()).or_default().push(s);
                }
            }
            out
        }

        /// Debug TSV: file line fn call mods native ok forwards reasons args.
        pub fn to_tsv(&self) -> String {
            let mut out = String::new();
            for s in &self.sites {
                out.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    s.file,
                    s.line,
                    s.fn_name,
                    s.call,
                    s.mods.join(","),
                    s.native.join(","),
                    u8::from(s.dynamic.is_empty()),
                    u8::from(s.forwards),
                    s.dynamic.join(" | "),
                    s.args.join(" ;; "),
                ));
            }
            out
        }
    }

    fn describe(name: &str, s: &Spec) -> String {
        let path: Vec<String> = s
            .path
            .iter()
            .map(|p| match p {
                Proj::Idx(i) => format!(".{i}"),
                Proj::Field(n) => format!(".{n}"),
            })
            .collect();
        format!(
            "{name}/{} arg{}{}{}",
            s.arity.map_or("*".to_string(), |a| a.to_string()),
            s.arg,
            path.concat(),
            s.recv
                .as_ref()
                .map_or(String::new(), |r| format!(" recv={r}")),
        )
    }

    fn scan_files(files: Vec<File>) -> Scan {
        let an = An::new(&files);
        let mut specs: HashMap<String, Vec<Spec>> = HashMap::new();
        for (name, spec) in roots() {
            specs.entry(name).or_default().push(spec);
        }
        let root_count: usize = specs.values().map(Vec::len).sum();
        let mut sites: Vec<Site> = Vec::new();
        for _round in 0..16 {
            sites.clear();
            let mut new: Vec<(String, Spec)> = Vec::new();
            for (fi, f) in files.iter().enumerate() {
                if f.rel.ends_with("rdna-compute/src/compiler.rs") {
                    // KernelCompiler implementation: forwards its own parameters.
                    continue;
                }
                for (name, list) in &specs {
                    let idxs = match f.calls.get(name) {
                        Some(i) => i,
                        None => continue,
                    };
                    for &i in idxs {
                        let fid = match inner_fn(f, i) {
                            Some(x) => x,
                            None => continue,
                        };
                        let fd = &f.fns[fid];
                        let o = i + 1;
                        let close = f.cl(o);
                        if close == NONE {
                            continue;
                        }
                        let method = f.prev(i) == ".";
                        let args = split_commas(f, o + 1, close, false);
                        for spec in list {
                            if method && !spec.has_self {
                                continue;
                            }
                            let off = if !method && spec.has_self { 1 } else { 0 };
                            if let Some(ar) = spec.arity {
                                if args.len() < off || args.len() - off != ar {
                                    continue;
                                }
                            }
                            if let Some(rv) = &spec.recv {
                                let ok_recv = method && i >= 2 && f.t(i - 2) == rv.as_str();
                                if !ok_recv {
                                    continue;
                                }
                            }
                            let idx = spec.arg + off;
                            if idx >= args.len() {
                                continue;
                            }
                            let (a, b) = args[idx];
                            let v = proj_path(&an.ev(fi, a, b), &spec.path);
                            for (p, ppath) in &v.params {
                                let w = Spec {
                                    arity: Some(fd.params.len()),
                                    arg: *p,
                                    path: ppath.clone(),
                                    has_self: fd.has_self,
                                    recv: None,
                                };
                                let known = specs.get(&fd.name).map_or(false, |l| l.contains(&w))
                                    || new.iter().any(|(n, s2)| n == &fd.name && s2 == &w);
                                if !known {
                                    new.push((fd.name.clone(), w));
                                }
                            }
                            sites.push(Site {
                                file: f.rel.clone(),
                                line: f.line(i),
                                fn_name: fd.name.clone(),
                                call: name.clone(),
                                expr: f.text(a, b).chars().take(200).collect(),
                                args: args
                                    .iter()
                                    .map(|&(x, y)| f.text(x, y).chars().take(160).collect())
                                    .collect(),
                                mods: v.strs.iter().cloned().collect(),
                                native: v.nstrs.iter().cloned().collect(),
                                dynamic: v.unk.clone(),
                                forwards: !v.params.is_empty(),
                                tok: i,
                            });
                        }
                    }
                }
            }
            if new.is_empty() {
                break;
            }
            for (name, spec) in new {
                specs.entry(name).or_default().push(spec);
            }
        }
        sites.sort_by(|x, y| {
            (x.file.as_str(), x.line, x.tok, x.call.as_str()).cmp(&(
                y.file.as_str(),
                y.line,
                y.tok,
                y.call.as_str(),
            ))
        });
        sites.dedup_by(|x, y| {
            x.file == y.file && x.tok == y.tok && x.call == y.call && x.expr == y.expr
        });
        // A wrapper forwards its parameter into a sink: its callers (traced
        // above) carry the static verdict, but the forwarding site itself is
        // always a dynamic site that must be allowlisted by `file::fn`. A
        // wrapper that no scanned code calls says so explicitly.
        let called: HashSet<(String, String)> = sites
            .iter()
            .map(|s| (s.call.clone(), s.fn_name.clone()))
            .collect();
        for s in sites.iter_mut() {
            if !s.forwards {
                continue;
            }
            s.dynamic.push(
                "caller-module parameter forwarded; variants audited at traced callers".to_string(),
            );
            if !called
                .iter()
                .any(|(call, caller)| call == &s.fn_name && caller != &s.fn_name)
            {
                s.dynamic.push(format!(
                    "wrapper `{}` forwards its module parameter but no scanned code calls it",
                    s.fn_name
                ));
            }
        }
        let mut wrappers: Vec<String> = Vec::new();
        for (name, list) in &specs {
            for s in list {
                if !roots().iter().any(|(n, r)| n == name && r == s) {
                    wrappers.push(describe(name, s));
                }
            }
        }
        wrappers.sort();
        let tokens = files.iter().map(File::n).sum();
        let _ = root_count;
        Scan {
            sites,
            wrappers,
            files: files.len(),
            tokens,
        }
    }

    /// Drop files that are `#[cfg(test)] mod x;` targets (and their children).
    fn drop_test_mod_files(files: Vec<File>) -> Vec<File> {
        let mut excluded: HashSet<String> = HashSet::new();
        for f in &files {
            if f.excl_mods.is_empty() {
                continue;
            }
            let (base, file) = match f.rel.rfind('/') {
                Some(p) => (&f.rel[..p], &f.rel[p + 1..]),
                None => ("", f.rel.as_str()),
            };
            let stem = file.strip_suffix(".rs").unwrap_or(file);
            let dir = if matches!(stem, "mod" | "lib" | "main") {
                base.to_string()
            } else if base.is_empty() {
                stem.to_string()
            } else {
                format!("{base}/{stem}")
            };
            for m in &f.excl_mods {
                let pre = if dir.is_empty() {
                    String::new()
                } else {
                    format!("{dir}/")
                };
                excluded.insert(format!("{pre}{m}.rs"));
                excluded.insert(format!("{pre}{m}/mod.rs"));
            }
        }
        let is_excluded = |rel: &str| -> bool {
            if excluded.contains(rel) {
                return true;
            }
            excluded.iter().any(|e| {
                if let Some(dir) = e.strip_suffix("mod.rs") {
                    rel.starts_with(dir)
                } else {
                    let dir = format!("{}/", e.strip_suffix(".rs").unwrap_or(e));
                    rel.starts_with(&dir)
                }
            })
        };
        files
            .into_iter()
            .filter(|f| !f.whole && !is_excluded(&f.rel))
            .collect()
    }

    fn run_scan(files: Vec<File>) -> Scan {
        let files = drop_test_mod_files(files);
        // Evaluation recurses through expression nesting; give it room.
        std::thread::Builder::new()
            .name("kernel-registry-coverage-scan".to_string())
            .stack_size(512 << 20)
            .spawn(move || scan_files(files))
            .expect("spawn scanner thread")
            .join()
            .expect("scanner thread panicked")
    }

    /// Scan in-memory sources `(workspace-relative path, text)`.
    pub fn scan_sources(sources: Vec<(String, String)>) -> Scan {
        let files: Vec<File> = sources
            .into_iter()
            .map(|(rel, src)| build_file(&rel, src))
            .collect();
        run_scan(files)
    }

    /// Scan `crates/*/src/**/*.rs` below `crates_dir` (paths are reported as
    /// `crates/<crate>/src/...`). `tests`, `benches`, `examples` and `target`
    /// directories are skipped.
    pub fn scan_workspace(crates_dir: &Path) -> Scan {
        let mut paths: Vec<(String, std::path::PathBuf)> = Vec::new();
        let mut crate_dirs: Vec<_> = std::fs::read_dir(crates_dir)
            .unwrap_or_else(|e| panic!("read {}: {e}", crates_dir.display()))
            .filter_map(Result::ok)
            .map(|e| e.path())
            .collect();
        crate_dirs.sort();
        for cdir in crate_dirs {
            let name = match cdir.file_name().and_then(|s| s.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            let src_dir = cdir.join("src");
            if !src_dir.is_dir() {
                continue;
            }
            let mut stack = vec![(src_dir, format!("crates/{name}/src"))];
            while let Some((dir, rel)) = stack.pop() {
                let rd = match std::fs::read_dir(&dir) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                for entry in rd.filter_map(Result::ok) {
                    let p = entry.path();
                    let fname = entry.file_name().to_string_lossy().to_string();
                    if p.is_dir() {
                        if matches!(fname.as_str(), "tests" | "benches" | "examples" | "target") {
                            continue;
                        }
                        stack.push((p, format!("{rel}/{fname}")));
                    } else if fname.ends_with(".rs") {
                        paths.push((format!("{rel}/{fname}"), p));
                    }
                }
            }
        }
        paths.sort();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(16);
        let mut files: Vec<File> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= paths.len() {
                                break;
                            }
                            let (rel, path) = &paths[i];
                            let bytes = std::fs::read(path)
                                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
                            let src = String::from_utf8_lossy(&bytes).into_owned();
                            out.push(build_file(rel, src));
                        }
                        out
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("lexer thread panicked"))
                .collect()
        });
        files.sort_by(|a, b| a.rel.cmp(&b.rel));
        run_scan(files)
    }

    /// Arch requirements implied by a module name. Each group is satisfied when
    /// the module is registered for at least one arch of the group; all groups
    /// must be satisfied. Tokens are the `_`-separated words after the first.
    pub fn required_arch_groups(module: &str) -> Vec<Vec<&'static str>> {
        let mut out: Vec<Vec<&'static str>> = Vec::new();
        for tok in module.split('_').skip(1) {
            let group: Vec<&'static str> = match tok {
                "gfx1201" | "gfx12" => vec!["gfx1201"],
                "gfx1100" => vec!["gfx1100"],
                "gfx1151" => vec!["gfx1151"],
                "gfx11" => vec!["gfx1100", "gfx1151"],
                "gfx942" => vec!["gfx942"],
                "gfx906" => vec!["gfx906"],
                _ => continue,
            };
            if !out.contains(&group) {
                out.push(group);
            }
        }
        out
    }
}
// FRJ-SCANNER-END

use scanner::{required_arch_groups, Scan};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const FIXTURE_REL: &str = "tests/fixtures/lazy-kernel-allowlist.tsv";
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/lazy-kernel-allowlist.tsv"
);

/// Allowed allowlist categories.
const CATEGORIES: &[&str] = &[
    "non-core-family",
    "non-target-arch",
    "developer-opt-in",
    "no-caller",
    "codec-outside-core-rungs",
    "config-opt-in",
    "dynamic-enumerated",
    "dynamic-lazy",
];

// ------------------------------------------------------------------ fixture

#[derive(Debug, Clone)]
struct Allow {
    kind: String,
    name: String,
    category: String,
    line: usize,
}

/// `file.ext:123` (or `file.ext:123-130`) somewhere in the reason.
fn has_file_line(reason: &str) -> bool {
    reason
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '`' | '[' | ']'))
        .any(|tok| {
            let tok = tok.trim_end_matches(&['.', ':'][..]);
            let Some((file, nums)) = tok.rsplit_once(':') else {
                return false;
            };
            let digits_ok = !nums.is_empty()
                && nums
                    .split('-')
                    .all(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
            let ext_ok = file.rsplit_once('.').is_some_and(|(stem, ext)| {
                !stem.is_empty()
                    && !ext.is_empty()
                    && ext.chars().all(|c| c.is_ascii_alphanumeric())
            });
            digits_ok && ext_ok
        })
}

fn parse_fixture(text: &str, errors: &mut Vec<String>) -> Vec<Allow> {
    let mut out: Vec<Allow> = Vec::new();
    let mut seen: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (idx, raw) in text.lines().enumerate() {
        let line = idx + 1;
        if raw.trim().is_empty() || raw.trim_start().starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = raw.split('\t').collect();
        if cols.len() != 4 {
            errors.push(format!(
                "{FIXTURE_REL}:{line}: expected 4 tab-separated columns (kind, name, category, reason), found {}",
                cols.len()
            ));
            continue;
        }
        let (kind, name, category, reason) = (
            cols[0].trim(),
            cols[1].trim(),
            cols[2].trim(),
            cols[3].trim(),
        );
        if kind == "kind" && name == "name" {
            continue;
        }
        let mut ok = true;
        if kind != "module" && kind != "dynamic" {
            errors.push(format!(
                "{FIXTURE_REL}:{line}: kind `{kind}` must be `module` or `dynamic`"
            ));
            ok = false;
        }
        if name.is_empty() {
            errors.push(format!("{FIXTURE_REL}:{line}: empty name"));
            ok = false;
        } else if kind == "dynamic" && !(name.starts_with("crates/") && name.contains(".rs::")) {
            errors.push(format!(
                "{FIXTURE_REL}:{line}: dynamic name `{name}` must look like `crates/<crate>/src/<file>.rs::<fn>`"
            ));
            ok = false;
        }
        if !CATEGORIES.contains(&category) {
            errors.push(format!(
                "{FIXTURE_REL}:{line}: category `{category}` is not one of {}",
                CATEGORIES.join(", ")
            ));
            ok = false;
        } else if kind == "module" && category.starts_with("dynamic-") {
            errors.push(format!(
                "{FIXTURE_REL}:{line}: `module` rows cannot use the `{category}` category"
            ));
            ok = false;
        }
        if reason.is_empty() {
            errors.push(format!("{FIXTURE_REL}:{line}: empty reason"));
            ok = false;
        } else if !has_file_line(reason) {
            errors.push(format!(
                "{FIXTURE_REL}:{line}: reason must cite a `file:line` (e.g. gemm.rs:123)"
            ));
            ok = false;
        }
        if !ok {
            continue;
        }
        let key = (kind.to_string(), name.to_string());
        if let Some(first) = seen.get(&key) {
            errors.push(format!(
                "{FIXTURE_REL}:{line}: duplicate {kind} row `{name}` (first at line {first})"
            ));
            continue;
        }
        seen.insert(key, line);
        out.push(Allow {
            kind: kind.to_string(),
            name: name.to_string(),
            category: category.to_string(),
            line,
        });
    }
    out
}

// ----------------------------------------------------------------- registry

/// module -> arches whose registry entries include it.
fn registry_modules() -> BTreeMap<String, BTreeSet<&'static str>> {
    let mut out: BTreeMap<String, BTreeSet<&'static str>> = BTreeMap::new();
    for &arch in rdna_compute::kernel_registry::SUPPORTED_ARCHES {
        let list = rdna_compute::kernel_registry::entries(arch, "")
            .unwrap_or_else(|e| panic!("kernel_registry::entries({arch}, \"\"): {e:?}"));
        for entry in list {
            out.entry(entry.module.to_string())
                .or_default()
                .insert(arch);
        }
    }
    out
}

/// Required groups the registry does not satisfy for `module` (empty = fine).
fn unmet_groups(
    module: &str,
    registry: &BTreeMap<String, BTreeSet<&'static str>>,
) -> Result<(), String> {
    let Some(arches) = registry.get(module) else {
        return Err("is not in the kernel registry for any supported arch".to_string());
    };
    let mut missing: Vec<String> = Vec::new();
    for group in required_arch_groups(module) {
        if !group.iter().any(|a| arches.contains(a)) {
            missing.push(group.join(" or "));
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "is named for {} but the registry only has it for [{}]",
            missing.join(" and "),
            arches.iter().copied().collect::<Vec<_>>().join(", ")
        ))
    }
}

fn locations(sites: &[&scanner::Site]) -> String {
    let mut out: Vec<String> = sites
        .iter()
        .map(|s| format!("{}:{} ({} in `{}`)", s.file, s.line, s.call, s.fn_name))
        .collect();
    out.sort();
    out.dedup();
    out.join("; ")
}

/// Core of the gate, separated for the scanner unit tests: returns failures.
fn check(
    scan: &Scan,
    registry: &BTreeMap<String, BTreeSet<&'static str>>,
    allow: &[Allow],
) -> Vec<String> {
    let mut failures: Vec<String> = Vec::new();
    let static_mods = scan.static_modules();
    let dynamic = scan.dynamic_sites();
    let allowed_modules: BTreeMap<&str, &Allow> = allow
        .iter()
        .filter(|a| a.kind == "module")
        .map(|a| (a.name.as_str(), a))
        .collect();
    let allowed_dynamic: BTreeMap<&str, &Allow> = allow
        .iter()
        .filter(|a| a.kind == "dynamic")
        .map(|a| (a.name.as_str(), a))
        .collect();

    for (module, sites) in &static_mods {
        if allowed_modules.contains_key(module.as_str()) {
            continue;
        }
        if let Err(why) = unmet_groups(module, registry) {
            let first = sites[0];
            failures.push(format!(
                "{}:{}: runtime module `{module}` {why}; register it for the right arch or allowlist it in {FIXTURE_REL} [all sites: {}]",
                first.file,
                first.line,
                locations(sites)
            ));
        }
    }
    for (key, sites) in &dynamic {
        if !allowed_dynamic.contains_key(key.as_str()) {
            let first = sites[0];
            let why: BTreeSet<&str> = sites
                .iter()
                .flat_map(|s| s.dynamic.iter().map(String::as_str))
                .collect();
            failures.push(format!(
                "{}:{}: dynamic module expression (`{}`; unresolved: {}) in `{key}` is not allowlisted in {FIXTURE_REL}",
                first.file,
                first.line,
                first.expr,
                why.into_iter().collect::<Vec<_>>().join("; ")
            ));
        }
    }
    for a in allow {
        match a.kind.as_str() {
            "module" => match static_mods.get(&a.name) {
                None => failures.push(format!(
                    "{FIXTURE_REL}:{}: stale module allowlist row `{}` ({}): no runtime call site names it any more",
                    a.line, a.name, a.category
                )),
                Some(_) => {
                    if unmet_groups(&a.name, registry).is_ok() {
                        failures.push(format!(
                            "{FIXTURE_REL}:{}: stale module allowlist row `{}` ({}): the registry now satisfies its required arches",
                            a.line, a.name, a.category
                        ));
                    }
                }
            },
            "dynamic" => {
                if !dynamic.contains_key(&a.name) {
                    failures.push(format!(
                        "{FIXTURE_REL}:{}: stale dynamic allowlist row `{}` ({}): no unresolved module expression remains in that function",
                        a.line, a.name, a.category
                    ));
                }
            }
            _ => {}
        }
    }
    failures
}

fn emit_dump(scan: &Scan) {
    let Ok(target) = std::env::var("KERNEL_REGISTRY_COVERAGE_DUMP") else {
        return;
    };
    let tsv = scan.to_tsv();
    if target == "-" || target.is_empty() {
        eprintln!("{tsv}");
    } else {
        std::fs::write(&target, tsv).unwrap_or_else(|e| panic!("write {target}: {e}"));
    }
}

#[test]
fn every_runtime_hip_module_is_registered_or_allowlisted() {
    let crates_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir");
    let scan = scanner::scan_workspace(crates_dir);
    emit_dump(&scan);

    let static_mods = scan.static_modules();
    let dynamic = scan.dynamic_sites();
    let registry = registry_modules();
    let native: BTreeSet<&str> = scan
        .sites
        .iter()
        .flat_map(|s| s.native.iter().map(String::as_str))
        .collect();
    eprintln!(
        "kernel_registry_coverage: {} files, {} tokens, {} sink sites, {} wrappers, {} static modules, {} dynamic sites in {} functions, {} native-only modules, {} registry modules",
        scan.files,
        scan.tokens,
        scan.sites.len(),
        scan.wrappers.len(),
        static_mods.len(),
        dynamic.values().map(Vec::len).sum::<usize>(),
        dynamic.len(),
        native.len(),
        registry.len()
    );
    for w in &scan.wrappers {
        eprintln!("  wrapper {w}");
    }
    for (key, sites) in &dynamic {
        for s in sites {
            eprintln!(
                "  dynamic {key} @{} `{}` ({})",
                s.line,
                s.expr,
                s.dynamic.join("; ")
            );
        }
    }
    for (module, sites) in &static_mods {
        let arches: Vec<&str> = registry
            .get(module)
            .map(|a| a.iter().copied().collect())
            .unwrap_or_default();
        eprintln!(
            "  static {module} [{}] {}",
            arches.join(","),
            locations(sites)
        );
    }

    assert!(
        scan.sites.len() > 100 && static_mods.len() > 100,
        "scanner found only {} sites / {} modules: the extraction is broken",
        scan.sites.len(),
        static_mods.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let text = std::fs::read_to_string(FIXTURE).unwrap_or_else(|e| panic!("read {FIXTURE}: {e}"));
    let allow = parse_fixture(&text, &mut failures);
    failures.extend(check(&scan, &registry, &allow));
    assert!(
        failures.is_empty(),
        "{} kernel registry coverage failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ------------------------------------------------------------ scanner tests

fn scan1(src: &str) -> Scan {
    scanner::scan_sources(vec![("crates/t/src/lib.rs".to_string(), src.to_string())])
}

fn mods(scan: &Scan) -> Vec<String> {
    scan.static_modules().keys().cloned().collect()
}

fn dyn_keys(scan: &Scan) -> Vec<String> {
    scan.dynamic_sites().keys().cloned().collect()
}

#[test]
fn lexer_ignores_comments_strings_raw_strings_and_chars() {
    let src = r####"
impl Gpu {
    fn go(&mut self) {
        // self.ensure_kernel("line_comment", S, "f");
        /* self.ensure_kernel("block", S, "f"); /* nested */ self.ensure_kernel("block2", S, "f"); */
        let _doc = r#"self.ensure_kernel("in_raw", S, "f")"#;
        let _src = r###"ensure_kernel("in_raw3", "#", "f") } fn x() {"###;
        let _s = "self.ensure_kernel(\"in_str\", S, \"f\")";
        let _c = '"';
        let _d = '\'';
        let _b = b'"';
        let _l: &'static str = "x";
        self.ensure_kernel("real", SRC, "real_fn").unwrap();
    }
}
"####;
    let s = scan1(src);
    assert_eq!(mods(&s), vec!["real".to_string()], "{:?}", s.sites);
    assert_eq!(s.sites.len(), 1);
    assert_eq!(s.sites[0].line, 13);
    assert_eq!(s.sites[0].fn_name, "go");
    assert!(dyn_keys(&s).is_empty());
}

#[test]
fn cfg_test_items_and_test_files_are_excluded() {
    let lib = r#"
impl Gpu {
    fn live(&mut self) { self.ensure_kernel("live", S, "f"); }
    #[cfg(not(test))]
    fn not_test(&mut self) { self.ensure_kernel("not_test", S, "f"); }
    #[cfg(any(test, feature = "x"))]
    fn any_test(&mut self) { self.ensure_kernel("any_test", S, "f"); }
    #[cfg(test)]
    fn gated(&mut self) { self.ensure_kernel("gated", S, "f"); }
    #[test]
    fn unit(&mut self) { self.ensure_kernel("unit", S, "f"); }
    #[cfg(all(test, feature = "y"))]
    #[allow(dead_code)]
    fn all_test(&mut self) { self.ensure_kernel("all_test", S, "f"); }
}
#[cfg(test)]
mod tests {
    fn t(g: &mut Gpu) { g.ensure_kernel("in_tests_mod", S, "f"); }
}
#[cfg(test)]
mod more_tests;
fn after(g: &mut Gpu) { g.ensure_kernel("after", S, "f"); }
"#;
    let more = r#"fn t(g: &mut Gpu) { g.ensure_kernel("in_tests_file", S, "f"); }"#;
    let s = scanner::scan_sources(vec![
        ("crates/t/src/lib.rs".to_string(), lib.to_string()),
        ("crates/t/src/more_tests.rs".to_string(), more.to_string()),
    ]);
    assert_eq!(
        mods(&s),
        vec![
            "after".to_string(),
            "any_test".to_string(),
            "live".to_string(),
            "not_test".to_string()
        ]
    );
    let whole = scanner::scan_sources(vec![(
        "crates/t/src/x.rs".to_string(),
        "#![cfg(test)]\nfn t(g: &mut Gpu) { g.ensure_kernel(\"w\", S, \"f\"); }".to_string(),
    )]);
    assert!(whole.sites.is_empty());
}

#[test]
fn consts_are_resolved_with_function_scope_first() {
    let src = r#"
const MODULE: &str = "file_module";
const ALIAS: &str = MODULE;
impl Gpu {
    fn a(&mut self) {
        const MODULE: &str = "local_a";
        self.ensure_kernel(MODULE, SRC, "f");
    }
    fn b(&mut self) {
        self.ensure_kernel(MODULE, SRC, "f");
        self.ensure_kernel(ALIAS, SRC, "f");
        self.ensure_kernel(Self::MODULE, SRC, "f");
    }
    fn c(&mut self) {
        const SYMBOL: &str = "sym";
        self.ensure_kernel_public(SYMBOL, SRC, SYMBOL);
        self.ensure_kernel("conc", SRC, concat!("a", "b"));
        self.ensure_kernel(concat!("m_", "x"), SRC, SYMBOL);
    }
}
"#;
    let s = scan1(src);
    assert_eq!(
        mods(&s),
        ["file_module", "local_a", "m_x", "sym", "conc"]
            .iter()
            .map(|x| x.to_string())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    );
    let a = s.sites.iter().find(|x| x.fn_name == "a").unwrap();
    assert_eq!(a.mods, vec!["local_a".to_string()]);
    assert!(dyn_keys(&s).is_empty(), "{:?}", dyn_keys(&s));
}

#[test]
fn tuple_branches_expose_only_the_module_position() {
    let src = r#"
const SRC_A: &str = include_str!("a.hip");
impl Gpu {
    fn pick(&mut self, wide: bool, k: usize) {
        let (src, module) = if wide { (SRC_A, "mod_wide") } else { (SRC_B, "mod_narrow") };
        self.ensure_kernel(module, src, "f");
        let (kname, ksrc) = match k {
            0 => ("m0", SRC_A),
            1 => { ("m1", SRC_B) }
            _ => return Err(e),
        };
        self.ensure_kernel(kname, ksrc, kname);
        let table = [("t0", "sym0", SRC_A), ("t1", "sym1", SRC_B)];
        for (module, sym, src) in table.iter() {
            self.ensure_kernel(module, src, sym);
        }
        for (name, src) in [("lit_a", SRC_A), ("lit_b", SRC_B)] {
            self.ensure_kernel_public(name, src, name);
        }
    }
}
const TABLE: &[(&str, &str, &str)] = &[("c0", "S", "sy0"), ("c1", "S", "sy1")];
fn run(g: &mut Gpu) {
    for &(m, s, f) in TABLE.iter() { g.ensure_kernel(m, s, f); }
}
"#;
    let s = scan1(src);
    assert_eq!(
        mods(&s),
        vec![
            "c0",
            "c1",
            "lit_a",
            "lit_b",
            "m0",
            "m1",
            "mod_narrow",
            "mod_wide",
            "t0",
            "t1"
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>()
    );
    assert!(
        dyn_keys(&s).is_empty(),
        "{:?}",
        s.dynamic_sites().keys().collect::<Vec<_>>()
    );
}

#[test]
fn wrappers_and_tuple_forwarders_are_followed_transitively() {
    let src = r#"
impl Gpu {
    fn helper(&mut self, a: u32, module: &str, src: &str, sym: &str) {
        self.ensure_kernel(module, src, sym).unwrap();
    }
    fn outer(&mut self, name: &str) {
        self.helper(1, name, SRC, "s");
    }
    fn user(&mut self) {
        self.outer("outer_literal");
        self.helper(0, "helper_literal", SRC, "s");
    }
    fn launch(&mut self, k: (&'static str, &'static str, &'static str)) {
        self.ensure_kernel_public(k.0, k.1, k.2).unwrap();
    }
    fn launch2(&mut self, k: (&'static str, &'static str, &'static str)) {
        let (module, src, sym) = k;
        self.ensure_kernel_public(module, src, sym).unwrap();
    }
    fn tuple_user(&mut self, pm: bool) {
        self.launch(("tuple_literal", SRC, "s"));
        self.launch2(if pm { ("t_pm", SRC, "s") } else { ("t_hip", SRC, "s") });
        self.launch2(sel_for_arch("gfx1100"));
    }
}
fn sel_for_arch(arch: &str) -> (&'static str, &'static str, &'static str) {
    match arch {
        "gfx1100" => ("sel_gfx1100", SRC, "s"),
        _ => ("sel_other", SRC, "s"),
    }
}
"#;
    let s = scan1(src);
    assert_eq!(
        mods(&s),
        vec![
            "helper_literal",
            "outer_literal",
            "sel_gfx1100",
            "sel_other",
            "t_hip",
            "t_pm",
            "tuple_literal"
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>(),
        "{:?}",
        s.sites
    );
    // Every forwarding wrapper is a dynamic site keyed by its own function.
    assert_eq!(
        dyn_keys(&s),
        ["helper", "launch", "launch2", "outer"]
            .iter()
            .map(|f| format!("crates/t/src/lib.rs::{f}"))
            .collect::<Vec<_>>()
    );
    assert!(
        s.wrappers.iter().any(|w| w.starts_with("helper/")),
        "{:?}",
        s.wrappers
    );
    assert!(
        s.wrappers.iter().any(|w| w.starts_with("outer/")),
        "{:?}",
        s.wrappers
    );
    assert!(
        s.wrappers.iter().any(|w| w.starts_with("launch/")),
        "{:?}",
        s.wrappers
    );
}

#[test]
fn selectors_and_struct_tables_expose_module_literals() {
    let src = r#"
impl Gpu {
    fn go(&mut self, caps: &ArchCaps) {
        let (src, module) = kernels::gemm_for_arch(caps);
        self.ensure_kernel(module, src, "f");
        let (module2, _) = Self::tile_module(2);
        self.ensure_kernel_public(module2, SRC, "g");
        let spec = &SERIAL;
        self.ensure_kernel(spec.module, spec.src, spec.symbol);
        if let Some((src3, mod3)) = maybe_for_arch(caps) {
            self.ensure_kernel(mod3, src3, "h");
        }
    }
}
struct Spec { module: &'static str, src: &'static str, symbol: &'static str }
const SERIAL: Spec = Spec { module: "serial_mod", src: SRC, symbol: "serial_sym" };
pub fn gemm_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    if caps.is_gfx1201() {
        return (SRC_A, "gemm_gfx1201");
    }
    match caps.arch() {
        "gfx1100" | "gfx1151" => (SRC_B, "gemm_gfx11"),
        _ => (SRC_C, "gemm_generic"),
    }
}
fn tile_module(n: usize) -> (&'static str, &'static str) {
    (if n == 2 { "tile_two" } else { "tile_n" }, SRC)
}
fn maybe_for_arch(caps: &ArchCaps) -> Option<(&'static str, &'static str)> {
    if caps.x() { Some((SRC, "maybe_mod")) } else { None }
}
"#;
    let s = scan1(src);
    assert_eq!(
        mods(&s),
        vec![
            "gemm_generic",
            "gemm_gfx11",
            "gemm_gfx1201",
            "maybe_mod",
            "serial_mod",
            "tile_n",
            "tile_two"
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>(),
        "{:?}",
        s.sites
    );
    assert!(dyn_keys(&s).is_empty(), "{:?}", dyn_keys(&s));
}

#[test]
fn local_closure_constructors_keep_native_images_out() {
    let src = r#"
fn launch(gpu: &mut Gpu, pm: bool) {
    const GFX1151: &str = "hip_gfx1151";
    const GFX1201: &str = "hip_gfx1201";
    const PM_GFX1151: &str = "pm_gfx1151";
    let hip = |module, src, kernel| -> Code { (module, Ok(src), kernel) };
    let built = |module, image, kernel| -> Code { (module, Err(image), kernel) };
    let attend = if pm {
        built(PM_GFX1151, IMAGE, "pm_kernel")
    } else {
        hip(GFX1151, SRC, "hip_kernel")
    };
    let (convert, attend) = match pm {
        false => (hip(GFX1201, SRC, "c"), attend),
        true => (built(PM_GFX1151, IMAGE, "c_pm"), attend),
    };
    for (module, code, kernel) in [convert, attend].into_iter().skip(usize::from(pm)) {
        match code {
            Ok(src) => { gpu.ensure_kernel_public(module, src, kernel).unwrap(); }
            Err(image) => { gpu.ensure_embedded_kernel(module, image, kernel).unwrap(); }
        }
    }
}
"#;
    let s = scan1(src);
    assert_eq!(
        mods(&s),
        vec!["hip_gfx1151".to_string(), "hip_gfx1201".to_string()],
        "{:?}",
        s.sites
    );
    assert_eq!(s.sites.len(), 1);
    assert_eq!(
        s.sites[0].native,
        vec!["pm_gfx1151".to_string()],
        "native modules must be tracked separately: {:?}",
        s.sites
    );
    assert!(dyn_keys(&s).is_empty());
}

#[test]
fn unresolved_and_format_expressions_stay_dynamic() {
    let src = r#"
impl Gpu {
    fn fmt(&mut self, base: &str) {
        let module = format!("{base}_x");
        self.ensure_kernel(&module, SRC, "f");
        self.ensure_kernel(&format!("{}_y", base), SRC, "f");
    }
    fn reassigned(&mut self, wide: bool) {
        let mut m = "first";
        if wide { m = "second"; }
        self.ensure_kernel(m, SRC, "f");
    }
    fn unknown_call(&mut self) {
        let m = lookup_module();
        self.ensure_kernel(m, SRC, "f");
    }
    fn partly(&mut self, wide: bool, n: &str) {
        let m = if wide { "known" } else { n_name(n) };
        self.ensure_kernel(m, SRC, "f");
    }
    fn closure_param(&mut self) {
        let go = |m: &str| self.ensure_kernel(m, SRC, "f");
        go("x");
    }
}
"#;
    let s = scan1(src);
    let keys = dyn_keys(&s);
    for f in [
        "fmt",
        "reassigned",
        "unknown_call",
        "partly",
        "closure_param",
    ] {
        assert!(
            keys.contains(&format!("crates/t/src/lib.rs::{f}")),
            "missing dynamic key for {f}: {keys:?}"
        );
    }
    // The resolved arm of a partly-known expression is still checked statically.
    assert_eq!(mods(&s), vec!["known".to_string()]);
    let fmt_sites: Vec<_> = s.sites.iter().filter(|x| x.fn_name == "fmt").collect();
    assert_eq!(fmt_sites.len(), 2);
    assert!(fmt_sites
        .iter()
        .all(|x| x.mods.is_empty() && !x.dynamic.is_empty()));
}

#[test]
fn receiver_scoped_compiler_sinks_and_extra_roots() {
    let src = r#"
impl Gpu {
    fn go(&mut self, ctx: &Ctx) {
        self.compiler.compile("kc_compile", SRC).unwrap();
        self.compiler.compile_for_symbol("kc_symbol", SRC, "s").unwrap();
        ctx.recorder.compile("recorder_compile", SRC).unwrap();
        other.compile("other_compile", SRC).unwrap();
        crate::scratch::compile_and_load_kernel(
            &mut self.compiler, &self.hip, &mut self.modules, &mut self.functions,
            &mut self.route, false, "loaded_mod", SRC, "f",
        ).unwrap();
        self.compiler.compile_batch_for_symbols(&[("batch_a", S, "a"), ("batch_b", S, "b")]).unwrap();
    }
}
"#;
    let s = scan1(src);
    assert_eq!(
        mods(&s),
        vec![
            "batch_a",
            "batch_b",
            "kc_compile",
            "kc_symbol",
            "loaded_mod"
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>(),
        "{:?}",
        s.sites
    );
}

#[test]
fn arch_suffix_requirements() {
    use scanner::required_arch_groups as req;
    assert_eq!(req("gemm_wmma_gfx1201"), vec![vec!["gfx1201"]]);
    assert_eq!(req("gemm_wmma_gfx12"), vec![vec!["gfx1201"]]);
    assert_eq!(req("gemm_wmma_gfx12_ldsstage"), vec![vec!["gfx1201"]]);
    assert_eq!(req("gemv_gfx1100"), vec![vec!["gfx1100"]]);
    assert_eq!(req("gemv_gfx1151_row"), vec![vec!["gfx1151"]]);
    assert_eq!(req("gemv_gfx11"), vec![vec!["gfx1100", "gfx1151"]]);
    assert_eq!(req("x_gfx942"), vec![vec!["gfx942"]]);
    assert_eq!(req("x_gfx906"), vec![vec!["gfx906"]]);
    assert_eq!(
        req("a_gfx1100_to_gfx1151"),
        vec![vec!["gfx1100"], vec!["gfx1151"]]
    );
    assert!(req("gemv_hfq4g256").is_empty());
    assert!(
        req("gfx1201").is_empty(),
        "the leading word is the family, not a suffix"
    );
    assert!(req("xgfx1201").is_empty());
    assert!(req("gemm_gfx12000").is_empty());

    let mut registry: BTreeMap<String, BTreeSet<&'static str>> = BTreeMap::new();
    registry.insert("k_gfx11".into(), ["gfx1151"].into_iter().collect());
    registry.insert("k_gfx1100".into(), ["gfx1151"].into_iter().collect());
    registry.insert(
        "k_gfx12_v2".into(),
        ["gfx1201", "gfx1100"].into_iter().collect(),
    );
    registry.insert("plain".into(), ["gfx1100"].into_iter().collect());
    assert!(
        unmet_groups("k_gfx11", &registry).is_ok(),
        "gfx11 needs at least one RDNA3 arch"
    );
    assert!(
        unmet_groups("k_gfx1100", &registry).is_err(),
        "gfx1100 is exact"
    );
    assert!(unmet_groups("k_gfx12_v2", &registry).is_ok());
    assert!(unmet_groups("plain", &registry).is_ok());
    assert!(unmet_groups("missing", &registry).is_err());
}

#[test]
fn allowlist_checks_report_missing_stale_and_dynamic_sites() {
    let src = r#"
impl Gpu {
    fn go(&mut self, base: &str) {
        self.ensure_kernel("present", SRC, "f");
        self.ensure_kernel("absent_gfx1201", SRC, "f");
        self.ensure_kernel("wrong_gfx1100", SRC, "f");
        self.ensure_kernel("allowed_mod", SRC, "f");
        self.ensure_kernel(&format!("{base}_x"), SRC, "f");
    }
}
"#;
    let scan = scan1(src);
    let mut registry: BTreeMap<String, BTreeSet<&'static str>> = BTreeMap::new();
    registry.insert("present".into(), ["gfx1201"].into_iter().collect());
    registry.insert("wrong_gfx1100".into(), ["gfx1201"].into_iter().collect());

    let allow = |kind: &str, name: &str| Allow {
        kind: kind.to_string(),
        name: name.to_string(),
        category: "no-caller".to_string(),
        line: 7,
    };
    let none = check(&scan, &registry, &[]);
    assert!(
        none.iter()
            .any(|f| f.starts_with("crates/t/src/lib.rs:5:") && f.contains("absent_gfx1201")),
        "{none:?}"
    );
    assert!(
        none.iter()
            .any(|f| f.starts_with("crates/t/src/lib.rs:6:") && f.contains("wrong_gfx1100")),
        "{none:?}"
    );
    assert!(
        none.iter()
            .any(|f| f.starts_with("crates/t/src/lib.rs:7:") && f.contains("allowed_mod")),
        "{none:?}"
    );
    assert!(
        none.iter()
            .any(|f| f.starts_with("crates/t/src/lib.rs:8:") && f.contains("dynamic")),
        "{none:?}"
    );
    assert!(!none.iter().any(|f| f.contains("`present`")), "{none:?}");

    let full = [
        allow("module", "absent_gfx1201"),
        allow("module", "wrong_gfx1100"),
        allow("module", "allowed_mod"),
        allow("dynamic", "crates/t/src/lib.rs::go"),
    ];
    assert!(
        check(&scan, &registry, &full).is_empty(),
        "{:?}",
        check(&scan, &registry, &full)
    );

    let stale = [
        allow("module", "absent_gfx1201"),
        allow("module", "wrong_gfx1100"),
        allow("module", "allowed_mod"),
        allow("module", "present"),
        allow("module", "vanished_mod"),
        allow("dynamic", "crates/t/src/lib.rs::go"),
        allow("dynamic", "crates/t/src/lib.rs::vanished"),
    ];
    let f = check(&scan, &registry, &stale);
    assert_eq!(f.len(), 3, "{f:?}");
    assert!(
        f.iter()
            .all(|x| x.starts_with(FIXTURE_REL) && x.contains("stale")),
        "{f:?}"
    );
}

#[test]
fn fixture_validation() {
    let text = "\
# comment
kind\tname\tcategory\treason
module\ta_gfx12\tnon-target-arch\tonly gfx1100 caller, dispatch.rs:12
module\ta_gfx12\tnon-target-arch\tduplicate gemm.rs:1
module\tb\tbogus-category\treason gemm.rs:1
module\tc\tdynamic-lazy\tmodule rows cannot be dynamic gemm.rs:1
module\td\tno-caller\t
module\te\tno-caller\tno line reference here
module\tf\tno-caller
dynamic\tnot-a-key\tdynamic-lazy\tgemm.rs:1
dynamic\tcrates/x/src/y.rs::f\tdynamic-enumerated\tenumerated from the registry, scratch.rs:700-710
dynamic\tcrates/x/src/y.rs::g\tconfig-opt-in\tKV mode `kv_mode`, tensor_ops.rs:99.
module\th\tcodec-outside-core-rungs\tMQ5 has no core artifact (gemm.rs:2)
weird\tz\tno-caller\tgemm.rs:1
";
    let mut errors = Vec::new();
    let rows = parse_fixture(text, &mut errors);
    let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "a_gfx12",
            "crates/x/src/y.rs::f",
            "crates/x/src/y.rs::g",
            "h"
        ]
    );
    assert_eq!(rows[0].line, 3);
    assert_eq!(rows[0].category, "non-target-arch");
    assert_eq!(errors.len(), 8, "{errors:#?}");
    for needle in [
        ":4: duplicate",
        ":5: category `bogus-category`",
        ":6: `module` rows cannot use",
        ":7: empty reason",
        ":8: reason must cite",
        ":9: expected 4 tab-separated columns",
        ":10: dynamic name",
        ":14: kind `weird`",
    ] {
        assert!(
            errors.iter().any(|e| e.contains(needle)),
            "missing {needle}: {errors:#?}"
        );
    }
    assert!(has_file_line("x scratch.rs:700-710 y"));
    assert!(has_file_line("see crates/a/b.rs:12."));
    assert!(!has_file_line("gemm.rs: no number"));
    assert!(!has_file_line("a:12"));
}

#[test]
fn wrapper_without_scanned_caller_is_dynamic() {
    let src = r#"
impl Gpu {
    pub fn orphan(&mut self, module: &str) { self.ensure_kernel(module, SRC, "f").unwrap(); }
    pub fn called(&mut self, module: &str) { self.ensure_kernel(module, SRC, "f").unwrap(); }
    fn user(&mut self) { self.called("called_literal"); }
}
"#;
    let s = scan1(src);
    assert_eq!(
        dyn_keys(&s),
        vec![
            "crates/t/src/lib.rs::called".to_string(),
            "crates/t/src/lib.rs::orphan".to_string()
        ]
    );
    let reasons = |f: &str| -> Vec<String> {
        s.sites
            .iter()
            .filter(|x| x.fn_name == f)
            .flat_map(|x| x.dynamic.clone())
            .collect()
    };
    assert!(
        reasons("called").iter().all(|r| r.contains("forwarded")),
        "{:?}",
        reasons("called")
    );
    assert!(reasons("orphan")
        .iter()
        .any(|r| r.contains("no scanned code calls it")));
    assert_eq!(mods(&s), vec!["called_literal".to_string()]);
}

#[test]
fn struct_variant_patterns_resolve_only_from_the_same_variant() {
    let src = r#"
#[derive(Clone, Copy)]
pub enum Mask {
    Embedded { module: &'static str, image: &'static [u8], symbol: &'static str },
    Hipcc { module: &'static str, source: &'static str, symbol: &'static str },
}
const HIP_MODULE: &str = "mask_hip_module";
static A: Table = Table {
    mask: Mask::Hipcc { module: HIP_MODULE, source: SRC, symbol: "s" },
};
static B: Table = Table {
    mask: Mask::Embedded { module: "mask_native_module", image: IMG, symbol: "s" },
};
static C: Table = Table {
    mask: Mask::Hipcc { module: "mask_hip_two", source: SRC, symbol: "s" },
};
impl Gpu {
    fn load(&mut self, arch: &Arch) {
        match arch.mask {
            Mask::Embedded { module, image, symbol } => {
                self.ensure_embedded_kernel(module, image, symbol).unwrap();
            }
            Mask::Hipcc {
                module,
                source,
                symbol,
            } => self.ensure_kernel(module, source, symbol).unwrap(),
        }
    }
    fn load_else(&mut self, arch: &Arch) {
        let Mask::Hipcc { module, source, .. } = arch.mask else { return };
        self.ensure_kernel_public(module, source, "x").unwrap();
    }
    fn skip_embedded(&mut self, arch: &Arch) {
        if let Mask::Embedded { module, .. } = arch.mask {
            self.helper_native(module);
        }
    }
}
"#;
    let s = scan1(src);
    assert_eq!(
        mods(&s),
        vec!["mask_hip_module".to_string(), "mask_hip_two".to_string()],
        "native `Embedded` module must not leak: {:?}",
        s.sites
    );
    assert!(
        dyn_keys(&s).is_empty(),
        "{:?}",
        s.dynamic_sites().keys().collect::<Vec<_>>()
    );
    assert_eq!(s.sites.len(), 2, "{:?}", s.sites);
}

#[test]
fn struct_variant_patterns_follow_typed_parameters_into_struct_tables() {
    let src = r#"
struct Info { one_module: &'static str, vt_module: &'static str, native_module: &'static str }
static QKV: Info = Info { one_module: "one_qkv", vt_module: "vt_qkv", native_module: "nat_qkv" };
static RES: Info = Info { one_module: "one_res", vt_module: "vt_res", native_module: "nat_res" };
enum Image { Hip { module: &'static str, source: &'static str }, Pm }
struct Plan { image: Image }
impl Gpu {
    fn plan(&self, info: &'static Info, wide: bool) -> Plan {
        if wide {
            return Plan { image: Image::Hip { module: info.vt_module, source: SRC_A } };
        }
        Plan { image: Image::Hip { module: info.one_module, source: SRC_B } }
    }
    fn load(&mut self, plan: &Plan) -> HipResult<()> {
        match plan.image {
            Image::Hip { module, source } => self.ensure_kernel(module, source, "f"),
            Image::Pm => self.ensure_embedded_kernel(PM_MODULE, IMG, "f"),
        }
    }
    fn opaque(&mut self, mystery: &Mystery) {
        let p = Plan { image: Image::Hip { module: mystery.name, source: SRC_A } };
        let _ = p;
    }
}
"#;
    let s = scan1(src);
    assert_eq!(
        mods(&s),
        ["one_qkv", "one_res", "vt_qkv", "vt_res"]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>(),
        "{:?}",
        s.sites
    );
    // `Mystery` has no literal, so its field stays unresolved (dynamic), not guessed.
    let load = s.sites.iter().find(|x| x.fn_name == "load").unwrap();
    assert!(
        load.dynamic
            .iter()
            .any(|r| r.contains("depends on parameter")),
        "unresolvable constructor field must surface as dynamic: {:?}",
        load.dynamic
    );
}

#[test]
fn struct_variant_patterns_do_not_count_patterns_as_constructors() {
    let src = r#"
enum E { A { module: &'static str }, B { module: &'static str } }
impl Gpu {
    fn go(&mut self, e: E) {
        match e {
            E::A { module } | E::B { module } => self.ensure_kernel(module, SRC, "f"),
        }
    }
    fn go2(&mut self, e: E) {
        let (E::A { module }, _) = (e, 1) else { return };
        self.ensure_kernel(module, SRC, "g");
    }
}
const X: E = E::A { module: "only_literal" };
"#;
    let s = scan1(src);
    // Shorthand names in patterns are bindings, never constructor fields.
    for site in &s.sites {
        assert!(
            site.mods.iter().all(|m| m == "only_literal"),
            "pattern shorthand leaked as a module: {site:?}"
        );
    }
}

#[test]
fn none_or_empty_source_slot_marks_a_native_descriptor_tuple() {
    let src = r#"
impl Gpu {
    fn none_arm(&mut self, pm: bool, hip: bool) {
        let (module, source, name, rows) = if hip {
            ("hip_wmma_gfx12_x4", Some(SRC_HIP), "n0", 64)
        } else if pm {
            ("pm_x4_gfx1151", None, "n1", 128)
        } else {
            ("hip_wmma_gfx11_bt", Some(SRC_HIP2), "n2", 64)
        };
        match source {
            Some(source) => self.ensure_kernel(module, source, name).unwrap(),
            None => self.ensure_embedded_kernel(module, IMAGE, name).unwrap(),
        }
    }
    fn empty_arm(&mut self, pm: bool, v2c: bool) {
        let (module, source, kernel) = match v2c {
            true if pm => ("iu4_pm_gfx1100", "", "k_pm"),
            true => ("iu4_v2c_gfx11", SRC_V2C, "k_v2c"),
            false => ("iu4_v2b_gfx11", SRC_V2B, "k_v2b"),
        };
        if pm {
            self.ensure_embedded_kernel(module, IMAGE, kernel).unwrap();
        } else {
            self.ensure_kernel(module, source, kernel).unwrap();
        }
    }
    fn optional_source(&mut self, c: bool) {
        // Only sometimes absent, and a 2-tuple: neither is a sentinel.
        let (m3, s3, n3) = ("maybe_src_mod", if c { Some(SRC_A) } else { None }, "n");
        self.ensure_kernel(m3, s3.unwrap(), n3).unwrap();
        let (m2, rows) = ("two_tuple_mod", None);
        self.ensure_kernel(m2, SRC_B, "n").unwrap();
        let _ = rows;
    }
}
"#;
    let s = scan1(src);
    assert_eq!(
        mods(&s),
        [
            "hip_wmma_gfx11_bt",
            "hip_wmma_gfx12_x4",
            "iu4_v2b_gfx11",
            "iu4_v2c_gfx11",
            "maybe_src_mod",
            "two_tuple_mod",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>(),
        "pm sentinel arms must not be static HIP modules: {:?}",
        s.sites
    );
    let native: std::collections::BTreeSet<&str> = s
        .sites
        .iter()
        .flat_map(|x| x.native.iter().map(String::as_str))
        .collect();
    assert!(native.contains("pm_x4_gfx1151"), "{native:?}");
    assert!(native.contains("iu4_pm_gfx1100"), "{native:?}");
}

#[test]
fn only_identifier_strings_are_retained_and_sources_stay_unresolved() {
    let src = r##"
const SRC: &str = r#"
extern "C" __global__ void k(float* x) { x[0] = 1.0f; }
"#;
const MODULE: &str = "ident_module_1";
impl Gpu {
    fn go(&mut self) {
        self.ensure_kernel(MODULE, SRC, "sym");
        let (module, source) = ("tuple_module", SRC);
        self.ensure_kernel(module, source, "sym");
        // A source that is (mis)used as the module must stay dynamic, not vanish.
        self.ensure_kernel(SRC, MODULE, "sym");
        self.ensure_kernel("not an identifier", SRC, "sym");
    }
}
"##;
    let s = scan1(src);
    assert_eq!(
        mods(&s),
        vec!["ident_module_1".to_string(), "tuple_module".to_string()]
    );
    let bad: Vec<_> = s.sites.iter().filter(|x| !x.dynamic.is_empty()).collect();
    assert_eq!(bad.len(), 2, "{:?}", s.sites);
    assert!(bad
        .iter()
        .all(|x| x.dynamic.iter().any(|r| r.contains("non-identifier"))));
    // No site retains the source text, only a bounded excerpt of the argument.
    assert!(
        s.sites
            .iter()
            .all(|x| x.args.iter().all(|a| a.len() <= 400)),
        "{:?}",
        s.sites
    );
}

#[test]
fn self_constructors_resolve_to_their_impl_type_not_every_enum() {
    let src = r#"
enum Image { Hip { module: &'static str, source: &'static str }, Pm }
enum Error { Hip { context: String, code: i32 } }
impl Error {
    fn new(context: String) -> Self { Self::Hip { context, code: 1 } }
}
impl Image {
    fn hip() -> Self { Self::Hip { module: "image_hip_module", source: S } }
}
impl Gpu {
    fn load(&mut self, image: &Image) {
        match image {
            Image::Hip { module, source } => self.ensure_kernel(module, source, "f"),
            Image::Pm => {}
        }
    }
}
"#;
    let s = scan1(src);
    assert_eq!(mods(&s), vec!["image_hip_module".to_string()]);
    assert!(
        dyn_keys(&s).is_empty(),
        "`Error::Hip` literal (no `module` field) must not be a phantom constructor of `Image::Hip`: {:?}",
        s.sites
    );
}

#[test]
fn generic_factory_names_stay_dynamic_and_same_file_calls_win() {
    let mut sources: Vec<(String, String)> = Vec::new();
    for i in 0..8 {
        sources.push((
            format!("crates/t/src/f{i}.rs"),
            format!("impl T{i} {{ pub fn new() -> &'static str {{ \"factory_{i}\" }} }}"),
        ));
    }
    sources.push((
        "crates/t/src/main_user.rs".to_string(),
        r#"
fn pick() -> &'static str { "local_pick" }
impl Gpu {
    fn go(&mut self) {
        let m = Entry::new();
        self.ensure_kernel(m, SRC, "f");
        self.ensure_kernel(pick(), SRC, "f");
    }
}
"#
        .to_string(),
    ));
    sources.push((
        "crates/t/src/other.rs".to_string(),
        "fn pick() -> &'static str { \"other_pick\" }".to_string(),
    ));
    let s = scanner::scan_sources(sources);
    assert_eq!(mods(&s), vec!["local_pick".to_string()], "{:?}", s.sites);
    let new_site = s.sites.iter().find(|x| x.expr == "m").unwrap();
    assert!(
        new_site
            .dynamic
            .iter()
            .any(|r| r.contains("ambiguous call new")),
        "{:?}",
        new_site.dynamic
    );
}
