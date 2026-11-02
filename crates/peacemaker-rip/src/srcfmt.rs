//! `.rip` source formatter: rustfmt over a masked copy of the source.
//!
//! The item keywords (`tile`, `chain`, `region`) are not Rust, so they are masked by byte offset
//! (found exactly as [`crate::preprocess`] finds them) into attribute-marked Rust items, formatted,
//! and restored. The result is accepted only if its token stream equals the input's.

use crate::preprocess::lex;
use proc_macro2::{Delimiter, TokenStream, TokenTree};
use std::io::Write;
use std::process::{Command, Stdio};

const TILE_ATTR: &str = "#[rip_tile]";
const CHAIN_ATTR: &str = "#[rip_chain]";
const REGION_MACRO: &str = "__rip_region!(";
const PUB_REGION_MACRO: &str = "__rip_pub_region!(";
/// Layout options only: no option that adds, drops or reorders tokens.
const RUSTFMT_CONFIG: &str = "max_width=120,trailing_semicolon=false,reorder_imports=false,reorder_modules=false,remove_nested_parens=false";
const MASK_NAMES: [&str; 4] = ["rip_tile", "rip_chain", "__rip_region", "__rip_pub_region"];

/// Format a `.rip` source with rustfmt (item keywords masked and restored); verifies that the token
/// stream is unchanged, else Err. Err also when rustfmt is not installed.
pub fn format_source(source: &str) -> Result<String, String> {
    let masked = mask(source)?;
    let formatted = run_rustfmt(&masked)?;
    let output = unmask(&formatted)?;
    verify_tokens(source, &output)?;
    Ok(output)
}

/// Replace the item keywords of `source` with rustfmt-parsable Rust:
/// `tile N {` -> `#[rip_tile] struct N {`, `chain f(` -> `#[rip_chain] fn f(`,
/// `region A, B;` -> `__rip_region!(A, B);` (`pub region` -> `__rip_pub_region!(`).
pub fn mask(source: &str) -> Result<String, String> {
    if let Some(name) = MASK_NAMES.iter().find(|n| source.contains(**n)) {
        return Err(format!(
            "source uses the reserved formatter marker `{name}`"
        ));
    }
    let toks: Vec<TokenTree> = lex(source)?.into_iter().collect();
    let lines = LineIndex::new(source);
    let mut edits = Vec::new();
    scan(&toks, true, &lines, &mut edits);
    edits.sort_by_key(|e| (e.start, e.end));
    let mut out = String::with_capacity(source.len() + edits.len() * 16);
    let mut at = 0;
    for e in &edits {
        out.push_str(&source[at..e.start]);
        out.push_str(&e.text);
        at = e.end;
    }
    out.push_str(&source[at..]);
    Ok(out)
}

/// Inverse of [`mask`] on rustfmt output (the attribute may have moved onto its own line).
pub fn unmask(text: &str) -> Result<String, String> {
    let text = unmask_items(text, TILE_ATTR, "struct", "tile")?;
    let text = unmask_items(&text, CHAIN_ATTR, "fn", "chain")?;
    let text = unmask_regions(&text, PUB_REGION_MACRO, "pub region ")?;
    unmask_regions(&text, REGION_MACRO, "region ")
}

struct Edit {
    start: usize,
    end: usize,
    text: String,
}

struct LineIndex<'a> {
    source: &'a str,
    starts: Vec<usize>,
}

impl<'a> LineIndex<'a> {
    fn new(source: &'a str) -> Self {
        let mut starts = vec![0];
        starts.extend(source.match_indices('\n').map(|(i, _)| i + 1));
        LineIndex { source, starts }
    }
    /// Byte offset of a proc_macro2 position (1-based line, 0-based char column).
    fn offset(&self, line: usize, column: usize) -> usize {
        let start = self.starts[line - 1];
        let rest = &self.source[start..];
        start
            + rest
                .char_indices()
                .nth(column)
                .map_or(rest.len(), |(b, _)| b)
    }
    fn start(&self, t: &TokenTree) -> usize {
        let p = t.span().start();
        self.offset(p.line, p.column)
    }
    fn end(&self, t: &TokenTree) -> usize {
        let p = t.span().end();
        self.offset(p.line, p.column)
    }
}

fn is_ident(t: &TokenTree, name: &str) -> bool {
    matches!(t, TokenTree::Ident(i) if i == name)
}
fn is_punct(t: &TokenTree, c: char) -> bool {
    matches!(t, TokenTree::Punct(p) if p.as_char() == c)
}
fn group_delim(t: &TokenTree, d: Delimiter) -> bool {
    matches!(t, TokenTree::Group(g) if g.delimiter() == d)
}

/// Same item walk as `preprocess::items`: only the first token of an item (after attributes and
/// visibility) is a keyword; `chain` also inside `impl`/`trait`/`mod` bodies.
fn scan(toks: &[TokenTree], top: bool, lines: &LineIndex, edits: &mut Vec<Edit>) {
    let mut i = 0;
    while i < toks.len() {
        let mut j = i;
        loop {
            if j < toks.len() && is_punct(&toks[j], '#') {
                j += 1;
                if j < toks.len() && is_punct(&toks[j], '!') {
                    j += 1;
                }
                if j < toks.len() && group_delim(&toks[j], Delimiter::Bracket) {
                    j += 1;
                    continue;
                }
            }
            break;
        }
        let mut pub_at = None;
        let mut pub_plain = false;
        if j < toks.len() && is_ident(&toks[j], "pub") {
            pub_at = Some(j);
            j += 1;
            if j < toks.len() && group_delim(&toks[j], Delimiter::Parenthesis) {
                j += 1;
            } else {
                pub_plain = true;
            }
        }
        let Some(kw) = toks.get(j) else {
            break;
        };
        let kw_name = match kw {
            TokenTree::Ident(id) => id.to_string(),
            _ => String::new(),
        };
        if kw_name == "region" && top {
            if let Some(end) = (j + 1..toks.len()).find(|&k| is_punct(&toks[k], ';')) {
                let names = &toks[j + 1..end];
                let valid = names.iter().any(|t| matches!(t, TokenTree::Ident(_)))
                    && names
                        .iter()
                        .all(|t| matches!(t, TokenTree::Ident(_)) || is_punct(t, ','));
                if valid {
                    let masked = match pub_at {
                        None => Some((lines.start(kw), REGION_MACRO)),
                        Some(p) if pub_plain => Some((lines.start(&toks[p]), PUB_REGION_MACRO)),
                        Some(_) => None,
                    };
                    if let Some((start, text)) = masked {
                        edits.push(Edit {
                            start,
                            end: lines.start(&names[0]),
                            text: text.into(),
                        });
                        let semi = &toks[end];
                        edits.push(Edit {
                            start: lines.end(&names[names.len() - 1]),
                            end: lines.end(semi),
                            text: ");".into(),
                        });
                        i = end + 1;
                        continue;
                    }
                }
            }
        }
        let semi_item = matches!(
            kw_name.as_str(),
            "const" | "static" | "type" | "use" | "extern"
        ) && !(kw_name == "const"
            && toks.get(j + 1).is_some_and(|t| {
                is_ident(t, "fn") || is_ident(t, "unsafe") || is_ident(t, "async")
            }))
            && !(kw_name == "extern"
                && toks
                    .get(j + 1)
                    .is_some_and(|t| matches!(t, TokenTree::Literal(_)) || is_ident(t, "fn")));
        let mut end = j;
        let mut closed_by_brace = false;
        while end < toks.len() {
            if is_punct(&toks[end], ';') {
                break;
            }
            if !semi_item && group_delim(&toks[end], Delimiter::Brace) {
                closed_by_brace = true;
                break;
            }
            end += 1;
        }
        let last = end.min(toks.len() - 1);
        let keyword = match kw_name.as_str() {
            "tile" if top => Some((TILE_ATTR, "struct")),
            "chain" => Some((CHAIN_ATTR, "fn")),
            _ => None,
        };
        if let Some((attr, rust)) = keyword {
            let kw_start = lines.start(kw);
            let attr_at = pub_at.map_or(kw_start, |p| lines.start(&toks[p]));
            if attr_at == kw_start {
                edits.push(Edit {
                    start: kw_start,
                    end: lines.end(kw),
                    text: format!("{attr} {rust}"),
                });
            } else {
                edits.push(Edit {
                    start: attr_at,
                    end: attr_at,
                    text: format!("{attr} "),
                });
                edits.push(Edit {
                    start: kw_start,
                    end: lines.end(kw),
                    text: rust.into(),
                });
            }
        }
        let recurse =
            closed_by_brace && matches!(kw_name.as_str(), "impl" | "trait" | "mod" | "unsafe");
        let is_container = recurse
            && (kw_name != "unsafe"
                || toks[j..=last]
                    .iter()
                    .any(|t| is_ident(t, "impl") || is_ident(t, "trait")));
        if is_container {
            if let TokenTree::Group(g) = &toks[last] {
                let inner: Vec<TokenTree> = g.stream().into_iter().collect();
                scan(&inner, false, lines, edits);
            }
        }
        i = last + 1;
    }
}

fn skip_ws(text: &str, mut at: usize) -> usize {
    while let Some(c) = text[at..].chars().next().filter(|c| c.is_whitespace()) {
        at += c.len_utf8();
    }
    at
}

/// `#[attr]<ws>[pub[(..)]<ws>]<rust>` -> `[pub[(..)]<ws>]<word>` for every occurrence.
fn unmask_items(text: &str, attr: &str, rust: &str, word: &str) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(p) = rest.find(attr) {
        out.push_str(&rest[..p]);
        let after = &rest[p + attr.len()..];
        let body = skip_ws(after, 0);
        let mut k = body;
        if after[k..].starts_with("pub") {
            k += 3;
            let w = skip_ws(after, k);
            if after[w..].starts_with('(') {
                let mut depth = 0usize;
                let close = after[w..]
                    .char_indices()
                    .find_map(|(b, c)| {
                        match c {
                            '(' => depth += 1,
                            ')' => depth -= 1,
                            _ => {}
                        }
                        (depth == 0).then_some(w + b + 1)
                    })
                    .ok_or_else(|| {
                        format!("unbalanced visibility after `{attr}` in formatted output")
                    })?;
                k = close;
            }
            k = skip_ws(after, k);
        }
        let tail = after[k..]
            .strip_prefix(rust)
            .filter(|t| t.chars().next().is_none_or(|c| c.is_whitespace()))
            .ok_or_else(|| format!("rustfmt moved `{attr}` away from its `{rust}` keyword"))?;
        out.push_str(&after[body..k]);
        out.push_str(word);
        rest = tail;
    }
    out.push_str(rest);
    Ok(out)
}

/// `__rip_region!(A, B);` -> `region A, B;` (and the `pub` form).
fn unmask_regions(text: &str, marker: &str, prefix: &str) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(p) = rest.find(marker) {
        out.push_str(&rest[..p]);
        let after = &rest[p + marker.len()..];
        let close = after
            .find(')')
            .ok_or("unterminated region marker in formatted output")?;
        let names: Vec<&str> = after[..close]
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .collect();
        let semi = skip_ws(after, close + 1);
        if !after[semi..].starts_with(';') {
            return Err("rustfmt changed a region declaration".into());
        }
        out.push_str(prefix);
        out.push_str(&names.join(", "));
        out.push(';');
        rest = &after[semi + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn run_rustfmt(masked: &str) -> Result<String, String> {
    let mut child = Command::new("rustfmt")
        .args([
            "--edition",
            "2021",
            "--emit",
            "stdout",
            "--config",
            RUSTFMT_CONFIG,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                "rustfmt not found on PATH: install it with `rustup component add rustfmt`"
                    .to_string()
            } else {
                format!("cannot run rustfmt: {e}")
            }
        })?;
    let mut stdin = child.stdin.take().ok_or("cannot open rustfmt stdin")?;
    let input = masked.to_owned();
    let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
    let output = child
        .wait_with_output()
        .map_err(|e| format!("rustfmt failed: {e}"))?;
    writer
        .join()
        .map_err(|_| "rustfmt stdin writer panicked".to_string())?
        .map_err(|e| format!("cannot write to rustfmt: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "rustfmt failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("rustfmt output is not UTF-8: {e}"))
}

fn is_fat_arrow(toks: &[TokenTree], k: usize) -> bool {
    k >= 1 && is_punct(&toks[k], '>') && is_punct(&toks[k - 1], '=')
}

/// Flattened token texts, ignoring spacing and the tokens rustfmt adds, drops or moves for layout
/// alone: a comma before a closing delimiter, the commas between match arms, and the braces around
/// a closure or match-arm body (`|x| { e }` / `A => { e }`).
fn flatten(ts: TokenStream, in_group: bool, out: &mut Vec<String>) {
    let mut toks: Vec<TokenTree> = ts.into_iter().collect();
    if in_group && toks.last().is_some_and(|t| is_punct(t, ',')) {
        toks.pop();
    }
    let has_arms = (0..toks.len()).any(|k| is_fat_arrow(&toks, k));
    for k in 0..toks.len() {
        match &toks[k] {
            TokenTree::Group(g) => {
                let (open, close) = match g.delimiter() {
                    Delimiter::Parenthesis => ("(", ")"),
                    Delimiter::Brace => ("{", "}"),
                    Delimiter::Bracket => ("[", "]"),
                    Delimiter::None => ("", ""),
                };
                let wrapper = g.delimiter() == Delimiter::Brace
                    && k > 0
                    && (is_punct(&toks[k - 1], '|') || is_fat_arrow(&toks, k - 1));
                if !wrapper {
                    out.push(open.into());
                }
                flatten(g.stream(), true, out);
                if !wrapper {
                    out.push(close.into());
                }
            }
            t if has_arms && is_punct(t, ',') => {}
            other => out.push(other.to_string()),
        }
    }
}

fn tokens(source: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    flatten(lex(source)?, false, &mut out);
    Ok(out)
}

fn verify_tokens(input: &str, output: &str) -> Result<(), String> {
    let before = tokens(input)?;
    let after = tokens(output).map_err(|e| format!("formatted output does not lex: {e}"))?;
    let at = before.iter().zip(&after).position(|(a, b)| a != b);
    match at {
        Some(k) => Err(format!(
            "formatting changed the token stream at token {} (input `{}`, output `{}`) after `{}`; refusing to rewrite",
            k + 1,
            before[k],
            after[k],
            before[k.saturating_sub(8)..k].join(" ")
        )),
        None if before.len() != after.len() => {
            let k = before.len().min(after.len());
            let (side, tok) = if before.len() > k { ("missing from the output", &before[k]) } else { ("added by formatting", &after[k]) };
            Err(format!("formatting changed the token stream at token {}: `{tok}` {side}; refusing to rewrite", k + 1))
        }
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_and_unmask_are_inverse() {
        let src = "/// doc\npub tile T { a: u8 }\nchain f(x: u32) { x }\nregion A, B;\nimpl X { chain g() {} }\n";
        let masked = mask(src).unwrap();
        assert!(masked.contains("#[rip_tile] pub struct T"));
        assert!(masked.contains("#[rip_chain] fn f("));
        assert!(masked.contains("__rip_region!(A, B);"));
        assert_eq!(unmask(&masked).unwrap(), src);
    }

    #[test]
    fn verification_ignores_layout_tokens_only() {
        verify_tokens(
            "fn f(a: u8, b: u8) { g(a, b) }",
            "fn f(\n a: u8,\n b: u8,\n) {\n g(a, b,)\n}",
        )
        .unwrap();
        verify_tokens(
            "fn f() { let g = |x| x + 1; }",
            "fn f() { let g = |x| { x + 1 }; }",
        )
        .unwrap();
        let err = verify_tokens("fn f() { a + b }", "fn f() { a - b }").unwrap_err();
        assert!(err.contains("`+`") && err.contains("`-`"), "{err}");
        assert!(verify_tokens("fn f() { a; b }", "fn f() { a; }").is_err());
    }
}
