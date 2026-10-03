//! Token-level pre-pass of the `.rip` surface syntax.
//!
//! A `.rip` file is Rust items plus three item-level keywords that make the
//! kernel vocabulary first-class at the parser boundary:
//!
//! - `tile Name { .. }`  is a compile-time register-tile layout (a `struct`);
//! - `chain name(..) { .. }` is an instruction chain / emission routine (a `fn`);
//! - `region Name;` (or `region A, B;`) declares an LDS region tag (an `enum`).
//!
//! The rewrite works on the token stream, never on text, so identifiers,
//! strings and comments that contain these words are untouched, and every
//! token keeps its original span (error positions point into the source).
//! Only the first token of an item is considered, so `tile`/`chain` stay
//! ordinary identifiers inside bodies (`let tile = ..`, `.chain(..)`).

use proc_macro2::{Delimiter, Group, Ident, TokenStream, TokenTree};

pub fn lex(source: &str) -> Result<TokenStream, String> {
    source.parse::<TokenStream>().map_err(|e| {
        let s = e.span().start();
        format!("{}:{}: {e}", s.line, s.column + 1)
    })
}

/// Rewrite `tile`/`chain`/`region` items of `ts` (recursing into `impl`,
/// `trait` and `mod` bodies for `chain`).
pub fn rewrite(ts: TokenStream) -> TokenStream {
    items(ts.into_iter().collect(), true)
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

fn items(toks: Vec<TokenTree>, top: bool) -> TokenStream {
    let mut out: Vec<TokenTree> = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        // Attributes and visibility belong to the item.
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
        if j < toks.len() && is_ident(&toks[j], "pub") {
            j += 1;
            if j < toks.len() && group_delim(&toks[j], Delimiter::Parenthesis) {
                j += 1;
            }
        }
        out.extend(toks[i..j].iter().cloned());
        let Some(kw) = toks.get(j) else {
            i = j;
            break;
        };
        let kw_name = match kw {
            TokenTree::Ident(id) => id.to_string(),
            _ => String::new(),
        };
        // `region A, B;`
        if kw_name == "region" && top {
            if let Some(end) = (j + 1..toks.len()).find(|&k| is_punct(&toks[k], ';')) {
                let names: Vec<&TokenTree> = toks[j + 1..end].iter().filter(|t| !is_punct(t, ',')).collect();
                if !names.is_empty() && names.iter().all(|t| matches!(t, TokenTree::Ident(_))) && toks[j + 1..end].iter().all(|t| matches!(t, TokenTree::Ident(_)) || is_punct(t, ',')) {
                    for n in names {
                        out.push(TokenTree::Ident(Ident::new("enum", kw.span())));
                        out.push(n.clone());
                        out.push(TokenTree::Group(Group::new(Delimiter::Brace, TokenStream::new())));
                    }
                    i = end + 1;
                    continue;
                }
            }
        }
        // Item end.
        let semi_item = matches!(kw_name.as_str(), "const" | "static" | "type" | "use" | "extern")
            && !(kw_name == "const" && toks.get(j + 1).is_some_and(|t| is_ident(t, "fn") || is_ident(t, "unsafe") || is_ident(t, "async")))
            && !(kw_name == "extern" && toks.get(j + 1).is_some_and(|t| matches!(t, TokenTree::Literal(_)) || is_ident(t, "fn")));
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
        // Keyword rewrite.
        let mut body_start = j;
        match kw_name.as_str() {
            "tile" if top => {
                out.push(TokenTree::Ident(Ident::new("struct", kw.span())));
                body_start = j + 1;
            }
            "chain" => {
                out.push(TokenTree::Ident(Ident::new("fn", kw.span())));
                body_start = j + 1;
            }
            _ => {}
        }
        // Items with a nested item list.
        let recurse = closed_by_brace && matches!(kw_name.as_str(), "impl" | "trait" | "mod" | "unsafe");
        let is_container = recurse && (kw_name != "unsafe" || toks[j..=last].iter().any(|t| is_ident(t, "impl") || is_ident(t, "trait")));
        let stop = if is_container { last } else { (last + 1).min(toks.len()) };
        out.extend(toks[body_start..stop].iter().cloned());
        if is_container {
            if let TokenTree::Group(g) = &toks[last] {
                let inner = items(g.stream().into_iter().collect(), false);
                let mut ng = Group::new(g.delimiter(), inner);
                ng.set_span(g.span());
                out.push(TokenTree::Group(ng));
            }
        }
        i = last + 1;
    }
    out.extend(toks[i.min(toks.len())..].iter().cloned());
    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(s: &str) -> String {
        rewrite(lex(s).unwrap()).to_string()
    }
    fn plain(s: &str) -> String {
        lex(s).unwrap().to_string()
    }

    #[test]
    fn tile_chain_region_become_rust_items() {
        assert_eq!(norm("tile T { a: u8 }"), plain("struct T { a: u8 }"));
        assert_eq!(norm("pub chain f(x: u8) -> u8 { x }"), plain("pub fn f(x: u8) -> u8 { x }"));
        assert_eq!(norm("region A;"), plain("enum A {}"));
        assert_eq!(norm("region A, B;"), plain("enum A {} enum B {}"));
    }

    #[test]
    fn keywords_are_only_rewritten_at_item_start() {
        // Inside a body, as a method name, and in strings they stay as written.
        let src = "fn f() { let tile = 1; let chain = tile; x.chain(y); \"tile T {}\"; }";
        assert_eq!(norm(src), plain(src));
    }

    #[test]
    fn attributes_and_docs_stay_attached() {
        assert_eq!(norm("/// doc\n#[derive(Clone)] pub tile T { a: u8 }"), plain("/// doc\n#[derive(Clone)] pub struct T { a: u8 }"));
    }

    #[test]
    fn a_const_block_does_not_end_the_item_early() {
        // The braces in a const initializer must not be taken for the item's end.
        let src = "const A: u8 = { 1 }; tile T { a: u8 } chain g() {}";
        assert_eq!(norm(src), plain("const A: u8 = { 1 }; struct T { a: u8 } fn g() {}"));
    }

    #[test]
    fn chain_inside_impl_is_a_method() {
        assert_eq!(norm("impl G { chain f(&self) {} fn h(&self) {} }"), plain("impl G { fn f(&self) {} fn h(&self) {} }"));
        // `tile` is a top-level item only.
        assert_eq!(norm("impl G { tile f(&self) {} }"), plain("impl G { tile f(&self) {} }"));
    }
}
