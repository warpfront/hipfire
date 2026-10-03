//! Macro elaboration. The evaluator knows no macro: every supported macro is
//! rewritten once, after parsing, into ordinary calls and expressions
//! (`format!` -> `__fmt(..)`, `vec!` -> array, `matches!` -> `match`, ...).
//! A macro it cannot elaborate becomes `__unsupported("..")`, which fails
//! only if it is executed.

use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use syn::parse::{Parse, ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::visit_mut::{self, VisitMut};
use syn::{Expr, Pat, Stmt, Token};

pub fn expand_file(file: &mut syn::File) {
    Expander.visit_file_mut(file);
}

struct Expander;

impl VisitMut for Expander {
    fn visit_stmt_mut(&mut self, s: &mut Stmt) {
        if let Stmt::Macro(sm) = s {
            let semi = sm.semi_token;
            let e = Expr::Macro(syn::ExprMacro { attrs: std::mem::take(&mut sm.attrs), mac: sm.mac.clone() });
            *s = Stmt::Expr(e, semi);
        }
        visit_mut::visit_stmt_mut(self, s);
    }
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        if let Expr::Macro(m) = e {
            *e = elaborate(&m.mac);
        }
        visit_mut::visit_expr_mut(self, e);
    }
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
    let f: Expr = syn::parse_str(name).expect("internal name");
    Expr::Call(syn::ExprCall { attrs: vec![], func: Box::new(f), paren_token: Default::default(), args: args.into_iter().collect() })
}

fn unsupported(msg: String) -> Expr {
    let lit = syn::LitStr::new(&msg, proc_macro2::Span::call_site());
    call("__unsupported", vec![syn::parse_quote!(#lit)])
}

fn args_of(tokens: TokenStream) -> syn::Result<Vec<Expr>> {
    Punctuated::<Expr, Token![,]>::parse_terminated.parse2(tokens).map(|p| p.into_iter().collect())
}

struct MatchesArgs {
    expr: Expr,
    pat: Pat,
    guard: Option<Expr>,
}
impl Parse for MatchesArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let expr: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let pat = Pat::parse_multi_with_leading_vert(input)?;
        let guard = if input.peek(Token![if]) {
            input.parse::<Token![if]>()?;
            Some(input.parse()?)
        } else {
            None
        };
        let _ = input.parse::<Option<Token![,]>>()?;
        Ok(Self { expr, pat, guard })
    }
}

fn elaborate(mac: &syn::Macro) -> Expr {
    let name = mac.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
    let toks = mac.tokens.clone();
    let bad = |e: syn::Error| {
        let s = e.span().start();
        unsupported(format!("{name}!: {}:{}: {e}", s.line, s.column + 1))
    };
    match name.as_str() {
        "format" | "format_args" => match args_of(toks) {
            Ok(a) if !a.is_empty() => call("__fmt", a),
            Ok(_) => unsupported("format! needs a format string".into()),
            Err(e) => bad(e),
        },
        "println" | "eprintln" | "print" | "eprint" => match args_of(toks) {
            Ok(a) => call(&format!("__{name}"), a),
            Err(e) => bad(e),
        },
        "panic" | "unreachable" | "todo" | "unimplemented" => match args_of(toks) {
            Ok(a) => call("__panic", if a.is_empty() { vec![syn::parse_quote!("")] } else { vec![call("__fmt", a)] }),
            Err(e) => bad(e),
        },
        "assert" | "debug_assert" => match args_of(toks) {
            Ok(mut a) if !a.is_empty() => {
                let cond = a.remove(0);
                if a.is_empty() {
                    call("__assert", vec![cond])
                } else {
                    call("__assert", vec![cond, call("__fmt", a)])
                }
            }
            Ok(_) => unsupported("assert! needs a condition".into()),
            Err(e) => bad(e),
        },
        "assert_eq" | "assert_ne" | "debug_assert_eq" | "debug_assert_ne" => match args_of(toks) {
            Ok(mut a) if a.len() >= 2 => {
                let ne = name.ends_with("_ne");
                let (l, r) = (a.remove(0), a.remove(0));
                let mut v = vec![l, r];
                if !a.is_empty() {
                    v.push(call("__fmt", a));
                }
                call(if ne { "__assert_ne" } else { "__assert_eq" }, v)
            }
            Ok(_) => unsupported("assert_eq! needs two operands".into()),
            Err(e) => bad(e),
        },
        "vec" => {
            let bracketed: TokenStream = std::iter::once(TokenTree::Group(Group::new(Delimiter::Bracket, toks))).collect();
            match syn::parse2::<Expr>(bracketed) {
                Ok(e) => e,
                Err(e) => bad(e),
            }
        }
        "matches" => match syn::parse2::<MatchesArgs>(toks) {
            Ok(MatchesArgs { expr, pat, guard }) => match guard {
                Some(g) => syn::parse_quote!(match #expr { #pat if #g => true, _ => false }),
                None => syn::parse_quote!(match #expr { #pat => true, _ => false }),
            },
            Err(e) => bad(e),
        },
        "stringify" => {
            let lit = syn::LitStr::new(&toks.to_string(), proc_macro2::Span::call_site());
            syn::parse_quote!(#lit)
        }
        other => unsupported(format!("macro `{other}!` is not supported by the .rip evaluator")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::ToTokens;

    fn expanded(src: &str) -> String {
        let mut e: Expr = syn::parse_str(src).unwrap();
        Expander.visit_expr_mut(&mut e);
        e.to_token_stream().to_string()
    }

    #[test]
    fn format_and_vec_become_calls_and_arrays() {
        assert_eq!(expanded("format!(\"{}\", x)"), "__fmt (\"{}\" , x)");
        assert_eq!(expanded("vec![1, 2]"), "[1 , 2]");
        assert_eq!(expanded("vec![0; 3]"), "[0 ; 3]");
    }

    #[test]
    fn matches_becomes_a_match_and_keeps_guards() {
        let m = expanded("matches!(a, A::X | A::Y)");
        assert!(m.starts_with("match a"), "{m}");
        assert!(m.contains("A :: X | A :: Y => true"), "{m}");
        assert!(expanded("matches!(a, Some(n) if n > 1)").contains("if n > 1 => true"));
    }

    #[test]
    fn unknown_macros_fail_only_when_executed() {
        assert!(expanded("weird!(x)").starts_with("__unsupported"));
    }
}
