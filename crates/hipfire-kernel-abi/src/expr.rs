// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Checked evaluation of [`Expr`] over launch params and export constants.

use crate::{ExportV1, Expr, KernelAbiV1};

/// Evaluate `expr` for `export`. `params` are in [`KernelAbiV1::params`]
/// order; a `Param(name)` resolves to the declared param of that name first
/// and to an [`ExportV1::consts`] entry otherwise. Overflow, division by
/// zero, an inexact [`Expr::DivExact`], a negative [`Expr::Sub`] and an
/// unknown name are errors; nothing wraps.
pub fn eval(
    expr: &Expr,
    abi: &KernelAbiV1,
    export: &ExportV1,
    params: &[u64],
) -> Result<u64, String> {
    match expr {
        Expr::Const(v) => Ok(*v),
        Expr::Param(name) => lookup(name, abi, export, params),
        Expr::Add(terms) => {
            let mut acc = 0u64;
            for t in terms.iter() {
                let v = eval(t, abi, export, params)?;
                acc = acc
                    .checked_add(v)
                    .ok_or_else(|| format!("overflow: {acc} + {v}"))?;
            }
            Ok(acc)
        }
        Expr::Mul(factors) => {
            let mut acc = 1u64;
            for t in factors.iter() {
                let v = eval(t, abi, export, params)?;
                acc = acc
                    .checked_mul(v)
                    .ok_or_else(|| format!("overflow: {acc} * {v}"))?;
            }
            Ok(acc)
        }
        Expr::Sub(a, b) => {
            let a = eval(a, abi, export, params)?;
            let b = eval(b, abi, export, params)?;
            a.checked_sub(b)
                .ok_or_else(|| format!("negative result: {a} - {b}"))
        }
        Expr::DivExact(a, d) => {
            let a = eval(a, abi, export, params)?;
            if *d == 0 {
                return Err(format!("division by zero: {a} / 0"));
            }
            if a % d != 0 {
                return Err(format!("inexact division: {a} is not a multiple of {d}"));
            }
            Ok(a / d)
        }
        Expr::DivCeil(a, d) => {
            let a = eval(a, abi, export, params)?;
            if *d == 0 {
                return Err(format!("division by zero: ceil({a} / 0)"));
            }
            Ok(a / d + u64::from(a % d != 0))
        }
        Expr::Div(a, d) => {
            let a = eval(a, abi, export, params)?;
            if *d == 0 {
                return Err(format!("division by zero: {a} / 0"));
            }
            Ok(a / d)
        }
    }
}

fn lookup(name: &str, abi: &KernelAbiV1, export: &ExportV1, params: &[u64]) -> Result<u64, String> {
    if let Some(i) = abi.param_index(name) {
        return params.get(i).copied().ok_or_else(|| {
            format!(
                "param {name} (index {i}) missing: {} values supplied",
                params.len()
            )
        });
    }
    export
        .consts
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            format!(
                "unknown name {name}: neither a param of {} nor a constant of {}",
                abi.id, export.symbol
            )
        })
}

/// Pre-order visit of every node of `expr`.
pub(crate) fn visit(expr: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(expr);
    match expr {
        Expr::Const(_) | Expr::Param(_) => {}
        Expr::Add(terms) | Expr::Mul(terms) => {
            for t in terms.iter() {
                visit(t, f);
            }
        }
        Expr::Sub(a, b) => {
            visit(a, f);
            visit(b, f);
        }
        Expr::DivExact(a, _) | Expr::DivCeil(a, _) | Expr::Div(a, _) => visit(a, f),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::toy::TOY;

    fn ev(e: &Expr, params: &[u64]) -> Result<u64, String> {
        eval(e, &TOY, &TOY.exports[0], params)
    }

    #[test]
    fn const_and_arithmetic() {
        assert_eq!(ev(&Expr::Const(7), &[]), Ok(7));
        let add = Expr::Add(&[Expr::Const(2), Expr::Const(3), Expr::Const(4)]);
        assert_eq!(ev(&add, &[]), Ok(9));
        assert_eq!(ev(&Expr::Add(&[]), &[]), Ok(0));
        let mul = Expr::Mul(&[Expr::Const(2), Expr::Const(3), Expr::Const(4)]);
        assert_eq!(ev(&mul, &[]), Ok(24));
        assert_eq!(ev(&Expr::Mul(&[]), &[]), Ok(1));
        let sub = Expr::Sub(&Expr::Const(9), &Expr::Const(4));
        assert_eq!(ev(&sub, &[]), Ok(5));
    }

    #[test]
    fn division_flavours() {
        const A: &Expr = &Expr::Const(10);
        assert_eq!(ev(&Expr::DivExact(A, 5), &[]), Ok(2));
        assert_eq!(ev(&Expr::DivCeil(A, 4), &[]), Ok(3));
        assert_eq!(ev(&Expr::DivCeil(A, 5), &[]), Ok(2));
        assert_eq!(ev(&Expr::Div(A, 4), &[]), Ok(2));
    }

    #[test]
    fn overflow_is_an_error() {
        let add = Expr::Add(&[Expr::Const(u64::MAX), Expr::Const(1)]);
        assert!(ev(&add, &[]).unwrap_err().contains("overflow"));
        let mul = Expr::Mul(&[Expr::Const(u64::MAX), Expr::Const(2)]);
        assert!(ev(&mul, &[]).unwrap_err().contains("overflow"));
    }

    #[test]
    fn division_errors() {
        const A: &Expr = &Expr::Const(10);
        assert!(ev(&Expr::DivExact(A, 0), &[])
            .unwrap_err()
            .contains("division by zero"));
        assert!(ev(&Expr::DivCeil(A, 0), &[])
            .unwrap_err()
            .contains("division by zero"));
        assert!(ev(&Expr::Div(A, 0), &[])
            .unwrap_err()
            .contains("division by zero"));
        assert!(ev(&Expr::DivExact(A, 4), &[])
            .unwrap_err()
            .contains("inexact"));
    }

    #[test]
    fn negative_sub_is_an_error() {
        let sub = Expr::Sub(&Expr::Const(1), &Expr::Const(2));
        assert!(ev(&sub, &[]).unwrap_err().contains("negative"));
    }

    #[test]
    fn param_then_const_lookup() {
        // TOY params are M, S, K; WIDTH is an export constant.
        assert_eq!(ev(&Expr::Param("S"), &[1, 2, 3]), Ok(2));
        assert_eq!(ev(&Expr::Param("WIDTH"), &[1, 2, 3]), Ok(1));
        let e2 = &TOY.exports[1];
        assert_eq!(eval(&Expr::Param("WIDTH"), &TOY, e2, &[1, 2, 3]), Ok(2));
    }

    #[test]
    fn unknown_name_and_short_params_are_errors() {
        assert!(ev(&Expr::Param("NOPE"), &[1, 2, 3])
            .unwrap_err()
            .contains("NOPE"));
        assert!(ev(&Expr::Param("K"), &[1, 2])
            .unwrap_err()
            .contains("missing"));
    }

    #[test]
    fn visit_reaches_every_node() {
        let e = Expr::DivCeil(
            &Expr::Add(&[
                Expr::Param("M"),
                Expr::Mul(&[Expr::Const(2), Expr::Param("K")]),
            ]),
            4,
        );
        let mut params = Vec::new();
        let mut nodes = 0;
        visit(&e, &mut |n| {
            nodes += 1;
            if let Expr::Param(p) = n {
                params.push(*p);
            }
        });
        assert_eq!(params, ["M", "K"]);
        assert_eq!(nodes, 6);
    }
}
