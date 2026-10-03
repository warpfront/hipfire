//! Operators, casts and declared-type coercion on values.

use crate::value::{values_eq, IntTy, Ty, Value};
use syn::BinOp;

/// Unify the types of two integer operands.
fn unify(a: Option<IntTy>, b: Option<IntTy>, op: &str) -> Result<Option<IntTy>, String> {
    match (a, b) {
        (Some(x), Some(y)) if x != y => Err(format!("mismatched integer types in `{op}`: {} and {}", x.name(), y.name())),
        (Some(x), _) | (_, Some(x)) => Ok(Some(x)),
        _ => Ok(None),
    }
}

fn check<'a>(v: i128, t: Option<IntTy>, what: &str) -> Result<Value<'a>, String> {
    match t {
        Some(t) if !t.fits(v) => Err(format!("attempt to {what} with overflow: {v} does not fit {}", t.name())),
        _ => Ok(Value::Int(v, t)),
    }
}

pub fn binop<'a>(op: &BinOp, l: &Value<'a>, r: &Value<'a>) -> Result<Value<'a>, String> {
    use BinOp::*;
    // Comparisons and equality work on every comparable pair.
    match op {
        Eq(_) => return Ok(Value::Bool(values_eq(l, r))),
        Ne(_) => return Ok(Value::Bool(!values_eq(l, r))),
        Lt(_) | Le(_) | Gt(_) | Ge(_) => {
            let ord = compare(l, r)?;
            return Ok(Value::Bool(match op {
                Lt(_) => ord.is_lt(),
                Le(_) => ord.is_le(),
                Gt(_) => ord.is_gt(),
                _ => ord.is_ge(),
            }));
        }
        _ => {}
    }
    match (l, r) {
        (Value::Int(a, ta), Value::Int(b, tb)) => {
            let (a, b) = (*a, *b);
            if matches!(op, Shl(_) | Shr(_)) {
                let t = *ta;
                if b < 0 || b >= i128::from(t.map_or(64, IntTy::bits)) {
                    return Err(format!("shift amount {b} out of range"));
                }
                let v = if matches!(op, Shl(_)) { a << b } else { a >> b };
                return Ok(Value::Int(t.map_or(v, |t| t.wrap(v)), t));
            }
            let t = unify(*ta, *tb, "arithmetic")?;
            let wrapped = |v: i128| check(v, t, "compute").map(|_| Value::Int(v, t));
            match op {
                Add(_) => wrapped(a + b).map_err(|_| format!("attempt to add with overflow: {a} + {b}")),
                Sub(_) => wrapped(a - b).map_err(|_| format!("attempt to subtract with overflow: {a} - {b}")),
                Mul(_) => wrapped(a.checked_mul(b).ok_or("attempt to multiply with overflow")?).map_err(|_| format!("attempt to multiply with overflow: {a} * {b}")),
                Div(_) => {
                    if b == 0 {
                        Err("attempt to divide by zero".into())
                    } else {
                        wrapped(a / b)
                    }
                }
                Rem(_) => {
                    if b == 0 {
                        Err("attempt to calculate the remainder with a divisor of zero".into())
                    } else {
                        wrapped(a % b)
                    }
                }
                BitAnd(_) => Ok(Value::Int(a & b, t)),
                BitOr(_) => Ok(Value::Int(a | b, t)),
                BitXor(_) => Ok(Value::Int(a ^ b, t)),
                _ => Err("unsupported integer operator".into()),
            }
        }
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(match op {
            Add(_) => a + b,
            Sub(_) => a - b,
            Mul(_) => a * b,
            Div(_) => a / b,
            Rem(_) => a % b,
            _ => return Err("unsupported float operator".into()),
        })),
        (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(match op {
            BitAnd(_) => a & b,
            BitOr(_) => a | b,
            BitXor(_) => a ^ b,
            _ => return Err("unsupported bool operator".into()),
        })),
        (Value::Str(a), Value::Str(b)) if matches!(op, Add(_)) => Ok(Value::str(format!("{a}{b}"))),
        _ => Err(format!("cannot apply operator to {} and {}", l.type_name(), r.type_name())),
    }
}

pub fn compare(l: &Value, r: &Value) -> Result<std::cmp::Ordering, String> {
    Ok(match (l, r) {
        (Value::Int(a, _), Value::Int(b, _)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).ok_or("NaN comparison")?,
        (Value::Str(a), Value::Str(b)) => a.cmp(b),
        (Value::Char(a), Value::Char(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Tuple(a), Value::Tuple(b)) | (Value::Array(a), Value::Array(b)) => {
            for (x, y) in a.iter().zip(b) {
                let o = compare(x, y)?;
                if o.is_ne() {
                    return Ok(o);
                }
            }
            a.len().cmp(&b.len())
        }
        (Value::Enum(_, v1, x), Value::Enum(_, v2, y)) if x.is_empty() && y.is_empty() => v1.cmp(v2),
        _ => return Err(format!("cannot order {} and {}", l.type_name(), r.type_name())),
    })
}

pub fn neg<'a>(v: &Value<'a>) -> Result<Value<'a>, String> {
    match v {
        Value::Int(i, t) => {
            if t.is_some_and(|t| !t.signed()) {
                return Err("cannot negate an unsigned integer".into());
            }
            Ok(check(-*i, *t, "negate")?)
        }
        Value::Float(f) => Ok(Value::Float(-f)),
        o => Err(format!("cannot negate {}", o.type_name())),
    }
}

pub fn not<'a>(v: &Value<'a>) -> Result<Value<'a>, String> {
    match v {
        Value::Bool(b) => Ok(Value::Bool(!b)),
        Value::Int(i, Some(t)) => Ok(Value::Int(t.wrap(!*i), Some(*t))),
        Value::Int(i, None) => Ok(Value::Int(!*i, None)),
        o => Err(format!("cannot apply `!` to {}", o.type_name())),
    }
}

/// `v as ty` for a primitive target type.
pub fn cast<'a>(v: &Value<'a>, ty: &str, discr: impl Fn(&str, &str) -> Option<i128>) -> Result<Value<'a>, String> {
    if let Some(t) = IntTy::parse(ty) {
        let raw = match v {
            Value::Int(i, _) => *i,
            Value::Bool(b) => i128::from(*b),
            Value::Char(c) => *c as i128,
            Value::Float(f) => {
                let x = f.trunc();
                if x.is_nan() {
                    0
                } else if x >= t.max() as f64 {
                    t.max()
                } else if x <= t.min() as f64 {
                    t.min()
                } else {
                    x as i128
                }
            }
            Value::Enum(e, var, p) if p.is_empty() => discr(e, var).ok_or_else(|| format!("cannot cast enum `{e}` to an integer"))?,
            o => return Err(format!("cannot cast {} to {ty}", o.type_name())),
        };
        return Ok(Value::Int(t.wrap(raw), Some(t)));
    }
    match (ty, v) {
        ("f32" | "f64", Value::Int(i, _)) => Ok(Value::Float(*i as f64)),
        ("f32", Value::Float(f)) => Ok(Value::Float(f64::from(*f as f32))),
        ("f64", Value::Float(f)) => Ok(Value::Float(*f)),
        ("char", Value::Int(i, _)) => char::from_u32(*i as u32).map(Value::Char).ok_or_else(|| "invalid char".into()),
        ("bool", Value::Bool(b)) => Ok(Value::Bool(*b)),
        _ => Err(format!("unsupported cast of {} to {ty}", v.type_name())),
    }
}

/// Give `v` the declared type `ty`: untyped integers take it (range checked),
/// typed ones must already match, containers are coerced elementwise.
pub fn coerce<'a>(v: Value<'a>, ty: &Ty) -> Result<Value<'a>, String> {
    match (ty, v) {
        (Ty::Any, v) => Ok(v),
        (Ty::Int(t), Value::Int(i, ti)) => {
            if let Some(x) = ti {
                if x != *t {
                    return Err(format!("mismatched types: expected `{}`, found `{}`", t.name(), x.name()));
                }
            }
            if !t.fits(i) {
                return Err(format!("literal {i} out of range for `{}`", t.name()));
            }
            Ok(Value::Int(i, Some(*t)))
        }
        (Ty::Int(_), v @ Value::Ref(_)) => Ok(v),
        (Ty::Array(e), Value::Array(xs)) => Ok(Value::Array(xs.into_iter().map(|x| coerce(x, e)).collect::<Result<_, _>>()?)),
        (Ty::Tuple(ts), Value::Tuple(xs)) if ts.len() == xs.len() => Ok(Value::Tuple(xs.into_iter().zip(ts).map(|(x, t)| coerce(x, t)).collect::<Result<_, _>>()?)),
        (Ty::Option(e), Value::Enum(t, v, mut p)) if &*t == "Option" && !p.is_empty() => {
            let x = coerce(p.remove(0), e)?;
            Ok(Value::Enum(t, v, vec![x]))
        }
        (Ty::Result(e), Value::Enum(t, v, mut p)) if &*t == "Result" && &*v == "Ok" && !p.is_empty() => {
            let x = coerce(p.remove(0), e)?;
            Ok(Value::Enum(t, v, vec![x]))
        }
        (_, v) => Ok(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin(op: &str, a: Value<'static>, b: Value<'static>) -> Result<Value<'static>, String> {
        let e: syn::ExprBinary = syn::parse_str(&format!("a {op} b")).unwrap();
        binop(&e.op, &a, &b)
    }
    fn u8v(v: i128) -> Value<'static> {
        Value::Int(v, Some(IntTy::U8))
    }
    fn lit(v: i128) -> Value<'static> {
        Value::Int(v, None)
    }

    #[test]
    fn untyped_literals_adopt_the_other_operand() {
        assert!(matches!(bin("+", u8v(200), lit(50)).unwrap(), Value::Int(250, Some(IntTy::U8))));
        assert!(matches!(bin("+", lit(1), lit(2)).unwrap(), Value::Int(3, None)));
    }

    #[test]
    fn typed_arithmetic_is_bounds_checked_at_the_type_boundary() {
        assert!(bin("+", u8v(255), lit(1)).unwrap_err().contains("overflow"));
        assert!(bin("-", u8v(0), lit(1)).unwrap_err().contains("overflow"));
        assert!(matches!(bin("+", u8v(255), lit(0)).unwrap(), Value::Int(255, _)));
        assert!(bin("/", u8v(1), lit(0)).unwrap_err().contains("zero"));
    }

    #[test]
    fn mismatched_typed_operands_are_rejected_but_shift_counts_are_free() {
        let u32v = Value::Int(1, Some(IntTy::U32));
        assert!(bin("+", u8v(1), u32v.clone()).unwrap_err().contains("mismatched"));
        assert!(matches!(bin("<<", u8v(0x81), lit(1)).unwrap(), Value::Int(0x02, Some(IntTy::U8))));
        assert!(matches!(bin("<<", u32v, u8v(31)).unwrap(), Value::Int(0x8000_0000, Some(IntTy::U32))));
        assert!(bin("<<", u8v(1), lit(8)).unwrap_err().contains("shift"));
    }

    #[test]
    fn casts_truncate_like_rust() {
        let no = |_: &str, _: &str| None;
        assert!(matches!(cast(&lit(0x1ff), "u8", no).unwrap(), Value::Int(0xff, Some(IntTy::U8))));
        assert!(matches!(cast(&lit(-1), "u32", no).unwrap(), Value::Int(0xffff_ffff, Some(IntTy::U32))));
        assert!(matches!(cast(&Value::Int(0xffff_ffff, Some(IntTy::U32)), "i32", no).unwrap(), Value::Int(-1, Some(IntTy::I32))));
    }

    #[test]
    fn coercion_checks_range_and_exact_types() {
        assert!(coerce(lit(300), &Ty::Int(IntTy::U8)).unwrap_err().contains("out of range"));
        assert!(coerce(Value::Int(1, Some(IntTy::U32)), &Ty::Int(IntTy::U8)).unwrap_err().contains("expected `u8`"));
        let arr = Value::Array(vec![lit(1), lit(2)]);
        let c = coerce(arr, &Ty::Array(Box::new(Ty::Int(IntTy::U8)))).unwrap();
        assert!(matches!(&c, Value::Array(x) if matches!(x[1], Value::Int(2, Some(IntTy::U8)))));
    }
}
