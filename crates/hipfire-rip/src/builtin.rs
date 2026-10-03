//! Methods of the std types a kernel script uses: integers, strings,
//! arrays / `Vec` / iterators (materialized eagerly), `Option`, `Result`.

use crate::interp::{Cx, Interp, R};
use crate::ops;
use crate::value::*;

fn is_enum(v: &Value, ty: &str) -> bool {
    matches!(v, Value::Enum(t, _, _) if &**t == ty)
}

/// The elements `for` / iterator adaptors walk.
pub fn items<'a>(v: &Value<'a>) -> Result<Vec<Value<'a>>, String> {
    match v {
        Value::Array(xs) => Ok(xs.clone()),
        Value::Range(r) => {
            let (Some(s), Some(e)) = (r.start, r.end) else { return Err("an unbounded range cannot be collected".into()) };
            let e = if r.inclusive { e + 1 } else { e };
            if e - s > (1 << 22) {
                return Err("range too large to materialize".into());
            }
            Ok((s..e.max(s)).map(|i| Value::Int(i, r.ty)).collect())
        }
        Value::Enum(t, var, p) if &**t == "Option" => Ok(if &**var == "Some" { p.clone() } else { vec![] }),
        Value::Str(s) => Ok(s.chars().map(Value::Char).collect()),
        o => Err(format!("`{}` is not iterable", o.type_name())),
    }
}

pub fn is_mutator(v: &Value, name: &str) -> bool {
    match v {
        Value::Array(_) => matches!(name, "push" | "pop" | "insert" | "remove" | "clear" | "extend" | "truncate" | "swap" | "reverse" | "sort" | "sort_unstable" | "append"),
        Value::Str(_) => matches!(name, "push_str" | "push" | "clear"),
        Value::Enum(t, _, _) if &**t == "Option" => matches!(name, "take" | "insert"),
        _ => false,
    }
}

fn idx(v: &Value, what: &str) -> Result<usize, String> {
    match v {
        Value::Int(i, _) if *i >= 0 => Ok(*i as usize),
        _ => Err(format!("{what} must be a non-negative integer")),
    }
}

pub fn call_mut<'a>(_it: &Interp<'a>, v: &mut Value<'a>, name: &str, args: Vec<Value<'a>>) -> Result<Value<'a>, String> {
    match (v, name) {
        (Value::Array(xs), "push") => {
            xs.push(args.into_iter().next().ok_or("push needs a value")?);
            Ok(Value::Unit)
        }
        (Value::Array(xs), "pop") => Ok(Value::option(xs.pop())),
        (Value::Array(xs), "insert") => {
            let i = idx(&args[0], "insert index")?;
            if i > xs.len() {
                return Err("insert index out of bounds".into());
            }
            xs.insert(i, args[1].clone());
            Ok(Value::Unit)
        }
        (Value::Array(xs), "remove") => {
            let i = idx(&args[0], "remove index")?;
            if i >= xs.len() {
                return Err("remove index out of bounds".into());
            }
            Ok(xs.remove(i))
        }
        (Value::Array(xs), "clear") => {
            xs.clear();
            Ok(Value::Unit)
        }
        (Value::Array(xs), "truncate") => {
            xs.truncate(idx(&args[0], "truncate length")?);
            Ok(Value::Unit)
        }
        (Value::Array(xs), "extend") | (Value::Array(xs), "append") => {
            xs.extend(items(&args[0])?);
            Ok(Value::Unit)
        }
        (Value::Array(xs), "swap") => {
            let (a, b) = (idx(&args[0], "swap index")?, idx(&args[1], "swap index")?);
            if a >= xs.len() || b >= xs.len() {
                return Err("swap index out of bounds".into());
            }
            xs.swap(a, b);
            Ok(Value::Unit)
        }
        (Value::Array(xs), "reverse") => {
            xs.reverse();
            Ok(Value::Unit)
        }
        (Value::Array(xs), "sort") | (Value::Array(xs), "sort_unstable") => {
            let mut err = None;
            xs.sort_by(|a, b| {
                ops::compare(a, b).unwrap_or_else(|e| {
                    err = Some(e);
                    std::cmp::Ordering::Equal
                })
            });
            err.map_or(Ok(Value::Unit), Err)
        }
        (Value::Str(s), "push_str") => {
            let Value::Str(t) = &args[0] else { return Err("push_str needs a string".into()) };
            *s = std::rc::Rc::from(format!("{s}{t}"));
            Ok(Value::Unit)
        }
        (Value::Str(s), "push") => {
            let Value::Char(c) = &args[0] else { return Err("push needs a char".into()) };
            *s = std::rc::Rc::from(format!("{s}{c}"));
            Ok(Value::Unit)
        }
        (Value::Str(s), "clear") => {
            *s = "".into();
            Ok(Value::Unit)
        }
        (v @ Value::Enum(..), "take") => Ok(std::mem::replace(v, Value::none())),
        (v @ Value::Enum(..), "insert") => {
            *v = Value::some(args[0].clone());
            Ok(args[0].clone())
        }
        (v, n) => Err(format!("no mutating method `{n}` on `{}`", v.type_name())),
    }
}

fn int_of(v: &Value) -> Result<(i128, Option<IntTy>), String> {
    match v {
        Value::Int(i, t) => Ok((*i, *t)),
        o => Err(format!("expected an integer, found {}", o.type_name())),
    }
}

fn bool_of(v: Value) -> Result<bool, String> {
    match v {
        Value::Bool(b) => Ok(b),
        o => Err(format!("closure must return bool, found {}", o.type_name())),
    }
}

pub fn call<'a>(it: &Interp<'a>, v: &Value<'a>, name: &str, targs: &[String], args: Vec<Value<'a>>, cx: &mut Cx<'_>) -> R<'a, Option<Value<'a>>> {
    let args: Vec<Value<'a>> = args
        .into_iter()
        .map(|a| match a {
            Value::Ref(_) => it.deref(a),
            a => a,
        })
        .collect();
    // Methods every value has.
    match name {
        "clone" | "to_owned" | "borrow" | "as_ref" | "as_mut" | "by_ref" | "borrow_mut" if args.is_empty() => return Ok(Some(v.clone())),
        "into" | "to_vec" if args.is_empty() && !matches!(v, Value::Range(_)) => return Ok(Some(v.clone())),
        "to_string" if args.is_empty() => return Ok(Some(Value::str(crate::fmt::display(v)?))),
        _ => {}
    }
    let res = match v {
        Value::Int(i, t) => int_method(*i, *t, name, &args)?,
        Value::Float(f) => float_method(*f, name, &args)?,
        Value::Bool(b) => match name {
            "then_some" => Some(if *b { Value::some(args[0].clone()) } else { Value::none() }),
            "then" => Some(if *b { Value::some(it.call_value(&args[0], vec![], cx)?) } else { Value::none() }),
            _ => None,
        },
        Value::Char(c) => match name {
            "is_ascii_digit" => Some(Value::Bool(c.is_ascii_digit())),
            "is_alphabetic" => Some(Value::Bool(c.is_alphabetic())),
            "is_alphanumeric" => Some(Value::Bool(c.is_alphanumeric())),
            "to_ascii_uppercase" => Some(Value::Char(c.to_ascii_uppercase())),
            "to_ascii_lowercase" => Some(Value::Char(c.to_ascii_lowercase())),
            _ => None,
        },
        Value::Str(s) => str_method(s, name, targs, &args)?,
        Value::Enum(t, var, p) if &**t == "Option" => option_method(it, var, p, name, &args, cx)?,
        Value::Enum(t, var, p) if &**t == "Result" => result_method(it, var, p, name, &args, cx)?,
        Value::Range(r) => match name {
            "contains" => {
                let (x, _) = int_of(&args[0])?;
                Some(Value::Bool(r.start.is_none_or(|s| x >= s) && r.end.is_none_or(|e| if r.inclusive { x <= e } else { x < e })))
            }
            "start" => r.start.map(|s| Value::Int(s, r.ty)),
            "end" => r.end.map(|s| Value::Int(s, r.ty)),
            "is_empty" => Some(Value::Bool(items(v)?.is_empty())),
            "len" => Some(Value::usize(items(v)?.len())),
            _ => return call(it, &Value::Array(items(v)?), name, targs, args, cx),
        },
        Value::Array(xs) => array_method(it, xs, name, targs, &args, cx)?,
        _ => None,
    };
    Ok(res)
}

fn int_method<'a>(i: i128, t: Option<IntTy>, name: &str, args: &[Value<'a>]) -> Result<Option<Value<'a>>, String> {
    let lim = |t: Option<IntTy>| (t.map_or(i128::from(i32::MIN), IntTy::min), t.map_or(i128::from(i32::MAX), IntTy::max));
    let other = |k: usize| -> Result<i128, String> {
        let (o, ot) = int_of(args.get(k).ok_or("missing integer argument")?)?;
        if let (Some(a), Some(b)) = (t, ot) {
            if a != b {
                return Err(format!("mismatched integer types: {} and {}", a.name(), b.name()));
            }
        }
        Ok(o)
    };
    let out_t = |ot: Option<IntTy>| t.or(ot);
    let arg_t = |k: usize| match args.get(k) {
        Some(Value::Int(_, ot)) => *ot,
        _ => None,
    };
    let checked = |v: i128, t: Option<IntTy>| -> Option<Value<'a>> {
        let (lo, hi) = lim(t);
        (v >= lo && v <= hi).then_some(Value::Int(v, t))
    };
    Ok(Some(match name {
        "abs" => Value::Int(i.abs(), t),
        "signum" => Value::Int(i.signum(), t),
        "pow" => {
            let e = other(0).map(|x| x as u32).or_else(|_| int_of(&args[0]).map(|x| x.0 as u32))?;
            let v = i.checked_pow(e).ok_or("attempt to multiply with overflow")?;
            let (lo, hi) = lim(t);
            if v < lo || v > hi {
                return Err("attempt to multiply with overflow".into());
            }
            Value::Int(v, t)
        }
        "min" => Value::Int(i.min(other(0)?), out_t(arg_t(0))),
        "max" => Value::Int(i.max(other(0)?), out_t(arg_t(0))),
        "clamp" => Value::Int(i.clamp(other(0)?, other(1)?), out_t(arg_t(0))),
        "abs_diff" => Value::Int((i - other(0)?).abs(), t.map(|t| match t {
            IntTy::I8 => IntTy::U8,
            IntTy::I16 => IntTy::U16,
            IntTy::I32 => IntTy::U32,
            IntTy::I64 => IntTy::U64,
            IntTy::Isize => IntTy::Usize,
            u => u,
        })),
        "saturating_sub" => {
            let (lo, hi) = lim(out_t(arg_t(0)));
            Value::Int((i - other(0)?).clamp(lo, hi), out_t(arg_t(0)))
        }
        "saturating_add" => {
            let (lo, hi) = lim(out_t(arg_t(0)));
            Value::Int((i + other(0)?).clamp(lo, hi), out_t(arg_t(0)))
        }
        "saturating_mul" => {
            let (lo, hi) = lim(out_t(arg_t(0)));
            Value::Int((i * other(0)?).clamp(lo, hi), out_t(arg_t(0)))
        }
        "wrapping_add" => Value::Int(t.map_or(i + other(0)?, |t| t.wrap(i + other(0).unwrap_or(0))), t),
        "wrapping_sub" => Value::Int(t.map_or(i - other(0)?, |t| t.wrap(i - other(0).unwrap_or(0))), t),
        "wrapping_mul" => Value::Int(t.map_or(i * other(0)?, |t| t.wrap(i * other(0).unwrap_or(0))), t),
        "checked_add" => Value::option(checked(i + other(0)?, out_t(arg_t(0)))),
        "checked_sub" => Value::option(checked(i - other(0)?, out_t(arg_t(0)))),
        "checked_mul" => Value::option(i.checked_mul(other(0)?).and_then(|v| checked(v, out_t(arg_t(0))))),
        "checked_div" => {
            let d = other(0)?;
            Value::option((d != 0).then(|| Value::Int(i / d, out_t(arg_t(0)))))
        }
        "div_ceil" => {
            let d = other(0)?;
            if d == 0 {
                return Err("attempt to divide by zero".into());
            }
            Value::Int((i + d - 1).div_euclid(d), out_t(arg_t(0)))
        }
        "next_multiple_of" => {
            let d = other(0)?;
            if d == 0 {
                return Err("attempt to calculate the remainder with a divisor of zero".into());
            }
            Value::Int((i + d - 1).div_euclid(d) * d, out_t(arg_t(0)))
        }
        "rem_euclid" => Value::Int(i.rem_euclid(other(0)?), out_t(arg_t(0))),
        "is_power_of_two" => Value::Bool(i > 0 && (i & (i - 1)) == 0),
        "count_ones" => Value::Int(i128::from((i as u128 & t.map_or(u128::from(u32::MAX), |t| if t.bits() == 64 { u128::from(u64::MAX) } else { (1u128 << t.bits()) - 1 })).count_ones()), Some(IntTy::U32)),
        "trailing_zeros" => {
            let bits = t.map_or(32, IntTy::bits);
            Value::Int(if i == 0 { i128::from(bits) } else { i128::from((i as u128).trailing_zeros()) }, Some(IntTy::U32))
        }
        "leading_zeros" => {
            let bits = t.map_or(32, IntTy::bits);
            let mask = if bits == 64 { u128::from(u64::MAX) } else { (1u128 << bits) - 1 };
            Value::Int(i128::from((i as u128 & mask).leading_zeros() - (128 - bits)), Some(IntTy::U32))
        }
        "is_positive" => Value::Bool(i > 0),
        "is_negative" => Value::Bool(i < 0),
        "try_into" => Value::ok(Value::Int(i, t)),
        _ => return Ok(None),
    }))
}

fn float_method<'a>(f: f64, name: &str, args: &[Value<'a>]) -> Result<Option<Value<'a>>, String> {
    let fl = |k: usize| match args.get(k) {
        Some(Value::Float(x)) => Ok(*x),
        Some(Value::Int(x, _)) => Ok(*x as f64),
        _ => Err("expected a float".to_string()),
    };
    Ok(Some(match name {
        "sqrt" => Value::Float(f.sqrt()),
        "abs" => Value::Float(f.abs()),
        "floor" => Value::Float(f.floor()),
        "ceil" => Value::Float(f.ceil()),
        "round" => Value::Float(f.round()),
        "trunc" => Value::Float(f.trunc()),
        "exp" => Value::Float(f.exp()),
        "ln" => Value::Float(f.ln()),
        "powi" => Value::Float(f.powi(int_of(&args[0])?.0 as i32)),
        "powf" => Value::Float(f.powf(fl(0)?)),
        "min" => Value::Float(f.min(fl(0)?)),
        "max" => Value::Float(f.max(fl(0)?)),
        "to_bits" => Value::Int(i128::from((f as f32).to_bits()), Some(IntTy::U32)),
        "is_nan" => Value::Bool(f.is_nan()),
        _ => return Ok(None),
    }))
}

fn str_method<'a>(s: &std::rc::Rc<str>, name: &str, targs: &[String], args: &[Value<'a>]) -> Result<Option<Value<'a>>, String> {
    let sarg = |k: usize| match args.get(k) {
        Some(Value::Str(x)) => Ok(x.to_string()),
        Some(Value::Char(c)) => Ok(c.to_string()),
        _ => Err("expected a string argument".to_string()),
    };
    Ok(Some(match name {
        "len" => Value::usize(s.len()),
        "is_empty" => Value::Bool(s.is_empty()),
        "as_str" | "as_string" | "trim_start_matches_none" => Value::Str(s.clone()),
        "starts_with" => Value::Bool(s.starts_with(&sarg(0)?)),
        "ends_with" => Value::Bool(s.ends_with(&sarg(0)?)),
        "contains" => Value::Bool(s.contains(&sarg(0)?)),
        "trim" => Value::str(s.trim()),
        "to_uppercase" | "to_ascii_uppercase" => Value::str(s.to_uppercase()),
        "to_lowercase" | "to_ascii_lowercase" => Value::str(s.to_lowercase()),
        "replace" => Value::str(s.replace(&sarg(0)?, &sarg(1)?)),
        "repeat" => Value::str(s.repeat(idx(&args[0], "repeat count")?)),
        "chars" => Value::Array(s.chars().map(Value::Char).collect()),
        "bytes" => Value::Array(s.bytes().map(Value::u8).collect()),
        "lines" => Value::Array(s.lines().map(Value::str).collect()),
        "split" => Value::Array(s.split(&sarg(0)?[..]).map(Value::str).collect()),
        "parse" => {
            let ty = targs.first().map(String::as_str).unwrap_or("i32");
            match IntTy::parse(ty) {
                Some(t) => match s.trim().parse::<i128>() {
                    Ok(v) if t.fits(v) => Value::ok(Value::Int(v, Some(t))),
                    Ok(_) => Value::err("number too large for the target type"),
                    Err(e) => Value::err(e.to_string()),
                },
                None if ty == "f64" || ty == "f32" => match s.trim().parse::<f64>() {
                    Ok(v) => Value::ok(Value::Float(v)),
                    Err(e) => Value::err(e.to_string()),
                },
                None => return Err(format!("parse::<{ty}>() is not supported")),
            }
        }
        _ => return Ok(None),
    }))
}

fn expect_fail(kind: &str, msg: &Value) -> String {
    format!("{kind}: {}", crate::fmt::debug(msg, false))
}

fn option_method<'a>(it: &Interp<'a>, var: &str, p: &[Value<'a>], name: &str, args: &[Value<'a>], cx: &mut Cx<'_>) -> R<'a, Option<Value<'a>>> {
    let some = var == "Some";
    let inner = p.first().cloned();
    Ok(Some(match name {
        "is_some" => Value::Bool(some),
        "is_none" => Value::Bool(!some),
        "is_some_and" => Value::Bool(some && bool_of(it.call_value(&args[0], vec![inner.expect("payload")], cx)?)?),
        "unwrap" => inner.ok_or("called `Option::unwrap()` on a `None` value")?,
        "expect" => inner.ok_or_else(|| expect_fail("expect failed", &args[0]))?,
        "unwrap_or" => inner.unwrap_or_else(|| args[0].clone()),
        "unwrap_or_else" => match inner {
            Some(v) => v,
            None => it.call_value(&args[0], vec![], cx)?,
        },
        "unwrap_or_default" => inner.unwrap_or(Value::Int(0, None)),
        "map" => match inner {
            Some(v) => Value::some(it.call_value(&args[0], vec![v], cx)?),
            None => Value::none(),
        },
        "and_then" => match inner {
            Some(v) => it.call_value(&args[0], vec![v], cx)?,
            None => Value::none(),
        },
        "map_or" => match inner {
            Some(v) => it.call_value(&args[1], vec![v], cx)?,
            None => args[0].clone(),
        },
        "filter" => match inner {
            Some(v) if bool_of(it.call_value(&args[0], vec![v.clone()], cx)?)? => Value::some(v),
            _ => Value::none(),
        },
        "ok_or" => match inner {
            Some(v) => Value::ok(v),
            None => Value::Enum("Result".into(), "Err".into(), vec![args[0].clone()]),
        },
        "ok_or_else" => match inner {
            Some(v) => Value::ok(v),
            None => Value::Enum("Result".into(), "Err".into(), vec![it.call_value(&args[0], vec![], cx)?]),
        },
        "or" => if some { Value::some(inner.expect("payload")) } else { args[0].clone() },
        "or_else" => if some { Value::some(inner.expect("payload")) } else { it.call_value(&args[0], vec![], cx)? },
        "cloned" | "copied" => Value::option(inner),
        "iter" | "into_iter" => Value::Array(inner.into_iter().collect()),
        _ => return Ok(None),
    }))
}

fn result_method<'a>(it: &Interp<'a>, var: &str, p: &[Value<'a>], name: &str, args: &[Value<'a>], cx: &mut Cx<'_>) -> R<'a, Option<Value<'a>>> {
    let ok = var == "Ok";
    let inner = p.first().cloned().unwrap_or(Value::Unit);
    Ok(Some(match name {
        "is_ok" => Value::Bool(ok),
        "is_err" => Value::Bool(!ok),
        "unwrap" => {
            if ok {
                inner
            } else {
                return Err(format!("called `Result::unwrap()` on an `Err` value: {}", crate::fmt::debug(&inner, false)).into());
            }
        }
        "expect" => {
            if ok {
                inner
            } else {
                return Err(format!("{}: {}", crate::fmt::display(&args[0]).unwrap_or_default(), crate::fmt::debug(&inner, false)).into());
            }
        }
        "unwrap_err" => {
            if ok {
                return Err("called `Result::unwrap_err()` on an `Ok` value".into());
            }
            inner
        }
        "unwrap_or" => if ok { inner } else { args[0].clone() },
        "unwrap_or_else" => if ok { inner } else { it.call_value(&args[0], vec![inner], cx)? },
        "unwrap_or_default" => if ok { inner } else { Value::Int(0, None) },
        "map" => if ok { Value::ok(it.call_value(&args[0], vec![inner], cx)?) } else { Value::Enum("Result".into(), "Err".into(), vec![inner]) },
        "map_err" => if ok { Value::ok(inner) } else { Value::Enum("Result".into(), "Err".into(), vec![it.call_value(&args[0], vec![inner], cx)?]) },
        "and_then" => if ok { it.call_value(&args[0], vec![inner], cx)? } else { Value::Enum("Result".into(), "Err".into(), vec![inner]) },
        "or_else" => if ok { Value::ok(inner) } else { it.call_value(&args[0], vec![inner], cx)? },
        "ok" => if ok { Value::some(inner) } else { Value::none() },
        "err" => if ok { Value::none() } else { Value::some(inner) },
        _ => return Ok(None),
    }))
}

fn array_method<'a>(it: &Interp<'a>, xs: &[Value<'a>], name: &str, targs: &[String], args: &[Value<'a>], cx: &mut Cx<'_>) -> R<'a, Option<Value<'a>>> {
    let call1 = |f: &Value<'a>, x: &Value<'a>, cx: &mut Cx<'_>| it.call_value(f, vec![x.clone()], cx);
    let arr = |v: Vec<Value<'a>>| Some(Value::Array(v));
    Ok(match name {
        "len" => Some(Value::usize(xs.len())),
        "is_empty" => Some(Value::Bool(xs.is_empty())),
        "iter" | "into_iter" | "iter_mut" | "cloned" | "copied" | "to_vec" | "drain" => arr(xs.to_vec()),
        "first" => Some(Value::option(xs.first().cloned())),
        "last" => Some(Value::option(xs.last().cloned())),
        "get" => Some(Value::option(match &args[0] {
            Value::Int(i, _) => usize::try_from(*i).ok().and_then(|i| xs.get(i)).cloned(),
            _ => return Err("get needs an integer index".into()),
        })),
        "contains" => Some(Value::Bool(xs.iter().any(|x| values_eq(x, &args[0])))),
        "join" => {
            let Value::Str(sep) = &args[0] else { return Err("join needs a separator string".into()) };
            let parts: Vec<String> = xs.iter().map(crate::fmt::display).collect::<Result<_, _>>()?;
            Some(Value::str(parts.join(sep)))
        }
        "concat" => {
            let mut out = Vec::new();
            for x in xs {
                out.extend(items(x)?);
            }
            arr(out)
        }
        "enumerate" => arr(xs.iter().enumerate().map(|(i, x)| Value::Tuple(vec![Value::usize(i), x.clone()])).collect()),
        "rev" => arr(xs.iter().rev().cloned().collect()),
        "take" => arr(xs.iter().take(idx(&args[0], "take count")?).cloned().collect()),
        "skip" => arr(xs.iter().skip(idx(&args[0], "skip count")?).cloned().collect()),
        "step_by" => arr(xs.iter().step_by(idx(&args[0], "step")?.max(1)).cloned().collect()),
        "chain" => {
            let mut v = xs.to_vec();
            v.extend(items(&args[0])?);
            arr(v)
        }
        "zip" => arr(xs.iter().zip(items(&args[0])?).map(|(a, b)| Value::Tuple(vec![a.clone(), b])).collect()),
        "map" => arr(xs.iter().map(|x| call1(&args[0], x, cx)).collect::<R<Vec<_>>>()?),
        "for_each" => {
            for x in xs {
                call1(&args[0], x, cx)?;
            }
            Some(Value::Unit)
        }
        "filter" => {
            let mut out = Vec::new();
            for x in xs {
                if bool_of(it.deref(call1(&args[0], x, cx)?))? {
                    out.push(x.clone());
                }
            }
            arr(out)
        }
        "filter_map" => {
            let mut out = Vec::new();
            for x in xs {
                if let Value::Enum(_, v, mut p) = it.deref(call1(&args[0], x, cx)?) {
                    if &*v == "Some" {
                        out.push(p.remove(0));
                    }
                }
            }
            arr(out)
        }
        "flat_map" => {
            let mut out = Vec::new();
            for x in xs {
                out.extend(items(&it.deref(call1(&args[0], x, cx)?))?);
            }
            arr(out)
        }
        "take_while" | "skip_while" => {
            let mut idx_stop = xs.len();
            for (i, x) in xs.iter().enumerate() {
                if !bool_of(it.deref(call1(&args[0], x, cx)?))? {
                    idx_stop = i;
                    break;
                }
            }
            arr(if name == "take_while" { xs[..idx_stop].to_vec() } else { xs[idx_stop..].to_vec() })
        }
        "any" => {
            for x in xs {
                if bool_of(it.deref(call1(&args[0], x, cx)?))? {
                    return Ok(Some(Value::Bool(true)));
                }
            }
            Some(Value::Bool(false))
        }
        "all" => {
            for x in xs {
                if !bool_of(it.deref(call1(&args[0], x, cx)?))? {
                    return Ok(Some(Value::Bool(false)));
                }
            }
            Some(Value::Bool(true))
        }
        "position" => {
            for (i, x) in xs.iter().enumerate() {
                if bool_of(it.deref(call1(&args[0], x, cx)?))? {
                    return Ok(Some(Value::some(Value::usize(i))));
                }
            }
            Some(Value::none())
        }
        "find" => {
            for x in xs {
                if bool_of(it.deref(call1(&args[0], x, cx)?))? {
                    return Ok(Some(Value::some(x.clone())));
                }
            }
            Some(Value::none())
        }
        "count" => Some(Value::usize(xs.len())),
        "nth" => Some(Value::option(xs.get(idx(&args[0], "nth index")?).cloned())),
        "next" => Some(Value::option(xs.first().cloned())),
        "fold" => {
            let mut acc = args[0].clone();
            for x in xs {
                acc = it.call_value(&args[1], vec![acc, x.clone()], cx)?;
            }
            Some(acc)
        }
        "sum" | "product" => {
            let mul = name == "product";
            let mut acc: Option<Value<'a>> = None;
            for x in xs {
                let x = it.deref(x.clone());
                acc = Some(match acc {
                    None => x,
                    Some(a) => ops::binop(&if mul { syn::BinOp::Mul(Default::default()) } else { syn::BinOp::Add(Default::default()) }, &a, &x)?,
                });
            }
            let ty = targs.first().and_then(|t| IntTy::parse(t));
            Some(match (acc, ty) {
                (Some(Value::Int(i, _)), Some(t)) => Value::Int(i, Some(t)),
                (Some(v), _) => v,
                (None, t) => Value::Int(i128::from(mul), t),
            })
        }
        "min" | "max" => {
            let mut best: Option<&Value<'a>> = None;
            for x in xs {
                best = match best {
                    None => Some(x),
                    Some(b) => {
                        let o = ops::compare(x, b)?;
                        if (name == "min" && o.is_lt()) || (name == "max" && o.is_ge()) {
                            Some(x)
                        } else {
                            Some(b)
                        }
                    }
                };
            }
            Some(Value::option(best.cloned()))
        }
        "max_by_key" | "min_by_key" => {
            let mut best: Option<(Value<'a>, Value<'a>)> = None;
            for x in xs {
                let k = it.deref(call1(&args[0], x, cx)?);
                best = match best {
                    None => Some((k, x.clone())),
                    Some((bk, bx)) => {
                        let o = ops::compare(&k, &bk)?;
                        if (name == "min_by_key" && o.is_lt()) || (name == "max_by_key" && o.is_ge()) {
                            Some((k, x.clone()))
                        } else {
                            Some((bk, bx))
                        }
                    }
                };
            }
            Some(Value::option(best.map(|b| b.1)))
        }
        "unzip" => {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for x in xs {
                let Value::Tuple(p) = it.deref(x.clone()) else { return Err("unzip needs pairs".into()) };
                a.push(p[0].clone());
                b.push(p[1].clone());
            }
            Some(Value::Tuple(vec![Value::Array(a), Value::Array(b)]))
        }
        "collect" => match targs.first().map(String::as_str) {
            Some("Result") => {
                let mut ok = Vec::new();
                for x in xs {
                    match it.deref(x.clone()) {
                        Value::Enum(t, v, mut p) if &*t == "Result" => {
                            if &*v == "Ok" {
                                ok.push(p.remove(0));
                            } else {
                                return Ok(Some(Value::Enum(t, v, p)));
                            }
                        }
                        o => return Err(format!("collect::<Result<..>> over a non-Result ({})", o.type_name()).into()),
                    }
                }
                Some(Value::ok(Value::Array(ok)))
            }
            Some("Option") => {
                let mut some = Vec::new();
                for x in xs {
                    match it.deref(x.clone()) {
                        Value::Enum(t, v, mut p) if &*t == "Option" => {
                            if &*v == "Some" {
                                some.push(p.remove(0));
                            } else {
                                return Ok(Some(Value::none()));
                            }
                        }
                        o => return Err(format!("collect::<Option<..>> over a non-Option ({})", o.type_name()).into()),
                    }
                }
                Some(Value::some(Value::Array(some)))
            }
            Some("String") => {
                let mut s = String::new();
                for x in xs {
                    s.push_str(&crate::fmt::display(x)?);
                }
                Some(Value::str(s))
            }
            _ => arr(xs.to_vec()),
        },
        "windows" => {
            let n = idx(&args[0], "window size")?;
            arr(xs.windows(n.max(1)).map(|w| Value::Array(w.to_vec())).collect())
        }
        "chunks" => {
            let n = idx(&args[0], "chunk size")?;
            arr(xs.chunks(n.max(1)).map(|w| Value::Array(w.to_vec())).collect())
        }
        _ => None,
    })
}

#[allow(dead_code)]
fn _is_enum_used(v: &Value) -> bool {
    is_enum(v, "Option")
}
