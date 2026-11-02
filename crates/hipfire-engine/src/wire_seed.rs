// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wire `seed` domain: optional non-negative 64-bit integer (OpenAI-compatible
//! deterministic-sampling field). Unlike `parse_wire_attempt_id`, an out-of-domain
//! value is an ERROR, not a silent fallback to unseeded: a client that sends
//! `seed: -1` or `seed: 1.5` is asking for reproducibility and must not get a
//! fresh entropy stream without notice.

pub fn parse_wire_seed(value: Option<&serde_json::Value>) -> Result<Option<u64>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(n) = value.as_number() else {
        return Err(format!("seed must be a non-negative integer, got {value}"));
    };
    if let Some(s) = n.as_u64() {
        return Ok(Some(s));
    }
    if let Some(s) = n.as_i64() {
        return Err(format!("seed must be non-negative, got {s}"));
    }
    Err(format!("seed must be an integer, got {value}"))
}

#[cfg(test)]
mod tests {
    use super::parse_wire_seed;

    fn parse(json: &str) -> Result<Option<u64>, String> {
        parse_wire_seed(Some(&serde_json::from_str(json).unwrap()))
    }

    #[test]
    fn absent_and_null_are_unseeded() {
        assert_eq!(parse_wire_seed(None), Ok(None));
        assert_eq!(parse("null"), Ok(None));
    }

    #[test]
    fn non_negative_integers_pass_through_verbatim() {
        assert_eq!(parse("0"), Ok(Some(0)));
        assert_eq!(parse("1234"), Ok(Some(1234)));
        assert_eq!(parse(&u64::MAX.to_string()), Ok(Some(u64::MAX)));
    }

    #[test]
    fn negative_seeds_are_rejected_not_treated_as_unseeded() {
        let err = parse("-1").unwrap_err();
        assert!(err.contains("non-negative"), "reason: {err}");
        assert!((0..100).all(|i| parse(&format!("-{i}")).is_err()));
    }

    #[test]
    fn fractional_and_non_numeric_seeds_are_rejected() {
        assert!(parse("1.5").is_err());
        assert!(parse("0.0").is_err());
        assert!(parse("\"42\"").is_err());
        assert!(parse("true").is_err());
        assert!(parse("[]").is_err());
        assert!(parse("{}").is_err());
    }
}
