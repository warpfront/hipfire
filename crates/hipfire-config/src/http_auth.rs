// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Validation of user-supplied values that become outgoing HTTP header values.
//!
//! The Hugging Face token is read from the environment and sent as
//! `Authorization: Bearer <token>`. A token pasted with a trailing newline or
//! space, wrapped in quotes (Windows `set HF_TOKEN="hf_..."` keeps them), or
//! holding a non-ASCII character is not a legal header value, and the HTTP
//! client reports it as the unrelated `http: failed to parse header value`.
//! [`hf_authorization`] checks it first and names the environment variable that
//! supplied it. A bad value is rejected, never trimmed or repaired silently.
//!
//! Errors describe the offending character and its position, never the token.

use crate::{ConfigError, Result};
use std::ffi::OsString;

/// Environment variables that may hold a Hugging Face token, in precedence
/// order. A variable that is set but empty counts as unset.
pub const HF_TOKEN_ENV_VARS: [&str; 2] = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"];

/// Why a bearer token is not usable as an HTTP header value. Never contains the
/// token; the character is named by kind or code point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BearerTokenProblem {
    /// 1-based character position of the first offending character.
    pub position: usize,
    /// What the offending character is, for example "a space" or
    /// "a non-ASCII character (U+00E9)".
    pub what: String,
    /// Extra guidance for the common cause of this kind of character.
    pub hint: &'static str,
}

impl std::fmt::Display for BearerTokenProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the token contains {} at character {}; {}",
            self.what, self.position, self.hint
        )
    }
}

/// Checks that `token` is a legal single-line bearer credential (RFC 6750
/// `b64token`): ASCII letters and digits plus `-` `.` `_` `~` `+` `/`, with `=`
/// allowed only as trailing padding. That excludes spaces, tabs, line breaks,
/// quotes, other control characters and every non-ASCII character, so the
/// value always forms a valid `Authorization` header. An empty token is the
/// caller's "unset" case and is not checked here.
pub fn validate_bearer_token(token: &str) -> std::result::Result<(), BearerTokenProblem> {
    let mut padding = false;
    for (index, ch) in token.chars().enumerate() {
        let valid = if ch == '=' {
            padding = true;
            true
        } else {
            !padding
                && (ch.is_ascii_alphanumeric() || matches!(ch, '-' | '.' | '_' | '~' | '+' | '/'))
        };
        if !valid {
            let (what, hint) = describe_invalid(ch, padding);
            return Err(BearerTokenProblem {
                position: index + 1,
                what,
                hint,
            });
        }
    }
    Ok(())
}

/// Names a character that cannot appear in a header credential, by kind or code
/// point ("a space", "a non-ASCII character (U+00E9)"). Never the character
/// itself for printable text. `ch` must not be `=`.
pub(crate) fn describe_char(ch: char) -> String {
    describe_invalid(ch, false).0
}

fn describe_invalid(ch: char, after_padding: bool) -> (String, &'static str) {
    const WHITESPACE: &str =
        "remove stray spaces and line breaks, such as one copied along with the token";
    const QUOTE: &str = "remove the quotation marks; they are not part of the token \
         (in Windows `set` they are kept literally)";
    const NON_ASCII: &str = "a token is plain ASCII, so this is usually a copy or paste mistake";
    const OTHER: &str = "a token holds only letters, digits and - . _ ~ + /";
    match ch {
        ' ' => ("a space".to_owned(), WHITESPACE),
        '\t' => ("a tab".to_owned(), WHITESPACE),
        '\n' => ("a newline".to_owned(), WHITESPACE),
        '\r' => ("a carriage return".to_owned(), WHITESPACE),
        '"' => ("a double quote".to_owned(), QUOTE),
        '\'' => ("a single quote".to_owned(), QUOTE),
        '=' => unreachable!("padding is valid"),
        c if after_padding && c.is_ascii_alphanumeric() => (
            "a character after the `=` padding".to_owned(),
            "a token holds only letters, digits and - . _ ~ + /, with `=` only at the end",
        ),
        c if c.is_whitespace() => (
            format!("a whitespace character (U+{:04X})", c as u32),
            WHITESPACE,
        ),
        c if c.is_control() => (format!("a control character (U+{:04X})", c as u32), OTHER),
        c if !c.is_ascii() => (
            format!("a non-ASCII character (U+{:04X})", c as u32),
            NON_ASCII,
        ),
        c => (
            format!("the unsupported character U+{:04X}", c as u32),
            OTHER,
        ),
    }
}

/// The `Authorization` header value (`Bearer <token>`) for Hugging Face
/// downloads, from [`HF_TOKEN_ENV_VARS`] in the process environment. `Ok(None)`
/// when no variable holds a token. An invalid token is an error that names the
/// variable it came from; the next variable is not tried instead.
pub fn hf_authorization() -> Result<Option<String>> {
    hf_authorization_from(|name| std::env::var_os(name))
}

/// [`hf_authorization`] over an arbitrary variable lookup.
pub fn hf_authorization_from(lookup: impl Fn(&str) -> Option<OsString>) -> Result<Option<String>> {
    for name in HF_TOKEN_ENV_VARS {
        let Some(raw) = lookup(name).filter(|raw| !raw.is_empty()) else {
            continue;
        };
        let invalid = |message: String| ConfigError::InvalidValue {
            key: format!("environment variable {name}"),
            message: format!(
                "{message}. Fix or unset {name} to pull{}",
                if name == HF_TOKEN_ENV_VARS[0] {
                    ""
                } else {
                    " (it is used because HF_TOKEN is not set)"
                }
            ),
        };
        let Some(token) = raw.to_str() else {
            return Err(invalid("the token is not valid UTF-8".to_owned()));
        };
        validate_bearer_token(token).map_err(|problem| invalid(problem.to_string()))?;
        return Ok(Some(format!("Bearer {token}")));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const TOKEN: &str = "hf_AbCdEf0123456789-._~+/xyz==";

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, OsString> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), OsString::from(*v)))
            .collect();
        move |name| map.get(name).cloned()
    }

    fn error_text(pairs: &[(&str, &str)]) -> String {
        hf_authorization_from(env(pairs)).unwrap_err().to_string()
    }

    #[test]
    fn accepts_realistic_tokens() {
        assert_eq!(validate_bearer_token(TOKEN), Ok(()));
        assert_eq!(validate_bearer_token("hf_abc"), Ok(()));
        assert_eq!(
            hf_authorization_from(env(&[("HF_TOKEN", "hf_abc")])).unwrap(),
            Some("Bearer hf_abc".to_owned())
        );
    }

    #[test]
    fn unset_and_empty_mean_no_token() {
        assert_eq!(hf_authorization_from(env(&[])).unwrap(), None);
        assert_eq!(
            hf_authorization_from(env(&[("HF_TOKEN", ""), ("HUGGING_FACE_HUB_TOKEN", "")]))
                .unwrap(),
            None
        );
    }

    #[test]
    fn hf_token_wins_and_empty_falls_through() {
        let both = env(&[("HF_TOKEN", "hf_a"), ("HUGGING_FACE_HUB_TOKEN", "hf_b")]);
        assert_eq!(hf_authorization_from(both).unwrap().unwrap(), "Bearer hf_a");
        let fallback = env(&[("HF_TOKEN", ""), ("HUGGING_FACE_HUB_TOKEN", "hf_b")]);
        assert_eq!(
            hf_authorization_from(fallback).unwrap().unwrap(),
            "Bearer hf_b"
        );
    }

    #[test]
    fn rejects_whitespace_and_line_breaks_with_position() {
        for (value, what, position) in [
            ("hf_abc\n", "a newline", 7),
            ("hf_abc\r\n", "a carriage return", 7),
            ("hf_abc ", "a space", 7),
            (" hf_abc", "a space", 1),
            ("hf_a bc", "a space", 5),
            ("hf_a\tbc", "a tab", 5),
            ("   ", "a space", 1),
            ("hf_a\u{a0}bc", "a whitespace character (U+00A0)", 5),
        ] {
            let problem = validate_bearer_token(value).unwrap_err();
            assert_eq!(problem.what, what, "{value:?}");
            assert_eq!(problem.position, position, "{value:?}");
        }
    }

    #[test]
    fn rejects_quotes_non_ascii_and_control_characters() {
        let quoted = validate_bearer_token("\"hf_abc\"").unwrap_err();
        assert_eq!(quoted.what, "a double quote");
        assert!(quoted.hint.contains("quotation marks"));
        assert_eq!(
            validate_bearer_token("'hf_abc'").unwrap_err().what,
            "a single quote"
        );
        assert_eq!(
            validate_bearer_token("hf_é").unwrap_err().what,
            "a non-ASCII character (U+00E9)"
        );
        assert_eq!(
            validate_bearer_token("hf_\u{1F511}").unwrap_err().what,
            "a non-ASCII character (U+1F511)"
        );
        assert_eq!(
            validate_bearer_token("hf_\u{0}").unwrap_err().what,
            "a control character (U+0000)"
        );
        assert_eq!(
            validate_bearer_token("hf\u{7f}").unwrap_err().what,
            "a control character (U+007F)"
        );
        assert_eq!(
            validate_bearer_token("hf_a#b").unwrap_err().what,
            "the unsupported character U+0023"
        );
    }

    #[test]
    fn padding_only_at_the_end() {
        assert_eq!(validate_bearer_token("abc=="), Ok(()));
        let problem = validate_bearer_token("ab=c").unwrap_err();
        assert_eq!(problem.position, 4);
        assert_eq!(problem.what, "a character after the `=` padding");
    }

    #[test]
    fn error_names_the_variable_and_never_echoes_the_token() {
        let text = error_text(&[("HF_TOKEN", "hf_SECRETVALUE\n")]);
        assert!(text.contains("environment variable HF_TOKEN"), "{text}");
        assert!(text.contains("a newline at character 15"), "{text}");
        assert!(text.contains("Fix or unset HF_TOKEN"), "{text}");
        assert!(!text.contains("SECRETVALUE"), "{text}");

        let text = error_text(&[("HF_TOKEN", "hf_é_SECRETVALUE")]);
        assert!(!text.contains("SECRETVALUE"), "{text}");
        assert!(text.contains("non-ASCII character (U+00E9)"), "{text}");
    }

    #[test]
    fn fallback_variable_is_named_when_it_is_the_source() {
        let text = error_text(&[("HUGGING_FACE_HUB_TOKEN", "hf_a b")]);
        assert!(
            text.contains("environment variable HUGGING_FACE_HUB_TOKEN"),
            "{text}"
        );
        assert!(text.contains("a space at character 5"), "{text}");
        assert!(text.contains("because HF_TOKEN is not set"), "{text}");
    }

    #[test]
    fn invalid_primary_is_not_masked_by_a_valid_fallback() {
        let text = error_text(&[
            ("HF_TOKEN", "bad token"),
            ("HUGGING_FACE_HUB_TOKEN", "hf_ok"),
        ]);
        assert!(text.contains("environment variable HF_TOKEN"), "{text}");
        assert!(!text.contains("HUGGING_FACE_HUB_TOKEN"), "{text}");
    }

    #[test]
    fn rejects_non_utf8_values() {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let lookup = |name: &str| {
                (name == "HF_TOKEN").then(|| OsString::from_vec(vec![b'h', b'f', 0xff]))
            };
            let text = hf_authorization_from(lookup).unwrap_err().to_string();
            assert!(text.contains("environment variable HF_TOKEN"), "{text}");
            assert!(text.contains("not valid UTF-8"), "{text}");
        }
    }
}
