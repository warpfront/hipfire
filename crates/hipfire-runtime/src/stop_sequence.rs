// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! OpenAI `stop` sequences: request parsing and a streaming matcher.
//!
//! [`parse_stop_field`] accepts both wire forms (`"stop": "\n"` and
//! `"stop": ["END", "\n\n"]`) and rejects values the decode loops cannot honour
//! instead of silently truncating them.
//!
//! [`StopMatcher`] runs on the client-visible answer text *before* it is
//! emitted. It holds back the longest tail that could still grow into a stop
//! sequence, so a stop that spans a token boundary is caught and the stop text
//! itself never reaches the client (llama.cpp and vLLM both exclude it).

use std::borrow::Cow;

/// Most stop sequences one request may carry (OpenAI's documented limit).
pub const MAX_STOP_SEQUENCES: usize = 4;
/// Longest stop sequence, in characters.
pub const MAX_STOP_CHARS: usize = 64;

/// Parse the OpenAI `stop` field: absent/`null`, one string, or an array of
/// strings. Empty strings are ignored. Anything else is a validation error;
/// the message starts with `invalid stop` so every gateway maps it to 400.
pub fn parse_stop_field(value: Option<&serde_json::Value>) -> Result<Vec<String>, String> {
    let entries: &[serde_json::Value] = match value {
        None | Some(serde_json::Value::Null) => return Ok(Vec::new()),
        Some(one @ serde_json::Value::String(_)) => std::slice::from_ref(one),
        Some(serde_json::Value::Array(arr)) => arr,
        Some(_) => {
            return Err("invalid stop: expected a string or an array of strings".to_owned());
        }
    };
    if entries.len() > MAX_STOP_SEQUENCES {
        return Err(format!(
            "invalid stop: at most {MAX_STOP_SEQUENCES} sequences are allowed, got {}",
            entries.len()
        ));
    }
    let mut stops = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(s) = entry.as_str() else {
            return Err("invalid stop: every entry must be a string".to_owned());
        };
        if s.chars().count() > MAX_STOP_CHARS {
            return Err(format!(
                "invalid stop: each sequence must be at most {MAX_STOP_CHARS} characters"
            ));
        }
        if !s.is_empty() {
            stops.push(s.to_owned());
        }
    }
    Ok(stops)
}

/// Streaming stop-sequence matcher over client-visible text.
///
/// Feed each chunk through [`push`](Self::push) and emit what it returns.
/// After a match, `push` returns the text before the earliest stop sequence
/// and [`matched`](Self::matched) latches; everything after it is dropped. At
/// end of stream without a match, [`finish`](Self::finish) releases the held
/// tail. With no stop sequences configured, `push` borrows its input and never
/// holds anything back.
#[derive(Debug, Clone, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    /// Longest stop sequence, in bytes (bounds the held tail).
    max_len: usize,
    /// Tail of the stream that is a proper prefix of some stop sequence.
    held: String,
    matched: bool,
}

impl StopMatcher {
    pub fn new(stops: &[String]) -> Self {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let max_len = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            max_len,
            held: String::new(),
            matched: false,
        }
    }

    /// Whether any stop sequence is configured.
    pub fn is_active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// Whether a stop sequence has matched; the stream is over once true.
    pub fn matched(&self) -> bool {
        self.matched
    }

    /// Feed visible text; returns the part that is now safe to emit.
    pub fn push<'a>(&mut self, text: &'a str) -> Cow<'a, str> {
        if self.matched {
            return Cow::Borrowed("");
        }
        if self.stops.is_empty() {
            return Cow::Borrowed(text);
        }
        self.held.push_str(text);
        let earliest = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()))
            .min();
        if let Some(at) = earliest {
            self.matched = true;
            self.held.truncate(at);
            return Cow::Owned(std::mem::take(&mut self.held));
        }
        let keep = self.partial_stop_suffix_len();
        let tail = self.held.split_off(self.held.len() - keep);
        Cow::Owned(std::mem::replace(&mut self.held, tail))
    }

    /// End of stream: the held tail when no stop matched, else nothing.
    pub fn finish(&mut self) -> String {
        if self.matched {
            return String::new();
        }
        std::mem::take(&mut self.held)
    }

    /// Byte length of the longest suffix of `held` that is a proper prefix of
    /// some stop sequence. Every such suffix starts on a char boundary, because
    /// the stop sequence it prefixes is valid UTF-8 starting at a char.
    fn partial_stop_suffix_len(&self) -> usize {
        let longest = self.held.len().min(self.max_len.saturating_sub(1));
        (1..=longest)
            .rev()
            .find(|&k| {
                let start = self.held.len() - k;
                self.held.is_char_boundary(start) && {
                    let suffix = &self.held[start..];
                    self.stops
                        .iter()
                        .any(|stop| stop.len() > k && stop.starts_with(suffix))
                }
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn stops(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Feed `chunks` and return (emitted text, matched).
    fn run(stop_list: &[&str], chunks: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(&stops(stop_list));
        let mut out = String::new();
        for chunk in chunks {
            out.push_str(&m.push(chunk));
            if m.matched() {
                break;
            }
        }
        out.push_str(&m.finish());
        (out, m.matched())
    }

    #[test]
    fn parse_accepts_string_and_array_forms() {
        assert_eq!(parse_stop_field(None), Ok(vec![]));
        assert_eq!(parse_stop_field(Some(&json!(null))), Ok(vec![]));
        assert_eq!(parse_stop_field(Some(&json!("\n"))), Ok(stops(&["\n"])));
        assert_eq!(
            parse_stop_field(Some(&json!(["END", "", "\n\n"]))),
            Ok(stops(&["END", "\n\n"]))
        );
        assert_eq!(parse_stop_field(Some(&json!(""))), Ok(vec![]));
    }

    #[test]
    fn parse_rejects_what_the_decoder_cannot_honour() {
        for bad in [
            json!(["a", "b", "c", "d", "e"]),
            json!(["ok", 7]),
            json!(7),
            json!({"stop": "x"}),
            json!("x".repeat(MAX_STOP_CHARS + 1)),
        ] {
            let err = parse_stop_field(Some(&bad)).expect_err("must reject");
            assert!(err.starts_with("invalid stop"), "{bad}: {err}");
        }
        // The limit is in characters, not bytes.
        let cjk = "中".repeat(MAX_STOP_CHARS);
        assert_eq!(parse_stop_field(Some(&json!(cjk))), Ok(vec![cjk]));
    }

    #[test]
    fn stop_text_is_never_emitted() {
        assert_eq!(run(&["END"], &["Hello END world"]), ("Hello ".into(), true));
        assert_eq!(run(&["\n"], &["line one\nline two"]), ("line one".into(), true));
    }

    #[test]
    fn stop_split_across_chunks_is_held_back_then_trimmed() {
        let mut m = StopMatcher::new(&stops(&["END"]));
        assert_eq!(m.push("abc E"), "abc ");
        assert_eq!(m.push("N"), "");
        assert_eq!(m.push("D tail"), "");
        assert!(m.matched());
        assert_eq!(m.push("more"), "");
        assert_eq!(m.finish(), "");
    }

    #[test]
    fn false_prefix_is_released_once_disambiguated() {
        let mut m = StopMatcher::new(&stops(&["END"]));
        assert_eq!(m.push("E"), "");
        assert_eq!(m.push("N"), "");
        assert_eq!(m.push("x"), "ENx");
        assert!(!m.matched());
        // A dangling prefix at end of stream is ordinary text.
        assert_eq!(m.push("EN"), "");
        assert_eq!(m.finish(), "EN");
    }

    #[test]
    fn holdback_never_exceeds_longest_stop_minus_one() {
        let mut m = StopMatcher::new(&stops(&["abcdef", "xy"]));
        assert_eq!(m.push("zzzabcde"), "zzz");
        assert_eq!(m.held.len(), 5);
        assert_eq!(m.push("Q"), "abcdeQ");
        assert!(m.held.is_empty());
    }

    #[test]
    fn earliest_of_several_stops_wins() {
        assert_eq!(
            run(&["world", "lo"], &["hello world"]),
            ("hel".into(), true)
        );
    }

    #[test]
    fn multibyte_stop_and_text() {
        assert_eq!(run(&["。"], &["你好", "世界。再见"]), ("你好世界".into(), true));
        // Held tail is a whole char, never a split code point.
        let mut m = StopMatcher::new(&stops(&["🎉!"]));
        assert_eq!(m.push("hi 🎉"), "hi ");
        assert_eq!(m.push("?"), "🎉?");
    }

    #[test]
    fn inactive_matcher_borrows_and_holds_nothing() {
        let mut m = StopMatcher::new(&[]);
        assert!(!m.is_active());
        assert!(matches!(m.push("abc"), Cow::Borrowed("abc")));
        assert_eq!(m.finish(), "");
        assert!(!StopMatcher::new(&stops(&[""])).is_active());
    }
}
