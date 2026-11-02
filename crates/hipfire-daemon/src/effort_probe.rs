//! Load-time `reasoning_effort` probe as published on the `loaded` ack.
//!
//! The runtime classifier (`probe_effort_capability`) renders the effective
//! chat template; this module adds the daemon-side cases it cannot see (no
//! tokenizer / no template) and the one-line diagnostic naming the template's
//! provenance, so a failed probe is observable instead of silently degrading
//! to `reasoning_effort_native: false`.

use hipfire_runtime::tokenizer::Tokenizer;

/// Outcome published in the `loaded` ack.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct EffortProbe {
    pub(crate) native: bool,
    pub(crate) efforts: Vec<&'static str>,
    /// `Some(reason)` iff the probe could not run or the baseline render
    /// failed; `None` when the probe ran, including templates that ignore
    /// effort.
    pub(crate) error: Option<String>,
}

impl EffortProbe {
    /// Non-QwenJinja contracts are never probed: nothing to report.
    pub(crate) fn not_probed() -> Self {
        Self {
            native: false,
            efforts: Vec::new(),
            error: None,
        }
    }

    /// The three `loaded`-ack fields every ack shape publishes, built from the
    /// one probe result so the single-slot and multi-slot acks cannot diverge.
    pub(crate) fn ack_fields(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut fields = serde_json::Map::new();
        fields.insert("reasoning_effort_native".into(), self.native.into());
        fields.insert("reasoning_efforts".into(), serde_json::json!(self.efforts));
        fields.insert(
            "reasoning_effort_probe_error".into(),
            serde_json::json!(self.error),
        );
        fields
    }

    /// Print the single failure diagnostic line when the probe failed.
    pub(crate) fn log_failure(&self, template_source: Option<&str>) {
        if let Some(error) = self.error.as_deref() {
            eprintln!("{}", probe_failure_line(error, template_source));
        }
    }
}

/// Probe the loaded model's effective template. A missing tokenizer or
/// template is a probe error (reported with its reason), never a silent
/// `native == false`.
pub(crate) fn probe_loaded_effort(
    tokenizer: Option<&Tokenizer>,
    template: Option<&str>,
) -> EffortProbe {
    match (tokenizer, template) {
        (Some(tokenizer), Some(template)) => {
            let cap = hipfire_runtime::prompt_frame::probe_effort_capability(tokenizer, template);
            EffortProbe {
                native: cap.native,
                efforts: cap.supported,
                error: cap.probe_error,
            }
        }
        (tokenizer, template) => {
            let missing = match (tokenizer.is_none(), template.is_none()) {
                (true, true) => "tokenizer and chat template",
                (true, false) => "tokenizer",
                _ => "chat template",
            };
            EffortProbe {
                native: false,
                efforts: Vec::new(),
                error: Some(format!(
                    "reasoning_effort probe not run: model has no {missing}"
                )),
            }
        }
    }
}

/// Ordinary-ack probe: probes only the `qwen_jinja` contract (every other
/// contract reports false/[]/null), logs a failed probe once with the
/// effective template source, and returns the ack's
/// `(reasoning_effort_native, reasoning_efforts, reasoning_effort_probe_error)`
/// values already serialized as JSON.
pub(crate) fn loaded_ack_parts(
    reasoning_contract: &str,
    tokenizer: Option<&Tokenizer>,
    template: Option<&str>,
    template_source: Option<&str>,
) -> (bool, String, String) {
    let probe = if reasoning_contract == "qwen_jinja" {
        probe_loaded_effort(tokenizer, template)
    } else {
        EffortProbe::not_probed()
    };
    probe.log_failure(template_source);
    (
        probe.native,
        serde_json::to_string(&probe.efforts).unwrap_or_else(|_| "[]".to_string()),
        serde_json::to_string(&probe.error).expect("effort probe error is serializable"),
    )
}

/// The single diagnostic line for a failed probe: the error (flattened to one
/// line) and the effective template source.
pub(crate) fn probe_failure_line(error: &str, template_source: Option<&str>) -> String {
    let mut line = String::with_capacity(96 + error.len() + template_source.map_or(4, str::len));
    line.push_str("[reasoning_effort_probe] failed: ");
    push_flat(&mut line, error);
    line.push_str(" (chat_template source: ");
    match template_source {
        Some(source) => push_flat(&mut line, source),
        None => line.push_str("none"),
    }
    line.push(')');
    line
}

/// Append `text` with every whitespace run (newlines included) collapsed to a
/// single space and no leading/trailing space, keeping the log to one line.
fn push_flat(out: &mut String, text: &str) {
    let mut words = text.split_whitespace();
    if let Some(first) = words.next() {
        out.push_str(first);
        for word in words {
            out.push(' ');
            out.push_str(word);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_tokenizer_or_template_is_a_probe_error_with_reason() {
        let both = probe_loaded_effort(None, None);
        assert!(!both.native && both.efforts.is_empty());
        assert_eq!(
            both.error.as_deref(),
            Some("reasoning_effort probe not run: model has no tokenizer and chat template")
        );
        let no_tokenizer = probe_loaded_effort(None, Some("{{ messages }}"));
        assert_eq!(
            no_tokenizer.error.as_deref(),
            Some("reasoning_effort probe not run: model has no tokenizer")
        );
    }

    #[test]
    fn not_probed_reports_no_error() {
        let probe = EffortProbe::not_probed();
        assert_eq!(probe.error, None);
        assert!(!probe.native && probe.efforts.is_empty());
    }

    #[test]
    fn failure_line_is_one_line_with_error_and_source() {
        let line = probe_failure_line(
            "render failed:\n  undefined value\n",
            Some("user override: per-model template /home/u/.hipfire/templates/m.mq6.j2"),
        );
        assert!(!line.contains('\n'));
        assert_eq!(
            line,
            "[reasoning_effort_probe] failed: render failed: undefined value (chat_template source: user override: per-model template /home/u/.hipfire/templates/m.mq6.j2)"
        );
        assert!(probe_failure_line("e", None).ends_with("(chat_template source: none)"));
    }

    #[test]
    fn failure_line_flattens_a_source_path_containing_newlines() {
        let line = probe_failure_line(
            "e",
            Some("user override: per-model template /home/u/.hipfire/templates/a\nb.j2\r\n"),
        );
        assert!(!line.contains('\n') && !line.contains('\r'));
        assert_eq!(
            line,
            "[reasoning_effort_probe] failed: e (chat_template source: user override: per-model template /home/u/.hipfire/templates/a b.j2)"
        );
    }

    /// Byte-exact `tokenizer_config.chat_template` shipped in the registry
    /// `qwen3.8:27b-mq6` artifact (see runtime `prompt_frame` probe tests).
    const QWEN38_27B_MQ6_REAL: &str =
        include_str!("../../hipfire-runtime/templates/eval/qwen38-27b-mq6-official.jinja");

    /// The probe only renders the template; it never encodes, so a
    /// special-token-only vocabulary is enough.
    fn probe_tokenizer() -> Tokenizer {
        Tokenizer::from_hf_json(
            r#"{
                "model": {"type": "BPE", "vocab": {
                    "<|im_start|>": 0, "<|im_end|>": 1, "<think>": 2, "</think>": 3,
                    "system": 4, "user": 5, "assistant": 6
                }, "merges": []},
                "added_tokens": [
                    {"id": 0, "content": "<|im_start|>", "special": true},
                    {"id": 1, "content": "<|im_end|>", "special": true}
                ]
            }"#,
        )
        .expect("hermetic probe tokenizer")
    }

    /// Regression for the multi-slot ack that hardcoded `false`/`[]`: the
    /// slot ack is built from `ack_fields()` of the same probe the ordinary
    /// path runs, and for the real MQ6 template it must advertise native
    /// low/medium/xhigh with a null error.
    #[test]
    fn real_mq6_template_ack_fields_are_native_low_medium_xhigh() {
        let tokenizer = probe_tokenizer();
        let probe = probe_loaded_effort(Some(&tokenizer), Some(QWEN38_27B_MQ6_REAL));
        let ordinary =
            hipfire_runtime::prompt_frame::probe_effort_capability(&tokenizer, QWEN38_27B_MQ6_REAL);
        assert_eq!(probe.native, ordinary.native);
        assert_eq!(probe.efforts, ordinary.supported);
        assert_eq!(probe.error, ordinary.probe_error);

        let fields = probe.ack_fields();
        assert_eq!(fields["reasoning_effort_native"], serde_json::json!(true));
        assert_eq!(
            fields["reasoning_efforts"],
            serde_json::json!(["low", "medium", "xhigh"])
        );
        assert_eq!(
            fields["reasoning_effort_probe_error"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn failed_probe_ack_fields_carry_the_reason_not_null() {
        let fields = probe_loaded_effort(None, None).ack_fields();
        assert_eq!(fields["reasoning_effort_native"], serde_json::json!(false));
        assert_eq!(fields["reasoning_efforts"], serde_json::json!([]));
        assert!(fields["reasoning_effort_probe_error"].is_string());
    }
}
