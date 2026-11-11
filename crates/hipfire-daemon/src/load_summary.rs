//! The one-line `hipfire load summary:` and its `loaded`-ack twin.
//!
//! Pure formatting: the engine's own decision sites record *why* something is
//! on or off through `hipfire_config::load_summary::note`; this module only
//! joins those notes with the facts the daemon already holds (KV mode and
//! backend, admitted context, speculator kind). A facet with no note gets an
//! honest default derived from daemon state, never an invented reason.

use hipfire_config::load_summary::LoadNotes;
use serde_json::{json, Value};

/// Daemon-held facts about the model that just loaded.
pub(crate) struct SummaryState<'a> {
    /// Model file name (no directory).
    pub model: &'a str,
    pub arch: &'a str,
    /// Resolved KV mode (`seq_kv`).
    pub kv_mode: &'a str,
    /// Resident KV backend (`vmm` / `legacy`).
    pub kv_backend: &'a str,
    /// `automatic` or `explicit` (`ResidentKvDiag::request`).
    pub kv_request: &'a str,
    pub max_seq: usize,
    /// `user`, `model`, `card`, `legacy`, ...
    pub bound: &'a str,
    pub model_ctx: Option<usize>,
    pub card_cap: Option<usize>,
    /// The generic speculator the loaded model carries, when it names one.
    pub speculator: Option<&'a str>,
    pub pp: usize,
    /// Expert-parallel batch route.
    pub ep: bool,
}

/// The formatted summary: the log line and the per-facet strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoadSummary {
    pub line: String,
    pub spec: String,
    pub kv: String,
    pub context: String,
    pub placement: String,
    pub prefill: String,
}

impl LoadSummary {
    /// The `load_summary` object of the `loaded` ack.
    pub(crate) fn to_json(&self) -> Value {
        json!({
            "line": self.line,
            "spec": self.spec,
            "kv": self.kv,
            "context": self.context,
            "placement": self.placement,
            "prefill": self.prefill,
        })
    }
}

fn spec_facet(notes: &LoadNotes, state: &SummaryState<'_>) -> String {
    if let Some(on) = &notes.spec_on {
        return match &notes.copy_lane {
            Some(lane) => format!("{on}, {lane}"),
            None => on.clone(),
        };
    }
    if let Some(name) = state.speculator {
        return name.to_string();
    }
    if notes.spec_off.is_empty() {
        "off (no speculator armed)".to_string()
    } else {
        format!("off ({})", notes.spec_off.join("; "))
    }
}

fn context_facet(notes: &LoadNotes, state: &SummaryState<'_>) -> String {
    let limits = match (state.model_ctx, state.card_cap) {
        (Some(model), Some(card)) => format!("model {model}, card {card}"),
        _ => "model/card limits unknown".to_string(),
    };
    let mut detail = match state.bound {
        "user" => format!("user-set; {limits}"),
        "model" | "card" => match (state.model_ctx, state.card_cap) {
            (Some(model), Some(card)) => {
                format!("auto: min(model {model}, card {card})")
            }
            _ => format!("auto: bound by {}", state.bound),
        },
        other => format!("bound={other}; {limits}"),
    };
    if let Some(fit) = &notes.context_fit {
        detail.push_str("; ");
        detail.push_str(fit);
    }
    format!("{} ({detail})", state.max_seq)
}

fn placement_facet(notes: &LoadNotes, state: &SummaryState<'_>) -> String {
    if !notes.placement.is_empty() {
        return notes.placement.join("; ");
    }
    if state.pp > 1 {
        format!("pipeline-parallel x{}", state.pp)
    } else if state.ep {
        "expert-parallel".to_string()
    } else {
        "fully resident".to_string()
    }
}

fn prefill_facet(notes: &LoadNotes) -> String {
    if notes.prefill.is_empty() {
        "per-prompt chunking".to_string()
    } else {
        notes.prefill.join("; ")
    }
}

/// Join the engine's notes with daemon state into the summary line + facets.
pub(crate) fn summarize(notes: &LoadNotes, state: &SummaryState<'_>) -> LoadSummary {
    let spec = spec_facet(notes, state);
    // The carrier's resolved layout (e.g. `fp8` for a requested `auto`) wins
    // over the daemon's requested mode.
    let kv = format!(
        "{} ({} backend, {})",
        notes.kv.as_deref().unwrap_or(state.kv_mode),
        state.kv_backend,
        state.kv_request
    );
    let context = context_facet(notes, state);
    let placement = placement_facet(notes, state);
    let prefill = prefill_facet(notes);
    let line = format!(
        "hipfire load summary: {} ({}) | spec: {spec} | kv: {kv} | context: {context} | placement: {placement} | prefill: {prefill}",
        state.model, state.arch
    );
    LoadSummary {
        line,
        spec,
        kv,
        context,
        placement,
        prefill,
    }
}

/// Add `load_summary` to a serialized `loaded` ack. A body that is not a JSON
/// object is returned unchanged rather than dropped.
pub(crate) fn ack_with_summary(ack: String, summary: &LoadSummary) -> String {
    let Ok(mut value) = serde_json::from_str::<Value>(&ack) else {
        return ack;
    };
    let Some(object) = value.as_object_mut() else {
        return ack;
    };
    object.insert("load_summary".to_string(), summary.to_json());
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_config::load_summary::{begin, note, take, Facet};

    fn state<'a>() -> SummaryState<'a> {
        SummaryState {
            model: "qwen3.5-9b.mq4",
            arch: "qwen3_5",
            kv_mode: "fp8",
            kv_backend: "vmm",
            kv_request: "automatic",
            max_seq: 131072,
            bound: "card",
            model_ctx: Some(262144),
            card_cap: Some(131072),
            speculator: None,
            pp: 1,
            ep: false,
        }
    }

    // The notes store is process-global; tests that fill it hold this lock.
    static STORE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn qwen35_mtp_on() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        begin();
        note(Facet::SpecOn, "mtp K=3");
        note(Facet::ContextFit, "measured fit unchanged");
        let summary = summarize(&take(), &state());
        assert_eq!(
            summary.line,
            "hipfire load summary: qwen3.5-9b.mq4 (qwen3_5) | spec: mtp K=3 | kv: fp8 (vmm backend, automatic) | context: 131072 (auto: min(model 262144, card 131072); measured fit unchanged) | placement: fully resident | prefill: per-prompt chunking"
        );
        assert_eq!(summary.to_json()["spec"], "mtp K=3");
        assert_eq!(
            summary.to_json()["line"],
            Value::String(summary.line.clone())
        );
    }

    #[test]
    fn qwen4_mtp_off_for_host_mapped_experts() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        begin();
        note(
            Facet::SpecOff,
            "MTP off: 96 expert tensors host-mapped on gfx1100 (MTP: off by default with host-mapped experts; opt in with --spec mtp)",
        );
        note(
            Facet::Placement,
            "experts: layers 0..30 in VRAM, 96 tensors (12.4 GiB) host-mapped",
        );
        note(Facet::Prefill, "chunk 512 rows");
        note(
            Facet::Prefill,
            "long-prefill expert staging on (>= 64 rows)",
        );
        let mut st = state();
        st.model = "qwen3.8-flash-next.mq4";
        st.arch = "qwen4";
        st.bound = "user";
        st.max_seq = 32768;
        let summary = summarize(&take(), &st);
        assert_eq!(
            summary.line,
            "hipfire load summary: qwen3.8-flash-next.mq4 (qwen4) | spec: off (MTP off: 96 expert tensors host-mapped on gfx1100 (MTP: off by default with host-mapped experts; opt in with --spec mtp)) | kv: fp8 (vmm backend, automatic) | context: 32768 (user-set; model 262144, card 131072) | placement: experts: layers 0..30 in VRAM, 96 tensors (12.4 GiB) host-mapped | prefill: chunk 512 rows; long-prefill expert staging on (>= 64 rows)"
        );
    }

    #[test]
    fn ar_with_no_notes_uses_honest_defaults() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        begin();
        let mut st = state();
        st.model = "llama-8b.mq4";
        st.arch = "llama";
        st.kv_mode = "q8";
        st.kv_backend = "legacy";
        st.kv_request = "explicit";
        st.bound = "legacy";
        st.max_seq = 4096;
        st.model_ctx = None;
        st.card_cap = None;
        let summary = summarize(&take(), &st);
        assert_eq!(
            summary.line,
            "hipfire load summary: llama-8b.mq4 (llama) | spec: off (no speculator armed) | kv: q8 (legacy backend, explicit) | context: 4096 (bound=legacy; model/card limits unknown) | placement: fully resident | prefill: per-prompt chunking"
        );
    }

    #[test]
    fn speculator_name_is_the_fallback_when_no_site_noted_one() {
        let mut st = state();
        st.speculator = Some("dflash");
        assert_eq!(summarize(&LoadNotes::default(), &st).spec, "dflash");
    }

    #[test]
    fn copy_lane_rides_with_spec_on_and_parallel_topologies_are_named() {
        let notes = LoadNotes {
            spec_on: Some("qwen4 native mtp K=3".into()),
            copy_lane: Some("n-gram copy lane on".into()),
            ..LoadNotes::default()
        };
        let mut st = state();
        st.pp = 2;
        let summary = summarize(&notes, &st);
        assert_eq!(summary.spec, "qwen4 native mtp K=3, n-gram copy lane on");
        assert_eq!(summary.placement, "pipeline-parallel x2");
        st.pp = 1;
        st.ep = true;
        assert_eq!(summarize(&notes, &st).placement, "expert-parallel");
    }

    #[test]
    fn ack_gains_a_load_summary_object() {
        let summary = summarize(&LoadNotes::default(), &state());
        let ack = ack_with_summary(
            r#"{"type":"loaded","arch":"qwen3_5"}"#.to_string(),
            &summary,
        );
        let value: Value = serde_json::from_str(&ack).unwrap();
        assert_eq!(value["type"], "loaded");
        assert_eq!(value["load_summary"]["kv"], "fp8 (vmm backend, automatic)");
        assert_eq!(value["load_summary"]["line"], Value::String(summary.line));
        // Not an object: unchanged.
        assert_eq!(
            ack_with_summary(
                "nope".to_string(),
                &summarize(&LoadNotes::default(), &state())
            ),
            "nope"
        );
    }
}
