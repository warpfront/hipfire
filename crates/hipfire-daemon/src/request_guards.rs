//! Daemon-loop helpers kept out of main.rs: guards that answer or end the
//! process before a request runs, and small load/generate message readers.

use std::io::Write;
use std::path::PathBuf;

/// Qwen MTP head sidecar resolved by the CLI (models dir, then beside the path
/// as typed, then beside the canonical trunk). The load path is canonical, so
/// without this a sidecar beside a symlinked trunk is not found. Absent → the
/// loader looks for `<trunk>.mtp`. `mtp_mode=off` never carries one.
pub(crate) fn mtp_sidecar_path(msg: &serde_json::Value, mtp_mode: &str) -> Option<PathBuf> {
    msg.get("params")
        .and_then(|p| p.get("mtp"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty() && mtp_mode != "off")
        .map(PathBuf::from)
}

/// The client omitted `max_tokens`: routes fit the default to the context left
/// after the prompt instead of refusing it, for as long as the guard lives.
pub(crate) fn fit_max_tokens(
    msg: &serde_json::Value,
) -> hipfire_generate::common::FitMaxTokensGuard {
    let fit = msg.get("max_tokens_fit").and_then(|v| v.as_bool());
    hipfire_generate::common::FitMaxTokensGuard::set(fit == Some(true))
}

/// A sticky GPU fault (HipError 700/719) leaves the HIP context dead; only a
/// new process gets a live one. The request that hit it has already been
/// answered, so exit 75 (EX_TEMPFAIL): `hipfire serve` respawns the daemon
/// and reloads the model instead of failing every later request.
pub(crate) fn exit_if_gpu_poisoned(stdout: &mut impl Write) {
    if let Some(poison) = hipfire_runtime::reset_core::gpu_poison() {
        eprintln!(
            "[daemon] GPU context dead after sticky HipError({}) at {}; exiting (75) for a process restart",
            poison.code, poison.site,
        );
        let _ = stdout.flush();
        std::process::exit(75);
    }
}

/// Why a tensor-parallel (EP) request cannot be served: EP decode has no
/// penalty sampler and the EP load carries no vision tower, so an image or an
/// explicit non-neutral penalty is refused instead of silently dropped.
fn ep_refusal(msg: &serde_json::Value, has_image: bool) -> Option<String> {
    if has_image {
        return Some(
            "images are not supported at tp>1 (the tensor-parallel load has no vision tower)"
                .to_owned(),
        );
    }
    [
        ("repeat_penalty", 1.0),
        ("repetition_penalty", 1.0),
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
    ]
    .into_iter()
    .find(|(key, neutral)| {
        msg.get(*key)
            .and_then(|v| v.as_f64())
            .is_some_and(|value| value != *neutral)
    })
    .map(|(key, _)| {
        format!(
            "{key} is not supported at tp>1 (tensor-parallel decode has no penalty sampler); omit it or serve at tp=1"
        )
    })
}

/// At tp>1, answer a request [`ep_refusal`] rejects with an `unsupported`
/// error. Returns true when the request was refused and must be skipped.
pub(crate) fn refuse_ep_request(
    stdout: &mut impl Write,
    id: &str,
    msg: &serde_json::Value,
    ep: bool,
    has_image: bool,
) -> bool {
    let Some(message) = ep.then(|| ep_refusal(msg, has_image)).flatten() else {
        return false;
    };
    hipfire_generate::dense::emit_active_attempt_error(
        stdout,
        Some(id),
        &message,
        "unsupported",
        false,
        false,
    );
    let _ = stdout.flush();
    true
}

/// Suffix for an admitted-load failure. Single-device/pp loads are
/// unload-first, so after admission the prior model is already retired and a
/// construction or staging failure leaves `model=None`; say so explicitly.
/// A deferred tp>1 load keeps its prior model and gets no suffix.
pub(crate) fn no_model_loaded_suffix(model_present: bool) -> &'static str {
    if model_present {
        ""
    } else {
        "; no model loaded (the prior model was unloaded before this load)"
    }
}

#[cfg(test)]
mod tests {
    use super::no_model_loaded_suffix;

    #[test]
    fn admitted_load_failure_names_missing_model_only_when_unload_first() {
        // Unload-first (single-device/pp, incl. Qwen4): prior already retired.
        assert!(no_model_loaded_suffix(false).contains("no model loaded"));
        // Deferred tp>1: prior model survives a failed load.
        assert_eq!(no_model_loaded_suffix(true), "");
    }
}
