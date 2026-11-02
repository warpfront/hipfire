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

/// DFlash2 proposal width of a load (`params.dflash_block`,
/// `speculation.dflash_block`): one width per serving configuration, resolved
/// from the staged route (`vmm_batch`, `lanes` = `continuous_batch_size`) and
/// never from the live batch, so batched lanes stay byte-identical to the
/// singleton. Also whether the width was `auto` (a producer may keep its own
/// default then). An invalid value is answered as a validation error (`None`).
pub(crate) fn dflash_block(
    stdout: &mut impl Write,
    msg: &serde_json::Value,
    vmm_batch: bool,
    lanes: usize,
) -> Option<(usize, bool)> {
    match hipfire_runtime::dflash::DflashBlockPolicy::from_param(
        msg.get("params").and_then(|p| p.get("dflash_block")),
    ) {
        Ok(policy) => Some((
            policy.resolve(vmm_batch, lanes),
            policy == hipfire_runtime::dflash::DflashBlockPolicy::Auto,
        )),
        Err(e) => {
            hipfire_generate::batch::emit_uncorrelated_error(
                stdout,
                None,
                &e,
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            None
        }
    }
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

/// Why an EP request cannot be served: the EP load carries no vision tower,
/// and the EP archs without a penalty stage (`penalties_supported` false) samples without a
/// penalty stage, so an image or a non-neutral penalty is refused instead of
/// silently dropped.
fn ep_refusal(
    msg: &serde_json::Value,
    has_image: bool,
    penalties_supported: bool,
) -> Option<String> {
    if has_image {
        return Some(
            "images are not supported at tp>1 (the tensor-parallel load has no vision tower)"
                .to_owned(),
        );
    }
    if penalties_supported {
        return None;
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
            "{key} is not supported at tp>1 for this architecture (its tp>1 decode has no penalty sampler); omit it or serve at tp=1"
        )
    })
}

/// At tp>1, answer a request [`ep_refusal`] rejects with an `unsupported`
/// error. Returns true when the request was refused and must be skipped.
/// `ep`: the load is expert/tensor parallel; `penalties_supported`: its serve
/// loop applies penalties ([`hipfire_loader::EpArch::applies_request_penalties`]).
pub(crate) fn refuse_ep_request(
    stdout: &mut impl Write,
    id: &str,
    msg: &serde_json::Value,
    ep: bool,
    penalties_supported: bool,
    has_image: bool,
) -> bool {
    let Some(message) = ep
        .then(|| ep_refusal(msg, has_image, penalties_supported))
        .flatten()
    else {
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

#[cfg(test)]
mod tests {
    use super::ep_refusal;
    use serde_json::json;

    #[test]
    fn a_penalty_is_refused_only_where_the_tp_loop_has_no_penalty_stage() {
        let msg = json!({ "presence_penalty": 1.5, "repeat_penalty": 1.05 });
        let refusal = ep_refusal(&msg, false, false).expect("ds4/minimax refuse a penalty");
        assert!(refusal.starts_with("repeat_penalty is not supported at tp>1"));
        assert_eq!(ep_refusal(&msg, false, true), None);
    }

    #[test]
    fn neutral_penalties_pass_everywhere() {
        let msg = json!({
            "repeat_penalty": 1.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        });
        assert_eq!(ep_refusal(&msg, false, false), None);
    }

    #[test]
    fn an_image_is_refused_at_tp_even_when_penalties_are_supported() {
        assert!(ep_refusal(&json!({}), true, true)
            .expect("image refused")
            .starts_with("images are not supported at tp>1"));
    }
}
