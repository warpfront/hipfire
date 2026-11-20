//! A CRLF/NBSP Glimmer request renders the same prompt ids on the singleton
//! route (daemon normalization + `glimmer_prompt_ids`) and on a batch lane
//! (`render_glimmer_lane_prompt`). Needs the Glimmer fixture's tokenizer and
//! chat template only, no GPU.

const DEFAULT_FIXTURE: &str = "muse-glimmer-30b.mq4";

#[test]
#[ignore = "needs a Glimmer .mq4 (HIPFIRE_GLIMMER_MODEL or ~/.hipfire/models/muse-glimmer-30b.mq4)"]
fn crlf_nbsp_prompt_renders_identically_on_singleton_and_lane() {
    let path = std::env::var("HIPFIRE_GLIMMER_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.hipfire/models/{DEFAULT_FIXTURE}",
            std::env::var("HOME").unwrap()
        )
    });
    let hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&path))
        .unwrap_or_else(|e| panic!("open {path}: {e}"));
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).unwrap();
    let template = hfq.chat_template();
    assert!(template.is_some(), "fixture has no embedded chat template");
    let bos = 2;

    let unused = std::cell::Cell::new(0);
    let singleton = |prompt: &str, messages: Option<&[_]>| {
        hipfire_generate::dense::glimmer_prompt_ids(
            &tokenizer,
            template.as_ref(),
            &mut hipfire_loader::AsstTurnCache::new_from_env(),
            bos,
            prompt,
            None,
            None,
            0,
            None,
            messages,
            &unused,
            &unused,
        )
        .unwrap()
    };

    let raw_prompt = "line one\r\nline\u{a0}two";
    let bare = serde_json::json!({ "prompt": raw_prompt });
    let with_messages = serde_json::json!({
        "prompt": raw_prompt,
        "messages": [
            {"role": "user", "content": "first\r\nturn\u{a0}here"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": raw_prompt},
        ],
    });
    for msg in [bare, with_messages] {
        // Singleton: what the daemon hands `generate_muse_glimmer`.
        let prompt = hipfire_engine::prompt::generate_prompt(&msg);
        let messages = hipfire_engine::prompt::parse_generate_messages(&msg).unwrap();
        let singleton_ids = singleton(&prompt, messages.as_deref());

        let lane_ids = hipfire_generate::lanes::render_glimmer_lane_prompt(
            &tokenizer,
            template.as_ref(),
            bos,
            &msg,
        )
        .unwrap();
        assert_eq!(lane_ids, singleton_ids, "{msg}");

        // Not vacuous: the un-normalized render differs.
        let raw_messages: Option<Vec<hipfire_runtime::prompt_frame::Message>> = msg
            .get("messages")
            .map(|v| serde_json::from_value(v.clone()).unwrap());
        assert_ne!(
            singleton(raw_prompt, raw_messages.as_deref()),
            singleton_ids,
            "{msg}"
        );
    }
}
