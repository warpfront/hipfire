# MTP context-floor prompt provenance

Generated 2026-10-07 for worktree revision 90d90306d837. Intended worktree tracking location: `benchmarks/prompts/mtpfloor/`; no git commit made.

Inputs:
- `capture-smallcap.request.json`: byte-for-byte copy of `/home/kaden/qcal/release-0.4.1/release-matrix/runs/halo-fn-90d90306d837/evals/tc-s123/http/00031/request.json`. Complete stock-price/calculator workload with system, user message and two tools retained for all tool contexts.
- `qwen-chat-reference.jinja`: byte-for-byte copy of `/home/kaden/ClaudeCode/warpfront/hipfire-mtpfloor/crates/hipfire-runtime/templates/eval/qwen35-official-reference.jinja`.
- `generate.py`: deterministic Python generator using Jinja2 3.1.6; run `python3 generate.py` from any directory. MD5SUMS covers all other artifact files including these inputs and provenance.

Initially generated tool inputs from capture 00055 (full meeting-booking history), then parent clarified capture selection could use any complete TC capture. Shortest complete capture containing tools was 00031: 2031 rendered UTF-8 bytes before padding. This permits approximate 1K without trimming. Original 00055 tool files were replaced only after runner confirmed they had not been consumed and explicitly requested fixed-basename replacement. Original text MD5s for historical identification: tool-1k 4361d983faf7706efbe94bc88dc4c391; tool-32k 5698ad428decdd6a161d539439740250; tool-128k af021cfffef2a456ff966f9cfdf78af3. These are obsolete, not current checksums. Code/prose prompts were not changed.

Existing construction inspected before choosing formatting: release-matrix/lib/decode.py sends plain prompt text and max_think_tokens=1; normal daemon applies the model's embedded Jinja template. Qwen generation uses JinjaChatFrame.render_messages for messages/tools. These `.txt` files explicitly render the checked-in Qwen official reference, including tool schema system block, XML function/parameter instructions and closed-think assistant generation prefix. Use `HIPFIRE_JINJA_CHAT=0` to avoid a second chat wrapper. Companion `.request.json` files alternatively preserve structured messages/tools for normal rendering. The reference is a declared formatting assumption, not a verified extraction of a model's embedded template. No inference or tokenizer runs were performed.

Captured system/messages/tools are retained exactly in tool request JSON; a single historical-background user message is inserted after system and before the original user. No captured turn is rewritten. The template trims message content. JSON-string tool arguments would be parsed into objects for template parameter iteration; capture 00031 has no historical assistant/tool turns. Tool calls are elicited in the Qwen XML format with JSON-valued compound parameters, rather than a conflicting standalone-JSON instruction.

Size targets are UTF-8 bytes = nominal tokens times four, including full rendered framing. No standalone tokenizer was discovered in matrix scripts; tokenizer lives in model metadata. Bytes/4 is only an explicit heuristic, especially unreliable for code/JSON, not a measured token count. See manifest.json for lengths. All three tool contexts use the same complete 00031 workload with deterministic padding. Code/prose end with long-continuation requests exceeding 256 tokens; absence of EOS is an elicitation goal, not measured or guaranteed. Actual token counts and context-window acceptance require runtime checks by the main agent.

Qcal and worktree artifact copies are identical.
