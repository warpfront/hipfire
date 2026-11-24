# hipfire-generate — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Generation — above `hipfire-engine` and `hipfire-loader`, below `hipfire-daemon`; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Per-architecture generation bodies lifted out of `crates/hipfire-daemon/src/main.rs` so the product binary can stay arch-free. Owns the `generate` ladder (`ar`, `qwen`, `dense`, `vision`, `batch`, plus `common` shared helpers) that dispatches on `LoadedModel`/`arch_id` to concrete arch forwards and spec-decode policies. The `//!` docs in [`src/lib.rs`](src/lib.rs) explain why `hipfire-engine` (arch-free) and `hipfire-loader` (arch-carrying) both sit below it.

## Gotchas

- Modules are family-partitioned so each has one owning agent — families depend on `common` and never on each other.
- `src/lib.rs` and `Cargo.toml` are scaffold-owned and belong to no agent; do not claim them for a family change.
- Speculative session reuse in `qwen.rs` is family-agnostic; no family branch on the path. A model opts in with `ArchModel::spec_session_reuse` and a speculator names its route with `Speculator::session_route` (Qwen4 native MTP -> `Mtp`, DFlash2 -> `Ar`; `spec_session_route`). `generate_dflash` then plans through `plan_spec_session` (`session_cache::plan_resumable` over `ArchModel::session_plan_capped`: the deepest reuse point `can_resume_at` accepts, else cold), `spec_prefill` begins an `Ar` route through `SpecTarget::session_begin` (an `Mtp` drafter begins it itself), and `commit_session` publishes through `ArchModel::session_commit_live` / `session_commit_live_mesh` at a committed terminal. `generate_qwen4_ar` asks `plan_qwen4_session` (PP mesh aware; AR resumes any reuse point). Models without spec session reuse keep the conversation plans; `ar.rs` and the Qwen3.5 path are unchanged.
- Qwen assistant-turn splicing keys normalized content and tool calls together with the producer's router-emitted reasoning text. Echoed nonempty `reasoning_content` selects an exact turn; missing/empty reasoning uses the content/tool index only when exactly one cached turn matches. Safe completed open-think terminals store the same whole-body content slot and text-only reasoning marker as closed thinking turns, so history does not synthesize a close inside their generated tokens. AR, MTP and DFlash `length` terminals with a complete call-less body (open-think length included) store the emitted token body under that same key and reasoning; the template's per-turn `im_end` framing is retained and no synthetic think close is inserted. Truncated or malformed tool output, suppressed hidden completed calls (not describable in the echo key; no calls are released at `length`), stop-sequence, abort and fail-closed turns remain unstored.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
