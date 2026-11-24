# hipfire-client — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Operator — synchronous typed transport for the daemon JSONL and OpenAI HTTP/SSE protocols; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Synchronous typed transport for the `hipfire-daemon` JSONL protocol (spawn, `flock`-aware engine handle, line framing, `Engine`/`load`/`generate`/`unload` verbs) plus OpenAI `chat/completions` HTTP/SSE client (`complete_openai_chat`, `stream_openai_chat`, `probe_host`/`service_ready`/`service_url`). Shared by `hipfire-cli`, serve, TUI, and Python tooling so process management is not duplicated. The `//!` docs in [`src/lib.rs`](src/lib.rs) are the contract.

## Gotchas

- Daemon is the GPU/runtime boundary — this crate must not embed model or quantization logic; it is transport only.
- Line framing is JSONL over `ChildStdin`/`BufReader`; partial-line buffering must not be mistaken for a completed response.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
