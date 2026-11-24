# hipfire-cli — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Operator — native control plane and HTTP service, above the daemon; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Native operator for `hipfire run` (tag → `~/.hipfire/models/` path, forced-local policy, one-shot daemon spawn vs `HTTP POST /v1/chat/completions` when serve is up). Owns `hipfire run/serve/pull/diag` surfaces, typed config/registry integration, and long-running serve/attach state. The `//!` docs in [`src/main.rs`](src/main.rs) are authoritative; this binary never shells out to JS/TS.

## Gotchas

- Forced-local gate (`HIPFIRE_LOCAL=1`, `--kv-mode`, `--json`, `--no-stream`) — an HTTP failure while serve is live must abort, not silently spawn a local daemon that collides on VRAM.
- Registry fetch preserves `fresh cache → network → stale cache → bundled` order; a malformed registry is rejected wholesale.
- `/v1/embeddings` input lowering is `src/serve/embeddings.rs` (`lower_embeddings_input`): a bare string, a one-element list and the object form (`{text, image_base64}` / `{text, images_base64}` / `{text, audios_base64}`) lower byte-identically to what the daemon saw before, the typed `[{"content": […]}]` form decodes its `image_url` and `input_audio` parts **here** (base64 or data URL — the route's one decode point; `input_audio.format` is checked here, the container bytes downstream) and re-encodes them canonical onto the daemon's `images_base64` / `audios_base64` lists, and a bare content-part array is refused by naming the `{"content": …}` wrapper. The daemon, not the route, owns media bytes, marker counts and the tower refusals. `hipfire embed` mirrors it with repeatable `--image`/`--audio`, where `--audio` checks the RIFF/WAVE magic client-side and refuses anything else with the `ffmpeg` conversion (the tower's fixable-from-the-caller side).

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
