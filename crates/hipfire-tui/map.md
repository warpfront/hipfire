# hipfire-tui — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Operator — terminal UI above the daemon/CLI transport; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Terminal UI for hipfire (`ratatui` + `crossterm`): chat surface over the daemon JSONL/client transport (`app.rs` `App`, `hipfire.rs` engine binding, `ui.rs`/`ui_chat.rs` rendering). Owns choice navigation, per-character typing stream, and `extract_tool_calls_from_text` bridging. The `//!` crate header is not module-documented; read [`src/main.rs`](src/main.rs) and `app.rs`.

## Gotchas

- Owns its own panic hook (`src/main.rs`) to restore the terminal — a new panic path that bypasses it will leave the terminal in raw mode.
- TUIs on different physical GPUs may coexist; TUIs targeting the same card (UUID, or PCI address for no-UUID cards) contend in the daemon's machine-wide lock directory (default `/run/lock/hipfire`, fallback `/tmp/hipfire-locks`), regardless of HOME.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
