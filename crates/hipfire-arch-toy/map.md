# hipfire-arch-toy — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — reference template that hangs off `hipfire-loader` via `Carrier` (daemon must not dispatch it, `arch_id = 0xFF`); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Minimum-viable `Architecture` trait implementation used as a copy-paste template for new arch crates. Hardcodes stub config/weights/state so `cargo build` stays workspace-clean before real model code is wired. The `//!` docs in [`src/lib.rs`](src/lib.rs) and [`README.md`](README.md) document what to keep vs replace and rough effort per component; `hipfire-arch-qwen35` is the production reference.

## Gotchas

- Not a real model — `arch_id = 0xFF` must never be dispatched by the daemon; it exists only to keep the workspace green.
- Only bring-up triple + carrier surface; forward path is a stub — do not cargo-cult its numerically trivial kernels into a real port.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
