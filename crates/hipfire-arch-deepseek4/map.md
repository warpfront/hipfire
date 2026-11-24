# hipfire-arch-deepseek4 — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 9); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

DeepSeek V4 Flash (`arch_id` = 9): hybrid with Hyper-Connections, compressed-KV indexer, tail-only RoPE, Q/O-LoRA, raw SWA. Owns `Deepseek4Bundle`/`Deepseek4Config`/`weights`/`state`/`forward` plus the `Architecture` bring-up and the MTP/DSpark speculative head and `deepseek4_chat` example. See the `//!` docs in [`src/lib.rs`](src/lib.rs) and [`arch.rs`](src/arch.rs).

## Gotchas

- Wrapper clippy allowances are structural — kernel launch signatures exceed 7 args by design; do not lint-fix them into tuples.
- DeepSeek4 `load_model_ep` expert-parallel rank state lives in `LoadedModel.ep`, not `ModelState` — teardown must respect that split.
- The native MTP layer (`Deepseek4MtpDrafter`, `spec_decode`) has no measured AR floor (audited): `MTP_MEASURED_AR_FLOOR` (`src/lib.rs`) is `false`, so `speculation.mtp = auto` keeps plain AR and `DeepseekV4Config::load_mtp` leaves the in-trunk MTP layer unloaded; explicit `on` / `--spec mtp` loads and arms it. The DSpark sidecar is a separate selector and unaffected.
- `HIPFIRE_DS4_EXPERT_SOURCE=mapped` (default `resident`, `src/expert_source.rs`) loads every routed expert into pinned host-mapped blobs (74.25 GiB with the MTP head) admitted before the first allocation; `routed_seal.rs` binds a DS4-local `MutableExpertResidency` per layer (owner generation + table extents) that the score decode, hash decode, rows program and MTP head each verify. It is deliberately not exposed through `RoutedExpertWeights::mutable_residency` (the shared dispatch hook is QT44/QT53-only; S3 owns it). No fill/evict engine yet: the pointer tables hold the canonical image for the whole owner generation. GPU gate: `examples/fn_ds4_mapped_parity.rs` (lab).
- A speculative verify (DSpark, native MTP) writes every row before acceptance, and the SWA ring and the compressor rings are finite: a rejected row overwrites committed history there. `src/verify_rollback.rs` saves it before the forward (`arm_verify_rollback`; the compressor ops stash the window's rows) and `commit_prefix` / `spec_decode` restore it (`rollback_verify`). New ring-shaped state written by the rows program needs the same treatment.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
