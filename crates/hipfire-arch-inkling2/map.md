# hipfire-arch-inkling2 — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — new arch crate for Inkling-2-Mini; Phase 0 only (config, visit schedule, CPU oracle), not yet wired into loader/daemon.

## Purpose

Loop geometry and CPU oracle for the 24-physical-layer, 48-visit Inkling-2-Mini MoE. Validation and visit schedule mirror llama.cpp `inkling2-loop.cpp` at merge `29b52d398b294509746ec0a3f28948a3107c0213`; the oracle operators mirror `inkling2.cpp`. GPU weight bundle + per-visit decode state (`gpu.rs`, brief 01); no forward or generation route yet.

## Gotchas

- Arch ids 24 (trunk) and 25 (MTP sidecar) are provisional educated guesses, not shipped identities.
- `LoopConfig::validate` mirrors upstream exactly: fixed loop needs >= 2 passes, varying >= 1, core block-aligned, `n_slot <= 512`.
- Physical weights are shared across visits (`layer_phys`); KV/conv state is per-visit (48 slots), allocated on GPU by `gpu.rs`.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
