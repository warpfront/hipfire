# hipfire-detect — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Observability — JSONL behavior detectors, side-DAG above the daemon output; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Observational coherence/behavior detectors that consume daemon JSONL and surface model weirdness — token attractors, special-token leaks, empty/stalled `<think>` blocks, n-gram density spikes, tool-call malformations, EOS-immediate, whitespace-only output. Strictly observational: never blocks, mutates, or interferes with generation; produces verdicts. Mirrors and extends the runtime guards (`hipfire-runtime/src/{loop_guard,sampler,eos_filter}.rs`). See the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

- Detectors are verdict-only — wiring them as generation blocking would duplicate and contradict `loop_guard`/`sampler` policy.
- Each detector is an independent `Verdict` pass over the same JSONL — panicking on malformed lines would hide the remaining signals.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
