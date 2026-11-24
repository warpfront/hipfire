# saddle-quant — map

> **Status:** `supported` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Quantization-quality toolkit. In the default build graph — unlike
> [`saddle-lab`](../saddle-lab/map.md), which is archived research and excluded
> from `cargo build` by design. This crate gates published quality claims, so it
> is built and tested on every change.

## Purpose

Single owner of the quantization-quality artifact formats and the measurement
methodology built on them: HFQ/HFQM containers, HFHS Hessians, GGUF imatrices,
KLD references, and score sequences.

It exists because these formats had no owner. `hipfire-quantize` is primarily a
binary; its `lib.rs` exports `hfqm`/`hessian_io`/`hfhs_diag` but not
`gguf_input`, and **no crate outside it takes the dependency**. Measured
2026-08-16: **29 files** across `crates/` and `scripts/` independently parse the
HFQ header. The tree has already had to delete redundant parsers by hand —
`3dfd1b3f5`, *"drop awq_recipe_fingerprint.py — third redundant HFQ parser"* —
and `crates/hipfire-runtime/examples/imatrix_collect.rs:258` documents a
duplication forced by the missing export. One parser per format, here.

The format layer is GPU-free and dependency-light on purpose, so CI, offline
tooling, and Python bindings can link it without a ROCm toolchain. Anything
requiring a forward pass sits behind the `eval` feature.

## Methodology this crate is written against

The previous KLD pipeline produced numbers that could not support the claims
made from them. The replacement is specified against those defects:

| defect in the old pipeline | what this crate does |
|---|---|
| `top_k=256` over a 248,320 vocab (0.103% of the distribution), never validated against full-vocab | `Estimator` is recorded in the reference header; `TopK` carries `bias_vs_full`, and an uncalibrated truncated estimator is reportable but not comparable |
| scored window hardcoded to `n_ctx/2 .. n_ctx-1` | `WindowSpec` is explicit and stored |
| KV cleared per chunk, so nothing past 2K was ever measured while `kv-mode q8` was on | `WindowSpec::carry_kv` makes depth a recorded property |
| teacher unvalidated — a BF16 passthrough once scored PPL 1392 on WikiText-2 and still yielded a structurally valid reference | `OracleStats` is stored in the header, not `eprintln`'d |
| no integrity gate; every run logged `manifest.json missing; skipping ref sha256 check` | `ArtifactId` digests for teacher and corpus, enforced by `verify_against` |
| `n_chunk=24` gave overlapping CIs across five arms spanning 0.044–0.067 | reduction reports intervals, `separated_from` refuses unsupported rankings, and `min_chunks_for_power` turns that into an actionable chunk count |

## Gotchas

- **Wire tags are permanent.** `format::QuantType` discriminants are persisted
  in every `.hfq` on disk. Never renumber; add new tags at the end.
- **The HFQ magic is `HFQM`** even for `.mq4` model files — it is not a
  separate container.
- **Never load a Hessian eagerly.** A real one is 31,481,139,200 B. AWQ needs
  only `diag(H)`; the full K×K is consumed solely by LDLQ/GPTQ. `hfhs::diagonal`
  strides the payload for exactly this reason.
- **Imatrix key order matters.** hipfire-produced imatrices are safetensors-keyed;
  llama.cpp ones are `blk.*`-keyed. `imatrix::lookup` tries safetensors first,
  then the ggml mapping. Reversing that order was a real past defect that
  silently turned AWQ into a no-op.
- An imatrix without an AWQ alpha changes nothing. `--imatrix` alone produced a
  byte-identical artifact (md5 `129909ad0fed`) and an identical KLD to six
  decimals; every consumer is gated on `AWQ_ALPHA`.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
