# PR #1064 — A4-aware AWQ alpha grid A/B, second pair: Qwen3.5-4B (dense) and Qwen3.6-35B-A3B (MoE), gfx1100

**Lifecycle:** `historical`. Fixture-bound measured evidence for one PR's
recipe decision; companion to
`2026-11-13-a4-awq-alpha-grid-kld-qwen36-27b-gfx1100.md` (same question, dense
27B). Not a current default, an automatic baseline, a product claim, or an
admission decision.

**Disposition:** the wider candidate grid (`0.05/0.15/0.25/0.35/0.55/0.75`)
is compared against the shipped grid (`0.35/0.45/0.55/0.65/0.75`) on the same
parent, imatrix, wire format and flags per model; only the grid differs.
Reason for the second pair: the review's counter-evidence was an A3B KLD
regression, so the MoE class needs its own measurement, and the 4B class is
the small dense tier.

## Fixture identity

| field | value |
|---|---|
| GPU | AMD Radeon RX 7900 XTX, `gfx1100`, 24 GB |
| host | `inference` (shared; every eval holds `~/gpu-coord` `gpu0`; quantize is CPU-only) |
| parents | `RAW/Qwen/Qwen3.5-4B` (BF16, 9.3 GB), `RAW/Qwen/Qwen3.6-35B-A3B/source` (BF16, 67 GB) |
| imatrices | 4B: `hipfire/Qwen/Qwen3.5-4B-fixtures/imatrix_unsloth.gguf_file` (248 `in_sum2` rows, sha256 `63718abd33c6ddcbb67f9089fb8bbaf0bc9996be4e23bba14ce17dcee76924d2`); A3B: `hipfire/Qwen/Qwen3.6-35B-A3B-imatrix/imatrix_unsloth.gguf_file` (510 rows, 390 usable dense after the loader's MoE `[k,n_mat]` skip) |
| references | 4B: `qwen3.5-4b-bf16.kldref.bin` (sha256 `d3ba9b5618f86bd9efcf2cd95bc718c41df288612ddc9f8f2592ab7dbb90bb75`, model-matched); A3B: `qwen3.6-27b-bf16.kldref.bin` (sha256 `8af83b38…`, **foreign teacher**) |
| tokenizer identity | `merges.txt` md5 `bc43d7e04aa9cbafa49f74301bfcad93` and `tokenizer.json` md5 `7b6074791b84e7ce3a5fc12746b82d48` are identical across 4B, 3.6-27B and 3.6-35B-A3B, so a ref built on one tokenizes the same stream the others would |
| `hipfire-quantize` | `60bd5defa8d7f758262dfe8ff0e120f0` (PR head) |
| `eval_hipfire` | `0e76bcf155385c78e30584bb53907b51` |

## Method

One arm = one full quantize of the same parent, changing only the grid:

```
hipfire-quantize --input <parent> --format mq4v2 --awq-a4-aware \
                 --imatrix <imatrix> --threads 14 [--awq-a4-grid wide]
```

Scoring, one fresh process per arm under the GPU lock:

```
eval_hipfire --model <arm> --ref <ref> --scoring-mode prefill \
             --kv-mode q8 --kv-v q8 --max-chunks 24
env: HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_VERIFY_GRAPH=0
```

The A3B ref is a **foreign teacher** (Qwen3.6-27B BF16). Because the tokenizers
are identical, the token stream is valid and the candidate's NLL/PPL are its
own; the KLD column is divergence from that teacher, so only the A3B's *delta*
between arms is meaningful, not its absolute KLD.

## Results

### Qwen3.6-35B-A3B (MoE, arch 6)

The A4-aware grid **cannot act on this model**: it changes no quantized
byte, so its KLD is identical by construction, not by measurement.

Verified by a full-file byte comparison of the two arms (`cmp -l`): the last
differing byte is at offset 17,318,317, while the tensor payload starts at
`data_offset` 17,321,984 (HFQM header field, identical in both files). The
2,908,008 differing bytes all lie in the header's metadata block — the
self-describing `awq_a4_grid: "wide"` key this PR writes for the opt-in grid,
which re-serialises the ~17 MB metadata region. **All 22,870,341,888 payload
bytes are identical**, so every tensor's encoded bytes are identical.

- 0 `AWQ A4:` search lines across the whole 40-layer quantize.
- The artifact carries 12,288 `awq_scale` sidecars and **every one is
  `…experts.<n>.…`**; zero non-expert sidecars.
- Why, from the sources: attention lands on `Q8_F16` (fixed tier) or
  `MQ6G256V2` (K-map) — never qt44, and the A4 search only runs inside the
  `use_mq4v2` arm; routed experts are the fixed-`--awq-alpha` arm and
  `--awq-a4-aware` is MQ4V2-only; and shared experts, which *are* qt44, miss
  `imatrix_weights_for` because `safetensors_to_ggml_name` has no
  `mlp.shared_expert.* -> ffn_*_shexp` case (`calibration.rs`), even though
  the imatrix carries those rows (`ffn_gate/up/down_shexp` 40 each).

Consequence: any A3B KLD movement the review measured travelled through the
per-expert call site, not through this grid — consistent with #1065 gating
that call site off.

Artifacts: 22 887 663 872 B each (shipped md5
`0d496c3ec3908403f4717c52258a4c66`). They do **not** fit this card for
scoring — `hipMalloc: out of memory` at
`layers.29.mlp.experts.233.gate_up_proj` — so no end-to-end A3B KLD is
reported for the tiered recipe.

### Qwen3.5-4B (dense, arch 5)

Selection moved on 57 of 128 searches (44.5 %), 28 of them into `< 0.35`:

| grid | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped | 0 | 0 | 0 | 25 | 19 | 37 | 24 | 23 |
| wide | 1 | 11 | 16 | 23 | 0 | 47 | 0 | 30 |

| arm | slice-mean KLD | mean NLL | PPL |
|---|---:|---:|---:|
| shipped | 0.161414 | 2.459354 | 11.6973 |
| wide | **0.154004** | **2.389114** | **10.9038** |
| Δ | **−0.007410 (−4.59 %)** | **−0.070240 (−2.86 %)** | **−0.7935 (−6.78 %)** |

Artifacts: 2 588 862 464 B each; shipped md5 `2a4c9ea39dffa35a9162f305c61a5f16`,
wide `1bb187cb9a259a691f4e9ab0fa754c3e`.

This is the one fixture where the wider grid wins on **all three** metrics at
once, and the largest KLD gain of the set (−4.59 %). It is also the smallest
model, i.e. the one under the most quantization pressure.

## What this record does not establish

- No throughput measurement of any kind.
- The A3B KLD is not comparable with a model-matched reference, and no
  model-matched A3B reference exists on this host (the 67 GB BF16 parent does
  not fit 24 GB VRAM or 26 GB RAM, so `build_kld_ref_native` cannot run).
- No routed-expert claim: `--awq-a4-aware` is MQ4V2-only, expert `down_proj`
  keeps the fixed `--awq-alpha` on both arms, and the loader skips the
  imatrix's MoE `[k,n_mat]` rows.
- No statement that either grid is optimal, or that the sign of the effect
  transfers between model classes.
