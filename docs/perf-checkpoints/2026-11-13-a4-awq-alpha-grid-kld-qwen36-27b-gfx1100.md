# PR #1064 — A4-aware AWQ alpha grid A/B on Qwen3.6-27B (gfx1100)

**Lifecycle:** `historical`. Fixture-bound measured evidence for one PR's
recipe decision. Not a current default, an automatic baseline, a product
claim, or an admission decision.

**Companion:** `2026-11-13-a4-awq-alpha-grid-kld-4b-a3b-gfx1100.md` measures
the same grid on Qwen3.5-4B (dense) and Qwen3.6-35B-A3B (MoE).

**Disposition:** selecting per-tensor AWQ smoothing exponents from the wider
candidate grid (`0.05/0.15/0.25/0.35/0.55/0.75`, the grid this PR adds) is
compared against the shipped grid (`0.35/0.45/0.55/0.65/0.75`) on the same
BF16 source, imatrix, wire format and flags. Reading: the shipped grid stays
the default; the wider grid is only reachable through `--awq-a4-grid wide`.

## Fixture identity

| field | value |
|---|---|
| GPU | AMD Radeon RX 7900 XTX, `gfx1100`, 24 GB |
| host | `inference` (shared; GPU work under `~/gpu-coord`, quantize is CPU-only) |
| parent checkpoint | `/home/ghazni/models/RAW/Qwen/Qwen3.6-27B` (BF16 safetensors, single 52 GB file) |
| imatrix | `qwen3.8-27b-imatrix/imatrix_unsloth.gguf` (496 `in_sum2` rows; same geometry as the 3.6-27B parent: 5120 hidden / 64 layers / 17408 ffn) |
| reference | `qwen3.6-27b-bf16.kldref.bin`, sha256 `8af83b38710fbc8e5ee46ce2b84b3545381c834f17bf6dfaa15fd817e4734446`, 2 480 989 032 B, top-256, n_ctx 2048, 1175 chunks (llama-perplexity, `benchmarks/quality-baselines/harness/manifest.json`) |
| slice | `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`, md5 `83b0205a304bf4e52172ecdb05f2e895` (from the reference header; eval reads tokens from the ref) |
| `hipfire-quantize` (beta, `b90644a40e`) | md5 `75aee667e817543ffaf86f3c6725985d` |
| `hipfire-quantize` (PR head) | md5 `60bd5defa8d7f758262dfe8ff0e120f0` |
| `eval_hipfire` (PR head) | md5 `0e76bcf155385c78e30584bb53907b51` |

## Method

One arm = one full quantize of the same parent, changing only the A4-aware
candidate grid:

```
hipfire-quantize --input <parent> --format mq4v2 --awq-a4-aware \
                 --imatrix <imatrix> --threads 14 [--awq-a4-grid wide]
```

`--format mq4v2` is the A4-aware contract (MQ4V2-only); routed experts do not
exist in this dense model, so every search here is a dense per-tensor search.
The beta arm runs the *unmodified* `b90644a40e` binary with no grid flag: it is
the byte-identity control for the shipped default.

Scoring, one fresh process per arm, GPU lock held:

```
eval_hipfire --model <arm> --ref <ref> --scoring-mode prefill \
             --kv-mode q8 --kv-v q8 --max-chunks 24
env: HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_VERIFY_GRAPH=0
```

`eval_hipfire` is deterministic under this method (greedy teacher-forced
prefill against a fixed reference), so one run per arm is the full result.
Checked, not assumed: a second fresh process on the shipped arm reproduced
`slice-mean KLD = 0.247137` and a byte-identical `.kldseq`
(md5 `578da2c03c67d0a51ef654b60a4c3769`).

Both arms also share two recipe caveats, which therefore cancel in the A/B
but bound the absolute numbers: the imatrix is a 3.8-27B file applied to a
geometry-identical 3.6-27B parent, and `--awq-fix-la-head-order` was not
passed, so the 48 `linear_attn.out_proj` AWQ scales use the GGUF converter's
tiled V-head order (the quantizer warns about this at load).

## Results

### Selection: the wider grid is not a superset

Shipped grid, 256 dense per-tensor searches (one log line each):

| alpha | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 |
|---|---:|---:|---:|---:|---:|
| tensors | 57 | 64 | 48 | 53 | 34 |

Wide grid, same 256 searches:

| alpha | 0.05 | 0.15 | 0.25 | 0.35 | 0.55 | 0.75 |
|---|---:|---:|---:|---:|---:|---:|
| tensors | 8 | 22 | 30 | 64 | 76 | 56 |

The selection moved on **141 of 256 (55.1 %)** tensors, 60 of them into the
band only the wide grid reaches (`< 0.35`).

The wide grid **drops `0.45` and `0.65`**, which the shipped search picked for
117 of 256 tensors here, so `wide` is not "the shipped grid plus more
candidates": for those tensors it removes the winning candidate. The measured
transitions include both directions — `0.45 -> 0.55` (20), `0.65 -> 0.55`
(17), `0.65 -> 0.75` (17), `0.45 -> 0.75` (10) *upward*, and `0.45 -> 0.25`
(13), `0.65 -> 0.15` (6), `0.45 -> 0.15` (7) *downward*. Its W4A4 surrogate
objective can therefore be worse on a tensor, so the KLD effect is a
measurement (below), not a corollary of the sweep.

### Byte identity of the default

The unmodified `beta` binary (`b90644a40e`, md5
`75aee667e817543ffaf86f3c6725985d`) and the PR head
(`60bd5defa8d7f758262dfe8ff0e120f0`), both run with `--awq-a4-aware` and no
grid flag, on the same parent: **byte-identical, 14 987 181 056 B,**

```
907ffaf680c8381a8096666d7466eb95  arm-beta-shipped.mq4v2
907ffaf680c8381a8096666d7466eb95  arm-patched-shipped.mq4v2
```
(`cmp` reports no differing byte, so this is full-file equality, not only a
matching digest.)

and the per-tensor selection maps agree on **256/256** searched tensors. The
shipped artifact also carries no `awq_a4_grid` metadata key, so the metadata
block is unchanged too.

The opt-in arm differs by design: same size (14 987 181 056 B), md5
`73b12391ec198dc97ae6a2d047e60d72`, metadata `awq_a4_grid: "wide"`.

### KLD (the decision number)

| arm | grid | slice-mean KLD | mean NLL | PPL |
|---|---|---:|---:|---:|
| shipped, beta binary (control) | 0.35/0.45/0.55/0.65/0.75 | 0.247137 | 1.989587 | 7.3125 |
| shipped, PR head (no flag) | same | 0.247137 | 1.989587 | 7.3125 |
| wide (`--awq-a4-grid wide`) | 0.05/0.15/0.25/0.35/0.55/0.75 | 0.242580 | 1.989769 | 7.3138 |

**Wide vs shipped: KLD −0.004557 (−1.84 %), NLL +0.000182 (+0.009 %), PPL
+0.0013 (+0.018 %).** `.kldseq` md5: shipped
`578da2c03c67d0a51ef654b60a4c3769`, wide `a8729605c20d159624065d2f987dec7e`.

Reading: the KLD gain is real and deterministic under this fixture, but it is
small, it is one model against one reference, and both likelihood summaries
move the other way — so the wider grid is **not** established as a general
improvement and does not become the default. It is also the opposite sign to
the reviewer's A3B measurement (+3.78 % KLD), which was taken with the
per-expert search that #1065 now gates off; the sign of the effect depends on
which tensors the search is allowed to move. Both facts argue for opt-in.

## What this record does not establish

- No claim about Qwen3.8-27B, the A3B MoE tiers, or any other arch/model: the
  imatrix is a 3.8-27B file used on a geometry-identical 3.6-27B parent, which
  both arms share, so it cancels in the A/B but the absolute KLD is not a
  published pin.
- No routed-expert claim. `--awq-a4-aware` is MQ4V2-only; expert `down_proj`
  keeps the fixed `--awq-alpha` on both arms.
- No throughput measurement of any kind.
- No statement that either grid is optimal; only that the wider grid is not
  automatically better because a fixed-alpha sweep preferred its band.
