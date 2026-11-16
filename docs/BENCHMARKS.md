# Benchmarks

## Current scoreboard: beta `ed92048043` (measured 2026-11-16)

**Truth state:** measured (not admitted). Every cell below names its commit,
model file, fixture and date, and its raw per-process JSON is cited by a
`raw` ledger row (ids `SB16-*`) in [`registry/receipts-v1.json`](../registry/receipts-v1.json)
`ledger` ([`PERF-LEDGER.md`](PERF-LEDGER.md)), which also carries each row's
95% CI. None of these numbers is a speed floor, admission or route
certificate; product admission stays in [`admissions.yml`](admissions.yml).

| Field | Value |
|---|---|
| Build | `ed920480432141e1e0414be337c72efc50ea3960` (beta), release archive from [`scripts/build-binary-archive.sh`](../scripts/build-binary-archive.sh), one per arch |
| Cards | R9700 (gfx1201, one card), 7900 XTX (gfx1100), Strix Halo (gfx1151); each cell an exclusive timing reservation |
| Models | `qwen3.8:27b` (= `qwen3.8-27b.mq4-xts`, sha256 `3e38ccba…`), `qwen3.6:35b-a3b-mq4rs` (`8806a742…`), `qwen3.8:27b-mq3-xt` (`3e04fc8d…`), and the unregistered symmetric file `qwen3.8-27b.mq3-xts` (`68d5e6b9…`) loaded by path; Flash-Next GPTQ3 monolith (`8b15b6fe…`, by path) on R9700 only |
| Config | defaults (speculation, KV mode, backend); clean `HIPFIRE_HOME` |
| Prefill / AR | `hipfire bench --matrix --pp 512,2048,8192 --ctx 128,8192 --tg 128 --warmups 2 --runs 3` (synthetic ids; AR decode at context 128 and 8192) |
| Speculative decode | default speculation, greedy, `--max-tokens 256 --min-tokens 256`; code `benchmarks/prompts/humaneval_0_has_close_elements.txt` (md5 `5333a1f7…`), prose `prose_river_short.txt` (md5 `07a78809…`); τ = tokens per verify window |
| TTFT | `hipfire bench --ttft --prompt-file benchmarks/prompts/ttft_511.txt` (511 tokens, md5 `7423e894…`), time to first streamed token |
| Statistic | 4 fresh processes per cell and fixture; per process the median of its runs, per cell the median of the 4 process medians; 95% bootstrap CI over processes in the ledger |

Prefill and AR decode in tok/s; speculative decode in tok/s with τ in brackets;
TTFT in ms. Each value is the median of 4 fresh processes; the last column is
the row's widest 95% CI half-width relative to its median (the ledger row has
every CI).

| card | model | pp512 | pp2048 | pp8192 | tg128 @128 | tg128 @8K | spec route | code tok/s (τ) | prose tok/s (τ) | TTFT 511 ms | widest CI ± |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| R9700 | `qwen3.8:27b` (mq4-xts) | 4,948 | 5,350 | 5,195 | 38.6 | 37.7 | dflash | 163.7 (6.26) | 52.2 (1.11) | 143 | 17.8% |
| R9700 | `qwen3.6:35b-a3b-mq4rs` | 11,612 | 18,268 | 20,586 | 249.7 | 232.4 | AR | 243.8 | 244.7 | 64 | 0.4% |
| R9700 | `qwen3.8:27b-mq3-xt` | 613 | 608 | 610 | 42.6 | 41.6 | dflash | 94.2 (2.75) | 55.5 (1.22) | 674 | 0.9% |
| R9700 | `qwen3.8-27b.mq3-xts` (file) | 3,164 | 3,547 | 3,532 | 47.3 | 45.9 | AR | 47.1 | 47.1 | 185 | 0.4% |
| R9700 | Flash-Next GPTQ3 monolith (file) | 1,229 | 2,508 | 6,782 | 32.6 | 31.6 | mtp+ngram | 73.7 (3.86) | 59.0 (0.97) | 637 | 8.8% |
| 7900 XTX | `qwen3.8:27b` (mq4-xts) | 2,570 | 2,804 | 2,712 | 50.7 | 49.2 | dflash | 82.8 (3.54) | 54.0 (1.45) | 319 | 0.2% |
| 7900 XTX | `qwen3.6:35b-a3b-mq4rs` | 6,364 | 5,675 | 3,856 | 213.0 | 202.6 | AR | 207.3 | 208.6 | 109 | 0.4% |
| 7900 XTX | `qwen3.8:27b-mq3-xt` | 514 | 514 | 507 | 51.7 | 49.6 | dflash | 53.2 (3.48) | 29.9 (1.30) | 1,015 | 0.3% |
| 7900 XTX | `qwen3.8-27b.mq3-xts` (file) | 1,890 | 2,125 | 2,087 | 58.6 | 56.0 | AR | 58.2 | 58.3 | 471 | 0.4% |
| Strix Halo | `qwen3.8:27b` (mq4-xts) | 1,230 | 1,214 | 1,156 | 15.0 | 14.6 | dflash | 35.8 (3.54) | 25.8 (1.45) | 780 | 6.2% |
| Strix Halo | `qwen3.6:35b-a3b-mq4rs` | 2,672 | 3,105 | 2,326 | 84.7 | 80.6 | AR | 84.5 | 84.7 | 218 | 0.2% |
| Strix Halo | `qwen3.8:27b-mq3-xt` | 260 | 258 | 255 | 18.6 | 18.0 | dflash | 31.3 (3.48) | 18.9 (1.30) | 1,999 | 1.3% |
| Strix Halo | `qwen3.8-27b.mq3-xts` (file) | 260 | 258 | 254 | 18.8 | 18.2 | AR | 18.8 | 18.8 | 1,949 | 0.6% |

Reading notes (details in the ledger rows' receipts):

- "spec route" is what the default configuration ran. The 27B registry tiers
  pair a DFlash draft by default; `qwen3.6:35b-a3b-mq4rs` runs plain AR (MTP
  `auto` declines without a measured AR floor); the unregistered
  `qwen3.8-27b.mq3-xts`, loaded by path, gets no draft and runs AR.
- The speculative rows decode exactly 256 tokens (`--min-tokens 256`). The code
  answer ends at 173 tokens, so the code row includes decoding past EOS, where
  DFlash acceptance drops: on the 7900 XTX `qwen3.8:27b` reaches 249.6 tok/s
  (τ 12.31) at its natural 173-token length against 82.8 (τ 3.54) at 256.
- On `qwen3.8:27b-mq3-xt`, default DFlash is slower than AR on the prose
  prompt on the 7900 XTX (29.9 vs 51.7 tok/s).
- R9700 `qwen3.8:27b` tg128 @128: one of the four processes ran at 31.7 tok/s
  (the others 38.6), hence its 17.8% CI.
- Flash-Next is the retired GPTQ3 monolith on one R9700 (host-mapped experts,
  default MTP + n-gram route), measured only where a card had room.

## Historical tables

Everything below this heading is a retained, fixture-bound observation from
an earlier campaign (April–July 2026 builds). None of these rows is a live
floor, product default, admission, or route certificate. Per
[`INDEX.md`](INDEX.md), a **measured** claim requires a named fixture, binary
and model identity, and a measurement date on the same report — these tables
do not carry that full identity manifest, so they are **historical** only
(the inventory date below is not a measurement date).

The tables below were taken on older builds, most with the `asym3` KV mode
removed in 0.5.0; they are not the current speed of any listed card or model.
Dated v0.4.4 measurements are in the README's
[Measured performance (v0.4.4)](../README.md#measured-performance-v044) table
and the [CHANGELOG](../CHANGELOG.md) entries it cites. The v0.4.5 Flash-Next
EP2 serve measurement (2x R9700, 2026-11-11) is the **measured** (not admitted)
record [`perf-checkpoints/2026-11-11-flash-next-ep2-cp20.md`](perf-checkpoints/2026-11-11-flash-next-ep2-cp20.md);
it has no [`admissions.yml`](admissions.yml) row and is not a product floor.

| Field | Value |
|---|---|
| Page state (tables below) | **historical** (see [`INDEX.md`](INDEX.md)) |
| Inventory date | 2026-07-19 |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` |
| Comparison base | `origin/beta` @ `9ffb18da9d1377dfbf759db82641ea039b2e522e` |

## How to read this page

1. Treat every table outside the current scoreboard as **historical** only. Do not upgrade a row
   to **measured** without a complete per-table date, fixture, and
   binary/model-identity evidence manifest **and** a fresh protocol-compliant
   run under
   [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md).
   Measurement and admission are independent: a complete **measured** row is
   still not a product default or route certificate. Any product-default or
   admission claim **additionally** requires an explicit row in
   [`admissions.yml`](admissions.yml) (schema v2; exactly one earned record — fail closed otherwise).
2. Many older rows used the then-default `asym3` KV mode (Givens asymmetric KV),
   which was **removed in 0.5.0** — historical
   reproduction only. Current
   clean configs resolve `kv_cache=auto` through the model registry and
   otherwise to the architecture default (Qwen native `fp8` on eligible exact
   gfx1201, `q8` elsewhere). Do not compare asym3 rows to q8 or fp8 rows as
   one A/B.
3. Speed floors used by tooling live in `tests/speed-baselines/<arch>.txt` and
   are exercised by [`scripts/speed-gate.sh`](../scripts/speed-gate.sh) when that
   path’s policy applies. Those files are **not** reproduced here.
4. Validation and promotion routes live only in [`VALIDATION.md`](VALIDATION.md).
   Retired batteries are **historical reproduction only** — never current
   acceptance for a bench claim.
5. Redline-attributed numbers require the certification ladder in
   [`REDLINE.md`](REDLINE.md). Throughput without timed-arm route proof is not a
   Redline certification.

## Claim language (fail closed)

| Allowed | Forbidden without fresh evidence |
|---|---|
| “On \<date\>, fixture \<id\>, median X tok/s” | “Current baseline is X” |
| “Historical DFlash genre table (asym3, max_tokens=120)” | “DFlash is 4× on 27B” as a present product fact |
| “Speed-gate floor in `tests/speed-baselines/…`” | Treating any table below as that floor |
| Link to a dated `perf-checkpoints/` file | Stitching harness exits into an admission |

A bench number without protocol + identity hashes is **rejected** as promotion
evidence ([`VALIDATION.md`](VALIDATION.md)).

## Reproducing measurements

Protocol owner: [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md)
(warmup, fresh-process, noise band, prompt MD5 discipline).

```bash
# Canonical CLI surface (pp/decode matrix depends on flags and model)
hipfire bench qwen3.5:9b

# Optional speed-floor check when that path’s policy applies
./scripts/speed-gate.sh --fast
```

For DFlash genre work, use prompt-MD5-pinned fixtures under
[`benchmarks/prompts/`](../benchmarks/prompts/). Prompt structure swings τ;
byte-identical prompts are mandatory across sessions.

Optional A/B tooling: [`scripts/probe_commits.sh`](../scripts/probe_commits.sh)
(also reachable from `scripts/gates.sh --perf`). Neither script creates an
admission row.

---

## Historical: autoregressive decode (no spec) — 7900 XTX (gfx1100)

**Truth state:** historical
**Fixture notes:** then-default engine config (asym3 KV, FlashAttention auto,
`prompt_normalize=on`). Medians across 5 runs unless noted. Not a current q8
methodology row. No per-row binary/model hash manifest on this page.

| Model | decode | prefill (peak) | effective BW |
|---|---:|---:|---:|
| Qwen 3.5 0.8B MQ4 | **391 tok/s** | **7383 tok/s** | 200 GiB/s |
| Qwen 3.5 4B MQ4 | **180 tok/s** | **2487 tok/s** | 433 GiB/s |
| Qwen 3.5 9B MQ4 | **132 tok/s** | **1663 tok/s** | **654 GiB/s** |
| Qwen 3.5 27B MQ4 | **47 tok/s** | **478 tok/s** | **651 GiB/s** |

Engineering note retained with the snapshot: 9B and 27B decode saturated
~650 GiB/s of the 7900 XTX’s 960 GB/s peak (~68% BW-efficient end-to-end across
weights + KV + activations). Prefill on the smaller sizes was WMMA-bound on the
MQ4 fused projections **under that fixture**.

## Historical: DFlash speculative decode by genre — 7900 XTX

**Truth state:** historical (**superseded methodology**)
**Do not use as a current DFlash baseline.**

This table used `asym3` KV and `max_tokens=120`. Current DFlash performance
claims require the protocol in
[`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) (including
q8 where that is the active KV path, `max_tokens=256` when that is the campaign
contract, ≥3 fresh-process runs, prompt and binary hashes) plus the claim-class
route in [`VALIDATION.md`](VALIDATION.md). A retired battery pass is **not**
acceptance evidence. No per-row
binary/model hash manifest on this page.

DFlash speedup in this snapshot was **genre-conditional**. Code prompts whose
target distribution matched the draft won; long-form prose where high-entropy
continuations diverged could net-lose.

5-run medians under the historical configuration: asym3 KV, `--no-chatml`,
`max_tokens=120`, `prompt_normalize=true`:

| Model | genre | AR tok/s | DFlash tok/s | speedup | τ |
|---|---|---:|---:|---:|---:|
| Qwen 3.5 27B | code (HumanEval/53) | 44.1 | **196.0** (peak 218.6) | **4.45×** | 9.82 |
| Qwen 3.5 27B | prose (Rome essay) | 44.0 | 49.6 | 1.13× | 1.67 |
| Qwen 3.5 27B | instruct (sky-color) | 44.6 | 44.7 | 1.00× | 1.39 |
| Qwen 3.5 9B | code (HumanEval/53) | 124.0 | **329.1** (peak 346.7) | **2.65×** | 6.76 |
| Qwen 3.5 9B | code (HumanEval/0) | 121.9 | **372.9** | **3.06×** | 8.23 |
| Qwen 3.5 9B | instruct (sky-color) | 124.4 | **246.9** | **1.99×** | 4.76 |
| Qwen 3.5 9B | prose (federalist) | **125.3** | 99.4 | 0.79× ✗ | 1.20 |
| Qwen 3.5 9B | prose (Rome) | **122.7** | 98.3 | 0.80× ✗ | 1.20 |
| Qwen 3.6 27B | code (HumanEval/53) | 44.2 | **185.5** | **4.19×** | 9.25 |

**Config context for the snapshot (not a timeless default):** CLI default
`dflash_mode` is `"off"` (`crates/hipfire-config/src/lib.rs`), except that the Qwen3.8-27B tiers with a registry-declared draft resolve `auto` ([CONFIG.md](CONFIG.md#qwen38-27b-paired-dflash-default)); DFlash is opt-in elsewhere until a campaign
proves a broader win. Enable globally with `hipfire config set dflash_mode auto`
(dense Qwen 3.5+ on, A3B off unless overridden) or per model with
`hipfire config qwen3.5:27b set dflash_mode on`. The numbers above were measured
with DFlash forced on.

## Historical: 7900 XTX small-model snapshot (hipfire only)

**Truth state:** historical
Same-machine snapshot of hipfire MQ4 with asym3 KV and FlashAttention at
matched ~140-token and ~530-token prompts and 128-token generation. No
per-row binary/model hash manifest on this page.

| Model | hf pp128 | hf pp512 | hf decode |
|---|---:|---:|---:|
| Qwen 3.5 0.8B | **10,861** | **12,962** | **353** |
| Qwen 3.5 4B | **3,304** | **3,321** | **165** |
| Qwen 3.5 9B | **1,920** | 1,919 | **122** |

## Historical: other arches (decode tok/s)

**Truth state:** historical
Then-default configuration; not a cross-arch speed-gate matrix and not an
admission of gfx12 product routes. No per-row binary/model hash manifest on
this page.

| Arch | Examples | 0.8B | 4B | 9B | 27B |
|---|---|---:|---:|---:|---:|
| RDNA2 (gfx1030) | V620 Pro, RX 6800 XT | 250 | — | 65 | 22 |
| RDNA1 (gfx1010) | RX 5700 XT | 190 | 61 | 43 (HF4) | OOM |
| APU (gfx1013) | BC-250 | 207 | 77 | 47 | OOM |
| GCN5 (gfx906) | MI50 / MI60 | 231 | 61 | 59 | 21 |
| MI300X (gfx942) | datacenter | 850 | 480 | 320 | 130 |

Supporting dated notes (still fixture-bound):

- MI300X is wave64 + MFMA — different kernel family from RDNA WMMA paths.
- RDNA4 (gfx1200 / gfx1201) has gfx12-specific WMMA paths across fused GEMM,
  attention, and MoE kernels; operations without a gfx12 sibling still fall
  through typed dispatch tables. Presence of a kernel path ≠ Redline admission
  and ≠ a row in [`admissions.yml`](admissions.yml).
- gfx906 (Vega 20) prefill batch≥16 used the nwarps=4 dp4a MMQ kernel
  ([`plans/gfx906-mmq-prd.md`](plans/gfx906-mmq-prd.md)). Decode batch=1 notes from the 2026-05-05
  investigation
  ([`perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`](perf-checkpoints/2026-05-05-gfx906-decode-investigation.md)): residual
  GEMV software-pipelined ILP variant (+4.8% on 9B under that fixture) and fused
  projections pre-quantizing x to Q8_1 with `v_dot4_i32_i8` (+9.3% on 9B).
  Combined historical row: 50.7 → 58.9 tok/s (+16.2%) on Qwen 3.5 9B.

## Where new numbers go

| Kind of result | Owner |
|---|---|
| How to measure | [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) |
| Bench-suite layout | [`methodology/bench-suite.md`](methodology/bench-suite.md) |
| Immutable campaign checkpoints | [`perf-checkpoints/`](perf-checkpoints/) (new dated file; do not rewrite old bodies) |
| Every measured or published perf number (history, provenance, guard rows) | [`PERF-LEDGER.md`](PERF-LEDGER.md) (`registry/receipts-v1.json` `ledger`) |
| Claim → validation route | [`VALIDATION.md`](VALIDATION.md) |
| Product admission | [`admissions.yml`](admissions.yml) only (schema v2; exactly one earned record) |
| Redline-attributed claims | [`REDLINE.md`](REDLINE.md) |
| Speculation capability inventory | [`speculation-support-inventory.md`](speculation-support-inventory.md) (verify in source) |

Do not paste mutable inventory matrices into this page. Do not promote a
historical row by recency alone.
