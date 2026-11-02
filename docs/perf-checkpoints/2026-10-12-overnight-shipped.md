# Overnight 2026-10-11: measurements behind five shipped changes

**Lifecycle:** `historical`

**Date:** 2026-10-12. All runs took place on 2026-10-11.

**Disposition:** these are measured local results for changes that are already
on `beta`. The record gives the fixture, method and every number that the
public summaries of these changes quote. It does not admit or change any
default and does not update a product benchmark. Nothing here transfers to
another model, quant, GPU, prompt set, route or method without a new record.
Numbers marked **derived** are arithmetic on the measured values beside them.
Where this record and an earlier summary disagree, this record gives the raw
values and says so (section 6).

Sections:

1. Qwen3.8-27B DFlash at C8 on one R9700: five A/B campaigns
2. Flash-Next exact lanes on Strix Halo: PeaceMaker row kernels
3. Expert parallelism: EP2 and EP4 equal to one card
4. Shared-weight host: one weight copy for two tenants
5. Release kernel packs on ROCm 10.1
6. Differences from earlier summaries

---

## 1. Qwen3.8-27B DFlash at C8 on one R9700: five A/B campaigns

### Fixture (all campaigns)

- **GPU:** one Radeon AI PRO R9700 (gfx1201, 32 GB). Every campaign used the
  same card.
- **Models:** `qwen3.8-27b.mq4-xts` (14,987,185,152 B, sha256
  `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`) with the
  draft `qwen3.8-27b-dflash.mq4v2` (1,209,603,072 B).
- **Serve:** `hipfire serve` (the HTTP serve path) with `serve.vmm_batch`,
  `continuous_batch_size = 8` and `HIPFIRE_CB_VERIFY_CHUNK128=1` (a developer
  variable that is off by default). One fresh `hipfire serve` process per
  measurement.
- **Workloads:**
  - `conc_steady` greedy: 64 code prompts × 256 output tokens, temperature 0,
    a closed-loop client with K concurrent clients (K = 1, 4 or 8), after an
    untimed warm-up. Metric: steady-window output tok/s (`steady.tok_s`).
  - BetterBench v1 sampled: 48 requests, T 0.7, wire seed 42, K = 8.
- **Order:** within a campaign the arms ran interleaved, rotating the order
  per round. Each campaign is its own A/B. The campaigns ran on different
  builds and, from campaign 1.2 on, with `HIPFIRE_CB_PHASES=1` (per-phase step
  timing) in every arm, so the base of one campaign is not the result of the
  previous one. The steps are not one chained measurement.
- **Statistic:** `predict ab` (CS95 on the effect) where given; otherwise the
  per-process values and their mean or median, as labelled.

### 1.1 `speculation.dflash_block = auto` (L = 8 on a batching serve)

**Change:** merged as `3ac1cdbe3` … `eda01276c`, default `auto` (kill switch
`HIPFIRE_DFLASH_BLOCK=16`). Build: pre-merge branch build, daemon md5
`d97a654938b4a170fa8508e5dc056a36`. No phase timing. 3 processes per arm.
F16 verify in both arms (the IU4 default came later).

| K | L16 tok/s (per process) | L16 mean | L8 tok/s (per process) | L8 mean | change, **derived** |
|---|---|---:|---|---:|---:|
| 1 | 114.88 / 115.20 / 114.40 | 114.8 | 108.29 / 107.13 / 107.96 | 107.8 | −6.1 % |
| 4 | 228.96 / 226.99 / 226.23 | 227.4 | 274.44 / 269.25 / 273.97 | 272.6 | +19.9 % |
| 8 | 253.25 / 249.70 / 260.92 | 254.6 | 329.84 / 327.03 / 316.82 | 324.6 | +27.5 % |

- Text: every K = 8 and K = 4 request equals K = 1 within an L value (L16 text
  md5 `0ec3cc44…`, L8 text md5 `2f262d21…`). Greedy text differs between L16
  and L8.
- A separate single K = 1 process at L8 on the same build read 107.55 tok/s
  (used as the base in 1.2's lone-request row).

### 1.2 IU4 verify on by default at L = 8

**Change:** merged as `dc585428e`, with the follow-up `ea8758b9b` (kill switch
`HIPFIRE_CB_VERIFY_IU4=0`). A numerics change: verify activations are 4-bit.
Build: pre-merge branch build at `c596026ae`, daemon md5
`a456e69ae0d8a2078d3a019e8d90332c`. Arms: IU4 default vs the kill switch, same
binary, 8 processes per arm, `HIPFIRE_CB_PHASES=1` in both.

| arm | K = 8 steady tok/s, per process | median |
|---|---|---:|
| F16 verify (`HIPFIRE_CB_VERIFY_IU4=0`) | 339.93 / 340.06 / 324.97 / 340.24 / 340.41 / 339.96 / 340.01 / 340.14 | 340.0 |
| IU4 verify (default) | 351.84 / 375.35 / 375.03 / 374.25 / 374.51 / 374.51 / 374.49 / 374.62 | 374.5 |

- `predict ab`: +9.9 %, CS95 +5.3 % … +14.7 %. Ratio of the medians,
  **derived**: +10.1 %.
- Text: IU4 K = 8 equals IU4 K = 1 on 64/64 prompts (md5 `fa168ee2…`). The kill
  switch reproduces the F16 L8 text (`2f262d21…`). F16 and IU4 text differ.
- **Lone request (K = 1) on the batching serve:** 107.55 tok/s at L8 with F16
  verify (1.1, one process) → 99.32 tok/s with IU4 (one process, same
  fixture, the IU4 build): −7.7 %.
- **Quality, paired task eval** (greedy DFlash singleton, same binary, F16 vs
  IU4):

| task | items | F16 | IU4 | only F16 | only IU4 |
|---|---:|---:|---:|---:|---:|
| HumanEval | 164 | 157 | 156 | 2 | 1 |
| GSM8K 5-shot | 1,319 | 1,282 | 1,282 | 5 | 5 |
| MMLU-Pro subset | 1,400 | 871 | 871 | 0 | 0 |

- Verify-row KLD against F16: mean 8.82e-3, top-1 agreement 97.23 %
  (7,965 / 8,192 rows).

### 1.3 Batched tape copy and draft overwrite GEMMs

**Changes:** `c98e3c2b3` (one batched tape-capture copy per DeltaNet layer,
kill switch `HIPFIRE_CB_TAPE_COPY_BATCH=0`) and `3effddc87` (draft overwrite
GEMM twins replace the pre-zero fill, kill switch
`HIPFIRE_DRAFT_COLLAPSE_OFF=1`). L8 + IU4 verify. Builds: base daemon md5
`5f9123942fcd7f8e4a5f5ec105967fb3`; both changes daemon md5
`7bceaee82ac8c3a9bcc863f5a4b23358` (the single-change arms are that binary
with the other change's kill switch set). 3 processes per arm, greedy K = 8,
`HIPFIRE_CB_PHASES=1`.

| arm | steady tok/s, per process | mean |
|---|---|---:|
| base | 373.89 / 374.77 / 374.68 | 374.4 |
| tape copy only | 386.27 / 384.91 / 385.51 | 385.6 |
| overwrite only | 377.53 / 378.54 / 378.07 | 378.0 |
| both | 388.79 / 389.43 / 388.75 | 389.0 |

- Both together, **derived** from the means: +3.9 %.
- BetterBench sampled K = 8, 4 processes per arm, steady tok/s means: base
  241.1, both 248.8.
- Text md5 in every greedy run: `fa168ee2…`; every sampled run: `91bde7c5…`.

### 1.4 Lane-batched DeltaNet verify and one 64-row draft forward

**Changes:** `504b933de` (lane-batched DeltaNet preamble and recurrence, kill
switch `HIPFIRE_CB_LANE_GDN_BATCH=0`) and `d101d62b1` (one 64-row draft forward
at C8, kill switch `HIPFIRE_CB_DRAFT_ROWS64=0`). Builds: base = the 1.3
"both" state; candidate daemon md5 `54dcbc5baf504ffaced37573c5d80f6b`; the
rebased head `d101d62b1` (daemon md5 `91ea854506ff88cb910491ccb764e882`) gave
the same text. 3 processes per arm, greedy K = 8, `HIPFIRE_CB_PHASES=1`.

| arm | steady tok/s, per process | mean |
|---|---|---:|
| base | 389.02 / 389.70 / 388.89 | 389.2 |
| lane-batched DeltaNet only | 410.76 / 410.25 / 410.06 | 410.4 |
| 64-row draft only | 403.02 / 403.23 / 403.83 | 403.4 |
| both | 425.58 / 425.68 / 425.91 | 425.7 |

- Both together, **derived** from the means: +9.4 %.
- Text md5 in every greedy run: `fa168ee2…`; every sampled run: `91bde7c5…`.
- Oracles: `lane_gdn_oracle` 18/18 cases bit-identical;
  `draft_overwrite_gfx12_oracle` 40/40; `cb_vmm_state_oracle` (block 8, IU4,
  8 lanes) passes.

### 1.5 Lane-batched full-attention verify and lane selector (A/B unresolved)

**Changes:** `beb5ad6fb` (kill switch `HIPFIRE_CB_LANE_FA_BATCH=0`) and
`2c25a2dab` (kill switch `HIPFIRE_DFLASH_SELECTOR_LANES=0`). Both are merged.
Base: the rebased 1.4 head (daemon md5 `91ea854506ff88cb910491ccb764e882`);
candidate daemon md5 `de50e3de8a3b5da2a019bd51a3077bc9`, with the single-change
arms set by the other change's kill switch. 4 processes per arm, greedy K = 8,
`HIPFIRE_CB_PHASES=1`.

| arm | steady tok/s, per process |
|---|---|
| base | 392.94 / 424.28 / 403.56 / 424.56 |
| full-attention verify only | 450.48 / 445.70 / 449.90 / 444.37 |
| selector only | 421.66 / 420.99 / 427.94 / 410.66 |
| both | 421.02 / 451.73 / 455.16 / 455.70 |

- **Verdict:** the tok/s `predict ab` came out **need more runs**. A host-side
  slow mode (draft phase ≈21 instead of ≈14.5 ms/step) hit 2 of 4 base
  processes and 1 of 4 processes with both changes.
- In processes outside the slow mode only: base 424.3 / 424.6, both changes
  451.7 / 455.2 / 455.7 (≈452–456).
- Phase timings that did resolve: verify phase, base vs full-attention verify,
  win, +9.1 % (CS95 +8.3 … +9.8 %); draft phase, full-attention verify vs both
  (slow-mode-free processes), win, +6.5 % (CS95 +3.3 … +9.7 %).
- Text md5 in every greedy run: `fa168ee2…`; every sampled run: `91bde7c5…`.
  `lane_fa_oracle` 18/18 bit-identical.

### 1.6 The lone request on a batching serve

The gains above are for 8 concurrent clients. A single request on the same
serve pays for them:

| state | K = 1 steady tok/s | source |
|---|---:|---|
| L16, F16 verify | 114.8 | 1.1, mean of 3 |
| L8, F16 verify | 107.8 | 1.1, mean of 3 |
| L8, IU4 verify | 99.32 | 1.2, one process |
| after 1.3 | 101.19 | one process |
| after 1.4 | 101.46 | one process |
| after 1.5 | 101.01 | one process |

- **Derived:** L8 −6.1 % (1.1) × IU4 −7.7 % (1.2): 0.939 × 0.923 = 0.867, a
  lone-request cost of ≈ −13 % (−13.3 %). These are two separate A/Bs.
- **Derived:** the current state against the L16 mean, different builds:
  101.01 / 114.8 = 0.880 (−12.0 %).
- Kill switches: `HIPFIRE_DFLASH_BLOCK=16` restores L16, and with it the F16
  verify (the IU4 default applies only at L8). `HIPFIRE_CB_VERIFY_IU4=0` keeps
  L8 and drops only IU4. The developer kill switches of 1.3–1.5 are named in
  their sections.

## 2. Flash-Next exact lanes on Strix Halo: PeaceMaker row kernels

**Changes:** `dcf9b37c8` (the lane AR floor prices row-group spill) and
`926b929cf` (`HIPFIRE_FN_LANE_PM_ROWS` default on for gfx1151, kill switch
`HIPFIRE_FN_LANE_PM_ROWS=0`).

### Fixture

- **Hardware:** Strix Halo (gfx1151, 128 GB unified memory).
- **Model:** `qwen3.8-flash-next-gptq3.mq4` (sha256
  `8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972`) with its
  `.ple-q4` sidecar.
- **Serve:** `hipfire serve` with `serve.vmm_batch`, `continuous_batch_size =
  8` (exact Flash-Next lanes).
- **Workload:** `conc_steady` greedy, 64 prompts × 256 output tokens after 8
  untimed warm-up prompts, closed loop with K clients. One fresh serve process
  per run. Metric: output tok/s (`out_tok_s`) and TTFT p50.
- **Build:** one pre-rebase branch binary (daemon md5 prefix `fc5e1a01`) that
  already had the AR floor. The PM-off and PM-on arms differ only in
  `HIPFIRE_FN_LANE_PM_ROWS` (unset vs `1`), interleaved. The flag was off by
  default in that build; `926b929cf` flips the default.

### PM rows off vs on, K = 8

| arm | output tok/s, per process | TTFT p50, ms |
|---|---|---|
| PM rows off | 65.91 / 63.40 / 66.35 | 847 / 905 / 1,006 |
| PM rows on | 78.98 / 79.42 / 79.42 / 79.30 | 978 / 996 / 953 / 976 |

- Rounded ranges: off 63.4–66.4, on 79.0–79.4 tok/s.
- `predict ab`: tok/s +21.6 % (CS95 +6.5 % … +38.8 %). TTFT p50 is
  unresolved: every PM-on value lies inside the PM-off range.
- Text md5 `b6b9e879a6d993e2fcf475930c160b7c` in every run.

### Concurrency, PM rows on (same binary)

| K | output tok/s | TTFT p50, ms | text md5 |
|---|---:|---:|---|
| 1 | 57.41 | 269 | `b6b9e879…` |
| 2 | 72.49 | 443 | `b6b9e879…` |
| 4 | 72.54 | 495 | `b6b9e879…` |
| 8 | 78.98 / 79.42 / 79.42 / 79.30 | 953–996 | `b6b9e879…` |

- K = 1, 2 and 4 are one process each.
- **Derived:** K = 8 mean 79.28 / K = 1 57.41 = 1.38×.
- The text is the same at every K: each request equals the request run alone.

### Lane AR floor

Before the floor (PM rows on, an earlier trace build, 3 processes): 71.93 /
73.00 / 74.77 tok/s (mean 73.23). With the floor (the PM-on runs above, first
three): 78.98 / 79.42 / 79.30 (mean 79.23). **Derived:** +8.2 %.

## 3. Expert parallelism: EP2 and EP4 equal to one card

**Change:** `7c3db6e00` (the expanded MQ4V2 MoE down GEMV pins its first-pair
contraction to the single-device D4 arithmetic; `4a0855cd2` regenerated the
crate maps). The runs used the pre-rebase commit `64afcc2fa`, whose code hunks
are identical to `7c3db6e00`.

### Fixture

- **GPUs:** R9700 (gfx1201) cards in one host: 2 for EP2, 4 for EP4, all
  physical (`HIPFIRE_EMULATE_GPUS` unset), plus the single-card arm.
- **Model:** Ornith 1.5 35B-A3B, `ornith-1.5-35b-a3b.mq4r` (18,700,570,368 B,
  sha256 `84103fcc8ade42aa2ac8ec01176df7a4ead5e94810597c9fae2f6763152a3ac6`),
  a qwen35 MoE with top-k 8 routing. This is not Flash-Next.
- **Oracle:** `route_oracle_mesh` tests `qwen35_ep2_vs_single_oracle` and
  `qwen35_ep4_vs_ep2_oracle` (deterministic mode, graphs off), 27 committed
  greedy positions, rank-0 logits compared byte for byte at every position.
  Fixed oracle binary md5 `6030d7b313f5437240ba384899223d41`.

### Results

| check | before the fix | after the fix |
|---|---|---|
| EP2 vs single, 27 positions: worst logit abs diff | 0.2546 (position 10) | 0 |
| EP4 vs EP2, 27 positions: worst logit abs diff | — | 0 |
| EP4 vs single, raw bytes: 201 logit/state files × 27 positions | — | 5,427 / 5,427 equal |
| single-card output before vs after the fix | — | byte-identical |

- The 5,427 files cover logits and every state the oracle dumps: K/V, DeltaNet
  S, scales, conv and error feedback.
- **First divergence before the fix:** position 0, layer 0, the post-MoE final
  residual, FP32 element 9: single `0x3b67de5a`, EP2 `0x3b67de58` (two ULPs).
  The fused single-device kernel computes `RN(d1·x1)` then `FMA(d0, x0, ·)`;
  the expanded EP kernel computed `RN(d0·x0)` then `FMA(d1, x1, ·)` in K = 512
  group 1, rows 0–2 of every 4.
- **Kernel gate:** `mq4v2_expanded_down_matches_pinned_d4_chain_bits` passes on
  gfx1201 for K = 512 and K = 768, every row class, and fails if the opposite
  orientation is not observably different.

## 4. Shared-weight host: one weight copy for two tenants

**Change:** the shared-weight host v1 series ending `5a0dd38fd` (developer,
Linux, gfx1201, default off; `HIPFIRE_WEIGHT_HOST=0` is the kill switch). The
runs used the pre-rebase head `e7befaa47`; all 24 commits of that series have
the same patch-id as the merged series.

### Fixture

- **GPU:** one R9700 (gfx1201, 32 GB).
- **Model:** `qwen3.8-27b.mq4-xts` (the file of section 1).
- **Tenants:** two builds with different kernel code objects (build A, and
  build B with a changed gfx1201 scheduler profile: 42 named modules differ).
  Oracle `shared_weight_state_oracle` (build A md5
  `65caaf58db2586fde6ee4e2873284e1a`, build B md5
  `dc49e689d38b3ff934d1eac113fd8042`); host `hipfire-weight-host` md5
  `b57cf1354a1a6826f98f3c87e532f0fd`.

### The weight epoch

- One host process loaded the model once and exported 1,347 weight owners:
  15,097,597,952 B (15.1 GB) of unique weight bytes.

### Tenant memory and load time

| per tenant process | private load | import from the host |
|---|---:|---:|
| device memory taken by the load | 15,768,485,888 B (15.8 GB) | 375,390,208 B (375 MB) |

| build / mode | private weights ms | import weights ms | ratio, **derived** |
|---|---:|---:|---:|
| A / eager | 3,356.1 | 79.6 | 2.37 % |
| A / graph | 3,309.6 | 77.0 | 2.33 % |
| B / eager | 3,331.8 | 85.0 | 2.55 % |
| B / graph | 3,288.0 | 80.6 | 2.45 % |

- Import time is 2.3–2.6 % of the private weight load (`weights_ms` from
  `QWEN35_LOAD_PHASES`). The load gate was import ≤ 20 % of private.

### Correctness (G3): PASS

- **Cells:** 4 prompt classes (short, long, context 512, context 8,192) × 2
  modes (eager, HIP graph) × 16 comparisons = 128 cells, plus 8 informational
  rows. Each compares logits, state, tokens and UTF-8 text byte for byte
  between private, imported and concurrent (A and B together on one epoch)
  tenants of both builds.
- **Result:** 128 / 128 equal (the 8 informational rows also equal).
- **Not run:** context 32,768 (two tenants plus the host do not fit in 32 GB at
  the declared budgets).
- **Negative controls detected:** a one-byte prompt change, one flipped logit
  bit, a wrong socket (fails closed).
- **Drain (G2 case D):** a host SIGTERM while a tenant was attached. In the
  passing run the host drained in 2,312 ms (`drain_ms`) and exited with code
  0, the tenant reported no model loaded within 0.5 s, and device memory
  returned to its baseline (delta 0 B). The first attempt met every check
  except the host's exit code; it was repeated, and the repeat is the result
  given here.
- **Timing diagnostic** (not a gate): short prompt, graph mode, 3 runs per
  arm: private median 36.25 tok/s, import median 35.88 tok/s.

## 5. Release kernel packs on ROCm 10.1

**Change:** `ed1b83835` (the release workflow builds kernel packs in
`docker.io/rocm/dev-ubuntu-26.04:10.1.0-full@sha256:4f5ed1bf6532a4b9920400401b0ad0f706af356dae61e40afefbfd6c8073f0ce`).

- **Arms:** a native build and a build in that image, same commit `ed1b83835`,
  same script, tag `v0.4.2`. Both report HIP 7.16.26385, clang 24.0.0git (the
  same clang binary sha256), ROCm 10.1.0, code object v6, kernel cache ABI 6.
- **Arch:** gfx1201 only. Other arches were not compared.
- **Result:** 228 / 228 module objects have equal sha256, with no module
  missing on either side, and the two tarballs are byte-identical (1,755,888 B,
  sha256 `45b8ba405cd60009bf46de8eb778d80f2d57578e6ffd492c70ba0cc264bee93d`).
- No GPU was used. The v0.4.1 gfx1201 pack had 216 modules (ROCm 10.0, HIP
  7.15, cache ABI 5).

## 6. Differences from earlier summaries

- **IU4 F16 arm:** the 8-process median of `steady.tok_s` is 340.03, so 340.0.
  The CHANGELOG IU4 entry says 340.1.
- **Tape copy and overwrite base:** the base arm's mean is 374.4 (median
  374.7). The CHANGELOG writes the base as 374.5, which is the IU4 campaign's
  median.
- **Lane AR floor:** the `dcf9b37c8` CHANGELOG line gives C8 72.5 → 79.2 tok/s.
  The run files give 73.23 → 79.23 (section 2); no run reads 72.5 at C8 (72.5
  is the K = 2 and K = 4 value).
- **Shared-weight host:** the CHANGELOG entry says G3 is pending. G3 passed
  (section 4), at contexts up to 8,192; 32,768 was not run.
- **EP exactness** was measured on the Ornith 1.5 35B-A3B MoE (section 3), not
  on Flash-Next.
- **L8 workload:** the C8 254.6 → 324.6 pair is the greedy `conc_steady` code
  set, not BetterBench.
