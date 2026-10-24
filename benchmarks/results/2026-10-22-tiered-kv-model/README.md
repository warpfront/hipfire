# #925 real-model positional proof, not product admission

Base: upstream beta `8c148af7861c624bba28ed0805bddc4379c84aad`.
Prior storage commit: `35ede67b608fd50e6cd05565d9b4055afa5b3366`.
Commands and explicit limitations:
[investigation](../../../docs/investigations/2026-10-22-tiered-kv-model-positions.md).

## Identity and scope

- Physical card: W7900, gfx1100, `0000:53:00.0`, UUID
  `GPU-dee503dba9717041`, 48,301,604,864 physical VRAM bytes. Same card,
  fresh process per arm; HIP 7.15.26333-0000000.
- Existing `rx7900xt` managed profile: **20 GiB**, capacity only. Device-wide
  50 ms sampled guard: 24 GiB minus 512 MiB; all four runs completed below
  the 23.5 GiB guard threshold. This is not a physical 24 GB card test.
- Artifact: `qwen3.8-27b.mq4-xts`, SHA-256
  `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`.
- Exact lab oracle binary SHA-256:
  `732c4acea35043c342f7a8e5ec65ddc6685b8c468a1de0c78374cbecedd3e09b`.
- Example source SHA-256:
  `80caba5930f69b8fb7ee0b1bb7309e4218677cb425a56c99f1984abc9f1e1a0d`.
- Q8/Q8 KV, explicitly `HIPFIRE_KV_BACKEND=legacy`; Q8 DeltaNet plus EF;
  eager HIP, speculation/vision/adaptive KV off. Incompatible VMM-only
  `HIPFIRE_GFX1100_FA2_SPLIT_VERIFY=0`. Prefill chunk/max batch 1024.
- Committed lighthouse prompt with 288 deterministic archive records:
  **6,175 input tokens**, requested/actual **1,024 generated tokens** per arm,
  length finish. Each input/hash/binary identity is in the raw JSON.

## Observed four-arm results (v2 receipts)

| Arm | Generated | Final visible KV rows | Managed device bytes after decode | Sampled device-wide peak bytes |
|---|---:|---:|---:|---:|
| ordinary AR baseline | 1024 | 7198 | 16,781,463,060 | 17,554,292,736 |
| positional identity, all history | 1024 | 7198 | 16,781,463,060 | 17,554,333,696 |
| bounded sink/recent history | 1024 | 2078 | 15,801,044,500 | 17,699,414,016 |
| deliberately wrong RoPE | 1024 | 2078 | 15,801,044,500 | 17,688,862,720 |

`comparison.json` passed the schema-v2 checks with four independent guard
receipts. Identity equals ordinary AR for **all 1024 full-logit hashes and
token IDs**, prefill state, final valid KV, hidden and DN/conv/scales/EF state.
Correct bounded first-new-FA-K equals AR; wrong-position first-new-FA-K differs.

Bounded retains absolute block starts
`[0,5248,5376,5504,5632,5760,5888,6016]` (1024 rows), then the 31-row partial
prompt tail, then a fixed 1024-row generation reserve. Before dropping the
original KV it independently compared **36,730,880 bytes over all 32 K/V
sides**, including every selected block and partial tail, with exact equality.
Host archive: **213,909,504 bytes = 204 MiB**, representing all 6144 complete
prompt rows. The generation reserve does not publish new archive blocks yet.

## What capacity advantage this does and does not show

The original 7199-row FA KV capacity is **250,640,384 bytes** (~239.03 MiB).
The bounded 2079-row capacity is **72,382,464 bytes** (~69.03 MiB):
**71.12% less device KV payload**. This is not a 71% total-VRAM reduction.

The post-decode managed counter also drops by 980,418,560 bytes, but this
includes draining prefill tensor pools when replacing KV. It must **not** be
attributed entirely to KV offload. Most importantly, **the measured whole-run
peak does not fall**: bounded is about 138.4 MiB (0.83%) higher in this run,
because full initial prefill and temporarily coexisting original/new backing
still occur. A bounded prefill implementation is required before any peak-
memory or real 256K-input fit claim.

The much larger 256K-history result in the separate storage checkpoint uses
synthetic resident pressure, not this 27B model. Do not combine those two
experiments into a claimed 256K real-model deployment.

## Decoded text inspection

Read ordinary and bounded `text` in the raw JSON. Both produce a coherent
lighthouse-restoration opening with consistent characters and dialogue,
without the small fixture's obvious repeated-sentence failure. Bounded changes
wording and details, as expected when history attention changes. Both end
mid-story at the configured length limit; neither completes the requested
5000-word story. This is **qualitative smoke evidence**, not LongBench accuracy,
retrieval recall or a claim that selecting history is quality-neutral.
The wrong-RoPE arm is deliberately broken; plausible prose cannot certify it.

No throughput is reported: per-step full-logit downloads and hashes perturb
timing. No production config, default, serving route or admission is added.

## Raw evidence and earlier failures

`raw-model-runs.tar.gz` contains immutable original JSON, text/logs, per-token
traces and every raw VRAM sample under `tiered-kv-model/`:

- `small-baseline.*`: initial harness read-size failure. Legacy Q8 DN S has
  a byte-count shape but a historical F32 tag; reading `numel * 4` exceeded
  its backing. Fixed the oracle to hash the actual encoded allocation.
- `small-v2-*`: earlier v1-schema 0.8B four-arm runs (799+1024). Identity
  was exact, but the model generated repetitive prose. Not quality acceptance.
- `27b-baseline.*`: refused before GPU inference because the gfx1100 verifier
  flag was incompatible with explicitly requested legacy KV. No OOM occurred.
- `27b-v2-*`: earlier v1-schema 27B four-arm runs (799+1024), identity exact.
  These do not contain the independent staged-history verification and are
  **not accepted by the v2 comparator**. Their file-name `v2` is an attempt
  identifier, not a schema version. Nothing was relabelled or overwritten.
- `27b-v3-long-*`: final schema-v2 four-arm 6175+1024 matrix used above.

Archive SHA-256:
`c0d3e35420eafc57609def6edcf0a733ec5fcb909e4f8fb34ef009e28fac98b9`.
The root `comparison.json` is an unedited copy of the final comparison.

## Checks so far

- Runtime storage CPU tests: 7/7.
- Positional admission/index CPU tests: 3/3.
- Independent staged-history verifier CPU tests: 3/3, including corruption.
- Comparator CPU tests: 25/25, including malformed/missing receipts and mixed
  binary, bad counts, old schema and short-decode negative controls.
- Sampled VRAM guard CPU tests: 19/19.
- Release lab example and native CLI/daemon build: passed.
- `scripts/pre-commit-ci.sh origin/beta`: passed (raw compressed log attached).
  Workspace clippy emits existing warnings; strict `-D warnings` was not
  clean in unchanged dependencies and is not claimed as passed.

Additional product/default-path regression checks are recorded separately;
none can turn this lab proof into tiered serving support.

## Default-path speed investigation, not a tiered speed claim

The existing `speed-gate.sh --fast` did **not** pass: 4B pp32 prefill was
1138.9 versus the committed 1014.0 tok/s floor; decode was 152.9 versus
178.0 (below the 169.1 tolerance-adjusted floor). The raw failure is retained
in `default-speed.log.gz`; no floor, tolerance or hook was changed.

To avoid attributing this to either hardware or this patch without evidence,
build the **clean pinned beta source** separately, save both executables,
warm each cell, then alternate fresh processes on GPU0. Same 4B artifact,
synthetic benchmark input, legacy `asym3` (the existing floor's historical
codec, not a new recommendation), graph on, DPM warmup 3 seconds, pp32,
5 untimed decode warmups, **256 measured decode tokens**. Two ABBA blocks:

| Arm | Fresh-process decode tok/s (all samples) | Median |
|---|---|---:|
| clean beta | 147.2, 147.7, 148.1, 146.9 | 147.45 |
| candidate, override unused | 147.4, 130.8, 146.7, 146.5 | 146.60 |

Median difference: **−0.58%**, not a demonstrated large default-path loss and
not a gain. The 130.8 candidate outlier is **not discarded**: p50=6.71 ms,
p99=16.50 ms, max=133.11 ms, compared with p50 about 6.65–6.71 ms in the
other runs. Its long-tail cause is not localized. Four samples per arm do
not establish equivalence or exclude rare regressions. Both clean beta and
candidate remain below the old 178 floor on this card; the clean-beta control
prevents assigning that whole gap to the new lab hook. This does **not** make
the failed hook green or authorize a baseline change.

Executable SHA-256:

- clean beta bench: `3bd98271fcad015c7529932dbdc106f233dbefd7b36106804a116f06aa076763`
- candidate bench: `4779eff7c6a906bbfa5bee20683a709e1f82c131d8dea5b39664360fffa8ea8b`

The original ordinary CLI/daemon binaries built from that candidate have MD5
`aa8e26d5c0584cee9b33a9e21e1a239c` /
`9ebcc0252eb67779010ebf1ca858e890` respectively. At that earlier checkpoint
only the completed pre-commit CI was reported pass; later results below do
not retroactively change the identity or status of those runs.

## Refresh onto beta 3a5ee848c

The branch was subsequently rebased onto
`3a5ee848c161b87f28d2437c3cb977de736b4a9c`. Only CHANGELOG had conflicts;
upstream and prototype entries were both retained. The replayed storage commit
is `68a420c61100e062f0f9724ea878f3850c489dea`.

### Source and binary binding

`rebased-source.patch.gz` is the actual staged **source** delta against that
pinned upstream beta (all changed crates/kernels and the four diagnostic Python
files, excluding docs/results). Apply its decompressed patch to a clean pinned
beta with `git apply --index`; it includes both storage and positional slices.
SHA-256: `98569060bb54188d68bfbdf0e14106d67f8cb607c6233b50520f3c0f3cce292d`.
This patch supplies the full touched-source snapshot missing from the original
dirty-tree report, rather than pretending an example hash pins all source.

- New oracle binary SHA-256:
  `d57c544fa4f3b8841a61a694e15d3175070fe9a73adc6a18481f2cd349e4d95b`.
- Forward source SHA-256 (unchanged by rebase):
  `d21f593c3f9c2fef75bc618635a7ae83b41f0677df9b122eff13e93e5ae3e002`.
- Example source SHA-256 remains `80caba5930f69b8fb7ee0b1bb7309e4218677cb425a56c99f1984abc9f1e1a0d`.
- New daemon MD5: `71ad91420669e5a5bdbc938b0f0b8ce2`.
- New native CLI MD5: `c5f455cf96d2917fd04b99ca4ab826d5`.

### Refreshed hardware checks

The same GPU0, model, exact 6175-token input, 1024-token decode budget, flags,
20 GiB managed cap and 23.5 GiB sampled threshold were used in four new
processes. All generated 1024 tokens; the v2 comparison again passed complete
identity-state equality, independent staged bytes and wrong-RoPE negative
control. KV capacity remains 239.03 -> 69.03 MiB; it is not a total-VRAM claim.

| Arm | Sampled device-wide peak bytes |
|---|---:|
| ordinary AR | 17,554,296,832 |
| positional identity | 17,562,693,632 |
| bounded | 17,562,943,488 |
| wrong RoPE | 17,573,552,128 |

Peak VRAM again does **not** decrease. No timing comparison is inferred from
these instrumented oracles or from peak differences between sessions.

On GPU1, **ordinary product route, no lab position override**:

- `serve_harness.py` battery: 5/5 turns, natural stop, generated tokens
  `[114,326,70,112,78]`, no runaway/empty/attractor/retrieval-miss flags.
  Read all full outputs: correct merging function, 210-mile arithmetic,
  three-sentence seasons explanation, coherent four-sentence story and exactly
  five code-maintenance tips. These are short product regression checks, not
  the separate 1024-token positional proof and not tiered serving support.
- First battery attempt failed before model load because an outer shell GPU
  lock conflicted with the native daemon's own lock. Retried with the native
  daemon as sole lock owner; no lock was removed or bypassed. Both attempts are
  preserved in the archive. Successful sampled peak: 16,922,497,024 bytes.
- `redline_daemon_harness.py`: stable capture, **659 dispatches / 16 kernels /
  b84cfd51ff978677**, AQL contracts valid; 16-position HIP/PM4/blob parity
  exact for logits, KV, recurrent state and GDN frame. Sampled peak:
  16,838,844,416 bytes. This protects the ordinary `None` route; the lab hook
  still refuses replay and no tiered PM4 admission or speedup is claimed.

`rebased-runs.tar.gz` retains the original refreshed matrix, failed/successful
battery logs/JSON, Redline report and every raw VRAM sample. SHA-256:
`c5bdfad34a9ca4a7a657986b9fcb7d316aca363405d88abbf8c88cde0fe4a270`.
Convenience copies: `rebased-comparison.json`, `serve-battery.json`,
`default-redline.json.gz` (lossless compression of the unedited JSON).

### Refreshed CPU/build status

- Release oracle and native CLI/daemon builds passed.
- Focused lab tests: 7 storage + 3 position + 3 staged verification, plus
  25 comparator + 19 guard tests passed. `rebased-focused-tests.log.gz`.
- `scripts/pre-commit-ci.sh origin/beta` passed. `rebased-precommit.log.gz`.
- The first full no-GPU CI on the old base failed at local TCP listeners
  prohibited by sandbox. Allowing those listeners made the affected CLI suite
  pass **770/770**; this does not make that first full run green.
- The refreshed full no-GPU CI, outside sandbox, **failed** at two unchanged
  registry source/index closure tests. Clean pinned upstream beta independently
  reproduces exactly those two failures (23 other registry tests pass):
  `source_export_evaluator_reproduces_the_sealed_gfx1100_object_and_index_gap`
  and `tensor_ops_index_is_the_source_export_closure_per_arch`. The new
  `hyper_write_bf16x8_gfx1201` export is missing from the expected index closure.
  No registry, kernel-pack index or gate workaround is included here. Raw logs:
  `rebased-no-gpu-ci.log.gz`, `clean-beta-registry-tests.log.gz`.
- Full workspace `--lib --no-fail-fast`, with GPUs hidden, completed with
  four failed cases across three targets (loader, railgun-cert, rdna-compute).
  One was ours: the recording inventory lacked the new lab refusal entry and
  still named the pre-extraction attention helper. Those rows were corrected
  without changing classification or scanning; all six railgun-cert library
  tests then passed. This metadata-only followup does not alter the measured
  Qwen code or inference binaries (railgun-cert is not their dependency).
  The remaining loader case
  `gfx1151_fwht_k_override_refuses_instead_of_legacy_fallback` independently
  fails on clean pinned beta too, as do the two registry cases above. The
  original full-run failure is retained rather than relabelled as green.
  Raw logs: `workspace-lib-tests.log.gz`, `recording-inventory-fixed.log.gz`,
  `candidate-loader-isolated.log.gz`, `clean-beta-loader-isolated.log.gz`.
- The old speed-floor failure and matched A/B investigation above remain
  disclosed. They were measured on the original pinned base, not silently
  promoted to measurements of this refreshed binary.

Final source delta including the recording-inventory followup:
`final-source.patch.gz`, SHA-256
`b9eacb9ca38b7c11b2746d242132e8d46fd6dee42005f7bf2d1fea2c677265a5`.
The earlier `rebased-source.patch.gz` remains the measured source snapshot;
the only subsequent source edit is the recording table, not an inference
dependency. Final pre-commit CI passed again (`pr-final-precommit.log.gz`).
All new functionality stays opt-in (`lab` plus explicit diagnostic invocation);
ordinary serving still supplies no position override.
