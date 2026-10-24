# Opt-in tiered KV: bounded initial prefill

Scope: dense Qwen3.8-27B MQ4-XTS, text AR, legacy Q8/Q8 KV, Q8+EF
DeltaNet state. This is a lab state/storage oracle, not production admission,
an end-to-end speed benchmark, or evidence that selected attention is
quality-neutral. Normal serving has no tiered-mode configuration.

Pinned upstream: `beta@3a5ee848c161b87f28d2437c3cb977de736b4a9c`.
Parent implementation: `1e18db10924a215ce452c4e83d72a82dbf2410bf`.

## Identities

- Target: `qwen3.8-27b.mq4-xts`, 14,987,185,152 bytes; SHA-256
  `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`.
- Measured lab executable SHA-256:
  `d0a64188858d24f6c453660df158f1f3f50c533d43cea74e816325d6efed535c`.
- `qwen35/prefill.rs` SHA-256:
  `10e1326e7093c31943e7073c93e1c8afcbe8c5a0b5f1cbaf11ca2171cf36a220`.
- `carrier.rs` SHA-256:
  `395bf6a5e525e68c0176ddedc087e3bf58bde14f21db475b027e7f8fb09f1e05`.
- Streaming example SHA-256:
  `f9200d4bdb1273a66c49c6bb6f403e1926cf589f33f6e9d878a28b00bf65fbb1`.
- Default non-lab daemon MD5: `54316d70e2be45110c400d6fd6a6a129`;
  CLI MD5: `20ddfbfe8cf7f03993274b670ea3c62c`.
- HIP 7.15.26333; GPU0 W7900 `0000:53:00.0`, GPU1 W7900 Dual Slot
  `0000:9e:00.0`, both actual gfx1100 with 48,301,604,864 B VRAM. No hardware
  identity override. `HIPFIRE_EMU=rx7900xt` imposes a **20 GiB allocation cap**,
  not a hardware/performance emulation. Independent sysfs guard: 24 GiB minus
  512 MiB, sampled every 50 ms.

## Completed runs

All arms use fixed 257-row PBS (no GDN tape), 128-row archive blocks, eager
HIP, graph off, no speculative decoder. Every arm generated the requested
1,024 tokens. Full-history ordinary/identity arms intentionally hold all
history and are not called memory-saving bounded arms.

| Arm | GPU | Input tokens | KV rows | KV bytes | Sampled device peak B |
|---|---:|---:|---:|---:|---:|
| `baseline-v2` | 0 | 6,175 | 7,423 | 258,439,168 | 16,477,356,032 |
| `identity` | 0 | 6,175 | 7,423 | 258,439,168 | 16,518,991,872 |
| `bounded` | 0 | 6,175 | 2,175 | 75,724,800 | 16,385,163,264 |
| `bounded-32k` | 1 | 32,383 | 2,175 | 75,724,800 | 16,751,845,376 |
| `bounded-128k` | 1 | 129,151 | 2,175 | 75,724,800 | 16,793,690,112 |

`comparison-v2.json` passes strict ordinary/identity chunk logits, before/after
source/staged KV/DN/hidden state, complete decode trace/text and final state,
plus bounded archive/map/carry and independent guard checks. Full-history
source bytes must remain unchanged during staging; that reference is read
directly from the GPU, not reconstructed from the archive's source plan.

`long-context-validation.json` records standalone schema/map/carry/guard
validation for the two GPU1 long inputs. Device KV is unchanged at 72.2168 MiB;
host archive grows from 1,123,024,896 B to 4,492,099,584 B. This establishes a
completed long-input path with a fixed device KV budget, not bounded host
history. Whole-device peaks are sampled, not hard high-water or 16 GiB-card
fit guarantees. Installation briefly owns two **bounded** KV slabs; all
allocations still pass the same managed cap. Point ledger readings are not
allocator peak measurements. No timing improvement is claimed.

The lighthouse continuations were read: coherent prose and dialogue, cut at
the requested length, not a repeated-token attractor. This is a smoke check;
no LongBench accuracy or quality-neutrality result is asserted.

Normal-route regression, not tiered serving/replay admission:

- Native serve battery: 5/5, lengths 114/326/70/112/78, natural stop, no empty,
  attractor, runaway, stream error or post-terminal bytes. Output includes a
  correct sorted-list merge, 210-mile arithmetic, seasons explanation,
  four-sentence story and five coding tips.
- Redline: stable 659 dispatches / 16 kernels / `b84cfd51ff978677`, AQL
  contracts valid; 16 positions of HIP/PM4 logits, KV, recurrent and GDN-frame
  parity pass.

## Raw artifacts, failures and CPU checks

`runs.tar.gz` is a lossless archive of `results/tiered-prefill-streaming/`:
all oracle JSON/text/trace, original logs, sampled guards, default serve and
Redline evidence, both successful and failed comparison attempts.
SHA-256: `6ec4c6ab78a8b4e87085df3957afabff988b1051b43201fe658ba7763b669f4f`.

Preserved failures:

- Initial `baseline.json` was refused before weight allocation because the
  gfx1100 FA2 verifier defaults to VMM-only. The exact subsequent legacy
  experiment explicitly sets `HIPFIRE_GFX1100_FA2_SPLIT_VERIFY=0`.
- `comparison.json` is the original failed analyzer output: its first version
  mistook normal post-child-exit guard cleanup signals for a memory trip.
  The analyzer now follows the guard's actual completed/exit/peak contract;
  a CPU test covers that distinction. `comparison-v2.json` is new output,
  not an edited receipt. Limit-exceeded and nonzero exits are still refused.

Compressed CPU/build logs are verbatim outputs. Focused passes: storage and
stream planner 12, carrier 13, positioned prefill 4, example source-copy 3,
recording inventory 6, guard 19, previous comparator 25, streaming comparator
29: **111 tests total**. `precommit-final.log.gz` records the final required
GPU-free pre-commit checks. No thresholds, ratchets, scan exclusions or gate
semantics were weakened. Previously documented upstream workspace failures
in the parent PR are not reclassified as passing by these focused runs.

`source.patch.gz` is the mechanically generated staged Rust/TOML/Python diff
against the pinned upstream, including the previous lab slices. SHA-256:
`91038b018cc4d8fa465e7d1feea3ba6f976d163a6649a3f150af6088523cd0cd`.
It excludes generated evidence and prose, and does not imply the final commit
SHA was measured before it existed.

## Reproduce / remaining scope

See [the implementation and commands](../../../docs/investigations/2026-10-22-tiered-kv-streaming-prefill.md).
For the two long GPU1 runs replace `--prefix-records 288` with `1536` and
`6144`, use `--mode bounded --window-blocks 8`, and choose GPU1's UUID/BDF.

Still absent: ordinary serve/session integration, query-based selection,
decode restaging, accepted-only DFlash transactions, checkpoint/resume,
vision, asynchronous copies, cross-architecture hardware validation, and
task-quality evaluation of selected attention. #925 is not closed by this
prototype.
