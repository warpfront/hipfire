# gfx1100 TP2 whole-token graph promotion receipt — 2026-10-16

**Lifecycle:** `historical`

**Disposition:** measured promotion evidence for the existing exact homogeneous
gfx1100 TP2 route. This record is not itself a product default, current
baseline, or admission decision.

## Question and scope

Determine whether the alternating-partial plus coherent whole-token hipGraph
route landed from #830 can become the gfx1100 default inside its existing
fail-closed envelope. The comparison changes only:

- A: `HIPFIRE_TP_PEER_ALTERNATING=0 HIPFIRE_TP_FULL_GRAPH=0`
- B: `HIPFIRE_TP_PEER_ALTERNATING=1 HIPFIRE_TP_FULL_GRAPH=1`

The route remains limited to an exact homogeneous gfx1100 TP2 pair, peer
direct access, alternating partial buffers and plain AR. Mixed architectures,
other rank counts and speculative decoding are outside this receipt.

Source under measurement:
`c8619378f9fd7fa673e75bacdd4854f4b5b274fb` (`warpfront/beta`). The TP2
whole-token implementation and admission are unchanged through the promotion
base `c1a6be48e`.

## Identities

- GPU 0: Radeon Pro W7900, exact `gfx1100`, 48,301,604,864 B VRAM,
  UUID `GPU-dee503dba9717041`, PCI `0000:53:00.0`.
- GPU 1: Radeon Pro W7900 Dual Slot, exact `gfx1100`, 48,301,604,864 B
  VRAM, UUID `GPU-ecb57a743c42951e`, PCI `0000:9e:00.0`.
- HIP runtime reported by hipfire: `7.15`; `hipconfig --version`:
  `7.15.26333-0000000`.
- Model: `qwen3.8-27b.mq4-xts`, SHA-256
  `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`,
  MD5 `6702fe8a56c0c8ec85aca1804b8df29e`.
- `hipfire` SHA-256 / MD5:
  `8850064d7648ed68655c3157f9e826ed764bbdd7078ebc86c56217ef8edef70b` /
  `d0089f1fa015f736ba33d33f8af272ac`.
- `daemon` SHA-256 / MD5:
  `5b2f6133422c3a3afa4f5e2ea5221737ccfd94d7842b8cfdf846283051dca631` /
  `10be1f635a9a5989b0cdc245c949f182`.

## Path-specific exact oracle

`tp2_cross_device_graph_reduce` ran with both cards visible and
`HIPFIRE_TP2_GRAPH_REPLAYS=100`:

```text
PASS gfx1100 TP2 graph fused reduce: 100 exact replays
```

The oracle overwrites both ranks' inputs before every replay and requires exact
rank-ordered reduced output on both devices.

## Fresh-process performance A/B

Common settings: TP2, Q8/Q8 VMM KV, pure AR (`--spec off`), greedy, thinking
off, `max_seq=16384`, 512 generated tokens, `--backend noslots`,
`--workload stateless`. Prompt
`benchmarks/prompts/mq4v2_k5120_long_ar.txt`, MD5
`9f64b7bc79c38aaa65cccc511b167390`, was 126 tokens as reported by the daemon.
This is a decode claim, not a prefill claim.

Each sample used a fresh CLI/daemon process and one built-in discarded warmup.
Order was `A,B,B,A,A,B`. `HIPFIRE_AR_GRAPH_TRACE` was unset for all timed
samples.

| arm | decode samples (tok/s) | median | population stddev |
|---|---|---:|---:|
| A — segmented TP2 | 48.0, 47.8, 47.8 | 47.8 | 0.094 |
| B — alternating + whole-token graph | 53.4, 53.2, 53.1 | 53.2 | 0.125 |

Median delta: **+11.30% (1.113x)**. Every report acknowledged effective
`kv_backend=vmm`, `kv_backend_legacy=false`, and `kv_mode=q8`. No sample logged
a panic, illegal memory access, load error, timeout, or poisoned route.

Separate trace-only runs witnessed the 128-barrier signal tape and
`captured whole-token tp=2` marker in B and their absence in A. Trace timings
are excluded because the segmented route logs every replay while tracing.

The promotion candidate based on `12daf4fe99017a88242469f60853b860609c20ab`
was also rebuilt and smoke-tested with the same two cards. With
`HIPFIRE_TP_FULL_GRAPH` and `HIPFIRE_TP_PEER_ALTERNATING` both unset, the
product path reported the 128-barrier signal tape and
`captured whole-token tp=2`; setting only `HIPFIRE_TP_FULL_GRAPH=0` restored
the 129-root/128-peer segmented graph. These trace-enabled 64-token runs are
route and kill-switch evidence only, not additional performance samples. The
candidate oracle passed 100/100 exact replays both before and after the final
rebase to `c1a6be48e`.

## Product E2E parity

`scripts/serve_harness.py` ran the six-prompt battery
`benchmarks/gemma4_eseries_long_decode.json` (MD5
`14fa7b959ddd90e6680fec319d8097ef`) with the same model, TP2, Q8 VMM, pure AR,
greedy decoding, thinking disabled, `max_seq=16384`, and 512 output tokens per
prompt.

- 6/6 turns completed in each arm; 3,072 generated tokens per arm.
- 0 empty outputs, attractors, retrieval misses, or stream errors.
- All six assistant outputs were byte-identical arm-to-arm.
- Concatenated assistant transcript MD5 in both arms:
  `84410b3e45e9ddbb506c33f5d12526de`.

Every turn reached the intentional 512-token length cap. Decoded previews were
inspected and remained readable and on-task. Timing from this trace-enabled
battery is not part of the performance claim.

## Interpretation

This evidence supports default-on only inside the existing exact homogeneous
gfx1100 TP2 eligibility gate, with `HIPFIRE_TP_FULL_GRAPH=0` retained as the
kill switch. It does not establish TP1-to-TP2 scaling, performance on another
host/model/prompt, speculative decode, mixed architectures, TP counts other
than two, or a prefill improvement.
