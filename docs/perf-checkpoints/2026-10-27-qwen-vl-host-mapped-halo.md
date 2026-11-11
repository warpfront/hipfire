# Qwen VL host-mapped vision on Strix Halo — 2026-10-27

Qwen3.8 27B and Qwen3.5 35B-A3B both run their full image-generation route with
host-mapped direct vision-weight reads. The resulting token IDs, visible text,
reasoning and finish payload agree exactly with resident weights. The existing
per-block streaming route is also compared. The benefit is GPU capacity; no
decode speed improvement is claimed.

## Source and implementation

- Beta base: `d6dd645cbba7d2c9b6c27eaecddb2a618a8eda9a`.
- Executed code: `e2fae6afa34c7cebee159c5fef4653123d2a3c95`. Canonical prompts were added in the subsequent
  data-only commit `a269ca1429ee3b8472e2da10767cf3403ae7726a`.
- Default-feature native daemon SHA-256: `5d90827417de0c6489a1cead1e9a6dca62b11714b9e7e8a13d78f007070a39fe`.
- Nahida: gfx1151 Strix Halo, PCI `0000:c5:00.0`, 96 GiB VRAM carve-out,
  Linux MemTotal 30.98 GiB. All builds, exports and GPU validation ran on Nahida.

`HIPFIRE_VISION_HOST_MAPPED=1` selects host-mapped matrices under
`vision.mode=auto` or `on`. Explicit `resident` wins; `off` skips the tower.
Unset/0 preserves the current streamed default. Only the 110 large per-block
and merger matrices move: patch projection and small tensors retain their
resident placement. Conversion, typed FC2 zero padding, transactional cleanup
and compute kernels are unchanged. The same shared loader serves dense and MoE
carriers; no new public vision mode is added.

## Checkpoints and component preparation

Both requested SSH hosts were inventoried read-only. X570 supplied the 27B
trunk and paired tower plus the 35B-A3B trunk via rsync (3 files,
35,578,421,440 bytes, successful return). The 27B trunk's SHA-256 matches the
repository registry. Exact copied index prefixes, sizes and mtimes were also
checked; full MD5/SHA-256 were computed once on Nahida, then exact stat identity
was used for runs. No existing 125 GB Flash-Next payload was rehashed.

No A3B tower was found at either host. Its 2048-wide output cannot use the
27B tower's 5120-wide merger. A visual-only safetensors component was obtained
from [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754),
revision `59d61f3ce65a6d9863b86d2e96597125219dc754`, using checked HTTP byte ranges
for all 333 visual tensors (893,142,496 payload bytes). The unrequested full
upstream shards were not downloaded or claimed fully hashed.

The frozen native quantizer exported the F16 matrices/F32 vectors. All 333
payloads match an independent mathematical round-toward-zero / exact-widen
reference. The first NumPy reference assumed round-to-nearest and failed on
subnormal values; that failed attempt is retained. The shipped HFQ encoder
explicitly specifies truncation in `hipfire-quantize/src/float16.rs`, so the
corrected reference uses that existing format contract. No precision recipe
was changed.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `qwen3.8-27b.mq4-xts` | 14,987,185,152 | `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae` |
| `qwen3.8-27b-vision.hfq` | 927,611,840 | `269457cb42adc9d2fd20b33d2911e1b3015a30ef2033927f500c7fbe508753c9` |
| `qwen3.5-35b-a3b.mq4` | 19,663,624,448 | `953c570cb242857c4de4d6118d5d532a06c3d8fd218e5b4d4b0835161189081e` |
| `qwen3.5-35b-a3b-vision.hfq` | 913,603,520 | `2ef96b31386e73dab9bf1eb0a3a3af0a00782e4cc8836709722ed74723fafb38` |
| `qwen3.5-35b-a3b-vision-lean.hfq` | 899,288,000 | `4cf4c552dbd64ed50c362ec09c72c8fc38a5c7262a37cc64d3f123a844b91d8a` |

The final A3B table uses `qwen3.5-35b-a3b-vision-lean.hfq`. The initial complete
input export retained text tokenizer/template metadata, increasing the index
prefix from 24,576 to 14,340,096 bytes. HFQ range identities retain cloned
metadata, so the streamed tower held substantial redundant host data. A second
vision-only input directory includes config, image preprocessing and the exact
same safetensors payload, without text tokenizer assets. The resulting tower's
vision config and all 333 tensor payloads are identical to the original export.

This reduces streamed A3B process RSS from 2020.23 to
518.41 MiB (about 1.47 GiB).
The old artifact and its complete experiment remain unchanged. Six tower dumps
and 54 complete generation outputs match old vs lean exactly. This is component
pack preparation; no global source-identity API or admission proof was weakened.

## Full-model AB

Three fresh-process groups per model ran resident, host-mapped, streamed in
that order. Every process runs A first, two more A requests, B, repeated A,
then a text-only seasons request. Each reported warm TTFT is the median of the
three process medians over those three subsequent A requests. No ABBA.

A is the committed `doge.jpeg` close-up; B is `doge_napping.png`. Tower grids
are 34x34 and 40x60 patches (289 and 600 projected image rows). Both prompts
are committed under `benchmarks/prompts/vision_offload_*.txt`; their exact bytes,
MD5s and SHA-256s are in `RESULT.json` and per-run provenance.

Requests use the production native JSONL daemon: greedy seed1234, max_seq4096,
96-token output cap, closed thinking prefix, MTP/DFlash off, eager HIP and the
same committed-token audit on all arms. TTFT is measured by the client before
JSON parsing and includes base64 decoding, image preprocessing, tower, token
framing and text prefill. The existing `generator_prefill_ms` starts AFTER the
tower and is not used as total TTFT. The actual commit-ready/commit/done protocol
is checked; normalized output comparisons retain generated IDs and full text.

| Model | Placement | Client warm TTFT, ms | Delta, ms | Daemon decode, tok/s | Idle-after VRAM, MiB | VRAM released, MiB | Process RSS, MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8 27B | resident | 778.34 | +0.00 | 14.8 | 17573.52 | 0.00 | 372.01 |
| Qwen3.8 27B | host-mapped | 788.49 | +10.14 | 14.9 | 16623.70 | 949.82 | 1354.07 |
| Qwen3.8 27B | streamed | 797.85 | +19.51 | 14.9 | 16732.06 | 841.46 | 469.81 |
| Qwen3.5 35B-A3B | resident | 5061.79 | +0.00 | 62.1 | 20573.67 | 0.00 | 399.50 |
| Qwen3.5 35B-A3B | host-mapped | 5060.95 | -0.84 | 62.1 | 19651.80 | 921.88 | 1356.00 |
| Qwen3.5 35B-A3B | streamed | 5071.47 | +9.68 | 62.2 | 19752.19 | 821.48 | 518.41 |

| Model / placement | Three process TTFT medians, ms | Paired deltas vs resident, ms |
|---|---|---|
| 27b / resident | 778.344 / 758.139 / 786.669 | +0.000 / +0.000 / +0.000 |
| 27b / host-mapped | 788.485 / 788.857 / 779.190 | +10.141 / +30.718 / -7.479 |
| 27b / streamed | 797.642 / 797.849 / 799.830 | +19.298 / +39.710 / +13.161 |
| a3b / resident | 5069.854 / 5061.790 / 5059.457 | +0.000 / +0.000 / +0.000 |
| a3b / host-mapped | 5079.821 / 5060.948 / 5050.683 | +9.967 / -0.843 / -8.774 |
| a3b / streamed | 5083.315 / 5064.470 / 5071.466 | +13.460 / +2.680 / +12.009 |

Device-wide VRAM was sampled with both physical leases held; deltas use the
same requests and idle point. RSS is a separate process view, not a physical
host-memory ledger. Shared mappings and GPU/process counters overlap on UMA
and must not be added. The memory result is relocation to the Linux host-RAM
partition, not removal of the weight storage.

The latency changes are small compared with complete-model work and sampling
variation. Decode is effectively unchanged. A3B's image prefill still executes
per token (the native log names `prefill: per-token (MoE layers)`), accounting
for roughly five seconds to first token on A. Batching that existing image
path is a separate, larger speed opportunity.

## Verification

- Required `pre-commit-ci.sh d6dd645cbba7d2c9b6c27eaecddb2a618a8eda9a` passed. GPU-free tests: VL 25/25,
  loader vision 16/16. No full QC/LongBench campaign.
- Both original towers: 12 fresh processes over two images, finite/nonzero and
  repeated-output checks, 8/8 candidate-vs-resident byte comparisons.
- Lean A3B tower: 6 processes, 4/4 placement comparisons and 6/6 old-vs-lean
  byte comparisons. Every converted source payload also agrees independently.
- Primary full-model results: 18 fresh processes, 72/72 exact final-output
  comparisons. The original full-metadata A3B formal run is retained separately;
  its results are not pooled into the final table.
- Explicit resident with the host-map flag remains resident and agrees with
  the reference. Off with the flag rejects an image and preserves the text
  answer. The final two model policy sets pass 10 exact output checks.
- Decoded answers were inspected: both models identify the Shiba close-up and
  its wide eyes, the other dog lying on its back with raised paws, and axial
  tilt as the cause of seasons. Other placements return identical IDs/text.

All formal timing follows completed transfers, hashing, compilation and an
initial full request sweep. Only the initial functional/tower warmups may overlap
preparation I/O. Both `/tmp/hipfire-gpu.lock` and
`/run/lock/hipfire/gpu-pci-0000:c5:00.0.lock` remain inherited by GPU children;
the private Cargo target and internal daemon lock directories are isolated.

Evidence is specific to these two artifacts, two images, single-GPU Halo eager
AR and native JSONL transport. It preserves the resident model output; it is
not a new HF task-accuracy admission, TP/PP result, text-expert offload result,
or proof that the entire remaining 32 GiB can be consumed.

## Portable review evidence

The sibling [JSON receipt](2026-10-27-qwen-vl-host-mapped-halo.json) contains
all 18 primary process rows, loaded KV/backend acknowledgements, exact output
hashes, reference token IDs/text, model/binary/fixture pins and check exit
receipts. The prepared PR preserves the tested runtime sources byte for byte;
the receipt records each changed runtime file's SHA-256 and original Git blob.
The A3B metadata reduction above is preparation of the measured component,
not a change to the production quantizer or range-identity implementation.

## Evidence location

Nahida: `/home/husrcf/Code/hipfireQC/tmpOutput/hipfire-vision-qwen27-a3b-halo-20261027`.

Key files: `RESULT.json`, `cpu-build-v1/binaries.json`, `checkpoint-manifest.json`,
`checkpoint-manifest-lean.json`, `a3b-export-lean-proof-v1/`, `tower-v1/`,
`lean-tower-v1/`, `formal-v1/`, `lean-formal-v1/`, `policy-v1/`, `lean-policy-v1/`.
They retain raw commands, process/lease receipts, returned events, generated
IDs/text, memory samples, hashes and all failed/reference-preparation attempts.
Dev-only orchestration scripts are in the same output directory.
