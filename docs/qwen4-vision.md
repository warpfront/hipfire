<!-- SPDX-License-Identifier: Apache-2.0 -->

# Flash Next vision: implementation and reproduction

Branch implementation for [issue #938](https://github.com/warpfront/hipfire/issues/938).
The registry has **no published, paired Flash Next vision sidecar**. Export one
and pass its path explicitly. The current validation route is
[Flash-Next image pathway in VALIDATION.md](VALIDATION.md).

Post-merge Halo validation is recorded below for source
`428e2e63627228cd3b5f79759ef40550979e123a` and the exact binaries in the
[2026-10-27 receipt](../benchmarks/vision/halo-gate-20261027-beta3a01.json). Default
`auto` loads with and without the tower have identical text; image serving and
automatic admission passed. **Stock beta could not load on this 96 GiB BIOS
carve-out configuration**; its separate text reference required
`HIPFIRE_OOM_GUARD=0`. That comparison is explicitly non-default. The earlier
2026-10-23 CPU tower reference and historical capture diagnostic remain scoped
to their original artifacts.

## Checkpoint and artifact

The pinned source is `Qwen/Qwen3.8-Flash-Next` at
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. Its
[official tensor index](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json)
contains **333 `model.visual.*` tensors**, all in
`model-00001-of-00131.safetensors`. These include the projector/merger. The
[official config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)
specifies a 27-block, 1152-wide vision tower with 16 attention heads, patch size
16, temporal patch size 2, spatial merge size 2, and **2560-wide output**. The
Qwen3.8-27B tower's 5120-wide output is incompatible with this trunk.

The implementation reuses hipfire's Qwen35-VL tower equations and image
preprocessing, with a Flash Next-specific configuration and complete-tensor
preflight. The text trunk remains architecture 16; image capability requires a
resident, validated tower.

Build and export from the pinned remote checkpoint:

```bash
cargo build --release
cargo run --release -p hipfire-quantize --bin hipfire-quantize -- \
  --qwen4-flash-next --vision-only \
  --input hf://Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --output "$HOME/.hipfire/models/qwen3.8-flash-next-vision.hfq"
```

Production component mode is the default. This route fetches the visual tensor
payload ranges and necessary metadata, without downloading the text weights.
It preserves the original configuration and preprocessor configuration; matrices
are stored as F16 and vectors/tables as F32. The generated local artifact was
checked against all 333 source tensors:

| Property | Local validation artifact |
| --- | --- |
| Filename | `qwen3.8-flash-next-vision.hfq` |
| Size | 904,008,640 bytes |
| Tensor layout | 111 F16 matrices, 222 F32 vectors/tables |
| SHA-256 | `acef5567d9ee0f253d9f8ebde994bd9c37e285754db6d227b3f1df2fea67cc08` |
| Visual BF16 source subset SHA-256 | `6eb9e554981f944118b1d8fd4d12ec5213d996a4510ecd21d3398f8f7d775804` |

The subset hash identifies a safetensors file with a rewritten visual-only
header and unchanged source payloads. These are local receipt identities,
not a registry publication or a claim that a future export has identical metadata.

## Serving and execution scope

With the separately downloaded text trunk already available:

```bash
HIPFIRE_DAEMON_BIN="$PWD/target/release/daemon" HIPFIRE_VISION_MODE=resident \
  target/release/hipfire serve --model qwen3.8:flash-next --spec off \
  --vision "$HOME/.hipfire/models/qwen3.8-flash-next-vision.hfq"
```

`--vision` overrides the environment/registry sidecar path. The shared
[vision configuration contract](CONFIG.md#vision-tower) accepts all four modes;
an omitted mode comes from `hipfire_config::vision_mode_default()` (currently
`auto`). **Flash Next is resident-only in this implementation:**

| Mode | Flash Next behavior |
| --- | --- |
| `auto` (default) | Load a discovered or explicit sidecar as a resident tower. If no sidecar is available, retain the text-only path. An explicit missing or invalid sidecar fails the load. |
| `resident` | Discover the sidecar as in `auto` and upload the whole tower at load. If no sidecar is available, retain the text-only path. An explicit missing or invalid sidecar fails the load. |
| `on` | Require a valid sidecar, using the same resident backend. Missing or invalid sidecars fail the load; there is no silent text-only fallback. |
| `off` | Skip the tower, including an explicit sidecar, and refuse image requests. |

The `auto`/`on` resident mapping is a Flash Next exception. The shared
Qwen3.5/3.8-VL route keeps beta's per-block streamed tower for `auto`/`on`; its
`resident` policy and the other vision routes are unchanged. Image discovery
reflects a successfully loaded tower, not merely a configured filename.

Before text placement, the loader validates the sidecar and accounts for its
resident device weights (including FC2 row padding) and maximum admitted image
scratch (including the projected 2560-wide rows). It passes this reservation
through `SequenceHint.reserve_bytes` and the shared `AutoReserve` budget, then
uploads the tower after placement. The reservation is bookkeeping, not a dummy
GPU allocation. Record the `vision_reserve=` budget entry and placement logs
when checking admission. Device allocation backing is queried separately from
integrated-GPU identity: a BIOS VRAM carve-out is not charged again against the
host RAM left outside it.

The implemented image route is **one image, one GPU, cold autoregressive
generation**. Video, multiple images, and distributed image execution are
rejected. Speculative image decoding and image prefix/session reuse are outside
this route. Standard sampling, stop sequences, and the normalized Qwen reasoning
controls are forwarded to the common AR producer. Final resize dimensions are
checked before allocating the resized image; the attention hard limit is 16,128
patches. Temporary GPU owners are transactional across upload, each layer, and
the merger. A failed fence quarantines allocations instead of recycling memory
that may still have in-flight readers.

The carrier follows these upstream semantics:

- Replace the expanded image-pad embedding rows with the tower's 2560-wide
  outputs **before HC stream expansion**. Keep the original image-pad token IDs
  for PLE/hash inputs.
- Retain the complete prompt's T/H/W positions. QSA query, key, and indexer query
  use interleaved mRoPE sections `[11, 11, 10]`; decoding uses the sequence
  position plus the image's `rope_delta`.
- Pool raw indexer keys, normalize, then rotate using the **first token's**
  T/H/W position in each compression block. Chunking cannot replace this with a
  scalar position or the block's last position.
- During the image request, bypass the NPU trunk plan, one-dimensional QSA
  fusions, lane promotion, retained PM4/AQL replay, and token-only prefix/session
  caches. Graph capture is refused for this carrier. Request reset releases the
  image context after GPU completion and restores normal text request policy.

The implementation lives in `hipfire-arch-qwen4`'s `vision` and
`bundle/image_context` modules, `hipfire-generate/src/qwen4_vision.rs`, and the
optional QSA mRoPE bindings in `rdna-compute/src/qsa_mrope.rs`. The three image
exports live in their own `qsa_mrope` module; the shared `tensor_ops` source is
byte-identical to beta, preserving the measured text-route pack hashes.

## Halo validation (2026-10-27)

This is fixture-bound evidence, not an admission record or a speed claim.
Tested source: `428e2e63627228cd3b5f79759ef40550979e123a`, including beta
`3a01cc447bceec99ec28ed81d07460a61b395f4f`. Hardware: gfx1151, HIP
`7.16.26385-0000000`, kernel `7.0.0-38-generic`, 96 GiB BIOS-reserved VRAM.
The [receipt](../benchmarks/vision/halo-gate-20261027-beta3a01.json) includes complete
text, request/prompt identities, image answers, load logs, memory snapshots,
and model/binary digests. Later documentation-only commits do not extend it
to changed runtime code.

Each arm used a fresh process and isolated `HIPFIRE_HOME`, the same monolithic
`8b15b6fe...` trunk, external Q4LM row-order PLE, and default gfx1151-zab-v1
trunk overlay. This is not a fresh split-layout pull. `vision_mode`, `max_seq`,
MTP, NPU trunk and graph policy were omitted from the serving command and
resolved through product defaults; only the tower arm supplied `--vision`.
The text harness attached with `--no-spawn --mtp auto --sampling greedy
--seed 0 --thinking off --thinking-effort none --max-tokens 512`. The prompts
are the committed five-genre battery in `scripts/serve_harness.py`.

| Check | Observed result |
| --- | --- |
| PR default text, no sidecar vs auto-loaded sidecar | 5/5 complete transcripts byte-identical, including reasoning/tool fields, request hashes, input/output token counts and terminals |
| Separate beta reference | Same 5/5 transcripts, but **OOM guard explicitly off**; not a stock-default pass |
| Auto-loaded image tower | 17/17 HTTP/lifecycle cases plus image capability discovery |
| No-sidecar text-only capability | 11/11 image-refusal and follow-up text cases |
| Text chain after image/cancellation | 5/5 coherent, clean stop terminals |
| CPU vision/load/image/resource/admission tests | 61 passed across six filters; the exploratory zero-match filter is not counted |
| Frozen flagship pack / kernel registry / sealed compiler checks | 5 / 61 / 1 passed; historical source-hash fixtures unchanged |
| GPU channel | 18 passed, zero failed/skipped |
| Independent QSA mRoPE oracle | Five cases pass; max norm/rotation error `8.9407e-7`, chunk/pooling comparisons exact |

RGB answers agree across stream modes. OCR transcribes all five direction
signs, including accents; both dog answers identify it sleeping on its back.
These were checked against the source images. Text serving actually ran native
MTP (`K=7`); default NPU planning selected GPU-only plans (`families=none`), so
this is not proof of NPU-executed inference. The transcript digest shared by
all three successful arms is `578a6f19180b6e65e95a36eb0fe6bd32`.

**Beta limitation.** Stock `3a01cc44` rejected the load because its host-union
check counted device allocations in the BIOS carve-out against the remaining
host RAM: 80.7 GiB required plus 4 GiB headroom versus 22.5 GiB available.
The PR's allocation-backing query avoids that double count. The separate
beta binary was rerun with `HIPFIRE_OOM_GUARD=0` only for the text reference;
that switch also disables other host-headroom checks. Routed experts were fully resident in both builds; KV/GDN format, automatic
context and MTP route matched. Retained PLE was availability-dependent: the
beta reference kept 5,590 MiB, versus 8,162 MiB in the no-tower PR arm. PR admission
was tested with the default guard enabled. Strict stock-beta default-load
parity is therefore **not established on this BIOS configuration**.

The loader reported these automatic-admission values (MiB are truncated log
values). The first free-memory sample is after text-rank construction, before
tower upload; it must not be mistaken for final resident free memory.

| Budget | No sidecar | Auto-loaded tower |
| --- | ---: | ---: |
| Free device memory at rank admission, MiB | 18,864 | 18,864 |
| Remaining context/gather/MTP request growth, MiB | 8,238 | 8,238 |
| Resident PLE retained, MiB | 8,162 | 6,519 |
| Device headroom, MiB | 2,048 | 2,048 |
| Admitted automatic context / card cap | 262,144 | 262,144 |
| Independent vision reserve, bytes | 0 | 2,158,224,320 |

The vision quote is 896,353,216 bytes of resident weights including FC2 padding
plus 1,261,871,104 bytes for maximum-image scratch, projected rows and allowance.
It reduces reclaimable PLE residency rather than taking the context's reserved
space. The observed PLE difference is 1,643 MiB, not the full vision reserve;
residency also depends on currently available memory. Both PR loads and
subsequent requests passed; all test services stopped,
and device usage returned to 302,886,912 bytes (desktop baseline). The admitted
262,144 capacity was not exercised with a full-length prompt; snapshots are
not peak-memory measurements. No throughput/TTFT or new PM4 claim is made.

PR binary MD5s: CLI `2ab92e9b4bdf3cf231f32d32d3195a9b`, daemon
`f1c9286c4f73ccc788dbe9c28eb04bfe`. Beta reference MD5s: CLI
`fe34a1c29e982f55df184b38c50bac85`, daemon
`9ac9197ce3e644b815e9d9ad95bf79d5`. Full SHA-256 identities and commands are in
the receipt. Local raw files are under
`.codeinsight+research/vision-938/halo-gate-20261027-beta3a01/`.

The [earlier same-day receipt](../benchmarks/vision/halo-gate-20261027.json)
remains unchanged and applies only to source `95e69b966` / beta `035c0ddd`.

## Validation

The commands below define the route for future runtime changes. The dated
post-merge receipt above covers its pinned source and fixture; historical
receipts later in this section do not validate newer changes.

CPU checks:

```bash
cargo test -p hipfire-arch-qwen4 --lib vision
cargo test -p hipfire-arch-qwen4 --lib image_context
cargo test -p hipfire-loader --lib vision
cargo test -p hipfire-generate --lib qwen4_vision
```

For the independent QSA kernel reference, build before acquiring the shared GPU
lock; run it separately from a resident serving process:

```bash
cargo build -p rdna-compute --features lab --example qsa_mrope_parity
mkdir -p .codeinsight+research/vision-938
source scripts/gpu-lock.sh
gpu_acquire qwen4-vision-mrope
trap gpu_release EXIT
HIPFIRE_REPLAY_BACKEND=hip timeout 120s target/debug/examples/qsa_mrope_parity \
  > .codeinsight+research/vision-938/qsa-mrope-parity.json \
  2> .codeinsight+research/vision-938/qsa-mrope-parity.log
gpu_release
trap - EXIT
```

The gfx1151/HIP 7.16 receipt covers rectangular image positions, gated Q,
contiguous K, indexer Q, chunk boundaries, decode delta, and incremental F32/BF16
key pooling. Maximum norm/rotation absolute error was `8.94e-7`; chunked versus
unchunked outputs and both pooling references matched exactly. Full local
receipt: `.codeinsight+research/vision-938/QSA-MROPE-RECEIPT.md`.

For tower validation, [dump_qwen4_reference.py](../benchmarks/vision/dump_qwen4_reference.py)
loads only the 333 original BF16 visual tensors into the official Transformers
vision model on CPU. Feed its `pixel_values.bin` and grid dimensions to
`cargo run --release -p hipfire-arch-qwen4 --features lab --example qwen4_vision_dump --`
with `--model`, `--patches`, `--grid-h`, `--grid-w`, and `--out`. Rerun the CPU
script with `--compare` pointing to that dump to compare every stage. Receipts
record source, input, configuration, implementation, and binary identities.
The existing gfx1151 comparison used a `[1,12,18]` grid and `[54,2560]` projected
output: final maximum absolute error `6.69e-4`, relative L2 `0.001404`, cosine
`0.999999017`. Its local receipt is
`.codeinsight+research/vision-938/tower-stage-comparison-complete.json`.
These are component numerical checks; neither supplies E2E quality or timing
acceptance.

Against the running service, save the HTTP report and inspect every answer:

```bash
python3 scripts/check_qwen4_vision.py --base-url http://127.0.0.1:11435 \
  --model qwen3.8:flash-next --expect-vision on \
  --output .codeinsight+research/vision-938/http-on.json
```

This covers streaming/non-streaming image requests, image replacement, OCR and
description fixtures, disconnect/follow-up behavior, and text after an image.
`--expect-vision` describes the service's loaded image capability; it does not
set `vision_mode`. Use `--expect-vision on` when `auto`, `resident`, or `on` has
loaded the tower. Restart with `HIPFIRE_VISION_MODE=off` and use
`--expect-vision off` with a separate report: discovery must omit image support,
image requests must be refused, and text must still work. An `auto` or `resident`
load without a sidecar also uses `--expect-vision off`.

Check load admission separately from this HTTP script: a missing sidecar under
`on`, and an explicit bad sidecar under `auto` or `resident`, must fail closed.
Compare default-mode text output in fresh processes with and without the valid
sidecar, using identical prompt bytes and settings. Save the load/placement logs
and `vision_reserve=` entry as well as the HTTP reports; the script does not
exercise load failures or prove the memory budget. Record exact trunk, sidecar,
CLI, and daemon identities alongside every receipt.

Historical local result (2026-10-23, gfx1151/HIP 7.16, single GPU, q8 KV, eager HIP,
`--spec off`, `--max-seq 8192`, NPU trunk off): **17/17 image HTTP cases passed**.
Red/green/blue answers matched across streaming modes; all five French direction
signs were transcribed correctly; the dog was correctly described as lying on
its back with closed eyes and raised paws. Stop sequences and low-effort capped
reasoning were honored. An extreme aspect-ratio image was rejected before tower
execution, and cancellation followed by text and another image remained usable.
The separate vision-off run passed all ten image refusals plus a text request.
Decoded OCR and description answers were inspected against the committed images.
These are functional results, not a performance comparison. Mixed speculative
text/image serving and NPU-enabled text loads were not exercised in this run.

Historical image-report identities (2026-10-23):

- Trunk SHA-256: `8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972`.
- CLI MD5: `9418bfeb06e3829e41253a61f51465c8`.
- Daemon MD5: `e69bc229887fc6a0ba4d4f5625c5a4c7`.
- Local full report: `.codeinsight+research/vision-938/http-final.json`, including
  exact image and prompt digests, complete answers, and stream terminals.
- Final text battery/chain reports and capture diagnostic live beside it;
  the first text attempt with only `--thinking off` retained template reasoning
  and exhausted its small output budget. Successful text runs explicitly pass
  `--thinking-effort none --max-tokens 512`; the unsuccessful receipt is retained.

### Historical text capture diagnostic on Halo

The additional 2026-10-23 text-only `redline_daemon_harness.py --qwen4 --pm4` check uses
BF16 QSA state, context 128, maximum context 2048, and five shadow positions
129–133. On gfx1151, `HIPFIRE_QWEN4_QSA_DECODE_SPLITK=0` is required to compare
the same attention reduction route: eager split-K is enabled by default on
Halo while recording uses the per-head kernel. With split-K disabled, all
positions matched ordinary HIP, recorded HIP, and PM4 bit for bit, with stable
1,177-launch capture and 33 passing kernel contracts. The same binary with
split-K enabled failed that exact-state comparison. Both reports are retained;
the default configuration is not claimed to pass. This diagnostic does not
admit a Flash Next PM4 product route on Halo, and image requests remain eager.
