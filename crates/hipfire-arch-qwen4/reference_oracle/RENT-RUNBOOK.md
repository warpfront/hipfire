# Flash-Next true-bf16 teacher: single-MI300X rental runbook

## Scope and current status

The local pilot was cancelled at Kaden's direction. No Flash-Next BF16 weight payload was downloaded on hiptrx, and no real teacher traversal, floor measurement or estimator-stability result was produced. The attempted tokenizer preparation had already exited; its 22 MiB partial cache was deleted. Do not restart a local teacher download or pilot.

The implementation is in branch `work/fn-kl-method`, based on `origin/beta` at `d7ae367`, in `/home/kaden/ClaudeCode/warpfront/hipfire-kl-method`. It is not pushed. Transfer the committed code bundle named in the handoff; do not substitute an older `beta` checkout. The source inspection and quality contract are in `KL-METHOD.md` alongside this file.

This runbook prepares a fresh **MI300X (gfx942, 192 GB)** rental and runs the real source-bf16 teacher plus its reduction-order floor. It does not run Paiton or any competitor plugin. Source weights live on the rental's own disk/host memory; the forward is layer-major with only one expert layer live on the GPU. Main must approve the full-corpus run after the rental pilot.

## 1. Provisioning and ROCm pin

Choose Linux x86_64, a working amdgpu driver with `/dev/kfd` and `/dev/dri` accessible, one exclusive MI300X, **at least 1 TB free fast local disk**, and preferably **512 GiB host RAM**. Disk-backed source access is supported; 512 GiB makes a warm core-weight page cache practical. Avoid a small root overlay for checkpoint/output storage. The source-plus-PLE estimate is roughly 352 GB, not a disk guarantee: setup sums the pinned HF file sizes and rejects insufficient capacity before downloading shards.

A 192-GiB interpretation of the card still falls short of a nominal 250-GB core by 40.83 GiB, before workspaces. Thus even this card cannot hold the entire BF16 model; a routed expert layer is only 5.033 GB and fits easily. Whole-corpus paired BF16 hidden streams for 327,680 input positions occupy about 13.4 GB at the four-way hyper-connection width, plus transient workspaces. Do not load all expert layers into VRAM.

**ROCm 10.1 is available and is the target, not an aspirational fallback.** Official AMD indices contain `torch==2.14.0+rocm10.1.0` and the matching `amd-torch-device-gfx942` wheels for CPython 3.10–3.14. Use Python 3.12 on the rental. The official installation recipe is:

```bash
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  'rocm[libraries,device-gfx942]==10.1.0'
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  'torch[device-gfx942]==2.14.0+rocm10.1.0'
```

The checked-in `rent_setup.py` implements this in a private venv, selecting/downloading the exact wheels and recording hashes. Do not install the bare split torch wheel without its gfx942 device package. `torchvision` and `torchaudio` are not needed.

Prefer a provider ROCm-10.1 image. AMD also documents the image `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0`; if using a container, expose `/dev/kfd` and `/dev/dri`, add the render/video groups, use adequate shared memory, and bind the rental's large local storage at the same run root. Pip installs userspace, not a working host kernel driver. Follow AMD's supported host/driver matrix rather than guessing kernel compatibility.

**Fallback policy:** only if no ABI-compatible ROCm-10.1 torch/core/gfx942 wheel exists at execution time may setup choose the explicitly stated **ROCm 7.14 / torch 2.14.1+rocm7.14** fallback. It checks the AMD index, not merely the absence of a `rocm10.1` directory on download.pytorch.org. Network errors, 403/429/5xx, TLS failures and timeouts are not evidence of absent wheels and fail closed. Use a matching fallback userspace image if this actually occurs; do not silently bypass a host-version mismatch. No fallback is expected for Python 3.12 given the indices inspected for this handoff.

Primary sources:

- [AMD ROCm 10.1 installation and gfx942 SDK extras](https://rocm.docs.amd.com/en/latest/install/rocm.html)
- [AMD PyTorch installation, gfx942 extras and ROCm-10.1 Docker images](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- [AMD ROCm-10.1 torch wheel index](https://stable.repo.amd.com/rocm/pytorch/whl-next/torch/)
- [AMD gfx942 torch device-wheel index](https://stable.repo.amd.com/rocm/pytorch/whl-next/amd-torch-device-gfx942/)
- [Explicit ROCm-7.14 fallback index](https://download.pytorch.org/whl/rocm7.14/torch/)
- [AMD host/driver compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)

## 2. Transfer code and small inputs, not local model weights

Set the rental run root to its large disk. Commands below are executed **on the rental**, not hiptrx:

```bash
export ROOT=/srv/fn-kl
export CODE="$ROOT/hipfire"
export ORACLE="$CODE/crates/hipfire-arch-qwen4/reference_oracle"
export REV=de4b8e4d43b917e7706784d8bb445c9af86a3540
export SRC="$ROOT/source/$REV"
export PY="$ROOT/venv/bin/python"
mkdir -p "$ROOT/incoming" "$ROOT/corpus" "$ROOT/outputs"
```

Transfer `fn-kl-method.bundle`, this runbook, `KL-METHOD.md`, and the existing HumanEval file into `$ROOT/incoming`. The HumanEval source is `/home/kaden/.hipfire/datasets/HumanEval.jsonl`; its SHA256 is:

`1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2`.

HumanEval is a **code-only timing/floor proxy**, not the representative text/assistant quality corpus. Its contents are not executed. Do not transfer the old MQ4 capture as source weights or teacher hidden states.

Install the committed code on the rental:

```bash
git clone --branch beta https://github.com/warpfront/hipfire.git "$CODE"
git -C "$CODE" fetch "$ROOT/incoming/fn-kl-method.bundle" work/fn-kl-method
git -C "$CODE" checkout --detach FETCH_HEAD
git -C "$CODE" rev-parse HEAD | tee "$ROOT/code-commit.txt"
```

Verify that the printed commit equals the handoff commit. Repository access credentials, if needed, stay on the rental; do not put them in reports. The bundle is generated from the local branch without pushing it.

## 3. Fresh-box setup script and source integrity

Run the checked-in setup script using the host Python. It creates its own venv, installs the selected ROCm torch, records runtime versions, then downloads the pinned source checkpoint **to this rental disk**:

```bash
python3 -I "$ORACLE/rent_setup.py" \
  --root "$ROOT" --python python3.12 \
  --rocm-version 10.1 --torch-version 2.14.0 \
  --fallback-rocm-version 7.14 --fallback-torch-version 2.14.1 \
  --corpus-gb 2 --output-gb 100 --jobs 8
```

An `HF_TOKEN` may be supplied in the environment if the model requires it; do not record its value. Do not pass `--allow-rocm-mismatch` in the standard run. Setup failure is not permission to switch to a CPU/CUDA wheel.

Pinned model: `Qwen/Qwen3.8-Flash-Next` at `$REV`.

- `config.json` SHA256: `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`.
- `model.safetensors.index.json` SHA256: `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`.
- The index maps tensors to 131 shards; it **does not supply the shard SHA256 values**. Setup gets each shard's LFS identity from the revision-pinned HF metadata and verifies the downloaded **full-file SHA256 against that LFS SHA**. Non-LFS small files are verified against their git-blob SHA1 as well as recording SHA256. Do not claim a hash came from the safetensors index when it did not.
- Selected upstream operator sources are checked against the existing hard-coded source hashes and pinned commits (Transformers `93c8b7b485963a10800c91f55304db6be211c2bd`, version 5.16.1; source-only vLLM adapters `e126687a9a828d513c01a07cd69f025f27d63280`). This does not install or execute a competitor plugin.
- AMD wheel SHA256 values are observed/recorded; the AMD index inspected here has no published hash fragments. Setup verifies a fragment where an index publishes one. Distinguish TLS/index/version identity from a separately published wheel hash.

Setup outputs:

- `$ROOT/setup-manifest.json`: must have `status: complete`; selected wheel/version, fallback reason if any, installed distributions, raw `torch.version.hip`, `torch.version.cuda`, GPU properties, host ROCm/hipcc/driver diagnostics, disk allocation and per-transfer bytes/seconds/MB/s.
- `$ROOT/source/source-manifest.json`: source directory, pinned identities, per-file sizes and verified hashes.
- `$ROOT/logs/pip-install-{core,torch}.log` and `pip-install-torch-report.json`.
- `$ROOT/wheels/`: selected wheels and recorded hashes; these stay on the rental.
- `$SRC`: the complete pinned source and tokenizer. The revision in the directory name also satisfies the corpus freezer's strict revision check.

The installed torch must expose the ROCm backend (`torch.version.hip` is non-null; `torch.version.cuda` is null), the selected ROCm build tag and gfx942 devices. Record the **raw HIP component version separately** from the ROCm distribution version; do not assume their version strings must always be identical. Setup runs a small BF16 GPU matmul on the rental as a device sanity check. No such GPU check was run on hiptrx.

The teacher's local source reader additionally validates tensor shapes, dtypes, safetensors extents/header hashes, index consistency and per-read range hashes. Local full-shard verification is done by setup; its presence is not replaced by an ETag-only claim. Preserve the setup source manifest with the teacher artifacts.

## 4. Freeze the small pilot corpus

The preparation utility is local-only: it loads the tokenizer from `$SRC`, never from an unpinned online tokenizer. Freeze four units per stratum, capped at 128 text/256 assistant input tokens, using the fixed HumanEval prompt/canonical-solution proxy:

```bash
"$PY" -I "$ORACLE/quality_corpus.py" \
  --tokenizer-dir "$SRC" --strict-revision \
  --human-eval-input "$ROOT/incoming/HumanEval.jsonl" \
  --expect-human-eval-sha256 1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2 \
  --max-input-tokens 128 --assistant-max-input-tokens 256 \
  --units-per-stratum 4 --min-scored-positions 16 \
  --seed 3800 --name qwen4-kl-pilot-humaneval-proxy \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --out "$ROOT/corpus/pilot.json"
```

The script reports the actual input/scored counts; the caps imply at most 1,536 inputs, not an invented exactly-1,536 count. Text plus assistant score positions are separate. Assistant masks refer to actual rendered assistant **content**, exclude role delimiters/scaffolds, and account for the chat template trimming whitespace. Every score position has the next token as target; final rows without targets are excluded. Independent sequences reset attention/GDN/PLE histories. Corpus metadata records source/tokenizer/template hashes and labels every HumanEval unit as a proxy.

For a representative full corpus, `quality_corpus.py --input` accepts frozen JSONL text records or conversations with fixed reference assistant answers; it never generates new answers. `KL-METHOD.md` specifies the full corpus and deterministic selection/masking policy. Do not silently promote this code-only pilot to that full corpus.

## 5. One layer-major pass: teacher, all-position tape and floor

On ROCm the torch device name is `cuda`, not `hip`. Do not carry hiptrx's GPU-hiding environment onto the rental:

```bash
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
nice -n 10 "$PY" -I "$ORACLE/quality.py" \
  --exact-floor --corpus "$ROOT/corpus/pilot.json" \
  --out-dir "$ROOT/outputs/pilot" --local-dir "$SRC" \
  --device cuda --threads 16 --memory-limit-gib 0 \
  --pass-tokens 0 --expert-cache-gib 8 --head-rows 4096 \
  --logits-storage bf16 --export-hidden --bootstrap 2000 --seed 3800
```

`--exact-floor` (`--paired-floor` alias) advances native and reverse-expert-order hidden states in lockstep through each layer, sharing that layer's immutable source weights. It does **not** stage all expert layers on the GPU. `--pass-tokens 0` keeps the corpus in one layer-major weight pass; split passes are available if measured activation/workspace usage requires it, at the cost of repeated local weight reads. `--expert-cache-gib 8` accommodates the 4.6875-GiB expert layer plus other layer weights. It is cleared at layer boundaries.

`--memory-limit-gib 0` is deliberate on the rental: a 64-GiB address-space limit can reject GPU virtual-address mappings even with small live host RSS. The exporter records host RSS and GPU allocated/reserved peaks. The CPU mode defaults to its 64-GiB limit and ≤16 threads; the cancelled local CPU pilot is not resumed by this command.

The floor changes only the actual pinned experts-forward accumulation loop (`expert_hit` versus `expert_hit.flip(0)`); expert products, source weights, routing computation and BF16 cast boundaries are preserved. Its transformed AST/diff identity is recorded separately. It is a bf16 reduction-order floor, not an independent-implementation floor and not a synthetic perturbation. The native→native self comparison remains an estimator sanity check, not the floor.

BF16 logit storage is **lossless here**, not a quantization choice: export refuses it unless the LM-head output really is BF16 and its F32 widening round-trips bit-exactly. Arbitrary F32 candidate logits must remain F32. The scorer reconstructs all vocabulary entries and uses F64 stable log-normalizers/KL; there is no top-32/256 surrogate in the primary metric.

Expected pilot files:

```text
outputs/pilot/native/export.json
outputs/pilot/native/score_index.u32le
outputs/pilot/native/logits.bf16le
outputs/pilot/native/route_ids.u16le
outputs/pilot/native/route_scores.f32le
outputs/pilot/native/final_hidden.bfloat16le
outputs/pilot/reverse-experts/(same files)
outputs/pilot/floor_native_vs_reverse.json
outputs/pilot/floor_native_vs_reverse.positions.npz
outputs/pilot/floor_self_native.json
outputs/pilot/floor_self_native.positions.npz
```

Read shapes/ordering/identities from `export.json`; do not guess binary interleaving. Route files cover **all input positions, all 48 MoE layers and ordered top-10 experts**, including unscored context, rather than just assistant/scored rows. UInt16 IDs are exact for 512 experts. Scores are stored as F32. A future candidate replay must implement the frozen primary **IDs-only** contract (candidate own logits gathered/normalized on teacher IDs); replaying teacher scores too is a separately named IDs+scores diagnostic. This implementation captures the source tape; it does not claim a production-engine replay hook already exists.

Acceptance before Main's full-run go:

1. Both exports are complete 48-layer source-bf16 forwards, with identical corpus/source identities, expected scored counts, full vocabulary and finite logits. Partial-layer timing probes must never be called full teacher references.
2. All-position tape counts/IDs/scores match the declared geometry and sequence boundaries.
3. Native→native KL is within the documented zero tolerance; native→reverse yields a real, reproducible floor with per-position values and document-bootstrap CIs.
4. Report the numerical stability flag (CI half-width ≤0.002 nats, relative criterion where applicable, plus split-half checks). A pass on four code-only units per stratum is **not** representative-corpus acceptance.
5. Report actual wall time, input/scored tok/s, source I/O versus compute time, local/remote bytes, host RSS, GPU peaks and `projection_327680`. With a complete `$SRC`, the teacher should not transfer remote weight bytes. Keep setup's cold network staging separate from this traversal.
6. Source/setup/export manifests must state exact ROCm and torch versions and whether any fallback occurred.

Recomputing the floor from saved exports requires no model/GPU:

```bash
"$PY" -I "$ORACLE/quality.py" --exact-compare \
  --reference "$ROOT/outputs/pilot/native" \
  --candidate "$ROOT/outputs/pilot/reverse-experts" \
  --out "$ROOT/outputs/pilot/floor-recomputed.json" --bootstrap 2000 --seed 3800
```

To compare a later candidate, supply its complete exact-logit export with the same corpus/index identity; do not feed the legacy top-k quality artifact to the exact scorer.

## 6. Full-corpus command — only after Main approves

Main supplies/freezes `$ROOT/corpus/quality-65536.json`: 32,768 text and 32,768 assistant target positions across independent units, with the method's context-length strata, target/role masks and input identities. The original planned mixed corpus forwards 327,680 input tokens. Verify its actual counts before projecting runtime.

After the pilot and Main's go:

```bash
nice -n 10 "$PY" -I "$ORACLE/quality.py" \
  --exact-floor --corpus "$ROOT/corpus/quality-65536.json" \
  --out-dir "$ROOT/outputs/full" --local-dir "$SRC" \
  --device cuda --threads 16 --memory-limit-gib 0 \
  --pass-tokens 0 --expert-cache-gib 8 --head-rows 4096 \
  --logits-storage bf16 --export-hidden --bootstrap 2000 --seed 3800
```

If the stability rule requires more independent units, append the predeclared deterministic doubling round and record the final sample size. Never drop high-KL rows, alter score masks after inspecting candidates, subtract the floor, or claim Paiton's numerical thresholds were reproduced on an undisclosed/different corpus.

## 7. Hash and ship the result bundle

For 65,536 scores and vocab 248,320, a BF16 full-vocab teacher is **32.548 GB**, versus 65.096 GB F32. All-context routes for 327,680 inputs are **300 MiB IDs + 600 MiB scores per implementation**, before small metadata. Two BF16 floor exports total roughly 65.1 GB on the rental. Preserve both there; the initial return bundle can ship native teacher logits plus both tapes and floor/per-position evidence, not the large reverse logits or the 352-GB source checkpoint.

Ship back:

- setup/source manifests and package/install reports (no credentials);
- the exact code commit, this runbook and the frozen corpus with its hashes;
- native `export.json`, score index, **full-vocabulary native BF16 logits**, native tape and final hidden;
- reverse `export.json`, tape and final hidden; its export retains the hash of the reverse logits kept on the rental;
- ordinary floor and self-sanity JSON plus per-position NPZ files and any recomputed report;
- command logs, runtime/throughput/RSS/GPU evidence and a SHA256+byte-count manifest for every returned file.

The full native reference plus tapes is roughly 35 GB, rather than returning a full checkpoint. Do not remove the rental's reverse logit file until Main has verified receipt and hashes and approved deletion. If full F32 storage was required instead, report the resulting 65.1-GB reference size before transfer.

The following rental-side packaging script avoids guessing which optional files exist and hashes the archive too:

```bash
export RUN="$ROOT/outputs/pilot"  # change to outputs/full for the approved full run
"$PY" -I - <<'PY'
import hashlib, json, os, tarfile
from pathlib import Path
root = Path(os.environ['ROOT'])
run = Path(os.environ['RUN'])
files = [root/'code-commit.txt', root/'setup-manifest.json', root/'source/source-manifest.json',
         root/'incoming/RENT-RUNBOOK.md', root/'incoming/KL-METHOD.md']
files += sorted((root/'logs').glob('*'))
files += sorted((root/'corpus').glob('*.json'))
files += sorted(p for p in run.rglob('*') if p.is_file()
                and not (p.parent.name == 'reverse-experts' and p.name.startswith('logits.')))
files = sorted(set(p for p in files if p.is_file()))
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
records = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size,
            'sha256': digest(p)} for p in files]
ret = root/'return'; ret.mkdir(exist_ok=True)
manifest = ret/'SHA256-MANIFEST.json'
manifest.write_text(json.dumps({'schema':'hipfire.qwen4.return.v1',
                               'files':records}, indent=2, sort_keys=True)+'\n')
with manifest.open('rb') as f: os.fsync(f.fileno())
archive = ret/'fn-kl-return.tar'
with tarfile.open(archive, 'w') as tar:
    tar.add(manifest, arcname='SHA256-MANIFEST.json')
    for p in files: tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
with archive.open('rb') as f: os.fsync(f.fileno())
seal = ret/'ARCHIVE-SHA256.json'
seal.write_text(json.dumps({'file':archive.name,'bytes':archive.stat().st_size,
                           'sha256':digest(archive)}, indent=2)+'\n')
with seal.open('rb') as f: os.fsync(f.fileno())
print(seal.read_text())
PY
```

Transfer `return/fn-kl-return.tar` and `ARCHIVE-SHA256.json`. Verify the archive hash **before** extraction and each file hash afterward. Keep code/source identities from the inner exports; the shipping manifest does not replace them. Returning outputs is separate from permission to retain model weights on hiptrx; **do not ship the source snapshot back**.

## 8. Expected hours and projection caveats

**[INFERENCE — not a rental benchmark]** Reserve approximately **1–8 hours total** for setup, source staging, the small pilot and a subsequent paired full run on one MI300X; reserve more if the provider's disk/network is slow or the unoptimized source kernels dominate. This is not an authorization to run the full corpus.

- A ~352-GB source fetch at assumed 50–500 MB/s takes **1.96–0.20 hours**, plus dependency setup and source hashing (roughly 0.25–0.75 hours combined). Setup reports actual bytes and throughput; the exact snapshot size comes from HF metadata.
- A staged 1–2K-input pilot should be budgeted **5–30 minutes**, dominated by loading most of the same expert weights and many small source-operator launches. No observed value exists yet.
- The two 327,680-token BF16 variants are approximately **7.9–13.1 PFLOP** under the method's 12–20 GFLOP/input planning envelope. At an assumed effective 5–30 TFLOP/s this is about **44–4 minutes of arithmetic alone**, not an end-to-end claim. Budget **0.5–4 hours** for paired traversal, operator/launch/transfer overhead, exact KL and evidence writes. The large-card peak FLOP spec is not the effective rate of small grouped experts.
- With `--pass-tokens 0`, a whole-corpus layer-major pass reads the core weights roughly once rather than the old 40×8K CPU-pass plan's ~10 TB. If GPU activation limits force 40 split passes, ~10 TB of local core reads at 0.5–3 GB/s adds a **5.56–0.93 hour I/O bound**. A large host page cache can reduce cold disk reads but does not remove GPU transfer/compute.
- Do not linearly scale a pilot wall time dominated by a nearly fixed 242-GB weight load. Project measured compute by input count/context shape, then add measured weight I/O for the actual number of passes, and keep initial source staging separate. Report the script's projection and this caveat together.

A one-time roughly **250-GB local core copy** (roughly 352 GB including full BF16 PLE/source extras) on the rental avoids repeated network transfers. Renting large host RAM/local disk is the approved approach; no equivalent copy is staged on hiptrx. The final rental pilot report must replace these estimates with observed throughput, memory, floor and stability results before deciding the full run.
