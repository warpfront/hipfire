# Ordinary MQ4 packed/wide prefill experiment

Lifecycle: **historical**. Disposition: opt-in experimental candidate; not a
default, admission, cross-model comparison, or quality-neutrality claim.

## Fixture and implementation

- Base: beta `8d302c342`; packed restoration `38d68e211` plus the accompanying
  native-MQ4 wide admission, memory budgeting and packed-tail changes.
- W7900/gfx1100, GPU1 PCI0000:9e:00.0, HIP7.15.26333.
- Qwen3.6-27B MQ4 SHA256
  `86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc`.
- Final daemon SHA256
  `1006155d92a091002e7eb7ad53792b5dda2d54135ccccab43ffc1671493709ed`.
- Prompt SHA256
  `a2312bc708f9c63a73ab6dbb893cf34fce475413c77f8b0b377de5a84a0e381d`,
  MD5 `15cefcf1cbf756accb2d5fcfdadf7f84`.
- Exact8192 input tokens, max_seq16384, Q8/VMM KV, Q8/EF recurrent state,
  speculation off, greedy, verify graph off, 1024 output tokens.

Native MQ4 retains its projection path; widening does not grant it the V2
producer or lean-PBS contract. Packed FFN composes with wide ordinary prefill
using full PBS and additional scratch admission. Both flags default OFF:
`HIPFIRE_GFX1100_PACKED_MQ4_PREFILL` and `HIPFIRE_GFX1100_MQ4_WIDE_PREFILL`.
`HIPFIRE_PREFILL_CHUNK_ROWS=4096` requests the measured candidate ceiling.
Captured/verify/TP/MoE paths are not widened. Recurrent commits stay512 rows.
Complete legacy512 tiles can merge; irregular tails retain their old grouping
and quantization route (4097 becomes3584+511+2, not4095+2).

## Direct ordinary-MQ4 result

Fresh-process ABBA, three full warmups per process, one measured request.
Each arm is the same binary/model/prompt. A=flagsOFF/chunk512;
B=bothflagsON/chunk4096. Raw hashes, executed routes and lengths checked.

| Arm | Prefill tok/s | Decode tok/s | Generated |
|---|---:|---:|---:|
| A | 784.7 | 36.7 | 1024 |
| B | 1051.7 | 36.7 | 1024 |
| B | 1050.9 | 36.6 | 1024 |
| A | 784.9 | 36.7 | 1024 |

Mean prefill784.8 ->1051.3 (+33.96%); decode36.70 ->36.65 (-0.14%).
This is two measurements per arm on one fixture, not a broad performance
admission. Sampled device peak rises17384370176 ->19726008320 bytes
(~2.18GiB); it is not an exact allocator delta.

Earlier incremental observations (different sessions, not additive gains):
packed512 vs native512 +21.88% PP; native-wide1024/2048/4096/8192 approximately
+7.46/+8.38/+10.31/+10.27% PP. Decode checked at every rung. The final direct
combined ABBA above is the authoritative result for this candidate binary.

## Correctness and compatibility coverage

- Prefill unit module:63 passed,6 ignored. Final build and whitespace check pass.
- Native512/native8192 at8193 rows:646 numeric records equal, all454
  raw-byte-comparable records equal, zero nonfinite; includes four
  teacher-forced continuation logits and KV/recurrent state.
- Packed512/packed4096 at4097 and8193 rows:646 records each equal; the
  diagnosed irregular-tail route mismatch is fixed. These are same-format
  scheduling oracles, NOT native-versus-packed numerical equivalence.
- Local historical MQ4V2/XT artifact compatibility: executed both packed
  FFN routes at4096, completed PP8192/TG256. Not canonical XT perf evidence.
- Five serving genres, contexts6014–6037,1024 outputs each: no empty,
  attractor or stream-error events. Three related turns: cached0/4096/7418;
  generated1024/333/607; clean terminal handling. Text read in full.
- Stock cold prefill-capture harness rejects both clean beta and candidate
  because FA2 Q16 scratch needs pre-growth. Preserved as a baseline limitation,
  not waived as a pass. Candidate stock decode-only harness passes.
- Explicitly warmed full harness passes on BOTH: prefill1332 launches/20
  kernels, decode659/16, stable sequences and AQL contracts; four-position
  HIP/PM4/blob and GDN-frame parity. Adapter logs every added warmup request.
  Capture uses the unchanged fallback; ordinary packed execution has separate
  daemon/state/serve coverage above.

Packed activation quantization changes greedy output versus native. Within
each arm output repeats exactly. Coherent output is not a quality-neutrality
proof: arithmetic/complexity errors occur in technical explanations; generated
code is truncated at1024 tokens. No broad accuracy or runnable-program claim.

## Reproduction and evidence

`benchmarks/results/mq4-step0-3-evidence.tar.gz` (PR #794 head `ee696bec`; not carried into the tree)
SHA256 `7c1fe65a945607700beec335d16cbcf35e32ca04f11bc8415bace73239d047b9`
contains immutable final raw daemon/serving/capture logs, manifests, per-tensor
comparison reports, prompts, campaign JSON, and orchestration/audit scripts.
Large raw state dumps remain local; regenerate them with the checked-in
`mq4_prefill_state_oracle` example and archived runner. Archive scripts contain
original local paths: substitute checkout/binary/output paths and GPU binding
for a new machine; do not reuse original output directories.

Build daemon and oracle with the same features:

```sh
cargo build --release --locked -p hipfire-daemon -p hipfire-arch-qwen35 \
  --features hipfire-daemon/flash-attn-ck,hipfire-arch-qwen35/flash-attn-ck \
  --bin daemon --example mq4_prefill_state_oracle
```

After extracting into a fresh directory, adjust only the campaign's prompt
path to its archived `results/mq4-tail-final-abba/prompt.txt`. Run under the
repository GPU lock:

```sh
python3 pp8192_native_matrix.py CHECKOUT PINNED_DAEMON NEW_OUTPUT GPU \
  final-mq4-tg1024.json
python3 summarize_mq4_abba.py NEW_OUTPUT
```

Do not compare HTTP cold/JIT timings with warmed daemon ABBA, mix different
model generations, or infer packed quantization is lossless from a within-route
state comparison. No registry/default changes or production promotion here.
