# Offload pass-back: a split spilled step across both engines — gfx1201 — 2026-10-01

**Lifecycle:** `historical`

**Amended in place 2026-10-01, later the same day**, on the record owner's
instruction. `docs/perf-checkpoints/README.md` routes corrections to a new dated
amendment file; that rule is deliberately overridden here, and every changed
passage is marked `[amended]` so a reader can still see what moved. Sections added:
§ 8 (offload-amount sweep + a 27B comparison point), § 9 (CLI paths), § 10 (the
32k-token long-range arc). § 4 and the identity table were corrected — see the
`[amended]` notes.

**Disposition:** measured evidence for `memory.offload_exec=passback` (the row-split
pass-back mode; design in
[`../plans/partial-gpu-offload-design.md`](../plans/partial-gpu-offload-design.md) § 6.2.2).
It is **not** a product baseline, not an admission, and not a
[`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim. One host, one prompt, one model, one
budget (24 of 32 layers resident, 8 spilled), greedy, `--spec off`. Nothing here
transfers to another host, link width, core count, model, or decode length without a
new record.

## What this establishes

1. **The mode is correct, and its two arms are the existing paths exactly.** A split
   step's GPU rows are bit-identical to a full `pcie` launch's rows and its CPU rows
   bit-identical to a whole-weight CPU GEMV's, over 15 cases (real 2B/9B
   MQ4/MQ3-Lloyd/MQ6/HFQ6 tensors, their odd-row-count prefixes, and synthetic
   MQ4G256V2/MQ4CG256/MQ3G256V2/HFQ4G256/Q8_0), for both the plain and the residual
   step form — including that no row at or past `m` is ever written. **12 of the 15
   cases mix numerically different engines** (max element delta 1.2e-7 – 1.5e-5,
   relative 1.1e-7 – 2.3e-7); in 3 (Mq4CG256, Hfq4G256, Q8F16) the two engines agree
   bitwise.
2. **`memory.offload_passback_share = 0` is byte-identical to `memory.offload_exec=cpu`**
   (555-byte 256-token and 642-byte 512-token completions compared with `cmp`), and prints no `split:`
   line at all: the disable path is exactly the whole-CPU step.
3. **A stray share in a non-`passback` mode changes nothing.** `pcie` with
   `HIPFIRE_OFFLOAD_PASSBACK_SHARE=0.4` produced byte-identical output to bare `pcie`
   and one informational line at load.
4. **Pass-back pays: +11.7 % decode over `cpu`, and `pcie` remains the loser at this
   spill size.** Three interleaved fresh-process rounds, budget 24 on the 9B:
   `cpu` 27.40 tok/s, `passback` 30.60 (+11.7 %),
   `pcie` 23.20 (-15.3 %). Per-arm spread was
   tiny (`cpu` [27.4, 27.5, 27.2], `passback` [30.8, 30.6, 30.2], `pcie` [23.2, 23.2, 23.2]); `vram_free_mb`
   `[amended]` spread 11308–11336 MB across **all nine** arms (per-arm values in
   § 1) — identical within the diagnostic's own granularity, as expected: a
   pass-back changes who multiplies, never what is resident. The pass-back's own
   values (11336/11336/11308) sit *inside* the cpu and pcie ranges, so nothing here
   is attributable to the split. The earlier wording of this bullet said
   "11308–11308 MB", which was wrong; the data were always the per-arm lists in § 1.
5. **The scheduler converges on its own to the share the standalone bench predicts.**
   With `share auto` the four covered shapes settled at **0.279–0.360** on the field
   shapes (2–27 MB) and froze there; `hipfire offload-bench` measures the same host's
   balance point independently at **0.28–0.45** by format and 0.36–0.40 for the same
   `k=4096` shapes and sizes. No host constant is load-bearing: the seeding probe times
   both engines on the step's own weight buffer and the online controller corrects it.
6. **The GPU arm of a split reads host-mapped weight bytes at 22.8–28.2 GB/s**, against
   34.7–68.4 GB/s for the CPU arm on the same bytes (§ 2's table, `--reps 7`, the
   corrected blocking sync) — i.e. the two-engine bound the headroom record predicted
   is real on this host, whose link measures 27.3 GB/s bulk. `[amended]` An earlier
   version of this bullet quoted 19.6–28.5 and 29–73 GB/s: the 19.6 and the 29 came
   from the pre-fix probe table § 3 keeps as a discarded comparison, so quoting them
   here contradicted § 2.
7. **Forensic: `Gpu::sync_with_deadline` is unusable for timing.** It polls completion
   with `std::thread::sleep(SYNC_POLL_INTERVAL)`, `SYNC_POLL_INTERVAL = 2 ms`
   (`crates/rdna-compute/src/dispatch.rs:1264`), so a probe that times
   `launch → sync_with_deadline` measures the *sleep*, not the kernel: it reported
   4.1 GB/s for an 8 MiB weight whose true rate is 24.4 GB/s, and the error looks exactly
   like a size-dependent fixed cost (~1.7 ms/launch). `offload_split`'s probe now uses
   `gpu.hip.device_synchronize()`; deadline-bearing paths keep `sync_with_deadline`.
   The first probe table below is the artifact, kept because the wrong number is the
   kind of thing that gets quoted.

## Fixture identity (measured)

- Host: 1 × Radeon RX 9070 XT `gfx1201`, 16304 MB, **PCIe 16.0 GT/s × 16** (4.0 x16),
  ROCm/HIP 7.2; Ryzen 7 7800X3D (8c/16t), 28 GB RAM. 1-min load average 4.6–5.9 during
  the arms. `[amended]` The load was **not** unrelated work and the host was never
  near the plan's `uptime ≈ 0.5`: the top consumer is `orcaslicer_main` at ~55 % of a
  core, plus konsole/kwin/easyeffects, so `nproc`-normalised load stays around 3–8
  through every arm. The interleaved round design is what makes the deltas readable;
  the **absolute rates are depressed relative to an idle host** and the balance point
  itself shifts with contention (a slower CPU arm favours a higher GPU share), so
  nothing here should be quoted as this machine's standalone engine rates. § 2's
  `offload-bench` table was taken in the same regime.
- Source HEAD `f9079882c1d66740e22cea5da1b9418b6f24531a`;
  `target/release/daemon` md5 `ec8f192755cc349b8e46b2dd578429eb`;
  `target/release/hipfire` md5 `c291d26c39f82013d492d94512b3f0c8`.
- `~/.hipfire/models/qwen3.5-9b.mq4` — 5,313,750,016 B, sha256
  `ba83acf5bfd5d4e334b0afc26d779734e31623bb7f74e807c3581dfecb3128ad` (the registry
  artifact `qwen3.5:9b`; the same file the headroom record used, and **not** the
  local-only 9B the 2026-09-27 CPU-exec records measured).
  Every projection in this model is qt 13 `Mq4G256`.
- Prompt `benchmarks/prompts/gpu_offload_probe.txt`, md5
  `5835c71e471849b4a72e1dc8e39695e7` (59 tokens, 215 chars).
- Budget: `HIPFIRE_GPU_LAYER_BUDGET=24` on a 32-layer model → **8 of 32 layers spilled**.

## Method

`hipfire bench qwen3.5:9b --spec off --runs 1 --warmups 1 --max-tokens 128 --backend
noslots --workload stateless --prompt-file benchmarks/prompts/gpu_offload_probe.txt
--json`, one fresh daemon per arm (`pkill -f target/release/daemon` first), arms
interleaved `cpu`, `passback`, `pcie` × 3 rounds, `HIPFIRE_OFFLOAD_EXEC` selecting the
mode. `passback` ran with `share auto` (the default) — no pinning anywhere in the
perf arms, so the number below is what a user gets without tuning.

Correctness arms use `hipfire run … -t 0 --spec off -n 256` (and 512) with the same
budget, and `cmp` on the stdout bytes. The row-offset parity arms are
`HIPFIRE_OFFLOAD_EXEC=passback cargo test -p hipfire-arch-qwen35 --release --test
gpu_gemv_parity -- --ignored --nocapture passback_row_offset`.

## 1. Mode A/B/C (3 interleaved rounds, budget 24, 8 of 32 spilled)

| round | `cpu` | `passback` (share auto) | `pcie` |
|---|---:|---:|---:|
| 1 | 27.4 | 30.8 | 23.2 |
| 2 | 27.5 | 30.6 | 23.2 |
| 3 | 27.2 | 30.2 | 23.2 |
| **median** | **27.40** | **30.60** | **23.20** |
| vs `cpu` | — | **+11.7 %** | -15.3 % |

Decode tok/s (the `decode_tok_s.median` of a single 128-token run per process). Whole
`--json` per arm in the appendix. Wall tok/s tracked decode
(`cpu` [25.8, 25.8, 25.6], `passback` [28.8, 28.5, 28.2], `pcie` [22.0, 22.0, 22.0]); prefill is a
59-token prompt and measures launch overhead, so it is not quoted as a prefill number
(`cpu` [195.5, 193.7, 197.9], `passback` [201.0, 198.6, 196.7], `pcie` [194.6, 195.5, 198.9]).
`vram_free_mb` (post-load, per arm): `cpu` [11308, 11336, 11308], `passback` [11336, 11336, 11308],
`pcie` [11336, 11336, 11336] — identical within the diagnostic's own granularity, as
expected: the pass-back changes who multiplies, never what is resident.

The pass-back's own accounting on the field shapes (round 3 of the bench equals the
run below; identical shares): 10,752 split steps out of 15,871, `0 host-mapped steps
still on GPU`, and no row ever charged to the CPU-idle numerator.

## 2. `hipfire offload-bench` (independent host measurement)

`./target/release/hipfire offload-bench --reps 7`, default matrix, `k=5120`,
192 MiB/format. `predicted_speedup` is the two-engine bound
`(r_gpu + r_cpu) / max(r_gpu, r_cpu)`.

```
arch gfx1201 — GPU share for a split offload step (memory.offload_exec=passback)
format              m      k      MiB     cpu GB/s     gpu GB/s   share   speedup
mq4g256         74018   5120    192.0         44.3         25.9   0.368     1.58x
mq4g256v2       74018   5120    192.0         46.8         23.5   0.334     1.50x
mq4cg256        74018   5120    192.0         41.8         22.8   0.353     1.55x
mq6g256         50332   5120    192.0         38.4         27.8   0.420     1.72x
mq6g256v2       50332   5120    192.0         42.4         28.2   0.399     1.66x
mq5g256v2       59919   5120    192.0         35.9         27.0   0.429     1.75x
mq3g256v2       96792   5120    192.0         34.7         23.0   0.399     1.66x
hfq4g256        74018   5120    192.0         48.7         24.4   0.334     1.50x
q8_0            37009   5120    192.0         68.4         25.9   0.275     1.38x

recommended for the 5 largest measured formats: memory.offload_passback_share = 0.334
```

## 3. The seeding probe, before and after the sync fix (same shape, `k=4096`)

`offload-bench --format mq4g256 --k 4096 --buffer-mb <n> --reps 7`:

| weight | with `sync_with_deadline` (the artifact) | with `device_synchronize` (shipped) |
|---|---:|---:|
| 8 MiB | 4.1 GB/s (share 0.093) | 24.4 GB/s (share 0.451) |
| 16 MiB | 8.1 (0.195) | 24.6 (0.395) |
| 24 MiB | 12.2 (0.221) | 25.8 (0.373) |
| 48 MiB | 12.2 (0.202) | 26.1 (0.360) |
| 96 MiB | 16.3 (0.253) | 26.5 (0.367) |
| 192 MiB | 24.4 (0.361) | 26.6 (0.340) |

The GPU rate is flat in the weight size once the sleep is out of the measurement; the
left column's "fixed cost" was the poll interval. CPU rates are unchanged between the
two (29–52 GB/s).

## 4. Share trajectory (in-process scheduler, `share auto`, budget 24)

Shape `m=12288 k=4096` (`gate_up_proj`), the four covered shapes all behave alike:

| calls | gpu_rows/12288 | share | controller |
|---:|---:|---:|---|
| 1 | 5480/12288 | 0.393 | gpu samples 1, waited false, target 0.393, 1 applied |
| 2 | 5156/12288 | 0.393 | gpu samples 1, waited false, target 0.393, 1 applied |
| 4 | 4858/12288 | 0.371 | gpu samples 2, waited false, target 0.371, 2 applied |
| 8 | 4461/12288 | 0.322 | gpu samples 5, waited true, target 0.322, 5 applied |
| 16 | 4272/12288 | 0.341 | gpu samples 13, waited false, target 0.341, 10 applied |
| 32 | 4306/12288 | 0.363 | gpu samples 31, waited false, target 0.363, 21 applied |
| 64 | 4407/12288 | 0.367 | gpu samples 68, waited false, target 0.367, 42 applied |
| 128 | 4457/12288 | 0.360 | gpu samples 169, waited false, target 0.360, 68 applied (frozen) |
| 256 | 4441/12288 | 0.360 | gpu samples 444, waited true, target 0.360, 68 applied (frozen) |
| 512 | 4432/12288 | 0.360 | gpu samples 1019, waited true, target 0.360, 68 applied (frozen) |
| 1024 | 4428/12288 | 0.360 | gpu samples 2243, waited true, target 0.360, 68 applied (frozen) |
| 2048 | 4426/12288 | 0.360 | gpu samples 4773, waited true, target 0.360, 68 applied (frozen) |
| 4096 | 4425/12288 | 0.360 | gpu samples 9931, waited true, target 0.360, 68 applied (frozen) |


Final `split:` lines of that run (mean per split step over the shape's calls):

```
split: step gemv m=8192 k=4096 quant=Mq4G256 gpu_rows=2945/8192 share=0.360 (gpu samples 9926, last waited=true, target=0.360, 68 applied, frozen) rotated=false residual=false awq=false | 2048 calls | 15866 steps (10747 split), 0 host-mapped steps still on GPU | mean per split step: d2h=0.03ms gemv=0.35ms join=0.05ms | cpu=30.5GB/s gpu≥15.3GB/s
split: step gemv m=1024 k=4096 quant=Mq4G256 gpu_rows=368/1024 share=0.360 (gpu samples 9928, last waited=true, target=0.360, 68 applied, frozen) rotated=false residual=false awq=false | 1024 calls | 15868 steps (10749 split), 0 host-mapped steps still on GPU | mean per split step: d2h=0.03ms gemv=0.09ms join=0.03ms | cpu=11.8GB/s gpu≥5.2GB/s
split: step gemv m=4096 k=4096 quant=Mq4G256 gpu_rows=1472/4096 share=0.360 (gpu samples 9929, last waited=true, target=0.360, 68 applied, frozen) rotated=true residual=true awq=false | 2048 calls | 15869 steps (10750 split), 0 host-mapped steps still on GPU | mean per split step: d2h=0.08ms gemv=0.24ms join=0.04ms | cpu=18.0GB/s gpu≥8.9GB/s
split: step gemv m=12288 k=4096 quant=Mq4G256 gpu_rows=4425/12288 share=0.360 (gpu samples 9931, last waited=true, target=0.360, 68 applied, frozen) rotated=false residual=false awq=false | 4096 calls | 15871 steps (10752 split), 0 host-mapped steps still on GPU | mean per split step: d2h=0.03ms gemv=0.46ms join=0.05ms | cpu=35.1GB/s gpu≥18.1GB/s
```

Read: `[amended]` the probe seeds ~0.36 (the `0.233` a pre-fix run started from is
quoted in the commit that fixed the probe's sync, not here), the controller pushes while
every join sits at the
shape's own no-wait floor, and it converges and freezes at 0.279–0.360 once joins
exceed that floor. `gpu≥GB/s` is a lower bound by construction (the CPU arm is the
straggler there), which is why the share rests on the sampled `target` and not on it.

## 5. Row-offset equivalence (the correctness arm)

`HIPFIRE_OFFLOAD_EXEC=passback cargo test -p hipfire-arch-qwen35 --release --test
gpu_gemv_parity -- --ignored --nocapture passback_row_offset` — every case's split
output is compared against **both** whole-step references, plus the arm-delta (how far
apart the two engines are on identical bytes):

```
GPU dev 0: gfx1201 (17.1 GB VRAM, HIP 7.2)
  pre-compiled kernels: /home/avery/.hipfire_kernels/gfx1201
qwen3.5-2b.mq4 up_proj                       m=6144   k=2048   row_bytes=1088   g=3072   (m-g=3072) exact; arm delta 1.490e-7 (rel 1.45e-7)
qwen3.5-2b.mq4 up_proj odd-m=1929            m=1929   k=2048   row_bytes=1088   g=960    (m-g=969) exact; arm delta 1.192e-7 (rel 1.35e-7)
qwen3.5-2b.mq3 up_proj                       m=6144   k=2048   row_bytes=896    g=3072   (m-g=3072) exact; arm delta 1.788e-7 (rel 1.88e-7)
qwen3.5-2b.mq3 up_proj odd-m=2341            m=2341   k=2048   row_bytes=896    g=1168   (m-g=1173) exact; arm delta 1.788e-7 (rel 2.03e-7)
qwen3.5-2b.mq6 up_proj                       m=6144   k=2048   row_bytes=1600   g=3072   (m-g=3072) exact; arm delta 2.384e-7 (rel 2.27e-7)
qwen3.5-2b.mq6 up_proj odd-m=1311            m=1311   k=2048   row_bytes=1600   g=648    (m-g=663) exact; arm delta 1.490e-7 (rel 1.64e-7)
qwen3.5-2b.hf6 up_proj                       m=6144   k=2048   row_bytes=1600   g=3072   (m-g=3072) exact; arm delta 1.192e-7 (rel 1.14e-7)
qwen3.5-2b.hf6 up_proj odd-m=1311            m=1311   k=2048   row_bytes=1600   g=648    (m-g=663) exact; arm delta 1.192e-7 (rel 1.30e-7)
qwen3.5-9b.mq4 up_proj                       m=12288  k=4096   row_bytes=2176   g=6144   (m-g=6144) exact; arm delta 3.576e-7 (rel 2.28e-7)
qwen3.5-9b.mq4 up_proj odd-m=965             m=965    k=4096   row_bytes=2176   g=480    (m-g=485) exact; arm delta 2.384e-7 (rel 2.02e-7)
synthetic qt=44 Mq4G256V2                    m=965    k=4096   row_bytes=2176   g=480    (m-g=485) exact; arm delta 1.526e-5 (rel 1.53e-7)
synthetic qt=45 Mq4CG256                     m=965    k=4096   row_bytes=2176   g=480    (m-g=485) exact; arm delta 0.000e0 (rel 0.00e0)
synthetic qt=49 Mq3G256V2                    m=1261   k=4096   row_bytes=1664   g=624    (m-g=637) exact; arm delta 7.629e-6 (rel 1.37e-7)
synthetic qt=6 Hfq4G256                      m=965    k=4096   row_bytes=2176   g=480    (m-g=485) exact; arm delta 0.000e0 (rel 0.00e0)
synthetic qt=3 Q8F16                         m=483    k=4096   row_bytes=4352   g=240    (m-g=243) exact; residual arm refused (no GPU residual kernel); arm delta 0.000e0 (rel 0.00e0)

pass-back row-offset equivalence: 15 case(s) bit-exact on gfx1201; 12 of them mix numerically different engines
test passback_row_offset_equivalence ... ok
```

## 6. Byte-identity checks

| comparison | result |
|---|---|
| `HIPFIRE_OFFLOAD_EXEC=cpu` vs `passback` with `HIPFIRE_OFFLOAD_PASSBACK_SHARE=0`, 256 tokens | **identical** (`cmp`), `0` `split:` lines |
| `pcie` vs `pcie` + stray `HIPFIRE_OFFLOAD_PASSBACK_SHARE=0.4`, 256 tokens | **identical**; stray-share notice printed |
| `cpu` vs `passback` (share auto), 256 tokens | **identical** |
| `cpu` vs `passback` (share auto), 512 tokens (EOS at 642 chars, so both runs end early) | **identical** |
| `cpu` vs `pcie`, 256 tokens | **identical** |

That last row is the honest headline for this fixture: on the registry 9B artifact,
**all three modes produce the same greedy token stream** for this prompt. The engines
*do* differ numerically per row (§ 5: 1.2e-7 – 1.5e-5 absolute on 12 of 15 cases), so
the mixture is genuine; this prompt/budget simply sits below the argmax sensitivity.
(The 2026-09-27 `cpu`-vs-`pcie` divergence the CPU-exec records report was measured on
a **different** 9B artifact — the local-only one the headroom record documents as
un-pullable — so the two are not comparable.)

## 7. Decoded text (coherence read, `passback`, share auto, budget 24)

```
{open('/tmp/v7b_passback.txt', errors='replace').read().strip()}
```

Coherent, correct for the prompt, and terminated by EOS; identical to the `cpu` and
`pcie` arms' text byte for byte.

## Not measured

- **The serve / slots path, and prefill.** Every arm here is `bench --backend noslots
  --workload stateless` and `run`; the multi-slot engine, batching, and long-context
  behavior are untouched by this mode but untested by this record.
- **Any other host, link width, core count, or AVX2 availability.** Non-AVX2 hosts are
  refused by the split entirely (the step stays on the CPU) and were not exercised.
- **Any other GPU arch or model.** gfx1201 only; the 9B (32 layers, all qt 13) only.
  The 27B/64-layer dense case, MoE/A3B, spec decode (DFlash/MTP), and any quant other
  than qt 13 in a *real* model are untested here.
- **Long-context decode.** 128-token bench runs; the scheduler's freeze rule and the
  Share trajectory over thousands of steps are covered only by the 256-token `run`
  arms.
- **The two-stream contingency.** Whether one non-blocking stream plus an event pair
  around the D2H would help is not measured; the same-stream ordering was sufficient
  (the copies are ~0.03–0.08 ms against 0.1–0.6 ms CPU arms).
- **`Ky` / host-DRAM contention** — the headroom record's subject, not repeated here.


## 8. Offload-amount sweep (9B) and one 27B comparison point

`[amended]` Added after the original. Same method as § 1 (`hipfire bench … --json`,
one fresh daemon per arm, arms interleaved, `--spec off --backend noslots
--workload stateless`, 128 tokens, greedy, the same committed prompt), sweeping the
resident-layer budget so the spilled prefix changes. Two rounds per point.

Bars: [`data-2026-10-01-offload-passback-split/mode-sweep-vs-spilled-layers.png`](data-2026-10-01-offload-passback-split/mode-sweep-vs-spilled-layers.png)
(+ `.svg`).

### 8.1 Qwen3.5-9B mq4 (32 layers)

| layers spilled | `cpu` tok/s | `passback` tok/s | `pcie` tok/s | passback vs `cpu` | passback vs `pcie` |
|---:|---:|---:|---:|---:|---:|
| 4 | 43.80 | 47.50 | 39.45 | **+8.4 %** | +20.4 % |
| 8 | 28.10 | 30.50 | 23.55 | **+8.5 %** | +29.5 % |
| 12 | 20.50 | 23.20 | 16.65 | **+13.2 %** | +39.3 % |
| 16 | 16.20 | 18.45 | 12.85 | **+13.9 %** | +43.6 % |

Per-round raw (the drift check; medians above are of these two):

```
  4   cpu       [43.6, 44.0]   passback [47.1, 47.9]   pcie [39.2, 39.7]
  8   cpu       [28.0, 28.2]   passback [31.0, 30.0]   pcie [23.7, 23.4]
 12   cpu       [20.5, 20.5]   passback [23.7, 22.7]   pcie [16.6, 16.7]
 16   cpu       [16.2, 16.2]   passback [18.4, 18.5]   pcie [12.9, 12.8]
```

Reading: **the pass-back's advantage grows with the spill** — +8.4 % at 4 of 32
spilled, +13.9 % at 16 of 32 — and its advantage over `pcie` grows faster still
(+20 % → +44 %), because `pcie` is the arm that pays the whole link read per step
while `cpu` and `passback` share the host's own bandwidth. The budget-8 point here
(cpu 28.10 / passback 30.50) is the same shape as § 1's three-round point (27.40 /
30.60) measured a few hours earlier on a differently loaded host — the pass-back
figure reproduced to 0.3 %, the `cpu` figure moved 0.7 tok/s, which is the honest
size of the pass-back gain's uncertainty at this budget (**+8.5 %…+11.7 %**, not a
single number).

### 8.2 Qwen3.8-27B mq3-xt (64 layers), 8 of 64 spilled

`~/.hipfire/models/qwen3.8-27b.mq3-xt` — 11,777,616,896 B, md5
`80bb9198e6a565fc006b2ae1b7c89eca` (the headroom record's pinned 27B; the registry
`qwen3.8:27b-mq4-xts` fixture is **not** on this disk, so this is the comparable
artifact, not the canonical one). One round; a second round was abandoned when the
`pcie` arm wedged under memory pressure (`free` at 0 GB / 12 GB available) — the
same stall the headroom record documents at 27B load.

| mode | decode tok/s | ttft ms | vram_free_mb |
|---|---:|---:|---:|
| `cpu` | 14.40 | 586 | 4730 |
| **`passback` (auto)** | **17.10** | 588 | 4730 |
| `pcie` | 14.40 | 591 | 4758 |

**+18.8 % over `cpu`.** The honest control is *spill fraction*, not spill count: 8 of
64 is 12.5 % of the model, the same fraction as the 9B's 4-of-32 point, which gains
+8.4 % — so the 27B more than doubles the gain at equal spill fraction, which isolates
model size from spill amount. (Comparing at equal *counts* — 8 of 64 versus 8 of 32 —
confounds the two: the 27B wins there as well, despite spilling half the fraction.)
The mechanism is per-step bytes: each spilled projection is a larger vector, so the
CPU arm's serialized host time per step is longer and there is more of it to
recover. Note
`pcie` ties `cpu` exactly here (14.40 both) rather than winning or losing — the
"GPU route is ~20 % slower per byte than the CPU route" from the headroom record
lands on a tie once the spill is only 8 of 64 layers and the resident work dominates.

## 9. `hipfire offload-bench` CLI paths

`[amended]` Added after the original — these are user-facing and were unrecorded.

| check | result |
|---|---|
| while a daemon is live | exit **1**, stdout empty (0 bytes), one line naming the live pid: `a hipfire daemon is live (pid N); stop it first (`hipfire stop`) — offload-bench measures the device directly, and a concurrent daemon makes both engine rates meaningless` |
| `--json` | exit 0, one object: `arch`, `recommended {format, share}`, `formats[] {format, m, k, bytes, cpu_bytes_per_s, gpu_bytes_per_s, share, predicted_speedup}` |
| `--write` | wrote `offload_passback_share = "0.377"` to `~/.hipfire/config.toml`; `hipfire config get memory.offload_passback_share` printed `0.377` |
| restore | `hipfire config reset memory.offload_passback_share` left the config byte-identical to its pre-test copy (`diff` clean) |

## 10. Not measured: a long-horizon text arc (attempted, then abandoned)

`[amended]` A 32k-token completion per backend was started and then dropped at the
record owner's request: the prompt was a self-verifying integer sequence rather than
the long *prose* they wanted, and re-scoping it is not this record's business. Its
outputs and prompt were deleted with it, so no numbers from the attempt are quoted
here. Two pitfalls it did establish, both worth knowing before anyone retries:

* **`hipfire run` has no reasoning flag, and a long budget is consumed by hidden
  reasoning.** With `reasoning.mode = on` (the config default, and the 9B is a
  reasoning model), two arms each spent ~20 minutes of real decode and wrote **1
  byte** — a bare `println!()` around an empty visible answer, because every token
  went into the invisible think block. `reasoning.mode = off` streamed text
  immediately, at 27 tok/s (matching the sweep's `cpu` point). AGENTS.md § 7 warns
  about the `bench` side of this ("a reasoning model cannot close `<think>` inside the
  token budget, and the daemon fails that turn closed"); `run` has only the config
  key.
* **`-n` does not set the arc's length — the model's own stop does.** Instructed to
  "keep going as long as you are allowed to", the 9B stopped by EOS at 1000 (997
  integers, 3887 bytes ≈ 3.9k tokens, 143 s). For a long arc, the length has to come
  from the prompt (a bounded range) rather than from the token budget.

What *is* measured about long-horizon behaviour in this record is § 6: the 256- and
512-token text comparisons across all three modes, and the element-level divergence
in § 5.

## Appendix: whole `hipfire bench --json` per arm (9 arms)

Identical flags to § 1; `samples` are single-run, so the per-arm spread across rounds
is the noise evidence, not the stdev within a run.

{bench_full}
