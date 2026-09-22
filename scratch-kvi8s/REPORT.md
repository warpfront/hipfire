# KvI8s: existing `Fp8` vs `Q8` KV on gfx1201 (card-B)

Worktree `wt-kvi8s` @ `34ccae74b` (branch `kvi8s`, off `wt-lloyd`).
Artifact `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`
(14,987,185,152 B; first two eval cells used the resolving symlink
`~/.hipfire/models/qwen3.8-27b.mq4-xt`, all later cells the absolute path —
same file, symlink untouched).
GPU `GPU-e475645fe0200397` (card-B), HOME `ab1`.
Evaluator `cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`;
daemon+CLI built in-worktree; `HIPFIRE_DAEMON_BIN=<wt>/target/release/daemon`.
No source change (no kernel edited). ab1 `config.toml` was varied for one
diagnostic pair only, then restored byte-identical (diff-verified against
`scratch-kvi8s/config.ab1.bak`).

## 1. Mechanism — what each mode actually does today

Both modes are selectable with no code change (`FULL_LADDER`,
`crates/hipfire-runtime/src/kv_mode.rs:78`).

| | `Q8` | `Fp8` (native E4M3FN) |
|---|---|---|
| Per-element encoding | symmetric int8, **no zero point**: `q = clamp(rn(x·127/amax), −127, 127)` | OCP E4M3FN codes via HW `cvt_pk_fp8_f32` (RNE, finite saturation) |
| Scale granularity | **per 32-element block** (8 blocks/head at D256): `scale = amax/127` stored f16 | **per (token, KV head)** over the full 256-elem head: `s = amax/448` stored f16 (zero row → 1.0; subnormal floor bump + next-up guard so `448·s ≥ amax`) |
| Zero point stored | No (zero maps to zero) | No (scale-only) |
| Row layout | 34 B/block = `[f16 scale][32×int8]`; 1088 B/side/token/layer at Hkv4/D256 | `[Hkv×D codes][Hkv f16 scales]` = **1032 B**/side/token/row at Hkv4/D256 |
| K vs V | K = Q8_0; V = Q8_0 by default, **independently re-encodable** to lloyd2/3/4 via `--kv-v` / `HIPFIRE_KV_V` | **Indivisible pair: fp8 on both sides.** Non-q8 V rejected at load and at dispatch. `--kv-v q8` is only the neutral spelling |
| VRAM (measured §4) | 2×1088 B/token/layer over 16 KV layers; VMM `mapped_prefix=1927`, `physical_cap=max_seq=262144` | 2×1032 B/token/layer over 16 KV layers; VMM `mapped_prefix=2032`, `physical_cap=max_seq=262144` |
| Decode FA, short ctx (scalar kernel) | **Dequant first**: `q·(scale·code)` accumulated in f32 FMA | **Dequant first**: `q·(ks·e4m3_to_f32(code))` in f32 FMA |
| Decode FA, long ctx (flash tile) | **Dequant first**: wave-cooperative f32 dot + 2-pass softmax | **Dequant first**: same structure, E4M3→f32 decode per element |
| Prefill FA2 (gfx1201) | **Dequant at fill**: K/V decoded once per 64-key tile into fragment-major **f16** LDS planes, then **f16 WMMA** body (Q/K/V/P f16, accumulators f32). Admission narrow: batch 64..=512, batch%16==0, **eager-only** (never under graph capture) | **In place, native dtype**: stage-b copies E4M3 codes verbatim into fp8 fragment planes (no decode, no f16 rounding; scales read header-direct) and runs **fp8 (E4M3) QK + PV WMMA legs** with f32 scores/softmax-state/O. Default route is qresident (first in dispatch order): batch 64..32768, capture-safe |

File:line citations (all paths under `wt-kvi8s`):

- Q8 quant rule (symmetric, per-32, no ZP): `kernels/src/kv_cache_write_q8_0_batched.hip:49-56`.
- Q8 layout math: `crates/saddle-core/src/kv.rs:1380-1384`.
- Q8 V default + VMode axis: `kv.rs:1430` (`v_mode: VMode::Q8`); `kv.rs:258-264` (VMode enum);
  `--kv-v` → V-only realloc: `crates/hipfire-runtime/examples/eval_hipfire.rs:588-596`;
  carrier gate (Lloyd-V needs FWHT-K): `crates/hipfire-arch-qwen35/src/carrier.rs:246-258`.
- fp8 scale/code rule: `kernels/src/kv_cache_write_q8_0_batched.hip:186-210`.
- fp8 row layout: `kv.rs:2856-2859` (1032 B @ Hkv4/D256); constructor `kv.rs:2948-2969`;
  `quant_fp8=true`, empty scales: `kv.rs:3001`.
- fp8 indivisible K/V: `kv.rs:28-34`, `kv.rs:2066-2070`, `kv.rs:2947`,
  `carrier.rs:322-326`, dispatch re-check
  `crates/hipfire-dispatch/src/families/kv_tier.rs:344-351`, eval enforcement
  `eval_hipfire.rs:449-454`.
- Q8 scalar decode (f32 FMA): `kernels/src/attention_q8_0_kv.hip:75-83`.
- fp8 scalar decode (f32 FMA after E4M3→f32): same file `:258-286`.
- Flash tile = f32 dot for both: `kernels/src/attention_flash_q8_0_tile.hip:5`; fp8 branch `:407+`.
- Q8 FA2 = f16 planes + f16 WMMA: `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:9-10`,
  KMODE doc `:48-60`; launcher `crates/rdna-compute/src/attention.rs:3644-3646`.
- Q8 FA2 admission (narrow, eager-only): `attention.rs:3633-3642`.
- fp8 stage-b = in-place fp8 WMMA: `attention.rs:4461-4474`;
  kernel fill `:734-745`, QK `:923-934`, PV `:1155`
  (`kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip`).
- fp8 dispatch order + admission: `crates/hipfire-dispatch/src/families/attention.rs:2310-2380`;
  capture-safe note `:2299-2301`; flag defaults ON:
  `crates/rdna-compute/src/feature_flags.rs:666-669`.
- Tier-exclusive dispatch: `crates/hipfire-dispatch/src/families/kv_tier.rs:352-356` (fp8 keys),
  `:1460-1472` (cross-tier rejection tests).
- Arch default auto→fp8: `crates/hipfire-runtime/src/kv_mode.rs:63-76`.

## 2. Quality — KLD vs teacher refs, iu4 route, prefill scoring, full 24×2048 refs

`HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 eval_hipfire --model <artifact>
--ref <ref> --kv-mode <mode> --kv-v q8 --scoring-mode prefill --max-chunks 24`

| cell | slice-mean KLD | NLL / PPL | repeats |
|---|---|---|---|
| fp8 WT2 c24 (control) | **0.083278** | 1.886010 / 6.5930 | — (c2 rep bit-exact: 0.065213 ×2) |
| q8 WT2 c24 | **0.083487** | 1.884666 / 6.5842 | — |
| fp8 agentic c24 (control) | **0.215471** | 1.996614 / 7.3641 | rerun bit-exact |
| q8 agentic c24 | **0.206581** | 2.001480 / 7.4000 | rerun bit-exact |

Deltas (same build, same setup): WT2 q8−fp8 = +0.000209 (**+0.25%, tied, noise**);
agentic q8−fp8 = −0.008890 (**−4.13%, q8 better, stable** — both arms reproduced
bit-exactly on rerun: 0.215471→0.215471, 0.206581→0.206581).

Control note: quoted controls 0.0825/0.2042 did NOT reproduce from this fresh-worktree
build. Per Main this matches sibling A4Fine's identical pair (WT2 0.0825→0.083278,
agentic 0.2042→0.215471) — a build-provenance effect (precompiled IU4 object).
Absolute KLD is therefore comparable only within one build; the fp8-vs-q8 delta above
is self-consistent and is what the verdict rests on. Do not chase 0.0825/0.2042.

Arm confirmation: fp8 logs `fp8/fp8 native KV (E4M3FN + f16/token/head)` +
`KV cache: fp8-e4m3 (16/64 layers carry KV…)`; q8 constructs via
`KvCache::new_gpu_q8` (`eval_hipfire.rs:435-441`) with `HIPFIRE_KV_MODE=q8` logged.

## 3. Perf — ABBA pairs, card-B, daemon bench, `HIPFIRE_GRAPH=1`

Matrix: `hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,8192 --ctx 128 --tg 128
--spec off --runs 3 --warmups 1 --kv-mode <mode> --json` (median of 3 shown).
TTFT: `--ttft --prompt-file benchmarks/prompts/ttft_5900.txt --runs 8 --warmups 2 --json`.
Every arm loader-line confirmed (`KV cache: Fp8/Q8 vmm …`, see per-run `.stderr`).

Matrix pair 1 (A1 B1 B2 A2) and pair 2 (A3 B3 B4 A4), tok/s:

| run | mode | pp512 | pp8192 | tg128 |
|---|---|---|---|---|
| A1 | fp8 | 3335.7 [3323.7, 3335.7, 3342.0] | 3665.5 [3669.8, 3665.5, 3661.0] | 36.492 |
| B1 | q8 | 2837.7 [2822.2, 2837.7, 2839.4] | 2627.3 [2630.2, 2627.3, 2624.9] | 36.517 |
| B2 | q8 | 2836.1 [2824.7, 2836.1, 2838.3] | 2627.2 [2632.5, 2627.2, 2622.9] | 36.492 |
| A2 | fp8 | 3364.7 [3354.4, 3364.7, 3365.7] | 3642.4 [3642.4, 3643.3, 3641.8] | 36.481 |
| A3 | fp8 | 3331.7 [3333.3, 3320.8, 3331.7] | 3668.7 [3673.4, 3668.7, 3663.6] | 36.513 |
| B3 | q8 | 2837.9 [2829.3, 2842.8, 2837.9] | 2626.2 [2627.2, 2626.2, 2623.6] | 36.512 |
| B4 | q8 | 2810.2 [2810.2, 2802.8, 2823.3] | 2625.7 [2627.3, 2625.7, 2625.0] | 36.544 |
| A4 | fp8 | 3321.8 [3321.8, 3321.7, 3343.0] | 3657.5 [3657.5, 3658.2, 3654.8] | 36.532 |

Means: fp8 pp512 **3338.5**, q8 pp512 **2830.5** → fp8 **+17.9%**.
fp8 pp8192 **3658.5**, q8 pp8192 **2626.6** → fp8 **+39.3%**.
tg128: fp8 mean 36.504, q8 mean 36.516 → **tied** (+0.03%).
Across-rep drift ≤0.9%, far below the 18–39% gaps.

TTFT pair (A1 B1 B2 A2), 5909-token prompt:

| run | mode | TTFT median ms | stdev | pp tok/s |
|---|---|---|---|---|
| A1 | fp8 | 1631.46 | 1.63 | 3621.9 |
| B1 | q8 | 2217.53 | 2.30 | 2664.7 |
| B2 | q8 | 2221.25 | 3.43 | 2660.2 |
| A2 | fp8 | 1631.32 | 0.93 | 3622.2 |

fp8 TTFT wins by **~588 ms (−36%)**; fully explained by prefill rate
(5909/3622 = 1.631 s vs 5909/2662 = 2.220 s). A1 reproduces the flag-free
reference (1628.2 ms / 3629.3 tok/s).

FA2 ablation (pp8192, `kernel.attn_qresident/gfx12_fa_packet/gfx12_fa2_prefill=false`,
config restored byte-identical after): fp8-noFA2 **308.0** tok/s (vs ~3660 with FA2,
**11.9× route gain**); q8-noFA2 **2006.9** (vs ~2626, +31%). The fp8 lead is
admission-dominated (wide capture-safe FA2 family vs q8's narrow eager-only FA2),
with fp8-WMMA-vs-f16-WMMA arithmetic secondary. Raw files: `abl_fp8_nofa2.json`,
`abl_q8_nofa2.json`.

## 4. VRAM footprint / long-context ceiling

`vram_free_mb` = 16838 under both modes (model-loaded). Per-token bytes from frozen
strides (§1): q8 1088 B vs fp8 1032 B per side/token/layer — near-identical, fp8 if
anything marginally smaller. Same ceiling: `physical_cap=max_seq=262144`, VMM backend,
16/64 KV layers both arms. **Neither mode changes the long-context ceiling.**

## 5. Battery under q8 — 5/5 `finish=stop`

`python3 scripts/serve_harness.py --mode battery --model qwen3.8:27b-mq4-xt
--thinking off --kv q8` (second run with `--out battery_q8_turns.json`; sampled
T=1.0 so texts differ run to run — both runs 5/5 stop). Full decoded texts from the
`--out` run, verbatim:

t1 [code], gen=316: "```python\ndef merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into a single
sorted list.\"\"\"\n    result = []\n    i, j = 0, 0\n\n    while i < len(a) and j <
len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i +=
1\n        else:\n            result.append(b[j])\n            j += 1\n\n
result.extend(a[i:])\n    result.extend(b[j:])\n\n    return result\n```"

t2 [reason], gen=186: "Step 1: First part of the trip \\n\\[\\n60 \\text{ mph} \\times 2.5
\\text{ hours} = 150 \\text{ miles}\\n\\]\\n\\nStep 2: Second part of the trip
\\n\\[\\n40 \\text{ mph} \\times 1.5 \\text{ hours} = 60 \\text{ miles}\\n\\]\\n\\nStep
3: Total distance \\n\\[\\n150 + 60 = 210 \\text{ miles}\\n\\]\\n\\n**Final answer: 210
miles**"

t3 [factual], gen=176: "Earth's axis is tilted about 23.5 degrees relative to its orbital
plane. As Earth orbits the Sun, different hemispheres receive more direct sunlight at
different times of year. This changing angle and duration of sunlight causes the seasonal
cycle."

t4 [prose], gen=288: "Each dawn, Elias counted the gray rocks below the lighthouse and
called them by names no one else knew. One fog-drenched morning, he found a small brass
lantern lying among the seaweed, its glass uncracked and its flame still glowing without oil
or wick. When he touched it, a child's voice whispered, \"I'm still looking for the way
home.\" He carried the lantern up to the lamp room and lit it beside the great beam, hoping
the light would show someone where they had lost their way."

t5 [instruct], gen=120: "1. Write clear, descriptive names for variables, functions, and
classes.\n2. Keep functions small and focused on a single responsibility.\n3. Avoid
duplicated code by reusing abstractions and shared utilities.\n4. Add meaningful comments
only where intent or logic is not obvious.\n5. Write tests to document expected behavior and
protect against regressions."

First run (no `--out`, log `battery_q8.log`) was likewise 5/5 stop
(avg prefill 1223.6 tok/s, decode 36.4 tok/s).

## 6. Verdict

The gfx1201 fp8-KV default is justified **by perf, not by quality** — and on quality it
is a small loss on the distribution actually served:

- **Quality: NO.** WT2 tied (+0.25%, noise); agentic favours q8 by a stable, bit-exact
  4.1%. The default is not a quality win anywhere measured.
- **Perf: YES, strongly on prefill.** fp8 +17.9% pp512, +39.3% pp8192, −36% TTFT;
  decode tied. The win is route-admission-dominated (ablation §3), i.e. it lives in
  the fp8 FA2 family that has no q8 counterpart — not in Stephan's scale-only fold.
- **VRAM/ceiling: NEITHER.** Near-identical footprint, same 262 K ceiling.

For the portable-8-bit question: nothing in these numbers defends fp8 KV on quality,
and the perf advantage is an FA-kernel-route property of this card, not an information
property of the E4M3 alphabet — a symmetric-int8 tier inherits the fold cost by
construction but would need its own FA2 route to inherit the speed.
