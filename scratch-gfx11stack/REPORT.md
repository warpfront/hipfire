# gfx11 stacked IU4 prefill integration: q8 ship gate

## Verdict and scope

**Main decision: make C the gfx11 baseline in a later integration step; this branch does not flip defaults.** C is fastest in both q8 matrix prefill rows on both cards, and its decode is within 0.1% of P. Its WT2 c24 score remains below the 0.10 budget, but the combination **does not compose quality**: C gives back both solo WT2 improvements and is marginally worse than P on both cards. This is a measured speed-versus-quality tradeoff, not a failure of co-dispatch. Main prioritizes the speed; default selection remains outside this worktree.

P = gridspec on, gfx1100 LF16 shape on / gfx1151 shape off, symfold off, A4=8; S = P plus symfold on; A = P plus A4=2; C = both on. The typed defaults remain `kernel.gfx11_iu4_symfold=false` and `kernel.gfx11_a4_candidates=8`. The two levers were integrated without changing the production baseline.

The **ship gate** is q8 matrix pp512/pp8192 with decode ctx128/tg128: no prefill row more than 1% slower than P and decode within 1% of P. TTFT-511 is an odd-fill/partial-N diagnostic, not a magnitude estimator; a real prompt length of 511 uses four 128-column tiles with one padded column. WT2 c24 under q8 K/V is the only quality gate; agentic scores are non-gating historical artifacts.

## Co-dispatch proof before timing

One **gfx1100 arm-C daemon process** had typed config `home-gfx1100-codispatch/config.toml` with `gfx11_iu4_symfold=true`, `gfx11_a4_candidates="2"`, gridspec true, and shape true. Its single `rocprofv3 --kernel-trace --stats` pass (`codispatch-gfx1100/daemon_kernel_stats.csv`) dispatched 272 SET and 128 ADD LF16 `_symfold` GEMMs alongside 64 fused SwiGLU+i4, 128 RMSNorm+rotate+i4, 48 gated-norm+i4, and 16 sigmoid/rotate+i4 producer calls. The daemon log asserts `GPU dev 0: gfx1100`. There were zero standalone-quantizer rows. This ties both kernel families to the same process, not separate solo runs.

The integrated-tree gfx1100 c2 producer code object (`isa/rms-gfx1100-c2.isa:733,1059`) has two `v_cndmask_b32_e64` choices between `2.0` and `0x3fdb6db7` and **zero** `s_cmp_lg_u32 ..., 8` search-loop comparisons; the gfx1151 c2 object has the same pattern at `:731,1055`. The integrated-tree symfold code object (`isa/symfold-gfx1100.isa`, `isa/symfold-gfx1151.isa`; summary `isa-evidence.txt`) contains `v_xor_b32 ... 0x88888888` and `v_wmma_i32_16x16x16_iu4 ... neg_lo:[1,1,0]` in `_symfold` entry families. Config acknowledgment files, ISA, and the runtime co-dispatch trace jointly establish both live knobs; the trace alone names kernels but does not encode the producer candidate count. The source-level compiler injection reads the typed `HIPFIRE_GFX11_A4_CANDIDATES` process config and emits `-DIU4_A4_CANDIDATES=2` for this arm.

## q8 matrix: primary throughput ship gate

Command per process: `hipfire bench <model> --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode q8 --json`, with `HIPFIRE_DAEMON_BIN` pinned to the integration daemon. Every JSON asserts `gpu.arch`, `kv_mode=q8`, three timed samples per row, and the requested shapes. All values below are tok/s, **higher is better**; brackets retain the three raw timed samples. Fresh daemon per process, one process at a time.

### gfx1100 matrix

| process | arm | pp512 tok/s (3 samples) | pp8192 tok/s (3 samples) | ctx128/tg128 tok/s (3 samples) |
|---|---|---|---|---|
| P1 | P | 1924.50 [1921.50, 1924.50, 1928.80] | 1889.90 [1894.70, 1889.90, 1888.30] | 49.1228 [49.1270, 49.1228, 49.1083] |
| S1 | S | 2012.30 [2010.90, 2013.20, 2012.30] | 1975.20 [1975.20, 1975.20, 1973.90] | 48.9833 [48.9615, 48.9975, 48.9833] |
| A1 | A | 2019.70 [2014.80, 2019.70, 2023.10] | 1966.50 [1969.00, 1966.50, 1965.00] | 49.1905 [49.1868, 49.1905, 49.1991] |
| C1 | C | 2128.00 [2129.50, 2127.50, 2128.00] | 2071.90 [2071.90, 2072.10, 2070.00] | 49.0785 [49.0845, 49.0785, 49.0759] |
| C2 | C | 2122.40 [2122.40, 2122.40, 2126.80] | 2069.90 [2072.20, 2069.70, 2069.90] | 48.9350 [48.9223, 48.9350, 48.9429] |
| A2 | A | 2009.00 [2010.00, 2008.20, 2009.00] | 1957.40 [1959.90, 1957.10, 1957.40] | 48.9846 [48.9846, 48.9589, 49.0016] |
| S2 | S | 2002.90 [1997.60, 2003.90, 2002.90] | 1969.50 [1968.70, 1971.50, 1969.50] | 49.0486 [49.0514, 49.0437, 49.0486] |
| P2 | P | 1900.90 [1893.40, 1900.90, 1901.00] | 1869.10 [1869.10, 1867.80, 1869.90] | 48.9784 [48.9784, 49.0094, 48.9769] |

Pooled means of process medians (tok/s):

| arm | pp512 | pp8192 | ctx128/tg128 | pp512 vs P | pp8192 vs P | decode vs P |
|---|---:|---:|---:|---:|---:|---:|
| P | 1912.700 | 1879.500 | 49.05063 | +0.000% | +0.000% | +0.000% |
| S | 2007.600 | 1972.350 | 49.01597 | +4.962% | +4.940% | -0.071% |
| A | 2014.350 | 1961.950 | 49.08756 | +5.314% | +4.387% | +0.075% |
| C | 2125.200 | 2070.900 | 49.00677 | +11.110% | +10.184% | -0.089% |

### gfx1151 matrix

| process | arm | pp512 tok/s (3 samples) | pp8192 tok/s (3 samples) | ctx128/tg128 tok/s (3 samples) |
|---|---|---|---|---|
| P1 | P | 767.80 [767.80, 768.70, 767.80] | 711.60 [720.40, 711.60, 697.00] | 14.8528 [14.8522, 14.8528, 14.8540] |
| S1 | S | 785.20 [790.10, 785.20, 777.50] | 723.40 [727.70, 723.40, 721.60] | 14.8555 [14.8555, 14.8555, 14.8552] |
| A1 | A | 772.30 [783.00, 772.30, 770.00] | 712.30 [713.20, 712.30, 709.10] | 14.8537 [14.8521, 14.8538, 14.8537] |
| C1 | C | 799.20 [809.10, 799.20, 794.10] | 738.50 [740.30, 738.00, 738.50] | 14.8505 [14.8503, 14.8511, 14.8505] |
| C2 | C | 797.10 [805.10, 797.10, 792.20] | 738.30 [739.20, 738.30, 736.20] | 14.8544 [14.8545, 14.8544, 14.8544] |
| A2 | A | 766.90 [779.00, 766.90, 763.30] | 705.30 [707.70, 705.30, 704.40] | 14.8549 [14.8538, 14.8551, 14.8549] |
| S2 | S | 767.70 [778.10, 767.70, 763.50] | 715.90 [717.90, 715.90, 714.70] | 14.8542 [14.8542, 14.8542, 14.8554] |
| P2 | P | 740.20 [751.10, 740.20, 736.30] | 686.00 [687.00, 686.00, 683.60] | 14.8566 [14.8566, 14.8576, 14.8563] |
| A3 | A | 763.40 [774.70, 763.40, 759.50] | 704.60 [707.90, 704.00, 704.60] | 14.8577 [14.8592, 14.8577, 14.8577] |
| C3 | C | 791.70 [804.00, 791.70, 789.20] | 735.10 [736.90, 735.10, 734.60] | 14.8614 [14.8608, 14.8614, 14.8619] |
| P3 | P | 741.90 [752.60, 741.90, 737.50] | 685.10 [687.80, 685.10, 685.00] | 14.8540 [14.8540, 14.8539, 14.8542] |
| S3 | S | 768.30 [778.90, 768.30, 766.50] | 718.00 [718.60, 718.00, 715.90] | 14.8554 [14.8555, 14.8547, 14.8554] |

Pooled means of process medians (tok/s):

| arm | pp512 | pp8192 | ctx128/tg128 | pp512 vs P | pp8192 vs P | decode vs P |
|---|---:|---:|---:|---:|---:|---:|
| P | 749.967 | 694.233 | 14.85445 | +0.000% | +0.000% | +0.000% |
| S | 773.733 | 719.100 | 14.85504 | +3.169% | +3.582% | +0.004% |
| A | 767.533 | 707.400 | 14.85542 | +2.342% | +1.897% | +0.007% |
| C | 796.000 | 737.300 | 14.85544 | +6.138% | +6.203% | +0.007% |

### Stacking arithmetic and arm choice

- gfx1100 pp512: multiplicative solo expectation `(S/P)*(A/P) = (2007.600/1912.700)*(2014.350/1912.700) = 1.105397x`; measured `C/P = 2125.200/1912.700 = 1.111099x`; **super-multiplicative by +0.570 percentage points**. C/S = 1.058577x; C/A = 1.055030x.
- gfx1100 pp8192: multiplicative solo expectation `(S/P)*(A/P) = (1972.350/1879.500)*(1961.950/1879.500) = 1.095437x`; measured `C/P = 2070.900/1879.500 = 1.101836x`; **super-multiplicative by +0.640 percentage points**. C/S = 1.049966x; C/A = 1.055531x.
- gfx1100 decode: C/P = 0.999106x (-0.089%), within the 1% unchanged-decode gate. All three non-P arms improve both prefill rows, so no row violates the 1% regression threshold.
- gfx1151 pp512: multiplicative solo expectation `(S/P)*(A/P) = (773.733/749.967)*(767.533/749.967) = 1.055856x`; measured `C/P = 796.000/749.967 = 1.061381x`; **super-multiplicative by +0.552 percentage points**. C/S = 1.028778x; C/A = 1.037089x.
- gfx1151 pp8192: multiplicative solo expectation `(S/P)*(A/P) = (719.100/694.233)*(707.400/694.233) = 1.055464x`; measured `C/P = 737.300/694.233 = 1.062035x`; **super-multiplicative by +0.657 percentage points**. C/S = 1.025309x; C/A = 1.042267x.
- gfx1151 decode: C/P = 1.000067x (+0.007%), within the 1% unchanged-decode gate. All three non-P arms improve both prefill rows, so no row violates the 1% regression threshold.

Halo three-window direction checks compare like-numbered P to S/A/C medians. All C/P matrix prefill comparisons are positive: pp512 +4.09%, +7.69%, +6.71%; pp8192 +3.78%, +7.62%, +7.30%. A/P pp512 is +0.59%, +3.61%, +2.90% and pp8192 +0.10%, +2.81%, +2.85%, i.e. positive in every window, with a small first-window margin. The process order is gfx1100 `P1 S1 A1 C1 C2 A2 S2 P2` and Halo the same first eight followed by `A3 C3 P3 S3`; the first eight reverse arm order to cancel first-slot bias. `matrix-summary.txt` and every `matrix-<arch>-<process>-<arm>.json` retain the machine-readable vectors.

## q8 TTFT-511: partial-N diagnostic

Command per process: `hipfire bench <model> --ttft --prompt-file benchmarks/prompts/ttft_511.txt --runs 4 --warmups 2 --spec off --kv-mode q8 --json`. The committed prompt MD5 is `7423e8940920082c6fa11576d23bc9a2`; **every** JSON asserts `prompt_tokens=511` and this MD5 at both top level and `ttft`. JSON does not serialize KV mode for this protocol; the pinned runner command is evidence of q8. Brackets retain the four nested `ttft.samples` in ms.

### gfx1100 TTFT-511

| process | arm | TTFT-511 median ms | four timed samples ms |
|---|---|---:|---|
| P1 | P | 312.669 | [310.974, 312.854, 312.484, 313.093] |
| S1 | S | 303.423 | [302.980, 303.481, 303.365, 303.587] |
| A1 | A | 299.863 | [299.734, 299.992, 300.468, 299.712] |
| C1 | C | 291.433 | [291.333, 291.533, 292.512, 291.120] |
| C2 | C | 291.401 | [290.687, 291.294, 292.056, 291.508] |
| A2 | A | 301.449 | [300.794, 301.444, 301.454, 301.984] |
| S2 | S | 304.318 | [303.494, 304.230, 304.406, 304.417] |
| P2 | P | 314.329 | [313.945, 314.261, 314.396, 314.572] |

Pooled means of process medians (ms; smaller is faster):

| arm | pooled ms | speedup vs P |
|---|---:|---:|
| P | 313.499 | 1.000000x |
| S | 303.870 | 1.031686x |
| A | 300.656 | 1.042716x |
| C | 291.417 | 1.075774x |

Stacking: `(P/S)*(P/A) = (313.499/303.870)*(313.499/300.656) = 1.075755x` expected versus `P/C = 313.499/291.417 = 1.075774x` measured (+0.002 percentage-point interaction). C/S = 1.042734x; C/A = 1.031704x.

### gfx1151 TTFT-511

| process | arm | TTFT-511 median ms | four timed samples ms |
|---|---|---:|---|
| P1 | P | 918.413 | [915.981, 917.860, 920.434, 918.967] |
| S1 | S | 880.008 | [880.664, 880.087, 879.278, 879.928] |
| A1 | A | 905.024 | [903.650, 905.064, 905.309, 904.984] |
| C1 | C | 867.007 | [867.221, 867.455, 866.366, 866.794] |
| C2 | C | 867.646 | [867.836, 867.241, 867.976, 867.456] |
| A2 | A | 904.013 | [904.122, 903.693, 903.939, 904.087] |
| S2 | S | 882.090 | [879.112, 882.753, 881.427, 883.334] |
| P2 | P | 919.986 | [923.593, 919.932, 919.581, 920.041] |
| A3 | A | 904.207 | [904.148, 904.266, 904.103, 904.541] |
| C3 | C | 865.794 | [864.427, 866.496, 865.092, 866.638] |
| P3 | P | 918.732 | [918.667, 918.796, 919.463, 918.391] |
| S3 | S | 883.710 | [884.141, 882.939, 885.517, 883.280] |

Pooled means of process medians (ms; smaller is faster):

| arm | pooled ms | speedup vs P |
|---|---:|---:|
| P | 919.044 | 1.000000x |
| S | 881.936 | 1.042075x |
| A | 904.415 | 1.016175x |
| C | 866.816 | 1.060253x |

Stacking: `(P/S)*(P/A) = (919.044/881.936)*(919.044/904.415) = 1.058931x` expected versus `P/C = 919.044/866.816 = 1.060253x` measured (+0.132 percentage-point interaction). C/S = 1.017443x; C/A = 1.043376x.

Halo odd-fill P/C per-window speedups are +5.93%, +6.03%, +6.11%, all same sign; P/A +1.48%, +1.77%, +1.61%; P/S +4.36%, +4.30%, +3.96%. This checks the partial-N path without treating 511-token magnitude as representative of a 5909-token pass. The earlier gfx1100 TTFT-5909 four-arm raw JSONs are retained but superseded as a competitive metric; they did not pin q8 and must not be mixed into these tables.

## q8/q8 WT2 c24: quality ship gate

Deterministic **one** evaluation per `(card, arm, WT2)` (no paired reruns): `HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 ./target/release/examples/eval_hipfire --model <artifact> --ref /home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks 24 --output <path>`. Each `c24-q8-<arch>-<arm>-wt2.stderr` asserts actual arch and forced `HIPFIRE_KV_MODE=q8 HIPFIRE_KV_V=q8`, and reports 24/24 chunks, 24,552 scored tokens. The evaluator does not print a separate `KV cache:` line; its own forced-mode line is the direct process-level assertion. WT2 budget ≤0.10.

| card | P | S | A | C | C − P | C − S | C − A | budget |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| gfx1100 | 0.076812 | 0.075286 | 0.074228 | 0.076879 | +0.000067 | +0.001593 | +0.002651 | all pass |
| gfx1151 | 0.076483 | 0.074972 | 0.075970 | 0.076901 | +0.000418 | +0.001929 | +0.000931 | all pass |

The **quality interaction is sub-additive/antagonistic**, even while throughput is slightly super-multiplicative: C gives back both solo WT2 gains on both cards (gfx1100 C is +0.000067 versus P and +0.002651 versus A; gfx1151 C is +0.000418 versus P and +0.000931 versus A). C remains under 0.10, but it is not quality-positive. A is best on gfx1100; S is best on gfx1151. The cross-architecture c24 scores do **not** match digit-for-digit under q8/q8: P gfx1100 − gfx1151 = +0.000329, S = +0.000314, A = −0.001742, C = −0.000022. This is a real difference and is not averaged; the solo reports' identity observation applied to their prior asym3 protocol, not a universal property of the LF16 path.

**Provenance correction:** P under q8/q8 reproduces the original campaign WT2 baselines exactly: gfx1100 0.076812, gfx1151 0.076483. Both solo unit reports attributed a later 0.088665 baseline to build provenance; the integrated runs show that it was caused by scoring under asym3 K-cache instead of shipped q8 K-cache. The numeric preregistrations `QUALITY_PREDICTION.txt` and `HALO_QUALITY_PREDICTION.txt` (0.085790/0.238812 for C, cross-arch identity predicted) were computed from those asym3 figures and were explicitly voided in `QUALITY_PROTOCOL_OVERRIDE.txt` before the q8/q8 quality screen. They are **not** q8 predictions; no manufactured q8 absolute prediction is substituted after observing results.

## q8 arm-C decoded batteries

Manual typed-config `hipfire serve ... --kv-mode q8 --kv-backend contiguous` with `HIPFIRE_DAEMON_BIN` pinned, followed by `scripts/serve_harness.py --no-spawn --mode battery --thinking off --speculation off --sampling greedy`. For each card, `battery-q8-<arch>.serve.log` asserts actual `GPU dev 0: <arch>` and **two** `KV cache: q8` loads (prewarm and request budget); this is resolved runtime state, not inferred from a flag. `battery-q8-<arch>.json` has 5/5 `finish=stop`, zero empty, runaway, attractor, and retrieval misses. All ten decoded responses were read verbatim in `battery-q8-decoded-verbatim.txt`: code returns a two-pointer `merge_sorted`, arithmetic concludes 210 miles, seasons explain axial tilt, fiction has Elias finding a music box, and the instruction prompt yields five clean-code bullets. The two cards differ slightly in the seasons sentence (`causes` versus `creates`), with both readable and on task.

## Superseded artifacts: not ship evidence

- `c24-gfx1100-{wt2,ag}.*` is the first arm-C evaluation under asym3 K/q8 V; its WT2/agentic numeric predictions were invalidated by the shipped q8 KV requirement. The earlier gfx1100 `battery-gfx1100.*` serve run resolved `K FWHT-3b` rather than q8; it too is void. These are retained to document the protocol correction, not used in verdict.
- A first Halo throughput window was intentionally killed at remote PID 1295000 on user directive. **Every partial Halo artifact from that window is void**; the complete matrix/TTFT-511 files cited above were overwritten in the subsequent clean window with explicit q8 and a clean pre/post GPU check. No partial-window cell was reused.
- An initial matrix command copied from gfx1201 used `--kv-mode fp8` and was rejected before producing samples: gfx11 requires q8. An initial c24 invocation with fp8 K was also rejected before scoring. These refusals are not treated as measurements.
- The historical non-gating q8/q8 agentic scores obtained before the WT2-only directive were gfx1100 P 0.203307, A 0.209906, C 0.204263 (`c24-q8-gfx1100-*-ag.stderr`). S/agentic and all Halo agentic were intentionally not measured; none of these numbers enters the ship gate. Older asym3 agentic values in the solo reports are not comparable to q8 scores.

## Provenance and reproducibility

- Integration worktree `/home/kaden/hipfire-gfx11stack`, branch `exp/gfx11-stack`: base `7a34a572a`, LF16 cherry-pick `e296c2bb9` -> `b226fa9b1` (resolved gridspec default true), A4 `0bf88f717` -> `01a204634`, symfold config `fdfcce429` -> `11ff7a6fb`, implementation `1e21ba14f` -> `8d1eba1fe`, clarification `ed670e025` -> `b12289ddb`. Subsequent commits record superseded preregistrations, the prompt, and protocol override; no defaults flipped. Native hipx `cargo build --release` and `cargo build --release --example eval_hipfire` succeeded. Never built through sshfs.
- Artifact `/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`; SHA256 `de8ee8256033c3690b0f1a2aff14e77cc88fff490e118648b04a833a3f2969b5` from prior certified artifact, identity re-verified by stat tuple `14987185152 1789995527 5398606`. Matrix/TTFT KV mode q8; evaluator q8 K/q8 V. The 511-token prompt MD5 is `7423e8940920082c6fa11576d23bc9a2`; original 5909-token prompt MD5 `ed720348b81a19fab64d4783c75c1ae3` is only historical.
- Final binary MD5s: `hipfire` `9c2d7abbf4ed87a1f20fadedc48ee3a6`; `daemon` `08df7bc3e7bd0ed873145d88fc69e35f`; `examples/eval_hipfire` `d0d03d551e600fb277a1e120eb45c194`. `HIPFIRE_DAEMON_BIN` pinned for every bench/serve process.
- gfx1100 HIP ordinal 0 is RX 7900 XTX, UID `0x43390a851e296ee5`, runtime logs assert `gfx1100`; HIP ordinal 1 was independently runtime-asserted via `hipfire diag` as gfx1151 and every JSON/evaluator/serve log asserts gfx1151. HIP ordinal 3 (gfx1010) was never used. `rocm-smi --showpids` pre/post of each relevant batch is under `pids-*.txt`; all checks were empty apart from persistent zero-VRAM `gpusentry`. One GPU process per card at a time, lease announced and explicitly released between windows.
- Supporting machine-readable evidence is under `scratch-gfx11stack/`: per-process `matrix-*.json` and `oddfill-*.json`; `matrix-summary.txt`, `oddfill-summary.txt`, `c24-q8-summary.txt`; `codispatch-gfx1100/daemon_{kernel_stats,kernel_trace,agent_info}.csv`; `isa-evidence.txt` and `isa/`; `battery-q8-*.json`, `battery-q8-*.serve.log`, `battery-q8-decoded-verbatim.txt`; typed config acknowledgments and process logs. The report and committed prompt remain readable even where raw scratch data is not committed.
