# gfx1100 IU4 workgroup-shape experiment

## Verdict

The clean rerun preserved the promising LF16 signal. Reusing the existing 16-wave low-footprint full-tile body on gfx1100 improved exact-N gate/up by **1.1192x** and down/add by **1.0709x**. Combined with grid specialization, the real partial-N gate/up shape at `N=5909` improved from **102.514 to 124.614 TOPS (1.2156x)**, or **40.47% to 49.20%** of the measured 253.296-TOPS ceiling. The result clears the 1.15x continuation bar but remains 37.366 TOPS below the 161.98-TOPS target (76.93% of target).

The candidate composes with grid specialization and has no attributable gfx1151 regression because its selector is strictly gated to gfx1100. Keep it as the default-off `kernel.gfx11_iu4_shape` experiment. It is a material scheduling win, not the complete path to 2,500 tok/s.

## Implementation

The branch starts from `mq4-lloyd-gfx11-base` and includes the four grid-specialization commits before this change.

- Registered typed bool `kernel.gfx11_iu4_shape` (legacy process spelling `HIPFIRE_GFX11_IU4_SHAPE`), default false.
- Extended compilation of the existing LF16 arithmetic body to gfx1100.
- Added gfx1100 row-major LF16 SET and ADD wrappers for graph/replay capture. Eager launches retain the column-adjacent wrapper; both wrappers call the same body and differ only in mapping `blockIdx.{x,y}`.
- On gfx1100, full SET and ADD tiles select LF16 when the knob is true. A partial-N launch selects it only when grid specialization has separated the unchecked interior from the guarded tail.
- On gfx1151, the knob is ignored. Its shipped SET-only LF16 decision and ordinary ADD route are unchanged.
- Grid specialization remains independently controlled; this worktree preserves its inherited default-off behavior.

## Environment and protocol

- Host: `hipx`, ROCm 7.15.26333.
- gfx1100 identity asserted by architecture on every XTX run; gfx1151 identity asserted likewise.
- Model: `qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`.
- One session SHA-256 check matched `de8ee8256033c3690b0f1a2aff14e77cc88fff490e118648b04a833a3f2969b5`; later windows used the required stat triple `14987185152 1789995527 5398606`.
- Every timing arm used fresh processes and F/R/R/F ordering. Microbenchmarks used two warmups and nine timed samples per process. `hipfire bench` used graph-forward on, `--spec off`, the release daemon, and architecture-bearing JSON.
- GPU process lists were clean before and after every lease except the zero-VRAM `gpusentry` process.

## Launch/resource sweep

The HIP runtime reports 48 processor units on the 96-CU gfx1100. The table therefore gives both raw occupancy-API residency per reported unit and the average normalized over 96 hardware CUs.

| Variant | Block | Waves/block | Blocks/reported unit | Waves/reported unit | Blocks/96-CU avg | Waves/96-CU avg | VGPR | SGPR | LDS | Scratch/spills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped | `[32,8,1]` | 8 | 2 | 16 | 1 | 8 | 189 | 22 | 30,720 B | 0 |
| LF16 | `[32,16,1]` | 16 | 2 | 32 | 1 | 16 | 96 | 26 | 30,720 B | 0 |
| LF32 scratch candidate | `[32,32,1]` | 32 | 2 | 64 | 1 | 32 | 96 | 25 | 30,720 B | 0 |

The selected LF16 variant is at the measured 16-wave/CU knee, uses an explicit 96-VGPR attribute rather than approaching the 240-256 compiler cap, and emitted zero scratch/spills. SET and ADD share these resources. LF32 also emitted zero spills but did not improve throughput enough to justify its shape: exact gate/up `N=8192` was 118.081 TOPS shipped versus 119.722 TOPS LF32, only **1.0139x**. It remained a scratch-only candidate.

### Exact daemon grids

| Shape | N | Rows of 128 | Columns of 128 | Blocks | Blocks / 96 CU | Specialized split when partial |
|---|---:|---:|---:|---:|---:|---:|
| gate/up (`M=17408,K=5120`) | 5909 | 136 | 47 | 6,392 | 66.58 | 6,256 interior + 136 tail |
| gate/up | 8192 | 136 | 64 | 8,704 | 90.67 | exact; no tail |
| down/add (`M=5120,K=17408`) | 5909 | 40 | 47 | 1,880 | 19.58 | 1,840 interior + 40 tail |
| down/add | 8192 | 40 | 64 | 2,560 | 26.67 | exact; no tail |

## gfx1100 microbenchmarks

Means below are over four fresh-process medians. Every process asserted gfx1100 and compared the candidate result against shipped output before timing.

| Shape | N | Candidate | Shipped TOPS | Candidate TOPS | Ratio | Shipped ceiling | Candidate ceiling |
|---|---:|---|---:|---:|---:|---:|---:|
| gate/up SET | 5909 | grid specialization + LF16 | 102.514 | 124.614 | **1.2156x** | 40.47% | **49.20%** |
| gate/up SET | 8192 | LF16 | 117.875 | 131.924 | **1.1192x** | 46.54% | **52.08%** |
| down ADD | 5909 | grid specialization + LF16 | 102.041 | 122.616 | **1.2016x** | 40.29% | **48.41%** |
| down ADD | 8192 | LF16 | 121.480 | 130.098 | **1.0709x** | 47.96% | **51.36%** |

Raw F/R/R/F medians (TOPS):

- gate/up 5909: shipped `[99.255, 105.648, 105.914, 99.238]`; candidate `[129.191, 119.771, 120.251, 129.244]`.
- gate/up 8192: shipped `[112.810, 122.606, 122.079, 114.003]`; LF16 `[137.152, 126.919, 126.240, 137.384]`.
- down 5909: shipped `[98.901, 105.342, 105.110, 98.812]`; candidate `[127.625, 118.172, 118.014, 126.653]`.
- down 8192: shipped `[117.375, 125.752, 126.439, 116.354]`; LF16 `[134.752, 124.915, 125.188, 135.535]`.

No measured cell regressed. LF16 ADD is beneficial on gfx1100 even though it is harmful on gfx1151, which is why the ADD selector is architecture-specific.

### 2x2 composition at gate/up N=5909

Each row is its own paired four-process comparison; independent references differ slightly with clock drift.

| Shape knob | Grid specialization | Paired shipped mean | Candidate mean | Ratio |
|---|---|---:|---:|---:|
| off | off | 102.779 | 102.779 | 1.0000x |
| off | on | 102.779 | 110.340 | **1.0736x** |
| on | off | 103.019 | 102.964 | 0.9995x |
| on | on | 102.514 | 124.614 | **1.2156x** |

Shape-only is intentionally neutral at partial N: without the split, the guarded partial grid stays on the shipped body. The exact-N LF16 factor (1.1192x) times the standalone grid factor (1.0736x) predicts 1.2016x; the measured combined result is 1.2156x. The two optimizations therefore compose rather than cannibalize one another.

## End-to-end gfx1100

Production graph-forward ABBA used A = both knobs off and B = both knobs on.

| Metric | A mean | B mean | B/A |
|---|---:|---:|---:|
| Matrix PP512 tok/s | 1,623.65 | 1,756.50 | **1.0818x** |
| Matrix PP8192 tok/s | 1,638.25 | 1,751.10 | **1.0689x** |
| TTFT, 5,909-token prompt | 3,671.149 ms | 3,325.307 ms | **0.9058x** (9.42% lower) |
| TTFT-derived prefill | 1,609.578 tok/s | 1,776.979 tok/s | **1.1040x** |
| Decode, ctx128/tg128 | 49.064 tok/s | 48.927 tok/s | 0.9972x |

The matrix medians were A1/A2 = `1636.8/1610.5` at PP512 and `1647.5/1629.0` at PP8192; B1/B2 = `1759.8/1753.2` and `1752.8/1749.4`. TTFT medians were A1/A2 = `3670.155/3672.142 ms` and B1/B2 = `3325.223/3325.390 ms`.

## gfx1151 non-regression and architecture split

The production check fixed grid specialization **true in both arms** and changed only `kernel.gfx11_iu4_shape`. Both arms reported gfx1151. The selector requires `arch == gfx1100`, so the candidate flag is a code-path no-op on Halo.

- Honest 5,909-token TTFT ABBA: 9,041.274 ms flag-off versus 9,030.708 ms flag-on (0.9988x time); derived prefill 653.559 versus 654.323 tok/s (1.0012x). This is neutral.
- Matrix PP512 raw aggregate: 712.35 versus 702.85 tok/s (0.9867x).
- Matrix PP8192 raw aggregate: 660.85 versus 650.75 tok/s (0.9847x).

The matrix aggregate is dominated by the first A process being high (`724.4/673.3`); the remaining B1/B2/A2 processes cluster at `707.2, 698.5, 700.3` for PP512 and `652.9, 648.6, 648.4` for PP8192. Because both arms resolve to identical kernels and the stable partial-N TTFT comparison is +0.12%, the matrix excursion is process drift, not an attributable regression.

A direct LF16-vs-ordinary diagnostic on gfx1151 also explains the per-architecture policy. Exact gate/up SET was 48.126 versus 48.717 TOPS (1.0123x), while exact down ADD was 48.015 versus 45.270 TOPS (0.9428x). Halo therefore retains shipped SET-only LF16 and ordinary ADD; gfx1100 enables both behind its own knob.

## Correctness and self-audit

All 16 final four-cell microbench comparisons were bit-exact. ADD comparisons used a nonzero 0.25 residual and returned zero mismatches over 30,254,080 elements at N=5909 and 41,943,040 at N=8192. SET returned zero over 102,863,872 and 142,606,336 elements. The grid specialization had already been established bit-exact across 64 fresh processes.

A separate c24 quality run was not useful after this proof: c24 is deterministic, while the candidate preserves the exact accumulation and fold order and produced identical output bits for every exercised element, including nonzero-residual ADD. There is no changed tensor for c24 to score.

### Loader and addressing audit

For each full 128x128 workgroup and K=256 group:

- A unique bytes: `128 * 136 = 17,408` (16,384-byte payload plus 1,024-byte headers).
- X unique bytes: two K=128 halves, each `128 * 72 = 9,216`, totaling 18,432.
- Total unique operand input: 35,840 bytes/group. Gate/up K=5120 uses 20 groups = 716,800 bytes/workgroup; down K=17408 uses 68 groups = 2,437,120 bytes/workgroup.
- Output traffic: SET writes 65,536 bytes/tile; ADD reads and writes 131,072 bytes/tile.

The LF16 loader issues exactly 4,096 payload dwords plus 256 header dwords for A and exactly 2,304 dwords per X half. No operand is redundantly fetched. All 512 lanes participate in A payload, X loads, compute, and stores; waves 0-7 alone fetch the 256 headers once.

Output coverage is one-to-one over all 16,384 tile elements: wave `w` chooses row `16*(w/2)` and column `64*(w%2)`; `c=0..3`, `l=0..7`, and lane `0..31` map to `i = wave_row + 2*l + lane/16`, `j = wave_col + 16*c + lane%16`. The full domain is covered without overlap. K groups remain increasing, each half remains ordered, the eight-WMMA chain and low/high pairing remain unchanged, and the same fold operation is used. The graph wrapper changes only which block-index component denotes row versus column.

## Battery and harness evidence

The stock `serve_harness.py` creates an isolated `HIPFIRE_HOME`; it therefore does **not** carry typed experimental knobs set in the caller's normal config. A stock-harness statement that a knob was on can actually have exercised the knob's default route. This was discovered when validating this change.

The valid workaround was to start one manual server per typed-config arm, with the release daemon and graph-forward enabled, then run the required battery through `serve_harness.py --mode battery ... --thinking off --no-spawn`. Both correctly routed arms completed 5/5 with `runaway=0`, `empty=0`, `attractor=0`, and `retrieval_miss=0`.

Decoded-text prefixes captured verbatim by the harness:

- Shipped/off: code `Merge two already-sorted lists into a single sorted...`; reason `Step 1: First leg distance...`; factual `The seasons are caused by Earth's axial tilt of about 23.5 degrees...`; prose `Every dawn, the keeper climbed the sea-slicked rocks...`; instruct `Use clear, descriptive names for variables, functions, and modules...`.
- Candidate/on: code `Merge two already-sorted lists into one sorted list...`; reason `First part: 60 mph times 2.5 hours = 150 miles...`; factual `The seasons are caused by Earth’s axial tilt of about 23.5 degrees...`; prose `Elias climbed the fog-wrapped rocks at dawn...`; instruct `Use clear, descriptive names for variables, functions, and modules...`.

Sampling was intentionally unseeded, so wording differs while all five answers are coherent and complete. Earlier battery-based knob claims in this campaign need the same typed-config/no-spawn treatment before they are cited as routing evidence. This includes the earlier grid-specialization battery; its correctness case instead rests on bit-identical outputs and independently proven interior/tail dispatch counts.

## Verification performed

- `cargo build --release` completed after adding both eager and graph/replay wrappers.
- `cargo test -p hipfire-config --lib`: 75 passed.
- Final code-object metadata: LF16 96 VGPR, 26 SGPR, zero scratch/spills; ordinary 189 VGPR, 22 SGPR, zero scratch/spills.
- Clean gfx1100 microbenchmark ABBA for all four required cells and the 2x2 composition.
- Graph-forward production matrix and 5,909-token TTFT ABBA on gfx1100.
- Graph-forward production matrix and 5,909-token TTFT ABBA on gfx1151 with grid specialization fixed true.
- One correctly routed, typed-config battery per gfx1100 arm through `--no-spawn`.
