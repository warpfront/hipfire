# GEMM v2 V2B production integration: gfx1151 (Halo)

Branch `gfx11-v2b-halo`, based on `mq4-lloyd` at `ef52914a4` (source tree = `8a2f42ef8` + the gfx1100 V2C LDS-constant fix; for gfx1151 it dispatches exactly like `84d2992a`/land-042). hipx worktree `/home/kaden/hipfire-v2bhalo`. Model: `/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq` (current MQ4V2 artifact; no file-format change, no weight sidecar, no repack).

**Decision:** on exact gfx1151, route symmetric MQ4V2 IU4 full-tile SET and ADD prefill GEMMs through a production-layout port of GemmV2Gfx11's `v2b_ms` whenever the M256×N256 grid has at least two dispatch rounds (`(M/256)·(N/256) ≥ CU count`, 40 on Halo); everything else keeps X5. Every Halo gate passes: SET+ADD GEMM time in the pp8192 trace drops **7,338 → 5,696 ms (−22.4%)**, fresh-process ABBA ×2 against mq4-lloyd gives **pp8192 +18.71% (835.5 → 991.8 tok/s)** and pp512 +17.78% with decode −0.02%, and WT2 c24 is byte-identical at 0.076901.

## Kernel: `kernels/src/gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip`

Compute is `v2b_ms` unchanged (frozen `scratch-gemmv2/gemmv2.hip`, `-DV2_TN=256 -DV2_MAGIC=1 -DV2_SCF32=1`):
- M256×N256 CTA, 16 waves (block 32×16), wave tile 64 rows × 64 tokens (4 row × 4 token fragments), 128 FP32 sums per lane;
- two 32 KiB K128 LDS slots (65,536 B), one CTA barrier per epoch, 1 CTA/WGP;
- A-row lane permutation, MAGIC integer chains (`0x4B400000`), SCF32 fold `sum = fma(RN(sc·d), float(C), sum)` per ascending K128 half, ADD epilogue `RN(oldY + sum)` with EG=2 residual grouping.

Operands are staged directly from the production layouts, following GemmV2Integ's V2C port (`gemm_mq4g256v2_residual_iu4_v2c.gfx11.hip`):
- **Weights:** MQ4G256V2 136-byte row groups; rebiased with `^0x88888888` while staging (symmetric `zp == −8·sc` contract).
- **Activations:** `block_i4_128` 72-byte blocks.
- **Permutation at the LDS store:** staging lane (hi, rr) of wave w owns row/token `16w+rr` and K16 slices `2i+hi`, so every b64 store fills one contiguous 256-byte LDS block (bank-conflict free) and each WMMA fragment is one 128-byte LDS run. With 16 waves this covers the 256-row/256-token tile with 4+4 b64 loads per thread per epoch.
- **W scales bypass LDS:** a wave owns 64 rows, so each lane holds two scale words (row fragments 0–1 and 2–3) and the fold gathers the eight rows it needs with DPP `row_share`. One addition over the V2C scheme, found in the variant screen below: the lane loads the whole 8-byte group header `[sc0 zp0 sc1 zp1]` once per group (even epoch) and keeps `sc0|sc1` packed in one word, instead of one `u16` load per epoch. Exact (same FP16 bits), −8 VGPRs, +1–2.6% on SET.
- **X scales `d`:** read straight from `Xq` by the consuming lanes.

ABI `(const char* A, const char* Xq, float* Y, int M, int K, int N)`; symbols `gemm_mq4g256v2_residual_iu4_v2b_{set,add}_gfx11`; grid `(N/256, M/256)`, block `[32,16,1]`, dynamic LDS 65,536 B. The TU also builds for `__gfx1100__` but is dispatched only on gfx1151.

## Dispatch (`crates/rdna-compute/src/gemm.rs`, `gemm_mq4g256v2_mmq_prequant_iu4`)

A new block right after V2C's selects V2B when all of these hold:
- symfold active, full M128/N128 tiles, eager column route (not capture/replay), positive dims — the X5 eligibility rule;
- `arch == "gfx1151"`;
- `M % 256 == 0 && N % 256 == 0`;
- `(M/256)·(N/256) ≥ cu_count_or_default()` (40 CUs = 2×20 WGPs on the 8060S: at least two rounds of the 1-CTA/WGP grid);
- `HIPFIRE_IU4_V2B` is not `0`.

`HIPFIRE_IU4_V2B=0` falls back to X5 (and X5's own fallbacks). `kernels.rs` registers `GEMM_MQ4G256V2_RESIDUAL_IU4_V2B_GFX11_SRC` as a separate module; no existing translation unit, kernel, or compiler policy changed. For the pp8192 bench every traced SET/ADD shape takes V2B; at pp512 all but SET M1024 (8 CTAs) do.

## Standalone (Halo, before integration)

Host `standalone/gemmv2-host.cpp` (GemmV2Integ's host with a `v2b_prod` arm, `V2BP_EXTRA` variant arms and ad-hoc `set|add:M:K:N` shapes), `standalone/run_standalone.sh`, `standalone/summ.py`. Every phase is a fresh process with `ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0`, the arch asserted from `hipGetDeviceProperties` (`gfx1151`, 20 MPs), rocm-smi pre/post (`logs/S*-smi-*.txt`).

Objects (`logs/S*-objects.sha256`):
- X5 control: the object hipfire mq4-lloyd (land-042 code) JIT-compiled on Halo, `…_x5_symfold.c4e902d85a7eb0e1.hsaco` sha `566abd34…` (same source hash as the X5 trace's object);
- native `v2b-ms-gfx1151.hsaco` `e7453b2b…` (GemmV2Gfx11 frozen build);
- production layout: S1/S2 first port `6f2937dd…`, S3 final `d01bed4f…` (the runtime-JIT object's ISA is identical to it, see Resources).

### Oracle (bit-exact vs X5)

`scratch-xreuse/oracle_inputs.hpp` (sha `e9e9a0c1…`), full corpus: hashed M256/N256 at K ∈ {256, 512, 5120, 6144, 17408}; UnequalHalfWitness K256/512; signed extrema K512; scale edges K1024; one-hot K ∈ {0,127,128,255,256,511} at K512 and K5119 at K5120; inter-CTA grids M512/K1024/N768 and M768/K1536/N512. SET and ADD (ADD from nonzero residual bytes restored before every arm).

| Run | Arms | Outputs compared | Differing |
|---|---|---:|---:|
| S1 (`logs/S1-oracle.log`) | v2b_ms, v2b_prod (first port) | 7,340,032 | **0** |
| S2 (`logs/S2-oracle.log`) | + base, hdr, dx, hdrdx variants | 22,020,096 | **0** |
| S3 (`logs/S3-oracle.log`) | v2b_ms, v2b_prod (final), hdr | 11,010,048 | **0** |

X5 vs the CPU reference of the same DAG: 0 differing in every CPU-checked case.

### Timing, final object (S3; mean of F and R process medians, median of 7 events, ≥0.6 s warm-up per arm; % of the 105.543 TOPS Halo reference)

| shape | M | K | N | v2b CTAs | x5 ms (% of 105.543) | v2b_ms ms (% of 105.543) | v2b_prod ms (% of 105.543) | X5 / v2b_prod | v2b_prod / v2b_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| set_m17408 | 17408 | 5120 | 8192 | 2176 | 26.279 (52.6%) | 18.784 (73.7%) | 19.778 (70.0%) | 1.329x | 1.053x |
| set_m12288 | 12288 | 5120 | 8192 | 1536 | 17.781 (54.9%) | 13.290 (73.5%) | 13.964 (69.9%) | 1.273x | 1.051x |
| set_m10240 | 10240 | 5120 | 8192 | 1280 | 15.662 (52.0%) | 11.076 (73.5%) | 11.717 (69.5%) | 1.337x | 1.058x |
| set_m6144 | 6144 | 5120 | 8192 | 768 | 9.357 (52.2%) | 6.753 (72.3%) | 7.184 (68.0%) | 1.302x | 1.064x |
| set_m1024 | 1024 | 5120 | 8192 | 128 | 1.709 (47.6%) | 1.311 (62.1%) | 1.431 (56.9%) | 1.195x | 1.091x |
| add_k6144 | 5120 | 6144 | 8192 | 640 | 12.207 (40.0%) | 8.912 (54.8%) | 9.689 (50.4%) | 1.260x | 1.087x |
| add_k17408 | 5120 | 17408 | 8192 | 640 | 26.243 (52.7%) | 20.112 (68.8%) | 21.303 (64.9%) | 1.232x | 1.059x |
| set:1024:5120:4096 | 1024 | 5120 | 4096 | 64 | 0.772 (52.7%) | 0.684 (59.5%) | 0.711 (57.2%) | 1.085x | 1.040x |
| set:1024:5120:2048 | 1024 | 5120 | 2048 | 32 | 0.402 (50.6%) | 0.339 (60.0%) | 0.355 (57.2%) | 1.132x | 1.048x |
| set:1024:5120:1024 | 1024 | 5120 | 1024 | 16 | 0.199 (51.1%) | 0.172 (59.3%) | 0.178 (57.2%) | 1.119x | 1.037x |
| set:6144:5120:768 | 6144 | 5120 | 768 | 72 | 0.864 (53.0%) | 0.701 (65.3%) | 0.739 (62.0%) | 1.170x | 1.054x |
| set:1024:5120:768 | 1024 | 5120 | 768 | 12 | 0.152 (50.1%) | 0.169 (45.1%) | 0.176 (43.5%) | 0.868x | 1.039x |
| add:5120:6144:768 | 5120 | 6144 | 768 | 60 | 0.862 (53.1%) | 0.663 (69.1%) | 0.697 (65.7%) | 1.238x | 1.051x |
| set_m17408_n512 | 17408 | 5120 | 512 | 136 | 1.566 (55.2%) | 1.256 (68.9%) | 1.328 (65.1%) | 1.179x | 1.058x |
| set_m12288_n512 | 12288 | 5120 | 512 | 96 | 1.111 (55.0%) | 0.893 (68.4%) | 0.947 (64.5%) | 1.173x | 1.061x |
| set_m10240_n512 | 10240 | 5120 | 512 | 80 | 0.951 (53.5%) | 0.723 (70.4%) | 0.755 (67.4%) | 1.260x | 1.045x |
| set_m6144_n512 | 6144 | 5120 | 512 | 48 | 0.570 (53.6%) | 0.507 (60.2%) | 0.528 (57.8%) | 1.079x | 1.041x |
| set_m1024_n512 | 1024 | 5120 | 512 | 8 | 0.116 (43.9%) | 0.167 (30.5%) | 0.173 (29.3%) | 0.669x | 1.039x |
| add_k6144_n512 | 5120 | 6144 | 512 | 40 | 0.594 (51.3%) | 0.436 (70.0%) | 0.459 (66.5%) | 1.295x | 1.052x |
| add_k17408_n512 | 5120 | 17408 | 512 | 40 | 1.695 (51.0%) | 1.215 (71.1%) | 1.267 (68.3%) | 1.338x | 1.042x |
| set:17408:5120:256 | 17408 | 5120 | 256 | 68 | 0.858 (50.4%) | 0.698 (62.0%) | 0.736 (58.8%) | 1.165x | 1.054x |
| set:12288:5120:256 | 12288 | 5120 | 256 | 48 | 0.644 (47.4%) | 0.514 (59.3%) | 0.538 (56.7%) | 1.197x | 1.045x |
| set:10240:5120:256 | 10240 | 5120 | 256 | 40 | 0.551 (46.2%) | 0.363 (70.0%) | 0.379 (67.1%) | 1.453x | 1.044x |
| set:6144:5120:256 | 6144 | 5120 | 256 | 24 | 0.311 (49.0%) | 0.331 (46.1%) | 0.344 (44.3%) | 0.904x | 1.040x |
| set:1024:5120:256 | 1024 | 5120 | 256 | 4 | 0.076 (33.3%) | 0.165 (15.4%) | 0.172 (14.8%) | 0.444x | 1.039x |
| add:5120:6144:256 | 5120 | 6144 | 256 | 20 | 0.267 (57.2%) | 0.224 (68.2%) | 0.234 (65.2%) | 1.141x | 1.046x |
| add:5120:17408:256 | 5120 | 17408 | 256 | 20 | 0.806 (53.6%) | 0.617 (70.1%) | 0.655 (66.0%) | 1.230x | 1.062x |

- N8192 rows are exactly the seven SET/ADD shapes of the Halo pp8192 X5 trace (N=8192 single chunk); N512 rows are the same GEMMs at pp512; the rest is a CTA-count sweep for the routing threshold.
- **Every grid ≥ 40 CTAs is faster than X5 (1.079–1.453×).** Grids of 4, 8, 12 and 24 CTAs lose (0.44–0.90×): a 1-CTA/WGP 256² tile quantizes badly into Halo's 20 WGPs, whereas X5's 128² tiles fill partial rounds. 16, 20 and 32 CTAs win but are excluded by the conservative ≥ 2-round rule (none occurs in the pp512/pp8192 gates; they are N ≤ 1024 edge shapes).
- **Prod vs native:** the production layout keeps 92–95% of native `v2b_ms` speed on SET/ADD K17408 (`v2b_prod / v2b_ms` 1.05–1.06×) and 92% on ADD K6144 (1.087×), in line with GemmV2Integ's 6–7% on XTX V2C (b64 vs b128 staging, scattered staging addresses, 72-byte-strided `d`).

### Variant screen (S2, same protocol; ratio = first-port prod ms / variant ms)

| N8192 shape | X5 ms | v2b_ms ms | first-port prod ms | base | hdr | dx | hdr+dx |
|---|---:|---:|---:|---:|---:|---:|---:|
| set_m17408 | 26.115 | 18.335 | 19.703 | 19.594 (1.006x) | 19.324 (1.020x) | 20.115 (0.980x) | 19.889 (0.991x) |
| set_m12288 | 17.722 | 13.006 | 14.068 | 14.027 (1.003x) | 13.713 (1.026x) | 14.151 (0.994x) | 13.955 (1.008x) |
| set_m10240 | 15.492 | 11.105 | 11.947 | 11.997 (0.996x) | 11.698 (1.021x) | 12.105 (0.987x) | 11.764 (1.016x) |
| set_m6144 | 9.381 | 6.784 | 7.311 | 7.305 (1.001x) | 7.193 (1.016x) | 7.481 (0.977x) | 7.380 (0.991x) |
| set_m1024 | 1.710 | 1.288 | 1.428 | 1.431 (0.998x) | 1.413 (1.011x) | 1.468 (0.973x) | 1.461 (0.978x) |
| add_k17408 | 26.344 | 20.071 | 21.072 | 21.236 (0.992x) | 21.049 (1.001x) | 21.704 (0.971x) | 21.570 (0.977x) |
| add_k6144 | 12.159 | 8.886 | 9.610 | 9.508 (1.011x) | 9.631 (0.998x) | 9.793 (0.981x) | 9.787 (0.982x) |

- `base` = the scratch variant TU with all switches off (noise reference, ±1%).
- `hdr` (b64 group header once per group, packed `sc0|sc1`): +1.1–2.6% on SET, neutral on ADD, 246 VGPR → **adopted** into the production kernel.
- `dx` (two `d` loads + `v_permlanex16` + selects instead of four `d` loads): −2–3% everywhere; rejected (matches V2C's `v2cp6` result on XTX).
- A paired-epoch unroll (compile-time slot parity, needed for a constant-parity `hdr`) spilled (256 VGPR, 26–28 spills) and was dropped; the adopted `hdr` uses a uniform runtime-parity branch.

## Resources: runtime-compiled production HSACO

Object `trace/gfx1151-8192/home/.hipfire_kernels/gfx1151/gemm_mq4g256v2_residual_iu4_v2b_gfx11.2d3e1a3c10c6bf7b.hsaco` (sha `4d361efd…`), JIT-compiled by the branch's `hipfire` with `--genco --offload-arch=gfx1151 -O3 --no-offload-compress`. Probe `resource.cpp` (`hipFuncGetAttribute` + `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`, 512 threads, 65,536 B) and ELF notes (`gfx1151-resources.txt`):

| Resource | SET | ADD |
|---|---:|---:|
| VGPRs | 246 | 246 |
| Private bytes | 0 | 0 |
| VGPR/SGPR spills (ELF) | 0 / 0 | 0 / 0 |
| Dynamic LDS | 65,536 B | 65,536 B |
| CTAs per MP (waves) | 1 (16) | 1 (16) |

Disassembly is identical (0 diff lines) to the standalone-timed S3 object `d01bed4f…`. rocprof reports `VGPR_Count` 248 (allocation granularity) and `Scratch_Size` 0 for every V2B dispatch.

## Gates (Halo gfx1151, ROCR=1 HIP=0, knob-free, VMM q8)

### 1. rocprof pp8192 trace

`run_profile.py` + `inventory.py`: warm request (different first words), then an uncached traced 8,192-token request; asserted `GPU dev 0: gfx1151`, `KV cache: Q8 vmm (`, ROCR=1/HIP=0 on server and daemon (`trace/gfx1151-8192/gpu-process-env.txt`); daemon stopped with `hipfire stop <port>` before the session ended; CSV flushed. The traced request ran 07:23:37–07:23:45 UTC with both hipx cards otherwise idle (peers' later gfx1100 load/JIT at ~07:26 overlapped only the untimed WT2).

| Kernel | Calls | ms | X5 (hipfire-x5 trace) calls / ms |
|---|---:|---:|---:|
| V2B SET M17408 (grid 32×68) | 128 | 2,560.05 | 128 / 3,355.67 |
| V2B SET M12288 | 16 | 224.95 | 16 / 284.30 |
| V2B SET M10240 | 48 | 556.32 | 48 / 738.68 |
| V2B SET M6144 | 48 | 348.20 | 48 / 445.63 |
| V2B SET M1024 | 32 | 44.83 | 32 / 53.79 |
| V2B ADD M5120 (K17408 + K6144) | 128 | 1,961.86 | 128 / 2,460.37 |
| **SET + ADD** | 400 | **5,696.20** | 400 / **7,338.44** |

- **Saved 1,642.2 ms (−22.38%)** of SET+ADD against X5's 7,338 ms; every X5 call site now dispatches V2B (identical call counts; kernel count 4,849 in both traces).
- Trace totals: kernel sum 8,106.1 ms, request wall 8,154.3 ms (X5 reference trace, older pre-Q16/RoPE stack: 9,989.9 / 10,042.2 ms).

### 2. Fresh-process ABBA ×2

`run_abba.py` against byte copies of the mq4-lloyd (land-042 code) release binaries, pinned in `baseline-mq4lloyd/` (`hipfire` `f1ae6d2e…`, `daemon` `a4f83b5c…`, identical to `/home/kaden/hipfire-land042/target/release`; for gfx1151 `ef52914a4` dispatches identically). B = this branch (`hipfire` `7a0c3428…`, `daemon` `b993ec50…`). Fresh HOME per run, `HIPFIRE_*` scrubbed, ROCR=1/HIP=0 attested from `/proc/<pid>/environ`, `GPU dev 0: gfx1151` and `KV cache: Q8 vmm (` asserted, JSON `kv_mode=q8`/`kv_backend=vmm`/`gpu.arch=gfx1151` checked. Both hipx cards were kept quiet by hub agreement for the whole run (07:33–07:48 UTC; rocm-smi pre-check showed only `gpusentry`).

`hipfire bench <model> --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode q8 --kv-backend vmm --json`

| # | Arm | pp512 | pp8192 | tg128@128 |
|---:|---|---:|---:|---:|
| 1 | A mq4-lloyd | 991.9 | 852.8 | 14.857 |
| 2 | B V2B | 1145.8 | 1000.6 | 14.854 |
| 3 | B V2B | 1137.0 | 991.0 | 14.848 |
| 4 | A mq4-lloyd | 955.9 | 830.1 | 14.856 |
| 5 | A mq4-lloyd | 953.0 | 829.0 | 14.859 |
| 6 | B V2B | 1130.8 | 989.1 | 14.861 |
| 7 | B V2B | 1127.9 | 986.5 | 14.851 |
| 8 | A mq4-lloyd | 955.1 | 829.9 | 14.855 |

| Row | mq4-lloyd mean | V2B mean | Pooled | Per cycle |
|---|---:|---:|---:|---|
| pp512 | 963.98 | 1135.38 | **1.17781×** | 1.17199, 1.18374 |
| pp8192 | 835.45 | **991.80** | **1.18714×** | 1.18343, 1.19091 |
| tg128 decode | 14.8565 | 14.8536 | 0.99980× (−0.02%) | 0.99965, 0.99995 |

**PASS** (ship bar): pp8192 and pp512 are both ≥ +1.5%, no row is more than 1% slower, decode is within 1%.

**Absolute pp8192 on Halo: 991.8 tok/s** (mean of the four V2B processes; range 986.5–1000.6; the first B process hit 1,000.6 and the later ones drifted ~1% lower, as did A after its first run). That is +19.9% over the 827.2 recorded for mq4-lloyd in `scratch-land042` and **0.8% short of the 1,000 target**. Raw data: `gfx1151/abba/summary.json`, per-run `*.json`, `*.log`, `*.env`, `*.command.json`, and `gfx1151-abba.out`.

### 3. Quality

**WT2** (`run_wt2.py`, this tree's `eval_hipfire` `2f6bb341…`): `--kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks 24`, ROCR=1/HIP=0, `HIPFIRE_GRAPH=0`. Slice-mean KLD **0.076901** (pinned Halo value), and the KLD sequence is **byte-identical** to mq4-lloyd's recorded `scratch-land042/gfx1151/wt2/B.kldseq`, SHA256 **`eefd7d257d736b9bdaddf1246e93217c710eb4291dfec3cafd7f62c863aaeb25`** (`gfx1151/wt2/run.out`, `B.log`, `B.env`).

**Battery** (`run_battery.py`): manual `hipfire serve` on port 11978 with a fresh HOME; asserted `GPU dev 0: gfx1151`, `KV cache: Q8 vmm (`, and ROCR=1/HIP=0 on the server and its daemon (`gfx1151/serve-env.txt`). Harness `scripts/serve_harness.py --no-spawn --mode battery --max-think-tokens 1 --max-tokens 512`: **5/5** `finish=stop` (code, reasoning, factual, prose, instruct); runaway 0, empty 0, attractor 0, retrieval-miss 0 (`gfx1151/battery.json`, `battery.log`). The first turn's 11.2 s prefill is the fresh-HOME kernel JIT.

**TTFT-511** (`odd_fill.py`, same server): 511 prompt tokens, 0 cached, `finish=stop`, 132 completion tokens, and a coherent answer that notices the truncated message and discusses the `merge_sorted` function (`gfx1151/ttft511.json`). N=511 is not a multiple of 128/256, so it exercises the partial-N fallback route, not V2B.

### 4. Kill switch

`killswitch.sh` (`gfx1151-killswitch.out`, `gfx1151/killswitch/bench.log`): `HIPFIRE_IU4_V2B=0`, fresh HOME, one uncontrolled matrix run. The fresh kernel cache holds `gemm_mq4g256v2_residual_mmq_iu4_gfx11_x5_symfold.c4e902d85a7eb0e1.hsaco` (the mq4-lloyd X5 source hash) and no V2B object; pp512 944.1 and pp8192 838.0 tok/s, i.e. the mq4-lloyd level.

GPU post-check at 07:50:59 UTC: `rocm-smi --showpids` shows only `gpusentry`; no process of this worktree remains.

## Build and provenance

- Release binaries built **locally** on hiptrx (same Ubuntu 26.04.1 / rustc / `/opt/rocm` HIP 7.15.26333 toolchain as hipx) from the committed tree: `cargo build --release` and `cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`; copied to `/home/kaden/hipfire-v2bhalo/target/release` (`hipfire` `7a0c3428…`, `daemon` `b993ec50…`, `eval_hipfire` `2f6bb341…`). The hipx worktree was only ever updated by `git fetch`/`checkout` of pushed commits (clean status before every run). No compile ran on hipx; kernels JIT on first use.
- Standalone objects and the host were built locally with `hipcc --genco --offload-arch=gfx1151 -O3 --no-offload-compress` (the runtime flags) and `hipcc -O2`.
- Every Halo GPU phase was announced on hub with pre/post `rocm-smi` checks; the trace and ABBA ran in agreed both-card quiet windows.
- Clean-room check: the assignment's case-insensitive `git log -p ef52914a4..HEAD` grep for the excluded token returns 0.

## Not done / next

- **Prod vs native gap.** The production layout still leaves 5–6% (SET, ADD K17408) and 8.7% (ADD K6144) against native `v2b_ms`. Closing it needs a native-layout X producer (the quantizer writing NativeIu4SoA1) and/or a W sidecar; both are file/producer changes outside this assignment. The two exact staging levers screened here beyond `hdr` (`dx`, paired epochs) lost.
- **Small grids.** 16/20/32-CTA grids win (1.12–1.23×) but are excluded by the ≥ 2-round rule; a finer rule is possible but untested on real prompts.
- **1,000 tok/s:** pp8192 is 991.8 mean (1,000.6 best process). The remaining ~0.8% has to come from outside SET/ADD (e.g. the beta/alpha fold or epilogue fusions) or from the native-layout gap above.


