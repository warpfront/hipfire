# gfx11 FA2 Q8: 16-query tile production gates

## Provenance and current status

Worktree `/home/kaden/hipfire-fa2q16`, branch `gfx11-fa2-q16`, based on landed `e5f944a3a`. The 16-query Q8 kernel is bit-exact against the incumbent at the real 16-layer H24/KV4/D256, 8192-query causal shape. It passes the standalone ≥5% screen on gfx1151 and gfx1100. **Production verdict:** gfx1100 passes the +1.5% prefill ship gate (+1.992% pp8192), all other rows stay within −1%; gfx1151 is **stackable-only** under Main's explicit exception (FA2 trace −19.78%, production pp8192 +0.576%, pp512 −0.698%, decode +0.037%). Quality and serve battery pass on both. The earlier whole-chunk Halo ABBA below was *void* because it did not exercise shipped default segmentation; it is not used in this verdict.

The source modifies `kernels/src/attention_q8_0_fa2_gqa.gfx11.hip` and `crates/rdna-compute/src/attention.rs`: exact gfx1100/gfx1151 use Q16, grid `ceil(batch/16)×4`, block 256, KT32, 32,768-B dynamic LDS; other gfx11 targets and fwht3-K retain Q8, grid `ceil(batch/8)×4`, block 128. Six compute waves hold the same per-wave 16-query accumulator; eight waves cooperate on K/V fills. All final gates below use the **rebuilt guarded source**, knob-free defaults and explicit Q8/VMM. Baseline `/home/kaden/hipfire-prof040/target/release/hipfire` came from unchanged `e5f944a3a`, SHA256 `0c10c3a6e1954b86e37c21ade0ea3822fdf63291df3ac3d4d6e1da56450e85f2`. Final candidate SHA256: CLI `644976c2b296b193d535b0ae0deb01800015613f90f4dbeee2bda3206efdb2f5`, daemon `5d33f55d9e732022d749d00f8876e70096cf5c463aff1d5bd6100b0a251fcbc4`, evaluator `163d27bb3b79e30f5f2b3987dbc87223821e5fce8210c76615eefae108f39658`. Native `cargo build --release` and targeted `cargo build --release -p hipfire-runtime --example eval_hipfire` both exited zero after the guard.

## Standalone screen and numerical oracle

Disposable source `scratch-fa2q16/screen.hip` uses 16 separate deterministic nonuniform Q8 K/V layer planes, contiguous causal positions 0–8191, H24/KV4/D256, and the same f16 Q for both kernels. It compares **all 805,306,368 f32 outputs** across 16 layers (50,331,648 elements per layer); both cards report zero unequal words, zero nonfinite elements, maximum absolute error zero. Each timed pass launches exactly 16 layer bodies, with incumbent grid 1024×4/block128 versus candidate grid 512×4/block256. GPU-event timings below are ms/pass, chronological A-B-B-A twice within one process (A=incumbent, B=Q16). Fresh-process production numbers must not be inferred from these isolated kernel figures.

| Card | A1 B1 B2 A2; A1 B1 B2 A2 (ms) | Uncontended A/B mean | Saving |
|---|---|---:|---:|
| gfx1151 | 914.904, 704.796, 704.830, 912.746; 916.155, 715.725, 712.562, 915.149 | 914.738 / 709.478 | 22.44% |
| gfx1100 | 420.298, 342.260, 336.545, 418.571; 419.619, 340.814, 343.320, **882.261** | 419.496 / 340.735 (exclude the single 882.261-ms A2 outlier) | 18.78% |

An earlier independent clean ABBA screen gave gfx1151 A/B 912.699/704.645 ms (22.80%), gfx1100 418.895/343.353 ms (18.03%), also bit-exact over one complete layer. The profiled landed-production comparison in `../hipfire-prof040/scratch-halonongemm/REPORT.md` was 1,034.634 ms Halo / 434.727 ms XTX, with *different production call partition* (256 Halo / 32 XTX); do not equate these with this 16-launch microbenchmark.

HIP ISA metadata for the Q16 body (`llvm-readelf --notes` on extracted gfx11 code objects): on **both** gfx1151 and gfx1100 `.vgpr_count=218`, `.sgpr_count=22`, `.vgpr_spill_count=0`, `.sgpr_spill_count=0`, `.private_segment_fixed_size=0`, `.group_segment_fixed_size=0`; dynamic LDS 32,768 bytes. The incumbent gfx1151 body uses 224 VGPR, 28 SGPR, no spills/private scratch. `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(..., 32768)` returned **2 resident blocks/CU on both cards** for Q16/block256 and the incumbent/block128. The production JIT cache for the initially built candidate also recorded 218 VGPR/22 SGPR, zero spills/private. The final guarded source is codegen-equivalent on the two selected cards; the production trace confirms block256 on Halo.

## Non-shipping Halo whole-chunk ABBA (void per Main)

Runner `scratch-fa2q16/run_matrix.py` used the exact protocol command `hipfire bench <model> --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode q8 --kv-backend vmm --json`, fresh processes A1/B1/B2/A2 and the *same* `kernel.gfx11_q8_fa2_wide=true` in all four arms. `ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0` were captured from each process `/proc/<pid>/environ`; every corresponding process log contains `GPU dev 0: gfx1151`, `KV cache: Q8 vmm (`, and `prefill_chunk: requested=8192 admitted=8192 commit_stride=512`. JSON, combined logs, environment and pre/post PID captures are `scratch-fa2q16/gfx1151-{A1,B1,B2,A2}.{json,log,env,pre-pids,post-pids}`.

| Halo process | pp512 median tok/s | pp8192 median tok/s | tg128@128 median tok/s |
|---|---:|---:|---:|
| A1 landed | 812.3 | 770.7 | 14.858479 |
| B1 Q16 | 808.7 | 761.8 | 14.856490 |
| B2 Q16 | 794.8 | 757.5 | 14.860726 |
| A2 landed | 792.9 | 741.9 | 14.862103 |
| A mean | 802.60 | 756.30 | 14.860291 |
| B mean | 801.75 | 759.65 | 14.858608 |
| B versus A | −0.106% | **+0.443%** | −0.011% |

The +0.443% whole-chunk Halo result is below the +1.5% gain threshold, but this optional route is not Halo's default and Main voided it as the shipping gate. First runner launch aborted before any measurement because it read `/proc/PID/environ` between `fork` and `exec`; the runner was corrected and the four complete fresh-process arms above are the valid measurements. No additional benchmark cells were silently discarded. Pre/post GPU inventory shows only persistent `gpusentry` PID 2970 after the gate.

## Knob-free default Halo profiler

`scratch-fa2q16/trace_default.py` runs the original Prof040 `rocprofv3 --kernel-trace` routine on the rebuilt candidate and the identical baseline prompt (`benchmarks/prompts/pp8192.txt` symlinked temporarily to the Prof040 original), one uncached 8192-token request with one completion after a distinct-prefix warm request. Server log asserts `GPU dev 0: gfx1151` and `KV cache: Q8 vmm (`; process environment has `ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0`; response reported prompt=8192, cached=0 and wall=10,739.681 ms. Symbol `attention_q8_0_fa2_gqa_gfx11` ran **256 calls, 829.973074 ms total**, grid `(8192,4)` workitems/block256. Landed default Prof040 was the same 256 calls, **1,034.634 ms**; candidate FA2 time decreased **19.78%**. This is the actual segmented default path, not the void whole-chunk arm. Trace details: `scratch-fa2q16/trace-default/gfx1151-8192/fa2-summary.json`, `daemon_kernel_trace.csv`, `trace-response.json`, `serve.log`.

## Knob-free fresh-process Q8/VMM ABBA

`scratch-fa2q16/run_matrix.py` runs A1/B1/B2/A2, A=landed `e5f944a3a`, B=rebuilt guarded candidate, with `hipfire bench <model> --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode q8 --kv-backend vmm --json`. It does **not** set any kernel geometry flags. Each arm is a new process; its `/proc/<pid>/environ` attests `ROCR_VISIBLE_DEVICES=1` Halo or `0` XTX and `HIP_VISIBLE_DEVICES=0`; log asserts corresponding `GPU dev 0: gfx1151/gfx1100` and `KV cache: Q8 vmm (`. Per-process JSON, combined logs, environment and pre/post PID captures: `scratch-fa2q16/{gfx1151,gfx1100}-default-{A1,B1,B2,A2}.{json,log,env,pre-pids,post-pids}`. Rows below are median tok/s, higher is better.

| Card | Process | pp512 | pp8192 | tg128@128 |
|---|---|---:|---:|---:|
| gfx1151 | A1 | 814.6 | 759.8 | 14.851485 |
| gfx1151 | B1 | 803.8 | 753.6 | 14.862604 |
| gfx1151 | B2 | 790.7 | 749.3 | 14.857869 |
| gfx1151 | A2 | 791.1 | 734.5 | 14.857952 |
| gfx1151 | A mean | 802.85 | 747.15 | 14.854719 |
| gfx1151 | B mean | 797.25 | 751.45 | 14.860237 |
| gfx1151 | B/A | **−0.698%** | **+0.576%** | **+0.037%** |
| gfx1100 | A1 | 2150.5 | 2188.8 | 49.229990 |
| gfx1100 | B1 | 2134.9 | 2226.0 | 48.989668 |
| gfx1100 | B2 | 2125.8 | 2222.6 | 48.974571 |
| gfx1100 | A2 | 2120.9 | 2172.9 | 49.151218 |
| gfx1100 | A mean | 2135.70 | 2180.85 | 49.190604 |
| gfx1100 | B mean | 2130.35 | 2224.30 | 48.982119 |
| gfx1100 | B/A | **−0.251%** | **+1.992%** | **−0.424%** |

No average row is >1% slower, and no paired-process row regressed >1%. XTX passes the +1.5% pp8192 ship threshold. Halo misses +1.5% throughput but satisfies Main's **stackable exception**, supported by its default-route −19.78% FA2 trace, bit-exact full-layer oracle, and no >1% slow row. Do not report Halo as a +1.5% end-to-end win.

## WT2 c24 Q8/Q8 and manual serve battery

`scratch-fa2q16/run_wt2.py` used `/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin`, `--kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks 24`, one run per baseline/candidate arm, comparing the landed evaluator in `/home/kaden/hipfire-land040` against the rebuilt candidate. Both evaluated 24/24 chunks and 24,552 tokens. **gfx1100 A/B KLD 0.076879/0.076879; gfx1151 A/B 0.076901/0.076901.** Difference zero on each, below 0.0005 tolerance and absolute 0.10 cap. Logs and `.kldseq`: `scratch-fa2q16/{gfx1100,gfx1151}-wt2-{A,B}.*`.

Both cards ran a manually started candidate `hipfire serve --model <absolute-model> --kv-mode q8 --kv-backend vmm 127.0.0.1:11520`, each with an isolated HOME, `HIPFIRE_GRAPH=1`, `ROCR_VISIBLE_DEVICES=0`/`1`, `HIP_VISIBLE_DEVICES=0`; `/proc/<daemon-pid>/environ` attested both visibility vars and service log asserted the appropriate card plus `KV cache: Q8 vmm (`. `scripts/serve_harness.py --no-spawn --mode battery --thinking off --model <absolute-model> --kv q8 --kv-backend vmm --port 11520` reported **5/5 finish=stop, runaway=0, empty=0, attractor=0, retrieval_miss=0** on each card. Per-turn output: `scratch-fa2q16/gfx1100-battery.json` and `gfx1151-battery.json`. After graceful shutdown, `rocm-smi --showpids` listed only persistent `gpusentry` PID 2970.
