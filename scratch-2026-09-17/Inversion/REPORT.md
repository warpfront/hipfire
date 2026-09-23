# Standalone-to-daemon inversion report

## Verdict

The two apparent inversions have different causes.

1. **Raster G8T0 was a synthetic launch-topology artifact.** The original rig timed one contiguous `M=34816` iu4 `full_set`, but the daemon emits two consecutive `M=17408` launches for gate and up. G8T0 removes a roughly 10% superlinear penalty of the synthetic combined launch; production already avoids that penalty by splitting the halves. At the real single-dispatch shape the win is only 1.32%, and in the complete daemon trace all `full_set` calls improve 0.82%. The missing 10.7% is not DPM, graph capture, or a neighbor effect.
2. **The current GDN scan win does transfer into the daemon kernel.** Standalone is 156.44 -> 84.32 us (-46.10%) under the ATT run; the daemon trace is 171.09 -> 93.02 us average (-45.63%). The scan's 44.96 ms request saving is diluted to a 30.92 ms (-1.54%) change in summed kernel time by unrelated 0.5-2.5% arm drift in other kernels. There is no occupancy overlap with a neighbor: all 4,148 dispatches are serialized. The current dedup is nevertheless a **KILL for correctness**: real-data c2 changes from 0.060029 to 0.063152 and the output md5 changes. Synthetic identity did not cover that state.

The corrected rig rule is: **a kernel rig must replay one actual runtime dispatch, including its exact `(M,K,N)`, launch decomposition, output/weight buffers, and immediately adjacent dispatches. Never concatenate logically related gate/up matrices merely to match their total mathematical width.** A 10% result from the combined `M=34816` proxy is not evidence for either of the daemon's `M=17408` launches.

## Setup and controls

- Base: `390d8228f`.
- Branch/worktree: `gfx1201-inversion`, `wt-inversion`.
- Device: Card-A by UUID only, `GPU-9eb7aeda51c88ffd`; home `ab0`.
- Fixed environment: `HOME=/home/kaden/.hipfire-homes/ab0`, `ROCR_VISIBLE_DEVICES=GPU-9eb7aeda51c88ffd`, `HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels`, `HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models`, `HIPFIRE_GRAPH=1`.
- Workload: `qwen3.8:27b-mq4-xt`, prompt md5 `ed720348b81a19fab64d4783c75c1ae3`, 5,909 tokens, one TTFT request per arm after a separate warm request.
- Instrumentation toggles: `HIPFIRE_GDN_DEDUP=0/1`, `HIPFIRE_RASTER_G8T0=0/1`.
- `rocprofv3 --kernel-trace --stats` captured the full dispatch timeline. ATT targeted one exact kernel occurrence. A profiling-only graceful shutdown control was added because hard-killing the profiler child prevented trace finalization.
- `sample-amdsmi.py` used the same AMD SMI library as `amd-smi` at 100 ms because the CLI's watch interval accepts only whole seconds.

No product commit was made.

## Raster G8T0

### Shape isolation

Each rig row is an internal median of 20 after five in-process warmups; A/B/A rows use the mean of A1 and A2. The first process after idle was discarded.

| One `full_set` launch | Baseline | G8T0 | Delta |
|---|---:|---:|---:|
| Original synthetic rig, `M=34816,K=5120,N=4096` | 6,656.790 us | 5,965.744 us | **-10.38%** |
| Token-count isolate, `M=34816,K=5120,N=5909` | 9,762.951 us | 8,748.879 us | **-10.39%** |
| Daemon-shape rig, `M=17408,K=5120,N=5909` | 4,443.839 us | 4,385.304 us | **-1.32%** |

Changing only the token count preserves the 10.4% result, so the 4,096-vs-5,909 token difference is not the cause. Halving the output-row extent to the daemon's actual launch shape removes nine tenths of the win.

The decisive arithmetic is visible without a model:

- Combined-shape baseline at `N=5909`: 9,762.951 us.
- Two split-shape baselines: `2 * 4,443.839 = 8,887.677 us`, already 8.96% faster than the synthetic combined launch.
- Combined-shape G8T0: 8,748.879 us, essentially the same regime as two split launches.

Thus G8T0 mostly repairs a penalty created by the proxy's synthetic concatenation. In the daemon, dispatches 501 and 502 are the two real halves. Their traced grid is 34,816 work-items / 256 threads = 136 workgroups in X, corresponding to 17,408 output rows at the 128-row tile. The ATT prologue also proves the candidate is live: 406 sampled waves take the grouped-remap path and zero hit the direct-map fallback.

Raw rig rows are in `raw/raster-rig-results.csv`; the exact binaries are `raw/raster-{shape,token}-{base,g8t0}.bin`.

### In-daemon kernel trace

| Kernel | Calls | Baseline total | G8T0 total | Delta |
|---|---:|---:|---:|---:|
| `...iu4_full_set` | 368 | 959.557 ms | 951.700 ms | -0.819% |
| `...iu4_full_add` | 128 | 458.013 ms | 456.147 ms | -0.407% |
| packet FA control | 192 | 136.409 ms | 136.932 ms | +0.383% |
| GDN scan control | 576 | 98.545 ms | 98.618 ms | +0.073% |
| all kernels | 4,148 | 2,014.063 ms | 2,005.634 ms | -0.419% |

For the dominant `full_set` production shape (grid X 34,816, grid Y 47, 128 calls), baseline is 647.407 ms and G8T0 is 641.715 ms (-0.879%). Other `full_set` shapes range from -0.08% to -0.85%. Resource metadata is unchanged: 208 VGPR, zero scratch, 256-thread workgroup.

One profiled request reported TTFT 2,093.371 ms baseline and 2,078.376 ms G8T0 (-0.72%). This is a one-row diagnostic, not a ship measurement; it agrees with the 8.43 ms reduction in summed kernel time and cannot support the original 10.7% projection.

### ATT

In-situ dispatch 501 was sampled in both arms. The decoder reported an incomplete final wave, so use category proportions and the kernel timeline rather than absolute ATT event counts.

| ATT quantity | Baseline | G8T0 |
|---|---:|---:|
| sampled latency events | 80,119,517 | 78,907,683 |
| sampled stall events | 45,907,639 | 44,435,938 |
| stall / latency | 57.30% | 56.31% |
| waitcnt stalls | 21,883,642 | 19,849,669 |
| VALU stalls | 13,302,346 | 13,497,919 |
| WMMA stalls | 6,799,267 | 7,038,247 |
| LDS stalls | 3,458,910 | 3,605,767 |

This is a small redistribution, not the 10.7% original-rig signature. It matches the per-dispatch trace (dispatch 501 is 4,851.603 vs 4,853.651 us, effectively flat) and the full 128-call shape total (-0.879%).

Shape-matched standalone ATT selected one SIMD and replayed `M=17408,K=5120,N=5909`. Its event timer is 4,464.790 vs 4,393.627 us (-1.59%), consistent with the unprofiled -1.32% A/B/A result:

| Shape-matched standalone ATT quantity | Baseline | G8T0 |
|---|---:|---:|
| sampled latency events | 852,633,847 | 840,134,547 |
| sampled stall events | 482,359,119 | 468,851,145 |
| stall / latency | 56.57% | 55.81% |
| waitcnt stalls | 197,921,438 | 177,989,319 |

The standalone and daemon stall fractions are nearly the same at the same launch shape. In both, G8T0 trims waitcnt stalls but shifts some sampled stalls to VALU/WMMA, yielding only a 1-2% kernel result. The old 10.7% number requires the synthetic combined output extent.

### Clock/power control

Active request segment, sampled at 100 ms and selected by socket power >=100 W:

| Arm | Samples | current GFX mean (range) | current UCLK mean | socket power mean (range) |
|---|---:|---:|---:|---:|
| baseline | 21 | 2,440.8 MHz (2,390-2,548) | 1,258 MHz | 294.7 W (150-333) |
| G8T0 | 21 | 2,450.3 MHz (2,402-2,627) | 1,258 MHz | 295.4 W (168-336) |

GFX differs by +0.39%, UCLK is identical, and power differs by +0.24%. DPM does not explain either the original rig win or its loss in production.

### Host-side fix assessment

The only plausible host lever is to concatenate each gate/up weight and output pair and issue one `M=34816` launch. The shape isolation says that baseline fusion would first introduce the 8.96% combined-launch penalty, after which G8T0 removes it. Compared with today's two split launches, combined G8T0 is only about 1.56% faster for that pair. The daemon trace already bounds the realized total request benefit to 7.86 ms in `full_set` plus 1.87 ms in `full_add`, under 0.5% of summed kernel time. That is below the +1.5% pp8192 ship gate and does not justify a host/output-layout rewrite. **No host fix is recommended.**

## GDN scan dedup timing specimen

### Standalone vs daemon

| Measurement | Baseline | Dedup | Delta |
|---|---:|---:|---:|
| standalone 512-row event under ATT run | 156.442 us | 84.320 us | -46.10% |
| daemon, 576 calls, average | 171.086 us | 93.023 us | -45.63% |
| daemon, 576 calls, total | 98.545 ms | 53.581 ms | -45.63% |

The daemon candidate has a 512-thread workgroup, 192 VGPR in trace metadata (compiler resource report: 190), 60,928 B LDS, and zero scratch. Baseline has 256 threads, 248 VGPR in trace metadata, the same LDS, and zero scratch.

The scan therefore does not lose its kernel win in situ. It saves 44.964 ms. Summed request kernel time moves only 2,014.063 -> 1,983.140 ms (-1.54%) because other kernels drift the other way:

| Kernel | Baseline total | Dedup-arm total | Delta |
|---|---:|---:|---:|
| GDN scan | 98.545 ms | 53.581 ms | -45.628% |
| iu4 `full_set` | 959.557 ms | 967.194 ms | +0.796% |
| iu4 `full_add` | 458.013 ms | 460.374 ms | +0.516% |
| gated norm | 57.468 ms | 58.925 ms | +2.535% |
| fused SiLU/mul | 115.794 ms | 117.225 ms | +1.236% |
| fused RMSNorm | 77.624 ms | 78.314 ms | +0.889% |
| packet FA control | 136.409 ms | 136.653 ms | +0.179% |

The 5,909-token single-row diagnostic reported 2,093.371 -> 2,001.047 ms TTFT (-4.41%). As above, this is diagnostic only. The scan's trace-derived end-to-end ceiling is 44.964 / 2,014.063 = 2.23% before any unrelated drift, so standalone percentages must not be applied directly to matrix or TTFT.

### Timeline and neighbors

All 4,148 dispatches are serialized in all arms; overlap count is zero. Median dispatch gap is 3.48 us baseline and 3.44 us dedup. For the 528 scans followed immediately by `gdn_chunk_kkt_solve`, the post-scan gap is 3.455 vs 3.366 us and solve duration is 21.286 vs 21.762 us. The terminal scan in each of 48 linear-attention layers is followed by gated norm; that successor is 1,197.24 vs 1,227.60 us (+2.54%), accounting for 1.46 ms, far smaller than the scan saving. Nothing is co-resident with the 60,928-B-LDS scan block, so there is no cross-kernel occupancy theft mechanism.

### ATT, in situ and standalone

| Sample | stall / latency | waitcnt stalls | total stalls | sampled hits |
|---|---:|---:|---:|---:|
| daemon baseline | 73.48% | 9,489,181 | 10,300,649 | 1,015,053 |
| daemon dedup | 55.52% | 4,537,907 | 6,480,642 | 759,456 |
| standalone baseline | 69.57% | 768,700 | 857,785 | 104,948 |
| standalone dedup | 56.54% | 513,994 | 730,209 | 93,644 |

ATT sampling volume differs between binaries, so absolute count ratios are not timing ratios. The stable result is the direction and stall mix: wait-dominated baseline drops from 92.1% to 70.0% of in-daemon stalls, while the measured kernel duration drops 45.6%. Standalone shows the same stall-fraction transition. The daemon is not injecting a new stall class.

### Clock/power control

| Arm | Samples | current GFX mean (range) | current UCLK mean | socket power mean (range) |
|---|---:|---:|---:|---:|
| baseline | 21 | 2,440.8 MHz (2,390-2,548) | 1,258 MHz | 294.7 W (150-333) |
| dedup | 20 | 2,423.1 MHz (2,367-2,579) | 1,258 MHz | 302.9 W (283-361) |

The dedup arm is 0.73% lower in mean GFX yet substantially faster; UCLK is identical. DPM cannot be the source of the scan win or an inversion.

### Correctness overrides timing

The copied current candidate passes the synthetic identity harness but is not bit-identical on the required real-data c2 path: OFF/ON KLD is 0.060029/0.063152 and output md5 differs. The timing result is therefore a mechanism specimen only. **Do not stage or ship this GDN body despite the real in-daemon speedup.** The source bug must be found before performance gates have meaning.

## Corrected measurement rule

For future kernel gates:

1. Extract exact kernel arguments and grid/workgroup dimensions from a daemon trace.
2. Replay one runtime dispatch, not a mathematically combined proxy; preserve the runtime's split/fused topology.
3. Use the same input/state semantics. Synthetic byte identity is insufficient for stateful GDN.
4. Run ATT on one named daemon dispatch and compare its stall mix to a shape-matched standalone dispatch.
5. Compute the request ceiling from `kernel_saved_ms / baseline_total_kernel_ms` before matrix runs.
6. Check clocks/power only as a control. Do not infer DPM from an end-to-end row when per-kernel trace already shows the win or loss.
7. Require real-data correctness before performance ship gates.

## Artifacts

- Kernel traces/stats: `raw/{base,gdn-dedup,raster-g8t0}-trace/`.
- 100 ms metrics: `raw/{base,gdn-dedup,raster-g8t0}-metrics.csv`.
- GDN ATT: `raw/gdn-{base,dedup}-att/` and `raw/gdn-standalone-{base,dedup}-att/`.
- Raster in-daemon ATT: `raw/raster-{base,g8t0}-att/`.
- Raster shape-matched standalone ATT: `raw/raster-shape-{base,g8t0}-att/`.
- Rig rows and exact replay binaries: `raw/raster-rig-results.csv`, `raw/raster-{shape,token}-{base,g8t0}.bin`.
- Profiling wrappers: `profile-daemon.sh`, `sample-amdsmi.py`.

ATT raw captures that reached the hardware buffer report an incomplete final wave; generated stats CSVs are retained and this limitation is reflected above.
