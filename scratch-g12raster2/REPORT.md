# G12Raster2: exact gfx1201 IU4 GEMM raster / staging / epilogue screen (PLAN.md §9 item 2)

## Verdict: KEEP (banded raster + wide epilogue); KILL deeper prefetch

- **What's kept.** Two changes, both default-on behind one kill switch, `HIPFIRE_G12_RASTER=0`:
  - a banded CTA raster: bands of 8 row tiles, rows fastest within a band, then tokens;
  - a wide full-tile epilogue: one [32 tok][132] f32 LDS tile, written out as 512-byte b128 token columns.
- **Exactness.** Outputs are bit-identical: every output bit matches on the full shapes plus M/N tails and edge corpora, on the symmetric and asymmetric routes and on the F1 SiLU entry. The OFF build is instruction-identical to the incumbent.
- **pp8192 rocprof, same binary off→on (card-E).**

  | Base | IU4 GEMM ms | GPU total ms | Traced prefill tok/s |
  |---|---:|---:|---:|
  | `4bd33ce40` (assigned base) | 1,608.4 → 1,550.8 (**−3.58%**) | −59.8 (−2.68%) | 3,695 → 3,798 (+2.77%) |
  | `865fd3d03` (current official, includes F1 and attention v2) | 1,640.3 → 1,585.4 (**−3.35%**) | −55.5 (−2.62%) | 3,891 → 3,996 (+2.69%) |

  Every shape is faster; no shape regresses.
- **WT2 c24 fp8 (card-D), lever on.** KLD 0.083278, kldseq md5 `9d0e860f41db992820ebdc9483c0a041` on both bases. That is byte-identical to the baseline.
- **Killed.**
  - Depth-2 register prefetch: −6.4% weighted.
  - Pure token-first raster: −9.5% weighted; FA-Q −26%, ADD −18%.
  - Direct no-LDS b128 epilogue: −0.5%.
  - ADD old-Y prefetch epilogue: loses 5–7 points on ADD K6144.
- **Commits.**
  - `4bd33ce40` base, branch `g12-raster2`: production **`e15d581d9`**; this evidence directory is committed on top of it.
  - `865fd3d03` base, branch `g12-raster2-865` (worktree `wt-g12raster2-865`): production port **`97bd7c6b7`**; the same evidence commit is cherry-picked on top.
  - The clean-room tag count (`git log -p <base>..HEAD`) is 0 on both branches.

## 1. Setup

- **Kernel.** `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip`: tile 128 rows × 128 tokens, 8 waves, LDS 20,480 B, 182 VGPR.
  - Occupancy is 3 CTAs/WGP, so about 96 CTAs are resident on 32 WGPs. **LDS, not VGPRs, is the limiter.**
  - Dispatch is in `gemm.rs::gemm_mq4g256v2_mmq_prequant_iu4`, gfx1201 branch.
- **Harness** (`host.cpp`, `build.sh`).
  - Device TUs are the runtime concatenation: symfold defines, then `block_i4_128_quant.hip`, then the kernel body. They are compiled with the runtime flags `hipcc --genco --offload-arch=gfx1201 -O3 --no-offload-compress` and loaded with `hipModuleLoad`.
  - Control = the unmodified `4bd33ce40` kernel (`base_kernel.hip`), or the `865fd3d03` kernel (`k865.hip`).
  - The host asserts one visible device, gfx1201, PCI c3:00.0 (card-E).
- **Inputs.** Non-periodic: a splitmix64 counter hash per byte, nibble word and scale.
  - fp16 scales span 2^-14..2^-3 with random sign; zp = −8·sc.
  - Token d spans 2^-9..2^3, with s = the exact signed nibble sum.
  - The residual is random with a random sign.
- **Oracle.** Every arm is compared with the control on **every output bit**.
  - SET runs into a NaN-prefilled Y; it must overwrite everything, and the control is checked to leave no stale NaN.
  - ADD runs from the non-trivial residual.
  - **Tail/edge corpus** (13 cases, SET+ADD):
    - M ∈ {48, 1000, 1002 (M%4=2, wide path off), 1024, 1920, 2944, 5120, 17024 (133-tile partial band), 17408};
    - N ∈ {1, 300, 643, 777, 1000, 4096, 4113, 4173, 4739, 8155, 8192};
    - K ∈ {256…17408};
    - edge rows: ±0 scales, all −8, all +7, all-zero codes, one-hot K ∈ {0, 127, 128, 255}; edge tokens: ±extremes, d = 0, cancellation.
  - **Full shapes** (N = 8192 and N = 4096 in every timing process): gate/up 17408, LA QKV 10240, LA Z 6144, β/α 48, FA Q 12288, FA K/V 1024, ADD K6144, ADD K17408. The `865fd3d03` runs also check the F1 SiLU 17408, plus 4 SiLU tail cases.
  - Asymmetric (non-symfold) route: full-shape and tail parity as well.
- **Timing.**
  - Arms are interleaved per round, with the order reversed on alternate rounds; 15 rounds; ≥1.5 s interleaved warmup.
  - Reported figures are the median ms and the median per-round paired speedup against the control.
  - Processes run in F and R shape order.
  - Weighted = Σ (pp8192 calls × ms). Calls: gate/up 128, down 64, QKV 48, Z 48, β/α 96, FA Q 16, FA K/V 32, out 64. On `865fd3d03` the 128 gate/up SETs become 64 SiLU calls.
  - **"cold" mode:** before every timed launch, a 512 MB scrub, then the ADD residual restore, then an X touch. This mimics production, where W arrives cold and X hot.
  - Clocks were 2.2–2.35 GHz under the board power limit.
- **Production protocol.**
  - rocprof: `run_trace.py`, the G12Port runner with only the paths changed. It runs the pp8192 fixture (md5 `8e588120…`) after a different-prefix warm request, on fp8 VMM, with `HIPFIRE_GRAPH=1` and a private HOME/kernel cache.
  - WT2: `run_wt2.py`.
  - Binaries: `bin/` (`4bd33ce40` + lever: hipfire md5 `36655083…`, daemon `d398cf5a…`, eval `dfdbe39e…`) and `bin865/` (hipfire `43694708…`, daemon `c1dc6c54…`, eval `4086180e…`). Both are git-ignored.

## 2. Screen: what was tried

Paired weighted saving is shown as % of control weighted time; positive = faster. Source logs are `logs/t1..t8`, `summary-*.txt`.

| Arm | Change | Weighted | Worst shape | Result |
|---|---|---:|---:|---|
| `r1g1` | token-fastest raster | **−9.45%** | FA Q −25.8% | kill |
| `r1g4` / `r1g8` / `r1g16` / `r1g32` | row band G | +1.75 / +1.9 / +1.7 / +0.2% | ≥ −1.3% | band 4–16 is good |
| `r2g4` / `r2g8` / `r2g16` | token band G, tokens fastest | −0.2 / +0.05 / −0.2% | FA Q −5.7% | kill |
| `epi1` | [32][132] LDS tile, b128 512-B columns, 8 barriers | +1.5 – +1.9% | ≥ −0.6% | keep; ADD K6144 +4–5% |
| `epi2` | direct b128 from the C layout, no LDS/barriers | −0.5 / −0.05% | gate −1.6% | kill (64-B segments) |
| `epi3` | epi1 + all old-Y ADD loads hoisted | +1.6% | ADD K6144 −1.7% when combined | kill |
| `pf2` | depth-2 register prefetch (next two slabs in flight, 188 VGPR) | **−6.4%** | −6.7% | kill |
| `r1g8e1p2` | band 8 + epi1 + pf2 | −3.2% | −7.8% | kill |
| `r1g{2,4,6,8,16}e1` | band + epi1 | +2.6 / +3.1–3.3 / +3.2–3.5 / +3.4–3.5 / +3.3–3.55% | ≥ −1.0% | **band 8 + epi1 chosen** |

The raster effect is W-working-set locality. With row-fastest order, about 96 CTAs cover roughly one token tile and 96 different W row tiles (about 33 MB). The full W (47 MB for gate/up) is therefore re-streamed from MALL for each of the 64 token tiles. A band of 8 row tiles × ~12 token tiles holds about 2.8 + 4.4 MB, which fits in L2. Token-first order does the opposite: it re-streams X (24–80 MB) per row tile and loses badly. That matches PLAN §4: token-first is **not** a universal win.

## 3. Chosen arm, standalone per-shape paired speedup (%) and weighted saving

`r1g8e1` is the screen body; `prod_g12r` is the production source built with `IU4_G12_RASTER`. They are instruction-equivalent.

| Run | gate/up SET | QKV | Z | β/α | FA Q | FA K/V | ADD K6144 | ADD K17408 | **Weighted** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t2 F N8192 warm | +3.86 | +3.62 | +3.28 | −0.16 | +2.95 | +0.89 | +5.89 | +3.42 | **+3.51%** |
| t3 F N8192 warm | +3.66 | +3.08 | +2.21 | −0.28 | +2.97 | +1.85 | +5.79 | +3.14 | **+3.40%** |
| t3 R N8192 warm | +3.23 | +3.60 | +3.37 | −0.07 | +2.67 | −0.08 | +6.42 | +2.93 | **+3.48%** |
| t4 F N8192 warm | +3.34 | +3.34 | +2.85 | −0.22 | +2.99 | +1.26 | +6.82 | +3.20 | **+3.60%** |
| t4 F N8192 cold | +4.69 | +4.48 | +3.00 | −0.88 | +4.53 | +0.16 | +5.68 | +3.22 | **+3.93%** |
| t4 R N8192 cold | +5.25 | +3.92 | +2.70 | −1.00 | +4.53 | +0.28 | +6.03 | +2.29 | **+4.21%** |
| t4 R N4096 warm | +4.55 | +2.51 | +0.45 | −0.19 | +2.09 | +1.32 | +7.29 | +3.73 | **+3.72%** |
| t5 F N512 warm | +1.84 | +2.51 | +2.40 | −0.39 | +1.96 | +1.26 | +8.49 | +10.39 | **+4.04%** |
| t5 R N2048 warm | +1.65 | +2.70 | +3.05 | −0.29 | +1.94 | +1.07 | +7.21 | +3.99 | **+2.75%** |
| t6 F N4096 cold (prod) | +5.16 | +4.49 | +2.89 | +0.00 | +5.07 | +0.34 | +6.98 | +2.93 | **+4.21%** |
| t6 F N8192 warm (prod) | +2.55 | +3.20 | +2.96 | +0.64 | +2.45 | +1.08 | +5.69 | +3.40 | **+2.88%** |
| t6 R N8192 cold (prod) | +4.45 | +3.83 | +2.24 | −0.74 | +3.78 | −0.11 | +5.50 | +2.82 | **+3.72%** |

- **Noise.** The identity arms (`var_r0`, `prod_off`), whose ISA equals the control's, read −0.4% to +0.3% weighted. Single shapes range up to ±1.6%.
- **The one sub-3% warm process** (t6 F warm, 2.88%) sits within that noise. Cold and production numbers are higher.
- **β/α M48** is a single partial row tile, so the lever cannot touch it (raster no-op, wide path off). Its −1.0…+0.6% readings are noise on a 53 µs kernel.
- **No shape exceeds −1.0% in any process.**

## 4. Production, `4bd33ce40` (branch `g12-raster2`)

- **Change.** In `IU4_G12_RASTER` builds only:
  - the raster remap at the top of the body;
  - the wide full-tile epilogue: `M % 4 == 0`, 16 B-aligned Y, and anything else takes the incumbent stores;
  - `IU4_RASTER_BAND` = 8.
- **Wiring.**
  - `kernels.rs` adds `GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_{,SYMFOLD_}G12R_SRC`, with `_g12r` / `_symfold_g12r` entry names.
  - `gemm.rs` selects them on gfx1201 unless `HIPFIRE_G12_RASTER=0`.
  - Grid, block, LDS and the ABI are unchanged. The capture/replay grid is unchanged; the kernel name is fixed per process.
- **Resources.** symfold g12r: 185 VGPR (control 182); asymmetric: 205 (control 202). Both have 0 spills, 0 private bytes and 3 CTAs/WGP.
- **OFF build.** Without the macro, the kernel ISA is identical to the pre-change file (only the cuid symbol differs), on both routes.
- **rocprof pp8192** (card-E, same binary): off1 → on1 → on2 → off2.
  - 3,729 dispatches each.
  - GPU ms: 2,222.8 / 2,168.0 / 2,172.2 / 2,236.9.
  - Prefill tok/s: 3,707.0 / 3,801.5 / 3,793.5 / 3,683.5.
  - ON runs contained only `_g12r` IU4 symbols; OFF runs only incumbent ones.
  - Serve log: `KV cache: Fp8 vmm`, fp8.

| Shape (calls) | Off ms | On ms | Δ ms | Δ% |
|---|---:|---:|---:|---:|
| gate/up SET M17408 (128) | 739.435 | 713.798 | −25.637 | −3.47 |
| down ADD K17408 (64) | 382.719 | 369.426 | −13.293 | −3.47 |
| LA QKV SET M10240 (48) | 166.549 | 161.477 | −5.073 | −3.05 |
| out ADD K6144 (64) | 135.737 | 127.692 | −8.044 | **−5.93** |
| LA Z SET M6144 (48) | 99.987 | 96.601 | −3.386 | −3.39 |
| FA Q SET M12288 (16) | 66.834 | 64.901 | −1.933 | −2.89 |
| FA K/V SET M1024 (32) | 11.410 | 11.203 | −0.207 | −1.81 |
| LA β/α SET M48 (96) | 5.747 | 5.663 | −0.085 | −1.47 |
| **IU4 total** | **1,608.419** | **1,550.762** | **−57.657** | **−3.58** |
| Other kernels | 621.436 | 619.332 | −2.104 | −0.34 |
| GPU total | 2,229.857 | 2,170.094 | −59.763 | −2.68 |

The production traces agree with the standalone cold numbers: per shape −1.5 to −5.9%, weighted −3.58%.

WT2 c24: `run_wt2.py g12r-on bin D` gives KLD **0.083278** and kldseq md5 **`9d0e860f41db992820ebdc9483c0a041`** (sha256 `1dd3caed…58711`). The evaluator compiled and used only the `_symfold_g12r` IU4 module.

## 5. Port onto the current official `865fd3d03` (branch `g12-raster2-865`)

The official `mq4-lloyd` moved during this task to `865fd3d03`, which includes F1-lite. F1 rewrites the same kernel file into an `EPI` template and adds a SiLU gate/up entry with its own two-group raster; this conflicts with the `4bd33ce40` patch. The port is therefore a separate commit, and on it `IU4_G12_RASTER` also:

- gives the SiLU entry the same band-8 raster, replacing `IU4_SILU_RGROUPS`;
- gives the SiLU entry a wide h epilogue: a [32 tok][68] tile and 256-byte b128 h columns.

Everything else is identical to §4. The OFF build is ISA-identical to `865fd3d03`. Parity against the `865fd3d03` control passes on the full corpus, the full shapes and the SiLU cases, on both routes.

**Standalone** (`logs/t7`, `t8`). Arms: `p865_g12r` = raster only on SiLU; `p865_g12r2` = raster plus wide h, the shipped variant.

| Run | SiLU | QKV | Z | β/α | FA Q | FA K/V | ADD K6144 | ADD K17408 | Weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t8 F N8192 warm, g12r2 | +2.94 | +2.74 | +3.09 | +0.00 | +2.55 | +1.06 | +5.77 | +2.70 | +2.98% |
| t8 R N8192 cold, g12r2 | +3.74 | +3.71 | +2.14 | −1.01 | +4.10 | +0.79 | +5.20 | +2.46 | +3.14% |
| t8 F N4096 cold, g12r2 | +3.80 | +4.41 | +2.41 | −0.39 | +4.88 | +0.50 | +6.75 | +3.88 | +4.05% |

**rocprof pp8192** (card-E, `bin865`, same binary): off1 → on1 → on2 → off2.

- 3,633 dispatches each.
- GPU ms: 2,108.4 / 2,056.6 / 2,063.3 / 2,122.5.
- Prefill tok/s: 3,905.6 / 4,003.0 / 3,988.7 / 3,877.0.

| Shape (calls) | Off ms | On ms | Δ% |
|---|---:|---:|---:|
| gate/up SiLU (F1) M17408 (64) | 761.133 | 738.072 | −3.03 |
| down ADD K17408 (64) | 384.490 | 371.534 | −3.37 |
| LA QKV SET (48) | 168.832 | 163.712 | −3.03 |
| out ADD K6144 (64) | 138.971 | 130.775 | −5.90 |
| LA Z SET (48) | 101.760 | 98.353 | −3.35 |
| FA Q SET (16) | 67.705 | 65.758 | −2.88 |
| FA K/V SET (32) | 11.611 | 11.441 | −1.46 |
| β/α SET (96) | 5.828 | 5.744 | −1.44 |
| **IU4 total** | **1,640.329** | **1,585.388** | **−3.35** |
| GPU total | 2,115.451 | 2,059.944 | −2.62 |

WT2 c24 on `865fd3d03` + lever (`run_wt2.py g12r865-on bin865 D`): KLD **0.083278**, kldseq md5 **`9d0e860f41db992820ebdc9483c0a041`** (sha256 `1dd3caed…58711`). This is byte-identical; the evaluator used the `865fd3d03` `_symfold_g12r` module.

## 6. Notes and caveats

- **Compiler bug found.** hipcc/clang 23 miscompiles `__builtin_bit_cast(T, ext_vector[i])` for i > 0: it returns element 0.
  - Symptom: `__builtin_amdgcn_raw_buffer_load_b128` looked "replicated". Assembling a `u32x4` by element + `__builtin_bit_cast` also splatted element 0.
  - The harness oracle caught both (75% mismatches).
  - The shipped code uses plain `float4` global accesses and no ext-vector bit casts.
  - Existing production code only bit-casts scalars, so it is unaffected.
- **Deeper prefetch loses.** Depth-2 prefetch loses about 6% even though VGPR stays under the occupancy cliff (188 vs the 256 allowed at 3 CTAs/WGP). The extra in-flight loads cost more than the latency they hide. [INFERENCE: VMEM issue/return contention with the WMMA/fold stream; not profiled.]
- **Remaining headroom.** The measured gain is locality plus store shape; neither removes the exact-fold VALU cap (PLAN §5.1). This lever takes about 3.5% of the IU4 time. The GEMM stays at about 37% of the 687-TOPS ceiling.
- **Clocks.** They were not locked, and the power limit applies. Standalone speedups are paired within a process; rocprof pairs are off/on/on/off in the same binary on one card. Per the lever protocol, no whole-model ABBA was run.
- **Integration.**
  - Because `865fd3d03` rewrote the kernel, `g12-raster2` (on `4bd33ce40`) does not apply cleanly on top of it. `g12-raster2-865` is the landable form.
  - `HIPFIRE_G12_RASTER=0` restores the incumbent modules exactly: the OFF ISA is identical.

## 7. Files

- **Harness:**
  - `build.sh`, `host.cpp`
  - `iu4_var.hip`: the screen body, with `IU4_RASTER` / `IU4_GROUP` / `IU4_EPI` / `IU4_PF2` knobs
  - `base_kernel.hip` (`4bd33ce40` kernel), `k865.hip` (`865fd3d03` kernel)
  - `run-screen.sh`, `analyze.py`, `summarize_arm.py`
- **Production runners:** `run_trace.py`, `run_wt2.py`, `trace_compare.py`, `trace_shapes.py`.
- **Logs:**
  - `logs/parity-*.txt`: tail/edge corpus, all arms, PARITY_SUMMARY fails=0.
  - `logs/fullparity-*.txt`.
  - `logs/t1..t8-*.txt`: timing with full-shape parity headers.
  - `logs/summary-*.txt`.
- **Traces:** `trace/<label>/{summary.json,symbols.tsv,serve.log,env.txt,gpu-pre/post.txt,*-response.json}`, `trace/compare-{4bd,865}.txt`, `trace/shapes-{4bd,865}.txt`. The raw CSVs are git-ignored.
- **WT2:** `wt2/g12r-on.{kldseq,log,env}`, `wt2/g12r865-on.*`.
