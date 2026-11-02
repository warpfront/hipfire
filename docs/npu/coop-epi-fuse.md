# Co-op epilogue fold: removing the gate_up / down NPU-column tail (Halo, IEF15 N0)

> **Historical research note: superseded by the 0.4.2 serving route.** This file records a killed co-op prototype (epilogue fold for the dense 27B gate_up/down NPU columns; verdict in section 6) from the 2026-10 NPU research cycle. It is not current guidance, and any figure in it is a dated fixture-bound measurement or an explicitly labelled historical estimate, not a product claim. The shipped 0.4.2 NPU route is the Flash-Next gfx1151 trunk input projection described in [npu-trunk.md](npu-trunk.md): `HIPFIRE_QWEN4_NPU_TRUNK=off|auto|on|emulate` (default `auto`, effectively `off` on every architecture except gfx1151), calibration schema 4, BF16-direct only. There is **no I32 serving contract** (no int32 partial, combine or fold exists in the product route). The out-projection program exists but is **not wired** into serving; `qsa.o` and the GDN o-projection are **not shipped**. Environment-variable names and flag syntax below are historical proposals or experiment switches, not supported configuration.

2026-10-03, research branch `npu/coop-epi-fuse` (on `npu/coop-down27` e6450c4; see [coop-down27.md](coop-down27.md)).
Model `qwen3.8-27b.mq4-xts` (sha256 `3e38ccba…f6f8ae`), L2 / L42 real weights. Inputs (md5): `L2.mlp.a4.bin` 4444070c…,
`L2.down.a4.bin` ed787a32…, `L42.down.a4.bin` 67d7d41e…. Binary `coop27` md5 6a64289b… (5e0ba76),
`ief_fold_gfx1151.co` md5 484200c1…. Windows: `tools/npu/steps/wf{1,2,3}.txt` (GPU only), `wc{1,2}.txt` (co-op). Logs are in
`logs/coopepi/` (copied from the measurement host). Labels: **M** measured, **E** estimate.

## 1. What was built (no hipfire edits; everything vendored into `crates/npu-tools/native/coop27/`)
* **The incumbent in PM text.** `v2b_pm_gfx1151.s` is `peacemaker-isa emit --kernel iu4_v2b --epi all --arch gfx1151` from
  the 0.4.1 hipfire beta checkout at 9aef0a35f. Peacemaker assembles it to an ELF byte-identical to the one inside the vendored
  `.hxaco` (M, `cmp`).
* **PM twins, the fold** (`native/build.sh`). `ief_gu_fold` and `ief_add_fold` are the incumbent gate_up SiLU and down ADD
  entries instruction for instruction, from the entry label to their end label. Their code bytes equal the incumbent's
  up to the final dealloc+endpgm (M, `cmp`). After their own epilogue stores, every wave runs `fold_sched.inc`, a
  work-claimed and flag-gated epilogue for the NPU columns:
  * the incumbent SiLU DAG verbatim, or `RN(x + Y)`, with the same C-tile and output index math as `ief_silu` /
    `ief_addc`;
  * one pass is one `ief_*` workgroup's worth (8 groups per lane, 32 b128 loads in flight);
  * claims of 16 wave-units go through an L2 atomic, pre-checked by a plain L2 read;
  * readiness: the host writes `flags[j] = round` (default-registered host page, clflush) after NPU command `j`
    completes. One wave per workgroup reads that uncached flag and mirrors it into the round's L2 slot; the rest scan
    the slot (one parallel 4-lane read);
  * the last 20 workgroups by linear id (dispatched last) drain: they wait until every command is ready and
    exhausted;
  * a poll cap writes a status word instead of hanging.

  Kernarg grows 44/40 → 160 B; the GEMM part is unchanged.
* **Merged wide tails** `ief_silu_wide` / `ief_addc_wide`: the same pass body in static mode, one launch for all J
  commands.
* `coop27 --mode fold` runs GPU-only checks and timing. `--mode coop --tail sep,wide,fold` measures the three tails ABBA
  in one process, against GPU-alone and the window. Every timed sample records the GPU temperature and sclk.

## 2. Correctness (M, every process)
* **GPU only (WF3).** Twin with no NPU work vs the full incumbent: 0 mismatches on its columns, NPU region untouched.
  * Flags published mid-kernel (gate_up at 8/16/25 ms; the incumbent's own g,u, chunks rotated across arenas each
    round): 0 mismatches.
  * Flags preset: 0 mismatches. Wide tails: 0. Slot census: every claim counter reached upc, status 0.
* **Co-op (WC1, WC2), first and last round of every tail.**
  * GPU columns are byte-identical to GPU-alone (0 of 117–134 M gate_up, 0 of 33.5 M down).
  * NPU C is byte-equal to the CPU IEF15 oracle.
  * NPU-column h equals the incumbent SiLU DAG of the oracle g,u (gate_up); x equals `RN(old + oracle Y)` (down).
  * D/b and the A stream match the CPU.
  * Fold slots are clean.
* **Error vs GPU-alone** is unchanged from coop-down27:

  | arm | pre-epilogue NRMSE | h NPU cols | all cols |
  |---|---:|---:|---:|
  | gate_up | 1.61e-5 | 1.39–1.45e-5 | 3.2e-6 (J=1), 4.5e-6 (J=2), 5.5e-6 (J=3) |
  | down L2 | 3.12e-5 | 4.37e-7 | 1.96e-7 |
  | down L42 | 2.24e-5 | 1.97e-6 | 8.79e-7 |

## 3. GPU-only fold cost (M, WF3, flags preset unless noted, ms)
| shape | window | +sep | +wide | fold | fold J=0 | tail sep / wide | fold, flags at 8/16/25 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate_up J=3 | 31.59 / 31.73 | 34.47 / 35.17 | 34.27 / 34.56 | 34.31 / 34.35 | 31.30 / 31.98 | 3.14 / 3.06 | 33.75–34.15 |
| gate_up J=1 | 37.03 | 37.71 | 37.64 | 37.66 | 36.77 | 0.79 / 0.78 | 36.48–36.72 |
| down n=1024 | 14.81 | 15.93 | 15.89 | 16.03 | 15.01 | 1.36 / 1.25 | — |

Earlier scheduler versions (WF1, WF2: per-wave tickets, then every exiting wave polling the uncached host flag) cost
+9…+12 ms at J=3. The flag poll was the cause; v3 fixed it.

## 4. Per-layer co-op table (M, 3 fresh processes per point, tails ABBA in-process, coarse transport, glue v2)
Delta = co-op − GPU-alone, ms/layer, per process. GPU-alone this session: gate_up 38.50–39.02 (one J=2 process 37.31),
down 18.32–18.41. GPU at 70–89 °C, median sclk 1.45–1.95 GHz, similar across the variants of a process (`*THERM` lines).

| arm | sep (`ief_silu`/`ief_addc` passes) | wide (one tail launch) | fold (twin epilogue) |
|---|---|---|---|
| gate_up J=1 | +0.59 / +0.58 / +0.49 | +0.38 / +0.39 / +0.32 | +0.36 / +0.18 / +0.27 |
| gate_up J=2 | +0.04 / −0.12 / +0.79 | 0.00 / −0.20 / +0.63 | −0.14 / −0.29 / +0.41 |
| gate_up J=3 | −0.39 / −0.46 / −0.36 | −0.35 / −0.51 / −0.44 | **−0.89 / −0.67 / −0.68** |
| down n=1024 L2 (+L42) | +1.26 / +1.16 / +1.24 (+1.19) | **+1.10 / +1.03 / +1.07 (+1.01)** | +1.30 / +1.27 / +1.30 (+1.06) |

Parts at J=3:
* **sep:** pack 1.48, window concurrent 34.0–34.4 (α 1.06–1.07), tail 2.77, NPU 30.0–31.0 ms (26.1–27.1 alone).
* **fold:** pack 1.44–1.51, window + folded SiLU 36.3–36.7 (α 1.13–1.15), tail 0.

The fold removes the 2.77 ms tail, but its window grows ~2.4 ms. The twin runs one 512-thread workgroup per WGP
(256 VGPRs), so a WGP doing NPU-column SiLU is a WGP not doing GEMM. Only ~0.3–0.5 ms of real overlap remains (the
helper's memory traffic beside other WGPs' WMMA, and no launch/drain bubble).

Down: the NPU finishes ~1 ms before the window, so the fold's addc lands on the last workgroups and it loses to
`ief_addc_wide`. The wide tail only saves ~0.1–0.2 ms over the separate passes.

**Pick:** fold for gate_up, wide for down.

## 5. Step estimate (E, 64 layers, baseline 6810.69 ms)
* **gate_up J=3 + fold alone:** −0.68 … −0.89 ms/layer ⇒ −43 … −57 ms, **−0.6 … −0.8 %**. J=1/J=2 are ≈ 0 or positive.
* **down n=1024 + wide alone:** +1.01 … +1.10 ms/layer ⇒ +65 … +70 ms (+1.0 %).
* **Both:** their sum, plus the NPU design switch twice per layer (N0 M: +1.02/+1.20 ms per alternating submit) ⇒ about
  +160 ms. Net loss.

## 6. Verdict
**KILL** wiring gate_up co-op as an opt-in hipfire route. It is exact, and the best arm (J=3, fold) is now reliably
negative, but only −0.7 ms/layer, ≤ 0.8 % of the step. The process-to-process spread (±0.4 ms/layer at J=2) is the same
size. Shipping it would cost:
* a PM twin of the incumbent kernel;
* a 160 B kernarg;
* a host flag page;
* the per-layer NPU prep and CPU relay.

The arithmetic for a GO:
* The NPU is still 30.5 ms busy for 3 commands (α 1.06–1.15 from its DDR traffic). The fold cannot hide work that needs
  the WGPs.
* The levers are the blocked-residency NPU tiling ([coop-down27.md](coop-down27.md) §7, E −0.6 ms/layer more), plus a twin whose
  helper fits beside a GEMM workgroup. That needs fewer GEMM VGPRs, or the NPU-column SiLU written as part of the
  GEMM's own 8-group epilogue for neighbour columns, which does not exist in the incumbent emitter.
* Together E ≈ −1.3 ms/layer (−1.2 %/step), still far below +10 %.
