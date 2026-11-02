# Dense Qwen3.8-27B (MQ4XTS) IU4 prefill on Strix Halo: NPU co-op study

> **Lifecycle: historical study (2026-10-02/03, hipfire 0.4.1 beta), verdict KILL; not a
> shipped path.** The int32-partials (`I32` epilogue, per-epoch `C_e`) route and the
> IEF fold candidate studied here were killed and never wired; no GPU fold of NPU
> partials exists in any serving route. Every number is a dated, fixture-bound measurement
> or labelled estimate for the dense-27B pp8192 step on one Halo host, not current
> performance or an admission. The 0.4.2 Flash-Next gfx1151 NPU trunk is a different route:
> input projections only, calibration schema 4, BF16-direct only (no I32 contract),
> out-projection unwired. See [`npu-trunk.md`](npu-trunk.md) for the current route and
> schema.

Dated 2026-10-02/03 (workspace branch `npu/dense27b`, not in this repository).

Sources (read-only):
- hipfire 0.4.1 beta source snapshot, commit f7960b4e8 (written `W:` below);
- the Halo kernel-trace study, workspace-local (written `R:`);
- the fixture `qwen3.8-27b.mq4-xts` (JSON header, `HFQMC`).

Every number is labelled **measured** (with its log) or **estimate**.

## 1. The work: every IU4 GEMM of one pp8192 step on Halo

Model: 64 layers. `layer_types` is 3× linear_attention (GDN) then 1× full_attention, repeated, giving 48 GDN + 16 FA layers. Hidden 5120, intermediate 17408, FA 24 q / 4 kv heads × 256, GDN 16 k / 48 v heads × 128 (fixture header).

The widened prefill route runs pp8192 as **one chunk** (`W:crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1445-1469`, default chunk_rows 8192 on gfx1151), so every GEMM has 8192 token rows. That matches the trace: 64 gate_up calls = 1 per layer.

Kernels on gfx1151: the PM-builder V2B entries `gemm_mq4g256v2_{residual_iu4_pm_v2b_set,residual_iu4_pm_v2b_add,gate_up_silu_iu4_pm_v2b}_gfx1151` and the hipcc `..._v2b_set_zba_gfx11`. All use a 256×256 tile, `v_wmma_i32_16x16x16_iu4`, and K128 epochs (`W:crates/rdna-compute/src/gemm.rs:20522-20555,729-756`).

GPU ms are from the measured Halo kernel trace (2nd prefill, 6810.69 ms = 1202.8 tok/s), `R:tops.md:23-32` and `R:raw/halo/dense-iu4/summary.txt`.

| GEMM | M (tokens) × N (features) × K | epochs K/128 | calls/step | GPU ms/call | GPU TOPS | ms/step | epilogue |
|---|---|---:|---:|---:|---:|---:|---|
| GDN qkv | 8192×10240×5120 | 40 | 48 | 11.463 | 74.9 | 550.2 | SET |
| GDN z+β+α (N = 6144 + 96 + 160 pad) | 8192×6400×5120 | 40 | 48 | 7.604 | 70.6 | 365.0 | SET + split (`_set_zba`) |
| FA q (+ output gate) | 8192×12288×5120 | 40 | 16 | 13.722 | 75.1 | 219.6 | SET |
| FA k, v | 8192×1024×5120 | 40 | 32 | 1.386 | 62.0 | 44.4 | SET |
| o_proj (GDN and FA) | 8192×5120×6144 | 48 | 64 | 6.706 | 76.9 | 429.2 | SET into delta (residual fold deferred) |
| gate_up + SiLU (F1-lite) | 8192×34816×5120 | 40 | 64 | 38.547 | 75.8 | 2467.0 | `h = g/(1+expf(-g))*u` |
| down | 8192×5120×17408 | 136 | 64 | 18.939 | 77.1 | 1212.1 | ADD `RN(old + sum)` |
| **total** | 399.6 TOP | | 336 | | **75.6** | **5287.4** (77.6% of 6798.8 ms kernel time) | |

The remaining 1512 ms are FA attention (457), GDN scan/prep/solve (370), the A4 producers (585: `fused_silu_mul_mq_rotate_awq_i4_hin` 191, `fused_rmsnorm_mq_rotate_awq_i4_fold` 171, `gated_norm_mq_rotate_awq_i4_gfx11` 115, `fused_rmsnorm_mq_rotate_awq_i4` 76, `sigmoid_mul_rotate_x_mq_awq_i4_gfx11` 32) and small glue.

lm_head is a 1-row GEMV in prefill, not an IU4 GEMM.

## 2. The GPU route's exact numeric contract

Sources: `W:kernels/src/gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip:14-33,146-157,244-290,346-371,443-483`, `W:crates/peacemaker-isa/src/kernels/iu4_fold.rs`, `W:crates/peacemaker-isa/src/kernels/iu4_v2b.rs:1-40,521-527`, `W:kernels/src/block_i4_128_quant.hip`.

1. **Weights (symmetric MQ4G256V2, "XTS").** Each feature row is `K/256` groups of 136 B. A group is `[f16 sc0, f16 zp0, f16 sc1, f16 zp1]` plus 128 nibble bytes. K half `h` (epoch `e = 2*group + h`) is the 64 B at `+8+64h`. Byte `j` holds K `2j` in its low nibble and `2j+1` in its high nibble.
   - Nibbles `u` are unsigned, with `zp == -8*sc`.
   - The GPU rebiases them once in staging with `XOR 0x88888888`, so the WMMA operand is **`u - 8` in [-8, 7]** (signed).
   - The epoch scale is **`sc_e` = f16 at header byte `4*(e&1)`**, widened exactly to f32. `zp` is never read.
2. **Activations (A4, `block_i4_128`, 72 B).** Layout is `[f32 d, i32 s, u8 qs[64]]`, block index `e*tokens + t`. `qs` holds K `128e+2j` in the low and `+1` in the high nibble, as two's-complement int4.
   - Producers are fused kernels (RMSNorm / gated norm / SiLU·mul / sigmoid·mul, AWQ scale and FWHT rotation, then the shared quant recipe). Per 128 block: `amax`, candidate scales `d` (MSE-clip grid; the gfx11 fused producers use the host-flag candidate set, `{5,7}` per `block_i4_128_quant.hip:20-26`; the flag value on this binary was not verified), then `q = clamp(rintf(x/d), -8, 7)`, and `s = sum q`.
   - `d` > 0, or `d = 1` when `amax == 0`.
   - The symmetric fold never reads `s`.
3. **Integer chain per epoch.** `C_e = sum_{j<128} (u-8)·q` exactly. It is 8 `v_wmma_i32_16x16x16_iu4` steps, signA = signB = true, seeded with `0x4B400000`, so `f32(C_e) = bits(C_e + magic) - 12582912.0` is exact (`|C_e| < 2^22`). The range is **`C_e` ∈ [−7168, 8192]**.
4. **Fold**, per output in ascending epoch from `sum = +0.0`: **`t = RN(f32(sc_e) * d_e)`**, then **`sum = fmaf(t, f32(C_e), sum)`**, single-rounded RNE, f32 denormals on. The `t` product is per (feature, token, epoch), so there are **two IEEE ops per output-epoch**.
5. **Epilogues.**
   - SET stores `sum` to `Y[token][feature]`.
   - ADD (down) stores `RN(old + sum)`.
   - o_proj SETs a delta, and the residual add happens in the next `_fold` producer.
   - gate_up stores `h = g/(1+expf(-g))*u` with hipcc's `expf`/`fdiv` DAG (`iu4_v2b.rs:753-759`). Its input `g` is a full f32, so no lookup table can reproduce it.
   - z+β+α splits rows 6144..6239 into β/α and drops the 160 padding rows.

CPU reference of 1–4 (Rust): `crates/peacemaker-npu/src/kernels/iu4.rs`, with `partials` and `fold_set`, operand decoders, the exact f16 widening, and tests including a case that separates single-rounded fma from multiply-then-add.

## 3. NPU kernel: native IU4 operands → bit-exact per-epoch int32 partials

`iu4::EpochGemm` (`crates/peacemaker-npu/src/kernels/iu4.rs`) is the silicon-proven **V8 i32** whole-array design run as an **epoch-batched GEMM**. Wave rows `e*Mpad + t` carry activation epoch `e`, and every wave of epoch `e` streams weight epoch `e` as its B tile (`gemm_array::ArrayDesign::pack_in_per_mw`, a host-layout-only extension). Core programs, routes, descriptors and TXN are unchanged. The output is `C_e[e][t][feature]` int32.

- The inputs are the hipfire-native buffers. The host packer expands nibbles to the int8 the V8 core consumes, applying the rebias, so the integers are identical and nothing about precision changes. Core-side int4 unpack (`VLDB.UNPACK` exists) would save 64 KiB of the 1.125 MiB moved per wave, which is not the binding resource (see §4).
- CLI: `npu-gemm TOKENS FEATURES K --variant V8 --epi i32 --iu4-epochs`. Every checked submit compares all partials with `iu4::partials`.

**Simulator** (exact PDI/TXN interpreter, `crates/peacemaker-npu/tests/iu4_epochs.rs`, log `logs/dense27b/sim-iu4-epochs.log`): exact for 512²×256 (Fast and Slow control), 300×700×512 (token/feature padding), 1024³ (32 waves, two token blocks per epoch), and 3 submits with different operands on one context. All runs include the extreme partials +8192 and −7168. The CPU fold of the NPU partials is bit-identical to the fold of the CPU partials.

**Silicon** (measured, window w1, `logs/dense27b/w1-be0be55.log`, 2026-10-02 23:39 UTC, fclk pinned, kernel log clean): 0 mismatches in every checked submit.

| shape tokens×features×K | check | result |
|---|---|---|
| 512×512×256 | 3 submits, one context | 0/524 288 ×3 |
| 1024×1024×1024 | 2 submits | 0/8 388 608 ×2 |
| 300×700×512, FastSlowCtl | fresh context | 0/1 433 600 |
| the four timing shapes below | first + final of each loop | 0 mismatches |

The f32 fold stays on the GPU in this variant.

## 4. Measured NPU rates at the 27B shapes

Window w1, busy TOPS = useful 2·M·N·K / summed submit→completion time.

| kernel | shape | busy TOPS | wall TOPS | µs/submit (last) | note |
|---|---|---:|---:|---:|---|
| IU4 epoch partials (exact contract `C_e`) | 512×3072×5120 (qkv / q / gate_up K) | **3.35** | 2.95 | 5230 | writes 251 MiB/submit, 52 GB/s (write ceiling) |
| same | 512×2560×6144 (o_proj K) | **3.35** | 2.96 | 5414 | |
| same | 512×512×17408 (down K) | **3.24** | 3.00 | 2977 | |
| same | 512×1024×5120 (k/v) | **3.07** | 2.90 | 1810 | |
| V8 int8 full K (exact int GEMM, not the contract) | 4096×4096×5120 | **26.94** | 26.19 | 6583 | |
| V8 i32 full K | 4096×4096×5120 | **26.56** | 25.80 | 6357 | |
| V8 int8 full K | 4096×5120×6144 | **27.16** | 26.12 | 9584 | |
| V8 int8 full K | 4096×5120×8704 (half of down K) | **27.53** | 26.49 | 13271 | |

**Derived per-GEMM time if the NPU did the whole GEMM** (ESTIMATE = ops / measured rate):

| GEMM | partials route | fold-free V8 bound |
|---|---:|---:|
| qkv | 256 ms | 32 ms |
| gate_up | 872 ms | 108 ms |
| down | 451 ms | 53 ms |

Against the GPU's 11.5 / 38.5 / 18.9 ms, the partials route is **write-bound and dead**: per output-epoch it writes 4 B (int32) or 2 B (int16; `C_e` fits). That caps the NPU at ≈3.4 / ≈6.9 TOPS, and the GPU would have to read 2–4 B per output-epoch to fold it. At ≤256 GB/s that read costs more than the GPU's own 75 TOPS compute of the same outputs (75 TOPS / 256 ops × 2 B = 586 GB/s).

Constraints hit: V9 (`K ≤ 3456`) and V10 (`kc·(2+NW) ≤ 112`) admit no 27B shape. V8 needs a K split for down (MAX_KC 160) and N slices (≤ 256 waves per submit).

## 5. Can the NPU do the f32 fold exactly? Lane-level models

`crates/peacemaker-npu/src/kernels/iu4_fold_model.rs`, log `logs/dense27b/fold-model.log`.

Both models compute `t = RN(sc·d)` then `fma(t, C, s)` from AIE2P lane primitives only, and charge each primitive to its VLIW slot (`crates/peacemaker-npu/src/isa/gen.rs`):
- **Mv**: `vadd/vsub.32`, `vband`, `vbor`, `vlt/vge/veqz`, `vsel`, `vabs`, `vmov`, `vups`; 16 i32 lanes, one per bundle.
- **St**: `vsrs` with a scalar shift and conv-even rounding.
- **Vec**: `vmul` 32×16 into 64-bit lanes, accumulator `vadd/vsub`, fp32-accumulator `vadd.f/vsub.f/vmul.f` (bf16 inputs only). This slot is shared with the GEMM's VMAC.

AIE2P has **no lane-variable shift, no vector clz and no xor**.

Both models are **bit-exact** against `f32` mul then `f32::mul_add` over:
- 800 000 chained 40-epoch elements;
- 400 000 adversarial elements (cancellation, signed zeros, subnormal sc/d/s, extreme C, any-magnitude s);
- 13 824 constructed final-rounding ties.

Lanes outside a fast path are flagged to a scalar fallback, and the fallback results are included in the exactness check.

| model | per element: Mv / St / Vec-int / Vec-fp ops (÷16 lanes) | cycles per output-epoch | fallback (realistic) |
|---|---|---:|---:|
| float-assisted: integer `t` and `Q = Mt·C`, exact hi+lo split via the GPU's magic-number conversion, Boldo–Melquiond `RN(hi+lo+s)` with round-to-odd from 2Sum plus a last-bit fix | 22 / 4 / 2 / 23 | 1.375 (fp32 acc 32 lanes, Mv-bound) – 1.81 (16 lanes, Vec + GEMM) | 0.16 % (`C = 0`) |
| integer only: 64-bit two-word align / add / clz / RNE via select ladders | 122 / 29 / 2 / 0 | 7.6 | 17 % (window `δ ∈ [−8, 24]`; widening costs more) |

The GEMM itself costs **0.331 cycles per output-epoch** per core (measured 1357 cycles per 128×64×64 chunk). The exact fold therefore costs **4.2–5.5×** the GEMM with the float-assisted path, ~3.7× with a perfect Mv/Vec rebalance, and ≥ 23× integer-only. None reaches ≤ 3×.

The float-assisted path also **assumes** the fp32-accumulator add is IEEE RNE on silicon, which is unverified. The probe is not run, pending a project decision.

NPU core rate with the fold (ESTIMATE) = 32 · 1.8 GHz · 256 / cycles: **8.1 / 10.7 / 12.0 TOPS** (16-lane / 32-lane / balanced).

## 6. GPU derate vs NPU traffic (measured)

Window m4, `logs/dense27b/m4-2c54965/`, 2026-10-02 23:51–23:53 UTC. One dense-27B daemon load. Prompt = `bench_prefill` built-in tokens (`10+(i%1000)`); requests md5 `f8b7d1b377185d6f94bae6accb2647a5`. MemAvailable was 28.0 GB after the load. The NPU ran a V8 int8 4096×4096×5120 loop throttled with `--gap-us`.

| NPU level | NPU DRAM GB/s | NPU TOPS busy / wall | GPU pp8192 median of 2 (ms) | slowdown, raw (vs mean alone 6676) | slowdown, drift-corrected |
|---|---:|---:|---:|---:|---:|
| gap 0 | 46.9 | 22.97 / 22.88 | 7348.6 | +10.06 % | +11.3 % |
| gap 13 ms | 16.0 | 19.32 / 7.79 | 7064.8 | +5.81 % | +5.9 % |
| gap 58 ms | 5.3 | 19.99 / 2.57 | 6893.1 | +3.24 % | +2.2 % |

**UNRELIABLE: GPU-alone drift +4.13 %.** The first pair was 6529.8 / 6552.6 ms and the last pair 6726.7 / 6896.2 ms. The second run of every pair was also slower, which suggests a thermal or power ramp. The drift correction uses a baseline linear in time between the two alone pairs. NPU first and final submits verified exact at every level.

Reading: the derate grows with NPU traffic, but sublinearly. That points to a power/thermal component on top of DRAM sharing.

## 7. Co-op step model: column split (ESTIMATE)

Per GEMM, the NPU owns `n` whole output features (weight rows; its share of B resident in NPU-packed form) and the GPU the other `N−n`.

- GPU time is `α · (1−n/N) · g` (trace `g`), with `α` taken from §6 at the NPU's own traffic. Traffic is V8 bytes/op 2.05 B/kop plus 0.39 B/kop of f32 output, times the NPU rate.
- NPU time is `(n/N) · ops / (P·β) + 0.215 ms` fixed per submit, with **β = 0.853** measured in §6 (22.97 / 26.94).
- Handoff `H` is added per GEMM: **0.27 ms** CPU relay or **0.02 ms** GPU-driven ring (both estimates). `n` is optimised per GEMM.
- Extra GPU work (estimates at 200 GB/s): a SiLU pass over the NPU's gate_up columns, and an NPU-layout int8 copy of each producer's output.
- The non-GEMM 1523 ms is unchanged.

| NPU rate P (exact fold) | α at its traffic | NPU share | step ms, relay | step ms, ring | tok/s | gain |
|---|---:|---:|---:|---:|---:|---:|
| 8.1 TOPS (fp acc 16-lane) | 1.060 | ≈ 9 % | 6821 | 6749 | 1201–1214 | −0.2 … +0.9 % |
| 10.7 TOPS (fp acc 32-lane) | 1.070 | ≈ 11 % | 6743 | 6665 | 1215–1229 | +1.0 … +2.2 % |
| 12.0 TOPS (perfect balance) | 1.074 | ≈ 12 % | 6701 | 6623 | 1222–1237 | +1.6 … +2.8 % |
| 26.9 TOPS, **fold free (not bit-exact)** | 1.113 | ≈ 23 % | 6223 | 6140 | 1316–1334 | +9.5 … +10.9 % |

Baseline: 6810.7 ms, 1202.8 tok/s.

The raw (not drift-corrected) α curve shifts every row by ≤ 0.3 %. Break-even is `P·β > (1 − 1/α)·75.6` TOPS.

## 8. Verdict

**KILL the end-to-end co-op prototype under the current bit-exact contract** (accepted 2026-10-03).
- Parity holds only for the int32 partials. They are measured-exact on silicon but write-bound at 3.3 TOPS, and the GPU cannot afford to read them.
- An exact NPU-side fold costs 4–5.5× the GEMM (≥ 23× without fp hardware) and rests on an unverified fp32-accumulator RNE assumption.
- The best modelled exact gain is +1 to +3 % of a pp8192 step. Getting it would require:
  - the fold kernel,
  - the fp probe,
  - NPU-layout producers,
  - a split SiLU pass,
  - GPU-driven sync.

The fp32-accumulator probe and a Flash-Next small-M window were dropped (project decision; the levers study measured those shapes in its w5, see `report.md`).

## 9. What a fold-contract change would have to look like (design note; no hipfire edit)

The NPU keeps its integer rate only if the per-epoch fold becomes **integer-exact and order-independent**. Today's fold rounds twice per output-epoch in f32 (`RN(sc·d)`, then `fma`), which on AIE2P costs 4–5.5× the GEMM. The candidate below makes the fold a single 32×16 integer MAC per output-epoch on both devices.

### Candidate contract "integer epoch fold" (IEF)

- **Weight scales**: at load, re-express each row's per-epoch f16 scales on one per-row exponent `a_r` = the max exponent over the row, as unsigned 16-bit integers `S_{r,e} = sc_{r,e} · 2^(-a_r)`, with a sign bit folded into the integer. This is lossless when a row's scale exponents span ≤ 5 binades (11 + 5 = 16 bits). Wider rows round the smallest-scale groups: RNE, one rounding at load.
- **Activation scales**: the A4 producer emits `d_{t,e} = D_{t,e} · 2^(b_t)`, with `D` 16-bit and a per-token exponent `b_t`. The producer's MSE-clip candidate search picks `d` on that grid, so each quantized code `q` is computed against the grid value it will be scaled by. This is the one producer change.
- **Fold**: `Y[t][r] = RN_f32( 2^(a_r + b_t) · Σ_e (S_{r,e} · D_{t,e}) · C_e )` with **exact int64 accumulation**. `S·D` is ≤ 32 bits; one term is `C_e` × that, ≤ 46 bits; 136 epochs stay under 2^54. There is exactly one rounding per output, at the end, and no per-epoch rounding at all. Accumulation order no longer matters, so a K split (and an NPU/GPU split along K) also becomes exact.
- **Epilogues**: SET stores `Y`. ADD stays `RN(old + Y)`. F1-lite SiLU keeps its hipcc DAG on the GPU. For NPU-owned columns the NPU emits `g` and `u` and the GPU's SiLU pass applies `silu(g)·u`, bit-identical because its inputs are the same f32 values.

### What changes in GPU numerics

- **Scales.** Weight scales are rounded only for rows wider than 16 bits of scale range. Activation scales move onto a 16-bit-per-token grid, a quantizer candidate-grid change.
- **Accumulation.** It becomes exact: no f32 rounding per epoch, one final RNE. In isolation this is more accurate than today, because 40–136 roundings become one.
- **GPU fold cost.** The per-output-epoch VALU sequence changes from `mul_f32, sub_f32, fma_f32` to `mul_u32` (`S·D`) plus a 64-bit multiply-add (`C·Sq`), and the int64 accumulators double the sum VGPRs: 128 → 256 per lane in V2B, which does not fit as-is. So the GPU kernels would need a re-tiling. On RDNA3.5, `v_mad_i64_i32` issues at a reduced rate (rate not measured here), so **the GPU's own fold may slow down**. That has to be measured before the NPU gain is credited.

### NPU cost (ESTIMATE)

Per output-epoch the NPU adds one Vec `vmul` 16×16 (`S·D`, or 32-bit) and one Vec 32×16 `vmac` of `C_e` into an acc64 lane: 2–3 Vec ops ÷ 16 lanes = 0.13–0.19 cycles, against the GEMM's 0.25 Vec / 0.331 total. That is **1.15–1.33×** the GEMM, all-integer, with no fp hardware assumption. `C_e` stays in core registers, so no partials ever leave the core.

The final int64 → f32 RNE costs once per output (normalize + `vsrs` conv-even ≈ 30 Mv/St ops ÷ 16 lanes), amortized over 40–136 epochs, under 0.05 cycles per output-epoch.

With the NPU at 19–22 TOPS, the §7 model gives **step 6311–6490 ms, 1262–1298 tok/s, +4.9 … +7.9 %**. The low ends use relay sync, the high ends ring sync.

### Gates, in order (each a kill point)

1. **Quality.** Run hipfire's dense-27B KLD gate of the IU4 route (the `iu4 c24` budget 0.10 KLD convention, `W:kernels/src/block_i4_128_quant.hip:23-24`), plus PPL, IEF vs the current route on the GPU alone. Abort if it is over budget or worse than the current route by more than run-to-run noise.
2. **GPU speed.** The IEF GPU kernels must not lose more than 1 % pp8192 tok/s on Halo GPU-alone; otherwise the co-op gain is eaten before it starts.
3. **Parity.** CPU model of IEF, then the NPU kernel on the exact simulator, bit-exact vs the GPU on captured real tiles of every GEMM type including the ADD and split epilogues. Then one ≤ 90 s silicon window comparing NPU bytes with GPU bytes.
4. **Co-op prototype** (relay sync first), with the derate re-measured. §6 is drift-flagged, so it needs a thermally stable protocol: interleaved alone/concurrent pairs and a cool-down. GO if the measured step gain is ≥ 3 % net of every extra pass.

IEF changes the GPU numerics of a shipped route, so it needs user approval before any hipfire work. Escalated for a user decision.

## 10. Files

- `crates/peacemaker-npu/src/kernels/iu4.rs`: contract, CPU reference, `EpochGemm`.
- `crates/peacemaker-npu/src/kernels/iu4_fold_model.rs`: fold lane models.
- `crates/peacemaker-npu/src/kernels/gemm_array.rs`: `pack_in_per_mw`, `wave_m`.
- `crates/peacemaker-npu/tests/iu4_epochs.rs`.
- `npu-gemm`: `--iu4-epochs`, `--gap-us`.
- `tools/npu/npu-dense27b-w1.sh`, `tools/npu/npu-dense27b-m4.sh`.
- Logs: `logs/dense27b/` (workspace-local, not in this repository).
