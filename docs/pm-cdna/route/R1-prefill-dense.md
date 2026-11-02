# R1 — gfx1201 native-FP8 dense/trunk prefill (`HIPFIRE_IU4_PREFILL=0`) and its gfx942 (CDNA3, wave64) mirror

Tree: `/home/kaden/qcal/release-0.4.2/pm-cdna3` (all bare `path:line` are relative to it). Plan: `subq4/review/GFX942-FP8-FN-PLAN.md` (relative to `/home/kaden/qcal/release-0.4.2`). ISA: `CDNA3 ISA §x p.N` = `pm-cdna3-ref/cdna3-isa.pages.txt`, printed page N of 553. CPU-only reading; nothing built or run. `[UNVERIFIED]` = confirm on MI300X.

## 0. Route identity

- KLD-pinned model: Qwen3.8-27B, "H2, symmetric MQ4V2 XT (qt44) with per-group AWQ alpha" (`AGENTS.md:600-601`). Native-FP8 pins: WT2 0.041859 / code24 0.029867, `--kv-mode fp8 --kv-v q8` (`AGENTS.md:641-642`). The pins are on `hipfire-arch-qwen35`. Qwen4 Flash-Next has **no** FP8 trunk arm (§4).
- Selection chain for a dense chunk: `HIPFIRE_IU4_PREFILL=0` → `iu4_prefill_enabled()=false`. IU4 producers return `None`. Then the FP8-stream producers are admitted (`prefill.rs:629-641`, `dispatch.rs:4519-4529`) and their `Mq4v2Fp8Prepared` planes go to the FP8 consumers. For N ≥ 512 the consumer is the PM builder **F2 Row** bundle (`gemm.rs:10263-10283`, the `!self.flags.iu4_prefill_enabled()` term at `:10268`). For 64 ≤ N < 512, or `HIPFIRE_G12_FP8_ISA=0`, it is the hipcc v2 / s2bt8 / bt tiles (`gemm.rs:10837-10936`).
- The pinned artifact uses symmetric uniform qt44 weights (`gpu.mq4v2_symmetric`, stamped at `crates/hipfire-runtime/src/model_load.rs:479-482`). F2 requires it (`gemm.rs:10266`). Lloyd qt=52 arms are siblings, not on the pinned route.
- **Brief-cited lines that are NOT on this route:**
  - `kernels.rs:4706-4818`: IU4 MMQ sources.
  - `scratch.rs:2198-2206`: `ensure_int4_mmq_x`, IU4 only.
  - `kernel_registry.rs:392-399`: inside the `gfx1100|gfx1151` block opened at `:377`. On gfx11, IU4=0 means Q8_1 MMQ.
  - `gemm.rs:21092-21101`: IU4 consumer doc.
  - `feature_flags.rs:42-45,774,2281-2295`: the IU4 flag itself and its test.
  - The final-norm/lm_head of Qwen3.5 prefill is a last-token `Step::Gemv` (`prefill.rs:13076-13094`), a decode GEMV. It is not FP8 prefill.

## 1. Levers

| lever (env / config) | where parsed (path:line) | gfx1201 default | semantics | gfx942 mirror: same name, default, where resolved |
|---|---|---|---|---|
| `HIPFIRE_IU4_PREFILL` / `kernel.iu4_prefill` | `feature_flags.rs:774`; registry `crates/hipfire-config/src/lib.rs:2925-2926` | on (gfx1100/1151/1201). The pinned route sets it to `0`. | `0` disables A4/IU4. That removes the `!iu4_prefill_enabled()` veto in F2 admission (`gemm.rs:10268`) and enables the FP8 qkvza arm of the GDN chunk-scan admission (`prefill.rs:14407-14412`). | Same name. Resolves **false** on gfx942 already (test `feature_flags.rs:2292-2294`), because gfx942 has no int4 MFMA. On gfx942 this env is a no-op, and native FP8 is the only quantized-activation prefill route. |
| `HIPFIRE_GFX12_MQ4V2_FP8_QKVZA` / `kernel.gfx12_mq4v2_fp8_qkvza` | `feature_flags.rs:950-951` (field `:306`); `lib.rs:2694-2702` | on (`arch=="gfx1201"`) | Family flag for the 4-way LA qkvza. Consulted by producer admission (`prefill.rs:620`), F2 entry (`gemm.rs:10827-10836`), GDN-fused F2 (`gemm.rs:10310`), v2 (`gemm.rs:10843-10845`), the unprepared route entry (`gemm.rs:11528-11539`) and chunk512 (`prefill.rs:1693-1698`). | Same name. **Default off on gfx942** (opt-in): plan §13.2 makes this a generator-only arm with its own KLD row (`subq4/review/GFX942-FP8-FN-PLAN.md:314-318`). Resolve at `feature_flags.rs:950` as `arch=="gfx1201"` default plus explicit opt-in for `gfx942`. Every `arch=="gfx1201"` gate listed in §5 must admit gfx942. |
| `HIPFIRE_GFX12_MQ4V2_FP8_GATEUP` | `feature_flags.rs:946-947` (field `:300`); `lib.rs:2676-2684` | on | FFN gate/up family (`prefill.rs:621-623`, also requires `!hfq4g256_ldsstage_wmma`). Gates `fp8_silu_h_active` (`gemm.rs:10536-10547`) and the route entry (`gemm.rs:34281-34291`). | Same. Off on gfx942 (opt-in), same site. |
| `HIPFIRE_GFX12_MQ4V2_FP8_QKV` | `feature_flags.rs:952-953` (field `:309`); `lib.rs:2703-2711` | on | FA 3-way qkv family (`prefill.rs:624`, `gemm.rs:11276-11283`, `:11291-11293`). | Same. Off on gfx942 (opt-in). |
| `HIPFIRE_GFX12_MQ4V2_FP8_RESID` | `feature_flags.rs:948-949` (field `:303`); `lib.rs:2685-2693` | on | Residual (wo / w_down) family (`prefill.rs:625`, `gemm.rs:36632-36637`, `:36883-36892`). | Same. Off on gfx942 (opt-in). |
| `HIPFIRE_GFX12_MQ4V2_FP8_V2` | `feature_flags.rs:959-960` (field `:350-356`); `lib.rs:2712-2720` | on | Prerequisite for F2 (`gemm.rs:10267`) and the staged v2 hipcc tiles at N ≥ 256 (`gemm.rs:10843-10845`). | Same. On wherever a family flag is on (gfx942: follows opt-in). |
| `HIPFIRE_GFX12_FP8_STREAM` / `kernel.gfx12_fp8_stream` | `feature_flags.rs:975-976`; `gfx12_fp8_stream_enabled()` `:1281-1283`; `lib.rs:2827-2828` | on | Fused producer → FP8 planes; the standalone pack disappears. Live gate `fp8_stream_active` (`dispatch.rs:4519-4529`): arch gfx1201, eager, batch ≥ 64, K%256==0, no A8, no IU4-verify scope. | Same name. Default on when a family flag is on. `feature_flags.rs:1282` and `dispatch.rs:4523` must admit gfx942. |
| `HIPFIRE_G12_FP8_ISA` | `gemm.rs:830-831` | on | PM F2 Row bundle for N ≥ `G12_FP8_F2_MIN_N=512` (`gemm.rs:835`). `0` → hipcc v2 (`gemm.rs:821-829`). | Same name. Selects the gfx942 PM twin bundle (§3 K7). Default off until that bundle is certified; the HIP twin K8 runs meanwhile. |
| `HIPFIRE_G12_FP8_ISA_FORCE_SMALL_N` | `gemm.rs:838-839` | off | Quality-only: admits F2 below 512 rows. | Same; off. |
| `HIPFIRE_G12_FP8_ISA_COVERAGE` | `gemm.rs:842-843`, print `:10503-10505` | off | Prints `fp8_f2_selected:` for each selected site. | Same; off. Needed for the gfx942 route census. |
| `HIPFIRE_G12_FP8_F2_BUNDLE=<path>` | `gemm.rs:10407-10418` | unset | Loads the F2 `.hxaco` from a file (A/B against embedded `kernels.rs:4899-4900`). | Same name, gfx942 bundle path. |
| `HIPFIRE_BF16_H_FP8` | `gemm.rs:818-819` | on | F2 silu stores a compact bf16 h (`gemm.rs:10399`). The down producer reads bf16 (`gemv.rs:4290-4301`). | Same; on. The integer RNE pack is portable (`epilogue.rs:9-34`). |
| `HIPFIRE_FP8_PROD_INREG` | `gemv.rs:88-92` (`k <= 17408`); `gemm.rs:10545` | on | Producers keep the FWHT row in registers (no `x_rot` store). Required for silu-h. | Same; on. Same K bound (9 groups/lane, `mq4v2_fp8_producer_pack.hip:157-161`). |
| `HIPFIRE_FP8_SILU_H` | `gemm.rs:10546` | on | Paired-row gate/up epilogue writes h. Needs symfold, 128×128 v2 geometry, N ≥ 256, gate_m==up_m (`gemm.rs:10536-10547`). | Same; on. |
| `HIPFIRE_RMSNORM_P1A_BATCHED` | `gemv.rs:93-98` | on | Selects the `_short` (P1A batched) vs `_seq` RMS FP8 producer objects (`gemv.rs:4008-4028`). Bit-identical. | Same. |
| `HIPFIRE_FP8_ROW_SCALE_SHIFT` | `gemv.rs:99-116` | 0, allowed {-1,0,1,2,3} | Multiplies the power-of-two row scale by 2^shift (`mq4v2_fp8_producer_pack.hip:120,180`). | Same. With the ×0.5 FNUZ fold, \|x/a\| ≤ 448 (shift −1) → ≤ 224 < 240, still in FNUZ range. |
| `HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM` | `gemm.rs:10514-10526` | 128×128×64, 8 waves | v2 hipcc tile geometry. | Same name; gfx942 geometries per §3 K8. |
| `HIPFIRE_FP8_SYMFOLD` | `gemm.rs:10528-10531` | on (symmetric + v2) | `q→E4M3(q−8)` with one `fma(s_h,P,T)` per half (`gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:68-73,889-903`). | Same. |
| `HIPFIRE_GFX12_MQ4V2_FP8_SLABS` | `gemm.rs:10852-10856` (same at `:11300-11304`, `:34457-34462`, `:36655-36659`) | 2 slabs when N%128==0 | s2bt8 vs single-slab bt tiles at N < 256. | Same. |
| `HIPFIRE_GDN_PREP_FUSED` | `norm.rs:5680-5687` | on (gfx1201) | F2 `qkvzagdn` epilogue + `gdn_chunk_prep_fixup` replace `gdn_chunk_prep` (`prefill.rs:7521-7581`). | Same. On gfx942 it is unreachable until a gfx942 GDN chunk scan exists (K7c). |
| `HIPFIRE_GFX12_GDN_CHUNK_SCAN` | `feature_flags.rs:963-964` | on (gfx1100/1151/1201) | Admits the chunk scan only if the qkvza projection route is IU4 or FP8 on gfx1201 (`prefill.rs:14396-14442`). | Same name. gfx942 is not in the arch list: GDN-fused F2 is blocked (plan P2 scan port). |
| `HIPFIRE_GDN_SCAN_OUT` | `norm.rs:5531-5549`; route default `prefill.rs:7069-7090` | FP8-stream reader → f32 | GDN scan `out` plane format read by the FP8 gated-norm producer. | Same; f32. |
| `HIPFIRE_HFQ4G256_LDSSTAGE` | `feature_flags.rs:899` | off | If on, it vetoes the FP8 gate/up producer (`prefill.rs:621-623`). | Same; off. |
| `HIPFIRE_A8_PREFILL` | `feature_flags.rs:777` | off | `a8_prefill_active` vetoes F2 and FP8 stream (`gemm.rs:10269`, `dispatch.rs:4521`). | Same; off. |
| `HIPFIRE_PREFILL_MAX_BATCH` / chunk512 | `prefill.rs:1724-1729`; `fp8_chunk512_for_gpu` `:1693-1698`; `:1710-1722` | 512 when GATEUP∧RESID∧QKVZA (else 384) | Ordinary chunk ceiling. | Same env. `fp8_chunk512_for_gpu` tests `arch=="gfx1201"`. gfx942 needs its own measured rung; do not inherit 512 blindly. |
| `HIPFIRE_PREFILL_CHUNK_ROWS` / `prefill.chunk_rows` | `prefill.rs:1731-1758` | 8192 widened (gfx1100/1151/1201) | Widened ordinary prefill. On gfx1201 the projection route requires all four FP8 flags (`prefill.rs:1958-1970`). Scratch sizing uses `fp8_row_bytes_wide` when `arch=="gfx1201"` (`prefill.rs:2138-2141,2251-2255,2361-2425`). | Same env. Every `arch=="gfx1201"` test at those lines must include gfx942. Rung sizes follow the FP8 plane extents (§2). |
| `HIPFIRE_CB_VERIFY_FP8` (separate route) | `prefill.rs:1239-1251`; `crates/rdna-compute/src/gemm_fp8_verify.rs:300-301` | off | Verify-only PM family `mq4_fp8_verify_*` (`kernels.rs:4912-4924`). **Not this route.** | Out of R1. |
| `HIPFIRE_FP8_WMMA` | `feature_flags.rs:806`; `dispatch.rs:320` | off | HFP4G32 FP8 path, not MQ4V2. **Not this route.** | n/a. |

## 2. Scratch / sidecar buffers (byte layouts are the gfx942 contract)

| buffer | layout | where | gfx942 |
|---|---|---|---|
| X8 | `[N,K]` u8 E4M3 codes | `scratch.rs:1123-1140`; grown `scratch.rs:2502-2546` | Byte-identical under the ×0.5 activation fold (D1). Exception: −0. |
| half_sums | `[N,K/256,2]` f32: sum of **decoded** codes per 128-half | same | Used only by non-symfold v2/bt. Keep in the OCP value domain (D1c). |
| row_scales (`D`) | `[N]` f32, 2^clamp(ceil(log2(amax/224)),−31,9) | `mq4v2_fp8_producer_pack.hip:70-79` | Identical. |
| F2 workspace Wf/Rw/Ew | Wf `Mp·K` B, Rw `Mp·K/128` f32, Ew `Mp` f32, Mp = ceil(Mt/256)·256 | `scratch.rs:1142-1149,1151-1161`, field `:407-409`; carved `gemm.rs:10382-10428` | Identical bytes (D2). |
| GDN fused targets q/k/v FP16, conv ring | `prefill.rs:7472-7483,7530-7543` | — | Blocked with K7c. |
| capture | Every FP8 entry is eager-only (`gemm.rs:10388-10392`, `dispatch.rs:4072-4077`, `gemv.rs:3993-3998`) | — | Same rule. |

## 3. Design contracts used by the kernel table

**D1 — FNUZ folds (exact).**
- **(a) Activations.** gfx1201 emits `cvt_pk_fp8_f32(x·inv_a)` with \|x/a\| ≤ 224 (`mq4v2_fp8_producer_pack.hip:134-142,195-202`; `pack_f16_to_fp8_mq4v2.gfx12.hip:265-273`). The gfx942 twin emits `cvt_pk_fp8_f32((x·inv_a)·0.5f)`. The FNUZ grid is the OCP grid halved, subnormals included (`m·2⁻¹⁰` vs `m·2⁻⁹`), so RNE gives the same byte, with FNUZ value = OCP value / 2. The ×0.5 is exact. FNUZ max is 240 (CDNA3 ISA §7.2 p.50). D and `inv_a` are unchanged.
- **(b) Weights.** The repack LUT `q→E4M3(q−8)` (`mq4v2_fp8_fragment_repack.gfx12.hip:85-99`; symfold table `gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:111-127`) contains no 0x80 (q=8 → 0x00; zero-scale halves → 0x00, repack `:147-158`). No remap is needed. FNUZ value = (q−8)/2.
- **(c) Product.** FNUZ×FNUZ = (x/a)(q−8)/4 → C₉₄₂ = C₁₂₀₁/4 exactly, for the same summation order. Fold ×4 into an exact power-of-two multiply:
  - F2: `D' = 4·D` at D staging. The epilogue is `RN(D·RN(Ew·C))` (`epilogue.rs:114-120`), and RN(4D·RN(Ew·C/4)) = RN(D·RN(Ew·C)).
  - v2/bt: `s_h' = 4·s_h` at LDS staging, so `fma(4s_h, P/4, T) = fma(s_h, P, T)`.
  - Rw ratios are unchanged.
- **(d) Half sums.** The in-register path decodes with `__builtin_amdgcn_cvt_pk_f32_fp8` (`mq4v2_fp8_producer_pack.hip:206-209`), which is FNUZ on gfx942 and gives value/2. The twin multiplies `lane_parts` by 2 (exact) or uses the software OCP decoder `fp8stream_e4m3_to_f32` (`:39-54`) on the identical byte. The from-F32 path already uses the software decoder (`:144-147`) and is unchanged.
- **(e) Caveats.** gfx1201 may emit 0x80 (OCP −0) for small negatives. gfx942 FNUZ has no −0, so the expected byte is 0x00 `[UNVERIFIED]`. Byte-diff harnesses must treat 0x80≡0x00. MFMA keeps denorms and forces RNE (CDNA3 ISA §7.1.2 p.42). The internal MFMA accumulation order is not documented.

**D2 — F2 Row on gfx942: MFMA 16x16x32 fp8 tile mapping.**
- gfx1201 F2:
  - grid `[ceil(Mt/256), ceil(N/128), 1]`, block 256 (8×wave32), LDS 19,456 B (`gemm.rs:10485-10487`, `spec.rs:16-17`).
  - Per wave: 64 rows (rg) × 64 tokens (token half) (`gdn_epilogue.rs:20-24`). acc index `8·(i·4+tt)+e` (`fold.rs:41`).
- gfx942 twin:
  - Block 256 = **4×wave64**, same grid and LDS budget. Wave w owns rg=w (64 rows) × all 128 tokens: 4 row16 (i) × 8 token16 (tt) tiles × 4 acc = **128 acc VGPRs in arch v0–v127**. They stay out of AGPRs so the VALU ratio fold can touch them; MFMA C/D may be VGPR or AccVGPR per ACC_CD (CDNA3 ISA §7.1 p.41).
  - **A = weights.** Lane l reads Wf (bytes identical to gfx1201) at block offset `((l&15) + 16·((l>>4)&1))·16 + (l>>5)·8`. This follows from the repack layout (`mq4v2_fp8_fragment_repack.gfx12.hip:26-30`) and the MFMA A rule lane = row + 16·(k/8), K_L=8 (CDNA3 ISA §7.1.4.1 p.47). The offset is a per-lane constant, and each 512 B block stays fully coalesced.
  - **B = activations.** X8 staged in LDS: slot s (K64) × K32 step st × tt → 512 B block at `s·8192 + st·4096 + tt·512`, lane l at `+l·8` = `X8[tok tt·16+(l&15)][k0+st·32+8·(l>>4)..+7]`. That is 2×8192 B = the gfx1201 activation slot budget (`publish.rs` per PmBuilders scout: `slot·8192 + step·2048 + tt·256`).
  - **Ratios** stay at `16384 + plane·1024` (`fold.rs:28`). Lane reads one `ds_read_b128` per i at row `w·64 + i·16 + 4·(l>>4)` (the four rows it owns).
  - **Output.** Lane holds `D[4·(l>>4)+r][l&15]` (CDNA3 ISA §7.1.4.2 p.48: H=4, M_I=4) = 4 consecutive weight rows × 1 token. That is exactly the "row quad" `store_quad` already assumes (`epilogue.rs:36-38`). Ew: 16 regs/lane (4 i × 4 r). D: 8 regs/lane (tt).
  - **Per K128 per wave:** 4 u × 4 i × 8 tt = 128 `v_mfma_f32_16x16x32_fp8_fp8` (16 cycles each, CDNA3 ISA §7.1.2 p.42). Ratio fold per half = 64 `v_pk_mul_f32` (acc pair × ratio pair; VOP3P op 49, CDNA3 ISA §12.10 p.263).
  - **SiLU pairing.** The family-1 repack interleaves gate/up row16 blocks i=2p (gate) and 2p+1 (up) for the same hrow (`mq4v2_fp8_fragment_repack.gfx12.hip:18-21,60-67`). gfx1201 pairs acc g with g+32 (`epilogue.rs:151-157`). On gfx942 the same lane holds both (i=2p, 2p+1), so h = silu(g)·u in-lane with no cross-lane traffic.
  - **Register budget** `[UNVERIFIED occupancy]`: ~128 acc + 16 A (double-buffered; may live in AGPRs, MFMA src A/B may be AccVGPR per ACC bits, CDNA3 ISA §7.1 p.41) + 16 B + 16 ratio + ~12 addr. Under 256 arch; allocation granule 8 (CDNA3 ISA §3.6.4 p.12).
  - **Alternatives:** 8×wave64 (block 512, 64 acc/lane, 64 rows × 64 tokens per wave, matching gfx1201 wave ownership 1:1); or `v_mfma_f32_32x32x16_fp8_fp8` (same cycles per FLOP, 16 distinct ratio rows per lane per tile; worse fold).

**D3 — hipcc producers on wave64.**
- Every FP8 producer runs 256 threads with logical `lane=tid&31, wave=tid>>5`:
  - `fused_rmsnorm_mq_rotate.hip:149-152`
  - `fused_silu_mul_mq_rotate_fp8.gfx12.hip:48-50`
  - `gated_norm_mq_rotate_fp8.gfx12.hip:50-52`
  - `mq_rotate_x_fp8.gfx12.hip:40-42`
  - `pack_f16_to_fp8_mq4v2.gfx12.hip:211-213`
- Every cross-lane op stays inside a 32-lane group:
  - FWHT `ds_swizzle` bitmask mode (`0x041F..0x401F`) acts "on each group of 32 within a wavefront" (CDNA3 ISA §12.12 p.426).
  - `__shfl_xor` offsets ≤ 16 (`mq4v2_fp8_producer_pack.hip:56-68,218-222`).
  - The RMS `__shfl_down` ≤ 16 chain is consumed only at lanes 0/32 (`fused_rmsnorm_mq_rotate.hip:313-318`), so it depends only on its own 32 lanes.
- So on gfx942 (2 logical warps per hw wave) the arithmetic is bit-identical by reading. Tier C still decides `[UNVERIFIED]`.
- The FWHT butterfly result is independent of lane mapping. A native wave64 FWHT already exists (`kernels/src/mq_rotate_x.gfx942.hip:7-12`) if the 32-lane form under-occupies.
- The only FP8 deltas are D1a/D1d. There are no gfx1201 PM producers (all hipcc: `kernels.rs:1317-1427`), so gfx942 needs HIP twins only.

## 4. Kernel table — one 512-row chunk, dense DeltaNet layer then FA layer, default gfx1201 route (F2 admitted, N ≥ 512)

| # | kernel (symbol) | source path:line | launch site path:line + predicate | computes / numerics | gfx942 PM twin design | gfx942 HIP twin | notes / risks |
|---|---|---|---|---|---|---|---|
| 1 | `fused_rmsnorm_mq_rotate_awq_mq4v2_fp8_inreg_short_gfx12` (K=5120 ≤ 6144; P1A batched; AWQ). Siblings `_seq`, non-short, non-inreg, non-AWQ. | `kernels.rs:1317-1388` = `mq4v2_fp8_producer_pack.hip` + `fused_rmsnorm_mq_rotate.hip:100-104,447-499,542-579` | `prefill.rs:6185` (qkvza input, `attn_norm`) → `try_gfx12_fp8_stream_rmsnorm_prepared` `prefill.rs:653-678` → `gemv.rs:3982-4178` (selection `:4008-4049`). Predicate: `gfx12_fp8_stream_admitted` (`prefill.rs:633-641`) = consumer-takes-FP8 (`:605-627`) ∧ `fp8_stream_active` (`dispatch.rs:4519-4529`) ∧ IU4 producer returned `None`. | rms = rsqrt(mean x²+eps); `((x·w)·rms/awq)·s1` → FWHT-256 (local strides 1,2,4 + swizzle 8..128) → `·0.0625·s2`. Row amax → a = pow2 ≥ amax/224 (clamp exp [−31,9], shift). Codes `cvt_pk_fp8_f32(f·inv_a)` OCP RNE. Half sums from decoded codes. Grid [N], block 256, LDS (K+256)·4 (`gemv.rs:4122-4141`). | None (hipcc on gfx1201). Optional later. | `new: kernels/src/pm_cdna3/fused_rmsnorm_mq_rotate_fp8.gfx942.hip`: same TU structure per D3 + D1a (`·0.5f` at `producer_pack.hip:195-201`) + D1d. | Byte-identical planes are the acceptance test (oracle `crates/hipfire-runtime/examples/fp8stream_producer_oracle.rs`, per Levers scout). LDS (5120+256)·4 = 21.5 KiB per block fits 64 KiB (CDNA3 ISA §2.2.1 p.6). |
| 2 | `mq4v2_fp8_fragment_repack_gfx1201` (family 0, dims [qkv,z,β,α]) | `kernels/src/mq4v2_fp8_fragment_repack.gfx12.hip:110-195`; shipped `.hsaco` SHA `ef2f7796…` (`kernels.rs:4925-4928`) | `gemm.rs:10449-10462` inside `fp8_f2_row_launch` (`:10352`). Predicate `fp8_f2_row_active` (`gemm.rs:10263-10283`) + ABI checks (`:10368-10381`). Grid [ceil(Mt/256)·4, K/256], block 256. | qt44 136 B groups → Wf = E4M3(q−8) in the WMMA lane layout. Rw[b] = RN(e[b−1]/e[b]) over the nonzero-scale chain. Ew = last nonzero scale (`:26-33,164-194`). Zero word ignored (symmetric). | n/a (hipcc). | `new: kernels/src/pm_cdna3/mq4v2_fp8_fragment_repack.gfx942.hip`: **same source**. It is per-thread only; `v_perm_b32` exists (CDNA3 ISA §12.11 p.343). Wf/Rw/Ew byte-identical (D2 reads gfx1201 Wf with a lane-offset constant). | Keeping Wf identical allows a CPU byte diff against gfx1201 dumps. The ×4 product fold goes into D, not Ew (D1c). |
| 3 | `gemm_mq4g256v2_fp8_qkvzagdn_row_b1` (when GDN chunk scan ∧ `gdn_prep_fused_enabled`), else `gemm_mq4g256v2_fp8_qkvza_row_b1` | PM builder `crates/peacemaker-isa/src/kernels/fp8_gemm/` (`spec.rs:7-66`, `fold.rs:9-48`, `epilogue.rs:88-245`, `gdn_epilogue.rs:5-38`); bundle `kernels/gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco` SHA `90bcc18b…` (`kernels.rs:4885-4900`) | GDN: `prefill.rs:6582-6599` → `gemm.rs:10293-10329` (also needs dims `[10240,·,48,48]` and target sizes). Plain: `prefill.rs:6601-6617` → `gemm.rs:10827-10836`. Both need `gfx12_mq4v2_fp8_qkvza ∧ fp8_f2_row_active ∧ scale_mode==1`. Launch `gemm.rs:10485-10501`. | `v_wmma_f32_16x16x16_fp8_fp8`. Acc in units of the current half's scale; at each K128 half `acc *= Rw` (Row fold, first half skipped, `fold.rs:17-48`). Epilogue `RN(D·RN(Ew·C))` with no FMA contraction (`epilogue.rs:114-120`). Segment-split f32 stores per Mi (`:181-228`). GDN variant: q/k/v row tiles → conv1d(4)+SiLU+head RMS (+q_scale) → FP16 q/k/v via a 19×1040 B LDS ring (`gdn_epilogue.rs:5-38`); raw rows kept for the fixup. 96 B ABI (144 B GDN) (`spec.rs:47-66`). | **D2.** Builder: add `Arch::Gfx942` (absent today; `spec.rs:28-29` rejects non-gfx1201; no `gfx942` anywhere in `crates/peacemaker-isa/src`). New kloop emitting `v_mfma_f32_16x16x32_fp8_fp8`. Fold with `v_pk_mul_f32`. Epilogue with `D'=4D` (D1c). Kernarg ABI unchanged. GDN epilogue (blocked, see #4): one wave64 = two 128-ch heads (one per 32-lane half), per-token head-RMS reduce within the half (DPP/ds_swizzle), same ring; norm/conv goldens re-sliced from a gfx942 hipcc object (gdn_region.rs header per PmBuilders scout). | `new: kernels/src/pm_cdna3/gemm_mq4g256v2_fp8_row_mfma.gfx942.hip` (oracle / fallback: same Wf/Rw/Ew/X8/D ABI, D2 mapping). | MFMA→VALU read hazard for the fold `[UNVERIFIED wait states]`. `qkvzagdn` is unreachable on gfx942: `gdn` targets exist only when the chunk scan is admitted (`prefill.rs:7530-7543`), and the scan excludes gfx942 (`prefill.rs:14399`, `feature_flags.rs:963-964`). |
| 4 | `gdn_chunk_prep_fixup` | `kernels/src/gdn_chunk_prep_fixup.gfx1201.hip:8-174` | `prefill.rs:7558-7581` (`prep_fused` true) → `norm.rs:5703-5734`. Grid (max(⌈T/64⌉,⌈T/128⌉), 11), block 256. | Gates (softplus/exp prefix, sigmoid β) for C64 chunks; tile heads 128j..128j+2 conv+SiLU+RMS → FP16; conv ring update. Width-32 shuffles explicit (`:80`, `:138`). | n/a. | Same source compiles for gfx942 (explicit width 32, per-thread otherwise): `new: kernels/src/pm_cdna3/gdn_chunk_prep_fixup.gfx942.hip` (verbatim twin). | **Blocked** with #3 GDN (no gfx942 chunk scan). Not FP8 math. |
| 5 | `gated_norm_mq_rotate_awq_mq4v2_fp8_inreg_gfx12` (f32 x; `_xbf16` / non-inreg / non-AWQ siblings) | `gated_norm_mq_rotate_fp8.gfx12.hip:26-193`; consts selected `gemv.rs:5326-5359` | `prefill.rs:7268` → `try_gfx12_fp8_stream_gdn_prepared` `prefill.rs:962-979` → `gemv.rs:5274-5465`. Predicate `gfx12_fp8_stream_gdn_admitted` (`prefill.rs:944-955`: head_dim 128, n_heads·hd==k, Residual takes FP8). inreg iff `fp8_prod_inreg ∧ (k+8)·4 ≤ 65536` (`gemv.rs:5325`). | Per head: rms over 128 → `x·inv_rms·w·silu(z)/awq` staged in LDS → FWHT → pack (same recipe as #1). LDS (K+8)·4 (`gemv.rs:5441`). | n/a. | `new: kernels/src/pm_cdna3/gated_norm_mq_rotate_fp8.gfx942.hip` (D3 + D1a/D1d). | 27B GDN out K = 48·128 = 6144 → 24.6 KiB LDS. |
| 6 | #2 repack + `gemm_mq4g256v2_fp8_add_row_b1` (LA `wo`) | as #2/#3 | `prefill.rs:7370-7371` → `dispatch_batched_fp8_lloyd_epilogue` `prefill.rs:1575-1605` → `gemm.rs:36604-36637` (`gfx12_mq4v2_fp8_resid ∧ fp8_f2_row_active`). | ADD epilogue: old Y loaded, one f32 add per element in scalar operand order (`epilogue.rs:36-38`). | D2, `add`. | as #3 (`add` entry). | — |
| 7 | #1 producer on `ffn_norm` | as #1 | `prefill.rs:8058` (GateUp consumer; `prefill.rs:621-623`) | as #1 | n/a | as #1 | — |
| 8 | #2 repack (family 1, gate/up paired) + `gemm_mq4g256v2_fp8_silu_row_b1_bf16` | as #2/#3; silu at `epilogue.rs:142-180`, bf16 pack `:9-34` | `prefill.rs:8354-8363` (`silu_h = fp8_silu_h_active ∧ w_down MQ4G256V2 ∧ Residual FP8`) → `gemm.rs:34433-34440` (`silu_h_requested ∧ fp8_f2_row_active(..,true)`). Symbol `gemm.rs:10399`. | h = silu(g)·u per (row, token) from paired accumulators. RNE bf16 (hip_bfloat16 bitwise, NaN/Inf guarded, integer ops only). Y1 untouched (`gemm.rs:10434`). | D2 SiLU pairing (i=2p gate, 2p+1 up in the same lane); SiLU region re-emitted for gfx9 VALU; integer bf16 pack is portable. | as #3 (`silu`/`silu_bf16` entries). | — |
| 9 | `fused_silu_mul_mq_rotate_awq_hin_bf16_fp8_gfx12` (h-in, bf16 h, inreg) | `kernels.rs:1415-1427` + `fused_silu_mul_mq_rotate_fp8.gfx12.hip:24-201` (HIN macro `:80-93`) | `prefill.rs:8589` → `try_gfx12_fp8_stream_silu_prepared` `prefill.rs:684-700` (`h_ready` → `fused_silu_hin_rotate_mq_fp8_gfx12_batched`) → `gemv.rs:4248-4452` (table `:4291-4332`). HIN needs inreg ∧ no x_rot (`gemv.rs:4270-4277`). | Reads h (bf16 widened exactly) /awq ·s1 → FWHT → pack. Grid [N], block 256, no dyn LDS (`gemv.rs:4410-4415`). | n/a | `new: kernels/src/pm_cdna3/fused_silu_mul_mq_rotate_fp8.gfx942.hip` (+ HIN/bf16 macros), D3 + D1a/D1d. | K = 17408 (27B FFN) → 9 groups/lane in registers (`fp8_prod_inreg` k ≤ 17408). VGPR pressure on wave64 `[UNVERIFIED]`. |
| 10 | #2 repack + `gemm_mq4g256v2_fp8_add_row_b1` (`w_down`) | as #6 | `prefill.rs:8625-8626` → `prefill.rs:1591` → `gemm.rs:36632-36637` | as #6 | D2 `add` | as #3 | — |
| 11 | #1 producer on FA `attn_norm` | as #1 | `prefill.rs:8872` (Qkv consumer) | as #1 | n/a | as #1 | — |
| 12 | #2 repack + `gemm_mq4g256v2_fp8_qkv_row_b1` | as #3 | `prefill.rs:9152-9177` → `gemm.rs:11276-11283` (`gfx12_mq4v2_fp8_qkv ∧ fp8_f2_row_active ∧ scale_mode==1`) | 3-segment f32 stores (`epilogue.rs:181-228`) | D2 `qkv` | as #3 | FA attention itself is out of R1. |
| 13 | `sigmoid_mul_rotate_x_mq_awq_mq4v2_fp8_inreg_gil_gfx12` (gate read in place) + siblings | `mq_rotate_x_fp8.gfx12.hip:19-194` (`_GATE_IL` `:47-50`); consts `gemv.rs:6176-6237` | `prefill.rs:9682-9686` → `try_gfx12_fp8_stream_sigmoid_prepared` `prefill.rs:833-848` → `gemv.rs:6138-6347`. Predicate: Residual takes FP8 ∧ `fp8_stream_active`. | `attn·sigmoid(gate)/awq·s1` → FWHT → pack. Grid [N], block 256. | n/a | `new: kernels/src/pm_cdna3/mq_rotate_x_fp8.gfx942.hip` (D3 + D1a/D1d). | — |
| 14 | #2 repack + `gemm_mq4g256v2_fp8_add_row_b1` (FA `wo`) | as #6 | `prefill.rs:9742-9743` | as #6 | D2 `add` | as #3 | — |
| 15 | FA-layer FFN = #7–#10 | as above | `prefill.rs:10830` (producer), `:11131` (gate/up), `:11349` (SwiGLU producer), `:11384` (down) | as #7–#10 | as #7–#10 | as #7–#10 | — |
| 16 | hipcc v2 staged tile (256 ≤ N < 512, or `HIPFIRE_G12_FP8_ISA=0`): `gemm_{qkvza,qkv}_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201_symfold`, `gemm_mq4g256v2_residual_wmma_fp8_v2_b128x128_gfx1201_symfold`, `gemm_gate_up_…_v2_b128x128_gfx1201_silu_h[_bf16]` | `gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip` (geometry/accumulation `:12-35`, symfold `:68-73,111-127,889-903`, family macros `:187-208`); consts `kernels.rs:5283-5423` | qkvza `gemm.rs:10843-10881`; qkv `gemm.rs:11291-11329`; gate/up `gemm.rs:34448-34504` (silu_h `:34470-34481`); residual `gemm.rs:36646-36684`. Predicate: family flag ∧ v2 ∧ N ≥ 256 ∧ F2 not admitted. LDS formula `gemm.rs:10551-10558`. | WMMA fp8, weights expanded in-kernel through the v_perm LUT. Symfold `T=fma(s_h,P,T)` per 128-half. Output `y = row_scale·s_final·acc` (`:17`). | n/a (hipcc on gfx1201). | `new: kernels/src/pm_cdna3/gemm_mq4g256v2_fp8_mfma.gfx942.hip`: one TU, same family macros/ABI. 4×wave64 per 128×128 tile, MFMA 16x16x32 (A = in-kernel LUT expand, written in D2 lane order into LDS; B = X8). `s_h' = 4·s_h` at the `lds_SZ` store (D1c). LDS rows padded like gfx12 (`vbk+8`). | The gfx12 TU `#error`s off gfx1200/1201 (`:107-110,154-156`), so a new TU is mandatory. F2 vs v2 bytes already differ on gfx1201 (ratio vs fma fold). Open: route 64 ≤ N < 512 to repack+F2-twin with masked tiles instead. |
| 17 | hipcc s2bt8 / bt12 / bt8 / bt4 (64 ≤ N < 256): `gemm_{qkvza,qkv}_mq4g256v2_wmma_fp8_gfx12_{s2bt8,bt12,bt8,bt4}`, `gemm_gate_up_…_gfx12_*`, `gemm_mq4g256v2_residual_wmma_fp8_gfx12_*` | same TU; consts per PmBuilders scout `kernels.rs:5152-5279` | qkvza `gemm.rs:10905-10936`; qkv `gemm.rs:11353-11384`; gate/up `gemm.rs:34528-34559`; residual `gemm.rs:36708-36739` | One wave32 per 16 rows × 16·BV tokens, no LDS (`:12-35`). Non-symfold (asymmetric) form: `fma(s_h,P,T); fma(z_h,S_h,T)` (`:905-925`). | n/a | Same new TU as #16, `BT` macros. One wave64 = 16 rows × 16·BV tokens, MFMA 16x16x32 (half the instruction count of K16). | Symmetric pin → symfold only on v2; the bt arms read half_sums (D1d needed). |
| 18 | `pack_f32_to_fp8_mq4v2_gfx12` (standalone pack; only when FP8_STREAM is off, or via the unprepared route entries) | `pack_f16_to_fp8_mq4v2.gfx12.hip:184-287` | `dispatch.rs:4065-4137` → `scratch.rs:1763-1793`. Callers: `gemm.rs:10597` (qkvza unprepared), route entries `gemm.rs:11528-11539`, `:34281-34291`, `:36883-36892` (batch ≥ 64), Lloyd `:35988-36025`. | Same scale/encode recipe as #1. Grid [N], block 256, 32 B LDS. | n/a | `new: kernels/src/pm_cdna3/pack_f32_to_fp8_mq4v2.gfx942.hip` (D1a; software half-sum decoder unchanged). | Default route does not launch it (FP8_STREAM on). |
| 19 | Non-HIN SwiGLU FP8 producer (`fused_silu_mul_mq_rotate_awq_mq4v2_fp8_inreg_gfx12`) + separate gate/up planes, when `fp8_silu_h_active` is false | `kernels.rs:1389-1414`; `fused_silu_mul_mq_rotate_fp8.gfx12.hip` | `prefill.rs:700-` → `gemv.rs:4184-4196`; gate/up via #16/#17 non-silu symbols | silu(g)·u computed in the producer | n/a | #9 twin without HIN | Off-default (`HIPFIRE_FP8_SILU_H=0` or `FP8_PROD_INREG=0`). |
| 20 | Lloyd qt=52 LUT twins (`*_lut` consts; `gemm_*_fp8_*_lloyd` launchers) | `gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:84-105,159-166` | `prefill.rs:6487-6557`, `:8287-8340`, `:9083-9142`, `:1424-1442`, `:1543-1554`, `:15078-15095` | Codebook `E4M3(L−7.5)` as 4 kernarg dwords | n/a | #16 TU with `HIPFIRE_FP8_LUT_ARG` | **Off-route** for the qt44 pin. Codebook bytes must be checked for 0x80 before FNUZ reinterpretation `[UNVERIFIED]`. |

Launch order inside a DeltaNet layer: 1 → 2 → 3 → 4 → (GDN chunk scan, out of R1) → 5 → 6 → 7 → 8 → 9 → 10 (`prefill.rs:14450-14501`). FA layer: 11 → 12 → (attention) → 13 → 14 → 15 (`prefill.rs:14503-14540`). Every F2 launch is preceded by its own repack, a single transaction on the stream (`gemm.rs:10332-10335`).

## 5. Qwen4 Flash-Next trunk on gfx1201 (IU4=0): no FP8 arm

| op | live path | FP8 arm |
|---|---|---|
| GDN/QSA in/out projections | `project_trunk` / `project_trunk_masked` (`crates/hipfire-dispatch/src/pipeline/layer_ops.rs:416,430`) → `project_weights_inner` (`layer_ops.rs:177-382`): MQ4/MQ6 shared-rotation F16 `gemm_mq{4,6}g256v2_xf16` (`:187-251`); BF16 `gemm_bf16_xf32_f16_wmma_qwen4` (`:262-281`); few-row (`:286-315`); else `project_rotated` (`layer_ops.rs:590-687`) → `gemm_mq4g256v2` → `gemm_mq4g256v2_residual_wmma` F16 (`gemm.rs:38965-38989,38949-38952,36972-37132`). Never the gfx12 FP8 intercept (`gemm.rs:36796`/`:36883`). | none |
| shared expert (BF16 per artifact) | `shared_gate_up` / `shared_down` (`crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:492,708`) | none |
| lm_head | `crates/hipfire-arch-qwen4/src/program.rs:583` → `project_weight` | none |
| HC | `hyper_*` ops (`layer_ops.rs:689-711`), no FP8 GEMM | none |
| chunk | `qwen4_prefill_chunk_default` gfx1201 = 4096 (`crates/hipfire-arch-qwen4/src/gpu_forward.rs:83-86`) | — |

Evidence: no `mq4v2_fp8` / `_fp8_gfx12` / `fp8_f2` / `prepare_mq4v2_fp8` call exists in `crates/hipfire-arch-qwen4` or `crates/hipfire-dispatch`. The only `fp8` hits are QSA FP8 KV attention (`crates/hipfire-dispatch/src/families/attention.rs:2611`) and Lloyd panic texts (`families/fused_qkv.rs:28-30`, `families/gemm.rs:95-98`). gfx942 mirror for Qwen4 R1: **nothing to mirror**. The plan's BF16 trunk → `GemmBf16Mfma` (`subq4/review/GFX942-FP8-FN-PLAN.md:47,60,62`) is the gfx942 trunk, and a native-FP8 trunk arm would be new design, not a mirror.

## 6. gfx942 resolution sites (every `arch=="gfx1201"` test the mirror must widen)

- Flag defaults: `feature_flags.rs:946-953,959-960,975-976`; `feature_flags.rs:1282`.
- Live gates: `dispatch.rs:4523`; `gemm.rs:10265`; producer arch checks `gemv.rs:4264,5296,6154`; consumer arch checks `gemm.rs:10587-10589,36614-36618`.
- Prefill policy: chunk512 `prefill.rs:1693-1698`; widened route `prefill.rs:1961-1967`; scratch sizing `prefill.rs:2251,2361`.
- Chunk-scan coupling: `prefill.rs:14399,14411-14414`, which also requires FP8 KV (`kv_cache.quant_fp8`, workstream R-KV).

## 7. Summary

**Portable as-is on wave64 (by reading; Tier C still required):**
- Fragment repack #2: per-thread code, byte-identical Wf/Rw/Ew.
- `gdn_chunk_prep_fixup` #4: explicit width 32.
- All producer arithmetic (#1, #5, #9, #13, #18): FWHT, RMS, scale, amax reductions (D3).
- F2 integer bf16 pack (`epilogue.rs:9-34`).
- Every scratch byte layout (§2).

**Needs a twin:**
- Producers: one `·0.5f` at the cvt and the ×2 half-sum fix (D1a/D1d). HIP only.
- F2 Row PM bundle (#3/#6/#8/#10/#12/#14): `Arch::Gfx942` builder, MFMA 16x16x32 kloop, `v_pk_mul_f32` fold, `D'=4D`, plus a HIP oracle twin.
- hipcc v2/bt consumers (#16/#17): new MFMA TU, `s_h'=4s_h`.
- `.hsaco` builds for gfx942 (repack, fixup).

**Blocked:**
- `qkvzagdn` F2 epilogue + fixup (#3 GDN, #4): no gfx942 GDN chunk scan (`prefill.rs:14399`). Additionally needs gfx942 region goldens.
- Qwen4 trunk FP8: no gfx1201 source to mirror.

**Open / `[UNVERIFIED]`:**
1. `v_cvt_pk_fp8_f32` on gfx942 for values that round to −0: 0x00 expected. Also the ISA table lists CVT_PK_FP8_F32 as "ignores: clamp" (CDNA3 ISA §7.2 p.50), contradicting "clamp=saturate". This is irrelevant while \|arg\| ≤ 224.
2. MFMA fp8 internal summation order. Outputs will not be byte-identical to gfx1201 WMMA even with identical operands. A gfx942 KLD row is mandatory (plan `:317`).
3. MFMA-dst → VALU-read wait states for the ratio fold and epilogue (16x16x32 fp8, 16 cycles).
4. Bank behavior of 64-lane `ds_read_b64` / `ds_read_b128` broadcast for the D2 LDS layout.
5. `v_pk_mul_f32` per-element rounding equals `v_mul_f32` (needed for the fold to reproduce `RN(acc·ratio)`).
6. Occupancy at ~180 arch VGPRs × 4 waves/CTA; whether 8×wave64 (64 acc) beats 4×wave64 (128 acc).
7. In-register producers at K=17408 (#9): 72 held floats/lane on wave64 VGPR budget.
8. gfx942 chunk rung (512 on gfx1201 is measured, `prefill.rs:1679-1682`), not inherited.
9. Whether 64 ≤ N < 512 should use masked F2 twin tiles instead of a v2/bt MFMA TU. This is a decision for the PM lead.
