# What else the XDNA2 NPU could do for hipfire models — lever brainstorm

> **Historical research note: superseded by the 0.4.2 serving route.** This file records a lever brainstorm (research only, nothing run) for what else the NPU could do for hipfire models from the 2026-10 NPU research cycle. It is not current guidance, and any figure in it is a dated fixture-bound measurement or an explicitly labelled historical estimate, not a product claim. The shipped 0.4.2 NPU route is the Flash-Next gfx1151 trunk input projection described in [npu-trunk.md](npu-trunk.md): `HIPFIRE_QWEN4_NPU_TRUNK=off|auto|on|emulate` (default `auto`, effectively `off` on every architecture except gfx1151), calibration schema 4, BF16-direct only. There is **no I32 serving contract** (no int32 partial, combine or fold exists in the product route). The out-projection program exists but is **not wired** into serving; `qsa.o` and the GDN o-projection are **not shipped**. Environment-variable names and flag syntax below are historical proposals or experiment switches, not supported configuration.
>
> Candidate levers here are unbuilt proposals. Env names written as "proposed" (`HIPFIRE_IU4_NPU_EC`, `HIPFIRE_QWEN4_SHARED_NPU`) were never implemented. `HIPFIRE_QWEN4_TRUNK_IU4` is a real, separate GPU IU4-trunk developer switch and is unrelated to the NPU route.

2026-10-03. Research only; nothing here was run. Every number is tagged **M** (measured, with
its source) or **E** (estimate / arithmetic on measured inputs). Paths: `npu/` = this repo; `W:` =
the 0.4.1 hipfire beta checkout (hipfire beta f7960b4e8, read only); `R:` = the 0.4.1 release-workspace measurement reports (not included in this repository).

## 0. The three walls every candidate hits (measured)

| wall | value | source |
|---|---|---|
| NPU↔DRAM port | ≈55 GB/s read, ≈55 write, 34+34 mixed; GPU gets 240 GB/s from the same LPDDR5X | `npu/report.md:329-331,883,1027-1037` **M** |
| GPU slowdown under NPU traffic | +2.2 / +5.9 / +11.3 % at 5 / 16 / 47 GB/s (drift-flagged) | `npu/docs/dense27b.md:140-148` **M** |
| per-launch floor | ≈215 µs full TXN, ≈50 µs lean TXN + 20–25 µs chained submit; ring handoff p50 4.5 µs; grouped experts 113–234 µs **per expert** at M ≤ 512 | `npu/report.md:599-602,622-627,646-653` **M** |
| int8 GEMM, exact | 25–33 TOPS at M=4096 dense shapes (lean/ring), 26.9–27.5 at dense-27B shapes, ≈5 TOPS at M=160 experts | `npu/report.md:707-713`, `dense27b.md:90-93` **M** |
| numerics | int32 core exact; vector FP = bf16-in / fp32-acc, no exact fp32 FMA; exact copy of the GPU fold costs 4.2–5.5× the GEMM | `dense27b.md:107-134` **M (model), silicon-exact partials** |

Consequences used below (E, arithmetic): (a) anything weight-streaming at M=1 is **≥4.4× slower on the NPU than on
the GPU** and adds latency if it sits on the token chain; (b) decode-scale work can afford **≤ ~10 NPU launches per
token** (30 ms AR step at 33 tok/s, `W:CHANGELOG.md:395-420` **M**) — so per-layer decode roles are dead on arrival,
and per-expert roles are dead at any M; (c) a GEMM-class role only pays when the NPU's job is **≤ ~20–60 % of the
ops of the GPU work it overlaps** (NPU 25 vs GPU 75 IU4 TOPS, `dense27b.md:31` **M**) *or* it sits off the GPU's
critical path; (d) the correction/output bytes the NPU writes, not its MACs, are usually the binding term.

Denominators: Flash-Next pp8192 IU4 step **4262 ms** (base, `R:fn-trunk-sel/report.md:20` **M**; buckets
`R:fn-halo-2500.md:39-66` **M**: MoE GU 486, down 324, glue 238, MQ6 trunk 1421, HC-down 413, shared GU 103 +
down 128 + selector 13, QSA attn 358, selection 73, GDN 218); dense Qwen3.8-27B pp8192 **6811 ms**, 5287 of it
IU4 GEMM (`dense27b.md:20-31` **M**); decode AR ≈30 ms/token, MTP 55–60 tok/s (`W:CHANGELOG.md:395-420` **M**).
Quality: Flash-Next IU4-route WT2 KLD **0.1027**, gate ≤ 0.1177; trunk A4 per family (`R:fn-trunk-sel/report.md:7-15`
**M**): gdn.qkv +0.0135 (act +0.0084), gdn.z/a/b +0.0066 (act +0.0061), gdn.out +0.0226 (act +0.0156),
qsa.q/k/v/idx +0.0085 (act +0.0063), qsa.o +0.0424 (act +0.0432), all +0.0790 (weights +0.0139, act +0.0651); "all"
saves 560 ms/step. Dense 27B: A4 0.070 vs native fp8 0.042 (`R:fn-requant/attribution.md:215` **M**). Flash-Next
F16-expert route 0.066 → IU4 experts 0.1027: **the expert A4 costs +0.037**, the largest A4 term anywhere
(`R:fn-trunk-iu4/report.md:30-32` **M**), and 43–52 % of Flash-Next KLD is routing flips (`attribution.md:150-163` **M**).

## 1. Candidates

### C1. Block-selective A8 "low half" for IU4 — residual GEMM on the NPU (the proposed error-correction idea, sized)

**Mechanism.** The A4 producer already forms `x' = FWHT256(x/awq)`, per-128 `amax`, the MSE-clip grid
`d_j = (amax/7)·½·(1+j/7)` and `q = clamp(rint(x'/d), -8, 7)` (`W:kernels/src/block_i4_128_quant.hip:20-33,53-55,
77-137`, `W:kernels/src/mq_rotate_x_i4.hip:5-25`). Clip ratios down to 0.5 mean clipped elements carry residuals up to
`amax/2`: the residual `r' = x' − d·q` is dominated by the blocks whose `d` is large, i.e. the 256-blocks that contain
outlier channels/heads — and those are static per tensor. The correction is `Y_corr = Σ_{e∈S} sc_e · d_lo,e · C_lo,e`,
`C_lo,e = Σ_j W'[n][128e+j]·q_lo[t][128e+j]` over a **static block set S** (|S| ≈ 20 % of K), with `q_lo` the int4
re-quant of `r'` (second plane, `d_lo = max|r'|/7`, emitted by the GPU producer: +1 nibble plane for S only). This is
exactly hipfire's two-plane A8 integer identity restricted to S (`R:fn-wmo/w4a8-moe.md:60-71,95-102` **M design**),
and the NPU core holds it exactly (`npu/peacemaker-npu/src/kernels/iu4.rs:1-25,121` — silicon-exact partials,
`dense27b.md:69-77` **M**).

**Numerics.** The correction is a *new* term: it needs **determinism, not parity with the GPU fold**. Fold it on the
NPU with the all-integer epoch fold (`S·D` ≤ 32 bit, `C·(S·D)` into int64, one final RNE, `dense27b.md:189-207`
**E**: 1.15–1.33× the GEMM) or bf16/fp32-acc (deterministic on a fixed program; exact-RNE-ness irrelevant). The GPU
adds `Y_corr` (bf16 or f32) with one add in the IU4 epilogue → new PM-builder epilogue variant → **non-bit-exact,
default-OFF, named kill switch** (proposed name only, never implemented and not a supported variable: `HIPFIRE_IU4_NPU_EC`), like `HIPFIRE_QWEN4_TRUNK_IU4`.

**Why the NPU and not the GPU.** The GPU can run the same S-blocks itself as extra K epochs at **+|S|/K ≈ +20 % of the
corrected GEMM's time** (`A8Split128` machinery exists, `w4a8-moe.md:75-85`). The NPU version instead costs the GPU
the contention of a near-saturated NPU port (≈−10 %) plus reading `Y_corr` (2 B/output; hidden under compute when
the GEMM is compute-bound). So the NPU's *incremental* value is ≈ 10 % of the corrected GEMMs' time, not 20 %.

**Honest ceiling (E).**
- Dense 27B: corrected GEMMs = 5287 ms → GPU-alone selective A8 **+1057 ms**; NPU route **+420…580 ms** (NPU busy
  ≈3.2–3.8 s per step at 45–55 GB/s). NPU advantage ≈ **480–640 ms/step (7–9 %)** — but the whole thing is a
  *quality purchase*: step 6811 → ≈7300 ms for KLD 0.070 → 0.070 − f·0.028, `f` = fraction of A4 error energy inside S.
- Flash-Next trunk: corrected GEMMs ≈ 860 ms (1421 − 560) → GPU-alone +172 ms, NPU +86 ms; advantage ≈ **40–90 ms (1–2 %)**.
  Budget math with `f = 0.5` on gdn.qkv + gdn.z/a/b: 0.1027 + 0.0051 + 0.0005 + 0.5·(0.0084+0.0061) = **0.1155 ≤ 0.1177**
  → unlocks both families: 149 + 96 = 245 ms saved vs 149 today = **+96 ms/step**, ≈ +70 after contention; with
  `f = 0.8`, add qsa.q/k/v/idx: 0.1146 → ≈ **+170 ms (4 %)**. "all" never fits (0.1165 weights-only + any act residual
  > 0.1177). Assumes family Δs add (unmeasured).
- **Per-GEMM feasibility is port-bound, not MAC-bound (E):** Flash-Next gdn.qkv (8192×10240, S = 4 of 20 blocks):
  reads ≈ 90 MB (V9 resident B) + writes 168 MB bf16 → ≈5 ms vs GPU 5.7 ms; dense gate_up (N=34816, S = 8 of 40):
  ≈570 MB write + 230–570 MB read → 25–34 ms vs GPU 38.5 ms. Fits only with wide waves; no slack for a bigger S.
- KLD recovery per budget: first order `ΔKLD_recovered ≈ f(S)·ΔKLD_act` (output-error energy ∝ KLD; Flash-Next is
  sub-additive through routing flips, `attribution.md:202`, so treat as an upper bound). `f(20 % blocks)` ∈ [0.2
  (white residual) … 0.8 (outlier-concentrated)] **E — this spread is the whole lever; §3 measures it on real data.**

**Cost.** NPU: V9/G80 epoch-batched int8 GEMM over an S-slice (exists as `EpochGemm` host packing,
`iu4.rs:20-25`), new integer fold epilogue on core, ring slot per GEMM (~240–336 per prefill step: ≈60 µs each,
≈15–20 ms/step **E**). GPU: producer second plane for S, epilogue add, NPU-layout A/B packing kernels
(`levers-joint.md:282`). Memory: W' S-slice copy in NPU pack order (20 % of trunk weights).

**Risks.** Port saturation → the measured −10…−11 % GPU derate (and the sub-linear power/thermal component,
`dense27b.md:146-148`) lands on every corrected GEMM; `Y_corr` read not hidden on the narrow-N GEMMs; deferred-add
variants (C3) change where the model sees the correction.

**Kill.** §3 experiment, stage 1: if the static top-20 % blocks capture `< 0.5` of Σ‖W' r'‖² on gdn.qkv/z and
qsa.q/k/v (Flash-Next) or `< 0.6` on qkv/gate_up (dense), the NPU form is dead (the GPU-only forms in C2 may survive).

**Prior attempt.** None for EC. Plain NPU IU4 offload under the bit-exact fold: killed (`dense27b.md:171-183`).

### C2. Low-rank corrections `U(V·r')` — finding: they move nothing to the NPU

**Mechanism.** Input-side: `V (r×K)` projects the residual, `U (N×r)` expands; ops `r/N + r/K` of the GEMM (≈5–10 %
at r=128, K=2560). Output-side on the block set S: precompute `Vᵀ W'_S (r×128|S|)`, so the per-token work is a tiny
`r×512` GEMV plus `U p` at `r/K` ≈ 5 %. In both forms the expensive part is already ≤ 10 % of the GEMM → **the GPU
does it for less than the NPU's contention cost**; the NPU has no role.

**Ceiling (E).** Capture fraction = energy of `W' r'` in the top-r singular subspace of `W'` (input form) or of
`W'_S` (output form). For `r = 128` of K = 2560 and typical LLM spectra: **0.1–0.3** (input form), **0.5–0.8 of the
S-captured energy** (output form, `W'_S` has rank ≤ 512). Unmeasured; §3 produces both curves from the GPTQ MQ4 trunk
payloads (`R:fn-trunk-iu4/payloads/`, 240 tensors) and the dense `~/.hipfire/models/qwen3.8-27b.mq4*` weights.

**Verdict.** Not an NPU lever. Hand "output-side low-rank selective-A8 correction" to GPU-side work as the cheapest
quality lever if §3 shows `f(S)` high and `W'_S` low-rank: ≈ +5 % GPU time on the corrected GEMMs for most of C1's
quality gain, no NPU, no port traffic.

### C3. Deferred-add EC for residual writers (Flash-Next gdn.out, qsa.o) — full A8 low half, hidden behind the MoE block

*Historical candidate: no NPU path for `qsa.o` or the GDN o-projection shipped in 0.4.2; the shipped trunk covers input projections only ([npu-trunk.md](npu-trunk.md)).*

**Mechanism.** `gdn.out`/`qsa.o` outputs enter the residual (HC write); their next *unavoidable* consumer after the
MoE block is the next layer's norm — ≈22 ms/layer of slack (`fn-halo-2500.md:41-43` **M**, 1048 ms/48). The NPU
computes the **full** low half (all K blocks) for these two tensors: 36·2·8192·2560·6144 + 12·2·8192·2560·6144 =
**12.1 TOP/step → ≈480 ms NPU, ≈10 ms/layer (E)**; the GPU adds `Y_corr` into the residual stream *after* the MoE
block. The MoE of the same layer sees the uncorrected residual; everything downstream (47 layers) sees the corrected one.

**Ceiling (E).** Recovers `η·(0.0432 + 0.0156)` of act error, `η` = deferral efficiency (unknown; one-layer leakage
suggests 0.7–0.9). With η = 0.9: qsa.o + gdn.out → 0.1027 + (−0.0008 + 0.0070) + 0.1·0.0588 = **0.1148 ≤ 0.1177** →
saves ≈61 (qsa.o) + gdn.out's unresolved 0…150 ms (`fn-trunk-sel/report.md:24-25`: unstable samples) ≈ **60–200
ms/step (1.5–4.5 %)**. With η = 0.7 it is over budget (0.1266). Fragile; the HC 4-stream mixing makes "add it later" a
non-standard model edit that must be modelled before any kernel.

**Why NPU.** This is the one EC where the NPU's time is *not* bounded by the GPU GEMM it corrects: 10 ms of NPU work
against 22 ms of GPU work it does not touch. Port: ≈ 100 MB/layer → ≈1 GB/s average, ≈ +1 % GPU (E).

**Kill.** §3 stage 2 (twin) with a deferred-add arm: η < 0.8 → dead.

### C4. Expert-side A8 correction — the biggest A4 target, unreachable by the NPU

**Fact.** Expert IU4 costs +0.037 KLD (0.066 → 0.1027), more than the entire trunk A4 budget. Correcting it is worth
more than any trunk EC. **But** the correction is per expert (W_e differs per slot): 1562 expert runs/layer
(`R:fn-wmo/moe-astat.md:19` **M**) at the measured 113–234 µs/expert (`report.md:646-653` **M**) = 180–360 ms/layer
against a 22 ms MoE phase. Dead by two orders of magnitude at any geometry; a shared-subspace trick (one `V` for all
experts) collapses to the block selector, whose per-expert part `W_e[:,S]·r'_S` is again per expert.

**Verdict.** Not an NPU lever. GPU-only: block-selective A8 on experts at +|S|/K ≈ +20 % of 810 ms = **+160 ms/step**
for `f(S)·0.037` KLD — probably the best quality-per-ms EC anywhere; hand to LeadW4A8's A8 machinery as a "partial A8"
mode. §3 stage 1 runs the same curves on the MoE input (`capture/L*.gin.bin` is the GDN block input, not the MoE
input; a MoE-input capture is needed — the FN_TRUNK_CAPTURE hook pattern, `R:fn-trunk-iu4/capture.sh`).

### C5. Shared expert (Flash-Next) as W8A8 on the NPU, concurrent with the routed experts

**Mechanism.** The shared expert is a dense per-token GEMM pair — gate/up `8192×(2×640)×2560`, down
`8192×2560×640` — run in BF16 on the GPU (`W:crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:155-242,
291-303`), 103 + 128 ms/step (`fn-halo-2500.md:47,49` **M**), and its only consumer is the combine/HC write after
the routed down. Its gate_up shape is **the NPU's measured best regime** (gate_up 4096×1280×2560: 28.7–30.2 TOPS,
`report.md:707-713` **M**); M = 8192 is better still (29.1–29.4 at 8192-row shapes, `report.md:438,453` **M**). The NPU
runs it in W8A8 (W8 = the Q8_0 load-time copy decode already keeps, `W:CHANGELOG.md:378-380` **M**; A8 = per-128
int8 of `x_rot`, the `A8Split128` producer's domain) with a deterministic NPU fold → bf16 g,u → GPU
`shared_expert_activation_bf16_f32` + A8 of the activation → NPU down → GPU scaled add + HC write
(`qt44_qt53_prefill.rs:244-289,416-460`).

**Why NPU.** Off the GPU critical path (parallel to the 22 ms/layer routed phase), no expert-shape intercept (one
GEMM per projection per layer), measured rates transfer directly, port duty ≈ 94 MB/layer ≈ 1 GB/s → GPU derate
≪ +1 % (E, below the 5 GB/s point).

**Ceiling (E).** NPU time per layer: 54 GOP / 28 TOPS + 27 / 24 ≈ **3 ms** against 22 ms slack. Saves up to 231 ms
minus GPU glue (activation kernel stays; scaled-add/HC write pass ≈ 0.35 ms/layer ≈ 17 ms/step; two A8 producers
≈ 0.4 ms/layer unless A8 routed experts already produce them) ≈ **180–200 ms/step (4–5 %)**. 2 kicks + 2 waits per
layer = 192 ring ops ≈ 1 ms. GPU-alone alternative: shared expert as IU4 on the GPU ≈ −150 ms at A4 quality; so the
NPU's advantage is ≈ 30–50 ms **plus A8 instead of A4 quality**; against the exact BF16 baseline it is the full 180–200.

**Numerics.** W8A8 is BF16-class error (int8 per-128 ≈ 2⁻⁸ relative, bf16 2⁻⁹); KLD cost **E ≤ +0.001–0.002**,
unmeasured. Not bit-exact → opt-in kill switch (proposed name only, never implemented and not a supported variable: `HIPFIRE_QWEN4_SHARED_NPU`). A bf16-VMAC fallback (exact
precision class, accumulation order only) exists in the ISA (`dense27b.md:114`) but its rate on our emitter is
unmeasured; use it only if the A8 gate fails.

**Cost.** NPU: W8A8 fold epilogue (int32 → `sc·d` → bf16; the `S·D` integer fold is 1.2–1.3× the GEMM, E), ring
slots sized for the two shapes, NPU-order packing of the Q8 copy at load (+0 memory, repack of 236 MB). GPU: A8
producer calls, epilogue reading NPU output, dispatch-graph change (kick after norm/rotate, wait before combine).

**Risks.** The decode Q8_0 copy is in the natural basis while `x_rot` is FWHT-256 — pick one basis (W8 in the rotated
basis requires a load-time requant from BF16, 236 MB, still +0 if it replaces the decode copy; **[INFERENCE]** the
decode path can read the rotated copy via its own rotated x). The router+selector+gate/up fused launch
(`qt44_qt53_prefill.rs:167-180`) loses its fusion → the router/selector keep a small GPU launch.

**Kill.** (1) Offline: the twin with a W8A8 shared-expert arm (`R:fn-requant/attr/oracle.py` already decodes Q8 and
MQ formats and runs F32 math) — KLD rise > 0.003 kills. (2) One ≤90 s window: V9/G80 at 8192×1280×2560 and
8192×2560×640 with int8 A/B in W8A8 layout, exact vs CPU, ≥ 20 TOPS each — else the 22 ms slack is not 7× but
< 2× and the lever is marginal.

**Prior attempt.** OddShapes (GPU retile, est. −23 ms) targets the same bucket; routed-expert NPU offload at
expert shapes was measured intercept-bound (`report.md:587-590`) — this lever deliberately avoids expert shapes.

### C6. Dense-27B deterministic non-exact column split — the bar C1 must beat for the same NPU minutes

Already modelled: NPU owns ≈23 % of output features with its own fold at 26.9 TOPS → **+9.5…10.9 % step (E,
`dense27b.md:160-165`)**; with IEF 19–22 TOPS → +4.9…7.9 % (`dense27b.md:208`). It was killed only under the
bit-exact contract (`dense27b.md:173`). It is a *speed* lever; C1 is a *quality* lever; both saturate the 55 GB/s
port, so they cannot run together at full rate. If a non-exact opt-in NPU route is admitted at all, this is the
cheaper one to build (no producer change, no new numerics class beyond the fold) and it should be the control arm
any EC claim is measured against.

### C7. MTP draft-head ranking on the NPU — drop

MQ2 copy of the LM head ranks ids < 100 000 (`W:crates/hipfire-arch-qwen4/src/mtp_gpu.rs:1102-1117`, `CHANGELOG.md:408-410`
**M**): ≈ 2560 × 100k × 0.25 B ≈ **70 MB per draft step (E)** → GPU ≈0.3 ms at 240 GB/s, NPU ≈1.3 ms at 55 GB/s, on the
MTP chain (hidden → MTP layer → draft head → next step). Moves 1 % of a 30 ms step off the GPU and adds +1 ms latency
per draft step. Net loss. Kill without an experiment.

### C8. DFlash / DSpark draft forward on the NPU — drop

Drafts are 0.9–1.66 GB (`W:docs/MODELS.md:106-115` **M**: qwen3.8 mq4 1.21 GB; dspark 0.91 GB in `~/.hipfire/models`).
One block draft reads the whole draft: GPU ≈5 ms, NPU ≈22 ms (E). The draft is conditioned on target features of the
*accepted* prefix, so it cannot be pipelined under the verify without speculating on acceptance (τ ≈ 1.7–9.8 by genre,
`W:docs/BENCHMARKS.md:119-122` historical **M**). Numerics are free (verification is target-lossless), which is the
only thing in its favour. Kill: the 4.4× bandwidth gap is measured; no experiment needed.

### C9. Router / expert-prefetch / acceptance predictors — drop

Halo is unified memory: there is no expert page-in to predict. The only paged store is the SSD-resident PLE row
table (`R:fn-cache/plan.md:100` **M**), already prefetched from MTP draft tokens (`CHANGELOG.md:412-413` **M**). The
acceptance/depth controller is weightless logic (`CHANGELOG.md:404-407`). A hidden-state → next-layer top-k MLP would
fit on-chip (≈3 MB int8, E) but has no consumer. Nothing to size.

### C10. Independent-request sidecars: vision tower, embedder/reranker — clean, unsized, expensive

The vision tower (27 layers, hidden 1152, intermediate 4304, 16 heads; `release-0.4.0/qwen4/parent/config.json`
`vision_config` **M**) ≈ 411 M params (E); per 1024-patch image ≈ 1 TOP → NPU int8 ≈ 40 ms + attention/LN/GELU on
the vector unit (unmeasured) vs GPU ≈ 20–25 ms (E at 40–57 F16 TOPS). It frees ≈ 1 decode token of GPU time per
image and couples to nothing on the token chain; same shape of argument for a 100–300 M-param embedder (10–20 ms
per query, E). Cost: a second inference stack on peacemaker-npu (bf16 attention, norms, GELU, int8 GEMMs with deterministic
folds) — the largest build in this list. Quality: int8 ViT/embedder is standard deployment practice, but OCR fidelity
needs its own gate. **Unsized**: measure the tower's GPU ms/image on gfx1151 first; if < 25 ms it is not worth a stack.

### C11. QSA indexer / long-context attention pieces / KV compression — drop

QSA selection is 73 ms/step (1.7 %), on the attention critical path, and bounded by `indexer_budget` 2048 so decode
KV reads stay small (≈25 MB/token at 12 QSA layers, E). Indexer scoring at 262k context is ≈400 MB/token of pooled
keys: GPU 1.7 ms, NPU 7 ms (E). Softmax/exp on bf16-in vector FP is not the GPU's numerics; QSA K/V fp8 conversion
(`R:fn-cache/plan.md:71`) needs exact f32 division the NPU lacks. GDN scan/solve is fp32 recurrent state. All
non-exact, all slower, all ≤ 2 %: kill.

### C12. Quantizer calibration / Hessians / A4 scale search on the NPU — drop

GPTQ Hessians for 240 trunk tensors are ≈ 206 TOP once (E, 65 536 rows): seconds on either device; the capture
dominated (266 s, `R:fn-trunk-iu4/report.md:12-13` **M**). Online scale search would make the NPU's bf16 MSE argmin
pick different `d` than the GPU's f32 → a different quantizer, on the producer's critical path, reading 168 MB of
f32 per producer at 55 GB/s. Zero product value; developer-time only.

### C13. gfx1152 (Krackan) — the one place plain GEMM offload clears the bar; EC inherits it

No Krackan measurement exists in this repo. **[INFERENCE from public specs]**: Radeon 860M = 8 RDNA 3.5 CUs
(Halo 8060S = 40) → ≈ 1/5 of Halo's 75 IU4 TOPS ≈ 15 TOPS; LPDDR5X-7500 128-bit ≈ 120 GB/s; same XDNA2 class NPU.
If the NPU port is again ≈55 GB/s it is ≈45 % of system bandwidth (vs 23 % on Halo), so the contention curve will not
transfer. Then: NPU int8 ≈ 25–30 TOPS **exceeds** the iGPU → the column-split (C6) and even a majority-NPU prefill
become the levers, the fold contract question (IEF, `dense27b.md:189-217`) becomes decisive, and EC (C1) is cheap
because the GPU GEMMs are 5× longer. Required measurements before any claim: `npu-bw --sweep` and the M4 derate
protocol on a Krackan box; IU4 GEMM rate on gfx1152.

### C14. Decode-time offload of any kind — structural drop

Per-token budget 30 ms; per-layer slot ≈ 0.6 ms; NPU launch ≥ 50 µs lean, ≥ 115 µs per grouped expert; port 55 GB/s.
Decode MoE share (≈1.2 GB expert bytes/token, E) on the NPU: ≤ 0.4 ms saved before 48 × ≥ 50 µs of launches. LM head
(MQ6 ≈ 0.5 GB): GPU 2 ms, NPU ≥ 11 ms. Verify-pass heads, DDTree, n-gram: CPU/GPU-trivial. Nothing survives.

## 2. Ranking

| rank | lever | honest ceiling (E) | status |
|---|---|---|---|
| 1 | **C5 shared expert W8A8 on NPU (Flash-Next)** | 180–200 ms/step (4–5 %), ≈0 contention, BF16-class quality | measured rates transfer; needs A8 quality gate + fold epilogue |
| 2 | **C1 block-selective EC on NPU, dense 27B** | quality purchase: −`f`·0.028 KLD for ≈ +7 % step, ≈480–640 ms cheaper than GPU-alone selective A8 | lives or dies on `f(S)` from §3 |
| 3 | **C6 dense-27B deterministic column split** | +9.5–10.9 % step (modelled) | control arm for any NPU route; needs admission of a non-exact opt-in |
| 4 | C1/C3 EC on Flash-Next trunk | +70–170 ms/step (1.6–4 %) if `f ≥ 0.5` and family Δs add; C3 needs η ≥ 0.8 | marginal; GPU-only C2/C4 forms are better quality-per-ms |
| 5 | C13 gfx1152 | unsized; ratio flips in the NPU's favour | needs a Krackan window, nothing else first |

Drop now, with the numbers above: C7, C8, C9, C11, C12, C14 (bandwidth- or launch-floor-killed), C10 (right shape,
wrong cost until a tower timing says otherwise). **Not NPU levers but the best EC findings:** C2 output-side low-rank
selective-A8 and C4 expert partial-A8 — hand both to GPU-side work.

Plain answer to the question this note set out to answer: the NPU is a 25–30 TOPS int8 engine behind a 55 GB/s straw. Error correction is
a real use only where the GPU work it hides behind is long (dense 27B GEMMs, the Flash-Next MoE phase) and only as a
quality purchase; tiny sidecars at decode are all slower on the NPU than on the GPU and sit on the token chain. The
single best *speed* use found is a GEMM after all — the shared expert — because it is the one dense GEMM that is
both the NPU's measured sweet-spot shape and off the GPU's critical path.

## 3. The single best first experiment — CPU, offline, real weights and real activations

**Goal.** Produce `f(S, b)` (block-concentration capture curve), the low-rank curves, and the A8 ceiling for every
A4 family, from data already on this machine; then the KLD per block/rank budget in the twin.

**Inputs (local).**
- Real captured activations, 65 536 rows per layer, f32, unrotated, exactly what each projection group reads:
  `R:fn-trunk-iu4/capture/L{i}.{gin,gout,ain,aout}.bin` (gin → gdn.qkv/z/a/b input K=2560; gout → gdn.out K=6144;
  ain → qsa.q/k/v/idx; aout → qsa.o K=6144), 103 GB (`capture.sh:4-5`).
- Rotated Hessians `R:fn-trunk-iu4/hess/L{i}.{tag}.npy` (96 files, `gptq_trunk.py:5-8`) for a cheap first pass
  (per-block variance = `diag(H_rot)` summed per 128-block).
- Trunk weights: GPTQ MQ4 payloads `R:fn-trunk-iu4/payloads/` (QT44, rotated basis; decoder in
  `R:fn-requant/attr/oracle.py` and `gptq_trunk.py:85-91`). Dense 27B weights: `~/.hipfire/models/qwen3.8-27b.mq4`
  (MQ4V2). Dense-27B *activations* are **not** captured locally — stage 1 runs Flash-Next; the dense run needs the
  same capture hook on the qwen35 prefill (named gap).

**Stage 1 (numpy, hours, no GPU).** For each (layer, tag) and each tensor reading it:
1. `x' = x · Rbd` (FWHT-256 as in `gptq_trunk.py:48-50`; AWQ divide if the family uses it), per-128 `amax`,
   the 8-candidate grid, `q`, `d`, `r' = x' − d·q` — the recipe of `block_i4_128_quant.hip:77-137`; parity with the
   GPU bytes is not required for an energy curve.
2. Error energy per block: `E_b = Σ_t ‖W'[:,b]·r'_t[b]‖²` (exact), total `E = Σ_t ‖W' r'_t‖²`. Curves: static top-`b`
   blocks by `E_b` (what C1 ships), dynamic per-token top-`b` by `d` (upper bound), for `b/K ∈ {5, 10, 20, 40 %}`.
3. Low-rank: `f_lr(r)` = energy of `W' r'` in the top-`r` left-singular subspace of `W'`, `r ∈ {32…512}`;
   output-side `f_S,lr(r)` on `W'_S`.
4. A8 ceiling: residual after the second int4 plane (`d_lo = max|r'_S|/7`) — the floor C1 can reach.
5. Report per family: `f(20 %)`, `f_lr(128)`, per-layer ranking (feeds "selected layers"), and the same on the
   `gin` tensor as a proxy for the MoE input.

**Kill thresholds (decisions, not facts).** C1-NPU needs `f_static(20 %) ≥ 0.5` on gdn.qkv/z and qsa.q/k/v; C4-GPU
needs `≥ 0.5` on the MoE-input proxy; C2 wins over C1 if `f_S,lr(128) ≥ 0.7`.

**Stage 2 (twin, ≈30 min and 26 GB RSS per arm, `attribution.md:243` M).** Add to `R:fn-requant/attr/oracle.py` a
fake-A4 arm for the trunk families and MoE experts (it has no A4 model today, `oracle.py:17-18`), then arms: + static
block-selective correction at `b` ∈ {10, 20 %}, + low-rank at `r` ∈ {64, 128, 256}, + deferred-add for gdn.out/qsa.o
(C3's η). KLD vs the HFKLDR teacher on the 8 chunks gives **KLD recovery per block budget and per rank** directly,
and the family-additivity check that the budget math in C1 assumes. Only after stage 2 does any NPU kernel work start.
