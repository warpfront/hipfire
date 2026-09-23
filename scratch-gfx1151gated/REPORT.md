# gfx1151 prefill exclusion audit

## Verdict

The gfx1100-shaped instance of this dispatch defect existed and paid 11.92%; the gfx1151-shaped instance does not exist on the measured Qwen3.8-27B MQ4V2 prefill path.

Halo is admitted to the material gfx11 paths: widened prefill, IU4 projections, grid specialization, column-adjacent full tiles, LF16 SET, producer-side IU4 fusion, GDN chunk scan, and gfx11 FA2. The one material, runnable existing entry from which it is excluded is LF16 ADD. Re-testing that exclusion confirmed the source comment: LF16 ADD is 0.9593x at N=5909 and 0.9474x at N=8192. The exclusion is deliberate and correct.

No production source or typed configuration changed. The candidate remains reachable only from the explicit scratch harness, so it is default-off. `gemm.rs` dispatch is untouched. There is no gfx1100 regression surface.

## Scope and method

- Source baseline: `7a34a572a` (production grid specialization enabled).
- Target graph: Qwen3.8-27B dense MQ4V2 ordinary prefill/TTFT, including projection GEMMs, Q8 attention, producer fusions, GDN, architecture defaults, and dispatch-family ingress.
- Trace used for cost attribution: `scratch-iu4gridspec/rocprof-grid-gfx1151/daemon_kernel_stats.csv` from the production gfx1151 grid-specialized run. Its summed GPU-kernel duration is 8.369674781 s.
- A table row is one logical predicate. Mirrored caller/implementation guards are grouped, but every site is named. Rows marked zero are absent from this production trace, not assumed free in other workloads.
- gfx1201-only entries are enumerated because they are adjacent prefill branches, but they were not promoted to runnable gfx1151 candidates: their source uses a different WMMA fragment/K width, activation or KV format, or both.
- Enumeration result: 43 logical predicate groups on the target call graph, plus 6 adjacent architecture predicates screened out as decode-, MoE-, or other-model-only. Repeated implementation guards for one route are grouped but all source sites are cited.
- Decode-only, speculative-verify-only, MoE-only, and other-model defaults found during the same audit are recorded separately so that they are not confused with missing production TTFT coverage.

## Complete predicate enumeration: target prefill graph

| ID | Predicate site | What it gates | Admitted | Excluded | Judgement and target consequence |
|---|---|---|---|---|---|
| A1 | `crates/rdna-compute/src/arch_caps.rs:130-139` | RDNA architecture molecules | gfx1151 is `is_rdna3` and `is_rdna3p5` | gfx1151 is not `is_rdna3_dgpu` or `is_rdna4` | Deliberate topology split. This classification is not itself a kernel denial. |
| A2 | `crates/rdna-compute/src/arch_caps.rs:143-147` | WMMA/MMQ capabilities | gfx1151 has WMMA, gfx11 wave32 WMMA, and MMQ | gfx1151 lacks gfx12 wave32 WMMA | Deliberate ISA split. Material gfx11 family keys remain admitted. |
| A3 | `crates/rdna-compute/src/arch_caps.rs:247-260` | MMQ minimum batch | gfx1151 takes the RDNA3.5 arm, cutoff 128 | batches below 128 | Deliberate measured cutoff; same numeric cutoff as gfx1100/gfx12. N=5909/8192 admitted. |
| F1 | `crates/rdna-compute/src/feature_flags.rs:526-527` | IU4 prefill and grid specialization defaults | grid specialization defaults on for gfx1151; IU4 is the production typed route | disabled override | Halo admitted; no exclusion. |
| F2 | `crates/rdna-compute/src/feature_flags.rs:551-561` | gfx1100 QKVZA K2048, residual stage-X32, and sigmoid-buffer defaults | exact gfx1100 by default | gfx1151 by default | Deliberate measured defaults for old HFQ4/F16/GEMV routes. None appears in target trace. |
| F3 | `crates/rdna-compute/src/feature_flags.rs:612-613` | RDNA3 RMSNorm vecsum default | exact gfx1100 | gfx1151 | Deliberate per-card tuning; target runs gfx11 IU4 producer fusions instead. |
| F4 | `crates/rdna-compute/src/feature_flags.rs:644-646` | residual and gate/up small-N LDS-stage defaults | exact gfx1100 | gfx1151 | Deliberate N<=16 verify policy. Target N=5909/8192 cannot reach it. |
| F5 | `crates/rdna-compute/src/feature_flags.rs:647-656` | gfx12 MQ4V2 FP8 projection defaults | exact gfx1201 | every gfx11, including gfx1151 | Deliberate data-format/ISA split. Halo uses IU4, not a withheld compatible entry. |
| F6 | `crates/rdna-compute/src/feature_flags.rs:657-658`; `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:6038-6049` | one-launch gfx12 GDN preamble | exact gfx1201 plus shape constraints | gfx1151 | Deliberate gfx12 source. Halo retains the gfx11 GDN sequence. Not a compatible entry to ungate. |
| F7 | `crates/rdna-compute/src/feature_flags.rs:659-660` | GDN chunk scan | gfx1100, gfx1151, gfx1201 | other arches | Halo admitted; trace proves 576 calls, 2.083%. |
| F8 | `crates/rdna-compute/src/feature_flags.rs:661-668` | gfx12 SiLU+quant, producer+quant, and FP8-stream defaults | exact gfx1201 | gfx1151 | Deliberate gfx12 format/source split. Halo has its own IU4 producer routes in F9. |
| F9 | `crates/rdna-compute/src/feature_flags.rs:665-666`; `crates/rdna-compute/src/dispatch.rs:3071-3084` | gfx11 producer+IU4 sidecar fusions | gfx1100 and gfx1151 | non-gfx11 target arches | Halo admitted. Trace proves the fused gfx11 producer family is active. |
| F10 | `crates/rdna-compute/src/feature_flags.rs:669-676` | gfx12 FA2/packet/Q-resident defaults and gfx11 FA2 default | gfx1201 for gfx12 variants; gfx1100/gfx1151 for gfx11 FA2 | Halo only from gfx12 variants | Deliberate source/format split. Halo's gfx11 FA2 is default-on and traced. |
| G1 | `crates/rdna-compute/src/gemm.rs:329-335` | MQV2 gfx11 batch-tile width admission | gfx1151 for 2/3/5/6-bit; gfx1100 for 5/6-bit | dtype-specific siblings | Halo is more broadly admitted; MQ4 uses the explicit policy in G3. |
| G2 | `crates/rdna-compute/src/gemm.rs:338-359` | same-row multi-wave policy | exact gfx1100 for MQ5/6 at measured ranges | gfx1151 | Deliberate: comment records neutral/negative gfx1151 measurements. Not target MQ4. |
| G3 | `crates/rdna-compute/src/gemm.rs:376-408` | architecture-specific MQV2 batch tiles | gfx1151 MQ4 gate/up BT12 and QKVZA/QKV/residual BT4 from N>=96 | unmatched arch/dtype/shape tuples | Halo has an explicit measured policy, not a fallback caused by exclusion. |
| G4 | `crates/rdna-compute/src/gemm.rs:19690-19698` | IU4 consumer capability gate | exact gfx1100, gfx1151, gfx1201 | all other arches | Halo admitted. |
| G5 | `crates/rdna-compute/src/gemm.rs:19706-19777` | K32 staged IU4 implementation | exact gfx1201 | gfx11 | Deliberate ISA/layout split: gfx12 K32 WMMA, fragment mapping, and source differ. Not executable as a gfx1151 ungate. |
| G6 | `crates/rdna-compute/src/gemm.rs:19783-19794`; `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:651-653` | grid-specialized partial-N interior and column-adjacent full entries | gfx1100 and gfx1151 | other arches, capture/replay | Halo admitted. The `_gfx1151` ordinary symbol suffix is incidental: its compile guard also admits gfx1100. This is the current production reference. |
| G7 | `crates/rdna-compute/src/gemm.rs:19795-19812`; `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:628-648` | 16-wave LF16 column entry | exact gfx1151, SET only | ADD on gfx1151; every other arch | Deliberate according to source comment, but this was the highest-value compatible exclusion and was re-tested below. |
| G8 | `crates/rdna-compute/src/gemm.rs:29317-29346`, `29923-29941`, `31120-31138`, `32765-32779` | QKVZA, QKV, gate/up, and residual IU4 ingress | gfx1100 and gfx1151, eager N>=64 | other arches/modes/small N | Halo admitted to every dense projection. These early returns make later F16-only gates inactive at target N. |
| G9 | `crates/rdna-compute/src/gemm.rs:29370-29391`, `29945-29966`, `31162-31180`, `32844-32860` | arch-specific F16 batch-tile launchers | separate gfx1100 and gfx1151 launchers | neither target is excluded | Incidental symbol split backed by explicit sibling implementations. |
| G10 | `crates/rdna-compute/src/gemm.rs:31140-31150` | gate/up MQ4 same-row MW-LDS | exact gfx1100, N>=384 | gfx1151 | Deliberate CU-count policy. At target N this code is unreachable because G8 returns through IU4 first. |
| G11 | `crates/rdna-compute/src/gemm.rs:31184-31200` | gate/up raw-slab LDS stage | exact gfx1100, N<=16 | gfx1151 and target N | Deliberate verify optimization; zero target calls. |
| G12 | `crates/rdna-compute/src/gemm.rs:32781-32817` | residual LDS-stage/split-K verify tier | exact gfx1100, N<=16 | gfx1151 and target N | Deliberate 96-CU small-grid policy; zero target calls. |
| G13 | `crates/rdna-compute/src/gemm.rs:32819-32834` | residual production MW4/MW8 | exact gfx1100 at N>=416 | gfx1151 | Deliberate CU-count policy. Inactive after the IU4 early return. The traced 1.44% `...gfx11_mw4_lds` is the shared small-M tail helper, not this withheld route. |
| P1 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1021-1032`, `1097-1104` | ordinary chunk default | exact gfx1100 512; exact gfx1201 384/512; gfx1151 all-MQ4 special 512 | nonmatching configurations | Halo gets 512 explicitly; no exclusion. |
| P2 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1042-1047`, `1173-1177` | widened prefill | gfx1100, gfx1151, gfx1201 | other arches | Halo admitted up to the same 8192-row ceiling. |
| P3 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1228-1238` | projection format within widened prefill | gfx1201 FP8; gfx1100/gfx1151 IU4 | arch whose matching format flags are off | Deliberate format split; Halo admitted via IU4. |
| P4 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:5452-5460`; `crates/rdna-compute/src/mq_f16_producers.rs:25-30,60-64,252-256,367-371,471-475` | fused F16 projection-input producers/consumers | exact gfx1100, chain verify, N<=16 | gfx1151 and ordinary prefill | Deliberate small-verify specialization; zero target calls. |
| P5 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:6230-6242`; `crates/rdna-compute/src/mq_f16_residual_producers.rs:103,201,290,362,438,510,589` | fused F16 residual-input producer family | exact gfx1100, chain verify, N<=16 | gfx1151 and ordinary prefill | Deliberate small-verify specialization; zero target calls. |
| P6 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:7924-7935` | fused full-attention preparation | exact gfx1100 in chain verify | gfx1151 and ordinary prefill | Deliberate small-verify launch fusion; zero target calls. |
| P7 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:8248-8268`; `crates/rdna-compute/src/attention.rs:6635-6656` | Q8 multi-row masked attention | gfx1100 or gfx1201, N=4..32 | gfx1151 | Deliberate measured verify allowlist; zero target calls at N=5909/8192. |
| P8 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:8309-8328` | native-FP8 packet/Q-resident widened attention | exact gfx1201 with FP8 KV | gfx1151 Q8 KV | Deliberate format/source split. Halo continues through the per-segment gfx11 FA2 route. |
| P9 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:10787-10816`; `crates/rdna-compute/src/attention.rs:3579-3591` | merge two 512-row FA halves and admit one 1024-row FA2 launch | exact gfx1151, eager/non-replay | every other arch from N=1024 | Deliberate positive Halo specialization; it is admitted rather than withheld. |
| P10 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:11837-11853` | dense GDN chunk-scan route and KV-format selection | gfx1100/gfx1151 IU4+Q8; gfx1201 IU4-or-FP8 with native FP8 KV | unsupported arch/format pairs | Halo explicitly admitted; trace proves the route active. |
| P11 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:2717-2740` | model-level MQV2 batched-prefill eligibility | WMMA-capable gfx11/gfx12 through the shared rule, including gfx1151 | non-WMMA arches or gfx11 kill switch | Halo admitted; this is the outer gate required to reach all dense projection predicates above. |
| P12 | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:3692-3701` | Q8 WMMA prefill eligibility | every arch reporting WMMA, including gfx1151 | non-WMMA arches or explicit opt-out | Halo admitted; no architecture-string exclusion remains here. |
| T1 | `crates/rdna-compute/src/attention.rs:3628-3675`; `crates/hipfire-dispatch/src/families/attention.rs:1820-1870,2310-2360` | FA2/packet/Q-resident source selection | exact gfx1201 gets gfx12 sources; gfx11 allowlist includes gfx1151 | Halo only from gfx12 sources | Deliberate ISA/KV-format split; trace proves gfx1151 runs 192 gfx11 FA2 calls (6.409%). |
| T2 | `crates/hipfire-dispatch/src/families/attention.rs:698-725` | one-launch Q8 K+V cache writer | exact gfx1100 | gfx1151 | Deliberate measured allowlist, but potentially compatible. Entire incumbent pair is only 0.0667% of traced GPU time; even deleting it is 1.00067x. |
| T3 | `crates/hipfire-dispatch/src/families/attention.rs:2026-2047` | flash-prefill default | every `gfx11*`, plus eligible gfx12 | other arches outside gfx12 envelope | Halo admitted; no exclusion. |
| T4 | `crates/hipfire-dispatch/src/families/attention.rs:2112-2125` | batched versus WMMA speculative flash variant | exact gfx1100 defaults to batched in speculative verify; siblings default WMMA | gfx1151 from batched default | Deliberate unmeasured-sibling policy; not ordinary TTFT. |
| T5 | `crates/hipfire-dispatch/src/families/attention.rs:887-888,2153-2156,2201-2205,2385-2389` | gfx12 query-16 default envelope, WMMA capability, and fallback crossover | exact gfx1201 gets query-16 default and 4096 crossover; gfx1151 has gfx11 WMMA and 8192 fallback crossover | gfx1151 from gfx12-only default | Deliberate measured tuning. The target returns through gfx11 FA2 first, so the fallback crossover has zero target calls. |

### Screened adjacent predicates that are not on this target graph

These are real architecture exclusions, but they cannot explain Halo TTFT for this dense artifact.

| Site | Predicate | Why it is outside the target graph | Judgement |
|---|---|---|---|
| `crates/rdna-compute/src/feature_flags.rs:468-493` | gfx1100 GEMV row/load-policy defaults | GEMV/decode compiler policy, not the IU4 prefill consumer | Deliberate measured default. |
| `crates/rdna-compute/src/attention.rs:58-86` | exact gfx1100 short-KV tile32; one exact gfx1151 Radiowave shape | scalar Q8 decode/tiled fallback; production target uses gfx11 FA2 | Deliberate shape-specific tuning. |
| `crates/rdna-compute/src/attention.rs:148-155` | exact gfx1100 asymmetric Q8 K+V pairing | graph-on decode | Deliberate measured decode optimization. |
| `crates/rdna-compute/src/feature_flags.rs:688-702` | Gemma4 embedding/PLE defaults admit gfx1100/gfx1201 | different model family | Deliberate validation allowlist; no Qwen cost. |
| `crates/hipfire-dispatch/src/pipeline/mod.rs:649-656` | exact gfx1100 MoE down-last combine | batch-one MoE decode | Deliberate experiment; unrelated model/path. |
| `crates/hipfire-dispatch/src/pipeline/mod.rs:754-774`, `1107-1112`, `2056-2061` | gfx1100/gfx1201 fused MQ4V2 MoE gate/shared-down | batch-one Ornith MoE decode | Deliberate exact-shape allowlist; unrelated model/path. |

## Cost ranking and candidate choice

The ranking first requires an existing entry that can plausibly run on gfx1151 without changing data format or WMMA ISA. This removes the adjacent gfx1201 K32/FP8/packet sources before performance ranking.

| Rank | Compatible exclusion | Traced incumbent cost | Absolute zero-cost ceiling | Decision |
|---:|---|---:|---:|---|
| 1 | LF16 ADD (`G7`) | ordinary ADD full tiles: 2.188365133 s, 26.146% | 1.3540x | Test. Exact gfx1151 symbol already exists; only host dispatch withholds it. |
| 2 | gfx1100 one-launch K+V cache write (`T2`) | both incumbent writes together: 5.579158 ms, 0.0667% | 1.00067x if both vanished; about 1.00033x if pairing merely halves them | Reject as immaterial. |
| 3 | exact-gfx1100 F16/MW/verify routes (`G10-G13`, `P4-P7`, `T4`) | zero calls at target N/mode | 1.0000x on this trace | Reject: earlier IU4/ordinary-prefill branches return first or N is <=32 only. |

Adjacent architecture-specific families do carry time, but they are not compatible ungates:

- gfx12 K32 IU4 sits beside 77.82% of the trace (`full SET + full ADD + tail`) but uses the gfx12 K32 WMMA source and fragment mapping. It is a port, not an admission flip.
- gfx12 attention variants sit beside the 6.409% gfx11 FA2 cost, but Halo is already executing the corresponding gfx11 FA2 entry.
- gfx12 producer fusions sit beside 7.807% of traced gfx11 fused producers; Halo is already executing its corresponding fused entries.
- gfx12 GDN pre-fuse sits beside a 3.704% total GDN family, but is a different source. Even erasing the entire family would only be 1.0385x, and the compatible KKT-to-scan lane was bounded separately.

Thus LF16 ADD was not an arbitrary pick: it was the only already-built, exact-gfx1151, material-cost entry withheld solely by a dispatch predicate.

## LF16 ADD experiment

### Candidate

Reference: the production grid-specialized split: unchecked full interior plus one guarded `...tail_gridspec` launch when N is partial. SET uses the shipped LF16 entry; ADD uses the shipped ordinary 8-wave entry.

Candidate: identical split and guarded tail, but dispatch full ADD interior tiles to existing `gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151` at `[32,16,1]`. Gate/up SET is intentionally source-identical between arms and is therefore a control.

Scratch source: `scratch-gfx1151gated/bench_lf16_add.hip`. Corrected binary SHA-256: `146745f703d433258319496d2b252259d1306b6dc221296861687e2a14f68e2e`.

### Reference defect caught and discarded

The first harness revision compared against the historical checked whole-grid generic launch for partial N rather than the now-production grid-specialized split. That would have attributed the already-landed interior/tail win to this candidate. All output from that revision was discarded. The reference was changed to the exact production split before the measurement below. This is the same class of control error as comparing a new route against a superseded dispatch baseline.

### Protocol

- Halo lease announced and released; one GPU process at a time.
- Empty before and after except the resident zero-VRAM `gpusentry` process.
- Every fresh process asserted `gcnArchName` contains `gfx1151`.
- Four fresh processes per cell, order forward/reverse/reverse/forward.
- Per process and arm: 2 warmups, then 9 HIP-event samples; process statistic is the median.
- Exact production dimensions: gate/up `(M,K)=(17408,5120)`, down `(5120,17408)`, N=5909 and N=8192.
- ADD outputs start from nonzero `0.25f`, so SET cannot accidentally pass as ADD.
- Each fresh process compares every output float bit against production before timing.

### Results

The table reports the mean of the four fresh-process medians.

| Shape | N | Production TOPS | LF16-ADD-route TOPS | Ratio | Candidate / 109.690 TOPS ceiling | Bit parity |
|---|---:|---:|---:|---:|---:|---|
| gate/up SET control | 5909 | 49.3315 | 49.3713 | 1.0008x | 45.01% | 0 / 411,455,488 differing bits across four processes |
| gate/up SET control | 8192 | 48.8215 | 48.8163 | 0.9999x | 44.50% | 0 / 570,425,344 differing bits across four processes |
| down ADD candidate | 5909 | 46.9070 | 44.9980 | 0.9593x | 41.02% | 0 / 121,016,320 differing bits across four processes |
| down ADD candidate | 8192 | 47.9803 | 45.4565 | 0.9474x | 41.44% | 0 / 167,772,160 differing bits across four processes |

Fresh-process TOPS values:

- gate/up N=5909 production: `49.347, 49.294, 49.213, 49.472`; control: `49.506, 49.454, 49.251, 49.274`.
- gate/up N=8192 production: `48.956, 48.836, 48.831, 48.663`; control: `48.909, 48.878, 48.637, 48.841`.
- down N=5909 production: `46.878, 47.109, 46.874, 46.767`; candidate: `44.969, 44.977, 45.194, 44.852`.
- down N=8192 production: `48.125, 47.987, 47.999, 47.810`; candidate: `45.323, 45.492, 45.539, 45.472`.

All 16 fresh processes were bit-exact. No daemon c24 was run: there is no production build or selectable daemon route, and the hard gate stopped before TTFT. The scratch harness instead compared every output bit for the exact kernel shapes.

### Gate decision

The required gate/up N=5909 threshold was 1.15x. The source-identical control was 1.0008x, as expected. LF16 ADD itself regressed 4.07% at N=5909 and 5.26% at N=8192. Therefore:

- **KILL** LF16 ADD.
- Do not run TTFT, matrix, or server battery.
- Do not add `kernel.gfx1151_route` or alter production dispatch; an inert production knob would add dead surface.
- Existing `!add` predicate is deliberate and should remain.

## Final premise assessment

The audit found no material gfx1151 route accidentally gated away on the production gfx11 prefill graph. Every high-cost bucket is either already admitted on Halo, backed by a Halo-specific sibling, or separated for a real ISA/data-format reason. The sole high-cost compatible exclusion was explicitly measured and lost. Remaining compatible exclusions have at most 0.067% traced cost or are absent at ordinary-prefill N.

This closes the gfx1151 mirror lane as a negative result: no production change, no TTFT claim, and no gfx1100 behavior change.
