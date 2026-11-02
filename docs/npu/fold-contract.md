# Dense Qwen3.8-27B fold/co-op decision

> **Lifecycle: historical dark research record (0.4.1, decided 2026-10-05); superseded as a
> design.** The full-IEF15 route (G0/G1) is **killed**, and the mixed-contract co-op plan
> in §§3–8 was never wired into any serving path. Only the dark CPU oracle and NPU
> pair-core pieces (`tools/npu/fold-model`, `peacemaker-npu::kernels::iu4_ief15*`) exist; the
> GPU sidecar/split modules named in §8 were not written. No `I32` or IEF15 contract
> is a shipped 0.4.2 path: the current Flash-Next NPU trunk is BF16-direct only
> (calibration schema 4, input projections only; out-projection unwired). See
> [`npu-trunk.md`](npu-trunk.md) for the current route and schema. Tables and
> payoff figures below are dated model estimates or fixture-bound measurements, not
> current performance, and are not an admission. `HIPFIRE_IU4_FOLD_CONTRACT`, named
> here as a proposal that was killed, exists nowhere in code or configuration.

Scope: 0.4.1 dark opt-in experiment, cutoff 2026-10-05 23:59 MST. `W:` below denotes paths in the hipfire 0.4.1 beta source snapshot, read-only. This branch contains CPU research and planning, not runtime changes. Flash-Next is deprioritized; no further Flash error/payoff work.

## 1. Measured decision: G0/G1 KILLED

**Do not implement or land the original full-IEF15 G1 config/sidecar/route slices.** G0 failed its1.03x speed gate by a large margin, despite exact new-contract bytes and zero spills. The project then requested **mixed-contract co-op**: current GPU fold on GPU-owned columns, IEF15 only on NPU-owned columns. Full IEF15 is retained only as a transferred gfx1152 sensitivity, not a revived implementation plan.

Measured Halo G0 experiment branch `exp/ief15-g0`@`7559ee957` (not in this repository); hardware logs `edge-1`, `correct-1`, `time-1`..`time-3` were kept on the test host. Each timed process used its own timing lock,1s DPM warmup and4ABBAblocks×4reps;3 fresh processes, byte-identical A/Xq between routes. S/a/D/b production was **not charged**, so a real producer only worsens this result. Comparison is tuned PM V2B SET/ADD/SiLU, and shipped `_set_zba` for Z|beta|alpha, not an upstream naive baseline.

| Shape, calls | incumbent ms r1/r2/r3 | IEF15 ms r1/r2/r3 | ratio r1/r2/r3 | IEF graph ms r1/r2/r3 |
|---|---|---|---|---|
|qkv,48|10.872/10.790/10.799|15.666/15.666/15.697|1.441/1.452/1.454|16.09/16.09/16.20|
|zba,48|7.193/7.250/7.323|10.011/10.063/10.137|1.392/1.388/1.384|10.07/10.09/10.18|
|FAq,16|13.095/13.180/13.218|19.230/19.269/19.426|1.468/1.462/1.470|19.30/19.40/19.53|
|FAk/v,32|1.324/1.326/1.323|1.736/1.736/1.750|1.312/1.309/1.323|1.73/1.72/1.72|
|o_proj,64|6.572/6.661/6.706|9.550/9.707/9.741|1.453/1.457/1.453|9.63/9.72/9.77|
|gate_up+SiLU,64|36.528/36.625/36.772|61.665/61.810/61.715|1.688/1.688/1.678|61.76/61.76/62.11|
|down ADD,64|17.615/17.695/17.813|26.561/26.780/26.854|1.508/1.513/1.508|26.56/27.41/27.58|
|weighted step GEMMs|5004.75/5022.00/5046.29|7853.42/7889.85/7898.75|1.5692/1.5711/1.5653|—|

Probe MD5 `ef9e1ee39ed28632d2dfc27023007c31`; PM bundle MD5 `cfa66665c06e010e0064f2151896f763`; all input MD5s are in the G0 structured report.243VGPR SET/ADD/SiLU, zero private segment/spill counts/scratch instructions. Exact SET/ADD/SiLU edge corpus and all seven dense shapes passed CPU oracle comparison; eager/graph-replay output bytes identical. Integer pack model passed~2M random plus~490k tie/carry/subnormal/overflow cases. These G0 results were observed in a separate hardware run, not re-run by this CPU-only study. G0 has no runtime selection or incumbent changes.

Cause supported by the implementation census: five integer VALU operations/output/epoch, noVOPD, plus M256×N128 retile,64int64states/wave,48KiBLDS. Incumbent uses three scalar operations/output packed into12VOPDpackets/8outputs (`W:crates/peacemaker-isa/src/kernels/iu4_fold.rs:30-48`), M256×N256 and128f32sumVGPRs (`iu4_v2b.rs:57,105-124`). Zero spills do not make integer carry chains or repeated weight reads free. The original G0/G1 path is killed, not 'almost viable'.

## 2. Denominator and honest ceilings

Measured dense pp8192 step6810.69ms, IU4GEMMs5287.408ms, remainder1523.282ms,399.6TOP (`docs/npu/dense27b.md:20-33,167`). Thus77.6% is the **entire-bucket time ceiling**, not a co-op prediction. Old fold-free26.9TOPS NPU model was only+10.9% throughput; old exactf32 emulation model+1–3% (`:150-175`). Full IEF15 now has a measured1.57x GPU penalty. **Halo+10–15% is not supported by either new model below.** gfx1152 remains the primary modeled payoff because its GPU is much smaller; there are no gfx1152 measurements here.

## 3. Frozen NPU IEF15 arithmetic and sidecars

For unchanged nibbles/activation codes:

`C[t,r,e]=sum(j=0..127) (u[r,128e+j]-8)*q[t,128e+j]`, exacti32,range[-7168,8192]. Current contract is ascending epoch `sum=fmaf(RN_f32(sc_e*d_e),float(C_e),sum)` from+0 (`W:iu4_fold.rs:3-9,36-48`). Original AWQ/FWHT and A4 clipping are untouched.

For each finite nonnegative scale vector choose the **smallest integer** a for which all `RNE_integer(v_e*2^-a)<=32767`; allzero usesa=0. For one weight row encode S15 across all its epochs with exponenta_r; for one token encode D15 across all its epochs with exponentb_t. Device exponent selection uses exponent bits and exact integer rounding, not approximate log2. Preserve q; this is post-quantization scale projection, not re-clipping.

`I[t,r]=sum_e i64(S[r,e])*i64(D[t,e])*i64(C[t,r,e])`; `Y=RNE_binary32(I*2^(a_r+b_t))`, **one** final rounding. E<=136 implies |I|<2^51. No atomics. K partitions retain/merge signed64 states before rounding; never sum rounded f32 partials. Integerzero maps+0; negative underflow maps-0. Handle subnormals/ties/overflow directly; i64->f64->f32 can double round and is forbidden. Nonfinite/negative scales reject before slot publication. CPU oracle `tools/npu/fold-model/src/lib.rs::{grid,fixed,round_scaled}`.

Version1 sidecars, LE, no MQ4 format change:
- S:u16 `[local_feature][K/128]`,a:i16 `[local_feature]`, everymantissa<=32767.
- D:u16 epoch-major `[K/128][token]`,b:i16 `[token]`; computed after every existing A4 producer write, not cached across changed activations.
- Immutable original QT44 bytes and72B A4 `[epoch][token]` blocks remain address-stable inputs. No flag-off allocations/launches.
- NPU dense output token-major f32 `[token][local_feature]`,pre-SiLU/pre-residual. GPU owns exact existingSiLU (`W:iu4_v2b.rs:753-758`) and `RN(old+Y)` ADD/deferred residual semantics. Padding outputs+0. Separate g/u outputs are paired by hidden-feature index, not independent column cuts.
- Existing ringv1 fixed-slot offsets/program identity from `docs/npu/railgun-npu.md` §3 remain; IEF15 needs a distinct program/output-contract cache identity, never an int8/int32 program reinterpretation.

## 4. Mixed contract: split is part of numeric identity

**GPU-owned output columns retain current ascending f32FMA, with no IEF15 GPU kernel or scale sidecar. NPU-owned columns use IEF15.** This intentionally changes the original shared-all-device contract after the measured kill decision.

For identical incoming A4/scales, GPU-column fold error vs current is exactly0; NPU-column error is the IEF15 measurement in§5. Error does not disappear merely because the other columns are exact. Representative pooled NRMSE might scale roughlysqrt(f) under uniform signal/error distribution, but that is **INFERENCE**, not a bound: offloaded rows may contain the largest errors. Raw maximum relative error remains cancellation-sensitive. The model does not prove end-to-end layer/logit error; after one changed projection, subsequent hidden states, d and q can change. SiLU nonlinearity also prevents reusing raw fold error as a post-SiLU guarantee.

Determinism requirements:
1. Freeze exact per-projection sorted feature ranges before execution/capture. A '17%' setting is insufficient: record integer ranges, paired g/u cuts, K segmentation, output layout, grid policy and finalpack version.
2. Bind that map plus model/weight identity and contractversion into graph/Redline/prefill-cache/run identity. A different split means a different numeric execution even with identical prompts. No load-adaptive/thermal/occupancy-dependent split changes within a reproducibility claim.
3. Disjoint columns publish once. GPU completion/NPU completion order cannot affect sums; every integer Kpartition joins beforeoneRNE, epilogue/ADD executes once, DONE only after all C channels join.
4. NPU timeout/unavailable/wedge **fails closed**. Silent fallback to currentGPU on NPU-owned columns changes bytes. A same-contract fallback would need to replay IEF15 on exactly those columns (G0 is too slow to be a promoted GPU route); not in this plan. `HIPFIRE_NPU_SPILLOVER=0` returns the original whole-GPU execution, a different explicitly selected numeric identity.
5. Keep only default-OFF co-op flag `kernel.npu_spillover` / `HIPFIRE_NPU_SPILLOVER` (`crates/hipfire-config/src/lib.rs`, `kernel.npu_spillover`: experimental, default off, gfx1151-only, needs a verified `.xdna.zip` registry sidecar). Do not add the killed proposed `HIPFIRE_IU4_FOLD_CONTRACT` GPU setting or route (it was never implemented; no such setting exists). The co-op program/map identifies NPU IEF15; caller owns its fixed-map schema.

GPU-current split speed is still an **assumption**, not proven by G0: incumbent kernels have no independent outputLDY/column offset. A PM-authored strided **current-fold** entry or compact+copy must preserve the exact fold and price all movement. Do not charge zero for a full output gather. A current-fold strided entry changes address arithmetic only, not WMMA/fold order; its own speed/parity gate precedes integration.

## 5. Dense CPU errors, per GEMM and all64layers

Artifact `qwen3.8-27b.mq4-xts` (test-host copy),14,987,185,152B. Stored scales/nibbles, never random weights.496tensors/all64layers,32evenly spaced feature rows/tensor,8synthetic rotated-domain tokens;126,976fold outputs,57,655,296weight bytes. Real dense A4 dumps were not located in inspected local workspaces; no new capture/hardware experiment here.

Deterministic seed0x6a09e667f3bcc909; CLTnormal activations with epoch amplitudeexp(0.4*N). Correct producer c2 **candidate indices**{5,7}, not divisors5/7; four-consecutive-elements/lane f32FMA MSE plus XORreduction mirror `W:kernels/src/block_i4_128_quant.hip:94-126,213-217`. All compared folds share identical modeled q/d. Earlier provisional divisor5/7 numbers are superseded. Device fast-reciprocal producer equality is not inferred from CPU math.

| Candidate | mean/max raw rel vs current | mean/max raw rel vs f64 | NRMSE current/f64 | maxabs vs current |
|---|---|---|---|---|
|IEF15|1.859095e-4 /1.280640|1.859144e-4 /1.279685|3.147933e-5 /3.147921e-5|4.940629e-4|
|IEF16|1.050309e-4 /2.122087|1.050427e-4 /2.118903|1.491533e-5 /1.491483e-5|1.972914e-4|
|BF16MAC|1.191080e-2 /76.63424|1.191043e-2 /76.55506|1.808243e-3 /1.808243e-3|2.274203e-2|
|group4|2.482183 /12491.63|2.482241 /12486.39|.3781659 /.3781659|3.800138|
|once|2.696805 /14935.05|2.696797 /14935.90|.4322918 /.4322918|4.491929|

0/847,872 **sampled** weight scales lost precision on15 or16bit grids, not an all-weight proof. CurrentNRMSEvsf641.438134e-7; zero reference mismatches0. IEF16 improves aggregate precision but not every nearzero output's relative error; maximum is not monotonic in mantissa bits.

F64reference is ascendingΣf64(sc)*f64(d)*f64(C), without per-epoch f32rounding, not an exact real-arithmetic superaccumulator. Mean relative error averagesabs(delta)/abs(reference) over nonzero samples; maximum uses the same cancellation-sensitive denominator. NRMSE=sqrt(Σdelta²/Σf64reference²). All metrics are **pre-epilogue**. Layer tables pool sampled projection outputs, not propagated layer-state/logit errors. No KLD used.

Full requested tables [`numeric-tables.md`](../../tools/npu/fold-model/numeric-tables.md):5-30 summary/perGEMM,34-99 every64layer. Per-candidate/per-tensor/kind/layer details [`dense-errors.json`](../../tools/npu/fold-model/dense-errors.json). Layer error/probabilistic quality and decoded-output review remain model-level gates; table values alone cannot promote changed numerics.

## 6. Candidate AIE/GPU costs

Slot lower bounds are **estimates/instruction census**, not NPU IEF measurements.16-lane issue model `crates/peacemaker-npu/src/kernels/iu4_fold_model.rs:7-46`; measured GEMM.331cycles/output/epoch (`docs/npu/dense27b.md:90-100`), nativeVMACfloor.25. Vecslot competes with GEMM; apparent.081slack is not automatically free.

| Contract | AIE2P additional work per16outputs/epoch | GPU gfx1151 / gfx1201 cost | Rank |
|---|---|---|---|
|current ascendingfmaf(RN(sc*d),C,sum)|bestexact emulation22Mv/4St/2Vec-int/23Vec-float,1.375–1.81cycles/output;fpaccRNEunverified|gfx1151:12VOPDpackets/8outputs;gfx1201stage2:80packets/64outputs plusstage1unbias (`W:iu4_gemm/fold.rs:35-47`)|keepdefault; exactNPU killed|
|IEF15 exacti64ΣS15D15C,oneRNE|VMUL32×16,VSRS0 tosafeP32,VMAC32×16 withsignedC16:2Vec+1St plusMv/widen/shuffle. Vec-onlytotalfloor.375cy/output vs.331GEMM;3Vecfloor.4375. Retile/localI loads+stores/finalpack extra.19–23TOPS **estimated** from26.9foldfree|fiveVALU/output/epoch,2sumVGPR/output. gfx1151 **measured1.565–1.571x** afterretile,0spills. gfx1201 can double64→128sumVGPR but higherWMMA rate makes fold relativelycostlier;1.57notgfx1201measurement|mixed-NPU only; fullGPU killed|
|IEF16 grids<=65535|P mayexceed signed32;unsigned16/signed32VMAC needs signbit corrections/decomposition;>=3–4Vec plusMv/constantshifts|unsignedS*D fitsu32 but signedC needs unsignedhigh/sign correction orwidermultiply,greaterthan5-opIEF15;same64bitstates|only ifIEF15 precision insufficient|
|BF16MAC: T=BF16(RN(sc*d)),C'=BF16(C),ascendingf32fma|nativebf16VMAC plusT formation/conversions;fpaccRNE/denormal semantics unverified|mul+roundT,intconvert+roundC,fmac;onef32sumVGPR but loses incumbent fold pairing|lowerprecision/unprovenhardware; notfirst|
|group4: separate mean(sc),mean(d),fma onΣC each4epochs|int32groupedC,foldcost/4,stillf32FMAemulation|onefold/4epochs,onef32sumstate|KILL .378NRMSE|
|once:mean(sc)*mean(d)*ΣC,onecast|one wholeKscale/convert, safeint32ΣC forE136|minimumoverhead|KILL .432NRMSE|

No uniform output-relative bound exists for the scale projection because C terms can cancel. Grid absolute scale error<=0.5*2^a is the actual arithmetic bound. Full exactfold writeback int32partials remains~3TOPS/writebound, not a fallback route (`docs/npu/dense27b.md:101-119`).

## 7. Halo/gfx1152 payoff: mixed versus full IEF15

Primary specs: [AMD RyzenAI7350](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-7-350.html):KrackanPoint,Radeon860M8graphicscores/3000MHz,2memorychannels,LPDDR5X8000 orDDR55600,28Wdefault/15–54WcTDP,NPUupto50TOPS. [AMD RDNA3WMMA](https://gpuopen.com/learn/wmma_on_rdna3/) gives1024denseIU4ops/CU/clock; [ROCm WMMA family support](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html) includesgfx1150–1153. gfx1152 is the user-specified target;860M is a representativeSKU, not allKrackanparts.

Derived860Mpeak8*3e9*1024=24.576IU4TOPS; assuming70%efficiency gives17.2usefulTOPS. Halo~75.6usefulTOPS,40physicalCUs@2.60–2.66GHz supports compute-ratio.225 (the Halo roofline trace `tops.md:12-17,24-32`, workspace-local). HIP may report20WGPs instead of40CUs; no double counting. Two64bit channels imply128GB/sLPDDR5X8000 or89.6GB/sDDR55600 **theoretical**, not sustained. NPU50marketingTOPS is not an IEF measurement.

Reproducible model `tools/npu/fold-model/src/bin/payoff.rs`, output [`payoff.json`](../../tools/npu/fold-model/payoff.json). Both modes optimize256weight-row cuts, **gate/up512combined rows =256paired hidden features** to retain incumbent V2B admission (`W:gemm.rs:33901-33924` calls `iu4_v2_tile` at20543-20549). Mixed GPU ratio1.00 assumes incumbent current-strided speed; full ratio1.57 transfers HaloG0 to gfx1152 as an estimate, not its measurement. Trace5287.408ms denominator remains fixed; G0's5004–5046ms microbench total is not substituted for the traced step.

`T=max(alpha*(1-f)*g*gpu_ratio, f*ops/(P*.853)+.215ms)+.020ms+metadata+NPUcolumn_epilogue`, with55GB/sNPUportcap. Measured-source traffic proxy2.44B/kop=V8input2.05+f32output.39 (`docs/npu/dense27b.md:154`);3.05B/kop is **assumed25%retile sensitivity, not worst-case bound**. Repeated-tile traffic matters; one host operand copy perGEMM is not the memory denominator. Locali64I traffic/retileactualport, producerpacking and stridedGPU costs are unmeasured. Metadata/epilogue rate200GB/s is assumed. beta.853 andGPUderate(+2.2/5.9/11.3%at5.3/16/46.9GB/s) are borrowed from Halo measurements flaggedunreliable4.13%drift (`docs/npu/dense27b.md:140-158`).

Old215usperNPUsubmit and20usringhandoff retained conservatively; persistentdenseIEF mayremove215us only afternewshapeproof. Co-op window log `coop-w2-0fe69f9.log` (workspace-local, see `report.md`) shows livealias andemptyringGPU p503.13uspub->done/5.77usconsecutivepublish/17.58uswall, andsmallfixture32/32GPU-read exactparity. These prove plumbing, notdenseIEFthroughput orgraphcorrectness. Replacing20uswith5.77us alone saves atmost4.90ms/344calls.

**All table gains/shares are ESTIMATES.** gfx1152 remainder scales2x(memoryproxy) or4.444x(computeproxy), GPU-alonestep26.55–30.27s. Endpoints include2.44/3.05B/kop andbothremaindercases, not guaranteed bounds on smallpackagepower/thermal behavior. Percentage is throughputgain, nottimesaved.

### Mixed: incumbent GPU + IEF15 NPU

| NPU raw assumed | Halo share | Halo gain | gfx1152 share | gfx1152 gain |
|---:|---:|---:|---:|---:|
|19TOPS|17.6%|+2.64–3.36%|50.9–51.2%|+48.91–60.68%|
|21TOPS|20.2%|+4.05–5.26%|54.7%|+52.94–65.24%|
|23TOPS|20.2–21.1%|+4.01–5.38%|54.7–55.8%|+53.53–72.06%|

### Full IEF15: measured Halo GPU1.57x penalty transferred

| NPU raw assumed | Halo share | Halo gain | gfx1152 share | gfx1152 gain |
|---:|---:|---:|---:|---:|
|19TOPS|26.0–26.2%|−21.97…−21.14%|61.5–63.4%|+29.29–35.89%|
|21TOPS|27.0–29.1%|−21.16…−20.80%|65.1%|+35.66–43.54%|
|23TOPS|29.5–30.6%|−21.00…−18.50%|65.1–66.8%|+35.86–48.58%|

JSON additionallyincludes26.9/30.2TOPS upper-reference scenarios, **not IEF measurements**;55GB/s cap flattens30.2. Full gfx1152's positive model does not revive killedHaloG0/G1. Mixed is strictly preferable here, avoids1.57xGPUcost, but loses split-independent whole-GEMM numeric identity. Measuring onlyHalo cannot promote gfx1152; its clocks/port/topology/power/actualNPUrates remainunknown.

## 8. Revised independent work slices (mixed only; historical plan, not implemented as written)

Reviewers own final admission. OriginalG0/G1 are killed. No project-widebuild/test/formatter midflight. This is an **implementation plan**, not a claim these modules exist. `docs/npu/coop-dense27b.md` owns buffer/sync/packing lifecycle; this document owns arithmetic/error/splitidentity.

### M0 — cheapest NPU arithmetic/rate kill gate

Slice: NPU fold. New `crates/peacemaker-npu/src/kernels/iu4_ief15.rs`, module registration, new scoped `crates/peacemaker-npu/tests/iu4_ief15.rs`, `npu-gemm` explicitIEFprobe. Reuse QT44/72BA4decoders `crates/peacemaker-npu/src/kernels/iu4.rs:36-110`, schedule/emitter `gemm_core.rs:74-115,393-437,785-820`, slotmodel `iu4_fold_model.rs:7-46`. IncumbentV8/V9/G80 untouched.

Frozen API: `Ief15Gemm::new(tokens:usize,features:usize,k:usize,ctl:Control)->Result<Self,String>`; `pack_in(&self,w:&Weights<'_>,x:&Acts<'_>,s:&[u16],a:&[i16],d:&[u16],b:&[i16])->Result<[Vec<u8>;2],String>`; `reference` samearguments->`Vec<u8>` LEf32. OwnedArrayDesignuses existing `append_run_body`/`append_lean_run_body` transactionAPI (`docs/npu/railgun-npu.md` §3.4); exportnewwavegeometry, neverassumeV9. Kernelfold:twoK64GEMMchunks ->C32;VMULS15D15->acc64,VSRS0->signedP32,VMAC32×signedC16->I64. Cconversion is lossless [-7168,8192], not saturating. Finalvariable normalization is compare/select plus fixedshifts; AIE has novectorclz/lanevariableshift. C scratch becomes finalf32output onlyafterallreadsretire.

Memory gate:128×64I alone64KiB is illegal;64×64 totals64KiB before metadata, also illegal. FreezeTM64/TN32: I16KiB+C8KiB+A ping/pong8KiB+B ping/pong4KiB=36KiB. Proposed disjoint byte offsets: A0 0x0000..0x1000,A1 0x1000..0x2000,B0 0x2000..0x2800,B1 0x2800..0x3000,C 0x3000..0x5000,I 0x5000..0x9000; maximum E136 S/D metadata0x9000..0xf600,a/b0xf600..0xf6c0, controls0xf6c0..0x10000. DMA alignment/lock/routes must pass before emit. Final f32 aliases expired C. K17408 retains one integer I across all resident-B segments, no f32 merge. Include local-I read/write cost; no19–23TOPS promise from Vecfloor.

Orderedgate:CPUoracle ->assembler/simulatorrange/RAW/memoryproof ->extremeC,cancellation,normal/subnormalties,±zero,padding,E5/20/40/48/136,changedoperands3submits -> declarednewdesignsoloNPUwindow<=90s AxQOS0/vendorBDword5, first/finalbytes,V9healthafter -> useful realdensegate_up/down/QKVrates. If measuredrate/traffic yields<+10% nettargetgain afterstridedGPU/packing, **abandon target's promotion before integration**; Halo alreadyfails+10model. Onwedge stop/report, no retry into sharedtiming.

### M1 — NPU-only sidecar production, no GPU IEF route

Slice: metadata. Own new `W:crates/rdna-compute/src/npu_ief15_scales.rs` and `W:crates/peacemaker-isa/src/kernels/npu_ief15_scales.rs`; Shared registrations are integrated centrally. Frozen CPU API `encode_npu_ief15_qt44_scales(payload:&[u8],rows:usize,k:usize)->Result<(Vec<u16>,Vec<i16>),String>` returns S/a from load-time original bytes. Frozen host API `Gpu::fill_npu_ief15_scales(&mut self,xq:&GpuTensor,d:&GpuTensor,b:&GpuTensor,tokens:usize,k:usize)->HipResult<()>` fills caller-preallocated D/b. PMargs `(Xq,D,b,tokens,K)` =threeu64ptrs+twou32,32B. Exact exponent-bit/RNE grid, zero/tie/subnormal/nonfinite rejection, no q changes or weight readback. Regenerate D/b after every A4 write; no flag-off buffers/launches or capture-time allocation. Direct-launch byte oracle with changed inputs, producer-cost measurement and emitted bundle/type handoff are acceptance; no GPU IEF fold/schema.

### M2 — current-fold strided split proof (independent of M0)

Slice: GPU current-route. Own new `W:crates/peacemaker-isa/src/kernels/iu4_current_split.rs` and `W:crates/rdna-compute/examples/iu4_current_split_probe.rs`; Shared registrations are integrated centrally. Never reuse G0's IEF five-op kernel. Anchor `W:peacemaker-isa/src/kernels/iu4_v2b.rs:92-100,508-555,687-748`; preserve WMMA/fmaf/SiLU, change feature-window addresses only. Proposed `emit(Spec{arch:Arch,epi:Epi})->Result<Emitted,String>` uses existing proof types. SET `(A,Xq,Y,M,K,N,LDY,COL0)` =threeu64ptrs+fiveu32,44B; ADD appendsGSHIFT,48B; SiLU `(G,U,Xq,Y,M,K,N,LDY,COL0)` =fourptrs+fiveu32,52B. Rebase A to local rows; Xq shared; Y offset `token*LDY+COL0+row`. SiLU M counts paired hidden features (34816combined rows=>M17408), LDY/COL0 are hidden-output units. Keep M%256 admission; paired cut256hidden/512combined. Flag-off config/schema/bundles unchanged.

Allsevennamedshapes, splitZ|beta|alphastoresemantics notplainSET, FAprojections, o_projdeferredresidual, gate_upanddownADD. Callsiteanchors `W:rdna-compute/src/gemm.rs:20522-20594,31555`; `hipfire-arch-qwen35/src/qwen35/prefill.rs:249-259,433,5744-5790`. RunLSPfullreferencesbeforeexportchanges. A4produceralreadyshared. CapturemustnotfallthrougholdV2Beageronlypredicate atgemm.rs20532-20533toanothernumericroute. GPU-ownedcolumns byte-equal incumbent; sentineluntouchedNPUcolumns,LDY>M,pairedGU,ADDoldYpadding. ScopedactualPMlaunch,eager/graph/Redlineexactparity;>=3freshprocesstiming identicalinputs. Current-strided>3%weightedslowdownorpackingcostbreakingtargetmodel killsmixedpromise; do notreuseG0cost-freeGPUassumptionafterfailure.

Buildsequence forM1/M2:scopedCPUoracle; `cargo build -p peacemaker-isa --features toolchain --bin peacemaker-isa`; RustPMnativeemitter `--co/--bundle` (`src/bin/peacemaker-isa.rs:23-25`), nohipcc; scopedrdna-computeprobe. Sharedregistrationwritesareintegratedcentrally; /emittedassetinstallationtoavoidraces. M0andM2 canrunindependently withserializedhardwarewindows; M1'sbuffers feedM0+integration afteritsoracle.

### M3 — caller-owned fixed-map integration/final gate

Slice:co-op,consumeM0/M1/M2interfacesaftergates. Ownring/bufferplan `docs/npu/coop-dense27b.md`; no arithmetic redefinition. Preserve ringv1 offsets/sequence-lastpub,16Cchanneljoin-beforeDONE, release/L2writeback/acquireinvalidate (`docs/npu/railgun-npu.md` §§3–4). Shared controlplaneRustonly. Newmixedidentityincludesexactcolumnmap; fixedmapduringcapture/replay/cacheuse; no silentcurrentfallback. No extraGu×Upmismatchedcut orpartialf32downsum. Newshape packing mustpriceV8-styleAduplicationoverfeaturewaves, notcompactMKonce.

Afterallland,sharedclaim-scopedvalidationrunsonce. Authority `W:docs/VALIDATION.md:24-38,279,292-301`:path-specificnumericoracle,test_kernelsexamplechannel,servesemantics,andRedlineladderwhereapplicable; stationaryfreshprocessmatchedperformance. Coherence-gatefamilyretiredat309-318, neveracceptance. >=3freshprocessruns, promptMD5andbinaryMD5,byte-identicalprompt,eyeballdecodedoutputs,nosingle-tokenattractors,actualhipGraph+PM4proof. Measurednetpp8192>=+10% is requiredforrequestedtargetclaim; belowthresholdisnotpromotedaswinforit.

Notdoing:Flashintegration,artifactrequantization,AWQ/FWHTretuning,KLD-basedchoice,defaultenablement,driver/firmware/QoSchanges,dynamicP2ring,TS/Pythonruntime,hipcc,localGPUwork,orautomaticportabilityfromgfx1151togfx1152/gfx1201/gfx906/gfx10xx/gfx94x. Archgateeverynewentry; gfx1152 needsitsownadmission/measurement.

## 9. Reproduction and observed CPU proof

```
export CARGO_TARGET_DIR=<scratch-target-dir>
cargo test --release --manifest-path tools/npu/fold-model/Cargo.toml --offline
cargo build --release --manifest-path tools/npu/fold-model/Cargo.toml --offline
# The CPU dense model run used a test-host copy of the artifact under the host's timing lock.
fold-contract-model <path>/qwen3.8-27b.mq4-xts dense-errors.json
report numeric-tables.md dense-errors.json
payoff payoff.json
report --payoff payoff.json
```

Observed here:7CPU tests passed; stored-artifact126,976fold comparisons and renderer/both payoff modes executed. rustc1.98.1(48a229cea2026-09-01). CPU model binary MD5 `8fcf75d6abe8bc315304b1070138dbea`; dense evidence MD5 `44d8d1c4fda0802a4b8674424a1f9fc8`; final aligned mixed/full payoff MD5 `f844e93bee923c9b32d7a5d8f19f4ebd`. Seed and weight/activation FNV64 in JSON. No real dense prompt MD5 exists in this CPU experiment; not device acceptance. Peer G0 proof is separate in§1. No GPU/NPU window in this study; no project-wide build/test/formatter/linter.

Archivalprior-to-deprioritizationFlashaudit:canonicalGPTQ3SHA256fresh `8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972`;553RTNfallbackexperts,bothprojections/allrows/epochs,21,235,200headers,zeroasymmetric/nonfinite/negativescales. [`gptq3-rtn-audit.json`](../../tools/npu/fold-model/gptq3-rtn-audit.json). Noaffineextensioncriticalpath. NoFlasherror/payofftableinthisdeliverable.
