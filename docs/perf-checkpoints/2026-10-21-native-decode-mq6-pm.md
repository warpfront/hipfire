# Native Decode MQ6 PM admission — 2026-10-21

**Lifecycle: adopted, fixture-scoped numerical and endpoint evidence.** Parent `b992a5684cb2a93d7d54f5369b3f8a608c180caf`, branch `experiment/halo-native-pd-20261021`. Local only, no push. The full GPU–NPU Prefill plus AR/MTP Decode objective remains active; the previous full NPU drafter remains stopped. This cycle changes native GPU Decode dispatch; NPU programs, shared arenas and Prefill bank geometry are unchanged.

## Removed work and admission

The existing certified gfx1151 MQ6 PM narrow kernels share X loads across four weight rows per wave and use `ceil(M/4)` workgroups (x4 sums the four ceilings). The prior native singleton Qwen4 executor did not enable lane_rows_scope, so its 2–8-token MTP projections used HIP row kernels even though the PM image was already in the repository. This cycle admits the existing exact arithmetic/image for those native calls. No GPU arithmetic, weight format, precision, quantization or reduction order is changed; no new conversion, per-forward allocation, GPU pass or GPU–NPU handoff is introduced.

The new predicate requires the existing `HIPFIRE_FN_LANE_PM_ROWS` flag, exact gfx1151, Qwen4 scope, no lane scope, actual rows2..8, and no canonical-Prefill row remapping. The old lane/capture guard and all Q8 selection remain. Calls whose actual row count exceeds8 explicitly disable only the new native policy in recursive fallback groups; one-row AR compute remains unchanged. AR initialization does additionally preload the PM image when this single-scope gfx1151 flag is on, so load behavior is not claimed unchanged.

`Qwen4GpuForward::new_inner` resolves both narrow exports through the existing pinned loader before scratch/capture. gfx1151 has no walked closed route, and `route_entries` continues to refuse partial plans. An actual registry fixture checks that refusal for AR and NativeMtp. An earlier V1 preload helper was placed on the unreachable walked path; V2 added the real constructor hook. Final V3 removes that unused helper/interner. V1's oracle initially failed compilation because FeatureFlags is Arc-owned; the lab driver now uses `Arc::make_mut`. All failed/intermediate sources and logs remain preserved.

Exact named replay contracts are single R@0/R@8/W@16 and x4 R@0/8/16/24/32, W@40/48/56/64. `launch_maybe_blob` owns and pads single raw36B to48B; x4 is96B. Unknown suffixes remain refused. The shipping fatbundle is unchanged, SHA256 `8ea2cac1b40aae940b26da08348d28db6f2fb67cc4797aeda6347499303be945`. The first metadata attempt treated the bundle as ELF and failed; the retained successful extraction reads its ELF member at offset4096,789752B, SHA256 `f164ffa5c7fbaad537ad4c654bb553e09e04aa335befe81cb2d6d049245f9ac2`.

## Numerical and runtime validation

- Final V3 hardware oracle:510 nonvacuous full-allocation/guard/finite exact checks against independent frozen HIP-row and per-token GEMV anchors. Ten shapes include native GDN QKV/out, QSA out, vocabulary head and QSA q/k/v/idx x4, plus ragged dimensions/K tails and empty x4 parts. Direct PM covers N1..8; actual native eager/forced-blob/record routes cover N2..8.
- Seventy recorded launches verify selected entries, source/image provenance, owned kernarg ABI, grids and pointer effects. Twenty-five native GraphState captures each replay A/B/A with changed X at the same address:75 actual graph replays, independent HIP reference each time, unchanged owned blob hashes, and actual B output differences. Both PM exports are tested, rather than only calling raw kernels.
- Four CPU contract tests pass, including exact scope exclusions, ABI/effect metadata and the actual gfx1151 closed-plan refusal. These do not replace hardware tests.
- Actual-model identity: five fresh child processes, pinned125GB MQ4-XTS model, Q4 PLE and trunk overlay, F32 GDN/q8 KV/HIP replay. AR plus batched, incremental, adaptive and full-head each generate256 identical IDs. Observed window counts are78 batched,73 incremental interleaved,84 adaptive batched,79 full-head batched. This arithmetic test deliberately generates through EOS; its post-EOS repeated special tokens are not natural service-quality evidence.
- Both frozen daemons pass scoped Redline BF16/AR, splitK0, NPU off: stable capture, valid AQL contracts and multi-position exact state/logit parity. This retained AR fixture does not itself prove multirow PM capture; the native GraphState oracle above supplies that evidence. Formal q8/MTP keeps default attention and chooser.
- Four normal service processes,16 turns: AR/MTP baseline/candidate full assistant/reasoning/tool-call/request transcripts match. Inputs21607/24/38/243 and outputs97/43/156/48 all end naturally with stop and one terminal. Decoded code summary, primes, merge sort and LRU were read. No empty/attractor/runaway/stream errors. Actual Qwen4 logs show q8/VMM and Q4 PLE. Null harness KV markers stay UNOBSERVED, separately from actual runtime evidence.

The PM image metadata reports255VGPR, single45/x4 85SGPR, zero LDS/scratch, and raw ABI36/96. Actual HIP occupancy API reports24 active blocks/MP for both PM kernels, versus64 for HIP R2 and56 for HIP R3/4/5. Different work per block makes this an inadequate standalone performance explanation; no achieved-occupancy/PMC/cache-bandwidth claim follows.

## Isolated exploratory screen

Before integration, a deterministic synthetic M10240/K2560 pilot tested N1/2/3/4/5/8 against unchanged HIP rows and per-token references, including ragged/header controls:48 cases per process across one check plus three timed fresh processes,192 exact records. The timed packet contains one each of N2/3/4/5; it is not the native1:12:8:1 row distribution. Each process has10s DPM warmup, calibration so each arm's packet sample lasts at least20ms, and seven alternating paired samples; all observed samples exceed10ms.

Three fresh HIP→PM packet medians:288.046→165.600us,290.227→166.676us,290.512→167.782us, time reductions42.51/42.57/42.25%. All raw samples/operand hashes remain. Only before/after telemetry was captured here, not per-sample power/time series. This is synthetic primitive evidence, not a42% whole-model or GPU–NPU speedup.

## Formal native endpoints

Twelve fresh daemon processes: AR candidate/baseline/baseline/candidate/candidate/baseline; MTP baseline/candidate/candidate/baseline/baseline/candidate. Each has one matching warmup and one measurement,10s DPM warmup, byte-identical21582-token prompt MD5 `b4d0b63cddcac872648ddf3cdd92cac2`, q8/VMM, NPUauto, resident weights, PLE8GiB, caches off, maxseq32768 and256 output budget. MTP actual warmup/measured output counts are256; AR CLI actual counts remain UNOBSERVED. No profiler, route dump, remote CPU build, heavy audit or model rehash overlaps formal timing. Global flock and native per-PCI lease handoff serialize GPU use. All raw power/PPT/clock/thermal/memory objects are retained.

| route | baseline TTFT ms raw | candidate TTFT ms raw | median TTFT delta | baseline Decode tok/s raw | candidate Decode tok/s raw | median Decode delta |
|---|---|---|---:|---|---|---:|
|off|9644.7, 9635.3, 9652.1|9581.6, 9695.3, 9641|-0.038%|33.7, 33.1, 33.8|33.7, 33.7, 33.7|+0.000%|
|mtp|9831.9, 9841.9, 9845.3|9792, 9774.4, 9797.5|-0.507%|51.5, 53.2, 53.6|55.6, 54.5, 55|+3.383%|

MTP tau raw: baseline 1.8, 1.87, 1.87; candidate 2.04, 1.9, 1.97. Walltime-dependent scheduling remains a confound; no sums of cross-cycle percentages or attribution of all endpoint variation to this kernel. Telemetry complete: True. Investigation flags: `[]`.

## Decision

Adopt the exact native MQ6 PM dispatch and capture/preload contracts. All three candidate MTP samples (55.6/54.5/55.0 tok/s) exceed all three baseline samples (51.5/53.2/53.6); medians53.2→55.0, observed+3.383%. AR medians33.7→33.7. MTP TTFT9841.9→9792.0ms (−0.507%); AR9644.7→9641.0ms (−0.038%). No ≥5% endpoint or persistent-decode-loss investigation flag fires. This supports the user's preference to retain verified work removal without requiring every small endpoint metric to improve.

MTP tau is1.80/1.87/1.87 versus2.04/1.90/1.97. The normal production speculative route retains its built-in n-gram lane; this is not a native-only ablation. Endpoint benefit includes whatever the normal chooser/schedule did; it is not proof of a pure-kernel3.4% critical-path saving. Measured-run aggregate GPU clocks are2739.0/2746.7/2743.1 versus2742.1/2754.5/2750.4MHz, socket power117.1/117.1/116.9 versus115.0/114.7/115.3W, slow-PPT residency55.5/56.8/56.9 versus53.1/50.3/51.3%. Full raw telemetry is retained; this is not power equivalence or an isolated explanation. The lower prior baseline51.5 and higher candidate55.6 remain in all statistics.

For the user's overnight status question, historical initial MTP median51.7 versus current55.0 is a descriptive+6.383% cross-cycle difference. Initial AR33.7 remains33.7; initial AR TTFT9683.6 versus9641.0 is−0.440%, initial MTP9885.1 versus9792.0 is−0.942%. Earliest AR/MTP baselines differ by one tiny bank-retirement change, and these were measured in different rounds. No cumulative matched A/B or stable general improvement is claimed; do not add local/cycle percentages. This current cycle's twelve fresh matched processes establish the narrower comparison above.


## Actual native dispatch and unchanged NPU pipeline

A separate 64-token ordinary-IPC rocprof diagnostic observes actual PM single: 1,700 calls / 178.441 ms, and x4: 240 calls / 32.571 ms between first-token and done host receipts; old HIP MQ6 row launches in that tail are0. These are dispatched kernels, not merely preloaded names. Expected four-output-row workgroup grids are verified. PM batch N is in kernargs and cannot be recovered from CSV grids.

The main routed GU remains960 calls/217.680ms, and routed down133.092ms. These duration sums are diagnostic attribution, can overlap and depend on the chosen windows. Do not directly subtract the previous profile's238.104ms/2134 MQ6 calls from this profile's different call counts and schedules to claim saved latency.

NPU trace retains96 records,48 layers × A/B, live8192-row banks at positions0/8192; actual plan leaves5198 tail rows on GPU. Visible-wait proxy `max(npu_done-wait_begin,0)` totals111.920ms, startup111.920ms, nonstartup0.000ms. Instrumented spans prove this observed Prefill plan/overlap scope; they are not Decode NPU latency or device utilization. Profiler/trace perturb the chooser, and approximate CLOCK_BOOTTIME host-receipt phase boundaries are preserved with crossing-interval handling and clock evidence. No PMC/DRAM/cache traffic or per-window exact kernel IDs were captured.

## Provenance, CI and follow-up

Candidate normal-feature daemon SHA256 `4a00fce149669b6f74e663071232674190536640195f77b562bb8c14363e4c31`; baseline `b205e0d092b2079a9826f133bb45602c5aae3eb80483b8c47e0f9a97d171f358`; common CLI `588beda36f29307aecd8ecd598a36ebc5af69c44cb2b2b6c3a630319e2c0e369`. The baseline is the earlier aa98 functional binary reused after checking aa98→b992 changes are docs only; this is not a fresh b992 binary identity. Final source copy exactly matches the five tested V3 runtime files plus env-var documentation. Lab example/Cargo manifest changes stay isolated; source/oracle snapshots are embedded in the receipt. No kernel image is rebuilt or replaced. Intermediate V1/V2 failures remain separate from V3 evidence.

Full real GPU-free `scripts/pre-commit-ci.sh origin/beta` passed on the copied production code; actual log and tool identities are retained. Commit hooks are recorded separately, and absent hardware fixtures remain skips rather than passes. Full125GB model/PLE/overlay pins are reused after exact size/mtime checks for formal benches; the separate existing identity fixture itself hashed them. Adjacent compressed receipt embeds raw logs, snapshots, scripts, numeric records, profile CSV and formal telemetry; large executables/model remain externally pinned.

Adoption follows exact correctness, verified removed launch/load work and endpoint review, consistent with retaining worthwhile work removal despite small fluctuations. It is separate from establishing a stable general whole-model speedup. Next isolate a PM kernel with a smaller capacity for native N2..5: the existing B8 image declares255VGPR even for the dominant N2..5 calls, so a smaller static capacity may remove unused register storage without changing the per-output F32 DAG. This is a source-supported hypothesis, not measured gain or a reason to change dispatch yet; compare independent old PM/HIP anchors and actual native shapes before any admission. Routed GU and down remain other large measured Decode costs. Do not return to the stopped full NPU drafter or infer Decode offload latency from long-Prefill NPU chains.

Evidence root: `.codeinsight+research/halo-native-pd-20261021/decode-mq6-pm-cycle-v1`. Goal remains active, no push.
