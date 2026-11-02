# G0 inventory and CPU kill experiment

> **Historical report.** This is the G0 (phase 0) investigation that preceded the load-time kernel JIT work. Baseline commit `83cb363e8a`; observations were taken on 2026-10-08. Every measured table below is bound to the fixtures, hosts and toolchain of that day and is **not** a statement about current beta or product performance. The pinned hashes are preserved as recorded. Labels such as A1–A5 name numbered work **phases** of that effort (not people or agents): A1/A2 job control and concurrency, A3 the closed route plan, A4 admission and loader reserves. Statements below that something "was not implemented" describe the G0 baseline only.
>
> **Current state.** The load-time JIT now exists: [`HIPFIRE_JIT_JOBS`](env-vars.md) is implemented in [`compile_jobs.rs`](../crates/rdna-compute/src/compile_jobs.rs) and [`hipfire-loader/src/lib.rs`](../crates/hipfire-loader/src/lib.rs), and the closed route plan is built by `route_entries` in [`kernel_registry.rs`](../crates/rdna-compute/src/kernel_registry.rs). The user-facing behavior and its limits (which routes are closed, what every other route does) are documented in [env-vars.md](env-vars.md) under "Load-time kernel barrier (limited routes)" and in the 0.4.2 entries of [CHANGELOG.md](../CHANGELOG.md). Read those, not this report, for what ships.

**G0 PASS, with the explicitly excluded Flash-Next (FN) gfx1100 fixture.** No product change from the later phases was included in G0. Temporary observation logging was removed after the experiment; the reproducer patch for it is not checked in.

## Six pinned inventories

Manifests are committed under [`crates/rdna-compute/tests/fixtures/jit-g0/`](../crates/rdna-compute/tests/fixtures/jit-g0/).

| Manifest | Modules | Default required symbols | Explicit helper superset symbols | Evidence |
|---|---:|---:|---:|---|
| [`27b-xts-gfx1201.routes.json`](../crates/rdna-compute/tests/fixtures/jit-g0/27b-xts-gfx1201.routes.json) | 45 | 51 | 8 | observed-point-routes |
| [`27b-xts-gfx1100.routes.json`](../crates/rdna-compute/tests/fixtures/jit-g0/27b-xts-gfx1100.routes.json) | 25 | 30 | 6 | observed-point-routes |
| [`27b-xts-gfx1151.routes.json`](../crates/rdna-compute/tests/fixtures/jit-g0/27b-xts-gfx1151.routes.json) | 27 | 31 | 4 | observed-point-routes |
| [`flash-next-gfx1201.routes.json`](../crates/rdna-compute/tests/fixtures/jit-g0/flash-next-gfx1201.routes.json) | 42 | 106 | 8 | observed-point-routes |
| [`flash-next-gfx1100.routes.json`](../crates/rdna-compute/tests/fixtures/jit-g0/flash-next-gfx1100.routes.json) | 29 | 67 | 6 | source-predicted; excluded fixture |
| [`flash-next-gfx1151.routes.json`](../crates/rdna-compute/tests/fixtures/jit-g0/flash-next-gfx1151.routes.json) | 17 | 28 | 4 | failed-default-request-observation |

Each HIP entry pins the exact source path/SHA256, full flags, architecture, scheduler profile, cache ABI, toolchain and required symbols; embedded entries pin image bytes/targets/exports and bypass HIP compilation. Default AR/native-MTP membership is separate from historical explicit-spec helper supersets and serial-compiler kill arms. The observed FN-gfx1201 AR/PM-off arm is separately recorded. `HIPFIRE_PACK_JOBS=1` was the existing experiment control; at the G0 baseline `HIPFIRE_JIT_JOBS=1` was a planned override recorded in the manifests (`planned_jit_override_status`) and not yet implemented — see the current-state note above.

All 24 cold-JIT source hashes recorded for the 0.4.2 gfx1201 cold-start observation are covered by selected-symbol entries and [`cold-24.json`](../crates/rdna-compute/tests/fixtures/jit-g0/cold-24.json). Mixed fixtures never establish default membership. All 508 TSV source MD5s matched their pinned files. The cold-start observation directory itself is not checked in.

These are G0 evidence inventories, **not closed ready-plan oracles**. All-row/tail/max-context/graph selector coverage and separately admitted drafter headers were left to phase A3 (see the committed [A3 closure audit](../crates/rdna-compute/tests/fixtures/A3-closure-audit.md) and the phase-A3 fixtures in [`jit-a3/`](../crates/rdna-compute/tests/fixtures/jit-a3/)). gfx11 dense inventories include completed prefill observations, not the reset ordinary-decode attempts. The 28 FN-gfx1151 entries are requests before a failed native prefill, not a successful full route or numerical proof; nothing downstream of that failure may be inferred from it.

The exact canonical Flash-Next model SHA256 is `8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972` (verified equal on both test machines); its header SHA256 is `8703bb1626ff48b976a84d5f4f7d18b4aad8e9d1d9b780a52f963d73cb83c013`. The 27B header SHA256 is `2a4f495e9077a1c68ec4b923b00b7fba5a2206acdad17c7c73ede9ea934a5de0`, likewise verified equal on both machines. Full tensor metadata is retained in the committed header fixtures ([`header-flash-next.json`](../crates/rdna-compute/tests/fixtures/jit-g0/header-flash-next.json), [`header-27b.json`](../crates/rdna-compute/tests/fixtures/jit-g0/header-27b.json)); only tokenizer metadata is omitted.

## Excluded fixture, not a JIT-resource defect

Canonical FN default AR on gfx1100 is source-predicted only. On a 24 GiB gfx1100 (XTX) test host with approximately 30 GiB of system memory, it is inadmissible: 96 trunk expert tensors alone total 64,172,851,200 bytes (59.765625 GiB). Even allocating **all** 24 GiB of VRAM to experts leaves at least 35.765625 GiB of host expert payloads. That host's `/proc/meminfo` reported MemTotal 31,960,744 KiB (30.480141 GiB), so the lower-bound shortfall is 5.285484 GiB before MTP, fixed device weights, staging, scratch, safety or compiler jobs. This fixture cannot load on that host at all; **no load-time JIT guarantee is claimed** for it. Admission and production loader reserves were left to phase A4. The detailed exclusion record is not checked in; the committed [`flash-next-gfx1100.routes.json`](../crates/rdna-compute/tests/fixtures/jit-g0/flash-next-gfx1100.routes.json) carries the source-predicted inventory only.

## CPU-only kill experiment

The per-process receipts (frozen input manifest, registry TSV, provenance, summary, required-export list, and nine per-process command/strace/RSS/time/result files) are not checked in; the tables and digests below are the recorded results.

The experiment used the existing bounded pack compiler (implementation commit `1767881e81e1`, prerequisite branch tip `74a8abf5e8`); its compiler source matches the baseline. Builds used `-j 28` and `ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1`. No new execution runtime or compiler executor was added.

Planned set: 44 HIP module aliases, **43 full source/flags/profile/ABI recipe groups**, 47 required HIP exports, plus one embedded image supplying 4 required exports (51 total). The two `qwen35_fa_prep_*` aliases retain their names and union both required symbols while compiling once. All 47 required HIP exports were found in the actual measured ELF objects. Installed-hit arms used a private executable-relative installed pack seeded from serial 1, with fresh private HOME/cache/lock/output directories. Every arm has three fresh pack processes; CPU-only env is pinned in each command receipt.

Budget: 64 affinity CPUs; default CPU request 56; MemAvailable 68,166,197,248 bytes; reserve 63,215,166,464 bytes (three times the 27B file bytes + 16 GiB staging/scratch + 1 GiB safety); remaining 4.611007 GiB; 1536 MiB/job clamp => **3 workers**. All inspected cgroup ancestors had `memory.max=max`, with no tighter finite cap. This is a conservative experiment admission bound, not a production loader-memory estimator. The 1536 MiB/job figure was kept despite lower observed RSS.

| Arm | Fresh process | Wall s | User+sys s | Sampled peak tree MiB | HIP compile invocations | Peak active compilers |
|---|---:|---:|---:|---:|---:|---:|
| serial | 1 | 25.711841 | 25.87 | 240.10 | 43 | 1 |
| bounded | 1 | 9.333515 | 27.04 | 580.79 | 43 | 3 |
| hit | 1 | 0.103486 | 0.22 | 52.90 | 0 | 0 |
| bounded | 2 | 9.249586 | 26.87 | 577.93 | 43 | 3 |
| serial | 2 | 25.927736 | 26.06 | 239.52 | 43 | 1 |
| hit | 2 | 0.103468 | 0.20 | 58.64 | 0 | 0 |
| serial | 3 | 25.765523 | 25.91 | 239.68 | 43 | 1 |
| bounded | 3 | 9.428430 | 27.23 | 587.24 | 43 | 3 |
| hit | 3 | 0.124684 | 0.26 | 148.05 | 0 | 0 |

All nine return codes were 0. Tree RSS was sampled every 20 ms; the highest bounded sample was 587.24 MiB, below the reserved compiler budget. `strace` counted compile invocations and peak active compiler processes; installed hits compiled **zero** recipes.

Every run's 129 object/hash/index files were byte-hash-identical. The common sorted filename-to-SHA256 artifact-set digest is:

`61000acde94372919a5bcec7e175fae0376c469436fa28a8cf0c4a5a7116d59a`

Frozen input manifest SHA256: `c5cb33aa4983fc7de0856e5124b27a00cfa97afec76f273e3761e66d24e186ff`.
Pack binary SHA256: `b20aa944c62144848cb86464a9bc258142e2104c370e0049a678da3673fc28c3`.

Median serial 25.765523 s, bounded 9.333515 s, installed hit 0.103486 s: **63.7752% compile-wall reduction**. Using the previously observed 2.834 s weight interval, `max(weight, compile)` predicts 25.765523 => 9.333515 s (63.7752% reduction), with installed-hit critical path 2.834 s. Thus the CPU kill gate exceeded the 20% compile-wall and 10% predicted critical-path thresholds with identical artifacts and a safe bounded resource model: **PASS**.

This does not measure real load overlap, driver loading, alias publication, ready/first-response latency, numerical equivalence or throughput. The prediction is not an observed end-to-end load improvement. No arithmetic/source reorder is included; the intended steady-state throughput ceiling remained 0% change.

## Source findings handed to phase A3

- Four live tensor_ops q8 exports are absent from legacy registry declarations/TSV export lists but present in actual ELF objects: `indexed_attention_attention_q8_batched`, `indexed_attention_attention_q8_batched_hg4`, `indexed_attention_cache_append_q8_batched`, `indexed_attention_decode_prologue_q8`. Inventories record the declaration gaps rather than hiding them; no product registry correction was part of G0. (Historical G0 measurement: `registry_missing_symbols` in the fixtures is kept as data. The later packaged-index closure repair, HOLD1-R2-REVIEW section 3, added these four plus `indexed_attention_attention_q8`, `_q8_serial`, `_q8_batched_serial` and the canonical `gated_delta_step_halves_state128_persistent256_canon_q8`/`_q8ef` to the gfx1100/gfx1151 and gfx1201 `tensor_ops` inventories; `jit_g0_route_data` now requires every measured symbol to be in the live registry.)
- Current canonical Halo HC-down selected `gemm_wmma_lds_160_64_32_64_k64_p`; the historical four-entry LDS census is not a closed default route. HC row-fold and all selected shapes must be audited before port ranking.
- Native MQ6 shape coverage is not blanket coverage: gfx1151's public thresholds at the time included M>=2560 and N>=2048; gfx1201 BF16/regions paths remain HIP. gfx1100 has no corresponding native image.
- `HIPFIRE_QWEN4_MOE_SYM_PM=0` restores symmetric MoE HIP only on gfx1151, not gfx1201; the HIP TU is already required for its symmetric-check export. No whole-TU saving is inferred.
- FN HC defaults differ (gfx1151 level 3, gfx1201 level 1, gfx1100 level 0); the gfx1100 host-mapped default is AR, not Halo's native-MTP default. QSA/HC producer and consumer coverage must share resolved policy, not mixed-fixture names.

## Observation validity

Recovered result mtimes on the second test machine (2026-10-08): 27B-gfx1100 `19:55:27.343Z`; 27B-gfx1151 `19:58:40.891Z`; original FN-gfx1151-native `20:02:13.228Z`. All precede an invalidated host-memory-pressure (OOM) window of 20:25–20:36Z. The original FN run returned 1 after an HSA fault; its request receipt is retained but not checked in, with **no causal kernel-fault attribution** and no memory/process-pressure telemetry. All queued reset/timed-out follow-ups and runs inside the invalid window are excluded from acceptance evidence. No retry wrapper or recursive relaunch was added to the experiment tooling.

## Pure route-data tests

Seven pure tests live in [`jit_g0_route_data.rs`](../crates/rdna-compute/tests/jit_g0_route_data.rs), using the six manifests, two header fixtures and the 24-source fixture in [`tests/fixtures/jit-g0/`](../crates/rdna-compute/tests/fixtures/jit-g0/). They cover baseline factories/recipe identities/embedded exports, all 24 cold sources, header quant/shapes and separated defaults, full-recipe alias union, conflicting public-entry image refusal, existing architecture/shape-selector constants, and G0-pass versus A3-readiness/excluded-fixture distinctions. They do not initialize HIP or add a model-route execution runtime. The serial/spec arms resolve their recipes through the manifests' `entries_ref` fields rather than inline `entries`.

At the time of the G0 handoff the suite was run CPU-hidden and recorded as 7 passed, 0 failed (0.03 s). That result is historical and was **not** re-run for this document. To reproduce from a source checkout root:

```sh
env ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 cargo test --release --locked -p rdna-compute --features deltanet --test jit_g0_route_data
```

Set `CARGO_TARGET_DIR=<dir>` in the environment to use a build directory other than the default `target/`.

G0 concluded **PASS with the FN-gfx1100 inadmissible-host exclusion**. The later phases (A1–A5) were dispatched separately; their current outcome is described by the source and documentation linked in the current-state note at the top, not by this report.
