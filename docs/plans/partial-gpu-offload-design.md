# Partial GPU Offload — Design Plan

**Status:** shipped — v1 covers the qwen35 dense path (§9 steps 1–2). The §5–§8 text
below is the *design-time* plan; read the as-shipped note first before implementing from it.
**Author:** (agent)
**Created:** 2026-09-23
**Companion:** `docs/VALIDATION.md`, `CLAUDE.md` § Runtime validation, AGENTS.md §6 pitfalls
**Related plans:** `docs/plans/dispatch_1.2_gpu_verify.md`, `docs/plans/multi-gpu-pp.md`,
`docs/plans/2026-07-28-pr549-vmm-all-kv-adaptive-design.md`

> **As-shipped note (2026-09-26).** §5 Option A (staging arena + drop graph capture) and the
> `MemoryClass`/`HostPinned` vocabulary below were the *design-time* plan and are **not** what
> shipped. What shipped reads host memory directly with the existing kernels, unchanged, via
> `hipHostMalloc(hipHostMallocMapped)` — and the choice of host mechanism turned out to be the
> single decision that makes the feature work at all. Read
> §6.1 "Implementation decisions settled during bring-up" **before** implementing anything from
> §5/§6.1's option list; it records the mechanism measurement, the reader seam, and the
> fail-closed rules as built.

> **Review ask.** This is a capability that does not currently exist in hipfire. The plan is
> correctness-first: get models that do not fit in VRAM to *run*, then optimize. Speed is a
> second-order concern and is explicitly bounded by PCIe bandwidth (see §7). Reviewers should
> focus on (§10) whether the memory-class approach is the right linchpin and whether the graph-
> capture gate in §4B is sufficient.

---

## 1. Problem & goal

### Single-architecture-first decision (v1 ships qwen3.8-27B dense only)

This plan implements partial offload for **exactly one LLM architecture first**, then broadens.
Grounded in the current tree:

- **The two breaking invariants (§4) are new code regardless of arch.** Spilling breaks kernel
  memory assumptions (WMMA/GEMV read device-local operands, §3) and hipGraph pointer stability
  (`OpBinding` binds fixed kernarg pointers that dangle if weights move — the documented silent-
  corruption class, AGENTS.md §6). `forward_slots.rs` captures ~1390 launches/step into per-context
  graphs (`SlotDecodeGraph`, `DECODE_GRAPH_CTX_BUCKET=256`), so (b) is high-stakes here. Isolating
  both to one arch before touching breadth is the correctness-first call, not a shortcut — spreading
  untested novel logic across five arches multiplies the failure surface with no parallelism gain.
- **qwen3.8-27B dense lives in `hipfire-arch-qwen35`** (the Qwen decoder family covers qwen2/3.5/
  3.6/3.8; there is no separate qwen38 crate). So "one arch" is not a proxy — it is the §1 primary
  agentic-coding target, implemented directly.
- **The dispatch substrate is arch-agnostic + shared.** `superop.rs` resolves every `OpBinding` by
  index against POD indices + `KernelKey` + flavor; the kernel family lives in an arch-side
  `ForwardBindings` impl (§6.3). A correct qwen35-dense implementation produces the pattern that
  broadens mechanically to llama/gemma4/lfm2moe — re-researching per arch costs more than doing it once.
- **The single-arch spike is also the broadening template.** §9 step 1's go/no-go (byte-identical vs
  resident) validates both the memory-class/staging/graph-gate mechanism *and* the per-arch migration
  shape, so v2+ arches reuse it rather than re-inventing placement.

### v1 scope guardrail — dense path only; MoE/EP excluded

The qwen35 crate bundles dense **and** A3B-MoE + sealed-expert-parallel machinery (a 7603-line
`forward.rs`, a 2984-line `weights.rs`). This plan touches the **dense** weight tables and the
**dense FFN** forward body — `weights.rs` `Qwen35Weights`, `forward_slots.rs` `dense_ffn_body_slots`
and the `run_*_layer_slots` dense runners. It deliberately does **not** touch MoE/EP — routed-expert
staging, sealed-EP binding, and `prefill_moe_ffn_body_batched` are orthogonal to offload (§10 Q5)
and ship in a later phase. The two use cases still share one mechanism:

1. **Long context on the fitted 27B.** Weights fit (~15 GB MQ4XT), KV is starved (~1 GB free → ~16–24K
   tokens). Offload a few layers to buy the context agentic coding wants. *(primary v1 use case)*
2. **Run-the-unfit (co-primary motivation, v2).** The 35B-A3B MoE at MQ4 exceeds VRAM and will not
   load; offload enough layers to fit. Co-primary as motivation but its MoE path is out of v1 scope.

The 9B row in the table above (plenty of KV headroom) needs no offload and ships nothing. The 35B-MoE
row motivates why the mechanism matters; it is a co-primary goal, not a side benefit — but MoE/EP is
v2+.

### Out of scope (for this plan).
- Broadening beyond qwen3.8-27B dense: llama, gemma4, lfm2moe (`hipfire-arch-*` crates), and the
  qwen35 A3B-MoE / sealed-EP path (§9 step 3).
- Multi-GPU tensor/pipeline parallelism (`docs/plans/multi-gpu-pp.md` owns that).
- Making offloaded layers fast. PCIe-bound compute is acceptable as the initial cost — both use
  cases trade speed for capacity, exactly as llama.cpp does with `-ngl`.
- Unified-memory APUs (Strix Halo): weights and KV already share one RAM pool, so per-layer
  VRAM/RAM splitting is largely moot there — the OOM guard (`memory.oom_guard`) owns that path.

### Definition of done. The qwen3.8-27B MQ4XT dense model runs at a target context length by offloading
the minimum layers to fit it (use case 1), and an over-capacity dense model loads and runs (use case
2); in both, output is byte-identical to the resident path, with a placement policy driven by a KV
budget (or weight-fit) and an opt-in config knob; validated on qwen3.8-27B dense before broadening
(§9). MoE/EP offload ships after this gate.

**Goal.** Add configurable partial offload: keep a chosen suffix of layers **and as much KV cache
as fits** resident in VRAM, spill the rest to host RAM, producing byte-identical output to the
pure-VRAM path. The two use cases share one mechanism and differ only in *how many* layers are
offloaded — long-context is "offload the minimum to fit the target KV budget"; run-the-unfit is
"offload enough to fit the weights at all."

**VRAM accounting on a 16 GB card (RX 9070 / 9070 XT), hipfire's real models:**

| model | MQ4 weight footprint | 16 GB fit? | offload needed for |
|---|---|---|---|
| Qwen3.5/3.6 **9B** MQ4 | ~5 GB | yes, comfortably | nothing — plenty of KV headroom already |
| Qwen3.8-27B **MQ4XT** dense | ~15 GB | barely (~1 GB free) | long context: offload a few layers to buy KV space |
| Qwen3.6 **35B-A3B MoE** MQ4 | ~16–17 GB+ (base + experts) | often no | run-the-unfit: offload enough just to fit (v2, MoE/EP out of v1 scope) |

---

## 2. What partial offload is NOT (scope guardrails)

- It is **not** `--n-gpu-layers` speed tuning. On a discrete card, spilled layers compute over
  PCIe (~12 GB/s read) vs HBM (~500+ GB/s). Expect ~10–50× slower decode on offloaded layers.
- It is **not** a new quant format or kernel micro-opt. It reuses existing kernels via staging
  or host-read variants.
- It is **not** automatic best-effort eviction during run. Placement is decided once at load by
  a policy; it does not thrash per-request (that would defeat the KV cache purpose).

---

## 3. Architecture facts that must be preserved

Grounded in the current tree:

- **Weights are device-local `GpuTensor`s** held in a model-owned table, referenced by
  `WeightSlot(pub u32)`. There is **no host/device memory-class tag** on tensors today.
- The per-token executor (`run_layer_program`, `hipfire-dispatch/.../superop.rs`) resolves each
  `OpBinding` **by index**, re-borrowing live `GpuTensor`s from the weight/scratch/state tables
  and calling the resolved kernel family method. `OpBinding` is pure POD (indices + `KernelKey`
  + flavor); it carries no pointers.
- Kernels are arch-gated `.gfxNNNN` files / `#if defined(__gfxNNNN__)` atoms
  (`arch_caps.rs`). The fast path uses WMMA GEMM (`gemm_hfq4g256_residual_wmma`, gfx11/12
  wave32) and 32-thread GEMV (`gemv_mq3g256`, `__launch_bounds__(32,16)`). Both assume device-
  local operands.
- **hipGraph capture** (`HIPFIRE_VERIFY_GRAPH=1`, default ON) captures `forward_scratch_layers`
  as one graph with fixed kernarg pointers bound to the live tensors. If a layer's weights move
  host↔device, those captured pointers dangle past `end_graph_capture` — the documented silent-
  corruption class (AGENTS.md §6).
- **KV cache** lives in a `SlotPool` arena + `kv_slots::preflight_alloc`, K/V formats asym4/3/2;
  written/read by `kv_cache_write_q8_0_batched_slots` and the flash-attention families with fixed
  per-row strides.
- The bridge (`hip-bridge/src/vmm.rs`) already exposes generic VMM primitives —
  `HipMemAllocationProp::device_pinned`, `mem_create`, `mem_get_allocation_granularity` — so
  allocating host-pinned memory is technically available at the bridge layer.

**Invariant to preserve:** output must be byte-identical to resident mode (per VALIDATION.md,
byte-exact token-ID comparison is stricter than prose coherence).

---

## 4. The core tension

Two hipfire invariants break when a layer leaves VRAM:

1. **Kernel memory assumption.** WMMA/GEMV kernels read weights from device pointers. Spilling
   means the kernel must either (a) have its weights copied into device scratch each time it runs,
   or (b) read them from host-mapped memory via a new slow-read kernel variant. Neither is the
   existing fast path.
2. **hipGraph pointer stability.** Captured graphs need fixed pointers for the process lifetime.
   Moving weights breaks this (§3).

Everything in §5–§8 exists to resolve these two without regressing the resident path.

---

## 5. Design options

### Option A — Staging arena + drop graph capture (recommended for first correctness)

- Allocate a fixed device scratch pool sized to the largest spilled layer, via the bridge VMM
  (`mem_create` with `device_pinned` staging → `hipMalloc` device scratch).
- Tag weight tensors (and KV slots) with residency: `Device` or `HostPinned`. A host-pinned
  tensor keeps its bytes in RAM; before each forward pass over that layer, memcpy its weights
  into the device scratch region; after, leave resident in scratch or evict.
- When **any** layer is spilled, disable hipGraph capture (`HIPFIRE_VERIFY_GRAPH=0` equivalent for
  this model) and run the `HIPFIRE_FORWARD_LOWERED` hand-loop / non-captured path. This avoids
  the dangling-pointer graph bug entirely — the cheapest correct choice.
- Pro: smallest kernel-surface change (reuse fast kernels via staging). Con: per-layer copy adds
  PCIe writes each step; no graph amortization.

### Option B — Mapped host memory, stable addresses, graph-compatible

- Allocate spilled weights as mapped host/GTT so their virtual addresses are process-stable →
  hipGraph capture survives.
- Kernels read them over PCIe with a new slow-read variant per GEMV/GEMM/attention family (no
  WMMA). Add in-graph async copy nodes only if some staging is still needed.
- Pro: graph capture works; fewer copies. Con: needs host-read kernel variants anyway; same PCIe
  ceiling; mapped-host reads on RDNA need the pointer to be visible/mapped to every wave — verify
  with a microbench (`bench_gtt_bandwidth.hip` pattern).

### Option C — Hybrid (final target, not first step)

- Resident layers use the fast device path + graph capture. Spilled FFN-heavy layers use mapped-
  host reads for weights but keep attention/KV fully resident to avoid cross-memory attention rows.
- Introduce only after A/B correctness is pinned and the bottleneck profiled.

**Recommendation:** start with **A** (fastest to correct), re-examine graph capture at §8 once
output is proven, then consider B/C for speed. Do not start with B — host-read kernel variants
are more surface area than staging and buy nothing until speed matters.

---

## 6. Work breakdown by layer

### 6.1 Tensor memory-class (linchpin) — qwen35 DENSE only
- `rdna-compute`: add a residency tag to `GpuTensor`/`DeviceBuffer` (e.g. `MemoryClass { Device,
  HostPinned }`) OR keep tensors device-only and model the split at the weight-table level. Prefer
  tagging at the table boundary so kernels still see only `GpuTensor`.
- **v1 = qwen35 dense weight tables only** (`weights.rs` `Qwen35Weights`, one residency field per
  dense layer). Propagate to `-llama` / `-gemma4` / `-lfm2moe` in §9 step 3. **Exclude the qwen35
  A3B-MoE / sealed-EP tables** — routed-expert staging is orthogonal to offload and ships later.
- `SlotPool` + KV arenas gain the same residency field, but only for the v1 dense path; KV spill is
  deferred (§6.5).
- Keep the resident code path byte-identical when no layer is spilled (zero-diff baseline). This is
  the regression guard: any offload change must not alter the non-spilled qwen3.8-27B dense trace.

#### Implementation decisions settled during bring-up (do not re-litigate)

- **`proj` needs a second reader seam; per-layer selection is impossible.** Quantized weights
  upload raw codes through the arch-supplied `read_proj` fn pointer, whose `&Gpu` cannot
  host-allocate: `Gpu::alloc_host_mapped_tensor` is `&mut self` because it records the host
  pointer in `self.host_mapped` (the registry `free_tensor` needs to `hipHostFree`). Selecting a
  different `read_proj` per layer does not help either — the
  backend is built per layer, but the parameter type is fixed across arches and cannot carry
  `&mut Gpu`. Ship a twin instead: `read_proj_host: Option<fn(&HfqFile, &mut Gpu, …)>`, which
  `proj` calls when `host_local` is set. `None` means the arch has no offload support, and an
  offloaded layer without a host reader is a **hard error**, never a silent device allocation —
  a half-configured offload must not masquerade as working. A `debug_assert!` is insufficient:
  it does not fire in release builds.
- **The f32-dequant fallback is fail-closed for offloaded layers.** `load_weight_tensor_raw`'s
  catch-all `_ =>` arm and qt 1 `Native` mode route to `dequant_weight_raw`, which has no host
  upload. Those arms refuse when the host path is requested, so f32-dequant quant types are
  unsupported for offload. Threading a `host` flag into `dequant_weight_raw` instead would
  change a signature shared with qwen2/llama/runtime for no gain on the dense target. The AWQ
  sidecar stays device-resident deliberately: it is a 1-D f16 vector of length K, so spilling
  it would cost PCIe bandwidth per GEMV to save kilobytes.
- **MoE is structurally out of reach, but the log line is the only guard.** `load_moe_ffn(hfq,
  gpu, …)` takes the raw `&HfqFile`/`&mut Gpu` and never sees an `HfqBackend`, so it cannot read
  `host_local` — that is what keeps A3B-MoE out of v1. It also allocates device-side
  unconditionally, so a *future* policy that offloaded a MoE layer would spill that layer's dense
  weights to host while its expert weights stayed in VRAM, and `offloaded=N` would still count
  the layer as offloaded. Unreachable in dense-only v1, but it is exactly the silent-partial
  shape the `read_proj_host: None` hard error exists to prevent: report counts from what was
  actually host-located, not from the policy that requested it.
- **The host memory class must be `hipHostMalloc(hipHostMallocMapped)`, not a host-located VMM
  arena.** This is the one decision that determines whether the feature does anything. A
  `hipMemCreate` PINNED/Host arena mapped into device VA — the primitive this branch was built
  on (`1f7588008`) — is charged against the **device heap 1:1** on gfx1201. Measured with a
  1 GiB `hipMalloc` ladder, holding 4096 MB:

  | held | device headroom | cost |
  |---|---|---|
  | nothing (control) | 15360 MB | — |
  | host-located VMM arena | 11264 MB | 4096 MB |
  | `hipHostMalloc(mapped)` | 15360 MB | 0 MB |

  So the VMM mechanism moved weight bytes to system RAM while charging the card for them
  anyway: `hipMemGetInfo` free did not rise, and the allocatable device headroom fell by exactly
  the spilled amount. Offload built on it was net-zero for the VRAM it exists to free, which is
  what the "does offload actually free VRAM?" blocker in the bring-up handoff was really seeing —
  it was not a measurement artifact. `hipHostMalloc` + `hipHostGetDevicePointer` gives the kernels
  the same dereferenceable pointer (device alias == host pointer on this box) at zero device cost.
  The VMM host primitive and `MemoryLocality::HostPinned` are therefore deleted, so the charging
  mechanism cannot be re-adopted; `Gpu::host_located` now means "allocated by
  `Gpu::alloc_host_mapped_tensor`". `examples/host_offload_headroom.rs` asserts the invariant,
  because byte-identity and coherence tests cannot see it — output stays correct either way and
  only the capacity win disappears.

### 6.2 Kernel paths for spilled weights
- Option A: implement staging in the executor — copy HostPinned weight bytes → device scratch
  before the op's kernel call; device tensors skip the copy. Reuse existing kernels unchanged.
- If profiling shows staging copies dominate, add Option B slow-read variants, arch-gated into
  `.gfxNNNN` files exactly like existing WMMA/GEMM splits — **do not** branch residency inside one
  kernel file (per `hipfire-arch-port`: family macros need concrete atoms).

#### 6.2.1 CPU execution of the spilled-weight ops (`memory.offload_exec=cpu`)

Shipped 2026-09-27. Placement is untouched — the same contiguous spilled prefix,
the same `hipHostMalloc` host-mapped weights, the same KV residency. What changes
is *who multiplies*, which is what bounds a spilled layer's per-token cost by the
PCIe link (27.1 GB/s measured, §7) instead of device DRAM: this is llama.cpp's
`-ngl` CPU backend, in hipfire's shape.

- **Config.** `memory.offload_exec` = `pcie` (default: byte-for-byte the
  behaviour above, and the zero-diff guard) | `cpu` (env
  `HIPFIRE_OFFLOAD_EXEC`). Unset, empty and unknown all fail closed to `pcie`.
  With `i_gpu_start == 0` it prints one informational line and changes nothing:
  no weight is host-mapped, so no step can move.
- **Seam.** One arch-agnostic place: `hipfire-dispatch`'s `execute_steps`. With
  CPU execution on, a fused entry that spans a CPU step is not matched (a fusion
  is one launch over several weights and cannot half-land on the CPU), and each
  `Step::Gemv` / `Step::GemvResidual` over a host-mapped weight runs on the CPU:
  D2H input → [AWQ divide] → FWHT → GEMV → H2D out, or in-place residual
  accumulate (the residual is the destination; a residual step never writes its
  `out` scratch). `weight_gemv_swiglu_residual` — the one op family that fuses a
  GEMV into a kernel the seam cannot see — splits itself: GPU `silu_mul_f32`,
  then the CPU GEMV plus residual.
- **Two launcher properties the CPU path must reproduce exactly**, both of which
  fail *silently* rather than erroring, and both now pinned by parity arms in
  `crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs`: the per-channel **AWQ**
  divide happens *inside the rotation* (`(W·s)·(x/s) = W·x`, 138 sidecars in the
  fixtures), and a **`Prerotated`** input must never be rotated again (the
  launcher checks the rotation tag; a CPU step cannot, since a bare `GpuTensor`
  carries no tag).
- **Graph capture.** A CPU step is a host sync point, so hipGraph capture is
  disabled for the model's lifetime whenever CPU steps are possible
  (`cpu_offload_active(i_gpu_start)`), logged once
  (`cpu exec: hipGraph capture disabled (CPU-executed steps present)`). The slot
  decode graph takes the same decision.
- **Footprint recipe for a large spilled prefix.** `hipHostMalloc` memory is
  pinned, so it is host RAM the kernel cannot reclaim: on this host a 16-layer
  spill of an 11.8 GB MQ3 checkpoint (≈2.5 GB pinned) stalled twice in load while
  the box had 0 GB free, and the same model at an 8-layer spill (≈1.2 GB) loaded
  and reported `8/8 spilled layers fully covered; uncovered quants: none`. Use
  the smallest spilled prefix that frees the VRAM you actually need.
- **Redline refused.** `memory.offload_exec=cpu` together with a retained-replay
  backend is a load error naming both keys: the tape records GPU launches, would
  omit the CPU-executed steps, and would replay stale activations (§6.7).
- **Coverage.** Every dense-decode projection of both layer types routes through
  the seam — DeltaNet `in_proj_qkv/z/a/b` and `out_proj`, FullAttn `q/k/v/o`,
  FFN `gate/up/down` — 7 distinct shapes on the 2B. Formats: **28**, i.e. every
  dense-capable entry in `docs/quant-formats/qt-register.txt` — qt 1/2/3/16
  element and Q8 blocks, qt 6/7/8/9/10/11/12 HFQ, qt 13/15/17/18/31 flat MQ,
  qt 19/20/30/51 Lloyd codebooks, qt 40/41 Bonsai ternary/binary, qt 44/45/47/48/49/50
  Magnum V2 + MQ4C. Each is pinned by
  `crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs` in **both** arms of the
  rotation contract — `Raw` (both sides rotate once) and `Prerotated` (neither
  rotates again) — against the production launcher, and additionally bit-exactly
  against the canonical `dequantize_to_f32` where that has an arm (19 formats).
  qt 49 (`qwen3.8-27b.mq3-xt`'s only projection format) has no canonical arm, so
  its oracle is a **real 27B tensor**: 2.4e-7 / 3.5e-7 relative at m=12288 k=5120
  (`HIPFIRE_PARITY_EXTRA_MODEL=<path> HIPFIRE_PARITY_EXTRA_QT=49`).
  Two formats decode but are arch-ineligible as `Step::Gemv` weights on gfx12 —
  qt 11/12 have no production dense GEMV kernel (`MissingImpl`), which the
  parity test reports rather than silently comparing nothing.
  The exclusions are structural: qt 5 `Q8HFQ` (rows padded to a 128-byte stride),
  qt 14 `MQ8G256` (`RotationPlan::Mq8Internal`: int8-quantized activations),
  qt 21/24/32-37 HFP4/MFP4(+Lloyd/P/E8) (per-row 16 B header + per-32 block
  scales/codewords, and an arch-owned upload path), qt 28/29 PARO (Givens
  rotation on the activation), qt 0/4 GGUF-side formats, and qt 38/39 GL +
  qt 22 `TidI32` (MoE-indexed only, no dense GEMV kernel exists / not a weight).
  Each stays on the GPU over PCIe and is *named* by the load-time line
  `cpu exec: {covered}/{spilled} spilled layers fully covered; uncovered quants: …`.
  `HIPFIRE_CPU_EXEC_TRACE=1` reports one line per distinct step shape (at the
  shape's first call and at every doubling of its call count, each with that
  shape's own mean D2H/GEMV/H2D) plus the running counters, whose second number
  ("host-mapped steps still on GPU") must be 0 for a fully covered model — a
  fused launch the guard missed or a call site that bypasses `execute_steps`
  shows up there instead of as a mystery. The `calls=` field on the line is what
  makes the number quotable: a shape's first line is its cold first step.
- **Not covered, deliberately.** The slots/serve body (`forward_batch_slots` →
  `dense_ffn_body_slots`) and prefill run batched GEMM kernels that never enter
  `execute_steps`, so their spilled weights are still read over PCIe; the seam
  covers their per-slot lm_head `Step::Gemv` only. The lm_head itself can never
  be host-mapped — it is part of `largest_fitting_tail`'s always-resident base.
- **Numerical contract.** llama.cpp-level, not bit-identity (the same record's
  "Correctness gate"): the two engines are independent implementations with
  different accumulation orders. Acceptance is coherence plus task-correct output
  plus a *measured* divergence — so the CPU kernels use ordinary f32 `expf`
  arithmetic where they need it and hand-written AVX2 row dots (`Mq4G256`,
  `Mq3G256V2`) that are intentionally not bit-equal to the scalar path
  (`crates/hipfire-cpu/src/simd`). Everything else decodes scalar; the seam's
  cost model is weight bytes over host bandwidth, so a format without a kernel
  is a throughput regression, never a correctness one.
  **Fidelity is the per-step parity (6.7e-7 worst case) and the bit-exact decode
  over 1695 tensors; the end-to-end text divergence is a downstream symptom of
  greedy argmax over numbers that differ at the seventh digit, not an error rate
  — read the shared-prefix length as a sensitivity, never as accuracy.**
- **Evidence** (gfx1201, HIP 7.2, 2026-09-27): `hipfire_cpu::dequant_group`
  reproduces the canonical decoder bit-for-bit over 1695 real tensors of the
  2B/9B fixtures; launcher-vs-CPU parity worst case 6.7e-7 relative across 8
  formats plus AWQ and pre-rotated arms; greedy output byte-identical to the
  parent-commit build for `pcie` with and without a spill (2B 1343 chars, 9B
  2734 chars). The `cpu` arm is *deterministic* (byte-identical across two fresh
  processes and across `RAYON_NUM_THREADS=1`, so the divergence is not CPU-side
  scheduling) and its first divergence from `pcie` on the 9B at 8 spilled layers
  is a **single whitespace token** after 723 characters (≈190 tokens) — a
  near-tie under greedy decode, after which the histories diverge and so do the
  completions. Numbers, method, fixture identity, ceilings and the superseded
  first readings:
  [`docs/perf-checkpoints/2026-09-27-gfx1201-cpu-exec-offload.md`](../perf-checkpoints/2026-09-27-gfx1201-cpu-exec-offload.md).

### 6.3 Dispatch substrate (`hipfire-dispatch/.../superop.rs`)
- The executor binds by index today; add a residency-aware bind step: for each `WeightSlot`, pick
  the device pointer (resident) or staging scratch pointer (spilled). `OpBinding` stays POD; the
  bind resolves against live tensor residency.

### 6.4 Graph capture (§3 risk #2)
- **Implemented:** capture stays ON for the default direct-PCIe spill (the weights' addresses are
  stable host-mapped pointers, so a captured graph replays correctly — verified byte-identical to a
  fully resident run). Only `memory.offload_exec=cpu` forces the non-captured path, for the model's
  lifetime, because a CPU step is a host sync point that can neither be recorded nor replayed
  (§6.2.1; `cpu_offload_active`, logged once).
- Later: if Option B, verify mapped-host pointer stability and re-enable capture with in-graph
  copy nodes; add a regression guard that fails loudly if a captured graph's pointers ever move.

### 6.5 KV cache split (defer / restrict)
- Attention kernels read K/V from device with fixed strides; cross-device rows need a second
  flash-attention path or per-step staging of active rows. **Initial scope:** spill only FFN-layer
  weights, keep attention + KV fully resident. Expand after correctness is proven.

### 6.6 Placement policy + admission (`hipfire-config`)

**Objective differs by use case but shares one mechanism (contiguous suffix on GPU):**

- **Long context:** given a target KV budget / max-seq, offload the *minimum* contiguous prefix of
  layers whose freed VRAM makes the KV cache fit. This is llama.cpp's `-ngl` in reverse — instead
  of "how many layers can I keep on GPU", it is "what must I drop to afford this context".
- **Run-the-unfit:** offload enough layers that total footprint drops below device memory, then fit
  the largest KV cache that remains.

**Placement heuristic.** Match llama.cpp (`llama-model.cpp:1467`): keep a contiguous suffix
`[i_gpu_start .. n_layer]` on GPU, spill the prefix to host RAM. Every layer reads its weights each
decode step (the residual stream passes through all of them), so no layer is bandwidth-free — but
a contiguous split keeps the hot path simple and matches the reference exactly. The input/embed and
output/lm_head layers stay resident regardless (tiny, used every step).

**Optimizer.** Beyond a raw knob, project per-layer VRAM and pick the placement that maximizes KV
capacity for the requested context (or minimizes offload for a target fit) — this is what
llama.cpp's `fit.cpp` does with its `set_ngl_tensor_split_tbo`. hipfire can reuse the OOM-guard
shape (`oom_guard_effective`, deployment-class logic + logging) as the template.

**Config knob.** e.g. `memory.gpu_layer_budget` (exact layer count, or a fraction), plus a target
max-seq so placement is driven by KV need rather than an opaque layer number; daemon logs
"N layers offloaded, M resident, KV budget X".

### 6.7 Redline path (separate effort)
- `hipfire-redline` bare-libdrm/direct-KMD bypasses the ROCm runtime; host/device memory mgmt
  there is not covered by the `hip-bridge` VMM primitives. Track as a follow-up; do not block v1.

### 6.8 Modularization — shared core vs per-arch adapter (start one, broaden others)

The change decomposes into a small arch-agnostic **core** and a thin **per-arch adapter**. That is
why expanding from qwen3.8-27B to other arches is ~one-crate edits, not re-research:

**Core — implement once, reused by every arch (all arch-agnostic):**
- Residency data model (`MemoryClass {Device, HostPinned}`) at the weight-table boundary — kernels
  still see only `GpuTensor`, so no kernel change.
- Staging arena (device scratch for spilled bytes), allocated via bridge VMM (`mem_create`/`device_pinned`);
  Option A reuses existing kernels unchanged (§6.2).
- Residency-aware bind in the shared executor loop (`superop.rs::run_layer_program`) — `OpBinding` stays POD,
  one bind step resolves device-vs-staging per `WeightSlot`; all arches run the same loop.
- Graph-capture gate (one decision: any HostPinned → non-captured hand path).
- Placement policy + `hipfire-config` knob (contiguous-suffix heuristic; each arch reports per-layer VRAM).

**Adapter — one per arch crate, mechanical for dense arches:**
- Add a residency field per layer to that crate's weight table (`Qwen35Weights`, etc.) and wire the bind to
  resolve it. Dense arches (llama, gemma4, lfm2moe) reuse the same `Proj/ResidualGemv/Attend` ops (§6.3);
  expansion ≈ one crate edit + shared-core reuse, no new kernel surface for v1.

**Why the qwen3.8 spike validates both at once:** §9 step 1's go/no-go proves the core *and* the adapter
shape simultaneously — the hard part is not done twice.

**Where modularity holds strongly (v1):** FFN-only spill (§6.5) keeps attention + KV resident, so v1 touches
only shared core + one dense adapter — no kernel changes, no MoE composition, no cross-device attention.

**Where modularity degrades — deferred precisely to keep v1 clean:**
- **Option B host-read kernel variants (speed pass).** If profiling forces PCIe reads over staging, per-GEMV/
  GEMM/attention `.gfxNNNN` atoms appear, arch-gated via `arch_caps.rs`. Real per-arch surface, but later
  (§9 step 5) and on the `hipfire-arch-port` family-macro path.
- **MoE / routed-expert composition (deepseek4, qwen35-A3B).** Routed experts already stage host tensors
  (`forward.rs`, §10 Q5); offload must compose with it or wait — v2. This is the genuine cross-arch friction
  point.
- **Cross-device attention rows (§6.5 KV spill).** Needs a second flash-attention path (arch-specific); deferred
  until FFN-only spill is proven and profiled.

**Expansion cost per next arch:** dense arch ≈ one weight-table edit against the shared core; MoE/deepseek4 ≈
dense edit + routed-expert composition (v2).

---

## 7. Performance reality check (do not over-promise)

- Offloaded layers are PCIe-bandwidth-bound, not HBM-bound. For a 27B/35B model this is the
  dominant cost; expect large decode regressions on spilled layers. **Both use cases pay the same
  cost** — long-context and run-the-unfit offload weights read every step, so tok/s drops by the
  same PCIe-bound factor regardless of which scenario motivated the spill.
- This is a "can run" capability. The acceptance criterion is correctness + fit, not tok/s parity
  (the user explicitly accepts llama.cpp's `-ngl` slowdown as the price for capacity).
- **CPU execution changes the bound, not the shape.** `memory.offload_exec=cpu` replaces "every
  spilled byte crosses PCIe once per token" with "every spilled byte is read from system DRAM and
  multiplied on the CPU", so the ceiling becomes the host's memory bandwidth plus one D2H/H2D pair
  per step (a few tens of microseconds each) rather than 27.1 GB/s over the link. It buys nothing
  when the spill is empty and nothing for the batched (slots/prefill) bodies, which bypass the seam.
- **Graph-capture loss compounds the penalty.** Option A drops hipGraph capture when any layer is
  spilled — a real host-overhead hit on top of the PCIe cost. Even at an "acceptable" speed target,
  weighing A (simplest, correct) against B (mapped-host stable pointers, keeps graph) still matters;
  profile before committing.
- **Measurement method, and the resident VRAM control.**
  `hipfire bench` already pins greedy decoding itself (`temperature: 0.0, top_p: 1.0` in the
  generated request, hipfire-cli/src/main.rs:4870), so a bench A/B needs no flag — the `-t 0`
  requirement applies to `hipfire run`, which samples at the configured temperature by default.
  When comparing decode throughput, **interleave the arms** (branch, stock, branch, stock, …)
  across fresh processes: batching same-arm runs consecutively on this box lets them drift into a
  ~2x lower regime, which reads as a fake regression. Interleaved 6-per-arm, branch and stock
  medians were 40.25 vs 40.35 tok/s — 0.25% apart, no regression.
  **Caveat: do not eyeball bench output.** Bench also sets `max_think_tokens: 1` and
  `assistant_prefix: "closed_think"` when reasoning is off (main.rs:4876-4878), so on a
  reasoning checkpoint every bench run decodes through the empty-think template whose degraded
  output is visible in the text. Bench is for rate/metadata only; the coherence eyeball must use
  `hipfire run`.
  **Resident VRAM control** — read it from the bench JSON, not by polling `rocm-smi`:
  `vram_free_before_mb - vram_free_mb` is the model's own footprint at load time. For
  `qwen3.8-27b.mq3-xt` on gfx1201 (16304 MB total) that is **13042 MB** on this branch and
  **13098 MB** on stock, across three loads — call it **~13070 MB**. An offloaded run is only
  proven if `vram_free_mb` lands meaningfully above ~3100 MB; an `offloaded=N` log line is not
  evidence on its own. Branch ≈ stock on this number also means the resident path costs no extra
  VRAM.
- On unified-memory APUs the feature is mostly redundant — route those through the existing OOM
  guard instead.

---

## 8. Validation route (per CLAUDE.md / VALIDATION.md)

No single correctness gate applies; select by what changed:

- **Kernel/dispatch/weight-table changes** → `scripts/redline_daemon_harness.py` if on the
  Redline path; otherwise byte-exact token-ID comparison against resident mode.
- **User-facing generation / state lifecycle** → `scripts/serve_harness.py battery` (varied
  prompts) + `chain` (related turns), decoded and eyeballed. Spilled output MUST match resident
  output token-for-token.
- **Per-GPU baselines:** use `tests/quality-baselines/{gfx1010,gfx1100}/` auto-detection; record
  model md5 + binary md5 + prompt md5 (byte-identical prompts, AGENTS.md §0 rule 2).
- **Eyeball check** is mandatory: a suspiciously tight stddev on the spilled path can hide single-
  token attractor failures (AGENTS.md §0 rule 3).
- **Byte-identity requires greedy decoding (`-t 0`).** Measured on gfx1201: at default
  temperature this engine is **nondeterministic** — the same binary, same prompt, same
  settings produces different output across runs, on both the feature branch and stock.
  A token-ID comparison at default temperature therefore fails even with zero regression,
  and a *matching* digest is luck rather than evidence. Verified: qwen3.5-2b and
  qwen3.5-9b are each self-deterministic under `-t 0` and byte-identical to stock
  `master`. AGENTS.md §0 governs byte-identical *prompts*; it says nothing about decoding
  determinism, which is the gap this closes.
- **Target fixture:** `~/.hipfire/models/qwen3.6-27b.mq4` — 14984158208 B, sha256
  `86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc`, an exact match to
  the AGENTS.md canonical qwen3.6-27b pin, so its numbers are quotable. Do **not** use
  `qwen3.8-27b.mq4-xt`: that name is not on disk. `qwen3.8-27b-mq3-xt`
  (11777616896 B) is the **MQ3** tier, not the MQ4XT pin AGENTS.md cites (14980361216 B),
  so its numbers are not comparable to that fixture — and being ~3 GiB smaller it leaves
  enough headroom on a 17.1 GB card that the layer budget may never engage, making it a
  weak subject for demonstrating VRAM relief.

---

## 9. Phased sequencing

1. **Spike — residency tag + qwen3.8-27B DENSE.** Add `MemoryClass` to the qwen35 *dense* weight
   table only (`weights.rs` `Qwen35Weights`; exclude A3B-MoE / sealed-EP). Implement manual "spill
   last N dense layers" via Option A staging; force non-captured path. Prove byte-identical output vs
   resident on gfx1100 (or available box) for the qwen3.8-27B MQ4XT dense checkpoint. *This is the
   go/no-go.* Do not touch MoE/EP in this spike — if MoE correctness depends on it, that is a
   separate follow-up.
2. **Placement policy + config.** Generalize from manual to a placement policy feeding `hipfire
   config`; daemon logging; preflight that refuses rather than OOMs when nothing fits. v1 knob is
   qwen3.8-27B KV-budget / weight-fit driven ("N layers offloaded, M resident, KV budget X").
3. **Broaden arches.** Propagate the proven dense pattern to llama, gemma4, lfm2moe — each a fresh
   `hipfire-arch-*` crate, mechanical because the dispatch substrate is shared/POD (§6.3). qwen35 A3B-MoE
   and deepseek4 routed-expert staging separately: both already carry complex host-staging semantics
   (`weights.rs`, `forward.rs`) and must prove MoE correctness first (§10 Q5).
4. **KV spill expansion** (§6.5) only after FFN-only spill is proven and profiled on qwen3.8-27B dense.
5. **Speed pass.** Re-examine graph capture; add Option B slow-read variants if profiling warrants.

---

## 10. Open questions for the reviewer

1. **Memory-class placement:** tag `GpuTensor` directly, or keep tensors device-only and model
   residency at the weight-table boundary? (Affects how invasive kernel changes are.)
2. **Graph gate (§4):** is forcing the non-captured hand path on spill sufficient for v1, or must
   graph capture survive from day one? Cost of each.
3. **Staging vs host-read:** for Option A, does per-layer memcpy every decode step dominate, or
   is it acceptable until speed matters? Expected copy cost for a 27B layer over PCIe.
4. **Granularity:** per-layer spill, or coarser (e.g. all FFN vs all attention) to limit the
   kernel surface?
5. **deepseek4 interaction:** its MoE routed experts already stage host tensors (`forward.rs`).
   Does offload compose with routed-expert staging, or must it wait until after MoE correctness?
6. **Redline:** should v1 target the HIP-runtime path only, or both? (Recommend HIP path first.)

---

## 11. Acceptance criteria (final)

Status against the evidence of record (`docs/perf-checkpoints/2026-09-27-*`). Ticked items
name the fixture that evidences them; unticked items have no run of record.

- [ ] qwen3.8-27B MQ4XT DENSE runs to completion via the spill path (use case 1 long-context, and
      use case 2 run-the-unfit on an over-capacity dense model); output byte-identical to resident
      mode on ≥3 fresh-process runs.
      **Not evidenced on MQ4XT.** The 27B run of record is `qwen3.8-27b.mq3-xt` (64 layers) at an
      8-layer spill: one 610 B completion, byte-identical to the `pcie` arm, single prompt, no
      ≥3-process byte diff. Cross-process byte-identity and the byte-diff guard were read on the 9B
      (`qwen3.5-9b.mq4`, 8 of 32 spilled). Under greedy decode the two arms agree for 723 characters
      (~190 tokens) and then differ by one whitespace token — the documented contract is coherence
      plus a *measured* divergence, not bit-identity.
- [x] Config knob controls placement; daemon logs residency decision — `memory.gpu_layer_budget`
      resolves `i_gpu_start` and logs `partial offload: N resident / M offloaded`.
- [x] Resident path is byte-identical when no layer is spilled (zero-diff baseline) — read on
      `qwen3.5-2b` and `qwen3.5-9b` (1343 / 2734 chars) against the parent build. The unset-budget
      branch of the same path is what the 27B fixture runs, but no 27B byte-diff was read.
- [x] Validated with `serve_harness.py battery` + chain, decoded and eyeballed; md5s recorded —
      `cpu` arm with a spill, 5/5 turns each, `runaway=0 empty=0 attractor=0 retrieval_miss=0`.
- [ ] MoE/EP tables untouched by v1 — the qwen3.8-27B dense trace matches resident bit-for-bit, proving
      no regression was introduced into the A3B path (§9 step 1 gate).
      **Half evidenced.** MoE is structurally out of reach (the MoE loader never sees the offload flag)
      and the other arches declare `host_local: false`; no A3B sparse trace of record was compared
      against resident.
