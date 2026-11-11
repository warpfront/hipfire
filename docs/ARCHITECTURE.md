# Architecture

How a `hipfire run` becomes tokens on the current modular tree. Runtime source
wins on conflict. Mutable command surfaces, model catalogs, and env inventories
live in their own owners ([`CLI.md`](CLI.md), [`MODELS.md`](MODELS.md),
[`env-vars.md`](env-vars.md)); this page maps crates, loaders, and data flow.

Truth-state labels follow [`INDEX.md`](INDEX.md). Implementation capability is
not product admission: a compiled path or runtime gate is not certification.

## Workspace map

Workspace members (`Cargo.toml`), grouped by role. Directory order is not
layering; the layer table below is. The list tracks `Cargo.toml` and is
deliberately uncounted: count members from the manifest, not this page.

```text
crates/
├── hip-bridge/               # dlopen HIP FFI (libamdhip64); VMM arenas and retired-VA accounting
├── hsa-bridge/               # thin public HSA queue/AQL helpers
├── va-bridge/                # dlopen VA-API (VCN JPEG) decode for VL images
├── rdna-compute/             # Gpu, kernel compile/cache, HIP launch families, replay controller
├── radiowave/                # policy-driven HIP compiler + AMDGPU code-object inspector
├── peacemaker-ir/            # typed RDNA machine-code representation, gfx12 instruction tables
├── peacemaker-lift/          # ELF/bundle/metadata lifter for typed AMDGPU programs
├── peacemaker-cdna/          # CDNA3 (gfx942, wave64) instruction model, encoder/decoder, CPU emulator; transitive via peacemaker-isa, never selected at runtime
├── peacemaker-author/        # typed kernel-authoring core (targets, LDS regions, wait tokens)
├── peacemaker-isa/           # checked instruction Builder (wait ledger, LDS, hazards) + peacemaker tooling
├── peacemaker-rip/           # standalone `.rip` kernel-script front end onto the peacemaker-isa builder
├── railgun/                  # certified dispatch-program model, gfx12 plan/lowering, shadow diff; feature `npu`: XDNA2 runtime
├── railgun-cert/             # offline certifier: kernarg facts as a receipt `railgun` section
├── railgun-corpus/           # runtime-side reader for railgun fact records / JIT receipt corpus
├── railgun-jitcorpus/        # offline JIT receipt corpus builder and coverage check
├── saddle-core/              # the substrate: grammar, KV, caps, sampling policy
├── saddle-quant/             # quantization-quality toolkit (KLD references, Hessians, imatrices, scores)
├── saddle-lab/               # archived research examples and one-off harnesses
├── hipfire-runtime/          # arch-agnostic infra + Architecture trait; LLaMA forward still lives here
├── hipfire-cpu/              # CPU execution core for host-mapped (spilled) quantized weights
├── hipfire-dispatch/         # dtype/arch family tables + MoE/attn pipelines
├── hipfire-dispatch-tests/   # dispatch coverage tests
├── hipfire-arch-llama/       # facade for dense LLaMA/Mistral/plain Qwen3 (ids 0/1)
├── hipfire-arch-qwen2/       # Qwen2 text (id 7)
├── hipfire-arch-qwen35/      # Qwen3.5/3.6/3.8 hybrid DeltaNet, dense and MoE (ids 5/6)
├── hipfire-arch-qwen35-vl/   # vision tower attached to Qwen3.5 ids 5/6
├── hipfire-arch-qwen4/       # Qwen3.8-Flash-Next / Qwen4 (id 16)
├── hipfire-arch-dots-ocr/    # dots.ocr (id 8)
├── hipfire-arch-deepseek4/   # DeepSeek V4 Flash (id 9)
├── hipfire-arch-minimax/     # MiniMax-M2 (id 10)
├── hipfire-arch-lfm2moe/     # LFM2.5 / LFM2.5-MoE (id 11)
├── hipfire-arch-lfm2-vl/     # LFM2-VL SigLIP2-NaFlex tower + projector (rides id 11)
├── hipfire-arch-cohere2moe/  # Cohere2-MoE (id 12)
├── hipfire-arch-gemma4/      # Gemma 4 text (id 13)
├── hipfire-arch-muse-glimmer/# Muse Glimmer dense text (id 14)
├── hipfire-arch-maple/       # Maple-Preview natively-ternary MoE (id 15)
├── hipfire-arch-diffusion/   # FLUX.1 / FLUX.2 image-generation components (ids 40/45)
├── hipfire-arch-toy/         # template only (0xFF); never dispatched
├── hipfire-ds4-parent/       # DeepSeek V4 parent-checkpoint calibration oracle
├── hipfire-pflash/           # PFlash prefill compression — deprecated since 0.4.0, removal in 0.5.0
├── hipfire-loader/           # Carrier registry, load_model, LoadedModel, batch staging
├── hipfire-engine/           # arch-free serve engine: scheduler, terminal, emit, prompt
├── hipfire-generate/         # generation bodies: ar, qwen, dense, vision, batch, img, redline fixtures
├── hipfire-daemon/           # the product binary — [[bin]] name = "daemon"
├── hipfire-quantize/         # CPU encoder (safetensors/GGUF → .mq* / .hfq*)
├── hipfire-detect/           # observational JSONL behavior detectors
├── hipfire-atlas/            # Kernel Atlas schema + emit helpers
├── hipfire-config/           # typed TOML config + migration
├── hipfire-registry/         # strict bundled/dynamic model registry
├── hipfire-client/           # daemon JSONL + OpenAI HTTP/SSE client
├── hipfire-cli/              # native operator and HTTP service
├── hipfire-tui/              # terminal UI
├── hipfire-reap/             # model-agnostic REAP expert pruning + selective re-quant overlay
├── redline/                  # experimental direct-KMD / bare-libdrm research
├── redline-dispatch/         # retained-tape record/replay + plan selection
├── redline-rocr/             # public ROCr/HSA ABI + PM4 packet builders
├── hipfire-xdna/             # minimal raw amdxdna ioctl control plane (opt-in, exact gfx1151)
├── peacemaker-npu/           # XDNA2/AIE2P program builder, kernel designs, CPU simulator
└── npu-tools/                # XDNA2 NPU diagnostics and measurement binaries (railgun feature npu)
```

HIP sources live under `kernels/src/`; there is no JavaScript/TypeScript
operator runtime.

### Layering

| Layer | Crates | Role |
|---|---|---|
| Operator | `hipfire-cli`, `hipfire-config`, `hipfire-registry`, `hipfire-client`, `hipfire-tui` | Tag resolve, typed config, pull, HTTP service/client, one-shot daemon spawn |
| Product binary | `hipfire-daemon` | `[[bin]] name = "daemon"`. Message dispatch and process lifetime only |
| Generation | `hipfire-generate` | The generate bodies: `ar` (route selector + dispatch), `qwen`, `dense`, `vision`, `batch`/`fn_batch`/`vmm_conv`, `img`, plus the Redline fixtures |
| Serve engine | `hipfire-engine` | Scheduler, terminal control, emit, prompt. **Zero arch dependencies** |
| Composition root | `hipfire-loader` | Carrier registry, single `load_model` dispatch, `LoadedModel`, continuous-batch staging |
| Arch forward | `hipfire-arch-*` | Config / weights / state / static-dispatch forward (LLaMA exception: canonical forward remains in runtime) |
| Shared infra | `hipfire-runtime` | HFQ/safetensors, tokenizer, sampler, framing, spec primitives, `Architecture` trait |
| Substrate | `saddle-core` | Grammar, KV cache, `ArchCaps`, sampling policy. Depends only on `rdna-compute`, `hip-bridge`, `serde` |
| Kernel select | `hipfire-dispatch`, `rdna-compute` | Family tables + `Gpu` methods that pick WMMA/dot/baseline |
| Device FFI | `hip-bridge`, `hsa-bridge` | HIP runtime; optional HSA AQL helpers |
| Encode | `hipfire-quantize` | No GPU deps; CI-safe quantizer |
| Observability | `hipfire-detect`, `hipfire-atlas` | Detectors; bench corpus schema |
| Kernel compile policy | `radiowave` | Policy-driven hipcc invocation and AMDGPU code-object inspection behind `rdna-compute`'s compile path |
| Retained replay | `redline-dispatch`, `redline-rocr`, `rdna-compute::replay` | Product-integrated Redline **transport**: record, retain, replay PM4/AQL (see below) |
| Typed authoring and validation | `peacemaker-ir`, `peacemaker-lift`, `peacemaker-cdna`, `peacemaker-author`, `peacemaker-isa`, `peacemaker-rip`, `railgun`, `railgun-cert`, `railgun-corpus`, `railgun-jitcorpus` | Machine-code types, lifter, checked builder, certified facts and dispatch-program model. `peacemaker-cdna` (CDNA3/gfx942 instruction model and CPU emulator) is reached only through `peacemaker-isa`'s opt-in `cdna` arm. It is a transitive compile-time dependency of `rdna-compute` (which depends on `peacemaker-isa`), but no runtime dispatch, callsite selection or default behavior uses it. Authoring and validation, **not** the retained transport (see railgun below) |
| Direct-KMD research | `redline`, `saddle-lab` | Not the serving transport |
| Deprecated | `hipfire-pflash` | PFlash prefill compression: deprecated since 0.4.0, removal in 0.5.0 |
| NPU (XDNA2/AIE2P) | `peacemaker-npu`, `railgun` feature `npu`, `hipfire-arch-qwen4` `npu_trunk`, `npu-tools`, `hipfire-xdna` | `hipfire-arch-qwen4` links `peacemaker-npu` and `railgun`'s `npu` feature on Linux; the Flash-Next trunk route defaults to `auto` and arms only on gfx1151 with an openable NPU and matching calibration (see [Qwen4 NPU trunk](#qwen4-npu-trunk-xdna2)). `npu-tools` and `hipfire-xdna` are diagnostic/experimental ([`npu/README.md`](npu/README.md)) |

The dependency edge runs one way and is checked by `cargo tree`:

```text
saddle-core → hipfire-runtime → hipfire-loader → hipfire-engine
                                      → hipfire-generate → hipfire-daemon
```

Two invariants hold today and are worth keeping:

- **`saddle-core` never depends upward.** Its only dependencies are
  `rdna-compute`, `hip-bridge` and `serde_json`; `cargo tree -p saddle-core -e normal`
  matches nothing under `hipfire-{runtime,loader,engine,generate,daemon,arch-*}`.
- **`hipfire-engine` declares no arch crate at all.** The daemon reaches an
  architecture only through `hipfire-loader`'s `Carrier` and the
  `hipfire-generate` entry points; `crates/hipfire-daemon/src/main.rs` names no
  `hipfire_arch_*` type (the daemon manifest lists only `hipfire-arch-qwen35`
  and `hipfire-arch-qwen35-vl`, for slot support).

`hipfire-runtime` lists the arch crates under `[dev-dependencies]` for its own
examples. That is not a cycle — dev-dependencies do not participate in the
build graph of anything that depends on it.

`hipfire-loader` depends on arch crates and is the **only** top-of-DAG
`load_model` entry the daemon uses. Forward stays **statically dispatched** on
concrete arch types after load (`Architecture` docs in
`crates/hipfire-runtime/src/arch.rs`): the trait is bring-up scaffolding, not a
hot-path `dyn` forward. **LLaMA exception:** `hipfire-arch-llama` is a facade —
canonical dense LLaMA/Mistral/plain-Qwen3 forward and shared transformer types
still live in `hipfire_runtime::llama`; the arch crate re-exports them.

### Qwen4 typed layer seam

The current Qwen4 ordinary-HIP path keeps family binding in
`crates/hipfire-arch-qwen4/src/program.rs` and `gpu_forward.rs`: typed
dimensions/descriptors, resident-resource binding, bounded scratch, row-shaped
router views, and whole-program preflight live there. Neutral stateful
operation contracts and execution remain in
`crates/hipfire-dispatch/src/pipeline/layer_ops.rs` and `steps.rs`; the sealed
QT44/QT53 MoE route remains shared in `pipeline/sealed_moe.rs`,
`moe_program.rs`, and `qt44_qt53_prefill.rs`. Fixed Qwen4 HIP helpers are
registered by `hipfire-arch-qwen4/src/gpu_ops.rs`, while
`rdna-compute/src/tensor_ops.rs` and `grouped_ops.rs` provide neutral wrapper
contracts. This is an implementation ownership map, not replay/PM4 admission,
physical-EP proof, or a performance promotion.

With `deltanet` enabled, the fixed-geometry QT44/QT53 route is admitted on
any AMD GPU for matching wire formats and operands; gfx1151-specific kernel
optimizations do not gate ordinary-HIP serving.

The grouped symmetric IU4 MoE prefill route (`HIPFIRE_QWEN4_MOE_SYM_IU4`) is
**on by default** on gfx1100, gfx1151 and gfx1201 for layers whose routed-expert
headers verify symmetric at load (a symmetric requant such as GPTQ3); shipped
asymmetric artifacts fail that check and stay on the F16 WMMA route.
`HIPFIRE_QWEN4_MOE_SYM_IU4=0` opts out and keeps the F16 route, which has the
lower KLD but the slower prefill; the measured trade for each architecture is
recorded with the knob in [`env-vars.md`](env-vars.md). On gfx1100 the regular
prefill and the S1 slow-link window route (`aggregate-sym-iu4`) read the same
request (`Gpu::qwen4_moe_sym_iu4_requested`), so one flag moves both.

On gfx1151, Flash-Next prefill tiles of 48 to 511 rows (agent-turn appends,
tool results, short prompt tails) run on the >= 512-row kernel family by
default. The tile engages `Gpu::prefill_invariant_scope`, so every row-count
routing predicate sees the canonical chunk while launches keep the real row
count. `Gpu::qwen4_append_f16_moe_scope` keeps that tile's MoE on the F16 WMMA
grouped arm, because the IU4 arm below 512 rows exceeds the KLD cap.
`HIPFIRE_QWEN4_APPEND_WMMA=0` restores the F32 SIMT routes.

### Qwen4 expert-parallel mesh (Flash-Next, EP2 through EP8)

`topology.ep` (`HIPFIRE_TOPOLOGY_EP`) takes an integer `1..=8`;
`HIPFIRE_QWEN4_EP=0` forces `1`. Qwen4 admits expert parallelism only with
`pp = 1`, `tp = 1`, on `ep` ranks of one exact arch, gfx1100 or gfx1201 (a
mixed set is refused), with the peers directly accessible from each
other (`qwen4_ep_geometry_refusal`, `qwen4_ep_preflight` in
`crates/hipfire-loader/src/admission.rs`; checked before rank 0 allocates).
Routed experts shard by id (`e % N`; an N that does not divide 512 gives
uneven per-rank counts, `Qwen4EpPlacement::local_count`); every rank runs the
unchanged single-device program. Under the developer-only
`HIPFIRE_EMULATE_GPUS` the ranks alias physical devices (rank `r` on device
`r % count`, `qwen4_ep_rank_devices`).

The mesh is owned by the rank-0 `Qwen4Bundle` (`attach_ep_mesh`,
`crates/hipfire-arch-qwen4/src/ep.rs`): peers are bundles on persistent worker
threads, each with its own `Gpu`. It is **not** `LoadedModel.ep`, so a Qwen4 EP
model still takes the ordinary `Qwen4Ar` / `Qwen4Spec` generation route rather
than `generate_ep`. EP refuses what the replicated program does not run
(DFlash, DSpark, n-gram drafting, DDTree, adaptive KV, an EAGLE drafter, CASK,
PFlash, an experts-per-token override, a non-Single compute placement; lane
steps never run under EP), and native MTP attaches under EP only on an explicit
`--spec mtp` (`qwen4_ep_option_refusal`, `qwen4_native_mtp`).

Three EP levers are **default-on under EP**, each with an exact-`0` kill switch
(`gpu_forward.rs`):

- `HIPFIRE_EP_BALANCED_OWNERS`: balanced per-call expert ownership on the
  host-mapped trunk MoE layers. It ships for exact gfx1201 at 2..=4 ranks only;
  a gfx1100 rank logs `balanced ownership not admitted (needs gfx1201); static
  split`, a gfx1201 rank at 5..=8 ranks `... (needs at most 4 ranks); static
  split`, and both run the exact static split.
- `HIPFIRE_EP_OWNER_ROUTED_PREFILL`: owner-routed chunk MoE, EP2 through EP8.
  A prefill chunk of at least 512 rows computes only its owned expert tiles and
  pulls the other ranks' down rows over P2P, instead of computing every expert
  on every rank. It needs the Flash-Next profile and one of two routes: exact
  gfx1201 on the symmetric IU4 route (every MoE layer verified symmetric on
  every rank), or exact gfx1100 on the F16 WMMA grouped route (BF16-out
  gate/up with the SwiGLU epilogue, BF16-out down; the per-layer verdict is
  the QT44/QT53 expert formats, and the grouping is the stable one so every
  rank shares the grouped layout).
- `HIPFIRE_EP_EXPERT_STAGE`: Flash-Next long-prefill expert staging. It exists
  only when owner-routed prefill is admitted and only on gfx1201; a gfx1100
  rank logs `expert staging not admitted (needs gfx1201)` and builds nothing.

Exact figures, memory arithmetic and the hardware-validation record live in
[`multi-gpu.md`](multi-gpu.md) and [`env-vars.md`](env-vars.md), not here.

### Qwen4 long context, trunk overlay and PLE rows

- **YaRN.** `attention.yarn_factor` (`HIPFIRE_YARN_FACTOR`) overrides the
  checkpoint's RoPE scaling at load (`hipfire_runtime::rope`): unset follows
  the checkpoint's `rope_parameters`, above `1` applies static YaRN over the
  original context, `1` disables it. It is wired for Qwen4 only. Admission
  bounds `max_seq` by the config's `context_limit()` (native positions, or the
  YaRN-extended factor × original) and `QWEN4_MAX_CONTEXT`
  (`crates/hipfire-arch-qwen4/src/config.rs`); semantics in
  [`CONFIG.md`](CONFIG.md).
- **zab trunk overlay.** A `<model>.trunk-overlay` sidecar (an HFQM container
  of pre-encoded QT44 `linear_attn.in_proj_{z,a,b}` tensors, bound to the exact
  base by length and header/index sha256) replaces the base's MQ6 tensors and
  arms the `gfx1151-zab-v1` trunk default
  (`crates/hipfire-arch-qwen4/src/trunk_overlay.rs`).
  `HIPFIRE_QWEN4_TRUNK_OVERLAY` is `auto` by default: applied only on exact
  gfx1151. The client pins the stock-artifact overlay; see
  [`MODELS.md`](MODELS.md) and [`env-vars.md`](env-vars.md).
- **PLE rows.** The per-layer-embedding rows are not uploaded wholesale. They
  come from the embedded Q8 rows, `<model>.ple-q8`, or the 4-bit Lloyd-Max
  `<model>.ple-q4` (`memory.ple_format` / `HIPFIRE_PLE_FORMAT`). `auto` picks
  **q4 on gfx1151** (unified-memory APU) and q8 elsewhere, never deriving or
  downloading: `resolve_qwen4_ple_rows` runs at admission and `hipfire run`
  ensures the sidecar before spawning the daemon.

### Qwen4 NPU trunk (XDNA2)

The Flash-Next family-B MQ6 trunk can run its projections on the XDNA2 NPU.
This is **not** dark: `hipfire-arch-qwen4` depends on `peacemaker-npu` (panel
programs, numerical reference) and, on Linux, on `railgun` with feature `npu`
(amdxdna runtime), and `Qwen4Bundle` prepares the route at load
(`prepare_npu_trunk`, `crates/hipfire-arch-qwen4/src/npu_trunk/`).

The mode is `fusions.npu_trunk` / `HIPFIRE_QWEN4_NPU_TRUNK`
(`off|auto|on|emulate`) and **defaults to `auto`**. `auto` resolves silently to
`off` on every architecture except gfx1151, with no NPU, index or calibration
I/O. On gfx1151 it arms only for a single-device Flash-Next load (hidden size
2560, stock MQ6G256V2 trunk projections, full 8192-row prefill scratch) when
the NPU opens and a matching calibration
(`HIPFIRE_QWEN4_NPU_TRUNK_CAL` / `fusions.npu_trunk_calibration`, else the
compiled-in measured one) selects a nonempty family set; a load refusal logs one
reason and stays GPU-only, while request-time errors stay fail-closed. `off` is
the kill switch; forced `on`/`emulate` refuse other architectures. Details:
[`npu/npu-trunk.md`](npu/npu-trunk.md), [`CONFIG.md`](CONFIG.md),
[`env-vars.md`](env-vars.md). The rest of the NPU stack (`npu-tools`,
`hipfire-xdna`, the AIE2P simulator, co-op prototypes) is experimental
([`npu/README.md`](npu/README.md)).

## Request lifecycle

```text
hipfire run <tag-or-path> "…"
        │
        ▼
Native CLI (`crates/hipfire-cli`)
  resolve registry tag → model path under ~/.hipfire/models/ (or local path)
  if serve up AND not forced local → HTTP POST /v1/chat/completions
    forced local when `HIPFIRE_LOCAL` is truthy or any of `--image`,
    `--kv-mode`, `--kv-backend`, `--kv-k`, `--kv-v`, `--tp` (> 1), `--max-seq`,
    `--head`, `--vision`, `--spec`/`--speculation`, `--model-draft`,
    `--draft-max`, `--dspark-conf-threshold` is passed
    (`run_should_force_local` in `crates/hipfire-cli/src/main.rs`;
    `--json`/`--no-stream` ride the HTTP route and do not force local)
    if HTTP fails while serve still live → abort (no local spawn; would collide)
  else → spawn one-shot daemon binary
        │
        ▼
Daemon (crates/hipfire-daemon/src/main.rs)
  hipfire_loader::load_model(path, …, &mut Gpu)
        │
        ▼
hipfire-loader
  ModelSource::from_path  →  HfqFile  |  SafetensorsSource dir
  Carrier registry probe on arch_id (+ is_dir namespace)
  carrier.load → LoadedModel { arch_id, state: Option<Box<dyn ArchModel>>, tokenizer, … }
  optional: draft/speculator, VL weights, EP/PP scaffolding (Qwen4 EP mesh rides rank 0's bundle)
  Qwen3.5-VL tower sidecar: `params.vision` / `HIPFIRE_VISION_SIDECAR` → separate `qwen3.8-27b-vision.hfq` validated at admission, tower sized by the trunk's `vision_config`
        │
        ▼
generate(…) ladder (`hipfire_generate::ar::generate`)
  select_generation_route (tool/stop gating) → early routes 13 Gemma4, 14 Muse Glimmer
  → route match: Qwen4 (16, no `m.ep`), EP (`m.ep`: 9/10, Qwen3.5 5/6),
    Qwen2 / DS4 / LFM / Cohere / Maple (AR only) / MiniMax / dots.ocr,
    PP `generate_mesh_carrier`, else qwen35/llama body (DFlash/spec, MTP, VL, …)
        │
        ▼
Arch forward (hipfire-arch-*/src)
  prefill (eager per-token and/or batched where implemented)
  decode_step / verify block
  final norm + lm_head when logits needed
        │
        ▼
rdna-compute Gpu methods  ↔  hipfire-dispatch family tables
  kernels/src/*.hip  →  precompiled hsaco or hipcc JIT cache
  ordinary HIP (hip-bridge)  |  retained replay fork (ReplayController → ROCr)
```

Serve long-running path: same daemon binary, HTTP surface documented in
[`SERVE.md`](SERVE.md); chat attach in [`CHAT.md`](CHAT.md).

**Kernel readiness at load.** `load_admitted_with_gemma4_drafter` (the common
single-GPU/PP loader) asks `kernel_registry::route_entries` for the load's
kernel closure. A closed route (today: Qwen3.6-27B MQ4G256V2 XTS on exact
gfx1201, AR or native MTP) begins its bounded batch compile
(`Gpu::begin_route_kernel_load`) while weights upload; the first kernel
request or launch drains it and loads every planned HIP and embedded
PeaceMaker image (`ensure_route_modules_preloaded`); after the model loads,
`finish_route_kernel_load` seals the route, so `loaded` implies no further
compile or module load. Refused routes keep lazy per-kernel JIT; unload and
the next load reset the route. Details and the worker budget:
[`env-vars.md`](env-vars.md) (`HIPFIRE_JIT_JOBS`).

## Model sources (not “two model paths”)

Load is **source × carrier**, not a two-file hard split.

### Sources

| Source | Type | Arch id origin |
|---|---|---|
| HFQ / MQ* artifact | `ModelSource::Hfq(HfqFile)` | Header `HfqFile::arch_id` |
| Safetensors / Paro directory | `ModelSource::Dir(SafetensorsSource)` | `derive_arch_id` from `config.json` (`architectures` / `model_type`) |

Defined in `crates/hipfire-runtime/src/loader_api.rs` and
`safetensors_source.rs`. Unrecognized dir `model_type` yields sentinel
`UNCLAIMED_ARCH_ID` (`u32::MAX`) so routing fails closed instead of defaulting
to Qwen3.5.

Tensor names follow family-dependent HuggingFace conventions (e.g. dense LLaMA
`model.layers.{i}.…`; Qwen3.5 nested `model.language_model.layers.…`). GGUF
inputs map only **recognized** llama.cpp names via `gguf_to_safetensors_name`
at quantize time; unknown GGUF names are retained unchanged.

### Carriers

Object-safe `Carrier` trait + static `REGISTRY` in
`crates/hipfire-loader/src/{lib,carriers}.rs`. `load_model` requires **exactly
one** matching carrier (zero → error; two → ambiguous error).

| Carrier | Claims `arch_id` | Arch crate / notes |
|---|---|---|
| `LlamaCarrier` | 0, 1 | `hipfire-arch-llama` (dense LLaMA/Mistral/plain Qwen3) |
| `Qwen35Carrier` | 5, 6 | `hipfire-arch-qwen35` (+ optional `hipfire-arch-qwen35-vl`) |
| `Qwen2Carrier` | 7 | `hipfire-arch-qwen2` (Q/K/V attention bias path) |
| `DotsOcrCarrier` | 8 | `hipfire-arch-dots-ocr` (vision + Qwen2 text decoder fields) |
| `Deepseek4Carrier` | 9 | `hipfire-arch-deepseek4` |
| `MinimaxCarrier` | 10 | `hipfire-arch-minimax` |
| `Lfm2MoeCarrier` | 11 | `hipfire-arch-lfm2moe` |
| `Cohere2MoeCarrier` | 12 | `hipfire-arch-cohere2moe` |
| `Gemma4Carrier` | 13, 22 | `hipfire-arch-gemma4`; 13 is the primary target, 22 is the EAGLE-drafter sidecar and is excluded from the early generate route |
| `MuseGlimmerCarrier` | 14 | `hipfire-arch-muse-glimmer`; bundle type lives in `hipfire-loader` |
| `MapleCarrier` | 15 | `hipfire-arch-maple`; plain AR only (no spec/MTP/DFlash route) |
| `Qwen4Carrier` | 16 (HFQM artifacts only) | `hipfire-arch-qwen4` (Flash-Next); claims no safetensors directory, so a dir is a conversion input and fails closed |
| `FluxDiffusionCarrier` | 40, 45 | `hipfire-arch-diffusion` (FLUX.1 / FLUX.2 components) |

Canonical numeric registry: [`architecture-ids.md`](architecture-ids.md).

`LoadedModel.state` is `Option<Box<dyn ArchModel>>`
(`hipfire_runtime::arch_model::ArchModel`), not a closed enum: each
architecture's bundle implements the trait in its own crate and owns its GPU
teardown (`free_gpu`), so adding an architecture edits no central match.
Consumers use typed `LoadedModel` accessors (`qwen35()`, `qwen4()`, `llama()`,
`gemma4()`, `gemma4_lowered_mut()`, `muse_glimmer()`, `maple()`, `lfm2moe()`,
`minimax()`, `cohere2moe()`, `deepseek4()`, `dots_ocr()`, `flux_pipe()`) that
downcast through `Any`. Gemma 4 is one of two mutually exclusive bundles:
eager `Gemma4Bundle` or lowered `Gemma4LoweredBundle`. EP (expert-parallel)
for DeepSeek4 / MiniMax stores rank state in `LoadedModel.ep`, not `state`;
Qwen4's EP mesh lives inside rank 0's `Qwen4Bundle`.

### Multi-device load variants

| API | Scope |
|---|---|
| `load_model` | Single GPU (Qwen3.5 PP when `pp > 1` inside the carrier; the Qwen4 carrier also attaches its EP mesh here, `attach_qwen4_ep_peers`) |
| `load_model_ep` | Expert-parallel: arch_id 9 and 10 only; staging reclaims completed/pushed ranks on failure (constructor-mid-failure can still leak that rank’s partial allocs) |

## Architecture crates

Each `hipfire-arch-*` crate owns family-specific config, weight load, GPU
state, and forward — **except LLaMA**: canonical dense forward and shared
transformer types remain in `hipfire_runtime::llama`; `hipfire-arch-llama` is
the facade/re-export plus bring-up/carrier surface. Other runtime-owned pieces
(sampler, prompt frame, EOS filter, loop guard, generic spec traits) stay in
`hipfire-runtime`.

| Crate | Family summary |
|---|---|
| `hipfire-arch-llama` | Facade for dense FA ids 0/1; forward body is `hipfire_runtime::llama` |
| `hipfire-arch-qwen2` | Standalone Qwen2 text |
| `hipfire-arch-qwen35` | Hybrid DeltaNet + full attention; dense (5) and MoE/A3B (6); DFlash/MTP hooks |
| `hipfire-arch-qwen35-vl` | Vision tower attached to Qwen3.5 ids 5/6 |
| `hipfire-arch-qwen4` | Qwen3.8-Flash-Next / Qwen4 (16): hybrid GatedDeltaNet + QSA sparse attention, PLE, 512-expert MoE, native MTP, EP (2..=8 ranks), NPU trunk |
| `hipfire-arch-dots-ocr` | dots.ocr / Qwen2-VL-class OCR |
| `hipfire-arch-deepseek4` | DeepSeek V4 Flash (HC, compressed-KV indexer, optional MTP/DSpark) |
| `hipfire-arch-minimax` | MiniMax-M2 MoE |
| `hipfire-arch-lfm2moe` | LFM2.5 dense + LFM2.5-MoE hybrid short-conv / GQA |
| `hipfire-arch-lfm2-vl` | LFM2-VL SigLIP2-NaFlex vision tower + projector on the arch-11 carrier |
| `hipfire-arch-cohere2moe` | Cohere2-MoE / North-Mini-Code |
| `hipfire-arch-gemma4` | Gemma 4 hybrid sliding/full attention text (13), eager dense and lowered/MoE bundles; EAGLE drafter id 22 |
| `hipfire-arch-muse-glimmer` | Muse Glimmer dense text (14): 3:1 sliding/NoPE-full attention; DFlash drafter id 23 |
| `hipfire-arch-maple` | Maple-Preview natively-ternary 256-expert MoE (15), plain AR |
| `hipfire-arch-diffusion` | Latent image diffusion: FLUX.1 MMDiT (40) and FLUX.2 Klein (45) — components, not chat trunks |
| `hipfire-arch-qwen35-embed` | Qwen3.5 multi-vector embedding encoder (17) — bidirectional full attention + causal GDN, SigLIP tower, 128-d late-interaction head; served only via the non-chat `embed` verb. Under `serve.multi_slot` its own engine merges concurrent requests into one batched forward (`slot_engine.rs`); the trait hook is the defaulted `SlotEngineHandle::encode`, so the generation engine is untouched. |
| `hipfire-arch-embeddinggemma2` | EmbeddingGemma 2 single-vector embedding encoder (18) — bidirectional Gemma-4-family encoder (sliding/full hybrid, PLE, per-layer head dims, 768-d Matryoshka head, mean pooling + L2); served only via the non-chat `embed` verb with the arch-17 slot-engine shape. Text-only is the `vision_mode=off` default: with the shared `.vl` pack resolved (`auto`/`resident`/`on`) the vision and audio towers are resident and media input serves on the `embed` verb; a tower-less load refuses it by modality. Video is not served yet. |
| `hipfire-arch-toy` | Template only (`arch_id = 0xFF`); daemon must not dispatch |

Bring-up contract: implement `hipfire_runtime::arch::Architecture` (see
`hipfire-arch-toy` and production `hipfire-arch-qwen35/src/arch.rs`).

### Image generation (ids 40–47)

Diffusion trunks are **components**: they load through `FluxDiffusionCarrier`
and are refused by text `generate`, so they implement `ArchModel` + `Carrier`
and deliberately not `Architecture` (a latent-step optimizer has no token
stream). `arch_id` 40 is FLUX.1 MMDiT; **45 is FLUX.2 Klein** (4B/9B), keyed on
`_class_name: "Flux2Transformer2DModel"` or `model_type: "flux2"`, daemon name
`flux2_mmdit`. FLUX.2 is a separate forward body in the same crate — bias-free
linears, one shared modulation vector, SwiGLU MLPs, 4-axis id-table RoPE, Qwen3
conditioning instead of T5+CLIP, and a 32-channel VAE whose latent statistics
live in an internal BatchNorm. The request wire is the daemon's
`img_generate`/`img_progress`/`img_done` and HTTP `/v1/images/generations`;
arch 45 adds an **`images`** field (up to four reference image paths, also
`hipfire img --image`) whose VAE-encoded tokens condition every denoise step
without being denoised, which is what makes reference editing a request option
rather than a separate route. Component ids and their detection rules:
[`architecture-ids.md`](architecture-ids.md) § Image-generation component ids.

### Forward shape (typical dense/hybrid layer)

Per layer (names vary by family): pre-norm → mixer (attention and/or short-conv
/ DeltaNet) → residual → FFN norm → gate/up/down or MoE → residual. Final
embedding norm + tied or untied `lm_head` when logits are required. Hybrid
families carry extra recurrent state (DeltaNet, LFM conv tails) alongside KV.

## Dispatch and kernels

### Two related dispatch surfaces

1. **`rdna-compute`** — `Gpu` methods (`gemm_*`, `gemv_*`, attention, norm, MoE,
   sampling, …) used directly by arch forwards. Arch capability predicates live
   in `arch_caps` / feature flags; implementations pick WMMA → specialized →
   baseline. Source modules: `dispatch.rs`, `gemm.rs`, `gemv.rs`,
   `attention.rs`, `moe.rs`, `norm.rs`, `kernels.rs`, `compiler.rs`, `replay.rs`.
2. **`hipfire-dispatch`** — unified family tables and pipelines so callers avoid
   matching on `DType` by hand (`families/`, `tables/`, `pipeline/`, `ops/`).

Principle (both layers): fast paths first; baseline last; drop redundant arch
clauses when a newer predicate subsumes them.

### Kernel build

```text
Rust ensure_kernel source expression (preambles, header stitching, selectors)
        │ hipfire-kernel-registry --arch <arch>
        ▼
exact six-column source/flags registry
        │ scripts/compile-kernels.sh → hipfire-kernel-pack
        ▼
<daemon-bin-dir>/kernels/compiled/<arch>/<module>.hsaco + .index.json + .hash
~/.hipfire_kernels/<arch>/<module>.<hash>.hsaco  # writable JIT cache
```

The runtime verifies the installed index and object SHA-256 before loading,
even with hipcc present. A missing or stale index falls back to JIT only when
hipcc is available; compiler-free loads fail closed. Packaged module names
come from the exact Rust registry, never HIP source basenames.
The installed pack is read-only to the runtime: a missing, stale or rejected
packaged object is recompiled into the hot JIT cache only and is never rewritten
under `kernels/compiled/<arch>`. Only the packaging tools (`hipfire-kernel-pack`
and `daemon --precompile`, which publish to an explicit output directory) write
a pack.

JIT and packaged objects share one core recipe: `--genco --offload-arch=<arch>
-O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6`, then the
extra, per-module and per-kernel flags (`KernelCompiler::recipe_for_source`).
`-fuse-cuid=none` drops hipcc's `__hip_cuid_*` symbol, which is named after the
input path and argv. On one host the object bytes therefore depend only on the
source, the recipe and the resolved toolchain and headers, not on the cache root
or temp name. `-mcode-object-version=6` pins the AMDGPU code-object version
instead of inheriting the compiler default, so a newer LLVM cannot change it.
The cache ABI is part of every hot key, packaging key and index.
It changes whenever the core recipe does, so objects from an older recipe miss
and are recompiled instead of being mixed in.

Some arch crates also ship crate-local HIP (registered through their own
`kernels.rs`) for family-specific ops.

## Serve / generate path

Composition root: `crates/hipfire-daemon/src/main.rs`; the generate bodies and
the route selector live in `crates/hipfire-generate/src/ar.rs`.

After `load_model`, request handling builds sampling defaults
(request → HFQ `generation_config` recommendations → arch ladder) and enters
`hipfire_generate::ar::generate`:

1. **Producer route.** `select_generation_route` (a pure function of
   `GenerationRouteInputs`) names the producer first; the name gates tools and
   stop sequences per route.
2. **Early routes** (`hipfire_loader::generation_early_route`, exact id match):
   `13` → Gemma 4 (eager dense `generate_gemma4`, or `generate_gemma4_lowered`
   for the lowered/MoE bundle), `14` → Muse Glimmer (`generate_muse_glimmer`).
   Arch 22 maps to no early route: it is a drafter sidecar, never a primary
   generate target.
3. **Route match** on the selected route:
   - **Qwen4 (16)** → `Qwen4Spec` when native MTP is attached and the request
     fits it (greedy, or sampled with `speculation.mtp_sampled`), else
     `Qwen4Ar`. Qwen4 never takes the `m.ep` branch: its EP mesh sits inside
     the bundle.
   - **EP** (`m.ep.is_some()`) → `generate_ep` (ds4 / MiniMax ranks and
     Qwen3.5 5/6 mesh loads).
   - **Per-arch bodies** with optional n-gram/spec when a `Speculator` is
     loaded and temp policy allows: ids 7, 9, 11, 12, 10, 8. **Maple (15)** is
     plain AR: it declares no DFlash or MTP and has no verify path, so a
     speculator never changes its route.
   - **Qwen3.5 / LLaMA body**: PP (`generate_mesh_carrier`), DFlash/spec
     (`generate_dflash` / MTP), VL, else dense AR loops.

Continuous batching is a separate exact-id staging route
(`hipfire_loader::continuous_batch_route`): Qwen3.5 (5/6) and Flash-Next
(16). Qwen4 batches only as exact AR lanes through the VMM branch
(`serve.vmm_batch`; there is no fixed-lane Qwen4 body). Lane steps run on one
card, replicated across EP ranks, or band-split across PP bands (sequential
bands, host-staged HC edges; each device's lane store must fit its own free
memory). Multi-card lanes are AR only: no native-MTP lanes and no promotion of a
running singleton; their GPU gates and performance are pending. LFM2 (11) has
no servable batch path.

Capability examples that are **implemented** but not automatically
“product-certified”:

- N-gram draft is model-free opt-in (`HIPFIRE_NGRAM_DRAFT`); many arches
  implement `SpecTarget`, but acceptance and product defaults are separate.
- DFlash draft resolution is CLI-side path/auto-match; daemon loads the path it
  is given. Mode toggles and MoE caveats: config/env owners, not this page.

Speculation inventory: [`speculation-support-inventory.md`](speculation-support-inventory.md)
is a source-derived capability inventory, not product admission; verify in
source before claims.

## Redline roles (capability vs certification)

Three crates + one integration controller. Normative contributor procedure:
[`REDLINE.md`](REDLINE.md) (**branch-implemented** vs `origin/beta` at the
comparison base in [`INDEX.md`](INDEX.md)).

| Piece | Owns | Does **not** own |
|---|---|---|
| `redline` | Experimental direct-KMD via libdrm_amdgpu (device, BO, PM4 CS, fences) | Product serve transport |
| `redline-dispatch` | Dispatch-DAG record/validate, artifact/kernarg identity, plan compile/selection, retained AQL/PM4 graph construction; HIP and AQL backends | ROCr ABI lifetimes; model admission policy |
| `redline-rocr` | Dynamically loaded public ROCr/HSA ABI; queues, signals, AQL packets, arch PM4-IB builders | Model scheduling / backend admission |
| `rdna-compute::replay::ReplayController` | Product integration on `Gpu`: record, prepare, route, poison, reset | A fourth transport crate |

Active retained-PM4 mental model (ordinary HIP launches remain the default path):

```text
recorder-aware launch site
  ├─ ordinary HIP → hip-bridge
  └─ retained replay (ReplayController fork when armed/ready)
       → retained tape (opcodes, kernargs, geometry, deps, bindings)
       → arch-specific PM4 lowering
       → public-HSA command memory
       → one PM4-IB AQL packet by default (PreparedPm4Graph::Single)
         or multiple queue/phase packets when explicitly enabled phased
         multi-queue replay is used (PreparedPm4Graph::Phased)
       → ROCr queue / doorbell / completion
```

**Runtime routing/default ≠ admission or certification.** Example runtime
automatic default (source capability/default fact only, `REDLINE.md`; not an
admission or certification record): `mq4r_redline_default` — exact GPU arch
`gfx1100`/`gfx1151`/`gfx1201` + PP=1 + TP=1 + case-insensitive `.mq4r`
extension configures model default replay backend (model-family agnostic; no
`arch_id` gate); `gfx1200` and all other arches remain opt-in. Built-in `hip`
config profile, explicit `HIPFIRE_REPLAY_BACKEND`, or manual capture disables
the automatic default / opts into broader capability.
Promotion and performance claims require the certification ladder in
`REDLINE.md` (parity, timed-arm route proof, matched stationary conditions).
`ReplayState::Ready` alone is not repository certification. Stitching manual
capture to product timed-arm proof without that ladder is **blocked**
([`INDEX.md`](INDEX.md)).

Do not describe Redline as “future only”: retained record/replay and ROCr PM4-IB
paths are **implemented**; what remains gated is **route certification and
default product claims**.

### railgun: typed authoring and validation, not transport

Do not conflate railgun with Redline. **Redline** is the retained transport:
it records launches, retains the tape and replays it as PM4/AQL through ROCr.
**railgun** is the typed authoring and validation side:

- `railgun` models a *certified dispatch program*: host words and affine
  formulas, resources (including pointer tables), copy and kernel nodes typed
  from certified facts, hazard edges, the gfx12 cache table, the PM4 boundary
  plan and a shadow diff (`crates/railgun/src/{lib,kernel,plan,shadow}.rs`).
- `railgun-cert` lifts a code object with `peacemaker-lift` and writes kernarg
  facts as a receipt `railgun` section; `railgun-jitcorpus` builds the offline
  receipt corpus of runtime-JIT modules and checks coverage; `railgun-corpus` is
  the runtime's read-only view of those records.
- Kernel authoring and instruction validation sit beneath it in
  `peacemaker-ir`, `peacemaker-lift`, `peacemaker-cdna`, `peacemaker-author`,
  `peacemaker-isa` and `peacemaker-rip`.

Its product-tree integration is developer-only and opt-in, and certifies no
route: `HIPFIRE_RAILGUN_SHADOW` diffs railgun's lowering against Redline's
recorded tape without submitting it; `HIPFIRE_RAILGUN_CHECK` byte-compares a
prepared PM4 replay against its HIP twin and poisons the route on a difference;
`HIPFIRE_RAILGUN_BACKEND=railgun` makes the gfx12 single-IB tape execute
railgun's lowering, and any refusal fails closed to HIP, never to the Redline
planner. Knobs and procedure: [`REDLINE.md`](REDLINE.md),
[`env-vars.md`](env-vars.md). The `npu` feature of `railgun` is the unrelated
XDNA2 runtime used by the Qwen4 NPU trunk above.

## KV cache (shared policy)

Resolved by `hipfire_runtime::kv_mode` per load-site policy (aliases and
accepted sets differ by carrier). Concrete modes include:

| Mode | Role (summary) |
|---|---|
| `q8` | Q8_0 K and V |
| `fp8` / `bf16` | Native K+V layout (`fp8`: Qwen single-GPU; `bf16`: Qwen quality-control arm, Maple default) |
| `fwht2` / `fwht3` / `fwht4` | FWHT-rotated K tiers + Q8 V; optional headroom modes |
| `asym2` / `asym3` / `asym4` | **Deprecated since 0.4.0, removal in 0.5.0**: legacy Givens-rotated Lloyd K + Q8 V (fwht3 supersedes Givens asym KV). On Qwen the bare names alias `fwhtN`; the Givens constructors need `legacy-asymN` |

**Qwen-family `auto` / unset** (arch-aware, Qwen only): native **fp8/fp8** on
exact `gfx1201` single GPU when the load is native-eligible (H24/Hkv4/D256
dense or H16/Hkv2/D256 A3B, D256, no adaptive, no CASK
(deprecated since 0.4.0, removal in 0.5.0), no slots/TP/PP split);
**q8/q8** everywhere else, gfx1100 and gfx1151 included. Native `bf16`
stays H24 dense-only. Non-Qwen family defaults are unchanged (Maple BF16, DeepSeek
compressor F32, Gemma layered policy, …). Full K/V axis overrides (`--kv-k` /
`--kv-v`) and precedence live in [`CONFIG.md`](CONFIG.md) / [`CLI.md`](CLI.md).

**Flash-Next (Qwen4, arch 16)** has its own `kv_cache` table for the QSA
full-attention K/V arenas: `auto` is `fp8` on exact gfx1201 (for a head
geometry the kernels implement), `q8` on exact gfx1151 and exact gfx1100
(head_dim 256) and the exact `bf16` state elsewhere; `q8` is opt-in on the
other gfx11/gfx12 arches; `fwhtN`/`asymN` are refused. The GatedDeltaNet state is
separate. Table and sizes: [`CONFIG.md`](CONFIG.md).

**Backend selection** (`--kv-backend` / `memory.kv_backend`): accept only
`legacy` \| `vmm`. Omitted request is **automatic** and prefers **VMM** on
certified combinations; otherwise falls back to legacy once per load with a
logged reason. Selecting legacy (explicit or automatic) emits exactly one
stderr warning whose stable token is the contiguous substring
`HIPFIRE_KV_BACKEND=legacy`. Old spelling `contiguous` is rejected with a
migration error that names `legacy`. Explicit `vmm` on an unsupported
combination fails closed before teardown.

**Physical layout:** the legacy backend still backs the trunk KV arena as one
physically contiguous allocation (stable base pointer for the full
reservation). VMM instead reserves a virtual address range and maps physical
pages on demand; both keep graph/retained base pointers stable within their
model. Private draft caches (DFlash/MTP) are owned separately and do not
relabel the trunk backend. Exact quant layouts and math:
[`QUANTIZATION.md`](QUANTIZATION.md). Hybrid linear layers (DeltaNet) use fixed
recurrent state instead of FA KV for those layers.

**VMM virtual-address lifetime:** a released `VmmArena` unmaps its segments and
frees its physical handles, but never returns its virtual range to the driver
(`hipMemAddressFree` is not called). The range stays reserved for the life of
the process and is counted by `hip_bridge::retired_va_bytes()`. The reason is a
driver behavior seen on ROCm 7.14+ with gfx1201: a VA that was mapped,
unmapped and mapped again to another handle kept translating to its first
backing, so kernels read the previous owner's pages (`crates/hip-bridge/src/vmm.rs`
module docs). A hint-less `hipMemAddressReserve` after a free returns exactly
the freed range, so without retirement every unload → load in one daemon
(model swap, serve idle eviction, `max_seq` change, failed-load retry) would
re-map VAs the GPU had translated before. New reservations are refused with a
restart hint once the retired total would exceed the `RETIRED_VA_CAP_BYTES`
cap of 64 TiB, half of the 47-bit (128 TiB) user VA space. Growth inside a
live arena only ever maps fresh offsets. A failed access reset in `map_next`
poisons the arena instead of letting a retry map a new handle at the address it
just unmapped.

### Qwen4 session cache and live continuation

Flash-Next (Qwen4) reuses prefill work through the engine-owned session cache
(`hipfire_runtime::session_cache`; keying, planning, LRU eviction, the memory
guard, placement and every copy live in the engine, the arch only describes
its state parts and prefill boundaries). It copies state; it does not alias
VMM granules or relocate banks. There is no AR-only fallback tier and no
hipGraph route (Qwen4 replays prepared PM4, below).

- **Snapshots are cold-exact and cross-session.** A prefill snapshots the
  state at every prefill-chunk multiple it crosses (8192 rows on gfx1151,
  4096 on gfx1201): GDN, PLE and hyper state, the selections and, with
  native MTP, the head's state and draft policy whole; of the append-only
  QSA K/V rows (and the head's) only the rows above the parent snapshot.
  A snapshot therefore costs one chunk of rows at any depth, and sessions
  that share a prefix share its links. A hit restores the deepest snapshot
  of the chain and prefills the rest; it equals a cold prefill of the same
  prompt. AR and native-MTP snapshots never cross routes.
- **Commit-gated.** A snapshot is published only after the client commits the
  turn. Snapshots never depend on live state, so a reset does not drop them.
- **Eviction and budget.** LRU over leaves; a snapshot with children is
  pinned. The budget is `memory.session_cache_bytes` /
  `HIPFIRE_SESSION_CACHE_BYTES` (default 8 GiB, `0` = cache off), a ceiling,
  not a reservation. Snapshots live in device memory (VRAM, or a unified-memory
  APU's GTT or carve-out) and yield to context growth. A capture is skipped,
  or older snapshots are evicted, unless free device memory
  (`Gpu::device_mem_info`: `hipMemGetInfo`, on a GTT-backed APU clamped to
  `MemAvailable` minus the CMA region plus TTM's page pool) still holds the live
  state's growth through the prompt in flight plus headroom: on a
  unified-memory APU `clamp(MemTotal / 16, 2 GiB, 8 GiB)`, on a discrete GPU
  256 MiB; every allocation path, disk staging included, keeps it.
  When a forward crosses a VMM growth boundary, snapshots are freed (LRU
  leaves first, then the turn's pending captures) until that growth fits.
- **Live continuation.** Inside one conversation, a turn whose canonical
  tokens strictly extend the previous turn's committed consumed history
  (same route; for native MTP the head is at the same position) prefills
  only the suffix in place, with no copy. It is session-exact (the state
  descends from decode) rather than cold-exact, so a live turn captures no
  snapshot and the shared cache stays cold-exact. When the live state and the
  longest snapshot reach equally far, the live state wins; a hit always
  prefills a non-empty suffix. It needs the session cache attached, so
  `HIPFIRE_SESSION_CACHE_BYTES=0` also turns it off.
- **Removed.** The RC3 end-of-prompt checkpoint, the multi-entry radix store
  with typed PM4 relocation, shared-turn anchors and periodic captures no
  longer exist for Qwen4; `HIPFIRE_QWEN_PROMPT_CACHE` and
  `HIPFIRE_QWEN_RADIX_CACHE` do not affect it (Qwen3.5 and dense keep their
  own conversation cache).
- **Concurrency.** Queued, not parallel: one request executes at a time per
  loaded model; HTTP concurrency queues and there is no GPU slot scheduler.

Switch: `HIPFIRE_SESSION_CACHE_BYTES` (see [`env-vars.md`](env-vars.md)).

## Observability hooks

Examples (full list: [`env-vars.md`](env-vars.md)):

- `HIPFIRE_GRAPH` — hipGraph capture (debug; AR-oriented)
- `HIPFIRE_PROFILE` / rocprof integration — internal vs external timing
- Daemon log under the serve data dir — load progress, JIT, dispatch decisions
- `hipfire-detect` — post-hoc JSONL detectors (non-blocking)
- `hipfire-atlas` — structured bench rows (`--emit-atlas`)

## Where to contribute

| Task | Start here |
|---|---|
| New model family | `.agents/skills/hipfire-arch-port/` + `hipfire-arch-toy`; register carrier + id |
| New GPU arch / WMMA port | arch-port skill; `rdna-compute` + kernel variants |
| Kernel micro-opt | `kernels/src/<name>.<chip>.hip`; dispatch/family wiring; methodology under `docs/methodology/` |
| Quant format / encoder | `hipfire-quantize`; [`QUANTIZATION.md`](QUANTIZATION.md) / [`QUANTIZE.md`](QUANTIZE.md) |
| Retained replay / PM4 | [`REDLINE.md`](REDLINE.md); do not treat `crates/redline` KMD as serve |
| Validation route | [`VALIDATION.md`](VALIDATION.md) only — no universal GPU gate |

## Related owners

| Concern | Owner |
|---|---|
| Arch id table | [`architecture-ids.md`](architecture-ids.md) |
| Docs lifecycle / truth states | [`INDEX.md`](INDEX.md) |
| Redline certification | [`REDLINE.md`](REDLINE.md) |
| Models / VRAM / sidecars | [`MODELS.md`](MODELS.md) |
| Multi-GPU ops | [`multi-gpu.md`](multi-gpu.md) |
| Perf claim protocol | [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) |
