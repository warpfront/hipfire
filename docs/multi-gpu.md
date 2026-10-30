# Multi-GPU operator guide

Operator reference for hipfire multi-device modes. Mutable env inventory lives in
[`env-vars.md`](env-vars.md). Validation route selection lives **only** in
[`VALIDATION.md`](VALIDATION.md) — this page does not invent minimum routes.
Bring-up narrative is historical:
[`multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md).

| Field | Value |
|---|---|
| Inventory date | 2026-10-13 |
| Historical audited source ref | `c7a732ad5b921706519cdba30fdb43a7e62aa2d4` (`origin/beta`); excludes the Unreleased Flash-Next PP route below |
| Page state | Existing routes are **shipped / ref-pinned** (see [`INDEX.md`](INDEX.md)); Flash-Next host-staged PP is **Unreleased / opt-in**, with no hardware qualification claimed here. The [Measured](#measured-flash-next-ep-on-2x-r9700-2026-10-13) tables are **measured** (fixture-bound, dated, not promises); appendix tables are **historical** only |
| Orchestration | `crates/hipfire-runtime/src/multi_gpu.rs` |
| Generic PP load path | `crates/hipfire-loader/src/carriers.rs` (`load_qwen35_pp`) |
| Flash-Next PP admission and refusals | `crates/hipfire-loader/src/admission.rs` (`qwen4_pp_option_refusal`, `qwen4_pp_env_refusal`, `admit_qwen4_pp_mesh`) |
| Daemon load / refusals | `crates/hipfire-daemon/src/main.rs` |
| Supporting multi-GPU script | [`scripts/pp-gate.sh`](../scripts/pp-gate.sh) (not a VALIDATION selector minimum route) |
| Admissions | [`admissions.yml`](admissions.yml) — schema v2, exactly one single-GPU retained-PM4 record; no multi-GPU records (fail closed; none inferred) |

## Modes

Four multi-device routes exist. Generic PP and `tp` are **mutually exclusive** at
load (`tp > 1 && pp > 1` → error); Flash-Next expert parallelism (`topology.ep`)
cannot be combined with either (`pp` or `tp` ≠ 1 is refused when `ep > 1`), and
Flash-Next pipeline parallelism (`topology.pp`) needs `tp = ep = 1`.

| Mode | Load knob | What it does | Source-wired runtime surface (audited ref) |
|---|---|---|---|
| **Pipeline parallel (PP), generic** | daemon load `params.pp` (`N`, default `1`) | Contiguous **layer bands** across `N` devices; the residual stream crosses bands via `boundary_copy`, a peer memcpy. The route is built on peer access (P2P) between the cards; hipfire has no host-staged edge service for it | Qwen3.5 / 3.6 **HFQ** only (`arch_id` 5 dense, 6 MoE/A3B) via `load_qwen35_pp` |
| **Expert parallel (EP / `tp`)** | `params.tp`, or CLI `hipfire serve --tp N` → `HIPFIRE_TP` | Within-layer expert sharding + all-reduce; every rank runs every layer (`Gpus::init_tp`) | MiniMax-M2 (`arch_id` 10) and DeepSeek V4 Flash (`arch_id` 9) via `load_model_ep` |
| **Flash-Next expert parallel (EP, N = 2..=8)** | config `topology.ep` (`hipfire config set topology.ep N`; env `HIPFIRE_TOPOLOGY_EP`; default `1`, allowed `1..=8`; `HIPFIRE_QWEN4_EP=0` forces `1`). **Not `--tp`.** | Routed experts shard by id (`e % N`; an N that does not divide 512 gives uneven per-rank counts); every rank holds the whole trunk and runs the unchanged single-device program on its own worker thread; a copy-only owner exchange feeds the unchanged slot-order combine. **Opt-in, experimental.** | Qwen3.8 Flash-Next (Qwen4, `arch_id` 16) on N = 2..=8 homogeneous cards of one exact arch, **gfx1100** (7900 XTX) or **gfx1201** (R9700), every directed card pair peer-accessible (Qwen4 carrier, `load_qwen4_rank`; admission in `crates/hipfire-loader/src/admission.rs`) |
| **Flash-Next pipeline parallel (PP, N = 2..=8), host-staged** | config `topology.pp` (`hipfire config set topology.pp N`; env `HIPFIRE_TOPOLOGY_PP`; default `1`, allowed `1..=8`) or daemon load `params.pp`. No load-time `--pp` flag; **not `--tp`.** | Contiguous **layer bands**, one per card, every band fully VRAM-resident; the hyper-connection (HC) streams cross each band edge through pinned host memory, so no peer access is needed. The output card (physical card 0) runs the last logical band. **Opt-in, experimental.** See [Flash-Next pipeline parallelism (host-staged)](#flash-next-pipeline-parallelism-host-staged) | Qwen3.8 Flash-Next (Qwen4, `arch_id` 16) on N = 2..=8 homogeneous cards of one exact arch, **gfx1100** (7900 XTX) or **gfx1201** (R9700), with `tp = ep = 1` (admission in `crates/hipfire-loader/src/admission.rs`, `admit_qwen4_pp_mesh`) |

`pp = 1`, `tp = 1` and `ep = 1` are single-GPU. Behavior matches the pre-multi-GPU paths.

**None of the listed PP/EP routes is an admission or product default.**
[`admissions.yml`](admissions.yml) has no multi-GPU records at schema v2 (the sole earned row is single-GPU `pp=tp=1`).
Source-wired means the load path exists in runtime at the audited ref — not that
it is promoted. Flash-Next EP is opt-in (`topology.ep` defaults to `1`) and
carries the `experimental` lifecycle status in [`CONFIG.md`](CONFIG.md); the
defaults listed in [Flash-Next EP: enabling and status](#flash-next-ep-enabling-and-status)
apply only once `topology.ep` is set to a value in `2..=8`. The measured and
exactness evidence on this page was taken on exact gfx1201 R9700s at EP2 and
EP4; it is not evidence for gfx1100 or for other rank counts.

Flash-Next PP is opt-in and experimental in the same way (`topology.pp` defaults
to `1`; lifecycle status in [`CONFIG.md`](CONFIG.md)). This page records no
measurement and no hardware qualification of it; the way to produce that
evidence on your own box is the [PP mode of `scripts/fn-ep-gate.sh`](#qualify-a-box-fn-ep-gate-mode-pp).

**PP is not tensor-parallel serving.** It does not give multi-user throughput.
It is a **capacity** tool: fit larger context / larger HFQ weights by splitting
layers. **No speedup is promised** for models that already fit on one card;
current historical measurements on one 2× gfx1100 station were slower under
sequential PP=2 (see [Appendix A](#appendix-a-historical-pp-evidence-immutable)).

## Which mode for which model

| Model / situation | Cards | Use | Why |
|---|---|---|---|
| Qwen3.8 Flash-Next (Qwen4) | 2..=8 cards of one exact arch, gfx1100 or gfx1201 | **EP** (`topology.ep` = N) | Experts are what does not fit one card. On 32 GB gfx1201 at 32K context EP2 keeps 30 of 49 routed layers in VRAM per rank (19 host-mapped) and EP4 keeps all 49, and EP2 prefill is measured faster than one card ([Measured](#measured-flash-next-ep-on-2x-r9700-2026-10-13)). For 24 GB gfx1100 cards see the [per-rank fit arithmetic](#flash-next-ep-per-rank-expert-fit); gfx1100 and N > 4 have no measurement on this page. |
| Qwen3.8 Flash-Next (Qwen4) on 2..=8 cards of one exact arch **without peer access** between the cards (or where bands are wanted instead of expert sharding) | 2..=8 | **PP, host-staged** (`topology.pp` = N) | EP refuses a missing peer pair; PP stages each band edge's HC streams through pinned host memory and never needs P2P. Every band keeps all its routed experts resident, so the split must fit each card ([budget](#flash-next-pipeline-parallelism-host-staged)); one request's decode visits the bands in order. No measured numbers on this page. |
| MiniMax-M2, DeepSeek V4 Flash | 2+ | **EP via `--tp N`** | The `load_model_ep` route above; no admission, no measured numbers on this page. |
| Dense Qwen 27B-class that fits one card (registry weights: 11.78 GB for `qwen3.8:27b-mq3-xt`, [`MODELS.md`](MODELS.md); the historical probe in Appendix A shows about 15.5 GB for the mq4 27B proxy) | 2+ | **One model per card** (one daemon per card, each pinned by `hardware.devices`; the per-card lock makes two daemons asking for `gfx1201` get different cards) or **batched serving** on one card ([`SERVE.md`](SERVE.md) continuous batching) rather than splitting | Splitting a model that already fits adds cross-card traffic and, under PP, a sequential decode penalty; it does not buy multi-user throughput. |
| Qwen3.5 / 3.6 HFQ that does not fit one card at the wanted context | 2+ (peer access between the cards) | **PP** (`params.pp`, generic route) | PP is a **capacity** tool, not a speedup: historical PP=2 decode measured about two thirds of single-card decode on short prompts (Appendix A). The generic route moves the residual with a peer memcpy; without peer access it is not a qualified configuration. |

**CLI note:** the shipped CLI forwards **`tp`** (`HIPFIRE_TP` / `--tp`) on serve
load messages; that is the knob for MiniMax-M2 / DeepSeek V4 Flash. **Do not
pass `--tp` for Flash-Next:** with `topology.ep > 1` a load with `tp` ≠ 1 is
refused (`cannot combine with pipeline/tensor parallelism`); use `topology.ep`
for expert parallelism and `topology.pp` for pipeline parallelism. There is **no
load-time `--pp` flag** on `hipfire run`, `serve` or `bench` (`hipfire bench --pp`
is the comma list of prompt sizes for `--matrix`, not a pipeline degree). A pipeline degree comes from the
`topology.pp` config key (env `HIPFIRE_TOPOLOGY_PP`), which `run`, `serve` and
`bench` honor, or from `params.pp` on a raw daemon JSONL `load` (an explicit
`params.pp` above `1` wins over the configured value). Generic Qwen3.5 / 3.6 PP
examples and `scripts/pp-gate.sh` set `params.pp` directly.

## Topology (generic Qwen3.5 / 3.6 PP)

Source of truth: `Gpus` in `multi_gpu.rs`. This section describes the generic
Qwen3.5 / 3.6 route; Flash-Next PP has its own placement, transport and budget
rules in [Flash-Next pipeline parallelism (host-staged)](#flash-next-pipeline-parallelism-host-staged).

1. **Device pick** — `hardware.devices` lists physical cards in logical order
   (see [Device selection](#device-selection)). Startup resolves it from the
   KFD topology, reserves the cards, installs ROCr selectors plus HIP logical
   `0..N-1`, and after HIP init aborts unless every logical device has the
   resolved card's arch and PCI bus ID. `Gpus` construction repeats that check.
2. **Layer map**
   - Default: `Gpus::init_uniform(pp, n_layers)` — contiguous bands,
     `base = n_layers / N`, remainder distributed so max−min ≤ 1 layer.
   - Escape hatch: `HIPFIRE_PP_LAYERS=a,b,…` → `Gpus::init_layers` (length must
     equal `pp`, sum must equal `n_layers`). Skips the uniform free-VRAM delta
     check; still enforces arch match unless overridden.
3. **Placement convention (Variant 2)** — `output_device = last` device holds
   `output_norm + lm_head`. Device 0 holds the embedding side of the split
   (generic route; Flash-Next PP puts the output card first in the device list
   and runs the last band on it).
4. **Boundary traffic** — at each band edge, `boundary_copy` issues a peer
   memcpy (`hipMemcpyPeerAsync`); the caller waits with `wait_boundary`. The
   route is built and validated on peer access. The source comment that HIP
   "transparently host-stages" a pair without peer access describes driver
   behaviour, not a hipfire-owned or admitted path; do not rely on it. Only
   Flash-Next PP has an admitted host-staged edge service.
5. **Peer access** — `enable_peer_all` must run **after** weights/KV/scratch that
   need peer maps are allocated. Treat a full peer matrix as a requirement of
   this route: `enable_peer_all` returns `false` for an incomplete matrix and
   the load is not blocked, but the copies then ride on an unqualified driver
   fallback. Partial pair failure does not abort capable pairs.
6. **Preflight**
   - Default: **exact `Gpu.arch` string match** across devices
     (`d.arch != arch0` in `preflight_vram` / `multi_gpu.rs`). This is not a
     loose “family” compare.
   - Free-VRAM delta ≤ `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` (default **2.0**) for
     `init_uniform` / `init_tp` only.
   - `HIPFIRE_ALLOW_MIXED_ARCH=1` opts into mixed-arch pairs (JIT per arch).
7. **Threading** — HIP work is **single-threaded** for the daemon lifetime.
   `bind_thread` before peer enable and before device-bound work. No rayon/tokio
   HIP callers in v1.

EP topology is different: `init_tp` sets every device’s layer map as “all layers
on rank 0” for PP helpers, while the EP forward ignores bands and shards experts.
RCCL all-reduce is used unless `HIPFIRE_TP_USE_RCCL=0` (host fallback not
implemented — that opt-out errors). Non-standard ROCm layouts (e.g. nixpkgs
splitting `librccl` out of the ROCm root) set `HIPFIRE_RCCL_LIB` to the full
`librccl.so` path; the loader tries that before the ROCm root candidates.

### Peer / fabric checks (host)

```sh
rocm-smi --showtopo
rocm-smi --showtoponuma
rocm-smi --showtopoaccess
```

A full `True` peer-access matrix is required by the generic PP and `tp` routes
and by Flash-Next EP (EP refuses a missing pair). Flash-Next PP does not need it:
its default edges are host-staged and never enable peer access.

### Device selection

`hardware.devices` (`HIPFIRE_DEVICES`, singular alias `HIPFIRE_DEVICE`) is a
comma-separated list; its order is the logical device order for PP/TP/EP.
Entries:

| Entry | Meaning |
|---|---|
| `N` | Index into the GPUs sorted by PCI address — the `rocm-smi` order. Not a ROCr or HIP ordinal. |
| `gfxNNNN` | First card of that arch, in PCI order, that is free (its daemon lock is not held) and not named elsewhere in the list. `gfx1201,gfx1201` gives two distinct cards. |
| `GPU-<hex>` | Exact UUID (`rocminfo` / `rocm-smi --showuniqueid`). |
| `DDDD:BB:DD.F` or `BB:DD.F` | Exact PCI address — the only exact form for cards whose UUID reads `GPU-XX`/`0x0`. |

Why a new index: ROCr numbers GPUs in KFD topology node order (`rocminfo`),
HIP numbers whatever ROCr exposed, and `rocm-smi` sorts by PCI bus. On a host
with a 7900 XTX (66:00.0), 6950 XT (6e:00.0), 5700 XT (99:00.0) and Strix Halo
iGPU (bf:00.0) the ROCr order is gfx1100, gfx1151, gfx1030, gfx1010 while the
PCI order is gfx1100, gfx1030, gfx1010, gfx1151. `HIPFIRE_DEVICES=2` is the
5700 XT there regardless of what ROCr calls it.

Resolution reads `/sys/class/kfd/kfd/topology/nodes/*/properties` only (no GPU
contact): nodes with SIMDs are GPUs, `gfx_target_version` gives the arch,
`unique_id` the UUID (0 = none), `domain` + `location_id` the PCI address. The
KFD GPU-node order equals ROCr's agent order (checked against `rocminfo` on a
5× gfx1201 host and the mixed host above). Each card lowers to
`ROCR_VISIBLE_DEVICES` as its UUID, or as its ROCr ordinal when it has none;
HIP gets `0..N-1`. Exact entries are claimed first, then each `gfxNNNN` entry in
list order. Unknown entries, misses, duplicates, a busy exact card, or no free
card of an arch fail closed and print the full device table.

Raw inherited `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` remain the expert
override when `hardware.devices` is unset: compatible pairs are normalized,
ambiguous pairs fail closed, and the daemon then identifies the HIP-visible
cards by PCI address to lock them.

Each daemon holds a non-blocking flock per card in `HIPFIRE_LOCK_DIR` (else
`/run/lock/hipfire`, else `/tmp/hipfire-locks`): `gpu-GPU-<uuid>.lock`, or
`gpu-pci-<dddd:bb:dd.f>.lock` for a card without a UUID. The `gfxNNNN` choice
and the reservation are one step, so two daemons asking for `gfx1201` get
different cards. `scripts/gpu-lock.sh` uses the same keys.

## Logical GPU emulation (developer-only)

Logical GPU emulation is a developer-only diagnostic route for exercising
expert-parallel rank, ownership, and combine behavior on one physical device.
It is not a multi-GPU product mode and does not change ordinary physical
admission.

`HIPFIRE_EMULATE_GPUS` is captured in the startup snapshot before GPU
initialization and parsed as `usize`. Only a successfully parsed integer `>=2`
enables emulation; an unset or malformed value, `0`, or `1` disables it. When
enabled, each requested logical device ID is aliased modulo the loaded
physical-device count:

| Physical devices loaded | Requested logical IDs | Resolved physical IDs |
|---|---|---|
| 1 | `[0, 1, 2, 3]` | `[0, 0, 0, 0]` |
| 2 | `[0, 1, 2, 3]` | `[0, 1, 0, 1]` |

The switch is only an enable switch; it does not choose the logical EP width.
`Gpus::init_ep(2, ...)` owns the two-rank layout and `Gpus::init_ep(4, ...)`
owns the four-rank layout; Flash-Next (Qwen4) EP takes its width from
`topology.ep` (see
[Flash-Next EP under emulation](#flash-next-qwen4-ep-under-emulation)).
Duplicate physical IDs remain refused in ordinary
mode; the snapshotted emulation state is the explicit diagnostic exception for
sealed ownership, load, and batch policy.

### Single-device proof workflow

Use one physical device and keep the selectors explicit:

```sh
unset HIPFIRE_DEVICES HIPFIRE_TP_USE_RCCL
export HIP_VISIBLE_DEVICES=0
export HIPFIRE_EMULATE_GPUS=4       # use 2 for the logical EP2 route
export HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1
export HIPFIRE_EP_PEER_ALLREDUCE=1
```

Use `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` for decode and
`HIPFIRE_EP_PEER_ALLREDUCE=1` for batched prefill/tick. These selectors must be
chosen explicitly for the corresponding route. Never set either selector to
`0`, and never set `HIPFIRE_TP_USE_RCCL=0`: those values select/request RCCL
behavior and are not a host all-reduce fallback.

Aliased logical ranks retain separate rank state, streams, buffers, and sealed
ownership. Same-device peer setup is fail-soft: when two logical ranks resolve
to the same physical device, peer capability/enable calls are skipped for that
pair and the route continues with peer access disabled rather than aborting.

### Flash-Next (Qwen4) EP under emulation

Flash-Next EP honors the same switch. With `HIPFIRE_EMULATE_GPUS >= 2` and
`topology.ep = N` (2..=8), the load maps rank `r` to physical device
`r % count` (`qwen4_ep_rank_devices`, the same alias rule as above), so one
visible device is enough for any N. Each rank keeps its own worker thread,
stream, buffers, expert shard and exchange state. What differs from physical
mode:

- **Device count:** at least one visible device, not N.
- **Arch:** the aliased devices are still checked: exact gfx1100 or exact
  gfx1201, one arch for every rank. Under emulation gfx1151 (Strix Halo, one
  unified-memory GPU) is admitted too; it proves byte parity and lifecycle on
  one device and is never a physical EP admission.
- **Peer checks:** ranks that resolve to the same physical device are never
  probed and no peer mapping is enabled between them (the row gather and the
  tile exchange accept ranks that share a device); each pair of distinct
  physical devices is probed once.
- **Free VRAM:** each rank's free VRAM is its physical device's free VRAM
  divided by the number of ranks aliased onto that device
  (`qwen4_ep_alias_free`), so the residency split is budgeted per alias and
  ranks sharing a card do not each claim all of it.
- **Log:** the plan prints `qwen4 EP logical emulation (HIPFIRE_EMULATE_GPUS):
  rank -> physical device 0->0, 1->1, 2->0, …`.

This is developer-only and not an admission. Under a `HIPFIRE_EMU` capacity
profile logical emulation is refused ([`EMU.md`](EMU.md)).

### Evidence boundary

The logical EP route-oracle suite is designed as byte-exact checks on one
gfx1151 device. A successful run validates sealed ownership, route broadcast,
canonical slot-order combine, result broadcast, inactive rows, and lifecycle
sequencing. The setup documented here is not itself evidence of a completed
lifecycle pass.

A passing logical-rank oracle does **not** prove RCCL, PCIe/xGMI or other
interconnect behavior, physical-device synchronization, throughput, product
gfx1201 admission, or G5 acceptance. The same boundary holds for Flash-Next
EP under emulation: same-device ranks skip peer setup and share one card's
compute and memory bandwidth, and the per-alias free-VRAM division is not a
real per-card fit. Do not describe this evidence as physical EP, interconnect,
or performance validation.

## Launch and config

### PP load (daemon JSONL, generic Qwen3.5 / 3.6)

```json
{"type":"load","model":"/path/to/qwen3.5-9b.mq4","params":{"max_seq":16384,"pp":2}}
```

Example process:

```sh
# Persist one device list for both HIP and ROCr (PCI-order indices).
hipfire config set hardware.devices 0,1

# Optional: asymmetric bands (must sum to n_layers, length == pp)
# export HIPFIRE_PP_LAYERS=16,16

# Bit-stable k-split reduction for pp=1 vs pp=2 parity work
export HIPFIRE_DETERMINISTIC=1

cargo run --release -p hipfire-daemon
# then send the load JSON above on stdin
```

### EP load (CLI): MiniMax-M2 / DeepSeek V4 Flash (`--tp`)

```sh
HIPFIRE_DEVICES=gfx1201,gfx1201 hipfire serve <minimax-or-deepseek4-tag> --tp 2
# equivalent: HIPFIRE_TP=2 …
```

EP does not honor a non-default `--kv-mode` the same way single-GPU load does
today — the CLI warns when both are set. Reload without a DFlash draft for EP.

### Flash-Next EP: enabling and status

Flash-Next (Qwen3.8 Flash-Next, Qwen4, registry tag `qwen3.8:flash-next`) uses
`topology.ep`, not `--tp`:

```sh
# N distinct cards of ONE arch (all gfx1100 or all gfx1201), in logical order; N = 4 shown
hipfire config set hardware.devices gfx1100,gfx1100,gfx1100,gfx1100   # or HIPFIRE_DEVICES=…
hipfire config set topology.ep 4                         # or HIPFIRE_TOPOLOGY_EP=4; any N in 2..=8
hipfire serve qwen3.8:flash-next
# optional: native MTP under EP is opt-in, `auto` stays AR
hipfire serve qwen3.8:flash-next --spec mtp
```

`topology.ep` is an integer in `1..=8` (anything else is rejected by config
validation naming that range). `HIPFIRE_QWEN4_EP=0` forces `topology.ep = 1`
whatever the config says. The first `ep` visible cards become the ranks. The
load is refused (naming `topology.ep` or the missing requirement) when `pp` or
`tp` is not `1`, when `ep` is above 8, when device 0 is not gfx1100 or gfx1201,
when the ranks are not one arch (a mixed gfx1100/gfx1201 set is refused:
`requires the same architecture on every rank`), when fewer than `ep` devices
are visible, or when a directed card pair is not peer-accessible. EP has no
host-staged fallback for a missing peer pair; the load refuses. EP also refuses
DFlash, DSpark, n-gram drafting, DDTree, adaptive KV, drafters, CASK, PFlash,
the retained Redline default and `HIPFIRE_MTP_SAMPLED_MODE=naive`. Exact AR
lane batching (`serve.vmm_batch`) is **not** refused under EP; see
[fn-lanes under PP and EP](#flash-next-fn-lanes-continuous-batching-under-pp-and-ep).
Flash-Next EP stays **opt-in and
experimental** (`topology.ep` defaults to `1`). EP decode is eager (no graph
capture).

| | exact gfx1201 (R9700), N = 2..=8 | exact gfx1100 (7900 XTX), N = 2..=8 |
|---|---|---|
| Load / decode / verify / native MTP (`--spec mtp`) | Admitted (opt-in, experimental) | Admitted (opt-in, experimental) |
| Balanced per-call expert ownership on host-mapped layers ("W1") | **On** by default under EP at N = 2..=4; kill switch `HIPFIRE_EP_BALANCED_OWNERS=0`. At N = 5..=8 **not admitted** (the certified owner-assign kernel covers 2..=4 ranks): each rank logs `balanced ownership not admitted (needs at most 4 ranks); static split` | **Not admitted**: each rank logs `balanced ownership not admitted (needs gfx1201); static split` and runs the static `e % N` split, which is exact |
| Owner-routed prefill chunks ("S1") | **On** by default when every admission condition holds; kill switch `HIPFIRE_EP_OWNER_ROUTED_PREFILL=0`. Symmetric IU4 route (every MoE layer verified symmetric on every rank) | Same default and kill switch. F16 WMMA grouped route (BF16-out gate/up with the SwiGLU epilogue, BF16-out down; QT44/QT53 experts on every layer); the stable grouping is used whenever the exchange is installed |
| Long-prefill expert staging ("S2") | **On** by default when S1 is admitted; kill switch `HIPFIRE_EP_EXPERT_STAGE=0` (pipeline-wide kill switch `HIPFIRE_QWEN4_EXPERT_STAGE`) | Never created (`expert staging not admitted (needs gfx1201)`) |

For the three levers, exactly `0` is off and unset or any other value is on
(they follow one rule, `ep_lever_on`). Single-card (`ep = 1`) behavior is
unchanged by all of them. The per-rank load log states what ran (`qwen4 EP
rank r/n: balanced ownership on N host-mapped MoE layers`; `owner-routed
prefill on …` or `owner-routed prefill not admitted (reason)`). How many routed
layers stay in VRAM depends on the card and N: see
[per-rank expert fit](#flash-next-ep-per-rank-expert-fit). On the 32 GB R9700
fixture at 32K context the changelog-reported figures are EP2: 30 of 49 routed
layers resident per rank (19 host-mapped), +0.34 GiB headroom; EP4: 49 of 49,
+3.77 GiB headroom.

**Exactness.** On the evidence below, EP output on exact gfx1201 is
bit-identical to the same build on one card, not merely close. The evidence
(`fn_ep_oracle` and the serve harnesses, 2×/4× R9700, pinned Flash-Next XTS
sha256 prefix `8b15b6fe`; fixture-bound) covers **exact gfx1201 at EP2 and EP4
only**; it is not evidence for gfx1100 or for other rank counts:

- EP2 and EP4 AR decode: logits and per-request state match one card at every
  step over 256 greedy tokens (ctx 512, 8192, 32768).
- Native MTP: greedy tokens, per-window results and trunk/head/drafter state
  match one card on EP2 (34/34 compared) and EP4 (52/52); greedy MTP equals AR.
- Sampled MTP (T 0.6, top-p 0.95, top-k 20, seed 7, two fresh processes) on
  EP2 with the defaults above: tokens, windows and state identical to one card,
  and per-seed stable. **No sampled-MTP oracle was run on EP4**, and the EP4
  cells above were taken when S1 was still refused at EP4, so no EP4 hardware
  cell for S1/S2 is recorded on this page.
- S1 + S2 on EP2: 25 of 25 compares against one card pass (greedy AR and MTP at
  p8k and p16k, a 4,396-token prompt whose last chunk is 300 rows, bitwise chunk
  digests at layers 0/2/3/31/47); flag-off EP2 equals the previous EP2.

### Flash-Next EP: expert ownership and uneven splits

Routed expert `e` is owned by rank `e % N` and stored at local slot `e / N` of
that rank's compact expert tensors (ascending expert id). N does not have to
divide the 512 experts: rank `r` holds `(512 + N - 1 - r) / N` experts, so the
low ranks hold one more. Per-rank counts (`Qwen4EpPlacement::local_count`):

| N | experts per rank |
|---|---|
| 2 | 256 / 256 |
| 3 | 171 / 171 / 170 |
| 4 | 128 each |
| 5 | 103 / 103 / 102 / 102 / 102 |
| 6 | 86 / 86 / 85 / 85 / 85 / 85 |
| 7 | 74 / 73 / 73 / 73 / 73 / 73 / 73 |
| 8 | 64 each |

Residency is decided per layer for all ranks together: every rank budgets the
**largest** share (`max_local_count = ceil(512 / N)` experts of each routed
stack) when it sizes how many MoE layers stay in VRAM, and the split every rank
applies is the minimum over the ranks' own free VRAM (`ep_vram_layers`). A
layer is therefore resident on every rank or on none; the full-table prefill
route relies on equal residency per layer across ranks. Pinned host bytes are
charged per rank at that rank's own share (an uneven split charges the low
ranks more), and the host RAM and GTT checks cover every rank that has not
pinned yet. Each rank prints its own fit line and `qwen4 EP rank r/n: loaded on
device d (K routed experts of 512 per stack, …)`.

Ownership is sealed once per load (`Qwen4EpSeal`, `crates/hipfire-arch-qwen4/src/ep_seal.rs`):
one EP mesh epoch, every rank's physical device, and one sealed expert plan
per routed-expert group (48 trunk layers plus the MTP head), planned by the
shared `sealed_moe::plan_packed_expert_parallel` from the source's tensor
index. Each rank's compact spans follow the `e % N` stride the exchange kernels
use; the seal checks that its plans give exactly that stride. Each rank loads
its weights under the seal's identity for it (mesh epoch, logical rank,
physical device). The seal also binds the source generation: the base file,
the PLE rows sidecar and the trunk-overlay sidecar, which every rank resolves
for itself. A rank whose files differ from the ones rank 0 sealed is refused
before it allocates, and the mesh refuses a peer bound to another seal, rank or
device. The load log names the epoch:
`qwen4 EP plan: N ranks on devices [...] (mesh epoch E, 49 sealed expert groups), …`.

### Flash-Next EP: per-rank expert fit

How many of the 49 MoE layers (48 trunk layers plus the MTP head's) stay
resident per rank is `floor((free − non-expert weights − reserve) / per-layer
routed bytes)` when they do not all fit, and all 49 when
`49 × per-layer bytes + non-expert weights + reserve` fits in free VRAM
(`ep_rank_layers`, `fits_fully_resident`). The table below is **arithmetic from
the single-card figures pinned in the unit test
`uneven_ep_split_uses_the_largest_rank_share`, not a measurement of an EP
load**: 24,520 MiB free per 24 GB gfx1100 card, 5,115 MiB of replicated
non-expert weights, a 6,656 MiB reserve (`AUTO_VRAM_RESERVE_BYTES`, the 32,768
token baseline), and 1,336,934,400 bytes of routed experts per MoE layer for
all 512 experts (2,611,200 bytes per expert). The real load computes the reserve
with `auto_vram_reserve`, which also depends on the prefill chunk, the KV
backend and native MTP, so the per-rank fit line a load prints is the figure
that applies.

| N | largest rank share (experts) | routed bytes per layer on that rank | MoE layers resident per rank (of 49) |
|---|---|---|---|
| 2 | 256 | 637.5 MiB | 19 |
| 3 | 171 | 425.8 MiB | 29 |
| 4 | 128 | 318.75 MiB | 39 |
| 5 | 103 | 256.5 MiB | 49 |
| 6 | 86 | 214.2 MiB | 49 |
| 7 | 74 | 184.3 MiB | 49 |
| 8 | 64 | 159.4 MiB | 49 |

N = 2 and N = 4 sit within 1 MiB of the next layer (the quotient is 19.998 and
39.997), so a slightly different free-VRAM or reserve figure moves them by one
layer. At N = 2, 3 and 4 the layers that do not fit are host-mapped (the MTP
layer always is when not every layer is resident).

### Flash-Next EP: AR lane reservation

A Flash-Next EP load that requests lane batching (`serve.vmm_batch` on and
`serve.continuous_batch_size` above `1`, the same request the daemon reads
for staging, on the VMM KV backend) charges the AR lane store to each rank
**before** the routed experts are split into resident and host-mapped layers.
The lane store is staged after the weights from the memory that is left, so
without the charge the plan can fill the card with expert layers and staging is
then refused for lack of room. The charge is the same figure staging checks
later: one shared helper computes the lane-store bytes from the lane count, the
load's `max_seq`, the requested prefill chunk, the VMM granularity and a row
budget of at least the lane count, and the load placement and
`stage_rank_lanes` read that one function (the loader's batch staging for EP
does too), so the reserved and the staged bytes cannot drift. The reservation
is planner capacity only: it changes which layers are resident, never a
forward. Singleton forward bytes and numerics are unchanged.

- **Off is identical.** With no batch request (no `serve.vmm_batch`, or
  `continuous_batch_size` of `1`), or on any load that is not a Flash-Next EP
  load, nothing is charged and the placement is the same plan as before. The
  request is a per-load field on the loader's admission, not an environment
  variable, and the plain `load_model` entry points leave it off.
- **Trade line.** A charged load logs exactly one line:
  `[daemon] qwen4 EP lane reservation: lanes=N max_seq=S -> X GiB/rank; resident expert layers A -> B`.
  `A` is the resident expert layer count the plan would have chosen without
  the charge, `B` the count with it. A smaller `B` is the trade: lane memory is
  bought with host-mapped expert layers.
- **Impossible reservation.** If the charge cannot be met even with every
  expert layer on the host, the load keeps the baseline (uncharged) plan and
  logs why. Staging then meets its existing per-rank `lane store bytes <= free`
  refusal, the load keeps the singleton route
  (`continuous_batch_capable = false`), and nothing else changes.
- **Oracle.** `fn_mesh_lane_oracle --reserve-lanes N` loads through the admitted
  route with that request set to `N` lanes and a row budget equal to the
  oracle's staged `--row-budget` (default `max(--ks)`; the loader takes the
  larger of that and `N`). `N = 1` is the same plan as off; without the flag the
  oracle loads as before. Pick `N = max(--ks)` to reserve what the oracle
  stages. The flag is planner capacity only: the oracle's
  singleton-versus-lanes comparison and its `--ks`, `--row-budget` and
  `--policies` are unchanged. The report records `reserve_lanes`.

This section records behavior only; it carries no measurement and no exactness
evidence. The changed placement is not covered by the EP2/EP4 evidence above.

### Flash-Next EP: card topology

All EP exchange traffic (the row gather on decode and few-row forwards, the
tile pull in owner-routed prefill) is peer-to-peer copies between the cards, so
exchange bandwidth is whatever the card-to-card links give. Consumer boards
commonly hang cards off chipset lanes or PCIe switches and run some slots at
x4 or x8; that lowers the exchange bandwidth, and every rank waits for the
slowest link. EP has no host-staged fallback: if any directed card pair is
not peer-accessible, the load refuses. The community gate below records each
card's PCIe link speed and width and the P2P topology in its receipt.

### Flash-Next EP: balanced expert ownership (default on under EP on gfx1201 at 2..=4 ranks; `HIPFIRE_EP_BALANCED_OWNERS=0` disables)

Under Flash-Next (Qwen4) expert parallelism rank `r` owns routed expert `e`
when `e % ranks == r`. On the trunk layers whose experts live in pinned host
memory (the host-mapped layers, read zero-copy over PCIe) the number of routed
experts a token draws from each rank differs call to call, so the rank with
fewer host experts finishes early and waits for the other at every such layer.

Balanced ownership (default on under EP; `HIPFIRE_EP_BALANCED_OWNERS=0` keeps
the static split) balances that per call. After the router wrote the call's
routed ids, one small device kernel deals the distinct routed experts to the
ranks alternately in ascending expert id, identically on every rank, and
rewrites the rank's live pointer tables for the layer: an expert the rank was
dealt keeps its real address (its own pinned tensor, or the owner peer's pinned
tensor, which every GPU can read), every other entry names the rank's zero
dummy expert. The routed kernels run unchanged on the unchanged ids, and the
row exchange then copies each expanded row bit-for-bit from the rank that
computed it. Every routed expert is computed from its own bytes by the same
kernel on exactly one rank, and the canonical combine is untouched, so the
output is byte-identical to the static split and to one card.

Scope: host-mapped trunk MoE layers on one-row decode and few-row forwards
(verify, replay and takeover, 2 to 8 rows). VRAM-resident layers, chunk
forwards (more than 8 rows) and the MTP head keep the static split. With
`HIPFIRE_EP_BALANCED_OWNERS=0`, no table, buffer, stage or kernel module is
added. Each rank logs `qwen4 EP rank r/n: balanced ownership on N host-mapped
MoE layers` at attach when it is on.

Balanced ownership is admitted on exact gfx1201 at 2..=4 ranks only (the
per-call owner-assign kernel ships for that arch and its certified contract
covers 2..=4 ranks). On gfx1100 each rank logs `qwen4 EP rank r/n: balanced
ownership not admitted (needs gfx1201); static split`, and on gfx1201 at 5..=8
ranks `... (needs at most 4 ranks); static split`; either way the rank builds
nothing and runs the static `e % ranks` split, which is exact.

### Flash-Next EP: owner-routed prefill chunks (default on under EP; `HIPFIRE_EP_OWNER_ROUTED_PREFILL=0` disables)

Without the lever every EP rank runs the whole 512-expert grouped MoE of every
prefill chunk, reading its peers' experts through P2P weight tables.
`HIPFIRE_EP_OWNER_ROUTED_PREFILL` (default ON under EP2..EP8 on exact gfx1201 (symmetric IU4 route) and exact gfx1100 (F16 WMMA grouped route); `=0` is the kill switch, any other value or unset is on; single card unchanged)
splits that work. All ranks still run the router, top-k and stable grouping
(a deterministic function of replicated bytes, so the grouped layout is
identical everywhere). Each rank runs the grouped gate/up and down on its own
experts' tiles only (`e % ranks`): the symmetric IU4 GEMMs on gfx1201, the F16
WMMA grouped GEMMs on gfx1100 (BF16-out gate/up with the SwiGLU epilogue and
BF16-out down). The stable grouping is used whenever the exchange is installed
(on gfx1100 the fused scatter's row order is not reproducible across ranks).
A PeaceMaker kernel (`ep_tile_mask_pm_gfx1201` or `ep_tile_mask_pm_gfx1100`) writes
a copy of the tile ids with every tile of a peer expert set to the `-1`
sentinel the grouped kernels already return on before any load, and the
kernels read the copy (the scatter's ids are never modified). The rank then
pulls every peer-owned tile's sixteen BF16 grouped rows of `y_down_grouped`
(the PeaceMaker tile pull: predicated 16-byte copies, two-source kernel on EP2
and eight-source kernel on EP3..EP8, approximately 105 MB per
layer/rank on EP2 or 157 MB on EP4 before grouping padding), and the unchanged
BF16-row combine reads exactly the bytes a single card stores. Per layer:
record ready after down, cross a host barrier, wait every peer's ready, pull,
record done and cross a second host barrier. Right before the next routed
down, wait every peer's previous done (one layer of trunk work later). Two
alternating event pairs make buffer reuse safe for every rank count.
Without an installed launch stream the waits are host
`hipEventSynchronize` calls, as in the decode row gather.

Scope: chunks of at least 512 rows, with a per-layer verdict that holds on every
rank and agrees across ranks: gfx1201 needs every MoE layer verified symmetric,
gfx1100 needs the QT44/QT53 expert formats on every layer (it reads no
symmetric headers). A tail chunk of 9 to 511 rows, few-row forwards (2 to 8
rows), decode and the MTP head keep today's route. Per rank one `qwen4 EP
rank r/n: owner-routed prefill not admitted (reason)` line explains a refusal
(ranks outside 2..=8, an arch other than gfx1100 or gfx1201, no Flash-Next
profile, the sym-IU4 route not requested (gfx1201) or the F16 WMMA grouped MoE
route not available (gfx1100), the compact scratch plan); per-rank verdicts
that disagree log `qwen4 EP: owner-routed prefill refused (per-rank admitted
layers … are not identical and complete)` and chunks keep the full-table
route. The rotation of the grouped
activations still runs over every slot (it has no tile input);
non-owned tiles' activation rows are never consumed. Diagnostics:
`fn_ep_oracle --chunk-dump` (`HIPFIRE_CHUNK_DUMP_DIR`) digests `y_down_grouped`
tile by tile, the FFN output and the HC streams of selected layers for
bitwise comparison with a single card (`fn-ep/coal/s1/compare_chunk.py`).

### Flash-Next EP: long-prefill expert staging

`HIPFIRE_EP_EXPERT_STAGE` (default ON under EP when the owner-routed prefill is admitted on gfx1201; `=0` is the kill switch, any other value or unset is on) admits the existing
`HIPFIRE_QWEN4_EXPERT_STAGE` pipeline under expert parallelism. It is gfx1201
only: on gfx1100 a rank logs `qwen4 EP rank r/n: expert staging not admitted
(needs gfx1201)` and builds nothing. Each rank
borrows its last two resident compact expert layers as alternating DMA stages,
then restores their original bytes before the forward returns. Residency is
unchanged; each host layer copies the rank's owned compact experts (637.5
MiB/rank on EP2 and 318.75 MiB/rank on EP4). Two donor mirrors/rank, each
charged at the largest rank's share, are charged for all ranks against one host's
RAM/GTT budgets; an all-resident load needs no host-layer stage.
Decode and verify continue to use mapped experts and host-wait row exchange,
independent of the staging launch stream. Staging stays outside capture/tapes.
EP staging requires every trunk MoE in the forward to bind the compact
owner-routed route: full-table forwards read peer donor VRAM and never stage.
`begin_forward(ep_compact_all_layers)` is the explicit per-forward admission
input. Before returning to any full-table EP chunk, every rank must call
`ep_quiesce(gpu)` and then cross a host barrier before any rank launches it,
so all donors' restore waits have completed before peer reads.

Stage tables preserve the bound runtime's non-owned entries: peer-mapped
experts for the full-table route, zero dummy experts for owner-routed compact
prefill. Owned expert `e` points at donor slot `e / ranks`. Internally
`Qwen4ExpertStage::prepare_runtime(gpu, layer, runtime)` prepares each full
table once before binding; compact tables are built at load. The compact-route
binder may call `tables_compact(layer)`; `tables(layer)` also selects the last
prepared runtime kind. Single-card tables retain their original parity layout.

### Rehostable owner-routed tile exchange

The PeaceMaker tile mask and pull kernels and runtime `EpTileExchange` are
model-independent building blocks for two to eight ranks on gfx1100 and
gfx1201 (`ep_tile_pm_arch`). The mask
reads `owner_table[expert]`; the pull uses the peer source on EP2 or selects
one of eight registered bases by that logical owner on EP3..EP8. Every rank waits all peers' ready
events before pulling, and all peers' done events before overwriting its
grouped-row buffer. Two alternating event pairs and two host barriers per
exchange preserve the buffer lifetime without another row bank.

The owner table can be derived from the generic runtime's
`ExpertExecutionPlan::rank_ownership()` (`ExpertRankOwnership`), retaining
logical ranks rather than physical device IDs. This transport does not change
the generic EP schedule's numerical combine contract or attach itself to a
model. Flash-Next binds these pieces through its rank-generic S1/S2 adapter
for 0.4.2; moving that adapter onto the generic engine is separate work.
The EP2 pull retains its 48-byte kernarg specialization behind the same API;
EP3..EP8 take the eight-source launch.

### Flash-Next EP: community gate

`scripts/fn-ep-gate.sh [options] MODEL.mq4 [N] [OUT_DIR]` is a one-command
gate for a host with N homogeneous gfx1100 or gfx1201 cards (N in 2..=8;
default: every visible card of the chosen arch, at most 8). From a hipfire
checkout it builds the binaries, runs one card (`topology.ep = 1`) and N cards
(`topology.ep = N`) through `fn_ep_oracle` over two committed prompts (about
500 and about 4,400 tokens, 128 greedy tokens, state digests at steps 1, 64 and
128), and `scripts/fn_ep_compare.py` compares the arms byte for byte: the
tokens, the rank-0 logits sha256, every peer rank's logits mirror sha256 and
every rank's state digests must equal the single card's. It then times
`hipfire bench --spec off` decode and prefill (3 fresh processes per arm by
default, arms interleaved). Each arm runs in a private HOME holding only
`topology.ep` unless `--no-isolate` is given.

`--ref-ranks R` (an integer, `1 <= R < N`, default 1) makes the reference arm
EP too. It is for hosts with less RAM than the one-card reference needs: a
single 24 GB card pins about 50 GiB of host RAM for the experts it cannot hold,
while EP4 on four 24 GB cards keeps all 49 expert layers in VRAM and needs
almost no host RAM. With `--ref-ranks R` the reference arm (`ref-epR`) runs
`topology.ep = R` on the first R entries of the card list, and the EP-N arm is
compared against it byte for byte exactly as above (rank-0 logits and token,
every peer mirror, every rank's state digests against reference rank 0);
`fn_ep_compare.py` also requires the reference to be internally exact (its R
logits mirrors and R state digests all equal its own rank 0) and N to be larger
than R. EP-R equal to one card was itself proven byte-exact for R up to 4 on
gfx1201, so an EP-R reference chains back to the single card. The receipt
config records `ref_ranks` and `ref_arm`. With R = 1 nothing else changes.

The receipt (`OUT_DIR/receipt.json`, schema `fn-ep-gate-receipt-v1`, with a
readable `summary.txt`, `compare.txt` and `gate.log`) holds the verdict and
reason, the configuration, the environment (git/build hash, ROCm, `rocm-smi`
topology and P2P access, host RAM), a per-rank table of arch, VRAM and PCIe
link speed and width, each arm's status, duration and peak VRAM per card (an EP
load that fails, for example on a missing peer pair or out of VRAM, is recorded
with its stderr tail instead of aborting the run), the compare verdict with the
first mismatch, and every speed run with its raw JSON. Exit status: 0 for a
byte-exact PASS, 1 for a mismatch or a failed arm, 2 for an environment or
usage problem. It is supporting manual evidence, not a
[`VALIDATION.md`](VALIDATION.md) selector route.

`--mode pp` runs the same gate for Flash-Next pipeline parallelism; see
[Qualify a box](#qualify-a-box-fn-ep-gate-mode-pp).

`crates/hipfire-runtime/examples/ep_transport_oracle.rs` (feature `lab`) is a
separate no-model synthetic oracle for the row-gather and tile launchers. Without
`--devices` it runs every rank on one device at N = 2..=8. With `--devices 0,1[,..]`
each rank is its own device and host thread, and it proves the host-staged transport
for rigs without peer access: one mapped-pinned host bank (two, alternating) aliased on
every device, each rank publishing its owned rows or tiles into the bank, system-release
ready events, a host sync of every producer before the existing gather kernels read
the bank, and bank reuse only after every reader's done event. It checks every byte,
coherency across bank reuse and a read-before-publish negative control (`xrows`,
`xtiles`, `xctl`), and times EP exchange shapes and pipeline boundary hops against P2P
(`xtime`, `xhop`). `--transport host|peer|both` (default from
`HIPFIRE_QWEN4_EP_TRANSPORT`, `auto` = host when any device pair lacks peer access);
the host transport never enables peer access. It is harness evidence for EP exchange
shapes only: no Flash-Next EP route uses host staging. The product host-staged route
is [Flash-Next pipeline parallelism](#flash-next-pipeline-parallelism-host-staged),
which has its own edge service.


### Environment (operator-facing)

Canonical table: [`env-vars.md`](env-vars.md) (`MULTI-GPU` group). Short map:

| Variable | Role |
|---|---|
| `hardware.devices` | Persistent device list (index, `gfxNNNN`, `GPU-<uuid>`, PCI address); resolved to physical cards, reserved, lowered to ROCr selectors plus HIP logical `0..N-1` |
| `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` | Expert one-shot filters when `hardware.devices` is unset; compatible pairs are normalized and ambiguous pairs fail closed |
| `HIPFIRE_DEVICES` / `HIPFIRE_DEVICE` | Compatibility aliases for `hardware.devices` |
| `HIPFIRE_PP_LAYERS` | Per-band layer counts in logical band order, for generic PP and Flash-Next PP (Flash-Next also needs the first band to hold at least 2 layers) |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | `init_uniform` / `init_tp` free-VRAM delta |
| `HIPFIRE_ALLOW_MIXED_ARCH` | Opt into mixed-arch device sets |
| `HIPFIRE_DETERMINISTIC` | Deterministic WMMA reduction path (parity / bisect) |
| `HIPFIRE_TP` | EP degree (CLI `--tp` sets this) |
| `HIPFIRE_TOPOLOGY_EP` | Flash-Next EP degree (`topology.ep`, default `1`, allowed `1..=8`); **not** `--tp` |
| `HIPFIRE_TOPOLOGY_PP` | Pipeline degree (`topology.pp`, default `1`, allowed `1..=8`; an explicit `params.pp` above `1` wins); the Flash-Next PP route takes `2..=8`. **No load-time `--pp` flag** |
| `HIPFIRE_MESH_TRANSPORT` | Developer-only, Flash-Next PP: how band edges move HC streams: `auto` (default, host-staged), `host` or `peer` (refused) |
| `HIPFIRE_QWEN4_EP` | `0` forces Flash-Next `topology.ep = 1` (developer kill switch) |
| `HIPFIRE_EMULATE_GPUS` | Developer-only logical rank aliasing (integer `>=2`); also maps Flash-Next EP rank `r` to physical device `r % count` (see [emulation](#flash-next-qwen4-ep-under-emulation)) |
| `HIPFIRE_EP_BALANCED_OWNERS` | Flash-Next EP: balanced per-call expert ownership on host-mapped layers (gfx1201 at 2..=4 ranks only; other arches and larger N keep the exact static split); on under EP, exact `0` disables |
| `HIPFIRE_EP_OWNER_ROUTED_PREFILL` | Flash-Next EP (N = 2..=8): owner-routed prefill chunks on gfx1201 and gfx1100; on under EP, exact `0` disables |
| `HIPFIRE_EP_EXPERT_STAGE` | Flash-Next EP: long-prefill expert staging (gfx1201 only); on when owner-routed prefill is admitted, exact `0` disables |
| `HIPFIRE_TP_USE_RCCL` | `0` opts out of RCCL (errors; no host AR yet) |
| `HIPFIRE_RCCL_LIB` | Explicit `librccl.so` path tried before the ROCm root (nixpkgs / split RCCL installs) |
| `HIPFIRE_PP_PFLASH=1` | **Experimental, generic Qwen3.5 / 3.6 PP only** — the daemon accepts PFlash compose with `pp>1` (not a product default; not route-certified). Flash-Next PP still refuses PFlash |
| `HIPFIRE_PP_DFLASH=1` | **Experimental, generic Qwen3.5 / 3.6 PP only** — the daemon accepts a DFlash draft field with `pp>1` (cross-card spec generate is **not** fully implemented; see daemon refusal text). Flash-Next PP still refuses DFlash |

Test-only: `HIPFIRE_HAVE_2_GPU`, `HIPFIRE_PP_PARITY_MODEL` (pp_parity harness).

## Refusals, silent paths, and experimental exceptions (`pp > 1`)

Many non-support cases are enforced at **load** (daemon and/or carrier) and fail
closed — do not expect silent degrade to single-GPU for those. **Exceptions and
carrier-specific wording matter**; do not treat the table as one universal
string or a blanket “always refuse at load.”

| Condition | Behavior |
|---|---|
| `tp > 1` and `pp > 1` | Error: mutually exclusive |
| DFlash `draft` set and `HIPFIRE_PP_DFLASH` unset | Error: DFlash requires `pp=1` (daemon, any arch) |
| DFlash `draft` set and `HIPFIRE_PP_DFLASH=1` | **Experimental exception, generic Qwen3.5 / 3.6 route only:** load may accept; cross-card speculative generate is **not** fully implemented (daemon message states PR2–4 of the hetero PFlash/DFlash plan are incomplete). Not an admission. Flash-Next PP still refuses a draft: `does not support DFlash` |
| CASK / TriAttention sidecar set | Error: requires `pp=1` (daemon); Flash-Next PP also names it: `does not support CASK` |
| PFlash drafter / mode on and `HIPFIRE_PP_PFLASH` unset | Error: PFlash requires `pp=1` (daemon, any arch) |
| PFlash on and `HIPFIRE_PP_PFLASH=1` | **Experimental exception, generic Qwen3.5 / 3.6 route only:** opt-in only; not a product default and not route-certified performance. Flash-Next PP still refuses PFlash: `does not support PFlash` |
| Non-PP carriers at `pp>1` | Carrier-specific error strings (not one universal quote). Examples at the audited ref: `qwen2: pipeline-parallel (pp>1) unsupported`; `llama:` / `dots_ocr:` / `deepseek4:` / `minimax:` / `lfm2moe:` `pipeline-parallel (pp>1) unsupported` on HFQ (and distinct `safetensors + pp>1 unsupported` on dirs); **`cohere2moe: pp>1 unsupported via registry`** (different wording). |
| Qwen3.5 **safetensors directory** + `pp>1` | Error: `qwen35: safetensors + pp>1 unsupported` |
| Qwen3.5 / 3.6 **VL HFQ** + `pp>1` | **Not a hard refuse.** `load_qwen35_pp` is the text HFQ loader only — vision weights are **not** loaded on that path, so VL artifacts **silently become text-only** under PP. Serve real VL at `pp=1`. |
| `hipfire bench --matrix` on a multi-GPU load (PP, TP or EP) | **Not a refusal.** The synthetic `bench_prefill` / `bench_decode` rows run on every topology `serve` admits (the earlier `bench_prefill requires a single-GPU model (pp=1, non-EP)` refusal is gone) and the JSON reports the loaded `topology` as `{pp, tp, ep, cards}` from the daemon's own answers. Rows are synthetic token ids (no prompt, speculation or HTTP path); time a real prompt with `hipfire bench --prompt-file`. See [CLI.md](CLI.md#bench-matrix-topology) |
| Flash-Next (Qwen4) `pp>1`, unsupported options | `qwen4: pipeline parallelism (topology.pp = N) does not support <subject>; drop the option or set topology.pp = 1`, subjects: DFlash, DSpark, n-gram drafting, DDTree, adaptive KV, an EAGLE drafter, CASK, PFlash, an experts-per-token override, a non-Single compute placement, the MTP n-gram modifier. Full list: [refusals](#flash-next-pipeline-parallelism-host-staged) |
| Flash-Next (Qwen4) `pp>1`, environment and config requests | Same sentence with subjects host-mapped routed experts (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`), the expert hotset, expert staging, session snapshots, the session cache disk tier, the NPU trunk |
| Flash-Next (Qwen4) `pp>1`, mesh, bands and budget | Refused before any allocation, naming `topology.pp`: pp outside `2..=8` or `tp`/`ep` ≠ 1, `HIPFIRE_EMULATE_GPUS`, too few visible devices, unknown PCI identity, an arch other than gfx1100/gfx1201, mixed arches, `HIPFIRE_MESH_TRANSPORT=peer`, a bad `HIPFIRE_PP_LAYERS`, a band that does not fit its card |
| Flash-Next (Qwen4) `pp>1` + native MTP | Admitted only on explicit request (`--spec mtp`), for greedy requests and for sampled requests with the default Leviathan verify and neutral penalties; penalized requests, `HIPFIRE_MTP_SAMPLED=0` and `HIPFIRE_MTP_SAMPLED_MODE=naive` sampled requests run AR, MTP sessions are not distributed, DFlash/DSpark/n-gram/DDTree stay refused. fn-lanes under PP are AR lanes only: native-MTP lanes are not staged ([fn-lanes under PP and EP](#flash-next-fn-lanes-continuous-batching-under-pp-and-ep)) |

EP-only: `tp>1` with a DFlash draft → refused; non-EP arch → `load_model_ep` error.
Flash-Next `topology.ep > 1` has its own refusals (pp/tp ≠ 1, `ep` above 8, an
arch other than gfx1100/gfx1201, mixed arches, too few devices, missing peer
access, DFlash/DSpark/n-gram/DDTree/adaptive KV/drafters/CASK/PFlash/Redline,
`HIPFIRE_MTP_SAMPLED_MODE=naive`): see
[Flash-Next EP: enabling and status](#flash-next-ep-enabling-and-status).

### Flash-Next pipeline parallelism (host-staged)

Flash-Next (Qwen3.8 Flash-Next, Qwen4, `arch_id` 16) can run as a **pipeline**
of layer bands, one band per card, over 2..=8 cards of one exact arch (gfx1100 or
gfx1201). It is for boxes where the cards have no peer access to each other, or
where the trunk plus every routed expert has to be spread over several cards'
VRAM. It is opt-in and experimental.

#### What it is

- The trunk layers are cut into `N` contiguous **bands**, one per card. Each band
  keeps every routed expert of its layers resident in its card's VRAM: there are
  no host-mapped experts, no expert hotset and no expert staging under PP.
- The model's hyper-connection (HC) residual streams cross each band edge through
  **pinned host memory** (two host banks per edge). No peer access is enabled or
  needed.
- The **output card is card 0**, the first entry of `hardware.devices`. It runs
  the **last logical band** together with `output_norm`, `lm_head`, the sampler
  and the native MTP head, so logits and tokens come out of card 0. The other
  cards run the earlier bands in the order they follow it in the list. With
  devices `0,1,2,3,4,5` and `N = 6` the bands run on cards 1, 2, 3, 4, 5, 0 (with
  `N = 2`, cards 1 then 0); the input band (embedding side, the PLE rows) is on
  the second listed card.
- `topology.pp = 1` (the default) leaves single-card loads, `topology.ep` and `tp`
  untouched.

#### Enable it

CLI, with the config key or its env spelling (there is no load-time `--pp` flag):

```sh
# six cards of ONE arch; the first entry is the output card and runs the last band
hipfire config set hardware.devices 0,1,2,3,4,5      # or HIPFIRE_DEVICES=…
hipfire config set topology.pp 6                      # or HIPFIRE_TOPOLOGY_PP=6; 1..=8
export HIPFIRE_PP_LAYERS=9,8,8,8,8,7                  # optional, see below
hipfire serve qwen3.8:flash-next
# optional: native MTP is opt-in under PP, `auto` stays AR
hipfire serve qwen3.8:flash-next --spec mtp
```

Daemon: `params.pp` on the load message (an explicit value above `1` wins over
`topology.pp`):

```json
{"type":"load","model":"/path/to/qwen3.8-flash-next.mq4","params":{"pp":6}}
```

The first `pp` visible cards become the ranks. A load needs `pp` in `2..=8`,
`tp = ep = 1`, and cards that are discrete GPUs of one exact arch. Unified-memory
cards and `HIPFIRE_EMULATE_GPUS` aliasing are refused.

#### Band placement: `HIPFIRE_PP_LAYERS`

- `HIPFIRE_PP_LAYERS=a,b,…` lists layer counts in **logical band order**: the
  first count is band 0 (the first layers), the last count is the output band
  on card 0. It is not the card order. The list needs `pp` entries, no zero entry
  and a sum equal to the trunk's layer count (48 for Flash-Next).
- Unset (or empty) gives the **uniform** split: `layers / N` per band, the
  remainder going to the first bands (`N = 5` on 48 layers: 10, 10, 10, 9, 9).
  Flash-Next keeps it whenever every card holds it and otherwise
  [auto-fits](#per-band-memory-budget) the split and the prefill chunk. A list
  pins the split: it is never searched.
- The PLE layer is layer 1, so **band 0 must hold at least 2 layers**; the same
  rule applies to a uniform split and to an explicit list.
- Example for six 24 GB cards and the mq4-xts model: `9,8,8,8,8,7`, so card 0
  (the output card, which also carries `lm_head`, the logits and the MTP head)
  runs the 7-layer last band. It is the split the qualification gate below
  takes as its target; it is an input, not a measured result, and whether it fits
  is decided by the per-band budget below.

#### Expert-slab builds (`mq3-xts`, `mq2-xts`)

The per-expert slab builds run under `topology.pp` as well as the packed `mq4`
artifact. The loader builds each layer's slab arena as it does on one card, and
each card loads only its own band's layers: that band's slab arenas, fully in
VRAM, with the same all-resident budget check as above. The MTP head's experts
are on the output card (card 0) with the rest of the head.

Which build runs on which card type is the same codec rule as on one card:

| Cards (one exact arch) | Builds that load under PP |
|---|---|
| R9700, gfx1201 | `mq3-xts` and `mq2-xts` (and packed `mq4`) |
| 7900 XTX, gfx1100 | `mq2-xts` (and packed `mq4`); `mq3-xts` is refused, because its MQ3 codec is not available on gfx1100, exactly as on one card |

Why it matters: `mq2-xts` is about 47 GB where the packed `mq4` trunk is about
68 GB, so it fits on fewer cards. Per card, the band holds roughly the file size
divided by `N` (the file already carries the non-expert weights), plus the
output card's extra for `lm_head` and the MTP head, and then the per-band
runtime terms from the budget above (F16 shadows, QSA context, state, scratch,
reserve). Approximate band bytes, before those runtime terms:

| Build (size) | N = 2 | N = 3 | N = 4 |
|---|---|---|---|
| `mq2-xts` (about 47 GB) | about 24 GB | about 16 GB | about 12 GB |
| `mq3-xts` (about 59 GB) | about 30 GB | about 20 GB | about 15 GB |
| packed `mq4` (about 68 GB trunk) | about 34 GB | about 23 GB | about 17 GB |

These are rough division figures, not a measurement; admission makes the real
decision per rank and names a card that is short. For example, on 32 GB cards
packed `mq4` does not fit at `N = 2` while `mq2-xts` leaves room; on 24 GB cards
`N = 2` is too tight for any of them once the runtime terms are added. The PLE
rows stay on the host as for every split base, so they are not part of the
per-card figures. Slab loads are otherwise unchanged under PP: the same
refusals apply (no host-mapped experts, no expert staging, no hotset).

#### Transport: `HIPFIRE_MESH_TRANSPORT`

Developer-only, read once at admission. The choice is made per directed edge
before any allocation or peer enable; there is no runtime fallback.

| Value | Behaviour |
|---|---|
| `auto` (default, or unset) | Host-staged edges. Admission has no byte-qualification evidence for peer edges, so `auto` resolves to host-mapped |
| `host` | Host-staged edges, explicitly: two pinned host banks per edge, peer access is never enabled |
| `peer` | Refused at admission: `Peer edges for Qwen4 PP need an admission-time qualification step (HIPFIRE_MESH_TRANSPORT=peer refused: …); … use auto or host` |
| anything else | Refused: `HIPFIRE_MESH_TRANSPORT="…" is not one of auto, peer, host` |

An edge's payload is at most `rows × HC streams × hidden × 4` bytes at the
canonical prefill chunk (the F32 worst case; BF16 halves it), staged in two pinned
host banks per edge, so a host needs that much pinned RAM per edge on top of
whatever else it pins.

#### Per-band memory budget

Every band is **all-resident**. Before any allocation, admission adds up, per
rank, the band's weights, the F16 shadows, the QSA context, the request state,
the forward scratch, the native MTP terms and the reserve, and compares the sum
with that card's free VRAM. With explicit native MTP the output band also charges
the head and its request buffers and every band charges its row capture. A card
that is short refuses the load, naming every failing rank and its shortfall:

```text
qwen4: pipeline parallelism (topology.pp = 6): band does not fit device memory: rank 3 (layers 25..33) needs … bytes (… weights + … F16 shadows + … QSA context + … state + … forward scratch + … native MTP + … reserve) but … bytes are free (short by …); rebalance HIPFIRE_PP_LAYERS or lower memory.max_seq (host-mapped band layers are not admitted)
```

There is no spill to host-mapped experts. The canonical prefill chunk is the
same on every band, and a band whose free VRAM cannot hold it at attach
refuses instead of shrinking it. What changes how a short card is handled is
the auto-fit below.

##### Auto-fit (`HIPFIRE_PP_LAYERS` and `HIPFIRE_PREFILL_CHUNK_ROWS` unset)

With neither variable set, the load decides the split and the chunk once,
before any allocation, with the same pure census as the refusal above:

1. **The default wins when it fits.** The uniform split at the default chunk
   (4096 rows on gfx1201) is taken whenever every card holds it; such a load
   is unchanged and logs nothing extra.
2. **Otherwise the chunk steps down.** The rungs are 2048, 1024, 512 and 256
   rows, only those below the default chunk (and never above `memory.max_seq`).
   The **largest** chunk at which some split fits wins. At each rung every
   split with at least 2 layers in band 0 is tried for `pp = 2`; for `pp > 2` a
   deterministic steepest-descent walk starts at the uniform split and moves
   one layer at a time between any two bands.
3. **The most balanced fitting split at that chunk is taken:** the lowest
   maximum of `required / free` over the cards; a tie goes to the split closest
   to uniform, then to larger earlier bands.
4. **One log line** states the move, ahead of the usual per-rank
   `need … MiB of … MiB free` lines:

   ```text
   qwen4 PP auto-fit: layers 25,23 chunk 1024 (uniform 24,24 at chunk 4096 short by 1754 MiB on rank 1)
   ```

5. **If nothing fits** the refusal above is raised for the nearest candidate
   and ends its text with the search: `auto-fit searched chunk rows … none fits;
   the nearest, layers … at chunk …, is short by … MiB on rank …`.

**Pin it.** `HIPFIRE_PP_LAYERS=a,b,…` turns the search off entirely: the split
and the default chunk are used as given and a short card refuses exactly as
before. `HIPFIRE_PREFILL_CHUNK_ROWS=N` keeps the chunk (it is never stepped
down) and only the split is searched.

**What it changes in the output.** The chunk changes numerics: output equals a
single-card run at the same chunk (see [What runs](#what-runs)), so a load the
auto-fit stepped down to chunk 1024 equals a single card at
`HIPFIRE_PREFILL_CHUNK_ROWS=1024`, not the 4096 default; pin the same chunk on
both sides to compare. The split does not: a cut-safe band split is designed
to equal one uncut card byte for byte and the PP gate checks it (the
`--ref-ranks` row of the [`fn-ep-gate` options](#qualify-a-box-fn-ep-gate-mode-pp)).
Both statements are the existing documented claims; this page records no
measurement of the auto-fit's own output.

**Worked example (arithmetic from measured loads, not a new measurement).**
Flash-Next `mq3-xts` (59.25 GB) on 2× R9700 with 32,550 MiB free per card at
`memory.max_seq` 16,384: uniform 24,24 at chunk 4096 needs 35.97 GB on rank 1
(30.87 weights, 0.62 F16 shadows, 0.16 QSA context, 0.03 state, 1.18 forward
scratch, 0.93 native MTP, 2.18 reserve), 1.84 GB short; rank 0 fits. No split
fits at 4096 or 2048 rows; at 1024 rows exactly 25,23 fits (the hand-tuned
split, rank 0 32,476 MiB, rank 1 32,377 MiB including 755 MiB of MTP). The
auto-fit therefore picks `25,23` at chunk 1024 with no environment variables.
A 64K context (`memory.max_seq` 66,048) needed chunk 256 in the measured
hand-tuned load; the auto-fit walks the same ladder and, from that
measurement, is expected to land on `25,23` at chunk 256.

#### What runs

- **AR decode** and **pipelined prefill**: a prompt is cut into canonical chunks
  that flow through the bands in order; only the last chunk computes the head.
  **gfx1201 prefill chunk:** under PP the canonical chunk defaults to **4096
  rows** (a single card stays at 16384 rows under the compact plan), so a 32K
  prompt is eight chunks in flight instead of two or three and the band pipeline
  fills. An explicit `HIPFIRE_PREFILL_CHUNK_ROWS` wins (`=16384` restores the
  previous chunk); the bands and the mesh planner share one chunk, the default
  chunk when every card holds it and otherwise the one the
  [auto-fit](#auto-fit-hipfire_pp_layers-and-hipfire_prefill_chunk_rows-unset)
  chose. Output equals a
  single-card run at the same chunk. Decode is unchanged (~54 tok/s). Measured
  2026-10-28, 4x R9700 (PP4), packed mq4-xts, beta c7c44097cc + explicit chunk
  env, greedy, one fresh process per arm:

  | Prefill tok/s | 16384 rows (old default) | 8192 rows | 4096 rows (new default) |
  |---|---:|---:|---:|
  | 8K | 4,838 | 6,039 | 6,709 |
  | 32K | 6,953 | 14,026 | 16,657 |
  | 64K | 13,445 | 16,735 | 18,260 |
  | 131K | 15,194 | 17,444 | 17,885 |
- **Per-band sessions**: every band attaches its own in-memory session cache
  (`memory.session_cache_bytes`, an independent opportunistic budget per band, see
  [`env-vars.md`](env-vars.md)); only client-committed all-band snapshots are
  reused. The disk tier and message-end snapshots are refused. A load with native
  MTP attaches no session cache (the output band logs
  `qwen4 session cache: off under topology.pp with native MTP`); AR loads keep theirs.
  **Known limitation:** only the all-band stride snapshots (every 4,096 tokens) are
  reusable under PP. The single-card resident tail/lineage snapshot is not shared
  across bands, so a prompt shorter than 4,096 tokens is always prefilled cold and
  a longer one reuses at most its last 4,096-token boundary.
- **Native MTP, explicit only**: `speculation.mtp = on` / `--spec mtp`.
  `auto` and `off` stay AR under PP. The output band drafts, samples and decides;
  every band runs the same verify block `[seed, drafts...]`, partial rejections
  are repaired per band, and the output band finishes the head. Its exactness
  contract is the single-card native-MTP result (tokens, window records, canonical
  state including the MTP head); qualify each hardware cell with `--mtp` in the gate below.
- **Sampled native MTP (Leviathan) with neutral penalties** runs on the mesh with
  the same route as greedy MTP: temperature > 0 requests (the default
  `speculation.mtp_sampled`, `HIPFIRE_MTP_SAMPLED_MODE` unset or `leviathan`)
  draft from the head's re-scored candidates, verify through the same
  `[seed, drafts...]` block and accept by speculative rejection sampling, so the
  emitted stream equals PP AR in distribution (not bit-for-bit; the sampled-MTP
  chi-square oracle `sampled_mtp_distribution_hw` runs it with
  `HIPFIRE_SAMPLED_MTP_TOPOLOGY=pp2`). Every draft and target-row read goes
  through the output band's device. **Penalized sampled requests**
  (repeat/presence/frequency), `HIPFIRE_MTP_SAMPLED_MODE=naive` (its draws come
  from the process-wide AR sampler) and `HIPFIRE_MTP_SAMPLED=0`
  (`speculation.mtp_sampled = false`, the kill switch) decode **AR** on the mesh
  instead of failing, as do penalized greedy requests. MTP sessions are not
  distributed: a native-MTP request on the mesh never reuses a session snapshot
  and publishes none. A speculative request that still reaches the mesh with
  DFlash/DSpark/n-gram/DDTree, KV eviction or a checkpoint resume is refused
  before any state changes.

#### Refusals

All of these fire at load, before any card is touched. The rows of the next table
share one sentence: `qwen4: pipeline parallelism (topology.pp = N) does not support
<subject>; drop the option or set topology.pp = 1`. Within the option rows and
within the environment rows, the first match wins.

| Subject | When |
|---|---|
| DFlash | a draft path or DFlash speculation is requested |
| DSpark | DSpark speculation is requested |
| n-gram drafting | n-gram drafting is requested |
| DDTree | DDTree is requested |
| adaptive KV | `kv_adaptive` is on |
| an EAGLE drafter | a drafter model (`params.drafter`) is given |
| CASK | a CASK / TriAttention sidecar is given |
| PFlash | PFlash is requested |
| an experts-per-token override | an experts-per-token override is given |
| a non-Single compute placement | a compute placement other than `single` is given |
| the MTP n-gram modifier (speculation.mtp_ngram = on; leave it auto or off) | native MTP is on and `speculation.mtp_ngram = on`; `auto` and `off` are accepted and build no n-gram pool |
| host-mapped routed experts (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`) | the variable is set (any value) |
| the expert hotset (`HIPFIRE_QWEN4_EXPERT_HOTSET`; set 0) | the variable is set to anything but `0`; unset is accepted and PP never applies a hotset |
| expert staging (`HIPFIRE_QWEN4_EXPERT_STAGE`) | exactly `1`; unset or `0` is accepted |
| session snapshots (`HIPFIRE_QWEN4_TURN_SNAPSHOTS`) | exactly `1`; unset or any other value is accepted |
| the session cache disk tier (`memory.session_cache_disk_bytes`; set 0) | the budget is above `0` |
| the NPU trunk (`fusions.npu_trunk = on` or `emulate`) | the mode resolved for the cards' arch is `on` or `emulate`; `off` and `auto` are accepted |

Mesh, rank, band and budget refusals use the prefix
`qwen4: pipeline parallelism (topology.pp = N)`:

- `is admitted for 2..=8 stages with tp = ep = 1 (tp=…, ep=…)`; `topology.ep` above
  `1` is refused as `cannot combine with expert parallelism`.
- `owns physical devices; HIPFIRE_EMULATE_GPUS aliasing is not admitted`.
- `needs N visible devices, M visible (hardware.devices / HIPFIRE_DEVICES)`.
- `needs a known PCI identity for every rank; rank R (device D) has none (KFD
  topology unreadable)`.
- `requires gfx1100 or gfx1201 on every rank; rank R is …`, `requires the same
  architecture on every rank; rank 0 is …, rank R is …`, and `requires every rank
  to match the primary device; ranks are …, device 0 is …`.
- `HIPFIRE_MESH_TRANSPORT=peer` and unknown transport values (table above).
- `HIPFIRE_PP_LAYERS parse: …`, `HIPFIRE_PP_LAYERS has K entries, expected pp=N`,
  `HIPFIRE_PP_LAYERS sum=S != n_layers=L`, a zero entry (`each device must own ≥1
  layer`), and a first band under two layers (`first PP band has … layers,
  requires at least 2 (the PLE layer lives in the first band)`).
- `band does not fit device memory: …` (the budget above).
- The band loader also refuses a unified-memory card, an automatic context, a
  trunk-overlay sidecar and a set `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`
  (`qwen4: topology.pp does not admit … (host-mapped expert layers)`).

Also refused under PP: the daemon-level `tp > 1 && pp > 1`
(`mutually exclusive`). Exact AR lane batching is admitted under PP; see
[fn-lanes under PP and EP](#flash-next-fn-lanes-continuous-batching-under-pp-and-ep).
The synthetic `hipfire bench --matrix` probes
(`bench_prefill` / `bench_decode`) are **not** refused under PP, TP or EP.

#### Flash-Next fn-lanes continuous batching under PP and EP

With `serve.vmm_batch` on, a Qwen4 load under `topology.pp` or `topology.ep`
(`2..=8`) stages the same exact **AR fn-lanes** route as a single card
(`continuous_batch_route = "fn-lanes"`, greedy-only lanes), default-off like
the single-card route. Exactness rests on two byte-identity receipts (PP2 mq2,
and EP2 mq4; see the stage-policy bullet below); the serve figures are in the
**Measured** bullet at the end of this section.

- **Staging and admission.** Lanes are staged after the weights, from each
  device's remaining free memory. The check is per device: every band (PP) or
  rank (EP) needs `lane store bytes <= free` on its own card. If any device does
  not fit, nothing is staged, the reason is logged and the load keeps the
  singleton route (`continuous_batch_capable = false`). Without the EP lane
  reservation below, staging does not change the singleton's own admission and
  residency plan. Under EP a lane request can reserve its bytes in that plan
  first: [Flash-Next EP: AR lane reservation](#flash-next-ep-ar-lane-reservation).
- **EP.** The trunk is replicated, so every rank holds a full lane-store replica
  and runs each lane entry through the same per-rank worker fan-out as the
  singleton EP forward.
- **PP.** Each band holds a band-scoped lane store. HC streams cross each band
  edge through pinned host memory (no peer access), exactly as the singleton
  PP forward. A decode step is cut into `G` contiguous, balanced, whole-lane
  micro-batches (a lane's rows are never split) that wavefront through the
  bands with sender completion events gating each receiver upload
  (`HIPFIRE_PP_LANE_OVERLAP=0` selects the sequential band walk). Prompt
  tiles run as singleton PP tiles. Per-step token staging uses pinned host
  memory with event-fenced asynchronous uploads; it must stay byte-identical
  to a synchronous upload, and no speedup from it is measured or claimed here.
- **PP micro-batch count `G`.** `G` ranges over `1..=min(lanes, 2 * stages)`.
  Stage-policy admission is per-stage proof-row based (see the stage-policy
  bullet below); nothing is assumed per-lane or shared by arch alone. Under a
  no-sharing policy (per-lane, no row-local sharing to trade against) `G` is
  the maximal overlap `min(lanes, 2 * stages)`. Under a shared policy `G` is
  adaptive: bounded exploration of each `G`, then the `G` with the cheapest
  observed complete-step wall (the step's final output sync included, not
  caller-side picks), ties broken deterministically; statistics are kept
  separately per exact row shape, stage count and policy, so a policy change
  never mixes timings. The adaptive path is production opt-in only. Two
  developer-only variables steer it (both in
  [env-vars.md](env-vars.md#qwen4--qwen38-flash-next-arch-16)):
  `HIPFIRE_PP_LANE_GROUPS=<G>` forces `G` (a positive value is clamped to
  `min(active lanes, 2 * stages)`; `0` or a non-integer refuses), overriding
  the per-lane maximal overlap too; `HIPFIRE_FN_LANE_POLICY=<spec>` (a
  `LaneStagePolicy::parse` spec, read at lane-store construction, applied to
  every band/rank store and logged; a bad spec refuses the load) selects the
  stage policy. Every `G` and every policy must keep lane outputs
  byte-identical; the choice is a scheduling decision, not a numerical one,
  and no measured gain is claimed on this page.
- **AR lanes only.** Native-MTP lanes are not staged under PP/EP
  (`continuous_batch_spec = false`, `continuous_batch_spec_mode = "off"`),
  whatever `speculation.mtp` requests.
- **No promotion.** A running singleton request is never promoted into a lane
  under PP/EP (`continuous_batch_promotion = false`); on one card it is `true`.
- **Stage-policy admission is per-stage proof rows.** Sharing is decided per
  row-local stage, from the weights each lane store actually bound, never from
  the production shape or from a whole-policy certificate or flag. A stage
  becomes `OnceAllRows` only if **every** owned weight-instance route of that
  stage matches a `RowTwinProof` row `{arch, stage, route: StageRoute,
  domain: &[LaneTopology], receipt}` of `LANE_ROW_TWIN_PROOFS`; one unmatched
  instance (an unproved projection, router, shared-expert or MoE route, or a
  topology outside the row's `domain`) leaves only **that** stage per lane,
  with the reason recorded, while the other stages keep their own proofs. The
  stateful stages (`ple_conv`, `gdn_core`, `qsa_core`) are always per lane.
  The routes are derived per bound instance (`LaneStageRoutes`) and resolved
  into a policy plus evidence and per-lane reasons (`LanePolicyResolution`).
  gfx1151 keeps all its historical topologies and expert classes: its rows are
  wildcard-route rows over every `LaneTopology`, preserving the old behavior
  under the exact receipt `GFX1151_LANE_RECEIPT`. The gfx1201 rows are
  route-bound and carry one receipt per proved topology.

  *PP rows.* The PP2 mq2 rows (PP domain) carry `GFX1201_PP2_MQ2_RECEIPT`
  (hold j1574, `fn_mesh_lane_oracle` on 2x R9700): with every row-local stage
  shared and with every stage per lane, each lane's ids and every logits row
  were byte-identical to the serialized PP singleton at K=1..8 (8 prompts x 64
  tokens) under adaptive, 1 and 4 decode micro-batches and with overlap off.
  Their routes are the actual all-MQ6 projection route of the PP2 artifact
  (GDN input, QSA and head projections MQ6, Q8 embedding), the BF16
  HC/router/shared routes, and the resident MQ4/E8P MoE route with per-expert
  `DecodeGemv` capability admission.

  *EP rows.* The EP rows (`[Ep]` domain) are twelve: eleven non-MoE rows
  (embedding, PLE rows, HC read, HC write, GDN in/out projections, QSA in/out
  projections, final HC read, head and argmax) and one MoE row. All carry
  `GFX1201_EP2_MQ4_RECEIPT` (hold j1570, `mcb-notes/runs/ep-diag-8e148906a-1`,
  2026-10-18, gfx1201 EP2 mq4-xts): production lane ids and logits were
  byte-identical to the serialized EP singleton at K=1, 2, 4 and 8 over 64
  tokens. The bound routes a non-MoE row requires are those the proved
  artifact binds: MQ6 for every GDN qkv/z/a/b/out and QSA indexer/q/k/v/out
  projection and for the head, Q8 embedding, BF16 HC, BF16 PLE rows and F32
  argmax.

  *EP MoE class.* The MoE row requires the route class
  `LaneExpertRoute::EpPackedStaticOrBalanced`, with BF16 router/shared routes
  and per-expert `DecodeGemv` capability admission. The class is decided per
  owned MoE layer, and **every** owned MoE layer must be one of exactly two
  things: (a) the actual static gathered runtime over fully resident packed
  QT44/QT53 experts, or (b) the actual balanced-live gathered runtime over
  fully host-mapped packed QT44/QT53 experts. In both, the gather must be
  installed; there must be no CPU (D1/D2), tier, slab, pruned or FP8 expert
  route; and the layer's dtype buffers and the runtime it selected must
  agree. Any mix of (a) and (b) layers is admitted, with no restriction on how
  many layers are resident (27 resident + 21 host-mapped, 28 + 20, and so
  on). A single MoE layer outside that class matches no row, so the MoE stage
  alone runs per lane.

  *Fallback and limits.* A different bound route, arch or topology (a mixed
  or otherwise different artifact route, or any expert route outside the EP
  MoE class above) matches no row, and the fallback is stage-specific: only
  the affected stage runs per lane with the singleton's typed ops. The `Ep`
  domain does not encode the rank count, and the measured EP cells are EP2
  mq4-xts at K=1, 2, 4 and 8 only: the receipt is exactness evidence, not
  performance, and it evidences no other EP width, topology count or latency
  (the admitted class covers both ownership runtimes above, but no
  throughput or per-stream figure is attached to either). Both receipts are
  exactness evidence, not performance. A future arch or route qualifies
  solely by adding a proof row; no resolver code changes. The evidence is the
  deduplicated set of receipts of the proofs actually used, logged once per
  store as `qwen4 lanes: stage policy <describe()> (proofs <receipts>;
  per-lane: <stage>=<reason>,...)` and reported through
  `fn_lanes_stage_policy` / `fn_lanes_stage_evidence` in the load ack. The
  developer `HIPFIRE_FN_LANE_POLICY` override is unchanged. On a stage that
  stays per lane, a step of C lanes runs that stage C times serially under
  both topologies; this is the policy, not a tuning gap.
- **Load ack.** `fn_lanes_topology` is `single`, `ep` or `pp`;
  `fn_lanes_device_receipts` has one entry per card (EP: rank order; PP: input
  band to output band). The scalar `fn_lanes_*` receipt fields are the root's:
  rank 0 under EP, the output band under PP.
- **Bench.** `hipfire bench --concurrency … --backend batch` accepts the `vmm`
  and `fn-lanes` routes. It loads through the same path as the other bench
  arms, which forwards the configured `topology.pp` (load `params.pp`) and the
  configured `topology.ep` (daemon process config), so the sweep measures the
  PP/EP topology configured for the model.
- **Measured (S1, 2x R9700 gfx1201, 2026-10-19).** Serve, greedy, thinking
  off, 16 prompts x 256 tokens, `continuous_batch_size` 8, K closed-loop
  clients, 3 fresh processes per arm (PP batching-off C1: 4), the
  packet binaries against its parent with batching off. Output tok/s (mean);
  ratios against the batching-off C1. Every cell's text equals the
  batching-off text, and the stacked oracle cells on the same binaries passed
  (PP2 K=1, 8; EP2 K=3, 5, 6, 7; ids and logits).

  | topology | off C1 (parent) | off C1 | on C1 | on C4 | on C8 |
  |---|---|---|---|---|---|
  | PP2 mq2-xts | 50.01 | 49.90 (median 49.99 vs 49.95) | 49.88 (−0.3%) | 79.19 (1.58x) | 98.61 (1.97x) |
  | EP2 mq4-xts | 45.80 | 45.46 (median 45.89 vs 45.82) | 44.78 (−2.2%) | 75.38 (1.65x) | 83.72 (1.83x) |

  A global default flip of `serve.vmm_batch` still needs single-card
  Flash-Next C1/C4/C8 with batching on (C1 no regression, C4/C8 above C1, 3
  fresh processes per arm).

#### Qualify a box: `fn-ep-gate` mode `pp`

`scripts/fn-ep-gate.sh --mode pp` is the one-command PP gate. It takes the same
`MODEL.mq4 [N] [OUT_DIR]` positionals and options as the EP gate (`--arch`,
`--devices`, `--tokens`, `--reps`, …) plus:

| Option | Meaning |
|---|---|
| `--mode ep\|pp` | `ep` (default) is the unchanged EP gate; `pp` runs pipeline parallelism |
| `--pp-layers COUNTS` | band counts in logical band order, exported as `HIPFIRE_PP_LAYERS` |
| `--transport auto\|host` | edge transport (`HIPFIRE_MESH_TRANSPORT`), default `host` |
| `--ref-ranks R` | the reference arm is PP-R on the first R cards instead of one card (default `1`, one card with host-mapped experts). A cut-safe band split is designed to equal one uncut card byte for byte and the gate checks it, so PP-R is a valid reference when one card would need more pinned RAM than the host has |
| `--mtp` | adds the native-MTP exactness arm under a fixed policy: `HIPFIRE_MTP_AR_FLOOR=0`, `HIPFIRE_MTP_INCREMENTAL=0`, `HIPFIRE_MTP_NGRAM=off`, oracle flags `--spec mtp --temp 0 --ngram off` |

```sh
# six cards of one arch, bands 9,8,8,8,8,7 (card 0 runs the last band); the reference
# is PP-R on the first R cards, with R chosen so those bands fit their cards and the host RAM
scripts/fn-ep-gate.sh --mode pp --pp-layers 9,8,8,8,8,7 --ref-ranks "$R" MODEL.mq4 6 OUT
# the same, plus the explicit native-MTP exactness arm
scripts/fn-ep-gate.sh --mode pp --pp-layers 9,8,8,8,8,7 --ref-ranks "$R" --mtp MODEL.mq4 6 OUT-mtp
```

The reference arm and the PP arm both run `fn_ep_oracle` with `--canonical-state`
(`--pp N` for the PP arm, `--pp R` for a PP-R reference) over the committed
prompts, and their records are compared byte for byte; each PP arm's `meta.json`
must record the requested `pp` and placement. The speed cells run `hipfire bench
--spec off` with `HIPFIRE_TOPOLOGY_PP=N` and check that the effective degree is
`N` (decode tok/s and prefill tok/s at the gate's cell sizes, fresh processes per
repetition). The receipt has the EP gate's structure plus `pp`, the bands, the
transport, each rank's PCI id, arch and PCIe link speed and width
(`/sys/bus/pci/devices/*/current_link_{speed,width}`) and the host RAM.

### Dense Qwen3.5/3.6 TP: native MTP

Dense Qwen3.5/3.6 at `tp>1` can speculate with MTP, **opt-in**: only an explicit
`speculation = "mtp"` (`speculation.mtp = on`, `--spec mtp`) takes the spec route.
The loader then replicates the MTP head on rank 0 (bundled `.mq4-mtp` trailer, the
CLI-resolved `.mtp` sidecar, or the trunk's sibling `.mtp`); the request drafts on
rank 0 and verifies across ranks. A missing head is a load error, as `mtp=on` is
on one GPU. `auto` (the default) and `off` keep the AR mesh loop even when a head
sits beside the trunk, so existing `tp>1` serves are unchanged. Block-verify
drafters (n-gram, DFlash, DSpark) stay refused on dense TP (an `auto`-paired DFlash draft is dropped with a log line at `tp>1` by `serve`/`bench`; explicit `on` is still refused), and the n-gram-mod
composition inside MTP (`HIPFIRE_MTP_NGRAM`) drafts natively there.

Dense-TP MTP is not a byte-identical replacement for TP AR: speculative
verification runs a multi-row trunk geometry, whereas AR advances one token at
a time. The H2 greedy gate observed different output in some workloads, so a
throughput improvement alone does not qualify this route for automatic use.

### Dense TP2 reduction protocol (developer)

Dense Qwen3.5/3.6 TP decode reduces each layer's attention and FFN partials
into the residual through `Gpus::all_reduce_sum_f32_peer_add_routed`
(`hipfire-runtime`). Every route evaluates the same rank-ordered sum, so the
output is the same bytes on each route: with a 16-bit exchange
(next section; `auto` picks it for multi-row and TP>=3 reductions) `x + (r(p0) + r(p1))`, with `HIPFIRE_TP_EXCHANGE=f32`
`x + (p0 + p1)`. With two
peer-access ranks the default is a symmetric direct-peer read; each reduction
waits on the peer's producer event and, before the next layer overwrites a
partial, on the peer's consumer-complete event.

Two developer controls, both tri-state (`1` forces on, `0` forces off, unset
takes the default), change only the synchronization. On an exact homogeneous
gfx1100 or gfx1201 pair the whole-token graph (and with it the alternating
protocol) is on by default; on every other topology both default off:

- `HIPFIRE_TP_PEER_ALTERNATING`: the attention partial and the FFN partial
  live in different buffers. Once a peer has produced the next partial, it
  has finished reading the previous one, so the reverse wait is dropped.
  Unset follows `HIPFIRE_TP_FULL_GRAPH`.
- `HIPFIRE_TP_FULL_GRAPH`: exact homogeneous gfx1100 pair or exact
  homogeneous gfx1201 pair, TP=2, peer access, plain AR decode (a mixed
  gfx1100/gfx1201 pair, any other arch, and any other rank count are not
  admitted). Each rank's whole decode token is captured as one hipGraph.
  Every reduction becomes one graph-resident kernel: a device-resident signal
  barrier plus the residual add, which reads the peer's partial through a
  host-mapped, system-coherent exchange. Captured peer-VRAM reads can be stale
  on gfx1100, and the same exchange is used on gfx1201. The loader prepares
  the signal tape and the exchange (about 80 KiB for Qwen3.8-27B). Enabling
  this control implies the alternating protocol unless
  `HIPFIRE_TP_PEER_ALTERNATING=0`.
- `HIPFIRE_TP_PEER_PUSH` (tri-state): inside the whole-token graph each rank
  pushes its partial into the other ranks' fine-grained VRAM mailboxes as
  16-byte tagged granules and polls its own mailbox, replacing the host
  exchange (TP2) or the pack/copy/fold/event route (TP4) with one kernel per
  rank and no fence. On a gfx1201 pair the f32 variant is bit-identical and
  drops the in-graph collective chain from 9.2 to 4.3 us, but 27B decode gains
  only 0.6%, so it is opt-in there. On four gfx1201 ranks it is the default:
  the bf16 variant is what lets the whole-token graph capture a TP4 decode
  token (next paragraph).

On four exact gfx1201 ranks on the generic mesh route the whole-token graph is
also the default (`HIPFIRE_TP_FULL_GRAPH=0` or `HIPFIRE_TP_PEER_PUSH=0` restore
the eager event route). The bf16 exchange is unchanged (`x + (r(p0) + r(p1) +
r(p2) + r(p3))`, rank order, fp32 adds), computed by one `tp_peer_pushn`
kernel per rank per collective: the rank packs its partial to bf16, writes it
to its slot of every peer's mailbox bank, waits (bounded) for the three other
slots in its own mailbox, and folds. Two banks alternate with a device-resident
per-rank generation, so a replayed graph needs no host reset; a rank that saw
all peers' granules for generation g knows each peer finished g-1. The load
qualifies the mailboxes with 64 eager exchanges compared bit for bit with a
host replay. The pushes go to one peer at a time with a fence and barrier between
destinations (concurrent write-through stores to several peers collapse to
about 1.5 GB/s per sender). Decode of Qwen3.8-27B mq4-xts on 4x R9700:
54.0 -> 77.5 tok/s (+43%).

On exact homogeneous gfx1100 and gfx1201 pairs the whole-token graph is the
default; `HIPFIRE_TP_FULL_GRAPH=0` restores the segmented-graph route with
event-synchronized reductions.

The gfx1100 promotion receipt was measured 2026-10-16 on 2x W7900, current
Qwen3.8-27B mq4-xts (SHA-256 `3e38ccba…`), Q8 VMM KV, pure AR, greedy, 512
generated tokens, three fresh processes per arm in `A,B,B,A,A,B` order:

| gfx1100 route | decode samples (tok/s) | median | delta |
|---|---|---:|---:|
| segmented (`HIPFIRE_TP_FULL_GRAPH=0`) | 48.0, 47.8, 47.8 | 47.8 | — |
| whole-token graph | 53.4, 53.2, 53.1 | 53.2 | +11.30% |

The path oracle passed 100/100 exact replays. Six product-path prompts produced
3,072 tokens per arm with byte-identical transcripts (MD5
`84410b3e45e9ddbb506c33f5d12526de`). See the dated
[performance checkpoint](perf-checkpoints/2026-10-16-gfx1100-tp2-whole-token-graph.md)
for fixture and binary identities.

The gfx1201 receipt was measured 2026-10-15 on 2x R9700 (PCIe 5.0 x16), Qwen3.8-27B
mq4-xts (sha256 `3e38ccba…`), greedy decode of 64 tokens after a 127- or
8191-token prefill, 3 fresh processes per arm, median ms/token after the
first token (fixture-bound, not a floor):

| ctx | 1 card (default FP8 KV) | TP2, `HIPFIRE_TP_FULL_GRAPH=0` (Q8 KV) | TP2 default, whole-token graph (Q8 KV) |
|---|---|---|---|
| 128 | 29.86 | 19.99 | 18.55 |
| 8192 | 26.82 | 20.59 | 18.97 |

Rank-0 logits were byte-identical between the two TP2 routes on every
token. The first decode token after load runs direct and the second captures
(about 25 ms). Validate either pair with the lab oracle
`tp2_cross_device_graph_reduce` (`--features lab`,
`HIPFIRE_TP2_GRAPH_REPLAYS`) and with greedy transcript identity against
`HIPFIRE_TP_FULL_GRAPH=0`.

### Dense TP partial exchange: 16-bit (developer)

The row-parallel stages (attention out-projection, FFN down-projection) leave
one f32 partial per rank, and the all-reduce moves those partials between
ranks. The residual stream is f32 and replicated. `HIPFIRE_TP_EXCHANGE`
selects the format the ranks exchange:

| value | exchange |
|---|---|
| unset / `auto` | the default: `bf16` for every multi-row reduction (prefill, speculative verify) at any TP degree and for single-row decode at TP>=3; `f32` for single-row decode at TP2, the one measured loss (bf16 there: -1.4% decode). One pure function, `tp_exchange::auto_exchange(rows, tp)` |
| `bf16` | forced everywhere: each rank converts its f32 partial to bf16 once (round to nearest even); every rank sums the rounded, rank-ordered partials in f32 into its f32 residual |
| `f16` | the same with IEEE half (three more mantissa bits, but a partial above 65,504 in magnitude becomes infinity and poisons the residual: overflow is not guarded) |
| `f32` | forced everywhere: the f32 exchange, rank-ordered f32 sums of the f32 partials, as before |

Only the exchange is 16-bit; the residual stream, the sums and the rooted
route's broadcast of the finished f32 residual stay f32. Every rank, the
producer included, adds the rounded value of every partial (`r(p) =
up(pack(p))`) in the same order, so the replicated residual stays bit-identical
across ranks and the hand and the generic dense TP routes, which reduce
through the same `Gpus::all_reduce_sum_f32_peer_add_routed` (hand route, and
the generic route's TP2) or the executor's `peer_reduce` (generic TP>2), stay
byte-identical to each other (the `tp_prefill_decode_matches_hand_route`
hardware test). The result is **not** byte-identical to the f32 exchange: the
16-bit route is a numeric route, judged by KLD against the f32 exchange and
the single-card pins, not by byte identity.

The 16-bit route covers every peer-access reduction between distinct devices
with 2 to 4 ranks and a row count x dim that is a multiple of 8:

- **TP2 direct-peer read**: each rank packs its partial into its own slot of a
  16-bit buffer (`tp_pack16_*`), records the producer event, waits the peer's,
  and `tp_fold16_*` adds `r(p0) + r(p1)`, reading the peer's slot through its
  P2P mapping, into the f32 residual. The alternating-slot protocol keeps one
  slot per partial buffer.
- **TP2 whole-token graph**: `tp_graph_signal_reduce16_*_gfx1100` publishes the
  16-bit copy into the peer's coherent shadow (half the bytes of the f32
  exchange), then reduces as above. The shadow segments the loader sizes for
  f32 hold either format.
- **TP>2**: the rooted fold gathers the peers' 16-bit partials (the root folds
  its own f32 partial rounded in register), adds the sum to its residual and
  broadcasts the f32 residual; the generic executor's small-payload all-pull
  pulls the 16-bit slots and every rank folds them itself.

The host-staged transport, same-device (`HIPFIRE_EMULATE_GPUS`) ranks, rank
counts above 4, the no-target `AllReduce` and the expert-parallel reductions
keep the f32 exchange. The dense TP load log reports the choice
(`[loader] dense qwen TP exchange=bf16 (HIPFIRE_TP_EXCHANGE)`).

Measured 2026-10-27 on hiptrx (R9700, gfx1201), Qwen3.8-27B mq4-xts (sha256
`3e38ccba…`), `eval_hipfire --scoring-mode prefill --max-chunks 24 --tp N
--kv-mode q8 --kv-v q8` (the hand dense-TP prefill with per-token hidden
capture; `HIPFIRE_GRAPH=0`, fresh HOME and kernel cache per run), 24 chunks,
mean KLD and the paired per-chunk difference against the f32 exchange of the
same TP degree (95% t interval):

| corpus | TP | f32 | bf16 | bf16 - f32 | f16 | f16 - f32 |
|---|---|---:|---:|---|---:|---|
| WT2 | 2 | 0.061277 | 0.059057 | -0.002220 [-0.004432, -0.000007] | 0.059953 | -0.001323 [-0.002996, +0.000350] |
| code24 | 2 | 0.045400 | 0.045685 | +0.000286 [-0.002197, +0.002768] | 0.043671 | -0.001728 [-0.005858, +0.002401] |
| WT2 | 4 | 0.059522 | 0.060439 | +0.000917 [-0.000328, +0.002161] | 0.059898 | +0.000375 [-0.001609, +0.002360] |
| code24 | 4 | 0.042194 | 0.044125 | +0.001931 [-0.002018, +0.005880] | 0.044575 | +0.002381 [-0.001549, +0.006311] |

Both 16-bit formats sit inside the f32 exchange's noise at TP2 and TP4. The
single-card pins (gfx1201 A4, `--kv-mode fp8 --kv-v q8`: WT2 0.069571, code24
0.052119) reproduce in the same harness. The TP route's own KLD is lower than
the single-card pin for every exchange, f32 included (the cause is not
established here), so the pin comparison does not bound the exchange error:
the judged quantity is the paired difference above, and none of the 16-bit
rows is worse than the single-card pin. bf16 is the default because it has the
f32 range, so no partial can overflow, whereas f16 turns a partial above
65,504 into infinity. A probe of the f32 partials over the same two TP2 runs
(6,144 reductions, both ranks, prefill rows) found a largest magnitude of
436.5 on WT2 and 412.9 on code24 and none above 1,000, so f16 has more than
100x headroom on these corpora; that is a measurement, not a bound for other
prompts.

### Generic mesh executor: TP collectives (developer)

`Gpus::provision_mesh_edges` admits pure PP meshes (TP = EP = 1) and pure TP
meshes (PP = EP = 1). TP×PP and any EP axis are refused before any mesh state
is written. For a pure TP mesh the executor runs the splitter's `AllReduceAdd`
and `AllReduce` collectives over the group of every rank in rank order, with
the hand dense-TP routes' numeric contract: `x + (p0 + p1 + …)` and
`p0 + p1 + …` (with the default 16-bit exchange the `AllReduceAdd` partials are
the rounded `r(p)` of the previous section; the no-target `AllReduce` stays
f32), the same bytes on every rank. The transport is chosen once at
provisioning, and each transport has exactly one route (no per-call choice;
`peer_access_enabled`, `HIPFIRE_TP_PEER_DIRECT` and `hardware.tp_use_rccl` do
not apply to it):

- **Peer**: every rank pair is one device under `HIPFIRE_EMULATE_GPUS` or
  passes `qualify_peer`. Each rank pulls every partial with the same kernel
  copy that `qualify_peer` byte-checks (never `hipMemcpyPeer`, which writes
  nothing into allocations made after peer enable) and folds the sum locally.
  Producer and read events order the cross-rank reads; nothing waits on the
  host.
  A TP2 `AllReduceAdd` between two distinct devices instead uses the hand
  route's direct-peer add kernel (`add_peer_residual_f32`, one launch per
  rank); other shapes (TP>2, `AllReduce`, same-device emulation) use the
  copy-fold.
  For TP>2 payloads above 256 KiB (prefill) the copy-fold is rooted instead:
  rank 0 gathers and folds in rank order, then each other rank pulls the
  finished sum (`2(N-1)` copies, identical bytes).
- **Host**: some pair is not peer-qualified. The hand route's host-staged
  rank-order reduction (`Gpus::all_reduce_sum_f32_host`), then `x + sum` on
  each rank. It synchronizes the rank streams. RCCL is never used, as in the
  hand dense-TP route (`qwen35` `dense_tp_all_reduce_sum_f32`).

Scratch, events, host staging and kernels are prepared at provisioning. The
whole epoch is validated before the first enqueue (group = every rank
ascending, one occurrence per rank, F32 descriptors equal on every rank,
16-byte aligned, no overlapping operands or targets, the replaced
`Accumulate` not scheduled), and a failed drain quarantines the mesh exactly
as for PP edges. Family bindings return the operand as `BoundaryIo::view`, the
`AllReduceAdd` target as `BoundaryIo::into`, and `Gpus::mesh_tp_epoch()` as
the descriptor epoch. An `AllReduceAdd` operand's contents are unspecified
afterwards. A boundary's memory attributes (plain local device memory of its
owner) are validated once per submission, in the preparation pass; the
`reduce`/`transfer` re-check only compares the binding's identity (addresses,
extents, descriptor, transport epoch, slot) and makes no HIP call.

Oracle: byte identity against the single-device rank-order sum over three
undrained epochs with distinct partials, for the provisioned transport and the
forced host transport, plus a TP×EP refusal. Every oracle buffer is allocated
after provisioning has enabled peer access. On one device:

```sh
cargo test -p hipfire-runtime --lib --no-run   # prints the unittests binary
for n in 2 4; do HIPFIRE_EMULATE_GPUS=$n <binary> \
  multi_gpu::executor::tests::tp_collectives_match_single_device_rank_order_sum \
  --ignored --nocapture --test-threads=1; done
```

Physical runbook (gfx1201, not run on a single-device box). Run the same
binary without `HIPFIRE_EMULATE_GPUS`: every visible device is one rank.

1. **Peer reads into post-enable allocations (the arm that guards the Peer
   transport).** `HIP_VISIBLE_DEVICES` = two, then four peer-capable cards.
   Pass: the `provisioned transport Peer` line (not `Host`) and the `host`
   line both report byte identity on every rank. Provisioning enables peer
   access before the oracle allocates its activations, so a copy mechanism
   that does not reach post-enable allocations fails the `x`/`n` comparison
   (the partials differ per epoch, so a stale read fails too). A `Host`
   provisioned line on cards expected to be peer-capable means `qualify_peer`
   refused the pair; record it.
2. **Optional reference.** The same arm with the binary from the first PR head
   (`99853a2db`, rooted `hipMemcpyPeerAsync` route) shows whether the
   `hipMemcpyPeer` trap reproduces on that ROCm; arm 1 must pass either way.
3. **Host transport.** Arm 1's `host` line already runs it on the same cards.
   If the box has a pair without peer access, run that pair too: its
   provisioned line must read `Host`.

Record per arm the `TP<n> oracle` lines (FNV-64 digests of `x` and `n`), the
binary MD5, `HIP_VISIBLE_DEVICES` and the ROCm version.

### Dense FFN tensor parallelism: `GatedMlp` split (developer)

`SplitRule::GatedMlp` splits a dense gated MLP along its intermediate: gate
and up `ColumnShard { axis: 0 }`, down `RowShard { axis: 1 }`. Every rank
runs the op's kernel ladder unchanged on its `intermediate / parts` slice
with the input whole (the pre-norm needs complete rows), and the output is a
partial sum that its consumer reduces: an `Accumulate` fuses one
`AllReduceAdd`, any other whole reader takes an `AllReduce`. All-`Replicate`
weights, or a single TP part (a PP stage), keep the op whole and local.
Admission refuses, without allocating, a mixed policy, gate/up/down that do
not share one intermediate, `intermediate % (parts * group)` on the down
weight, and a ladder that has not opted in (`MlpLadder::tp_shardable`). Every
ladder admits its `Write` form; the sandwich ladders (`SandwichWide`,
`SandwichSteps`) split before their post-norm epilogue, which cannot take a
partial sum, so a TP route binds the epilogue as a separate whole-row op
after the reduction.

The first end-to-end use is the Qwen4 dense developer profile: each rank
holds the whole model with only its BF16 FFN slice
(`Qwen4Weights::shard_dense_ffn`, which records `(rank, parts)`),
`qwen4_tp_plan` reduces each layer's FFN output before its `HyperWrite`
(and refuses a packed FFN), and `Qwen4TpBindings` runs a rank's program in
one segment per collective after checking the plan is that of exactly its
shards. A rank's FFN hands its F32 partial to the reduction
(`MlpStorage::Bf16SourcePartial`); the HC write rounds the sum once, as the
unsharded FFN output is rounded once. Everything except the FFN is
replicated. Sessions, checkpoints and prefix repair are not admitted on TP
yet, and the loader has no Qwen4 TP route. Oracle (tokens and logits
byte-identical to a single-device run of the same rank shards that sums the
FFN partials in rank order, and reported against the plain single card):

```sh
cargo test --release -p hipfire-arch-qwen4 --features qwen4-dev-profiles --lib --no-run
for n in 2 4; do HIPFIRE_EMULATE_GPUS=$n HIPFIRE_QWEN4_DEV_ARTIFACT=<tiny-dense-dev .hfq> \
  <binary> mesh_tp::hw_tests --ignored --nocapture --test-threads=1; done
```

The second use is Qwen3.5/3.6/3.8 dense prefill and decode (`qwen35::mesh_tp`,
#927 item 4), with the hand dense-TP route's work split. Each rank holds an
even contiguous block of the query, KV, DeltaNet key and value heads and of
the FFN intermediate (`tp_rank_layouts`, loaded through
`load_weights_dense_tp_rank`; uneven TP3/TP5 layouts are refused). Each
layer's mixer (`SplitRule::HeadMixer`: whole input, head-split input
projections and state, `RowShard { axis: 1 }` output projection) and its
`GatedMlp` each write the rank's partial, and the residual `Accumulate` after
each becomes one rank-ordered `AllReduceAdd` into `x`: two reductions per
layer, as in the hand route. A decode epoch is one token; a prefill epoch is
up to the rank's batch-scratch rows (the provisioned `max_bytes` must hold
`rows * dim * 4`). `Qwen35TpBindings` is checked against the plan of exactly
its shards (ranks come only from `Qwen35TpRank::load`, which loads each
rank's slice with the layout it records); a prefill past the batch scratch,
sessions, checkpoints and prefix repair are refused before any launch.
Prefix repair is refused through the host-only
`MeshStepBindings::admits_prefix_repair` capability, checked before the
first callback, so the mesh stays healthy. The loader reaches this route by
default (`HIPFIRE_DENSE_TP_ROUTE=auto`, below); `hand` is the rollback to the
hand route (see `docs/design/qwen35-declarative-program.md`,
"What stays on the hand route").
Oracle (`HIPFIRE_QWEN35_TP_ROWS` prompt rows, default 16, as prefill epochs
of at most `HIPFIRE_QWEN35_TP_CHUNK` rows, default 16; then 16 tokens
teacher-forced to the plain card's greedy tokens, the first and the ninth as
1-row Prefill epochs; byte identity against the hand dense-TP route over the
same shards and epochs, reported against the plain card). The 1-row rows are
`ROWS=1025 CHUNK=64` and `ROWS=577 CHUNK=96`, which end the prompt in a
1-row epoch:

```sh
cargo test --release -p hipfire-arch-qwen35 --lib --no-run
for n in 2 4; do HIPFIRE_EMULATE_GPUS=$n HIPFIRE_QWEN35_TP_MODEL=<dense .mq4> \
  <binary> qwen35::mesh_tp::hw_tests --ignored --nocapture --test-threads=1; done
```

Physical TP2 runbook (gfx1201, e.g. hiptrx; not runnable on a one-card box).
Fixtures: `qwen3.5-9b.mq4` (sha256 `ba83acf5…28ad`, 5313750016 B) and
`qwen3.8-27b.mq4-xt` (`9f91556f…eeb7`, 14980361216 B). Without
`HIPFIRE_EMULATE_GPUS`, the rank count comes from `HIPFIRE_QWEN35_TP_PARTS`
and rank `i` runs on visible device `i` for both the hand route and the
generic route; the plain card runs on device 0.

```sh
for model in qwen3.5-9b.mq4 qwen3.8-27b.mq4-xt; do
  HIP_VISIBLE_DEVICES=0,1 HIPFIRE_QWEN35_TP_PARTS=2 HIPFIRE_QWEN35_TP_MODEL=<models>/$model \
    <binary> qwen35::mesh_tp::hw_tests --ignored --nocapture --test-threads=1
done
```

Pass: each `TP2 <model>` line shows the hand and TP FNV-64 equal (the test
asserts it) on the transport `provision_mesh_edges` chose (Peer on
peer-qualified cards, else Host). The hand route reduces through its own
peer-rooted or host-staged route; a mismatch here that the emulated arm does
not show localizes to the two transports. Record per model the two `TP2` lines, the
binary MD5, `HIP_VISIBLE_DEVICES`, the ROCm version and, from the gfx1201
emulated arm (`HIPFIRE_EMULATE_GPUS=2`, one card), the same lines for a
same-box reference. Repeat with `0,1,2,3` and `HIPFIRE_QWEN35_TP_PARTS=4`.
On a TP2 pair the runtime admits for the whole-token graph (both arms prepare
it as the loaders do) the test also prints `whole-token graph admitted: true`
and each rank's `whole-token graph epochs` counts, and asserts at least one
capture and two replays among the 14 decode epochs; where the graph is not
admitted (TP4) it asserts that decode ran no graphs.

Loader route (developer): `HIPFIRE_DENSE_TP_ROUTE` picks the route
`hipfire serve --tp N` loads for a dense Qwen3.5/3.6/3.8 model. `auto` (the
default; also unset or empty) loads the generic mesh route whenever it admits
the load, else the hand route (`load_model_tp_qwen35_dense`,
`EpArch::Qwen35DenseTp`) with the log line
`[loader] dense qwen TP route=auto -> hand: <reason>`; the reasons are
`speculation.mtp = on` (the mesh route has no MTP verify) and an uneven
head/FFN split (TP3/TP5). `mesh` forces the mesh route and refuses MTP and
uneven splits; `hand` is the rollback to the hand route. Any other value is a
load error naming the accepted values (`parse_dense_tp_route` /
`resolve_dense_tp_route`, `crates/hipfire-loader/src/lib.rs`). The mesh load
builds one `Qwen35TpRank` per device (`EpArch::Qwen35MeshTp`),
prepares the TP2 whole-token graph exactly as the hand loader does (signal
tape and payload exchange before peer access is enabled; a no-op unless the
runtime admits the graph), provisions the reduction with
`provision_mesh_edges` (peer when every rank pair passes byte qualification,
else the host-staged reduction), calls `enable_peer_all` once the ranks'
buffers exist when the transport is peer, and builds the program and plan
once (`Qwen35TpProgram`; the load is refused unless the plan holds two
collectives per layer, the signal tape's barriers). Each request epoch runs
through `Qwen35TpProgram::run_epoch`. It serves the hand route's loop:
chunked prefill (the rank's batch scratch rows, the hand route's prefill
chunk) and single-token AR decode, on every KV backend (the VMM default
included).
DFlash and n-gram drafters are not built, and sessions, checkpoints and
prefix repair are refused by the bindings. The load prints
`[loader] dense qwen TP route=mesh (generic mesh executor): <tp> ranks,
transport=<peer|host>, prefill chunk=<rows>, <n> collectives/epoch`; the
hand route prints `[loader] dense qwen TP route=hand ...`. A failed epoch
poisons the mesh (every later epoch is refused) until the model is reloaded.
`serve` only enqueues mesh epochs (`run_epoch` returns the executor ticket) and
drains them together at the rank-0 logits read, like the hand route's
enqueue-then-download, so the first token no longer waits on a drained epoch.
The mesh route is byte-identical to the hand route at TP2 and TP4 (physical
hardware test, logits fnv64 equal).

Measured on a 4x gfx1201 (R9700) host, Qwen3.8-27B mq4-xts,
`hipfire bench --matrix --spec off --runs 3`, 3 fresh processes per arm,
median tok/s:

| Case | Mesh | Hand | Single card |
|---|---:|---:|---:|
| TP2 pp512 | 3289.5 | 2873.5 | 4892.6 |
| TP2 pp2048 | 3355.2 | 2753.8 | 5251.9 |
| TP2 tg64@128 | 54.67 | 54.94 | 41.36 |
| TP2 tg64@4096 | 53.99 | 54.27 | 40.78 |
| TP4 pp512 | 2061.0 | 1919.4 | - |
| TP4 pp2048 | 2113.4 | 1997.4 | - |
| TP4 tg64@128 | 51.77 | 43.96 | - |
| TP4 tg64@4096 | 51.28 | 43.56 | - |

Decode graphs on the mesh route: TP2 decode on an exact gfx1201 or gfx1100
pair uses the same whole-token graph as the hand route
(`HIPFIRE_TP_FULL_GRAPH`, on by default for those pairs). `run_epoch` runs each
rank's embed eagerly, captures the executor submission of the rest on both
ranks (the reductions record the signal-barrier kernels) and replays the
graphs on every later token, so a replayed token costs the host only the
epoch bookkeeping. The decode partials alternate two slots
(`BoundaryIo::partial_slot`: mixer 0, FFN 1), which selects the alternating
direct-peer reduction the graph can capture; prefill reuses one partial
buffer and takes the ordinary protocol. Any other decode (TP4, another pair,
`HIPFIRE_TP_FULL_GRAPH=0`, `HIPFIRE_GRAPH=0`, a KV eviction offset) decodes
eager through the executor. Per-segment hipGraphs (one per rank program
segment between the collectives) measured slower than eager at TP4 on
gfx1201 (45.99 vs 51.13 tok/s, Qwen3.8-27B) and were removed: TP2 uses the
whole-token graph, TP>2 decodes eager. `HIPFIRE_AR_GRAPH_TRACE=1` prints
`[qwen-mesh-tp-graph]` lines for whole-token captures and replays. Every mode
computes the same bytes.

The third use is Muse Glimmer (`muse_glimmer::mesh_tp`, #927 item 5). Both
Glimmer sublayers end in the sandwich epilogue `x + post_norm(out)`, and the
RMS post-norm cannot take a partial sum, so each sublayer is split at that
boundary: the attention (`SplitRule::HeadMixer`, `SandwichAttention`) writes
the rank's `o_proj` partial and the MLP (`SplitRule::GatedMlp`,
`MlpLadder::SandwichSteps`) its `down` partial, and the whole-row epilogue op
after each takes one rank-ordered `AllReduce` of the partial, then runs the
single-card post-norm residual kernels on the reduced rows
(`execute_sandwich_attention_partial`, `execute_sandwich_mlp_partial`,
`execute_sandwich_epilogue`; the single-card ops are unchanged). Each rank
holds an even block of the query and KV heads, the `o_proj` columns and the
FFN intermediate in whole G256 groups (`GlimmerWeights::load_shard`); the
embedding and LM head are replicated. `muse-glimmer-30b` (32 query / 2 KV
heads, intermediate 19968) splits over two ranks only: TP4 is refused by the
plan (`n_heads=2 not divisible by Tp=4`). The loader still serves Glimmer on
one device only (`muse_glimmer: single-device only`); this route is the
executor program and its oracle, with no loader TP route yet. Oracle
(`HIPFIRE_GLIMMER_TP_ROWS` prompt rows, default 16, as prefill epochs of at
most `HIPFIRE_GLIMMER_TP_CHUNK` rows, default 16; then
`HIPFIRE_GLIMMER_TP_GREEDY` tokens, default 480, teacher-forced to the plain
card's greedy tokens, the first and the ninth as 1-row Prefill epochs; byte
identity of the executor against the single-device rank-order oracle, which
runs every rank's program on one device and folds the partials
`p0 + p1 + …` with the executor's kernels; tokens and the maximum logit
difference reported against the plain card):

```sh
cargo test --release -p hipfire-arch-muse-glimmer --lib --no-run
HIPFIRE_EMULATE_GPUS=2 HIPFIRE_GLIMMER_TP_MODEL=<models>/muse-glimmer-30b.mq4 \
  <binary> mesh_tp::hw_tests --ignored --nocapture --test-threads=1
```

Physical Glimmer TP2 runbook (gfx1201, e.g. hiptrx). Fixture
`muse-glimmer-30b.mq4` (sha256 `87bc776a…c052`, 18614785024 B; about 10.7 GB
of weights per rank, the Q8 embedding and LM head replicated, so it fits two
16 GB cards; the plain card and the oracle arm load
the whole model on device 0, which needs a 32 GB card such as the R9700):

```sh
HIP_VISIBLE_DEVICES=0,1 HIPFIRE_GLIMMER_TP_PARTS=2 \
  HIPFIRE_GLIMMER_TP_MODEL=<models>/muse-glimmer-30b.mq4 \
  <binary> mesh_tp::hw_tests --ignored --nocapture --test-threads=1
```

Pass: the test asserts the executor rows bitwise equal to the oracle's; record
the `TP2 muse-glimmer-30b.mq4` line (oracle, TP and plain FNV-64, argmax
agreement, max logit difference), the binary MD5, `HIP_VISIBLE_DEVICES`, the
ROCm version and the transport `provision_mesh_edges` chose, plus the same
line from the gfx1201 emulated arm (`HIPFIRE_EMULATE_GPUS=2`, one card) as a
same-box reference.

### Architectural limits (current)

- Homogeneous **exact arch string** by default (`ALLOW_MIXED_ARCH` is opt-in).
- No automatic VRAM-weighted split.
- PP Qwen3.5 decode (`generate_multi`) emits through the same
  `QwenArSemanticProducer` as single-card AR: `<think>` spans stream as
  `reasoning` events, only answer text as `token`, and `finish_reason` and
  held tail text come from the shared finish. `stop` and tools are still
  refused on PP upstream.
- Generic PP decode is sequential across bands (no async multi-band pipeline /
  per-band graph capture as a documented product path); Flash-Next PP pipelines
  prefill chunks across its bands.
- `HIPFIRE_PP_PFLASH` / `HIPFIRE_PP_DFLASH` are experimental flags of the generic
  Qwen3.5 / 3.6 route. They are **not** admissions, are not route-certified
  performance features, and do not change what Flash-Next PP refuses.

## Memory budget

Weights, KV, and scratch land on the devices that own each band. Last device
also carries `output_norm + lm_head`. Per-card headroom depends on quant, KV
mode, `max_seq`, and split.

Flash-Next PP uses a per-band budget instead (all-resident bands, each card
checked against its own free VRAM before allocation); see
[Per-band memory budget](#per-band-memory-budget).

**Reproduce on your hardware** (2+ visible GPUs, deltanet feature):

```sh
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
  -p hipfire-runtime --example pp2_vram_probe -- \
  ~/.hipfire/models/qwen3.5-9b.mq4 4096
```

Historical VRAM and throughput tables from the original multi-GPU PP doc are
preserved **verbatim** in [Appendix A](#appendix-a-historical-pp-evidence-immutable).
They are **historical** only: they do not carry a complete measured-identity
manifest (measurement date + binary md5 + model identity on the same report),
so they must not be labeled **measured**, must not be treated as floors, and
must not be edited in place. Re-run the probe / perf protocol before claiming
fit or speed on other SKUs.

PP KV policy for Qwen3.5 multi-GPU defaults through `QWEN35_PP_POLICY` (see
`kv_mode.rs`) — do not assume single-GPU `auto` KV defaults apply unchanged:
PP and dense TP / MoE EP have no native constructor, so `auto` is `q8/q8` on
every arch there (gfx1201 included), and PP accepts only `q8`, `fwht2`,
`fwht3` and legacy `legacy-asym3` K.

## Measured: Flash-Next EP on 2× R9700 (2026-10-13)

> **Truth state: measured, fixture-bound** ([`INDEX.md`](INDEX.md)). These are
> observations on the named fixture, binaries and date below — **not** promises,
> floors, defaults or admissions. They were taken on the pre-merge branch build
> that became the Flash-Next S1/S2 entries of the 0.4.2 changelog, not re-run on
> the audited ref, and become **historical** the moment the code, the fixture
> or the cards change. Any other model, prompt length or card pair is
> unmeasured here.

| Field | Value |
|---|---|
| Date | 2026-10-13 |
| Hardware | 2× Radeon R9700 (gfx1201), EP2, peer access on |
| Model | Qwen3.8 Flash-Next MQ4 XTS, sha256 prefix `8b15b6fe` |
| Binaries (md5 prefixes) | prefill: `ade91299` daemon, `83c40bb1` CLI (S1+S2 build, flags forced on); default-on re-run: `6eeeed5f` daemon, `ac01b942` CLI; decode no-trade: `f7970c02` daemon, `2a9a2091` CLI |
| Single-card arm | same binary, `topology.ep = 1` |
| EP2 arm | S1 + S2 + balanced owners on (S1/S2 default on in the default-on re-run) |

### Prefill (`hipfire bench --spec off`)

Method: committed prompts (p8k 9,319 tokens, md5 prefix `c3ccdc43`; p16k 16,343
tokens, md5 prefix `68b768d0`), warm (`--runs 1 --warmups 1` per process), **3
fresh processes per arm**, arms interleaved, median of the three, prefill
tok/s.

| Prompt | Single card (3 runs) | Median | EP2 (3 runs) | Median | EP2 / single |
|---|---|---|---|---|---|
| p8k | 3030.1 / 3034.0 / 3032.5 | 3032.5 | 3942.3 / 3946.4 / 3942.6 | 3942.6 | 1.300 |
| p16k | 3754.7 / 3752.1 / 3753.9 | 3753.9 | 4301.6 / 4307.6 / 4308.4 | 4307.6 | 1.147 |

Every EP2 run was above the single-card median at both lengths. A
default-flags re-run at p8k (no `HIPFIRE_EP_*` set, three fresh processes per
arm, interleaved) gave single 3039.8 / 3037.4 / 3035.2 (median 3037.4) and EP2
3948.1 / 3941.6 / 3936.7 (median 3941.6), ratio 1.298. **EP4 prefill was not
measured** (S1 was refused on EP4 in the measured build); the changelog's earlier EP4 run measured a
42-token prompt at 408 ms to first token against 286 ms on one card.

### Decode: no trade for the prefill levers (EP2)

Method: `hipfire bench --runs 3 --warmups 2 --max-tokens 256`, greedy, 3 fresh
processes per arm, one round, EP2 with balanced owners only ("W1 only") against
S1 + S2 on ("combined"). Short outputs; decode tok/s, not end-to-end.

| Workload | W1 only (3 runs) | Median | Combined (3 runs) | Median | Change |
|---|---|---|---|---|---|
| MTP code | 84.0 / 84.0 / 85.3 | 84.0 | 85.1 / 84.1 / 83.6 | 84.1 | +0.1 % |
| MTP prose | 61.7 / 62.2 / 65.6 | 62.2 | 61.4 / 65.9 / 65.9 | 65.9 | +5.9 % (noise: the W1-only arm itself spans 61.7–65.6) |
| AR code | 50.4 / 50.4 / 50.5 | 50.4 | 50.2 / 50.3 / 50.3 | 50.3 | −0.2 % |

Resident routed layers per rank are identical in both arms (29 of 49 under MTP,
30 of 49 under AR). Read this as "S1/S2 did not slow decode on this fixture",
not as a speedup.

### Decode against one card (changelog-reported, EP work window 2026-10-11 to 2026-10-13)

Same pinned model, R9700s, greedy, committed prompts, 3 fresh processes per
arm, `hipfire bench`; the per-run receipts are held by the maintainers.

| Workload | One card | EP2 | EP4 |
|---|---|---|---|
| MTP code (`humaneval_0`, 134 tokens), tok/s | 44.4 | 81.7 | 121.6 |
| MTP prose (`fiction_lighthouse`, 44 tokens), tok/s | 34.5 | 63.4 | 87.1 |
| AR decode, tg128 `--spec off`, tok/s | 36.1 | — | 61.2 |

These are short outputs and decode tok/s, not end-to-end. The changelog line
does not record which `HIPFIRE_EP_*` defaults were in force for the EP2 rows,
and the AR row is one card against EP4 only.

## Validation routes

There is **no universal GPU gate** ([`VALIDATION.md`](VALIDATION.md)). Multi-GPU
claims are **manual** and path-specific. **`scripts/pp-gate.sh` is not named in
the VALIDATION claim→route selector** and therefore is **not** a canonical
minimum route ([`VALIDATION.md`](VALIDATION.md) § retired / unnamed gate
scripts). Treat it only as **supporting manual evidence**.

Map claims through the selector’s classes:

| Claim class (selector) | What applies for multi-GPU | Notes |
|---|---|---|
| Forward / fusion / KV **numerical or state parity** | Path-specific parity/state oracle when one exists for the surface | Supporting tool today: `pp_parity_chatml` example (also invoked by `pp-gate.sh`). If no oracle exists for a surface → **blocked**. Not `serve_harness.py`. |
| Forward / serve **user-facing semantics** | `scripts/serve_harness.py` with the exact model (after parity if numbers/state can break) | Semantics only |
| Perf improvement under PP or EP | [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) + stationary matched runs; `speed-gate.sh` / `gates.sh` perf arm when applicable | Bench numbers without protocol/identity are not promotion evidence. Historical appendix rows are not floors. |
| Arch port (new device family behavior) | [`methodology/arch-port-validation.md`](methodology/arch-port-validation.md) | Channel + speed; no retired coherence battery as acceptance |
| Model/route **admission** | Row in [`admissions.yml`](admissions.yml) | Schema v2 exact-row only — multi-GPU PP/EP are **not** admitted |
| Docs-only edits | No-GPU CI / `scripts/no-gpu-ci.sh` | Never substitutes for GPU parity |
| Unknown multi-device surface | **Blocked** until VALIDATION grows a row | Fail closed |

Supporting manual commands (not selector minimums):

```sh
# Supporting multi-GPU battery (parity + daemon e2e + refusals). Skips cleanly
# with <2 usable devices. Not automatic merge proof; not a VALIDATION minimum.
./scripts/pp-gate.sh

# Faster: parity example only
./scripts/pp-gate.sh --skip-end-to-end

# Topology filter dry-run (no GPU work)
./scripts/pp-gate.sh --dry-run

# Direct parity example
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
  -p hipfire-runtime --example pp_parity_chatml -- \
  ~/.hipfire/models/qwen3.5-0.8b.mq4
```

**pp-gate knobs** (see script header): `PP_GATE_DEVICES`,
`HIPFIRE_PP_GATE_INCLUDE_IGPU`, `HIPFIRE_PP_GATE_HETEROGENEOUS`,
`HIPFIRE_PP_GATE_MODEL`, `HIPFIRE_PP_GATE_REQUIRE_SYSFS`. Filters drop known APU
iGPUs and skip heterogeneous ISA families unless overridden.

Pre-commit (when hooks installed) runs pp-gate if staged paths match the
multi-GPU hotspot regex in [`.githooks/pre-commit`](../.githooks/pre-commit).
That is a **path-gated local guard**, not full product admission.

Retired `scripts/coherence-gate-*.sh` batteries are **not** acceptance for PP
([`VALIDATION.md`](VALIDATION.md) § retired).

## Related

- Env inventory: [`env-vars.md`](env-vars.md)
- Serve / EP CLI: [`SERVE.md`](SERVE.md), [`CLI.md`](CLI.md)
- Validation selector: [`VALIDATION.md`](VALIDATION.md)
- Historical bring-up: [`multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md)
- Issue tracker context: [#58](https://github.com/warpfront/hipfire/issues/58)

---

## Appendix A — Historical PP evidence (immutable)

> **Lifecycle — historical only.** The block below is the prior multi-GPU PP
> document body retained for provenance. It is **not** current procedure, **not**
> a product floor, and **not** an admission. Truth state: **historical** (not
> **measured** — the retained tables lack a complete same-report measurement
> date + binary identity + model identity manifest required by
> [`INDEX.md`](INDEX.md)). Do not edit the evidence body; amend only by adding
> new dated sections outside this appendix. Warnings in the active sections
> above supersede any stronger claim language inside the retained body
> (including “measured”, “Status: v1 feature-complete”, absolute speedup
> wording, and gate-as-acceptance framing).
>
> The same applies to the host-staging fallback wording under “Verifying peer
> access” and to the v1 limitation lists: the generic Qwen3.5 / 3.6 PP route is a
> peer-memcpy route that needs peer access, and only Flash-Next PP has an
> admitted host-staged edge service (see
> [Flash-Next pipeline parallelism](#flash-next-pipeline-parallelism-host-staged)).
>
> **What the appendix tables are.** The July 2026 tables below (PP=2 VRAM
> probe and PP=2 throughput baseline) were taken on 2× Radeon RX 7900 XTX
> (gfx1100) for Qwen3.5-family models, with **short prompts only** (15 to 322
> tokens), before expert parallelism for Flash-Next existed. They are
> **historical**, not evidence for any current route, and are superseded for
> Flash-Next by the dated [Measured](#measured-flash-next-ep-on-2x-r9700-2026-10-13)
> section, which is a different model, different cards and 8k to 16k prompts.

# Multi-GPU Pipeline-Parallel

**Status:** v1 feature-complete on `feat/multi-gpu-pp` branch — tracking
issue [#58](https://github.com/warpfront/hipfire/issues/58). Stages
0–9 of the v2 plan are merged; refusal contracts (DFlash / VL / CASK +
pp>1) are wired and validated. This doc is the source of truth for
memory budget, deployment recipes, throughput, and known limitations.

## Why PP

hipfire on a single 24 GB card hits VRAM walls on:

- 27B at `--max-ctx ≥ 16K` with `kv_mode=asym3` (`AGENTS.md:356`)
- 35B-A3B at `--max-ctx ≥ 4K` with FP32 KV
- hypothetical 80B-A3B at any context

Pipeline-Parallel (PP) shards layers across N devices. Each device owns a contiguous "band"
of consecutive layers. The residual stream `s.x` flows through the bands sequentially: dev_0
runs layers `0..k1`, copies `s.x` to dev_1, dev_1 runs layers `k1..k2`, and so on. Final
`output_norm + lm_head` run on the last device (dev_last) — its `s.logits` is read by the
sampler in place.

**What PP gives you on 2× 24 GB:**
- Run 27B / 35B-A3B that don't fit on one card with extended context
- Unlock max_ctx on 27B beyond single-GPU OOM limits
- ~50-70% of single-GPU throughput on already-fitting models (sequential PP=2 is slower per token)

**What PP does NOT give you:**
- Faster multi-user serving — that's TP (tensor parallel), separate roadmap
- Speedup on models that already fit on one card

## Memory budget (per-card, PP=2)

Numbers below are **measured** on 2× Radeon RX 7900 XTX (gfx1100, 25.8 GiB VRAM
each) via `crates/hipfire-runtime/examples/pp2_vram_probe.rs` —
`hipMemGetInfo` deltas captured at each allocation stage
(`load_weights_multi`, `Qwen35ScratchSet`,
`KvCache::new_gpu_asym3_capped_multi`, `DeltaNetState`). Per-card columns
report the worst-of-two (the device that holds more — typically dev_last,
which carries `output_norm + lm_head`). `total` is the sum across both cards.

| Model | quant | n_layers | dim | KV mode | ctx | weights | KV/card | scratch+DN/card | total | per-card max | fits 24 GiB? |
|-------|-------|----------|-----|---------|-----|---------|---------|-----------------|-------|--------------|--------------|
| qwen3.5:0.8b | mq4 | 24 | 1024 | asym3 | 4096 | 1.3 GB | 50 MB | 8 MB | 1.3 GB | 0.7 GB | yes |
| qwen3.5:4b | mq4 | 32 | 2560 | asym3 | 4096 | 4.0 GB | 134 MB | 15 MB | 4.0 GB | 2.0 GB | yes |
| qwen3.5:9b | mq4 | 32 | 4096 | asym3 | 4096 | 5.6 GB | 134 MB | 19 MB | 5.6 GB | 2.8 GB | yes |
| qwen3.5:9b | mq4 | 32 | 4096 | asym3 | 16K | 6.2 GB | 436 MB | 46 MB | 6.2 GB | 3.1 GB | yes |
| qwen3.5:9b | mq3 | 32 | 4096 | asym3 | 4096 | 4.4 GB | 134 MB | 19 MB | 4.4 GB | 2.2 GB | yes |
| qwen3.5:27b *(via 3.6 proxy)* | mq4 | 64 | 5120 | asym3 | 4096 | 15.5 GB | 268 MB | 42 MB | 15.5 GB | 7.8 GB | yes |
| qwen3.5:27b *(via 3.6 proxy)* | mq4 | 64 | 5120 | asym3 | 16K | 16.8 GB | 872 MB | 80 MB | 16.8 GB | 8.4 GB | yes |
| qwen3.5:27b | mq3 | 64 | 5120 | asym3 | 4096 | 12.6 GB | 268 MB | 40 MB | 12.6 GB | 6.3 GB | yes |
| qwen3.5:27b | mq3 | 64 | 5120 | asym3 | 16K | 13.8 GB | 872 MB | 78 MB | 13.8 GB | 6.9 GB | yes |
| qwen3.6:35b-a3b | mq4 | 40 | 2048 | asym3 | 4096 | 23.5 GB | 103 MB | 21 MB | 23.5 GB | 11.8 GB | yes |
| (hypothetical) 80B-a3b | mq4 | 80 | 8192 | asym3 | 4096 | ~42 GB | ~250 MB | ~2 GB | ~44 GB | ~22 GB | estimate, tight |

Notes:
- `qwen3.5:27b mq4` rows use `qwen3.6:27b mq4` as a measurement proxy (same
  `n_layers=64`, `dim=5120`, `head_dim=256`, `n_kv_heads=4` — VRAM-equivalent).
  When `qwen3.5:27b mq4` lands as a downloadable artifact the rows can be
  re-measured directly.
- `qwen3.6:35b-a3b mq4` is the current-generation A3B; `qwen3.5:35b-a3b mq4`
  ships local-only with the same MoE shape — measurement carries over.
- The 80B row stays an estimate — no public artifact exists.
- "scratch+DN/card" combines `Qwen35ScratchSet::per_device[i]` (residual
  stream, attention/FFN scratch, flash partials, logits) and the
  `DeltaNetState` slice owned by that device's LA-layer band.

**Asymmetry under Variant 2 (lm_head on dev_last):**
- dev_0 carries `token_embd`
- dev_last carries `output_norm + lm_head`
- Layer count shifts by ±1 for `n_layers % 2 != 0` (uniform split formula
  `base + (i < rem ? 1 : 0)`)

Probed on this hardware: per-card max < 24 GiB at every measured shape. The
A3B model at 11.8 GB/card has the largest headroom consumer; everything
else stays under 8.5 GB/card with 4 K context, under 9 GB/card at 16 K.

To reproduce on your hardware:

```sh
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
    -p hipfire-runtime --example pp2_vram_probe -- \
    ~/.hipfire/models/qwen3.5-9b.mq4 4096
```

## Deployment recipes

The daemon takes a `pp` field in the load message (default `1` =
single-GPU, identical behavior to pre-PP code paths):

```sh
# Filter to the two 7900 XTX (drop the iGPU)
HIP_VISIBLE_DEVICES=0,1 hipfire run qwen3.5:27b --max-ctx 16384 "Hi"

# Bypass the inherited `gemm_..._wmma_ksplit` non-determinism (k-split
# atomicAdd reduction varies by warp scheduling — see commit f54ca71
# and kernels/src/gemm_hfq4g256_residual_wmma_ksplit.hip:22). Required
# for byte-equivalent output across processes / pp configurations.
HIPFIRE_DETERMINISTIC=1 HIP_VISIBLE_DEVICES=0,1 hipfire run qwen3.5:9b "Hi"

# Override uniform-VRAM tolerance (default 2 GiB; arches must match)
HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=4 hipfire run qwen3.5:27b "Hi"
```

Direct daemon JSON (driving without the CLI):

```json
{"type":"load","model":".../qwen3.5-27b.mq4","params":{"max_seq":16384,"pp":2}}
```

### Environment variables

| Variable | Effect |
|----------|--------|
| `hardware.devices = "3,1"` | PCI-order cards 3 and 1 become logical devices 0 and 1; ROCr receives their UUIDs (or ROCr ordinals for no-UUID cards) |
| `HIPFIRE_DETERMINISTIC=1` | Force k2 WMMA reduction (no atomicAdd) — bit-identical across processes/pp configs at ~33% perf cost on small-batch decode |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=N` | Pre-flight VRAM-asymmetry tolerance for `Gpus::init_uniform` (default 2.0) |
| `HIPFIRE_PREFILL_BATCHED=0` | Disable batched WMMA prefill (per-token fallback). Diagnostic for ksplit non-det isolation |
| `HIPFIRE_PREFILL_MAX_BATCH=N` | Override per-chunk prefill batch. When unset/invalid: arch defaults are **512** on exact `gfx1100`, **384** on exact `gfx1201`, else **256** (`PREFILL_MAX_BATCH`). Under TP, `prefill_max_batch_tp` uses default×`tp` (cap **2048**). An explicit `HIPFIRE_PREFILL_MAX_BATCH` wins over both the arch default and the TP scale. Chunks > N split with peer-copy at the boundary |
| `HIPFIRE_WO_WMMA_VARIANT={k2,ksplit,k4,…}` | Manual override of the wo-residual GEMM variant — see `dispatch.rs` auto-dispatch |

### Refusal matrix at load (`pp > 1`)

| Feature | Behavior | Why |
|---------|----------|-----|
| arch_id ∈ {5, 6} (Qwen3.5 dense + MoE/A3B) | Accepted | Validated end-to-end |
| arch_id = others (LLaMA / Qwen3) | Refused | Single-GPU only in v1 |
| VL models (vision_config + vision tensors) | Refused | v1.1 |
| DFlash draft (`draft` field set) | Refused | v1.1 — see `feedback_cask_mfold_dflash_broken.md` for the v1 ship-blocker |
| CASK / TriAttention sidecar | Refused | Eviction context is single-device — v1.1 |

## Throughput baseline (gfx1100 × 2)

Measured on 2× Radeon RX 7900 XTX, ROCm 6.4.3, with
`HIPFIRE_DETERMINISTIC=1` (bit-equivalent pp=1 ↔ pp=2 output).

| Model | Prompt | pp=1 prefill | pp=2 prefill | pp=1 decode | pp=2 decode | pp=2/pp=1 decode |
|-------|--------|--------------|--------------|-------------|-------------|------------------|
| 0.8B mq4 | 22 tok | 838 tok/s | 588 tok/s | 332 tok/s | 227 tok/s | 68% |
| 0.8B mq4 | 322 tok (chunked) | 6493 tok/s | 5490 tok/s | 315 tok/s | 212 tok/s | 67% |
| 35B-A3B mq4 (MoE) | 15 tok | 331 tok/s | 258 tok/s | 142 tok/s | 97 tok/s | 68% |

The pp=2 decode penalty is inherent to v1: per-token
`forward_scratch_multi` pays one HIP launch per kernel per layer with
no graph capture (vs pp=1 which captures + replays the AR-step graph
after warmup). Pipelined decode + per-band graph capture lift this in
v1.1.

## Limitations (v1)

Refused at load time:
- `pp > 1` + DFlash speculative decode
- `pp > 1` + CASK/TriAttention sidecar (eviction is single-device)
- `pp > 1` + VL models (vision encoder is single-device)
- `pp > 1` + arch_id ∉ {5, 6} (LLaMA / Qwen3 dense are pp=1 only)

Architectural limits in v1:
- Homogeneous arch only (`init_uniform` hard-fails on arch mismatch)
- Uniform layer split — `init_layers(per_device)` is the manual escape hatch
- Per-token decode (no async stream pipeline / per-band graph capture) — v1.1
- Pipelined prefill (chunk N+1 on dev_0 while chunk N processes on dev_1) — v1.1

## Validation (Stage 9)

```sh
# Multi-GPU gate. Skips silently when fewer than 2 GPU visible.
./scripts/pp-gate.sh

# Just the parity smoke (no daemon end-to-end), faster
./scripts/pp-gate.sh --skip-end-to-end

# Underlying byte-equivalence example
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
    -p hipfire-runtime --example pp_parity_chatml -- \
    ~/.hipfire/models/qwen3.5-0.8b.mq4
```

The `pp-gate.sh` battery checks:
1. Per-token `forward_scratch_multi` ≡ `forward_scratch` bit-exact (pp_parity_chatml)
2. Daemon `pp=1` ≡ `pp=2` byte-identical with `HIPFIRE_DETERMINISTIC=1` (greedy ChatML)
3. DFlash + pp=2 refusal at load
4. CASK + pp=2 refusal at load

A pre-commit hook calls `pp-gate.sh` automatically when staged files
match the `multi_gpu|pp_|peer_access|pipeline|stages` hotspot regex.

## Verifying peer access on your hardware

```sh
rocm-smi --showtopo            # weights, hops, link types
rocm-smi --showtoponuma        # NUMA placement
rocm-smi --showtopoaccess      # peer accessibility matrix
```

A `True` in every cell of `--showtopoaccess` for the cards you plan to use means peer-access
should work. If not, hipfire falls back to host-staging via pinned buffers (slower but correct).

## Open questions (will be filled in as Stages land)

- DFlash + PP integration scope — pending maintainer guidance on issue #58
- Whether mixed-arch should be soft-warn or hard-fail — currently hard-fail
- `hardware.devices` is the physical device source of truth; see [Device selection](#device-selection)
