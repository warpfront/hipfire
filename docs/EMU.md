# SKU emulation: application capacity and derating estimates

Developer-only and opt-in. Off unless `HIPFIRE_EMU` (or `hipfire bench --emu`)
names a profile; with it unset every allocation, output and JSON line is
unchanged.

## What it is

Two separate things, never blended into one number:

1. **An application-capacity profile.** `HIPFIRE_EMU=<id>` makes one hipfire
   process behave, for memory, like a smaller card (or an APU laptop scenario):
   the managed application allocation (device buffers, pinned host buffers, VMM
   backing, ROCr kernargs) is charged to a process-local ledger *before* the raw
   allocation, an over-cap request fails as an ordinary out-of-memory without
   reaching the driver, and every sizing and admission query reports the capped
   total and free. The host keeps executing at its own speed on all of its CUs
   with its own gfx. This is "R9700 execution at 16-GiB application capacity",
   never "measured RX 9070 XT tok/s".
2. **A measured-input derating model** (`hipfire emu estimate`). It takes
   measured kernel observations and host/target resource rates and *reports* an
   estimated wall time for a target. It never rewrites a measured figure, never
   injects waits and never starts a daemon.

The registry (`hipfire emu list`) is the single authoritative profile table. The
frozen design (EMU-DESIGN.md §1–§8) is kept with the SKU-emulation work, not in
this tree.

## CU-count emulation is NOT available on RDNA

There is no way to run "64 CUs of a 96-CU card" here and none is approved.
CU masks (`HSA_CU_MASK`, `ROC_GLOBAL_CU_MASK`, stream, queue, AQL or PM4 masks)
hung an R9700's MES scheduler on 2026-10-14, forced resets and took the host
down. They are refused at runtime init on RDNA (gfx10/gfx11/gfx12) and at every
mask entry point; see the `CHANGELOG.md` entry *CU masking is refused on RDNA*.
A profile's target CU count is metadata for reports and the estimate model
only: the 9070 profile still uses all 64 host CUs, the 7900 XT profile all 96,
and a UMA profile all of Strix Halo's 40. No emu path creates, reads or rewrites
a mask, and the contention probe that used masks is withdrawn.

Also not emulated, ever, by this feature: the target ISA/gfx (kernels compile and
run for the real host gfx), cache sizes, register-file size/occupancy and
scheduling geometry.

## Clocks and power cannot be emulated without root

Clock, power-cap and memory-clock controls are root-owned. No privileged write,
sysfs write or `rocm-smi` setting is attempted. Receipts only *read* world-readable
telemetry (the `bench_power` precedent) and never claim target board power,
sustained target clocks or energy per token. Nominal boost clocks are metadata,
not a measured compute rate.

## Bandwidth limiter: a proposal, not shipped

A "bandwidth-contention limiter" (a finite read/copy kernel running beside the
workload to eat memory bandwidth) is a separately approved, default-off
**laboratory proposal** with a kill-first gate (EMU-DESIGN §4). It is **not
implemented, not wired to any CLI or daemon path, and not enabled by `--emu`**.
It would not isolate a real bus-width or memory-clock cap (an unmasked hog also
changes occupancy, cache pollution, power and scheduling), and it needs explicit
maintainer approval after review of the crash forensics. The 9070 and 9070 XT
profiles share the R9700's 640 GB/s external spec, so they would not need one.
Bandwidth differences appear only as *estimate* inputs.

## Using it

```text
hipfire emu list [--json]
HIPFIRE_EMU=rx9070xt hipfire bench <model> ...        # environment
hipfire bench <model> --emu rx9070xt ...              # every spawned bench arm
hipfire bench <model> --emu off ...                   # overrides HIPFIRE_EMU
hipfire emu validate rx9070xt [--json] [--mib 256 --mib 1024]
hipfire emu estimate rx9060xt --input derating-input.json [--json]
```

- The profile reaches a daemon only as `developer.emu` in the startup
  `ProcessConfig` snapshot (process start only; a late `set_var` is invisible).
  `--emu` overrides the environment, including `--emu off` (which also drops the
  UMA override knobs, since they are invalid without a profile). The effective
  value is logged on stderr; nothing is logged when no profile is requested.
- Unknown ids, UMA knobs without a UMA profile and invalid scenarios fail
  **before any daemon starts**.
- A daemon refuses a profile the host cannot stand in for: the detected gfx must
  match exactly, integrated/discrete must match, physical CUs must be at least
  the target's, and real memory must cover the target capacity. Unknown host
  facts fail closed.
- `emu validate` starts a fresh model-free daemon with the profile installed and
  runs the `emu_validate` verb. The daemon refuses it while a model is loaded,
  with no profile installed, or when the request id differs from the installed
  profile. It exits nonzero when the `capacity`, `host_topology` or `graph`
  section fails (or the receipt `status` is `fail`); `unavailable` is reported,
  not failed. Run it on hardware only through the fabric scheduler.
- `emu estimate` is CPU-only. Exit 0 includes an `unavailable` estimate.
- Not supported under a profile (refused, never run uncapped): logical
  `HIPFIRE_EMULATE_GPUS`, PP/TP/EP > 1, mixed physical devices and
  shared-weight-host import.
- The direct-DRM Redline research backend (`redline::Device`, `crates/redline`)
  is not budget-accounted; no daemon or CLI process links it (only its own
  research examples call it), and hipfire processes allocate only through
  HIP/ROCr, which the ledger charges.

## Profiles

Capacity is GiB (`n << 30`), caches MiB, bandwidth decimal GB/s. Advertised
capacity is not guaranteed allocatable heap: an R9700's observed HIP total is
below `32 << 30`, so receipts report both. Target gfx/CUs/caches are
**metadata**; the host executes its own.

| ID | Host gfx / kind | Target gfx | Target CUs | Capacity | Peak BW | IC / L2 | VGPR file | Fidelity |
|---|---|---|---:|---|---:|---|---:|---|
| `rx9070xt` | gfx1201 discrete | gfx1201 | 64 | 16 GiB | 640 | 64 / 8 | 768 KiB | capacity-only, closest |
| `rx9070` | gfx1201 discrete | gfx1201 | 56 | 16 GiB | 640 | 64 / 8 | 768 KiB | capacity + estimate |
| `rx9060xt` | gfx1201 discrete | gfx1200 | 32 | 16 GiB | 320 | 32 / 4 | 768 KiB | cross-die estimate |
| `rx9060xt-8g` | gfx1201 discrete | gfx1200 | 32 | 8 GiB | 320 | 32 / 4 | 768 KiB | cross-die estimate |
| `rx7900xt` | gfx1100 discrete | gfx1100 | 84 | 20 GiB | 800 | 80 / 6 | 768 KiB | capacity + estimate |
| `rx7800xt` | gfx1100 discrete | gfx1101 | 60 | 16 GiB | 624 | 64 / 4 | 768 KiB | cross-die estimate |
| `rx7700xt` | gfx1100 discrete | gfx1101 | 54 | 12 GiB | 432 | 48 / 4 | 768 KiB | cross-die estimate |
| `rx7600` | gfx1100 discrete | gfx1102 | 32 | 8 GiB | 288 | 32 / 2 | 512 KiB | cross-die/occupancy estimate |
| `strix-point` | gfx1151 integrated | gfx1150 | 16 | UMA scenario | 128 | none / 2 | 512 KiB | platform scenario, not HX370 execution |
| `krackan` | gfx1151 integrated | gfx1152 | 8 | UMA scenario | 128 | none / 1 | 512 KiB | platform scenario, not AI7-350 execution |

Sources (also in each registry row's doc comment):

- gfx, CUs, cache and register file: [ROCm GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html).
- External bandwidth: AMD product pages for the
  [9070 XT](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html),
  [9070](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070.html),
  [9060 XT](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html),
  [7900 XT](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html),
  [7800 XT](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7800-xt.html),
  [7700 XT](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7700-xt.html),
  [7600](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7600.html),
  [Ryzen AI 9 HX 370](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html) and
  [Ryzen AI 7 350](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-7-350.html).
  The APU 128 GB/s is the LPDDR5X-8000 platform maximum, not a laptop guarantee.
- The AMD 9070 page confirms 640 GB/s, 2520 MHz boost, 220 W and PCIe 5.0 x16; the
  9060 XT page confirms 320 GB/s, 3130 MHz boost, 160 W (16 GB) and PCIe 5.0 x16.
  Sustained clocks and the PCIe link of the 7000-series rows are not recorded in the
  registry: an estimate for host-mapped offload needs the real negotiated link of
  the measuring host or a target receipt.

## UMA scenario defaults (`strix-point`, `krackan`)

Not hardware specifications: explicit scenario assumptions that firmware and the
OS can change, overridable with `HIPFIRE_EMU_HOST_BYTES`,
`HIPFIRE_EMU_CARVEOUT_BYTES`, `HIPFIRE_EMU_HOST_RESERVE_BYTES` and
`HIPFIRE_EMU_GTT_BYTES` ([env-vars.md](env-vars.md)).

```text
H = 32 GiB  host RAM of the emulated laptop
C =  1 GiB  BIOS/display carve-out reservation
R =  4 GiB  OS/ordinary CPU application headroom reservation
G = 16 GiB  aggregate virtual GTT limit  (= the emulated device capacity)
S = H - C - R = 27 GiB  managed shared-host allowance
```

Required: `C + R < H`, `0 < G <= S`, checked arithmetic, a real host with at least
`H` of RAM, and readable GTT/host safety queries. The emulated GPU allocates from
the virtual GTT; it does not gain `C + G`. Device backing counts against device,
GTT and shared-host; GTT-pinned host backing against host, GTT and shared-host;
userptr/explicit host-application backing against host and shared-host only.
`device_mem_info` reports `total = G` and
`free = min(real UMA-safe free, G - used_gtt, S - used_shared)`. Ordinary malloc,
the page cache, the CPU's cores, the NPU and the rest of a 32 GiB laptop OS are not
constrained: `R` is an accounting reservation, not a cgroup or RAM ballast.

## Reading a receipt and the memory reports

Three kinds of figure never share a field, and a reader should always be able to
tell which one they hold:

| Kind | Meaning | Where it appears |
|---|---|---|
| **Physical** (raw) | The real hardware and driver answer: raw HIP free/total, physical CUs, the HIP multiprocessor count (WGPs on RDNA, half the CUs), KFD count, observed HW_ID set, telemetry. Observed CUs must match the **host**, never the target. | Receipt physical/host sections; daemon `diag` `vram_free_mb`/`vram_total_mb` (unchanged, named in `emu.physical_fields`); the `physical:` figures in load-failure messages. |
| **Emulated** (budgeted) | This process's *managed application charge* against the profile cap (`accounting_kind = "managed_application_charge"`): profile capacity, budgeted free/total, committed and pending bytes per class (device, host, GTT, shared-host), pool-parked bytes. Not whole-driver residency: runtime code objects, queues and other processes are opaque to it. | Receipt capacity section; daemon `diag` `emu` object (`budgeted_free_mb`, `budgeted_total_mb`, `pool_parked_mb`, `ledger`); the `emu profile ... budgeted:` part of load-failure messages. |
| **Estimated** | Model output from measured inputs, labelled `[INFERENCE]` where it relies on a spec ratio. `estimate.status` is `unavailable` when time closure, rates or run identity are missing; no scalar is invented. `TopologyScenario` quality is never target-validated. | Receipt `estimate` section; `hipfire emu estimate` (`estimate.kind = "model_estimate_not_measured"`). |

Receipt (schema v1): top-level `status` (`pass|fail|unavailable`) plus separate
`capacity`, `host_topology`, `bandwidth`, `graph` and `estimate` sections, each with
its own `status` and an optional `reason`. There is no single "fidelity pass"
boolean. `cu_emulation` is `none`. In capacity-only mode the receipt makes no
target-bandwidth claim and records that no limiter is active. Synthetic runs
carry a deterministic input digest and a null prompt md5 with a reason, never a
made-up prompt hash.

With **no** profile the daemon's `diag` line, bench JSON and load-error text are
byte-for-byte what they were: the `emu` object and the `physical:` wording appear
only while a profile is installed.

`hipfire emu estimate` output keeps `measured` (input data), `estimate` (model
output, never presented as a measurement) and `target_profile` (registry metadata)
in separate objects.

## Fidelity limits

- **Guaranteed target:** deterministic process-local managed capacity accounting,
  admission queries and hard allocation refusal on a compatible host. It is not
  whole-driver memory virtualization: allocation rounding, runtime/code/queue
  overhead, other processes, fragmentation and driver behavior can differ from a
  real smaller board.
- **Highest relative fidelity:** R9700 to RX 9070 XT (same gfx, CU, cache and
  register geometry and external-memory spec), still different in board clocks,
  power, thermals and available driver heap.
- **Compute stays host-sized.** Fewer effective CUs, wave scheduling, SE/SA
  layout and small-kernel latency are estimates, not reproduced behavior.
- **ISA/code path:** target gfx paths (gfx1200/1101/1102/1150/1152) are not
  executed; everything compiles for the real host.
- **IC/L2 and occupancy:** the host cache is often larger; a 512 vs 768 KiB VGPR
  file changes target occupancy and cannot be recreated by a CU/clock ratio.
- **Clocks/power:** no privileged writes; nominal boost times CU count is not
  sustained compute.
- **PCIe/host offload:** the real link, NUMA, host DRAM, page placement, CPU and
  storage remain host properties. GPU bandwidth ratios do not scale host-mapped
  experts or CPU PLE/SSD phases.
- **APU:** capacity/GTT/carve-out are a declared scenario; CPU-shared bandwidth,
  package limits and the CPU/NPU of the laptop are not reproduced.
- **Numerics:** a capacity cap changes no arithmetic, but a smaller cap can change
  *placement* (resident vs host-mapped experts), which needs its own parity oracle.
  Profile the already-capped run, not an all-resident 32 GiB run.
- **Not done:** CU masks, root/sysfs writes, clock/power caps, persistent hogs,
  ballast allocations, fake target gfx/CU tuning, a universal tok/s multiplier,
  timer fabrication, NPU or multi-GPU redesign, route admissions.

## Measured sizing finding: Flash-Next on a 16 GB card

An RX 9070 XT (16 GB) host with 64 GB of RAM (TTM cap 31 GiB) **cannot load
Qwen3.8 Flash-Next**: its 38–61 GB of host-mapped routed experts exceed that
cap. Flash-Next on a 16 GB card needs **at least 128 GB of host RAM**. The same
line is in the Flash-Next model docs ([MODELS.md](MODELS.md)). The capacity
profile reproduces the device side of this (a 16 GiB application cap) on a 32 GB
R9700; it does not add host RAM, so it cannot make such a host load.

## Validation route

[VALIDATION.md](VALIDATION.md) row *SKU application-capacity emulation /
estimation*: CPU tests for the registry, ledger, model and CLI, plus
`hipfire emu validate <id>` on each compatible host through the fabric scheduler.
A passing receipt proves capacity enforcement and the host topology readout; it is
not a performance claim and not a route or model admission.
