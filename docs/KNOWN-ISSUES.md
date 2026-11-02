# Known issues

Open problems seen in hipfire testing that are not fixed yet. Each entry says
what was observed and on what, what was ruled out, and what to capture if you
hit it.

## gfx1151 (Strix Halo): rare GPU memory fault during deep prefill

**State:** measured; root cause unresolved. One occurrence in 41 deep
long-context runs on the test host.

**Symptom.** The daemon log shows a ROCr `HSA_STATUS_ERROR_MEMORY_FAULT` that
names a kernel. After that every HIP call fails with error 700
(`hipErrorIllegalAddress`). The error is sticky: restart the daemon to recover.
The kernel log (`dmesg`) shows an `amdgpu` GPU page fault
(`GCVM_L2_PROTECTION_FAULT_STATUS`).

**The occurrence (2026-10-15).**

- Build: beta `61d89f411`. Model: Qwen3.8 Flash-Next MQ4
  (`qwen3.8-flash-next-gptq3.mq4`), F32 QSA K/V state, NPU trunk pairs on,
  greedy AR.
- Request: a ~128K-token prompt in a session that had already served 8K and
  32K prompts; the fault hit the incremental prefill after row 81920.
- Daemon log: `HSA_STATUS_ERROR_MEMORY_FAULT` in
  `indexed_attention_select_scores_rows16_bf16_pm_gfx1151` (grid
  `[22528,46,1]`, address `0xeff040d90000`). The sticky 700 then surfaced at
  the next NPU pair's QSA arena growth (`QSA context map of 8388608 bytes at
  offset 46137344 failed`).
- Kernel log: only the first printed fault has a decoded status:
  `Faulty UTCL2 client ID: SQC (inst)` (shader instruction fetch),
  `PERMISSION_FAULTS: 0xb`, page `0x18000000000`. Five more addresses were
  printed without a per-address decode (`0x2200000000`, `0x2c800000000`,
  `0xbf3b0000`, `0x18000000000`, `0xeff040d90000`); only `0xeff040d90000` lies
  above 2^47. A preceding `38 callbacks suppressed` line cannot be attributed
  entirely to this process.
- Host: Strix Halo (gfx1151), Linux `7.3.0-rc7-perfopt+`, ROCm 10.1.0
  packages (HIP 7.16.26385, ROCr 1.21.0), MES firmware `0x86`,
  `amdgpu.cwsr_enable=1`.

**Not reproduced.**

- 40 further deep runs (one session each: 8K, 32K and 128K prompts, some to
  256K) on the faulting build and its neighbours, including split-K off, the
  map-ahead parent and an extra device sync before each NPU pair's memory
  growth: 0 faults, and no GPU fault line in the monitored runs' kernel logs.
  Every monitored run saw routine KFD queue evictions (0.9-9.6 s per run),
  with the NPU trunk on or off.
- Forced KFD queue evictions (a developer userptr-churn stressor firing every
  5 ms) while the PeaceMaker QSA score and select kernels ran: about 7,600
  eviction increments over 4 runs, 0 faults. This constrains only the tested
  conditions; it does not rule out every preemption or context-restore path.
- The developer extent check (`HIPFIRE_QWEN4_QSA_EXTENT_CHECK=1`) ran in all
  soak runs and never fired.
- A static audit found the PeaceMaker kernel descriptor and prologue of the
  faulting kernel at parity with its hipcc twin; no missing field that context
  save/restore depends on.

**Hypotheses (none proven).**

1. A GPU execution, address-translation or context save/restore defect
   (driver, firmware, CWSR, MES, TLB). Fits the first SQC(inst) fault; the
   forced-eviction negatives constrain only the tested conditions.
2. Loaded instruction or kernel-descriptor bytes corrupted by a stray write or
   a mapping problem. `HIPFIRE_QWEN4_QSA_CODE_CHECK=1` tests persistent
   CPU-visible corruption only, not every variant.
3. A data-arena extent, lifetime or access defect. The later, undecoded faults
   do not rule it out; the audited arena invariants show no failing window.
   The extent check is fail-fast hardening, not a fix.
4. Other hardware or runtime state faults. This list is not exhaustive.

The rarity fits timing, allocation or eviction state, memory layout or
transient corruption, but none of these is established. No synchronization,
padding or kernel change ships as a repair for this fault.

**If you hit it.** The daemon prints one line saying what to capture. Please
report:

- `sudo dmesg | grep -iE 'page fault|SQC|UTCL2|gfxhub|amdgpu'`, including the
  lines around the fault and any `callbacks suppressed` line;
- `uname -r`, the ROCm version (`cat /opt/rocm/.info/version` or your package
  manager's ROCm version) and `hipconfig --version`;
- the MES firmware version: `cat /sys/class/drm/card*/device/fw_version/mes_fw_version`;
- `cat /sys/module/amdgpu/parameters/cwsr_enable`;
- the daemon log from model load to the fault, the model, and the prompt length.

If it repeats, run with the developer diagnostics below and include their
lines.

**Developer diagnostics** (default off; fault localization, not serving
settings):

- `HIPFIRE_QWEN4_QSA_EXTENT_CHECK=1` checks QSA kernel byte extents against
  the mapped or allocated bytes before each launch and aborts with the arena,
  shape, limit and launch site on overflow. `2` also synchronizes after each
  checked launch, which serializes the GPU and is much slower.
- `HIPFIRE_QWEN4_QSA_CODE_CHECK=1` keeps a baseline of every loaded code
  object's read-only bytes (code and kernel descriptors), read through a CPU
  alias that is verified at load time, and compares once, on the CPU, at the
  first HIP error 700, logging one line. A difference shows the loaded code was
  overwritten by the time of the capture, not who wrote it. Equality does not
  show the bytes at the moment of the fault and rules out nothing about
  instruction caches, page tables, address translation, transient corruption
  or a bad program counter. Ranges without a verified CPU alias are reported as
  unavailable. A process that aborts before a host-side error 700 produces no
  capture.
  To check that the snapshot reads genuine CPU aliases on your system before
  relying on it, run
  `HIPFIRE_QWEN4_QSA_CODE_CHECK=1 cargo run --release -p rdna-compute --features lab --example qsa_code_check_selftest`;
  it loads PeaceMaker modules without launching kernels and prints one
  `hipfire: code-check self-test:` line.

**References.** ROCm source read during the analysis, pinned at
ROCm/rocm-systems `18f797bd` (same lineage as, but not verified identical to,
the HIP 7.16 / ROCr 1.21 build above):

- [CLR `hip_vm.cpp` 361-403](https://github.com/ROCm/rocm-systems/blob/18f797bd457c75d8ffb3ea95212825b9f2939389/projects/clr/hipamd/src/hip_vm.cpp#L361-L403)
  and [493-550](https://github.com/ROCm/rocm-systems/blob/18f797bd457c75d8ffb3ea95212825b9f2939389/projects/clr/hipamd/src/hip_vm.cpp#L493-L550):
  map and set-access paths, no stream drain.
- [ROCr `runtime.cpp` 4774-4882](https://github.com/ROCm/rocm-systems/blob/18f797bd457c75d8ffb3ea95212825b9f2939389/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp#L4774-L4882):
  `VMemorySetAccess` per-handle behaviour.
- [ROCr `runtime.cpp` 1329-1372](https://github.com/ROCm/rocm-systems/blob/18f797bd457c75d8ffb3ea95212825b9f2939389/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp#L1329-L1372)
  and [`amd_loader_context.cpp` 243-266](https://github.com/ROCm/rocm-systems/blob/18f797bd457c75d8ffb3ea95212825b9f2939389/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_loader_context.cpp#L243-L266):
  why the code check reads `hsa_amd_pointer_info` host aliases and not the
  loader's host shadow.

## Strix Halo on Ubuntu 26.04-class kernels: ~13 GiB of KHO scratch unusable by the GPU

**State:** understood; mitigated in hipfire (subtracted from host sizing, warned
about), fixed only by a kernel boot parameter.

**Symptom.** On a 128 GB Strix Halo running a kernel with Kexec HandOver (KHO)
enabled by default (Ubuntu 26.04's 7.0 generic kernel:
`CONFIG_KEXEC_HANDOVER=y` and `CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y`),
`/proc/meminfo` shows `CmaTotal: 0 kB` yet a `CmaFree` of ~13 GiB right after
boot, and `MemAvailable` reports that memory as available. KHO reserves
"scratch" memory at boot and releases it to the allocator as `MIGRATE_CMA`
pageblocks without counting it in `CmaTotal`. Only movable allocations can use
those pages; pinned GTT, `hipHostMalloc` and kernel allocations cannot, and
page cache that fills them still counts in `MemAvailable`. Sizing from
`MemAvailable` therefore over-reports by the whole region (measured:
13,536 MiB = 760 + 8,476 + 4,300 MiB of scratch, equal to the sum of the
per-zone `cma` lines in `/proc/zoneinfo`).

**What hipfire does.** Host sizing subtracts the whole CMA region, the larger
of `CmaFree` and the CMA pageblocks (`CmaTotal` or the `/proc/zoneinfo` `cma`
sum, which includes KHO scratch). A model load on a unified-memory GPU prints
`KHO scratch reserves N GiB unusable by the GPU; boot with kho=off to reclaim
it` once per process when the scratch is at least 1 GiB, and `hipfire diag`
reports it as a `kho_scratch` warning.

**Fix.** Boot with `kho=off` to return the memory to the system, or shrink the
reservation with the `kho_scratch=` kernel parameter.

**Not affected.** Fedora and openSUSE Tumbleweed kernels ship KHO with the
default-enable option off. Hosts where KHO fails to reserve its low-memory
scratch log `KHO: Failed to reserve ... disabling kexec handover` at boot and
have no scratch.
