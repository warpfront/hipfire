# Signed int8 GEMM on AIE2P

> **Lifecycle: historical kernel/bring-up note (2026-10-02), plus a dma-buf import source
> review.** It documents the single-core `gemm_i8` design and the first int8 plumbing,
> not the 0.4.2 serving route. Where it says a step is "offline" or awaits a hardware gate,
> that was the status of that day; silicon results are in [`report.md`](report.md). The
> int8/int32 epilogues here are lab designs; the 0.4.2 Flash-Next gfx1151 NPU trunk is
> BF16-direct only (calibration schema 4, no I32 contract, input projections only;
> out-projection unwired). See [`npu-trunk.md`](npu-trunk.md) for the current route.

The Rust kernel is `peacemaker_npu::kernels::gemm_i8`. It emits `isa::Program`
bytes directly; neither build nor execution invokes a vendor compiler. This
slice was offline when written; the Halo execution gate followed (see `report.md`).

## Instruction and layout contract

One dense VMUL/VMAC computes **8×8 times 8×8**, producing 64 signed int32
accumulators: 512 MACs per instruction. Both 64-byte source vectors and the
64-lane result are row-major. VMUL initializes the accumulator, so no C clear
or old-C load is required. Subsequent VMACs accumulate the reduction.

The control word is `0x308`: X signed bit9, Y signed bit8, amode0,
bmode1 at bits3–4, variant0, no subtraction/shift. Instruction fields come
from llvm-aie `AIE2PGenInstrInfo.td` (`VMAC_vmul_cm_core_X_X`,
`VMUL_vmul_cm_core_X_X`, `VLDA_dmx_lda_x_pstm_nrm_imm`,
`VST_dmx_sts_bm_pstm_nrm_imm`) and the generated register operand emitter.
TableGen specifies fields, not matrix arithmetic; the independent semantic
reference is `aie2p_vmult.h`'s signed `mul_8x8_8x8`/`mac_8x8_8x8` overloads
and `aie2p_compute_control`. References are Apache-2.0 WITH LLVM-exception.
The `032eb944d6cdbf2df2d4b8d9` vendor int8 GEMM disassembly initializes this
same control word and supplies the instruction-byte oracle.

Host A[M,K] and B[K,N] are row-major int8; C[M,N] is row-major int32.
Packing arranges K8 source microtiles in increasing reduction order. A DMA
chunk holds eight A microtiles (512 bytes), followed by eight B microtiles
(512 bytes), covering K64. No transpose, zero-point correction, bias, or
saturation is applied. K≤256 bounds the worst signed product sum to 4,194,304,
within int32. CPU reference uses wrapping int32 addition.

## One core

DMA-view tile addresses (core view is `0x70000 + address`):

| Buffer | Address | Bytes |
|---|---:|---:|
| A ping / pong | 0x0000 / 0x1000 | 512 each |
| B ping / pong | 0x2000 / 0x3000 | 512 each |
| C | 0x4000 | 256 |

S2MM0 cyclic BDs0→1→2→3→0 fill A0,B0,A1,B1. Empty/full semaphore pairs
are A0=(0,1), B0=(2,3), A1=(4,5), B1=(6,7). Every empty lock starts1;
every full lock starts0. DMA acquires empty by −1 and releases full by +1.
The core acquires full, computes, then releases empty. Core lock operands
use own-memory IDs48+id; BD operands use id directly. Ping/pong alternation
continues **across output jobs**, including odd K64 chunk counts.

C uses empty/full locks8/9. The core acquires Cempty before a job, stores
accumulator quarters LL/LH/HL/HH, waits for store completion, then releases
Cfull. MM2S0 BD4 drains 256 bytes, releases Cempty, and loops to itself.
This prevents the next job overwriting a still-draining C buffer.

The bounded program loops over pairs of jobs and emits a final odd job when
needed, rather than unrolling all output jobs into 16KiB program memory.
K is a positive multiple of64 up to256; jobs1..256. The schedule intentionally
uses standalone instructions and explicit NOPs: source-load latency7,
VMAC result latency6, and store memory stage5 from `AIE2PGenSchedule.td`.
It is a correctness baseline, **not a peak-throughput implementation**.

## 8 columns × 4 rows

An array wave covers M32×N64: core `(column c, physical row 2+r)` owns
output microtile `(wave_m*4+r, wave_n*8+c)`. Waves are M-major then N-major.
All32 cores have the same number of jobs. Supported array shapes are
M divisible32, N divisible64, K divisible64≤256, and at most256 waves;
there is no padded edge or tail path.

Each column has its own shim input/output stream. `pack_columns` orders
input as wave, then core row0..3, then K64 chunks, with A512 followed by
B512. `unpack_columns` gathers output wave/row-major C microtiles into host
row-major C. Input traffic deliberately replicates A across columns and B
across rows; no multicast/reuse claim is made.

One shim MM2S0 stream feeds memtile S2MM0, which stages four complete jobs
in separate row buffers. Memtile MM2S channels2..5 feed four independent
combined A/B streams up north lanes0..3. Intermediate compute rows pass
higher-row lanes through. Four core C streams return on south lanes0..3
to memtile S2MM channels2..5. Memtile MM2S0 gathers the four C microtiles
into shim S2MM0. Per-row double-buffered memtile staging and semaphore
pairs isolate producer/consumer progress. Channel parity selects the two
memtile BD banks. Memtile DMA own-memory addresses add0x80000; own-lock
DMA IDs add64. Shim routing uses the NOC mux/demux through South3 slave
(MM2S0) and South2 master (S2MM0), not nonexistent shim DMA switch ports.

The plan uses the DMA slice's structural `Bd`, `Task`, `Location`, lock writes,
`route::Circuit`, and `route::ShimDma` APIs. Shim addresses are supplied
by the caller; mapping/DDR patching and device submission remained the
runtime slice's responsibility. Cyclic tile/memtile tasks persist until configuration reset;
only finite shim output tasks issue completion tokens.

## Runnable single-core deployment

`design(&[0], 64, 64, 64)` builds a PDI and TXN for one core, column0/row2,
with64 output jobs. Its two firmware-patched arguments are combined AB input
(65,536 bytes) and packed C output (16,384 bytes). `Design::pack_in(a,b)`
returns the single **In** buffer; allocate the **Out** buffer from `args[1]`.
`Design::unpack_out` returns4,096 row-major int32 values. Other supported
single-core shapes are multiples8 with K64/128/256 and1..256 microtiles.

This deployment bypasses memtile DMA: its switch passes the input north
and output south. CDO loads the actual encoded program, resets/enables the
core, initializes semaphore locks, programs/starts the tile BD rings,
and configures all circuit/NOC routes. TXN programs contiguous shim BDs,
patches arguments0/1, queues transfers, and waits for the output token.
The shim TILE_CTRL packet route to SOUTH0 is included; without it data
can arrive while SYNC waits forever (the npu-hello hardware finding).
Use a fresh configuration/context per submission: the core runs once,
and cyclic tile DMA rings remain active afterward.

## Cycle model — estimates only

At a supplied clock f, the ideal dense hardware ceiling is
`512 MAC/core/cycle × 32 cores × f`. Counting multiplication and addition
separately gives `32,768 × f` operations/s:

- hypothetical 1.0 GHz: **16.384 TMAC/s = 32.768 TOP/s**;
- hypothetical 1.6 GHz: **26.2144 TMAC/s = 52.4288 TOP/s**.

These clocks are examples, not observed Halo clocks. At perfect utilization,
MAC cycles are `M*N*K / (512*32)`: cubic64/128/256 require16/128/1024 cycles.
The emitted conservative program has the separate issue-count estimate
`18 + (jobs>=2 ? 8 : 0) + jobs*(42 + 121*K/64) + floor(jobs/2)*16`.
It assumes one standalone instruction issue/cycle, includes explicit NOPs
and loop instructions, and excludes fetch stalls, lock/DMA stalls and any
additional hardware hazards. It is not measured execution time. For cubic
64/128/256, this standalone issue-count model estimates368/2,362/17,114 cycles.

Per column, shim input traffic is `waves*4*16*K` bytes and output is
`waves*4*256` bytes. Across eight columns that is `M*N*K/4 + 4*M*N`
bytes. With aggregate sustainable shim bandwidth B, the transfer lower
bound is this byte count/B; achievable latency cannot beat either that
bound or the compute bound. No bandwidth, overlap efficiency, power, or
concurrent-GPU performance has been measured by this slice.

## Offline gates

- `cargo test -p peacemaker-npu gemm_i8 -- --nocapture`: independent oracle bytes,
  signed packing/reference, and labelled peak-model output.
- `cargo test -p peacemaker-npu-sim gemm_i8_27_shapes_dma_cpu_exact -- --nocapture`:
  all27 combinations M,N,K in{64,128,256}, executing encoded core programs,
  semaphore locks, cyclic tile DMA, bounded stream FIFOs, and shim transfers;
  every output word is compared bit-exactly with the CPU reference.
- `cargo run -p peacemaker-npu-sim -- gemm 64` and `-- gemm 256`: actual CLI
  smokes of3 jobs each, including odd-job and cross-job ping/pong transitions.

Additional offline smoke exercised `pack_columns` through all 32 cores'
encoded programs and gathered every output for all 27 array shapes: CPU-exact.
Each emitted array plan had 432 distinct BD assignments and 256 conflict-free
circuit masters; CDO and TXN emission also completed.

The exact single-core Design PDI/TXN was loaded by the simulator's configuration
interpreter and submitted with packed host arguments for M=N=64 and
K=64/128/256. All 4,096 output words matched CPU in each case:

| K | PDI bytes | TXN bytes | Input bytes | Output bytes | Functional ticks |
|---:|---:|---:|---:|---:|---:|
| 64 | 2,144 | 300 | 65,536 | 16,384 | 16,591 |
| 128 | 2,768 | 300 | 131,072 | 16,384 | 32,975 |
| 256 | 4,000 | 300 | 262,144 | 16,384 | 65,743 |

Functional ticks are simulator progress counts, **not hardware cycles**.

The simulator gate proves the implemented arithmetic/dataflow model, not
hardware timing or the full memtile circuit on silicon. Vendor corpus
checks use absolute reference paths and skip optional corpus inspection
when the artifacts are absent. Halo execution remains unmeasured.

## HIP-written arguments via dma-buf (V8/V9/V10)

**Amendment 2026-10-03 (report.md, "HIP VMM dma-buf export does not live-alias"):** every run in this section wrote
its HIP buffer before the export and accessed it from the NPU only afterwards. Measured since: after the amdxdna import,
GPU writes to the HIP VA are not seen through the dma-buf and dma-buf writes are not seen by the GPU. A live shared
GPU/NPU buffer uses anonymous host pages pinned by amdxdna (`Device::userptr_bo`) and `hipHostRegister`ed for the GPU;
gate `npu-coop --mode alias`.

**Verdict: supported in the source, with an allocation caveat.** Ordinary
`hipMalloc` / `hipExtMallocWithFlags` device allocations are **supported-with-copy**
when initially VRAM-resident: amdxdna is a non-P2P importer and amdgpu pins them
in GTT, migrating their contents. Already-GTT/system pages are shared in place.
The probe uses **host-backed HIP VMM physical memory**, not `hipHostMalloc`.
The pinned host-VMM path is CPU-exact on silicon (report.md, 2026-10-02 21:46 UTC).

```
npu-gemm 512 512 64 --variant V8 --core FastSlowCtl --a-from-hip --iters 1
npu-gemm 512 512 64 --variant V8 --core FastSlowCtl --a-from-hip --iters 3 --reuse-ctx
```

`--a-from-hip` retains the arg0-only pinned-host probe. `--memory native`
uses normal BOs; `--memory vmm-host|vmm-host-uncached` instead uses
exported HIP VMM physical allocations for **all** A/B/C arguments. These
flags are mutually exclusive with `--a-from-hip`. Host uncached selects
`hipMemAllocationTypeUncached`, not a device-malloc flag; runtime failures
are reported without substituting another memory type. Uncached host
support depends on ROCr exposing its extended fine-grained pool.

`--burst 0..3` sets every shim BD's burst (64/128/256/512 B).
Normal emitters preserve vendor word 5: AxCACHE 2, AxQoS 0. Any other
cache/QoS is rejected before device access. `npu-bw --allow-hazardous-axi`
is a distinct explicit single-probe opt-in, never an automatic sweep;
declare such a run first, isolate it last in its own window, and stop/
report on timeout or device failure. QoS 15 wedged Halo (report.md).
`npu-bw --sweep` and `--sweep-axi` accept the same memory choices for both
arguments; the automatic AXI sweep varies only burst, keeping word 5.
`--ctl-probe nocompute` may use imported memory but remains timing-only;
it is not an exact GEMM. See report.md for simulator/default-byte proof
and measured bandwidth; simulator functional exactness does not model
cache coherence or AXI throughput.

The device opens before HIP is loaded. There is no build-time ROCm dependency.
The probe uses `hipMemCreate` with Pinned/Host/PosixFileDescriptor properties,
granularity-rounded allocation, VA reservation/map and GPU read/write access;
`hipMemcpyHtoD` writes packed A and `hipDeviceSynchronize` completes the write.
`hipMemGetHandleForAddressRange` exports the VMM allocation as a dma-buf fd
(DmaBufFd type, flags 0). `Device::import_dmabuf` maps it as arg0, without
copying packed A into an NPU-owned input BO. B/C and the exact CPU comparison
are unchanged. HIP memory and its fd outlive all imported BOs/contexts.

### Evidence chain

Kernel citations below are relative to the Linux 7.0 kernel source tree (a local reference checkout that is not part of this repository):

- `drivers/accel/amdxdna/amdxdna_pci_drv.c:249` installs the custom
  `gem_prime_import` hook (not the generic driver's default implementation).
  `amdxdna_gem.c:644-677` attaches/maps the dma-buf SG table bidirectionally,
  calls `drm_gem_shmem_prime_import_sg_table`, and sets the BO type to SHMEM.
  UAPI `include/uapi/drm/amdxdna_accel.h:176-188` also exposes dma-buf import
  through CREATE_BO's VA table.
- Addressing is not an SG-first-address shortcut. `amdxdna_gem.c:557-559`
  initializes addresses invalid; `:342-346` mmap registers HMM and
  `:248-249` assigns its host VA. Identity/SVA uses that VA/PASID
  (`amdxdna_gem.h:74-77`, `amdxdna_pci_drv.c:78-90`).
  Optional force-IOVA mode maps the page-backed SG table into a contiguous
  IOVA and assigns `dma_addr` (`amdxdna_iommu.c:38-75,132-139`).
- EXEC_CMD looks up/pins arbitrary argument handles
  (`amdxdna_ctx.c:420-461,584-625`); import pin is a no-op because attachment
  mapping already pins (`amdxdna_gem.c:869-870`). Our
  `crates/railgun/src/npu/mod.rs`, `HwCtx::submit`, embeds each `Bo::dev_addr()` into
  firmware bo0..bo4 and separately passes argument handles. V8's shim DDR
  patches use arg0=A (`crates/peacemaker-npu/src/kernels/gemm_array.rs:50-58,80-83`).
- The NPU is non-cache-coherent; SYNC_BO flushes an imported SG table
  (`amdxdna_gem.c:977-982,1007-1012`). The probe flushes the imported CPU
  mapping but never writes it. Explicit HIP synchronization is required;
  do not assume automatic GPU-producer dependency import. NPU submit adds
  WRITE completion fences to BO reservations and a timeline syncobj point
  (`aie2_ctx.c:991-1025,1031`).

Exporter citations are Linux stable **v7.0.14** (the installed 7.0.0-38
headers identify this baseline), available from
[kernel.org](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/?h=v7.0.14):

- `drivers/dma-buf/dma-buf.c:1033-1040,1072-1075`: plain `dma_buf_attach`
  supplies no importer operations, leaving `peer2peer=false`.
  `:926-932,1179-1215` pins static imports and waits KERNEL-use fences.
- `drivers/gpu/drm/amd/amdgpu/amdgpu_dma_buf.c:127-157,188-209`: non-P2P
  attachments exclude VRAM from pin/map placement; GTT SG pages are DMA-mapped.
  `amdgpu_object.c:979-986` validates placement. Already-compatible placement
  skips moving (`drivers/gpu/drm/ttm/ttm_bo.c:845-848`); a VRAM→TT move copies
  via blit or memcpy (`amdgpu_ttm.c:390-425,591-604`).
- `amdgpu_dma_buf.c:302-327` may migrate unpinned BOs to GTT for CPU reads.
  Generic `dma_buf_begin_cpu_access` also waits reservation fences
  (`drivers/dma-buf/dma-buf.c:1443-1499`); mmap alone is not this CPU-access
  synchronization operation.

HIP/ROCR citations are from
[ROCm/rocm-systems d1cf05c0dfbd3b63a7e637422294c4e864e44d56](https://github.com/ROCm/rocm-systems/tree/d1cf05c0dfbd3b63a7e637422294c4e864e44d56),
with ABI checked against HIP 7.15.26333 / ROCm core 10.0.0 headers:

- `projects/clr/hipamd/src/hip_memory.cpp:402-483,830-851` distinguishes
  host fine-grain allocations from device Finegrained/Uncached flags.
  `projects/clr/rocclr/device/rocm/rocmemory.cpp:938-1002` and
  `rocdevice.cpp:2496-2524` select GPU-local pools for device malloc flags.
  Finegrained does **not** mean system/GTT.
- Portable HSA export requires one GPU-agent pool allocation; CPU-agent
  allocations such as `hipHostMalloc` are rejected
  (`projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp:4241-4287`).
  Calling `hsa_amd_portable_export_dmabuf` directly does not fix this.
- Host-backed VMM uses a system allocation with `AllocateGTTAccess` and a
  GPU DRM owner (`runtime.cpp:4394-4421`), and shareable export uses that
  owner (`:4930-4960`). HIP Host properties select this route
  (`projects/clr/hipamd/src/hip_vm.cpp:91-167`).
  Address-range export recognizes VMM and uses vmem shareable export, not
  the GPU-agent-only portable path
  (`projects/clr/rocclr/device/device.cpp:1373-1390`,
  `device/rocm/rocmemory.cpp:1277-1325`).
  Ranges must be contained within an allocation; VMM size/map constraints
  require granularity alignment. Missing runtime support is a hard failure,
  never a fallback to a copying device allocation.

The original missing-device local smoke preceded the silicon results.
Current measured exactness/bandwidth, logs and safety incidents are in
report.md; no-migration claims distinguish host VMM from device VRAM.
