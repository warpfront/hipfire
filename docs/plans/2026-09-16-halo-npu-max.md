# Halo XDNA2 maximum-throughput and bare-dispatch investigation

Date: 2026-09-16. Coordinator/research: HaloNpuMax. Execution is delegated through Main. Source worktree is `/home/kaden/ClaudeCode/warpfront/wt-lloyd`; all device artifacts belong under `hipx:/home/kaden/npu-screen`. No production wiring or commits. This document is updated as executable evidence arrives; unmeasured entries are not performance claims.

## 1. Findings, scope, and evidence discipline

This investigation separates (A) the throughput of a properly tuned **int8→int32** dense GEMM, (B) direct amdxdna submission without XRT userspace, and (C) the additional contract required to replace hipfire IU4 rows. Success in A or B is not proof of C.

Existing measured evidence, inherited from `2026-09-16-halo-npu-probe-results.md` and `history://HaloNpuProbe`:

| M×K×N | stock tile m/k/n; columns | NPU average µs | minimum / maximum µs | average TOPS | unique-payload GB/s | validation |
|---|---|---:|---:|---:|---:|---|
| 2048×2048×2048 | 64/64/32; 8 | 4018.3 | 3704.0 / 4307.4 | 4.28 | 6.26 | NumPy int64 reference PASS |
| 17408×5120×512 | 64/64/32; 8 | 16099.4 | 15735.6 / 16819.9 | 5.67 | 7.91 | NumPy int64 reference PASS |

These are **averages, not medians**, 10 warmups and 20 timed samples. Raw logs: `/home/kaden/npu-screen/logs/sanity_2048.log` and `gateup_17408x5120x512.log`. End-to-end averages were 4828.5 and 16976.6 µs respectively. The down-shape result and tuning results are not yet supplied to this document.

Hardware/software inherited state: Ubuntu 26.04, kernel 7.0.0-31, amdxdna, `/dev/accel/accel0`, RyzenAI-npu5, firmware 1.1.2.65; distro XRT 2.21.75. mlir-aie f50bef7 / wheel 1.4.4.dev3 and Peano 22.0.0.2026090701. Plain SSH still has 8 MiB memlock; run device programs in `sudo -n -u kaden -i` login shells. The previous probe recorded the limits.d and systemd memlock changes. No new system changes are required for research.

**Corrections to the preceding screen:** a slower NPU on the *whole* matrix does not imply that a concurrent row-slice loses; ideal split throughput adds. Activations are indexed by K and N and are reused by **every** M-slice, not row-partitioned with weights. A single full-K i32 output is insufficient for IU4: each 128-K half needs its own dot before a differently scaled fold. Finally, the paper's 450.6 MAC/core/cycle and 38 TOPS headline concern narrowed i8 output, not automatically i32 output.

## 2. Throughput ceiling (A), tuning, tracing, and standalone bandwidth

Primary source: [Taka et al., arXiv:2512.13282v1](https://arxiv.org/html/2512.13282v1), particularly §§4.2–4.5 and Tables 1–3. It reports **384 MAC/core/cycle for int8→int32** (48×280×48 single-core tile), versus 450.6 for int8→int8. It uses xchesscc, hardware traces, turbo mode, and a Krackan Point DDR5 mini-PC—not this Halo/Peano system.

Table 3's best displayed int8→int32 point is **24.74 TOPS** at 4224×4224×4608, tile 96/64/96 and 256 MAC/core/cycle. The 37.35 TOPS point is i8→i8. Thus 30+ TOPS for full-precision output is an aspiration here, not a result already established by the cited comparison.

Read source on hipx:

- `mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py`: `_build_design`, `_device_for`, `core_fn`, `sequence`, `_validate_shape_args`, `_numpy_reference`, `_run_and_verify`.
- `mlir-aie/aie_kernels/aie2p/mm.cc`: `matmul_vectorized_2x2_mmul`; four independent accumulators, explicit `event0()`/`event1()` delimiters, reload/store C per k-tile. Verify the emitted kernel uses this AIE2P source, not the AIE2 version.
- `mlir-aie/python/utils/trace/__init__.py`: `configure_trace`, `start_trace`, `parse_trace`, `TraceConfig`; `Program.enable_trace` is the IRON frontend.

Source facts: npu2 uses the unrestricted eight-column device. `Worker.grid(4,n_aie_cols,...)` creates 32 workers at cols=8. All FIFO depths are currently 2; L2 input tiles are only m×k and k×n, so merely increasing FIFO depth does not implement the paper's independent large k_mt contiguous transfers. Stock stack allocation is 0xD00 bytes per core. CLI additionally requires M/(4m) even. A/B in L1, C in L1, all simultaneous, must fit including stack, alignment, and bank placement.

For this output-stationary mapping, distinguish unique payload bytes `MK+KN+4MN` from scheduled DMA bytes:

`A_read = MKN/(n*cols); B_read = MKN/(m*4); C_write = 4MN`.

The stock gate shape schedules A twice and B 68 times. Its 7.91 GB/s unique-payload rate is **not a measured DDR ceiling**. TAP-derived byte totals and actual bus/cache traffic must not be conflated either.

### Executable composer slice A: tuning and profiler evidence

**Ownership:** only `hipx:/home/kaden/npu-screen/max-tune/` (copy the original design and any changed kernel there), outputs under that directory. Do not modify the pinned upstream source, production worktree, or other workers' artifacts. Skip formatters, linters, and project suites. Coordinate CPU-heavy build phases and all device phases through Main; no builds during GPU timings. Builds may proceed while the direct-dispatch worker performs source/host-only work; NPU execution must be serialized between workers.

**Frozen artifact interface:** each passing configuration directory contains `design.xclbin`, `insts.txt` (32-bit hexadecimal words), `insts.bin` (little-endian u32), generated/lowered MLIR, AIE ELF/object and disassembly, build log, deterministic `a.bin`, `b.bin`, `c_xrt.bin`, plus `manifest.json`. Manifest fields: `M,K,N,m,k,n,cols,dtype_in,dtype_out,b_col_maj,c_col_maj,l1_a_depth,l1_b_depth,l1_c_depth,l2_a_depth,l2_b_depth,l2_c_depth,k_mt,stack_bytes,compiler,source_revision,kernel_name,artifact_sha256,argument_sizes,argument_offsets`. Record extra fields if required, never silently assume argument ABI. `results.jsonl` stores config id, every timed sample in ns, warm count, timed count, median/p90/mean/min/max µs, logical operations, unique payload bytes, TAP-derived DMA bytes, correctness status, power/clocks, and trace paths. The raw-dispatch worker consumes this directory, without rebuilding the GEMM.

**Commands / starting point:** under a `sudo -n -u kaden -i` shell, source `/home/kaden/npu-screen/mlir-aie/ironenv/bin/activate` and upstream `utils/env_setup.sh`; use the copied `whole_array.py --dtype_in i8 --dtype_out i32 --dev npu2 --n-aie-cols 8 -M 17408 -K 5120 -N 512 -m 64 -k 64 -n 32 -w 10 -i 50`. Ahead-of-time flags supported by the source are `--xclbin-path=<file> --insts-path=<file>`. Build offline, then benchmark cached artifacts with no CPU reference computation during timing. Generate/reference input outside the measurement phase. Save full command lines. Do not label narrowed i8/i16 output results as satisfying this contract.

**Step A1, bounded legal sweep:** retain stock baseline; then at cols=8 try m/k/n = 64/128/64, 64/128/32, 128/64/32, 32/256/64, 64/256/32, and 64/64/64 (all input/output FIFO depths initially 2). Test B row-major and column-major on the best two. Try cols=4 and cols=2 with the best legal tile, allowing n=64 or 128 where N divisibility and L1 capacity permit. Reject before compilation if `dA*m*k + dB*k*n + dC*4*m*n + 0xD00 > 65536`; physical bank allocation remains a further gate. For example 64/256/32 with double C is 65536 bytes *before* stack, so it must use single C or be rejected. Preserve exact-shape legality; padding results must separately report payload and padded operations.

**Step A2, paper-derived changes (not just flags):** decouple A/B/C L1 depths and L2 depths; single-buffer C at L1 to free capacity, double-buffer A/B, try 64/256/32 and 128/64/64 when legal. Keep L2 C aggregation by four rows. Add an independent `k_mt` at multiples of k, initially 256,512,1024, constrained by each 512 KiB MemTile and neighbors. L2 A m×k_mt and B k_mt×n for column-major B; L2→L1 emits consecutive k-sized subtiles with the paper's split MemTile-MM2S and CompTile-S2MM transformation. Merely widening a FIFO without adjusting the DMA transforms is invalid. Compare buffering depths 2 and 3 only where real allocation fits. Keep A broadcast across M rows and B across columns. Inspect whether A stream placement uses the intended even-column/neighbor capacity mapping for the asymmetric 4×8 array.

**Step A3, compute and DMA diagnosis:** retain untraced performance numbers. For stock and winner, build a traced artifact. Use `Program.enable_trace` before resolve; consult the pinned method signature. Trace representative first/last columns, not all 32 cores at maximal event rate. Existing `event0/event1` around matmul provide exact compute-call duration; report `m*k*n/cycles` MAC/cycle. Select named AIE2P events for INSTR_EVENT_0, INSTR_EVENT_1, INSTR_VECTOR, LOCK_STALL, STREAM_STALL, memory stalls, and selected DMA port RUNNING/STALLED events (max eight per trace unit). Port selectors must match lowered physical DMA channels, not guessed defaults. Parse with `aie.utils.trace.parse_trace(raw, lowered_mlir)` and summarize with `get_cycles_summary`, `get_vector_time`; save raw buffer and parsed Perfetto JSON. Attribute time to core active / input-lock wait / output backpressure / transfer gaps, detect trace overflow, and compare traced versus untraced time to bound perturbation. Do not call INSTR_VECTOR a direct MAC counter; combine event0/1 duration with known logical MAC count. If compute low, inspect Peano software pipeline and intrinsic instruction selection; test kernel unrolling/accumulator layout variants only against measured active-cycle evidence. If DMA low, compare contiguous vs strided transfers and k_mt, and then the paper's rolling BD reuse (up to 15 of 16 descriptors/shim) against the stock two transfer-block TaskGroup schedule. Reconfigure only retired BDs; no DMA or FIFO ownership violation.

**Step A4, actual dispatch shapes:** measure winner at gate full 17408×5120×512 and down full 5120×17408×512; measure M slices 1024,2048,3072,4096 for K=5120,N=512 and K=17408,N=512 (omit only genuinely illegal sizes with reason; also a 512-row down slice if legal). At least ten warm and fifty timed resident-buffer samples, medians/p90. Export winner artifacts promptly so raw-dispatch can work independently.

**Step A5, NPU→DDR only:** build a stream-out microbenchmark in the owned directory. Initialize a known pattern on AIE/L2 before the timed interval, then repeatedly S2MM to disjoint contiguous output ranges across 1,2,4,8 shims, large enough to amortize start/finish and avoid mistaking a warm tiny buffer for sustained DRAM traffic. Cover ≥4,16,64 MiB written where allocation permits. No matmul and no paired DDR input reads in the timed region. Validate the full output, record bytes/time GB/s plus dispatch and cache-sync times separately; do not call memcpy or BO SYNC_BO timing NPU→DDR bandwidth. A copy benchmark may be supplemental but is explicitly bidirectional. For a generated constant pattern, core generation can bound throughput; disclose/measure that ceiling or use replayed on-chip contents.

**Acceptance/abandon:** output every attempted config and precise compiler/runtime failures, best full and slice measurements, standalone write bandwidth, traces for stock/winner or the exact trace-tool failure and successfully attempted alternatives. Reach 30+ TOPS **for i32** or establish a measured active-core/DMA/command-gap bound, not 'stock was slow'. Stop an individual novel kernel idea after controlled paired trials show no repeatable gain and counters do not support its proposed bottleneck; retain the best passing implementation, not speculative complexity. No unmeasured universal throughput-ceiling claim is accepted.

## 3. Contention: GPU and NPU slowdown

Not yet measured in this followup. The definitive comparison must use the tuned winner, not only the stock loop. The prebuilt GPU oracle is:

`HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-lf3 HIPFIRE_GFX11_MQ4V2_IU4=1 TIME=1 /home/kaden/wt-lloyd/target/release/examples/tmp_halo_iu4_oracle /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt`

Reserve a build-free measurement phase with Main and HaloLF7. Perform GPU-alone, NPU-alone, overlapped, then GPU-alone again; preserve per-sample timestamps and prove actual overlap. Report gate and down GPU medians separately and NPU medians for the overlap window, not a long NPU loop average dominated by GPU idle intervals. No NumPy verification, compiler, decompressor, or unrelated heavy CPU work during this phase. Kill if GPU median slowdown exceeds 20%; report environmental drift and clocks, not just one aggregate percentage.

## 4. Direct dispatch (B): source-grounded ABI and proof

The header path in the assignment has moved: current AMD repository provides [`src/include/uapi/drm_local/amdxdna_accel.h`](https://github.com/amd/xdna-driver/blob/main/src/include/uapi/drm_local/amdxdna_accel.h) and a separate in-tree UAPI copy. Runtime compatibility must be checked against the installed Ubuntu driver/shim revision, not assumed from latest main.

[`src/shim/host/platform_host.cpp`](https://github.com/amd/xdna-driver/blob/main/src/shim/host/platform_host.cpp) directly implements `create_ctx`, `config_ctx_cu_config`, `create_drm_bo`, `sync_bo`, `submit_cmd`, `wait_cmd_ioctl`, and timeline-syncobj wait. It demonstrates that XRT is a userspace policy/encoding layer over public ioctls; this is source evidence of feasibility, **not yet a working no-XRT proof**.

Critical known ABI details: CREATE_HWCTX receives QoS pointer, tile count and maximum operations/cycle, and returns context and timeline-syncobj handles; PDI CU configuration is a separate CONFIG_HWCTX(CU) operation, not an xclbin pointer passed to CREATE_HWCTX. CREATE_BO and GET_BO_INFO supply mmap offset and device address. Single-command EXEC_CMD puts the **command handle itself** in `cmd_handles` (not a pointer), while `args` points to an array of u32 BO handles and `cmd_count=1`. WAIT_CMD takes context, timeout in milliseconds (zero=infinite), and returned u64 sequence. Generic GEM_CLOSE and SYNCOBJ_DESTROY manage resources. PRIME import/export exists in the shim but its success for GPU BOs is a separate driver/memory-type gate.

### Flat offline artifact generation and exact initialization

The pinned `programming_guide/compilation_stages.md` documents that every JIT cache directory keeps `main.pdi`, `insts.bin`, `main_aie_partition.json`, `main_kernels.json`, `main_mem_topology.json`, lowered `input_with_addresses.mlir`, `main_core_<col>_<row>.elf`, the CDO init/ELF/enable blobs, and `main_design.bif`. The PDI is therefore **already a flat file**, not locked inside XRT. `specialized_design.compile(xclbin_path=...,inst_path=...,pdi_path=...)` explicitly emits the three files; in cache mode `specialized_design.get_pdi_path()` finds the PDI without recompilation. The old `.txt` path is hexadecimal words; distinguish it from binary `insts.bin` by content, not by filename alone.

Observed example metadata in `hipx:/home/kaden/.npu/cache/280360844a039cedca472303` (illustrates the actual toolchain ABI; do not claim this one-core artifact is the full-array winner): AIE partition `column_width=8`, `operations_per_cycle=2048`; CU `MLIR_AIE:MLIRAIE`, functional=0, kernel id 0x901 mapped through PDI cdo_groups. Kernel argument byte offsets in the CU payload: opcode u64 at 0x00, instr u64 address at 0x08, ninstr u32 at 0x10, bo0 u64 at 0x14, bo1 at 0x1c, bo2 at 0x24, bo3 at 0x2c, bo4 at 0x34. The last pointers are **not naturally u64-aligned**; write bytes/u32 words, not an ordinary padded C struct. The metadata floors the BO count for firmware command-chain requirements; preserve zeroed unused argument slots.

`src/shim/hwctx.cpp::xclbin_parser` maps each kernel's `kernel_id` through AIE-partition PDI CDO groups, records `functional`, `ops_per_cycle`, and column count. `hwctx::ctx::create` passes `num_tiles=column_count*device_core_rows` to CREATE_HWCTX. `src/shim/kmq/hwctx.cpp::hwctx_kmq` allocates cacheable/device PDI BOs, copies and syncs the PDI, and submits CONFIG_HWCTX(CU) with the handle and functional byte. `src/shim/kmq/pcidev.cpp::pdev_kmq::on_first_open` creates the device heap (initial 64 MiB, 64 MiB alignment; latest shim can grow it). `src/shim/buffer.cpp::bo_flags_to_type` selects DEV for CACHEABLE when that heap exists, SHARE for host-only data, CMD for execute buffers.

Important version distinction: upstream Linux **v7.0 has no WAIT_CMD ioctl** (slot 9 is absent), whereas AMD's private header declares it. The installed in-tree path should wait on the CREATE_HWCTX timeline syncobj with `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`, point=`EXEC_CMD.seq`. A WAIT_CMD ENOTTY on this driver would be an ABI difference, **not an XRT dependency**. Probe/record which wait actually works.

### Executable composer slice B: raw-ioctl proof, no XRT at runtime

**Ownership:** `hipx:/home/kaden/npu-screen/raw-ioctl/` (C is preferred for a minimal self-contained ABI proof), optionally copied into a new standalone `wt-lloyd/tools/npu-probe/` outside workspace members. Do not edit production or the tuning worker's files. Skip formatters/linters/project suites. Obtain a device-time reservation through Main; host-side source reading and C compilation are independent of the expensive tuning build, but no NPU overlap and no CPU build during GPU timing.

**Input/output contract:** consume the config-directory manifest and flat artifacts from slice A (§2); start using existing cached stock artifacts while A tunes, then re-run against the actual winner. Produce `raw_npu.c`, vendored minimal UAPI definitions or version-pinned header, an offline-only metadata exporter if necessary, binary, complete build/ldd/readelf logs, per-ioctl trace, deterministic input and both output files, SHA256/byte comparison, raw timing JSON with every sample. Extend the manifest with PDI filename/hash, `column_width`, `ops_per_cycle`, CU index/function/kernel id, payload byte span and typed argument offsets. The runtime binary must not dlopen XRT and must not launch Python or an XRT helper.

**Exact structs on x86_64:** use fixed-width members, zero reserved fields, and `_Static_assert(sizeof/offsetof)`:

| ioctl (DRM command-base + id) | structure size | field order |
|---|---:|---|
| CREATE_HWCTX (0) | 56 | u64 ext,ext_flags,qos_p; u32 umq_bo,log_buf_bo,max_opc,num_tiles,mem_size,umq_doorbell,handle,syncobj_handle |
| DESTROY_HWCTX (1) | 8 | u32 handle,pad |
| CONFIG_HWCTX (2) | 24 | u32 handle,param_type; u64 param_val; u32 param_val_size,pad |
| CREATE_BO (3) | 32 | u64 flags,vaddr,size; u32 type,handle |
| GET_BO_INFO (4) | 48 | u64 ext,ext_flags; u32 handle,pad; u64 map_offset,vaddr,xdna_addr |
| SYNC_BO (5) | 24 | u32 handle,direction; u64 offset,size |
| EXEC_CMD (6) | 56 | u64 ext,ext_flags; u32 hwctx,type; u64 cmd_handles,args; u32 cmd_count,arg_count; u64 seq |
| WAIT_CMD (AMD private id 9 only) | 16 | u32 hwctx,timeout_ms; u64 seq |

Mainline v7.0 QoS is six u32 values: gops,fps,dma_bandwidth,latency,frame_exec_time,priority. Latest private header adds user_start_col/reserved; use the installed driver's contract, with no speculative nonzero extension. CU config payload is `u16 num_cus; u16 pad[3];` then each `{u32 cu_bo; u8 cu_func; u8 pad[3];}`; size 16 bytes for one CU. BO types are SHARE/SHMEM=1, DEV_HEAP=2, DEV=3, CMD=4; direction TO=0/FROM=1. IOCTL encoding is DRM_IOWR, type 'd', base 0x40; calculate with the vendored macros rather than unchecked magic numbers.

**Initialization and execution sequence:**

1. `open("/dev/accel/accel0",O_RDWR|O_CLOEXEC)`; query DRM version/device metadata/firmware and identify KMQ. Read artifact manifest and flat PDI/instructions, validate byte sizes and hashes offline before device writes. All arrays/pointers and the device fd remain alive through completion.
2. CREATE_BO(DEV_HEAP,64 MiB), GET_BO_INFO, reserve an appropriately 64-MiB-aligned CPU range and map the returned fake offset `MAP_SHARED|MAP_LOCKED` as the installed shim does. This must precede DEV BO allocations and context setup if the driver requires an existing heap. Do not put 89-MB weights in the 64-MB heap: A/B/C are SHARE BOs; PDI and instructions are small DEV allocations.
3. CREATE_HWCTX with zeroed ext/UMQ/log/mem fields, QoS matching reference, max_opc from manifest, num_tiles=column_width*queried_core_rows (32 for eight columns/four compute rows). Keep returned context handle and syncobj. Do not call a PM4 doorbell or invent an UMQ for npu5/KMQ.
4. CREATE_BO(DEV) for PDI, GET_BO_INFO; if map_offset is invalid, derive CPU pointer as heap_cpu+(pdi_xdna_addr-heap_xdna_addr), exactly as `buffer::vaddr`. Copy PDI, SYNC_BO TO (or explicitly replicate the installed shim's fenced cache-line flush if driver sync is unsupported; report which). CONFIG_HWCTX with CU record {pdi handle,function=manifest}.
5. CREATE_BO(DEV) for instruction words and CREATE_BO(SHARE) for A/B/C; GET_BO_INFO and mmap SHARE BOs. Device arguments use xdna_addr when valid; SHARE may return INVALID_ADDR and then use the actual pinned user mapping VA, as `buffer::paddr` does. The CPU VA is not necessarily a raw physical address. Fill deterministic A/B; initialize C with a changing sentinel, **flush C before execution** so later CPU dirty-line eviction cannot corrupt device output; flush/sync A/B/instructions. Keep heap, PDI, instruction, input/output BOs resident throughout repetitions.
6. CREATE_BO(CMD) and mmap the ERT command. Build `ert_start_kernel_cmd` with state NEW=1, opcode ERT_START_CU=0, type ERT_CU=3, extra_cu_masks=0, cu_mask=`1<<cu_index`. Real bit positions from installed `/usr/include/xrt/detail/ert.h`: state bits0–3, stat_enabled4, extra masks10–11, count12–22, opcode23–27, type28–31 (the header's comment says 27–31, but its actual 4-bit field starts at 28). Count is number of u32 words **after the 4-byte header**, including CU mask; not bytes and not `sizeof(struct)`. Populate the CU payload at manifest offsets: runtime opcode=3 for the ordinary mlir-aie transaction path, instr address, ninstr in **u32 words**, A/B/C addresses, zero unused slots. Confirm opcode and payload against a same-artifact XRT reference capture before blaming firmware on failure. ERT_START_NPU=20 is a different instruction-data packet, not an interchangeable replacement.
7. EXEC_CMD with hwctx, type=0, cmd_handles=the CMD BO handle value, cmd_count=1, args=pointer to u32 list of all used data/instruction BO handles, arg_count=list length; these handles ensure pinning/lifetime, while addresses live in the ERT payload. Save returned sequence.
8. WAIT_CMD only if supported; otherwise generic timeline wait on returned syncobj/sequence, with a finite absolute monotonic deadline and WAIT_FOR_SUBMIT as needed. Record the exact wait flags. Check the ERT state for COMPLETED, not merely a successful wait syscall. SYNC_BO FROM C / installed equivalent cache invalidation, then byte-compare the **entire** C buffer with `c_xrt.bin`. A successful wait with ERROR/TIMEOUT is failure.
9. For reuse, only after successful completion/read ownership restore: reset command state NEW, restore changing sentinel/output ownership if that is the test phase, update inputs if desired, repeat. Do not remap/allocate/reconfigure PDI inside the hot timing loop. Teardown after quiescence: destroy syncobj/context in driver-supported order (shim destroys syncobj then context), GEM_CLOSE buffers, unmap, close fd. Failed/timeout work must be quiesced by context teardown before freeing backing storage; never continue timing on a poisoned context.

**Proof and timing commands:** compile a C binary using `cc -O2 -std=c11 ... -o raw_npu` with libc and headers only, no `-lxrt*`; save `ldd raw_npu`, `readelf -d raw_npu`, and while it runs verify `/proc/<pid>/maps` contains no libxrt. Run inside the fixed-memlock login shell. Use `CLOCK_MONOTONIC_RAW` before EXEC_CMD through completion-state check, excluding configuration/allocation and input/output sync; also separately report host-visible call including required output sync. At least ten warmups and 100 samples for full GEMM, ≥1000 for small/empty. Build a real minimal valid empty transaction or smallest known-good GEMM separately; report which, its operation count, and do **not** call a small GEMM time pure ioctl overhead. No arbitrary malformed 'empty' packet.

Reference path may use Python/XRT only to generate/check `c_xrt.bin` offline. Compare raw→XRT bit-for-bit at both full shapes and the winner's slice; verify multiple deterministic inputs, change sentinels between correctness runs, and repeat after destroy/recreate to rule out stale output. Time from the same artifacts/layout as XRT. If a stage fails, preserve errno, ioctl fields, dmesg/firmware result if authorized, and contrast a captured same-artifact XRT sequence; work around private-WAIT absence with standard syncobj. A claim that XRT is fundamentally required needs a specific unreplicable signed/encrypted/init dependency; no such dependency is apparent in these sources.

**Firmware boundary:** driver `aie2_message.c::aie2_config_cu` encodes CU function plus shifted PDI device address into MSG_OP_CONFIG_CU. `aie2_init_exec_cu_req` copies ERT CU payload and CU index to MSG_OP_EXECUTE_BUFFER_CF; current drivers may use a firmware command-list wrapper selected by feature bits. Driver owns mailbox, PASID binding, firmware loading/security and scheduling. Bare userspace reproduces the **UAPI command**, not those kernel-internal firmware packets.

**Acceptance:** probe source exists, no-libxrt linkage/maps evidence, exact whole-output match, measured small and full/slice timings, reproducible commands and artifact hashes; or a stage-specific experimentally isolated blocker despite testing standard timeline wait and matching the installed source version. CPU reference generation and build work must never overlap GPU timing.

## 5. Bit-exact slice contract and ownership

Read-only source: `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`, `IU4_FOLD_RN` (182–190), `vec_dot_i4_x128` (194 onward), `gemm_iu4_body` (246 onward). The integer dot is reset for each 128-K half. Unsigned weight codes 0..15 times signed activation codes -8..7 fit exact i32 (`|C_half| ≤ 15360`); float conversion of C_half and exact code sum is exact. Half headers convert exactly to f32, subject to the device's exceptional/subnormal rules.

Required DAG, h0 then h1 in increasing 256-K group order:

`t1=RN(sc*d); p=RN(t1*float(C_half)); t2=RN(zp*d); term=RN_FMA(t2,float(s),p); sum=RN(sum+term)`.

No full-K integer reduction may replace these half dots. No reassociation, alternative FMA contraction, bf16 fold, or tolerance policy is authorized. The residual add/set output behavior must also match the chosen GPU entry exactly.

Each call is a transaction: weights and activation codes/headers are immutable while either device reads them; GPU and NPU own disjoint output rows; NPU raw partials are private until all half data are complete and synchronized; downstream consumers may observe Y only after both row owners and the required epilogue complete. A timeout is not permission to reuse a BO or overwrite NPU-owned output while the device may still write it. No production fallback/replay design is introduced by this probe.

Two numerical choices remain to be characterized: on-AIE f32 fold with proved instruction semantics, or GPU fold of the per-half i32 stream. The latter multiplies C traffic by K/128 and is not the cost of one ordinary i32 GEMM output. Weight expansion to an int8 slice mirror costs exactly `M_slice*K` additional bytes (GPU MQ4 remains resident); activation expansion is `K*N` bytes independent of M_slice. Zero-copy availability must be established with supported BO import/host registration and explicit visibility transitions, not inferred from unified physical LPDDR.

## 6. Verdict, projection, and ordered implementation gates

No tuned or raw-dispatch performance verdict is established yet. The stock numbers alone are not a universal kill of a tuned split design. A source-correct projection uses measured slice times, contention, preparation, waits and fold costs, and never scales the headline 38 TOPS i8-output number into an exact IU4 result.

For a preliminary linear model only, let G be full GPU time, D full-equivalent NPU time, s_g GPU slowdown fraction, s_n NPU slowdown fraction, H fixed publish/submit/join cost, and f NPU row fraction. Then `T(f)=max((1-f)*G*(1+s_g), f*D*(1+s_n))+H` before non-overlapped fold/copy cost, with `f*=G*(1+s_g)/(G*(1+s_g)+D*(1+s_n))`. Real choices must enumerate legal M-slices and measured NPU times, not rely on linearity.

For supplied 700 tok/s pp2048 and 78% GEMM share, use `R=700/(0.22+0.78*q)`, where `q=(1088*T_gate_new+512*T_down_new)/(1088*T_gate_old+512*T_down_old)` uses consistent baseline units. Oracle timings from another run are not an absolute reconstruction of the 700 tok/s campaign; report the normalization explicitly. FA2, GDN recurrence, bandwidth-bound kernels, decode, other architectures and production flags remain non-goals.

Implementation is contingent on measured gates and reviewer veto: artifact/ABI probe first; numeric half-dot/fold parity second; memory interoperability third; row-ownership/runtime integration only after split critical-path and contention measurements justify it. This document will freeze exact runtime symbols and executable slices before any proposed production work.
