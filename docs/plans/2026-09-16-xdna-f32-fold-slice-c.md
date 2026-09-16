# Slice C — proposed M2048 AIE hardware-precision f32-row experiment

**Status: specification only; NOT dispatched or authorized to run.** The user decides whether the conditional, roughly 4% end-to-end upside warrants this experiment. The passing i32 partials sidecar remains an immutable, performance-negative research artifact. This proposal neither changes its contract nor admits an approximate runtime path.

## 1. Decision and measured starting point

Test exactly one shape: NPU-owned gate rows **M=2048, K=5120, N=512**, with the original quantization and forty separate K128 integer dots. Fold those dots on the AIE and emit only final f32 rows. Do not replace four independently scaled half-dots by one K512 integer sum: the scale/zero and activation d/s terms differ by half, and the GPU's ordered rounding cannot be reconstructed from that sum.

Measured exact-partials service is 3696.305 µs median, 2.904906 equivalent TOPS, writing 167,772,160 bytes. The dense m128/k64/n64/k_mt512 M2048 reference in `hipx:/home/kaden/npu-screen/max-tune2/RESULTS.md:45–52` is 705.0 µs submit+wait / 816.9 µs host E2E. This is an **optimistic reference**, not a projection that includes the proposed fold. F32 output is 4,194,304 bytes, forty times smaller; total service is not expected to shrink fortyfold.

**Predeclared performance gate: complete NPU row-path service median ≤1640 µs**, including every per-invocation expansion, copy/visibility operation, NPU submit/wait, return transfer, final GPU scatter and join. Report p90, min/max and all samples. Passing the resident kernel alone is insufficient. The comparison is against a fresh same-window GPU-alone full gate measurement, nominally 1855–1856 µs, not against the throttled GPU during overlap. Under an ideal linear GPU-row model, M2048 leaves 1636.8 µs of GPU work and saves only 218.2 µs; serial overhead above that already prevents a win. The old 497.8 µs host-copy proxy would fail this gate. No free DMA overlap, zero-copy, dense speed, or serving gain is assumed.

## 2. Source map and immutable references

Read/reuse these exact patterns; do not edit their originals:

- `hipx:/home/kaden/npu-screen/max-tune2/whole_array_k.py::_build_design`, its A k_mt composition, `Worker.grid`, dense `sequence`, and C row-major join/drain. Keep the existing autoplacer rather than retrying the measured-slower explicit placement.
- `hipx:/home/kaden/npu-screen/part-h/gen_ph.py` R6 and generated `whole_array_ph.py::_build_design.core_fn`: two k64 integer matmuls per K128 dot, reset at every half. Reuse arithmetic, NOT the expensive partials drain schedule or JIT invocation.
- Generated `part-h/cfg-ph-wait-M2048/design.prj/mm.cc::matmul_vectorized_2x2_mmul`: C is blocked `[m/8,n/8,8,8]`, tile-major then row-major within each 8×8 tile. Its stores at 200–207 and input packing at 124–185 define the fold's lane mapping.
- `fold-probe/fold/fold_probe_vecmul.cc::fold_vecmul` at 41–51: working native vector `aie::mul(...).to_vector<float>()` plus separate `aie::add`. Do not inherit its scalar per-lane conversion/gather buffers or its single-dispatch workaround.
- `fold-probe/fold/results.json`, `fold_probe_vec.cc::fold_vector/fold_vector_fused`, and `fold_design.py::_build`: evidence of float-MAC all-zero output, missing scalar fmaf, and repeated-launch failures; not passing reusable implementations.
- Local worktree `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip::IU4_FOLD_RN` lines182–190, `vec_dot_i4_x128`, `load_iu4_tile` lines144–158: authoritative GPU arithmetic and MQ4 header/code interpretation.
- `crates/hipfire-runtime/examples/tmp_halo_iu4_oracle.rs::{load_mq4v2,dtoh_raw,launch_once,run_case}`, `LF_SET`, `IU4_SRC`, `MODULE`: existing fixture/GPU launch pattern. `run_case:274–287` invokes the real `Gpu::ensure_int4_mmq_x`, then snapshots the prepared 72-byte records. `run_case:324–345` executes LF16 and maps GPU output index to row/token. Consume the actual pinned GPU binary/cache identity, recording SHA256 and kernel symbol; do not substitute a CPU-only oracle.
- Toolchain `python/iron/buffer.py::Buffer`: a buffer passed in one Worker's fn_args is core-owned and auto-placed with that Worker. `aie.utils.npukernel.NPUKernel` executes exact AOT paths. Global-helper edits are NOT reliably covered by the current IRON JIT cache key; never use the stale-cache benchmark path.

## 3. One bounded design: m64 retile with embedded metadata

Select **m64/k64/n64, 4×8 cores, k_mt512**, not an m128 lifetime experiment. At m128, persistent f32 output plus an i32 half-dot alone occupy all 64 KiB of core L1. Pretending an i32→f32 reinterpretation removes one array is incorrect: both values are live while folding. M64 has an explicit legal storage budget and reuses the known integer microkernel; dense-class throughput remains unproven.

### Frozen input records: no extra shim channels

Do NOT add four weight-header plus eight activation-header streams: A/B already use twelve MM2S streams across eight shims, and twelve more exceed the sixteen shim MM2S channels. Embed headers in the same objects as their codes.

A half-record and B half-record are each **8704 bytes**, aligned at least32 bytes:

| byte range | A half-record (one64-row weight tile) | B half-record (one64-token tile) |
|---|---|---|
| 0..4095 | first K64 codes, native mm A blocked `[8,8,8,8]` layout | first K64 codes, native mm B blocked `[8,8,8,8]` layout |
| 4096..8191 | second K64 codes, same layout | second K64 codes, same layout |
| 8192..8447 | sc[64], exact f16→f32 header conversion | d[64], original prepared-X f32 bits |
| 8448..8703 | zp[64], exact f16→f32 header conversion | s[64], original prepared-X exact int32 sums |

A codes are unsigned nibble values0…15 stored in int8. B codes are sign-extended −8…7, never requantized. For each 256-K weight group at `((row*(K/256)+kb)*136)`, half h0/h1 header is the u32 at `+4*(half%2)`, low16=sc and high16=zp; codes come from the existing MQ4 nibble layout. Read and reuse the source, rather than assuming nibble order. X source is `block_i4_128` at `(half*N+token)*72`: d bits, signed exact s, packed codes. Confirm byte-for-byte conversion against the actual GPU-prepared fixture.

A DDR mirror order: `[row_tile=M/64][kg=K/512][half_in_kg=4][8704 bytes]`; total **11,141,120 bytes**, prepared once for immutable weights. A L2 object comprises four consecutive half-records =34816 bytes; forward/split into four8704-byte L1 objects without the old code-only dimensions-to-stream permutation, since the records are prepacked. This retains K512 shim batching.

B DDR order: `[col_tile=N/64][half=K/128][8704 bytes]`; total **2,785,280 bytes**. B L2 and L1 objects are one half-record. B pack/expansion is dynamic and charged every invocation. Reuse the original A row broadcast and B column broadcast topology. Metadata is an input, not a baked scale, even though it shares an argument BO with codes.

Artifact ABI remains three BO address slots A20/B28/Y36, but the BO formats are DISTINCT from the partials sidecar. C/Y DDR output is ordinary row-major f32 `[2048,512]` (4,194,304 bytes). Final GPU publication transposes/scatters it to `Y[token*full_M+row_start+local_row]`; its cost belongs in the complete-path timer. Do not claim a ready GPU output merely because host DDR contains row-major floats.

### Core state and schedule

Per core:

- A half-record FIFO depth2:17408 bytes; B half-record FIFO depth1:8704 bytes.
- Private i32 half-dot scratch `[64,64]`:16384 bytes, one `Buffer` per Worker, never shared.
- Final f32 C FIFO producer object `[64,64]`, depth1:16384 bytes; this object IS the persistent accumulator.
- Stack0xD00:3328 bytes. Nominal total **62208 bytes**, leaving3328 bytes for alignment/spills/other placement. The compiled map, not this arithmetic, is the legality gate. No per-lane float gather arrays or second output copy on stack.
- A-L2 depth2/B-L2 depth2/C-L2 depth2 start from existing patterns. All actual MemTile allocations must fit512 KiB after routing/joins. No depth sweep is authorized.

For each output tile acquire C once, initialize all lanes to +0. For h=0..39 in order: acquire one A and one B half-record; zero i32 scratch; call the existing64×64×64 int8 matmul twice, using offsets0 and4096; apply the hardware-policy fold with that half's metadata into persistent C; release both half-records only AFTER the fold has consumed their headers. After h39 release C exactly once. No K128 or K512 partial reaches DDR in the production experiment artifact.

C scratch/accum both use mm's blocked index `(((r/8)*8+c/8)*64+(r%8)*8+c%8)`. A fold vector of16 contiguous lanes represents two rows×eight columns; broadcast the corresponding two row headers and repeat the eight d/s lanes correctly. Validate this mapping with row-, token- and half-distinguishing inputs, not only uniform ones.

Use native vector integer-to-float conversion for bounded C/s where the pinned API supports it; verify emitted lowering. Integer values fit exactly in f32. Do not silently replace vector conversion with thousands of scalar softfloat calls and still project dense speed. A conversion limitation is a measured design issue, not permission to change quantization.

Runtime sends full-K input sequences and one final C drain for each dense output transfer block. Restore the dense sequence's bounded-BD grouping; every C completion now implies consumption of all40 halves for the covered rows. Retain explicit input completion waits before freeing descriptors. Physical shim BD ids are shared across direction/channel: never free in-flight full-K inputs against a short partial drain. Outer tap repeat counts must equal the actual object count, with four-dimensional normalization and no implicit double counting.

### Declared arithmetic policy

Name the experiment policy **`aie2p_f32_unfused_half128_ordered_v1`**. For each output/half:

`t1=HW_MUL(sc,d); p=HW_MUL(t1,float(C128)); t2=HW_MUL(zp,d); z=HW_MUL(t2,float(s)); term=HW_ADD(z,p); sum=HW_ADD(sum,term)`.

Use the established separate native vector mul/add spelling, with contraction disabled, increasing h order, and f32 stores/loads as required by actual hardware behavior. This is deliberately NOT IU4_FOLD_RN's fused `term=RN_FMA(t2,s,p)` and is NOT certified IEEE-RN for subnormal/signed-zero cases. Do not call it exact, bf16, a K512 fold, or a fused implementation. No extra quantization is authorized.

## 4. Float-MAC and stale-output hazard gate

Prior `aie::mac(accfloat,f32,f32)` variants returned all zeros despite ERT COMPLETED; scalar fmaf did not link. The chosen path avoids floating MAC using the working mul→vector plus separate add; integer mmul MAC remains required. This does not claim to fix the compiler's float-MAC implementation.

Before full timing, retain the exact AOT source/IR/ELF/disassembly/PDI/insts hashes and prove the chosen float fold actually runs:

1. Run row/token/half-distinguishing fixtures with known nonzero final values, both zero-point and scale contributions, positive/negative d, cancellation, all-zero followed by nonzero, and a changed A/B/header generation. Use nonzero output sentinels before every diagnostic run.
2. In the SAME context and BOs perform at least100 submissions alternating two independently expected input generations, plus a fresh-context repeat. Check every output each time outside any timed region. No one-process-per-dispatch workaround, alternating correct/zero drains, stale outputs, unmodified sentinel, or partial coverage is accepted.
3. Stage-diagnostic outputs may expose t1/p/t2/z/term/sum in a SEPARATE diagnostic build; final artifact outputs only Y. Compare simple exactly representable cases against independent arithmetic and compare real fixtures against the pinned GPU. ERT COMPLETED or merely nonzero output is never a numerical PASS.
4. Inspect emitted fold code for unsupported `fmaf`, unintended float MAC contraction, and scalar conversion/multiply helper hot loops. Use the known float-MAC failure as a regression target, not an instruction to repeat a known-broken spelling. If source changes do not change executed-artifact hashes, stop and fix artifact selection before reasoning about arithmetic.

## 5. Fixture, numerical and complete-path gates

### Fixture contract and GPU oracle

Use the actual qwen3.8-27b.mq4-xt layer0 `gate_proj.weight` from model SHA256 `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7`; full_M17408, slice row_start15360,row_count2048. Follow the existing oracle's real quantizer and LF16 SET launch; do not edit/rebuild the pinned oracle during measurement windows. A standalone fixture companion outside the production checkout may reuse its source pattern and same kernel artifact. Capture the packed MQ4 slice, exact prepared-X bytes and generation, GPU rows, model/tensor identity, compiler/kernel hashes and seed. Fixture paths and hashes are inputs to every subsequent report.

For each of2048 rows report over512 tokens:

- `max_abs_error[row] = max_c |Y_npu[row,c]-Y_gpu[c,full_row]|`;
- `mean_abs_error[row] = sum_c |difference| /512`.

Emit all2048 pairs, global maximum/mean, finite/nonfinite counts and signed-zero/subnormal diagnostic buckets. Check A/prepared-X immutability and row canaries; comparing the wrong row stride is a failure. Require finite outputs wherever the real fixture's GPU outputs are finite; do not drop or sanitize exceptions. No numerical tolerance was preapproved here: measurements are required, but nonzero errors cannot be declared acceptable by inventing an epsilon after seeing them. User/reviewer must approve predeclared model-level limits before admission.

### Timers and transport

Preallocate/reuse all buffers, contexts and commands. Weight record preparation is load-time and reported separately. For each of50 timed samples use a fresh prepared-X generation and charge:

`GPU prepared-X ready → actual source visibility/copy → nibble expansion + B-record pack → XDNA visibility → submit/wait → output visibility/copy → GPU row scatter → GPU completion/join ready`.

If packing runs on GPU, include its event/time and actual transfer. If packing runs on CPU, charge copying prepared-X to a readable allocation and its interference. Do not assume hipMalloc is cheaply CPU-readable, use a memcpy microbenchmark as actual staging time, or time a pre-expanded B repeatedly and call that complete service. The common quantizer may be excluded only if it is also excluded from the GPU baseline and starts from the same ready-generation boundary. No reference computation, NumPy validation, compile, allocation, or logging in timing regions.

Report resident submit+wait separately (host perf_counter around XRT launch+wait, NOT hardware cycles), complete-path median/p90/min/max and all50 samples. The hard experiment gate is **complete-path median ≤1640 µs**. A complete-path implementation unavailable on the actual host is a failed/unmeasured gate, not a zero-cost placeholder. Recheck GPU-alone→overlap→GPU-alone with the same fixture/cache, full_M stride and true join only after the isolated candidate passes. Require a net full-operation win beyond observed drift; ≤1640 alone is necessary, not sufficient, because preparation may serialize with remaining GPU work. Row-range partition remains GPU[0,15360), NPU[15360,17408).

### Admission is a later separate gate

Even a timing/fixture PASS must remain experimental. Admission requires a model-specific numerical certificate under policy `aie2p_f32_unfused_half128_ordered_v1`, per-layer diagnostics, output/logit max/mean/p99 KLD against the same GPU-only model, and the existing `scripts/serve_harness.py --model <same-model>` prompt battery including long-prefill/reuse. Same prompts/tokens/settings and predeclared release thresholds are mandatory. No KLD/serve threshold is invented by this plan; absent limits or reports blocks admission. No NPU runtime production edits, model serving, or KLD campaign are authorized merely by dispatching this kernel experiment.

## 6. Frozen composer slices (dispatch only after user approval)

Root for any later authorized work: **hipx `/home/kaden/npu-screen/f32-fold-c/`**, never part-h, max-tune2, hipfire-beta or hipfire production sources. No commits, formatters, linters, or project-wide builds/tests. Main owns shared validation and NPU/GPU/build scheduling; independent reviewers own the final veto.

**C1 — AIE artifact owner:** create `whole_array_f32row.py::_build_design/whole_array/main`, `fold_f32row.cc::{zero_f32_tile,fold_half128_f32_unfused}`, and the diagnostic variant. Consume exactly the packed record/ABI/policy above. Use `kernels.mm(dim_m=64,dim_k=64,dim_n=64,input_dtype=i8,output_dtype=i32)`, core-private Buffer, dense C join/drain and exact AOT compilation. Deliver real memory maps and executable artifact hashes, not scaffold. No fixture/GPU harness ownership.

**C2 — fixture/transport owner, independent of C1 until device execution:** create `fixture/` standalone companion using the existing GPU oracle source pattern, plus `harness_f32row.py::{pack_weights,pack_activations,run_exact_aot,compare_rows,time_complete_path}`. Consume the frozen records and three-BO ABI. Prepare independent references/pack round trips while C1 builds; later consume its hash-frozen artifacts. Own actual producer/consumer transport and fresh-generation timings, never a simulated copy path. No NPU design edits. C1 and C2 may work concurrently on separate files; NPU/GPU runs are serialized/coordinated through Main.

After both deliver, an independent reviewer checks half/header mapping, signedness, lifetime/BD/replay invariants, complete timing charge, immutable artifact identity and honest numeric-policy labeling. No further architecture abstraction or production integration belongs to these slices.

Required future CLI contract (commands shown now are a SPECIFICATION, not existing executables or executed proof):

```sh
# hipx kaden login shell; build phase only
export PYTHONPATH="${PYTHONPATH-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source /home/kaden/npu-screen/mlir-aie/ironenv/bin/activate
source /home/kaden/npu-screen/mlir-aie/utils/env_setup.sh
cd /home/kaden/npu-screen/f32-fold-c
python3 whole_array_f32row.py -M 2048 -K 5120 -N 512 -m 64 -k 64 -n 64 --n-aie-cols 8 --k-mt 512 --xclbin-path=cfg-f32-M2048/design.xclbin --insts-path=cfg-f32-M2048/insts.bin
# only after fixture export and Main's device window
python3 harness_f32row.py --fixture fixture/manifest.json --xclbin cfg-f32-M2048/design.xclbin --insts cfg-f32-M2048/insts.bin --mode numeric-reuse --replays 100
python3 harness_f32row.py --fixture fixture/manifest.json --xclbin cfg-f32-M2048/design.xclbin --insts cfg-f32-M2048/insts.bin --mode complete-path --warmup 10 --iters 50
```

The fixture companion must provide an export mode and a timed pack/publish/scatter mode backed by real GPU operations; a saved GPU array cannot stand in for timed live transport.

## 7. Ownership invariants, identity and park rule

| transition | required before → after invariant |
|---|---|
| Model fixture → weight mirror | identified MQ4 bytes immutable; exact nibble/header extraction → immutable packed A tied to model/tensor/row range/hash |
| Prepared-X → packed B | producer completed, prepared generation pinned → codes/d/s derive from that exact generation; old B never reused as fresh work |
| Packed buffers → submitted | visibility complete; A/B/Y BOs owned by invocation → every BO/address pinned until terminal completion |
| Half h acquired → folded | i32 scratch zero; both K64 blocks and metadata belong to same h → one ordered fold; only then release input records |
| Tile initialized → C publication | C privately accumulates from +0 through h0…h39 → exactly one final float tile, no partial publication |
| NPU complete → GPU rows ready | terminal ERT plus real visibility → final scatter writes only owned rows at full_M stride; GPU fence proves readiness |
| Both partitions → consumer | disjoint row owners complete → one complete gate output; no consumer observes half-published rows |
| Timeout/error → quiescent | no retry/overwrite/unmap while hardware may write → supported terminal/reset/quiescence before reuse; failed output never published |

Distinct identity is mandatory: `cfg-f32-M2048/` artifacts and a separate experiment metadata document containing arithmetic policy, dtype/layout, packed-header contracts, model/tensor/row slice, code/toolchain hashes, sample provenance and numerical report. Do NOT overwrite `sidecar/qwen3.8-27b.mq4-xt.xdna.zip`, reuse its P layout name, or label the proposed f32 artifact as the existing v1 int32 manifest. Frozen runtime v1 lacks an explicit numerical certificate/policy field: shipping approximate rows requires a separately reviewed admission/manifest evolution, not an unknown-field injection or name-only loophole.

Only gfx1151+npu5, eager gate SET, M2048/K5120/N512 are in this experiment. Flag-off and other architectures remain unchanged; no gfx1201/gfx1100 fallback, down/ADD, decode, FA2, GDN/KV state, graph capture, retained replay, automatic split, CPU spillover, or hidden numeric-policy substitution.

**Park immediately** if the selected design cannot fit the compiled memory map, exact integer half dots or metadata packing fail, repeated exact-AOT submissions exhibit zero/stale/incomplete output, the chosen mul/add path does not execute with the declared policy, complete-path median exceeds1640 µs, or a same-window joined split loses against GPU-alone beyond drift. No tile/depth/kernel search or compiler redesign is preauthorized. Preserve a measured negative. Even if those gates pass, park admission until user-approved model numerical limits, KLD and serve_harness gates pass. The dense705 µs anchor plus unknown retile/fold/transport costs is the only defensible projection; a model serving speedup remains unmeasured.
