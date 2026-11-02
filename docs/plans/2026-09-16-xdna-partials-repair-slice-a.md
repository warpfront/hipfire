# XDNA partials repair — slice A: correct the proven stream layout, isolate K5120

## Ownership / non-goals

Composer owns only hipx `/home/kaden/npu-screen/part-h/gen_ph.py` (generator), its regenerated `whole_array_ph.py`, and `smoke_taps.py`. No hipfire production code, no hipfire-beta, no commits, no formatter/linter/project suite. Main owns shared validation; reviewer owns veto. NPU is exclusively available; use kaden login shell for memlock. Do not touch pristine `whole_array_k_ref.py` or max-tune2. This slice does not claim to solve the still-unlocalized K5120 timeout.

## Measured root cause

`gen_ph.py` R8 / generated `_build_design.sequence` iterates row-pair → K512 group → column → two row blocks. Each block emits four K128 objects. Its monotonically incrementing `c_index` writes a flat contiguous stream in that order. R9 `device_order_to_canonical` instead assumes row-pair → column → row-block → all K128 segments. These agree only when K=512. The earlier analyzeK.py searched different halves/rows but kept column fixed, so it could not discover the column permutation.

Read-only experiment by XdnaArtifactMax against EXISTING `cfg-ph-K1024/{a.bin,b.bin,c_ph.bin}`: reshape raw into `[M/1024,K/512,8,2,4,512,64]`. For every `(pair,g,col,b,j)`, compare the `[512,64]` block against exact int64 `A[pair*1024+b*512:...,(g*4+j)*128:... ] @ B[(g*4+j)*128:...,col*64:...]`. Result: **0 / 4,194,304 mismatches**. All later fills and all columns in that saved run are exact. No device rerun and no source edit was needed for that result. The lost-later-fills hypothesis is disproven for K1024. K5120 remains unproven.

## Frozen interface / minimal change

Keep all compute/FIFO/tap/fence code unchanged initially, including CURRENT `wait=True` on A/B fills. The existing K5120 AOT `design.prj/aie.mlir:1302–1320` has no input issue_token; it predates that last source change. Only K1024 was tested with input waits, so a fresh K5120 run now is a meaningful changed experiment, not rerunning the same reported failure.

Replace generator R9's `device_order_to_canonical` body with the correct general mapping for the admitted geometry:

- H = `dev_flat.size // (M*N)`; G = H/4; `block_rows = 4*m`, `pair_rows = 2*block_rows`.
- Require `N == n*n_aie_cols`, `M % pair_rows == 0`, `H % 4 == 0`; this sidecar supports m128/n64/cols8 only.
- View raw as `(M//pair_rows, H//4, n_aie_cols, 2, 4, block_rows, n)`.
- Canonical `[M,H,N]` is `.transpose(0,3,5,1,4,2,6).reshape(M,H,N)`.
- Update R7/R8/R9 generated comments and docstrings that falsely say full-H block-major/halves-inner or fills occur only in first subgroup. Do NOT change emitted tap addresses or core loops in this slice.
- Update existing smoke_taps.py's synthetic round-trip generator to this layout. Distinct values must identify row, column, and half; exercise K1024 (G>1), not merely K512.

Before next change, the generator remains the sole source of truth: run `python3 gen_ph.py` in part-h. Decoder consumers are `harness_ph.py:106`, `validate_saved.py`, generated `_run_and_verify:754`, and smoke_taps.py; signatures stay unchanged.

## Commands / scoped proof

From a remote kaden login shell:

```sh
export PYTHONPATH="${PYTHONPATH-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source /home/kaden/npu-screen/mlir-aie/ironenv/bin/activate
source /home/kaden/npu-screen/mlir-aie/utils/env_setup.sh
cd /home/kaden/npu-screen/part-h
python3 gen_ph.py
python3 smoke_taps.py
K=1024 python3 validate_saved.py cfg-ph-K1024
python3 whole_array_ph.py --dtype_in i8 --dtype_out i32 --dev npu2 --n-aie-cols 8 -M 1024 -K 5120 -N 512 -m 128 -k 64 -n 64 --b-col-maj 0 --c-col-maj 0 --a-l1 2 --b-l1 2 --c-l1 1 --a-l2 2 --b-l2 2 --c-l2 2 --k-mt 512 --xclbin-path=cfg-ph-wait-M1024/design.xclbin --insts-path=cfg-ph-wait-M1024/insts.bin
python3 harness_ph.py --config-id cfg-ph-wait-M1024 -M 1024 -K 5120 -N 512 -m 128 -k 64 -n 64 --n-aie-cols 8 --k-mt 512 -w 1 -i 1 --outdir cfg-ph-wait-M1024
```

Create cfg-ph-wait-M1024 first. Run CPU-heavy compile supervised; do not leave unsupervised timeout jobs. Device command is via `ssh kaden@hipx 'sudo -n -u kaden -i ...'`. Preserve stdout/stderr and any driver timeout diagnostics. `harness_ph.py` uses JIT and does NOT execute the named AOT xclbin; report this explicitly. Same source/kwargs are necessary but insufficient to certify packaged PDI bytes. Later artifact replay must execute exact AOT output.

If K5120 completes and parity passes, continue this same source for M1024/2048/4096 at K5120 with 10 warm + 50 timed samples; save all raw ns and mismatch counts. Do NOT package yet. If it times out, stop further NPU experiments, preserve evidence, return exact phase/ERT state and fresh emitted IR to Main/Max for slice B. Do not call a timeout a parity result or claim wait=True fixed it without success.

## State invariants

Host A/B immutable and pinned through submission. For each group each A stream owns exactly two m×512 objects; each B stream owns two sequences of eight 64×64 objects; each C stream returns eight 512×64 i32 objects. Each core zeroes before exactly two k64 multiplies; no partial may mix K128 segments. All group outputs are awaited before task reuse. Current explicit input waits additionally witness each input descriptor completion before reuse. Group order is a layout choice, not corruption, because the core has no persistent accumulator across emitted K128 objects. A terminal successful submission produces exactly M*N*(K/128) i32 values; canonical decoding changes only the CPU interpretation, never the artifact arithmetic.

## Acceptance / abandon gate

Required: saved K1024 exact PASS under new decoder, CPU map covering G>1 and both row blocks, then fresh K5120 status with no invented timing. If full profiles pass, return n=50 median/p90/min/max NPU and E2E µs, TOPS=2*M*K*N/(median_us*1e6), exact write bytes `4*M*N*(K/128)` and extra vs dense `4*M*N*(K/128-1)`. Do not preserve extra input waits as a claimed performance optimization; this experiment only isolates completion semantics. If the timeout persists, abandon the four-drain fencing approach pending Max's bounded-BD redesign rather than trying arbitrary FIFO-depth/tile changes.

## Authoritative addendum: exact-AOT execution and the actual K5120 cause

This addendum supersedes the instruction above to run the unchanged JIT harness. Extend this slice's ownership to `harness_ph.py`. Require `outdir/design.xclbin` and `outdir/insts.bin`, instantiate `aie.utils.npukernel.NPUKernel(xclbin_path=..., insts_path=..., num_host_bos=3)`, and use `kernel(A_t,B_t,C_t)` for both warmup and timed calls. Remove now-unused compile kwargs. No JIT or missing-artifact fallback. Record absolute executed paths and SHA256 of both files outside the timed region.

Read-only cache inspection with exactly the harness compile kwargs computed `/home/kaden/.npu/cache/3baae72a8c336b8ff6b338e9`. Its `insts.bin` is 55,376 bytes, versus the newer sliced AOT design. Its cached `aie.mlir:1303,1308` still has full-K A `[1,10,128,512]` / length 655360 and B `[1,80,64,64]` / length 327680. `whole_array`'s bytecode only names `_build_design`, `iron`, `get_current_device`; installed `utils/compile/jit/_hash.py:138–148` hashes wrapper bytecode, not referenced global functions. Sources are empty and the cached dependency manifest contains no Python. Thus the alleged post-slicing K5120 repro actually reran the old full-fill artifact. The K1024 wait=True rerun is likewise not evidence that new input waits were exercised.

The cached artifact's lifetime violation is now proven with a scoped compiler experiment:

```sh
aie-opt /home/kaden/.npu/cache/3baae72a8c336b8ff6b338e9/input_with_addresses.mlir --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids
```

This exited 0. On physical shim `(0,0)`, A_L3L2_0's full-K MM2S tasks `%4/%6` receive BD0/BD1; its two row blocks total 80 partial objects. C_L2L3_7 shares the same shim and first receives BD2..5. After only eight objects drain, `TaskGroup.finish()` frees the still-active/queued input tasks without an input completion token. The NEXT group's C7 tasks `%84/%85` receive BD0/BD1, overwriting the unfinished A descriptors as S2MM output descriptors. On shim `(6,0)`, A_L3L2_3 and C_L2L3_5 exhibit the same BD0/BD1 reuse. The allocator is tile-wide, not separate per direction/channel. `AIEAssignRuntimeSequenceBDIDs.cpp:295–318` erases free operations after allocation: free is not a device wait. Four-dimensional taps have outer size 1 and hence repeat_count 0; this is not a repeat-count multiplication bug.

The fix is the existing balanced sliced-input schedule plus actual execution of the freshly built artifact; current explicit input waits make reuse ownership directly witnessed. Do not change the kernel geometry or FIFO depth merely because stale-cache runs appeared to reject this schedule. The original stale cache is evidence: preserve it, do not delete it.

After timed parity, exercise at least one changed A/B generation using weight-domain A codes 0…15 and activation-domain B codes −8…7, resetting P to a sentinel before submit; require exact parity again. This is not another timing campaign: it guards against stale-output/unchanged-input replay. Package only hashes that were actually run.
