# gfx1201 BF16 singleton reference objects (A3 Phase A)

Owner: A3 reference worker (`BatchInvariantA.A3Bf16Twin.Bf16References`). No GPU was used. Nothing here
was run on hardware; every arithmetic claim comes from the disassembly (see `true-cb/DAG-GFX1201-BF16.md`). The lift/emulator probe below was run by the integrator, not by this worker.

## Files

| file | what | sha256 |
|---|---|---|
| `gemv_bf16_xf32.hip` | copy of `kernels/src/gemv_bf16_xf32.hip` (module `gemv_bf16_xf32`, source of `kernels::gemv_bf16_xf32_src(false)` = `GEMV_BF16_XF32_SRC`) | `80ca9c63f6d74d061213fa4013f6a1da6f87c024ba60d145113f00d6c17bef42` |
| `gemv_bf16_xf32.gfx1201.hsaco` | uncompressed HIP bundle (`hipv4-amdgcn-amd-amdhsa--gfx1201`), 1 kernel `gemv_bf16_xf32` | `a94c1c43f95811c81213a42a4d8f4489fbbc3ec0639e4a5837b3298eaf219601` |
| `qwen4_gemv_bf16_xf32.hip` | copy of `kernels/src/qwen4_gemv_bf16_xf32.hip` (`QWEN4_GEMV_BF16_XF32_SRC`, module `qwen4_gemv_bf16_xf32`) | `995b0bb6355dedc7edca0660c95cf9a39ede8441d2d31855f39f5a7614faed25` |
| `qwen4_gemv_bf16_xf32.gfx1201.hsaco` | bundle, 30 kernels (see below) | `a610695fa6df3aaaf2771d98416fcfe1e229948b8ccae868bf03eed001850325` |
| `gemv_bf16_xf32.gfx1201.dis`, `gemv_bf16_xf32_x4.gfx1201.dis`, `gemv_bf16_xf32_k4.gfx1201.dis` | `llvm-objdump -d --mcpu=gfx1201` text of the three singleton symbols (trimmed after `s_endpgm`; line numbers are the DAG-doc citations). `gemv_bf16_xf32.gfx1201.dis` is from the shipped single-module object; the `gemv_bf16_xf32` symbol of the qwen4 object is instruction-text identical (only addresses/pad differ) | — |
| `analyze_rows.py`, `rows_static_analysis.txt` | static (no-execution) comparison of every `x4_rows_r2..r8` / `k4_rows_r2..r8` symbol against the singletons; output committed | — |
| `gemv_bf16_xf32_k4_rows_r2.gfx1201.dis` | disassembly of the HIP HC rows incumbent r2 (VALU epilogue citation) | — |
| `trace_fma_chains.py` | read-only symbolic FMA-chain tracer used for the DAG; `python3 trace_fma_chains.py gemv_bf16_xf32_x4` reads the fixture `.dis` beside it | — |
| `disassemble_reference.sh` | unbundle + `llvm-objdump` of a checked-in `.hsaco` (reproduces the ELF shas above; verified) | — |
| `compile_reference.sh` | the exact reference compile (compile only; `set -euo pipefail`) | — |

Device ELF (unbundled with `clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1201`):
`gemv_bf16_xf32.elf` sha256 `a628c16889a4dfb0466e3a7ee6506356271c551746c4afbabf253b28a92109a7`,
`qwen4_gemv_bf16_xf32.elf` sha256 `b1b97370b7afefac5add36b45ad5732202c8648a3e03f3e6f1dbcf46177a665b`.

`lift_object` consumes the bundles (`.hsaco`) as checked in.

### Kernels in `qwen4_gemv_bf16_xf32.gfx1201.hsaco` (symbol order)
`gemv_bf16_xf32`, `gemv_bf16_xf32_k4`, `gemv_bf16_xf32_k4_rows_r2..r8`, `gemv_bf16_xf32_k4_rows_tiled_r2..r8`,
`hyper_write_norm_f32`, `gemv_bf16_xf32_bf16_scaled_add`, `gemv_bf16_xf32_x4`, `gemv_bf16_xf32_x4_rows_r2..r8`,
`gemv_bf16_xf32_x4_rows_wide_r16`, `..._wide_r32`, `gemv_bf16_xf32_k4_rows_wide_r16`, `..._wide_r32`.

## Toolchain

`/opt/rocm/core/bin/hipcc` -> `/opt/rocm/core-10.1/bin/hipcc` (sha256 `0605893237d8c97c4cb2ecfb7a13445f0f9143005cdeb5c8ac787f1eb639a764`)

```
HIP version: 7.16.26385-0000000
AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git d6f6cb691863e9258034ab48c1dfce21689d5606)
Target: x86_64-unknown-linux-gnu
```
ROCm `.info/version` = 10.1.0. The runtime toolchain id folded into the hot-cache key is the first line, `HIP version: 7.16.26385-0000000`.

## Exact compile argv (reference)

Runtime recipe (`crates/rdna-compute/src/compiler.rs`):
`core_hipcc_flags("gfx1201")` (`:397`) = `--genco --offload-arch=gfx1201 -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6`;
then `hipcc_passthrough` (`:1526`): `--rocm-path=<root> --hip-path=<root>` (only when the root has `include/hip/hip_runtime.h`), `-I<root>/include`, `extra_flags` (empty), `module_flags` (see below), per-kernel `// HIPFIRE_COMPILER_FLAGS:` tags (none in either source); then `-o <obj> <src>`.

Module flags for `gemv_bf16_xf32` and `qwen4_gemv_bf16_xf32` on gfx1201: `module_flags_for` (`:683`) returns none (neither is in the gdn list, no gfx1151 cumode) and `scheduler_profile_for_module("gfx1201", name)` (`kernels.rs:5075`) = `SchedulerProfile::Default` (no llvm args). So there are no module/scheduler flags.

Reference command actually run (`ROCM_PATH=/opt/rocm/core`), see `compile_reference.sh`:

```
/opt/rocm/core/bin/hipcc --genco --offload-arch=gfx1201 -O3 --no-offload-compress -fuse-cuid=none \
  -mcode-object-version=6 --rocm-path=/opt/rocm/core --hip-path=/opt/rocm/core -I/opt/rocm/core/include \
  -o <m>.hsaco <m>.hip
```
`-fuse-cuid=none` makes the bytes independent of input path/argv, so the include/root spelling cannot change them. The runtime's resolved `<root>` (`/opt/rocm` vs `/opt/rocm/core`) was not observed (no runtime process was run); the object-sha equality below is the evidence that it does not matter.

## Hot-cache cross-check (`~/.hipfire_kernels/gfx1201/`)

| entry | `.hip` equals current source | object sha256 | toolchain (radiowave json) | equals reference |
|---|---|---|---|---|
| `gemv_bf16_xf32.249b031d8c4c842a` (2026-10-08) | yes (sha 80ca9c63…) | `a94c1c43…219601` | clang 24.0.0git d6f6cb691863, HIP 7.16.26385 | **identical** |
| `gemv_bf16_xf32.493e203a543aa941` (2026-09-10) | yes | `a7e8f4846a03…ac45ac` | clang 23.0.0git, HIP 7.15.26333 | differs (older toolchain; stale key) |
| `gemv_bf16_xf32.5ab13e23a44d7412` (2026-10-01) | yes | `6d7c479235f0…517ec6a` | clang 23.0.0git, HIP 7.15.26333 | differs (older toolchain) |
| `qwen4_gemv_bf16_xf32.2360726eb8cfd46d` (2026-10-16) | yes (sha 995b0bb6…) | `a610695f…850325` | clang 24.0.0git d6f6cb691863 | **identical** |
| `qwen4_gemv_bf16_xf32.070b6bfd…`, `.2ae06a7e…`, `.7f2dd132…` | no (older source revisions) | — | — | not current |

So the current-source, current-toolchain runtime objects are byte-identical to the reference compile. The cache key hash itself (`KernelCompiler::cache_hash`) was not recomputed here (no cargo run).

`radiowave.json` per-kernel resources (`.note` metadata, equal to the table the DAG doc uses): `gemv_bf16_xf32` v26/s18, `_x4` v102/s28, `_k4` v106/s20; none spill (`private_segment_fixed_size = 0`).

## Dispatch binding at the production sites

Documented in `true-cb/DAG-GFX1201-BF16.md` §0. Short form: rows==1 binds `gemv_bf16_xf32` (module `gemv_bf16_xf32`, single-matrix) and `gemv_bf16_xf32_x4` (module `qwen4_gemv_bf16_xf32`, MoE router/selector/shared); HC rows==1 binds `gemv_bf16_xf32_k4`.

## Lift / emulator probe (reported by the integrator; not re-run by the reference worker)

A standalone `rustc` probe (read-only A2 base-core libs, no cargo, no GPU) ran `peacemaker_lift::lift_object` and `Emulator::new` on both bundles:

- **Both bundles lift** (`lift_object(.., Frontend::Hipcc)` succeeds for `gemv_bf16_xf32.gfx1201.hsaco` and `qwen4_gemv_bf16_xf32.gfx1201.hsaco`).
- `Emulator::new` fails with a missing-semantics error: single `gemv_bf16_xf32` and HC `gemv_bf16_xf32_k4` (and every HC rows incumbent `k4_rows_r2..r8`): `v_dual_mov_b32`; `gemv_bf16_xf32_x4` and every `x4_rows_r2..r8`: `s_load_b512`. The ordinary kernels' unsupported ops are integer/control only; HC additionally has scalar-FP ops (`s_add_f32`, `s_mul_f32`, `s_rndne_f32`, `s_fmamk_f32`, `s_cmp_*_f32`, `s_cvt_i32_f32`) and `v_s_exp_f32` with no row.
- These are first-failure messages per symbol; the full missing-op union per kernel is in `true-cb/DAG-GFX1201-BF16.md` §6 (heuristic name match against `semantics.tbl`, may over-report ops the lifter names differently).
- Consequence: the reference singletons cannot yet be executed by the emulator. Per contract §4.1 the CPU reference falls back to the DAG transcription and the numeric claim is left to H0.

## HIP rows incumbents vs singleton

Static structural comparison (no execution): `analyze_rows.py` (reads the full disassembly from `disassemble_reference.sh`) and its retained output `rows_static_analysis.txt`; verdicts in `true-cb/DAG-GFX1201-BF16.md` §5. `gemv_bf16_xf32_k4_rows_r2.gfx1201.dis` is the VALU-epilogue citation file.
