# gfx1201 MQ6G256V2 singleton and HIP-rows reference fixtures

Binding contract: `/home/kaden/qcal/release-0.4.3/true-cb/CONTRACT-A.md` §1, §4. DAG: `true-cb/DAG-GFX1201-MQ6.md`.
Worktree `release-0.4.3/bi-a1`, branch `bi/a1-mq6`, base `8e328f6c3`.

## Production binding (source of truth: `crates/rdna-compute/src`)

| Route | Launcher | Module | Symbol | Source | Geometry |
|---|---|---|---|---|---|
| MQ6 single | `Gpu::gemv_mq6g256v2` (`gemv.rs:22788`); module via `kernels::gemv_mq6g256v2_for_arch` (`kernels.rs:6994`): gfx1201 -> `(GEMV_MQ6G256V2_SRC, "gemv_mq6g256v2")`, module name `gemv_mq6g256v2_mq6v2` | `gemv_mq6g256v2_mq6v2` | `gemv_mq6g256v2` | `kernels/src/gemv_mq6g256v2.hip` | grid `[M]`, block 32, kernarg 32 B (`A@0 x@8 y@16 M@24 K@28`) |
| MQ6 x4 | `Gpu::gemv_mq6g256v2_x4` (`gemv.rs:22841`) | `qwen4_gemv_mq6g256v2` (`QWEN4_GEMV_MQ6G256V2_SRC`) | `gemv_mq6g256v2_x4` | `kernels/src/qwen4_gemv_mq6g256v2.hip` | grid `[M0+M1+M2+M3]`, block 32, kernarg 92 B (`A0..A3@0..24 x@32 y0..y3@40..64 M0..M3@72..84 K@88`) |
| HIP rows incumbent (r2..r8, x4_r2..r8; wide r16/r32) | `Gpu::gemm_mq6g256v2_f32_rows` (`gemv.rs:22902`), `gemm_mq6g256v2_f32_rows_x4` (`gemv.rs:20655`) | `gemm_mq6g256v2_f32_rows` | `gemm_mq6g256v2_f32_rows_r{2..8}`, `..._x4_r{2..8}` (+ wide) | `kernels/src/gemm_mq6g256v2_f32_rows.hip` | grid `[M]` / `[sum Mi]`, block 32; kernarg 32 B / 92 B |

The `qwen4_gemv_mq6g256v2` module also exports `gemv_mq6g256v2` (same body as the single module; vector stream identical, see DAG §6).
`gemv_mq6g256v2_x4` is reached by `Gpu::gemm_mq6g256v2_f32_rows_x4` at `rows == 1` groups.

## Compile recipe (exact runtime)

`compiler.rs` `core_hipcc_flags("gfx1201")` + `hipcc_passthrough` + `-o OBJ SRC`:

```
/opt/rocm/core/bin/hipcc --genco --offload-arch=gfx1201 -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6 \
  --rocm-path=/opt/rocm/core-10.1 --hip-path=/opt/rocm/core-10.1 -I/opt/rocm/core-10.1/include -o OBJ SRC
```

- extra flags (`HIPFIRE_HIPCC_EXTRA_FLAGS` / `FeatureFlags::hipcc_extra_flags`): empty on gfx1201 (the only gfx-specific appends are gfx1100).
- `module_flags_for("gfx1201", name)`: empty for all three module names (only gdn_chunk_* modules get flags).
- `scheduler_profile_for_module("gfx1201", name)` (`kernels.rs:5075`): `Default` for `gemv_mq6g256v2_mq6v2`, `qwen4_gemv_mq6g256v2`, `gemm_mq6g256v2_f32_rows` (only three gfx1201 mmq modules get `IterativeIlp`) -> no `-mllvm` args. Hot-cache `radiowave.json` agrees (`scheduler_profile: default`).
- per-kernel `HIPFIRE_COMPILER_FLAGS:` magic comments: none in any of the three sources.
- Verified the ROCm root spelling does not change the bytes: `--rocm-path=/opt/rocm/core-10.1`, `/opt/rocm/core` and `/opt/rocm` all produced identical objects.
- Toolchain: `hipcc --version`: `HIP version: 7.16.26385-0000000`, `AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git d6f6cb691863e9258034ab48c1dfce21689d5606)`, InstalledDir `/opt/rocm/core-10.1/lib/llvm/bin`.
- Script: `rebuild.sh [OUTDIR [ROCM_ROOT]]` in this directory reproduces the three `.hsaco` files.

## Files

`*.hsaco` = hipcc output verbatim (clang offload bundle, targets `hipv4-amdgcn-amd-amdhsa--gfx1201` + `host-x86_64-unknown-linux-gnu-`). `*.elf` = the gfx1201 member unbundled with `clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 --unbundle` (raw ELF64 AMDGPU, feed this to `lift_object`). `*.disasm.txt` = `llvm-objdump -d --mcpu=gfx1201` of the ELF (line citations in the DAG doc refer to these files).

| Fixture | Source sha256 | hsaco sha256 | elf sha256 |
|---|---|---|---|
| `gemv_mq6g256v2` (module `gemv_mq6g256v2_mq6v2`, sym `gemv_mq6g256v2`) | `50af54313e803ae12013fc6c0cdde93889390debbdd69936f7c566bf5a438016` | `18b2536f2c4f6e01505d214f5816d9703af8880ea5a8063cb0ff23fc6c27df5d` | `d141a32fa1f2d7ad1aaeec63d63afa48ad6d53c3f512e9237f9e307f85094023` |
| `qwen4_gemv_mq6g256v2` (syms `gemv_mq6g256v2`, `gemv_mq6g256v2_x4`) | `7f1d4a49e921e57b21d3d669d517742c039e18c5fbdcaa771834ec9a1def28bc` | `18c105a770a1f073da20d84d4a05aba2e393fb56dd31a254ce48812f2b1cf4d3` | `9f5ec178ea5618b3f6bd0552a9d8d3836519ec4dee9bc55a25f58257aa842959` |
| `gemm_mq6g256v2_f32_rows` (18 syms, below) | `a93dfdbcfeb46ebae8641177984abc1074eabe161041729b498dc5f18ac91abe` | `cf64fc627308e1defa6000d3adbc2b087a517bd0d04eb1ea3f29bbe84eb58626` | `3cb4d9b3a4b2b0ad3ccb1721640af47ee44829037b0d68e3b2d05be5ecffc6c5` |
| `gemv_mq6g256v2.disasm.txt` | | | `1159fd1d56918ad89699325630efcc32326d9b7cee23eb805accf163e38a3a7c` |
| `qwen4_gemv_mq6g256v2.disasm.txt` | | | `fb2de340a09375424d26d4854b64818858419b40e8d2146a9efa095fd70d517b` |

Symbols in the rows object: `gemm_mq6g256v2_f32_rows_r2..r8`, `gemm_mq6g256v2_f32_rows_x4_r2..r8`, `gemm_mq6g256v2_f32_rows_wide_r16/r32`, `gemm_mq6g256v2_f32_rows_x4_wide_r16/r32` (wide kernarg 36/96 B, LDS 8192 B).

## Hot-cache cross-check (`~/.hipfire_kernels/gfx1201`, toolchain d6f6cb69, files dated 2026-10-08 / 2026-10-16)

| Hot object | `.hip` == current source | `radiowave.json` source_sha256 | hot hsaco sha256 | equals fixture |
|---|---|---|---|---|
| `gemv_mq6g256v2_mq6v2.18ae38b13b48c8a8.hsaco` | yes (`cmp`) | `50af5431...` | `18b2536f...27df5d` | **yes** |
| `qwen4_gemv_mq6g256v2.27843e484b1a18e8.hsaco` | yes (`cmp`) | `7f1d4a49...` | `18c105a7...cf4d3` | **yes** |
| `gemm_mq6g256v2_f32_rows.3a9fa239f50cb6b4.hsaco` | yes (`cmp`) | `a93dfdbc...` | `cf64fc62...58626` | **yes** |

Other hot entries with the same `.hip` text but different object bytes (older toolchain ids, different cache hash): `gemv_mq6g256v2_mq6v2.{5d01d0c1...,db9dcf3f...}`, `qwen4_gemv_mq6g256v2.a2588c958998d611`, `gemm_mq6g256v2_f32_rows.{26261342...,42ec7394...}`; not used. The hot hash name itself embeds the runtime toolchain string and was not recomputed; identity is established by source bytes + radiowave `output_sha256` + byte-equal recompile.

## Descriptor facts (from `.rodata` kernel descriptors and AMDGPU note)

| Symbol | VGPR | SGPR | LDS | private | kernarg | rsrc1 |
|---|---|---|---|---|---|---|
| `gemv_mq6g256v2` | 76 | 24 (asm 22) | 0 | 0 | 32 | `0xE00F0009` |
| `gemv_mq6g256v2_x4` | 76 | 27 | 0 | 0 | 92 | `0xE00F0009` |
| rows r2,r3,r4,r5,r6,r7,r8 | 96,98,101,99,100,98,68 | 28,30,32,34,37,39,34 | 0 | 0 | 32 | |
| rows x4_r2..r8 | 96,98,101,99,100,189,215 | 28,30,31,33,36,38,40 | 0 | 0 | 92 | |

`0xE00F0009`: RNE rounding (modes 0), f32 and f16/f64 denorm modes 3 (preserved), WGP mode, memory ordered, forward progress; gfx12 has no IEEE_MODE/DX10_CLAMP descriptor bits (the asm emits no such directives). `__launch_bounds__(32,16)` (occupancy 16) for single and x4. No atomics, no scratch, no `buffer_` ops in the single object.

## Additional fixture files

| File | Content | sha256 |
|---|---|---|
| `gemm_mq6g256v2_f32_rows.r2_r8.disasm.txt` | `llvm-objdump -d --mcpu=gfx1201` of `gemm_mq6g256v2_f32_rows_r{2..8}` and `_x4_r{2..8}` from `gemm_mq6g256v2_f32_rows.elf` | `449ec2601ccd278df8f6857f6b22cfe5705e1b678eb4e4e2afdccbe80ca7af80` |

## HIP rows incumbent vs singleton (static DAG reading, not a byte proof; the symbolic reader used for it was an investigation-only throwaway script and is not checked in)

For all 14 incumbent symbols and every output column the per-output DAG equals the singleton's (element order 1,0,2..7 with x_i paired with w_i, one fma_mix dequant, separate group add acc-first into stream g%4, tail into streams 0..2, fold `(a0+a1)+(a3+a2)`, swizzle16 + bpermute 8/4/2/1 own+other, lane-0 store). One difference: exactly one column per kernel multiplies `x*w` (the singleton orientation) and the other R-1 columns multiply `w*x` (commuted operands: value-identical except NaN payload selection). Details with instruction addresses: `true-cb/DAG-GFX1201-MQ6.md` section 7. The numerical CPU observation (every incumbent column equal to the singleton on the probed shapes) is recorded below; H0 remains required for hardware bytes.

## Execution evidence (what was and was not run)

- Fixture generation (hipcc compiles, unbundling, disassembly, hot-cache cross-check, static analysis): no GPU, no emulator, no cargo.
- CPU numerical results (parent `BatchInvariantA.A1Mq6Twin`, after emulator core `931aaa84a` merged; not run by the fixture author; own test file 14/14 passed):
  - The lifted hipcc singleton executes in the emulator, with NO fallback to the DAG transcription; the transcription equals the hipcc singleton on 112 words.
  - PM entries (all 10 symbols, all C in range) vs the actual singleton and vs PM `c1` per row: single 17,292 words, x4 24,354 words, 0 mismatches.
  - HIP rows incumbents C=2..8: every one EQUAL to the singleton; single 17*C words, x4 27*C words per launch (r2 34/54, r3 51/81, r4 68/108, r5 85/135, r6 102/162, r7 119/189, r8 136/216, single/x4).
  - 60 invalid-C launches wrote nothing; the default pass also checks guards, input/weight immutability and ragged waves. M7 and image identity unchanged.
- FULL sweep (parent run, not by the fixture author): `LANE_ROWS_MQ6_FULL=1 cargo test -p peacemaker-isa --features toolchain --test lane_rows_mq6_gfx1201 -- --nocapture --test-threads=1` PASSED 15/15 in 183.18 s (larger matrix; the default 14/14 above is a separate smaller matrix and keeps its own counts). K up to 6144; 295 single and 160 x4 entry-case sweeps; each PM output is asserted equal to the DAG transcription, the lifted hipcc singleton and an independent PM `c1` run per row. For x4 the test ALSO executes the actual bound fixture `qwen4_gemv_mq6g256v2.elf` / `gemv_mq6g256v2_x4` (92 B ABI) once per pool row in every x4 case; no skips, no fallbacks. 0 mismatches. Words compared per entry (single / x4): c1 465 / 408; xbatch_r4 5,362 / 4,792; xbatch_r8 18,164 / 16,112; xbatch_r4_wg8 5,362 / 4,792; xbatch_r8_wg8 18,164 / 16,112; totals 47,517 + 42,216 = 89,733. Incumbent per-C EQUAL counts, the 112-word transcription check, 60 untouched invalid-C launches and all-10 M7 / wait-replay / image identity are unchanged. CPU emulator observations only; H0 still required for hardware.
- gfx1151 preservation (parent-reported): the unchanged `qwen4_lane_rows` feature-toolchain test re-run after the core merge PASSED 9/9 in 185.19 s (committed identity, narrow text identity, wait-replay/M7).
- HISTORICAL (before core `931aaa84a`, superseded, not current blockers): the singleton lifted but `Emulator::new` failed with `no semantics for s_mul_u64 (probe K=256 M=2)` (transcription fallback, 0 hipcc words compared); incumbent rows failed at `Emulator::new` with `no semantics for v_cmp_gt_u32_e32` (single C=2..8) and `no semantics for s_load_b512` (x4 C=2..8), 0 words compared; `transcription_decoder_known_answers` passed (K=256/512/1280, M=2, 6 words, transcription decoder only).
- H0 on the R9700 remains required for hardware bytes, throughput and hazards; the CPU emulator observations above do not establish them.
