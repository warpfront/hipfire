# aco-export — offline PM-ACO producer (PM-ACO-PARITY.md F1 / P0)

Compiles real SPIR-V compute shaders through the **pinned RADV SPIR-V → NIR → ACO** path and
exports `pm-aco-artifact-v1`: the final ACO executable words, shader config (RSRC1/2/3, FP mode,
LDS, scratch, actual vs allocated registers), symbols, and the physical entry-register ABI.
Developer tooling only: no Mesa/C++ code enters any shipping crate; this crate is **not** a
workspace member (`[workspace]` in its Cargo.toml).

## Capture boundary

`mesa/0001-radv-aco-export-hook.patch` (sha256 in `mesa/pin.json`, applied to an *isolated* detached
worktree of Mesa `mesa-26.0.8` / `60e95b787857afbc9a00b693b91c0d9c8923a430`; the original
`/home/kaden/qcal/peacemaker/mesa` is never touched) adds:

* `radv_aco_export.inc`, included by `radv_shader.c` right before `radv_aco_build_shader_binary()`.
  The `aco_callback` stashes code/exec_size/config/symbols/ACO disasm; `shader_compile()` calls the
  emitter after `radv_postprocess_binary_config()` so it sees the final RSRC1/2/3, the
  `radv_shader_args`/`ac_shader_args` slot table (field names resolved, aliasing detected),
  `user_sgprs_locs`, inline push mask, `radv_shader_info`, aco options/info, device identity, RADV's own
  `radv_precompute_registers()` result, and the final NIR. Only active when `RADV_ACO_EXPORT_DIR` is set.
* `aco_export_ra_max_used_{vgprs,sgprs}` written by ACO register allocation (actual use vs the
  granule-rounded allocation).

No code generation changes. gfx12 workgroup ids have no `ac_arg` slot (`radv_shader_args.c:572-578`);
they are the cited TTMP facts (`aco_select_nir_intrinsics.cpp:4056-4073`: x=ttmp9, y/z=ttmp7).
Anything not understood (unnamed/aliased slot, non-inlined or partially inlined 64-bit push pointer,
scratch/spills, ACO symbols, unknown RSRC bits, packed gfx12 y/z ids) is a hard `REJECT`.

## Commands (run once, in order; nothing here was executed by the producer agent)

```sh
W=/home/kaden/qcal/release-0.4.2/worktrees/pm-aco-pilot-20261011
MESA=/home/kaden/qcal/release-0.4.2/worktrees/mesa-aco-pilot-20261011   # isolated worktree, patch already applied
P=/home/kaden/qcal/release-0.4.2/pm-aco-pilot                            # outputs (outside both trees)

ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 cargo build --release --offline -j 12 --manifest-path $W/tools/aco-export/Cargo.toml
X=$W/tools/aco-export/target/release/aco-export

# 1. patched RADV (+ drm-shim). meson/ninja; options are pinned in mesa/pin.json (-Dllvm=disabled).
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 $X build-mesa --mesa-src $MESA --build-dir $P/mesa-build --jobs 12

# 2. plain-Vulkan host
$X build-host --out $P/bin/vk_aco_host

# 3a. REAL device compile (card B, exact PCI identity; Vulkan ignores ROCR_VISIBLE_DEVICES)
F=/home/kaden/qcal/locks/fabric/fabric-run.sh
for K in copy_scale_f32 gemv_mq4g256v2_residual; do
  $F --class correct --card GPU-e475645fe0200397 --mem-gb 1 --hostmem-gb 2 -- \
    $X export --kernel $K --out $P/artifacts/$K --mesa-src $MESA --mesa-build $P/mesa-build \
    --host $P/bin/vk_aco_host --pci-bdf 0000:13:00.0
done
# 3b. (alternative, labelled OFFLINE, no hardware identity) drm-shim gfx1201:
#     ... export ... --offline-shim

# 4. existing radiowave oracle command on the ACO dump (also printed by `export`)
cargo run -q -p radiowave --bin radiowave --manifest-path $W/Cargo.toml -- \
   oracle aco --input $P/artifacts/gemv_mq4g256v2_residual/shader.aco.dump \
   --input-artifact $P/artifacts/gemv_mq4g256v2_residual/shader.spv --kernel gemv_mq4g256v2_residual \
   --arch gfx1201 --wavefront 32 --workgroup 32,1,1 > $P/gemv_mq4g256v2_residual.radiowave-oracle.json   # report goes OUTSIDE the immutable artifact
# or: $X oracle --artifact $P/artifacts/gemv_mq4g256v2_residual --radiowave <path-to-radiowave-bin> --out $P/gemv_mq4g256v2_residual.radiowave-oracle.json

# 5. HIP contract (pm-aco-hip-contract-v1) from the sidecar/rawspec + real capture
CAP=/home/kaden/qcal/release-0.4.2/pm-decode-twins/oracle-inputs/gemv_mq4g256v2_residual/occ0
$X contract --artifact $P/artifacts/gemv_mq4g256v2_residual --capture $CAP --out $P/artifacts/gemv_mq4g256v2_residual.contract.json
# (odd/tail support: add --odd-tail-evidence-sha256 <sha of the Vulkan odd-tail receipt bucket> after step 7)

# 6. Oracle framed inputs (existing example, gated behind lab)
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 cargo build --release --offline -j 12 --manifest-path $W/Cargo.toml -p rdna-compute --features lab --example pm_aco_oracle
$W/target/release/examples/pm_aco_oracle prepare --capture $CAP --out $P/prepare

# 7. original Vulkan dispatch on the exact device -> aco-vulkan-ref-v1
$F --class correct --card GPU-e475645fe0200397 --mem-gb 1 --hostmem-gb 2 -- \
   $X reference --artifact $P/artifacts/gemv_mq4g256v2_residual --capture $CAP --prepare-dir $P/prepare \
   --out $P/vk-ref --pci-bdf 0000:13:00.0 --mesa-src $MESA --mesa-build $P/mesa-build --host $P/bin/vk_aco_host
```

`export` requires an empty output directory; it fails if the tree is not exactly pin + patch, if the
capture does not contain exactly one ACO compile, or if anything is rejected. `reference` refuses a
receipt unless the ACO binary re-compiled for the dispatch is byte-identical to `shader.bin`.

After the producer/reference steps, build `pm-raw-census`, `aco_hsa_emit` (feature `toolchain`) and
`pm_aco_oracle` (feature `lab`) with GPUs hidden and `-j 12`:

```sh
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 cargo build --release --offline -j 12 --manifest-path $W/Cargo.toml -p peacemaker-isa --features toolchain --example aco_hsa_emit
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 cargo build --release --offline -j 12 --manifest-path $W/Cargo.toml -p rdna-compute --features lab --example pm_aco_oracle
```

Run `pm-raw-census shader.rawspec.json`, then
`$W/target/release/examples/aco_hsa_emit shader.rawspec.json CONTRACT.json OUT`. The latter writes native ELF/bundle images,
their manifests, and `pm-aco-certificate-v1` with separately closed Raw/HSA obligations; it is
static evidence, not a GPU receipt. Use the exercised Vulkan odd-tail bucket hash when generating
an odd/tail-enabled contract. Do not edit a frozen artifact to add these external receipts.

Run `pm_aco_oracle full --capture CAP --candidate OUT/candidate.co --candidate-manifest
OUT/candidate.co.manifest.json --vulkan-receipt REF/receipt.json --out REPORT` through the same
card-B correctness wrapper. It requires exact whole-buffer/guard equality, negative controls and
eager/hipGraph replay. Only after PASS, register `predict add --needs-run --reason "required pilot
diagnostic"` with an explicit forecast/noise band, then run the separate `time` mode through
`fabric-run.sh --class timing`, supplying `--parity-report REPORT/report.json`. This is one
diagnostic comparison against the tuned hipcc and accepted native PM twin, not product admission.

### Native PM twin vs hipcc: `twin-time` (no candidate)

To check only whether the default-on native PM decode twin costs kernel time against the pinned
hipcc object, run `pm_aco_oracle twin-time --capture CAP --out DIR [--iters N --reps R --warmup W
--cache warm|cold]` (once per captured occurrence, e.g. `occ0` WO 5120x6144 and `occ127` DOWN
5120x17408) through `fabric-run.sh --class timing` on the card under test, after registering the
prediction. It needs no candidate, ACO manifest, Vulkan receipt or parity report and refuses those
options. The inventory-verified hipcc object and the accepted twin (`pm_decode_twins`) are loaded,
run eagerly and through hipGraph on the captured M/K/grid/block, and their whole A/x/y buffers
plus guards must equal the captured real post bytes and each other (six comparisons, nonzero
compared bytes, zero differing bytes); any mismatch fails closed and no timing is taken. Timing is
exactly two arms, rep-major with the order alternating per rep (executed order recorded), event
timing, no ACO arm. `DIR/twin-time-report.json` (schema `pm-twin-time-v1`) binds the executable
(sha256/md5), card (device id, HIP UUID, PCI bus), capture meta/file hashes, arm images, shape
geometry, parity counts and per-arm samples with `twin_over_hipcc`.

This is a kernel-only residual-launch diagnostic. It is not the product A/B (`allpm_decode`
default on vs off through the stock `hipfire bench` path, per
`docs/methodology/perf-benchmarking.md`), which measures model-level decode speed and is
never replaced by it. The existing `time` mode is unchanged and still requires the candidate,
manifest and a passing `full` report.

## Artifact layout (`pm-aco-artifact-v1`)

| file | content |
|---|---|
| `shader.spv` | glslang output (validated; DenormPreserve/SignedZeroInfNanPreserve 32 + PhysicalStorageBufferAddresses asserted) |
| `shader.bin` | WHOLE ACO code buffer (executable range + any trailing data), byte-exact |
| `shader.abi.json` | `pm-aco-abi-v1` entry ABI sidecar |
| `shader.rawspec.json` | `pm-raw-kernel-spec-v1` (RawLift) |
| `shader.aco.dump` | radiowave-format dump: ROCm llvm-mc disassembly of the executable range + RADV executable statistics |
| `capture/` | `capture.json` (raw hook output), `final.nir.txt`, `aco.disasm.txt`, `host-stats.txt`, `identity.json`, `host.stderr.txt` |
| `artifact.json` | canonical JSON: files{rel:{sha256}}, provenance{compiler_sha256, mesa_patch_sha256, mesa_commit}, source/SPIR-V/compiler/patch/driver/host/env identities, pipeline layout/specialization hashes, binary{sha256 whole, executable{offset,len,sha256}, trailing}, config, abi/rawspec hashes, bindings/extents, cache/scope/robustness, radiowave argv |

The MQ4 shader reproduces the incumbent numerics order exactly (see the header comment of
`crates/peacemaker-lift/tests/fixtures/aco/src/gemv_mq4g256v2_residual.comp`). Whether ACO/RADV
preserves that order bit-for-bit is exactly what the ACO-vs-Vulkan-vs-hipcc oracle decides; it is not
assumed.

## First MQ4 kill verdict (2026-10-11)

Card B (`GPU-e475645fe0200397`, PCI `0000:13:00.0`, gfx1201) passed the actual producer,
typed 985-instruction round trip, native HSA relift, zero Raw/HSA obligations, and the whole-buffer
Vulkan/HIP/hipcc/native-twin eager/graph oracle on M5120/K6144 and the real-prefix odd/tail case
M5119/K5888. The body is 5,456 unchanged bytes; the entry prologue is 28 bytes. CPU gates passed
504 tests (five ignored); affected source maps match.

Evidence is under `/home/kaden/qcal/release-0.4.2/pm-aco-pilot/`: immutable
`artifacts/gemv_mq4g256v2_residual-v2/`, `hsa-final/certificate.json`,
`vk-ref-v2/receipt.json`, `oracle-full/report.json`, and `oracle-timing/time-report.json`.
The original v1 artifact and `mesa-build` driver are preserved separately.

The one fabric timing diagnostic, preceded by prediction P0062, measured warm-cache medians
21.743 µs ACO/HIP, 18.243 µs hipcc, and 21.857 µs accepted native PM. ACO/HIP is 19.2% slower
than hipcc; its 0.5% difference from the native twin is not a resolved win. **Stop performance
spending and default-cutover work for this pilot.** Retain the proven explicit import/parity path;
no default, model-level, retained-route, other-arch or full-floating-domain promotion is claimed.
