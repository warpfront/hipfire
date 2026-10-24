# NPU (XDNA2 / AIE2P): serving route and research stack

hipfire's NPU stack includes an AIE2P program builder, a pure-CPU simulator,
and a direct `amdxdna` runtime (DRM ioctls only, no libxrt or vendor compiler).
The research stack was folded in at `c1bf57a`; the Flash-Next input-projection
route now ships in 0.4.2 on gfx1151 with `HIPFIRE_QWEN4_NPU_TRUNK=auto` by
default. Other architectures resolve auto to off.

## Current serving status (0.4.2)

See [`npu-trunk.md`](npu-trunk.md) for the serving contract and
[`../env-vars.md`](../env-vars.md) for authoritative environment defaults.
Admission requires an open NPU and a matching schema-4 BF16-direct
calibration. The planner supports rebalanced pairs and row-adaptive
256-row-wave chains, with at most four distinct wave counts per load;
a pair needing a fifth runs GPU-native.

The I32 serving contract is removed. The out-projection program is not
wired into serving: neither `qsa.o` nor GDN o-proj ships on this route.
Historical design notes and the research report below are not current
performance claims or admission evidence; use the root
[`CHANGELOG.md`](../../CHANGELOG.md) for release changes.

## Layout

| Piece | Where | What |
|---|---|---|
| `peacemaker-npu` | `crates/peacemaker-npu` | AIE2P emitter: VLIW encoder/scheduler (`isa`, tables generated from llvm-aie TableGen), CDO/PDI writer, DPU transaction writer, DMA/route/registers, kernel designs (`kernels`: int8 GEMM, whole-array V8–V10/G80, IU4/IEF15, ring, experts, bw probe), and the offline simulator (`sim`). Bins `peacemaker-npu-objdump`, `peacemaker-npu-sim`. |
| `railgun::npu` | `crates/railgun/src/npu` (feature `npu`; enabled by `hipfire-arch-qwen4` on Linux) | Runtime: device query, DEV_HEAP, hardware context, CONFIG_CU, BOs, eager/prepared/chained submit, persistent ring, `tape`, FCLK guard (`fclk`), dlopen'd HIP interop (`hip_runtime`). |
| `hipfire_runtime::coproc` | `crates/hipfire-runtime/src/coproc` | Architecture-neutral co-processor stage: op-level stage map, family ownership table, `bf16_direct` numerics contract, schema-4 calibration, planner and live rate scales, layer-major chunk-pair schedule. See [`npu-trunk.md`](npu-trunk.md#generic-co-processor-stage-hipfire_runtimecoproc-044). |
| `npu-tools` | `crates/npu-tools` | Every binary: `npu-tools` (`aie-status`, `txn-dump`, `soc-metrics`, `m4`, `tblgen2rs`, `gpu-bw`, `fclk`), `npu-gemm`, `npu-hello`, `npu-experts`, `npu-ring`, `npu-railgun`, `npu-bw`, `npu-coop`, and `coop27` (`--features lab`). |
| Scripts | `tools/npu/` | Silicon-window harnesses (`npu-window.sh` pins FCLK and takes the timing lock), batch/bisect drivers, the `a4-ec` and `fold-model` offline analyses (standalone cargo projects). |
| Design notes | `docs/npu/*.md` | `railgun-npu.md` (runtime design), `gemm_i8.md`, `fold-contract.md`, `ief15-n0.md`, `g80-feasibility.md`, `dense27b.md`, `coop-*.md`, `levers-*.md`, `a4-ec.md`. |
| Evidence | `docs/npu/report.md` | Historical silicon results under the stated fixtures, not current product claims. `docs/npu/CHANGELOG.md` records the research workspace's history; the root changelog records releases. |

## Relationship to `crates/hipfire-xdna`

`hipfire-xdna` is the older, minimal `amdxdna` ioctl control plane (one
session, one ERT packet, opt-in gfx1151 spillover experiments driven by
`HIPFIRE_XDNA_*`). `railgun::npu` is a separate, broader runtime
(prepared/chained submit, ring, tape, HIP dma-buf interop) used by the
Flash-Next serving route and the research tools. The two control planes
share no code; do not confuse the older spillover experiment with the
calibrated `HIPFIRE_QWEN4_NPU_TRUNK` route.

## Building and testing

```sh
cargo build -p railgun --features npu      # runtime (CI builds this)
cargo build -p npu-tools --features lab    # all binaries incl. coop27 (CI builds this)
cargo test  -p peacemaker-npu --release            # emitter + simulator suites, pure CPU
cargo test  -p railgun --features npu --lib
```

All `peacemaker-npu` tests are pure CPU (`golden_bytes`, `lean`, `ring`, `experts`,
`g80`, `iu4_ief15`, …); `array` and `g80` take minutes in debug. Vendor
byte-oracle cross-checks run only with `AIE2P_VENDOR_CORPUS` set (they skip
otherwise); `isa_decode::vendor_decode_oracle_roundtrip` is `#[ignore]`d
because it also needs an llvm-aie `llvm-objdump` (`AIE2P_OBJDUMP`). Running the
binaries needs XDNA2 silicon, and the co-op paths a gfx1151 iGPU with a pinned
fabric clock (`NPU_FCLK_GUARD`). Environment variables are listed in
[`../env-vars.md`](../env-vars.md#manual--npu-research-stack-dark-not-hipfire_).

## Licensing

Material derived from Xilinx/llvm-aie, Xilinx/mlir-aie, Xilinx/aie-rt and the
Linux amdxdna UAPI is listed, with the affected files, in the root `NOTICE`.
