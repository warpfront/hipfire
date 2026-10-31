#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Route A seal of the gfx1100 / gfx1201 hipcc twins into embedded `kernels/*.hxaco`:
#   kernels/gated_delta_vector_pm_gfx1100.hip          (RDA serial + depthwise conv rows, XTX judge)
#   kernels/gemm_mq4g256v2_moe_grouped_k2_pm_gfx1100.hip
#   kernels/gated_delta_vector_pm_gfx1201.hip          (the same two bodies, R9700 card judge)
#
# Each source is compiled with EXACTLY the runtime JIT argv (crates/rdna-compute/src/compiler.rs
# `core_hipcc_flags` + the rocm path flags the card judge recorded in candidate.co.json), the
# shipped incumbent source (kernels/src) is compiled the same way as the kernarg `--abi-ref`, and
# the `hip_twin` example seals every twin symbol (lift identity via emit::checked, hazards,
# SGPR >= 16, kernarg ABI == incumbent). Each twin symbol's `.text` sha256 must equal the
# card-judged candidate's `code_sha256`; a mismatch aborts (the rename must not change code).
#
# Writes kernels/<module>.hxaco, kernels/<module>.hxaco.sha256 and kernels/<module>.twin.json.
#
# Env: ROCM_PATH (default /opt/rocm/core), CARGO_TARGET_DIR, PM_TWIN_WORK (scratch dir, default
#      <repo>/target/pm-hip-twin), PM_TWIN_ARCH (seal only that arch's modules; default all).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM="${ROCM_PATH:-/opt/rocm/core}"
WORK="${PM_TWIN_WORK:-$REPO/target/pm-hip-twin}"
ONLY_ARCH="${PM_TWIN_ARCH:-}"
mkdir -p "$WORK"

# Mirror of compiler.rs `core_hipcc_flags(arch)` followed by the recorded rocm path flags.
hipcc_compile() { # src out arch
  local src="$1" out="$2" arch="$3"
  "$ROCM/bin/hipcc" --genco "--offload-arch=$arch" -O3 --no-offload-compress -fuse-cuid=none \
    -mcode-object-version=6 "--rocm-path=$ROCM" "--hip-path=$ROCM" "-I$ROCM/include" \
    "-fmacro-prefix-map=$(dirname "$src")/=" -o "$out" "$src"
}

( cd "$REPO" && cargo build --release -p peacemaker-isa --features lab --example hip_twin )
TWIN_BIN="${CARGO_TARGET_DIR:-$REPO/target}/release/examples/hip_twin"

# seal_module <arch> <module> <twin-src> <spec>...
#   spec = twin_sym:incumbent_src:incumbent_sym:judged_candidate:judged_code_sha256
seal_module() {
  local arch="$1" module="$2" src="$3"
  shift 3
  if [[ -n "$ONLY_ARCH" && "$ONLY_ARCH" != "$arch" ]]; then return 0; fi
  local dir="$WORK/$module"
  mkdir -p "$dir"
  hipcc_compile "$REPO/kernels/$src" "$dir/twin.co" "$arch"
  local args=(--in "$dir/twin.co" --arch "$arch" --out "$REPO/kernels/$module.hxaco" --report "$dir/report.json")
  local spec i=0 tsym isrc isym
  for spec in "$@"; do
    IFS=: read -r tsym isrc isym _ _ <<<"$spec"
    hipcc_compile "$REPO/kernels/src/$isrc" "$dir/incumbent$i.co" "$arch"
    args+=(--kernel "$tsym" --abi-ref "$dir/incumbent$i.co" --abi-ref-symbol "$isym")
    i=$((i + 1))
  done
  "$TWIN_BIN" "${args[@]}" >/dev/null
  python3 - "$dir/report.json" "$REPO/kernels/$module.twin.json" "$module" "$arch" "$@" <<'PY'
import json, sys
report_path, out_path, module, arch, *specs = sys.argv[1:]
reports = json.load(open(report_path))
if isinstance(reports, dict):
    reports = [reports]
assert len(reports) == len(specs), (len(reports), len(specs))
judged, bad = [], []
for rep, spec in zip(reports, specs):
    tsym, _isrc, _isym, label, want = spec.split(":")
    got = rep["text_sha256"]
    judged.append({"symbol": tsym, "judged_candidate": label, "judged_code_sha256": want,
                   "twin_text_sha256": got, "equal": want == got})
    if want != got:
        bad.append(f"{tsym}: twin .text {got} != judged {want} ({label})")
with open(out_path, "w") as f:
    json.dump({"module": module, "arch": arch, "reports": reports, "judged_text_check": judged}, f, indent=2)
    f.write("\n")
if bad:
    sys.exit("pm-hip-twin: rename/compile changed .text vs the card-judged candidate:\n  " + "\n  ".join(bad))
PY
  echo "sealed kernels/$module.hxaco sha256 $(cat "$REPO/kernels/$module.hxaco.sha256")"
}

# spec = twin_sym:incumbent_src:incumbent_sym:judged_candidate:judged_code_sha256 (the candidate's `.text` sha256 as
# compiled and timed by the card judge: gfx1100 hip_loop R1/R2 on the XTX, gfx1201 hip_loop on R9700 E; bit-identical
# to the incumbent on the respective card).
seal_module gfx1100 gated_delta_vector_pm_gfx1100 gated_delta_vector_pm_gfx1100.hip \
  "gated_delta_net_vector_serial_f32_pm_gfx1100:gated_delta_net_vector_serial.hip:gated_delta_net_vector_serial_f32:R2Rda-05-rs-tv8-prio:6d0b57fb7bb4d8f5f8ab1bbd3ceffeb31ffdb08bd1a761dbfebc8b8b60e8113f" \
  "causal_conv_depthwise_silu_rows_f32_pm_gfx1100:causal_conv_depthwise_rows.hip:causal_conv_depthwise_silu_rows_f32:R1Conv-04-rpw32-k4:95629e55cb81c016bbca77dd366aebce2830c2612e7c8e9d4d927570b20e1a0a"

seal_module gfx1100 gemm_mq4g256v2_moe_grouped_k2_pm_gfx1100 gemm_mq4g256v2_moe_grouped_k2_pm_gfx1100.hip \
  "gemm_mq4g256v2_moe_grouped_wmma_k2_pm_gfx1100:gemm_mq4g256v2_moe_grouped_wmma_k2.hip:gemm_mq4g256v2_moe_grouped_wmma_k2:R1Moe-04-xlds-w4-t2:33675b778bc94940ad4fa6600111953b32eaeb2eb503ed530cd4b1b78ef4228d"

# gfx1201: the identical bodies (symbol rename only), judged on R9700 E by hip_loop (`rda_serial`, `conv_rows`).
seal_module gfx1201 gated_delta_vector_pm_gfx1201 gated_delta_vector_pm_gfx1201.hip \
  "gated_delta_net_vector_serial_f32_pm_gfx1201:gated_delta_net_vector_serial.hip:gated_delta_net_vector_serial_f32:R2Rda-05-rs-tv8-prio:4b0f755c125e8024da75103a4c1c7d6e73bf329815b77ee15e164c67fd1485ec" \
  "causal_conv_depthwise_silu_rows_f32_pm_gfx1201:causal_conv_depthwise_rows.hip:causal_conv_depthwise_silu_rows_f32:R1Conv-04-rpw32-k4:35985539865d2a7e5a1ea193f1ee2287722fa528f551f115bd4e55f8c8fb8e83"
