#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Offline build of kernels/src/ggml_mmvq_shim.hip for gfx1201 (no GPU needed).
# Flags = rdna-compute/src/compiler.rs `core_hipcc_flags(arch)` verbatim, plus
# the include/define set the ggml snapshot needs. No -ffast-math. FP contraction
# policy, recorded explicitly: -ffp-contract=fast-honor-pragmas, which IS the
# clang HIP default (measured: the object is byte-identical with and without the
# flag; -ffp-contract=off changes it), i.e. the policy of every hipfire kernel
# module except the GDN chunk kernels (-ffp-contract=off, compiler.rs
# module_flags_for) and of an unflagged upstream ggml build.
#
# usage: scripts/build-ggml-mmvq-shim.sh [OUT_DIR]   (default: target/ggml-mmvq)
# env:   GGML_SHIM_DEFS="-DGGML_SHIM_ONLY=1"  evidence builds of a single export
#        (never for the shipped object); HIPCC, GGML_SHIM_ARCH
# output: OUT_DIR/ggml_mmvq_shim_gfx1201.co
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/target/ggml-mmvq}"
arch="${GGML_SHIM_ARCH:-gfx1201}"
hipcc="${HIPCC:-hipcc}"
mkdir -p "$out"
exec "$hipcc" \
  --genco "--offload-arch=$arch" -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6 \
  -ffp-contract=fast-honor-pragmas -DGGML_USE_HIP ${GGML_SHIM_DEFS:-} \
  "-I$root/third_party/ggml/include" "-I$root/third_party/ggml/src" "-I$root/third_party/ggml/src/ggml-cuda" \
  "$root/kernels/src/ggml_mmvq_shim.hip" -o "$out/ggml_mmvq_shim_${arch}.co"
