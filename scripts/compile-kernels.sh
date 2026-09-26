#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Compile only admitted runtime modules from rdna-compute's exact Rust source
# registry. Basename enumeration of kernels/src/*.hip cannot reproduce the
# runtime's preambles, stripped headers, substituted samplers or module keys.
# Usage: ./scripts/compile-kernels.sh [gfx1201 gfx1100 gfx1151 gfx906 gfx942]
#        ./scripts/compile-kernels.sh --print-registry [arch ...]
#        ./scripts/compile-kernels.sh --print-rocm-resolution
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="$SCRIPT_DIR/kernels/compiled"
PRINT_REGISTRY=0
if [ "${1:-}" = "--print-registry" ]; then
    PRINT_REGISTRY=1
    shift
fi
if [ $# -gt 0 ] && [ "$1" != "--print-rocm-resolution" ]; then
    ARCHS=("$@")
else
    ARCHS=(gfx1201 gfx1100 gfx1151 gfx906 gfx942)
fi

ROCM_RESOLUTION="$(
    cargo run --quiet --locked --manifest-path "$SCRIPT_DIR/Cargo.toml" \
        -p hipfire-config --bin hipfire-rocm-resolve
)"
SELECTED_ROCM_ROOT="$(printf '%s\n' "$ROCM_RESOLUTION" | sed -n 's/^ROCM_ROOT=//p')"
HIPCC_BIN="$(printf '%s\n' "$ROCM_RESOLUTION" | sed -n 's/^HIPCC=//p')"
if [ -z "$SELECTED_ROCM_ROOT" ] || [ -z "$HIPCC_BIN" ]; then
    echo "ERROR: ROCm resolver returned incomplete output" >&2
    exit 1
fi
if [ "${1:-}" = "--print-rocm-resolution" ]; then
    printf '%s\n' "$ROCM_RESOLUTION"
    exit 0
fi

cargo build --quiet --locked --manifest-path "$SCRIPT_DIR/Cargo.toml" \
    -p rdna-compute --bin hipfire-kernel-registry --bin hipfire-kernel-pack
TARGET_DIR="${CARGO_TARGET_DIR:-$SCRIPT_DIR/target}"
REGISTRY_BIN="$TARGET_DIR/debug/hipfire-kernel-registry"
PACK_BIN="$TARGET_DIR/debug/hipfire-kernel-pack"
if [ ! -x "$REGISTRY_BIN" ] || [ ! -x "$PACK_BIN" ]; then
    echo "ERROR: kernel registry or pack binary missing after build" >&2
    exit 1
fi

STAGING="$(mktemp -d "$SCRIPT_DIR/.kernel-registry.XXXXXXXX")"
trap 'rm -rf "$STAGING"' EXIT
for arch in "${ARCHS[@]}"; do
    source_dir="$STAGING/$arch"
    "$REGISTRY_BIN" --arch "$arch" --out-dir "$source_dir"
    pack_args=(--arch "$arch" --registry "$source_dir/registry.tsv" --output "$OUT_BASE/$arch")
    # The exporter evaluates FeatureFlags for this arch just as Gpu::init
    # does; raw HIPFIRE_HIPCC_EXTRA_FLAGS alone misses default IU4 flags.
    effective_flags="$(< "$source_dir/extra-flags.txt")"
    if [ -n "$effective_flags" ]; then
        pack_args+=(--extra-flags "$effective_flags")
    fi
    if [ "$PRINT_REGISTRY" -eq 1 ]; then
        while IFS=$'\t' read -r row_arch module symbols source flags profile; do
            printf '%s\t%s\t%s\t%s\t%s\n' "$row_arch" "$module" "$symbols" "$flags" "$profile"
        done < "$source_dir/registry.tsv"
    else
        echo "=== Packaging $arch from exact Rust registry (ROCm: $SELECTED_ROCM_ROOT) ==="
        ROCM_PATH="$SELECTED_ROCM_ROOT" "$PACK_BIN" "${pack_args[@]}"
    fi
done
