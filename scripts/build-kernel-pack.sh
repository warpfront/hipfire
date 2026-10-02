#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Build release kernel packs from one exact commit. For each architecture the
# commit's own scripts/compile-kernels.sh packages the exact Rust registry into
# kernels/compiled/<arch>, then hipfire-kernel-manifest re-verifies every index
# against the sources and the build compiler and records the provenance.
#
# Outputs, per arch, in --out:
#   hipfire-kernels-<tag>-<arch>.tar.gz         manifest.json + <arch>/*.{hsaco,hash,index.json}
#   hipfire-kernels-<tag>-<arch>.tar.gz.sha256  `sha256sum` line for the tarball
#   hipfire-kernels-<tag>-<arch>.manifest.json  copy of the manifest in the tarball
#
# Needs git, cargo, GNU tar, gzip and hipcc; no GPU. The source is
# `git archive` of the commit, so local edits never reach a pack, and the
# build runs with a fresh HOME and no HIPFIRE_* feature overrides, so the
# objects match what a default install compiles. Tarballs are reproducible:
# sorted members, owner 0, the commit's timestamp, gzip -n.
#
# Usage: scripts/build-kernel-pack.sh --tag TAG [--commit REV] [--out DIR]
#            [--target-dir DIR] [--rocm-min X.Y] [--rocm-max-exclusive X.Y]
#            [arch ...]
#   --commit defaults to the local tag TAG; archs default to every admitted
#   registry arch (gfx1201 gfx1100 gfx1151 gfx906 gfx942).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG=""
REV=""
OUT="$PWD/dist"
TARGET_DIR=""
ROCM_MIN=""
ROCM_MAX=""
ARCHS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --tag|--commit|--out|--target-dir|--rocm-min|--rocm-max-exclusive)
            [ "$#" -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            case "$1" in
                --tag) TAG="$2" ;;
                --commit) REV="$2" ;;
                --out) OUT="$2" ;;
                --target-dir) TARGET_DIR="$2" ;;
                --rocm-min) ROCM_MIN="$2" ;;
                --rocm-max-exclusive) ROCM_MAX="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            sed -n '7,27p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            echo "ERROR: unknown option $1" >&2
            exit 2
            ;;
        *)
            ARCHS+=("$1")
            shift
            ;;
    esac
done
[ "${#ARCHS[@]}" -gt 0 ] || ARCHS=(gfx1201 gfx1100 gfx1151 gfx906 gfx942)
[ -n "$TAG" ] || { echo "ERROR: --tag is required" >&2; exit 2; }
if ! git check-ref-format "refs/tags/$TAG" >/dev/null 2>&1 || [[ "$TAG" == -* ]]; then
    echo "ERROR: invalid tag name '$TAG'" >&2
    exit 2
fi
if { [ -n "$ROCM_MIN" ] && [ -z "$ROCM_MAX" ]; } || { [ -z "$ROCM_MIN" ] && [ -n "$ROCM_MAX" ]; }; then
    echo "ERROR: --rocm-min and --rocm-max-exclusive go together" >&2
    exit 2
fi

if [ -n "$REV" ]; then
    COMMIT="$(git -C "$REPO" rev-parse --verify --quiet "$REV^{commit}")" \
        || { echo "ERROR: --commit $REV is not a commit in $REPO" >&2; exit 1; }
else
    COMMIT="$(git -C "$REPO" rev-parse --verify --quiet "refs/tags/$TAG^{commit}")" \
        || { echo "ERROR: no local tag $TAG; fetch it or pass --commit REV" >&2; exit 1; }
fi
COMMIT_EPOCH="$(git -C "$REPO" show -s --format=%ct "$COMMIT")"

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/hipfire-kernel-pack.XXXXXXXX")"
trap 'rm -rf "$WORK"' EXIT
SRC="$WORK/src"
mkdir -p "$SRC" "$WORK/home" "$WORK/stage"
git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$SRC"
[ -n "$TARGET_DIR" ] || TARGET_DIR="$WORK/target"
mkdir -p "$TARGET_DIR"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

# Clean build environment: toolchain selection passes through, feature
# overrides and the caller's hipfire config do not.
BUILD_ENV=(
    env -i
    "PATH=$PATH"
    "HOME=$WORK/home"
    "LANG=C.UTF-8"
    "CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}"
    "RUSTUP_HOME=${RUSTUP_HOME:-$HOME/.rustup}"
    "CARGO_TARGET_DIR=$TARGET_DIR"
    "HIPFIRE_KERNEL_CACHE=$WORK/kernel-cache"
)
for name in ROCM_PATH HIP_PATH HIPFIRE_ROCM_ROOT HIPFIRE_ROCM_PATH HIPFIRE_HIPCC HIPFIRE_ROCM_STRICT RUSTUP_TOOLCHAIN; do
    if [ -n "${!name:-}" ]; then
        BUILD_ENV+=("$name=${!name}")
    fi
done

cd "$WORK"
RESOLUTION="$("${BUILD_ENV[@]}" "$SRC/scripts/compile-kernels.sh" --print-rocm-resolution)"
ROCM_ROOT="$(printf '%s\n' "$RESOLUTION" | sed -n 's/^ROCM_ROOT=//p')"
[ -n "$ROCM_ROOT" ] || { echo "ERROR: ROCm resolver returned no ROCM_ROOT" >&2; exit 1; }
"${BUILD_ENV[@]}" cargo build --quiet --locked --manifest-path "$SRC/Cargo.toml" \
    -p rdna-compute --bin hipfire-kernel-manifest
MANIFEST_BIN="$TARGET_DIR/debug/hipfire-kernel-manifest"
range_args=()
if [ -n "$ROCM_MIN" ]; then
    range_args=(--rocm-min "$ROCM_MIN" --rocm-max-exclusive "$ROCM_MAX")
fi

echo "=== hipfire kernel packs: $TAG @ $COMMIT (ROCm $ROCM_ROOT) ==="
for arch in "${ARCHS[@]}"; do
    compiled="$SRC/kernels/compiled/$arch"
    rm -rf "$compiled"
    "${BUILD_ENV[@]}" "$SRC/scripts/compile-kernels.sh" "$arch"

    stage="$WORK/stage/$arch"
    mkdir -p "$stage"
    cp -a "$compiled" "$stage/$arch"
    "${BUILD_ENV[@]}" ROCM_PATH="$ROCM_ROOT" "$MANIFEST_BIN" --arch "$arch" --dir "$stage/$arch" \
        --tag "$TAG" --commit "$COMMIT" --rocm-root "$ROCM_ROOT" \
        --out "$stage/manifest.json" "${range_args[@]}"

    stem="hipfire-kernels-$TAG-$arch"
    tar --format=ustar --sort=name --owner=0 --group=0 --numeric-owner \
        --mtime="@$COMMIT_EPOCH" --mode='a+rX,u+w,go-w' \
        -C "$stage" -cf - manifest.json "$arch" | gzip -9n > "$OUT/$stem.tar.gz"
    cp "$stage/manifest.json" "$OUT/$stem.manifest.json"
    (cd "$OUT" && sha256sum "$stem.tar.gz" > "$stem.tar.gz.sha256")
    echo "packed $OUT/$stem.tar.gz ($(wc -c < "$OUT/$stem.tar.gz") bytes, sha256 $(cut -d' ' -f1 "$OUT/$stem.tar.gz.sha256"))"
done
