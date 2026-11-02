#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Build release kernel packs from one exact commit. For each architecture the
# commit's own scripts/compile-kernels.sh packages the exact Rust registry into
# kernels/compiled/<arch>, then hipfire-kernel-manifest re-verifies every index
# against the sources, the build compiler and each object's kernel exports, and records the provenance.
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
#            [--target-dir DIR] [--jobs N] [--rocm-min X.Y] [--rocm-max-exclusive X.Y]
#            [arch ...]
#   --commit defaults to the local tag TAG; archs default to every admitted
#   registry arch (gfx1201 gfx1100 gfx1151 gfx906 gfx942).
#   --jobs limits concurrent architectures (default 1). The host-scaled total
#   compiler budget is split across them, not multiplied; HIPFIRE_PACK_JOBS
#   overrides that total (still bounded by cores and available RAM).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG=""
REV=""
OUT="$PWD/dist"
TARGET_DIR=""
ROCM_MIN=""
ROCM_MAX=""
ARCHS=()
JOBS=1
PACK_JOBS="${HIPFIRE_PACK_JOBS:-}"
PIDS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --tag|--commit|--out|--target-dir|--jobs|--rocm-min|--rocm-max-exclusive)
            [ "$#" -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            case "$1" in
                --tag) TAG="$2" ;;
                --commit) REV="$2" ;;
                --out) OUT="$2" ;;
                --target-dir) TARGET_DIR="$2" ;;
                --jobs) JOBS="$2" ;;
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
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { [ -n "$PACK_JOBS" ] && ! [[ "$PACK_JOBS" =~ ^[1-9][0-9]*$ ]]; }; then
    echo "ERROR: --jobs and HIPFIRE_PACK_JOBS must be positive integers" >&2
    exit 2
fi
# nproc respects the process CPU affinity just as available_parallelism does.
# Budget all arch workers together, including the approximately 1.5 GiB/job RAM
# limit. /proc's MemAvailable includes reclaimable caches, not only unused RAM.
CORES="$(nproc)"
RESERVE=$((CORES / 8))
[ "$RESERVE" -ge 1 ] || RESERVE=1
BUDGET=$((CORES - RESERVE))
[ "$BUDGET" -ge 1 ] || BUDGET=1
if [ -n "$PACK_JOBS" ]; then
    # Clamp before shell arithmetic, so arbitrarily large decimal inputs cannot
    # wrap around and accidentally select an unbounded number of workers.
    if [ "${#PACK_JOBS}" -gt "${#CORES}" ] || { [ "${#PACK_JOBS}" -eq "${#CORES}" ] && [[ "$PACK_JOBS" > "$CORES" ]]; }; then
        BUDGET="$CORES"
    else
        BUDGET="$PACK_JOBS"
    fi
fi
MEM_KB=0
while read -r key value unit; do
    if [ "$key" = "MemAvailable:" ]; then MEM_KB="$value"; break; fi
done < /proc/meminfo
MEM_JOBS=$((MEM_KB / 1572864))
[ "$MEM_JOBS" -ge 1 ] || MEM_JOBS=1
[ "$BUDGET" -le "$MEM_JOBS" ] || BUDGET="$MEM_JOBS"
ARCH_JOBS="${#ARCHS[@]}"
if [ "${#JOBS}" -lt "${#ARCH_JOBS}" ] || { [ "${#JOBS}" -eq "${#ARCH_JOBS}" ] && [[ "$JOBS" < "$ARCH_JOBS" ]]; }; then ARCH_JOBS="$JOBS"; fi
[ "$ARCH_JOBS" -le "$BUDGET" ] || ARCH_JOBS="$BUDGET"
MODULE_JOBS=$((BUDGET / ARCH_JOBS))
# Duplicate arches would concurrently remove and overwrite the same directory.
declare -A SEEN_ARCHS=()
for arch in "${ARCHS[@]}"; do
    if [ -n "${SEEN_ARCHS[$arch]:-}" ]; then echo "ERROR: duplicate architecture $arch" >&2; exit 2; fi
    SEEN_ARCHS[$arch]=1
done

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
cleanup() {
    local pid
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    rm -rf "$WORK"
}
trap cleanup EXIT
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
BUILD_ENV+=("HIPFIRE_PACK_JOBS=$MODULE_JOBS")
# Caller device/job selection passes through untouched: the pack build itself
# runs no GPU work, but the caller's ROCR/HIP visibility and cargo job budget
# must reach the compile workers unchanged (never defaulted or clamped here).
for name in ROCM_PATH HIP_PATH HIPFIRE_ROCM_ROOT HIPFIRE_ROCM_PATH HIPFIRE_HIPCC HIPFIRE_ROCM_STRICT RUSTUP_TOOLCHAIN ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES HIPFIRE_DEVICES CUDA_VISIBLE_DEVICES CARGO_BUILD_JOBS; do
    if [ -n "${!name:-}" ]; then
        BUILD_ENV+=("$name=${!name}")
    fi
done

cd "$WORK"
RESOLUTION="$("${BUILD_ENV[@]}" "$SRC/scripts/compile-kernels.sh" --print-rocm-resolution)"
ROCM_ROOT="$(printf '%s\n' "$RESOLUTION" | sed -n 's/^ROCM_ROOT=//p')"
[ -n "$ROCM_ROOT" ] || { echo "ERROR: ROCm resolver returned no ROCM_ROOT" >&2; exit 1; }
"${BUILD_ENV[@]}" cargo build --quiet --locked --manifest-path "$SRC/Cargo.toml" \
    -p rdna-compute --bin hipfire-kernel-manifest --bin hipfire-kernel-registry --bin hipfire-kernel-pack
MANIFEST_BIN="$TARGET_DIR/debug/hipfire-kernel-manifest"
range_args=()
if [ -n "$ROCM_MIN" ]; then
    range_args=(--rocm-min "$ROCM_MIN" --rocm-max-exclusive "$ROCM_MAX")
fi

echo "=== hipfire kernel packs: $TAG @ $COMMIT (ROCm $ROCM_ROOT) ==="
echo "=== $ARCH_JOBS concurrent arches, $MODULE_JOBS workers each (total budget $BUDGET / $CORES cores) ==="
build_arch() {
    local arch="$1" compiled stage stem
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
}

# Bounded batches keep the sum of per-arch pools within the shared budget.
# Wait for every sibling on failure before the EXIT trap removes their sources.
wait_batch() {
    local pid status=0
    for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
    PIDS=()
    return "$status"
}
for arch in "${ARCHS[@]}"; do
    build_arch "$arch" &
    PIDS+=("$!")
    if [ "${#PIDS[@]}" -eq "$ARCH_JOBS" ]; then wait_batch; fi
done
wait_batch
