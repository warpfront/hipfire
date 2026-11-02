#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Build prebuilt Linux binary archives from one exact commit: the hipfire CLI,
# its GPU daemon and the kernel pack for one architecture, for hosts that
# already have ROCm (the HIP runtime is not bundled).
#
# Outputs, per arch, in --out:
#   hipfire-<tag>-linux-rocm-<arch>-x64.tar.gz         the archive
#   hipfire-<tag>-linux-rocm-<arch>-x64.tar.gz.sha256  `sha256sum` line for it
#
# Archive layout (extract anywhere; the daemon loads kernels from
# kernels/compiled/<arch> beside itself):
#   bin/hipfire
#   bin/daemon
#   bin/kernels/compiled/<arch>/   indexed kernel objects for <arch> only
#   LICENSE
#   NOTICE
# Run it with `HIPFIRE_DAEMON_BIN=<dir>/bin/daemon <dir>/bin/hipfire ...` when
# it is not installed under $HIPFIRE_HOME/bin.
#
# Needs git, cargo, GNU tar, gzip, strip (binutils) and hipcc; no GPU. The source is
# `git archive` of the commit, so local edits never reach an archive. The
# binaries link the build host's glibc, so a host needs a glibc at least that
# new; the release workflow builds in the ROCm 10.1 Ubuntu 26.04 image, the same
# toolchain as the kernel packs. Tarballs are reproducible given the same
# binaries: sorted members, owner 0, the commit's timestamp, gzip -n.
#
# Usage: scripts/build-binary-archive.sh --tag TAG [--commit REV] [--out DIR]
#            [--target-dir DIR] [--packs DIR] [arch ...]
#   --commit defaults to the local tag TAG; archs default to gfx1151 gfx1201
#   gfx1100. The CLI and daemon are built once and shared by every arch.
#   --packs DIR takes an arch's kernels from DIR/hipfire-kernels-TAG-ARCH.tar.gz
#   (scripts/build-kernel-pack.sh output, SHA-256 checked) instead of compiling
#   them again; an arch with no pack there is compiled.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG=""
REV=""
OUT="$PWD/dist"
TARGET_DIR=""
PACKS=""
ARCHS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --tag|--commit|--out|--target-dir|--packs)
            [ "$#" -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            case "$1" in
                --tag) TAG="$2" ;;
                --commit) REV="$2" ;;
                --out) OUT="$2" ;;
                --target-dir) TARGET_DIR="$2" ;;
                --packs) PACKS="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            sed -n '7,40p' "$0" | sed 's/^# \{0,1\}//'
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
command -v strip >/dev/null 2>&1 || { echo "ERROR: strip (binutils) is required" >&2; exit 2; }
[ "${#ARCHS[@]}" -gt 0 ] || ARCHS=(gfx1151 gfx1201 gfx1100)
[ -n "$TAG" ] || { echo "ERROR: --tag is required" >&2; exit 2; }
if ! git check-ref-format "refs/tags/$TAG" >/dev/null 2>&1 || [[ "$TAG" == -* ]]; then
    echo "ERROR: invalid tag name '$TAG'" >&2
    exit 2
fi
declare -A SEEN_ARCHS=()
for arch in "${ARCHS[@]}"; do
    [[ "$arch" =~ ^gfx[0-9a-f]+$ ]] || { echo "ERROR: invalid architecture $arch" >&2; exit 2; }
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

[ -z "$PACKS" ] || PACKS="$(cd "$PACKS" && pwd)" || exit 2
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/hipfire-binary-archive.XXXXXXXX")"
trap 'rm -rf "$WORK"' EXIT
SRC="$WORK/src"
mkdir -p "$SRC" "$WORK/home" "$WORK/stage"
git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$SRC"
[ -n "$TARGET_DIR" ] || TARGET_DIR="$WORK/target"
mkdir -p "$TARGET_DIR"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

# Clean build environment: toolchain and device/job selection pass through,
# feature overrides and the caller's hipfire config do not. The build runs no
# GPU work; the empty device defaults keep it off any GPU the host has.
BUILD_ENV=(
    env -i
    "PATH=$PATH"
    "HOME=$WORK/home"
    "LANG=C.UTF-8"
    "CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}"
    "RUSTUP_HOME=${RUSTUP_HOME:-$HOME/.rustup}"
    "CARGO_TARGET_DIR=$TARGET_DIR"
    "HIPFIRE_KERNEL_CACHE=$WORK/kernel-cache"
    "HIPFIRE_BUILD_COMMIT_OVERRIDE=$COMMIT"
    "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:--1}"
    "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:--1}"
)
for name in ROCM_PATH HIP_PATH HIPFIRE_ROCM_ROOT HIPFIRE_ROCM_PATH HIPFIRE_HIPCC HIPFIRE_ROCM_STRICT RUSTUP_TOOLCHAIN HIPFIRE_DEVICES CUDA_VISIBLE_DEVICES CARGO_BUILD_JOBS HIPFIRE_PACK_JOBS; do
    if [ -n "${!name:-}" ]; then
        BUILD_ENV+=("$name=${!name}")
    fi
done

cd "$SRC"
echo "=== hipfire binary archives: $TAG @ $COMMIT (${ARCHS[*]}) ==="
"${BUILD_ENV[@]}" cargo build --release --locked -p hipfire-daemon -p hipfire-cli
[ -x "$TARGET_DIR/release/hipfire" ] && [ -x "$TARGET_DIR/release/daemon" ] \
    || { echo "ERROR: hipfire or daemon missing after build" >&2; exit 1; }

for arch in "${ARCHS[@]}"; do
    compiled="$SRC/kernels/compiled/$arch"
    rm -rf "$compiled"
    pack="${PACKS:+$PACKS/hipfire-kernels-$TAG-$arch.tar.gz}"
    if [ -n "$pack" ] && [ -f "$pack" ]; then
        (cd "$PACKS" && sha256sum --check --quiet "$(basename "$pack").sha256")
        mkdir -p "$WORK/packs/$arch"
        tar -xzf "$pack" -C "$WORK/packs/$arch" "$arch"
        compiled="$WORK/packs/$arch/$arch"
    else
        "${BUILD_ENV[@]}" "$SRC/scripts/compile-kernels.sh" "$arch"
    fi
    [ -d "$compiled" ] && [ -n "$(ls -A "$compiled")" ] \
        || { echo "ERROR: no kernels for $arch" >&2; exit 1; }

    stage="$WORK/stage/$arch"
    rm -rf "$stage"
    mkdir -p "$stage/bin/kernels/compiled"
    cp "$TARGET_DIR/release/hipfire" "$TARGET_DIR/release/daemon" "$stage/bin/"
    # The release binaries carry ~100 MB of symbols; the staged copies (not the
    # cargo target dir) are stripped.
    strip --strip-unneeded "$stage/bin/hipfire" "$stage/bin/daemon"
    cp -a "$compiled" "$stage/bin/kernels/compiled/$arch"
    cp "$SRC/LICENSE" "$SRC/NOTICE" "$stage/"

    stem="hipfire-$TAG-linux-rocm-$arch-x64"
    tar --format=ustar --sort=name --owner=0 --group=0 --numeric-owner \
        --mtime="@$COMMIT_EPOCH" --mode='a+rX,u+w,go-w' \
        -C "$stage" -cf - bin LICENSE NOTICE | gzip -9n > "$OUT/$stem.tar.gz"
    (cd "$OUT" && sha256sum "$stem.tar.gz" > "$stem.tar.gz.sha256")
    echo "packed $OUT/$stem.tar.gz ($(wc -c < "$OUT/$stem.tar.gz") bytes, sha256 $(cut -d' ' -f1 "$OUT/$stem.tar.gz.sha256"))"
done
