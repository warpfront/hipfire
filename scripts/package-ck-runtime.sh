#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIDECAR="${SIDECAR:-${ROOT}/experiments/flash-attn-ck-sidecar/build/libhipfire_flash_attn_ck.so}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/dist}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
VERSION="${VERSION:-$(git -C "${ROOT}" rev-parse --short=12 HEAD)}"
ALLOW_DIRTY=0

usage() {
    cat <<'EOF'
Usage: package-ck-runtime.sh [options]

Package one exact-architecture CK runtime sidecar. The archive contains no
hipfire binary, ROCm SDK, CK source tree, or model-specific routing policy.

Options:
  --sidecar PATH     ABI v2 CK runtime library
  --output-dir PATH  Destination directory (default: dist)
  --gpu-arch ARCH    gfx1100, gfx1151, or gfx1201
  --version NAME     Bundle version (default: current git commit)
  --allow-dirty      Permit a development bundle from a dirty worktree
  --help             Show this help
EOF
}

while (($#)); do
    case "$1" in
        --sidecar) SIDECAR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gpu-arch) GPU_ARCH="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${GPU_ARCH}" in gfx1100|gfx1151|gfx1201) ;; *) echo "unsupported GPU arch: ${GPU_ARCH}" >&2; exit 2 ;; esac
[[ "${VERSION}" =~ ^[A-Za-z0-9._+-]+$ ]] || { echo "unsafe bundle version: ${VERSION}" >&2; exit 2; }
if [[ "${ALLOW_DIRTY}" -eq 0 && -n "$(git -C "${ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    echo "refusing to package a dirty worktree; commit first or use --allow-dirty" >&2
    exit 2
fi
for command in git readelf sha256sum tar; do
    command -v "${command}" >/dev/null || { echo "missing command: ${command}" >&2; exit 2; }
done
[[ -f "${SIDECAR}" ]] || { echo "missing CK sidecar: ${SIDECAR}" >&2; exit 2; }
for symbol in hipfire_flash_attn_ck_abi_version hipfire_flash_attn_ck_capabilities \
    hipfire_flash_attn_ck_fwd_workspace_bytes hipfire_flash_attn_ck_fwd_supported \
    hipfire_flash_attn_ck_fwd; do
    readelf -Ws "${SIDECAR}" | grep -Fq "${symbol}" || { echo "sidecar is missing ${symbol}" >&2; exit 2; }
done

BUNDLE_NAME="hipfire-ck-runtime-${GPU_ARCH}-${VERSION}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
STAGING="$(mktemp -d "${OUTPUT_DIR}/.${BUNDLE_NAME}.XXXXXX")"
trap 'rm -rf "${STAGING}"' EXIT
BUNDLE_ROOT="${STAGING}/${BUNDLE_NAME}"
mkdir -p "${BUNDLE_ROOT}/lib"
install -m 0755 "${SIDECAR}" "${BUNDLE_ROOT}/lib/libhipfire_flash_attn_ck.so"
cat >"${BUNDLE_ROOT}/manifest.env" <<EOF
BUNDLE_FORMAT_VERSION=1
CK_RUNTIME_ABI_VERSION=2
BUNDLE_VERSION=${VERSION}
GIT_COMMIT=$(git -C "${ROOT}" rev-parse HEAD)
GPU_ARCH=${GPU_ARCH}
EOF
(
    cd "${BUNDLE_ROOT}"
    sha256sum lib/libhipfire_flash_attn_ck.so manifest.env > SHA256SUMS
)
ARCHIVE="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
tar --sort=name --owner=0 --group=0 --numeric-owner -C "${STAGING}" -czf "${ARCHIVE}" "${BUNDLE_NAME}"
(cd "${OUTPUT_DIR}" && sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256")
echo "bundle=${ARCHIVE}"
echo "checksum=${ARCHIVE}.sha256"
