#!/usr/bin/env bash
set -euo pipefail

PREFIX="${HIPFIRE_CK_PREFIX:-${HOME}/.local/opt/hipfire-ck}"
BUNDLE=""
URL=""
EXPECTED_SHA256=""
SKIP_GPU_CHECK=0

usage() {
    cat <<'EOF'
Usage: install-ck-runtime.sh (--bundle PATH | --url URL) [options]

Install one exact-architecture CK runtime sidecar atomically.

Options:
  --bundle PATH     Local bundle archive
  --url URL         Download bundle with curl
  --sha256 HEX      Required archive checksum for --url
  --prefix PATH     Install root (default: ~/.local/opt/hipfire-ck)
  --skip-gpu-check  Permit installation without a matching visible GPU
  --help            Show this help
EOF
}

while (($#)); do
    case "$1" in
        --bundle) BUNDLE="$2"; shift 2 ;;
        --url) URL="$2"; shift 2 ;;
        --sha256) EXPECTED_SHA256="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --skip-gpu-check) SKIP_GPU_CHECK=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ -n "${BUNDLE}" && -n "${URL}" ]] || [[ -z "${BUNDLE}" && -z "${URL}" ]]; then
    echo "select exactly one of --bundle or --url" >&2; exit 2
fi
if [[ -n "${URL}" && ! "${EXPECTED_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "--url requires --sha256 with 64 hexadecimal characters" >&2; exit 2
fi
for command in sha256sum tar readelf ldd; do
    command -v "${command}" >/dev/null || { echo "missing command: ${command}" >&2; exit 2; }
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/hipfire-ck-install.XXXXXX")"
STAGED=""
cleanup() { rm -rf "${WORK}"; [[ -z "${STAGED}" || ! -e "${STAGED}" ]] || rm -rf "${STAGED}"; }
trap cleanup EXIT
if [[ -n "${URL}" ]]; then
    command -v curl >/dev/null || { echo "curl is required for --url" >&2; exit 2; }
    BUNDLE="${WORK}/bundle.tar.gz"
    curl --fail --location --retry 3 --output "${BUNDLE}" "${URL}"
else
    BUNDLE="$(cd "$(dirname "${BUNDLE}")" && pwd)/$(basename "${BUNDLE}")"
fi
[[ -f "${BUNDLE}" ]] || { echo "bundle not found: ${BUNDLE}" >&2; exit 2; }
if [[ -n "${EXPECTED_SHA256}" ]]; then
    ACTUAL="$(sha256sum "${BUNDLE}" | awk '{print $1}')"
    [[ "${ACTUAL}" == "${EXPECTED_SHA256,,}" ]] || { echo "bundle SHA256 mismatch" >&2; exit 2; }
fi
while IFS= read -r member; do
    case "${member}" in /*|../*|*/../*|*/..) echo "unsafe archive member: ${member}" >&2; exit 2 ;; esac
done < <(tar -tzf "${BUNDLE}")
tar -xzf "${BUNDLE}" -C "${WORK}"
mapfile -t ROOTS < <(find "${WORK}" -mindepth 1 -maxdepth 1 -type d -name 'hipfire-ck-runtime-*' | sort)
[[ "${#ROOTS[@]}" -eq 1 ]] || { echo "bundle must contain exactly one runtime root" >&2; exit 2; }
SOURCE="${ROOTS[0]}"
for path in lib/libhipfire_flash_attn_ck.so manifest.env SHA256SUMS; do
    [[ -f "${SOURCE}/${path}" ]] || { echo "bundle is missing ${path}" >&2; exit 2; }
done
(cd "${SOURCE}" && sha256sum --check --strict --quiet SHA256SUMS)
manifest_value() { sed -n "s/^$1=//p" "${SOURCE}/manifest.env" | tail -n1; }
[[ "$(manifest_value BUNDLE_FORMAT_VERSION)" == 1 ]] || { echo "unsupported bundle format" >&2; exit 2; }
[[ "$(manifest_value CK_RUNTIME_ABI_VERSION)" == 2 ]] || { echo "unsupported CK runtime ABI" >&2; exit 2; }
ARCH="$(manifest_value GPU_ARCH)"
VERSION="$(manifest_value BUNDLE_VERSION)"
case "${ARCH}" in gfx1100|gfx1151|gfx1201) ;; *) echo "unsupported GPU arch: ${ARCH}" >&2; exit 2 ;; esac
[[ "${VERSION}" =~ ^[A-Za-z0-9._+-]+$ ]] || { echo "unsafe bundle version: ${VERSION}" >&2; exit 2; }
if ldd "${SOURCE}/lib/libhipfire_flash_attn_ck.so" | grep -q 'not found'; then
    ldd "${SOURCE}/lib/libhipfire_flash_attn_ck.so" >&2
    echo "bundle has unresolved dynamic dependencies" >&2; exit 2
fi
if [[ "${SKIP_GPU_CHECK}" -eq 0 ]]; then
    command -v rocminfo >/dev/null || { echo "rocminfo is required for the exact-arch check" >&2; exit 2; }
    grep -Eq "Name:[[:space:]]+${ARCH}([[:space:]]|$)" < <(rocminfo 2>/dev/null) || {
        echo "no matching ${ARCH} GPU is visible" >&2; exit 2;
    }
fi
DEST_ROOT="${PREFIX}/${ARCH}"
mkdir -p "${DEST_ROOT}/releases"
DEST_ROOT="$(cd "${DEST_ROOT}" && pwd)"
DEST="${DEST_ROOT}/releases/${VERSION}"
[[ ! -e "${DEST}" ]] || { echo "release already exists: ${DEST}" >&2; exit 2; }
STAGED="${DEST_ROOT}/releases/.${VERSION}.installing.$$"
cp -a "${SOURCE}" "${STAGED}"
mv "${STAGED}" "${DEST}"
STAGED=""
ln -sfn "releases/${VERSION}" "${DEST_ROOT}/current"
echo "installed=${DEST}"
echo "sidecar=${DEST_ROOT}/current/lib/libhipfire_flash_attn_ck.so"
