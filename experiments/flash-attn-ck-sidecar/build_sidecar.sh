#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
FLASH_ATTN_ROOT="${FLASH_ATTN_ROOT:?set FLASH_ATTN_ROOT to a flash-attention checkout containing composable_kernel}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
MAX_JOBS="${MAX_JOBS:-8}"
OUT="${OUT:-${ROOT}/build/libhipfire_flash_attn_ck.so}"
HEAD_DIMS="${HEAD_DIMS:-64,128,256}"

case "${GPU_ARCH}" in
    gfx1100)
        TARGET_DEFINE=HIPFIRE_CK_TARGET_GFX1100
        GENERATOR_TARGET=gfx11
        EXPECTED_SOURCES=17
        ;;
    gfx1151)
        TARGET_DEFINE=HIPFIRE_CK_TARGET_GFX1151
        GENERATOR_TARGET=gfx11
        EXPECTED_SOURCES=17
        ;;
    gfx1201)
        TARGET_DEFINE=HIPFIRE_CK_TARGET_GFX1201
        GENERATOR_TARGET=gfx12
        EXPECTED_SOURCES=13
        ;;
    *) echo "unsupported exact CK artifact arch: ${GPU_ARCH}" >&2; exit 2 ;;
esac
BUILD_DIR="$(dirname "${OUT}")"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
HIP_LIB_DIR="${HIP_ROOT}/lib"
EXTERNAL_CK_ROOT="${FLASH_ATTN_ROOT}/csrc/composable_kernel"
REQUIRED_CK_REV="13f6d635653bd5ffbfcac8577f1ef09590c23d78"
RECIPE_PATCH="${ROOT}/gfx11_ck_recipe.patch"
REQUIRED_RECIPE_SHA256="b43ea8d12e14cef04518225acaa69b63e62991ba4a83efcd596fc108105ac765"
CK_ROOT="${BUILD_DIR}/ck-source"

if [[ ! -f "${HIP_LIB_DIR}/libamdhip64.so" ]]; then
    echo "missing HIP runtime under ${HIP_LIB_DIR}" >&2
    exit 2
fi

if [[ ! -d "${EXTERNAL_CK_ROOT}/.git" && ! -f "${EXTERNAL_CK_ROOT}/.git" ]]; then
    echo "missing composable_kernel git checkout under ${EXTERNAL_CK_ROOT}" >&2
    exit 2
fi
if ! git -C "${EXTERNAL_CK_ROOT}" cat-file -e "${REQUIRED_CK_REV}^{commit}"; then
    echo "composable_kernel checkout does not contain required revision ${REQUIRED_CK_REV}" >&2
    exit 2
fi
if [[ ! -f "${RECIPE_PATCH}" ]]; then
    echo "missing bundled CK recipe ${RECIPE_PATCH}" >&2
    exit 2
fi
RECIPE_SHA256="$(sha256sum "${RECIPE_PATCH}" | cut -d' ' -f1)"
if [[ "${RECIPE_SHA256}" != "${REQUIRED_RECIPE_SHA256}" ]]; then
    echo "gfx11 CK recipe SHA256 mismatch: ${RECIPE_SHA256}" >&2
    exit 2
fi

reset_dir() {
    local directory="$1"
    mkdir -p "${directory}"
    find "${directory}" -mindepth 1 -delete
}

reset_dir "${BUILD_DIR}/generated"
reset_dir "${BUILD_DIR}/objects"
reset_dir "${CK_ROOT}"

git -C "${EXTERNAL_CK_ROOT}" archive "${REQUIRED_CK_REV}" | tar -x -C "${CK_ROOT}"
patch -d "${CK_ROOT}" -p1 < "${RECIPE_PATCH}"

FMHA_DIR="${CK_ROOT}/example/ck_tile/01_fmha"
GENERATOR="${FMHA_DIR}/generate.py"
if [[ ! -f "${FMHA_DIR}/fmha_fwd.hpp" || ! -f "${GENERATOR}" ]]; then
    echo "missing CK FMHA source under ${FMHA_DIR}" >&2
    exit 2
fi

LIST="${BUILD_DIR}/sources.list"
FILTER="*d*_fp16_batch*_nlogits_nbias_*nlse_ndropout_nskip_nqscale_ntrload*"

python3 "${GENERATOR}" \
    --targets "${GENERATOR_TARGET}" \
    --api fwd \
    --receipt 2 \
    --optdim "${HEAD_DIMS}" \
    --filter "${FILTER}" \
    --list_blobs "${LIST}"

if [[ "$(wc -l < "${LIST}")" -ne "${EXPECTED_SOURCES}" ]]; then
    echo "expected ${EXPECTED_SOURCES} ${GENERATOR_TARGET} FP16 D64/D128/D256 generated sources" >&2
    echo "the supplied FlashAttention tree does not carry the required CK recipe" >&2
    exit 2
fi

python3 "${GENERATOR}" \
    --targets "${GENERATOR_TARGET}" \
    --api fwd \
    --receipt 2 \
    --optdim "${HEAD_DIMS}" \
    --filter "${FILTER}" \
    --output_dir "${BUILD_DIR}/generated"

COMMON_FLAGS=(
    -std=c++20
    -O3
    -fPIC
    --offload-arch="${GPU_ARCH}"
    -D"${TARGET_DEFINE}"=1
    -DCK_TILE_FMHA_FWD_FAST_EXP2=1
    -fgpu-flush-denormals-to-zero
    -DCK_ENABLE_BF16
    -DCK_ENABLE_FP16
    -DCK_ENABLE_FP32
    -DCK_ENABLE_FP64
    -DCK_ENABLE_INT8
    -D__HIP_PLATFORM_HCC__=1
    -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3
    -Wno-pass-failed
    -mllvm --lsr-drop-solution=1
    -fno-offload-uniform-block
    -mllvm -enable-post-misched=0
    -mllvm -amdgpu-early-inline-all=true
    -mllvm -amdgpu-function-calls=false
    -I"${ROOT}"
    -I"${REPO_ROOT}/kernels/src"
    -I"${FMHA_DIR}"
    -I"${CK_ROOT}/include"
    -I"${CK_ROOT}/library/include"
)

mapfile -t GENERATED_SOURCES < <(find "${BUILD_DIR}/generated" -maxdepth 2 -type f -name 'fmha_fwd*.cpp' | sort)
if [[ "${#GENERATED_SOURCES[@]}" -ne "${EXPECTED_SOURCES}" ]]; then
    echo "generated ${#GENERATED_SOURCES[@]} sources, expected ${EXPECTED_SOURCES}" >&2
    exit 2
fi

compile_one() {
    local source="$1"
    local stem
    stem="$(basename "${source}" .cpp)"
    "${ROCM_PATH}/bin/hipcc" "${COMMON_FLAGS[@]}" -x hip -c "${source}" \
        -o "${BUILD_DIR}/objects/${stem}.o"
}
export -f compile_one

for source in "${GENERATED_SOURCES[@]}" "${ROOT}/hipfire_flash_attn_ck.cpp"; do
    compile_one "${source}" &
    while (( "$(jobs -pr | wc -l)" >= MAX_JOBS )); do
        wait -n
    done
done
wait

"${ROCM_PATH}/bin/hipcc" \
    -shared \
    -Wl,-z,defs \
    -Wl,-rpath,"${HIP_LIB_DIR}" \
    "${BUILD_DIR}"/objects/*.o \
    -lamdhip64 \
    -o "${OUT}"

"${ROCM_PATH}/bin/hipcc" \
    -std=c++20 \
    -O2 \
    -D"${TARGET_DEFINE}"=1 \
    -I"${ROOT}" \
    "${ROOT}/smoke_raw_abi.cpp" \
    -L"${BUILD_DIR}" \
    -Wl,-rpath,'$ORIGIN' \
    -Wl,-rpath,"${HIP_LIB_DIR}" \
    -lhipfire_flash_attn_ck \
    -o "${BUILD_DIR}/smoke_raw_abi"

file "${OUT}"
du -h "${OUT}"
echo "${OUT}"
