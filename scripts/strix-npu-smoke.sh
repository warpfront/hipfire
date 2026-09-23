#!/usr/bin/env bash
# Probe a Strix Halo/Ryzen AI XDNA NPU and optionally run a tiny MLIR-AIE
# vector-scalar compute smoke.
#
# Default mode is non-mutating and only checks device/runtime readiness:
#   scripts/strix-npu-smoke.sh
#
# Full smoke mode expects a local mlir-aie checkout with the IRON venv:
#   HIPFIRE_NPU_RUN_MLIR_AIE_SMOKE=1 scripts/strix-npu-smoke.sh
#
# Useful overrides:
#   MLIR_AIE_DIR=/path/to/mlir-aie
#   HIPFIRE_NPU_MIN_MEMLOCK_BYTES=134217728
#   HIPFIRE_NPU_DEVICE_NAME=npu2

set -euo pipefail

MIN_MEMLOCK_BYTES="${HIPFIRE_NPU_MIN_MEMLOCK_BYTES:-134217728}"
RUN_MLIR_AIE_SMOKE="${HIPFIRE_NPU_RUN_MLIR_AIE_SMOKE:-0}"
MLIR_AIE_DIR="${MLIR_AIE_DIR:-$HOME/mlir-aie}"
NPU_DEVICE_NAME="${HIPFIRE_NPU_DEVICE_NAME:-npu2}"

say() {
    printf '[strix-npu] %s\n' "$*"
}

fail() {
    printf '[strix-npu] ERROR: %s\n' "$*" >&2
    exit 1
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

memlock_bytes() {
    python3 - <<'PY'
import resource
soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
print(soft if soft >= 0 else 2**63 - 1)
PY
}

groups_csv() {
    id -G | tr ' ' ','
}

run_with_raised_memlock() {
    if ! have_cmd sudo || ! have_cmd prlimit || ! have_cmd setpriv; then
        return 127
    fi
    sudo -n prlimit --memlock="${MIN_MEMLOCK_BYTES}:${MIN_MEMLOCK_BYTES}" -- \
        setpriv --reuid="$(id -u)" --regid="$(id -g)" --groups="$(groups_csv)" "$@"
}

run_xrt_smi() {
    xrt-smi examine 2>&1
}

probe_xrt() {
    have_cmd xrt-smi || fail "xrt-smi not found"
    [[ -e /dev/accel/accel0 ]] || fail "/dev/accel/accel0 not found"

    say "device node: $(ls -l /dev/accel/accel0)"
    say "memlock soft limit: $(memlock_bytes) bytes"

    local out rc
    set +e
    out="$(run_xrt_smi)"
    rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
        say "xrt-smi examine works with current user limits"
        printf '%s\n' "$out"
        return 0
    fi

    say "xrt-smi examine failed with current user limits"
    printf '%s\n' "$out" >&2

    if [[ "$out" == *"Resource temporarily unavailable"* ]]; then
        say "failure matches low RLIMIT_MEMLOCK; retrying with ${MIN_MEMLOCK_BYTES} byte memlock"
        run_with_raised_memlock xrt-smi examine
        return 0
    fi

    return "$rc"
}

derive_peano_dir() {
    python3 - <<'PY'
import os
import site

for base in site.getsitepackages():
    candidate = os.path.join(base, "llvm-aie")
    if os.path.isdir(candidate):
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

derive_mlir_aie_include_dir() {
    python3 - <<'PY'
import os
import site

for base in site.getsitepackages():
    candidate = os.path.join(base, "mlir_aie", "include")
    if os.path.isdir(candidate):
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_mlir_aie_smoke() {
    [[ -d "$MLIR_AIE_DIR" ]] || fail "MLIR_AIE_DIR not found: $MLIR_AIE_DIR"
    [[ -f "$MLIR_AIE_DIR/ironenv/bin/activate" ]] || fail "IRON venv not found under $MLIR_AIE_DIR/ironenv"

    # shellcheck disable=SC1091
    source "$MLIR_AIE_DIR/ironenv/bin/activate"

    local mlir_aie_include_dir
    mlir_aie_include_dir="${MLIR_AIE_INCLUDE_DIR:-$(derive_mlir_aie_include_dir)}"
    [[ -d "$mlir_aie_include_dir" ]] || fail "mlir_aie include dir not found: $mlir_aie_include_dir"
    export CPLUS_INCLUDE_PATH="${mlir_aie_include_dir}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"

    local example_dir="$MLIR_AIE_DIR/programming_examples/basic/vector_scalar_mul"
    [[ -d "$example_dir" ]] || fail "vector_scalar_mul example not found: $example_dir"

    local peano_dir
    peano_dir="${PEANO_INSTALL_DIR:-$(derive_peano_dir)}"
    [[ -x "$peano_dir/bin/clang++" ]] || fail "llvm-aie clang++ not found under PEANO_INSTALL_DIR=$peano_dir"

    say "building MLIR-AIE vector_scalar_mul for ${NPU_DEVICE_NAME}"
    (
        cd "$example_dir"
        make clean >/dev/null 2>&1 || true
        PEANO_INSTALL_DIR="$peano_dir" make "devicename=${NPU_DEVICE_NAME}" \
            build/final_in1_size.xclbin build/insts_in1_size.bin

        rm -rf _build
        mkdir -p _build
        cd _build
        cmake "$example_dir" \
            -DTARGET_NAME=vector_scalar_mul_in1_size \
            -DIN1_SIZE=8192 \
            -DIN2_SIZE=4 \
            -DOUT_SIZE=8192 \
            -DINT_BIT_WIDTH=16 \
            -DXRT_INC_DIR=/usr/include \
            -DXRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
            -DCMAKE_C_COMPILER=gcc \
            -DCMAKE_CXX_COMPILER=g++
        cmake --build . --config Release
        cp vector_scalar_mul_in1_size ../vector_scalar_mul_in1_size.exe
    )

    say "running MLIR-AIE vector_scalar_mul"
    (
        cd "$example_dir"
        run_with_raised_memlock ./vector_scalar_mul_in1_size.exe \
            -x build/final_in1_size.xclbin \
            -i build/insts_in1_size.bin \
            -k MLIR_AIE
    )
}

probe_xrt

if [[ "$RUN_MLIR_AIE_SMOKE" == "1" ]]; then
    run_mlir_aie_smoke
else
    say "set HIPFIRE_NPU_RUN_MLIR_AIE_SMOKE=1 to run the MLIR-AIE compute smoke"
fi
