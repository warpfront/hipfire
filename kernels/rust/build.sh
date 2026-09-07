#!/usr/bin/env bash
set -euo pipefail
# Build the Rust-native VCN preprocess kernels for one gfx target.
# Usage: ./build.sh <gfxNNNN> [--fp-contract]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GFX="${1:?usage: build.sh <gfxNNNN> [--fp-contract]}"
FPC=0
if [[ "${2:-}" == "--fp-contract" ]]; then FPC=1; fi
if [[ $FPC == 1 ]]; then
  export RUSTFLAGS="-C target-cpu=${GFX} -C llvm-args=-fp-contract=fast"
  export CARGO_TARGET_DIR="${HERE}/target/build-fpc"
  OUT="${HERE}/target/${GFX}/vl_yuv_preprocess-fpc.elf"
else
  export RUSTFLAGS="-C target-cpu=${GFX}"
  export CARGO_TARGET_DIR="${HERE}/target/build-default"
  OUT="${HERE}/target/${GFX}/vl_yuv_preprocess.elf"
fi
cd "${HERE}"
cargo +nightly build --release --target amdgcn-amd-amdhsa -Zbuild-std=core
mkdir -p "$(dirname "${OUT}")"
cp "${CARGO_TARGET_DIR}/amdgcn-amd-amdhsa/release/vl_yuv_preprocess.elf" "${OUT}"
echo "wrote ${OUT}"
