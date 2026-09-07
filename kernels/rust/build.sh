#!/usr/bin/env bash
set -euo pipefail
# Build one Rust-native kernel crate for one gfx target.
# Usage: ./build.sh <crate> <gfxNNNN> [--fp-contract]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="${1:?usage: build.sh <crate> <gfxNNNN> [--fp-contract]}"
GFX="${2:?usage: build.sh <crate> <gfxNNNN> [--fp-contract]}"
FPC=0
if [[ "${3:-}" == "--fp-contract" ]]; then FPC=1; fi
if [[ $FPC == 1 ]]; then
  export RUSTFLAGS="-C target-cpu=${GFX} -C llvm-args=-fp-contract=fast"
  export CARGO_TARGET_DIR="${HERE}/target/build-fpc"
  OUT="${HERE}/target/${GFX}/${CRATE}-fpc.elf"
else
  export RUSTFLAGS="-C target-cpu=${GFX}"
  export CARGO_TARGET_DIR="${HERE}/target/build-default"
  OUT="${HERE}/target/${GFX}/${CRATE}.elf"
fi
cd "${HERE}"
cargo +nightly build --release -p "${CRATE}" --target amdgcn-amd-amdhsa -Zbuild-std=core
mkdir -p "$(dirname "${OUT}")"
cp "${CARGO_TARGET_DIR}/amdgcn-amd-amdhsa/release/${CRATE}.elf" "${OUT}"
echo "wrote ${OUT}"
