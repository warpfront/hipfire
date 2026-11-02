#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# Regenerate the PM-native coop ring code object embedded by npu-tools (`coop_gpu`); no vendor assembler/compiler.
# Build peacemaker from the approved PM tree with its lockfile and a separate target (see build-gpu-bw.sh):
# CARGO_TARGET_DIR=/path/out cargo build --locked --manifest-path /path/PM/Cargo.toml -p peacemaker-isa --features toolchain --bin peacemaker
# PM reference used for this artifact: wt-land-041m f7960b4e8, peacemaker_isa::native::assemble.
# PEACEMAKER=/path/out/debug/peacemaker bash crates/npu-tools/native/build-coop-ring.sh
set -euo pipefail
here=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"${PEACEMAKER:-peacemaker}" native --arch=gfx1151 "$here/coop_ring_gfx1151.s" \
    --co="$here/coop_ring_gfx1151.co"
