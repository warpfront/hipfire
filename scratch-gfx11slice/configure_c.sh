#!/usr/bin/env bash
set -euo pipefail
root=/home/kaden/hipfire-gfx11slice
out="$root/scratch-gfx11slice"
cli="$root/target/release/hipfire"
export HIP_VISIBLE_DEVICES=0 HIPFIRE_HOME="${HIPFIRE_HOME:-$out/home-triple}"
mkdir -p "$HIPFIRE_HOME" "$out/triple-trace"
"$cli" config set kernel.gfx11_iu4_gridspec true
"$cli" config set kernel.gfx11_iu4_shape true
"$cli" config set kernel.gfx11_iu4_symfold true
"$cli" config set kernel.gfx11_a4_candidates 2
"$cli" config set kernel.gfx11_lean_pbs false
