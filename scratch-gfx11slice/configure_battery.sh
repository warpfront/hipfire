#!/usr/bin/env bash
set -euo pipefail
root=/home/kaden/hipfire-gfx11slice
out="$root/scratch-gfx11slice"
export HIPFIRE_HOME="$out/home-battery"
bash "$out/configure_c.sh"
"$root/target/release/hipfire" config set speculation.mode off
"$root/target/release/hipfire" config set kernel.gfx11_lean_pbs true
