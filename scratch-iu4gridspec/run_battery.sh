#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 gfx1100|gfx1151 HIP_VISIBLE_DEVICES LABEL" >&2
  exit 2
fi
arch="$1"
device="$2"
label="$3"
root=/home/kaden/hipfire-iu4gridspec
model=/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq
export HIP_VISIBLE_DEVICES="$device"
export HIPFIRE_DAEMON_BIN="$root/target/release/daemon"
export HIPFIRE_CLI_BIN="$root/target/release/hipfire"
python3 "$root/scripts/serve_harness.py" --mode battery --model "$model" --thinking off \
  2>&1 | tee "$root/scratch-iu4gridspec/battery-${label}.console"
