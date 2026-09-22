#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 gfx1100|gfx1151 HIP_VISIBLE_DEVICES" >&2
  exit 2
fi
arch="$1"
device="$2"
cd /home/kaden/hipfire-gfx1100shape/scratch-iu4resid
export HIP_VISIBLE_DEVICES="$device"
log="exact-${arch}.log"
: > "$log"
for shape in gate down; do
  for n in 5909 8192; do
    EXPECT_ARCH="$arch" "./bench_resid_${arch}" "$shape" "$n" | tee -a "$log"
    EXPECT_ARCH="$arch" "./bench_resid_${arch}" "$shape" "$n" --reverse | tee -a "$log"
    EXPECT_ARCH="$arch" "./bench_resid_${arch}" "$shape" "$n" --reverse | tee -a "$log"
    EXPECT_ARCH="$arch" "./bench_resid_${arch}" "$shape" "$n" | tee -a "$log"
  done
done
