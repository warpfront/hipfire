#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 gfx1100|gfx1151 HIP_VISIBLE_DEVICES [label]" >&2
  exit 2
fi
arch="$1"
device="$2"
label="${3:-}"
cd /home/kaden/hipfire-iu4gridspec/scratch-iu4gridspec
export HIP_VISIBLE_DEVICES="$device"
log="exact-${label:+${label}-}${arch}.log"
mode=()
if [[ "$label" == "branch" ]]; then
  mode=(--branch)
elif [[ "$label" == "ordinary" ]]; then
  mode=(--ordinary-split)
fi
: > "$log"
for shape in gate down; do
  for n in 5909 8192; do
    EXPECT_ARCH="$arch" "./bench_gridspec_${arch}" "$shape" "$n" "${mode[@]}" | tee -a "$log"
    EXPECT_ARCH="$arch" "./bench_gridspec_${arch}" "$shape" "$n" --reverse "${mode[@]}" | tee -a "$log"
    EXPECT_ARCH="$arch" "./bench_gridspec_${arch}" "$shape" "$n" --reverse "${mode[@]}" | tee -a "$log"
    EXPECT_ARCH="$arch" "./bench_gridspec_${arch}" "$shape" "$n" "${mode[@]}" | tee -a "$log"
  done
done
