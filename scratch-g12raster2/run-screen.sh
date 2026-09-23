#!/usr/bin/env bash
# run-screen.sh <tag> <arms> <rounds> <config...>   config = ORDER:N:MODE, e.g. F:8192:warm
set -uo pipefail
cd "$(dirname "$0")"
tag=$1; arms=$2; rounds=$3; shift 3
export REF=${REF:-control}
export ROCR_VISIBLE_DEVICES=GPU-05f92432f2312a0e HIP_VISIBLE_DEVICES=GPU-05f92432f2312a0e
mkdir -p logs
for cfg in "$@"; do
  IFS=: read -r order n mode <<<"$cfg"
  out=logs/$tag-$order-$n-$mode.txt
  ./host time "$order" "$n" "$mode" "$arms" "${SHAPES:-}" "$rounds" > "$out" 2>&1
  echo "$cfg rc=$? parity_bad=$(grep FULLPARITY "$out" | grep -vc 'mismatches=0$')"
done
