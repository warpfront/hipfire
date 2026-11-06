#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Ratchet: arch-id match forms in the multi-card front doors.
#
# Multi-GPU routing is declared by the carrier (`Carrier::topology_caps`,
# crates/hipfire-loader/src/topology_caps.rs), not matched on an arch id. This
# counts the arch-id match forms that remain, in non-test code, in the files
# that front the multi-card routes, so a new `matches!(arch_id, ..)`,
# `arch_id == N` or `match .. arch_id` there fails the gate. The ceilings live
# in scripts/leanup-thresholds.txt (metrics `topology_arch_arms_*`); lowering
# one is encouraged, raising one needs a RATCHET-RAISE line in the commit
# message (scripts/ratchet-diff.sh).
#
# Forms counted (comment-only lines and `#[cfg(test)]` items are skipped):
#   matches!(.. arch_id ..          e.g. matches!(arch_id, 5 | 6)
#   arch_id[()] ==/!= <value>       e.g. arch_id == QWEN4_ARCH_ID
#   match .. arch_id .. {           e.g. match hfq.arch_id {
#
# Usage (repository root or any directory):
#   scripts/check-topology-arch-arms.sh             print metrics, assert thresholds
#   scripts/check-topology-arch-arms.sh --metrics   print metrics only (used by
#                                                   scripts/leanup-ratchets.sh,
#                                                   which asserts them)
# Exit: 0 within ceilings, 1 a ceiling exceeded, 2 harness error.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

METRICS_ONLY=0
if [ "${1:-}" = "--metrics" ]; then METRICS_ONLY=1; fi

# metric-name  file
FILES=(
  "admission crates/hipfire-loader/src/admission.rs"
  "lib crates/hipfire-loader/src/lib.rs"
  "bench_loader crates/hipfire-loader/src/bench_topology.rs"
  "caps crates/hipfire-loader/src/topology_caps.rs"
  "generate_qwen crates/hipfire-generate/src/qwen.rs"
  "generate_bench crates/hipfire-generate/src/bench_topology.rs"
)

# Count the forms in one file, skipping comment lines and `#[cfg(test)]` items
# (a `mod` block ends at the next column-0 `}`; a one-line item at its `;`).
count_file() {
  awk '
    function counted(line) {
      if (line ~ /matches!\([[:space:]]*[A-Za-z_.&*()]*arch_id/) return 1
      if (line ~ /arch_id(\(\))?[[:space:]]*(==|!=)[[:space:]]*[A-Za-z0-9_:(]/) return 1
      if (line ~ /(^|[^A-Za-z_])match[[:space:]]+[A-Za-z0-9_.&*]*arch_id(\(\))?[[:space:]]*\{/) return 1
      return 0
    }
    skipping == 1 {
      if ($0 ~ /^}/) { skipping = 0 }
      else if ($0 ~ /;[[:space:]]*$/ && !seen_open) { skipping = 0 }
      if ($0 ~ /\{/) seen_open = 1
      next
    }
    /^#\[cfg\(test\)\]/ { skipping = 1; seen_open = 0; next }
    /^[[:space:]]*\/\// { next }
    { n += counted($0) }
    END { print n + 0 }
  ' "$1"
}

declare -A METRIC
total=0
for entry in "${FILES[@]}"; do
  name="${entry%% *}"
  file="${entry#* }"
  if [ ! -f "$file" ]; then
    echo "check-topology-arch-arms: missing $file" >&2
    exit 2
  fi
  n="$(count_file "$file")"
  METRIC["topology_arch_arms_$name"]="$n"
  total=$((total + n))
  printf 'topology_arch_arms_%s %s\n' "$name" "$n"
done
METRIC["topology_arch_arms_total"]="$total"
printf 'topology_arch_arms_total %s\n' "$total"

if [ "$METRICS_ONLY" -eq 1 ]; then exit 0; fi

THRESH="scripts/leanup-thresholds.txt"
[ -f "$THRESH" ] || { echo "check-topology-arch-arms: $THRESH missing" >&2; exit 2; }
fails=0
checked=0
while read -r metric op want; do
  case "$metric" in topology_arch_arms_*) ;; *) continue ;; esac
  got="${METRIC[$metric]:-}"
  if [ -z "$got" ]; then
    echo "check-topology-arch-arms: FAIL threshold names unknown metric '$metric'"
    fails=$((fails + 1))
    continue
  fi
  checked=$((checked + 1))
  case "$op" in
    "==") [ "$got" = "$want" ] || { echo "check-topology-arch-arms: FAIL $metric = $got, must be $want"; fails=$((fails + 1)); } ;;
    "<=") if [ "$got" -gt "$want" ]; then
            echo "check-topology-arch-arms: FAIL $metric = $got, ceiling $want"
            fails=$((fails + 1))
          fi ;;
    *) echo "check-topology-arch-arms: FAIL unknown operator '$op' for $metric"; fails=$((fails + 1)) ;;
  esac
done < "$THRESH"
if [ "$checked" -eq 0 ]; then
  echo "check-topology-arch-arms: FAIL no topology_arch_arms_* threshold in $THRESH (nothing asserted)"
  exit 1
fi
if [ "$fails" -gt 0 ]; then
  echo "check-topology-arch-arms: $fails violation(s). Declare the capability on the carrier"
  echo "  (Carrier::topology_caps) instead of matching an arch id, or see scripts/ratchet-diff.sh to raise."
  exit 1
fi
echo "check-topology-arch-arms: OK ($checked metric(s) asserted)"
