#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# Runner for the sampled-MTP lane suite
# (`crates/hipfire-arch-qwen4/tests/sampled_mtp_distribution_hw.rs`,
# `HIPFIRE_SAMPLED_MTP_EXECUTOR=lanes`): one lane arm over one `START..END`
# trial shard, the CPU shard merge, the CPU/judging verdict, and the CPU
# self-test. It only sets the environment each arm requires (the test refuses
# a stale variable) and execs `cargo test`; device selection, the fabric
# reservation and the job time limit are the caller's.
#
# Usage:
#   scripts/sampled_lane_suite.sh shard ARM START..END   one arm shard (GPU)
#   scripts/sampled_lane_suite.sh merge                  join shards (CPU)
#   scripts/sampled_lane_suite.sh verdict                judge merged arms (CPU)
#   scripts/sampled_lane_suite.sh selftest               host-only tests (CPU)
#
# Required environment (shard/merge/verdict):
#   MODEL     canonical artifact (HIPFIRE_MTP_IDENTITY_MODEL)
#   OUT       one fresh directory PER executor/topology/lane count
#   TOPOLOGY  single | pp2 | ep2           (HIPFIRE_SAMPLED_MTP_TOPOLOGY)
#   LANES     1 | 8                        (HIPFIRE_SAMPLED_MTP_LANES)
# Optional: TRIALS (default 2000), PROMPT (default prose), CARGO_TARGET_DIR,
#   DRY_RUN=1 (print the environment and command, run nothing).
#
# ARM is one of lane_ar lane_batched lane_adaptive lane_copy. Cover 0..TRIALS
# with disjoint shards for EACH arm, then `merge`, then `verdict`; `verdict`
# never launches an arm. Until the sampled lane executor lands (S3/S5, S4) a
# `shard` run validates its configuration and fails with "not yet
# implemented by S3/S5"; `selftest` is the only runnable part.

set -euo pipefail

TEST_BIN=(cargo test --release -p hipfire-arch-qwen4 --test sampled_mtp_distribution_hw)
TEST_ARGS=(-- --exact --ignored --nocapture --test-threads=1)
ARMS="lane_ar lane_batched lane_adaptive lane_copy"

die() { echo "sampled_lane_suite: $*" >&2; exit 2; }

run() {
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "+ $*"
  else
    "$@"
  fi
}

common_env() {
  : "${MODEL:?set MODEL}" "${OUT:?set OUT}" "${TOPOLOGY:?set TOPOLOGY=single|pp2|ep2}" "${LANES:?set LANES=1|8}"
  case "$TOPOLOGY" in single | pp2 | ep2) ;; *) die "TOPOLOGY=$TOPOLOGY: expected single|pp2|ep2" ;; esac
  case "$LANES" in 1 | 8) ;; *) die "LANES=$LANES: expected 1|8" ;; esac
  export HIPFIRE_MTP_IDENTITY_MODEL="$MODEL" HIPFIRE_SAMPLED_MTP_OUT="$OUT"
  export HIPFIRE_SAMPLED_MTP_EXECUTOR=lanes HIPFIRE_SAMPLED_MTP_LANES="$LANES"
  export HIPFIRE_SAMPLED_MTP_TOPOLOGY="$TOPOLOGY"
  export HIPFIRE_SAMPLED_MTP_TRIALS="${TRIALS:-2000}" HIPFIRE_SAMPLED_MTP_PROMPT="${PROMPT:-prose}"
  export HIPFIRE_MTP_SAMPLED=1 HIPFIRE_MTP_SAMPLED_MODE=leviathan
  unset HIPFIRE_MTP_TRACE HIPFIRE_MTP_PHASE_TIMING HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS
  unset HIPFIRE_MTP_INCREMENTAL HIPFIRE_MTP_AR_FLOOR HIPFIRE_MTP_NGRAM_COPY
  unset HIPFIRE_SAMPLED_MTP_ARM HIPFIRE_SAMPLED_MTP_SHARD
}

show_env() {
  if [ "${DRY_RUN:-0}" = 1 ]; then
    env | grep -E '^(HIPFIRE_|HIP_VISIBLE|ROCR_VISIBLE)' | sort | sed 's/^/env /'
  fi
}

case "${1:-}" in
  shard)
    arm="${2:?usage: $0 shard ARM START..END}"
    range="${3:?usage: $0 shard ARM START..END}"
    case " $ARMS " in *" $arm "*) ;; *) die "arm $arm: expected one of: $ARMS" ;; esac
    case "$range" in [0-9]*..[0-9]*) ;; *) die "range $range: expected START..END" ;; esac
    common_env
    export HIPFIRE_SAMPLED_MTP_ARM="$arm" HIPFIRE_SAMPLED_MTP_SHARD="$range"
    case "$arm" in
      lane_batched) export HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_AR_FLOOR=0 ;;
      lane_copy) export HIPFIRE_MTP_NGRAM_COPY=on ;;
    esac
    show_env
    run "${TEST_BIN[@]}" sampled_mtp_distribution_arm "${TEST_ARGS[@]}"
    ;;
  merge)
    common_env
    show_env
    run "${TEST_BIN[@]}" sampled_mtp_merge_short_shards "${TEST_ARGS[@]}"
    ;;
  verdict)
    common_env
    show_env
    run "${TEST_BIN[@]}" sampled_mtp_matches_ar_distribution_on_flash_next "${TEST_ARGS[@]}"
    ;;
  selftest)
    # Host-only: GPUs hidden, no model. A zero-test filter is a failure.
    export HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
    log="$(mktemp "${TMPDIR:-/tmp}/lane-suite-selftest.XXXXXX")"
    trap 'rm -f "$log"' EXIT
    run cargo test -p hipfire-arch-qwen4 --test sampled_mtp_distribution_hw -- lane_suite lane_run 2>&1 | tee "$log"
    if [ "${DRY_RUN:-0}" != 1 ] && ! grep -Eq 'test result: ok\. [1-9][0-9]* passed' "$log"; then
      die "selftest ran zero tests (a zero-test filter is not a pass)"
    fi
    ;;
  *)
    sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
