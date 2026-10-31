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
#   STATE_QUANT (GDN test state, default q8), DRY_RUN=1 (print the environment
#   and command, run nothing), TEST_EXE (a staged test executable run instead
#   of `cargo test`: the same executable MUST run every shard, the merge and
#   the verdict, its hash is part of every config).
#
# ARM is one of lane_ar lane_batched lane_adaptive lane_copy. Cover 0..TRIALS
# with disjoint shards for EACH arm, then `merge`, then `verdict`; `verdict`
# never launches an arm. TOPOLOGY=ep2 (cards C,B; HIPFIRE_DEVICES from the
# fabric job, a HIPFIRE_HOME whose cache/verified-artifacts.json holds the
# model, its .ple-q4 and .expert-hotset hashes) runs the EP2 lane executor
# (`lane_ep.rs`): one process per arm and shard, `START..END` of a few hundred
# trials. HIPFIRE_TOPOLOGY_EP=2 is exported here, the native-MTP arms pin
# HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=8 (the 8-lane arm would otherwise run AR
# windows); give `merge` and `verdict` the same job environment as the shards
# (the config identity compares every HIPFIRE_* variable). `lane_copy` needs a
# fixture the copy pool hits: set PROMPT=repeat for the whole cohort. The single
# and pp2 topologies validate their configuration and refuse ("ep2 only");
# `selftest` is the host-only part.

set -euo pipefail

TEST_BIN=(cargo test --release -p hipfire-arch-qwen4 --test sampled_mtp_distribution_hw)
TEST_ARGS=(-- --exact --ignored --nocapture --test-threads=1)
ARMS="lane_ar lane_batched lane_adaptive lane_copy"

die() { echo "sampled_lane_suite: $*" >&2; exit 2; }

# The test command for test NAME: cargo test, or the staged executable.
test_cmd() {
  if [ -n "${TEST_EXE:-}" ]; then
    [ -x "$TEST_EXE" ] || die "TEST_EXE=$TEST_EXE is not an executable"
    run "$TEST_EXE" "$1" --exact --ignored --nocapture --test-threads=1
  else
    run "${TEST_BIN[@]}" "$1" "${TEST_ARGS[@]}"
  fi
}

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
  export HIPFIRE_GDN_TEST_STATE_QUANT="${STATE_QUANT:-q8}"
  [ "$TOPOLOGY" = ep2 ] && export HIPFIRE_TOPOLOGY_EP=2
  unset HIPFIRE_MTP_TRACE HIPFIRE_MTP_PHASE_TIMING HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS
  unset HIPFIRE_MTP_INCREMENTAL HIPFIRE_MTP_AR_FLOOR HIPFIRE_MTP_NGRAM_COPY HIPFIRE_EP_MTP_LANE_MAX_ACTIVE
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
      lane_batched) export HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_AR_FLOOR=0 HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=8 ;;
      lane_adaptive) export HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=8 ;;
      lane_copy) export HIPFIRE_MTP_NGRAM_COPY=on HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=8 ;;
    esac
    show_env
    test_cmd sampled_mtp_distribution_arm
    ;;
  merge)
    common_env
    show_env
    test_cmd sampled_mtp_merge_short_shards
    ;;
  verdict)
    common_env
    show_env
    test_cmd sampled_mtp_matches_ar_distribution_on_flash_next
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
