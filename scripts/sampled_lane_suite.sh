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
#   LANES     1..8 (any count 1 to 8)      (HIPFIRE_SAMPLED_MTP_LANES)
#   LANE_K    2 | 3 (default 2 = the shipped lane draft depth; 3 = the older
#             default, a sensitivity run)  (HIPFIRE_SAMPLED_MTP_LANE_K). Every
#             shard, the merge and the verdict of one cohort MUST use the same
#             value: it is part of every shard's merge key and identity.
# Optional: TRIALS (default 2000), PROMPT (default prose), CARGO_TARGET_DIR,
#   STATE_QUANT (GDN test state, default q8), DRY_RUN=1 (print the environment
#   and command, run nothing), TEST_EXE (a staged test executable run instead
#   of `cargo test`: the same executable MUST run every shard, the merge and
#   the verdict, its hash is part of every config).
#
# ARM is one of lane_ar lane_batched lane_adaptive lane_copy lane_cross. Cover
# 0..TRIALS with disjoint shards for EACH arm of the cohort, then `merge`, then
# `verdict`; `verdict` never launches an arm. TOPOLOGY=ep2 (cards C,B;
# HIPFIRE_DEVICES from the fabric job, a HIPFIRE_HOME whose
# cache/verified-artifacts.json holds the model, its .ple-q4 and .expert-hotset
# hashes) runs the EP2 lane executor (`lane_ep.rs`): one process per arm and
# shard, `START..END` of a few hundred trials. HIPFIRE_TOPOLOGY_EP=2 is
# exported here. The fixed-count native-MTP arms (lane_batched, lane_adaptive,
# lane_copy) pin HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=8 so that N lanes run native
# windows at every N (a native-window coverage arm, not the shipped switching
# policy). lane_cross (ep2, LANES=8 only) is the shipped switching policy: it
# pins HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=3 (the shipped lane_switch default) and
# admits 3 of a trial's 8 requests at tick 0 and 5 more once the first group
# has prefilled, so early lanes' first window runs native (3 active) and later
# ones AR (8 active; 4 active is the hold case). The native arms (lane_batched,
# lane_adaptive, lane_cross) also pin HIPFIRE_EP_MTP_LANE_K=$LANE_K; lane_ar
# pins nothing.
# Give `merge` and `verdict` the same job environment as the shards (the config
# identity compares every HIPFIRE_* variable). The ep2 cohort is lane_ar
# lane_batched lane_adaptive, plus lane_cross when LANES=8 (`lane_copy` is
# refused under ep2). The single and pp2 topologies validate their
# configuration and refuse ("ep2 only"); `selftest` is the host-only part.
#
# Draw route: with HIPFIRE_QWEN4_LANE_DEVICE_SAMPLE on (the default on gfx1201)
# every sampled draw of every ep2 arm (AR rows, the prompt-completing first
# token, native and copy verify windows) is one lane-wide device launch;
# `=0` restores the host sampler. Each arm records the route it RESOLVED
# (`device` or `host`) as lane_suite.lane_draw_route in its config: ids of one
# route are a config error under the other, shards of two routes never merge,
# and `verdict` prints every arm's route and refuses a cohort that mixes them.
# Run every arm of a cohort under the same setting.

set -euo pipefail

TEST_BIN=(cargo test --release -p hipfire-arch-qwen4 --test sampled_mtp_distribution_hw)
TEST_ARGS=(-- --exact --ignored --nocapture --test-threads=1)
ARMS="lane_ar lane_batched lane_adaptive lane_copy lane_cross"

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
  : "${MODEL:?set MODEL}" "${OUT:?set OUT}" "${TOPOLOGY:?set TOPOLOGY=single|pp2|ep2}" "${LANES:?set LANES=1..8}"
  LANE_K="${LANE_K:-2}"
  case "$LANE_K" in 2 | 3) ;; *) die "LANE_K=$LANE_K: expected 2|3" ;; esac
  case "$TOPOLOGY" in single | pp2 | ep2) ;; *) die "TOPOLOGY=$TOPOLOGY: expected single|pp2|ep2" ;; esac
  case "$LANES" in 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8) ;; *) die "LANES=$LANES: expected 1..8" ;; esac
  export HIPFIRE_MTP_IDENTITY_MODEL="$MODEL" HIPFIRE_SAMPLED_MTP_OUT="$OUT"
  export HIPFIRE_SAMPLED_MTP_EXECUTOR=lanes HIPFIRE_SAMPLED_MTP_LANES="$LANES" HIPFIRE_SAMPLED_MTP_LANE_K="$LANE_K"
  export HIPFIRE_SAMPLED_MTP_TOPOLOGY="$TOPOLOGY"
  export HIPFIRE_SAMPLED_MTP_TRIALS="${TRIALS:-2000}" HIPFIRE_SAMPLED_MTP_PROMPT="${PROMPT:-prose}"
  export HIPFIRE_MTP_SAMPLED=1 HIPFIRE_MTP_SAMPLED_MODE=leviathan
  export HIPFIRE_GDN_TEST_STATE_QUANT="${STATE_QUANT:-q8}"
  [ "$TOPOLOGY" = ep2 ] && export HIPFIRE_TOPOLOGY_EP=2
  unset HIPFIRE_MTP_TRACE HIPFIRE_MTP_PHASE_TIMING HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS
  unset HIPFIRE_MTP_INCREMENTAL HIPFIRE_MTP_AR_FLOOR HIPFIRE_MTP_NGRAM_COPY HIPFIRE_EP_MTP_LANE_MAX_ACTIVE HIPFIRE_EP_MTP_LANE_K
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
      lane_batched) export HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_AR_FLOOR=0 HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=8 HIPFIRE_EP_MTP_LANE_K="$LANE_K" ;;
      lane_adaptive) export HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=8 HIPFIRE_EP_MTP_LANE_K="$LANE_K" ;;
      lane_copy) export HIPFIRE_MTP_NGRAM_COPY=on HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=8 ;;
      lane_cross)
        [ "$TOPOLOGY" = ep2 ] || die "lane_cross runs under TOPOLOGY=ep2 only"
        [ "$LANES" = 8 ] || die "lane_cross needs LANES=8"
        export HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_AR_FLOOR=0 HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=3 HIPFIRE_EP_MTP_LANE_K="$LANE_K"
        ;;
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
    run cargo test -p hipfire-arch-qwen4 --test sampled_mtp_distribution_hw -- lane_suite lane_run lane_ep 2>&1 | tee "$log"
    if [ "${DRY_RUN:-0}" != 1 ] && ! grep -Eq 'test result: ok\. [1-9][0-9]* passed' "$log"; then
      die "selftest ran zero tests (a zero-test filter is not a pass)"
    fi
    ;;
  *)
    sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
