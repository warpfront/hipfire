#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# job_sampled_lane_shards.sh SPEC... -- inside ONE fabric `correct` job on cards C,B (EP2): runs
# EP2 sampled lane-suite shards one after another, each in its own fresh process
# (`scripts/sampled_lane_suite.sh shard`, topology ep2). A SPEC is ARM:LANES:START..END, e.g.
#   lane_ar:8:0..8 lane_ar:1:0..8 lane_batched:8:0..8 lane_batched:1:0..8
#
#   /home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --cards C,B --mem-gb 30 --hostmem-gb 40 \
#     --hostmem-lock --max-secs 1500 --owner OWNER -- env BIN=... RUNROOT=... TAG=... \
#     "$BIN/job_sampled_lane_shards.sh" lane_ar:8:0..8
#
# Required env:  BIN      directory holding the staged `sampled_mtp_distribution_hw` test executable, its
#                         SOURCE, and the copies of this script and sampled_lane_suite.sh (the same BIN for
#                         every shard, the merge and the verdict: the executable's hash is in every config)
#                RUNROOT  run root; the shard outputs go to $RUNROOT/out-ep2-L<LANES> (one directory per
#                         executor/topology/lane count, shared by every arm and shard of that lane count)
#                TAG      this job's name; logs and summary.tsv go to $RUNROOT/$TAG (must not exist)
# Optional env:  MODEL (default mgfn/model/qwen3.8-flash-next-base.mq4-xts), KCACHE (default
#                mgfn/kcache-hiptrx), VHOME (HIPFIRE_HOME with cache/verified-artifacts.json holding the model,
#                .ple-q4 and .expert-hotset hashes; default $RUNROOT/hipfire-home), TRIALS (default 2000, the
#                cohort total, identical for every shard), PROMPT (default prose, identical for every arm),
#                BUDGET_SECS (default 1440: the job's own clock; a shard that cannot finish in what is left
#                is not started), EPLANE_DRY=1 (print the environment and commands, run nothing).
#
# Every process runs under the same scrubbed environment: each HIPFIRE_/HIP_/ROC_/HSA_/GPU_/AMD_/MIOPEN_
# variable of the caller is dropped except the fabric selectors, then the private HIPFIRE_HOME (verified
# cache), the kernel cache and the lane-suite environment (`sampled_lane_suite.sh`) are set. Cleanup kills
# only the shard process this script started (its own pid).
set -uo pipefail
G=/home/kaden/qcal/release-0.4.3/mgfn
SD=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
[[ $# -ge 1 ]] || { sed -n '2,/^set -uo/{/^set -uo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}" >&2; exit 2; }
BIN=${BIN:?BIN required}; BIN=$(readlink -f "$BIN"); RUNROOT=${RUNROOT:?RUNROOT required}; TAG=${TAG:?TAG required}
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad TAG '$TAG'" >&2; exit 2; }
EXE=$BIN/sampled_mtp_distribution_hw
MODEL=${MODEL:-$G/model/qwen3.8-flash-next-base.mq4-xts}; KCACHE=${KCACHE:-$G/kcache-hiptrx}
VHOME=${VHOME:-$RUNROOT/hipfire-home}; TRIALS=${TRIALS:-2000}; PROMPT=${PROMPT:-prose}; BUDGET=${BUDGET_SECS:-1440}
DRY=${EPLANE_DRY:-0}
[[ -x $EXE && -x $SD/sampled_lane_suite.sh ]] || { echo "BIN $BIN lacks the test executable or sampled_lane_suite.sh" >&2; exit 2; }
[[ -f $MODEL && -f $VHOME/cache/verified-artifacts.json ]] || { echo "model or $VHOME/cache/verified-artifacts.json missing" >&2; exit 2; }
specs=()
for spec in "$@"; do
  [[ $spec =~ ^(lane_ar|lane_batched|lane_adaptive|lane_copy):(1|8):[0-9]+\.\.[0-9]+$ ]] || { echo "bad SPEC '$spec' (ARM:LANES:START..END)" >&2; exit 2; }
  specs+=("$spec")
done
O=$RUNROOT/$TAG
if [[ $DRY != 1 ]]; then
  [[ -n "${FABRIC_JOB:-}" ]] || { echo "not inside a fabric job" >&2; exit 2; }
  [[ ! -e $O ]] || { echo "exists: $O" >&2; exit 2; }
fi
IFS=, read -r -a CARDS <<< "${FABRIC_CARDS:-}"
[[ ${#CARDS[@]} == 2 && -n "${HIPFIRE_DEVICES:-}" ]] || { echo "need exactly 2 fabric cards with HIPFIRE_DEVICES; got '${FABRIC_CARDS:-}' / '${HIPFIRE_DEVICES:-}'" >&2; exit 2; }

EV=(env)
for v in $(compgen -e); do
  case "$v" in FABRIC_*|HIPFIRE_FABRIC_ADMISSION|HIPFIRE_DEVICES) ;; HIPFIRE_*|HIP_*|ROC_*|HSA_*|GPU_*|AMD_*|MIOPEN_*) EV+=(-u "$v");; esac
done
EV+=(HOME="$RUNROOT/home" HIPFIRE_HOME="$VHOME" HIPFIRE_LOCK_DIR="$RUNROOT/lockdir" HIPFIRE_KERNEL_CACHE="$KCACHE"
     HIPFIRE_NO_REGISTRY_FETCH=1 HIPFIRE_PLE_FORMAT=q4 HIPFIRE_SESSION_CACHE_BYTES=0)
if [[ $DRY != 1 ]]; then mkdir -p "$O/logs" "$RUNROOT/home" "$RUNROOT/lockdir" || exit 2; fi

PID=""
cleanup() {
  trap - EXIT TERM INT
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID" 2>/dev/null
    for _ in $(seq 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    kill -0 "$PID" 2>/dev/null && kill -KILL "$PID" 2>/dev/null
  fi
  [[ $DRY == 1 ]] || date -u +%FT%TZ > "$O/DONE"
}
trap 'cleanup; exit 143' TERM INT
trap cleanup EXIT
if [[ $DRY != 1 ]]; then
  {
    echo "FABRIC_JOB=$FABRIC_JOB FABRIC_CARDS=$FABRIC_CARDS HIPFIRE_DEVICES=$HIPFIRE_DEVICES ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} date=$(date -u +%FT%TZ) tag=$TAG"
    echo "TRIALS=$TRIALS PROMPT=$PROMPT specs: ${specs[*]}"
    sha256sum "$EXE"; echo SOURCE:; cat "$BIN/SOURCE" 2>/dev/null || echo "(no $BIN/SOURCE)"
    echo "model=$MODEL"; stat -c 'model size=%s mtime=%y inode=%i' "$MODEL"
    echo "kernel_cache=$KCACHE verified_home=$VHOME"; uname -r; uptime; free -g
  } > "$O/pre.txt" 2>&1
  printf 'spec\trc\tstart\tend\twall_s\n' > "$O/summary.tsv"
fi
T0=$(date +%s); worst=0
for spec in "${specs[@]}"; do
  IFS=: read -r arm lanes range <<< "$spec"
  out=$RUNROOT/out-ep2-L$lanes
  left=$(( BUDGET - ($(date +%s) - T0) ))
  cmd=("${EV[@]}" MODEL="$MODEL" OUT="$out" TOPOLOGY=ep2 LANES="$lanes" TRIALS="$TRIALS" PROMPT="$PROMPT" TEST_EXE="$EXE"
       "$SD/sampled_lane_suite.sh" shard "$arm" "$range")
  if [[ $DRY == 1 ]]; then printf 'timeout -k 15 <left> '; printf '%q ' "${cmd[@]}"; echo; continue; fi
  if (( left < 120 )); then echo "skip $spec: $left s left of the $BUDGET s budget" | tee -a "$O/logs/skipped.txt"; worst=1; continue; fi
  mkdir -p "$out"
  s=$(date +%s); log=$O/logs/${arm}-L${lanes}-${range/../-}
  setsid timeout -k 15 "$left" "${cmd[@]}" > "$log.out" 2> "$log.err" &
  PID=$!
  wait "$PID"; rc=$?
  PID=""
  e=$(date +%s)
  printf '%s\t%s\t%s\t%s\t%s\n' "$spec" "$rc" "$(date -u -d @"$s" +%FT%TZ)" "$(date -u -d @"$e" +%FT%TZ)" "$((e - s))" >> "$O/summary.tsv"
  (( rc == 0 )) || worst=$rc
done
[[ $DRY == 1 ]] || echo "$worst" > "$O/rc.txt"
exit "$worst"
