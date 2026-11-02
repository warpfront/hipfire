#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-window.sh CMD...  — run one NPU submission window on hipx safely.
#
# MUST be invoked under the exclusive Halo timing lock, e.g.
#   ~/pm-wave/hipx-lock.sh timing -- ~/qcal/npu/tools/npu-window.sh ./target/release/npu-hello
# hipx-lock.sh opens fd 8 on hipx-e2e.lock, takes `flock -x` on it and `exec`s us. flock locks
# belong to the open file description, and /proc/$$/fdinfo/8 lists a `lock:` line only for the
# description that holds it; we refuse unless that line is an exclusive (WRITE) FLOCK on the
# lock file's inode.
#
# Lifecycle (timestamped to stderr and $NPU_WINDOW_LOG):
#   1. `npu-tools fclk pin` (crates/railgun/src/npu/fclk.rs, the one pin/verify/restore implementation): refuse (exit 6)
#      unless the Strix Halo iGPU (0000:bf:00.0) perf level is `auto` (driver policy) — restoring a prior manual
#      fclk mask is not attempted; pin the fabric clock through the standard amdgpu sysfs interface (kernel docs:
#      power_dpm_force_performance_level / pp_dpm_fclk): perf level `manual`, then only the top pp_dpm_fclk level,
#      then wait up to 30 s for the read-back: `manual` AND exactly one `*`, on the top line. No read-back → exit 5,
#      no NPU work. Writes go through `sudo -n tee` (passwordless sudo on hipx).
#   2. raise RLIMIT_MEMLOCK for this process tree only (prlimit on $$; not a persistent host
#      setting — /etc/security/limits.d already grants it to login sessions, ssh shells here lack it)
#   3. run CMD under `timeout` in the background and wait; INT/TERM/HUP kill and reap it first.
#      Concurrent GPU+NPU binaries (npu-coop, npu-tools m4) check the pin themselves (NPU_FCLK_GUARD).
#   4. ALWAYS restore once a pin was attempted (EXIT trap): `npu-tools fclk restore` writes `auto`, verifies it.
#   5. print amdxdna/amdgpu kernel-log lines emitted during the window.
#
# The fclk tool is $NPU_FCLK_TOOL, default <repo>/target/release/npu-tools (copy it with the other binaries).
#
# Only the Halo iGPU is pinned: it shares the SoC data fabric with the NPU. The discrete cards on
# hipx sit behind PCIe on their own fabric and are used by other agents.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
FCLK_TOOL=${NPU_FCLK_TOOL:-$ROOT/target/release/npu-tools}
LOCKF=/home/kaden/pm-wave/hipx-e2e.lock
LOG=${NPU_WINDOW_LOG:-/home/kaden/qcal/npu/logs/npu-window.log}
TMO=${NPU_WINDOW_TIMEOUT:-120}
mkdir -p "$(dirname "$LOG")"
log() { printf '%s npu-window[%s]: %s\n' "$(date -u +%FT%T.%3NZ)" "$$" "$*" | tee -a "$LOG" >&2; }

# Exclusive-lock ownership: fdinfo line "lock:	1: FLOCK  ADVISORY  WRITE <pid> <maj>:<min>:<inode> 0 EOF"
INODE=$(stat -c %i "$LOCKF")
if ! grep -Eq "^lock:.*FLOCK +ADVISORY +WRITE +[0-9]+ +[0-9a-f]+:[0-9a-f]+:${INODE} " /proc/$$/fdinfo/8 2>/dev/null; then
  log "REFUSE: fd 8 of pid $$ does not hold an exclusive flock on $LOCKF (run via ~/pm-wave/hipx-lock.sh timing --)"
  exit 3
fi

[ -x "$FCLK_TOOL" ] || { log "REFUSE: fclk tool $FCLK_TOOL missing or not executable (NPU_FCLK_TOOL)"; exit 7; }
# fclk SUBCMD: run `npu-tools fclk SUBCMD`, log its output, return its status.
fclk() {
  local out rc=0
  out=$("$FCLK_TOOL" fclk "$@" 2>&1) || rc=$?
  while IFS= read -r l; do log "$l"; done <<< "$out"
  return "$rc"
}

log "start cmd=[$*] lock=exclusive(owner $$) fclk_tool=$FCLK_TOOL"
T0=$(date +%s)
CHILD=
PINNED=0
restore() {
  rc=$?
  trap - EXIT INT TERM HUP
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
    log "killing child $CHILD before restore"; kill -TERM "$CHILD" 2>/dev/null || true
    for _ in $(seq 100); do kill -0 "$CHILD" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$CHILD" 2>/dev/null || true; wait "$CHILD" 2>/dev/null || true
  fi
  if [ "$PINNED" = 1 ]; then fclk restore || rc=4; fi
  log "kernel log during window:"
  sudo -n journalctl -k --since "@$T0" --no-pager 2>/dev/null | grep -iE 'amdxdna|amdgpu|xdna' | tee -a "$LOG" >&2 || true
  log "end rc=$rc"
  exit "$rc"
}
trap restore EXIT
trap 'log "signal received"; exit 130' INT TERM HUP

PINNED=1
fclk pin || { prc=$?; [ "$prc" = 6 ] && PINNED=0; exit "$prc"; }

sudo -n prlimit --pid $$ --memlock=unlimited:unlimited
timeout --signal=TERM --kill-after=10 "$TMO" "$@" &
CHILD=$!
set +e
wait "$CHILD"
crc=$?
set -e
CHILD=
log "cmd rc=$crc"
if fclk status; then log "pin still held after cmd"; else log "WARNING: pin NOT held after cmd"; fi
exit "$crc"
