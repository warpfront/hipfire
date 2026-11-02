#!/usr/bin/env bash
# Detached launcher on hipx: ONE fabric timing hold on xtx running gap-job.sh over the ABBAAB plan.
#   gap-launch.sh NAME PLAN_DIR [MAX_SECS]
# PLAN_DIR comes from `gap_plan.py make` (plan.tsv + pins.json + candidate.toml). Everything lands in $GAP_RUNS/NAME
# (must not exist). Fabric admission is fixed by the qualification contract: class timing, card xtx, mem 22.4 GiB,
# host-mem 67 GiB lock, max 1800 s, priority P1, D2 off (pinned in the cells). The hold runs under the frozen R7
# kernel-watcher wrapper in its own session (nohup setsid, every fd > 2 closed). Waiters only test for
# $GAP_RUNS/NAME/watch.exit. MemAvailable >= 80 GiB is enforced by gap-cell.sh at each cell start inside the hold.
# Wrapper default is the canonical wt-hold1-r7-harness path; GAP_WRAP_OVERRIDE (absolute path) wins when set.
WRAP=${GAP_WRAP_OVERRIDE:-/home/kaden/qcal/release-0.4.3/slowlink-prefill/wt-hold1-r7-harness/scripts/slowlink-holds/run-with-watcher.sh}
FABRIC=/home/kaden/qcal/locks/fabric/fabric-run.sh
set -u
export PYTHONDONTWRITEBYTECODE=1
NAME=$1 PLAN_DIR=${2%/} MAXS=${3:-1800}
PRIO=P1
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=${GAP_RUNS:-/home/kaden/qcal/release-0.4.3/slowlink-prefill/gap2000/runs}
ROOT=$RUNS/$NAME
case "$NAME" in ''|*[!A-Za-z0-9._-]*) echo "bad NAME" >&2; exit 2 ;; esac
case "$PLAN_DIR" in /*) ;; *) echo "PLAN_DIR must be absolute" >&2; exit 2 ;; esac
case "$MAXS" in ''|*[!0-9]*) echo "bad MAX_SECS" >&2; exit 2 ;; esac
[ "$MAXS" -le 1800 ] || { echo "max secs > 1800" >&2; exit 2; }
[ -e "$ROOT" ] && { echo "refusing existing $ROOT" >&2; exit 2; }
[ -x "$FABRIC" ] || { echo "missing $FABRIC" >&2; exit 2; }
[ -f "$WRAP" ] || { echo "missing $WRAP" >&2; exit 2; }
python3 "$HERE/gap_plan.py" check-plan "$PLAN_DIR/plan.tsv" >/dev/null || exit 2
avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
mkdir -p "$ROOT"
cp -p -- "$PLAN_DIR/plan.tsv" "$PLAN_DIR/pins.json" "$PLAN_DIR/candidate.toml" "$ROOT/"
mv "$ROOT/plan.tsv" "$ROOT/plan.submitted.tsv"
{
  (cd "$HERE" && sha256sum gap-*.sh gap_*.py candidate.toml)
  sha256sum "$WRAP" "$PLAN_DIR/plan.tsv" "$PLAN_DIR/pins.json" "$PLAN_DIR/candidate.toml"
} >"$ROOT/scripts.sha256"
{ printf 'harness_git_head='; git -C "$HERE" rev-parse HEAD 2>&1; git -C "$HERE" status --short -- . 2>&1; } \
  >"$ROOT/harness-git.txt" || true
nohup setsid bash -c '
  for f in /proc/$$/fd/*; do n=${f##*/}; [ "$n" -gt 2 ] 2>/dev/null && eval "exec $n>&-"; done
  exec env OUT_PREFIX="$1/watch" PYTHONDONTWRITEBYTECODE=1 bash "$2" "$3" --class timing --card xtx --mem-gb 22.4 \
    --hostmem-gb 67 --hostmem-lock --max-secs "$4" --priority "$5" --owner S1Gap -- \
    bash "$6/gap-job.sh" "$7" "$1/cells"
' launcher "$ROOT" "$WRAP" "$FABRIC" "$MAXS" "$PRIO" "$HERE" "$PLAN_DIR/plan.tsv" \
  </dev/null >"$ROOT/launcher.log" 2>&1 &
LPID=$!
sleep 1
printf '{"name":"%s","pid":%s,"pgid":"%s","boot_id":"%s","mem_available_kib_at_launch":%s,"priority":"%s","max_secs":%s,"launch_unix":%s}\n' \
  "$NAME" "$LPID" "$(ps -o pgid= -p "$LPID" 2>/dev/null | tr -d ' ')" "$(cat /proc/sys/kernel/random/boot_id)" \
  "$avail_kb" "$PRIO" "$MAXS" "$(date +%s.%N)" >"$ROOT/launch.json"
cat "$ROOT/launch.json"
