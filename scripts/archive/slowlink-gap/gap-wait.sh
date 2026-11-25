#!/usr/bin/env bash
# Waiter (run on hipx): blocks until $GAP_RUNS/NAME/watch.exit exists, then prints the summary. Read-only on the run.
#   gap-wait.sh NAME [MAX_WAIT_SECS]            blocking
#   gap-wait.sh --detach NAME [MAX_WAIT_SECS]   nohup setsid (every fd > 2 closed); output -> $GAP_RUNS/NAME/waiter.log,
#                                               exit code -> $GAP_RUNS/NAME/waiter.exit (fresh files; refused if present)
# Exit: 0 job rc 0 and clean; 1 job failed / survivors; 3 XTX drop (INCOMPLETE: stop, tell the parent, NO retry);
#       75 wait timeout. Detection only; this script never kills anything.
set -u
export PYTHONDONTWRITEBYTECODE=1
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=gap-lib.sh
. "$HERE/gap-lib.sh"
RUNS=${GAP_RUNS:-/home/kaden/qcal/release-0.4.3/slowlink-prefill/gap2000/runs}
if [ "${1:-}" = --detach ]; then
  shift
  NAME=$1 MAXW=${2:-3500}
  ROOT=$RUNS/$NAME
  [ -d "$ROOT" ] || { echo "no run $ROOT" >&2; exit 2; }
  for f in "$ROOT/waiter.log" "$ROOT/waiter.exit"; do [ -e "$f" ] && { echo "refusing existing $f" >&2; exit 2; }; done
  nohup setsid bash -c '
    for f in /proc/$$/fd/*; do n=${f##*/}; [ "$n" -gt 2 ] 2>/dev/null && eval "exec $n>&-"; done
    bash "$1" "$2" "$3"; echo $? >"$4"
  ' waiter "$HERE/gap-wait.sh" "$NAME" "$MAXW" "$ROOT/waiter.exit" </dev/null >"$ROOT/waiter.log" 2>&1 &
  printf '{"waiter_pid":%s,"log":"%s","exit_file":"%s"}\n' "$!" "$ROOT/waiter.log" "$ROOT/waiter.exit"
  exit 0
fi
NAME=$1 MAXW=${2:-3500}
ROOT=$RUNS/$NAME
end=$((SECONDS + MAXW))
while [ ! -e "$ROOT/watch.exit" ]; do
  [ "$SECONDS" -ge "$end" ] && { echo "WAIT_TIMEOUT $NAME"; tail -5 "$ROOT/watch.fabric.log" 2>/dev/null; exit 75; }
  sleep 15
done
WEXIT=$(cat "$ROOT/watch.exit")
echo "exit=$WEXIT"
tail -8 "$ROOT/watch.fabric.log" 2>/dev/null
cat "$ROOT/cells/job.log" 2>/dev/null
bad=0
for c in "$ROOT"/cells/*/; do
  [ -d "$c" ] || continue
  echo "== $c exit=$(cat "$c/exit" 2>/dev/null) leftover=$(cat "$c/leftover.txt" 2>/dev/null | wc -l) leftover-after=$(cat "$c/leftover-after.txt" 2>/dev/null | wc -l)"
  ls "$c" | tr '\n' ' '; echo
  [ "$(cat "$c/exit" 2>/dev/null)" = 0 ] || bad=1
done
if [ -e "$ROOT/cells/XTX_DROP" ]; then
  echo "XTX_DROP PRESENT: INCOMPLETE. Stop; tell the parent; do NOT retry."
  cat "$ROOT/cells/XTX_DROP"
  exit 3
fi
# Post-job survivor detection: processes the job owned (cell HIPFIRE_HOME under this run) + job/cell scripts by run name.
left=$(gap_survivors "" "$ROOT/cells/")
if [ -n "$left" ]; then
  echo "POST_JOB_SURVIVORS (detect only, nothing killed):"
  gap_describe $left
  bad=1
else
  echo "post-job: no process owned by $NAME (HIPFIRE_HOME tag scan)"
fi
pgrep -af "gap-job.sh|gap-cell.sh" | grep -F -- "$ROOT" && bad=1 || echo "post-job: no gap-job/gap-cell process for $NAME"
cat "$ROOT/watch.watchers-after.txt" 2>/dev/null
[ "$WEXIT" = 0 ] && [ "$bad" = 0 ] && exit 0
exit 1
