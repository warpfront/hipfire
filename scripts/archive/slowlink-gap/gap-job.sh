#!/usr/bin/env bash
# Runs inside ONE fabric hold: the six ABBAAB cells of PLAN_FILE in order, each a fresh process via gap-cell.sh.
#   gap-job.sh PLAN_FILE OUT_ROOT
# PLAN_FILE = plan.tsv from gap_plan.py make (validated here: exactly A1 B1 B2 A2 A3 B3, one pins file).
# Stops at the first failed cell (no automatic rerun). XTX drop -> exit 3 and an XTX_DROP marker in OUT_ROOT.
# After the last cell (or the first failure) nothing owned by the job may survive (post-job sweep, detection only
# beyond the per-cell exact-pid cleanup); then a sha256 manifest is written and OUT_ROOT is made read-only.
set -u
set +m
PLAN=$1 ROOT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=gap-lib.sh
. "$HERE/gap-lib.sh"
case "$ROOT" in /*) ;; *) echo "OUT_ROOT must be absolute" >&2; exit 2 ;; esac
case "$PLAN" in /*) ;; *) echo "PLAN_FILE must be absolute" >&2; exit 2 ;; esac
python3 "$HERE/gap_plan.py" check-plan "$PLAN" >/dev/null || exit 2
[ -e "$ROOT" ] && { echo "refusing existing OUT_ROOT $ROOT" >&2; exit 2; }
mkdir -p "$ROOT"
cp -- "$PLAN" "$ROOT/plan.tsv"
printf '%s\n' "$(date -u +%FT%TZ) job start fabric_job=${FABRIC_JOB:-} cards=${FABRIC_CARDS:-}" >>"$ROOT/job.log"
rc=0
while IFS=$'\t' read -r label bin config rest; do
  case "$label" in ''|'#'*) continue ;; esac
  extras=()
  if [ -n "${rest:-}" ]; then IFS=$'\t' read -r -a extras <<<"$rest"; fi
  printf '%s\n' "$(date -u +%FT%TZ) cell $label start" >>"$ROOT/job.log"
  bash "$HERE/gap-cell.sh" "$label" "$bin" "$config" "$ROOT/$label" "${extras[@]}"
  rc=$?
  printf '%s\n' "$(date -u +%FT%TZ) cell $label exit $rc" >>"$ROOT/job.log"
  if [ -e "$ROOT/$label/XTX_DROP" ]; then cp -- "$ROOT/$label/XTX_DROP" "$ROOT/XTX_DROP"; rc=3; break; fi
  [ "$rc" -eq 0 ] || break
done <"$PLAN"
# Post-job sweep: any process still carrying a cell HIPFIRE_HOME under this root is a failure.
left=$(gap_survivors "" "$ROOT/")
if [ -n "$left" ]; then
  gap_describe $left >"$ROOT/post-job-survivors.txt"
  printf '%s\n' "$(date -u +%FT%TZ) POST_JOB_SURVIVORS: $(echo $left)" >>"$ROOT/job.log"
  [ "$rc" -eq 0 ] && rc=4
else
  : >"$ROOT/post-job-survivors.txt"
fi
printf '%s\n' "$(date -u +%FT%TZ) job end rc=$rc" >>"$ROOT/job.log"
# Receipt: sha256 of every receipt file (kernel/home caches excluded), then read-only.
(cd "$ROOT" && find . -type f ! -path './*/kcache/*' ! -path './*/home/*' ! -name MANIFEST.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >MANIFEST.sha256) || { [ "$rc" -eq 0 ] && rc=5; }
