#!/usr/bin/env bash
# ONE fresh `hipfire bench` process (H2 argv) inside an already-admitted fabric hold on hipx/xtx.
#   gap-cell.sh LABEL BIN_DIR CONFIG_TOML OUT_DIR GAP_PINS=/abs/pins.json GAP_ARM=A|B
# Adapted for the ABBAAB qualification. Differences, all fail-closed:
#  * no free-form env: the cell env is derived from pins.json (gap_plan.py cell-env)
#    (combination `place_iu4`: HIPFIRE_STATE_QUANT=q8 HIPFIRE_KV_MODE=q8 D2 off QSA_WMMA_GATHER=0
#    for both arms; arm B adds ONLY placement + IU4). No Q8 trunk WMMA lever on this runtime; trace dropped.
#  * full stage identity before launch (symlink/readlink, canonical target, ino/size/mtime_ns, sha256+md5, kernel
#    pack tree hash), stat-only recheck after; model identity = cached stat only (never read/hashed).
#  * every fd > 2 is closed in the model process group before exec; verified on /proc/PID/fd.
#  * model group = own session/PGID; on exit TERM -> drain -> KILL of that group, then any surviving process the
#    cell owns (group/session/HIPFIRE_HOME tag, exact pids only) is a hard failure (rc 4).
#  * device watchdog: vendor/device read every 2 s; disappearance => XTX_DROP marker, group killed, rc 3.
# Env: every HIPFIRE_/HIP_/ROCR_/ROC_/HSA_/GPU_/AMD_ variable inherited is removed except FABRIC_* and
# HIPFIRE_FABRIC_ADMISSION; then exactly the pinned set + the per-cell paths are applied.
# Exit: 0 ok, 2 refused/preflight, 3 XTX drop, 4 leftover process, 6 stage changed, 7 card did not drain, 8 lifecycle.
set -u
set +m
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=gap-lib.sh
. "$HERE/gap-lib.sh"
[ "$#" -ge 4 ] || { echo "usage: gap-cell.sh LABEL BIN_DIR CONFIG OUT GAP_PINS=... GAP_ARM=A|B" >&2; exit 2; }
LABEL=$1 BIN_DIR=$2 CONFIG=$3 OUT=$4
shift 4
MODEL=/home/kaden/pm-wave/fngptq/models/qwen3.8-flash-next-gptq3.mq4
UUID=GPU-43390a851e296ee5
BDF=0000:67:00.0
CAP_SECS=${GAP_CELL_CAP_SECS:-600}
PLAN=$HERE/gap_plan.py
PINS= ARM=
for kv in "$@"; do
  case "$kv" in
    GAP_PINS=/*) PINS=${kv#*=} ;;
    GAP_ARM=[AB]) ARM=${kv#*=} ;;
    *) echo "extra argument not allowed: $kv" >&2; exit 2 ;;
  esac
done
case "$LABEL" in [AB][123]) ;; *) echo "bad label $LABEL (A1..A3,B1..B3)" >&2; exit 2 ;; esac
[ "${LABEL:0:1}" = "$ARM" ] || { echo "label $LABEL does not match GAP_ARM=$ARM" >&2; exit 2; }
[ -f "$PINS" ] || { echo "missing GAP_PINS $PINS" >&2; exit 2; }
case "$OUT" in /*) ;; *) echo "OUT must be absolute" >&2; exit 2 ;; esac
case "$BIN_DIR" in /*) ;; *) echo "BIN_DIR must be absolute" >&2; exit 2 ;; esac
[ -e "$OUT" ] && { echo "refusing existing $OUT" >&2; exit 2; }
for f in "$BIN_DIR/hipfire" "$BIN_DIR/daemon" "$CONFIG" "$MODEL" "$MODEL.ple-q4"; do
  [ -e "$f" ] || { echo "missing $f" >&2; exit 2; }
done
[ -d "$BIN_DIR/kernels/compiled/gfx1100" ] || { echo "missing pack under $BIN_DIR" >&2; exit 2; }
mkdir -p "$OUT" "$OUT/home/.hipfire/bin" "$OUT/kcache" "$OUT/lockdir"

CPID= WDOG=
cleanup() {
  local rc=$? left p _
  trap - EXIT
  trap '' TERM INT HUP
  if [ -n "$WDOG" ]; then
    : >"$OUT/.cli-done"
    kill -- "$WDOG" 2>/dev/null || true
    wait "$WDOG" 2>/dev/null || true
  fi
  if [ -n "$CPID" ]; then
    kill -TERM -- "-$CPID" 2>/dev/null || true
    for _ in $(seq 150); do [ -n "$(gap_live $(pgrep -g "$CPID" 2>/dev/null))" ] || break; sleep 0.1; done
    kill -KILL -- "-$CPID" 2>/dev/null || true
    wait "$CPID" 2>/dev/null || true
  fi
  # Exact-pid sweep of anything this cell still owns (group, session, HIPFIRE_HOME tag): TERM, then KILL.
  left=$(gap_survivors "$CPID" "$OUT/home/.hipfire")
  if [ -n "$left" ]; then
    gap_describe $left >"$OUT/leftover-term.txt"
    # shellcheck disable=SC2086
    kill -TERM $left 2>/dev/null || true
    sleep 2
    # shellcheck disable=SC2086
    kill -KILL $(gap_live $left) 2>/dev/null || true
    sleep 1
  fi
  left=$(gap_survivors "$CPID" "$OUT/home/.hipfire")
  if [ -n "$left" ]; then gap_describe $left >"$OUT/leftover-after.txt"; [ "$rc" -eq 0 ] && rc=4
  else : >"$OUT/leftover-after.txt"; fi
  if ! gap_vendor_ok "$BDF"; then
    echo "XTX_DROP after cell: $BDF no longer reads 0x1002:0x744c" | tee "$OUT/XTX_DROP" >&2; rc=3
  fi
  printf '{"end_unix":%s,"vram_used_after":%s,"mem_available_kib_after":%s}\n' "$(date +%s.%N)" \
    "$(cat /sys/bus/pci/devices/$BDF/mem_info_vram_used 2>/dev/null || echo -1)" \
    "$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)" >"$OUT/postflight.json"
  printf '%s\n' "$rc" >"$OUT/exit"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

cp -- "$CONFIG" "$OUT/home/.hipfire/config.toml"
ln -s "$BIN_DIR/daemon" "$OUT/home/.hipfire/bin/daemon"
# HOME fallback for the kernel pack when the daemon is a symlink into an artifact tree without a pack beside it.
ln -s "$BIN_DIR/kernels" "$OUT/home/.hipfire/bin/kernels"
# Device + host preflight.
vendor=$(cat /sys/bus/pci/devices/$BDF/vendor 2>/dev/null || echo missing)
device=$(cat /sys/bus/pci/devices/$BDF/device 2>/dev/null || echo missing)
used=$(cat /sys/bus/pci/devices/$BDF/mem_info_vram_used 2>/dev/null || echo -1)
avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
{
  printf '{"label":"%s","arm":"%s","host":"%s","vendor":"%s","device":"%s","vram_used_before":%s,"mem_available_kib_before":%s,' \
    "$LABEL" "$ARM" "$(hostname)" "$vendor" "$device" "$used" "$avail_kb"
  printf '"fabric_job":"%s","fabric_cards":"%s","boot_id":"%s","pins":"%s","pins_sha256":"%s","start_unix":%s}\n' \
    "${FABRIC_JOB:-}" "${FABRIC_CARDS:-}" "$(cat /proc/sys/kernel/random/boot_id)" "$PINS" \
    "$(sha256sum <"$PINS" | cut -d' ' -f1)" "$(date +%s.%N)"
} >"$OUT/preflight.json"
if [ "$vendor" != 0x1002 ] || [ "$device" != 0x744c ]; then
  echo "XTX_DROP: device $BDF reads vendor=$vendor device=$device" | tee "$OUT/XTX_DROP" >&2; exit 3
fi
if [ "$used" -ge $((2 * 1024 * 1024 * 1024)) ]; then echo "card not idle: vram used $used" >&2; exit 2; fi
if [ "$avail_kb" -lt $((80 * 1024 * 1024)) ]; then echo "MemAvailable ${avail_kb} KiB < 80 GiB" >&2; exit 2; fi

# Pinned identities: config, model (cached stat only), stage (symlinks, stat, sha256, md5, kernel pack tree hash).
python3 "$PLAN" check-config "$PINS" "$CONFIG" || exit 2
python3 "$PLAN" check-model "$PINS" >"$OUT/model-stat.txt" || exit 2
python3 "$PLAN" verify-stage "$BIN_DIR" "$PINS" "$OUT/stage-identity.json" --binaries-out "$OUT/binaries.txt" || exit 2
{
  for e in "$BIN_DIR"/* "$BIN_DIR/kernels/compiled/gfx1100"; do
    printf '%s\n  ls: %s\n  readlink: %s\n  canonical: %s\n  stat -L: %s\n' "$e" \
      "$(ls -ld --time-style=full-iso -- "$e")" "$(readlink -- "$e" || echo '-')" "$(readlink -f -- "$e")" \
      "$(stat -L -c 'ino=%i size=%s mtime=%.9Y' -- "$e")"
  done
} >"$OUT/stage-links.txt"

# Cell env from the pins (validated against the harness definition), then the sanitised inherited env.
mapfile -t PINENV < <(python3 "$PLAN" cell-env "$PINS" "$ARM") || exit 2
[ "${#PINENV[@]}" -ge 3 ] || { echo "empty pinned env" >&2; exit 2; }
ENVV=()
: >"$OUT/env-stripped.txt"
while IFS= read -r -d '' kv; do
  k=${kv%%=*}
  case "$k" in
    FABRIC_*|HIPFIRE_FABRIC_ADMISSION) ENVV+=("$kv") ;;
    HIPFIRE_*|HIP_*|ROCR_*|ROC_*|HSA_*|GPU_*|AMD_*) printf '%s\n' "$k" >>"$OUT/env-stripped.txt" ;;
    HOME) ;;
    *) ENVV+=("$kv") ;;
  esac
done < <(env -0)
ENVV+=("HOME=$OUT/home" "HIPFIRE_HOME=$OUT/home/.hipfire" "HIPFIRE_LOCK_DIR=$OUT/lockdir"
  "HIPFIRE_KERNEL_CACHE=$OUT/kcache" "HIPFIRE_DEVICES=$UUID" "ROCR_VISIBLE_DEVICES=$UUID"
  "HIP_VISIBLE_DEVICES=0" "HIPFIRE_NO_REGISTRY_FETCH=1" "${PINENV[@]}")
printf '%s\n' "${ENVV[@]}" | grep -E '^(HIPFIRE_|HIP_|ROCR_|HOME=)' | sort >"$OUT/env-applied.txt"
# H2 argv, in the spelling the earlier gap2000 cells used.
ARGV=("$BIN_DIR/hipfire" bench "$MODEL" --matrix --pp 16384 --tg 1 --ctx 1 --runs 3 --warmups 1 --max-seq 16416 --json)
printf '%q ' "${ARGV[@]}" >"$OUT/command.txt"; printf '\n' >>"$OUT/command.txt"

# Model group: own session/PGID, every fd > 2 closed before exec, cwd = the cell dir.
LAUNCH='for f in /proc/$$/fd/*; do n=${f##*/}; [ "$n" -gt 2 ] 2>/dev/null && eval "exec $n>&-"; done; cd "$1" || exit 97; shift; exec "$@"'
setsid bash -c "$LAUNCH" gap-model-launch "$OUT" env -i "${ENVV[@]}" \
  timeout --signal=TERM --kill-after=20 "$CAP_SECS" "${ARGV[@]}" \
  >"$OUT/stdout.json" 2>"$OUT/stderr.log" </dev/null &
CPID=$!
printf '%s\n' "$CPID" >"$OUT/cli.pgid"
# Lifecycle check: the launcher must be its own session+PGID leader and, once exec'd into `timeout`, hold no fd > 2.
RC=
for _ in $(seq 50); do
  [ "$(cat "/proc/$CPID/comm" 2>/dev/null)" = timeout ] && break
  [ -d "/proc/$CPID" ] || break
  sleep 0.1
done
if [ -d "/proc/$CPID" ]; then
  ls "/proc/$CPID/fd" >"$OUT/fds-model.txt" 2>&1
  ps -o pid=,pgid=,sid=,comm= -p "$CPID" >"$OUT/model-session.txt" 2>&1
  read -r _p PGID_ SID_ _c <"$OUT/model-session.txt" || true
  if [ "${PGID_:-}" != "$CPID" ] || [ "${SID_:-}" != "$CPID" ] || [ "$(cat "/proc/$CPID/comm" 2>/dev/null)" != timeout ] \
    || awk '$1 > 2 { bad = 1 } END { exit !bad }' "$OUT/fds-model.txt"; then
    echo "LIFECYCLE: model launcher not (own session/PGID, exec'd, fds<=2): $(tr '\n' ' ' <"$OUT/model-session.txt") fds: $(tr '\n' ' ' <"$OUT/fds-model.txt")" >&2
    RC=8
  fi
fi
if [ -z "$RC" ]; then
  # Device watchdog: vendor/device must stay readable and unchanged for the life of the model process.
  (
    while [ ! -e "$OUT/.cli-done" ]; do
      if ! gap_vendor_ok "$BDF"; then
        echo "XTX_DROP during cell: $BDF no longer reads 0x1002:0x744c" >"$OUT/XTX_DROP"
        kill -TERM -- "-$CPID" 2>/dev/null; sleep 5; kill -KILL -- "-$CPID" 2>/dev/null
        exit 0
      fi
      sleep 2
    done
  ) &
  WDOG=$!
  wait "$CPID"
  RC=$?
  : >"$OUT/.cli-done"
  wait "$WDOG" 2>/dev/null
  WDOG=
fi
[ -e "$OUT/XTX_DROP" ] && RC=3
# The CLI exited: nothing it owns may survive (group, session, HIPFIRE_HOME tag). Cleanup then reaps and re-checks.
left=$(gap_survivors "$CPID" "$OUT/home/.hipfire")
if [ -n "$left" ]; then
  gap_describe $left >"$OUT/leftover.txt"
  echo "LIFECYCLE_LEFTOVER: cell-owned process(es) still live after CLI exit: $(echo $left)" >&2
  [ "$RC" -eq 0 ] && RC=4
fi
# Stage + model must be unchanged while the cell ran (stat-only recheck; no re-hash, no model read).
if ! python3 "$PLAN" recheck-stage "$BIN_DIR" "$PINS" >"$OUT/stage-recheck.txt" 2>&1; then
  echo "stage changed during cell: $(cat "$OUT/stage-recheck.txt")" >&2
  [ "$RC" -eq 0 ] && RC=6
fi
python3 "$PLAN" check-model "$PINS" >"$OUT/model-stat-after.txt" 2>&1 || { [ "$RC" -eq 0 ] && RC=6; }
# Settle: the card must drain back below 2 GiB before the next cell.
drained=0
for _ in $(seq 90); do
  u=$(cat /sys/bus/pci/devices/$BDF/mem_info_vram_used 2>/dev/null || echo -1)
  if [ "$u" -ge 0 ] && [ "$u" -lt $((2 * 1024 * 1024 * 1024)) ]; then drained=1; break; fi
  sleep 1
done
if [ "$drained" != 1 ]; then echo "card did not drain: vram used $u" >&2; [ "$RC" -eq 0 ] && RC=7; fi
exit "$RC"
