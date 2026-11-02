#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# Post-land GPU smoke orchestrator (docs/post-land-smoke.md).
#
#   post-land-smoke.sh --env FILE --land SHA [--prev SHA] [--out DIR] [--dry-run] [--skip-host H]...
#   post-land-smoke.sh --env FILE --watch SECONDS
#
# --land: smoke one landed commit. Extracts that commit's own runner and plan, verifies the
# queue-built binaries, selects the lanes the diff requires, runs each lane as a fabric hold
# (local lanes directly, remote lanes through a generated script run over ssh), merges the lane
# results into VERDICT.json and applies the pin policy.
# Exit: 0 PASS, 1 FAIL, 3 CHANGED, 4 INCOMPLETE, 2 usage/infrastructure.
#
# Process safety: the only processes ever signalled are the process groups and ssh PIDs this script
# recorded at launch. Nothing is ever killed by name or pattern; fabric jobs are cancelled by exact
# id (fabric-run.sh --cancel <ID>) by the owner.
#
# Env file keys (bash KEY=VALUE, sourced): REPO BINS_ROOT PINS OUT_ROOT FABRIC_RUN OWNER
# [PRIORITY=P1] [WAIT_SECS=3600] LOCAL_HOST LOCAL_MODEL_ENV LOCAL_HOME_MODELS, and per remote host
# label H (upper-cased, non-alphanumerics -> _): REMOTE_H_SSH REMOTE_H_ROOT REMOTE_H_FABRIC_RUN
# REMOTE_H_MODEL_ENV REMOTE_H_HOME_MODELS. Optional: HOLD_EXTRA_ARGS (words appended to every
# `hold` command, e.g. a --watch-cmd for CPU tests). Env overrides: SSH_BIN, RSYNC_BIN, PYTHON,
# WATCH_ONCE=1 (one --watch iteration, for tests).
set -euo pipefail

EXIT_USAGE=2

usage() {
    cat >&2 <<'EOF'
usage: post-land-smoke.sh --env FILE --land SHA [--prev SHA] [--out DIR] [--dry-run] [--skip-host H]...
       post-land-smoke.sh --env FILE --watch SECONDS
EOF
    exit "$EXIT_USAGE"
}

die_usage() {
    echo "post-land-smoke: $*" >&2
    exit "$EXIT_USAGE"
}

log() { echo "[post-land-smoke $(date -u +%H:%M:%SZ)] $*" >&2; }

# ---------------------------------------------------------------------------------------------
# process-group helpers. These two are also emitted verbatim into generated remote launch scripts
# (declare -f), so they must stay self-contained.
# ---------------------------------------------------------------------------------------------

# start_group PIDFILE LOGFILE CMD...: run CMD in its own session/process group with stdin from
# /dev/null and stdout+stderr to LOGFILE. Sets GROUP_WAITER (pid to `wait` on), GROUP_PID and
# GROUP_PGID (recorded in PIDFILE as "PID PGID"). The inner shell refuses to exec CMD unless it is
# its own group leader, so GROUP_PGID is never this script's own group.
start_group() {
    local pidfile=$1 logfile=$2
    shift 2
    rm -f "$pidfile" "$pidfile.tmp"
    setsid -w bash -c '
        pg=$(ps -o pgid= -p "$$" | tr -d " ")
        if [ "$pg" != "$$" ]; then echo "start_group: not a process-group leader" >&2; exit 99; fi
        echo "$$ $pg" > "$1.tmp" && mv "$1.tmp" "$1"
        shift
        exec "$@"' _ "$pidfile" "$@" >"$logfile" 2>&1 </dev/null &
    GROUP_WAITER=$!
    local i=0
    while [ ! -s "$pidfile" ]; do
        kill -0 "$GROUP_WAITER" 2>/dev/null || break
        i=$((i + 1))
        [ "$i" -le 400 ] || break
        sleep 0.05
    done
    GROUP_PID="" GROUP_PGID=""
    if [ -s "$pidfile" ]; then
        read -r GROUP_PID GROUP_PGID <"$pidfile"
        return 0
    fi
    return 1
}

# signal_group PGID SIG: signal a recorded process group, never this script's own group.
signal_group() {
    local pgid=$1 sig=$2 own
    [[ $pgid =~ ^[0-9]+$ ]] || return 0
    [ "$pgid" -gt 1 ] || return 0
    own=$(ps -o pgid= -p "$$" | tr -d ' ')
    [ "$pgid" != "$own" ] || return 0
    kill "-$sig" -- "-$pgid" 2>/dev/null || true
}

# wait_pid PID: wait for a child we started, surviving trapped signals; sets WAIT_RC to its exit
# status. (Not usable in $(...): only the shell that started the child can wait for it.)
wait_pid() {
    local pid=$1
    while :; do
        if wait "$pid"; then WAIT_RC=0; else WAIT_RC=$?; fi
        if kill -0 "$pid" 2>/dev/null; then continue; fi
        break
    done
}

# read_kv_file FILE ARRAY: KEY=VALUE lines (blank and #-comment lines ignored) -> ARRAY elements.
read_kv_file() {
    local file=$1 line
    local -n _out=$2
    _out=()
    [ -r "$file" ] || die_usage "model env file unreadable: $file"
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
        '' | '#'*) continue ;;
        [A-Za-z_]*=*) _out+=("$line") ;;
        *) die_usage "bad line in $file (want KEY=VALUE): $line" ;;
        esac
    done <"$file"
}

host_var() { # host suffix -> value of REMOTE_<HOST>_<suffix>
    local h=${1^^}
    h=${h//[^A-Z0-9]/_}
    local v="REMOTE_${h}_$2"
    printf '%s' "${!v:-}"
}

# ---------------------------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------------------------

ENV_FILE="" LAND="" PREV="" OUT_DIR="" DRY_RUN=0 WATCH_SECS="" SKIP_HOSTS=()
while [ $# -gt 0 ]; do
    case "$1" in
    --env) [ $# -ge 2 ] || usage; ENV_FILE=$2; shift 2 ;;
    --land) [ $# -ge 2 ] || usage; LAND=$2; shift 2 ;;
    --prev) [ $# -ge 2 ] || usage; PREV=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || usage; OUT_DIR=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-host) [ $# -ge 2 ] || usage; SKIP_HOSTS+=("$2"); shift 2 ;;
    --watch) [ $# -ge 2 ] || usage; WATCH_SECS=$2; shift 2 ;;
    -h | --help) usage ;;
    *) echo "post-land-smoke: unknown argument: $1" >&2; usage ;;
    esac
done
[ -n "$ENV_FILE" ] || usage
[ -r "$ENV_FILE" ] || die_usage "env file unreadable: $ENV_FILE"
if [ -n "$WATCH_SECS" ]; then
    [ -z "$LAND" ] || usage
    [[ $WATCH_SECS =~ ^[0-9]+$ ]] || die_usage "--watch SECONDS must be an integer"
else
    [ -n "$LAND" ] || usage
fi

REPO="" BINS_ROOT="" PINS="" OUT_ROOT="" FABRIC_RUN="" OWNER="" PRIORITY="" WAIT_SECS="" LOCAL_HOST=""
LOCAL_MODEL_ENV="" LOCAL_HOME_MODELS="" HOLD_EXTRA_ARGS=""
# shellcheck disable=SC1090
source "$ENV_FILE"
PRIORITY=${PRIORITY:-P1}
WAIT_SECS=${WAIT_SECS:-3600}
SSH_BIN=${SSH_BIN:-ssh}
RSYNC_BIN=${RSYNC_BIN:-rsync}
PYTHON=${PYTHON:-python3}
read -r -a SSH_CMD <<<"$SSH_BIN"
read -r -a RSYNC_CMD <<<"$RSYNC_BIN"
read -r -a HOLD_EXTRA <<<"$HOLD_EXTRA_ARGS"

for k in REPO BINS_ROOT PINS OWNER; do
    [ -n "${!k}" ] || die_usage "env file must set $k"
done
[ -n "$OUT_ROOT" ] || [ -n "$OUT_DIR" ] || die_usage "env file must set OUT_ROOT (or pass --out)"
[ -d "$REPO" ] || die_usage "REPO is not a directory: $REPO"
[[ $WAIT_SECS =~ ^[0-9]+$ ]] || die_usage "WAIT_SECS must be an integer"

SELF=$(readlink -f "${BASH_SOURCE[0]}")

# ---------------------------------------------------------------------------------------------
# --watch mode
# ---------------------------------------------------------------------------------------------

if [ -n "$WATCH_SECS" ]; then
    [ -n "$OUT_ROOT" ] || die_usage "--watch needs OUT_ROOT"
    mkdir -p "$OUT_ROOT"
    exec 8>"$OUT_ROOT/watch.lock"
    flock -n 8 || die_usage "another watcher holds $OUT_ROOT/watch.lock"
    WATCH_CHILD=""
    trap '[ -z "$WATCH_CHILD" ] || kill -TERM "$WATCH_CHILD" 2>/dev/null || true; exit 143' INT TERM HUP
    while :; do
        if ! git -C "$REPO" fetch -q origin beta; then log "git fetch origin beta failed; retrying in ${WATCH_SECS}s"; fi
        sha=$(git -C "$REPO" rev-parse --verify "origin/beta^{commit}" 2>/dev/null || true)
        last=$(cat "$OUT_ROOT/last-smoked" 2>/dev/null || true)
        if [ -n "$sha" ] && [ "$sha" != "$last" ] && [ -e "$BINS_ROOT/$sha/LANDED.json" ]; then
            log "origin/beta ${sha:0:12} (last smoked ${last:0:12}): running --land"
            args=(--env "$ENV_FILE" --land "$sha")
            [ -z "$last" ] || args+=(--prev "$last")
            bash "$SELF" "${args[@]}" 8>&- &
            WATCH_CHILD=$!
            wait_pid "$WATCH_CHILD"
            WATCH_CHILD=""
            log "smoke of ${sha:0:12} exited $WAIT_RC"
            printf '%s\n' "$sha" >"$OUT_ROOT/last-smoked.tmp" && mv "$OUT_ROOT/last-smoked.tmp" "$OUT_ROOT/last-smoked"
        fi
        [ "${WATCH_ONCE:-0}" != "1" ] || exit 0
        sleep "$WATCH_SECS" &
        WATCH_CHILD=$!
        wait "$WATCH_CHILD" || true
        WATCH_CHILD=""
    done
fi

# ---------------------------------------------------------------------------------------------
# --land mode
# ---------------------------------------------------------------------------------------------

# 1. resolve shas
FULL=$(git -C "$REPO" rev-parse --verify "$LAND^{commit}" 2>/dev/null) || die_usage "cannot resolve --land $LAND in $REPO"
if [ -z "$PREV" ]; then
    pinned=""
    if [ -r "$PINS" ]; then
        pinned=$("$PYTHON" - "$PINS" <<'PY'
import json, sys
try:
    hist = json.load(open(sys.argv[1])).get("history") or []
except (OSError, ValueError):
    hist = []
print(next((h["sha"] for h in reversed(hist) if h.get("sha")), ""))
PY
)
    fi
    if [ -n "$pinned" ]; then
        PREV=$pinned
    else
        PREV=$(git -C "$REPO" rev-parse --verify "$FULL^" 2>/dev/null) || die_usage "$FULL has no parent; pass --prev"
    fi
fi
PREV_FULL=$(git -C "$REPO" rev-parse --verify "$PREV^{commit}" 2>/dev/null) || die_usage "cannot resolve --prev $PREV in $REPO (fetch it or pass --prev)"
PREV=$PREV_FULL

# 2. run directory and the land's own runner/plan
OUT_BASE=${OUT_DIR:-$OUT_ROOT}
mkdir -p "$OUT_BASE"
OUT_BASE=$(cd "$OUT_BASE" && pwd)
RUN_BASE="${FULL}-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID=$RUN_BASE
n=1
until mkdir "$OUT_BASE/$RUN_ID" 2>/dev/null; do # a rerun within the same second gets a -N suffix
    n=$((n + 1))
    [ "$n" -le 99 ] || die_usage "cannot create a run dir under $OUT_BASE"
    RUN_ID="$RUN_BASE-$n"
done
RUN="$OUT_BASE/$RUN_ID"
mkdir -p "$RUN/lanes" "$RUN/bin"
log "land $FULL prev $PREV run $RUN"
git -C "$REPO" show "$FULL:scripts/post_land_smoke.py" >"$RUN/post_land_smoke.py" 2>"$RUN/git-show.err" ||
    die_usage "$FULL has no scripts/post_land_smoke.py: $(cat "$RUN/git-show.err")"
git -C "$REPO" show "$FULL:scripts/post-land-smoke/plan.json" >"$RUN/plan.json" 2>"$RUN/git-show.err" ||
    die_usage "$FULL has no scripts/post-land-smoke/plan.json: $(cat "$RUN/git-show.err")"
chmod +x "$RUN/post_land_smoke.py"

# 3. binaries: verify, hardlink into the run dir, verify again
BIN_SRC="$BINS_ROOT/$FULL"
[ -d "$BIN_SRC" ] || die_usage "no binaries at $BIN_SRC"
[ -r "$BIN_SRC/SHA256SUMS" ] || die_usage "no SHA256SUMS in $BIN_SRC"
for f in daemon hipfire; do [ -f "$BIN_SRC/$f" ] || die_usage "missing $BIN_SRC/$f"; done
(cd "$BIN_SRC" && sha256sum -c --quiet SHA256SUMS >"$RUN/bins-verify.log" 2>&1) ||
    die_usage "binary checksum mismatch in $BIN_SRC (see $RUN/bins-verify.log)"
for f in daemon hipfire; do
    ln "$BIN_SRC/$f" "$RUN/bin/$f" 2>"$RUN/ln.err" || die_usage "cannot hardlink $BIN_SRC/$f into $RUN/bin (cross-device?): $(cat "$RUN/ln.err")"
done
grep -E '[[:space:]]\*?(daemon|hipfire)$' "$BIN_SRC/SHA256SUMS" >"$RUN/bin/SHA256SUMS" || true
[ "$(wc -l <"$RUN/bin/SHA256SUMS")" -eq 2 ] || die_usage "SHA256SUMS in $BIN_SRC must list both daemon and hipfire"
(cd "$RUN/bin" && sha256sum -c --quiet SHA256SUMS >>"$RUN/bins-verify.log" 2>&1) ||
    die_usage "hardlinked binary checksum mismatch in $RUN/bin"

# 4. what changed
git -C "$REPO" diff --name-only "$PREV" "$FULL" >"$RUN/changed.txt"
git -C "$REPO" diff -U0 --no-color "$PREV" "$FULL" >"$RUN/land.diff"

# 5. lanes and plan
LOCAL_ENV_ARGS=()
if [ -n "$LOCAL_MODEL_ENV" ]; then read_kv_file "$LOCAL_MODEL_ENV" LOCAL_ENV_ARGS; fi
runner() { env "${LOCAL_ENV_ARGS[@]}" "$PYTHON" "$RUN/post_land_smoke.py" "$@"; }
sel_args=(--plan "$RUN/plan.json" --changed-paths "$RUN/changed.txt" --diff "$RUN/land.diff")
runner plan "${sel_args[@]}" >"$RUN/plan.txt" || die_usage "runner plan failed"
runner lanes "${sel_args[@]}" >"$RUN/lanes.json" || die_usage "runner lanes failed"

mapfile -t LANE_LINES < <("$PYTHON" - "$RUN/lanes.json" <<'PY'
import json, shlex, sys
for lane in json.load(open(sys.argv[1])):
    print("\t".join([lane["name"], lane["host"], shlex.join(lane["fabric_args"]), " ".join(lane["loads"])]))
PY
)
[ "${#LANE_LINES[@]}" -gt 0 ] || die_usage "no lane has a selected load"

LANE_NAMES=() LANE_HOSTS=() LANE_FA=() LANE_LOADS=() LANE_KIND=()
config_errors=()
is_skipped_host() { local h; for h in "${SKIP_HOSTS[@]}"; do [ "$h" = "$1" ] && return 0; done; return 1; }
for line in "${LANE_LINES[@]}"; do
    IFS=$'\t' read -r name host fa loads <<<"$line"
    LANE_NAMES+=("$name") LANE_HOSTS+=("$host") LANE_FA+=("$fa") LANE_LOADS+=("$loads")
    if is_skipped_host "$host"; then
        LANE_KIND+=(skipped)
    elif [ "$host" = "$LOCAL_HOST" ]; then
        LANE_KIND+=(local)
        for k in FABRIC_RUN LOCAL_MODEL_ENV LOCAL_HOME_MODELS; do
            [ -n "${!k}" ] || config_errors+=("lane $name (local host $host): env file must set $k")
        done
    elif [ -n "$(host_var "$host" SSH)" ]; then
        LANE_KIND+=(remote)
        root=$(host_var "$host" ROOT)
        for k in ROOT FABRIC_RUN MODEL_ENV HOME_MODELS; do
            [ -n "$(host_var "$host" "$k")" ] || config_errors+=("lane $name (remote host $host): env file must set REMOTE_${host^^}_$k")
        done
        [[ $root =~ ^/[A-Za-z0-9._/+-]+$ ]] || config_errors+=("lane $name: REMOTE_${host^^}_ROOT must be a plain absolute path, got '$root'")
    else
        LANE_KIND+=(noconfig)
        config_errors+=("lane $name: host '$host' has no configuration (LOCAL_HOST=$LOCAL_HOST, no REMOTE_${host^^}_SSH); fix the env file or pass --skip-host $host")
    fi
done
if [ "${#config_errors[@]}" -gt 0 ]; then
    printf 'post-land-smoke: %s\n' "${config_errors[@]}" >&2
    exit "$EXIT_USAGE"
fi

# build_cmd KIND LANE FABRIC_ARGS_STR FABRIC_RUN BASE LANE_OUT HOME_MODELS: sets LANE_PRE (fabric
# invocation up to and including `env`) and LANE_POST (the runner `hold` command); the model vars
# go between them.
build_cmd() {
    local lane=$2 fa=$3 fabric=$4 base=$5 laneout=$6 homemodels=$7
    local -a words
    eval "words=($fa)"
    LANE_PRE=("$fabric" "${words[@]}" --priority "$PRIORITY" --owner "$OWNER" --wait-secs "$WAIT_SECS" -- env)
    LANE_POST=(python3 "$base/post_land_smoke.py" hold --plan "$base/plan.json" --lane "$lane" --daemon "$base/bin/daemon"
        --out "$laneout" --changed-paths "$base/changed.txt" --diff "$base/land.diff" --home-models "$homemodels" "${HOLD_EXTRA[@]}")
}

if [ "$DRY_RUN" = 1 ]; then
    echo "== lanes.json =="; cat "$RUN/lanes.json"
    echo "== plan =="; cat "$RUN/plan.txt"
    echo "== fabric commands =="
    for i in "${!LANE_NAMES[@]}"; do
        name=${LANE_NAMES[$i]} host=${LANE_HOSTS[$i]} kind=${LANE_KIND[$i]}
        case "$kind" in
        local)
            build_cmd local "$name" "${LANE_FA[$i]}" "$FABRIC_RUN" "$RUN" "$RUN/lanes/$name" "$LOCAL_HOME_MODELS"
            echo "[local $host] lane $name:"
            printf '  setsid '; printf '%q ' "${LANE_PRE[@]}" "${LOCAL_ENV_ARGS[@]}" "${LANE_POST[@]}"; echo ;;
        remote)
            remote="$(host_var "$host" ROOT)/$RUN_ID"
            build_cmd remote "$name" "${LANE_FA[$i]}" "$(host_var "$host" FABRIC_RUN)" "$remote" "$remote/lanes/$name" "$(host_var "$host" HOME_MODELS)"
            echo "[remote $host via $(host_var "$host" SSH)] lane $name: ssh $remote/launch-$name.sh runs:"
            printf '  '; printf '%q ' "${LANE_PRE[@]}" "<model vars from $(host_var "$host" MODEL_ENV)>" "${LANE_POST[@]}"; echo ;;
        skipped) echo "[skipped host $host] lane $name: not submitted (its cells are FAIL \"not run\")" ;;
        esac
    done
    echo "dry run: nothing submitted; run dir $RUN"
    exit 0
fi

# ---------------------------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------------------------

LIVE_GROUPS=()   # recorded local process groups (PGID)
LIVE_SSH=()      # recorded local ssh PIDs
INTERRUPTED=0
FINISHED=0

term_recorded() {
    local g p
    for g in "${LIVE_GROUPS[@]}"; do signal_group "$g" TERM; done
    for p in "${LIVE_SSH[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
}
on_signal() {
    INTERRUPTED=1
    log "caught $1: forwarding SIGTERM to recorded lane groups and ssh pids only (cancel remote/queued fabric jobs by exact id)"
    term_recorded
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP
trap '[ "$FINISHED" = 1 ] || term_recorded' EXIT

# remote preparation: one dir per host, verified on the remote before anything is submitted
prepare_remote() {
    local host=$1 target root remote dir
    target=$(host_var "$host" SSH)
    root=$(host_var "$host" ROOT)
    remote="$root/$RUN_ID"
    dir="$RUN/remote-$host"
    mkdir -p "$dir/bin" "$dir/lanes"
    cp "$RUN/post_land_smoke.py" "$RUN/plan.json" "$RUN/changed.txt" "$RUN/land.diff" "$dir/"
    ln "$RUN/bin/daemon" "$dir/bin/daemon" || die_usage "cannot hardlink the daemon into $dir/bin"
    cat >"$dir/verify.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
sha256sum -c MANIFEST.sha256
EOF
    cat >"$dir/manifest.sh" <<'EOF'
#!/usr/bin/env bash
# manifest.sh LANE: sha256 of every file under lanes/LANE (paths relative to this dir) -> RESULTS-LANE.sha256
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
lane=$1
[ -d "lanes/$lane" ] || { : >"RESULTS-$lane.sha256"; exit 0; }
find "lanes/$lane" -type f -print0 | sort -z | xargs -0 -r sha256sum >"RESULTS-$lane.sha256.tmp"
mv "RESULTS-$lane.sha256.tmp" "RESULTS-$lane.sha256"
EOF
    local i
    for i in "${!LANE_NAMES[@]}"; do
        [ "${LANE_KIND[$i]}" = remote ] && [ "${LANE_HOSTS[$i]}" = "$host" ] || continue
        local name=${LANE_NAMES[$i]}
        build_cmd remote "$name" "${LANE_FA[$i]}" "$(host_var "$host" FABRIC_RUN)" "$remote" "$remote/lanes/$name" "$(host_var "$host" HOME_MODELS)"
        {
            echo '#!/usr/bin/env bash'
            echo "# generated by post-land-smoke.sh for land $FULL, lane $name"
            echo 'set -euo pipefail'
            echo 'cd "$(dirname "$(readlink -f "$0")")"'
            echo 'EXIT_USAGE=2'
            declare -f start_group signal_group wait_pid read_kv_file die_usage
            printf 'LANE=%q\n' "$name"
            printf 'PRE=('; printf '%q ' "${LANE_PRE[@]}"; echo ')'
            printf 'POST=('; printf '%q ' "${LANE_POST[@]}"; echo ')'
            printf 'MODEL_ENV_FILE=%q\n' "$(host_var "$host" MODEL_ENV)"
            cat <<'EOF'
MODEL_ARGS=()
read_kv_file "$MODEL_ENV_FILE" MODEL_ARGS
mkdir -p "lanes/$LANE"
GROUP_PGID=""
finish() {
    local rc=$?
    trap - EXIT
    [ ! -f "lanes/$LANE.fabric.log" ] || cat "lanes/$LANE.fabric.log"
    ./manifest.sh "$LANE" || true
    exit "$rc"
}
trap finish EXIT
trap 'signal_group "$GROUP_PGID" TERM' INT TERM HUP
start_group "lanes/$LANE.pid" "lanes/$LANE.fabric.log" "${PRE[@]}" "${MODEL_ARGS[@]}" "${POST[@]}" || { echo "launch: could not start lane $LANE" >&2; exit 2; }
wait_pid "$GROUP_WAITER"
exit "$WAIT_RC"
EOF
        } >"$dir/launch-$name.sh"
        chmod +x "$dir/launch-$name.sh"
    done
    chmod +x "$dir/verify.sh" "$dir/manifest.sh" "$dir/post_land_smoke.py"
    (cd "$dir" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum >MANIFEST.sha256)
    (cd "$dir" && sha256sum -c --quiet MANIFEST.sha256) || die_usage "local manifest of $dir does not verify"
    log "rsync $dir/ -> $target:$remote/"
    "${RSYNC_CMD[@]}" -a -e "$SSH_BIN" "$dir/" "$target:$remote/" >"$RUN/remote-$host.rsync.log" 2>&1 ||
        die_usage "rsync to $target:$remote failed (see $RUN/remote-$host.rsync.log)"
    "${SSH_CMD[@]}" "$target" "$remote/verify.sh" >"$RUN/remote-$host.verify.log" 2>&1 </dev/null ||
        die_usage "remote sha256 verification failed on $target (see $RUN/remote-$host.verify.log)"
}

prepared_hosts=" "
for i in "${!LANE_NAMES[@]}"; do
    if [ "${LANE_KIND[$i]}" = remote ] && [[ $prepared_hosts != *" ${LANE_HOSTS[$i]} "* ]]; then
        prepare_remote "${LANE_HOSTS[$i]}"
        prepared_hosts+="${LANE_HOSTS[$i]} "
    fi
done

[ "$INTERRUPTED" = 0 ] || die_usage "interrupted before any lane was submitted"
LANE_WAITER=() LANE_RC=() LANE_NOTE=()
for i in "${!LANE_NAMES[@]}"; do
    name=${LANE_NAMES[$i]} host=${LANE_HOSTS[$i]} kind=${LANE_KIND[$i]}
    LANE_WAITER[i]="" LANE_RC[i]="" LANE_NOTE[i]=""
    # A signal caught while an earlier lane was being launched: submit nothing more.
    if [ "$INTERRUPTED" = 1 ] && [ "$kind" != skipped ]; then
        LANE_NOTE[i]="not submitted (interrupted)"
        continue
    fi
    case "$kind" in
    local)
        build_cmd local "$name" "${LANE_FA[$i]}" "$FABRIC_RUN" "$RUN" "$RUN/lanes/$name" "$LOCAL_HOME_MODELS"
        log "submit local lane $name on $host: ${LANE_PRE[*]:1}"
        if start_group "$RUN/lanes/$name.pid" "$RUN/lanes/$name.fabric.log" "${LANE_PRE[@]}" "${LOCAL_ENV_ARGS[@]}" "${LANE_POST[@]}"; then
            LANE_WAITER[i]=$GROUP_WAITER
            LIVE_GROUPS+=("$GROUP_PGID")
            # The trap only signals groups already in LIVE_GROUPS; a signal caught while this lane
            # was starting must still reach it.
            [ "$INTERRUPTED" = 0 ] || signal_group "$GROUP_PGID" TERM
        else
            LANE_RC[i]=2 LANE_NOTE[i]="could not start"
        fi ;;
    remote)
        remote="$(host_var "$host" ROOT)/$RUN_ID"
        target=$(host_var "$host" SSH)
        log "submit remote lane $name on $host: $target $remote/launch-$name.sh"
        "${SSH_CMD[@]}" "$target" "$remote/launch-$name.sh" >"$RUN/lanes/$name.fabric.log" 2>&1 </dev/null &
        LANE_WAITER[i]=$!
        echo "$! ssh" >"$RUN/lanes/$name.pid"
        LIVE_SSH+=("$!")
        [ "$INTERRUPTED" = 0 ] || kill -TERM "${LANE_WAITER[$i]}" 2>/dev/null || true ;;
    skipped) LANE_NOTE[i]="host skipped (--skip-host)" ;;
    esac
done

# ---------------------------------------------------------------------------------------------
# wait, fetch, verify
# ---------------------------------------------------------------------------------------------

infra_errors=()
for i in "${!LANE_NAMES[@]}"; do
    [ -n "${LANE_WAITER[$i]}" ] || continue
    wait_pid "${LANE_WAITER[$i]}"
    LANE_RC[i]=$WAIT_RC
    log "lane ${LANE_NAMES[$i]} (${LANE_KIND[$i]}) exited ${LANE_RC[$i]}"
done
FINISHED=1

for i in "${!LANE_NAMES[@]}"; do
    [ "${LANE_KIND[$i]}" = remote ] || continue
    name=${LANE_NAMES[$i]} host=${LANE_HOSTS[$i]}
    target=$(host_var "$host" SSH)
    remote="$(host_var "$host" ROOT)/$RUN_ID"
    mkdir -p "$RUN/lanes/$name"
    # -a copies symlinks as links (each daemon HOME's .hipfire/models is one); the excludes keep a
    # model store or model file from ever being pulled onto this disk.
    if ! "${RSYNC_CMD[@]}" -a --exclude='.hipfire/models' --exclude='*.hfq' --exclude='*.mq4*' -e "$SSH_BIN" \
        "$target:$remote/lanes/$name/" "$RUN/lanes/$name/" >"$RUN/lanes/$name.rsync.log" 2>&1 ||
        ! "${RSYNC_CMD[@]}" -a -e "$SSH_BIN" "$target:$remote/RESULTS-$name.sha256" "$RUN/remote-$host/" >>"$RUN/lanes/$name.rsync.log" 2>&1; then
        infra_errors+=("lane $name: fetching results from $target:$remote failed (see $RUN/lanes/$name.rsync.log)")
        LANE_NOTE[i]="results fetch failed"
        continue
    fi
    if [ ! -s "$RUN/remote-$host/RESULTS-$name.sha256" ]; then
        infra_errors+=("lane $name: remote manifest is empty (no result files)")
        LANE_NOTE[i]="no remote results"
    elif ! (cd "$RUN" && sha256sum -c --quiet "remote-$host/RESULTS-$name.sha256" >"$RUN/lanes/$name.verify.log" 2>&1); then
        infra_errors+=("lane $name: fetched results do not match the remote manifest (see $RUN/lanes/$name.verify.log)")
        LANE_NOTE[i]="results sha256 mismatch"
        rm -f "$RUN/lanes/$name/run.json" "$RUN/lanes/$name/hold.json"  # never judge unverified results
    fi
done

# ---------------------------------------------------------------------------------------------
# verdict and pins
# ---------------------------------------------------------------------------------------------

short=${FULL:0:9}
result_dirs=()
for i in "${!LANE_NAMES[@]}"; do
    [ -f "$RUN/lanes/${LANE_NAMES[$i]}/run.json" ] && result_dirs+=("$RUN/lanes/${LANE_NAMES[$i]}") ||
        { [ "${LANE_KIND[$i]}" = skipped ] || infra_errors+=("lane ${LANE_NAMES[$i]}: produced no run.json"); }
done

OVERALL="ERROR" VERDICT_RC=$EXIT_USAGE PIN_NOTE="no pin (no verdict)"
mkdir -p "$(dirname "$PINS")"
if [ "${#result_dirs[@]}" -gt 0 ]; then
    exec 9>"$PINS.lock"
    flock 9
    if "$PYTHON" "$RUN/post_land_smoke.py" verdict --results "${result_dirs[@]}" --plan "$RUN/plan.json" \
        --pins "$PINS" --sha "$FULL" --out "$RUN/VERDICT.json" >"$RUN/verdict.txt" 2>"$RUN/verdict.err"; then
        VERDICT_RC=0
    else
        VERDICT_RC=$?
    fi
    if [ -s "$RUN/VERDICT.json" ]; then
        OVERALL=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["overall"])' "$RUN/VERDICT.json" 2>/dev/null || echo ERROR)
    fi
    if [ "$INTERRUPTED" = 1 ]; then
        PIN_NOTE="no pin (interrupted)"
    else
        case "$VERDICT_RC" in
        0)
            if "$PYTHON" "$RUN/post_land_smoke.py" pin --verdict "$RUN/VERDICT.json" --pins "$PINS" >"$RUN/pin.txt" 2>&1; then
                PIN_NOTE="pinned: $(cat "$RUN/pin.txt")"
            else
                PIN_NOTE="PIN FAILED: $(cat "$RUN/pin.txt")"
            fi ;;
        1)
            failed=$("$PYTHON" - "$RUN/VERDICT.json" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
print(",".join(sorted(c for c, d in v.get("cells", {}).items() if d.get("status") == "FAIL")))
PY
)
            if "$PYTHON" "$RUN/post_land_smoke.py" pin --verdict "$RUN/VERDICT.json" --pins "$PINS" \
                --allow-failed-cells "post-land FAIL at $short: $failed" >"$RUN/pin.txt" 2>&1; then
                PIN_NOTE="pinned the non-failing cells (failed: $failed): $(cat "$RUN/pin.txt")"
            else
                PIN_NOTE="PIN FAILED: $(cat "$RUN/pin.txt")"
            fi ;;
        3) PIN_NOTE="no pin: CHANGED cells; owner runs: post_land_smoke.py pin --verdict $RUN/VERDICT.json --pins $PINS --accept-changed REASON" ;;
        4) PIN_NOTE="no pin: INCOMPLETE (stop and report; a kernel watcher hit or lost device ends the lane)" ;;
        *) PIN_NOTE="no pin (verdict exit $VERDICT_RC)" ;;
        esac
    fi
    flock -u 9
else
    VERDICT_RC=$EXIT_USAGE
    infra_errors+=("no lane produced results: no verdict")
fi
[ "$INTERRUPTED" = 0 ] || { OVERALL="INTERRUPTED"; VERDICT_RC=$EXIT_USAGE; }
if [ "${#infra_errors[@]}" -gt 0 ] && [ "$VERDICT_RC" = 0 ]; then
    : # the verdict itself is authoritative; infrastructure notes are listed in SUMMARY.txt
fi

# ---------------------------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------------------------

hold_status() { # lane dir -> "status: reason"
    "$PYTHON" - "$1/hold.json" <<'PY' 2>/dev/null || echo "no hold.json"
import json, sys
h = json.load(open(sys.argv[1]))
print(f'{h.get("status")}: {h.get("reason")}')
PY
}
{
    echo "post-land smoke: $FULL"
    echo "prev:      $PREV"
    echo "run dir:   $RUN"
    echo "verdict:   $OVERALL (exit $VERDICT_RC)  $RUN/VERDICT.json"
    echo "pins:      $PIN_NOTE"
    echo
    [ ! -s "$RUN/verdict.txt" ] || { cat "$RUN/verdict.txt"; echo; }
    echo "lanes:"
    for i in "${!LANE_NAMES[@]}"; do
        name=${LANE_NAMES[$i]}
        job=$(grep -o 'queued as j[0-9]*' "$RUN/lanes/$name.fabric.log" 2>/dev/null | head -n1 | sed 's/queued as //' || true)
        printf '  %-16s host=%s (%s) rc=%s fabric_job=%s hold=%s%s\n' "$name" "${LANE_HOSTS[$i]}" "${LANE_KIND[$i]}" \
            "${LANE_RC[$i]:--}" "${job:--}" "$(hold_status "$RUN/lanes/$name")" "${LANE_NOTE[$i]:+  [${LANE_NOTE[$i]}]}"
        echo "    loads: ${LANE_LOADS[$i]}"
        echo "    log:   $RUN/lanes/$name.fabric.log"
    done
    if [ "${#infra_errors[@]}" -gt 0 ]; then
        echo
        echo "infrastructure notes:"
        printf '  %s\n' "${infra_errors[@]}"
    fi
    echo
    echo "files: $RUN/{plan.txt,lanes.json,changed.txt,land.diff,VERDICT.json,verdict.txt,lanes/}"
    echo "POST-LAND-SMOKE $FULL $OVERALL $RUN/VERDICT.json"
} >"$RUN/SUMMARY.txt"
cat "$RUN/SUMMARY.txt"
exit "$VERDICT_RC"
