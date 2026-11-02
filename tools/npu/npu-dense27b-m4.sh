#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# Run inside ONE npu-window.sh hold; NPU_LEVELS are --gap-us values.
# Pair medians equal pair means; slowdown uses all four GPU-alone runs.
# Requests, builtin prompt identity, drift and verified NPU summaries are logged.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: $0 OUTDIR" >&2; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p -- "$1"
out=$(cd -- "$1" && pwd)
: > "$out/marks.txt"
: > "$out/req.jsonl"
start=$SECONDS
daemon_pid= npu_pid= fifo_open=0
mark() { printf '%s %s\n' "$(date +%s%N)" "$*" | tee -a "$out/marks.txt"; }
fail() { mark "FAIL: $*" >&2; exit 1; }
stop_child() {
    local pid=$1 end=$((SECONDS + 3))
    kill "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < end )); do sleep 0.05; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
    wait "$pid" 2>/dev/null || true
}
cleanup() {
    local rc=$?
    trap - EXIT
    [[ -z $npu_pid ]] || stop_child "$npu_pid"
    if (( fifo_open )); then exec 9>&-; fi
    [[ -z $daemon_pid ]] || stop_child "$daemon_pid"
    rm -f -- "$out/daemon.fifo"
    local digest
    digest=$(md5sum "$out/req.jsonl")
    mark "requests_md5=${digest%% *} file=req.jsonl"
    mark "end rc=$rc elapsed_s=$((SECONDS - start))"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mem_available() {
    avail=0
    while read -r key value rest; do
        if [[ $key == MemAvailable: ]]; then avail=$value; break; fi
    done < /proc/meminfo
    mark "meminfo $1 MemAvailable=$avail kB"
}
mem_available before-load
if (( avail < 16 * 1024 * 1024 )); then
    mark 'REFUSE: MemAvailable < 16 GiB'
    exit 3
fi
loop_s=${NPU_LOOP_S:-16}
[[ $loop_s =~ ^[0-9]+$ ]] && (( 10#$loop_s > 0 )) || fail 'NPU_LOOP_S must be a positive integer'
loop_s=$((10#$loop_s))
read -r -a levels <<< "${NPU_LEVELS:-0 13000 58000}"
(( ${#levels[@]} > 0 )) || fail 'NPU_LEVELS must not be empty'
for gap in "${levels[@]}"; do
    [[ $gap =~ ^[0-9]+$ ]] || fail "invalid gap: $gap"
done
mkdir -p "$out/home/.hipfire" "$HOME/pm-wave/iu4roof/lockdir"
printf '[reasoning]\neffort = "none"\n[memory]\nmax_seq = 16384\n' > "$out/home/.hipfire/config.toml"
: > "$out/resp.jsonl"
mkfifo "$out/daemon.fifo"
exec 9<> "$out/daemon.fifo"
fifo_open=1
sync
printf '2\n' | sudo -n tee /proc/sys/vm/drop_caches > /dev/null
env -u HIP_VISIBLE_DEVICES -u HIPFIRE_QWEN4_MOE_SYM_IU4 -u HIPFIRE_QWEN4_MOE_SYM_PM -u HIPFIRE_HOME -u HIPFIRE_KV_MODE -u HIPFIRE_STATE_QUANT -u HIPFIRE_IU4_PREFILL \
    HOME="$out/home" HIPFIRE_DEVICES=0000:bf:00.0 ROCR_VISIBLE_DEVICES=1 \
    HIPFIRE_LOCK_DIR="$HOME/pm-wave/iu4roof/lockdir" HIPFIRE_GRAPH=1 HIPFIRE_DPM_WARMUP_SECS=10 \
    HIPFIRE_KERNEL_CACHE="$HOME/pm-wave/iu4roof/kcache" \
    "$HOME/pm-wave/integrate041e/bin/daemon" 9>&- < "$out/daemon.fifo" > "$out/resp.jsonl" 2> "$out/daemon.log" &
daemon_pid=$!
seen=0
response=
req() {
    local json=$1 want=$2 timeout=$3 end=$((SECONDS + $3)) count line
    printf '%s\n' "$json" >&9
    printf '%s\n' "$json" >> "$out/req.jsonl"
    while (( SECONDS < end )); do
        count=0
        while IFS= read -r line; do
            count=$((count + 1))
            (( count > seen )) || continue
            seen=$count
            if [[ $line =~ \"type\"[[:space:]]*:[[:space:]]*\"error\" ]]; then fail "daemon error: $line"; fi
            if [[ $line =~ \"type\"[[:space:]]*:[[:space:]]*\"$want\" ]]; then response=$line; return; fi
        done < "$out/resp.jsonl"
        kill -0 "$daemon_pid" 2>/dev/null || fail "daemon exited before $want"
        sleep 0.05
    done
    fail "timeout waiting for $want (${timeout}s)"
}
wait_child() {
    local pid=$1 timeout=$2 label=$3 end=$((SECONDS + $2))
    while kill -0 "$pid" 2>/dev/null; do
        (( SECONDS < end )) || fail "$label did not exit within ${timeout}s"
        sleep 0.05
    done
    wait "$pid" || fail "$label exited unsuccessfully"
}
# Keep GPU pair medians in integer microseconds (ms truncated at 3 decimals).
ms_us() {
    local whole frac
    [[ $1 =~ ^([0-9]+)(\.([0-9]+))?$ ]] || fail "invalid prefill ms: $1"
    whole=${BASH_REMATCH[1]} frac=${BASH_REMATCH[3]:-0}000
    REPLY=$((10#$whole * 1000 + 10#${frac:0:3}))
}
alone_sum=0
last_us=0
prefill() {
    local tag=$1 ms tok_s
    mark "prefill-start $tag prompt=bench_prefill-builtin"
    req '{"type":"bench_prefill","tokens":8192}' prefill_result 60
    [[ $response =~ \"ms\"[[:space:]]*:[[:space:]]*([0-9]+(\.[0-9]+)?) ]] || fail "missing ms: $response"
    ms=${BASH_REMATCH[1]}
    [[ $response =~ \"tok_s\"[[:space:]]*:[[:space:]]*([0-9]+(\.[0-9]+)?) ]] || fail "missing tok_s: $response"
    tok_s=${BASH_REMATCH[1]}
    mark "prefill-end $tag ms=$ms tok_s=$tok_s"
    ms_us "$ms"
    last_us=$REPLY
}
mark load-start
req '{"type":"load","model":"/home/kaden/pm-wave/g11fa2/models/qwen3.8-27b.mq4-xts","params":{"max_seq":16384}}' loaded 170
mark load-end
mem_available after-load
if (( avail < 16 * 1024 * 1024 )); then
    mark 'REFUSE: after load MemAvailable < 16 GiB'
    req '{"type":"unload"}' unloaded 60
    mark unloaded
    exit 3
fi
prefill warm
for i in 1 2; do prefill "gpu-alone-$i"; alone_sum=$((alone_sum + last_us)); done
first_pair_sum=$alone_sum
level_sums=() npu_summaries=()
for gap in "${levels[@]}"; do
    log="$out/npu-g$gap.log"
    mark "npu-start gap_us=$gap loop_s=$loop_s"
    (cd "$repo" && exec ./target/release/npu-gemm 4096 4096 5120 --variant V8 --loop "$loop_s" --gap-us "$gap" --timeout-ms 3000) > "$log" 2>&1 &
    npu_pid=$!
    end=$((SECONDS + 30))
    while ! grep -q 'submit 0:' "$log"; do
        kill -0 "$npu_pid" 2>/dev/null || fail "NPU gap=$gap exited before submit 0"
        (( SECONDS < end )) || fail "NPU gap=$gap first-submit timeout"
        sleep 0.05
    done
    mark "npu-first-submit gap_us=$gap"
    sum=0
    for i in 1 2; do
        kill -0 "$npu_pid" 2>/dev/null || fail "NPU gap=$gap ended before concurrent prefill $i"
        prefill "concurrent-g$gap-$i"
        sum=$((sum + last_us))
    done
    wait_child "$npu_pid" "$((loop_s + 15))" "NPU gap=$gap"
    npu_pid=
    grep -q '^verify: first=exact last=exact$' "$log" || fail "NPU gap=$gap first/last verification failed"
    summary=$(grep '^summary:.*npu_dram_GBps=' "$log") || fail "NPU gap=$gap missing bandwidth summary"
    level_sums+=("$sum")
    npu_summaries+=("$summary; verify: first=exact last=exact")
    mark "npu-end gap_us=$gap rc=0"
done
for i in 3 4; do prefill "gpu-alone-$i"; alone_sum=$((alone_sum + last_us)); done
req '{"type":"unload"}' unloaded 60
mark unloaded
exec 9>&-
fifo_open=0
wait_child "$daemon_pid" 15 daemon
daemon_pid=
(( alone_sum > 0 )) || fail 'zero GPU-alone mean'
mean_alone=$((alone_sum / 4))
mark "summary gpu_alone_mean_ms=$((mean_alone / 1000)).$(printf '%03d' "$((mean_alone % 1000))") samples=4"
last_pair_sum=$((alone_sum - first_pair_sum))
(( first_pair_sum > 0 )) || fail 'zero first GPU-alone pair median'
drift=$(((last_pair_sum - first_pair_sum) * 10000 / first_pair_sum))
sign=
if (( drift < 0 )); then sign=-; drift=$((-drift)); fi
mark "drift_pct=$sign$((drift / 100)).$(printf '%02d' "$((drift % 100))")"
diff=$((last_pair_sum - first_pair_sum))
if (( diff < 0 )); then diff=$((-diff)); fi
if (( diff * 100 > first_pair_sum * 3 )); then mark 'UNRELIABLE drift>3%'; fi
for idx in "${!levels[@]}"; do
    mean=$((level_sums[idx] / 2))
    slowdown=$(((2 * level_sums[idx] - alone_sum) * 10000 / alone_sum))
    sign=
    if (( slowdown < 0 )); then sign=-; slowdown=$((-slowdown)); fi
    mark "summary gap_us=${levels[idx]} gpu_median_ms=$((mean / 1000)).$(printf '%03d' "$((mean % 1000))") slowdown_pct=$sign$((slowdown / 100)).$(printf '%02d' "$((slowdown % 100))") ${npu_summaries[idx]}"
done
