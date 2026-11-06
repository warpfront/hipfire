#!/bin/bash
# mq3wire M1: XTX (gfx1100) `qwen3.8-flash-next-base.mq3-xts` decode-only tier. ONE fabric timing job (<=25 min).
# Fresh `hipfire bench` process per cell. Arms (env per arm; every other HIPFIRE_* is unset):
#   D  HIPFIRE_OFFLOAD_EXEC=cpu    D2 armed: every cold/host expert (W3 planar decoders + MQ4) executes on the CPU
#   P  HIPFIRE_OFFLOAD_EXEC=pcie   D2 off: the W3 K1 reads host-mapped arenas over PCIe (kernel-on-card check)
# Every arm: HIPFIRE_PREFILL_CHUNK_ROWS=256 (placement + scratch sized for a decode-only load; the forward caps the
# prompt tile at the 8-row K1 width itself, there is no W3 K2 on gfx1100).
# Cells (name:arm:spec:prompt:warmups:runs:max-tokens), run in order, skipped when the cap leaves no room:
CELLS_DEFAULT="ar-code-128:D:off:code-128:1:3:256 mtp-code-128:D:mtp:code-128:1:3:256 ar-prose-128:D:off:prose-128:1:3:256 mtp-prose-128:D:mtp:prose-128:1:3:256 ar-code-128-pcie:P:off:code-128:1:2:128 ar-code-1k:D:off:code-1k:0:1:64"
# usage (on hipx, inside fabric-run.sh):  [CELLS="..."] [CAP=1380] bash xjob-mq3.sh
X=/home/kaden/qcal/release-0.4.5/mgmtp/singlecard/mq3wire; XN=/home/kaden/qcal/release-0.4.5/mgmtp/xtxnow; B=${BIN:-$X/bin}; O=${OUT:-$X/ev-$(date -u +%m%d-%H%M%S)}; D=/sys/bus/pci/devices/0000:67:00.0
CELLS=${CELLS:-$CELLS_DEFAULT}; CAP=$((${CAP:-1380}))
MD=$X/home/.hipfire/models; MODEL=qwen3.8-flash-next-base.mq3-xts; SRC=/home/kaden/.hipfire/models
mkdir -p $O $MD $X/prompts
for f in "" .ple-q4; do [ -e $MD/$MODEL$f ] || ln $SRC/$MODEL$f $MD/ || { echo "cannot link $MODEL$f" >&2; exit 2; }; done
for p in code prose; do head -c 512 $XN/prompts/$p-1k.txt > $X/prompts/$p-128.txt; cp -f $XN/prompts/$p-1k.txt $X/prompts/$p-1k.txt; done
export HIPFIRE_HOME=$X/home/.hipfire HOME=$X/home HIPFIRE_DAEMON_BIN=$B/daemon HIPFIRE_REGISTRY_URL=http://127.0.0.1:1/v1.json HIPFIRE_NO_REGISTRY_FETCH=1
for v in $(compgen -e); do case $v in HIPFIRE_DEVICES|HIPFIRE_LOCK_DIR|HIPFIRE_FABRIC_ADMISSION|HIPFIRE_HOME|HIPFIRE_DAEMON_BIN|HIPFIRE_REGISTRY_URL|HIPFIRE_NO_REGISTRY_FETCH) ;; HIPFIRE_*|GPU_MAX_HW_QUEUES) unset $v;; esac; done
export HIPFIRE_JINJA_CHAT=0
T0=$(date +%s)
{ echo "date=$(date -u +%FT%TZ) FABRIC_JOB=${FABRIC_JOB:-} FABRIC_CARDS=${FABRIC_CARDS:-}"; uptime; cat /proc/loadavg; echo "GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-unset (HIP default 4)}"
  env | grep -E '^(HIPFIRE|GPU_MAX|ROCR|HSA)' | sort; echo "CELLS=$CELLS CAP=$CAP"
  md5sum $B/hipfire $B/daemon $X/prompts/*; wc -c $X/prompts/*; sha256sum $MD/$MODEL | cut -c1-16; stat -L -c "%n %s" $MD/$MODEL $MD/$MODEL.ple-q4
  echo "link: $(cat $D/current_link_width) x $(cat $D/current_link_speed)"; free -g | head -2; cat $X/SOURCE; } > $O/pre.txt 2>&1
LS=$(cat $D/current_link_width)
( while sleep 1; do N=$(cat $D/current_link_width 2>/dev/null); if [ "$N" != "$LS" ] || [ ! -e $D/vendor ]; then echo "LINK CHANGE $LS -> $N $(date -u)" > $O/LINK_DROPPED; [ -s $O/cur.pid ] && kill -KILL -- -$(cat $O/cur.pid); break; fi; done ) & W=$!
arm_env() { case $1 in D) echo "HIPFIRE_OFFLOAD_EXEC=cpu HIPFIRE_PREFILL_CHUNK_ROWS=256" ;; P) echo "HIPFIRE_OFFLOAD_EXEC=pcie HIPFIRE_PREFILL_CHUNK_ROWS=256" ;; *) echo "bad arm $1" >&2; exit 2 ;; esac; }
cell() { local tag=$1 arm=$2; shift 2; local o=$O/$tag; mkdir -p $o
  echo "start $(date -u +%T) uptime: $(cat /proc/loadavg) arm=$arm env=$(arm_env $arm)" > $o/meta.txt
  ( while sleep 0.5; do cat $D/mem_info_vram_used; done > $o/vram.txt 2>/dev/null ) & local V=$!
  setsid env $(arm_env $arm) timeout 900 $B/hipfire bench $MODEL "$@" --json > $o/out.json 2> $o/err.log & echo $! > $O/cur.pid; wait $!; echo "rc=$? end $(date -u +%T)" >> $o/meta.txt
  kill $V 2>/dev/null; sort -n $o/vram.txt | tail -1 | awk '{printf "vram_peak_MiB=%.0f\n", $1/1048576}' >> $o/meta.txt
  $B/hipfire stop >/dev/null 2>&1; sleep 3; }
last=0
for c in $CELLS; do
  IFS=: read -r name arm spec pr warm runs maxtok <<< "$c"
  el=$(( $(date +%s) - T0 )); need=$(( last / 2 + 120 ))
  [ -e $O/LINK_DROPPED ] && { echo "link dropped; stop"; break; }
  if [ $el -gt $((CAP - need)) ]; then echo "SKIP $name (elapsed $el)"; continue; fi
  s=$(date +%s)
  cell $name $arm --prompt-file $X/prompts/$pr.txt --max-tokens $maxtok --warmups $warm --runs $runs --spec $spec
  last=$(( $(date +%s) - s )); echo "done $name ${last}s (elapsed $(( $(date +%s) - T0 )))"
done
kill $W 2>/dev/null; echo "ALLDONE $(date -u)"
