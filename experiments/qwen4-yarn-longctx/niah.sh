#!/usr/bin/env bash
# niah.sh (YarnQwen): multi-needle retrieval through a native `hipfire serve` of the pinned Flash-Next XTS on
# gfx1151, inside ONE fabric hold on hipx. One fresh serve process per arm.
#   fabric-run.sh --class timing --card halo --mem-gb 120 --hostmem-lock --max-secs 1790 --priority P2 \
#     --owner YarnQwen -- /home/kaden/pm-wave/yarn-qwen/niah.sh ARM...
#   ARM = FIXTURE:FACTOR:MAXSEQ:KV   e.g. 48k:native:57344:auto  48k:2:57344:auto  500k:2:524288:q8
# Standing rule: nothing above 57K context on the Halo without Main's explicit OK (the 300k/500k arms are
# prepared, not run, until then). FACTOR native = no override; F = HIPFIRE_YARN_FACTOR=F (engine YaRN).
# The 500K memory plan: kv q8 (QSA K/V Q8_0 + BF16 index arenas), session cache off, MTP off.
set -uo pipefail
R=/home/kaden/pm-wave/yarn-qwen
BIN=${BIN:-$R/bin}
M=/home/kaden/pm-wave/fngptq/models/qwen3.8-flash-next-gptq3.mq4
[ "${HIPFIRE_DEVICES:-}" = gfx1151 ] || { echo "device guard: HIPFIRE_DEVICES=${HIPFIRE_DEVICES:-} (run under fabric --card halo)" >&2; exit 4; }
rc=0
for spec in "$@"; do
  IFS=: read -r fixture factor maxseq kv <<< "$spec"
  if [ "$maxseq" -gt 57344 ] && [ "${MAIN_OK_LONGCTX:-}" != 1 ]; then
    echo "refused $spec: max_seq $maxseq > 57344 needs Main's explicit OK (set MAIN_OK_LONGCTX=1 only then)" >&2
    rc=3; continue
  fi
  O=$R/runs/niah-$fixture-f$factor-$kv; rm -rf "$O"; mkdir -p "$O/home" "$O/lockdir"
  E=(env)
  for v in $(compgen -e); do
    case "$v" in HIPFIRE_DEVICES|ROCR_VISIBLE_DEVICES|FABRIC_*|HIPFIRE_FABRIC_ADMISSION) ;;
      HIPFIRE_*|HIP_VISIBLE*|HSA_*|GPU_*) E+=(-u "$v");; esac
  done
  E+=(HOME="$O/home" HIPFIRE_LOCK_DIR="$O/lockdir" HIPFIRE_KERNEL_CACHE="$R/kcache" HIPFIRE_NO_REGISTRY_FETCH=1
    HIPFIRE_DAEMON_BIN="$BIN/daemon" HIPFIRE_CLI_BIN="$BIN/hipfire" HIPFIRE_SESSION_CACHE_BYTES=0)
  [ "$factor" != native ] && E+=(HIPFIRE_YARN_FACTOR="$factor")
  H=(python3 -I "$R/serve_harness.py" --model "$M" --mtp off --dflash off --mode battery --sampling greedy
    --thinking off --max-tokens 160 --max-seq "$maxseq" --kv "$kv" --niah-file "$R/niah/niah_long_$fixture.jsonl"
    --port ${PORT:-19571} --serve-log "$O/serve.log" --out "$O/niah.json")
  printf '%q ' "${E[@]}" "${H[@]}" > "$O/command.txt"
  md5sum "$BIN/daemon" "$BIN/hipfire" "$R/niah/niah_long_$fixture.jsonl" > "$O/identity.md5"
  "${E[@]}" timeout ${TMO:-1700} "${H[@]}" > "$O/niah.out" 2> "$O/niah.err"
  r=$?; echo "$(date -u +%T) niah $spec rc=$r" | tee -a "$R/runs/run.log"; [ $r != 0 ] && rc=$r
done
exit $rc
