#!/bin/bash
# Bitwise producer oracle for HIPFIRE_G12_A4C2 on gfx1201.
# Reference: b_c2/ = base-tree (0f6cea0dd) kernels.rs concatenation compiled
# with the production argv + -DIU4_A4_CANDIDATES=2 (the flag build).
# Candidate: n_c2/ = this tree, same argv and flag (one-pass producer-layout
# emit). Every gfx1201-compiled source that calls
# emit_iu4_sidecar_from_producer8 is covered at its production shape; the
# four pp8192 producers (and plain twins) also run tail shapes (odd N, N=1,
# group counts that leave a partial wave/workgroup), normal and special data
# (zero, -0, denormal, huge, Inf/NaN rows), several seeds, x_rot on/off.
cd "$(dirname "$0")"
fail=0
declare -A SYM
while read -r c s; do SYM[$c]=$s; done < syms.txt
fam_of() {
  case $1 in
    FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_FOLD_SRC) echo rmsfold ;;
    FUSED_RMSNORM_MQ_ROTATE_I4_FOLD_SRC) echo rmsfoldn ;;
    FUSED_RMSNORM_MQ_ROTATE_AWQ_*) echo rms ;;
    FUSED_RMSNORM_MQ_ROTATE_I4_*) echo rmsn ;;
    GATED_NORM_MQ_ROTATE_AWQ_*) echo gated ;;
    GATED_NORM_MQ_ROTATE_I4_*) echo gatedn ;;
    SIGMOID_*) echo sig ;;
    MQ_ROTATE_X_AWQ_*) echo rotawq ;;
    MQ_ROTATE_X_I4_*) echo rot ;;
    FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN*) echo hin ;;
    FUSED_SILU_MUL_MQ_ROTATE_AWQ_*) echo siluawq ;;
    FUSED_SILU_MUL_MQ_ROTATE_I4_*) echo silu ;;
  esac
}
geom() {  # const K N xrot -> gx:gy:block:shmem
  local c=$1 K=$2 N=$3 xr=$4 g=$(( $2 / 256 ))
  case $(fam_of "$c") in
    rms*) echo "$N:1:256:$(( xr ? (K + 256) * 4 : 1024 ))" ;;
    gated*) if [[ $c == *_V2_SRC ]]; then echo "$(( (g + 1) / 2 )):$N:64:0"; else echo "$g:$N:64:0"; fi ;;
    sig|rot|rotawq) echo "$(( N * g )):1:32:0" ;;
    *) echo "$g:$N:32:0" ;;
  esac
}
n_run=0
run() {  # const K N seed xrot data   (SKIP=n resumes after the first n cases)
  n_run=$((n_run + 1)); [ "$n_run" -le "${SKIP:-0}" ] && return
  local c=$1 f; f=$(fam_of "$1")
  local gm; gm=$(geom "$c" "$2" "$3" "$5")
  local out line
  out=$(DATA=$6 ./harness "$f" oracle "$2" "$3" "$4" "$5" \
        "A:b_c2/$c.hsaco:${SYM[$c]}:$gm" "A:n_c2/$c.hsaco:${SYM[$c]}:$gm" 2>&1)
  line=$(echo "$out" | grep -E 'ORACLE_(PASS|FAIL)' || echo "NO_RESULT $(echo "$out" | tail -1)")
  printf '%-46s %-8s K=%-5s N=%-4s seed=%-2s xrot=%s %-7s | %s | %s\n' "$c" "$f" "$2" "$3" "$4" "$5" "$6" \
    "$(echo "$out" | grep -oE 'untouched blocks: [0-9]+')" "$(echo "$line" | sed 's/^.*: i4/i4/')"
  echo "$line" | grep -q ORACLE_PASS || fail=1
}
prodK() {
  case $(fam_of "$1") in
    rms*) echo 5120 ;; gated*|sig|rot*) echo 6144 ;; *) echo 17408 ;;
  esac
}
# 1. Every emit source at its production shape, normal + special data.
for c in $(cat consts.txt); do
  for data in normal special; do run "$c" "$(prodK "$c")" 8192 11 0 $data; done
done
# 2. Tails and x_rot for the pp8192 producers and their plain twins.
for c in FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SRC FUSED_RMSNORM_MQ_ROTATE_I4_GFX12_V2_SRC; do
  for data in normal special; do
    for kn in "5120 8192 12 1" "5120 1000 13 0" "5120 17 14 1" "5120 1 15 0" "4096 777 16 0" \
              "2560 513 17 1" "1024 64 18 0" "6144 333 19 0" "7168 65 20 1"; do
      set -- $kn; run "$c" "$1" "$2" "$3" "$4" $data
    done
  done
done
for c in GATED_NORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SRC GATED_NORM_MQ_ROTATE_I4_GFX12_V2_SRC; do
  for data in normal special; do
    for kn in "6144 8192 12 1" "6144 1000 13 0" "6144 1 14 0" "4096 513 15 1" "3328 77 16 0" \
              "512 64 17 0" "8192 33 18 1"; do
      set -- $kn; run "$c" "$1" "$2" "$3" "$4" $data
    done
  done
done
for c in SIGMOID_MUL_MQ_ROTATE_X_AWQ_I4_GFX12_SRC; do
  for data in normal special; do
    for kn in "6144 8192 12 1" "6144 1000 13 0" "6144 1 14 0" "5120 333 15 0" "3328 77 16 1"; do
      set -- $kn; run "$c" "$1" "$2" "$3" "$4" $data
    done
  done
done
for c in FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN_GFX12_SRC FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_GFX12_SRC; do
  for data in normal special; do
    for kn in "17408 8192 12 0" "17408 1000 13 1" "17408 1 14 0" "17408 333 15 0" "4096 777 16 1" \
              "13824 65 17 0"; do
      set -- $kn; run "$c" "$1" "$2" "$3" "$4" $data
    done
  done
done
[ $fail = 0 ] && echo "ORACLE_MATRIX_PASS" || echo "ORACLE_MATRIX_FAIL"
