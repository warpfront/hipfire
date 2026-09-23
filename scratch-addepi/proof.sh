#!/bin/bash
# AddEpilogue Halo proof on the stack-1 port (one both-cards-quiet lease):
# pp8192 rocprof A (HIPFIRE_V2B_ADDEPI=0) and B in the same binaries; oracle of
# the runtime JIT objects (trace-B cache) and of hipcc objects from this tree
# against stack-1 (8397738d4) reference objects, N8192/N512 F/R timing; WT2
# c24 q8/q8 off vs on (byte identity). usage: proof.sh [trace|jit|standalone|wt2]...
set -uo pipefail
d=/home/kaden/hipfire-addepi-stack/scratch-addepi
cd $d
for ph in "$@"; do
  case $ph in
    standalone)
      ADDEPI_PROD=1 ADDEPI_REF=1 ADDEPI_VARS=prod ./run_standalone.sh P1 gfx1151 oracle F R
      ADDEPI_PROD=1 ADDEPI_VARS=prod ADDEPI_N=512 ./run_standalone.sh P1n512 gfx1151 F R ;;
    jit)
      # Runtime JIT objects (trace-B daemon cache) vs pre-lever reference
      # objects built from stack-1 8397738d4 (build_ref.sh): the production-object oracle.
      src=$d/trace/gfx1151-8192-B/home/.hipfire_kernels/gfx1151; j=$d/binjit
      rm -rf $j && mkdir -p $j && cp bin/v2b_ref_gfx1151.hsaco bin/norm_ref_gfx1151.hsaco $j/
      g=$(ls $src/gemm_mq4g256v2_residual_iu4_v2b_gfx11.*.hsaco)
      cp $g $j/v2b_prod_gfx1151.hsaco && cp $g $j/v2b_prodadd_gfx1151.hsaco
      cp $(ls $src/fused_rmsnorm_mq_rotate_awq_i4.*.hsaco) $j/norm_base_prod_gfx1151.hsaco
      cp $(ls $src/fused_rmsnorm_mq_rotate_awq_i4_fold.*.hsaco) $j/norm_fold_prod_gfx1151.hsaco
      ADDEPI_BIN=$j ADDEPI_PROD=1 ADDEPI_REF=1 ADDEPI_VARS=prod,prodadd ./run_standalone.sh J1 gfx1151 oracle F R ;;
    trace)
      python3 run_profile.py gfx1151 8192 A && python3 run_profile.py gfx1151 8192 B && python3 inventory.py && python3 trace_delta.py ;;
    wt2)
      python3 run_wt2.py gfx1151 ;;
  esac
  echo "proof phase $ph rc=$?"
done
rocm-smi --showpids
