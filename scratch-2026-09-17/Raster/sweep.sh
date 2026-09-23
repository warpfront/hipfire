#!/bin/bash
# Raster/cpol sweep driver: set variant defines in the iu4 gfx12 kernel,
# rebuild the probe, run oracle + rig, log to evidence dir.
# Usage: ./sweep.sh <variant-name> [G] [T] [A_CPOL] [W_CPOL]
#   G/T/A_CPOL/W_CPOL: values for IU4_SWIZZLE_G/T, IU4_A/W_CPOL (default 0)
R=/home/kaden/ClaudeCode/warpfront/wt-raster
K=$R/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip
EV=$R/scratch-2026-09-17/Raster
V=$1; G=${2:-0}; T=${3:-0}; AC=${4:-0}; WC=${5:-0}
export HOME=/home/kaden/.hipfire-homes/ab3 ROCR_VISIBLE_DEVICES=3
export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab3/.hipfire_kernels
export HIPFIRE_GRAPH=1 HIPFIRE_LLOYD_GFX12=1
sed -i -e "s/^#define IU4_SWIZZLE_G .*/#define IU4_SWIZZLE_G $G/" \
       -e "s/^#define IU4_SWIZZLE_T .*/#define IU4_SWIZZLE_T $T/" \
       -e "s/^#define IU4_A_CPOL .*/#define IU4_A_CPOL $AC/" \
       -e "s/^#define IU4_W_CPOL .*/#define IU4_W_CPOL $WC/" $K
grep -n "^#define IU4_\(SWIZZLE_G\|SWIZZLE_T\|A_CPOL\|W_CPOL\)" $K
cargo build --release -p saddle-lab --example kx3_foldfree_probe 2>$EV/$V.build.log | tail -2
./target/release/examples/kx3_foldfree_probe 2>$EV/$V.resource.log | tee $EV/$V.rig.log
