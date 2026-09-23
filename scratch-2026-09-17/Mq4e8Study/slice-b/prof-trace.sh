#!/bin/bash
# rocprofv3 kernel-trace of one oracle time call (xmaster on), K=5120.
export HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0
export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels
export HIPFIRE_GRAPH=0 HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=1
cd /home/kaden/ClaudeCode/warpfront/wt-mq4e8
/opt/rocm/core/bin/rocprofv3 --kernel-trace -d scratch-2026-09-17/Mq4e8Study/slice-b/prof-k5120 -- ./target/release/examples/tmp_iu4_xmaster_oracle --k 5120 time 2>scratch-2026-09-17/Mq4e8Study/slice-b/prof-k5120.stderr
echo EXIT=$?
