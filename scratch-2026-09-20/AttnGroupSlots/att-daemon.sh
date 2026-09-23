#!/bin/sh
exec rocprofv3 --att --att-target-cu 0 \
  --kernel-include-regex 'attention_fp8_e4m3_fa2_gqa_packet' \
  -d /home/kaden/ClaudeCode/warpfront/wt-attnslots/scratch-2026-09-20/AttnGroupSlots/att-candidate-replay \
  -o att -- \
  /home/kaden/ClaudeCode/warpfront/wt-attnslots/target/release/daemon "$@"
