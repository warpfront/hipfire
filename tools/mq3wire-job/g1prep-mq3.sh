#!/bin/bash
# mq3wire G1 prep (CPU only, no card, no fabric): real mq3-xts samples + the Down chain for the W3 K2 byte oracle.
# Run on hipx (inputs staged under singlecard/mq3wire/g1). Idempotent: steps whose output exists are skipped.
#   bash g1prep-mq3.sh
X=/home/kaden/qcal/release-0.4.5/mgmtp/singlecard/mq3wire; G=$X/g1; B=$X/bin2
MODEL=/home/kaden/.hipfire/models/qwen3.8-flash-next-base.mq3-xts
SHA=d35896ca69b87d3e93176847a036db3e2cb1554931f130a842b13718b486238b   # published mq3-xts (59,109,362,200 B), registry/README
PROBE=$G/probe/observations.json; DIV=$G/probe/divide-observations.json
set -e
# needs on hipx: $X/tools/{mq3_k2_extract_samples,e8p_k1_extract_samples,qwen4_moe_gfx1100_chain_a4,mq_i4_a4_oracle}.py and $G/probe/{observations,divide-observations,intrinsic-merged.c1.r3}.json
[ -e $G/samples-mq3/sample.json ] || nice -n 19 python3 $X/tools/mq3_k2_extract_samples.py --artifact $MODEL --expect-sha256 $SHA --output-dir $G/samples-mq3
if [ ! -e $G/chain/w3/chain-a4.json ]; then
  nice -n 19 $B/qwen4_moe_sym_gfx1100_g1_oracle --prepare-chain $G/chain/w3 --codec w3 --sample-dir $G/samples-mq3 --probe-table $PROBE --divide-table $DIV --intrinsic-table $G/probe/intrinsic-merged.c1.r3.json --out $G/prepare-w3.json
  python3 $X/tools/qwen4_moe_gfx1100_chain_a4.py --chain-dir $G/chain/w3
fi
echo "g1prep OK: $(ls $G/samples-mq3 | wc -l) sample files, chain $(ls $G/chain/w3 | wc -l) files"
