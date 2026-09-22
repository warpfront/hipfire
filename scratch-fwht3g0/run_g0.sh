#!/bin/sh
set -eu
export HIP_VISIBLE_DEVICES=0
h=/home/kaden/hipfire-fwht3g0/scratch-fwht3g0/fwht3_g0
for start in 0 32768 65536 131072 245760 253952; do
    "$h" "$start" parity all
done
for block in 1 2 3; do
    for start in 0 32768 65536 131072 245760 253952; do
        for arm in F S S F; do
            echo "PROCESS block=$block start=$start arm=$arm"
            "$h" "$start" "$arm" all
        done
    done
done
echo 'G0 COMPLETE'
