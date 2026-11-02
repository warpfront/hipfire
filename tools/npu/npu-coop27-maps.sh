#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# npu-coop27-maps.sh — one npu-window.sh hold: the shared-page mapping matrix (coop27 --mode maps; no NPU command, the NPU
# device is opened only for userptr import and its BO mapping). Reserves hugetlb pages for the hugetlb rows (160 x 2 MiB,
# 1 x 1 GiB; best effort) and restores the incoming counts on exit.
#   ~/pm-wave/hipx-lock.sh timing -- env NPU_WINDOW_TIMEOUT=90 NPU_FCLK_TOOL=$BIN/npu-tools BIN=.. DATA=.. tools/npu/npu-window.sh tools/npu/npu-coop27-maps.sh
set -u
BIN=${BIN:?BIN}; DATA=${DATA:?DATA}
H2=/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
H1=/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
OLD2=$(cat $H2); OLD1=$(cat $H1)
restore() { echo "$OLD2" | sudo -n tee $H2 >/dev/null; echo "$OLD1" | sudo -n tee $H1 >/dev/null; echo "hugepages restored: 2M=$(cat $H2) 1G=$(cat $H1)"; }
trap restore EXIT
echo 160 | sudo -n tee $H2 >/dev/null; echo 1 | sudo -n tee $H1 >/dev/null
echo "hugepages reserved: 2M=$(cat $H2) 1G=$(cat $H1)"
timeout --foreground --signal=TERM --kill-after=2 75 "$BIN/coop27" --mode maps --layer 2 --xq "$DATA/L2.down.a4.bin"
rc=$?
echo "step rc=$rc"
exit $rc
