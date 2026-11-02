#!/bin/bash
# run.sh <gfx1201|gfx1151> <binary> <outdir> <suites> [card-selector]
# One bounded fabric hold (class correct, max 300 s) around numerics_probe. Nothing else runs on the card.
#   gfx1201: local hiptrx card (default B = GPU-e475645fe0200397; C = GPU-085289909a86cc63)
#   gfx1151: ssh hipx, card `halo` (no ROCr UUID: identity is the KFD unique_id/location_id recorded below)
# After the hold the host-side identity (rocminfo Uuid/BDFID, KFD node properties, md5 of the probe binary) is written
# next to the receipts as identity.host.txt.
set -euo pipefail
ARCH=${1:?arch}; BIN=${2:?binary}; OUT=${3:?outdir}; SUITES=${4:-all}; CARD=${5:-}
MEMGB=${NP_MEM_GB:-1}
case "$ARCH" in
  gfx1201)
    CARD=${CARD:-GPU-e475645fe0200397}
    mkdir -p "$OUT"
    /home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card "$CARD" --mem-gb "$MEMGB" --max-secs 300 --wait-secs ${NP_WAIT:-3300} -- "$BIN" "$OUT" "$SUITES"
    { echo "host=$(hostname) card=$CARD"; md5sum "$BIN"; rocminfo | grep -E "Name:.*gfx|Uuid|BDFID|Marketing" ; } > "$OUT/identity.host.txt"
    ;;
  gfx1151)
    CARD=${CARD:-halo}
    R=pm-wave/numerics/$(basename "$OUT")
    ssh hipx "mkdir -p ~/$R"
    scp -q "$BIN" "hipx:$R/numerics_probe"
    ssh hipx "~/qcal/locks/fabric/fabric-run.sh --class correct --card $CARD --mem-gb $MEMGB --max-secs 300 --wait-secs ${NP_WAIT:-3300} -- \$HOME/$R/numerics_probe \$HOME/$R/out $SUITES"
    ssh hipx "cd ~/$R && { echo host=\$(hostname) card=$CARD; md5sum numerics_probe; rocminfo | grep -E 'Name:.*gfx|Uuid|BDFID|Marketing'; for n in /sys/class/kfd/kfd/topology/nodes/*; do grep -qE '^simd_count [1-9]' \$n/properties && { echo \$n; grep -E '^(gpu_id|unique_id|location_id|domain|device_id|vendor_id|drm_render_minor|simd_count|gfx_target_version) ' \$n/properties; }; done; lspci -nn -d 1002: 2>/dev/null; } > out/identity.host.txt"
    mkdir -p "$OUT"
    rsync -a "hipx:$R/out/" "$OUT/"
    ;;
  *) echo "unsupported arch $ARCH" >&2; exit 2 ;;
esac
