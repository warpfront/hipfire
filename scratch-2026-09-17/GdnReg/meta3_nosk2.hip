#include "stub3_nosk2.hip"
#include <cstdio>
int main() {
    hipFuncAttributes c;
    if (hipFuncGetAttributes(&c, (const void*)gated_delta_net_q8_register_scan_gfx1201) != hipSuccess) return 1;
    int occ = 0;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, gated_delta_net_q8_register_scan_gfx1201, 256, 0);
    printf("vgpr=%d sgpr=%d lds=%zu scratch=%zu vspill=%d sspill=%d occ=%d\n",
        c.numRegs, 0, c.sharedSizeBytes, c.localSizeBytes, 0, 0, occ);
    return 0;
}
