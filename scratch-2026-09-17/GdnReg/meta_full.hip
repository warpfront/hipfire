#include "full_isa.hip"
int main() {
    hipFuncAttributes c;
    hipFuncGetAttributes(&c, (const void*)gated_delta_net_q8_register_scan_gfx1201);
    printf("vgpr=%d\n", c.numRegs);
    return 0;
}
