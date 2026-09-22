// The unchanged production FA2 source, built in a separate KMODE=0 translation unit.
#define HIPFIRE_FA2_KT 32
#define HIPFIRE_FA2_KMODE 0
#define attention_q8_0_fa2_gqa_gfx11 attention_q8_0_fa2_gqa_q8_probe
#include "../kernels/src/attention_q8_0_fa2_gqa.gfx11.hip"
