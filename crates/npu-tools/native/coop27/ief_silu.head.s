; PM-native gfx1151 gate/up SiLU epilogue for the coop27 gate_up arm. Assembled by native/build.sh as
; ief_silu.head.s + silu_dag_gfx1151.inc + ief_silu.tail.s (no vendor assembler).
; silu_dag_gfx1151.inc is the verbatim body of one SiLU group of the incumbent `gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151`
; epilogue (`peacemaker-isa emit --kernel iu4_v2b --epi all --arch gfx1151`, wt-land-041m 798d63b3a, `iu4_v2b::silu_group` ->
; `Epilogue::silu_dense`): h[k] = g[k] / (1 + expf(-g[k])) * u[k], gate in v32..v39, up in v64..v71, scratch v160..v207,
; s48..s95, vcc; same float mode bits as that entry (denorm 3, ieee 1, round 0).
;
; ief_silu  kernarg 40 B: c u64 @0, y u64 @8, ldy u32 @16, col0 u32 @20, waves u32 @24, NW u32 @28, half u32 @32.
;           Block 256, grid [2*half, 2*MW, 8]. The NPU command's features are [g rows | u rows] (half N-waves each); workgroup
;           (nw*2 + h, mw*2 + s, col), nw < half, reads the g tile of wave mw*NW + nw and the u tile of wave mw*NW + nw + half
;           (same in-tile offsets, +half*16 KiB) and stores h[t*ldy + col0 + r] for its 64 tokens x 32 hidden features.
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected ief_silu
.globl ief_silu
.p2align 8
.type ief_silu,@function
ief_silu:
    s_load_b256 s[16:23], s[0:1], null
    s_load_b32 s24, s[0:1], 0x20
    s_lshr_b32 s5, s2, 1
    s_and_b32 s6, s2, 1
    s_lshr_b32 s7, s3, 1
    s_and_b32 s8, s3, 1
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s9, s7, s23
    s_add_u32 s9, s9, s5
    s_lshl_b32 s10, s4, 2
    s_lshl_b32 s11, s8, 1
    s_add_u32 s10, s10, s11
    s_mul_i32 s10, s10, s22
    s_lshl_b32 s11, s9, 1
    s_add_u32 s11, s11, s6
    s_add_u32 s10, s10, s11
    s_lshl_b32 s10, s10, 13
    s_add_u32 s12, s16, s10
    s_addc_u32 s13, s17, 0
    s_and_b32 s11, s4, 1
    s_lshl_b32 s14, s8, 1
    s_add_u32 s14, s14, s11
    s_lshl_b32 s14, s14, 6
    s_lshl_b32 s15, s7, 8
    s_add_u32 s14, s14, s15
    s_lshl_b32 s15, s5, 8
    s_lshr_b32 s11, s4, 1
    s_lshl_b32 s11, s11, 6
    s_add_u32 s15, s15, s11
    s_lshl_b32 s11, s6, 5
    s_add_u32 s15, s15, s11
    s_add_u32 s15, s15, s21
    s_lshl_b32 s25, s24, 14
    v_and_b32_e32 v1, 7, v0
    v_bfe_u32 v2, v0, 3, 2
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshl_add_u32 v4, v3, 3, v1
    v_add_nc_u32_e32 v4, s14, v4
    v_mul_lo_u32 v5, v4, s20
    v_lshl_add_u32 v6, v2, 3, s15
    v_add_nc_u32_e32 v5, v5, v6
    v_lshlrev_b32_e32 v5, 2, v5
    v_lshlrev_b32_e32 v7, 5, v0
    v_add_nc_u32_e32 v8, s25, v7
    global_load_b128 v[32:35], v7, s[12:13] glc dlc
    global_load_b128 v[36:39], v7, s[12:13] offset:16 glc dlc
    global_load_b128 v[64:67], v8, s[12:13] glc dlc
    global_load_b128 v[68:71], v8, s[12:13] offset:16 glc dlc
    s_waitcnt vmcnt(0)
