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
v_mul_f32_e32 v160, 0xbfb8aa3b, v32
v_mul_f32_e32 v166, 0xbfb8aa3b, v33
v_mul_f32_e32 v172, 0xbfb8aa3b, v34
v_mul_f32_e32 v178, 0xbfb8aa3b, v35
v_mul_f32_e32 v184, 0xbfb8aa3b, v36
v_mul_f32_e32 v190, 0xbfb8aa3b, v37
v_mul_f32_e32 v196, 0xbfb8aa3b, v38
v_mul_f32_e32 v202, 0xbfb8aa3b, v39
v_fma_f32 v161, 0xbfb8aa3b, v32, -v160
v_fma_f32 v167, 0xbfb8aa3b, v33, -v166
v_fma_f32 v173, 0xbfb8aa3b, v34, -v172
v_fma_f32 v179, 0xbfb8aa3b, v35, -v178
v_fma_f32 v185, 0xbfb8aa3b, v36, -v184
v_fma_f32 v191, 0xbfb8aa3b, v37, -v190
v_fma_f32 v197, 0xbfb8aa3b, v38, -v196
v_fma_f32 v203, 0xbfb8aa3b, v39, -v202
v_fmac_f32_e32 v161, 0xb2a5705f, v32
v_fmac_f32_e32 v167, 0xb2a5705f, v33
v_fmac_f32_e32 v173, 0xb2a5705f, v34
v_fmac_f32_e32 v179, 0xb2a5705f, v35
v_fmac_f32_e32 v185, 0xb2a5705f, v36
v_fmac_f32_e32 v191, 0xb2a5705f, v37
v_fmac_f32_e32 v197, 0xb2a5705f, v38
v_fmac_f32_e32 v203, 0xb2a5705f, v39
v_rndne_f32_e32 v162, v160
v_rndne_f32_e32 v168, v166
v_rndne_f32_e32 v174, v172
v_rndne_f32_e32 v180, v178
v_rndne_f32_e32 v186, v184
v_rndne_f32_e32 v192, v190
v_rndne_f32_e32 v198, v196
v_rndne_f32_e32 v204, v202
v_sub_f32_e32 v160, v160, v162
v_sub_f32_e32 v166, v166, v168
v_sub_f32_e32 v172, v172, v174
v_sub_f32_e32 v178, v178, v180
v_sub_f32_e32 v184, v184, v186
v_sub_f32_e32 v190, v190, v192
v_sub_f32_e32 v196, v196, v198
v_sub_f32_e32 v202, v202, v204
v_add_f32_e32 v160, v160, v161
v_add_f32_e32 v166, v166, v167
v_add_f32_e32 v172, v172, v173
v_add_f32_e32 v178, v178, v179
v_add_f32_e32 v184, v184, v185
v_add_f32_e32 v190, v190, v191
v_add_f32_e32 v196, v196, v197
v_add_f32_e32 v202, v202, v203
v_exp_f32_e32 v160, v160
v_exp_f32_e32 v166, v166
v_exp_f32_e32 v172, v172
v_exp_f32_e32 v178, v178
v_exp_f32_e32 v184, v184
v_exp_f32_e32 v190, v190
v_exp_f32_e32 v196, v196
v_exp_f32_e32 v202, v202
v_cvt_i32_f32_e32 v162, v162
v_cvt_i32_f32_e32 v168, v168
v_cvt_i32_f32_e32 v174, v174
v_cvt_i32_f32_e32 v180, v180
v_cvt_i32_f32_e32 v186, v186
v_cvt_i32_f32_e32 v192, v192
v_cvt_i32_f32_e32 v198, v198
v_cvt_i32_f32_e32 v204, v204
v_ldexp_f32 v160, v160, v162
v_ldexp_f32 v166, v166, v168
v_ldexp_f32 v172, v172, v174
v_ldexp_f32 v178, v178, v180
v_ldexp_f32 v184, v184, v186
v_ldexp_f32 v190, v190, v192
v_ldexp_f32 v196, v196, v198
v_ldexp_f32 v202, v202, v204
v_cmp_nlt_f32_e64 s48, 0x42ce8ed0, v32
v_cmp_nlt_f32_e64 s50, 0x42ce8ed0, v33
v_cmp_nlt_f32_e64 s52, 0x42ce8ed0, v34
v_cmp_nlt_f32_e64 s54, 0x42ce8ed0, v35
v_cmp_nlt_f32_e64 s56, 0x42ce8ed0, v36
v_cmp_nlt_f32_e64 s58, 0x42ce8ed0, v37
v_cmp_nlt_f32_e64 s60, 0x42ce8ed0, v38
v_cmp_nlt_f32_e64 s62, 0x42ce8ed0, v39
v_cndmask_b32_e64 v160, 0, v160, s48
v_cndmask_b32_e64 v166, 0, v166, s50
v_cndmask_b32_e64 v172, 0, v172, s52
v_cndmask_b32_e64 v178, 0, v178, s54
v_cndmask_b32_e64 v184, 0, v184, s56
v_cndmask_b32_e64 v190, 0, v190, s58
v_cndmask_b32_e64 v196, 0, v196, s60
v_cndmask_b32_e64 v202, 0, v202, s62
v_cmp_ngt_f32_e64 s64, 0xc2b17218, v32
v_cmp_ngt_f32_e64 s66, 0xc2b17218, v33
v_cmp_ngt_f32_e64 s68, 0xc2b17218, v34
v_cmp_ngt_f32_e64 s70, 0xc2b17218, v35
v_cmp_ngt_f32_e64 s72, 0xc2b17218, v36
v_cmp_ngt_f32_e64 s74, 0xc2b17218, v37
v_cmp_ngt_f32_e64 s76, 0xc2b17218, v38
v_cmp_ngt_f32_e64 s78, 0xc2b17218, v39
v_cndmask_b32_e64 v160, 0x7f800000, v160, s64
v_cndmask_b32_e64 v166, 0x7f800000, v166, s66
v_cndmask_b32_e64 v172, 0x7f800000, v172, s68
v_cndmask_b32_e64 v178, 0x7f800000, v178, s70
v_cndmask_b32_e64 v184, 0x7f800000, v184, s72
v_cndmask_b32_e64 v190, 0x7f800000, v190, s74
v_cndmask_b32_e64 v196, 0x7f800000, v196, s76
v_cndmask_b32_e64 v202, 0x7f800000, v202, s78
v_add_f32_e32 v160, 1.0, v160
v_add_f32_e32 v166, 1.0, v166
v_add_f32_e32 v172, 1.0, v172
v_add_f32_e32 v178, 1.0, v178
v_add_f32_e32 v184, 1.0, v184
v_add_f32_e32 v190, 1.0, v190
v_add_f32_e32 v196, 1.0, v196
v_add_f32_e32 v202, 1.0, v202
v_div_scale_f32 v161, null, v160, v160, v32
v_div_scale_f32 v167, null, v166, v166, v33
v_div_scale_f32 v173, null, v172, v172, v34
v_div_scale_f32 v179, null, v178, v178, v35
v_div_scale_f32 v185, null, v184, v184, v36
v_div_scale_f32 v191, null, v190, v190, v37
v_div_scale_f32 v197, null, v196, v196, v38
v_div_scale_f32 v203, null, v202, v202, v39
v_div_scale_f32 v162, s80, v32, v160, v32
v_div_scale_f32 v168, s82, v33, v166, v33
v_div_scale_f32 v174, s84, v34, v172, v34
v_div_scale_f32 v180, s86, v35, v178, v35
v_div_scale_f32 v186, s88, v36, v184, v36
v_div_scale_f32 v192, s90, v37, v190, v37
v_div_scale_f32 v198, s92, v38, v196, v38
v_div_scale_f32 v204, s94, v39, v202, v39
v_rcp_f32_e32 v163, v161
v_rcp_f32_e32 v169, v167
v_rcp_f32_e32 v175, v173
v_rcp_f32_e32 v181, v179
v_rcp_f32_e32 v187, v185
v_rcp_f32_e32 v193, v191
v_rcp_f32_e32 v199, v197
v_rcp_f32_e32 v205, v203
v_fma_f32 v164, -v161, v163, 1.0
v_fma_f32 v170, -v167, v169, 1.0
v_fma_f32 v176, -v173, v175, 1.0
v_fma_f32 v182, -v179, v181, 1.0
v_fma_f32 v188, -v185, v187, 1.0
v_fma_f32 v194, -v191, v193, 1.0
v_fma_f32 v200, -v197, v199, 1.0
v_fma_f32 v206, -v203, v205, 1.0
v_fmac_f32_e32 v163, v164, v163
v_fmac_f32_e32 v169, v170, v169
v_fmac_f32_e32 v175, v176, v175
v_fmac_f32_e32 v181, v182, v181
v_fmac_f32_e32 v187, v188, v187
v_fmac_f32_e32 v193, v194, v193
v_fmac_f32_e32 v199, v200, v199
v_fmac_f32_e32 v205, v206, v205
v_mul_f32_e32 v164, v162, v163
v_mul_f32_e32 v170, v168, v169
v_mul_f32_e32 v176, v174, v175
v_mul_f32_e32 v182, v180, v181
v_mul_f32_e32 v188, v186, v187
v_mul_f32_e32 v194, v192, v193
v_mul_f32_e32 v200, v198, v199
v_mul_f32_e32 v206, v204, v205
v_fma_f32 v165, -v161, v164, v162
v_fma_f32 v171, -v167, v170, v168
v_fma_f32 v177, -v173, v176, v174
v_fma_f32 v183, -v179, v182, v180
v_fma_f32 v189, -v185, v188, v186
v_fma_f32 v195, -v191, v194, v192
v_fma_f32 v201, -v197, v200, v198
v_fma_f32 v207, -v203, v206, v204
v_fmac_f32_e32 v164, v165, v163
v_fmac_f32_e32 v170, v171, v169
v_fmac_f32_e32 v176, v177, v175
v_fmac_f32_e32 v182, v183, v181
v_fmac_f32_e32 v188, v189, v187
v_fmac_f32_e32 v194, v195, v193
v_fmac_f32_e32 v200, v201, v199
v_fmac_f32_e32 v206, v207, v205
v_fma_f32 v165, -v161, v164, v162
v_fma_f32 v171, -v167, v170, v168
v_fma_f32 v177, -v173, v176, v174
v_fma_f32 v183, -v179, v182, v180
v_fma_f32 v189, -v185, v188, v186
v_fma_f32 v195, -v191, v194, v192
v_fma_f32 v201, -v197, v200, v198
v_fma_f32 v207, -v203, v206, v204
s_mov_b32 vcc_lo, s80
v_div_fmas_f32 v165, v165, v163, v164
s_mov_b32 vcc_lo, s82
v_div_fmas_f32 v171, v171, v169, v170
s_mov_b32 vcc_lo, s84
v_div_fmas_f32 v177, v177, v175, v176
s_mov_b32 vcc_lo, s86
v_div_fmas_f32 v183, v183, v181, v182
s_mov_b32 vcc_lo, s88
v_div_fmas_f32 v189, v189, v187, v188
s_mov_b32 vcc_lo, s90
v_div_fmas_f32 v195, v195, v193, v194
s_mov_b32 vcc_lo, s92
v_div_fmas_f32 v201, v201, v199, v200
s_mov_b32 vcc_lo, s94
v_div_fmas_f32 v207, v207, v205, v206
v_div_fixup_f32 v165, v165, v160, v32
v_div_fixup_f32 v171, v171, v166, v33
v_div_fixup_f32 v177, v177, v172, v34
v_div_fixup_f32 v183, v183, v178, v35
v_div_fixup_f32 v189, v189, v184, v36
v_div_fixup_f32 v195, v195, v190, v37
v_div_fixup_f32 v201, v201, v196, v38
v_div_fixup_f32 v207, v207, v202, v39
v_mul_f32_e32 v32, v165, v64
v_mul_f32_e32 v33, v171, v65
v_mul_f32_e32 v34, v177, v66
v_mul_f32_e32 v35, v183, v67
v_mul_f32_e32 v36, v189, v68
v_mul_f32_e32 v37, v195, v69
v_mul_f32_e32 v38, v201, v70
v_mul_f32_e32 v39, v207, v71
    global_store_b128 v5, v[32:35], s[18:19]
    global_store_b128 v5, v[36:39], s[18:19] offset:16
    s_endpgm
.Lief_silu_end:
.size ief_silu, .Lief_silu_end-ief_silu
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel ief_silu
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_queue_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_user_sgpr_private_segment_size 0
    .amdhsa_wavefront_size32 1
    .amdhsa_uses_dynamic_stack 0
    .amdhsa_enable_private_segment 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 208
    .amdhsa_next_free_sgpr 96
    .amdhsa_reserve_vcc 1
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_fp16_overflow 0
    .amdhsa_workgroup_processor_mode 1
    .amdhsa_memory_ordered 1
    .amdhsa_forward_progress 1
    .amdhsa_dx10_clamp 1
    .amdhsa_ieee_mode 1
    .amdhsa_shared_vgpr_count 0
    .amdhsa_exception_fp_ieee_invalid_op 0
    .amdhsa_exception_fp_denorm_src 0
    .amdhsa_exception_fp_ieee_div_zero 0
    .amdhsa_exception_fp_ieee_overflow 0
    .amdhsa_exception_fp_ieee_underflow 0
    .amdhsa_exception_fp_ieee_inexact 0
    .amdhsa_exception_int_div_zero 0
    .amdhsa_inst_pref_size ((instprefsize(.Lief_silu_end-ief_silu)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space: global
        .name: c
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .name: ldy
        .offset: 16
        .size: 4
        .value_kind: by_value
      - .name: col0
        .offset: 20
        .size: 4
        .value_kind: by_value
      - .name: waves
        .offset: 24
        .size: 4
        .value_kind: by_value
      - .name: nw
        .offset: 28
        .size: 4
        .value_kind: by_value
      - .name: half
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .name: ief_silu
    .private_segment_fixed_size: 0
    .sgpr_count: 98
    .sgpr_spill_count: 0
    .symbol: ief_silu.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 208
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1151
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
