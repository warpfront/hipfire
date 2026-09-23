	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	qkv_full                ; -- Begin function qkv_full
	.globl	qkv_full
	.p2align	8
	.type	qkv_full,@function
qkv_full:                               ; @qkv_full
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[24:27], s[0:1], 0x48
	v_bfe_u32 v2, v0, 4, 1
	v_lshrrev_b32_e32 v1, 1, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v161, 3, v2
	s_wait_kmcnt 0x0
	s_add_co_i32 s30, s25, s24
	s_cmp_gt_i32 s27, 0xff
	s_cbranch_scc1 .LBB0_2
; %bb.1:
	v_lshlrev_b32_e32 v2, 3, v2
	s_mov_b32 s2, 0
	s_branch .LBB0_3
.LBB0_2:
	s_mov_b32 s2, -1
                                        ; implicit-def: $vgpr2
.LBB0_3:
	s_clause 0x1
	s_load_b512 s[8:23], s[0:1], 0x0
	s_load_b32 s31, s[0:1], 0x58
	v_and_b32_e32 v160, 15, v0
	v_and_b32_e32 v158, 0x60, v0
	v_and_b32_e32 v159, 64, v1
	s_lshl_b32 s33, ttmp7, 7
	s_add_co_i32 s34, s30, s26
	s_lshl_b32 s35, ttmp9, 7
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_mov_b32 s29, 0
	s_cbranch_vccnz .LBB0_20
; %bb.4:
	v_lshrrev_b32_e32 v2, 2, v0
	s_add_co_i32 s7, s34, -1
	s_ashr_i32 s2, s27, 31
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v4, s8 :: v_dual_mov_b32 v5, s9
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v3, s35, v2
	v_and_b32_e32 v7, 0x7f, v0
	v_add_nc_u32_e32 v9, s33, v2
	s_lshr_b32 s2, s2, 24
	s_delay_alu instid0(VALU_DEP_3)
	v_min_i32_e32 v6, s7, v3
	v_add_nc_u32_e32 v12, 64, v3
	v_cmp_gt_i32_e64 s3, s34, v3
	s_add_co_i32 s2, s27, s2
	v_or_b32_e32 v11, 64, v2
	v_cmp_gt_i32_e32 vcc_lo, s24, v6
	v_cmp_gt_i32_e64 s5, s30, v6
	v_or_b32_e32 v16, s35, v7
	v_mul_u32_u24_e32 v22, 0x48, v2
	v_mad_u32_u24 v28, 0x48, v2, 0
	v_cndmask_b32_e64 v10, s24, 0, vcc_lo
	v_cndmask_b32_e32 v14, s11, v5, vcc_lo
	v_dual_mov_b32 v2, v8 :: v_dual_add_nc_u32 v13, 64, v9
	s_ashr_i32 s36, s2, 8
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v10, s30, v10, s5
	s_mul_i32 s8, s36, 0x88
	v_min_i32_e32 v17, s7, v16
	v_cndmask_b32_e64 v14, s13, v14, s5
	s_add_co_i32 s28, s31, -1
	v_sub_nc_u32_e32 v3, v6, v10
	v_min_i32_e32 v6, s7, v12
	v_cndmask_b32_e32 v10, s10, v4, vcc_lo
	v_cmp_gt_i32_e64 s7, s24, v17
	v_mul_u32_u24_e32 v21, 0x48, v11
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v3, s8, v3
	v_cmp_gt_i32_e32 vcc_lo, s24, v6
	v_cmp_gt_i32_e64 s6, s30, v6
	v_cndmask_b32_e64 v10, s12, v10, s5
	v_add_nc_u32_e32 v19, s33, v1
	v_cmp_gt_i32_e64 s4, s31, v13
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v15, s24, 0, vcc_lo
	v_cndmask_b32_e32 v18, s11, v5, vcc_lo
	v_add_co_u32 v3, s5, v10, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v14, s5
	v_cndmask_b32_e64 v15, s30, v15, s6
	v_cndmask_b32_e32 v14, s10, v4, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s30, v17
	v_cndmask_b32_e64 v5, s11, v5, s7
	v_cndmask_b32_e64 v4, s10, v4, s7
	v_sub_nc_u32_e32 v6, v6, v15
	v_cndmask_b32_e64 v15, s24, 0, s7
	v_lshlrev_b32_e32 v23, 2, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, s13, v5, vcc_lo
	v_cmp_gt_i32_e64 s5, s34, v12
	v_mul_lo_u32 v6, s8, v6
	v_cndmask_b32_e32 v15, s30, v15, vcc_lo
	v_cndmask_b32_e64 v12, s12, v14, s6
	v_cndmask_b32_e64 v14, s13, v18, s6
	v_cndmask_b32_e32 v4, s12, v4, vcc_lo
	v_or_b32_e32 v1, v159, v160
	v_sub_nc_u32_e32 v15, v17, v15
	v_cmp_gt_i32_e64 s2, s31, v9
	v_add_co_u32 v6, s6, v12, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, 0, v14, s6
	v_mul_lo_u32 v14, s8, v15
	v_and_b32_e32 v15, 3, v0
	v_min_i32_e32 v9, s28, v9
	v_add_nc_u32_e32 v27, 0, v161
	v_mul_u32_u24_e32 v30, 0x48, v1
	v_lshlrev_b32_e32 v31, 2, v1
	v_lshlrev_b32_e32 v20, 4, v15
	v_cmp_gt_i32_e64 s7, s34, v16
	v_add_co_u32 v163, vcc_lo, v4, v14
	v_min_i32_e32 v4, s28, v13
	v_mov_b32_e32 v11, v8
	v_lshlrev_b32_e32 v13, 3, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v164, null, 0, v5, vcc_lo
	v_mad_co_u64_u32 v[152:153], null, v4, s27, v[20:21]
	v_and_b32_e32 v4, 0x6f, v0
	v_lshrrev_b32_e32 v5, 7, v0
	v_add_co_u32 v166, vcc_lo, v3, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v167, null, 0, v10, vcc_lo
	v_mad_u32_u24 v26, 0x48, v4, 0
	v_or_b32_e32 v4, v161, v158
	v_add_co_u32 v168, vcc_lo, v6, v13
	v_mad_co_u64_u32 v[153:154], null, v9, s27, v[20:21]
	v_lshl_add_u32 v25, v5, 2, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v32, 3, v4
	v_mov_b32_e32 v4, v8
	v_min_i32_e32 v18, s28, v19
	v_cmp_gt_i32_e64 s6, s31, v19
	v_mov_b32_e32 v9, v8
	v_dual_mov_b32 v10, v8 :: v_dual_add_nc_u32 v29, 0, v20
	v_dual_mov_b32 v1, v8 :: v_dual_lshlrev_b32 v24, 3, v7
	v_mov_b32_e32 v19, v8
	v_mul_lo_u32 v162, v18, s36
	v_lshlrev_b32_e32 v165, 4, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v169, null, 0, v12, vcc_lo
	v_dual_mov_b32 v3, v8 :: v_dual_mov_b32 v16, v8
	v_dual_mov_b32 v5, v8 :: v_dual_mov_b32 v18, v8
	v_dual_mov_b32 v6, v8 :: v_dual_mov_b32 v7, v8
	v_add_nc_u32_e32 v170, 0, v23
	v_mov_b32_e32 v12, v8
	v_dual_mov_b32 v14, v8 :: v_dual_mov_b32 v13, v8
	v_dual_mov_b32 v15, v8 :: v_dual_add_nc_u32 v172, 0, v31
	v_dual_mov_b32 v17, v8 :: v_dual_add_nc_u32 v178, v27, v30
	v_mov_b32_e32 v151, v11
	v_add3_u32 v175, v29, v22, 0x2400
	v_add3_u32 v176, v29, v21, 0x2400
	v_mov_b32_e32 v150, v10
	v_dual_mov_b32 v148, v8 :: v_dual_add_nc_u32 v171, v25, v24
	v_add_nc_u32_e32 v173, 0, v32
	v_add_nc_u32_e32 v174, v28, v20
	v_add_nc_u32_e32 v177, v26, v161
	v_mov_b32_e32 v27, v19
	v_dual_mov_b32 v35, v19 :: v_dual_mov_b32 v30, v14
	v_dual_mov_b32 v43, v19 :: v_dual_mov_b32 v38, v14
	v_dual_mov_b32 v51, v19 :: v_dual_mov_b32 v46, v14
	v_dual_mov_b32 v59, v19 :: v_dual_mov_b32 v54, v14
	v_mov_b32_e32 v67, v19
	v_mov_b32_e32 v75, v19
	v_dual_mov_b32 v83, v8 :: v_dual_mov_b32 v26, v18
	v_dual_mov_b32 v25, v17 :: v_dual_mov_b32 v22, v14
	v_dual_mov_b32 v24, v16 :: v_dual_mov_b32 v23, v15
	v_dual_mov_b32 v34, v18 :: v_dual_mov_b32 v21, v13
	v_mov_b32_e32 v32, v16
	v_dual_mov_b32 v20, v12 :: v_dual_mov_b32 v33, v17
	v_dual_mov_b32 v28, v12 :: v_dual_mov_b32 v31, v15
	v_dual_mov_b32 v42, v18 :: v_dual_mov_b32 v29, v13
	v_dual_mov_b32 v40, v16 :: v_dual_mov_b32 v41, v17
	v_dual_mov_b32 v36, v12 :: v_dual_mov_b32 v39, v15
	v_dual_mov_b32 v50, v18 :: v_dual_mov_b32 v37, v13
	v_dual_mov_b32 v48, v16 :: v_dual_mov_b32 v49, v17
	v_dual_mov_b32 v44, v12 :: v_dual_mov_b32 v47, v15
	v_dual_mov_b32 v58, v18 :: v_dual_mov_b32 v45, v13
	v_dual_mov_b32 v56, v16 :: v_dual_mov_b32 v57, v17
	v_dual_mov_b32 v52, v12 :: v_dual_mov_b32 v55, v15
	v_dual_mov_b32 v66, v18 :: v_dual_mov_b32 v53, v13
	v_dual_mov_b32 v64, v16 :: v_dual_mov_b32 v149, v9
	v_dual_mov_b32 v60, v12 :: v_dual_mov_b32 v65, v17
	v_dual_mov_b32 v72, v16 :: v_dual_mov_b32 v63, v15
	v_mov_b32_e32 v70, v14
	v_dual_mov_b32 v62, v14 :: v_dual_mov_b32 v61, v13
	v_mov_b32_e32 v68, v12
	v_dual_mov_b32 v74, v18 :: v_dual_mov_b32 v73, v17
	v_dual_mov_b32 v80, v5 :: v_dual_mov_b32 v71, v15
	v_dual_mov_b32 v78, v3 :: v_dual_mov_b32 v69, v13
	v_mov_b32_e32 v76, v1
	v_dual_mov_b32 v82, v7 :: v_dual_mov_b32 v81, v6
	v_mov_b32_e32 v79, v4
	v_mov_b32_e32 v77, v2
	s_mov_b32 s10, 0x4e4c4a48
	s_mov_b32 s11, 0x4040404
	s_mov_b32 s12, 0
                                        ; implicit-def: $vgpr84_vgpr85_vgpr86_vgpr87_vgpr88_vgpr89_vgpr90_vgpr91
                                        ; implicit-def: $vgpr92_vgpr93_vgpr94_vgpr95_vgpr96_vgpr97_vgpr98_vgpr99
                                        ; implicit-def: $vgpr100_vgpr101_vgpr102_vgpr103_vgpr104_vgpr105_vgpr106_vgpr107
                                        ; implicit-def: $vgpr108_vgpr109_vgpr110_vgpr111_vgpr112_vgpr113_vgpr114_vgpr115
                                        ; implicit-def: $vgpr116_vgpr117_vgpr118_vgpr119_vgpr120_vgpr121_vgpr122_vgpr123
                                        ; implicit-def: $vgpr124_vgpr125_vgpr126_vgpr127_vgpr128_vgpr129_vgpr130_vgpr131
                                        ; implicit-def: $vgpr132_vgpr133_vgpr134_vgpr135_vgpr136_vgpr137_vgpr138_vgpr139
                                        ; implicit-def: $vgpr140_vgpr141_vgpr142_vgpr143_vgpr144_vgpr145_vgpr146_vgpr147
	s_branch .LBB0_6
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_add_co_i32 s12, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, s36
	s_cbranch_scc1 .LBB0_21
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_8 Depth 2
                                        ;       Child Loop BB0_10 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	v_add_lshl_u32 v11, s12, v162, 1
	s_lshl_b32 s28, s12, 8
	s_mul_i32 s13, s12, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[14:15], s[28:29]
	s_mov_b32 s27, -1
	s_mov_b32 s37, 0
	s_branch .LBB0_8
.LBB0_7:                                ;   in Loop: Header=BB0_8 Depth=2
	s_mov_b32 s37, 1
	s_and_b32 vcc_lo, exec_lo, s27
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_5
.LBB0_8:                                ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_10 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s38, s37, 2
	s_lshl_b32 s28, s37, 7
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s38, s38, s13
	v_or_b32_e32 v7, s37, v11
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v163, s38
	s_add_nc_u64 s[38:39], s[8:9], s[28:29]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v164, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, s28, s38, v153
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s39, 0, s28
	s_lshl_b32 s28, s37, 6
	v_lshlrev_b64_e32 v[183:184], 2, v[7:8]
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s28, s13, s28
	v_add_co_u32 v9, s37, s38, v152
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v154, vcc_lo, v166, s28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v155, null, 0, v167, vcc_lo
	v_add_co_u32 v156, vcc_lo, v168, s28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v157, null, 0, v169, vcc_lo
	v_add_co_u32 v183, vcc_lo, s16, v183
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s39, 0, s37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v184, null, s17, v184, vcc_lo
	global_load_b32 v189, v[1:2], off
	s_clause 0x1
	global_load_b128 v[1:4], v[5:6], off
	global_load_b128 v[179:182], v[9:10], off
	global_load_b64 v[185:186], v[154:155], off offset:8
	global_load_b64 v[187:188], v[156:157], off offset:8
	global_load_b32 v183, v[183:184], off
	v_add_nc_u32_e32 v7, 0x1200, v174
	s_mov_b32 s38, -1
	s_xor_b32 s27, s27, -1
	s_mov_b32 s37, 0
	s_wait_loadcnt 0x5
	v_lshrrev_b32_e32 v184, v165, v189
	s_wait_loadcnt 0x4
	v_cndmask_b32_e64 v2, 0, v2, s2
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v185, 0, v185, s3
	v_cndmask_b32_e64 v186, 0, v186, s3
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v187, 0, v187, s5
	v_cndmask_b32_e64 v188, 0, v188, s5
	v_cvt_f32_f16_e64 v184, v184.l
	v_and_b32_e32 v189, 0xf0f0f0f, v185
	v_lshrrev_b32_e32 v185, 4, v185
	v_and_b32_e32 v190, 0xf0f0f0f, v186
	v_lshrrev_b32_e32 v186, 4, v186
	v_and_b32_e32 v191, 0xf0f0f0f, v187
	v_lshrrev_b32_e32 v187, 4, v187
	v_and_b32_e32 v192, 0xf0f0f0f, v188
	v_lshrrev_b32_e32 v188, 4, v188
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v183, 0, v183, s6
	v_cndmask_b32_e64 v193, 0, v184, s7
	v_and_b32_e32 v184, 0xf0f0f0f, v185
	v_and_b32_e32 v185, 0xf0f0f0f, v186
	v_and_b32_e32 v186, 0xf0f0f0f, v187
	v_and_b32_e32 v187, 0xf0f0f0f, v188
	ds_store_b32 v170, v183 offset:18432
	v_perm_b32 v183, v184, v189, 0x5010400
	v_perm_b32 v184, v184, v189, 0x7030602
	v_perm_b32 v188, v185, v190, 0x5010400
	v_perm_b32 v185, v185, v190, 0x7030602
	v_perm_b32 v189, v186, v191, 0x5010400
	v_perm_b32 v186, v186, v191, 0x7030602
	v_perm_b32 v190, v187, v192, 0x5010400
	v_perm_b32 v187, v187, v192, 0x7030602
	v_and_b32_e32 v191, 0x7070707, v183
	v_lshrrev_b32_e32 v183, 1, v183
	v_and_b32_e32 v192, 0x7070707, v184
	v_lshrrev_b32_e32 v184, 1, v184
	v_and_b32_e32 v194, 0x7070707, v188
	v_lshrrev_b32_e32 v188, 1, v188
	v_and_b32_e32 v195, 0x7070707, v185
	v_lshrrev_b32_e32 v185, 1, v185
	v_and_b32_e32 v196, 0x7070707, v189
	v_lshrrev_b32_e32 v189, 1, v189
	v_and_b32_e32 v197, 0x7070707, v186
	v_lshrrev_b32_e32 v186, 1, v186
	v_and_b32_e32 v198, 0x7070707, v190
	v_lshrrev_b32_e32 v190, 1, v190
	v_and_b32_e32 v199, 0x7070707, v187
	v_lshrrev_b32_e32 v187, 1, v187
	v_perm_b32 v200, s10, 0x44403800, v191
	v_or_b32_e32 v191, 0x50505050, v191
	v_and_or_b32 v183, v183, s11, 0x3020100
	v_perm_b32 v201, s10, 0x44403800, v192
	v_or_b32_e32 v192, 0x50505050, v192
	v_and_or_b32 v184, v184, s11, 0x3020100
	v_perm_b32 v202, s10, 0x44403800, v194
	v_or_b32_e32 v194, 0x50505050, v194
	v_and_or_b32 v188, v188, s11, 0x3020100
	v_perm_b32 v203, s10, 0x44403800, v195
	v_or_b32_e32 v195, 0x50505050, v195
	v_and_or_b32 v204, v185, s11, 0x3020100
	v_perm_b32 v205, s10, 0x44403800, v196
	v_or_b32_e32 v196, 0x50505050, v196
	v_and_or_b32 v189, v189, s11, 0x3020100
	v_perm_b32 v206, s10, 0x44403800, v197
	v_or_b32_e32 v197, 0x50505050, v197
	v_and_or_b32 v207, v186, s11, 0x3020100
	v_perm_b32 v208, s10, 0x44403800, v198
	v_or_b32_e32 v198, 0x50505050, v198
	v_and_or_b32 v190, v190, s11, 0x3020100
	v_perm_b32 v209, s10, 0x44403800, v199
	v_or_b32_e32 v199, 0x50505050, v199
	v_and_or_b32 v210, v187, s11, 0x3020100
	v_cndmask_b32_e64 v1, 0, v1, s2
	v_cndmask_b32_e64 v4, 0, v4, s2
	v_cndmask_b32_e64 v3, 0, v3, s2
	v_cndmask_b32_e64 v180, 0, v180, s4
	v_cndmask_b32_e64 v179, 0, v179, s4
	v_cndmask_b32_e64 v182, 0, v182, s4
	v_cndmask_b32_e64 v181, 0, v181, s4
	v_perm_b32 v183, v191, v200, v183
	v_perm_b32 v184, v192, v201, v184
	v_perm_b32 v185, v194, v202, v188
	v_perm_b32 v186, v195, v203, v204
	v_perm_b32 v187, v196, v205, v189
	v_perm_b32 v188, v197, v206, v207
	v_perm_b32 v189, v198, v208, v190
	v_perm_b32 v190, v199, v209, v210
	ds_store_b32 v171, v193 offset:18944
	ds_store_2addr_b64 v174, v[1:2], v[3:4] offset1:1
	ds_store_2addr_b64 v7, v[179:180], v[181:182] offset1:1
	ds_store_2addr_b64 v175, v[183:184], v[185:186] offset1:1
	ds_store_2addr_b64 v176, v[187:188], v[189:190] offset1:1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_10
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=3
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s37, -1
	s_mov_b32 s38, 0
	s_and_b32 vcc_lo, exec_lo, s28
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_7
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_8 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s28, s38, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=3
	s_clause 0x1
	global_load_b128 v[1:4], v[5:6], off offset:64
	global_load_b128 v[80:83], v[9:10], off offset:64
	global_load_b64 v[148:149], v[154:155], off offset:40
	global_load_b64 v[150:151], v[156:157], off offset:40
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v77, 0, v2, s2
	v_cndmask_b32_e64 v76, 0, v1, s2
	v_cndmask_b32_e64 v79, 0, v4, s2
	v_cndmask_b32_e64 v78, 0, v3, s2
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v81, 0, v81, s4
	v_cndmask_b32_e64 v80, 0, v80, s4
	v_cndmask_b32_e64 v83, 0, v83, s4
	v_cndmask_b32_e64 v82, 0, v82, s4
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v148, 0, v148, s3
	v_cndmask_b32_e64 v149, 0, v149, s3
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v150, 0, v150, s5
	v_cndmask_b32_e64 v151, 0, v151, s5
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=3
	v_add_nc_u32_e32 v180, 0x2400, v177
	v_add_nc_u32_e32 v179, 0x800, v178
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_mov_b32 s39, -1
	ds_load_2addr_b64 v[1:4], v180 offset1:144
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_16
; %bb.13:                               ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_17
.LBB0_14:                               ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s38
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_18
.LBB0_15:                               ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s37
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_9
	s_branch .LBB0_19
.LBB0_16:                               ;   in Loop: Header=BB0_10 Depth=3
	ds_load_2addr_b64 v[181:184], v178 offset1:144
	ds_load_2addr_b64 v[185:188], v179 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[181:182], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[181:182], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[183:184], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[183:184], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[185:186], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[185:186], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[187:188], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[187:188], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[181:184], v180 offset0:2 offset1:146
	ds_load_2addr_b64 v[185:188], v178 offset0:2 offset1:146
	ds_load_2addr_b64 v[189:192], v179 offset0:34 offset1:178
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[181:182], v[185:186], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[183:184], v[185:186], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[181:182], v[187:188], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[183:184], v[187:188], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[181:182], v[189:190], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[183:184], v[189:190], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[181:182], v[191:192], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[183:184], v[191:192], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[181:184], v180 offset0:4 offset1:148
	ds_load_2addr_b64 v[185:188], v178 offset0:4 offset1:148
	ds_load_2addr_b64 v[189:192], v179 offset0:36 offset1:180
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[181:182], v[185:186], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[183:184], v[185:186], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[181:182], v[187:188], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[183:184], v[187:188], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[181:182], v[189:190], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[183:184], v[189:190], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[181:182], v[191:192], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[183:184], v[191:192], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[181:184], v180 offset0:6 offset1:150
	ds_load_2addr_b64 v[185:188], v178 offset0:6 offset1:150
	ds_load_2addr_b64 v[189:192], v179 offset0:38 offset1:182
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[181:182], v[185:186], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[183:184], v[185:186], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[181:182], v[187:188], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[183:184], v[187:188], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[181:182], v[189:190], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[183:184], v[189:190], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[181:182], v[191:192], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[183:184], v[191:192], v[84:91]
	; sched_barrier mask(0x00000000)
	s_cbranch_execnz .LBB0_14
.LBB0_17:                               ;   in Loop: Header=BB0_10 Depth=3
	ds_load_2addr_b64 v[84:87], v178 offset1:144
	ds_load_2addr_b64 v[181:184], v179 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_nop
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[84:85], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[84:85], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[86:87], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[86:87], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[181:182], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[181:182], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[183:184], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[183:184], 0
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[1:4], v180 offset0:2 offset1:146
	ds_load_2addr_b64 v[181:184], v178 offset0:2 offset1:146
	ds_load_2addr_b64 v[185:188], v179 offset0:34 offset1:178
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[181:182], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[181:182], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[183:184], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[183:184], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[185:186], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[185:186], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[187:188], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[187:188], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[1:4], v180 offset0:4 offset1:148
	ds_load_2addr_b64 v[181:184], v178 offset0:4 offset1:148
	ds_load_2addr_b64 v[185:188], v179 offset0:36 offset1:180
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[181:182], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[181:182], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[183:184], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[183:184], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[185:186], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[185:186], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[187:188], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[187:188], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[1:4], v180 offset0:6 offset1:150
	ds_load_2addr_b64 v[180:183], v178 offset0:6 offset1:150
	ds_load_2addr_b64 v[184:187], v179 offset0:38 offset1:182
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[180:181], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[180:181], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[182:183], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[182:183], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[184:185], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[184:185], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[186:187], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[186:187], v[84:91]
	; sched_barrier mask(0x00000000)
	s_and_not1_b32 vcc_lo, exec_lo, s38
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_15
.LBB0_18:                               ;   in Loop: Header=BB0_10 Depth=3
	s_wait_dscnt 0x0
	v_lshrrev_b32_e32 v1, 4, v148
	v_lshrrev_b32_e32 v2, 4, v149
	v_and_b32_e32 v3, 0xf0f0f0f, v148
	v_and_b32_e32 v4, 0xf0f0f0f, v149
	s_wait_loadcnt 0x0
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	v_and_b32_e32 v2, 0xf0f0f0f, v2
	s_barrier_signal -1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v179, v1, v3, 0x5010400
	v_perm_b32 v1, v1, v3, 0x7030602
	v_perm_b32 v3, v2, v4, 0x5010400
	v_perm_b32 v4, v2, v4, 0x7030602
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v2, 0x7070707, v179
	v_lshrrev_b32_e32 v179, 1, v179
	v_and_b32_e32 v180, 0x7070707, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_and_b32_e32 v181, 0x7070707, v3
	v_perm_b32 v182, s10, 0x44403800, v2
	v_or_b32_e32 v2, 0x50505050, v2
	v_and_or_b32 v179, v179, s11, 0x3020100
	v_lshrrev_b32_e32 v3, 1, v3
	v_perm_b32 v183, s10, 0x44403800, v180
	v_or_b32_e32 v180, 0x50505050, v180
	v_and_or_b32 v184, v1, s11, 0x3020100
	v_perm_b32 v1, v2, v182, v179
	v_lshrrev_b32_e32 v2, 4, v150
	v_perm_b32 v179, s10, 0x44403800, v181
	v_or_b32_e32 v181, 0x50505050, v181
	v_and_or_b32 v3, v3, s11, 0x3020100
	v_and_b32_e32 v182, 0xf0f0f0f, v150
	v_and_b32_e32 v185, 0xf0f0f0f, v2
	v_perm_b32 v2, v180, v183, v184
	v_lshrrev_b32_e32 v183, 4, v151
	v_perm_b32 v3, v181, v179, v3
	v_and_b32_e32 v186, 0x7070707, v4
	v_perm_b32 v179, v185, v182, 0x5010400
	v_perm_b32 v182, v185, v182, 0x7030602
	v_and_b32_e32 v185, 0xf0f0f0f, v151
	v_and_b32_e32 v183, 0xf0f0f0f, v183
	v_lshrrev_b32_e32 v4, 1, v4
	v_and_b32_e32 v184, 0x7070707, v179
	v_lshrrev_b32_e32 v179, 1, v179
	v_and_b32_e32 v187, 0x7070707, v182
	v_perm_b32 v188, v183, v185, 0x5010400
	v_perm_b32 v183, v183, v185, 0x7030602
	v_lshrrev_b32_e32 v182, 1, v182
	v_perm_b32 v180, s10, 0x44403800, v186
	v_or_b32_e32 v181, 0x50505050, v186
	v_and_b32_e32 v185, 0x7070707, v188
	v_lshrrev_b32_e32 v188, 1, v188
	v_and_b32_e32 v190, 0x7070707, v183
	v_lshrrev_b32_e32 v183, 1, v183
	v_and_or_b32 v4, v4, s11, 0x3020100
	v_perm_b32 v186, s10, 0x44403800, v184
	v_or_b32_e32 v184, 0x50505050, v184
	v_and_or_b32 v179, v179, s11, 0x3020100
	v_perm_b32 v189, s10, 0x44403800, v187
	v_or_b32_e32 v187, 0x50505050, v187
	v_and_or_b32 v182, v182, s11, 0x3020100
	v_perm_b32 v191, s10, 0x44403800, v185
	v_or_b32_e32 v185, 0x50505050, v185
	v_and_or_b32 v188, v188, s11, 0x3020100
	v_perm_b32 v192, s10, 0x44403800, v190
	v_or_b32_e32 v190, 0x50505050, v190
	v_and_or_b32 v183, v183, s11, 0x3020100
	v_perm_b32 v4, v181, v180, v4
	v_perm_b32 v179, v184, v186, v179
	v_perm_b32 v180, v187, v189, v182
	v_perm_b32 v181, v185, v191, v188
	v_perm_b32 v182, v190, v192, v183
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v174, v[76:77], v[78:79] offset1:1
	ds_store_2addr_b64 v7, v[80:81], v[82:83] offset1:1
	ds_store_2addr_b64 v175, v[1:2], v[3:4] offset1:1
	ds_store_2addr_b64 v176, v[179:180], v[181:182] offset1:1
	s_and_not1_b32 vcc_lo, exec_lo, s37
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_9
.LBB0_19:                               ;   in Loop: Header=BB0_10 Depth=3
	v_add_nc_u32_e32 v179, 0x4a10, v173
	v_add_nc_u32_e32 v183, 0x4800, v172
	ds_load_2addr_b32 v[179:180], v179 offset1:1
	ds_load_2addr_b32 v[181:182], v183 offset1:16
	ds_load_2addr_b32 v[183:184], v183 offset0:32 offset1:48
	s_wait_dscnt 0x3
	v_add_nc_u32_e32 v1, 0x4a00, v173
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	v_add_nc_u32_e32 v3, 0x4a08, v173
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	s_wait_dscnt 0x4
	v_fma_f32 v54, v179, v126, v54
	v_add_nc_u32_e32 v185, 0x4a18, v173
	v_fma_f32 v38, v179, v110, v38
	v_add_nc_u32_e32 v187, 0x4a20, v173
	v_fma_f32 v70, v179, v142, v70
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v54, v180, v182
	v_fma_f32 v22, v179, v94, v22
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v38, v180, v183
	ds_load_2addr_b32 v[185:186], v185 offset1:1
	ds_load_2addr_b32 v[187:188], v187 offset1:1
	s_wait_dscnt 0x3
	v_fma_f32 v68, v1, v140, v68
	v_fma_f32 v52, v1, v124, v52
	v_fma_f32 v36, v1, v108, v36
	v_fma_f32 v20, v1, v92, v20
	v_dual_fmac_f32 v22, v180, v184 :: v_dual_add_nc_u32 v1, 0x4a28, v173
	s_wait_dscnt 0x2
	v_fma_f32 v69, v3, v141, v69
	v_fma_f32 v53, v3, v125, v53
	v_fma_f32 v37, v3, v109, v37
	v_fma_f32 v21, v3, v93, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v68, v2, v181 :: v_dual_fmac_f32 v69, v4, v181
	v_fmac_f32_e32 v53, v4, v182
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v37, v4, v183
	v_dual_fmac_f32 v21, v4, v184 :: v_dual_fmac_f32 v52, v2, v182
	v_fmac_f32_e32 v36, v2, v183
	v_fmac_f32_e32 v20, v2, v184
	s_wait_dscnt 0x1
	v_fma_f32 v71, v185, v143, v71
	v_fma_f32 v55, v185, v127, v55
	v_fma_f32 v39, v185, v111, v39
	v_fma_f32 v23, v185, v95, v23
	s_wait_dscnt 0x0
	v_fma_f32 v72, v187, v144, v72
	v_fma_f32 v56, v187, v128, v56
	v_add_nc_u32_e32 v3, 0x4a30, v173
	v_fma_f32 v40, v187, v112, v40
	v_fma_f32 v24, v187, v96, v24
	v_add_nc_u32_e32 v185, 0x4a90, v173
	v_add_nc_u32_e32 v179, 0x4a38, v173
	v_add_nc_u32_e32 v187, 0x4a98, v173
	v_dual_fmac_f32 v70, v180, v181 :: v_dual_fmac_f32 v71, v186, v181
	v_dual_fmac_f32 v55, v186, v182 :: v_dual_fmac_f32 v72, v188, v181
	v_dual_fmac_f32 v39, v186, v183 :: v_dual_fmac_f32 v24, v188, v184
	v_dual_fmac_f32 v23, v186, v184 :: v_dual_fmac_f32 v56, v188, v182
	v_fmac_f32_e32 v40, v188, v183
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	ds_load_2addr_b32 v[179:180], v179 offset1:1
	ds_load_2addr_b32 v[185:186], v185 offset1:1
	ds_load_2addr_b32 v[187:188], v187 offset1:1
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	s_wait_dscnt 0x4
	v_fma_f32 v74, v3, v146, v74
	v_fma_f32 v58, v3, v130, v58
	v_fma_f32 v42, v3, v114, v42
	v_fma_f32 v26, v3, v98, v26
	s_wait_dscnt 0x2
	v_fma_f32 v46, v185, v118, v46
	v_add_nc_u32_e32 v3, 0x4a88, v173
	v_fmac_f32_e32 v74, v4, v181
	v_fmac_f32_e32 v58, v4, v182
	v_fmac_f32_e32 v42, v4, v183
	v_fmac_f32_e32 v26, v4, v184
	v_fmac_f32_e32 v46, v186, v182
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	s_wait_dscnt 0x1
	v_fma_f32 v73, v1, v145, v73
	v_fma_f32 v57, v1, v129, v57
	v_fma_f32 v41, v1, v113, v41
	v_fma_f32 v25, v1, v97, v25
	v_add_nc_u32_e32 v1, 0x4a80, v173
	v_fmac_f32_e32 v73, v2, v181
	v_fmac_f32_e32 v57, v2, v182
	v_fmac_f32_e32 v41, v2, v183
	v_fmac_f32_e32 v25, v2, v184
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	v_fmac_f32_e32 v27, v179, v99
	v_fma_f32 v62, v185, v134, v62
	v_fma_f32 v30, v185, v102, v30
	v_fma_f32 v14, v185, v86, v14
	v_fmac_f32_e32 v75, v179, v147
	v_add_nc_u32_e32 v185, 0x4ab8, v173
	v_dual_fmac_f32 v27, v180, v184 :: v_dual_fmac_f32 v62, v186, v181
	v_fmac_f32_e32 v30, v186, v183
	s_wait_dscnt 0x1
	v_fma_f32 v61, v3, v133, v61
	v_fma_f32 v45, v3, v117, v45
	v_fma_f32 v29, v3, v101, v29
	v_fma_f32 v13, v3, v85, v13
	v_add_nc_u32_e32 v3, 0x4aa8, v173
	v_fmac_f32_e32 v61, v4, v181
	v_fmac_f32_e32 v45, v4, v182
	v_fmac_f32_e32 v29, v4, v183
	v_fmac_f32_e32 v13, v4, v184
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	s_wait_dscnt 0x1
	v_fma_f32 v60, v1, v132, v60
	v_fma_f32 v44, v1, v116, v44
	v_fma_f32 v28, v1, v100, v28
	v_fma_f32 v12, v1, v84, v12
	v_add_nc_u32_e32 v1, 0x4aa0, v173
	v_fmac_f32_e32 v60, v2, v181
	v_fmac_f32_e32 v44, v2, v182
	v_fmac_f32_e32 v28, v2, v183
	v_fmac_f32_e32 v12, v2, v184
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	v_dual_fmac_f32 v14, v186, v184 :: v_dual_fmac_f32 v75, v180, v181
	v_fma_f32 v63, v187, v135, v63
	v_fma_f32 v47, v187, v119, v47
	v_fma_f32 v31, v187, v103, v31
	v_fma_f32 v15, v187, v87, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v63, v188, v181
	v_fmac_f32_e32 v47, v188, v182
	s_wait_dscnt 0x1
	v_fma_f32 v65, v3, v137, v65
	v_fma_f32 v49, v3, v121, v49
	v_fma_f32 v33, v3, v105, v33
	v_fma_f32 v17, v3, v89, v17
	v_fmac_f32_e32 v31, v188, v183
	v_fmac_f32_e32 v15, v188, v184
	v_fmac_f32_e32 v65, v4, v181
	v_fmac_f32_e32 v49, v4, v182
	v_fmac_f32_e32 v33, v4, v183
	v_fmac_f32_e32 v17, v4, v184
	s_wait_dscnt 0x0
	v_fma_f32 v48, v1, v120, v48
	v_fmac_f32_e32 v59, v179, v131
	v_fma_f32 v16, v1, v88, v16
	v_fmac_f32_e32 v43, v179, v115
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v48, v2, v182 :: v_dual_add_nc_u32 v179, 0x4ab0, v173
	v_dual_fmac_f32 v59, v180, v182 :: v_dual_fmac_f32 v16, v2, v184
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v43, v180, v183
	ds_load_2addr_b32 v[179:180], v179 offset1:1
	ds_load_2addr_b32 v[185:186], v185 offset1:1
	v_fma_f32 v64, v1, v136, v64
	v_fma_f32 v32, v1, v104, v32
	v_fmac_f32_e32 v64, v2, v181
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v32, v2, v183
	s_wait_dscnt 0x1
	v_fma_f32 v66, v179, v138, v66
	v_fma_f32 v50, v179, v122, v50
	v_fma_f32 v34, v179, v106, v34
	v_fma_f32 v18, v179, v90, v18
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v67, v185, v139
	v_fmac_f32_e32 v51, v185, v123
	v_fmac_f32_e32 v35, v185, v107
	v_dual_fmac_f32 v19, v185, v91 :: v_dual_fmac_f32 v66, v180, v181
	v_fmac_f32_e32 v50, v180, v182
	v_fmac_f32_e32 v34, v180, v183
	v_dual_fmac_f32 v18, v180, v184 :: v_dual_fmac_f32 v67, v186, v181
	v_fmac_f32_e32 v51, v186, v182
	v_fmac_f32_e32 v35, v186, v183
	v_fmac_f32_e32 v19, v186, v184
	s_branch .LBB0_9
.LBB0_20:
	v_mov_b32_e32 v68, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v69, v68 :: v_dual_mov_b32 v70, v68
	v_dual_mov_b32 v71, v68 :: v_dual_mov_b32 v72, v68
	v_dual_mov_b32 v73, v68 :: v_dual_mov_b32 v74, v68
	v_mov_b32_e32 v75, v68
	v_dual_mov_b32 v60, v68 :: v_dual_mov_b32 v61, v69
	v_dual_mov_b32 v52, v68 :: v_dual_mov_b32 v53, v69
	v_dual_mov_b32 v44, v68 :: v_dual_mov_b32 v45, v69
	v_dual_mov_b32 v36, v68 :: v_dual_mov_b32 v37, v69
	v_dual_mov_b32 v28, v68 :: v_dual_mov_b32 v29, v69
	v_dual_mov_b32 v20, v68 :: v_dual_mov_b32 v21, v69
	v_dual_mov_b32 v12, v68 :: v_dual_mov_b32 v13, v69
	v_dual_mov_b32 v62, v70 :: v_dual_mov_b32 v63, v71
	v_dual_mov_b32 v64, v72 :: v_dual_mov_b32 v65, v73
	v_dual_mov_b32 v66, v74 :: v_dual_mov_b32 v67, v75
	v_dual_mov_b32 v54, v70 :: v_dual_mov_b32 v55, v71
	v_dual_mov_b32 v56, v72 :: v_dual_mov_b32 v57, v73
	v_dual_mov_b32 v58, v74 :: v_dual_mov_b32 v59, v75
	v_dual_mov_b32 v46, v70 :: v_dual_mov_b32 v47, v71
	v_dual_mov_b32 v48, v72 :: v_dual_mov_b32 v49, v73
	v_dual_mov_b32 v50, v74 :: v_dual_mov_b32 v51, v75
	v_dual_mov_b32 v38, v70 :: v_dual_mov_b32 v39, v71
	v_dual_mov_b32 v40, v72 :: v_dual_mov_b32 v41, v73
	v_dual_mov_b32 v42, v74 :: v_dual_mov_b32 v43, v75
	v_dual_mov_b32 v30, v70 :: v_dual_mov_b32 v31, v71
	v_dual_mov_b32 v32, v72 :: v_dual_mov_b32 v33, v73
	v_dual_mov_b32 v34, v74 :: v_dual_mov_b32 v35, v75
	v_dual_mov_b32 v22, v70 :: v_dual_mov_b32 v23, v71
	v_dual_mov_b32 v24, v72 :: v_dual_mov_b32 v25, v73
	v_dual_mov_b32 v26, v74 :: v_dual_mov_b32 v27, v75
	v_dual_mov_b32 v14, v70 :: v_dual_mov_b32 v15, v71
	v_dual_mov_b32 v16, v72 :: v_dual_mov_b32 v17, v73
	v_dual_mov_b32 v18, v74 :: v_dual_mov_b32 v19, v75
	s_branch .LBB0_22
.LBB0_21:
	v_mov_b32_e32 v2, v161
.LBB0_22:
	v_lshlrev_b32_e32 v1, 6, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v2, v2, v160
	s_load_b64 s[4:5], s[0:1], 0x40
	v_and_b32_e32 v9, 31, v0
	v_and_b32_e32 v1, 0x3800, v1
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v4, 17, v2
	v_add_nc_u32_e32 v5, 18, v2
	v_add_nc_u32_e32 v10, 19, v2
	v_add_nc_u32_e32 v76, 20, v2
	v_add_nc_u32_e32 v3, 0, v1
	v_and_b32_e32 v1, 31, v2
	v_and_b32_e32 v4, 31, v4
	v_and_b32_e32 v8, 31, v5
	v_add_nc_u32_e32 v77, 21, v2
	v_lshl_add_u32 v11, v160, 7, v3
	v_xor_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v5, v2, 2, v11
	v_lshl_add_u32 v6, v1, 2, v11
	v_add_nc_u32_e32 v1, 22, v2
	v_add_nc_u32_e32 v2, 23, v2
	v_lshl_add_u32 v7, v4, 2, v11
	v_and_b32_e32 v4, 31, v10
	v_and_b32_e32 v10, 31, v76
	v_lshl_add_u32 v8, v8, 2, v11
	ds_store_2addr_b32 v5, v68, v69 offset1:1
	ds_store_2addr_b32 v5, v70, v71 offset0:2 offset1:3
	ds_store_2addr_b32 v5, v72, v73 offset0:4 offset1:5
	ds_store_2addr_b32 v5, v74, v75 offset0:6 offset1:7
	ds_store_b32 v6, v60
	ds_store_b32 v7, v61
	ds_store_b32 v8, v62
	v_and_b32_e32 v60, 31, v77
	v_and_b32_e32 v1, 31, v1
	v_and_b32_e32 v2, 31, v2
	v_lshl_add_u32 v62, v4, 2, v11
	v_lshl_add_u32 v68, v10, 2, v11
	v_lshl_add_u32 v69, v60, 2, v11
	v_lshl_add_u32 v70, v1, 2, v11
	v_lshl_add_u32 v71, v2, 2, v11
	ds_store_b32 v62, v63
	ds_store_b32 v68, v64
	ds_store_b32 v69, v65
	ds_store_b32 v70, v66
	ds_store_b32 v71, v67
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or3_b32 v4, s35, v158, v9
	v_add_nc_u32_e32 v1, s33, v159
	v_lshl_add_u32 v9, v9, 2, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s2, s34, v4
	s_wait_kmcnt 0x0
	v_cmp_gt_i32_e64 s3, s31, v1
	v_cmp_le_i32_e64 s1, s34, v4
	v_cmp_gt_i32_e64 s0, s30, v4
	v_ashrrev_i32_e32 v2, 31, v1
	v_cmp_gt_i32_e32 vcc_lo, s24, v4
	s_and_b32 s2, s3, s2
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	ds_load_b32 v61, v9
	v_dual_mov_b32 v63, s22 :: v_dual_mov_b32 v64, s23
	v_add_co_u32 v10, s2, s18, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s19, v11, s2
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v63, s4, v63, s0
	v_cndmask_b32_e64 v64, s5, v64, s0
	global_load_b32 v60, v[10:11], off
	v_dual_mov_b32 v10, s24 :: v_dual_mov_b32 v11, s25
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v63, v63, s20, vcc_lo
	v_cndmask_b32_e64 v64, v64, s21, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v10, s30, v10, s0
	v_cndmask_b32_e64 v11, s26, v11, s0
	v_cndmask_b32_e64 v10, v10, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v11, v11, s24, vcc_lo
	v_sub_nc_u32_e32 v10, v4, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[10:11], null, v11, v1, v[10:11]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v11, 0 :: v_dual_mul_f32 v60, v60, v61
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s2, v63, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v64, v11, s2
	global_store_b32 v[10:11], v60, off
.LBB0_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v11, 1, v1
	v_add_nc_u32_e32 v10, 1, v0
	s_xor_b32 s3, s1, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s31, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_lshlrev_b64_e32 v[60:61], 2, v[1:2]
	v_mov_b32_e32 v64, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v64, s26, v64, s0
	v_add_co_u32 v60, s1, s18, v60
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v61, null, s19, v61, s1
	global_load_b32 v63, v[60:61], off offset:4
	v_dual_mov_b32 v60, s24 :: v_dual_and_b32 v61, 31, v10
	v_cndmask_b32_e64 v60, s30, v60, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v61, v61, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v60, v60, 0, vcc_lo
	ds_load_b32 v65, v61 offset:128
	v_cndmask_b32_e64 v61, v64, s24, vcc_lo
	v_mov_b32_e32 v64, s22
	v_sub_nc_u32_e32 v60, v4, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[60:61], null, v61, v11, v[60:61]
	v_mov_b32_e32 v11, s23
	v_mov_b32_e32 v61, 0
	v_cndmask_b32_e64 v64, s4, v64, s0
	v_cndmask_b32_e64 v11, s5, v11, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]
	v_cndmask_b32_e64 v64, v64, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v11, v11, s21, vcc_lo
	v_add_co_u32 v60, s1, v64, v60
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v61, null, v11, v61, s1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v63, v63, v65
	global_store_b32 v[60:61], v63, off
.LBB0_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v60, 2, v1
	v_add_nc_u32_e32 v11, 2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v60
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_lshlrev_b64_e32 v[63:64], 2, v[1:2]
	v_mov_b32_e32 v61, s24
	v_dual_mov_b32 v65, s25 :: v_dual_mov_b32 v66, s22
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v63, s1, s18, v63
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v64, null, s19, v64, s1
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v65, s26, v65, s0
	v_cndmask_b32_e64 v66, s4, v66, s0
	global_load_b32 v63, v[63:64], off offset:8
	v_and_b32_e32 v64, 31, v11
	v_cndmask_b32_e64 v61, s30, v61, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v65, v65, s24, vcc_lo
	v_cndmask_b32_e64 v66, v66, s20, vcc_lo
	v_lshl_add_u32 v64, v64, 2, v3
	v_cndmask_b32_e64 v61, v61, 0, vcc_lo
	ds_load_b32 v64, v64 offset:256
	v_sub_nc_u32_e32 v61, v4, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[60:61], null, v65, v60, v[61:62]
	v_mov_b32_e32 v65, s23
	v_mov_b32_e32 v61, 0
	v_cndmask_b32_e64 v65, s5, v65, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]
	v_cndmask_b32_e64 v65, v65, s21, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v60, s1, v66, v60
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v61, null, v65, v61, s1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v63, v63, v64
	global_store_b32 v[60:61], v63, off
.LBB0_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v61, 3, v1
	v_add_nc_u32_e32 v60, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v61
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_lshlrev_b64_e32 v[63:64], 2, v[1:2]
	v_mov_b32_e32 v66, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v66, s26, v66, s0
	v_add_co_u32 v63, s1, s18, v63
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v64, null, s19, v64, s1
	global_load_b32 v65, v[63:64], off offset:12
	v_dual_mov_b32 v63, s24 :: v_dual_and_b32 v64, 31, v60
	v_cndmask_b32_e64 v63, s30, v63, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v64, v64, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v63, v63, 0, vcc_lo
	ds_load_b32 v67, v64 offset:384
	v_cndmask_b32_e64 v64, v66, s24, vcc_lo
	v_mov_b32_e32 v66, s22
	v_sub_nc_u32_e32 v63, v4, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[63:64], null, v64, v61, v[63:64]
	v_dual_mov_b32 v61, s23 :: v_dual_mov_b32 v64, 0
	v_cndmask_b32_e64 v61, s5, v61, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[63:64], 2, v[63:64]
	v_cndmask_b32_e64 v61, v61, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v65, v65, v67
	v_cndmask_b32_e64 v66, s4, v66, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, s20, vcc_lo
	v_add_co_u32 v63, s1, v66, v63
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v64, null, v61, v64, s1
	global_store_b32 v[63:64], v65, off
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v63, 4, v1
	v_add_nc_u32_e32 v61, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v63
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_lshlrev_b64_e32 v[64:65], 2, v[1:2]
	v_dual_mov_b32 v67, s25 :: v_dual_and_b32 v66, 31, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v64, s1, s18, v64
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v65, null, s19, v65, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v67, s26, v67, s0
	global_load_b32 v65, v[64:65], off offset:16
	v_mov_b32_e32 v64, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v67, v67, s24, vcc_lo
	v_cndmask_b32_e64 v64, s30, v64, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, 0, vcc_lo
	v_sub_nc_u32_e32 v64, v4, v64
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[63:64], null, v67, v63, v[64:65]
	v_mov_b32_e32 v67, s23
	v_lshl_add_u32 v66, v66, 2, v3
	v_mov_b32_e32 v64, 0
	v_cndmask_b32_e64 v67, s5, v67, s0
	ds_load_b32 v66, v66 offset:512
	v_mov_b32_e32 v72, s22
	v_lshlrev_b64_e32 v[63:64], 2, v[63:64]
	v_cndmask_b32_e64 v67, v67, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v65, v66
	v_cndmask_b32_e64 v72, s4, v72, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, s20, vcc_lo
	v_add_co_u32 v63, s1, v72, v63
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v64, null, v67, v64, s1
	global_store_b32 v[63:64], v65, off
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v64, 5, v1
	v_add_nc_u32_e32 v63, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v64
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_lshlrev_b64_e32 v[65:66], 2, v[1:2]
	v_dual_mov_b32 v72, s25 :: v_dual_and_b32 v67, 31, v63
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v65, s1, s18, v65
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v66, null, s19, v66, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v72, s26, v72, s0
	global_load_b32 v66, v[65:66], off offset:20
	v_mov_b32_e32 v65, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v72, v72, s24, vcc_lo
	v_cndmask_b32_e64 v65, s30, v65, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, 0, vcc_lo
	v_sub_nc_u32_e32 v65, v4, v65
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[64:65], null, v72, v64, v[65:66]
	v_mov_b32_e32 v72, s23
	v_lshl_add_u32 v67, v67, 2, v3
	v_mov_b32_e32 v65, 0
	v_cndmask_b32_e64 v72, s5, v72, s0
	ds_load_b32 v67, v67 offset:640
	v_mov_b32_e32 v73, s22
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_cndmask_b32_e64 v72, v72, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v66, v66, v67
	v_cndmask_b32_e64 v73, s4, v73, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v73, v73, s20, vcc_lo
	v_add_co_u32 v64, s1, v73, v64
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v65, null, v72, v65, s1
	global_store_b32 v[64:65], v66, off
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v65, 6, v1
	v_add_nc_u32_e32 v64, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v65
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_lshlrev_b64_e32 v[66:67], 2, v[1:2]
	v_dual_mov_b32 v73, s25 :: v_dual_and_b32 v72, 31, v64
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v66, s1, s18, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v67, null, s19, v67, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v73, s26, v73, s0
	global_load_b32 v67, v[66:67], off offset:24
	v_mov_b32_e32 v66, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v73, v73, s24, vcc_lo
	v_cndmask_b32_e64 v66, s30, v66, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, 0, vcc_lo
	v_sub_nc_u32_e32 v66, v4, v66
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, v73, v65, v[66:67]
	v_mov_b32_e32 v73, s23
	v_lshl_add_u32 v72, v72, 2, v3
	v_mov_b32_e32 v66, 0
	v_cndmask_b32_e64 v73, s5, v73, s0
	ds_load_b32 v72, v72 offset:768
	v_mov_b32_e32 v74, s22
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	v_cndmask_b32_e64 v73, v73, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v67, v72
	v_cndmask_b32_e64 v74, s4, v74, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v74, v74, s20, vcc_lo
	v_add_co_u32 v65, s1, v74, v65
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v66, null, v73, v66, s1
	global_store_b32 v[65:66], v67, off
.LBB0_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v66, 7, v1
	v_add_nc_u32_e32 v65, 7, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v66
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_lshlrev_b64_e32 v[72:73], 2, v[1:2]
	v_dual_mov_b32 v67, s24 :: v_dual_mov_b32 v74, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v67, s30, v67, s0
	v_add_co_u32 v72, s1, s18, v72
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v73, null, s19, v73, s1
	v_cndmask_b32_e64 v74, s26, v74, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v67, v67, 0, vcc_lo
	global_load_b32 v72, v[72:73], off offset:28
	v_and_b32_e32 v73, 31, v65
	v_cndmask_b32_e64 v74, v74, s24, vcc_lo
	v_sub_nc_u32_e32 v67, v4, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[66:67], null, v74, v66, v[67:68]
	v_mov_b32_e32 v74, s23
	v_lshl_add_u32 v73, v73, 2, v3
	v_mov_b32_e32 v67, 0
	v_cndmask_b32_e64 v74, s5, v74, s0
	ds_load_b32 v73, v73 offset:896
	v_mov_b32_e32 v75, s22
	v_lshlrev_b64_e32 v[66:67], 2, v[66:67]
	v_cndmask_b32_e64 v74, v74, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v72, v72, v73
	v_cndmask_b32_e64 v75, s4, v75, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v75, v75, s20, vcc_lo
	v_add_co_u32 v66, s1, v75, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v67, null, v74, v67, s1
	global_store_b32 v[66:67], v72, off
.LBB0_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v67, 8, v1
	v_add_nc_u32_e32 v66, 8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v67
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_lshlrev_b64_e32 v[72:73], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v72, s1, s18, v72
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s19, v73, s1
	global_load_b32 v74, v[72:73], off offset:32
	v_dual_mov_b32 v72, s24 :: v_dual_and_b32 v73, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v72, s30, v72, s0
	v_lshl_add_u32 v73, v73, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v72, v72, 0, vcc_lo
	ds_load_b32 v76, v73 offset:1024
	v_mov_b32_e32 v75, s25
	v_sub_nc_u32_e32 v72, v4, v72
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v74, v74, v76
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v75, s26, v75, s0
	v_cndmask_b32_e64 v73, v75, s24, vcc_lo
	v_mov_b32_e32 v75, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[72:73], null, v73, v67, v[72:73]
	v_mov_b32_e32 v67, s23
	v_mov_b32_e32 v73, 0
	v_cndmask_b32_e64 v75, s4, v75, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v67, s5, v67, s0
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v75, v75, s20, vcc_lo
	v_cndmask_b32_e64 v67, v67, s21, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v72, s1, v75, v72
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, v67, v73, s1
	global_store_b32 v[72:73], v74, off
.LBB0_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v72, 9, v1
	v_add_nc_u32_e32 v67, 9, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v72
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_lshlrev_b64_e32 v[73:74], 2, v[1:2]
	v_dual_mov_b32 v76, s25 :: v_dual_and_b32 v75, 31, v67
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v73, s1, s18, v73
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v74, null, s19, v74, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v76, s26, v76, s0
	global_load_b32 v74, v[73:74], off offset:36
	v_mov_b32_e32 v73, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v76, v76, s24, vcc_lo
	v_cndmask_b32_e64 v73, s30, v73, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v73, v73, 0, vcc_lo
	v_sub_nc_u32_e32 v73, v4, v73
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[72:73], null, v76, v72, v[73:74]
	v_mov_b32_e32 v76, s23
	v_lshl_add_u32 v75, v75, 2, v3
	v_mov_b32_e32 v73, 0
	v_cndmask_b32_e64 v76, s5, v76, s0
	ds_load_b32 v75, v75 offset:1152
	v_mov_b32_e32 v77, s22
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	v_cndmask_b32_e64 v76, v76, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v74, v74, v75
	v_cndmask_b32_e64 v77, s4, v77, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v77, v77, s20, vcc_lo
	v_add_co_u32 v72, s1, v77, v72
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, v76, v73, s1
	global_store_b32 v[72:73], v74, off
.LBB0_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v73, 10, v1
	v_add_nc_u32_e32 v72, 10, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v73
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_lshlrev_b64_e32 v[74:75], 2, v[1:2]
	v_dual_mov_b32 v77, s25 :: v_dual_and_b32 v76, 31, v72
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v74, s1, s18, v74
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s19, v75, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v77, s26, v77, s0
	global_load_b32 v75, v[74:75], off offset:40
	v_mov_b32_e32 v74, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v77, v77, s24, vcc_lo
	v_cndmask_b32_e64 v74, s30, v74, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v74, v74, 0, vcc_lo
	v_sub_nc_u32_e32 v74, v4, v74
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[73:74], null, v77, v73, v[74:75]
	v_mov_b32_e32 v77, s23
	v_lshl_add_u32 v76, v76, 2, v3
	v_mov_b32_e32 v74, 0
	v_cndmask_b32_e64 v77, s5, v77, s0
	ds_load_b32 v76, v76 offset:1280
	v_mov_b32_e32 v78, s22
	v_lshlrev_b64_e32 v[73:74], 2, v[73:74]
	v_cndmask_b32_e64 v77, v77, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v75, v75, v76
	v_cndmask_b32_e64 v78, s4, v78, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v78, v78, s20, vcc_lo
	v_add_co_u32 v73, s1, v78, v73
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v74, null, v77, v74, s1
	global_store_b32 v[73:74], v75, off
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v74, 11, v1
	v_add_nc_u32_e32 v73, 11, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v74
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_lshlrev_b64_e32 v[75:76], 2, v[1:2]
	v_dual_mov_b32 v78, s25 :: v_dual_and_b32 v77, 31, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v75, s1, s18, v75
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v76, null, s19, v76, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v78, s26, v78, s0
	global_load_b32 v76, v[75:76], off offset:44
	v_mov_b32_e32 v75, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v78, v78, s24, vcc_lo
	v_cndmask_b32_e64 v75, s30, v75, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v75, v75, 0, vcc_lo
	v_sub_nc_u32_e32 v75, v4, v75
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[74:75], null, v78, v74, v[75:76]
	v_mov_b32_e32 v78, s23
	v_lshl_add_u32 v77, v77, 2, v3
	v_mov_b32_e32 v75, 0
	v_cndmask_b32_e64 v78, s5, v78, s0
	ds_load_b32 v77, v77 offset:1408
	v_mov_b32_e32 v79, s22
	v_lshlrev_b64_e32 v[74:75], 2, v[74:75]
	v_cndmask_b32_e64 v78, v78, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v76, v76, v77
	v_cndmask_b32_e64 v79, s4, v79, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v79, v79, s20, vcc_lo
	v_add_co_u32 v74, s1, v79, v74
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, v78, v75, s1
	global_store_b32 v[74:75], v76, off
.LBB0_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v75, 12, v1
	v_add_nc_u32_e32 v74, 12, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v75
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_lshlrev_b64_e32 v[76:77], 2, v[1:2]
	v_dual_mov_b32 v79, s25 :: v_dual_and_b32 v78, 31, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v76, s1, s18, v76
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v77, null, s19, v77, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v79, s26, v79, s0
	global_load_b32 v77, v[76:77], off offset:48
	v_mov_b32_e32 v76, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v79, v79, s24, vcc_lo
	v_cndmask_b32_e64 v76, s30, v76, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, 0, vcc_lo
	v_sub_nc_u32_e32 v76, v4, v76
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[75:76], null, v79, v75, v[76:77]
	v_mov_b32_e32 v79, s23
	v_lshl_add_u32 v78, v78, 2, v3
	v_mov_b32_e32 v76, 0
	v_cndmask_b32_e64 v79, s5, v79, s0
	ds_load_b32 v78, v78 offset:1536
	v_mov_b32_e32 v80, s22
	v_lshlrev_b64_e32 v[75:76], 2, v[75:76]
	v_cndmask_b32_e64 v79, v79, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v77, v77, v78
	v_cndmask_b32_e64 v80, s4, v80, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, s20, vcc_lo
	v_add_co_u32 v75, s1, v80, v75
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v76, null, v79, v76, s1
	global_store_b32 v[75:76], v77, off
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v76, 13, v1
	v_add_nc_u32_e32 v75, 13, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v76
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_lshlrev_b64_e32 v[77:78], 2, v[1:2]
	v_dual_mov_b32 v80, s25 :: v_dual_and_b32 v79, 31, v75
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v77, s1, s18, v77
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v78, null, s19, v78, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v80, s26, v80, s0
	global_load_b32 v78, v[77:78], off offset:52
	v_mov_b32_e32 v77, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v80, v80, s24, vcc_lo
	v_cndmask_b32_e64 v77, s30, v77, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v77, v77, 0, vcc_lo
	v_sub_nc_u32_e32 v77, v4, v77
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[76:77], null, v80, v76, v[77:78]
	v_mov_b32_e32 v80, s23
	v_lshl_add_u32 v79, v79, 2, v3
	v_mov_b32_e32 v77, 0
	v_cndmask_b32_e64 v80, s5, v80, s0
	ds_load_b32 v79, v79 offset:1664
	v_mov_b32_e32 v81, s22
	v_lshlrev_b64_e32 v[76:77], 2, v[76:77]
	v_cndmask_b32_e64 v80, v80, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v78, v78, v79
	v_cndmask_b32_e64 v81, s4, v81, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v81, v81, s20, vcc_lo
	v_add_co_u32 v76, s1, v81, v76
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v77, null, v80, v77, s1
	global_store_b32 v[76:77], v78, off
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v77, 14, v1
	v_add_nc_u32_e32 v76, 14, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v77
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_lshlrev_b64_e32 v[78:79], 2, v[1:2]
	v_dual_mov_b32 v81, s25 :: v_dual_and_b32 v80, 31, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v78, s1, s18, v78
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s19, v79, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v81, s26, v81, s0
	global_load_b32 v79, v[78:79], off offset:56
	v_mov_b32_e32 v78, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v81, v81, s24, vcc_lo
	v_cndmask_b32_e64 v78, s30, v78, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v78, v78, 0, vcc_lo
	v_sub_nc_u32_e32 v78, v4, v78
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[77:78], null, v81, v77, v[78:79]
	v_mov_b32_e32 v81, s23
	v_lshl_add_u32 v80, v80, 2, v3
	v_mov_b32_e32 v78, 0
	v_cndmask_b32_e64 v81, s5, v81, s0
	ds_load_b32 v80, v80 offset:1792
	v_mov_b32_e32 v82, s22
	v_lshlrev_b64_e32 v[77:78], 2, v[77:78]
	v_cndmask_b32_e64 v81, v81, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v79, v79, v80
	v_cndmask_b32_e64 v82, s4, v82, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v82, v82, s20, vcc_lo
	v_add_co_u32 v77, s1, v82, v77
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v78, null, v81, v78, s1
	global_store_b32 v[77:78], v79, off
.LBB0_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v77, 15, v1
	v_add_nc_u32_e32 v0, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s31, v77
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_54
; %bb.53:
	v_lshlrev_b64_e32 v[78:79], 2, v[1:2]
	v_dual_mov_b32 v81, s25 :: v_dual_and_b32 v80, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v78, s1, s18, v78
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s19, v79, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v81, s26, v81, s0
	global_load_b32 v79, v[78:79], off offset:60
	v_mov_b32_e32 v78, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v81, v81, s24, vcc_lo
	v_cndmask_b32_e64 v78, s30, v78, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v78, v78, 0, vcc_lo
	v_sub_nc_u32_e32 v78, v4, v78
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[77:78], null, v81, v77, v[78:79]
	v_mov_b32_e32 v81, s23
	v_lshl_add_u32 v80, v80, 2, v3
	v_mov_b32_e32 v78, 0
	v_cndmask_b32_e64 v81, s5, v81, s0
	ds_load_b32 v80, v80 offset:1920
	v_mov_b32_e32 v82, s22
	v_lshlrev_b64_e32 v[77:78], 2, v[77:78]
	v_cndmask_b32_e64 v81, v81, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v79, v79, v80
	v_cndmask_b32_e64 v82, s4, v82, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v82, v82, s20, vcc_lo
	v_add_co_u32 v77, s1, v82, v77
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v78, null, v81, v78, s1
	global_store_b32 v[77:78], v79, off
.LBB0_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v52, v53 offset1:1
	ds_store_2addr_b32 v5, v54, v55 offset0:2 offset1:3
	ds_store_2addr_b32 v5, v56, v57 offset0:4 offset1:5
	ds_store_2addr_b32 v5, v58, v59 offset0:6 offset1:7
	ds_store_b32 v6, v44
	ds_store_b32 v7, v45
	ds_store_b32 v8, v46
	ds_store_b32 v62, v47
	ds_store_b32 v68, v48
	ds_store_b32 v69, v49
	ds_store_b32 v70, v50
	ds_store_b32 v71, v51
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v44, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	ds_load_b32 v48, v9
	v_mov_b32_e32 v47, s25
	v_mov_b32_e32 v49, s22
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	v_cndmask_b32_e64 v47, s26, v47, s0
	v_cndmask_b32_e64 v49, s4, v49, s0
	global_load_b32 v46, v[45:46], off offset:64
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v47, v47, s24, vcc_lo
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, s30, v45, s0
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[44:45], null, v47, v44, v[45:46]
	v_mov_b32_e32 v47, s23
	s_wait_dscnt 0x0
	v_dual_mov_b32 v45, 0 :: v_dual_mul_f32 v46, v46, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v47, s5, v47, s0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v47, v47, s21, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v47, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 17, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:68
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:128
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:72
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:256
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 19, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v60
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:76
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:384
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 20, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:80
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:512
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 21, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v63
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:84
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:640
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 22, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v64
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:88
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:768
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 23, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v65
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:92
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:896
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 24, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v66
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:96
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:1024
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 25, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_74
; %bb.73:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v67
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:100
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:1152
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 26, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_76
; %bb.75:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v72
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:104
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:1280
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 27, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_78
; %bb.77:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:108
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:1408
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 28, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_80
; %bb.79:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:112
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:1536
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 29, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_82
; %bb.81:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v75
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:116
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:1664
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 30, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_84
; %bb.83:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:120
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:1792
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v44, 31, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v44
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_86
; %bb.85:
	v_lshlrev_b64_e32 v[45:46], 2, v[1:2]
	v_dual_mov_b32 v48, s25 :: v_dual_and_b32 v47, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v45, s1, s18, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, s19, v46, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, s26, v48, s0
	global_load_b32 v46, v[45:46], off offset:124
	v_mov_b32_e32 v45, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v48, v48, s24, vcc_lo
	v_cndmask_b32_e64 v45, s30, v45, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, 0, vcc_lo
	v_sub_nc_u32_e32 v45, v4, v45
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[44:45], null, v48, v44, v[45:46]
	v_mov_b32_e32 v48, s23
	v_lshl_add_u32 v47, v47, 2, v3
	v_mov_b32_e32 v45, 0
	v_cndmask_b32_e64 v48, s5, v48, s0
	ds_load_b32 v47, v47 offset:1920
	v_mov_b32_e32 v49, s22
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_cndmask_b32_e64 v48, v48, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	v_cndmask_b32_e64 v49, s4, v49, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, s20, vcc_lo
	v_add_co_u32 v44, s1, v49, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, v48, v45, s1
	global_store_b32 v[44:45], v46, off
.LBB0_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v36, v37 offset1:1
	ds_store_2addr_b32 v5, v38, v39 offset0:2 offset1:3
	ds_store_2addr_b32 v5, v40, v41 offset0:4 offset1:5
	ds_store_2addr_b32 v5, v42, v43 offset0:6 offset1:7
	ds_store_b32 v6, v28
	ds_store_b32 v7, v29
	ds_store_b32 v8, v30
	ds_store_b32 v62, v31
	ds_store_b32 v68, v32
	ds_store_b32 v69, v33
	ds_store_b32 v70, v34
	ds_store_b32 v71, v35
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v28, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_88
; %bb.87:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	ds_load_b32 v32, v9
	v_mov_b32_e32 v31, s25
	v_mov_b32_e32 v33, s22
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	v_cndmask_b32_e64 v31, s26, v31, s0
	v_cndmask_b32_e64 v33, s4, v33, s0
	global_load_b32 v30, v[29:30], off offset:128
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v31, v31, s24, vcc_lo
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, s30, v29, s0
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[28:29], null, v31, v28, v[29:30]
	v_mov_b32_e32 v31, s23
	s_wait_dscnt 0x0
	v_dual_mov_b32 v29, 0 :: v_dual_mul_f32 v30, v30, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v31, s5, v31, s0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v31, v31, s21, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v31, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 33, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_90
; %bb.89:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:132
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:128
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 34, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_92
; %bb.91:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:136
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:256
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 35, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_94
; %bb.93:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v60
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:140
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:384
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 36, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_96
; %bb.95:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:144
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:512
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 37, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_98
; %bb.97:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v63
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:148
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:640
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 38, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_100
; %bb.99:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v64
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:152
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:768
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 39, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_102
; %bb.101:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v65
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:156
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:896
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 40, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_104
; %bb.103:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v66
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:160
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:1024
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 41, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v67
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:164
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:1152
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 42, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v72
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:168
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:1280
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 43, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_110
; %bb.109:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:172
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:1408
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 44, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_112
; %bb.111:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:176
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:1536
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 45, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_114
; %bb.113:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v75
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:180
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:1664
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 46, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_116
; %bb.115:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:184
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:1792
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 47, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_118
; %bb.117:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:188
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s30, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:1920
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v20, v21 offset1:1
	ds_store_2addr_b32 v5, v22, v23 offset0:2 offset1:3
	ds_store_2addr_b32 v5, v24, v25 offset0:4 offset1:5
	ds_store_2addr_b32 v5, v26, v27 offset0:6 offset1:7
	ds_store_b32 v6, v12
	ds_store_b32 v7, v13
	ds_store_b32 v8, v14
	ds_store_b32 v62, v15
	ds_store_b32 v68, v16
	ds_store_b32 v69, v17
	ds_store_b32 v70, v18
	ds_store_b32 v71, v19
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v5, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_120
; %bb.119:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	ds_load_b32 v9, v9
	v_mov_b32_e32 v8, s25
	v_mov_b32_e32 v12, s22
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v12, s4, v12, s0
	global_load_b32 v7, v[6:7], off offset:192
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v8, v8, s24, vcc_lo
	v_cndmask_b32_e64 v12, v12, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, s30, v6, s0
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[5:6], null, v8, v5, v[6:7]
	v_mov_b32_e32 v8, s23
	s_wait_dscnt 0x0
	v_dual_mov_b32 v6, 0 :: v_dual_mul_f32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v8, s5, v8, s0
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v8, v8, s21, vcc_lo
	v_add_co_u32 v5, s1, v12, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v8, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 49, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_122
; %bb.121:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:196
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:128
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_124
; %bb.123:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:200
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:256
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 51, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_126
; %bb.125:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v60
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:204
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:384
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 52, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_128
; %bb.127:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:208
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:512
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 53, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_130
; %bb.129:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v63
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:212
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:640
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 54, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_132
; %bb.131:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v64
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:216
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:768
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 55, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_134
; %bb.133:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v65
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:220
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:896
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 56, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_136
; %bb.135:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v66
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:224
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:1024
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 57, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_138
; %bb.137:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v67
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:228
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:1152
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 58, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_140
; %bb.139:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v72
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:232
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:1280
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 59, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_142
; %bb.141:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:236
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:1408
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_144
; %bb.143:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:240
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:1536
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 61, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_146
; %bb.145:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v75
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:244
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:1664
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 62, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_148
; %bb.147:
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	v_dual_mov_b32 v9, s25 :: v_dual_and_b32 v8, 31, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, s1, s18, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s19, v7, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, s26, v9, s0
	global_load_b32 v7, v[6:7], off offset:248
	v_mov_b32_e32 v6, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s24, vcc_lo
	v_cndmask_b32_e64 v6, s30, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_sub_nc_u32_e32 v6, v4, v6
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v9, v5, v[6:7]
	v_mov_b32_e32 v9, s23
	v_lshl_add_u32 v8, v8, 2, v3
	v_mov_b32_e32 v6, 0
	v_cndmask_b32_e64 v9, s5, v9, s0
	ds_load_b32 v8, v8 offset:1792
	v_mov_b32_e32 v10, s22
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v9, v9, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, s4, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, s20, vcc_lo
	v_add_co_u32 v5, s1, v10, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v9, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_148:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 63, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s31, v5
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_150
; %bb.149:
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_mov_b32_e32 v6, s25
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v1, s1, s18, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s19, v2, s1
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_add_u32 v0, v0, 2, v3
	v_cndmask_b32_e64 v3, s26, v6, s0
	global_load_b32 v2, v[1:2], off offset:252
	v_mov_b32_e32 v1, s24
	ds_load_b32 v6, v0 offset:1920
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	v_cndmask_b32_e64 v1, s30, v1, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_sub_nc_u32_e32 v0, v4, v1
	v_mov_b32_e32 v4, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[0:1], null, v3, v5, v[0:1]
	v_mov_b32_e32 v3, s23
	v_mov_b32_e32 v1, 0
	v_cndmask_b32_e64 v4, s4, v4, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s5, v3, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v4, v4, s20, vcc_lo
	v_cndmask_b32_e64 v3, v3, s21, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v3, v1, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v2, v2, v6
	global_store_b32 v[0:1], v2, off
.LBB0_150:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end0:
	.size	qkv_full, .Lfunc_end0-qkv_full
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel qkv_full
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 92
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
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 211
		.amdhsa_next_free_sgpr 40
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-qkv_full)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lqkv_full.num_vgpr, 211
	.set .Lqkv_full.num_agpr, 0
	.set .Lqkv_full.numbered_sgpr, 40
	.set .Lqkv_full.num_named_barrier, 0
	.set .Lqkv_full.private_seg_size, 0
	.set .Lqkv_full.uses_vcc, 1
	.set .Lqkv_full.uses_flat_scratch, 0
	.set .Lqkv_full.has_dyn_sized_stack, 0
	.set .Lqkv_full.has_recursion, 0
	.set .Lqkv_full.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 23208
; TotalNumSgprs: 42
; NumVgprs: 211
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 26
; NumSGPRsForWavesPerEU: 42
; NumVGPRsForWavesPerEU: 211
; Occupancy: 7
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_bc3acd0dade202d1,@object ; @__hip_cuid_bc3acd0dade202d1
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_bc3acd0dade202d1
__hip_cuid_bc3acd0dade202d1:
	.byte	0                               ; 0x0
	.size	__hip_cuid_bc3acd0dade202d1, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_bc3acd0dade202d1
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 92
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           qkv_full
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         qkv_full.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     211
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
