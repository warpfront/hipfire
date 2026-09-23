	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	attention_q8_0_fa2_gqa_gfx1201 ; -- Begin function attention_q8_0_fa2_gqa_gfx1201
	.globl	attention_q8_0_fa2_gqa_gfx1201
	.p2align	8
	.type	attention_q8_0_fa2_gqa_gfx1201,@function
attention_q8_0_fa2_gqa_gfx1201:         ; @attention_q8_0_fa2_gqa_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_141
; %bb.1:
	s_cmp_gt_i32 ttmp7, 3
	s_cbranch_scc1 .LBB0_141
; %bb.2:
	s_load_b32 s16, s[0:1], 0x38
	s_lshl_b32 s3, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s3, s7
	s_cbranch_scc1 .LBB0_141
; %bb.3:
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v184, 0 :: v_dual_mov_b32 v1, 0
	v_mov_b32_e32 v2, 0
	v_lshrrev_b32_e32 v195, 3, v0
	v_mov_b32_e32 v186, 0
	s_mov_b32 s17, 0
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_5
; %bb.4:
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_or_b32 v184, v0, 7, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 6, v1
	v_cmp_gt_i32_e32 vcc_lo, s7, v184
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, ttmp7, 6, v[1:2]
	s_and_b32 s17, vcc_lo, exec_lo
	v_and_or_b32 v1, v195, 1, v1
	v_lshlrev_b32_e32 v186, 8, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x1800, v184, v[186:187]
	v_mov_b32_e32 v2, 0
.LBB0_5:
	s_or_b32 exec_lo, exec_lo, s4
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x20
	v_and_b32_e32 v6, 31, v0
	v_mov_b32_e32 v4, -1
	v_bfrev_b32_e32 v5, -2
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_u32_e32 8, v6
	s_cbranch_execz .LBB0_9
; %bb.6:
	v_or_b32_e32 v3, s3, v6
	v_bfrev_b32_e32 v5, -2
	v_mov_b32_e32 v4, -1
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s7, v3
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_ashrrev_i32_e32 v4, 31, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, vcc_lo, s0, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v4, null, s1, v4, vcc_lo
	global_load_b32 v4, v[3:4], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v5, v4
.LBB0_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_9:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s4
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_lshl_add_u32 v199, v6, 4, 0
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshrrev_b32_e32 v9, 1, v0
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	v_xor_b32_e32 v7, 16, v3
	v_xor_b32_e32 v15, 8, v3
	v_lshlrev_b64_e32 v[128:129], 2, v[184:185]
	v_xor_b32_e32 v16, 4, v3
	v_xor_b32_e32 v17, 2, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	v_xor_b32_e32 v18, 1, v3
	v_dual_mov_b32 v225, 0xff800000 :: v_dual_and_b32 v8, 7, v0
	v_or_b32_e32 v11, 0x80, v0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v3, v7 :: v_dual_lshlrev_b32 v196, 8, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v15
	v_lshlrev_b32_e32 v198, 6, v0
	v_or_b32_e32 v12, 0x100, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v80, 0 :: v_dual_lshlrev_b32 v197, 2, v7
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v15, v3, v15 :: v_dual_and_b32 v130, 16, v6
	v_lshrrev_b32_e32 v6, 1, v6
	ds_bpermute_b32 v7, v197, v4
	ds_bpermute_b32 v10, v197, v5
	v_dual_mov_b32 v200, 0 :: v_dual_lshlrev_b32 v15, 2, v15
	v_dual_mov_b32 v82, v80 :: v_dual_and_b32 v185, 8, v6
	v_cmp_gt_u32_e32 vcc_lo, 32, v16
	v_and_b32_e32 v201, 16, v9
	v_or_b32_e32 v13, 0x180, v0
	v_dual_mov_b32 v83, v80 :: v_dual_lshlrev_b32 v14, 3, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, v3, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v17
	v_mov_b32_e32 v224, v185
	v_dual_mov_b32 v85, v80 :: v_dual_lshlrev_b32 v0, 4, v0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v17, v3, v17 :: v_dual_lshlrev_b32 v16, 2, v16
	s_wait_kmcnt 0x0
	v_add_co_u32 v131, vcc_lo, s8, v1
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v7
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v132, null, s9, v2, vcc_lo
	v_lshlrev_b32_e32 v1, 2, v17
	ds_bpermute_b32 v6, v15, v4
	ds_bpermute_b32 v7, v15, v5
	v_cmp_gt_u32_e32 vcc_lo, 32, v18
	v_lshl_or_b32 v8, ttmp7, 3, v8
	v_dual_mov_b32 v86, v80 :: v_dual_and_b32 v9, 12, v9
	v_dual_mov_b32 v84, v80 :: v_dual_and_b32 v205, 0x200, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_cndmask_b32 v3, v3, v18 :: v_dual_add_nc_u32 v206, 0, v9
	v_or_b32_e32 v14, 0x600, v0
	v_or_b32_e32 v15, 0x2600, v0
	v_or_b32_e32 v19, 0x2e00, v0
	v_or_b32_e32 v20, 0x3600, v0
	v_or_b32_e32 v21, 0x3e00, v0
	v_or_b32_e32 v22, 0x4600, v0
	v_or_b32_e32 v23, 0x5e00, v0
	v_or_b32_e32 v25, 0x6600, v0
	v_or_b32_e32 v26, 0x6e00, v0
	v_or_b32_e32 v27, 0x7600, v0
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v7
	v_or_b32_e32 v6, 0x4e00, v0
	v_or_b32_e32 v7, 0x5600, v0
	v_or_b32_e32 v0, 0x7e00, v0
	ds_bpermute_b32 v24, v16, v4
	ds_bpermute_b32 v16, v16, v5
	v_dual_mov_b32 v87, v80 :: v_dual_lshlrev_b32 v10, 3, v11
	v_mov_b32_e32 v81, v80
	v_mad_co_i64_i32 v[187:188], null, v8, 34, s[10:11]
	v_mad_co_i64_i32 v[189:190], null, v8, 34, s[12:13]
	v_lshlrev_b32_e32 v8, 2, v3
	v_add_nc_u32_e32 v213, 0, v19
	v_add_nc_u32_e32 v214, 0, v20
	v_add_nc_u32_e32 v215, 0, v21
	v_add_nc_u32_e32 v216, 0, v22
	v_add_nc_u32_e32 v219, 0, v23
	v_add_nc_u32_e32 v223, 0, v0
	v_add_co_u32 v191, vcc_lo, s0, v128
	v_lshrrev_b32_e32 v202, 3, v11
	v_lshrrev_b32_e32 v203, 3, v12
	v_lshrrev_b32_e32 v204, 3, v13
	v_lshl_or_b32 v11, v11, 4, 0x600
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v4, v24
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v5, v16
	v_dual_mov_b32 v16, v80 :: v_dual_mov_b32 v17, v81
	v_lshl_or_b32 v12, v12, 4, 0x600
	ds_bpermute_b32 v5, v1, v2
	ds_bpermute_b32 v1, v1, v4
	v_lshl_or_b32 v13, v13, 4, 0x600
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v192, null, s1, v129, vcc_lo
	v_add_co_u32 v193, vcc_lo, v131, v130
	v_and_b32_e32 v207, 0x600, v10
	v_add_nc_u32_e32 v208, 0, v14
	v_add_nc_u32_e32 v209, 0, v11
	v_add_nc_u32_e32 v210, 0, v12
	v_add_nc_u32_e32 v211, 0, v13
	v_add_nc_u32_e32 v212, 0, v15
	v_add_nc_u32_e32 v217, 0, v6
	v_add_nc_u32_e32 v218, 0, v7
	v_add_nc_u32_e32 v220, 0, v25
	v_add_nc_u32_e32 v221, 0, v26
	v_add_nc_u32_e32 v222, 0, v27
	v_dual_mov_b32 v24, v80 :: v_dual_mov_b32 v25, v81
	s_wait_dscnt 0x1
	v_max_i32_e32 v133, v2, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v134, v4, v1
	v_mov_b32_e32 v0, v80
	v_dual_mov_b32 v32, v80 :: v_dual_mov_b32 v33, v81
	ds_bpermute_b32 v135, v8, v133
	ds_bpermute_b32 v136, v8, v134
	v_mov_b32_e32 v8, v80
	v_dual_mov_b32 v40, v80 :: v_dual_mov_b32 v41, v81
	v_dual_mov_b32 v48, v80 :: v_dual_mov_b32 v49, v81
	v_dual_mov_b32 v56, v80 :: v_dual_mov_b32 v57, v81
	v_dual_mov_b32 v64, v80 :: v_dual_mov_b32 v65, v81
	v_dual_mov_b32 v72, v80 :: v_dual_mov_b32 v73, v81
	v_dual_mov_b32 v95, v87 :: v_dual_mov_b32 v94, v86
	v_dual_mov_b32 v103, v87 :: v_dual_mov_b32 v102, v86
	v_dual_mov_b32 v111, v87 :: v_dual_mov_b32 v110, v86
	v_dual_mov_b32 v119, v87 :: v_dual_mov_b32 v118, v86
	s_wait_dscnt 0x1
	v_max_i32_e32 v128, v133, v135
	s_wait_dscnt 0x0
	v_min_i32_e32 v129, v134, v136
	v_dual_mov_b32 v127, v87 :: v_dual_mov_b32 v126, v86
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v194, null, 0, v132, vcc_lo
	v_readfirstlane_b32 s11, v128
	v_readfirstlane_b32 s12, v129
	v_dual_mov_b32 v135, v87 :: v_dual_mov_b32 v134, v86
	v_dual_mov_b32 v1, v81 :: v_dual_mov_b32 v2, v82
	v_dual_mov_b32 v3, v83 :: v_dual_mov_b32 v4, v84
	v_dual_mov_b32 v5, v85 :: v_dual_mov_b32 v6, v86
	v_mov_b32_e32 v7, v87
	v_dual_mov_b32 v9, v81 :: v_dual_mov_b32 v10, v82
	v_dual_mov_b32 v11, v83 :: v_dual_mov_b32 v12, v84
	v_dual_mov_b32 v13, v85 :: v_dual_mov_b32 v14, v86
	v_dual_mov_b32 v15, v87 :: v_dual_mov_b32 v18, v82
	v_dual_mov_b32 v19, v83 :: v_dual_mov_b32 v20, v84
	v_dual_mov_b32 v21, v85 :: v_dual_mov_b32 v22, v86
	v_dual_mov_b32 v23, v87 :: v_dual_mov_b32 v26, v82
	v_dual_mov_b32 v27, v83 :: v_dual_mov_b32 v28, v84
	v_dual_mov_b32 v29, v85 :: v_dual_mov_b32 v30, v86
	v_dual_mov_b32 v31, v87 :: v_dual_mov_b32 v34, v82
	v_dual_mov_b32 v35, v83 :: v_dual_mov_b32 v36, v84
	v_dual_mov_b32 v37, v85 :: v_dual_mov_b32 v38, v86
	v_dual_mov_b32 v39, v87 :: v_dual_mov_b32 v42, v82
	v_dual_mov_b32 v43, v83 :: v_dual_mov_b32 v44, v84
	v_dual_mov_b32 v45, v85 :: v_dual_mov_b32 v46, v86
	v_dual_mov_b32 v47, v87 :: v_dual_mov_b32 v50, v82
	v_dual_mov_b32 v51, v83 :: v_dual_mov_b32 v52, v84
	v_dual_mov_b32 v53, v85 :: v_dual_mov_b32 v54, v86
	v_dual_mov_b32 v55, v87 :: v_dual_mov_b32 v58, v82
	v_dual_mov_b32 v59, v83 :: v_dual_mov_b32 v60, v84
	v_dual_mov_b32 v61, v85 :: v_dual_mov_b32 v62, v86
	v_dual_mov_b32 v63, v87 :: v_dual_mov_b32 v66, v82
	v_dual_mov_b32 v67, v83 :: v_dual_mov_b32 v68, v84
	v_dual_mov_b32 v69, v85 :: v_dual_mov_b32 v70, v86
	v_dual_mov_b32 v71, v87 :: v_dual_mov_b32 v74, v82
	v_dual_mov_b32 v75, v83 :: v_dual_mov_b32 v76, v84
	v_dual_mov_b32 v77, v85 :: v_dual_mov_b32 v78, v86
	v_mov_b32_e32 v79, v87
	v_dual_mov_b32 v93, v85 :: v_dual_mov_b32 v92, v84
	v_dual_mov_b32 v91, v83 :: v_dual_mov_b32 v90, v82
	v_dual_mov_b32 v89, v81 :: v_dual_mov_b32 v88, v80
	v_dual_mov_b32 v101, v85 :: v_dual_mov_b32 v100, v84
	v_dual_mov_b32 v99, v83 :: v_dual_mov_b32 v98, v82
	v_dual_mov_b32 v97, v81 :: v_dual_mov_b32 v96, v80
	v_dual_mov_b32 v109, v85 :: v_dual_mov_b32 v108, v84
	v_dual_mov_b32 v107, v83 :: v_dual_mov_b32 v106, v82
	v_dual_mov_b32 v105, v81 :: v_dual_mov_b32 v104, v80
	v_dual_mov_b32 v117, v85 :: v_dual_mov_b32 v116, v84
	v_dual_mov_b32 v115, v83 :: v_dual_mov_b32 v114, v82
	v_dual_mov_b32 v113, v81 :: v_dual_mov_b32 v112, v80
	v_dual_mov_b32 v125, v85 :: v_dual_mov_b32 v124, v84
	v_dual_mov_b32 v123, v83 :: v_dual_mov_b32 v122, v82
	v_dual_mov_b32 v121, v81 :: v_dual_mov_b32 v120, v80
	v_dual_mov_b32 v133, v85 :: v_dual_mov_b32 v132, v84
	v_dual_mov_b32 v131, v83 :: v_dual_mov_b32 v130, v82
	v_dual_mov_b32 v129, v81 :: v_dual_mov_b32 v128, v80
	s_mov_b32 s13, 0
	s_movk_i32 s18, 0x780
	s_mov_b32 s19, 0
	s_branch .LBB0_12
.LBB0_10:                               ;   in Loop: Header=BB0_12 Depth=1
	s_or_b32 exec_lo, exec_lo, s21
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB0_11:                               ;   in Loop: Header=BB0_12 Depth=1
	s_add_co_i32 s19, s19, 1
	v_add_nc_u32_e32 v224, 64, v224
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s19, 0x3fffffff
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s13, s13, 64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s20, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_139
.LBB0_12:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_19 Depth 2
                                        ;     Child Loop BB0_27 Depth 2
                                        ;     Child Loop BB0_35 Depth 2
                                        ;     Child Loop BB0_43 Depth 2
                                        ;     Child Loop BB0_55 Depth 2
                                        ;     Child Loop BB0_69 Depth 2
                                        ;     Child Loop BB0_130 Depth 2
	s_lshl_b32 s3, s19, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s3, s11
	s_cselect_b32 s20, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s20
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_11
; %bb.13:                               ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v82, 0 :: v_dual_add_nc_u32 v83, s3, v195
	v_mov_b32_e32 v81, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s11, v83
; %bb.14:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_u64_u32 v[81:82], null, 0x440, v83, v[187:188]
; %bb.15:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u64_e32 vcc_lo, 0, v[81:82]
	v_mov_b32_e32 v84, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_17
; %bb.16:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_d16_b16 v84, v[81:82], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v84, v84.l
.LBB0_17:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v81, s0, v81, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v82, null, 0, v82, s0
	v_mov_b32_e32 v85, v196
	s_mov_b64 s[4:5], 0
	s_mov_b32 s1, 0
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_b32_e32 v87, 0x780, v85
	s_and_b32 s0, s1, 16
	s_wait_loadcnt 0x0
	v_bfe_i32 v136, v86, 0, 8
	v_bfe_i32 v137, v86, 8, 8
	v_bfe_i32 v138, v86, 16, 8
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v87, v87, s0, v195
	v_ashrrev_i32_e32 v86, 24, v86
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_lshlrev_b32_e32 v87, 4, v87
	v_cvt_f32_i32_e32 v86, v86
	s_and_b32 s0, s4, 4
	v_fma_mixlo_f16 v136, v84, v136, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s0, 1
	v_fma_mixhi_f16 v136, v84, v137, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v87, s0, 0, v87
	v_fma_mixlo_f16 v137, v84, v138, 0
	v_fma_mixhi_f16 v137, v84, v86, 0
	v_add_nc_u32_e32 v85, 32, v85
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_add_co_i32 s1, s1, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 32
	ds_store_2addr_b32 v87, v136, v137 offset1:1
	s_cbranch_scc1 .LBB0_21
.LBB0_19:                               ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v86, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB0_18
; %bb.20:                               ;   in Loop: Header=BB0_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v86, s0, v81, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v87, null, s5, v82, s0
	global_load_b32 v86, v[86:87], off
	s_branch .LBB0_18
.LBB0_21:                               ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v82, 0 :: v_dual_add_nc_u32 v85, s3, v202
	v_mov_b32_e32 v81, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s11, v85
; %bb.22:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_u64_u32 v[81:82], null, 0x440, v85, v[187:188]
; %bb.23:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u64_e32 vcc_lo, 0, v[81:82]
	v_mov_b32_e32 v84, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_d16_b16 v84, v[81:82], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v84, v84.l
.LBB0_25:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v81, s0, v81, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v82, null, 0, v82, s0
	v_mov_b32_e32 v86, v196
	s_mov_b64 s[4:5], 0
	s_mov_b32 s1, 0
	s_branch .LBB0_27
.LBB0_26:                               ;   in Loop: Header=BB0_27 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_or_b32 v136, 0x780, v86, 32
	s_and_b32 s0, s1, 16
	s_wait_loadcnt 0x0
	v_bfe_i32 v137, v87, 0, 8
	v_bfe_i32 v138, v87, 8, 8
	v_bfe_i32 v139, v87, 16, 8
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v136, v136, s0, v195
	v_ashrrev_i32_e32 v87, 24, v87
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_lshlrev_b32_e32 v136, 4, v136
	v_cvt_f32_i32_e32 v87, v87
	s_and_b32 s0, s4, 4
	v_fma_mixlo_f16 v137, v84, v137, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s0, 1
	v_fma_mixhi_f16 v137, v84, v138, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v136, s0, 0, v136
	v_fma_mixlo_f16 v138, v84, v139, 0
	v_fma_mixhi_f16 v138, v84, v87, 0
	v_add_nc_u32_e32 v86, 32, v86
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_add_co_i32 s1, s1, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 32
	ds_store_2addr_b32 v136, v137, v138 offset1:1
	s_cbranch_scc0 .LBB0_29
.LBB0_27:                               ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v87, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB0_26
; %bb.28:                               ;   in Loop: Header=BB0_27 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v136, s0, v81, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v137, null, s5, v82, s0
	global_load_b32 v87, v[136:137], off
	s_branch .LBB0_26
.LBB0_29:                               ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v81, 0 :: v_dual_add_nc_u32 v84, s3, v203
	v_mov_b32_e32 v82, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s11, v84
; %bb.30:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_u64_u32 v[81:82], null, 0x440, v84, v[187:188]
; %bb.31:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u64_e32 vcc_lo, 0, v[81:82]
	v_mov_b32_e32 v84, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_d16_b16 v84, v[81:82], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v84, v84.l
.LBB0_33:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v81, s0, v81, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v82, null, 0, v82, s0
	v_mov_b32_e32 v86, v196
	s_mov_b64 s[4:5], 0
	s_mov_b32 s1, 0
	s_branch .LBB0_35
.LBB0_34:                               ;   in Loop: Header=BB0_35 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_or_b32 v136, 0x780, v86, 64
	s_and_b32 s0, s1, 16
	s_wait_loadcnt 0x0
	v_bfe_i32 v137, v87, 0, 8
	v_bfe_i32 v138, v87, 8, 8
	v_bfe_i32 v139, v87, 16, 8
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v136, v136, s0, v195
	v_ashrrev_i32_e32 v87, 24, v87
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_lshlrev_b32_e32 v136, 4, v136
	v_cvt_f32_i32_e32 v87, v87
	s_and_b32 s0, s4, 4
	v_fma_mixlo_f16 v137, v84, v137, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s0, 1
	v_fma_mixhi_f16 v137, v84, v138, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v136, s0, 0, v136
	v_fma_mixlo_f16 v138, v84, v139, 0
	v_fma_mixhi_f16 v138, v84, v87, 0
	v_add_nc_u32_e32 v86, 32, v86
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_add_co_i32 s1, s1, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 32
	ds_store_2addr_b32 v136, v137, v138 offset1:1
	s_cbranch_scc0 .LBB0_37
.LBB0_35:                               ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v87, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB0_34
; %bb.36:                               ;   in Loop: Header=BB0_35 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v136, s0, v81, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v137, null, s5, v82, s0
	global_load_b32 v87, v[136:137], off
	s_branch .LBB0_34
.LBB0_37:                               ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v81, 0 :: v_dual_add_nc_u32 v84, s3, v204
	v_mov_b32_e32 v82, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s11, v84
; %bb.38:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_u64_u32 v[81:82], null, 0x440, v84, v[187:188]
; %bb.39:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u64_e32 vcc_lo, 0, v[81:82]
	v_mov_b32_e32 v84, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_d16_b16 v84, v[81:82], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v84, v84.l
.LBB0_41:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v81, s0, v81, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v82, null, 0, v82, s0
	v_mov_b32_e32 v86, v196
	s_mov_b64 s[4:5], 0
	s_mov_b32 s1, 0
	s_branch .LBB0_43
.LBB0_42:                               ;   in Loop: Header=BB0_43 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_or_b32 v136, v86, s18, 0x60
	s_and_b32 s0, s1, 16
	s_wait_loadcnt 0x0
	v_bfe_i32 v137, v87, 0, 8
	v_bfe_i32 v138, v87, 8, 8
	v_bfe_i32 v139, v87, 16, 8
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v136, v136, s0, v195
	v_ashrrev_i32_e32 v87, 24, v87
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_lshlrev_b32_e32 v136, 4, v136
	v_cvt_f32_i32_e32 v87, v87
	s_and_b32 s0, s4, 4
	v_fma_mixlo_f16 v137, v84, v137, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s0, 1
	v_fma_mixhi_f16 v137, v84, v138, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v136, s0, 0, v136
	v_fma_mixlo_f16 v138, v84, v139, 0
	v_fma_mixhi_f16 v138, v84, v87, 0
	v_add_nc_u32_e32 v86, 32, v86
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_add_co_i32 s1, s1, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 32
	ds_store_2addr_b32 v136, v137, v138 offset1:1
	s_cbranch_scc0 .LBB0_45
.LBB0_43:                               ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v87, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB0_42
; %bb.44:                               ;   in Loop: Header=BB0_43 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v136, s0, v81, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v137, null, s5, v82, s0
	global_load_b32 v87, v[136:137], off
	s_branch .LBB0_42
.LBB0_45:                               ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v83, 0 :: v_dual_add_nc_u32 v86, v83, v195
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v81, 0
	v_mov_b32_e32 v82, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s11, v86
; %bb.46:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_u64_u32 v[81:82], null, 0x440, v86, v[189:190]
; %bb.47:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s11, v86
; %bb.48:                               ;   in Loop: Header=BB0_12 Depth=1
	v_or_b32_e32 v83, 1, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[83:84], null, 0x440, v83, v[189:190]
; %bb.49:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[81:82]
	v_dual_mov_b32 v86, 0 :: v_dual_mov_b32 v87, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_51
; %bb.50:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_d16_b16 v87, v[81:82], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v87, v87.l
.LBB0_51:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e64 s0, 0, v[83:84]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_53
; %bb.52:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_d16_b16 v86, v[83:84], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v86, v86.l
.LBB0_53:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_co_u32 v83, s1, v83, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v84, null, 0, v84, s1
	v_add_co_u32 v81, s1, v81, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v82, null, 0, v82, s1
	v_mov_b32_e32 v136, v198
	s_mov_b64 s[4:5], 0
	s_branch .LBB0_55
.LBB0_54:                               ;   in Loop: Header=BB0_55 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_b32_e32 v139, 0x1e0, v136
	s_wait_loadcnt 0x0
	v_bfe_i32 v140, v138, 0, 8
	v_and_or_b32 v141, s4, 12, v205
	v_bfe_i32 v142, v137, 0, 8
	v_bfe_i32 v143, v138, 8, 8
	v_bfe_i32 v144, v137, 16, 8
	v_cvt_f32_i32_e32 v140, v140
	v_or3_b32 v139, v141, v139, v201
	v_cvt_f32_i32_e32 v141, v142
	v_bfe_i32 v142, v137, 8, 8
	v_cvt_f32_i32_e32 v143, v143
	v_fma_mixlo_f16 v140, v87, v140, 0
	v_ashrrev_i32_e32 v137, 24, v137
	v_fma_mixhi_f16 v140, v86, v141, 0
	v_cvt_f32_i32_e32 v141, v142
	v_fma_mixlo_f16 v142, v87, v143, 0
	v_bfe_i32 v143, v138, 16, 8
	v_ashrrev_i32_e32 v138, 24, v138
	v_lshl_add_u32 v139, v139, 4, v206
	v_fma_mixhi_f16 v142, v86, v141, 0
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v141, v143
	v_cvt_f32_i32_e32 v143, v144
	v_cvt_f32_i32_e32 v138, v138
	v_add_nc_u32_e32 v139, 0x8000, v139
	v_add_nc_u32_e32 v136, 8, v136
	v_fma_mixlo_f16 v141, v87, v141, 0
	v_fma_mixhi_f16 v141, v86, v143, 0
	v_fma_mixlo_f16 v138, v87, v138, 0
	v_fma_mixhi_f16 v138, v86, v137, 0
	s_add_nc_u64 s[4:5], s[4:5], 4
	ds_store_2addr_b32 v139, v140, v142 offset1:4
	ds_store_2addr_b32 v139, v141, v138 offset0:8 offset1:12
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 32
	s_cbranch_scc1 .LBB0_59
.LBB0_55:                               ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v138, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB0_57
; %bb.56:                               ;   in Loop: Header=BB0_55 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v138, s1, v81, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v139, null, s5, v82, s1
	global_load_b32 v138, v[138:139], off
.LBB0_57:                               ;   in Loop: Header=BB0_55 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s0
	s_cbranch_execz .LBB0_54
; %bb.58:                               ;   in Loop: Header=BB0_55 Depth=2
	v_add_co_u32 v139, s1, v83, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v140, null, s5, v84, s1
	global_load_b32 v137, v[139:140], off
	s_branch .LBB0_54
.LBB0_59:                               ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v84, 0 :: v_dual_add_nc_u32 v85, v85, v202
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v82, 0
	v_mov_b32_e32 v81, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s11, v85
; %bb.60:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_u64_u32 v[81:82], null, 0x440, v85, v[189:190]
; %bb.61:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s11, v85
; %bb.62:                               ;   in Loop: Header=BB0_12 Depth=1
	v_or_b32_e32 v83, 1, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[83:84], null, 0x440, v83, v[189:190]
; %bb.63:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[81:82]
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v86, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_65
; %bb.64:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_d16_b16 v86, v[81:82], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v86, v86.l
.LBB0_65:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e64 s0, 0, v[83:84]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_67
; %bb.66:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_d16_b16 v85, v[83:84], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v85, v85.l
.LBB0_67:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_co_u32 v83, s1, v83, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v84, null, 0, v84, s1
	v_add_co_u32 v81, s1, v81, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v82, null, 0, v82, s1
	v_mov_b32_e32 v87, v198
	s_mov_b64 s[4:5], 0
	s_branch .LBB0_69
.LBB0_68:                               ;   in Loop: Header=BB0_69 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_b32_e32 v138, 0x1e0, v87
	s_wait_loadcnt 0x0
	v_bfe_i32 v139, v137, 0, 8
	v_and_or_b32 v140, s4, 12, v207
	v_bfe_i32 v141, v136, 0, 8
	v_bfe_i32 v142, v137, 8, 8
	v_bfe_i32 v143, v136, 16, 8
	v_cvt_f32_i32_e32 v139, v139
	v_or3_b32 v138, v140, v138, v201
	v_cvt_f32_i32_e32 v140, v141
	v_bfe_i32 v141, v136, 8, 8
	v_cvt_f32_i32_e32 v142, v142
	v_fma_mixlo_f16 v139, v86, v139, 0
	v_ashrrev_i32_e32 v136, 24, v136
	v_fma_mixhi_f16 v139, v85, v140, 0
	v_cvt_f32_i32_e32 v140, v141
	v_fma_mixlo_f16 v141, v86, v142, 0
	v_bfe_i32 v142, v137, 16, 8
	v_ashrrev_i32_e32 v137, 24, v137
	v_lshl_add_u32 v138, v138, 4, v206
	v_fma_mixhi_f16 v141, v85, v140, 0
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v140, v142
	v_cvt_f32_i32_e32 v142, v143
	v_cvt_f32_i32_e32 v137, v137
	v_add_nc_u32_e32 v138, 0x8000, v138
	v_add_nc_u32_e32 v87, 8, v87
	v_fma_mixlo_f16 v140, v86, v140, 0
	v_fma_mixhi_f16 v140, v85, v142, 0
	v_fma_mixlo_f16 v137, v86, v137, 0
	v_fma_mixhi_f16 v137, v85, v136, 0
	s_add_nc_u64 s[4:5], s[4:5], 4
	ds_store_2addr_b32 v138, v139, v141 offset1:4
	ds_store_2addr_b32 v138, v140, v137 offset0:8 offset1:12
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 32
	s_cbranch_scc0 .LBB0_73
.LBB0_69:                               ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB0_71
; %bb.70:                               ;   in Loop: Header=BB0_69 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v137, s1, v81, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v138, null, s5, v82, s1
	global_load_b32 v137, v[137:138], off
.LBB0_71:                               ;   in Loop: Header=BB0_69 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s0
	s_cbranch_execz .LBB0_68
; %bb.72:                               ;   in Loop: Header=BB0_69 Depth=2
	v_add_co_u32 v138, s1, v83, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v139, null, s5, v84, s1
	global_load_b32 v136, v[138:139], off
	s_branch .LBB0_68
.LBB0_73:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s21, s2
	s_cbranch_execz .LBB0_10
; %bb.74:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mov_b32_e32 v168, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v173, v168
	v_dual_mov_b32 v174, v168 :: v_dual_mov_b32 v175, v168
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[172:175], v[193:194], off
.LBB0_76:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_mov_b32_e32 v171, 0
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[168:171], v[193:194], off offset:32
.LBB0_78:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v181, 0
	v_dual_mov_b32 v180, 0 :: v_dual_mov_b32 v183, 0
	v_mov_b32_e32 v182, 0
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[180:183], v[193:194], off offset:64
.LBB0_80:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v178, 0
	v_mov_b32_e32 v179, 0
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[176:179], v[193:194], off offset:96
.LBB0_82:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[81:84], v199
	ds_load_b128 v[144:147], v199 offset:2048
	ds_load_b128 v[148:151], v199 offset:4096
	ds_load_b128 v[152:155], v199 offset:6144
	s_or_b32 s1, s3, 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s11
	s_cselect_b32 s0, -1, 0
	s_cmp_gt_i32 s1, s11
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[136:143], v[81:84], v[172:175], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[144:147], v[168:171], v[136:143]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[136:143], v[148:151], v[180:183], v[136:143]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[152:155], v[176:179], v[136:143]
	s_cbranch_scc1 .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[81:84], v199 offset:512
	ds_load_b128 v[152:155], v199 offset:2560
	ds_load_b128 v[156:159], v199 offset:4608
	ds_load_b128 v[160:163], v199 offset:6656
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[144:151], v[81:84], v[172:175], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[152:155], v[168:171], v[144:151]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[144:151], v[156:159], v[180:183], v[144:151]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[160:163], v[176:179], v[144:151]
	s_branch .LBB0_85
.LBB0_84:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mov_b32_e32 v87, v80
	v_dual_mov_b32 v81, v80 :: v_dual_mov_b32 v82, v80
	v_dual_mov_b32 v83, v80 :: v_dual_mov_b32 v84, v80
	v_dual_mov_b32 v85, v80 :: v_dual_mov_b32 v86, v80
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v151, v87
	v_dual_mov_b32 v147, v83 :: v_dual_mov_b32 v146, v82
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v149, v85 :: v_dual_mov_b32 v148, v84
	v_dual_mov_b32 v150, v86 :: v_dual_mov_b32 v145, v81
	v_mov_b32_e32 v144, v80
.LBB0_85:                               ;   in Loop: Header=BB0_12 Depth=1
	s_or_b32 s4, s3, 32
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s11
	s_cselect_b32 s1, -1, 0
	s_cmp_gt_i32 s4, s11
	s_cbranch_scc1 .LBB0_87
; %bb.86:                               ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[81:84], v199 offset:1024
	ds_load_b128 v[160:163], v199 offset:3072
	ds_load_b128 v[164:167], v199 offset:5120
	ds_load_b128 v[226:229], v199 offset:7168
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[152:159], v[81:84], v[172:175], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[160:163], v[168:171], v[152:159]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[152:159], v[164:167], v[180:183], v[152:159]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[226:229], v[176:179], v[152:159]
	s_branch .LBB0_88
.LBB0_87:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mov_b32_e32 v87, v80
	v_dual_mov_b32 v81, v80 :: v_dual_mov_b32 v82, v80
	v_dual_mov_b32 v83, v80 :: v_dual_mov_b32 v84, v80
	v_dual_mov_b32 v85, v80 :: v_dual_mov_b32 v86, v80
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v159, v87
	v_dual_mov_b32 v155, v83 :: v_dual_mov_b32 v154, v82
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v157, v85 :: v_dual_mov_b32 v156, v84
	v_dual_mov_b32 v158, v86 :: v_dual_mov_b32 v153, v81
	v_mov_b32_e32 v152, v80
.LBB0_88:                               ;   in Loop: Header=BB0_12 Depth=1
	s_or_b32 s4, s3, 48
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s11
	s_cselect_b32 s3, -1, 0
	s_cmp_gt_i32 s4, s11
	s_cbranch_scc1 .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[81:84], v208
	ds_load_b128 v[226:229], v209
	ds_load_b128 v[230:233], v210
	ds_load_b128 v[234:237], v211
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[160:167], v[81:84], v[172:175], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[226:229], v[168:171], v[160:167]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[160:167], v[230:233], v[180:183], v[160:167]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[234:237], v[176:179], v[160:167]
	s_branch .LBB0_91
.LBB0_90:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mov_b32_e32 v87, v80
	v_dual_mov_b32 v81, v80 :: v_dual_mov_b32 v82, v80
	v_dual_mov_b32 v83, v80 :: v_dual_mov_b32 v84, v80
	v_dual_mov_b32 v85, v80 :: v_dual_mov_b32 v86, v80
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v167, v87
	v_dual_mov_b32 v163, v83 :: v_dual_mov_b32 v162, v82
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v165, v85 :: v_dual_mov_b32 v164, v84
	v_dual_mov_b32 v166, v86 :: v_dual_mov_b32 v161, v81
	v_mov_b32_e32 v160, v80
.LBB0_91:                               ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_mov_b32_e32 v171, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s4, s17
	s_cbranch_execz .LBB0_93
; %bb.92:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[168:171], v[193:194], off offset:128
.LBB0_93:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v83, 0
	v_mov_b32_e32 v84, 0
	s_and_saveexec_b32 s4, s17
	s_cbranch_execz .LBB0_95
; %bb.94:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[81:84], v[193:194], off offset:160
.LBB0_95:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v179, 0
	v_mov_b32_e32 v178, 0
	s_and_saveexec_b32 s4, s17
	s_cbranch_execz .LBB0_97
; %bb.96:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[176:179], v[193:194], off offset:192
.LBB0_97:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, 0
	v_mov_b32_e32 v175, 0
	s_and_saveexec_b32 s4, s17
	s_cbranch_execz .LBB0_99
; %bb.98:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[172:175], v[193:194], off offset:224
.LBB0_99:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	ds_load_b128 v[180:183], v199 offset:8192
	ds_load_b128 v[226:229], v199 offset:10240
	ds_load_b128 v[230:233], v199 offset:12288
	ds_load_b128 v[234:237], v199 offset:14336
	v_cndmask_b32_e64 v85, 0, 1, s0
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[136:143], v[180:183], v[168:171], v[136:143]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[226:229], v[81:84], v[136:143]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[136:143], v[230:233], v[176:179], v[136:143]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[234:237], v[172:175], v[136:143]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_133
; %bb.100:                              ;   in Loop: Header=BB0_12 Depth=1
	v_cndmask_b32_e64 v86, 0, 1, s1
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_134
.LBB0_101:                              ;   in Loop: Header=BB0_12 Depth=1
	v_cndmask_b32_e64 v87, 0, 1, s3
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_103
.LBB0_102:                              ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[180:183], v212
	ds_load_b128 v[226:229], v213
	ds_load_b128 v[230:233], v214
	ds_load_b128 v[234:237], v215
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[160:167], v[180:183], v[168:171], v[160:167]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[226:229], v[81:84], v[160:167]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[160:167], v[230:233], v[176:179], v[160:167]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[234:237], v[172:175], v[160:167]
.LBB0_103:                              ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_mov_b32_e32 v171, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_105
; %bb.104:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[168:171], v[193:194], off offset:256
.LBB0_105:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v83, 0
	v_mov_b32_e32 v84, 0
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_107
; %bb.106:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[81:84], v[193:194], off offset:288
.LBB0_107:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v179, 0
	v_mov_b32_e32 v178, 0
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_109
; %bb.108:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[176:179], v[193:194], off offset:320
.LBB0_109:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, 0
	v_mov_b32_e32 v175, 0
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_111
; %bb.110:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[172:175], v[193:194], off offset:352
.LBB0_111:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[180:183], v199 offset:16384
	ds_load_b128 v[226:229], v199 offset:18432
	ds_load_b128 v[230:233], v199 offset:20480
	ds_load_b128 v[234:237], v199 offset:22528
	v_cmp_ne_u32_e32 vcc_lo, 1, v85
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[136:143], v[180:183], v[168:171], v[136:143]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[226:229], v[81:84], v[136:143]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[136:143], v[230:233], v[176:179], v[136:143]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[234:237], v[172:175], v[136:143]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_135
; %bb.112:                              ;   in Loop: Header=BB0_12 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v86
	s_cbranch_vccz .LBB0_136
.LBB0_113:                              ;   in Loop: Header=BB0_12 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v87
	s_cbranch_vccnz .LBB0_115
.LBB0_114:                              ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[180:183], v216
	ds_load_b128 v[226:229], v217
	ds_load_b128 v[230:233], v218
	ds_load_b128 v[234:237], v219
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[160:167], v[180:183], v[168:171], v[160:167]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[226:229], v[81:84], v[160:167]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[160:167], v[230:233], v[176:179], v[160:167]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[234:237], v[172:175], v[160:167]
.LBB0_115:                              ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_mov_b32_e32 v171, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_117
; %bb.116:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[168:171], v[193:194], off offset:384
.LBB0_117:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v83, 0
	v_mov_b32_e32 v84, 0
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_119
; %bb.118:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[81:84], v[193:194], off offset:416
.LBB0_119:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v179, 0
	v_mov_b32_e32 v178, 0
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_121
; %bb.120:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[176:179], v[193:194], off offset:448
.LBB0_121:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, 0
	v_mov_b32_e32 v175, 0
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_123
; %bb.122:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[172:175], v[193:194], off offset:480
.LBB0_123:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[180:183], v199 offset:24576
	ds_load_b128 v[226:229], v199 offset:26624
	ds_load_b128 v[230:233], v199 offset:28672
	ds_load_b128 v[234:237], v199 offset:30720
	v_cmp_ne_u32_e32 vcc_lo, 1, v85
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[136:143], v[180:183], v[168:171], v[136:143]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[226:229], v[81:84], v[136:143]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[136:143], v[230:233], v[176:179], v[136:143]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[234:237], v[172:175], v[136:143]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_137
; %bb.124:                              ;   in Loop: Header=BB0_12 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v86
	s_cbranch_vccz .LBB0_138
.LBB0_125:                              ;   in Loop: Header=BB0_12 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v87
	s_cbranch_vccnz .LBB0_127
.LBB0_126:                              ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[180:183], v220
	ds_load_b128 v[226:229], v221
	ds_load_b128 v[230:233], v222
	ds_load_b128 v[234:237], v223
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[160:167], v[180:183], v[168:171], v[160:167]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[226:229], v[81:84], v[160:167]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[160:167], v[230:233], v[176:179], v[160:167]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[234:237], v[172:175], v[160:167]
.LBB0_127:                              ;   in Loop: Header=BB0_12 Depth=1
	v_mov_b32_e32 v81, v199
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s22, 0
	s_mov_b32 s23, 0
	s_branch .LBB0_130
.LBB0_128:                              ;   in Loop: Header=BB0_130 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v83, s22, v224
	s_cmp_lt_u32 s23, 2
	v_cndmask_b32_e64 v168, v165, v157, s1
	s_cselect_b32 s10, -1, 0
	v_cndmask_b32_e64 v169, v164, v156, s1
	v_add_nc_u32_e32 v84, 2, v83
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s7, v83, v82
	v_cmp_lt_i32_e64 s8, v83, v82
	v_cndmask_b32_e64 v170, v163, v155, s1
	v_cndmask_b32_e64 v171, v161, v153, s1
	v_cmp_le_i32_e32 vcc_lo, v84, v82
	v_add_nc_u32_e32 v84, 3, v83
	v_cndmask_b32_e64 v172, v160, v152, s1
	v_cndmask_b32_e64 v173, v162, v154, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, v84, v82
	v_add_nc_u32_e32 v84, 4, v83
	v_cmp_le_i32_e64 s4, v84, v82
	v_add_nc_u32_e32 v84, 5, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cmp_le_i32_e64 s5, v84, v82
	v_add_nc_u32_e32 v84, 6, v83
	v_add_nc_u32_e32 v83, 7, v83
	v_cmp_le_i32_e64 s6, v84, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_i32_e64 s9, v83, v82
	v_cndmask_b32_e64 v82, v151, v143, s0
	v_cndmask_b32_e64 v83, v167, v159, s1
	v_cndmask_b32_e64 v84, v166, v158, s1
	s_or_b32 s1, s24, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v82, v83, v82, s10
	v_cndmask_b32_e64 v83, v150, v142, s0
	v_mul_f32_e32 v82, s16, v82
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v83, v84, v83, s10
	v_cndmask_b32_e64 v84, v149, v141, s0
	v_mul_f32_e32 v83, s16, v83
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v84, v168, v84, s10
	v_cndmask_b32_e64 v168, v148, v140, s0
	v_mul_f32_e32 v84, s16, v84
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v168, v169, v168, s10
	v_cndmask_b32_e64 v169, v147, v139, s0
	v_mul_f32_e32 v168, s16, v168
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v169, v170, v169, s10
	v_cndmask_b32_e64 v170, v145, v137, s0
	v_mul_f32_e32 v169, s16, v169
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v170, v171, v170, s10
	v_cndmask_b32_e64 v171, v144, v136, s0
	v_mul_f32_e32 v170, s16, v170
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v171, v172, v171, s10
	v_cndmask_b32_e64 v172, v146, v138, s0
	s_or_b32 s0, s24, s7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s17, s0
	v_mul_f32_e32 v171, s16, v171
	v_cndmask_b32_e64 v172, v173, v172, s10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v171, 0xff800000, v171, s0
	s_and_b32 s0, s17, s1
	v_mul_f32_e32 v172, s16, v172
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v170, 0xff800000, v170, s0
	s_or_b32 s0, s24, vcc_lo
	s_or_b32 s1, s24, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s17, s0
	s_or_b32 s0, s24, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v172, 0xff800000, v172, vcc_lo
	s_and_b32 vcc_lo, s17, s1
	s_or_b32 s1, s24, s5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v169, 0xff800000, v169, vcc_lo
	s_and_b32 vcc_lo, s17, s0
	v_max3_num_f32 v173, v171, 0xff800000, v170
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v168, 0xff800000, v168, vcc_lo
	s_and_b32 vcc_lo, s17, s1
	s_or_b32 s0, s24, s6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v84, 0xff800000, v84, vcc_lo
	s_and_b32 vcc_lo, s17, s0
	s_or_b32 s1, s24, s9
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v83, 0xff800000, v83, vcc_lo
	v_max3_num_f32 v173, v173, v172, v169
	s_and_b32 vcc_lo, s17, s1
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v174, 0xff800000, v82, vcc_lo
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v225
	v_max3_num_f32 v82, v173, v168, v84
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v82, v82, v83, v174
	ds_bpermute_b32 v173, v197, v82
	s_wait_dscnt 0x0
	v_max3_num_f32 v82, v225, v82, v173
	v_dual_sub_f32 v169, v169, v82 :: v_dual_sub_f32 v170, v170, v82
	v_sub_f32_e32 v83, v83, v82
	v_cmp_eq_f32_e64 s0, 0xff800000, v82
	v_sub_f32_e32 v173, v225, v82
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v169, 0x3fb8aa3b, v169 :: v_dual_mul_f32 v170, 0x3fb8aa3b, v170
	v_mul_f32_e32 v83, 0x3fb8aa3b, v83
	v_sub_f32_e32 v171, v171, v82
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v173, 0x3fb8aa3b, v173 :: v_dual_sub_f32 v168, v168, v82
	v_exp_f32_e32 v170, v170
	v_dual_sub_f32 v84, v84, v82 :: v_dual_mov_b32 v225, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v173, v173
	v_mul_f32_e32 v171, 0x3fb8aa3b, v171
	v_exp_f32_e32 v169, v169
	v_exp_f32_e32 v83, v83
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v177, v170, 0, s0
	v_sub_f32_e32 v170, v174, v82
	v_exp_f32_e32 v171, v171
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v182, 0, v173, vcc_lo
	v_mul_f32_e32 v168, 0x3fb8aa3b, v168
	v_mul_f32_e32 v84, 0x3fb8aa3b, v84
	v_mul_f32_e32 v170, 0x3fb8aa3b, v170
	v_cndmask_b32_e64 v179, v169, 0, s0
	v_mul_f32_e32 v135, v135, v182
	v_mul_f32_e32 v133, v133, v182
	v_mul_f32_e32 v131, v131, v182
	v_exp_f32_e32 v170, v170
	v_cndmask_b32_e64 v176, v171, 0, s0
	v_exp_f32_e32 v171, v168
	v_exp_f32_e32 v84, v84
	v_cndmask_b32_e64 v83, v83, 0, s0
	v_cvt_f16_f32_e64 v168.h, v177
	v_cvt_f16_f32_e64 v168.l, v176
	v_cvt_f16_f32_e64 v169.h, v179
	v_dual_mul_f32 v134, v134, v182 :: v_dual_mul_f32 v129, v129, v182
	v_cndmask_b32_e64 v181, v170, 0, s0
	v_sub_f32_e32 v172, v172, v82
	v_cndmask_b32_e64 v180, v171, 0, s0
	v_cndmask_b32_e64 v84, v84, 0, s0
	v_cvt_f16_f32_e64 v171.l, v83
	v_cvt_f16_f32_e64 v171.h, v181
	v_mul_f32_e32 v172, 0x3fb8aa3b, v172
	v_cvt_f16_f32_e64 v170.l, v180
	v_cvt_f16_f32_e64 v170.h, v84
	v_dual_mul_f32 v132, v132, v182 :: v_dual_mul_f32 v127, v127, v182
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v172, v172
	v_dual_mul_f32 v130, v130, v182 :: v_dual_mul_f32 v125, v125, v182
	v_dual_mul_f32 v128, v128, v182 :: v_dual_mul_f32 v123, v123, v182
	v_dual_mul_f32 v126, v126, v182 :: v_dual_mul_f32 v121, v121, v182
	v_dual_mul_f32 v124, v124, v182 :: v_dual_mul_f32 v119, v119, v182
	v_cndmask_b32_e64 v178, v172, 0, s0
	ds_load_b128 v[172:175], v81 offset:32768
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v122, v122, v182 :: v_dual_mul_f32 v117, v117, v182
	v_cvt_f16_f32_e64 v169.l, v178
	v_dual_mul_f32 v120, v120, v182 :: v_dual_mul_f32 v115, v115, v182
	v_dual_mul_f32 v118, v118, v182 :: v_dual_mul_f32 v113, v113, v182
	v_dual_mul_f32 v116, v116, v182 :: v_dual_mul_f32 v111, v111, v182
	v_dual_mul_f32 v114, v114, v182 :: v_dual_mul_f32 v109, v109, v182
	v_dual_mul_f32 v112, v112, v182 :: v_dual_mul_f32 v107, v107, v182
	v_dual_mul_f32 v110, v110, v182 :: v_dual_mul_f32 v105, v105, v182
	v_dual_mul_f32 v108, v108, v182 :: v_dual_mul_f32 v103, v103, v182
	v_dual_mul_f32 v106, v106, v182 :: v_dual_mul_f32 v101, v101, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[128:135], v[172:175], v[168:171], v[128:135]
	ds_load_b128 v[172:175], v81 offset:33280
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v104, v104, v182 :: v_dual_mul_f32 v99, v99, v182
	v_dual_mul_f32 v102, v102, v182 :: v_dual_mul_f32 v97, v97, v182
	v_dual_mul_f32 v100, v100, v182 :: v_dual_mul_f32 v95, v95, v182
	v_dual_mul_f32 v98, v98, v182 :: v_dual_mul_f32 v93, v93, v182
	v_dual_mul_f32 v96, v96, v182 :: v_dual_mul_f32 v91, v91, v182
	v_dual_mul_f32 v94, v94, v182 :: v_dual_mul_f32 v89, v89, v182
	v_dual_mul_f32 v92, v92, v182 :: v_dual_mul_f32 v79, v79, v182
	v_dual_mul_f32 v90, v90, v182 :: v_dual_mul_f32 v77, v77, v182
	v_dual_mul_f32 v88, v88, v182 :: v_dual_mul_f32 v75, v75, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[120:127], v[172:175], v[168:171], v[120:127]
	ds_load_b128 v[172:175], v81 offset:33792
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v78, v78, v182 :: v_dual_mul_f32 v73, v73, v182
	v_dual_mul_f32 v76, v76, v182 :: v_dual_mul_f32 v71, v71, v182
	v_dual_mul_f32 v74, v74, v182 :: v_dual_mul_f32 v69, v69, v182
	v_dual_mul_f32 v72, v72, v182 :: v_dual_mul_f32 v67, v67, v182
	v_dual_mul_f32 v70, v70, v182 :: v_dual_mul_f32 v65, v65, v182
	v_dual_mul_f32 v68, v68, v182 :: v_dual_mul_f32 v63, v63, v182
	v_dual_mul_f32 v66, v66, v182 :: v_dual_mul_f32 v61, v61, v182
	v_dual_mul_f32 v64, v64, v182 :: v_dual_mul_f32 v59, v59, v182
	v_dual_mul_f32 v62, v62, v182 :: v_dual_mul_f32 v57, v57, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[112:119], v[172:175], v[168:171], v[112:119]
	ds_load_b128 v[172:175], v81 offset:34304
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v60, v60, v182 :: v_dual_mul_f32 v55, v55, v182
	v_dual_mul_f32 v58, v58, v182 :: v_dual_mul_f32 v53, v53, v182
	v_dual_mul_f32 v56, v56, v182 :: v_dual_mul_f32 v51, v51, v182
	v_dual_mul_f32 v54, v54, v182 :: v_dual_mul_f32 v49, v49, v182
	v_dual_mul_f32 v52, v52, v182 :: v_dual_mul_f32 v47, v47, v182
	v_dual_mul_f32 v50, v50, v182 :: v_dual_mul_f32 v45, v45, v182
	v_dual_mul_f32 v48, v48, v182 :: v_dual_mul_f32 v43, v43, v182
	v_dual_mul_f32 v46, v46, v182 :: v_dual_mul_f32 v41, v41, v182
	v_dual_mul_f32 v44, v44, v182 :: v_dual_mul_f32 v39, v39, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[104:111], v[172:175], v[168:171], v[104:111]
	ds_load_b128 v[172:175], v81 offset:34816
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v42, v42, v182 :: v_dual_mul_f32 v37, v37, v182
	v_dual_mul_f32 v40, v40, v182 :: v_dual_mul_f32 v35, v35, v182
	v_dual_mul_f32 v38, v38, v182 :: v_dual_mul_f32 v33, v33, v182
	v_dual_mul_f32 v36, v36, v182 :: v_dual_mul_f32 v31, v31, v182
	v_dual_mul_f32 v34, v34, v182 :: v_dual_mul_f32 v29, v29, v182
	v_dual_mul_f32 v32, v32, v182 :: v_dual_mul_f32 v27, v27, v182
	v_dual_mul_f32 v30, v30, v182 :: v_dual_mul_f32 v25, v25, v182
	v_dual_mul_f32 v28, v28, v182 :: v_dual_mul_f32 v23, v23, v182
	v_dual_mul_f32 v26, v26, v182 :: v_dual_mul_f32 v21, v21, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[96:103], v[172:175], v[168:171], v[96:103]
	ds_load_b128 v[172:175], v81 offset:35328
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v24, v24, v182 :: v_dual_mul_f32 v19, v19, v182
	v_dual_mul_f32 v22, v22, v182 :: v_dual_mul_f32 v17, v17, v182
	v_dual_mul_f32 v20, v20, v182 :: v_dual_mul_f32 v15, v15, v182
	v_dual_mul_f32 v18, v18, v182 :: v_dual_mul_f32 v13, v13, v182
	v_dual_mul_f32 v16, v16, v182 :: v_dual_mul_f32 v11, v11, v182
	v_dual_mul_f32 v14, v14, v182 :: v_dual_mul_f32 v9, v9, v182
	v_dual_mul_f32 v12, v12, v182 :: v_dual_mul_f32 v7, v7, v182
	v_dual_mul_f32 v10, v10, v182 :: v_dual_mul_f32 v5, v5, v182
	v_dual_mul_f32 v8, v8, v182 :: v_dual_mul_f32 v3, v3, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[88:95], v[172:175], v[168:171], v[88:95]
	ds_load_b128 v[172:175], v81 offset:35840
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v6, v6, v182 :: v_dual_mul_f32 v1, v1, v182
	v_mul_f32_e32 v4, v4, v182
	v_mul_f32_e32 v2, v2, v182
	v_mul_f32_e32 v0, v0, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[72:79], v[172:175], v[168:171], v[72:79]
	ds_load_b128 v[172:175], v81 offset:36352
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[64:71], v[172:175], v[168:171], v[64:71]
	ds_load_b128 v[172:175], v81 offset:36864
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[56:63], v[172:175], v[168:171], v[56:63]
	ds_load_b128 v[172:175], v81 offset:37376
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[48:55], v[172:175], v[168:171], v[48:55]
	ds_load_b128 v[172:175], v81 offset:37888
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[40:47], v[172:175], v[168:171], v[40:47]
	ds_load_b128 v[172:175], v81 offset:38400
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[32:39], v[172:175], v[168:171], v[32:39]
	ds_load_b128 v[172:175], v81 offset:38912
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[24:31], v[172:175], v[168:171], v[24:31]
	ds_load_b128 v[172:175], v81 offset:39424
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[16:23], v[172:175], v[168:171], v[16:23]
	ds_load_b128 v[172:175], v81 offset:39936
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[8:15], v[172:175], v[168:171], v[8:15]
	ds_load_b128 v[172:175], v81 offset:40448
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[172:175], v[168:171], v[0:7]
	v_add_f32_e32 v168, v176, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v168, v178, v168
	v_add_f32_e32 v168, v179, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v168, v180, v168
	v_add_f32_e32 v84, v84, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v83, v83, v84
	v_add_f32_e32 v83, v181, v83
	ds_bpermute_b32 v84, v197, v83
	s_wait_dscnt 0x0
	v_add_f32_e32 v83, v83, v84
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v83, v200, v182
	v_mov_b32_e32 v200, v83
.LBB0_129:                              ;   in Loop: Header=BB0_130 Depth=2
	v_add_nc_u32_e32 v81, 0x2000, v81
	s_add_co_i32 s22, s22, 16
	s_add_co_i32 s23, s23, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s22, 64
	s_cbranch_scc0 .LBB0_10
.LBB0_130:                              ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s22, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s22, 32
	s_cselect_b32 s1, -1, 0
	s_cmp_eq_u32 s22, 16
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v82, v87, v86, s1
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v82, v82, v85, vcc_lo
	v_and_b32_e32 v82, 1, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 1, v82
	s_or_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_129
; %bb.131:                              ;   in Loop: Header=BB0_130 Depth=2
	s_add_co_i32 s3, s13, s22
	v_mov_b32_e32 v82, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s3, 15
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s12
	s_cselect_b32 s24, -1, 0
	s_cmp_gt_i32 s3, s12
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s3, s17
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB0_128
; %bb.132:                              ;   in Loop: Header=BB0_130 Depth=2
	global_load_b32 v82, v[191:192], off
	s_branch .LBB0_128
.LBB0_133:                              ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[180:183], v199 offset:8704
	ds_load_b128 v[226:229], v199 offset:10752
	ds_load_b128 v[230:233], v199 offset:12800
	ds_load_b128 v[234:237], v199 offset:14848
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[144:151], v[180:183], v[168:171], v[144:151]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[226:229], v[81:84], v[144:151]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[144:151], v[230:233], v[176:179], v[144:151]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[234:237], v[172:175], v[144:151]
	v_cndmask_b32_e64 v86, 0, 1, s1
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_101
.LBB0_134:                              ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[180:183], v199 offset:9216
	ds_load_b128 v[226:229], v199 offset:11264
	ds_load_b128 v[230:233], v199 offset:13312
	ds_load_b128 v[234:237], v199 offset:15360
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[152:159], v[180:183], v[168:171], v[152:159]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[226:229], v[81:84], v[152:159]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[152:159], v[230:233], v[176:179], v[152:159]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[234:237], v[172:175], v[152:159]
	v_cndmask_b32_e64 v87, 0, 1, s3
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_102
	s_branch .LBB0_103
.LBB0_135:                              ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[180:183], v199 offset:16896
	ds_load_b128 v[226:229], v199 offset:18944
	ds_load_b128 v[230:233], v199 offset:20992
	ds_load_b128 v[234:237], v199 offset:23040
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[144:151], v[180:183], v[168:171], v[144:151]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[226:229], v[81:84], v[144:151]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[144:151], v[230:233], v[176:179], v[144:151]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[234:237], v[172:175], v[144:151]
	v_cmp_ne_u32_e32 vcc_lo, 1, v86
	s_cbranch_vccnz .LBB0_113
.LBB0_136:                              ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[180:183], v199 offset:17408
	ds_load_b128 v[226:229], v199 offset:19456
	ds_load_b128 v[230:233], v199 offset:21504
	ds_load_b128 v[234:237], v199 offset:23552
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[152:159], v[180:183], v[168:171], v[152:159]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[226:229], v[81:84], v[152:159]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[152:159], v[230:233], v[176:179], v[152:159]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[234:237], v[172:175], v[152:159]
	v_cmp_ne_u32_e32 vcc_lo, 1, v87
	s_cbranch_vccz .LBB0_114
	s_branch .LBB0_115
.LBB0_137:                              ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[180:183], v199 offset:25088
	ds_load_b128 v[226:229], v199 offset:27136
	ds_load_b128 v[230:233], v199 offset:29184
	ds_load_b128 v[234:237], v199 offset:31232
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[144:151], v[180:183], v[168:171], v[144:151]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[226:229], v[81:84], v[144:151]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[144:151], v[230:233], v[176:179], v[144:151]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[234:237], v[172:175], v[144:151]
	v_cmp_ne_u32_e32 vcc_lo, 1, v86
	s_cbranch_vccnz .LBB0_125
.LBB0_138:                              ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[180:183], v199 offset:25600
	ds_load_b128 v[226:229], v199 offset:27648
	ds_load_b128 v[230:233], v199 offset:29696
	ds_load_b128 v[234:237], v199 offset:31744
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[152:159], v[180:183], v[168:171], v[152:159]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[226:229], v[81:84], v[152:159]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[152:159], v[230:233], v[176:179], v[152:159]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[234:237], v[172:175], v[152:159]
	v_cmp_ne_u32_e32 vcc_lo, 1, v87
	s_cbranch_vccz .LBB0_126
	s_branch .LBB0_127
.LBB0_139:
	s_and_saveexec_b32 s0, s17
	s_cbranch_execz .LBB0_141
; %bb.140:
	v_div_scale_f32 v82, null, v200, v200, 1.0
	v_div_scale_f32 v84, vcc_lo, 1.0, v200, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v83, v82
	v_fma_f32 v80, -v82, v83, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v83, v80, v83
	v_mul_f32_e32 v85, v84, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v80, -v82, v85, v84
	v_fmac_f32_e32 v85, v80, v83
	v_mad_co_u64_u32 v[80:81], null, 0x1800, v184, v[186:187]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v82, -v82, v85, v84
	v_or_b32_e32 v80, v80, v185
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v82, v82, v83, v85
	v_cmp_lt_f32_e32 vcc_lo, 0, v200
	v_div_fixup_f32 v82, v82, v200, 1.0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v138, 0, v82 :: v_dual_mov_b32 v81, 0
	v_mul_f32_e32 v82, v130, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	v_dual_mul_f32 v83, v131, v138 :: v_dual_mul_f32 v120, v120, v138
	v_dual_mul_f32 v85, v133, v138 :: v_dual_mul_f32 v122, v122, v138
	v_mul_f32_e32 v86, v134, v138
	v_add_co_u32 v136, vcc_lo, s14, v80
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v137, null, s15, v81, vcc_lo
	v_dual_mul_f32 v80, v128, v138 :: v_dual_mul_f32 v81, v129, v138
	v_dual_mul_f32 v84, v132, v138 :: v_dual_mul_f32 v87, v135, v138
	v_dual_mul_f32 v124, v124, v138 :: v_dual_mul_f32 v121, v121, v138
	v_dual_mul_f32 v126, v126, v138 :: v_dual_mul_f32 v123, v123, v138
	v_dual_mul_f32 v112, v112, v138 :: v_dual_mul_f32 v125, v125, v138
	v_dual_mul_f32 v114, v114, v138 :: v_dual_mul_f32 v127, v127, v138
	v_dual_mul_f32 v116, v116, v138 :: v_dual_mul_f32 v113, v113, v138
	v_dual_mul_f32 v118, v118, v138 :: v_dual_mul_f32 v115, v115, v138
	v_dual_mul_f32 v104, v104, v138 :: v_dual_mul_f32 v107, v107, v138
	v_dual_mul_f32 v96, v96, v138 :: v_dual_mul_f32 v109, v109, v138
	v_dual_mul_f32 v98, v98, v138 :: v_dual_mul_f32 v97, v97, v138
	v_mul_f32_e32 v99, v99, v138
	v_dual_mul_f32 v75, v75, v138 :: v_dual_mul_f32 v64, v64, v138
	v_dual_mul_f32 v77, v77, v138 :: v_dual_mul_f32 v66, v66, v138
	v_dual_mul_f32 v65, v65, v138 :: v_dual_mul_f32 v70, v70, v138
	v_dual_mul_f32 v67, v67, v138 :: v_dual_mul_f32 v56, v56, v138
	v_dual_mul_f32 v51, v51, v138 :: v_dual_mul_f32 v40, v40, v138
	v_dual_mul_f32 v53, v53, v138 :: v_dual_mul_f32 v42, v42, v138
	v_dual_mul_f32 v41, v41, v138 :: v_dual_mul_f32 v46, v46, v138
	v_dual_mul_f32 v43, v43, v138 :: v_dual_mul_f32 v32, v32, v138
	v_dual_mul_f32 v27, v27, v138 :: v_dual_mul_f32 v16, v16, v138
	v_dual_mul_f32 v29, v29, v138 :: v_dual_mul_f32 v18, v18, v138
	v_dual_mul_f32 v17, v17, v138 :: v_dual_mul_f32 v22, v22, v138
	v_dual_mul_f32 v19, v19, v138 :: v_dual_mul_f32 v8, v8, v138
	v_dual_mul_f32 v117, v117, v138 :: v_dual_mul_f32 v106, v106, v138
	v_dual_mul_f32 v119, v119, v138 :: v_dual_mul_f32 v108, v108, v138
	v_dual_mul_f32 v105, v105, v138 :: v_dual_mul_f32 v110, v110, v138
	v_mul_f32_e32 v111, v111, v138
	s_clause 0x7
	global_store_b128 v[136:137], v[80:83], off
	global_store_b128 v[136:137], v[84:87], off offset:16
	global_store_b128 v[136:137], v[120:123], off offset:64
	global_store_b128 v[136:137], v[124:127], off offset:80
	global_store_b128 v[136:137], v[112:115], off offset:128
	global_store_b128 v[136:137], v[116:119], off offset:144
	global_store_b128 v[136:137], v[104:107], off offset:192
	global_store_b128 v[136:137], v[108:111], off offset:208
	v_dual_mul_f32 v80, v100, v138 :: v_dual_mul_f32 v81, v101, v138
	v_mul_f32_e32 v86, v90, v138
	v_dual_mul_f32 v82, v102, v138 :: v_dual_mul_f32 v83, v103, v138
	v_dual_mul_f32 v79, v79, v138 :: v_dual_mul_f32 v68, v68, v138
	v_dual_mul_f32 v69, v69, v138 :: v_dual_mul_f32 v58, v58, v138
	v_dual_mul_f32 v71, v71, v138 :: v_dual_mul_f32 v60, v60, v138
	v_dual_mul_f32 v55, v55, v138 :: v_dual_mul_f32 v44, v44, v138
	v_dual_mul_f32 v45, v45, v138 :: v_dual_mul_f32 v34, v34, v138
	v_dual_mul_f32 v47, v47, v138 :: v_dual_mul_f32 v36, v36, v138
	v_dual_mul_f32 v31, v31, v138 :: v_dual_mul_f32 v20, v20, v138
	v_dual_mul_f32 v21, v21, v138 :: v_dual_mul_f32 v10, v10, v138
	v_dual_mul_f32 v23, v23, v138 :: v_dual_mul_f32 v12, v12, v138
	v_dual_mul_f32 v84, v88, v138 :: v_dual_mul_f32 v85, v89, v138
	v_dual_mul_f32 v90, v94, v138 :: v_dual_mul_f32 v87, v91, v138
	v_dual_mul_f32 v72, v72, v138 :: v_dual_mul_f32 v57, v57, v138
	v_dual_mul_f32 v62, v62, v138 :: v_dual_mul_f32 v59, v59, v138
	v_dual_mul_f32 v48, v48, v138 :: v_dual_mul_f32 v33, v33, v138
	v_dual_mul_f32 v38, v38, v138 :: v_dual_mul_f32 v35, v35, v138
	v_dual_mul_f32 v24, v24, v138 :: v_dual_mul_f32 v9, v9, v138
	v_dual_mul_f32 v14, v14, v138 :: v_dual_mul_f32 v11, v11, v138
	v_mul_f32_e32 v0, v0, v138
	v_dual_mul_f32 v88, v92, v138 :: v_dual_mul_f32 v89, v93, v138
	v_dual_mul_f32 v74, v74, v138 :: v_dual_mul_f32 v91, v95, v138
	v_dual_mul_f32 v76, v76, v138 :: v_dual_mul_f32 v61, v61, v138
	v_dual_mul_f32 v50, v50, v138 :: v_dual_mul_f32 v63, v63, v138
	v_dual_mul_f32 v52, v52, v138 :: v_dual_mul_f32 v37, v37, v138
	v_dual_mul_f32 v26, v26, v138 :: v_dual_mul_f32 v39, v39, v138
	v_dual_mul_f32 v28, v28, v138 :: v_dual_mul_f32 v13, v13, v138
	v_dual_mul_f32 v2, v2, v138 :: v_dual_mul_f32 v15, v15, v138
	v_dual_mul_f32 v4, v4, v138 :: v_dual_mul_f32 v73, v73, v138
	v_dual_mul_f32 v78, v78, v138 :: v_dual_mul_f32 v49, v49, v138
	v_dual_mul_f32 v54, v54, v138 :: v_dual_mul_f32 v25, v25, v138
	v_dual_mul_f32 v30, v30, v138 :: v_dual_mul_f32 v1, v1, v138
	v_dual_mul_f32 v6, v6, v138 :: v_dual_mul_f32 v3, v3, v138
	s_clause 0x11
	global_store_b128 v[136:137], v[96:99], off offset:256
	global_store_b128 v[136:137], v[80:83], off offset:272
	global_store_b128 v[136:137], v[84:87], off offset:320
	global_store_b128 v[136:137], v[88:91], off offset:336
	global_store_b128 v[136:137], v[72:75], off offset:384
	global_store_b128 v[136:137], v[76:79], off offset:400
	global_store_b128 v[136:137], v[64:67], off offset:448
	global_store_b128 v[136:137], v[68:71], off offset:464
	global_store_b128 v[136:137], v[56:59], off offset:512
	global_store_b128 v[136:137], v[60:63], off offset:528
	global_store_b128 v[136:137], v[48:51], off offset:576
	global_store_b128 v[136:137], v[52:55], off offset:592
	global_store_b128 v[136:137], v[40:43], off offset:640
	global_store_b128 v[136:137], v[44:47], off offset:656
	global_store_b128 v[136:137], v[32:35], off offset:704
	global_store_b128 v[136:137], v[36:39], off offset:720
	global_store_b128 v[136:137], v[24:27], off offset:768
	global_store_b128 v[136:137], v[28:31], off offset:784
	v_mul_f32_e32 v5, v5, v138
	v_mul_f32_e32 v7, v7, v138
	s_clause 0x5
	global_store_b128 v[136:137], v[16:19], off offset:832
	global_store_b128 v[136:137], v[20:23], off offset:848
	global_store_b128 v[136:137], v[8:11], off offset:896
	global_store_b128 v[136:137], v[12:15], off offset:912
	global_store_b128 v[136:137], v[0:3], off offset:960
	global_store_b128 v[136:137], v[4:7], off offset:976
.LBB0_141:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	attention_q8_0_fa2_gqa_gfx1201, .Lfunc_end0-attention_q8_0_fa2_gqa_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_q8_0_fa2_gqa_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 60
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
		.amdhsa_next_free_vgpr 238
		.amdhsa_next_free_sgpr 25
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-attention_q8_0_fa2_gqa_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_q8_0_fa2_gqa_gfx1201.num_vgpr, 238
	.set .Lattention_q8_0_fa2_gqa_gfx1201.num_agpr, 0
	.set .Lattention_q8_0_fa2_gqa_gfx1201.numbered_sgpr, 25
	.set .Lattention_q8_0_fa2_gqa_gfx1201.num_named_barrier, 0
	.set .Lattention_q8_0_fa2_gqa_gfx1201.private_seg_size, 0
	.set .Lattention_q8_0_fa2_gqa_gfx1201.uses_vcc, 1
	.set .Lattention_q8_0_fa2_gqa_gfx1201.uses_flat_scratch, 0
	.set .Lattention_q8_0_fa2_gqa_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_q8_0_fa2_gqa_gfx1201.has_recursion, 0
	.set .Lattention_q8_0_fa2_gqa_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10136
; TotalNumSgprs: 27
; NumVgprs: 238
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 27
; NumVGPRsForWavesPerEU: 238
; Occupancy: 6
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_fa2_q_preconvert_gfx1201 ; -- Begin function attention_fa2_q_preconvert_gfx1201
	.globl	attention_fa2_q_preconvert_gfx1201
	.p2align	8
	.type	attention_fa2_q_preconvert_gfx1201,@function
attention_fa2_q_preconvert_gfx1201:     ; @attention_fa2_q_preconvert_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x20
	v_lshrrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v1, ttmp9, 2, v1
	s_wait_kmcnt 0x0
	s_mul_i32 s2, s2, 24
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e32 vcc_lo, s2, v1
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB1_2
; %bb.1:
	v_mul_hi_i32 v2, 0x2aaaaaab, v1
	s_load_b128 s[0:3], s[0:1], 0x0
	v_mov_b32_e32 v9, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v11, v9
	v_lshrrev_b32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v2, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v2, v2, v3
	v_mul_lo_u32 v3, v2, 24
	v_mul_lo_u32 v8, 0x1800, v2
	v_lshlrev_b32_e32 v2, 3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v12, 0xf8, v2
	v_sub_nc_u32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v4, 2, v12
	v_lshlrev_b32_e32 v10, 8, v1
	v_lshlrev_b64_e32 v[0:1], 2, v[8:9]
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]
	v_lshlrev_b32_e32 v12, 1, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[2:3], 2, v[10:11]
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]
	s_wait_kmcnt 0x0
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, s2, v8
	s_clause 0x1
	global_load_b128 v[0:3], v[4:5], off
	global_load_b128 v[4:7], v[4:5], off offset:16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s3, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_loadcnt 0x1
	v_cvt_f16_f32_e32 v0.l, v0
	v_cvt_f16_f32_e32 v0.h, v1
	v_cvt_f16_f32_e32 v1.l, v2
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v2.l, v4
	v_add_co_u32 v4, vcc_lo, v8, v12
	v_cvt_f16_f32_e32 v1.h, v3
	v_cvt_f16_f32_e32 v2.h, v5
	v_cvt_f16_f32_e32 v3.l, v6
	v_cvt_f16_f32_e32 v3.h, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v9, vcc_lo
	global_store_b128 v[4:5], v[0:3], off
.LBB1_2:
	s_endpgm
.Lfunc_end1:
	.size	attention_fa2_q_preconvert_gfx1201, .Lfunc_end1-attention_fa2_q_preconvert_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fa2_q_preconvert_gfx1201
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 13
		.amdhsa_next_free_sgpr 4
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-attention_fa2_q_preconvert_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fa2_q_preconvert_gfx1201.num_vgpr, 13
	.set .Lattention_fa2_q_preconvert_gfx1201.num_agpr, 0
	.set .Lattention_fa2_q_preconvert_gfx1201.numbered_sgpr, 4
	.set .Lattention_fa2_q_preconvert_gfx1201.num_named_barrier, 0
	.set .Lattention_fa2_q_preconvert_gfx1201.private_seg_size, 0
	.set .Lattention_fa2_q_preconvert_gfx1201.uses_vcc, 1
	.set .Lattention_fa2_q_preconvert_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fa2_q_preconvert_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fa2_q_preconvert_gfx1201.has_recursion, 0
	.set .Lattention_fa2_q_preconvert_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 388
; TotalNumSgprs: 6
; NumVgprs: 13
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 6
; NumVGPRsForWavesPerEU: 13
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_q8_0_fa2_gqa_partial_gfx1201 ; -- Begin function attention_q8_0_fa2_gqa_partial_gfx1201
	.globl	attention_q8_0_fa2_gqa_partial_gfx1201
	.p2align	8
	.type	attention_q8_0_fa2_gqa_partial_gfx1201,@function
attention_q8_0_fa2_gqa_partial_gfx1201: ; @attention_q8_0_fa2_gqa_partial_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x28
	s_load_b64 s[16:17], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s4, s17, -9
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s4, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB2_145
; %bb.1:
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB2_145
; %bb.2:
	s_lshl_b32 s4, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s4, s7
	s_cbranch_scc1 .LBB2_145
; %bb.3:
	s_lshr_b32 s18, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s18, s17
	s_cbranch_scc1 .LBB2_145
; %bb.4:
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v184, 0 :: v_dual_mov_b32 v1, 0
	v_mov_b32_e32 v2, 0
	v_lshrrev_b32_e32 v197, 3, v0
	v_mov_b32_e32 v186, 0
	s_mov_b32 s19, 0
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB2_6
; %bb.5:
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_or_b32 v184, v0, 7, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 6, v1
	v_cmp_gt_i32_e32 vcc_lo, s7, v184
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[1:2], null, s3, 6, v[1:2]
	v_mul_lo_u32 v2, 0x1800, v184
	s_and_b32 s19, vcc_lo, exec_lo
	v_and_or_b32 v186, v197, 1, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v1, v186, 8, v2
	v_mov_b32_e32 v2, 0
.LBB2_6:
	s_or_b32 exec_lo, exec_lo, s5
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x20
	v_and_b32_e32 v6, 31, v0
	v_mov_b32_e32 v4, -1
	v_bfrev_b32_e32 v5, -2
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_u32_e32 8, v6
	s_cbranch_execz .LBB2_10
; %bb.7:
	v_or_b32_e32 v3, s4, v6
	v_bfrev_b32_e32 v5, -2
	v_mov_b32_e32 v4, -1
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s7, v3
	s_cbranch_execz .LBB2_9
; %bb.8:
	v_ashrrev_i32_e32 v4, 31, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, vcc_lo, s0, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v4, null, s1, v4, vcc_lo
	global_load_b32 v4, v[3:4], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v5, v4
.LBB2_9:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB2_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_lshrrev_b32_e32 v198, 4, v6
	v_lshl_add_u32 v202, v6, 4, 0
	v_dual_mov_b32 v128, 0 :: v_dual_lshlrev_b32 v199, 8, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_xor_b32_e32 v7, 16, v3
	v_xor_b32_e32 v14, 8, v3
	v_xor_b32_e32 v15, 4, v3
	v_lshrrev_b32_e32 v9, 1, v0
	v_dual_mov_b32 v187, 0xff800000 :: v_dual_lshlrev_b32 v200, 6, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	v_or_b32_e32 v10, 0x80, v0
	v_or_b32_e32 v11, 0x100, v0
	v_or_b32_e32 v12, 0x180, v0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v3, v7 :: v_dual_and_b32 v8, 7, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v14
	v_dual_mov_b32 v130, v128 :: v_dual_lshlrev_b32 v13, 3, v0
	v_dual_mov_b32 v129, v128 :: v_dual_lshlrev_b32 v0, 4, v0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v14, v3, v14 :: v_dual_lshlrev_b32 v201, 2, v7
	v_cmp_gt_u32_e32 vcc_lo, 32, v15
	v_dual_mov_b32 v131, v128 :: v_dual_and_b32 v42, 16, v6
	ds_bpermute_b32 v6, v201, v4
	ds_bpermute_b32 v7, v201, v5
	v_lshlrev_b32_e32 v14, 2, v14
	v_ashrrev_i32_e32 v185, 31, v184
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, v3, v15, vcc_lo
	v_dual_mov_b32 v132, v128 :: v_dual_and_b32 v203, 16, v9
	v_mov_b32_e32 v133, v128
	v_lshlrev_b64_e32 v[40:41], 2, v[184:185]
	v_dual_mov_b32 v134, v128 :: v_dual_lshlrev_b32 v185, 3, v198
	v_xor_b32_e32 v16, 2, v3
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	v_xor_b32_e32 v17, 1, v3
	v_lshrrev_b32_e32 v205, 3, v11
	v_lshl_or_b32 v11, v11, 4, 0x600
	v_cmp_gt_u32_e32 vcc_lo, 32, v16
	v_and_b32_e32 v207, 0x200, v13
	s_cvt_f32_u32 s4, s17
	s_add_co_i32 s5, s17, 0x1ff
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, v3, v16, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v43, vcc_lo, s8, v1
	ds_bpermute_b32 v6, v14, v4
	ds_bpermute_b32 v7, v14, v5
	v_or_b32_e32 v14, 0x600, v0
	v_mov_b32_e32 v135, v128
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s9, v2, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v17
	v_add_nc_u32_e32 v210, 0, v14
	v_lshlrev_b32_e32 v15, 2, v15
	v_lshlrev_b32_e32 v1, 2, v16
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s6, s4
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v3, v3, v17 :: v_dual_add_nc_u32 v212, 0, v11
	s_and_b32 s5, s5, 0xffff
	v_lshl_or_b32 v8, s3, 3, v8
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s5, s5
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v9, 12, v9
	v_or_b32_e32 v18, 0x2600, v0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s3, s5, s6
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v7
	v_or_b32_e32 v19, 0x2e00, v0
	s_wait_alu depctr_sa_sdst(0)
	s_trunc_f32 s3, s3
	v_or_b32_e32 v20, 0x3600, v0
	ds_bpermute_b32 v24, v15, v4
	ds_bpermute_b32 v15, v15, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, s3, 0x80000000
	v_or_b32_e32 v21, 0x3e00, v0
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s5, s6, s4
	v_or_b32_e32 v22, 0x4600, v0
	v_or_b32_e32 v6, 0x4e00, v0
	v_or_b32_e32 v7, 0x5600, v0
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s5, 31
	v_or_b32_e32 v23, 0x5e00, v0
	v_or_b32_e32 v25, 0x6600, v0
	v_or_b32_e32 v26, 0x6e00, v0
	v_or_b32_e32 v27, 0x7600, v0
	v_or_b32_e32 v0, 0x7e00, v0
	s_cvt_u32_f32 s3, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s5, s4
	v_add_co_u32 v193, vcc_lo, v43, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v194, null, 0, v44, vcc_lo
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v4, v24
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v5, v15
	s_add_co_ci_u32 s3, s3, 0
	v_add_co_u32 v195, vcc_lo, s0, v40
	ds_bpermute_b32 v5, v1, v2
	ds_bpermute_b32 v1, v1, v4
	v_add_nc_u32_e32 v219, 0, v6
	v_add_nc_u32_e32 v220, 0, v7
	v_add_nc_u32_e32 v225, 0, v0
	v_lshrrev_b32_e32 v204, 3, v10
	v_lshrrev_b32_e32 v206, 3, v12
	v_lshl_or_b32 v12, v12, 4, 0x600
	v_mad_co_u64_u32 v[191:192], null, v8, 34, s[12:13]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s3, 0xffff
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v196, null, s1, v41, vcc_lo
	v_mad_co_u64_u32 v[189:190], null, v8, 34, s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s11, s18, s3
	s_movk_i32 s21, 0x780
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s12, s11, 6
	s_add_co_i32 s22, s11, s3
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v226, s12, v185
	v_mov_b32_e32 v188, 0
	s_wait_dscnt 0x1
	v_max_i32_e32 v45, v2, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v46, v4, v1
	ds_bpermute_b32 v47, v3, v45
	ds_bpermute_b32 v48, v3, v46
	v_mov_b32_e32 v0, v128
	v_add_nc_u32_e32 v208, 0, v9
	v_dual_mov_b32 v1, v129 :: v_dual_add_nc_u32 v214, 0, v18
	v_add_nc_u32_e32 v215, 0, v19
	v_dual_mov_b32 v3, v131 :: v_dual_add_nc_u32 v216, 0, v20
	v_add_nc_u32_e32 v217, 0, v21
	v_dual_mov_b32 v5, v133 :: v_dual_add_nc_u32 v218, 0, v22
	v_add_nc_u32_e32 v221, 0, v23
	v_add_nc_u32_e32 v222, 0, v25
	v_add_nc_u32_e32 v223, 0, v26
	v_add_nc_u32_e32 v224, 0, v27
	v_mov_b32_e32 v16, v128
	v_dual_mov_b32 v24, v128 :: v_dual_lshlrev_b32 v13, 3, v10
	v_mov_b32_e32 v2, v130
	s_wait_dscnt 0x1
	v_max_i32_e32 v42, v45, v47
	s_wait_dscnt 0x0
	v_min_i32_e32 v43, v46, v48
	v_lshl_or_b32 v10, v10, 4, 0x600
	v_dual_mov_b32 v17, v129 :: v_dual_mov_b32 v32, v128
	v_mov_b32_e32 v27, v131
	v_readfirstlane_b32 s13, v42
	v_readfirstlane_b32 s20, v43
	v_dual_mov_b32 v40, v128 :: v_dual_mov_b32 v35, v131
	v_dual_mov_b32 v48, v128 :: v_dual_mov_b32 v43, v131
	v_dual_mov_b32 v56, v128 :: v_dual_mov_b32 v51, v131
	v_dual_mov_b32 v64, v128 :: v_dual_mov_b32 v59, v131
	v_dual_mov_b32 v72, v128 :: v_dual_mov_b32 v67, v131
	v_dual_mov_b32 v80, v128 :: v_dual_mov_b32 v75, v131
	v_dual_mov_b32 v88, v128 :: v_dual_mov_b32 v83, v131
	v_dual_mov_b32 v96, v128 :: v_dual_mov_b32 v91, v131
	v_dual_mov_b32 v104, v128 :: v_dual_and_b32 v209, 0x600, v13
	v_add_nc_u32_e32 v211, 0, v10
	v_dual_mov_b32 v8, v128 :: v_dual_add_nc_u32 v213, 0, v12
	v_dual_mov_b32 v99, v131 :: v_dual_mov_b32 v112, v128
	v_dual_mov_b32 v107, v131 :: v_dual_mov_b32 v120, v128
	v_mov_b32_e32 v4, v132
	v_dual_mov_b32 v6, v134 :: v_dual_mov_b32 v7, v135
	v_dual_mov_b32 v9, v129 :: v_dual_mov_b32 v10, v130
	v_dual_mov_b32 v11, v131 :: v_dual_mov_b32 v12, v132
	v_dual_mov_b32 v13, v133 :: v_dual_mov_b32 v28, v132
	v_dual_mov_b32 v14, v134 :: v_dual_mov_b32 v15, v135
	v_mov_b32_e32 v30, v134
	v_dual_mov_b32 v18, v130 :: v_dual_mov_b32 v19, v131
	v_mov_b32_e32 v34, v130
	v_dual_mov_b32 v20, v132 :: v_dual_mov_b32 v21, v133
	v_mov_b32_e32 v36, v132
	v_dual_mov_b32 v22, v134 :: v_dual_mov_b32 v23, v135
	v_dual_mov_b32 v38, v134 :: v_dual_mov_b32 v25, v129
	v_dual_mov_b32 v26, v130 :: v_dual_mov_b32 v29, v133
	v_dual_mov_b32 v42, v130 :: v_dual_mov_b32 v31, v135
	v_dual_mov_b32 v44, v132 :: v_dual_mov_b32 v33, v129
	v_dual_mov_b32 v46, v134 :: v_dual_mov_b32 v37, v133
	v_dual_mov_b32 v50, v130 :: v_dual_mov_b32 v39, v135
	v_dual_mov_b32 v52, v132 :: v_dual_mov_b32 v41, v129
	v_dual_mov_b32 v54, v134 :: v_dual_mov_b32 v45, v133
	v_dual_mov_b32 v58, v130 :: v_dual_mov_b32 v47, v135
	v_dual_mov_b32 v60, v132 :: v_dual_mov_b32 v49, v129
	v_dual_mov_b32 v62, v134 :: v_dual_mov_b32 v53, v133
	v_dual_mov_b32 v66, v130 :: v_dual_mov_b32 v55, v135
	v_dual_mov_b32 v68, v132 :: v_dual_mov_b32 v57, v129
	v_dual_mov_b32 v70, v134 :: v_dual_mov_b32 v61, v133
	v_dual_mov_b32 v74, v130 :: v_dual_mov_b32 v63, v135
	v_dual_mov_b32 v76, v132 :: v_dual_mov_b32 v65, v129
	v_dual_mov_b32 v78, v134 :: v_dual_mov_b32 v69, v133
	v_dual_mov_b32 v82, v130 :: v_dual_mov_b32 v71, v135
	v_dual_mov_b32 v84, v132 :: v_dual_mov_b32 v73, v129
	v_dual_mov_b32 v86, v134 :: v_dual_mov_b32 v77, v133
	v_dual_mov_b32 v90, v130 :: v_dual_mov_b32 v79, v135
	v_dual_mov_b32 v92, v132 :: v_dual_mov_b32 v81, v129
	v_dual_mov_b32 v94, v134 :: v_dual_mov_b32 v85, v133
	v_dual_mov_b32 v98, v130 :: v_dual_mov_b32 v87, v135
	v_dual_mov_b32 v100, v132 :: v_dual_mov_b32 v89, v129
	v_dual_mov_b32 v102, v134 :: v_dual_mov_b32 v93, v133
	v_dual_mov_b32 v106, v130 :: v_dual_mov_b32 v95, v135
	v_dual_mov_b32 v108, v132 :: v_dual_mov_b32 v97, v129
	v_dual_mov_b32 v110, v134 :: v_dual_mov_b32 v101, v133
	v_dual_mov_b32 v114, v130 :: v_dual_mov_b32 v103, v135
	v_dual_mov_b32 v116, v132 :: v_dual_mov_b32 v105, v129
	v_dual_mov_b32 v118, v134 :: v_dual_mov_b32 v109, v133
	v_dual_mov_b32 v122, v130 :: v_dual_mov_b32 v111, v135
	v_dual_mov_b32 v124, v132 :: v_dual_mov_b32 v113, v129
	v_dual_mov_b32 v126, v134 :: v_dual_mov_b32 v115, v131
	v_mov_b32_e32 v117, v133
	v_mov_b32_e32 v119, v135
	v_mov_b32_e32 v121, v129
	v_mov_b32_e32 v123, v131
	v_mov_b32_e32 v125, v133
	v_mov_b32_e32 v127, v135
	s_branch .LBB2_13
.LBB2_11:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s24
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB2_12:                               ;   in Loop: Header=BB2_13 Depth=1
	s_add_co_i32 s11, s11, 1
	v_add_nc_u32_e32 v226, 64, v226
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s11, s22
	s_cselect_b32 s0, -1, 0
	s_xor_b32 s1, s23, -1
	s_add_co_i32 s12, s12, 64
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_140
.LBB2_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_20 Depth 2
                                        ;     Child Loop BB2_28 Depth 2
                                        ;     Child Loop BB2_36 Depth 2
                                        ;     Child Loop BB2_44 Depth 2
                                        ;     Child Loop BB2_56 Depth 2
                                        ;     Child Loop BB2_70 Depth 2
                                        ;     Child Loop BB2_131 Depth 2
	s_lshl_b32 s3, s11, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s3, s13
	s_cselect_b32 s23, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s23
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_12
; %bb.14:                               ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v130, 0 :: v_dual_add_nc_u32 v131, s3, v197
	v_mov_b32_e32 v129, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s13, v131
; %bb.15:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mad_co_i64_i32 v[129:130], null, 0x440, v131, v[189:190]
; %bb.16:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u64_e32 vcc_lo, 0, v[129:130]
	v_mov_b32_e32 v132, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_18
; %bb.17:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_d16_b16 v132, v[129:130], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v132, v132.l
.LBB2_18:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v129, s0, v129, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, 0, v130, s0
	v_mov_b32_e32 v133, v199
	s_mov_b64 s[4:5], 0
	s_mov_b32 s1, 0
	s_branch .LBB2_20
.LBB2_19:                               ;   in Loop: Header=BB2_20 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_b32_e32 v135, 0x780, v133
	s_and_b32 s0, s1, 16
	s_wait_loadcnt 0x0
	v_bfe_i32 v136, v134, 0, 8
	v_bfe_i32 v137, v134, 8, 8
	v_bfe_i32 v138, v134, 16, 8
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v135, v135, s0, v197
	v_ashrrev_i32_e32 v134, 24, v134
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_lshlrev_b32_e32 v135, 4, v135
	v_cvt_f32_i32_e32 v134, v134
	s_and_b32 s0, s4, 4
	v_fma_mixlo_f16 v136, v132, v136, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s0, 1
	v_fma_mixhi_f16 v136, v132, v137, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v135, s0, 0, v135
	v_fma_mixlo_f16 v137, v132, v138, 0
	v_fma_mixhi_f16 v137, v132, v134, 0
	v_add_nc_u32_e32 v133, 32, v133
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_add_co_i32 s1, s1, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 32
	ds_store_2addr_b32 v135, v136, v137 offset1:1
	s_cbranch_scc1 .LBB2_22
.LBB2_20:                               ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v134, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB2_19
; %bb.21:                               ;   in Loop: Header=BB2_20 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v134, s0, v129, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v135, null, s5, v130, s0
	global_load_b32 v134, v[134:135], off
	s_branch .LBB2_19
.LBB2_22:                               ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v130, 0 :: v_dual_add_nc_u32 v133, s3, v204
	v_mov_b32_e32 v129, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s13, v133
; %bb.23:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mad_co_i64_i32 v[129:130], null, 0x440, v133, v[189:190]
; %bb.24:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u64_e32 vcc_lo, 0, v[129:130]
	v_mov_b32_e32 v132, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_26
; %bb.25:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_d16_b16 v132, v[129:130], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v132, v132.l
.LBB2_26:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v129, s0, v129, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, 0, v130, s0
	v_mov_b32_e32 v134, v199
	s_mov_b64 s[4:5], 0
	s_mov_b32 s1, 0
	s_branch .LBB2_28
.LBB2_27:                               ;   in Loop: Header=BB2_28 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_or_b32 v136, 0x780, v134, 32
	s_and_b32 s0, s1, 16
	s_wait_loadcnt 0x0
	v_bfe_i32 v137, v135, 0, 8
	v_bfe_i32 v138, v135, 8, 8
	v_bfe_i32 v139, v135, 16, 8
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v136, v136, s0, v197
	v_ashrrev_i32_e32 v135, 24, v135
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_lshlrev_b32_e32 v136, 4, v136
	v_cvt_f32_i32_e32 v135, v135
	s_and_b32 s0, s4, 4
	v_fma_mixlo_f16 v137, v132, v137, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s0, 1
	v_fma_mixhi_f16 v137, v132, v138, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v136, s0, 0, v136
	v_fma_mixlo_f16 v138, v132, v139, 0
	v_fma_mixhi_f16 v138, v132, v135, 0
	v_add_nc_u32_e32 v134, 32, v134
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_add_co_i32 s1, s1, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 32
	ds_store_2addr_b32 v136, v137, v138 offset1:1
	s_cbranch_scc0 .LBB2_30
.LBB2_28:                               ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v135, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB2_27
; %bb.29:                               ;   in Loop: Header=BB2_28 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v135, s0, v129, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v136, null, s5, v130, s0
	global_load_b32 v135, v[135:136], off
	s_branch .LBB2_27
.LBB2_30:                               ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v129, 0 :: v_dual_add_nc_u32 v132, s3, v205
	v_mov_b32_e32 v130, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s13, v132
; %bb.31:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mad_co_i64_i32 v[129:130], null, 0x440, v132, v[189:190]
; %bb.32:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u64_e32 vcc_lo, 0, v[129:130]
	v_mov_b32_e32 v132, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_34
; %bb.33:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_d16_b16 v132, v[129:130], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v132, v132.l
.LBB2_34:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v129, s0, v129, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, 0, v130, s0
	v_mov_b32_e32 v134, v199
	s_mov_b64 s[4:5], 0
	s_mov_b32 s1, 0
	s_branch .LBB2_36
.LBB2_35:                               ;   in Loop: Header=BB2_36 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_or_b32 v136, 0x780, v134, 64
	s_and_b32 s0, s1, 16
	s_wait_loadcnt 0x0
	v_bfe_i32 v137, v135, 0, 8
	v_bfe_i32 v138, v135, 8, 8
	v_bfe_i32 v139, v135, 16, 8
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v136, v136, s0, v197
	v_ashrrev_i32_e32 v135, 24, v135
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_lshlrev_b32_e32 v136, 4, v136
	v_cvt_f32_i32_e32 v135, v135
	s_and_b32 s0, s4, 4
	v_fma_mixlo_f16 v137, v132, v137, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s0, 1
	v_fma_mixhi_f16 v137, v132, v138, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v136, s0, 0, v136
	v_fma_mixlo_f16 v138, v132, v139, 0
	v_fma_mixhi_f16 v138, v132, v135, 0
	v_add_nc_u32_e32 v134, 32, v134
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_add_co_i32 s1, s1, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 32
	ds_store_2addr_b32 v136, v137, v138 offset1:1
	s_cbranch_scc0 .LBB2_38
.LBB2_36:                               ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v135, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB2_35
; %bb.37:                               ;   in Loop: Header=BB2_36 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v135, s0, v129, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v136, null, s5, v130, s0
	global_load_b32 v135, v[135:136], off
	s_branch .LBB2_35
.LBB2_38:                               ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v129, 0 :: v_dual_add_nc_u32 v132, s3, v206
	v_mov_b32_e32 v130, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s13, v132
; %bb.39:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mad_co_i64_i32 v[129:130], null, 0x440, v132, v[189:190]
; %bb.40:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u64_e32 vcc_lo, 0, v[129:130]
	v_mov_b32_e32 v132, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_42
; %bb.41:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_d16_b16 v132, v[129:130], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v132, v132.l
.LBB2_42:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v129, s0, v129, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, 0, v130, s0
	v_mov_b32_e32 v134, v199
	s_mov_b64 s[4:5], 0
	s_mov_b32 s1, 0
	s_branch .LBB2_44
.LBB2_43:                               ;   in Loop: Header=BB2_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_or_b32 v136, v134, s21, 0x60
	s_and_b32 s0, s1, 16
	s_wait_loadcnt 0x0
	v_bfe_i32 v137, v135, 0, 8
	v_bfe_i32 v138, v135, 8, 8
	v_bfe_i32 v139, v135, 16, 8
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v136, v136, s0, v197
	v_ashrrev_i32_e32 v135, 24, v135
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_lshlrev_b32_e32 v136, 4, v136
	v_cvt_f32_i32_e32 v135, v135
	s_and_b32 s0, s4, 4
	v_fma_mixlo_f16 v137, v132, v137, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s0, 1
	v_fma_mixhi_f16 v137, v132, v138, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v136, s0, 0, v136
	v_fma_mixlo_f16 v138, v132, v139, 0
	v_fma_mixhi_f16 v138, v132, v135, 0
	v_add_nc_u32_e32 v134, 32, v134
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_add_co_i32 s1, s1, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 32
	ds_store_2addr_b32 v136, v137, v138 offset1:1
	s_cbranch_scc0 .LBB2_46
.LBB2_44:                               ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v135, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB2_43
; %bb.45:                               ;   in Loop: Header=BB2_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v135, s0, v129, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v136, null, s5, v130, s0
	global_load_b32 v135, v[135:136], off
	s_branch .LBB2_43
.LBB2_46:                               ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v131, 0 :: v_dual_add_nc_u32 v134, v131, v197
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v129, 0
	v_mov_b32_e32 v130, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s13, v134
; %bb.47:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mad_co_i64_i32 v[129:130], null, 0x440, v134, v[191:192]
; %bb.48:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s13, v134
; %bb.49:                               ;   in Loop: Header=BB2_13 Depth=1
	v_or_b32_e32 v131, 1, v134
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[131:132], null, 0x440, v131, v[191:192]
; %bb.50:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[129:130]
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_52
; %bb.51:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_d16_b16 v135, v[129:130], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v135, v135.l
.LBB2_52:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e64 s0, 0, v[131:132]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB2_54
; %bb.53:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_d16_b16 v134, v[131:132], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v134, v134.l
.LBB2_54:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_co_u32 v131, s1, v131, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v132, null, 0, v132, s1
	v_add_co_u32 v129, s1, v129, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, 0, v130, s1
	v_mov_b32_e32 v136, v200
	s_mov_b64 s[4:5], 0
	s_branch .LBB2_56
.LBB2_55:                               ;   in Loop: Header=BB2_56 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_b32_e32 v139, 0x1e0, v136
	s_wait_loadcnt 0x0
	v_bfe_i32 v140, v138, 0, 8
	v_and_or_b32 v141, s4, 12, v207
	v_bfe_i32 v142, v137, 0, 8
	v_bfe_i32 v143, v138, 8, 8
	v_bfe_i32 v144, v137, 16, 8
	v_cvt_f32_i32_e32 v140, v140
	v_or3_b32 v139, v141, v139, v203
	v_cvt_f32_i32_e32 v141, v142
	v_bfe_i32 v142, v137, 8, 8
	v_cvt_f32_i32_e32 v143, v143
	v_fma_mixlo_f16 v140, v135, v140, 0
	v_ashrrev_i32_e32 v137, 24, v137
	v_fma_mixhi_f16 v140, v134, v141, 0
	v_cvt_f32_i32_e32 v141, v142
	v_fma_mixlo_f16 v142, v135, v143, 0
	v_bfe_i32 v143, v138, 16, 8
	v_ashrrev_i32_e32 v138, 24, v138
	v_lshl_add_u32 v139, v139, 4, v208
	v_fma_mixhi_f16 v142, v134, v141, 0
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v141, v143
	v_cvt_f32_i32_e32 v143, v144
	v_cvt_f32_i32_e32 v138, v138
	v_add_nc_u32_e32 v139, 0x8000, v139
	v_add_nc_u32_e32 v136, 8, v136
	v_fma_mixlo_f16 v141, v135, v141, 0
	v_fma_mixhi_f16 v141, v134, v143, 0
	v_fma_mixlo_f16 v138, v135, v138, 0
	v_fma_mixhi_f16 v138, v134, v137, 0
	s_add_nc_u64 s[4:5], s[4:5], 4
	ds_store_2addr_b32 v139, v140, v142 offset1:4
	ds_store_2addr_b32 v139, v141, v138 offset0:8 offset1:12
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 32
	s_cbranch_scc1 .LBB2_60
.LBB2_56:                               ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v138, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB2_58
; %bb.57:                               ;   in Loop: Header=BB2_56 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v138, s1, v129, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v139, null, s5, v130, s1
	global_load_b32 v138, v[138:139], off
.LBB2_58:                               ;   in Loop: Header=BB2_56 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s0
	s_cbranch_execz .LBB2_55
; %bb.59:                               ;   in Loop: Header=BB2_56 Depth=2
	v_add_co_u32 v139, s1, v131, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v140, null, s5, v132, s1
	global_load_b32 v137, v[139:140], off
	s_branch .LBB2_55
.LBB2_60:                               ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v132, 0 :: v_dual_add_nc_u32 v133, v133, v204
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v130, 0
	v_mov_b32_e32 v129, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s13, v133
; %bb.61:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mad_co_i64_i32 v[129:130], null, 0x440, v133, v[191:192]
; %bb.62:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s13, v133
; %bb.63:                               ;   in Loop: Header=BB2_13 Depth=1
	v_or_b32_e32 v131, 1, v133
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[131:132], null, 0x440, v131, v[191:192]
; %bb.64:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[129:130]
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v134, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_66
; %bb.65:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_d16_b16 v134, v[129:130], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v134, v134.l
.LBB2_66:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e64 s0, 0, v[131:132]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB2_68
; %bb.67:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_d16_b16 v133, v[131:132], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v133, v133.l
.LBB2_68:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_co_u32 v131, s1, v131, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v132, null, 0, v132, s1
	v_add_co_u32 v129, s1, v129, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, 0, v130, s1
	v_mov_b32_e32 v135, v200
	s_mov_b64 s[4:5], 0
	s_branch .LBB2_70
.LBB2_69:                               ;   in Loop: Header=BB2_70 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_b32_e32 v138, 0x1e0, v135
	s_wait_loadcnt 0x0
	v_bfe_i32 v139, v137, 0, 8
	v_and_or_b32 v140, s4, 12, v209
	v_bfe_i32 v141, v136, 0, 8
	v_bfe_i32 v142, v137, 8, 8
	v_bfe_i32 v143, v136, 16, 8
	v_cvt_f32_i32_e32 v139, v139
	v_or3_b32 v138, v140, v138, v203
	v_cvt_f32_i32_e32 v140, v141
	v_bfe_i32 v141, v136, 8, 8
	v_cvt_f32_i32_e32 v142, v142
	v_fma_mixlo_f16 v139, v134, v139, 0
	v_ashrrev_i32_e32 v136, 24, v136
	v_fma_mixhi_f16 v139, v133, v140, 0
	v_cvt_f32_i32_e32 v140, v141
	v_fma_mixlo_f16 v141, v134, v142, 0
	v_bfe_i32 v142, v137, 16, 8
	v_ashrrev_i32_e32 v137, 24, v137
	v_lshl_add_u32 v138, v138, 4, v208
	v_fma_mixhi_f16 v141, v133, v140, 0
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v140, v142
	v_cvt_f32_i32_e32 v142, v143
	v_cvt_f32_i32_e32 v137, v137
	v_add_nc_u32_e32 v138, 0x8000, v138
	v_add_nc_u32_e32 v135, 8, v135
	v_fma_mixlo_f16 v140, v134, v140, 0
	v_fma_mixhi_f16 v140, v133, v142, 0
	v_fma_mixlo_f16 v137, v134, v137, 0
	v_fma_mixhi_f16 v137, v133, v136, 0
	s_add_nc_u64 s[4:5], s[4:5], 4
	ds_store_2addr_b32 v138, v139, v141 offset1:4
	ds_store_2addr_b32 v138, v140, v137 offset0:8 offset1:12
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 32
	s_cbranch_scc0 .LBB2_74
.LBB2_70:                               ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB2_72
; %bb.71:                               ;   in Loop: Header=BB2_70 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v137, s1, v129, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v138, null, s5, v130, s1
	global_load_b32 v137, v[137:138], off
.LBB2_72:                               ;   in Loop: Header=BB2_70 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s0
	s_cbranch_execz .LBB2_69
; %bb.73:                               ;   in Loop: Header=BB2_70 Depth=2
	v_add_co_u32 v138, s1, v131, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v139, null, s5, v132, s1
	global_load_b32 v136, v[138:139], off
	s_branch .LBB2_69
.LBB2_74:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s24, s2
	s_cbranch_execz .LBB2_11
; %bb.75:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v168, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v173, v168
	v_dual_mov_b32 v174, v168 :: v_dual_mov_b32 v175, v168
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_77
; %bb.76:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[172:175], v[193:194], off
.LBB2_77:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_mov_b32_e32 v171, 0
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_79
; %bb.78:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[168:171], v[193:194], off offset:32
.LBB2_79:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v181, 0
	v_dual_mov_b32 v180, 0 :: v_dual_mov_b32 v183, 0
	v_mov_b32_e32 v182, 0
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_81
; %bb.80:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[180:183], v[193:194], off offset:64
.LBB2_81:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v178, 0
	v_mov_b32_e32 v179, 0
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_83
; %bb.82:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[176:179], v[193:194], off offset:96
.LBB2_83:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[129:132], v202
	ds_load_b128 v[144:147], v202 offset:2048
	ds_load_b128 v[148:151], v202 offset:4096
	ds_load_b128 v[152:155], v202 offset:6144
	s_or_b32 s1, s3, 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s13
	s_cselect_b32 s0, -1, 0
	s_cmp_gt_i32 s1, s13
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[136:143], v[129:132], v[172:175], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[144:147], v[168:171], v[136:143]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[136:143], v[148:151], v[180:183], v[136:143]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[152:155], v[176:179], v[136:143]
	s_cbranch_scc1 .LBB2_85
; %bb.84:                               ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[129:132], v202 offset:512
	ds_load_b128 v[152:155], v202 offset:2560
	ds_load_b128 v[156:159], v202 offset:4608
	ds_load_b128 v[160:163], v202 offset:6656
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[144:151], v[129:132], v[172:175], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[152:155], v[168:171], v[144:151]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[144:151], v[156:159], v[180:183], v[144:151]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[160:163], v[176:179], v[144:151]
	s_branch .LBB2_86
.LBB2_85:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v135, v128
	v_dual_mov_b32 v129, v128 :: v_dual_mov_b32 v130, v128
	v_dual_mov_b32 v131, v128 :: v_dual_mov_b32 v132, v128
	v_dual_mov_b32 v133, v128 :: v_dual_mov_b32 v134, v128
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v151, v135
	v_dual_mov_b32 v147, v131 :: v_dual_mov_b32 v146, v130
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v149, v133 :: v_dual_mov_b32 v148, v132
	v_dual_mov_b32 v150, v134 :: v_dual_mov_b32 v145, v129
	v_mov_b32_e32 v144, v128
.LBB2_86:                               ;   in Loop: Header=BB2_13 Depth=1
	s_or_b32 s4, s3, 32
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s13
	s_cselect_b32 s1, -1, 0
	s_cmp_gt_i32 s4, s13
	s_cbranch_scc1 .LBB2_88
; %bb.87:                               ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[129:132], v202 offset:1024
	ds_load_b128 v[160:163], v202 offset:3072
	ds_load_b128 v[164:167], v202 offset:5120
	ds_load_b128 v[227:230], v202 offset:7168
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[152:159], v[129:132], v[172:175], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[160:163], v[168:171], v[152:159]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[152:159], v[164:167], v[180:183], v[152:159]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[227:230], v[176:179], v[152:159]
	s_branch .LBB2_89
.LBB2_88:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v135, v128
	v_dual_mov_b32 v129, v128 :: v_dual_mov_b32 v130, v128
	v_dual_mov_b32 v131, v128 :: v_dual_mov_b32 v132, v128
	v_dual_mov_b32 v133, v128 :: v_dual_mov_b32 v134, v128
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v159, v135
	v_dual_mov_b32 v155, v131 :: v_dual_mov_b32 v154, v130
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v157, v133 :: v_dual_mov_b32 v156, v132
	v_dual_mov_b32 v158, v134 :: v_dual_mov_b32 v153, v129
	v_mov_b32_e32 v152, v128
.LBB2_89:                               ;   in Loop: Header=BB2_13 Depth=1
	s_or_b32 s4, s3, 48
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s13
	s_cselect_b32 s3, -1, 0
	s_cmp_gt_i32 s4, s13
	s_cbranch_scc1 .LBB2_91
; %bb.90:                               ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[129:132], v210
	ds_load_b128 v[227:230], v211
	ds_load_b128 v[231:234], v212
	ds_load_b128 v[235:238], v213
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[160:167], v[129:132], v[172:175], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[227:230], v[168:171], v[160:167]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[160:167], v[231:234], v[180:183], v[160:167]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[235:238], v[176:179], v[160:167]
	s_branch .LBB2_92
.LBB2_91:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v135, v128
	v_dual_mov_b32 v129, v128 :: v_dual_mov_b32 v130, v128
	v_dual_mov_b32 v131, v128 :: v_dual_mov_b32 v132, v128
	v_dual_mov_b32 v133, v128 :: v_dual_mov_b32 v134, v128
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v167, v135
	v_dual_mov_b32 v163, v131 :: v_dual_mov_b32 v162, v130
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v165, v133 :: v_dual_mov_b32 v164, v132
	v_dual_mov_b32 v166, v134 :: v_dual_mov_b32 v161, v129
	v_mov_b32_e32 v160, v128
.LBB2_92:                               ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_mov_b32_e32 v171, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s4, s19
	s_cbranch_execz .LBB2_94
; %bb.93:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[168:171], v[193:194], off offset:128
.LBB2_94:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_mov_b32_e32 v132, 0
	s_and_saveexec_b32 s4, s19
	s_cbranch_execz .LBB2_96
; %bb.95:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[129:132], v[193:194], off offset:160
.LBB2_96:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v179, 0
	v_mov_b32_e32 v178, 0
	s_and_saveexec_b32 s4, s19
	s_cbranch_execz .LBB2_98
; %bb.97:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[176:179], v[193:194], off offset:192
.LBB2_98:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, 0
	v_mov_b32_e32 v175, 0
	s_and_saveexec_b32 s4, s19
	s_cbranch_execz .LBB2_100
; %bb.99:                               ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[172:175], v[193:194], off offset:224
.LBB2_100:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	ds_load_b128 v[180:183], v202 offset:8192
	ds_load_b128 v[227:230], v202 offset:10240
	ds_load_b128 v[231:234], v202 offset:12288
	ds_load_b128 v[235:238], v202 offset:14336
	v_cndmask_b32_e64 v133, 0, 1, s0
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[136:143], v[180:183], v[168:171], v[136:143]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[227:230], v[129:132], v[136:143]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[136:143], v[231:234], v[176:179], v[136:143]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[235:238], v[172:175], v[136:143]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_134
; %bb.101:                              ;   in Loop: Header=BB2_13 Depth=1
	v_cndmask_b32_e64 v134, 0, 1, s1
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_135
.LBB2_102:                              ;   in Loop: Header=BB2_13 Depth=1
	v_cndmask_b32_e64 v135, 0, 1, s3
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_104
.LBB2_103:                              ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[180:183], v214
	ds_load_b128 v[227:230], v215
	ds_load_b128 v[231:234], v216
	ds_load_b128 v[235:238], v217
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[160:167], v[180:183], v[168:171], v[160:167]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[227:230], v[129:132], v[160:167]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[160:167], v[231:234], v[176:179], v[160:167]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[235:238], v[172:175], v[160:167]
.LBB2_104:                              ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_mov_b32_e32 v171, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_106
; %bb.105:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[168:171], v[193:194], off offset:256
.LBB2_106:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_mov_b32_e32 v132, 0
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_108
; %bb.107:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[129:132], v[193:194], off offset:288
.LBB2_108:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v179, 0
	v_mov_b32_e32 v178, 0
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_110
; %bb.109:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[176:179], v[193:194], off offset:320
.LBB2_110:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, 0
	v_mov_b32_e32 v175, 0
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_112
; %bb.111:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[172:175], v[193:194], off offset:352
.LBB2_112:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[180:183], v202 offset:16384
	ds_load_b128 v[227:230], v202 offset:18432
	ds_load_b128 v[231:234], v202 offset:20480
	ds_load_b128 v[235:238], v202 offset:22528
	v_cmp_ne_u32_e32 vcc_lo, 1, v133
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[136:143], v[180:183], v[168:171], v[136:143]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[227:230], v[129:132], v[136:143]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[136:143], v[231:234], v[176:179], v[136:143]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[235:238], v[172:175], v[136:143]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_136
; %bb.113:                              ;   in Loop: Header=BB2_13 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v134
	s_cbranch_vccz .LBB2_137
.LBB2_114:                              ;   in Loop: Header=BB2_13 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v135
	s_cbranch_vccnz .LBB2_116
.LBB2_115:                              ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[180:183], v218
	ds_load_b128 v[227:230], v219
	ds_load_b128 v[231:234], v220
	ds_load_b128 v[235:238], v221
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[160:167], v[180:183], v[168:171], v[160:167]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[227:230], v[129:132], v[160:167]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[160:167], v[231:234], v[176:179], v[160:167]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[235:238], v[172:175], v[160:167]
.LBB2_116:                              ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_mov_b32_e32 v171, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_118
; %bb.117:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[168:171], v[193:194], off offset:384
.LBB2_118:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_mov_b32_e32 v132, 0
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_120
; %bb.119:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[129:132], v[193:194], off offset:416
.LBB2_120:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v179, 0
	v_mov_b32_e32 v178, 0
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_122
; %bb.121:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[176:179], v[193:194], off offset:448
.LBB2_122:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, 0
	v_mov_b32_e32 v175, 0
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB2_124
; %bb.123:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[172:175], v[193:194], off offset:480
.LBB2_124:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[180:183], v202 offset:24576
	ds_load_b128 v[227:230], v202 offset:26624
	ds_load_b128 v[231:234], v202 offset:28672
	ds_load_b128 v[235:238], v202 offset:30720
	v_cmp_ne_u32_e32 vcc_lo, 1, v133
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[136:143], v[180:183], v[168:171], v[136:143]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[227:230], v[129:132], v[136:143]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[136:143], v[231:234], v[176:179], v[136:143]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[136:143], v[235:238], v[172:175], v[136:143]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_138
; %bb.125:                              ;   in Loop: Header=BB2_13 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v134
	s_cbranch_vccz .LBB2_139
.LBB2_126:                              ;   in Loop: Header=BB2_13 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v135
	s_cbranch_vccnz .LBB2_128
.LBB2_127:                              ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[180:183], v222
	ds_load_b128 v[227:230], v223
	ds_load_b128 v[231:234], v224
	ds_load_b128 v[235:238], v225
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[160:167], v[180:183], v[168:171], v[160:167]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[227:230], v[129:132], v[160:167]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[160:167], v[231:234], v[176:179], v[160:167]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[160:167], v[235:238], v[172:175], v[160:167]
.LBB2_128:                              ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v129, v202
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s25, 0
	s_mov_b32 s26, 0
	s_branch .LBB2_131
.LBB2_129:                              ;   in Loop: Header=BB2_131 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v131, s25, v226
	s_cmp_lt_u32 s26, 2
	v_cndmask_b32_e64 v168, v165, v157, s1
	s_cselect_b32 s10, -1, 0
	v_cndmask_b32_e64 v169, v164, v156, s1
	v_add_nc_u32_e32 v132, 2, v131
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s7, v131, v130
	v_cmp_lt_i32_e64 s8, v131, v130
	v_cndmask_b32_e64 v170, v163, v155, s1
	v_cndmask_b32_e64 v171, v161, v153, s1
	v_cmp_le_i32_e32 vcc_lo, v132, v130
	v_add_nc_u32_e32 v132, 3, v131
	v_cndmask_b32_e64 v172, v160, v152, s1
	v_cndmask_b32_e64 v173, v162, v154, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, v132, v130
	v_add_nc_u32_e32 v132, 4, v131
	v_cmp_le_i32_e64 s4, v132, v130
	v_add_nc_u32_e32 v132, 5, v131
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cmp_le_i32_e64 s5, v132, v130
	v_add_nc_u32_e32 v132, 6, v131
	v_add_nc_u32_e32 v131, 7, v131
	v_cmp_le_i32_e64 s6, v132, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_i32_e64 s9, v131, v130
	v_cndmask_b32_e64 v130, v151, v143, s0
	v_cndmask_b32_e64 v131, v167, v159, s1
	v_cndmask_b32_e64 v132, v166, v158, s1
	s_or_b32 s1, s27, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v130, v131, v130, s10
	v_cndmask_b32_e64 v131, v150, v142, s0
	v_mul_f32_e32 v130, s16, v130
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v131, v132, v131, s10
	v_cndmask_b32_e64 v132, v149, v141, s0
	v_mul_f32_e32 v131, s16, v131
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v132, v168, v132, s10
	v_cndmask_b32_e64 v168, v148, v140, s0
	v_mul_f32_e32 v132, s16, v132
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v168, v169, v168, s10
	v_cndmask_b32_e64 v169, v147, v139, s0
	v_mul_f32_e32 v168, s16, v168
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v169, v170, v169, s10
	v_cndmask_b32_e64 v170, v145, v137, s0
	v_mul_f32_e32 v169, s16, v169
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v170, v171, v170, s10
	v_cndmask_b32_e64 v171, v144, v136, s0
	v_mul_f32_e32 v170, s16, v170
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v171, v172, v171, s10
	v_cndmask_b32_e64 v172, v146, v138, s0
	s_or_b32 s0, s27, s7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s19, s0
	v_mul_f32_e32 v171, s16, v171
	v_cndmask_b32_e64 v172, v173, v172, s10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v171, 0xff800000, v171, s0
	s_and_b32 s0, s19, s1
	v_mul_f32_e32 v172, s16, v172
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v170, 0xff800000, v170, s0
	s_or_b32 s0, s27, vcc_lo
	s_or_b32 s1, s27, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s19, s0
	s_or_b32 s0, s27, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v172, 0xff800000, v172, vcc_lo
	s_and_b32 vcc_lo, s19, s1
	s_or_b32 s1, s27, s5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v169, 0xff800000, v169, vcc_lo
	s_and_b32 vcc_lo, s19, s0
	s_or_b32 s0, s27, s6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v168, 0xff800000, v168, vcc_lo
	s_and_b32 vcc_lo, s19, s1
	s_or_b32 s1, s27, s9
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v173, 0xff800000, v132, vcc_lo
	v_max3_num_f32 v132, v171, 0xff800000, v170
	s_and_b32 vcc_lo, s19, s0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v174, 0xff800000, v131, vcc_lo
	s_and_b32 vcc_lo, s19, s1
	v_max3_num_f32 v132, v132, v172, v169
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v175, 0xff800000, v130, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v130, v132, v168, v173
	v_max3_num_f32 v130, v130, v174, v175
	ds_bpermute_b32 v131, v201, v130
	s_wait_dscnt 0x0
	v_max3_num_f32 v130, v187, v130, v131
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v131, v171, v130
	v_dual_sub_f32 v169, v169, v130 :: v_dual_sub_f32 v132, v170, v130
	v_sub_f32_e32 v175, v175, v130
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v130
	v_mul_f32_e32 v131, 0x3fb8aa3b, v131
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v169, 0x3fb8aa3b, v169 :: v_dual_mul_f32 v132, 0x3fb8aa3b, v132
	v_dual_sub_f32 v171, v187, v130 :: v_dual_sub_f32 v170, v172, v130
	v_exp_f32_e32 v131, v131
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v169, v169
	v_exp_f32_e32 v132, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v171, 0x3fb8aa3b, v171
	v_sub_f32_e32 v173, v173, v130
	v_exp_f32_e32 v171, v171
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v131, v131, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v177, v169, 0, vcc_lo
	v_cndmask_b32_e64 v132, v132, 0, vcc_lo
	v_mul_f32_e32 v170, 0x3fb8aa3b, v170
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e64 v169.l, v131
	v_add_f32_e32 v131, v131, v132
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v170, v170
	v_sub_f32_e32 v172, v168, v130
	v_cvt_f16_f32_e64 v169.h, v132
	v_cndmask_b32_e64 v168, v170, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v131, v168, v131 :: v_dual_sub_f32 v174, v174, v130
	v_dual_add_f32 v131, v177, v131 :: v_dual_mul_f32 v170, 0x3fb8aa3b, v172
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v172, 0x3fb8aa3b, v173 :: v_dual_mul_f32 v173, 0x3fb8aa3b, v174
	v_mul_f32_e32 v174, 0x3fb8aa3b, v175
	v_exp_f32_e32 v175, v170
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v173, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v174, v174
	v_cvt_f16_f32_e64 v170.l, v168
	v_cvt_f16_f32_e64 v170.h, v177
	v_cndmask_b32_e64 v178, v175, 0, vcc_lo
	v_cndmask_b32_e64 v179, v172, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e64 v180, v173, 0, vcc_lo
	v_cndmask_b32_e64 v181, v174, 0, vcc_lo
	ds_load_b128 v[173:176], v129 offset:32768
	v_add_f32_e32 v131, v178, v131
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v187
	v_cvt_f16_f32_e64 v172.l, v180
	v_cvt_f16_f32_e64 v172.h, v181
	;;#ASMSTART
	;;#ASMEND
	v_add_f32_e32 v131, v179, v131
	v_mov_b32_e32 v187, v130
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v131, v180, v131
	v_add_f32_e32 v131, v181, v131
	ds_bpermute_b32 v132, v201, v131
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v182, 0, v171, vcc_lo
	v_cvt_f16_f32_e64 v171.l, v178
	v_cvt_f16_f32_e64 v171.h, v179
	s_wait_dscnt 0x0
	v_add_f32_e32 v131, v131, v132
	v_dual_mul_f32 v127, v127, v182 :: v_dual_mul_f32 v126, v126, v182
	v_mul_f32_e32 v123, v123, v182
	v_dual_mul_f32 v125, v125, v182 :: v_dual_mul_f32 v124, v124, v182
	v_dual_mul_f32 v121, v121, v182 :: v_dual_mul_f32 v122, v122, v182
	v_dual_mul_f32 v119, v119, v182 :: v_dual_mul_f32 v120, v120, v182
	v_dual_mul_f32 v117, v117, v182 :: v_dual_mul_f32 v118, v118, v182
	v_dual_mul_f32 v115, v115, v182 :: v_dual_mul_f32 v116, v116, v182
	v_mul_f32_e32 v113, v113, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[120:127], v[173:176], v[169:172], v[120:127]
	ds_load_b128 v[173:176], v129 offset:33280
	v_dual_mul_f32 v114, v114, v182 :: v_dual_mul_f32 v111, v111, v182
	v_dual_mul_f32 v112, v112, v182 :: v_dual_mul_f32 v109, v109, v182
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v110, v110, v182 :: v_dual_mul_f32 v107, v107, v182
	v_dual_mul_f32 v108, v108, v182 :: v_dual_mul_f32 v105, v105, v182
	v_dual_mul_f32 v106, v106, v182 :: v_dual_mul_f32 v103, v103, v182
	v_dual_mul_f32 v104, v104, v182 :: v_dual_mul_f32 v101, v101, v182
	v_dual_mul_f32 v102, v102, v182 :: v_dual_mul_f32 v99, v99, v182
	v_dual_mul_f32 v100, v100, v182 :: v_dual_mul_f32 v97, v97, v182
	v_dual_mul_f32 v98, v98, v182 :: v_dual_mul_f32 v95, v95, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[112:119], v[173:176], v[169:172], v[112:119]
	ds_load_b128 v[173:176], v129 offset:33792
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v96, v96, v182 :: v_dual_mul_f32 v93, v93, v182
	v_dual_mul_f32 v94, v94, v182 :: v_dual_mul_f32 v91, v91, v182
	v_dual_mul_f32 v92, v92, v182 :: v_dual_mul_f32 v89, v89, v182
	v_dual_mul_f32 v90, v90, v182 :: v_dual_mul_f32 v87, v87, v182
	v_dual_mul_f32 v88, v88, v182 :: v_dual_mul_f32 v85, v85, v182
	v_dual_mul_f32 v86, v86, v182 :: v_dual_mul_f32 v83, v83, v182
	v_dual_mul_f32 v84, v84, v182 :: v_dual_mul_f32 v81, v81, v182
	v_dual_mul_f32 v82, v82, v182 :: v_dual_mul_f32 v79, v79, v182
	v_dual_mul_f32 v80, v80, v182 :: v_dual_mul_f32 v77, v77, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[104:111], v[173:176], v[169:172], v[104:111]
	ds_load_b128 v[173:176], v129 offset:34304
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v78, v78, v182 :: v_dual_mul_f32 v75, v75, v182
	v_dual_mul_f32 v76, v76, v182 :: v_dual_mul_f32 v73, v73, v182
	v_dual_mul_f32 v74, v74, v182 :: v_dual_mul_f32 v71, v71, v182
	v_dual_mul_f32 v72, v72, v182 :: v_dual_mul_f32 v69, v69, v182
	v_dual_mul_f32 v70, v70, v182 :: v_dual_mul_f32 v67, v67, v182
	v_dual_mul_f32 v68, v68, v182 :: v_dual_mul_f32 v65, v65, v182
	v_dual_mul_f32 v66, v66, v182 :: v_dual_mul_f32 v63, v63, v182
	v_dual_mul_f32 v64, v64, v182 :: v_dual_mul_f32 v61, v61, v182
	v_dual_mul_f32 v62, v62, v182 :: v_dual_mul_f32 v59, v59, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[96:103], v[173:176], v[169:172], v[96:103]
	ds_load_b128 v[173:176], v129 offset:34816
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v60, v60, v182 :: v_dual_mul_f32 v57, v57, v182
	v_dual_mul_f32 v58, v58, v182 :: v_dual_mul_f32 v55, v55, v182
	v_dual_mul_f32 v56, v56, v182 :: v_dual_mul_f32 v53, v53, v182
	v_dual_mul_f32 v54, v54, v182 :: v_dual_mul_f32 v51, v51, v182
	v_dual_mul_f32 v52, v52, v182 :: v_dual_mul_f32 v49, v49, v182
	v_dual_mul_f32 v50, v50, v182 :: v_dual_mul_f32 v47, v47, v182
	v_dual_mul_f32 v48, v48, v182 :: v_dual_mul_f32 v45, v45, v182
	v_dual_mul_f32 v46, v46, v182 :: v_dual_mul_f32 v43, v43, v182
	v_dual_mul_f32 v44, v44, v182 :: v_dual_mul_f32 v41, v41, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[88:95], v[173:176], v[169:172], v[88:95]
	ds_load_b128 v[173:176], v129 offset:35328
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v42, v42, v182 :: v_dual_mul_f32 v39, v39, v182
	v_dual_mul_f32 v40, v40, v182 :: v_dual_mul_f32 v37, v37, v182
	v_dual_mul_f32 v38, v38, v182 :: v_dual_mul_f32 v35, v35, v182
	v_dual_mul_f32 v36, v36, v182 :: v_dual_mul_f32 v33, v33, v182
	v_dual_mul_f32 v34, v34, v182 :: v_dual_mul_f32 v31, v31, v182
	v_dual_mul_f32 v32, v32, v182 :: v_dual_mul_f32 v29, v29, v182
	v_dual_mul_f32 v30, v30, v182 :: v_dual_mul_f32 v27, v27, v182
	v_dual_mul_f32 v28, v28, v182 :: v_dual_mul_f32 v25, v25, v182
	v_dual_mul_f32 v26, v26, v182 :: v_dual_mul_f32 v23, v23, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[80:87], v[173:176], v[169:172], v[80:87]
	ds_load_b128 v[173:176], v129 offset:35840
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v24, v24, v182 :: v_dual_mul_f32 v21, v21, v182
	v_dual_mul_f32 v22, v22, v182 :: v_dual_mul_f32 v19, v19, v182
	v_dual_mul_f32 v20, v20, v182 :: v_dual_mul_f32 v17, v17, v182
	v_dual_mul_f32 v18, v18, v182 :: v_dual_mul_f32 v15, v15, v182
	v_dual_mul_f32 v16, v16, v182 :: v_dual_mul_f32 v13, v13, v182
	v_dual_mul_f32 v14, v14, v182 :: v_dual_mul_f32 v11, v11, v182
	v_dual_mul_f32 v12, v12, v182 :: v_dual_mul_f32 v9, v9, v182
	v_dual_mul_f32 v10, v10, v182 :: v_dual_mul_f32 v7, v7, v182
	v_dual_mul_f32 v8, v8, v182 :: v_dual_mul_f32 v5, v5, v182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[72:79], v[173:176], v[169:172], v[72:79]
	ds_load_b128 v[173:176], v129 offset:36352
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v6, v6, v182 :: v_dual_mul_f32 v3, v3, v182
	v_dual_mul_f32 v4, v4, v182 :: v_dual_mul_f32 v1, v1, v182
	v_dual_mul_f32 v2, v2, v182 :: v_dual_fmac_f32 v131, v188, v182
	v_mul_f32_e32 v0, v0, v182
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v188, v131
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[64:71], v[173:176], v[169:172], v[64:71]
	ds_load_b128 v[173:176], v129 offset:36864
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[56:63], v[173:176], v[169:172], v[56:63]
	ds_load_b128 v[173:176], v129 offset:37376
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[48:55], v[173:176], v[169:172], v[48:55]
	ds_load_b128 v[173:176], v129 offset:37888
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[40:47], v[173:176], v[169:172], v[40:47]
	ds_load_b128 v[173:176], v129 offset:38400
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[32:39], v[173:176], v[169:172], v[32:39]
	ds_load_b128 v[173:176], v129 offset:38912
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[24:31], v[173:176], v[169:172], v[24:31]
	ds_load_b128 v[173:176], v129 offset:39424
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[16:23], v[173:176], v[169:172], v[16:23]
	ds_load_b128 v[173:176], v129 offset:39936
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[8:15], v[173:176], v[169:172], v[8:15]
	ds_load_b128 v[173:176], v129 offset:40448
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[173:176], v[169:172], v[0:7]
.LBB2_130:                              ;   in Loop: Header=BB2_131 Depth=2
	v_add_nc_u32_e32 v129, 0x2000, v129
	s_add_co_i32 s25, s25, 16
	s_add_co_i32 s26, s26, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s25, 64
	s_cbranch_scc0 .LBB2_11
.LBB2_131:                              ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s25, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s25, 32
	s_cselect_b32 s1, -1, 0
	s_cmp_eq_u32 s25, 16
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v130, v135, v134, s1
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v130, v130, v133, vcc_lo
	v_and_b32_e32 v130, 1, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 1, v130
	s_or_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_130
; %bb.132:                              ;   in Loop: Header=BB2_131 Depth=2
	s_add_co_i32 s3, s12, s25
	v_mov_b32_e32 v130, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s3, 15
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s20
	s_cselect_b32 s27, -1, 0
	s_cmp_gt_i32 s3, s20
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s3, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB2_129
; %bb.133:                              ;   in Loop: Header=BB2_131 Depth=2
	global_load_b32 v130, v[195:196], off
	s_branch .LBB2_129
.LBB2_134:                              ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[180:183], v202 offset:8704
	ds_load_b128 v[227:230], v202 offset:10752
	ds_load_b128 v[231:234], v202 offset:12800
	ds_load_b128 v[235:238], v202 offset:14848
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[144:151], v[180:183], v[168:171], v[144:151]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[227:230], v[129:132], v[144:151]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[144:151], v[231:234], v[176:179], v[144:151]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[235:238], v[172:175], v[144:151]
	v_cndmask_b32_e64 v134, 0, 1, s1
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_102
.LBB2_135:                              ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[180:183], v202 offset:9216
	ds_load_b128 v[227:230], v202 offset:11264
	ds_load_b128 v[231:234], v202 offset:13312
	ds_load_b128 v[235:238], v202 offset:15360
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[152:159], v[180:183], v[168:171], v[152:159]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[227:230], v[129:132], v[152:159]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[152:159], v[231:234], v[176:179], v[152:159]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[235:238], v[172:175], v[152:159]
	v_cndmask_b32_e64 v135, 0, 1, s3
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_103
	s_branch .LBB2_104
.LBB2_136:                              ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[180:183], v202 offset:16896
	ds_load_b128 v[227:230], v202 offset:18944
	ds_load_b128 v[231:234], v202 offset:20992
	ds_load_b128 v[235:238], v202 offset:23040
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[144:151], v[180:183], v[168:171], v[144:151]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[227:230], v[129:132], v[144:151]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[144:151], v[231:234], v[176:179], v[144:151]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[235:238], v[172:175], v[144:151]
	v_cmp_ne_u32_e32 vcc_lo, 1, v134
	s_cbranch_vccnz .LBB2_114
.LBB2_137:                              ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[180:183], v202 offset:17408
	ds_load_b128 v[227:230], v202 offset:19456
	ds_load_b128 v[231:234], v202 offset:21504
	ds_load_b128 v[235:238], v202 offset:23552
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[152:159], v[180:183], v[168:171], v[152:159]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[227:230], v[129:132], v[152:159]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[152:159], v[231:234], v[176:179], v[152:159]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[235:238], v[172:175], v[152:159]
	v_cmp_ne_u32_e32 vcc_lo, 1, v135
	s_cbranch_vccz .LBB2_115
	s_branch .LBB2_116
.LBB2_138:                              ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[180:183], v202 offset:25088
	ds_load_b128 v[227:230], v202 offset:27136
	ds_load_b128 v[231:234], v202 offset:29184
	ds_load_b128 v[235:238], v202 offset:31232
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[144:151], v[180:183], v[168:171], v[144:151]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[227:230], v[129:132], v[144:151]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[144:151], v[231:234], v[176:179], v[144:151]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[144:151], v[235:238], v[172:175], v[144:151]
	v_cmp_ne_u32_e32 vcc_lo, 1, v134
	s_cbranch_vccnz .LBB2_126
.LBB2_139:                              ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[180:183], v202 offset:25600
	ds_load_b128 v[227:230], v202 offset:27648
	ds_load_b128 v[231:234], v202 offset:29696
	ds_load_b128 v[235:238], v202 offset:31744
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[152:159], v[180:183], v[168:171], v[152:159]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[227:230], v[129:132], v[152:159]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[152:159], v[231:234], v[176:179], v[152:159]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[152:159], v[235:238], v[172:175], v[152:159]
	v_cmp_ne_u32_e32 vcc_lo, 1, v135
	s_cbranch_vccz .LBB2_127
	s_branch .LBB2_128
.LBB2_140:
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB2_145
; %bb.141:
	v_cmp_eq_u32_e32 vcc_lo, 0, v198
	s_and_b32 s1, vcc_lo, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_143
; %bb.142:
	v_mad_co_u64_u32 v[128:129], null, v184, 24, v[186:187]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[128:129], null, v128, s17, s[18:19]
	v_mov_b32_e32 v129, 0
	v_mul_lo_u32 v128, 0x102, v128
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[128:129], 2, v[128:129]
	v_add_co_u32 v128, vcc_lo, s14, v128
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v129, null, s15, v129, vcc_lo
	global_store_b64 v[128:129], v[187:188], off
.LBB2_143:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s19
	s_cbranch_execz .LBB2_145
; %bb.144:
	v_mad_co_u64_u32 v[128:129], null, v184, 24, v[186:187]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[128:129], null, v128, s17, s[18:19]
	v_mul_lo_u32 v128, 0x102, v128
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v129, 0 :: v_dual_add_nc_u32 v130, v128, v185
	v_lshlrev_b64_e32 v[131:132], 2, v[128:129]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v128, 2, v130
	v_add_co_u32 v131, vcc_lo, s14, v131
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v132, null, s15, v132, vcc_lo
	v_lshlrev_b64_e32 v[133:134], 2, v[128:129]
	v_add_nc_u32_e32 v128, 18, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[135:136], 2, v[128:129]
	v_add_nc_u32_e32 v128, 34, v130
	v_add_co_u32 v133, vcc_lo, v131, v133
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v134, null, v132, v134, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[137:138], 2, v[128:129]
	v_add_nc_u32_e32 v128, 50, v130
	v_add_co_u32 v135, vcc_lo, v131, v135
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v136, null, v132, v136, vcc_lo
	v_lshlrev_b64_e32 v[139:140], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x42, v130
	v_add_co_u32 v137, vcc_lo, v131, v137
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, v132, v138, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[141:142], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x52, v130
	v_add_co_u32 v139, vcc_lo, v131, v139
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v140, null, v132, v140, vcc_lo
	v_lshlrev_b64_e32 v[143:144], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x62, v130
	s_clause 0x3
	global_store_b64 v[133:134], v[120:121], off
	global_store_b64 v[135:136], v[112:113], off
	global_store_b64 v[137:138], v[104:105], off
	global_store_b64 v[139:140], v[96:97], off
	v_add_co_u32 v141, vcc_lo, v131, v141
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v142, null, v132, v142, vcc_lo
	v_lshlrev_b64_e32 v[96:97], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x72, v130
	v_add_co_u32 v104, vcc_lo, v131, v143
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v105, null, v132, v144, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[112:113], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x82, v130
	v_add_co_u32 v96, vcc_lo, v131, v96
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v97, null, v132, v97, vcc_lo
	v_lshlrev_b64_e32 v[120:121], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x92, v130
	v_add_co_u32 v112, vcc_lo, v131, v112
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v113, null, v132, v113, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[133:134], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xa2, v130
	s_clause 0x3
	global_store_b64 v[141:142], v[88:89], off
	global_store_b64 v[104:105], v[80:81], off
	global_store_b64 v[96:97], v[72:73], off
	global_store_b64 v[112:113], v[64:65], off
	v_add_co_u32 v120, vcc_lo, v131, v120
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v121, null, v132, v121, vcc_lo
	v_lshlrev_b64_e32 v[64:65], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xb2, v130
	v_add_co_u32 v72, vcc_lo, v131, v133
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, v132, v134, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[80:81], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xc2, v130
	v_add_co_u32 v64, vcc_lo, v131, v64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, v132, v65, vcc_lo
	v_lshlrev_b64_e32 v[88:89], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xd2, v130
	v_add_co_u32 v80, vcc_lo, v131, v80
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, v132, v81, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[96:97], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xe2, v130
	v_add_co_u32 v88, vcc_lo, v131, v88
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v89, null, v132, v89, vcc_lo
	v_lshlrev_b64_e32 v[104:105], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xf2, v130
	v_add_co_u32 v96, vcc_lo, v131, v96
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v97, null, v132, v97, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[112:113], 2, v[128:129]
	v_add_nc_u32_e32 v128, 4, v130
	v_add_co_u32 v104, vcc_lo, v131, v104
	s_clause 0x3
	global_store_b64 v[120:121], v[56:57], off
	global_store_b64 v[72:73], v[48:49], off
	global_store_b64 v[64:65], v[40:41], off
	global_store_b64 v[80:81], v[32:33], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v105, null, v132, v105, vcc_lo
	v_lshlrev_b64_e32 v[32:33], 2, v[128:129]
	v_add_nc_u32_e32 v128, 20, v130
	v_add_co_u32 v112, vcc_lo, v131, v112
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v113, null, v132, v113, vcc_lo
	s_clause 0x3
	global_store_b64 v[88:89], v[24:25], off
	global_store_b64 v[96:97], v[16:17], off
	global_store_b64 v[104:105], v[8:9], off
	global_store_b64 v[112:113], v[0:1], off
	v_lshlrev_b64_e32 v[0:1], 2, v[128:129]
	v_add_nc_u32_e32 v128, 36, v130
	v_add_co_u32 v8, vcc_lo, v131, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v132, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[128:129]
	v_add_nc_u32_e32 v128, 52, v130
	v_add_co_u32 v0, vcc_lo, v131, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v132, v1, vcc_lo
	v_lshlrev_b64_e32 v[24:25], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x44, v130
	v_add_co_u32 v16, vcc_lo, v131, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v132, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[32:33], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x54, v130
	v_add_co_u32 v24, vcc_lo, v131, v24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, v132, v25, vcc_lo
	v_lshlrev_b64_e32 v[40:41], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x64, v130
	v_add_co_u32 v32, vcc_lo, v131, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v132, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[48:49], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x74, v130
	v_add_co_u32 v40, vcc_lo, v131, v40
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v41, null, v132, v41, vcc_lo
	v_lshlrev_b64_e32 v[56:57], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x84, v130
	v_add_co_u32 v48, vcc_lo, v131, v48
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v49, null, v132, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[64:65], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x94, v130
	v_add_co_u32 v56, vcc_lo, v131, v56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v57, null, v132, v57, vcc_lo
	v_lshlrev_b64_e32 v[72:73], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xa4, v130
	v_add_co_u32 v64, vcc_lo, v131, v64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, v132, v65, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[80:81], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xb4, v130
	v_add_co_u32 v72, vcc_lo, v131, v72
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, v132, v73, vcc_lo
	v_lshlrev_b64_e32 v[88:89], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xc4, v130
	v_add_co_u32 v80, vcc_lo, v131, v80
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, v132, v81, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[96:97], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xd4, v130
	v_add_co_u32 v88, vcc_lo, v131, v88
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v89, null, v132, v89, vcc_lo
	v_lshlrev_b64_e32 v[104:105], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xe4, v130
	v_add_co_u32 v96, vcc_lo, v131, v96
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v97, null, v132, v97, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[112:113], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xf4, v130
	v_add_co_u32 v104, vcc_lo, v131, v104
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v105, null, v132, v105, vcc_lo
	v_lshlrev_b64_e32 v[120:121], 2, v[128:129]
	v_add_nc_u32_e32 v128, 6, v130
	v_add_co_u32 v112, vcc_lo, v131, v112
	s_clause 0x7
	global_store_b64 v[8:9], v[122:123], off
	global_store_b64 v[0:1], v[114:115], off
	global_store_b64 v[16:17], v[106:107], off
	global_store_b64 v[24:25], v[98:99], off
	global_store_b64 v[32:33], v[90:91], off
	global_store_b64 v[40:41], v[82:83], off
	global_store_b64 v[48:49], v[74:75], off
	global_store_b64 v[56:57], v[66:67], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v113, null, v132, v113, vcc_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[128:129]
	v_add_nc_u32_e32 v128, 22, v130
	v_add_co_u32 v120, vcc_lo, v131, v120
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v121, null, v132, v121, vcc_lo
	s_clause 0x3
	global_store_b64 v[96:97], v[26:27], off
	global_store_b64 v[104:105], v[18:19], off
	global_store_b64 v[112:113], v[10:11], off
	global_store_b64 v[120:121], v[2:3], off
	v_lshlrev_b64_e32 v[2:3], 2, v[128:129]
	v_add_nc_u32_e32 v128, 38, v130
	s_clause 0x3
	global_store_b64 v[64:65], v[58:59], off
	global_store_b64 v[72:73], v[50:51], off
	global_store_b64 v[80:81], v[42:43], off
	global_store_b64 v[88:89], v[34:35], off
	v_add_co_u32 v0, vcc_lo, v131, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v132, v1, vcc_lo
	v_lshlrev_b64_e32 v[8:9], 2, v[128:129]
	v_add_nc_u32_e32 v128, 54, v130
	v_add_co_u32 v2, vcc_lo, v131, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, v132, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x46, v130
	v_add_co_u32 v8, vcc_lo, v131, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v132, v9, vcc_lo
	v_lshlrev_b64_e32 v[16:17], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x56, v130
	v_add_co_u32 v10, vcc_lo, v131, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v132, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x66, v130
	v_add_co_u32 v16, vcc_lo, v131, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v132, v17, vcc_lo
	v_lshlrev_b64_e32 v[24:25], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x76, v130
	v_add_co_u32 v18, vcc_lo, v131, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v132, v19, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[26:27], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x86, v130
	v_add_co_u32 v24, vcc_lo, v131, v24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, v132, v25, vcc_lo
	v_lshlrev_b64_e32 v[32:33], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x96, v130
	v_add_co_u32 v26, vcc_lo, v131, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v132, v27, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[34:35], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xa6, v130
	v_add_co_u32 v32, vcc_lo, v131, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v132, v33, vcc_lo
	v_lshlrev_b64_e32 v[40:41], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xb6, v130
	v_add_co_u32 v34, vcc_lo, v131, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, v132, v35, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[42:43], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xc6, v130
	v_add_co_u32 v40, vcc_lo, v131, v40
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v41, null, v132, v41, vcc_lo
	v_lshlrev_b64_e32 v[48:49], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xd6, v130
	v_add_co_u32 v42, vcc_lo, v131, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, v132, v43, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[50:51], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xe6, v130
	v_add_co_u32 v48, vcc_lo, v131, v48
	s_clause 0x7
	global_store_b64 v[0:1], v[124:125], off
	global_store_b64 v[2:3], v[116:117], off
	global_store_b64 v[8:9], v[108:109], off
	global_store_b64 v[10:11], v[100:101], off
	global_store_b64 v[16:17], v[92:93], off
	global_store_b64 v[18:19], v[84:85], off
	global_store_b64 v[24:25], v[76:77], off
	global_store_b64 v[26:27], v[68:69], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v49, null, v132, v49, vcc_lo
	v_lshlrev_b64_e32 v[56:57], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xf6, v130
	v_add_co_u32 v50, vcc_lo, v131, v50
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, v132, v51, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[58:59], 2, v[128:129]
	v_add_nc_u32_e32 v128, 8, v130
	v_add_co_u32 v56, vcc_lo, v131, v56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v57, null, v132, v57, vcc_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[128:129]
	v_add_nc_u32_e32 v128, 24, v130
	v_add_co_u32 v58, vcc_lo, v131, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, v132, v59, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 2, v[128:129]
	v_add_nc_u32_e32 v128, 40, v130
	s_clause 0x3
	global_store_b64 v[48:49], v[28:29], off
	global_store_b64 v[50:51], v[20:21], off
	global_store_b64 v[56:57], v[12:13], off
	global_store_b64 v[58:59], v[4:5], off
	v_add_co_u32 v0, vcc_lo, v131, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v132, v1, vcc_lo
	v_lshlrev_b64_e32 v[4:5], 2, v[128:129]
	v_add_nc_u32_e32 v128, 56, v130
	v_add_co_u32 v2, vcc_lo, v131, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, v132, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x48, v130
	v_add_co_u32 v4, vcc_lo, v131, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v132, v5, vcc_lo
	v_lshlrev_b64_e32 v[10:11], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x58, v130
	v_add_co_u32 v8, vcc_lo, v131, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v132, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x68, v130
	v_add_co_u32 v10, vcc_lo, v131, v10
	s_clause 0x3
	global_store_b64 v[32:33], v[60:61], off
	global_store_b64 v[34:35], v[52:53], off
	global_store_b64 v[40:41], v[44:45], off
	global_store_b64 v[42:43], v[36:37], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v132, v11, vcc_lo
	v_lshlrev_b64_e32 v[16:17], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x78, v130
	v_add_co_u32 v12, vcc_lo, v131, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v132, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x88, v130
	v_add_co_u32 v16, vcc_lo, v131, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v132, v17, vcc_lo
	v_lshlrev_b64_e32 v[20:21], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0x98, v130
	v_add_co_u32 v18, vcc_lo, v131, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v132, v19, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[24:25], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xa8, v130
	v_add_co_u32 v20, vcc_lo, v131, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, v132, v21, vcc_lo
	v_lshlrev_b64_e32 v[26:27], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xb8, v130
	v_add_co_u32 v24, vcc_lo, v131, v24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, v132, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[28:29], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xc8, v130
	v_add_co_u32 v26, vcc_lo, v131, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v132, v27, vcc_lo
	v_lshlrev_b64_e32 v[32:33], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xd8, v130
	v_add_co_u32 v28, vcc_lo, v131, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v132, v29, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[34:35], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xe8, v130
	v_add_co_u32 v32, vcc_lo, v131, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v132, v33, vcc_lo
	v_lshlrev_b64_e32 v[36:37], 2, v[128:129]
	v_add_nc_u32_e32 v128, 0xf8, v130
	v_add_co_u32 v34, vcc_lo, v131, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, v132, v35, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[40:41], 2, v[128:129]
	v_add_co_u32 v36, vcc_lo, v131, v36
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v37, null, v132, v37, vcc_lo
	v_add_co_u32 v40, vcc_lo, v131, v40
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v41, null, v132, v41, vcc_lo
	s_clause 0xf
	global_store_b64 v[0:1], v[126:127], off
	global_store_b64 v[2:3], v[118:119], off
	global_store_b64 v[4:5], v[110:111], off
	global_store_b64 v[8:9], v[102:103], off
	global_store_b64 v[10:11], v[94:95], off
	global_store_b64 v[12:13], v[86:87], off
	global_store_b64 v[16:17], v[78:79], off
	global_store_b64 v[18:19], v[70:71], off
	global_store_b64 v[20:21], v[62:63], off
	global_store_b64 v[24:25], v[54:55], off
	global_store_b64 v[26:27], v[46:47], off
	global_store_b64 v[28:29], v[38:39], off
	global_store_b64 v[32:33], v[30:31], off
	global_store_b64 v[34:35], v[22:23], off
	global_store_b64 v[36:37], v[14:15], off
	global_store_b64 v[40:41], v[6:7], off
.LBB2_145:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end2:
	.size	attention_q8_0_fa2_gqa_partial_gfx1201, .Lfunc_end2-attention_q8_0_fa2_gqa_partial_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_q8_0_fa2_gqa_partial_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
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
		.amdhsa_next_free_vgpr 239
		.amdhsa_next_free_sgpr 28
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-attention_q8_0_fa2_gqa_partial_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_q8_0_fa2_gqa_partial_gfx1201.num_vgpr, 239
	.set .Lattention_q8_0_fa2_gqa_partial_gfx1201.num_agpr, 0
	.set .Lattention_q8_0_fa2_gqa_partial_gfx1201.numbered_sgpr, 28
	.set .Lattention_q8_0_fa2_gqa_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_q8_0_fa2_gqa_partial_gfx1201.private_seg_size, 0
	.set .Lattention_q8_0_fa2_gqa_partial_gfx1201.uses_vcc, 1
	.set .Lattention_q8_0_fa2_gqa_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_q8_0_fa2_gqa_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_q8_0_fa2_gqa_partial_gfx1201.has_recursion, 0
	.set .Lattention_q8_0_fa2_gqa_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 12408
; TotalNumSgprs: 30
; NumVgprs: 239
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 239
; Occupancy: 6
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_q8_0_fa2_gqa_merge_gfx1201 ; -- Begin function attention_q8_0_fa2_gqa_merge_gfx1201
	.globl	attention_q8_0_fa2_gqa_merge_gfx1201
	.p2align	8
	.type	attention_q8_0_fa2_gqa_merge_gfx1201,@function
attention_q8_0_fa2_gqa_merge_gfx1201:   ; @attention_q8_0_fa2_gqa_merge_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s5, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s5, s7, -9
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s5, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB3_10
; %bb.1:
	v_lshrrev_b32_e32 v1, 5, v0
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v4, ttmp9, 3, v1
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB3_10
; %bb.2:
	s_load_b128 s[0:3], s[0:1], 0x0
	v_mad_co_u64_u32 v[1:2], null, v4, s7, 0
	v_ashrrev_i32_e32 v5, 31, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[6:7], null, v5, s7, v[2:3]
	s_wait_kmcnt 0x0
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, s[0:1]
	s_mov_b32 s0, s7
	v_mad_co_u64_u32 v[2:3], null, 0x408, v6, v[2:3]
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v8, 0xff800000 :: v_dual_mov_b32 v7, v2
	v_mov_b32_e32 v6, v1
.LBB3_3:                                ; =>This Inner Loop Header: Depth=1
	global_load_b32 v3, v[6:7], off
	v_max_num_f32_e32 v8, v8, v8
	v_add_co_u32 v6, vcc_lo, 0x408, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	s_wait_loadcnt 0x0
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v8, v8, v3
	s_cbranch_scc0 .LBB3_3
; %bb.4:
	v_and_b32_e32 v3, 31, v0
	v_lshlrev_b64_e32 v[5:6], 10, v[4:5]
	v_mov_b32_e32 v4, 0
	s_mov_b32 s1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v9, 2, v3
	v_add_co_u32 v0, vcc_lo, s2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s3, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v9, s0, v9, 8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, 0, s0
	s_branch .LBB3_6
.LBB3_5:                                ;   in Loop: Header=BB3_6 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_scale_f32 v5, null, v11, v11, v12
	v_div_scale_f32 v14, vcc_lo, v12, v11, v12
	v_rcp_f32_e32 v6, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v13, -v5, v6, 1.0
	v_fmac_f32_e32 v6, v13, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v13, v14, v6
	v_fma_f32 v15, -v5, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v13, v15, v6
	v_fma_f32 v5, -v5, v13, v14
	v_add_nc_u32_e32 v14, 32, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v13, v5, v6, v13
	v_lshlrev_b64_e32 v[5:6], 2, v[3:4]
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v3
	v_div_fixup_f32 v3, v13, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v5, s0, v0, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v7, v6, s0
	v_cmp_lt_f32_e64 s0, 0, v11
	s_or_b32 s1, vcc_lo, s1
	v_cndmask_b32_e64 v11, 0, v3, s0
	v_add_co_u32 v9, s0, 0x80, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0
	v_mov_b32_e32 v3, v14
	global_store_b32 v[5:6], v11, off
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB3_10
.LBB3_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_8 Depth 2
	v_dual_mov_b32 v6, v2 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	s_mov_b32 s0, s7
	s_branch .LBB3_8
.LBB3_7:                                ;   in Loop: Header=BB3_8 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_co_u32 v15, vcc_lo, v5, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v6, v10, vcc_lo
	v_add_co_u32 v5, vcc_lo, 0x408, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	global_load_b32 v15, v[15:16], off
	v_fmac_f32_e32 v11, v13, v14
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v12, v14, v15
	s_cbranch_scc1 .LBB3_5
.LBB3_8:                                ;   Parent Loop BB3_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	s_mov_b32 s2, exec_lo
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	s_cbranch_execz .LBB3_7
; %bb.9:                                ;   in Loop: Header=BB3_8 Depth=2
	global_load_b32 v14, v[5:6], off
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	v_exp_f32_e32 v14, v14
	s_branch .LBB3_7
.LBB3_10:
	s_endpgm
.Lfunc_end3:
	.size	attention_q8_0_fa2_gqa_merge_gfx1201, .Lfunc_end3-attention_q8_0_fa2_gqa_merge_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_q8_0_fa2_gqa_merge_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 17
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-attention_q8_0_fa2_gqa_merge_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_q8_0_fa2_gqa_merge_gfx1201.num_vgpr, 17
	.set .Lattention_q8_0_fa2_gqa_merge_gfx1201.num_agpr, 0
	.set .Lattention_q8_0_fa2_gqa_merge_gfx1201.numbered_sgpr, 8
	.set .Lattention_q8_0_fa2_gqa_merge_gfx1201.num_named_barrier, 0
	.set .Lattention_q8_0_fa2_gqa_merge_gfx1201.private_seg_size, 0
	.set .Lattention_q8_0_fa2_gqa_merge_gfx1201.uses_vcc, 1
	.set .Lattention_q8_0_fa2_gqa_merge_gfx1201.uses_flat_scratch, 0
	.set .Lattention_q8_0_fa2_gqa_merge_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_q8_0_fa2_gqa_merge_gfx1201.has_recursion, 0
	.set .Lattention_q8_0_fa2_gqa_merge_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 728
; TotalNumSgprs: 10
; NumVgprs: 17
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 17
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
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
	.type	__hip_cuid_9f03258719b67a61,@object ; @__hip_cuid_9f03258719b67a61
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9f03258719b67a61
__hip_cuid_9f03258719b67a61:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9f03258719b67a61, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_9f03258719b67a61
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 60
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           attention_q8_0_fa2_gqa_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     27
    .sgpr_spill_count: 0
    .symbol:         attention_q8_0_fa2_gqa_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     238
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           attention_fa2_q_preconvert_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .sgpr_spill_count: 0
    .symbol:         attention_fa2_q_preconvert_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     13
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           attention_q8_0_fa2_gqa_partial_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         attention_q8_0_fa2_gqa_partial_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     239
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           attention_q8_0_fa2_gqa_merge_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         attention_q8_0_fa2_gqa_merge_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     17
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
