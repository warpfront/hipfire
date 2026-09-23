	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_gfx1201:     ; @attention_fp8_e4m3_fa2_gqa_gfx1201
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
	s_cbranch_vccnz .LBB0_89
; %bb.1:
	s_cmp_gt_i32 ttmp7, 3
	s_cbranch_scc1 .LBB0_89
; %bb.2:
	s_load_b32 s22, s[0:1], 0x38
	s_lshl_b32 s3, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s3, s7
	s_cbranch_scc1 .LBB0_89
; %bb.3:
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v3, 0
	v_lshrrev_b32_e32 v156, 4, v0
	v_lshrrev_b32_e32 v157, 3, v0
	v_mov_b32_e32 v147, 0
	s_mov_b32 s23, 0
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_5
; %bb.4:
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_or_b32 v145, v0, 7, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 6, v1
	v_mul_lo_u32 v3, v145, 24
	v_cmp_gt_i32_e32 vcc_lo, s7, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, ttmp7, 6, v[1:2]
	v_lshrrev_b32_e32 v2, 3, v0
	s_and_b32 s23, vcc_lo, exec_lo
	v_and_or_b32 v147, v2, 1, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v2, v147, v3, 8
	v_mov_b32_e32 v3, 0
.LBB0_5:
	s_or_b32 exec_lo, exec_lo, s4
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x20
	v_dual_mov_b32 v5, -1 :: v_dual_and_b32 v6, 31, v0
	v_bfrev_b32_e32 v7, -2
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_u32_e32 8, v6
	s_cbranch_execz .LBB0_9
; %bb.6:
	v_or_b32_e32 v4, s3, v6
	v_bfrev_b32_e32 v7, -2
	v_mov_b32_e32 v5, -1
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s7, v4
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_ashrrev_i32_e32 v5, 31, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_add_co_u32 v4, vcc_lo, s0, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s1, v5, vcc_lo
	global_load_b32 v5, v[4:5], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v7, v5
.LBB0_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_9:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	s_or_b32 exec_lo, exec_lo, s4
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_cndmask_b32_e64 v14, 0, v3, s23
	v_dual_mov_b32 v166, 1.0 :: v_dual_and_b32 v159, 15, v0
	v_lshrrev_b32_e32 v161, 5, v0
	v_xor_b32_e32 v1, 16, v11
	v_xor_b32_e32 v3, 4, v11
	v_cndmask_b32_e64 v15, 0, v2, s23
	v_ashrrev_i32_e32 v146, 31, v145
	v_lshrrev_b32_e32 v2, 1, v6
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
	v_xor_b32_e32 v17, 1, v11
	v_lshl_add_u32 v160, v6, 3, 0
	v_lshlrev_b64_e32 v[129:130], 2, v[145:146]
	s_mov_b32 s17, 0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v11, v1, vcc_lo
	s_mul_i32 s16, s7, 0x1800
	s_mov_b32 s4, ttmp7
	s_ashr_i32 s5, ttmp7, 31
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[18:19], s[8:9], s[16:17]
	v_lshlrev_b32_e32 v158, 2, v1
	v_xor_b32_e32 v1, 8, v11
	s_lshl_b64 s[4:5], s[4:5], 8
	s_lshl_b32 s6, ttmp7, 1
	s_add_nc_u64 s[20:21], s[10:11], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s7, s6, 31
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
	s_mov_b32 s16, s17
	s_wait_alu depctr_va_vcc(0)
	v_dual_mov_b32 v168, 0xff800000 :: v_dual_cndmask_b32 v9, v11, v1
	ds_bpermute_b32 v4, v158, v5
	ds_bpermute_b32 v8, v158, v7
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	v_mov_b32_e32 v1, 0
	v_lshlrev_b32_e32 v9, 2, v9
	v_and_b32_e32 v146, 8, v2
	v_lshlrev_b32_e32 v162, 7, v159
	v_lshlrev_b32_e32 v18, 4, v159
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v2, v1
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v5, v4
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v7, v8
	v_lshrrev_b32_e32 v7, 1, v0
	v_lshlrev_b32_e32 v8, 3, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v11, v3, vcc_lo
	ds_bpermute_b32 v12, v9, v4
	ds_bpermute_b32 v13, v9, v5
	v_mov_b32_e32 v167, 0
	v_and_b32_e32 v19, 0xf8, v8
	v_lshlrev_b32_e32 v0, 2, v0
	v_and_b32_e32 v163, 8, v7
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v7, v1
	v_lshl_or_b32 v133, v161, 8, v19
	v_mad_co_u64_u32 v[9:10], null, v145, 24, v[147:148]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v10, v1 :: v_dual_add_nc_u32 v165, 0, v133
	v_lshlrev_b64_e32 v[131:132], 2, v[9:10]
	s_wait_dscnt 0x1
	v_max_i32_e32 v12, v4, v12
	s_wait_dscnt 0x0
	v_min_i32_e32 v13, v5, v13
	v_xor_b32_e32 v4, 2, v11
	v_mov_b32_e32 v5, v1
	v_add_co_u32 v148, s3, s20, v18
	ds_bpermute_b32 v16, v0, v12
	ds_bpermute_b32 v0, v0, v13
	v_cmp_gt_u32_e32 vcc_lo, 32, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v149, null, s21, 0, s3
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[20:21], s[10:11], s[6:7]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v11, v4, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v17
	v_mov_b32_e32 v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v66, v11, v17, vcc_lo
	v_add_co_u32 v134, vcc_lo, v15, v146
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v135, null, 0, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v89, 2, v66
	v_lshlrev_b32_e32 v20, 2, v8
	v_mov_b32_e32 v8, v1
	s_wait_dscnt 0x1
	v_max_i32_e32 v65, v12, v16
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v13, v0
	v_add_co_u32 v150, vcc_lo, s0, v129
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v11, v3
	v_mov_b32_e32 v13, v5
	ds_bpermute_b32 v67, v20, v65
	ds_bpermute_b32 v68, v20, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v151, null, s1, v130, vcc_lo
	v_add_co_u32 v129, vcc_lo, s8, v134
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, s9, v135, vcc_lo
	v_add_co_u32 v152, vcc_lo, s18, v131
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, s19, v132, vcc_lo
	v_add_co_u32 v154, vcc_lo, v129, 48
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v33, v1
	s_wait_dscnt 0x1
	v_max_i32_e32 v136, v65, v67
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v68
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v41, v1
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v49, v1
	ds_bpermute_b32 v137, v89, v136
	ds_bpermute_b32 v138, v89, v0
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v57, v1
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v65, v1
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v73, v1
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v81, v1
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v89, v1
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v97, v1
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v105, v1
	v_dual_mov_b32 v120, v8 :: v_dual_mov_b32 v113, v1
	v_dual_mov_b32 v128, v8 :: v_dual_mov_b32 v121, v1
	s_wait_dscnt 0x1
	v_max_i32_e32 v131, v136, v137
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v138
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v155, null, 0, v130, vcc_lo
	v_dual_mov_b32 v15, v7 :: v_dual_lshlrev_b32 v164, 4, v161
	v_readfirstlane_b32 s24, v131
	v_dual_mov_b32 v136, v8 :: v_dual_mov_b32 v129, v1
	v_dual_mov_b32 v14, v6 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v12, v4 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v10, v2 :: v_dual_mov_b32 v19, v3
	v_dual_mov_b32 v9, v1 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v39, v7 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v35, v3 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v71, v7 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v79, v7 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v77, v5 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v87, v7 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v85, v5 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v95, v7 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v93, v5 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v103, v7 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v101, v5 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v99, v3 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v111, v7 :: v_dual_mov_b32 v100, v4
	v_dual_mov_b32 v109, v5 :: v_dual_mov_b32 v98, v2
	v_dual_mov_b32 v107, v3 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v119, v7 :: v_dual_mov_b32 v108, v4
	v_dual_mov_b32 v117, v5 :: v_dual_mov_b32 v106, v2
	v_dual_mov_b32 v115, v3 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v127, v7 :: v_dual_mov_b32 v116, v4
	v_dual_mov_b32 v125, v5 :: v_dual_mov_b32 v114, v2
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v135, v7 :: v_dual_mov_b32 v124, v4
	v_dual_mov_b32 v133, v5 :: v_dual_mov_b32 v122, v2
	v_mov_b32_e32 v131, v3
	v_readfirstlane_b32 s25, v0
	v_mov_b32_e32 v134, v6
	v_mov_b32_e32 v132, v4
	v_mov_b32_e32 v130, v2
	s_add_nc_u64 s[18:19], s[12:13], s[4:5]
	s_add_nc_u64 s[12:13], s[12:13], s[6:7]
	s_branch .LBB0_13
.LBB0_10:                               ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b32_e32 v168, v2
.LBB0_11:                               ;   in Loop: Header=BB0_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s27
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB0_12:                               ;   in Loop: Header=BB0_13 Depth=1
	s_add_co_i32 s16, s16, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s16, 0x3fffffff
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s17, s17, 64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s11, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_87
.LBB0_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_16 Depth 2
                                        ;     Child Loop BB0_22 Depth 2
                                        ;     Child Loop BB0_44 Depth 2
                                        ;       Child Loop BB0_49 Depth 3
	s_lshl_b32 s26, s16, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s26, s24
	s_cselect_b32 s11, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s11
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_12
; %bb.14:                               ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b32_e32 v0, v156
	v_mov_b32_e32 v6, v157
	s_mov_b32 s0, 8
	s_branch .LBB0_16
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_and_b32_e32 v7, 0xe0, v6
	v_add_nc_u32_e32 v6, 16, v6
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	v_add_nc_u32_e32 v7, v7, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v7, v0, 15, v7
	v_add_nc_u32_e32 v0, 8, v0
	v_lshl_add_u32 v7, v7, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v7, v[2:3], v[4:5] offset1:16
	s_cbranch_scc1 .LBB0_18
.LBB0_16:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v7, s17, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v5, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_ge_i32_e64 s24, v7
	s_cbranch_execz .LBB0_15
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v7, v[148:149]
	global_load_b128 v[2:5], v[2:3], off
	s_branch .LBB0_15
.LBB0_18:                               ;   in Loop: Header=BB0_13 Depth=1
	v_or_b32_e32 v6, s26, v163
	v_dual_mov_b32 v7, v164 :: v_dual_mov_b32 v8, v161
	s_movk_i32 s0, 0xc000
	s_branch .LBB0_22
.LBB0_19:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB0_20:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_21:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v0, s0, v165
	v_add_nc_u32_e32 v8, 4, v8
	v_add_nc_u32_e32 v7, 64, v7
	s_addk_co_i32 s0, 0x400
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	ds_store_b64 v0, v[2:3] offset:32768
	s_cbranch_scc1 .LBB0_38
.LBB0_22:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0x70, v8
	v_add_nc_u32_e32 v137, v0, v6
	v_and_or_b32 v0, 0xf0, v7, v159
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v2, 7, v137
	v_cmp_ge_i32_e32 vcc_lo, s24, v2
                                        ; implicit-def: $vgpr2_vgpr3
	s_and_saveexec_b32 s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_22 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v137, s[18:19]
	v_mov_b16_e32 v5.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v139.l, v5.h
	v_add_co_u32 v2, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_clause 0x7
	global_load_u8 v0, v[2:3], off offset:1032
	global_load_u8 v4, v[2:3], off offset:2064
	global_load_d16_u8 v5, v[2:3], off
	global_load_u8 v137, v[2:3], off offset:3096
	global_load_u8 v138, v[2:3], off offset:5160
	global_load_u8 v140, v[2:3], off offset:4128
	global_load_u8 v141, v[2:3], off offset:7224
	global_load_d16_hi_u8 v139, v[2:3], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v0, 8, v0
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v137
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v138
	v_or_b32_e32 v0, v0, v5
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v0, v0, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v140, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v141
	v_or3_b32 v0, v0, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v3, v2, v139, v3
	v_or3_b32 v2, v0, 0, 0
                                        ; implicit-def: $vgpr0
.LBB0_24:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_21
; %bb.25:                               ;   in Loop: Header=BB0_22 Depth=2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_ge_i32_e64 s24, v137
	s_cbranch_execz .LBB0_20
; %bb.26:                               ;   in Loop: Header=BB0_22 Depth=2
	v_add_co_u32 v4, s4, s18, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s19, 0, s4
	v_mov_b16_e32 v0.h, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[2:3], null, 0x408, v137, v[4:5]
	global_load_d16_u8 v0, v[2:3], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
	v_cmpx_gt_i32_e64 s24, v137
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_22 Depth=2
	v_or_b32_e32 v2, 1, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, 0x408, v2, v[4:5]
	global_load_u8 v2, v[2:3], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v0, v2, 8, v0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
.LBB0_28:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 2, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_22 Depth=2
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 16, v2
.LBB0_30:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 3, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_22 Depth=2
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 24, v2
.LBB0_32:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 4, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_22 Depth=2
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
.LBB0_34:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 5, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_22 Depth=2
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 8, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v3, v0, v3
.LBB0_36:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 6, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_19
; %bb.37:                               ;   in Loop: Header=BB0_22 Depth=2
	v_mad_co_u64_u32 v[4:5], null, 0x408, v0, v[4:5]
	v_mov_b16_e32 v0.l, 0
	global_load_d16_hi_u8 v0, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
	s_branch .LBB0_19
.LBB0_38:                               ;   in Loop: Header=BB0_13 Depth=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s27, s2
	s_cbranch_execz .LBB0_11
; %bb.39:                               ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b32_e32 v0, s22
	s_and_saveexec_b32 s0, s23
	s_cbranch_execz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_13 Depth=1
	global_load_b32 v0, v[152:153], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v0, s22, v0
.LBB0_41:                               ;   in Loop: Header=BB0_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_or_b32 s0, s26, 16
	v_mov_b32_e32 v4, v160
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s24
	s_mov_b32 s28, 0
	s_cselect_b32 s29, -1, 0
	s_branch .LBB0_44
.LBB0_42:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_sub_f32_e32 v143, v168, v2
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v168
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	v_div_scale_f32 v172, null, v3, v3, v140
	v_div_scale_f32 v174, null, v3, v3, v139
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	v_div_scale_f32 v177, null, v3, v3, v137
	v_div_scale_f32 v176, null, v3, v3, v8
	v_rcp_f32_e32 v173, v172
	v_lshl_add_u32 v188, s28, 12, v160
	v_exp_f32_e32 v143, v143
	v_rcp_f32_e32 v180, v177
	v_rcp_f32_e32 v179, v176
	v_fma_f32 v175, -v172, v173, 1.0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v143, 0, v143, vcc_lo
	v_dual_fmac_f32 v173, v175, v173 :: v_dual_mul_f32 v144, v166, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_scale_f32 v166, null, v3, v3, v144
	v_div_scale_f32 v170, vcc_lo, v144, v3, v144
	v_rcp_f32_e32 v168, v166
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v169, -v166, v168, 1.0
	v_fmac_f32_e32 v168, v169, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v169, v170, v168
	v_fma_f32 v171, -v166, v169, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v169, v171, v168
	v_fma_f32 v166, -v166, v169, v170
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v166, v166, v168, v169
	v_div_fixup_f32 v6, v166, v3, v144
	v_div_scale_f32 v166, null, v3, v3, v141
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v134, v134, v6
	v_dual_mul_f32 v130, v130, v6 :: v_dual_fmac_f32 v5, v167, v143
	v_mul_f32_e32 v128, v128, v6
	v_div_scale_f32 v143, null, v3, v3, v142
	v_rcp_f32_e32 v168, v166
	v_dual_mul_f32 v136, v136, v6 :: v_dual_mul_f32 v135, v135, v6
	v_mul_f32_e32 v126, v126, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v144, v143
	v_dual_mul_f32 v133, v133, v6 :: v_dual_mul_f32 v124, v124, v6
	v_dual_mul_f32 v132, v132, v6 :: v_dual_mul_f32 v131, v131, v6
	v_mul_f32_e32 v122, v122, v6
	v_fma_f32 v170, -v166, v168, 1.0
	v_mul_f32_e32 v25, v25, v6
	v_dual_mul_f32 v129, v129, v6 :: v_dual_mul_f32 v120, v120, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v167, -v143, v144, 1.0
	v_dual_mul_f32 v33, v33, v6 :: v_dual_fmac_f32 v168, v170, v168
	v_div_scale_f32 v170, s0, v141, v3, v141
	v_dual_mul_f32 v127, v127, v6 :: v_dual_mul_f32 v118, v118, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v144, v167, v144
	v_div_scale_f32 v167, vcc_lo, v142, v3, v142
	v_dual_mul_f32 v16, v16, v6 :: v_dual_mul_f32 v125, v125, v6
	v_dual_mul_f32 v116, v116, v6 :: v_dual_mul_f32 v123, v123, v6
	v_dual_mul_f32 v114, v114, v6 :: v_dual_mul_f32 v169, v167, v144
	v_dual_mul_f32 v121, v121, v6 :: v_dual_mul_f32 v112, v112, v6
	v_dual_mul_f32 v119, v119, v6 :: v_dual_mul_f32 v110, v110, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v171, -v143, v169, v167
	v_dual_mul_f32 v117, v117, v6 :: v_dual_mul_f32 v108, v108, v6
	v_dual_mul_f32 v115, v115, v6 :: v_dual_mul_f32 v106, v106, v6
	v_fmac_f32_e32 v169, v171, v144
	v_mul_f32_e32 v171, v170, v168
	v_dual_mul_f32 v113, v113, v6 :: v_dual_mul_f32 v104, v104, v6
	v_dual_mul_f32 v111, v111, v6 :: v_dual_mul_f32 v102, v102, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v143, -v143, v169, v167
	v_fma_f32 v167, -v166, v171, v170
	v_dual_mul_f32 v109, v109, v6 :: v_dual_mul_f32 v100, v100, v6
	v_dual_mul_f32 v107, v107, v6 :: v_dual_mul_f32 v98, v98, v6
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v169
	v_fmac_f32_e32 v171, v167, v168
	v_div_scale_f32 v167, s1, v140, v3, v140
	s_mov_b32 vcc_lo, s0
	v_rcp_f32_e32 v144, v174
	v_div_fixup_f32 v143, v143, v3, v142
	v_fma_f32 v142, -v166, v171, v170
	v_mul_f32_e32 v166, v167, v173
	v_div_scale_f32 v170, null, v3, v3, v7
	v_dual_mul_f32 v105, v105, v6 :: v_dual_mul_f32 v96, v96, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v168, v142, v168, v171
	v_fma_f32 v171, -v172, v166, v167
	s_mov_b32 vcc_lo, s1
	v_rcp_f32_e32 v175, v170
	v_fma_f32 v169, -v174, v144, 1.0
	v_mul_f32_e32 v19, v19, v6
	v_fmac_f32_e32 v166, v171, v173
	v_mov_b16_e64 v142.h, 0
	v_mov_b16_e64 v142.l, v1.l
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, s0, v139, v3, v139
	v_fma_f32 v167, -v172, v166, v167
	v_fma_f32 v178, -v170, v175, 1.0
	v_div_fixup_f32 v168, v168, v3, v141
	v_mov_b16_e64 v141.h, v142.h
	v_mov_b16_e64 v141.l, v142.l
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v166, v167, v173, v166
	v_div_scale_f32 v167, null, v3, v3, v138
	v_fma_f32 v173, -v176, v179, 1.0
	s_mov_b32 vcc_lo, s0
	v_cvt_pk_fp8_f32 v141.l, v143, v168
	v_div_fixup_f32 v140, v166, v3, v140
	v_fma_f32 v166, -v177, v180, 1.0
	v_mul_f32_e32 v15, v15, v6
	v_fmac_f32_e32 v179, v173, v179
	v_div_scale_f32 v173, s3, v8, v3, v8
	v_dual_mul_f32 v103, v103, v6 :: v_dual_mul_f32 v94, v94, v6
	v_fmac_f32_e32 v180, v166, v180
	v_div_scale_f32 v166, s4, v137, v3, v137
	v_dual_mul_f32 v10, v10, v6 :: v_dual_mul_f32 v101, v101, v6
	v_dual_mul_f32 v92, v92, v6 :: v_dual_mul_f32 v99, v99, v6
	v_mul_f32_e32 v90, v90, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v182, v166, v180
	v_fmac_f32_e32 v175, v178, v175
	v_mul_f32_e32 v171, v169, v144
	v_rcp_f32_e32 v178, v167
	v_dual_mul_f32 v97, v97, v6 :: v_dual_mul_f32 v88, v88, v6
	v_dual_mul_f32 v95, v95, v6 :: v_dual_mul_f32 v86, v86, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v172, -v174, v171, v169
	v_dual_mul_f32 v93, v93, v6 :: v_dual_mul_f32 v84, v84, v6
	v_dual_mul_f32 v91, v91, v6 :: v_dual_mul_f32 v82, v82, v6
	v_fmac_f32_e32 v171, v172, v144
	v_div_scale_f32 v172, s1, v7, v3, v7
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v181, -v167, v178, 1.0
	v_mul_f32_e32 v17, v17, v6
	v_dual_mul_f32 v89, v89, v6 :: v_dual_mul_f32 v80, v80, v6
	v_fma_f32 v169, -v174, v171, v169
	v_mul_f32_e32 v174, v172, v175
	v_fmac_f32_e32 v178, v181, v178
	v_div_scale_f32 v181, s0, v138, v3, v138
	v_dual_mul_f32 v87, v87, v6 :: v_dual_mul_f32 v78, v78, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v169, v144, v171
	v_mul_f32_e32 v171, v173, v179
	v_fma_f32 v169, -v170, v174, v172
	v_mul_f32_e32 v183, v181, v178
	s_mov_b32 vcc_lo, s1
	v_div_fixup_f32 v139, v144, v3, v139
	v_fma_f32 v144, -v176, v171, v173
	v_fmac_f32_e32 v174, v169, v175
	v_fma_f32 v169, -v177, v182, v166
	v_dual_mul_f32 v85, v85, v6 :: v_dual_mul_f32 v76, v76, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v171, v144, v179
	v_fma_f32 v144, -v167, v183, v181
	v_fma_f32 v170, -v170, v174, v172
	v_fmac_f32_e32 v182, v169, v180
	v_cvt_pk_fp8_f32 v141.h, v140, v139
	v_dual_mul_f32 v83, v83, v6 :: v_dual_mul_f32 v74, v74, v6
	v_fmac_f32_e32 v183, v144, v178
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v169, v170, v175, v174
	v_fma_f32 v170, -v176, v171, v173
	v_fma_f32 v144, -v177, v182, v166
	s_mov_b32 vcc_lo, s3
	v_fma_f32 v167, -v167, v183, v181
	v_div_fixup_f32 v7, v169, v3, v7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v166, v170, v179, v171
	s_mov_b32 vcc_lo, s4
	v_dual_mul_f32 v81, v81, v6 :: v_dual_mul_f32 v72, v72, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v180, v182
	s_mov_b32 vcc_lo, s0
	v_div_fixup_f32 v8, v166, v3, v8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v167, v167, v178, v183
	v_dual_mul_f32 v79, v79, v6 :: v_dual_mul_f32 v70, v70, v6
	v_div_fixup_f32 v137, v144, v3, v137
	v_cvt_pk_fp8_f32 v142.l, v7, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v138, v167, v3, v138
	ds_load_b64 v[7:8], v188 offset:16384
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v77, v77, v6 :: v_dual_mul_f32 v68, v68, v6
	v_cvt_pk_fp8_f32 v142.h, v137, v138
	ds_load_b64 v[137:138], v188 offset:16640
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[139:140], v188 offset:16896
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[143:144], v188 offset:17152
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[166:167], v188 offset:17408
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[168:169], v188 offset:17664
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[170:171], v188 offset:17920
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[172:173], v188 offset:18176
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[174:175], v188 offset:18432
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[176:177], v188 offset:18688
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[178:179], v188 offset:18944
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[180:181], v188 offset:19200
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[182:183], v188 offset:19456
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[184:185], v188 offset:19712
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[186:187], v188 offset:19968
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[188:189], v188 offset:20224
	v_dual_mul_f32 v75, v75, v6 :: v_dual_mul_f32 v66, v66, v6
	v_dual_mul_f32 v73, v73, v6 :: v_dual_mul_f32 v64, v64, v6
	v_dual_mul_f32 v71, v71, v6 :: v_dual_mul_f32 v62, v62, v6
	v_dual_mul_f32 v69, v69, v6 :: v_dual_mul_f32 v60, v60, v6
	v_dual_mul_f32 v67, v67, v6 :: v_dual_mul_f32 v58, v58, v6
	v_dual_mul_f32 v65, v65, v6 :: v_dual_mul_f32 v56, v56, v6
	v_dual_mul_f32 v63, v63, v6 :: v_dual_mul_f32 v54, v54, v6
	v_dual_mul_f32 v61, v61, v6 :: v_dual_mul_f32 v52, v52, v6
	v_dual_mul_f32 v59, v59, v6 :: v_dual_mul_f32 v50, v50, v6
	v_dual_mul_f32 v57, v57, v6 :: v_dual_mul_f32 v48, v48, v6
	v_dual_mul_f32 v55, v55, v6 :: v_dual_mul_f32 v46, v46, v6
	v_dual_mul_f32 v53, v53, v6 :: v_dual_mul_f32 v44, v44, v6
	v_dual_mul_f32 v51, v51, v6 :: v_dual_mul_f32 v42, v42, v6
	v_dual_mul_f32 v49, v49, v6 :: v_dual_mul_f32 v40, v40, v6
	v_dual_mul_f32 v47, v47, v6 :: v_dual_mul_f32 v38, v38, v6
	v_dual_mul_f32 v45, v45, v6 :: v_dual_mul_f32 v36, v36, v6
	v_dual_mul_f32 v43, v43, v6 :: v_dual_mul_f32 v34, v34, v6
	v_dual_mul_f32 v41, v41, v6 :: v_dual_mul_f32 v32, v32, v6
	v_dual_mul_f32 v39, v39, v6 :: v_dual_mul_f32 v30, v30, v6
	v_dual_mul_f32 v37, v37, v6 :: v_dual_mul_f32 v28, v28, v6
	v_dual_mul_f32 v35, v35, v6 :: v_dual_mul_f32 v26, v26, v6
	v_dual_mul_f32 v31, v31, v6 :: v_dual_mul_f32 v24, v24, v6
	v_dual_mul_f32 v29, v29, v6 :: v_dual_mul_f32 v22, v22, v6
	v_dual_mul_f32 v27, v27, v6 :: v_dual_mul_f32 v20, v20, v6
	v_dual_mul_f32 v23, v23, v6 :: v_dual_mul_f32 v18, v18, v6
	v_dual_mul_f32 v21, v21, v6 :: v_dual_mul_f32 v14, v14, v6
	v_dual_mul_f32 v13, v13, v6 :: v_dual_mul_f32 v12, v12, v6
	v_mul_f32_e32 v11, v11, v6
	v_mul_f32_e32 v9, v9, v6
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[7:8], v[141:142], v[129:136]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[137:138], v[141:142], v[121:128]
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[139:140], v[141:142], v[113:120]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[143:144], v[141:142], v[105:112]
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[166:167], v[141:142], v[97:104]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[168:169], v[141:142], v[89:96]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[170:171], v[141:142], v[81:88]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[172:173], v[141:142], v[73:80]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[174:175], v[141:142], v[65:72]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[176:177], v[141:142], v[57:64]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[178:179], v[141:142], v[49:56]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[180:181], v[141:142], v[41:48]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[182:183], v[141:142], v[33:40]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[184:185], v[141:142], v[25:32]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[186:187], v[141:142], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[188:189], v[141:142], v[9:16]
	v_dual_mov_b32 v167, v5 :: v_dual_mov_b32 v166, v3
	;;#ASMSTART
	;;#ASMEND
.LBB0_43:                               ;   in Loop: Header=BB0_44 Depth=2
	v_add_nc_u32_e32 v4, 0x100, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v168, v2
	s_add_co_i32 s28, s28, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s28, 4
	s_cbranch_scc0 .LBB0_10
.LBB0_44:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_49 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s28, 1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB0_47
; %bb.45:                               ;   in Loop: Header=BB0_44 Depth=2
	s_cmp_eq_u32 s28, 1
	s_mov_b32 s0, s29
	s_cbranch_scc1 .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_44 Depth=2
	s_cmp_eq_u32 s28, 2
	s_cselect_b32 s0, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s24, s0
	s_cselect_b32 s0, -1, 0
.LBB0_47:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_84
; %bb.48:                               ;   in Loop: Header=BB0_44 Depth=2
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v2, v154
	v_mov_b32_e32 v3, v155
	s_movk_i32 s0, 0xc000
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_mov_b32_e32 v144, v137
.LBB0_49:                               ;   Parent Loop BB0_13 Depth=1
                                        ;     Parent Loop BB0_44 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_clause 0x3
	global_load_b64 v[173:174], v[2:3], off offset:-48
	global_load_b64 v[175:176], v[2:3], off offset:-32
	global_load_b64 v[177:178], v[2:3], off offset:-16
	global_load_b64 v[179:180], v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v169, s0, v4
	v_add_co_u32 v2, vcc_lo, v2, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	ds_load_2addr_stride64_b64 v[5:8], v169 offset0:32 offset1:34
	ds_load_2addr_stride64_b64 v[169:172], v169 offset0:36 offset1:38
	s_addk_co_i32 s0, 0x1000
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[5:6], v[173:174], v[137:144]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[7:8], v[175:176], v[137:144]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[169:170], v[177:178], v[137:144]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[171:172], v[179:180], v[137:144]
	s_cbranch_scc1 .LBB0_49
; %bb.50:                               ;   in Loop: Header=BB0_44 Depth=2
	s_lshl4_add_u32 s0, s28, s26
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v170, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 15
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s25
	s_cselect_b32 s30, -1, 0
	s_cmp_gt_i32 s1, s25
	s_cselect_b32 s1, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s1, s23
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s3
	s_cbranch_execz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_44 Depth=2
	global_load_b32 v170, v[150:151], off
.LBB0_52:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, s0, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e32 vcc_lo, s24, v2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[5:6], null, 0x408, v2, s[20:21]
	global_load_d16_b16 v3, v[5:6], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v173, v3.l
.LBB0_54:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 1, v2
	v_cmp_gt_i32_e64 s0, s24, v2
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[6:7], null, 0x408, v5, s[20:21]
	global_load_d16_b16 v3, v[6:7], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v175, v3.l
.LBB0_56:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v6, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s1, s24, v6
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[7:8], null, 0x408, v6, s[20:21]
	global_load_d16_b16 v3, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v3.l
.LBB0_58:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 3, v2
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v178, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e64 s3, s24, v7
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[171:172], null, 0x408, v7, s[20:21]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v178, v3.l
.LBB0_60:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v8, 4, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s4, s24, v8
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[171:172], null, 0x408, v8, s[20:21]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v176, v3.l
.LBB0_62:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v169, 5, v2
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e64 s5, s24, v169
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[171:172], null, 0x408, v169, s[20:21]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v179, v3.l
.LBB0_64:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v171, 6, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s6, s24, v171
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[180:181], null, 0x408, v171, s[20:21]
	global_load_d16_b16 v3, v[180:181], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v177, v3.l
.LBB0_66:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v172, 7, v2
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v180, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_ge_i32_e64 s7, s24, v172
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[180:181], null, 0x408, v172, s[20:21]
	global_load_d16_b16 v180, v[180:181], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v180, v180.l
.LBB0_68:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mul_f32 v137, v0, v137 :: v_dual_mul_f32 v138, v0, v138
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s8, v2, v170
	v_cmp_lt_i32_e64 s9, v2, v170
	v_dual_mul_f32 v139, v0, v139 :: v_dual_mul_f32 v140, v0, v140
	v_cmp_le_i32_e64 s10, v6, v170
	v_dual_mul_f32 v137, v137, v173 :: v_dual_mul_f32 v138, v138, v175
	s_or_b32 s8, s30, s8
	s_or_b32 s9, s30, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s23, s8
	v_dual_mul_f32 v173, v139, v174 :: v_dual_mul_f32 v144, v0, v144
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v137, 0xff800000, v137, s8
	s_or_b32 s10, s30, s10
	v_cmp_le_i32_e64 s8, v7, v170
	s_and_b32 s9, s23, s9
	v_mul_f32_e32 v142, v0, v142
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v139, 0xff800000, v138, s9
	s_and_b32 s9, s23, s10
	v_mul_f32_e32 v140, v140, v178
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, 0xff800000, v173, s9
	v_mul_f32_e32 v173, v0, v141
	s_or_b32 s10, s30, s8
	v_cmp_le_i32_e64 s8, v8, v170
	v_cmp_le_i32_e64 s9, v169, v170
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s10, s23, s10
	v_mul_f32_e32 v143, v0, v143
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v141, 0xff800000, v140, s10
	v_mul_f32_e32 v140, v173, v176
	s_or_b32 s8, s30, s8
	v_mul_f32_e32 v173, v142, v179
	s_or_b32 s9, s30, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s23, s8
	v_dual_mul_f32 v143, v143, v177 :: v_dual_mul_f32 v144, v144, v180
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v142, 0xff800000, v140, s8
	s_and_b32 s8, s23, s9
	v_cmp_le_i32_e64 s9, v172, v170
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v140, 0xff800000, v173, s8
	v_cmp_le_i32_e64 s8, v171, v170
	v_max3_num_f32 v170, v137, 0xff800000, v139
	s_or_b32 s9, s30, s9
	s_or_b32 s8, s30, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v170, v170, v138, v141
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s23, s8
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v173, 0xff800000, v143, s8
	s_and_b32 s8, s23, s9
	v_max3_num_f32 v143, v170, v142, v140
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v144, 0xff800000, v144, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v175, v143, v173, v144
	ds_bpermute_b32 v176, v158, v175
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v2, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v3, v2.l
.LBB0_70:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v170, 0
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[177:178], null, 0x408, v5, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v170, v2.l
.LBB0_72:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[5:6], null, 0x408, v6, s[12:13]
	global_load_d16_b16 v2, v[5:6], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v143, v2.l
.LBB0_74:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[6:7], null, 0x408, v7, s[12:13]
	global_load_d16_b16 v2, v[6:7], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v6, v2.l
.LBB0_76:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[7:8], null, 0x408, v8, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v5, v2.l
.LBB0_78:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v8, 0
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB0_85
; %bb.79:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execnz .LBB0_86
.LBB0_80:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB0_82
.LBB0_81:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[171:172], null, 0x408, v172, s[12:13]
	global_load_d16_b16 v2, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v169, v2.l
.LBB0_82:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_dscnt 0x0
	v_max3_num_f32 v2, v168, v175, v176
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v7, v137, v2 :: v_dual_sub_f32 v138, v138, v2
	v_dual_sub_f32 v137, v139, v2 :: v_dual_sub_f32 v140, v140, v2
	v_dual_sub_f32 v139, v141, v2 :: v_dual_sub_f32 v144, v144, v2
	v_dual_mul_f32 v7, 0x3fb8aa3b, v7 :: v_dual_mul_f32 v138, 0x3fb8aa3b, v138
	v_dual_sub_f32 v141, v142, v2 :: v_dual_sub_f32 v142, v173, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_mul_f32 v140, 0x3fb8aa3b, v140
	v_exp_f32_e32 v7, v7
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v2
	v_exp_f32_e32 v138, v138
	v_dual_mul_f32 v139, 0x3fb8aa3b, v139 :: v_dual_mul_f32 v144, 0x3fb8aa3b, v144
	v_exp_f32_e32 v137, v137
	v_exp_f32_e32 v140, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v139, v139
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, v7, 0, vcc_lo
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_mul_f32_e32 v171, 0x3fb8aa3b, v142
	v_exp_f32_e32 v144, v144
	v_mul_f32_e32 v142, v7, v3
	v_exp_f32_e32 v172, v141
	v_cndmask_b32_e64 v137, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	v_add_f32_e32 v3, v7, v137
	v_mul_f32_e32 v141, v137, v170
	v_cndmask_b32_e64 v170, v139, 0, vcc_lo
	v_exp_f32_e32 v137, v171
	v_cndmask_b32_e64 v171, v172, 0, vcc_lo
	v_cndmask_b32_e64 v172, v140, 0, vcc_lo
	v_mul_f32_e32 v140, v138, v143
	v_cndmask_b32_e64 v143, v144, 0, vcc_lo
	v_mul_f32_e32 v139, v170, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v7, v171, v5 :: v_dual_mul_f32 v8, v172, v8
	v_dual_add_f32 v3, v138, v3 :: v_dual_mul_f32 v138, v143, v169
	v_max3_num_f32 v6, v142, 0, v141
	v_cndmask_b32_e64 v5, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v3, v170, v3
	v_max3_num_f32 v6, v6, v140, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v137, v5, v174
	v_add_f32_e32 v3, v171, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v6, v6, v7, v8
	v_add_f32_e32 v3, v172, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v144, v6, v137, v138
	v_add_f32_e32 v3, v5, v3
	ds_bpermute_b32 v169, v158, v144
	v_add_f32_e32 v5, v143, v3
	ds_bpermute_b32 v6, v158, v5
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v3, v169, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v143, v144, v3
	v_mov_b32_e32 v3, v166
	v_cmpx_lt_f32_e32 0, v143
	s_cbranch_execz .LBB0_42
; %bb.83:                               ;   in Loop: Header=BB0_44 Depth=2
	v_div_scale_f32 v3, null, 0x43e00000, 0x43e00000, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v144, v3
	v_fma_f32 v169, -v3, v144, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, vcc_lo, v143, 0x43e00000, v143
	v_mul_f32_e32 v170, v169, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v171, -v3, v170, v169
	v_fmac_f32_e32 v170, v171, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v3, v170, v169
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v144, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v3, 0x43e00000, v143
	v_max_num_f32_e32 v3, 0x1f800000, v3
	s_branch .LBB0_42
.LBB0_84:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mov_b32_e32 v2, v168
	s_branch .LBB0_43
.LBB0_85:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[7:8], null, 0x408, v169, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v8, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execz .LBB0_80
.LBB0_86:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[177:178], null, 0x408, v171, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execnz .LBB0_81
	s_branch .LBB0_82
.LBB0_87:
	s_and_saveexec_b32 s0, s23
	s_cbranch_execz .LBB0_89
; %bb.88:
	v_div_scale_f32 v0, null, v167, v167, 1.0
	v_div_scale_f32 v3, vcc_lo, 1.0, v167, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v1, v0
	v_fma_f32 v2, -v0, v1, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v2, v1
	v_mul_f32_e32 v2, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v0, v2, v3
	v_fmac_f32_e32 v2, v4, v1
	v_mul_lo_u32 v4, 0x1800, v145
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v0, -v0, v2, v3
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v0, v0, v1, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v2, v147, 8, v4
	v_cmp_lt_f32_e32 vcc_lo, 0, v167
	v_mov_b32_e32 v1, 0
	v_div_fixup_f32 v3, v0, v167, 1.0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_or_b32_e32 v0, v2, v146
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_mul_f32_e32 v139, v166, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v137, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, s15, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v0, v129, v139 :: v_dual_mul_f32 v3, v132, v139
	v_dual_mul_f32 v1, v130, v139 :: v_dual_mul_f32 v2, v131, v139
	v_dual_mul_f32 v5, v134, v139 :: v_dual_mul_f32 v4, v133, v139
	v_dual_mul_f32 v7, v136, v139 :: v_dual_mul_f32 v6, v135, v139
	v_dual_mul_f32 v121, v121, v139 :: v_dual_mul_f32 v122, v122, v139
	v_dual_mul_f32 v123, v123, v139 :: v_dual_mul_f32 v124, v124, v139
	v_dual_mul_f32 v125, v125, v139 :: v_dual_mul_f32 v126, v126, v139
	v_dual_mul_f32 v127, v127, v139 :: v_dual_mul_f32 v128, v128, v139
	v_dual_mul_f32 v113, v113, v139 :: v_dual_mul_f32 v112, v112, v139
	v_dual_mul_f32 v97, v97, v139 :: v_dual_mul_f32 v98, v98, v139
	v_dual_mul_f32 v99, v99, v139 :: v_dual_mul_f32 v100, v100, v139
	v_dual_mul_f32 v101, v101, v139 :: v_dual_mul_f32 v114, v114, v139
	v_dual_mul_f32 v115, v115, v139 :: v_dual_mul_f32 v116, v116, v139
	v_dual_mul_f32 v117, v117, v139 :: v_dual_mul_f32 v102, v102, v139
	v_dual_mul_f32 v103, v103, v139 :: v_dual_mul_f32 v104, v104, v139
	v_dual_mul_f32 v118, v118, v139 :: v_dual_mul_f32 v119, v119, v139
	v_dual_mul_f32 v120, v120, v139 :: v_dual_mul_f32 v105, v105, v139
	v_dual_mul_f32 v106, v106, v139 :: v_dual_mul_f32 v107, v107, v139
	v_dual_mul_f32 v108, v108, v139 :: v_dual_mul_f32 v109, v109, v139
	v_dual_mul_f32 v110, v110, v139 :: v_dual_mul_f32 v111, v111, v139
	s_clause 0x7
	global_store_b128 v[137:138], v[0:3], off
	global_store_b128 v[137:138], v[4:7], off offset:16
	global_store_b128 v[137:138], v[121:124], off offset:64
	global_store_b128 v[137:138], v[125:128], off offset:80
	global_store_b128 v[137:138], v[113:116], off offset:128
	global_store_b128 v[137:138], v[117:120], off offset:144
	global_store_b128 v[137:138], v[105:108], off offset:192
	global_store_b128 v[137:138], v[109:112], off offset:208
	v_dual_mul_f32 v0, v89, v139 :: v_dual_mul_f32 v3, v92, v139
	v_dual_mul_f32 v1, v90, v139 :: v_dual_mul_f32 v2, v91, v139
	v_dual_mul_f32 v5, v94, v139 :: v_dual_mul_f32 v4, v93, v139
	v_dual_mul_f32 v7, v96, v139 :: v_dual_mul_f32 v6, v95, v139
	v_dual_mul_f32 v81, v81, v139 :: v_dual_mul_f32 v82, v82, v139
	v_dual_mul_f32 v83, v83, v139 :: v_dual_mul_f32 v84, v84, v139
	v_dual_mul_f32 v85, v85, v139 :: v_dual_mul_f32 v86, v86, v139
	v_dual_mul_f32 v87, v87, v139 :: v_dual_mul_f32 v88, v88, v139
	s_clause 0x5
	global_store_b128 v[137:138], v[97:100], off offset:256
	global_store_b128 v[137:138], v[101:104], off offset:272
	global_store_b128 v[137:138], v[0:3], off offset:320
	global_store_b128 v[137:138], v[4:7], off offset:336
	global_store_b128 v[137:138], v[81:84], off offset:384
	global_store_b128 v[137:138], v[85:88], off offset:400
	v_dual_mul_f32 v0, v73, v139 :: v_dual_mul_f32 v3, v76, v139
	v_dual_mul_f32 v1, v74, v139 :: v_dual_mul_f32 v2, v75, v139
	v_dual_mul_f32 v5, v78, v139 :: v_dual_mul_f32 v4, v77, v139
	v_dual_mul_f32 v7, v80, v139 :: v_dual_mul_f32 v6, v79, v139
	v_dual_mul_f32 v65, v65, v139 :: v_dual_mul_f32 v66, v66, v139
	v_dual_mul_f32 v67, v67, v139 :: v_dual_mul_f32 v68, v68, v139
	v_dual_mul_f32 v69, v69, v139 :: v_dual_mul_f32 v70, v70, v139
	v_dual_mul_f32 v71, v71, v139 :: v_dual_mul_f32 v72, v72, v139
	v_dual_mul_f32 v57, v57, v139 :: v_dual_mul_f32 v58, v58, v139
	v_dual_mul_f32 v59, v59, v139 :: v_dual_mul_f32 v60, v60, v139
	v_dual_mul_f32 v61, v61, v139 :: v_dual_mul_f32 v62, v62, v139
	v_dual_mul_f32 v63, v63, v139 :: v_dual_mul_f32 v64, v64, v139
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:448
	global_store_b128 v[137:138], v[4:7], off offset:464
	global_store_b128 v[137:138], v[65:68], off offset:512
	global_store_b128 v[137:138], v[69:72], off offset:528
	global_store_b128 v[137:138], v[57:60], off offset:576
	global_store_b128 v[137:138], v[61:64], off offset:592
	v_dual_mul_f32 v0, v49, v139 :: v_dual_mul_f32 v3, v52, v139
	v_dual_mul_f32 v1, v50, v139 :: v_dual_mul_f32 v2, v51, v139
	v_dual_mul_f32 v5, v54, v139 :: v_dual_mul_f32 v4, v53, v139
	v_dual_mul_f32 v7, v56, v139 :: v_dual_mul_f32 v6, v55, v139
	v_dual_mul_f32 v41, v41, v139 :: v_dual_mul_f32 v42, v42, v139
	v_dual_mul_f32 v43, v43, v139 :: v_dual_mul_f32 v44, v44, v139
	v_dual_mul_f32 v45, v45, v139 :: v_dual_mul_f32 v46, v46, v139
	v_dual_mul_f32 v47, v47, v139 :: v_dual_mul_f32 v48, v48, v139
	v_dual_mul_f32 v33, v33, v139 :: v_dual_mul_f32 v34, v34, v139
	v_dual_mul_f32 v35, v35, v139 :: v_dual_mul_f32 v36, v36, v139
	v_dual_mul_f32 v37, v37, v139 :: v_dual_mul_f32 v38, v38, v139
	v_dual_mul_f32 v39, v39, v139 :: v_dual_mul_f32 v40, v40, v139
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:640
	global_store_b128 v[137:138], v[4:7], off offset:656
	global_store_b128 v[137:138], v[41:44], off offset:704
	global_store_b128 v[137:138], v[45:48], off offset:720
	global_store_b128 v[137:138], v[33:36], off offset:768
	global_store_b128 v[137:138], v[37:40], off offset:784
	v_dual_mul_f32 v0, v25, v139 :: v_dual_mul_f32 v3, v28, v139
	v_dual_mul_f32 v1, v26, v139 :: v_dual_mul_f32 v2, v27, v139
	v_dual_mul_f32 v5, v30, v139 :: v_dual_mul_f32 v4, v29, v139
	v_dual_mul_f32 v7, v32, v139 :: v_dual_mul_f32 v6, v31, v139
	v_dual_mul_f32 v17, v17, v139 :: v_dual_mul_f32 v18, v18, v139
	v_dual_mul_f32 v19, v19, v139 :: v_dual_mul_f32 v20, v20, v139
	v_dual_mul_f32 v21, v21, v139 :: v_dual_mul_f32 v22, v22, v139
	v_dual_mul_f32 v23, v23, v139 :: v_dual_mul_f32 v24, v24, v139
	v_dual_mul_f32 v8, v9, v139 :: v_dual_mul_f32 v9, v10, v139
	v_dual_mul_f32 v10, v11, v139 :: v_dual_mul_f32 v11, v12, v139
	v_dual_mul_f32 v12, v13, v139 :: v_dual_mul_f32 v13, v14, v139
	v_dual_mul_f32 v14, v15, v139 :: v_dual_mul_f32 v15, v16, v139
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:832
	global_store_b128 v[137:138], v[4:7], off offset:848
	global_store_b128 v[137:138], v[17:20], off offset:896
	global_store_b128 v[137:138], v[21:24], off offset:912
	global_store_b128 v[137:138], v[8:11], off offset:960
	global_store_b128 v[137:138], v[12:15], off offset:976
.LBB0_89:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	attention_fp8_e4m3_fa2_gqa_gfx1201, .Lfunc_end0-attention_fp8_e4m3_fa2_gqa_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_gfx1201
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
		.amdhsa_next_free_vgpr 190
		.amdhsa_next_free_sgpr 31
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-attention_fp8_e4m3_fa2_gqa_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.num_vgpr, 190
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.numbered_sgpr, 31
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7996
; TotalNumSgprs: 33
; NumVgprs: 190
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 33
; NumVGPRsForWavesPerEU: 190
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
	.globl	attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201,@function
attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201: ; @attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
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
	.size	attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201, .Lfunc_end1-attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.num_vgpr, 13
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.numbered_sgpr, 4
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.has_indirect_call, 0
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
	.protected	attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
	.globl	attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201,@function
attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201: ; @attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x18
	v_lshrrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v1, ttmp9, 2, v1
	s_wait_kmcnt 0x0
	s_mul_i32 s2, s2, 24
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e32 vcc_lo, s2, v1
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB2_3
; %bb.1:
	v_mul_hi_i32 v2, 0x2aaaaaab, v1
	s_load_b128 s[8:11], s[0:1], 0x0
	v_mbcnt_lo_u32_b32 v12, -1, 0
	v_and_b32_e32 v14, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v13, 8, v12
	v_lshrrev_b32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v2, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v3, 0 :: v_dual_add_nc_u32 v2, v2, v3
	v_mul_lo_u32 v4, v2, 24
	v_mul_lo_u32 v2, 0x1800, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_nc_u32_e32 v6, v1, v4
	v_lshlrev_b64_e32 v[4:5], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v2, 8, v6
	s_wait_kmcnt 0x0
	v_add_co_u32 v0, vcc_lo, s8, v4
	v_lshlrev_b32_e32 v4, 5, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[6:7], 2, v[2:3]
	v_add_co_ci_u32_e64 v2, null, s9, v5, vcc_lo
	v_add_co_u32 v0, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v7, vcc_lo
	v_add_co_u32 v8, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, 0, v2, vcc_lo
	s_clause 0x1
	global_load_b128 v[4:7], v[8:9], off
	global_load_b128 v[8:11], v[8:9], off offset:16
	s_wait_loadcnt 0x1
	v_max_num_f32_e64 v0, |v5|, |v5|
	v_max_num_f32_e64 v2, |v4|, |v4|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v2, v0
	v_xor_b32_e32 v2, 16, v12
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v12, v2, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v13, v12, v13 :: v_dual_lshlrev_b32 v2, 2, v2
	v_lshlrev_b32_e32 v13, 2, v13
	v_max3_num_f32 v0, v0, |v6|, |v7|
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v0, v0, |v8|, |v9|
	v_max3_num_f32 v0, v0, |v10|, |v11|
	ds_bpermute_b32 v2, v2, v0
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v2
	ds_bpermute_b32 v2, v13, v0
	v_xor_b32_e32 v13, 4, v12
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v12, v13, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v2, v2, v2 :: v_dual_lshlrev_b32 v13, 2, v13
	v_max_num_f32_e32 v0, v0, v2
	ds_bpermute_b32 v2, v13, v0
	v_xor_b32_e32 v13, 2, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v12, v13, vcc_lo
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v2, v2, v2 :: v_dual_lshlrev_b32 v13, 2, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v2
	ds_bpermute_b32 v2, v13, v0
	v_xor_b32_e32 v13, 1, v12
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v12, 2, v12
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v2, v2, v2
	v_max_num_f32_e32 v0, v0, v2
	ds_bpermute_b32 v2, v12, v0
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v2
	v_div_scale_f32 v2, null, 0x43e00000, 0x43e00000, v0
	v_div_scale_f32 v15, vcc_lo, v0, 0x43e00000, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v12, v2
	v_fma_f32 v13, -v2, v12, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v12, v13, v12
	v_mul_f32_e32 v13, v15, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v2, v13, v15
	v_fmac_f32_e32 v13, v16, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f32 v2, -v2, v13, v15
	v_lshlrev_b32_e32 v15, 3, v14
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v2, v2, v12, v13
	v_cmp_neq_f32_e32 vcc_lo, 0, v0
	v_mov_b16_e32 v13.l, v3.l
	v_mov_b16_e32 v13.h, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v2, v2, 0x43e00000, v0
	v_mov_b16_e32 v12.l, v13.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v12.h, v13.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 1.0, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v30, null, v0, v0, v11
	v_div_scale_f32 v16, null, v0, v0, v4
	v_div_scale_f32 v18, null, v0, v0, v5
	v_div_scale_f32 v20, null, v0, v0, v6
	v_div_scale_f32 v22, null, v0, v0, v7
	v_rcp_f32_e32 v38, v30
	v_rcp_f32_e32 v31, v16
	v_rcp_f32_e32 v32, v18
	v_div_scale_f32 v24, null, v0, v0, v8
	v_rcp_f32_e32 v33, v20
	v_div_scale_f32 v26, null, v0, v0, v9
	v_rcp_f32_e32 v34, v22
	v_div_scale_f32 v39, s8, v11, v0, v11
	v_div_scale_f32 v28, null, v0, v0, v10
	v_fma_f32 v47, -v30, v38, 1.0
	v_rcp_f32_e32 v35, v24
	v_fma_f32 v40, -v16, v31, 1.0
	v_rcp_f32_e32 v36, v26
	v_fma_f32 v41, -v18, v32, 1.0
	v_fmac_f32_e32 v38, v47, v38
	v_rcp_f32_e32 v37, v28
	v_fma_f32 v42, -v20, v33, 1.0
	v_div_scale_f32 v17, vcc_lo, v4, v0, v4
	v_fmac_f32_e32 v31, v40, v31
	v_mul_f32_e32 v47, v39, v38
	v_fma_f32 v43, -v22, v34, 1.0
	v_div_scale_f32 v19, s2, v5, v0, v5
	v_fma_f32 v44, -v24, v35, 1.0
	v_div_scale_f32 v21, s3, v6, v0, v6
	v_fma_f32 v55, -v30, v47, v39
	v_dual_fmac_f32 v32, v41, v32 :: v_dual_fmac_f32 v33, v42, v33
	v_fma_f32 v45, -v26, v36, 1.0
	v_mul_f32_e32 v40, v17, v31
	v_div_scale_f32 v23, s4, v7, v0, v7
	v_fmac_f32_e32 v34, v43, v34
	v_fma_f32 v46, -v28, v37, 1.0
	v_fmac_f32_e32 v47, v55, v38
	v_mul_f32_e32 v41, v19, v32
	v_div_scale_f32 v25, s5, v8, v0, v8
	v_dual_fmac_f32 v35, v44, v35 :: v_dual_mul_f32 v42, v21, v33
	v_div_scale_f32 v27, s6, v9, v0, v9
	v_fmac_f32_e32 v36, v45, v36
	v_fma_f32 v48, -v16, v40, v17
	v_mul_f32_e32 v43, v23, v34
	v_div_scale_f32 v29, s7, v10, v0, v10
	v_fmac_f32_e32 v37, v46, v37
	v_fma_f32 v49, -v18, v41, v19
	v_mul_f32_e32 v44, v25, v35
	v_fma_f32 v50, -v20, v42, v21
	v_dual_mul_f32 v45, v27, v36 :: v_dual_fmac_f32 v40, v48, v31
	v_fma_f32 v51, -v22, v43, v23
	v_mul_f32_e32 v46, v29, v37
	v_fmac_f32_e32 v41, v49, v32
	v_fma_f32 v52, -v24, v44, v25
	v_fmac_f32_e32 v42, v50, v33
	v_fma_f32 v53, -v26, v45, v27
	v_fma_f32 v16, -v16, v40, v17
	v_dual_fmac_f32 v43, v51, v34 :: v_dual_lshlrev_b32 v2, 8, v1
	v_fma_f32 v54, -v28, v46, v29
	v_fma_f32 v17, -v18, v41, v19
	v_fmac_f32_e32 v44, v52, v35
	v_fma_f32 v18, -v20, v42, v21
	v_fmac_f32_e32 v45, v53, v36
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v31, v40
	s_mov_b32 vcc_lo, s2
	v_fma_f32 v19, -v22, v43, v23
	v_fmac_f32_e32 v46, v54, v37
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v32, v41
	s_mov_b32 vcc_lo, s3
	v_fma_f32 v20, -v24, v44, v25
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v33, v42
	s_mov_b32 vcc_lo, s4
	v_fma_f32 v21, -v26, v45, v27
	v_div_fixup_f32 v4, v16, v0, v4
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v16, v19, v34, v43
	s_mov_b32 vcc_lo, s5
	v_fma_f32 v22, -v28, v46, v29
	v_div_fixup_f32 v5, v17, v0, v5
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v20, v35, v44
	s_mov_b32 vcc_lo, s6
	v_fma_f32 v23, -v30, v47, v39
	v_div_fixup_f32 v6, v18, v0, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v21, v36, v45
	s_mov_b32 vcc_lo, s7
	v_add_co_u32 v2, s2, s10, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v19, v22, v37, v46
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v7, v16, v0, v7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v20, v23, v38, v47
	v_div_fixup_f32 v8, v17, v0, v8
	v_div_fixup_f32 v9, v18, v0, v9
	v_div_fixup_f32 v10, v19, v0, v10
	v_add_co_ci_u32_e64 v16, null, s11, 0, s2
	v_div_fixup_f32 v11, v20, v0, v11
	v_cvt_pk_fp8_f32 v12.l, v4, v5
	v_add_co_u32 v4, vcc_lo, v2, v15
	v_cvt_pk_fp8_f32 v12.h, v6, v7
	v_cvt_pk_fp8_f32 v13.l, v8, v9
	v_cvt_pk_fp8_f32 v13.h, v10, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v16, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	global_store_b64 v[4:5], v[12:13], off
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB2_3
; %bb.2:
	s_load_b64 s[0:1], s[0:1], 0x10
	v_mov_b32_e32 v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	global_store_b32 v[1:2], v0, off
.LBB2_3:
	s_endpgm
.Lfunc_end2:
	.size	attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201, .Lfunc_end2-attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
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
		.amdhsa_next_free_vgpr 56
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.num_vgpr, 56
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.numbered_sgpr, 12
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1484
; TotalNumSgprs: 14
; NumVgprs: 56
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 56
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
	.protected	attention_fp8_e4m3_fa2_gqa_partial_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_partial_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_partial_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_partial_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_partial_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_partial_gfx1201
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
	s_cbranch_vccnz .LBB3_93
; %bb.1:
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB3_93
; %bb.2:
	s_lshl_b32 s4, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s4, s7
	s_cbranch_scc1 .LBB3_93
; %bb.3:
	s_lshr_b32 s18, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s18, s17
	s_cbranch_scc1 .LBB3_93
; %bb.4:
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_lshrrev_b32_e32 v156, 4, v0
	v_lshrrev_b32_e32 v157, 3, v0
	s_mov_b32 s19, 0
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB3_6
; %bb.5:
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_or_b32 v2, v0, 7, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 6, v1
	v_cmp_gt_i32_e32 vcc_lo, s7, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[3:4], null, s3, 6, v[1:2]
	v_lshrrev_b32_e32 v1, 3, v0
	v_mul_lo_u32 v4, v2, 24
	s_and_b32 s19, vcc_lo, exec_lo
	v_and_or_b32 v5, v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v5, v4, 8
	v_mov_b32_e32 v4, 0
.LBB3_6:
	s_or_b32 exec_lo, exec_lo, s5
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x20
	v_and_b32_e32 v9, 31, v0
	v_mov_b32_e32 v7, -1
	v_bfrev_b32_e32 v8, -2
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_u32_e32 8, v9
	s_cbranch_execz .LBB3_10
; %bb.7:
	v_or_b32_e32 v6, s4, v9
	v_bfrev_b32_e32 v8, -2
	v_mov_b32_e32 v7, -1
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s7, v6
	s_cbranch_execz .LBB3_9
; %bb.8:
	v_ashrrev_i32_e32 v7, 31, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_wait_kmcnt 0x0
	v_add_co_u32 v6, vcc_lo, s0, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	global_load_b32 v7, v[6:7], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v8, v7
.LBB3_9:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB3_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_mbcnt_lo_u32_b32 v10, -1, 0
	v_lshrrev_b32_e32 v159, 4, v9
	v_lshrrev_b32_e32 v13, 1, v0
	v_dual_mov_b32 v167, 1.0 :: v_dual_lshlrev_b32 v14, 3, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_xor_b32_e32 v1, 16, v10
	v_xor_b32_e32 v12, 8, v10
	v_lshl_add_u32 v161, v9, 3, 0
	v_cndmask_b32_e64 v9, 0, v4, s19
	v_lshrrev_b32_e32 v162, 5, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
	v_xor_b32_e32 v15, 1, v10
	s_cvt_f32_u32 s4, s17
	s_add_co_i32 s6, s17, 0x1ff
	s_mov_b32 s5, 0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, v10, v1 :: v_dual_and_b32 v160, 15, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v12
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s20, s4
	s_and_b32 s6, s6, 0xffff
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v158, 2, v1
	v_and_b32_e32 v165, 8, v13
	v_and_b32_e32 v13, 0xf8, v14
	v_lshlrev_b32_e32 v163, 3, v159
	ds_bpermute_b32 v6, v158, v7
	ds_bpermute_b32 v11, v158, v8
	v_lshlrev_b32_e32 v166, 4, v162
	v_lshlrev_b32_e32 v133, 4, v160
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s6, s6
	v_dual_mov_b32 v154, 0xff800000 :: v_dual_mov_b32 v155, 0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s20, s6, s20
	s_trunc_f32 s20, s20
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[145:146], null, v2, 24, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v10, v12, vcc_lo
	v_max_i32_e32 v4, v7, v6
	v_xor_b32_e32 v7, 4, v10
	v_cndmask_b32_e64 v12, 0, v3, s19
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b32_e32 v0, 2, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v8, v11
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	v_lshlrev_b32_e32 v164, 7, v160
	v_xor_b32_e32 v11, 2, v10
	ds_bpermute_b32 v6, v0, v4
	ds_bpermute_b32 v0, v0, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v10, v7, vcc_lo
	v_lshlrev_b64_e32 v[129:130], 2, v[2:3]
	v_mov_b32_e32 v8, v1
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, v1 :: v_dual_lshlrev_b32 v7, 2, v7
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v146, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, v10, v11, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v15
	v_lshlrev_b64_e32 v[131:132], 2, v[145:146]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v134, v10, v15, vcc_lo
	s_wait_dscnt 0x1
	v_max_i32_e32 v41, v4, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v5, v0
	v_dual_mov_b32 v5, v1 :: v_dual_mov_b32 v4, v1
	v_mov_b32_e32 v6, v1
	ds_bpermute_b32 v42, v7, v41
	ds_bpermute_b32 v43, v7, v0
	v_mov_b32_e32 v7, v1
	v_lshlrev_b32_e32 v65, 2, v44
	v_add_co_u32 v136, vcc_lo, v12, v163
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v137, null, 0, v9, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v146, vcc_lo, s0, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v147, null, s1, v130, vcc_lo
	v_lshlrev_b32_e32 v129, 2, v134
	s_xor_b32 s0, s20, 0x80000000
	v_lshl_or_b32 v135, v162, 8, v13
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s6, s0, s4
	s_cvt_u32_f32 s0, s20
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v7
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s6, 31
	s_wait_dscnt 0x1
	v_max_i32_e32 v138, v41, v42
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v43
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v41, v1
	v_mov_b32_e32 v56, v8
	ds_bpermute_b32 v139, v65, v138
	ds_bpermute_b32 v140, v65, v0
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v64, v8
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v72, v8
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v80, v8
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v88, v8
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v96, v8
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v104, v8
	v_dual_mov_b32 v97, v1 :: v_dual_mov_b32 v112, v8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s6, s4
	v_dual_mov_b32 v105, v1 :: v_dual_mov_b32 v120, v8
	v_dual_mov_b32 v113, v1 :: v_dual_mov_b32 v128, v8
	s_wait_dscnt 0x1
	v_max_i32_e32 v130, v138, v139
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v140
	v_dual_mov_b32 v121, v1 :: v_dual_add_nc_u32 v168, 0, v135
	v_add_co_u32 v135, vcc_lo, s8, v136
	ds_bpermute_b32 v134, v129, v130
	ds_bpermute_b32 v129, v129, v0
	s_add_co_ci_u32 s6, s0, 0
	s_lshl_b32 s4, s3, 8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v136, null, s9, v137, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[4:5]
	s_add_nc_u64 s[20:21], s[12:13], s[4:5]
	s_mul_i32 s4, s7, 0x1800
	v_add_co_u32 v148, vcc_lo, v135, 48
	s_and_b32 s22, s6, 0xffff
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[8:9], s[4:5]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v149, null, 0, v136, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v152, vcc_lo, s6, v131
	v_add_co_u32 v150, s0, s0, v133
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	s_wait_dscnt 0x1
	v_max_i32_e32 v130, v130, v134
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v129
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, s7, v132, vcc_lo
	v_readfirstlane_b32 s26, v130
	v_mov_b32_e32 v136, v8
	v_dual_mov_b32 v14, v6 :: v_dual_mov_b32 v13, v5
	v_dual_mov_b32 v12, v4 :: v_dual_mov_b32 v11, v3
	v_dual_mov_b32 v10, v2 :: v_dual_mov_b32 v9, v1
	v_dual_mov_b32 v22, v6 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v20, v4 :: v_dual_mov_b32 v19, v3
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v30, v6 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v28, v4 :: v_dual_mov_b32 v27, v3
	v_dual_mov_b32 v26, v2 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v36, v4 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v33, v1
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v71, v7 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v79, v7 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v77, v5 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v87, v7 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v85, v5 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v95, v7 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v93, v5 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v103, v7 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v101, v5 :: v_dual_mov_b32 v100, v4
	v_dual_mov_b32 v99, v3 :: v_dual_mov_b32 v98, v2
	v_dual_mov_b32 v111, v7 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v109, v5 :: v_dual_mov_b32 v108, v4
	v_dual_mov_b32 v107, v3 :: v_dual_mov_b32 v106, v2
	v_dual_mov_b32 v119, v7 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v117, v5 :: v_dual_mov_b32 v116, v4
	v_dual_mov_b32 v115, v3 :: v_dual_mov_b32 v114, v2
	v_dual_mov_b32 v127, v7 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v125, v5 :: v_dual_mov_b32 v124, v4
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v122, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v151, null, s1, 0, s0
	v_readfirstlane_b32 s27, v0
	v_dual_mov_b32 v135, v7 :: v_dual_mov_b32 v134, v6
	v_dual_mov_b32 v133, v5 :: v_dual_mov_b32 v132, v4
	v_dual_mov_b32 v131, v3 :: v_dual_mov_b32 v130, v2
	v_mov_b32_e32 v129, v1
	s_mul_i32 s24, s18, s22
	s_lshl_b32 s4, s3, 1
	s_add_co_i32 s25, s24, s22
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[10:11], s[4:5]
	s_add_nc_u64 s[12:13], s[12:13], s[4:5]
	s_lshl_b32 s11, s24, 6
	s_branch .LBB3_14
.LBB3_11:                               ;   in Loop: Header=BB3_14 Depth=1
	v_mov_b32_e32 v154, v2
.LBB3_12:                               ;   in Loop: Header=BB3_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB3_13:                               ;   in Loop: Header=BB3_14 Depth=1
	s_add_co_i32 s24, s24, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s24, s25
	s_cselect_b32 s0, -1, 0
	s_xor_b32 s1, s28, -1
	s_add_co_i32 s11, s11, 64
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_88
.LBB3_14:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_17 Depth 2
                                        ;     Child Loop BB3_23 Depth 2
                                        ;     Child Loop BB3_45 Depth 2
                                        ;       Child Loop BB3_50 Depth 3
	s_lshl_b32 s29, s24, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s29, s26
	s_cselect_b32 s28, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_13
; %bb.15:                               ;   in Loop: Header=BB3_14 Depth=1
	v_mov_b32_e32 v0, v156
	v_mov_b32_e32 v6, v157
	s_mov_b32 s0, 8
	s_branch .LBB3_17
.LBB3_16:                               ;   in Loop: Header=BB3_17 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_and_b32_e32 v7, 0xe0, v6
	v_add_nc_u32_e32 v6, 16, v6
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	v_add_nc_u32_e32 v7, v7, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v7, v0, 15, v7
	v_add_nc_u32_e32 v0, 8, v0
	v_lshl_add_u32 v7, v7, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v7, v[2:3], v[4:5] offset1:16
	s_cbranch_scc1 .LBB3_19
.LBB3_17:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v7, s11, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v5, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_ge_i32_e64 s26, v7
	s_cbranch_execz .LBB3_16
; %bb.18:                               ;   in Loop: Header=BB3_17 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v7, v[150:151]
	global_load_b128 v[2:5], v[2:3], off
	s_branch .LBB3_16
.LBB3_19:                               ;   in Loop: Header=BB3_14 Depth=1
	v_or_b32_e32 v6, s29, v165
	v_dual_mov_b32 v7, v166 :: v_dual_mov_b32 v8, v162
	s_movk_i32 s0, 0xc000
	s_branch .LBB3_23
.LBB3_20:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB3_21:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB3_22:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v0, s0, v168
	v_add_nc_u32_e32 v8, 4, v8
	v_add_nc_u32_e32 v7, 64, v7
	s_addk_co_i32 s0, 0x400
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	ds_store_b64 v0, v[2:3] offset:32768
	s_cbranch_scc1 .LBB3_39
.LBB3_23:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0x70, v8
	v_add_nc_u32_e32 v137, v0, v6
	v_and_or_b32 v0, 0xf0, v7, v160
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v2, 7, v137
	v_cmp_ge_i32_e32 vcc_lo, s26, v2
                                        ; implicit-def: $vgpr2_vgpr3
	s_and_saveexec_b32 s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB3_25
; %bb.24:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v137, s[20:21]
	v_mov_b16_e32 v5.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v139.l, v5.h
	v_add_co_u32 v2, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_clause 0x7
	global_load_u8 v0, v[2:3], off offset:1032
	global_load_u8 v4, v[2:3], off offset:2064
	global_load_d16_u8 v5, v[2:3], off
	global_load_u8 v137, v[2:3], off offset:3096
	global_load_u8 v138, v[2:3], off offset:5160
	global_load_u8 v140, v[2:3], off offset:4128
	global_load_u8 v141, v[2:3], off offset:7224
	global_load_d16_hi_u8 v139, v[2:3], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v0, 8, v0
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v137
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v138
	v_or_b32_e32 v0, v0, v5
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v0, v0, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v140, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v141
	v_or3_b32 v0, v0, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v3, v2, v139, v3
	v_or3_b32 v2, v0, 0, 0
                                        ; implicit-def: $vgpr0
.LBB3_25:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB3_22
; %bb.26:                               ;   in Loop: Header=BB3_23 Depth=2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_ge_i32_e64 s26, v137
	s_cbranch_execz .LBB3_21
; %bb.27:                               ;   in Loop: Header=BB3_23 Depth=2
	v_add_co_u32 v4, s4, s20, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s21, 0, s4
	v_mov_b16_e32 v0.h, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, 0x408, v137, v[4:5]
	global_load_d16_u8 v0, v[2:3], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
	v_cmpx_gt_i32_e64 s26, v137
	s_cbranch_execz .LBB3_29
; %bb.28:                               ;   in Loop: Header=BB3_23 Depth=2
	v_or_b32_e32 v2, 1, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, v[4:5]
	global_load_u8 v2, v[2:3], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v0, v2, 8, v0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
.LBB3_29:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 2, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_31
; %bb.30:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 16, v2
.LBB3_31:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 3, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_33
; %bb.32:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 24, v2
.LBB3_33:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 4, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_35
; %bb.34:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
.LBB3_35:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 5, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_37
; %bb.36:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 8, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v3, v0, v3
.LBB3_37:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 6, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_20
; %bb.38:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[4:5], null, 0x408, v0, v[4:5]
	v_mov_b16_e32 v0.l, 0
	global_load_d16_hi_u8 v0, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
	s_branch .LBB3_20
.LBB3_39:                               ;   in Loop: Header=BB3_14 Depth=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB3_12
; %bb.40:                               ;   in Loop: Header=BB3_14 Depth=1
	v_mov_b32_e32 v0, s16
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB3_42
; %bb.41:                               ;   in Loop: Header=BB3_14 Depth=1
	global_load_b32 v0, v[152:153], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v0, s16, v0
.LBB3_42:                               ;   in Loop: Header=BB3_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_or_b32 s0, s29, 16
	v_mov_b32_e32 v4, v161
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s26
	s_mov_b32 s31, 0
	s_cselect_b32 s33, -1, 0
	s_branch .LBB3_45
.LBB3_43:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_sub_f32_e32 v143, v154, v2
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v154
	v_div_scale_f32 v172, null, v3, v3, v140
	v_div_scale_f32 v174, null, v3, v3, v139
	v_div_scale_f32 v176, null, v3, v3, v8
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	v_div_scale_f32 v177, null, v3, v3, v137
	v_rcp_f32_e32 v173, v172
	v_lshl_add_u32 v189, s31, 12, v161
	v_rcp_f32_e32 v179, v176
	v_exp_f32_e32 v143, v143
	v_rcp_f32_e32 v180, v177
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	v_fma_f32 v175, -v172, v173, 1.0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v143, 0, v143, vcc_lo
	v_fmac_f32_e32 v173, v175, v173
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v144, v167, v143
	v_div_scale_f32 v154, null, v3, v3, v144
	v_div_scale_f32 v170, vcc_lo, v144, v3, v144
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v167, v154
	v_fma_f32 v169, -v154, v167, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v167, v169, v167
	v_mul_f32_e32 v169, v170, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v171, -v154, v169, v170
	v_fmac_f32_e32 v169, v171, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v154, v169, v170
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v154, v154, v167, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v6, v154, v3, v144
	v_div_scale_f32 v154, null, v3, v3, v141
	v_dual_mul_f32 v134, v134, v6 :: v_dual_fmac_f32 v5, v155, v143
	v_mul_f32_e32 v126, v126, v6
	v_div_scale_f32 v143, null, v3, v3, v142
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v167, v154
	v_dual_mul_f32 v136, v136, v6 :: v_dual_mul_f32 v135, v135, v6
	v_mul_f32_e32 v124, v124, v6
	v_rcp_f32_e32 v144, v143
	v_dual_mul_f32 v133, v133, v6 :: v_dual_mul_f32 v122, v122, v6
	v_dual_mul_f32 v132, v132, v6 :: v_dual_mul_f32 v131, v131, v6
	v_mul_f32_e32 v120, v120, v6
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v170, -v154, v167, 1.0
	v_dual_mul_f32 v130, v130, v6 :: v_dual_mul_f32 v129, v129, v6
	v_mul_f32_e32 v118, v118, v6
	v_fma_f32 v155, -v143, v144, 1.0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v167, v170, v167
	v_div_scale_f32 v170, s0, v141, v3, v141
	v_dual_mul_f32 v128, v128, v6 :: v_dual_mul_f32 v127, v127, v6
	v_mul_f32_e32 v116, v116, v6
	v_fmac_f32_e32 v144, v155, v144
	v_div_scale_f32 v155, vcc_lo, v142, v3, v142
	v_dual_mul_f32 v125, v125, v6 :: v_dual_mul_f32 v114, v114, v6
	v_dual_mul_f32 v123, v123, v6 :: v_dual_mul_f32 v112, v112, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v169, v155, v144
	v_dual_mul_f32 v121, v121, v6 :: v_dual_mul_f32 v110, v110, v6
	v_dual_mul_f32 v119, v119, v6 :: v_dual_mul_f32 v108, v108, v6
	v_fma_f32 v171, -v143, v169, v155
	v_dual_mul_f32 v12, v12, v6 :: v_dual_mul_f32 v117, v117, v6
	v_dual_mul_f32 v106, v106, v6 :: v_dual_mul_f32 v115, v115, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v104, v104, v6 :: v_dual_fmac_f32 v169, v171, v144
	v_mul_f32_e32 v171, v170, v167
	v_dual_mul_f32 v113, v113, v6 :: v_dual_mul_f32 v102, v102, v6
	v_mul_f32_e32 v10, v10, v6
	v_fma_f32 v143, -v143, v169, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v155, -v154, v171, v170
	v_dual_mul_f32 v111, v111, v6 :: v_dual_mul_f32 v100, v100, v6
	v_dual_mul_f32 v109, v109, v6 :: v_dual_mul_f32 v98, v98, v6
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v169
	v_fmac_f32_e32 v171, v155, v167
	v_div_scale_f32 v155, s1, v140, v3, v140
	v_rcp_f32_e32 v144, v174
	s_mov_b32 vcc_lo, s0
	v_div_fixup_f32 v143, v143, v3, v142
	v_fma_f32 v142, -v154, v171, v170
	v_mul_f32_e32 v154, v155, v173
	v_div_scale_f32 v170, null, v3, v3, v7
	v_dual_mul_f32 v107, v107, v6 :: v_dual_mul_f32 v96, v96, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v167, v142, v167, v171
	v_fma_f32 v171, -v172, v154, v155
	v_fma_f32 v169, -v174, v144, 1.0
	v_mul_f32_e32 v21, v21, v6
	v_rcp_f32_e32 v175, v170
	v_mul_f32_e32 v23, v23, v6
	v_fmac_f32_e32 v154, v171, v173
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, s0, v139, v3, v139
	s_mov_b32 vcc_lo, s1
	v_mov_b16_e64 v142.h, 0
	v_fma_f32 v155, -v172, v154, v155
	v_mov_b16_e64 v142.l, v1.l
	v_mul_f32_e32 v171, v169, v144
	v_fma_f32 v178, -v170, v175, 1.0
	v_div_fixup_f32 v167, v167, v3, v141
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v154, v155, v173, v154
	v_div_scale_f32 v155, null, v3, v3, v138
	v_fma_f32 v172, -v174, v171, v169
	v_fmac_f32_e32 v175, v178, v175
	v_fma_f32 v173, -v176, v179, 1.0
	v_div_fixup_f32 v140, v154, v3, v140
	v_rcp_f32_e32 v178, v155
	v_fmac_f32_e32 v171, v172, v144
	v_div_scale_f32 v172, s1, v7, v3, v7
	v_fma_f32 v154, -v177, v180, 1.0
	v_fmac_f32_e32 v179, v173, v179
	v_div_scale_f32 v173, s3, v8, v3, v8
	v_fma_f32 v169, -v174, v171, v169
	v_dual_mul_f32 v19, v19, v6 :: v_dual_mul_f32 v174, v172, v175
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v181, -v155, v178, 1.0
	v_dual_mul_f32 v17, v17, v6 :: v_dual_fmac_f32 v180, v154, v180
	v_div_scale_f32 v154, s4, v137, v3, v137
	s_mov_b32 vcc_lo, s0
	v_fmac_f32_e32 v178, v181, v178
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v169, v144, v171
	v_mul_f32_e32 v171, v173, v179
	v_div_scale_f32 v181, s0, v138, v3, v138
	v_fma_f32 v169, -v170, v174, v172
	v_dual_mul_f32 v15, v15, v6 :: v_dual_mul_f32 v182, v154, v180
	v_div_fixup_f32 v139, v144, v3, v139
	v_fma_f32 v144, -v176, v171, v173
	v_mul_f32_e32 v183, v181, v178
	v_fmac_f32_e32 v174, v169, v175
	v_fma_f32 v169, -v177, v182, v154
	s_mov_b32 vcc_lo, s1
	v_fmac_f32_e32 v171, v144, v179
	v_fma_f32 v144, -v155, v183, v181
	v_fma_f32 v170, -v170, v174, v172
	v_fmac_f32_e32 v182, v169, v180
	v_mov_b16_e64 v141.h, v142.h
	v_mov_b16_e64 v141.l, v142.l
	v_fmac_f32_e32 v183, v144, v178
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v169, v170, v175, v174
	v_fma_f32 v170, -v176, v171, v173
	v_fma_f32 v144, -v177, v182, v154
	s_mov_b32 vcc_lo, s3
	v_fma_f32 v155, -v155, v183, v181
	v_div_fixup_f32 v7, v169, v3, v7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v154, v170, v179, v171
	s_mov_b32 vcc_lo, s4
	v_cvt_pk_fp8_f32 v141.h, v140, v139
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v180, v182
	s_mov_b32 vcc_lo, s0
	v_div_fixup_f32 v8, v154, v3, v8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v155, v155, v178, v183
	v_cvt_pk_fp8_f32 v141.l, v143, v167
	v_div_fixup_f32 v137, v144, v3, v137
	v_dual_mul_f32 v105, v105, v6 :: v_dual_mul_f32 v94, v94, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v138, v155, v3, v138
	v_cvt_pk_fp8_f32 v142.l, v7, v8
	ds_load_b64 v[7:8], v189 offset:16384
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v103, v103, v6 :: v_dual_mul_f32 v92, v92, v6
	v_cvt_pk_fp8_f32 v142.h, v137, v138
	ds_load_b64 v[137:138], v189 offset:16640
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[139:140], v189 offset:16896
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[143:144], v189 offset:17152
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[154:155], v189 offset:17408
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[169:170], v189 offset:17664
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[171:172], v189 offset:17920
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[173:174], v189 offset:18176
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[175:176], v189 offset:18432
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[177:178], v189 offset:18688
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[179:180], v189 offset:18944
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[181:182], v189 offset:19200
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[183:184], v189 offset:19456
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[185:186], v189 offset:19712
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[187:188], v189 offset:19968
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[189:190], v189 offset:20224
	v_dual_mul_f32 v101, v101, v6 :: v_dual_mul_f32 v90, v90, v6
	v_dual_mul_f32 v99, v99, v6 :: v_dual_mul_f32 v88, v88, v6
	v_dual_mul_f32 v97, v97, v6 :: v_dual_mul_f32 v86, v86, v6
	v_dual_mul_f32 v95, v95, v6 :: v_dual_mul_f32 v84, v84, v6
	v_dual_mul_f32 v93, v93, v6 :: v_dual_mul_f32 v82, v82, v6
	v_dual_mul_f32 v91, v91, v6 :: v_dual_mul_f32 v80, v80, v6
	v_dual_mul_f32 v89, v89, v6 :: v_dual_mul_f32 v78, v78, v6
	v_dual_mul_f32 v87, v87, v6 :: v_dual_mul_f32 v76, v76, v6
	v_dual_mul_f32 v85, v85, v6 :: v_dual_mul_f32 v74, v74, v6
	v_dual_mul_f32 v83, v83, v6 :: v_dual_mul_f32 v72, v72, v6
	v_dual_mul_f32 v81, v81, v6 :: v_dual_mul_f32 v70, v70, v6
	v_dual_mul_f32 v79, v79, v6 :: v_dual_mul_f32 v68, v68, v6
	v_dual_mul_f32 v77, v77, v6 :: v_dual_mul_f32 v66, v66, v6
	v_dual_mul_f32 v75, v75, v6 :: v_dual_mul_f32 v64, v64, v6
	v_dual_mul_f32 v73, v73, v6 :: v_dual_mul_f32 v62, v62, v6
	v_dual_mul_f32 v71, v71, v6 :: v_dual_mul_f32 v60, v60, v6
	v_dual_mul_f32 v69, v69, v6 :: v_dual_mul_f32 v58, v58, v6
	v_dual_mul_f32 v67, v67, v6 :: v_dual_mul_f32 v56, v56, v6
	v_dual_mul_f32 v65, v65, v6 :: v_dual_mul_f32 v54, v54, v6
	v_dual_mul_f32 v63, v63, v6 :: v_dual_mul_f32 v52, v52, v6
	v_dual_mul_f32 v61, v61, v6 :: v_dual_mul_f32 v50, v50, v6
	v_dual_mul_f32 v59, v59, v6 :: v_dual_mul_f32 v48, v48, v6
	v_dual_mul_f32 v57, v57, v6 :: v_dual_mul_f32 v46, v46, v6
	v_dual_mul_f32 v55, v55, v6 :: v_dual_mul_f32 v44, v44, v6
	v_dual_mul_f32 v53, v53, v6 :: v_dual_mul_f32 v42, v42, v6
	v_dual_mul_f32 v51, v51, v6 :: v_dual_mul_f32 v40, v40, v6
	v_dual_mul_f32 v49, v49, v6 :: v_dual_mul_f32 v38, v38, v6
	v_dual_mul_f32 v47, v47, v6 :: v_dual_mul_f32 v36, v36, v6
	v_dual_mul_f32 v45, v45, v6 :: v_dual_mul_f32 v34, v34, v6
	v_dual_mul_f32 v43, v43, v6 :: v_dual_mul_f32 v32, v32, v6
	v_dual_mul_f32 v41, v41, v6 :: v_dual_mul_f32 v30, v30, v6
	v_dual_mul_f32 v39, v39, v6 :: v_dual_mul_f32 v28, v28, v6
	v_dual_mul_f32 v37, v37, v6 :: v_dual_mul_f32 v26, v26, v6
	v_dual_mul_f32 v35, v35, v6 :: v_dual_mul_f32 v24, v24, v6
	v_dual_mul_f32 v33, v33, v6 :: v_dual_mul_f32 v22, v22, v6
	v_dual_mul_f32 v31, v31, v6 :: v_dual_mul_f32 v20, v20, v6
	v_dual_mul_f32 v29, v29, v6 :: v_dual_mul_f32 v18, v18, v6
	v_dual_mul_f32 v27, v27, v6 :: v_dual_mul_f32 v16, v16, v6
	v_dual_mul_f32 v25, v25, v6 :: v_dual_mul_f32 v14, v14, v6
	v_mul_f32_e32 v13, v13, v6
	v_mul_f32_e32 v11, v11, v6
	v_mul_f32_e32 v9, v9, v6
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[7:8], v[141:142], v[129:136]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[137:138], v[141:142], v[121:128]
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[139:140], v[141:142], v[113:120]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[143:144], v[141:142], v[105:112]
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[154:155], v[141:142], v[97:104]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[169:170], v[141:142], v[89:96]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[171:172], v[141:142], v[81:88]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[173:174], v[141:142], v[73:80]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[175:176], v[141:142], v[65:72]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[177:178], v[141:142], v[57:64]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[179:180], v[141:142], v[49:56]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[181:182], v[141:142], v[41:48]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[183:184], v[141:142], v[33:40]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[185:186], v[141:142], v[25:32]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[187:188], v[141:142], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[189:190], v[141:142], v[9:16]
	v_mov_b32_e32 v155, v5
	v_mov_b32_e32 v167, v3
	;;#ASMSTART
	;;#ASMEND
.LBB3_44:                               ;   in Loop: Header=BB3_45 Depth=2
	v_add_nc_u32_e32 v4, 0x100, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v154, v2
	s_add_co_i32 s31, s31, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s31, 4
	s_cbranch_scc0 .LBB3_11
.LBB3_45:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB3_50 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s31, 1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB3_48
; %bb.46:                               ;   in Loop: Header=BB3_45 Depth=2
	s_cmp_eq_u32 s31, 1
	s_mov_b32 s0, s33
	s_cbranch_scc1 .LBB3_48
; %bb.47:                               ;   in Loop: Header=BB3_45 Depth=2
	s_cmp_eq_u32 s31, 2
	s_cselect_b32 s0, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, s29
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s26, s0
	s_cselect_b32 s0, -1, 0
.LBB3_48:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_85
; %bb.49:                               ;   in Loop: Header=BB3_45 Depth=2
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v2, v148
	v_mov_b32_e32 v3, v149
	s_movk_i32 s0, 0xc000
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_mov_b32_e32 v144, v137
.LBB3_50:                               ;   Parent Loop BB3_14 Depth=1
                                        ;     Parent Loop BB3_45 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_clause 0x3
	global_load_b64 v[173:174], v[2:3], off offset:-48
	global_load_b64 v[175:176], v[2:3], off offset:-32
	global_load_b64 v[177:178], v[2:3], off offset:-16
	global_load_b64 v[179:180], v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v169, s0, v4
	v_add_co_u32 v2, vcc_lo, v2, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	ds_load_2addr_stride64_b64 v[5:8], v169 offset0:32 offset1:34
	ds_load_2addr_stride64_b64 v[169:172], v169 offset0:36 offset1:38
	s_addk_co_i32 s0, 0x1000
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[5:6], v[173:174], v[137:144]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[7:8], v[175:176], v[137:144]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[169:170], v[177:178], v[137:144]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[171:172], v[179:180], v[137:144]
	s_cbranch_scc1 .LBB3_50
; %bb.51:                               ;   in Loop: Header=BB3_45 Depth=2
	s_lshl4_add_u32 s0, s31, s29
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v170, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 15
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s27
	s_cselect_b32 s34, -1, 0
	s_cmp_gt_i32 s1, s27
	s_cselect_b32 s1, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s1, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s3
	s_cbranch_execz .LBB3_53
; %bb.52:                               ;   in Loop: Header=BB3_45 Depth=2
	global_load_b32 v170, v[146:147], off
.LBB3_53:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, s0, v163
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e32 vcc_lo, s26, v2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB3_55
; %bb.54:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[5:6], null, 0x408, v2, s[22:23]
	global_load_d16_b16 v3, v[5:6], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v173, v3.l
.LBB3_55:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 1, v2
	v_cmp_gt_i32_e64 s0, s26, v2
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB3_57
; %bb.56:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[6:7], null, 0x408, v5, s[22:23]
	global_load_d16_b16 v3, v[6:7], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v175, v3.l
.LBB3_57:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v6, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s1, s26, v6
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB3_59
; %bb.58:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[7:8], null, 0x408, v6, s[22:23]
	global_load_d16_b16 v3, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v3.l
.LBB3_59:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 3, v2
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v178, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e64 s3, s26, v7
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_61
; %bb.60:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[171:172], null, 0x408, v7, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v178, v3.l
.LBB3_61:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v8, 4, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s4, s26, v8
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB3_63
; %bb.62:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[171:172], null, 0x408, v8, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v176, v3.l
.LBB3_63:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v169, 5, v2
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e64 s5, s26, v169
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_65
; %bb.64:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[171:172], null, 0x408, v169, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v179, v3.l
.LBB3_65:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v171, 6, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s6, s26, v171
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB3_67
; %bb.66:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[180:181], null, 0x408, v171, s[22:23]
	global_load_d16_b16 v3, v[180:181], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v177, v3.l
.LBB3_67:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v172, 7, v2
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v180, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_ge_i32_e64 s7, s26, v172
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_69
; %bb.68:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[180:181], null, 0x408, v172, s[22:23]
	global_load_d16_b16 v180, v[180:181], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v180, v180.l
.LBB3_69:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mul_f32 v137, v0, v137 :: v_dual_mul_f32 v138, v0, v138
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s8, v2, v170
	v_cmp_lt_i32_e64 s9, v2, v170
	v_dual_mul_f32 v139, v0, v139 :: v_dual_mul_f32 v140, v0, v140
	v_cmp_le_i32_e64 s10, v6, v170
	v_dual_mul_f32 v137, v137, v173 :: v_dual_mul_f32 v138, v138, v175
	s_or_b32 s8, s34, s8
	s_or_b32 s9, s34, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	v_dual_mul_f32 v173, v139, v174 :: v_dual_mul_f32 v144, v0, v144
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v137, 0xff800000, v137, s8
	s_or_b32 s10, s34, s10
	v_cmp_le_i32_e64 s8, v7, v170
	s_and_b32 s9, s19, s9
	v_mul_f32_e32 v142, v0, v142
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v139, 0xff800000, v138, s9
	s_and_b32 s9, s19, s10
	v_mul_f32_e32 v140, v140, v178
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, 0xff800000, v173, s9
	v_mul_f32_e32 v173, v0, v141
	s_or_b32 s10, s34, s8
	v_cmp_le_i32_e64 s8, v8, v170
	v_cmp_le_i32_e64 s9, v169, v170
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s10, s19, s10
	v_mul_f32_e32 v143, v0, v143
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v141, 0xff800000, v140, s10
	v_mul_f32_e32 v140, v173, v176
	s_or_b32 s8, s34, s8
	v_mul_f32_e32 v173, v142, v179
	s_or_b32 s9, s34, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	v_dual_mul_f32 v143, v143, v177 :: v_dual_mul_f32 v144, v144, v180
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v142, 0xff800000, v140, s8
	s_and_b32 s8, s19, s9
	v_cmp_le_i32_e64 s9, v172, v170
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v140, 0xff800000, v173, s8
	v_cmp_le_i32_e64 s8, v171, v170
	v_max3_num_f32 v170, v137, 0xff800000, v139
	s_or_b32 s9, s34, s9
	s_or_b32 s8, s34, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v170, v170, v138, v141
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v173, 0xff800000, v143, s8
	s_and_b32 s8, s19, s9
	v_max3_num_f32 v143, v170, v142, v140
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v144, 0xff800000, v144, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v175, v143, v173, v144
	ds_bpermute_b32 v176, v158, v175
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB3_71
; %bb.70:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v3, v2.l
.LBB3_71:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v170, 0
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB3_73
; %bb.72:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[177:178], null, 0x408, v5, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v170, v2.l
.LBB3_73:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_75
; %bb.74:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[5:6], null, 0x408, v6, s[12:13]
	global_load_d16_b16 v2, v[5:6], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v143, v2.l
.LBB3_75:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB3_77
; %bb.76:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[6:7], null, 0x408, v7, s[12:13]
	global_load_d16_b16 v2, v[6:7], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v6, v2.l
.LBB3_77:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB3_79
; %bb.78:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[7:8], null, 0x408, v8, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v5, v2.l
.LBB3_79:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v8, 0
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB3_86
; %bb.80:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execnz .LBB3_87
.LBB3_81:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB3_83
.LBB3_82:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[171:172], null, 0x408, v172, s[12:13]
	global_load_d16_b16 v2, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v169, v2.l
.LBB3_83:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_dscnt 0x0
	v_max3_num_f32 v2, v154, v175, v176
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v7, v137, v2 :: v_dual_sub_f32 v138, v138, v2
	v_dual_sub_f32 v137, v139, v2 :: v_dual_sub_f32 v140, v140, v2
	v_dual_sub_f32 v139, v141, v2 :: v_dual_sub_f32 v144, v144, v2
	v_dual_mul_f32 v7, 0x3fb8aa3b, v7 :: v_dual_mul_f32 v138, 0x3fb8aa3b, v138
	v_dual_sub_f32 v141, v142, v2 :: v_dual_sub_f32 v142, v173, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_mul_f32 v140, 0x3fb8aa3b, v140
	v_exp_f32_e32 v7, v7
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v2
	v_exp_f32_e32 v138, v138
	v_dual_mul_f32 v139, 0x3fb8aa3b, v139 :: v_dual_mul_f32 v144, 0x3fb8aa3b, v144
	v_exp_f32_e32 v137, v137
	v_exp_f32_e32 v140, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v139, v139
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, v7, 0, vcc_lo
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_mul_f32_e32 v171, 0x3fb8aa3b, v142
	v_exp_f32_e32 v144, v144
	v_mul_f32_e32 v142, v7, v3
	v_exp_f32_e32 v172, v141
	v_cndmask_b32_e64 v137, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	v_add_f32_e32 v3, v7, v137
	v_mul_f32_e32 v141, v137, v170
	v_cndmask_b32_e64 v170, v139, 0, vcc_lo
	v_exp_f32_e32 v137, v171
	v_cndmask_b32_e64 v171, v172, 0, vcc_lo
	v_cndmask_b32_e64 v172, v140, 0, vcc_lo
	v_mul_f32_e32 v140, v138, v143
	v_cndmask_b32_e64 v143, v144, 0, vcc_lo
	v_mul_f32_e32 v139, v170, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v7, v171, v5 :: v_dual_mul_f32 v8, v172, v8
	v_dual_add_f32 v3, v138, v3 :: v_dual_mul_f32 v138, v143, v169
	v_max3_num_f32 v6, v142, 0, v141
	v_cndmask_b32_e64 v5, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v3, v170, v3
	v_max3_num_f32 v6, v6, v140, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v137, v5, v174
	v_add_f32_e32 v3, v171, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v6, v6, v7, v8
	v_add_f32_e32 v3, v172, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v144, v6, v137, v138
	v_add_f32_e32 v3, v5, v3
	ds_bpermute_b32 v169, v158, v144
	v_add_f32_e32 v5, v143, v3
	ds_bpermute_b32 v6, v158, v5
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v3, v169, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v143, v144, v3
	v_mov_b32_e32 v3, v167
	v_cmpx_lt_f32_e32 0, v143
	s_cbranch_execz .LBB3_43
; %bb.84:                               ;   in Loop: Header=BB3_45 Depth=2
	v_div_scale_f32 v3, null, 0x43e00000, 0x43e00000, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v144, v3
	v_fma_f32 v169, -v3, v144, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, vcc_lo, v143, 0x43e00000, v143
	v_mul_f32_e32 v170, v169, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v171, -v3, v170, v169
	v_fmac_f32_e32 v170, v171, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v3, v170, v169
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v144, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v3, 0x43e00000, v143
	v_max_num_f32_e32 v3, 0x1f800000, v3
	s_branch .LBB3_43
.LBB3_85:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mov_b32_e32 v2, v154
	s_branch .LBB3_44
.LBB3_86:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[7:8], null, 0x408, v169, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v8, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execz .LBB3_81
.LBB3_87:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[177:178], null, 0x408, v171, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execnz .LBB3_82
	s_branch .LBB3_83
.LBB3_88:
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB3_93
; %bb.89:
	v_cmp_eq_u32_e32 vcc_lo, 0, v159
	s_and_b32 s1, vcc_lo, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_91
; %bb.90:
	v_mad_co_u64_u32 v[0:1], null, v145, s17, s[18:19]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v0, 0x102, v0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	global_store_b64 v[0:1], v[154:155], off
.LBB3_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s19
	s_cbranch_execz .LBB3_93
; %bb.92:
	v_mad_co_u64_u32 v[1:2], null, v145, s17, s[18:19]
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v5, 2, v163
	v_dual_mul_f32 v0, v167, v129 :: v_dual_mul_f32 v105, v105, v167
	v_mul_f32_e32 v4, v121, v167
	v_mul_f32_e32 v8, v113, v167
	v_mul_f32_e32 v81, v81, v167
	v_mul_lo_u32 v1, 0x102, v1
	v_dual_mul_f32 v7, v124, v167 :: v_dual_mul_f32 v66, v66, v167
	v_dual_mul_f32 v57, v57, v167 :: v_dual_mul_f32 v106, v106, v167
	v_dual_mul_f32 v113, v9, v167 :: v_dual_mul_f32 v34, v34, v167
	v_mul_f32_e32 v9, v114, v167
	v_lshlrev_b64_e32 v[2:3], 2, v[1:2]
	v_dual_mul_f32 v1, v167, v130 :: v_dual_mul_f32 v26, v26, v167
	v_dual_mul_f32 v107, v107, v167 :: v_dual_mul_f32 v114, v10, v167
	v_mul_f32_e32 v91, v91, v167
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, vcc_lo, s14, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s15, v3, vcc_lo
	v_dual_mul_f32 v2, v167, v131 :: v_dual_mul_f32 v129, v167, v133
	v_add_co_u32 v137, vcc_lo, v6, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v138, null, 0, v3, vcc_lo
	v_mul_f32_e32 v3, v167, v132
	v_dual_mul_f32 v5, v122, v167 :: v_dual_mul_f32 v10, v115, v167
	v_mul_f32_e32 v6, v123, v167
	v_dual_mul_f32 v83, v83, v167 :: v_dual_mul_f32 v108, v108, v167
	v_mul_f32_e32 v75, v75, v167
	v_dual_mul_f32 v115, v11, v167 :: v_dual_mul_f32 v44, v44, v167
	s_clause 0x1
	global_store_b128 v[137:138], v[0:3], off offset:8
	global_store_b128 v[137:138], v[4:7], off offset:72
	v_dual_mul_f32 v11, v116, v167 :: v_dual_mul_f32 v2, v119, v167
	v_dual_mul_f32 v1, v118, v167 :: v_dual_mul_f32 v6, v111, v167
	v_dual_mul_f32 v4, v109, v167 :: v_dual_mul_f32 v5, v110, v167
	v_mul_f32_e32 v36, v36, v167
	v_dual_mul_f32 v7, v112, v167 :: v_dual_mul_f32 v0, v117, v167
	v_mul_f32_e32 v3, v120, v167
	v_dual_mul_f32 v97, v97, v167 :: v_dual_mul_f32 v130, v167, v134
	v_dual_mul_f32 v89, v89, v167 :: v_dual_mul_f32 v132, v167, v136
	v_dual_mul_f32 v73, v73, v167 :: v_dual_mul_f32 v122, v126, v167
	v_dual_mul_f32 v65, v65, v167 :: v_dual_mul_f32 v124, v128, v167
	v_dual_mul_f32 v49, v49, v167 :: v_dual_mul_f32 v98, v98, v167
	v_dual_mul_f32 v41, v41, v167 :: v_dual_mul_f32 v90, v90, v167
	v_dual_mul_f32 v33, v33, v167 :: v_dual_mul_f32 v82, v82, v167
	v_mul_f32_e32 v131, v167, v135
	v_dual_mul_f32 v25, v25, v167 :: v_dual_mul_f32 v74, v74, v167
	v_dual_mul_f32 v121, v125, v167 :: v_dual_mul_f32 v58, v58, v167
	v_dual_mul_f32 v123, v127, v167 :: v_dual_mul_f32 v50, v50, v167
	v_dual_mul_f32 v17, v17, v167 :: v_dual_mul_f32 v42, v42, v167
	v_dual_mul_f32 v18, v18, v167 :: v_dual_mul_f32 v99, v99, v167
	v_dual_mul_f32 v67, v67, v167 :: v_dual_mul_f32 v100, v100, v167
	v_dual_mul_f32 v59, v59, v167 :: v_dual_mul_f32 v92, v92, v167
	v_dual_mul_f32 v51, v51, v167 :: v_dual_mul_f32 v84, v84, v167
	v_dual_mul_f32 v43, v43, v167 :: v_dual_mul_f32 v76, v76, v167
	v_dual_mul_f32 v35, v35, v167 :: v_dual_mul_f32 v68, v68, v167
	v_dual_mul_f32 v27, v27, v167 :: v_dual_mul_f32 v60, v60, v167
	v_dual_mul_f32 v19, v19, v167 :: v_dual_mul_f32 v52, v52, v167
	v_dual_mul_f32 v28, v28, v167 :: v_dual_mul_f32 v85, v85, v167
	global_store_b128 v[137:138], v[8:11], off offset:136
	v_dual_mul_f32 v20, v20, v167 :: v_dual_mul_f32 v77, v77, v167
	v_dual_mul_f32 v116, v12, v167 :: v_dual_mul_f32 v69, v69, v167
	v_mul_f32_e32 v8, v101, v167
	v_mul_f32_e32 v12, v93, v167
	s_clause 0x1
	global_store_b128 v[137:138], v[105:108], off offset:200
	global_store_b128 v[137:138], v[4:7], off offset:216
	v_dual_mul_f32 v61, v61, v167 :: v_dual_mul_f32 v10, v103, v167
	v_mul_f32_e32 v9, v102, v167
	v_dual_mul_f32 v11, v104, v167 :: v_dual_mul_f32 v4, v13, v167
	v_dual_mul_f32 v53, v53, v167 :: v_dual_mul_f32 v86, v86, v167
	v_dual_mul_f32 v45, v45, v167 :: v_dual_mul_f32 v78, v78, v167
	v_dual_mul_f32 v37, v37, v167 :: v_dual_mul_f32 v70, v70, v167
	v_dual_mul_f32 v29, v29, v167 :: v_dual_mul_f32 v62, v62, v167
	v_mul_f32_e32 v13, v94, v167
	v_dual_mul_f32 v54, v54, v167 :: v_dual_mul_f32 v87, v87, v167
	v_dual_mul_f32 v46, v46, v167 :: v_dual_mul_f32 v79, v79, v167
	v_dual_mul_f32 v38, v38, v167 :: v_dual_mul_f32 v71, v71, v167
	v_dual_mul_f32 v30, v30, v167 :: v_dual_mul_f32 v63, v63, v167
	v_dual_mul_f32 v5, v14, v167 :: v_dual_mul_f32 v6, v15, v167
	v_mul_f32_e32 v14, v95, v167
	v_dual_mul_f32 v55, v55, v167 :: v_dual_mul_f32 v88, v88, v167
	v_dual_mul_f32 v47, v47, v167 :: v_dual_mul_f32 v80, v80, v167
	v_dual_mul_f32 v39, v39, v167 :: v_dual_mul_f32 v72, v72, v167
	v_dual_mul_f32 v31, v31, v167 :: v_dual_mul_f32 v64, v64, v167
	v_mul_f32_e32 v15, v96, v167
	v_mul_f32_e32 v56, v56, v167
	v_mul_f32_e32 v48, v48, v167
	v_mul_f32_e32 v40, v40, v167
	v_mul_f32_e32 v32, v32, v167
	global_store_b128 v[137:138], v[0:3], off offset:152
	v_dual_mul_f32 v0, v21, v167 :: v_dual_mul_f32 v1, v22, v167
	v_dual_mul_f32 v2, v23, v167 :: v_dual_mul_f32 v3, v24, v167
	s_clause 0x13
	global_store_b128 v[137:138], v[129:132], off offset:24
	global_store_b128 v[137:138], v[121:124], off offset:88
	global_store_b128 v[137:138], v[97:100], off offset:264
	global_store_b128 v[137:138], v[8:11], off offset:280
	global_store_b128 v[137:138], v[89:92], off offset:328
	global_store_b128 v[137:138], v[12:15], off offset:344
	global_store_b128 v[137:138], v[81:84], off offset:392
	global_store_b128 v[137:138], v[85:88], off offset:408
	global_store_b128 v[137:138], v[73:76], off offset:456
	global_store_b128 v[137:138], v[77:80], off offset:472
	global_store_b128 v[137:138], v[65:68], off offset:520
	global_store_b128 v[137:138], v[69:72], off offset:536
	global_store_b128 v[137:138], v[57:60], off offset:584
	global_store_b128 v[137:138], v[61:64], off offset:600
	global_store_b128 v[137:138], v[49:52], off offset:648
	global_store_b128 v[137:138], v[53:56], off offset:664
	global_store_b128 v[137:138], v[41:44], off offset:712
	global_store_b128 v[137:138], v[45:48], off offset:728
	global_store_b128 v[137:138], v[33:36], off offset:776
	global_store_b128 v[137:138], v[37:40], off offset:792
	v_mul_f32_e32 v7, v16, v167
	s_clause 0x5
	global_store_b128 v[137:138], v[25:28], off offset:840
	global_store_b128 v[137:138], v[29:32], off offset:856
	global_store_b128 v[137:138], v[17:20], off offset:904
	global_store_b128 v[137:138], v[0:3], off offset:920
	global_store_b128 v[137:138], v[113:116], off offset:968
	global_store_b128 v[137:138], v[4:7], off offset:984
.LBB3_93:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end3:
	.size	attention_fp8_e4m3_fa2_gqa_partial_gfx1201, .Lfunc_end3-attention_fp8_e4m3_fa2_gqa_partial_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_partial_gfx1201
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
		.amdhsa_next_free_vgpr 191
		.amdhsa_next_free_sgpr 35
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-attention_fp8_e4m3_fa2_gqa_partial_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.num_vgpr, 191
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.numbered_sgpr, 35
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8148
; TotalNumSgprs: 37
; NumVgprs: 191
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 37
; NumVGPRsForWavesPerEU: 191
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_merge_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_merge_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_merge_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_merge_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_merge_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_merge_gfx1201
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
	s_cbranch_vccnz .LBB4_10
; %bb.1:
	v_lshrrev_b32_e32 v1, 5, v0
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v4, ttmp9, 3, v1
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB4_10
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
.LBB4_3:                                ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_scc0 .LBB4_3
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
	s_branch .LBB4_6
.LBB4_5:                                ;   in Loop: Header=BB4_6 Depth=1
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
	s_cbranch_execz .LBB4_10
.LBB4_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_8 Depth 2
	v_dual_mov_b32 v6, v2 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	s_mov_b32 s0, s7
	s_branch .LBB4_8
.LBB4_7:                                ;   in Loop: Header=BB4_8 Depth=2
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
	s_cbranch_scc1 .LBB4_5
.LBB4_8:                                ;   Parent Loop BB4_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	s_mov_b32 s2, exec_lo
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	s_cbranch_execz .LBB4_7
; %bb.9:                                ;   in Loop: Header=BB4_8 Depth=2
	global_load_b32 v14, v[5:6], off
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	v_exp_f32_e32 v14, v14
	s_branch .LBB4_7
.LBB4_10:
	s_endpgm
.Lfunc_end4:
	.size	attention_fp8_e4m3_fa2_gqa_merge_gfx1201, .Lfunc_end4-attention_fp8_e4m3_fa2_gqa_merge_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_merge_gfx1201
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-attention_fp8_e4m3_fa2_gqa_merge_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.num_vgpr, 17
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.numbered_sgpr, 8
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.has_indirect_call, 0
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
	.protected	attention_fp8_e4m3_fa2_gqa_packet_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_packet_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_packet_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_packet_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_packet_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_packet_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[12:15], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s12, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s13, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s14, 0x100
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB5_60
; %bb.1:
	s_cmp_gt_i32 ttmp7, 3
	s_cbranch_scc1 .LBB5_60
; %bb.2:
	s_load_b32 s22, s[0:1], 0x38
	s_lshl_b32 s14, ttmp9, 7
	s_mul_i32 s19, s15, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s14, s19
	s_cbranch_scc1 .LBB5_60
; %bb.3:
	s_clause 0x1
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x20
	v_and_b32_e32 v5, 31, v0
	v_mov_b32_e32 v3, -1
	v_bfrev_b32_e32 v4, -2
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_u32_e32 22, v5
	s_cbranch_execz .LBB5_7
; %bb.4:
	s_mul_hi_i32 s1, s14, 0x2aaaaaab
	v_bfrev_b32_e32 v4, -2
	s_lshr_b32 s2, s1, 31
	v_mov_b32_e32 v3, -1
	v_add3_u32 v1, s1, s2, v5
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v1
	s_cbranch_execz .LBB5_6
; %bb.5:
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	global_load_b32 v3, v[1:2], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v4, v3
.LBB5_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
.LBB5_7:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	s_or_b32 exec_lo, exec_lo, s0
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_dual_mov_b32 v16, 0x6020400 :: v_dual_lshlrev_b32 v11, 3, v5
	v_lshrrev_b32_e32 v6, 5, v0
	v_lshrrev_b32_e32 v9, 4, v5
	v_xor_b32_e32 v2, 16, v1
	v_xor_b32_e32 v19, 8, v1
	v_xor_b32_e32 v25, 4, v1
	v_bfe_u32 v14, v0, 2, 2
	v_lshl_add_u32 v156, v5, 4, 0
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	v_xor_b32_e32 v27, 1, v1
	v_dual_mov_b32 v17, 0x5040100 :: v_dual_lshlrev_b32 v20, 4, v0
	v_or_b32_e32 v21, 0x100, v0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v1, v2 :: v_dual_and_b32 v8, 15, v0
	v_mov_b32_e32 v2, 0
	v_cmp_gt_u32_e32 vcc_lo, 32, v19
	v_and_b32_e32 v10, 0x7f, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v7, 2, v7
	v_and_b32_e32 v15, 1, v0
	v_and_b32_e32 v13, 3, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, v1, v19, vcc_lo
	v_and_b32_e32 v5, 16, v0
	ds_bpermute_b32 v12, v7, v3
	ds_bpermute_b32 v7, v7, v4
	v_cmp_eq_u32_e32 vcc_lo, 0, v15
	v_lshlrev_b32_e32 v19, 2, v19
	v_add_nc_u32_e32 v162, 0, v5
	v_xor_b32_e32 v26, 2, v1
	v_bfe_u32 v18, v0, 5, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v160, 0x3070105, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v25
	v_or_b32_e32 v22, 0x200, v0
	v_or_b32_e32 v23, 0x300, v0
	s_or_b32 s1, s14, 0x7f
	v_lshl_add_u32 v24, v6, 11, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s1, s19
	v_dual_mov_b32 v214, 0xff800000 :: v_dual_lshlrev_b32 v159, 3, v6
	v_cmp_eq_u32_e64 s1, v9, v18
	v_lshrrev_b32_e32 v18, 5, v22
	s_mov_b32 s13, 0
	s_mul_i32 s12, s15, 0x1800
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v12
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v4, v7
	s_mov_b32 s2, ttmp7
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[20:21], s[4:5], s[12:13]
	s_cselect_b32 s12, -1, 0
	ds_bpermute_b32 v7, v19, v3
	ds_bpermute_b32 v12, v19, v4
	v_dual_mov_b32 v212, 1.0 :: v_dual_lshlrev_b32 v19, 4, v6
	s_ashr_i32 s3, ttmp7, 31
	v_lshlrev_b32_e32 v158, 3, v9
	v_and_or_b32 v18, v18, 20, v9
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v16, v19, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, v1, v25, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 2, v13
	v_and_or_b32 v164, v19, 48, v8
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[2:3], s[2:3], 8
	v_xad_u32 v165, 0x120, v11, v24
	v_xad_u32 v166, 0x124, v11, v24
	v_xad_u32 v167, 0x240, v11, v24
	v_xad_u32 v168, 0x244, v11, v24
	v_xad_u32 v169, 0x360, v11, v24
	v_xad_u32 v171, 0x364, v11, v24
	v_xad_u32 v172, 0x520, v11, v24
	v_xad_u32 v173, 0x524, v11, v24
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v7
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v4, v12
	v_add_nc_u32_e32 v12, s14, v16
	v_lshlrev_b32_e32 v15, 2, v15
	v_lshl_or_b32 v7, v13, 6, v14
	v_and_or_b32 v16, 0x180, v21, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v161, 0x3020706, v17, vcc_lo
	v_mul_hi_i32 v17, 0x2aaaaaab, v12
	ds_bpermute_b32 v5, v15, v3
	ds_bpermute_b32 v14, v15, v4
	v_cmp_gt_u32_e32 vcc_lo, 32, v26
	v_lshlrev_b32_e32 v28, 3, v13
	v_and_or_b32 v15, v6, 4, v9
	v_lshrrev_b32_e32 v6, 5, v21
	v_and_or_b32 v21, 0x280, v22, v10
	v_lshrrev_b32_e32 v22, 5, v23
	v_and_or_b32 v10, 0x380, v23, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v23, v1, v26, vcc_lo
	v_lshrrev_b32_e32 v25, 31, v17
	v_cmp_gt_u32_e32 vcc_lo, 32, v27
	v_and_or_b32 v19, v6, 12, v9
	v_and_or_b32 v22, v22, 28, v9
	v_lshlrev_b32_e32 v163, 2, v23
	v_add_nc_u32_e32 v146, v17, v25
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[14:15], s[6:7], s[2:3]
	s_add_nc_u64 s[2:3], s[8:9], s[2:3]
	v_xad_u32 v174, 0x640, v11, v24
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v150, s2, s2, v11
	s_wait_dscnt 0x1
	v_max_i32_e32 v5, v3, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v14, v4, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v1, v27, vcc_lo
	v_mul_lo_u32 v1, v146, 6
	v_ashrrev_i32_e32 v147, 31, v146
	ds_bpermute_b32 v8, v163, v5
	ds_bpermute_b32 v17, v163, v14
	v_lshlrev_b32_e32 v170, 2, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v151, null, s3, 0, s2
	v_cmp_le_i32_e64 s2, s19, v12
	v_sub_nc_u32_e32 v1, v12, v1
	v_cmp_gt_i32_e64 s3, s19, v12
	v_xad_u32 v175, 0x764, v11, v24
	v_lshlrev_b32_e32 v16, 4, v16
	v_lshlrev_b32_e32 v10, 4, v10
	v_mad_co_u64_u32 v[148:149], null, ttmp7, 6, v[1:2]
	v_add_nc_u32_e32 v149, v24, v11
	s_lshl_b32 s18, ttmp7, 1
	v_cmp_gt_u32_e64 s0, 64, v0
	v_lshl_add_u32 v157, v0, 1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s19, s18, 31
	v_mov_b32_e32 v213, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[8:9], s[18:19]
	v_mad_co_u64_u32 v[3:4], null, v146, 24, v[148:149]
	s_wait_dscnt 0x1
	v_max_i32_e32 v6, v5, v8
	s_wait_dscnt 0x0
	v_min_i32_e32 v8, v14, v17
	v_lshlrev_b64_e32 v[4:5], 2, v[146:147]
	v_xad_u32 v147, 0x760, v11, v24
	ds_bpermute_b32 v9, v170, v6
	ds_bpermute_b32 v14, v170, v8
	v_lshlrev_b32_e32 v1, 8, v3
	v_add_co_u32 v152, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v12, v1, 0, s2
	v_cndmask_b32_e64 v1, v3, 0, s2
	s_add_nc_u64 s[16:17], s[6:7], s[18:19]
	s_mov_b32 s18, 0x76543210
	v_add_co_u32 v5, s4, s4, v12
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b64_e32 v[3:4], 2, v[1:2]
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s5, 0, s4
	s_wait_dscnt 0x1
	v_max_i32_e32 v1, v6, v9
	s_wait_dscnt 0x0
	v_min_i32_e32 v6, v8, v14
	v_add_co_u32 v154, vcc_lo, s20, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v155, null, s21, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s21, v6
	v_or_b32_e32 v6, 12, v7
	v_or_b32_e32 v3, 8, v7
	v_or_b32_e32 v4, 0x108, v7
	v_lshlrev_b32_e32 v8, 6, v0
	v_or_b32_e32 v9, 0x10c, v7
	v_xor_b32_e32 v6, v6, v28
	v_xor_b32_e32 v3, v3, v28
	v_readfirstlane_b32 s20, v1
	v_xad_u32 v1, 0x644, v11, v24
	v_xor_b32_e32 v4, v4, v28
	v_lshl_add_u32 v179, v6, 2, v24
	v_or_b32_e32 v6, 20, v7
	v_lshl_add_u32 v177, v3, 2, v24
	v_or_b32_e32 v3, 16, v7
	v_lshl_add_u32 v178, v4, 2, v24
	v_or_b32_e32 v4, 0x110, v7
	v_xor_b32_e32 v6, v6, v28
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, v3, v28
	v_xor_b32_e32 v4, v4, v28
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v183, v6, 2, v24
	v_or_b32_e32 v6, 0x11c, v7
	v_lshl_add_u32 v181, v3, 2, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v182, v4, 2, v24
	v_or_b32_e32 v4, 0x118, v7
	v_xor_b32_e32 v6, v6, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v4, v4, v28
	v_lshl_add_u32 v188, v6, 2, v24
	v_or_b32_e32 v6, 48, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v186, v4, 2, v24
	v_xor_b32_e32 v6, v6, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v193, v6, 2, v24
	v_or_b32_e32 v6, 0x138, v7
	v_xor_b32_e32 v6, v6, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v198, v6, 2, v24
	v_dual_mov_b32 v6, v2 :: v_dual_and_b32 v11, 0x3000, v8
	v_xor_b32_e32 v8, v9, v28
	v_or_b32_e32 v9, 24, v7
	v_lshl_add_u32 v180, v8, 2, v24
	v_or_b32_e32 v8, 0x114, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, v9, v28
	v_or_b32_e32 v9, 0x128, v7
	v_xor_b32_e32 v8, v8, v28
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v185, v3, 2, v24
	v_or_b32_e32 v3, 28, v7
	v_xor_b32_e32 v4, v9, v28
	v_or_b32_e32 v9, 52, v7
	v_lshl_add_u32 v184, v8, 2, v24
	v_or_b32_e32 v8, 40, v7
	v_xor_b32_e32 v3, v3, v28
	v_lshl_add_u32 v190, v4, 2, v24
	v_or_b32_e32 v4, 0x12c, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v8, v8, v28
	v_lshl_add_u32 v187, v3, 2, v24
	v_or_b32_e32 v3, 44, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v4, v4, v28
	v_lshl_add_u32 v189, v8, 2, v24
	v_or_b32_e32 v8, 0x130, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v3, v3, v28
	v_lshl_add_u32 v192, v4, 2, v24
	v_or_b32_e32 v4, 0x134, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v8, v8, v28
	v_lshl_add_u32 v191, v3, 2, v24
	v_xor_b32_e32 v3, v9, v28
	s_delay_alu instid0(VALU_DEP_4)
	v_xor_b32_e32 v4, v4, v28
	v_mov_b32_e32 v9, v2
	v_lshl_add_u32 v194, v8, 2, v24
	v_or_b32_e32 v8, 60, v7
	v_lshl_add_u32 v195, v3, 2, v24
	v_or_b32_e32 v3, 56, v7
	v_lshl_add_u32 v196, v4, 2, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v8, v8, v28
	v_xor_b32_e32 v3, v3, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v199, v8, 2, v24
	v_mov_b32_e32 v8, v2
	v_lshl_add_u32 v197, v3, 2, v24
	v_mov_b32_e32 v3, v2
	v_add_co_u32 v201, vcc_lo, v5, v158
	v_mov_b32_e32 v5, v2
	v_lshlrev_b32_e32 v17, 2, v7
	v_or_b32_e32 v7, 0x13c, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v202, null, 0, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v4, v7, v28
	v_mov_b32_e32 v7, v2
	v_lshl_add_u32 v200, v4, 2, v24
	v_dual_mov_b32 v4, v2 :: v_dual_lshlrev_b32 v13, 5, v13
	v_lshlrev_b32_e32 v21, 4, v21
	v_lshlrev_b32_e32 v203, 3, v15
	v_lshlrev_b32_e32 v204, 3, v19
	v_add_nc_u32_e32 v205, 0, v16
	v_add3_u32 v176, v24, v17, v13
	v_add_nc_u32_e32 v208, 0, v10
	v_add_nc_u32_e32 v209, v156, v11
	v_dual_mov_b32 v17, v9 :: v_dual_lshlrev_b32 v206, 3, v18
	v_mov_b32_e32 v15, v7
	v_lshlrev_b32_e32 v207, 3, v22
	v_dual_mov_b32 v11, v3 :: v_dual_add_nc_u32 v210, 0, v20
	v_dual_mov_b32 v10, v2 :: v_dual_add_nc_u32 v211, 0, v21
	v_mov_b32_e32 v25, v9
	v_mov_b32_e32 v33, v9
	v_mov_b32_e32 v41, v9
	v_mov_b32_e32 v49, v9
	v_mov_b32_e32 v57, v9
	v_mov_b32_e32 v65, v9
	v_mov_b32_e32 v73, v9
	v_mov_b32_e32 v81, v9
	v_mov_b32_e32 v89, v9
	v_mov_b32_e32 v97, v9
	v_mov_b32_e32 v105, v9
	v_mov_b32_e32 v113, v9
	v_mov_b32_e32 v121, v9
	v_mov_b32_e32 v129, v9
	v_dual_mov_b32 v137, v9 :: v_dual_mov_b32 v16, v8
	v_dual_mov_b32 v14, v6 :: v_dual_mov_b32 v13, v5
	v_mov_b32_e32 v12, v4
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	v_mov_b32_e32 v36, v4
	v_dual_mov_b32 v22, v6 :: v_dual_mov_b32 v21, v5
	v_mov_b32_e32 v34, v2
	v_dual_mov_b32 v20, v4 :: v_dual_mov_b32 v19, v3
	v_mov_b32_e32 v48, v8
	v_mov_b32_e32 v18, v2
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	v_mov_b32_e32 v44, v4
	v_dual_mov_b32 v30, v6 :: v_dual_mov_b32 v29, v5
	v_mov_b32_e32 v42, v2
	v_dual_mov_b32 v28, v4 :: v_dual_mov_b32 v27, v3
	v_mov_b32_e32 v56, v8
	v_mov_b32_e32 v26, v2
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_mov_b32_e32 v52, v4
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v50, v2 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v47, v7
	v_mov_b32_e32 v60, v4
	v_dual_mov_b32 v46, v6 :: v_dual_mov_b32 v45, v5
	v_dual_mov_b32 v58, v2 :: v_dual_mov_b32 v43, v3
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v55, v7
	v_mov_b32_e32 v68, v4
	v_dual_mov_b32 v54, v6 :: v_dual_mov_b32 v53, v5
	v_dual_mov_b32 v66, v2 :: v_dual_mov_b32 v51, v3
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v63, v7
	v_mov_b32_e32 v76, v4
	v_dual_mov_b32 v62, v6 :: v_dual_mov_b32 v61, v5
	v_dual_mov_b32 v74, v2 :: v_dual_mov_b32 v59, v3
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v71, v7
	v_mov_b32_e32 v84, v4
	v_dual_mov_b32 v70, v6 :: v_dual_mov_b32 v69, v5
	v_dual_mov_b32 v82, v2 :: v_dual_mov_b32 v67, v3
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v79, v7
	v_mov_b32_e32 v92, v4
	v_dual_mov_b32 v78, v6 :: v_dual_mov_b32 v77, v5
	v_dual_mov_b32 v90, v2 :: v_dual_mov_b32 v75, v3
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v87, v7
	v_mov_b32_e32 v100, v4
	v_dual_mov_b32 v86, v6 :: v_dual_mov_b32 v85, v5
	v_dual_mov_b32 v98, v2 :: v_dual_mov_b32 v83, v3
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v95, v7
	v_mov_b32_e32 v108, v4
	v_dual_mov_b32 v94, v6 :: v_dual_mov_b32 v93, v5
	v_dual_mov_b32 v106, v2 :: v_dual_mov_b32 v91, v3
	v_dual_mov_b32 v120, v8 :: v_dual_mov_b32 v103, v7
	v_mov_b32_e32 v116, v4
	v_dual_mov_b32 v102, v6 :: v_dual_mov_b32 v101, v5
	v_dual_mov_b32 v114, v2 :: v_dual_mov_b32 v99, v3
	v_dual_mov_b32 v128, v8 :: v_dual_mov_b32 v111, v7
	v_mov_b32_e32 v124, v4
	v_dual_mov_b32 v110, v6 :: v_dual_mov_b32 v109, v5
	v_dual_mov_b32 v122, v2 :: v_dual_mov_b32 v107, v3
	v_dual_mov_b32 v136, v8 :: v_dual_mov_b32 v119, v7
	v_mov_b32_e32 v132, v4
	v_dual_mov_b32 v118, v6 :: v_dual_mov_b32 v117, v5
	v_dual_mov_b32 v130, v2 :: v_dual_mov_b32 v115, v3
	v_dual_mov_b32 v127, v7 :: v_dual_mov_b32 v126, v6
	v_mov_b32_e32 v125, v5
	v_mov_b32_e32 v123, v3
	v_dual_mov_b32 v135, v7 :: v_dual_mov_b32 v134, v6
	v_mov_b32_e32 v133, v5
	v_mov_b32_e32 v131, v3
	s_branch .LBB5_10
.LBB5_8:                                ;   in Loop: Header=BB5_10 Depth=1
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v214, v3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB5_9:                                ;   in Loop: Header=BB5_10 Depth=1
	s_add_co_i32 s13, s13, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 s13, 0x3fffffff
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s19, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_58
.LBB5_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_47 Depth 2
                                        ;       Child Loop BB5_52 Depth 3
	s_lshl_b32 s23, s13, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s23, s20
	s_cselect_b32 s19, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s19
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_9
; %bb.11:                               ;   in Loop: Header=BB5_10 Depth=1
	v_or_b32_e32 v3, s23, v164
                                        ; implicit-def: $vgpr6
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[138:139], null, 0x408, v3, s[14:15]
	v_cmp_ge_i32_e32 vcc_lo, s20, v3
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s4
	s_cbranch_execz .LBB5_13
; %bb.12:                               ;   in Loop: Header=BB5_10 Depth=1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_u32 v3, s4, v138, v203
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, s4
	v_add_co_u32 v8, s4, v138, v204
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, 0, v139, s4
	s_clause 0x3
	global_load_b64 v[140:141], v[3:4], off
	global_load_b64 v[142:143], v[3:4], off offset:16
	global_load_b64 v[6:7], v[8:9], off
	global_load_b64 v[8:9], v[8:9], off offset:16
	s_wait_loadcnt 0x2
	ds_store_b128 v210, v[140:143]
.LBB5_13:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s5
	s_cbranch_execz .LBB5_15
; %bb.14:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v4, v2
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v6, v9
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	ds_store_b128 v210, v[2:5]
.LBB5_15:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	ds_store_b128 v205, v[6:9]
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB5_17
; %bb.16:                               ;   in Loop: Header=BB5_10 Depth=1
	v_add_co_u32 v3, vcc_lo, v138, v206
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, vcc_lo
	v_add_co_u32 v8, vcc_lo, v138, v207
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, 0, v139, vcc_lo
	s_clause 0x3
	global_load_b64 v[138:139], v[3:4], off
	global_load_b64 v[140:141], v[3:4], off offset:16
	global_load_b64 v[6:7], v[8:9], off
	global_load_b64 v[8:9], v[8:9], off offset:16
	s_wait_loadcnt 0x2
	ds_store_b128 v211, v[138:141]
.LBB5_17:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB5_19
; %bb.18:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v4, v2
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v6, v9
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	ds_store_b128 v211, v[2:5]
.LBB5_19:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v139, 0 :: v_dual_add_nc_u32 v140, s23, v159
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v3, 0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v208, v[6:9]
	v_cmpx_ge_i32_e64 s20, v140
	s_cbranch_execz .LBB5_21
; %bb.20:                               ;   in Loop: Header=BB5_10 Depth=1
	v_mad_co_u64_u32 v[3:4], null, 0x408, v140, v[150:151]
	global_load_b64 v[3:4], v[3:4], off
.LBB5_21:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v5, 0x8000, v149
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v5, v3, v4 offset1:1
	v_cmpx_gt_i32_e64 s20, v140
	s_cbranch_execz .LBB5_23
; %bb.22:                               ;   in Loop: Header=BB5_10 Depth=1
	v_or_b32_e32 v3, 1, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[3:4], null, 0x408, v3, v[150:151]
	global_load_b64 v[138:139], v[3:4], off
.LBB5_23:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v7, 2, v140
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v165, v138 offset:32768
	ds_store_b32 v166, v139 offset:32768
	v_cmpx_ge_i32_e64 s20, v7
	s_cbranch_execz .LBB5_25
; %bb.24:                               ;   in Loop: Header=BB5_10 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v7, v[150:151]
	global_load_b64 v[5:6], v[5:6], off
.LBB5_25:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v7, 3, v140
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v167, v5 offset:32768
	ds_store_b32 v168, v6 offset:32768
	v_cmpx_ge_i32_e64 s20, v7
	s_cbranch_execz .LBB5_27
; %bb.26:                               ;   in Loop: Header=BB5_10 Depth=1
	v_mad_co_u64_u32 v[3:4], null, 0x408, v7, v[150:151]
	global_load_b64 v[3:4], v[3:4], off
.LBB5_27:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v9, 4, v140
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v169, v3 offset:32768
	ds_store_b32 v171, v4 offset:32768
	v_cmpx_ge_i32_e64 s20, v9
	s_cbranch_execz .LBB5_29
; %bb.28:                               ;   in Loop: Header=BB5_10 Depth=1
	v_mad_co_u64_u32 v[3:4], null, 0x408, v9, v[150:151]
	global_load_b64 v[7:8], v[3:4], off
.LBB5_29:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v3, 5, v140
	v_add_nc_u32_e32 v4, 0x8400, v149
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v4, v7, v8 offset1:1
	v_cmpx_ge_i32_e64 s20, v3
	s_cbranch_execz .LBB5_31
; %bb.30:                               ;   in Loop: Header=BB5_10 Depth=1
	v_mad_co_u64_u32 v[3:4], null, 0x408, v3, v[150:151]
	global_load_b64 v[5:6], v[3:4], off
.LBB5_31:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v9, 6, v140
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v172, v5 offset:32768
	ds_store_b32 v173, v6 offset:32768
	v_cmpx_ge_i32_e64 s20, v9
	s_cbranch_execz .LBB5_33
; %bb.32:                               ;   in Loop: Header=BB5_10 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v9, v[150:151]
	global_load_b64 v[7:8], v[5:6], off
.LBB5_33:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v5, 7, v140
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v174, v7 offset:32768
	ds_store_b32 v1, v8 offset:32768
	v_cmpx_ge_i32_e64 s20, v5
	s_cbranch_execz .LBB5_35
; %bb.34:                               ;   in Loop: Header=BB5_10 Depth=1
	v_mad_co_u64_u32 v[3:4], null, 0x408, v5, v[150:151]
	global_load_b64 v[3:4], v[3:4], off
.LBB5_35:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	ds_store_b32 v147, v3 offset:32768
	ds_store_b32 v175, v4 offset:32768
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, s1
	s_cbranch_execz .LBB5_37
; %bb.36:                               ;   in Loop: Header=BB5_10 Depth=1
	v_add_nc_u32_e32 v7, 0x8000, v176
	v_add_nc_u32_e32 v8, 0x8400, v176
	ds_load_2addr_b32 v[3:4], v7 offset1:4
	ds_load_2addr_b32 v[5:6], v8 offset1:4
	s_wait_dscnt 0x1
	ds_bpermute_b32 v9, v170, v3
	s_wait_dscnt 0x1
	ds_bpermute_b32 v138, v170, v5
	ds_bpermute_b32 v139, v170, v4
	ds_bpermute_b32 v140, v170, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v5, v138, v5, v160
	s_wait_dscnt 0x1
	v_perm_b32 v9, v139, v4, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v160
	ds_bpermute_b32 v4, v163, v3
	ds_bpermute_b32 v138, v163, v5
	ds_bpermute_b32 v139, v163, v9
	ds_bpermute_b32 v140, v163, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v4, v3, v161
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v5, v161
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v9, v161
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v161
	ds_store_b128 v209, v[3:6] offset:16384
	ds_load_b32 v3, v177 offset:32768
	ds_load_b32 v4, v178 offset:32768
	ds_load_b32 v5, v179 offset:32768
	ds_load_b32 v6, v180 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v170, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v170, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v170, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v170, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v160
	ds_bpermute_b32 v9, v163, v3
	ds_bpermute_b32 v138, v163, v4
	ds_bpermute_b32 v139, v163, v5
	ds_bpermute_b32 v140, v163, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v161
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v161
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v161
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v161
	ds_store_b128 v209, v[3:6] offset:16896
	ds_load_b32 v3, v181 offset:32768
	ds_load_b32 v4, v182 offset:32768
	ds_load_b32 v5, v183 offset:32768
	ds_load_b32 v6, v184 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v170, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v170, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v170, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v170, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v160
	ds_bpermute_b32 v9, v163, v3
	ds_bpermute_b32 v138, v163, v4
	ds_bpermute_b32 v139, v163, v5
	ds_bpermute_b32 v140, v163, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v161
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v161
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v161
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v161
	ds_store_b128 v209, v[3:6] offset:17408
	ds_load_b32 v3, v185 offset:32768
	ds_load_b32 v4, v186 offset:32768
	ds_load_b32 v5, v187 offset:32768
	ds_load_b32 v6, v188 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v170, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v170, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v170, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v170, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v160
	ds_bpermute_b32 v9, v163, v3
	ds_bpermute_b32 v138, v163, v4
	ds_bpermute_b32 v139, v163, v5
	ds_bpermute_b32 v140, v163, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v161
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v161
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v161
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v161
	ds_store_b128 v209, v[3:6] offset:17920
	ds_load_2addr_b32 v[3:4], v7 offset0:32 offset1:36
	ds_load_2addr_b32 v[5:6], v8 offset0:32 offset1:36
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v170, v3
	s_wait_dscnt 0x1
	ds_bpermute_b32 v8, v170, v5
	ds_bpermute_b32 v9, v170, v4
	ds_bpermute_b32 v138, v170, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v5, v8, v5, v160
	s_wait_dscnt 0x1
	v_perm_b32 v7, v9, v4, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v160
	ds_bpermute_b32 v4, v163, v3
	ds_bpermute_b32 v8, v163, v5
	ds_bpermute_b32 v9, v163, v7
	ds_bpermute_b32 v138, v163, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v4, v3, v161
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v5, v161
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v7, v161
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v161
	ds_store_b128 v209, v[3:6] offset:18432
	ds_load_b32 v3, v189 offset:32768
	ds_load_b32 v4, v190 offset:32768
	ds_load_b32 v5, v191 offset:32768
	ds_load_b32 v6, v192 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v170, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v170, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v170, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v170, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v160
	ds_bpermute_b32 v7, v163, v3
	ds_bpermute_b32 v8, v163, v4
	ds_bpermute_b32 v9, v163, v5
	ds_bpermute_b32 v138, v163, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v161
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v161
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v161
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v161
	ds_store_b128 v209, v[3:6] offset:18944
	ds_load_b32 v3, v193 offset:32768
	ds_load_b32 v4, v194 offset:32768
	ds_load_b32 v5, v195 offset:32768
	ds_load_b32 v6, v196 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v170, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v170, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v170, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v170, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v160
	ds_bpermute_b32 v7, v163, v3
	ds_bpermute_b32 v8, v163, v4
	ds_bpermute_b32 v9, v163, v5
	ds_bpermute_b32 v138, v163, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v161
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v161
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v161
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v161
	ds_store_b128 v209, v[3:6] offset:19456
	ds_load_b32 v3, v197 offset:32768
	ds_load_b32 v4, v198 offset:32768
	ds_load_b32 v5, v199 offset:32768
	ds_load_b32 v6, v200 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v170, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v170, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v170, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v170, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v160
	ds_bpermute_b32 v7, v163, v3
	ds_bpermute_b32 v8, v163, v4
	ds_bpermute_b32 v9, v163, v5
	ds_bpermute_b32 v138, v163, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v161
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v161
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v161
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v161
	ds_store_b128 v209, v[3:6] offset:19968
.LBB5_37:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB5_41
; %bb.38:                               ;   in Loop: Header=BB5_10 Depth=1
	v_or_b32_e32 v4, s23, v0
	v_mov_b32_e32 v3, 0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s20, v4
	s_cbranch_execz .LBB5_40
; %bb.39:                               ;   in Loop: Header=BB5_10 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v4, s[16:17]
	v_mad_co_u64_u32 v[7:8], null, 0x408, v4, s[8:9]
	global_load_d16_b16 v3, v[5:6], off offset:1024
	global_load_d16_hi_b16 v3, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v4.h, 8, v3.l
	v_lshrrev_b16 v4.l, 8, v3.h
	v_and_b16 v5.h, 0xff, v3.l
	v_and_b16 v5.l, 0xff, v3.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_lshlrev_b16 v3, 8, v4 op_sel_hi:[0,1]
	v_or_b32_e32 v3, v3, v5
.LBB5_40:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	ds_store_b16_d16_hi v157, v3 offset:49152
	ds_store_b16 v157, v3 offset:49280
.LBB5_41:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v6, s22
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_43
; %bb.42:                               ;   in Loop: Header=BB5_10 Depth=1
	global_load_b32 v3, v[154:155], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v6, s22, v3
.LBB5_43:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_or_b32 s4, s23, 63
	v_mov_b32_e32 v7, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s21
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s24, s12, s4
	s_wait_alu depctr_sa_sdst(0)
	s_nor_b32 s5, s24, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s5
	s_cbranch_execz .LBB5_45
; %bb.44:                               ;   in Loop: Header=BB5_10 Depth=1
	global_load_b32 v7, v[152:153], off
.LBB5_45:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_or_b32 s4, s23, 16
	s_mov_b32 s25, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s20
	s_cselect_b32 s26, -1, 0
	s_branch .LBB5_47
.LBB5_46:                               ;   in Loop: Header=BB5_47 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_sub_f32_e32 v145, v214, v3
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v214
	v_div_scale_f32 v219, null, v4, v4, v142
	v_mov_b16_e64 v237.l, v2.l
	v_div_scale_f32 v221, null, v4, v4, v141
	v_mul_f32_e32 v145, 0x3fb8aa3b, v145
	v_mov_b16_e64 v237.h, 0
	v_rcp_f32_e32 v220, v219
	v_mov_b16_e64 v236.l, v237.l
	v_lshl_add_u32 v232, s25, 12, v156
	v_exp_f32_e32 v145, v145
	v_mov_b16_e64 v236.h, v237.h
	v_add_f32_e32 v5, v5, v8
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v222, -v219, v220, 1.0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v145, 0, v145 :: v_dual_fmac_f32 v220, v222, v220
	v_div_scale_f32 v222, null, v4, v4, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v212, v212, v145
	v_div_scale_f32 v214, null, v4, v4, v212
	v_div_scale_f32 v217, vcc_lo, v212, v4, v212
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v215, v214
	v_fma_f32 v216, -v214, v215, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v215, v216, v215
	v_mul_f32_e32 v216, v217, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v218, -v214, v216, v217
	v_fmac_f32_e32 v216, v218, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v214, -v214, v216, v217
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v214, v214, v215, v216
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v8, v214, v4, v212
	v_fmac_f32_e32 v5, v213, v145
	v_div_scale_f32 v145, null, v4, v4, v144
	v_div_scale_f32 v213, null, v4, v4, v143
	v_dual_mul_f32 v137, v137, v8 :: v_dual_mul_f32 v130, v130, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v212, v145
	v_rcp_f32_e32 v215, v213
	v_dual_mul_f32 v136, v136, v8 :: v_dual_mul_f32 v135, v135, v8
	v_mul_f32_e32 v128, v128, v8
	v_dual_mul_f32 v134, v134, v8 :: v_dual_mul_f32 v133, v133, v8
	v_mul_f32_e32 v126, v126, v8
	v_mul_f32_e32 v132, v132, v8
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v214, -v145, v212, 1.0
	v_fma_f32 v217, -v213, v215, 1.0
	v_dual_mul_f32 v131, v131, v8 :: v_dual_mul_f32 v124, v124, v8
	v_dual_mul_f32 v129, v129, v8 :: v_dual_mul_f32 v122, v122, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v212, v214, v212 :: v_dual_fmac_f32 v215, v217, v215
	v_div_scale_f32 v214, vcc_lo, v144, v4, v144
	v_div_scale_f32 v217, s4, v143, v4, v143
	v_mul_f32_e32 v23, v23, v8
	v_dual_mul_f32 v127, v127, v8 :: v_dual_mul_f32 v120, v120, v8
	v_mul_f32_e32 v216, v214, v212
	v_dual_mul_f32 v125, v125, v8 :: v_dual_mul_f32 v118, v118, v8
	v_dual_mul_f32 v123, v123, v8 :: v_dual_mul_f32 v116, v116, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v218, -v145, v216, v214
	v_dual_mul_f32 v121, v121, v8 :: v_dual_mul_f32 v114, v114, v8
	v_dual_mul_f32 v119, v119, v8 :: v_dual_mul_f32 v112, v112, v8
	v_fmac_f32_e32 v216, v218, v212
	v_mul_f32_e32 v218, v217, v215
	v_dual_mul_f32 v117, v117, v8 :: v_dual_mul_f32 v110, v110, v8
	v_dual_mul_f32 v115, v115, v8 :: v_dual_mul_f32 v108, v108, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v145, -v145, v216, v214
	v_fma_f32 v214, -v213, v218, v217
	v_mul_f32_e32 v21, v21, v8
	v_dual_mul_f32 v113, v113, v8 :: v_dual_mul_f32 v106, v106, v8
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v145, v145, v212, v216
	v_fmac_f32_e32 v218, v214, v215
	v_div_scale_f32 v214, s5, v142, v4, v142
	s_mov_b32 vcc_lo, s4
	v_rcp_f32_e32 v212, v221
	v_div_fixup_f32 v144, v145, v4, v144
	v_fma_f32 v145, -v213, v218, v217
	v_mul_f32_e32 v213, v214, v220
	v_dual_mul_f32 v111, v111, v8 :: v_dual_mul_f32 v104, v104, v8
	v_dual_mul_f32 v109, v109, v8 :: v_dual_mul_f32 v102, v102, v8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v145, v145, v215, v218
	v_fma_f32 v217, -v219, v213, v214
	v_div_scale_f32 v215, null, v4, v4, v140
	s_mov_b32 vcc_lo, s5
	v_fma_f32 v216, -v221, v212, 1.0
	v_div_fixup_f32 v143, v145, v4, v143
	v_fmac_f32_e32 v213, v217, v220
	v_rcp_f32_e32 v145, v215
	v_div_scale_f32 v218, null, v4, v4, v139
	v_dual_mul_f32 v107, v107, v8 :: v_dual_mul_f32 v100, v100, v8
	v_cvt_pk_fp8_f32 v236.l, v144, v143
	v_fma_f32 v143, -v219, v213, v214
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v219, v218
	v_dual_mul_f32 v105, v105, v8 :: v_dual_mul_f32 v98, v98, v8
	v_fma_f32 v214, -v215, v145, 1.0
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v143, v143, v220, v213
	v_rcp_f32_e32 v213, v222
	v_fmac_f32_e32 v212, v216, v212
	v_div_scale_f32 v216, s4, v141, v4, v141
	v_fmac_f32_e32 v145, v214, v145
	s_mov_b32 vcc_lo, s4
	v_div_fixup_f32 v142, v143, v4, v142
	v_div_scale_f32 v143, null, v4, v4, v138
	v_fma_f32 v214, -v218, v219, 1.0
	v_dual_mul_f32 v103, v103, v8 :: v_dual_mul_f32 v96, v96, v8
	v_fma_f32 v223, -v222, v213, 1.0
	v_dual_mul_f32 v101, v101, v8 :: v_dual_mul_f32 v94, v94, v8
	v_dual_mul_f32 v99, v99, v8 :: v_dual_mul_f32 v92, v92, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v213, v223, v213
	v_div_scale_f32 v223, s4, v9, v4, v9
	v_mul_f32_e32 v217, v216, v212
	v_dual_mul_f32 v97, v97, v8 :: v_dual_mul_f32 v90, v90, v8
	v_dual_mul_f32 v95, v95, v8 :: v_dual_mul_f32 v88, v88, v8
	v_fma_f32 v144, -v221, v217, v216
	v_dual_mul_f32 v93, v93, v8 :: v_dual_mul_f32 v86, v86, v8
	v_dual_mul_f32 v91, v91, v8 :: v_dual_mul_f32 v84, v84, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v217, v144, v212
	v_div_scale_f32 v144, s5, v140, v4, v140
	v_mul_f32_e32 v15, v15, v8
	v_dual_mul_f32 v89, v89, v8 :: v_dual_mul_f32 v82, v82, v8
	v_fma_f32 v216, -v221, v217, v216
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v220, v144, v145
	v_rcp_f32_e32 v221, v143
	v_dual_mul_f32 v12, v12, v8 :: v_dual_mul_f32 v87, v87, v8
	v_mul_f32_e32 v80, v80, v8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v212, v216, v212, v217
	v_fma_f32 v216, -v215, v220, v144
	s_mov_b32 vcc_lo, s5
	v_dual_mul_f32 v85, v85, v8 :: v_dual_mul_f32 v78, v78, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v224, -v143, v221, 1.0
	v_fmac_f32_e32 v220, v216, v145
	v_dual_mul_f32 v216, v223, v213 :: v_dual_fmac_f32 v219, v214, v219
	v_div_scale_f32 v214, s6, v139, v4, v139
	v_div_fixup_f32 v141, v212, v4, v141
	v_fmac_f32_e32 v221, v224, v221
	v_div_scale_f32 v224, s7, v138, v4, v138
	v_fma_f32 v144, -v215, v220, v144
	v_mul_f32_e32 v217, v214, v219
	v_mul_f32_e32 v10, v10, v8
	v_cvt_pk_fp8_f32 v236.h, v142, v141
	v_mul_f32_e32 v215, v224, v221
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v145, v220
	v_fma_f32 v212, -v218, v217, v214
	s_mov_b32 vcc_lo, s6
	v_dual_mul_f32 v83, v83, v8 :: v_dual_mul_f32 v76, v76, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v140, v144, v4, v140
	v_fmac_f32_e32 v217, v212, v219
	v_fma_f32 v212, -v222, v216, v223
	v_dual_mul_f32 v81, v81, v8 :: v_dual_mul_f32 v74, v74, v8
	v_dual_mul_f32 v79, v79, v8 :: v_dual_mul_f32 v72, v72, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v145, -v218, v217, v214
	v_fma_f32 v214, -v143, v215, v224
	v_dual_fmac_f32 v216, v212, v213 :: v_dual_mul_f32 v77, v77, v8
	v_mul_f32_e32 v70, v70, v8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v145, v145, v219, v217
	v_fmac_f32_e32 v215, v214, v221
	v_fma_f32 v144, -v222, v216, v223
	s_mov_b32 vcc_lo, s4
	v_dual_mul_f32 v75, v75, v8 :: v_dual_mul_f32 v68, v68, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v143, -v143, v215, v224
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v213, v216
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v139, v145, v4, v139
	v_dual_mul_f32 v73, v73, v8 :: v_dual_mul_f32 v66, v66, v8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v143, v143, v221, v215
	v_div_fixup_f32 v9, v144, v4, v9
	v_cvt_pk_fp8_f32 v237.l, v140, v139
	v_dual_mul_f32 v71, v71, v8 :: v_dual_mul_f32 v64, v64, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v238, v143, v4, v138
	ds_load_b128 v[138:141], v232 offset:16384
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[142:145], v232 offset:16896
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[212:215], v232 offset:17408
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[216:219], v232 offset:17920
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[220:223], v232 offset:18432
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[224:227], v232 offset:18944
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[228:231], v232 offset:19456
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[232:235], v232 offset:19968
	v_dual_mul_f32 v69, v69, v8 :: v_dual_mul_f32 v62, v62, v8
	v_dual_mul_f32 v67, v67, v8 :: v_dual_mul_f32 v60, v60, v8
	v_dual_mul_f32 v65, v65, v8 :: v_dual_mul_f32 v58, v58, v8
	v_dual_mul_f32 v63, v63, v8 :: v_dual_mul_f32 v56, v56, v8
	v_dual_mul_f32 v61, v61, v8 :: v_dual_mul_f32 v54, v54, v8
	v_dual_mul_f32 v59, v59, v8 :: v_dual_mul_f32 v52, v52, v8
	v_dual_mul_f32 v57, v57, v8 :: v_dual_mul_f32 v50, v50, v8
	v_dual_mul_f32 v55, v55, v8 :: v_dual_mul_f32 v48, v48, v8
	v_dual_mul_f32 v53, v53, v8 :: v_dual_mul_f32 v46, v46, v8
	v_dual_mul_f32 v51, v51, v8 :: v_dual_mul_f32 v44, v44, v8
	v_dual_mul_f32 v49, v49, v8 :: v_dual_mul_f32 v42, v42, v8
	v_dual_mul_f32 v47, v47, v8 :: v_dual_mul_f32 v40, v40, v8
	v_dual_mul_f32 v45, v45, v8 :: v_dual_mul_f32 v38, v38, v8
	v_dual_mul_f32 v43, v43, v8 :: v_dual_mul_f32 v36, v36, v8
	v_dual_mul_f32 v41, v41, v8 :: v_dual_mul_f32 v34, v34, v8
	v_dual_mul_f32 v39, v39, v8 :: v_dual_mul_f32 v32, v32, v8
	v_dual_mul_f32 v37, v37, v8 :: v_dual_mul_f32 v30, v30, v8
	v_dual_mul_f32 v35, v35, v8 :: v_dual_mul_f32 v28, v28, v8
	v_dual_mul_f32 v33, v33, v8 :: v_dual_mul_f32 v26, v26, v8
	v_dual_mul_f32 v31, v31, v8 :: v_dual_mul_f32 v24, v24, v8
	v_dual_mul_f32 v29, v29, v8 :: v_dual_mul_f32 v22, v22, v8
	v_dual_mul_f32 v27, v27, v8 :: v_dual_mul_f32 v20, v20, v8
	v_dual_mul_f32 v25, v25, v8 :: v_dual_mul_f32 v18, v18, v8
	v_dual_mul_f32 v19, v19, v8 :: v_dual_mul_f32 v16, v16, v8
	v_dual_mul_f32 v17, v17, v8 :: v_dual_mul_f32 v14, v14, v8
	v_cvt_pk_fp8_f32 v237.h, v9, v238
	v_mul_f32_e32 v13, v13, v8
	v_mul_f32_e32 v11, v11, v8
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[130:137], v[138:139], v[236:237], v[130:137]
	v_wmma_f32_16x16x16_fp8_fp8 v[122:129], v[140:141], v[236:237], v[122:129]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[142:143], v[236:237], v[114:121]
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[144:145], v[236:237], v[106:113]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[212:213], v[236:237], v[98:105]
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[214:215], v[236:237], v[90:97]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[216:217], v[236:237], v[82:89]
	v_wmma_f32_16x16x16_fp8_fp8 v[74:81], v[218:219], v[236:237], v[74:81]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[66:73], v[220:221], v[236:237], v[66:73]
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[222:223], v[236:237], v[58:65]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[224:225], v[236:237], v[50:57]
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[226:227], v[236:237], v[42:49]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[228:229], v[236:237], v[34:41]
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[230:231], v[236:237], v[26:33]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[232:233], v[236:237], v[18:25]
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[234:235], v[236:237], v[10:17]
	v_dual_mov_b32 v213, v5 :: v_dual_mov_b32 v212, v4
	v_mov_b32_e32 v214, v3
	s_add_co_i32 s25, s25, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s25, 4
	s_cbranch_scc1 .LBB5_8
.LBB5_47:                               ;   Parent Loop BB5_10 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_52 Depth 3
	s_cmp_lt_i32 s25, 1
	s_mov_b32 s4, -1
	s_cbranch_scc1 .LBB5_50
; %bb.48:                               ;   in Loop: Header=BB5_47 Depth=2
	s_cmp_eq_u32 s25, 1
	s_mov_b32 s4, s26
	s_cbranch_scc1 .LBB5_50
; %bb.49:                               ;   in Loop: Header=BB5_47 Depth=2
	s_cmp_eq_u32 s25, 2
	s_cselect_b32 s4, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s4, s23
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s20, s4
	s_cselect_b32 s4, -1, 0
.LBB5_50:                               ;   in Loop: Header=BB5_47 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_57
; %bb.51:                               ;   in Loop: Header=BB5_47 Depth=2
	v_mov_b32_e32 v138, 0
	v_lshl_add_u32 v3, s25, 9, v156
	s_mov_b32 s4, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v139, v138 :: v_dual_mov_b32 v140, v138
	v_dual_mov_b32 v141, v138 :: v_dual_mov_b32 v142, v138
	v_dual_mov_b32 v143, v138 :: v_dual_mov_b32 v144, v138
	v_mov_b32_e32 v145, v138
.LBB5_52:                               ;   Parent Loop BB5_10 Depth=1
                                        ;     Parent Loop BB5_47 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s5, s4, 6
	v_lshl_add_u32 v219, s4, 12, v3
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, v201, s5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v202, vcc_lo
	ds_load_b128 v[215:218], v219
	ds_load_b128 v[219:222], v219 offset:2048
	s_add_co_i32 s4, s4, 1
	s_clause 0x3
	global_load_b64 v[8:9], v[4:5], off
	global_load_b64 v[223:224], v[4:5], off offset:16
	global_load_b64 v[225:226], v[4:5], off offset:32
	global_load_b64 v[4:5], v[4:5], off offset:48
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 4
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[215:216], v[8:9], v[138:145]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[217:218], v[223:224], v[138:145]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[219:220], v[225:226], v[138:145]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[221:222], v[4:5], v[138:145]
	s_cbranch_scc1 .LBB5_52
; %bb.53:                               ;   in Loop: Header=BB5_47 Depth=2
	v_lshl_add_u32 v8, s25, 5, v162
	s_lshl_b32 s4, s25, 4
	v_mov_b32_e32 v9, 0xff800000
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s4, s23
	ds_load_b96 v[3:5], v8 offset:49154
	ds_load_u16_d16 v215, v8 offset:49166
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s4, 15
	v_or_b32_e32 v216, s4, v158
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s5, s21
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s24, s4
	s_and_saveexec_b32 s5, s3
	s_cbranch_execz .LBB5_55
; %bb.54:                               ;   in Loop: Header=BB5_47 Depth=2
	ds_load_u16_d16 v9, v8 offset:49152
	v_cmp_le_i32_e32 vcc_lo, v216, v7
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s4, vcc_lo
	v_mul_f32_e32 v138, v6, v138
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v9, v138, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v9, 0xff800000, v9, vcc_lo
.LBB5_55:                               ;   in Loop: Header=BB5_47 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_mul_f32_e32 v138, v6, v139
	v_or_b32_e32 v139, 2, v216
	v_cmp_ge_i32_e32 vcc_lo, v216, v7
	s_xor_b32 s5, s4, -1
	v_mul_f32_e32 v140, v6, v140
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s4, v139, v7
	v_or_b32_e32 v139, 3, v216
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s6, s5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s2, s6
	s_and_b32 s4, s5, s4
	v_cmp_gt_i32_e32 vcc_lo, v139, v7
	v_mul_f32_e32 v139, v6, v141
	s_wait_dscnt 0x1
	v_fma_mix_f32 v138, v3, v138, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v3, v3, v140, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_or_b32_e32 v140, 4, v216
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	v_cndmask_b32_e64 v138, v138, 0xff800000, s6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v141, v3, 0xff800000, s4
	v_fma_mix_f32 v3, v4, v139, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v139, v6, v142
	s_and_b32 s4, s5, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v140, v7
	v_or_b32_e32 v140, 5, v216
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v142, v3, 0xff800000, s4
	v_fma_mix_f32 v3, v4, v139, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v4, v6, v143
	s_and_b32 s4, s5, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v140, v7
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	v_or_b32_e32 v140, 7, v216
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v139, v3, 0xff800000, s4
	v_fma_mix_f32 v3, v5, v4, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v4, 6, v216
	s_and_b32 s4, s5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v143, v3, 0xff800000, s4
	v_cmp_gt_i32_e32 vcc_lo, v4, v7
	v_mul_f32_e32 v3, v6, v144
	v_cmp_gt_i32_e64 s4, v140, v7
	v_mul_f32_e32 v4, v6, v145
	v_max3_num_f32 v140, v9, 0xff800000, v138
	s_and_b32 s6, s5, vcc_lo
	v_fma_mix_f32 v3, v5, v3, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_and_b32 s4, s5, s4
	s_wait_dscnt 0x0
	v_fma_mix_f32 v4, v215, v4, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v5, v140, v141, v142
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s2, s6
	s_or_b32 s4, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v140, v3, 0xff800000, s5
	v_cndmask_b32_e64 v4, v4, 0xff800000, s4
	v_max3_num_f32 v3, v5, v139, v143
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v3, v3, v140, v4
	v_mov_b32_e32 v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v5, v5, s18, 0xfedcba98
	v_max3_num_f32 v3, v214, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v4, v4, v3 :: v_dual_sub_f32 v5, v9, v3
	v_dual_sub_f32 v9, v138, v3 :: v_dual_add_nc_u32 v138, 0xc080, v8
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v3
	v_dual_mul_f32 v4, 0x3fb8aa3b, v4 :: v_dual_mul_f32 v5, 0x3fb8aa3b, v5
	v_dual_sub_f32 v143, v143, v3 :: v_dual_sub_f32 v140, v140, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v144, v4
	v_dual_sub_f32 v4, v141, v3 :: v_dual_sub_f32 v141, v142, v3
	v_exp_f32_e32 v142, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v140, 0x3fb8aa3b, v140
	v_mul_f32_e32 v145, 0x3fb8aa3b, v4
	ds_load_2addr_b32 v[4:5], v138 offset1:1
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_exp_f32_e32 v140, v140
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v142, v142, 0, vcc_lo
	v_add_nc_u32_e32 v8, 0xc088, v8
	v_exp_f32_e32 v141, v141
	ds_load_2addr_b32 v[215:216], v8 offset1:1
	v_dual_mul_f32 v9, 0x3fb8aa3b, v9 :: v_dual_sub_f32 v138, v139, v3
	v_cndmask_b32_e64 v219, v140, 0, vcc_lo
	v_cndmask_b32_e64 v218, v141, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v9, v9
	v_mul_f32_e32 v138, 0x3fb8aa3b, v138
	v_exp_f32_e32 v8, v145
	v_cndmask_b32_e64 v145, v144, 0, vcc_lo
	s_wait_dscnt 0x1
	v_fma_mix_f32 v144, v4, v142, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v141, v5, v218, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v9, v9, 0, vcc_lo
	v_cndmask_b32_e64 v8, v8, 0, vcc_lo
	v_mul_f32_e32 v139, 0x3fb8aa3b, v143
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_mix_f32 v143, v4, v9, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v142, v9
	v_fma_mix_f32 v142, v5, v8, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_cndmask_b32_e64 v4, v138, 0, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v138, v216, v145, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_max3_num_f32 v5, v144, 0, v143
	v_add_f32_e32 v8, v8, v9
	v_exp_f32_e32 v139, v139
	v_fma_mix_f32 v140, v215, v4, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v9, v216, v219, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v5, v5, v142, v141
	v_add_f32_e32 v8, v218, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v217, v139, 0, vcc_lo
	v_add_f32_e32 v4, v4, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_mix_f32 v139, v215, v217, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v4, v217, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v5, v5, v140, v139
	v_add_f32_e32 v4, v219, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v215, v5, v9, v138
	v_dual_add_f32 v5, v145, v4 :: v_dual_mov_b32 v8, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v8, v8, s18, 0xfedcba98
	v_max_num_f32_e32 v4, v8, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v8, v5 :: v_dual_max_num_f32 v145, v215, v4
	v_permlanex16_b32 v8, v8, s18, 0xfedcba98
	v_mov_b32_e32 v4, v212
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_lt_f32_e32 0, v145
	s_cbranch_execz .LBB5_46
; %bb.56:                               ;   in Loop: Header=BB5_47 Depth=2
	v_div_scale_f32 v4, null, 0x43e00000, 0x43e00000, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v215, v4
	v_fma_f32 v216, -v4, v215, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v215, v216, v215
	v_div_scale_f32 v216, vcc_lo, v145, 0x43e00000, v145
	v_mul_f32_e32 v217, v216, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v218, -v4, v217, v216
	v_fmac_f32_e32 v217, v218, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v4, v217, v216
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v4, v4, v215, v217
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v4, v4, 0x43e00000, v145
	v_max_num_f32_e32 v4, 0x1f800000, v4
	s_branch .LBB5_46
.LBB5_57:                               ;   in Loop: Header=BB5_47 Depth=2
	v_mov_b32_e32 v3, v214
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v214, v3
	s_add_co_i32 s25, s25, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s25, 4
	s_cbranch_scc0 .LBB5_47
	s_branch .LBB5_8
.LBB5_58:
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB5_60
; %bb.59:
	v_div_scale_f32 v0, null, v213, v213, 1.0
	v_div_scale_f32 v3, vcc_lo, 1.0, v213, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v1, v0
	v_fma_f32 v2, -v0, v1, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v2, v1
	v_mul_f32_e32 v2, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v0, v2, v3
	v_fmac_f32_e32 v2, v4, v1
	v_mul_lo_u32 v4, 0x1800, v146
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v0, -v0, v2, v3
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v0, v0, v1, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v2, v148, 8, v4
	v_cmp_lt_f32_e32 vcc_lo, 0, v213
	v_mov_b32_e32 v1, 0
	v_div_fixup_f32 v3, v0, v213, 1.0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_or_b32_e32 v0, v2, v158
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_mul_f32_e32 v140, v212, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v138, vcc_lo, s10, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v139, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v0, v130, v140 :: v_dual_mul_f32 v3, v133, v140
	v_dual_mul_f32 v1, v131, v140 :: v_dual_mul_f32 v2, v132, v140
	v_dual_mul_f32 v5, v135, v140 :: v_dual_mul_f32 v4, v134, v140
	v_dual_mul_f32 v7, v137, v140 :: v_dual_mul_f32 v6, v136, v140
	v_dual_mul_f32 v123, v123, v140 :: v_dual_mul_f32 v122, v122, v140
	v_dual_mul_f32 v125, v125, v140 :: v_dual_mul_f32 v124, v124, v140
	v_dual_mul_f32 v127, v127, v140 :: v_dual_mul_f32 v126, v126, v140
	v_dual_mul_f32 v129, v129, v140 :: v_dual_mul_f32 v128, v128, v140
	v_dual_mul_f32 v115, v115, v140 :: v_dual_mul_f32 v112, v112, v140
	v_dual_mul_f32 v99, v99, v140 :: v_dual_mul_f32 v98, v98, v140
	v_dual_mul_f32 v101, v101, v140 :: v_dual_mul_f32 v100, v100, v140
	v_dual_mul_f32 v103, v103, v140 :: v_dual_mul_f32 v114, v114, v140
	v_dual_mul_f32 v117, v117, v140 :: v_dual_mul_f32 v116, v116, v140
	v_dual_mul_f32 v119, v119, v140 :: v_dual_mul_f32 v102, v102, v140
	v_dual_mul_f32 v105, v105, v140 :: v_dual_mul_f32 v104, v104, v140
	v_dual_mul_f32 v118, v118, v140 :: v_dual_mul_f32 v121, v121, v140
	v_dual_mul_f32 v120, v120, v140 :: v_dual_mul_f32 v107, v107, v140
	v_dual_mul_f32 v106, v106, v140 :: v_dual_mul_f32 v109, v109, v140
	v_dual_mul_f32 v108, v108, v140 :: v_dual_mul_f32 v111, v111, v140
	v_dual_mul_f32 v110, v110, v140 :: v_dual_mul_f32 v113, v113, v140
	s_clause 0x7
	global_store_b128 v[138:139], v[0:3], off
	global_store_b128 v[138:139], v[4:7], off offset:16
	global_store_b128 v[138:139], v[122:125], off offset:64
	global_store_b128 v[138:139], v[126:129], off offset:80
	global_store_b128 v[138:139], v[114:117], off offset:128
	global_store_b128 v[138:139], v[118:121], off offset:144
	global_store_b128 v[138:139], v[106:109], off offset:192
	global_store_b128 v[138:139], v[110:113], off offset:208
	v_dual_mul_f32 v0, v90, v140 :: v_dual_mul_f32 v3, v93, v140
	v_dual_mul_f32 v1, v91, v140 :: v_dual_mul_f32 v2, v92, v140
	v_dual_mul_f32 v5, v95, v140 :: v_dual_mul_f32 v4, v94, v140
	v_dual_mul_f32 v7, v97, v140 :: v_dual_mul_f32 v6, v96, v140
	v_dual_mul_f32 v83, v83, v140 :: v_dual_mul_f32 v82, v82, v140
	v_dual_mul_f32 v85, v85, v140 :: v_dual_mul_f32 v84, v84, v140
	v_dual_mul_f32 v87, v87, v140 :: v_dual_mul_f32 v86, v86, v140
	v_dual_mul_f32 v89, v89, v140 :: v_dual_mul_f32 v88, v88, v140
	s_clause 0x5
	global_store_b128 v[138:139], v[98:101], off offset:256
	global_store_b128 v[138:139], v[102:105], off offset:272
	global_store_b128 v[138:139], v[0:3], off offset:320
	global_store_b128 v[138:139], v[4:7], off offset:336
	global_store_b128 v[138:139], v[82:85], off offset:384
	global_store_b128 v[138:139], v[86:89], off offset:400
	v_dual_mul_f32 v0, v74, v140 :: v_dual_mul_f32 v3, v77, v140
	v_dual_mul_f32 v1, v75, v140 :: v_dual_mul_f32 v2, v76, v140
	v_dual_mul_f32 v5, v79, v140 :: v_dual_mul_f32 v4, v78, v140
	v_dual_mul_f32 v7, v81, v140 :: v_dual_mul_f32 v6, v80, v140
	v_dual_mul_f32 v67, v67, v140 :: v_dual_mul_f32 v66, v66, v140
	v_dual_mul_f32 v69, v69, v140 :: v_dual_mul_f32 v68, v68, v140
	v_dual_mul_f32 v71, v71, v140 :: v_dual_mul_f32 v70, v70, v140
	v_dual_mul_f32 v73, v73, v140 :: v_dual_mul_f32 v72, v72, v140
	v_dual_mul_f32 v59, v59, v140 :: v_dual_mul_f32 v58, v58, v140
	v_dual_mul_f32 v61, v61, v140 :: v_dual_mul_f32 v60, v60, v140
	v_dual_mul_f32 v63, v63, v140 :: v_dual_mul_f32 v62, v62, v140
	v_dual_mul_f32 v65, v65, v140 :: v_dual_mul_f32 v64, v64, v140
	s_clause 0x5
	global_store_b128 v[138:139], v[0:3], off offset:448
	global_store_b128 v[138:139], v[4:7], off offset:464
	global_store_b128 v[138:139], v[66:69], off offset:512
	global_store_b128 v[138:139], v[70:73], off offset:528
	global_store_b128 v[138:139], v[58:61], off offset:576
	global_store_b128 v[138:139], v[62:65], off offset:592
	v_dual_mul_f32 v0, v50, v140 :: v_dual_mul_f32 v3, v53, v140
	v_dual_mul_f32 v1, v51, v140 :: v_dual_mul_f32 v2, v52, v140
	v_dual_mul_f32 v5, v55, v140 :: v_dual_mul_f32 v4, v54, v140
	v_dual_mul_f32 v7, v57, v140 :: v_dual_mul_f32 v6, v56, v140
	v_dual_mul_f32 v43, v43, v140 :: v_dual_mul_f32 v42, v42, v140
	v_dual_mul_f32 v45, v45, v140 :: v_dual_mul_f32 v44, v44, v140
	v_dual_mul_f32 v47, v47, v140 :: v_dual_mul_f32 v46, v46, v140
	v_dual_mul_f32 v49, v49, v140 :: v_dual_mul_f32 v48, v48, v140
	v_dual_mul_f32 v35, v35, v140 :: v_dual_mul_f32 v34, v34, v140
	v_dual_mul_f32 v37, v37, v140 :: v_dual_mul_f32 v36, v36, v140
	v_dual_mul_f32 v39, v39, v140 :: v_dual_mul_f32 v38, v38, v140
	v_dual_mul_f32 v41, v41, v140 :: v_dual_mul_f32 v40, v40, v140
	s_clause 0x5
	global_store_b128 v[138:139], v[0:3], off offset:640
	global_store_b128 v[138:139], v[4:7], off offset:656
	global_store_b128 v[138:139], v[42:45], off offset:704
	global_store_b128 v[138:139], v[46:49], off offset:720
	global_store_b128 v[138:139], v[34:37], off offset:768
	global_store_b128 v[138:139], v[38:41], off offset:784
	v_dual_mul_f32 v0, v26, v140 :: v_dual_mul_f32 v3, v29, v140
	v_dual_mul_f32 v1, v27, v140 :: v_dual_mul_f32 v2, v28, v140
	v_dual_mul_f32 v5, v31, v140 :: v_dual_mul_f32 v4, v30, v140
	v_dual_mul_f32 v7, v33, v140 :: v_dual_mul_f32 v6, v32, v140
	v_dual_mul_f32 v19, v19, v140 :: v_dual_mul_f32 v18, v18, v140
	v_dual_mul_f32 v21, v21, v140 :: v_dual_mul_f32 v20, v20, v140
	v_dual_mul_f32 v23, v23, v140 :: v_dual_mul_f32 v22, v22, v140
	v_dual_mul_f32 v25, v25, v140 :: v_dual_mul_f32 v24, v24, v140
	v_dual_mul_f32 v9, v11, v140 :: v_dual_mul_f32 v8, v10, v140
	v_dual_mul_f32 v11, v13, v140 :: v_dual_mul_f32 v10, v12, v140
	v_dual_mul_f32 v13, v15, v140 :: v_dual_mul_f32 v12, v14, v140
	v_dual_mul_f32 v15, v17, v140 :: v_dual_mul_f32 v14, v16, v140
	s_clause 0x5
	global_store_b128 v[138:139], v[0:3], off offset:832
	global_store_b128 v[138:139], v[4:7], off offset:848
	global_store_b128 v[138:139], v[18:21], off offset:896
	global_store_b128 v[138:139], v[22:25], off offset:912
	global_store_b128 v[138:139], v[8:11], off offset:960
	global_store_b128 v[138:139], v[12:15], off offset:976
.LBB5_60:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end5:
	.size	attention_fp8_e4m3_fa2_gqa_packet_gfx1201, .Lfunc_end5-attention_fp8_e4m3_fa2_gqa_packet_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_gfx1201
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
		.amdhsa_next_free_vgpr 239
		.amdhsa_next_free_sgpr 27
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-attention_fp8_e4m3_fa2_gqa_packet_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_vgpr, 239
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.numbered_sgpr, 27
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9968
; TotalNumSgprs: 29
; NumVgprs: 239
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 29
; NumVGPRsForWavesPerEU: 239
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
	.protected	attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x28
	s_load_b64 s[16:17], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s12, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s13, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s14, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s4, s17, -9
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s4, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB6_63
; %bb.1:
	s_and_b32 s2, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s2, 3
	s_cbranch_scc1 .LBB6_63
; %bb.2:
	s_lshl_b32 s14, ttmp9, 7
	s_mul_i32 s3, s15, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s14, s3
	s_cbranch_scc1 .LBB6_63
; %bb.3:
	s_lshr_b32 s12, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s12, s17
	s_cbranch_scc1 .LBB6_63
; %bb.4:
	s_clause 0x1
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b64 s[18:19], s[0:1], 0x20
	v_and_b32_e32 v5, 31, v0
	v_mov_b32_e32 v3, -1
	v_bfrev_b32_e32 v4, -2
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_u32_e32 22, v5
	s_cbranch_execz .LBB6_8
; %bb.5:
	s_mul_hi_i32 s1, s14, 0x2aaaaaab
	v_bfrev_b32_e32 v4, -2
	s_lshr_b32 s13, s1, 31
	v_mov_b32_e32 v3, -1
	v_add3_u32 v1, s1, s13, v5
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v1
	s_cbranch_execz .LBB6_7
; %bb.6:
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, vcc_lo, s18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s19, v2, vcc_lo
	global_load_b32 v3, v[1:2], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v4, v3
.LBB6_7:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
.LBB6_8:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	s_or_b32 exec_lo, exec_lo, s0
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_lshrrev_b32_e32 v154, 4, v5
	v_lshl_add_u32 v155, v5, 4, 0
	v_dual_mov_b32 v16, 0x5040100 :: v_dual_and_b32 v11, 3, v0
	v_xor_b32_e32 v2, 16, v1
	v_xor_b32_e32 v24, 4, v1
	v_lshrrev_b32_e32 v6, 5, v0
	v_or_b32_e32 v20, 0x100, v0
	v_xor_b32_e32 v25, 2, v1
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	v_and_b32_e32 v7, 15, v0
	v_lshl_add_u32 v23, v6, 11, 0
	v_bfe_u32 v17, v0, 5, 1
	v_or_b32_e32 v21, 0x200, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v1, v2, vcc_lo
	v_dual_mov_b32 v2, 0 :: v_dual_and_b32 v9, 0x7f, v0
	v_lshlrev_b32_e32 v158, 3, v6
	v_xor_b32_e32 v26, 1, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v15, 0x6020400 :: v_dual_lshlrev_b32 v8, 2, v8
	v_lshlrev_b32_e32 v10, 3, v5
	v_xor_b32_e32 v5, 8, v1
	s_add_co_i32 s1, s17, 0x1ff
	ds_bpermute_b32 v13, v8, v3
	ds_bpermute_b32 v8, v8, v4
	s_cvt_f32_u32 s13, s17
	v_cmp_gt_u32_e32 vcc_lo, 32, v5
	v_and_b32_e32 v14, 1, v0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s21, 0
	s_mul_i32 s20, s15, 0x1800
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v1, v5, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s15, s1
	v_cmp_eq_u32_e64 s1, v154, v17
	v_and_or_b32 v17, 0x280, v21, v9
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[24:25], s[4:5], s[20:21]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v159, 0x3070105, v15, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 2, v11
	v_lshlrev_b32_e32 v5, 2, v5
	v_and_or_b32 v15, v6, 4, v154
	v_s_rcp_f32 s20, s13
	v_or_b32_e32 v22, 0x300, v0
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v13
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v4, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v160, 0x3020706, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v24
	v_lshlrev_b32_e32 v27, 3, v11
	ds_bpermute_b32 v13, v5, v3
	ds_bpermute_b32 v5, v5, v4
	v_lshlrev_b32_e32 v8, 4, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v1, v24, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v25
	v_and_b32_e32 v18, 16, v0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s20, s15, s20
	v_and_or_b32 v162, v8, 48, v7
	v_lshlrev_b32_e32 v6, 2, v14
	v_lshrrev_b32_e32 v14, 5, v21
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v21, v1, v25, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v26
	v_add_nc_u32_e32 v161, 0, v18
	v_lshrrev_b32_e32 v18, 5, v22
	s_trunc_f32 s20, s20
	v_lshlrev_b32_e32 v164, 2, v21
	s_mul_i32 s28, s2, 6
	v_bfe_u32 v12, v0, 2, 2
	s_xor_b32 s22, s20, 0x80000000
	s_cvt_u32_f32 s20, s20
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v13
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v4, v5
	v_lshrrev_b32_e32 v5, 5, v20
	v_and_or_b32 v13, 0x180, v20, v9
	v_or_b32_e32 v20, v8, v7
	ds_bpermute_b32 v16, v6, v3
	ds_bpermute_b32 v6, v6, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v26, vcc_lo
	v_and_or_b32 v9, 0x380, v22, v9
	v_add_nc_u32_e32 v20, s14, v20
	v_and_or_b32 v22, v5, 12, v154
	s_fmac_f32 s15, s22, s13
	v_lshl_or_b32 v12, v11, 6, v12
	v_lshlrev_b32_e32 v21, 4, v9
	v_mul_hi_i32 v7, 0x2aaaaaab, v20
	v_mov_b32_e32 v9, v2
	v_lshlrev_b32_e32 v165, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s15, 31
	v_lshlrev_b32_e32 v19, 4, v0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s15, s13
	v_lshlrev_b32_e32 v157, 3, v154
	v_xad_u32 v166, 0x120, v10, v23
	v_xad_u32 v167, 0x124, v10, v23
	s_add_co_ci_u32 s13, s20, 0
	s_addk_co_i32 s14, 0x7f
	s_wait_dscnt 0x1
	v_max_i32_e32 v5, v3, v16
	s_wait_dscnt 0x0
	v_min_i32_e32 v6, v4, v6
	v_lshrrev_b32_e32 v3, 31, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s15, s13, 0xffff
	v_xad_u32 v168, 0x240, v10, v23
	ds_bpermute_b32 v8, v164, v5
	ds_bpermute_b32 v16, v164, v6
	v_dual_mov_b32 v152, 0xff800000 :: v_dual_add_nc_u32 v3, v7, v3
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s12, s15
	v_xad_u32 v169, 0x244, v10, v23
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s22, s13, s15
	v_mul_lo_u32 v1, v3, 6
	v_mul_lo_u32 v7, v3, 24
	s_cmp_lt_i32 s14, s3
	v_ashrrev_i32_e32 v4, 31, v3
	s_cselect_b32 s23, -1, 0
	s_lshl_b32 s20, s2, 8
	v_xad_u32 v170, 0x360, v10, v23
	s_add_nc_u64 s[14:15], s[6:7], s[20:21]
	v_sub_nc_u32_e32 v1, v20, v1
	s_add_nc_u64 s[26:27], s[8:9], s[20:21]
	s_lshl_b32 s20, s2, 1
	v_cmp_gt_i32_e64 s2, s3, v20
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_wait_dscnt 0x1
	v_max_i32_e32 v5, v5, v8
	s_wait_dscnt 0x0
	v_min_i32_e32 v6, v6, v16
	v_add3_u32 v163, v1, s28, v7
	v_add_co_u32 v146, s3, s26, v10
	ds_bpermute_b32 v7, v165, v5
	ds_bpermute_b32 v8, v165, v6
	v_lshlrev_b32_e32 v1, 8, v163
	v_add_co_u32 v148, vcc_lo, s18, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v149, null, s19, v4, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v16, 0, v1, s2
	v_cndmask_b32_e64 v1, 0, v163, s2
	v_xad_u32 v171, 0x364, v10, v23
	v_xad_u32 v172, 0x520, v10, v23
	v_xad_u32 v173, 0x524, v10, v23
	v_xad_u32 v174, 0x640, v10, v23
	v_lshlrev_b64_e32 v[3:4], 2, v[1:2]
	v_xad_u32 v175, 0x644, v10, v23
	v_xad_u32 v176, 0x760, v10, v23
	v_mov_b32_e32 v153, 0
	v_xad_u32 v177, 0x764, v10, v23
	v_add_co_ci_u32_e64 v147, null, s27, 0, s3
	v_add_co_u32 v150, vcc_lo, s24, v3
	s_wait_dscnt 0x1
	v_max_i32_e32 v1, v5, v7
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v151, null, s25, v4, vcc_lo
	v_lshlrev_b32_e32 v3, 2, v12
	v_lshlrev_b32_e32 v4, 5, v11
	v_readfirstlane_b32 s25, v5
	v_or_b32_e32 v5, 12, v12
	v_or_b32_e32 v6, 0x10c, v12
	v_lshlrev_b32_e32 v7, 6, v0
	v_add3_u32 v178, v23, v3, v4
	v_or_b32_e32 v3, 8, v12
	v_or_b32_e32 v4, 0x108, v12
	v_xor_b32_e32 v5, v5, v27
	v_xor_b32_e32 v6, v6, v27
	v_readfirstlane_b32 s24, v1
	v_xor_b32_e32 v3, v3, v27
	v_xor_b32_e32 v4, v4, v27
	v_lshl_add_u32 v181, v5, 2, v23
	v_lshl_add_u32 v182, v6, 2, v23
	v_or_b32_e32 v5, 20, v12
	v_lshl_add_u32 v179, v3, 2, v23
	v_lshl_add_u32 v180, v4, 2, v23
	v_or_b32_e32 v3, 16, v12
	v_or_b32_e32 v4, 0x110, v12
	v_or_b32_e32 v6, 0x114, v12
	v_xor_b32_e32 v5, v5, v27
	v_mov_b32_e32 v8, v2
	v_xor_b32_e32 v3, v3, v27
	v_xor_b32_e32 v4, v4, v27
	v_xor_b32_e32 v6, v6, v27
	v_lshl_add_u32 v185, v5, 2, v23
	v_or_b32_e32 v5, 0x11c, v12
	v_lshl_add_u32 v183, v3, 2, v23
	v_lshl_add_u32 v184, v4, 2, v23
	v_lshl_add_u32 v186, v6, 2, v23
	v_or_b32_e32 v3, 0x118, v12
	v_or_b32_e32 v4, 28, v12
	v_or_b32_e32 v6, 40, v12
	v_xor_b32_e32 v5, v5, v27
	v_add_co_u32 v16, s3, s4, v16
	v_xor_b32_e32 v3, v3, v27
	v_xor_b32_e32 v4, v4, v27
	v_xor_b32_e32 v6, v6, v27
	v_lshl_add_u32 v190, v5, 2, v23
	v_or_b32_e32 v5, 48, v12
	v_lshl_add_u32 v188, v3, 2, v23
	v_lshl_add_u32 v189, v4, 2, v23
	v_lshl_add_u32 v191, v6, 2, v23
	v_or_b32_e32 v3, 44, v12
	v_or_b32_e32 v4, 0x12c, v12
	v_or_b32_e32 v6, 0x130, v12
	v_xor_b32_e32 v5, v5, v27
	v_dual_mov_b32 v214, 1.0 :: v_dual_lshlrev_b32 v13, 4, v13
	v_xor_b32_e32 v3, v3, v27
	v_xor_b32_e32 v4, v4, v27
	v_xor_b32_e32 v6, v6, v27
	v_lshl_add_u32 v195, v5, 2, v23
	v_or_b32_e32 v5, 0x138, v12
	v_lshl_add_u32 v193, v3, 2, v23
	v_lshl_add_u32 v194, v4, 2, v23
	v_lshl_add_u32 v196, v6, 2, v23
	v_or_b32_e32 v3, 0x134, v12
	v_or_b32_e32 v4, 56, v12
	v_or_b32_e32 v6, 60, v12
	v_xor_b32_e32 v5, v5, v27
	v_and_or_b32 v14, v14, 20, v154
	v_xor_b32_e32 v3, v3, v27
	v_xor_b32_e32 v4, v4, v27
	v_xor_b32_e32 v6, v6, v27
	v_lshl_add_u32 v200, v5, 2, v23
	v_mov_b32_e32 v5, v2
	v_lshl_add_u32 v198, v3, 2, v23
	v_lshl_add_u32 v199, v4, 2, v23
	v_lshl_add_u32 v201, v6, 2, v23
	v_dual_mov_b32 v3, v2 :: v_dual_mov_b32 v4, v2
	v_dual_mov_b32 v6, v2 :: v_dual_add_nc_u32 v1, v23, v10
	v_and_b32_e32 v10, 0x3000, v7
	v_or_b32_e32 v7, 24, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v20, null, s5, 0, s3
	v_and_or_b32 v18, v18, 28, v154
	v_add_co_u32 v203, vcc_lo, v16, v157
	v_xor_b32_e32 v7, v7, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v204, null, 0, v20, vcc_lo
	v_cmp_gt_u32_e64 s0, 64, v0
	v_lshl_add_u32 v156, v0, 1, 0
	v_lshl_add_u32 v187, v7, 2, v23
	v_or_b32_e32 v7, 0x128, v12
	s_add_nc_u64 s[18:19], s[6:7], s[20:21]
	s_add_nc_u64 s[8:9], s[8:9], s[20:21]
	s_mov_b32 s7, 0x76543210
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v7, v7, v27
	v_lshl_add_u32 v192, v7, 2, v23
	v_or_b32_e32 v7, 52, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v7, v7, v27
	v_lshl_add_u32 v197, v7, 2, v23
	v_or_b32_e32 v7, 0x13c, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v7, v7, v27
	v_lshl_add_u32 v202, v7, 2, v23
	v_mov_b32_e32 v7, v2
	v_lshlrev_b32_e32 v17, 4, v17
	v_lshlrev_b32_e32 v205, 3, v15
	v_add_nc_u32_e32 v207, 0, v13
	v_lshlrev_b32_e32 v208, 3, v14
	v_add_nc_u32_e32 v211, v155, v10
	v_add_nc_u32_e32 v213, 0, v17
	v_dual_mov_b32 v17, v9 :: v_dual_lshlrev_b32 v206, 3, v22
	v_mov_b32_e32 v15, v7
	v_lshlrev_b32_e32 v209, 3, v18
	v_add_nc_u32_e32 v212, 0, v19
	v_mov_b32_e32 v10, v2
	v_dual_mov_b32 v11, v3 :: v_dual_add_nc_u32 v210, 0, v21
	v_mov_b32_e32 v25, v9
	v_mov_b32_e32 v33, v9
	v_mov_b32_e32 v41, v9
	v_mov_b32_e32 v49, v9
	v_mov_b32_e32 v57, v9
	v_mov_b32_e32 v65, v9
	v_mov_b32_e32 v73, v9
	v_mov_b32_e32 v81, v9
	v_mov_b32_e32 v89, v9
	v_mov_b32_e32 v97, v9
	v_mov_b32_e32 v105, v9
	v_mov_b32_e32 v113, v9
	v_mov_b32_e32 v121, v9
	v_mov_b32_e32 v129, v9
	v_dual_mov_b32 v137, v9 :: v_dual_mov_b32 v16, v8
	v_dual_mov_b32 v14, v6 :: v_dual_mov_b32 v13, v5
	v_mov_b32_e32 v12, v4
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	v_mov_b32_e32 v28, v4
	v_dual_mov_b32 v22, v6 :: v_dual_mov_b32 v21, v5
	v_mov_b32_e32 v26, v2
	v_dual_mov_b32 v20, v4 :: v_dual_mov_b32 v19, v3
	v_mov_b32_e32 v40, v8
	v_mov_b32_e32 v18, v2
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	v_mov_b32_e32 v36, v4
	v_dual_mov_b32 v30, v6 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v27, v3
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v39, v7
	v_mov_b32_e32 v44, v4
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v42, v2 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v47, v7
	v_mov_b32_e32 v52, v4
	v_dual_mov_b32 v46, v6 :: v_dual_mov_b32 v45, v5
	v_dual_mov_b32 v50, v2 :: v_dual_mov_b32 v43, v3
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v55, v7
	v_mov_b32_e32 v60, v4
	v_dual_mov_b32 v54, v6 :: v_dual_mov_b32 v53, v5
	v_dual_mov_b32 v58, v2 :: v_dual_mov_b32 v51, v3
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v63, v7
	v_mov_b32_e32 v68, v4
	v_dual_mov_b32 v62, v6 :: v_dual_mov_b32 v61, v5
	v_dual_mov_b32 v66, v2 :: v_dual_mov_b32 v59, v3
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v71, v7
	v_mov_b32_e32 v76, v4
	v_dual_mov_b32 v70, v6 :: v_dual_mov_b32 v69, v5
	v_dual_mov_b32 v74, v2 :: v_dual_mov_b32 v67, v3
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v79, v7
	v_mov_b32_e32 v84, v4
	v_dual_mov_b32 v78, v6 :: v_dual_mov_b32 v77, v5
	v_dual_mov_b32 v82, v2 :: v_dual_mov_b32 v75, v3
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v87, v7
	v_mov_b32_e32 v92, v4
	v_dual_mov_b32 v86, v6 :: v_dual_mov_b32 v85, v5
	v_dual_mov_b32 v90, v2 :: v_dual_mov_b32 v83, v3
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v95, v7
	v_mov_b32_e32 v100, v4
	v_dual_mov_b32 v94, v6 :: v_dual_mov_b32 v93, v5
	v_dual_mov_b32 v98, v2 :: v_dual_mov_b32 v91, v3
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v103, v7
	v_mov_b32_e32 v108, v4
	v_dual_mov_b32 v102, v6 :: v_dual_mov_b32 v101, v5
	v_dual_mov_b32 v106, v2 :: v_dual_mov_b32 v99, v3
	v_dual_mov_b32 v120, v8 :: v_dual_mov_b32 v111, v7
	v_mov_b32_e32 v116, v4
	v_dual_mov_b32 v110, v6 :: v_dual_mov_b32 v109, v5
	v_dual_mov_b32 v114, v2 :: v_dual_mov_b32 v107, v3
	v_dual_mov_b32 v128, v8 :: v_dual_mov_b32 v119, v7
	v_mov_b32_e32 v124, v4
	v_dual_mov_b32 v118, v6 :: v_dual_mov_b32 v117, v5
	v_dual_mov_b32 v122, v2 :: v_dual_mov_b32 v115, v3
	v_dual_mov_b32 v136, v8 :: v_dual_mov_b32 v127, v7
	v_mov_b32_e32 v132, v4
	v_dual_mov_b32 v126, v6 :: v_dual_mov_b32 v125, v5
	v_dual_mov_b32 v130, v2 :: v_dual_mov_b32 v123, v3
	v_dual_mov_b32 v135, v7 :: v_dual_mov_b32 v134, v6
	v_mov_b32_e32 v133, v5
	v_mov_b32_e32 v131, v3
	s_branch .LBB6_11
.LBB6_9:                                ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v152, v3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB6_10:                               ;   in Loop: Header=BB6_11 Depth=1
	s_add_co_i32 s13, s13, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s13, s22
	s_cselect_b32 s3, -1, 0
	s_xor_b32 s4, s20, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s4, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_59
.LBB6_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_48 Depth 2
                                        ;       Child Loop BB6_53 Depth 3
	s_lshl_b32 s21, s13, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s21, s24
	s_cselect_b32 s20, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s20
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_10
; %bb.12:                               ;   in Loop: Header=BB6_11 Depth=1
	v_or_b32_e32 v3, s21, v162
                                        ; implicit-def: $vgpr6
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[138:139], null, 0x408, v3, s[14:15]
	v_cmp_ge_i32_e32 vcc_lo, s24, v3
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s3
	s_cbranch_execz .LBB6_14
; %bb.13:                               ;   in Loop: Header=BB6_11 Depth=1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_u32 v3, s3, v138, v205
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, s3
	v_add_co_u32 v8, s3, v138, v206
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, 0, v139, s3
	s_clause 0x3
	global_load_b64 v[140:141], v[3:4], off
	global_load_b64 v[142:143], v[3:4], off offset:16
	global_load_b64 v[6:7], v[8:9], off
	global_load_b64 v[8:9], v[8:9], off offset:16
	s_wait_loadcnt 0x2
	ds_store_b128 v212, v[140:143]
.LBB6_14:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s4
	s_cbranch_execz .LBB6_16
; %bb.15:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v4, v2
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v6, v9
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	ds_store_b128 v212, v[2:5]
.LBB6_16:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt 0x0
	ds_store_b128 v207, v[6:9]
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB6_18
; %bb.17:                               ;   in Loop: Header=BB6_11 Depth=1
	v_add_co_u32 v3, vcc_lo, v138, v208
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, vcc_lo
	v_add_co_u32 v8, vcc_lo, v138, v209
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, 0, v139, vcc_lo
	s_clause 0x3
	global_load_b64 v[138:139], v[3:4], off
	global_load_b64 v[140:141], v[3:4], off offset:16
	global_load_b64 v[6:7], v[8:9], off
	global_load_b64 v[8:9], v[8:9], off offset:16
	s_wait_loadcnt 0x2
	ds_store_b128 v213, v[138:141]
.LBB6_18:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB6_20
; %bb.19:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v4, v2
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v6, v9
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	ds_store_b128 v213, v[2:5]
.LBB6_20:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v139, 0 :: v_dual_add_nc_u32 v140, s21, v158
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v3, 0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v210, v[6:9]
	v_cmpx_ge_i32_e64 s24, v140
	s_cbranch_execz .LBB6_22
; %bb.21:                               ;   in Loop: Header=BB6_11 Depth=1
	v_mad_co_i64_i32 v[3:4], null, 0x408, v140, v[146:147]
	global_load_b64 v[3:4], v[3:4], off
.LBB6_22:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v5, 0x8000, v1
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v5, v3, v4 offset1:1
	v_cmpx_gt_i32_e64 s24, v140
	s_cbranch_execz .LBB6_24
; %bb.23:                               ;   in Loop: Header=BB6_11 Depth=1
	v_or_b32_e32 v3, 1, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[3:4], null, 0x408, v3, v[146:147]
	global_load_b64 v[138:139], v[3:4], off
.LBB6_24:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 2, v140
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v166, v138 offset:32768
	ds_store_b32 v167, v139 offset:32768
	v_cmpx_ge_i32_e64 s24, v7
	s_cbranch_execz .LBB6_26
; %bb.25:                               ;   in Loop: Header=BB6_11 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v7, v[146:147]
	global_load_b64 v[5:6], v[5:6], off
.LBB6_26:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 3, v140
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v168, v5 offset:32768
	ds_store_b32 v169, v6 offset:32768
	v_cmpx_ge_i32_e64 s24, v7
	s_cbranch_execz .LBB6_28
; %bb.27:                               ;   in Loop: Header=BB6_11 Depth=1
	v_mad_co_i64_i32 v[3:4], null, 0x408, v7, v[146:147]
	global_load_b64 v[3:4], v[3:4], off
.LBB6_28:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v9, 4, v140
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v170, v3 offset:32768
	ds_store_b32 v171, v4 offset:32768
	v_cmpx_ge_i32_e64 s24, v9
	s_cbranch_execz .LBB6_30
; %bb.29:                               ;   in Loop: Header=BB6_11 Depth=1
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[146:147]
	global_load_b64 v[7:8], v[3:4], off
.LBB6_30:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v3, 5, v140
	v_add_nc_u32_e32 v4, 0x8400, v1
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v4, v7, v8 offset1:1
	v_cmpx_ge_i32_e64 s24, v3
	s_cbranch_execz .LBB6_32
; %bb.31:                               ;   in Loop: Header=BB6_11 Depth=1
	v_mad_co_i64_i32 v[3:4], null, 0x408, v3, v[146:147]
	global_load_b64 v[5:6], v[3:4], off
.LBB6_32:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v9, 6, v140
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v172, v5 offset:32768
	ds_store_b32 v173, v6 offset:32768
	v_cmpx_ge_i32_e64 s24, v9
	s_cbranch_execz .LBB6_34
; %bb.33:                               ;   in Loop: Header=BB6_11 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v9, v[146:147]
	global_load_b64 v[7:8], v[5:6], off
.LBB6_34:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v5, 7, v140
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v174, v7 offset:32768
	ds_store_b32 v175, v8 offset:32768
	v_cmpx_ge_i32_e64 s24, v5
	s_cbranch_execz .LBB6_36
; %bb.35:                               ;   in Loop: Header=BB6_11 Depth=1
	v_mad_co_i64_i32 v[3:4], null, 0x408, v5, v[146:147]
	global_load_b64 v[3:4], v[3:4], off
.LBB6_36:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt 0x0
	ds_store_b32 v176, v3 offset:32768
	ds_store_b32 v177, v4 offset:32768
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB6_38
; %bb.37:                               ;   in Loop: Header=BB6_11 Depth=1
	v_add_nc_u32_e32 v7, 0x8000, v178
	v_add_nc_u32_e32 v8, 0x8400, v178
	ds_load_2addr_b32 v[3:4], v7 offset1:4
	ds_load_2addr_b32 v[5:6], v8 offset1:4
	s_wait_dscnt 0x1
	ds_bpermute_b32 v9, v165, v3
	s_wait_dscnt 0x1
	ds_bpermute_b32 v138, v165, v5
	ds_bpermute_b32 v139, v165, v4
	ds_bpermute_b32 v140, v165, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v159
	s_wait_dscnt 0x2
	v_perm_b32 v5, v138, v5, v159
	s_wait_dscnt 0x1
	v_perm_b32 v9, v139, v4, v159
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v159
	ds_bpermute_b32 v4, v164, v3
	ds_bpermute_b32 v138, v164, v5
	ds_bpermute_b32 v139, v164, v9
	ds_bpermute_b32 v140, v164, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v4, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v5, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v9, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v160
	ds_store_b128 v211, v[3:6] offset:16384
	ds_load_b32 v3, v179 offset:32768
	ds_load_b32 v4, v180 offset:32768
	ds_load_b32 v5, v181 offset:32768
	ds_load_b32 v6, v182 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v165, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v165, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v165, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v165, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v159
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v159
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v159
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v159
	ds_bpermute_b32 v9, v164, v3
	ds_bpermute_b32 v138, v164, v4
	ds_bpermute_b32 v139, v164, v5
	ds_bpermute_b32 v140, v164, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v160
	ds_store_b128 v211, v[3:6] offset:16896
	ds_load_b32 v3, v183 offset:32768
	ds_load_b32 v4, v184 offset:32768
	ds_load_b32 v5, v185 offset:32768
	ds_load_b32 v6, v186 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v165, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v165, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v165, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v165, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v159
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v159
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v159
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v159
	ds_bpermute_b32 v9, v164, v3
	ds_bpermute_b32 v138, v164, v4
	ds_bpermute_b32 v139, v164, v5
	ds_bpermute_b32 v140, v164, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v160
	ds_store_b128 v211, v[3:6] offset:17408
	ds_load_b32 v3, v187 offset:32768
	ds_load_b32 v4, v188 offset:32768
	ds_load_b32 v5, v189 offset:32768
	ds_load_b32 v6, v190 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v165, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v165, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v165, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v165, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v159
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v159
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v159
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v159
	ds_bpermute_b32 v9, v164, v3
	ds_bpermute_b32 v138, v164, v4
	ds_bpermute_b32 v139, v164, v5
	ds_bpermute_b32 v140, v164, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v9, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v138, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v139, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v140, v6, v160
	ds_store_b128 v211, v[3:6] offset:17920
	ds_load_2addr_b32 v[3:4], v7 offset0:32 offset1:36
	ds_load_2addr_b32 v[5:6], v8 offset0:32 offset1:36
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v165, v3
	s_wait_dscnt 0x1
	ds_bpermute_b32 v8, v165, v5
	ds_bpermute_b32 v9, v165, v4
	ds_bpermute_b32 v138, v165, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v159
	s_wait_dscnt 0x2
	v_perm_b32 v5, v8, v5, v159
	s_wait_dscnt 0x1
	v_perm_b32 v7, v9, v4, v159
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v159
	ds_bpermute_b32 v4, v164, v3
	ds_bpermute_b32 v8, v164, v5
	ds_bpermute_b32 v9, v164, v7
	ds_bpermute_b32 v138, v164, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v4, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v5, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v7, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v160
	ds_store_b128 v211, v[3:6] offset:18432
	ds_load_b32 v3, v191 offset:32768
	ds_load_b32 v4, v192 offset:32768
	ds_load_b32 v5, v193 offset:32768
	ds_load_b32 v6, v194 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v165, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v165, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v165, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v165, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v159
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v159
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v159
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v159
	ds_bpermute_b32 v7, v164, v3
	ds_bpermute_b32 v8, v164, v4
	ds_bpermute_b32 v9, v164, v5
	ds_bpermute_b32 v138, v164, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v160
	ds_store_b128 v211, v[3:6] offset:18944
	ds_load_b32 v3, v195 offset:32768
	ds_load_b32 v4, v196 offset:32768
	ds_load_b32 v5, v197 offset:32768
	ds_load_b32 v6, v198 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v165, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v165, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v165, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v165, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v159
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v159
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v159
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v159
	ds_bpermute_b32 v7, v164, v3
	ds_bpermute_b32 v8, v164, v4
	ds_bpermute_b32 v9, v164, v5
	ds_bpermute_b32 v138, v164, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v160
	ds_store_b128 v211, v[3:6] offset:19456
	ds_load_b32 v3, v199 offset:32768
	ds_load_b32 v4, v200 offset:32768
	ds_load_b32 v5, v201 offset:32768
	ds_load_b32 v6, v202 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v165, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v165, v4
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v165, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v165, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v159
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v159
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v159
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v159
	ds_bpermute_b32 v7, v164, v3
	ds_bpermute_b32 v8, v164, v4
	ds_bpermute_b32 v9, v164, v5
	ds_bpermute_b32 v138, v164, v6
	s_wait_dscnt 0x3
	v_perm_b32 v3, v7, v3, v160
	s_wait_dscnt 0x2
	v_perm_b32 v4, v8, v4, v160
	s_wait_dscnt 0x1
	v_perm_b32 v5, v9, v5, v160
	s_wait_dscnt 0x0
	v_perm_b32 v6, v138, v6, v160
	ds_store_b128 v211, v[3:6] offset:19968
.LBB6_38:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB6_42
; %bb.39:                               ;   in Loop: Header=BB6_11 Depth=1
	v_or_b32_e32 v4, s21, v0
	v_mov_b32_e32 v3, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s24, v4
	s_cbranch_execz .LBB6_41
; %bb.40:                               ;   in Loop: Header=BB6_11 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v4, s[18:19]
	v_mad_co_i64_i32 v[7:8], null, 0x408, v4, s[8:9]
	global_load_d16_b16 v3, v[5:6], off offset:1024
	global_load_d16_hi_b16 v3, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v4.h, 8, v3.l
	v_lshrrev_b16 v4.l, 8, v3.h
	v_and_b16 v5.h, 0xff, v3.l
	v_and_b16 v5.l, 0xff, v3.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_lshlrev_b16 v3, 8, v4 op_sel_hi:[0,1]
	v_or_b32_e32 v3, v3, v5
.LBB6_41:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	ds_store_b16_d16_hi v156, v3 offset:49152
	ds_store_b16 v156, v3 offset:49280
.LBB6_42:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v6, s16
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB6_44
; %bb.43:                               ;   in Loop: Header=BB6_11 Depth=1
	global_load_b32 v3, v[150:151], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v6, s16, v3
.LBB6_44:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_or_b32 s3, s21, 63
	v_mov_b32_e32 v7, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s25
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s26, s23, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s26, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s3, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB6_46
; %bb.45:                               ;   in Loop: Header=BB6_11 Depth=1
	global_load_b32 v7, v[148:149], off
.LBB6_46:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_or_b32 s3, s21, 16
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s24
	s_cselect_b32 s28, -1, 0
	s_branch .LBB6_48
.LBB6_47:                               ;   in Loop: Header=BB6_48 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_sub_f32_e32 v145, v152, v3
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v152
	v_add_f32_e32 v5, v5, v8
	v_div_scale_f32 v219, null, v4, v4, v142
	v_div_scale_f32 v221, null, v4, v4, v141
	v_mul_f32_e32 v145, 0x3fb8aa3b, v145
	v_div_scale_f32 v223, null, v4, v4, v139
	v_div_scale_f32 v224, null, v4, v4, v9
	v_rcp_f32_e32 v220, v219
	v_lshl_add_u32 v234, s27, 12, v155
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v222, -v219, v220, 1.0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v145, 0, v145 :: v_dual_fmac_f32 v220, v222, v220
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v152, v214, v145
	v_div_scale_f32 v214, null, v4, v4, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v215, v214
	v_fma_f32 v216, -v214, v215, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v215, v216, v215
	v_div_scale_f32 v217, vcc_lo, v152, v4, v152
	v_mul_f32_e32 v216, v217, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v218, -v214, v216, v217
	v_fmac_f32_e32 v216, v218, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v214, -v214, v216, v217
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v214, v214, v215, v216
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v8, v214, v4, v152
	v_dual_mul_f32 v136, v136, v8 :: v_dual_fmac_f32 v5, v153, v145
	v_mul_f32_e32 v132, v132, v8
	v_div_scale_f32 v145, null, v4, v4, v144
	v_div_scale_f32 v153, null, v4, v4, v143
	v_dual_mul_f32 v137, v137, v8 :: v_dual_mul_f32 v130, v130, v8
	v_dual_mul_f32 v135, v135, v8 :: v_dual_mul_f32 v128, v128, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v152, v145
	v_rcp_f32_e32 v215, v153
	v_dual_mul_f32 v134, v134, v8 :: v_dual_mul_f32 v133, v133, v8
	v_dual_mul_f32 v126, v126, v8 :: v_dual_mul_f32 v131, v131, v8
	v_dual_mul_f32 v124, v124, v8 :: v_dual_mul_f32 v129, v129, v8
	v_mul_f32_e32 v122, v122, v8
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v214, -v145, v152, 1.0
	v_fma_f32 v217, -v153, v215, 1.0
	v_dual_mul_f32 v127, v127, v8 :: v_dual_mul_f32 v120, v120, v8
	v_dual_mul_f32 v125, v125, v8 :: v_dual_mul_f32 v118, v118, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v152, v214, v152 :: v_dual_fmac_f32 v215, v217, v215
	v_div_scale_f32 v214, vcc_lo, v144, v4, v144
	v_div_scale_f32 v217, s3, v143, v4, v143
	v_mul_f32_e32 v23, v23, v8
	v_dual_mul_f32 v123, v123, v8 :: v_dual_mul_f32 v116, v116, v8
	v_mul_f32_e32 v216, v214, v152
	v_dual_mul_f32 v121, v121, v8 :: v_dual_mul_f32 v114, v114, v8
	v_dual_mul_f32 v119, v119, v8 :: v_dual_mul_f32 v112, v112, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v218, -v145, v216, v214
	v_dual_mul_f32 v117, v117, v8 :: v_dual_mul_f32 v110, v110, v8
	v_dual_mul_f32 v115, v115, v8 :: v_dual_mul_f32 v108, v108, v8
	v_fmac_f32_e32 v216, v218, v152
	v_mul_f32_e32 v218, v217, v215
	v_dual_mul_f32 v113, v113, v8 :: v_dual_mul_f32 v106, v106, v8
	v_dual_mul_f32 v111, v111, v8 :: v_dual_mul_f32 v104, v104, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v145, -v145, v216, v214
	v_fma_f32 v214, -v153, v218, v217
	v_mul_f32_e32 v21, v21, v8
	v_dual_mul_f32 v109, v109, v8 :: v_dual_mul_f32 v102, v102, v8
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v145, v145, v152, v216
	v_fmac_f32_e32 v218, v214, v215
	v_div_scale_f32 v214, s4, v142, v4, v142
	v_rcp_f32_e32 v216, v221
	s_mov_b32 vcc_lo, s3
	v_div_fixup_f32 v144, v145, v4, v144
	v_fma_f32 v145, -v153, v218, v217
	v_mul_f32_e32 v217, v214, v220
	v_mov_b16_e64 v153.l, v2.l
	v_div_scale_f32 v222, s3, v141, v4, v141
	v_mov_b16_e64 v153.h, 0
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v145, v145, v215, v218
	v_fma_f32 v152, -v221, v216, 1.0
	v_fma_f32 v218, -v219, v217, v214
	v_div_scale_f32 v215, null, v4, v4, v140
	s_mov_b32 vcc_lo, s4
	v_div_fixup_f32 v143, v145, v4, v143
	v_fmac_f32_e32 v216, v152, v216
	v_mov_b16_e64 v152.l, v153.l
	v_fmac_f32_e32 v217, v218, v220
	v_rcp_f32_e32 v145, v215
	v_mov_b16_e64 v152.h, v153.h
	v_mul_f32_e32 v218, v222, v216
	v_cvt_pk_fp8_f32 v152.l, v144, v143
	v_fma_f32 v143, -v219, v217, v214
	v_rcp_f32_e32 v219, v223
	v_dual_mul_f32 v107, v107, v8 :: v_dual_mul_f32 v100, v100, v8
	v_fma_f32 v144, -v221, v218, v222
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v143, v143, v220, v217
	v_fma_f32 v214, -v215, v145, 1.0
	s_mov_b32 vcc_lo, s3
	v_dual_mul_f32 v105, v105, v8 :: v_dual_mul_f32 v98, v98, v8
	v_fmac_f32_e32 v218, v144, v216
	v_div_fixup_f32 v142, v143, v4, v142
	v_div_scale_f32 v143, null, v4, v4, v138
	v_fmac_f32_e32 v145, v214, v145
	v_rcp_f32_e32 v214, v224
	v_div_scale_f32 v144, s4, v140, v4, v140
	v_fma_f32 v217, -v223, v219, 1.0
	v_fma_f32 v220, -v221, v218, v222
	v_rcp_f32_e32 v222, v143
	v_mul_f32_e32 v13, v13, v8
	v_mul_f32_e32 v221, v144, v145
	v_fmac_f32_e32 v219, v217, v219
	v_div_scale_f32 v217, s5, v139, v4, v139
	v_fma_f32 v225, -v224, v214, 1.0
	v_mul_f32_e32 v15, v15, v8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v216, v220, v216, v218
	v_fma_f32 v218, -v215, v221, v144
	v_mul_f32_e32 v220, v217, v219
	v_fma_f32 v226, -v143, v222, 1.0
	v_fmac_f32_e32 v214, v225, v214
	v_div_scale_f32 v225, s3, v9, v4, v9
	v_div_fixup_f32 v141, v216, v4, v141
	v_fmac_f32_e32 v221, v218, v145
	v_fma_f32 v216, -v223, v220, v217
	v_fmac_f32_e32 v222, v226, v222
	v_div_scale_f32 v226, s6, v138, v4, v138
	v_mul_f32_e32 v218, v225, v214
	v_fma_f32 v144, -v215, v221, v144
	v_dual_fmac_f32 v220, v216, v219 :: v_dual_mul_f32 v11, v11, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v216, v226, v222
	v_fma_f32 v215, -v224, v218, v225
	s_mov_b32 vcc_lo, s4
	v_cvt_pk_fp8_f32 v152.h, v142, v141
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v145, v221
	v_fma_f32 v145, -v223, v220, v217
	v_fma_f32 v217, -v143, v216, v226
	v_fmac_f32_e32 v218, v215, v214
	s_mov_b32 vcc_lo, s5
	v_div_fixup_f32 v140, v144, v4, v140
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v145, v145, v219, v220
	v_fmac_f32_e32 v216, v217, v222
	v_fma_f32 v144, -v224, v218, v225
	s_mov_b32 vcc_lo, s3
	v_dual_mul_f32 v103, v103, v8 :: v_dual_mul_f32 v96, v96, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v143, -v143, v216, v226
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v214, v218
	s_mov_b32 vcc_lo, s6
	v_div_fixup_f32 v139, v145, v4, v139
	v_dual_mul_f32 v101, v101, v8 :: v_dual_mul_f32 v94, v94, v8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v143, v143, v222, v216
	v_div_fixup_f32 v9, v144, v4, v9
	v_cvt_pk_fp8_f32 v153.l, v140, v139
	v_dual_mul_f32 v99, v99, v8 :: v_dual_mul_f32 v92, v92, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v238, v143, v4, v138
	ds_load_b128 v[138:141], v234 offset:16384
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[142:145], v234 offset:16896
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[214:217], v234 offset:17408
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[218:221], v234 offset:17920
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[222:225], v234 offset:18432
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[226:229], v234 offset:18944
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[230:233], v234 offset:19456
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[234:237], v234 offset:19968
	v_dual_mul_f32 v97, v97, v8 :: v_dual_mul_f32 v90, v90, v8
	v_dual_mul_f32 v95, v95, v8 :: v_dual_mul_f32 v88, v88, v8
	v_dual_mul_f32 v93, v93, v8 :: v_dual_mul_f32 v86, v86, v8
	v_dual_mul_f32 v91, v91, v8 :: v_dual_mul_f32 v84, v84, v8
	v_dual_mul_f32 v89, v89, v8 :: v_dual_mul_f32 v82, v82, v8
	v_dual_mul_f32 v87, v87, v8 :: v_dual_mul_f32 v80, v80, v8
	v_dual_mul_f32 v85, v85, v8 :: v_dual_mul_f32 v78, v78, v8
	v_dual_mul_f32 v83, v83, v8 :: v_dual_mul_f32 v76, v76, v8
	v_dual_mul_f32 v81, v81, v8 :: v_dual_mul_f32 v74, v74, v8
	v_dual_mul_f32 v79, v79, v8 :: v_dual_mul_f32 v72, v72, v8
	v_dual_mul_f32 v77, v77, v8 :: v_dual_mul_f32 v70, v70, v8
	v_dual_mul_f32 v75, v75, v8 :: v_dual_mul_f32 v68, v68, v8
	v_dual_mul_f32 v73, v73, v8 :: v_dual_mul_f32 v66, v66, v8
	v_dual_mul_f32 v71, v71, v8 :: v_dual_mul_f32 v64, v64, v8
	v_dual_mul_f32 v69, v69, v8 :: v_dual_mul_f32 v62, v62, v8
	v_dual_mul_f32 v67, v67, v8 :: v_dual_mul_f32 v60, v60, v8
	v_dual_mul_f32 v65, v65, v8 :: v_dual_mul_f32 v58, v58, v8
	v_dual_mul_f32 v63, v63, v8 :: v_dual_mul_f32 v56, v56, v8
	v_dual_mul_f32 v61, v61, v8 :: v_dual_mul_f32 v54, v54, v8
	v_dual_mul_f32 v59, v59, v8 :: v_dual_mul_f32 v52, v52, v8
	v_dual_mul_f32 v57, v57, v8 :: v_dual_mul_f32 v50, v50, v8
	v_dual_mul_f32 v55, v55, v8 :: v_dual_mul_f32 v48, v48, v8
	v_dual_mul_f32 v53, v53, v8 :: v_dual_mul_f32 v46, v46, v8
	v_dual_mul_f32 v51, v51, v8 :: v_dual_mul_f32 v44, v44, v8
	v_dual_mul_f32 v49, v49, v8 :: v_dual_mul_f32 v42, v42, v8
	v_dual_mul_f32 v47, v47, v8 :: v_dual_mul_f32 v40, v40, v8
	v_dual_mul_f32 v45, v45, v8 :: v_dual_mul_f32 v38, v38, v8
	v_dual_mul_f32 v43, v43, v8 :: v_dual_mul_f32 v36, v36, v8
	v_dual_mul_f32 v41, v41, v8 :: v_dual_mul_f32 v34, v34, v8
	v_dual_mul_f32 v39, v39, v8 :: v_dual_mul_f32 v32, v32, v8
	v_dual_mul_f32 v37, v37, v8 :: v_dual_mul_f32 v30, v30, v8
	v_dual_mul_f32 v35, v35, v8 :: v_dual_mul_f32 v28, v28, v8
	v_dual_mul_f32 v33, v33, v8 :: v_dual_mul_f32 v26, v26, v8
	v_dual_mul_f32 v31, v31, v8 :: v_dual_mul_f32 v24, v24, v8
	v_dual_mul_f32 v29, v29, v8 :: v_dual_mul_f32 v22, v22, v8
	v_dual_mul_f32 v27, v27, v8 :: v_dual_mul_f32 v20, v20, v8
	v_dual_mul_f32 v25, v25, v8 :: v_dual_mul_f32 v18, v18, v8
	v_dual_mul_f32 v19, v19, v8 :: v_dual_mul_f32 v16, v16, v8
	v_dual_mul_f32 v17, v17, v8 :: v_dual_mul_f32 v14, v14, v8
	v_cvt_pk_fp8_f32 v153.h, v9, v238
	v_mul_f32_e32 v12, v12, v8
	v_mul_f32_e32 v10, v10, v8
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[130:137], v[138:139], v[152:153], v[130:137]
	v_wmma_f32_16x16x16_fp8_fp8 v[122:129], v[140:141], v[152:153], v[122:129]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[142:143], v[152:153], v[114:121]
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[144:145], v[152:153], v[106:113]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[214:215], v[152:153], v[98:105]
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[216:217], v[152:153], v[90:97]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[218:219], v[152:153], v[82:89]
	v_wmma_f32_16x16x16_fp8_fp8 v[74:81], v[220:221], v[152:153], v[74:81]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[66:73], v[222:223], v[152:153], v[66:73]
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[224:225], v[152:153], v[58:65]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[226:227], v[152:153], v[50:57]
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[228:229], v[152:153], v[42:49]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[230:231], v[152:153], v[34:41]
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[232:233], v[152:153], v[26:33]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[234:235], v[152:153], v[18:25]
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[236:237], v[152:153], v[10:17]
	v_dual_mov_b32 v153, v5 :: v_dual_mov_b32 v214, v4
	v_mov_b32_e32 v152, v3
	s_add_co_i32 s27, s27, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s27, 4
	s_cbranch_scc1 .LBB6_9
.LBB6_48:                               ;   Parent Loop BB6_11 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB6_53 Depth 3
	s_cmp_lt_i32 s27, 1
	s_mov_b32 s3, -1
	s_cbranch_scc1 .LBB6_51
; %bb.49:                               ;   in Loop: Header=BB6_48 Depth=2
	s_cmp_eq_u32 s27, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s3, s28
	s_cbranch_scc1 .LBB6_51
; %bb.50:                               ;   in Loop: Header=BB6_48 Depth=2
	s_cmp_eq_u32 s27, 2
	s_cselect_b32 s3, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s3, s21
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s24, s3
	s_cselect_b32 s3, -1, 0
.LBB6_51:                               ;   in Loop: Header=BB6_48 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_58
; %bb.52:                               ;   in Loop: Header=BB6_48 Depth=2
	v_mov_b32_e32 v138, 0
	v_lshl_add_u32 v3, s27, 9, v155
	s_mov_b32 s3, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v139, v138 :: v_dual_mov_b32 v140, v138
	v_dual_mov_b32 v141, v138 :: v_dual_mov_b32 v142, v138
	v_dual_mov_b32 v143, v138 :: v_dual_mov_b32 v144, v138
	v_mov_b32_e32 v145, v138
.LBB6_53:                               ;   Parent Loop BB6_11 Depth=1
                                        ;     Parent Loop BB6_48 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, s3, 6
	v_lshl_add_u32 v219, s3, 12, v3
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, v203, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v204, vcc_lo
	ds_load_b128 v[215:218], v219
	ds_load_b128 v[219:222], v219 offset:2048
	s_add_co_i32 s3, s3, 1
	s_clause 0x3
	global_load_b64 v[8:9], v[4:5], off
	global_load_b64 v[223:224], v[4:5], off offset:16
	global_load_b64 v[225:226], v[4:5], off offset:32
	global_load_b64 v[4:5], v[4:5], off offset:48
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s3, 4
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[215:216], v[8:9], v[138:145]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[217:218], v[223:224], v[138:145]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[219:220], v[225:226], v[138:145]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[221:222], v[4:5], v[138:145]
	s_cbranch_scc1 .LBB6_53
; %bb.54:                               ;   in Loop: Header=BB6_48 Depth=2
	v_lshl_add_u32 v8, s27, 5, v161
	s_lshl_b32 s3, s27, 4
	v_mov_b32_e32 v9, 0xff800000
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s3, s21
	ds_load_b96 v[3:5], v8 offset:49154
	ds_load_u16_d16 v215, v8 offset:49166
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s3, 15
	v_or_b32_e32 v216, s3, v157
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s25
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s26, s3
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB6_56
; %bb.55:                               ;   in Loop: Header=BB6_48 Depth=2
	ds_load_u16_d16 v9, v8 offset:49152
	v_cmp_le_i32_e32 vcc_lo, v216, v7
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s4, vcc_lo
	v_mul_f32_e32 v138, v6, v138
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v9, v138, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v9, 0xff800000, v9, vcc_lo
.LBB6_56:                               ;   in Loop: Header=BB6_48 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mul_f32_e32 v138, v6, v139
	v_or_b32_e32 v139, 2, v216
	v_cmp_lt_i32_e32 vcc_lo, v216, v7
	v_mul_f32_e32 v140, v6, v140
	s_wait_dscnt 0x1
	v_fma_mix_f32 v138, v3, v138, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e64 s3, v139, v7
	v_or_b32_e32 v139, 3, v216
	s_or_b32 s5, s4, vcc_lo
	v_fma_mix_f32 v3, v3, v140, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s2, s5
	s_or_b32 s3, s4, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v138, 0xff800000, v138, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v139, v7
	v_mul_f32_e32 v139, v6, v141
	v_or_b32_e32 v140, 4, v216
	s_and_b32 s3, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v141, 0xff800000, v3, s3
	v_fma_mix_f32 v3, v4, v139, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v139, v6, v142
	s_or_b32 s3, s4, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v140, v7
	v_or_b32_e32 v140, 5, v216
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v142, 0xff800000, v3, s3
	v_fma_mix_f32 v3, v4, v139, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v4, v6, v143
	s_or_b32 s3, s4, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v140, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s2, s3
	v_or_b32_e32 v140, 7, v216
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v139, 0xff800000, v3, s3
	v_fma_mix_f32 v3, v5, v4, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v4, 6, v216
	s_or_b32 s3, s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s2, s3
	v_cmp_le_i32_e64 s3, v140, v7
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v143, 0xff800000, v3, vcc_lo
	v_mul_f32_e32 v3, v6, v144
	v_cmp_le_i32_e32 vcc_lo, v4, v7
	v_mul_f32_e32 v4, v6, v145
	v_max3_num_f32 v140, v9, 0xff800000, v138
	s_or_b32 s3, s4, s3
	v_fma_mix_f32 v3, v5, v3, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_or_b32 s5, s4, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v4, v215, v4, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v5, v140, v141, v142
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s2, s5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v140, 0xff800000, v3, vcc_lo
	s_and_b32 vcc_lo, s2, s3
	v_max3_num_f32 v3, v5, v139, v143
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v4, 0xff800000, v4, vcc_lo
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v3, v3, v140, v4
	v_mov_b32_e32 v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v5, v5, s7, 0xfedcba98
	v_max3_num_f32 v3, v152, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v5, v9, v3
	v_dual_sub_f32 v9, v138, v3 :: v_dual_sub_f32 v4, v4, v3
	v_add_nc_u32_e32 v138, 0xc080, v8
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v3
	v_mul_f32_e32 v5, 0x3fb8aa3b, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v9, 0x3fb8aa3b, v9 :: v_dual_mul_f32 v4, 0x3fb8aa3b, v4
	v_dual_sub_f32 v143, v143, v3 :: v_dual_sub_f32 v140, v140, v3
	v_exp_f32_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v144, v4
	v_dual_sub_f32 v4, v141, v3 :: v_dual_sub_f32 v141, v142, v3
	v_exp_f32_e32 v142, v5
	v_mul_f32_e32 v140, 0x3fb8aa3b, v140
	v_mul_f32_e32 v145, 0x3fb8aa3b, v4
	ds_load_2addr_b32 v[4:5], v138 offset1:1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, 0, vcc_lo
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_exp_f32_e32 v140, v140
	v_cndmask_b32_e64 v142, v142, 0, vcc_lo
	v_add_nc_u32_e32 v8, 0xc088, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v141, v141
	ds_load_2addr_b32 v[215:216], v8 offset1:1
	v_sub_f32_e32 v138, v139, v3
	v_mul_f32_e32 v139, 0x3fb8aa3b, v143
	v_exp_f32_e32 v8, v145
	v_cndmask_b32_e64 v145, v144, 0, vcc_lo
	v_cndmask_b32_e64 v218, v141, 0, vcc_lo
	v_mul_f32_e32 v138, 0x3fb8aa3b, v138
	v_exp_f32_e32 v139, v139
	v_cndmask_b32_e64 v219, v140, 0, vcc_lo
	s_wait_dscnt 0x1
	v_fma_mix_f32 v144, v4, v142, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v143, v4, v9, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v138, v138
	v_cndmask_b32_e64 v8, v8, 0, vcc_lo
	v_add_f32_e32 v9, v142, v9
	v_fma_mix_f32 v141, v5, v218, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v217, v139, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_fma_mix_f32 v142, v5, v8, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v5, v144, 0, v143
	v_add_f32_e32 v8, v8, v9
	v_cndmask_b32_e64 v4, v138, 0, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v139, v215, v217, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v9, v216, v219, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v5, v5, v142, v141
	v_add_f32_e32 v8, v218, v8
	v_fma_mix_f32 v140, v215, v4, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v138, v216, v145, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v4, v4, v8
	v_max3_num_f32 v5, v5, v140, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v217, v4
	v_max3_num_f32 v215, v5, v9, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v219, v4
	v_dual_mov_b32 v8, v215 :: v_dual_add_f32 v5, v145, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v8, v8, s7, 0xfedcba98
	v_max_num_f32_e32 v4, v8, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v8, v5 :: v_dual_max_num_f32 v145, v215, v4
	v_permlanex16_b32 v8, v8, s7, 0xfedcba98
	v_mov_b32_e32 v4, v214
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_lt_f32_e32 0, v145
	s_cbranch_execz .LBB6_47
; %bb.57:                               ;   in Loop: Header=BB6_48 Depth=2
	v_div_scale_f32 v4, null, 0x43e00000, 0x43e00000, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v215, v4
	v_fma_f32 v216, -v4, v215, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v215, v216, v215
	v_div_scale_f32 v216, vcc_lo, v145, 0x43e00000, v145
	v_mul_f32_e32 v217, v216, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v218, -v4, v217, v216
	v_fmac_f32_e32 v217, v218, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v4, v217, v216
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v4, v4, v215, v217
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v4, v4, 0x43e00000, v145
	v_max_num_f32_e32 v4, 0x1f800000, v4
	s_branch .LBB6_47
.LBB6_58:                               ;   in Loop: Header=BB6_48 Depth=2
	v_mov_b32_e32 v3, v152
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v152, v3
	s_add_co_i32 s27, s27, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s27, 4
	s_cbranch_scc0 .LBB6_48
	s_branch .LBB6_9
.LBB6_59:
	v_cmp_eq_u32_e32 vcc_lo, 0, v154
	s_and_b32 s1, vcc_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_61
; %bb.60:
	v_mad_co_u64_u32 v[0:1], null, s17, v163, s[12:13]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v0, 0x102, v0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s10, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	global_store_b64 v[0:1], v[152:153], off
.LBB6_61:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB6_63
; %bb.62:
	v_mad_co_u64_u32 v[1:2], null, s17, v163, s[12:13]
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v3, 2, v157
	v_mul_f32_e32 v0, v214, v130
	v_dual_mul_f32 v4, v122, v214 :: v_dual_mul_f32 v5, v123, v214
	v_dual_mul_f32 v8, v114, v214 :: v_dual_mul_f32 v9, v115, v214
	v_mul_lo_u32 v1, 0x102, v1
	v_dual_mul_f32 v106, v106, v214 :: v_dual_mul_f32 v107, v107, v214
	v_dual_mul_f32 v98, v98, v214 :: v_dual_mul_f32 v99, v99, v214
	v_dual_mul_f32 v90, v90, v214 :: v_dual_mul_f32 v91, v91, v214
	v_dual_mul_f32 v82, v82, v214 :: v_dual_mul_f32 v83, v83, v214
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_dual_mul_f32 v74, v74, v214 :: v_dual_mul_f32 v75, v75, v214
	v_dual_mul_f32 v66, v66, v214 :: v_dual_mul_f32 v67, v67, v214
	v_dual_mul_f32 v58, v58, v214 :: v_dual_mul_f32 v59, v59, v214
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, s10, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s11, v2, vcc_lo
	v_dual_mul_f32 v50, v50, v214 :: v_dual_mul_f32 v51, v51, v214
	v_add_co_u32 v138, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v139, null, 0, v2, vcc_lo
	v_dual_mul_f32 v1, v214, v131 :: v_dual_mul_f32 v2, v214, v132
	v_mul_f32_e32 v3, v214, v133
	v_dual_mul_f32 v42, v42, v214 :: v_dual_mul_f32 v43, v43, v214
	v_dual_mul_f32 v34, v34, v214 :: v_dual_mul_f32 v35, v35, v214
	v_dual_mul_f32 v26, v26, v214 :: v_dual_mul_f32 v27, v27, v214
	v_dual_mul_f32 v114, v10, v214 :: v_dual_mul_f32 v115, v11, v214
	v_dual_mul_f32 v6, v124, v214 :: v_dual_mul_f32 v131, v214, v135
	v_dual_mul_f32 v10, v116, v214 :: v_dual_mul_f32 v133, v214, v137
	v_dual_mul_f32 v108, v108, v214 :: v_dual_mul_f32 v7, v125, v214
	v_dual_mul_f32 v100, v100, v214 :: v_dual_mul_f32 v11, v117, v214
	v_dual_mul_f32 v92, v92, v214 :: v_dual_mul_f32 v109, v109, v214
	v_dual_mul_f32 v84, v84, v214 :: v_dual_mul_f32 v101, v101, v214
	v_dual_mul_f32 v76, v76, v214 :: v_dual_mul_f32 v93, v93, v214
	v_dual_mul_f32 v68, v68, v214 :: v_dual_mul_f32 v85, v85, v214
	v_dual_mul_f32 v60, v60, v214 :: v_dual_mul_f32 v77, v77, v214
	v_dual_mul_f32 v52, v52, v214 :: v_dual_mul_f32 v69, v69, v214
	v_dual_mul_f32 v44, v44, v214 :: v_dual_mul_f32 v61, v61, v214
	v_dual_mul_f32 v36, v36, v214 :: v_dual_mul_f32 v53, v53, v214
	v_dual_mul_f32 v28, v28, v214 :: v_dual_mul_f32 v45, v45, v214
	v_dual_mul_f32 v20, v20, v214 :: v_dual_mul_f32 v37, v37, v214
	v_dual_mul_f32 v116, v12, v214 :: v_dual_mul_f32 v29, v29, v214
	v_mul_f32_e32 v130, v214, v134
	v_dual_mul_f32 v132, v214, v136 :: v_dual_mul_f32 v21, v21, v214
	global_store_b128 v[138:139], v[0:3], off offset:8
	v_dual_mul_f32 v117, v13, v214 :: v_dual_mul_f32 v0, v126, v214
	v_dual_mul_f32 v12, v118, v214 :: v_dual_mul_f32 v1, v127, v214
	v_dual_mul_f32 v110, v110, v214 :: v_dual_mul_f32 v3, v129, v214
	v_dual_mul_f32 v102, v102, v214 :: v_dual_mul_f32 v13, v119, v214
	v_dual_mul_f32 v94, v94, v214 :: v_dual_mul_f32 v111, v111, v214
	v_dual_mul_f32 v86, v86, v214 :: v_dual_mul_f32 v103, v103, v214
	v_dual_mul_f32 v78, v78, v214 :: v_dual_mul_f32 v95, v95, v214
	v_dual_mul_f32 v70, v70, v214 :: v_dual_mul_f32 v87, v87, v214
	v_dual_mul_f32 v62, v62, v214 :: v_dual_mul_f32 v79, v79, v214
	v_dual_mul_f32 v54, v54, v214 :: v_dual_mul_f32 v71, v71, v214
	v_dual_mul_f32 v46, v46, v214 :: v_dual_mul_f32 v63, v63, v214
	v_dual_mul_f32 v38, v38, v214 :: v_dual_mul_f32 v55, v55, v214
	v_dual_mul_f32 v2, v128, v214 :: v_dual_mul_f32 v47, v47, v214
	v_dual_mul_f32 v30, v30, v214 :: v_dual_mul_f32 v39, v39, v214
	v_dual_mul_f32 v22, v22, v214 :: v_dual_mul_f32 v31, v31, v214
	v_dual_mul_f32 v118, v14, v214 :: v_dual_mul_f32 v23, v23, v214
	v_dual_mul_f32 v119, v15, v214 :: v_dual_mul_f32 v14, v120, v214
	v_dual_mul_f32 v112, v112, v214 :: v_dual_mul_f32 v15, v121, v214
	v_dual_mul_f32 v104, v104, v214 :: v_dual_mul_f32 v113, v113, v214
	v_dual_mul_f32 v96, v96, v214 :: v_dual_mul_f32 v105, v105, v214
	v_dual_mul_f32 v88, v88, v214 :: v_dual_mul_f32 v97, v97, v214
	v_dual_mul_f32 v80, v80, v214 :: v_dual_mul_f32 v89, v89, v214
	v_dual_mul_f32 v72, v72, v214 :: v_dual_mul_f32 v81, v81, v214
	v_dual_mul_f32 v64, v64, v214 :: v_dual_mul_f32 v73, v73, v214
	v_dual_mul_f32 v56, v56, v214 :: v_dual_mul_f32 v65, v65, v214
	v_dual_mul_f32 v48, v48, v214 :: v_dual_mul_f32 v57, v57, v214
	v_dual_mul_f32 v40, v40, v214 :: v_dual_mul_f32 v49, v49, v214
	v_dual_mul_f32 v32, v32, v214 :: v_dual_mul_f32 v41, v41, v214
	v_dual_mul_f32 v24, v24, v214 :: v_dual_mul_f32 v33, v33, v214
	v_dual_mul_f32 v18, v18, v214 :: v_dual_mul_f32 v19, v19, v214
	v_dual_mul_f32 v120, v16, v214 :: v_dual_mul_f32 v25, v25, v214
	s_clause 0x18
	global_store_b128 v[138:139], v[130:133], off offset:24
	global_store_b128 v[138:139], v[4:7], off offset:72
	global_store_b128 v[138:139], v[0:3], off offset:88
	global_store_b128 v[138:139], v[8:11], off offset:136
	global_store_b128 v[138:139], v[12:15], off offset:152
	global_store_b128 v[138:139], v[106:109], off offset:200
	global_store_b128 v[138:139], v[110:113], off offset:216
	global_store_b128 v[138:139], v[98:101], off offset:264
	global_store_b128 v[138:139], v[102:105], off offset:280
	global_store_b128 v[138:139], v[90:93], off offset:328
	global_store_b128 v[138:139], v[94:97], off offset:344
	global_store_b128 v[138:139], v[82:85], off offset:392
	global_store_b128 v[138:139], v[86:89], off offset:408
	global_store_b128 v[138:139], v[74:77], off offset:456
	global_store_b128 v[138:139], v[78:81], off offset:472
	global_store_b128 v[138:139], v[66:69], off offset:520
	global_store_b128 v[138:139], v[70:73], off offset:536
	global_store_b128 v[138:139], v[58:61], off offset:584
	global_store_b128 v[138:139], v[62:65], off offset:600
	global_store_b128 v[138:139], v[50:53], off offset:648
	global_store_b128 v[138:139], v[54:57], off offset:664
	global_store_b128 v[138:139], v[42:45], off offset:712
	global_store_b128 v[138:139], v[46:49], off offset:728
	global_store_b128 v[138:139], v[34:37], off offset:776
	global_store_b128 v[138:139], v[38:41], off offset:792
	v_mul_f32_e32 v121, v17, v214
	s_clause 0x5
	global_store_b128 v[138:139], v[26:29], off offset:840
	global_store_b128 v[138:139], v[30:33], off offset:856
	global_store_b128 v[138:139], v[18:21], off offset:904
	global_store_b128 v[138:139], v[22:25], off offset:920
	global_store_b128 v[138:139], v[114:117], off offset:968
	global_store_b128 v[138:139], v[118:121], off offset:984
.LBB6_63:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end6:
	.size	attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201, .Lfunc_end6-attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
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
		.amdhsa_next_free_sgpr 29
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_vgpr, 239
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.numbered_sgpr, 29
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9952
; TotalNumSgprs: 31
; NumVgprs: 239
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 31
; NumVGPRsForWavesPerEU: 239
; Occupancy: 6
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
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
	s_cbranch_vccnz .LBB7_10
; %bb.1:
	v_lshrrev_b32_e32 v1, 5, v0
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v4, ttmp9, 3, v1
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB7_10
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
.LBB7_3:                                ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_scc0 .LBB7_3
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
	s_branch .LBB7_6
.LBB7_5:                                ;   in Loop: Header=BB7_6 Depth=1
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
	s_cbranch_execz .LBB7_10
.LBB7_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB7_8 Depth 2
	v_dual_mov_b32 v6, v2 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	s_mov_b32 s0, s7
	s_branch .LBB7_8
.LBB7_7:                                ;   in Loop: Header=BB7_8 Depth=2
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
	s_cbranch_scc1 .LBB7_5
.LBB7_8:                                ;   Parent Loop BB7_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	s_mov_b32 s2, exec_lo
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	s_cbranch_execz .LBB7_7
; %bb.9:                                ;   in Loop: Header=BB7_8 Depth=2
	global_load_b32 v14, v[5:6], off
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	v_exp_f32_e32 v14, v14
	s_branch .LBB7_7
.LBB7_10:
	s_endpgm
.Lfunc_end7:
	.size	attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201, .Lfunc_end7-attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end7-attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.num_vgpr, 17
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.numbered_sgpr, 8
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.has_indirect_call, 0
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
	.type	__hip_cuid_851bb07fbc4375f,@object ; @__hip_cuid_851bb07fbc4375f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_851bb07fbc4375f
__hip_cuid_851bb07fbc4375f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_851bb07fbc4375f, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_851bb07fbc4375f
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
    .name:           attention_fp8_e4m3_fa2_gqa_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     190
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
    .name:           attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.kd
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     56
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
    .name:           attention_fp8_e4m3_fa2_gqa_partial_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     37
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_partial_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     191
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
    .name:           attention_fp8_e4m3_fa2_gqa_merge_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_merge_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     17
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 60
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           attention_fp8_e4m3_fa2_gqa_packet_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     29
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_gfx1201.kd
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
    .max_flat_workgroup_size: 256
    .name:           attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.kd
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
    .name:           attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.kd
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
