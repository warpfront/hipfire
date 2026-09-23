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
	s_load_b128 s[4:7], s[0:1], 0x30
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
	s_cbranch_vccnz .LBB5_62
; %bb.1:
	s_and_b32 s25, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s25, 3
	s_cbranch_scc1 .LBB5_62
; %bb.2:
	s_load_b32 s24, s[0:1], 0x40
	s_lshr_b32 s2, ttmp7, 7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, 0x1fffe00
	s_cmp_gt_i32 s7, 0x200
	s_cselect_b32 s22, s2, 0
	s_cmp_le_i32 s7, s22
	s_cbranch_scc1 .LBB5_62
; %bb.3:
	s_sub_co_i32 s2, s7, s22
	s_lshl_b32 s20, ttmp9, 7
	s_min_i32 s23, s2, 0x200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s21, s23, 6
	s_cmp_ge_i32 s20, s21
	s_cbranch_scc1 .LBB5_62
; %bb.4:
	v_lshrrev_b32_e32 v7, 5, v0
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v154, s24 :: v_dual_and_b32 v9, 15, v0
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b128 s[16:19], s[0:1], 0x20
	v_lshlrev_b32_e32 v10, 4, v7
	v_bfe_u32 v153, v0, 4, 1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v1, v10, v9
	v_add_nc_u32_e32 v3, s20, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_mul_hi_i32 v1, 0x2aaaaaab, v3
	v_cmp_le_i32_e64 s1, s21, v3
	v_cmp_gt_i32_e64 s0, s21, v3
	v_lshrrev_b32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v2, v1, v2
	v_mul_lo_u32 v1, v2, 6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v3, v1
	v_mad_co_u64_u32 v[146:147], null, s25, 6, v[1:2]
	v_add_nc_u32_e32 v147, s22, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v147, 24, v[146:147]
	v_lshlrev_b32_e32 v8, 8, v1
	s_and_saveexec_b32 s26, s0
	s_cbranch_execz .LBB5_8
; %bb.5:
	v_mov_b32_e32 v2, 0
	s_mov_b32 s2, 0x76543210
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[135:136], 10, v[1:2]
	v_lshlrev_b32_e32 v1, 5, v153
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, vcc_lo, s8, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v136, vcc_lo
	v_add_co_u32 v131, vcc_lo, v3, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v132, null, 0, v4, vcc_lo
	s_clause 0x1f
	global_load_b128 v[3:6], v[131:132], off
	global_load_b128 v[11:14], v[131:132], off offset:16
	global_load_b128 v[15:18], v[131:132], off offset:64
	global_load_b128 v[19:22], v[131:132], off offset:80
	global_load_b128 v[23:26], v[131:132], off offset:128
	global_load_b128 v[27:30], v[131:132], off offset:144
	global_load_b128 v[31:34], v[131:132], off offset:192
	global_load_b128 v[35:38], v[131:132], off offset:208
	global_load_b128 v[39:42], v[131:132], off offset:256
	global_load_b128 v[43:46], v[131:132], off offset:272
	global_load_b128 v[47:50], v[131:132], off offset:320
	global_load_b128 v[51:54], v[131:132], off offset:336
	global_load_b128 v[55:58], v[131:132], off offset:384
	global_load_b128 v[59:62], v[131:132], off offset:400
	global_load_b128 v[63:66], v[131:132], off offset:448
	global_load_b128 v[67:70], v[131:132], off offset:464
	global_load_b128 v[71:74], v[131:132], off offset:512
	global_load_b128 v[75:78], v[131:132], off offset:528
	global_load_b128 v[79:82], v[131:132], off offset:576
	global_load_b128 v[83:86], v[131:132], off offset:592
	global_load_b128 v[87:90], v[131:132], off offset:640
	global_load_b128 v[91:94], v[131:132], off offset:656
	global_load_b128 v[95:98], v[131:132], off offset:704
	global_load_b128 v[99:102], v[131:132], off offset:720
	global_load_b128 v[103:106], v[131:132], off offset:768
	global_load_b128 v[107:110], v[131:132], off offset:784
	global_load_b128 v[111:114], v[131:132], off offset:832
	global_load_b128 v[115:118], v[131:132], off offset:848
	global_load_b128 v[119:122], v[131:132], off offset:896
	global_load_b128 v[123:126], v[131:132], off offset:912
	global_load_b128 v[127:130], v[131:132], off offset:960
	global_load_b128 v[131:134], v[131:132], off offset:976
	s_wait_loadcnt 0x1f
	v_max3_num_f32 v1, |v3|, 0, |v4|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v5|, |v6|
	s_wait_loadcnt 0x1e
	v_max3_num_f32 v1, v1, |v11|, |v12|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v13|, |v14|
	s_wait_loadcnt 0x1d
	v_max3_num_f32 v1, v1, |v15|, |v16|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v17|, |v18|
	s_wait_loadcnt 0x1c
	v_max3_num_f32 v1, v1, |v19|, |v20|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v21|, |v22|
	s_wait_loadcnt 0x1b
	v_max3_num_f32 v1, v1, |v23|, |v24|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v25|, |v26|
	s_wait_loadcnt 0x1a
	v_max3_num_f32 v1, v1, |v27|, |v28|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v29|, |v30|
	s_wait_loadcnt 0x19
	v_max3_num_f32 v1, v1, |v31|, |v32|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v33|, |v34|
	s_wait_loadcnt 0x18
	v_max3_num_f32 v1, v1, |v35|, |v36|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v37|, |v38|
	s_wait_loadcnt 0x17
	v_max3_num_f32 v1, v1, |v39|, |v40|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v41|, |v42|
	s_wait_loadcnt 0x16
	v_max3_num_f32 v1, v1, |v43|, |v44|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v45|, |v46|
	s_wait_loadcnt 0x15
	v_max3_num_f32 v1, v1, |v47|, |v48|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v49|, |v50|
	s_wait_loadcnt 0x14
	v_max3_num_f32 v1, v1, |v51|, |v52|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v53|, |v54|
	s_wait_loadcnt 0x13
	v_max3_num_f32 v1, v1, |v55|, |v56|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v57|, |v58|
	s_wait_loadcnt 0x12
	v_max3_num_f32 v1, v1, |v59|, |v60|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v61|, |v62|
	s_wait_loadcnt 0x11
	v_max3_num_f32 v1, v1, |v63|, |v64|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v65|, |v66|
	s_wait_loadcnt 0x10
	v_max3_num_f32 v1, v1, |v67|, |v68|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v69|, |v70|
	s_wait_loadcnt 0xf
	v_max3_num_f32 v1, v1, |v71|, |v72|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v73|, |v74|
	s_wait_loadcnt 0xe
	v_max3_num_f32 v1, v1, |v75|, |v76|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v77|, |v78|
	s_wait_loadcnt 0xd
	v_max3_num_f32 v1, v1, |v79|, |v80|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v81|, |v82|
	s_wait_loadcnt 0xc
	v_max3_num_f32 v1, v1, |v83|, |v84|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v85|, |v86|
	s_wait_loadcnt 0xb
	v_max3_num_f32 v1, v1, |v87|, |v88|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v89|, |v90|
	s_wait_loadcnt 0xa
	v_max3_num_f32 v1, v1, |v91|, |v92|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v93|, |v94|
	s_wait_loadcnt 0x9
	v_max3_num_f32 v1, v1, |v95|, |v96|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v97|, |v98|
	s_wait_loadcnt 0x8
	v_max3_num_f32 v1, v1, |v99|, |v100|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v101|, |v102|
	s_wait_loadcnt 0x7
	v_max3_num_f32 v1, v1, |v103|, |v104|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v105|, |v106|
	s_wait_loadcnt 0x6
	v_max3_num_f32 v1, v1, |v107|, |v108|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v109|, |v110|
	s_wait_loadcnt 0x5
	v_max3_num_f32 v1, v1, |v111|, |v112|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v113|, |v114|
	s_wait_loadcnt 0x4
	v_max3_num_f32 v1, v1, |v115|, |v116|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v117|, |v118|
	s_wait_loadcnt 0x3
	v_max3_num_f32 v1, v1, |v119|, |v120|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v121|, |v122|
	s_wait_loadcnt 0x2
	v_max3_num_f32 v1, v1, |v123|, |v124|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v125|, |v126|
	s_wait_loadcnt 0x1
	v_max3_num_f32 v1, v1, |v127|, |v128|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v129|, |v130|
	s_wait_loadcnt 0x0
	v_max3_num_f32 v1, v1, |v131|, |v132|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v133|, |v134|
	v_mov_b32_e32 v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v3, v3, s2, 0xfedcba98
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v1, v1, v3
	v_div_scale_f32 v3, null, 0x43e00000, 0x43e00000, v1
	v_div_scale_f32 v6, vcc_lo, v1, 0x43e00000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v4, v3
	v_fma_f32 v5, -v3, v4, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v5, v4
	v_mul_f32_e32 v5, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v11, -v3, v5, v6
	v_fmac_f32_e32 v5, v11, v4
	v_lshl_or_b32 v11, v153, 5, v135
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f32 v3, -v3, v5, v6
	v_lshl_or_b32 v6, v153, 3, v8
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v5, s2, s10, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s11, 0, s2
	v_div_fixup_f32 v12, v3, 0x43e00000, v1
	v_add_co_u32 v3, vcc_lo, s8, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s9, v136, vcc_lo
	v_cmp_neq_f32_e32 vcc_lo, 0, v1
	s_mov_b32 s9, 16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 1.0, v12, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, 4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
.LBB5_6:                                ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[11:14], v[3:4], off
	global_load_b128 v[15:18], v[3:4], off offset:16
	v_add_co_u32 v3, vcc_lo, v3, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	v_mov_b16_e32 v20.l, v2.l
	v_mov_b16_e32 v20.h, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s9, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s9, 0
	v_mov_b16_e32 v19.l, v20.l
	v_mov_b16_e32 v19.h, v20.h
	s_wait_loadcnt 0x1
	v_div_scale_f32 v21, null, v1, v1, v11
	v_div_scale_f32 v23, null, v1, v1, v12
	v_div_scale_f32 v25, null, v1, v1, v13
	v_div_scale_f32 v27, null, v1, v1, v14
	s_wait_loadcnt 0x0
	v_div_scale_f32 v29, null, v1, v1, v15
	v_rcp_f32_e32 v37, v21
	v_rcp_f32_e32 v38, v23
	v_rcp_f32_e32 v39, v25
	v_div_scale_f32 v31, null, v1, v1, v16
	v_rcp_f32_e32 v40, v27
	v_div_scale_f32 v33, null, v1, v1, v17
	v_rcp_f32_e32 v41, v29
	v_div_scale_f32 v35, null, v1, v1, v18
	v_rcp_f32_e32 v42, v31
	v_fma_f32 v45, -v21, v37, 1.0
	v_fma_f32 v46, -v23, v38, 1.0
	v_rcp_f32_e32 v43, v33
	v_fma_f32 v47, -v25, v39, 1.0
	v_div_scale_f32 v22, vcc_lo, v11, v1, v11
	v_rcp_f32_e32 v44, v35
	v_dual_fmac_f32 v37, v45, v37 :: v_dual_fmac_f32 v38, v46, v38
	v_fma_f32 v48, -v27, v40, 1.0
	v_div_scale_f32 v24, s2, v12, v1, v12
	v_fma_f32 v49, -v29, v41, 1.0
	v_div_scale_f32 v26, s3, v13, v1, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v39, v47, v39 :: v_dual_fmac_f32 v40, v48, v40
	v_fma_f32 v50, -v31, v42, 1.0
	v_dual_mul_f32 v45, v22, v37 :: v_dual_mul_f32 v46, v24, v38
	v_div_scale_f32 v28, s4, v14, v1, v14
	v_fma_f32 v51, -v33, v43, 1.0
	v_div_scale_f32 v30, s5, v15, v1, v15
	v_dual_fmac_f32 v41, v49, v41 :: v_dual_fmac_f32 v42, v50, v42
	v_fma_f32 v52, -v35, v44, 1.0
	v_dual_mul_f32 v47, v26, v39 :: v_dual_mul_f32 v48, v28, v40
	v_div_scale_f32 v32, s6, v16, v1, v16
	v_fma_f32 v53, -v21, v45, v22
	v_div_scale_f32 v34, s7, v17, v1, v17
	v_dual_fmac_f32 v43, v51, v43 :: v_dual_fmac_f32 v44, v52, v44
	v_fma_f32 v54, -v23, v46, v24
	v_dual_mul_f32 v49, v30, v41 :: v_dual_mul_f32 v50, v32, v42
	v_div_scale_f32 v36, s8, v18, v1, v18
	v_fma_f32 v55, -v25, v47, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v45, v53, v37 :: v_dual_fmac_f32 v46, v54, v38
	v_fma_f32 v56, -v27, v48, v28
	v_dual_mul_f32 v51, v34, v43 :: v_dual_mul_f32 v52, v36, v44
	v_fma_f32 v57, -v29, v49, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v47, v55, v39 :: v_dual_fmac_f32 v48, v56, v40
	v_fma_f32 v58, -v31, v50, v32
	v_fma_f32 v21, -v21, v45, v22
	v_fma_f32 v59, -v33, v51, v34
	v_fma_f32 v22, -v23, v46, v24
	v_dual_fmac_f32 v49, v57, v41 :: v_dual_fmac_f32 v50, v58, v42
	v_fma_f32 v60, -v35, v52, v36
	v_fma_f32 v23, -v25, v47, v26
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v21, v21, v37, v45
	s_mov_b32 vcc_lo, s2
	v_fma_f32 v24, -v27, v48, v28
	v_dual_fmac_f32 v51, v59, v43 :: v_dual_fmac_f32 v52, v60, v44
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v22, v22, v38, v46
	s_mov_b32 vcc_lo, s3
	v_fma_f32 v25, -v29, v49, v30
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v23, v23, v39, v47
	s_mov_b32 vcc_lo, s4
	v_fma_f32 v26, -v31, v50, v32
	v_div_fixup_f32 v11, v21, v1, v11
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v21, v24, v40, v48
	s_mov_b32 vcc_lo, s5
	v_fma_f32 v27, -v33, v51, v34
	v_div_fixup_f32 v12, v22, v1, v12
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v22, v25, v41, v49
	s_mov_b32 vcc_lo, s6
	v_fma_f32 v28, -v35, v52, v36
	v_div_fixup_f32 v13, v23, v1, v13
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v23, v26, v42, v50
	s_mov_b32 vcc_lo, s7
	v_cvt_pk_fp8_f32 v19.l, v11, v12
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v11, v27, v43, v51
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v14, v21, v1, v14
	v_div_fixup_f32 v12, v22, v1, v15
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v15, v28, v44, v52
	v_div_fixup_f32 v16, v23, v1, v16
	v_div_fixup_f32 v11, v11, v1, v17
	v_cvt_pk_fp8_f32 v19.h, v13, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v13, v15, v1, v18
	v_cvt_pk_fp8_f32 v20.l, v12, v16
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_fp8_f32 v20.h, v11, v13
	global_store_b64 v[5:6], v[19:20], off offset:-4
	v_add_co_u32 v5, vcc_lo, v5, 16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	s_cbranch_scc0 .LBB5_6
; %bb.7:
	v_mul_f32_e32 v154, s24, v1
.LBB5_8:
	s_or_b32 exec_lo, exec_lo, s26
	v_dual_mov_b32 v2, -1 :: v_dual_and_b32 v3, 31, v0
	v_bfrev_b32_e32 v4, -2
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_u32_e32 22, v3
	s_cbranch_execz .LBB5_12
; %bb.9:
	v_or_b32_e32 v1, s22, v3
	s_mul_hi_i32 s3, s20, 0x2aaaaaab
	v_bfrev_b32_e32 v4, -2
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s4, s3, 31
	v_mov_b32_e32 v2, -1
	v_add3_u32 v1, s3, s4, v1
	s_add_co_i32 s23, s23, s22
	s_mov_b32 s3, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s23, v1
	s_cbranch_execz .LBB5_11
; %bb.10:
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, vcc_lo, s18, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s19, v2, vcc_lo
	global_load_b32 v2, v[1:2], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v4, v2
.LBB5_11:
	s_or_b32 exec_lo, exec_lo, s3
.LBB5_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mbcnt_lo_u32_b32 v1, -1, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s9, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v5, 16, v1
	v_xor_b32_e32 v11, 8, v1
	v_cmp_gt_u32_e32 vcc_lo, 32, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v1, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	v_lshlrev_b32_e32 v5, 2, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v1, v11, vcc_lo
	ds_bpermute_b32 v6, v5, v2
	ds_bpermute_b32 v5, v5, v4
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v2, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v4, v5
	ds_bpermute_b32 v5, v11, v2
	ds_bpermute_b32 v6, v11, v4
	v_xor_b32_e32 v11, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v1, v11, vcc_lo
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v2, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v4, v6
	ds_bpermute_b32 v5, v11, v2
	ds_bpermute_b32 v6, v11, v4
	v_xor_b32_e32 v11, 2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v1, v11, vcc_lo
	v_lshlrev_b32_e32 v155, 2, v11
	v_xor_b32_e32 v11, 1, v1
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v2, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v4, v6
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	ds_bpermute_b32 v5, v155, v2
	ds_bpermute_b32 v6, v155, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v156, 2, v1
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v2, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v1, v4, v6
	ds_bpermute_b32 v4, v156, v2
	ds_bpermute_b32 v5, v156, v1
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v2, v4
	s_wait_dscnt 0x0
	v_min_i32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s22, v2
	v_readfirstlane_b32 s2, v1
	s_cmp_lt_i32 s22, 0
	s_cbranch_scc1 .LBB5_59
; %bb.13:
	v_lshrrev_b32_e32 v4, 6, v0
	v_bfe_u32 v1, v0, 5, 1
	v_dual_mov_b32 v12, 0x6020400 :: v_dual_and_b32 v11, 3, v0
	v_and_or_b32 v157, v10, 48, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v65, 0 :: v_dual_lshlrev_b32 v2, 4, v4
	v_dual_mov_b32 v14, 0x5040100 :: v_dual_lshlrev_b32 v13, 3, v11
	v_ashrrev_i32_e32 v148, 31, v147
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v158, v1, 3, v2
	v_dual_mov_b32 v69, v65 :: v_dual_and_b32 v2, 1, v0
	s_addk_co_i32 s20, 0x7f
	v_lshrrev_b32_e32 v9, 2, v9
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s20, s21
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	v_lshlrev_b32_e32 v10, 3, v3
	s_cselect_b32 s23, -1, 0
	s_lshl_b32 s8, s25, 8
	v_or_b32_e32 v16, 0x300, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v159, 0x3070105, v12, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 2, v11
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[14:15], s[8:9]
	v_dual_mov_b32 v67, v65 :: v_dual_and_b32 v6, 0x7f, v0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v149, s3, s4, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v160, 0x3020706, v14, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, v153, v1
	v_lshlrev_b64_e32 v[1:2], 2, v[147:148]
	v_or_b32_e32 v14, 0x200, v0
	v_lshl_or_b32 v9, v11, 6, v9
	v_lshrrev_b32_e32 v18, 5, v16
	v_lshl_add_u32 v5, v7, 11, 0
	v_dual_mov_b32 v71, v65 :: v_dual_lshlrev_b32 v162, 3, v153
	v_add_co_u32 v151, s4, s18, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v152, null, s19, v2, s4
	v_and_b32_e32 v1, 16, v0
	v_or_b32_e32 v2, 0x100, v0
	v_lshrrev_b32_e32 v17, 5, v14
	v_dual_mov_b32 v68, v65 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v66, v65 :: v_dual_add_nc_u32 v161, 0, v1
	v_lshrrev_b32_e32 v1, 5, v2
	v_and_or_b32 v2, 0x180, v2, v6
	v_and_or_b32 v14, 0x280, v14, v6
	v_and_or_b32 v6, 0x380, v16, v6
	v_and_or_b32 v16, v17, 20, v153
	v_and_or_b32 v17, v18, 28, v153
	v_dual_mov_b32 v70, v65 :: v_dual_lshlrev_b32 v11, 5, v11
	v_dual_mov_b32 v72, v65 :: v_dual_lshlrev_b32 v3, 4, v3
	v_lshlrev_b32_e32 v18, 2, v9
	v_dual_mov_b32 v214, 0xff800000 :: v_dual_add_nc_u32 v163, v5, v10
	v_xad_u32 v164, 0x120, v10, v5
	v_xad_u32 v165, 0x124, v10, v5
	v_xad_u32 v166, 0x240, v10, v5
	v_xad_u32 v167, 0x244, v10, v5
	v_xad_u32 v168, 0x360, v10, v5
	v_xad_u32 v169, 0x364, v10, v5
	v_xad_u32 v170, 0x520, v10, v5
	v_xad_u32 v171, 0x524, v10, v5
	v_xad_u32 v172, 0x640, v10, v5
	v_xad_u32 v173, 0x644, v10, v5
	v_xad_u32 v174, 0x760, v10, v5
	v_xad_u32 v175, 0x764, v10, v5
	v_add3_u32 v176, v5, v18, v11
	v_or_b32_e32 v10, 8, v9
	v_or_b32_e32 v11, 0x108, v9
	v_or_b32_e32 v18, 12, v9
	v_or_b32_e32 v19, 0x10c, v9
	v_or_b32_e32 v20, 24, v9
	v_xor_b32_e32 v10, v10, v13
	v_xor_b32_e32 v11, v11, v13
	v_xor_b32_e32 v18, v18, v13
	v_xor_b32_e32 v19, v19, v13
	v_xor_b32_e32 v20, v20, v13
	v_lshl_add_u32 v177, v10, 2, v5
	v_lshl_add_u32 v178, v11, 2, v5
	v_lshl_add_u32 v179, v18, 2, v5
	v_lshl_add_u32 v180, v19, 2, v5
	v_or_b32_e32 v10, 16, v9
	v_or_b32_e32 v11, 0x110, v9
	v_or_b32_e32 v18, 20, v9
	v_or_b32_e32 v19, 0x114, v9
	v_lshl_add_u32 v185, v20, 2, v5
	v_xor_b32_e32 v10, v10, v13
	v_xor_b32_e32 v11, v11, v13
	v_xor_b32_e32 v18, v18, v13
	v_xor_b32_e32 v19, v19, v13
	v_or_b32_e32 v20, 0x128, v9
	v_lshl_add_u32 v181, v10, 2, v5
	v_lshl_add_u32 v182, v11, 2, v5
	v_lshl_add_u32 v183, v18, 2, v5
	v_lshl_add_u32 v184, v19, 2, v5
	v_or_b32_e32 v10, 0x118, v9
	v_or_b32_e32 v11, 28, v9
	v_or_b32_e32 v18, 0x11c, v9
	v_or_b32_e32 v19, 40, v9
	v_xor_b32_e32 v20, v20, v13
	v_xor_b32_e32 v10, v10, v13
	v_xor_b32_e32 v11, v11, v13
	v_xor_b32_e32 v18, v18, v13
	v_xor_b32_e32 v19, v19, v13
	v_cndmask_b32_e64 v8, v8, 0, s1
	v_lshl_add_u32 v186, v10, 2, v5
	v_lshl_add_u32 v187, v11, 2, v5
	v_lshl_add_u32 v188, v18, 2, v5
	v_lshl_add_u32 v189, v19, 2, v5
	v_or_b32_e32 v10, 44, v9
	v_or_b32_e32 v11, 0x12c, v9
	v_or_b32_e32 v18, 48, v9
	v_or_b32_e32 v19, 0x130, v9
	v_lshl_add_u32 v190, v20, 2, v5
	v_xor_b32_e32 v10, v10, v13
	v_xor_b32_e32 v11, v11, v13
	v_xor_b32_e32 v18, v18, v13
	v_xor_b32_e32 v19, v19, v13
	v_or_b32_e32 v20, 52, v9
	v_lshl_add_u32 v191, v10, 2, v5
	v_lshl_add_u32 v192, v11, 2, v5
	v_lshl_add_u32 v193, v18, 2, v5
	v_lshl_add_u32 v194, v19, 2, v5
	v_or_b32_e32 v10, 0x134, v9
	v_or_b32_e32 v11, 56, v9
	v_or_b32_e32 v18, 0x138, v9
	v_or_b32_e32 v19, 60, v9
	v_or_b32_e32 v9, 0x13c, v9
	s_min_i32 s24, s22, s2
	v_cmp_gt_u32_e64 s2, 0x100, v0
	v_add_co_u32 v8, s4, s10, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s11, 0, s4
	v_and_or_b32 v7, v7, 4, v153
	v_and_or_b32 v1, v1, 12, v153
	v_dual_mov_b32 v213, 1.0 :: v_dual_lshlrev_b32 v2, 4, v2
	v_lshlrev_b32_e32 v14, 4, v14
	v_lshlrev_b32_e32 v6, 4, v6
	v_lshl_add_u32 v4, v4, 12, 0
	v_xor_b32_e32 v20, v20, v13
	v_xor_b32_e32 v10, v10, v13
	v_xor_b32_e32 v11, v11, v13
	v_xor_b32_e32 v18, v18, v13
	v_xor_b32_e32 v19, v19, v13
	v_xor_b32_e32 v9, v9, v13
	s_and_b32 s18, s2, vcc_lo
	v_add_co_u32 v202, vcc_lo, v8, v162
	v_lshl_add_u32 v195, v20, 2, v5
	v_lshl_add_u32 v196, v10, 2, v5
	v_lshl_add_u32 v197, v11, 2, v5
	v_lshl_add_u32 v198, v18, 2, v5
	v_lshl_add_u32 v199, v19, 2, v5
	v_lshl_add_u32 v200, v9, 2, v5
	v_add_nc_u32_e32 v201, 0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v203, null, 0, v12, vcc_lo
	v_lshlrev_b32_e32 v204, 3, v7
	v_lshlrev_b32_e32 v205, 3, v1
	v_add_nc_u32_e32 v206, 0, v2
	v_lshlrev_b32_e32 v207, 3, v16
	v_add_nc_u32_e32 v209, 0, v6
	v_add_nc_u32_e32 v210, v4, v3
	v_add_nc_u32_e32 v211, 0, v15
	v_dual_mov_b32 v1, v65 :: v_dual_add_nc_u32 v212, 0, v14
	v_dual_mov_b32 v9, v65 :: v_dual_lshlrev_b32 v208, 3, v17
	v_dual_mov_b32 v11, v67 :: v_dual_mov_b32 v12, v68
	v_dual_mov_b32 v17, v65 :: v_dual_mov_b32 v20, v68
	v_dual_mov_b32 v25, v65 :: v_dual_mov_b32 v28, v68
	v_dual_mov_b32 v33, v65 :: v_dual_mov_b32 v36, v68
	v_dual_mov_b32 v41, v65 :: v_dual_mov_b32 v44, v68
	v_dual_mov_b32 v49, v65 :: v_dual_mov_b32 v52, v68
	v_dual_mov_b32 v57, v65 :: v_dual_mov_b32 v80, v72
	v_dual_mov_b32 v73, v65 :: v_dual_mov_b32 v88, v72
	v_dual_mov_b32 v81, v65 :: v_dual_mov_b32 v96, v72
	v_dual_mov_b32 v89, v65 :: v_dual_mov_b32 v104, v72
	v_dual_mov_b32 v97, v65 :: v_dual_mov_b32 v112, v72
	v_dual_mov_b32 v105, v65 :: v_dual_mov_b32 v120, v72
	v_dual_mov_b32 v113, v65 :: v_dual_mov_b32 v128, v72
	v_dual_mov_b32 v121, v65 :: v_dual_mov_b32 v136, v72
	v_mov_b32_e32 v129, v65
	v_add_co_ci_u32_e64 v150, null, s5, 0, s3
	v_cmp_gt_u32_e64 s3, 64, v0
	v_lshl_add_u32 v148, v0, 1, 0
	v_dual_mov_b32 v2, v66 :: v_dual_mov_b32 v3, v67
	v_dual_mov_b32 v4, v68 :: v_dual_mov_b32 v5, v69
	v_dual_mov_b32 v6, v70 :: v_dual_mov_b32 v7, v71
	v_mov_b32_e32 v8, v72
	v_dual_mov_b32 v10, v66 :: v_dual_mov_b32 v13, v69
	v_dual_mov_b32 v14, v70 :: v_dual_mov_b32 v19, v67
	v_dual_mov_b32 v15, v71 :: v_dual_mov_b32 v16, v72
	v_dual_mov_b32 v21, v69 :: v_dual_mov_b32 v18, v66
	v_dual_mov_b32 v23, v71 :: v_dual_mov_b32 v22, v70
	v_dual_mov_b32 v27, v67 :: v_dual_mov_b32 v24, v72
	v_dual_mov_b32 v29, v69 :: v_dual_mov_b32 v26, v66
	v_dual_mov_b32 v31, v71 :: v_dual_mov_b32 v30, v70
	v_dual_mov_b32 v35, v67 :: v_dual_mov_b32 v32, v72
	v_dual_mov_b32 v37, v69 :: v_dual_mov_b32 v34, v66
	v_dual_mov_b32 v39, v71 :: v_dual_mov_b32 v38, v70
	v_dual_mov_b32 v43, v67 :: v_dual_mov_b32 v40, v72
	v_dual_mov_b32 v45, v69 :: v_dual_mov_b32 v42, v66
	v_dual_mov_b32 v47, v71 :: v_dual_mov_b32 v46, v70
	v_dual_mov_b32 v51, v67 :: v_dual_mov_b32 v48, v72
	v_dual_mov_b32 v53, v69 :: v_dual_mov_b32 v50, v66
	v_dual_mov_b32 v55, v71 :: v_dual_mov_b32 v54, v70
	v_dual_mov_b32 v59, v67 :: v_dual_mov_b32 v56, v72
	v_dual_mov_b32 v61, v69 :: v_dual_mov_b32 v58, v66
	v_dual_mov_b32 v63, v71 :: v_dual_mov_b32 v60, v68
	v_dual_mov_b32 v79, v71 :: v_dual_mov_b32 v62, v70
	v_dual_mov_b32 v77, v69 :: v_dual_mov_b32 v64, v72
	v_dual_mov_b32 v75, v67 :: v_dual_mov_b32 v78, v70
	v_dual_mov_b32 v87, v71 :: v_dual_mov_b32 v76, v68
	v_dual_mov_b32 v85, v69 :: v_dual_mov_b32 v74, v66
	v_dual_mov_b32 v83, v67 :: v_dual_mov_b32 v86, v70
	v_dual_mov_b32 v95, v71 :: v_dual_mov_b32 v84, v68
	v_dual_mov_b32 v93, v69 :: v_dual_mov_b32 v82, v66
	v_dual_mov_b32 v91, v67 :: v_dual_mov_b32 v94, v70
	v_dual_mov_b32 v103, v71 :: v_dual_mov_b32 v92, v68
	v_dual_mov_b32 v101, v69 :: v_dual_mov_b32 v90, v66
	v_dual_mov_b32 v99, v67 :: v_dual_mov_b32 v102, v70
	v_dual_mov_b32 v111, v71 :: v_dual_mov_b32 v100, v68
	v_dual_mov_b32 v109, v69 :: v_dual_mov_b32 v98, v66
	v_dual_mov_b32 v107, v67 :: v_dual_mov_b32 v110, v70
	v_dual_mov_b32 v119, v71 :: v_dual_mov_b32 v108, v68
	v_dual_mov_b32 v117, v69 :: v_dual_mov_b32 v106, v66
	v_dual_mov_b32 v115, v67 :: v_dual_mov_b32 v118, v70
	v_dual_mov_b32 v127, v71 :: v_dual_mov_b32 v116, v68
	v_dual_mov_b32 v125, v69 :: v_dual_mov_b32 v114, v66
	v_dual_mov_b32 v123, v67 :: v_dual_mov_b32 v126, v70
	v_dual_mov_b32 v135, v71 :: v_dual_mov_b32 v124, v68
	v_dual_mov_b32 v133, v69 :: v_dual_mov_b32 v122, v66
	v_dual_mov_b32 v131, v67 :: v_dual_mov_b32 v134, v70
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v132, v68
	v_mov_b32_e32 v130, v66
	s_add_nc_u64 s[20:21], s[12:13], s[8:9]
	s_lshl_b32 s8, s25, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[10:11], s[12:13], s[8:9]
	s_add_nc_u64 s[12:13], s[14:15], s[8:9]
	s_mov_b32 s8, 0x76543210
	s_branch .LBB5_15
.LBB5_14:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v214, v66
	s_add_co_i32 s9, s9, 64
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s9, s22
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB5_60
.LBB5_15:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_51 Depth 2
                                        ;       Child Loop BB5_53 Depth 3
	v_or_b32_e32 v66, s9, v157
                                        ; implicit-def: $vgpr69
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[138:139], null, 0x408, v66, s[20:21]
	v_cmp_ge_i32_e32 vcc_lo, s22, v66
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s4
	s_cbranch_execz .LBB5_17
; %bb.16:                               ;   in Loop: Header=BB5_15 Depth=1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_u32 v66, s4, v138, v204
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v67, null, 0, v139, s4
	v_add_co_u32 v71, s4, v138, v205
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v72, null, 0, v139, s4
	s_clause 0x3
	global_load_b64 v[140:141], v[66:67], off
	global_load_b64 v[142:143], v[66:67], off offset:16
	global_load_b64 v[69:70], v[71:72], off
	global_load_b64 v[71:72], v[71:72], off offset:16
	s_wait_loadcnt 0x2
	ds_store_b128 v211, v[140:143]
.LBB5_17:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s5
	s_cbranch_execz .LBB5_19
; %bb.18:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v67, v65
	v_mov_b32_e32 v66, v65
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v68, v65 :: v_dual_mov_b32 v69, v72
	v_dual_mov_b32 v71, v72 :: v_dual_mov_b32 v70, v72
	ds_store_b128 v211, v[65:68]
.LBB5_19:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	ds_store_b128 v206, v[69:72]
                                        ; implicit-def: $vgpr69
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execnz .LBB5_22
; %bb.20:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execnz .LBB5_23
.LBB5_21:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	ds_store_b128 v209, v[69:72]
	s_and_saveexec_b32 s4, s2
	s_cbranch_execnz .LBB5_24
	s_branch .LBB5_41
.LBB5_22:                               ;   in Loop: Header=BB5_15 Depth=1
	v_add_co_u32 v66, vcc_lo, v138, v207
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, 0, v139, vcc_lo
	v_add_co_u32 v71, vcc_lo, v138, v208
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v72, null, 0, v139, vcc_lo
	s_clause 0x3
	global_load_b64 v[138:139], v[66:67], off
	global_load_b64 v[140:141], v[66:67], off offset:16
	global_load_b64 v[69:70], v[71:72], off
	global_load_b64 v[71:72], v[71:72], off offset:16
	s_wait_loadcnt 0x2
	ds_store_b128 v212, v[138:141]
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB5_21
.LBB5_23:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v67, v65
	v_mov_b32_e32 v66, v65
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v68, v65 :: v_dual_mov_b32 v69, v72
	v_dual_mov_b32 v71, v72 :: v_dual_mov_b32 v70, v72
	ds_store_b128 v212, v[65:68]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	ds_store_b128 v209, v[69:72]
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB5_41
.LBB5_24:                               ;   in Loop: Header=BB5_15 Depth=1
	v_or_b32_e32 v72, s9, v158
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s22, v72
	s_cbranch_execz .LBB5_26
; %bb.25:                               ;   in Loop: Header=BB5_15 Depth=1
	v_mad_co_u64_u32 v[66:67], null, 0x408, v72, v[149:150]
	global_load_b64 v[66:67], v[66:67], off
.LBB5_26:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_add_nc_u32_e32 v70, 0x8000, v163
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v70, v66, v67 offset1:1
	v_cmpx_gt_i32_e64 s22, v72
	s_cbranch_execz .LBB5_28
; %bb.27:                               ;   in Loop: Header=BB5_15 Depth=1
	v_or_b32_e32 v66, 1, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[66:67], null, 0x408, v66, v[149:150]
	global_load_b64 v[68:69], v[66:67], off
.LBB5_28:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v138, 2, v72
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v164, v68 offset:32768
	ds_store_b32 v165, v69 offset:32768
	v_cmpx_ge_i32_e64 s22, v138
	s_cbranch_execz .LBB5_30
; %bb.29:                               ;   in Loop: Header=BB5_15 Depth=1
	v_mad_co_u64_u32 v[68:69], null, 0x408, v138, v[149:150]
	global_load_b64 v[70:71], v[68:69], off
.LBB5_30:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v68, 3, v72
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v166, v70 offset:32768
	ds_store_b32 v167, v71 offset:32768
	v_cmpx_ge_i32_e64 s22, v68
	s_cbranch_execz .LBB5_32
; %bb.31:                               ;   in Loop: Header=BB5_15 Depth=1
	v_mad_co_u64_u32 v[66:67], null, 0x408, v68, v[149:150]
	global_load_b64 v[66:67], v[66:67], off
.LBB5_32:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v138, 4, v72
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v168, v66 offset:32768
	ds_store_b32 v169, v67 offset:32768
	v_cmpx_ge_i32_e64 s22, v138
	s_cbranch_execz .LBB5_34
; %bb.33:                               ;   in Loop: Header=BB5_15 Depth=1
	v_mad_co_u64_u32 v[66:67], null, 0x408, v138, v[149:150]
	global_load_b64 v[70:71], v[66:67], off
.LBB5_34:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v66, 5, v72
	v_add_nc_u32_e32 v67, 0x8400, v163
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v67, v70, v71 offset1:1
	v_cmpx_ge_i32_e64 s22, v66
	s_cbranch_execz .LBB5_36
; %bb.35:                               ;   in Loop: Header=BB5_15 Depth=1
	v_mad_co_u64_u32 v[66:67], null, 0x408, v66, v[149:150]
	global_load_b64 v[68:69], v[66:67], off
.LBB5_36:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v138, 6, v72
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v170, v68 offset:32768
	ds_store_b32 v171, v69 offset:32768
	v_cmpx_ge_i32_e64 s22, v138
	s_cbranch_execz .LBB5_38
; %bb.37:                               ;   in Loop: Header=BB5_15 Depth=1
	v_mad_co_u64_u32 v[68:69], null, 0x408, v138, v[149:150]
	global_load_b64 v[70:71], v[68:69], off
.LBB5_38:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v68, 7, v72
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v172, v70 offset:32768
	ds_store_b32 v173, v71 offset:32768
	v_cmpx_ge_i32_e64 s22, v68
	s_cbranch_execz .LBB5_40
; %bb.39:                               ;   in Loop: Header=BB5_15 Depth=1
	v_mad_co_u64_u32 v[66:67], null, 0x408, v68, v[149:150]
	global_load_b64 v[66:67], v[66:67], off
.LBB5_40:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_wait_loadcnt 0x0
	ds_store_b32 v174, v66 offset:32768
	ds_store_b32 v175, v67 offset:32768
.LBB5_41:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, s18
	s_cbranch_execz .LBB5_43
; %bb.42:                               ;   in Loop: Header=BB5_15 Depth=1
	v_add_nc_u32_e32 v70, 0x8000, v176
	v_add_nc_u32_e32 v71, 0x8400, v176
	ds_load_2addr_b32 v[66:67], v70 offset1:4
	ds_load_2addr_b32 v[68:69], v71 offset1:4
	s_wait_dscnt 0x1
	ds_bpermute_b32 v72, v156, v66
	s_wait_dscnt 0x1
	ds_bpermute_b32 v138, v156, v68
	ds_bpermute_b32 v139, v156, v67
	ds_bpermute_b32 v140, v156, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v72, v66, v159
	s_wait_dscnt 0x2
	v_perm_b32 v68, v138, v68, v159
	s_wait_dscnt 0x1
	v_perm_b32 v72, v139, v67, v159
	s_wait_dscnt 0x0
	v_perm_b32 v69, v140, v69, v159
	ds_bpermute_b32 v67, v155, v66
	ds_bpermute_b32 v138, v155, v68
	ds_bpermute_b32 v139, v155, v72
	ds_bpermute_b32 v140, v155, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v67, v66, v160
	s_wait_dscnt 0x2
	v_perm_b32 v67, v138, v68, v160
	s_wait_dscnt 0x1
	v_perm_b32 v68, v139, v72, v160
	s_wait_dscnt 0x0
	v_perm_b32 v69, v140, v69, v160
	ds_store_b128 v210, v[66:69] offset:16384
	ds_load_b32 v66, v177 offset:32768
	ds_load_b32 v67, v178 offset:32768
	ds_load_b32 v68, v179 offset:32768
	ds_load_b32 v69, v180 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v72, v156, v66
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v156, v67
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v156, v68
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v156, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v72, v66, v159
	s_wait_dscnt 0x2
	v_perm_b32 v67, v138, v67, v159
	s_wait_dscnt 0x1
	v_perm_b32 v68, v139, v68, v159
	s_wait_dscnt 0x0
	v_perm_b32 v69, v140, v69, v159
	ds_bpermute_b32 v72, v155, v66
	ds_bpermute_b32 v138, v155, v67
	ds_bpermute_b32 v139, v155, v68
	ds_bpermute_b32 v140, v155, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v72, v66, v160
	s_wait_dscnt 0x2
	v_perm_b32 v67, v138, v67, v160
	s_wait_dscnt 0x1
	v_perm_b32 v68, v139, v68, v160
	s_wait_dscnt 0x0
	v_perm_b32 v69, v140, v69, v160
	ds_store_b128 v210, v[66:69] offset:16896
	ds_load_b32 v66, v181 offset:32768
	ds_load_b32 v67, v182 offset:32768
	ds_load_b32 v68, v183 offset:32768
	ds_load_b32 v69, v184 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v72, v156, v66
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v156, v67
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v156, v68
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v156, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v72, v66, v159
	s_wait_dscnt 0x2
	v_perm_b32 v67, v138, v67, v159
	s_wait_dscnt 0x1
	v_perm_b32 v68, v139, v68, v159
	s_wait_dscnt 0x0
	v_perm_b32 v69, v140, v69, v159
	ds_bpermute_b32 v72, v155, v66
	ds_bpermute_b32 v138, v155, v67
	ds_bpermute_b32 v139, v155, v68
	ds_bpermute_b32 v140, v155, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v72, v66, v160
	s_wait_dscnt 0x2
	v_perm_b32 v67, v138, v67, v160
	s_wait_dscnt 0x1
	v_perm_b32 v68, v139, v68, v160
	s_wait_dscnt 0x0
	v_perm_b32 v69, v140, v69, v160
	ds_store_b128 v210, v[66:69] offset:17408
	ds_load_b32 v66, v185 offset:32768
	ds_load_b32 v67, v186 offset:32768
	ds_load_b32 v68, v187 offset:32768
	ds_load_b32 v69, v188 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v72, v156, v66
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v156, v67
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v156, v68
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v156, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v72, v66, v159
	s_wait_dscnt 0x2
	v_perm_b32 v67, v138, v67, v159
	s_wait_dscnt 0x1
	v_perm_b32 v68, v139, v68, v159
	s_wait_dscnt 0x0
	v_perm_b32 v69, v140, v69, v159
	ds_bpermute_b32 v72, v155, v66
	ds_bpermute_b32 v138, v155, v67
	ds_bpermute_b32 v139, v155, v68
	ds_bpermute_b32 v140, v155, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v72, v66, v160
	s_wait_dscnt 0x2
	v_perm_b32 v67, v138, v67, v160
	s_wait_dscnt 0x1
	v_perm_b32 v68, v139, v68, v160
	s_wait_dscnt 0x0
	v_perm_b32 v69, v140, v69, v160
	ds_store_b128 v210, v[66:69] offset:17920
	ds_load_2addr_b32 v[66:67], v70 offset0:32 offset1:36
	ds_load_2addr_b32 v[68:69], v71 offset0:32 offset1:36
	s_wait_dscnt 0x1
	ds_bpermute_b32 v70, v156, v66
	s_wait_dscnt 0x1
	ds_bpermute_b32 v71, v156, v68
	ds_bpermute_b32 v72, v156, v67
	ds_bpermute_b32 v138, v156, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v70, v66, v159
	s_wait_dscnt 0x2
	v_perm_b32 v68, v71, v68, v159
	s_wait_dscnt 0x1
	v_perm_b32 v70, v72, v67, v159
	s_wait_dscnt 0x0
	v_perm_b32 v69, v138, v69, v159
	ds_bpermute_b32 v67, v155, v66
	ds_bpermute_b32 v71, v155, v68
	ds_bpermute_b32 v72, v155, v70
	ds_bpermute_b32 v138, v155, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v67, v66, v160
	s_wait_dscnt 0x2
	v_perm_b32 v67, v71, v68, v160
	s_wait_dscnt 0x1
	v_perm_b32 v68, v72, v70, v160
	s_wait_dscnt 0x0
	v_perm_b32 v69, v138, v69, v160
	ds_store_b128 v210, v[66:69] offset:18432
	ds_load_b32 v66, v189 offset:32768
	ds_load_b32 v67, v190 offset:32768
	ds_load_b32 v68, v191 offset:32768
	ds_load_b32 v69, v192 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v70, v156, v66
	s_wait_dscnt 0x3
	ds_bpermute_b32 v71, v156, v67
	s_wait_dscnt 0x3
	ds_bpermute_b32 v72, v156, v68
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v156, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v70, v66, v159
	s_wait_dscnt 0x2
	v_perm_b32 v67, v71, v67, v159
	s_wait_dscnt 0x1
	v_perm_b32 v68, v72, v68, v159
	s_wait_dscnt 0x0
	v_perm_b32 v69, v138, v69, v159
	ds_bpermute_b32 v70, v155, v66
	ds_bpermute_b32 v71, v155, v67
	ds_bpermute_b32 v72, v155, v68
	ds_bpermute_b32 v138, v155, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v70, v66, v160
	s_wait_dscnt 0x2
	v_perm_b32 v67, v71, v67, v160
	s_wait_dscnt 0x1
	v_perm_b32 v68, v72, v68, v160
	s_wait_dscnt 0x0
	v_perm_b32 v69, v138, v69, v160
	ds_store_b128 v210, v[66:69] offset:18944
	ds_load_b32 v66, v193 offset:32768
	ds_load_b32 v67, v194 offset:32768
	ds_load_b32 v68, v195 offset:32768
	ds_load_b32 v69, v196 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v70, v156, v66
	s_wait_dscnt 0x3
	ds_bpermute_b32 v71, v156, v67
	s_wait_dscnt 0x3
	ds_bpermute_b32 v72, v156, v68
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v156, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v70, v66, v159
	s_wait_dscnt 0x2
	v_perm_b32 v67, v71, v67, v159
	s_wait_dscnt 0x1
	v_perm_b32 v68, v72, v68, v159
	s_wait_dscnt 0x0
	v_perm_b32 v69, v138, v69, v159
	ds_bpermute_b32 v70, v155, v66
	ds_bpermute_b32 v71, v155, v67
	ds_bpermute_b32 v72, v155, v68
	ds_bpermute_b32 v138, v155, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v70, v66, v160
	s_wait_dscnt 0x2
	v_perm_b32 v67, v71, v67, v160
	s_wait_dscnt 0x1
	v_perm_b32 v68, v72, v68, v160
	s_wait_dscnt 0x0
	v_perm_b32 v69, v138, v69, v160
	ds_store_b128 v210, v[66:69] offset:19456
	ds_load_b32 v66, v197 offset:32768
	ds_load_b32 v67, v198 offset:32768
	ds_load_b32 v68, v199 offset:32768
	ds_load_b32 v69, v200 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v70, v156, v66
	s_wait_dscnt 0x3
	ds_bpermute_b32 v71, v156, v67
	s_wait_dscnt 0x3
	ds_bpermute_b32 v72, v156, v68
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v156, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v70, v66, v159
	s_wait_dscnt 0x2
	v_perm_b32 v67, v71, v67, v159
	s_wait_dscnt 0x1
	v_perm_b32 v68, v72, v68, v159
	s_wait_dscnt 0x0
	v_perm_b32 v69, v138, v69, v159
	ds_bpermute_b32 v70, v155, v66
	ds_bpermute_b32 v71, v155, v67
	ds_bpermute_b32 v72, v155, v68
	ds_bpermute_b32 v138, v155, v69
	s_wait_dscnt 0x3
	v_perm_b32 v66, v70, v66, v160
	s_wait_dscnt 0x2
	v_perm_b32 v67, v71, v67, v160
	s_wait_dscnt 0x1
	v_perm_b32 v68, v72, v68, v160
	s_wait_dscnt 0x0
	v_perm_b32 v69, v138, v69, v160
	ds_store_b128 v210, v[66:69] offset:19968
.LBB5_43:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_47
; %bb.44:                               ;   in Loop: Header=BB5_15 Depth=1
	v_or_b32_e32 v67, s9, v0
	v_mov_b32_e32 v66, 0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s22, v67
	s_cbranch_execz .LBB5_46
; %bb.45:                               ;   in Loop: Header=BB5_15 Depth=1
	v_mad_co_u64_u32 v[68:69], null, 0x408, v67, s[10:11]
	v_mad_co_u64_u32 v[70:71], null, 0x408, v67, s[12:13]
	global_load_d16_b16 v66, v[68:69], off offset:1024
	global_load_d16_hi_b16 v66, v[70:71], off offset:1024
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v67.h, 8, v66.l
	v_lshrrev_b16 v67.l, 8, v66.h
	v_and_b16 v68.h, 0xff, v66.l
	v_and_b16 v68.l, 0xff, v66.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_lshlrev_b16 v66, 8, v67 op_sel_hi:[0,1]
	v_or_b32_e32 v66, v66, v68
.LBB5_46:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	ds_store_b16_d16_hi v148, v66 offset:49152
	ds_store_b16 v148, v66 offset:49280
.LBB5_47:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_or_b32 s4, s9, 63
	v_mov_b32_e32 v69, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s24
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s14, s23, s4
	s_wait_alu depctr_sa_sdst(0)
	s_nor_b32 s5, s1, s14
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s5
	s_cbranch_execz .LBB5_49
; %bb.48:                               ;   in Loop: Header=BB5_15 Depth=1
	global_load_b32 v69, v[151:152], off
.LBB5_49:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_mov_b32 s15, 0
	s_branch .LBB5_51
.LBB5_50:                               ;   in Loop: Header=BB5_51 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v218, null, v67, v67, v141
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v214
	v_add_f32_e32 v68, v68, v70
	v_div_scale_f32 v220, null, v67, v67, v140
	v_mov_b16_e64 v238.l, v65.l
	v_rcp_f32_e32 v219, v218
	v_mov_b16_e64 v238.h, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v237.l, v238.l
	v_mov_b16_e64 v237.h, v238.h
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v221, -v218, v219, 1.0
	v_dual_sub_f32 v144, v214, v66 :: v_dual_fmac_f32 v219, v221, v219
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v144, 0x3fb8aa3b, v144
	v_div_scale_f32 v221, null, v67, v67, v71
	v_exp_f32_e32 v144, v144
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v144, 0, v144, vcc_lo
	v_mul_f32_e32 v145, v213, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v213, null, v67, v67, v145
	v_rcp_f32_e32 v214, v213
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v215, -v213, v214, 1.0
	v_fmac_f32_e32 v214, v215, v214
	v_div_scale_f32 v216, vcc_lo, v145, v67, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v215, v216, v214
	v_fma_f32 v217, -v213, v215, v216
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v215, v217, v214
	v_fma_f32 v213, -v213, v215, v216
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v213, v213, v214, v215
	v_div_fixup_f32 v70, v213, v67, v145
	v_div_scale_f32 v145, null, v67, v67, v142
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v135, v135, v70 :: v_dual_fmac_f32 v68, v137, v144
	v_mul_f32_e32 v131, v131, v70
	v_div_scale_f32 v137, null, v67, v67, v143
	v_rcp_f32_e32 v214, v145
	v_dual_mul_f32 v136, v136, v70 :: v_dual_mul_f32 v133, v133, v70
	v_dual_mul_f32 v134, v134, v70 :: v_dual_mul_f32 v129, v129, v70
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v144, v137
	v_dual_mul_f32 v132, v132, v70 :: v_dual_mul_f32 v127, v127, v70
	v_dual_mul_f32 v130, v130, v70 :: v_dual_mul_f32 v125, v125, v70
	v_fma_f32 v216, -v145, v214, 1.0
	v_dual_mul_f32 v128, v128, v70 :: v_dual_mul_f32 v123, v123, v70
	v_mul_f32_e32 v15, v15, v70
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v213, -v137, v144, 1.0
	v_fmac_f32_e32 v214, v216, v214
	v_div_scale_f32 v216, s4, v142, v67, v142
	v_dual_mul_f32 v126, v126, v70 :: v_dual_mul_f32 v121, v121, v70
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v144, v213, v144
	v_div_scale_f32 v213, vcc_lo, v143, v67, v143
	v_mul_f32_e32 v20, v20, v70
	v_dual_mul_f32 v124, v124, v70 :: v_dual_mul_f32 v119, v119, v70
	v_dual_mul_f32 v122, v122, v70 :: v_dual_mul_f32 v117, v117, v70
	v_dual_mul_f32 v215, v213, v144 :: v_dual_mul_f32 v120, v120, v70
	v_dual_mul_f32 v115, v115, v70 :: v_dual_mul_f32 v118, v118, v70
	v_mul_f32_e32 v113, v113, v70
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fma_f32 v217, -v137, v215, v213
	v_mul_f32_e32 v14, v14, v70
	v_dual_mul_f32 v116, v116, v70 :: v_dual_mul_f32 v111, v111, v70
	v_dual_mul_f32 v114, v114, v70 :: v_dual_mul_f32 v109, v109, v70
	v_dual_fmac_f32 v215, v217, v144 :: v_dual_mul_f32 v112, v112, v70
	v_dual_mul_f32 v107, v107, v70 :: v_dual_mul_f32 v110, v110, v70
	v_mul_f32_e32 v105, v105, v70
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fma_f32 v137, -v137, v215, v213
	v_dual_mul_f32 v108, v108, v70 :: v_dual_mul_f32 v103, v103, v70
	v_dual_mul_f32 v106, v106, v70 :: v_dual_mul_f32 v101, v101, v70
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v137, v137, v144, v215
	v_rcp_f32_e32 v144, v220
	s_mov_b32 vcc_lo, s4
	v_dual_mul_f32 v104, v104, v70 :: v_dual_mul_f32 v99, v99, v70
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_div_fixup_f32 v137, v137, v67, v143
	v_dual_mul_f32 v102, v102, v70 :: v_dual_mul_f32 v97, v97, v70
	v_dual_mul_f32 v100, v100, v70 :: v_dual_mul_f32 v95, v95, v70
	v_fma_f32 v215, -v220, v144, 1.0
	v_mul_f32_e32 v217, v216, v214
	v_dual_mul_f32 v98, v98, v70 :: v_dual_mul_f32 v93, v93, v70
	v_dual_mul_f32 v96, v96, v70 :: v_dual_mul_f32 v91, v91, v70
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v144, v215, v144
	v_fma_f32 v213, -v145, v217, v216
	v_div_scale_f32 v215, s4, v140, v67, v140
	v_dual_mul_f32 v94, v94, v70 :: v_dual_mul_f32 v89, v89, v70
	v_dual_mul_f32 v92, v92, v70 :: v_dual_mul_f32 v87, v87, v70
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v217, v213, v214
	v_div_scale_f32 v213, s5, v141, v67, v141
	v_dual_mul_f32 v90, v90, v70 :: v_dual_mul_f32 v85, v85, v70
	v_dual_mul_f32 v88, v88, v70 :: v_dual_mul_f32 v83, v83, v70
	v_fma_f32 v143, -v145, v217, v216
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v145, v213, v219 :: v_dual_mul_f32 v86, v86, v70
	v_dual_mul_f32 v81, v81, v70 :: v_dual_mul_f32 v84, v84, v70
	v_mul_f32_e32 v79, v79, v70
	v_fma_f32 v216, -v218, v145, v213
	v_mul_f32_e32 v6, v6, v70
	v_dual_mul_f32 v82, v82, v70 :: v_dual_mul_f32 v77, v77, v70
	v_dual_mul_f32 v80, v80, v70 :: v_dual_mul_f32 v75, v75, v70
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v145, v216, v219 :: v_dual_mul_f32 v216, v215, v144
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v143, v143, v214, v217
	v_div_scale_f32 v214, null, v67, v67, v139
	v_div_scale_f32 v217, null, v67, v67, v138
	s_mov_b32 vcc_lo, s5
	v_dual_mul_f32 v78, v78, v70 :: v_dual_mul_f32 v73, v73, v70
	v_div_fixup_f32 v142, v143, v67, v142
	v_rcp_f32_e32 v143, v214
	v_dual_mul_f32 v76, v76, v70 :: v_dual_mul_f32 v63, v63, v70
	v_dual_mul_f32 v74, v74, v70 :: v_dual_mul_f32 v61, v61, v70
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	v_cvt_pk_fp8_f32 v237.l, v137, v142
	v_fma_f32 v137, -v218, v145, v213
	v_fma_f32 v142, -v220, v216, v215
	v_rcp_f32_e32 v218, v217
	v_fma_f32 v213, -v214, v143, 1.0
	v_dual_mul_f32 v64, v64, v70 :: v_dual_mul_f32 v59, v59, v70
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v137, v137, v219, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v216, v142, v144 :: v_dual_fmac_f32 v143, v213, v143
	v_rcp_f32_e32 v145, v221
	v_div_scale_f32 v142, s5, v139, v67, v139
	v_div_fixup_f32 v137, v137, v67, v141
	v_div_scale_f32 v141, null, v67, v67, v72
	v_fma_f32 v213, -v217, v218, 1.0
	v_fma_f32 v215, -v220, v216, v215
	v_mul_f32_e32 v219, v142, v143
	s_mov_b32 vcc_lo, s4
	v_rcp_f32_e32 v220, v141
	v_fmac_f32_e32 v218, v213, v218
	v_div_scale_f32 v213, s6, v138, v67, v138
	v_fma_f32 v222, -v221, v145, 1.0
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v215, v144, v216
	v_fma_f32 v215, -v214, v219, v142
	s_mov_b32 vcc_lo, s5
	v_dual_mul_f32 v216, v213, v218 :: v_dual_fmac_f32 v145, v222, v145
	v_div_scale_f32 v222, s4, v71, v67, v71
	v_fma_f32 v223, -v141, v220, 1.0
	v_div_fixup_f32 v140, v144, v67, v140
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v144, -v217, v216, v213
	v_fmac_f32_e32 v219, v215, v143
	v_dual_mul_f32 v215, v222, v145 :: v_dual_fmac_f32 v220, v223, v220
	v_div_scale_f32 v223, s7, v72, v67, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v216, v144, v218
	v_fma_f32 v142, -v214, v219, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v144, -v221, v215, v222
	v_mul_f32_e32 v3, v3, v70
	v_mul_f32_e32 v214, v223, v220
	v_cvt_pk_fp8_f32 v237.h, v137, v140
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v142, v142, v143, v219
	v_fma_f32 v143, -v217, v216, v213
	v_fmac_f32_e32 v215, v144, v145
	v_fma_f32 v213, -v141, v214, v223
	s_mov_b32 vcc_lo, s6
	v_div_fixup_f32 v139, v142, v67, v139
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v143, v143, v218, v216
	v_fma_f32 v142, -v221, v215, v222
	v_fmac_f32_e32 v214, v213, v220
	s_mov_b32 vcc_lo, s4
	v_dual_mul_f32 v62, v62, v70 :: v_dual_mul_f32 v57, v57, v70
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v142, v142, v145, v215
	v_fma_f32 v141, -v141, v214, v223
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v138, v143, v67, v138
	v_lshl_add_u32 v145, s15, 12, v201
	v_div_fixup_f32 v71, v142, v67, v71
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v141, v141, v220, v214
	v_dual_mul_f32 v60, v60, v70 :: v_dual_mul_f32 v55, v55, v70
	v_cvt_pk_fp8_f32 v238.l, v139, v138
	ds_load_b128 v[137:140], v145 offset:16384
	;;#ASMSTART
	;;#ASMEND
	v_div_fixup_f32 v72, v141, v67, v72
	ds_load_b128 v[141:144], v145 offset:16896
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[213:216], v145 offset:17408
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[217:220], v145 offset:17920
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[221:224], v145 offset:18432
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[225:228], v145 offset:18944
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[229:232], v145 offset:19456
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[233:236], v145 offset:19968
	v_dual_mul_f32 v58, v58, v70 :: v_dual_mul_f32 v53, v53, v70
	v_dual_mul_f32 v56, v56, v70 :: v_dual_mul_f32 v51, v51, v70
	v_dual_mul_f32 v54, v54, v70 :: v_dual_mul_f32 v49, v49, v70
	v_dual_mul_f32 v52, v52, v70 :: v_dual_mul_f32 v47, v47, v70
	v_dual_mul_f32 v50, v50, v70 :: v_dual_mul_f32 v45, v45, v70
	v_dual_mul_f32 v48, v48, v70 :: v_dual_mul_f32 v43, v43, v70
	v_dual_mul_f32 v46, v46, v70 :: v_dual_mul_f32 v41, v41, v70
	v_dual_mul_f32 v44, v44, v70 :: v_dual_mul_f32 v39, v39, v70
	v_dual_mul_f32 v42, v42, v70 :: v_dual_mul_f32 v37, v37, v70
	v_dual_mul_f32 v40, v40, v70 :: v_dual_mul_f32 v35, v35, v70
	v_dual_mul_f32 v38, v38, v70 :: v_dual_mul_f32 v33, v33, v70
	v_dual_mul_f32 v36, v36, v70 :: v_dual_mul_f32 v31, v31, v70
	v_dual_mul_f32 v34, v34, v70 :: v_dual_mul_f32 v29, v29, v70
	v_dual_mul_f32 v32, v32, v70 :: v_dual_mul_f32 v27, v27, v70
	v_dual_mul_f32 v30, v30, v70 :: v_dual_mul_f32 v25, v25, v70
	v_dual_mul_f32 v28, v28, v70 :: v_dual_mul_f32 v23, v23, v70
	v_dual_mul_f32 v26, v26, v70 :: v_dual_mul_f32 v21, v21, v70
	v_dual_mul_f32 v24, v24, v70 :: v_dual_mul_f32 v19, v19, v70
	v_dual_mul_f32 v22, v22, v70 :: v_dual_mul_f32 v17, v17, v70
	v_dual_mul_f32 v18, v18, v70 :: v_dual_mul_f32 v13, v13, v70
	v_dual_mul_f32 v16, v16, v70 :: v_dual_mul_f32 v11, v11, v70
	v_dual_mul_f32 v12, v12, v70 :: v_dual_mul_f32 v9, v9, v70
	v_dual_mul_f32 v10, v10, v70 :: v_dual_mul_f32 v7, v7, v70
	v_dual_mul_f32 v8, v8, v70 :: v_dual_mul_f32 v5, v5, v70
	v_cvt_pk_fp8_f32 v238.h, v71, v72
	v_dual_mul_f32 v4, v4, v70 :: v_dual_mul_f32 v1, v1, v70
	v_mul_f32_e32 v2, v2, v70
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[137:138], v[237:238], v[129:136]
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[139:140], v[237:238], v[121:128]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[141:142], v[237:238], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[143:144], v[237:238], v[105:112]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[213:214], v[237:238], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[215:216], v[237:238], v[89:96]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[217:218], v[237:238], v[81:88]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[219:220], v[237:238], v[73:80]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[221:222], v[237:238], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[223:224], v[237:238], v[49:56]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[225:226], v[237:238], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[227:228], v[237:238], v[33:40]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[229:230], v[237:238], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[231:232], v[237:238], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[233:234], v[237:238], v[9:16]
	v_mov_b32_e32 v137, v68
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[235:236], v[237:238], v[1:8]
	v_dual_mov_b32 v213, v67 :: v_dual_mov_b32 v214, v66
	s_add_co_i32 s15, s15, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s15, 4
	s_cbranch_scc1 .LBB5_14
.LBB5_51:                               ;   Parent Loop BB5_15 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_53 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s5, s15, 4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s5, s9
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s4, s22
	s_cbranch_scc1 .LBB5_58
; %bb.52:                               ;   in Loop: Header=BB5_51 Depth=2
	v_mov_b32_e32 v138, 0
	v_lshl_add_u32 v66, s15, 9, v201
	s_mov_b32 s6, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v139, v138 :: v_dual_mov_b32 v140, v138
	v_dual_mov_b32 v141, v138 :: v_dual_mov_b32 v142, v138
	v_dual_mov_b32 v143, v138 :: v_dual_mov_b32 v144, v138
	v_mov_b32_e32 v145, v138
.LBB5_53:                               ;   Parent Loop BB5_15 Depth=1
                                        ;     Parent Loop BB5_51 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s7, s6, 6
	v_lshl_add_u32 v72, s6, 12, v66
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v67, vcc_lo, v202, s7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v68, null, 0, v203, vcc_lo
	ds_load_b128 v[215:218], v72
	ds_load_b128 v[219:222], v72 offset:2048
	s_add_co_i32 s6, s6, 1
	s_clause 0x3
	global_load_b64 v[70:71], v[67:68], off
	global_load_b64 v[223:224], v[67:68], off offset:16
	global_load_b64 v[225:226], v[67:68], off offset:32
	global_load_b64 v[67:68], v[67:68], off offset:48
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s6, 4
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[215:216], v[70:71], v[138:145]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[217:218], v[223:224], v[138:145]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[219:220], v[225:226], v[138:145]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[138:145], v[221:222], v[67:68], v[138:145]
	s_cbranch_scc1 .LBB5_53
; %bb.54:                               ;   in Loop: Header=BB5_51 Depth=2
	v_lshl_add_u32 v70, s5, 1, v161
	s_or_b32 s5, s4, 15
	v_or_b32_e32 v215, s4, v162
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s5, s24
	v_mov_b32_e32 v71, 0xff800000
	ds_load_b96 v[66:68], v70 offset:49154
	ds_load_u16_d16 v72, v70 offset:49166
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s14, s4
	s_and_saveexec_b32 s5, s0
	s_cbranch_execz .LBB5_56
; %bb.55:                               ;   in Loop: Header=BB5_51 Depth=2
	ds_load_u16_d16 v71, v70 offset:49152
	v_cmp_le_i32_e32 vcc_lo, v215, v69
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s4, vcc_lo
	v_mul_f32_e32 v138, v154, v138
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v71, v71, v138, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v71, 0xff800000, v71, vcc_lo
.LBB5_56:                               ;   in Loop: Header=BB5_51 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_mul_f32_e32 v138, v154, v139
	v_or_b32_e32 v139, 2, v215
	v_cmp_ge_i32_e32 vcc_lo, v215, v69
	s_xor_b32 s5, s4, -1
	v_mul_f32_e32 v140, v154, v140
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s4, v139, v69
	v_or_b32_e32 v139, 3, v215
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s6, s5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s1, s6
	s_and_b32 s4, s5, s4
	v_cmp_gt_i32_e32 vcc_lo, v139, v69
	v_mul_f32_e32 v139, v154, v141
	s_wait_dscnt 0x1
	v_fma_mix_f32 v138, v66, v138, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v66, v66, v140, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_or_b32_e32 v140, 4, v215
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	v_cndmask_b32_e64 v138, v138, 0xff800000, s6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v141, v66, 0xff800000, s4
	v_fma_mix_f32 v66, v67, v139, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v139, v154, v142
	s_and_b32 s4, s5, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v140, v69
	v_or_b32_e32 v140, 5, v215
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v142, v66, 0xff800000, s4
	v_fma_mix_f32 v66, v67, v139, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v67, v154, v143
	s_and_b32 s4, s5, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v140, v69
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	v_or_b32_e32 v140, 7, v215
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v139, v66, 0xff800000, s4
	v_fma_mix_f32 v66, v68, v67, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v67, 6, v215
	s_and_b32 s4, s5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v143, v66, 0xff800000, s4
	v_cmp_gt_i32_e32 vcc_lo, v67, v69
	v_mul_f32_e32 v66, v154, v144
	v_cmp_gt_i32_e64 s4, v140, v69
	v_mul_f32_e32 v67, v154, v145
	v_max3_num_f32 v140, v71, 0xff800000, v138
	s_and_b32 s6, s5, vcc_lo
	v_fma_mix_f32 v66, v68, v66, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_and_b32 s4, s5, s4
	s_wait_dscnt 0x0
	v_fma_mix_f32 v67, v72, v67, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v68, v140, v141, v142
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s1, s6
	s_or_b32 s4, s1, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v72, v66, 0xff800000, s5
	v_cndmask_b32_e64 v67, v67, 0xff800000, s4
	v_max3_num_f32 v66, v68, v139, v143
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v66, v66, v72, v67
	v_mov_b32_e32 v68, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v68, v68, s8, 0xfedcba98
	v_max3_num_f32 v66, v214, v66, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v68, v71, v66 :: v_dual_sub_f32 v71, v138, v66
	v_dual_sub_f32 v67, v67, v66 :: v_dual_sub_f32 v72, v72, v66
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v66
	v_dual_mul_f32 v68, 0x3fb8aa3b, v68 :: v_dual_mul_f32 v71, 0x3fb8aa3b, v71
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v67, 0x3fb8aa3b, v67
	v_add_nc_u32_e32 v138, 0xc080, v70
	v_exp_f32_e32 v144, v71
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v140, v67
	v_sub_f32_e32 v71, v143, v66
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v144, v144, 0, vcc_lo
	v_sub_f32_e32 v67, v141, v66
	v_sub_f32_e32 v141, v142, v66
	v_exp_f32_e32 v142, v68
	s_delay_alu instid0(TRANS32_DEP_2)
	v_cndmask_b32_e64 v215, v140, 0, vcc_lo
	v_mul_f32_e32 v145, 0x3fb8aa3b, v72
	v_mul_f32_e32 v143, 0x3fb8aa3b, v67
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	ds_load_2addr_b32 v[67:68], v138 offset1:1
	v_sub_f32_e32 v138, v139, v66
	v_exp_f32_e32 v145, v145
	v_exp_f32_e32 v139, v141
	v_cndmask_b32_e64 v140, v142, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v138, 0x3fb8aa3b, v138
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v217, v139, 0, vcc_lo
	v_mul_f32_e32 v71, 0x3fb8aa3b, v71
	v_exp_f32_e32 v141, v71
	s_wait_dscnt 0x0
	v_fma_mix_f32 v142, v67, v144, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_cndmask_b32_e64 v216, v141, 0, vcc_lo
	v_add_nc_u32_e32 v70, 0xc088, v70
	ds_load_2addr_b32 v[71:72], v70 offset1:1
	v_exp_f32_e32 v70, v143
	v_fma_mix_f32 v143, v67, v140, neg(0) op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v67, v138, 0, vcc_lo
	v_add_f32_e32 v138, v140, v144
	v_fma_mix_f32 v140, v68, v217, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v144, v145, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, 0, vcc_lo
	v_fma_mix_f32 v141, v68, v70, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v68, v143, 0, v142
	v_add_f32_e32 v70, v70, v138
	s_wait_dscnt 0x0
	v_fma_mix_f32 v139, v71, v67, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v138, v71, v216, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_max3_num_f32 v68, v68, v141, v140
	v_add_f32_e32 v70, v217, v70
	v_fma_mix_f32 v71, v72, v144, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v72, v215, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_max3_num_f32 v68, v68, v139, v138
	v_add_f32_e32 v67, v67, v70
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v145, v68, v71, v72
	v_dual_add_f32 v67, v216, v67 :: v_dual_mov_b32 v70, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v67, v144, v67
	v_permlanex16_b32 v70, v70, s8, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v68, v215, v67 :: v_dual_max_num_f32 v67, v70, v70
	v_mov_b32_e32 v70, v68
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v144, v145, v67
	v_permlanex16_b32 v70, v70, s8, 0xfedcba98
	v_mov_b32_e32 v67, v213
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_lt_f32_e32 0, v144
	s_cbranch_execz .LBB5_50
; %bb.57:                               ;   in Loop: Header=BB5_51 Depth=2
	v_div_scale_f32 v67, null, 0x43e00000, 0x43e00000, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v145, v67
	v_fma_f32 v215, -v67, v145, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v145, v215, v145
	v_div_scale_f32 v215, vcc_lo, v144, 0x43e00000, v144
	v_mul_f32_e32 v216, v215, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v217, -v67, v216, v215
	v_fmac_f32_e32 v216, v217, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v67, -v67, v216, v215
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v67, v67, v145, v216
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v67, v67, 0x43e00000, v144
	v_max_num_f32_e32 v67, 0x1f800000, v67
	s_branch .LBB5_50
.LBB5_58:                               ;   in Loop: Header=BB5_51 Depth=2
	v_mov_b32_e32 v66, v214
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v214, v66
	s_add_co_i32 s15, s15, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s15, 4
	s_cbranch_scc0 .LBB5_51
	s_branch .LBB5_14
.LBB5_59:
	v_mov_b32_e32 v137, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v213, 1.0 :: v_dual_mov_b32 v144, v137
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_dual_mov_b32 v129, v137 :: v_dual_mov_b32 v130, v138
	v_dual_mov_b32 v121, v137 :: v_dual_mov_b32 v122, v138
	v_dual_mov_b32 v113, v137 :: v_dual_mov_b32 v114, v138
	v_dual_mov_b32 v105, v137 :: v_dual_mov_b32 v106, v138
	v_dual_mov_b32 v97, v137 :: v_dual_mov_b32 v98, v138
	v_dual_mov_b32 v89, v137 :: v_dual_mov_b32 v90, v138
	v_dual_mov_b32 v81, v137 :: v_dual_mov_b32 v82, v138
	v_dual_mov_b32 v73, v137 :: v_dual_mov_b32 v74, v138
	v_dual_mov_b32 v57, v137 :: v_dual_mov_b32 v58, v138
	v_dual_mov_b32 v49, v137 :: v_dual_mov_b32 v50, v138
	v_dual_mov_b32 v41, v137 :: v_dual_mov_b32 v42, v138
	v_dual_mov_b32 v33, v137 :: v_dual_mov_b32 v34, v138
	v_dual_mov_b32 v25, v137 :: v_dual_mov_b32 v26, v138
	v_dual_mov_b32 v17, v137 :: v_dual_mov_b32 v18, v138
	v_dual_mov_b32 v9, v137 :: v_dual_mov_b32 v10, v138
	v_dual_mov_b32 v1, v137 :: v_dual_mov_b32 v2, v138
	v_dual_mov_b32 v131, v139 :: v_dual_mov_b32 v132, v140
	v_dual_mov_b32 v133, v141 :: v_dual_mov_b32 v134, v142
	v_dual_mov_b32 v135, v143 :: v_dual_mov_b32 v136, v144
	v_dual_mov_b32 v123, v139 :: v_dual_mov_b32 v124, v140
	v_dual_mov_b32 v125, v141 :: v_dual_mov_b32 v126, v142
	v_dual_mov_b32 v127, v143 :: v_dual_mov_b32 v128, v144
	v_dual_mov_b32 v115, v139 :: v_dual_mov_b32 v116, v140
	v_dual_mov_b32 v117, v141 :: v_dual_mov_b32 v118, v142
	v_dual_mov_b32 v119, v143 :: v_dual_mov_b32 v120, v144
	v_dual_mov_b32 v107, v139 :: v_dual_mov_b32 v108, v140
	v_dual_mov_b32 v109, v141 :: v_dual_mov_b32 v110, v142
	v_dual_mov_b32 v111, v143 :: v_dual_mov_b32 v112, v144
	v_dual_mov_b32 v99, v139 :: v_dual_mov_b32 v100, v140
	v_dual_mov_b32 v101, v141 :: v_dual_mov_b32 v102, v142
	v_dual_mov_b32 v103, v143 :: v_dual_mov_b32 v104, v144
	v_dual_mov_b32 v91, v139 :: v_dual_mov_b32 v92, v140
	v_dual_mov_b32 v93, v141 :: v_dual_mov_b32 v94, v142
	v_dual_mov_b32 v95, v143 :: v_dual_mov_b32 v96, v144
	v_dual_mov_b32 v83, v139 :: v_dual_mov_b32 v84, v140
	v_dual_mov_b32 v85, v141 :: v_dual_mov_b32 v86, v142
	v_dual_mov_b32 v87, v143 :: v_dual_mov_b32 v88, v144
	v_dual_mov_b32 v75, v139 :: v_dual_mov_b32 v76, v140
	v_dual_mov_b32 v77, v141 :: v_dual_mov_b32 v78, v142
	v_dual_mov_b32 v79, v143 :: v_dual_mov_b32 v80, v144
	v_dual_mov_b32 v59, v139 :: v_dual_mov_b32 v60, v140
	v_dual_mov_b32 v61, v141 :: v_dual_mov_b32 v62, v142
	v_dual_mov_b32 v63, v143 :: v_dual_mov_b32 v64, v144
	v_dual_mov_b32 v51, v139 :: v_dual_mov_b32 v52, v140
	v_dual_mov_b32 v53, v141 :: v_dual_mov_b32 v54, v142
	v_dual_mov_b32 v55, v143 :: v_dual_mov_b32 v56, v144
	v_dual_mov_b32 v43, v139 :: v_dual_mov_b32 v44, v140
	v_dual_mov_b32 v45, v141 :: v_dual_mov_b32 v46, v142
	v_dual_mov_b32 v47, v143 :: v_dual_mov_b32 v48, v144
	v_dual_mov_b32 v35, v139 :: v_dual_mov_b32 v36, v140
	v_dual_mov_b32 v37, v141 :: v_dual_mov_b32 v38, v142
	v_dual_mov_b32 v39, v143 :: v_dual_mov_b32 v40, v144
	v_dual_mov_b32 v27, v139 :: v_dual_mov_b32 v28, v140
	v_dual_mov_b32 v29, v141 :: v_dual_mov_b32 v30, v142
	v_dual_mov_b32 v31, v143 :: v_dual_mov_b32 v32, v144
	v_dual_mov_b32 v19, v139 :: v_dual_mov_b32 v20, v140
	v_dual_mov_b32 v21, v141 :: v_dual_mov_b32 v22, v142
	v_dual_mov_b32 v23, v143 :: v_dual_mov_b32 v24, v144
	v_dual_mov_b32 v11, v139 :: v_dual_mov_b32 v12, v140
	v_dual_mov_b32 v13, v141 :: v_dual_mov_b32 v14, v142
	v_dual_mov_b32 v15, v143 :: v_dual_mov_b32 v16, v144
	v_dual_mov_b32 v3, v139 :: v_dual_mov_b32 v4, v140
	v_dual_mov_b32 v5, v141 :: v_dual_mov_b32 v6, v142
	v_dual_mov_b32 v7, v143 :: v_dual_mov_b32 v8, v144
.LBB5_60:
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_62
; %bb.61:
	v_div_scale_f32 v0, null, v137, v137, 1.0
	v_div_scale_f32 v67, vcc_lo, 1.0, v137, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v65, v0
	v_fma_f32 v66, -v0, v65, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v65, v66, v65
	v_mul_f32_e32 v66, v67, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v68, -v0, v66, v67
	v_fmac_f32_e32 v66, v68, v65
	v_mul_lo_u32 v68, 0x1800, v147
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v0, -v0, v66, v67
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v0, v0, v65, v66
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v65, v146, 8, v68
	v_cmp_lt_f32_e32 vcc_lo, 0, v137
	v_mov_b32_e32 v66, 0
	v_div_fixup_f32 v0, v0, v137, 1.0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_or_b32 v65, v153, 3, v65
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	v_mul_f32_e32 v139, v213, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v137, vcc_lo, s16, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, s17, v66, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v65, v129, v139 :: v_dual_mul_f32 v66, v130, v139
	v_mul_f32_e32 v69, v133, v139
	v_dual_mul_f32 v67, v131, v139 :: v_dual_mul_f32 v68, v132, v139
	v_dual_mul_f32 v71, v135, v139 :: v_dual_mul_f32 v70, v134, v139
	v_dual_mul_f32 v121, v121, v139 :: v_dual_mul_f32 v72, v136, v139
	v_dual_mul_f32 v123, v123, v139 :: v_dual_mul_f32 v122, v122, v139
	v_dual_mul_f32 v125, v125, v139 :: v_dual_mul_f32 v124, v124, v139
	v_dual_mul_f32 v127, v127, v139 :: v_dual_mul_f32 v126, v126, v139
	v_dual_mul_f32 v113, v113, v139 :: v_dual_mul_f32 v128, v128, v139
	v_dual_mul_f32 v115, v115, v139 :: v_dual_mul_f32 v110, v110, v139
	v_dual_mul_f32 v97, v97, v139 :: v_dual_mul_f32 v112, v112, v139
	v_dual_mul_f32 v99, v99, v139 :: v_dual_mul_f32 v98, v98, v139
	v_dual_mul_f32 v101, v101, v139 :: v_dual_mul_f32 v100, v100, v139
	v_dual_mul_f32 v103, v103, v139 :: v_dual_mul_f32 v114, v114, v139
	v_dual_mul_f32 v117, v117, v139 :: v_dual_mul_f32 v116, v116, v139
	v_dual_mul_f32 v119, v119, v139 :: v_dual_mul_f32 v102, v102, v139
	v_mul_f32_e32 v104, v104, v139
	v_dual_mul_f32 v118, v118, v139 :: v_dual_mul_f32 v105, v105, v139
	v_dual_mul_f32 v120, v120, v139 :: v_dual_mul_f32 v107, v107, v139
	v_dual_mul_f32 v106, v106, v139 :: v_dual_mul_f32 v109, v109, v139
	v_dual_mul_f32 v108, v108, v139 :: v_dual_mul_f32 v111, v111, v139
	s_clause 0x7
	global_store_b128 v[137:138], v[65:68], off
	global_store_b128 v[137:138], v[69:72], off offset:16
	global_store_b128 v[137:138], v[121:124], off offset:64
	global_store_b128 v[137:138], v[125:128], off offset:80
	global_store_b128 v[137:138], v[113:116], off offset:128
	global_store_b128 v[137:138], v[117:120], off offset:144
	global_store_b128 v[137:138], v[105:108], off offset:192
	global_store_b128 v[137:138], v[109:112], off offset:208
	v_dual_mul_f32 v65, v89, v139 :: v_dual_mul_f32 v66, v90, v139
	v_mul_f32_e32 v69, v93, v139
	v_dual_mul_f32 v67, v91, v139 :: v_dual_mul_f32 v68, v92, v139
	v_dual_mul_f32 v71, v95, v139 :: v_dual_mul_f32 v70, v94, v139
	v_dual_mul_f32 v81, v81, v139 :: v_dual_mul_f32 v72, v96, v139
	v_dual_mul_f32 v83, v83, v139 :: v_dual_mul_f32 v82, v82, v139
	v_dual_mul_f32 v85, v85, v139 :: v_dual_mul_f32 v84, v84, v139
	v_dual_mul_f32 v87, v87, v139 :: v_dual_mul_f32 v86, v86, v139
	v_mul_f32_e32 v88, v88, v139
	s_clause 0x5
	global_store_b128 v[137:138], v[97:100], off offset:256
	global_store_b128 v[137:138], v[101:104], off offset:272
	global_store_b128 v[137:138], v[65:68], off offset:320
	global_store_b128 v[137:138], v[69:72], off offset:336
	global_store_b128 v[137:138], v[81:84], off offset:384
	global_store_b128 v[137:138], v[85:88], off offset:400
	v_dual_mul_f32 v65, v73, v139 :: v_dual_mul_f32 v66, v74, v139
	v_mul_f32_e32 v69, v77, v139
	v_dual_mul_f32 v67, v75, v139 :: v_dual_mul_f32 v68, v76, v139
	v_dual_mul_f32 v71, v79, v139 :: v_dual_mul_f32 v54, v54, v139
	v_dual_mul_f32 v41, v41, v139 :: v_dual_mul_f32 v56, v56, v139
	v_dual_mul_f32 v43, v43, v139 :: v_dual_mul_f32 v42, v42, v139
	v_dual_mul_f32 v45, v45, v139 :: v_dual_mul_f32 v44, v44, v139
	v_dual_mul_f32 v47, v47, v139 :: v_dual_mul_f32 v30, v30, v139
	v_dual_mul_f32 v17, v17, v139 :: v_dual_mul_f32 v32, v32, v139
	v_dual_mul_f32 v19, v19, v139 :: v_dual_mul_f32 v18, v18, v139
	v_dual_mul_f32 v21, v21, v139 :: v_dual_mul_f32 v20, v20, v139
	v_dual_mul_f32 v23, v23, v139 :: v_dual_mul_f32 v70, v78, v139
	v_dual_mul_f32 v57, v57, v139 :: v_dual_mul_f32 v72, v80, v139
	v_dual_mul_f32 v59, v59, v139 :: v_dual_mul_f32 v46, v46, v139
	v_dual_mul_f32 v33, v33, v139 :: v_dual_mul_f32 v48, v48, v139
	v_dual_mul_f32 v35, v35, v139 :: v_dual_mul_f32 v22, v22, v139
	v_dual_mul_f32 v9, v9, v139 :: v_dual_mul_f32 v24, v24, v139
	v_dual_mul_f32 v11, v11, v139 :: v_dual_mul_f32 v58, v58, v139
	v_dual_mul_f32 v61, v61, v139 :: v_dual_mul_f32 v60, v60, v139
	v_dual_mul_f32 v63, v63, v139 :: v_dual_mul_f32 v34, v34, v139
	v_dual_mul_f32 v37, v37, v139 :: v_dual_mul_f32 v36, v36, v139
	v_dual_mul_f32 v39, v39, v139 :: v_dual_mul_f32 v10, v10, v139
	v_dual_mul_f32 v13, v13, v139 :: v_dual_mul_f32 v12, v12, v139
	v_dual_mul_f32 v15, v15, v139 :: v_dual_mul_f32 v62, v62, v139
	v_dual_mul_f32 v49, v49, v139 :: v_dual_mul_f32 v64, v64, v139
	v_dual_mul_f32 v51, v51, v139 :: v_dual_mul_f32 v38, v38, v139
	v_dual_mul_f32 v25, v25, v139 :: v_dual_mul_f32 v40, v40, v139
	v_dual_mul_f32 v27, v27, v139 :: v_dual_mul_f32 v14, v14, v139
	v_mul_f32_e32 v16, v16, v139
	v_dual_mul_f32 v50, v50, v139 :: v_dual_mul_f32 v53, v53, v139
	v_dual_mul_f32 v52, v52, v139 :: v_dual_mul_f32 v55, v55, v139
	v_dual_mul_f32 v26, v26, v139 :: v_dual_mul_f32 v29, v29, v139
	v_dual_mul_f32 v28, v28, v139 :: v_dual_mul_f32 v31, v31, v139
	v_dual_mul_f32 v0, v1, v139 :: v_dual_mul_f32 v1, v2, v139
	v_dual_mul_f32 v2, v3, v139 :: v_dual_mul_f32 v3, v4, v139
	s_clause 0xb
	global_store_b128 v[137:138], v[65:68], off offset:448
	global_store_b128 v[137:138], v[69:72], off offset:464
	global_store_b128 v[137:138], v[57:60], off offset:512
	global_store_b128 v[137:138], v[61:64], off offset:528
	global_store_b128 v[137:138], v[49:52], off offset:576
	global_store_b128 v[137:138], v[53:56], off offset:592
	global_store_b128 v[137:138], v[41:44], off offset:640
	global_store_b128 v[137:138], v[45:48], off offset:656
	global_store_b128 v[137:138], v[33:36], off offset:704
	global_store_b128 v[137:138], v[37:40], off offset:720
	global_store_b128 v[137:138], v[25:28], off offset:768
	global_store_b128 v[137:138], v[29:32], off offset:784
	v_dual_mul_f32 v4, v5, v139 :: v_dual_mul_f32 v5, v6, v139
	v_dual_mul_f32 v6, v7, v139 :: v_dual_mul_f32 v7, v8, v139
	s_clause 0x5
	global_store_b128 v[137:138], v[17:20], off offset:832
	global_store_b128 v[137:138], v[21:24], off offset:848
	global_store_b128 v[137:138], v[9:12], off offset:896
	global_store_b128 v[137:138], v[13:16], off offset:912
	global_store_b128 v[137:138], v[0:3], off offset:960
	global_store_b128 v[137:138], v[4:7], off offset:976
.LBB5_62:
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
		.amdhsa_kernarg_size 68
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
; codeLenInByte = 12728
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
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x30
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
	s_cbranch_vccnz .LBB6_81
; %bb.1:
	s_and_b32 s23, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s23, 3
	s_cbranch_scc1 .LBB6_81
; %bb.2:
	s_load_b32 s21, s[0:1], 0x40
	s_lshr_b32 s2, ttmp7, 7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, 0x1fffe00
	s_cmp_gt_i32 s7, 0x200
	s_cselect_b32 s20, s2, 0
	s_cmp_le_i32 s7, s20
	s_cbranch_scc1 .LBB6_81
; %bb.3:
	s_sub_co_i32 s2, s7, s20
	s_mul_i32 s24, ttmp9, 0x180
	s_min_i32 s25, s2, 0x200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s22, s25, 6
	s_cmp_ge_i32 s24, s22
	s_cbranch_scc1 .LBB6_81
; %bb.4:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v152, s21 :: v_dual_and_b32 v151, 15, v0
	v_lshrrev_b32_e32 v8, 5, v0
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b128 s[16:19], s[0:1], 0x20
	v_or_b32_e32 v1, s24, v151
	v_bfe_u32 v150, v0, 4, 1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v3, v8, 4, v1
	v_mul_hi_i32 v1, 0x2aaaaaab, v3
	v_cmp_le_i32_e64 s1, s22, v3
	v_cmp_gt_i32_e64 s0, s22, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v2, 31, v1
	v_add_nc_u32_e32 v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v1, v2, 6
	v_sub_nc_u32_e32 v1, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[140:141], null, s23, 6, v[1:2]
	v_add_nc_u32_e32 v141, s20, v2
	v_mad_co_u64_u32 v[1:2], null, v141, 24, v[140:141]
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 8, v1
	s_and_saveexec_b32 s26, s0
	s_cbranch_execz .LBB6_8
; %bb.5:
	v_mov_b32_e32 v2, 0
	s_mov_b32 s2, 0x76543210
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[133:134], 10, v[1:2]
	v_lshlrev_b32_e32 v1, 5, v150
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, vcc_lo, s8, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v134, vcc_lo
	v_add_co_u32 v129, vcc_lo, v3, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v130, null, 0, v4, vcc_lo
	s_clause 0x1f
	global_load_b128 v[3:6], v[129:130], off
	global_load_b128 v[9:12], v[129:130], off offset:16
	global_load_b128 v[13:16], v[129:130], off offset:64
	global_load_b128 v[17:20], v[129:130], off offset:80
	global_load_b128 v[21:24], v[129:130], off offset:128
	global_load_b128 v[25:28], v[129:130], off offset:144
	global_load_b128 v[29:32], v[129:130], off offset:192
	global_load_b128 v[33:36], v[129:130], off offset:208
	global_load_b128 v[37:40], v[129:130], off offset:256
	global_load_b128 v[41:44], v[129:130], off offset:272
	global_load_b128 v[45:48], v[129:130], off offset:320
	global_load_b128 v[49:52], v[129:130], off offset:336
	global_load_b128 v[53:56], v[129:130], off offset:384
	global_load_b128 v[57:60], v[129:130], off offset:400
	global_load_b128 v[61:64], v[129:130], off offset:448
	global_load_b128 v[65:68], v[129:130], off offset:464
	global_load_b128 v[69:72], v[129:130], off offset:512
	global_load_b128 v[73:76], v[129:130], off offset:528
	global_load_b128 v[77:80], v[129:130], off offset:576
	global_load_b128 v[81:84], v[129:130], off offset:592
	global_load_b128 v[85:88], v[129:130], off offset:640
	global_load_b128 v[89:92], v[129:130], off offset:656
	global_load_b128 v[93:96], v[129:130], off offset:704
	global_load_b128 v[97:100], v[129:130], off offset:720
	global_load_b128 v[101:104], v[129:130], off offset:768
	global_load_b128 v[105:108], v[129:130], off offset:784
	global_load_b128 v[109:112], v[129:130], off offset:832
	global_load_b128 v[113:116], v[129:130], off offset:848
	global_load_b128 v[117:120], v[129:130], off offset:896
	global_load_b128 v[121:124], v[129:130], off offset:912
	global_load_b128 v[125:128], v[129:130], off offset:960
	global_load_b128 v[129:132], v[129:130], off offset:976
	s_wait_loadcnt 0x1f
	v_max3_num_f32 v1, |v3|, 0, |v4|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v5|, |v6|
	s_wait_loadcnt 0x1e
	v_max3_num_f32 v1, v1, |v9|, |v10|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v11|, |v12|
	s_wait_loadcnt 0x1d
	v_max3_num_f32 v1, v1, |v13|, |v14|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v15|, |v16|
	s_wait_loadcnt 0x1c
	v_max3_num_f32 v1, v1, |v17|, |v18|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v19|, |v20|
	s_wait_loadcnt 0x1b
	v_max3_num_f32 v1, v1, |v21|, |v22|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v23|, |v24|
	s_wait_loadcnt 0x1a
	v_max3_num_f32 v1, v1, |v25|, |v26|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v27|, |v28|
	s_wait_loadcnt 0x19
	v_max3_num_f32 v1, v1, |v29|, |v30|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v31|, |v32|
	s_wait_loadcnt 0x18
	v_max3_num_f32 v1, v1, |v33|, |v34|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v35|, |v36|
	s_wait_loadcnt 0x17
	v_max3_num_f32 v1, v1, |v37|, |v38|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v39|, |v40|
	s_wait_loadcnt 0x16
	v_max3_num_f32 v1, v1, |v41|, |v42|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v43|, |v44|
	s_wait_loadcnt 0x15
	v_max3_num_f32 v1, v1, |v45|, |v46|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v47|, |v48|
	s_wait_loadcnt 0x14
	v_max3_num_f32 v1, v1, |v49|, |v50|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v51|, |v52|
	s_wait_loadcnt 0x13
	v_max3_num_f32 v1, v1, |v53|, |v54|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v55|, |v56|
	s_wait_loadcnt 0x12
	v_max3_num_f32 v1, v1, |v57|, |v58|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v59|, |v60|
	s_wait_loadcnt 0x11
	v_max3_num_f32 v1, v1, |v61|, |v62|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v63|, |v64|
	s_wait_loadcnt 0x10
	v_max3_num_f32 v1, v1, |v65|, |v66|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v67|, |v68|
	s_wait_loadcnt 0xf
	v_max3_num_f32 v1, v1, |v69|, |v70|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v71|, |v72|
	s_wait_loadcnt 0xe
	v_max3_num_f32 v1, v1, |v73|, |v74|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v75|, |v76|
	s_wait_loadcnt 0xd
	v_max3_num_f32 v1, v1, |v77|, |v78|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v79|, |v80|
	s_wait_loadcnt 0xc
	v_max3_num_f32 v1, v1, |v81|, |v82|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v83|, |v84|
	s_wait_loadcnt 0xb
	v_max3_num_f32 v1, v1, |v85|, |v86|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v87|, |v88|
	s_wait_loadcnt 0xa
	v_max3_num_f32 v1, v1, |v89|, |v90|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v91|, |v92|
	s_wait_loadcnt 0x9
	v_max3_num_f32 v1, v1, |v93|, |v94|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v95|, |v96|
	s_wait_loadcnt 0x8
	v_max3_num_f32 v1, v1, |v97|, |v98|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v99|, |v100|
	s_wait_loadcnt 0x7
	v_max3_num_f32 v1, v1, |v101|, |v102|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v103|, |v104|
	s_wait_loadcnt 0x6
	v_max3_num_f32 v1, v1, |v105|, |v106|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v107|, |v108|
	s_wait_loadcnt 0x5
	v_max3_num_f32 v1, v1, |v109|, |v110|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v111|, |v112|
	s_wait_loadcnt 0x4
	v_max3_num_f32 v1, v1, |v113|, |v114|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v115|, |v116|
	s_wait_loadcnt 0x3
	v_max3_num_f32 v1, v1, |v117|, |v118|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v119|, |v120|
	s_wait_loadcnt 0x2
	v_max3_num_f32 v1, v1, |v121|, |v122|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v123|, |v124|
	s_wait_loadcnt 0x1
	v_max3_num_f32 v1, v1, |v125|, |v126|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v127|, |v128|
	s_wait_loadcnt 0x0
	v_max3_num_f32 v1, v1, |v129|, |v130|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, |v131|, |v132|
	v_mov_b32_e32 v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v3, v3, s2, 0xfedcba98
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v1, v1, v3
	v_div_scale_f32 v3, null, 0x43e00000, 0x43e00000, v1
	v_div_scale_f32 v6, vcc_lo, v1, 0x43e00000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v4, v3
	v_fma_f32 v5, -v3, v4, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v5, v4
	v_mul_f32_e32 v5, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v9, -v3, v5, v6
	v_fmac_f32_e32 v5, v9, v4
	v_lshl_or_b32 v9, v150, 5, v133
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f32 v3, -v3, v5, v6
	v_lshl_or_b32 v6, v150, 3, v7
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v5, s2, s10, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s11, 0, s2
	v_div_fixup_f32 v10, v3, 0x43e00000, v1
	v_add_co_u32 v3, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s9, v134, vcc_lo
	v_cmp_neq_f32_e32 vcc_lo, 0, v1
	s_mov_b32 s9, 16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 1.0, v10, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, 4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
.LBB6_6:                                ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[9:12], v[3:4], off
	global_load_b128 v[13:16], v[3:4], off offset:16
	v_add_co_u32 v3, vcc_lo, v3, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	v_mov_b16_e32 v18.l, v2.l
	v_mov_b16_e32 v18.h, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s9, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s9, 0
	v_mov_b16_e32 v17.l, v18.l
	v_mov_b16_e32 v17.h, v18.h
	s_wait_loadcnt 0x1
	v_div_scale_f32 v19, null, v1, v1, v9
	v_div_scale_f32 v21, null, v1, v1, v10
	v_div_scale_f32 v23, null, v1, v1, v11
	v_div_scale_f32 v25, null, v1, v1, v12
	s_wait_loadcnt 0x0
	v_div_scale_f32 v27, null, v1, v1, v13
	v_rcp_f32_e32 v35, v19
	v_rcp_f32_e32 v36, v21
	v_rcp_f32_e32 v37, v23
	v_div_scale_f32 v29, null, v1, v1, v14
	v_rcp_f32_e32 v38, v25
	v_div_scale_f32 v31, null, v1, v1, v15
	v_rcp_f32_e32 v39, v27
	v_div_scale_f32 v33, null, v1, v1, v16
	v_rcp_f32_e32 v40, v29
	v_fma_f32 v43, -v19, v35, 1.0
	v_fma_f32 v44, -v21, v36, 1.0
	v_rcp_f32_e32 v41, v31
	v_fma_f32 v45, -v23, v37, 1.0
	v_div_scale_f32 v20, vcc_lo, v9, v1, v9
	v_rcp_f32_e32 v42, v33
	v_dual_fmac_f32 v35, v43, v35 :: v_dual_fmac_f32 v36, v44, v36
	v_fma_f32 v46, -v25, v38, 1.0
	v_div_scale_f32 v22, s2, v10, v1, v10
	v_fma_f32 v47, -v27, v39, 1.0
	v_div_scale_f32 v24, s3, v11, v1, v11
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v37, v45, v37 :: v_dual_fmac_f32 v38, v46, v38
	v_fma_f32 v48, -v29, v40, 1.0
	v_dual_mul_f32 v43, v20, v35 :: v_dual_mul_f32 v44, v22, v36
	v_div_scale_f32 v26, s4, v12, v1, v12
	v_fma_f32 v49, -v31, v41, 1.0
	v_div_scale_f32 v28, s5, v13, v1, v13
	v_dual_fmac_f32 v39, v47, v39 :: v_dual_fmac_f32 v40, v48, v40
	v_fma_f32 v50, -v33, v42, 1.0
	v_dual_mul_f32 v45, v24, v37 :: v_dual_mul_f32 v46, v26, v38
	v_div_scale_f32 v30, s6, v14, v1, v14
	v_fma_f32 v51, -v19, v43, v20
	v_div_scale_f32 v32, s7, v15, v1, v15
	v_dual_fmac_f32 v41, v49, v41 :: v_dual_fmac_f32 v42, v50, v42
	v_fma_f32 v52, -v21, v44, v22
	v_dual_mul_f32 v47, v28, v39 :: v_dual_mul_f32 v48, v30, v40
	v_div_scale_f32 v34, s8, v16, v1, v16
	v_fma_f32 v53, -v23, v45, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v43, v51, v35 :: v_dual_fmac_f32 v44, v52, v36
	v_fma_f32 v54, -v25, v46, v26
	v_dual_mul_f32 v49, v32, v41 :: v_dual_mul_f32 v50, v34, v42
	v_fma_f32 v55, -v27, v47, v28
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v45, v53, v37 :: v_dual_fmac_f32 v46, v54, v38
	v_fma_f32 v56, -v29, v48, v30
	v_fma_f32 v19, -v19, v43, v20
	v_fma_f32 v57, -v31, v49, v32
	v_fma_f32 v20, -v21, v44, v22
	v_dual_fmac_f32 v47, v55, v39 :: v_dual_fmac_f32 v48, v56, v40
	v_fma_f32 v58, -v33, v50, v34
	v_fma_f32 v21, -v23, v45, v24
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v19, v19, v35, v43
	s_mov_b32 vcc_lo, s2
	v_fma_f32 v22, -v25, v46, v26
	v_dual_fmac_f32 v49, v57, v41 :: v_dual_fmac_f32 v50, v58, v42
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v20, v20, v36, v44
	s_mov_b32 vcc_lo, s3
	v_fma_f32 v23, -v27, v47, v28
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v21, v21, v37, v45
	s_mov_b32 vcc_lo, s4
	v_fma_f32 v24, -v29, v48, v30
	v_div_fixup_f32 v9, v19, v1, v9
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v19, v22, v38, v46
	s_mov_b32 vcc_lo, s5
	v_fma_f32 v25, -v31, v49, v32
	v_div_fixup_f32 v10, v20, v1, v10
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v20, v23, v39, v47
	s_mov_b32 vcc_lo, s6
	v_fma_f32 v26, -v33, v50, v34
	v_div_fixup_f32 v11, v21, v1, v11
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v21, v24, v40, v48
	s_mov_b32 vcc_lo, s7
	v_cvt_pk_fp8_f32 v17.l, v9, v10
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v9, v25, v41, v49
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v12, v19, v1, v12
	v_div_fixup_f32 v10, v20, v1, v13
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v13, v26, v42, v50
	v_div_fixup_f32 v14, v21, v1, v14
	v_div_fixup_f32 v9, v9, v1, v15
	v_cvt_pk_fp8_f32 v17.h, v11, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v11, v13, v1, v16
	v_cvt_pk_fp8_f32 v18.l, v10, v14
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_fp8_f32 v18.h, v9, v11
	global_store_b64 v[5:6], v[17:18], off offset:-4
	v_add_co_u32 v5, vcc_lo, v5, 16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	s_cbranch_scc0 .LBB6_6
; %bb.7:
	v_mul_f32_e32 v152, s21, v1
.LBB6_8:
	s_or_b32 exec_lo, exec_lo, s26
	v_dual_mov_b32 v4, -1 :: v_dual_and_b32 v153, 31, v0
	v_bfrev_b32_e32 v3, -2
	s_add_co_i32 s25, s25, s20
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v1, s20, v153
	v_lshl_add_u32 v1, ttmp9, 6, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_ashrrev_i32_e32 v2, 31, v1
	s_wait_alu depctr_sa_sdst(0)
	v_cmpx_gt_i32_e64 s25, v1
	s_cbranch_execz .LBB6_10
; %bb.9:
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, vcc_lo, s18, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s19, v4, vcc_lo
	global_load_b32 v3, v[3:4], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v4, v3
.LBB6_10:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 32, v1
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s25, v5
	s_cbranch_execz .LBB6_12
; %bb.11:
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s18, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s19, v2, vcc_lo
	global_load_b32 v1, v[1:2], off offset:128
	s_wait_loadcnt 0x0
	v_max_i32_e32 v4, v4, v1
	v_min_i32_e32 v3, v3, v1
.LBB6_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mbcnt_lo_u32_b32 v1, -1, 0
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v2, 16, v1
	v_xor_b32_e32 v6, 8, v1
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v1, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	v_lshlrev_b32_e32 v2, 2, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v1, v6, vcc_lo
	ds_bpermute_b32 v5, v2, v4
	ds_bpermute_b32 v2, v2, v3
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v3, v2
	ds_bpermute_b32 v3, v6, v4
	ds_bpermute_b32 v5, v6, v2
	v_xor_b32_e32 v6, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v1, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v4, v3
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v2, v5
	ds_bpermute_b32 v4, v6, v3
	ds_bpermute_b32 v5, v6, v2
	v_xor_b32_e32 v6, 2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v1, v6, vcc_lo
	v_lshlrev_b32_e32 v154, 2, v6
	v_xor_b32_e32 v6, 1, v1
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v4
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v2, v5
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v6, vcc_lo
	ds_bpermute_b32 v4, v154, v3
	ds_bpermute_b32 v5, v154, v2
	v_lshlrev_b32_e32 v155, 2, v1
	s_wait_dscnt 0x1
	v_max_i32_e32 v1, v3, v4
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v2, v5
	ds_bpermute_b32 v3, v155, v1
	ds_bpermute_b32 v4, v155, v2
	s_wait_dscnt 0x1
	v_max_i32_e32 v1, v1, v3
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v2, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s20, v1
	v_readfirstlane_b32 s2, v2
	v_cmpx_eq_u32_e32 0, v8
	s_cbranch_execz .LBB6_14
; %bb.13:
	s_mov_b32 m0, 0x80001
	s_barrier_init m0 
	s_mov_b32 m0, 0x80002
	s_barrier_init m0 
	s_mov_b32 m0, 0x80003
	s_barrier_init m0 
.LBB6_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s7, 0
	s_cmp_lt_i32 s20, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB6_78
; %bb.15:
	v_lshrrev_b32_e32 v143, 8, v0
	v_dual_mov_b32 v156, 0 :: v_dual_and_b32 v3, 7, v8
	v_ashrrev_i32_e32 v142, 31, v141
	s_addk_co_i32 s24, 0x17f
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v11, 0x5040100 :: v_dual_lshlrev_b32 v4, 12, v143
	v_dual_mov_b32 v6, 0x6020400 :: v_dual_lshlrev_b32 v5, 11, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[1:2], 2, v[141:142]
	v_lshlrev_b32_e32 v142, 3, v150
	s_min_i32 s21, s20, s2
	v_add3_u32 v4, 0, v4, v5
	v_mad_u32_u24 v5, v3, 3, v143
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s24, s22
	s_wait_kmcnt 0x0
	v_add_co_u32 v144, vcc_lo, s18, v1
	v_lshl_or_b32 v1, v3, 5, v142
	v_lshl_add_u32 v9, v5, 9, 0
	v_and_b32_e32 v5, 1, v0
	s_cselect_b32 s22, -1, 0
	s_lshl_b32 s6, s23, 8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v145, null, s19, v2, vcc_lo
	v_and_b32_e32 v2, 3, v0
	v_lshrrev_b32_e32 v10, 2, v151
	s_add_nc_u64 s[8:9], s[12:13], s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v146, s5, s8, v1
	v_lshlrev_b32_e32 v1, 3, v153
	v_lshl_or_b32 v5, v2, 6, v10
	v_add_co_ci_u32_e64 v147, null, s9, 0, s5
	s_add_nc_u64 s[8:9], s[14:15], s[6:7]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v158, 0x3070105, v6, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v148, s5, s8, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v149, null, s9, 0, s5
	v_lshlrev_b32_e32 v12, 3, v2
	v_cmp_gt_u32_e64 s5, 2, v2
	v_cndmask_b32_e64 v6, v7, 0, s1
	v_dual_mov_b32 v206, 1.0 :: v_dual_lshlrev_b32 v7, 2, v5
	v_lshlrev_b32_e32 v2, 5, v2
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v159, 0x3020706, v11, s5
	v_and_b32_e32 v11, 16, v0
	v_lshlrev_b32_e32 v0, 4, v0
	v_lshlrev_b32_e32 v157, 3, v3
	v_add3_u32 v175, v4, v7, v2
	v_or_b32_e32 v7, 12, v5
	v_dual_mov_b32 v207, 0xff800000 :: v_dual_add_nc_u32 v162, v4, v1
	v_and_b32_e32 v176, 0x31f0, v0
	v_or_b32_e32 v0, 16, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_xor_b32_e32 v7, v7, v12
	v_xad_u32 v163, 0x120, v1, v4
	v_xad_u32 v164, 0x124, v1, v4
	v_xad_u32 v165, 0x240, v1, v4
	v_xor_b32_e32 v0, v0, v12
	v_lshl_add_u32 v179, v7, 2, v4
	v_or_b32_e32 v7, 0x114, v5
	v_xad_u32 v166, 0x244, v1, v4
	v_xad_u32 v167, 0x360, v1, v4
	v_lshl_add_u32 v181, v0, 2, v4
	v_or_b32_e32 v0, 0x118, v5
	v_xor_b32_e32 v7, v7, v12
	v_xad_u32 v168, 0x364, v1, v4
	v_xad_u32 v169, 0x520, v1, v4
	v_xad_u32 v170, 0x524, v1, v4
	v_xor_b32_e32 v0, v0, v12
	v_lshl_add_u32 v184, v7, 2, v4
	v_or_b32_e32 v7, 40, v5
	v_xad_u32 v171, 0x640, v1, v4
	v_xad_u32 v172, 0x644, v1, v4
	v_lshl_add_u32 v186, v0, 2, v4
	v_or_b32_e32 v0, 44, v5
	v_xor_b32_e32 v7, v7, v12
	v_xad_u32 v173, 0x760, v1, v4
	v_xad_u32 v174, 0x764, v1, v4
	v_or_b32_e32 v1, 8, v5
	v_xor_b32_e32 v0, v0, v12
	v_lshl_add_u32 v189, v7, 2, v4
	v_or_b32_e32 v7, 0x130, v5
	v_or_b32_e32 v2, 0x108, v5
	v_or_b32_e32 v13, 0x10c, v5
	v_lshl_add_u32 v191, v0, 2, v4
	v_or_b32_e32 v0, 0x134, v5
	v_xor_b32_e32 v7, v7, v12
	v_cmp_gt_u32_e64 s3, 2, v3
	v_cmp_eq_u32_e32 vcc_lo, 0, v3
	v_cmp_eq_u32_e64 s5, v150, v3
	v_xor_b32_e32 v0, v0, v12
	v_lshl_add_u32 v194, v7, 2, v4
	v_or_b32_e32 v7, 60, v5
	v_lshl_add_u32 v3, v143, 4, v153
	v_cmp_gt_u32_e64 s6, 16, v153
	v_lshl_add_u32 v196, v0, 2, v4
	v_mov_b32_e32 v0, 0
	v_xor_b32_e32 v7, v7, v12
	v_lshl_add_u32 v160, v3, 1, 0
	v_add_co_u32 v3, s8, s10, v6
	v_lshlrev_b32_e32 v8, 4, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v199, v7, 2, v4
	v_mov_b32_e32 v7, v0
	v_xor_b32_e32 v1, v1, v12
	v_xor_b32_e32 v2, v2, v12
	v_xor_b32_e32 v13, v13, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s11, 0, s8
	v_lshl_add_u32 v177, v1, 2, v4
	v_or_b32_e32 v1, 0x110, v5
	v_lshl_add_u32 v178, v2, 2, v4
	v_or_b32_e32 v2, 20, v5
	v_lshl_add_u32 v180, v13, 2, v4
	v_or_b32_e32 v13, 24, v5
	v_xor_b32_e32 v1, v1, v12
	v_lshl_add_u32 v10, v143, 5, 0
	v_xor_b32_e32 v2, v2, v12
	s_and_b32 s18, vcc_lo, s6
	v_xor_b32_e32 v13, v13, v12
	v_lshl_add_u32 v182, v1, 2, v4
	v_or_b32_e32 v1, 28, v5
	v_lshl_add_u32 v183, v2, 2, v4
	v_or_b32_e32 v2, 0x11c, v5
	v_lshl_add_u32 v185, v13, 2, v4
	v_or_b32_e32 v13, 0x128, v5
	v_xor_b32_e32 v1, v1, v12
	v_cmp_ne_u32_e64 s2, 0, v143
	v_xor_b32_e32 v2, v2, v12
	v_cmp_ne_u32_e64 s4, 1, v143
	v_xor_b32_e32 v13, v13, v12
	v_lshl_add_u32 v187, v1, 2, v4
	v_or_b32_e32 v1, 0x12c, v5
	v_lshl_add_u32 v188, v2, 2, v4
	v_or_b32_e32 v2, 48, v5
	v_lshl_add_u32 v190, v13, 2, v4
	v_or_b32_e32 v13, 52, v5
	v_xor_b32_e32 v1, v1, v12
	v_mov_b32_e32 v128, v0
	v_xor_b32_e32 v2, v2, v12
	s_lshl_b32 s6, s23, 1
	v_xor_b32_e32 v13, v13, v12
	v_lshl_add_u32 v192, v1, 2, v4
	v_or_b32_e32 v1, 56, v5
	v_lshl_add_u32 v193, v2, 2, v4
	v_or_b32_e32 v2, 0x138, v5
	v_or_b32_e32 v5, 0x13c, v5
	v_lshl_add_u32 v195, v13, 2, v4
	v_xor_b32_e32 v1, v1, v12
	s_add_nc_u64 s[8:9], s[12:13], s[6:7]
	v_xor_b32_e32 v2, v2, v12
	v_xor_b32_e32 v5, v5, v12
	s_add_nc_u64 s[10:11], s[14:15], s[6:7]
	v_lshl_add_u32 v197, v1, 2, v4
	v_mov_b32_e32 v1, v0
	v_add_co_u32 v201, vcc_lo, v3, v142
	v_lshl_add_u32 v198, v2, 2, v4
	v_lshl_add_u32 v200, v5, 2, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v202, null, 0, v6, vcc_lo
	v_dual_mov_b32 v2, v0 :: v_dual_mov_b32 v3, v0
	v_dual_mov_b32 v4, v0 :: v_dual_add_nc_u32 v161, 0, v8
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_add_nc_u32_e32 v203, v9, v8
	v_dual_mov_b32 v15, v7 :: v_dual_add_nc_u32 v204, v10, v11
	v_dual_mov_b32 v8, v0 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v16, v0 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v24, v0 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v32, v0 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v40, v0 :: v_dual_mov_b32 v55, v7
	v_dual_mov_b32 v48, v0 :: v_dual_mov_b32 v63, v7
	v_dual_mov_b32 v56, v0 :: v_dual_mov_b32 v71, v7
	v_dual_mov_b32 v64, v0 :: v_dual_mov_b32 v79, v7
	v_dual_mov_b32 v72, v0 :: v_dual_mov_b32 v87, v7
	v_dual_mov_b32 v80, v0 :: v_dual_mov_b32 v95, v7
	v_dual_mov_b32 v88, v0 :: v_dual_mov_b32 v103, v7
	v_dual_mov_b32 v96, v0 :: v_dual_mov_b32 v111, v7
	v_dual_mov_b32 v104, v0 :: v_dual_mov_b32 v119, v7
	v_dual_mov_b32 v112, v0 :: v_dual_mov_b32 v127, v7
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
	v_dual_mov_b32 v46, v6 :: v_dual_mov_b32 v45, v5
	v_dual_mov_b32 v44, v4 :: v_dual_mov_b32 v43, v3
	v_dual_mov_b32 v42, v2 :: v_dual_mov_b32 v41, v1
	v_dual_mov_b32 v54, v6 :: v_dual_mov_b32 v53, v5
	v_dual_mov_b32 v52, v4 :: v_dual_mov_b32 v51, v3
	v_dual_mov_b32 v50, v2 :: v_dual_mov_b32 v49, v1
	v_dual_mov_b32 v62, v6 :: v_dual_mov_b32 v61, v5
	v_dual_mov_b32 v60, v4 :: v_dual_mov_b32 v59, v3
	v_dual_mov_b32 v58, v2 :: v_dual_mov_b32 v57, v1
	v_dual_mov_b32 v70, v6 :: v_dual_mov_b32 v69, v5
	v_dual_mov_b32 v68, v4 :: v_dual_mov_b32 v67, v3
	v_dual_mov_b32 v66, v2 :: v_dual_mov_b32 v65, v1
	v_dual_mov_b32 v78, v6 :: v_dual_mov_b32 v77, v5
	v_dual_mov_b32 v76, v4 :: v_dual_mov_b32 v75, v3
	v_dual_mov_b32 v74, v2 :: v_dual_mov_b32 v73, v1
	v_dual_mov_b32 v86, v6 :: v_dual_mov_b32 v85, v5
	v_dual_mov_b32 v84, v4 :: v_dual_mov_b32 v83, v3
	v_dual_mov_b32 v82, v2 :: v_dual_mov_b32 v81, v1
	v_dual_mov_b32 v94, v6 :: v_dual_mov_b32 v93, v5
	v_dual_mov_b32 v92, v4 :: v_dual_mov_b32 v91, v3
	v_dual_mov_b32 v90, v2 :: v_dual_mov_b32 v89, v1
	v_dual_mov_b32 v102, v6 :: v_dual_mov_b32 v101, v5
	v_dual_mov_b32 v100, v4 :: v_dual_mov_b32 v99, v3
	v_dual_mov_b32 v98, v2 :: v_dual_mov_b32 v97, v1
	v_dual_mov_b32 v110, v6 :: v_dual_mov_b32 v109, v5
	v_dual_mov_b32 v108, v4 :: v_dual_mov_b32 v107, v3
	v_dual_mov_b32 v106, v2 :: v_dual_mov_b32 v105, v1
	v_dual_mov_b32 v118, v6 :: v_dual_mov_b32 v117, v5
	v_dual_mov_b32 v116, v4 :: v_dual_mov_b32 v115, v3
	v_dual_mov_b32 v114, v2 :: v_dual_mov_b32 v113, v1
	v_dual_mov_b32 v126, v6 :: v_dual_mov_b32 v125, v5
	v_dual_mov_b32 v124, v4 :: v_dual_mov_b32 v123, v3
	v_dual_mov_b32 v122, v2 :: v_dual_mov_b32 v121, v1
	v_mov_b32_e32 v120, v0
	s_mov_b32 s12, 0x76543210
	s_branch .LBB6_17
.LBB6_16:                               ;   in Loop: Header=BB6_17 Depth=1
	v_mov_b32_e32 v207, v129
	s_add_co_i32 s7, s7, 48
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s7, s20
	s_cbranch_scc1 .LBB6_79
.LBB6_17:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_21 Depth 2
                                        ;       Child Loop BB6_60 Depth 3
	s_add_co_i32 s6, s7, 47
	v_mov_b32_e32 v205, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s6, s21
	s_cselect_b32 s6, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s13, s22, s6
	s_wait_alu depctr_sa_sdst(0)
	s_nor_b32 s14, s1, s13
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s14
	s_cbranch_execz .LBB6_19
; %bb.18:                               ;   in Loop: Header=BB6_17 Depth=1
	global_load_b32 v205, v[144:145], off
.LBB6_19:                               ;   in Loop: Header=BB6_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_mov_b32 s14, 0
	s_branch .LBB6_21
.LBB6_20:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_sub_f32_e32 v209, v207, v129
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v207
	v_add_f32_e32 v131, v131, v208
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v209, 0x3fb8aa3b, v209
	v_exp_f32_e32 v209, v209
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v207, 0, v209, vcc_lo
	v_dual_fmac_f32 v131, v128, v207 :: v_dual_mul_f32 v128, v206, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v206, null, v130, v130, v128
	v_rcp_f32_e32 v207, v206
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v208, -v206, v207, 1.0
	v_fmac_f32_e32 v207, v208, v207
	v_div_scale_f32 v208, vcc_lo, v128, v130, v128
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v209, v208, v207
	v_fma_f32 v210, -v206, v209, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v209, v210, v207
	v_fma_f32 v206, -v206, v209, v208
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v206, v206, v207, v209
	v_div_fixup_f32 v128, v206, v130, v128
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v127, v127, v128 :: v_dual_mul_f32 v116, v116, v128
	v_dual_mul_f32 v126, v126, v128 :: v_dual_mul_f32 v125, v125, v128
	v_mul_f32_e32 v112, v112, v128
	v_dual_mul_f32 v124, v124, v128 :: v_dual_mul_f32 v123, v123, v128
	v_mul_f32_e32 v110, v110, v128
	v_dual_mul_f32 v122, v122, v128 :: v_dual_mul_f32 v121, v121, v128
	v_mul_f32_e32 v108, v108, v128
	v_dual_mul_f32 v120, v120, v128 :: v_dual_mul_f32 v119, v119, v128
	v_mul_f32_e32 v106, v106, v128
	v_dual_mul_f32 v118, v118, v128 :: v_dual_mul_f32 v117, v117, v128
	v_dual_mul_f32 v104, v104, v128 :: v_dual_mul_f32 v115, v115, v128
	v_mul_f32_e32 v102, v102, v128
	v_dual_mul_f32 v114, v114, v128 :: v_dual_mul_f32 v113, v113, v128
	v_dual_mul_f32 v100, v100, v128 :: v_dual_mul_f32 v111, v111, v128
	v_dual_mul_f32 v98, v98, v128 :: v_dual_mul_f32 v109, v109, v128
	v_dual_mul_f32 v96, v96, v128 :: v_dual_mul_f32 v107, v107, v128
	v_dual_mul_f32 v94, v94, v128 :: v_dual_mul_f32 v105, v105, v128
	v_dual_mul_f32 v92, v92, v128 :: v_dual_mul_f32 v103, v103, v128
	v_dual_mul_f32 v90, v90, v128 :: v_dual_mul_f32 v101, v101, v128
	v_dual_mul_f32 v88, v88, v128 :: v_dual_mul_f32 v99, v99, v128
	v_dual_mul_f32 v86, v86, v128 :: v_dual_mul_f32 v97, v97, v128
	v_dual_mul_f32 v84, v84, v128 :: v_dual_mul_f32 v95, v95, v128
	v_dual_mul_f32 v82, v82, v128 :: v_dual_mul_f32 v93, v93, v128
	v_dual_mul_f32 v80, v80, v128 :: v_dual_mul_f32 v91, v91, v128
	v_dual_mul_f32 v78, v78, v128 :: v_dual_mul_f32 v89, v89, v128
	v_dual_mul_f32 v76, v76, v128 :: v_dual_mul_f32 v87, v87, v128
	v_dual_mul_f32 v74, v74, v128 :: v_dual_mul_f32 v85, v85, v128
	v_dual_mul_f32 v72, v72, v128 :: v_dual_mul_f32 v83, v83, v128
	v_dual_mul_f32 v70, v70, v128 :: v_dual_mul_f32 v81, v81, v128
	v_dual_mul_f32 v68, v68, v128 :: v_dual_mul_f32 v79, v79, v128
	v_dual_mul_f32 v66, v66, v128 :: v_dual_mul_f32 v77, v77, v128
	v_dual_mul_f32 v64, v64, v128 :: v_dual_mul_f32 v75, v75, v128
	v_dual_mul_f32 v62, v62, v128 :: v_dual_mul_f32 v73, v73, v128
	v_dual_mul_f32 v60, v60, v128 :: v_dual_mul_f32 v71, v71, v128
	v_dual_mul_f32 v58, v58, v128 :: v_dual_mul_f32 v69, v69, v128
	v_dual_mul_f32 v56, v56, v128 :: v_dual_mul_f32 v67, v67, v128
	v_dual_mul_f32 v54, v54, v128 :: v_dual_mul_f32 v65, v65, v128
	v_dual_mul_f32 v52, v52, v128 :: v_dual_mul_f32 v63, v63, v128
	v_dual_mul_f32 v50, v50, v128 :: v_dual_mul_f32 v61, v61, v128
	v_dual_mul_f32 v48, v48, v128 :: v_dual_mul_f32 v59, v59, v128
	v_dual_mul_f32 v46, v46, v128 :: v_dual_mul_f32 v57, v57, v128
	v_dual_mul_f32 v44, v44, v128 :: v_dual_mul_f32 v55, v55, v128
	v_dual_mul_f32 v42, v42, v128 :: v_dual_mul_f32 v53, v53, v128
	v_dual_mul_f32 v40, v40, v128 :: v_dual_mul_f32 v51, v51, v128
	v_dual_mul_f32 v38, v38, v128 :: v_dual_mul_f32 v49, v49, v128
	v_dual_mul_f32 v36, v36, v128 :: v_dual_mul_f32 v47, v47, v128
	v_dual_mul_f32 v34, v34, v128 :: v_dual_mul_f32 v45, v45, v128
	v_dual_mul_f32 v32, v32, v128 :: v_dual_mul_f32 v43, v43, v128
	v_dual_mul_f32 v30, v30, v128 :: v_dual_mul_f32 v41, v41, v128
	v_dual_mul_f32 v28, v28, v128 :: v_dual_mul_f32 v39, v39, v128
	v_dual_mul_f32 v26, v26, v128 :: v_dual_mul_f32 v37, v37, v128
	v_dual_mul_f32 v24, v24, v128 :: v_dual_mul_f32 v35, v35, v128
	v_dual_mul_f32 v22, v22, v128 :: v_dual_mul_f32 v33, v33, v128
	v_dual_mul_f32 v20, v20, v128 :: v_dual_mul_f32 v31, v31, v128
	v_dual_mul_f32 v18, v18, v128 :: v_dual_mul_f32 v29, v29, v128
	v_dual_mul_f32 v16, v16, v128 :: v_dual_mul_f32 v27, v27, v128
	v_dual_mul_f32 v14, v14, v128 :: v_dual_mul_f32 v25, v25, v128
	v_dual_mul_f32 v12, v12, v128 :: v_dual_mul_f32 v23, v23, v128
	v_dual_mul_f32 v10, v10, v128 :: v_dual_mul_f32 v21, v21, v128
	v_dual_mul_f32 v8, v8, v128 :: v_dual_mul_f32 v19, v19, v128
	v_dual_mul_f32 v6, v6, v128 :: v_dual_mul_f32 v17, v17, v128
	v_dual_mul_f32 v4, v4, v128 :: v_dual_mul_f32 v15, v15, v128
	v_dual_mul_f32 v2, v2, v128 :: v_dual_mul_f32 v13, v13, v128
	v_dual_mul_f32 v0, v0, v128 :: v_dual_mul_f32 v11, v11, v128
	v_mul_f32_e32 v9, v9, v128
	v_mul_f32_e32 v7, v7, v128
	v_mul_f32_e32 v5, v5, v128
	v_mul_f32_e32 v3, v3, v128
	v_mul_f32_e32 v1, v1, v128
	v_div_scale_f32 v128, null, v130, v130, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v206, v128
	v_fma_f32 v207, -v128, v206, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v206, v207, v206
	v_div_scale_f32 v207, vcc_lo, v139, v130, v139
	v_mul_f32_e32 v208, v207, v206
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v209, -v128, v208, v207
	v_fmac_f32_e32 v208, v209, v206
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v128, -v128, v208, v207
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v128, v128, v206, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v128, v128, v130, v139
	v_div_scale_f32 v139, null, v130, v130, v138
	v_rcp_f32_e32 v206, v139
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v207, -v139, v206, 1.0
	v_fmac_f32_e32 v206, v207, v206
	v_div_scale_f32 v207, vcc_lo, v138, v130, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v208, v207, v206
	v_fma_f32 v209, -v139, v208, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v208, v209, v206
	v_fma_f32 v139, -v139, v208, v207
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v139, v139, v206, v208
	v_div_fixup_f32 v206, v139, v130, v138
	v_mov_b16_e64 v139.l, v156.l
	v_mov_b16_e64 v139.h, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v138.l, v139.l
	v_mov_b16_e64 v138.h, v139.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v138.l, v128, v206
	v_div_scale_f32 v128, null, v130, v130, v137
	v_rcp_f32_e32 v206, v128
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v207, -v128, v206, 1.0
	v_fmac_f32_e32 v206, v207, v206
	v_div_scale_f32 v207, vcc_lo, v137, v130, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v208, v207, v206
	v_fma_f32 v209, -v128, v208, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v208, v209, v206
	v_fma_f32 v128, -v128, v208, v207
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v128, v128, v206, v208
	v_div_fixup_f32 v128, v128, v130, v137
	v_div_scale_f32 v137, null, v130, v130, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v206, v137
	v_fma_f32 v207, -v137, v206, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v206, v207, v206
	v_div_scale_f32 v207, vcc_lo, v136, v130, v136
	v_mul_f32_e32 v208, v207, v206
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v209, -v137, v208, v207
	v_fmac_f32_e32 v208, v209, v206
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v137, -v137, v208, v207
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v137, v137, v206, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v136, v137, v130, v136
	v_cvt_pk_fp8_f32 v138.h, v128, v136
	v_div_scale_f32 v128, null, v130, v130, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v136, v128
	v_fma_f32 v137, -v128, v136, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v136, v137, v136
	v_div_scale_f32 v137, vcc_lo, v135, v130, v135
	v_mul_f32_e32 v206, v137, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v207, -v128, v206, v137
	v_fmac_f32_e32 v206, v207, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v128, -v128, v206, v137
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v128, v128, v136, v206
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v128, v128, v130, v135
	v_div_scale_f32 v135, null, v130, v130, v134
	v_rcp_f32_e32 v136, v135
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v137, -v135, v136, 1.0
	v_fmac_f32_e32 v136, v137, v136
	v_div_scale_f32 v137, vcc_lo, v134, v130, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v206, v137, v136
	v_fma_f32 v207, -v135, v206, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v206, v207, v136
	v_fma_f32 v135, -v135, v206, v137
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v135, v135, v136, v206
	v_mov_b32_e32 v206, v130
	v_div_fixup_f32 v134, v135, v130, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v139.l, v128, v134
	v_div_scale_f32 v128, null, v130, v130, v133
	v_rcp_f32_e32 v134, v128
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v135, -v128, v134, 1.0
	v_fmac_f32_e32 v134, v135, v134
	v_div_scale_f32 v135, vcc_lo, v133, v130, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v136, v135, v134
	v_fma_f32 v137, -v128, v136, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v136, v137, v134
	v_fma_f32 v128, -v128, v136, v135
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v128, v128, v134, v136
	v_div_fixup_f32 v128, v128, v130, v133
	v_div_scale_f32 v133, null, v130, v130, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v134, v133
	v_fma_f32 v135, -v133, v134, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v134, v135, v134
	v_div_scale_f32 v135, vcc_lo, v132, v130, v132
	v_mul_f32_e32 v136, v135, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v137, -v133, v136, v135
	v_fmac_f32_e32 v136, v137, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v133, -v133, v136, v135
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v133, v133, v134, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v132, v133, v130, v132
	v_cvt_pk_fp8_f32 v139.h, v128, v132
	v_add_nc_u32_e32 v128, 0, v176
	ds_load_b128 v[132:135], v128 offset:12288
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[132:133], v[138:139], v[120:127]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[134:135], v[138:139], v[112:119]
	ds_load_b128 v[132:135], v128 offset:12800
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[132:133], v[138:139], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[134:135], v[138:139], v[96:103]
	ds_load_b128 v[132:135], v128 offset:13312
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[132:133], v[138:139], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[134:135], v[138:139], v[80:87]
	ds_load_b128 v[132:135], v128 offset:13824
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[132:133], v[138:139], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[134:135], v[138:139], v[64:71]
	ds_load_b128 v[132:135], v128 offset:14336
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[132:133], v[138:139], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[134:135], v[138:139], v[48:55]
	ds_load_b128 v[132:135], v128 offset:14848
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[132:133], v[138:139], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[134:135], v[138:139], v[32:39]
	ds_load_b128 v[132:135], v128 offset:15360
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[132:133], v[138:139], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[134:135], v[138:139], v[16:23]
	ds_load_b128 v[132:135], v128 offset:15872
	v_mov_b32_e32 v128, v131
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[132:133], v[138:139], v[8:15]
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[134:135], v[138:139], v[0:7]
	v_mov_b32_e32 v207, v129
	s_add_co_i32 s14, s14, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s14, 3
	s_cbranch_scc1 .LBB6_16
.LBB6_21:                               ;   Parent Loop BB6_17 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB6_60 Depth 3
	s_and_saveexec_b32 s6, s2
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB6_27
; %bb.22:                               ;   in Loop: Header=BB6_21 Depth=2
	s_and_saveexec_b32 s15, s4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s15, exec_lo, s15
	s_cbranch_execz .LBB6_24
; %bb.23:                               ;   in Loop: Header=BB6_21 Depth=2
	s_mov_b32 m0, 3
	s_barrier_join m0 
.LBB6_24:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s15, s15
	s_cbranch_execz .LBB6_26
; %bb.25:                               ;   in Loop: Header=BB6_21 Depth=2
	s_mov_b32 m0, 2
	s_barrier_join m0 
.LBB6_26:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB6_27:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB6_29
; %bb.28:                               ;   in Loop: Header=BB6_21 Depth=2
	s_mov_b32 m0, 1
	s_barrier_join m0 
.LBB6_29:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_lshl4_add_u32 s6, s14, s7
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v131, 0
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v133, s6, v151
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v129, 0
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s20, v133
	s_cbranch_execz .LBB6_31
; %bb.30:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mad_co_u64_u32 v[131:132], null, 0x408, v133, v[146:147]
	s_clause 0x1
	global_load_b64 v[129:130], v[131:132], off
	global_load_b64 v[131:132], v[131:132], off offset:16
.LBB6_31:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_wait_loadcnt 0x0
	ds_store_b128 v203, v[129:132]
	s_and_saveexec_b32 s15, s3
	s_cbranch_execz .LBB6_50
; %bb.32:                               ;   in Loop: Header=BB6_21 Depth=2
	v_or_b32_e32 v135, s6, v157
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v130, 0
	s_mov_b32 s19, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s20, v135
	s_cbranch_execz .LBB6_34
; %bb.33:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mad_co_u64_u32 v[129:130], null, 0x408, v135, v[148:149]
	global_load_b64 v[129:130], v[129:130], off
.LBB6_34:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_add_nc_u32_e32 v133, 0x6000, v162
	s_mov_b32 s19, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v133, v129, v130 offset1:1
	v_cmpx_gt_i32_e64 s20, v135
	s_cbranch_execz .LBB6_36
; %bb.35:                               ;   in Loop: Header=BB6_21 Depth=2
	v_or_b32_e32 v129, 1, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[129:130], null, 0x408, v129, v[148:149]
	global_load_b64 v[131:132], v[129:130], off
.LBB6_36:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_or_b32_e32 v136, 2, v135
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v134, 0
	s_mov_b32 s19, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v163, v131 offset:24576
	ds_store_b32 v164, v132 offset:24576
	v_cmpx_ge_i32_e64 s20, v136
	s_cbranch_execz .LBB6_38
; %bb.37:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mad_co_u64_u32 v[131:132], null, 0x408, v136, v[148:149]
	global_load_b64 v[133:134], v[131:132], off
.LBB6_38:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_or_b32_e32 v131, 3, v135
	s_mov_b32 s19, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v165, v133 offset:24576
	ds_store_b32 v166, v134 offset:24576
	v_cmpx_ge_i32_e64 s20, v131
	s_cbranch_execz .LBB6_40
; %bb.39:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mad_co_u64_u32 v[129:130], null, 0x408, v131, v[148:149]
	global_load_b64 v[129:130], v[129:130], off
.LBB6_40:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_or_b32_e32 v136, 4, v135
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v134, 0
	s_mov_b32 s19, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v167, v129 offset:24576
	ds_store_b32 v168, v130 offset:24576
	v_cmpx_ge_i32_e64 s20, v136
	s_cbranch_execz .LBB6_42
; %bb.41:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mad_co_u64_u32 v[129:130], null, 0x408, v136, v[148:149]
	global_load_b64 v[133:134], v[129:130], off
.LBB6_42:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_or_b32_e32 v129, 5, v135
	v_add_nc_u32_e32 v130, 0x6400, v162
	s_mov_b32 s19, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v130, v133, v134 offset1:1
	v_cmpx_ge_i32_e64 s20, v129
	s_cbranch_execz .LBB6_44
; %bb.43:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mad_co_u64_u32 v[129:130], null, 0x408, v129, v[148:149]
	global_load_b64 v[131:132], v[129:130], off
.LBB6_44:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_or_b32_e32 v136, 6, v135
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v134, 0
	s_mov_b32 s19, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v169, v131 offset:24576
	ds_store_b32 v170, v132 offset:24576
	v_cmpx_ge_i32_e64 s20, v136
	s_cbranch_execz .LBB6_46
; %bb.45:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mad_co_u64_u32 v[131:132], null, 0x408, v136, v[148:149]
	global_load_b64 v[133:134], v[131:132], off
.LBB6_46:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_or_b32_e32 v131, 7, v135
	s_mov_b32 s19, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v171, v133 offset:24576
	ds_store_b32 v172, v134 offset:24576
	v_cmpx_ge_i32_e64 s20, v131
	s_cbranch_execz .LBB6_48
; %bb.47:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mad_co_u64_u32 v[129:130], null, 0x408, v131, v[148:149]
	global_load_b64 v[129:130], v[129:130], off
.LBB6_48:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	s_wait_loadcnt 0x0
	ds_store_b32 v173, v129 offset:24576
	ds_store_b32 v174, v130 offset:24576
	;;#ASMSTART
	s_wait_loadcnt 0
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 exec_lo, exec_lo, s5
	s_cbranch_execz .LBB6_50
; %bb.49:                               ;   in Loop: Header=BB6_21 Depth=2
	v_add_nc_u32_e32 v133, 0x6000, v175
	v_add_nc_u32_e32 v134, 0x6400, v175
	v_add_nc_u32_e32 v139, 0, v176
	ds_load_2addr_b32 v[129:130], v133 offset1:4
	ds_load_2addr_b32 v[131:132], v134 offset1:4
	s_wait_dscnt 0x1
	ds_bpermute_b32 v135, v155, v129
	s_wait_dscnt 0x1
	ds_bpermute_b32 v136, v155, v131
	ds_bpermute_b32 v137, v155, v130
	ds_bpermute_b32 v138, v155, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v135, v129, v158
	s_wait_dscnt 0x2
	v_perm_b32 v131, v136, v131, v158
	s_wait_dscnt 0x1
	v_perm_b32 v135, v137, v130, v158
	s_wait_dscnt 0x0
	v_perm_b32 v132, v138, v132, v158
	ds_bpermute_b32 v130, v154, v129
	ds_bpermute_b32 v136, v154, v131
	ds_bpermute_b32 v137, v154, v135
	ds_bpermute_b32 v138, v154, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v130, v129, v159
	s_wait_dscnt 0x2
	v_perm_b32 v130, v136, v131, v159
	s_wait_dscnt 0x1
	v_perm_b32 v131, v137, v135, v159
	s_wait_dscnt 0x0
	v_perm_b32 v132, v138, v132, v159
	ds_store_b128 v139, v[129:132] offset:12288
	ds_load_b32 v129, v177 offset:24576
	ds_load_b32 v130, v178 offset:24576
	ds_load_b32 v131, v179 offset:24576
	ds_load_b32 v132, v180 offset:24576
	ds_load_b32 v135, v182 offset:24576
	ds_load_b32 v136, v184 offset:24576
	s_wait_dscnt 0x5
	ds_bpermute_b32 v137, v155, v129
	s_wait_dscnt 0x5
	ds_bpermute_b32 v138, v155, v130
	s_wait_dscnt 0x5
	ds_bpermute_b32 v208, v155, v131
	s_wait_dscnt 0x5
	ds_bpermute_b32 v209, v155, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v137, v129, v158
	s_wait_dscnt 0x2
	v_perm_b32 v130, v138, v130, v158
	s_wait_dscnt 0x1
	v_perm_b32 v131, v208, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v132, v209, v132, v158
	ds_bpermute_b32 v137, v154, v129
	ds_bpermute_b32 v138, v154, v130
	ds_bpermute_b32 v208, v154, v131
	ds_bpermute_b32 v209, v154, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v137, v129, v159
	s_wait_dscnt 0x2
	v_perm_b32 v130, v138, v130, v159
	s_wait_dscnt 0x1
	v_perm_b32 v131, v208, v131, v159
	s_wait_dscnt 0x0
	v_perm_b32 v132, v209, v132, v159
	ds_store_b128 v139, v[129:132] offset:12800
	ds_load_b32 v129, v181 offset:24576
	ds_load_b32 v130, v183 offset:24576
	ds_bpermute_b32 v131, v155, v135
	ds_bpermute_b32 v132, v155, v136
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v155, v129
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v155, v130
	s_wait_dscnt 0x3
	v_perm_b32 v131, v131, v135, v158
	s_wait_dscnt 0x2
	v_perm_b32 v132, v132, v136, v158
	ds_bpermute_b32 v136, v154, v132
	s_wait_dscnt 0x2
	v_perm_b32 v129, v137, v129, v158
	s_wait_dscnt 0x1
	v_perm_b32 v135, v138, v130, v158
	ds_bpermute_b32 v130, v154, v131
	ds_bpermute_b32 v137, v154, v129
	ds_bpermute_b32 v138, v154, v135
	s_wait_dscnt 0x3
	v_perm_b32 v132, v136, v132, v159
	s_wait_dscnt 0x2
	v_perm_b32 v130, v130, v131, v159
	s_wait_dscnt 0x1
	v_perm_b32 v129, v137, v129, v159
	s_wait_dscnt 0x0
	v_perm_b32 v131, v138, v135, v159
	ds_store_b128 v139, v[129:132] offset:13312
	ds_load_b32 v129, v185 offset:24576
	ds_load_b32 v130, v186 offset:24576
	ds_load_b32 v131, v187 offset:24576
	ds_load_b32 v132, v188 offset:24576
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v155, v129
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v155, v130
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v155, v131
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v155, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v135, v129, v158
	s_wait_dscnt 0x2
	v_perm_b32 v130, v136, v130, v158
	s_wait_dscnt 0x1
	v_perm_b32 v131, v137, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v132, v138, v132, v158
	ds_bpermute_b32 v135, v154, v129
	ds_bpermute_b32 v136, v154, v130
	ds_bpermute_b32 v137, v154, v131
	ds_bpermute_b32 v138, v154, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v135, v129, v159
	s_wait_dscnt 0x2
	v_perm_b32 v130, v136, v130, v159
	s_wait_dscnt 0x1
	v_perm_b32 v131, v137, v131, v159
	s_wait_dscnt 0x0
	v_perm_b32 v132, v138, v132, v159
	ds_store_b128 v139, v[129:132] offset:13824
	ds_load_2addr_b32 v[129:130], v133 offset0:32 offset1:36
	ds_load_2addr_b32 v[131:132], v134 offset0:32 offset1:36
	s_wait_dscnt 0x1
	ds_bpermute_b32 v133, v155, v129
	s_wait_dscnt 0x1
	ds_bpermute_b32 v134, v155, v131
	ds_bpermute_b32 v135, v155, v130
	ds_bpermute_b32 v136, v155, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v133, v129, v158
	s_wait_dscnt 0x2
	v_perm_b32 v131, v134, v131, v158
	s_wait_dscnt 0x1
	v_perm_b32 v133, v135, v130, v158
	s_wait_dscnt 0x0
	v_perm_b32 v132, v136, v132, v158
	ds_bpermute_b32 v130, v154, v129
	ds_bpermute_b32 v134, v154, v131
	ds_bpermute_b32 v135, v154, v133
	ds_bpermute_b32 v136, v154, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v130, v129, v159
	s_wait_dscnt 0x2
	v_perm_b32 v130, v134, v131, v159
	s_wait_dscnt 0x1
	v_perm_b32 v131, v135, v133, v159
	s_wait_dscnt 0x0
	v_perm_b32 v132, v136, v132, v159
	ds_store_b128 v139, v[129:132] offset:14336
	ds_load_b32 v129, v189 offset:24576
	ds_load_b32 v130, v190 offset:24576
	ds_load_b32 v131, v191 offset:24576
	ds_load_b32 v132, v192 offset:24576
	s_wait_dscnt 0x3
	ds_bpermute_b32 v133, v155, v129
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v155, v130
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v155, v131
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v155, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v133, v129, v158
	s_wait_dscnt 0x2
	v_perm_b32 v130, v134, v130, v158
	s_wait_dscnt 0x1
	v_perm_b32 v131, v135, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v132, v136, v132, v158
	ds_bpermute_b32 v133, v154, v129
	ds_bpermute_b32 v134, v154, v130
	ds_bpermute_b32 v135, v154, v131
	ds_bpermute_b32 v136, v154, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v133, v129, v159
	s_wait_dscnt 0x2
	v_perm_b32 v130, v134, v130, v159
	s_wait_dscnt 0x1
	v_perm_b32 v131, v135, v131, v159
	s_wait_dscnt 0x0
	v_perm_b32 v132, v136, v132, v159
	ds_store_b128 v139, v[129:132] offset:14848
	ds_load_b32 v129, v193 offset:24576
	ds_load_b32 v130, v194 offset:24576
	ds_load_b32 v131, v195 offset:24576
	ds_load_b32 v132, v196 offset:24576
	s_wait_dscnt 0x3
	ds_bpermute_b32 v133, v155, v129
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v155, v130
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v155, v131
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v155, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v133, v129, v158
	s_wait_dscnt 0x2
	v_perm_b32 v130, v134, v130, v158
	s_wait_dscnt 0x1
	v_perm_b32 v131, v135, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v132, v136, v132, v158
	ds_bpermute_b32 v133, v154, v129
	ds_bpermute_b32 v134, v154, v130
	ds_bpermute_b32 v135, v154, v131
	ds_bpermute_b32 v136, v154, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v133, v129, v159
	s_wait_dscnt 0x2
	v_perm_b32 v130, v134, v130, v159
	s_wait_dscnt 0x1
	v_perm_b32 v131, v135, v131, v159
	s_wait_dscnt 0x0
	v_perm_b32 v132, v136, v132, v159
	ds_store_b128 v139, v[129:132] offset:15360
	ds_load_b32 v129, v197 offset:24576
	ds_load_b32 v130, v198 offset:24576
	ds_load_b32 v131, v199 offset:24576
	ds_load_b32 v132, v200 offset:24576
	s_wait_dscnt 0x3
	ds_bpermute_b32 v133, v155, v129
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v155, v130
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v155, v131
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v155, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v133, v129, v158
	s_wait_dscnt 0x2
	v_perm_b32 v130, v134, v130, v158
	s_wait_dscnt 0x1
	v_perm_b32 v131, v135, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v132, v136, v132, v158
	ds_bpermute_b32 v133, v154, v129
	ds_bpermute_b32 v134, v154, v130
	ds_bpermute_b32 v135, v154, v131
	ds_bpermute_b32 v136, v154, v132
	s_wait_dscnt 0x3
	v_perm_b32 v129, v133, v129, v159
	s_wait_dscnt 0x2
	v_perm_b32 v130, v134, v130, v159
	s_wait_dscnt 0x1
	v_perm_b32 v131, v135, v131, v159
	s_wait_dscnt 0x0
	v_perm_b32 v132, v136, v132, v159
	ds_store_b128 v139, v[129:132] offset:15872
.LBB6_50:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_saveexec_b32 s15, s18
	s_cbranch_execz .LBB6_54
; %bb.51:                               ;   in Loop: Header=BB6_21 Depth=2
	v_or_b32_e32 v130, s6, v153
	v_mov_b32_e32 v129, 0
	s_mov_b32 s19, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s20, v130
	s_cbranch_execz .LBB6_53
; %bb.52:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mad_co_u64_u32 v[131:132], null, 0x408, v130, s[8:9]
	v_mad_co_u64_u32 v[133:134], null, 0x408, v130, s[10:11]
	global_load_d16_b16 v129, v[131:132], off offset:1024
	global_load_d16_hi_b16 v129, v[133:134], off offset:1024
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v130.h, 8, v129.l
	v_lshrrev_b16 v130.l, 8, v129.h
	v_and_b16 v131.h, 0xff, v129.l
	v_and_b16 v131.l, 0xff, v129.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_lshlrev_b16 v129, 8, v130 op_sel_hi:[0,1]
	v_or_b32_e32 v129, v129, v131
.LBB6_53:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	ds_store_b16_d16_hi v160, v129 offset:36864
	ds_store_b16 v160, v129 offset:36960
.LBB6_54:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_wait_dscnt 0x0
	s_and_saveexec_b32 s15, s2
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s15, exec_lo, s15
	s_cbranch_execnz .LBB6_65
; %bb.55:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s15, s15
	s_cbranch_execnz .LBB6_70
.LBB6_56:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_saveexec_b32 s15, s2
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s15, exec_lo, s15
	s_cbranch_execnz .LBB6_71
.LBB6_57:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s15, s15
	s_cbranch_execnz .LBB6_76
.LBB6_58:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_cmp_gt_i32 s6, s20
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB6_77
.LBB6_59:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mov_b32_e32 v129, 0
	s_mov_b32 s15, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v130, v129 :: v_dual_mov_b32 v131, v129
	v_dual_mov_b32 v132, v129 :: v_dual_mov_b32 v133, v129
	v_dual_mov_b32 v134, v129 :: v_dual_mov_b32 v135, v129
	v_mov_b32_e32 v136, v129
.LBB6_60:                               ;   Parent Loop BB6_17 Depth=1
                                        ;     Parent Loop BB6_21 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[137:138], null, s15, 6, v[143:144]
	s_lshl_b32 s19, s15, 6
	s_add_co_i32 s15, s15, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s15, 4
	v_lshl_add_u32 v139, v137, 9, v161
	v_add_co_u32 v137, vcc_lo, v201, s19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, 0, v202, vcc_lo
	ds_load_b128 v[208:211], v139
	s_clause 0x1
	global_load_b64 v[212:213], v[137:138], off
	global_load_b64 v[214:215], v[137:138], off offset:16
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[208:209], v[212:213], v[129:136]
	s_clause 0x1
	global_load_b64 v[212:213], v[137:138], off offset:32
	global_load_b64 v[137:138], v[137:138], off offset:48
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[210:211], v[214:215], v[129:136]
	ds_load_b128 v[208:211], v139 offset:1536
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[208:209], v[212:213], v[129:136]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[210:211], v[137:138], v[129:136]
	s_cbranch_scc1 .LBB6_60
; %bb.61:                               ;   in Loop: Header=BB6_21 Depth=2
	ds_load_b96 v[137:139], v204 offset:36866
	ds_load_u16_d16 v209, v204 offset:36878
	s_or_b32 s15, s6, 15
	v_or_b32_e32 v210, s6, v142
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s15, s21
	v_mov_b32_e32 v208, 0xff800000
	s_cselect_b32 s6, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s13, s6
	s_and_saveexec_b32 s15, s0
	s_cbranch_execz .LBB6_63
; %bb.62:                               ;   in Loop: Header=BB6_21 Depth=2
	ds_load_u16_d16 v208, v204 offset:36864
	v_mul_f32_e32 v129, v152, v129
	v_cmp_le_i32_e32 vcc_lo, v210, v205
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s6, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v129, v208, v129, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v208, 0xff800000, v129, vcc_lo
.LBB6_63:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_mul_f32_e32 v129, v152, v130
	v_or_b32_e32 v130, 2, v210
	v_cmp_ge_i32_e32 vcc_lo, v210, v205
	s_xor_b32 s15, s6, -1
	v_mul_f32_e32 v131, v152, v131
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s6, v130, v205
	v_or_b32_e32 v130, 3, v210
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s19, s15, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s19, s1, s19
	s_and_b32 s6, s15, s6
	v_cmp_gt_i32_e32 vcc_lo, v130, v205
	v_mul_f32_e32 v130, v152, v132
	s_wait_dscnt 0x1
	v_fma_mix_f32 v129, v137, v129, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s1, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v211, v129, 0xff800000, s19
	v_fma_mix_f32 v129, v137, v131, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_or_b32_e32 v131, 4, v210
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v132, v129, 0xff800000, s6
	v_fma_mix_f32 v129, v138, v130, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v130, v152, v133
	s_and_b32 s6, s15, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v131, v205
	v_or_b32_e32 v131, 5, v210
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s1, s6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v133, v129, 0xff800000, s6
	v_fma_mix_f32 v129, v138, v130, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v130, v152, v134
	s_and_b32 s6, s15, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v131, v205
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s1, s6
	v_or_b32_e32 v131, 7, v210
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v134, v129, 0xff800000, s6
	v_fma_mix_f32 v129, v139, v130, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v130, 6, v210
	s_and_b32 s6, s15, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s1, s6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v137, v129, 0xff800000, s6
	v_cmp_gt_i32_e32 vcc_lo, v130, v205
	v_mul_f32_e32 v129, v152, v135
	v_cmp_gt_i32_e64 s6, v131, v205
	v_mul_f32_e32 v130, v152, v136
	v_max3_num_f32 v131, v208, 0xff800000, v211
	s_and_b32 s19, s15, vcc_lo
	v_fma_mix_f32 v129, v139, v129, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_and_b32 s6, s15, s6
	s_wait_dscnt 0x0
	v_fma_mix_f32 v130, v209, v130, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v131, v131, v132, v133
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s15, s1, s19
	s_or_b32 s6, s1, s6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v135, v129, 0xff800000, s15
	v_cndmask_b32_e64 v130, v130, 0xff800000, s6
	v_max3_num_f32 v129, v131, v134, v137
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v135, v130
	v_mov_b32_e32 v131, v129
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v131, v131, s12, 0xfedcba98
	v_max3_num_f32 v129, v207, v129, v131
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_sub_f32 v130, v130, v129 :: v_dual_sub_f32 v131, v208, v129
	v_sub_f32_e32 v136, v211, v129
	v_sub_f32_e32 v135, v135, v129
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v129
	v_dual_mul_f32 v130, 0x3fb8aa3b, v130 :: v_dual_mul_f32 v131, 0x3fb8aa3b, v131
	v_sub_f32_e32 v137, v137, v129
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v135, 0x3fb8aa3b, v135
	v_exp_f32_e32 v138, v130
	v_mul_f32_e32 v130, 0x3fb8aa3b, v136
	v_exp_f32_e32 v136, v131
	v_dual_sub_f32 v131, v132, v129 :: v_dual_sub_f32 v132, v133, v129
	v_add_nc_u32_e32 v133, 0x9060, v204
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v139, v130
	v_exp_f32_e32 v135, v135
	v_mul_f32_e32 v210, 0x3fb8aa3b, v131
	v_mul_f32_e32 v132, 0x3fb8aa3b, v132
	ds_load_2addr_b32 v[130:131], v133 offset1:1
	v_sub_f32_e32 v133, v134, v129
	v_mul_f32_e32 v134, 0x3fb8aa3b, v137
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v136, v136, 0, vcc_lo
	v_exp_f32_e32 v132, v132
	v_cndmask_b32_e64 v211, v139, 0, vcc_lo
	v_mul_f32_e32 v133, 0x3fb8aa3b, v133
	v_exp_f32_e32 v134, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v133, v133
	v_cndmask_b32_e64 v132, v132, 0, vcc_lo
	v_add_nc_u32_e32 v137, 0x9068, v204
	s_delay_alu instid0(TRANS32_DEP_2)
	v_cndmask_b32_e64 v212, v134, 0, vcc_lo
	ds_load_2addr_b32 v[208:209], v137 offset1:1
	v_exp_f32_e32 v137, v210
	v_cndmask_b32_e64 v210, v138, 0, vcc_lo
	s_wait_dscnt 0x1
	v_fma_mix_f32 v139, v130, v136, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v138, v130, v211, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v130, v133, 0, vcc_lo
	v_add_f32_e32 v133, v136, v211
	v_fma_mix_f32 v136, v131, v132, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v211, v135, 0, vcc_lo
	v_cndmask_b32_e64 v134, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v137, v131, v134, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v131, v139, 0, v138
	v_add_f32_e32 v133, v134, v133
	s_wait_dscnt 0x0
	v_fma_mix_f32 v135, v208, v130, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v134, v208, v212, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_max3_num_f32 v131, v131, v137, v136
	v_add_f32_e32 v208, v132, v133
	v_fma_mix_f32 v133, v209, v211, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v132, v209, v210, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_max3_num_f32 v131, v131, v135, v134
	v_add_f32_e32 v130, v130, v208
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v209, v131, v133, v132
	v_add_f32_e32 v130, v212, v130
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b32_e32 v208, v209
	v_add_f32_e32 v130, v211, v130
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v208, v208, s12, 0xfedcba98
	v_dual_add_f32 v131, v210, v130 :: v_dual_max_num_f32 v130, v208, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v208, v131 :: v_dual_max_num_f32 v209, v209, v130
	v_permlanex16_b32 v208, v208, s12, 0xfedcba98
	v_mov_b32_e32 v130, v206
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_lt_f32_e32 0, v209
	s_cbranch_execz .LBB6_20
; %bb.64:                               ;   in Loop: Header=BB6_21 Depth=2
	v_div_scale_f32 v130, null, 0x43e00000, 0x43e00000, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v210, v130
	v_fma_f32 v211, -v130, v210, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v210, v211, v210
	v_div_scale_f32 v211, vcc_lo, v209, 0x43e00000, v209
	v_mul_f32_e32 v212, v211, v210
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v213, -v130, v212, v211
	v_fmac_f32_e32 v212, v213, v210
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v130, -v130, v212, v211
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v130, v130, v210, v212
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v130, v130, 0x43e00000, v209
	v_max_num_f32_e32 v130, 0x1f800000, v130
	s_branch .LBB6_20
.LBB6_65:                               ;   in Loop: Header=BB6_21 Depth=2
	s_and_saveexec_b32 s19, s4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s19, exec_lo, s19
	s_cbranch_execz .LBB6_67
; %bb.66:                               ;   in Loop: Header=BB6_21 Depth=2
	s_barrier_signal 3
.LBB6_67:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s19, s19
	s_cbranch_execz .LBB6_69
; %bb.68:                               ;   in Loop: Header=BB6_21 Depth=2
	s_barrier_signal 2
.LBB6_69:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	s_and_not1_saveexec_b32 s15, s15
	s_cbranch_execz .LBB6_56
.LBB6_70:                               ;   in Loop: Header=BB6_21 Depth=2
	s_barrier_signal 1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_saveexec_b32 s15, s2
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s15, exec_lo, s15
	s_cbranch_execz .LBB6_57
.LBB6_71:                               ;   in Loop: Header=BB6_21 Depth=2
	s_and_saveexec_b32 s19, s4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s19, exec_lo, s19
	s_cbranch_execz .LBB6_73
; %bb.72:                               ;   in Loop: Header=BB6_21 Depth=2
	s_barrier_wait 3
.LBB6_73:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s19, s19
	s_cbranch_execz .LBB6_75
; %bb.74:                               ;   in Loop: Header=BB6_21 Depth=2
	s_barrier_wait 2
.LBB6_75:                               ;   in Loop: Header=BB6_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	s_and_not1_saveexec_b32 s15, s15
	s_cbranch_execz .LBB6_58
.LBB6_76:                               ;   in Loop: Header=BB6_21 Depth=2
	s_barrier_wait 1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_cmp_gt_i32 s6, s20
	global_inv scope:SCOPE_SE
	s_cbranch_scc0 .LBB6_59
.LBB6_77:                               ;   in Loop: Header=BB6_21 Depth=2
	v_mov_b32_e32 v129, v207
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v207, v129
	s_add_co_i32 s14, s14, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s14, 3
	s_cbranch_scc0 .LBB6_21
	s_branch .LBB6_16
.LBB6_78:
	v_mov_b32_e32 v128, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v206, 1.0 :: v_dual_mov_b32 v135, v128
	v_dual_mov_b32 v129, v128 :: v_dual_mov_b32 v130, v128
	v_dual_mov_b32 v131, v128 :: v_dual_mov_b32 v132, v128
	v_dual_mov_b32 v133, v128 :: v_dual_mov_b32 v134, v128
	v_dual_mov_b32 v120, v128 :: v_dual_mov_b32 v121, v129
	v_dual_mov_b32 v112, v128 :: v_dual_mov_b32 v113, v129
	v_dual_mov_b32 v104, v128 :: v_dual_mov_b32 v105, v129
	v_dual_mov_b32 v96, v128 :: v_dual_mov_b32 v97, v129
	v_dual_mov_b32 v88, v128 :: v_dual_mov_b32 v89, v129
	v_dual_mov_b32 v80, v128 :: v_dual_mov_b32 v81, v129
	v_dual_mov_b32 v72, v128 :: v_dual_mov_b32 v73, v129
	v_dual_mov_b32 v64, v128 :: v_dual_mov_b32 v65, v129
	v_dual_mov_b32 v56, v128 :: v_dual_mov_b32 v57, v129
	v_dual_mov_b32 v48, v128 :: v_dual_mov_b32 v49, v129
	v_dual_mov_b32 v40, v128 :: v_dual_mov_b32 v41, v129
	v_dual_mov_b32 v32, v128 :: v_dual_mov_b32 v33, v129
	v_dual_mov_b32 v24, v128 :: v_dual_mov_b32 v25, v129
	v_dual_mov_b32 v16, v128 :: v_dual_mov_b32 v17, v129
	v_dual_mov_b32 v8, v128 :: v_dual_mov_b32 v9, v129
	v_dual_mov_b32 v0, v128 :: v_dual_mov_b32 v1, v129
	v_dual_mov_b32 v122, v130 :: v_dual_mov_b32 v123, v131
	v_dual_mov_b32 v124, v132 :: v_dual_mov_b32 v125, v133
	v_dual_mov_b32 v126, v134 :: v_dual_mov_b32 v127, v135
	v_dual_mov_b32 v114, v130 :: v_dual_mov_b32 v115, v131
	v_dual_mov_b32 v116, v132 :: v_dual_mov_b32 v117, v133
	v_dual_mov_b32 v118, v134 :: v_dual_mov_b32 v119, v135
	v_dual_mov_b32 v106, v130 :: v_dual_mov_b32 v107, v131
	v_dual_mov_b32 v108, v132 :: v_dual_mov_b32 v109, v133
	v_dual_mov_b32 v110, v134 :: v_dual_mov_b32 v111, v135
	v_dual_mov_b32 v98, v130 :: v_dual_mov_b32 v99, v131
	v_dual_mov_b32 v100, v132 :: v_dual_mov_b32 v101, v133
	v_dual_mov_b32 v102, v134 :: v_dual_mov_b32 v103, v135
	v_dual_mov_b32 v90, v130 :: v_dual_mov_b32 v91, v131
	v_dual_mov_b32 v92, v132 :: v_dual_mov_b32 v93, v133
	v_dual_mov_b32 v94, v134 :: v_dual_mov_b32 v95, v135
	v_dual_mov_b32 v82, v130 :: v_dual_mov_b32 v83, v131
	v_dual_mov_b32 v84, v132 :: v_dual_mov_b32 v85, v133
	v_dual_mov_b32 v86, v134 :: v_dual_mov_b32 v87, v135
	v_dual_mov_b32 v74, v130 :: v_dual_mov_b32 v75, v131
	v_dual_mov_b32 v76, v132 :: v_dual_mov_b32 v77, v133
	v_dual_mov_b32 v78, v134 :: v_dual_mov_b32 v79, v135
	v_dual_mov_b32 v66, v130 :: v_dual_mov_b32 v67, v131
	v_dual_mov_b32 v68, v132 :: v_dual_mov_b32 v69, v133
	v_dual_mov_b32 v70, v134 :: v_dual_mov_b32 v71, v135
	v_dual_mov_b32 v58, v130 :: v_dual_mov_b32 v59, v131
	v_dual_mov_b32 v60, v132 :: v_dual_mov_b32 v61, v133
	v_dual_mov_b32 v62, v134 :: v_dual_mov_b32 v63, v135
	v_dual_mov_b32 v50, v130 :: v_dual_mov_b32 v51, v131
	v_dual_mov_b32 v52, v132 :: v_dual_mov_b32 v53, v133
	v_dual_mov_b32 v54, v134 :: v_dual_mov_b32 v55, v135
	v_dual_mov_b32 v42, v130 :: v_dual_mov_b32 v43, v131
	v_dual_mov_b32 v44, v132 :: v_dual_mov_b32 v45, v133
	v_dual_mov_b32 v46, v134 :: v_dual_mov_b32 v47, v135
	v_dual_mov_b32 v34, v130 :: v_dual_mov_b32 v35, v131
	v_dual_mov_b32 v36, v132 :: v_dual_mov_b32 v37, v133
	v_dual_mov_b32 v38, v134 :: v_dual_mov_b32 v39, v135
	v_dual_mov_b32 v26, v130 :: v_dual_mov_b32 v27, v131
	v_dual_mov_b32 v28, v132 :: v_dual_mov_b32 v29, v133
	v_dual_mov_b32 v30, v134 :: v_dual_mov_b32 v31, v135
	v_dual_mov_b32 v18, v130 :: v_dual_mov_b32 v19, v131
	v_dual_mov_b32 v20, v132 :: v_dual_mov_b32 v21, v133
	v_dual_mov_b32 v22, v134 :: v_dual_mov_b32 v23, v135
	v_dual_mov_b32 v10, v130 :: v_dual_mov_b32 v11, v131
	v_dual_mov_b32 v12, v132 :: v_dual_mov_b32 v13, v133
	v_dual_mov_b32 v14, v134 :: v_dual_mov_b32 v15, v135
	v_dual_mov_b32 v2, v130 :: v_dual_mov_b32 v3, v131
	v_dual_mov_b32 v4, v132 :: v_dual_mov_b32 v5, v133
	v_dual_mov_b32 v6, v134 :: v_dual_mov_b32 v7, v135
.LBB6_79:
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB6_81
; %bb.80:
	v_div_scale_f32 v129, null, v128, v128, 1.0
	v_div_scale_f32 v132, vcc_lo, 1.0, v128, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v130, v129
	v_fma_f32 v131, -v129, v130, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v130, v131, v130
	v_mul_f32_e32 v131, v132, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v133, -v129, v131, v132
	v_fmac_f32_e32 v131, v133, v130
	v_mul_lo_u32 v133, 0x1800, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v129, -v129, v131, v132
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v129, v129, v130, v131
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v131, v140, 8, v133
	v_cmp_lt_f32_e32 vcc_lo, 0, v128
	v_mov_b32_e32 v130, 0
	v_div_fixup_f32 v132, v129, v128, 1.0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_or_b32 v129, v150, 3, v131
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v131, 0, v132, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[128:129], 2, v[129:130]
	v_mul_f32_e32 v130, v206, v131
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v128, vcc_lo, s16, v128
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v129, null, s17, v129, vcc_lo
	v_dual_mul_f32 v120, v120, v130 :: v_dual_mul_f32 v121, v121, v130
	v_dual_mul_f32 v122, v122, v130 :: v_dual_mul_f32 v123, v123, v130
	v_dual_mul_f32 v124, v124, v130 :: v_dual_mul_f32 v103, v103, v130
	v_dual_mul_f32 v88, v88, v130 :: v_dual_mul_f32 v89, v89, v130
	v_dual_mul_f32 v90, v90, v130 :: v_dual_mul_f32 v91, v91, v130
	v_dual_mul_f32 v92, v92, v130 :: v_dual_mul_f32 v79, v79, v130
	v_dual_mul_f32 v64, v64, v130 :: v_dual_mul_f32 v65, v65, v130
	v_dual_mul_f32 v66, v66, v130 :: v_dual_mul_f32 v67, v67, v130
	v_dual_mul_f32 v68, v68, v130 :: v_dual_mul_f32 v55, v55, v130
	v_dual_mul_f32 v40, v40, v130 :: v_dual_mul_f32 v41, v41, v130
	v_dual_mul_f32 v42, v42, v130 :: v_dual_mul_f32 v43, v43, v130
	v_dual_mul_f32 v44, v44, v130 :: v_dual_mul_f32 v31, v31, v130
	v_dual_mul_f32 v16, v16, v130 :: v_dual_mul_f32 v17, v17, v130
	v_dual_mul_f32 v18, v18, v130 :: v_dual_mul_f32 v19, v19, v130
	v_dual_mul_f32 v20, v20, v130 :: v_dual_mul_f32 v125, v125, v130
	v_dual_mul_f32 v126, v126, v130 :: v_dual_mul_f32 v127, v127, v130
	v_dual_mul_f32 v112, v112, v130 :: v_dual_mul_f32 v93, v93, v130
	v_dual_mul_f32 v94, v94, v130 :: v_dual_mul_f32 v95, v95, v130
	v_dual_mul_f32 v80, v80, v130 :: v_dual_mul_f32 v69, v69, v130
	v_dual_mul_f32 v70, v70, v130 :: v_dual_mul_f32 v71, v71, v130
	v_dual_mul_f32 v56, v56, v130 :: v_dual_mul_f32 v45, v45, v130
	v_dual_mul_f32 v46, v46, v130 :: v_dual_mul_f32 v47, v47, v130
	v_dual_mul_f32 v32, v32, v130 :: v_dual_mul_f32 v21, v21, v130
	v_dual_mul_f32 v22, v22, v130 :: v_dual_mul_f32 v23, v23, v130
	v_dual_mul_f32 v8, v8, v130 :: v_dual_mul_f32 v113, v113, v130
	v_dual_mul_f32 v114, v114, v130 :: v_dual_mul_f32 v115, v115, v130
	v_dual_mul_f32 v116, v116, v130 :: v_dual_mul_f32 v81, v81, v130
	v_dual_mul_f32 v82, v82, v130 :: v_dual_mul_f32 v83, v83, v130
	v_dual_mul_f32 v84, v84, v130 :: v_dual_mul_f32 v57, v57, v130
	v_dual_mul_f32 v58, v58, v130 :: v_dual_mul_f32 v59, v59, v130
	v_dual_mul_f32 v60, v60, v130 :: v_dual_mul_f32 v33, v33, v130
	v_dual_mul_f32 v34, v34, v130 :: v_dual_mul_f32 v35, v35, v130
	v_dual_mul_f32 v36, v36, v130 :: v_dual_mul_f32 v9, v9, v130
	v_dual_mul_f32 v10, v10, v130 :: v_dual_mul_f32 v11, v11, v130
	v_dual_mul_f32 v12, v12, v130 :: v_dual_mul_f32 v117, v117, v130
	v_dual_mul_f32 v118, v118, v130 :: v_dual_mul_f32 v119, v119, v130
	v_dual_mul_f32 v104, v104, v130 :: v_dual_mul_f32 v85, v85, v130
	v_dual_mul_f32 v86, v86, v130 :: v_dual_mul_f32 v87, v87, v130
	v_dual_mul_f32 v72, v72, v130 :: v_dual_mul_f32 v61, v61, v130
	v_dual_mul_f32 v62, v62, v130 :: v_dual_mul_f32 v63, v63, v130
	v_dual_mul_f32 v48, v48, v130 :: v_dual_mul_f32 v37, v37, v130
	v_dual_mul_f32 v38, v38, v130 :: v_dual_mul_f32 v39, v39, v130
	v_dual_mul_f32 v24, v24, v130 :: v_dual_mul_f32 v13, v13, v130
	v_dual_mul_f32 v14, v14, v130 :: v_dual_mul_f32 v15, v15, v130
	v_dual_mul_f32 v0, v0, v130 :: v_dual_mul_f32 v105, v105, v130
	v_dual_mul_f32 v106, v106, v130 :: v_dual_mul_f32 v107, v107, v130
	v_dual_mul_f32 v108, v108, v130 :: v_dual_mul_f32 v73, v73, v130
	v_dual_mul_f32 v74, v74, v130 :: v_dual_mul_f32 v75, v75, v130
	v_dual_mul_f32 v76, v76, v130 :: v_dual_mul_f32 v49, v49, v130
	v_dual_mul_f32 v50, v50, v130 :: v_dual_mul_f32 v51, v51, v130
	v_dual_mul_f32 v52, v52, v130 :: v_dual_mul_f32 v25, v25, v130
	v_dual_mul_f32 v26, v26, v130 :: v_dual_mul_f32 v27, v27, v130
	v_dual_mul_f32 v28, v28, v130 :: v_dual_mul_f32 v1, v1, v130
	v_dual_mul_f32 v2, v2, v130 :: v_dual_mul_f32 v3, v3, v130
	v_dual_mul_f32 v4, v4, v130 :: v_dual_mul_f32 v109, v109, v130
	v_dual_mul_f32 v110, v110, v130 :: v_dual_mul_f32 v111, v111, v130
	v_dual_mul_f32 v96, v96, v130 :: v_dual_mul_f32 v97, v97, v130
	v_dual_mul_f32 v98, v98, v130 :: v_dual_mul_f32 v99, v99, v130
	v_dual_mul_f32 v100, v100, v130 :: v_dual_mul_f32 v101, v101, v130
	v_mul_f32_e32 v102, v102, v130
	s_clause 0x7
	global_store_b128 v[128:129], v[120:123], off
	global_store_b128 v[128:129], v[124:127], off offset:16
	global_store_b128 v[128:129], v[112:115], off offset:64
	global_store_b128 v[128:129], v[116:119], off offset:80
	global_store_b128 v[128:129], v[104:107], off offset:128
	global_store_b128 v[128:129], v[108:111], off offset:144
	global_store_b128 v[128:129], v[96:99], off offset:192
	global_store_b128 v[128:129], v[100:103], off offset:208
	v_dual_mul_f32 v77, v77, v130 :: v_dual_mul_f32 v78, v78, v130
	s_clause 0x5
	global_store_b128 v[128:129], v[88:91], off offset:256
	global_store_b128 v[128:129], v[92:95], off offset:272
	global_store_b128 v[128:129], v[80:83], off offset:320
	global_store_b128 v[128:129], v[84:87], off offset:336
	global_store_b128 v[128:129], v[72:75], off offset:384
	global_store_b128 v[128:129], v[76:79], off offset:400
	v_dual_mul_f32 v53, v53, v130 :: v_dual_mul_f32 v54, v54, v130
	s_clause 0x5
	global_store_b128 v[128:129], v[64:67], off offset:448
	global_store_b128 v[128:129], v[68:71], off offset:464
	global_store_b128 v[128:129], v[56:59], off offset:512
	global_store_b128 v[128:129], v[60:63], off offset:528
	global_store_b128 v[128:129], v[48:51], off offset:576
	global_store_b128 v[128:129], v[52:55], off offset:592
	v_dual_mul_f32 v29, v29, v130 :: v_dual_mul_f32 v30, v30, v130
	s_clause 0x5
	global_store_b128 v[128:129], v[40:43], off offset:640
	global_store_b128 v[128:129], v[44:47], off offset:656
	global_store_b128 v[128:129], v[32:35], off offset:704
	global_store_b128 v[128:129], v[36:39], off offset:720
	global_store_b128 v[128:129], v[24:27], off offset:768
	global_store_b128 v[128:129], v[28:31], off offset:784
	v_dual_mul_f32 v5, v5, v130 :: v_dual_mul_f32 v6, v6, v130
	v_mul_f32_e32 v7, v7, v130
	s_clause 0x5
	global_store_b128 v[128:129], v[16:19], off offset:832
	global_store_b128 v[128:129], v[20:23], off offset:848
	global_store_b128 v[128:129], v[8:11], off offset:896
	global_store_b128 v[128:129], v[12:15], off offset:912
	global_store_b128 v[128:129], v[0:3], off offset:960
	global_store_b128 v[128:129], v[4:7], off offset:976
.LBB6_81:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end6:
	.size	attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201, .Lfunc_end6-attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 68
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
		.amdhsa_next_free_vgpr 216
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.num_vgpr, 216
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.numbered_sgpr, 27
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.num_named_barrier, 3
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 12716
; TotalNumSgprs: 29
; NumVgprs: 216
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 26
; NumSGPRsForWavesPerEU: 29
; NumVGPRsForWavesPerEU: 216
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
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
	s_cbranch_vccnz .LBB7_62
; %bb.1:
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB7_62
; %bb.2:
	s_lshl_b32 s14, ttmp9, 7
	s_mul_i32 s2, s15, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s14, s2
	s_cbranch_scc1 .LBB7_62
; %bb.3:
	s_lshr_b32 s12, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s12, s17
	s_cbranch_scc1 .LBB7_62
; %bb.4:
	v_lshrrev_b32_e32 v5, 5, v0
	v_and_b32_e32 v6, 15, v0
	s_clause 0x1
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b64 s[18:19], s[0:1], 0x20
	s_mul_i32 s1, s3, 6
	v_dual_mov_b32 v148, s16 :: v_dual_lshlrev_b32 v7, 4, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v1, v7, v6
	v_add_nc_u32_e32 v2, s14, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_hi_i32 v1, 0x2aaaaaab, v2
	v_cmp_gt_i32_e64 s0, s2, v2
	v_lshrrev_b32_e32 v3, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v1, v1, v3
	v_mul_lo_u32 v3, v1, 6
	v_mul_lo_u32 v4, v1, 24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v3, v2, v3
	v_add3_u32 v143, v3, s1, v4
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB7_6
; %bb.5:
	v_mov_b32_e32 v144, 0
	s_mul_i32 s20, s15, 0x1800
	s_mov_b32 s21, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[20:21], s[4:5], s[20:21]
	v_lshlrev_b64_e32 v[2:3], 2, v[143:144]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s20, v2
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	global_load_b32 v2, v[2:3], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v148, s16, v2
.LBB7_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_dual_mov_b32 v3, -1 :: v_dual_and_b32 v4, 31, v0
	v_bfrev_b32_e32 v8, -2
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_u32_e32 22, v4
	s_cbranch_execz .LBB7_10
; %bb.7:
	s_mul_hi_i32 s13, s14, 0x2aaaaaab
	v_bfrev_b32_e32 v8, -2
	s_lshr_b32 s16, s13, 31
	v_mov_b32_e32 v3, -1
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v2, s13, s16, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v2
	s_cbranch_execz .LBB7_9
; %bb.8:
	v_ashrrev_i32_e32 v3, 31, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	v_add_co_u32 v2, vcc_lo, s18, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v3, null, s19, v3, vcc_lo
	global_load_b32 v3, v[2:3], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v8, v3
.LBB7_9:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB7_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_cvt_f32_u32 s1, s17
	s_add_co_i32 s13, s17, 0x1ff
	v_lshrrev_b32_e32 v149, 4, v4
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s13, s13, 0xffff
	v_xor_b32_e32 v9, 16, v2
	v_xor_b32_e32 v11, 8, v2
	v_s_rcp_f32 s15, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s13, s13
	s_mov_b32 s21, 0
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(TRANS32_DEP_1)
	s_mul_f32 s15, s13, s15
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v2, v9, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_wait_alu depctr_sa_sdst(0)
	s_trunc_f32 s15, s15
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v9, 2, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v2, v11, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s16, s15, 0x80000000
	s_cvt_u32_f32 s15, s15
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s13, s16, s1
	ds_bpermute_b32 v10, v9, v3
	ds_bpermute_b32 v9, v9, v8
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s13, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s13, s1
	s_add_co_ci_u32 s1, s15, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s12, s1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s1, s13, s1
	s_lshl_b32 s13, s13, 6
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s1, s1, 6
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v10
	s_wait_dscnt 0x0
	v_min_i32_e32 v8, v8, v9
	ds_bpermute_b32 v10, v11, v3
	ds_bpermute_b32 v9, v11, v8
	v_xor_b32_e32 v11, 4, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v2, v11, vcc_lo
	v_lshlrev_b32_e32 v11, 2, v11
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v10
	s_wait_dscnt 0x0
	v_min_i32_e32 v8, v8, v9
	ds_bpermute_b32 v10, v11, v3
	ds_bpermute_b32 v9, v11, v8
	v_xor_b32_e32 v11, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v2, v11, vcc_lo
	v_lshlrev_b32_e32 v150, 2, v11
	v_xor_b32_e32 v11, 1, v2
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v10
	s_wait_dscnt 0x0
	v_min_i32_e32 v8, v8, v9
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	ds_bpermute_b32 v10, v150, v3
	ds_bpermute_b32 v9, v150, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v2, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v151, 2, v2
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v10
	s_wait_dscnt 0x0
	v_min_i32_e32 v8, v8, v9
	ds_bpermute_b32 v2, v151, v3
	ds_bpermute_b32 v9, v151, v8
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s15, v2
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v8, v9
	s_add_co_i32 s15, s15, 1
	v_readfirstlane_b32 s16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_min_i32 s22, s15, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s13, s22
	s_cbranch_scc1 .LBB7_57
; %bb.11:
	v_lshrrev_b32_e32 v3, 6, v0
	v_and_or_b32 v152, v7, 48, v6
	v_bfe_u32 v7, v0, 5, 1
	v_dual_mov_b32 v12, 0x6020400 :: v_dual_and_b32 v11, 3, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v131, 0 :: v_dual_lshlrev_b32 v2, 4, v3
	s_addk_co_i32 s14, 0x7f
	v_dual_mov_b32 v14, 0x5040100 :: v_dual_lshlrev_b32 v13, 3, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_or_b32 v153, v7, 3, v2
	v_dual_mov_b32 v133, v131 :: v_dual_and_b32 v2, 1, v0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s14, s2
	v_dual_mov_b32 v132, v131 :: v_dual_and_b32 v9, 0x7f, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	v_ashrrev_i32_e32 v2, 31, v1
	s_cselect_b32 s23, -1, 0
	s_lshl_b32 s20, s3, 8
	v_lshrrev_b32_e32 v6, 2, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v154, 0x3070105, v12, vcc_lo
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_cmp_gt_u32_e32 vcc_lo, 2, v11
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[14:15], s[6:7], s[20:21]
	s_add_nc_u64 s[24:25], s[8:9], s[20:21]
	s_lshl_b32 s20, s3, 1
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v155, 0x3020706, v14 :: v_dual_lshlrev_b32 v10, 3, v4
	v_add_co_u32 v146, s3, s18, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v147, null, s19, v2, s3
	v_dual_mov_b32 v134, v131 :: v_dual_and_b32 v1, 16, v0
	v_or_b32_e32 v2, 0x100, v0
	v_or_b32_e32 v14, 0x200, v0
	v_or_b32_e32 v16, 0x300, v0
	v_lshl_or_b32 v6, v11, 6, v6
	v_dual_mov_b32 v136, v131 :: v_dual_add_nc_u32 v157, 0, v1
	v_lshrrev_b32_e32 v1, 5, v2
	v_and_or_b32 v2, 0x180, v2, v9
	v_lshrrev_b32_e32 v17, 5, v14
	v_lshrrev_b32_e32 v18, 5, v16
	v_lshl_add_u32 v8, v5, 11, 0
	v_dual_mov_b32 v135, v131 :: v_dual_lshlrev_b32 v158, 3, v149
	v_dual_mov_b32 v138, v131 :: v_dual_lshlrev_b32 v15, 4, v0
	v_and_or_b32 v14, 0x280, v14, v9
	v_and_or_b32 v9, 0x380, v16, v9
	v_dual_mov_b32 v137, v131 :: v_dual_lshlrev_b32 v2, 4, v2
	v_and_or_b32 v16, v17, 20, v149
	v_and_or_b32 v17, v18, 28, v149
	v_lshlrev_b32_e32 v18, 2, v6
	v_lshlrev_b32_e32 v11, 5, v11
	v_add_co_u32 v144, s2, s24, v10
	v_add_nc_u32_e32 v159, v8, v10
	v_xad_u32 v160, 0x120, v10, v8
	v_xad_u32 v161, 0x124, v10, v8
	v_xad_u32 v162, 0x240, v10, v8
	v_xad_u32 v163, 0x244, v10, v8
	v_xad_u32 v164, 0x360, v10, v8
	v_xad_u32 v165, 0x364, v10, v8
	v_xad_u32 v166, 0x520, v10, v8
	v_xad_u32 v167, 0x524, v10, v8
	v_xad_u32 v168, 0x640, v10, v8
	v_xad_u32 v169, 0x644, v10, v8
	v_xad_u32 v170, 0x760, v10, v8
	v_xad_u32 v171, 0x764, v10, v8
	v_add3_u32 v172, v8, v18, v11
	v_or_b32_e32 v10, 8, v6
	v_or_b32_e32 v11, 0x108, v6
	v_or_b32_e32 v18, 12, v6
	v_or_b32_e32 v19, 0x10c, v6
	v_or_b32_e32 v20, 24, v6
	v_xor_b32_e32 v10, v10, v13
	v_xor_b32_e32 v11, v11, v13
	v_xor_b32_e32 v18, v18, v13
	v_xor_b32_e32 v19, v19, v13
	v_xor_b32_e32 v20, v20, v13
	v_lshl_add_u32 v173, v10, 2, v8
	v_lshl_add_u32 v174, v11, 2, v8
	v_lshl_add_u32 v175, v18, 2, v8
	v_lshl_add_u32 v176, v19, 2, v8
	v_or_b32_e32 v10, 16, v6
	v_or_b32_e32 v11, 0x110, v6
	v_or_b32_e32 v18, 20, v6
	v_or_b32_e32 v19, 0x114, v6
	v_lshl_add_u32 v181, v20, 2, v8
	v_xor_b32_e32 v10, v10, v13
	v_xor_b32_e32 v11, v11, v13
	v_xor_b32_e32 v18, v18, v13
	v_xor_b32_e32 v19, v19, v13
	v_or_b32_e32 v20, 0x128, v6
	v_lshl_add_u32 v177, v10, 2, v8
	v_lshl_add_u32 v178, v11, 2, v8
	v_lshl_add_u32 v179, v18, 2, v8
	v_lshl_add_u32 v180, v19, 2, v8
	v_or_b32_e32 v10, 0x118, v6
	v_or_b32_e32 v11, 28, v6
	v_or_b32_e32 v18, 0x11c, v6
	v_or_b32_e32 v19, 40, v6
	v_cmp_eq_u32_e32 vcc_lo, v149, v7
	v_xor_b32_e32 v10, v10, v13
	v_xor_b32_e32 v11, v11, v13
	v_xor_b32_e32 v18, v18, v13
	v_xor_b32_e32 v19, v19, v13
	v_lshlrev_b32_e32 v7, 8, v143
	v_lshl_add_u32 v182, v10, 2, v8
	v_lshl_add_u32 v183, v11, 2, v8
	v_lshl_add_u32 v184, v18, 2, v8
	v_lshl_add_u32 v185, v19, 2, v8
	v_or_b32_e32 v10, 44, v6
	v_or_b32_e32 v11, 0x12c, v6
	v_or_b32_e32 v18, 48, v6
	v_or_b32_e32 v19, 0x130, v6
	v_xor_b32_e32 v20, v20, v13
	v_xor_b32_e32 v10, v10, v13
	v_xor_b32_e32 v11, v11, v13
	v_xor_b32_e32 v18, v18, v13
	v_xor_b32_e32 v19, v19, v13
	v_cndmask_b32_e64 v7, 0, v7, s0
	v_lshl_add_u32 v186, v20, 2, v8
	v_or_b32_e32 v20, 52, v6
	v_lshl_add_u32 v187, v10, 2, v8
	v_lshl_add_u32 v188, v11, 2, v8
	v_lshl_add_u32 v189, v18, 2, v8
	v_lshl_add_u32 v190, v19, 2, v8
	v_or_b32_e32 v10, 0x134, v6
	v_or_b32_e32 v11, 56, v6
	v_or_b32_e32 v18, 0x138, v6
	v_or_b32_e32 v19, 60, v6
	v_or_b32_e32 v6, 0x13c, v6
	v_cmp_gt_u32_e64 s1, 0x100, v0
	v_add_co_u32 v7, s3, s4, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s5, 0, s3
	v_and_or_b32 v5, v5, 4, v149
	v_dual_mov_b32 v209, 1.0 :: v_dual_lshlrev_b32 v14, 4, v14
	v_dual_mov_b32 v210, 0xff800000 :: v_dual_lshlrev_b32 v9, 4, v9
	v_lshlrev_b32_e32 v4, 4, v4
	v_lshl_add_u32 v3, v3, 12, 0
	v_xor_b32_e32 v20, v20, v13
	v_xor_b32_e32 v10, v10, v13
	v_xor_b32_e32 v11, v11, v13
	v_xor_b32_e32 v18, v18, v13
	v_xor_b32_e32 v19, v19, v13
	v_xor_b32_e32 v6, v6, v13
	v_and_or_b32 v1, v1, 12, v149
	s_and_b32 s24, s1, vcc_lo
	v_add_co_u32 v198, vcc_lo, v7, v158
	v_lshl_add_u32 v191, v20, 2, v8
	v_lshl_add_u32 v192, v10, 2, v8
	v_lshl_add_u32 v193, v11, 2, v8
	v_lshl_add_u32 v194, v18, 2, v8
	v_lshl_add_u32 v195, v19, 2, v8
	v_lshl_add_u32 v196, v6, 2, v8
	v_add_nc_u32_e32 v197, 0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v199, null, 0, v12, vcc_lo
	v_lshlrev_b32_e32 v200, 3, v5
	v_lshlrev_b32_e32 v203, 3, v16
	v_lshlrev_b32_e32 v204, 3, v17
	v_add_nc_u32_e32 v205, 0, v9
	v_add_nc_u32_e32 v206, v3, v4
	v_add_nc_u32_e32 v207, 0, v15
	v_dual_mov_b32 v3, v131 :: v_dual_add_nc_u32 v208, 0, v14
	v_mov_b32_e32 v11, v131
	v_dual_mov_b32 v19, v131 :: v_dual_mov_b32 v20, v132
	v_dual_mov_b32 v27, v131 :: v_dual_mov_b32 v28, v132
	v_dual_mov_b32 v35, v131 :: v_dual_mov_b32 v36, v132
	v_dual_mov_b32 v43, v131 :: v_dual_mov_b32 v44, v132
	v_dual_mov_b32 v51, v131 :: v_dual_mov_b32 v52, v132
	v_dual_mov_b32 v59, v131 :: v_dual_mov_b32 v60, v132
	v_dual_mov_b32 v67, v131 :: v_dual_mov_b32 v68, v132
	v_dual_mov_b32 v75, v131 :: v_dual_mov_b32 v76, v132
	v_dual_mov_b32 v83, v131 :: v_dual_mov_b32 v84, v132
	v_dual_mov_b32 v91, v131 :: v_dual_mov_b32 v92, v132
	v_dual_mov_b32 v99, v131 :: v_dual_mov_b32 v100, v132
	v_dual_mov_b32 v107, v131 :: v_dual_mov_b32 v108, v132
	v_dual_mov_b32 v115, v131 :: v_dual_mov_b32 v116, v132
	v_dual_mov_b32 v123, v131 :: v_dual_mov_b32 v124, v132
	v_add_co_ci_u32_e64 v145, null, s25, 0, s2
	v_cmp_gt_u32_e64 s2, 64, v0
	v_lshl_add_u32 v156, v0, 1, 0
	v_dual_mov_b32 v10, v138 :: v_dual_lshlrev_b32 v201, 3, v1
	v_dual_mov_b32 v9, v137 :: v_dual_add_nc_u32 v202, 0, v2
	v_dual_mov_b32 v4, v132 :: v_dual_mov_b32 v5, v133
	v_dual_mov_b32 v6, v134 :: v_dual_mov_b32 v7, v135
	v_mov_b32_e32 v8, v136
	v_dual_mov_b32 v12, v132 :: v_dual_mov_b32 v13, v133
	v_dual_mov_b32 v14, v134 :: v_dual_mov_b32 v15, v135
	v_dual_mov_b32 v16, v136 :: v_dual_mov_b32 v17, v137
	v_dual_mov_b32 v18, v138 :: v_dual_mov_b32 v21, v133
	v_dual_mov_b32 v22, v134 :: v_dual_mov_b32 v23, v135
	v_dual_mov_b32 v24, v136 :: v_dual_mov_b32 v25, v137
	v_dual_mov_b32 v26, v138 :: v_dual_mov_b32 v29, v133
	v_dual_mov_b32 v30, v134 :: v_dual_mov_b32 v31, v135
	v_dual_mov_b32 v32, v136 :: v_dual_mov_b32 v33, v137
	v_dual_mov_b32 v34, v138 :: v_dual_mov_b32 v37, v133
	v_dual_mov_b32 v38, v134 :: v_dual_mov_b32 v39, v135
	v_dual_mov_b32 v40, v136 :: v_dual_mov_b32 v41, v137
	v_dual_mov_b32 v42, v138 :: v_dual_mov_b32 v45, v133
	v_dual_mov_b32 v46, v134 :: v_dual_mov_b32 v47, v135
	v_dual_mov_b32 v48, v136 :: v_dual_mov_b32 v49, v137
	v_dual_mov_b32 v50, v138 :: v_dual_mov_b32 v53, v133
	v_dual_mov_b32 v54, v134 :: v_dual_mov_b32 v55, v135
	v_dual_mov_b32 v56, v136 :: v_dual_mov_b32 v57, v137
	v_dual_mov_b32 v58, v138 :: v_dual_mov_b32 v61, v133
	v_dual_mov_b32 v62, v134 :: v_dual_mov_b32 v63, v135
	v_dual_mov_b32 v64, v136 :: v_dual_mov_b32 v65, v137
	v_dual_mov_b32 v66, v138 :: v_dual_mov_b32 v69, v133
	v_dual_mov_b32 v70, v134 :: v_dual_mov_b32 v71, v135
	v_dual_mov_b32 v72, v136 :: v_dual_mov_b32 v73, v137
	v_dual_mov_b32 v74, v138 :: v_dual_mov_b32 v77, v133
	v_dual_mov_b32 v78, v134 :: v_dual_mov_b32 v79, v135
	v_dual_mov_b32 v80, v136 :: v_dual_mov_b32 v81, v137
	v_dual_mov_b32 v82, v138 :: v_dual_mov_b32 v85, v133
	v_dual_mov_b32 v86, v134 :: v_dual_mov_b32 v87, v135
	v_dual_mov_b32 v88, v136 :: v_dual_mov_b32 v89, v137
	v_dual_mov_b32 v90, v138 :: v_dual_mov_b32 v93, v133
	v_dual_mov_b32 v94, v134 :: v_dual_mov_b32 v95, v135
	v_dual_mov_b32 v96, v136 :: v_dual_mov_b32 v97, v137
	v_dual_mov_b32 v98, v138 :: v_dual_mov_b32 v101, v133
	v_dual_mov_b32 v102, v134 :: v_dual_mov_b32 v103, v135
	v_dual_mov_b32 v104, v136 :: v_dual_mov_b32 v105, v137
	v_dual_mov_b32 v106, v138 :: v_dual_mov_b32 v109, v133
	v_dual_mov_b32 v110, v134 :: v_dual_mov_b32 v111, v135
	v_dual_mov_b32 v112, v136 :: v_dual_mov_b32 v113, v137
	v_dual_mov_b32 v114, v138 :: v_dual_mov_b32 v117, v133
	v_dual_mov_b32 v118, v134 :: v_dual_mov_b32 v119, v135
	v_dual_mov_b32 v120, v136 :: v_dual_mov_b32 v121, v137
	v_dual_mov_b32 v122, v138 :: v_dual_mov_b32 v125, v133
	v_dual_mov_b32 v126, v134 :: v_dual_mov_b32 v127, v135
	v_dual_mov_b32 v128, v136 :: v_dual_mov_b32 v129, v137
	v_mov_b32_e32 v130, v138
	v_mov_b32_e32 v2, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[18:19], s[6:7], s[20:21]
	s_add_nc_u64 s[8:9], s[8:9], s[20:21]
	s_mov_b32 s7, 0x76543210
	s_branch .LBB7_13
.LBB7_12:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v210, v1
	s_add_co_i32 s13, s13, 64
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s13, s22
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB7_58
.LBB7_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB7_49 Depth 2
                                        ;       Child Loop BB7_51 Depth 3
	v_or_b32_e32 v1, s13, v152
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[136:137], null, 0x408, v1, s[14:15]
	v_cmp_le_i32_e32 vcc_lo, s22, v1
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
; %bb.14:                               ;   in Loop: Header=BB7_13 Depth=1
	v_dual_mov_b32 v132, v131 :: v_dual_mov_b32 v133, v131
	v_mov_b32_e32 v134, v131
	ds_store_b128 v207, v[131:134]
; %bb.15:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s4, s3
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v132, 0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s4
	s_cbranch_execz .LBB7_17
; %bb.16:                               ;   in Loop: Header=BB7_13 Depth=1
	v_add_co_u32 v132, s3, v136, v200
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v133, null, 0, v137, s3
	v_add_co_u32 v134, s3, v136, v201
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v135, null, 0, v137, s3
	s_clause 0x3
	global_load_b64 v[138:139], v[132:133], off
	global_load_b64 v[140:141], v[132:133], off offset:16
	global_load_b64 v[132:133], v[134:135], off
	global_load_b64 v[134:135], v[134:135], off offset:16
	s_wait_loadcnt 0x2
	ds_store_b128 v207, v[138:141]
.LBB7_17:                               ;   in Loop: Header=BB7_13 Depth=1
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	ds_store_b128 v202, v[132:135]
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
; %bb.18:                               ;   in Loop: Header=BB7_13 Depth=1
	v_dual_mov_b32 v132, v131 :: v_dual_mov_b32 v133, v131
	v_mov_b32_e32 v134, v131
                                        ; implicit-def: $vgpr136_vgpr137
	ds_store_b128 v208, v[131:134]
; %bb.19:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s3, s3
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v132, 0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB7_21
; %bb.20:                               ;   in Loop: Header=BB7_13 Depth=1
	v_add_co_u32 v132, vcc_lo, v136, v203
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v133, null, 0, v137, vcc_lo
	v_add_co_u32 v134, vcc_lo, v136, v204
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v135, null, 0, v137, vcc_lo
	s_clause 0x3
	global_load_b64 v[136:137], v[132:133], off
	global_load_b64 v[138:139], v[132:133], off offset:16
	global_load_b64 v[132:133], v[134:135], off
	global_load_b64 v[134:135], v[134:135], off offset:16
	s_wait_loadcnt 0x2
	ds_store_b128 v208, v[136:139]
.LBB7_21:                               ;   in Loop: Header=BB7_13 Depth=1
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt 0x0
	ds_store_b128 v205, v[132:135]
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB7_39
; %bb.22:                               ;   in Loop: Header=BB7_13 Depth=1
	v_or_b32_e32 v1, s13, v153
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s22, v1
	s_cbranch_execz .LBB7_24
; %bb.23:                               ;   in Loop: Header=BB7_13 Depth=1
	v_mad_co_i64_i32 v[132:133], null, 0x408, v1, v[144:145]
	global_load_b64 v[132:133], v[132:133], off
.LBB7_24:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v136, 1, v1
	v_add_nc_u32_e32 v137, 0x8000, v159
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v137, v132, v133 offset1:1
	v_cmpx_gt_i32_e64 s22, v136
	s_cbranch_execz .LBB7_26
; %bb.25:                               ;   in Loop: Header=BB7_13 Depth=1
	v_mad_co_i64_i32 v[132:133], null, 0x408, v136, v[144:145]
	global_load_b64 v[134:135], v[132:133], off
.LBB7_26:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v138, 2, v1
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v160, v134 offset:32768
	ds_store_b32 v161, v135 offset:32768
	v_cmpx_gt_i32_e64 s22, v138
	s_cbranch_execz .LBB7_28
; %bb.27:                               ;   in Loop: Header=BB7_13 Depth=1
	v_mad_co_i64_i32 v[134:135], null, 0x408, v138, v[144:145]
	global_load_b64 v[136:137], v[134:135], off
.LBB7_28:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v134, 3, v1
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v162, v136 offset:32768
	ds_store_b32 v163, v137 offset:32768
	v_cmpx_gt_i32_e64 s22, v134
	s_cbranch_execz .LBB7_30
; %bb.29:                               ;   in Loop: Header=BB7_13 Depth=1
	v_mad_co_i64_i32 v[132:133], null, 0x408, v134, v[144:145]
	global_load_b64 v[132:133], v[132:133], off
.LBB7_30:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v138, 4, v1
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v164, v132 offset:32768
	ds_store_b32 v165, v133 offset:32768
	v_cmpx_gt_i32_e64 s22, v138
	s_cbranch_execz .LBB7_32
; %bb.31:                               ;   in Loop: Header=BB7_13 Depth=1
	v_mad_co_i64_i32 v[132:133], null, 0x408, v138, v[144:145]
	global_load_b64 v[136:137], v[132:133], off
.LBB7_32:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v132, 5, v1
	v_add_nc_u32_e32 v133, 0x8400, v159
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v133, v136, v137 offset1:1
	v_cmpx_gt_i32_e64 s22, v132
	s_cbranch_execz .LBB7_34
; %bb.33:                               ;   in Loop: Header=BB7_13 Depth=1
	v_mad_co_i64_i32 v[132:133], null, 0x408, v132, v[144:145]
	global_load_b64 v[134:135], v[132:133], off
.LBB7_34:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v138, 6, v1
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v166, v134 offset:32768
	ds_store_b32 v167, v135 offset:32768
	v_cmpx_gt_i32_e64 s22, v138
	s_cbranch_execz .LBB7_36
; %bb.35:                               ;   in Loop: Header=BB7_13 Depth=1
	v_mad_co_i64_i32 v[134:135], null, 0x408, v138, v[144:145]
	global_load_b64 v[136:137], v[134:135], off
.LBB7_36:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v1, 7, v1
	s_mov_b32 s4, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v168, v136 offset:32768
	ds_store_b32 v169, v137 offset:32768
	v_cmpx_gt_i32_e64 s22, v1
	s_cbranch_execz .LBB7_38
; %bb.37:                               ;   in Loop: Header=BB7_13 Depth=1
	v_mad_co_i64_i32 v[132:133], null, 0x408, v1, v[144:145]
	global_load_b64 v[132:133], v[132:133], off
.LBB7_38:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	ds_store_b32 v170, v132 offset:32768
	ds_store_b32 v171, v133 offset:32768
.LBB7_39:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s3, s24
	s_cbranch_execz .LBB7_41
; %bb.40:                               ;   in Loop: Header=BB7_13 Depth=1
	v_add_nc_u32_e32 v1, 0x8000, v172
	v_add_nc_u32_e32 v136, 0x8400, v172
	ds_load_2addr_b32 v[132:133], v1 offset1:4
	ds_load_2addr_b32 v[134:135], v136 offset1:4
	s_wait_dscnt 0x1
	ds_bpermute_b32 v137, v151, v132
	s_wait_dscnt 0x1
	ds_bpermute_b32 v138, v151, v134
	ds_bpermute_b32 v139, v151, v133
	ds_bpermute_b32 v140, v151, v135
	s_wait_dscnt 0x3
	v_perm_b32 v132, v137, v132, v154
	s_wait_dscnt 0x2
	v_perm_b32 v134, v138, v134, v154
	s_wait_dscnt 0x1
	v_perm_b32 v137, v139, v133, v154
	s_wait_dscnt 0x0
	v_perm_b32 v135, v140, v135, v154
	ds_bpermute_b32 v133, v150, v132
	ds_bpermute_b32 v138, v150, v134
	ds_bpermute_b32 v139, v150, v137
	ds_bpermute_b32 v140, v150, v135
	s_wait_dscnt 0x3
	v_perm_b32 v132, v133, v132, v155
	s_wait_dscnt 0x2
	v_perm_b32 v133, v138, v134, v155
	s_wait_dscnt 0x1
	v_perm_b32 v134, v139, v137, v155
	s_wait_dscnt 0x0
	v_perm_b32 v135, v140, v135, v155
	ds_store_b128 v206, v[132:135] offset:16384
	ds_load_b32 v132, v173 offset:32768
	ds_load_b32 v133, v174 offset:32768
	ds_load_b32 v134, v175 offset:32768
	ds_load_b32 v135, v176 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v151, v132
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v151, v133
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v151, v134
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v151, v135
	s_wait_dscnt 0x3
	v_perm_b32 v132, v137, v132, v154
	s_wait_dscnt 0x2
	v_perm_b32 v133, v138, v133, v154
	s_wait_dscnt 0x1
	v_perm_b32 v134, v139, v134, v154
	s_wait_dscnt 0x0
	v_perm_b32 v135, v140, v135, v154
	ds_bpermute_b32 v137, v150, v132
	ds_bpermute_b32 v138, v150, v133
	ds_bpermute_b32 v139, v150, v134
	ds_bpermute_b32 v140, v150, v135
	s_wait_dscnt 0x3
	v_perm_b32 v132, v137, v132, v155
	s_wait_dscnt 0x2
	v_perm_b32 v133, v138, v133, v155
	s_wait_dscnt 0x1
	v_perm_b32 v134, v139, v134, v155
	s_wait_dscnt 0x0
	v_perm_b32 v135, v140, v135, v155
	ds_store_b128 v206, v[132:135] offset:16896
	ds_load_b32 v132, v177 offset:32768
	ds_load_b32 v133, v178 offset:32768
	ds_load_b32 v134, v179 offset:32768
	ds_load_b32 v135, v180 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v151, v132
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v151, v133
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v151, v134
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v151, v135
	s_wait_dscnt 0x3
	v_perm_b32 v132, v137, v132, v154
	s_wait_dscnt 0x2
	v_perm_b32 v133, v138, v133, v154
	s_wait_dscnt 0x1
	v_perm_b32 v134, v139, v134, v154
	s_wait_dscnt 0x0
	v_perm_b32 v135, v140, v135, v154
	ds_bpermute_b32 v137, v150, v132
	ds_bpermute_b32 v138, v150, v133
	ds_bpermute_b32 v139, v150, v134
	ds_bpermute_b32 v140, v150, v135
	s_wait_dscnt 0x3
	v_perm_b32 v132, v137, v132, v155
	s_wait_dscnt 0x2
	v_perm_b32 v133, v138, v133, v155
	s_wait_dscnt 0x1
	v_perm_b32 v134, v139, v134, v155
	s_wait_dscnt 0x0
	v_perm_b32 v135, v140, v135, v155
	ds_store_b128 v206, v[132:135] offset:17408
	ds_load_b32 v132, v181 offset:32768
	ds_load_b32 v133, v182 offset:32768
	ds_load_b32 v134, v183 offset:32768
	ds_load_b32 v135, v184 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v151, v132
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v151, v133
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v151, v134
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v151, v135
	s_wait_dscnt 0x3
	v_perm_b32 v132, v137, v132, v154
	s_wait_dscnt 0x2
	v_perm_b32 v133, v138, v133, v154
	s_wait_dscnt 0x1
	v_perm_b32 v134, v139, v134, v154
	s_wait_dscnt 0x0
	v_perm_b32 v135, v140, v135, v154
	ds_bpermute_b32 v137, v150, v132
	ds_bpermute_b32 v138, v150, v133
	ds_bpermute_b32 v139, v150, v134
	ds_bpermute_b32 v140, v150, v135
	s_wait_dscnt 0x3
	v_perm_b32 v132, v137, v132, v155
	s_wait_dscnt 0x2
	v_perm_b32 v133, v138, v133, v155
	s_wait_dscnt 0x1
	v_perm_b32 v134, v139, v134, v155
	s_wait_dscnt 0x0
	v_perm_b32 v135, v140, v135, v155
	ds_store_b128 v206, v[132:135] offset:17920
	ds_load_2addr_b32 v[132:133], v1 offset0:32 offset1:36
	ds_load_2addr_b32 v[134:135], v136 offset0:32 offset1:36
	s_wait_dscnt 0x1
	ds_bpermute_b32 v1, v151, v132
	s_wait_dscnt 0x1
	ds_bpermute_b32 v136, v151, v134
	ds_bpermute_b32 v137, v151, v133
	ds_bpermute_b32 v138, v151, v135
	s_wait_dscnt 0x3
	v_perm_b32 v1, v1, v132, v154
	s_wait_dscnt 0x2
	v_perm_b32 v134, v136, v134, v154
	s_wait_dscnt 0x1
	v_perm_b32 v136, v137, v133, v154
	s_wait_dscnt 0x0
	v_perm_b32 v135, v138, v135, v154
	ds_bpermute_b32 v132, v150, v1
	ds_bpermute_b32 v133, v150, v134
	ds_bpermute_b32 v137, v150, v136
	ds_bpermute_b32 v138, v150, v135
	s_wait_dscnt 0x3
	v_perm_b32 v132, v132, v1, v155
	s_wait_dscnt 0x2
	v_perm_b32 v133, v133, v134, v155
	s_wait_dscnt 0x1
	v_perm_b32 v134, v137, v136, v155
	s_wait_dscnt 0x0
	v_perm_b32 v135, v138, v135, v155
	ds_store_b128 v206, v[132:135] offset:18432
	ds_load_b32 v1, v185 offset:32768
	ds_load_b32 v132, v186 offset:32768
	ds_load_b32 v133, v187 offset:32768
	ds_load_b32 v134, v188 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v151, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v151, v132
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v151, v133
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v151, v134
	s_wait_dscnt 0x3
	v_perm_b32 v1, v135, v1, v154
	s_wait_dscnt 0x2
	v_perm_b32 v135, v136, v132, v154
	s_wait_dscnt 0x1
	v_perm_b32 v136, v137, v133, v154
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v134, v154
	ds_bpermute_b32 v132, v150, v1
	ds_bpermute_b32 v133, v150, v135
	ds_bpermute_b32 v134, v150, v136
	ds_bpermute_b32 v138, v150, v137
	s_wait_dscnt 0x3
	v_perm_b32 v132, v132, v1, v155
	s_wait_dscnt 0x2
	v_perm_b32 v133, v133, v135, v155
	s_wait_dscnt 0x1
	v_perm_b32 v134, v134, v136, v155
	s_wait_dscnt 0x0
	v_perm_b32 v135, v138, v137, v155
	ds_store_b128 v206, v[132:135] offset:18944
	ds_load_b32 v1, v189 offset:32768
	ds_load_b32 v132, v190 offset:32768
	ds_load_b32 v133, v191 offset:32768
	ds_load_b32 v134, v192 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v151, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v151, v132
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v151, v133
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v151, v134
	s_wait_dscnt 0x3
	v_perm_b32 v1, v135, v1, v154
	s_wait_dscnt 0x2
	v_perm_b32 v135, v136, v132, v154
	s_wait_dscnt 0x1
	v_perm_b32 v136, v137, v133, v154
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v134, v154
	ds_bpermute_b32 v132, v150, v1
	ds_bpermute_b32 v133, v150, v135
	ds_bpermute_b32 v134, v150, v136
	ds_bpermute_b32 v138, v150, v137
	s_wait_dscnt 0x3
	v_perm_b32 v132, v132, v1, v155
	s_wait_dscnt 0x2
	v_perm_b32 v133, v133, v135, v155
	s_wait_dscnt 0x1
	v_perm_b32 v134, v134, v136, v155
	s_wait_dscnt 0x0
	v_perm_b32 v135, v138, v137, v155
	ds_store_b128 v206, v[132:135] offset:19456
	ds_load_b32 v1, v193 offset:32768
	ds_load_b32 v132, v194 offset:32768
	ds_load_b32 v133, v195 offset:32768
	ds_load_b32 v134, v196 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v151, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v151, v132
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v151, v133
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v151, v134
	s_wait_dscnt 0x3
	v_perm_b32 v1, v135, v1, v154
	s_wait_dscnt 0x2
	v_perm_b32 v135, v136, v132, v154
	s_wait_dscnt 0x1
	v_perm_b32 v136, v137, v133, v154
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v134, v154
	ds_bpermute_b32 v132, v150, v1
	ds_bpermute_b32 v133, v150, v135
	ds_bpermute_b32 v134, v150, v136
	ds_bpermute_b32 v138, v150, v137
	s_wait_dscnt 0x3
	v_perm_b32 v132, v132, v1, v155
	s_wait_dscnt 0x2
	v_perm_b32 v133, v133, v135, v155
	s_wait_dscnt 0x1
	v_perm_b32 v134, v134, v136, v155
	s_wait_dscnt 0x0
	v_perm_b32 v135, v138, v137, v155
	ds_store_b128 v206, v[132:135] offset:19968
.LBB7_41:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB7_45
; %bb.42:                               ;   in Loop: Header=BB7_13 Depth=1
	v_or_b32_e32 v132, s13, v0
	v_mov_b32_e32 v1, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s22, v132
	s_cbranch_execz .LBB7_44
; %bb.43:                               ;   in Loop: Header=BB7_13 Depth=1
	v_mad_co_i64_i32 v[133:134], null, 0x408, v132, s[18:19]
	v_mad_co_i64_i32 v[135:136], null, 0x408, v132, s[8:9]
	global_load_d16_b16 v1, v[133:134], off offset:1024
	global_load_d16_hi_b16 v1, v[135:136], off offset:1024
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v132.h, 8, v1.l
	v_lshrrev_b16 v132.l, 8, v1.h
	v_and_b16 v133.h, 0xff, v1.l
	v_and_b16 v133.l, 0xff, v1.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_lshlrev_b16 v1, 8, v132 op_sel_hi:[0,1]
	v_or_b32_e32 v1, v1, v133
.LBB7_44:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	ds_store_b16_d16_hi v156, v1 offset:49152
	ds_store_b16 v156, v1 offset:49280
.LBB7_45:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_or_b32 s3, s13, 63
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s3, s22
	v_mov_b32_e32 v211, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_le_i32 s3, s16
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s4, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s20, s23, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s20, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s0, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB7_47
; %bb.46:                               ;   in Loop: Header=BB7_13 Depth=1
	global_load_b32 v211, v[146:147], off
.LBB7_47:                               ;   in Loop: Header=BB7_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_mov_b32 s21, 0
	s_branch .LBB7_49
.LBB7_48:                               ;   in Loop: Header=BB7_49 Depth=2
	s_or_b32 exec_lo, exec_lo, s3
	v_sub_f32_e32 v212, v210, v1
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v210
	v_div_scale_f32 v217, null, v132, v132, v140
	v_div_scale_f32 v219, null, v132, v132, v139
	v_div_scale_f32 v222, null, v132, v132, v135
	v_mul_f32_e32 v212, 0x3fb8aa3b, v212
	v_div_scale_f32 v221, null, v132, v132, v137
	v_rcp_f32_e32 v218, v217
	v_lshl_add_u32 v232, s21, 12, v197
	v_add_f32_e32 v133, v133, v134
	v_exp_f32_e32 v212, v212
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v220, -v217, v218, 1.0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v210, 0, v212, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v218, v220, v218
	v_mul_f32_e32 v209, v209, v210
	v_fmac_f32_e32 v133, v2, v210
	v_div_scale_f32 v2, null, v132, v132, v142
	v_div_scale_f32 v210, null, v132, v132, v141
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_scale_f32 v212, null, v132, v132, v209
	v_div_scale_f32 v215, vcc_lo, v209, v132, v209
	v_rcp_f32_e32 v213, v212
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v214, -v212, v213, 1.0
	v_fmac_f32_e32 v213, v214, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v214, v215, v213
	v_fma_f32 v216, -v212, v214, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v214, v216, v213
	v_fma_f32 v212, -v212, v214, v215
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v212, v212, v213, v214
	v_rcp_f32_e32 v213, v210
	v_div_fixup_f32 v134, v212, v132, v209
	v_rcp_f32_e32 v209, v2
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v215, -v210, v213, 1.0
	v_mul_f32_e32 v16, v16, v134
	v_dual_mul_f32 v130, v130, v134 :: v_dual_mul_f32 v127, v127, v134
	v_mul_f32_e32 v129, v129, v134
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fma_f32 v212, -v2, v209, 1.0
	v_dual_mul_f32 v22, v22, v134 :: v_dual_fmac_f32 v213, v215, v213
	v_div_scale_f32 v215, s3, v141, v132, v141
	v_dual_mul_f32 v128, v128, v134 :: v_dual_mul_f32 v125, v125, v134
	v_fmac_f32_e32 v209, v212, v209
	v_div_scale_f32 v212, vcc_lo, v142, v132, v142
	v_dual_mul_f32 v126, v126, v134 :: v_dual_mul_f32 v123, v123, v134
	v_dual_mul_f32 v15, v15, v134 :: v_dual_mul_f32 v124, v124, v134
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v121, v121, v134 :: v_dual_mul_f32 v214, v212, v209
	v_dual_mul_f32 v122, v122, v134 :: v_dual_mul_f32 v119, v119, v134
	v_dual_mul_f32 v120, v120, v134 :: v_dual_mul_f32 v117, v117, v134
	v_fma_f32 v216, -v2, v214, v212
	v_mul_f32_e32 v11, v11, v134
	v_dual_mul_f32 v9, v9, v134 :: v_dual_mul_f32 v118, v118, v134
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v115, v115, v134 :: v_dual_fmac_f32 v214, v216, v209
	v_mul_f32_e32 v216, v215, v213
	v_dual_mul_f32 v116, v116, v134 :: v_dual_mul_f32 v113, v113, v134
	v_dual_mul_f32 v114, v114, v134 :: v_dual_mul_f32 v111, v111, v134
	v_fma_f32 v2, -v2, v214, v212
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v212, -v210, v216, v215
	v_dual_mul_f32 v112, v112, v134 :: v_dual_mul_f32 v109, v109, v134
	v_dual_mul_f32 v110, v110, v134 :: v_dual_mul_f32 v107, v107, v134
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v2, v2, v209, v214
	v_fmac_f32_e32 v216, v212, v213
	v_div_scale_f32 v212, s4, v140, v132, v140
	v_rcp_f32_e32 v214, v219
	s_mov_b32 vcc_lo, s3
	v_div_fixup_f32 v2, v2, v132, v142
	v_fma_f32 v142, -v210, v216, v215
	v_mul_f32_e32 v215, v212, v218
	v_mov_b16_e64 v210.l, v131.l
	v_div_scale_f32 v220, s3, v139, v132, v139
	v_mov_b16_e64 v210.h, 0
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v142, v142, v213, v216
	v_div_scale_f32 v213, null, v132, v132, v138
	v_fma_f32 v209, -v219, v214, 1.0
	v_fma_f32 v216, -v217, v215, v212
	s_mov_b32 vcc_lo, s4
	v_div_fixup_f32 v141, v142, v132, v141
	v_rcp_f32_e32 v142, v213
	v_fmac_f32_e32 v214, v209, v214
	v_mov_b16_e64 v209.l, v210.l
	v_fmac_f32_e32 v215, v216, v218
	v_mov_b16_e64 v209.h, v210.h
	v_dual_mul_f32 v108, v108, v134 :: v_dual_mul_f32 v105, v105, v134
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_pk_fp8_f32 v209.l, v2, v141
	v_fma_f32 v2, -v217, v215, v212
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fma_f32 v212, -v213, v142, 1.0
	v_mul_f32_e32 v216, v220, v214
	v_rcp_f32_e32 v217, v221
	v_dual_mul_f32 v106, v106, v134 :: v_dual_mul_f32 v103, v103, v134
	v_fmac_f32_e32 v142, v212, v142
	v_rcp_f32_e32 v212, v222
	v_fma_f32 v141, -v219, v216, v220
	v_dual_mul_f32 v104, v104, v134 :: v_dual_mul_f32 v101, v101, v134
	v_dual_mul_f32 v102, v102, v134 :: v_dual_mul_f32 v99, v99, v134
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v216, v141, v214
	v_div_scale_f32 v141, s4, v138, v132, v138
	v_dual_mul_f32 v100, v100, v134 :: v_dual_mul_f32 v97, v97, v134
	v_fma_f32 v223, -v222, v212, 1.0
	v_dual_mul_f32 v98, v98, v134 :: v_dual_mul_f32 v95, v95, v134
	v_dual_mul_f32 v96, v96, v134 :: v_dual_mul_f32 v93, v93, v134
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v212, v223, v212
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v2, v2, v218, v215
	v_fma_f32 v215, -v221, v217, 1.0
	v_fma_f32 v218, -v219, v216, v220
	v_mul_f32_e32 v10, v10, v134
	s_mov_b32 vcc_lo, s3
	v_div_fixup_f32 v2, v2, v132, v140
	v_div_scale_f32 v140, null, v132, v132, v136
	v_fmac_f32_e32 v217, v215, v217
	v_div_scale_f32 v215, s5, v137, v132, v137
	v_mul_f32_e32 v219, v141, v142
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v214, v218, v214, v216
	v_rcp_f32_e32 v220, v140
	v_div_scale_f32 v223, s3, v135, v132, v135
	v_mul_f32_e32 v218, v215, v217
	v_fma_f32 v216, -v213, v219, v141
	v_div_fixup_f32 v139, v214, v132, v139
	v_mul_f32_e32 v3, v3, v134
	s_mov_b32 vcc_lo, s4
	v_fma_f32 v214, -v221, v218, v215
	v_dual_fmac_f32 v219, v216, v142 :: v_dual_mul_f32 v216, v223, v212
	v_fma_f32 v224, -v140, v220, 1.0
	v_cvt_pk_fp8_f32 v209.h, v2, v139
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v218, v214, v217
	v_fma_f32 v141, -v213, v219, v141
	v_fma_f32 v213, -v222, v216, v223
	v_fmac_f32_e32 v220, v224, v220
	v_div_scale_f32 v224, s6, v136, v132, v136
	v_dual_mul_f32 v94, v94, v134 :: v_dual_mul_f32 v91, v91, v134
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v141, v141, v142, v219
	v_fma_f32 v142, -v221, v218, v215
	v_mul_f32_e32 v214, v224, v220
	v_fmac_f32_e32 v216, v213, v212
	s_mov_b32 vcc_lo, s5
	v_div_fixup_f32 v138, v141, v132, v138
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v142, v142, v217, v218
	v_fma_f32 v215, -v140, v214, v224
	v_fma_f32 v141, -v222, v216, v223
	s_mov_b32 vcc_lo, s3
	v_dual_mul_f32 v92, v92, v134 :: v_dual_mul_f32 v89, v89, v134
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v214, v215, v220
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v141, v141, v212, v216
	s_mov_b32 vcc_lo, s6
	v_div_fixup_f32 v137, v142, v132, v137
	v_dual_mul_f32 v90, v90, v134 :: v_dual_mul_f32 v87, v87, v134
	v_fma_f32 v140, -v140, v214, v224
	v_div_fixup_f32 v2, v141, v132, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_pk_fp8_f32 v210.l, v138, v137
	v_dual_mul_f32 v88, v88, v134 :: v_dual_mul_f32 v85, v85, v134
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v140, v140, v220, v214
	v_dual_mul_f32 v86, v86, v134 :: v_dual_mul_f32 v83, v83, v134
	v_dual_mul_f32 v84, v84, v134 :: v_dual_mul_f32 v81, v81, v134
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v236, v140, v132, v136
	ds_load_b128 v[135:138], v232 offset:16384
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[139:142], v232 offset:16896
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
	v_dual_mul_f32 v82, v82, v134 :: v_dual_mul_f32 v79, v79, v134
	v_dual_mul_f32 v80, v80, v134 :: v_dual_mul_f32 v77, v77, v134
	v_dual_mul_f32 v78, v78, v134 :: v_dual_mul_f32 v75, v75, v134
	v_dual_mul_f32 v76, v76, v134 :: v_dual_mul_f32 v73, v73, v134
	v_dual_mul_f32 v74, v74, v134 :: v_dual_mul_f32 v71, v71, v134
	v_dual_mul_f32 v72, v72, v134 :: v_dual_mul_f32 v69, v69, v134
	v_dual_mul_f32 v70, v70, v134 :: v_dual_mul_f32 v67, v67, v134
	v_dual_mul_f32 v68, v68, v134 :: v_dual_mul_f32 v65, v65, v134
	v_dual_mul_f32 v66, v66, v134 :: v_dual_mul_f32 v63, v63, v134
	v_dual_mul_f32 v64, v64, v134 :: v_dual_mul_f32 v61, v61, v134
	v_dual_mul_f32 v62, v62, v134 :: v_dual_mul_f32 v59, v59, v134
	v_dual_mul_f32 v60, v60, v134 :: v_dual_mul_f32 v57, v57, v134
	v_dual_mul_f32 v58, v58, v134 :: v_dual_mul_f32 v55, v55, v134
	v_dual_mul_f32 v56, v56, v134 :: v_dual_mul_f32 v53, v53, v134
	v_dual_mul_f32 v54, v54, v134 :: v_dual_mul_f32 v51, v51, v134
	v_dual_mul_f32 v52, v52, v134 :: v_dual_mul_f32 v49, v49, v134
	v_dual_mul_f32 v50, v50, v134 :: v_dual_mul_f32 v47, v47, v134
	v_dual_mul_f32 v48, v48, v134 :: v_dual_mul_f32 v45, v45, v134
	v_dual_mul_f32 v46, v46, v134 :: v_dual_mul_f32 v43, v43, v134
	v_dual_mul_f32 v44, v44, v134 :: v_dual_mul_f32 v41, v41, v134
	v_dual_mul_f32 v42, v42, v134 :: v_dual_mul_f32 v39, v39, v134
	v_dual_mul_f32 v40, v40, v134 :: v_dual_mul_f32 v37, v37, v134
	v_dual_mul_f32 v38, v38, v134 :: v_dual_mul_f32 v35, v35, v134
	v_dual_mul_f32 v36, v36, v134 :: v_dual_mul_f32 v33, v33, v134
	v_dual_mul_f32 v34, v34, v134 :: v_dual_mul_f32 v31, v31, v134
	v_dual_mul_f32 v32, v32, v134 :: v_dual_mul_f32 v29, v29, v134
	v_dual_mul_f32 v30, v30, v134 :: v_dual_mul_f32 v27, v27, v134
	v_dual_mul_f32 v28, v28, v134 :: v_dual_mul_f32 v25, v25, v134
	v_dual_mul_f32 v26, v26, v134 :: v_dual_mul_f32 v23, v23, v134
	v_dual_mul_f32 v24, v24, v134 :: v_dual_mul_f32 v21, v21, v134
	v_dual_mul_f32 v20, v20, v134 :: v_dual_mul_f32 v19, v19, v134
	v_dual_mul_f32 v18, v18, v134 :: v_dual_mul_f32 v17, v17, v134
	v_dual_mul_f32 v14, v14, v134 :: v_dual_mul_f32 v13, v13, v134
	v_dual_mul_f32 v12, v12, v134 :: v_dual_mul_f32 v7, v7, v134
	v_dual_mul_f32 v8, v8, v134 :: v_dual_mul_f32 v5, v5, v134
	v_cvt_pk_fp8_f32 v210.h, v2, v236
	v_mul_f32_e32 v6, v6, v134
	v_mul_f32_e32 v4, v4, v134
	v_mov_b32_e32 v2, v133
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[123:130], v[135:136], v[209:210], v[123:130]
	v_wmma_f32_16x16x16_fp8_fp8 v[115:122], v[137:138], v[209:210], v[115:122]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[107:114], v[139:140], v[209:210], v[107:114]
	v_wmma_f32_16x16x16_fp8_fp8 v[99:106], v[141:142], v[209:210], v[99:106]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[91:98], v[212:213], v[209:210], v[91:98]
	v_wmma_f32_16x16x16_fp8_fp8 v[83:90], v[214:215], v[209:210], v[83:90]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[75:82], v[216:217], v[209:210], v[75:82]
	v_wmma_f32_16x16x16_fp8_fp8 v[67:74], v[218:219], v[209:210], v[67:74]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[59:66], v[220:221], v[209:210], v[59:66]
	v_wmma_f32_16x16x16_fp8_fp8 v[51:58], v[222:223], v[209:210], v[51:58]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[43:50], v[224:225], v[209:210], v[43:50]
	v_wmma_f32_16x16x16_fp8_fp8 v[35:42], v[226:227], v[209:210], v[35:42]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[27:34], v[228:229], v[209:210], v[27:34]
	v_wmma_f32_16x16x16_fp8_fp8 v[19:26], v[230:231], v[209:210], v[19:26]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[11:18], v[232:233], v[209:210], v[11:18]
	v_wmma_f32_16x16x16_fp8_fp8 v[3:10], v[234:235], v[209:210], v[3:10]
	v_dual_mov_b32 v209, v132 :: v_dual_mov_b32 v210, v1
	s_add_co_i32 s21, s21, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s21, 4
	s_cbranch_scc1 .LBB7_12
.LBB7_49:                               ;   Parent Loop BB7_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB7_51 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, s21, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s4, s13
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s3, s22
	s_cbranch_scc1 .LBB7_56
; %bb.50:                               ;   in Loop: Header=BB7_49 Depth=2
	v_mov_b32_e32 v132, 0
	v_lshl_add_u32 v1, s21, 9, v197
	s_mov_b32 s5, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v133, v132 :: v_dual_mov_b32 v134, v132
	v_dual_mov_b32 v135, v132 :: v_dual_mov_b32 v136, v132
	v_dual_mov_b32 v137, v132 :: v_dual_mov_b32 v138, v132
	v_mov_b32_e32 v139, v132
.LBB7_51:                               ;   Parent Loop BB7_13 Depth=1
                                        ;     Parent Loop BB7_49 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s6, s5, 6
	v_lshl_add_u32 v142, s5, 12, v1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v140, vcc_lo, v198, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v141, null, 0, v199, vcc_lo
	ds_load_b128 v[212:215], v142
	ds_load_b128 v[216:219], v142 offset:2048
	s_add_co_i32 s5, s5, 1
	s_clause 0x3
	global_load_b64 v[220:221], v[140:141], off
	global_load_b64 v[222:223], v[140:141], off offset:16
	global_load_b64 v[224:225], v[140:141], off offset:32
	global_load_b64 v[140:141], v[140:141], off offset:48
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s5, 4
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[212:213], v[220:221], v[132:139]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[214:215], v[222:223], v[132:139]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[216:217], v[224:225], v[132:139]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[218:219], v[140:141], v[132:139]
	s_cbranch_scc1 .LBB7_51
; %bb.52:                               ;   in Loop: Header=BB7_49 Depth=2
	v_lshl_add_u32 v212, s4, 1, v157
	s_or_b32 s4, s3, 15
	v_or_b32_e32 v214, s3, v158
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s4, s22
	v_mov_b32_e32 v213, 0xff800000
	ds_load_b96 v[140:142], v212 offset:49154
	ds_load_u16_d16 v1, v212 offset:49166
	s_cselect_b32 s5, -1, 0
	s_cmp_le_i32 s4, s16
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s5, s4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s20, s3
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB7_54
; %bb.53:                               ;   in Loop: Header=BB7_49 Depth=2
	ds_load_u16_d16 v213, v212 offset:49152
	v_mul_f32_e32 v132, v148, v132
	v_cmp_le_i32_e32 vcc_lo, v214, v211
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s4, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v132, v213, v132, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v213, 0xff800000, v132, vcc_lo
.LBB7_54:                               ;   in Loop: Header=BB7_49 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mul_f32_e32 v132, v148, v133
	v_or_b32_e32 v133, 2, v214
	v_cmp_lt_i32_e32 vcc_lo, v214, v211
	v_dual_mul_f32 v134, v148, v134 :: v_dual_mul_f32 v137, v148, v137
	s_wait_dscnt 0x1
	v_fma_mix_f32 v132, v140, v132, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e64 s3, v133, v211
	v_or_b32_e32 v133, 3, v214
	s_or_b32 s5, s4, vcc_lo
	v_fma_mix_f32 v134, v140, v134, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s5
	s_or_b32 s3, s4, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v132, 0xff800000, v132, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v133, v211
	v_mul_f32_e32 v133, v148, v135
	v_or_b32_e32 v135, 4, v214
	s_and_b32 s3, s0, s3
	v_mul_f32_e32 v136, v148, v136
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v134, 0xff800000, v134, s3
	s_or_b32 s3, s4, vcc_lo
	v_fma_mix_f32 v133, v141, v133, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e32 vcc_lo, v135, v211
	v_or_b32_e32 v135, 5, v214
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s0, s3
	v_fma_mix_f32 v136, v141, v136, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v133, 0xff800000, v133, s3
	s_or_b32 s3, s4, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v135, v211
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s0, s3
	v_or_b32_e32 v140, 7, v214
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v135, 0xff800000, v136, s3
	v_fma_mix_f32 v136, v142, v137, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v137, 6, v214
	s_or_b32 s3, s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s3
	v_cmp_le_i32_e64 s3, v140, v211
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v136, 0xff800000, v136, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v137, v211
	v_dual_mul_f32 v137, v148, v138 :: v_dual_mul_f32 v138, v148, v139
	v_max3_num_f32 v139, v213, 0xff800000, v132
	s_or_b32 s3, s4, s3
	s_or_b32 s5, s4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_mix_f32 v137, v142, v137, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v1, v1, v138, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v138, v139, v134, v133
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s5
	v_add_nc_u32_e32 v140, 0xc080, v212
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v137, 0xff800000, v137, vcc_lo
	s_and_b32 vcc_lo, s0, s3
	s_mov_b32 s3, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v139, 0xff800000, v1, vcc_lo
	v_max3_num_f32 v1, v138, v135, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, v137, v139
	v_mov_b32_e32 v138, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v138, v138, s7, 0xfedcba98
	v_max3_num_f32 v1, v210, v1, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v138, v139, v1 :: v_dual_sub_f32 v133, v133, v1
	v_dual_sub_f32 v132, v132, v1 :: v_dual_sub_f32 v135, v135, v1
	v_sub_f32_e32 v136, v136, v1
	v_dual_mul_f32 v138, 0x3fb8aa3b, v138 :: v_dual_sub_f32 v137, v137, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v132, 0x3fb8aa3b, v132 :: v_dual_sub_f32 v139, v213, v1
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v1
	v_exp_f32_e32 v138, v138
	v_mul_f32_e32 v142, 0x3fb8aa3b, v133
	v_dual_mul_f32 v136, 0x3fb8aa3b, v136 :: v_dual_mul_f32 v137, 0x3fb8aa3b, v137
	v_exp_f32_e32 v141, v132
	ds_load_2addr_b32 v[132:133], v140 offset1:1
	v_add_nc_u32_e32 v140, 0xc088, v212
	v_mul_f32_e32 v212, 0x3fb8aa3b, v135
	v_mul_f32_e32 v139, 0x3fb8aa3b, v139
	v_exp_f32_e32 v213, v136
	v_exp_f32_e32 v142, v142
	ds_load_2addr_b32 v[135:136], v140 offset1:1
	v_exp_f32_e32 v140, v212
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v212, v138, 0, vcc_lo
	v_sub_f32_e32 v134, v134, v1
	v_exp_f32_e32 v139, v139
	v_exp_f32_e32 v137, v137
	v_cndmask_b32_e64 v213, v213, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v134, 0x3fb8aa3b, v134
	v_cndmask_b32_e64 v214, v142, 0, vcc_lo
	v_exp_f32_e32 v134, v134
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v138, v139, 0, vcc_lo
	v_cndmask_b32_e64 v139, v141, 0, vcc_lo
	v_cndmask_b32_e64 v215, v137, 0, vcc_lo
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v142, v132, v138, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v141, v132, v139, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v138, v138, v139
	s_delay_alu instid0(TRANS32_DEP_1)
	v_cndmask_b32_e64 v134, v134, 0, vcc_lo
	v_cndmask_b32_e64 v132, v140, 0, vcc_lo
	v_fma_mix_f32 v139, v133, v214, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v137, v135, v213, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v140, v133, v134, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v133, v142, 0, v141
	v_add_f32_e32 v134, v134, v138
	v_fma_mix_f32 v138, v135, v132, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v135, v136, v215, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v136, v136, v212, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_max3_num_f32 v133, v133, v140, v139
	v_add_f32_e32 v134, v214, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v133, v133, v138, v137
	v_add_f32_e32 v132, v132, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v214, v133, v135, v136
	v_add_f32_e32 v132, v213, v132
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b32_e32 v134, v214
	v_add_f32_e32 v132, v215, v132
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v134, v134, s7, 0xfedcba98
	v_dual_add_f32 v133, v212, v132 :: v_dual_max_num_f32 v132, v134, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b32_e32 v134, v133
	v_max_num_f32_e32 v212, v214, v132
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_permlanex16_b32 v134, v134, s7, 0xfedcba98
	v_mov_b32_e32 v132, v209
	v_cmpx_lt_f32_e32 0, v212
	s_cbranch_execz .LBB7_48
; %bb.55:                               ;   in Loop: Header=BB7_49 Depth=2
	v_div_scale_f32 v132, null, 0x43e00000, 0x43e00000, v212
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v213, v132
	v_fma_f32 v214, -v132, v213, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v213, v214, v213
	v_div_scale_f32 v214, vcc_lo, v212, 0x43e00000, v212
	v_mul_f32_e32 v215, v214, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v216, -v132, v215, v214
	v_fmac_f32_e32 v215, v216, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v132, -v132, v215, v214
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v132, v132, v213, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v132, v132, 0x43e00000, v212
	v_max_num_f32_e32 v132, 0x1f800000, v132
	s_branch .LBB7_48
.LBB7_56:                               ;   in Loop: Header=BB7_49 Depth=2
	v_mov_b32_e32 v1, v210
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v210, v1
	s_add_co_i32 s21, s21, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s21, 4
	s_cbranch_scc0 .LBB7_49
	s_branch .LBB7_12
.LBB7_57:
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v209, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_mov_b32_e32 v9, v2
	v_dual_mov_b32 v3, v2 :: v_dual_mov_b32 v4, v2
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v6, v2
	v_dual_mov_b32 v7, v2 :: v_dual_mov_b32 v8, v2
	v_dual_mov_b32 v1, 0xff800000 :: v_dual_mov_b32 v130, v9
	v_dual_mov_b32 v123, v2 :: v_dual_mov_b32 v122, v9
	v_dual_mov_b32 v115, v2 :: v_dual_mov_b32 v114, v9
	v_dual_mov_b32 v107, v2 :: v_dual_mov_b32 v106, v9
	v_dual_mov_b32 v99, v2 :: v_dual_mov_b32 v98, v9
	v_dual_mov_b32 v91, v2 :: v_dual_mov_b32 v90, v9
	v_dual_mov_b32 v83, v2 :: v_dual_mov_b32 v82, v9
	v_dual_mov_b32 v75, v2 :: v_dual_mov_b32 v74, v9
	v_dual_mov_b32 v67, v2 :: v_dual_mov_b32 v66, v9
	v_dual_mov_b32 v59, v2 :: v_dual_mov_b32 v58, v9
	v_dual_mov_b32 v51, v2 :: v_dual_mov_b32 v50, v9
	v_dual_mov_b32 v43, v2 :: v_dual_mov_b32 v42, v9
	v_dual_mov_b32 v35, v2 :: v_dual_mov_b32 v34, v9
	v_dual_mov_b32 v27, v2 :: v_dual_mov_b32 v26, v9
	v_dual_mov_b32 v19, v2 :: v_dual_mov_b32 v18, v9
	v_dual_mov_b32 v129, v8 :: v_dual_mov_b32 v128, v7
	v_dual_mov_b32 v127, v6 :: v_dual_mov_b32 v126, v5
	v_dual_mov_b32 v125, v4 :: v_dual_mov_b32 v124, v3
	v_dual_mov_b32 v121, v8 :: v_dual_mov_b32 v120, v7
	v_dual_mov_b32 v119, v6 :: v_dual_mov_b32 v118, v5
	v_dual_mov_b32 v117, v4 :: v_dual_mov_b32 v116, v3
	v_dual_mov_b32 v113, v8 :: v_dual_mov_b32 v112, v7
	v_dual_mov_b32 v111, v6 :: v_dual_mov_b32 v110, v5
	v_dual_mov_b32 v109, v4 :: v_dual_mov_b32 v108, v3
	v_dual_mov_b32 v105, v8 :: v_dual_mov_b32 v104, v7
	v_dual_mov_b32 v103, v6 :: v_dual_mov_b32 v102, v5
	v_dual_mov_b32 v101, v4 :: v_dual_mov_b32 v100, v3
	v_dual_mov_b32 v97, v8 :: v_dual_mov_b32 v96, v7
	v_dual_mov_b32 v95, v6 :: v_dual_mov_b32 v94, v5
	v_dual_mov_b32 v93, v4 :: v_dual_mov_b32 v92, v3
	v_dual_mov_b32 v89, v8 :: v_dual_mov_b32 v88, v7
	v_dual_mov_b32 v87, v6 :: v_dual_mov_b32 v86, v5
	v_dual_mov_b32 v85, v4 :: v_dual_mov_b32 v84, v3
	v_dual_mov_b32 v81, v8 :: v_dual_mov_b32 v80, v7
	v_dual_mov_b32 v79, v6 :: v_dual_mov_b32 v78, v5
	v_dual_mov_b32 v77, v4 :: v_dual_mov_b32 v76, v3
	v_dual_mov_b32 v73, v8 :: v_dual_mov_b32 v72, v7
	v_dual_mov_b32 v71, v6 :: v_dual_mov_b32 v70, v5
	v_dual_mov_b32 v69, v4 :: v_dual_mov_b32 v68, v3
	v_dual_mov_b32 v65, v8 :: v_dual_mov_b32 v64, v7
	v_dual_mov_b32 v63, v6 :: v_dual_mov_b32 v62, v5
	v_dual_mov_b32 v61, v4 :: v_dual_mov_b32 v60, v3
	v_dual_mov_b32 v57, v8 :: v_dual_mov_b32 v56, v7
	v_dual_mov_b32 v55, v6 :: v_dual_mov_b32 v54, v5
	v_dual_mov_b32 v53, v4 :: v_dual_mov_b32 v52, v3
	v_dual_mov_b32 v49, v8 :: v_dual_mov_b32 v48, v7
	v_dual_mov_b32 v47, v6 :: v_dual_mov_b32 v46, v5
	v_dual_mov_b32 v45, v4 :: v_dual_mov_b32 v44, v3
	v_dual_mov_b32 v41, v8 :: v_dual_mov_b32 v40, v7
	v_dual_mov_b32 v39, v6 :: v_dual_mov_b32 v38, v5
	v_dual_mov_b32 v37, v4 :: v_dual_mov_b32 v36, v3
	v_dual_mov_b32 v33, v8 :: v_dual_mov_b32 v32, v7
	v_dual_mov_b32 v31, v6 :: v_dual_mov_b32 v30, v5
	v_dual_mov_b32 v29, v4 :: v_dual_mov_b32 v28, v3
	v_dual_mov_b32 v25, v8 :: v_dual_mov_b32 v24, v7
	v_dual_mov_b32 v23, v6 :: v_dual_mov_b32 v22, v5
	v_dual_mov_b32 v21, v4 :: v_dual_mov_b32 v20, v3
	v_dual_mov_b32 v17, v8 :: v_dual_mov_b32 v16, v7
	v_dual_mov_b32 v15, v6 :: v_dual_mov_b32 v14, v5
	v_dual_mov_b32 v13, v4 :: v_dual_mov_b32 v12, v3
	v_dual_mov_b32 v11, v2 :: v_dual_mov_b32 v10, v9
	v_mov_b32_e32 v9, v8
	v_mov_b32_e32 v8, v7
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
.LBB7_58:
	v_cmp_eq_u32_e32 vcc_lo, 0, v149
	s_and_b32 s2, vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB7_60
; %bb.59:
	v_mad_co_u64_u32 v[131:132], null, s17, v143, s[12:13]
	v_mov_b32_e32 v132, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v131, 0x102, v131
	v_lshlrev_b64_e32 v[131:132], 2, v[131:132]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v131, vcc_lo, s10, v131
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v132, null, s11, v132, vcc_lo
	global_store_b64 v[131:132], v[1:2], off
.LBB7_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB7_62
; %bb.61:
	v_mad_co_u64_u32 v[1:2], null, s17, v143, s[12:13]
	v_mov_b32_e32 v2, 0
	v_dual_mul_f32 v0, v209, v123 :: v_dual_mul_f32 v115, v115, v209
	v_lshlrev_b32_e32 v123, 5, v149
	v_dual_mul_f32 v107, v107, v209 :: v_dual_mul_f32 v116, v116, v209
	v_dual_mul_f32 v99, v99, v209 :: v_dual_mul_f32 v108, v108, v209
	v_mul_lo_u32 v1, 0x102, v1
	v_dual_mul_f32 v91, v91, v209 :: v_dual_mul_f32 v100, v100, v209
	v_dual_mul_f32 v83, v83, v209 :: v_dual_mul_f32 v92, v92, v209
	v_dual_mul_f32 v75, v75, v209 :: v_dual_mul_f32 v84, v84, v209
	v_dual_mul_f32 v67, v67, v209 :: v_dual_mul_f32 v76, v76, v209
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_dual_mul_f32 v59, v59, v209 :: v_dual_mul_f32 v68, v68, v209
	v_dual_mul_f32 v51, v51, v209 :: v_dual_mul_f32 v60, v60, v209
	v_dual_mul_f32 v43, v43, v209 :: v_dual_mul_f32 v52, v52, v209
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, s10, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s11, v2, vcc_lo
	v_dual_mul_f32 v35, v35, v209 :: v_dual_mul_f32 v44, v44, v209
	v_add_co_u32 v131, vcc_lo, v1, v123
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v132, null, 0, v2, vcc_lo
	v_dual_mul_f32 v123, v3, v209 :: v_dual_mul_f32 v12, v12, v209
	v_dual_mul_f32 v1, v209, v124 :: v_dual_mul_f32 v124, v4, v209
	v_dual_mul_f32 v2, v209, v125 :: v_dual_mul_f32 v3, v209, v126
	v_dual_mul_f32 v27, v27, v209 :: v_dual_mul_f32 v36, v36, v209
	v_dual_mul_f32 v19, v19, v209 :: v_dual_mul_f32 v28, v28, v209
	v_dual_mul_f32 v11, v11, v209 :: v_dual_mul_f32 v20, v20, v209
	v_dual_mul_f32 v117, v117, v209 :: v_dual_mul_f32 v118, v118, v209
	v_dual_mul_f32 v109, v109, v209 :: v_dual_mul_f32 v110, v110, v209
	v_dual_mul_f32 v101, v101, v209 :: v_dual_mul_f32 v102, v102, v209
	v_dual_mul_f32 v93, v93, v209 :: v_dual_mul_f32 v94, v94, v209
	v_dual_mul_f32 v85, v85, v209 :: v_dual_mul_f32 v86, v86, v209
	v_dual_mul_f32 v77, v77, v209 :: v_dual_mul_f32 v78, v78, v209
	v_dual_mul_f32 v69, v69, v209 :: v_dual_mul_f32 v70, v70, v209
	v_dual_mul_f32 v61, v61, v209 :: v_dual_mul_f32 v62, v62, v209
	v_dual_mul_f32 v53, v53, v209 :: v_dual_mul_f32 v54, v54, v209
	v_dual_mul_f32 v45, v45, v209 :: v_dual_mul_f32 v46, v46, v209
	v_dual_mul_f32 v37, v37, v209 :: v_dual_mul_f32 v38, v38, v209
	v_dual_mul_f32 v29, v29, v209 :: v_dual_mul_f32 v30, v30, v209
	v_dual_mul_f32 v21, v21, v209 :: v_dual_mul_f32 v22, v22, v209
	v_dual_mul_f32 v127, v209, v127 :: v_dual_mul_f32 v128, v209, v128
	v_dual_mul_f32 v129, v209, v129 :: v_dual_mul_f32 v130, v209, v130
	v_dual_mul_f32 v125, v5, v209 :: v_dual_mul_f32 v126, v6, v209
	global_store_b128 v[131:132], v[0:3], off offset:8
	v_dual_mul_f32 v0, v119, v209 :: v_dual_mul_f32 v1, v120, v209
	v_dual_mul_f32 v4, v111, v209 :: v_dual_mul_f32 v3, v122, v209
	v_dual_mul_f32 v103, v103, v209 :: v_dual_mul_f32 v2, v121, v209
	v_dual_mul_f32 v95, v95, v209 :: v_dual_mul_f32 v104, v104, v209
	v_dual_mul_f32 v87, v87, v209 :: v_dual_mul_f32 v96, v96, v209
	v_dual_mul_f32 v79, v79, v209 :: v_dual_mul_f32 v88, v88, v209
	v_dual_mul_f32 v71, v71, v209 :: v_dual_mul_f32 v80, v80, v209
	v_dual_mul_f32 v63, v63, v209 :: v_dual_mul_f32 v72, v72, v209
	v_dual_mul_f32 v55, v55, v209 :: v_dual_mul_f32 v64, v64, v209
	v_dual_mul_f32 v47, v47, v209 :: v_dual_mul_f32 v56, v56, v209
	v_dual_mul_f32 v39, v39, v209 :: v_dual_mul_f32 v48, v48, v209
	v_dual_mul_f32 v31, v31, v209 :: v_dual_mul_f32 v40, v40, v209
	v_dual_mul_f32 v23, v23, v209 :: v_dual_mul_f32 v32, v32, v209
	v_dual_mul_f32 v15, v15, v209 :: v_dual_mul_f32 v24, v24, v209
	v_dual_mul_f32 v111, v7, v209 :: v_dual_mul_f32 v16, v16, v209
	v_dual_mul_f32 v5, v112, v209 :: v_dual_mul_f32 v6, v113, v209
	v_dual_mul_f32 v112, v8, v209 :: v_dual_mul_f32 v105, v105, v209
	v_dual_mul_f32 v97, v97, v209 :: v_dual_mul_f32 v106, v106, v209
	v_dual_mul_f32 v89, v89, v209 :: v_dual_mul_f32 v98, v98, v209
	v_dual_mul_f32 v81, v81, v209 :: v_dual_mul_f32 v90, v90, v209
	v_dual_mul_f32 v73, v73, v209 :: v_dual_mul_f32 v82, v82, v209
	v_dual_mul_f32 v65, v65, v209 :: v_dual_mul_f32 v74, v74, v209
	v_dual_mul_f32 v57, v57, v209 :: v_dual_mul_f32 v66, v66, v209
	v_dual_mul_f32 v49, v49, v209 :: v_dual_mul_f32 v58, v58, v209
	v_dual_mul_f32 v41, v41, v209 :: v_dual_mul_f32 v50, v50, v209
	v_dual_mul_f32 v33, v33, v209 :: v_dual_mul_f32 v42, v42, v209
	v_dual_mul_f32 v25, v25, v209 :: v_dual_mul_f32 v34, v34, v209
	v_dual_mul_f32 v17, v17, v209 :: v_dual_mul_f32 v26, v26, v209
	v_mul_f32_e32 v7, v114, v209
	v_dual_mul_f32 v13, v13, v209 :: v_dual_mul_f32 v14, v14, v209
	v_dual_mul_f32 v113, v9, v209 :: v_dual_mul_f32 v18, v18, v209
	s_clause 0x18
	global_store_b128 v[131:132], v[127:130], off offset:24
	global_store_b128 v[131:132], v[115:118], off offset:72
	global_store_b128 v[131:132], v[0:3], off offset:88
	global_store_b128 v[131:132], v[107:110], off offset:136
	global_store_b128 v[131:132], v[4:7], off offset:152
	global_store_b128 v[131:132], v[99:102], off offset:200
	global_store_b128 v[131:132], v[103:106], off offset:216
	global_store_b128 v[131:132], v[91:94], off offset:264
	global_store_b128 v[131:132], v[95:98], off offset:280
	global_store_b128 v[131:132], v[83:86], off offset:328
	global_store_b128 v[131:132], v[87:90], off offset:344
	global_store_b128 v[131:132], v[75:78], off offset:392
	global_store_b128 v[131:132], v[79:82], off offset:408
	global_store_b128 v[131:132], v[67:70], off offset:456
	global_store_b128 v[131:132], v[71:74], off offset:472
	global_store_b128 v[131:132], v[59:62], off offset:520
	global_store_b128 v[131:132], v[63:66], off offset:536
	global_store_b128 v[131:132], v[51:54], off offset:584
	global_store_b128 v[131:132], v[55:58], off offset:600
	global_store_b128 v[131:132], v[43:46], off offset:648
	global_store_b128 v[131:132], v[47:50], off offset:664
	global_store_b128 v[131:132], v[35:38], off offset:712
	global_store_b128 v[131:132], v[39:42], off offset:728
	global_store_b128 v[131:132], v[27:30], off offset:776
	global_store_b128 v[131:132], v[31:34], off offset:792
	v_mul_f32_e32 v114, v10, v209
	s_clause 0x5
	global_store_b128 v[131:132], v[19:22], off offset:840
	global_store_b128 v[131:132], v[23:26], off offset:856
	global_store_b128 v[131:132], v[11:14], off offset:904
	global_store_b128 v[131:132], v[15:18], off offset:920
	global_store_b128 v[131:132], v[123:126], off offset:968
	global_store_b128 v[131:132], v[111:114], off offset:984
.LBB7_62:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end7:
	.size	attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201, .Lfunc_end7-attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
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
		.amdhsa_next_free_vgpr 237
		.amdhsa_next_free_sgpr 26
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end7-attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_vgpr, 237
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.numbered_sgpr, 26
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10552
; TotalNumSgprs: 28
; NumVgprs: 237
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 28
; NumVGPRsForWavesPerEU: 237
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
	.protected	attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201
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
	s_cbranch_vccnz .LBB8_81
; %bb.1:
	s_and_b32 s23, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s23, 3
	s_cbranch_scc1 .LBB8_81
; %bb.2:
	s_mul_i32 s5, ttmp9, 0x180
	s_mul_i32 s4, s7, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s5, s4
	s_cbranch_scc1 .LBB8_81
; %bb.3:
	s_lshr_b32 s18, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s18, s17
	s_cbranch_scc1 .LBB8_81
; %bb.4:
	v_dual_mov_b32 v150, s16 :: v_dual_and_b32 v149, 15, v0
	v_lshrrev_b32_e32 v4, 5, v0
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x20
	s_mul_i32 s1, s23, 6
	v_or_b32_e32 v1, s5, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v2, v4, 4, v1
	v_mul_hi_i32 v1, 0x2aaaaaab, v2
	v_cmp_gt_i32_e64 s0, s4, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v3, 31, v1
	v_add_nc_u32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v3, v1, 6
	v_mul_lo_u32 v5, v1, 24
	v_sub_nc_u32_e32 v3, v2, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v141, v3, s1, v5
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB8_6
; %bb.5:
	v_mov_b32_e32 v142, 0
	s_mul_i32 s20, s7, 0x1800
	s_mov_b32 s21, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[20:21], s[8:9], s[20:21]
	v_lshlrev_b64_e32 v[2:3], 2, v[141:142]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s20, v2
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	global_load_b32 v2, v[2:3], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v150, s16, v2
.LBB8_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_dual_mov_b32 v6, -1 :: v_dual_and_b32 v151, 31, v0
	v_bfrev_b32_e32 v5, -2
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v2, ttmp9, 6, v151
	v_ashrrev_i32_e32 v3, 31, v2
	v_cmpx_gt_i32_e64 s7, v2
	s_cbranch_execz .LBB8_8
; %bb.7:
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[2:3]
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s2, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
	global_load_b32 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v6, v5
.LBB8_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, 32, v2
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s7, v7
	s_cbranch_execz .LBB8_10
; %bb.9:
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s2, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v3, vcc_lo
	global_load_b32 v2, v[2:3], off offset:128
	s_wait_loadcnt 0x0
	v_max_i32_e32 v6, v6, v2
	v_min_i32_e32 v5, v5, v2
.LBB8_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v3, 16, v2
	v_xor_b32_e32 v8, 8, v2
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v2, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v8, v2, v8 :: v_dual_lshlrev_b32 v3, 2, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 2, v8
	ds_bpermute_b32 v7, v3, v6
	ds_bpermute_b32 v3, v3, v5
	s_wait_dscnt 0x1
	v_max_i32_e32 v6, v6, v7
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v5, v3
	ds_bpermute_b32 v5, v8, v6
	ds_bpermute_b32 v7, v8, v3
	v_xor_b32_e32 v8, 4, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v2, v8, vcc_lo
	v_lshlrev_b32_e32 v8, 2, v8
	s_wait_dscnt 0x1
	v_max_i32_e32 v5, v6, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v7
	ds_bpermute_b32 v6, v8, v5
	ds_bpermute_b32 v7, v8, v3
	v_xor_b32_e32 v8, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v2, v8, vcc_lo
	v_lshlrev_b32_e32 v153, 2, v8
	v_xor_b32_e32 v8, 1, v2
	s_wait_dscnt 0x1
	v_max_i32_e32 v5, v5, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v7
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v2, v8, vcc_lo
	ds_bpermute_b32 v6, v153, v5
	ds_bpermute_b32 v7, v153, v3
	v_lshlrev_b32_e32 v154, 2, v2
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v5, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v7
	ds_bpermute_b32 v5, v154, v2
	ds_bpermute_b32 v6, v154, v3
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v2, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s1, v2
	v_readfirstlane_b32 s16, v3
	v_cmpx_eq_u32_e32 0, v4
	s_cbranch_execz .LBB8_12
; %bb.11:
	s_mov_b32 m0, 0x80001
	s_barrier_init m0 
	s_mov_b32 m0, 0x80002
	s_barrier_init m0 
	s_mov_b32 m0, 0x80003
	s_barrier_init m0 
.LBB8_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_cvt_f32_u32 s6, s17
	s_add_co_i32 s7, s17, 0x1ff
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s7, s7, 0xffff
	v_s_rcp_f32 s19, s6
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s7, s7
	s_barrier_signal -1
	v_lshrrev_b32_e32 v152, 4, v151
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s19, s7, s19
	s_trunc_f32 s19, s19
	s_delay_alu instid0(SALU_CYCLE_3)
	s_xor_b32 s20, s19, 0x80000000
	s_cvt_u32_f32 s19, s19
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s7, s20, s6
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s7, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s7, s6
	s_add_co_ci_u32 s6, s19, 0
	s_add_co_i32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s6, s6, 0xffff
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s7, s18, s6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s7, s6
	s_lshl_b32 s19, s7, 6
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s6, s6, 6
	s_mov_b32 s7, 0
	s_wait_alu depctr_sa_sdst(0)
	s_min_i32 s20, s1, s6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s19, s20
	s_cbranch_scc1 .LBB8_76
; %bb.13:
	v_lshrrev_b32_e32 v142, 8, v0
	v_dual_mov_b32 v156, 0 :: v_dual_and_b32 v3, 7, v4
	v_ashrrev_i32_e32 v2, 31, v1
	s_addk_co_i32 s5, 0x17f
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v7, 0x6020400 :: v_dual_lshlrev_b32 v4, 12, v142
	v_mad_u32_u24 v6, v3, 3, v142
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s5, s4
	v_lshrrev_b32_e32 v8, 2, v149
	s_cselect_b32 s21, -1, 0
	v_lshl_add_u32 v10, v6, 9, 0
	v_and_b32_e32 v6, 1, v0
	s_wait_kmcnt 0x0
	v_add_co_u32 v143, vcc_lo, s2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v144, null, s3, v2, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	v_lshlrev_b32_e32 v155, 3, v152
	s_lshl_b32 s6, s23, 8
	v_lshlrev_b32_e32 v5, 11, v3
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[10:11], s[6:7]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v158, 0x3070105, v7, vcc_lo
	v_lshl_or_b32 v2, v3, 5, v155
	v_mov_b32_e32 v9, 0x5040100
	v_add3_u32 v4, 0, v4, v5
	v_cmp_gt_u32_e64 s2, 2, v3
	v_lshl_add_u32 v12, v142, 5, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v145, s4, s4, v2
	v_dual_mov_b32 v207, 0xff800000 :: v_dual_and_b32 v2, 3, v0
	v_mov_b32_e32 v206, 1.0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v146, null, s5, 0, s4
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_or_b32 v7, v2, 6, v8
	v_lshlrev_b32_e32 v11, 3, v2
	v_cmp_gt_u32_e32 vcc_lo, 2, v2
	v_lshlrev_b32_e32 v2, 5, v2
	s_add_nc_u64 s[4:5], s[12:13], s[6:7]
	v_lshlrev_b32_e32 v8, 2, v7
	v_cmp_ne_u32_e64 s1, 0, v142
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v159, 0x3020706, v9, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v3
	v_or_b32_e32 v9, 0x10c, v7
	v_add3_u32 v175, v4, v8, v2
	v_or_b32_e32 v2, 8, v7
	v_or_b32_e32 v8, 12, v7
	v_cmp_ne_u32_e64 s3, 1, v142
	v_xor_b32_e32 v9, v9, v11
	s_lshl_b32 s6, s23, 1
	v_xor_b32_e32 v2, v2, v11
	v_xor_b32_e32 v8, v8, v11
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[10:11], s[10:11], s[6:7]
	v_lshl_add_u32 v180, v9, 2, v4
	v_or_b32_e32 v9, 24, v7
	v_lshl_add_u32 v177, v2, 2, v4
	v_or_b32_e32 v2, 0x110, v7
	v_lshl_add_u32 v179, v8, 2, v4
	v_or_b32_e32 v8, 0x114, v7
	v_xor_b32_e32 v9, v9, v11
	s_add_nc_u64 s[12:13], s[12:13], s[6:7]
	v_xor_b32_e32 v2, v2, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v8, v8, v11
	v_lshl_add_u32 v185, v9, 2, v4
	v_or_b32_e32 v9, 0x128, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v182, v2, 2, v4
	v_or_b32_e32 v2, 28, v7
	v_lshl_add_u32 v184, v8, 2, v4
	v_or_b32_e32 v8, 40, v7
	v_xor_b32_e32 v9, v9, v11
	v_xor_b32_e32 v2, v2, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v8, v8, v11
	v_lshl_add_u32 v190, v9, 2, v4
	v_or_b32_e32 v9, 52, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v187, v2, 2, v4
	v_or_b32_e32 v2, 0x12c, v7
	v_lshl_add_u32 v189, v8, 2, v4
	v_or_b32_e32 v8, 0x130, v7
	v_xor_b32_e32 v9, v9, v11
	v_xor_b32_e32 v2, v2, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v8, v8, v11
	v_lshl_add_u32 v195, v9, 2, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v192, v2, 2, v4
	v_or_b32_e32 v2, 56, v7
	v_lshl_add_u32 v194, v8, 2, v4
	v_or_b32_e32 v8, 60, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v2, v2, v11
	v_xor_b32_e32 v8, v8, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v197, v2, 2, v4
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v5, 3, v151
	v_lshlrev_b32_e32 v6, 8, v141
	v_lshlrev_b32_e32 v157, 3, v3
	v_lshl_add_u32 v199, v8, 2, v4
	v_mov_b32_e32 v8, v2
	v_add_co_u32 v147, s4, s4, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v148, null, s5, 0, s4
	v_cmp_gt_u32_e64 s5, 16, v151
	v_cmp_eq_u32_e64 s4, v152, v3
	v_lshl_add_u32 v3, v142, 4, v151
	v_cndmask_b32_e64 v6, 0, v6, s0
	v_mov_b32_e32 v9, v2
	s_and_b32 s22, vcc_lo, s5
	v_lshlrev_b32_e32 v1, 4, v151
	v_lshl_add_u32 v160, v3, 1, 0
	v_add_co_u32 v3, s5, s8, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s9, 0, s5
	s_mov_b32 s9, 0x76543210
	v_add_co_u32 v201, vcc_lo, v3, v155
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v202, null, 0, v6, vcc_lo
	v_mov_b32_e32 v6, v2
	v_add_nc_u32_e32 v162, v4, v5
	v_xad_u32 v163, 0x120, v5, v4
	v_xad_u32 v164, 0x124, v5, v4
	v_xad_u32 v165, 0x240, v5, v4
	v_xad_u32 v166, 0x244, v5, v4
	v_xad_u32 v167, 0x360, v5, v4
	v_xad_u32 v168, 0x364, v5, v4
	v_xad_u32 v169, 0x520, v5, v4
	v_xad_u32 v170, 0x524, v5, v4
	v_xad_u32 v171, 0x640, v5, v4
	v_xad_u32 v172, 0x644, v5, v4
	v_xad_u32 v173, 0x760, v5, v4
	v_xad_u32 v174, 0x764, v5, v4
	v_or_b32_e32 v5, 0x108, v7
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v5, v5, v11
	v_lshl_add_u32 v178, v5, 2, v4
	v_or_b32_e32 v5, 20, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v5, v5, v11
	v_lshl_add_u32 v183, v5, 2, v4
	v_or_b32_e32 v5, 0x11c, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v5, v5, v11
	v_lshl_add_u32 v188, v5, 2, v4
	v_or_b32_e32 v5, 48, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v5, v5, v11
	v_lshl_add_u32 v193, v5, 2, v4
	v_or_b32_e32 v5, 0x138, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v5, v5, v11
	v_lshl_add_u32 v198, v5, 2, v4
	v_mov_b32_e32 v5, v2
	v_and_b32_e32 v13, 16, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v204, v12, v13
	v_lshlrev_b32_e32 v0, 4, v0
	v_and_b32_e32 v176, 0x31f0, v0
	v_or_b32_e32 v0, 16, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v0, v0, v11
	v_lshl_add_u32 v181, v0, 2, v4
	v_or_b32_e32 v0, 0x118, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v0, v0, v11
	v_lshl_add_u32 v186, v0, 2, v4
	v_or_b32_e32 v0, 44, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v0, v0, v11
	v_lshl_add_u32 v191, v0, 2, v4
	v_or_b32_e32 v0, 0x134, v7
	v_or_b32_e32 v7, 0x13c, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v0, v0, v11
	v_xor_b32_e32 v7, v7, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v196, v0, 2, v4
	v_lshl_add_u32 v200, v7, 2, v4
	v_dual_mov_b32 v4, v2 :: v_dual_mov_b32 v7, v2
	v_add_nc_u32_e32 v203, v10, v1
	v_mov_b32_e32 v17, v9
	v_dual_mov_b32 v25, v9 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v33, v9 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v41, v9 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v49, v9 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v57, v9 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v65, v9 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v73, v9 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v81, v9 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v89, v9 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v97, v9 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v105, v9 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v113, v9 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v121, v9 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v129, v9 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v16, v8 :: v_dual_add_nc_u32 v161, 0, v1
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v10, v2
	v_dual_mov_b32 v12, v4 :: v_dual_mov_b32 v11, v3
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v20, v4 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v19, v3
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v28, v4 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v26, v2 :: v_dual_mov_b32 v27, v3
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v36, v4 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v44, v4 :: v_dual_mov_b32 v45, v5
	v_dual_mov_b32 v42, v2 :: v_dual_mov_b32 v43, v3
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v55, v7
	v_dual_mov_b32 v52, v4 :: v_dual_mov_b32 v53, v5
	v_dual_mov_b32 v50, v2 :: v_dual_mov_b32 v51, v3
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v63, v7
	v_dual_mov_b32 v60, v4 :: v_dual_mov_b32 v61, v5
	v_dual_mov_b32 v58, v2 :: v_dual_mov_b32 v59, v3
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v71, v7
	v_dual_mov_b32 v68, v4 :: v_dual_mov_b32 v69, v5
	v_dual_mov_b32 v66, v2 :: v_dual_mov_b32 v67, v3
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v79, v7
	v_dual_mov_b32 v76, v4 :: v_dual_mov_b32 v77, v5
	v_dual_mov_b32 v74, v2 :: v_dual_mov_b32 v75, v3
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v87, v7
	v_dual_mov_b32 v84, v4 :: v_dual_mov_b32 v85, v5
	v_dual_mov_b32 v82, v2 :: v_dual_mov_b32 v83, v3
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v95, v7
	v_dual_mov_b32 v92, v4 :: v_dual_mov_b32 v93, v5
	v_dual_mov_b32 v90, v2 :: v_dual_mov_b32 v91, v3
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v103, v7
	v_dual_mov_b32 v100, v4 :: v_dual_mov_b32 v101, v5
	v_dual_mov_b32 v98, v2 :: v_dual_mov_b32 v99, v3
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v111, v7
	v_dual_mov_b32 v108, v4 :: v_dual_mov_b32 v109, v5
	v_dual_mov_b32 v106, v2 :: v_dual_mov_b32 v107, v3
	v_dual_mov_b32 v120, v8 :: v_dual_mov_b32 v119, v7
	v_dual_mov_b32 v116, v4 :: v_dual_mov_b32 v117, v5
	v_dual_mov_b32 v114, v2 :: v_dual_mov_b32 v115, v3
	v_dual_mov_b32 v128, v8 :: v_dual_mov_b32 v127, v7
	v_dual_mov_b32 v124, v4 :: v_dual_mov_b32 v125, v5
	v_dual_mov_b32 v122, v2 :: v_dual_mov_b32 v123, v3
	v_mov_b32_e32 v1, v2
	s_branch .LBB8_15
.LBB8_14:                               ;   in Loop: Header=BB8_15 Depth=1
	v_mov_b32_e32 v207, v0
	s_add_co_i32 s19, s19, 48
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s19, s20
	s_cbranch_scc1 .LBB8_77
.LBB8_15:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB8_19 Depth 2
                                        ;       Child Loop BB8_58 Depth 3
	s_add_co_i32 s5, s19, 47
	v_mov_b32_e32 v205, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s5, s20
	s_cselect_b32 s6, -1, 0
	s_cmp_le_i32 s5, s16
	s_cselect_b32 s5, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s6, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s23, s21, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s5, s23, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s6, s0, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s6
	s_cbranch_execz .LBB8_17
; %bb.16:                               ;   in Loop: Header=BB8_15 Depth=1
	global_load_b32 v205, v[143:144], off
.LBB8_17:                               ;   in Loop: Header=BB8_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_mov_b32 s24, 0
	s_branch .LBB8_19
.LBB8_18:                               ;   in Loop: Header=BB8_19 Depth=2
	s_or_b32 exec_lo, exec_lo, s5
	v_sub_f32_e32 v208, v207, v0
	v_add_f32_e32 v131, v131, v132
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v207
	v_div_scale_f32 v213, null, v130, v130, v138
	v_div_scale_f32 v215, null, v130, v130, v137
	v_mul_f32_e32 v208, 0x3fb8aa3b, v208
	v_mov_b16_e64 v231.l, v156.l
	v_mov_b16_e64 v231.h, 0
	v_rcp_f32_e32 v214, v213
	v_add_nc_u32_e32 v226, 0, v176
	v_exp_f32_e32 v208, v208
	v_mov_b16_e64 v230.l, v231.l
	v_mov_b16_e64 v230.h, v231.h
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v216, -v213, v214, 1.0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v207, 0, v208 :: v_dual_fmac_f32 v214, v216, v214
	v_div_scale_f32 v216, null, v130, v130, v133
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v206, v206, v207
	v_div_scale_f32 v208, null, v130, v130, v206
	v_div_scale_f32 v211, vcc_lo, v206, v130, v206
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v209, v208
	v_fma_f32 v210, -v208, v209, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v209, v210, v209
	v_mul_f32_e32 v210, v211, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v212, -v208, v210, v211
	v_fmac_f32_e32 v210, v212, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v208, -v208, v210, v211
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v208, v208, v209, v210
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v132, v208, v130, v206
	v_dual_mul_f32 v128, v128, v132 :: v_dual_fmac_f32 v131, v1, v207
	v_div_scale_f32 v1, null, v130, v130, v140
	v_div_scale_f32 v207, null, v130, v130, v139
	v_mul_f32_e32 v19, v19, v132
	v_mul_f32_e32 v7, v7, v132
	v_mul_f32_e32 v3, v3, v132
	v_rcp_f32_e32 v206, v1
	v_rcp_f32_e32 v209, v207
	v_mul_f32_e32 v129, v129, v132
	v_dual_mul_f32 v127, v127, v132 :: v_dual_mul_f32 v126, v126, v132
	v_dual_mul_f32 v125, v125, v132 :: v_dual_mul_f32 v124, v124, v132
	v_dual_mul_f32 v123, v123, v132 :: v_dual_mul_f32 v122, v122, v132
	v_mul_f32_e32 v121, v121, v132
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v208, -v1, v206, 1.0
	v_fma_f32 v211, -v207, v209, 1.0
	v_dual_mul_f32 v120, v120, v132 :: v_dual_mul_f32 v119, v119, v132
	v_dual_mul_f32 v118, v118, v132 :: v_dual_mul_f32 v117, v117, v132
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v206, v208, v206
	v_div_scale_f32 v208, vcc_lo, v140, v130, v140
	v_fmac_f32_e32 v209, v211, v209
	v_div_scale_f32 v211, s5, v139, v130, v139
	v_dual_mul_f32 v116, v116, v132 :: v_dual_mul_f32 v115, v115, v132
	v_mul_f32_e32 v210, v208, v206
	v_dual_mul_f32 v114, v114, v132 :: v_dual_mul_f32 v113, v113, v132
	v_dual_mul_f32 v112, v112, v132 :: v_dual_mul_f32 v111, v111, v132
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v212, -v1, v210, v208
	v_dual_mul_f32 v110, v110, v132 :: v_dual_mul_f32 v109, v109, v132
	v_dual_mul_f32 v108, v108, v132 :: v_dual_mul_f32 v107, v107, v132
	v_fmac_f32_e32 v210, v212, v206
	v_mul_f32_e32 v212, v211, v209
	v_dual_mul_f32 v106, v106, v132 :: v_dual_mul_f32 v105, v105, v132
	v_dual_mul_f32 v104, v104, v132 :: v_dual_mul_f32 v103, v103, v132
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v1, -v1, v210, v208
	v_fma_f32 v208, -v207, v212, v211
	v_dual_mul_f32 v102, v102, v132 :: v_dual_mul_f32 v101, v101, v132
	v_dual_mul_f32 v100, v100, v132 :: v_dual_mul_f32 v99, v99, v132
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v1, v1, v206, v210
	v_rcp_f32_e32 v206, v215
	s_mov_b32 vcc_lo, s5
	v_dual_mul_f32 v98, v98, v132 :: v_dual_mul_f32 v97, v97, v132
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_div_fixup_f32 v1, v1, v130, v140
	v_dual_mul_f32 v96, v96, v132 :: v_dual_mul_f32 v95, v95, v132
	v_dual_mul_f32 v94, v94, v132 :: v_dual_mul_f32 v93, v93, v132
	v_fma_f32 v210, -v215, v206, 1.0
	v_dual_mul_f32 v92, v92, v132 :: v_dual_mul_f32 v91, v91, v132
	v_dual_mul_f32 v90, v90, v132 :: v_dual_mul_f32 v89, v89, v132
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v206, v210, v206
	v_div_scale_f32 v210, s5, v137, v130, v137
	v_fmac_f32_e32 v212, v208, v209
	v_div_scale_f32 v208, s6, v138, v130, v138
	v_dual_mul_f32 v88, v88, v132 :: v_dual_mul_f32 v87, v87, v132
	v_dual_mul_f32 v86, v86, v132 :: v_dual_mul_f32 v85, v85, v132
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v140, -v207, v212, v211
	v_mul_f32_e32 v207, v208, v214
	v_dual_mul_f32 v84, v84, v132 :: v_dual_mul_f32 v83, v83, v132
	v_dual_mul_f32 v82, v82, v132 :: v_dual_mul_f32 v81, v81, v132
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fma_f32 v211, -v213, v207, v208
	v_mul_f32_e32 v10, v10, v132
	v_dual_mul_f32 v80, v80, v132 :: v_dual_mul_f32 v79, v79, v132
	v_dual_mul_f32 v78, v78, v132 :: v_dual_mul_f32 v77, v77, v132
	v_fmac_f32_e32 v207, v211, v214
	v_mul_f32_e32 v211, v210, v206
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v140, v140, v209, v212
	v_div_scale_f32 v209, null, v130, v130, v136
	v_div_scale_f32 v212, null, v130, v130, v135
	s_mov_b32 vcc_lo, s6
	v_dual_mul_f32 v76, v76, v132 :: v_dual_mul_f32 v75, v75, v132
	v_div_fixup_f32 v139, v140, v130, v139
	v_rcp_f32_e32 v140, v209
	v_dual_mul_f32 v74, v74, v132 :: v_dual_mul_f32 v73, v73, v132
	v_dual_mul_f32 v72, v72, v132 :: v_dual_mul_f32 v71, v71, v132
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	v_cvt_pk_fp8_f32 v230.l, v1, v139
	v_fma_f32 v1, -v213, v207, v208
	v_rcp_f32_e32 v213, v212
	v_fma_f32 v139, -v215, v211, v210
	v_fma_f32 v208, -v209, v140, 1.0
	v_dual_mul_f32 v70, v70, v132 :: v_dual_mul_f32 v69, v69, v132
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v1, v1, v214, v207
	s_mov_b32 vcc_lo, s5
	v_fmac_f32_e32 v140, v208, v140
	v_rcp_f32_e32 v207, v216
	v_fma_f32 v208, -v212, v213, 1.0
	v_div_fixup_f32 v1, v1, v130, v138
	v_div_scale_f32 v138, null, v130, v130, v134
	v_dual_mul_f32 v68, v68, v132 :: v_dual_mul_f32 v67, v67, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v213, v208, v213
	v_dual_mul_f32 v8, v8, v132 :: v_dual_fmac_f32 v211, v139, v206
	v_div_scale_f32 v208, s7, v135, v130, v135
	v_mul_f32_e32 v6, v6, v132
	v_div_scale_f32 v139, s6, v136, v130, v136
	v_fma_f32 v217, -v216, v207, 1.0
	v_fma_f32 v210, -v215, v211, v210
	v_rcp_f32_e32 v215, v138
	v_dual_mul_f32 v66, v66, v132 :: v_dual_mul_f32 v65, v65, v132
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v207, v217, v207
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v206, v210, v206, v211
	v_mul_f32_e32 v211, v208, v213
	v_div_scale_f32 v217, s5, v133, v130, v133
	s_mov_b32 vcc_lo, s6
	v_dual_mul_f32 v64, v64, v132 :: v_dual_mul_f32 v63, v63, v132
	v_fma_f32 v218, -v138, v215, 1.0
	v_div_fixup_f32 v137, v206, v130, v137
	v_fma_f32 v206, -v212, v211, v208
	v_dual_mul_f32 v5, v5, v132 :: v_dual_mul_f32 v62, v62, v132
	v_mul_f32_e32 v61, v61, v132
	v_dual_fmac_f32 v215, v218, v215 :: v_dual_mul_f32 v214, v139, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v211, v206, v213
	v_div_scale_f32 v218, s8, v134, v130, v134
	v_cvt_pk_fp8_f32 v230.h, v1, v137
	v_dual_mul_f32 v60, v60, v132 :: v_dual_mul_f32 v59, v59, v132
	v_fma_f32 v210, -v209, v214, v139
	v_dual_mul_f32 v58, v58, v132 :: v_dual_mul_f32 v57, v57, v132
	v_dual_mul_f32 v56, v56, v132 :: v_dual_mul_f32 v55, v55, v132
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v214, v210, v140
	v_mul_f32_e32 v210, v217, v207
	v_dual_mul_f32 v54, v54, v132 :: v_dual_mul_f32 v53, v53, v132
	v_dual_mul_f32 v52, v52, v132 :: v_dual_mul_f32 v51, v51, v132
	v_fma_f32 v139, -v209, v214, v139
	v_mul_f32_e32 v209, v218, v215
	v_fma_f32 v206, -v216, v210, v217
	v_dual_mul_f32 v50, v50, v132 :: v_dual_mul_f32 v49, v49, v132
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v139, v139, v140, v214
	v_fma_f32 v140, -v212, v211, v208
	v_fma_f32 v208, -v138, v209, v218
	v_fmac_f32_e32 v210, v206, v207
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v136, v139, v130, v136
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v140, v140, v213, v211
	v_fmac_f32_e32 v209, v208, v215
	v_fma_f32 v139, -v216, v210, v217
	s_mov_b32 vcc_lo, s5
	v_dual_mul_f32 v48, v48, v132 :: v_dual_mul_f32 v47, v47, v132
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v138, -v138, v209, v218
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v139, v139, v207, v210
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v135, v140, v130, v135
	v_dual_mul_f32 v46, v46, v132 :: v_dual_mul_f32 v45, v45, v132
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v138, v138, v215, v209
	v_div_fixup_f32 v1, v139, v130, v133
	v_cvt_pk_fp8_f32 v231.l, v136, v135
	v_dual_mul_f32 v44, v44, v132 :: v_dual_mul_f32 v43, v43, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v232, v138, v130, v134
	ds_load_b128 v[133:136], v226 offset:12288
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[137:140], v226 offset:12800
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[206:209], v226 offset:13312
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[210:213], v226 offset:13824
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[214:217], v226 offset:14336
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[218:221], v226 offset:14848
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[222:225], v226 offset:15360
	;;#ASMSTART
	;;#ASMEND
	ds_load_b128 v[226:229], v226 offset:15872
	v_dual_mul_f32 v42, v42, v132 :: v_dual_mul_f32 v41, v41, v132
	v_dual_mul_f32 v40, v40, v132 :: v_dual_mul_f32 v39, v39, v132
	v_dual_mul_f32 v38, v38, v132 :: v_dual_mul_f32 v37, v37, v132
	v_dual_mul_f32 v36, v36, v132 :: v_dual_mul_f32 v35, v35, v132
	v_dual_mul_f32 v34, v34, v132 :: v_dual_mul_f32 v33, v33, v132
	v_dual_mul_f32 v32, v32, v132 :: v_dual_mul_f32 v31, v31, v132
	v_dual_mul_f32 v30, v30, v132 :: v_dual_mul_f32 v29, v29, v132
	v_dual_mul_f32 v28, v28, v132 :: v_dual_mul_f32 v27, v27, v132
	v_dual_mul_f32 v26, v26, v132 :: v_dual_mul_f32 v25, v25, v132
	v_dual_mul_f32 v24, v24, v132 :: v_dual_mul_f32 v23, v23, v132
	v_dual_mul_f32 v22, v22, v132 :: v_dual_mul_f32 v21, v21, v132
	v_dual_mul_f32 v20, v20, v132 :: v_dual_mul_f32 v17, v17, v132
	v_dual_mul_f32 v18, v18, v132 :: v_dual_mul_f32 v15, v15, v132
	v_dual_mul_f32 v16, v16, v132 :: v_dual_mul_f32 v13, v13, v132
	v_dual_mul_f32 v14, v14, v132 :: v_dual_mul_f32 v11, v11, v132
	v_dual_mul_f32 v12, v12, v132 :: v_dual_mul_f32 v9, v9, v132
	v_cvt_pk_fp8_f32 v231.h, v1, v232
	v_dual_mul_f32 v4, v4, v132 :: v_dual_mov_b32 v1, v131
	v_mul_f32_e32 v2, v2, v132
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[122:129], v[133:134], v[230:231], v[122:129]
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[135:136], v[230:231], v[114:121]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[137:138], v[230:231], v[106:113]
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[139:140], v[230:231], v[98:105]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[206:207], v[230:231], v[90:97]
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[208:209], v[230:231], v[82:89]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[74:81], v[210:211], v[230:231], v[74:81]
	v_wmma_f32_16x16x16_fp8_fp8 v[66:73], v[212:213], v[230:231], v[66:73]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[214:215], v[230:231], v[58:65]
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[216:217], v[230:231], v[50:57]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[218:219], v[230:231], v[42:49]
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[220:221], v[230:231], v[34:41]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[222:223], v[230:231], v[26:33]
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[224:225], v[230:231], v[18:25]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[226:227], v[230:231], v[10:17]
	v_wmma_f32_16x16x16_fp8_fp8 v[2:9], v[228:229], v[230:231], v[2:9]
	v_dual_mov_b32 v206, v130 :: v_dual_mov_b32 v207, v0
	s_add_co_i32 s24, s24, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 s24, 3
	s_cbranch_scc1 .LBB8_14
.LBB8_19:                               ;   Parent Loop BB8_15 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB8_58 Depth 3
	s_and_saveexec_b32 s5, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB8_25
; %bb.20:                               ;   in Loop: Header=BB8_19 Depth=2
	s_and_saveexec_b32 s6, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB8_22
; %bb.21:                               ;   in Loop: Header=BB8_19 Depth=2
	s_mov_b32 m0, 3
	s_barrier_join m0 
.LBB8_22:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB8_24
; %bb.23:                               ;   in Loop: Header=BB8_19 Depth=2
	s_mov_b32 m0, 2
	s_barrier_join m0 
.LBB8_24:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
.LBB8_25:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s5
	s_cbranch_execz .LBB8_27
; %bb.26:                               ;   in Loop: Header=BB8_19 Depth=2
	s_mov_b32 m0, 1
	s_barrier_join m0 
.LBB8_27:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_lshl4_add_u32 s5, s24, s19
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v132, 0
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v0, s5, v149
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v130, 0
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s20, v0
	s_cbranch_execz .LBB8_29
; %bb.28:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mad_co_i64_i32 v[132:133], null, 0x408, v0, v[145:146]
	s_clause 0x1
	global_load_b64 v[130:131], v[132:133], off
	global_load_b64 v[132:133], v[132:133], off offset:16
.LBB8_29:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	ds_store_b128 v203, v[130:133]
	s_and_saveexec_b32 s6, s2
	s_cbranch_execz .LBB8_48
; %bb.30:                               ;   in Loop: Header=BB8_19 Depth=2
	v_or_b32_e32 v0, s5, v157
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s20, v0
	s_cbranch_execz .LBB8_32
; %bb.31:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mad_co_i64_i32 v[130:131], null, 0x408, v0, v[147:148]
	global_load_b64 v[130:131], v[130:131], off
.LBB8_32:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v134, 1, v0
	v_add_nc_u32_e32 v135, 0x6000, v162
	s_mov_b32 s7, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v135, v130, v131 offset1:1
	v_cmpx_gt_i32_e64 s20, v134
	s_cbranch_execz .LBB8_34
; %bb.33:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mad_co_i64_i32 v[130:131], null, 0x408, v134, v[147:148]
	global_load_b64 v[132:133], v[130:131], off
.LBB8_34:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v136, 2, v0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	s_mov_b32 s7, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v163, v132 offset:24576
	ds_store_b32 v164, v133 offset:24576
	v_cmpx_gt_i32_e64 s20, v136
	s_cbranch_execz .LBB8_36
; %bb.35:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mad_co_i64_i32 v[132:133], null, 0x408, v136, v[147:148]
	global_load_b64 v[134:135], v[132:133], off
.LBB8_36:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v132, 3, v0
	s_mov_b32 s7, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v165, v134 offset:24576
	ds_store_b32 v166, v135 offset:24576
	v_cmpx_gt_i32_e64 s20, v132
	s_cbranch_execz .LBB8_38
; %bb.37:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mad_co_i64_i32 v[130:131], null, 0x408, v132, v[147:148]
	global_load_b64 v[130:131], v[130:131], off
.LBB8_38:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v136, 4, v0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	s_mov_b32 s7, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v167, v130 offset:24576
	ds_store_b32 v168, v131 offset:24576
	v_cmpx_gt_i32_e64 s20, v136
	s_cbranch_execz .LBB8_40
; %bb.39:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mad_co_i64_i32 v[130:131], null, 0x408, v136, v[147:148]
	global_load_b64 v[134:135], v[130:131], off
.LBB8_40:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v130, 5, v0
	v_add_nc_u32_e32 v131, 0x6400, v162
	s_mov_b32 s7, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v131, v134, v135 offset1:1
	v_cmpx_gt_i32_e64 s20, v130
	s_cbranch_execz .LBB8_42
; %bb.41:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mad_co_i64_i32 v[130:131], null, 0x408, v130, v[147:148]
	global_load_b64 v[132:133], v[130:131], off
.LBB8_42:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v136, 6, v0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	s_mov_b32 s7, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v169, v132 offset:24576
	ds_store_b32 v170, v133 offset:24576
	v_cmpx_gt_i32_e64 s20, v136
	s_cbranch_execz .LBB8_44
; %bb.43:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mad_co_i64_i32 v[132:133], null, 0x408, v136, v[147:148]
	global_load_b64 v[134:135], v[132:133], off
.LBB8_44:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v0, 7, v0
	s_mov_b32 s7, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v171, v134 offset:24576
	ds_store_b32 v172, v135 offset:24576
	v_cmpx_gt_i32_e64 s20, v0
	s_cbranch_execz .LBB8_46
; %bb.45:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mad_co_i64_i32 v[130:131], null, 0x408, v0, v[147:148]
	global_load_b64 v[130:131], v[130:131], off
.LBB8_46:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt 0x0
	ds_store_b32 v173, v130 offset:24576
	ds_store_b32 v174, v131 offset:24576
	;;#ASMSTART
	s_wait_loadcnt 0
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 exec_lo, exec_lo, s4
	s_cbranch_execz .LBB8_48
; %bb.47:                               ;   in Loop: Header=BB8_19 Depth=2
	v_add_nc_u32_e32 v0, 0x6000, v175
	v_add_nc_u32_e32 v134, 0x6400, v175
	v_add_nc_u32_e32 v139, 0, v176
	ds_load_2addr_b32 v[130:131], v0 offset1:4
	ds_load_2addr_b32 v[132:133], v134 offset1:4
	s_wait_dscnt 0x1
	ds_bpermute_b32 v135, v154, v130
	s_wait_dscnt 0x1
	ds_bpermute_b32 v136, v154, v132
	ds_bpermute_b32 v137, v154, v131
	ds_bpermute_b32 v138, v154, v133
	s_wait_dscnt 0x3
	v_perm_b32 v130, v135, v130, v158
	s_wait_dscnt 0x2
	v_perm_b32 v132, v136, v132, v158
	s_wait_dscnt 0x1
	v_perm_b32 v135, v137, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v133, v138, v133, v158
	ds_bpermute_b32 v131, v153, v130
	ds_bpermute_b32 v136, v153, v132
	ds_bpermute_b32 v137, v153, v135
	ds_bpermute_b32 v138, v153, v133
	s_wait_dscnt 0x3
	v_perm_b32 v130, v131, v130, v159
	s_wait_dscnt 0x2
	v_perm_b32 v131, v136, v132, v159
	s_wait_dscnt 0x1
	v_perm_b32 v132, v137, v135, v159
	s_wait_dscnt 0x0
	v_perm_b32 v133, v138, v133, v159
	ds_store_b128 v139, v[130:133] offset:12288
	ds_load_b32 v130, v177 offset:24576
	ds_load_b32 v131, v178 offset:24576
	ds_load_b32 v132, v179 offset:24576
	ds_load_b32 v133, v180 offset:24576
	ds_load_b32 v135, v182 offset:24576
	ds_load_b32 v136, v184 offset:24576
	s_wait_dscnt 0x5
	ds_bpermute_b32 v137, v154, v130
	s_wait_dscnt 0x5
	ds_bpermute_b32 v138, v154, v131
	s_wait_dscnt 0x5
	ds_bpermute_b32 v140, v154, v132
	s_wait_dscnt 0x5
	ds_bpermute_b32 v208, v154, v133
	s_wait_dscnt 0x3
	v_perm_b32 v130, v137, v130, v158
	s_wait_dscnt 0x2
	v_perm_b32 v131, v138, v131, v158
	s_wait_dscnt 0x1
	v_perm_b32 v132, v140, v132, v158
	s_wait_dscnt 0x0
	v_perm_b32 v133, v208, v133, v158
	ds_bpermute_b32 v137, v153, v130
	ds_bpermute_b32 v138, v153, v131
	ds_bpermute_b32 v140, v153, v132
	ds_bpermute_b32 v208, v153, v133
	s_wait_dscnt 0x3
	v_perm_b32 v130, v137, v130, v159
	s_wait_dscnt 0x2
	v_perm_b32 v131, v138, v131, v159
	s_wait_dscnt 0x1
	v_perm_b32 v132, v140, v132, v159
	s_wait_dscnt 0x0
	v_perm_b32 v133, v208, v133, v159
	ds_store_b128 v139, v[130:133] offset:12800
	ds_load_b32 v130, v181 offset:24576
	ds_load_b32 v131, v183 offset:24576
	ds_bpermute_b32 v132, v154, v135
	ds_bpermute_b32 v133, v154, v136
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v154, v130
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v154, v131
	s_wait_dscnt 0x3
	v_perm_b32 v132, v132, v135, v158
	s_wait_dscnt 0x2
	v_perm_b32 v133, v133, v136, v158
	ds_bpermute_b32 v136, v153, v133
	s_wait_dscnt 0x2
	v_perm_b32 v130, v137, v130, v158
	s_wait_dscnt 0x1
	v_perm_b32 v135, v138, v131, v158
	ds_bpermute_b32 v131, v153, v132
	ds_bpermute_b32 v137, v153, v130
	ds_bpermute_b32 v138, v153, v135
	s_wait_dscnt 0x3
	v_perm_b32 v133, v136, v133, v159
	s_wait_dscnt 0x2
	v_perm_b32 v131, v131, v132, v159
	s_wait_dscnt 0x1
	v_perm_b32 v130, v137, v130, v159
	s_wait_dscnt 0x0
	v_perm_b32 v132, v138, v135, v159
	ds_store_b128 v139, v[130:133] offset:13312
	ds_load_b32 v130, v185 offset:24576
	ds_load_b32 v131, v186 offset:24576
	ds_load_b32 v132, v187 offset:24576
	ds_load_b32 v133, v188 offset:24576
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v154, v130
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v154, v131
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v154, v132
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v154, v133
	s_wait_dscnt 0x3
	v_perm_b32 v130, v135, v130, v158
	s_wait_dscnt 0x2
	v_perm_b32 v131, v136, v131, v158
	s_wait_dscnt 0x1
	v_perm_b32 v132, v137, v132, v158
	s_wait_dscnt 0x0
	v_perm_b32 v133, v138, v133, v158
	ds_bpermute_b32 v135, v153, v130
	ds_bpermute_b32 v136, v153, v131
	ds_bpermute_b32 v137, v153, v132
	ds_bpermute_b32 v138, v153, v133
	s_wait_dscnt 0x3
	v_perm_b32 v130, v135, v130, v159
	s_wait_dscnt 0x2
	v_perm_b32 v131, v136, v131, v159
	s_wait_dscnt 0x1
	v_perm_b32 v132, v137, v132, v159
	s_wait_dscnt 0x0
	v_perm_b32 v133, v138, v133, v159
	ds_store_b128 v139, v[130:133] offset:13824
	ds_load_2addr_b32 v[130:131], v0 offset0:32 offset1:36
	ds_load_2addr_b32 v[132:133], v134 offset0:32 offset1:36
	s_wait_dscnt 0x1
	ds_bpermute_b32 v0, v154, v130
	s_wait_dscnt 0x1
	ds_bpermute_b32 v134, v154, v132
	ds_bpermute_b32 v135, v154, v131
	ds_bpermute_b32 v136, v154, v133
	s_wait_dscnt 0x3
	v_perm_b32 v0, v0, v130, v158
	s_wait_dscnt 0x2
	v_perm_b32 v132, v134, v132, v158
	s_wait_dscnt 0x1
	v_perm_b32 v134, v135, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v133, v136, v133, v158
	ds_bpermute_b32 v130, v153, v0
	ds_bpermute_b32 v131, v153, v132
	ds_bpermute_b32 v135, v153, v134
	ds_bpermute_b32 v136, v153, v133
	s_wait_dscnt 0x3
	v_perm_b32 v130, v130, v0, v159
	s_wait_dscnt 0x2
	v_perm_b32 v131, v131, v132, v159
	s_wait_dscnt 0x1
	v_perm_b32 v132, v135, v134, v159
	s_wait_dscnt 0x0
	v_perm_b32 v133, v136, v133, v159
	ds_store_b128 v139, v[130:133] offset:14336
	ds_load_b32 v0, v189 offset:24576
	ds_load_b32 v130, v190 offset:24576
	ds_load_b32 v131, v191 offset:24576
	ds_load_b32 v132, v192 offset:24576
	s_wait_dscnt 0x3
	ds_bpermute_b32 v133, v154, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v154, v130
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v154, v131
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v154, v132
	s_wait_dscnt 0x3
	v_perm_b32 v0, v133, v0, v158
	s_wait_dscnt 0x2
	v_perm_b32 v133, v134, v130, v158
	s_wait_dscnt 0x1
	v_perm_b32 v134, v135, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v135, v136, v132, v158
	ds_bpermute_b32 v130, v153, v0
	ds_bpermute_b32 v131, v153, v133
	ds_bpermute_b32 v132, v153, v134
	ds_bpermute_b32 v136, v153, v135
	s_wait_dscnt 0x3
	v_perm_b32 v130, v130, v0, v159
	s_wait_dscnt 0x2
	v_perm_b32 v131, v131, v133, v159
	s_wait_dscnt 0x1
	v_perm_b32 v132, v132, v134, v159
	s_wait_dscnt 0x0
	v_perm_b32 v133, v136, v135, v159
	ds_store_b128 v139, v[130:133] offset:14848
	ds_load_b32 v0, v193 offset:24576
	ds_load_b32 v130, v194 offset:24576
	ds_load_b32 v131, v195 offset:24576
	ds_load_b32 v132, v196 offset:24576
	s_wait_dscnt 0x3
	ds_bpermute_b32 v133, v154, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v154, v130
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v154, v131
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v154, v132
	s_wait_dscnt 0x3
	v_perm_b32 v0, v133, v0, v158
	s_wait_dscnt 0x2
	v_perm_b32 v133, v134, v130, v158
	s_wait_dscnt 0x1
	v_perm_b32 v134, v135, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v135, v136, v132, v158
	ds_bpermute_b32 v130, v153, v0
	ds_bpermute_b32 v131, v153, v133
	ds_bpermute_b32 v132, v153, v134
	ds_bpermute_b32 v136, v153, v135
	s_wait_dscnt 0x3
	v_perm_b32 v130, v130, v0, v159
	s_wait_dscnt 0x2
	v_perm_b32 v131, v131, v133, v159
	s_wait_dscnt 0x1
	v_perm_b32 v132, v132, v134, v159
	s_wait_dscnt 0x0
	v_perm_b32 v133, v136, v135, v159
	ds_store_b128 v139, v[130:133] offset:15360
	ds_load_b32 v0, v197 offset:24576
	ds_load_b32 v130, v198 offset:24576
	ds_load_b32 v131, v199 offset:24576
	ds_load_b32 v132, v200 offset:24576
	s_wait_dscnt 0x3
	ds_bpermute_b32 v133, v154, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v154, v130
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v154, v131
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v154, v132
	s_wait_dscnt 0x3
	v_perm_b32 v0, v133, v0, v158
	s_wait_dscnt 0x2
	v_perm_b32 v133, v134, v130, v158
	s_wait_dscnt 0x1
	v_perm_b32 v134, v135, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v135, v136, v132, v158
	ds_bpermute_b32 v130, v153, v0
	ds_bpermute_b32 v131, v153, v133
	ds_bpermute_b32 v132, v153, v134
	ds_bpermute_b32 v136, v153, v135
	s_wait_dscnt 0x3
	v_perm_b32 v130, v130, v0, v159
	s_wait_dscnt 0x2
	v_perm_b32 v131, v131, v133, v159
	s_wait_dscnt 0x1
	v_perm_b32 v132, v132, v134, v159
	s_wait_dscnt 0x0
	v_perm_b32 v133, v136, v135, v159
	ds_store_b128 v139, v[130:133] offset:15872
.LBB8_48:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s22
	s_cbranch_execz .LBB8_52
; %bb.49:                               ;   in Loop: Header=BB8_19 Depth=2
	v_or_b32_e32 v130, s5, v151
	v_mov_b32_e32 v0, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s20, v130
	s_cbranch_execz .LBB8_51
; %bb.50:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mad_co_i64_i32 v[131:132], null, 0x408, v130, s[10:11]
	v_mad_co_i64_i32 v[133:134], null, 0x408, v130, s[12:13]
	global_load_d16_b16 v0, v[131:132], off offset:1024
	global_load_d16_hi_b16 v0, v[133:134], off offset:1024
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v130.h, 8, v0.l
	v_lshrrev_b16 v130.l, 8, v0.h
	v_and_b16 v131.h, 0xff, v0.l
	v_and_b16 v131.l, 0xff, v0.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_lshlrev_b16 v0, 8, v130 op_sel_hi:[0,1]
	v_or_b32_e32 v0, v0, v131
.LBB8_51:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_store_b16_d16_hi v160, v0 offset:36864
	ds_store_b16 v160, v0 offset:36960
.LBB8_52:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x0
	s_and_saveexec_b32 s6, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execnz .LBB8_63
; %bb.53:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execnz .LBB8_68
.LBB8_54:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execnz .LBB8_69
.LBB8_55:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execnz .LBB8_74
.LBB8_56:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_cmp_ge_i32 s5, s20
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB8_75
.LBB8_57:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mov_b32_e32 v130, 0
	s_mov_b32 s6, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v131, v130 :: v_dual_mov_b32 v132, v130
	v_dual_mov_b32 v133, v130 :: v_dual_mov_b32 v134, v130
	v_dual_mov_b32 v135, v130 :: v_dual_mov_b32 v136, v130
	v_mov_b32_e32 v137, v130
.LBB8_58:                               ;   Parent Loop BB8_15 Depth=1
                                        ;     Parent Loop BB8_19 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s7, s6, 6
	v_mad_co_u64_u32 v[208:209], null, s6, 6, v[142:143]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v138, vcc_lo, v201, s7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v139, null, 0, v202, vcc_lo
	s_add_co_i32 s6, s6, 1
	s_clause 0x3
	global_load_b64 v[216:217], v[138:139], off
	global_load_b64 v[218:219], v[138:139], off offset:16
	global_load_b64 v[220:221], v[138:139], off offset:32
	global_load_b64 v[138:139], v[138:139], off offset:48
	v_lshl_add_u32 v0, v208, 9, v161
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s6, 4
	ds_load_b128 v[208:211], v0
	ds_load_b128 v[212:215], v0 offset:1536
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[130:137], v[208:209], v[216:217], v[130:137]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[130:137], v[210:211], v[218:219], v[130:137]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[130:137], v[212:213], v[220:221], v[130:137]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[130:137], v[214:215], v[138:139], v[130:137]
	s_cbranch_scc1 .LBB8_58
; %bb.59:                               ;   in Loop: Header=BB8_19 Depth=2
	ds_load_b96 v[138:140], v204 offset:36866
	ds_load_u16_d16 v0, v204 offset:36878
	s_or_b32 s6, s5, 15
	v_or_b32_e32 v209, s5, v155
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s6, s20
	v_mov_b32_e32 v208, 0xff800000
	s_cselect_b32 s7, -1, 0
	s_cmp_le_i32 s6, s16
	s_cselect_b32 s6, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s23, s5
	s_and_saveexec_b32 s5, s0
	s_cbranch_execz .LBB8_61
; %bb.60:                               ;   in Loop: Header=BB8_19 Depth=2
	ds_load_u16_d16 v208, v204 offset:36864
	v_mul_f32_e32 v130, v150, v130
	v_cmp_le_i32_e32 vcc_lo, v209, v205
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s6, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v130, v208, v130, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v208, 0xff800000, v130, vcc_lo
.LBB8_61:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_mul_f32_e32 v130, v150, v131
	v_or_b32_e32 v131, 2, v209
	v_cmp_lt_i32_e32 vcc_lo, v209, v205
	v_dual_mul_f32 v132, v150, v132 :: v_dual_mul_f32 v135, v150, v135
	s_wait_dscnt 0x1
	v_fma_mix_f32 v130, v138, v130, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e64 s5, v131, v205
	v_or_b32_e32 v131, 3, v209
	s_or_b32 s7, s6, vcc_lo
	v_fma_mix_f32 v132, v138, v132, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s7
	s_or_b32 s5, s6, s5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v130, 0xff800000, v130, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v131, v205
	v_mul_f32_e32 v131, v150, v133
	v_or_b32_e32 v133, 4, v209
	s_and_b32 s5, s0, s5
	v_mul_f32_e32 v134, v150, v134
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v132, 0xff800000, v132, s5
	s_or_b32 s5, s6, vcc_lo
	v_fma_mix_f32 v131, v139, v131, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e32 vcc_lo, v133, v205
	v_or_b32_e32 v133, 5, v209
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s0, s5
	v_fma_mix_f32 v134, v139, v134, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v131, 0xff800000, v131, s5
	s_or_b32 s5, s6, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v133, v205
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s0, s5
	v_or_b32_e32 v138, 7, v209
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v133, 0xff800000, v134, s5
	v_fma_mix_f32 v134, v140, v135, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v135, 6, v209
	s_or_b32 s5, s6, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s5
	v_cmp_le_i32_e64 s5, v138, v205
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v134, 0xff800000, v134, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v135, v205
	v_dual_mul_f32 v135, v150, v136 :: v_dual_mul_f32 v136, v150, v137
	v_max3_num_f32 v137, v208, 0xff800000, v130
	s_or_b32 s5, s6, s5
	s_or_b32 s7, s6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_mix_f32 v135, v140, v135, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v0, v0, v136, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v136, v137, v132, v131
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s7
	v_add_nc_u32_e32 v138, 0x9060, v204
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v135, 0xff800000, v135, vcc_lo
	s_and_b32 vcc_lo, s0, s5
	s_mov_b32 s5, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v137, 0xff800000, v0, vcc_lo
	v_max3_num_f32 v0, v136, v133, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v0, v0, v135, v137
	v_mov_b32_e32 v136, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v136, v136, s9, 0xfedcba98
	v_max3_num_f32 v0, v207, v0, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v130, v130, v0 :: v_dual_sub_f32 v133, v133, v0
	v_dual_sub_f32 v136, v137, v0 :: v_dual_sub_f32 v131, v131, v0
	v_dual_sub_f32 v137, v208, v0 :: v_dual_mul_f32 v130, 0x3fb8aa3b, v130
	v_dual_sub_f32 v135, v135, v0 :: v_dual_sub_f32 v134, v134, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v140, 0x3fb8aa3b, v131
	v_mul_f32_e32 v136, 0x3fb8aa3b, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v139, v130
	ds_load_2addr_b32 v[130:131], v138 offset1:1
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_sub_f32 v132, v132, v0
	v_mul_f32_e32 v134, 0x3fb8aa3b, v134
	v_add_nc_u32_e32 v138, 0x9068, v204
	v_mul_f32_e32 v135, 0x3fb8aa3b, v135
	v_exp_f32_e32 v136, v136
	v_exp_f32_e32 v137, v137
	v_mul_f32_e32 v132, 0x3fb8aa3b, v132
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v0
	v_mul_f32_e32 v208, 0x3fb8aa3b, v133
	v_exp_f32_e32 v209, v134
	ds_load_2addr_b32 v[133:134], v138 offset1:1
	v_exp_f32_e32 v140, v140
	v_exp_f32_e32 v132, v132
	v_exp_f32_e32 v138, v208
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v208, v136, 0, vcc_lo
	v_cndmask_b32_e64 v136, v137, 0, vcc_lo
	v_cndmask_b32_e64 v137, v139, 0, vcc_lo
	v_exp_f32_e32 v135, v135
	v_cndmask_b32_e64 v209, v209, 0, vcc_lo
	v_cndmask_b32_e64 v210, v140, 0, vcc_lo
	v_cndmask_b32_e64 v132, v132, 0, vcc_lo
	s_wait_dscnt 0x1
	v_fma_mix_f32 v140, v130, v136, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v139, v130, v137, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v136, v136, v137
	v_cndmask_b32_e64 v130, v138, 0, vcc_lo
	v_fma_mix_f32 v138, v131, v132, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v137, v131, v210, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_max3_num_f32 v131, v140, 0, v139
	v_add_f32_e32 v132, v132, v136
	v_cndmask_b32_e64 v211, v135, 0, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v136, v133, v130, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v135, v133, v209, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_max3_num_f32 v131, v131, v138, v137
	v_add_f32_e32 v132, v210, v132
	v_fma_mix_f32 v133, v134, v211, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v134, v134, v208, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_max3_num_f32 v131, v131, v136, v135
	v_add_f32_e32 v130, v130, v132
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v210, v131, v133, v134
	v_add_f32_e32 v130, v209, v130
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b32_e32 v132, v210
	v_add_f32_e32 v130, v211, v130
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_permlanex16_b32 v132, v132, s9, 0xfedcba98
	v_add_f32_e32 v131, v208, v130
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v130, v132, v132
	v_mov_b32_e32 v132, v131
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v208, v210, v130
	v_permlanex16_b32 v132, v132, s9, 0xfedcba98
	v_mov_b32_e32 v130, v206
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_lt_f32_e32 0, v208
	s_cbranch_execz .LBB8_18
; %bb.62:                               ;   in Loop: Header=BB8_19 Depth=2
	v_div_scale_f32 v130, null, 0x43e00000, 0x43e00000, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v209, v130
	v_fma_f32 v210, -v130, v209, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v209, v210, v209
	v_div_scale_f32 v210, vcc_lo, v208, 0x43e00000, v208
	v_mul_f32_e32 v211, v210, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v212, -v130, v211, v210
	v_fmac_f32_e32 v211, v212, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v130, -v130, v211, v210
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v130, v130, v209, v211
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v130, v130, 0x43e00000, v208
	v_max_num_f32_e32 v130, 0x1f800000, v130
	s_branch .LBB8_18
.LBB8_63:                               ;   in Loop: Header=BB8_19 Depth=2
	s_and_saveexec_b32 s7, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB8_65
; %bb.64:                               ;   in Loop: Header=BB8_19 Depth=2
	s_barrier_signal 3
.LBB8_65:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB8_67
; %bb.66:                               ;   in Loop: Header=BB8_19 Depth=2
	s_barrier_signal 2
.LBB8_67:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB8_54
.LBB8_68:                               ;   in Loop: Header=BB8_19 Depth=2
	s_barrier_signal 1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB8_55
.LBB8_69:                               ;   in Loop: Header=BB8_19 Depth=2
	s_and_saveexec_b32 s7, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB8_71
; %bb.70:                               ;   in Loop: Header=BB8_19 Depth=2
	s_barrier_wait 3
.LBB8_71:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB8_73
; %bb.72:                               ;   in Loop: Header=BB8_19 Depth=2
	s_barrier_wait 2
.LBB8_73:                               ;   in Loop: Header=BB8_19 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB8_56
.LBB8_74:                               ;   in Loop: Header=BB8_19 Depth=2
	s_barrier_wait 1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_cmp_ge_i32 s5, s20
	global_inv scope:SCOPE_SE
	s_cbranch_scc0 .LBB8_57
.LBB8_75:                               ;   in Loop: Header=BB8_19 Depth=2
	v_mov_b32_e32 v0, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_mov_b32_e32 v207, v0
	s_add_co_i32 s24, s24, 1
	s_cmp_eq_u32 s24, 3
	s_cbranch_scc0 .LBB8_19
	s_branch .LBB8_14
.LBB8_76:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v206, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_mov_b32_e32 v8, v1
	v_dual_mov_b32 v2, v1 :: v_dual_mov_b32 v3, v1
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	v_dual_mov_b32 v0, 0xff800000 :: v_dual_mov_b32 v129, v8
	v_dual_mov_b32 v122, v1 :: v_dual_mov_b32 v121, v8
	v_dual_mov_b32 v114, v1 :: v_dual_mov_b32 v113, v8
	v_dual_mov_b32 v106, v1 :: v_dual_mov_b32 v105, v8
	v_dual_mov_b32 v98, v1 :: v_dual_mov_b32 v97, v8
	v_dual_mov_b32 v90, v1 :: v_dual_mov_b32 v89, v8
	v_dual_mov_b32 v82, v1 :: v_dual_mov_b32 v81, v8
	v_dual_mov_b32 v74, v1 :: v_dual_mov_b32 v73, v8
	v_dual_mov_b32 v66, v1 :: v_dual_mov_b32 v65, v8
	v_dual_mov_b32 v58, v1 :: v_dual_mov_b32 v57, v8
	v_dual_mov_b32 v50, v1 :: v_dual_mov_b32 v49, v8
	v_dual_mov_b32 v42, v1 :: v_dual_mov_b32 v41, v8
	v_dual_mov_b32 v34, v1 :: v_dual_mov_b32 v33, v8
	v_dual_mov_b32 v26, v1 :: v_dual_mov_b32 v25, v8
	v_dual_mov_b32 v18, v1 :: v_dual_mov_b32 v17, v8
	v_dual_mov_b32 v128, v7 :: v_dual_mov_b32 v127, v6
	v_dual_mov_b32 v126, v5 :: v_dual_mov_b32 v125, v4
	v_dual_mov_b32 v124, v3 :: v_dual_mov_b32 v123, v2
	v_dual_mov_b32 v120, v7 :: v_dual_mov_b32 v119, v6
	v_dual_mov_b32 v118, v5 :: v_dual_mov_b32 v117, v4
	v_dual_mov_b32 v116, v3 :: v_dual_mov_b32 v115, v2
	v_dual_mov_b32 v112, v7 :: v_dual_mov_b32 v111, v6
	v_dual_mov_b32 v110, v5 :: v_dual_mov_b32 v109, v4
	v_dual_mov_b32 v108, v3 :: v_dual_mov_b32 v107, v2
	v_dual_mov_b32 v104, v7 :: v_dual_mov_b32 v103, v6
	v_dual_mov_b32 v102, v5 :: v_dual_mov_b32 v101, v4
	v_dual_mov_b32 v100, v3 :: v_dual_mov_b32 v99, v2
	v_dual_mov_b32 v96, v7 :: v_dual_mov_b32 v95, v6
	v_dual_mov_b32 v94, v5 :: v_dual_mov_b32 v93, v4
	v_dual_mov_b32 v92, v3 :: v_dual_mov_b32 v91, v2
	v_dual_mov_b32 v88, v7 :: v_dual_mov_b32 v87, v6
	v_dual_mov_b32 v86, v5 :: v_dual_mov_b32 v85, v4
	v_dual_mov_b32 v84, v3 :: v_dual_mov_b32 v83, v2
	v_dual_mov_b32 v80, v7 :: v_dual_mov_b32 v79, v6
	v_dual_mov_b32 v78, v5 :: v_dual_mov_b32 v77, v4
	v_dual_mov_b32 v76, v3 :: v_dual_mov_b32 v75, v2
	v_dual_mov_b32 v72, v7 :: v_dual_mov_b32 v71, v6
	v_dual_mov_b32 v70, v5 :: v_dual_mov_b32 v69, v4
	v_dual_mov_b32 v68, v3 :: v_dual_mov_b32 v67, v2
	v_dual_mov_b32 v64, v7 :: v_dual_mov_b32 v63, v6
	v_dual_mov_b32 v62, v5 :: v_dual_mov_b32 v61, v4
	v_dual_mov_b32 v60, v3 :: v_dual_mov_b32 v59, v2
	v_dual_mov_b32 v56, v7 :: v_dual_mov_b32 v55, v6
	v_dual_mov_b32 v54, v5 :: v_dual_mov_b32 v53, v4
	v_dual_mov_b32 v52, v3 :: v_dual_mov_b32 v51, v2
	v_dual_mov_b32 v48, v7 :: v_dual_mov_b32 v47, v6
	v_dual_mov_b32 v46, v5 :: v_dual_mov_b32 v45, v4
	v_dual_mov_b32 v44, v3 :: v_dual_mov_b32 v43, v2
	v_dual_mov_b32 v40, v7 :: v_dual_mov_b32 v39, v6
	v_dual_mov_b32 v38, v5 :: v_dual_mov_b32 v37, v4
	v_dual_mov_b32 v36, v3 :: v_dual_mov_b32 v35, v2
	v_dual_mov_b32 v32, v7 :: v_dual_mov_b32 v31, v6
	v_dual_mov_b32 v30, v5 :: v_dual_mov_b32 v29, v4
	v_dual_mov_b32 v28, v3 :: v_dual_mov_b32 v27, v2
	v_dual_mov_b32 v24, v7 :: v_dual_mov_b32 v23, v6
	v_dual_mov_b32 v22, v5 :: v_dual_mov_b32 v21, v4
	v_dual_mov_b32 v20, v3 :: v_dual_mov_b32 v19, v2
	v_dual_mov_b32 v16, v7 :: v_dual_mov_b32 v15, v6
	v_dual_mov_b32 v14, v5 :: v_dual_mov_b32 v13, v4
	v_dual_mov_b32 v12, v3 :: v_dual_mov_b32 v11, v2
	v_dual_mov_b32 v10, v1 :: v_dual_mov_b32 v9, v8
	v_mov_b32_e32 v8, v7
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v1
.LBB8_77:
	v_cmp_eq_u32_e32 vcc_lo, 0, v152
	s_wait_kmcnt 0x0
	s_and_b32 s2, vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB8_79
; %bb.78:
	v_mad_co_u64_u32 v[130:131], null, s17, v141, s[18:19]
	v_mov_b32_e32 v131, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v130, 0x102, v130
	v_lshlrev_b64_e32 v[130:131], 2, v[130:131]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v130, vcc_lo, s14, v130
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v131, null, s15, v131, vcc_lo
	global_store_b64 v[130:131], v[0:1], off
.LBB8_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB8_81
; %bb.80:
	v_mad_co_u64_u32 v[130:131], null, s17, v141, s[18:19]
	v_dual_mov_b32 v131, 0 :: v_dual_mul_f32 v0, v206, v122
	v_dual_mul_f32 v114, v114, v206 :: v_dual_lshlrev_b32 v1, 5, v152
	v_mul_f32_e32 v108, v108, v206
	v_dual_mul_f32 v106, v106, v206 :: v_dual_mul_f32 v115, v115, v206
	v_mul_lo_u32 v130, 0x102, v130
	v_dual_mul_f32 v98, v98, v206 :: v_dual_mul_f32 v107, v107, v206
	v_dual_mul_f32 v90, v90, v206 :: v_dual_mul_f32 v99, v99, v206
	v_dual_mul_f32 v82, v82, v206 :: v_dual_mul_f32 v91, v91, v206
	v_dual_mul_f32 v74, v74, v206 :: v_dual_mul_f32 v83, v83, v206
	v_lshlrev_b64_e32 v[130:131], 2, v[130:131]
	v_dual_mul_f32 v66, v66, v206 :: v_dual_mul_f32 v75, v75, v206
	v_dual_mul_f32 v58, v58, v206 :: v_dual_mul_f32 v67, v67, v206
	v_dual_mul_f32 v50, v50, v206 :: v_dual_mul_f32 v59, v59, v206
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_co_u32 v122, vcc_lo, s14, v130
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v131, null, s15, v131, vcc_lo
	v_dual_mul_f32 v42, v42, v206 :: v_dual_mul_f32 v51, v51, v206
	v_add_co_u32 v130, vcc_lo, v122, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v131, null, 0, v131, vcc_lo
	v_dual_mul_f32 v122, v2, v206 :: v_dual_mul_f32 v11, v11, v206
	v_dual_mul_f32 v1, v206, v123 :: v_dual_mul_f32 v2, v206, v124
	v_dual_mul_f32 v123, v3, v206 :: v_dual_mul_f32 v116, v116, v206
	v_mul_f32_e32 v3, v206, v125
	v_dual_mul_f32 v34, v34, v206 :: v_dual_mul_f32 v43, v43, v206
	v_dual_mul_f32 v26, v26, v206 :: v_dual_mul_f32 v35, v35, v206
	v_dual_mul_f32 v18, v18, v206 :: v_dual_mul_f32 v27, v27, v206
	v_dual_mul_f32 v10, v10, v206 :: v_dual_mul_f32 v19, v19, v206
	v_dual_mul_f32 v100, v100, v206 :: v_dual_mul_f32 v127, v206, v127
	v_dual_mul_f32 v92, v92, v206 :: v_dual_mul_f32 v129, v206, v129
	v_dual_mul_f32 v84, v84, v206 :: v_dual_mul_f32 v117, v117, v206
	v_dual_mul_f32 v76, v76, v206 :: v_dual_mul_f32 v109, v109, v206
	v_dual_mul_f32 v68, v68, v206 :: v_dual_mul_f32 v101, v101, v206
	v_dual_mul_f32 v60, v60, v206 :: v_dual_mul_f32 v93, v93, v206
	v_dual_mul_f32 v52, v52, v206 :: v_dual_mul_f32 v85, v85, v206
	v_dual_mul_f32 v44, v44, v206 :: v_dual_mul_f32 v77, v77, v206
	v_dual_mul_f32 v36, v36, v206 :: v_dual_mul_f32 v69, v69, v206
	v_dual_mul_f32 v28, v28, v206 :: v_dual_mul_f32 v61, v61, v206
	v_dual_mul_f32 v20, v20, v206 :: v_dual_mul_f32 v53, v53, v206
	v_dual_mul_f32 v12, v12, v206 :: v_dual_mul_f32 v45, v45, v206
	v_dual_mul_f32 v124, v4, v206 :: v_dual_mul_f32 v37, v37, v206
	v_mul_f32_e32 v126, v206, v126
	v_dual_mul_f32 v128, v206, v128 :: v_dual_mul_f32 v29, v29, v206
	v_mul_f32_e32 v21, v21, v206
	global_store_b128 v[130:131], v[0:3], off offset:8
	v_dual_mul_f32 v13, v13, v206 :: v_dual_mul_f32 v4, v110, v206
	v_dual_mul_f32 v125, v5, v206 :: v_dual_mul_f32 v102, v102, v206
	v_mul_f32_e32 v0, v118, v206
	v_dual_mul_f32 v94, v94, v206 :: v_dual_mul_f32 v1, v119, v206
	v_dual_mul_f32 v86, v86, v206 :: v_dual_mul_f32 v3, v121, v206
	v_dual_mul_f32 v78, v78, v206 :: v_dual_mul_f32 v5, v111, v206
	v_dual_mul_f32 v70, v70, v206 :: v_dual_mul_f32 v103, v103, v206
	v_dual_mul_f32 v62, v62, v206 :: v_dual_mul_f32 v95, v95, v206
	v_dual_mul_f32 v54, v54, v206 :: v_dual_mul_f32 v87, v87, v206
	v_dual_mul_f32 v46, v46, v206 :: v_dual_mul_f32 v79, v79, v206
	v_dual_mul_f32 v38, v38, v206 :: v_dual_mul_f32 v71, v71, v206
	v_dual_mul_f32 v30, v30, v206 :: v_dual_mul_f32 v63, v63, v206
	v_dual_mul_f32 v2, v120, v206 :: v_dual_mul_f32 v55, v55, v206
	v_dual_mul_f32 v22, v22, v206 :: v_dual_mul_f32 v47, v47, v206
	v_dual_mul_f32 v14, v14, v206 :: v_dual_mul_f32 v39, v39, v206
	v_dual_mul_f32 v110, v6, v206 :: v_dual_mul_f32 v31, v31, v206
	v_dual_mul_f32 v23, v23, v206 :: v_dual_mul_f32 v6, v112, v206
	v_dual_mul_f32 v15, v15, v206 :: v_dual_mul_f32 v104, v104, v206
	v_dual_mul_f32 v111, v7, v206 :: v_dual_mul_f32 v96, v96, v206
	v_dual_mul_f32 v88, v88, v206 :: v_dual_mul_f32 v7, v113, v206
	v_dual_mul_f32 v80, v80, v206 :: v_dual_mul_f32 v105, v105, v206
	v_dual_mul_f32 v72, v72, v206 :: v_dual_mul_f32 v97, v97, v206
	v_dual_mul_f32 v64, v64, v206 :: v_dual_mul_f32 v89, v89, v206
	v_dual_mul_f32 v56, v56, v206 :: v_dual_mul_f32 v81, v81, v206
	v_dual_mul_f32 v48, v48, v206 :: v_dual_mul_f32 v73, v73, v206
	v_dual_mul_f32 v40, v40, v206 :: v_dual_mul_f32 v65, v65, v206
	v_dual_mul_f32 v32, v32, v206 :: v_dual_mul_f32 v57, v57, v206
	v_dual_mul_f32 v24, v24, v206 :: v_dual_mul_f32 v49, v49, v206
	v_dual_mul_f32 v16, v16, v206 :: v_dual_mul_f32 v41, v41, v206
	v_dual_mul_f32 v112, v8, v206 :: v_dual_mul_f32 v33, v33, v206
	v_mul_f32_e32 v25, v25, v206
	v_mul_f32_e32 v17, v17, v206
	s_clause 0x18
	global_store_b128 v[130:131], v[126:129], off offset:24
	global_store_b128 v[130:131], v[114:117], off offset:72
	global_store_b128 v[130:131], v[0:3], off offset:88
	global_store_b128 v[130:131], v[106:109], off offset:136
	global_store_b128 v[130:131], v[4:7], off offset:152
	global_store_b128 v[130:131], v[98:101], off offset:200
	global_store_b128 v[130:131], v[102:105], off offset:216
	global_store_b128 v[130:131], v[90:93], off offset:264
	global_store_b128 v[130:131], v[94:97], off offset:280
	global_store_b128 v[130:131], v[82:85], off offset:328
	global_store_b128 v[130:131], v[86:89], off offset:344
	global_store_b128 v[130:131], v[74:77], off offset:392
	global_store_b128 v[130:131], v[78:81], off offset:408
	global_store_b128 v[130:131], v[66:69], off offset:456
	global_store_b128 v[130:131], v[70:73], off offset:472
	global_store_b128 v[130:131], v[58:61], off offset:520
	global_store_b128 v[130:131], v[62:65], off offset:536
	global_store_b128 v[130:131], v[50:53], off offset:584
	global_store_b128 v[130:131], v[54:57], off offset:600
	global_store_b128 v[130:131], v[42:45], off offset:648
	global_store_b128 v[130:131], v[46:49], off offset:664
	global_store_b128 v[130:131], v[34:37], off offset:712
	global_store_b128 v[130:131], v[38:41], off offset:728
	global_store_b128 v[130:131], v[26:29], off offset:776
	global_store_b128 v[130:131], v[30:33], off offset:792
	v_mul_f32_e32 v113, v9, v206
	s_clause 0x5
	global_store_b128 v[130:131], v[18:21], off offset:840
	global_store_b128 v[130:131], v[22:25], off offset:856
	global_store_b128 v[130:131], v[10:13], off offset:904
	global_store_b128 v[130:131], v[14:17], off offset:920
	global_store_b128 v[130:131], v[122:125], off offset:968
	global_store_b128 v[130:131], v[110:113], off offset:984
.LBB8_81:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end8:
	.size	attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201, .Lfunc_end8-attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201
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
		.amdhsa_next_free_vgpr 233
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end8-attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.num_vgpr, 233
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.numbered_sgpr, 25
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.num_named_barrier, 3
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10620
; TotalNumSgprs: 27
; NumVgprs: 233
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 27
; NumVGPRsForWavesPerEU: 233
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
	s_cbranch_vccnz .LBB9_10
; %bb.1:
	v_lshrrev_b32_e32 v1, 5, v0
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v4, ttmp9, 3, v1
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB9_10
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
.LBB9_3:                                ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_scc0 .LBB9_3
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
	s_branch .LBB9_6
.LBB9_5:                                ;   in Loop: Header=BB9_6 Depth=1
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
	s_cbranch_execz .LBB9_10
.LBB9_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB9_8 Depth 2
	v_dual_mov_b32 v6, v2 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	s_mov_b32 s0, s7
	s_branch .LBB9_8
.LBB9_7:                                ;   in Loop: Header=BB9_8 Depth=2
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
	s_cbranch_scc1 .LBB9_5
.LBB9_8:                                ;   Parent Loop BB9_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	s_mov_b32 s2, exec_lo
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	s_cbranch_execz .LBB9_7
; %bb.9:                                ;   in Loop: Header=BB9_8 Depth=2
	global_load_b32 v14, v[5:6], off
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	v_exp_f32_e32 v14, v14
	s_branch .LBB9_7
.LBB9_10:
	s_endpgm
.Lfunc_end9:
	.size	attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201, .Lfunc_end9-attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end9-attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201)<<4)&4080)>>4
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
	.type	__hip_cuid_b554d5921064ddc,@object ; @__hip_cuid_b554d5921064ddc
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_b554d5921064ddc
__hip_cuid_b554d5921064ddc:
	.byte	0                               ; 0x0
	.size	__hip_cuid_b554d5921064ddc, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym fa2_pkt_kv_ready0
	.addrsig_sym fa2_pkt_kv_ready1
	.addrsig_sym fa2_pkt_kv_ready2
	.addrsig_sym __hip_cuid_b554d5921064ddc
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
      - .address_space:  global
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
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
      - .offset:         64
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 68
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
      - .address_space:  global
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
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
      - .offset:         64
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 68
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 768
    .name:           attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     29
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     216
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
    .sgpr_count:     28
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     237
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
    .max_flat_workgroup_size: 768
    .name:           attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     27
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_q384_partial_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     233
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
