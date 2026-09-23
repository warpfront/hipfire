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
	s_cbranch_vccnz .LBB0_87
; %bb.1:
	s_cmp_gt_i32 ttmp7, 3
	s_cbranch_scc1 .LBB0_87
; %bb.2:
	s_load_b32 s22, s[0:1], 0x38
	s_lshl_b32 s3, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s3, s7
	s_cbranch_scc1 .LBB0_87
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
	v_dual_mov_b32 v1, -1 :: v_dual_and_b32 v6, 31, v0
	v_bfrev_b32_e32 v5, -2
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_u32_e32 8, v6
	s_cbranch_execz .LBB0_9
; %bb.6:
	v_or_b32_e32 v4, s3, v6
	v_bfrev_b32_e32 v5, -2
	v_mov_b32_e32 v1, -1
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
	global_load_b32 v1, v[4:5], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v5, v1
.LBB0_8:                                ; %Flow283
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_9:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s4
	v_mbcnt_lo_u32_b32 v4, -1, 0
	v_lshrrev_b32_e32 v161, 5, v0
	v_lshl_add_u32 v160, v6, 3, 0
	v_ashrrev_i32_e32 v146, 31, v145
	v_lshrrev_b32_e32 v6, 1, v6
	v_xor_b32_e32 v7, 16, v4
	v_xor_b32_e32 v9, 8, v4
	v_xor_b32_e32 v13, 1, v4
	v_lshlrev_b64_e32 v[129:130], 2, v[145:146]
	v_cndmask_b32_e64 v2, 0, v2, s23
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	v_cndmask_b32_e64 v3, 0, v3, s23
	s_mov_b32 s4, ttmp7
	s_mov_b32 s17, 0
	s_ashr_i32 s5, ttmp7, 31
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v4, v7, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_mul_i32 s16, s7, 0x1800
	s_lshl_b64 s[4:5], s[4:5], 8
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[18:19], s[8:9], s[16:17]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v9, v4, v9 :: v_dual_lshlrev_b32 v158, 2, v7
	s_add_nc_u64 s[20:21], s[10:11], s[4:5]
	s_lshl_b32 s6, ttmp7, 1
	s_mov_b32 s16, s17
	ds_bpermute_b32 v7, v158, v1
	ds_bpermute_b32 v8, v158, v5
	v_dual_mov_b32 v162, 0 :: v_dual_lshlrev_b32 v9, 2, v9
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s7, s6, 31
	s_wait_dscnt 0x1
	v_max_i32_e32 v1, v1, v7
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v8
	v_lshrrev_b32_e32 v7, 1, v0
	v_lshlrev_b32_e32 v8, 3, v0
	ds_bpermute_b32 v11, v9, v1
	ds_bpermute_b32 v12, v9, v5
	v_mad_co_u64_u32 v[9:10], null, v145, 24, v[147:148]
	v_xor_b32_e32 v10, 4, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v0, v4, v10 :: v_dual_and_b32 v159, 15, v0
	v_mov_b32_e32 v185, 0
	v_dual_mov_b32 v167, 0xff800000 :: v_dual_lshlrev_b32 v14, 4, v159
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v0, 2, v0
	v_dual_mov_b32 v189, v185 :: v_dual_and_b32 v146, 8, v6
	s_wait_dscnt 0x1
	v_max_i32_e32 v1, v1, v11
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v12
	v_xor_b32_e32 v11, 2, v4
	v_mov_b32_e32 v192, v185
	v_dual_mov_b32 v187, v185 :: v_dual_and_b32 v164, 8, v7
	ds_bpermute_b32 v12, v0, v1
	ds_bpermute_b32 v0, v0, v5
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	v_lshlrev_b32_e32 v165, 4, v161
	v_dual_mov_b32 v186, v185 :: v_dual_and_b32 v7, 0xf8, v8
	v_dual_mov_b32 v188, v185 :: v_dual_mov_b32 v191, v185
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v4, v11, vcc_lo
	v_mov_b32_e32 v190, v185
	v_dual_mov_b32 v200, v192 :: v_dual_mov_b32 v195, v187
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v199, v191 :: v_dual_lshlrev_b32 v6, 2, v6
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	v_dual_mov_b32 v208, v192 :: v_dual_mov_b32 v203, v187
	v_dual_mov_b32 v204, v188 :: v_dual_mov_b32 v25, v185
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v4, v4, v13 :: v_dual_mov_b32 v197, v189
	s_wait_dscnt 0x1
	v_max_i32_e32 v1, v1, v12
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v5, v0
	v_dual_mov_b32 v30, v190 :: v_dual_mov_b32 v33, v185
	v_lshlrev_b32_e32 v4, 2, v4
	ds_bpermute_b32 v5, v6, v1
	ds_bpermute_b32 v6, v6, v0
	v_dual_mov_b32 v38, v190 :: v_dual_mov_b32 v41, v185
	v_dual_mov_b32 v46, v190 :: v_dual_mov_b32 v49, v185
	v_dual_mov_b32 v54, v190 :: v_dual_mov_b32 v57, v185
	v_dual_mov_b32 v62, v190 :: v_dual_mov_b32 v65, v185
	v_mov_b32_e32 v69, v189
	v_dual_mov_b32 v10, v185 :: v_dual_lshlrev_b32 v163, 7, v159
	v_add_co_u32 v2, vcc_lo, v2, v146
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[131:132], 2, v[9:10]
	v_add_co_u32 v150, vcc_lo, s0, v129
	s_wait_dscnt 0x1
	v_max_i32_e32 v1, v1, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v6
	v_dual_mov_b32 v68, v188 :: v_dual_mov_b32 v73, v185
	v_dual_mov_b32 v76, v188 :: v_dual_mov_b32 v81, v185
	ds_bpermute_b32 v5, v4, v1
	ds_bpermute_b32 v4, v4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v151, null, s1, v130, vcc_lo
	v_add_co_u32 v2, vcc_lo, s8, v2
	v_dual_mov_b32 v84, v188 :: v_dual_mov_b32 v89, v185
	v_lshl_or_b32 v7, v161, 8, v7
	v_dual_mov_b32 v92, v188 :: v_dual_mov_b32 v97, v185
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s9, v3, vcc_lo
	v_dual_mov_b32 v100, v188 :: v_dual_mov_b32 v105, v185
	v_add_co_u32 v152, vcc_lo, s18, v131
	v_dual_mov_b32 v108, v188 :: v_dual_mov_b32 v113, v185
	v_add_co_u32 v148, s3, s20, v14
	s_wait_dscnt 0x1
	v_max_i32_e32 v1, v1, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v4
	v_dual_mov_b32 v116, v188 :: v_dual_mov_b32 v121, v185
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, s19, v132, vcc_lo
	v_add_co_u32 v154, vcc_lo, v2, 48
	v_dual_mov_b32 v124, v188 :: v_dual_mov_b32 v129, v185
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v149, null, s21, 0, s3
	v_dual_mov_b32 v198, v190 :: v_dual_mov_b32 v193, v185
	v_dual_mov_b32 v196, v188 :: v_dual_mov_b32 v207, v191
	v_dual_mov_b32 v194, v186 :: v_dual_mov_b32 v205, v189
	v_dual_mov_b32 v206, v190 :: v_dual_mov_b32 v201, v185
	v_dual_mov_b32 v202, v186 :: v_dual_mov_b32 v27, v187
	v_dual_mov_b32 v26, v186 :: v_dual_mov_b32 v29, v189
	v_dual_mov_b32 v28, v188 :: v_dual_mov_b32 v31, v191
	v_dual_mov_b32 v32, v192 :: v_dual_mov_b32 v35, v187
	v_dual_mov_b32 v34, v186 :: v_dual_mov_b32 v37, v189
	v_dual_mov_b32 v36, v188 :: v_dual_mov_b32 v39, v191
	v_dual_mov_b32 v40, v192 :: v_dual_mov_b32 v43, v187
	v_dual_mov_b32 v42, v186 :: v_dual_mov_b32 v45, v189
	v_dual_mov_b32 v44, v188 :: v_dual_mov_b32 v47, v191
	v_dual_mov_b32 v48, v192 :: v_dual_mov_b32 v51, v187
	v_dual_mov_b32 v50, v186 :: v_dual_mov_b32 v53, v189
	v_dual_mov_b32 v52, v188 :: v_dual_mov_b32 v55, v191
	v_dual_mov_b32 v56, v192 :: v_dual_mov_b32 v59, v187
	v_dual_mov_b32 v58, v186 :: v_dual_mov_b32 v61, v189
	v_dual_mov_b32 v60, v188 :: v_dual_mov_b32 v63, v191
	v_dual_mov_b32 v64, v192 :: v_dual_mov_b32 v67, v187
	v_dual_mov_b32 v66, v186 :: v_dual_mov_b32 v71, v191
	v_dual_mov_b32 v70, v190 :: v_dual_mov_b32 v75, v187
	v_dual_mov_b32 v72, v192 :: v_dual_mov_b32 v77, v189
	v_dual_mov_b32 v74, v186 :: v_dual_mov_b32 v79, v191
	v_dual_mov_b32 v78, v190 :: v_dual_mov_b32 v83, v187
	v_dual_mov_b32 v80, v192 :: v_dual_mov_b32 v85, v189
	v_dual_mov_b32 v82, v186 :: v_dual_mov_b32 v87, v191
	v_dual_mov_b32 v86, v190 :: v_dual_mov_b32 v91, v187
	v_dual_mov_b32 v88, v192 :: v_dual_mov_b32 v93, v189
	v_dual_mov_b32 v90, v186 :: v_dual_mov_b32 v95, v191
	v_dual_mov_b32 v94, v190 :: v_dual_mov_b32 v99, v187
	v_dual_mov_b32 v96, v192 :: v_dual_mov_b32 v101, v189
	v_dual_mov_b32 v98, v186 :: v_dual_mov_b32 v103, v191
	v_dual_mov_b32 v102, v190 :: v_dual_mov_b32 v107, v187
	v_dual_mov_b32 v104, v192 :: v_dual_mov_b32 v109, v189
	v_dual_mov_b32 v106, v186 :: v_dual_mov_b32 v111, v191
	v_dual_mov_b32 v110, v190 :: v_dual_mov_b32 v115, v187
	v_dual_mov_b32 v112, v192 :: v_dual_mov_b32 v117, v189
	v_dual_mov_b32 v114, v186 :: v_dual_mov_b32 v119, v191
	v_dual_mov_b32 v118, v190 :: v_dual_mov_b32 v123, v187
	v_dual_mov_b32 v120, v192 :: v_dual_mov_b32 v125, v189
	v_dual_mov_b32 v122, v186 :: v_dual_mov_b32 v127, v191
	v_dual_mov_b32 v126, v190 :: v_dual_mov_b32 v131, v187
	v_dual_mov_b32 v128, v192 :: v_dual_mov_b32 v133, v189
	v_dual_mov_b32 v135, v191 :: v_dual_add_nc_u32 v166, 0, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v155, null, 0, v3, vcc_lo
	v_readfirstlane_b32 s24, v1
	v_readfirstlane_b32 s25, v0
	v_mov_b32_e32 v130, v186
	v_mov_b32_e32 v132, v188
	v_mov_b32_e32 v134, v190
	v_mov_b32_e32 v136, v192
	s_add_nc_u64 s[18:19], s[12:13], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[20:21], s[10:11], s[6:7]
	s_add_nc_u64 s[12:13], s[12:13], s[6:7]
	s_branch .LBB0_13
.LBB0_10:                               ; %Flow277
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b32_e32 v167, v7
.LBB0_11:                               ; %Flow278
                                        ;   in Loop: Header=BB0_13 Depth=1
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
	s_cbranch_vccz .LBB0_85
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
; %bb.14:                               ; %.preheader416.i.preheader
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b32_e32 v0, v156
	v_mov_b32_e32 v6, v157
	s_mov_b32 s0, 8
	s_branch .LBB0_16
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_and_b32_e32 v1, 0xe0, v6
	v_add_nc_u32_e32 v6, 16, v6
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	v_add_nc_u32_e32 v1, v1, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v1, v0, 15, v1
	v_add_nc_u32_e32 v0, 8, v0
	v_lshl_add_u32 v1, v1, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v1, v[2:3], v[4:5] offset1:16
	s_cbranch_scc1 .LBB0_18
.LBB0_16:                               ; %.preheader416.i
                                        ;   Parent Loop BB0_13 Depth=1
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
.LBB0_18:                               ; %.preheader415.i
                                        ;   in Loop: Header=BB0_13 Depth=1
	v_or_b32_e32 v6, s26, v164
	v_dual_mov_b32 v7, v165 :: v_dual_mov_b32 v8, v161
	s_movk_i32 s0, 0xc000
	s_branch .LBB0_22
.LBB0_19:                               ; %Flow279
                                        ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB0_20:                               ; %Flow280
                                        ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_21:                               ; %.loopexit413.i
                                        ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v0, s0, v166
	v_add_nc_u32_e32 v8, 4, v8
	v_add_nc_u32_e32 v7, 64, v7
	s_addk_co_i32 s0, 0x400
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	ds_store_b64 v0, v[2:3] offset:32768
	s_cbranch_scc1 .LBB0_38
.LBB0_22:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0x70, v8
	s_mov_b32 s1, exec_lo
                                        ; implicit-def: $vgpr2_vgpr3
	v_add_nc_u32_e32 v137, v0, v6
	v_and_or_b32 v0, 0xf0, v7, v159
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v1, 7, v137
	v_cmpx_ge_i32_e64 s24, v1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_22 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v137, s[18:19]
	v_mov_b16_e32 v4.h, 0
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v10.l, v4.h
	v_add_co_u32 v2, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_clause 0x7
	global_load_u8 v0, v[2:3], off offset:1032
	global_load_u8 v1, v[2:3], off offset:2064
	global_load_d16_u8 v4, v[2:3], off
	global_load_u8 v5, v[2:3], off offset:3096
	global_load_u8 v9, v[2:3], off offset:5160
	global_load_u8 v11, v[2:3], off offset:4128
	global_load_u8 v12, v[2:3], off offset:7224
	global_load_d16_hi_u8 v10, v[2:3], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v0, 8, v0
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v1, 16, v1
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v2, 24, v5
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v3, 8, v9
	v_or_b32_e32 v0, v0, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v0, v0, v1, v2
	s_wait_loadcnt 0x2
	v_or3_b32 v1, 0, v11, v3
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v2, 24, v12
	v_or3_b32 v0, v0, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v3, v1, v10, v2
	v_or3_b32 v2, v0, 0, 0
                                        ; implicit-def: $vgpr0
.LBB0_24:                               ; %Flow281
                                        ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_21
; %bb.25:                               ;   in Loop: Header=BB0_22 Depth=2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_ge_i32_e64 s24, v137
	s_cbranch_execz .LBB0_20
; %bb.26:                               ; %.preheader412.i
                                        ;   in Loop: Header=BB0_22 Depth=2
	v_add_co_u32 v4, s4, s18, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s19, 0, s4
	v_mov_b16_e64 v184.h, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[2:3], null, 0x408, v137, v[4:5]
	global_load_d16_u8 v184, v[2:3], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v2, v184 :: v_dual_mov_b32 v3, v185
	v_cmpx_gt_i32_e64 s24, v137
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_22 Depth=2
	v_or_b32_e32 v0, 1, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[2:3], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v184, v0, 8, v184
	v_dual_mov_b32 v2, v184 :: v_dual_mov_b32 v3, v185
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
	v_mov_b32_e32 v186, s22
	s_and_saveexec_b32 s0, s23
	s_cbranch_execz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_13 Depth=1
	global_load_b32 v0, v[152:153], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v186, s22, v0
.LBB0_41:                               ;   in Loop: Header=BB0_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_or_b32 s0, s26, 16
	v_mov_b32_e32 v6, v160
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s24
	s_mov_b32 s28, 0
	s_cselect_b32 s29, -1, 0
	s_branch .LBB0_44
.LBB0_42:                               ; %_ZL11fa2_scale_nPKhiii.exit381.7.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_sub_f32_e32 v0, v167, v7
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v167
	v_lshl_add_u32 v168, s28, 12, v160
	v_cvt_f16_f32_e64 v2.l, v144
	v_cvt_f16_f32_e64 v2.h, v174
	s_wait_dscnt 0x0
	v_dual_mul_f32 v0, 0x3fb8aa3b, v0 :: v_dual_add_f32 v167, v169, v170
	v_cvt_f16_f32_e64 v4.l, v177
	v_cvt_f16_f32_e64 v3.h, v176
	v_cvt_f16_f32_e64 v5.l, v179
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v0, v0
	v_cvt_f16_f32_e64 v3.l, v175
	v_cvt_f16_f32_e64 v4.h, v178
	v_cvt_f16_f32_e64 v5.h, v180
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v144, 0, v0, vcc_lo
	ds_load_b64 v[0:1], v168 offset:16384
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[9:10], v168 offset:16640
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v136, v136, v144 :: v_dual_mul_f32 v135, v135, v144
	v_dual_mul_f32 v134, v134, v144 :: v_dual_mul_f32 v131, v131, v144
	v_dual_mul_f32 v133, v133, v144 :: v_dual_mul_f32 v132, v132, v144
	v_dual_mul_f32 v129, v129, v144 :: v_dual_mul_f32 v130, v130, v144
	v_dual_mul_f32 v127, v127, v144 :: v_dual_mul_f32 v128, v128, v144
	v_dual_mul_f32 v125, v125, v144 :: v_dual_mul_f32 v126, v126, v144
	v_dual_mul_f32 v123, v123, v144 :: v_dual_mul_f32 v124, v124, v144
	v_dual_mul_f32 v121, v121, v144 :: v_dual_mul_f32 v122, v122, v144
	v_mul_f32_e32 v119, v119, v144
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[11:12], v0
	v_cvt_pk_f32_fp8_e64 v[13:14], v0 op_sel:[1,0]
	v_cvt_pk_f32_fp8_e32 v[15:16], v1
	v_cvt_pk_f32_fp8_e64 v[0:1], v1 op_sel:[1,0]
	v_dual_mul_f32 v120, v120, v144 :: v_dual_mul_f32 v117, v117, v144
	v_fma_mixlo_f16 v169, v8, v11, 0
	v_fma_mixhi_f16 v169, v138, v12, 0
	ds_load_b64 v[11:12], v168 offset:16896
	v_fma_mixlo_f16 v172, v141, v0, 0
	v_fma_mixhi_f16 v172, v143, v1, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e64 v[0:1], v9 op_sel:[1,0]
	;;#ASMSTART
	;;#ASMEND
	v_fma_mixlo_f16 v170, v137, v13, 0
	v_fma_mixhi_f16 v170, v140, v14, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v9
	v_fma_mixlo_f16 v171, v139, v15, 0
	v_fma_mixhi_f16 v171, v142, v16, 0
	v_cvt_pk_f32_fp8_e32 v[15:16], v10
	ds_load_b64 v[17:18], v168 offset:17152
	v_fma_mixlo_f16 v174, v137, v0, 0
	v_fma_mixhi_f16 v174, v140, v1, 0
	v_cvt_pk_f32_fp8_e64 v[0:1], v10 op_sel:[1,0]
	;;#ASMSTART
	;;#ASMEND
	v_fma_mixlo_f16 v175, v139, v15, 0
	v_fma_mixhi_f16 v175, v142, v16, 0
	v_fma_mixlo_f16 v173, v8, v13, 0
	v_fma_mixlo_f16 v176, v141, v0, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[9:10], v11
	v_fma_mixhi_f16 v176, v143, v1, 0
	v_cvt_pk_f32_fp8_e32 v[0:1], v12
	v_fma_mixhi_f16 v173, v138, v14, 0
	v_cvt_pk_f32_fp8_e64 v[13:14], v11 op_sel:[1,0]
	v_fma_mixlo_f16 v177, v8, v9, 0
	v_fma_mixhi_f16 v177, v138, v10, 0
	ds_load_b64 v[9:10], v168 offset:17408
	v_fma_mixlo_f16 v179, v139, v0, 0
	v_fma_mixhi_f16 v179, v142, v1, 0
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e64 v[0:1], v17 op_sel:[1,0]
	ds_load_b64 v[15:16], v168 offset:17664
	;;#ASMSTART
	;;#ASMEND
	v_cvt_pk_f32_fp8_e64 v[11:12], v12 op_sel:[1,0]
	v_wmma_f32_16x16x16_f16 v[129:136], v[169:172], v[2:5], v[129:136]
	v_fma_mixlo_f16 v182, v137, v0, 0
	v_fma_mixhi_f16 v182, v140, v1, 0
	v_fma_mixlo_f16 v178, v137, v13, 0
	v_fma_mixlo_f16 v180, v141, v11, 0
	v_fma_mixhi_f16 v180, v143, v12, 0
	v_cvt_pk_f32_fp8_e32 v[11:12], v18
	v_fma_mixhi_f16 v178, v140, v14, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v17
	v_dual_mul_f32 v118, v118, v144 :: v_dual_mul_f32 v115, v115, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_mixlo_f16 v183, v139, v11, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[0:1], v9
	v_fma_mixhi_f16 v183, v142, v12, 0
	v_cvt_pk_f32_fp8_e64 v[11:12], v9 op_sel:[1,0]
	v_fma_mixlo_f16 v181, v8, v13, 0
	v_fma_mixhi_f16 v181, v138, v14, 0
	v_fma_mixlo_f16 v169, v8, v0, 0
	v_fma_mixhi_f16 v169, v138, v1, 0
	ds_load_b64 v[0:1], v168 offset:17920
	v_cvt_pk_f32_fp8_e64 v[13:14], v18 op_sel:[1,0]
	v_fma_mixlo_f16 v170, v137, v11, 0
	v_fma_mixhi_f16 v170, v140, v12, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[11:12], v15
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v116, v116, v144 :: v_dual_mul_f32 v113, v113, v144
	v_dual_mul_f32 v114, v114, v144 :: v_dual_mul_f32 v111, v111, v144
	v_fma_mixlo_f16 v184, v141, v13, 0
	v_fma_mixhi_f16 v184, v143, v14, 0
	v_wmma_f32_16x16x16_f16 v[121:128], v[173:176], v[2:5], v[121:128]
	v_cvt_pk_f32_fp8_e32 v[13:14], v10
	v_cvt_pk_f32_fp8_e64 v[9:10], v10 op_sel:[1,0]
	v_fma_mixlo_f16 v173, v8, v11, 0
	ds_load_b64 v[17:18], v168 offset:18176
	v_fma_mixhi_f16 v173, v138, v12, 0
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_f16 v[113:120], v[177:180], v[2:5], v[113:120]
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[11:12], v0
	v_fma_mixlo_f16 v172, v141, v9, 0
	v_fma_mixhi_f16 v172, v143, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v15 op_sel:[1,0]
	v_fma_mixlo_f16 v171, v139, v13, 0
	v_fma_mixlo_f16 v177, v8, v11, 0
	v_fma_mixhi_f16 v177, v138, v12, 0
	ds_load_b64 v[11:12], v168 offset:18432
	v_fma_mixhi_f16 v171, v142, v14, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v16
	v_fma_mixlo_f16 v174, v137, v9, 0
	v_fma_mixhi_f16 v174, v140, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v16 op_sel:[1,0]
	v_dual_mul_f32 v112, v112, v144 :: v_dual_mul_f32 v109, v109, v144
	v_fma_mixlo_f16 v175, v139, v13, 0
	v_fma_mixhi_f16 v175, v142, v14, 0
	v_cvt_pk_f32_fp8_e64 v[13:14], v0 op_sel:[1,0]
	v_fma_mixlo_f16 v176, v141, v9, 0
	v_fma_mixhi_f16 v176, v143, v10, 0
	v_cvt_pk_f32_fp8_e32 v[9:10], v1
	v_cvt_pk_f32_fp8_e64 v[0:1], v1 op_sel:[1,0]
	v_dual_mul_f32 v110, v110, v144 :: v_dual_mul_f32 v107, v107, v144
	v_dual_mul_f32 v108, v108, v144 :: v_dual_mul_f32 v105, v105, v144
	v_dual_mul_f32 v106, v106, v144 :: v_dual_mul_f32 v103, v103, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_mixlo_f16 v180, v141, v0, 0
	v_fma_mixhi_f16 v180, v143, v1, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[0:1], v18
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v104, v104, v144 :: v_dual_mul_f32 v101, v101, v144
	v_dual_mul_f32 v102, v102, v144 :: v_dual_mul_f32 v99, v99, v144
	v_dual_mul_f32 v100, v100, v144 :: v_dual_mul_f32 v97, v97, v144
	v_dual_mul_f32 v98, v98, v144 :: v_dual_mul_f32 v95, v95, v144
	v_wmma_f32_16x16x16_f16 v[105:112], v[181:184], v[2:5], v[105:112]
	ds_load_b64 v[15:16], v168 offset:18688
	v_fma_mixlo_f16 v183, v139, v0, 0
	v_fma_mixhi_f16 v183, v142, v1, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[0:1], v11
	;;#ASMSTART
	;;#ASMEND
	v_fma_mixlo_f16 v178, v137, v13, 0
	v_fma_mixhi_f16 v178, v140, v14, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v17
	v_wmma_f32_16x16x16_f16 v[97:104], v[169:172], v[2:5], v[97:104]
	v_fma_mixlo_f16 v169, v8, v0, 0
	v_fma_mixhi_f16 v169, v138, v1, 0
	ds_load_b64 v[0:1], v168 offset:18944
	v_fma_mixlo_f16 v179, v139, v9, 0
	v_fma_mixhi_f16 v179, v142, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v17 op_sel:[1,0]
	v_fma_mixlo_f16 v181, v8, v13, 0
	v_fma_mixhi_f16 v181, v138, v14, 0
	v_cvt_pk_f32_fp8_e64 v[13:14], v18 op_sel:[1,0]
	;;#ASMSTART
	;;#ASMEND
	v_fma_mixlo_f16 v182, v137, v9, 0
	v_fma_mixhi_f16 v182, v140, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v11 op_sel:[1,0]
	v_fma_mixlo_f16 v184, v141, v13, 0
	v_fma_mixhi_f16 v184, v143, v14, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v12
	v_cvt_pk_f32_fp8_e64 v[11:12], v12 op_sel:[1,0]
	v_dual_mul_f32 v96, v96, v144 :: v_dual_mul_f32 v93, v93, v144
	v_dual_mul_f32 v94, v94, v144 :: v_dual_mul_f32 v91, v91, v144
	v_dual_mul_f32 v92, v92, v144 :: v_dual_mul_f32 v89, v89, v144
	v_dual_mul_f32 v90, v90, v144 :: v_dual_mul_f32 v87, v87, v144
	v_fma_mixlo_f16 v172, v141, v11, 0
	v_fma_mixhi_f16 v172, v143, v12, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e64 v[11:12], v15 op_sel:[1,0]
	ds_load_b64 v[17:18], v168 offset:19200
	v_dual_mul_f32 v88, v88, v144 :: v_dual_mul_f32 v85, v85, v144
	v_dual_mul_f32 v86, v86, v144 :: v_dual_mul_f32 v83, v83, v144
	v_dual_mul_f32 v84, v84, v144 :: v_dual_mul_f32 v81, v81, v144
	v_dual_mul_f32 v82, v82, v144 :: v_dual_mul_f32 v79, v79, v144
	v_wmma_f32_16x16x16_f16 v[89:96], v[173:176], v[2:5], v[89:96]
	v_fma_mixlo_f16 v170, v137, v9, 0
	v_fma_mixhi_f16 v170, v140, v10, 0
	v_cvt_pk_f32_fp8_e32 v[9:10], v15
	v_fma_mixlo_f16 v174, v137, v11, 0
	v_fma_mixhi_f16 v174, v140, v12, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[11:12], v0
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_f16 v[81:88], v[177:180], v[2:5], v[81:88]
	v_fma_mixlo_f16 v171, v139, v13, 0
	v_fma_mixhi_f16 v171, v142, v14, 0
	v_fma_mixlo_f16 v173, v8, v9, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v16
	v_fma_mixhi_f16 v173, v138, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v16 op_sel:[1,0]
	v_fma_mixlo_f16 v177, v8, v11, 0
	v_fma_mixhi_f16 v177, v138, v12, 0
	ds_load_b64 v[11:12], v168 offset:19456
	v_fma_mixlo_f16 v175, v139, v13, 0
	v_fma_mixhi_f16 v175, v142, v14, 0
	v_cvt_pk_f32_fp8_e64 v[13:14], v0 op_sel:[1,0]
	v_fma_mixlo_f16 v176, v141, v9, 0
	v_fma_mixhi_f16 v176, v143, v10, 0
	v_cvt_pk_f32_fp8_e32 v[9:10], v1
	v_cvt_pk_f32_fp8_e64 v[0:1], v1 op_sel:[1,0]
	v_dual_mul_f32 v80, v80, v144 :: v_dual_mul_f32 v77, v77, v144
	v_dual_mul_f32 v78, v78, v144 :: v_dual_mul_f32 v75, v75, v144
	v_dual_mul_f32 v76, v76, v144 :: v_dual_mul_f32 v73, v73, v144
	v_dual_mul_f32 v74, v74, v144 :: v_dual_mul_f32 v71, v71, v144
	v_fma_mixlo_f16 v180, v141, v0, 0
	v_fma_mixhi_f16 v180, v143, v1, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[0:1], v18
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_f16 v[73:80], v[181:184], v[2:5], v[73:80]
	v_fma_mixlo_f16 v178, v137, v13, 0
	v_fma_mixhi_f16 v178, v140, v14, 0
	v_fma_mixlo_f16 v179, v139, v9, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v17
	v_fma_mixhi_f16 v179, v142, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v17 op_sel:[1,0]
	v_fma_mixlo_f16 v183, v139, v0, 0
	v_fma_mixhi_f16 v183, v142, v1, 0
	ds_load_b64 v[0:1], v168 offset:19712
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v72, v72, v144 :: v_dual_mul_f32 v69, v69, v144
	v_dual_mul_f32 v70, v70, v144 :: v_dual_mul_f32 v67, v67, v144
	v_dual_mul_f32 v68, v68, v144 :: v_dual_mul_f32 v65, v65, v144
	v_dual_mul_f32 v66, v66, v144 :: v_dual_mul_f32 v63, v63, v144
	v_fma_mixlo_f16 v181, v8, v13, 0
	v_fma_mixhi_f16 v181, v138, v14, 0
	v_fma_mixlo_f16 v182, v137, v9, 0
	v_fma_mixhi_f16 v182, v140, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v18 op_sel:[1,0]
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e64 v[13:14], v11 op_sel:[1,0]
	ds_load_b64 v[17:18], v168 offset:19968
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_f16 v[65:72], v[169:172], v[2:5], v[65:72]
	v_fma_mixlo_f16 v184, v141, v9, 0
	v_fma_mixhi_f16 v184, v143, v10, 0
	v_cvt_pk_f32_fp8_e32 v[9:10], v11
	v_fma_mixlo_f16 v170, v137, v13, 0
	v_fma_mixhi_f16 v170, v140, v14, 0
	ds_load_b64 v[13:14], v168 offset:20224
	v_dual_mul_f32 v64, v64, v144 :: v_dual_mul_f32 v61, v61, v144
	v_fma_mixlo_f16 v169, v8, v9, 0
	v_fma_mixhi_f16 v169, v138, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v12 op_sel:[1,0]
	v_dual_mul_f32 v62, v62, v144 :: v_dual_mul_f32 v59, v59, v144
	v_dual_mul_f32 v60, v60, v144 :: v_dual_mul_f32 v57, v57, v144
	v_dual_mul_f32 v58, v58, v144 :: v_dual_mul_f32 v55, v55, v144
	v_cvt_pk_f32_fp8_e32 v[15:16], v12
	v_fma_mixlo_f16 v172, v141, v9, 0
	s_wait_dscnt 0x2
	v_cvt_pk_f32_fp8_e32 v[11:12], v0
	v_fma_mixhi_f16 v172, v143, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v0 op_sel:[1,0]
	v_dual_mul_f32 v56, v56, v144 :: v_dual_mul_f32 v53, v53, v144
	v_dual_mul_f32 v54, v54, v144 :: v_dual_mul_f32 v51, v51, v144
	v_dual_mul_f32 v52, v52, v144 :: v_dual_mul_f32 v49, v49, v144
	v_dual_mul_f32 v50, v50, v144 :: v_dual_mul_f32 v47, v47, v144
	v_wmma_f32_16x16x16_f16 v[57:64], v[173:176], v[2:5], v[57:64]
	v_fma_mixlo_f16 v171, v139, v15, 0
	v_fma_mixhi_f16 v171, v142, v16, 0
	v_fma_mixlo_f16 v173, v8, v11, 0
	v_cvt_pk_f32_fp8_e32 v[15:16], v1
	v_fma_mixhi_f16 v173, v138, v12, 0
	v_fma_mixlo_f16 v174, v137, v9, 0
	v_cvt_pk_f32_fp8_e64 v[0:1], v1 op_sel:[1,0]
	v_fma_mixhi_f16 v174, v140, v10, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[9:10], v17
	v_cvt_pk_f32_fp8_e64 v[11:12], v17 op_sel:[1,0]
	v_dual_mul_f32 v48, v48, v144 :: v_dual_mul_f32 v45, v45, v144
	v_dual_mul_f32 v46, v46, v144 :: v_dual_mul_f32 v43, v43, v144
	v_dual_mul_f32 v44, v44, v144 :: v_dual_mul_f32 v41, v41, v144
	v_dual_mul_f32 v42, v42, v144 :: v_dual_mul_f32 v39, v39, v144
	v_wmma_f32_16x16x16_f16 v[49:56], v[177:180], v[2:5], v[49:56]
	v_fma_mixlo_f16 v176, v141, v0, 0
	v_fma_mixhi_f16 v176, v143, v1, 0
	v_fma_mixlo_f16 v177, v8, v9, 0
	v_cvt_pk_f32_fp8_e32 v[0:1], v18
	v_fma_mixhi_f16 v177, v138, v10, 0
	v_fma_mixlo_f16 v178, v137, v11, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v18 op_sel:[1,0]
	v_fma_mixhi_f16 v178, v140, v12, 0
	s_wait_dscnt 0x0
	v_cvt_pk_f32_fp8_e32 v[11:12], v13
	v_wmma_f32_16x16x16_f16 v[41:48], v[181:184], v[2:5], v[41:48]
	v_fma_mixlo_f16 v179, v139, v0, 0
	v_fma_mixhi_f16 v179, v142, v1, 0
	v_fma_mixlo_f16 v180, v141, v9, 0
	v_fma_mixhi_f16 v180, v143, v10, 0
	v_cvt_pk_f32_fp8_e64 v[0:1], v13 op_sel:[1,0]
	v_fma_mixlo_f16 v181, v8, v11, 0
	v_cvt_pk_f32_fp8_e32 v[8:9], v14
	v_cvt_pk_f32_fp8_e64 v[10:11], v14 op_sel:[1,0]
	v_dual_mul_f32 v40, v40, v144 :: v_dual_mul_f32 v37, v37, v144
	v_dual_mul_f32 v38, v38, v144 :: v_dual_mul_f32 v35, v35, v144
	v_dual_mul_f32 v36, v36, v144 :: v_dual_mul_f32 v33, v33, v144
	v_dual_mul_f32 v34, v34, v144 :: v_dual_mul_f32 v31, v31, v144
	v_dual_mul_f32 v32, v32, v144 :: v_dual_mul_f32 v29, v29, v144
	v_dual_mul_f32 v30, v30, v144 :: v_dual_mul_f32 v27, v27, v144
	v_dual_mul_f32 v28, v28, v144 :: v_dual_mul_f32 v25, v25, v144
	v_dual_mul_f32 v26, v26, v144 :: v_dual_mul_f32 v207, v207, v144
	v_dual_mul_f32 v208, v208, v144 :: v_dual_mul_f32 v205, v205, v144
	v_dual_mul_f32 v206, v206, v144 :: v_dual_mul_f32 v203, v203, v144
	v_dual_mul_f32 v204, v204, v144 :: v_dual_mul_f32 v201, v201, v144
	v_dual_mul_f32 v202, v202, v144 :: v_dual_mul_f32 v199, v199, v144
	v_dual_mul_f32 v200, v200, v144 :: v_dual_mul_f32 v197, v197, v144
	v_dual_mul_f32 v198, v198, v144 :: v_dual_mul_f32 v195, v195, v144
	v_dual_mul_f32 v196, v196, v144 :: v_dual_mul_f32 v193, v193, v144
	v_mul_f32_e32 v194, v194, v144
	v_fma_mixlo_f16 v175, v139, v15, 0
	v_fma_mixhi_f16 v175, v142, v16, 0
	v_fma_mixhi_f16 v181, v138, v12, 0
	v_fma_mixlo_f16 v182, v137, v0, 0
	v_fma_mixhi_f16 v182, v140, v1, 0
	v_fma_mixlo_f16 v183, v139, v8, 0
	v_fma_mixhi_f16 v183, v142, v9, 0
	v_fma_mixlo_f16 v184, v141, v10, 0
	v_fma_mixhi_f16 v184, v143, v11, 0
	v_fmac_f32_e32 v167, v162, v144
	v_wmma_f32_16x16x16_f16 v[33:40], v[169:172], v[2:5], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[173:176], v[2:5], v[25:32]
	v_wmma_f32_16x16x16_f16 v[201:208], v[177:180], v[2:5], v[201:208]
	v_wmma_f32_16x16x16_f16 v[193:200], v[181:184], v[2:5], v[193:200]
	v_mov_b32_e32 v162, v167
	;;#ASMSTART
	;;#ASMEND
.LBB0_43:                               ; %.loopexit407.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v167, v7 :: v_dual_add_nc_u32 v6, 0x100, v6
	s_add_co_i32 s28, s28, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s28, 4
	s_cbranch_scc0 .LBB0_10
.LBB0_44:                               ; %NodeBlock
                                        ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_49 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s28, 1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB0_47
; %bb.45:                               ; %LeafBlock
                                        ;   in Loop: Header=BB0_44 Depth=2
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
.LBB0_47:                               ; %Flow275
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_81
; %bb.48:                               ; %.preheader411.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v2, v154
	v_mov_b32_e32 v3, v155
	s_movk_i32 s0, 0xc000
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_mov_b32_e32 v144, v137
.LBB0_49:                               ; %.preheader.i
                                        ;   Parent Loop BB0_13 Depth=1
                                        ;     Parent Loop BB0_44 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_clause 0x3
	global_load_b64 v[4:5], v[2:3], off offset:-48
	global_load_b64 v[7:8], v[2:3], off offset:-32
	global_load_b64 v[176:177], v[2:3], off offset:-16
	global_load_b64 v[178:179], v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v0, s0, v6
	v_add_co_u32 v2, vcc_lo, v2, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	ds_load_2addr_stride64_b64 v[168:171], v0 offset0:32 offset1:34
	ds_load_2addr_stride64_b64 v[172:175], v0 offset0:36 offset1:38
	s_addk_co_i32 s0, 0x1000
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[168:169], v[4:5], v[137:144]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[170:171], v[7:8], v[137:144]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[172:173], v[176:177], v[137:144]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[174:175], v[178:179], v[137:144]
	s_cbranch_scc1 .LBB0_49
; %bb.50:                               ;   in Loop: Header=BB0_44 Depth=2
	s_lshl4_add_u32 s0, s28, s26
	v_mov_b32_e32 v169, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 15
	v_mov_b32_e32 v7, 0
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
	global_load_b32 v7, v[150:151], off
.LBB0_52:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, s0, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e32 vcc_lo, s24, v2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[3:4], null, 0x408, v2, s[20:21]
	global_load_d16_b16 v3, v[3:4], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v169, v3.l
.LBB0_54:                               ; %_ZL11fa2_scale_nPKhiii.exit.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v3, 1, v2
	v_cmp_gt_i32_e64 s0, s24, v2
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v174, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[4:5], null, 0x408, v3, s[20:21]
	global_load_d16_b16 v4, v[4:5], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v4.l
.LBB0_56:                               ; %_ZL11fa2_scale_nPKhiii.exit.1.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v4, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s1, s24, v4
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[170:171], null, 0x408, v4, s[20:21]
	global_load_d16_b16 v5, v[170:171], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v170, v5.l
.LBB0_58:                               ; %_ZL11fa2_scale_nPKhiii.exit.2.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v5, 3, v2
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v177, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e64 s3, s24, v5
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[171:172], null, 0x408, v5, s[20:21]
	global_load_d16_b16 v8, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v177, v8.l
.LBB0_60:                               ; %_ZL11fa2_scale_nPKhiii.exit.3.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v168, 4, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s4, s24, v168
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[171:172], null, 0x408, v168, s[20:21]
	global_load_d16_b16 v8, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v175, v8.l
.LBB0_62:                               ; %_ZL11fa2_scale_nPKhiii.exit.4.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v171, 5, v2
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v178, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e64 s5, s24, v171
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[172:173], null, 0x408, v171, s[20:21]
	global_load_d16_b16 v8, v[172:173], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v178, v8.l
.LBB0_64:                               ; %_ZL11fa2_scale_nPKhiii.exit.5.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v172, 6, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s6, s24, v172
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[179:180], null, 0x408, v172, s[20:21]
	global_load_d16_b16 v8, v[179:180], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v176, v8.l
.LBB0_66:                               ; %_ZL11fa2_scale_nPKhiii.exit.6.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v173, 7, v2
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v179, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_ge_i32_e64 s7, s24, v173
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[179:180], null, 0x408, v173, s[20:21]
	global_load_d16_b16 v179, v[179:180], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v179, v179.l
.LBB0_68:                               ; %_ZL11fa2_scale_nPKhiii.exit.7.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mul_f32 v0, v186, v137 :: v_dual_mul_f32 v1, v186, v138
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s8, v2, v7
	v_cmp_lt_i32_e64 s9, v2, v7
	v_cmp_le_i32_e64 s10, v4, v7
	v_dual_mul_f32 v0, v0, v169 :: v_dual_mul_f32 v9, v186, v139
	s_or_b32 s8, s30, s8
	v_dual_mul_f32 v1, v1, v174 :: v_dual_mul_f32 v10, v186, v140
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s23, s8
	s_or_b32 s9, s30, s9
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v0, 0xff800000, v0, s8
	v_cmp_le_i32_e64 s8, v5, v7
	v_dual_mul_f32 v9, v9, v170 :: v_dual_mul_f32 v10, v10, v177
	s_or_b32 s10, s30, s10
	s_and_b32 s9, s23, s9
	v_dual_mul_f32 v11, v186, v141 :: v_dual_mul_f32 v12, v186, v142
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v1, 0xff800000, v1, s9
	s_and_b32 s9, s23, s10
	s_or_b32 s10, s30, s8
	v_cmp_le_i32_e64 s8, v168, v7
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v9, 0xff800000, v9, s9
	v_cmp_le_i32_e64 s9, v171, v7
	v_dual_mul_f32 v11, v11, v175 :: v_dual_mul_f32 v12, v12, v178
	s_or_b32 s8, s30, s8
	s_and_b32 s10, s23, s10
	s_or_b32 s9, s30, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s23, s8
	v_mul_f32_e32 v13, v186, v143
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v11, 0xff800000, v11, s8
	s_and_b32 s8, s23, s9
	v_cndmask_b32_e64 v10, 0xff800000, v10, s10
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v12, 0xff800000, v12, s8
	v_cmp_le_i32_e64 s8, v172, v7
	v_cmp_le_i32_e64 s9, v173, v7
	v_mul_f32_e32 v7, v186, v144
	v_max3_num_f32 v14, v0, 0xff800000, v1
	v_mul_f32_e32 v13, v13, v176
	s_or_b32 s8, s30, s8
	s_or_b32 s9, s30, s9
	v_mul_f32_e32 v7, v7, v179
	v_max3_num_f32 v14, v14, v9, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s23, s8
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v13, 0xff800000, v13, s8
	s_and_b32 s8, s23, s9
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v15, 0xff800000, v7, s8
	v_max3_num_f32 v7, v14, v11, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v7, v7, v13, v15
	ds_bpermute_b32 v14, v158, v7
	s_wait_dscnt 0x0
	v_max3_num_f32 v7, v167, v7, v14
	v_dual_sub_f32 v0, v0, v7 :: v_dual_sub_f32 v1, v1, v7
	v_sub_f32_e32 v10, v10, v7
	v_cmp_eq_f32_e64 s8, 0xff800000, v7
	v_dual_sub_f32 v9, v9, v7 :: v_dual_sub_f32 v12, v12, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v0, 0x3fb8aa3b, v0 :: v_dual_mul_f32 v1, 0x3fb8aa3b, v1
	v_dual_mul_f32 v10, 0x3fb8aa3b, v10 :: v_dual_mul_f32 v9, 0x3fb8aa3b, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v10, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v144, v0, 0, s8
	v_sub_f32_e32 v0, v13, v7
	v_exp_f32_e32 v1, v1
	v_cndmask_b32_e64 v176, v10, 0, s8
	v_sub_f32_e32 v10, v15, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v11, v11, v7 :: v_dual_mul_f32 v0, 0x3fb8aa3b, v0
	v_exp_f32_e32 v9, v9
	v_mul_f32_e32 v10, 0x3fb8aa3b, v10
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v174, v1, 0, s8
	v_mul_f32_e32 v1, 0x3fb8aa3b, v12
	v_exp_f32_e32 v0, v0
	v_cndmask_b32_e64 v175, v9, 0, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v1, v1
	v_cndmask_b32_e64 v179, v0, 0, s8
	v_add_f32_e32 v9, v144, v174
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v178, v1, 0, s8
	v_mul_f32_e32 v11, 0x3fb8aa3b, v11
	v_add_f32_e32 v9, v175, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v11, v11
	v_add_f32_e32 v9, v176, v9
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v177, v11, 0, s8
	v_add_f32_e32 v1, v177, v9
	v_exp_f32_e32 v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v178, v1
	v_add_f32_e32 v0, v179, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v180, v9, 0, s8
	v_add_f32_e32 v169, v180, v0
	ds_bpermute_b32 v170, v158, v169
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[137:138], null, 0x408, v2, s[12:13]
	global_load_d16_b16 v2, v[137:138], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v8, v2.l
.LBB0_70:                               ; %_ZL11fa2_scale_nPKhiii.exit381.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v138, 0
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v3, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v138, v2.l
.LBB0_72:                               ; %_ZL11fa2_scale_nPKhiii.exit381.1.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v4, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v137, v2.l
.LBB0_74:                               ; %_ZL11fa2_scale_nPKhiii.exit381.2.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v140, 0
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v5, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v140, v2.l
.LBB0_76:                               ; %_ZL11fa2_scale_nPKhiii.exit381.3.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v168, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v139, v2.l
.LBB0_78:                               ; %_ZL11fa2_scale_nPKhiii.exit381.4.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v142, 0
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB0_82
; %bb.79:                               ; %_ZL11fa2_scale_nPKhiii.exit381.5.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execnz .LBB0_83
.LBB0_80:                               ; %_ZL11fa2_scale_nPKhiii.exit381.6.i
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v143, 0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB0_42
	s_branch .LBB0_84
.LBB0_81:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mov_b32_e32 v7, v167
	s_branch .LBB0_43
.LBB0_82:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v171, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v142, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execz .LBB0_80
.LBB0_83:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v172, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v141, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v143, 0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB0_42
.LBB0_84:                               ;   in Loop: Header=BB0_44 Depth=2
	v_mad_co_u64_u32 v[2:3], null, 0x408, v173, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v143, v2.l
	s_branch .LBB0_42
.LBB0_85:
	s_and_saveexec_b32 s0, s23
	s_cbranch_execz .LBB0_87
; %bb.86:                               ; %.loopexit.loopexit.i
	v_div_scale_f32 v0, null, v162, v162, 1.0
	v_div_scale_f32 v3, vcc_lo, 1.0, v162, 1.0
	v_mul_lo_u32 v5, 0x1800, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v2, v0
	v_fma_f32 v1, -v0, v2, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v1, v2
	v_mul_f32_e32 v4, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v1, -v0, v4, v3
	v_dual_fmac_f32 v4, v1, v2 :: v_dual_mov_b32 v1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f32 v0, -v0, v4, v3
	v_lshl_add_u32 v3, v147, 8, v5
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v2, v0, v2, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or_b32_e32 v0, v3, v146
	v_cmp_lt_f32_e32 vcc_lo, 0, v162
	v_div_fixup_f32 v2, v2, v162, 1.0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v137, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, s15, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v0, v129, v15 :: v_dual_mul_f32 v3, v132, v15
	v_dual_mul_f32 v1, v130, v15 :: v_dual_mul_f32 v2, v131, v15
	v_dual_mul_f32 v5, v134, v15 :: v_dual_mul_f32 v4, v133, v15
	v_dual_mul_f32 v7, v136, v15 :: v_dual_mul_f32 v6, v135, v15
	v_dual_mul_f32 v121, v121, v15 :: v_dual_mul_f32 v122, v122, v15
	v_dual_mul_f32 v123, v123, v15 :: v_dual_mul_f32 v124, v124, v15
	v_dual_mul_f32 v125, v125, v15 :: v_dual_mul_f32 v126, v126, v15
	v_dual_mul_f32 v127, v127, v15 :: v_dual_mul_f32 v128, v128, v15
	v_dual_mul_f32 v113, v113, v15 :: v_dual_mul_f32 v114, v114, v15
	v_dual_mul_f32 v115, v115, v15 :: v_dual_mul_f32 v116, v116, v15
	v_dual_mul_f32 v117, v117, v15 :: v_dual_mul_f32 v112, v112, v15
	v_dual_mul_f32 v97, v97, v15 :: v_dual_mul_f32 v98, v98, v15
	v_dual_mul_f32 v99, v99, v15 :: v_dual_mul_f32 v100, v100, v15
	v_dual_mul_f32 v118, v118, v15 :: v_dual_mul_f32 v119, v119, v15
	v_dual_mul_f32 v120, v120, v15 :: v_dual_mul_f32 v105, v105, v15
	v_dual_mul_f32 v106, v106, v15 :: v_dual_mul_f32 v107, v107, v15
	v_dual_mul_f32 v108, v108, v15 :: v_dual_mul_f32 v109, v109, v15
	v_dual_mul_f32 v110, v110, v15 :: v_dual_mul_f32 v111, v111, v15
	s_clause 0x7
	global_store_b128 v[137:138], v[0:3], off
	global_store_b128 v[137:138], v[4:7], off offset:16
	global_store_b128 v[137:138], v[121:124], off offset:64
	global_store_b128 v[137:138], v[125:128], off offset:80
	global_store_b128 v[137:138], v[113:116], off offset:128
	global_store_b128 v[137:138], v[117:120], off offset:144
	global_store_b128 v[137:138], v[105:108], off offset:192
	global_store_b128 v[137:138], v[109:112], off offset:208
	v_dual_mul_f32 v0, v101, v15 :: v_dual_mul_f32 v3, v104, v15
	v_dual_mul_f32 v1, v102, v15 :: v_dual_mul_f32 v2, v103, v15
	v_dual_mul_f32 v5, v90, v15 :: v_dual_mul_f32 v4, v89, v15
	v_dual_mul_f32 v7, v92, v15 :: v_dual_mul_f32 v6, v91, v15
	v_dual_mul_f32 v89, v93, v15 :: v_dual_mul_f32 v90, v94, v15
	v_dual_mul_f32 v91, v95, v15 :: v_dual_mul_f32 v92, v96, v15
	v_dual_mul_f32 v81, v81, v15 :: v_dual_mul_f32 v82, v82, v15
	v_dual_mul_f32 v83, v83, v15 :: v_dual_mul_f32 v84, v84, v15
	v_dual_mul_f32 v85, v85, v15 :: v_dual_mul_f32 v86, v86, v15
	v_dual_mul_f32 v87, v87, v15 :: v_dual_mul_f32 v88, v88, v15
	s_clause 0x5
	global_store_b128 v[137:138], v[97:100], off offset:256
	global_store_b128 v[137:138], v[0:3], off offset:272
	global_store_b128 v[137:138], v[4:7], off offset:320
	global_store_b128 v[137:138], v[89:92], off offset:336
	global_store_b128 v[137:138], v[81:84], off offset:384
	global_store_b128 v[137:138], v[85:88], off offset:400
	v_dual_mul_f32 v0, v73, v15 :: v_dual_mul_f32 v3, v76, v15
	v_dual_mul_f32 v1, v74, v15 :: v_dual_mul_f32 v2, v75, v15
	v_dual_mul_f32 v5, v78, v15 :: v_dual_mul_f32 v4, v77, v15
	v_dual_mul_f32 v7, v80, v15 :: v_dual_mul_f32 v6, v79, v15
	v_dual_mul_f32 v65, v65, v15 :: v_dual_mul_f32 v66, v66, v15
	v_dual_mul_f32 v67, v67, v15 :: v_dual_mul_f32 v68, v68, v15
	v_dual_mul_f32 v69, v69, v15 :: v_dual_mul_f32 v70, v70, v15
	v_dual_mul_f32 v71, v71, v15 :: v_dual_mul_f32 v72, v72, v15
	v_dual_mul_f32 v57, v57, v15 :: v_dual_mul_f32 v58, v58, v15
	v_dual_mul_f32 v59, v59, v15 :: v_dual_mul_f32 v60, v60, v15
	v_dual_mul_f32 v61, v61, v15 :: v_dual_mul_f32 v62, v62, v15
	v_dual_mul_f32 v63, v63, v15 :: v_dual_mul_f32 v64, v64, v15
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:448
	global_store_b128 v[137:138], v[4:7], off offset:464
	global_store_b128 v[137:138], v[65:68], off offset:512
	global_store_b128 v[137:138], v[69:72], off offset:528
	global_store_b128 v[137:138], v[57:60], off offset:576
	global_store_b128 v[137:138], v[61:64], off offset:592
	v_dual_mul_f32 v0, v49, v15 :: v_dual_mul_f32 v3, v52, v15
	v_dual_mul_f32 v1, v50, v15 :: v_dual_mul_f32 v2, v51, v15
	v_dual_mul_f32 v5, v54, v15 :: v_dual_mul_f32 v4, v53, v15
	v_dual_mul_f32 v7, v56, v15 :: v_dual_mul_f32 v6, v55, v15
	v_dual_mul_f32 v41, v41, v15 :: v_dual_mul_f32 v42, v42, v15
	v_dual_mul_f32 v43, v43, v15 :: v_dual_mul_f32 v44, v44, v15
	v_dual_mul_f32 v45, v45, v15 :: v_dual_mul_f32 v46, v46, v15
	v_dual_mul_f32 v47, v47, v15 :: v_dual_mul_f32 v48, v48, v15
	v_dual_mul_f32 v33, v33, v15 :: v_dual_mul_f32 v34, v34, v15
	v_dual_mul_f32 v35, v35, v15 :: v_dual_mul_f32 v36, v36, v15
	v_dual_mul_f32 v37, v37, v15 :: v_dual_mul_f32 v38, v38, v15
	v_dual_mul_f32 v39, v39, v15 :: v_dual_mul_f32 v40, v40, v15
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:640
	global_store_b128 v[137:138], v[4:7], off offset:656
	global_store_b128 v[137:138], v[41:44], off offset:704
	global_store_b128 v[137:138], v[45:48], off offset:720
	global_store_b128 v[137:138], v[33:36], off offset:768
	global_store_b128 v[137:138], v[37:40], off offset:784
	v_dual_mul_f32 v0, v25, v15 :: v_dual_mul_f32 v3, v28, v15
	v_dual_mul_f32 v1, v26, v15 :: v_dual_mul_f32 v2, v27, v15
	v_dual_mul_f32 v5, v30, v15 :: v_dual_mul_f32 v4, v29, v15
	v_dual_mul_f32 v7, v32, v15 :: v_dual_mul_f32 v6, v31, v15
	v_dual_mul_f32 v17, v201, v15 :: v_dual_mul_f32 v18, v202, v15
	v_dual_mul_f32 v19, v203, v15 :: v_dual_mul_f32 v20, v204, v15
	v_dual_mul_f32 v21, v205, v15 :: v_dual_mul_f32 v22, v206, v15
	v_dual_mul_f32 v23, v207, v15 :: v_dual_mul_f32 v24, v208, v15
	v_dual_mul_f32 v9, v194, v15 :: v_dual_mul_f32 v8, v193, v15
	v_dual_mul_f32 v11, v196, v15 :: v_dual_mul_f32 v10, v195, v15
	v_dual_mul_f32 v13, v198, v15 :: v_dual_mul_f32 v12, v197, v15
	v_dual_mul_f32 v14, v199, v15 :: v_dual_mul_f32 v15, v200, v15
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:832
	global_store_b128 v[137:138], v[4:7], off offset:848
	global_store_b128 v[137:138], v[17:20], off offset:896
	global_store_b128 v[137:138], v[21:24], off offset:912
	global_store_b128 v[137:138], v[8:11], off offset:960
	global_store_b128 v[137:138], v[12:15], off offset:976
.LBB0_87:                               ; %_Z16fa2_stageb_nbodyILb0EEvPKhS1_S1_PfPKiifiiiiii.exit
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
		.amdhsa_next_free_vgpr 209
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
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.num_vgpr, 209
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
; codeLenInByte = 8416
; TotalNumSgprs: 33
; NumVgprs: 209
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 26
; NumSGPRsForWavesPerEU: 33
; NumVGPRsForWavesPerEU: 209
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
	s_cbranch_vccnz .LBB3_91
; %bb.1:
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB3_91
; %bb.2:
	s_lshl_b32 s4, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s4, s7
	s_cbranch_scc1 .LBB3_91
; %bb.3:
	s_lshr_b32 s18, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s18, s17
	s_cbranch_scc1 .LBB3_91
; %bb.4:
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_lshrrev_b32_e32 v157, 4, v0
	v_lshrrev_b32_e32 v158, 3, v0
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
	v_dual_mov_b32 v7, -1 :: v_dual_and_b32 v8, 31, v0
	v_bfrev_b32_e32 v9, -2
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_u32_e32 8, v8
	s_cbranch_execz .LBB3_10
; %bb.7:
	v_or_b32_e32 v6, s4, v8
	v_bfrev_b32_e32 v9, -2
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
	v_mov_b32_e32 v9, v7
.LBB3_9:                                ; %Flow295
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB3_10:                               ; %.lr.ph.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_lshrrev_b32_e32 v12, 1, v0
	v_lshrrev_b32_e32 v163, 5, v0
	v_lshrrev_b32_e32 v160, 4, v8
	v_lshl_add_u32 v162, v8, 3, 0
	v_xor_b32_e32 v6, 16, v1
	v_xor_b32_e32 v11, 8, v1
	v_cndmask_b32_e64 v8, 0, v3, s19
	v_ashrrev_i32_e32 v3, 31, v2
	s_cvt_f32_u32 s4, s17
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_add_co_i32 s6, s17, 0x1ff
	v_dual_mov_b32 v147, 0xff800000 :: v_dual_lshlrev_b32 v164, 3, v160
	v_lshlrev_b64_e32 v[129:130], 2, v[2:3]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v6, v1, v6 :: v_dual_and_b32 v161, 15, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	v_mov_b32_e32 v185, 0
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s20, s4
	v_dual_mov_b32 v148, 0 :: v_dual_lshlrev_b32 v159, 2, v6
	v_lshlrev_b32_e32 v14, 4, v161
	v_lshlrev_b32_e32 v13, 3, v0
	v_dual_mov_b32 v191, v185 :: v_dual_mov_b32 v192, v185
	ds_bpermute_b32 v6, v159, v7
	ds_bpermute_b32 v10, v159, v9
	v_dual_mov_b32 v188, v185 :: v_dual_lshlrev_b32 v165, 7, v161
	v_dual_mov_b32 v190, v185 :: v_dual_lshlrev_b32 v167, 4, v163
	s_and_b32 s6, s6, 0xffff
	v_cndmask_b32_e64 v4, 0, v4, s19
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s6, s6
	s_mov_b32 s5, 0
	v_dual_mov_b32 v187, v185 :: v_dual_and_b32 v166, 8, v12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s20, s6, s20
	v_dual_mov_b32 v186, v185 :: v_dual_mov_b32 v189, v185
	v_dual_mov_b32 v33, v185 :: v_dual_mov_b32 v36, v188
	s_delay_alu instid0(SALU_CYCLE_1)
	s_trunc_f32 s20, s20
	v_dual_mov_b32 v25, v185 :: v_dual_mov_b32 v28, v188
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[145:146], null, v2, 24, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v1, v11, vcc_lo
	v_max_i32_e32 v6, v7, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v7, v9, v10
	v_xor_b32_e32 v9, 4, v1
	v_xor_b32_e32 v10, 2, v1
	v_dual_mov_b32 v146, v185 :: v_dual_lshlrev_b32 v5, 2, v5
	v_and_b32_e32 v12, 0xf8, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	v_xor_b32_e32 v11, 1, v1
	ds_bpermute_b32 v0, v5, v6
	ds_bpermute_b32 v5, v5, v7
	v_lshlrev_b64_e32 v[131:132], 2, v[145:146]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v1, v9, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	v_dual_mov_b32 v41, v185 :: v_dual_mov_b32 v42, v186
	v_dual_mov_b32 v49, v185 :: v_dual_mov_b32 v50, v186
	v_dual_mov_b32 v57, v185 :: v_dual_mov_b32 v58, v186
	v_dual_mov_b32 v65, v185 :: v_dual_mov_b32 v66, v186
	v_dual_mov_b32 v73, v185 :: v_dual_mov_b32 v74, v186
	v_dual_mov_b32 v81, v185 :: v_dual_mov_b32 v82, v186
	v_dual_mov_b32 v89, v185 :: v_dual_mov_b32 v90, v186
	v_dual_mov_b32 v97, v185 :: v_dual_mov_b32 v98, v186
	s_wait_dscnt 0x1
	v_max_i32_e32 v0, v6, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v1, v10, vcc_lo
	v_lshlrev_b32_e32 v2, 2, v9
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v7, v5
	v_mov_b32_e32 v39, v191
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	v_lshlrev_b32_e32 v6, 2, v6
	ds_bpermute_b32 v5, v2, v0
	ds_bpermute_b32 v2, v2, v3
	v_lshl_or_b32 v7, v163, 8, v12
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, v1, v11 :: v_dual_mov_b32 v200, v192
	v_add_co_u32 v8, vcc_lo, v8, v164
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v149, vcc_lo, s0, v129
	s_xor_b32 s0, s20, 0x80000000
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v150, null, s1, v130, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s6, s0, s4
	s_cvt_u32_f32 s0, s20
	v_lshlrev_b32_e32 v1, 2, v1
	v_add_nc_u32_e32 v146, 0, v7
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s6, 31
	v_dual_mov_b32 v193, v185 :: v_dual_mov_b32 v208, v192
	s_wait_dscnt 0x1
	v_max_i32_e32 v0, v0, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v3, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s6, s4
	v_dual_mov_b32 v105, v185 :: v_dual_mov_b32 v106, v186
	ds_bpermute_b32 v3, v6, v0
	ds_bpermute_b32 v5, v6, v2
	s_add_co_ci_u32 s0, s0, 0
	s_lshl_b32 s4, s3, 8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s22, s0, 0xffff
	s_add_nc_u64 s[0:1], s[10:11], s[4:5]
	s_add_nc_u64 s[20:21], s[12:13], s[4:5]
	s_mul_i32 s4, s7, 0x1800
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v153, s0, s0, v14
	s_add_nc_u64 s[6:7], s[8:9], s[4:5]
	v_dual_mov_b32 v113, v185 :: v_dual_mov_b32 v114, v186
	v_dual_mov_b32 v121, v185 :: v_dual_mov_b32 v122, v186
	v_dual_mov_b32 v199, v191 :: v_dual_mov_b32 v198, v190
	v_dual_mov_b32 v197, v189 :: v_dual_mov_b32 v196, v188
	v_dual_mov_b32 v195, v187 :: v_dual_mov_b32 v194, v186
	s_wait_dscnt 0x1
	v_max_i32_e32 v0, v0, v3
	v_add_co_u32 v3, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s9, v4, vcc_lo
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v2, v5
	v_add_co_u32 v151, vcc_lo, v3, 48
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v152, null, 0, v4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v155, vcc_lo, s6, v131
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v156, null, s7, v132, vcc_lo
	v_dual_mov_b32 v129, v185 :: v_dual_mov_b32 v130, v186
	ds_bpermute_b32 v5, v1, v0
	ds_bpermute_b32 v1, v1, v2
	v_dual_mov_b32 v207, v191 :: v_dual_mov_b32 v206, v190
	v_dual_mov_b32 v205, v189 :: v_dual_mov_b32 v204, v188
	v_dual_mov_b32 v203, v187 :: v_dual_mov_b32 v202, v186
	v_dual_mov_b32 v201, v185 :: v_dual_mov_b32 v26, v186
	v_dual_mov_b32 v27, v187 :: v_dual_mov_b32 v30, v190
	v_dual_mov_b32 v29, v189 :: v_dual_mov_b32 v32, v192
	v_dual_mov_b32 v31, v191 :: v_dual_mov_b32 v34, v186
	v_dual_mov_b32 v35, v187 :: v_dual_mov_b32 v38, v190
	v_dual_mov_b32 v37, v189 :: v_dual_mov_b32 v40, v192
	s_wait_dscnt 0x1
	v_max_i32_e32 v0, v0, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v1, v2, v1
	v_dual_mov_b32 v43, v187 :: v_dual_mov_b32 v44, v188
	v_dual_mov_b32 v45, v189 :: v_dual_mov_b32 v46, v190
	v_dual_mov_b32 v47, v191 :: v_dual_mov_b32 v48, v192
	v_dual_mov_b32 v51, v187 :: v_dual_mov_b32 v52, v188
	v_dual_mov_b32 v53, v189 :: v_dual_mov_b32 v54, v190
	v_dual_mov_b32 v55, v191 :: v_dual_mov_b32 v56, v192
	v_dual_mov_b32 v59, v187 :: v_dual_mov_b32 v60, v188
	v_dual_mov_b32 v61, v189 :: v_dual_mov_b32 v62, v190
	v_dual_mov_b32 v63, v191 :: v_dual_mov_b32 v64, v192
	v_dual_mov_b32 v67, v187 :: v_dual_mov_b32 v68, v188
	v_dual_mov_b32 v69, v189 :: v_dual_mov_b32 v70, v190
	v_dual_mov_b32 v71, v191 :: v_dual_mov_b32 v72, v192
	v_dual_mov_b32 v75, v187 :: v_dual_mov_b32 v76, v188
	v_dual_mov_b32 v77, v189 :: v_dual_mov_b32 v78, v190
	v_dual_mov_b32 v79, v191 :: v_dual_mov_b32 v80, v192
	v_dual_mov_b32 v83, v187 :: v_dual_mov_b32 v84, v188
	v_dual_mov_b32 v85, v189 :: v_dual_mov_b32 v86, v190
	v_dual_mov_b32 v87, v191 :: v_dual_mov_b32 v88, v192
	v_dual_mov_b32 v91, v187 :: v_dual_mov_b32 v92, v188
	v_dual_mov_b32 v93, v189 :: v_dual_mov_b32 v94, v190
	v_dual_mov_b32 v95, v191 :: v_dual_mov_b32 v96, v192
	v_dual_mov_b32 v99, v187 :: v_dual_mov_b32 v100, v188
	v_dual_mov_b32 v101, v189 :: v_dual_mov_b32 v102, v190
	v_dual_mov_b32 v103, v191 :: v_dual_mov_b32 v104, v192
	v_dual_mov_b32 v107, v187 :: v_dual_mov_b32 v108, v188
	v_dual_mov_b32 v109, v189 :: v_dual_mov_b32 v110, v190
	v_dual_mov_b32 v111, v191 :: v_dual_mov_b32 v112, v192
	v_dual_mov_b32 v115, v187 :: v_dual_mov_b32 v116, v188
	v_dual_mov_b32 v117, v189 :: v_dual_mov_b32 v118, v190
	v_dual_mov_b32 v119, v191 :: v_dual_mov_b32 v120, v192
	v_dual_mov_b32 v123, v187 :: v_dual_mov_b32 v124, v188
	v_dual_mov_b32 v125, v189 :: v_dual_mov_b32 v126, v190
	v_dual_mov_b32 v127, v191 :: v_dual_mov_b32 v128, v192
	v_add_co_ci_u32_e64 v154, null, s1, 0, s0
	v_readfirstlane_b32 s26, v0
	v_readfirstlane_b32 s27, v1
	v_dual_mov_b32 v131, v187 :: v_dual_mov_b32 v132, v188
	v_dual_mov_b32 v133, v189 :: v_dual_mov_b32 v134, v190
	v_dual_mov_b32 v135, v191 :: v_dual_mov_b32 v136, v192
	s_mul_i32 s24, s18, s22
	s_lshl_b32 s4, s3, 1
	s_add_co_i32 s25, s24, s22
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[10:11], s[4:5]
	s_add_nc_u64 s[12:13], s[12:13], s[4:5]
	s_lshl_b32 s11, s24, 6
	s_branch .LBB3_14
.LBB3_11:                               ; %Flow289
                                        ;   in Loop: Header=BB3_14 Depth=1
	v_mov_b32_e32 v147, v7
.LBB3_12:                               ; %Flow290
                                        ;   in Loop: Header=BB3_14 Depth=1
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
	s_cbranch_vccz .LBB3_86
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
; %bb.15:                               ; %.preheader430.i.preheader
                                        ;   in Loop: Header=BB3_14 Depth=1
	v_mov_b32_e32 v0, v157
	v_mov_b32_e32 v6, v158
	s_mov_b32 s0, 8
	s_branch .LBB3_17
.LBB3_16:                               ;   in Loop: Header=BB3_17 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_and_b32_e32 v1, 0xe0, v6
	v_add_nc_u32_e32 v6, 16, v6
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	v_add_nc_u32_e32 v1, v1, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v1, v0, 15, v1
	v_add_nc_u32_e32 v0, 8, v0
	v_lshl_add_u32 v1, v1, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v1, v[2:3], v[4:5] offset1:16
	s_cbranch_scc1 .LBB3_19
.LBB3_17:                               ; %.preheader430.i
                                        ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v7, s11, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v5, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_ge_i32_e64 s26, v7
	s_cbranch_execz .LBB3_16
; %bb.18:                               ;   in Loop: Header=BB3_17 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v7, v[153:154]
	global_load_b128 v[2:5], v[2:3], off
	s_branch .LBB3_16
.LBB3_19:                               ; %.preheader429.i
                                        ;   in Loop: Header=BB3_14 Depth=1
	v_or_b32_e32 v6, s29, v166
	v_dual_mov_b32 v7, v167 :: v_dual_mov_b32 v8, v163
	s_movk_i32 s0, 0xc000
	s_branch .LBB3_23
.LBB3_20:                               ; %Flow291
                                        ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB3_21:                               ; %Flow292
                                        ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB3_22:                               ; %.loopexit427.i
                                        ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v0, s0, v146
	v_add_nc_u32_e32 v8, 4, v8
	v_add_nc_u32_e32 v7, 64, v7
	s_addk_co_i32 s0, 0x400
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	ds_store_b64 v0, v[2:3] offset:32768
	s_cbranch_scc1 .LBB3_39
.LBB3_23:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0x70, v8
	s_mov_b32 s1, exec_lo
                                        ; implicit-def: $vgpr2_vgpr3
	v_add_nc_u32_e32 v137, v0, v6
	v_and_or_b32 v0, 0xf0, v7, v161
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v1, 7, v137
	v_cmpx_ge_i32_e64 s26, v1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB3_25
; %bb.24:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v137, s[20:21]
	v_mov_b16_e32 v4.h, 0
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v10.l, v4.h
	v_add_co_u32 v2, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_clause 0x7
	global_load_u8 v0, v[2:3], off offset:1032
	global_load_u8 v1, v[2:3], off offset:2064
	global_load_d16_u8 v4, v[2:3], off
	global_load_u8 v5, v[2:3], off offset:3096
	global_load_u8 v9, v[2:3], off offset:5160
	global_load_u8 v11, v[2:3], off offset:4128
	global_load_u8 v12, v[2:3], off offset:7224
	global_load_d16_hi_u8 v10, v[2:3], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v0, 8, v0
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v1, 16, v1
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v2, 24, v5
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v3, 8, v9
	v_or_b32_e32 v0, v0, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v0, v0, v1, v2
	s_wait_loadcnt 0x2
	v_or3_b32 v1, 0, v11, v3
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v2, 24, v12
	v_or3_b32 v0, v0, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v3, v1, v10, v2
	v_or3_b32 v2, v0, 0, 0
                                        ; implicit-def: $vgpr0
.LBB3_25:                               ; %Flow293
                                        ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB3_22
; %bb.26:                               ;   in Loop: Header=BB3_23 Depth=2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_ge_i32_e64 s26, v137
	s_cbranch_execz .LBB3_21
; %bb.27:                               ; %.preheader426.i
                                        ;   in Loop: Header=BB3_23 Depth=2
	v_add_co_u32 v4, s4, s20, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s21, 0, s4
	v_mov_b16_e64 v184.h, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, 0x408, v137, v[4:5]
	global_load_d16_u8 v184, v[2:3], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v2, v184 :: v_dual_mov_b32 v3, v185
	v_cmpx_gt_i32_e64 s26, v137
	s_cbranch_execz .LBB3_29
; %bb.28:                               ;   in Loop: Header=BB3_23 Depth=2
	v_or_b32_e32 v0, 1, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[2:3], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[2:3], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v184, v0, 8, v184
	v_dual_mov_b32 v2, v184 :: v_dual_mov_b32 v3, v185
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
	v_mov_b32_e32 v186, s16
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB3_42
; %bb.41:                               ;   in Loop: Header=BB3_14 Depth=1
	global_load_b32 v0, v[155:156], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v186, s16, v0
.LBB3_42:                               ;   in Loop: Header=BB3_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_or_b32 s0, s29, 16
	v_mov_b32_e32 v6, v162
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s26
	s_mov_b32 s31, 0
	s_cselect_b32 s33, -1, 0
	s_branch .LBB3_45
.LBB3_43:                               ; %_ZL11fa2_scale_nPKhiii.exit395.7.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_sub_f32_e32 v0, v147, v7
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v147
	v_lshl_add_u32 v168, s31, 12, v162
	v_cvt_f16_f32_e64 v2.l, v144
	v_cvt_f16_f32_e64 v2.h, v174
	s_wait_dscnt 0x0
	v_dual_mul_f32 v0, 0x3fb8aa3b, v0 :: v_dual_add_f32 v147, v169, v170
	v_cvt_f16_f32_e64 v4.l, v177
	v_cvt_f16_f32_e64 v3.h, v176
	v_cvt_f16_f32_e64 v5.l, v179
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v0, v0
	v_cvt_f16_f32_e64 v3.l, v175
	v_cvt_f16_f32_e64 v4.h, v178
	v_cvt_f16_f32_e64 v5.h, v180
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v144, 0, v0, vcc_lo
	ds_load_b64 v[0:1], v168 offset:16384
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[9:10], v168 offset:16640
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v136, v136, v144 :: v_dual_mul_f32 v135, v135, v144
	v_dual_mul_f32 v134, v134, v144 :: v_dual_mul_f32 v131, v131, v144
	v_dual_mul_f32 v133, v133, v144 :: v_dual_mul_f32 v132, v132, v144
	v_dual_mul_f32 v129, v129, v144 :: v_dual_mul_f32 v130, v130, v144
	v_dual_mul_f32 v127, v127, v144 :: v_dual_mul_f32 v128, v128, v144
	v_dual_mul_f32 v125, v125, v144 :: v_dual_mul_f32 v126, v126, v144
	v_dual_mul_f32 v123, v123, v144 :: v_dual_mul_f32 v124, v124, v144
	v_dual_mul_f32 v121, v121, v144 :: v_dual_mul_f32 v122, v122, v144
	v_mul_f32_e32 v119, v119, v144
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[11:12], v0
	v_cvt_pk_f32_fp8_e64 v[13:14], v0 op_sel:[1,0]
	v_cvt_pk_f32_fp8_e32 v[15:16], v1
	v_cvt_pk_f32_fp8_e64 v[0:1], v1 op_sel:[1,0]
	v_dual_mul_f32 v120, v120, v144 :: v_dual_mul_f32 v117, v117, v144
	v_fma_mixlo_f16 v169, v8, v11, 0
	v_fma_mixhi_f16 v169, v138, v12, 0
	ds_load_b64 v[11:12], v168 offset:16896
	v_fma_mixlo_f16 v172, v141, v0, 0
	v_fma_mixhi_f16 v172, v143, v1, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e64 v[0:1], v9 op_sel:[1,0]
	;;#ASMSTART
	;;#ASMEND
	v_fma_mixlo_f16 v170, v137, v13, 0
	v_fma_mixhi_f16 v170, v140, v14, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v9
	v_fma_mixlo_f16 v171, v139, v15, 0
	v_fma_mixhi_f16 v171, v142, v16, 0
	v_cvt_pk_f32_fp8_e32 v[15:16], v10
	ds_load_b64 v[17:18], v168 offset:17152
	v_fma_mixlo_f16 v174, v137, v0, 0
	v_fma_mixhi_f16 v174, v140, v1, 0
	v_cvt_pk_f32_fp8_e64 v[0:1], v10 op_sel:[1,0]
	;;#ASMSTART
	;;#ASMEND
	v_fma_mixlo_f16 v175, v139, v15, 0
	v_fma_mixhi_f16 v175, v142, v16, 0
	v_fma_mixlo_f16 v173, v8, v13, 0
	v_fma_mixlo_f16 v176, v141, v0, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[9:10], v11
	v_fma_mixhi_f16 v176, v143, v1, 0
	v_cvt_pk_f32_fp8_e32 v[0:1], v12
	v_fma_mixhi_f16 v173, v138, v14, 0
	v_cvt_pk_f32_fp8_e64 v[13:14], v11 op_sel:[1,0]
	v_fma_mixlo_f16 v177, v8, v9, 0
	v_fma_mixhi_f16 v177, v138, v10, 0
	ds_load_b64 v[9:10], v168 offset:17408
	v_fma_mixlo_f16 v179, v139, v0, 0
	v_fma_mixhi_f16 v179, v142, v1, 0
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e64 v[0:1], v17 op_sel:[1,0]
	ds_load_b64 v[15:16], v168 offset:17664
	;;#ASMSTART
	;;#ASMEND
	v_cvt_pk_f32_fp8_e64 v[11:12], v12 op_sel:[1,0]
	v_wmma_f32_16x16x16_f16 v[129:136], v[169:172], v[2:5], v[129:136]
	v_fma_mixlo_f16 v182, v137, v0, 0
	v_fma_mixhi_f16 v182, v140, v1, 0
	v_fma_mixlo_f16 v178, v137, v13, 0
	v_fma_mixlo_f16 v180, v141, v11, 0
	v_fma_mixhi_f16 v180, v143, v12, 0
	v_cvt_pk_f32_fp8_e32 v[11:12], v18
	v_fma_mixhi_f16 v178, v140, v14, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v17
	v_dual_mul_f32 v118, v118, v144 :: v_dual_mul_f32 v115, v115, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_mixlo_f16 v183, v139, v11, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[0:1], v9
	v_fma_mixhi_f16 v183, v142, v12, 0
	v_cvt_pk_f32_fp8_e64 v[11:12], v9 op_sel:[1,0]
	v_fma_mixlo_f16 v181, v8, v13, 0
	v_fma_mixhi_f16 v181, v138, v14, 0
	v_fma_mixlo_f16 v169, v8, v0, 0
	v_fma_mixhi_f16 v169, v138, v1, 0
	ds_load_b64 v[0:1], v168 offset:17920
	v_cvt_pk_f32_fp8_e64 v[13:14], v18 op_sel:[1,0]
	v_fma_mixlo_f16 v170, v137, v11, 0
	v_fma_mixhi_f16 v170, v140, v12, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[11:12], v15
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v116, v116, v144 :: v_dual_mul_f32 v113, v113, v144
	v_dual_mul_f32 v114, v114, v144 :: v_dual_mul_f32 v111, v111, v144
	v_fma_mixlo_f16 v184, v141, v13, 0
	v_fma_mixhi_f16 v184, v143, v14, 0
	v_wmma_f32_16x16x16_f16 v[121:128], v[173:176], v[2:5], v[121:128]
	v_cvt_pk_f32_fp8_e32 v[13:14], v10
	v_cvt_pk_f32_fp8_e64 v[9:10], v10 op_sel:[1,0]
	v_fma_mixlo_f16 v173, v8, v11, 0
	ds_load_b64 v[17:18], v168 offset:18176
	v_fma_mixhi_f16 v173, v138, v12, 0
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_f16 v[113:120], v[177:180], v[2:5], v[113:120]
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[11:12], v0
	v_fma_mixlo_f16 v172, v141, v9, 0
	v_fma_mixhi_f16 v172, v143, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v15 op_sel:[1,0]
	v_fma_mixlo_f16 v171, v139, v13, 0
	v_fma_mixlo_f16 v177, v8, v11, 0
	v_fma_mixhi_f16 v177, v138, v12, 0
	ds_load_b64 v[11:12], v168 offset:18432
	v_fma_mixhi_f16 v171, v142, v14, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v16
	v_fma_mixlo_f16 v174, v137, v9, 0
	v_fma_mixhi_f16 v174, v140, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v16 op_sel:[1,0]
	v_dual_mul_f32 v112, v112, v144 :: v_dual_mul_f32 v109, v109, v144
	v_fma_mixlo_f16 v175, v139, v13, 0
	v_fma_mixhi_f16 v175, v142, v14, 0
	v_cvt_pk_f32_fp8_e64 v[13:14], v0 op_sel:[1,0]
	v_fma_mixlo_f16 v176, v141, v9, 0
	v_fma_mixhi_f16 v176, v143, v10, 0
	v_cvt_pk_f32_fp8_e32 v[9:10], v1
	v_cvt_pk_f32_fp8_e64 v[0:1], v1 op_sel:[1,0]
	v_dual_mul_f32 v110, v110, v144 :: v_dual_mul_f32 v107, v107, v144
	v_dual_mul_f32 v108, v108, v144 :: v_dual_mul_f32 v105, v105, v144
	v_dual_mul_f32 v106, v106, v144 :: v_dual_mul_f32 v103, v103, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_mixlo_f16 v180, v141, v0, 0
	v_fma_mixhi_f16 v180, v143, v1, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[0:1], v18
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v104, v104, v144 :: v_dual_mul_f32 v101, v101, v144
	v_dual_mul_f32 v102, v102, v144 :: v_dual_mul_f32 v99, v99, v144
	v_dual_mul_f32 v100, v100, v144 :: v_dual_mul_f32 v97, v97, v144
	v_dual_mul_f32 v98, v98, v144 :: v_dual_mul_f32 v95, v95, v144
	v_wmma_f32_16x16x16_f16 v[105:112], v[181:184], v[2:5], v[105:112]
	ds_load_b64 v[15:16], v168 offset:18688
	v_fma_mixlo_f16 v183, v139, v0, 0
	v_fma_mixhi_f16 v183, v142, v1, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[0:1], v11
	;;#ASMSTART
	;;#ASMEND
	v_fma_mixlo_f16 v178, v137, v13, 0
	v_fma_mixhi_f16 v178, v140, v14, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v17
	v_wmma_f32_16x16x16_f16 v[97:104], v[169:172], v[2:5], v[97:104]
	v_fma_mixlo_f16 v169, v8, v0, 0
	v_fma_mixhi_f16 v169, v138, v1, 0
	ds_load_b64 v[0:1], v168 offset:18944
	v_fma_mixlo_f16 v179, v139, v9, 0
	v_fma_mixhi_f16 v179, v142, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v17 op_sel:[1,0]
	v_fma_mixlo_f16 v181, v8, v13, 0
	v_fma_mixhi_f16 v181, v138, v14, 0
	v_cvt_pk_f32_fp8_e64 v[13:14], v18 op_sel:[1,0]
	;;#ASMSTART
	;;#ASMEND
	v_fma_mixlo_f16 v182, v137, v9, 0
	v_fma_mixhi_f16 v182, v140, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v11 op_sel:[1,0]
	v_fma_mixlo_f16 v184, v141, v13, 0
	v_fma_mixhi_f16 v184, v143, v14, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v12
	v_cvt_pk_f32_fp8_e64 v[11:12], v12 op_sel:[1,0]
	v_dual_mul_f32 v96, v96, v144 :: v_dual_mul_f32 v93, v93, v144
	v_dual_mul_f32 v94, v94, v144 :: v_dual_mul_f32 v91, v91, v144
	v_dual_mul_f32 v92, v92, v144 :: v_dual_mul_f32 v89, v89, v144
	v_dual_mul_f32 v90, v90, v144 :: v_dual_mul_f32 v87, v87, v144
	v_fma_mixlo_f16 v172, v141, v11, 0
	v_fma_mixhi_f16 v172, v143, v12, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e64 v[11:12], v15 op_sel:[1,0]
	ds_load_b64 v[17:18], v168 offset:19200
	v_dual_mul_f32 v88, v88, v144 :: v_dual_mul_f32 v85, v85, v144
	v_dual_mul_f32 v86, v86, v144 :: v_dual_mul_f32 v83, v83, v144
	v_dual_mul_f32 v84, v84, v144 :: v_dual_mul_f32 v81, v81, v144
	v_dual_mul_f32 v82, v82, v144 :: v_dual_mul_f32 v79, v79, v144
	v_wmma_f32_16x16x16_f16 v[89:96], v[173:176], v[2:5], v[89:96]
	v_fma_mixlo_f16 v170, v137, v9, 0
	v_fma_mixhi_f16 v170, v140, v10, 0
	v_cvt_pk_f32_fp8_e32 v[9:10], v15
	v_fma_mixlo_f16 v174, v137, v11, 0
	v_fma_mixhi_f16 v174, v140, v12, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[11:12], v0
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_f16 v[81:88], v[177:180], v[2:5], v[81:88]
	v_fma_mixlo_f16 v171, v139, v13, 0
	v_fma_mixhi_f16 v171, v142, v14, 0
	v_fma_mixlo_f16 v173, v8, v9, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v16
	v_fma_mixhi_f16 v173, v138, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v16 op_sel:[1,0]
	v_fma_mixlo_f16 v177, v8, v11, 0
	v_fma_mixhi_f16 v177, v138, v12, 0
	ds_load_b64 v[11:12], v168 offset:19456
	v_fma_mixlo_f16 v175, v139, v13, 0
	v_fma_mixhi_f16 v175, v142, v14, 0
	v_cvt_pk_f32_fp8_e64 v[13:14], v0 op_sel:[1,0]
	v_fma_mixlo_f16 v176, v141, v9, 0
	v_fma_mixhi_f16 v176, v143, v10, 0
	v_cvt_pk_f32_fp8_e32 v[9:10], v1
	v_cvt_pk_f32_fp8_e64 v[0:1], v1 op_sel:[1,0]
	v_dual_mul_f32 v80, v80, v144 :: v_dual_mul_f32 v77, v77, v144
	v_dual_mul_f32 v78, v78, v144 :: v_dual_mul_f32 v75, v75, v144
	v_dual_mul_f32 v76, v76, v144 :: v_dual_mul_f32 v73, v73, v144
	v_dual_mul_f32 v74, v74, v144 :: v_dual_mul_f32 v71, v71, v144
	v_fma_mixlo_f16 v180, v141, v0, 0
	v_fma_mixhi_f16 v180, v143, v1, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[0:1], v18
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_f16 v[73:80], v[181:184], v[2:5], v[73:80]
	v_fma_mixlo_f16 v178, v137, v13, 0
	v_fma_mixhi_f16 v178, v140, v14, 0
	v_fma_mixlo_f16 v179, v139, v9, 0
	v_cvt_pk_f32_fp8_e32 v[13:14], v17
	v_fma_mixhi_f16 v179, v142, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v17 op_sel:[1,0]
	v_fma_mixlo_f16 v183, v139, v0, 0
	v_fma_mixhi_f16 v183, v142, v1, 0
	ds_load_b64 v[0:1], v168 offset:19712
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v72, v72, v144 :: v_dual_mul_f32 v69, v69, v144
	v_dual_mul_f32 v70, v70, v144 :: v_dual_mul_f32 v67, v67, v144
	v_dual_mul_f32 v68, v68, v144 :: v_dual_mul_f32 v65, v65, v144
	v_dual_mul_f32 v66, v66, v144 :: v_dual_mul_f32 v63, v63, v144
	v_fma_mixlo_f16 v181, v8, v13, 0
	v_fma_mixhi_f16 v181, v138, v14, 0
	v_fma_mixlo_f16 v182, v137, v9, 0
	v_fma_mixhi_f16 v182, v140, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v18 op_sel:[1,0]
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e64 v[13:14], v11 op_sel:[1,0]
	ds_load_b64 v[17:18], v168 offset:19968
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_f16 v[65:72], v[169:172], v[2:5], v[65:72]
	v_fma_mixlo_f16 v184, v141, v9, 0
	v_fma_mixhi_f16 v184, v143, v10, 0
	v_cvt_pk_f32_fp8_e32 v[9:10], v11
	v_fma_mixlo_f16 v170, v137, v13, 0
	v_fma_mixhi_f16 v170, v140, v14, 0
	ds_load_b64 v[13:14], v168 offset:20224
	v_dual_mul_f32 v64, v64, v144 :: v_dual_mul_f32 v61, v61, v144
	v_fma_mixlo_f16 v169, v8, v9, 0
	v_fma_mixhi_f16 v169, v138, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v12 op_sel:[1,0]
	v_dual_mul_f32 v62, v62, v144 :: v_dual_mul_f32 v59, v59, v144
	v_dual_mul_f32 v60, v60, v144 :: v_dual_mul_f32 v57, v57, v144
	v_dual_mul_f32 v58, v58, v144 :: v_dual_mul_f32 v55, v55, v144
	v_cvt_pk_f32_fp8_e32 v[15:16], v12
	v_fma_mixlo_f16 v172, v141, v9, 0
	s_wait_dscnt 0x2
	v_cvt_pk_f32_fp8_e32 v[11:12], v0
	v_fma_mixhi_f16 v172, v143, v10, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v0 op_sel:[1,0]
	v_dual_mul_f32 v56, v56, v144 :: v_dual_mul_f32 v53, v53, v144
	v_dual_mul_f32 v54, v54, v144 :: v_dual_mul_f32 v51, v51, v144
	v_dual_mul_f32 v52, v52, v144 :: v_dual_mul_f32 v49, v49, v144
	v_dual_mul_f32 v50, v50, v144 :: v_dual_mul_f32 v47, v47, v144
	v_wmma_f32_16x16x16_f16 v[57:64], v[173:176], v[2:5], v[57:64]
	v_fma_mixlo_f16 v171, v139, v15, 0
	v_fma_mixhi_f16 v171, v142, v16, 0
	v_fma_mixlo_f16 v173, v8, v11, 0
	v_cvt_pk_f32_fp8_e32 v[15:16], v1
	v_fma_mixhi_f16 v173, v138, v12, 0
	v_fma_mixlo_f16 v174, v137, v9, 0
	v_cvt_pk_f32_fp8_e64 v[0:1], v1 op_sel:[1,0]
	v_fma_mixhi_f16 v174, v140, v10, 0
	s_wait_dscnt 0x1
	v_cvt_pk_f32_fp8_e32 v[9:10], v17
	v_cvt_pk_f32_fp8_e64 v[11:12], v17 op_sel:[1,0]
	v_dual_mul_f32 v48, v48, v144 :: v_dual_mul_f32 v45, v45, v144
	v_dual_mul_f32 v46, v46, v144 :: v_dual_mul_f32 v43, v43, v144
	v_dual_mul_f32 v44, v44, v144 :: v_dual_mul_f32 v41, v41, v144
	v_dual_mul_f32 v42, v42, v144 :: v_dual_mul_f32 v39, v39, v144
	v_wmma_f32_16x16x16_f16 v[49:56], v[177:180], v[2:5], v[49:56]
	v_fma_mixlo_f16 v176, v141, v0, 0
	v_fma_mixhi_f16 v176, v143, v1, 0
	v_fma_mixlo_f16 v177, v8, v9, 0
	v_cvt_pk_f32_fp8_e32 v[0:1], v18
	v_fma_mixhi_f16 v177, v138, v10, 0
	v_fma_mixlo_f16 v178, v137, v11, 0
	v_cvt_pk_f32_fp8_e64 v[9:10], v18 op_sel:[1,0]
	v_fma_mixhi_f16 v178, v140, v12, 0
	s_wait_dscnt 0x0
	v_cvt_pk_f32_fp8_e32 v[11:12], v13
	v_wmma_f32_16x16x16_f16 v[41:48], v[181:184], v[2:5], v[41:48]
	v_fma_mixlo_f16 v179, v139, v0, 0
	v_fma_mixhi_f16 v179, v142, v1, 0
	v_fma_mixlo_f16 v180, v141, v9, 0
	v_fma_mixhi_f16 v180, v143, v10, 0
	v_cvt_pk_f32_fp8_e64 v[0:1], v13 op_sel:[1,0]
	v_fma_mixlo_f16 v181, v8, v11, 0
	v_cvt_pk_f32_fp8_e32 v[8:9], v14
	v_cvt_pk_f32_fp8_e64 v[10:11], v14 op_sel:[1,0]
	v_dual_mul_f32 v40, v40, v144 :: v_dual_mul_f32 v37, v37, v144
	v_dual_mul_f32 v38, v38, v144 :: v_dual_mul_f32 v35, v35, v144
	v_dual_mul_f32 v36, v36, v144 :: v_dual_mul_f32 v33, v33, v144
	v_dual_mul_f32 v34, v34, v144 :: v_dual_mul_f32 v31, v31, v144
	v_dual_mul_f32 v32, v32, v144 :: v_dual_mul_f32 v29, v29, v144
	v_dual_mul_f32 v30, v30, v144 :: v_dual_mul_f32 v27, v27, v144
	v_dual_mul_f32 v28, v28, v144 :: v_dual_mul_f32 v25, v25, v144
	v_dual_mul_f32 v26, v26, v144 :: v_dual_mul_f32 v207, v207, v144
	v_dual_mul_f32 v208, v208, v144 :: v_dual_mul_f32 v205, v205, v144
	v_dual_mul_f32 v206, v206, v144 :: v_dual_mul_f32 v203, v203, v144
	v_dual_mul_f32 v204, v204, v144 :: v_dual_mul_f32 v201, v201, v144
	v_dual_mul_f32 v202, v202, v144 :: v_dual_mul_f32 v199, v199, v144
	v_dual_mul_f32 v200, v200, v144 :: v_dual_mul_f32 v197, v197, v144
	v_dual_mul_f32 v198, v198, v144 :: v_dual_mul_f32 v195, v195, v144
	v_dual_mul_f32 v196, v196, v144 :: v_dual_mul_f32 v193, v193, v144
	v_dual_mul_f32 v194, v194, v144 :: v_dual_fmac_f32 v147, v148, v144
	v_fma_mixlo_f16 v175, v139, v15, 0
	v_fma_mixhi_f16 v175, v142, v16, 0
	v_fma_mixhi_f16 v181, v138, v12, 0
	v_fma_mixlo_f16 v182, v137, v0, 0
	v_fma_mixhi_f16 v182, v140, v1, 0
	v_fma_mixlo_f16 v183, v139, v8, 0
	v_fma_mixhi_f16 v183, v142, v9, 0
	v_fma_mixlo_f16 v184, v141, v10, 0
	v_fma_mixhi_f16 v184, v143, v11, 0
	v_wmma_f32_16x16x16_f16 v[33:40], v[169:172], v[2:5], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[173:176], v[2:5], v[25:32]
	v_wmma_f32_16x16x16_f16 v[201:208], v[177:180], v[2:5], v[201:208]
	v_mov_b32_e32 v148, v147
	v_wmma_f32_16x16x16_f16 v[193:200], v[181:184], v[2:5], v[193:200]
	;;#ASMSTART
	;;#ASMEND
.LBB3_44:                               ; %.loopexit421.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v147, v7 :: v_dual_add_nc_u32 v6, 0x100, v6
	s_add_co_i32 s31, s31, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s31, 4
	s_cbranch_scc0 .LBB3_11
.LBB3_45:                               ; %NodeBlock
                                        ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB3_50 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s31, 1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB3_48
; %bb.46:                               ; %LeafBlock
                                        ;   in Loop: Header=BB3_45 Depth=2
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
.LBB3_48:                               ; %Flow287
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_82
; %bb.49:                               ; %.preheader425.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v2, v151
	v_mov_b32_e32 v3, v152
	s_movk_i32 s0, 0xc000
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_mov_b32_e32 v144, v137
.LBB3_50:                               ; %.preheader.i
                                        ;   Parent Loop BB3_14 Depth=1
                                        ;     Parent Loop BB3_45 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_clause 0x3
	global_load_b64 v[4:5], v[2:3], off offset:-48
	global_load_b64 v[7:8], v[2:3], off offset:-32
	global_load_b64 v[176:177], v[2:3], off offset:-16
	global_load_b64 v[178:179], v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v0, s0, v6
	v_add_co_u32 v2, vcc_lo, v2, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	ds_load_2addr_stride64_b64 v[168:171], v0 offset0:32 offset1:34
	ds_load_2addr_stride64_b64 v[172:175], v0 offset0:36 offset1:38
	s_addk_co_i32 s0, 0x1000
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[168:169], v[4:5], v[137:144]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[170:171], v[7:8], v[137:144]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[172:173], v[176:177], v[137:144]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[174:175], v[178:179], v[137:144]
	s_cbranch_scc1 .LBB3_50
; %bb.51:                               ;   in Loop: Header=BB3_45 Depth=2
	s_lshl4_add_u32 s0, s31, s29
	v_mov_b32_e32 v169, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 15
	v_mov_b32_e32 v7, 0
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
	global_load_b32 v7, v[149:150], off
.LBB3_53:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, s0, v164
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e32 vcc_lo, s26, v2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB3_55
; %bb.54:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[3:4], null, 0x408, v2, s[22:23]
	global_load_d16_b16 v3, v[3:4], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v169, v3.l
.LBB3_55:                               ; %_ZL11fa2_scale_nPKhiii.exit.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v3, 1, v2
	v_cmp_gt_i32_e64 s0, s26, v2
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v174, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB3_57
; %bb.56:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[4:5], null, 0x408, v3, s[22:23]
	global_load_d16_b16 v4, v[4:5], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v4.l
.LBB3_57:                               ; %_ZL11fa2_scale_nPKhiii.exit.1.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v4, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s1, s26, v4
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB3_59
; %bb.58:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[170:171], null, 0x408, v4, s[22:23]
	global_load_d16_b16 v5, v[170:171], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v170, v5.l
.LBB3_59:                               ; %_ZL11fa2_scale_nPKhiii.exit.2.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v5, 3, v2
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v177, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e64 s3, s26, v5
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_61
; %bb.60:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[171:172], null, 0x408, v5, s[22:23]
	global_load_d16_b16 v8, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v177, v8.l
.LBB3_61:                               ; %_ZL11fa2_scale_nPKhiii.exit.3.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v168, 4, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s4, s26, v168
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB3_63
; %bb.62:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[171:172], null, 0x408, v168, s[22:23]
	global_load_d16_b16 v8, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v175, v8.l
.LBB3_63:                               ; %_ZL11fa2_scale_nPKhiii.exit.4.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v171, 5, v2
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v178, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e64 s5, s26, v171
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_65
; %bb.64:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[172:173], null, 0x408, v171, s[22:23]
	global_load_d16_b16 v8, v[172:173], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v178, v8.l
.LBB3_65:                               ; %_ZL11fa2_scale_nPKhiii.exit.5.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v172, 6, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s6, s26, v172
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB3_67
; %bb.66:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[179:180], null, 0x408, v172, s[22:23]
	global_load_d16_b16 v8, v[179:180], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v176, v8.l
.LBB3_67:                               ; %_ZL11fa2_scale_nPKhiii.exit.6.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v173, 7, v2
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v179, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_ge_i32_e64 s7, s26, v173
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_69
; %bb.68:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[179:180], null, 0x408, v173, s[22:23]
	global_load_d16_b16 v179, v[179:180], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v179, v179.l
.LBB3_69:                               ; %_ZL11fa2_scale_nPKhiii.exit.7.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mul_f32 v0, v186, v137 :: v_dual_mul_f32 v1, v186, v138
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s8, v2, v7
	v_cmp_lt_i32_e64 s9, v2, v7
	v_cmp_le_i32_e64 s10, v4, v7
	v_dual_mul_f32 v0, v0, v169 :: v_dual_mul_f32 v9, v186, v139
	s_or_b32 s8, s34, s8
	v_dual_mul_f32 v1, v1, v174 :: v_dual_mul_f32 v10, v186, v140
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	s_or_b32 s9, s34, s9
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v0, 0xff800000, v0, s8
	v_cmp_le_i32_e64 s8, v5, v7
	v_dual_mul_f32 v9, v9, v170 :: v_dual_mul_f32 v10, v10, v177
	s_or_b32 s10, s34, s10
	s_and_b32 s9, s19, s9
	v_dual_mul_f32 v11, v186, v141 :: v_dual_mul_f32 v12, v186, v142
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v1, 0xff800000, v1, s9
	s_and_b32 s9, s19, s10
	s_or_b32 s10, s34, s8
	v_cmp_le_i32_e64 s8, v168, v7
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v9, 0xff800000, v9, s9
	v_cmp_le_i32_e64 s9, v171, v7
	v_dual_mul_f32 v11, v11, v175 :: v_dual_mul_f32 v12, v12, v178
	s_or_b32 s8, s34, s8
	s_and_b32 s10, s19, s10
	s_or_b32 s9, s34, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	v_mul_f32_e32 v13, v186, v143
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v11, 0xff800000, v11, s8
	s_and_b32 s8, s19, s9
	v_cndmask_b32_e64 v10, 0xff800000, v10, s10
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v12, 0xff800000, v12, s8
	v_cmp_le_i32_e64 s8, v172, v7
	v_cmp_le_i32_e64 s9, v173, v7
	v_mul_f32_e32 v7, v186, v144
	v_max3_num_f32 v14, v0, 0xff800000, v1
	v_mul_f32_e32 v13, v13, v176
	s_or_b32 s8, s34, s8
	s_or_b32 s9, s34, s9
	v_mul_f32_e32 v7, v7, v179
	v_max3_num_f32 v14, v14, v9, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v13, 0xff800000, v13, s8
	s_and_b32 s8, s19, s9
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v15, 0xff800000, v7, s8
	v_max3_num_f32 v7, v14, v11, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v7, v7, v13, v15
	ds_bpermute_b32 v14, v159, v7
	s_wait_dscnt 0x0
	v_max3_num_f32 v7, v147, v7, v14
	v_dual_sub_f32 v0, v0, v7 :: v_dual_sub_f32 v1, v1, v7
	v_sub_f32_e32 v10, v10, v7
	v_cmp_eq_f32_e64 s8, 0xff800000, v7
	v_dual_sub_f32 v9, v9, v7 :: v_dual_sub_f32 v12, v12, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v0, 0x3fb8aa3b, v0 :: v_dual_mul_f32 v1, 0x3fb8aa3b, v1
	v_dual_mul_f32 v10, 0x3fb8aa3b, v10 :: v_dual_mul_f32 v9, 0x3fb8aa3b, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v10, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v144, v0, 0, s8
	v_sub_f32_e32 v0, v13, v7
	v_exp_f32_e32 v1, v1
	v_cndmask_b32_e64 v176, v10, 0, s8
	v_sub_f32_e32 v10, v15, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v11, v11, v7 :: v_dual_mul_f32 v0, 0x3fb8aa3b, v0
	v_exp_f32_e32 v9, v9
	v_mul_f32_e32 v10, 0x3fb8aa3b, v10
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v174, v1, 0, s8
	v_mul_f32_e32 v1, 0x3fb8aa3b, v12
	v_exp_f32_e32 v0, v0
	v_cndmask_b32_e64 v175, v9, 0, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v1, v1
	v_cndmask_b32_e64 v179, v0, 0, s8
	v_add_f32_e32 v9, v144, v174
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v178, v1, 0, s8
	v_mul_f32_e32 v11, 0x3fb8aa3b, v11
	v_add_f32_e32 v9, v175, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v11, v11
	v_add_f32_e32 v9, v176, v9
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v177, v11, 0, s8
	v_add_f32_e32 v1, v177, v9
	v_exp_f32_e32 v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v178, v1
	v_add_f32_e32 v0, v179, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v180, v9, 0, s8
	v_add_f32_e32 v169, v180, v0
	ds_bpermute_b32 v170, v159, v169
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB3_71
; %bb.70:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[137:138], null, 0x408, v2, s[12:13]
	global_load_d16_b16 v2, v[137:138], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v8, v2.l
.LBB3_71:                               ; %_ZL11fa2_scale_nPKhiii.exit395.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v138, 0
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB3_73
; %bb.72:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v3, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v138, v2.l
.LBB3_73:                               ; %_ZL11fa2_scale_nPKhiii.exit395.1.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_75
; %bb.74:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v4, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v137, v2.l
.LBB3_75:                               ; %_ZL11fa2_scale_nPKhiii.exit395.2.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v140, 0
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB3_77
; %bb.76:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v5, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v140, v2.l
.LBB3_77:                               ; %_ZL11fa2_scale_nPKhiii.exit395.3.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB3_79
; %bb.78:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v168, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v139, v2.l
.LBB3_79:                               ; %_ZL11fa2_scale_nPKhiii.exit395.4.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v142, 0
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB3_83
; %bb.80:                               ; %_ZL11fa2_scale_nPKhiii.exit395.5.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execnz .LBB3_84
.LBB3_81:                               ; %_ZL11fa2_scale_nPKhiii.exit395.6.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v143, 0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB3_43
	s_branch .LBB3_85
.LBB3_82:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mov_b32_e32 v7, v147
	s_branch .LBB3_44
.LBB3_83:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v171, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v142, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execz .LBB3_81
.LBB3_84:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v172, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v141, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v143, 0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB3_43
.LBB3_85:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v173, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v143, v2.l
	s_branch .LBB3_43
.LBB3_86:                               ; %._crit_edge.i
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB3_91
; %bb.87:
	v_cmp_eq_u32_e32 vcc_lo, 0, v160
	s_and_b32 s1, vcc_lo, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_89
; %bb.88:
	v_mad_co_u64_u32 v[0:1], null, v145, s17, s[18:19]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v0, 0x102, v0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	global_store_b64 v[0:1], v[147:148], off
.LBB3_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s19
	s_cbranch_execz .LBB3_91
; %bb.90:                               ; %.loopexit.loopexit.i
	v_mad_co_u64_u32 v[0:1], null, v145, s17, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v0, 0x102, v0
	v_dual_mov_b32 v1, 0 :: v_dual_add_nc_u32 v2, v0, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[3:4], 2, v[0:1]
	v_add_nc_u32_e32 v0, 2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v3, vcc_lo, s14, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s15, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_add_nc_u32_e32 v0, 18, v2
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_add_nc_u32_e32 v0, 34, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v5, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_add_nc_u32_e32 v0, 50, v2
	v_add_co_u32 v7, vcc_lo, v3, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v4, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x42, v2
	v_add_co_u32 v9, vcc_lo, v3, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v4, v10, vcc_lo
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x52, v2
	v_add_co_u32 v11, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x62, v2
	v_add_co_u32 v13, vcc_lo, v3, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v4, v14, vcc_lo
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x72, v2
	v_add_co_u32 v15, vcc_lo, v3, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v4, v16, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x82, v2
	v_add_co_u32 v17, vcc_lo, v3, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v4, v18, vcc_lo
	v_lshlrev_b64_e32 v[21:22], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x92, v2
	v_add_co_u32 v19, vcc_lo, v3, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v4, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[23:24], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xa2, v2
	s_clause 0x7
	global_store_b64 v[5:6], v[129:130], off
	global_store_b64 v[7:8], v[121:122], off
	global_store_b64 v[9:10], v[113:114], off
	global_store_b64 v[11:12], v[105:106], off
	global_store_b64 v[13:14], v[97:98], off
	global_store_b64 v[15:16], v[89:90], off
	global_store_b64 v[17:18], v[81:82], off
	global_store_b64 v[19:20], v[73:74], off
	v_add_co_u32 v21, vcc_lo, v3, v21
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, v4, v22, vcc_lo
	v_lshlrev_b64_e32 v[137:138], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xb2, v2
	v_add_co_u32 v23, vcc_lo, v3, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, v4, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[139:140], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xc2, v2
	v_add_co_u32 v137, vcc_lo, v3, v137
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, v4, v138, vcc_lo
	v_lshlrev_b64_e32 v[141:142], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xd2, v2
	v_add_co_u32 v139, vcc_lo, v3, v139
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v140, null, v4, v140, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[143:144], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xe2, v2
	v_add_co_u32 v141, vcc_lo, v3, v141
	s_clause 0x3
	global_store_b64 v[21:22], v[65:66], off
	global_store_b64 v[23:24], v[57:58], off
	global_store_b64 v[137:138], v[49:50], off
	global_store_b64 v[139:140], v[41:42], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v142, null, v4, v142, vcc_lo
	v_lshlrev_b64_e32 v[145:146], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xf2, v2
	v_add_co_u32 v143, vcc_lo, v3, v143
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v144, null, v4, v144, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[147:148], 2, v[0:1]
	v_add_nc_u32_e32 v0, 4, v2
	v_add_co_u32 v145, vcc_lo, v3, v145
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v146, null, v4, v146, vcc_lo
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_add_nc_u32_e32 v0, 20, v2
	v_add_co_u32 v147, vcc_lo, v3, v147
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v148, null, v4, v148, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_add_nc_u32_e32 v0, 36, v2
	s_clause 0x3
	global_store_b64 v[141:142], v[33:34], off
	global_store_b64 v[143:144], v[25:26], off
	global_store_b64 v[145:146], v[201:202], off
	global_store_b64 v[147:148], v[193:194], off
	v_add_co_u32 v5, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_add_nc_u32_e32 v0, 52, v2
	v_add_co_u32 v7, vcc_lo, v3, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v4, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x44, v2
	v_add_co_u32 v9, vcc_lo, v3, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v4, v10, vcc_lo
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x54, v2
	v_add_co_u32 v11, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x64, v2
	v_add_co_u32 v13, vcc_lo, v3, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v4, v14, vcc_lo
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x74, v2
	v_add_co_u32 v15, vcc_lo, v3, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v4, v16, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x84, v2
	v_add_co_u32 v17, vcc_lo, v3, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v4, v18, vcc_lo
	v_lshlrev_b64_e32 v[21:22], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x94, v2
	v_add_co_u32 v19, vcc_lo, v3, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v4, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[23:24], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xa4, v2
	s_clause 0x7
	global_store_b64 v[5:6], v[131:132], off
	global_store_b64 v[7:8], v[123:124], off
	global_store_b64 v[9:10], v[115:116], off
	global_store_b64 v[11:12], v[107:108], off
	global_store_b64 v[13:14], v[99:100], off
	global_store_b64 v[15:16], v[91:92], off
	global_store_b64 v[17:18], v[83:84], off
	global_store_b64 v[19:20], v[75:76], off
	v_add_co_u32 v21, vcc_lo, v3, v21
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, v4, v22, vcc_lo
	v_lshlrev_b64_e32 v[25:26], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xb4, v2
	v_add_co_u32 v23, vcc_lo, v3, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, v4, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[33:34], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xc4, v2
	v_add_co_u32 v25, vcc_lo, v3, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v26, vcc_lo
	v_lshlrev_b64_e32 v[41:42], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xd4, v2
	v_add_co_u32 v33, vcc_lo, v3, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, v4, v34, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[49:50], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xe4, v2
	s_clause 0x3
	global_store_b64 v[21:22], v[67:68], off
	global_store_b64 v[23:24], v[59:60], off
	global_store_b64 v[25:26], v[51:52], off
	global_store_b64 v[33:34], v[43:44], off
	v_add_co_u32 v41, vcc_lo, v3, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v4, v42, vcc_lo
	v_lshlrev_b64_e32 v[57:58], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xf4, v2
	v_add_co_u32 v49, vcc_lo, v3, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v50, null, v4, v50, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[65:66], 2, v[0:1]
	v_add_nc_u32_e32 v0, 6, v2
	v_add_co_u32 v57, vcc_lo, v3, v57
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v58, null, v4, v58, vcc_lo
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_add_nc_u32_e32 v0, 22, v2
	v_add_co_u32 v65, vcc_lo, v3, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v4, v66, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_add_nc_u32_e32 v0, 38, v2
	s_clause 0x3
	global_store_b64 v[41:42], v[35:36], off
	global_store_b64 v[49:50], v[27:28], off
	global_store_b64 v[57:58], v[203:204], off
	global_store_b64 v[65:66], v[195:196], off
	v_add_co_u32 v5, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_add_nc_u32_e32 v0, 54, v2
	v_add_co_u32 v7, vcc_lo, v3, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v4, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x46, v2
	v_add_co_u32 v9, vcc_lo, v3, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v4, v10, vcc_lo
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x56, v2
	v_add_co_u32 v11, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x66, v2
	v_add_co_u32 v13, vcc_lo, v3, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v4, v14, vcc_lo
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x76, v2
	v_add_co_u32 v15, vcc_lo, v3, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v4, v16, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x86, v2
	v_add_co_u32 v17, vcc_lo, v3, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v4, v18, vcc_lo
	v_lshlrev_b64_e32 v[21:22], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x96, v2
	v_add_co_u32 v19, vcc_lo, v3, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v4, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[23:24], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xa6, v2
	s_clause 0x7
	global_store_b64 v[5:6], v[133:134], off
	global_store_b64 v[7:8], v[125:126], off
	global_store_b64 v[9:10], v[117:118], off
	global_store_b64 v[11:12], v[109:110], off
	global_store_b64 v[13:14], v[101:102], off
	global_store_b64 v[15:16], v[93:94], off
	global_store_b64 v[17:18], v[85:86], off
	global_store_b64 v[19:20], v[77:78], off
	v_add_co_u32 v21, vcc_lo, v3, v21
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, v4, v22, vcc_lo
	v_lshlrev_b64_e32 v[25:26], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xb6, v2
	v_add_co_u32 v23, vcc_lo, v3, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, v4, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[27:28], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xc6, v2
	v_add_co_u32 v25, vcc_lo, v3, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v26, vcc_lo
	v_lshlrev_b64_e32 v[33:34], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xd6, v2
	v_add_co_u32 v27, vcc_lo, v3, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, v4, v28, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[35:36], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xe6, v2
	s_clause 0x3
	global_store_b64 v[21:22], v[69:70], off
	global_store_b64 v[23:24], v[61:62], off
	global_store_b64 v[25:26], v[53:54], off
	global_store_b64 v[27:28], v[45:46], off
	v_add_co_u32 v33, vcc_lo, v3, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, v4, v34, vcc_lo
	v_lshlrev_b64_e32 v[41:42], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xf6, v2
	v_add_co_u32 v35, vcc_lo, v3, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, v4, v36, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[43:44], 2, v[0:1]
	v_add_nc_u32_e32 v0, 8, v2
	v_add_co_u32 v41, vcc_lo, v3, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v4, v42, vcc_lo
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_add_nc_u32_e32 v0, 24, v2
	v_add_co_u32 v43, vcc_lo, v3, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, v4, v44, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_add_nc_u32_e32 v0, 40, v2
	v_add_co_u32 v5, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_add_nc_u32_e32 v0, 56, v2
	v_add_co_u32 v7, vcc_lo, v3, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v4, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x48, v2
	v_add_co_u32 v9, vcc_lo, v3, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v4, v10, vcc_lo
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x58, v2
	v_add_co_u32 v11, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x68, v2
	v_add_co_u32 v13, vcc_lo, v3, v13
	s_clause 0x3
	global_store_b64 v[33:34], v[37:38], off
	global_store_b64 v[35:36], v[29:30], off
	global_store_b64 v[41:42], v[205:206], off
	global_store_b64 v[43:44], v[197:198], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v4, v14, vcc_lo
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x78, v2
	v_add_co_u32 v15, vcc_lo, v3, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v4, v16, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x88, v2
	v_add_co_u32 v17, vcc_lo, v3, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v4, v18, vcc_lo
	v_lshlrev_b64_e32 v[21:22], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x98, v2
	v_add_co_u32 v19, vcc_lo, v3, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v4, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[23:24], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xa8, v2
	v_add_co_u32 v21, vcc_lo, v3, v21
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, v4, v22, vcc_lo
	v_lshlrev_b64_e32 v[25:26], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xb8, v2
	v_add_co_u32 v23, vcc_lo, v3, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, v4, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[27:28], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xc8, v2
	v_add_co_u32 v25, vcc_lo, v3, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v26, vcc_lo
	v_lshlrev_b64_e32 v[29:30], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xd8, v2
	v_add_co_u32 v27, vcc_lo, v3, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, v4, v28, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[33:34], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xe8, v2
	v_add_co_u32 v29, vcc_lo, v3, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, v4, v30, vcc_lo
	v_lshlrev_b64_e32 v[35:36], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xf8, v2
	v_add_co_u32 v33, vcc_lo, v3, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, v4, v34, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v35, vcc_lo, v3, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, v4, v36, vcc_lo
	v_add_co_u32 v0, vcc_lo, v3, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v4, v1, vcc_lo
	s_clause 0xf
	global_store_b64 v[5:6], v[135:136], off
	global_store_b64 v[7:8], v[127:128], off
	global_store_b64 v[9:10], v[119:120], off
	global_store_b64 v[11:12], v[111:112], off
	global_store_b64 v[13:14], v[103:104], off
	global_store_b64 v[15:16], v[95:96], off
	global_store_b64 v[17:18], v[87:88], off
	global_store_b64 v[19:20], v[79:80], off
	global_store_b64 v[21:22], v[71:72], off
	global_store_b64 v[23:24], v[63:64], off
	global_store_b64 v[25:26], v[55:56], off
	global_store_b64 v[27:28], v[47:48], off
	global_store_b64 v[29:30], v[39:40], off
	global_store_b64 v[33:34], v[31:32], off
	global_store_b64 v[35:36], v[207:208], off
	global_store_b64 v[0:1], v[199:200], off
.LBB3_91:                               ; %_Z16fa2_stageb_nbodyILb1EEvPKhS1_S1_PfPKiifiiiiii.exit
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
		.amdhsa_next_free_vgpr 209
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
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.num_vgpr, 209
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
; codeLenInByte = 10564
; TotalNumSgprs: 37
; NumVgprs: 209
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 26
; NumSGPRsForWavesPerEU: 37
; NumVGPRsForWavesPerEU: 209
; Occupancy: 7
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
; %bb.2:                                ; %.lr.ph
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
; %bb.4:                                ; %.preheader63
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
.LBB4_5:                                ; %._crit_edge
                                        ;   in Loop: Header=BB4_6 Depth=1
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
.LBB4_6:                                ; %.preheader
                                        ; =>This Loop Header: Depth=1
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
.LBB4_10:                               ; %.loopexit
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
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_91e6e03ad624f8df,@object ; @__hip_cuid_91e6e03ad624f8df
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_91e6e03ad624f8df
__hip_cuid_91e6e03ad624f8df:
	.byte	0                               ; 0x0
	.size	__hip_cuid_91e6e03ad624f8df, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_91e6e03ad624f8df
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
    .vgpr_count:     209
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
    .vgpr_count:     209
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
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
