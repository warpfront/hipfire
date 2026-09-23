	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_f16_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_f16_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_f16_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_f16_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_f16_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_f16_gfx1201
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
	s_cbranch_vccnz .LBB0_279
; %bb.1:
	s_cmp_gt_i32 ttmp7, 3
	s_cbranch_scc1 .LBB0_279
; %bb.2:
	s_load_b32 s11, s[0:1], 0x38
	s_lshl_b32 s3, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s3, s7
	s_cbranch_scc1 .LBB0_279
; %bb.3:
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v185, 0 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v1, 0
	v_lshrrev_b32_e32 v192, 3, v0
	v_mov_b32_e32 v187, 0
	s_mov_b32 s20, 0
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_5
; %bb.4:
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_or_b32 v185, v0, 7, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 6, v1
	v_cmp_gt_i32_e32 vcc_lo, s7, v185
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, ttmp7, 6, v[1:2]
	s_and_b32 s20, vcc_lo, exec_lo
	v_and_or_b32 v1, v192, 1, v1
	v_lshlrev_b32_e32 v187, 8, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x1800, v185, v[187:188]
	v_mov_b32_e32 v2, 0
.LBB0_5:
	s_or_b32 exec_lo, exec_lo, s4
	s_clause 0x1
	s_load_b256 s[12:19], s[0:1], 0x0
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
.LBB0_8:                                ; %Flow868
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_9:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s4
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	v_lshrrev_b32_e32 v8, 1, v0
	v_lshl_add_u32 v196, v6, 4, 0
	v_ashrrev_i32_e32 v186, 31, v185
	v_xor_b32_e32 v7, 16, v3
	v_xor_b32_e32 v12, 8, v3
	v_xor_b32_e32 v16, 4, v3
	v_xor_b32_e32 v17, 2, v3
	v_xor_b32_e32 v18, 1, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	v_lshlrev_b64_e32 v[129:130], 2, v[185:186]
	v_lshrrev_b32_e32 v11, 2, v0
	v_or_b32_e32 v14, 0x80, v0
	s_mov_b32 s4, ttmp7
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v3, v7 :: v_dual_mov_b32 v222, 0xff800000
	v_cmp_gt_u32_e32 vcc_lo, 32, v12
	v_and_b32_e32 v131, 16, v6
	v_lshrrev_b32_e32 v6, 1, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v57, 0 :: v_dual_lshlrev_b32 v194, 2, v7
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, v3, v12, vcc_lo
	v_lshlrev_b32_e32 v7, 5, v0
	v_mov_b32_e32 v197, 0
	ds_bpermute_b32 v9, v194, v4
	ds_bpermute_b32 v10, v194, v5
	v_lshlrev_b32_e32 v12, 2, v12
	v_mov_b32_e32 v60, v57
	v_dual_mov_b32 v59, v57 :: v_dual_and_b32 v198, 0xe0, v7
	v_dual_mov_b32 v64, v57 :: v_dual_lshlrev_b32 v13, 3, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v16
	v_dual_mov_b32 v61, v57 :: v_dual_and_b32 v186, 8, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v201, 0x200, v13
	v_dual_mov_b32 v58, v57 :: v_dual_lshlrev_b32 v193, 8, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, v3, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v17
	v_and_b32_e32 v6, 12, v8
	v_dual_mov_b32 v62, v57 :: v_dual_lshlrev_b32 v195, 6, v0
	v_dual_mov_b32 v63, v57 :: v_dual_and_b32 v200, 30, v11
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v9
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, v3, v17, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v132, vcc_lo, s12, v1
	ds_bpermute_b32 v7, v12, v4
	ds_bpermute_b32 v9, v12, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v133, null, s13, v2, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v18
	v_lshlrev_b32_e32 v16, 2, v16
	v_lshlrev_b32_e32 v1, 2, v17
	v_lshlrev_b32_e32 v15, 4, v0
	v_and_b32_e32 v199, 16, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v3, v18, vcc_lo
	v_lshrrev_b32_e32 v8, 2, v14
	v_dual_mov_b32 v49, v57 :: v_dual_add_nc_u32 v202, 0, v6
	v_dual_mov_b32 v56, v64 :: v_dual_mov_b32 v51, v59
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v72, v64 :: v_dual_and_b32 v203, 62, v8
	v_dual_mov_b32 v69, v61 :: v_dual_mov_b32 v80, v64
	v_dual_mov_b32 v77, v61 :: v_dual_mov_b32 v88, v64
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v7
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v9
	v_or_b32_e32 v7, 0x4e00, v15
	v_or_b32_e32 v9, 0x5600, v15
	v_dual_mov_b32 v221, v186 :: v_dual_lshlrev_b32 v10, 3, v14
	ds_bpermute_b32 v25, v16, v4
	ds_bpermute_b32 v16, v16, v5
	v_add_nc_u32_e32 v214, 0, v7
	v_add_nc_u32_e32 v215, 0, v9
	v_lshlrev_b32_e32 v9, 2, v3
	v_lshl_or_b32 v12, v14, 4, 0x600
	v_dual_mov_b32 v85, v61 :: v_dual_mov_b32 v96, v64
	v_add_co_u32 v188, vcc_lo, v132, v131
	v_dual_mov_b32 v93, v61 :: v_dual_mov_b32 v104, v64
	v_dual_mov_b32 v101, v61 :: v_dual_mov_b32 v112, v64
	v_dual_mov_b32 v109, v61 :: v_dual_mov_b32 v120, v64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v189, null, 0, v133, vcc_lo
	v_add_co_u32 v190, vcc_lo, s0, v129
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v4, v25
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v5, v16
	v_dual_mov_b32 v117, v61 :: v_dual_mov_b32 v128, v64
	v_and_b32_e32 v204, 0x600, v10
	ds_bpermute_b32 v5, v1, v2
	ds_bpermute_b32 v1, v1, v4
	v_dual_mov_b32 v33, v57 :: v_dual_add_nc_u32 v206, 0, v12
	v_dual_mov_b32 v40, v64 :: v_dual_mov_b32 v41, v57
	v_dual_mov_b32 v48, v64 :: v_dual_mov_b32 v125, v61
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v191, null, s1, v130, vcc_lo
	v_dual_mov_b32 v35, v59 :: v_dual_mov_b32 v42, v58
	v_dual_mov_b32 v37, v61 :: v_dual_mov_b32 v44, v60
	v_dual_mov_b32 v39, v63 :: v_dual_mov_b32 v46, v62
	v_dual_mov_b32 v43, v59 :: v_dual_mov_b32 v50, v58
	v_dual_mov_b32 v45, v61 :: v_dual_mov_b32 v52, v60
	s_wait_dscnt 0x1
	v_max_i32_e32 v134, v2, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v135, v4, v1
	v_dual_mov_b32 v1, v57 :: v_dual_mov_b32 v2, v58
	v_or_b32_e32 v26, 0x6600, v15
	v_or_b32_e32 v27, 0x6e00, v15
	v_or_b32_e32 v28, 0x7600, v15
	v_or_b32_e32 v11, 0x600, v15
	v_or_b32_e32 v13, 0x1600, v15
	v_add_nc_u32_e32 v217, 0, v26
	v_add_nc_u32_e32 v218, 0, v27
	v_add_nc_u32_e32 v219, 0, v28
	v_dual_mov_b32 v25, v57 :: v_dual_mov_b32 v32, v64
	v_mov_b32_e32 v26, v58
	ds_bpermute_b32 v136, v9, v134
	ds_bpermute_b32 v137, v9, v135
	v_or_b32_e32 v14, 0x1e00, v15
	v_or_b32_e32 v19, 0x2600, v15
	v_or_b32_e32 v20, 0x2e00, v15
	v_or_b32_e32 v21, 0x3600, v15
	v_or_b32_e32 v22, 0x3e00, v15
	v_or_b32_e32 v23, 0x4600, v15
	v_or_b32_e32 v24, 0x5e00, v15
	v_or_b32_e32 v15, 0x7e00, v15
	v_add_nc_u32_e32 v205, 0, v11
	v_add_nc_u32_e32 v207, 0, v13
	v_dual_mov_b32 v7, v63 :: v_dual_add_nc_u32 v208, 0, v14
	v_add_nc_u32_e32 v209, 0, v19
	v_add_nc_u32_e32 v210, 0, v20
	v_add_nc_u32_e32 v211, 0, v21
	v_add_nc_u32_e32 v212, 0, v22
	v_add_nc_u32_e32 v213, 0, v23
	v_add_nc_u32_e32 v216, 0, v24
	s_wait_dscnt 0x1
	v_max_i32_e32 v131, v134, v136
	s_wait_dscnt 0x0
	v_min_i32_e32 v132, v135, v137
	v_dual_mov_b32 v9, v57 :: v_dual_add_nc_u32 v220, 0, v15
	v_mov_b32_e32 v17, v57
	v_readfirstlane_b32 s23, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_readfirstlane_b32 s24, v132
	v_dual_mov_b32 v136, v64 :: v_dual_mov_b32 v3, v59
	v_dual_mov_b32 v4, v60 :: v_dual_mov_b32 v5, v61
	v_mov_b32_e32 v6, v62
	v_mov_b32_e32 v8, v64
	v_dual_mov_b32 v10, v58 :: v_dual_mov_b32 v11, v59
	v_dual_mov_b32 v12, v60 :: v_dual_mov_b32 v13, v61
	v_dual_mov_b32 v14, v62 :: v_dual_mov_b32 v15, v63
	v_mov_b32_e32 v16, v64
	v_dual_mov_b32 v18, v58 :: v_dual_mov_b32 v19, v59
	v_dual_mov_b32 v20, v60 :: v_dual_mov_b32 v21, v61
	v_mov_b32_e32 v28, v60
	v_dual_mov_b32 v22, v62 :: v_dual_mov_b32 v23, v63
	v_mov_b32_e32 v30, v62
	v_dual_mov_b32 v24, v64 :: v_dual_mov_b32 v27, v59
	v_dual_mov_b32 v34, v58 :: v_dual_mov_b32 v29, v61
	v_dual_mov_b32 v36, v60 :: v_dual_mov_b32 v31, v63
	v_dual_mov_b32 v38, v62 :: v_dual_mov_b32 v47, v63
	v_dual_mov_b32 v54, v62 :: v_dual_mov_b32 v53, v61
	v_dual_mov_b32 v70, v62 :: v_dual_mov_b32 v55, v63
	v_dual_mov_b32 v68, v60 :: v_dual_mov_b32 v71, v63
	v_dual_mov_b32 v66, v58 :: v_dual_mov_b32 v67, v59
	v_dual_mov_b32 v78, v62 :: v_dual_mov_b32 v65, v57
	v_dual_mov_b32 v76, v60 :: v_dual_mov_b32 v79, v63
	v_dual_mov_b32 v74, v58 :: v_dual_mov_b32 v75, v59
	v_dual_mov_b32 v86, v62 :: v_dual_mov_b32 v73, v57
	v_dual_mov_b32 v84, v60 :: v_dual_mov_b32 v87, v63
	v_dual_mov_b32 v82, v58 :: v_dual_mov_b32 v83, v59
	v_dual_mov_b32 v94, v62 :: v_dual_mov_b32 v81, v57
	v_dual_mov_b32 v92, v60 :: v_dual_mov_b32 v95, v63
	v_dual_mov_b32 v90, v58 :: v_dual_mov_b32 v91, v59
	v_dual_mov_b32 v102, v62 :: v_dual_mov_b32 v89, v57
	v_dual_mov_b32 v100, v60 :: v_dual_mov_b32 v103, v63
	v_dual_mov_b32 v98, v58 :: v_dual_mov_b32 v99, v59
	v_dual_mov_b32 v110, v62 :: v_dual_mov_b32 v97, v57
	v_dual_mov_b32 v108, v60 :: v_dual_mov_b32 v111, v63
	v_dual_mov_b32 v106, v58 :: v_dual_mov_b32 v107, v59
	v_dual_mov_b32 v118, v62 :: v_dual_mov_b32 v105, v57
	v_dual_mov_b32 v116, v60 :: v_dual_mov_b32 v119, v63
	v_dual_mov_b32 v114, v58 :: v_dual_mov_b32 v115, v59
	v_dual_mov_b32 v126, v62 :: v_dual_mov_b32 v113, v57
	v_dual_mov_b32 v124, v60 :: v_dual_mov_b32 v127, v63
	v_dual_mov_b32 v122, v58 :: v_dual_mov_b32 v123, v59
	v_dual_mov_b32 v134, v62 :: v_dual_mov_b32 v121, v57
	v_dual_mov_b32 v132, v60 :: v_dual_mov_b32 v135, v63
	v_dual_mov_b32 v130, v58 :: v_dual_mov_b32 v133, v61
	v_mov_b32_e32 v131, v59
	v_mov_b32_e32 v129, v57
	s_lshl_b32 s22, ttmp7, 1
	s_ashr_i32 s5, ttmp7, 31
	s_mov_b32 s21, 0
	s_mov_b32 s25, 0
	s_ashr_i32 s26, s22, 31
	s_lshl_b64 s[12:13], s[4:5], 8
	s_branch .LBB0_12
.LBB0_10:                               ; %Flow806
                                        ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s28
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB0_11:                               ;   in Loop: Header=BB0_12 Depth=1
	s_add_co_i32 s25, s25, 1
	v_add_nc_u32_e32 v221, 64, v221
	s_cmp_eq_u32 s25, 0x3fffffff
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s21, s21, 64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s27, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_277
.LBB0_12:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_15 Depth 2
                                        ;       Child Loop BB0_22 Depth 3
                                        ;     Child Loop BB0_66 Depth 2
                                        ;     Child Loop BB0_144 Depth 2
                                        ;     Child Loop BB0_268 Depth 2
	s_lshl_b32 s3, s25, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s3, s23
	s_cselect_b32 s27, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s27
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_11
; %bb.13:                               ; %.preheader594.i.preheader
                                        ;   in Loop: Header=BB0_12 Depth=1
	s_mov_b32 s1, 0
	s_branch .LBB0_15
.LBB0_14:                               ;   in Loop: Header=BB0_15 Depth=2
	s_add_co_i32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s1, 4
	s_cbranch_scc0 .LBB0_55
.LBB0_15:                               ; %.preheader594.i
                                        ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_22 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v61, s1, 7, v0
	s_mov_b32 s0, exec_lo
	v_mov_b32_e32 v59, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v58, 3, v61
	v_add_nc_u32_e32 v60, s3, v58
	v_mov_b32_e32 v58, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s23, v60
; %bb.16:                               ;   in Loop: Header=BB0_15 Depth=2
	v_mad_co_u64_u32 v[58:59], null, 0x408, v60, s[14:15]
; %bb.17:                               ;   in Loop: Header=BB0_15 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u64_e32 vcc_lo, 0, v[58:59]
	v_mov_b32_e32 v60, 0
	s_and_saveexec_b32 s4, vcc_lo
	s_cbranch_execz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_15 Depth=2
	v_add_co_u32 v62, s0, v58, s22
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s26, v59, s0
	global_load_d16_b16 v60, v[62:63], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v60, v60.l
.LBB0_19:                               ;   in Loop: Header=BB0_15 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v58, s0, v58, s12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v62, null, s13, v59, s0
	v_lshrrev_b32_e32 v63, 2, v61
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v59, s0, v58, v198
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v61, null, 0, v62, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v63, v193 :: v_dual_and_b32 v62, 0x1e0, v63
	s_mov_b64 s[4:5], 0
	s_mov_b32 s6, 0
	s_branch .LBB0_22
.LBB0_20:                               ; %Flow856
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_or_b32 exec_lo, exec_lo, s8
.LBB0_21:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit518.i
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_and_b32_e32 v64, 0x780, v63
	s_and_b32 s0, s6, 16
	v_fma_mixlo_f16 v137, v60, v137, 0
	v_fma_mixhi_f16 v137, v60, v58, 0
	v_fma_mixlo_f16 v58, v60, v138, 0
	v_add_nc_u32_e32 v64, v64, v62
	v_fma_mixhi_f16 v58, v60, v139, 0
	v_add_nc_u32_e32 v63, 32, v63
	s_add_co_i32 s6, s6, 8
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v64, v192, s0, v64
	s_and_b32 s0, s4, 4
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s0, 1
	s_cmp_eq_u32 s4, 32
	v_lshlrev_b32_e32 v64, 4, v64
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v64, s0, 0, v64
	ds_store_2addr_b32 v64, v137, v58 offset1:1
	s_cbranch_scc1 .LBB0_14
.LBB0_22:                               ;   Parent Loop BB0_12 Depth=1
                                        ;     Parent Loop BB0_15 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v64, 0
	s_and_saveexec_b32 s7, vcc_lo
	s_cbranch_execz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v137, s0, v59, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v138, null, s5, v61, s0
	global_load_b32 v64, v[137:138], off
.LBB0_24:                               ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt 0x0
	v_bfe_i32 v58, v64, 0, 8
	v_bfe_u32 v139, v64, 3, 4
	v_and_b32_e32 v138, 7, v64
	s_mov_b32 s7, exec_lo
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v139
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_22 Depth=3
	v_cvt_f32_ubyte0_e32 v137, v138
	v_cmp_ne_u32_e64 s0, 7, v138
                                        ; implicit-def: $vgpr138
                                        ; implicit-def: $vgpr139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v137, 0x3e000000, v137, 1.0
	v_mul_f32_e32 v137, 0x43800000, v137
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v137, 0x7fc00000, v137, s0
	v_cmp_gt_i16_e64 s0, 0, v58.l
                                        ; implicit-def: $vgpr58_lo16
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v137, v137, -v137, s0
.LBB0_26:                               ; %Flow866
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_32
; %bb.27:                               ; %LeafBlock
                                        ;   in Loop: Header=BB0_22 Depth=3
	v_cmp_ne_u32_e64 s0, 0, v139
                                        ; implicit-def: $vgpr137
	s_and_saveexec_b32 s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s0, exec_lo, s8
; %bb.28:                               ;   in Loop: Header=BB0_22 Depth=3
	v_lshlrev_b32_e32 v58, 24, v64
	v_lshlrev_b32_e32 v137, 23, v139
	v_lshlrev_b32_e32 v138, 20, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v58, 0x80000000, v58
	v_or3_b32 v58, v137, v138, v58
                                        ; implicit-def: $vgpr138
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v137, 0x3c000000, v58
                                        ; implicit-def: $vgpr58_lo16
; %bb.29:                               ; %Flow864
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s0
; %bb.30:                               ;   in Loop: Header=BB0_22 Depth=3
	v_cvt_f32_ubyte0_e32 v137, v138
	v_cmp_gt_i16_e64 s0, 0, v58.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v137, 0x3b000000, v137
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v137, v137, -v137, s0
; %bb.31:                               ; %Flow865
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_or_b32 exec_lo, exec_lo, s8
.LBB0_32:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit.i
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_lshrrev_b32_e32 v139, 8, v64
	v_bfe_u32 v140, v64, 11, 4
	s_mov_b32 s7, exec_lo
                                        ; implicit-def: $vgpr58
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v138, 7, v139
	v_cmpx_lt_i32_e32 14, v140
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_22 Depth=3
	v_cvt_f32_ubyte0_e32 v58, v138
	v_bfe_i32 v139, v139, 0, 8
	v_cmp_ne_u32_e64 s0, 7, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, 0x3e000000, v58, 1.0
	v_mul_f32_e32 v140, 0x43800000, v58
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v58.l, v139.l
                                        ; implicit-def: $vgpr139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v138, 0x7fc00000, v140, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s0, 0, v58.l
                                        ; implicit-def: $vgpr140
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v58, v138, -v138, s0
                                        ; implicit-def: $vgpr138
.LBB0_34:                               ; %Flow863
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_40
; %bb.35:                               ; %LeafBlock728
                                        ;   in Loop: Header=BB0_22 Depth=3
	v_cmp_ne_u32_e64 s0, 0, v140
                                        ; implicit-def: $vgpr58
	s_and_saveexec_b32 s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s0, exec_lo, s8
; %bb.36:                               ;   in Loop: Header=BB0_22 Depth=3
	v_lshlrev_b32_e32 v58, 24, v139
	v_lshlrev_b32_e32 v139, 23, v140
	v_lshlrev_b32_e32 v138, 20, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v58, 0x80000000, v58
	v_or3_b32 v58, v139, v138, v58
                                        ; implicit-def: $vgpr139
                                        ; implicit-def: $vgpr138
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v58, 0x3c000000, v58
; %bb.37:                               ; %Flow861
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s0
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_22 Depth=3
	v_bfe_i32 v58, v139, 0, 8
	v_cvt_f32_ubyte0_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i16_e64 s0, 0, v58.l
	v_mul_f32_e32 v138, 0x3b000000, v138
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v138, -v138, s0
.LBB0_39:                               ; %Flow862
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_or_b32 exec_lo, exec_lo, s8
.LBB0_40:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit508.i
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_mov_b16_e64 v140.h, 0
	v_mov_b16_e64 v140.l, v64.h
	v_bfe_u32 v141, v64, 19, 4
	s_mov_b32 s7, exec_lo
                                        ; implicit-def: $vgpr138
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v139, 7, v140
	v_cmpx_lt_i32_e32 14, v141
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_22 Depth=3
	v_cvt_f32_ubyte0_e32 v138, v139
	v_mov_b16_e64 v140.l, v64.h
	v_cmp_ne_u32_e64 s0, 7, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v138, 0x3e000000, v138, 1.0
	v_bfe_i32 v140, v140, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v141, 0x43800000, v138
	v_mov_b16_e64 v138.l, v140.l
                                        ; implicit-def: $vgpr140
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v139, 0x7fc00000, v141, s0
	v_cmp_gt_i16_e64 s0, 0, v138.l
                                        ; implicit-def: $vgpr141
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v138, v139, -v139, s0
                                        ; implicit-def: $vgpr139
.LBB0_42:                               ; %Flow860
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_48
; %bb.43:                               ; %LeafBlock732
                                        ;   in Loop: Header=BB0_22 Depth=3
	v_cmp_ne_u32_e64 s0, 0, v141
                                        ; implicit-def: $vgpr138
	s_and_saveexec_b32 s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s0, exec_lo, s8
; %bb.44:                               ;   in Loop: Header=BB0_22 Depth=3
	v_lshlrev_b32_e32 v138, 24, v140
	v_lshlrev_b32_e32 v140, 23, v141
	v_lshlrev_b32_e32 v139, 20, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v138, 0x80000000, v138
	v_or3_b32 v138, v140, v139, v138
                                        ; implicit-def: $vgpr139
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v138, 0x3c000000, v138
; %bb.45:                               ; %Flow858
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s0
	s_cbranch_execz .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_22 Depth=3
	v_mov_b16_e64 v138.l, v64.h
	v_cvt_f32_ubyte0_e32 v139, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_bfe_i32 v138, v138, 0, 8
	v_mul_f32_e32 v139, 0x3b000000, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s0, 0, v138.l
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v138, v139, -v139, s0
.LBB0_47:                               ; %Flow859
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_or_b32 exec_lo, exec_lo, s8
.LBB0_48:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit513.i
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_bfe_u32 v141, v64, 27, 4
	v_bfe_u32 v140, v64, 24, 3
	s_mov_b32 s7, exec_lo
                                        ; implicit-def: $vgpr139
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v141
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_22 Depth=3
	v_cvt_f32_ubyte0_e32 v139, v140
	v_cmp_ne_u32_e64 s0, 7, v140
                                        ; implicit-def: $vgpr140
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v139, 0x3e000000, v139, 1.0
	v_mul_f32_e32 v139, 0x43800000, v139
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v139, 0x7fc00000, v139, s0
	v_cmp_gt_i32_e64 s0, 0, v64
                                        ; implicit-def: $vgpr64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s0
.LBB0_50:                               ; %Flow857
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_21
; %bb.51:                               ; %LeafBlock736
                                        ;   in Loop: Header=BB0_22 Depth=3
	v_cmp_ne_u32_e64 s0, 0, v141
                                        ; implicit-def: $vgpr139
	s_and_saveexec_b32 s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s0, exec_lo, s8
; %bb.52:                               ;   in Loop: Header=BB0_22 Depth=3
	v_and_b32_e32 v64, 0x80000000, v64
	v_lshlrev_b32_e32 v139, 23, v141
	v_lshlrev_b32_e32 v140, 20, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v64, v139, v140, v64
                                        ; implicit-def: $vgpr140
	v_add_nc_u32_e32 v139, 0x3c000000, v64
                                        ; implicit-def: $vgpr64
; %bb.53:                               ; %Flow855
                                        ;   in Loop: Header=BB0_22 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s0
	s_cbranch_execz .LBB0_20
; %bb.54:                               ;   in Loop: Header=BB0_22 Depth=3
	v_cvt_f32_ubyte0_e32 v139, v140
	v_cmp_gt_i32_e64 s0, 0, v64
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v139, 0x3b000000, v139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s0
	s_branch .LBB0_20
.LBB0_55:                               ; %.preheader593.preheader.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v59, 0 :: v_dual_add_nc_u32 v62, s3, v200
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v61, 0
	v_mov_b32_e32 v60, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s23, v62
; %bb.56:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_u64_u32 v[60:61], null, 0x408, v62, s[16:17]
; %bb.57:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s23, v62
; %bb.58:                               ;   in Loop: Header=BB0_12 Depth=1
	v_or_b32_e32 v58, 1, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[58:59], null, 0x408, v58, s[16:17]
; %bb.59:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[60:61]
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v63, 0
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_61
; %bb.60:                               ;   in Loop: Header=BB0_12 Depth=1
	v_add_co_u32 v63, s0, v60, s22
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v64, null, s26, v61, s0
	global_load_d16_b16 v63, v[63:64], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v63, v63.l
.LBB0_61:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_ne_u64_e64 s0, 0, v[58:59]
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB0_63
; %bb.62:                               ;   in Loop: Header=BB0_12 Depth=1
	v_add_co_u32 v137, s1, v58, s22
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v138, null, s26, v59, s1
	global_load_d16_b16 v62, v[137:138], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v62, v62.l
.LBB0_63:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v60, s1, v60, s12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v61, null, s13, v61, s1
	v_add_co_u32 v58, s1, v58, s12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v64, null, s13, v59, s1
	v_add_co_u32 v59, s1, v60, v198
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v60, null, 0, v61, s1
	v_add_co_u32 v61, s1, v58, v198
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v64, null, 0, v64, s1
	v_mov_b32_e32 v137, v195
	s_mov_b64 s[4:5], 0
	s_branch .LBB0_66
.LBB0_64:                               ; %Flow832
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_65:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit528.3.i
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_fma_mixlo_f16 v138, v63, v140, 0
	v_fma_mixhi_f16 v138, v62, v139, 0
	v_add_nc_u32_e32 v137, 8, v137
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 32
	ds_store_b32 v58, v138 offset:32816
	s_cbranch_scc1 .LBB0_133
.LBB0_66:                               ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v139, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v139, s1, v59, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v140, null, s5, v60, s1
	global_load_b32 v139, v[139:140], off
.LBB0_68:                               ; %._crit_edge24
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s0
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_66 Depth=2
	v_add_co_u32 v140, s1, v61, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v141, null, s5, v64, s1
	global_load_b32 v138, v[140:141], off
.LBB0_70:                               ; %._crit_edge784.i
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_bfe_i32 v58, v139, 0, 8
	v_bfe_u32 v142, v139, 3, 4
	v_and_b32_e32 v141, 7, v139
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr140
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v140, 0x43800000, v140
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v140, 0x7fc00000, v140, s1
	v_cmp_gt_i16_e64 s1, 0, v58.l
                                        ; implicit-def: $vgpr58_lo16
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
.LBB0_72:                               ; %Flow854
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_78
; %bb.73:                               ; %LeafBlock772
                                        ;   in Loop: Header=BB0_66 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.74:                               ;   in Loop: Header=BB0_66 Depth=2
	v_lshlrev_b32_e32 v58, 24, v139
	v_lshlrev_b32_e32 v140, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v58, 0x80000000, v58
	v_or3_b32 v58, v140, v141, v58
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v58
                                        ; implicit-def: $vgpr58_lo16
; %bb.75:                               ; %Flow852
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.76:                               ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_gt_i16_e64 s1, 0, v58.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v140, 0x3b000000, v140
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
; %bb.77:                               ; %Flow853
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_78:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit523.i
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_bfe_i32 v58, v138, 0, 8
	v_bfe_u32 v143, v138, 3, 4
	v_and_b32_e32 v142, 7, v138
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v141, v142
	v_cmp_ne_u32_e64 s1, 7, v142
                                        ; implicit-def: $vgpr142
                                        ; implicit-def: $vgpr143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v141, 0x3e000000, v141, 1.0
	v_mul_f32_e32 v141, 0x43800000, v141
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v141, 0x7fc00000, v141, s1
	v_cmp_gt_i16_e64 s1, 0, v58.l
                                        ; implicit-def: $vgpr58_lo16
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, v141, -v141, s1
.LBB0_80:                               ; %Flow851
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_86
; %bb.81:                               ; %LeafBlock776
                                        ;   in Loop: Header=BB0_66 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr141
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.82:                               ;   in Loop: Header=BB0_66 Depth=2
	v_lshlrev_b32_e32 v58, 24, v138
	v_lshlrev_b32_e32 v141, 23, v143
	v_lshlrev_b32_e32 v142, 20, v142
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v58, 0x80000000, v58
	v_or3_b32 v58, v141, v142, v58
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v141, 0x3c000000, v58
                                        ; implicit-def: $vgpr58_lo16
; %bb.83:                               ; %Flow849
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.84:                               ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v141, v142
	v_cmp_gt_i16_e64 s1, 0, v58.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, v141, -v141, s1
; %bb.85:                               ; %Flow850
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_86:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit528.i
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_b32_e32 v58, 0x1e0, v137
	v_and_or_b32 v142, s4, 12, v201
	v_fma_mixlo_f16 v140, v63, v140, 0
	v_fma_mixhi_f16 v140, v62, v141, 0
	v_bfe_u32 v143, v139, 11, 4
	s_mov_b32 s6, exec_lo
	v_or3_b32 v58, v142, v58, v199
	v_lshrrev_b32_e32 v142, 8, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v58, 4, v58
	v_and_b32_e32 v141, 7, v142
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v58, v202, v58
	ds_store_b32 v58, v140 offset:32768
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_bfe_i32 v142, v142, 0, 8
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v143, 0x43800000, v140
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v140.l, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v140.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB0_88:                               ; %Flow848
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_94
; %bb.89:                               ; %LeafBlock780
                                        ;   in Loop: Header=BB0_66 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.90:                               ;   in Loop: Header=BB0_66 Depth=2
	v_lshlrev_b32_e32 v140, 24, v142
	v_lshlrev_b32_e32 v142, 23, v143
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v140, 0x80000000, v140
	v_or3_b32 v140, v142, v141, v140
                                        ; implicit-def: $vgpr142
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v140
; %bb.91:                               ; %Flow846
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB0_93
; %bb.92:                               ;   in Loop: Header=BB0_66 Depth=2
	v_bfe_i32 v140, v142, 0, 8
	v_cvt_f32_ubyte0_e32 v141, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i16_e64 s1, 0, v140.l
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v140, v141, -v141, s1
.LBB0_93:                               ; %Flow847
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_94:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit523.1.i
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_lshrrev_b32_e32 v142, 8, v138
	v_bfe_u32 v144, v138, 11, 4
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v141, 7, v142
	v_cmpx_lt_i32_e32 14, v144
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_96
; %bb.95:                               ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v143, v141
	v_bfe_i32 v142, v142, 0, 8
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr144
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v143, 0x3e000000, v143, 1.0
	v_mul_f32_e32 v143, 0x43800000, v143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	v_cmp_gt_i16_e64 s1, 0, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v143, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB0_96:                               ; %Flow845
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_102
; %bb.97:                               ; %LeafBlock784
                                        ;   in Loop: Header=BB0_66 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v144
                                        ; implicit-def: $vgpr143
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.98:                               ;   in Loop: Header=BB0_66 Depth=2
	v_lshlrev_b32_e32 v142, 24, v142
	v_lshlrev_b32_e32 v143, 23, v144
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v142, 0x80000000, v142
	v_or3_b32 v141, v143, v141, v142
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v143, 0x3c000000, v141
                                        ; implicit-def: $vgpr141
; %bb.99:                               ; %Flow843
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.100:                              ;   in Loop: Header=BB0_66 Depth=2
	v_bfe_i32 v142, v142, 0, 8
	v_cvt_f32_ubyte0_e32 v143, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v141.l, v142.l
	v_mul_f32_e32 v142, 0x3b000000, v143
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v141.l
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v143, v142, -v142, s1
; %bb.101:                              ; %Flow844
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_102:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit528.1.i
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_mov_b16_e64 v142.h, 0
	v_mov_b16_e64 v142.l, v139.h
	v_fma_mixlo_f16 v140, v63, v140, 0
	v_fma_mixhi_f16 v140, v62, v143, 0
	v_bfe_u32 v143, v139, 19, 4
	s_mov_b32 s6, exec_lo
	v_and_b32_e32 v141, 7, v142
	ds_store_b32 v58, v140 offset:32784
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_104
; %bb.103:                              ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_mov_b16_e64 v142.l, v139.h
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_bfe_i32 v142, v142, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v143, 0x43800000, v140
	v_mov_b16_e64 v140.l, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	v_cmp_gt_i16_e64 s1, 0, v140.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v140, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB0_104:                              ; %Flow842
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_110
; %bb.105:                              ; %LeafBlock788
                                        ;   in Loop: Header=BB0_66 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.106:                              ;   in Loop: Header=BB0_66 Depth=2
	v_lshlrev_b32_e32 v140, 24, v142
	v_lshlrev_b32_e32 v142, 23, v143
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v140, 0x80000000, v140
	v_or3_b32 v140, v142, v141, v140
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v140
; %bb.107:                              ; %Flow840
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB0_109
; %bb.108:                              ;   in Loop: Header=BB0_66 Depth=2
	v_mov_b16_e64 v140.l, v139.h
	v_cvt_f32_ubyte0_e32 v141, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_bfe_i32 v140, v140, 0, 8
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v140.l
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v141, -v141, s1
.LBB0_109:                              ; %Flow841
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_110:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit523.2.i
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_mov_b16_e64 v143.h, 0
	v_mov_b16_e64 v143.l, v138.h
	v_bfe_u32 v144, v138, 19, 4
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v141, 7, v143
	v_cmpx_lt_i32_e32 14, v144
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_112
; %bb.111:                              ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v142, v141
	v_mov_b16_e64 v143.l, v138.h
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v142, 0x3e000000, v142, 1.0
	v_bfe_i32 v143, v143, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v144, 0x43800000, v142
	v_mov_b16_e64 v142.l, v143.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v141, 0x7fc00000, v144, s1
	v_cmp_gt_i16_e64 s1, 0, v142.l
                                        ; implicit-def: $vgpr144
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v142, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB0_112:                              ; %Flow839
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_118
; %bb.113:                              ; %LeafBlock792
                                        ;   in Loop: Header=BB0_66 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v144
                                        ; implicit-def: $vgpr142
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.114:                              ;   in Loop: Header=BB0_66 Depth=2
	v_lshlrev_b32_e32 v142, 24, v143
	v_lshlrev_b32_e32 v143, 23, v144
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v142, 0x80000000, v142
	v_or3_b32 v141, v143, v141, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v142, 0x3c000000, v141
                                        ; implicit-def: $vgpr141
; %bb.115:                              ; %Flow837
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB0_117
; %bb.116:                              ;   in Loop: Header=BB0_66 Depth=2
	v_mov_b16_e64 v142.l, v138.h
	v_cvt_f32_ubyte0_e32 v143, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_bfe_i32 v142, v142, 0, 8
	v_mov_b16_e64 v141.l, v142.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v142, 0x3b000000, v143
	v_cmp_gt_i16_e64 s1, 0, v141.l
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v142, v142, -v142, s1
.LBB0_117:                              ; %Flow838
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_118:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit528.2.i
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_fma_mixlo_f16 v140, v63, v140, 0
	v_fma_mixhi_f16 v140, v62, v142, 0
	v_bfe_u32 v142, v139, 27, 4
	v_bfe_u32 v141, v139, 24, 3
	s_mov_b32 s6, exec_lo
	ds_store_b32 v58, v140 offset:32800
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_120
; %bb.119:                              ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v140, 0x43800000, v140
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v140, 0x7fc00000, v140, s1
	v_cmp_gt_i32_e64 s1, 0, v139
                                        ; implicit-def: $vgpr139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
.LBB0_120:                              ; %Flow836
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_126
; %bb.121:                              ; %LeafBlock796
                                        ;   in Loop: Header=BB0_66 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.122:                              ;   in Loop: Header=BB0_66 Depth=2
	v_and_b32_e32 v139, 0x80000000, v139
	v_lshlrev_b32_e32 v140, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v139, v140, v141, v139
                                        ; implicit-def: $vgpr141
	v_add_nc_u32_e32 v140, 0x3c000000, v139
                                        ; implicit-def: $vgpr139
; %bb.123:                              ; %Flow834
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.124:                              ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_gt_i32_e64 s1, 0, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v140, 0x3b000000, v140
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
; %bb.125:                              ; %Flow835
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_126:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit523.3.i
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_bfe_u32 v142, v138, 27, 4
	v_bfe_u32 v141, v138, 24, 3
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr139
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_128
; %bb.127:                              ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v139, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v139, 0x3e000000, v139, 1.0
	v_mul_f32_e32 v139, 0x43800000, v139
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v139, 0x7fc00000, v139, s1
	v_cmp_gt_i32_e64 s1, 0, v138
                                        ; implicit-def: $vgpr138
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s1
.LBB0_128:                              ; %Flow833
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_65
; %bb.129:                              ; %LeafBlock800
                                        ;   in Loop: Header=BB0_66 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr139
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.130:                              ;   in Loop: Header=BB0_66 Depth=2
	v_and_b32_e32 v138, 0x80000000, v138
	v_lshlrev_b32_e32 v139, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v138, v139, v141, v138
                                        ; implicit-def: $vgpr141
	v_add_nc_u32_e32 v139, 0x3c000000, v138
                                        ; implicit-def: $vgpr138
; %bb.131:                              ; %Flow831
                                        ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB0_64
; %bb.132:                              ;   in Loop: Header=BB0_66 Depth=2
	v_cvt_f32_ubyte0_e32 v139, v141
	v_cmp_gt_i32_e64 s1, 0, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v139, 0x3b000000, v139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s1
	s_branch .LBB0_64
.LBB0_133:                              ; %.preheader593.1.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v59, 0 :: v_dual_add_nc_u32 v62, s3, v203
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v61, 0
	v_mov_b32_e32 v60, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s23, v62
; %bb.134:                              ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_u64_u32 v[60:61], null, 0x408, v62, s[16:17]
; %bb.135:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s23, v62
; %bb.136:                              ;   in Loop: Header=BB0_12 Depth=1
	v_or_b32_e32 v58, 1, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[58:59], null, 0x408, v58, s[16:17]
; %bb.137:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[60:61]
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v63, 0
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_139
; %bb.138:                              ;   in Loop: Header=BB0_12 Depth=1
	v_add_co_u32 v63, s0, v60, s22
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v64, null, s26, v61, s0
	global_load_d16_b16 v63, v[63:64], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v63, v63.l
.LBB0_139:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_ne_u64_e64 s0, 0, v[58:59]
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB0_141
; %bb.140:                              ;   in Loop: Header=BB0_12 Depth=1
	v_add_co_u32 v137, s1, v58, s22
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v138, null, s26, v59, s1
	global_load_d16_b16 v62, v[137:138], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v62, v62.l
.LBB0_141:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v60, s1, v60, s12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v61, null, s13, v61, s1
	v_add_co_u32 v58, s1, v58, s12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v64, null, s13, v59, s1
	v_add_co_u32 v59, s1, v60, v198
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v60, null, 0, v61, s1
	v_add_co_u32 v61, s1, v58, v198
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v64, null, 0, v64, s1
	v_mov_b32_e32 v137, v195
	s_mov_b64 s[4:5], 0
	s_branch .LBB0_144
.LBB0_142:                              ; %Flow808
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_143:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit528.3.1.i
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_fma_mixlo_f16 v138, v63, v140, 0
	v_fma_mixhi_f16 v138, v62, v139, 0
	v_add_nc_u32_e32 v137, 8, v137
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 32
	ds_store_b32 v58, v138 offset:32816
	s_cbranch_scc0 .LBB0_211
.LBB0_144:                              ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v139, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB0_146
; %bb.145:                              ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v139, s1, v59, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v140, null, s5, v60, s1
	global_load_b32 v139, v[139:140], off
.LBB0_146:                              ; %._crit_edge
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s0
	s_cbranch_execz .LBB0_148
; %bb.147:                              ;   in Loop: Header=BB0_144 Depth=2
	v_add_co_u32 v140, s1, v61, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v141, null, s5, v64, s1
	global_load_b32 v138, v[140:141], off
.LBB0_148:                              ; %._crit_edge.i
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_bfe_i32 v58, v139, 0, 8
	v_bfe_u32 v142, v139, 3, 4
	v_and_b32_e32 v141, 7, v139
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr140
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_150
; %bb.149:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v140, 0x43800000, v140
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v140, 0x7fc00000, v140, s1
	v_cmp_gt_i16_e64 s1, 0, v58.l
                                        ; implicit-def: $vgpr58_lo16
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
.LBB0_150:                              ; %Flow830
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_156
; %bb.151:                              ; %LeafBlock740
                                        ;   in Loop: Header=BB0_144 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.152:                              ;   in Loop: Header=BB0_144 Depth=2
	v_lshlrev_b32_e32 v58, 24, v139
	v_lshlrev_b32_e32 v140, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v58, 0x80000000, v58
	v_or3_b32 v58, v140, v141, v58
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v58
                                        ; implicit-def: $vgpr58_lo16
; %bb.153:                              ; %Flow828
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.154:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_gt_i16_e64 s1, 0, v58.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v140, 0x3b000000, v140
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
; %bb.155:                              ; %Flow829
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_156:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit523.1689.i
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_bfe_i32 v58, v138, 0, 8
	v_bfe_u32 v143, v138, 3, 4
	v_and_b32_e32 v142, 7, v138
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_158
; %bb.157:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v141, v142
	v_cmp_ne_u32_e64 s1, 7, v142
                                        ; implicit-def: $vgpr142
                                        ; implicit-def: $vgpr143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v141, 0x3e000000, v141, 1.0
	v_mul_f32_e32 v141, 0x43800000, v141
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v141, 0x7fc00000, v141, s1
	v_cmp_gt_i16_e64 s1, 0, v58.l
                                        ; implicit-def: $vgpr58_lo16
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, v141, -v141, s1
.LBB0_158:                              ; %Flow827
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_164
; %bb.159:                              ; %LeafBlock744
                                        ;   in Loop: Header=BB0_144 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr141
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.160:                              ;   in Loop: Header=BB0_144 Depth=2
	v_lshlrev_b32_e32 v58, 24, v138
	v_lshlrev_b32_e32 v141, 23, v143
	v_lshlrev_b32_e32 v142, 20, v142
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v58, 0x80000000, v58
	v_or3_b32 v58, v141, v142, v58
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v141, 0x3c000000, v58
                                        ; implicit-def: $vgpr58_lo16
; %bb.161:                              ; %Flow825
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.162:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v141, v142
	v_cmp_gt_i16_e64 s1, 0, v58.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, v141, -v141, s1
; %bb.163:                              ; %Flow826
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_164:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit528.1696.i
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_b32_e32 v58, 0x1e0, v137
	v_and_or_b32 v142, s4, 12, v204
	v_fma_mixlo_f16 v140, v63, v140, 0
	v_fma_mixhi_f16 v140, v62, v141, 0
	v_bfe_u32 v143, v139, 11, 4
	s_mov_b32 s6, exec_lo
	v_or3_b32 v58, v142, v58, v199
	v_lshrrev_b32_e32 v142, 8, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v58, 4, v58
	v_and_b32_e32 v141, 7, v142
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v58, v202, v58
	ds_store_b32 v58, v140 offset:32768
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_166
; %bb.165:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_bfe_i32 v142, v142, 0, 8
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v143, 0x43800000, v140
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v140.l, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v140.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB0_166:                              ; %Flow824
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_172
; %bb.167:                              ; %LeafBlock748
                                        ;   in Loop: Header=BB0_144 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.168:                              ;   in Loop: Header=BB0_144 Depth=2
	v_lshlrev_b32_e32 v140, 24, v142
	v_lshlrev_b32_e32 v142, 23, v143
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v140, 0x80000000, v140
	v_or3_b32 v140, v142, v141, v140
                                        ; implicit-def: $vgpr142
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v140
; %bb.169:                              ; %Flow822
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB0_171
; %bb.170:                              ;   in Loop: Header=BB0_144 Depth=2
	v_bfe_i32 v140, v142, 0, 8
	v_cvt_f32_ubyte0_e32 v141, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i16_e64 s1, 0, v140.l
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v140, v141, -v141, s1
.LBB0_171:                              ; %Flow823
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_172:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit523.1.1.i
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_lshrrev_b32_e32 v142, 8, v138
	v_bfe_u32 v144, v138, 11, 4
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v141, 7, v142
	v_cmpx_lt_i32_e32 14, v144
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_174
; %bb.173:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v143, v141
	v_bfe_i32 v142, v142, 0, 8
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr144
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v143, 0x3e000000, v143, 1.0
	v_mul_f32_e32 v143, 0x43800000, v143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	v_cmp_gt_i16_e64 s1, 0, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v143, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB0_174:                              ; %Flow821
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_180
; %bb.175:                              ; %LeafBlock752
                                        ;   in Loop: Header=BB0_144 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v144
                                        ; implicit-def: $vgpr143
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.176:                              ;   in Loop: Header=BB0_144 Depth=2
	v_lshlrev_b32_e32 v142, 24, v142
	v_lshlrev_b32_e32 v143, 23, v144
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v142, 0x80000000, v142
	v_or3_b32 v141, v143, v141, v142
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v143, 0x3c000000, v141
                                        ; implicit-def: $vgpr141
; %bb.177:                              ; %Flow819
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.178:                              ;   in Loop: Header=BB0_144 Depth=2
	v_bfe_i32 v142, v142, 0, 8
	v_cvt_f32_ubyte0_e32 v143, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v141.l, v142.l
	v_mul_f32_e32 v142, 0x3b000000, v143
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v141.l
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v143, v142, -v142, s1
; %bb.179:                              ; %Flow820
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_180:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit528.1.1.i
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_mov_b16_e64 v142.h, 0
	v_mov_b16_e64 v142.l, v139.h
	v_fma_mixlo_f16 v140, v63, v140, 0
	v_fma_mixhi_f16 v140, v62, v143, 0
	v_bfe_u32 v143, v139, 19, 4
	s_mov_b32 s6, exec_lo
	v_and_b32_e32 v141, 7, v142
	ds_store_b32 v58, v140 offset:32784
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_182
; %bb.181:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_mov_b16_e64 v142.l, v139.h
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_bfe_i32 v142, v142, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v143, 0x43800000, v140
	v_mov_b16_e64 v140.l, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	v_cmp_gt_i16_e64 s1, 0, v140.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v140, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB0_182:                              ; %Flow818
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_188
; %bb.183:                              ; %LeafBlock756
                                        ;   in Loop: Header=BB0_144 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.184:                              ;   in Loop: Header=BB0_144 Depth=2
	v_lshlrev_b32_e32 v140, 24, v142
	v_lshlrev_b32_e32 v142, 23, v143
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v140, 0x80000000, v140
	v_or3_b32 v140, v142, v141, v140
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v140
; %bb.185:                              ; %Flow816
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB0_187
; %bb.186:                              ;   in Loop: Header=BB0_144 Depth=2
	v_mov_b16_e64 v140.l, v139.h
	v_cvt_f32_ubyte0_e32 v141, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_bfe_i32 v140, v140, 0, 8
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v140.l
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v141, -v141, s1
.LBB0_187:                              ; %Flow817
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_188:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit523.2.1.i
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_mov_b16_e64 v143.h, 0
	v_mov_b16_e64 v143.l, v138.h
	v_bfe_u32 v144, v138, 19, 4
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v141, 7, v143
	v_cmpx_lt_i32_e32 14, v144
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_190
; %bb.189:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v142, v141
	v_mov_b16_e64 v143.l, v138.h
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v142, 0x3e000000, v142, 1.0
	v_bfe_i32 v143, v143, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v144, 0x43800000, v142
	v_mov_b16_e64 v142.l, v143.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v141, 0x7fc00000, v144, s1
	v_cmp_gt_i16_e64 s1, 0, v142.l
                                        ; implicit-def: $vgpr144
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v142, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB0_190:                              ; %Flow815
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_196
; %bb.191:                              ; %LeafBlock760
                                        ;   in Loop: Header=BB0_144 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v144
                                        ; implicit-def: $vgpr142
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.192:                              ;   in Loop: Header=BB0_144 Depth=2
	v_lshlrev_b32_e32 v142, 24, v143
	v_lshlrev_b32_e32 v143, 23, v144
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v142, 0x80000000, v142
	v_or3_b32 v141, v143, v141, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v142, 0x3c000000, v141
                                        ; implicit-def: $vgpr141
; %bb.193:                              ; %Flow813
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB0_195
; %bb.194:                              ;   in Loop: Header=BB0_144 Depth=2
	v_mov_b16_e64 v142.l, v138.h
	v_cvt_f32_ubyte0_e32 v143, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_bfe_i32 v142, v142, 0, 8
	v_mov_b16_e64 v141.l, v142.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v142, 0x3b000000, v143
	v_cmp_gt_i16_e64 s1, 0, v141.l
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v142, v142, -v142, s1
.LBB0_195:                              ; %Flow814
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_196:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit528.2.1.i
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_fma_mixlo_f16 v140, v63, v140, 0
	v_fma_mixhi_f16 v140, v62, v142, 0
	v_bfe_u32 v142, v139, 27, 4
	v_bfe_u32 v141, v139, 24, 3
	s_mov_b32 s6, exec_lo
	ds_store_b32 v58, v140 offset:32800
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_198
; %bb.197:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v140, 0x43800000, v140
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v140, 0x7fc00000, v140, s1
	v_cmp_gt_i32_e64 s1, 0, v139
                                        ; implicit-def: $vgpr139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
.LBB0_198:                              ; %Flow812
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_204
; %bb.199:                              ; %LeafBlock764
                                        ;   in Loop: Header=BB0_144 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.200:                              ;   in Loop: Header=BB0_144 Depth=2
	v_and_b32_e32 v139, 0x80000000, v139
	v_lshlrev_b32_e32 v140, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v139, v140, v141, v139
                                        ; implicit-def: $vgpr141
	v_add_nc_u32_e32 v140, 0x3c000000, v139
                                        ; implicit-def: $vgpr139
; %bb.201:                              ; %Flow810
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.202:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_gt_i32_e64 s1, 0, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v140, 0x3b000000, v140
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
; %bb.203:                              ; %Flow811
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB0_204:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit523.3.1.i
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_bfe_u32 v142, v138, 27, 4
	v_bfe_u32 v141, v138, 24, 3
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr139
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_206
; %bb.205:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v139, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v139, 0x3e000000, v139, 1.0
	v_mul_f32_e32 v139, 0x43800000, v139
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v139, 0x7fc00000, v139, s1
	v_cmp_gt_i32_e64 s1, 0, v138
                                        ; implicit-def: $vgpr138
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s1
.LBB0_206:                              ; %Flow809
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB0_143
; %bb.207:                              ; %LeafBlock768
                                        ;   in Loop: Header=BB0_144 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr139
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.208:                              ;   in Loop: Header=BB0_144 Depth=2
	v_and_b32_e32 v138, 0x80000000, v138
	v_lshlrev_b32_e32 v139, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v138, v139, v141, v138
                                        ; implicit-def: $vgpr141
	v_add_nc_u32_e32 v139, 0x3c000000, v138
                                        ; implicit-def: $vgpr138
; %bb.209:                              ; %Flow807
                                        ;   in Loop: Header=BB0_144 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB0_142
; %bb.210:                              ;   in Loop: Header=BB0_144 Depth=2
	v_cvt_f32_ubyte0_e32 v139, v141
	v_cmp_gt_i32_e64 s1, 0, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v139, 0x3b000000, v139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s1
	s_branch .LBB0_142
.LBB0_211:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s28, s2
	s_cbranch_execz .LBB0_10
; %bb.212:                              ; %.preheader591.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	v_mov_b32_e32 v169, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, v169
	v_dual_mov_b32 v175, v169 :: v_dual_mov_b32 v176, v169
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_214
; %bb.213:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[173:176], v[188:189], off
.LBB0_214:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v171, 0
	v_mov_b32_e32 v172, 0
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_216
; %bb.215:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[169:172], v[188:189], off offset:32
.LBB0_216:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v181, 0 :: v_dual_mov_b32 v184, 0
	v_mov_b32_e32 v183, 0
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_218
; %bb.217:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[181:184], v[188:189], off offset:64
.LBB0_218:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v178, 0 :: v_dual_mov_b32 v179, 0
	v_mov_b32_e32 v180, 0
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_220
; %bb.219:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[177:180], v[188:189], off offset:96
.LBB0_220:                              ; %.preheader588.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[58:61], v196
	ds_load_b128 v[145:148], v196 offset:2048
	ds_load_b128 v[149:152], v196 offset:4096
	ds_load_b128 v[153:156], v196 offset:6144
	s_or_b32 s1, s3, 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s23
	s_cselect_b32 s0, -1, 0
	s_cmp_gt_i32 s1, s23
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[137:144], v[58:61], v[173:176], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[145:148], v[169:172], v[137:144]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[137:144], v[149:152], v[181:184], v[137:144]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[153:156], v[177:180], v[137:144]
	s_cbranch_scc1 .LBB0_222
; %bb.221:                              ; %.preheader585.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[58:61], v196 offset:512
	ds_load_b128 v[153:156], v196 offset:2560
	ds_load_b128 v[157:160], v196 offset:4608
	ds_load_b128 v[161:164], v196 offset:6656
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[145:152], v[58:61], v[173:176], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[153:156], v[169:172], v[145:152]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[145:152], v[157:160], v[181:184], v[145:152]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[161:164], v[177:180], v[145:152]
	s_branch .LBB0_223
.LBB0_222:                              ;   in Loop: Header=BB0_12 Depth=1
	v_mov_b32_e32 v64, v57
	v_dual_mov_b32 v58, v57 :: v_dual_mov_b32 v59, v57
	v_dual_mov_b32 v60, v57 :: v_dual_mov_b32 v61, v57
	v_dual_mov_b32 v62, v57 :: v_dual_mov_b32 v63, v57
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v152, v64
	v_dual_mov_b32 v148, v60 :: v_dual_mov_b32 v147, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v150, v62 :: v_dual_mov_b32 v149, v61
	v_dual_mov_b32 v151, v63 :: v_dual_mov_b32 v146, v58
	v_mov_b32_e32 v145, v57
.LBB0_223:                              ; %.loopexit584.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	s_or_b32 s4, s3, 32
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s23
	s_cselect_b32 s1, -1, 0
	s_cmp_gt_i32 s4, s23
	s_cbranch_scc1 .LBB0_225
; %bb.224:                              ; %.preheader582.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[58:61], v196 offset:1024
	ds_load_b128 v[161:164], v196 offset:3072
	ds_load_b128 v[165:168], v196 offset:5120
	ds_load_b128 v[223:226], v196 offset:7168
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[153:160], v[58:61], v[173:176], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[161:164], v[169:172], v[153:160]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[153:160], v[165:168], v[181:184], v[153:160]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[223:226], v[177:180], v[153:160]
	s_branch .LBB0_226
.LBB0_225:                              ;   in Loop: Header=BB0_12 Depth=1
	v_mov_b32_e32 v64, v57
	v_dual_mov_b32 v58, v57 :: v_dual_mov_b32 v59, v57
	v_dual_mov_b32 v60, v57 :: v_dual_mov_b32 v61, v57
	v_dual_mov_b32 v62, v57 :: v_dual_mov_b32 v63, v57
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v160, v64
	v_dual_mov_b32 v156, v60 :: v_dual_mov_b32 v155, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v158, v62 :: v_dual_mov_b32 v157, v61
	v_dual_mov_b32 v159, v63 :: v_dual_mov_b32 v154, v58
	v_mov_b32_e32 v153, v57
.LBB0_226:                              ;   in Loop: Header=BB0_12 Depth=1
	s_or_b32 s4, s3, 48
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s23
	s_cselect_b32 s3, -1, 0
	s_cmp_gt_i32 s4, s23
	s_cbranch_scc1 .LBB0_228
; %bb.227:                              ; %.preheader581.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[58:61], v205
	ds_load_b128 v[223:226], v206
	ds_load_b128 v[227:230], v207
	ds_load_b128 v[231:234], v208
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[161:168], v[58:61], v[173:176], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[223:226], v[169:172], v[161:168]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[161:168], v[227:230], v[181:184], v[161:168]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[231:234], v[177:180], v[161:168]
	s_branch .LBB0_229
.LBB0_228:                              ;   in Loop: Header=BB0_12 Depth=1
	v_mov_b32_e32 v64, v57
	v_dual_mov_b32 v58, v57 :: v_dual_mov_b32 v59, v57
	v_dual_mov_b32 v60, v57 :: v_dual_mov_b32 v61, v57
	v_dual_mov_b32 v62, v57 :: v_dual_mov_b32 v63, v57
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v168, v64
	v_dual_mov_b32 v164, v60 :: v_dual_mov_b32 v163, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v166, v62 :: v_dual_mov_b32 v165, v61
	v_dual_mov_b32 v167, v63 :: v_dual_mov_b32 v162, v58
	v_mov_b32_e32 v161, v57
.LBB0_229:                              ; %.preheader589.1.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v171, 0
	v_mov_b32_e32 v172, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s4, s20
	s_cbranch_execz .LBB0_231
; %bb.230:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[169:172], v[188:189], off offset:128
.LBB0_231:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v60, 0
	v_mov_b32_e32 v61, 0
	s_and_saveexec_b32 s4, s20
	s_cbranch_execz .LBB0_233
; %bb.232:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[58:61], v[188:189], off offset:160
.LBB0_233:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v180, 0
	v_mov_b32_e32 v179, 0
	s_and_saveexec_b32 s4, s20
	s_cbranch_execz .LBB0_235
; %bb.234:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[177:180], v[188:189], off offset:192
.LBB0_235:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	v_mov_b32_e32 v176, 0
	s_and_saveexec_b32 s4, s20
	s_cbranch_execz .LBB0_237
; %bb.236:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[173:176], v[188:189], off offset:224
.LBB0_237:                              ; %.preheader588.1.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	ds_load_b128 v[181:184], v196 offset:8192
	ds_load_b128 v[223:226], v196 offset:10240
	ds_load_b128 v[227:230], v196 offset:12288
	ds_load_b128 v[231:234], v196 offset:14336
	v_cndmask_b32_e64 v62, 0, 1, s0
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[137:144], v[181:184], v[169:172], v[137:144]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[223:226], v[58:61], v[137:144]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[137:144], v[227:230], v[177:180], v[137:144]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[231:234], v[173:176], v[137:144]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_271
; %bb.238:                              ; %.loopexit584.1.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	v_cndmask_b32_e64 v63, 0, 1, s1
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_272
.LBB0_239:                              ;   in Loop: Header=BB0_12 Depth=1
	v_cndmask_b32_e64 v64, 0, 1, s3
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_241
.LBB0_240:                              ; %.preheader581.1.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[181:184], v209
	ds_load_b128 v[223:226], v210
	ds_load_b128 v[227:230], v211
	ds_load_b128 v[231:234], v212
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[161:168], v[181:184], v[169:172], v[161:168]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[223:226], v[58:61], v[161:168]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[161:168], v[227:230], v[177:180], v[161:168]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[231:234], v[173:176], v[161:168]
.LBB0_241:                              ; %.preheader589.2.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v171, 0
	v_mov_b32_e32 v172, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_243
; %bb.242:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[169:172], v[188:189], off offset:256
.LBB0_243:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v60, 0
	v_mov_b32_e32 v61, 0
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_245
; %bb.244:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[58:61], v[188:189], off offset:288
.LBB0_245:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v180, 0
	v_mov_b32_e32 v179, 0
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_247
; %bb.246:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[177:180], v[188:189], off offset:320
.LBB0_247:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	v_mov_b32_e32 v176, 0
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_249
; %bb.248:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[173:176], v[188:189], off offset:352
.LBB0_249:                              ; %.preheader588.2.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[181:184], v196 offset:16384
	ds_load_b128 v[223:226], v196 offset:18432
	ds_load_b128 v[227:230], v196 offset:20480
	ds_load_b128 v[231:234], v196 offset:22528
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[137:144], v[181:184], v[169:172], v[137:144]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[223:226], v[58:61], v[137:144]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[137:144], v[227:230], v[177:180], v[137:144]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[231:234], v[173:176], v[137:144]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_273
; %bb.250:                              ; %.loopexit584.2.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v63
	s_cbranch_vccz .LBB0_274
.LBB0_251:                              ;   in Loop: Header=BB0_12 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v64
	s_cbranch_vccnz .LBB0_253
.LBB0_252:                              ; %.preheader581.2.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[181:184], v213
	ds_load_b128 v[223:226], v214
	ds_load_b128 v[227:230], v215
	ds_load_b128 v[231:234], v216
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[161:168], v[181:184], v[169:172], v[161:168]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[223:226], v[58:61], v[161:168]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[161:168], v[227:230], v[177:180], v[161:168]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[231:234], v[173:176], v[161:168]
.LBB0_253:                              ; %.preheader589.3.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v171, 0
	v_mov_b32_e32 v172, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_255
; %bb.254:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[169:172], v[188:189], off offset:384
.LBB0_255:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v60, 0
	v_mov_b32_e32 v61, 0
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_257
; %bb.256:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[58:61], v[188:189], off offset:416
.LBB0_257:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v180, 0
	v_mov_b32_e32 v179, 0
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_259
; %bb.258:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[177:180], v[188:189], off offset:448
.LBB0_259:                              ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	v_mov_b32_e32 v176, 0
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_261
; %bb.260:                              ;   in Loop: Header=BB0_12 Depth=1
	global_load_b128 v[173:176], v[188:189], off offset:480
.LBB0_261:                              ; %.preheader588.3.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[181:184], v196 offset:24576
	ds_load_b128 v[223:226], v196 offset:26624
	ds_load_b128 v[227:230], v196 offset:28672
	ds_load_b128 v[231:234], v196 offset:30720
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[137:144], v[181:184], v[169:172], v[137:144]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[223:226], v[58:61], v[137:144]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[137:144], v[227:230], v[177:180], v[137:144]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[231:234], v[173:176], v[137:144]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_275
; %bb.262:                              ; %.loopexit584.3.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v63
	s_cbranch_vccz .LBB0_276
.LBB0_263:                              ;   in Loop: Header=BB0_12 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v64
	s_cbranch_vccnz .LBB0_265
.LBB0_264:                              ; %.preheader581.3.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[181:184], v217
	ds_load_b128 v[223:226], v218
	ds_load_b128 v[227:230], v219
	ds_load_b128 v[231:234], v220
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[161:168], v[181:184], v[169:172], v[161:168]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[223:226], v[58:61], v[161:168]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[161:168], v[227:230], v[177:180], v[161:168]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[231:234], v[173:176], v[161:168]
.LBB0_265:                              ; %.loopexit592.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	v_mov_b32_e32 v58, v196
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s29, 0
	s_mov_b32 s30, 0
	s_branch .LBB0_268
.LBB0_266:                              ; %.critedge.i
                                        ;   in Loop: Header=BB0_268 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v60, s29, v221
	s_cmp_lt_u32 s30, 2
	v_cndmask_b32_e64 v169, v166, v158, s1
	s_cselect_b32 s10, -1, 0
	v_cndmask_b32_e64 v170, v165, v157, s1
	v_add_nc_u32_e32 v61, 2, v60
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s7, v60, v59
	v_cmp_lt_i32_e64 s8, v60, v59
	v_cndmask_b32_e64 v171, v164, v156, s1
	v_cndmask_b32_e64 v172, v162, v154, s1
	v_cmp_le_i32_e32 vcc_lo, v61, v59
	v_add_nc_u32_e32 v61, 3, v60
	v_cndmask_b32_e64 v173, v161, v153, s1
	v_cndmask_b32_e64 v174, v163, v155, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, v61, v59
	v_add_nc_u32_e32 v61, 4, v60
	v_cmp_le_i32_e64 s4, v61, v59
	v_add_nc_u32_e32 v61, 5, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cmp_le_i32_e64 s5, v61, v59
	v_add_nc_u32_e32 v61, 6, v60
	v_add_nc_u32_e32 v60, 7, v60
	v_cmp_le_i32_e64 s6, v61, v59
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_i32_e64 s9, v60, v59
	v_cndmask_b32_e64 v59, v152, v144, s0
	v_cndmask_b32_e64 v60, v168, v160, s1
	v_cndmask_b32_e64 v61, v167, v159, s1
	s_or_b32 s1, s31, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v59, v60, v59, s10
	v_cndmask_b32_e64 v60, v151, v143, s0
	v_mul_f32_e32 v59, s11, v59
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v60, v61, v60, s10
	v_cndmask_b32_e64 v61, v150, v142, s0
	v_mul_f32_e32 v60, s11, v60
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v61, v169, v61, s10
	v_cndmask_b32_e64 v169, v149, v141, s0
	v_mul_f32_e32 v61, s11, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v169, v170, v169, s10
	v_cndmask_b32_e64 v170, v148, v140, s0
	v_mul_f32_e32 v169, s11, v169
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v170, v171, v170, s10
	v_cndmask_b32_e64 v171, v146, v138, s0
	v_mul_f32_e32 v170, s11, v170
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v171, v172, v171, s10
	v_cndmask_b32_e64 v172, v145, v137, s0
	v_mul_f32_e32 v171, s11, v171
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v172, v173, v172, s10
	v_cndmask_b32_e64 v173, v147, v139, s0
	s_or_b32 s0, s31, s7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s20, s0
	v_mul_f32_e32 v172, s11, v172
	v_cndmask_b32_e64 v173, v174, v173, s10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v172, 0xff800000, v172, s0
	s_and_b32 s0, s20, s1
	v_mul_f32_e32 v173, s11, v173
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v171, 0xff800000, v171, s0
	s_or_b32 s0, s31, vcc_lo
	s_or_b32 s1, s31, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s20, s0
	s_or_b32 s0, s31, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v173, 0xff800000, v173, vcc_lo
	s_and_b32 vcc_lo, s20, s1
	s_or_b32 s1, s31, s5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v170, 0xff800000, v170, vcc_lo
	s_and_b32 vcc_lo, s20, s0
	v_max3_num_f32 v174, v172, 0xff800000, v171
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v169, 0xff800000, v169, vcc_lo
	s_and_b32 vcc_lo, s20, s1
	s_or_b32 s0, s31, s6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v61, 0xff800000, v61, vcc_lo
	s_and_b32 vcc_lo, s20, s0
	s_or_b32 s1, s31, s9
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v60, 0xff800000, v60, vcc_lo
	v_max3_num_f32 v174, v174, v173, v170
	s_and_b32 vcc_lo, s20, s1
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v175, 0xff800000, v59, vcc_lo
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v222
	v_max3_num_f32 v59, v174, v169, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v59, v59, v60, v175
	ds_bpermute_b32 v174, v194, v59
	s_wait_dscnt 0x0
	v_max3_num_f32 v59, v222, v59, v174
	v_dual_sub_f32 v170, v170, v59 :: v_dual_sub_f32 v171, v171, v59
	v_sub_f32_e32 v60, v60, v59
	v_cmp_eq_f32_e64 s0, 0xff800000, v59
	v_sub_f32_e32 v174, v222, v59
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v170, 0x3fb8aa3b, v170 :: v_dual_mul_f32 v171, 0x3fb8aa3b, v171
	v_mul_f32_e32 v60, 0x3fb8aa3b, v60
	v_sub_f32_e32 v172, v172, v59
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v174, 0x3fb8aa3b, v174 :: v_dual_sub_f32 v169, v169, v59
	v_exp_f32_e32 v171, v171
	v_dual_sub_f32 v61, v61, v59 :: v_dual_mov_b32 v222, v59
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v174, v174
	v_mul_f32_e32 v172, 0x3fb8aa3b, v172
	v_exp_f32_e32 v170, v170
	v_exp_f32_e32 v60, v60
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v178, v171, 0, s0
	v_sub_f32_e32 v171, v175, v59
	v_exp_f32_e32 v172, v172
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, 0, v174, vcc_lo
	v_mul_f32_e32 v169, 0x3fb8aa3b, v169
	v_mul_f32_e32 v61, 0x3fb8aa3b, v61
	v_mul_f32_e32 v171, 0x3fb8aa3b, v171
	v_cndmask_b32_e64 v180, v170, 0, s0
	v_mul_f32_e32 v136, v136, v183
	v_mul_f32_e32 v134, v134, v183
	v_mul_f32_e32 v132, v132, v183
	v_exp_f32_e32 v171, v171
	v_cndmask_b32_e64 v177, v172, 0, s0
	v_exp_f32_e32 v172, v169
	v_exp_f32_e32 v61, v61
	v_cndmask_b32_e64 v60, v60, 0, s0
	v_cvt_f16_f32_e64 v169.h, v178
	v_cvt_f16_f32_e64 v169.l, v177
	v_cvt_f16_f32_e64 v170.h, v180
	v_dual_mul_f32 v135, v135, v183 :: v_dual_mul_f32 v130, v130, v183
	v_cndmask_b32_e64 v182, v171, 0, s0
	v_sub_f32_e32 v173, v173, v59
	v_cndmask_b32_e64 v181, v172, 0, s0
	v_cndmask_b32_e64 v61, v61, 0, s0
	v_cvt_f16_f32_e64 v172.l, v60
	v_cvt_f16_f32_e64 v172.h, v182
	v_mul_f32_e32 v173, 0x3fb8aa3b, v173
	v_cvt_f16_f32_e64 v171.l, v181
	v_cvt_f16_f32_e64 v171.h, v61
	v_dual_mul_f32 v133, v133, v183 :: v_dual_mul_f32 v128, v128, v183
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v173, v173
	v_dual_mul_f32 v131, v131, v183 :: v_dual_mul_f32 v126, v126, v183
	v_dual_mul_f32 v129, v129, v183 :: v_dual_mul_f32 v124, v124, v183
	v_dual_mul_f32 v127, v127, v183 :: v_dual_mul_f32 v122, v122, v183
	v_dual_mul_f32 v125, v125, v183 :: v_dual_mul_f32 v120, v120, v183
	v_cndmask_b32_e64 v179, v173, 0, s0
	ds_load_b128 v[173:176], v58 offset:32768
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v123, v123, v183 :: v_dual_mul_f32 v118, v118, v183
	v_cvt_f16_f32_e64 v170.l, v179
	v_dual_mul_f32 v121, v121, v183 :: v_dual_mul_f32 v116, v116, v183
	v_dual_mul_f32 v119, v119, v183 :: v_dual_mul_f32 v114, v114, v183
	v_dual_mul_f32 v117, v117, v183 :: v_dual_mul_f32 v112, v112, v183
	v_dual_mul_f32 v115, v115, v183 :: v_dual_mul_f32 v110, v110, v183
	v_dual_mul_f32 v113, v113, v183 :: v_dual_mul_f32 v108, v108, v183
	v_dual_mul_f32 v111, v111, v183 :: v_dual_mul_f32 v106, v106, v183
	v_dual_mul_f32 v109, v109, v183 :: v_dual_mul_f32 v104, v104, v183
	v_dual_mul_f32 v107, v107, v183 :: v_dual_mul_f32 v102, v102, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[129:136], v[173:176], v[169:172], v[129:136]
	ds_load_b128 v[173:176], v58 offset:33280
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v105, v105, v183 :: v_dual_mul_f32 v100, v100, v183
	v_dual_mul_f32 v103, v103, v183 :: v_dual_mul_f32 v98, v98, v183
	v_dual_mul_f32 v101, v101, v183 :: v_dual_mul_f32 v96, v96, v183
	v_dual_mul_f32 v99, v99, v183 :: v_dual_mul_f32 v94, v94, v183
	v_dual_mul_f32 v97, v97, v183 :: v_dual_mul_f32 v92, v92, v183
	v_dual_mul_f32 v95, v95, v183 :: v_dual_mul_f32 v90, v90, v183
	v_dual_mul_f32 v93, v93, v183 :: v_dual_mul_f32 v88, v88, v183
	v_dual_mul_f32 v91, v91, v183 :: v_dual_mul_f32 v86, v86, v183
	v_dual_mul_f32 v89, v89, v183 :: v_dual_mul_f32 v84, v84, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[173:176], v[169:172], v[121:128]
	ds_load_b128 v[173:176], v58 offset:33792
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v87, v87, v183 :: v_dual_mul_f32 v82, v82, v183
	v_dual_mul_f32 v85, v85, v183 :: v_dual_mul_f32 v80, v80, v183
	v_dual_mul_f32 v83, v83, v183 :: v_dual_mul_f32 v78, v78, v183
	v_dual_mul_f32 v81, v81, v183 :: v_dual_mul_f32 v76, v76, v183
	v_dual_mul_f32 v79, v79, v183 :: v_dual_mul_f32 v74, v74, v183
	v_dual_mul_f32 v77, v77, v183 :: v_dual_mul_f32 v72, v72, v183
	v_dual_mul_f32 v75, v75, v183 :: v_dual_mul_f32 v70, v70, v183
	v_dual_mul_f32 v73, v73, v183 :: v_dual_mul_f32 v68, v68, v183
	v_dual_mul_f32 v71, v71, v183 :: v_dual_mul_f32 v66, v66, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[113:120], v[173:176], v[169:172], v[113:120]
	ds_load_b128 v[173:176], v58 offset:34304
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v69, v69, v183 :: v_dual_mul_f32 v56, v56, v183
	v_dual_mul_f32 v67, v67, v183 :: v_dual_mul_f32 v54, v54, v183
	v_dual_mul_f32 v65, v65, v183 :: v_dual_mul_f32 v52, v52, v183
	v_dual_mul_f32 v55, v55, v183 :: v_dual_mul_f32 v50, v50, v183
	v_dual_mul_f32 v53, v53, v183 :: v_dual_mul_f32 v48, v48, v183
	v_dual_mul_f32 v51, v51, v183 :: v_dual_mul_f32 v46, v46, v183
	v_dual_mul_f32 v49, v49, v183 :: v_dual_mul_f32 v44, v44, v183
	v_dual_mul_f32 v47, v47, v183 :: v_dual_mul_f32 v42, v42, v183
	v_dual_mul_f32 v45, v45, v183 :: v_dual_mul_f32 v40, v40, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[105:112], v[173:176], v[169:172], v[105:112]
	ds_load_b128 v[173:176], v58 offset:34816
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v43, v43, v183 :: v_dual_mul_f32 v38, v38, v183
	v_dual_mul_f32 v41, v41, v183 :: v_dual_mul_f32 v36, v36, v183
	v_dual_mul_f32 v39, v39, v183 :: v_dual_mul_f32 v34, v34, v183
	v_dual_mul_f32 v37, v37, v183 :: v_dual_mul_f32 v32, v32, v183
	v_dual_mul_f32 v35, v35, v183 :: v_dual_mul_f32 v30, v30, v183
	v_dual_mul_f32 v33, v33, v183 :: v_dual_mul_f32 v28, v28, v183
	v_dual_mul_f32 v31, v31, v183 :: v_dual_mul_f32 v26, v26, v183
	v_dual_mul_f32 v29, v29, v183 :: v_dual_mul_f32 v24, v24, v183
	v_dual_mul_f32 v27, v27, v183 :: v_dual_mul_f32 v22, v22, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[97:104], v[173:176], v[169:172], v[97:104]
	ds_load_b128 v[173:176], v58 offset:35328
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v25, v25, v183 :: v_dual_mul_f32 v20, v20, v183
	v_dual_mul_f32 v23, v23, v183 :: v_dual_mul_f32 v18, v18, v183
	v_dual_mul_f32 v21, v21, v183 :: v_dual_mul_f32 v16, v16, v183
	v_dual_mul_f32 v19, v19, v183 :: v_dual_mul_f32 v14, v14, v183
	v_dual_mul_f32 v17, v17, v183 :: v_dual_mul_f32 v12, v12, v183
	v_dual_mul_f32 v15, v15, v183 :: v_dual_mul_f32 v10, v10, v183
	v_dual_mul_f32 v13, v13, v183 :: v_dual_mul_f32 v8, v8, v183
	v_dual_mul_f32 v11, v11, v183 :: v_dual_mul_f32 v6, v6, v183
	v_dual_mul_f32 v9, v9, v183 :: v_dual_mul_f32 v4, v4, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[173:176], v[169:172], v[89:96]
	ds_load_b128 v[173:176], v58 offset:35840
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v7, v7, v183 :: v_dual_mul_f32 v2, v2, v183
	v_mul_f32_e32 v5, v5, v183
	v_mul_f32_e32 v3, v3, v183
	v_mul_f32_e32 v1, v1, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[81:88], v[173:176], v[169:172], v[81:88]
	ds_load_b128 v[173:176], v58 offset:36352
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[73:80], v[173:176], v[169:172], v[73:80]
	ds_load_b128 v[173:176], v58 offset:36864
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[65:72], v[173:176], v[169:172], v[65:72]
	ds_load_b128 v[173:176], v58 offset:37376
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[49:56], v[173:176], v[169:172], v[49:56]
	ds_load_b128 v[173:176], v58 offset:37888
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[41:48], v[173:176], v[169:172], v[41:48]
	ds_load_b128 v[173:176], v58 offset:38400
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[33:40], v[173:176], v[169:172], v[33:40]
	ds_load_b128 v[173:176], v58 offset:38912
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[173:176], v[169:172], v[25:32]
	ds_load_b128 v[173:176], v58 offset:39424
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[17:24], v[173:176], v[169:172], v[17:24]
	ds_load_b128 v[173:176], v58 offset:39936
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[9:16], v[173:176], v[169:172], v[9:16]
	ds_load_b128 v[173:176], v58 offset:40448
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[173:176], v[169:172], v[1:8]
	v_add_f32_e32 v169, v177, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v169, v179, v169
	v_add_f32_e32 v169, v180, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v169, v181, v169
	v_add_f32_e32 v61, v61, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v61
	v_add_f32_e32 v60, v182, v60
	ds_bpermute_b32 v61, v194, v60
	s_wait_dscnt 0x0
	v_add_f32_e32 v60, v60, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v60, v197, v183
	v_mov_b32_e32 v197, v60
.LBB0_267:                              ; %.loopexit579.i
                                        ;   in Loop: Header=BB0_268 Depth=2
	v_add_nc_u32_e32 v58, 0x2000, v58
	s_add_co_i32 s29, s29, 16
	s_add_co_i32 s30, s30, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s29, 64
	s_cbranch_scc0 .LBB0_10
.LBB0_268:                              ;   Parent Loop BB0_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s29, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s29, 32
	s_cselect_b32 s1, -1, 0
	s_cmp_eq_u32 s29, 16
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v59, v64, v63, s1
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v59, v59, v62, vcc_lo
	v_and_b32_e32 v59, 1, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 1, v59
	s_or_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_267
; %bb.269:                              ;   in Loop: Header=BB0_268 Depth=2
	s_add_co_i32 s3, s21, s29
	v_mov_b32_e32 v59, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s3, 15
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s24
	s_cselect_b32 s31, -1, 0
	s_cmp_gt_i32 s3, s24
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s3, s20
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB0_266
; %bb.270:                              ;   in Loop: Header=BB0_268 Depth=2
	global_load_b32 v59, v[190:191], off
	s_branch .LBB0_266
.LBB0_271:                              ; %.preheader585.1.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[181:184], v196 offset:8704
	ds_load_b128 v[223:226], v196 offset:10752
	ds_load_b128 v[227:230], v196 offset:12800
	ds_load_b128 v[231:234], v196 offset:14848
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[145:152], v[181:184], v[169:172], v[145:152]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[223:226], v[58:61], v[145:152]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[145:152], v[227:230], v[177:180], v[145:152]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[231:234], v[173:176], v[145:152]
	v_cndmask_b32_e64 v63, 0, 1, s1
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_239
.LBB0_272:                              ; %.preheader582.1.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[181:184], v196 offset:9216
	ds_load_b128 v[223:226], v196 offset:11264
	ds_load_b128 v[227:230], v196 offset:13312
	ds_load_b128 v[231:234], v196 offset:15360
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[153:160], v[181:184], v[169:172], v[153:160]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[223:226], v[58:61], v[153:160]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[153:160], v[227:230], v[177:180], v[153:160]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[231:234], v[173:176], v[153:160]
	v_cndmask_b32_e64 v64, 0, 1, s3
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_240
	s_branch .LBB0_241
.LBB0_273:                              ; %.preheader585.2.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[181:184], v196 offset:16896
	ds_load_b128 v[223:226], v196 offset:18944
	ds_load_b128 v[227:230], v196 offset:20992
	ds_load_b128 v[231:234], v196 offset:23040
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[145:152], v[181:184], v[169:172], v[145:152]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[223:226], v[58:61], v[145:152]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[145:152], v[227:230], v[177:180], v[145:152]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[231:234], v[173:176], v[145:152]
	v_cmp_ne_u32_e32 vcc_lo, 1, v63
	s_cbranch_vccnz .LBB0_251
.LBB0_274:                              ; %.preheader582.2.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[181:184], v196 offset:17408
	ds_load_b128 v[223:226], v196 offset:19456
	ds_load_b128 v[227:230], v196 offset:21504
	ds_load_b128 v[231:234], v196 offset:23552
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[153:160], v[181:184], v[169:172], v[153:160]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[223:226], v[58:61], v[153:160]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[153:160], v[227:230], v[177:180], v[153:160]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[231:234], v[173:176], v[153:160]
	v_cmp_ne_u32_e32 vcc_lo, 1, v64
	s_cbranch_vccz .LBB0_252
	s_branch .LBB0_253
.LBB0_275:                              ; %.preheader585.3.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[181:184], v196 offset:25088
	ds_load_b128 v[223:226], v196 offset:27136
	ds_load_b128 v[227:230], v196 offset:29184
	ds_load_b128 v[231:234], v196 offset:31232
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[145:152], v[181:184], v[169:172], v[145:152]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[223:226], v[58:61], v[145:152]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[145:152], v[227:230], v[177:180], v[145:152]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[231:234], v[173:176], v[145:152]
	v_cmp_ne_u32_e32 vcc_lo, 1, v63
	s_cbranch_vccnz .LBB0_263
.LBB0_276:                              ; %.preheader582.3.i
                                        ;   in Loop: Header=BB0_12 Depth=1
	ds_load_b128 v[181:184], v196 offset:25600
	ds_load_b128 v[223:226], v196 offset:27648
	ds_load_b128 v[227:230], v196 offset:29696
	ds_load_b128 v[231:234], v196 offset:31744
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[153:160], v[181:184], v[169:172], v[153:160]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[223:226], v[58:61], v[153:160]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[153:160], v[227:230], v[177:180], v[153:160]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[231:234], v[173:176], v[153:160]
	v_cmp_ne_u32_e32 vcc_lo, 1, v64
	s_cbranch_vccz .LBB0_264
	s_branch .LBB0_265
.LBB0_277:
	s_and_saveexec_b32 s0, s20
	s_cbranch_execz .LBB0_279
; %bb.278:                              ; %.loopexit.loopexit.i
	v_div_scale_f32 v0, null, v197, v197, 1.0
	v_div_scale_f32 v60, vcc_lo, 1.0, v197, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v59, v0
	v_fma_f32 v57, -v0, v59, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v59, v57, v59
	v_mul_f32_e32 v61, v60, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v57, -v0, v61, v60
	v_fmac_f32_e32 v61, v57, v59
	v_mad_co_u64_u32 v[57:58], null, 0x1800, v185, v[187:188]
	v_mov_b32_e32 v58, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v0, -v0, v61, v60
	v_or_b32_e32 v57, v57, v186
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v0, v0, v59, v61
	v_cmp_lt_f32_e32 vcc_lo, 0, v197
	v_lshlrev_b64_e32 v[57:58], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v0, v0, v197, 1.0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v139, 0, v0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v137, vcc_lo, s18, v57
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, s19, v58, vcc_lo
	v_dual_mul_f32 v57, v129, v139 :: v_dual_mul_f32 v58, v130, v139
	v_dual_mul_f32 v59, v131, v139 :: v_dual_mul_f32 v60, v132, v139
	v_dual_mul_f32 v61, v133, v139 :: v_dual_mul_f32 v62, v134, v139
	v_dual_mul_f32 v63, v135, v139 :: v_dual_mul_f32 v64, v136, v139
	v_dual_mul_f32 v121, v121, v139 :: v_dual_mul_f32 v122, v122, v139
	v_dual_mul_f32 v123, v123, v139 :: v_dual_mul_f32 v124, v124, v139
	v_dual_mul_f32 v125, v125, v139 :: v_dual_mul_f32 v126, v126, v139
	v_dual_mul_f32 v127, v127, v139 :: v_dual_mul_f32 v128, v128, v139
	v_dual_mul_f32 v113, v113, v139 :: v_dual_mul_f32 v114, v114, v139
	v_dual_mul_f32 v115, v115, v139 :: v_dual_mul_f32 v116, v116, v139
	v_dual_mul_f32 v97, v97, v139 :: v_dual_mul_f32 v98, v98, v139
	v_dual_mul_f32 v99, v99, v139 :: v_dual_mul_f32 v100, v100, v139
	v_dual_mul_f32 v117, v117, v139 :: v_dual_mul_f32 v118, v118, v139
	v_dual_mul_f32 v119, v119, v139 :: v_dual_mul_f32 v120, v120, v139
	v_dual_mul_f32 v105, v105, v139 :: v_dual_mul_f32 v106, v106, v139
	v_dual_mul_f32 v107, v107, v139 :: v_dual_mul_f32 v108, v108, v139
	v_dual_mul_f32 v109, v109, v139 :: v_dual_mul_f32 v110, v110, v139
	v_dual_mul_f32 v111, v111, v139 :: v_dual_mul_f32 v112, v112, v139
	s_clause 0x7
	global_store_b128 v[137:138], v[57:60], off
	global_store_b128 v[137:138], v[61:64], off offset:16
	global_store_b128 v[137:138], v[121:124], off offset:64
	global_store_b128 v[137:138], v[125:128], off offset:80
	global_store_b128 v[137:138], v[113:116], off offset:128
	global_store_b128 v[137:138], v[117:120], off offset:144
	global_store_b128 v[137:138], v[105:108], off offset:192
	global_store_b128 v[137:138], v[109:112], off offset:208
	v_dual_mul_f32 v57, v101, v139 :: v_dual_mul_f32 v58, v102, v139
	v_dual_mul_f32 v59, v103, v139 :: v_dual_mul_f32 v60, v104, v139
	v_dual_mul_f32 v61, v89, v139 :: v_dual_mul_f32 v62, v90, v139
	v_dual_mul_f32 v63, v91, v139 :: v_dual_mul_f32 v64, v92, v139
	v_dual_mul_f32 v89, v93, v139 :: v_dual_mul_f32 v90, v94, v139
	v_dual_mul_f32 v91, v95, v139 :: v_dual_mul_f32 v92, v96, v139
	v_dual_mul_f32 v81, v81, v139 :: v_dual_mul_f32 v82, v82, v139
	v_dual_mul_f32 v83, v83, v139 :: v_dual_mul_f32 v84, v84, v139
	v_dual_mul_f32 v85, v85, v139 :: v_dual_mul_f32 v86, v86, v139
	v_dual_mul_f32 v87, v87, v139 :: v_dual_mul_f32 v88, v88, v139
	s_clause 0x5
	global_store_b128 v[137:138], v[97:100], off offset:256
	global_store_b128 v[137:138], v[57:60], off offset:272
	global_store_b128 v[137:138], v[61:64], off offset:320
	global_store_b128 v[137:138], v[89:92], off offset:336
	global_store_b128 v[137:138], v[81:84], off offset:384
	global_store_b128 v[137:138], v[85:88], off offset:400
	v_dual_mul_f32 v57, v73, v139 :: v_dual_mul_f32 v58, v74, v139
	v_dual_mul_f32 v59, v75, v139 :: v_dual_mul_f32 v60, v76, v139
	v_dual_mul_f32 v41, v41, v139 :: v_dual_mul_f32 v42, v42, v139
	v_dual_mul_f32 v43, v43, v139 :: v_dual_mul_f32 v44, v44, v139
	v_dual_mul_f32 v17, v17, v139 :: v_dual_mul_f32 v18, v18, v139
	v_dual_mul_f32 v19, v19, v139 :: v_dual_mul_f32 v20, v20, v139
	v_dual_mul_f32 v61, v77, v139 :: v_dual_mul_f32 v62, v78, v139
	v_dual_mul_f32 v63, v79, v139 :: v_dual_mul_f32 v64, v80, v139
	v_dual_mul_f32 v45, v45, v139 :: v_dual_mul_f32 v46, v46, v139
	v_dual_mul_f32 v47, v47, v139 :: v_dual_mul_f32 v48, v48, v139
	v_dual_mul_f32 v21, v21, v139 :: v_dual_mul_f32 v22, v22, v139
	v_dual_mul_f32 v23, v23, v139 :: v_dual_mul_f32 v24, v24, v139
	v_dual_mul_f32 v65, v65, v139 :: v_dual_mul_f32 v66, v66, v139
	v_dual_mul_f32 v67, v67, v139 :: v_dual_mul_f32 v68, v68, v139
	v_dual_mul_f32 v33, v33, v139 :: v_dual_mul_f32 v34, v34, v139
	v_dual_mul_f32 v35, v35, v139 :: v_dual_mul_f32 v36, v36, v139
	v_dual_mul_f32 v9, v9, v139 :: v_dual_mul_f32 v10, v10, v139
	v_dual_mul_f32 v11, v11, v139 :: v_dual_mul_f32 v12, v12, v139
	v_dual_mul_f32 v69, v69, v139 :: v_dual_mul_f32 v70, v70, v139
	v_dual_mul_f32 v71, v71, v139 :: v_dual_mul_f32 v72, v72, v139
	v_dual_mul_f32 v37, v37, v139 :: v_dual_mul_f32 v38, v38, v139
	v_dual_mul_f32 v39, v39, v139 :: v_dual_mul_f32 v40, v40, v139
	v_dual_mul_f32 v13, v13, v139 :: v_dual_mul_f32 v14, v14, v139
	v_dual_mul_f32 v15, v15, v139 :: v_dual_mul_f32 v16, v16, v139
	v_dual_mul_f32 v49, v49, v139 :: v_dual_mul_f32 v50, v50, v139
	v_dual_mul_f32 v51, v51, v139 :: v_dual_mul_f32 v52, v52, v139
	v_dual_mul_f32 v25, v25, v139 :: v_dual_mul_f32 v26, v26, v139
	v_dual_mul_f32 v27, v27, v139 :: v_dual_mul_f32 v28, v28, v139
	v_dual_mul_f32 v0, v1, v139 :: v_dual_mul_f32 v1, v2, v139
	v_dual_mul_f32 v2, v3, v139 :: v_dual_mul_f32 v3, v4, v139
	v_dual_mul_f32 v53, v53, v139 :: v_dual_mul_f32 v54, v54, v139
	v_dual_mul_f32 v55, v55, v139 :: v_dual_mul_f32 v56, v56, v139
	s_clause 0x5
	global_store_b128 v[137:138], v[57:60], off offset:448
	global_store_b128 v[137:138], v[61:64], off offset:464
	global_store_b128 v[137:138], v[65:68], off offset:512
	global_store_b128 v[137:138], v[69:72], off offset:528
	global_store_b128 v[137:138], v[49:52], off offset:576
	global_store_b128 v[137:138], v[53:56], off offset:592
	v_dual_mul_f32 v29, v29, v139 :: v_dual_mul_f32 v30, v30, v139
	v_dual_mul_f32 v31, v31, v139 :: v_dual_mul_f32 v32, v32, v139
	s_clause 0x5
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
.LBB0_279:                              ; %_Z12fa2_gqa_bodyILb0EEvPKDF16_PKhS3_PfPKiifiiiiii.exit
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	attention_fp8_e4m3_fa2_gqa_f16_gfx1201, .Lfunc_end0-attention_fp8_e4m3_fa2_gqa_f16_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_f16_gfx1201
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
		.amdhsa_next_free_vgpr 235
		.amdhsa_next_free_sgpr 32
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-attention_fp8_e4m3_fa2_gqa_f16_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_f16_gfx1201.num_vgpr, 235
	.set .Lattention_fp8_e4m3_fa2_gqa_f16_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_f16_gfx1201.numbered_sgpr, 32
	.set .Lattention_fp8_e4m3_fa2_gqa_f16_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_f16_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_f16_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_f16_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_f16_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_f16_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_f16_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14652
; TotalNumSgprs: 34
; NumVgprs: 235
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 235
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
	.protected	attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x28
	s_load_b64 s[20:21], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s4, s21, -9
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s4, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB2_283
; %bb.1:
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB2_283
; %bb.2:
	s_lshl_b32 s4, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s4, s7
	s_cbranch_scc1 .LBB2_283
; %bb.3:
	s_lshr_b32 s22, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s22, s21
	s_cbranch_scc1 .LBB2_283
; %bb.4:
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v185, 0 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v1, 0
	v_lshrrev_b32_e32 v194, 3, v0
	v_mov_b32_e32 v187, 0
	s_mov_b32 s11, 0
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB2_6
; %bb.5:
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_or_b32 v185, v0, 7, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 6, v1
	v_cmp_gt_i32_e32 vcc_lo, s7, v185
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[1:2], null, s3, 6, v[1:2]
	v_mul_lo_u32 v2, 0x1800, v185
	s_and_b32 s11, vcc_lo, exec_lo
	v_and_or_b32 v187, v194, 1, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v1, v187, 8, v2
	v_mov_b32_e32 v2, 0
.LBB2_6:
	s_or_b32 exec_lo, exec_lo, s5
	s_clause 0x1
	s_load_b256 s[12:19], s[0:1], 0x0
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
.LBB2_9:                                ; %Flow880
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB2_10:                               ; %.lr.ph.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_lshrrev_b32_e32 v195, 4, v6
	v_dual_mov_b32 v129, 0 :: v_dual_lshlrev_b32 v8, 5, v0
	v_and_b32_e32 v59, 16, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_xor_b32_e32 v7, 16, v3
	v_lshl_add_u32 v199, v6, 4, 0
	v_xor_b32_e32 v14, 8, v3
	v_xor_b32_e32 v15, 4, v3
	v_xor_b32_e32 v16, 2, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	v_lshlrev_b32_e32 v196, 8, v0
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	v_xor_b32_e32 v17, 1, v3
	v_lshrrev_b32_e32 v9, 1, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v3, v7, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v14
	v_lshlrev_b32_e32 v11, 3, v0
	v_ashrrev_i32_e32 v186, 31, v185
	v_lshrrev_b32_e32 v10, 2, v0
	v_dual_mov_b32 v189, 0 :: v_dual_lshlrev_b32 v198, 2, v7
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v3, v14, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v15
	v_and_b32_e32 v200, 0xe0, v8
	ds_bpermute_b32 v6, v198, v4
	ds_bpermute_b32 v7, v198, v5
	v_dual_mov_b32 v131, v129 :: v_dual_lshlrev_b32 v8, 2, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, v3, v15, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v16
	v_or_b32_e32 v12, 0x80, v0
	v_dual_mov_b32 v132, v129 :: v_dual_lshlrev_b32 v13, 4, v0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v16, v3, v16 :: v_dual_lshlrev_b32 v15, 2, v15
	s_wait_kmcnt 0x0
	v_add_co_u32 v60, vcc_lo, s12, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v61, null, s13, v2, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v17
	v_dual_mov_b32 v134, v129 :: v_dual_and_b32 v201, 16, v9
	v_lshlrev_b64_e32 v[57:58], 2, v[185:186]
	v_dual_mov_b32 v133, v129 :: v_dual_lshlrev_b32 v186, 3, v195
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v7
	v_dual_mov_b32 v135, v129 :: v_dual_and_b32 v202, 30, v10
	v_dual_mov_b32 v136, v129 :: v_dual_and_b32 v203, 0x200, v11
	ds_bpermute_b32 v6, v8, v4
	ds_bpermute_b32 v7, v8, v5
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, v3, v17 :: v_dual_lshlrev_b32 v10, 3, v12
	v_mov_b32_e32 v130, v129
	v_lshlrev_b32_e32 v2, 2, v16
	s_cvt_f32_u32 s4, s21
	v_or_b32_e32 v18, 0x1e00, v13
	v_or_b32_e32 v19, 0x2600, v13
	v_or_b32_e32 v20, 0x2e00, v13
	v_or_b32_e32 v21, 0x3600, v13
	v_or_b32_e32 v22, 0x3e00, v13
	v_or_b32_e32 v23, 0x4600, v13
	v_or_b32_e32 v24, 0x4e00, v13
	v_lshlrev_b32_e32 v1, 2, v1
	s_add_co_i32 s5, s21, 0x1ff
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s6, s4
	v_add_nc_u32_e32 v210, 0, v18
	v_add_nc_u32_e32 v211, 0, v19
	v_add_nc_u32_e32 v212, 0, v20
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v7
	v_add_nc_u32_e32 v213, 0, v21
	v_add_nc_u32_e32 v214, 0, v22
	v_add_nc_u32_e32 v215, 0, v23
	ds_bpermute_b32 v27, v15, v4
	ds_bpermute_b32 v15, v15, v5
	v_dual_mov_b32 v17, v129 :: v_dual_add_nc_u32 v216, 0, v24
	s_and_b32 s5, s5, 0xffff
	v_mov_b32_e32 v20, v132
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s5, s5
	v_or_b32_e32 v25, 0x5600, v13
	v_or_b32_e32 v26, 0x6e00, v13
	v_or_b32_e32 v28, 0x7600, v13
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s6, s5, s6
	v_add_co_u32 v190, vcc_lo, v60, v59
	v_add_nc_u32_e32 v217, 0, v25
	s_wait_alu depctr_sa_sdst(0)
	s_trunc_f32 s6, s6
	v_dual_mov_b32 v21, v133 :: v_dual_add_nc_u32 v220, 0, v26
	v_dual_mov_b32 v18, v130 :: v_dual_add_nc_u32 v221, 0, v28
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, s6, 0x80000000
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v4, v27
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v5, v15
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s5, s7, s4
	v_dual_mov_b32 v22, v134 :: v_dual_mov_b32 v25, v129
	ds_bpermute_b32 v5, v2, v3
	ds_bpermute_b32 v2, v2, v4
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s5, 31
	s_cvt_u32_f32 s6, s6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s5, s4
	v_dual_mov_b32 v30, v134 :: v_dual_mov_b32 v33, v129
	v_dual_mov_b32 v38, v134 :: v_dual_mov_b32 v41, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v191, null, 0, v61, vcc_lo
	v_add_co_u32 v192, vcc_lo, s0, v57
	v_dual_mov_b32 v46, v134 :: v_dual_mov_b32 v49, v129
	v_mov_b32_e32 v54, v134
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v193, null, s1, v58, vcc_lo
	s_add_co_ci_u32 s4, s6, 0
	v_and_b32_e32 v8, 12, v9
	s_wait_dscnt 0x1
	v_max_i32_e32 v62, v3, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v63, v4, v2
	v_lshrrev_b32_e32 v9, 2, v12
	v_or_b32_e32 v11, 0x600, v13
	v_lshl_or_b32 v12, v12, 4, 0x600
	ds_bpermute_b32 v64, v1, v62
	ds_bpermute_b32 v65, v1, v63
	v_or_b32_e32 v14, 0x1600, v13
	v_or_b32_e32 v6, 0x5e00, v13
	v_or_b32_e32 v7, 0x6600, v13
	v_or_b32_e32 v13, 0x7e00, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s4, 0xffff
	v_add_nc_u32_e32 v204, 0, v8
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s12, s22, s4
	v_and_b32_e32 v205, 62, v9
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s13, s12, 6
	v_and_b32_e32 v206, 0x600, v10
	v_add_nc_u32_e32 v207, 0, v11
	v_add_nc_u32_e32 v208, 0, v12
	v_add_nc_u32_e32 v209, 0, v14
	v_dual_mov_b32 v19, v131 :: v_dual_add_nc_u32 v218, 0, v6
	v_add_nc_u32_e32 v219, 0, v7
	v_dual_mov_b32 v23, v135 :: v_dual_add_nc_u32 v222, 0, v13
	s_wait_dscnt 0x1
	v_max_i32_e32 v59, v62, v64
	s_wait_dscnt 0x0
	v_min_i32_e32 v60, v63, v65
	v_mov_b32_e32 v1, v129
	v_mov_b32_e32 v9, v129
	v_mov_b32_e32 v65, v129
	v_readfirstlane_b32 s23, v59
	v_readfirstlane_b32 s24, v60
	v_dual_mov_b32 v57, v129 :: v_dual_mov_b32 v62, v134
	v_dual_mov_b32 v70, v134 :: v_dual_mov_b32 v73, v129
	v_dual_mov_b32 v78, v134 :: v_dual_mov_b32 v81, v129
	v_dual_mov_b32 v86, v134 :: v_dual_mov_b32 v89, v129
	v_dual_mov_b32 v94, v134 :: v_dual_mov_b32 v97, v129
	v_dual_mov_b32 v102, v134 :: v_dual_mov_b32 v105, v129
	v_dual_mov_b32 v110, v134 :: v_dual_mov_b32 v113, v129
	v_dual_mov_b32 v118, v134 :: v_dual_mov_b32 v121, v129
	v_dual_mov_b32 v188, 0xff800000 :: v_dual_lshlrev_b32 v197, 6, v0
	v_dual_mov_b32 v2, v130 :: v_dual_mov_b32 v3, v131
	v_dual_mov_b32 v4, v132 :: v_dual_mov_b32 v5, v133
	v_dual_mov_b32 v6, v134 :: v_dual_mov_b32 v7, v135
	v_mov_b32_e32 v8, v136
	v_dual_mov_b32 v10, v130 :: v_dual_mov_b32 v11, v131
	v_dual_mov_b32 v12, v132 :: v_dual_mov_b32 v13, v133
	v_dual_mov_b32 v14, v134 :: v_dual_mov_b32 v15, v135
	v_mov_b32_e32 v16, v136
	v_dual_mov_b32 v24, v136 :: v_dual_mov_b32 v27, v131
	v_dual_mov_b32 v26, v130 :: v_dual_mov_b32 v29, v133
	v_dual_mov_b32 v28, v132 :: v_dual_mov_b32 v31, v135
	v_dual_mov_b32 v32, v136 :: v_dual_mov_b32 v35, v131
	v_dual_mov_b32 v34, v130 :: v_dual_mov_b32 v37, v133
	v_dual_mov_b32 v36, v132 :: v_dual_mov_b32 v39, v135
	v_dual_mov_b32 v40, v136 :: v_dual_mov_b32 v43, v131
	v_dual_mov_b32 v42, v130 :: v_dual_mov_b32 v45, v133
	v_dual_mov_b32 v44, v132 :: v_dual_mov_b32 v47, v135
	v_dual_mov_b32 v48, v136 :: v_dual_mov_b32 v51, v131
	v_dual_mov_b32 v50, v130 :: v_dual_mov_b32 v53, v133
	v_dual_mov_b32 v52, v132 :: v_dual_mov_b32 v55, v135
	v_dual_mov_b32 v56, v136 :: v_dual_mov_b32 v59, v131
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v223, s13, v186
	v_dual_mov_b32 v58, v130 :: v_dual_mov_b32 v61, v133
	v_dual_mov_b32 v60, v132 :: v_dual_mov_b32 v63, v135
	v_dual_mov_b32 v64, v136 :: v_dual_mov_b32 v67, v131
	v_dual_mov_b32 v66, v130 :: v_dual_mov_b32 v69, v133
	v_dual_mov_b32 v68, v132 :: v_dual_mov_b32 v71, v135
	v_dual_mov_b32 v72, v136 :: v_dual_mov_b32 v75, v131
	v_dual_mov_b32 v74, v130 :: v_dual_mov_b32 v77, v133
	v_dual_mov_b32 v76, v132 :: v_dual_mov_b32 v79, v135
	v_dual_mov_b32 v80, v136 :: v_dual_mov_b32 v83, v131
	v_dual_mov_b32 v82, v130 :: v_dual_mov_b32 v85, v133
	v_dual_mov_b32 v84, v132 :: v_dual_mov_b32 v87, v135
	v_dual_mov_b32 v88, v136 :: v_dual_mov_b32 v91, v131
	v_dual_mov_b32 v90, v130 :: v_dual_mov_b32 v93, v133
	v_dual_mov_b32 v92, v132 :: v_dual_mov_b32 v95, v135
	v_dual_mov_b32 v96, v136 :: v_dual_mov_b32 v99, v131
	v_dual_mov_b32 v98, v130 :: v_dual_mov_b32 v101, v133
	v_dual_mov_b32 v100, v132 :: v_dual_mov_b32 v103, v135
	v_dual_mov_b32 v104, v136 :: v_dual_mov_b32 v107, v131
	v_dual_mov_b32 v106, v130 :: v_dual_mov_b32 v109, v133
	v_dual_mov_b32 v108, v132 :: v_dual_mov_b32 v111, v135
	v_dual_mov_b32 v112, v136 :: v_dual_mov_b32 v115, v131
	v_dual_mov_b32 v114, v130 :: v_dual_mov_b32 v117, v133
	v_dual_mov_b32 v116, v132 :: v_dual_mov_b32 v119, v135
	v_dual_mov_b32 v120, v136 :: v_dual_mov_b32 v123, v131
	v_dual_mov_b32 v122, v130 :: v_dual_mov_b32 v125, v133
	v_dual_mov_b32 v124, v132 :: v_dual_mov_b32 v127, v135
	v_mov_b32_e32 v126, v134
	v_mov_b32_e32 v128, v136
	s_lshl_b32 s25, s3, 1
	s_lshl_b32 s26, s3, 8
	s_add_co_i32 s27, s12, s4
	s_branch .LBB2_13
.LBB2_11:                               ; %Flow818
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b32 exec_lo, exec_lo, s29
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB2_12:                               ;   in Loop: Header=BB2_13 Depth=1
	s_add_co_i32 s12, s12, 1
	v_add_nc_u32_e32 v223, 64, v223
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s12, s27
	s_cselect_b32 s0, -1, 0
	s_xor_b32 s1, s28, -1
	s_add_co_i32 s13, s13, 64
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_278
.LBB2_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_16 Depth 2
                                        ;       Child Loop BB2_23 Depth 3
                                        ;     Child Loop BB2_67 Depth 2
                                        ;     Child Loop BB2_145 Depth 2
                                        ;     Child Loop BB2_269 Depth 2
	s_lshl_b32 s3, s12, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s3, s23
	s_cselect_b32 s28, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_12
; %bb.14:                               ; %.preheader608.i.preheader
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_mov_b32 s1, 0
	s_branch .LBB2_16
.LBB2_15:                               ;   in Loop: Header=BB2_16 Depth=2
	s_add_co_i32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s1, 4
	s_cbranch_scc0 .LBB2_56
.LBB2_16:                               ; %.preheader608.i
                                        ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_23 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v133, s1, 7, v0
	s_mov_b32 s0, exec_lo
	v_mov_b32_e32 v131, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v130, 3, v133
	v_add_nc_u32_e32 v132, s3, v130
	v_mov_b32_e32 v130, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s23, v132
; %bb.17:                               ;   in Loop: Header=BB2_16 Depth=2
	v_mad_co_i64_i32 v[130:131], null, 0x408, v132, s[14:15]
; %bb.18:                               ;   in Loop: Header=BB2_16 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u64_e32 vcc_lo, 0, v[130:131]
	v_mov_b32_e32 v132, 0
	s_and_saveexec_b32 s4, vcc_lo
	s_cbranch_execz .LBB2_20
; %bb.19:                               ;   in Loop: Header=BB2_16 Depth=2
	v_add_co_u32 v134, s0, v130, s25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v135, null, 0, v131, s0
	global_load_d16_b16 v132, v[134:135], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v132, v132.l
.LBB2_20:                               ;   in Loop: Header=BB2_16 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v130, s0, v130, s26
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v134, null, 0, v131, s0
	v_lshrrev_b32_e32 v135, 2, v133
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v131, s0, v130, v200
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v133, null, 0, v134, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v135, v196 :: v_dual_and_b32 v134, 0x1e0, v135
	s_mov_b64 s[4:5], 0
	s_mov_b32 s6, 0
	s_branch .LBB2_23
.LBB2_21:                               ; %Flow868
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_or_b32 exec_lo, exec_lo, s8
.LBB2_22:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit532.i
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_and_b32_e32 v136, 0x780, v135
	s_and_b32 s0, s6, 16
	v_fma_mixlo_f16 v137, v132, v137, 0
	v_fma_mixhi_f16 v137, v132, v130, 0
	v_fma_mixlo_f16 v130, v132, v138, 0
	v_add_nc_u32_e32 v136, v136, v134
	v_fma_mixhi_f16 v130, v132, v139, 0
	v_add_nc_u32_e32 v135, 32, v135
	s_add_co_i32 s6, s6, 8
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v136, v194, s0, v136
	s_and_b32 s0, s4, 4
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s0, 1
	s_cmp_eq_u32 s4, 32
	v_lshlrev_b32_e32 v136, 4, v136
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v136, s0, 0, v136
	ds_store_2addr_b32 v136, v137, v130 offset1:1
	s_cbranch_scc1 .LBB2_15
.LBB2_23:                               ;   Parent Loop BB2_13 Depth=1
                                        ;     Parent Loop BB2_16 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v136, 0
	s_and_saveexec_b32 s7, vcc_lo
	s_cbranch_execz .LBB2_25
; %bb.24:                               ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v136, s0, v131, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v137, null, s5, v133, s0
	global_load_b32 v136, v[136:137], off
.LBB2_25:                               ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt 0x0
	v_bfe_i32 v130, v136, 0, 8
	v_bfe_u32 v139, v136, 3, 4
	v_and_b32_e32 v138, 7, v136
	s_mov_b32 s7, exec_lo
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v139
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB2_27
; %bb.26:                               ;   in Loop: Header=BB2_23 Depth=3
	v_cvt_f32_ubyte0_e32 v137, v138
	v_cmp_ne_u32_e64 s0, 7, v138
                                        ; implicit-def: $vgpr138
                                        ; implicit-def: $vgpr139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v137, 0x3e000000, v137, 1.0
	v_mul_f32_e32 v137, 0x43800000, v137
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v137, 0x7fc00000, v137, s0
	v_cmp_gt_i16_e64 s0, 0, v130.l
                                        ; implicit-def: $vgpr130_lo16
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v137, v137, -v137, s0
.LBB2_27:                               ; %Flow878
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB2_33
; %bb.28:                               ; %LeafBlock
                                        ;   in Loop: Header=BB2_23 Depth=3
	v_cmp_ne_u32_e64 s0, 0, v139
                                        ; implicit-def: $vgpr137
	s_and_saveexec_b32 s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s0, exec_lo, s8
; %bb.29:                               ;   in Loop: Header=BB2_23 Depth=3
	v_lshlrev_b32_e32 v130, 24, v136
	v_lshlrev_b32_e32 v137, 23, v139
	v_lshlrev_b32_e32 v138, 20, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v130, 0x80000000, v130
	v_or3_b32 v130, v137, v138, v130
                                        ; implicit-def: $vgpr138
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v137, 0x3c000000, v130
                                        ; implicit-def: $vgpr130_lo16
; %bb.30:                               ; %Flow876
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s0
; %bb.31:                               ;   in Loop: Header=BB2_23 Depth=3
	v_cvt_f32_ubyte0_e32 v137, v138
	v_cmp_gt_i16_e64 s0, 0, v130.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v137, 0x3b000000, v137
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v137, v137, -v137, s0
; %bb.32:                               ; %Flow877
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_or_b32 exec_lo, exec_lo, s8
.LBB2_33:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit.i
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_lshrrev_b32_e32 v139, 8, v136
	v_bfe_u32 v140, v136, 11, 4
	s_mov_b32 s7, exec_lo
                                        ; implicit-def: $vgpr130
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v138, 7, v139
	v_cmpx_lt_i32_e32 14, v140
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB2_35
; %bb.34:                               ;   in Loop: Header=BB2_23 Depth=3
	v_cvt_f32_ubyte0_e32 v130, v138
	v_bfe_i32 v139, v139, 0, 8
	v_cmp_ne_u32_e64 s0, 7, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v130, 0x3e000000, v130, 1.0
	v_mul_f32_e32 v140, 0x43800000, v130
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v130.l, v139.l
                                        ; implicit-def: $vgpr139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v138, 0x7fc00000, v140, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s0, 0, v130.l
                                        ; implicit-def: $vgpr140
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v130, v138, -v138, s0
                                        ; implicit-def: $vgpr138
.LBB2_35:                               ; %Flow875
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB2_41
; %bb.36:                               ; %LeafBlock739
                                        ;   in Loop: Header=BB2_23 Depth=3
	v_cmp_ne_u32_e64 s0, 0, v140
                                        ; implicit-def: $vgpr130
	s_and_saveexec_b32 s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s0, exec_lo, s8
; %bb.37:                               ;   in Loop: Header=BB2_23 Depth=3
	v_lshlrev_b32_e32 v130, 24, v139
	v_lshlrev_b32_e32 v139, 23, v140
	v_lshlrev_b32_e32 v138, 20, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v130, 0x80000000, v130
	v_or3_b32 v130, v139, v138, v130
                                        ; implicit-def: $vgpr139
                                        ; implicit-def: $vgpr138
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v130, 0x3c000000, v130
; %bb.38:                               ; %Flow873
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s0
	s_cbranch_execz .LBB2_40
; %bb.39:                               ;   in Loop: Header=BB2_23 Depth=3
	v_bfe_i32 v130, v139, 0, 8
	v_cvt_f32_ubyte0_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i16_e64 s0, 0, v130.l
	v_mul_f32_e32 v138, 0x3b000000, v138
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v130, v138, -v138, s0
.LBB2_40:                               ; %Flow874
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_or_b32 exec_lo, exec_lo, s8
.LBB2_41:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit522.i
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_mov_b16_e64 v140.h, 0
	v_mov_b16_e64 v140.l, v136.h
	v_bfe_u32 v141, v136, 19, 4
	s_mov_b32 s7, exec_lo
                                        ; implicit-def: $vgpr138
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v139, 7, v140
	v_cmpx_lt_i32_e32 14, v141
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB2_43
; %bb.42:                               ;   in Loop: Header=BB2_23 Depth=3
	v_cvt_f32_ubyte0_e32 v138, v139
	v_mov_b16_e64 v140.l, v136.h
	v_cmp_ne_u32_e64 s0, 7, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v138, 0x3e000000, v138, 1.0
	v_bfe_i32 v140, v140, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v141, 0x43800000, v138
	v_mov_b16_e64 v138.l, v140.l
                                        ; implicit-def: $vgpr140
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v139, 0x7fc00000, v141, s0
	v_cmp_gt_i16_e64 s0, 0, v138.l
                                        ; implicit-def: $vgpr141
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v138, v139, -v139, s0
                                        ; implicit-def: $vgpr139
.LBB2_43:                               ; %Flow872
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB2_49
; %bb.44:                               ; %LeafBlock743
                                        ;   in Loop: Header=BB2_23 Depth=3
	v_cmp_ne_u32_e64 s0, 0, v141
                                        ; implicit-def: $vgpr138
	s_and_saveexec_b32 s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s0, exec_lo, s8
; %bb.45:                               ;   in Loop: Header=BB2_23 Depth=3
	v_lshlrev_b32_e32 v138, 24, v140
	v_lshlrev_b32_e32 v140, 23, v141
	v_lshlrev_b32_e32 v139, 20, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v138, 0x80000000, v138
	v_or3_b32 v138, v140, v139, v138
                                        ; implicit-def: $vgpr139
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v138, 0x3c000000, v138
; %bb.46:                               ; %Flow870
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s0
	s_cbranch_execz .LBB2_48
; %bb.47:                               ;   in Loop: Header=BB2_23 Depth=3
	v_mov_b16_e64 v138.l, v136.h
	v_cvt_f32_ubyte0_e32 v139, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_bfe_i32 v138, v138, 0, 8
	v_mul_f32_e32 v139, 0x3b000000, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s0, 0, v138.l
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v138, v139, -v139, s0
.LBB2_48:                               ; %Flow871
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_or_b32 exec_lo, exec_lo, s8
.LBB2_49:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit527.i
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_bfe_u32 v141, v136, 27, 4
	v_bfe_u32 v140, v136, 24, 3
	s_mov_b32 s7, exec_lo
                                        ; implicit-def: $vgpr139
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v141
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB2_51
; %bb.50:                               ;   in Loop: Header=BB2_23 Depth=3
	v_cvt_f32_ubyte0_e32 v139, v140
	v_cmp_ne_u32_e64 s0, 7, v140
                                        ; implicit-def: $vgpr140
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v139, 0x3e000000, v139, 1.0
	v_mul_f32_e32 v139, 0x43800000, v139
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v139, 0x7fc00000, v139, s0
	v_cmp_gt_i32_e64 s0, 0, v136
                                        ; implicit-def: $vgpr136
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s0
.LBB2_51:                               ; %Flow869
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB2_22
; %bb.52:                               ; %LeafBlock747
                                        ;   in Loop: Header=BB2_23 Depth=3
	v_cmp_ne_u32_e64 s0, 0, v141
                                        ; implicit-def: $vgpr139
	s_and_saveexec_b32 s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s0, exec_lo, s8
; %bb.53:                               ;   in Loop: Header=BB2_23 Depth=3
	v_and_b32_e32 v136, 0x80000000, v136
	v_lshlrev_b32_e32 v139, 23, v141
	v_lshlrev_b32_e32 v140, 20, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v136, v139, v140, v136
                                        ; implicit-def: $vgpr140
	v_add_nc_u32_e32 v139, 0x3c000000, v136
                                        ; implicit-def: $vgpr136
; %bb.54:                               ; %Flow867
                                        ;   in Loop: Header=BB2_23 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s0
	s_cbranch_execz .LBB2_21
; %bb.55:                               ;   in Loop: Header=BB2_23 Depth=3
	v_cvt_f32_ubyte0_e32 v139, v140
	v_cmp_gt_i32_e64 s0, 0, v136
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v139, 0x3b000000, v139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s0
	s_branch .LBB2_21
.LBB2_56:                               ; %.preheader607.preheader.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v131, 0 :: v_dual_add_nc_u32 v134, s3, v202
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v133, 0
	v_mov_b32_e32 v132, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s23, v134
; %bb.57:                               ;   in Loop: Header=BB2_13 Depth=1
	v_mad_co_i64_i32 v[132:133], null, 0x408, v134, s[16:17]
; %bb.58:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s23, v134
; %bb.59:                               ;   in Loop: Header=BB2_13 Depth=1
	v_or_b32_e32 v130, 1, v134
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[130:131], null, 0x408, v130, s[16:17]
; %bb.60:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[132:133]
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB2_62
; %bb.61:                               ;   in Loop: Header=BB2_13 Depth=1
	v_add_co_u32 v135, s0, v132, s25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v136, null, 0, v133, s0
	global_load_d16_b16 v135, v[135:136], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v135, v135.l
.LBB2_62:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_ne_u64_e64 s0, 0, v[130:131]
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB2_64
; %bb.63:                               ;   in Loop: Header=BB2_13 Depth=1
	v_add_co_u32 v136, s1, v130, s25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v137, null, 0, v131, s1
	global_load_d16_b16 v134, v[136:137], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v134, v134.l
.LBB2_64:                               ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v132, s1, v132, s26
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v133, null, 0, v133, s1
	v_add_co_u32 v130, s1, v130, s26
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v136, null, 0, v131, s1
	v_add_co_u32 v131, s1, v132, v200
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v132, null, 0, v133, s1
	v_add_co_u32 v133, s1, v130, v200
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v136, null, 0, v136, s1
	v_mov_b32_e32 v137, v197
	s_mov_b64 s[4:5], 0
	s_branch .LBB2_67
.LBB2_65:                               ; %Flow844
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_66:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit542.3.i
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_fma_mixlo_f16 v138, v135, v140, 0
	v_fma_mixhi_f16 v138, v134, v139, 0
	v_add_nc_u32_e32 v137, 8, v137
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 32
	ds_store_b32 v130, v138 offset:32816
	s_cbranch_scc1 .LBB2_134
.LBB2_67:                               ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v139, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB2_69
; %bb.68:                               ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v139, s1, v131, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v140, null, s5, v132, s1
	global_load_b32 v139, v[139:140], off
.LBB2_69:                               ; %._crit_edge45
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s0
	s_cbranch_execz .LBB2_71
; %bb.70:                               ;   in Loop: Header=BB2_67 Depth=2
	v_add_co_u32 v140, s1, v133, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v141, null, s5, v136, s1
	global_load_b32 v138, v[140:141], off
.LBB2_71:                               ; %._crit_edge912.i
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_bfe_i32 v130, v139, 0, 8
	v_bfe_u32 v142, v139, 3, 4
	v_and_b32_e32 v141, 7, v139
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr140
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_73
; %bb.72:                               ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v140, 0x43800000, v140
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v140, 0x7fc00000, v140, s1
	v_cmp_gt_i16_e64 s1, 0, v130.l
                                        ; implicit-def: $vgpr130_lo16
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
.LBB2_73:                               ; %Flow866
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_79
; %bb.74:                               ; %LeafBlock783
                                        ;   in Loop: Header=BB2_67 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.75:                               ;   in Loop: Header=BB2_67 Depth=2
	v_lshlrev_b32_e32 v130, 24, v139
	v_lshlrev_b32_e32 v140, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v130, 0x80000000, v130
	v_or3_b32 v130, v140, v141, v130
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v130
                                        ; implicit-def: $vgpr130_lo16
; %bb.76:                               ; %Flow864
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.77:                               ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_gt_i16_e64 s1, 0, v130.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v140, 0x3b000000, v140
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
; %bb.78:                               ; %Flow865
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_79:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit537.i
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_bfe_i32 v130, v138, 0, 8
	v_bfe_u32 v143, v138, 3, 4
	v_and_b32_e32 v142, 7, v138
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_81
; %bb.80:                               ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v141, v142
	v_cmp_ne_u32_e64 s1, 7, v142
                                        ; implicit-def: $vgpr142
                                        ; implicit-def: $vgpr143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v141, 0x3e000000, v141, 1.0
	v_mul_f32_e32 v141, 0x43800000, v141
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v141, 0x7fc00000, v141, s1
	v_cmp_gt_i16_e64 s1, 0, v130.l
                                        ; implicit-def: $vgpr130_lo16
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, v141, -v141, s1
.LBB2_81:                               ; %Flow863
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_87
; %bb.82:                               ; %LeafBlock787
                                        ;   in Loop: Header=BB2_67 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr141
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.83:                               ;   in Loop: Header=BB2_67 Depth=2
	v_lshlrev_b32_e32 v130, 24, v138
	v_lshlrev_b32_e32 v141, 23, v143
	v_lshlrev_b32_e32 v142, 20, v142
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v130, 0x80000000, v130
	v_or3_b32 v130, v141, v142, v130
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v141, 0x3c000000, v130
                                        ; implicit-def: $vgpr130_lo16
; %bb.84:                               ; %Flow861
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.85:                               ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v141, v142
	v_cmp_gt_i16_e64 s1, 0, v130.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, v141, -v141, s1
; %bb.86:                               ; %Flow862
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_87:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit542.i
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_b32_e32 v130, 0x1e0, v137
	v_and_or_b32 v142, s4, 12, v203
	v_fma_mixlo_f16 v140, v135, v140, 0
	v_fma_mixhi_f16 v140, v134, v141, 0
	v_bfe_u32 v143, v139, 11, 4
	s_mov_b32 s6, exec_lo
	v_or3_b32 v130, v142, v130, v201
	v_lshrrev_b32_e32 v142, 8, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v130, 4, v130
	v_and_b32_e32 v141, 7, v142
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v130, v204, v130
	ds_store_b32 v130, v140 offset:32768
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_89
; %bb.88:                               ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_bfe_i32 v142, v142, 0, 8
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v143, 0x43800000, v140
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v140.l, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v140.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB2_89:                               ; %Flow860
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_95
; %bb.90:                               ; %LeafBlock791
                                        ;   in Loop: Header=BB2_67 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.91:                               ;   in Loop: Header=BB2_67 Depth=2
	v_lshlrev_b32_e32 v140, 24, v142
	v_lshlrev_b32_e32 v142, 23, v143
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v140, 0x80000000, v140
	v_or3_b32 v140, v142, v141, v140
                                        ; implicit-def: $vgpr142
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v140
; %bb.92:                               ; %Flow858
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB2_94
; %bb.93:                               ;   in Loop: Header=BB2_67 Depth=2
	v_bfe_i32 v140, v142, 0, 8
	v_cvt_f32_ubyte0_e32 v141, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i16_e64 s1, 0, v140.l
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v140, v141, -v141, s1
.LBB2_94:                               ; %Flow859
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_95:                               ; %_ZL19fa2_fp8_e4m3_to_f32h.exit537.1.i
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_lshrrev_b32_e32 v142, 8, v138
	v_bfe_u32 v144, v138, 11, 4
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v141, 7, v142
	v_cmpx_lt_i32_e32 14, v144
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_97
; %bb.96:                               ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v143, v141
	v_bfe_i32 v142, v142, 0, 8
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr144
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v143, 0x3e000000, v143, 1.0
	v_mul_f32_e32 v143, 0x43800000, v143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	v_cmp_gt_i16_e64 s1, 0, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v143, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB2_97:                               ; %Flow857
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_103
; %bb.98:                               ; %LeafBlock795
                                        ;   in Loop: Header=BB2_67 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v144
                                        ; implicit-def: $vgpr143
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.99:                               ;   in Loop: Header=BB2_67 Depth=2
	v_lshlrev_b32_e32 v142, 24, v142
	v_lshlrev_b32_e32 v143, 23, v144
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v142, 0x80000000, v142
	v_or3_b32 v141, v143, v141, v142
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v143, 0x3c000000, v141
                                        ; implicit-def: $vgpr141
; %bb.100:                              ; %Flow855
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.101:                              ;   in Loop: Header=BB2_67 Depth=2
	v_bfe_i32 v142, v142, 0, 8
	v_cvt_f32_ubyte0_e32 v143, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v141.l, v142.l
	v_mul_f32_e32 v142, 0x3b000000, v143
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v141.l
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v143, v142, -v142, s1
; %bb.102:                              ; %Flow856
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_103:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit542.1.i
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_mov_b16_e64 v142.h, 0
	v_mov_b16_e64 v142.l, v139.h
	v_fma_mixlo_f16 v140, v135, v140, 0
	v_fma_mixhi_f16 v140, v134, v143, 0
	v_bfe_u32 v143, v139, 19, 4
	s_mov_b32 s6, exec_lo
	v_and_b32_e32 v141, 7, v142
	ds_store_b32 v130, v140 offset:32784
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_105
; %bb.104:                              ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_mov_b16_e64 v142.l, v139.h
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_bfe_i32 v142, v142, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v143, 0x43800000, v140
	v_mov_b16_e64 v140.l, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	v_cmp_gt_i16_e64 s1, 0, v140.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v140, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB2_105:                              ; %Flow854
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_111
; %bb.106:                              ; %LeafBlock799
                                        ;   in Loop: Header=BB2_67 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.107:                              ;   in Loop: Header=BB2_67 Depth=2
	v_lshlrev_b32_e32 v140, 24, v142
	v_lshlrev_b32_e32 v142, 23, v143
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v140, 0x80000000, v140
	v_or3_b32 v140, v142, v141, v140
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v140
; %bb.108:                              ; %Flow852
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB2_110
; %bb.109:                              ;   in Loop: Header=BB2_67 Depth=2
	v_mov_b16_e64 v140.l, v139.h
	v_cvt_f32_ubyte0_e32 v141, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_bfe_i32 v140, v140, 0, 8
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v140.l
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v141, -v141, s1
.LBB2_110:                              ; %Flow853
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_111:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit537.2.i
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_mov_b16_e64 v143.h, 0
	v_mov_b16_e64 v143.l, v138.h
	v_bfe_u32 v144, v138, 19, 4
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v141, 7, v143
	v_cmpx_lt_i32_e32 14, v144
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_113
; %bb.112:                              ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v142, v141
	v_mov_b16_e64 v143.l, v138.h
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v142, 0x3e000000, v142, 1.0
	v_bfe_i32 v143, v143, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v144, 0x43800000, v142
	v_mov_b16_e64 v142.l, v143.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v141, 0x7fc00000, v144, s1
	v_cmp_gt_i16_e64 s1, 0, v142.l
                                        ; implicit-def: $vgpr144
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v142, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB2_113:                              ; %Flow851
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_119
; %bb.114:                              ; %LeafBlock803
                                        ;   in Loop: Header=BB2_67 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v144
                                        ; implicit-def: $vgpr142
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.115:                              ;   in Loop: Header=BB2_67 Depth=2
	v_lshlrev_b32_e32 v142, 24, v143
	v_lshlrev_b32_e32 v143, 23, v144
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v142, 0x80000000, v142
	v_or3_b32 v141, v143, v141, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v142, 0x3c000000, v141
                                        ; implicit-def: $vgpr141
; %bb.116:                              ; %Flow849
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB2_118
; %bb.117:                              ;   in Loop: Header=BB2_67 Depth=2
	v_mov_b16_e64 v142.l, v138.h
	v_cvt_f32_ubyte0_e32 v143, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_bfe_i32 v142, v142, 0, 8
	v_mov_b16_e64 v141.l, v142.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v142, 0x3b000000, v143
	v_cmp_gt_i16_e64 s1, 0, v141.l
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v142, v142, -v142, s1
.LBB2_118:                              ; %Flow850
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_119:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit542.2.i
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_fma_mixlo_f16 v140, v135, v140, 0
	v_fma_mixhi_f16 v140, v134, v142, 0
	v_bfe_u32 v142, v139, 27, 4
	v_bfe_u32 v141, v139, 24, 3
	s_mov_b32 s6, exec_lo
	ds_store_b32 v130, v140 offset:32800
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_121
; %bb.120:                              ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v140, 0x43800000, v140
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v140, 0x7fc00000, v140, s1
	v_cmp_gt_i32_e64 s1, 0, v139
                                        ; implicit-def: $vgpr139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
.LBB2_121:                              ; %Flow848
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_127
; %bb.122:                              ; %LeafBlock807
                                        ;   in Loop: Header=BB2_67 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.123:                              ;   in Loop: Header=BB2_67 Depth=2
	v_and_b32_e32 v139, 0x80000000, v139
	v_lshlrev_b32_e32 v140, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v139, v140, v141, v139
                                        ; implicit-def: $vgpr141
	v_add_nc_u32_e32 v140, 0x3c000000, v139
                                        ; implicit-def: $vgpr139
; %bb.124:                              ; %Flow846
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.125:                              ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_gt_i32_e64 s1, 0, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v140, 0x3b000000, v140
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
; %bb.126:                              ; %Flow847
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_127:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit537.3.i
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_bfe_u32 v142, v138, 27, 4
	v_bfe_u32 v141, v138, 24, 3
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr139
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_129
; %bb.128:                              ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v139, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v139, 0x3e000000, v139, 1.0
	v_mul_f32_e32 v139, 0x43800000, v139
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v139, 0x7fc00000, v139, s1
	v_cmp_gt_i32_e64 s1, 0, v138
                                        ; implicit-def: $vgpr138
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s1
.LBB2_129:                              ; %Flow845
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_66
; %bb.130:                              ; %LeafBlock811
                                        ;   in Loop: Header=BB2_67 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr139
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.131:                              ;   in Loop: Header=BB2_67 Depth=2
	v_and_b32_e32 v138, 0x80000000, v138
	v_lshlrev_b32_e32 v139, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v138, v139, v141, v138
                                        ; implicit-def: $vgpr141
	v_add_nc_u32_e32 v139, 0x3c000000, v138
                                        ; implicit-def: $vgpr138
; %bb.132:                              ; %Flow843
                                        ;   in Loop: Header=BB2_67 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB2_65
; %bb.133:                              ;   in Loop: Header=BB2_67 Depth=2
	v_cvt_f32_ubyte0_e32 v139, v141
	v_cmp_gt_i32_e64 s1, 0, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v139, 0x3b000000, v139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s1
	s_branch .LBB2_65
.LBB2_134:                              ; %.preheader607.1.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v131, 0 :: v_dual_add_nc_u32 v134, s3, v205
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v133, 0
	v_mov_b32_e32 v132, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s23, v134
; %bb.135:                              ;   in Loop: Header=BB2_13 Depth=1
	v_mad_co_i64_i32 v[132:133], null, 0x408, v134, s[16:17]
; %bb.136:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s23, v134
; %bb.137:                              ;   in Loop: Header=BB2_13 Depth=1
	v_or_b32_e32 v130, 1, v134
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[130:131], null, 0x408, v130, s[16:17]
; %bb.138:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_ne_u64_e32 vcc_lo, 0, v[132:133]
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB2_140
; %bb.139:                              ;   in Loop: Header=BB2_13 Depth=1
	v_add_co_u32 v135, s0, v132, s25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v136, null, 0, v133, s0
	global_load_d16_b16 v135, v[135:136], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v135, v135.l
.LBB2_140:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_ne_u64_e64 s0, 0, v[130:131]
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB2_142
; %bb.141:                              ;   in Loop: Header=BB2_13 Depth=1
	v_add_co_u32 v136, s1, v130, s25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v137, null, 0, v131, s1
	global_load_d16_b16 v134, v[136:137], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v134, v134.l
.LBB2_142:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v132, s1, v132, s26
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v133, null, 0, v133, s1
	v_add_co_u32 v130, s1, v130, s26
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v136, null, 0, v131, s1
	v_add_co_u32 v131, s1, v132, v200
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v132, null, 0, v133, s1
	v_add_co_u32 v133, s1, v130, v200
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v136, null, 0, v136, s1
	v_mov_b32_e32 v137, v197
	s_mov_b64 s[4:5], 0
	s_branch .LBB2_145
.LBB2_143:                              ; %Flow820
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_144:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit542.3.1.i
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_fma_mixlo_f16 v138, v135, v140, 0
	v_fma_mixhi_f16 v138, v134, v139, 0
	v_add_nc_u32_e32 v137, 8, v137
	s_add_nc_u64 s[4:5], s[4:5], 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 32
	ds_store_b32 v130, v138 offset:32816
	s_cbranch_scc0 .LBB2_212
.LBB2_145:                              ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v139, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB2_147
; %bb.146:                              ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v139, s1, v131, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v140, null, s5, v132, s1
	global_load_b32 v139, v[139:140], off
.LBB2_147:                              ; %._crit_edge
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s0
	s_cbranch_execz .LBB2_149
; %bb.148:                              ;   in Loop: Header=BB2_145 Depth=2
	v_add_co_u32 v140, s1, v133, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v141, null, s5, v136, s1
	global_load_b32 v138, v[140:141], off
.LBB2_149:                              ; %._crit_edge911.i
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_bfe_i32 v130, v139, 0, 8
	v_bfe_u32 v142, v139, 3, 4
	v_and_b32_e32 v141, 7, v139
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr140
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_151
; %bb.150:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v140, 0x43800000, v140
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v140, 0x7fc00000, v140, s1
	v_cmp_gt_i16_e64 s1, 0, v130.l
                                        ; implicit-def: $vgpr130_lo16
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
.LBB2_151:                              ; %Flow842
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_157
; %bb.152:                              ; %LeafBlock751
                                        ;   in Loop: Header=BB2_145 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.153:                              ;   in Loop: Header=BB2_145 Depth=2
	v_lshlrev_b32_e32 v130, 24, v139
	v_lshlrev_b32_e32 v140, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v130, 0x80000000, v130
	v_or3_b32 v130, v140, v141, v130
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v130
                                        ; implicit-def: $vgpr130_lo16
; %bb.154:                              ; %Flow840
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.155:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_gt_i16_e64 s1, 0, v130.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v140, 0x3b000000, v140
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
; %bb.156:                              ; %Flow841
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_157:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit537.1705.i
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_bfe_i32 v130, v138, 0, 8
	v_bfe_u32 v143, v138, 3, 4
	v_and_b32_e32 v142, 7, v138
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_159
; %bb.158:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v141, v142
	v_cmp_ne_u32_e64 s1, 7, v142
                                        ; implicit-def: $vgpr142
                                        ; implicit-def: $vgpr143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v141, 0x3e000000, v141, 1.0
	v_mul_f32_e32 v141, 0x43800000, v141
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v141, 0x7fc00000, v141, s1
	v_cmp_gt_i16_e64 s1, 0, v130.l
                                        ; implicit-def: $vgpr130_lo16
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, v141, -v141, s1
.LBB2_159:                              ; %Flow839
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_165
; %bb.160:                              ; %LeafBlock755
                                        ;   in Loop: Header=BB2_145 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr141
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.161:                              ;   in Loop: Header=BB2_145 Depth=2
	v_lshlrev_b32_e32 v130, 24, v138
	v_lshlrev_b32_e32 v141, 23, v143
	v_lshlrev_b32_e32 v142, 20, v142
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v130, 0x80000000, v130
	v_or3_b32 v130, v141, v142, v130
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v141, 0x3c000000, v130
                                        ; implicit-def: $vgpr130_lo16
; %bb.162:                              ; %Flow837
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.163:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v141, v142
	v_cmp_gt_i16_e64 s1, 0, v130.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, v141, -v141, s1
; %bb.164:                              ; %Flow838
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_165:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit542.1712.i
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_and_b32_e32 v130, 0x1e0, v137
	v_and_or_b32 v142, s4, 12, v206
	v_fma_mixlo_f16 v140, v135, v140, 0
	v_fma_mixhi_f16 v140, v134, v141, 0
	v_bfe_u32 v143, v139, 11, 4
	s_mov_b32 s6, exec_lo
	v_or3_b32 v130, v142, v130, v201
	v_lshrrev_b32_e32 v142, 8, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v130, 4, v130
	v_and_b32_e32 v141, 7, v142
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v130, v204, v130
	ds_store_b32 v130, v140 offset:32768
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_167
; %bb.166:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_bfe_i32 v142, v142, 0, 8
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v143, 0x43800000, v140
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v140.l, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v140.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB2_167:                              ; %Flow836
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_173
; %bb.168:                              ; %LeafBlock759
                                        ;   in Loop: Header=BB2_145 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.169:                              ;   in Loop: Header=BB2_145 Depth=2
	v_lshlrev_b32_e32 v140, 24, v142
	v_lshlrev_b32_e32 v142, 23, v143
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v140, 0x80000000, v140
	v_or3_b32 v140, v142, v141, v140
                                        ; implicit-def: $vgpr142
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v140
; %bb.170:                              ; %Flow834
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB2_172
; %bb.171:                              ;   in Loop: Header=BB2_145 Depth=2
	v_bfe_i32 v140, v142, 0, 8
	v_cvt_f32_ubyte0_e32 v141, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i16_e64 s1, 0, v140.l
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v140, v141, -v141, s1
.LBB2_172:                              ; %Flow835
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_173:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit537.1.1.i
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_lshrrev_b32_e32 v142, 8, v138
	v_bfe_u32 v144, v138, 11, 4
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v141, 7, v142
	v_cmpx_lt_i32_e32 14, v144
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_175
; %bb.174:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v143, v141
	v_bfe_i32 v142, v142, 0, 8
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr144
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v143, 0x3e000000, v143, 1.0
	v_mul_f32_e32 v143, 0x43800000, v143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	v_cmp_gt_i16_e64 s1, 0, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v143, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB2_175:                              ; %Flow833
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_181
; %bb.176:                              ; %LeafBlock763
                                        ;   in Loop: Header=BB2_145 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v144
                                        ; implicit-def: $vgpr143
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.177:                              ;   in Loop: Header=BB2_145 Depth=2
	v_lshlrev_b32_e32 v142, 24, v142
	v_lshlrev_b32_e32 v143, 23, v144
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v142, 0x80000000, v142
	v_or3_b32 v141, v143, v141, v142
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v143, 0x3c000000, v141
                                        ; implicit-def: $vgpr141
; %bb.178:                              ; %Flow831
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.179:                              ;   in Loop: Header=BB2_145 Depth=2
	v_bfe_i32 v142, v142, 0, 8
	v_cvt_f32_ubyte0_e32 v143, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v141.l, v142.l
	v_mul_f32_e32 v142, 0x3b000000, v143
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v141.l
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v143, v142, -v142, s1
; %bb.180:                              ; %Flow832
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_181:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit542.1.1.i
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_mov_b16_e64 v142.h, 0
	v_mov_b16_e64 v142.l, v139.h
	v_fma_mixlo_f16 v140, v135, v140, 0
	v_fma_mixhi_f16 v140, v134, v143, 0
	v_bfe_u32 v143, v139, 19, 4
	s_mov_b32 s6, exec_lo
	v_and_b32_e32 v141, 7, v142
	ds_store_b32 v130, v140 offset:32784
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v143
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_183
; %bb.182:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_mov_b16_e64 v142.l, v139.h
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_bfe_i32 v142, v142, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v143, 0x43800000, v140
	v_mov_b16_e64 v140.l, v142.l
                                        ; implicit-def: $vgpr142
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v141, 0x7fc00000, v143, s1
	v_cmp_gt_i16_e64 s1, 0, v140.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v140, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB2_183:                              ; %Flow830
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_189
; %bb.184:                              ; %LeafBlock767
                                        ;   in Loop: Header=BB2_145 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v143
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.185:                              ;   in Loop: Header=BB2_145 Depth=2
	v_lshlrev_b32_e32 v140, 24, v142
	v_lshlrev_b32_e32 v142, 23, v143
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v140, 0x80000000, v140
	v_or3_b32 v140, v142, v141, v140
                                        ; implicit-def: $vgpr141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v140, 0x3c000000, v140
; %bb.186:                              ; %Flow828
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB2_188
; %bb.187:                              ;   in Loop: Header=BB2_145 Depth=2
	v_mov_b16_e64 v140.l, v139.h
	v_cvt_f32_ubyte0_e32 v141, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_bfe_i32 v140, v140, 0, 8
	v_mul_f32_e32 v141, 0x3b000000, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i16_e64 s1, 0, v140.l
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v141, -v141, s1
.LBB2_188:                              ; %Flow829
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_189:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit537.2.1.i
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_mov_b16_e64 v143.h, 0
	v_mov_b16_e64 v143.l, v138.h
	v_bfe_u32 v144, v138, 19, 4
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v141, 7, v143
	v_cmpx_lt_i32_e32 14, v144
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_191
; %bb.190:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v142, v141
	v_mov_b16_e64 v143.l, v138.h
	v_cmp_ne_u32_e64 s1, 7, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v142, 0x3e000000, v142, 1.0
	v_bfe_i32 v143, v143, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v144, 0x43800000, v142
	v_mov_b16_e64 v142.l, v143.l
                                        ; implicit-def: $vgpr143
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v141, 0x7fc00000, v144, s1
	v_cmp_gt_i16_e64 s1, 0, v142.l
                                        ; implicit-def: $vgpr144
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v142, v141, -v141, s1
                                        ; implicit-def: $vgpr141
.LBB2_191:                              ; %Flow827
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_197
; %bb.192:                              ; %LeafBlock771
                                        ;   in Loop: Header=BB2_145 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v144
                                        ; implicit-def: $vgpr142
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.193:                              ;   in Loop: Header=BB2_145 Depth=2
	v_lshlrev_b32_e32 v142, 24, v143
	v_lshlrev_b32_e32 v143, 23, v144
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v142, 0x80000000, v142
	v_or3_b32 v141, v143, v141, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v142, 0x3c000000, v141
                                        ; implicit-def: $vgpr141
; %bb.194:                              ; %Flow825
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB2_196
; %bb.195:                              ;   in Loop: Header=BB2_145 Depth=2
	v_mov_b16_e64 v142.l, v138.h
	v_cvt_f32_ubyte0_e32 v143, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_bfe_i32 v142, v142, 0, 8
	v_mov_b16_e64 v141.l, v142.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v142, 0x3b000000, v143
	v_cmp_gt_i16_e64 s1, 0, v141.l
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v142, v142, -v142, s1
.LBB2_196:                              ; %Flow826
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_197:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit542.2.1.i
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_fma_mixlo_f16 v140, v135, v140, 0
	v_fma_mixhi_f16 v140, v134, v142, 0
	v_bfe_u32 v142, v139, 27, 4
	v_bfe_u32 v141, v139, 24, 3
	s_mov_b32 s6, exec_lo
	ds_store_b32 v130, v140 offset:32800
                                        ; implicit-def: $vgpr140
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_199
; %bb.198:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v140, 0x3e000000, v140, 1.0
	v_mul_f32_e32 v140, 0x43800000, v140
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v140, 0x7fc00000, v140, s1
	v_cmp_gt_i32_e64 s1, 0, v139
                                        ; implicit-def: $vgpr139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
.LBB2_199:                              ; %Flow824
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_205
; %bb.200:                              ; %LeafBlock775
                                        ;   in Loop: Header=BB2_145 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr140
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.201:                              ;   in Loop: Header=BB2_145 Depth=2
	v_and_b32_e32 v139, 0x80000000, v139
	v_lshlrev_b32_e32 v140, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v139, v140, v141, v139
                                        ; implicit-def: $vgpr141
	v_add_nc_u32_e32 v140, 0x3c000000, v139
                                        ; implicit-def: $vgpr139
; %bb.202:                              ; %Flow822
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
; %bb.203:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v140, v141
	v_cmp_gt_i32_e64 s1, 0, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v140, 0x3b000000, v140
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v140, v140, -v140, s1
; %bb.204:                              ; %Flow823
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.LBB2_205:                              ; %_ZL19fa2_fp8_e4m3_to_f32h.exit537.3.1.i
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_bfe_u32 v142, v138, 27, 4
	v_bfe_u32 v141, v138, 24, 3
	s_mov_b32 s6, exec_lo
                                        ; implicit-def: $vgpr139
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_i32_e32 14, v142
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB2_207
; %bb.206:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v139, v141
	v_cmp_ne_u32_e64 s1, 7, v141
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v139, 0x3e000000, v139, 1.0
	v_mul_f32_e32 v139, 0x43800000, v139
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v139, 0x7fc00000, v139, s1
	v_cmp_gt_i32_e64 s1, 0, v138
                                        ; implicit-def: $vgpr138
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s1
.LBB2_207:                              ; %Flow821
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB2_144
; %bb.208:                              ; %LeafBlock779
                                        ;   in Loop: Header=BB2_145 Depth=2
	v_cmp_ne_u32_e64 s1, 0, v142
                                        ; implicit-def: $vgpr139
	s_and_saveexec_b32 s7, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s7
; %bb.209:                              ;   in Loop: Header=BB2_145 Depth=2
	v_and_b32_e32 v138, 0x80000000, v138
	v_lshlrev_b32_e32 v139, 23, v142
	v_lshlrev_b32_e32 v141, 20, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v138, v139, v141, v138
                                        ; implicit-def: $vgpr141
	v_add_nc_u32_e32 v139, 0x3c000000, v138
                                        ; implicit-def: $vgpr138
; %bb.210:                              ; %Flow819
                                        ;   in Loop: Header=BB2_145 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s1
	s_cbranch_execz .LBB2_143
; %bb.211:                              ;   in Loop: Header=BB2_145 Depth=2
	v_cvt_f32_ubyte0_e32 v139, v141
	v_cmp_gt_i32_e64 s1, 0, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v139, 0x3b000000, v139
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v139, v139, -v139, s1
	s_branch .LBB2_143
.LBB2_212:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s29, s2
	s_cbranch_execz .LBB2_11
; %bb.213:                              ; %.preheader605.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v169, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, v169
	v_dual_mov_b32 v175, v169 :: v_dual_mov_b32 v176, v169
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_215
; %bb.214:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[173:176], v[190:191], off
.LBB2_215:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v171, 0
	v_mov_b32_e32 v172, 0
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_217
; %bb.216:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[169:172], v[190:191], off offset:32
.LBB2_217:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v181, 0 :: v_dual_mov_b32 v184, 0
	v_mov_b32_e32 v183, 0
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_219
; %bb.218:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[181:184], v[190:191], off offset:64
.LBB2_219:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v178, 0 :: v_dual_mov_b32 v179, 0
	v_mov_b32_e32 v180, 0
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_221
; %bb.220:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[177:180], v[190:191], off offset:96
.LBB2_221:                              ; %.preheader602.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[130:133], v199
	ds_load_b128 v[145:148], v199 offset:2048
	ds_load_b128 v[149:152], v199 offset:4096
	ds_load_b128 v[153:156], v199 offset:6144
	s_or_b32 s1, s3, 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s23
	s_cselect_b32 s0, -1, 0
	s_cmp_gt_i32 s1, s23
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[137:144], v[130:133], v[173:176], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[145:148], v[169:172], v[137:144]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[137:144], v[149:152], v[181:184], v[137:144]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[153:156], v[177:180], v[137:144]
	s_cbranch_scc1 .LBB2_223
; %bb.222:                              ; %.preheader599.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[130:133], v199 offset:512
	ds_load_b128 v[153:156], v199 offset:2560
	ds_load_b128 v[157:160], v199 offset:4608
	ds_load_b128 v[161:164], v199 offset:6656
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[145:152], v[130:133], v[173:176], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[153:156], v[169:172], v[145:152]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[145:152], v[157:160], v[181:184], v[145:152]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[161:164], v[177:180], v[145:152]
	s_branch .LBB2_224
.LBB2_223:                              ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v136, v129
	v_dual_mov_b32 v130, v129 :: v_dual_mov_b32 v131, v129
	v_dual_mov_b32 v132, v129 :: v_dual_mov_b32 v133, v129
	v_dual_mov_b32 v134, v129 :: v_dual_mov_b32 v135, v129
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v152, v136
	v_dual_mov_b32 v148, v132 :: v_dual_mov_b32 v147, v131
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v150, v134 :: v_dual_mov_b32 v149, v133
	v_dual_mov_b32 v151, v135 :: v_dual_mov_b32 v146, v130
	v_mov_b32_e32 v145, v129
.LBB2_224:                              ; %.loopexit598.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_or_b32 s4, s3, 32
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s23
	s_cselect_b32 s1, -1, 0
	s_cmp_gt_i32 s4, s23
	s_cbranch_scc1 .LBB2_226
; %bb.225:                              ; %.preheader596.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[130:133], v199 offset:1024
	ds_load_b128 v[161:164], v199 offset:3072
	ds_load_b128 v[165:168], v199 offset:5120
	ds_load_b128 v[224:227], v199 offset:7168
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[153:160], v[130:133], v[173:176], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[161:164], v[169:172], v[153:160]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[153:160], v[165:168], v[181:184], v[153:160]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[224:227], v[177:180], v[153:160]
	s_branch .LBB2_227
.LBB2_226:                              ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v136, v129
	v_dual_mov_b32 v130, v129 :: v_dual_mov_b32 v131, v129
	v_dual_mov_b32 v132, v129 :: v_dual_mov_b32 v133, v129
	v_dual_mov_b32 v134, v129 :: v_dual_mov_b32 v135, v129
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v160, v136
	v_dual_mov_b32 v156, v132 :: v_dual_mov_b32 v155, v131
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v158, v134 :: v_dual_mov_b32 v157, v133
	v_dual_mov_b32 v159, v135 :: v_dual_mov_b32 v154, v130
	v_mov_b32_e32 v153, v129
.LBB2_227:                              ;   in Loop: Header=BB2_13 Depth=1
	s_or_b32 s4, s3, 48
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s23
	s_cselect_b32 s3, -1, 0
	s_cmp_gt_i32 s4, s23
	s_cbranch_scc1 .LBB2_229
; %bb.228:                              ; %.preheader595.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[130:133], v207
	ds_load_b128 v[224:227], v208
	ds_load_b128 v[228:231], v209
	ds_load_b128 v[232:235], v210
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[161:168], v[130:133], v[173:176], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[224:227], v[169:172], v[161:168]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[161:168], v[228:231], v[181:184], v[161:168]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[232:235], v[177:180], v[161:168]
	s_branch .LBB2_230
.LBB2_229:                              ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v136, v129
	v_dual_mov_b32 v130, v129 :: v_dual_mov_b32 v131, v129
	v_dual_mov_b32 v132, v129 :: v_dual_mov_b32 v133, v129
	v_dual_mov_b32 v134, v129 :: v_dual_mov_b32 v135, v129
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b32_e32 v168, v136
	v_dual_mov_b32 v164, v132 :: v_dual_mov_b32 v163, v131
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v166, v134 :: v_dual_mov_b32 v165, v133
	v_dual_mov_b32 v167, v135 :: v_dual_mov_b32 v162, v130
	v_mov_b32_e32 v161, v129
.LBB2_230:                              ; %.preheader603.1.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v171, 0
	v_mov_b32_e32 v172, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s4, s11
	s_cbranch_execz .LBB2_232
; %bb.231:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[169:172], v[190:191], off offset:128
.LBB2_232:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_mov_b32_e32 v133, 0
	s_and_saveexec_b32 s4, s11
	s_cbranch_execz .LBB2_234
; %bb.233:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[130:133], v[190:191], off offset:160
.LBB2_234:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v180, 0
	v_mov_b32_e32 v179, 0
	s_and_saveexec_b32 s4, s11
	s_cbranch_execz .LBB2_236
; %bb.235:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[177:180], v[190:191], off offset:192
.LBB2_236:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	v_mov_b32_e32 v176, 0
	s_and_saveexec_b32 s4, s11
	s_cbranch_execz .LBB2_238
; %bb.237:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[173:176], v[190:191], off offset:224
.LBB2_238:                              ; %.preheader602.1.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	ds_load_b128 v[181:184], v199 offset:8192
	ds_load_b128 v[224:227], v199 offset:10240
	ds_load_b128 v[228:231], v199 offset:12288
	ds_load_b128 v[232:235], v199 offset:14336
	v_cndmask_b32_e64 v134, 0, 1, s0
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[137:144], v[181:184], v[169:172], v[137:144]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[224:227], v[130:133], v[137:144]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[137:144], v[228:231], v[177:180], v[137:144]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[232:235], v[173:176], v[137:144]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_272
; %bb.239:                              ; %.loopexit598.1.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_cndmask_b32_e64 v135, 0, 1, s1
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_273
.LBB2_240:                              ;   in Loop: Header=BB2_13 Depth=1
	v_cndmask_b32_e64 v136, 0, 1, s3
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_242
.LBB2_241:                              ; %.preheader595.1.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[181:184], v211
	ds_load_b128 v[224:227], v212
	ds_load_b128 v[228:231], v213
	ds_load_b128 v[232:235], v214
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[161:168], v[181:184], v[169:172], v[161:168]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[224:227], v[130:133], v[161:168]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[161:168], v[228:231], v[177:180], v[161:168]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[232:235], v[173:176], v[161:168]
.LBB2_242:                              ; %.preheader603.2.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v171, 0
	v_mov_b32_e32 v172, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_244
; %bb.243:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[169:172], v[190:191], off offset:256
.LBB2_244:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_mov_b32_e32 v133, 0
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_246
; %bb.245:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[130:133], v[190:191], off offset:288
.LBB2_246:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v180, 0
	v_mov_b32_e32 v179, 0
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_248
; %bb.247:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[177:180], v[190:191], off offset:320
.LBB2_248:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	v_mov_b32_e32 v176, 0
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_250
; %bb.249:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[173:176], v[190:191], off offset:352
.LBB2_250:                              ; %.preheader602.2.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[181:184], v199 offset:16384
	ds_load_b128 v[224:227], v199 offset:18432
	ds_load_b128 v[228:231], v199 offset:20480
	ds_load_b128 v[232:235], v199 offset:22528
	v_cmp_ne_u32_e32 vcc_lo, 1, v134
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[137:144], v[181:184], v[169:172], v[137:144]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[224:227], v[130:133], v[137:144]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[137:144], v[228:231], v[177:180], v[137:144]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[232:235], v[173:176], v[137:144]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_274
; %bb.251:                              ; %.loopexit598.2.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v135
	s_cbranch_vccz .LBB2_275
.LBB2_252:                              ;   in Loop: Header=BB2_13 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v136
	s_cbranch_vccnz .LBB2_254
.LBB2_253:                              ; %.preheader595.2.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[181:184], v215
	ds_load_b128 v[224:227], v216
	ds_load_b128 v[228:231], v217
	ds_load_b128 v[232:235], v218
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[161:168], v[181:184], v[169:172], v[161:168]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[224:227], v[130:133], v[161:168]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[161:168], v[228:231], v[177:180], v[161:168]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[232:235], v[173:176], v[161:168]
.LBB2_254:                              ; %.preheader603.3.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v171, 0
	v_mov_b32_e32 v172, 0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_256
; %bb.255:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[169:172], v[190:191], off offset:384
.LBB2_256:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_mov_b32_e32 v133, 0
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_258
; %bb.257:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[130:133], v[190:191], off offset:416
.LBB2_258:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v180, 0
	v_mov_b32_e32 v179, 0
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_260
; %bb.259:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[177:180], v[190:191], off offset:448
.LBB2_260:                              ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	v_mov_b32_e32 v176, 0
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_262
; %bb.261:                              ;   in Loop: Header=BB2_13 Depth=1
	global_load_b128 v[173:176], v[190:191], off offset:480
.LBB2_262:                              ; %.preheader602.3.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b128 v[181:184], v199 offset:24576
	ds_load_b128 v[224:227], v199 offset:26624
	ds_load_b128 v[228:231], v199 offset:28672
	ds_load_b128 v[232:235], v199 offset:30720
	v_cmp_ne_u32_e32 vcc_lo, 1, v134
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[137:144], v[181:184], v[169:172], v[137:144]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[224:227], v[130:133], v[137:144]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[137:144], v[228:231], v[177:180], v[137:144]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[137:144], v[232:235], v[173:176], v[137:144]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_276
; %bb.263:                              ; %.loopexit598.3.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v135
	s_cbranch_vccz .LBB2_277
.LBB2_264:                              ;   in Loop: Header=BB2_13 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v136
	s_cbranch_vccnz .LBB2_266
.LBB2_265:                              ; %.preheader595.3.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[181:184], v219
	ds_load_b128 v[224:227], v220
	ds_load_b128 v[228:231], v221
	ds_load_b128 v[232:235], v222
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[161:168], v[181:184], v[169:172], v[161:168]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[224:227], v[130:133], v[161:168]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[161:168], v[228:231], v[177:180], v[161:168]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[161:168], v[232:235], v[173:176], v[161:168]
.LBB2_266:                              ; %.loopexit606.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	v_mov_b32_e32 v130, v199
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s30, 0
	s_mov_b32 s31, 0
	s_branch .LBB2_269
.LBB2_267:                              ; %.critedge.i
                                        ;   in Loop: Header=BB2_269 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v132, s30, v223
	s_cmp_lt_u32 s31, 2
	v_cndmask_b32_e64 v169, v166, v158, s1
	s_cselect_b32 s10, -1, 0
	v_cndmask_b32_e64 v170, v165, v157, s1
	v_add_nc_u32_e32 v133, 2, v132
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s7, v132, v131
	v_cmp_lt_i32_e64 s8, v132, v131
	v_cndmask_b32_e64 v171, v164, v156, s1
	v_cndmask_b32_e64 v172, v162, v154, s1
	v_cmp_le_i32_e32 vcc_lo, v133, v131
	v_add_nc_u32_e32 v133, 3, v132
	v_cndmask_b32_e64 v173, v161, v153, s1
	v_cndmask_b32_e64 v174, v163, v155, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s3, v133, v131
	v_add_nc_u32_e32 v133, 4, v132
	v_cmp_le_i32_e64 s4, v133, v131
	v_add_nc_u32_e32 v133, 5, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cmp_le_i32_e64 s5, v133, v131
	v_add_nc_u32_e32 v133, 6, v132
	v_add_nc_u32_e32 v132, 7, v132
	v_cmp_le_i32_e64 s6, v133, v131
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_i32_e64 s9, v132, v131
	v_cndmask_b32_e64 v131, v152, v144, s0
	v_cndmask_b32_e64 v132, v168, v160, s1
	v_cndmask_b32_e64 v133, v167, v159, s1
	s_or_b32 s1, s33, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v131, v132, v131, s10
	v_cndmask_b32_e64 v132, v151, v143, s0
	v_mul_f32_e32 v131, s20, v131
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v132, v133, v132, s10
	v_cndmask_b32_e64 v133, v150, v142, s0
	v_mul_f32_e32 v132, s20, v132
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v133, v169, v133, s10
	v_cndmask_b32_e64 v169, v149, v141, s0
	v_mul_f32_e32 v133, s20, v133
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v169, v170, v169, s10
	v_cndmask_b32_e64 v170, v148, v140, s0
	v_mul_f32_e32 v169, s20, v169
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v170, v171, v170, s10
	v_cndmask_b32_e64 v171, v146, v138, s0
	v_mul_f32_e32 v170, s20, v170
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v171, v172, v171, s10
	v_cndmask_b32_e64 v172, v145, v137, s0
	v_mul_f32_e32 v171, s20, v171
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v172, v173, v172, s10
	v_cndmask_b32_e64 v173, v147, v139, s0
	s_or_b32 s0, s33, s7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s11, s0
	v_mul_f32_e32 v172, s20, v172
	v_cndmask_b32_e64 v173, v174, v173, s10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v172, 0xff800000, v172, s0
	s_and_b32 s0, s11, s1
	v_mul_f32_e32 v173, s20, v173
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v171, 0xff800000, v171, s0
	s_or_b32 s0, s33, vcc_lo
	s_or_b32 s1, s33, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s11, s0
	s_or_b32 s0, s33, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v173, 0xff800000, v173, vcc_lo
	s_and_b32 vcc_lo, s11, s1
	s_or_b32 s1, s33, s5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v170, 0xff800000, v170, vcc_lo
	s_and_b32 vcc_lo, s11, s0
	s_or_b32 s0, s33, s6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v169, 0xff800000, v169, vcc_lo
	s_and_b32 vcc_lo, s11, s1
	s_or_b32 s1, s33, s9
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v174, 0xff800000, v133, vcc_lo
	v_max3_num_f32 v133, v172, 0xff800000, v171
	s_and_b32 vcc_lo, s11, s0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v175, 0xff800000, v132, vcc_lo
	s_and_b32 vcc_lo, s11, s1
	v_max3_num_f32 v133, v133, v173, v170
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v176, 0xff800000, v131, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v131, v133, v169, v174
	v_max3_num_f32 v131, v131, v175, v176
	ds_bpermute_b32 v132, v198, v131
	s_wait_dscnt 0x0
	v_max3_num_f32 v131, v188, v131, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v132, v172, v131
	v_dual_sub_f32 v170, v170, v131 :: v_dual_sub_f32 v133, v171, v131
	v_sub_f32_e32 v176, v176, v131
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v131
	v_mul_f32_e32 v132, 0x3fb8aa3b, v132
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v170, 0x3fb8aa3b, v170 :: v_dual_mul_f32 v133, 0x3fb8aa3b, v133
	v_dual_sub_f32 v172, v188, v131 :: v_dual_sub_f32 v171, v173, v131
	v_exp_f32_e32 v132, v132
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v170, v170
	v_exp_f32_e32 v133, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v172, 0x3fb8aa3b, v172
	v_sub_f32_e32 v174, v174, v131
	v_exp_f32_e32 v172, v172
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v132, v132, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v178, v170, 0, vcc_lo
	v_cndmask_b32_e64 v133, v133, 0, vcc_lo
	v_mul_f32_e32 v171, 0x3fb8aa3b, v171
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e64 v170.l, v132
	v_add_f32_e32 v132, v132, v133
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v171, v171
	v_sub_f32_e32 v173, v169, v131
	v_cvt_f16_f32_e64 v170.h, v133
	v_cndmask_b32_e64 v169, v171, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v132, v169, v132 :: v_dual_sub_f32 v175, v175, v131
	v_dual_add_f32 v132, v178, v132 :: v_dual_mul_f32 v171, 0x3fb8aa3b, v173
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v173, 0x3fb8aa3b, v174 :: v_dual_mul_f32 v174, 0x3fb8aa3b, v175
	v_mul_f32_e32 v175, 0x3fb8aa3b, v176
	v_exp_f32_e32 v176, v171
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v173, v173
	v_exp_f32_e32 v174, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v175, v175
	v_cvt_f16_f32_e64 v171.l, v169
	v_cvt_f16_f32_e64 v171.h, v178
	v_cndmask_b32_e64 v179, v176, 0, vcc_lo
	v_cndmask_b32_e64 v180, v173, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e64 v181, v174, 0, vcc_lo
	v_cndmask_b32_e64 v182, v175, 0, vcc_lo
	ds_load_b128 v[174:177], v130 offset:32768
	v_add_f32_e32 v132, v179, v132
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v188
	v_cvt_f16_f32_e64 v173.l, v181
	v_cvt_f16_f32_e64 v173.h, v182
	;;#ASMSTART
	;;#ASMEND
	v_add_f32_e32 v132, v180, v132
	v_mov_b32_e32 v188, v131
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v132, v181, v132
	v_add_f32_e32 v132, v182, v132
	ds_bpermute_b32 v133, v198, v132
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, 0, v172, vcc_lo
	v_cvt_f16_f32_e64 v172.l, v179
	v_cvt_f16_f32_e64 v172.h, v180
	s_wait_dscnt 0x0
	v_add_f32_e32 v132, v132, v133
	v_dual_mul_f32 v128, v128, v183 :: v_dual_mul_f32 v127, v127, v183
	v_mul_f32_e32 v124, v124, v183
	v_dual_mul_f32 v126, v126, v183 :: v_dual_mul_f32 v125, v125, v183
	v_dual_mul_f32 v122, v122, v183 :: v_dual_mul_f32 v123, v123, v183
	v_dual_mul_f32 v120, v120, v183 :: v_dual_mul_f32 v121, v121, v183
	v_dual_mul_f32 v118, v118, v183 :: v_dual_mul_f32 v119, v119, v183
	v_dual_mul_f32 v116, v116, v183 :: v_dual_mul_f32 v117, v117, v183
	v_mul_f32_e32 v114, v114, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[121:128], v[174:177], v[170:173], v[121:128]
	ds_load_b128 v[174:177], v130 offset:33280
	v_dual_mul_f32 v115, v115, v183 :: v_dual_mul_f32 v112, v112, v183
	v_dual_mul_f32 v113, v113, v183 :: v_dual_mul_f32 v110, v110, v183
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v111, v111, v183 :: v_dual_mul_f32 v108, v108, v183
	v_dual_mul_f32 v109, v109, v183 :: v_dual_mul_f32 v106, v106, v183
	v_dual_mul_f32 v107, v107, v183 :: v_dual_mul_f32 v104, v104, v183
	v_dual_mul_f32 v105, v105, v183 :: v_dual_mul_f32 v102, v102, v183
	v_dual_mul_f32 v103, v103, v183 :: v_dual_mul_f32 v100, v100, v183
	v_dual_mul_f32 v101, v101, v183 :: v_dual_mul_f32 v98, v98, v183
	v_dual_mul_f32 v99, v99, v183 :: v_dual_mul_f32 v96, v96, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[113:120], v[174:177], v[170:173], v[113:120]
	ds_load_b128 v[174:177], v130 offset:33792
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v97, v97, v183 :: v_dual_mul_f32 v94, v94, v183
	v_dual_mul_f32 v95, v95, v183 :: v_dual_mul_f32 v92, v92, v183
	v_dual_mul_f32 v93, v93, v183 :: v_dual_mul_f32 v90, v90, v183
	v_dual_mul_f32 v91, v91, v183 :: v_dual_mul_f32 v88, v88, v183
	v_dual_mul_f32 v89, v89, v183 :: v_dual_mul_f32 v86, v86, v183
	v_dual_mul_f32 v87, v87, v183 :: v_dual_mul_f32 v84, v84, v183
	v_dual_mul_f32 v85, v85, v183 :: v_dual_mul_f32 v82, v82, v183
	v_dual_mul_f32 v83, v83, v183 :: v_dual_mul_f32 v80, v80, v183
	v_dual_mul_f32 v81, v81, v183 :: v_dual_mul_f32 v78, v78, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[105:112], v[174:177], v[170:173], v[105:112]
	ds_load_b128 v[174:177], v130 offset:34304
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v79, v79, v183 :: v_dual_mul_f32 v76, v76, v183
	v_dual_mul_f32 v77, v77, v183 :: v_dual_mul_f32 v74, v74, v183
	v_dual_mul_f32 v75, v75, v183 :: v_dual_mul_f32 v72, v72, v183
	v_dual_mul_f32 v73, v73, v183 :: v_dual_mul_f32 v70, v70, v183
	v_dual_mul_f32 v71, v71, v183 :: v_dual_mul_f32 v68, v68, v183
	v_dual_mul_f32 v69, v69, v183 :: v_dual_mul_f32 v66, v66, v183
	v_dual_mul_f32 v67, v67, v183 :: v_dual_mul_f32 v64, v64, v183
	v_dual_mul_f32 v65, v65, v183 :: v_dual_mul_f32 v62, v62, v183
	v_dual_mul_f32 v63, v63, v183 :: v_dual_mul_f32 v60, v60, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[97:104], v[174:177], v[170:173], v[97:104]
	ds_load_b128 v[174:177], v130 offset:34816
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v61, v61, v183 :: v_dual_mul_f32 v58, v58, v183
	v_dual_mul_f32 v59, v59, v183 :: v_dual_mul_f32 v56, v56, v183
	v_dual_mul_f32 v57, v57, v183 :: v_dual_mul_f32 v54, v54, v183
	v_dual_mul_f32 v55, v55, v183 :: v_dual_mul_f32 v52, v52, v183
	v_dual_mul_f32 v53, v53, v183 :: v_dual_mul_f32 v50, v50, v183
	v_dual_mul_f32 v51, v51, v183 :: v_dual_mul_f32 v48, v48, v183
	v_dual_mul_f32 v49, v49, v183 :: v_dual_mul_f32 v46, v46, v183
	v_dual_mul_f32 v47, v47, v183 :: v_dual_mul_f32 v44, v44, v183
	v_dual_mul_f32 v45, v45, v183 :: v_dual_mul_f32 v42, v42, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[174:177], v[170:173], v[89:96]
	ds_load_b128 v[174:177], v130 offset:35328
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v43, v43, v183 :: v_dual_mul_f32 v40, v40, v183
	v_dual_mul_f32 v41, v41, v183 :: v_dual_mul_f32 v38, v38, v183
	v_dual_mul_f32 v39, v39, v183 :: v_dual_mul_f32 v36, v36, v183
	v_dual_mul_f32 v37, v37, v183 :: v_dual_mul_f32 v34, v34, v183
	v_dual_mul_f32 v35, v35, v183 :: v_dual_mul_f32 v32, v32, v183
	v_dual_mul_f32 v33, v33, v183 :: v_dual_mul_f32 v30, v30, v183
	v_dual_mul_f32 v31, v31, v183 :: v_dual_mul_f32 v28, v28, v183
	v_dual_mul_f32 v29, v29, v183 :: v_dual_mul_f32 v26, v26, v183
	v_dual_mul_f32 v27, v27, v183 :: v_dual_mul_f32 v24, v24, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[81:88], v[174:177], v[170:173], v[81:88]
	ds_load_b128 v[174:177], v130 offset:35840
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v25, v25, v183 :: v_dual_mul_f32 v22, v22, v183
	v_dual_mul_f32 v23, v23, v183 :: v_dual_mul_f32 v20, v20, v183
	v_dual_mul_f32 v21, v21, v183 :: v_dual_mul_f32 v18, v18, v183
	v_dual_mul_f32 v19, v19, v183 :: v_dual_mul_f32 v16, v16, v183
	v_dual_mul_f32 v17, v17, v183 :: v_dual_mul_f32 v14, v14, v183
	v_dual_mul_f32 v15, v15, v183 :: v_dual_mul_f32 v12, v12, v183
	v_dual_mul_f32 v13, v13, v183 :: v_dual_mul_f32 v10, v10, v183
	v_dual_mul_f32 v11, v11, v183 :: v_dual_mul_f32 v8, v8, v183
	v_dual_mul_f32 v9, v9, v183 :: v_dual_mul_f32 v6, v6, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[73:80], v[174:177], v[170:173], v[73:80]
	ds_load_b128 v[174:177], v130 offset:36352
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v7, v7, v183 :: v_dual_mul_f32 v4, v4, v183
	v_dual_mul_f32 v5, v5, v183 :: v_dual_mul_f32 v2, v2, v183
	v_dual_mul_f32 v3, v3, v183 :: v_dual_fmac_f32 v132, v189, v183
	v_mul_f32_e32 v1, v1, v183
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v189, v132
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[65:72], v[174:177], v[170:173], v[65:72]
	ds_load_b128 v[174:177], v130 offset:36864
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[57:64], v[174:177], v[170:173], v[57:64]
	ds_load_b128 v[174:177], v130 offset:37376
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[49:56], v[174:177], v[170:173], v[49:56]
	ds_load_b128 v[174:177], v130 offset:37888
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[41:48], v[174:177], v[170:173], v[41:48]
	ds_load_b128 v[174:177], v130 offset:38400
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[33:40], v[174:177], v[170:173], v[33:40]
	ds_load_b128 v[174:177], v130 offset:38912
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[174:177], v[170:173], v[25:32]
	ds_load_b128 v[174:177], v130 offset:39424
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[17:24], v[174:177], v[170:173], v[17:24]
	ds_load_b128 v[174:177], v130 offset:39936
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[9:16], v[174:177], v[170:173], v[9:16]
	ds_load_b128 v[174:177], v130 offset:40448
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[174:177], v[170:173], v[1:8]
.LBB2_268:                              ; %.loopexit593.i
                                        ;   in Loop: Header=BB2_269 Depth=2
	v_add_nc_u32_e32 v130, 0x2000, v130
	s_add_co_i32 s30, s30, 16
	s_add_co_i32 s31, s31, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s30, 64
	s_cbranch_scc0 .LBB2_11
.LBB2_269:                              ;   Parent Loop BB2_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s30, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s30, 32
	s_cselect_b32 s1, -1, 0
	s_cmp_eq_u32 s30, 16
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v131, v136, v135, s1
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v131, v131, v134, vcc_lo
	v_and_b32_e32 v131, 1, v131
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 1, v131
	s_or_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_268
; %bb.270:                              ;   in Loop: Header=BB2_269 Depth=2
	s_add_co_i32 s3, s13, s30
	v_mov_b32_e32 v131, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s3, 15
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s24
	s_cselect_b32 s33, -1, 0
	s_cmp_gt_i32 s3, s24
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s3, s11
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB2_267
; %bb.271:                              ;   in Loop: Header=BB2_269 Depth=2
	global_load_b32 v131, v[192:193], off
	s_branch .LBB2_267
.LBB2_272:                              ; %.preheader599.1.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[181:184], v199 offset:8704
	ds_load_b128 v[224:227], v199 offset:10752
	ds_load_b128 v[228:231], v199 offset:12800
	ds_load_b128 v[232:235], v199 offset:14848
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[145:152], v[181:184], v[169:172], v[145:152]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[224:227], v[130:133], v[145:152]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[145:152], v[228:231], v[177:180], v[145:152]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[232:235], v[173:176], v[145:152]
	v_cndmask_b32_e64 v135, 0, 1, s1
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_240
.LBB2_273:                              ; %.preheader596.1.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[181:184], v199 offset:9216
	ds_load_b128 v[224:227], v199 offset:11264
	ds_load_b128 v[228:231], v199 offset:13312
	ds_load_b128 v[232:235], v199 offset:15360
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[153:160], v[181:184], v[169:172], v[153:160]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[224:227], v[130:133], v[153:160]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[153:160], v[228:231], v[177:180], v[153:160]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[232:235], v[173:176], v[153:160]
	v_cndmask_b32_e64 v136, 0, 1, s3
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_241
	s_branch .LBB2_242
.LBB2_274:                              ; %.preheader599.2.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[181:184], v199 offset:16896
	ds_load_b128 v[224:227], v199 offset:18944
	ds_load_b128 v[228:231], v199 offset:20992
	ds_load_b128 v[232:235], v199 offset:23040
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[145:152], v[181:184], v[169:172], v[145:152]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[224:227], v[130:133], v[145:152]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[145:152], v[228:231], v[177:180], v[145:152]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[232:235], v[173:176], v[145:152]
	v_cmp_ne_u32_e32 vcc_lo, 1, v135
	s_cbranch_vccnz .LBB2_252
.LBB2_275:                              ; %.preheader596.2.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[181:184], v199 offset:17408
	ds_load_b128 v[224:227], v199 offset:19456
	ds_load_b128 v[228:231], v199 offset:21504
	ds_load_b128 v[232:235], v199 offset:23552
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[153:160], v[181:184], v[169:172], v[153:160]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[224:227], v[130:133], v[153:160]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[153:160], v[228:231], v[177:180], v[153:160]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[232:235], v[173:176], v[153:160]
	v_cmp_ne_u32_e32 vcc_lo, 1, v136
	s_cbranch_vccz .LBB2_253
	s_branch .LBB2_254
.LBB2_276:                              ; %.preheader599.3.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[181:184], v199 offset:25088
	ds_load_b128 v[224:227], v199 offset:27136
	ds_load_b128 v[228:231], v199 offset:29184
	ds_load_b128 v[232:235], v199 offset:31232
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[145:152], v[181:184], v[169:172], v[145:152]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[224:227], v[130:133], v[145:152]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[145:152], v[228:231], v[177:180], v[145:152]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[145:152], v[232:235], v[173:176], v[145:152]
	v_cmp_ne_u32_e32 vcc_lo, 1, v135
	s_cbranch_vccnz .LBB2_264
.LBB2_277:                              ; %.preheader596.3.i
                                        ;   in Loop: Header=BB2_13 Depth=1
	ds_load_b128 v[181:184], v199 offset:25600
	ds_load_b128 v[224:227], v199 offset:27648
	ds_load_b128 v[228:231], v199 offset:29696
	ds_load_b128 v[232:235], v199 offset:31744
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[153:160], v[181:184], v[169:172], v[153:160]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[224:227], v[130:133], v[153:160]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[153:160], v[228:231], v[177:180], v[153:160]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[153:160], v[232:235], v[173:176], v[153:160]
	v_cmp_ne_u32_e32 vcc_lo, 1, v136
	s_cbranch_vccz .LBB2_265
	s_branch .LBB2_266
.LBB2_278:                              ; %._crit_edge.i
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB2_283
; %bb.279:
	v_cmp_eq_u32_e32 vcc_lo, 0, v195
	s_and_b32 s1, vcc_lo, s11
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_281
; %bb.280:
	v_mad_co_u64_u32 v[129:130], null, v185, 24, v[187:188]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[129:130], null, v129, s21, s[22:23]
	v_mov_b32_e32 v130, 0
	v_mul_lo_u32 v129, 0x102, v129
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[129:130], 2, v[129:130]
	v_add_co_u32 v129, vcc_lo, s18, v129
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v130, null, s19, v130, vcc_lo
	global_store_b64 v[129:130], v[188:189], off
.LBB2_281:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s11
	s_cbranch_execz .LBB2_283
; %bb.282:                              ; %.loopexit.loopexit.i
	v_mad_co_u64_u32 v[129:130], null, v185, 24, v[187:188]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[129:130], null, v129, s21, s[22:23]
	v_mul_lo_u32 v129, 0x102, v129
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v0, v129, v186
	v_mov_b32_e32 v130, 0
	v_lshlrev_b64_e32 v[131:132], 2, v[129:130]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v129, 2, v0
	v_lshlrev_b64_e32 v[133:134], 2, v[129:130]
	v_add_nc_u32_e32 v129, 18, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v131, vcc_lo, s18, v131
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v132, null, s19, v132, vcc_lo
	v_lshlrev_b64_e32 v[135:136], 2, v[129:130]
	v_add_nc_u32_e32 v129, 34, v0
	v_add_co_u32 v133, vcc_lo, v131, v133
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v134, null, v132, v134, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[137:138], 2, v[129:130]
	v_add_nc_u32_e32 v129, 50, v0
	v_add_co_u32 v135, vcc_lo, v131, v135
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v136, null, v132, v136, vcc_lo
	v_lshlrev_b64_e32 v[139:140], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x42, v0
	v_add_co_u32 v137, vcc_lo, v131, v137
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, v132, v138, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[141:142], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x52, v0
	v_add_co_u32 v139, vcc_lo, v131, v139
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v140, null, v132, v140, vcc_lo
	v_lshlrev_b64_e32 v[143:144], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x62, v0
	s_clause 0x3
	global_store_b64 v[133:134], v[121:122], off
	global_store_b64 v[135:136], v[113:114], off
	global_store_b64 v[137:138], v[105:106], off
	global_store_b64 v[139:140], v[97:98], off
	v_add_co_u32 v141, vcc_lo, v131, v141
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v142, null, v132, v142, vcc_lo
	v_lshlrev_b64_e32 v[97:98], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x72, v0
	v_add_co_u32 v105, vcc_lo, v131, v143
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v106, null, v132, v144, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[113:114], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x82, v0
	v_add_co_u32 v97, vcc_lo, v131, v97
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v98, null, v132, v98, vcc_lo
	v_lshlrev_b64_e32 v[121:122], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x92, v0
	v_add_co_u32 v113, vcc_lo, v131, v113
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v114, null, v132, v114, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[133:134], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xa2, v0
	s_clause 0x3
	global_store_b64 v[141:142], v[89:90], off
	global_store_b64 v[105:106], v[81:82], off
	global_store_b64 v[97:98], v[73:74], off
	global_store_b64 v[113:114], v[65:66], off
	v_add_co_u32 v121, vcc_lo, v131, v121
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v122, null, v132, v122, vcc_lo
	v_lshlrev_b64_e32 v[65:66], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xb2, v0
	v_add_co_u32 v73, vcc_lo, v131, v133
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, v132, v134, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[81:82], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xc2, v0
	v_add_co_u32 v65, vcc_lo, v131, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v132, v66, vcc_lo
	v_lshlrev_b64_e32 v[89:90], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xd2, v0
	v_add_co_u32 v81, vcc_lo, v131, v81
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, v132, v82, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[97:98], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xe2, v0
	v_add_co_u32 v89, vcc_lo, v131, v89
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v90, null, v132, v90, vcc_lo
	v_lshlrev_b64_e32 v[105:106], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xf2, v0
	v_add_co_u32 v97, vcc_lo, v131, v97
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v98, null, v132, v98, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[113:114], 2, v[129:130]
	v_add_nc_u32_e32 v129, 4, v0
	v_add_co_u32 v105, vcc_lo, v131, v105
	s_clause 0x3
	global_store_b64 v[121:122], v[57:58], off
	global_store_b64 v[73:74], v[49:50], off
	global_store_b64 v[65:66], v[41:42], off
	global_store_b64 v[81:82], v[33:34], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v106, null, v132, v106, vcc_lo
	v_lshlrev_b64_e32 v[33:34], 2, v[129:130]
	v_add_nc_u32_e32 v129, 20, v0
	v_add_co_u32 v113, vcc_lo, v131, v113
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v114, null, v132, v114, vcc_lo
	s_clause 0x3
	global_store_b64 v[89:90], v[25:26], off
	global_store_b64 v[97:98], v[17:18], off
	global_store_b64 v[105:106], v[9:10], off
	global_store_b64 v[113:114], v[1:2], off
	v_lshlrev_b64_e32 v[1:2], 2, v[129:130]
	v_add_nc_u32_e32 v129, 36, v0
	v_add_co_u32 v9, vcc_lo, v131, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v132, v34, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[129:130]
	v_add_nc_u32_e32 v129, 52, v0
	v_add_co_u32 v1, vcc_lo, v131, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v132, v2, vcc_lo
	v_lshlrev_b64_e32 v[25:26], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x44, v0
	v_add_co_u32 v17, vcc_lo, v131, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v132, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[33:34], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x54, v0
	v_add_co_u32 v25, vcc_lo, v131, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v132, v26, vcc_lo
	v_lshlrev_b64_e32 v[41:42], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x64, v0
	v_add_co_u32 v33, vcc_lo, v131, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, v132, v34, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[49:50], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x74, v0
	v_add_co_u32 v41, vcc_lo, v131, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v132, v42, vcc_lo
	v_lshlrev_b64_e32 v[57:58], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x84, v0
	v_add_co_u32 v49, vcc_lo, v131, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v50, null, v132, v50, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[65:66], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x94, v0
	v_add_co_u32 v57, vcc_lo, v131, v57
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v58, null, v132, v58, vcc_lo
	v_lshlrev_b64_e32 v[73:74], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xa4, v0
	v_add_co_u32 v65, vcc_lo, v131, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v132, v66, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[81:82], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xb4, v0
	v_add_co_u32 v73, vcc_lo, v131, v73
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, v132, v74, vcc_lo
	v_lshlrev_b64_e32 v[89:90], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xc4, v0
	v_add_co_u32 v81, vcc_lo, v131, v81
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, v132, v82, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[97:98], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xd4, v0
	v_add_co_u32 v89, vcc_lo, v131, v89
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v90, null, v132, v90, vcc_lo
	v_lshlrev_b64_e32 v[105:106], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xe4, v0
	v_add_co_u32 v97, vcc_lo, v131, v97
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v98, null, v132, v98, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[113:114], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xf4, v0
	v_add_co_u32 v105, vcc_lo, v131, v105
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v106, null, v132, v106, vcc_lo
	v_lshlrev_b64_e32 v[121:122], 2, v[129:130]
	v_add_nc_u32_e32 v129, 6, v0
	v_add_co_u32 v113, vcc_lo, v131, v113
	s_clause 0x7
	global_store_b64 v[9:10], v[123:124], off
	global_store_b64 v[1:2], v[115:116], off
	global_store_b64 v[17:18], v[107:108], off
	global_store_b64 v[25:26], v[99:100], off
	global_store_b64 v[33:34], v[91:92], off
	global_store_b64 v[41:42], v[83:84], off
	global_store_b64 v[49:50], v[75:76], off
	global_store_b64 v[57:58], v[67:68], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v114, null, v132, v114, vcc_lo
	v_lshlrev_b64_e32 v[1:2], 2, v[129:130]
	v_add_nc_u32_e32 v129, 22, v0
	v_add_co_u32 v121, vcc_lo, v131, v121
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v122, null, v132, v122, vcc_lo
	s_clause 0x3
	global_store_b64 v[97:98], v[27:28], off
	global_store_b64 v[105:106], v[19:20], off
	global_store_b64 v[113:114], v[11:12], off
	global_store_b64 v[121:122], v[3:4], off
	v_lshlrev_b64_e32 v[3:4], 2, v[129:130]
	v_add_nc_u32_e32 v129, 38, v0
	s_clause 0x3
	global_store_b64 v[65:66], v[59:60], off
	global_store_b64 v[73:74], v[51:52], off
	global_store_b64 v[81:82], v[43:44], off
	global_store_b64 v[89:90], v[35:36], off
	v_add_co_u32 v1, vcc_lo, v131, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v132, v2, vcc_lo
	v_lshlrev_b64_e32 v[9:10], 2, v[129:130]
	v_add_nc_u32_e32 v129, 54, v0
	v_add_co_u32 v3, vcc_lo, v131, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v132, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x46, v0
	v_add_co_u32 v9, vcc_lo, v131, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v132, v10, vcc_lo
	v_lshlrev_b64_e32 v[17:18], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x56, v0
	v_add_co_u32 v11, vcc_lo, v131, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v132, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x66, v0
	v_add_co_u32 v17, vcc_lo, v131, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v132, v18, vcc_lo
	v_lshlrev_b64_e32 v[25:26], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x76, v0
	v_add_co_u32 v19, vcc_lo, v131, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v132, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[27:28], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x86, v0
	v_add_co_u32 v25, vcc_lo, v131, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v132, v26, vcc_lo
	v_lshlrev_b64_e32 v[33:34], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x96, v0
	v_add_co_u32 v27, vcc_lo, v131, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, v132, v28, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[35:36], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xa6, v0
	v_add_co_u32 v33, vcc_lo, v131, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, v132, v34, vcc_lo
	v_lshlrev_b64_e32 v[41:42], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xb6, v0
	v_add_co_u32 v35, vcc_lo, v131, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, v132, v36, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[43:44], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xc6, v0
	v_add_co_u32 v41, vcc_lo, v131, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v132, v42, vcc_lo
	v_lshlrev_b64_e32 v[49:50], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xd6, v0
	v_add_co_u32 v43, vcc_lo, v131, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, v132, v44, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[51:52], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xe6, v0
	v_add_co_u32 v49, vcc_lo, v131, v49
	s_clause 0x7
	global_store_b64 v[1:2], v[125:126], off
	global_store_b64 v[3:4], v[117:118], off
	global_store_b64 v[9:10], v[109:110], off
	global_store_b64 v[11:12], v[101:102], off
	global_store_b64 v[17:18], v[93:94], off
	global_store_b64 v[19:20], v[85:86], off
	global_store_b64 v[25:26], v[77:78], off
	global_store_b64 v[27:28], v[69:70], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v50, null, v132, v50, vcc_lo
	v_lshlrev_b64_e32 v[57:58], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xf6, v0
	v_add_co_u32 v51, vcc_lo, v131, v51
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, v132, v52, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[59:60], 2, v[129:130]
	v_add_nc_u32_e32 v129, 8, v0
	v_add_co_u32 v57, vcc_lo, v131, v57
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v58, null, v132, v58, vcc_lo
	v_lshlrev_b64_e32 v[1:2], 2, v[129:130]
	v_add_nc_u32_e32 v129, 24, v0
	v_add_co_u32 v59, vcc_lo, v131, v59
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v60, null, v132, v60, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[3:4], 2, v[129:130]
	v_add_nc_u32_e32 v129, 40, v0
	s_clause 0x3
	global_store_b64 v[49:50], v[29:30], off
	global_store_b64 v[51:52], v[21:22], off
	global_store_b64 v[57:58], v[13:14], off
	global_store_b64 v[59:60], v[5:6], off
	v_add_co_u32 v1, vcc_lo, v131, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v132, v2, vcc_lo
	v_lshlrev_b64_e32 v[5:6], 2, v[129:130]
	v_add_nc_u32_e32 v129, 56, v0
	v_add_co_u32 v3, vcc_lo, v131, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v132, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x48, v0
	v_add_co_u32 v5, vcc_lo, v131, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v132, v6, vcc_lo
	v_lshlrev_b64_e32 v[11:12], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x58, v0
	v_add_co_u32 v9, vcc_lo, v131, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v132, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x68, v0
	v_add_co_u32 v11, vcc_lo, v131, v11
	s_clause 0x3
	global_store_b64 v[33:34], v[61:62], off
	global_store_b64 v[35:36], v[53:54], off
	global_store_b64 v[41:42], v[45:46], off
	global_store_b64 v[43:44], v[37:38], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v132, v12, vcc_lo
	v_lshlrev_b64_e32 v[17:18], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x78, v0
	v_add_co_u32 v13, vcc_lo, v131, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v132, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x88, v0
	v_add_co_u32 v17, vcc_lo, v131, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v132, v18, vcc_lo
	v_lshlrev_b64_e32 v[21:22], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0x98, v0
	v_add_co_u32 v19, vcc_lo, v131, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v132, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[25:26], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xa8, v0
	v_add_co_u32 v21, vcc_lo, v131, v21
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, v132, v22, vcc_lo
	v_lshlrev_b64_e32 v[27:28], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xb8, v0
	v_add_co_u32 v25, vcc_lo, v131, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v132, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[29:30], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xc8, v0
	v_add_co_u32 v27, vcc_lo, v131, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, v132, v28, vcc_lo
	v_lshlrev_b64_e32 v[33:34], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xd8, v0
	v_add_co_u32 v29, vcc_lo, v131, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, v132, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[35:36], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xe8, v0
	v_add_co_u32 v33, vcc_lo, v131, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, v132, v34, vcc_lo
	v_lshlrev_b64_e32 v[37:38], 2, v[129:130]
	v_add_nc_u32_e32 v129, 0xf8, v0
	v_add_co_u32 v35, vcc_lo, v131, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, v132, v36, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[41:42], 2, v[129:130]
	v_add_co_u32 v37, vcc_lo, v131, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v38, null, v132, v38, vcc_lo
	v_add_co_u32 v41, vcc_lo, v131, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v132, v42, vcc_lo
	s_clause 0xf
	global_store_b64 v[1:2], v[127:128], off
	global_store_b64 v[3:4], v[119:120], off
	global_store_b64 v[5:6], v[111:112], off
	global_store_b64 v[9:10], v[103:104], off
	global_store_b64 v[11:12], v[95:96], off
	global_store_b64 v[13:14], v[87:88], off
	global_store_b64 v[17:18], v[79:80], off
	global_store_b64 v[19:20], v[71:72], off
	global_store_b64 v[21:22], v[63:64], off
	global_store_b64 v[25:26], v[55:56], off
	global_store_b64 v[27:28], v[47:48], off
	global_store_b64 v[29:30], v[39:40], off
	global_store_b64 v[33:34], v[31:32], off
	global_store_b64 v[35:36], v[23:24], off
	global_store_b64 v[37:38], v[15:16], off
	global_store_b64 v[41:42], v[7:8], off
.LBB2_283:                              ; %_Z12fa2_gqa_bodyILb1EEvPKDF16_PKhS3_PfPKiifiiiiii.exit
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end2:
	.size	attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201, .Lfunc_end2-attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201
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
		.amdhsa_next_free_vgpr 236
		.amdhsa_next_free_sgpr 34
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.num_vgpr, 236
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.numbered_sgpr, 34
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16876
; TotalNumSgprs: 36
; NumVgprs: 236
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 36
; NumVGPRsForWavesPerEU: 236
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
	.protected	attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201
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
	s_branch .LBB3_6
.LBB3_5:                                ; %._crit_edge
                                        ;   in Loop: Header=BB3_6 Depth=1
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
.LBB3_6:                                ; %.preheader
                                        ; =>This Loop Header: Depth=1
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
.LBB3_10:                               ; %.loopexit
	s_endpgm
.Lfunc_end3:
	.size	attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201, .Lfunc_end3-attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.num_vgpr, 17
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.numbered_sgpr, 8
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.has_indirect_call, 0
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
	.type	__hip_cuid_ff43023e03900cd2,@object ; @__hip_cuid_ff43023e03900cd2
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_ff43023e03900cd2
__hip_cuid_ff43023e03900cd2:
	.byte	0                               ; 0x0
	.size	__hip_cuid_ff43023e03900cd2, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_ff43023e03900cd2
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
    .name:           attention_fp8_e4m3_fa2_gqa_f16_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_f16_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     235
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
    .name:           attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     36
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     236
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
    .name:           attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201.kd
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
