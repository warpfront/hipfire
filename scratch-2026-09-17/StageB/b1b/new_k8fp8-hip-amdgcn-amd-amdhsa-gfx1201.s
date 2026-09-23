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
	s_cbranch_vccnz .LBB0_321
; %bb.1:
	s_cmp_gt_i32 ttmp7, 3
	s_cbranch_scc1 .LBB0_321
; %bb.2:
	s_lshl_b32 s2, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s2, s15
	s_cbranch_scc1 .LBB0_321
; %bb.3:
	s_load_b256 s[4:11], s[0:1], 0x8
	v_dual_mov_b32 v2, -1 :: v_dual_and_b32 v1, 31, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e32 8, v1
	s_cbranch_execz .LBB0_7
; %bb.4:
	v_or_b32_e32 v1, s2, v1
	v_mov_b32_e32 v2, -1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s15, v1
	s_cbranch_execz .LBB0_6
; %bb.5:
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, vcc_lo, s10, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s11, v2, vcc_lo
	global_load_b32 v2, v[1:2], off
.LBB0_6:                                ; %Flow1239
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_7:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	s_or_b32 exec_lo, exec_lo, s0
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_dual_mov_b32 v13, 0 :: v_dual_and_b32 v4, 15, v0
	v_or_b32_e32 v16, 0x380, v0
	v_or_b32_e32 v12, 0x280, v0
	v_xor_b32_e32 v3, 16, v1
	v_xor_b32_e32 v18, 2, v1
	v_xor_b32_e32 v15, 8, v1
	v_lshrrev_b32_e32 v52, 4, v16
	v_lshrrev_b32_e32 v27, 1, v16
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	v_lshlrev_b32_e32 v31, 3, v16
	v_xor_b32_e32 v19, 1, v1
	v_lshrrev_b32_e32 v50, 4, v12
	v_lshrrev_b32_e32 v24, 1, v12
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v3, v1, v3 :: v_dual_lshlrev_b32 v6, 3, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v15
	v_or_b32_e32 v8, 0x80, v0
	v_or_b32_e32 v9, 0x100, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v3, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v15, v1, v15 :: v_dual_lshlrev_b32 v20, 7, v4
	v_or_b32_e32 v10, 0x180, v0
	v_or_b32_e32 v11, 0x200, v0
	s_wait_loadcnt 0x0
	ds_bpermute_b32 v3, v3, v2
	v_or_b32_e32 v14, 0x300, v0
	v_lshlrev_b32_e32 v15, 2, v15
	v_lshrrev_b32_e32 v17, 5, v0
	s_wait_kmcnt 0x0
	s_mov_b32 s10, ttmp7
	s_ashr_i32 s11, ttmp7, 31
	v_lshrrev_b32_e32 v5, 1, v0
	v_lshlrev_b32_e32 v21, 4, v4
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[16:17], s[10:11], 8
	v_lshrrev_b32_e32 v44, 4, v0
	v_lshrrev_b32_e32 v46, 4, v8
	v_lshrrev_b32_e32 v47, 4, v9
	v_lshrrev_b32_e32 v48, 4, v10
	v_lshrrev_b32_e32 v49, 4, v11
	v_lshrrev_b32_e32 v51, 4, v14
	v_lshl_or_b32 v23, v17, 4, v4
	v_lshrrev_b32_e32 v8, 5, v8
	s_mov_b32 s14, ttmp9
	s_ashr_i32 s15, ttmp9, 31
	s_add_nc_u64 s[20:21], s[4:5], s[16:17]
	s_wait_dscnt 0x0
	v_max_i32_e32 v2, v2, v3
	v_xor_b32_e32 v3, 4, v1
	v_lshrrev_b32_e32 v9, 5, v9
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[18:19], s[14:15], 11
	s_add_nc_u64 s[14:15], s[6:7], s[16:17]
	ds_bpermute_b32 v15, v15, v2
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	v_and_b32_e32 v22, 0xf8, v6
	v_and_b32_e32 v45, 8, v5
	v_and_or_b32 v5, v5, 48, v4
	v_lshrrev_b32_e32 v25, 1, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v1, v3, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v18
	v_lshlrev_b32_e32 v26, 3, v14
	v_add_co_u32 v14, s1, s20, v21
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v16, v1, v18 :: v_dual_lshlrev_b32 v3, 2, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v19
	v_lshlrev_b32_e32 v12, 3, v12
	v_lshrrev_b32_e32 v10, 5, v10
	v_or_b32_e32 v30, 0x480, v0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, v1, v19 :: v_dual_lshlrev_b32 v16, 2, v16
	v_add_nc_u32_e32 v54, 0, v22
	s_wait_dscnt 0x0
	v_max_i32_e32 v2, v2, v15
	v_add_co_ci_u32_e64 v15, null, s21, 0, s1
	v_lshlrev_b32_e32 v1, 2, v1
	v_or_b32_e32 v21, v44, v20
	ds_bpermute_b32 v3, v3, v2
	v_or_b32_e32 v22, v20, v46
	v_and_or_b32 v28, v47, 7, v20
	v_and_or_b32 v29, v48, 15, v20
	v_and_or_b32 v32, v50, 15, v20
	v_and_or_b32 v33, v51, 7, v20
	v_lshl_or_b32 v35, v9, 4, v4
	v_lshlrev_b32_e32 v53, 8, v17
	v_lshl_or_b32 v36, v10, 4, v4
	v_lshl_or_b32 v76, v32, 3, 0x200
	v_or_b32_e32 v32, 0x500, v0
	v_lshlrev_b32_e32 v71, 3, v21
	v_lshlrev_b32_e32 v72, 3, v22
	v_and_or_b32 v37, 0xb0, v25, v4
	v_and_or_b32 v38, 0xf0, v27, v4
	v_and_b32_e32 v84, 0x1b00, v26
	v_lshl_or_b32 v73, v28, 3, 0x100
	v_lshl_or_b32 v74, v29, 3, 0x100
	v_lshl_or_b32 v77, v33, 3, 0x300
	v_or_b32_e32 v39, 0x700, v0
	s_wait_dscnt 0x0
	v_max_i32_e32 v2, v2, v3
	v_and_or_b32 v3, v49, 7, v20
	v_and_or_b32 v20, v52, 15, v20
	v_or_b32_e32 v40, 0x780, v0
	v_lshrrev_b32_e32 v41, 1, v39
	ds_bpermute_b32 v34, v16, v2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v16, s1, s14, v23
	v_lshl_or_b32 v23, v8, 4, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s15, 0, s1
	v_add_co_u32 v18, s1, s14, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v19, null, s15, 0, s1
	v_lshl_or_b32 v78, v20, 3, 0x300
	v_add_co_u32 v20, s1, s14, v23
	v_and_or_b32 v5, 0x70, v24, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v21, null, s15, 0, s1
	v_add_co_u32 v22, s1, s14, v35
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v23, null, s15, 0, s1
	s_wait_dscnt 0x0
	v_max_i32_e32 v2, v2, v34
	v_add_co_u32 v24, s1, s14, v36
	v_lshrrev_b32_e32 v34, 1, v32
	v_or_b32_e32 v36, 0x580, v0
	ds_bpermute_b32 v1, v1, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v25, null, s15, 0, s1
	v_add_co_u32 v26, s1, s14, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v27, null, s15, 0, s1
	v_add_co_u32 v28, s1, s14, v37
	v_lshlrev_b32_e32 v5, 3, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, s15, 0, s1
	v_lshlrev_b32_e32 v35, 3, v32
	v_lshrrev_b32_e32 v37, 1, v36
	v_lshlrev_b32_e32 v36, 3, v36
	v_lshlrev_b32_e32 v42, 3, v39
	v_lshrrev_b32_e32 v43, 1, v40
	v_and_b32_e32 v88, 0x2b00, v35
	v_lshlrev_b32_e32 v11, 3, v11
	v_and_b32_e32 v89, 0x2f00, v36
	s_wait_dscnt 0x0
	v_max_i32_e32 v1, v2, v1
	v_lshrrev_b32_e32 v2, 1, v30
	v_add_co_u32 v30, s2, s14, v38
	v_or_b32_e32 v38, 0x680, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_readfirstlane_b32 s1, v1
	v_and_or_b32 v2, 0x70, v2, v4
	v_and_b32_e32 v1, 0x1f00, v31
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, s15, 0, s2
	v_or_b32_e32 v6, 0x3000, v6
	v_add_co_u32 v32, s2, s14, v2
	v_and_or_b32 v2, 0xb0, v34, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v33, null, s15, 0, s2
	v_and_b32_e32 v92, 0x3b00, v42
	v_lshlrev_b32_e32 v42, 3, v40
	v_add_co_u32 v34, s2, s14, v2
	v_and_or_b32 v2, 0xf0, v37, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v35, null, s15, 0, s2
	v_lshlrev_b32_e32 v7, 4, v0
	v_lshlrev_b32_e32 v8, 8, v8
	v_add_co_u32 v36, s2, s14, v2
	v_lshrrev_b32_e32 v2, 1, v38
	v_lshlrev_b32_e32 v38, 3, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v37, null, s15, 0, s2
	v_lshlrev_b32_e32 v9, 8, v9
	v_and_or_b32 v2, 0x70, v2, v4
	v_and_b32_e32 v91, 0x3700, v38
	v_lshlrev_b32_e32 v10, 8, v10
	v_and_b32_e32 v11, 0x1300, v11
	v_and_b32_e32 v12, 0x1700, v12
	v_add_co_u32 v38, s2, s14, v2
	v_and_or_b32 v2, 0xb0, v41, v4
	v_and_or_b32 v4, 0xf0, v43, v4
	v_lshl_or_b32 v3, v3, 3, 0x200
	v_or_b32_e32 v86, 0x2000, v53
	v_and_b32_e32 v5, 0x2700, v5
	v_and_b32_e32 v6, 0x3300, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v39, null, s15, 0, s2
	v_add_co_u32 v40, s2, s14, v2
	v_and_b32_e32 v2, 0x3f00, v42
	v_lshl_add_u32 v55, v0, 8, 0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v41, null, s15, 0, s2
	v_add_co_u32 v42, s2, s14, v4
	s_lshl_b32 s12, ttmp7, 1
	v_cmp_gt_u32_e64 s0, 64, v0
	v_add_co_ci_u32_e64 v43, null, s15, 0, s2
	v_add_nc_u32_e32 v56, 16, v55
	v_add_nc_u32_e32 v57, 32, v55
	v_add_nc_u32_e32 v58, 48, v55
	v_add_nc_u32_e32 v59, 64, v55
	v_add_nc_u32_e32 v60, 0x50, v55
	v_add_nc_u32_e32 v61, 0x60, v55
	v_add_nc_u32_e32 v62, 0x70, v55
	v_add_nc_u32_e32 v63, 0x80, v55
	v_add_nc_u32_e32 v64, 0x90, v55
	v_add_nc_u32_e32 v65, 0xa0, v55
	v_add_nc_u32_e32 v66, 0xb0, v55
	v_add_nc_u32_e32 v67, 0xc0, v55
	v_add_nc_u32_e32 v68, 0xd0, v55
	v_add_nc_u32_e32 v69, 0xe0, v55
	v_add_nc_u32_e32 v70, 0xf0, v55
	v_add_nc_u32_e32 v71, 0, v71
	v_add_nc_u32_e32 v72, 0, v72
	v_add_nc_u32_e32 v73, 0, v73
	v_add_nc_u32_e32 v74, 0, v74
	v_add_nc_u32_e32 v75, 0, v3
	v_add_nc_u32_e32 v76, 0, v76
	v_add_nc_u32_e32 v77, 0, v77
	v_add_nc_u32_e32 v78, 0, v78
	v_add_nc_u32_e32 v79, v54, v8
	v_add_nc_u32_e32 v80, v54, v9
	v_add_nc_u32_e32 v81, v54, v10
	v_add_nc_u32_e32 v82, v54, v11
	v_add_nc_u32_e32 v83, v54, v12
	v_add_nc_u32_e32 v84, v54, v84
	v_add_nc_u32_e32 v85, v54, v1
	v_add_nc_u32_e32 v86, v54, v86
	v_add_nc_u32_e32 v87, v54, v5
	v_add_nc_u32_e32 v88, v54, v88
	v_add_nc_u32_e32 v89, v54, v89
	v_add_nc_u32_e32 v90, v54, v6
	v_add_nc_u32_e32 v91, v54, v91
	v_add_nc_u32_e32 v92, v54, v92
	v_add_nc_u32_e32 v93, v54, v2
	v_lshlrev_b32_e32 v94, 4, v7
	v_lshlrev_b32_e32 v95, 2, v0
	s_mov_b32 s3, 0
	s_lshl_b64 s[10:11], s[10:11], 9
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[10:11], s[10:11], s[18:19]
	s_add_nc_u64 s[4:5], s[4:5], s[12:13]
	s_add_nc_u64 s[6:7], s[6:7], s[12:13]
	s_mov_b32 s2, s3
	s_branch .LBB0_11
.LBB0_8:                                ; %_ZL11fa2_scale_nPKhiii.exit166.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	global_store_b32 v[1:2], v3, off offset:33024
.LBB0_9:                                ; %Flow
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s2, s2, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s2, 0x3fffffff
	s_cselect_b32 s12, -1, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB0_10:                               ; %Flow1237
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s12
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_321
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s14, s2, 6
	s_mov_b32 s12, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s14, s1
	s_cbranch_scc1 .LBB0_10
; %bb.12:                               ; %.preheader178.preheader.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v3, s14, v44
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_14
; %bb.13:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[3:4], null, 0x408, v3, v[14:15]
	global_load_b128 v[5:8], v[3:4], off
.LBB0_14:                               ; %.preheader178.1.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_dual_mov_b32 v4, 0 :: v_dual_add_nc_u32 v9, s14, v46
	v_mov_b32_e32 v3, 0
	s_mov_b32 s12, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v71, v[5:6], v[7:8] offset1:16
	v_cmpx_ge_i32_e64 s1, v9
	s_cbranch_execz .LBB0_16
; %bb.15:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v9, v[14:15]
	global_load_b128 v[1:4], v[1:2], off
.LBB0_16:                               ; %.preheader178.2.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_dual_mov_b32 v6, 0 :: v_dual_add_nc_u32 v7, s14, v47
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v12, 0
	v_mov_b32_e32 v11, 0
	s_mov_b32 s12, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v72, v[1:2], v[3:4] offset1:16
	v_cmpx_ge_i32_e64 s1, v7
	s_cbranch_execz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v7, v[14:15]
	global_load_b128 v[9:12], v[1:2], off
.LBB0_18:                               ; %.preheader178.3.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v1, s14, v48
	v_mov_b32_e32 v7, 0
	s_mov_b32 s12, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v73, v[9:10], v[11:12] offset1:16
	v_cmpx_ge_i32_e64 s1, v1
	s_cbranch_execz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[14:15]
	global_load_b128 v[5:8], v[1:2], off
.LBB0_20:                               ; %.preheader178.4.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v3, s14, v49
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v12, 0
	v_mov_b32_e32 v11, 0
	s_mov_b32 s12, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v74, v[5:6], v[7:8] offset1:16
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[3:4], null, 0x408, v3, v[14:15]
	global_load_b128 v[9:12], v[3:4], off
.LBB0_22:                               ; %.preheader178.5.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_dual_mov_b32 v4, 0 :: v_dual_add_nc_u32 v5, s14, v50
	v_mov_b32_e32 v3, 0
	s_mov_b32 s12, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v75, v[9:10], v[11:12] offset1:16
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v5, v[14:15]
	global_load_b128 v[1:4], v[1:2], off
.LBB0_24:                               ; %.preheader178.6.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_dual_mov_b32 v6, 0 :: v_dual_add_nc_u32 v7, s14, v51
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v12, 0
	v_mov_b32_e32 v11, 0
	s_mov_b32 s12, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v76, v[1:2], v[3:4] offset1:16
	v_cmpx_ge_i32_e64 s1, v7
	s_cbranch_execz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v7, v[14:15]
	global_load_b128 v[9:12], v[1:2], off
.LBB0_26:                               ; %.preheader178.7.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v1, s14, v52
	v_mov_b32_e32 v7, 0
	s_mov_b32 s12, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v77, v[9:10], v[11:12] offset1:16
	v_cmpx_ge_i32_e64 s1, v1
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[14:15]
	global_load_b128 v[5:8], v[1:2], off
.LBB0_28:                               ; %.preheader177.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_or_b32_e32 v3, s14, v45
	s_mov_b32 s12, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v78, v[5:6], v[7:8] offset1:16
                                        ; implicit-def: $vgpr1_vgpr2
	v_or_b32_e32 v4, 7, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[16:17]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB0_30:                               ; %Flow1236
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_46
; %bb.31:                               ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_45
; %bb.32:                               ; %.preheader.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[16:17]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v3
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[16:17]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_34:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB0_36:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB0_38:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_40:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB0_42:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[16:17]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB0_44:                               ; %Flow1234
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_45:                               ; %Flow1235
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_46:                               ; %.loopexit.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_add_nc_u32_e32 v5, v54, v53
	s_mov_b32 s12, exec_lo
	ds_store_b64 v5, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_48
; %bb.47:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[20:21]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB0_48:                               ; %Flow1233
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_64
; %bb.49:                               ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_63
; %bb.50:                               ; %.preheader.1.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[20:21]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v3
	s_cbranch_execz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[20:21]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_52:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[20:21]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB0_54:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[20:21]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB0_56:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[20:21]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_58:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[20:21]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB0_60:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[20:21]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB0_62:                               ; %Flow1231
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_63:                               ; %Flow1232
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_64:                               ; %.loopexit.1.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	ds_store_b64 v79, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[22:23]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB0_66:                               ; %Flow1230
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_82
; %bb.67:                               ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_81
; %bb.68:                               ; %.preheader.2.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[22:23]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v3
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[22:23]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_70:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[22:23]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB0_72:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[22:23]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB0_74:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[22:23]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_76:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[22:23]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB0_78:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[22:23]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB0_80:                               ; %Flow1228
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_81:                               ; %Flow1229
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_82:                               ; %.loopexit.2.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	ds_store_b64 v80, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[24:25]
	v_mov_b16_e32 v6.h, 0
	s_clause 0x4
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v5, v[1:2], off offset:2064
	global_load_d16_u8 v6, v[1:2], off
	global_load_u8 v7, v[1:2], off offset:3096
	global_load_u8 v8, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v9.l, v6.h
	s_clause 0x2
	global_load_u8 v10, v[1:2], off offset:4128
	global_load_u8 v11, v[1:2], off offset:7224
	global_load_d16_hi_u8 v9, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v4
	v_lshlrev_b32_e32 v2, 16, v5
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v7
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v5, 8, v8
	v_or_b32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v10, v5
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v11
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v9, v4
	v_or3_b32 v1, v1, 0, 0
.LBB0_84:                               ; %Flow1227
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_100
; %bb.85:                               ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_99
; %bb.86:                               ; %.preheader.3.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[24:25]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v3
	s_cbranch_execz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[24:25]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_88:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 2, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[24:25]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v4, 16, v1
.LBB0_90:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 3, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_92
; %bb.91:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[24:25]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v4, 24, v1
.LBB0_92:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 4, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_94
; %bb.93:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[24:25]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v4, v2
.LBB0_94:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 5, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_96
; %bb.95:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[24:25]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v4, 8, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v4, v2
.LBB0_96:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 6, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_98
; %bb.97:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[24:25]
	v_mov_b16_e32 v6.l, 0
	global_load_d16_hi_u8 v6, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB0_98:                               ; %Flow1225
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_99:                               ; %Flow1226
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_100:                              ; %.loopexit.3.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_or_b32_e32 v4, 16, v3
	s_mov_b32 s12, exec_lo
	ds_store_b64 v81, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_or_b32_e32 v5, 7, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_102
; %bb.101:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[18:19]
	v_mov_b16_e32 v8.h, 0
	s_clause 0x4
	global_load_u8 v6, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v8, v[1:2], off
	global_load_u8 v9, v[1:2], off offset:3096
	global_load_u8 v10, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v11.l, v8.h
	s_clause 0x2
	global_load_u8 v12, v[1:2], off offset:4128
	global_load_u8 v96, v[1:2], off offset:7224
	global_load_d16_hi_u8 v11, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v6, 24, v9
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v10
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v6
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v12, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 24, v96
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v11, v6
	v_or3_b32 v1, v1, 0, 0
.LBB0_102:                              ; %Flow1224
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_118
; %bb.103:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_117
; %bb.104:                              ; %.preheader.4.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[18:19]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v4
	s_cbranch_execz .LBB0_106
; %bb.105:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[18:19]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_106:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 2, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_108
; %bb.107:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[18:19]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 16, v1
.LBB0_108:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 3, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_110
; %bb.109:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[18:19]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 24, v1
.LBB0_110:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 4, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_112
; %bb.111:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[18:19]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB0_112:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 5, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_114
; %bb.113:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[18:19]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v6, v2
.LBB0_114:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 6, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_116
; %bb.115:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[18:19]
	v_mov_b16_e32 v8.l, 0
	global_load_d16_hi_u8 v8, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB0_116:                              ; %Flow1222
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_117:                              ; %Flow1223
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_118:                              ; %.loopexit.4.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	ds_store_b64 v82, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_120
; %bb.119:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[26:27]
	v_mov_b16_e32 v8.h, 0
	s_clause 0x4
	global_load_u8 v6, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v8, v[1:2], off
	global_load_u8 v9, v[1:2], off offset:3096
	global_load_u8 v10, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v11.l, v8.h
	s_clause 0x2
	global_load_u8 v12, v[1:2], off offset:4128
	global_load_u8 v96, v[1:2], off offset:7224
	global_load_d16_hi_u8 v11, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v6, 24, v9
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v10
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v6
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v12, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 24, v96
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v11, v6
	v_or3_b32 v1, v1, 0, 0
.LBB0_120:                              ; %Flow1221
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_136
; %bb.121:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_135
; %bb.122:                              ; %.preheader.5.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[26:27]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v4
	s_cbranch_execz .LBB0_124
; %bb.123:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[26:27]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_124:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 2, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_126
; %bb.125:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[26:27]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 16, v1
.LBB0_126:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 3, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_128
; %bb.127:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[26:27]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 24, v1
.LBB0_128:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 4, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_130
; %bb.129:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[26:27]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB0_130:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 5, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_132
; %bb.131:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[26:27]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v6, v2
.LBB0_132:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 6, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_134
; %bb.133:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[26:27]
	v_mov_b16_e32 v8.l, 0
	global_load_d16_hi_u8 v8, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB0_134:                              ; %Flow1219
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_135:                              ; %Flow1220
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_136:                              ; %.loopexit.5.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	ds_store_b64 v83, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_138
; %bb.137:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[28:29]
	v_mov_b16_e32 v8.h, 0
	s_clause 0x4
	global_load_u8 v6, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v8, v[1:2], off
	global_load_u8 v9, v[1:2], off offset:3096
	global_load_u8 v10, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v11.l, v8.h
	s_clause 0x2
	global_load_u8 v12, v[1:2], off offset:4128
	global_load_u8 v96, v[1:2], off offset:7224
	global_load_d16_hi_u8 v11, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v6, 24, v9
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v10
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v6
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v12, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 24, v96
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v11, v6
	v_or3_b32 v1, v1, 0, 0
.LBB0_138:                              ; %Flow1218
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_154
; %bb.139:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_153
; %bb.140:                              ; %.preheader.6.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[28:29]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v4
	s_cbranch_execz .LBB0_142
; %bb.141:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[28:29]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_142:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 2, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_144
; %bb.143:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[28:29]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 16, v1
.LBB0_144:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 3, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_146
; %bb.145:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[28:29]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 24, v1
.LBB0_146:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 4, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_148
; %bb.147:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[28:29]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB0_148:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 5, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_150
; %bb.149:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[28:29]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v6, v2
.LBB0_150:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 6, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_152
; %bb.151:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[28:29]
	v_mov_b16_e32 v8.l, 0
	global_load_d16_hi_u8 v8, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB0_152:                              ; %Flow1216
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_153:                              ; %Flow1217
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_154:                              ; %.loopexit.6.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	ds_store_b64 v84, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_156
; %bb.155:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[30:31]
	v_mov_b16_e32 v6.h, 0
	s_clause 0x4
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v5, v[1:2], off offset:2064
	global_load_d16_u8 v6, v[1:2], off
	global_load_u8 v7, v[1:2], off offset:3096
	global_load_u8 v8, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v9.l, v6.h
	s_clause 0x2
	global_load_u8 v10, v[1:2], off offset:4128
	global_load_u8 v11, v[1:2], off offset:7224
	global_load_d16_hi_u8 v9, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v4
	v_lshlrev_b32_e32 v2, 16, v5
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v7
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v5, 8, v8
	v_or_b32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v10, v5
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v11
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v9, v4
                                        ; implicit-def: $vgpr4
	v_or3_b32 v1, v1, 0, 0
.LBB0_156:                              ; %Flow1215
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_172
; %bb.157:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_171
; %bb.158:                              ; %.preheader.7.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[30:31]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v4
	s_cbranch_execz .LBB0_160
; %bb.159:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[30:31]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_160:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 2, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_162
; %bb.161:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[30:31]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB0_162:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 3, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_164
; %bb.163:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[30:31]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB0_164:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 4, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_166
; %bb.165:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[30:31]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_166:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 5, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_168
; %bb.167:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[30:31]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB0_168:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 6, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_170
; %bb.169:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[30:31]
	v_mov_b16_e32 v6.l, 0
	global_load_d16_hi_u8 v6, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB0_170:                              ; %Flow1213
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_171:                              ; %Flow1214
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_172:                              ; %.loopexit.7.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_or_b32_e32 v4, 32, v3
	s_mov_b32 s12, exec_lo
	ds_store_b64 v85, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_or_b32_e32 v5, 7, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_174
; %bb.173:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[16:17]
	v_mov_b16_e32 v6.h, 0
	s_clause 0x4
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v5, v[1:2], off offset:2064
	global_load_d16_u8 v6, v[1:2], off
	global_load_u8 v7, v[1:2], off offset:3096
	global_load_u8 v8, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v9.l, v6.h
	s_clause 0x2
	global_load_u8 v10, v[1:2], off offset:4128
	global_load_u8 v11, v[1:2], off offset:7224
	global_load_d16_hi_u8 v9, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v4
	v_lshlrev_b32_e32 v2, 16, v5
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v7
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v5, 8, v8
	v_or_b32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v10, v5
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v11
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v9, v4
                                        ; implicit-def: $vgpr4
	v_or3_b32 v1, v1, 0, 0
.LBB0_174:                              ; %Flow1212
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_190
; %bb.175:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_189
; %bb.176:                              ; %.preheader.8.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[16:17]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v4
	s_cbranch_execz .LBB0_178
; %bb.177:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[16:17]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_178:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 2, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_180
; %bb.179:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB0_180:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 3, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_182
; %bb.181:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB0_182:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 4, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_184
; %bb.183:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_184:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 5, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_186
; %bb.185:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB0_186:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 6, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_188
; %bb.187:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[16:17]
	v_mov_b16_e32 v6.l, 0
	global_load_d16_hi_u8 v6, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB0_188:                              ; %Flow1210
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_189:                              ; %Flow1211
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_190:                              ; %.loopexit.8.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_add_nc_u32_e32 v4, 32, v3
	s_mov_b32 s12, exec_lo
	ds_store_b64 v86, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_or_b32_e32 v5, 7, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_192
; %bb.191:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[32:33]
	v_mov_b16_e32 v8.h, 0
	s_clause 0x4
	global_load_u8 v6, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v8, v[1:2], off
	global_load_u8 v9, v[1:2], off offset:3096
	global_load_u8 v10, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v11.l, v8.h
	s_clause 0x2
	global_load_u8 v12, v[1:2], off offset:4128
	global_load_u8 v96, v[1:2], off offset:7224
	global_load_d16_hi_u8 v11, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v6, 24, v9
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v10
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v6
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v12, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 24, v96
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v11, v6
	v_or3_b32 v1, v1, 0, 0
.LBB0_192:                              ; %Flow1209
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_208
; %bb.193:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_207
; %bb.194:                              ; %.preheader.9.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[32:33]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v4
	s_cbranch_execz .LBB0_196
; %bb.195:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[32:33]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_196:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 2, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_198
; %bb.197:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[32:33]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 16, v1
.LBB0_198:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 3, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_200
; %bb.199:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[32:33]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 24, v1
.LBB0_200:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 4, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_202
; %bb.201:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[32:33]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB0_202:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 5, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_204
; %bb.203:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[32:33]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v6, v2
.LBB0_204:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 6, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_206
; %bb.205:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[32:33]
	v_mov_b16_e32 v8.l, 0
	global_load_d16_hi_u8 v8, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB0_206:                              ; %Flow1207
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_207:                              ; %Flow1208
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_208:                              ; %.loopexit.9.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	ds_store_b64 v87, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_210
; %bb.209:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[34:35]
	v_mov_b16_e32 v8.h, 0
	s_clause 0x4
	global_load_u8 v6, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v8, v[1:2], off
	global_load_u8 v9, v[1:2], off offset:3096
	global_load_u8 v10, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v11.l, v8.h
	s_clause 0x2
	global_load_u8 v12, v[1:2], off offset:4128
	global_load_u8 v96, v[1:2], off offset:7224
	global_load_d16_hi_u8 v11, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v6, 24, v9
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v10
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v6
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v12, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 24, v96
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v11, v6
	v_or3_b32 v1, v1, 0, 0
.LBB0_210:                              ; %Flow1206
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_226
; %bb.211:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_225
; %bb.212:                              ; %.preheader.10.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[34:35]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v4
	s_cbranch_execz .LBB0_214
; %bb.213:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[34:35]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_214:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 2, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_216
; %bb.215:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[34:35]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 16, v1
.LBB0_216:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 3, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_218
; %bb.217:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[34:35]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 24, v1
.LBB0_218:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 4, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_220
; %bb.219:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[34:35]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB0_220:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 5, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_222
; %bb.221:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[34:35]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v6, v2
.LBB0_222:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v6, 6, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v6
	s_cbranch_execz .LBB0_224
; %bb.223:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[6:7], null, 0x408, v6, v[34:35]
	v_mov_b16_e32 v8.l, 0
	global_load_d16_hi_u8 v8, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB0_224:                              ; %Flow1204
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_225:                              ; %Flow1205
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_226:                              ; %.loopexit.10.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	ds_store_b64 v88, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_228
; %bb.227:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[36:37]
	v_mov_b16_e32 v6.h, 0
	s_clause 0x4
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v5, v[1:2], off offset:2064
	global_load_d16_u8 v6, v[1:2], off
	global_load_u8 v7, v[1:2], off offset:3096
	global_load_u8 v8, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v9.l, v6.h
	s_clause 0x2
	global_load_u8 v10, v[1:2], off offset:4128
	global_load_u8 v11, v[1:2], off offset:7224
	global_load_d16_hi_u8 v9, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v4
	v_lshlrev_b32_e32 v2, 16, v5
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v7
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v5, 8, v8
	v_or_b32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v10, v5
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v11
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v9, v4
                                        ; implicit-def: $vgpr4
	v_or3_b32 v1, v1, 0, 0
.LBB0_228:                              ; %Flow1203
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_244
; %bb.229:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_243
; %bb.230:                              ; %.preheader.11.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, v[36:37]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v4
	s_cbranch_execz .LBB0_232
; %bb.231:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[36:37]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_232:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 2, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_234
; %bb.233:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[36:37]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB0_234:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 3, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_236
; %bb.235:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[36:37]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB0_236:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 4, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_238
; %bb.237:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[36:37]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_238:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 5, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_240
; %bb.239:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[36:37]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB0_240:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 6, v4
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_242
; %bb.241:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[36:37]
	v_mov_b16_e32 v6.l, 0
	global_load_d16_hi_u8 v6, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB0_242:                              ; %Flow1201
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_243:                              ; %Flow1202
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_244:                              ; %.loopexit.11.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_add_nc_u32_e32 v3, 48, v3
	s_mov_b32 s12, exec_lo
	ds_store_b64 v89, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_or_b32_e32 v4, 7, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_246
; %bb.245:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[18:19]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB0_246:                              ; %Flow1200
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_262
; %bb.247:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_261
; %bb.248:                              ; %.preheader.12.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[18:19]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v3
	s_cbranch_execz .LBB0_250
; %bb.249:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[18:19]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_250:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_252
; %bb.251:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[18:19]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB0_252:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_254
; %bb.253:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[18:19]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB0_254:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_256
; %bb.255:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[18:19]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_256:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_258
; %bb.257:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[18:19]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB0_258:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_260
; %bb.259:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[18:19]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB0_260:                              ; %Flow1198
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_261:                              ; %Flow1199
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_262:                              ; %.loopexit.12.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	ds_store_b64 v90, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_264
; %bb.263:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[38:39]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB0_264:                              ; %Flow1197
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_280
; %bb.265:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_279
; %bb.266:                              ; %.preheader.13.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[38:39]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v3
	s_cbranch_execz .LBB0_268
; %bb.267:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[38:39]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_268:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_270
; %bb.269:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[38:39]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB0_270:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_272
; %bb.271:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[38:39]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB0_272:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_274
; %bb.273:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[38:39]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_274:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_276
; %bb.275:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[38:39]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB0_276:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_278
; %bb.277:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[38:39]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB0_278:                              ; %Flow1195
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_279:                              ; %Flow1196
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_280:                              ; %.loopexit.13.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	ds_store_b64 v91, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_282
; %bb.281:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[40:41]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB0_282:                              ; %Flow1194
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_298
; %bb.283:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_297
; %bb.284:                              ; %.preheader.14.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[40:41]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v3
	s_cbranch_execz .LBB0_286
; %bb.285:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[40:41]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_286:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_288
; %bb.287:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[40:41]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB0_288:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_290
; %bb.289:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[40:41]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB0_290:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_292
; %bb.291:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[40:41]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_292:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_294
; %bb.293:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[40:41]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB0_294:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v5
	s_cbranch_execz .LBB0_296
; %bb.295:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[5:6], null, 0x408, v5, v[40:41]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB0_296:                              ; %Flow1192
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_297:                              ; %Flow1193
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_298:                              ; %.loopexit.14.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	ds_store_b64 v92, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s1, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, exec_lo, s12
	s_cbranch_execz .LBB0_300
; %bb.299:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[42:43]
	v_mov_b16_e32 v5.h, 0
	s_clause 0x4
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v5, v[1:2], off
	global_load_u8 v6, v[1:2], off offset:3096
	global_load_u8 v7, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v8.l, v5.h
	s_clause 0x2
	global_load_u8 v9, v[1:2], off offset:4128
	global_load_u8 v10, v[1:2], off offset:7224
	global_load_d16_hi_u8 v8, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v3
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v6
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v7
	v_or_b32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v9, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v10
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v8, v3
                                        ; implicit-def: $vgpr3
	v_or3_b32 v1, v1, 0, 0
.LBB0_300:                              ; %Flow1191
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s12, s12
	s_cbranch_execz .LBB0_316
; %bb.301:                              ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_315
; %bb.302:                              ; %.preheader.15.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v3, v[42:43]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s15, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s1, v3
	s_cbranch_execz .LBB0_304
; %bb.303:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, v[42:43]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB0_304:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 2, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_306
; %bb.305:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[42:43]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v4, 16, v1
.LBB0_306:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 3, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_308
; %bb.307:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[42:43]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v4, 24, v1
.LBB0_308:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 4, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_310
; %bb.309:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[42:43]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v4, v2
.LBB0_310:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v4, 5, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v4
	s_cbranch_execz .LBB0_312
; %bb.311:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, v[42:43]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v4, 8, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v4, v2
.LBB0_312:                              ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v3, 6, v3
	s_mov_b32 s15, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s1, v3
	s_cbranch_execz .LBB0_314
; %bb.313:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[3:4], null, 0x408, v3, v[42:43]
	v_mov_b16_e32 v5.l, 0
	global_load_d16_hi_u8 v5, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_314:                              ; %Flow1189
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_315:                              ; %Flow1190
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_316:                              ; %.loopexit.15.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	ds_store_b64 v93, v[1:2] offset:16384
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_add_nc_u64 s[12:13], s[10:11], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[12:13], s[12:13], 0x8200
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[8:9], s[12:13]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_b32 v[3:4], v55 offset0:2 offset1:3
	ds_load_2addr_b32 v[7:8], v56 offset0:2 offset1:3
	ds_load_2addr_b32 v[5:6], v56 offset1:1
	ds_load_2addr_b32 v[1:2], v55 offset1:1
	ds_load_2addr_b32 v[11:12], v57 offset0:2 offset1:3
	ds_load_2addr_b32 v[98:99], v58 offset0:2 offset1:3
	ds_load_2addr_b32 v[96:97], v58 offset1:1
	ds_load_2addr_b32 v[9:10], v57 offset1:1
	ds_load_2addr_b32 v[102:103], v59 offset0:2 offset1:3
	ds_load_2addr_b32 v[106:107], v60 offset0:2 offset1:3
	ds_load_2addr_b32 v[104:105], v60 offset1:1
	ds_load_2addr_b32 v[100:101], v59 offset1:1
	ds_load_2addr_b32 v[110:111], v61 offset0:2 offset1:3
	ds_load_2addr_b32 v[114:115], v62 offset0:2 offset1:3
	ds_load_2addr_b32 v[112:113], v62 offset1:1
	ds_load_2addr_b32 v[108:109], v61 offset1:1
	ds_load_2addr_b32 v[118:119], v63 offset0:2 offset1:3
	ds_load_2addr_b32 v[122:123], v64 offset0:2 offset1:3
	ds_load_2addr_b32 v[120:121], v64 offset1:1
	ds_load_2addr_b32 v[116:117], v63 offset1:1
	ds_load_2addr_b32 v[126:127], v65 offset0:2 offset1:3
	ds_load_2addr_b32 v[130:131], v66 offset0:2 offset1:3
	ds_load_2addr_b32 v[128:129], v66 offset1:1
	ds_load_2addr_b32 v[124:125], v65 offset1:1
	ds_load_2addr_b32 v[134:135], v67 offset0:2 offset1:3
	ds_load_2addr_b32 v[138:139], v68 offset0:2 offset1:3
	ds_load_2addr_b32 v[136:137], v68 offset1:1
	ds_load_2addr_b32 v[132:133], v67 offset1:1
	ds_load_2addr_b32 v[142:143], v69 offset0:2 offset1:3
	ds_load_2addr_b32 v[144:145], v70 offset1:1
	ds_load_2addr_b32 v[140:141], v69 offset1:1
	ds_load_2addr_b32 v[146:147], v70 offset0:2 offset1:3
	s_wait_dscnt 0x1c
	s_clause 0x1
	global_store_b128 v94, v[1:4], s[12:13]
	global_store_b128 v94, v[5:8], s[12:13] offset:16
	s_wait_dscnt 0x18
	s_clause 0x1
	global_store_b128 v94, v[9:12], s[12:13] offset:32
	global_store_b128 v94, v[96:99], s[12:13] offset:48
	s_wait_dscnt 0x14
	s_clause 0x1
	global_store_b128 v94, v[100:103], s[12:13] offset:64
	global_store_b128 v94, v[104:107], s[12:13] offset:80
	s_wait_dscnt 0x10
	s_clause 0x1
	global_store_b128 v94, v[108:111], s[12:13] offset:96
	global_store_b128 v94, v[112:115], s[12:13] offset:112
	s_wait_dscnt 0xc
	s_clause 0x1
	global_store_b128 v94, v[116:119], s[12:13] offset:128
	global_store_b128 v94, v[120:123], s[12:13] offset:144
	s_wait_dscnt 0x8
	s_clause 0x1
	global_store_b128 v94, v[124:127], s[12:13] offset:160
	global_store_b128 v94, v[128:131], s[12:13] offset:176
	s_wait_dscnt 0x4
	s_clause 0x1
	global_store_b128 v94, v[132:135], s[12:13] offset:192
	global_store_b128 v94, v[136:139], s[12:13] offset:208
	s_wait_dscnt 0x1
	global_store_b128 v94, v[140:143], s[12:13] offset:224
	s_wait_dscnt 0x0
	global_store_b128 v94, v[144:147], s[12:13] offset:240
	s_and_saveexec_b32 s15, s0
	s_cbranch_execz .LBB0_9
; %bb.317:                              ;   in Loop: Header=BB0_11 Depth=1
	v_or_b32_e32 v4, s14, v0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v5, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e32 vcc_lo, s1, v4
	s_and_saveexec_b32 s14, vcc_lo
	s_cbranch_execz .LBB0_319
; %bb.318:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[1:2], null, 0x408, v4, s[4:5]
	global_load_d16_b16 v1, v[1:2], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v5, v1.l
.LBB0_319:                              ; %_ZL11fa2_scale_nPKhiii.exit.i
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	v_add_co_u32 v1, s14, s12, v95
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s13, 0, s14
	global_store_b32 v95, v5, s[12:13] offset:32768
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB0_8
; %bb.320:                              ;   in Loop: Header=BB0_11 Depth=1
	v_mad_co_u64_u32 v[3:4], null, 0x408, v4, s[6:7]
	global_load_d16_b16 v3, v[3:4], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v3, v3.l
	s_branch .LBB0_8
.LBB0_321:                              ; %_Z21fa2_stageb_nfill_dumpILb0EEvPKhS1_PfPKiiiiiiii.exit
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
		.amdhsa_next_free_vgpr 148
		.amdhsa_next_free_sgpr 22
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
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.num_vgpr, 148
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.numbered_sgpr, 22
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15424
; TotalNumSgprs: 24
; NumVgprs: 148
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 18
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 148
; Occupancy: 9
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
	s_load_b128 s[12:15], s[0:1], 0x28
	s_load_b32 s16, s[0:1], 0x3c
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s12, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s13, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s14, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s4, s16, -9
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s4, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB3_322
; %bb.1:
	s_and_b32 s12, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s12, 3
	s_cbranch_scc1 .LBB3_322
; %bb.2:
	s_lshl_b32 s2, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s2, s15
	s_cbranch_scc1 .LBB3_322
; %bb.3:
	s_lshr_b32 s14, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s14, s16
	s_cbranch_scc1 .LBB3_322
; %bb.4:
	s_load_b256 s[4:11], s[0:1], 0x8
	v_dual_mov_b32 v2, -1 :: v_dual_and_b32 v1, 31, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e32 8, v1
	s_cbranch_execz .LBB3_8
; %bb.5:
	v_or_b32_e32 v1, s2, v1
	v_mov_b32_e32 v2, -1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s15, v1
	s_cbranch_execz .LBB3_7
; %bb.6:
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, vcc_lo, s10, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s11, v2, vcc_lo
	global_load_b32 v2, v[1:2], off
.LBB3_7:                                ; %Flow1264
	s_or_b32 exec_lo, exec_lo, s1
.LBB3_8:                                ; %.lr.ph.i
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	s_or_b32 exec_lo, exec_lo, s0
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_dual_mov_b32 v13, 0 :: v_dual_lshlrev_b32 v6, 3, v0
	v_lshrrev_b32_e32 v15, 5, v0
	s_cvt_f32_u32 s3, s16
	v_xor_b32_e32 v3, 16, v1
	v_xor_b32_e32 v14, 8, v1
	v_xor_b32_e32 v16, 2, v1
	v_lshlrev_b32_e32 v53, 8, v15
	v_xor_b32_e32 v17, 1, v1
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	s_add_co_i32 s1, s16, 0x1ff
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s10, s3
	s_and_b32 s1, s1, 0xffff
	v_or_b32_e32 v8, 0x80, v0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v3, v1, v3 :: v_dual_and_b32 v4, 15, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v14
	s_cvt_f32_u32 s1, s1
	v_or_b32_e32 v9, 0x100, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v3, 2, v3
	v_lshl_or_b32 v21, v15, 4, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v1, v14, vcc_lo
	s_mul_f32 s10, s1, s10
	v_or_b32_e32 v10, 0x180, v0
	s_wait_loadcnt 0x0
	ds_bpermute_b32 v3, v3, v2
	v_or_b32_e32 v11, 0x200, v0
	s_wait_alu depctr_sa_sdst(0)
	s_trunc_f32 s10, s10
	v_or_b32_e32 v12, 0x280, v0
	v_or_b32_e32 v26, 0x300, v0
	v_or_b32_e32 v28, 0x380, v0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s18, s10, 0x80000000
	v_lshrrev_b32_e32 v44, 4, v0
	s_fmac_f32 s1, s18, s3
	s_cvt_u32_f32 s10, s10
	s_mov_b32 s13, 0
	v_lshrrev_b32_e32 v46, 4, v8
	s_bitset0_b32 s1, 31
	v_lshrrev_b32_e32 v47, 4, v9
	s_cmp_ge_f32 s1, s3
	v_lshrrev_b32_e32 v48, 4, v10
	v_lshrrev_b32_e32 v49, 4, v11
	v_lshrrev_b32_e32 v50, 4, v12
	v_lshrrev_b32_e32 v51, 4, v26
	v_lshrrev_b32_e32 v52, 4, v28
	s_wait_dscnt 0x0
	v_max_i32_e32 v2, v2, v3
	v_xor_b32_e32 v3, 4, v1
	v_lshrrev_b32_e32 v8, 5, v8
	v_lshlrev_b32_e32 v19, 4, v4
	s_mov_b32 s11, s13
	v_lshrrev_b32_e32 v9, 5, v9
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	v_lshlrev_b32_e32 v14, 2, v14
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_ci_u32 s1, s10, 0
	s_lshl_b32 s10, s12, 8
	s_mov_b32 s2, ttmp9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v1, v3, vcc_lo
	ds_bpermute_b32 v14, v14, v2
	v_cmp_gt_u32_e32 vcc_lo, 32, v16
	v_and_b32_e32 v20, 0xf8, v6
	v_lshrrev_b32_e32 v10, 5, v10
	v_lshlrev_b32_e32 v3, 2, v3
	s_ashr_i32 s3, ttmp9, 31
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, v1, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v17
	v_lshlrev_b32_e32 v18, 7, v4
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[20:21], s[4:5], s[10:11]
	v_add_nc_u32_e32 v54, 0, v20
	v_lshlrev_b32_e32 v15, 2, v15
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v17, vcc_lo
	v_or_b32_e32 v16, v44, v18
	v_or_b32_e32 v17, v18, v46
	v_and_or_b32 v20, v47, 7, v18
	v_and_or_b32 v22, v48, 15, v18
	v_lshlrev_b32_e32 v1, 2, v1
	v_and_or_b32 v23, v49, 7, v18
	v_and_or_b32 v24, v50, 15, v18
	s_wait_dscnt 0x0
	v_max_i32_e32 v2, v2, v14
	v_lshrrev_b32_e32 v14, 1, v12
	v_and_or_b32 v25, v51, 7, v18
	v_and_or_b32 v18, v52, 15, v18
	v_lshl_or_b32 v27, v8, 4, v4
	ds_bpermute_b32 v3, v3, v2
	v_lshrrev_b32_e32 v5, 1, v0
	v_and_or_b32 v30, 0x70, v14, v4
	s_add_nc_u64 s[18:19], s[6:7], s[10:11]
	s_lshl_b64 s[10:11], s[2:3], 2
	v_add_co_u32 v14, s3, s20, v19
	v_lshl_or_b32 v29, v9, 4, v4
	v_lshlrev_b32_e32 v71, 3, v16
	v_lshlrev_b32_e32 v72, 3, v17
	v_lshl_or_b32 v78, v18, 3, 0x300
	v_and_b32_e32 v45, 8, v5
	v_and_or_b32 v5, v5, 48, v4
	v_lshl_or_b32 v73, v20, 3, 0x100
	v_lshlrev_b32_e32 v12, 3, v12
	s_mov_b32 s17, s13
	v_lshl_or_b32 v74, v22, 3, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[10:11], s[10:11], s[12:13]
	s_mov_b32 s15, s13
	v_lshl_or_b32 v75, v23, 3, 0x200
	s_wait_dscnt 0x0
	v_max_i32_e32 v2, v2, v3
	v_lshl_or_b32 v3, v10, 4, v4
	v_lshl_or_b32 v76, v24, 3, 0x200
	s_and_b32 s1, s1, 0xffff
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[10:11], s[10:11], s[16:17]
	ds_bpermute_b32 v15, v15, v2
	v_lshl_or_b32 v77, v25, 3, 0x300
	s_mul_i32 s2, s14, s1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[10:11], s[10:11], s[14:15]
	v_or_b32_e32 v33, 0x500, v0
	v_or_b32_e32 v36, 0x580, v0
	v_or_b32_e32 v38, 0x680, v0
	v_or_b32_e32 v39, 0x700, v0
	v_or_b32_e32 v40, 0x780, v0
	v_lshrrev_b32_e32 v34, 1, v33
	v_lshrrev_b32_e32 v37, 1, v36
	v_lshlrev_b32_e32 v35, 3, v33
	v_lshlrev_b32_e32 v36, 3, v36
	v_lshrrev_b32_e32 v41, 1, v39
	v_lshlrev_b32_e32 v42, 3, v39
	v_lshrrev_b32_e32 v43, 1, v40
	v_and_b32_e32 v88, 0x2b00, v35
	v_and_b32_e32 v89, 0x2f00, v36
	v_lshlrev_b32_e32 v11, 3, v11
	v_or_b32_e32 v6, 0x3000, v6
	s_wait_dscnt 0x0
	v_max_i32_e32 v2, v2, v15
	v_add_co_ci_u32_e64 v15, null, s21, 0, s3
	v_add_co_u32 v16, s3, s18, v21
	ds_bpermute_b32 v1, v1, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s19, 0, s3
	v_add_co_u32 v18, s3, s18, v27
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v19, null, s19, 0, s3
	v_add_co_u32 v20, s3, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v21, null, s19, 0, s3
	v_add_co_u32 v22, s3, s18, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v23, null, s19, 0, s3
	v_add_co_u32 v24, s3, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v25, null, s19, 0, s3
	v_lshlrev_b32_e32 v3, 3, v26
	s_wait_dscnt 0x0
	v_max_i32_e32 v1, v2, v1
	v_lshrrev_b32_e32 v2, 1, v26
	v_add_co_u32 v26, s3, s18, v30
	v_lshrrev_b32_e32 v5, 1, v28
	s_delay_alu instid0(VALU_DEP_4)
	v_readfirstlane_b32 s14, v1
	v_and_b32_e32 v1, 0x1700, v12
	v_and_or_b32 v2, 0xb0, v2, v4
	v_lshlrev_b32_e32 v12, 3, v28
	v_or_b32_e32 v30, 0x480, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v27, null, s19, 0, s3
	v_add_co_u32 v28, s3, s18, v2
	v_and_or_b32 v2, 0xf0, v5, v4
	v_and_b32_e32 v5, 0x1f00, v12
	v_lshrrev_b32_e32 v12, 1, v30
	v_lshlrev_b32_e32 v32, 3, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, s19, 0, s3
	v_add_co_u32 v30, s3, s18, v2
	v_and_or_b32 v12, 0x70, v12, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, s19, 0, s3
	v_and_b32_e32 v87, 0x2700, v32
	v_and_b32_e32 v92, 0x3b00, v42
	v_add_co_u32 v32, s3, s18, v12
	v_and_or_b32 v12, 0xb0, v34, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v33, null, s19, 0, s3
	v_lshlrev_b32_e32 v42, 3, v40
	v_lshlrev_b32_e32 v7, 4, v0
	v_add_co_u32 v34, s3, s18, v12
	v_and_or_b32 v12, 0xf0, v37, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v35, null, s19, 0, s3
	v_lshlrev_b32_e32 v8, 8, v8
	v_lshlrev_b32_e32 v9, 8, v9
	v_add_co_u32 v36, s3, s18, v12
	v_lshrrev_b32_e32 v12, 1, v38
	v_lshlrev_b32_e32 v38, 3, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v37, null, s19, 0, s3
	v_lshlrev_b32_e32 v10, 8, v10
	v_and_or_b32 v12, 0x70, v12, v4
	v_and_b32_e32 v91, 0x3700, v38
	v_and_b32_e32 v11, 0x1300, v11
	v_and_b32_e32 v3, 0x1b00, v3
	v_or_b32_e32 v2, 0x2000, v53
	v_add_co_u32 v38, s3, s18, v12
	v_and_or_b32 v12, 0xb0, v41, v4
	v_and_or_b32 v4, 0xf0, v43, v4
	v_and_b32_e32 v6, 0x3300, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v39, null, s19, 0, s3
	v_add_co_u32 v40, s3, s18, v12
	v_and_b32_e32 v12, 0x3f00, v42
	v_lshl_add_u32 v55, v0, 8, 0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v41, null, s19, 0, s3
	v_add_co_u32 v42, s3, s18, v4
	v_cmp_gt_u32_e64 s0, 64, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v43, null, s19, 0, s3
	v_add_nc_u32_e32 v56, 16, v55
	v_add_nc_u32_e32 v57, 32, v55
	v_add_nc_u32_e32 v58, 48, v55
	v_add_nc_u32_e32 v59, 64, v55
	v_add_nc_u32_e32 v60, 0x50, v55
	v_add_nc_u32_e32 v61, 0x60, v55
	v_add_nc_u32_e32 v62, 0x70, v55
	v_add_nc_u32_e32 v63, 0x80, v55
	v_add_nc_u32_e32 v64, 0x90, v55
	v_add_nc_u32_e32 v65, 0xa0, v55
	v_add_nc_u32_e32 v66, 0xb0, v55
	v_add_nc_u32_e32 v67, 0xc0, v55
	v_add_nc_u32_e32 v68, 0xd0, v55
	v_add_nc_u32_e32 v69, 0xe0, v55
	v_add_nc_u32_e32 v70, 0xf0, v55
	v_add_nc_u32_e32 v71, 0, v71
	v_add_nc_u32_e32 v72, 0, v72
	v_add_nc_u32_e32 v73, 0, v73
	v_add_nc_u32_e32 v74, 0, v74
	v_add_nc_u32_e32 v75, 0, v75
	v_add_nc_u32_e32 v76, 0, v76
	v_add_nc_u32_e32 v77, 0, v77
	v_add_nc_u32_e32 v78, 0, v78
	v_add_nc_u32_e32 v79, v54, v8
	v_add_nc_u32_e32 v80, v54, v9
	v_add_nc_u32_e32 v81, v54, v10
	v_add_nc_u32_e32 v82, v54, v11
	v_add_nc_u32_e32 v83, v54, v1
	v_add_nc_u32_e32 v84, v54, v3
	v_add_nc_u32_e32 v85, v54, v5
	v_add_nc_u32_e32 v86, v54, v2
	v_add_nc_u32_e32 v87, v54, v87
	v_add_nc_u32_e32 v88, v54, v88
	v_add_nc_u32_e32 v89, v54, v89
	v_add_nc_u32_e32 v90, v54, v6
	v_add_nc_u32_e32 v91, v54, v91
	v_add_nc_u32_e32 v92, v54, v92
	v_add_nc_u32_e32 v93, v54, v12
	v_lshlrev_b32_e32 v94, 4, v7
	v_lshlrev_b32_e32 v95, 2, v0
	s_lshl_b32 s12, s12, 1
	s_add_co_i32 s1, s2, s1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[10:11], s[10:11], 9
	s_add_nc_u64 s[4:5], s[4:5], s[12:13]
	s_add_nc_u64 s[6:7], s[6:7], s[12:13]
	s_branch .LBB3_12
.LBB3_9:                                ; %_ZL11fa2_scale_nPKhiii.exit166.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	global_store_b32 v[1:2], v3, off offset:33024
.LBB3_10:                               ; %Flow
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s2, s2, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s2, s1
	s_cselect_b32 s3, -1, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB3_11:                               ; %Flow1262
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_322
.LBB3_12:                               ; =>This Inner Loop Header: Depth=1
	s_lshl_b32 s15, s2, 6
	s_mov_b32 s3, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s15, s14
	s_cbranch_scc1 .LBB3_11
; %bb.13:                               ; %.preheader178.preheader.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v3, s15, v44
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_15
; %bb.14:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[3:4], null, 0x408, v3, v[14:15]
	global_load_b128 v[5:8], v[3:4], off
.LBB3_15:                               ; %.preheader178.1.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v4, 0 :: v_dual_add_nc_u32 v9, s15, v46
	v_mov_b32_e32 v3, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v71, v[5:6], v[7:8] offset1:16
	v_cmpx_ge_i32_e64 s14, v9
	s_cbranch_execz .LBB3_17
; %bb.16:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v9, v[14:15]
	global_load_b128 v[1:4], v[1:2], off
.LBB3_17:                               ; %.preheader178.2.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v6, 0 :: v_dual_add_nc_u32 v7, s15, v47
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v12, 0
	v_mov_b32_e32 v11, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v72, v[1:2], v[3:4] offset1:16
	v_cmpx_ge_i32_e64 s14, v7
	s_cbranch_execz .LBB3_19
; %bb.18:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[14:15]
	global_load_b128 v[9:12], v[1:2], off
.LBB3_19:                               ; %.preheader178.3.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v1, s15, v48
	v_mov_b32_e32 v7, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v73, v[9:10], v[11:12] offset1:16
	v_cmpx_ge_i32_e64 s14, v1
	s_cbranch_execz .LBB3_21
; %bb.20:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[14:15]
	global_load_b128 v[5:8], v[1:2], off
.LBB3_21:                               ; %.preheader178.4.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v3, s15, v49
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v12, 0
	v_mov_b32_e32 v11, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v74, v[5:6], v[7:8] offset1:16
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_23
; %bb.22:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[3:4], null, 0x408, v3, v[14:15]
	global_load_b128 v[9:12], v[3:4], off
.LBB3_23:                               ; %.preheader178.5.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v4, 0 :: v_dual_add_nc_u32 v5, s15, v50
	v_mov_b32_e32 v3, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v75, v[9:10], v[11:12] offset1:16
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_25
; %bb.24:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v5, v[14:15]
	global_load_b128 v[1:4], v[1:2], off
.LBB3_25:                               ; %.preheader178.6.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v6, 0 :: v_dual_add_nc_u32 v7, s15, v51
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v12, 0
	v_mov_b32_e32 v11, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v76, v[1:2], v[3:4] offset1:16
	v_cmpx_ge_i32_e64 s14, v7
	s_cbranch_execz .LBB3_27
; %bb.26:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[14:15]
	global_load_b128 v[9:12], v[1:2], off
.LBB3_27:                               ; %.preheader178.7.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v1, s15, v52
	v_mov_b32_e32 v7, 0
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v77, v[9:10], v[11:12] offset1:16
	v_cmpx_ge_i32_e64 s14, v1
	s_cbranch_execz .LBB3_29
; %bb.28:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[14:15]
	global_load_b128 v[5:8], v[1:2], off
.LBB3_29:                               ; %.preheader177.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v3, s15, v45
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v78, v[5:6], v[7:8] offset1:16
                                        ; implicit-def: $vgpr1_vgpr2
	v_or_b32_e32 v4, 7, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_31
; %bb.30:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[16:17]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB3_31:                               ; %Flow1261
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_47
; %bb.32:                               ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_46
; %bb.33:                               ; %.preheader.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[16:17]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v3
	s_cbranch_execz .LBB3_35
; %bb.34:                               ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[16:17]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_35:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_37
; %bb.36:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB3_37:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_39
; %bb.38:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB3_39:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_41
; %bb.40:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB3_41:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_43
; %bb.42:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB3_43:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_45
; %bb.44:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[16:17]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB3_45:                               ; %Flow1259
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_46:                               ; %Flow1260
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_47:                               ; %.loopexit.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v5, v54, v53
	s_mov_b32 s3, exec_lo
	ds_store_b64 v5, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_49
; %bb.48:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[18:19]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB3_49:                               ; %Flow1258
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_65
; %bb.50:                               ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_64
; %bb.51:                               ; %.preheader.1.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[18:19]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v3
	s_cbranch_execz .LBB3_53
; %bb.52:                               ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[18:19]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_53:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_55
; %bb.54:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[18:19]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB3_55:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_57
; %bb.56:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[18:19]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB3_57:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_59
; %bb.58:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[18:19]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB3_59:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_61
; %bb.60:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[18:19]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB3_61:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_63
; %bb.62:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[18:19]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB3_63:                               ; %Flow1256
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_64:                               ; %Flow1257
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_65:                               ; %.loopexit.1.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	ds_store_b64 v79, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_67
; %bb.66:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[20:21]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB3_67:                               ; %Flow1255
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_83
; %bb.68:                               ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_82
; %bb.69:                               ; %.preheader.2.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[20:21]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v3
	s_cbranch_execz .LBB3_71
; %bb.70:                               ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[20:21]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_71:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_73
; %bb.72:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[20:21]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB3_73:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_75
; %bb.74:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[20:21]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB3_75:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_77
; %bb.76:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[20:21]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB3_77:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_79
; %bb.78:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[20:21]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB3_79:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_81
; %bb.80:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[20:21]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB3_81:                               ; %Flow1253
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_82:                               ; %Flow1254
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_83:                               ; %.loopexit.2.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	ds_store_b64 v80, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_85
; %bb.84:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[22:23]
	v_mov_b16_e32 v6.h, 0
	s_clause 0x4
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v5, v[1:2], off offset:2064
	global_load_d16_u8 v6, v[1:2], off
	global_load_u8 v7, v[1:2], off offset:3096
	global_load_u8 v8, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v9.l, v6.h
	s_clause 0x2
	global_load_u8 v10, v[1:2], off offset:4128
	global_load_u8 v11, v[1:2], off offset:7224
	global_load_d16_hi_u8 v9, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v4
	v_lshlrev_b32_e32 v2, 16, v5
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v7
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v5, 8, v8
	v_or_b32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v10, v5
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v11
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v9, v4
	v_or3_b32 v1, v1, 0, 0
.LBB3_85:                               ; %Flow1252
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_101
; %bb.86:                               ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_100
; %bb.87:                               ; %.preheader.3.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[22:23]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v3
	s_cbranch_execz .LBB3_89
; %bb.88:                               ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[22:23]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_89:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 2, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_91
; %bb.90:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[22:23]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v4, 16, v1
.LBB3_91:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 3, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_93
; %bb.92:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[22:23]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v4, 24, v1
.LBB3_93:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 4, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_95
; %bb.94:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[22:23]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v4, v2
.LBB3_95:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 5, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_97
; %bb.96:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[22:23]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v4, 8, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v4, v2
.LBB3_97:                               ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 6, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_99
; %bb.98:                               ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[22:23]
	v_mov_b16_e32 v6.l, 0
	global_load_d16_hi_u8 v6, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB3_99:                               ; %Flow1250
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_100:                              ; %Flow1251
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_101:                              ; %.loopexit.3.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v4, 16, v3
	s_mov_b32 s3, exec_lo
	ds_store_b64 v81, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_or_b32_e32 v5, 7, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_103
; %bb.102:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[24:25]
	v_mov_b16_e32 v8.h, 0
	s_clause 0x4
	global_load_u8 v6, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v8, v[1:2], off
	global_load_u8 v9, v[1:2], off offset:3096
	global_load_u8 v10, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v11.l, v8.h
	s_clause 0x2
	global_load_u8 v12, v[1:2], off offset:4128
	global_load_u8 v96, v[1:2], off offset:7224
	global_load_d16_hi_u8 v11, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v6, 24, v9
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v10
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v6
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v12, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 24, v96
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v11, v6
	v_or3_b32 v1, v1, 0, 0
.LBB3_103:                              ; %Flow1249
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_119
; %bb.104:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_118
; %bb.105:                              ; %.preheader.4.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[24:25]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v4
	s_cbranch_execz .LBB3_107
; %bb.106:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[24:25]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_107:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 2, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_109
; %bb.108:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[24:25]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 16, v1
.LBB3_109:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 3, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_111
; %bb.110:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[24:25]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 24, v1
.LBB3_111:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 4, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_113
; %bb.112:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[24:25]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB3_113:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 5, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_115
; %bb.114:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[24:25]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v6, v2
.LBB3_115:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 6, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_117
; %bb.116:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[24:25]
	v_mov_b16_e32 v8.l, 0
	global_load_d16_hi_u8 v8, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB3_117:                              ; %Flow1247
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_118:                              ; %Flow1248
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_119:                              ; %.loopexit.4.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	ds_store_b64 v82, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_121
; %bb.120:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[26:27]
	v_mov_b16_e32 v8.h, 0
	s_clause 0x4
	global_load_u8 v6, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v8, v[1:2], off
	global_load_u8 v9, v[1:2], off offset:3096
	global_load_u8 v10, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v11.l, v8.h
	s_clause 0x2
	global_load_u8 v12, v[1:2], off offset:4128
	global_load_u8 v96, v[1:2], off offset:7224
	global_load_d16_hi_u8 v11, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v6, 24, v9
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v10
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v6
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v12, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 24, v96
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v11, v6
	v_or3_b32 v1, v1, 0, 0
.LBB3_121:                              ; %Flow1246
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_137
; %bb.122:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_136
; %bb.123:                              ; %.preheader.5.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[26:27]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v4
	s_cbranch_execz .LBB3_125
; %bb.124:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[26:27]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_125:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 2, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_127
; %bb.126:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[26:27]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 16, v1
.LBB3_127:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 3, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_129
; %bb.128:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[26:27]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 24, v1
.LBB3_129:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 4, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_131
; %bb.130:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[26:27]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB3_131:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 5, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_133
; %bb.132:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[26:27]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v6, v2
.LBB3_133:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 6, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_135
; %bb.134:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[26:27]
	v_mov_b16_e32 v8.l, 0
	global_load_d16_hi_u8 v8, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB3_135:                              ; %Flow1244
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_136:                              ; %Flow1245
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_137:                              ; %.loopexit.5.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	ds_store_b64 v83, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_139
; %bb.138:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[28:29]
	v_mov_b16_e32 v8.h, 0
	s_clause 0x4
	global_load_u8 v6, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v8, v[1:2], off
	global_load_u8 v9, v[1:2], off offset:3096
	global_load_u8 v10, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v11.l, v8.h
	s_clause 0x2
	global_load_u8 v12, v[1:2], off offset:4128
	global_load_u8 v96, v[1:2], off offset:7224
	global_load_d16_hi_u8 v11, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v6, 24, v9
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v10
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v6
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v12, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 24, v96
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v11, v6
	v_or3_b32 v1, v1, 0, 0
.LBB3_139:                              ; %Flow1243
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_155
; %bb.140:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_154
; %bb.141:                              ; %.preheader.6.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[28:29]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v4
	s_cbranch_execz .LBB3_143
; %bb.142:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[28:29]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_143:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 2, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_145
; %bb.144:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[28:29]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 16, v1
.LBB3_145:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 3, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_147
; %bb.146:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[28:29]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 24, v1
.LBB3_147:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 4, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_149
; %bb.148:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[28:29]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB3_149:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 5, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_151
; %bb.150:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[28:29]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v6, v2
.LBB3_151:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 6, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_153
; %bb.152:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[28:29]
	v_mov_b16_e32 v8.l, 0
	global_load_d16_hi_u8 v8, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB3_153:                              ; %Flow1241
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_154:                              ; %Flow1242
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_155:                              ; %.loopexit.6.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	ds_store_b64 v84, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_157
; %bb.156:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[30:31]
	v_mov_b16_e32 v6.h, 0
	s_clause 0x4
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v5, v[1:2], off offset:2064
	global_load_d16_u8 v6, v[1:2], off
	global_load_u8 v7, v[1:2], off offset:3096
	global_load_u8 v8, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v9.l, v6.h
	s_clause 0x2
	global_load_u8 v10, v[1:2], off offset:4128
	global_load_u8 v11, v[1:2], off offset:7224
	global_load_d16_hi_u8 v9, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v4
	v_lshlrev_b32_e32 v2, 16, v5
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v7
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v5, 8, v8
	v_or_b32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v10, v5
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v11
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v9, v4
                                        ; implicit-def: $vgpr4
	v_or3_b32 v1, v1, 0, 0
.LBB3_157:                              ; %Flow1240
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_173
; %bb.158:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_172
; %bb.159:                              ; %.preheader.7.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[30:31]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v4
	s_cbranch_execz .LBB3_161
; %bb.160:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[30:31]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_161:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 2, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_163
; %bb.162:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[30:31]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB3_163:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 3, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_165
; %bb.164:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[30:31]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB3_165:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 4, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_167
; %bb.166:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[30:31]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB3_167:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 5, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_169
; %bb.168:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[30:31]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB3_169:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 6, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_171
; %bb.170:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[30:31]
	v_mov_b16_e32 v6.l, 0
	global_load_d16_hi_u8 v6, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB3_171:                              ; %Flow1238
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_172:                              ; %Flow1239
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_173:                              ; %.loopexit.7.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v4, 32, v3
	s_mov_b32 s3, exec_lo
	ds_store_b64 v85, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_or_b32_e32 v5, 7, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_175
; %bb.174:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[16:17]
	v_mov_b16_e32 v6.h, 0
	s_clause 0x4
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v5, v[1:2], off offset:2064
	global_load_d16_u8 v6, v[1:2], off
	global_load_u8 v7, v[1:2], off offset:3096
	global_load_u8 v8, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v9.l, v6.h
	s_clause 0x2
	global_load_u8 v10, v[1:2], off offset:4128
	global_load_u8 v11, v[1:2], off offset:7224
	global_load_d16_hi_u8 v9, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v4
	v_lshlrev_b32_e32 v2, 16, v5
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v7
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v5, 8, v8
	v_or_b32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v10, v5
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v11
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v9, v4
                                        ; implicit-def: $vgpr4
	v_or3_b32 v1, v1, 0, 0
.LBB3_175:                              ; %Flow1237
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_191
; %bb.176:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_190
; %bb.177:                              ; %.preheader.8.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[16:17]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v4
	s_cbranch_execz .LBB3_179
; %bb.178:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[16:17]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_179:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 2, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_181
; %bb.180:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB3_181:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 3, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_183
; %bb.182:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB3_183:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 4, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_185
; %bb.184:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB3_185:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 5, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_187
; %bb.186:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[16:17]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB3_187:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 6, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_189
; %bb.188:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[16:17]
	v_mov_b16_e32 v6.l, 0
	global_load_d16_hi_u8 v6, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB3_189:                              ; %Flow1235
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_190:                              ; %Flow1236
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_191:                              ; %.loopexit.8.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v4, 32, v3
	s_mov_b32 s3, exec_lo
	ds_store_b64 v86, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_or_b32_e32 v5, 7, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_193
; %bb.192:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[32:33]
	v_mov_b16_e32 v8.h, 0
	s_clause 0x4
	global_load_u8 v6, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v8, v[1:2], off
	global_load_u8 v9, v[1:2], off offset:3096
	global_load_u8 v10, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v11.l, v8.h
	s_clause 0x2
	global_load_u8 v12, v[1:2], off offset:4128
	global_load_u8 v96, v[1:2], off offset:7224
	global_load_d16_hi_u8 v11, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v6, 24, v9
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v10
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v6
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v12, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 24, v96
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v11, v6
	v_or3_b32 v1, v1, 0, 0
.LBB3_193:                              ; %Flow1234
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_209
; %bb.194:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_208
; %bb.195:                              ; %.preheader.9.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[32:33]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v4
	s_cbranch_execz .LBB3_197
; %bb.196:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[32:33]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_197:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 2, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_199
; %bb.198:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[32:33]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 16, v1
.LBB3_199:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 3, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_201
; %bb.200:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[32:33]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 24, v1
.LBB3_201:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 4, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_203
; %bb.202:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[32:33]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB3_203:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 5, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_205
; %bb.204:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[32:33]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v6, v2
.LBB3_205:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 6, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_207
; %bb.206:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[32:33]
	v_mov_b16_e32 v8.l, 0
	global_load_d16_hi_u8 v8, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB3_207:                              ; %Flow1232
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_208:                              ; %Flow1233
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_209:                              ; %.loopexit.9.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	ds_store_b64 v87, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_211
; %bb.210:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[34:35]
	v_mov_b16_e32 v8.h, 0
	s_clause 0x4
	global_load_u8 v6, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v8, v[1:2], off
	global_load_u8 v9, v[1:2], off offset:3096
	global_load_u8 v10, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v11.l, v8.h
	s_clause 0x2
	global_load_u8 v12, v[1:2], off offset:4128
	global_load_u8 v96, v[1:2], off offset:7224
	global_load_d16_hi_u8 v11, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v6, 24, v9
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v10
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v6
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v12, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 24, v96
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v11, v6
	v_or3_b32 v1, v1, 0, 0
.LBB3_211:                              ; %Flow1231
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_227
; %bb.212:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_226
; %bb.213:                              ; %.preheader.10.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[34:35]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v4
	s_cbranch_execz .LBB3_215
; %bb.214:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[34:35]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_215:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 2, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_217
; %bb.216:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[34:35]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 16, v1
.LBB3_217:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 3, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_219
; %bb.218:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[34:35]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v6, 24, v1
.LBB3_219:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 4, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_221
; %bb.220:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[34:35]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB3_221:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 5, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_223
; %bb.222:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[34:35]
	global_load_u8 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v6, v2
.LBB3_223:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v6, 6, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v6
	s_cbranch_execz .LBB3_225
; %bb.224:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[6:7], null, 0x408, v6, v[34:35]
	v_mov_b16_e32 v8.l, 0
	global_load_d16_hi_u8 v8, v[6:7], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB3_225:                              ; %Flow1229
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_226:                              ; %Flow1230
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_227:                              ; %.loopexit.10.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	ds_store_b64 v88, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_229
; %bb.228:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[36:37]
	v_mov_b16_e32 v6.h, 0
	s_clause 0x4
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v5, v[1:2], off offset:2064
	global_load_d16_u8 v6, v[1:2], off
	global_load_u8 v7, v[1:2], off offset:3096
	global_load_u8 v8, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v9.l, v6.h
	s_clause 0x2
	global_load_u8 v10, v[1:2], off offset:4128
	global_load_u8 v11, v[1:2], off offset:7224
	global_load_d16_hi_u8 v9, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v4
	v_lshlrev_b32_e32 v2, 16, v5
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v7
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v5, 8, v8
	v_or_b32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v10, v5
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v11
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v9, v4
                                        ; implicit-def: $vgpr4
	v_or3_b32 v1, v1, 0, 0
.LBB3_229:                              ; %Flow1228
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_245
; %bb.230:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_244
; %bb.231:                              ; %.preheader.11.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, v[36:37]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v4
	s_cbranch_execz .LBB3_233
; %bb.232:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[36:37]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_233:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 2, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_235
; %bb.234:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[36:37]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB3_235:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 3, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_237
; %bb.236:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[36:37]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB3_237:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 4, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_239
; %bb.238:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[36:37]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB3_239:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 5, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_241
; %bb.240:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[36:37]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB3_241:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 6, v4
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_243
; %bb.242:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[36:37]
	v_mov_b16_e32 v6.l, 0
	global_load_d16_hi_u8 v6, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB3_243:                              ; %Flow1226
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_244:                              ; %Flow1227
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_245:                              ; %.loopexit.11.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v3, 48, v3
	s_mov_b32 s3, exec_lo
	ds_store_b64 v89, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_or_b32_e32 v4, 7, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_247
; %bb.246:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[24:25]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB3_247:                              ; %Flow1225
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_263
; %bb.248:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_262
; %bb.249:                              ; %.preheader.12.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[24:25]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v3
	s_cbranch_execz .LBB3_251
; %bb.250:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[24:25]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_251:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_253
; %bb.252:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[24:25]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB3_253:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_255
; %bb.254:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[24:25]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB3_255:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_257
; %bb.256:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[24:25]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB3_257:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_259
; %bb.258:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[24:25]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB3_259:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_261
; %bb.260:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[24:25]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB3_261:                              ; %Flow1223
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_262:                              ; %Flow1224
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_263:                              ; %.loopexit.12.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	ds_store_b64 v90, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_265
; %bb.264:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[38:39]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB3_265:                              ; %Flow1222
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_281
; %bb.266:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_280
; %bb.267:                              ; %.preheader.13.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[38:39]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v3
	s_cbranch_execz .LBB3_269
; %bb.268:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[38:39]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_269:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_271
; %bb.270:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[38:39]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB3_271:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_273
; %bb.272:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[38:39]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB3_273:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_275
; %bb.274:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[38:39]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB3_275:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_277
; %bb.276:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[38:39]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB3_277:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_279
; %bb.278:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[38:39]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB3_279:                              ; %Flow1220
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_280:                              ; %Flow1221
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_281:                              ; %.loopexit.13.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	ds_store_b64 v91, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_283
; %bb.282:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[40:41]
	v_mov_b16_e32 v7.h, 0
	s_clause 0x4
	global_load_u8 v5, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v10.l, v7.h
	s_clause 0x2
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v5
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v5, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v5
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v5
	v_or3_b32 v1, v1, 0, 0
.LBB3_283:                              ; %Flow1219
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_299
; %bb.284:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_298
; %bb.285:                              ; %.preheader.14.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[40:41]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v3
	s_cbranch_execz .LBB3_287
; %bb.286:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[40:41]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_287:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 2, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_289
; %bb.288:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[40:41]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB3_289:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 3, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_291
; %bb.290:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[40:41]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB3_291:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 4, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_293
; %bb.292:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[40:41]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB3_293:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 5, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_295
; %bb.294:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[40:41]
	global_load_u8 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB3_295:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v5, 6, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v5
	s_cbranch_execz .LBB3_297
; %bb.296:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[5:6], null, 0x408, v5, v[40:41]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[5:6], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB3_297:                              ; %Flow1217
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_298:                              ; %Flow1218
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_299:                              ; %.loopexit.14.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	ds_store_b64 v92, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_ge_i32_e64 s14, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_301
; %bb.300:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[42:43]
	v_mov_b16_e32 v5.h, 0
	s_clause 0x4
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v5, v[1:2], off
	global_load_u8 v6, v[1:2], off offset:3096
	global_load_u8 v7, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v8.l, v5.h
	s_clause 0x2
	global_load_u8 v9, v[1:2], off offset:4128
	global_load_u8 v10, v[1:2], off offset:7224
	global_load_d16_hi_u8 v8, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v3
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v6
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v7
	v_or_b32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v9, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v10
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v8, v3
                                        ; implicit-def: $vgpr3
	v_or3_b32 v1, v1, 0, 0
.LBB3_301:                              ; %Flow1216
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB3_317
; %bb.302:                              ;   in Loop: Header=BB3_12 Depth=1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_316
; %bb.303:                              ; %.preheader.15.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[42:43]
	v_mov_b16_e32 v12.h, 0
	s_mov_b32 s13, exec_lo
	global_load_d16_u8 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
	v_cmpx_gt_i32_e64 s14, v3
	s_cbranch_execz .LBB3_305
; %bb.304:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[42:43]
	global_load_u8 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v12, v1, 8, v12
	v_dual_mov_b32 v1, v12 :: v_dual_mov_b32 v2, v13
.LBB3_305:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 2, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_307
; %bb.306:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[42:43]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v4, 16, v1
.LBB3_307:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 3, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_309
; %bb.308:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[42:43]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v4, 24, v1
.LBB3_309:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 4, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_311
; %bb.310:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[42:43]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v4, v2
.LBB3_311:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v4, 5, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v4
	s_cbranch_execz .LBB3_313
; %bb.312:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[42:43]
	global_load_u8 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v4, 8, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v4, v2
.LBB3_313:                              ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v3, 6, v3
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s14, v3
	s_cbranch_execz .LBB3_315
; %bb.314:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[3:4], null, 0x408, v3, v[42:43]
	v_mov_b16_e32 v5.l, 0
	global_load_d16_hi_u8 v5, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB3_315:                              ; %Flow1214
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB3_316:                              ; %Flow1215
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
.LBB3_317:                              ; %.loopexit.15.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	ds_store_b64 v93, v[1:2] offset:16384
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_ashr_i32 s3, s2, 31
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[10:11], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[12:13], s[12:13], 0x8200
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[8:9], s[12:13]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_b32 v[3:4], v55 offset0:2 offset1:3
	ds_load_2addr_b32 v[7:8], v56 offset0:2 offset1:3
	ds_load_2addr_b32 v[5:6], v56 offset1:1
	ds_load_2addr_b32 v[1:2], v55 offset1:1
	ds_load_2addr_b32 v[11:12], v57 offset0:2 offset1:3
	ds_load_2addr_b32 v[98:99], v58 offset0:2 offset1:3
	ds_load_2addr_b32 v[96:97], v58 offset1:1
	ds_load_2addr_b32 v[9:10], v57 offset1:1
	ds_load_2addr_b32 v[102:103], v59 offset0:2 offset1:3
	ds_load_2addr_b32 v[106:107], v60 offset0:2 offset1:3
	ds_load_2addr_b32 v[104:105], v60 offset1:1
	ds_load_2addr_b32 v[100:101], v59 offset1:1
	ds_load_2addr_b32 v[110:111], v61 offset0:2 offset1:3
	ds_load_2addr_b32 v[114:115], v62 offset0:2 offset1:3
	ds_load_2addr_b32 v[112:113], v62 offset1:1
	ds_load_2addr_b32 v[108:109], v61 offset1:1
	ds_load_2addr_b32 v[118:119], v63 offset0:2 offset1:3
	ds_load_2addr_b32 v[122:123], v64 offset0:2 offset1:3
	ds_load_2addr_b32 v[120:121], v64 offset1:1
	ds_load_2addr_b32 v[116:117], v63 offset1:1
	ds_load_2addr_b32 v[126:127], v65 offset0:2 offset1:3
	ds_load_2addr_b32 v[130:131], v66 offset0:2 offset1:3
	ds_load_2addr_b32 v[128:129], v66 offset1:1
	ds_load_2addr_b32 v[124:125], v65 offset1:1
	ds_load_2addr_b32 v[134:135], v67 offset0:2 offset1:3
	ds_load_2addr_b32 v[138:139], v68 offset0:2 offset1:3
	ds_load_2addr_b32 v[136:137], v68 offset1:1
	ds_load_2addr_b32 v[132:133], v67 offset1:1
	ds_load_2addr_b32 v[142:143], v69 offset0:2 offset1:3
	ds_load_2addr_b32 v[144:145], v70 offset1:1
	ds_load_2addr_b32 v[140:141], v69 offset1:1
	ds_load_2addr_b32 v[146:147], v70 offset0:2 offset1:3
	s_wait_dscnt 0x1c
	s_clause 0x1
	global_store_b128 v94, v[1:4], s[12:13]
	global_store_b128 v94, v[5:8], s[12:13] offset:16
	s_wait_dscnt 0x18
	s_clause 0x1
	global_store_b128 v94, v[9:12], s[12:13] offset:32
	global_store_b128 v94, v[96:99], s[12:13] offset:48
	s_wait_dscnt 0x14
	s_clause 0x1
	global_store_b128 v94, v[100:103], s[12:13] offset:64
	global_store_b128 v94, v[104:107], s[12:13] offset:80
	s_wait_dscnt 0x10
	s_clause 0x1
	global_store_b128 v94, v[108:111], s[12:13] offset:96
	global_store_b128 v94, v[112:115], s[12:13] offset:112
	s_wait_dscnt 0xc
	s_clause 0x1
	global_store_b128 v94, v[116:119], s[12:13] offset:128
	global_store_b128 v94, v[120:123], s[12:13] offset:144
	s_wait_dscnt 0x8
	s_clause 0x1
	global_store_b128 v94, v[124:127], s[12:13] offset:160
	global_store_b128 v94, v[128:131], s[12:13] offset:176
	s_wait_dscnt 0x4
	s_clause 0x1
	global_store_b128 v94, v[132:135], s[12:13] offset:192
	global_store_b128 v94, v[136:139], s[12:13] offset:208
	s_wait_dscnt 0x1
	global_store_b128 v94, v[140:143], s[12:13] offset:224
	s_wait_dscnt 0x0
	global_store_b128 v94, v[144:147], s[12:13] offset:240
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB3_10
; %bb.318:                              ;   in Loop: Header=BB3_12 Depth=1
	v_or_b32_e32 v4, s15, v0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v5, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e32 vcc_lo, s14, v4
	s_and_saveexec_b32 s15, vcc_lo
	s_cbranch_execz .LBB3_320
; %bb.319:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[1:2], null, 0x408, v4, s[4:5]
	global_load_d16_b16 v1, v[1:2], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v5, v1.l
.LBB3_320:                              ; %_ZL11fa2_scale_nPKhiii.exit.i
                                        ;   in Loop: Header=BB3_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_add_co_u32 v1, s15, s12, v95
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s13, 0, s15
	global_store_b32 v95, v5, s[12:13] offset:32768
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB3_9
; %bb.321:                              ;   in Loop: Header=BB3_12 Depth=1
	v_mad_co_i64_i32 v[3:4], null, 0x408, v4, s[6:7]
	global_load_d16_b16 v3, v[3:4], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v3, v3.l
	s_branch .LBB3_9
.LBB3_322:                              ; %_Z21fa2_stageb_nfill_dumpILb1EEvPKhS1_PfPKiiiiiiii.exit
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
		.amdhsa_next_free_vgpr 148
		.amdhsa_next_free_sgpr 22
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
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.num_vgpr, 148
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.numbered_sgpr, 22
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15592
; TotalNumSgprs: 24
; NumVgprs: 148
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 18
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 148
; Occupancy: 9
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
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     148
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
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_partial_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     148
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
