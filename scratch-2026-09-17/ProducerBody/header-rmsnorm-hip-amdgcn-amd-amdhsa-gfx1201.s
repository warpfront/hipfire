	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	fused_rmsnorm_mq_rotate_awq_i4_gfx12 ; -- Begin function fused_rmsnorm_mq_rotate_awq_i4_gfx12
	.globl	fused_rmsnorm_mq_rotate_awq_i4_gfx12
	.p2align	8
	.type	fused_rmsnorm_mq_rotate_awq_i4_gfx12,@function
fused_rmsnorm_mq_rotate_awq_i4_gfx12:   ; @fused_rmsnorm_mq_rotate_awq_i4_gfx12
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x3
	s_load_b96 s[20:22], s[0:1], 0x38
	s_load_b32 s2, s[0:1], 0x54
	s_load_b256 s[12:19], s[0:1], 0x0
	s_load_b256 s[4:11], s[0:1], 0x20
	v_mov_b32_e32 v42, 0
	s_mov_b32 s24, ttmp9
	s_mov_b32 s25, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s11, exec_lo
	s_ashr_i32 s1, s20, 31
	s_mov_b32 s0, s20
	s_and_b32 s10, s2, 0xffff
	s_mul_u64 s[2:3], s[0:1], s[24:25]
	v_cmpx_gt_i32_e64 s20, v0
	s_cbranch_execz .LBB0_4
; %bb.1:                                ; %.lr.ph.preheader
	v_dual_mov_b32 v42, 0 :: v_dual_lshlrev_b32 v1, 2, v0
	s_lshl_b64 s[26:27], s[2:3], 2
	v_mov_b32_e32 v3, v0
	s_add_nc_u64 s[26:27], s[12:13], s[26:27]
	s_lshl_b32 s23, s10, 2
	v_add_co_u32 v1, s0, s26, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s27, 0, s0
.LBB0_2:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	global_load_b32 v4, v[1:2], off
	v_add_nc_u32_e32 v3, s10, v3
	v_add_co_u32 v1, vcc_lo, v1, s23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v42, v4, v4
	v_cmp_le_i32_e64 s0, s20, v3
	s_or_b32 s25, s0, s25
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s25
	s_cbranch_execnz .LBB0_2
; %bb.3:                                ; %Flow794
	s_or_b32 exec_lo, exec_lo, s25
.LBB0_4:                                ; %Flow795
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s11
	s_lshr_b32 s24, s10, 5
	s_lshr_b32 s1, s1, 24
	s_cvt_f32_u32 s0, s24
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s1, s20, s1
	s_lshl_b64 s[2:3], s[2:3], 2
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s11, s1, 8
	v_s_rcp_f32 s0, s0
	s_sub_co_i32 s1, 0, s24
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s10, s11, s24
	v_lshrrev_b32_e32 v43, 5, v0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s10, s10, -1
	v_and_b32_e32 v41, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s23, s10
	s_ashr_i32 s10, s10, 31
	s_mul_f32 s0, s0, 0x4f7ffffe
	s_add_nc_u64 s[12:13], s[12:13], s[2:3]
	v_lshlrev_b32_e32 v68, 3, v41
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr21
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s0, s0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s1, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s0, s23, s0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s1, s0, s24
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s1, s23, s1
	s_add_co_i32 s23, s0, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s25, s1, s24
	s_cmp_ge_u32 s1, s24
	s_cselect_b32 s0, s23, s0
	s_cselect_b32 s1, s25, s1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s23, s0, 1
	s_cmp_ge_u32 s1, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s0, s23, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, s0, s10
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s23, s0, s10
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v67, s23, v43
	s_cmp_gt_i32 s23, 0
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v67
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s10, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s10
	s_cbranch_execz .LBB0_6
; %bb.5:
	v_lshl_or_b32 v1, v67, 8, v68
	v_mov_b32_e32 v2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v3, vcc_lo, s12, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s13, v2, vcc_lo
	v_add_co_u32 v5, vcc_lo, s14, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s15, v2, vcc_lo
	v_add_co_u32 v9, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v2, vcc_lo
	s_clause 0x1
	global_load_b128 v[21:24], v[3:4], off
	global_load_b128 v[17:20], v[3:4], off offset:16
	s_clause 0x1
	global_load_b128 v[13:16], v[5:6], off
	global_load_b128 v[5:8], v[5:6], off offset:16
	s_clause 0x1
	global_load_b128 v[1:4], v[9:10], off
	global_load_b128 v[9:12], v[9:10], off offset:16
.LBB0_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_swizzle_b32 v45, v42 offset:swizzle(BITMASK_PERM,"1pppp")
	v_lshlrev_b32_e32 v37, 2, v68
	v_mbcnt_lo_u32_b32 v44, -1, 0
	v_cmp_eq_u32_e64 s0, 0, v41
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_u32_e32 vcc_lo, 24, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v46, 0, 8, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 28, v44
	s_wait_dscnt 0x0
	v_add_f32_e32 v42, v42, v45
	s_clause 0x3
	global_load_b128 v[25:28], v37, s[18:19] offset:16
	global_load_b128 v[29:32], v37, s[18:19]
	global_load_b128 v[33:36], v37, s[4:5] offset:16
	global_load_b128 v[37:40], v37, s[4:5]
	v_add_lshl_u32 v45, v46, v44, 2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v46, 0, 4, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 30, v44
	ds_bpermute_b32 v45, v45, v42
	v_add_nc_u32_e32 v46, v46, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v47, 0, 2, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 31, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v48, null, 0, v44, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v45, v42, v45 :: v_dual_lshlrev_b32 v42, 2, v46
	ds_bpermute_b32 v46, v42, v45
	s_wait_dscnt 0x0
	v_dual_add_f32 v46, v45, v46 :: v_dual_add_nc_u32 v47, v47, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v45, 2, v47
	ds_bpermute_b32 v47, v45, v46
	s_wait_dscnt 0x0
	v_dual_add_f32 v47, v46, v47 :: v_dual_lshlrev_b32 v46, 2, v48
	ds_bpermute_b32 v48, v46, v47
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_lshl_add_u32 v43, v43, 2, 0
	s_wait_dscnt 0x0
	v_add_f32_e32 v47, v47, v48
	ds_store_b32 v43, v47
.LBB0_8:
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB0_13
; %bb.9:
	v_mov_b32_e32 v43, 0
	s_mov_b32 s5, exec_lo
	v_cmpx_gt_u32_e64 s24, v0
; %bb.10:
	v_lshl_add_u32 v43, v0, 2, 0
	ds_load_b32 v43, v43
; %bb.11:
	s_or_b32 exec_lo, exec_lo, s5
	s_wait_dscnt 0x0
	ds_bpermute_b32 v42, v42, v43
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_wait_dscnt 0x0
	v_add_f32_e32 v42, v43, v42
	ds_bpermute_b32 v43, v45, v42
	s_wait_dscnt 0x0
	v_add_f32_e32 v42, v42, v43
	ds_bpermute_b32 v43, v46, v42
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_13
; %bb.12:
	s_wait_dscnt 0x0
	v_add_f32_e32 v42, v42, v43
	s_cvt_f32_i32 s5, s20
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_3)
	v_div_scale_f32 v43, null, s5, s5, v42
	v_div_scale_f32 v47, vcc_lo, v42, s5, v42
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v45, v43
	v_fma_f32 v46, -v43, v45, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v45, v46, v45
	v_mul_f32_e32 v46, v47, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v48, -v43, v46, v47
	v_fmac_f32_e32 v46, v48, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v43, -v43, v46, v47
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v43, v43, v45, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v42, v43, s5, v42
	v_add_f32_e32 v42, s21, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v43, 0x4b800000, v42
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v42
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v42, v42, v43, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rsq_f32_e32 v42, v42
	v_mul_f32_e32 v43, 0x45800000, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_cndmask_b32 v42, v42, v43 :: v_dual_mov_b32 v43, 0
	ds_store_b32 v43, v42
.LBB0_13:                               ; %Flow793
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_94
; %bb.14:                               ; %.lr.ph625
	v_dual_mov_b32 v66, 0 :: v_dual_and_b32 v43, 2, v0
	s_add_nc_u64 s[18:19], s[6:7], s[2:3]
	v_xor_b32_e32 v46, 2, v44
	v_xor_b32_e32 v47, 1, v44
	ds_load_b32 v69, v66
	v_cmp_eq_u32_e64 s2, 0, v43
	v_xor_b32_e32 v43, 8, v44
	v_and_b32_e32 v42, 1, v0
	s_cmp_lg_u64 s[6:7], 0
	s_mov_b32 s20, ttmp9
	s_cselect_b32 s25, -1, 0
	s_ashr_i32 s21, ttmp9, 31
	v_cmp_eq_u32_e64 s1, 0, v42
	v_and_b32_e32 v42, 8, v0
	v_lshlrev_b32_e32 v70, 1, v41
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[6:7], s[20:21], 0x48
	s_mov_b32 s24, 0
	s_add_nc_u64 s[20:21], s[8:9], s[6:7]
	v_cmp_eq_u32_e64 s4, 0, v42
	v_xor_b32_e32 v42, 16, v44
	s_xor_b32 s26, s10, -1
	s_mov_b32 s27, 0xc1000000
	s_mov_b32 s28, 0
                                        ; implicit-def: $sgpr29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 32, v42
	v_and_b32_e32 v45, 4, v0
	v_and_b32_e32 v0, 16, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v42, v44, v42, vcc_lo
	v_cmp_eq_u32_e64 s3, 0, v45
	v_xor_b32_e32 v45, 4, v44
	v_cmp_gt_u32_e32 vcc_lo, 32, v43
	v_cmp_eq_u32_e64 s5, 0, v0
	v_lshrrev_b32_e32 v0, 1, v41
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v43, v44, v43 :: v_dual_lshlrev_b32 v72, 2, v42
	v_cmp_gt_u32_e32 vcc_lo, 32, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v48, 16, v0
	v_lshlrev_b32_e32 v73, 2, v43
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v45, v44, v45, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v46
	v_lshlrev_b32_e32 v71, 2, v48
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v74, 2, v45
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, v44, v46, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v47
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v44, v44, v47 :: v_dual_lshlrev_b32 v75, 2, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v76, 2, v44
	s_wait_dscnt 0x0
	s_branch .LBB0_17
.LBB0_15:                               ; %_Z31emit_iu4_sidecar_from_producer8ffffffffP12block_i4_128iiii.exit
                                        ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_add_co_i32 s28, s28, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s28, s23
	s_cselect_b32 s6, -1, 0
	s_and_not1_b32 s7, s29, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s6, s6, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s29, s7, s6
.LBB0_16:                               ; %Flow789
                                        ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s6, exec_lo, s29
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s24, s6, s24
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s24
	s_cbranch_execz .LBB0_94
.LBB0_17:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_25 Depth 2
                                        ;     Child Loop BB0_61 Depth 2
	v_add_nc_u32_e32 v77, s28, v67
	s_or_b32 s29, s29, exec_lo
	s_mov_b32 s30, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s11, v77
	s_cbranch_execz .LBB0_16
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s28, 0
	v_lshl_or_b32 v65, v77, 8, v68
	v_dual_mov_b32 v41, v9 :: v_dual_mov_b32 v42, v10
	v_dual_mov_b32 v43, v11 :: v_dual_mov_b32 v44, v12
	s_wait_dscnt 0x0
	v_dual_mov_b32 v45, v1 :: v_dual_mov_b32 v46, v2
	v_dual_mov_b32 v47, v3 :: v_dual_mov_b32 v48, v4
	v_dual_mov_b32 v49, v5 :: v_dual_mov_b32 v50, v6
	v_dual_mov_b32 v51, v7 :: v_dual_mov_b32 v52, v8
	v_dual_mov_b32 v57, v13 :: v_dual_mov_b32 v58, v14
	v_dual_mov_b32 v59, v15 :: v_dual_mov_b32 v60, v16
	v_dual_mov_b32 v53, v17 :: v_dual_mov_b32 v54, v18
	v_dual_mov_b32 v55, v19 :: v_dual_mov_b32 v56, v20
	v_dual_mov_b32 v61, v21 :: v_dual_mov_b32 v62, v22
	v_dual_mov_b32 v63, v23 :: v_dual_mov_b32 v64, v24
	s_cselect_b32 s6, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s7, s26, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s7
	s_cbranch_execz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_17 Depth=1
	v_lshlrev_b64_e32 v[41:42], 2, v[65:66]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, vcc_lo, s12, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s13, v42, vcc_lo
	v_add_co_u32 v45, vcc_lo, s14, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s15, v42, vcc_lo
	v_add_co_u32 v41, vcc_lo, s16, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s17, v42, vcc_lo
	s_clause 0x1
	global_load_b128 v[61:64], v[43:44], off
	global_load_b128 v[53:56], v[43:44], off offset:16
	s_clause 0x1
	global_load_b128 v[57:60], v[45:46], off
	global_load_b128 v[49:52], v[45:46], off offset:16
	s_clause 0x1
	global_load_b128 v[45:48], v[41:42], off
	global_load_b128 v[41:44], v[41:42], off offset:16
.LBB0_20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v57, v57, v61 :: v_dual_mul_f32 v58, v58, v62
	v_dual_mul_f32 v59, v59, v63 :: v_dual_mul_f32 v60, v60, v64
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v49, v49, v53 :: v_dual_mul_f32 v50, v50, v54
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v57, v69, v57 :: v_dual_mul_f32 v58, v69, v58
	v_dual_mul_f32 v53, v69, v59 :: v_dual_mul_f32 v52, v52, v56
	v_dual_mul_f32 v51, v51, v55 :: v_dual_mul_f32 v60, v69, v60
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v54, null, v45, v45, v57
	v_div_scale_f32 v59, null, v46, v46, v58
	v_div_scale_f32 v55, null, v47, v47, v53
	v_div_scale_f32 v63, vcc_lo, v57, v45, v57
	v_div_scale_f32 v79, s6, v58, v46, v58
	v_rcp_f32_e32 v56, v54
	v_rcp_f32_e32 v61, v59
	v_rcp_f32_e32 v62, v55
	v_div_scale_f32 v80, s7, v53, v47, v53
	v_dual_mul_f32 v49, v69, v49 :: v_dual_mul_f32 v50, v69, v50
	v_dual_mul_f32 v51, v69, v51 :: v_dual_mul_f32 v52, v69, v52
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v64, -v54, v56, 1.0
	v_fma_f32 v78, -v59, v61, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v81, -v55, v62, 1.0
	s_wait_loadcnt 0x0
	v_div_scale_f32 v83, null, v41, v41, v49
	v_div_scale_f32 v88, null, v42, v42, v50
	v_dual_fmac_f32 v56, v64, v56 :: v_dual_fmac_f32 v61, v78, v61
	v_fmac_f32_e32 v62, v81, v62
	v_div_scale_f32 v64, null, v48, v48, v60
	v_div_scale_f32 v89, null, v43, v43, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v81, v63, v56
	v_dual_mul_f32 v82, v79, v61 :: v_dual_mul_f32 v85, v80, v62
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v84, v64
	v_div_scale_f32 v78, s8, v60, v48, v60
	v_fma_f32 v86, -v54, v81, v63
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v87, -v59, v82, v79
	v_fma_f32 v90, -v55, v85, v80
	v_dual_fmac_f32 v81, v86, v56 :: v_dual_fmac_f32 v82, v87, v61
	v_rcp_f32_e32 v86, v83
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v87, -v64, v84, 1.0
	v_fmac_f32_e32 v85, v90, v62
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v54, -v54, v81, v63
	v_fma_f32 v59, -v59, v82, v79
	v_rcp_f32_e32 v63, v88
	v_rcp_f32_e32 v79, v89
	v_div_scale_f32 v90, null, v44, v44, v52
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v54, v54, v56, v81
	s_mov_b32 vcc_lo, s6
	v_fmac_f32_e32 v84, v87, v84
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v56, v59, v61, v82
	v_fma_f32 v59, -v83, v86, 1.0
	v_fma_f32 v55, -v55, v85, v80
	v_div_fixup_f32 v45, v54, v45, v57
	v_fma_f32 v57, -v88, v63, 1.0
	v_div_fixup_f32 v46, v56, v46, v58
	v_rcp_f32_e32 v58, v90
	v_mul_f32_e32 v54, v78, v84
	v_fmac_f32_e32 v86, v59, v86
	v_div_scale_f32 v56, s6, v49, v41, v49
	v_fma_f32 v59, -v89, v79, 1.0
	s_mov_b32 vcc_lo, s7
	v_fmac_f32_e32 v63, v57, v63
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v55, v55, v62, v85
	v_fma_f32 v61, -v64, v54, v78
	v_div_scale_f32 v57, s7, v50, v42, v50
	v_dual_fmac_f32 v79, v59, v79 :: v_dual_mul_f32 v62, v56, v86
	v_fma_f32 v80, -v90, v58, 1.0
	s_mov_b32 vcc_lo, s8
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v81, v57, v63
	v_div_scale_f32 v59, s9, v51, v43, v51
	v_mul_f32_e32 v46, v30, v46
	v_fmac_f32_e32 v58, v80, v58
	v_div_scale_f32 v80, s10, v52, v44, v52
	v_fmac_f32_e32 v54, v61, v84
	v_fma_f32 v61, -v83, v62, v56
	v_div_fixup_f32 v47, v55, v47, v53
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v62, v61, v86
	v_fma_f32 v61, -v88, v81, v57
	v_fma_f32 v56, -v83, v62, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v81, v61, v63
	v_mul_f32_e32 v85, v80, v58
	v_fma_f32 v64, -v64, v54, v78
	v_fma_f32 v57, -v88, v81, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v61, -v90, v85, v80
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v54, v64, v84, v54
	s_mov_b32 vcc_lo, s6
	v_mul_f32_e32 v82, v59, v79
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v56, v56, v86, v62
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v48, v54, v48, v60
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v57, v57, v63, v81
	v_fmac_f32_e32 v85, v61, v58
	v_fma_f32 v78, -v89, v82, v59
	v_div_fixup_f32 v41, v56, v41, v49
	v_fma_f32 v49, v29, v45, v46
	v_div_fixup_f32 v42, v57, v42, v50
	v_fma_f32 v45, v29, v45, -v46
	v_mul_f32_e32 v48, v32, v48
	v_fma_f32 v61, -v90, v85, v80
	s_mov_b32 vcc_lo, s9
	v_mul_f32_e32 v42, v26, v42
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v46, v31, v47, v48
	v_fma_f32 v47, v31, v47, -v48
	v_fma_f32 v48, v25, v41, v42
	v_fmac_f32_e32 v82, v78, v79
	v_fma_f32 v41, v25, v41, -v42
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v59, -v89, v82, v59
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v59, v59, v79, v82
	s_mov_b32 vcc_lo, s10
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v58, v61, v58, v85
	s_and_not1_b32 vcc_lo, exec_lo, s25
	v_div_fixup_f32 v43, v59, v43, v51
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v44, v58, v44, v52
	v_mul_f32_e32 v44, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v42, v27, v43, v44
	v_fma_f32 v43, v27, v43, -v44
	v_add_f32_e32 v44, v49, v46
	v_sub_f32_e32 v46, v49, v46
	v_add_f32_e32 v49, v45, v47
	v_sub_f32_e32 v45, v45, v47
	v_dual_add_f32 v47, v48, v42 :: v_dual_add_f32 v50, v41, v43
	v_dual_sub_f32 v41, v41, v43 :: v_dual_sub_f32 v42, v48, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v43, v47, v44 :: v_dual_add_f32 v52, v41, v45
	v_sub_f32_e32 v41, v45, v41
	v_add_f32_e32 v51, v42, v46
	ds_swizzle_b32 v53, v43 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v43, -v43, v43, s1
	ds_swizzle_b32 v55, v51 offset:swizzle(SWAP,1)
	s_wait_dscnt 0x1
	v_dual_add_f32 v43, v43, v53 :: v_dual_sub_f32 v44, v44, v47
	ds_swizzle_b32 v53, v43 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v43, -v43, v43, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v53
	v_dual_add_f32 v48, v50, v49 :: v_dual_sub_f32 v47, v49, v50
	v_cndmask_b32_e64 v49, -v51, v51, s1
	ds_swizzle_b32 v51, v41 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v41, -v41, v41, s1
	ds_swizzle_b32 v54, v48 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v45, -v48, v48, s1
	v_add_f32_e32 v49, v49, v55
	ds_swizzle_b32 v48, v47 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v47, -v47, v47, s1
	ds_swizzle_b32 v53, v43 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v43, -v43, v43, s3
	ds_swizzle_b32 v55, v49 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v49, -v49, v49, s2
	s_wait_dscnt 0x4
	v_dual_sub_f32 v42, v46, v42 :: v_dual_add_f32 v41, v41, v51
	ds_swizzle_b32 v56, v52 offset:swizzle(SWAP,1)
	s_wait_dscnt 0x4
	v_add_f32_e32 v45, v45, v54
	v_cndmask_b32_e64 v52, -v52, v52, s1
	ds_swizzle_b32 v46, v44 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v44, -v44, v44, s1
	ds_swizzle_b32 v54, v45 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v45, -v45, v45, s2
	s_wait_dscnt 0x3
	v_add_f32_e32 v49, v49, v55
	ds_swizzle_b32 v50, v42 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v42, -v42, v42, s1
	v_add_f32_e32 v43, v43, v53
	ds_swizzle_b32 v55, v49 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v49, -v49, v49, s3
	ds_swizzle_b32 v53, v43 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v43, -v43, v43, s4
	s_wait_dscnt 0x3
	v_dual_add_f32 v52, v52, v56 :: v_dual_add_f32 v45, v45, v54
	ds_swizzle_b32 v56, v52 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v51, -v52, v52, s2
	ds_swizzle_b32 v52, v41 offset:swizzle(SWAP,2)
	v_add_f32_e32 v44, v44, v46
	ds_swizzle_b32 v54, v45 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v41, -v41, v41, s2
	s_wait_dscnt 0x5
	v_add_f32_e32 v42, v42, v50
	v_cndmask_b32_e64 v45, -v45, v45, s3
	s_wait_dscnt 0x4
	v_dual_add_f32 v46, v47, v48 :: v_dual_add_f32 v49, v49, v55
	s_wait_dscnt 0x3
	v_add_f32_e32 v43, v43, v53
	ds_swizzle_b32 v55, v49 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v49, -v49, v49, s4
	ds_swizzle_b32 v53, v43 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v43, -v43, v43, s5
	s_wait_dscnt 0x4
	v_add_f32_e32 v51, v51, v56
	ds_swizzle_b32 v47, v44 offset:swizzle(SWAP,2)
	s_wait_dscnt 0x4
	v_add_f32_e32 v41, v41, v52
	ds_swizzle_b32 v50, v42 offset:swizzle(SWAP,2)
	s_wait_dscnt 0x4
	v_add_f32_e32 v45, v45, v54
	ds_swizzle_b32 v48, v46 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v56, v51 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v44, -v44, v44, s2
	v_cndmask_b32_e64 v42, -v42, v42, s2
	v_cndmask_b32_e64 v46, -v46, v46, s2
	v_cndmask_b32_e64 v51, -v51, v51, s3
	ds_swizzle_b32 v52, v41 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v41, -v41, v41, s3
	ds_swizzle_b32 v54, v45 offset:swizzle(SWAP,8)
	s_wait_dscnt 0x7
	v_add_f32_e32 v49, v49, v55
	v_cndmask_b32_e64 v45, -v45, v45, s4
	s_wait_dscnt 0x5
	v_dual_add_f32 v43, v43, v53 :: v_dual_add_f32 v44, v44, v47
	s_wait_dscnt 0x4
	v_add_f32_e32 v42, v42, v50
	s_wait_dscnt 0x3
	v_add_f32_e32 v46, v46, v48
	ds_swizzle_b32 v47, v44 offset:swizzle(SWAP,4)
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v56
	ds_swizzle_b32 v50, v42 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v44, -v44, v44, s3
	ds_swizzle_b32 v48, v46 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v42, -v42, v42, s3
	s_wait_dscnt 0x4
	v_add_f32_e32 v41, v41, v52
	ds_swizzle_b32 v56, v51 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v46, -v46, v46, s3
	v_cndmask_b32_e64 v51, -v51, v51, s4
	ds_swizzle_b32 v52, v41 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v41, -v41, v41, s4
	s_wait_dscnt 0x4
	v_add_f32_e32 v44, v44, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v42, v42, v50
	s_wait_dscnt 0x2
	v_add_f32_e32 v46, v46, v48
	ds_swizzle_b32 v47, v44 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v44, -v44, v44, s4
	ds_swizzle_b32 v50, v42 offset:swizzle(SWAP,8)
	v_add_f32_e32 v45, v45, v54
	ds_swizzle_b32 v48, v46 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v42, -v42, v42, s4
	s_wait_dscnt 0x4
	v_add_f32_e32 v51, v51, v56
	v_cndmask_b32_e64 v46, -v46, v46, s4
	s_wait_dscnt 0x2
	v_add_f32_e32 v44, v44, v47
	ds_swizzle_b32 v54, v45 offset:swizzle(SWAP,16)
	s_wait_dscnt 0x2
	v_add_f32_e32 v42, v42, v50
	ds_swizzle_b32 v47, v49 offset:swizzle(SWAP,16)
	s_wait_dscnt 0x2
	v_add_f32_e32 v46, v46, v48
	ds_swizzle_b32 v50, v44 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v48, v51 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v44, -v44, v44, s5
	v_add_f32_e32 v41, v41, v52
	ds_swizzle_b32 v55, v46 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v52, v42 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v45, -v45, v45, s5
	v_cndmask_b32_e64 v49, -v49, v49, s5
	v_cndmask_b32_e64 v51, -v51, v51, s5
	v_cndmask_b32_e64 v46, -v46, v46, s5
	v_cndmask_b32_e64 v42, -v42, v42, s5
	s_wait_dscnt 0x5
	v_add_f32_e32 v45, v45, v54
	s_wait_dscnt 0x3
	v_add_f32_e32 v44, v44, v50
	ds_swizzle_b32 v56, v41 offset:swizzle(SWAP,16)
	s_wait_dscnt 0x3
	v_dual_add_f32 v48, v51, v48 :: v_dual_add_f32 v47, v49, v47
	v_cndmask_b32_e64 v41, -v41, v41, s5
	s_wait_dscnt 0x2
	v_add_f32_e32 v46, v46, v55
	s_wait_dscnt 0x1
	v_dual_add_f32 v49, v42, v52 :: v_dual_mul_f32 v42, 0x3d800000, v43
	v_dual_mul_f32 v43, 0x3d800000, v45 :: v_dual_mul_f32 v48, 0x3d800000, v48
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v45, 0x3d800000, v47 :: v_dual_mul_f32 v46, 0x3d800000, v46
	s_wait_dscnt 0x0
	v_dual_mul_f32 v46, v34, v46 :: v_dual_add_f32 v47, v41, v56
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v41, v37, v42 :: v_dual_mul_f32 v42, v38, v43
	v_mul_f32_e32 v43, v39, v45
	v_mul_f32_e32 v45, 0x3d800000, v44
	v_mul_f32_e32 v44, v40, v48
	v_dual_mul_f32 v48, 0x3d800000, v49 :: v_dual_mul_f32 v49, 0x3d800000, v47
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v45, v33, v45
	v_dual_mul_f32 v47, v35, v48 :: v_dual_mul_f32 v48, v36, v49
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	v_lshlrev_b64_e32 v[49:50], 2, v[65:66]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v49, vcc_lo, s18, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v50, null, s19, v50, vcc_lo
	s_clause 0x1
	global_store_b128 v[49:50], v[41:44], off
	global_store_b128 v[49:50], v[45:48], off offset:16
.LBB0_22:                               ;   in Loop: Header=BB0_17 Depth=1
	v_lshlrev_b32_e32 v49, 2, v0
	s_mov_b32 s9, exec_lo
	ds_bpermute_b32 v50, v49, v41
	ds_bpermute_b32 v51, v49, v42
	ds_bpermute_b32 v52, v49, v43
	ds_bpermute_b32 v54, v49, v44
	ds_bpermute_b32 v53, v49, v45
	ds_bpermute_b32 v55, v49, v46
	ds_bpermute_b32 v56, v49, v47
	ds_bpermute_b32 v49, v49, v48
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v53, v53, v50, s1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v51, v55, v51, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v50, v56, v52, s1
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v49, v49, v54, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v52, |v53|, |v51|, |v50|
	v_max_num_f32_e64 v54, |v49|, |v49|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v52, v52, v54
	ds_bpermute_b32 v54, v72, v52
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v54, v54, v54
	v_max_num_f32_e32 v52, v52, v54
	ds_bpermute_b32 v54, v73, v52
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v54, v54, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v52, v52, v54
	ds_bpermute_b32 v54, v74, v52
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v54, v54, v54
	v_max_num_f32_e32 v52, v52, v54
	ds_bpermute_b32 v54, v75, v52
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v54, v54, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v52, v52, v54
	ds_bpermute_b32 v54, v76, v52
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v54, v54, v54
	v_max_num_f32_e32 v54, v52, v54
	v_mov_b32_e32 v52, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_neq_f32_e32 0, v54
	s_cbranch_execz .LBB0_48
; %bb.23:                               ;   in Loop: Header=BB0_17 Depth=1
	v_div_scale_f32 v52, null, 0x40e00000, 0x40e00000, v54
	v_div_scale_f32 v57, vcc_lo, v54, 0x40e00000, v54
	s_mov_b32 s10, 0
	v_rcp_f32_e32 v55, v52
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v56, -v52, v55, 1.0
	v_fmac_f32_e32 v55, v56, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v56, v57, v55
	v_fma_f32 v58, -v52, v56, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v56, v58, v55
	v_fma_f32 v52, -v52, v56, v57
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v52, v52, v55, v56
	v_mov_b32_e32 v56, 0x7149f2ca
	v_div_fixup_f32 v55, v52, 0x40e00000, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v52, 1.0 :: v_dual_mul_f32 v55, 0.5, v55
	s_branch .LBB0_25
.LBB0_24:                               ; %.preheader.preheader.i.i
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_mul_f32_e32 v57, s6, v55
	s_add_co_i32 s10, s10, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s10, 8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v58, null, v57, v57, v53
	v_rcp_f32_e32 v62, v58
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v80, -v58, v62, 1.0
	v_fmac_f32_e32 v62, v80, v62
	v_div_scale_f32 v59, null, v57, v57, v51
	v_div_scale_f32 v60, null, v57, v57, v50
	v_div_scale_f32 v61, null, v57, v57, v49
	v_div_scale_f32 v78, vcc_lo, v53, v57, v53
	v_div_scale_f32 v79, s6, v51, v57, v51
	v_rcp_f32_e32 v63, v59
	v_rcp_f32_e32 v64, v60
	v_rcp_f32_e32 v65, v61
	v_div_scale_f32 v83, s7, v50, v57, v50
	v_div_scale_f32 v80, s8, v49, v57, v49
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v81, -v59, v63, 1.0
	v_fma_f32 v82, -v60, v64, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v84, -v61, v65, 1.0
	v_dual_fmac_f32 v63, v81, v63 :: v_dual_fmac_f32 v64, v82, v64
	v_mul_f32_e32 v81, v78, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v65, v84, v65 :: v_dual_mul_f32 v82, v79, v63
	v_mul_f32_e32 v84, v83, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v86, -v58, v81, v78
	v_fma_f32 v87, -v59, v82, v79
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v88, -v60, v84, v83
	v_fmac_f32_e32 v81, v86, v62
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v85, v80, v65 :: v_dual_fmac_f32 v82, v87, v63
	v_fmac_f32_e32 v84, v88, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v58, -v58, v81, v78
	v_fma_f32 v89, -v61, v85, v80
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v59, -v59, v82, v79
	v_fma_f32 v60, -v60, v84, v83
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v58, v58, v62, v81
	s_mov_b32 vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v59, v59, v63, v82
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v58, v58, v57, v53
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v60, v60, v64, v84
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v59, v59, v57, v51
	v_rndne_f32_e32 v58, v58
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v60, v60, v57, v50
	v_rndne_f32_e32 v59, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v58, v58, s27, 0x40e00000
	v_rndne_f32_e32 v60, v60
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v59, v59, s27, 0x40e00000
	v_fma_f32 v58, -v58, v57, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v60, v60, s27, 0x40e00000
	v_fma_f32 v59, -v59, v57, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v58, v58, v58, 0
	v_fmac_f32_e32 v85, v89, v65
	v_fma_f32 v60, -v60, v57, v50
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v58, v59, v59
	v_fma_f32 v61, -v61, v85, v80
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v58, v60, v60
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v61, v61, v65, v85
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v61, v61, v57, v49
	v_rndne_f32_e32 v61, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_med3_num_f32 v61, v61, s27, 0x40e00000
	v_fma_f32 v59, -v61, v57, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v58, v59, v59
	ds_bpermute_b32 v59, v72, v58
	s_wait_dscnt 0x0
	v_add_f32_e32 v58, v58, v59
	ds_bpermute_b32 v59, v73, v58
	s_wait_dscnt 0x0
	v_add_f32_e32 v58, v58, v59
	ds_bpermute_b32 v59, v74, v58
	s_wait_dscnt 0x0
	v_add_f32_e32 v58, v58, v59
	ds_bpermute_b32 v59, v75, v58
	s_wait_dscnt 0x0
	v_add_f32_e32 v58, v58, v59
	ds_bpermute_b32 v59, v76, v58
	s_wait_dscnt 0x0
	v_add_f32_e32 v58, v58, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v58, v56
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v56, v56, v58, vcc_lo
	v_cndmask_b32_e32 v52, v52, v57, vcc_lo
	s_cbranch_scc0 .LBB0_48
.LBB0_25:                               ; %NodeBlock746
                                        ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s10, 3
	s_mov_b32 s7, -1
                                        ; implicit-def: $sgpr6
	s_cbranch_scc1 .LBB0_42
; %bb.26:                               ; %NodeBlock744
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_cmp_lt_i32 s10, 5
                                        ; implicit-def: $sgpr6
	s_cbranch_scc1 .LBB0_36
; %bb.27:                               ; %NodeBlock742
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_cmp_lt_i32 s10, 6
                                        ; implicit-def: $sgpr6
	s_cbranch_scc1 .LBB0_33
; %bb.28:                               ; %LeafBlock
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_cmp_lg_u32 s10, 6
	s_mov_b32 s6, -1
	s_cbranch_scc0 .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_25 Depth=2
	s_mov_b32 s6, 0
.LBB0_30:                               ; %Flow775
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s6, 2.0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_25 Depth=2
	s_mov_b32 s6, 0x3fedb6dc
.LBB0_32:                               ; %Flow776
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_mov_b32 s7, 0
.LBB0_33:                               ; %Flow777
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_35
; %bb.34:                               ;   in Loop: Header=BB0_25 Depth=2
	s_mov_b32 s6, 0x3fdb6db7
.LBB0_35:                               ; %Flow778
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_mov_b32 s7, 0
.LBB0_36:                               ; %Flow781
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_41
; %bb.37:                               ; %NodeBlock740
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_cmp_gt_i32 s10, 3
	s_mov_b32 s6, -1
	s_cbranch_scc0 .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_25 Depth=2
	s_mov_b32 s6, 0
.LBB0_39:                               ; %Flow779
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s6, 0x3fc92492
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_25 Depth=2
	s_mov_b32 s6, 0x3fb6db6e
.LBB0_41:                               ; %Flow782
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_mov_b32 s7, 0
.LBB0_42:                               ; %Flow786
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_24
; %bb.43:                               ; %NodeBlock738
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_cmp_lt_i32 s10, 1
	s_mov_b32 s6, 1.0
	s_cbranch_scc1 .LBB0_24
; %bb.44:                               ; %NodeBlock
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_cmp_gt_i32 s10, 1
	s_mov_b32 s6, -1
	s_cbranch_scc0 .LBB0_46
; %bb.45:                               ;   in Loop: Header=BB0_25 Depth=2
	s_mov_b32 s6, 0
.LBB0_46:                               ; %Flow783
                                        ;   in Loop: Header=BB0_25 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s6, 0x3fa49249
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_24
; %bb.47:                               ;   in Loop: Header=BB0_25 Depth=2
	s_mov_b32 s6, 0x3f924925
	s_branch .LBB0_24
.LBB0_48:                               ; %Flow788
                                        ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s9
	v_cmp_neq_f32_e64 s6, 0, v54
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v56, 0
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_17 Depth=1
	v_div_scale_f32 v54, null, v52, v52, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v56, v54
	v_fma_f32 v57, -v54, v56, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v56, v57, v56
	v_div_scale_f32 v57, vcc_lo, v53, v52, v53
	v_mul_f32_e32 v58, v57, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v59, -v54, v58, v57
	v_fmac_f32_e32 v58, v59, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v54, -v54, v58, v57
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v54, v54, v56, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v53, v54, v52, v53
	v_rndne_f32_e32 v53, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v53, v53, s27, 0x40e00000
	v_cvt_i32_f32_e32 v56, v53
.LBB0_50:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_17 Depth=1
	v_div_scale_f32 v53, null, v52, v52, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v54, v53
	v_fma_f32 v55, -v53, v54, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v54, v55, v54
	v_div_scale_f32 v55, vcc_lo, v51, v52, v51
	v_mul_f32_e32 v57, v55, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v53, v57, v55
	v_fmac_f32_e32 v57, v58, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v53, -v53, v57, v55
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v53, v53, v54, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v51, v53, v52, v51
	v_rndne_f32_e32 v51, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v51, v51, s27, 0x40e00000
	v_cvt_i32_f32_e32 v55, v51
.LBB0_52:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v53, 0
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_17 Depth=1
	v_div_scale_f32 v53, null, v52, v52, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v54, v53
	v_fma_f32 v57, -v53, v54, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v54, v57, v54
	v_div_scale_f32 v57, vcc_lo, v50, v52, v50
	v_mul_f32_e32 v58, v57, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v59, -v53, v58, v57
	v_fmac_f32_e32 v58, v59, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v53, -v53, v58, v57
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v53, v53, v54, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v50, v53, v52, v50
	v_rndne_f32_e32 v50, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v50, v50, s27, 0x40e00000
	v_cvt_i32_f32_e32 v53, v50
.LBB0_54:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_17 Depth=1
	v_div_scale_f32 v50, null, v52, v52, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v51, v50
	v_fma_f32 v54, -v50, v51, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v51, v54, v51
	v_div_scale_f32 v54, vcc_lo, v49, v52, v49
	v_mul_f32_e32 v57, v54, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v50, v57, v54
	v_fmac_f32_e32 v57, v58, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v50, -v50, v57, v54
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v50, v50, v51, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v49, v50, v52, v49
	v_rndne_f32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v49, v49, s27, 0x40e00000
	v_cvt_i32_f32_e32 v51, v49
.LBB0_56:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v49, v55, v56
	v_lshlrev_b32_e32 v54, 1, v77
	v_and_b32_e32 v56, 15, v56
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add3_u32 v49, v49, v53, v51
	v_mad_co_i64_i32 v[58:59], null, v54, s22, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_or_b32 v55, v55, 4, v56
	ds_bpermute_b32 v50, v72, v49
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v49, v49, v50
	ds_bpermute_b32 v50, v73, v49
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v49, v49, v50
	ds_bpermute_b32 v50, v74, v49
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v49, v49, v50
	ds_bpermute_b32 v50, v75, v49
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v57, v49, v50
	v_mad_co_u64_u32 v[49:50], null, 0x48, v58, s[20:21]
	v_and_b32_e32 v58, 15, v53
	ds_bpermute_b32 v53, v76, v57
	v_lshl_or_b32 v58, v51, 4, v58
	v_mad_co_u64_u32 v[50:51], null, 0x48, v59, v[50:51]
	v_and_b16 v51.h, 0xff, v55.l
	v_add_co_u32 v55, vcc_lo, v49, v70
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v51.l, 8, v58.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v56, null, 0, v50, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v51.l, v51.h, v51.l
	global_store_b16 v[55:56], v51, off offset:8
	s_and_saveexec_b32 s6, s0
	s_cbranch_execz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v53, v57, v53
	global_store_b64 v[49:50], v[52:53], off
.LBB0_58:                               ; %_Z26quantize_block_i4_128_waveILb1EEvPKfP12block_i4_128i.exit.i
                                        ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	ds_bpermute_b32 v41, v71, v41
	ds_bpermute_b32 v42, v71, v42
	ds_bpermute_b32 v49, v71, v43
	ds_bpermute_b32 v44, v71, v44
	ds_bpermute_b32 v43, v71, v45
	ds_bpermute_b32 v46, v71, v46
	ds_bpermute_b32 v47, v71, v47
	ds_bpermute_b32 v48, v71, v48
	s_mov_b32 s9, exec_lo
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v45, v43, v41, s1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v43, v46, v42, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v42, v47, v49, s1
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v41, v48, v44, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v44, |v45|, |v43|, |v42|
	v_max_num_f32_e64 v46, |v41|, |v41|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v44, v44, v46
	ds_bpermute_b32 v46, v72, v44
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v46, v46, v46
	v_max_num_f32_e32 v44, v44, v46
	ds_bpermute_b32 v46, v73, v44
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v46, v46, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v44, v44, v46
	ds_bpermute_b32 v46, v74, v44
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v46, v46, v46
	v_max_num_f32_e32 v44, v44, v46
	ds_bpermute_b32 v46, v75, v44
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v46, v46, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v44, v44, v46
	ds_bpermute_b32 v46, v76, v44
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v46, v46, v46
	v_max_num_f32_e32 v46, v44, v46
	v_mov_b32_e32 v44, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_neq_f32_e32 0, v46
	s_cbranch_execz .LBB0_84
; %bb.59:                               ;   in Loop: Header=BB0_17 Depth=1
	v_div_scale_f32 v44, null, 0x40e00000, 0x40e00000, v46
	v_div_scale_f32 v49, vcc_lo, v46, 0x40e00000, v46
	s_mov_b32 s10, 0
	v_rcp_f32_e32 v47, v44
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v48, -v44, v47, 1.0
	v_fmac_f32_e32 v47, v48, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v48, v49, v47
	v_fma_f32 v50, -v44, v48, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v48, v50, v47
	v_fma_f32 v44, -v44, v48, v49
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v44, v44, v47, v48
	v_mov_b32_e32 v48, 0x7149f2ca
	v_div_fixup_f32 v47, v44, 0x40e00000, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v44, 1.0 :: v_dual_mul_f32 v47, 0.5, v47
	s_branch .LBB0_61
.LBB0_60:                               ; %.preheader.preheader.i.1.i
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_mul_f32_e32 v49, s6, v47
	s_add_co_i32 s10, s10, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s10, 8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v51, null, v49, v49, v43
	v_rcp_f32_e32 v56, v51
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v62, -v51, v56, 1.0
	v_fmac_f32_e32 v56, v62, v56
	v_div_scale_f32 v50, null, v49, v49, v45
	v_div_scale_f32 v52, null, v49, v49, v42
	v_div_scale_f32 v53, null, v49, v49, v41
	v_div_scale_f32 v59, vcc_lo, v45, v49, v45
	v_div_scale_f32 v60, s6, v43, v49, v43
	v_rcp_f32_e32 v55, v50
	v_rcp_f32_e32 v57, v52
	v_rcp_f32_e32 v58, v53
	v_div_scale_f32 v64, s7, v42, v49, v42
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v61, -v50, v55, 1.0
	v_fma_f32 v63, -v52, v57, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v65, -v53, v58, 1.0
	v_fmac_f32_e32 v55, v61, v55
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v57, v63, v57 :: v_dual_fmac_f32 v58, v65, v58
	v_div_scale_f32 v61, s8, v41, v49, v41
	v_dual_mul_f32 v63, v60, v56 :: v_dual_mul_f32 v62, v59, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v79, -v51, v63, v60
	v_fma_f32 v78, -v50, v62, v59
	v_mul_f32_e32 v65, v64, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v63, v79, v56 :: v_dual_fmac_f32 v62, v78, v55
	v_fma_f32 v80, -v52, v65, v64
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v51, -v51, v63, v60
	v_fma_f32 v50, -v50, v62, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v65, v80, v57
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v50, v50, v55, v62
	s_mov_b32 vcc_lo, s6
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v52, -v52, v65, v64
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v51, v51, v56, v63
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v50, v50, v49, v45
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v52, v52, v57, v65
	v_div_fixup_f32 v51, v51, v49, v43
	s_mov_b32 vcc_lo, s8
	v_rndne_f32_e32 v50, v50
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v52, v52, v49, v42
	v_rndne_f32_e32 v51, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v50, v50, s27, 0x40e00000
	v_rndne_f32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v51, v51, s27, 0x40e00000
	v_fma_f32 v50, -v50, v49, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v52, v52, s27, 0x40e00000
	v_fma_f32 v51, -v51, v49, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v50, v50, v50, 0
	v_mul_f32_e32 v77, v61, v58
	v_fma_f32 v52, -v52, v49, v42
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v50, v51, v51
	v_fma_f32 v81, -v53, v77, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v50, v52, v52 :: v_dual_fmac_f32 v77, v81, v58
	v_fma_f32 v53, -v53, v77, v61
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v53, v53, v58, v77
	v_div_fixup_f32 v53, v53, v49, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v53, v53
	v_med3_num_f32 v53, v53, s27, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v51, -v53, v49, v41
	v_fmac_f32_e32 v50, v51, v51
	ds_bpermute_b32 v51, v72, v50
	s_wait_dscnt 0x0
	v_add_f32_e32 v50, v50, v51
	ds_bpermute_b32 v51, v73, v50
	s_wait_dscnt 0x0
	v_add_f32_e32 v50, v50, v51
	ds_bpermute_b32 v51, v74, v50
	s_wait_dscnt 0x0
	v_add_f32_e32 v50, v50, v51
	ds_bpermute_b32 v51, v75, v50
	s_wait_dscnt 0x0
	v_add_f32_e32 v50, v50, v51
	ds_bpermute_b32 v51, v76, v50
	s_wait_dscnt 0x0
	v_add_f32_e32 v50, v50, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v50, v48
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, v48, v50, vcc_lo
	v_cndmask_b32_e32 v44, v44, v49, vcc_lo
	s_cbranch_scc0 .LBB0_84
.LBB0_61:                               ; %NodeBlock760
                                        ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s10, 3
	s_mov_b32 s7, -1
                                        ; implicit-def: $sgpr6
	s_cbranch_scc1 .LBB0_78
; %bb.62:                               ; %NodeBlock758
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_cmp_lt_i32 s10, 5
                                        ; implicit-def: $sgpr6
	s_cbranch_scc1 .LBB0_72
; %bb.63:                               ; %NodeBlock756
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_cmp_lt_i32 s10, 6
                                        ; implicit-def: $sgpr6
	s_cbranch_scc1 .LBB0_69
; %bb.64:                               ; %LeafBlock754
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_cmp_lg_u32 s10, 6
	s_mov_b32 s6, -1
	s_cbranch_scc0 .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_61 Depth=2
	s_mov_b32 s6, 0
.LBB0_66:                               ; %Flow
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s6, 2.0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_61 Depth=2
	s_mov_b32 s6, 0x3fedb6dc
.LBB0_68:                               ; %Flow762
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_mov_b32 s7, 0
.LBB0_69:                               ; %Flow763
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_71
; %bb.70:                               ;   in Loop: Header=BB0_61 Depth=2
	s_mov_b32 s6, 0x3fdb6db7
.LBB0_71:                               ; %Flow764
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_mov_b32 s7, 0
.LBB0_72:                               ; %Flow767
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_77
; %bb.73:                               ; %NodeBlock752
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_cmp_gt_i32 s10, 3
	s_mov_b32 s6, -1
	s_cbranch_scc0 .LBB0_75
; %bb.74:                               ;   in Loop: Header=BB0_61 Depth=2
	s_mov_b32 s6, 0
.LBB0_75:                               ; %Flow765
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s6, 0x3fc92492
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_77
; %bb.76:                               ;   in Loop: Header=BB0_61 Depth=2
	s_mov_b32 s6, 0x3fb6db6e
.LBB0_77:                               ; %Flow768
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_mov_b32 s7, 0
.LBB0_78:                               ; %Flow772
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_60
; %bb.79:                               ; %NodeBlock750
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_cmp_lt_i32 s10, 1
	s_mov_b32 s6, 1.0
	s_cbranch_scc1 .LBB0_60
; %bb.80:                               ; %NodeBlock748
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_cmp_gt_i32 s10, 1
	s_mov_b32 s6, -1
	s_cbranch_scc0 .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_61 Depth=2
	s_mov_b32 s6, 0
.LBB0_82:                               ; %Flow769
                                        ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s6, 0x3fa49249
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_60
; %bb.83:                               ;   in Loop: Header=BB0_61 Depth=2
	s_mov_b32 s6, 0x3f924925
	s_branch .LBB0_60
.LBB0_84:                               ; %Flow774
                                        ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s9
	v_cmp_neq_f32_e64 s6, 0, v46
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_86
; %bb.85:                               ;   in Loop: Header=BB0_17 Depth=1
	v_div_scale_f32 v47, null, v44, v44, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v48, v47
	v_fma_f32 v49, -v47, v48, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v48, v49, v48
	v_div_scale_f32 v49, vcc_lo, v45, v44, v45
	v_mul_f32_e32 v50, v49, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v51, -v47, v50, v49
	v_fmac_f32_e32 v50, v51, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v47, -v47, v50, v49
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v47, v47, v48, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v45, v47, v44, v45
	v_rndne_f32_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v45, v45, s27, 0x40e00000
	v_cvt_i32_f32_e32 v47, v45
.LBB0_86:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_17 Depth=1
	v_div_scale_f32 v45, null, v44, v44, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v46, v45
	v_fma_f32 v48, -v45, v46, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v46, v48, v46
	v_div_scale_f32 v48, vcc_lo, v43, v44, v43
	v_mul_f32_e32 v49, v48, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v50, -v45, v49, v48
	v_fmac_f32_e32 v49, v50, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v45, -v45, v49, v48
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v45, v45, v46, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v43, v45, v44, v43
	v_rndne_f32_e32 v43, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v43, v43, s27, 0x40e00000
	v_cvt_i32_f32_e32 v46, v43
.LBB0_88:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v45, 0
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_17 Depth=1
	v_div_scale_f32 v45, null, v44, v44, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v48, v45
	v_fma_f32 v49, -v45, v48, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v48, v49, v48
	v_div_scale_f32 v49, vcc_lo, v42, v44, v42
	v_mul_f32_e32 v50, v49, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v51, -v45, v50, v49
	v_fmac_f32_e32 v50, v51, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v45, -v45, v50, v49
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v45, v45, v48, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v42, v45, v44, v42
	v_rndne_f32_e32 v42, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v42, v42, s27, 0x40e00000
	v_cvt_i32_f32_e32 v45, v42
.LBB0_90:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_92
; %bb.91:                               ;   in Loop: Header=BB0_17 Depth=1
	v_div_scale_f32 v42, null, v44, v44, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v43, v42
	v_fma_f32 v48, -v42, v43, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v43, v48, v43
	v_div_scale_f32 v48, vcc_lo, v41, v44, v41
	v_mul_f32_e32 v49, v48, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v50, -v42, v49, v48
	v_fmac_f32_e32 v49, v50, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v42, -v42, v49, v48
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v42, v42, v43, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v41, v42, v44, v41
	v_rndne_f32_e32 v41, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v41, v41, s27, 0x40e00000
	v_cvt_i32_f32_e32 v43, v41
.LBB0_92:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v41, v46, v47
	v_or_b32_e32 v48, 1, v54
	v_and_b32_e32 v47, 15, v47
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add3_u32 v41, v41, v45, v43
	v_mad_co_i64_i32 v[49:50], null, v48, s22, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_or_b32 v46, v46, 4, v47
	ds_bpermute_b32 v42, v72, v41
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v41, v41, v42
	ds_bpermute_b32 v42, v73, v41
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v41, v41, v42
	ds_bpermute_b32 v42, v74, v41
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v41, v41, v42
	ds_bpermute_b32 v42, v75, v41
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v48, v41, v42
	v_mad_co_u64_u32 v[41:42], null, 0x48, v49, s[20:21]
	v_and_b32_e32 v49, 15, v45
	ds_bpermute_b32 v45, v76, v48
	v_lshl_or_b32 v49, v43, 4, v49
	v_mad_co_u64_u32 v[42:43], null, 0x48, v50, v[42:43]
	v_and_b16 v43.h, 0xff, v46.l
	v_add_co_u32 v46, vcc_lo, v41, v70
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v43.l, 8, v49.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, 0, v42, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v43.l, v43.h, v43.l
	global_store_b16 v[46:47], v43, off offset:8
	s_and_saveexec_b32 s6, s0
	s_cbranch_execz .LBB0_15
; %bb.93:                               ;   in Loop: Header=BB0_17 Depth=1
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v45, v48, v45
	global_store_b64 v[41:42], v[44:45], off
	s_branch .LBB0_15
.LBB0_94:                               ; %.critedge
	s_endpgm
.Lfunc_end0:
	.size	fused_rmsnorm_mq_rotate_awq_i4_gfx12, .Lfunc_end0-fused_rmsnorm_mq_rotate_awq_i4_gfx12
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel fused_rmsnorm_mq_rotate_awq_i4_gfx12
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 328
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
		.amdhsa_next_free_vgpr 91
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-fused_rmsnorm_mq_rotate_awq_i4_gfx12)<<4)&4080)>>4
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
	.set .Lfused_rmsnorm_mq_rotate_awq_i4_gfx12.num_vgpr, 91
	.set .Lfused_rmsnorm_mq_rotate_awq_i4_gfx12.num_agpr, 0
	.set .Lfused_rmsnorm_mq_rotate_awq_i4_gfx12.numbered_sgpr, 31
	.set .Lfused_rmsnorm_mq_rotate_awq_i4_gfx12.num_named_barrier, 0
	.set .Lfused_rmsnorm_mq_rotate_awq_i4_gfx12.private_seg_size, 0
	.set .Lfused_rmsnorm_mq_rotate_awq_i4_gfx12.uses_vcc, 1
	.set .Lfused_rmsnorm_mq_rotate_awq_i4_gfx12.uses_flat_scratch, 0
	.set .Lfused_rmsnorm_mq_rotate_awq_i4_gfx12.has_dyn_sized_stack, 0
	.set .Lfused_rmsnorm_mq_rotate_awq_i4_gfx12.has_recursion, 0
	.set .Lfused_rmsnorm_mq_rotate_awq_i4_gfx12.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8216
; TotalNumSgprs: 33
; NumVgprs: 91
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 33
; NumVGPRsForWavesPerEU: 91
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
	.type	__hip_cuid_46a8726b8fa3f8e,@object ; @__hip_cuid_46a8726b8fa3f8e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_46a8726b8fa3f8e
__hip_cuid_46a8726b8fa3f8e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_46a8726b8fa3f8e, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_46a8726b8fa3f8e
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
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         76
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         80
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         84
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         86
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         88
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         90
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         92
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         94
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         136
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         192
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 328
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           fused_rmsnorm_mq_rotate_awq_i4_gfx12
    .private_segment_fixed_size: 0
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         fused_rmsnorm_mq_rotate_awq_i4_gfx12.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     91
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
