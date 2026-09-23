	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	attention_flash_q8_0_tile ; -- Begin function attention_flash_q8_0_tile
	.globl	attention_flash_q8_0_tile
	.p2align	8
	.type	attention_flash_q8_0_tile,@function
attention_flash_q8_0_tile:              ; @attention_flash_q8_0_tile
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[12:15], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 ttmp9, s12
	s_cbranch_scc1 .LBB0_95
; %bb.1:
	s_load_b64 s[2:3], s[0:1], 0x20
	s_wait_kmcnt 0x0
	s_load_b32 s2, s[2:3], 0x0
	s_load_b96 s[16:18], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_add_co_i32 s2, s2, 1
	s_mul_i32 s24, s17, ttmp7
	s_min_i32 s19, s2, s15
	s_cmp_gt_i32 s18, 0
	s_cselect_b32 s28, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s3, s28, exec_lo
	s_cselect_b32 s20, s19, s2
	s_mov_b32 s3, 0
	s_cmp_ge_i32 s24, s20
	s_cbranch_scc1 .LBB0_95
; %bb.2:
	s_abs_i32 s2, s13
	s_abs_i32 s6, s12
	s_cvt_f32_u32 s4, s2
	s_sub_co_i32 s5, 0, s2
	v_lshlrev_b32_e32 v12, 2, v0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s4, s4
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s4, s4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s5, s5, s4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s5, s4, s5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s4, s5
	s_xor_b32 s5, s12, s13
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s4, s6, s4
	s_ashr_i32 s5, s5, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s7, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s6, s6, s7
	s_add_co_i32 s7, s4, 1
	s_sub_co_i32 s8, s6, s2
	s_cmp_ge_u32 s6, s2
	s_cselect_b32 s4, s7, s4
	s_cselect_b32 s6, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s4, 1
	s_cmp_ge_u32 s6, s2
	s_cselect_b32 s2, s7, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s2, s2, s5
	s_sub_co_i32 s21, s2, s5
	s_load_b256 s[4:11], s[0:1], 0x0
	s_abs_i32 s12, s21
	s_add_co_i32 s1, s14, 0x7f
	s_cvt_f32_u32 s2, s12
	s_ashr_i32 s22, s1, 31
	s_sub_co_i32 s23, 0, s12
	s_lshr_b32 s22, s22, 25
	v_s_rcp_f32 s2, s2
	s_add_co_i32 s1, s1, s22
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	s_ashr_i32 s27, s1, 7
	s_mov_b32 s1, s3
	s_mul_f32 s0, s2, 0x4f7ffffe
	s_abs_i32 s2, ttmp9
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s0, s0
	s_mul_i32 s23, s23, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s22, s0, s23
	s_add_co_i32 s0, s0, s22
	s_cmp_gt_i32 s14, 0
	s_cselect_b32 s29, -1, 0
	s_cmp_lt_i32 s14, 1
	s_cbranch_scc1 .LBB0_5
; %bb.3:
	s_mul_i32 s30, s14, ttmp9
	v_mov_b32_e32 v2, 0
	s_ashr_i32 s31, s30, 31
	s_mov_b32 s22, 0
	s_lshl_b64 s[30:31], s[30:31], 2
	s_lshl2_add_u32 s23, s17, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[4:5], s[30:31]
	s_max_i32 s25, s27, 1
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	v_lshl_or_b32 v1, s22, 7, v12
	s_add_co_i32 s22, s22, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s22, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[3:4], 2, v[1:2]
	v_lshl_add_u32 v1, v1, 2, s23
	v_add_co_u32 v3, vcc_lo, s4, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v4, null, s5, v4, vcc_lo
	global_load_b128 v[3:6], v[3:4], off
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v1, v3, v4 offset1:1
	ds_store_2addr_b32 v1, v5, v6 offset0:2 offset1:3
	s_cbranch_scc0 .LBB0_4
.LBB0_5:
	s_wait_kmcnt 0x0
	s_add_co_i32 s5, s24, s17
	s_ashr_i32 s4, ttmp9, 31
	s_wait_alu depctr_sa_sdst(0)
	s_min_i32 s20, s5, s20
	s_ashr_i32 s5, s21, 31
	s_sub_co_i32 s25, s20, s24
	s_sub_co_i32 s18, s19, s18
	s_and_b32 s19, s28, exec_lo
	s_cselect_b32 s18, s18, 0
	s_mul_u64 s[0:1], s[2:3], s[0:1]
	s_sub_co_i32 s18, s18, s24
	v_cmp_eq_u32_e64 s0, 0, v0
	s_max_i32 s18, s18, 0
	s_mov_b32 s3, 0
	s_min_i32 s30, s18, s25
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s30, 1
	; wave barrier
	s_cbranch_scc1 .LBB0_17
; %bb.6:
	s_and_b32 s18, s30, 7
	s_cmp_lt_u32 s30, 8
	s_mov_b32 s19, -1
	s_cbranch_scc1 .LBB0_12
; %bb.7:
	v_mov_b32_e32 v1, 0xff800000
	s_and_b32 s3, s30, 0x7ffffff8
	s_mov_b32 s19, 0
	s_mov_b32 s20, 0
	s_branch .LBB0_9
.LBB0_8:                                ;   in Loop: Header=BB0_9 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s21
	s_add_co_i32 s19, s19, 8
	s_add_co_i32 s20, s20, 32
	s_cmp_eq_u32 s3, s19
	s_cbranch_scc1 .LBB0_11
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	s_and_saveexec_b32 s21, s0
	s_cbranch_execz .LBB0_8
; %bb.10:                               ;   in Loop: Header=BB0_9 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v2, s20
	ds_store_2addr_b32 v2, v1, v1 offset1:1
	ds_store_2addr_b32 v2, v1, v1 offset0:2 offset1:3
	ds_store_2addr_b32 v2, v1, v1 offset0:4 offset1:5
	ds_store_2addr_b32 v2, v1, v1 offset0:6 offset1:7
	s_branch .LBB0_8
.LBB0_11:
	s_cmp_lg_u32 s18, 0
	s_cselect_b32 s19, -1, 0
.LBB0_12:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s19
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_17
; %bb.13:
	v_mov_b32_e32 v1, 0xff800000
	s_lshl2_add_u32 s3, s3, 0
	s_lshl_b32 s18, s18, 2
	s_branch .LBB0_15
.LBB0_14:                               ;   in Loop: Header=BB0_15 Depth=1
	s_or_b32 exec_lo, exec_lo, s19
	s_add_co_i32 s18, s18, -4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s3, 4
	s_cmp_lg_u32 s18, 0
	s_cbranch_scc0 .LBB0_17
.LBB0_15:                               ; =>This Inner Loop Header: Depth=1
	s_and_saveexec_b32 s19, s0
	s_cbranch_execz .LBB0_14
; %bb.16:                               ;   in Loop: Header=BB0_15 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v2, s3
	ds_store_b32 v2, v1
	s_branch .LBB0_14
.LBB0_17:
	s_mul_i32 s0, s1, s12
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s4, s5
	s_sub_co_i32 s0, s2, s0
	s_add_co_i32 s2, s1, 1
	s_sub_co_i32 s4, s0, s12
	s_cmp_ge_u32 s0, s12
	v_and_b32_e32 v13, 28, v12
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s1, s2, s1
	s_cselect_b32 s0, s4, s0
	s_add_co_i32 s2, s1, 1
	s_cmp_ge_u32 s0, s12
	v_cndmask_b32_e64 v14, 0, 1, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s0, s2, s1
	s_ashr_i32 s1, s14, 31
	s_xor_b32 s0, s0, s3
	s_lshr_b32 s1, s1, 27
	s_sub_co_i32 s26, s0, s3
	s_add_co_i32 s1, s14, s1
	v_cndmask_b32_e64 v3, 0, 1, s29
	s_ashr_i32 s1, s1, 5
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_cmp_eq_u32_e64 s0, 0, v0
	s_add_co_i32 s2, s30, 3
	s_mul_i32 s26, s26, s1
	s_mul_i32 s1, s13, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s2, s25
	s_mul_i32 s4, s1, 34
	s_cbranch_scc1 .LBB0_34
; %bb.18:
	v_bfi_b32 v2, v1, 0, 32
	v_xor_b32_e32 v5, 16, v1
	v_xor_b32_e32 v6, 8, v1
	v_xor_b32_e32 v7, 4, v1
	v_xor_b32_e32 v8, 2, v1
	s_abs_i32 s31, s15
	v_cmp_lt_u32_e32 vcc_lo, v5, v2
	s_cvt_f32_u32 s1, s31
	v_xor_b32_e32 v10, 1, v1
	v_lshrrev_b32_e32 v4, 3, v0
	s_sub_co_i32 s2, 0, s31
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v1, v5, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, v6, v2
	v_s_rcp_f32 s1, s1
	s_lshl_b32 s3, s17, 2
	s_mul_i32 s12, s26, 34
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 31
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v1, v6, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, v7, v2
	v_lshlrev_b32_e32 v5, 2, v5
	s_max_i32 s33, s27, 1
	s_mul_f32 s1, s1, 0x4f7ffffe
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v1, v7 :: v_dual_lshlrev_b32 v6, 2, v6
	v_cmp_lt_u32_e32 vcc_lo, v8, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v1, v8, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, v10, v2
	v_and_b32_e32 v9, 7, v0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s2, s2, s1
	v_lshlrev_b32_e32 v7, 2, v7
	v_lshlrev_b32_e32 v8, 2, v8
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v10, v1, v10 :: v_dual_lshlrev_b32 v9, 4, v9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s2, s1, s2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s34, s1, s2
	v_lshl_or_b32 v9, v4, 7, v9
	v_mad_u32_u24 v4, v4, 34, s12
	s_delay_alu instid0(VALU_DEP_2)
	v_add3_u32 v2, 0, s3, v9
	v_lshlrev_b32_e32 v9, 2, v10
	s_branch .LBB0_20
.LBB0_19:                               ;   in Loop: Header=BB0_20 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_add_co_i32 s2, s30, 7
	s_add_co_i32 s1, s30, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s2, s25
	s_mov_b32 s30, s1
	s_cbranch_scc1 .LBB0_35
.LBB0_20:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_30 Depth 2
	s_add_co_i32 s1, s30, s24
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s2, s1
	s_cbranch_vccnz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_20 Depth=1
	s_abs_i32 s2, s1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s3, s2, s34
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s3, s3, s31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s2, s2, s3
	s_ashr_i32 s3, s1, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s12, s2, s31
	s_cmp_ge_u32 s2, s31
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s2, s12, s2
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s12, s2, s31
	s_cmp_ge_u32 s2, s31
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s2, s12, s2
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s2, s2, s3
.LBB0_22:                               ;   in Loop: Header=BB0_20 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_add_co_i32 s12, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_20 Depth=1
	s_abs_i32 s3, s12
	s_ashr_i32 s12, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s13, s3, s34
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s3, s3, s13
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s13, s3, s31
	s_cmp_ge_u32 s3, s31
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s3, s13, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s13, s3, s31
	s_cmp_ge_u32 s3, s31
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s3, s13, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s3, s12
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s12, s3, s12
.LBB0_24:                               ;   in Loop: Header=BB0_20 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_add_co_i32 s20, s1, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_20 Depth=1
	s_abs_i32 s3, s20
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s13, s3, s34
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s3, s3, s13
	s_ashr_i32 s13, s20, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s18, s3, s31
	s_cmp_ge_u32 s3, s31
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s3, s18, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s18, s3, s31
	s_cmp_ge_u32 s3, s31
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s3, s18, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s3, s13
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s20, s3, s13
.LBB0_26:                               ;   in Loop: Header=BB0_20 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_add_co_i32 s22, s1, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_20 Depth=1
	s_abs_i32 s1, s22
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s3, s1, s34
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s3, s3, s31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s1, s1, s3
	s_ashr_i32 s3, s22, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s13, s1, s31
	s_cmp_ge_u32 s1, s31
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s1, s13, s1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s13, s1, s31
	s_cmp_ge_u32 s1, s31
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s1, s13, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s22, s1, s3
.LBB0_28:                               ;   in Loop: Header=BB0_20 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_31
; %bb.29:                               ;   in Loop: Header=BB0_20 Depth=1
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[2:3], s[2:3], s[4:5]
	s_ashr_i32 s21, s20, 31
	s_ashr_i32 s23, s22, 31
	s_wait_dscnt 0x2
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v17, v4
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v16, 0
	s_mul_u64 s[18:19], s[12:13], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[6:7], s[2:3]
	s_mul_u64 s[2:3], s[20:21], s[4:5]
	s_mul_u64 s[22:23], s[22:23], s[4:5]
	s_add_nc_u64 s[18:19], s[6:7], s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[20:21], s[6:7], s[2:3]
	s_add_nc_u64 s[22:23], s[6:7], s[22:23]
	s_mov_b32 s35, s33
.LBB0_30:                               ;   Parent Loop BB0_20 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_ashrrev_i32_e32 v24, 31, v17
	s_wait_dscnt 0x1
	v_add_co_u32 v19, vcc_lo, s18, v17
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v21, s1, s20, v17
	v_add_co_u32 v23, s2, s22, v17
	v_add_co_u32 v25, s3, s12, v17
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, s13, v24, s3
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, s19, v24, vcc_lo
	v_add_co_ci_u32_e64 v22, null, s21, v24, s1
	v_add_co_ci_u32_e64 v24, null, s23, v24, s2
	v_add_co_u32 v27, vcc_lo, v19, v13
	v_add_co_u32 v31, s2, v23, v13
	v_add_co_u32 v29, s1, v21, v13
	v_add_co_u32 v33, s3, v25, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, 0, v20, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v32, null, 0, v24, s2
	v_add_co_ci_u32_e64 v34, null, 0, v26, s3
	v_add_co_ci_u32_e64 v30, null, 0, v22, s1
	s_clause 0x7
	global_load_d16_b16 v25, v[25:26], off
	global_load_d16_b16 v26, v[19:20], off
	global_load_d16_b16 v35, v[21:22], off
	global_load_d16_b16 v23, v[23:24], off
	global_load_b32 v24, v[31:32], off offset:2
	global_load_b32 v31, v[33:34], off offset:2
	global_load_b32 v27, v[27:28], off offset:2
	global_load_b32 v28, v[29:30], off offset:2
	ds_load_2addr_b32 v[19:20], v18 offset0:2 offset1:3
	ds_load_2addr_b32 v[21:22], v18 offset1:1
	s_add_co_i32 s35, s35, -1
	v_add_nc_u32_e32 v17, 0x88, v17
	s_cmp_eq_u32 s35, 0
	s_wait_loadcnt 0x3
	v_ashrrev_i16 v30.l, 8, v24.l
	s_wait_loadcnt 0x2
	v_ashrrev_i16 v32.l, 8, v31.h
	v_ashrrev_i16 v33.l, 8, v31.l
	s_wait_loadcnt 0x1
	v_ashrrev_i16 v37.l, 8, v27.l
	s_wait_loadcnt 0x0
	v_ashrrev_i16 v40.l, 8, v28.l
	v_bfe_i32 v34, v31, 0, 8
	v_bfe_i32 v31, v31, 16, 8
	v_ashrrev_i16 v36.l, 8, v27.h
	v_bfe_i32 v33, v33, 0, 16
	v_bfe_i32 v32, v32, 0, 16
	v_bfe_i32 v37, v37, 0, 16
	v_bfe_i32 v40, v40, 0, 16
	v_bfe_i32 v30, v30, 0, 16
	v_bfe_i32 v38, v27, 0, 8
	v_bfe_i32 v27, v27, 16, 8
	v_bfe_i32 v41, v28, 0, 8
	v_bfe_i32 v42, v24, 0, 8
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v31, v31
	v_bfe_i32 v36, v36, 0, 16
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v37, v37
	v_cvt_f32_i32_e32 v40, v40
	v_cvt_f32_i32_e32 v30, v30
	v_ashrrev_i16 v29.l, 8, v24.h
	v_ashrrev_i16 v39.l, 8, v28.h
	v_bfe_i32 v28, v28, 16, 8
	v_bfe_i32 v24, v24, 16, 8
	v_cvt_f32_i32_e32 v38, v38
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v41, v41
	v_cvt_f32_i32_e32 v42, v42
	v_fma_mix_f32 v31, v25, v31, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v34, v25, v34, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v36, v36
	v_fma_mix_f32 v32, v25, v32, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v25, v25, v33, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v33, v26, v37, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v37, v35, v40, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v30, v23, v30, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v28, v28
	v_bfe_i32 v39, v39, 0, 16
	v_cvt_f32_i32_e32 v24, v24
	v_bfe_i32 v29, v29, 0, 16
	v_fma_mix_f32 v27, v26, v27, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v38, v26, v38, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v41, v35, v41, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v42, v23, v42, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v26, v26, v36, neg(0) op_sel_hi:[1,0,0]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v25, v25, v22
	v_mul_f32_e32 v33, v33, v22
	v_mul_f32_e32 v36, v37, v22
	v_mul_f32_e32 v22, v30, v22
	v_cvt_f32_i32_e32 v39, v39
	v_fma_mix_f32 v28, v35, v28, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v29, v29
	v_fma_mix_f32 v24, v23, v24, neg(0) op_sel_hi:[1,0,0]
	v_dual_fmac_f32 v25, v34, v21 :: v_dual_fmac_f32 v36, v41, v21
	v_fmac_f32_e32 v22, v42, v21
	v_fmac_f32_e32 v33, v38, v21
	v_fma_mix_f32 v30, v35, v39, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v21, v23, v29, neg(0) op_sel_hi:[1,0,0]
	v_dual_fmac_f32 v25, v31, v19 :: v_dual_fmac_f32 v36, v28, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v33, v27, v19 :: v_dual_fmac_f32 v22, v24, v19
	v_dual_fmac_f32 v25, v32, v20 :: v_dual_add_nc_u32 v18, 0x200, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v36, v30, v20
	v_dual_fmac_f32 v33, v26, v20 :: v_dual_fmac_f32 v22, v21, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v10, v10, v25 :: v_dual_add_f32 v15, v15, v36
	v_dual_add_f32 v11, v11, v33 :: v_dual_add_f32 v16, v16, v22
	s_cbranch_scc0 .LBB0_30
	s_branch .LBB0_32
.LBB0_31:                               ;   in Loop: Header=BB0_20 Depth=1
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v16, 0
.LBB0_32:                               ;   in Loop: Header=BB0_20 Depth=1
	s_wait_dscnt 0x2
	ds_bpermute_b32 v17, v5, v10
	ds_bpermute_b32 v18, v5, v11
	s_wait_dscnt 0x3
	ds_bpermute_b32 v19, v5, v15
	s_wait_dscnt 0x3
	ds_bpermute_b32 v20, v5, v16
	s_wait_dscnt 0x2
	v_dual_add_f32 v10, v10, v17 :: v_dual_add_f32 v11, v11, v18
	s_wait_dscnt 0x0
	v_dual_add_f32 v15, v15, v19 :: v_dual_add_f32 v16, v16, v20
	ds_bpermute_b32 v17, v6, v10
	ds_bpermute_b32 v18, v6, v11
	ds_bpermute_b32 v19, v6, v15
	ds_bpermute_b32 v20, v6, v16
	s_wait_dscnt 0x2
	v_dual_add_f32 v10, v10, v17 :: v_dual_add_f32 v11, v11, v18
	s_wait_dscnt 0x0
	v_dual_add_f32 v15, v15, v19 :: v_dual_add_f32 v16, v16, v20
	ds_bpermute_b32 v17, v7, v10
	ds_bpermute_b32 v18, v7, v11
	ds_bpermute_b32 v19, v7, v15
	ds_bpermute_b32 v20, v7, v16
	s_wait_dscnt 0x2
	v_dual_add_f32 v10, v10, v17 :: v_dual_add_f32 v17, v11, v18
	s_wait_dscnt 0x1
	v_add_f32_e32 v18, v15, v19
	ds_bpermute_b32 v11, v8, v10
	s_wait_dscnt 0x1
	v_add_f32_e32 v16, v16, v20
	ds_bpermute_b32 v15, v8, v17
	ds_bpermute_b32 v19, v8, v18
	s_wait_dscnt 0x2
	v_add_f32_e32 v11, v10, v11
	ds_bpermute_b32 v20, v8, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v15, v17, v15
	s_wait_dscnt 0x1
	v_add_f32_e32 v10, v18, v19
	ds_bpermute_b32 v18, v9, v11
	ds_bpermute_b32 v17, v9, v10
	s_wait_dscnt 0x2
	v_add_f32_e32 v16, v16, v20
	ds_bpermute_b32 v19, v9, v15
	ds_bpermute_b32 v20, v9, v16
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_19
; %bb.33:                               ;   in Loop: Header=BB0_20 Depth=1
	s_wait_dscnt 0x0
	v_dual_add_f32 v11, v11, v18 :: v_dual_add_f32 v16, v16, v20
	v_dual_add_f32 v15, v15, v19 :: v_dual_add_f32 v10, v10, v17
	s_lshl2_add_u32 s2, s30, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v11, s16, v11
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v17, s2
	v_dual_mul_f32 v15, s16, v15 :: v_dual_mul_f32 v16, s16, v16
	v_mul_f32_e32 v10, s16, v10
	ds_store_2addr_b32 v17, v11, v15 offset1:1
	ds_store_2addr_b32 v17, v10, v16 offset0:2 offset1:3
	s_branch .LBB0_19
.LBB0_34:
	s_mov_b32 s1, s30
.LBB0_35:
	v_cmp_eq_u32_e64 s0, 0, v0
	s_cmp_ge_i32 s1, s25
	s_cbranch_scc1 .LBB0_46
; %bb.36:
	v_bfi_b32 v2, v1, 0, 32
	v_xor_b32_e32 v5, 16, v1
	v_xor_b32_e32 v6, 8, v1
	v_xor_b32_e32 v7, 4, v1
	v_xor_b32_e32 v8, 2, v1
	s_abs_i32 s12, s15
	v_cmp_lt_u32_e32 vcc_lo, v5, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s2, s12
	v_xor_b32_e32 v10, 1, v1
	v_lshrrev_b32_e32 v4, 3, v0
	s_sub_co_i32 s3, 0, s12
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v1, v5, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, v6, v2
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s2, s2
	s_lshl_b32 s13, s17, 2
	s_mul_i32 s18, s26, 34
	s_ashr_i32 s5, s4, 31
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v1, v6, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, v7, v2
	v_lshlrev_b32_e32 v5, 2, v5
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v1, v7 :: v_dual_lshlrev_b32 v6, 2, v6
	v_cmp_lt_u32_e32 vcc_lo, v8, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s2, s2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v1, v8, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, v10, v2
	v_and_b32_e32 v9, 7, v0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s3, s3, s2
	v_lshlrev_b32_e32 v7, 2, v7
	v_lshlrev_b32_e32 v8, 2, v8
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v10, v1, v10 :: v_dual_lshlrev_b32 v9, 4, v9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s3, s2, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_lshl_or_b32 v9, v4, 7, v9
	v_mad_u32_u24 v4, v4, 34, s18
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s18, s2, s3
	v_add3_u32 v2, 0, s13, v9
	v_lshlrev_b32_e32 v9, 2, v10
	s_max_i32 s13, s27, 1
	s_branch .LBB0_38
.LBB0_37:                               ;   in Loop: Header=BB0_38 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s1, s25
	s_cbranch_scc0 .LBB0_46
.LBB0_38:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_42 Depth 2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_add_co_i32 s2, s1, s24
	s_cbranch_vccnz .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_38 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s3, s2
	s_ashr_i32 s2, s2, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s19, s3, s18
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s19, s19, s12
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s3, s3, s19
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s19, s3, s12
	s_cmp_ge_u32 s3, s12
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s3, s19, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s19, s3, s12
	s_cmp_ge_u32 s3, s12
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s3, s19, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s3, s2
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s2, s3, s2
.LBB0_40:                               ;   in Loop: Header=BB0_38 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v3
	s_cbranch_vccnz .LBB0_43
; %bb.41:                               ;   in Loop: Header=BB0_38 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s3, s2, 31
	s_wait_dscnt 0x0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, v4
	v_mov_b32_e32 v15, v2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[2:3], s[2:3], s[4:5]
	s_mov_b32 s19, s13
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[6:7], s[2:3]
.LBB0_42:                               ;   Parent Loop BB0_38 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_dscnt 0x2
	v_ashrrev_i32_e32 v17, 31, v11
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v16, vcc_lo, s2, v11
	v_add_nc_u32_e32 v11, 0x88, v11
	s_add_co_i32 s19, s19, -1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s3, v17, vcc_lo
	v_add_co_u32 v18, vcc_lo, v16, v13
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s19, 0
	s_wait_dscnt 0x1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, 0, v17, vcc_lo
	s_wait_dscnt 0x0
	s_clause 0x1
	global_load_d16_b16 v20, v[16:17], off
	global_load_b32 v21, v[18:19], off offset:2
	ds_load_2addr_b32 v[16:17], v15 offset0:2 offset1:3
	ds_load_2addr_b32 v[18:19], v15 offset1:1
	v_add_nc_u32_e32 v15, 0x200, v15
	s_wait_loadcnt 0x0
	v_ashrrev_i16 v22.l, 8, v21.l
	v_bfe_i32 v23, v21, 0, 8
	v_bfe_i32 v24, v21, 16, 8
	v_ashrrev_i32_e32 v21, 24, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_bfe_i32 v22, v22, 0, 16
	v_cvt_f32_i32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_i32_e32 v24, v24
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_i32_e32 v22, v22
	v_fma_mix_f32 v23, v20, v23, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v22, v20, v22, neg(0) op_sel_hi:[1,0,0]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v19, v19, v22
	v_fma_mix_f32 v22, v20, v24, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v19, v18, v23
	v_fma_mix_f32 v18, v20, v21, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v19, v16, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v17, v18
	v_add_f32_e32 v10, v10, v19
	s_cbranch_scc0 .LBB0_42
	s_branch .LBB0_44
.LBB0_43:                               ;   in Loop: Header=BB0_38 Depth=1
	v_mov_b32_e32 v10, 0
.LBB0_44:                               ;   in Loop: Header=BB0_38 Depth=1
	s_wait_dscnt 0x0
	ds_bpermute_b32 v11, v5, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v10, v10, v11
	ds_bpermute_b32 v11, v6, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v10, v10, v11
	ds_bpermute_b32 v11, v7, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v10, v10, v11
	ds_bpermute_b32 v11, v8, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v10, v10, v11
	ds_bpermute_b32 v11, v9, v10
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_37
; %bb.45:                               ;   in Loop: Header=BB0_38 Depth=1
	s_lshl2_add_u32 s3, s1, 0
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_dual_add_f32 v10, v10, v11 :: v_dual_mov_b32 v11, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v10, s16, v10
	ds_store_b32 v11, v10
	s_branch .LBB0_37
.LBB0_46:
	v_cmp_gt_i32_e32 vcc_lo, s25, v0
	v_mov_b32_e32 v2, 0xf149f2ca
	; wave barrier
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_50
; %bb.47:
	v_dual_mov_b32 v5, v0 :: v_dual_add_nc_u32 v4, 0, v12
	v_mov_b32_e32 v2, 0xf149f2ca
	s_mov_b32 s2, 0
.LBB0_48:                               ; =>This Inner Loop Header: Depth=1
	ds_load_b32 v6, v4
	v_dual_max_num_f32 v2, v2, v2 :: v_dual_add_nc_u32 v5, 32, v5
	v_add_nc_u32_e32 v4, 0x80, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e64 s0, s25, v5
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s0, s2
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v6, v6, v6
	v_max_num_f32_e32 v2, v2, v6
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_48
; %bb.49:
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_bfi_b32 v8, v1, 0, 32
	v_xor_b32_e32 v4, 16, v1
	v_xor_b32_e32 v6, 8, v1
	v_xor_b32_e32 v7, 4, v1
	v_xor_b32_e32 v9, 2, v1
	s_mov_b32 s1, 0
	v_cmp_lt_u32_e64 s0, v4, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v4, v1, v4, s0
	v_cmp_lt_u32_e64 s0, v6, v8
	v_lshlrev_b32_e32 v4, 2, v4
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v6, v1, v6, s0
	v_cmp_lt_u32_e64 s0, v7, v8
	ds_bpermute_b32 v5, v4, v2
	v_max_num_f32_e32 v2, v2, v2
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v7, v1, v7, s0
	v_cmp_lt_u32_e64 s0, v9, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v7, 2, v7
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v9, v1, v9, s0
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v2, v2, v5
	ds_bpermute_b32 v5, v6, v2
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	v_max_num_f32_e32 v2, v2, v5
	ds_bpermute_b32 v5, v7, v2
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v10, v5, v5
	v_lshlrev_b32_e32 v5, 2, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v9, v2, v10
	v_xor_b32_e32 v10, 1, v1
	ds_bpermute_b32 v2, v5, v9
	v_cmp_lt_u32_e64 s0, v10, v8
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v1, v1, v10, s0
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v8, v2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v1, v9, v8 :: v_dual_lshlrev_b32 v2, 2, v1
	ds_bpermute_b32 v8, v2, v1
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v8, v8, v8
	v_dual_max_num_f32 v1, v1, v8 :: v_dual_mov_b32 v8, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_54
; %bb.51:
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v9, 0, v12
	v_mov_b32_e32 v10, v0
.LBB0_52:                               ; =>This Inner Loop Header: Depth=1
	ds_load_b32 v11, v9
	s_wait_dscnt 0x0
	v_dual_sub_f32 v11, v11, v1 :: v_dual_add_nc_u32 v10, 32, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v15, 0x3fb8aa3b, v11
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v11
	v_fma_f32 v16, 0x3fb8aa3b, v11, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v16, 0x32a5705f, v11 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v11
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, 0x7f800000, v15, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, s25, v10
	ds_store_b32 v9, v11
	v_dual_add_f32 v8, v8, v11 :: v_dual_add_nc_u32 v9, 0x80, v9
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_52
; %bb.53:
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_bpermute_b32 v4, v4, v8
	s_abs_i32 s0, s17
	s_add_co_i32 s3, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s1, s0
	s_sub_co_i32 s2, 0, s0
	s_add_co_i32 s3, s3, -1
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s1, s1
	s_abs_i32 s5, s3
	; wave barrier
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s1, s1, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s1, s1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v8, v4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s2, s2, s1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s2, s1, s2
	ds_bpermute_b32 v6, v6, v4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s1, s1, s2
	s_xor_b32 s2, s3, s17
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s1, s5, s1
	s_ashr_i32 s2, s2, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s3, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s3, s5, s3
	s_add_co_i32 s5, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s6, s3, s0
	s_cmp_ge_u32 s3, s0
	s_cselect_b32 s1, s5, s1
	s_cselect_b32 s3, s6, s3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s5, s1, 1
	s_cmp_ge_u32 s3, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s0, s5, s1
	s_add_co_i32 s1, s14, 2
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, s0, s2
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v6
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s0, s0, s2
	s_mov_b32 s2, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s0, s0, ttmp9
	ds_bpermute_b32 v6, v7, v4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, ttmp7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v6
	ds_bpermute_b32 v5, v5, v4
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	ds_bpermute_b32 v2, v2, v4
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .LBB0_56
; %bb.55:
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v4, v2
	v_mov_b32_e32 v0, 0
	global_store_b64 v0, v[1:2], s[0:1]
.LBB0_56:
	s_or_b32 exec_lo, exec_lo, s2
	v_cmp_ne_u32_e32 vcc_lo, 1, v3
	s_cbranch_vccnz .LBB0_95
; %bb.57:
	s_cmp_gt_i32 s25, 7
	v_mov_b32_e32 v5, 0
	s_cselect_b32 s2, -1, 0
	s_abs_i32 s3, s15
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s5, s3
	s_sub_co_i32 s6, 0, s3
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s5, s5
	s_mul_f32 s5, s5, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s7, s5
	s_max_i32 s5, s27, 1
	s_mul_i32 s6, s6, s7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_hi_u32 s10, s7, s6
	s_mov_b32 s6, 0
	s_add_co_i32 s7, s7, s10
	s_branch .LBB0_59
.LBB0_58:                               ;   in Loop: Header=BB0_59 Depth=1
	v_lshlrev_b64_e32 v[6:7], 2, v[4:5]
	s_add_co_i32 s6, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s6, s5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, s0, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	global_store_b128 v[6:7], v[0:3], off offset:8
	s_cbranch_scc0 .LBB0_95
.LBB0_59:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_62 Depth 2
                                        ;     Child Loop BB0_81 Depth 2
                                        ;     Child Loop BB0_93 Depth 2
	v_lshl_or_b32 v4, s6, 7, v12
	v_mov_b32_e32 v6, v5
	s_wait_dscnt 0x0
	v_mov_b32_e32 v2, v5
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_mov_b32 s11, 0
	v_lshrrev_b32_e32 v0, 5, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v3, v6 :: v_dual_add_nc_u32 v0, s26, v0
	v_mul_lo_u32 v15, v0, 34
	v_dual_mov_b32 v0, v5 :: v_dual_mov_b32 v1, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_ashrrev_i32_e32 v16, 31, v15
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_78
; %bb.60:                               ;   in Loop: Header=BB0_59 Depth=1
	v_add_co_u32 v6, vcc_lo, s8, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s9, v16, vcc_lo
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s10, 0
	s_branch .LBB0_62
.LBB0_61:                               ;   in Loop: Header=BB0_62 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[17:18], null, s11, s4, v[6:7]
	v_mad_co_i64_i32 v[19:20], null, s13, s4, v[6:7]
	v_mad_co_i64_i32 v[21:22], null, s14, s4, v[6:7]
	v_mad_co_i64_i32 v[27:28], null, s12, s4, v[6:7]
	s_add_co_i32 s12, s10, 15
	s_add_co_i32 s11, s10, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s25
	s_mov_b32 s10, s11
	v_add_co_u32 v23, vcc_lo, v17, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, 0, v18, vcc_lo
	v_add_co_u32 v25, vcc_lo, v19, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, 0, v20, vcc_lo
	s_clause 0x1
	global_load_b32 v29, v[23:24], off offset:2
	global_load_b32 v30, v[25:26], off offset:2
	v_add_co_u32 v23, vcc_lo, v21, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, 0, v22, vcc_lo
	v_add_co_u32 v25, vcc_lo, v27, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, 0, v28, vcc_lo
	s_clause 0x5
	global_load_b32 v23, v[23:24], off offset:2
	global_load_b32 v24, v[25:26], off offset:2
	global_load_d16_b16 v17, v[17:18], off
	global_load_d16_b16 v18, v[27:28], off
	global_load_d16_b16 v21, v[21:22], off
	global_load_d16_b16 v19, v[19:20], off
	s_wait_loadcnt 0x7
	v_bfe_i32 v20, v29, 0, 8
	v_bfe_i32 v22, v29, 8, 8
	v_bfe_i32 v25, v29, 16, 8
	v_ashrrev_i32_e32 v29, 24, v29
	s_wait_loadcnt 0x6
	v_bfe_i32 v26, v30, 0, 8
	v_bfe_i32 v27, v30, 8, 8
	v_bfe_i32 v28, v30, 16, 8
	v_ashrrev_i32_e32 v30, 24, v30
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v22, v22
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v29, v29
	s_wait_loadcnt 0x5
	v_bfe_i32 v31, v23, 0, 8
	v_bfe_i32 v32, v23, 8, 8
	v_bfe_i32 v33, v23, 16, 8
	v_ashrrev_i32_e32 v23, 24, v23
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v30, v30
	s_wait_loadcnt 0x3
	v_fma_mix_f32 v20, v17, v20, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v22, v17, v22, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v25, v17, v25, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v17, v17, v29, neg(0) op_sel_hi:[1,0,0]
	v_bfe_i32 v34, v24, 0, 8
	v_bfe_i32 v35, v24, 8, 8
	v_bfe_i32 v36, v24, 16, 8
	v_ashrrev_i32_e32 v24, 24, v24
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v23, v23
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v30, v19, v30, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v26, v19, v26, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v27, v19, v27, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v19, v19, v28, neg(0) op_sel_hi:[1,0,0]
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v0, v10, v20 :: v_dual_fmac_f32 v1, v10, v22
	v_fmac_f32_e32 v2, v10, v25
	v_fmac_f32_e32 v3, v10, v17
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v24, v24
	v_fma_mix_f32 v23, v21, v23, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v28, v21, v31, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v29, v21, v32, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v21, v21, v33, neg(0) op_sel_hi:[1,0,0]
	v_dual_fmac_f32 v0, v11, v26 :: v_dual_fmac_f32 v1, v11, v27
	v_dual_fmac_f32 v2, v11, v19 :: v_dual_fmac_f32 v3, v11, v30
	v_fma_mix_f32 v24, v18, v24, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v31, v18, v34, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v10, v18, v35, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v11, v18, v36, neg(0) op_sel_hi:[1,0,0]
	v_dual_fmac_f32 v0, v8, v28 :: v_dual_fmac_f32 v1, v8, v29
	v_dual_fmac_f32 v2, v8, v21 :: v_dual_fmac_f32 v3, v8, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v0, v9, v31 :: v_dual_fmac_f32 v1, v9, v10
	v_dual_fmac_f32 v2, v9, v11 :: v_dual_fmac_f32 v3, v9, v24
	s_cbranch_scc1 .LBB0_78
.LBB0_62:                               ;   Parent Loop BB0_59 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl2_add_u32 s11, s10, 0
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s11
	s_add_co_i32 s12, s10, s24
	ds_load_2addr_b32 v[10:11], v8 offset1:1
	ds_load_2addr_b32 v[8:9], v8 offset0:2 offset1:3
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s13, s12
	s_cbranch_vccnz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_62 Depth=2
	s_abs_i32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s14, s13, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s14, s14, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s13, s13, s14
	s_ashr_i32 s14, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s13, s3
	s_cmp_ge_u32 s13, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s13, s15, s13
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s13, s3
	s_cmp_ge_u32 s13, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s13, s15, s13
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s13, s13, s14
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s13, s13, s14
.LBB0_64:                               ;   in Loop: Header=BB0_62 Depth=2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_add_co_i32 s14, s12, 1
	s_cbranch_vccnz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_62 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s15, s14
	s_ashr_i32 s14, s14, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s16, s15, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s16, s16, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s15, s16
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s15, s3
	s_cmp_ge_u32 s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s15, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s15, s3
	s_cmp_ge_u32 s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s15, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s15, s15, s14
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s14, s15, s14
.LBB0_66:                               ;   in Loop: Header=BB0_62 Depth=2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_add_co_i32 s15, s12, 2
	s_cbranch_vccnz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_62 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s16, s15
	s_ashr_i32 s15, s15, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s17, s16, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s17, s17, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s16, s17
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s17, s16, s3
	s_cmp_ge_u32 s16, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s16, s17, s16
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s17, s16, s3
	s_cmp_ge_u32 s16, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s16, s17, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s16, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s16, s15
.LBB0_68:                               ;   in Loop: Header=BB0_62 Depth=2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_add_co_i32 s16, s12, 3
	s_cbranch_vccnz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_62 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s17, s16
	s_ashr_i32 s16, s16, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s18, s17, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s18, s18, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s17, s17, s18
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s18, s17, s3
	s_cmp_ge_u32 s17, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s17, s18, s17
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s18, s17, s3
	s_cmp_ge_u32 s17, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s17, s18, s17
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s17, s17, s16
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s17, s16
.LBB0_70:                               ;   in Loop: Header=BB0_62 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[17:18], null, s13, s4, v[6:7]
	v_mad_co_i64_i32 v[19:20], null, s14, s4, v[6:7]
	v_mad_co_i64_i32 v[21:22], null, s15, s4, v[6:7]
	v_mad_co_i64_i32 v[27:28], null, s16, s4, v[6:7]
	s_add_co_i32 s12, s12, 4
	v_add_co_u32 v23, vcc_lo, v17, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, 0, v18, vcc_lo
	v_add_co_u32 v25, vcc_lo, v19, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, 0, v20, vcc_lo
	s_clause 0x1
	global_load_b32 v29, v[23:24], off offset:2
	global_load_b32 v30, v[25:26], off offset:2
	v_add_co_u32 v23, vcc_lo, v21, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, 0, v22, vcc_lo
	v_add_co_u32 v25, vcc_lo, v27, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, 0, v28, vcc_lo
	s_clause 0x5
	global_load_b32 v23, v[23:24], off offset:2
	global_load_b32 v24, v[25:26], off offset:2
	global_load_d16_b16 v17, v[17:18], off
	global_load_d16_b16 v18, v[19:20], off
	global_load_d16_b16 v19, v[21:22], off
	global_load_d16_b16 v20, v[27:28], off
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt 0x7
	v_bfe_i32 v21, v29, 0, 8
	v_bfe_i32 v22, v29, 8, 8
	v_bfe_i32 v25, v29, 16, 8
	v_ashrrev_i32_e32 v26, 24, v29
	s_wait_loadcnt 0x6
	v_bfe_i32 v27, v30, 0, 8
	v_bfe_i32 v28, v30, 8, 8
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_bfe_i32 v29, v30, 16, 8
	v_ashrrev_i32_e32 v30, 24, v30
	s_wait_loadcnt 0x5
	v_bfe_i32 v31, v23, 0, 8
	v_bfe_i32 v32, v23, 8, 8
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	s_wait_loadcnt 0x3
	v_fma_mix_f32 v21, v17, v21, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v22, v17, v22, neg(0) op_sel_hi:[1,0,0]
	v_bfe_i32 v33, v23, 16, 8
	v_ashrrev_i32_e32 v23, 24, v23
	v_bfe_i32 v35, v24, 8, 8
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_fma_mix_f32 v25, v17, v25, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v17, v17, v26, neg(0) op_sel_hi:[1,0,0]
	s_wait_loadcnt 0x2
	v_fma_mix_f32 v26, v18, v27, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v27, v18, v28, neg(0) op_sel_hi:[1,0,0]
	s_wait_dscnt 0x1
	v_dual_fmac_f32 v0, v10, v21 :: v_dual_fmac_f32 v1, v10, v22
	v_bfe_i32 v34, v24, 0, 8
	v_bfe_i32 v36, v24, 16, 8
	v_ashrrev_i32_e32 v24, 24, v24
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v23, v23
	v_cvt_f32_i32_e32 v35, v35
	v_fma_mix_f32 v28, v18, v29, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v18, v18, v30, neg(0) op_sel_hi:[1,0,0]
	s_wait_loadcnt 0x1
	v_fma_mix_f32 v29, v19, v31, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v30, v19, v32, neg(0) op_sel_hi:[1,0,0]
	v_dual_fmac_f32 v2, v10, v25 :: v_dual_fmac_f32 v1, v11, v27
	v_dual_fmac_f32 v3, v10, v17 :: v_dual_fmac_f32 v0, v11, v26
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v24, v24
	v_fma_mix_f32 v31, v19, v33, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v19, v19, v23, neg(0) op_sel_hi:[1,0,0]
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v32, v20, v35, neg(0) op_sel_hi:[1,0,0]
	v_dual_fmac_f32 v2, v11, v28 :: v_dual_fmac_f32 v3, v11, v18
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v0, v8, v29 :: v_dual_fmac_f32 v1, v8, v30
	v_cvt_f32_i32_e32 v34, v34
	v_fma_mix_f32 v10, v20, v36, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v11, v20, v24, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v2, v8, v31 :: v_dual_fmac_f32 v1, v9, v32
	v_fmac_f32_e32 v3, v8, v19
	v_fma_mix_f32 v23, v20, v34, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v2, v9, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v10, s11 :: v_dual_fmac_f32 v3, v9, v11
	v_fmac_f32_e32 v0, v9, v23
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[8:9], v10 offset0:6 offset1:7
	ds_load_2addr_b32 v[10:11], v10 offset0:4 offset1:5
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s11, s12
	s_cbranch_vccnz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_62 Depth=2
	s_abs_i32 s11, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s13, s11, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s11, s11, s13
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s14, s11, s3
	s_cmp_ge_u32 s11, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s11, s14, s11
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s14, s11, s3
	s_cmp_ge_u32 s11, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s11, s14, s11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s11, s11, s13
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s11, s11, s13
.LBB0_72:                               ;   in Loop: Header=BB0_62 Depth=2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_add_co_i32 s13, s12, 1
	s_cbranch_vccnz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_62 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s14, s13
	s_ashr_i32 s13, s13, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s15, s14, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s15, s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s14, s14, s15
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s14, s3
	s_cmp_ge_u32 s14, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s14, s15, s14
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s14, s3
	s_cmp_ge_u32 s14, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s14, s15, s14
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s14, s14, s13
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s13, s14, s13
.LBB0_74:                               ;   in Loop: Header=BB0_62 Depth=2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_add_co_i32 s14, s12, 2
	s_cbranch_vccnz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_62 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s15, s14
	s_ashr_i32 s14, s14, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s16, s15, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s16, s16, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s15, s16
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s15, s3
	s_cmp_ge_u32 s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s15, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s15, s3
	s_cmp_ge_u32 s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s15, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s15, s15, s14
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s14, s15, s14
.LBB0_76:                               ;   in Loop: Header=BB0_62 Depth=2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_add_co_i32 s12, s12, 3
	s_cbranch_vccnz .LBB0_61
; %bb.77:                               ;   in Loop: Header=BB0_62 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s15, s12
	s_ashr_i32 s12, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s16, s15, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s16, s16, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s15, s16
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s15, s3
	s_cmp_ge_u32 s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s15, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s15, s3
	s_cmp_ge_u32 s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s15, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s15, s15, s12
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s12, s15, s12
	s_branch .LBB0_61
.LBB0_78:                               ;   in Loop: Header=BB0_59 Depth=1
	s_or_b32 s10, s11, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s10, s25
	s_cbranch_scc1 .LBB0_89
; %bb.79:                               ;   in Loop: Header=BB0_59 Depth=1
	v_add_co_u32 v6, vcc_lo, s8, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s9, v16, vcc_lo
	s_branch .LBB0_81
.LBB0_80:                               ;   in Loop: Header=BB0_81 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[17:18], null, s10, s4, v[6:7]
	v_mad_co_i64_i32 v[19:20], null, s13, s4, v[6:7]
	v_mad_co_i64_i32 v[21:22], null, s14, s4, v[6:7]
	v_mad_co_i64_i32 v[23:24], null, s12, s4, v[6:7]
	s_add_co_i32 s12, s11, 7
	s_add_co_i32 s10, s11, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s25
	s_mov_b32 s11, s10
	v_add_co_u32 v25, vcc_lo, v17, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, 0, v18, vcc_lo
	v_add_co_u32 v27, vcc_lo, v19, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, 0, v20, vcc_lo
	global_load_b32 v31, v[25:26], off offset:2
	v_add_co_u32 v29, vcc_lo, v21, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v22, vcc_lo
	v_add_co_u32 v25, vcc_lo, v23, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, 0, v24, vcc_lo
	s_clause 0x6
	global_load_b32 v27, v[27:28], off offset:2
	global_load_b32 v28, v[29:30], off offset:2
	global_load_b32 v25, v[25:26], off offset:2
	global_load_d16_b16 v17, v[17:18], off
	global_load_d16_b16 v18, v[19:20], off
	global_load_d16_b16 v19, v[21:22], off
	global_load_d16_b16 v20, v[23:24], off
	s_wait_loadcnt 0x7
	v_bfe_i32 v21, v31, 0, 8
	v_bfe_i32 v22, v31, 8, 8
	v_bfe_i32 v33, v31, 16, 8
	v_ashrrev_i32_e32 v31, 24, v31
	s_wait_loadcnt 0x6
	v_bfe_i32 v24, v27, 0, 8
	v_cvt_f32_i32_e32 v21, v21
	v_bfe_i32 v23, v27, 8, 8
	s_wait_loadcnt 0x5
	v_bfe_i32 v26, v28, 0, 8
	v_ashrrev_i32_e32 v34, 24, v27
	v_bfe_i32 v27, v27, 16, 8
	v_cvt_f32_i32_e32 v22, v22
	v_cvt_f32_i32_e32 v24, v24
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v31, v31
	s_wait_loadcnt 0x3
	v_fma_mix_f32 v21, v17, v21, neg(0) op_sel_hi:[1,0,0]
	v_bfe_i32 v29, v28, 8, 8
	v_bfe_i32 v32, v25, 0, 8
	v_bfe_i32 v35, v28, 16, 8
	v_ashrrev_i32_e32 v28, 24, v28
	v_cvt_f32_i32_e32 v23, v23
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v27, v27
	v_fma_mix_f32 v22, v17, v22, neg(0) op_sel_hi:[1,0,0]
	s_wait_loadcnt 0x2
	v_fma_mix_f32 v24, v18, v24, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v31, v17, v31, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v17, v17, v33, neg(0) op_sel_hi:[1,0,0]
	s_wait_dscnt 0x1
	v_fma_f32 v0, v10, v21, v0
	v_bfe_i32 v30, v25, 8, 8
	v_ashrrev_i32_e32 v36, 24, v25
	v_bfe_i32 v25, v25, 16, 8
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v28, v28
	v_fma_mix_f32 v23, v18, v23, neg(0) op_sel_hi:[1,0,0]
	s_wait_loadcnt 0x1
	v_fma_mix_f32 v26, v19, v26, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v27, v18, v27, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v18, v18, v34, neg(0) op_sel_hi:[1,0,0]
	v_dual_fmac_f32 v1, v10, v22 :: v_dual_fmac_f32 v0, v11, v24
	v_fmac_f32_e32 v3, v10, v31
	v_fma_f32 v2, v10, v17, v2
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v25, v25
	v_fma_mix_f32 v29, v19, v29, neg(0) op_sel_hi:[1,0,0]
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v32, v20, v32, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v28, v19, v28, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v19, v19, v35, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v2, v11, v27
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v0, v8, v26 :: v_dual_fmac_f32 v1, v11, v23
	v_fmac_f32_e32 v3, v11, v18
	v_fma_mix_f32 v30, v20, v30, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v10, v20, v25, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v11, v20, v36, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v2, v8, v19 :: v_dual_fmac_f32 v3, v8, v28
	v_dual_fmac_f32 v0, v9, v32 :: v_dual_fmac_f32 v1, v8, v29
	v_dual_fmac_f32 v2, v9, v10 :: v_dual_fmac_f32 v3, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v1, v9, v30
	s_cbranch_scc1 .LBB0_90
.LBB0_81:                               ;   Parent Loop BB0_59 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl2_add_u32 s10, s11, 0
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s10
	s_add_co_i32 s12, s11, s24
	ds_load_2addr_b32 v[10:11], v8 offset1:1
	ds_load_2addr_b32 v[8:9], v8 offset0:2 offset1:3
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s10, s12
	s_cbranch_vccnz .LBB0_83
; %bb.82:                               ;   in Loop: Header=BB0_81 Depth=2
	s_abs_i32 s10, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s13, s10, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s10, s10, s13
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s14, s10, s3
	s_cmp_ge_u32 s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s10, s14, s10
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s14, s10, s3
	s_cmp_ge_u32 s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s10, s14, s10
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, s10, s13
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s10, s10, s13
.LBB0_83:                               ;   in Loop: Header=BB0_81 Depth=2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_add_co_i32 s13, s12, 1
	s_cbranch_vccnz .LBB0_85
; %bb.84:                               ;   in Loop: Header=BB0_81 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s14, s13
	s_ashr_i32 s13, s13, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s15, s14, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s15, s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s14, s14, s15
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s14, s3
	s_cmp_ge_u32 s14, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s14, s15, s14
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s14, s3
	s_cmp_ge_u32 s14, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s14, s15, s14
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s14, s14, s13
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s13, s14, s13
.LBB0_85:                               ;   in Loop: Header=BB0_81 Depth=2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_add_co_i32 s14, s12, 2
	s_cbranch_vccnz .LBB0_87
; %bb.86:                               ;   in Loop: Header=BB0_81 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s15, s14
	s_ashr_i32 s14, s14, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s16, s15, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s16, s16, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s15, s16
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s15, s3
	s_cmp_ge_u32 s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s15, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s15, s3
	s_cmp_ge_u32 s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s15, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s15, s15, s14
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s14, s15, s14
.LBB0_87:                               ;   in Loop: Header=BB0_81 Depth=2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_add_co_i32 s12, s12, 3
	s_cbranch_vccnz .LBB0_80
; %bb.88:                               ;   in Loop: Header=BB0_81 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s15, s12
	s_ashr_i32 s12, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s16, s15, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s16, s16, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s15, s16
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s15, s3
	s_cmp_ge_u32 s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s15, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s16, s15, s3
	s_cmp_ge_u32 s15, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s15, s16, s15
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s15, s15, s12
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s12, s15, s12
	s_branch .LBB0_80
.LBB0_89:                               ;   in Loop: Header=BB0_59 Depth=1
	s_mov_b32 s10, s11
.LBB0_90:                               ;   in Loop: Header=BB0_59 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s10, s25
	s_cbranch_scc1 .LBB0_58
; %bb.91:                               ;   in Loop: Header=BB0_59 Depth=1
	v_add_co_u32 v6, vcc_lo, s8, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s9, v16, vcc_lo
	s_branch .LBB0_93
.LBB0_92:                               ;   in Loop: Header=BB0_93 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[9:10], null, s11, s4, v[6:7]
	s_add_co_i32 s10, s10, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s10, s25
	v_add_co_u32 v15, vcc_lo, v9, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, 0, v10, vcc_lo
	s_clause 0x1
	global_load_b32 v11, v[15:16], off offset:2
	global_load_d16_b16 v9, v[9:10], off
	s_wait_loadcnt 0x1
	v_bfe_i32 v10, v11, 0, 8
	v_bfe_i32 v15, v11, 8, 8
	v_bfe_i32 v16, v11, 16, 8
	v_ashrrev_i32_e32 v11, 24, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_i32_e32 v16, v16
	v_cvt_f32_i32_e32 v11, v11
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v10, v9, v10, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v15, v9, v15, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v16, v9, v16, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v9, v9, v11, neg(0) op_sel_hi:[1,0,0]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v0, v8, v10 :: v_dual_fmac_f32 v1, v8, v15
	v_dual_fmac_f32 v2, v8, v16 :: v_dual_fmac_f32 v3, v8, v9
	s_cbranch_scc0 .LBB0_58
.LBB0_93:                               ;   Parent Loop BB0_59 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_lshl2_add_u32 s11, s10, 0
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s11
	s_add_co_i32 s11, s10, s24
	ds_load_b32 v8, v8
	s_cbranch_vccnz .LBB0_92
; %bb.94:                               ;   in Loop: Header=BB0_93 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s12, s11
	s_ashr_i32 s11, s11, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s13, s12, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s12, s12, s13
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s13, s12, s3
	s_cmp_ge_u32 s12, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s12, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s13, s12, s3
	s_cmp_ge_u32 s12, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s12, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s12, s12, s11
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s11, s12, s11
	s_branch .LBB0_92
.LBB0_95:
	s_endpgm
.Lfunc_end0:
	.size	attention_flash_q8_0_tile, .Lfunc_end0-attention_flash_q8_0_tile
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_flash_q8_0_tile
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 72
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
		.amdhsa_next_free_vgpr 43
		.amdhsa_next_free_sgpr 36
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-attention_flash_q8_0_tile)<<4)&4080)>>4
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
	.set .Lattention_flash_q8_0_tile.num_vgpr, 43
	.set .Lattention_flash_q8_0_tile.num_agpr, 0
	.set .Lattention_flash_q8_0_tile.numbered_sgpr, 36
	.set .Lattention_flash_q8_0_tile.num_named_barrier, 0
	.set .Lattention_flash_q8_0_tile.private_seg_size, 0
	.set .Lattention_flash_q8_0_tile.uses_vcc, 1
	.set .Lattention_flash_q8_0_tile.uses_flat_scratch, 0
	.set .Lattention_flash_q8_0_tile.has_dyn_sized_stack, 0
	.set .Lattention_flash_q8_0_tile.has_recursion, 0
	.set .Lattention_flash_q8_0_tile.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8660
; TotalNumSgprs: 38
; NumVgprs: 43
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 38
; NumVGPRsForWavesPerEU: 43
; Occupancy: 16
; WaveLimiterHint : 0
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
	.type	__hip_cuid_5a8209df69b32f98,@object ; @__hip_cuid_5a8209df69b32f98
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5a8209df69b32f98
__hip_cuid_5a8209df69b32f98:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5a8209df69b32f98, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5a8209df69b32f98
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
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           attention_flash_q8_0_tile
    .private_segment_fixed_size: 0
    .sgpr_count:     38
    .sgpr_spill_count: 0
    .symbol:         attention_flash_q8_0_tile.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     43
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
