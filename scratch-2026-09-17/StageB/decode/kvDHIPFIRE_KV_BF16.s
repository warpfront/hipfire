	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	attention_q8_0_kv       ; -- Begin function attention_q8_0_kv
	.globl	attention_q8_0_kv
	.p2align	8
	.type	attention_q8_0_kv,@function
attention_q8_0_kv:                      ; @attention_q8_0_kv
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[28:30], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 ttmp9, s28
	s_cbranch_scc1 .LBB0_64
; %bb.1:
	s_abs_i32 s2, s29
	s_abs_i32 s5, s28
	s_cvt_f32_u32 s3, s2
	s_sub_co_i32 s4, 0, s2
	v_lshlrev_b32_e32 v20, 2, v0
	s_mov_b32 s37, 0
	v_s_rcp_f32 s3, s3
	s_mul_i32 s34, s30, ttmp9
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s3, s3
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s4, s4, s3
	s_mul_hi_u32 s4, s3, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s3, s3, s4
	s_xor_b32 s4, s28, s29
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s3, s5, s3
	s_ashr_i32 s7, s4, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s6, s3, s2
	s_add_co_i32 s8, s3, 1
	s_sub_co_i32 s6, s5, s6
	s_load_b64 s[4:5], s[0:1], 0x20
	s_sub_co_i32 s9, s6, s2
	s_cmp_ge_u32 s6, s2
	s_cselect_b32 s3, s8, s3
	s_cselect_b32 s6, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s3, 1
	s_cmp_ge_u32 s6, s2
	s_load_b32 s6, s[0:1], 0x4c
	s_cselect_b32 s2, s8, s3
	s_abs_i32 s36, ttmp9
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, s2, s7
	s_ashr_i32 s35, s34, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s3, s2, s7
	s_load_b256 s[20:27], s[0:1], 0x0
	s_wait_kmcnt 0x0
	s_load_b32 s31, s[4:5], 0x0
	s_abs_i32 s33, s3
	s_mov_b32 s5, s37
	s_cvt_f32_u32 s2, s33
	s_sub_co_i32 s4, 0, s33
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s2, s2
	s_and_b32 s28, s6, 0xffff
	s_mov_b32 s6, exec_lo
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s2, s2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s4, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s4, s2, s4
	s_add_co_i32 s4, s2, s4
	v_cmpx_gt_i32_e64 s30, v0
	s_cbranch_execz .LBB0_4
; %bb.2:
	s_lshl_b64 s[8:9], s[34:35], 2
	v_dual_mov_b32 v4, v0 :: v_dual_add_nc_u32 v3, 0, v20
	s_add_nc_u64 s[8:9], s[20:21], s[8:9]
	s_lshl_b32 s7, s28, 2
	v_add_co_u32 v1, s2, s8, v20
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s9, 0, s2
	s_mov_b32 s8, s37
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	global_load_b32 v5, v[1:2], off
	v_add_nc_u32_e32 v4, s28, v4
	v_add_co_u32 v1, vcc_lo, v1, s7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_i32_e64 s2, s30, v4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s8, s2, s8
	s_wait_loadcnt 0x0
	ds_store_b32 v3, v5 offset:8192
	v_add_nc_u32_e32 v3, s7, v3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_3
.LBB0_4:
	s_or_b32 exec_lo, exec_lo, s6
	s_mul_u64 s[20:21], s[36:37], s[4:5]
	s_ashr_i32 s20, s3, 31
	s_cvt_f32_u32 s3, s28
	s_sub_co_i32 s13, 0, s28
	s_add_co_i32 s5, s30, s28
	s_ashr_i32 s2, ttmp9, 31
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s3, s3
	s_add_co_i32 s16, s5, -1
	s_wait_dscnt 0x0
	s_abs_i32 s17, s16
	s_ashr_i32 s37, s16, 31
	s_barrier_signal -1
	s_mov_b32 s4, 0
	s_delay_alu instid0(TRANS32_DEP_1)
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_mov_b32 s5, s4
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s3, s3
	s_mov_b32 s8, s4
	s_mov_b32 s9, s4
	s_mov_b32 s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s14, s13, s3
	s_mov_b32 s11, s4
	s_mul_hi_u32 s14, s3, s14
	s_mov_b32 s12, s4
	s_add_co_i32 s3, s3, s14
	s_mov_b32 s13, s4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s3, s17, s3
	s_mov_b32 s14, s4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s18, s3, s28
	s_mov_b32 s15, s4
	s_sub_co_i32 s17, s17, s18
	s_add_co_i32 s18, s3, 1
	s_sub_co_i32 s19, s17, s28
	s_cmp_ge_u32 s17, s28
	s_mov_b32 s16, s4
	s_cselect_b32 s3, s18, s3
	s_cselect_b32 s17, s19, s17
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s18, s3, 1
	s_cmp_ge_u32 s17, s28
	s_mov_b32 s17, s4
	s_cselect_b32 s3, s18, s3
	s_mov_b32 s18, s4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s3, s37
	s_mov_b32 s19, s4
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s3, s3, s37
	v_dual_mov_b32 v1, s4 :: v_dual_mov_b32 v2, s5
	v_dual_mov_b32 v3, s6 :: v_dual_mov_b32 v4, s7
	v_dual_mov_b32 v5, s8 :: v_dual_mov_b32 v6, s9
	v_dual_mov_b32 v7, s10 :: v_dual_mov_b32 v8, s11
	v_dual_mov_b32 v9, s12 :: v_dual_mov_b32 v10, s13
	v_dual_mov_b32 v11, s14 :: v_dual_mov_b32 v12, s15
	v_dual_mov_b32 v13, s16 :: v_dual_mov_b32 v14, s17
	v_dual_mov_b32 v15, s18 :: v_dual_mov_b32 v16, s19
	s_wait_alu depctr_sa_sdst(0)
	s_min_i32 s8, s3, 16
	s_cmp_gt_i32 s3, 0
	s_barrier_wait -1
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s3, 1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB0_7
; %bb.5:
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v15, 0
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v13, 0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 m0, s4
	s_add_co_i32 s4, s4, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, s4
	s_cbranch_scc0 .LBB0_6
.LBB0_7:
	v_mov_b32_e32 v19, 0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s31, 0
	s_mov_b32 s3, 0
	s_cbranch_scc1 .LBB0_59
; %bb.8:
	s_mul_i32 s4, s21, s33
	s_xor_b32 s2, s2, s20
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s4, s36, s4
	s_add_co_i32 s5, s21, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s6, s4, s33
	s_cmp_ge_u32 s4, s33
	s_load_b32 s13, s[0:1], 0x38
	s_cselect_b32 s5, s5, s21
	s_cselect_b32 s4, s6, s4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s5, 1
	s_cmp_ge_u32 s4, s33
	s_mov_b32 s21, 0xf149f2ca
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s4, s6, s5
	s_ashr_i32 s5, s30, 31
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, s4, s2
	s_lshr_b32 s5, s5, 27
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s15, s4, s2
	s_add_co_i32 s5, s30, s5
	s_lshl2_add_u32 s10, s30, 0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s12, s5, 5
	s_add_co_i32 s11, s31, 1
	s_cmp_gt_i32 s30, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s4, s12, s29
	s_mul_i32 s15, s15, s12
	s_cselect_b32 s14, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 31
	s_mul_i32 s0, s15, 34
	s_lshr_b32 s16, s28, 1
	v_dual_mov_b32 v24, 0 :: v_dual_add_nc_u32 v21, s10, v20
	v_dual_mov_b32 v23, s10 :: v_dual_add_nc_u32 v22, 0, v20
	s_cselect_b32 s17, -1, 0
	s_ashr_i32 s1, s0, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[4:5], s[4:5], 34
	s_add_nc_u64 s[6:7], s[24:25], s[0:1]
	s_lshl_b32 s18, s28, 2
	s_mov_b32 s19, s11
	s_mov_b32 s20, 0
	s_branch .LBB0_11
.LBB0_9:                                ;   in Loop: Header=BB0_11 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_10:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_fmac_f32_e32 v19, v24, v17
	s_addk_co_i32 s20, 0x800
	s_addk_co_i32 s19, 0xf800
	s_mov_b32 s21, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s20, s31
	v_mov_b32_e32 v24, v19
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB0_59
.LBB0_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_14 Depth 2
                                        ;       Child Loop BB0_16 Depth 3
                                        ;     Child Loop BB0_19 Depth 2
                                        ;     Child Loop BB0_34 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_27 Depth 2
                                        ;     Child Loop BB0_37 Depth 2
                                        ;     Child Loop BB0_41 Depth 2
                                        ;       Child Loop BB0_44 Depth 3
                                        ;       Child Loop BB0_50 Depth 3
                                        ;     Child Loop BB0_57 Depth 2
	s_sub_co_i32 s1, s11, s20
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_gt_i32_e64 s0, s1, v0
	s_min_i32 s2, s1, 0x800
	s_and_saveexec_b32 s24, s0
	s_cbranch_execz .LBB0_17
; %bb.12:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mov_b32_e32 v25, v0
	s_mov_b32 s25, 0
	s_branch .LBB0_14
.LBB0_13:                               ;   in Loop: Header=BB0_14 Depth=2
	v_lshl_add_u32 v17, v25, 2, 0
	s_wait_kmcnt 0x0
	v_dual_mul_f32 v18, s13, v19 :: v_dual_add_nc_u32 v25, s28, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_i32_e32 vcc_lo, s2, v25
	ds_store_b32 v17, v18
	s_or_b32 s25, vcc_lo, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s25
	s_cbranch_execz .LBB0_17
.LBB0_14:                               ;   Parent Loop BB0_11 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_16 Depth 3
	v_mov_b32_e32 v19, 0
	s_and_not1_b32 vcc_lo, exec_lo, s14
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_13
; %bb.15:                               ;   in Loop: Header=BB0_14 Depth=2
	v_add_nc_u32_e32 v19, s20, v25
	s_mov_b32 s29, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[17:18], null, s4, v19, s[22:23]
	v_mad_co_u64_u32 v[18:19], null, s5, v19, v[18:19]
	v_mov_b32_e32 v19, 0
.LBB0_16:                               ;   Parent Loop BB0_11 Depth=1
                                        ;     Parent Loop BB0_14 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s33, s29, s15
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s33, s33, 34
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s36, s33, 31
	v_add_co_u32 v30, vcc_lo, v17, s33
	s_wait_alu depctr_sa_sdst(0) depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s36, v18, vcc_lo
	s_lshl_b32 s33, s29, 7
	s_add_co_i32 s29, s29, 1
	s_clause 0x2
	global_load_b128 v[26:29], v[30:31], off offset:2
	global_load_d16_b16 v42, v[30:31], off
	global_load_b128 v[30:33], v[30:31], off offset:18
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s36, s33, 0x2000
	s_add_co_i32 s38, s33, 0x2010
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v34, s36 :: v_dual_mov_b32 v39, s38
	s_add_co_i32 s37, s33, 0x2008
	s_add_co_i32 s39, s33, 0x2018
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v36, s37 :: v_dual_mov_b32 v43, s39
	ds_load_2addr_b32 v[34:35], v34 offset1:1
	s_add_co_i32 s40, s33, 0x2020
	s_add_co_i32 s41, s33, 0x2028
	ds_load_2addr_b32 v[36:37], v36 offset1:1
	s_add_co_i32 s42, s33, 0x2030
	s_add_co_i32 s43, s33, 0x2038
	s_add_co_i32 s44, s33, 0x2040
	s_add_co_i32 s45, s33, 0x2050
	s_add_co_i32 s46, s33, 0x2058
	s_add_co_i32 s47, s33, 0x2060
	s_add_co_i32 s49, s33, 0x2070
	s_add_co_i32 s36, s33, 0x2048
	s_add_co_i32 s48, s33, 0x2068
	s_addk_co_i32 s33, 0x2078
	s_cmp_eq_u32 s29, s12
	s_wait_loadcnt 0x2
	v_bfe_i32 v38, v26, 0, 8
	v_bfe_i32 v40, v26, 8, 8
	v_bfe_i32 v44, v26, 16, 8
	v_ashrrev_i32_e32 v26, 24, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_i32_e32 v41, v38
	v_cvt_f32_i32_e32 v45, v40
	ds_load_2addr_b32 v[38:39], v39 offset1:1
	v_cvt_f32_i32_e32 v26, v26
	s_wait_loadcnt 0x1
	v_fma_mix_f32 v46, v42, v41, neg(0) op_sel_hi:[1,0,0]
	ds_load_2addr_b32 v[40:41], v43 offset1:1
	v_cvt_f32_i32_e32 v43, v44
	v_fma_mix_f32 v44, v42, v45, neg(0) op_sel_hi:[1,0,0]
	v_bfe_i32 v45, v27, 0, 8
	s_wait_dscnt 0x3
	s_wait_alu depctr_sa_sdst(0)
	v_dual_fmac_f32 v19, v34, v46 :: v_dual_mov_b32 v34, s40
	v_fma_mix_f32 v43, v42, v43, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v26, v42, v26, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v19, v35, v44 :: v_dual_mov_b32 v44, s41
	v_bfe_i32 v35, v27, 8, 8
	v_fma_mix_f32 v45, v42, v45, neg(0) op_sel_hi:[1,0,0]
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v19, v36, v43 :: v_dual_mov_b32 v36, s42
	v_bfe_i32 v43, v27, 16, 8
	v_cvt_f32_i32_e32 v46, v35
	ds_load_2addr_b32 v[34:35], v34 offset1:1
	v_fmac_f32_e32 v19, v37, v26
	v_ashrrev_i32_e32 v26, 24, v27
	v_cvt_f32_i32_e32 v27, v43
	v_fma_mix_f32 v37, v42, v46, neg(0) op_sel_hi:[1,0,0]
	v_bfe_i32 v43, v28, 0, 8
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v19, v38, v45
	v_cvt_f32_i32_e32 v45, v26
	v_fma_mix_f32 v46, v42, v27, neg(0) op_sel_hi:[1,0,0]
	ds_load_2addr_b32 v[26:27], v44 offset1:1
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v19, v39, v37
	v_bfe_i32 v39, v28, 8, 8
	ds_load_2addr_b32 v[36:37], v36 offset1:1
	v_fma_mix_f32 v44, v42, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v43, v42, v43, neg(0) op_sel_hi:[1,0,0]
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v19, v40, v46
	v_bfe_i32 v40, v28, 16, 8
	v_cvt_f32_i32_e32 v45, v39
	v_ashrrev_i32_e32 v28, 24, v28
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v38, s43 :: v_dual_fmac_f32 v19, v41, v44
	v_cvt_f32_i32_e32 v40, v40
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_mix_f32 v41, v42, v45, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v28, v28
	ds_load_2addr_b32 v[38:39], v38 offset1:1
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v19, v34, v43
	v_bfe_i32 v43, v29, 0, 8
	v_fma_mix_f32 v40, v42, v40, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v28, v42, v28, neg(0) op_sel_hi:[1,0,0]
	v_bfe_i32 v44, v29, 16, 8
	v_fmac_f32_e32 v19, v35, v41
	v_bfe_i32 v41, v29, 8, 8
	v_cvt_f32_i32_e32 v43, v43
	v_dual_mov_b32 v34, s44 :: v_dual_mov_b32 v35, s36
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v19, v26, v40
	v_cvt_f32_i32_e32 v41, v41
	v_fma_mix_f32 v43, v42, v43, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v40, s45 :: v_dual_fmac_f32 v19, v27, v28
	v_ashrrev_i32_e32 v28, 24, v29
	v_cvt_f32_i32_e32 v29, v44
	ds_load_2addr_b32 v[26:27], v34 offset1:1
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v19, v36, v43
	v_cvt_f32_i32_e32 v43, v28
	v_fma_mix_f32 v44, v42, v29, neg(0) op_sel_hi:[1,0,0]
	ds_load_2addr_b32 v[28:29], v35 offset1:1
	v_fma_mix_f32 v34, v42, v41, neg(0) op_sel_hi:[1,0,0]
	s_wait_loadcnt 0x0
	v_bfe_i32 v41, v30, 0, 8
	v_fma_mix_f32 v43, v42, v43, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v36, s46 :: v_dual_fmac_f32 v19, v37, v34
	v_bfe_i32 v37, v30, 8, 8
	v_cvt_f32_i32_e32 v41, v41
	ds_load_2addr_b32 v[34:35], v40 offset1:1
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v19, v38, v44
	v_bfe_i32 v38, v30, 16, 8
	v_cvt_f32_i32_e32 v40, v37
	v_fma_mix_f32 v41, v42, v41, neg(0) op_sel_hi:[1,0,0]
	v_ashrrev_i32_e32 v30, 24, v30
	v_fmac_f32_e32 v19, v39, v43
	v_cvt_f32_i32_e32 v38, v38
	v_fma_mix_f32 v39, v42, v40, neg(0) op_sel_hi:[1,0,0]
	v_bfe_i32 v40, v31, 0, 8
	v_cvt_f32_i32_e32 v30, v30
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v19, v26, v41
	v_fma_mix_f32 v38, v42, v38, neg(0) op_sel_hi:[1,0,0]
	ds_load_2addr_b32 v[36:37], v36 offset1:1
	v_cvt_f32_i32_e32 v40, v40
	v_fma_mix_f32 v30, v42, v30, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v19, v27, v39
	v_bfe_i32 v27, v31, 8, 8
	v_mov_b32_e32 v26, s47
	v_fma_mix_f32 v40, v42, v40, neg(0) op_sel_hi:[1,0,0]
	v_mov_b32_e32 v39, s48
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v19, v28, v38
	v_bfe_i32 v28, v31, 16, 8
	v_cvt_f32_i32_e32 v41, v27
	ds_load_2addr_b32 v[26:27], v26 offset1:1
	v_dual_mov_b32 v38, s49 :: v_dual_fmac_f32 v19, v29, v30
	v_ashrrev_i32_e32 v29, 24, v31
	v_cvt_f32_i32_e32 v28, v28
	v_fma_mix_f32 v30, v42, v41, neg(0) op_sel_hi:[1,0,0]
	v_bfe_i32 v31, v32, 0, 8
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v19, v34, v40 :: v_dual_mov_b32 v34, s33
	v_cvt_f32_i32_e32 v40, v29
	v_fma_mix_f32 v41, v42, v28, neg(0) op_sel_hi:[1,0,0]
	ds_load_2addr_b32 v[28:29], v39 offset1:1
	v_fmac_f32_e32 v19, v35, v30
	v_bfe_i32 v35, v32, 8, 8
	v_cvt_f32_i32_e32 v39, v31
	v_fma_mix_f32 v40, v42, v40, neg(0) op_sel_hi:[1,0,0]
	ds_load_2addr_b32 v[30:31], v38 offset1:1
	v_cvt_f32_i32_e32 v38, v35
	ds_load_2addr_b32 v[34:35], v34 offset1:1
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v19, v36, v41
	v_bfe_i32 v36, v32, 16, 8
	v_fma_mix_f32 v39, v42, v39, neg(0) op_sel_hi:[1,0,0]
	v_ashrrev_i32_e32 v32, 24, v32
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v19, v37, v40
	v_cvt_f32_i32_e32 v36, v36
	v_fma_mix_f32 v37, v42, v38, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v19, v26, v39
	v_bfe_i32 v26, v33, 0, 8
	v_fma_mix_f32 v36, v42, v36, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v32, v42, v32, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v19, v27, v37
	v_bfe_i32 v27, v33, 8, 8
	v_cvt_f32_i32_e32 v26, v26
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v19, v28, v36
	v_bfe_i32 v28, v33, 16, 8
	v_cvt_f32_i32_e32 v27, v27
	v_fma_mix_f32 v26, v42, v26, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v19, v29, v32
	v_ashrrev_i32_e32 v29, 24, v33
	v_cvt_f32_i32_e32 v28, v28
	v_fma_mix_f32 v27, v42, v27, neg(0) op_sel_hi:[1,0,0]
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v19, v30, v26
	v_cvt_f32_i32_e32 v26, v29
	v_fma_mix_f32 v28, v42, v28, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v19, v31, v27
	v_fma_mix_f32 v26, v42, v26, neg(0) op_sel_hi:[1,0,0]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v34, v28
	v_fmac_f32_e32 v19, v35, v26
	s_cbranch_scc0 .LBB0_16
	s_branch .LBB0_13
.LBB0_17:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s24
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v17, 0xf149f2ca
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s24, s0
	s_cbranch_execz .LBB0_21
; %bb.18:                               ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v17, 0xf149f2ca :: v_dual_mov_b32 v18, v22
	v_mov_b32_e32 v19, v0
	s_mov_b32 s25, 0
.LBB0_19:                               ;   Parent Loop BB0_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_load_b32 v25, v18
	v_add_nc_u32_e32 v19, s28, v19
	s_wait_alu depctr_sa_sdst(0)
	v_dual_max_num_f32 v17, v17, v17 :: v_dual_add_nc_u32 v18, s18, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e32 vcc_lo, s2, v19
	s_or_b32 s25, vcc_lo, s25
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v25, v25, v25
	v_max_num_f32_e32 v17, v17, v25
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s25
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_11 Depth=1
	s_or_b32 exec_lo, exec_lo, s25
.LBB0_21:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s24
	ds_store_b32 v21, v17 offset:8192
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s24, s16
	s_and_not1_b32 vcc_lo, exec_lo, s17
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_34
.LBB0_22:                               ;   in Loop: Header=BB0_11 Depth=1
	ds_load_b32 v17, v23 offset:8192
	v_max_num_f32_e64 v18, s21, s21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s24, v18
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v17, v17, v17
	v_readfirstlane_b32 s25, v17
	s_wait_alu depctr_sa_sdst(0)
	s_max_num_f32 s24, s24, s25
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_sub_f32 s21, s21, s24
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s25, s21, 0x3fb8aa3b
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2)
	s_xor_b32 s29, s25, 0x80000000
	s_rndne_f32 s33, s25
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s29, s21, 0x3fb8aa3b, s29
	s_cmp_nlt_f32 s21, 0xc2ce8ed0
	s_sub_f32 s25, s25, s33
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s29, s21, 0x32a5705f, s29
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s21, 0x42b17218
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s25, s25, s29
	s_cvt_i32_f32 s29, s33
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s25, s25
	s_wait_alu depctr_va_sdst(0)
	v_ldexp_f32 v17, s25, s29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v17, 0, v17, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v17, 0x7f800000, v17, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s9
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_11 Depth=1
	s_mov_b32 s21, 0
.LBB0_24:                               ;   Parent Loop BB0_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 m0, s21
	s_add_co_i32 s21, s21, 1
	v_movrels_b32_e32 v18, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, s21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v18, v17, v18
	v_movreld_b32_e32 v1, v18
	s_cbranch_scc0 .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mov_b32_e32 v18, 0
	s_and_saveexec_b32 s21, s0
	s_cbranch_execz .LBB0_29
; %bb.26:                               ;   in Loop: Header=BB0_11 Depth=1
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, v22
	v_mov_b32_e32 v25, v0
	s_mov_b32 s0, 0
.LBB0_27:                               ;   Parent Loop BB0_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_load_b32 v26, v19
	s_wait_dscnt 0x0
	v_dual_subrev_f32 v26, s24, v26 :: v_dual_add_nc_u32 v25, s28, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v27, 0x3fb8aa3b, v26
	v_fma_f32 v28, 0x3fb8aa3b, v26, -v27
	v_rndne_f32_e32 v29, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v27, v27, v29
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v26
	v_fmac_f32_e32 v28, 0x32a5705f, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v27, v27, v28
	v_cvt_i32_f32_e32 v28, v29
	v_exp_f32_e32 v27, v27
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v27, v27, v28
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v27, 0, v27, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 0x7f800000, v27, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, s2, v25
	ds_store_b32 v19, v26
	v_dual_add_f32 v18, v18, v26 :: v_dual_add_nc_u32 v19, s18, v19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_27
; %bb.28:                               ;   in Loop: Header=BB0_11 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
.LBB0_29:                               ;   in Loop: Header=BB0_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s21
	ds_store_b32 v21, v18 offset:8192
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, s16
	s_and_not1_b32 vcc_lo, exec_lo, s17
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_37
.LBB0_30:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mov_b32_e32 v18, s10
	s_cmp_lt_i32 s1, 1
	ds_load_b32 v19, v18 offset:8192
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB0_10
; %bb.31:                               ;   in Loop: Header=BB0_11 Depth=1
	s_cmp_lt_i32 s19, 2
	s_cbranch_scc1 .LBB0_52
; %bb.32:                               ;   in Loop: Header=BB0_11 Depth=1
	v_med3_i32 v18, s19, 1, 0x800
	s_mov_b32 s25, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s0, v18
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s21, s0, 0xffe
	s_branch .LBB0_41
.LBB0_33:                               ;   in Loop: Header=BB0_34 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s25
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_lshr_b32 s24, s24, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc0 .LBB0_22
.LBB0_34:                               ;   Parent Loop BB0_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s25, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	v_cmpx_gt_u32_e64 s24, v0
	s_cbranch_execz .LBB0_33
; %bb.35:                               ;   in Loop: Header=BB0_34 Depth=2
	v_lshl_add_u32 v17, s24, 2, v21
	ds_load_b32 v17, v17 offset:8192
	ds_load_b32 v18, v21 offset:8192
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v17, v17, v17 :: v_dual_max_num_f32 v18, v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v17, v18, v17
	ds_store_b32 v21, v17 offset:8192
	s_branch .LBB0_33
.LBB0_36:                               ;   in Loop: Header=BB0_37 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_lshr_b32 s0, s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc0 .LBB0_30
.LBB0_37:                               ;   Parent Loop BB0_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s2, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	v_cmpx_gt_u32_e64 s0, v0
	s_cbranch_execz .LBB0_36
; %bb.38:                               ;   in Loop: Header=BB0_37 Depth=2
	v_lshl_add_u32 v18, s0, 2, v21
	ds_load_b32 v18, v18 offset:8192
	ds_load_b32 v19, v21 offset:8192
	s_wait_dscnt 0x0
	v_add_f32_e32 v18, v18, v19
	ds_store_b32 v21, v18 offset:8192
	s_branch .LBB0_36
.LBB0_39:                               ;   in Loop: Header=BB0_41 Depth=2
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_40:                               ;   in Loop: Header=BB0_41 Depth=2
	s_add_co_i32 s25, s25, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s25, s21
	s_cbranch_scc1 .LBB0_53
.LBB0_41:                               ;   Parent Loop BB0_11 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_44 Depth 3
                                        ;       Child Loop BB0_50 Depth 3
	s_and_not1_b32 vcc_lo, exec_lo, s9
	s_wait_alu depctr_sa_sdst(0)
	s_lshl2_add_u32 s29, s25, 0
	s_cbranch_vccnz .LBB0_47
; %bb.42:                               ;   in Loop: Header=BB0_41 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v25, s29 :: v_dual_mov_b32 v26, v0
	s_add_co_i32 s2, s25, s20
	s_mov_b32 s33, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[4:5], s[2:3]
	ds_load_b32 v25, v25
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]
	s_mov_b32 s2, 0
                                        ; implicit-def: $sgpr36
	s_wait_dscnt 0x0
	s_branch .LBB0_44
.LBB0_43:                               ;   in Loop: Header=BB0_44 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s37
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s37, exec_lo, s36
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s37, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_46
.LBB0_44:                               ;   Parent Loop BB0_11 Depth=1
                                        ;     Parent Loop BB0_41 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_or_b32 s36, s36, exec_lo
	s_mov_b32 s37, exec_lo
	v_cmpx_gt_i32_e64 s30, v26
	s_cbranch_execz .LBB0_43
; %bb.45:                               ;   in Loop: Header=BB0_44 Depth=3
	v_lshrrev_b32_e32 v27, 5, v26
	s_mov_b32 m0, s33
	s_add_co_i32 s33, s33, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, s33
	v_mul_lo_u32 v29, v27, 34
	v_and_b32_e32 v27, 31, v26
	v_add_nc_u32_e32 v26, s28, v26
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v28, s38, s0, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s1, 0, s38
	s_cselect_b32 s38, -1, 0
	v_add_co_u32 v27, vcc_lo, v28, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, 0, v30, vcc_lo
	s_clause 0x1
	global_load_i8 v27, v[27:28], off offset:2
	global_load_d16_b16 v28, v29, s[0:1]
	v_movrels_b32_e32 v29, v1
	s_and_not1_b32 s36, s36, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s38, s38, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s36, s36, s38
	s_wait_loadcnt 0x1
	v_cvt_f32_i32_e32 v27, v27
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v27, v28, v27, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v29, v25, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v1, v29
	s_branch .LBB0_43
.LBB0_46:                               ;   in Loop: Header=BB0_41 Depth=2
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_47:                               ;   in Loop: Header=BB0_41 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s9
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_40
; %bb.48:                               ;   in Loop: Header=BB0_41 Depth=2
	v_dual_mov_b32 v25, s29 :: v_dual_mov_b32 v26, v0
	s_add_co_i32 s0, s20, s25
	s_mov_b32 s29, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s0, 1
	ds_load_b32 v25, v25 offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[4:5], s[2:3]
	s_mov_b32 s2, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]
                                        ; implicit-def: $sgpr33
	s_wait_dscnt 0x0
	s_branch .LBB0_50
.LBB0_49:                               ;   in Loop: Header=BB0_50 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s36
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s36, exec_lo, s33
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s36, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_39
.LBB0_50:                               ;   Parent Loop BB0_11 Depth=1
                                        ;     Parent Loop BB0_41 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_or_b32 s33, s33, exec_lo
	s_mov_b32 s36, exec_lo
	v_cmpx_gt_i32_e64 s30, v26
	s_cbranch_execz .LBB0_49
; %bb.51:                               ;   in Loop: Header=BB0_50 Depth=3
	v_lshrrev_b32_e32 v27, 5, v26
	s_mov_b32 m0, s29
	s_add_co_i32 s29, s29, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, s29
	v_mul_lo_u32 v29, v27, 34
	v_and_b32_e32 v27, 31, v26
	v_add_nc_u32_e32 v26, s28, v26
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v28, s37, s0, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s1, 0, s37
	s_cselect_b32 s37, -1, 0
	v_add_co_u32 v27, vcc_lo, v28, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, 0, v30, vcc_lo
	s_clause 0x1
	global_load_i8 v27, v[27:28], off offset:2
	global_load_d16_b16 v28, v29, s[0:1]
	v_movrels_b32_e32 v29, v1
	s_and_not1_b32 s33, s33, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s37, s37, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s33, s33, s37
	s_wait_loadcnt 0x1
	v_cvt_f32_i32_e32 v27, v27
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v27, v28, v27, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v29, v25, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v1, v29
	s_branch .LBB0_49
.LBB0_52:                               ;   in Loop: Header=BB0_11 Depth=1
	s_mov_b32 s21, 0
	s_cbranch_execz .LBB0_10
	s_branch .LBB0_54
.LBB0_53:                               ;   in Loop: Header=BB0_11 Depth=1
	v_and_b32_e32 v18, 1, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e64 s0, 1, v18
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_10
.LBB0_54:                               ;   in Loop: Header=BB0_11 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s9
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_10
; %bb.55:                               ;   in Loop: Header=BB0_11 Depth=1
	s_lshl2_add_u32 s0, s21, 0
	s_add_co_i32 s2, s21, s20
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v18, s0 :: v_dual_mov_b32 v25, v0
	s_mul_u64 s[0:1], s[4:5], s[2:3]
	s_mov_b32 s21, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]
	ds_load_b32 v18, v18
	s_mov_b32 s2, 0
                                        ; implicit-def: $sgpr25
	s_wait_dscnt 0x0
	s_branch .LBB0_57
.LBB0_56:                               ;   in Loop: Header=BB0_57 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s29
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s29, exec_lo, s25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s29, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_9
.LBB0_57:                               ;   Parent Loop BB0_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_b32 s25, s25, exec_lo
	s_mov_b32 s29, exec_lo
	v_cmpx_gt_i32_e64 s30, v25
	s_cbranch_execz .LBB0_56
; %bb.58:                               ;   in Loop: Header=BB0_57 Depth=2
	v_lshrrev_b32_e32 v26, 5, v25
	s_mov_b32 m0, s21
	s_add_co_i32 s21, s21, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, s21
	v_mul_lo_u32 v28, v26, 34
	v_and_b32_e32 v26, 31, v25
	v_add_nc_u32_e32 v25, s28, v25
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v27, s33, s0, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, s1, 0, s33
	s_cselect_b32 s33, -1, 0
	v_add_co_u32 v26, vcc_lo, v27, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, 0, v29, vcc_lo
	s_clause 0x1
	global_load_i8 v26, v[26:27], off offset:2
	global_load_d16_b16 v27, v28, s[0:1]
	v_movrels_b32_e32 v28, v1
	s_and_not1_b32 s25, s25, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s33, s33, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s25, s25, s33
	s_wait_loadcnt 0x1
	v_cvt_f32_i32_e32 v26, v26
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v26, v27, v26, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v28, v18, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v1, v28
	s_branch .LBB0_56
.LBB0_59:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s9
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_64
; %bb.60:
	v_div_scale_f32 v17, null, v19, v19, 1.0
	v_div_scale_f32 v22, vcc_lo, 1.0, v19, 1.0
	s_lshl_b64 s[0:1], s[34:35], 2
	v_subrev_nc_u32_e32 v0, s28, v0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[26:27], s[0:1]
	v_rcp_f32_e32 v18, v17
	v_xor_b32_e32 v17, 0x80000000, v17
	s_mov_b32 s3, 0
                                        ; implicit-def: $sgpr2
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v21, v17, v18, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v21, v18
	v_mul_f32_e32 v21, v22, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v23, v17, v21, v22
	v_fmac_f32_e32 v21, v23, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v22, v17, v21
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v17, s0, s0, v20
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v21, v22, v18, v21
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v18, null, s1, 0, s0
	s_lshl_b32 s0, s28, 2
	s_mov_b32 s1, 0
	v_div_fixup_f32 v19, v21, v19, 1.0
	s_branch .LBB0_62
.LBB0_61:                               ;   in Loop: Header=BB0_62 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s4, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s4, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_64
.LBB0_62:                               ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v0, s28, v0
	s_or_b32 s2, s2, exec_lo
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s30, v0
	s_cbranch_execz .LBB0_61
; %bb.63:                               ;   in Loop: Header=BB0_62 Depth=1
	s_mov_b32 m0, s3
	s_add_co_i32 s3, s3, 1
	v_movrels_b32_e32 v20, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, s3
	s_cselect_b32 s5, -1, 0
	s_and_not1_b32 s2, s2, exec_lo
	v_mul_f32_e32 v22, v19, v20
	v_add_co_u32 v20, vcc_lo, v17, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, 0, v18, vcc_lo
	global_store_b32 v[17:18], v22, off
	v_mov_b32_e32 v17, v20
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s5, exec_lo
	v_mov_b32_e32 v18, v21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s2, s5
	s_branch .LBB0_61
.LBB0_64:
	s_endpgm
.Lfunc_end0:
	.size	attention_q8_0_kv, .Lfunc_end0-attention_q8_0_kv
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_q8_0_kv
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
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
		.amdhsa_next_free_vgpr 47
		.amdhsa_next_free_sgpr 50
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-attention_q8_0_kv)<<4)&4080)>>4
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
	.set .Lattention_q8_0_kv.num_vgpr, 47
	.set .Lattention_q8_0_kv.num_agpr, 0
	.set .Lattention_q8_0_kv.numbered_sgpr, 50
	.set .Lattention_q8_0_kv.num_named_barrier, 0
	.set .Lattention_q8_0_kv.private_seg_size, 0
	.set .Lattention_q8_0_kv.uses_vcc, 1
	.set .Lattention_q8_0_kv.uses_flat_scratch, 0
	.set .Lattention_q8_0_kv.has_dyn_sized_stack, 0
	.set .Lattention_q8_0_kv.has_recursion, 0
	.set .Lattention_q8_0_kv.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4952
; TotalNumSgprs: 52
; NumVgprs: 47
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 52
; NumVGPRsForWavesPerEU: 47
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
	.protected	attention_bf16_kv       ; -- Begin function attention_bf16_kv
	.globl	attention_bf16_kv
	.p2align	8
	.type	attention_bf16_kv,@function
attention_bf16_kv:                      ; @attention_bf16_kv
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[28:30], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 ttmp9, s28
	s_cbranch_scc1 .LBB1_64
; %bb.1:
	s_abs_i32 s2, s29
	s_abs_i32 s5, s28
	s_cvt_f32_u32 s3, s2
	s_sub_co_i32 s4, 0, s2
	v_lshlrev_b32_e32 v25, 2, v0
	s_mov_b32 s37, 0
	v_s_rcp_f32 s3, s3
	s_mul_i32 s34, s30, ttmp9
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s3, s3
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s4, s4, s3
	s_mul_hi_u32 s4, s3, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s3, s3, s4
	s_xor_b32 s4, s28, s29
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s3, s5, s3
	s_ashr_i32 s7, s4, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s6, s3, s2
	s_add_co_i32 s8, s3, 1
	s_sub_co_i32 s6, s5, s6
	s_load_b64 s[4:5], s[0:1], 0x20
	s_sub_co_i32 s9, s6, s2
	s_cmp_ge_u32 s6, s2
	s_cselect_b32 s3, s8, s3
	s_cselect_b32 s6, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s3, 1
	s_cmp_ge_u32 s6, s2
	s_load_b32 s6, s[0:1], 0x4c
	s_cselect_b32 s2, s8, s3
	s_abs_i32 s36, ttmp9
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, s2, s7
	s_ashr_i32 s35, s34, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s3, s2, s7
	s_load_b256 s[20:27], s[0:1], 0x0
	s_wait_kmcnt 0x0
	s_load_b32 s33, s[4:5], 0x0
	s_abs_i32 s31, s3
	s_mov_b32 s5, s37
	s_cvt_f32_u32 s2, s31
	s_sub_co_i32 s4, 0, s31
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s2, s2
	s_and_b32 s28, s6, 0xffff
	s_mov_b32 s6, exec_lo
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s2, s2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s4, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s4, s2, s4
	s_add_co_i32 s4, s2, s4
	v_cmpx_gt_i32_e64 s30, v0
	s_cbranch_execz .LBB1_4
; %bb.2:
	s_lshl_b64 s[8:9], s[34:35], 2
	v_dual_mov_b32 v4, v0 :: v_dual_add_nc_u32 v3, 0, v25
	s_add_nc_u64 s[8:9], s[20:21], s[8:9]
	s_lshl_b32 s7, s28, 2
	v_add_co_u32 v1, s2, s8, v25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s9, 0, s2
	s_mov_b32 s8, s37
.LBB1_3:                                ; =>This Inner Loop Header: Depth=1
	global_load_b32 v5, v[1:2], off
	v_add_nc_u32_e32 v4, s28, v4
	v_add_co_u32 v1, vcc_lo, v1, s7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_i32_e64 s2, s30, v4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s8, s2, s8
	s_wait_loadcnt 0x0
	ds_store_b32 v3, v5 offset:8192
	v_add_nc_u32_e32 v3, s7, v3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB1_3
.LBB1_4:
	s_or_b32 exec_lo, exec_lo, s6
	s_mul_u64 s[20:21], s[36:37], s[4:5]
	s_ashr_i32 s20, s3, 31
	s_cvt_f32_u32 s3, s28
	s_sub_co_i32 s13, 0, s28
	s_add_co_i32 s5, s30, s28
	s_ashr_i32 s2, ttmp9, 31
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s3, s3
	s_add_co_i32 s16, s5, -1
	s_wait_dscnt 0x0
	s_abs_i32 s17, s16
	s_ashr_i32 s37, s16, 31
	s_barrier_signal -1
	s_mov_b32 s4, 0
	s_delay_alu instid0(TRANS32_DEP_1)
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_mov_b32 s5, s4
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s3, s3
	s_mov_b32 s8, s4
	s_mov_b32 s9, s4
	s_mov_b32 s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s14, s13, s3
	s_mov_b32 s11, s4
	s_mul_hi_u32 s14, s3, s14
	s_mov_b32 s12, s4
	s_add_co_i32 s3, s3, s14
	s_mov_b32 s13, s4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s3, s17, s3
	s_mov_b32 s14, s4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s18, s3, s28
	s_mov_b32 s15, s4
	s_sub_co_i32 s17, s17, s18
	s_add_co_i32 s18, s3, 1
	s_sub_co_i32 s19, s17, s28
	s_cmp_ge_u32 s17, s28
	s_mov_b32 s16, s4
	s_cselect_b32 s3, s18, s3
	s_cselect_b32 s17, s19, s17
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s18, s3, 1
	s_cmp_ge_u32 s17, s28
	s_mov_b32 s17, s4
	s_cselect_b32 s3, s18, s3
	s_mov_b32 s18, s4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s3, s37
	s_mov_b32 s19, s4
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s3, s3, s37
	v_dual_mov_b32 v1, s4 :: v_dual_mov_b32 v2, s5
	v_dual_mov_b32 v3, s6 :: v_dual_mov_b32 v4, s7
	v_dual_mov_b32 v5, s8 :: v_dual_mov_b32 v6, s9
	v_dual_mov_b32 v7, s10 :: v_dual_mov_b32 v8, s11
	v_dual_mov_b32 v9, s12 :: v_dual_mov_b32 v10, s13
	v_dual_mov_b32 v11, s14 :: v_dual_mov_b32 v12, s15
	v_dual_mov_b32 v13, s16 :: v_dual_mov_b32 v14, s17
	v_dual_mov_b32 v15, s18 :: v_dual_mov_b32 v16, s19
	s_wait_alu depctr_sa_sdst(0)
	s_min_i32 s12, s3, 16
	s_cmp_gt_i32 s3, 0
	s_barrier_wait -1
	s_cselect_b32 s13, -1, 0
	s_cmp_lt_i32 s3, 1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_7
; %bb.5:
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v15, 0
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v13, 0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
.LBB1_6:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 m0, s4
	s_add_co_i32 s4, s4, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, s4
	s_cbranch_scc0 .LBB1_6
.LBB1_7:
	v_mov_b32_e32 v24, 0
	v_subrev_nc_u32_e32 v26, s28, v0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s33, 0
	s_mov_b32 s3, 0
	s_cbranch_scc1 .LBB1_59
; %bb.8:
	s_mul_i32 s4, s21, s31
	s_xor_b32 s2, s2, s20
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s4, s36, s4
	s_add_co_i32 s5, s21, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s6, s4, s31
	s_cmp_ge_u32 s4, s31
	v_mad_co_i64_i32 v[17:18], null, s29, v0, 0
	s_cselect_b32 s5, s5, s21
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s4, s6, s4
	s_add_co_i32 s6, s5, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s31
	s_load_b32 s20, s[0:1], 0x38
	s_cselect_b32 s4, s6, s5
	s_ashr_i32 s11, s29, 31
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, s4, s2
	s_ashr_i32 s31, s30, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s4, s4, s2
	s_lshl2_add_u32 s14, s30, 0
	s_add_co_i32 s15, s33, 1
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 31
	v_lshlrev_b64_e32 v[17:18], 1, v[17:18]
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s10, s29
	s_cselect_b32 s16, -1, 0
	s_lshr_b32 s17, s28, 1
	s_cselect_b32 s18, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mul_u64 s[6:7], s[30:31], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v17, vcc_lo, v17, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s5, v18, vcc_lo
	s_mov_b32 s29, s3
	v_mul_lo_u32 v19, v17, s31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[6:7], s[28:29]
	v_mul_lo_u32 v20, v18, s30
	v_mad_co_u64_u32 v[17:18], null, v17, s30, s[22:23]
	s_lshl_b64 s[8:9], s[0:1], 1
	v_dual_mov_b32 v30, s14 :: v_dual_add_nc_u32 v27, s14, v25
	v_dual_mov_b32 v31, 0 :: v_dual_add_nc_u32 v28, 0, v25
	v_subrev_nc_u32_e32 v29, s28, v0
	s_mov_b32 s19, 1
	v_add3_u32 v18, v20, v18, v19
	v_lshlrev_b32_e32 v19, 1, v0
	s_lshl_b64 s[6:7], s[6:7], 12
	s_lshl_b32 s21, s28, 2
	s_lshl_b64 s[10:11], s[10:11], 1
	s_lshl_b32 s22, s28, 1
	v_add_co_u32 v19, s0, s24, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v20, null, s25, 0, s0
	s_mov_b32 s36, 0xf149f2ca
	s_mov_b32 s23, s15
	s_mov_b32 s24, 0
	s_branch .LBB1_11
.LBB1_9:                                ;   in Loop: Header=BB1_11 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
.LBB1_10:                               ;   in Loop: Header=BB1_11 Depth=1
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_fmac_f32_e32 v24, v31, v23
	v_add_co_u32 v17, vcc_lo, v17, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s7, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mov_b32_e32 v31, v24
	s_addk_co_i32 s24, 0x800
	s_addk_co_i32 s23, 0xf800
	s_addk_co_i32 s19, 0x800
	s_mov_b32 s36, s25
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s24, s33
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_59
.LBB1_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_14 Depth 2
                                        ;       Child Loop BB1_16 Depth 3
                                        ;     Child Loop BB1_19 Depth 2
                                        ;     Child Loop BB1_34 Depth 2
                                        ;     Child Loop BB1_24 Depth 2
                                        ;     Child Loop BB1_27 Depth 2
                                        ;     Child Loop BB1_37 Depth 2
                                        ;     Child Loop BB1_41 Depth 2
                                        ;       Child Loop BB1_44 Depth 3
                                        ;       Child Loop BB1_50 Depth 3
                                        ;     Child Loop BB1_57 Depth 2
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s2, s15, s24
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_gt_i32_e64 s0, s2, v0
	s_min_i32 s29, s2, 0x800
	s_and_saveexec_b32 s25, s0
	s_cbranch_execz .LBB1_17
; %bb.12:                               ;   in Loop: Header=BB1_11 Depth=1
	v_dual_mov_b32 v22, v18 :: v_dual_mov_b32 v21, v17
	v_mov_b32_e32 v32, v0
	s_mov_b32 s37, 0
	s_branch .LBB1_14
.LBB1_13:                               ;   in Loop: Header=BB1_14 Depth=2
	v_lshl_add_u32 v23, v32, 2, 0
	v_add_nc_u32_e32 v32, s28, v32
	v_add_co_u32 v21, s1, v21, s8
	s_wait_kmcnt 0x0
	v_mul_f32_e32 v24, s20, v33
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v22, null, s9, v22, s1
	v_cmp_le_i32_e32 vcc_lo, s29, v32
	ds_store_b32 v23, v24
	s_or_b32 s37, vcc_lo, s37
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s37
	s_cbranch_execz .LBB1_17
.LBB1_14:                               ;   Parent Loop BB1_11 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_16 Depth 3
	v_mov_b32_e32 v33, 0
	s_and_not1_b32 vcc_lo, exec_lo, s16
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_13
; %bb.15:                               ;   in Loop: Header=BB1_14 Depth=2
	v_dual_mov_b32 v24, v22 :: v_dual_mov_b32 v23, v21
	s_mov_b32 s1, 0
	s_mov_b32 s38, s30
.LBB1_16:                               ;   Parent Loop BB1_11 Depth=1
                                        ;     Parent Loop BB1_14 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_u16 v34, v[23:24], off
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v35, s1
	v_add_co_u32 v23, vcc_lo, v23, 2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, 0, v24, vcc_lo
	s_add_co_i32 s38, s38, -1
	s_add_co_i32 s1, s1, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s38, 0
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v34, 16, v34
	ds_load_b32 v35, v35 offset:8192
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v33, v35, v34
	s_cbranch_scc0 .LBB1_16
	s_branch .LBB1_13
.LBB1_17:                               ;   in Loop: Header=BB1_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s25
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v21, 0xf149f2ca
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_21
; %bb.18:                               ;   in Loop: Header=BB1_11 Depth=1
	v_dual_mov_b32 v21, 0xf149f2ca :: v_dual_mov_b32 v22, v28
	v_mov_b32_e32 v23, v0
	s_mov_b32 s25, 0
.LBB1_19:                               ;   Parent Loop BB1_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_load_b32 v24, v22
	v_add_nc_u32_e32 v23, s28, v23
	v_dual_max_num_f32 v21, v21, v21 :: v_dual_add_nc_u32 v22, s21, v22
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v24, v24, v24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_le_i32_e32 vcc_lo, s29, v23
	v_max_num_f32_e32 v21, v21, v24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s25, vcc_lo, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s25
	s_cbranch_execnz .LBB1_19
; %bb.20:                               ;   in Loop: Header=BB1_11 Depth=1
	s_or_b32 exec_lo, exec_lo, s25
.LBB1_21:                               ;   in Loop: Header=BB1_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	ds_store_b32 v27, v21 offset:8192
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s1, s17
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_34
.LBB1_22:                               ;   in Loop: Header=BB1_11 Depth=1
	ds_load_b32 v21, v30 offset:8192
	v_max_num_f32_e64 v22, s36, s36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s1, v22
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v21, v21, v21
	v_readfirstlane_b32 s25, v21
	s_wait_alu depctr_sa_sdst(0)
	s_max_num_f32 s25, s1, s25
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_sub_f32 s1, s36, s25
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s36, s1, 0x3fb8aa3b
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2)
	s_xor_b32 s37, s36, 0x80000000
	s_rndne_f32 s38, s36
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s37, s1, 0x3fb8aa3b, s37
	s_cmp_nlt_f32 s1, 0xc2ce8ed0
	s_sub_f32 s36, s36, s38
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s37, s1, 0x32a5705f, s37
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s1, 0x42b17218
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s36, s36, s37
	s_cvt_i32_f32 s37, s38
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s36, s36
	s_wait_alu depctr_va_sdst(0)
	v_ldexp_f32 v21, s36, s37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v23, 0x7f800000, v21, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_25
; %bb.23:                               ;   in Loop: Header=BB1_11 Depth=1
	s_mov_b32 s1, 0
.LBB1_24:                               ;   Parent Loop BB1_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 m0, s1
	s_add_co_i32 s1, s1, 1
	v_movrels_b32_e32 v21, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v21, v23, v21
	v_movreld_b32_e32 v1, v21
	s_cbranch_scc0 .LBB1_24
.LBB1_25:                               ;   in Loop: Header=BB1_11 Depth=1
	v_mov_b32_e32 v21, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_29
; %bb.26:                               ;   in Loop: Header=BB1_11 Depth=1
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, v28
	v_mov_b32_e32 v24, v0
	s_mov_b32 s0, 0
.LBB1_27:                               ;   Parent Loop BB1_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_load_b32 v32, v22
	v_add_nc_u32_e32 v24, s28, v24
	s_wait_dscnt 0x0
	v_subrev_f32_e32 v32, s25, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v33, 0x3fb8aa3b, v32
	v_fma_f32 v34, 0x3fb8aa3b, v32, -v33
	v_rndne_f32_e32 v35, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v33, v33, v35
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v32
	v_fmac_f32_e32 v34, 0x32a5705f, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v33, v33, v34
	v_cvt_i32_f32_e32 v34, v35
	v_exp_f32_e32 v33, v33
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v33, v33, v34
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v33, 0, v33, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v32, 0x7f800000, v33, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, s29, v24
	ds_store_b32 v22, v32
	v_dual_add_f32 v21, v21, v32 :: v_dual_add_nc_u32 v22, s21, v22
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB1_27
; %bb.28:                               ;   in Loop: Header=BB1_11 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
.LBB1_29:                               ;   in Loop: Header=BB1_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	ds_store_b32 v27, v21 offset:8192
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, s17
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_37
.LBB1_30:                               ;   in Loop: Header=BB1_11 Depth=1
	v_mov_b32_e32 v21, s14
	s_cmp_lt_i32 s2, 1
	ds_load_b32 v24, v21 offset:8192
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_10
; %bb.31:                               ;   in Loop: Header=BB1_11 Depth=1
	s_cmp_lt_i32 s23, 2
	s_cbranch_scc1 .LBB1_52
; %bb.32:                               ;   in Loop: Header=BB1_11 Depth=1
	v_med3_i32 v32, s23, 1, 0x800
	s_mov_b32 s36, 0
	s_mov_b32 s0, s19
	s_mov_b32 s2, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s1, v32
	s_and_b32 s29, s1, 0xffe
	s_branch .LBB1_41
.LBB1_33:                               ;   in Loop: Header=BB1_34 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s25
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_lshr_b32 s1, s1, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc0 .LBB1_22
.LBB1_34:                               ;   Parent Loop BB1_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s25, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	v_cmpx_gt_u32_e64 s1, v0
	s_cbranch_execz .LBB1_33
; %bb.35:                               ;   in Loop: Header=BB1_34 Depth=2
	v_lshl_add_u32 v21, s1, 2, v27
	ds_load_b32 v21, v21 offset:8192
	ds_load_b32 v22, v27 offset:8192
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v21, v21, v21 :: v_dual_max_num_f32 v22, v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v21, v22, v21
	ds_store_b32 v27, v21 offset:8192
	s_branch .LBB1_33
.LBB1_36:                               ;   in Loop: Header=BB1_37 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_lshr_b32 s0, s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc0 .LBB1_30
.LBB1_37:                               ;   Parent Loop BB1_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s1, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	v_cmpx_gt_u32_e64 s0, v0
	s_cbranch_execz .LBB1_36
; %bb.38:                               ;   in Loop: Header=BB1_37 Depth=2
	v_lshl_add_u32 v21, s0, 2, v27
	ds_load_b32 v21, v21 offset:8192
	ds_load_b32 v22, v27 offset:8192
	s_wait_dscnt 0x0
	v_add_f32_e32 v21, v21, v22
	ds_store_b32 v27, v21 offset:8192
	s_branch .LBB1_36
.LBB1_39:                               ;   in Loop: Header=BB1_41 Depth=2
	s_or_b32 exec_lo, exec_lo, s1
.LBB1_40:                               ;   in Loop: Header=BB1_41 Depth=2
	s_add_co_i32 s36, s36, 2
	s_add_co_i32 s2, s2, 2
	s_add_co_i32 s0, s0, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s36, s29
	s_cbranch_scc1 .LBB1_53
.LBB1_41:                               ;   Parent Loop BB1_11 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_44 Depth 3
                                        ;       Child Loop BB1_50 Depth 3
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_wait_alu depctr_sa_sdst(0)
	s_lshl2_add_u32 s1, s36, 0
	s_cbranch_vccnz .LBB1_47
; %bb.42:                               ;   in Loop: Header=BB1_41 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v21, s1 :: v_dual_mov_b32 v34, v29
	s_mul_u64 s[38:39], s[10:11], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[4:5], s[38:39]
	ds_load_b32 v33, v21
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[21:22], null, s30, s38, v[19:20]
	s_mul_i32 s37, s31, s38
	s_mul_i32 s38, s30, s39
	s_mov_b32 s39, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v22, s38, s37, v22
	s_mov_b32 s37, 0
                                        ; implicit-def: $sgpr38
	s_wait_dscnt 0x0
	s_branch .LBB1_44
.LBB1_43:                               ;   in Loop: Header=BB1_44 Depth=3
	s_or_b32 exec_lo, exec_lo, s40
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s40, exec_lo, s38
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s37, s40, s37
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s37
	s_cbranch_execz .LBB1_46
.LBB1_44:                               ;   Parent Loop BB1_11 Depth=1
                                        ;     Parent Loop BB1_41 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_nc_u32_e32 v34, s28, v34
	s_or_b32 s38, s38, exec_lo
	s_mov_b32 s40, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s30, v34
	s_cbranch_execz .LBB1_43
; %bb.45:                               ;   in Loop: Header=BB1_44 Depth=3
	global_load_u16 v35, v[21:22], off
	s_mov_b32 m0, s39
	s_add_co_i32 s39, s39, 1
	v_movrels_b32_e32 v36, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, s39
	v_add_co_u32 v21, vcc_lo, v21, s22
	s_cselect_b32 s41, -1, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, 0, v22, vcc_lo
	s_and_not1_b32 s38, s38, exec_lo
	s_and_b32 s41, s41, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s38, s38, s41
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v35, 16, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v36, v33, v35
	v_movreld_b32_e32 v1, v36
	s_branch .LBB1_43
.LBB1_46:                               ;   in Loop: Header=BB1_41 Depth=2
	s_or_b32 exec_lo, exec_lo, s37
.LBB1_47:                               ;   in Loop: Header=BB1_41 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_40
; %bb.48:                               ;   in Loop: Header=BB1_41 Depth=2
	v_dual_mov_b32 v21, s1 :: v_dual_mov_b32 v34, v29
	s_mov_b32 s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[38:39], s[10:11], s[0:1]
	ds_load_b32 v33, v21 offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[4:5], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[21:22], null, s30, s38, v[19:20]
	s_mul_i32 s1, s31, s38
	s_mul_i32 s37, s30, s39
	s_mov_b32 s38, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v22, s37, s1, v22
	s_mov_b32 s1, 0
                                        ; implicit-def: $sgpr37
	s_wait_dscnt 0x0
	s_branch .LBB1_50
.LBB1_49:                               ;   in Loop: Header=BB1_50 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s39, exec_lo, s37
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s39, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB1_39
.LBB1_50:                               ;   Parent Loop BB1_11 Depth=1
                                        ;     Parent Loop BB1_41 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_nc_u32_e32 v34, s28, v34
	s_or_b32 s37, s37, exec_lo
	s_mov_b32 s39, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s30, v34
	s_cbranch_execz .LBB1_49
; %bb.51:                               ;   in Loop: Header=BB1_50 Depth=3
	global_load_u16 v35, v[21:22], off
	s_mov_b32 m0, s38
	s_add_co_i32 s38, s38, 1
	v_movrels_b32_e32 v36, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, s38
	v_add_co_u32 v21, vcc_lo, v21, s22
	s_cselect_b32 s40, -1, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, 0, v22, vcc_lo
	s_and_not1_b32 s37, s37, exec_lo
	s_and_b32 s40, s40, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s37, s37, s40
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v35, 16, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v36, v33, v35
	v_movreld_b32_e32 v1, v36
	s_branch .LBB1_49
.LBB1_52:                               ;   in Loop: Header=BB1_11 Depth=1
	s_mov_b32 s29, 0
	s_cbranch_execz .LBB1_10
	s_branch .LBB1_54
.LBB1_53:                               ;   in Loop: Header=BB1_11 Depth=1
	v_and_b32_e32 v21, 1, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e64 s0, 1, v21
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_10
.LBB1_54:                               ;   in Loop: Header=BB1_11 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_10
; %bb.55:                               ;   in Loop: Header=BB1_11 Depth=1
	s_lshl2_add_u32 s0, s29, 0
	s_add_co_i32 s2, s29, s24
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v21, s0
	s_mul_u64 s[0:1], s[10:11], s[2:3]
	v_mov_b32_e32 v33, v29
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[4:5], s[0:1]
	s_mov_b32 s2, 0
	ds_load_b32 v32, v21
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[21:22], null, s30, s0, v[19:20]
	s_mul_i32 s0, s31, s0
	s_mul_i32 s1, s30, s1
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v22, s1, s0, v22
	s_mov_b32 s0, 0
                                        ; implicit-def: $sgpr1
	s_wait_dscnt 0x0
	s_branch .LBB1_57
.LBB1_56:                               ;   in Loop: Header=BB1_57 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s29
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s29, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s29, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB1_9
.LBB1_57:                               ;   Parent Loop BB1_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_nc_u32_e32 v33, s28, v33
	s_or_b32 s1, s1, exec_lo
	s_mov_b32 s29, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s30, v33
	s_cbranch_execz .LBB1_56
; %bb.58:                               ;   in Loop: Header=BB1_57 Depth=2
	global_load_u16 v34, v[21:22], off
	s_mov_b32 m0, s2
	s_add_co_i32 s2, s2, 1
	v_movrels_b32_e32 v35, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, s2
	v_add_co_u32 v21, vcc_lo, v21, s22
	s_cselect_b32 s36, -1, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, 0, v22, vcc_lo
	s_and_not1_b32 s1, s1, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s36, s36, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s1, s36
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v34, 16, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v35, v32, v34
	v_movreld_b32_e32 v1, v35
	s_branch .LBB1_56
.LBB1_59:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_64
; %bb.60:
	v_div_scale_f32 v0, null, v24, v24, 1.0
	v_div_scale_f32 v19, vcc_lo, 1.0, v24, 1.0
	s_lshl_b64 s[0:1], s[34:35], 2
	s_mov_b32 s3, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[26:27], s[0:1]
	v_rcp_f32_e32 v17, v0
	v_xor_b32_e32 v0, 0x80000000, v0
                                        ; implicit-def: $sgpr2
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v0, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v19, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, v0, v18, v19
	v_fmac_f32_e32 v18, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v0, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v0, v19, v17, v18
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v17, s0, s0, v25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v18, null, s1, 0, s0
	v_div_fixup_f32 v0, v0, v24, 1.0
	s_lshl_b32 s0, s28, 2
	s_mov_b32 s1, 0
	s_branch .LBB1_62
.LBB1_61:                               ;   in Loop: Header=BB1_62 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s4, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s4, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB1_64
.LBB1_62:                               ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v26, s28, v26
	s_or_b32 s2, s2, exec_lo
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s30, v26
	s_cbranch_execz .LBB1_61
; %bb.63:                               ;   in Loop: Header=BB1_62 Depth=1
	s_mov_b32 m0, s3
	s_add_co_i32 s3, s3, 1
	v_movrels_b32_e32 v19, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, s3
	s_cselect_b32 s5, -1, 0
	s_and_not1_b32 s2, s2, exec_lo
	v_mul_f32_e32 v21, v0, v19
	v_add_co_u32 v19, vcc_lo, v17, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, 0, v18, vcc_lo
	global_store_b32 v[17:18], v21, off
	v_mov_b32_e32 v17, v19
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s5, exec_lo
	v_mov_b32_e32 v18, v20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s2, s5
	s_branch .LBB1_61
.LBB1_64:
	s_endpgm
.Lfunc_end1:
	.size	attention_bf16_kv, .Lfunc_end1-attention_bf16_kv
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_bf16_kv
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
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
		.amdhsa_next_free_vgpr 37
		.amdhsa_next_free_sgpr 42
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-attention_bf16_kv)<<4)&4080)>>4
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
	.set .Lattention_bf16_kv.num_vgpr, 37
	.set .Lattention_bf16_kv.num_agpr, 0
	.set .Lattention_bf16_kv.numbered_sgpr, 42
	.set .Lattention_bf16_kv.num_named_barrier, 0
	.set .Lattention_bf16_kv.private_seg_size, 0
	.set .Lattention_bf16_kv.uses_vcc, 1
	.set .Lattention_bf16_kv.uses_flat_scratch, 0
	.set .Lattention_bf16_kv.has_dyn_sized_stack, 0
	.set .Lattention_bf16_kv.has_recursion, 0
	.set .Lattention_bf16_kv.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3756
; TotalNumSgprs: 44
; NumVgprs: 37
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 37
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
	.type	__hip_cuid_47ead4c4071cf047,@object ; @__hip_cuid_47ead4c4071cf047
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_47ead4c4071cf047
__hip_cuid_47ead4c4071cf047:
	.byte	0                               ; 0x0
	.size	__hip_cuid_47ead4c4071cf047, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_47ead4c4071cf047
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
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         184
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           attention_q8_0_kv
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .sgpr_spill_count: 0
    .symbol:         attention_q8_0_kv.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     47
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
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         184
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           attention_bf16_kv
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         attention_bf16_kv.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     37
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
