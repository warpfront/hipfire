	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	t_old_fill_dump         ; -- Begin function t_old_fill_dump
	.globl	t_old_fill_dump
	.p2align	8
	.type	t_old_fill_dump,@function
t_old_fill_dump:                        ; @t_old_fill_dump
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x18
	s_mov_b32 s2, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_4
; %bb.1:
	v_mov_b32_e32 v1, 0
	v_lshl_add_u32 v2, v0, 2, 0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v3, s7
	s_mov_b32 s8, 0
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, -1, v1
	ds_store_b32 v2, v3
	v_add_nc_u32_e32 v2, 0x200, v2
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_or_b32 s8, vcc_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_2
; %bb.3:
	s_or_b32 exec_lo, exec_lo, s8
.LBB0_4:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v1, s7 :: v_dual_mov_b32 v2, v0
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x1bff, v2
	v_lshl_add_u32 v3, v2, 2, 0
	ds_store_2addr_stride64_b32 v3, v1, v1 offset0:8 offset1:10
	ds_store_2addr_stride64_b32 v3, v1, v1 offset0:12 offset1:14
	v_mov_b32_e32 v2, v4
	s_or_b32 s2, vcc_lo, s2
	ds_store_2addr_stride64_b32 v3, v1, v1 offset1:2
	ds_store_2addr_stride64_b32 v3, v1, v1 offset0:4 offset1:6
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_5
; %bb.6:
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b128 s[8:11], s[0:1], 0x0
	v_dual_mov_b32 v2, 0 :: v_dual_and_b32 v15, 15, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_ashr_i32 s7, s6, 31
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 4, v15
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[12:13], s[6:7], 8
	v_lshrrev_b32_e32 v3, 4, v0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v8, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v7, 0 :: v_dual_add_nc_u32 v4, s4, v3
	s_barrier_wait -1
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[2:3], s[8:9], s[12:13]
	global_inv scope:SCOPE_SE
	v_add_co_u32 v13, s2, s2, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s3, 0, s2
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_i32_e64 s5, v4
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[13:14]
	global_load_b128 v[5:8], v[4:5], off
.LBB0_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v20, 0x80, v0
	v_dual_mov_b32 v4, 0 :: v_dual_lshlrev_b32 v21, 7, v15
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v9, 4, v20
	v_or_b32_e32 v3, v3, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v10, s4, v9
	v_lshl_add_u32 v11, v3, 3, 0
	v_mov_b32_e32 v3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v11, v[5:6], v[7:8] offset1:16
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, v[13:14]
	global_load_b128 v[1:4], v[1:2], off
.LBB0_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v19, 0x100, v0
	v_or_b32_e32 v5, v21, v9
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v7, 4, v19
	v_lshl_add_u32 v16, v5, 3, 0
	v_mov_b32_e32 v12, 0
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v5, 0 :: v_dual_add_nc_u32 v8, s4, v7
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v16, v[1:2], v[3:4] offset1:16
	v_cmpx_gt_i32_e64 s5, v8
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v8, v[13:14]
	global_load_b128 v[9:12], v[1:2], off
.LBB0_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_lshrrev_b32_e32 v1, 3, v19
	v_or_b32_e32 v18, 0x180, v0
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v1, 32, v1
	v_lshrrev_b32_e32 v2, 4, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v1, v1, v21
	v_and_or_b32 v3, v7, 15, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v1, s4, v2
	v_mov_b32_e32 v7, 0
	v_lshl_add_u32 v3, v3, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v3, v[9:10], v[11:12] offset1:16
	v_cmpx_gt_i32_e64 s5, v1
	s_cbranch_execz .LBB0_14
; %bb.13:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v1, v[13:14]
	global_load_b128 v[5:8], v[3:4], off
.LBB0_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_lshrrev_b32_e32 v1, 3, v18
	v_or_b32_e32 v17, 0x200, v0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v12, 0
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, 0 :: v_dual_and_b32 v1, 32, v1
	v_dual_mov_b32 v10, 0 :: v_dual_add_nc_u32 v3, v1, v21
	v_lshrrev_b32_e32 v1, 4, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v11, v2, 15, v3
	v_dual_mov_b32 v3, 0 :: v_dual_add_nc_u32 v2, s4, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_add_u32 v16, v11, 3, 0
	v_mov_b32_e32 v11, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v16, v[5:6], v[7:8] offset1:16
	v_cmpx_gt_i32_e64 s5, v2
	s_cbranch_execz .LBB0_16
; %bb.15:
	v_mad_co_i64_i32 v[5:6], null, 0x408, v2, v[13:14]
	global_load_b128 v[9:12], v[5:6], off
.LBB0_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_lshrrev_b32_e32 v2, 3, v17
	s_load_b64 s[2:3], s[0:1], 0x10
	v_or_b32_e32 v16, 0x280, v0
	s_mov_b32 s0, exec_lo
	v_mov_b32_e32 v6, 0
	v_and_b32_e32 v2, 0x60, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v5, v2, v21
	v_lshrrev_b32_e32 v2, 4, v16
	v_and_or_b32 v5, v1, 15, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v1, s4, v2
	v_lshl_add_u32 v7, v5, 3, 0
	v_mov_b32_e32 v5, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v7, v[9:10], v[11:12] offset1:16
	v_cmpx_gt_i32_e64 s5, v1
	s_cbranch_execz .LBB0_18
; %bb.17:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v1, v[13:14]
	global_load_b128 v[3:6], v[3:4], off
.LBB0_18:
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v1, 3, v16
	v_or_b32_e32 v12, 0x300, v0
	v_mov_b32_e32 v7, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v1, 0x60, v1
	v_lshrrev_b32_e32 v22, 4, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v10, 0 :: v_dual_add_nc_u32 v1, v1, v21
	v_add_nc_u32_e32 v11, s4, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v9, v2, 15, v1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	v_lshl_add_u32 v23, v9, 3, 0
	v_mov_b32_e32 v9, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v23, v[3:4], v[5:6] offset1:16
	v_cmpx_gt_i32_e64 s5, v11
	s_cbranch_execz .LBB0_20
; %bb.19:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v11, v[13:14]
	global_load_b128 v[7:10], v[3:4], off
.LBB0_20:
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v3, 3, v12
	v_or_b32_e32 v11, 0x380, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v4, 0 :: v_dual_and_b32 v3, 0x60, v3
	v_lshrrev_b32_e32 v5, 4, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v3, v3, v21
	v_add_nc_u32_e32 v6, s4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v3, v22, 15, v3
	v_lshl_add_u32 v22, v3, 3, 0
	v_mov_b32_e32 v3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v22, v[7:8], v[9:10] offset1:16
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB0_22
; %bb.21:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, v[13:14]
	global_load_b128 v[1:4], v[1:2], off
.LBB0_22:
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v7, 3, v11
	v_lshrrev_b32_e32 v6, 1, v0
	s_add_nc_u64 s[0:1], s[10:11], s[12:13]
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v7, 0x60, v7
	v_and_b32_e32 v8, 8, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v9, v7, v21
	v_add_nc_u32_e32 v7, s4, v8
	v_lshrrev_b32_e32 v8, 5, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_or_b32 v5, v5, 15, v9
	v_add_nc_u32_e32 v10, 7, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_or_b32 v9, v8, 4, v15
	v_lshl_add_u32 v5, v5, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v5, v[1:2], v[3:4] offset1:16
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v10
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_38
; %bb.23:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB0_37
; %bb.24:
	v_add_co_u32 v3, s13, s0, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v5, 1, v7
	s_mov_b32 s13, exec_lo
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_mad_co_i64_i32 v[13:14], null, 0x408, v5, v[3:4]
	global_load_u8 v5, v[13:14], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 8, v1
.LBB0_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v5, 2, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_mad_co_i64_i32 v[13:14], null, 0x408, v5, v[3:4]
	global_load_u8 v5, v[13:14], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB0_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v5, 3, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_mad_co_i64_i32 v[13:14], null, 0x408, v5, v[3:4]
	global_load_u8 v5, v[13:14], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v5, 4, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_mad_co_i64_i32 v[13:14], null, 0x408, v5, v[3:4]
	global_load_u8 v5, v[13:14], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v5, 5, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_mad_co_i64_i32 v[13:14], null, 0x408, v5, v[3:4]
	global_load_u8 v5, v[13:14], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v5, 6, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v5, v[3:4]
	v_mov_b16_e32 v5.l, 0
	global_load_d16_hi_u8 v5, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB0_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_37:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s12
.LBB0_38:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, s[0:1]
	v_mov_b16_e32 v5.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v21.l, v5.h
	v_add_co_u32 v1, vcc_lo, v1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v5, v[1:2], off
	global_load_u8 v13, v[1:2], off offset:3096
	global_load_u8 v14, v[1:2], off offset:5160
	global_load_u8 v22, v[1:2], off offset:4128
	global_load_u8 v23, v[1:2], off offset:7224
	global_load_d16_hi_u8 v21, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v13
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v14
	v_or_b32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v22, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v23
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v21, v3
	v_or3_b32 v1, v1, 0, 0
.LBB0_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_lshlrev_b32_e32 v3, 3, v0
	v_lshrrev_b32_e32 v13, 5, v20
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0xf8, v3
	v_add_nc_u32_e32 v5, 0, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_or_b32 v3, v13, 4, v15
	v_lshl_add_u32 v4, v8, 8, v5
	ds_store_b64 v4, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v10
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_56
; %bb.41:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB0_55
; %bb.42:
	v_add_co_u32 v3, s13, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v14, 1, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_mad_co_i64_i32 v[20:21], null, 0x408, v14, v[3:4]
	global_load_u8 v14, v[20:21], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v14, 8, v1
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v14, 2, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_mad_co_i64_i32 v[20:21], null, 0x408, v14, v[3:4]
	global_load_u8 v14, v[20:21], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v14, 16, v1
.LBB0_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v14, 3, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_mad_co_i64_i32 v[20:21], null, 0x408, v14, v[3:4]
	global_load_u8 v14, v[20:21], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v14, 24, v1
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v14, 4, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_mad_co_i64_i32 v[20:21], null, 0x408, v14, v[3:4]
	global_load_u8 v14, v[20:21], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v14, v2
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v14, 5, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_mad_co_i64_i32 v[20:21], null, 0x408, v14, v[3:4]
	global_load_u8 v14, v[20:21], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v14, 8, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v14, v2
.LBB0_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v14, 6, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB0_54
; %bb.53:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v14, v[3:4]
	v_mov_b16_e32 v14.l, 0
	global_load_d16_hi_u8 v14, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v14, v2
.LBB0_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr3
.LBB0_56:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, s[0:1]
	v_mov_b16_e32 v14.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v22.l, v14.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v14, v[1:2], off
	global_load_u8 v20, v[1:2], off offset:3096
	global_load_u8 v21, v[1:2], off offset:5160
	global_load_u8 v23, v[1:2], off offset:4128
	global_load_u8 v24, v[1:2], off offset:7224
	global_load_d16_hi_u8 v22, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v20
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v21
	v_or_b32_e32 v1, v1, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v23, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v24
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v22, v3
	v_or3_b32 v1, v1, 0, 0
.LBB0_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_lshrrev_b32_e32 v14, 5, v19
	v_lshl_add_u32 v4, v13, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v3, v14, 4, v15
	ds_store_b64 v4, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v10
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_74
; %bb.59:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB0_73
; %bb.60:
	v_add_co_u32 v3, s13, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v13, 1, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_mad_co_i64_i32 v[19:20], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[19:20], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 8, v1
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v13, 2, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_mad_co_i64_i32 v[19:20], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[19:20], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 16, v1
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v13, 3, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_mad_co_i64_i32 v[19:20], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[19:20], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 24, v1
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v13, 4, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_mad_co_i64_i32 v[19:20], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[19:20], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v13, v2
.LBB0_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v13, 5, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_mad_co_i64_i32 v[19:20], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[19:20], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v13, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v13, v2
.LBB0_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v13, 6, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v13, v[3:4]
	v_mov_b16_e32 v13.l, 0
	global_load_d16_hi_u8 v13, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v13, v2
.LBB0_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr3
.LBB0_74:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_76
; %bb.75:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, s[0:1]
	v_mov_b16_e32 v13.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v21.l, v13.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v13, v[1:2], off
	global_load_u8 v19, v[1:2], off offset:3096
	global_load_u8 v20, v[1:2], off offset:5160
	global_load_u8 v22, v[1:2], off offset:4128
	global_load_u8 v23, v[1:2], off offset:7224
	global_load_d16_hi_u8 v21, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v19
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v20
	v_or_b32_e32 v1, v1, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v22, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v23
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v21, v3
	v_or3_b32 v1, v1, 0, 0
.LBB0_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_lshrrev_b32_e32 v13, 5, v18
	v_lshl_add_u32 v4, v14, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v3, v13, 4, v15
	ds_store_b64 v4, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v10
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_92
; %bb.77:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB0_91
; %bb.78:
	v_add_co_u32 v3, s13, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v10, 1, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_80
; %bb.79:
	v_mad_co_i64_i32 v[18:19], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[18:19], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 8, v1
.LBB0_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 2, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_82
; %bb.81:
	v_mad_co_i64_i32 v[18:19], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[18:19], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 16, v1
.LBB0_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 3, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_84
; %bb.83:
	v_mad_co_i64_i32 v[18:19], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[18:19], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 24, v1
.LBB0_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 4, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_86
; %bb.85:
	v_mad_co_i64_i32 v[18:19], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[18:19], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB0_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 5, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_88
; %bb.87:
	v_mad_co_i64_i32 v[18:19], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[18:19], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v10, 8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v10, v2
.LBB0_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 6, v7
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_90
; %bb.89:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v10, v[3:4]
	v_mov_b16_e32 v10.l, 0
	global_load_d16_hi_u8 v10, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB0_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr3
.LBB0_92:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_94
; %bb.93:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, s[0:1]
	v_mov_b16_e32 v10.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v19.l, v10.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v10, v[1:2], off
	global_load_u8 v14, v[1:2], off offset:3096
	global_load_u8 v18, v[1:2], off offset:5160
	global_load_u8 v20, v[1:2], off offset:4128
	global_load_u8 v21, v[1:2], off offset:7224
	global_load_d16_hi_u8 v19, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v14
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v18
	v_or_b32_e32 v1, v1, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v20, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v21
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v19, v3
	v_or3_b32 v1, v1, 0, 0
.LBB0_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_lshrrev_b32_e32 v10, 5, v17
	v_lshl_add_u32 v13, v13, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_and_b32_e32 v3, 16, v10
	ds_store_b64 v13, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_add_nc_u32_e32 v14, v7, v3
	v_lshrrev_b32_e32 v3, 1, v17
	v_add_nc_u32_e32 v4, 7, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v3, 0xf0, v3, v15
	v_cmpx_le_i32_e64 s5, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_110
; %bb.95:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB0_109
; %bb.96:
	v_add_co_u32 v3, s13, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v13, 1, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_98
; %bb.97:
	v_mad_co_i64_i32 v[17:18], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[17:18], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 8, v1
.LBB0_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v13, 2, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_100
; %bb.99:
	v_mad_co_i64_i32 v[17:18], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[17:18], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 16, v1
.LBB0_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v13, 3, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_102
; %bb.101:
	v_mad_co_i64_i32 v[17:18], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[17:18], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 24, v1
.LBB0_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v13, 4, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_104
; %bb.103:
	v_mad_co_i64_i32 v[17:18], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[17:18], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v13, v2
.LBB0_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v13, 5, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_mad_co_i64_i32 v[17:18], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[17:18], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v13, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v13, v2
.LBB0_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v13, 6, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v13, v[3:4]
	v_mov_b16_e32 v13.l, 0
	global_load_d16_hi_u8 v13, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v13, v2
.LBB0_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_109:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $vgpr3
.LBB0_110:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_112
; %bb.111:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, s[0:1]
	v_mov_b16_e32 v13.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v18.l, v13.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v13, v[1:2], off
	global_load_u8 v14, v[1:2], off offset:3096
	global_load_u8 v17, v[1:2], off offset:5160
	global_load_u8 v19, v[1:2], off offset:4128
	global_load_u8 v20, v[1:2], off offset:7224
	global_load_d16_hi_u8 v18, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v14
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v17
	v_or_b32_e32 v1, v1, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v19, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v20
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v18, v3
	v_or3_b32 v1, v1, 0, 0
.LBB0_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_lshrrev_b32_e32 v13, 5, v16
	v_lshl_add_u32 v10, v10, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_and_b32_e32 v3, 16, v13
	ds_store_b64 v10, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_add_nc_u32_e32 v14, v7, v3
	v_lshrrev_b32_e32 v3, 1, v16
	v_add_nc_u32_e32 v4, 7, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v3, 0xf0, v3, v15
	v_cmpx_le_i32_e64 s5, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_128
; %bb.113:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB0_127
; %bb.114:
	v_add_co_u32 v3, s13, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v10, 1, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_116
; %bb.115:
	v_mad_co_i64_i32 v[16:17], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[16:17], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 8, v1
.LBB0_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 2, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_118
; %bb.117:
	v_mad_co_i64_i32 v[16:17], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[16:17], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 16, v1
.LBB0_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 3, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_120
; %bb.119:
	v_mad_co_i64_i32 v[16:17], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[16:17], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 24, v1
.LBB0_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 4, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_122
; %bb.121:
	v_mad_co_i64_i32 v[16:17], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[16:17], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB0_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 5, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_124
; %bb.123:
	v_mad_co_i64_i32 v[16:17], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[16:17], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v10, 8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v10, v2
.LBB0_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 6, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_126
; %bb.125:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v10, v[3:4]
	v_mov_b16_e32 v10.l, 0
	global_load_d16_hi_u8 v10, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB0_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_127:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $vgpr3
.LBB0_128:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_130
; %bb.129:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, s[0:1]
	v_mov_b16_e32 v10.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v17.l, v10.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v10, v[1:2], off
	global_load_u8 v14, v[1:2], off offset:3096
	global_load_u8 v16, v[1:2], off offset:5160
	global_load_u8 v18, v[1:2], off offset:4128
	global_load_u8 v19, v[1:2], off offset:7224
	global_load_d16_hi_u8 v17, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v14
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v16
	v_or_b32_e32 v1, v1, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v18, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v19
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v17, v3
	v_or3_b32 v1, v1, 0, 0
.LBB0_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_lshrrev_b32_e32 v10, 5, v12
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 16, v10
	v_add_nc_u32_e32 v14, v7, v3
	v_lshrrev_b32_e32 v3, 1, v12
	v_lshl_add_u32 v12, v13, 8, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v4, 7, v14
	v_and_or_b32 v3, 0xf0, v3, v15
	ds_store_b64 v12, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_146
; %bb.131:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB0_145
; %bb.132:
	v_add_co_u32 v3, s13, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v12, 1, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB0_134
; %bb.133:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v12, v[3:4]
	global_load_u8 v12, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v12, 8, v1
.LBB0_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v12, 2, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB0_136
; %bb.135:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v12, v[3:4]
	global_load_u8 v12, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v12, 16, v1
.LBB0_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v12, 3, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB0_138
; %bb.137:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v12, v[3:4]
	global_load_u8 v12, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v12, 24, v1
.LBB0_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v12, 4, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB0_140
; %bb.139:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v12, v[3:4]
	global_load_u8 v12, v[12:13], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v12, v2
.LBB0_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v12, 5, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB0_142
; %bb.141:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v12, v[3:4]
	global_load_u8 v12, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v12, 8, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v12, v2
.LBB0_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v12, 6, v14
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB0_144
; %bb.143:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v12, v[3:4]
	v_mov_b16_e32 v12.l, 0
	global_load_d16_hi_u8 v12, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v12, v2
.LBB0_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_145:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $vgpr3
.LBB0_146:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_148
; %bb.147:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, s[0:1]
	v_mov_b16_e32 v12.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v16.l, v12.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v12, v[1:2], off
	global_load_u8 v13, v[1:2], off offset:3096
	global_load_u8 v14, v[1:2], off offset:5160
	global_load_u8 v17, v[1:2], off offset:4128
	global_load_u8 v18, v[1:2], off offset:7224
	global_load_d16_hi_u8 v16, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v13
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v14
	v_or_b32_e32 v1, v1, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v17, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v18
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v16, v3
	v_or3_b32 v1, v1, 0, 0
.LBB0_148:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_lshrrev_b32_e32 v12, 5, v11
	v_lshl_add_u32 v10, v10, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_and_b32_e32 v3, 16, v12
	ds_store_b64 v10, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_add_nc_u32_e32 v13, v7, v3
	v_lshrrev_b32_e32 v3, 1, v11
	v_add_nc_u32_e32 v4, 7, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v3, 0xf0, v3, v15
	v_cmpx_le_i32_e64 s5, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_164
; %bb.149:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB0_163
; %bb.150:
	v_add_co_u32 v3, s13, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v10, 1, v13
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v13, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_152
; %bb.151:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 8, v1
.LBB0_152:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 2, v13
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_154
; %bb.153:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 16, v1
.LBB0_154:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 3, v13
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_156
; %bb.155:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 24, v1
.LBB0_156:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 4, v13
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_158
; %bb.157:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB0_158:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 5, v13
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_160
; %bb.159:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v10, 8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v10, v2
.LBB0_160:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v10, 6, v13
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_162
; %bb.161:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v10, v[3:4]
	v_mov_b16_e32 v10.l, 0
	global_load_d16_hi_u8 v10, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB0_162:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_163:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr3
.LBB0_164:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_166
; %bb.165:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v13, s[0:1]
	v_mov_b16_e32 v10.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v14.l, v10.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v10, v[1:2], off
	global_load_u8 v11, v[1:2], off offset:3096
	global_load_u8 v13, v[1:2], off offset:5160
	global_load_u8 v16, v[1:2], off offset:4128
	global_load_u8 v17, v[1:2], off offset:7224
	global_load_d16_hi_u8 v14, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v11
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v13
	v_or_b32_e32 v1, v1, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v16, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v17
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v14, v3
	v_or3_b32 v1, v1, 0, 0
.LBB0_166:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v11, 39, v7
	v_lshl_add_u32 v3, v12, 8, v5
	v_add_nc_u32_e32 v10, 32, v7
	s_mov_b32 s7, exec_lo
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_182
; %bb.167:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_181
; %bb.168:
	v_add_co_u32 v3, s13, s0, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v9, 1, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_170
; %bb.169:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB0_170:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 2, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_172
; %bb.171:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB0_172:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 3, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_174
; %bb.173:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB0_174:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 4, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_176
; %bb.175:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_176:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 5, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_178
; %bb.177:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB0_178:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 6, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_180
; %bb.179:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_180:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_181:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr9
.LBB0_182:
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s7, s7
	v_or_b32_e32 v3, 32, v8
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB0_184
; %bb.183:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, s[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_mov_b16_e32 v9.h, 0
	s_clause 0x4
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v8, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v12, v[1:2], off offset:3096
	global_load_u8 v13, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v14.l, v9.h
	s_clause 0x2
	global_load_u8 v16, v[1:2], off offset:4128
	global_load_u8 v17, v[1:2], off offset:7224
	global_load_d16_hi_u8 v14, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v4
	v_lshlrev_b32_e32 v2, 16, v8
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v12
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v8, 8, v13
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v16, v8
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v17
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v14, v4
	v_or3_b32 v1, v1, 0, 0
.LBB0_184:
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v8, 0x480, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v8
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0x70, v4, v15
	v_cmpx_le_i32_e64 s5, v11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_200
; %bb.185:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_199
; %bb.186:
	v_add_co_u32 v3, s13, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v9, 1, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_188
; %bb.187:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB0_188:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 2, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_190
; %bb.189:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB0_190:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 3, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_192
; %bb.191:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB0_192:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 4, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_194
; %bb.193:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_194:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 5, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_196
; %bb.195:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB0_196:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 6, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_198
; %bb.197:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_198:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_199:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr4
.LBB0_200:
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s7, s7
	v_lshrrev_b32_e32 v3, 5, v8
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB0_202
; %bb.201:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v14.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v8, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v12, v[1:2], off offset:3096
	global_load_u8 v13, v[1:2], off offset:5160
	global_load_u8 v16, v[1:2], off offset:4128
	global_load_u8 v17, v[1:2], off offset:7224
	global_load_d16_hi_u8 v14, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v8
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v12
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v8, 8, v13
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v16, v8
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v17
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v14, v4
	v_or3_b32 v1, v1, 0, 0
.LBB0_202:
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v8, 0x500, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v8
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0xb0, v4, v15
	v_cmpx_le_i32_e64 s5, v11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_218
; %bb.203:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_217
; %bb.204:
	v_add_co_u32 v3, s13, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v9, 1, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_206
; %bb.205:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB0_206:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 2, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_208
; %bb.207:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB0_208:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 3, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_210
; %bb.209:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB0_210:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 4, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_212
; %bb.211:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_212:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 5, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_214
; %bb.213:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB0_214:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 6, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_216
; %bb.215:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_216:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_217:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr4
.LBB0_218:
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s7, s7
	v_lshrrev_b32_e32 v3, 5, v8
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB0_220
; %bb.219:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v14.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v8, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v12, v[1:2], off offset:3096
	global_load_u8 v13, v[1:2], off offset:5160
	global_load_u8 v16, v[1:2], off offset:4128
	global_load_u8 v17, v[1:2], off offset:7224
	global_load_d16_hi_u8 v14, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v8
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v12
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v8, 8, v13
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v16, v8
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v17
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v14, v4
	v_or3_b32 v1, v1, 0, 0
.LBB0_220:
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v8, 0x580, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v8
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0xf0, v4, v15
	v_cmpx_le_i32_e64 s5, v11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_236
; %bb.221:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB0_235
; %bb.222:
	v_add_co_u32 v3, s13, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v9, 1, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_224
; %bb.223:
	v_mad_co_i64_i32 v[11:12], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB0_224:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 2, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_226
; %bb.225:
	v_mad_co_i64_i32 v[11:12], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB0_226:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 3, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_228
; %bb.227:
	v_mad_co_i64_i32 v[11:12], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB0_228:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 4, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_230
; %bb.229:
	v_mad_co_i64_i32 v[11:12], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[11:12], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_230:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 5, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_232
; %bb.231:
	v_mad_co_i64_i32 v[11:12], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB0_232:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 6, v10
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_234
; %bb.233:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_234:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_235:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr10
                                        ; implicit-def: $vgpr4
.LBB0_236:
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s7, s7
	v_lshrrev_b32_e32 v3, 5, v8
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB0_238
; %bb.237:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v12.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v8, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v10, v[1:2], off offset:3096
	global_load_u8 v11, v[1:2], off offset:5160
	global_load_u8 v13, v[1:2], off offset:4128
	global_load_u8 v14, v[1:2], off offset:7224
	global_load_d16_hi_u8 v12, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v8
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v10
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v8, 8, v11
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v13, v8
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v14
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v12, v4
	v_or3_b32 v1, v1, 0, 0
.LBB0_238:
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v8, 55, v7
	v_lshl_add_u32 v3, v3, 8, v5
	v_or_b32_e32 v9, 0x600, v0
	v_and_or_b32 v4, v6, 48, v15
	v_add_nc_u32_e32 v6, 48, v7
	s_mov_b32 s7, exec_lo
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v8
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_254
; %bb.239:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB0_253
; %bb.240:
	v_add_co_u32 v3, s13, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v7, 1, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB0_242
; %bb.241:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v7, v[3:4]
	global_load_u8 v7, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v7, 8, v1
.LBB0_242:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v7, 2, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB0_244
; %bb.243:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v7, v[3:4]
	global_load_u8 v7, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v7, 16, v1
.LBB0_244:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v7, 3, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB0_246
; %bb.245:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v7, v[3:4]
	global_load_u8 v7, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v7, 24, v1
.LBB0_246:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v7, 4, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB0_248
; %bb.247:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v7, v[3:4]
	global_load_u8 v7, v[10:11], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB0_248:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v7, 5, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB0_250
; %bb.249:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v7, v[3:4]
	global_load_u8 v7, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v7, 8, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v7, v2
.LBB0_250:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v7, 6, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB0_252
; %bb.251:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v7, v[3:4]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB0_252:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_253:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr4
.LBB0_254:
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s7, s7
	v_lshrrev_b32_e32 v3, 5, v9
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB0_256
; %bb.255:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v12.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v10, v[1:2], off offset:3096
	global_load_u8 v11, v[1:2], off offset:5160
	global_load_u8 v13, v[1:2], off offset:4128
	global_load_u8 v14, v[1:2], off offset:7224
	global_load_d16_hi_u8 v12, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v10
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v11
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v13, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v14
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v12, v4
	v_or3_b32 v1, v1, 0, 0
.LBB0_256:
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v7, 0x680, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v7
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0x70, v4, v15
	v_cmpx_le_i32_e64 s5, v8
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_272
; %bb.257:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB0_271
; %bb.258:
	v_add_co_u32 v3, s13, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v9, 1, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_260
; %bb.259:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB0_260:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 2, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_262
; %bb.261:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB0_262:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 3, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_264
; %bb.263:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB0_264:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 4, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_266
; %bb.265:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_266:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 5, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_268
; %bb.267:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB0_268:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 6, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_270
; %bb.269:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_270:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_271:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr4
.LBB0_272:
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s7, s7
	v_lshrrev_b32_e32 v3, 5, v7
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB0_274
; %bb.273:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v12.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v10, v[1:2], off offset:3096
	global_load_u8 v11, v[1:2], off offset:5160
	global_load_u8 v13, v[1:2], off offset:4128
	global_load_u8 v14, v[1:2], off offset:7224
	global_load_d16_hi_u8 v12, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v10
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v11
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v13, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v14
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v12, v4
	v_or3_b32 v1, v1, 0, 0
.LBB0_274:
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v7, 0x700, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v7
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0xb0, v4, v15
	v_cmpx_le_i32_e64 s5, v8
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_290
; %bb.275:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB0_289
; %bb.276:
	v_add_co_u32 v3, s13, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v9, 1, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_278
; %bb.277:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB0_278:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 2, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_280
; %bb.279:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB0_280:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 3, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_282
; %bb.281:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB0_282:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 4, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_284
; %bb.283:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_284:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 5, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_286
; %bb.285:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB0_286:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v9, 6, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB0_288
; %bb.287:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB0_288:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_289:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr4
.LBB0_290:
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s7, s7
	v_lshrrev_b32_e32 v3, 5, v7
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB0_292
; %bb.291:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v12.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v10, v[1:2], off offset:3096
	global_load_u8 v11, v[1:2], off offset:5160
	global_load_u8 v13, v[1:2], off offset:4128
	global_load_u8 v14, v[1:2], off offset:7224
	global_load_d16_hi_u8 v12, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v10
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v11
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v13, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v14
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v12, v4
	v_or3_b32 v1, v1, 0, 0
.LBB0_292:
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v7, 0x780, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v7
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0xf0, v4, v15
	v_cmpx_le_i32_e64 s5, v8
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execz .LBB0_308
; %bb.293:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB0_307
; %bb.294:
	v_add_co_u32 v3, s13, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s13
	v_add_nc_u32_e32 v8, 1, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v8
	s_cbranch_execz .LBB0_296
; %bb.295:
	v_mad_co_i64_i32 v[8:9], null, 0x408, v8, v[3:4]
	global_load_u8 v8, v[8:9], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v8, 8, v1
.LBB0_296:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v8, 2, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v8
	s_cbranch_execz .LBB0_298
; %bb.297:
	v_mad_co_i64_i32 v[8:9], null, 0x408, v8, v[3:4]
	global_load_u8 v8, v[8:9], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v8, 16, v1
.LBB0_298:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v8, 3, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v8
	s_cbranch_execz .LBB0_300
; %bb.299:
	v_mad_co_i64_i32 v[8:9], null, 0x408, v8, v[3:4]
	global_load_u8 v8, v[8:9], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v8, 24, v1
.LBB0_300:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v8, 4, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v8
	s_cbranch_execz .LBB0_302
; %bb.301:
	v_mad_co_i64_i32 v[8:9], null, 0x408, v8, v[3:4]
	global_load_u8 v8, v[8:9], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v8, v2
.LBB0_302:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v8, 5, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v8
	s_cbranch_execz .LBB0_304
; %bb.303:
	v_mad_co_i64_i32 v[8:9], null, 0x408, v8, v[3:4]
	global_load_u8 v8, v[8:9], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v8, 8, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v8, v2
.LBB0_304:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v6, 6, v6
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB0_306
; %bb.305:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v6, v[3:4]
	v_mov_b16_e32 v6.l, 0
	global_load_d16_hi_u8 v6, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v6, v2
.LBB0_306:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_307:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr4
.LBB0_308:
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s7, s7
	v_lshrrev_b32_e32 v3, 5, v7
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB0_310
; %bb.309:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, s[0:1]
	v_mov_b16_e32 v7.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v10.l, v7.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v6, v[1:2], off offset:2064
	global_load_d16_u8 v7, v[1:2], off
	global_load_u8 v8, v[1:2], off offset:3096
	global_load_u8 v9, v[1:2], off offset:5160
	global_load_u8 v11, v[1:2], off offset:4128
	global_load_u8 v12, v[1:2], off offset:7224
	global_load_d16_hi_u8 v10, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v6
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v8
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 8, v9
	v_or_b32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v11, v6
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v12
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v10, v4
	v_or3_b32 v1, v1, 0, 0
.LBB0_310:
	s_or_b32 exec_lo, exec_lo, s7
	v_lshl_add_u32 v3, v3, 8, v5
	v_lshlrev_b32_e32 v65, 8, v0
	s_mov_b32 s0, exec_lo
	ds_store_b64 v3, v[1:2] offset:16384
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v63, 0, v65
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_b32 v[3:4], v63 offset0:2 offset1:3
	ds_load_2addr_b32 v[7:8], v63 offset0:6 offset1:7
	ds_load_2addr_b32 v[5:6], v63 offset0:4 offset1:5
	ds_load_2addr_b32 v[1:2], v63 offset1:1
	ds_load_2addr_b32 v[11:12], v63 offset0:10 offset1:11
	ds_load_2addr_b32 v[15:16], v63 offset0:14 offset1:15
	ds_load_2addr_b32 v[13:14], v63 offset0:12 offset1:13
	ds_load_2addr_b32 v[9:10], v63 offset0:8 offset1:9
	ds_load_2addr_b32 v[19:20], v63 offset0:18 offset1:19
	ds_load_2addr_b32 v[23:24], v63 offset0:22 offset1:23
	ds_load_2addr_b32 v[21:22], v63 offset0:20 offset1:21
	ds_load_2addr_b32 v[17:18], v63 offset0:16 offset1:17
	ds_load_2addr_b32 v[27:28], v63 offset0:26 offset1:27
	ds_load_2addr_b32 v[31:32], v63 offset0:30 offset1:31
	ds_load_2addr_b32 v[29:30], v63 offset0:28 offset1:29
	ds_load_2addr_b32 v[25:26], v63 offset0:24 offset1:25
	ds_load_2addr_b32 v[35:36], v63 offset0:34 offset1:35
	ds_load_2addr_b32 v[39:40], v63 offset0:38 offset1:39
	ds_load_2addr_b32 v[37:38], v63 offset0:36 offset1:37
	ds_load_2addr_b32 v[33:34], v63 offset0:32 offset1:33
	ds_load_2addr_b32 v[43:44], v63 offset0:42 offset1:43
	ds_load_2addr_b32 v[47:48], v63 offset0:46 offset1:47
	ds_load_2addr_b32 v[45:46], v63 offset0:44 offset1:45
	ds_load_2addr_b32 v[41:42], v63 offset0:40 offset1:41
	ds_load_2addr_b32 v[51:52], v63 offset0:50 offset1:51
	ds_load_2addr_b32 v[55:56], v63 offset0:54 offset1:55
	ds_load_2addr_b32 v[53:54], v63 offset0:52 offset1:53
	ds_load_2addr_b32 v[49:50], v63 offset0:48 offset1:49
	ds_load_2addr_b32 v[59:60], v63 offset0:58 offset1:59
	ds_load_2addr_b32 v[61:62], v63 offset0:60 offset1:61
	ds_load_2addr_b32 v[57:58], v63 offset0:56 offset1:57
	ds_load_2addr_b32 v[63:64], v63 offset0:62 offset1:63
	s_wait_dscnt 0x1c
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_store_b128 v65, v[1:4], s[2:3]
	global_store_b128 v65, v[5:8], s[2:3] offset:16
	s_wait_dscnt 0x18
	s_clause 0x1
	global_store_b128 v65, v[9:12], s[2:3] offset:32
	global_store_b128 v65, v[13:16], s[2:3] offset:48
	s_wait_dscnt 0x14
	s_clause 0x1
	global_store_b128 v65, v[17:20], s[2:3] offset:64
	global_store_b128 v65, v[21:24], s[2:3] offset:80
	s_wait_dscnt 0x10
	s_clause 0x1
	global_store_b128 v65, v[25:28], s[2:3] offset:96
	global_store_b128 v65, v[29:32], s[2:3] offset:112
	s_wait_dscnt 0xc
	s_clause 0x1
	global_store_b128 v65, v[33:36], s[2:3] offset:128
	global_store_b128 v65, v[37:40], s[2:3] offset:144
	s_wait_dscnt 0x8
	s_clause 0x1
	global_store_b128 v65, v[41:44], s[2:3] offset:160
	global_store_b128 v65, v[45:48], s[2:3] offset:176
	s_wait_dscnt 0x4
	s_clause 0x1
	global_store_b128 v65, v[49:52], s[2:3] offset:192
	global_store_b128 v65, v[53:56], s[2:3] offset:208
	s_wait_dscnt 0x1
	global_store_b128 v65, v[57:60], s[2:3] offset:224
	s_wait_dscnt 0x0
	global_store_b128 v65, v[61:64], s[2:3] offset:240
	v_cmpx_gt_u32_e32 64, v0
	s_cbranch_execz .LBB0_316
; %bb.311:
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v3, s4, v0
	v_mov_b32_e32 v4, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s5, v3
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_313
; %bb.312:
	v_mad_co_i64_i32 v[4:5], null, 0x408, v3, s[8:9]
	s_lshl_b32 s0, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s4, s0, 31
	v_add_co_u32 v4, s0, v4, s0
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s4, v5, s0
	global_load_d16_b16 v1, v[4:5], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v4, v1.l
.LBB0_313:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_lshlrev_b32_e32 v5, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v0, s0, s2, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s3, 0, s0
	global_store_b32 v5, v4, s[2:3] offset:32768
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_315
; %bb.314:
	v_mad_co_i64_i32 v[2:3], null, 0x408, v3, s[10:11]
	s_lshl_b32 s1, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s2, s1, 31
	v_add_co_u32 v2, vcc_lo, v2, s1
	s_wait_alu depctr_sa_sdst(0) depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s2, v3, vcc_lo
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v2, v2.l
.LBB0_315:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	global_store_b32 v[0:1], v2, off offset:33024
.LBB0_316:
	s_endpgm
.Lfunc_end0:
	.size	t_old_fill_dump, .Lfunc_end0-t_old_fill_dump
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel t_old_fill_dump
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
		.amdhsa_next_free_vgpr 66
		.amdhsa_next_free_sgpr 14
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-t_old_fill_dump)<<4)&4080)>>4
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
	.set .Lt_old_fill_dump.num_vgpr, 66
	.set .Lt_old_fill_dump.num_agpr, 0
	.set .Lt_old_fill_dump.numbered_sgpr, 14
	.set .Lt_old_fill_dump.num_named_barrier, 0
	.set .Lt_old_fill_dump.private_seg_size, 0
	.set .Lt_old_fill_dump.uses_vcc, 1
	.set .Lt_old_fill_dump.uses_flat_scratch, 0
	.set .Lt_old_fill_dump.has_dyn_sized_stack, 0
	.set .Lt_old_fill_dump.has_recursion, 0
	.set .Lt_old_fill_dump.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15476
; TotalNumSgprs: 16
; NumVgprs: 66
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 66
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	t_old_fill              ; -- Begin function t_old_fill
	.globl	t_old_fill
	.p2align	8
	.type	t_old_fill,@function
t_old_fill:                             ; @t_old_fill
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b96 s[4:6], s[0:1], 0x10
	s_load_b128 s[0:3], s[0:1], 0x0
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v15, 15, v0
	v_lshrrev_b32_e32 v3, 4, v0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_lshlrev_b32_e32 v9, 4, v15
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v8, 0
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v7, 0 :: v_dual_add_nc_u32 v4, s4, v3
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[0:1], s[0:1], s[6:7]
	v_add_co_u32 v13, s0, s0, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s1, 0, s0
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s5, v4
	s_cbranch_execz .LBB1_2
; %bb.1:
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, v[13:14]
	global_load_b128 v[5:8], v[4:5], off
.LBB1_2:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v20, 0x80, v0
	v_dual_mov_b32 v4, 0 :: v_dual_lshlrev_b32 v21, 7, v15
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v9, 4, v20
	v_or_b32_e32 v3, v3, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v10, s4, v9
	v_lshl_add_u32 v11, v3, 3, 0
	v_mov_b32_e32 v3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v11, v[5:6], v[7:8] offset1:16
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_4
; %bb.3:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, v[13:14]
	global_load_b128 v[1:4], v[1:2], off
.LBB1_4:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v19, 0x100, v0
	v_or_b32_e32 v5, v21, v9
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v7, 4, v19
	v_lshl_add_u32 v16, v5, 3, 0
	v_mov_b32_e32 v12, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v5, 0 :: v_dual_add_nc_u32 v8, s4, v7
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v16, v[1:2], v[3:4] offset1:16
	v_cmpx_gt_i32_e64 s5, v8
	s_cbranch_execz .LBB1_6
; %bb.5:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v8, v[13:14]
	global_load_b128 v[9:12], v[1:2], off
.LBB1_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v1, 3, v19
	v_or_b32_e32 v18, 0x180, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v1, 32, v1
	v_lshrrev_b32_e32 v2, 4, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v1, v1, v21
	v_and_or_b32 v3, v7, 15, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v1, s4, v2
	v_mov_b32_e32 v7, 0
	v_lshl_add_u32 v3, v3, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v3, v[9:10], v[11:12] offset1:16
	v_cmpx_gt_i32_e64 s5, v1
	s_cbranch_execz .LBB1_8
; %bb.7:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v1, v[13:14]
	global_load_b128 v[5:8], v[3:4], off
.LBB1_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v1, 3, v18
	v_or_b32_e32 v17, 0x200, v0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v12, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, 0 :: v_dual_and_b32 v1, 32, v1
	v_dual_mov_b32 v10, 0 :: v_dual_add_nc_u32 v3, v1, v21
	v_lshrrev_b32_e32 v1, 4, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v11, v2, 15, v3
	v_dual_mov_b32 v3, 0 :: v_dual_add_nc_u32 v2, s4, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_add_u32 v16, v11, 3, 0
	v_mov_b32_e32 v11, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v16, v[5:6], v[7:8] offset1:16
	v_cmpx_gt_i32_e64 s5, v2
	s_cbranch_execz .LBB1_10
; %bb.9:
	v_mad_co_i64_i32 v[5:6], null, 0x408, v2, v[13:14]
	global_load_b128 v[9:12], v[5:6], off
.LBB1_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v2, 3, v17
	v_or_b32_e32 v16, 0x280, v0
	s_mov_b32 s0, exec_lo
	v_mov_b32_e32 v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x60, v2
	v_add_nc_u32_e32 v5, v2, v21
	v_lshrrev_b32_e32 v2, 4, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v5, v1, 15, v5
	v_add_nc_u32_e32 v1, s4, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_add_u32 v7, v5, 3, 0
	v_mov_b32_e32 v5, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v7, v[9:10], v[11:12] offset1:16
	v_cmpx_gt_i32_e64 s5, v1
	s_cbranch_execz .LBB1_12
; %bb.11:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v1, v[13:14]
	global_load_b128 v[3:6], v[3:4], off
.LBB1_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v1, 3, v16
	v_or_b32_e32 v12, 0x300, v0
	v_mov_b32_e32 v7, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v1, 0x60, v1
	v_lshrrev_b32_e32 v22, 4, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v10, 0 :: v_dual_add_nc_u32 v1, v1, v21
	v_add_nc_u32_e32 v11, s4, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v9, v2, 15, v1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	v_lshl_add_u32 v23, v9, 3, 0
	v_mov_b32_e32 v9, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v23, v[3:4], v[5:6] offset1:16
	v_cmpx_gt_i32_e64 s5, v11
	s_cbranch_execz .LBB1_14
; %bb.13:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v11, v[13:14]
	global_load_b128 v[7:10], v[3:4], off
.LBB1_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v3, 3, v12
	v_or_b32_e32 v11, 0x380, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v4, 0 :: v_dual_and_b32 v3, 0x60, v3
	v_lshrrev_b32_e32 v5, 4, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v3, v3, v21
	v_add_nc_u32_e32 v6, s4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v3, v22, 15, v3
	v_lshl_add_u32 v22, v3, 3, 0
	v_mov_b32_e32 v3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v22, v[7:8], v[9:10] offset1:16
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB1_16
; %bb.15:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, v[13:14]
	global_load_b128 v[1:4], v[1:2], off
.LBB1_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v7, 3, v11
	v_lshrrev_b32_e32 v6, 1, v0
	s_add_nc_u64 s[0:1], s[2:3], s[6:7]
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v7, 0x60, v7
	v_and_b32_e32 v8, 8, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v9, v7, v21
	v_add_nc_u32_e32 v7, s4, v8
	v_lshrrev_b32_e32 v8, 5, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_or_b32 v5, v5, 15, v9
	v_add_nc_u32_e32 v10, 7, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_or_b32 v9, v8, 4, v15
	v_lshl_add_u32 v5, v5, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v5, v[1:2], v[3:4] offset1:16
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v10
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_32
; %bb.17:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_31
; %bb.18:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v5, 1, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB1_20
; %bb.19:
	v_mad_co_i64_i32 v[13:14], null, 0x408, v5, v[3:4]
	global_load_u8 v5, v[13:14], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 8, v1
.LBB1_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v5, 2, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB1_22
; %bb.21:
	v_mad_co_i64_i32 v[13:14], null, 0x408, v5, v[3:4]
	global_load_u8 v5, v[13:14], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 16, v1
.LBB1_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v5, 3, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB1_24
; %bb.23:
	v_mad_co_i64_i32 v[13:14], null, 0x408, v5, v[3:4]
	global_load_u8 v5, v[13:14], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v5, 24, v1
.LBB1_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v5, 4, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB1_26
; %bb.25:
	v_mad_co_i64_i32 v[13:14], null, 0x408, v5, v[3:4]
	global_load_u8 v5, v[13:14], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB1_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v5, 5, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB1_28
; %bb.27:
	v_mad_co_i64_i32 v[13:14], null, 0x408, v5, v[3:4]
	global_load_u8 v5, v[13:14], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v5, v2
.LBB1_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v5, 6, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v5
	s_cbranch_execz .LBB1_30
; %bb.29:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v5, v[3:4]
	v_mov_b16_e32 v5.l, 0
	global_load_d16_hi_u8 v5, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v5, v2
.LBB1_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_31:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
.LBB1_32:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB1_34
; %bb.33:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, s[0:1]
	v_mov_b16_e32 v5.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v21.l, v5.h
	v_add_co_u32 v1, vcc_lo, v1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v5, v[1:2], off
	global_load_u8 v13, v[1:2], off offset:3096
	global_load_u8 v14, v[1:2], off offset:5160
	global_load_u8 v22, v[1:2], off offset:4128
	global_load_u8 v23, v[1:2], off offset:7224
	global_load_d16_hi_u8 v21, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v13
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v14
	v_or_b32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v22, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v23
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v21, v3
	v_or3_b32 v1, v1, 0, 0
.LBB1_34:
	s_or_b32 exec_lo, exec_lo, s2
	v_lshlrev_b32_e32 v3, 3, v0
	v_lshrrev_b32_e32 v13, 5, v20
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0xf8, v3
	v_add_nc_u32_e32 v5, 0, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_or_b32 v3, v13, 4, v15
	v_lshl_add_u32 v4, v8, 8, v5
	ds_store_b64 v4, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v10
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_50
; %bb.35:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_49
; %bb.36:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v14, 1, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB1_38
; %bb.37:
	v_mad_co_i64_i32 v[20:21], null, 0x408, v14, v[3:4]
	global_load_u8 v14, v[20:21], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v14, 8, v1
.LBB1_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v14, 2, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB1_40
; %bb.39:
	v_mad_co_i64_i32 v[20:21], null, 0x408, v14, v[3:4]
	global_load_u8 v14, v[20:21], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v14, 16, v1
.LBB1_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v14, 3, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB1_42
; %bb.41:
	v_mad_co_i64_i32 v[20:21], null, 0x408, v14, v[3:4]
	global_load_u8 v14, v[20:21], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v14, 24, v1
.LBB1_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v14, 4, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB1_44
; %bb.43:
	v_mad_co_i64_i32 v[20:21], null, 0x408, v14, v[3:4]
	global_load_u8 v14, v[20:21], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v14, v2
.LBB1_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v14, 5, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB1_46
; %bb.45:
	v_mad_co_i64_i32 v[20:21], null, 0x408, v14, v[3:4]
	global_load_u8 v14, v[20:21], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v14, 8, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v14, v2
.LBB1_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v14, 6, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB1_48
; %bb.47:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v14, v[3:4]
	v_mov_b16_e32 v14.l, 0
	global_load_d16_hi_u8 v14, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v14, v2
.LBB1_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_49:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr3
.LBB1_50:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB1_52
; %bb.51:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, s[0:1]
	v_mov_b16_e32 v14.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v22.l, v14.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v14, v[1:2], off
	global_load_u8 v20, v[1:2], off offset:3096
	global_load_u8 v21, v[1:2], off offset:5160
	global_load_u8 v23, v[1:2], off offset:4128
	global_load_u8 v24, v[1:2], off offset:7224
	global_load_d16_hi_u8 v22, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v20
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v21
	v_or_b32_e32 v1, v1, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v23, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v24
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v22, v3
	v_or3_b32 v1, v1, 0, 0
.LBB1_52:
	s_or_b32 exec_lo, exec_lo, s2
	v_lshrrev_b32_e32 v14, 5, v19
	v_lshl_add_u32 v4, v13, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v3, v14, 4, v15
	ds_store_b64 v4, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v10
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_68
; %bb.53:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_67
; %bb.54:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v13, 1, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_56
; %bb.55:
	v_mad_co_i64_i32 v[19:20], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[19:20], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 8, v1
.LBB1_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v13, 2, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_58
; %bb.57:
	v_mad_co_i64_i32 v[19:20], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[19:20], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 16, v1
.LBB1_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v13, 3, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_60
; %bb.59:
	v_mad_co_i64_i32 v[19:20], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[19:20], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 24, v1
.LBB1_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v13, 4, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_62
; %bb.61:
	v_mad_co_i64_i32 v[19:20], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[19:20], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v13, v2
.LBB1_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v13, 5, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_64
; %bb.63:
	v_mad_co_i64_i32 v[19:20], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[19:20], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v13, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v13, v2
.LBB1_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v13, 6, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_66
; %bb.65:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v13, v[3:4]
	v_mov_b16_e32 v13.l, 0
	global_load_d16_hi_u8 v13, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v13, v2
.LBB1_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_67:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr3
.LBB1_68:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB1_70
; %bb.69:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, s[0:1]
	v_mov_b16_e32 v13.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v21.l, v13.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v13, v[1:2], off
	global_load_u8 v19, v[1:2], off offset:3096
	global_load_u8 v20, v[1:2], off offset:5160
	global_load_u8 v22, v[1:2], off offset:4128
	global_load_u8 v23, v[1:2], off offset:7224
	global_load_d16_hi_u8 v21, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v19
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v20
	v_or_b32_e32 v1, v1, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v22, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v23
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v21, v3
	v_or3_b32 v1, v1, 0, 0
.LBB1_70:
	s_or_b32 exec_lo, exec_lo, s2
	v_lshrrev_b32_e32 v13, 5, v18
	v_lshl_add_u32 v4, v14, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v3, v13, 4, v15
	ds_store_b64 v4, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v10
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_86
; %bb.71:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_85
; %bb.72:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v10, 1, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_74
; %bb.73:
	v_mad_co_i64_i32 v[18:19], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[18:19], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 8, v1
.LBB1_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 2, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_76
; %bb.75:
	v_mad_co_i64_i32 v[18:19], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[18:19], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 16, v1
.LBB1_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 3, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_78
; %bb.77:
	v_mad_co_i64_i32 v[18:19], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[18:19], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 24, v1
.LBB1_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 4, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_80
; %bb.79:
	v_mad_co_i64_i32 v[18:19], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[18:19], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB1_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 5, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_82
; %bb.81:
	v_mad_co_i64_i32 v[18:19], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[18:19], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v10, 8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v10, v2
.LBB1_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 6, v7
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_84
; %bb.83:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v10, v[3:4]
	v_mov_b16_e32 v10.l, 0
	global_load_d16_hi_u8 v10, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB1_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_85:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr3
.LBB1_86:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB1_88
; %bb.87:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, s[0:1]
	v_mov_b16_e32 v10.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v19.l, v10.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v10, v[1:2], off
	global_load_u8 v14, v[1:2], off offset:3096
	global_load_u8 v18, v[1:2], off offset:5160
	global_load_u8 v20, v[1:2], off offset:4128
	global_load_u8 v21, v[1:2], off offset:7224
	global_load_d16_hi_u8 v19, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v14
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v18
	v_or_b32_e32 v1, v1, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v20, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v21
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v19, v3
	v_or3_b32 v1, v1, 0, 0
.LBB1_88:
	s_or_b32 exec_lo, exec_lo, s2
	v_lshrrev_b32_e32 v10, 5, v17
	v_lshl_add_u32 v13, v13, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_and_b32_e32 v3, 16, v10
	ds_store_b64 v13, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_add_nc_u32_e32 v14, v7, v3
	v_lshrrev_b32_e32 v3, 1, v17
	v_add_nc_u32_e32 v4, 7, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v3, 0xf0, v3, v15
	v_cmpx_le_i32_e64 s5, v4
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_104
; %bb.89:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB1_103
; %bb.90:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v13, 1, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_92
; %bb.91:
	v_mad_co_i64_i32 v[17:18], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[17:18], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 8, v1
.LBB1_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v13, 2, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_94
; %bb.93:
	v_mad_co_i64_i32 v[17:18], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[17:18], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 16, v1
.LBB1_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v13, 3, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_96
; %bb.95:
	v_mad_co_i64_i32 v[17:18], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[17:18], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v13, 24, v1
.LBB1_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v13, 4, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_98
; %bb.97:
	v_mad_co_i64_i32 v[17:18], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[17:18], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v13, v2
.LBB1_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v13, 5, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_100
; %bb.99:
	v_mad_co_i64_i32 v[17:18], null, 0x408, v13, v[3:4]
	global_load_u8 v13, v[17:18], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v13, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v13, v2
.LBB1_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v13, 6, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_102
; %bb.101:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v13, v[3:4]
	v_mov_b16_e32 v13.l, 0
	global_load_d16_hi_u8 v13, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v13, v2
.LBB1_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_103:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $vgpr3
.LBB1_104:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB1_106
; %bb.105:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, s[0:1]
	v_mov_b16_e32 v13.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v18.l, v13.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v13, v[1:2], off
	global_load_u8 v14, v[1:2], off offset:3096
	global_load_u8 v17, v[1:2], off offset:5160
	global_load_u8 v19, v[1:2], off offset:4128
	global_load_u8 v20, v[1:2], off offset:7224
	global_load_d16_hi_u8 v18, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v14
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v17
	v_or_b32_e32 v1, v1, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v19, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v20
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v18, v3
	v_or3_b32 v1, v1, 0, 0
.LBB1_106:
	s_or_b32 exec_lo, exec_lo, s2
	v_lshrrev_b32_e32 v13, 5, v16
	v_lshl_add_u32 v10, v10, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_and_b32_e32 v3, 16, v13
	ds_store_b64 v10, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_add_nc_u32_e32 v14, v7, v3
	v_lshrrev_b32_e32 v3, 1, v16
	v_add_nc_u32_e32 v4, 7, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v3, 0xf0, v3, v15
	v_cmpx_le_i32_e64 s5, v4
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_122
; %bb.107:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB1_121
; %bb.108:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v10, 1, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_110
; %bb.109:
	v_mad_co_i64_i32 v[16:17], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[16:17], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 8, v1
.LBB1_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 2, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_112
; %bb.111:
	v_mad_co_i64_i32 v[16:17], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[16:17], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 16, v1
.LBB1_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 3, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_114
; %bb.113:
	v_mad_co_i64_i32 v[16:17], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[16:17], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 24, v1
.LBB1_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 4, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_116
; %bb.115:
	v_mad_co_i64_i32 v[16:17], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[16:17], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB1_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 5, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_118
; %bb.117:
	v_mad_co_i64_i32 v[16:17], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[16:17], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v10, 8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v10, v2
.LBB1_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 6, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_120
; %bb.119:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v10, v[3:4]
	v_mov_b16_e32 v10.l, 0
	global_load_d16_hi_u8 v10, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB1_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_121:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $vgpr3
.LBB1_122:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB1_124
; %bb.123:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, s[0:1]
	v_mov_b16_e32 v10.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v17.l, v10.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v10, v[1:2], off
	global_load_u8 v14, v[1:2], off offset:3096
	global_load_u8 v16, v[1:2], off offset:5160
	global_load_u8 v18, v[1:2], off offset:4128
	global_load_u8 v19, v[1:2], off offset:7224
	global_load_d16_hi_u8 v17, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v14
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v16
	v_or_b32_e32 v1, v1, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v18, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v19
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v17, v3
	v_or3_b32 v1, v1, 0, 0
.LBB1_124:
	s_or_b32 exec_lo, exec_lo, s2
	v_lshrrev_b32_e32 v10, 5, v12
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 16, v10
	v_add_nc_u32_e32 v14, v7, v3
	v_lshrrev_b32_e32 v3, 1, v12
	v_lshl_add_u32 v12, v13, 8, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v4, 7, v14
	v_and_or_b32 v3, 0xf0, v3, v15
	ds_store_b64 v12, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v4
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_140
; %bb.125:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v14
	s_cbranch_execz .LBB1_139
; %bb.126:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v12, 1, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB1_128
; %bb.127:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v12, v[3:4]
	global_load_u8 v12, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v12, 8, v1
.LBB1_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v12, 2, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB1_130
; %bb.129:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v12, v[3:4]
	global_load_u8 v12, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v12, 16, v1
.LBB1_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v12, 3, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB1_132
; %bb.131:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v12, v[3:4]
	global_load_u8 v12, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v12, 24, v1
.LBB1_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v12, 4, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB1_134
; %bb.133:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v12, v[3:4]
	global_load_u8 v12, v[12:13], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v12, v2
.LBB1_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v12, 5, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB1_136
; %bb.135:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v12, v[3:4]
	global_load_u8 v12, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v12, 8, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v12, v2
.LBB1_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v12, 6, v14
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v12
	s_cbranch_execz .LBB1_138
; %bb.137:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v12, v[3:4]
	v_mov_b16_e32 v12.l, 0
	global_load_d16_hi_u8 v12, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v12, v2
.LBB1_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_139:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $vgpr3
.LBB1_140:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB1_142
; %bb.141:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v14, s[0:1]
	v_mov_b16_e32 v12.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v16.l, v12.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v12, v[1:2], off
	global_load_u8 v13, v[1:2], off offset:3096
	global_load_u8 v14, v[1:2], off offset:5160
	global_load_u8 v17, v[1:2], off offset:4128
	global_load_u8 v18, v[1:2], off offset:7224
	global_load_d16_hi_u8 v16, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v13
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v14
	v_or_b32_e32 v1, v1, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v17, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v18
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v16, v3
	v_or3_b32 v1, v1, 0, 0
.LBB1_142:
	s_or_b32 exec_lo, exec_lo, s2
	v_lshrrev_b32_e32 v12, 5, v11
	v_lshl_add_u32 v10, v10, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_and_b32_e32 v3, 16, v12
	ds_store_b64 v10, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_add_nc_u32_e32 v13, v7, v3
	v_lshrrev_b32_e32 v3, 1, v11
	v_add_nc_u32_e32 v4, 7, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v3, 0xf0, v3, v15
	v_cmpx_le_i32_e64 s5, v4
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_158
; %bb.143:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v13
	s_cbranch_execz .LBB1_157
; %bb.144:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v10, 1, v13
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v13, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_146
; %bb.145:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 8, v1
.LBB1_146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 2, v13
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_148
; %bb.147:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 16, v1
.LBB1_148:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 3, v13
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_150
; %bb.149:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v10, 24, v1
.LBB1_150:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 4, v13
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_152
; %bb.151:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB1_152:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 5, v13
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_154
; %bb.153:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v10, v[3:4]
	global_load_u8 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v10, 8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v10, v2
.LBB1_154:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v10, 6, v13
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_156
; %bb.155:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v10, v[3:4]
	v_mov_b16_e32 v10.l, 0
	global_load_d16_hi_u8 v10, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v10, v2
.LBB1_156:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_157:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr3
.LBB1_158:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB1_160
; %bb.159:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v13, s[0:1]
	v_mov_b16_e32 v10.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v14.l, v10.h
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[1:2], off offset:1032
	global_load_u8 v4, v[1:2], off offset:2064
	global_load_d16_u8 v10, v[1:2], off
	global_load_u8 v11, v[1:2], off offset:3096
	global_load_u8 v13, v[1:2], off offset:5160
	global_load_u8 v16, v[1:2], off offset:4128
	global_load_u8 v17, v[1:2], off offset:7224
	global_load_d16_hi_u8 v14, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v11
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v13
	v_or_b32_e32 v1, v1, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v16, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v17
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v14, v3
	v_or3_b32 v1, v1, 0, 0
.LBB1_160:
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v11, 39, v7
	v_lshl_add_u32 v3, v12, 8, v5
	v_add_nc_u32_e32 v10, 32, v7
	s_mov_b32 s2, exec_lo
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v11
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_176
; %bb.161:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_175
; %bb.162:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v9, 1, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_164
; %bb.163:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB1_164:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 2, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_166
; %bb.165:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB1_166:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 3, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_168
; %bb.167:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB1_168:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 4, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_170
; %bb.169:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_170:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 5, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_172
; %bb.171:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB1_172:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 6, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_174
; %bb.173:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_174:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_175:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr9
.LBB1_176:
	s_or_saveexec_b32 s2, s2
	v_or_b32_e32 v3, 32, v8
	s_xor_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB1_178
; %bb.177:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, s[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_mov_b16_e32 v9.h, 0
	s_clause 0x4
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v8, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v12, v[1:2], off offset:3096
	global_load_u8 v13, v[1:2], off offset:5160
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v14.l, v9.h
	s_clause 0x2
	global_load_u8 v16, v[1:2], off offset:4128
	global_load_u8 v17, v[1:2], off offset:7224
	global_load_d16_hi_u8 v14, v[1:2], off offset:6192
	v_lshlrev_b32_e32 v1, 8, v4
	v_lshlrev_b32_e32 v2, 16, v8
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v12
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v8, 8, v13
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v16, v8
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v17
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v14, v4
	v_or3_b32 v1, v1, 0, 0
.LBB1_178:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x480, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v8
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0x70, v4, v15
	v_cmpx_le_i32_e64 s5, v11
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_194
; %bb.179:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_193
; %bb.180:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v9, 1, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_182
; %bb.181:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB1_182:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 2, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_184
; %bb.183:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB1_184:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 3, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_186
; %bb.185:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB1_186:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 4, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_188
; %bb.187:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_188:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 5, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_190
; %bb.189:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB1_190:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 6, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_192
; %bb.191:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_192:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_193:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr4
.LBB1_194:
	s_or_saveexec_b32 s2, s2
	v_lshrrev_b32_e32 v3, 5, v8
	s_xor_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB1_196
; %bb.195:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v14.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v8, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v12, v[1:2], off offset:3096
	global_load_u8 v13, v[1:2], off offset:5160
	global_load_u8 v16, v[1:2], off offset:4128
	global_load_u8 v17, v[1:2], off offset:7224
	global_load_d16_hi_u8 v14, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v8
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v12
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v8, 8, v13
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v16, v8
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v17
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v14, v4
	v_or3_b32 v1, v1, 0, 0
.LBB1_196:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x500, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v8
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0xb0, v4, v15
	v_cmpx_le_i32_e64 s5, v11
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_212
; %bb.197:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_211
; %bb.198:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v9, 1, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_200
; %bb.199:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB1_200:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 2, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_202
; %bb.201:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB1_202:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 3, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_204
; %bb.203:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB1_204:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 4, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_206
; %bb.205:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_206:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 5, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_208
; %bb.207:
	v_mad_co_i64_i32 v[12:13], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[12:13], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB1_208:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 6, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_210
; %bb.209:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_210:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_211:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr4
.LBB1_212:
	s_or_saveexec_b32 s2, s2
	v_lshrrev_b32_e32 v3, 5, v8
	s_xor_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB1_214
; %bb.213:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v14.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v8, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v12, v[1:2], off offset:3096
	global_load_u8 v13, v[1:2], off offset:5160
	global_load_u8 v16, v[1:2], off offset:4128
	global_load_u8 v17, v[1:2], off offset:7224
	global_load_d16_hi_u8 v14, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v8
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v12
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v8, 8, v13
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v16, v8
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v17
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v14, v4
	v_or3_b32 v1, v1, 0, 0
.LBB1_214:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x580, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v8
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0xf0, v4, v15
	v_cmpx_le_i32_e64 s5, v11
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_230
; %bb.215:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v10
	s_cbranch_execz .LBB1_229
; %bb.216:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v9, 1, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_218
; %bb.217:
	v_mad_co_i64_i32 v[11:12], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB1_218:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 2, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_220
; %bb.219:
	v_mad_co_i64_i32 v[11:12], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB1_220:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 3, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_222
; %bb.221:
	v_mad_co_i64_i32 v[11:12], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB1_222:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 4, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_224
; %bb.223:
	v_mad_co_i64_i32 v[11:12], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[11:12], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_224:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 5, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_226
; %bb.225:
	v_mad_co_i64_i32 v[11:12], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB1_226:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 6, v10
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_228
; %bb.227:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_228:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_229:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr10
                                        ; implicit-def: $vgpr4
.LBB1_230:
	s_or_saveexec_b32 s2, s2
	v_lshrrev_b32_e32 v3, 5, v8
	s_xor_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB1_232
; %bb.231:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v10, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v12.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v8, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v10, v[1:2], off offset:3096
	global_load_u8 v11, v[1:2], off offset:5160
	global_load_u8 v13, v[1:2], off offset:4128
	global_load_u8 v14, v[1:2], off offset:7224
	global_load_d16_hi_u8 v12, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v8
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v10
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v8, 8, v11
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v13, v8
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v14
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v12, v4
	v_or3_b32 v1, v1, 0, 0
.LBB1_232:
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v8, 55, v7
	v_lshl_add_u32 v3, v3, 8, v5
	v_or_b32_e32 v9, 0x600, v0
	v_and_or_b32 v4, v6, 48, v15
	v_add_nc_u32_e32 v6, 48, v7
	s_mov_b32 s2, exec_lo
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_cmpx_le_i32_e64 s5, v8
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_248
; %bb.233:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB1_247
; %bb.234:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v7, 1, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_236
; %bb.235:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v7, v[3:4]
	global_load_u8 v7, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v7, 8, v1
.LBB1_236:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v7, 2, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_238
; %bb.237:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v7, v[3:4]
	global_load_u8 v7, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v7, 16, v1
.LBB1_238:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v7, 3, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_240
; %bb.239:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v7, v[3:4]
	global_load_u8 v7, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v7, 24, v1
.LBB1_240:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v7, 4, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_242
; %bb.241:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v7, v[3:4]
	global_load_u8 v7, v[10:11], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB1_242:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v7, 5, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_244
; %bb.243:
	v_mad_co_i64_i32 v[10:11], null, 0x408, v7, v[3:4]
	global_load_u8 v7, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v7, 8, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v7, v2
.LBB1_244:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v7, 6, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_246
; %bb.245:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v7, v[3:4]
	v_mov_b16_e32 v7.l, 0
	global_load_d16_hi_u8 v7, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v7, v2
.LBB1_246:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_247:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr4
.LBB1_248:
	s_or_saveexec_b32 s2, s2
	v_lshrrev_b32_e32 v3, 5, v9
	s_xor_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB1_250
; %bb.249:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v12.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v10, v[1:2], off offset:3096
	global_load_u8 v11, v[1:2], off offset:5160
	global_load_u8 v13, v[1:2], off offset:4128
	global_load_u8 v14, v[1:2], off offset:7224
	global_load_d16_hi_u8 v12, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v10
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v11
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v13, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v14
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v12, v4
	v_or3_b32 v1, v1, 0, 0
.LBB1_250:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v7, 0x680, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v7
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0x70, v4, v15
	v_cmpx_le_i32_e64 s5, v8
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_266
; %bb.251:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB1_265
; %bb.252:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v9, 1, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_254
; %bb.253:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB1_254:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 2, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_256
; %bb.255:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB1_256:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 3, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_258
; %bb.257:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB1_258:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 4, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_260
; %bb.259:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_260:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 5, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_262
; %bb.261:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB1_262:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 6, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_264
; %bb.263:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_264:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_265:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr4
.LBB1_266:
	s_or_saveexec_b32 s2, s2
	v_lshrrev_b32_e32 v3, 5, v7
	s_xor_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB1_268
; %bb.267:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v12.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v10, v[1:2], off offset:3096
	global_load_u8 v11, v[1:2], off offset:5160
	global_load_u8 v13, v[1:2], off offset:4128
	global_load_u8 v14, v[1:2], off offset:7224
	global_load_d16_hi_u8 v12, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v10
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v11
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v13, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v14
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v12, v4
	v_or3_b32 v1, v1, 0, 0
.LBB1_268:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v7, 0x700, v0
	v_lshl_add_u32 v3, v3, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v4, 1, v7
	ds_store_b64 v3, v[1:2] offset:16384
                                        ; implicit-def: $vgpr1_vgpr2
	v_and_or_b32 v4, 0xb0, v4, v15
	v_cmpx_le_i32_e64 s5, v8
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_284
; %bb.269:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB1_283
; %bb.270:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s4, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s1, 0, s4
	v_add_nc_u32_e32 v9, 1, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, v[3:4]
	global_load_d16_u8 v1, v[1:2], off
	v_mov_b32_e32 v2, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.h, 0
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_272
; %bb.271:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 8, v1
.LBB1_272:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 2, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_274
; %bb.273:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 16, v1
.LBB1_274:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 3, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_276
; %bb.275:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v1, v9, 24, v1
.LBB1_276:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 4, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_278
; %bb.277:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_278:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 5, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_280
; %bb.279:
	v_mad_co_i64_i32 v[9:10], null, 0x408, v9, v[3:4]
	global_load_u8 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v2, v9, v2
.LBB1_280:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v9, 6, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB1_282
; %bb.281:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[3:4]
	v_mov_b16_e32 v9.l, 0
	global_load_d16_hi_u8 v9, v[3:4], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v2, v9, v2
.LBB1_282:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_283:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr4
.LBB1_284:
	s_or_saveexec_b32 s2, s2
	v_lshrrev_b32_e32 v3, 5, v7
	s_xor_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB1_286
; %bb.285:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x408, v6, s[0:1]
	v_mov_b16_e32 v9.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v12.l, v9.h
	v_add_co_u32 v1, vcc_lo, v1, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x7
	global_load_u8 v4, v[1:2], off offset:1032
	global_load_u8 v7, v[1:2], off offset:2064
	global_load_d16_u8 v9, v[1:2], off
	global_load_u8 v10, v[1:2], off offset:3096
	global_load_u8 v11, v[1:2], off offset:5160
	global_load_u8 v13, v[1:2], off offset:4128
	global_load_u8 v14, v[1:2], off offset:7224
	global_load_d16_hi_u8 v12, v[1:2], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v1, 8, v4
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v7
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v4, 24, v10
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v7, 8, v11
	v_or_b32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v1, v1, v2, v4
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v13, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 24, v14
	v_or3_b32 v1, v1, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v2, v2, v12, v4
	v_or3_b32 v1, v1, 0, 0
.LBB1_286:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v4, 0x780, v0
	v_lshl_add_u32 v7, v3, 8, v5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v0, 1, v4
	ds_store_b64 v7, v[1:2] offset:16384
	v_and_or_b32 v3, 0xf0, v0, v15
                                        ; implicit-def: $vgpr0_vgpr1
	v_cmpx_le_i32_e64 s5, v8
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB1_302
; %bb.287:
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB1_301
; %bb.288:
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, s4, s0, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s1, 0, s4
	v_add_nc_u32_e32 v7, 1, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[0:1], null, 0x408, v6, v[2:3]
	global_load_d16_u8 v0, v[0:1], off
	v_mov_b32_e32 v1, 0
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v0.h, 0
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_290
; %bb.289:
	v_mad_co_i64_i32 v[7:8], null, 0x408, v7, v[2:3]
	global_load_u8 v7, v[7:8], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v0, v7, 8, v0
.LBB1_290:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v7, 2, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_292
; %bb.291:
	v_mad_co_i64_i32 v[7:8], null, 0x408, v7, v[2:3]
	global_load_u8 v7, v[7:8], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v0, v7, 16, v0
.LBB1_292:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v7, 3, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_294
; %bb.293:
	v_mad_co_i64_i32 v[7:8], null, 0x408, v7, v[2:3]
	global_load_u8 v7, v[7:8], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v0, v7, 24, v0
.LBB1_294:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v7, 4, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_296
; %bb.295:
	v_mad_co_i64_i32 v[7:8], null, 0x408, v7, v[2:3]
	global_load_u8 v7, v[7:8], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v1, v7, v1
.LBB1_296:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v7, 5, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB1_298
; %bb.297:
	v_mad_co_i64_i32 v[7:8], null, 0x408, v7, v[2:3]
	global_load_u8 v7, v[7:8], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v7, 8, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v1, v7, v1
.LBB1_298:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v6, 6, v6
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v6
	s_cbranch_execz .LBB1_300
; %bb.299:
	v_mad_co_i64_i32 v[2:3], null, 0x408, v6, v[2:3]
	v_mov_b16_e32 v6.l, 0
	global_load_d16_hi_u8 v6, v[2:3], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v1, v6, v1
.LBB1_300:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB1_301:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr3
.LBB1_302:
	s_or_saveexec_b32 s2, s2
	v_lshrrev_b32_e32 v2, 5, v4
	s_xor_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB1_304
; %bb.303:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[0:1], null, 0x408, v6, s[0:1]
	v_mov_b16_e32 v6.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v9.l, v6.h
	v_add_co_u32 v0, vcc_lo, v0, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_clause 0x7
	global_load_u8 v3, v[0:1], off offset:1032
	global_load_u8 v4, v[0:1], off offset:2064
	global_load_d16_u8 v6, v[0:1], off
	global_load_u8 v7, v[0:1], off offset:3096
	global_load_u8 v8, v[0:1], off offset:5160
	global_load_u8 v10, v[0:1], off offset:4128
	global_load_u8 v11, v[0:1], off offset:7224
	global_load_d16_hi_u8 v9, v[0:1], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v0, 8, v3
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v1, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v7
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v8
	v_or_b32_e32 v0, v0, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v0, v0, v1, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v1, 0, v10, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v11
	v_or3_b32 v0, v0, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v1, v1, v9, v3
	v_or3_b32 v0, v0, 0, 0
.LBB1_304:
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v2, v2, 8, v5
	ds_store_b64 v2, v[0:1] offset:16384
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end1:
	.size	t_old_fill, .Lfunc_end1-t_old_fill
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel t_old_fill
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
		.amdhsa_next_free_vgpr 25
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-t_old_fill)<<4)&4080)>>4
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
	.set .Lt_old_fill.num_vgpr, 25
	.set .Lt_old_fill.num_agpr, 0
	.set .Lt_old_fill.numbered_sgpr, 8
	.set .Lt_old_fill.num_named_barrier, 0
	.set .Lt_old_fill.private_seg_size, 0
	.set .Lt_old_fill.uses_vcc, 1
	.set .Lt_old_fill.uses_flat_scratch, 0
	.set .Lt_old_fill.has_dyn_sized_stack, 0
	.set .Lt_old_fill.has_recursion, 0
	.set .Lt_old_fill.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14452
; TotalNumSgprs: 10
; NumVgprs: 25
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 25
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	t_new_fill_dump         ; -- Begin function t_new_fill_dump
	.globl	t_new_fill_dump
	.p2align	8
	.type	t_new_fill_dump,@function
t_new_fill_dump:                        ; @t_new_fill_dump
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_sub_nc_u32_e32 v1, 0x303f, v0
	s_load_b128 s[4:7], s[0:1], 0x18
	s_mov_b32 s2, 0
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v1, 8, v1
	v_add_nc_u32_e32 v1, 1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v1, v0 :: v_dual_and_b32 v4, 7, v1
	v_cmpx_ne_u32_e32 0, v4
	s_cbranch_execz .LBB2_4
; %bb.1:
	v_lshl_add_u32 v2, v0, 2, 0
	v_lshlrev_b32_e32 v3, 2, v4
	v_lshl_or_b32 v1, v4, 8, v0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v4, s7
	s_mov_b32 s8, 0
.LBB2_2:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v3, -4, v3
	ds_store_b32 v2, v4
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_eq_u32_e32 vcc_lo, 0, v3
	s_or_b32 s8, vcc_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB2_2
; %bb.3:
	s_or_b32 exec_lo, exec_lo, s8
.LBB2_4:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v2, s7
.LBB2_5:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v4, 0x800, v1
	v_cmp_lt_u32_e32 vcc_lo, 0x283f, v1
	v_lshl_add_u32 v3, v1, 2, 0
	ds_store_2addr_stride64_b32 v3, v2, v2 offset0:16 offset1:20
	ds_store_2addr_stride64_b32 v3, v2, v2 offset0:24 offset1:28
	v_mov_b32_e32 v1, v4
	s_or_b32 s2, vcc_lo, s2
	ds_store_2addr_stride64_b32 v3, v2, v2 offset1:4
	ds_store_2addr_stride64_b32 v3, v2, v2 offset0:8 offset1:12
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB2_5
; %bb.6:
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b128 s[8:11], s[0:1], 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v1, 15, v0
	v_lshrrev_b32_e32 v2, 1, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_ashr_i32 s7, s6, 31
	v_lshrrev_b32_e32 v14, 5, v0
	v_and_or_b32 v1, v2, 48, v1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[12:13], s[6:7], 8
	v_mov_b32_e32 v4, 0
	v_bfe_u32 v13, v0, 4, 1
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v2, 0
	v_add_nc_u32_e32 v1, s4, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s5, v1
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, 0x408, v1, s[8:9]
	v_mov_b32_e32 v1, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_add_co_u32 v15, s2, v5, s12
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v16, null, s13, v6, s2
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB2_8
; %bb.7:
	v_and_or_b32 v1, v14, 4, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 3, v1
	v_add_co_u32 v3, s2, v15, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v16, s2
	s_clause 0x1
	global_load_b64 v[1:2], v[3:4], off
	global_load_b64 v[3:4], v[3:4], off offset:16
.LBB2_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_load_b64 s[2:3], s[0:1], 0x10
	v_lshl_add_u32 v10, v0, 4, 0
	v_or_b32_e32 v9, 0x100, v0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v6, 0
	v_mov_b32_e32 v5, 0
	s_wait_loadcnt 0x0
	ds_store_b128 v10, v[1:4]
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB2_10
; %bb.9:
	v_lshrrev_b32_e32 v1, 5, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, v1, 12, v13
	v_lshlrev_b32_e32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s0, v15, v1
	v_add_co_ci_u32_e64 v2, null, 0, v16, s0
	s_clause 0x1
	global_load_b64 v[5:6], v[1:2], off
	global_load_b64 v[7:8], v[1:2], off offset:16
.LBB2_10:
	s_or_b32 exec_lo, exec_lo, s1
	v_dual_mov_b32 v4, 0 :: v_dual_and_b32 v17, 0x7f, v0
	v_or_b32_e32 v1, 0x200, v0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v11, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v2, 0x180, v9, v17
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v9, 0
	v_lshl_add_u32 v2, v2, 4, 0
	s_wait_loadcnt 0x0
	ds_store_b128 v2, v[5:8]
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB2_12
; %bb.11:
	v_lshrrev_b32_e32 v2, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v2, v2, 28, v13
	v_lshlrev_b32_e32 v2, 3, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v2, s0, v15, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, 0, v16, s0
	s_clause 0x1
	global_load_b64 v[9:10], v[2:3], off
	global_load_b64 v[11:12], v[2:3], off offset:16
.LBB2_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_and_or_b32 v1, 0x380, v1, v17
	v_or_b32_e32 v5, 0x300, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v2, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_add_u32 v6, v1, 4, 0
	v_mov_b32_e32 v1, 0
	s_wait_loadcnt 0x0
	ds_store_b128 v6, v[9:12]
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_14
; %bb.13:
	v_lshrrev_b32_e32 v1, 5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, v1, 28, v13
	v_lshlrev_b32_e32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v3, vcc_lo, v15, v1
	v_add_co_ci_u32_e64 v4, null, 0, v16, vcc_lo
	s_clause 0x1
	global_load_b64 v[1:2], v[3:4], off
	global_load_b64 v[3:4], v[3:4], off offset:16
.LBB2_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v13, 31, v0
	v_and_or_b32 v5, 0x380, v5, v17
	s_add_nc_u64 s[0:1], s[10:11], s[12:13]
	v_lshl_add_u32 v15, v14, 3, s4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v7, 0 :: v_dual_lshlrev_b32 v6, 3, v13
	v_lshl_add_u32 v11, v5, 4, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v10, 0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v5, s0, s0, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s1, 0, s0
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v11, v[1:4]
	v_cmpx_gt_i32_e64 s5, v15
	s_cbranch_execz .LBB2_16
; %bb.15:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v15, v[5:6]
	global_load_b64 v[9:10], v[1:2], off
.LBB2_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshlrev_b32_e32 v11, 11, v14
	v_add_nc_u32_e32 v1, 1, v15
	v_lshlrev_b32_e32 v14, 1, v13
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v12, 0, v11
	v_lshl_add_u32 v16, v13, 3, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x8000, v16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v2, v9, v10 offset1:1
	v_cmpx_gt_i32_e64 s5, v1
	s_cbranch_execz .LBB2_18
; %bb.17:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[5:6]
	global_load_b64 v[7:8], v[1:2], off
.LBB2_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v10, 0x48, v14
	v_xor_b32_e32 v17, 0x49, v14
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v9, 2, v15
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v3, 0
	v_lshl_add_u32 v10, v10, 2, v12
	v_lshl_add_u32 v17, v17, 2, v12
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v10, v7 offset:32768
	ds_store_b32 v17, v8 offset:32768
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB2_20
; %bb.19:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[5:6]
	global_load_b64 v[3:4], v[3:4], off
.LBB2_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v8, 0x90, v14
	v_xor_b32_e32 v9, 0x91, v14
	v_add_nc_u32_e32 v7, 3, v15
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v8, v8, 2, v12
	v_lshl_add_u32 v9, v9, 2, v12
	s_wait_loadcnt 0x0
	ds_store_b32 v8, v3 offset:32768
	ds_store_b32 v9, v4 offset:32768
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB2_22
; %bb.21:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[5:6]
	global_load_b64 v[1:2], v[1:2], off
.LBB2_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v10, 0xd8, v14
	v_xor_b32_e32 v17, 0xd9, v14
	v_dual_mov_b32 v4, 0 :: v_dual_add_nc_u32 v9, 4, v15
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v8, 0
	v_mov_b32_e32 v7, 0
	v_lshl_add_u32 v10, v10, 2, v12
	v_lshl_add_u32 v17, v17, 2, v12
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v10, v1 offset:32768
	ds_store_b32 v17, v2 offset:32768
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB2_24
; %bb.23:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v9, v[5:6]
	global_load_b64 v[7:8], v[1:2], off
.LBB2_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_nc_u32_e32 v1, 5, v15
	v_add_nc_u32_e32 v2, 0x8400, v16
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v2, v7, v8 offset1:1
	v_cmpx_gt_i32_e64 s5, v1
	s_cbranch_execz .LBB2_26
; %bb.25:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[5:6]
	global_load_b64 v[3:4], v[1:2], off
.LBB2_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v10, 0x148, v14
	v_xor_b32_e32 v16, 0x149, v14
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v9, 6, v15
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v8, 0
	v_mov_b32_e32 v7, 0
	v_lshl_add_u32 v10, v10, 2, v12
	v_lshl_add_u32 v16, v16, 2, v12
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v10, v3 offset:32768
	ds_store_b32 v16, v4 offset:32768
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB2_28
; %bb.27:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[5:6]
	global_load_b64 v[7:8], v[3:4], off
.LBB2_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v4, 0x190, v14
	v_xor_b32_e32 v9, 0x191, v14
	v_add_nc_u32_e32 v3, 7, v15
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v4, v4, 2, v12
	v_lshl_add_u32 v9, v9, 2, v12
	s_wait_loadcnt 0x0
	ds_store_b32 v4, v7 offset:32768
	ds_store_b32 v9, v8 offset:32768
	v_cmpx_gt_i32_e64 s5, v3
	s_cbranch_execz .LBB2_30
; %bb.29:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[5:6]
	global_load_b64 v[1:2], v[1:2], off
.LBB2_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v3, 0x1d8, v14
	v_xor_b32_e32 v4, 0x1d9, v14
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v3, v3, 2, v12
	v_lshl_add_u32 v4, v4, 2, v12
	s_wait_loadcnt 0x0
	ds_store_b32 v3, v1 offset:32768
	ds_store_b32 v4, v2 offset:32768
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_lshrrev_b32_e32 v1, 4, v13
	v_bfe_u32 v2, v0, 5, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_eq_u32_e64 v1, v2
	s_cbranch_execz .LBB2_32
; %bb.31:
	v_mbcnt_lo_u32_b32 v9, -1, 0
	v_lshrrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v19, 1, v0
	v_lshl_add_u32 v13, v13, 4, 0
	v_mov_b32_e32 v20, 0x6020400
	v_xor_b32_e32 v10, 1, v9
	v_xor_b32_e32 v18, 2, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v10, v9, v10 :: v_dual_and_b32 v5, 3, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v9, v9, v18 :: v_dual_lshlrev_b32 v2, 6, v5
	v_and_or_b32 v6, v1, 3, v2
	v_lshlrev_b32_e32 v1, 5, v5
	v_cmp_eq_u32_e32 vcc_lo, 0, v19
	v_mov_b32_e32 v19, 0x5040100
	v_lshlrev_b32_e32 v9, 2, v9
	v_lshlrev_b32_e32 v2, 2, v6
	v_or_b32_e32 v21, 8, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v18, 0x3070105, v20, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 2, v5
	v_lshlrev_b32_e32 v10, 2, v10
	v_add3_u32 v1, v12, v2, v1
	v_lshlrev_b32_e32 v5, 3, v5
	v_or_b32_e32 v22, 0x108, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, 0x3020706, v19, vcc_lo
	v_or_b32_e32 v23, 12, v6
	v_add_nc_u32_e32 v7, 0x8000, v1
	v_add_nc_u32_e32 v8, 0x8400, v1
	v_or_b32_e32 v24, 0x10c, v6
	ds_load_2addr_b32 v[1:2], v7 offset1:4
	ds_load_2addr_b32 v[3:4], v8 offset1:4
	v_and_b32_e32 v20, 0x3000, v11
	v_xor_b32_e32 v21, v21, v5
	v_xor_b32_e32 v22, v22, v5
	v_xor_b32_e32 v23, v23, v5
	v_xor_b32_e32 v24, v24, v5
	v_add_nc_u32_e32 v20, v13, v20
	s_wait_dscnt 0x1
	ds_bpermute_b32 v14, v10, v1
	s_wait_dscnt 0x1
	ds_bpermute_b32 v15, v10, v3
	ds_bpermute_b32 v16, v10, v2
	ds_bpermute_b32 v17, v10, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v14, v1, v18
	s_wait_dscnt 0x2
	v_perm_b32 v3, v15, v3, v18
	s_wait_dscnt 0x1
	v_perm_b32 v14, v16, v2, v18
	s_wait_dscnt 0x0
	v_perm_b32 v4, v17, v4, v18
	ds_bpermute_b32 v2, v9, v1
	ds_bpermute_b32 v15, v9, v3
	ds_bpermute_b32 v16, v9, v14
	ds_bpermute_b32 v17, v9, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v2, v1, v19
	s_wait_dscnt 0x2
	v_perm_b32 v2, v15, v3, v19
	s_wait_dscnt 0x1
	v_perm_b32 v3, v16, v14, v19
	s_wait_dscnt 0x0
	v_perm_b32 v4, v17, v4, v19
	v_lshl_add_u32 v14, v21, 2, v12
	v_lshl_add_u32 v15, v22, 2, v12
	v_lshl_add_u32 v16, v23, 2, v12
	v_lshl_add_u32 v17, v24, 2, v12
	ds_store_b128 v20, v[1:4] offset:16384
	ds_load_b32 v1, v14 offset:32768
	ds_load_b32 v2, v15 offset:32768
	ds_load_b32 v3, v16 offset:32768
	ds_load_b32 v4, v17 offset:32768
	v_or_b32_e32 v21, 16, v6
	v_or_b32_e32 v22, 0x110, v6
	v_or_b32_e32 v23, 20, v6
	v_or_b32_e32 v24, 0x114, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v21, v21, v5
	v_xor_b32_e32 v22, v22, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v23, v23, v5
	v_xor_b32_e32 v24, v24, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v14, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v10, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v10, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v14, v1, v18
	s_wait_dscnt 0x2
	v_perm_b32 v2, v15, v2, v18
	s_wait_dscnt 0x1
	v_perm_b32 v3, v16, v3, v18
	s_wait_dscnt 0x0
	v_perm_b32 v4, v17, v4, v18
	ds_bpermute_b32 v14, v9, v1
	ds_bpermute_b32 v15, v9, v2
	ds_bpermute_b32 v16, v9, v3
	ds_bpermute_b32 v17, v9, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v14, v1, v19
	s_wait_dscnt 0x2
	v_perm_b32 v2, v15, v2, v19
	s_wait_dscnt 0x1
	v_perm_b32 v3, v16, v3, v19
	s_wait_dscnt 0x0
	v_perm_b32 v4, v17, v4, v19
	v_lshl_add_u32 v14, v21, 2, v12
	v_lshl_add_u32 v15, v22, 2, v12
	v_lshl_add_u32 v16, v23, 2, v12
	v_lshl_add_u32 v17, v24, 2, v12
	ds_store_b128 v20, v[1:4] offset:16896
	ds_load_b32 v1, v14 offset:32768
	ds_load_b32 v2, v15 offset:32768
	ds_load_b32 v3, v16 offset:32768
	ds_load_b32 v4, v17 offset:32768
	v_or_b32_e32 v21, 24, v6
	v_or_b32_e32 v22, 0x118, v6
	v_or_b32_e32 v23, 28, v6
	v_or_b32_e32 v24, 0x11c, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v21, v21, v5
	v_xor_b32_e32 v22, v22, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v23, v23, v5
	v_xor_b32_e32 v24, v24, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v14, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v10, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v10, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v14, v1, v18
	s_wait_dscnt 0x2
	v_perm_b32 v2, v15, v2, v18
	s_wait_dscnt 0x1
	v_perm_b32 v3, v16, v3, v18
	s_wait_dscnt 0x0
	v_perm_b32 v4, v17, v4, v18
	ds_bpermute_b32 v14, v9, v1
	ds_bpermute_b32 v15, v9, v2
	ds_bpermute_b32 v16, v9, v3
	ds_bpermute_b32 v17, v9, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v14, v1, v19
	s_wait_dscnt 0x2
	v_perm_b32 v2, v15, v2, v19
	s_wait_dscnt 0x1
	v_perm_b32 v3, v16, v3, v19
	s_wait_dscnt 0x0
	v_perm_b32 v4, v17, v4, v19
	v_lshl_add_u32 v14, v21, 2, v12
	v_lshl_add_u32 v15, v22, 2, v12
	v_lshl_add_u32 v16, v23, 2, v12
	v_lshl_add_u32 v17, v24, 2, v12
	ds_store_b128 v20, v[1:4] offset:17408
	ds_load_b32 v1, v14 offset:32768
	ds_load_b32 v2, v15 offset:32768
	ds_load_b32 v3, v16 offset:32768
	ds_load_b32 v4, v17 offset:32768
	v_or_b32_e32 v21, 44, v6
	v_or_b32_e32 v22, 0x12c, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v21, v21, v5
	v_xor_b32_e32 v22, v22, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v14, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v10, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v10, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v14, v1, v18
	s_wait_dscnt 0x2
	v_perm_b32 v2, v15, v2, v18
	s_wait_dscnt 0x1
	v_perm_b32 v3, v16, v3, v18
	s_wait_dscnt 0x0
	v_perm_b32 v4, v17, v4, v18
	ds_bpermute_b32 v14, v9, v1
	ds_bpermute_b32 v15, v9, v2
	ds_bpermute_b32 v16, v9, v3
	ds_bpermute_b32 v17, v9, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v14, v1, v19
	s_wait_dscnt 0x2
	v_perm_b32 v2, v15, v2, v19
	s_wait_dscnt 0x1
	v_perm_b32 v3, v16, v3, v19
	s_wait_dscnt 0x0
	v_perm_b32 v4, v17, v4, v19
	v_or_b32_e32 v17, 40, v6
	v_or_b32_e32 v16, 0x800, v11
	ds_store_b128 v20, v[1:4] offset:17920
	ds_load_2addr_b32 v[1:2], v7 offset0:32 offset1:36
	ds_load_2addr_b32 v[3:4], v8 offset0:32 offset1:36
	v_or_b32_e32 v20, 0x128, v6
	v_xor_b32_e32 v17, v17, v5
	v_add_nc_u32_e32 v16, v13, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_xor_b32_e32 v20, v20, v5
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v10, v1
	s_wait_dscnt 0x1
	ds_bpermute_b32 v8, v10, v3
	ds_bpermute_b32 v14, v10, v2
	ds_bpermute_b32 v15, v10, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v7, v1, v18
	s_wait_dscnt 0x2
	v_perm_b32 v3, v8, v3, v18
	s_wait_dscnt 0x1
	v_perm_b32 v7, v14, v2, v18
	s_wait_dscnt 0x0
	v_perm_b32 v4, v15, v4, v18
	ds_bpermute_b32 v2, v9, v1
	ds_bpermute_b32 v8, v9, v3
	ds_bpermute_b32 v14, v9, v7
	ds_bpermute_b32 v15, v9, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v2, v1, v19
	s_wait_dscnt 0x2
	v_perm_b32 v2, v8, v3, v19
	s_wait_dscnt 0x1
	v_perm_b32 v3, v14, v7, v19
	s_wait_dscnt 0x0
	v_perm_b32 v4, v15, v4, v19
	v_lshl_add_u32 v7, v17, 2, v12
	v_lshl_add_u32 v8, v20, 2, v12
	v_lshl_add_u32 v14, v21, 2, v12
	v_lshl_add_u32 v15, v22, 2, v12
	ds_store_b128 v16, v[1:4] offset:16384
	ds_load_b32 v1, v7 offset:32768
	ds_load_b32 v2, v8 offset:32768
	ds_load_b32 v3, v14 offset:32768
	ds_load_b32 v4, v15 offset:32768
	v_or_b32_e32 v17, 48, v6
	v_or_b32_e32 v20, 0x130, v6
	v_or_b32_e32 v21, 52, v6
	v_or_b32_e32 v22, 0x134, v6
	v_or_b32_e32 v16, 0xa00, v11
	v_xor_b32_e32 v17, v17, v5
	v_xor_b32_e32 v20, v20, v5
	v_xor_b32_e32 v21, v21, v5
	v_xor_b32_e32 v22, v22, v5
	v_add_nc_u32_e32 v16, v13, v16
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v14, v10, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v10, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v7, v1, v18
	s_wait_dscnt 0x2
	v_perm_b32 v2, v8, v2, v18
	s_wait_dscnt 0x1
	v_perm_b32 v3, v14, v3, v18
	s_wait_dscnt 0x0
	v_perm_b32 v4, v15, v4, v18
	ds_bpermute_b32 v7, v9, v1
	ds_bpermute_b32 v8, v9, v2
	ds_bpermute_b32 v14, v9, v3
	ds_bpermute_b32 v15, v9, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v7, v1, v19
	s_wait_dscnt 0x2
	v_perm_b32 v2, v8, v2, v19
	s_wait_dscnt 0x1
	v_perm_b32 v3, v14, v3, v19
	s_wait_dscnt 0x0
	v_perm_b32 v4, v15, v4, v19
	v_lshl_add_u32 v7, v17, 2, v12
	v_lshl_add_u32 v8, v20, 2, v12
	v_lshl_add_u32 v14, v21, 2, v12
	v_lshl_add_u32 v15, v22, 2, v12
	ds_store_b128 v16, v[1:4] offset:16384
	ds_load_b32 v1, v7 offset:32768
	ds_load_b32 v2, v8 offset:32768
	ds_load_b32 v3, v14 offset:32768
	ds_load_b32 v4, v15 offset:32768
	v_or_b32_e32 v17, 56, v6
	v_or_b32_e32 v20, 0x138, v6
	v_or_b32_e32 v21, 60, v6
	v_or_b32_e32 v6, 0x13c, v6
	v_or_b32_e32 v16, 0xc00, v11
	v_xor_b32_e32 v17, v17, v5
	v_xor_b32_e32 v20, v20, v5
	v_xor_b32_e32 v21, v21, v5
	v_xor_b32_e32 v5, v6, v5
	v_add_nc_u32_e32 v16, v13, v16
	v_lshl_add_u32 v6, v17, 2, v12
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_add_u32 v5, v5, 2, v12
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v14, v10, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v10, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v7, v1, v18
	s_wait_dscnt 0x2
	v_perm_b32 v2, v8, v2, v18
	s_wait_dscnt 0x1
	v_perm_b32 v3, v14, v3, v18
	s_wait_dscnt 0x0
	v_perm_b32 v4, v15, v4, v18
	ds_bpermute_b32 v7, v9, v1
	ds_bpermute_b32 v8, v9, v2
	ds_bpermute_b32 v14, v9, v3
	ds_bpermute_b32 v15, v9, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v7, v1, v19
	s_wait_dscnt 0x2
	v_perm_b32 v2, v8, v2, v19
	s_wait_dscnt 0x1
	v_perm_b32 v3, v14, v3, v19
	s_wait_dscnt 0x0
	v_perm_b32 v4, v15, v4, v19
	v_lshl_add_u32 v7, v20, 2, v12
	v_lshl_add_u32 v8, v21, 2, v12
	ds_store_b128 v16, v[1:4] offset:16384
	ds_load_b32 v1, v6 offset:32768
	ds_load_b32 v2, v7 offset:32768
	ds_load_b32 v3, v8 offset:32768
	ds_load_b32 v4, v5 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v5, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v10, v3
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v10, v4
	s_wait_dscnt 0x3
	v_perm_b32 v1, v5, v1, v18
	s_wait_dscnt 0x2
	v_perm_b32 v2, v6, v2, v18
	s_wait_dscnt 0x1
	v_perm_b32 v3, v7, v3, v18
	s_wait_dscnt 0x0
	v_perm_b32 v4, v8, v4, v18
	ds_bpermute_b32 v5, v9, v1
	ds_bpermute_b32 v6, v9, v2
	ds_bpermute_b32 v7, v9, v3
	ds_bpermute_b32 v8, v9, v4
	v_or_b32_e32 v9, 0xe00, v11
	s_wait_dscnt 0x3
	v_perm_b32 v1, v5, v1, v19
	s_wait_dscnt 0x2
	v_perm_b32 v2, v6, v2, v19
	s_wait_dscnt 0x1
	v_perm_b32 v3, v7, v3, v19
	s_wait_dscnt 0x0
	v_perm_b32 v4, v8, v4, v19
	v_add_nc_u32_e32 v5, v13, v9
	ds_store_b128 v5, v[1:4] offset:16384
.LBB2_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_gt_u32_e32 vcc_lo, 64, v0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB2_36
; %bb.33:
	v_dual_mov_b32 v1, 0 :: v_dual_add_nc_u32 v2, s4, v0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s5, v2
	s_cbranch_execz .LBB2_35
; %bb.34:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v2, s[8:9]
	v_mad_co_i64_i32 v[1:2], null, 0x408, v2, s[10:11]
	s_lshl_b32 s5, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s6, s5, 31
	v_add_co_u32 v3, s0, v3, s5
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s6, v4, s0
	v_add_co_u32 v5, s0, v1, s5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s6, v2, s0
	global_load_d16_b16 v1, v[3:4], off offset:1024
	global_load_d16_hi_b16 v1, v[5:6], off offset:1024
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v2.h, 8, v1.l
	v_lshrrev_b16 v2.l, 8, v1.h
	v_and_b16 v3.h, 0xff, v1.l
	v_and_b16 v3.l, 0xff, v1.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_lshlrev_b16 v1, 8, v2 op_sel_hi:[0,1]
	v_or_b32_e32 v1, v1, v3
.LBB2_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_lshl_add_u32 v2, v0, 1, 0
	ds_store_b16_d16_hi v2, v1 offset:49152
	ds_store_b16 v2, v1 offset:49280
.LBB2_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_lshlrev_b32_e32 v33, 7, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 0, v33
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_b32 v[3:4], v31 offset0:2 offset1:3
	ds_load_2addr_b32 v[7:8], v31 offset0:6 offset1:7
	ds_load_2addr_b32 v[5:6], v31 offset0:4 offset1:5
	ds_load_2addr_b32 v[1:2], v31 offset1:1
	ds_load_2addr_b32 v[11:12], v31 offset0:10 offset1:11
	ds_load_2addr_b32 v[15:16], v31 offset0:14 offset1:15
	ds_load_2addr_b32 v[13:14], v31 offset0:12 offset1:13
	ds_load_2addr_b32 v[9:10], v31 offset0:8 offset1:9
	ds_load_2addr_b32 v[19:20], v31 offset0:18 offset1:19
	ds_load_2addr_b32 v[23:24], v31 offset0:22 offset1:23
	ds_load_2addr_b32 v[21:22], v31 offset0:20 offset1:21
	ds_load_2addr_b32 v[17:18], v31 offset0:16 offset1:17
	ds_load_2addr_b32 v[27:28], v31 offset0:26 offset1:27
	ds_load_2addr_b32 v[29:30], v31 offset0:28 offset1:29
	ds_load_2addr_b32 v[25:26], v31 offset0:24 offset1:25
	ds_load_2addr_b32 v[31:32], v31 offset0:30 offset1:31
	s_wait_dscnt 0xc
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_store_b128 v33, v[1:4], s[2:3]
	global_store_b128 v33, v[5:8], s[2:3] offset:16
	s_wait_dscnt 0x8
	s_clause 0x1
	global_store_b128 v33, v[9:12], s[2:3] offset:32
	global_store_b128 v33, v[13:16], s[2:3] offset:48
	s_wait_dscnt 0x4
	s_clause 0x1
	global_store_b128 v33, v[17:20], s[2:3] offset:64
	global_store_b128 v33, v[21:24], s[2:3] offset:80
	s_wait_dscnt 0x1
	global_store_b128 v33, v[25:28], s[2:3] offset:96
	s_wait_dscnt 0x0
	global_store_b128 v33, v[29:32], s[2:3] offset:112
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_38
; %bb.37:
	v_lshl_add_u32 v2, v0, 1, 0
	v_lshlrev_b32_e32 v0, 2, v0
	ds_load_u16_d16 v1, v2 offset:49152
	ds_load_u16_d16_hi v1, v2 offset:49280
	s_wait_dscnt 0x0
	v_cvt_f32_f16_e32 v2, v1.l
	v_cvt_f32_f16_e32 v1, v1.h
	s_clause 0x1
	global_store_b32 v0, v2, s[2:3] offset:32768
	global_store_b32 v0, v1, s[2:3] offset:33024
.LBB2_38:
	s_endpgm
.Lfunc_end2:
	.size	t_new_fill_dump, .Lfunc_end2-t_new_fill_dump
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel t_new_fill_dump
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
		.amdhsa_next_free_vgpr 34
		.amdhsa_next_free_sgpr 14
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-t_new_fill_dump)<<4)&4080)>>4
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
	.set .Lt_new_fill_dump.num_vgpr, 34
	.set .Lt_new_fill_dump.num_agpr, 0
	.set .Lt_new_fill_dump.numbered_sgpr, 14
	.set .Lt_new_fill_dump.num_named_barrier, 0
	.set .Lt_new_fill_dump.private_seg_size, 0
	.set .Lt_new_fill_dump.uses_vcc, 1
	.set .Lt_new_fill_dump.uses_flat_scratch, 0
	.set .Lt_new_fill_dump.has_dyn_sized_stack, 0
	.set .Lt_new_fill_dump.has_recursion, 0
	.set .Lt_new_fill_dump.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4896
; TotalNumSgprs: 16
; NumVgprs: 34
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 34
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	t_new_fill              ; -- Begin function t_new_fill
	.globl	t_new_fill
	.p2align	8
	.type	t_new_fill,@function
t_new_fill:                             ; @t_new_fill
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b96 s[4:6], s[0:1], 0x10
	s_load_b128 s[0:3], s[0:1], 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v1, 15, v0
	v_lshrrev_b32_e32 v2, 1, v0
	v_mov_b32_e32 v3, 0
	v_lshrrev_b32_e32 v14, 5, v0
	v_bfe_u32 v13, v0, 4, 1
	v_mov_b32_e32 v4, 0
	v_and_or_b32 v1, v2, 48, v1
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v1, s4, v1
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[5:6], null, 0x408, v1, s[0:1]
	v_cmp_gt_i32_e32 vcc_lo, s5, v1
	v_mov_b32_e32 v1, 0
	v_add_co_u32 v15, s0, v5, s6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v16, null, s7, v6, s0
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB3_2
; %bb.1:
	v_and_or_b32 v1, v14, 4, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 3, v1
	v_add_co_u32 v3, s0, v15, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v16, s0
	s_clause 0x1
	global_load_b64 v[1:2], v[3:4], off
	global_load_b64 v[3:4], v[3:4], off offset:16
.LBB3_2:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_lshl_add_u32 v10, v0, 4, 0
	v_or_b32_e32 v9, 0x100, v0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v6, 0
	v_mov_b32_e32 v5, 0
	s_wait_loadcnt 0x0
	ds_store_b128 v10, v[1:4]
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB3_4
; %bb.3:
	v_lshrrev_b32_e32 v1, 5, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, v1, 12, v13
	v_lshlrev_b32_e32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s0, v15, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, 0, v16, s0
	s_clause 0x1
	global_load_b64 v[5:6], v[1:2], off
	global_load_b64 v[7:8], v[1:2], off offset:16
.LBB3_4:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_dual_mov_b32 v4, 0 :: v_dual_and_b32 v17, 0x7f, v0
	v_or_b32_e32 v1, 0x200, v0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v11, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v2, 0x180, v9, v17
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v9, 0
	v_lshl_add_u32 v2, v2, 4, 0
	s_wait_loadcnt 0x0
	ds_store_b128 v2, v[5:8]
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB3_6
; %bb.5:
	v_lshrrev_b32_e32 v2, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v2, v2, 28, v13
	v_lshlrev_b32_e32 v2, 3, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v2, s0, v15, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, 0, v16, s0
	s_clause 0x1
	global_load_b64 v[9:10], v[2:3], off
	global_load_b64 v[11:12], v[2:3], off offset:16
.LBB3_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_and_or_b32 v1, 0x380, v1, v17
	v_or_b32_e32 v5, 0x300, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v2, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_add_u32 v6, v1, 4, 0
	v_mov_b32_e32 v1, 0
	s_wait_loadcnt 0x0
	ds_store_b128 v6, v[9:12]
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB3_8
; %bb.7:
	v_lshrrev_b32_e32 v1, 5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, v1, 28, v13
	v_lshlrev_b32_e32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v3, vcc_lo, v15, v1
	v_add_co_ci_u32_e64 v4, null, 0, v16, vcc_lo
	s_clause 0x1
	global_load_b64 v[1:2], v[3:4], off
	global_load_b64 v[3:4], v[3:4], off offset:16
.LBB3_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v13, 31, v0
	v_and_or_b32 v5, 0x380, v5, v17
	s_add_nc_u64 s[0:1], s[2:3], s[6:7]
	v_lshl_add_u32 v15, v14, 3, s4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v7, 0 :: v_dual_lshlrev_b32 v6, 3, v13
	v_lshl_add_u32 v11, v5, 4, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v10, 0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v5, s0, s0, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s1, 0, s0
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v11, v[1:4]
	v_cmpx_gt_i32_e64 s5, v15
	s_cbranch_execz .LBB3_10
; %bb.9:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v15, v[5:6]
	global_load_b64 v[9:10], v[1:2], off
.LBB3_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshlrev_b32_e32 v11, 11, v14
	v_add_nc_u32_e32 v1, 1, v15
	v_lshlrev_b32_e32 v14, 1, v13
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v12, 0, v11
	v_lshl_add_u32 v16, v13, 3, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x8000, v16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v2, v9, v10 offset1:1
	v_cmpx_gt_i32_e64 s5, v1
	s_cbranch_execz .LBB3_12
; %bb.11:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[5:6]
	global_load_b64 v[7:8], v[1:2], off
.LBB3_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v10, 0x48, v14
	v_xor_b32_e32 v17, 0x49, v14
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v9, 2, v15
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v3, 0
	v_lshl_add_u32 v10, v10, 2, v12
	v_lshl_add_u32 v17, v17, 2, v12
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v10, v7 offset:32768
	ds_store_b32 v17, v8 offset:32768
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB3_14
; %bb.13:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[5:6]
	global_load_b64 v[3:4], v[3:4], off
.LBB3_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v8, 0x90, v14
	v_xor_b32_e32 v9, 0x91, v14
	v_add_nc_u32_e32 v7, 3, v15
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v8, v8, 2, v12
	v_lshl_add_u32 v9, v9, 2, v12
	s_wait_loadcnt 0x0
	ds_store_b32 v8, v3 offset:32768
	ds_store_b32 v9, v4 offset:32768
	v_cmpx_gt_i32_e64 s5, v7
	s_cbranch_execz .LBB3_16
; %bb.15:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v7, v[5:6]
	global_load_b64 v[1:2], v[1:2], off
.LBB3_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v10, 0xd8, v14
	v_xor_b32_e32 v17, 0xd9, v14
	v_dual_mov_b32 v4, 0 :: v_dual_add_nc_u32 v9, 4, v15
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v8, 0
	v_mov_b32_e32 v7, 0
	v_lshl_add_u32 v10, v10, 2, v12
	v_lshl_add_u32 v17, v17, 2, v12
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v10, v1 offset:32768
	ds_store_b32 v17, v2 offset:32768
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB3_18
; %bb.17:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v9, v[5:6]
	global_load_b64 v[7:8], v[1:2], off
.LBB3_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_nc_u32_e32 v1, 5, v15
	v_add_nc_u32_e32 v2, 0x8400, v16
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v2, v7, v8 offset1:1
	v_cmpx_gt_i32_e64 s5, v1
	s_cbranch_execz .LBB3_20
; %bb.19:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v1, v[5:6]
	global_load_b64 v[3:4], v[1:2], off
.LBB3_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v10, 0x148, v14
	v_xor_b32_e32 v16, 0x149, v14
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v9, 6, v15
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v8, 0
	v_mov_b32_e32 v7, 0
	v_lshl_add_u32 v10, v10, 2, v12
	v_lshl_add_u32 v16, v16, 2, v12
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b32 v10, v3 offset:32768
	ds_store_b32 v16, v4 offset:32768
	v_cmpx_gt_i32_e64 s5, v9
	s_cbranch_execz .LBB3_22
; %bb.21:
	v_mad_co_i64_i32 v[3:4], null, 0x408, v9, v[5:6]
	global_load_b64 v[7:8], v[3:4], off
.LBB3_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v4, 0x190, v14
	v_xor_b32_e32 v9, 0x191, v14
	v_add_nc_u32_e32 v3, 7, v15
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v4, v4, 2, v12
	v_lshl_add_u32 v9, v9, 2, v12
	s_wait_loadcnt 0x0
	ds_store_b32 v4, v7 offset:32768
	ds_store_b32 v9, v8 offset:32768
	v_cmpx_gt_i32_e64 s5, v3
	s_cbranch_execz .LBB3_24
; %bb.23:
	v_mad_co_i64_i32 v[1:2], null, 0x408, v3, v[5:6]
	global_load_b64 v[1:2], v[1:2], off
.LBB3_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_xor_b32_e32 v3, 0x1d8, v14
	v_xor_b32_e32 v4, 0x1d9, v14
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v3, v3, 2, v12
	v_lshl_add_u32 v4, v4, 2, v12
	s_wait_loadcnt 0x0
	ds_store_b32 v3, v1 offset:32768
	ds_store_b32 v4, v2 offset:32768
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_lshrrev_b32_e32 v1, 4, v13
	v_bfe_u32 v2, v0, 5, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_eq_u32_e64 v1, v2
	s_cbranch_execz .LBB3_26
; %bb.25:
	v_mbcnt_lo_u32_b32 v9, -1, 0
	v_lshrrev_b32_e32 v1, 2, v0
	v_lshl_add_u32 v13, v13, 4, 0
	v_mov_b32_e32 v19, 0x6020400
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v10, 1, v9
	v_xor_b32_e32 v18, 2, v9
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v10, v9, v10 :: v_dual_and_b32 v5, 3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v2, 6, v5
	v_cmp_gt_u32_e32 vcc_lo, 32, v18
	v_and_b32_e32 v0, 1, v0
	v_and_or_b32 v6, v1, 3, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v9, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	v_lshlrev_b32_e32 v1, 5, v5
	v_lshlrev_b32_e32 v2, 2, v6
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v18, 0x3070105, v19 :: v_dual_lshlrev_b32 v9, 2, v9
	v_cmp_gt_u32_e32 vcc_lo, 2, v5
	v_lshlrev_b32_e32 v10, 2, v10
	v_add3_u32 v1, v12, v2, v1
	v_lshlrev_b32_e32 v5, 3, v5
	v_or_b32_e32 v20, 8, v6
	v_or_b32_e32 v21, 0x108, v6
	v_or_b32_e32 v22, 12, v6
	v_add_nc_u32_e32 v7, 0x8000, v1
	v_add_nc_u32_e32 v8, 0x8400, v1
	v_or_b32_e32 v23, 0x10c, v6
	v_and_b32_e32 v19, 0x3000, v11
	ds_load_2addr_b32 v[1:2], v7 offset1:4
	ds_load_2addr_b32 v[3:4], v8 offset1:4
	v_xor_b32_e32 v20, v20, v5
	v_xor_b32_e32 v21, v21, v5
	v_xor_b32_e32 v22, v22, v5
	v_xor_b32_e32 v23, v23, v5
	v_add_nc_u32_e32 v19, v13, v19
	s_wait_dscnt 0x1
	ds_bpermute_b32 v14, v10, v1
	s_wait_dscnt 0x1
	ds_bpermute_b32 v15, v10, v3
	ds_bpermute_b32 v17, v10, v4
	s_wait_dscnt 0x2
	v_perm_b32 v0, v14, v1, v18
	s_wait_dscnt 0x1
	v_perm_b32 v1, v15, v3, v18
	s_wait_dscnt 0x0
	v_perm_b32 v3, v17, v4, v18
	v_mov_b32_e32 v17, 0x5040100
	ds_bpermute_b32 v4, v9, v0
	ds_bpermute_b32 v14, v9, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, 0x3020706, v17, vcc_lo
	ds_bpermute_b32 v16, v10, v2
	s_wait_dscnt 0x2
	v_perm_b32 v0, v4, v0, v17
	s_wait_dscnt 0x1
	v_perm_b32 v1, v14, v1, v17
	v_lshl_add_u32 v4, v20, 2, v12
	v_lshl_add_u32 v14, v21, 2, v12
	s_wait_dscnt 0x0
	v_perm_b32 v2, v16, v2, v18
	ds_bpermute_b32 v16, v9, v3
	v_or_b32_e32 v20, 16, v6
	v_or_b32_e32 v21, 0x110, v6
	ds_bpermute_b32 v15, v9, v2
	v_xor_b32_e32 v20, v20, v5
	v_xor_b32_e32 v21, v21, v5
	s_wait_dscnt 0x1
	v_perm_b32 v3, v16, v3, v17
	v_lshl_add_u32 v16, v23, 2, v12
	v_or_b32_e32 v23, 0x114, v6
	s_wait_dscnt 0x0
	v_perm_b32 v2, v15, v2, v17
	v_lshl_add_u32 v15, v22, 2, v12
	v_or_b32_e32 v22, 20, v6
	v_xor_b32_e32 v23, v23, v5
	ds_store_b128 v19, v[0:3] offset:16384
	ds_load_b32 v0, v4 offset:32768
	ds_load_b32 v1, v14 offset:32768
	ds_load_b32 v2, v15 offset:32768
	ds_load_b32 v3, v16 offset:32768
	v_xor_b32_e32 v22, v22, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v4, v10, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v14, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v10, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v18
	s_wait_dscnt 0x2
	v_perm_b32 v1, v14, v1, v18
	s_wait_dscnt 0x1
	v_perm_b32 v2, v15, v2, v18
	s_wait_dscnt 0x0
	v_perm_b32 v3, v16, v3, v18
	ds_bpermute_b32 v4, v9, v0
	ds_bpermute_b32 v14, v9, v1
	ds_bpermute_b32 v15, v9, v2
	ds_bpermute_b32 v16, v9, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v17
	s_wait_dscnt 0x2
	v_perm_b32 v1, v14, v1, v17
	s_wait_dscnt 0x1
	v_perm_b32 v2, v15, v2, v17
	s_wait_dscnt 0x0
	v_perm_b32 v3, v16, v3, v17
	v_lshl_add_u32 v4, v20, 2, v12
	v_lshl_add_u32 v14, v21, 2, v12
	v_lshl_add_u32 v15, v22, 2, v12
	v_lshl_add_u32 v16, v23, 2, v12
	ds_store_b128 v19, v[0:3] offset:16896
	ds_load_b32 v0, v4 offset:32768
	ds_load_b32 v1, v14 offset:32768
	ds_load_b32 v2, v15 offset:32768
	ds_load_b32 v3, v16 offset:32768
	v_or_b32_e32 v20, 24, v6
	v_or_b32_e32 v21, 0x118, v6
	v_or_b32_e32 v22, 28, v6
	v_or_b32_e32 v23, 0x11c, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v20, v20, v5
	v_xor_b32_e32 v21, v21, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v22, v22, v5
	v_xor_b32_e32 v23, v23, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v4, v10, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v14, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v10, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v18
	s_wait_dscnt 0x2
	v_perm_b32 v1, v14, v1, v18
	s_wait_dscnt 0x1
	v_perm_b32 v2, v15, v2, v18
	s_wait_dscnt 0x0
	v_perm_b32 v3, v16, v3, v18
	ds_bpermute_b32 v4, v9, v0
	ds_bpermute_b32 v14, v9, v1
	ds_bpermute_b32 v15, v9, v2
	ds_bpermute_b32 v16, v9, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v17
	s_wait_dscnt 0x2
	v_perm_b32 v1, v14, v1, v17
	s_wait_dscnt 0x1
	v_perm_b32 v2, v15, v2, v17
	s_wait_dscnt 0x0
	v_perm_b32 v3, v16, v3, v17
	v_lshl_add_u32 v4, v20, 2, v12
	v_lshl_add_u32 v14, v21, 2, v12
	v_lshl_add_u32 v15, v22, 2, v12
	v_lshl_add_u32 v16, v23, 2, v12
	ds_store_b128 v19, v[0:3] offset:17408
	ds_load_b32 v0, v4 offset:32768
	ds_load_b32 v1, v14 offset:32768
	ds_load_b32 v2, v15 offset:32768
	ds_load_b32 v3, v16 offset:32768
	v_or_b32_e32 v20, 44, v6
	v_or_b32_e32 v21, 0x12c, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v20, v20, v5
	v_xor_b32_e32 v21, v21, v5
	s_wait_dscnt 0x3
	ds_bpermute_b32 v4, v10, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v14, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v10, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v18
	s_wait_dscnt 0x2
	v_perm_b32 v1, v14, v1, v18
	s_wait_dscnt 0x1
	v_perm_b32 v2, v15, v2, v18
	s_wait_dscnt 0x0
	v_perm_b32 v3, v16, v3, v18
	ds_bpermute_b32 v4, v9, v0
	ds_bpermute_b32 v14, v9, v1
	ds_bpermute_b32 v15, v9, v2
	ds_bpermute_b32 v16, v9, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v17
	s_wait_dscnt 0x2
	v_perm_b32 v1, v14, v1, v17
	s_wait_dscnt 0x1
	v_perm_b32 v2, v15, v2, v17
	s_wait_dscnt 0x0
	v_perm_b32 v3, v16, v3, v17
	v_or_b32_e32 v16, 40, v6
	v_or_b32_e32 v15, 0x800, v11
	ds_store_b128 v19, v[0:3] offset:17920
	ds_load_2addr_b32 v[0:1], v7 offset0:32 offset1:36
	ds_load_2addr_b32 v[2:3], v8 offset0:32 offset1:36
	v_or_b32_e32 v19, 0x128, v6
	v_xor_b32_e32 v16, v16, v5
	v_add_nc_u32_e32 v15, v13, v15
	s_delay_alu instid0(VALU_DEP_3)
	v_xor_b32_e32 v19, v19, v5
	s_wait_dscnt 0x1
	ds_bpermute_b32 v4, v10, v0
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v10, v2
	ds_bpermute_b32 v8, v10, v1
	ds_bpermute_b32 v14, v10, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v18
	s_wait_dscnt 0x2
	v_perm_b32 v2, v7, v2, v18
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v1, v18
	s_wait_dscnt 0x0
	v_perm_b32 v3, v14, v3, v18
	ds_bpermute_b32 v1, v9, v0
	ds_bpermute_b32 v7, v9, v2
	ds_bpermute_b32 v8, v9, v4
	ds_bpermute_b32 v14, v9, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v1, v0, v17
	s_wait_dscnt 0x2
	v_perm_b32 v1, v7, v2, v17
	s_wait_dscnt 0x1
	v_perm_b32 v2, v8, v4, v17
	s_wait_dscnt 0x0
	v_perm_b32 v3, v14, v3, v17
	v_lshl_add_u32 v4, v16, 2, v12
	v_lshl_add_u32 v7, v19, 2, v12
	v_lshl_add_u32 v8, v20, 2, v12
	v_lshl_add_u32 v14, v21, 2, v12
	ds_store_b128 v15, v[0:3] offset:16384
	ds_load_b32 v0, v4 offset:32768
	ds_load_b32 v1, v7 offset:32768
	ds_load_b32 v2, v8 offset:32768
	ds_load_b32 v3, v14 offset:32768
	v_or_b32_e32 v16, 48, v6
	v_or_b32_e32 v19, 0x130, v6
	v_or_b32_e32 v20, 52, v6
	v_or_b32_e32 v21, 0x134, v6
	v_or_b32_e32 v15, 0xa00, v11
	v_xor_b32_e32 v16, v16, v5
	v_xor_b32_e32 v19, v19, v5
	v_xor_b32_e32 v20, v20, v5
	v_xor_b32_e32 v21, v21, v5
	v_add_nc_u32_e32 v15, v13, v15
	s_wait_dscnt 0x3
	ds_bpermute_b32 v4, v10, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v14, v10, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v18
	s_wait_dscnt 0x2
	v_perm_b32 v1, v7, v1, v18
	s_wait_dscnt 0x1
	v_perm_b32 v2, v8, v2, v18
	s_wait_dscnt 0x0
	v_perm_b32 v3, v14, v3, v18
	ds_bpermute_b32 v4, v9, v0
	ds_bpermute_b32 v7, v9, v1
	ds_bpermute_b32 v8, v9, v2
	ds_bpermute_b32 v14, v9, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v17
	s_wait_dscnt 0x2
	v_perm_b32 v1, v7, v1, v17
	s_wait_dscnt 0x1
	v_perm_b32 v2, v8, v2, v17
	s_wait_dscnt 0x0
	v_perm_b32 v3, v14, v3, v17
	v_lshl_add_u32 v4, v16, 2, v12
	v_lshl_add_u32 v7, v19, 2, v12
	v_lshl_add_u32 v8, v20, 2, v12
	v_lshl_add_u32 v14, v21, 2, v12
	ds_store_b128 v15, v[0:3] offset:16384
	ds_load_b32 v0, v4 offset:32768
	ds_load_b32 v1, v7 offset:32768
	ds_load_b32 v2, v8 offset:32768
	ds_load_b32 v3, v14 offset:32768
	v_or_b32_e32 v16, 56, v6
	v_or_b32_e32 v19, 0x138, v6
	v_or_b32_e32 v20, 60, v6
	v_or_b32_e32 v6, 0x13c, v6
	v_or_b32_e32 v15, 0xc00, v11
	v_xor_b32_e32 v16, v16, v5
	v_xor_b32_e32 v19, v19, v5
	v_xor_b32_e32 v20, v20, v5
	v_xor_b32_e32 v5, v6, v5
	v_add_nc_u32_e32 v15, v13, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v6, v19, 2, v12
	v_lshl_add_u32 v5, v5, 2, v12
	s_wait_dscnt 0x3
	ds_bpermute_b32 v4, v10, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v14, v10, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v18
	s_wait_dscnt 0x2
	v_perm_b32 v1, v7, v1, v18
	s_wait_dscnt 0x1
	v_perm_b32 v2, v8, v2, v18
	s_wait_dscnt 0x0
	v_perm_b32 v3, v14, v3, v18
	ds_bpermute_b32 v4, v9, v0
	ds_bpermute_b32 v7, v9, v1
	ds_bpermute_b32 v8, v9, v2
	ds_bpermute_b32 v14, v9, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v17
	s_wait_dscnt 0x2
	v_perm_b32 v1, v7, v1, v17
	s_wait_dscnt 0x1
	v_perm_b32 v2, v8, v2, v17
	s_wait_dscnt 0x0
	v_perm_b32 v3, v14, v3, v17
	v_lshl_add_u32 v4, v16, 2, v12
	v_lshl_add_u32 v7, v20, 2, v12
	v_or_b32_e32 v8, 0xe00, v11
	ds_store_b128 v15, v[0:3] offset:16384
	ds_load_b32 v0, v4 offset:32768
	ds_load_b32 v1, v6 offset:32768
	ds_load_b32 v2, v7 offset:32768
	ds_load_b32 v3, v5 offset:32768
	s_wait_dscnt 0x3
	ds_bpermute_b32 v4, v10, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v5, v10, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v10, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v10, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v18
	s_wait_dscnt 0x2
	v_perm_b32 v1, v5, v1, v18
	s_wait_dscnt 0x1
	v_perm_b32 v2, v6, v2, v18
	s_wait_dscnt 0x0
	v_perm_b32 v3, v7, v3, v18
	ds_bpermute_b32 v4, v9, v0
	ds_bpermute_b32 v5, v9, v1
	ds_bpermute_b32 v6, v9, v2
	ds_bpermute_b32 v7, v9, v3
	s_wait_dscnt 0x3
	v_perm_b32 v0, v4, v0, v17
	s_wait_dscnt 0x2
	v_perm_b32 v1, v5, v1, v17
	s_wait_dscnt 0x1
	v_perm_b32 v2, v6, v2, v17
	s_wait_dscnt 0x0
	v_perm_b32 v3, v7, v3, v17
	v_add_nc_u32_e32 v4, v13, v8
	ds_store_b128 v4, v[0:3] offset:16384
.LBB3_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end3:
	.size	t_new_fill, .Lfunc_end3-t_new_fill
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel t_new_fill
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
		.amdhsa_next_free_vgpr 24
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-t_new_fill)<<4)&4080)>>4
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
	.set .Lt_new_fill.num_vgpr, 24
	.set .Lt_new_fill.num_agpr, 0
	.set .Lt_new_fill.numbered_sgpr, 8
	.set .Lt_new_fill.num_named_barrier, 0
	.set .Lt_new_fill.private_seg_size, 0
	.set .Lt_new_fill.uses_vcc, 1
	.set .Lt_new_fill.uses_flat_scratch, 0
	.set .Lt_new_fill.has_dyn_sized_stack, 0
	.set .Lt_new_fill.has_recursion, 0
	.set .Lt_new_fill.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4048
; TotalNumSgprs: 10
; NumVgprs: 24
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 24
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
	.type	__hip_cuid_645de2bebc2578e,@object ; @__hip_cuid_645de2bebc2578e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_645de2bebc2578e
__hip_cuid_645de2bebc2578e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_645de2bebc2578e, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_645de2bebc2578e
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
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
    .name:           t_old_fill_dump
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         t_old_fill_dump.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     66
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
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
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
    .name:           t_old_fill
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         t_old_fill.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     25
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
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
    .max_flat_workgroup_size: 256
    .name:           t_new_fill_dump
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         t_new_fill_dump.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     34
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
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
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
    .max_flat_workgroup_size: 256
    .name:           t_new_fill
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         t_new_fill.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     24
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
