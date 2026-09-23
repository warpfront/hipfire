	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z4ringPKjPji           ; -- Begin function _Z4ringPKjPji
	.globl	_Z4ringPKjPji
	.p2align	8
	.type	_Z4ringPKjPji,@function
_Z4ringPKjPji:                          ; @_Z4ringPKjPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 16, v0
; %bb.1:
	v_lshl_add_u32 v1, v0, 2, 0
	v_mov_b32_e32 v2, 0
	ds_store_b32 v1, v2 offset:40960
; %bb.2:
	s_or_b32 exec_lo, exec_lo, s2
	s_clause 0x1
	s_load_b32 s3, s[0:1], 0x10
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_bfe_u32 v1, v0, 5, 2
	v_lshrrev_b32_e32 v2, 7, v0
	v_and_b32_e32 v4, 31, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_ne_u32_e32 vcc_lo, 3, v1
	v_mad_u32_u24 v5, v2, 3, v1
	s_barrier_wait -1
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s3, 1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB0_52
; %bb.3:                                ; %.lr.ph
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v6, 2, v4
	v_lshl_add_u32 v7, v5, 2, 0
	v_lshl_add_u32 v8, v2, 2, 0
	v_cmp_eq_u32_e64 s0, 0, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v11, v2, 13, v6
	v_or_b32_e32 v9, 0xffffffe0, v4
	v_lshl_or_b32 v0, v2, 11, v4
	v_dual_mov_b32 v3, 0 :: v_dual_add_nc_u32 v10, 0, v6
	v_dual_mov_b32 v6, v1 :: v_dual_add_nc_u32 v11, 0, v11
	s_mov_b32 s8, 0
	s_mov_b32 s9, 0
	s_branch .LBB0_6
.LBB0_4:                                ; %Flow149
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_5:                                ; %Flow165
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s10
	v_cmp_eq_u32_e64 s1, s3, v12
	v_dual_mov_b32 v3, v12 :: v_dual_add_nc_u32 v0, 0x1000, v0
	s_xor_b32 s9, s9, -1
	s_or_b32 s8, s1, s8
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execz .LBB0_51
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_10 Depth 2
                                        ;     Child Loop BB0_14 Depth 2
                                        ;     Child Loop BB0_16 Depth 2
                                        ;     Child Loop BB0_25 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
                                        ;     Child Loop BB0_33 Depth 2
                                        ;     Child Loop BB0_37 Depth 2
                                        ;     Child Loop BB0_41 Depth 2
                                        ;     Child Loop BB0_45 Depth 2
                                        ;     Child Loop BB0_48 Depth 2
	v_cndmask_b32_e64 v2, 0, 1, s9
	v_and_b32_e32 v13, 1, v3
	v_add_nc_u32_e32 v12, 1, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v2, 14, v2
	s_and_saveexec_b32 s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, exec_lo, s1
	s_cbranch_execz .LBB0_20
; %bb.7:                                ; %.preheader71
                                        ;   in Loop: Header=BB0_6 Depth=1
	v_lshlrev_b32_e32 v14, 3, v13
	v_add_nc_u32_e32 v12, 1, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v3, 0, v14
	s_and_saveexec_b32 s10, s0
	s_cbranch_execz .LBB0_11
; %bb.8:                                ; %.preheader.i60
                                        ;   in Loop: Header=BB0_6 Depth=1
	ds_load_b32 v14, v3 offset:40960
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_ne_u32_e64 s1, v14, v12
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_11
; %bb.9:                                ; %.lr.ph.i62.preheader
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s11, 0
.LBB0_10:                               ; %.lr.ph.i62
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	ds_load_b32 v14, v3 offset:40960
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_eq_u32_e64 s1, v14, v12
	s_or_b32 s11, s1, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execnz .LBB0_10
.LBB0_11:                               ; %_Z11await_epochPjj.exit64
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	; wave barrier
	s_and_saveexec_b32 s10, s0
	s_cbranch_execz .LBB0_15
; %bb.12:                               ; %.preheader.i60.1
                                        ;   in Loop: Header=BB0_6 Depth=1
	ds_load_b32 v14, v3 offset:40964
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_ne_u32_e64 s1, v14, v12
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_15
; %bb.13:                               ; %.lr.ph.i62.1.preheader
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s11, 0
.LBB0_14:                               ; %.lr.ph.i62.1
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	ds_load_b32 v14, v3 offset:40964
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_eq_u32_e64 s1, v14, v12
	s_or_b32 s11, s1, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execnz .LBB0_14
.LBB0_15:                               ; %_Z11await_epochPjj.exit64.1
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	v_dual_mov_b32 v3, v9 :: v_dual_add_nc_u32 v2, v10, v2
	s_mov_b32 s10, 0
	; wave barrier
.LBB0_16:                               ; %.preheader70
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_load_b32 v14, v2
	v_add_nc_u32_e32 v3, 32, v3
	v_add_nc_u32_e32 v2, 0x80, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_lt_u32_e64 s1, 0xfdf, v3
	s_or_b32 s10, s1, s10
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v6, v14, v6
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execnz .LBB0_16
; %bb.17:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	; wave barrier
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mad_u32_u24 v2, v13, 24, v7
	s_wait_loadcnt 0x0
	ds_store_b32 v2, v12 offset:40976
.LBB0_19:                               ; %Flow
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr3
.LBB0_20:                               ; %Flow164
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s2
	s_cbranch_execz .LBB0_5
; %bb.21:                               ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_u32_e32 1, v3
	s_cbranch_execz .LBB0_47
; %bb.22:                               ; %.preheader
                                        ;   in Loop: Header=BB0_6 Depth=1
	v_mul_u32_u24_e32 v14, 24, v13
	v_add_nc_u32_e32 v3, -1, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v14, 0, v14
	s_and_saveexec_b32 s11, s0
	s_cbranch_execz .LBB0_26
; %bb.23:                               ; %.preheader.i
                                        ;   in Loop: Header=BB0_6 Depth=1
	ds_load_b32 v15, v14 offset:40976
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_ne_u32_e64 s1, v15, v3
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_26
; %bb.24:                               ; %.lr.ph.i.preheader
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s12, 0
.LBB0_25:                               ; %.lr.ph.i
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	ds_load_b32 v15, v14 offset:40976
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_eq_u32_e64 s1, v15, v3
	s_or_b32 s12, s1, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_25
.LBB0_26:                               ; %_Z11await_epochPjj.exit
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	; wave barrier
	s_and_saveexec_b32 s11, s0
	s_cbranch_execz .LBB0_30
; %bb.27:                               ; %.preheader.i.1
                                        ;   in Loop: Header=BB0_6 Depth=1
	ds_load_b32 v15, v14 offset:40980
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_ne_u32_e64 s1, v15, v3
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_30
; %bb.28:                               ; %.lr.ph.i.1.preheader
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s12, 0
.LBB0_29:                               ; %.lr.ph.i.1
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	ds_load_b32 v15, v14 offset:40980
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_eq_u32_e64 s1, v15, v3
	s_or_b32 s12, s1, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_29
.LBB0_30:                               ; %_Z11await_epochPjj.exit.1
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	; wave barrier
	s_and_saveexec_b32 s11, s0
	s_cbranch_execz .LBB0_34
; %bb.31:                               ; %.preheader.i.2
                                        ;   in Loop: Header=BB0_6 Depth=1
	ds_load_b32 v15, v14 offset:40984
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_ne_u32_e64 s1, v15, v3
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_34
; %bb.32:                               ; %.lr.ph.i.2.preheader
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s12, 0
.LBB0_33:                               ; %.lr.ph.i.2
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	ds_load_b32 v15, v14 offset:40984
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_eq_u32_e64 s1, v15, v3
	s_or_b32 s12, s1, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_33
.LBB0_34:                               ; %_Z11await_epochPjj.exit.2
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	; wave barrier
	s_and_saveexec_b32 s11, s0
	s_cbranch_execz .LBB0_38
; %bb.35:                               ; %.preheader.i.3
                                        ;   in Loop: Header=BB0_6 Depth=1
	ds_load_b32 v15, v14 offset:40988
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_ne_u32_e64 s1, v15, v3
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_38
; %bb.36:                               ; %.lr.ph.i.3.preheader
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s12, 0
.LBB0_37:                               ; %.lr.ph.i.3
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	ds_load_b32 v15, v14 offset:40988
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_eq_u32_e64 s1, v15, v3
	s_or_b32 s12, s1, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_37
.LBB0_38:                               ; %_Z11await_epochPjj.exit.3
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	; wave barrier
	s_and_saveexec_b32 s11, s0
	s_cbranch_execz .LBB0_42
; %bb.39:                               ; %.preheader.i.4
                                        ;   in Loop: Header=BB0_6 Depth=1
	ds_load_b32 v15, v14 offset:40992
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_ne_u32_e64 s1, v15, v3
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_42
; %bb.40:                               ; %.lr.ph.i.4.preheader
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s12, 0
.LBB0_41:                               ; %.lr.ph.i.4
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	ds_load_b32 v15, v14 offset:40992
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_eq_u32_e64 s1, v15, v3
	s_or_b32 s12, s1, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_41
.LBB0_42:                               ; %_Z11await_epochPjj.exit.4
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	; wave barrier
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_46
; %bb.43:                               ; %.preheader.i.5
                                        ;   in Loop: Header=BB0_6 Depth=1
	ds_load_b32 v15, v14 offset:40996
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_ne_u32_e64 s1, v15, v3
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_46
; %bb.44:                               ; %.lr.ph.i.5.preheader
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s11, 0
.LBB0_45:                               ; %.lr.ph.i.5
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	ds_load_b32 v15, v14 offset:40996
	s_wait_dscnt 0x0
	global_inv scope:SCOPE_SE
	v_cmp_eq_u32_e64 s1, v15, v3
	s_or_b32 s11, s1, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execnz .LBB0_45
.LBB0_46:                               ; %_Z11await_epochPjj.exit.5
                                        ;   in Loop: Header=BB0_6 Depth=1
	; wave barrier
.LBB0_47:                               ; %Flow163
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_add_nc_u32_e32 v14, v11, v2
	s_mov_b32 s11, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, s1, s4, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s5, v16, s1
	v_mov_b32_e32 v15, v9
.LBB0_48:                               ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v16, v[2:3], off
	v_add_nc_u32_e32 v15, 32, v15
	v_add_co_u32 v2, s1, 0x80, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, s1
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s2, 0x7df, v15
	s_or_b32 s11, s2, s11
	s_wait_loadcnt 0x0
	ds_store_b32 v14, v16
	v_add_nc_u32_e32 v14, 0x80, v14
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execnz .LBB0_48
; %bb.49:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	; wave barrier
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_4
; %bb.50:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u32 v2, v13, 3, v8
	s_wait_dscnt 0x0
	ds_store_b32 v2, v12 offset:40960
	s_branch .LBB0_4
.LBB0_51:                               ; %Flow166
	s_or_b32 exec_lo, exec_lo, s8
	s_branch .LBB0_53
.LBB0_52:
	v_mov_b32_e32 v6, 0
.LBB0_53:                               ; %Flow167
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_55
; %bb.54:
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s0, ttmp9, 0xc0
	v_mov_b32_e32 v1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_add_u32 v0, v5, 5, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v0, v0, v4
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s6, v0
	v_add_co_ci_u32_e64 v1, null, s7, v1, vcc_lo
	global_store_b32 v[0:1], v6, off
.LBB0_55:
	s_endpgm
.Lfunc_end0:
	.size	_Z4ringPKjPji, .Lfunc_end0-_Z4ringPKjPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z4ringPKjPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_sgpr 13
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_Z4ringPKjPji)<<4)&4080)>>4
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
	.set .L_Z4ringPKjPji.num_vgpr, 17
	.set .L_Z4ringPKjPji.num_agpr, 0
	.set .L_Z4ringPKjPji.numbered_sgpr, 13
	.set .L_Z4ringPKjPji.num_named_barrier, 0
	.set .L_Z4ringPKjPji.private_seg_size, 0
	.set .L_Z4ringPKjPji.uses_vcc, 1
	.set .L_Z4ringPKjPji.uses_flat_scratch, 0
	.set .L_Z4ringPKjPji.has_dyn_sized_stack, 0
	.set .L_Z4ringPKjPji.has_recursion, 0
	.set .L_Z4ringPKjPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1624
; TotalNumSgprs: 15
; NumVgprs: 17
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 15
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
	.type	__hip_cuid_f70806c800e446c8,@object ; @__hip_cuid_f70806c800e446c8
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_f70806c800e446c8
__hip_cuid_f70806c800e446c8:
	.byte	0                               ; 0x0
	.size	__hip_cuid_f70806c800e446c8, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_f70806c800e446c8
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z4ringPKjPji
    .private_segment_fixed_size: 0
    .sgpr_count:     15
    .sgpr_spill_count: 0
    .symbol:         _Z4ringPKjPji.kd
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
