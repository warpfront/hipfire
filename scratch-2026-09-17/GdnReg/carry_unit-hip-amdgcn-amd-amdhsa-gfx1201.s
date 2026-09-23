	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z10carry_unitPKfPfi    ; -- Begin function _Z10carry_unitPKfPfi
	.globl	_Z10carry_unitPKfPfi
	.p2align	8
	.type	_Z10carry_unitPKfPfi,@function
_Z10carry_unitPKfPfi:                   ; @_Z10carry_unitPKfPfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s13, s[0:1], 0x10
	v_lshlrev_b32_e32 v1, 7, v0
	v_lshrrev_b32_e32 v2, 6, v0
	v_or_b32_e32 v3, 0x400, v0
	s_load_b128 s[0:3], s[0:1], 0x0
	v_mov_b16_e32 v41.l, 0
	v_and_b32_e32 v1, 0x1f80, v1
	s_delay_alu instid0(VALU_DEP_3)
	v_lshrrev_b32_e32 v3, 6, v3
	s_wait_kmcnt 0x0
	s_lshl_b32 s12, s13, 6
	s_delay_alu instid0(VALU_DEP_2) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v24, s12, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_or_b32_e32 v1, v24, v2
	v_or_b32_e32 v3, v24, v3
	v_add_nc_u32_e32 v5, 4, v1
	v_add_nc_u32_e32 v7, 8, v1
	v_ashrrev_i32_e32 v2, 31, v1
	v_add_nc_u32_e32 v9, 12, v1
	v_ashrrev_i32_e32 v4, 31, v3
	v_ashrrev_i32_e32 v6, 31, v5
	v_ashrrev_i32_e32 v8, 31, v7
	v_add_nc_u32_e32 v11, 20, v1
	v_lshlrev_b64_e32 v[16:17], 2, v[1:2]
	v_ashrrev_i32_e32 v10, 31, v9
	v_add_nc_u32_e32 v13, 24, v1
	v_lshlrev_b64_e32 v[2:3], 2, v[3:4]
	v_lshlrev_b64_e32 v[4:5], 2, v[5:6]
	v_add_nc_u32_e32 v15, 28, v1
	v_lshlrev_b64_e32 v[6:7], 2, v[7:8]
	v_ashrrev_i32_e32 v12, 31, v11
	v_lshlrev_b64_e32 v[8:9], 2, v[9:10]
	v_add_co_u32 v18, vcc_lo, s0, v16
	v_ashrrev_i32_e32 v14, 31, v13
	v_add_co_ci_u32_e64 v19, null, s1, v17, vcc_lo
	v_add_co_u32 v4, vcc_lo, s0, v4
	v_ashrrev_i32_e32 v16, 31, v15
	v_lshlrev_b64_e32 v[10:11], 2, v[11:12]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s1, v5, vcc_lo
	v_add_co_u32 v6, vcc_lo, s0, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	v_add_co_u32 v8, vcc_lo, s0, v8
	v_lshlrev_b64_e32 v[12:13], 2, v[13:14]
	v_or_b32_e32 v17, 0x800, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	v_add_co_u32 v2, vcc_lo, s0, v2
	v_lshlrev_b64_e32 v[14:15], 2, v[15:16]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v3, vcc_lo
	v_add_co_u32 v10, vcc_lo, s0, v10
	v_lshrrev_b32_e32 v16, 6, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s1, v11, vcc_lo
	v_add_co_u32 v12, vcc_lo, s0, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s1, v13, vcc_lo
	v_add_co_u32 v14, vcc_lo, s0, v14
	v_or_b32_e32 v16, v24, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s1, v15, vcc_lo
	s_clause 0x7
	global_load_b32 v18, v[18:19], off
	global_load_b32 v19, v[4:5], off
	global_load_b32 v25, v[6:7], off
	global_load_b32 v26, v[8:9], off
	global_load_b32 v27, v[2:3], off
	global_load_b32 v28, v[10:11], off
	global_load_b32 v29, v[12:13], off
	global_load_b32 v30, v[14:15], off
	v_or_b32_e32 v7, 0xc00, v0
	v_add_nc_u32_e32 v20, 36, v1
	v_ashrrev_i32_e32 v17, 31, v16
	v_add_nc_u32_e32 v22, 40, v1
	v_add_nc_u32_e32 v6, 44, v1
	v_lshrrev_b32_e32 v10, 6, v7
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshlrev_b64_e32 v[2:3], 2, v[16:17]
	v_add_nc_u32_e32 v12, 52, v1
	v_add_nc_u32_e32 v14, 56, v1
	v_dual_mov_b32 v1, 0 :: v_dual_add_nc_u32 v16, 60, v1
	v_or_b32_e32 v10, v24, v10
	v_ashrrev_i32_e32 v23, 31, v22
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshlrev_b64_e32 v[4:5], 2, v[20:21]
	v_ashrrev_i32_e32 v13, 31, v12
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshlrev_b64_e32 v[8:9], 2, v[22:23]
	v_add_co_u32 v2, vcc_lo, s0, v2
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_ashrrev_i32_e32 v15, 31, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v3, vcc_lo
	v_add_co_u32 v4, vcc_lo, s0, v4
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_ashrrev_i32_e32 v17, 31, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s1, v5, vcc_lo
	v_add_co_u32 v8, vcc_lo, s0, v8
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	v_add_co_u32 v6, vcc_lo, s0, v6
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	v_add_co_u32 v10, vcc_lo, s0, v10
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s1, v11, vcc_lo
	v_add_co_u32 v12, vcc_lo, s0, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s1, v13, vcc_lo
	v_add_co_u32 v14, vcc_lo, s0, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s1, v15, vcc_lo
	v_add_co_u32 v16, vcc_lo, s0, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s1, v17, vcc_lo
	s_clause 0x7
	global_load_b32 v20, v[2:3], off
	global_load_b32 v21, v[4:5], off
	global_load_b32 v9, v[8:9], off
	global_load_b32 v22, v[6:7], off
	global_load_b32 v10, v[10:11], off
	global_load_b32 v11, v[12:13], off
	global_load_b32 v12, v[14:15], off
	global_load_b32 v13, v[16:17], off
	v_dual_mov_b32 v15, 0 :: v_dual_lshlrev_b32 v14, 1, v0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_mov_b32_e32 v16, 0
	s_mov_b32 s0, 0
	s_mov_b32 s1, 0
	s_wait_loadcnt 0xf
	v_cvt_f16_f32_e32 v4.l, v18
	s_wait_loadcnt 0xe
	v_cvt_f16_f32_e32 v5.l, v19
	s_wait_loadcnt 0xd
	v_cvt_f16_f32_e32 v5.h, v25
	s_wait_loadcnt 0xc
	v_cvt_f16_f32_e32 v6.l, v26
	s_wait_loadcnt 0xb
	v_cvt_f16_f32_e32 v6.h, v27
	s_wait_loadcnt 0xa
	v_cvt_f16_f32_e32 v7.l, v28
	s_wait_loadcnt 0x9
	v_cvt_f16_f32_e32 v7.h, v29
	s_wait_loadcnt 0x8
	v_cvt_f16_f32_e32 v8.l, v30
	s_wait_loadcnt 0x7
	v_cvt_f16_f32_e32 v4.h, v20
	s_wait_loadcnt 0x6
	v_cvt_f16_f32_e32 v8.h, v21
	s_wait_loadcnt 0x5
	v_cvt_f16_f32_e32 v9.l, v9
	s_wait_loadcnt 0x4
	v_cvt_f16_f32_e32 v9.h, v22
	s_wait_loadcnt 0x3
	v_cvt_f16_f32_e32 v10.l, v10
	s_wait_loadcnt 0x2
	v_cvt_f16_f32_e32 v10.h, v11
	s_wait_loadcnt 0x1
	v_cvt_f16_f32_e32 v11.l, v12
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v11.h, v13
	ds_store_b16 v14, v4
	ds_store_b16 v14, v5 offset:512
	ds_store_b16_d16_hi v14, v5 offset:1024
	ds_store_b16 v14, v6 offset:1536
	ds_store_b16_d16_hi v14, v6 offset:2048
	ds_store_b16 v14, v7 offset:2560
	ds_store_b16_d16_hi v14, v7 offset:3072
	ds_store_b16 v14, v8 offset:3584
	ds_store_b16_d16_hi v14, v4 offset:4096
	ds_store_b16_d16_hi v14, v8 offset:4608
	ds_store_b16 v14, v9 offset:5120
	ds_store_b16_d16_hi v14, v9 offset:5632
	ds_store_b16 v14, v10 offset:6144
	ds_store_b16_d16_hi v14, v10 offset:6656
	ds_store_b16 v14, v11 offset:7168
	ds_store_b16_d16_hi v14, v11 offset:7680
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	v_mov_b32_e32 v14, 0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB0_1:                                ; %.preheader122
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 m0, s1, 2
	s_add_co_i32 s4, s0, 1
	v_movrels_b32_e32 v20, v1
	v_movrels_b32_e32 v19, v2
	v_movrels_b32_e32 v18, v3
	v_movrels_b32_e32 v17, v4
	s_add_co_i32 s6, s0, 2
	v_lshrrev_b32_e32 v20, 16, v20
	v_lshrrev_b32_e32 v21, 16, v19
	s_lshl_b32 s5, s0, 1
	s_add_co_i32 s7, s0, 3
	s_add_co_i32 s8, s0, 4
	s_add_co_i32 s9, s0, 5
	s_add_co_i32 s10, s0, 6
	s_add_co_i32 s11, s0, 7
	s_and_b32 s14, s4, 31
	v_lshrrev_b32_e32 v22, 16, v18
	s_and_b32 s6, s6, 31
	v_lshrrev_b32_e32 v23, 16, v17
	v_mov_b16_e32 v45.l, v20.l
	s_and_b32 s4, s5, 62
	s_and_b32 s7, s7, 31
	s_and_b32 s8, s8, 31
	s_and_b32 s9, s9, 31
	s_and_b32 s10, s10, 31
	s_and_b32 s11, s11, 31
	s_lshl_b32 s5, s14, 1
	v_mov_b16_e32 v42.l, v19.l
	s_lshl_b32 s6, s6, 1
	v_mov_b16_e32 v46.l, v21.l
	s_or_b32 s14, s4, 0x1c0
	s_lshl_b32 s7, s7, 1
	s_lshl_b32 s8, s8, 1
	s_lshl_b32 s9, s9, 1
	s_lshl_b32 s10, s10, 1
	s_lshl_b32 s11, s11, 1
	s_or_b32 s15, s5, 0x1c0
	v_mov_b16_e32 v43.l, v18.l
	s_or_b32 s16, s6, 0x1c0
	v_mov_b16_e32 v47.l, v22.l
	s_clause 0x3
	scratch_store_b128 off, v[13:16], off offset:496
	scratch_store_b128 off, v[9:12], off offset:480
	scratch_store_b128 off, v[5:8], off offset:464
	scratch_store_b128 off, v[1:4], off offset:448
	s_or_b32 s17, s7, 0x1c0
	s_or_b32 s18, s8, 0x1c0
	s_or_b32 s19, s9, 0x1c0
	s_or_b32 s20, s10, 0x1c0
	s_or_b32 s21, s11, 0x1c0
	scratch_store_b16 off, v41, s14
	v_mov_b16_e32 v44.l, v17.l
	v_mov_b16_e32 v48.l, v23.l
	scratch_store_b16 off, v45, s15
	scratch_store_b16 off, v42, s16
	scratch_store_b16 off, v46, s17
	scratch_store_b16 off, v43, s18
	scratch_store_b16 off, v47, s19
	scratch_store_b16 off, v44, s20
	scratch_store_b16 off, v48, s21
	s_clause 0x3
	scratch_load_b128 v[29:32], off, off offset:496
	scratch_load_b128 v[25:28], off, off offset:480
	scratch_load_b128 v[21:24], off, off offset:464
	scratch_load_b128 v[17:20], off, off offset:448
	s_or_b32 s14, s4, 0x180
	s_or_b32 s15, s5, 0x180
	s_or_b32 s16, s6, 0x180
	s_or_b32 s17, s7, 0x180
	s_or_b32 s18, s8, 0x180
	s_or_b32 s19, s9, 0x180
	s_or_b32 s20, s10, 0x180
	s_or_b32 s21, s11, 0x180
	s_clause 0x3
	scratch_store_b128 off, v[13:16], off offset:432
	scratch_store_b128 off, v[9:12], off offset:416
	scratch_store_b128 off, v[5:8], off offset:400
	scratch_store_b128 off, v[1:4], off offset:384
	scratch_store_b16 off, v41, s14
	scratch_store_b16 off, v45, s15
	scratch_store_b16 off, v42, s16
	scratch_store_b16 off, v46, s17
	scratch_store_b16 off, v43, s18
	scratch_store_b16 off, v47, s19
	scratch_store_b16 off, v44, s20
	scratch_store_b16 off, v48, s21
	s_add_co_i32 s0, s0, 8
	s_add_co_i32 s1, s1, 1
	s_wait_loadcnt 0x0
	v_movrels_b32_e32 v36, v17
	v_movrels_b32_e32 v35, v18
	v_movrels_b32_e32 v34, v19
	v_movrels_b32_e32 v33, v20
	scratch_store_b16 off, v36, s14
	scratch_store_b16 off, v41, s15
	scratch_store_b16 off, v35, s16
	scratch_store_d16_hi_b16 off, v35, s17
	scratch_store_b16 off, v34, s18
	scratch_store_d16_hi_b16 off, v34, s19
	scratch_store_b16 off, v33, s20
	scratch_store_d16_hi_b16 off, v33, s21
	s_clause 0x3
	scratch_load_b128 v[29:32], off, off offset:432
	scratch_load_b128 v[25:28], off, off offset:416
	scratch_load_b128 v[21:24], off, off offset:400
	scratch_load_b128 v[17:20], off, off offset:384
	s_or_b32 s14, s4, 0x140
	s_or_b32 s15, s5, 0x140
	s_or_b32 s16, s6, 0x140
	s_or_b32 s17, s7, 0x140
	s_or_b32 s18, s8, 0x140
	s_or_b32 s19, s9, 0x140
	s_or_b32 s20, s10, 0x140
	s_or_b32 s21, s11, 0x140
	s_clause 0x3
	scratch_store_b128 off, v[13:16], off offset:368
	scratch_store_b128 off, v[9:12], off offset:352
	scratch_store_b128 off, v[5:8], off offset:336
	scratch_store_b128 off, v[1:4], off offset:320
	scratch_store_b16 off, v41, s14
	scratch_store_b16 off, v45, s15
	scratch_store_b16 off, v42, s16
	scratch_store_b16 off, v46, s17
	scratch_store_b16 off, v43, s18
	scratch_store_b16 off, v47, s19
	scratch_store_b16 off, v44, s20
	scratch_store_b16 off, v48, s21
	scratch_store_b16 off, v36, s14
	scratch_store_b16 off, v41, s15
	scratch_store_b16 off, v35, s16
	scratch_store_d16_hi_b16 off, v35, s17
	scratch_store_b16 off, v34, s18
	scratch_store_d16_hi_b16 off, v34, s19
	scratch_store_b16 off, v33, s20
	scratch_store_d16_hi_b16 off, v33, s21
	s_wait_loadcnt 0x0
	v_movrels_b32_e32 v40, v17
	v_movrels_b32_e32 v39, v18
	v_movrels_b32_e32 v38, v19
	v_movrels_b32_e32 v37, v20
	scratch_store_b16 off, v40, s14
	scratch_store_d16_hi_b16 off, v40, s15
	scratch_store_b16 off, v41, s16
	scratch_store_d16_hi_b16 off, v39, s17
	scratch_store_b16 off, v38, s18
	scratch_store_d16_hi_b16 off, v38, s19
	scratch_store_b16 off, v37, s20
	scratch_store_d16_hi_b16 off, v37, s21
	s_clause 0x3
	scratch_load_b128 v[29:32], off, off offset:368
	scratch_load_b128 v[25:28], off, off offset:352
	scratch_load_b128 v[21:24], off, off offset:336
	scratch_load_b128 v[17:20], off, off offset:320
	s_or_b32 s14, s4, 0x100
	s_or_b32 s15, s5, 0x100
	s_or_b32 s16, s6, 0x100
	s_or_b32 s17, s7, 0x100
	s_or_b32 s18, s8, 0x100
	s_or_b32 s19, s9, 0x100
	s_or_b32 s20, s10, 0x100
	s_or_b32 s21, s11, 0x100
	s_clause 0x3
	scratch_store_b128 off, v[13:16], off offset:304
	scratch_store_b128 off, v[9:12], off offset:288
	scratch_store_b128 off, v[5:8], off offset:272
	scratch_store_b128 off, v[1:4], off offset:256
	scratch_store_b16 off, v41, s14
	scratch_store_b16 off, v45, s15
	scratch_store_b16 off, v42, s16
	scratch_store_b16 off, v46, s17
	scratch_store_b16 off, v43, s18
	scratch_store_b16 off, v47, s19
	scratch_store_b16 off, v44, s20
	scratch_store_b16 off, v48, s21
	scratch_store_b16 off, v36, s14
	scratch_store_b16 off, v41, s15
	scratch_store_b16 off, v35, s16
	scratch_store_d16_hi_b16 off, v35, s17
	scratch_store_b16 off, v34, s18
	scratch_store_d16_hi_b16 off, v34, s19
	scratch_store_b16 off, v33, s20
	scratch_store_d16_hi_b16 off, v33, s21
	scratch_store_b16 off, v40, s14
	scratch_store_d16_hi_b16 off, v40, s15
	scratch_store_b16 off, v41, s16
	scratch_store_d16_hi_b16 off, v39, s17
	scratch_store_b16 off, v38, s18
	scratch_store_d16_hi_b16 off, v38, s19
	scratch_store_b16 off, v37, s20
	scratch_store_d16_hi_b16 off, v37, s21
	s_wait_loadcnt 0x0
	v_movrels_b32_e32 v49, v17
	v_movrels_b32_e32 v50, v18
	v_movrels_b32_e32 v52, v19
	v_movrels_b32_e32 v51, v20
	scratch_store_b16 off, v49, s14
	scratch_store_d16_hi_b16 off, v49, s15
	scratch_store_b16 off, v50, s16
	scratch_store_b16 off, v41, s17
	scratch_store_b16 off, v52, s18
	scratch_store_d16_hi_b16 off, v52, s19
	scratch_store_b16 off, v51, s20
	scratch_store_d16_hi_b16 off, v51, s21
	s_clause 0x3
	scratch_load_b128 v[29:32], off, off offset:304
	scratch_load_b128 v[25:28], off, off offset:288
	scratch_load_b128 v[21:24], off, off offset:272
	scratch_load_b128 v[17:20], off, off offset:256
	s_or_b32 s14, s4, 0xc0
	s_or_b32 s15, s5, 0xc0
	s_or_b32 s16, s6, 0xc0
	s_or_b32 s17, s7, 0xc0
	s_or_b32 s18, s8, 0xc0
	s_or_b32 s19, s9, 0xc0
	s_or_b32 s20, s10, 0xc0
	s_or_b32 s21, s11, 0xc0
	s_clause 0x3
	scratch_store_b128 off, v[13:16], off offset:240
	scratch_store_b128 off, v[9:12], off offset:224
	scratch_store_b128 off, v[5:8], off offset:208
	scratch_store_b128 off, v[1:4], off offset:192
	scratch_store_b16 off, v41, s14
	scratch_store_b16 off, v45, s15
	scratch_store_b16 off, v42, s16
	scratch_store_b16 off, v46, s17
	scratch_store_b16 off, v43, s18
	scratch_store_b16 off, v47, s19
	scratch_store_b16 off, v44, s20
	scratch_store_b16 off, v48, s21
	scratch_store_b16 off, v36, s14
	scratch_store_b16 off, v41, s15
	scratch_store_b16 off, v35, s16
	scratch_store_d16_hi_b16 off, v35, s17
	scratch_store_b16 off, v34, s18
	scratch_store_d16_hi_b16 off, v34, s19
	scratch_store_b16 off, v33, s20
	scratch_store_d16_hi_b16 off, v33, s21
	scratch_store_b16 off, v40, s14
	scratch_store_d16_hi_b16 off, v40, s15
	scratch_store_b16 off, v41, s16
	scratch_store_d16_hi_b16 off, v39, s17
	scratch_store_b16 off, v38, s18
	scratch_store_d16_hi_b16 off, v38, s19
	scratch_store_b16 off, v37, s20
	scratch_store_d16_hi_b16 off, v37, s21
	scratch_store_b16 off, v49, s14
	scratch_store_d16_hi_b16 off, v49, s15
	scratch_store_b16 off, v50, s16
	scratch_store_b16 off, v41, s17
	scratch_store_b16 off, v52, s18
	scratch_store_d16_hi_b16 off, v52, s19
	scratch_store_b16 off, v51, s20
	scratch_store_d16_hi_b16 off, v51, s21
	s_wait_loadcnt 0x0
	v_movrels_b32_e32 v53, v17
	v_movrels_b32_e32 v54, v18
	v_movrels_b32_e32 v55, v20
	v_movrels_b32_e32 v56, v19
	scratch_store_b16 off, v53, s14
	scratch_store_d16_hi_b16 off, v53, s15
	scratch_store_b16 off, v54, s16
	scratch_store_d16_hi_b16 off, v54, s17
	scratch_store_b16 off, v41, s18
	scratch_store_d16_hi_b16 off, v56, s19
	scratch_store_b16 off, v55, s20
	scratch_store_d16_hi_b16 off, v55, s21
	s_clause 0x3
	scratch_load_b128 v[29:32], off, off offset:240
	scratch_load_b128 v[25:28], off, off offset:224
	scratch_load_b128 v[21:24], off, off offset:208
	scratch_load_b128 v[17:20], off, off offset:192
	s_or_b32 s14, s4, 0x80
	s_or_b32 s15, s5, 0x80
	s_or_b32 s16, s6, 0x80
	s_or_b32 s17, s7, 0x80
	s_or_b32 s18, s8, 0x80
	s_or_b32 s19, s9, 0x80
	s_or_b32 s20, s10, 0x80
	s_or_b32 s21, s11, 0x80
	s_clause 0x3
	scratch_store_b128 off, v[13:16], off offset:176
	scratch_store_b128 off, v[9:12], off offset:160
	scratch_store_b128 off, v[5:8], off offset:144
	scratch_store_b128 off, v[1:4], off offset:128
	scratch_store_b16 off, v41, s14
	scratch_store_b16 off, v45, s15
	scratch_store_b16 off, v42, s16
	scratch_store_b16 off, v46, s17
	scratch_store_b16 off, v43, s18
	scratch_store_b16 off, v47, s19
	scratch_store_b16 off, v44, s20
	scratch_store_b16 off, v48, s21
	scratch_store_b16 off, v36, s14
	scratch_store_b16 off, v41, s15
	scratch_store_b16 off, v35, s16
	scratch_store_d16_hi_b16 off, v35, s17
	scratch_store_b16 off, v34, s18
	scratch_store_d16_hi_b16 off, v34, s19
	scratch_store_b16 off, v33, s20
	scratch_store_d16_hi_b16 off, v33, s21
	scratch_store_b16 off, v40, s14
	scratch_store_d16_hi_b16 off, v40, s15
	scratch_store_b16 off, v41, s16
	scratch_store_d16_hi_b16 off, v39, s17
	scratch_store_b16 off, v38, s18
	scratch_store_d16_hi_b16 off, v38, s19
	scratch_store_b16 off, v37, s20
	scratch_store_d16_hi_b16 off, v37, s21
	scratch_store_b16 off, v49, s14
	scratch_store_d16_hi_b16 off, v49, s15
	scratch_store_b16 off, v50, s16
	scratch_store_b16 off, v41, s17
	scratch_store_b16 off, v52, s18
	scratch_store_d16_hi_b16 off, v52, s19
	scratch_store_b16 off, v51, s20
	scratch_store_d16_hi_b16 off, v51, s21
	scratch_store_b16 off, v53, s14
	scratch_store_d16_hi_b16 off, v53, s15
	scratch_store_b16 off, v54, s16
	scratch_store_d16_hi_b16 off, v54, s17
	scratch_store_b16 off, v41, s18
	scratch_store_d16_hi_b16 off, v56, s19
	scratch_store_b16 off, v55, s20
	scratch_store_d16_hi_b16 off, v55, s21
	s_wait_loadcnt 0x0
	v_movrels_b32_e32 v57, v17
	v_movrels_b32_e32 v58, v18
	v_movrels_b32_e32 v60, v19
	v_movrels_b32_e32 v59, v20
	scratch_store_b16 off, v57, s14
	scratch_store_d16_hi_b16 off, v57, s15
	scratch_store_b16 off, v58, s16
	scratch_store_d16_hi_b16 off, v58, s17
	scratch_store_b16 off, v60, s18
	scratch_store_b16 off, v41, s19
	scratch_store_b16 off, v59, s20
	scratch_store_d16_hi_b16 off, v59, s21
	s_clause 0x3
	scratch_load_b128 v[29:32], off, off offset:176
	scratch_load_b128 v[25:28], off, off offset:160
	scratch_load_b128 v[21:24], off, off offset:144
	scratch_load_b128 v[17:20], off, off offset:128
	s_or_b32 s14, s4, 64
	s_or_b32 s15, s5, 64
	s_or_b32 s16, s6, 64
	s_or_b32 s17, s7, 64
	s_or_b32 s18, s8, 64
	s_or_b32 s19, s9, 64
	s_or_b32 s20, s10, 64
	s_or_b32 s21, s11, 64
	s_clause 0x3
	scratch_store_b128 off, v[13:16], off offset:112
	scratch_store_b128 off, v[9:12], off offset:96
	scratch_store_b128 off, v[5:8], off offset:80
	scratch_store_b128 off, v[1:4], off offset:64
	scratch_store_b16 off, v41, s14
	scratch_store_b16 off, v45, s15
	scratch_store_b16 off, v42, s16
	scratch_store_b16 off, v46, s17
	scratch_store_b16 off, v43, s18
	scratch_store_b16 off, v47, s19
	scratch_store_b16 off, v44, s20
	scratch_store_b16 off, v48, s21
	scratch_store_b16 off, v36, s14
	scratch_store_b16 off, v41, s15
	scratch_store_b16 off, v35, s16
	scratch_store_d16_hi_b16 off, v35, s17
	scratch_store_b16 off, v34, s18
	scratch_store_d16_hi_b16 off, v34, s19
	scratch_store_b16 off, v33, s20
	scratch_store_d16_hi_b16 off, v33, s21
	scratch_store_b16 off, v40, s14
	scratch_store_d16_hi_b16 off, v40, s15
	scratch_store_b16 off, v41, s16
	scratch_store_d16_hi_b16 off, v39, s17
	scratch_store_b16 off, v38, s18
	scratch_store_d16_hi_b16 off, v38, s19
	scratch_store_b16 off, v37, s20
	scratch_store_d16_hi_b16 off, v37, s21
	scratch_store_b16 off, v49, s14
	scratch_store_d16_hi_b16 off, v49, s15
	scratch_store_b16 off, v50, s16
	scratch_store_b16 off, v41, s17
	scratch_store_b16 off, v52, s18
	scratch_store_d16_hi_b16 off, v52, s19
	scratch_store_b16 off, v51, s20
	scratch_store_d16_hi_b16 off, v51, s21
	scratch_store_b16 off, v53, s14
	scratch_store_d16_hi_b16 off, v53, s15
	scratch_store_b16 off, v54, s16
	scratch_store_d16_hi_b16 off, v54, s17
	scratch_store_b16 off, v41, s18
	scratch_store_d16_hi_b16 off, v56, s19
	scratch_store_b16 off, v55, s20
	scratch_store_d16_hi_b16 off, v55, s21
	scratch_store_b16 off, v57, s14
	scratch_store_d16_hi_b16 off, v57, s15
	scratch_store_b16 off, v58, s16
	scratch_store_d16_hi_b16 off, v58, s17
	scratch_store_b16 off, v60, s18
	scratch_store_b16 off, v41, s19
	scratch_store_b16 off, v59, s20
	scratch_store_d16_hi_b16 off, v59, s21
	s_or_b32 s4, s4, 0
	s_or_b32 s5, s5, 0
	s_or_b32 s6, s6, 0
	s_or_b32 s7, s7, 0
	s_or_b32 s8, s8, 0
	s_or_b32 s9, s9, 0
	s_or_b32 s10, s10, 0
	s_or_b32 s11, s11, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 32
	s_wait_loadcnt 0x0
	v_movrels_b32_e32 v61, v17
	v_movrels_b32_e32 v62, v18
	v_movrels_b32_e32 v64, v19
	v_movrels_b32_e32 v63, v20
	scratch_store_b16 off, v61, s14
	scratch_store_d16_hi_b16 off, v61, s15
	scratch_store_b16 off, v62, s16
	scratch_store_d16_hi_b16 off, v62, s17
	scratch_store_b16 off, v64, s18
	scratch_store_d16_hi_b16 off, v64, s19
	scratch_store_b16 off, v41, s20
	scratch_store_d16_hi_b16 off, v63, s21
	s_clause 0x3
	scratch_load_b128 v[29:32], off, off offset:112
	scratch_load_b128 v[25:28], off, off offset:96
	scratch_load_b128 v[21:24], off, off offset:80
	scratch_load_b128 v[17:20], off, off offset:64
	s_clause 0x3
	scratch_store_b128 off, v[13:16], off offset:48
	scratch_store_b128 off, v[9:12], off offset:32
	scratch_store_b128 off, v[5:8], off offset:16
	scratch_store_b128 off, v[1:4], off
	scratch_store_b16 off, v41, s4
	scratch_store_b16 off, v45, s5
	scratch_store_b16 off, v42, s6
	scratch_store_b16 off, v46, s7
	scratch_store_b16 off, v43, s8
	scratch_store_b16 off, v47, s9
	scratch_store_b16 off, v44, s10
	scratch_store_b16 off, v48, s11
	scratch_store_b16 off, v36, s4
	scratch_store_b16 off, v41, s5
	scratch_store_b16 off, v35, s6
	scratch_store_d16_hi_b16 off, v35, s7
	scratch_store_b16 off, v34, s8
	scratch_store_d16_hi_b16 off, v34, s9
	scratch_store_b16 off, v33, s10
	scratch_store_d16_hi_b16 off, v33, s11
	scratch_store_b16 off, v40, s4
	scratch_store_d16_hi_b16 off, v40, s5
	scratch_store_b16 off, v41, s6
	scratch_store_d16_hi_b16 off, v39, s7
	scratch_store_b16 off, v38, s8
	scratch_store_d16_hi_b16 off, v38, s9
	scratch_store_b16 off, v37, s10
	scratch_store_d16_hi_b16 off, v37, s11
	scratch_store_b16 off, v49, s4
	scratch_store_d16_hi_b16 off, v49, s5
	scratch_store_b16 off, v50, s6
	scratch_store_b16 off, v41, s7
	scratch_store_b16 off, v52, s8
	scratch_store_d16_hi_b16 off, v52, s9
	scratch_store_b16 off, v51, s10
	scratch_store_d16_hi_b16 off, v51, s11
	scratch_store_b16 off, v53, s4
	scratch_store_d16_hi_b16 off, v53, s5
	scratch_store_b16 off, v54, s6
	scratch_store_d16_hi_b16 off, v54, s7
	scratch_store_b16 off, v41, s8
	scratch_store_d16_hi_b16 off, v56, s9
	scratch_store_b16 off, v55, s10
	scratch_store_d16_hi_b16 off, v55, s11
	scratch_store_b16 off, v57, s4
	scratch_store_d16_hi_b16 off, v57, s5
	scratch_store_b16 off, v58, s6
	scratch_store_d16_hi_b16 off, v58, s7
	scratch_store_b16 off, v60, s8
	scratch_store_b16 off, v41, s9
	scratch_store_b16 off, v59, s10
	scratch_store_d16_hi_b16 off, v59, s11
	scratch_store_b16 off, v61, s4
	scratch_store_d16_hi_b16 off, v61, s5
	scratch_store_b16 off, v62, s6
	scratch_store_d16_hi_b16 off, v62, s7
	scratch_store_b16 off, v64, s8
	scratch_store_d16_hi_b16 off, v64, s9
	scratch_store_b16 off, v41, s10
	scratch_store_d16_hi_b16 off, v63, s11
	s_wait_loadcnt 0x0
	v_movrels_b32_e32 v1, v17
	v_movrels_b32_e32 v2, v18
	v_movrels_b32_e32 v4, v19
	v_movrels_b32_e32 v3, v20
	scratch_store_b16 off, v1, s4
	scratch_store_d16_hi_b16 off, v1, s5
	scratch_store_b16 off, v2, s6
	scratch_store_d16_hi_b16 off, v2, s7
	scratch_store_b16 off, v4, s8
	scratch_store_d16_hi_b16 off, v4, s9
	scratch_store_b16 off, v3, s10
	scratch_store_b16 off, v41, s11
	s_clause 0x3
	scratch_load_b128 v[1:4], off, off
	scratch_load_b128 v[5:8], off, off offset:16
	scratch_load_b128 v[9:12], off, off offset:32
	scratch_load_b128 v[13:16], off, off offset:48
	s_wait_loadcnt 0x3
	v_readfirstlane_b32 s6, v1
	v_readfirstlane_b32 s7, v2
	v_readfirstlane_b32 s8, v3
	v_readfirstlane_b32 s9, v4
	s_wait_loadcnt 0x2
	v_readfirstlane_b32 s18, v5
	v_readfirstlane_b32 s20, v6
	v_readfirstlane_b32 s22, v7
	v_readfirstlane_b32 s23, v8
	s_wait_loadcnt 0x1
	v_readfirstlane_b32 s15, v9
	v_readfirstlane_b32 s16, v10
	v_readfirstlane_b32 s19, v11
	v_readfirstlane_b32 s21, v12
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s4, v13
	v_readfirstlane_b32 s5, v14
	v_readfirstlane_b32 s14, v15
	v_readfirstlane_b32 s17, v16
	s_cbranch_scc1 .LBB0_1
; %bb.2:                                ; %.preheader121
	v_bfe_u32 v33, v0, 4, 1
	s_mov_b32 s27, 0x44004300
	s_mov_b32 s26, 0x42004100
	s_mov_b32 s25, 0x40003e00
	s_mov_b32 s24, 0x3c003800
	v_and_b32_e32 v34, 15, v0
	v_dual_mov_b32 v94, s27 :: v_dual_lshlrev_b32 v1, 4, v33
	v_mov_b32_e32 v93, s26
	s_cvt_hi_f32_f16 s0, s9
	s_cvt_hi_f32_f16 s1, s8
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v9, v34, 7, v1
	s_cvt_hi_f32_f16 s28, s7
	s_cvt_hi_f32_f16 s29, s6
	s_cvt_f32_f16 s30, s9
	ds_load_b128 v[1:4], v9
	ds_load_b128 v[35:38], v9 offset:32
	ds_load_b128 v[39:42], v9 offset:64
	ds_load_b128 v[43:46], v9 offset:96
	ds_load_b128 v[5:8], v9 offset:2048
	ds_load_b128 v[47:50], v9 offset:2080
	ds_load_b128 v[51:54], v9 offset:2112
	ds_load_b128 v[55:58], v9 offset:2144
	ds_load_b128 v[59:62], v9 offset:4096
	ds_load_b128 v[63:66], v9 offset:4128
	ds_load_b128 v[67:70], v9 offset:4160
	ds_load_b128 v[71:74], v9 offset:4192
	ds_load_b128 v[75:78], v9 offset:6144
	ds_load_b128 v[79:82], v9 offset:6176
	ds_load_b128 v[83:86], v9 offset:6208
	ds_load_b128 v[87:90], v9 offset:6240
	s_cvt_f32_f16 s31, s8
	s_cvt_f32_f16 s33, s7
	s_cvt_f32_f16 s34, s6
	s_cvt_hi_f32_f16 s35, s23
	s_cvt_hi_f32_f16 s36, s22
	s_cvt_hi_f32_f16 s37, s20
	s_cvt_hi_f32_f16 s38, s18
	s_cvt_f32_f16 s23, s23
	s_cvt_f32_f16 s22, s22
	s_cvt_f32_f16 s20, s20
	s_cvt_f32_f16 s18, s18
	s_cvt_hi_f32_f16 s39, s21
	s_cvt_hi_f32_f16 s40, s19
	s_cvt_hi_f32_f16 s41, s16
	s_cvt_hi_f32_f16 s42, s15
	s_cvt_f32_f16 s21, s21
	s_cvt_f32_f16 s19, s19
	s_cvt_f32_f16 s16, s16
	s_cvt_f32_f16 s15, s15
	v_dual_mov_b32 v92, s25 :: v_dual_mov_b32 v91, s24
	v_dual_mov_b32 v25, s34 :: v_dual_mov_b32 v26, s29
	v_dual_mov_b32 v27, s33 :: v_dual_mov_b32 v28, s28
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v29, s31 :: v_dual_mov_b32 v30, s1
	v_dual_mov_b32 v31, s30 :: v_dual_mov_b32 v32, s0
	v_dual_mov_b32 v17, s18 :: v_dual_mov_b32 v18, s38
	v_dual_mov_b32 v19, s20 :: v_dual_mov_b32 v20, s37
	v_dual_mov_b32 v21, s22 :: v_dual_mov_b32 v22, s36
	v_dual_mov_b32 v23, s23 :: v_dual_mov_b32 v24, s35
	s_cvt_hi_f32_f16 s43, s17
	s_cvt_hi_f32_f16 s44, s14
	s_cvt_hi_f32_f16 s45, s5
	s_cvt_hi_f32_f16 s46, s4
	s_cvt_f32_f16 s17, s17
	s_cvt_f32_f16 s0, s14
	s_cvt_f32_f16 s1, s4
	s_cvt_f32_f16 s4, s5
	v_dual_mov_b32 v9, s15 :: v_dual_mov_b32 v10, s42
	v_dual_mov_b32 v11, s16 :: v_dual_mov_b32 v12, s41
	v_dual_mov_b32 v13, s19 :: v_dual_mov_b32 v14, s40
	v_dual_mov_b32 v15, s21 :: v_dual_mov_b32 v16, s39
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x16_f16 v[25:32], v[1:4], v[91:94], v[25:32]
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x16_f16 v[17:24], v[5:8], v[91:94], v[17:24]
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v1, s1 :: v_dual_mov_b32 v2, s46
	v_dual_mov_b32 v3, s4 :: v_dual_mov_b32 v4, s45
	v_dual_mov_b32 v5, s0 :: v_dual_mov_b32 v6, s44
	v_dual_mov_b32 v7, s17 :: v_dual_mov_b32 v8, s43
	s_mov_b32 s11, 0x48004780
	s_mov_b32 s10, 0x47004680
	s_mov_b32 s9, 0x46004580
	s_mov_b32 s8, 0x45004480
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_f16 v[9:16], v[59:62], v[91:94], v[9:16]
	v_dual_mov_b32 v62, s11 :: v_dual_mov_b32 v61, s10
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[1:8], v[75:78], v[91:94], v[1:8]
	v_dual_mov_b32 v60, s9 :: v_dual_mov_b32 v59, s8
	s_mov_b32 s7, 0x4a0049c0
	s_mov_b32 s6, 0x49804940
	s_mov_b32 s5, 0x490048c0
	s_mov_b32 s4, 0x48804840
	v_dual_mov_b32 v78, s7 :: v_dual_mov_b32 v77, s6
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v76, s5 :: v_dual_mov_b32 v75, s4
	v_wmma_f32_16x16x16_f16 v[25:32], v[35:38], v[59:62], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[47:50], v[59:62], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[63:66], v[59:62], v[9:16]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_f16 v[1:8], v[79:82], v[59:62], v[1:8]
	s_mov_b32 s19, 0x4c004bc0
	s_mov_b32 s18, 0x4b804b40
	s_mov_b32 s17, 0x4b004ac0
	s_mov_b32 s16, 0x4a804a40
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v38, s19 :: v_dual_mov_b32 v37, s18
	v_dual_mov_b32 v36, s17 :: v_dual_mov_b32 v35, s16
	v_wmma_f32_16x16x16_f16 v[25:32], v[39:42], v[75:78], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[51:54], v[75:78], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[67:70], v[75:78], v[9:16]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[1:8], v[83:86], v[75:78], v[1:8]
	v_bfe_u32 v91, v0, 5, 1
	v_wmma_f32_16x16x16_f16 v[25:32], v[43:46], v[35:38], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[55:58], v[35:38], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[71:74], v[35:38], v[9:16]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[87:90], v[35:38], v[1:8]
	s_mov_b32 s0, exec_lo
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_eq_u32_e64 s13, v91
	s_cbranch_execz .LBB0_4
; %bb.3:                                ; %.preheader
	v_cvt_f16_f32_e32 v23.l, v23
	v_cvt_f16_f32_e32 v23.h, v24
	v_lshrrev_b32_e32 v24, 2, v0
	v_cvt_f16_f32_e32 v0.h, v19
	v_cvt_f16_f32_e32 v19.l, v20
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_and_or_b32 v20, 0xf0, v24, v34
	v_cvt_f16_f32_e32 v31.l, v31
	v_cvt_f16_f32_e32 v31.h, v32
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_lshl_add_u32 v16, v20, 7, s12
	v_cvt_f16_f32_e32 v18.l, v7
	v_cvt_f16_f32_e32 v18.h, v8
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_lshl_or_b32 v32, v33, 3, v16
	v_mov_b32_e32 v33, 0
	v_cvt_f16_f32_e32 v16.l, v5
	v_cvt_f16_f32_e32 v14.h, v4
	v_cvt_f16_f32_e32 v25.l, v25
	v_cvt_f16_f32_e32 v29.l, v29
	v_lshlrev_b64_e32 v[7:8], 2, v[32:33]
	v_ashrrev_i32_e32 v33, 31, v32
	v_cvt_f16_f32_e32 v27.l, v27
	v_cvt_f16_f32_e32 v27.h, v28
	v_cvt_f16_f32_e32 v25.h, v26
	v_cvt_f16_f32_e32 v29.h, v30
	v_lshlrev_b64_e32 v[4:5], 2, v[32:33]
	v_add_co_u32 v32, vcc_lo, s2, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, s3, v8, vcc_lo
	v_cvt_f16_f32_e32 v21.l, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v34, vcc_lo, s2, v4
	v_cvt_f16_f32_e32 v0.l, v22
	v_cvt_f16_f32_e32 v19.h, v1
	v_cvt_f32_f16_e32 v1, v25.l
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_cvt_f16_f32_e32 v12.l, v9
	v_cvt_f16_f32_e32 v12.h, v10
	v_cvt_f16_f32_e32 v16.h, v6
	v_cvt_f16_f32_e32 v14.l, v3
	v_cvt_f32_f16_e32 v3, v25.h
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, s3, v5, vcc_lo
	v_cvt_f32_f16_e32 v5, v27.h
	v_cvt_f32_f16_e32 v4, v27.l
	v_cvt_f32_f16_e32 v6, v29.l
	v_cvt_f32_f16_e32 v24, v29.h
	v_cvt_f32_f16_e32 v26, v31.h
	v_cvt_f32_f16_e32 v25, v31.l
	v_cvt_f32_f16_e32 v8, v17.h
	v_cvt_f32_f16_e32 v7, v17.l
	v_cvt_f32_f16_e32 v10, v19.l
	v_cvt_f32_f16_e32 v9, v0.h
	v_cvt_f16_f32_e32 v17.l, v2
	s_clause 0x3
	global_store_b32 v[32:33], v1, off
	global_store_b128 v[34:35], v[3:6], off offset:4
	global_store_b96 v[34:35], v[24:26], off offset:20
	global_store_b128 v[34:35], v[7:10], off offset:64
	v_cvt_f32_f16_e32 v1, v0.l
	v_cvt_f32_f16_e32 v0, v21.l
	v_cvt_f32_f16_e32 v3, v23.h
	v_cvt_f32_f16_e32 v2, v23.l
	v_cvt_f32_f16_e32 v5, v12.h
	v_cvt_f32_f16_e32 v4, v12.l
	v_cvt_f32_f16_e32 v7, v11.h
	v_cvt_f32_f16_e32 v6, v11.l
	v_cvt_f32_f16_e32 v9, v13.h
	v_cvt_f32_f16_e32 v8, v13.l
	v_cvt_f32_f16_e32 v11, v15.h
	v_cvt_f32_f16_e32 v10, v15.l
	v_cvt_f32_f16_e32 v13, v17.l
	v_cvt_f32_f16_e32 v12, v19.h
	v_cvt_f32_f16_e32 v15, v14.h
	v_cvt_f32_f16_e32 v14, v14.l
	v_cvt_f32_f16_e32 v17, v16.h
	v_cvt_f32_f16_e32 v16, v16.l
	v_cvt_f32_f16_e32 v19, v18.h
	v_cvt_f32_f16_e32 v18, v18.l
	s_clause 0x4
	global_store_b128 v[34:35], v[0:3], off offset:80
	global_store_b128 v[34:35], v[4:7], off offset:128
	global_store_b128 v[34:35], v[8:11], off offset:144
	global_store_b128 v[34:35], v[12:15], off offset:192
	global_store_b128 v[34:35], v[16:19], off offset:208
.LBB0_4:                                ; %.loopexit
	s_endpgm
.Lfunc_end0:
	.size	_Z10carry_unitPKfPfi, .Lfunc_end0-_Z10carry_unitPKfPfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10carry_unitPKfPfi
		.amdhsa_group_segment_fixed_size 8192
		.amdhsa_private_segment_fixed_size 576
		.amdhsa_kernarg_size 20
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 95
		.amdhsa_next_free_sgpr 47
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_Z10carry_unitPKfPfi)<<4)&4080)>>4
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
	.set .L_Z10carry_unitPKfPfi.num_vgpr, 95
	.set .L_Z10carry_unitPKfPfi.num_agpr, 0
	.set .L_Z10carry_unitPKfPfi.numbered_sgpr, 47
	.set .L_Z10carry_unitPKfPfi.num_named_barrier, 0
	.set .L_Z10carry_unitPKfPfi.private_seg_size, 576
	.set .L_Z10carry_unitPKfPfi.uses_vcc, 1
	.set .L_Z10carry_unitPKfPfi.uses_flat_scratch, 1
	.set .L_Z10carry_unitPKfPfi.has_dyn_sized_stack, 0
	.set .L_Z10carry_unitPKfPfi.has_recursion, 0
	.set .L_Z10carry_unitPKfPfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7588
; TotalNumSgprs: 49
; NumVgprs: 95
; ScratchSize: 576
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 8192 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 49
; NumVGPRsForWavesPerEU: 95
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.type	__hip_cuid_e5d389edef073a4,@object ; @__hip_cuid_e5d389edef073a4
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e5d389edef073a4
__hip_cuid_e5d389edef073a4:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e5d389edef073a4, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e5d389edef073a4
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
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z10carry_unitPKfPfi
    .private_segment_fixed_size: 576
    .sgpr_count:     49
    .sgpr_spill_count: 0
    .symbol:         _Z10carry_unitPKfPfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     95
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
