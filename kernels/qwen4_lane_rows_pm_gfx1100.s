.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
.amdhsa_code_object_version 6
.text
.protected gemv_mq6g256v2_rows_t8_pm_gfx1100
.globl gemv_mq6g256v2_rows_t8_pm_gfx1100
.p2align 8
.type gemv_mq6g256v2_rows_t8_pm_gfx1100,@function
gemv_mq6g256v2_rows_t8_pm_gfx1100:
	s_load_b64 s[4:5], s[0:1], null
	s_load_b64 s[16:17], s[0:1], 0x8
	s_load_b64 s[32:33], s[0:1], 0x10
	s_load_b32 s34, s[0:1], 0x18
	s_load_b32 s30, s[0:1], 0x1c
	s_load_b32 s31, s[0:1], 0x20
	v_and_b32_e32 v5, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v5
	v_lshrrev_b32_e32 v12, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s40, v12
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v12, 10, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v10, v12, v11
	v_lshrrev_b32_e32 v13, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v13, 2, v13
	v_mul_u32_u24_e32 v14, 6, v5
	s_waitcnt lgkmcnt(0)
	s_max_i32 s39, s34, 0
	s_add_i32 s39, s39, 7
	s_lshr_b32 s39, s39, 3
	s_cmp_ge_u32 s14, s39
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_eq_u32 s31, 0
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_gt_u32 s31, 64
	s_cbranch_scc1 .Lmq6t8_end
	s_mov_b32 s35, 0
	s_cmp_le_i32 s30, 0
	s_cbranch_scc1 .Lmq6t8_end
	s_and_b32 s44, s30, 0xff
	s_cmp_lg_u32 s44, 0
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_ge_u32 s15, 8
	s_cbranch_scc1 .Lmq6t8_end
	s_lshl_b32 s54, s15, 3
	s_cmp_ge_u32 s54, s31
	s_cbranch_scc1 .Lmq6t8_end
	s_sub_u32 s31, s31, s54
	s_min_u32 s31, s31, 8
	s_mul_i32 s44, s54, s30
	s_lshl_b32 s44, s44, 2
	s_add_u32 s16, s16, s44
	s_addc_u32 s17, s17, 0
	s_mul_i32 s44, s54, s34
	s_lshl_b32 s44, s44, 2
	s_add_u32 s32, s32, s44
	s_addc_u32 s33, s33, 0
	s_sub_i32 s41, s14, s35
	s_lshl_b32 s41, s41, 3
	s_add_i32 s41, s41, s40
	s_cmp_lt_i32 s41, s34
	s_cselect_b32 s52, 1, 0
	s_lshr_b32 s42, s30, 8
	s_mul_i32 s43, s42, 0xc8
	s_mul_i32 s45, s41, s43
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_lshl_b32 s49, s30, 2
	s_lshl_b32 s50, s34, 2
	s_lshl_b32 s53, s41, 2
	s_add_i32 s44, s40, 0
	s_mul_i32 s56, s44, s49
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	s_mov_b32 s48, 0
	s_cmp_eq_u32 s42, 0
	s_cbranch_scc1 .Lmq6t8_fold
	.Lmq6t8_group:
	s_mul_i32 s46, s48, 0xc8
	s_lshl_b32 s47, s48, 10
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s47, v3
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6t8_s0t9_wdone
	v_add_nc_u32_e32 v1, s46, v13
	v_add_nc_u32_e32 v2, s46, v14
	buffer_load_b32 v7, v2, s[4:7], s45 offen offset:8
	buffer_load_u16 v8, v2, s[4:7], s45 offen offset:12
	buffer_load_b32 v6, v1, s[4:7], s45 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v16, 63, v7
	v_bfe_u32 v17, v7, 6, 6
	v_bfe_u32 v18, v7, 12, 6
	v_bfe_u32 v19, v7, 18, 6
	v_bfe_u32 v20, v7, 24, 6
	v_alignbit_b32 v21, v8, v7, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v21, 63, v21
	v_bfe_u32 v22, v8, 4, 6
	v_bfe_u32 v23, v8, 10, 6
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v6, v16, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v6, v17, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v6, v18, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v6, v19, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v6, v20, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v6, v21, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v6, v22, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v6, v23, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	.Lmq6t8_s0t9_wdone:
	s_add_i32 s44, s40, 0
	s_cmp_ge_u32 s44, s31
	s_cbranch_scc1 .Lmq6t8_s0t0_nofill
	v_add_nc_u32_e32 v12, s56, v3
	global_load_b128 v[24:27], v12, s[16:17]
	global_load_b128 v[28:31], v12, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v10, v24
	ds_store_b32 v10, v25 offset:128
	ds_store_b32 v10, v26 offset:256
	ds_store_b32 v10, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v10, v28 offset:512
	ds_store_b32 v10, v29 offset:640
	ds_store_b32 v10, v30 offset:768
	ds_store_b32 v10, v31 offset:896
	s_waitcnt lgkmcnt(0)
	.Lmq6t8_s0t0_nofill:
	s_barrier
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:128
	ds_load_b32 v24, v11
	ds_load_b32 v26, v11 offset:256
	ds_load_b32 v27, v11 offset:384
	ds_load_b32 v28, v11 offset:512
	ds_load_b32 v29, v11 offset:640
	ds_load_b32 v30, v11 offset:768
	ds_load_b32 v31, v11 offset:896
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v9
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:1152
	ds_load_b32 v24, v11 offset:1024
	ds_load_b32 v26, v11 offset:1280
	ds_load_b32 v27, v11 offset:1408
	ds_load_b32 v28, v11 offset:1536
	ds_load_b32 v29, v11 offset:1664
	ds_load_b32 v30, v11 offset:1792
	ds_load_b32 v31, v11 offset:1920
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v9
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:2176
	ds_load_b32 v24, v11 offset:2048
	ds_load_b32 v26, v11 offset:2304
	ds_load_b32 v27, v11 offset:2432
	ds_load_b32 v28, v11 offset:2560
	ds_load_b32 v29, v11 offset:2688
	ds_load_b32 v30, v11 offset:2816
	ds_load_b32 v31, v11 offset:2944
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v9
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:3200
	ds_load_b32 v24, v11 offset:3072
	ds_load_b32 v26, v11 offset:3328
	ds_load_b32 v27, v11 offset:3456
	ds_load_b32 v28, v11 offset:3584
	ds_load_b32 v29, v11 offset:3712
	ds_load_b32 v30, v11 offset:3840
	ds_load_b32 v31, v11 offset:3968
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v9
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:4224
	ds_load_b32 v24, v11 offset:4096
	ds_load_b32 v26, v11 offset:4352
	ds_load_b32 v27, v11 offset:4480
	ds_load_b32 v28, v11 offset:4608
	ds_load_b32 v29, v11 offset:4736
	ds_load_b32 v30, v11 offset:4864
	ds_load_b32 v31, v11 offset:4992
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v9
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:5248
	ds_load_b32 v24, v11 offset:5120
	ds_load_b32 v26, v11 offset:5376
	ds_load_b32 v27, v11 offset:5504
	ds_load_b32 v28, v11 offset:5632
	ds_load_b32 v29, v11 offset:5760
	ds_load_b32 v30, v11 offset:5888
	ds_load_b32 v31, v11 offset:6016
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v9
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:6272
	ds_load_b32 v24, v11 offset:6144
	ds_load_b32 v26, v11 offset:6400
	ds_load_b32 v27, v11 offset:6528
	ds_load_b32 v28, v11 offset:6656
	ds_load_b32 v29, v11 offset:6784
	ds_load_b32 v30, v11 offset:6912
	ds_load_b32 v31, v11 offset:7040
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v9
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:7296
	ds_load_b32 v24, v11 offset:7168
	ds_load_b32 v26, v11 offset:7424
	ds_load_b32 v27, v11 offset:7552
	ds_load_b32 v28, v11 offset:7680
	ds_load_b32 v29, v11 offset:7808
	ds_load_b32 v30, v11 offset:7936
	ds_load_b32 v31, v11 offset:8064
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v9
	.Lmq6t8_s0t0_cdone:
	s_barrier
	.Lmq6t8_s0t0_tile_done:
	s_add_i32 s48, s48, 1
	s_cmp_ge_u32 s48, s42
	s_cbranch_scc1 .Lmq6t8_fold
	s_mul_i32 s46, s48, 0xc8
	s_lshl_b32 s47, s48, 10
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s47, v3
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6t8_s1t9_wdone
	v_add_nc_u32_e32 v1, s46, v13
	v_add_nc_u32_e32 v2, s46, v14
	buffer_load_b32 v7, v2, s[4:7], s45 offen offset:8
	buffer_load_u16 v8, v2, s[4:7], s45 offen offset:12
	buffer_load_b32 v6, v1, s[4:7], s45 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v16, 63, v7
	v_bfe_u32 v17, v7, 6, 6
	v_bfe_u32 v18, v7, 12, 6
	v_bfe_u32 v19, v7, 18, 6
	v_bfe_u32 v20, v7, 24, 6
	v_alignbit_b32 v21, v8, v7, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v21, 63, v21
	v_bfe_u32 v22, v8, 4, 6
	v_bfe_u32 v23, v8, 10, 6
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v6, v16, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v6, v17, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v6, v18, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v6, v19, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v6, v20, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v6, v21, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v6, v22, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v6, v23, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	.Lmq6t8_s1t9_wdone:
	s_add_i32 s44, s40, 0
	s_cmp_ge_u32 s44, s31
	s_cbranch_scc1 .Lmq6t8_s1t0_nofill
	v_add_nc_u32_e32 v12, s56, v3
	global_load_b128 v[24:27], v12, s[16:17]
	global_load_b128 v[28:31], v12, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v10, v24
	ds_store_b32 v10, v25 offset:128
	ds_store_b32 v10, v26 offset:256
	ds_store_b32 v10, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v10, v28 offset:512
	ds_store_b32 v10, v29 offset:640
	ds_store_b32 v10, v30 offset:768
	ds_store_b32 v10, v31 offset:896
	s_waitcnt lgkmcnt(0)
	.Lmq6t8_s1t0_nofill:
	s_barrier
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:128
	ds_load_b32 v24, v11
	ds_load_b32 v26, v11 offset:256
	ds_load_b32 v27, v11 offset:384
	ds_load_b32 v28, v11 offset:512
	ds_load_b32 v29, v11 offset:640
	ds_load_b32 v30, v11 offset:768
	ds_load_b32 v31, v11 offset:896
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v9
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:1152
	ds_load_b32 v24, v11 offset:1024
	ds_load_b32 v26, v11 offset:1280
	ds_load_b32 v27, v11 offset:1408
	ds_load_b32 v28, v11 offset:1536
	ds_load_b32 v29, v11 offset:1664
	ds_load_b32 v30, v11 offset:1792
	ds_load_b32 v31, v11 offset:1920
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v9
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:2176
	ds_load_b32 v24, v11 offset:2048
	ds_load_b32 v26, v11 offset:2304
	ds_load_b32 v27, v11 offset:2432
	ds_load_b32 v28, v11 offset:2560
	ds_load_b32 v29, v11 offset:2688
	ds_load_b32 v30, v11 offset:2816
	ds_load_b32 v31, v11 offset:2944
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v9
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:3200
	ds_load_b32 v24, v11 offset:3072
	ds_load_b32 v26, v11 offset:3328
	ds_load_b32 v27, v11 offset:3456
	ds_load_b32 v28, v11 offset:3584
	ds_load_b32 v29, v11 offset:3712
	ds_load_b32 v30, v11 offset:3840
	ds_load_b32 v31, v11 offset:3968
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v9
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:4224
	ds_load_b32 v24, v11 offset:4096
	ds_load_b32 v26, v11 offset:4352
	ds_load_b32 v27, v11 offset:4480
	ds_load_b32 v28, v11 offset:4608
	ds_load_b32 v29, v11 offset:4736
	ds_load_b32 v30, v11 offset:4864
	ds_load_b32 v31, v11 offset:4992
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v9
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:5248
	ds_load_b32 v24, v11 offset:5120
	ds_load_b32 v26, v11 offset:5376
	ds_load_b32 v27, v11 offset:5504
	ds_load_b32 v28, v11 offset:5632
	ds_load_b32 v29, v11 offset:5760
	ds_load_b32 v30, v11 offset:5888
	ds_load_b32 v31, v11 offset:6016
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v9
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:6272
	ds_load_b32 v24, v11 offset:6144
	ds_load_b32 v26, v11 offset:6400
	ds_load_b32 v27, v11 offset:6528
	ds_load_b32 v28, v11 offset:6656
	ds_load_b32 v29, v11 offset:6784
	ds_load_b32 v30, v11 offset:6912
	ds_load_b32 v31, v11 offset:7040
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v9
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:7296
	ds_load_b32 v24, v11 offset:7168
	ds_load_b32 v26, v11 offset:7424
	ds_load_b32 v27, v11 offset:7552
	ds_load_b32 v28, v11 offset:7680
	ds_load_b32 v29, v11 offset:7808
	ds_load_b32 v30, v11 offset:7936
	ds_load_b32 v31, v11 offset:8064
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v9
	.Lmq6t8_s1t0_cdone:
	s_barrier
	.Lmq6t8_s1t0_tile_done:
	s_add_i32 s48, s48, 1
	s_cmp_ge_u32 s48, s42
	s_cbranch_scc1 .Lmq6t8_fold
	s_mul_i32 s46, s48, 0xc8
	s_lshl_b32 s47, s48, 10
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s47, v3
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6t8_s2t9_wdone
	v_add_nc_u32_e32 v1, s46, v13
	v_add_nc_u32_e32 v2, s46, v14
	buffer_load_b32 v7, v2, s[4:7], s45 offen offset:8
	buffer_load_u16 v8, v2, s[4:7], s45 offen offset:12
	buffer_load_b32 v6, v1, s[4:7], s45 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v16, 63, v7
	v_bfe_u32 v17, v7, 6, 6
	v_bfe_u32 v18, v7, 12, 6
	v_bfe_u32 v19, v7, 18, 6
	v_bfe_u32 v20, v7, 24, 6
	v_alignbit_b32 v21, v8, v7, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v21, 63, v21
	v_bfe_u32 v22, v8, 4, 6
	v_bfe_u32 v23, v8, 10, 6
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v6, v16, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v6, v17, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v6, v18, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v6, v19, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v6, v20, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v6, v21, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v6, v22, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v6, v23, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	.Lmq6t8_s2t9_wdone:
	s_add_i32 s44, s40, 0
	s_cmp_ge_u32 s44, s31
	s_cbranch_scc1 .Lmq6t8_s2t0_nofill
	v_add_nc_u32_e32 v12, s56, v3
	global_load_b128 v[24:27], v12, s[16:17]
	global_load_b128 v[28:31], v12, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v10, v24
	ds_store_b32 v10, v25 offset:128
	ds_store_b32 v10, v26 offset:256
	ds_store_b32 v10, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v10, v28 offset:512
	ds_store_b32 v10, v29 offset:640
	ds_store_b32 v10, v30 offset:768
	ds_store_b32 v10, v31 offset:896
	s_waitcnt lgkmcnt(0)
	.Lmq6t8_s2t0_nofill:
	s_barrier
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:128
	ds_load_b32 v24, v11
	ds_load_b32 v26, v11 offset:256
	ds_load_b32 v27, v11 offset:384
	ds_load_b32 v28, v11 offset:512
	ds_load_b32 v29, v11 offset:640
	ds_load_b32 v30, v11 offset:768
	ds_load_b32 v31, v11 offset:896
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v9
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:1152
	ds_load_b32 v24, v11 offset:1024
	ds_load_b32 v26, v11 offset:1280
	ds_load_b32 v27, v11 offset:1408
	ds_load_b32 v28, v11 offset:1536
	ds_load_b32 v29, v11 offset:1664
	ds_load_b32 v30, v11 offset:1792
	ds_load_b32 v31, v11 offset:1920
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v9
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:2176
	ds_load_b32 v24, v11 offset:2048
	ds_load_b32 v26, v11 offset:2304
	ds_load_b32 v27, v11 offset:2432
	ds_load_b32 v28, v11 offset:2560
	ds_load_b32 v29, v11 offset:2688
	ds_load_b32 v30, v11 offset:2816
	ds_load_b32 v31, v11 offset:2944
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v9
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:3200
	ds_load_b32 v24, v11 offset:3072
	ds_load_b32 v26, v11 offset:3328
	ds_load_b32 v27, v11 offset:3456
	ds_load_b32 v28, v11 offset:3584
	ds_load_b32 v29, v11 offset:3712
	ds_load_b32 v30, v11 offset:3840
	ds_load_b32 v31, v11 offset:3968
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v9
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:4224
	ds_load_b32 v24, v11 offset:4096
	ds_load_b32 v26, v11 offset:4352
	ds_load_b32 v27, v11 offset:4480
	ds_load_b32 v28, v11 offset:4608
	ds_load_b32 v29, v11 offset:4736
	ds_load_b32 v30, v11 offset:4864
	ds_load_b32 v31, v11 offset:4992
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v9
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:5248
	ds_load_b32 v24, v11 offset:5120
	ds_load_b32 v26, v11 offset:5376
	ds_load_b32 v27, v11 offset:5504
	ds_load_b32 v28, v11 offset:5632
	ds_load_b32 v29, v11 offset:5760
	ds_load_b32 v30, v11 offset:5888
	ds_load_b32 v31, v11 offset:6016
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v9
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:6272
	ds_load_b32 v24, v11 offset:6144
	ds_load_b32 v26, v11 offset:6400
	ds_load_b32 v27, v11 offset:6528
	ds_load_b32 v28, v11 offset:6656
	ds_load_b32 v29, v11 offset:6784
	ds_load_b32 v30, v11 offset:6912
	ds_load_b32 v31, v11 offset:7040
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v9
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:7296
	ds_load_b32 v24, v11 offset:7168
	ds_load_b32 v26, v11 offset:7424
	ds_load_b32 v27, v11 offset:7552
	ds_load_b32 v28, v11 offset:7680
	ds_load_b32 v29, v11 offset:7808
	ds_load_b32 v30, v11 offset:7936
	ds_load_b32 v31, v11 offset:8064
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v9
	.Lmq6t8_s2t0_cdone:
	s_barrier
	.Lmq6t8_s2t0_tile_done:
	s_add_i32 s48, s48, 1
	s_cmp_ge_u32 s48, s42
	s_cbranch_scc1 .Lmq6t8_fold
	s_mul_i32 s46, s48, 0xc8
	s_lshl_b32 s47, s48, 10
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s47, v3
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6t8_s3t9_wdone
	v_add_nc_u32_e32 v1, s46, v13
	v_add_nc_u32_e32 v2, s46, v14
	buffer_load_b32 v7, v2, s[4:7], s45 offen offset:8
	buffer_load_u16 v8, v2, s[4:7], s45 offen offset:12
	buffer_load_b32 v6, v1, s[4:7], s45 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v16, 63, v7
	v_bfe_u32 v17, v7, 6, 6
	v_bfe_u32 v18, v7, 12, 6
	v_bfe_u32 v19, v7, 18, 6
	v_bfe_u32 v20, v7, 24, 6
	v_alignbit_b32 v21, v8, v7, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v21, 63, v21
	v_bfe_u32 v22, v8, 4, 6
	v_bfe_u32 v23, v8, 10, 6
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v6, v16, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v6, v17, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v6, v18, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v6, v19, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v6, v20, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v6, v21, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v6, v22, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v6, v23, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	.Lmq6t8_s3t9_wdone:
	s_add_i32 s44, s40, 0
	s_cmp_ge_u32 s44, s31
	s_cbranch_scc1 .Lmq6t8_s3t0_nofill
	v_add_nc_u32_e32 v12, s56, v3
	global_load_b128 v[24:27], v12, s[16:17]
	global_load_b128 v[28:31], v12, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v10, v24
	ds_store_b32 v10, v25 offset:128
	ds_store_b32 v10, v26 offset:256
	ds_store_b32 v10, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v10, v28 offset:512
	ds_store_b32 v10, v29 offset:640
	ds_store_b32 v10, v30 offset:768
	ds_store_b32 v10, v31 offset:896
	s_waitcnt lgkmcnt(0)
	.Lmq6t8_s3t0_nofill:
	s_barrier
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:128
	ds_load_b32 v24, v11
	ds_load_b32 v26, v11 offset:256
	ds_load_b32 v27, v11 offset:384
	ds_load_b32 v28, v11 offset:512
	ds_load_b32 v29, v11 offset:640
	ds_load_b32 v30, v11 offset:768
	ds_load_b32 v31, v11 offset:896
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v9
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:1152
	ds_load_b32 v24, v11 offset:1024
	ds_load_b32 v26, v11 offset:1280
	ds_load_b32 v27, v11 offset:1408
	ds_load_b32 v28, v11 offset:1536
	ds_load_b32 v29, v11 offset:1664
	ds_load_b32 v30, v11 offset:1792
	ds_load_b32 v31, v11 offset:1920
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v9
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:2176
	ds_load_b32 v24, v11 offset:2048
	ds_load_b32 v26, v11 offset:2304
	ds_load_b32 v27, v11 offset:2432
	ds_load_b32 v28, v11 offset:2560
	ds_load_b32 v29, v11 offset:2688
	ds_load_b32 v30, v11 offset:2816
	ds_load_b32 v31, v11 offset:2944
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v9
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:3200
	ds_load_b32 v24, v11 offset:3072
	ds_load_b32 v26, v11 offset:3328
	ds_load_b32 v27, v11 offset:3456
	ds_load_b32 v28, v11 offset:3584
	ds_load_b32 v29, v11 offset:3712
	ds_load_b32 v30, v11 offset:3840
	ds_load_b32 v31, v11 offset:3968
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v9
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:4224
	ds_load_b32 v24, v11 offset:4096
	ds_load_b32 v26, v11 offset:4352
	ds_load_b32 v27, v11 offset:4480
	ds_load_b32 v28, v11 offset:4608
	ds_load_b32 v29, v11 offset:4736
	ds_load_b32 v30, v11 offset:4864
	ds_load_b32 v31, v11 offset:4992
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v9
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:5248
	ds_load_b32 v24, v11 offset:5120
	ds_load_b32 v26, v11 offset:5376
	ds_load_b32 v27, v11 offset:5504
	ds_load_b32 v28, v11 offset:5632
	ds_load_b32 v29, v11 offset:5760
	ds_load_b32 v30, v11 offset:5888
	ds_load_b32 v31, v11 offset:6016
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v9
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:6272
	ds_load_b32 v24, v11 offset:6144
	ds_load_b32 v26, v11 offset:6400
	ds_load_b32 v27, v11 offset:6528
	ds_load_b32 v28, v11 offset:6656
	ds_load_b32 v29, v11 offset:6784
	ds_load_b32 v30, v11 offset:6912
	ds_load_b32 v31, v11 offset:7040
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v9
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:7296
	ds_load_b32 v24, v11 offset:7168
	ds_load_b32 v26, v11 offset:7424
	ds_load_b32 v27, v11 offset:7552
	ds_load_b32 v28, v11 offset:7680
	ds_load_b32 v29, v11 offset:7808
	ds_load_b32 v30, v11 offset:7936
	ds_load_b32 v31, v11 offset:8064
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v9
	.Lmq6t8_s3t0_cdone:
	s_barrier
	.Lmq6t8_s3t0_tile_done:
	s_add_i32 s48, s48, 1
	s_cmp_lt_u32 s48, s42
	s_cbranch_scc1 .Lmq6t8_group
	.Lmq6t8_fold:
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_le_u32 s31, 0
	s_cbranch_scc1 .Lmq6t8_fold_done
	v_add_f32_e32 v40, v40, v41
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v43
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6t8_fold_done
	v_add_f32_e32 v44, v44, v45
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v47
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6t8_fold_done
	v_add_f32_e32 v48, v48, v49
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v51
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6t8_fold_done
	v_add_f32_e32 v52, v52, v53
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v55
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6t8_fold_done
	v_add_f32_e32 v56, v56, v57
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v59
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6t8_fold_done
	v_add_f32_e32 v60, v60, v61
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v63
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6t8_fold_done
	v_add_f32_e32 v64, v64, v65
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v67
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6t8_fold_done
	v_add_f32_e32 v68, v68, v69
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v71
	.Lmq6t8_fold_done:
	s_cmp_le_u32 s31, 0
	s_cbranch_scc1 .Lmq6t8_reduce_done
	ds_swizzle_b32 v24, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v40, v40, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v40, v40, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v40, v40, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v40, v40, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v40, v40, v24
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6t8_reduce_done
	ds_swizzle_b32 v24, v44 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v44, v44, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v44, v44, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v44, v44, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v44, v44, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v44, v44, v24
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6t8_reduce_done
	ds_swizzle_b32 v24, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v48, v48, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v48, v48, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v48, v48, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v48, v48, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v48, v48, v24
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6t8_reduce_done
	ds_swizzle_b32 v24, v52 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v52, v52, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v52, v52, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v52, v52, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v52, v52, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v52, v52, v24
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6t8_reduce_done
	ds_swizzle_b32 v24, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v56, v56, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v56, v56, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v56, v56, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v56, v56, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v56, v56, v24
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6t8_reduce_done
	ds_swizzle_b32 v24, v60 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v60, v60, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v60, v60, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v60, v60, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v60, v60, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v60, v60, v24
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6t8_reduce_done
	ds_swizzle_b32 v24, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v64, v64, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v64, v64, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v64, v64, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v64, v64, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v64, v64, v24
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6t8_reduce_done
	ds_swizzle_b32 v24, v68 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v68, v68, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v68, v68, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v68, v68, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v68, v68, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v68, v68, v24
	.Lmq6t8_reduce_done:
	v_cmpx_eq_u32_e32 0, v5
	s_cmp_le_u32 s31, 0
	s_cbranch_scc1 .Lmq6t8_end
	s_mul_i32 s51, s50, 0
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 0
	s_cbranch_scc1 .Lmq6t8_stored0
	global_store_b32 v24, v40, s[32:33]
	.Lmq6t8_stored0:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6t8_end
	s_mul_i32 s51, s50, 1
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6t8_stored1
	global_store_b32 v24, v44, s[32:33]
	.Lmq6t8_stored1:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6t8_end
	s_mul_i32 s51, s50, 2
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6t8_stored2
	global_store_b32 v24, v48, s[32:33]
	.Lmq6t8_stored2:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6t8_end
	s_mul_i32 s51, s50, 3
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6t8_stored3
	global_store_b32 v24, v52, s[32:33]
	.Lmq6t8_stored3:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6t8_end
	s_mul_i32 s51, s50, 4
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6t8_stored4
	global_store_b32 v24, v56, s[32:33]
	.Lmq6t8_stored4:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6t8_end
	s_mul_i32 s51, s50, 5
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6t8_stored5
	global_store_b32 v24, v60, s[32:33]
	.Lmq6t8_stored5:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6t8_end
	s_mul_i32 s51, s50, 6
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6t8_stored6
	global_store_b32 v24, v64, s[32:33]
	.Lmq6t8_stored6:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6t8_end
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6t8_end
	s_mul_i32 s51, s50, 7
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6t8_stored7
	global_store_b32 v24, v68, s[32:33]
	.Lmq6t8_stored7:
	s_waitcnt_vscnt null, 0x0
	.Lmq6t8_end:
	s_endpgm
.Lgemv_mq6g256v2_rows_t8_pm_gfx1100_end:
.size gemv_mq6g256v2_rows_t8_pm_gfx1100, .Lgemv_mq6g256v2_rows_t8_pm_gfx1100_end-gemv_mq6g256v2_rows_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_mq6g256v2_rows_t8_pm_gfx1100
	.amdhsa_group_segment_fixed_size 8192
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 36
	.amdhsa_user_sgpr_count 14
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
	.amdhsa_next_free_vgpr 72
	.amdhsa_next_free_sgpr 60
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_mq6g256v2_rows_t8_pm_gfx1100_end-gemv_mq6g256v2_rows_t8_pm_gfx1100)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.text
.protected gemv_mq6g256v2_rows_x4_t8_pm_gfx1100
.globl gemv_mq6g256v2_rows_x4_t8_pm_gfx1100
.p2align 8
.type gemv_mq6g256v2_rows_x4_t8_pm_gfx1100,@function
gemv_mq6g256v2_rows_x4_t8_pm_gfx1100:
	s_load_b256 s[64:71], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	v_and_b32_e32 v5, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v5
	v_lshrrev_b32_e32 v12, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s40, v12
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v12, 10, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v10, v12, v11
	v_lshrrev_b32_e32 v13, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v13, 2, v13
	v_mul_u32_u24_e32 v14, 6, v5
	s_waitcnt lgkmcnt(0)
	s_max_i32 s36, s26, 0
	s_add_i32 s36, s36, 7
	s_lshr_b32 s36, s36, 3
	s_max_i32 s44, s27, 0
	s_add_i32 s44, s44, 7
	s_lshr_b32 s44, s44, 3
	s_add_i32 s37, s36, s44
	s_max_i32 s44, s28, 0
	s_add_i32 s44, s44, 7
	s_lshr_b32 s44, s44, 3
	s_add_i32 s38, s37, s44
	s_max_i32 s44, s29, 0
	s_add_i32 s44, s44, 7
	s_lshr_b32 s44, s44, 3
	s_add_i32 s39, s38, s44
	s_cmp_ge_u32 s14, s39
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_eq_u32 s31, 0
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_gt_u32 s31, 64
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_lt_u32 s14, s38
	s_cselect_b64 s[4:5], s[68:69], s[70:71]
	s_cselect_b64 s[32:33], s[22:23], s[24:25]
	s_cselect_b32 s34, s28, s29
	s_cselect_b32 s35, s37, s38
	s_cmp_lt_u32 s14, s37
	s_cselect_b64 s[4:5], s[66:67], s[4:5]
	s_cselect_b64 s[32:33], s[20:21], s[32:33]
	s_cselect_b32 s34, s27, s34
	s_cselect_b32 s35, s36, s35
	s_cmp_lt_u32 s14, s36
	s_cselect_b64 s[4:5], s[64:65], s[4:5]
	s_cselect_b64 s[32:33], s[18:19], s[32:33]
	s_cselect_b32 s34, s26, s34
	s_cselect_b32 s35, 0, s35
	s_cmp_le_i32 s30, 0
	s_cbranch_scc1 .Lmq6x4t8_end
	s_and_b32 s44, s30, 0xff
	s_cmp_lg_u32 s44, 0
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_ge_u32 s15, 8
	s_cbranch_scc1 .Lmq6x4t8_end
	s_lshl_b32 s54, s15, 3
	s_cmp_ge_u32 s54, s31
	s_cbranch_scc1 .Lmq6x4t8_end
	s_sub_u32 s31, s31, s54
	s_min_u32 s31, s31, 8
	s_mul_i32 s44, s54, s30
	s_lshl_b32 s44, s44, 2
	s_add_u32 s16, s16, s44
	s_addc_u32 s17, s17, 0
	s_mul_i32 s44, s54, s34
	s_lshl_b32 s44, s44, 2
	s_add_u32 s32, s32, s44
	s_addc_u32 s33, s33, 0
	s_sub_i32 s41, s14, s35
	s_lshl_b32 s41, s41, 3
	s_add_i32 s41, s41, s40
	s_cmp_lt_i32 s41, s34
	s_cselect_b32 s52, 1, 0
	s_lshr_b32 s42, s30, 8
	s_mul_i32 s43, s42, 0xc8
	s_mul_i32 s45, s41, s43
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_lshl_b32 s49, s30, 2
	s_lshl_b32 s50, s34, 2
	s_lshl_b32 s53, s41, 2
	s_add_i32 s44, s40, 0
	s_mul_i32 s56, s44, s49
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	s_mov_b32 s48, 0
	s_cmp_eq_u32 s42, 0
	s_cbranch_scc1 .Lmq6x4t8_fold
	.Lmq6x4t8_group:
	s_mul_i32 s46, s48, 0xc8
	s_lshl_b32 s47, s48, 10
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s47, v3
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6x4t8_s0t9_wdone
	v_add_nc_u32_e32 v1, s46, v13
	v_add_nc_u32_e32 v2, s46, v14
	buffer_load_b32 v7, v2, s[4:7], s45 offen offset:8
	buffer_load_u16 v8, v2, s[4:7], s45 offen offset:12
	buffer_load_b32 v6, v1, s[4:7], s45 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v16, 63, v7
	v_bfe_u32 v17, v7, 6, 6
	v_bfe_u32 v18, v7, 12, 6
	v_bfe_u32 v19, v7, 18, 6
	v_bfe_u32 v20, v7, 24, 6
	v_alignbit_b32 v21, v8, v7, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v21, 63, v21
	v_bfe_u32 v22, v8, 4, 6
	v_bfe_u32 v23, v8, 10, 6
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v6, v16, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v6, v17, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v6, v18, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v6, v19, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v6, v20, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v6, v21, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v6, v22, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v6, v23, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	.Lmq6x4t8_s0t9_wdone:
	s_add_i32 s44, s40, 0
	s_cmp_ge_u32 s44, s31
	s_cbranch_scc1 .Lmq6x4t8_s0t0_nofill
	v_add_nc_u32_e32 v12, s56, v3
	global_load_b128 v[24:27], v12, s[16:17]
	global_load_b128 v[28:31], v12, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v10, v24
	ds_store_b32 v10, v25 offset:128
	ds_store_b32 v10, v26 offset:256
	ds_store_b32 v10, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v10, v28 offset:512
	ds_store_b32 v10, v29 offset:640
	ds_store_b32 v10, v30 offset:768
	ds_store_b32 v10, v31 offset:896
	s_waitcnt lgkmcnt(0)
	.Lmq6x4t8_s0t0_nofill:
	s_barrier
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6x4t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:128
	ds_load_b32 v24, v11
	ds_load_b32 v26, v11 offset:256
	ds_load_b32 v27, v11 offset:384
	ds_load_b32 v28, v11 offset:512
	ds_load_b32 v29, v11 offset:640
	ds_load_b32 v30, v11 offset:768
	ds_load_b32 v31, v11 offset:896
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v9
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6x4t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:1152
	ds_load_b32 v24, v11 offset:1024
	ds_load_b32 v26, v11 offset:1280
	ds_load_b32 v27, v11 offset:1408
	ds_load_b32 v28, v11 offset:1536
	ds_load_b32 v29, v11 offset:1664
	ds_load_b32 v30, v11 offset:1792
	ds_load_b32 v31, v11 offset:1920
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v9
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6x4t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:2176
	ds_load_b32 v24, v11 offset:2048
	ds_load_b32 v26, v11 offset:2304
	ds_load_b32 v27, v11 offset:2432
	ds_load_b32 v28, v11 offset:2560
	ds_load_b32 v29, v11 offset:2688
	ds_load_b32 v30, v11 offset:2816
	ds_load_b32 v31, v11 offset:2944
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v9
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6x4t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:3200
	ds_load_b32 v24, v11 offset:3072
	ds_load_b32 v26, v11 offset:3328
	ds_load_b32 v27, v11 offset:3456
	ds_load_b32 v28, v11 offset:3584
	ds_load_b32 v29, v11 offset:3712
	ds_load_b32 v30, v11 offset:3840
	ds_load_b32 v31, v11 offset:3968
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v9
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6x4t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:4224
	ds_load_b32 v24, v11 offset:4096
	ds_load_b32 v26, v11 offset:4352
	ds_load_b32 v27, v11 offset:4480
	ds_load_b32 v28, v11 offset:4608
	ds_load_b32 v29, v11 offset:4736
	ds_load_b32 v30, v11 offset:4864
	ds_load_b32 v31, v11 offset:4992
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v9
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6x4t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:5248
	ds_load_b32 v24, v11 offset:5120
	ds_load_b32 v26, v11 offset:5376
	ds_load_b32 v27, v11 offset:5504
	ds_load_b32 v28, v11 offset:5632
	ds_load_b32 v29, v11 offset:5760
	ds_load_b32 v30, v11 offset:5888
	ds_load_b32 v31, v11 offset:6016
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v9
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6x4t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:6272
	ds_load_b32 v24, v11 offset:6144
	ds_load_b32 v26, v11 offset:6400
	ds_load_b32 v27, v11 offset:6528
	ds_load_b32 v28, v11 offset:6656
	ds_load_b32 v29, v11 offset:6784
	ds_load_b32 v30, v11 offset:6912
	ds_load_b32 v31, v11 offset:7040
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v9
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6x4t8_s0t0_cdone
	ds_load_b32 v25, v11 offset:7296
	ds_load_b32 v24, v11 offset:7168
	ds_load_b32 v26, v11 offset:7424
	ds_load_b32 v27, v11 offset:7552
	ds_load_b32 v28, v11 offset:7680
	ds_load_b32 v29, v11 offset:7808
	ds_load_b32 v30, v11 offset:7936
	ds_load_b32 v31, v11 offset:8064
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v9
	.Lmq6x4t8_s0t0_cdone:
	s_barrier
	.Lmq6x4t8_s0t0_tile_done:
	s_add_i32 s48, s48, 1
	s_cmp_ge_u32 s48, s42
	s_cbranch_scc1 .Lmq6x4t8_fold
	s_mul_i32 s46, s48, 0xc8
	s_lshl_b32 s47, s48, 10
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s47, v3
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6x4t8_s1t9_wdone
	v_add_nc_u32_e32 v1, s46, v13
	v_add_nc_u32_e32 v2, s46, v14
	buffer_load_b32 v7, v2, s[4:7], s45 offen offset:8
	buffer_load_u16 v8, v2, s[4:7], s45 offen offset:12
	buffer_load_b32 v6, v1, s[4:7], s45 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v16, 63, v7
	v_bfe_u32 v17, v7, 6, 6
	v_bfe_u32 v18, v7, 12, 6
	v_bfe_u32 v19, v7, 18, 6
	v_bfe_u32 v20, v7, 24, 6
	v_alignbit_b32 v21, v8, v7, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v21, 63, v21
	v_bfe_u32 v22, v8, 4, 6
	v_bfe_u32 v23, v8, 10, 6
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v6, v16, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v6, v17, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v6, v18, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v6, v19, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v6, v20, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v6, v21, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v6, v22, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v6, v23, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	.Lmq6x4t8_s1t9_wdone:
	s_add_i32 s44, s40, 0
	s_cmp_ge_u32 s44, s31
	s_cbranch_scc1 .Lmq6x4t8_s1t0_nofill
	v_add_nc_u32_e32 v12, s56, v3
	global_load_b128 v[24:27], v12, s[16:17]
	global_load_b128 v[28:31], v12, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v10, v24
	ds_store_b32 v10, v25 offset:128
	ds_store_b32 v10, v26 offset:256
	ds_store_b32 v10, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v10, v28 offset:512
	ds_store_b32 v10, v29 offset:640
	ds_store_b32 v10, v30 offset:768
	ds_store_b32 v10, v31 offset:896
	s_waitcnt lgkmcnt(0)
	.Lmq6x4t8_s1t0_nofill:
	s_barrier
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6x4t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:128
	ds_load_b32 v24, v11
	ds_load_b32 v26, v11 offset:256
	ds_load_b32 v27, v11 offset:384
	ds_load_b32 v28, v11 offset:512
	ds_load_b32 v29, v11 offset:640
	ds_load_b32 v30, v11 offset:768
	ds_load_b32 v31, v11 offset:896
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v9
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6x4t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:1152
	ds_load_b32 v24, v11 offset:1024
	ds_load_b32 v26, v11 offset:1280
	ds_load_b32 v27, v11 offset:1408
	ds_load_b32 v28, v11 offset:1536
	ds_load_b32 v29, v11 offset:1664
	ds_load_b32 v30, v11 offset:1792
	ds_load_b32 v31, v11 offset:1920
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v9
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6x4t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:2176
	ds_load_b32 v24, v11 offset:2048
	ds_load_b32 v26, v11 offset:2304
	ds_load_b32 v27, v11 offset:2432
	ds_load_b32 v28, v11 offset:2560
	ds_load_b32 v29, v11 offset:2688
	ds_load_b32 v30, v11 offset:2816
	ds_load_b32 v31, v11 offset:2944
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v9
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6x4t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:3200
	ds_load_b32 v24, v11 offset:3072
	ds_load_b32 v26, v11 offset:3328
	ds_load_b32 v27, v11 offset:3456
	ds_load_b32 v28, v11 offset:3584
	ds_load_b32 v29, v11 offset:3712
	ds_load_b32 v30, v11 offset:3840
	ds_load_b32 v31, v11 offset:3968
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v9
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6x4t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:4224
	ds_load_b32 v24, v11 offset:4096
	ds_load_b32 v26, v11 offset:4352
	ds_load_b32 v27, v11 offset:4480
	ds_load_b32 v28, v11 offset:4608
	ds_load_b32 v29, v11 offset:4736
	ds_load_b32 v30, v11 offset:4864
	ds_load_b32 v31, v11 offset:4992
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v9
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6x4t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:5248
	ds_load_b32 v24, v11 offset:5120
	ds_load_b32 v26, v11 offset:5376
	ds_load_b32 v27, v11 offset:5504
	ds_load_b32 v28, v11 offset:5632
	ds_load_b32 v29, v11 offset:5760
	ds_load_b32 v30, v11 offset:5888
	ds_load_b32 v31, v11 offset:6016
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v9
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6x4t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:6272
	ds_load_b32 v24, v11 offset:6144
	ds_load_b32 v26, v11 offset:6400
	ds_load_b32 v27, v11 offset:6528
	ds_load_b32 v28, v11 offset:6656
	ds_load_b32 v29, v11 offset:6784
	ds_load_b32 v30, v11 offset:6912
	ds_load_b32 v31, v11 offset:7040
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v9
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6x4t8_s1t0_cdone
	ds_load_b32 v25, v11 offset:7296
	ds_load_b32 v24, v11 offset:7168
	ds_load_b32 v26, v11 offset:7424
	ds_load_b32 v27, v11 offset:7552
	ds_load_b32 v28, v11 offset:7680
	ds_load_b32 v29, v11 offset:7808
	ds_load_b32 v30, v11 offset:7936
	ds_load_b32 v31, v11 offset:8064
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v9
	.Lmq6x4t8_s1t0_cdone:
	s_barrier
	.Lmq6x4t8_s1t0_tile_done:
	s_add_i32 s48, s48, 1
	s_cmp_ge_u32 s48, s42
	s_cbranch_scc1 .Lmq6x4t8_fold
	s_mul_i32 s46, s48, 0xc8
	s_lshl_b32 s47, s48, 10
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s47, v3
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6x4t8_s2t9_wdone
	v_add_nc_u32_e32 v1, s46, v13
	v_add_nc_u32_e32 v2, s46, v14
	buffer_load_b32 v7, v2, s[4:7], s45 offen offset:8
	buffer_load_u16 v8, v2, s[4:7], s45 offen offset:12
	buffer_load_b32 v6, v1, s[4:7], s45 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v16, 63, v7
	v_bfe_u32 v17, v7, 6, 6
	v_bfe_u32 v18, v7, 12, 6
	v_bfe_u32 v19, v7, 18, 6
	v_bfe_u32 v20, v7, 24, 6
	v_alignbit_b32 v21, v8, v7, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v21, 63, v21
	v_bfe_u32 v22, v8, 4, 6
	v_bfe_u32 v23, v8, 10, 6
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v6, v16, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v6, v17, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v6, v18, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v6, v19, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v6, v20, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v6, v21, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v6, v22, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v6, v23, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	.Lmq6x4t8_s2t9_wdone:
	s_add_i32 s44, s40, 0
	s_cmp_ge_u32 s44, s31
	s_cbranch_scc1 .Lmq6x4t8_s2t0_nofill
	v_add_nc_u32_e32 v12, s56, v3
	global_load_b128 v[24:27], v12, s[16:17]
	global_load_b128 v[28:31], v12, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v10, v24
	ds_store_b32 v10, v25 offset:128
	ds_store_b32 v10, v26 offset:256
	ds_store_b32 v10, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v10, v28 offset:512
	ds_store_b32 v10, v29 offset:640
	ds_store_b32 v10, v30 offset:768
	ds_store_b32 v10, v31 offset:896
	s_waitcnt lgkmcnt(0)
	.Lmq6x4t8_s2t0_nofill:
	s_barrier
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6x4t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:128
	ds_load_b32 v24, v11
	ds_load_b32 v26, v11 offset:256
	ds_load_b32 v27, v11 offset:384
	ds_load_b32 v28, v11 offset:512
	ds_load_b32 v29, v11 offset:640
	ds_load_b32 v30, v11 offset:768
	ds_load_b32 v31, v11 offset:896
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v9
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6x4t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:1152
	ds_load_b32 v24, v11 offset:1024
	ds_load_b32 v26, v11 offset:1280
	ds_load_b32 v27, v11 offset:1408
	ds_load_b32 v28, v11 offset:1536
	ds_load_b32 v29, v11 offset:1664
	ds_load_b32 v30, v11 offset:1792
	ds_load_b32 v31, v11 offset:1920
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v9
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6x4t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:2176
	ds_load_b32 v24, v11 offset:2048
	ds_load_b32 v26, v11 offset:2304
	ds_load_b32 v27, v11 offset:2432
	ds_load_b32 v28, v11 offset:2560
	ds_load_b32 v29, v11 offset:2688
	ds_load_b32 v30, v11 offset:2816
	ds_load_b32 v31, v11 offset:2944
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v9
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6x4t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:3200
	ds_load_b32 v24, v11 offset:3072
	ds_load_b32 v26, v11 offset:3328
	ds_load_b32 v27, v11 offset:3456
	ds_load_b32 v28, v11 offset:3584
	ds_load_b32 v29, v11 offset:3712
	ds_load_b32 v30, v11 offset:3840
	ds_load_b32 v31, v11 offset:3968
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v9
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6x4t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:4224
	ds_load_b32 v24, v11 offset:4096
	ds_load_b32 v26, v11 offset:4352
	ds_load_b32 v27, v11 offset:4480
	ds_load_b32 v28, v11 offset:4608
	ds_load_b32 v29, v11 offset:4736
	ds_load_b32 v30, v11 offset:4864
	ds_load_b32 v31, v11 offset:4992
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v9
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6x4t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:5248
	ds_load_b32 v24, v11 offset:5120
	ds_load_b32 v26, v11 offset:5376
	ds_load_b32 v27, v11 offset:5504
	ds_load_b32 v28, v11 offset:5632
	ds_load_b32 v29, v11 offset:5760
	ds_load_b32 v30, v11 offset:5888
	ds_load_b32 v31, v11 offset:6016
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v9
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6x4t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:6272
	ds_load_b32 v24, v11 offset:6144
	ds_load_b32 v26, v11 offset:6400
	ds_load_b32 v27, v11 offset:6528
	ds_load_b32 v28, v11 offset:6656
	ds_load_b32 v29, v11 offset:6784
	ds_load_b32 v30, v11 offset:6912
	ds_load_b32 v31, v11 offset:7040
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v9
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6x4t8_s2t0_cdone
	ds_load_b32 v25, v11 offset:7296
	ds_load_b32 v24, v11 offset:7168
	ds_load_b32 v26, v11 offset:7424
	ds_load_b32 v27, v11 offset:7552
	ds_load_b32 v28, v11 offset:7680
	ds_load_b32 v29, v11 offset:7808
	ds_load_b32 v30, v11 offset:7936
	ds_load_b32 v31, v11 offset:8064
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v9
	.Lmq6x4t8_s2t0_cdone:
	s_barrier
	.Lmq6x4t8_s2t0_tile_done:
	s_add_i32 s48, s48, 1
	s_cmp_ge_u32 s48, s42
	s_cbranch_scc1 .Lmq6x4t8_fold
	s_mul_i32 s46, s48, 0xc8
	s_lshl_b32 s47, s48, 10
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s47, v3
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6x4t8_s3t9_wdone
	v_add_nc_u32_e32 v1, s46, v13
	v_add_nc_u32_e32 v2, s46, v14
	buffer_load_b32 v7, v2, s[4:7], s45 offen offset:8
	buffer_load_u16 v8, v2, s[4:7], s45 offen offset:12
	buffer_load_b32 v6, v1, s[4:7], s45 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v16, 63, v7
	v_bfe_u32 v17, v7, 6, 6
	v_bfe_u32 v18, v7, 12, 6
	v_bfe_u32 v19, v7, 18, 6
	v_bfe_u32 v20, v7, 24, 6
	v_alignbit_b32 v21, v8, v7, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v21, 63, v21
	v_bfe_u32 v22, v8, 4, 6
	v_bfe_u32 v23, v8, 10, 6
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v6, v16, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v6, v17, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v6, v18, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v6, v19, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v6, v20, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v6, v21, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v6, v22, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v6, v23, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	.Lmq6x4t8_s3t9_wdone:
	s_add_i32 s44, s40, 0
	s_cmp_ge_u32 s44, s31
	s_cbranch_scc1 .Lmq6x4t8_s3t0_nofill
	v_add_nc_u32_e32 v12, s56, v3
	global_load_b128 v[24:27], v12, s[16:17]
	global_load_b128 v[28:31], v12, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v10, v24
	ds_store_b32 v10, v25 offset:128
	ds_store_b32 v10, v26 offset:256
	ds_store_b32 v10, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v10, v28 offset:512
	ds_store_b32 v10, v29 offset:640
	ds_store_b32 v10, v30 offset:768
	ds_store_b32 v10, v31 offset:896
	s_waitcnt lgkmcnt(0)
	.Lmq6x4t8_s3t0_nofill:
	s_barrier
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6x4t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:128
	ds_load_b32 v24, v11
	ds_load_b32 v26, v11 offset:256
	ds_load_b32 v27, v11 offset:384
	ds_load_b32 v28, v11 offset:512
	ds_load_b32 v29, v11 offset:640
	ds_load_b32 v30, v11 offset:768
	ds_load_b32 v31, v11 offset:896
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v9
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6x4t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:1152
	ds_load_b32 v24, v11 offset:1024
	ds_load_b32 v26, v11 offset:1280
	ds_load_b32 v27, v11 offset:1408
	ds_load_b32 v28, v11 offset:1536
	ds_load_b32 v29, v11 offset:1664
	ds_load_b32 v30, v11 offset:1792
	ds_load_b32 v31, v11 offset:1920
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v9
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6x4t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:2176
	ds_load_b32 v24, v11 offset:2048
	ds_load_b32 v26, v11 offset:2304
	ds_load_b32 v27, v11 offset:2432
	ds_load_b32 v28, v11 offset:2560
	ds_load_b32 v29, v11 offset:2688
	ds_load_b32 v30, v11 offset:2816
	ds_load_b32 v31, v11 offset:2944
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v9
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6x4t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:3200
	ds_load_b32 v24, v11 offset:3072
	ds_load_b32 v26, v11 offset:3328
	ds_load_b32 v27, v11 offset:3456
	ds_load_b32 v28, v11 offset:3584
	ds_load_b32 v29, v11 offset:3712
	ds_load_b32 v30, v11 offset:3840
	ds_load_b32 v31, v11 offset:3968
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v9
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6x4t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:4224
	ds_load_b32 v24, v11 offset:4096
	ds_load_b32 v26, v11 offset:4352
	ds_load_b32 v27, v11 offset:4480
	ds_load_b32 v28, v11 offset:4608
	ds_load_b32 v29, v11 offset:4736
	ds_load_b32 v30, v11 offset:4864
	ds_load_b32 v31, v11 offset:4992
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v9
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6x4t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:5248
	ds_load_b32 v24, v11 offset:5120
	ds_load_b32 v26, v11 offset:5376
	ds_load_b32 v27, v11 offset:5504
	ds_load_b32 v28, v11 offset:5632
	ds_load_b32 v29, v11 offset:5760
	ds_load_b32 v30, v11 offset:5888
	ds_load_b32 v31, v11 offset:6016
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v9
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6x4t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:6272
	ds_load_b32 v24, v11 offset:6144
	ds_load_b32 v26, v11 offset:6400
	ds_load_b32 v27, v11 offset:6528
	ds_load_b32 v28, v11 offset:6656
	ds_load_b32 v29, v11 offset:6784
	ds_load_b32 v30, v11 offset:6912
	ds_load_b32 v31, v11 offset:7040
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v9
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6x4t8_s3t0_cdone
	ds_load_b32 v25, v11 offset:7296
	ds_load_b32 v24, v11 offset:7168
	ds_load_b32 v26, v11 offset:7424
	ds_load_b32 v27, v11 offset:7552
	ds_load_b32 v28, v11 offset:7680
	ds_load_b32 v29, v11 offset:7808
	ds_load_b32 v30, v11 offset:7936
	ds_load_b32 v31, v11 offset:8064
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v9, v25, v17
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v24, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v26, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v31, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v9
	.Lmq6x4t8_s3t0_cdone:
	s_barrier
	.Lmq6x4t8_s3t0_tile_done:
	s_add_i32 s48, s48, 1
	s_cmp_lt_u32 s48, s42
	s_cbranch_scc1 .Lmq6x4t8_group
	.Lmq6x4t8_fold:
	s_cmp_eq_u32 s52, 0
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_le_u32 s31, 0
	s_cbranch_scc1 .Lmq6x4t8_fold_done
	v_add_f32_e32 v40, v40, v41
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v43
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6x4t8_fold_done
	v_add_f32_e32 v44, v44, v45
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v47
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6x4t8_fold_done
	v_add_f32_e32 v48, v48, v49
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v51
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6x4t8_fold_done
	v_add_f32_e32 v52, v52, v53
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v55
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6x4t8_fold_done
	v_add_f32_e32 v56, v56, v57
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v59
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6x4t8_fold_done
	v_add_f32_e32 v60, v60, v61
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v63
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6x4t8_fold_done
	v_add_f32_e32 v64, v64, v65
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v67
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6x4t8_fold_done
	v_add_f32_e32 v68, v68, v69
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v71
	.Lmq6x4t8_fold_done:
	s_cmp_le_u32 s31, 0
	s_cbranch_scc1 .Lmq6x4t8_reduce_done
	ds_swizzle_b32 v24, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v40, v40, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v40, v40, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v40, v40, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v40, v40, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v40, v40, v24
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6x4t8_reduce_done
	ds_swizzle_b32 v24, v44 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v44, v44, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v44, v44, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v44, v44, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v44, v44, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v44, v44, v24
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6x4t8_reduce_done
	ds_swizzle_b32 v24, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v48, v48, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v48, v48, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v48, v48, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v48, v48, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v48, v48, v24
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6x4t8_reduce_done
	ds_swizzle_b32 v24, v52 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v52, v52, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v52, v52, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v52, v52, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v52, v52, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v52, v52, v24
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6x4t8_reduce_done
	ds_swizzle_b32 v24, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v56, v56, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v56, v56, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v56, v56, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v56, v56, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v56, v56, v24
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6x4t8_reduce_done
	ds_swizzle_b32 v24, v60 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v60, v60, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v60, v60, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v60, v60, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v60, v60, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v60, v60, v24
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6x4t8_reduce_done
	ds_swizzle_b32 v24, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v64, v64, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v64, v64, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v64, v64, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v64, v64, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v64, v64, v24
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6x4t8_reduce_done
	ds_swizzle_b32 v24, v68 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v68, v68, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v68, v68, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v68, v68, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v68, v68, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v68, v68, v24
	.Lmq6x4t8_reduce_done:
	v_cmpx_eq_u32_e32 0, v5
	s_cmp_le_u32 s31, 0
	s_cbranch_scc1 .Lmq6x4t8_end
	s_mul_i32 s51, s50, 0
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 0
	s_cbranch_scc1 .Lmq6x4t8_stored0
	global_store_b32 v24, v40, s[32:33]
	.Lmq6x4t8_stored0:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6x4t8_end
	s_mul_i32 s51, s50, 1
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lmq6x4t8_stored1
	global_store_b32 v24, v44, s[32:33]
	.Lmq6x4t8_stored1:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6x4t8_end
	s_mul_i32 s51, s50, 2
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lmq6x4t8_stored2
	global_store_b32 v24, v48, s[32:33]
	.Lmq6x4t8_stored2:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6x4t8_end
	s_mul_i32 s51, s50, 3
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lmq6x4t8_stored3
	global_store_b32 v24, v52, s[32:33]
	.Lmq6x4t8_stored3:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6x4t8_end
	s_mul_i32 s51, s50, 4
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lmq6x4t8_stored4
	global_store_b32 v24, v56, s[32:33]
	.Lmq6x4t8_stored4:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6x4t8_end
	s_mul_i32 s51, s50, 5
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lmq6x4t8_stored5
	global_store_b32 v24, v60, s[32:33]
	.Lmq6x4t8_stored5:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6x4t8_end
	s_mul_i32 s51, s50, 6
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lmq6x4t8_stored6
	global_store_b32 v24, v64, s[32:33]
	.Lmq6x4t8_stored6:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6x4t8_end
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6x4t8_end
	s_mul_i32 s51, s50, 7
	s_add_i32 s51, s51, s53
	v_mov_b32_e32 v24, s51
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lmq6x4t8_stored7
	global_store_b32 v24, v68, s[32:33]
	.Lmq6x4t8_stored7:
	s_waitcnt_vscnt null, 0x0
	.Lmq6x4t8_end:
	s_endpgm
.Lgemv_mq6g256v2_rows_x4_t8_pm_gfx1100_end:
.size gemv_mq6g256v2_rows_x4_t8_pm_gfx1100, .Lgemv_mq6g256v2_rows_x4_t8_pm_gfx1100_end-gemv_mq6g256v2_rows_x4_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_mq6g256v2_rows_x4_t8_pm_gfx1100
	.amdhsa_group_segment_fixed_size 8192
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
	.amdhsa_user_sgpr_count 14
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
	.amdhsa_next_free_vgpr 72
	.amdhsa_next_free_sgpr 72
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_mq6g256v2_rows_x4_t8_pm_gfx1100_end-gemv_mq6g256v2_rows_x4_t8_pm_gfx1100)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.text
.protected gemv_q8_0_k8_rows_t8_pm_gfx1100
.globl gemv_q8_0_k8_rows_t8_pm_gfx1100
.p2align 8
.type gemv_q8_0_k8_rows_t8_pm_gfx1100,@function
gemv_q8_0_k8_rows_t8_pm_gfx1100:
	s_load_b128 s[4:7], s[0:1], null
	s_load_b64 s[8:9], s[0:1], 0x10
	s_load_b64 s[10:11], s[0:1], 0x18
	s_load_b64 s[12:13], s[0:1], 0x20
	s_load_b32 s50, s[0:1], 0x28
	v_and_b32_e32 v1, 31, v0
	v_lshrrev_b32_e32 v13, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s49, v13
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 24, v1
	v_cndmask_b32_e64 v2, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v2, v2, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 28, v1
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 30, v1
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 31, v1
	v_cndmask_b32_e64 v5, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, v1, 2
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_mov_b32 s40, vcc_lo
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_end
	s_cmp_gt_u32 s12, 64
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_end
	s_and_b32 s39, s11, 0xff
	s_cmp_lg_u32 s39, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_end
	s_cmp_eq_u32 s11, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_end
	s_cmp_le_i32 s11, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_end
	s_lshl_b32 s16, s14, 1
	s_cmp_ge_i32 s16, s10
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_end
	s_lshl_b32 s15, s15, 3
	s_cmp_ge_u32 s15, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_end
	s_sub_u32 s12, s12, s15
	s_min_u32 s12, s12, 8
	s_mul_i32 s38, s15, s11
	s_lshl_b32 s38, s38, 2
	s_add_u32 s6, s6, s38
	s_addc_u32 s7, s7, 0
	s_mul_i32 s38, s15, s10
	s_lshl_b32 s38, s38, 2
	s_add_u32 s8, s8, s38
	s_addc_u32 s9, s9, 0
	s_lshr_b32 s45, s49, 3
	s_and_b32 s47, s49, 7
	s_add_i32 s17, s16, s45
	s_cmp_lt_i32 s17, s10
	s_cselect_b32 s18, 1, 0
	s_lshr_b32 s20, s11, 8
	s_mul_i32 s19, s20, 34
	s_lshl_b32 s48, s17, 3
	s_add_i32 s48, s48, s47
	s_mul_i32 s48, s48, s19
	v_mov_b32_e32 v10, s48
	v_add_nc_u32_e32 v11, s48, v1
	s_lshl_b32 s43, s11, 2
	s_mul_i32 s44, s45, s43
	s_lshr_b32 s39, s11, 1
	s_mul_i32 s39, s39, s47
	s_add_i32 s44, s44, s39
	s_lshr_b32 s21, s20, 3
	s_and_b32 s22, s20, 7
	v_lshlrev_b32_e32 v8, 2, v1
	s_lshl_b32 s39, s49, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v6, s39, v8
	s_lshl_b32 s39, s47, 10
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, s39, v8
	s_mul_i32 s39, s45, 0x100
	s_lshl_b32 s38, s47, 2
	s_add_i32 s39, s39, s38
	v_mov_b32_e32 v14, s39
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	s_mov_b32 s23, s21
	s_cmp_eq_u32 s23, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tails
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kloop:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	global_load_i8 v30, v11, s[4:5] offset:206
	global_load_u16 v22, v10, s[4:5] offset:204
	global_load_i8 v31, v11, s[4:5] offset:240
	global_load_u16 v23, v10, s[4:5] offset:238
	s_waitcnt vmcnt(15)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(14)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v30, v30
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v22, v30, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v31, v31
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v23, v31, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	global_load_b32 v39, v9, s[6:7] offset:896
	s_waitcnt vmcnt(7)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v38 offset:768
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v39 offset:896
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	ds_load_b32 v39, v7 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v39, v23
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	ds_load_b32 v38, v7 offset:8960
	ds_load_b32 v39, v7 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v39, v23
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t0_done:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t1_done
	s_add_i32 s38, s45, 2
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t1_nofill
	s_mul_i32 s38, s43, 2
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	global_load_b32 v39, v9, s[6:7] offset:896
	s_waitcnt vmcnt(7)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v38 offset:768
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v39 offset:896
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t1_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t1_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	ds_load_b32 v39, v7 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v39, v23
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t1_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	ds_load_b32 v38, v7 offset:8960
	ds_load_b32 v39, v7 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v39, v23
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t1_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t1_done:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t2_done
	s_add_i32 s38, s45, 4
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t2_nofill
	s_mul_i32 s38, s43, 4
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	global_load_b32 v39, v9, s[6:7] offset:896
	s_waitcnt vmcnt(7)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v38 offset:768
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v39 offset:896
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t2_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t2_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	ds_load_b32 v39, v7 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v39, v23
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t2_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	ds_load_b32 v38, v7 offset:8960
	ds_load_b32 v39, v7 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v39, v23
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t2_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t2_done:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t3_done
	s_add_i32 s38, s45, 6
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t3_nofill
	s_mul_i32 s38, s43, 6
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	global_load_b32 v39, v9, s[6:7] offset:896
	s_waitcnt vmcnt(7)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v38 offset:768
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v39 offset:896
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t3_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t3_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	ds_load_b32 v39, v7 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v39, v23
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t3_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	ds_load_b32 v38, v7 offset:8960
	ds_load_b32 v39, v7 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v39, v23
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t3_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_full8_t3_done:
	v_add_nc_u32_e32 v8, 0x400, v8
	v_add_nc_u32_e32 v10, 0x110, v10
	v_add_nc_u32_e32 v11, 0x110, v11
	s_add_i32 s23, s23, -1
	s_cmp_lg_u32 s23, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kloop
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tails:
	s_cmp_eq_u32 s22, 1
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail1
	s_cmp_eq_u32 s22, 2
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail2
	s_cmp_eq_u32 s22, 3
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail3
	s_cmp_eq_u32 s22, 4
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail4
	s_cmp_eq_u32 s22, 5
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail5
	s_cmp_eq_u32 s22, 6
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail6
	s_cmp_eq_u32 s22, 7
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail7
	s_branch .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail1:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v32
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v40, v32, v16
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:8192
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v41, v32, v16
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t0_done:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t1_done
	s_add_i32 s38, s45, 2
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t1_nofill
	s_mul_i32 s38, s43, 2
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v32
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t1_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t1_cdone
	ds_load_b32 v32, v7
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v42, v32, v16
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t1_cdone
	ds_load_b32 v32, v7 offset:8192
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v43, v32, v16
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t1_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t1_done:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t2_done
	s_add_i32 s38, s45, 4
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t2_nofill
	s_mul_i32 s38, s43, 4
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v32
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t2_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t2_cdone
	ds_load_b32 v32, v7
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v44, v32, v16
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t2_cdone
	ds_load_b32 v32, v7 offset:8192
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v45, v32, v16
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t2_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t2_done:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t3_done
	s_add_i32 s38, s45, 6
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t3_nofill
	s_mul_i32 s38, s43, 6
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v32
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t3_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t3_cdone
	ds_load_b32 v32, v7
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v46, v32, v16
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t3_cdone
	ds_load_b32 v32, v7 offset:8192
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v47, v32, v16
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t3_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part1_t3_done:
	s_branch .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail2:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v33 offset:128
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t0_done:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t1_done
	s_add_i32 s38, s45, 2
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t1_nofill
	s_mul_i32 s38, s43, 2
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v33 offset:128
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t1_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t1_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t1_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t1_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t1_done:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t2_done
	s_add_i32 s38, s45, 4
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t2_nofill
	s_mul_i32 s38, s43, 4
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v33 offset:128
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t2_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t2_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t2_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t2_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t2_done:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t3_done
	s_add_i32 s38, s45, 6
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t3_nofill
	s_mul_i32 s38, s43, 6
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v33 offset:128
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t3_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t3_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t3_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t3_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part2_t3_done:
	s_branch .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail3:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v34 offset:256
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t0_done:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t1_done
	s_add_i32 s38, s45, 2
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t1_nofill
	s_mul_i32 s38, s43, 2
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v34 offset:256
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t1_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t1_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t1_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t1_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t1_done:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t2_done
	s_add_i32 s38, s45, 4
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t2_nofill
	s_mul_i32 s38, s43, 4
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v34 offset:256
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t2_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t2_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t2_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t2_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t2_done:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t3_done
	s_add_i32 s38, s45, 6
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t3_nofill
	s_mul_i32 s38, s43, 6
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v34 offset:256
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t3_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t3_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t3_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t3_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part3_t3_done:
	s_branch .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail4:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v35 offset:384
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t0_done:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t1_done
	s_add_i32 s38, s45, 2
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t1_nofill
	s_mul_i32 s38, s43, 2
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v35 offset:384
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t1_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t1_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t1_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t1_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t1_done:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t2_done
	s_add_i32 s38, s45, 4
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t2_nofill
	s_mul_i32 s38, s43, 4
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v35 offset:384
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t2_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t2_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t2_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t2_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t2_done:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t3_done
	s_add_i32 s38, s45, 6
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t3_nofill
	s_mul_i32 s38, s43, 6
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v35 offset:384
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t3_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t3_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t3_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t3_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part4_t3_done:
	s_branch .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail5:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v36 offset:512
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t0_done:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t1_done
	s_add_i32 s38, s45, 2
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t1_nofill
	s_mul_i32 s38, s43, 2
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v36 offset:512
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t1_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t1_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t1_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t1_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t1_done:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t2_done
	s_add_i32 s38, s45, 4
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t2_nofill
	s_mul_i32 s38, s43, 4
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v36 offset:512
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t2_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t2_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t2_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t2_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t2_done:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t3_done
	s_add_i32 s38, s45, 6
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t3_nofill
	s_mul_i32 s38, s43, 6
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v36 offset:512
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t3_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t3_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t3_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t3_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part5_t3_done:
	s_branch .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail6:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v37 offset:640
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v37, v21
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v37, v21
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t0_done:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t1_done
	s_add_i32 s38, s45, 2
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t1_nofill
	s_mul_i32 s38, s43, 2
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v37 offset:640
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t1_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t1_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v37, v21
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t1_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v37, v21
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t1_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t1_done:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t2_done
	s_add_i32 s38, s45, 4
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t2_nofill
	s_mul_i32 s38, s43, 4
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v37 offset:640
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t2_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t2_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v37, v21
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t2_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v37, v21
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t2_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t2_done:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t3_done
	s_add_i32 s38, s45, 6
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t3_nofill
	s_mul_i32 s38, s43, 6
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v37 offset:640
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t3_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t3_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v37, v21
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t3_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v37, v21
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t3_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part6_t3_done:
	s_branch .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_tail7:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	global_load_i8 v30, v11, s[4:5] offset:206
	global_load_u16 v22, v10, s[4:5] offset:204
	s_waitcnt vmcnt(13)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v30, v30
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v22, v30, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v38 offset:768
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v38, v22
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	ds_load_b32 v38, v7 offset:8960
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v38, v22
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t0_done:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t1_done
	s_add_i32 s38, s45, 2
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t1_nofill
	s_mul_i32 s38, s43, 2
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v38 offset:768
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t1_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t1_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v38, v22
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t1_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	ds_load_b32 v38, v7 offset:8960
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v38, v22
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t1_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t1_done:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t2_done
	s_add_i32 s38, s45, 4
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t2_nofill
	s_mul_i32 s38, s43, 4
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v38 offset:768
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t2_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t2_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v38, v22
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t2_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	ds_load_b32 v38, v7 offset:8960
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v38, v22
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t2_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t2_done:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t3_done
	s_add_i32 s38, s45, 6
	s_cmp_ge_u32 s38, s12
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t3_nofill
	s_mul_i32 s38, s43, 6
	s_add_i32 s38, s38, s44
	v_add_nc_u32_e32 v9, s38, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v38 offset:768
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t3_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t3_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v38, v22
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t3_cdone
	ds_load_b32 v32, v7 offset:8192
	ds_load_b32 v33, v7 offset:8320
	ds_load_b32 v34, v7 offset:8448
	ds_load_b32 v35, v7 offset:8576
	ds_load_b32 v36, v7 offset:8704
	ds_load_b32 v37, v7 offset:8832
	ds_load_b32 v38, v7 offset:8960
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v38, v22
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t3_cdone:
	s_barrier
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_part7_t3_done:
	s_branch .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_kdone:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_parked
	s_cmp_le_u32 s12, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip0
	ds_swizzle_b32 v32, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v2, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v3, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v5, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	s_mov_b32 exec_lo, s40
	ds_store_b32 v14, v40
	s_mov_b32 exec_lo, -1
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip0:
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip1
	ds_swizzle_b32 v32, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v2, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v3, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v4, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v5, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	s_mov_b32 exec_lo, s40
	ds_store_b32 v14, v41 offset:32
	s_mov_b32 exec_lo, -1
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip1:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip2
	ds_swizzle_b32 v32, v42 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v2, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v3, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v4, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v5, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	s_mov_b32 exec_lo, s40
	ds_store_b32 v14, v42 offset:64
	s_mov_b32 exec_lo, -1
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip2:
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip3
	ds_swizzle_b32 v32, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v2, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v3, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v5, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	s_mov_b32 exec_lo, s40
	ds_store_b32 v14, v43 offset:96
	s_mov_b32 exec_lo, -1
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip3:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip4
	ds_swizzle_b32 v32, v44 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v2, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v3, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v5, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	s_mov_b32 exec_lo, s40
	ds_store_b32 v14, v44 offset:128
	s_mov_b32 exec_lo, -1
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip4:
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip5
	ds_swizzle_b32 v32, v45 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v2, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v3, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v4, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v5, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	s_mov_b32 exec_lo, s40
	ds_store_b32 v14, v45 offset:160
	s_mov_b32 exec_lo, -1
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip5:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip6
	ds_swizzle_b32 v32, v46 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v2, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v3, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v4, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v5, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	s_mov_b32 exec_lo, s40
	ds_store_b32 v14, v46 offset:192
	s_mov_b32 exec_lo, -1
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip6:
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip7
	ds_swizzle_b32 v32, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v2, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v3, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v5, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	s_mov_b32 exec_lo, s40
	ds_store_b32 v14, v47 offset:224
	s_mov_b32 exec_lo, -1
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_partial_skip7:
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_parked:
	v_and_b32_e32 v12, 7, v0
	v_lshrrev_b32_e32 v13, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v13, s16, v13
	v_lshlrev_b32_e32 v9, 5, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v15, v12, s10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v15, v13, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v15, 2, v15
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_mov_b32 s46, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s12, v12
	s_and_b32 s46, s46, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s10, v13
	s_and_b32 exec_lo, s46, vcc_lo
	s_cbranch_execz .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_end
	ds_load_b128 v[16:19], v9
	ds_load_b128 v[20:23], v9 offset:16
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v20, v20, v21
	v_add_f32_e32 v22, v22, v23
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v20, v20, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v20
	s_cmp_eq_u32 s50, 0
	s_cbranch_scc1 .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_store
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v24, 0x7f800000, v16
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v25, v16, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v25, v16, v25, 0x7fff
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v26, 0xffff, v16
	v_or_b32_e32 v27, 0x10000, v16
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v26, v27, v16, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v16, 0xffff0000, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, s13, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v24, 0x7f800000, v16
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v25, v16, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v25, v16, v25, 0x7fff
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v26, 0xffff, v16
	v_or_b32_e32 v27, 0x10000, v16
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v26, v27, v16, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v16, 0xffff0000, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, 0xbfb8aa3b, v16, -v24
	s_delay_alu instid0(VALU_DEP_2)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_f32_e32 v24, v24, v26
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v16
	s_delay_alu instid0(VALU_DEP_4)
	v_fmamk_f32 v25, v16, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v25, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v16
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, v16, v24, v16
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v25, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v16, v25, v24, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v24, 0x7f800000, v16
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v25, v16, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v25, v16, v25, 0x7fff
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v26, 0xffff, v16
	v_or_b32_e32 v27, 0x10000, v16
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v26, v27, v16, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v16, 0xffff0000, v25
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_store:
	global_store_b32 v15, v16, s[8:9]
	s_waitcnt_vscnt null, 0x0
	.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_body_end:
	s_endpgm
.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_end:
.size gemv_q8_0_k8_rows_t8_pm_gfx1100, .Lgemv_q8_0_k8_rows_t8_pm_gfx1100_end-gemv_q8_0_k8_rows_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_q8_0_k8_rows_t8_pm_gfx1100
	.amdhsa_group_segment_fixed_size 16384
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 44
	.amdhsa_user_sgpr_count 14
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
	.amdhsa_next_free_vgpr 48
	.amdhsa_next_free_sgpr 51
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_q8_0_k8_rows_t8_pm_gfx1100_end-gemv_q8_0_k8_rows_t8_pm_gfx1100)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.text
.protected gemv_q8_0_k320_rows_t8_pm_gfx1100
.globl gemv_q8_0_k320_rows_t8_pm_gfx1100
.p2align 8
.type gemv_q8_0_k320_rows_t8_pm_gfx1100,@function
gemv_q8_0_k320_rows_t8_pm_gfx1100:
	s_load_b128 s[4:7], s[0:1], null
	s_load_b64 s[8:9], s[0:1], 0x10
	s_load_b64 s[10:11], s[0:1], 0x18
	s_load_b32 s12, s[0:1], 0x20
	v_and_b32_e32 v1, 31, v0
	v_lshrrev_b32_e32 v13, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s49, v13
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 24, v1
	v_cndmask_b32_e64 v2, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v2, v2, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 28, v1
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 30, v1
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 31, v1
	v_cndmask_b32_e64 v5, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, v1, 2
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_mov_b32 s40, vcc_lo
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_end
	s_cmp_gt_u32 s12, 64
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_end
	s_and_b32 s39, s11, 31
	s_cmp_lg_u32 s39, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_end
	s_cmp_lg_u32 s11, 0x140
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_end
	s_lshl_b32 s16, s14, 3
	s_cmp_ge_i32 s16, s10
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_end
	s_lshl_b32 s15, s15, 3
	s_cmp_ge_u32 s15, s12
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_end
	s_sub_u32 s12, s12, s15
	s_min_u32 s12, s12, 8
	s_mul_i32 s38, s15, s11
	s_lshl_b32 s38, s38, 2
	s_add_u32 s6, s6, s38
	s_addc_u32 s7, s7, 0
	s_mul_i32 s38, s15, s10
	s_lshl_b32 s38, s38, 2
	s_add_u32 s8, s8, s38
	s_addc_u32 s9, s9, 0
	s_mov_b32 s45, s49
	s_add_i32 s17, s16, s45
	s_cmp_lt_i32 s17, s10
	s_cselect_b32 s18, 1, 0
	s_lshr_b32 s20, s11, 5
	s_mul_i32 s19, s20, 34
	s_mul_i32 s48, s17, s19
	v_mov_b32_e32 v10, s48
	v_add_nc_u32_e32 v11, s48, v1
	s_lshl_b32 s43, s11, 2
	s_mul_i32 s44, s45, s43
	s_lshr_b32 s21, s20, 3
	s_and_b32 s22, s20, 7
	v_lshlrev_b32_e32 v8, 2, v1
	s_lshl_b32 s39, s49, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v6, s39, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v7, v8
	s_lshl_b32 s41, s10, 2
	s_lshl_b32 s42, s17, 2
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	s_mov_b32 s23, s21
	s_cmp_eq_u32 s23, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tails
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kloop:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	global_load_i8 v30, v11, s[4:5] offset:206
	global_load_u16 v22, v10, s[4:5] offset:204
	global_load_i8 v31, v11, s[4:5] offset:240
	global_load_u16 v23, v10, s[4:5] offset:238
	s_waitcnt vmcnt(15)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(14)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v30, v30
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v22, v30, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v31, v31
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v23, v31, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	global_load_b32 v39, v9, s[6:7] offset:896
	s_waitcnt vmcnt(7)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v38 offset:768
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v39 offset:896
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	ds_load_b32 v39, v7 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v23, v39
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	ds_load_b32 v38, v7 offset:1792
	ds_load_b32 v39, v7 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v23, v39
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	ds_load_b32 v38, v7 offset:2816
	ds_load_b32 v39, v7 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v23, v39
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	ds_load_b32 v38, v7 offset:3840
	ds_load_b32 v39, v7 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v23, v39
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	ds_load_b32 v38, v7 offset:4864
	ds_load_b32 v39, v7 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v23, v39
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	ds_load_b32 v38, v7 offset:5888
	ds_load_b32 v39, v7 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v23, v39
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	ds_load_b32 v38, v7 offset:6912
	ds_load_b32 v39, v7 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v23, v39
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	ds_load_b32 v38, v7 offset:7936
	ds_load_b32 v39, v7 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v23, v39
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_full8_t0_done:
	v_add_nc_u32_e32 v8, 0x400, v8
	v_add_nc_u32_e32 v10, 0x110, v10
	v_add_nc_u32_e32 v11, 0x110, v11
	s_add_i32 s23, s23, -1
	s_cmp_lg_u32 s23, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kloop
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tails:
	s_cmp_eq_u32 s22, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail1
	s_cmp_eq_u32 s22, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail2
	s_cmp_eq_u32 s22, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail3
	s_cmp_eq_u32 s22, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail4
	s_cmp_eq_u32 s22, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail5
	s_cmp_eq_u32 s22, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail6
	s_cmp_eq_u32 s22, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail7
	s_branch .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail1:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v32
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part1_t0_done:
	s_branch .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail2:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v33 offset:128
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v12
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part2_t0_done:
	s_branch .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail3:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v34 offset:256
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v12
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part3_t0_done:
	s_branch .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail4:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v35 offset:384
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v40, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v41, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v42, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v43, v19, v35
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v44, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v45, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v46, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v47, v19, v35
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v48, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v49, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v50, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v51, v19, v35
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v52, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v53, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v54, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v55, v19, v35
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v56, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v57, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v58, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v59, v19, v35
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v60, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v61, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v62, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v63, v19, v35
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v64, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v65, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v66, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v67, v19, v35
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v68, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v69, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v70, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v71, v19, v35
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part4_t0_done:
	s_branch .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail5:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v36 offset:512
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v40, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v41, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v42, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v43, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v44, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v45, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v46, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v47, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v48, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v49, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v50, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v51, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v52, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v53, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v54, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v55, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v56, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v57, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v58, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v59, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v60, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v61, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v62, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v63, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v64, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v65, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v66, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v67, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v68, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v69, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v70, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v71, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part5_t0_done:
	s_branch .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail6:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v37 offset:640
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v40, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v41, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v42, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v43, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v44, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v45, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v46, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v47, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v48, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v49, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v50, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v51, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v52, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v53, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v54, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v55, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v56, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v57, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v58, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v59, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v60, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v61, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v62, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v63, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v64, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v65, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v66, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v67, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v68, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v69, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v70, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v71, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v12
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part6_t0_done:
	s_branch .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_tail7:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	global_load_i8 v30, v11, s[4:5] offset:206
	global_load_u16 v22, v10, s[4:5] offset:204
	s_waitcnt vmcnt(13)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v30, v30
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v22, v30, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v38 offset:768
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v40, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v41, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v42, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v43, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	ds_load_b32 v38, v7 offset:1792
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v44, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v45, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v46, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v47, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	ds_load_b32 v38, v7 offset:2816
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v48, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v49, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v50, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v51, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	ds_load_b32 v38, v7 offset:3840
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v52, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v53, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v54, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v55, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	ds_load_b32 v38, v7 offset:4864
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v56, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v57, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v58, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v59, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	ds_load_b32 v38, v7 offset:5888
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v60, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v61, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v62, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v63, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	ds_load_b32 v38, v7 offset:6912
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v64, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v65, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v66, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v67, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	ds_load_b32 v38, v7 offset:7936
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v68, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v69, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v70, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v71, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v12
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_part7_t0_done:
	s_branch .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_kdone:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_end
	s_cmp_le_u32 s12, 0
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip0
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v43, v41
	ds_swizzle_b32 v32, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v2, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v3, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v5, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip0:
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip1
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v47, v45
	ds_swizzle_b32 v32, v44 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v2, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v3, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v5, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip1:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip2
	v_add_f32_e32 v49, v49, v48
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v51, v49
	ds_swizzle_b32 v32, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v32
	ds_bpermute_b32 v32, v2, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v32
	ds_bpermute_b32 v32, v3, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v32
	ds_bpermute_b32 v32, v4, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v32
	ds_bpermute_b32 v32, v5, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v32
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip2:
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip3
	v_add_f32_e32 v53, v53, v52
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v55, v53
	ds_swizzle_b32 v32, v52 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v32
	ds_bpermute_b32 v32, v2, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v32
	ds_bpermute_b32 v32, v3, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v32
	ds_bpermute_b32 v32, v4, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v32
	ds_bpermute_b32 v32, v5, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v32
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip3:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip4
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v59, v57
	ds_swizzle_b32 v32, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v32
	ds_bpermute_b32 v32, v2, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v32
	ds_bpermute_b32 v32, v3, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v32
	ds_bpermute_b32 v32, v4, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v32
	ds_bpermute_b32 v32, v5, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v32
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip4:
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip5
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v63, v61
	ds_swizzle_b32 v32, v60 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v32
	ds_bpermute_b32 v32, v2, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v32
	ds_bpermute_b32 v32, v3, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v32
	ds_bpermute_b32 v32, v4, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v32
	ds_bpermute_b32 v32, v5, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v32
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip5:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip6
	v_add_f32_e32 v65, v65, v64
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v67, v65
	ds_swizzle_b32 v32, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v32
	ds_bpermute_b32 v32, v2, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v32
	ds_bpermute_b32 v32, v3, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v32
	ds_bpermute_b32 v32, v4, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v32
	ds_bpermute_b32 v32, v5, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v32
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip6:
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip7
	v_add_f32_e32 v69, v69, v68
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v71, v69
	ds_swizzle_b32 v32, v68 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v32
	ds_bpermute_b32 v32, v2, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v32
	ds_bpermute_b32 v32, v3, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v32
	ds_bpermute_b32 v32, v4, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v32
	ds_bpermute_b32 v32, v5, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v32
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_reduce_skip7:
	v_mov_b32_e32 v24, s42
	s_mul_i32 s38, s41, 1
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v25, s38
	s_mul_i32 s38, s41, 2
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v26, s38
	s_mul_i32 s38, s41, 3
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v27, s38
	s_mul_i32 s38, s41, 4
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v28, s38
	s_mul_i32 s38, s41, 5
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v29, s38
	s_mul_i32 s38, s41, 6
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v30, s38
	s_mul_i32 s38, s41, 7
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v31, s38
	s_mov_b32 exec_lo, s40
	global_store_b32 v24, v40, s[8:9]
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v25, v44, s[8:9]
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v26, v48, s[8:9]
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v27, v52, s[8:9]
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v28, v56, s[8:9]
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v29, v60, s[8:9]
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v30, v64, s[8:9]
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v31, v68, s[8:9]
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_stored0:
	s_mov_b32 exec_lo, -1
	s_waitcnt_vscnt null, 0x0
	.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_body_end:
	s_endpgm
.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_end:
.size gemv_q8_0_k320_rows_t8_pm_gfx1100, .Lgemv_q8_0_k320_rows_t8_pm_gfx1100_end-gemv_q8_0_k320_rows_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_q8_0_k320_rows_t8_pm_gfx1100
	.amdhsa_group_segment_fixed_size 8192
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 36
	.amdhsa_user_sgpr_count 14
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
	.amdhsa_next_free_vgpr 72
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
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_q8_0_k320_rows_t8_pm_gfx1100_end-gemv_q8_0_k320_rows_t8_pm_gfx1100)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.text
.protected gemv_q8_0_k2560_rows_t8_pm_gfx1100
.globl gemv_q8_0_k2560_rows_t8_pm_gfx1100
.p2align 8
.type gemv_q8_0_k2560_rows_t8_pm_gfx1100,@function
gemv_q8_0_k2560_rows_t8_pm_gfx1100:
	s_load_b128 s[4:7], s[0:1], null
	s_load_b64 s[8:9], s[0:1], 0x10
	s_load_b64 s[10:11], s[0:1], 0x18
	s_load_b32 s12, s[0:1], 0x20
	v_and_b32_e32 v1, 31, v0
	v_lshrrev_b32_e32 v13, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s49, v13
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 24, v1
	v_cndmask_b32_e64 v2, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v2, v2, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 28, v1
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 30, v1
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 31, v1
	v_cndmask_b32_e64 v5, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, v1, 2
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_mov_b32 s40, vcc_lo
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_end
	s_cmp_gt_u32 s12, 64
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_end
	s_and_b32 s39, s11, 31
	s_cmp_lg_u32 s39, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_end
	s_cmp_le_i32 s11, 0x600
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_end
	s_lshl_b32 s16, s14, 3
	s_cmp_ge_i32 s16, s10
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_end
	s_lshl_b32 s15, s15, 3
	s_cmp_ge_u32 s15, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_end
	s_sub_u32 s12, s12, s15
	s_min_u32 s12, s12, 8
	s_mul_i32 s38, s15, s11
	s_lshl_b32 s38, s38, 2
	s_add_u32 s6, s6, s38
	s_addc_u32 s7, s7, 0
	s_mul_i32 s38, s15, s10
	s_lshl_b32 s38, s38, 2
	s_add_u32 s8, s8, s38
	s_addc_u32 s9, s9, 0
	s_mov_b32 s45, s49
	s_add_i32 s17, s16, s45
	s_cmp_lt_i32 s17, s10
	s_cselect_b32 s18, 1, 0
	s_lshr_b32 s20, s11, 5
	s_mul_i32 s19, s20, 34
	s_mul_i32 s48, s17, s19
	v_mov_b32_e32 v10, s48
	v_add_nc_u32_e32 v11, s48, v1
	s_lshl_b32 s43, s11, 2
	s_mul_i32 s44, s45, s43
	s_lshr_b32 s21, s20, 3
	s_and_b32 s22, s20, 7
	v_lshlrev_b32_e32 v8, 2, v1
	s_lshl_b32 s39, s49, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v6, s39, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v7, v8
	s_lshl_b32 s41, s10, 2
	s_lshl_b32 s42, s17, 2
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	s_mov_b32 s23, s21
	s_cmp_eq_u32 s23, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tails
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kloop:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	global_load_i8 v30, v11, s[4:5] offset:206
	global_load_u16 v22, v10, s[4:5] offset:204
	global_load_i8 v31, v11, s[4:5] offset:240
	global_load_u16 v23, v10, s[4:5] offset:238
	s_waitcnt vmcnt(15)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(14)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v30, v30
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v22, v30, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v31, v31
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v23, v31, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	global_load_b32 v39, v9, s[6:7] offset:896
	s_waitcnt vmcnt(7)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v38 offset:768
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v39 offset:896
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	ds_load_b32 v39, v7 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v39, v23
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	ds_load_b32 v38, v7 offset:1792
	ds_load_b32 v39, v7 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v39, v23
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	ds_load_b32 v38, v7 offset:2816
	ds_load_b32 v39, v7 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v39, v23
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	ds_load_b32 v38, v7 offset:3840
	ds_load_b32 v39, v7 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v39, v23
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	ds_load_b32 v38, v7 offset:4864
	ds_load_b32 v39, v7 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v39, v23
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	ds_load_b32 v38, v7 offset:5888
	ds_load_b32 v39, v7 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v39, v23
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	ds_load_b32 v38, v7 offset:6912
	ds_load_b32 v39, v7 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v39, v23
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	ds_load_b32 v38, v7 offset:7936
	ds_load_b32 v39, v7 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v39, v23
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_full8_t0_done:
	v_add_nc_u32_e32 v8, 0x400, v8
	v_add_nc_u32_e32 v10, 0x110, v10
	v_add_nc_u32_e32 v11, 0x110, v11
	s_add_i32 s23, s23, -1
	s_cmp_lg_u32 s23, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kloop
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tails:
	s_cmp_eq_u32 s22, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail1
	s_cmp_eq_u32 s22, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail2
	s_cmp_eq_u32 s22, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail3
	s_cmp_eq_u32 s22, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail4
	s_cmp_eq_u32 s22, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail5
	s_cmp_eq_u32 s22, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail6
	s_cmp_eq_u32 s22, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail7
	s_branch .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail1:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v32
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v40, v32, v16
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v41, v32, v16
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v42, v32, v16
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v43, v32, v16
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v44, v32, v16
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v45, v32, v16
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v46, v32, v16
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v47, v32, v16
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part1_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail2:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v33 offset:128
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part2_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail3:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v34 offset:256
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part3_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail4:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v35 offset:384
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part4_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail5:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v36 offset:512
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part5_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail6:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v37 offset:640
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v37, v21
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v37, v21
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v37, v21
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v37, v21
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v37, v21
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v37, v21
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v37, v21
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v37, v21
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part6_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_tail7:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	global_load_i8 v30, v11, s[4:5] offset:206
	global_load_u16 v22, v10, s[4:5] offset:204
	s_waitcnt vmcnt(13)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v30, v30
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v22, v30, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v38 offset:768
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v38, v22
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	ds_load_b32 v38, v7 offset:1792
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v38, v22
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	ds_load_b32 v38, v7 offset:2816
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v38, v22
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	ds_load_b32 v38, v7 offset:3840
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v38, v22
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	ds_load_b32 v38, v7 offset:4864
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v38, v22
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	ds_load_b32 v38, v7 offset:5888
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v38, v22
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	ds_load_b32 v38, v7 offset:6912
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v38, v22
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	ds_load_b32 v38, v7 offset:7936
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v38, v22
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_part7_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_kdone:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_end
	s_cmp_le_u32 s12, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip0
	ds_swizzle_b32 v32, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v2, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v3, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v5, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip0:
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip1
	ds_swizzle_b32 v32, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v2, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v3, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v4, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v5, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip1:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip2
	ds_swizzle_b32 v32, v42 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v2, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v3, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v4, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v5, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip2:
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip3
	ds_swizzle_b32 v32, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v2, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v3, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v5, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip3:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip4
	ds_swizzle_b32 v32, v44 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v2, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v3, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v5, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip4:
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip5
	ds_swizzle_b32 v32, v45 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v2, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v3, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v4, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v5, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip5:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip6
	ds_swizzle_b32 v32, v46 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v2, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v3, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v4, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v5, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip6:
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip7
	ds_swizzle_b32 v32, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v2, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v3, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v5, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_reduce_skip7:
	v_mov_b32_e32 v24, s42
	s_mul_i32 s38, s41, 1
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v25, s38
	s_mul_i32 s38, s41, 2
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v26, s38
	s_mul_i32 s38, s41, 3
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v27, s38
	s_mul_i32 s38, s41, 4
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v28, s38
	s_mul_i32 s38, s41, 5
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v29, s38
	s_mul_i32 s38, s41, 6
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v30, s38
	s_mul_i32 s38, s41, 7
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v31, s38
	s_mov_b32 exec_lo, s40
	global_store_b32 v24, v40, s[8:9]
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v25, v41, s[8:9]
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v26, v42, s[8:9]
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v27, v43, s[8:9]
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v28, v44, s[8:9]
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v29, v45, s[8:9]
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v30, v46, s[8:9]
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v31, v47, s[8:9]
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_stored0:
	s_mov_b32 exec_lo, -1
	s_waitcnt_vscnt null, 0x0
	.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_body_end:
	s_endpgm
.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_end:
.size gemv_q8_0_k2560_rows_t8_pm_gfx1100, .Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_end-gemv_q8_0_k2560_rows_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_q8_0_k2560_rows_t8_pm_gfx1100
	.amdhsa_group_segment_fixed_size 8192
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 36
	.amdhsa_user_sgpr_count 14
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
	.amdhsa_next_free_vgpr 48
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
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_q8_0_k2560_rows_t8_pm_gfx1100_end-gemv_q8_0_k2560_rows_t8_pm_gfx1100)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.text
.protected gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100
.globl gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100
.p2align 8
.type gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100,@function
gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100:
	s_load_b128 s[24:27], s[0:1], null
	s_load_b128 s[28:31], s[0:1], 0x10
	s_load_b64 s[32:33], s[0:1], 0x20
	s_load_b64 s[34:35], s[0:1], 0x28
	s_load_b64 s[36:37], s[0:1], 0x30
	v_and_b32_e32 v1, 31, v0
	v_lshrrev_b32_e32 v13, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s49, v13
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 24, v1
	v_cndmask_b32_e64 v2, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v2, v2, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 28, v1
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 30, v1
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 31, v1
	v_cndmask_b32_e64 v5, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, v1, 2
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_mov_b32 s40, vcc_lo
	s_waitcnt lgkmcnt(0)
	s_add_i32 s39, s34, 7
	s_lshr_b32 s39, s39, 3
	s_cmp_ge_u32 s14, s39
	s_cselect_b64 s[4:5], s[26:27], s[24:25]
	s_cselect_b64 s[8:9], s[32:33], s[30:31]
	s_cselect_b32 s10, s35, s34
	s_cselect_b32 s39, s39, 0
	s_sub_i32 s38, s14, s39
	s_mov_b64 s[6:7], s[28:29]
	s_mov_b32 s11, s36
	s_mov_b32 s12, s37
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_end
	s_cmp_gt_u32 s12, 64
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_end
	s_and_b32 s39, s11, 31
	s_cmp_lg_u32 s39, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_end
	s_cmp_lg_u32 s11, 0xa00
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_end
	s_lshl_b32 s16, s38, 3
	s_cmp_ge_i32 s16, s10
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_end
	s_lshl_b32 s15, s15, 3
	s_cmp_ge_u32 s15, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_end
	s_sub_u32 s12, s12, s15
	s_min_u32 s12, s12, 8
	s_mul_i32 s38, s15, s11
	s_lshl_b32 s38, s38, 2
	s_add_u32 s6, s6, s38
	s_addc_u32 s7, s7, 0
	s_mul_i32 s38, s15, s10
	s_lshl_b32 s38, s38, 2
	s_add_u32 s8, s8, s38
	s_addc_u32 s9, s9, 0
	s_mov_b32 s45, s49
	s_add_i32 s17, s16, s45
	s_cmp_lt_i32 s17, s10
	s_cselect_b32 s18, 1, 0
	s_lshr_b32 s20, s11, 5
	s_mul_i32 s19, s20, 34
	s_mul_i32 s48, s17, s19
	v_mov_b32_e32 v10, s48
	v_add_nc_u32_e32 v11, s48, v1
	s_lshl_b32 s43, s11, 2
	s_mul_i32 s44, s45, s43
	s_lshr_b32 s21, s20, 3
	s_and_b32 s22, s20, 7
	v_lshlrev_b32_e32 v8, 2, v1
	s_lshl_b32 s39, s49, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v6, s39, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v7, v8
	s_lshl_b32 s41, s10, 2
	s_lshl_b32 s42, s17, 2
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	s_mov_b32 s23, s21
	s_cmp_eq_u32 s23, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tails
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kloop:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	global_load_i8 v30, v11, s[4:5] offset:206
	global_load_u16 v22, v10, s[4:5] offset:204
	global_load_i8 v31, v11, s[4:5] offset:240
	global_load_u16 v23, v10, s[4:5] offset:238
	s_waitcnt vmcnt(15)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(14)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v30, v30
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v22, v30, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v31, v31
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v23, v31, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	global_load_b32 v39, v9, s[6:7] offset:896
	s_waitcnt vmcnt(7)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v38 offset:768
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v39 offset:896
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	ds_load_b32 v39, v7 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v39, v23
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	ds_load_b32 v38, v7 offset:1792
	ds_load_b32 v39, v7 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v39, v23
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	ds_load_b32 v38, v7 offset:2816
	ds_load_b32 v39, v7 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v39, v23
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	ds_load_b32 v38, v7 offset:3840
	ds_load_b32 v39, v7 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v39, v23
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	ds_load_b32 v38, v7 offset:4864
	ds_load_b32 v39, v7 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v39, v23
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	ds_load_b32 v38, v7 offset:5888
	ds_load_b32 v39, v7 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v39, v23
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	ds_load_b32 v38, v7 offset:6912
	ds_load_b32 v39, v7 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v39, v23
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	ds_load_b32 v38, v7 offset:7936
	ds_load_b32 v39, v7 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v37, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v38, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v39, v23
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_full8_t0_done:
	v_add_nc_u32_e32 v8, 0x400, v8
	v_add_nc_u32_e32 v10, 0x110, v10
	v_add_nc_u32_e32 v11, 0x110, v11
	s_add_i32 s23, s23, -1
	s_cmp_lg_u32 s23, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kloop
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tails:
	s_cmp_eq_u32 s22, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail1
	s_cmp_eq_u32 s22, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail2
	s_cmp_eq_u32 s22, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail3
	s_cmp_eq_u32 s22, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail4
	s_cmp_eq_u32 s22, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail5
	s_cmp_eq_u32 s22, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail6
	s_cmp_eq_u32 s22, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail7
	s_branch .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail1:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v32
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v40, v32, v16
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v41, v32, v16
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v42, v32, v16
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v43, v32, v16
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v44, v32, v16
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v45, v32, v16
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v46, v32, v16
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v47, v32, v16
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part1_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail2:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v33 offset:128
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part2_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail3:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v34 offset:256
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part3_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail4:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v35 offset:384
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part4_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail5:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v36 offset:512
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part5_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail6:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v37 offset:640
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v37, v21
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v37, v21
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v37, v21
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v37, v21
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v37, v21
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v37, v21
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v37, v21
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v37, v21
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part6_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_tail7:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	global_load_i8 v30, v11, s[4:5] offset:206
	global_load_u16 v22, v10, s[4:5] offset:204
	s_waitcnt vmcnt(13)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v30, v30
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v22, v30, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v38 offset:768
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v40, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v38, v22
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	ds_load_b32 v38, v7 offset:1792
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v38, v22
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	ds_load_b32 v38, v7 offset:2816
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v42, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v38, v22
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	ds_load_b32 v38, v7 offset:3840
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v43, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v43, v38, v22
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	ds_load_b32 v38, v7 offset:4864
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v44, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v38, v22
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	ds_load_b32 v38, v7 offset:5888
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v45, v38, v22
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	ds_load_b32 v38, v7 offset:6912
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v46, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v46, v38, v22
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	ds_load_b32 v38, v7 offset:7936
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v47, v32, v16
	s_waitcnt lgkmcnt(5)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v33, v17
	s_waitcnt lgkmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v34, v18
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v35, v19
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v36, v20
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v37, v21
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v38, v22
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_cdone:
	s_barrier
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_part7_t0_done:
	s_branch .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_kdone:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_end
	s_cmp_le_u32 s12, 0
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip0
	ds_swizzle_b32 v32, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v2, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v3, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v5, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip0:
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip1
	ds_swizzle_b32 v32, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v2, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v3, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v4, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	ds_bpermute_b32 v32, v5, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v32
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip1:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip2
	ds_swizzle_b32 v32, v42 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v2, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v3, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v4, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	ds_bpermute_b32 v32, v5, v42
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v32
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip2:
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip3
	ds_swizzle_b32 v32, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v2, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v3, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	ds_bpermute_b32 v32, v5, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v32
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip3:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip4
	ds_swizzle_b32 v32, v44 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v2, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v3, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v5, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip4:
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip5
	ds_swizzle_b32 v32, v45 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v2, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v3, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v4, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	ds_bpermute_b32 v32, v5, v45
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v32
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip5:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip6
	ds_swizzle_b32 v32, v46 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v2, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v3, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v4, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	ds_bpermute_b32 v32, v5, v46
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v32
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip6:
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip7
	ds_swizzle_b32 v32, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v2, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v3, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	ds_bpermute_b32 v32, v5, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v32
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_reduce_skip7:
	v_mov_b32_e32 v24, s42
	s_mul_i32 s38, s41, 1
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v25, s38
	s_mul_i32 s38, s41, 2
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v26, s38
	s_mul_i32 s38, s41, 3
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v27, s38
	s_mul_i32 s38, s41, 4
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v28, s38
	s_mul_i32 s38, s41, 5
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v29, s38
	s_mul_i32 s38, s41, 6
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v30, s38
	s_mul_i32 s38, s41, 7
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v31, s38
	s_mov_b32 exec_lo, s40
	global_store_b32 v24, v40, s[8:9]
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_stored0
	global_store_b32 v25, v41, s[8:9]
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_stored0
	global_store_b32 v26, v42, s[8:9]
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_stored0
	global_store_b32 v27, v43, s[8:9]
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_stored0
	global_store_b32 v28, v44, s[8:9]
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_stored0
	global_store_b32 v29, v45, s[8:9]
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_stored0
	global_store_b32 v30, v46, s[8:9]
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_stored0
	global_store_b32 v31, v47, s[8:9]
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_stored0:
	s_mov_b32 exec_lo, -1
	s_waitcnt_vscnt null, 0x0
	.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_body_end:
	s_endpgm
.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_end:
.size gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100, .Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_end-gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100
	.amdhsa_group_segment_fixed_size 8192
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 56
	.amdhsa_user_sgpr_count 14
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
	.amdhsa_next_free_vgpr 48
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
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_q8_0_k2560_rows_x2_t8_pm_gfx1100_end-gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.text
.protected gemv_q8_0_wide_rows_t8_pm_gfx1100
.globl gemv_q8_0_wide_rows_t8_pm_gfx1100
.p2align 8
.type gemv_q8_0_wide_rows_t8_pm_gfx1100,@function
gemv_q8_0_wide_rows_t8_pm_gfx1100:
	s_load_b128 s[4:7], s[0:1], null
	s_load_b64 s[8:9], s[0:1], 0x10
	s_load_b64 s[10:11], s[0:1], 0x18
	s_load_b32 s12, s[0:1], 0x20
	v_and_b32_e32 v1, 31, v0
	v_lshrrev_b32_e32 v13, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s49, v13
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 24, v1
	v_cndmask_b32_e64 v2, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v2, v2, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 28, v1
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 30, v1
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v1, 2
	v_cmp_gt_u32_e32 vcc_lo, 31, v1
	v_cndmask_b32_e64 v5, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, v1, 2
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_mov_b32 s40, vcc_lo
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_end
	s_cmp_gt_u32 s12, 64
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_end
	s_and_b32 s39, s11, 31
	s_cmp_lg_u32 s39, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_end
	s_cmp_eq_u32 s11, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_end
	s_cmp_gt_u32 s11, 0x600
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_end
	s_lshl_b32 s16, s14, 3
	s_cmp_ge_i32 s16, s10
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_end
	s_lshl_b32 s15, s15, 3
	s_cmp_ge_u32 s15, s12
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_end
	s_sub_u32 s12, s12, s15
	s_min_u32 s12, s12, 8
	s_mul_i32 s38, s15, s11
	s_lshl_b32 s38, s38, 2
	s_add_u32 s6, s6, s38
	s_addc_u32 s7, s7, 0
	s_mul_i32 s38, s15, s10
	s_lshl_b32 s38, s38, 2
	s_add_u32 s8, s8, s38
	s_addc_u32 s9, s9, 0
	s_mov_b32 s45, s49
	s_add_i32 s17, s16, s45
	s_cmp_lt_i32 s17, s10
	s_cselect_b32 s18, 1, 0
	s_lshr_b32 s20, s11, 5
	s_mul_i32 s19, s20, 34
	s_mul_i32 s48, s17, s19
	v_mov_b32_e32 v10, s48
	v_add_nc_u32_e32 v11, s48, v1
	s_lshl_b32 s43, s11, 2
	s_mul_i32 s44, s45, s43
	s_lshr_b32 s21, s20, 3
	s_and_b32 s22, s20, 7
	v_lshlrev_b32_e32 v8, 2, v1
	s_lshl_b32 s39, s49, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v6, s39, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v7, v8
	s_lshl_b32 s41, s10, 2
	s_lshl_b32 s42, s17, 2
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	s_mov_b32 s23, s21
	s_cmp_eq_u32 s23, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tails
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kloop:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	global_load_i8 v30, v11, s[4:5] offset:206
	global_load_u16 v22, v10, s[4:5] offset:204
	global_load_i8 v31, v11, s[4:5] offset:240
	global_load_u16 v23, v10, s[4:5] offset:238
	s_waitcnt vmcnt(15)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(14)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v30, v30
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v22, v30, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v31, v31
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v23, v31, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	global_load_b32 v39, v9, s[6:7] offset:896
	s_waitcnt vmcnt(7)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v38 offset:768
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v39 offset:896
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	ds_load_b32 v39, v7 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v23, v39
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	ds_load_b32 v38, v7 offset:1792
	ds_load_b32 v39, v7 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v23, v39
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	ds_load_b32 v38, v7 offset:2816
	ds_load_b32 v39, v7 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v23, v39
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	ds_load_b32 v38, v7 offset:3840
	ds_load_b32 v39, v7 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v23, v39
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	ds_load_b32 v38, v7 offset:4864
	ds_load_b32 v39, v7 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v23, v39
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	ds_load_b32 v38, v7 offset:5888
	ds_load_b32 v39, v7 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v23, v39
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	ds_load_b32 v38, v7 offset:6912
	ds_load_b32 v39, v7 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v23, v39
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	ds_load_b32 v38, v7 offset:7936
	ds_load_b32 v39, v7 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v16, v32
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v17, v33
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v18, v34
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v19, v35
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v20, v36
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v21, v37
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v22, v38
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v23, v39
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_cdone:
	s_barrier
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_full8_t0_done:
	v_add_nc_u32_e32 v8, 0x400, v8
	v_add_nc_u32_e32 v10, 0x110, v10
	v_add_nc_u32_e32 v11, 0x110, v11
	s_add_i32 s23, s23, -1
	s_cmp_lg_u32 s23, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kloop
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tails:
	s_cmp_eq_u32 s22, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail1
	s_cmp_eq_u32 s22, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail2
	s_cmp_eq_u32 s22, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail3
	s_cmp_eq_u32 s22, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail4
	s_cmp_eq_u32 s22, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail5
	s_cmp_eq_u32 s22, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail6
	s_cmp_eq_u32 s22, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail7
	s_branch .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail1:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v32
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_cdone:
	s_barrier
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part1_t0_done:
	s_branch .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail2:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v33 offset:128
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v12
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_cdone:
	s_barrier
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part2_t0_done:
	s_branch .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail3:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v34 offset:256
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v32, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v33, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v34, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v12
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_cdone:
	s_barrier
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part3_t0_done:
	s_branch .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail4:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v35 offset:384
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v40, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v41, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v42, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v43, v19, v35
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v44, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v45, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v46, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v47, v19, v35
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v48, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v49, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v50, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v51, v19, v35
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v52, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v53, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v54, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v55, v19, v35
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v56, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v57, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v58, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v59, v19, v35
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v60, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v61, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v62, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v63, v19, v35
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v64, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v65, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v66, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v67, v19, v35
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v68, v16, v32
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v69, v17, v33
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v70, v18, v34
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v71, v19, v35
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_cdone:
	s_barrier
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part4_t0_done:
	s_branch .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail5:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v36 offset:512
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v40, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v41, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v42, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v43, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v44, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v45, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v46, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v47, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v48, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v49, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v50, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v51, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v52, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v53, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v54, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v55, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v56, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v57, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v58, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v59, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v60, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v61, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v62, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v63, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v64, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v65, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v66, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v67, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v68, v16, v32
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v69, v17, v33
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v70, v18, v34
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v71, v19, v35
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_cdone:
	s_barrier
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part5_t0_done:
	s_branch .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail6:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v37 offset:640
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v40, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v41, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v42, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v43, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v44, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v45, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v46, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v47, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v48, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v49, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v50, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v51, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v52, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v53, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v54, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v55, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v56, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v57, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v58, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v59, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v60, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v61, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v62, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v63, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v64, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v65, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v66, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v67, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v68, v16, v32
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v69, v17, v33
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v70, v18, v34
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v71, v19, v35
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v12
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_cdone:
	s_barrier
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part6_t0_done:
	s_branch .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_tail7:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_wdone
	global_load_i8 v24, v11, s[4:5] offset:2
	global_load_u16 v16, v10, s[4:5]
	global_load_i8 v25, v11, s[4:5] offset:36
	global_load_u16 v17, v10, s[4:5] offset:34
	global_load_i8 v26, v11, s[4:5] offset:70
	global_load_u16 v18, v10, s[4:5] offset:68
	global_load_i8 v27, v11, s[4:5] offset:104
	global_load_u16 v19, v10, s[4:5] offset:102
	global_load_i8 v28, v11, s[4:5] offset:138
	global_load_u16 v20, v10, s[4:5] offset:136
	global_load_i8 v29, v11, s[4:5] offset:172
	global_load_u16 v21, v10, s[4:5] offset:170
	global_load_i8 v30, v11, s[4:5] offset:206
	global_load_u16 v22, v10, s[4:5] offset:204
	s_waitcnt vmcnt(13)
	v_cvt_f32_i32_e32 v24, v24
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v16, v24, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(11)
	v_cvt_f32_i32_e32 v25, v25
	s_waitcnt vmcnt(10)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v17, v25, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(9)
	v_cvt_f32_i32_e32 v26, v26
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v18, v26, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(7)
	v_cvt_f32_i32_e32 v27, v27
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v19, v27, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_i32_e32 v28, v28
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v20, v28, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(3)
	v_cvt_f32_i32_e32 v29, v29
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v21, v29, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(1)
	v_cvt_f32_i32_e32 v30, v30
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v22, v30, neg(0) op_sel_hi:[1,0,0]
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_wdone:
	s_cmp_ge_u32 s45, s12
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_nofill
	v_add_nc_u32_e32 v9, s44, v8
	global_load_b32 v32, v9, s[6:7]
	global_load_b32 v33, v9, s[6:7] offset:128
	global_load_b32 v34, v9, s[6:7] offset:256
	global_load_b32 v35, v9, s[6:7] offset:384
	global_load_b32 v36, v9, s[6:7] offset:512
	global_load_b32 v37, v9, s[6:7] offset:640
	global_load_b32 v38, v9, s[6:7] offset:768
	s_waitcnt vmcnt(6)
	ds_store_b32 v6, v32
	s_waitcnt vmcnt(5)
	ds_store_b32 v6, v33 offset:128
	s_waitcnt vmcnt(4)
	ds_store_b32 v6, v34 offset:256
	s_waitcnt vmcnt(3)
	ds_store_b32 v6, v35 offset:384
	s_waitcnt vmcnt(2)
	ds_store_b32 v6, v36 offset:512
	s_waitcnt vmcnt(1)
	ds_store_b32 v6, v37 offset:640
	s_waitcnt vmcnt(0)
	ds_store_b32 v6, v38 offset:768
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_nofill:
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7
	ds_load_b32 v33, v7 offset:128
	ds_load_b32 v34, v7 offset:256
	ds_load_b32 v35, v7 offset:384
	ds_load_b32 v36, v7 offset:512
	ds_load_b32 v37, v7 offset:640
	ds_load_b32 v38, v7 offset:768
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v40, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v41, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v42, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v43, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v12
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:1024
	ds_load_b32 v33, v7 offset:1152
	ds_load_b32 v34, v7 offset:1280
	ds_load_b32 v35, v7 offset:1408
	ds_load_b32 v36, v7 offset:1536
	ds_load_b32 v37, v7 offset:1664
	ds_load_b32 v38, v7 offset:1792
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v44, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v45, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v46, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v47, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v12
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:2048
	ds_load_b32 v33, v7 offset:2176
	ds_load_b32 v34, v7 offset:2304
	ds_load_b32 v35, v7 offset:2432
	ds_load_b32 v36, v7 offset:2560
	ds_load_b32 v37, v7 offset:2688
	ds_load_b32 v38, v7 offset:2816
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v48, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v49, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v50, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v51, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v12
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:3072
	ds_load_b32 v33, v7 offset:3200
	ds_load_b32 v34, v7 offset:3328
	ds_load_b32 v35, v7 offset:3456
	ds_load_b32 v36, v7 offset:3584
	ds_load_b32 v37, v7 offset:3712
	ds_load_b32 v38, v7 offset:3840
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v52, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v53, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v54, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v55, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v12
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:4096
	ds_load_b32 v33, v7 offset:4224
	ds_load_b32 v34, v7 offset:4352
	ds_load_b32 v35, v7 offset:4480
	ds_load_b32 v36, v7 offset:4608
	ds_load_b32 v37, v7 offset:4736
	ds_load_b32 v38, v7 offset:4864
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v56, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v57, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v58, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v59, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v12
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:5120
	ds_load_b32 v33, v7 offset:5248
	ds_load_b32 v34, v7 offset:5376
	ds_load_b32 v35, v7 offset:5504
	ds_load_b32 v36, v7 offset:5632
	ds_load_b32 v37, v7 offset:5760
	ds_load_b32 v38, v7 offset:5888
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v60, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v61, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v62, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v63, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v12
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:6144
	ds_load_b32 v33, v7 offset:6272
	ds_load_b32 v34, v7 offset:6400
	ds_load_b32 v35, v7 offset:6528
	ds_load_b32 v36, v7 offset:6656
	ds_load_b32 v37, v7 offset:6784
	ds_load_b32 v38, v7 offset:6912
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v64, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v65, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v66, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v67, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v12
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_cdone
	ds_load_b32 v32, v7 offset:7168
	ds_load_b32 v33, v7 offset:7296
	ds_load_b32 v34, v7 offset:7424
	ds_load_b32 v35, v7 offset:7552
	ds_load_b32 v36, v7 offset:7680
	ds_load_b32 v37, v7 offset:7808
	ds_load_b32 v38, v7 offset:7936
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v68, v16, v32
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v69, v17, v33
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v70, v18, v34
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v71, v19, v35
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v12, v36, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, v37, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v38, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v12
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_cdone:
	s_barrier
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_part7_t0_done:
	s_branch .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kdone
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_kdone:
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_end
	s_cmp_le_u32 s12, 0
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip0
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v43, v41
	ds_swizzle_b32 v32, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v2, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v3, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v4, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	ds_bpermute_b32 v32, v5, v40
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v32
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip0:
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip1
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v47, v45
	ds_swizzle_b32 v32, v44 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v2, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v3, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v4, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	ds_bpermute_b32 v32, v5, v44
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v32
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip1:
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip2
	v_add_f32_e32 v49, v49, v48
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v51, v49
	ds_swizzle_b32 v32, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v32
	ds_bpermute_b32 v32, v2, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v32
	ds_bpermute_b32 v32, v3, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v32
	ds_bpermute_b32 v32, v4, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v32
	ds_bpermute_b32 v32, v5, v48
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v32
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip2:
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip3
	v_add_f32_e32 v53, v53, v52
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v55, v53
	ds_swizzle_b32 v32, v52 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v32
	ds_bpermute_b32 v32, v2, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v32
	ds_bpermute_b32 v32, v3, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v32
	ds_bpermute_b32 v32, v4, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v32
	ds_bpermute_b32 v32, v5, v52
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v32
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip3:
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip4
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v59, v57
	ds_swizzle_b32 v32, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v32
	ds_bpermute_b32 v32, v2, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v32
	ds_bpermute_b32 v32, v3, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v32
	ds_bpermute_b32 v32, v4, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v32
	ds_bpermute_b32 v32, v5, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v32
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip4:
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip5
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v63, v61
	ds_swizzle_b32 v32, v60 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v32
	ds_bpermute_b32 v32, v2, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v32
	ds_bpermute_b32 v32, v3, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v32
	ds_bpermute_b32 v32, v4, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v32
	ds_bpermute_b32 v32, v5, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v32
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip5:
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip6
	v_add_f32_e32 v65, v65, v64
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v67, v65
	ds_swizzle_b32 v32, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v32
	ds_bpermute_b32 v32, v2, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v32
	ds_bpermute_b32 v32, v3, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v32
	ds_bpermute_b32 v32, v4, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v32
	ds_bpermute_b32 v32, v5, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v32
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip6:
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip7
	v_add_f32_e32 v69, v69, v68
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v71, v69
	ds_swizzle_b32 v32, v68 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v32
	ds_bpermute_b32 v32, v2, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v32
	ds_bpermute_b32 v32, v3, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v32
	ds_bpermute_b32 v32, v4, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v32
	ds_bpermute_b32 v32, v5, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v32
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_reduce_skip7:
	v_mov_b32_e32 v24, s42
	s_mul_i32 s38, s41, 1
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v25, s38
	s_mul_i32 s38, s41, 2
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v26, s38
	s_mul_i32 s38, s41, 3
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v27, s38
	s_mul_i32 s38, s41, 4
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v28, s38
	s_mul_i32 s38, s41, 5
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v29, s38
	s_mul_i32 s38, s41, 6
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v30, s38
	s_mul_i32 s38, s41, 7
	s_add_i32 s38, s38, s42
	v_mov_b32_e32 v31, s38
	s_mov_b32 exec_lo, s40
	global_store_b32 v24, v40, s[8:9]
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v25, v44, s[8:9]
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v26, v48, s[8:9]
	s_cmp_le_u32 s12, 3
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v27, v52, s[8:9]
	s_cmp_le_u32 s12, 4
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v28, v56, s[8:9]
	s_cmp_le_u32 s12, 5
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v29, v60, s[8:9]
	s_cmp_le_u32 s12, 6
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v30, v64, s[8:9]
	s_cmp_le_u32 s12, 7
	s_cbranch_scc1 .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_stored0
	global_store_b32 v31, v68, s[8:9]
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_stored0:
	s_mov_b32 exec_lo, -1
	s_waitcnt_vscnt null, 0x0
	.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_body_end:
	s_endpgm
.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_end:
.size gemv_q8_0_wide_rows_t8_pm_gfx1100, .Lgemv_q8_0_wide_rows_t8_pm_gfx1100_end-gemv_q8_0_wide_rows_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_q8_0_wide_rows_t8_pm_gfx1100
	.amdhsa_group_segment_fixed_size 8192
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 36
	.amdhsa_user_sgpr_count 14
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
	.amdhsa_next_free_vgpr 72
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
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_q8_0_wide_rows_t8_pm_gfx1100_end-gemv_q8_0_wide_rows_t8_pm_gfx1100)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 256
    .name: gemv_mq6g256v2_rows_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 62
    .sgpr_spill_count: 0
    .symbol: gemv_mq6g256v2_rows_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 72
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A2
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A3
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y2
        .offset: 56
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y3
        .offset: 64
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: M2
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: M3
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: gemv_mq6g256v2_rows_x4_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 74
    .sgpr_spill_count: 0
    .symbol: gemv_mq6g256v2_rows_x4_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 72
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: scale
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: hc_act
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 44
    .max_flat_workgroup_size: 512
    .name: gemv_q8_0_k8_rows_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 53
    .sgpr_spill_count: 0
    .symbol: gemv_q8_0_k8_rows_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 48
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 256
    .name: gemv_q8_0_k320_rows_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 52
    .sgpr_spill_count: 0
    .symbol: gemv_q8_0_k320_rows_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 72
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 256
    .name: gemv_q8_0_k2560_rows_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 52
    .sgpr_spill_count: 0
    .symbol: gemv_q8_0_k2560_rows_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 48
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 40
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 44
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 48
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 52
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .max_flat_workgroup_size: 256
    .name: gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 52
    .sgpr_spill_count: 0
    .symbol: gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 48
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 256
    .name: gemv_q8_0_wide_rows_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 52
    .sgpr_spill_count: 0
    .symbol: gemv_q8_0_wide_rows_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 72
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1100
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
