.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected topk8_partial_f32_blocks_pm_gfx1201
.globl topk8_partial_f32_blocks_pm_gfx1201
.p2align 8
.type topk8_partial_f32_blocks_pm_gfx1201,@function
topk8_partial_f32_blocks_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b128 s[16:19], s[0:1], 0x20
	v_lshlrev_b32_e32 v1, 2, v0
	v_lshrrev_b32_e32 v12, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 3, v12
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v10, 0xff800000
	v_mov_b32_e32 v11, 0x7fffffff
	s_mov_b32 s45, 0
	s_mov_b32 s47, 0
	s_mov_b32 s49, 0
	s_mov_b32 s52, 0x76543210
	s_mov_b32 s29, 4
	s_mov_b32 s30, 32
	s_wait_kmcnt 0x0
	s_add_co_i32 s20, ttmp9, s19
	s_mul_i32 s21, s20, s11
	s_add_co_i32 s23, s21, s11
	s_min_i32 s23, s10, s23
	s_sub_co_i32 s22, s23, s21
	s_max_i32 s22, s22, 0
	s_add_co_i32 s26, s16, s18
	s_sub_co_i32 s27, s17, s16
	s_sub_co_i32 s28, 0, s18
	s_mul_i32 s24, s21, s29
	s_mul_hi_u32 s25, s21, s29
	s_add_co_u32 s32, s8, s24
	s_add_co_ci_u32 s33, s9, s25
	s_lshl_b32 s23, s22, 2
	s_mov_b32 s34, s23
	s_mov_b32 s35, 0x31004000
	s_mul_i32 s24, s20, s30
	s_mul_hi_u32 s25, s20, s30
	s_add_co_u32 s36, s12, s24
	s_add_co_ci_u32 s37, s13, s25
	s_mov_b32 s38, s30
	s_mov_b32 s39, 0x31004000
	s_mul_i32 s24, s20, s30
	s_mul_hi_u32 s25, s20, s30
	s_add_co_u32 s40, s14, s24
	s_add_co_ci_u32 s41, s15, s25
	s_mov_b32 s42, s30
	s_mov_b32 s43, 0x31004000
	buffer_load_b32 v16, v1, s[32:35], null offen
	buffer_load_b32 v17, v1, s[32:35], null offen offset:1024
	buffer_load_b32 v18, v1, s[32:35], null offen offset:2048
	buffer_load_b32 v19, v1, s[32:35], null offen offset:3072
	buffer_load_b32 v20, v1, s[32:35], null offen offset:4096
	buffer_load_b32 v21, v1, s[32:35], null offen offset:5120
	buffer_load_b32 v22, v1, s[32:35], null offen offset:6144
	buffer_load_b32 v23, v1, s[32:35], null offen offset:7168
	v_add_nc_u32_e32 v12, s21, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v24, v12
	v_cmp_gt_i32_e64 s44, s22, v0
	s_wait_loadcnt 0x7
	v_cndmask_b32_e64 v16, v10, v16, s44
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v24, v11, v24, s44
	v_add_nc_u32_e32 v25, 0x100, v12
	v_add_nc_u32_e32 v13, 0x100, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s44, s22, v13
	s_wait_loadcnt 0x6
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, v10, v17, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v25, v11, v25, s44
	v_add_nc_u32_e32 v26, 0x200, v12
	v_add_nc_u32_e32 v13, 0x200, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s44, s22, v13
	s_wait_loadcnt 0x5
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v18, v10, v18, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v26, v11, v26, s44
	v_add_nc_u32_e32 v27, 0x300, v12
	v_add_nc_u32_e32 v13, 0x300, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s44, s22, v13
	s_wait_loadcnt 0x4
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, v10, v19, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v27, v11, v27, s44
	v_add_nc_u32_e32 v28, 0x400, v12
	v_add_nc_u32_e32 v13, 0x400, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s44, s22, v13
	s_wait_loadcnt 0x3
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, v10, v20, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v28, v11, v28, s44
	v_add_nc_u32_e32 v29, 0x500, v12
	v_add_nc_u32_e32 v13, 0x500, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s44, s22, v13
	s_wait_loadcnt 0x2
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v21, v10, v21, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v29, v11, v29, s44
	v_add_nc_u32_e32 v30, 0x600, v12
	v_add_nc_u32_e32 v13, 0x600, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s44, s22, v13
	s_wait_loadcnt 0x1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v22, v10, v22, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v30, v11, v30, s44
	v_add_nc_u32_e32 v31, 0x700, v12
	v_add_nc_u32_e32 v13, 0x700, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s44, s22, v13
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v23, v10, v23, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v31, v11, v31, s44
	v_mov_b32_e32 v6, v10
	v_mov_b32_e32 v7, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v24, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v16, s44
	v_cndmask_b32_e64 v7, v7, v24, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v25, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v17, s44
	v_cndmask_b32_e64 v7, v7, v25, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v26, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v18, s44
	v_cndmask_b32_e64 v7, v7, v26, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v27, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v19, s44
	v_cndmask_b32_e64 v7, v7, v27, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v28, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v20, s44
	v_cndmask_b32_e64 v7, v7, v28, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v29, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v21, s44
	v_cndmask_b32_e64 v7, v7, v29, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v30, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v22, s44
	v_cndmask_b32_e64 v7, v7, v30, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v31, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v23, s44
	v_cndmask_b32_e64 v7, v7, v31, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v8, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v9, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v4, v8, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v5, v9, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	ds_store_b64 v2, v[8:9]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[32:35], v3
	ds_load_b128 v[36:39], v3 offset:16
	ds_load_b128 v[40:43], v3 offset:32
	ds_load_b128 v[44:47], v3 offset:48
	s_wait_dscnt 0x3
	v_mov_b32_e32 v6, v32
	v_mov_b32_e32 v7, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v35, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v34, s44
	v_cndmask_b32_e64 v7, v7, v35, s44
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v37, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v36, s44
	v_cndmask_b32_e64 v7, v7, v37, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v39, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v38, s44
	v_cndmask_b32_e64 v7, v7, v39, s44
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v41, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v40, s44
	v_cndmask_b32_e64 v7, v7, v41, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v43, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v42, s44
	v_cndmask_b32_e64 v7, v7, v43, s44
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v45, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v44, s44
	v_cndmask_b32_e64 v7, v7, v45, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v47, v7
	s_and_b32 s46, s46, s48
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v46, s44
	v_cndmask_b32_e64 v7, v7, v47, s44
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e64 s44, v7, v24
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v16, v16, v10, s44
	v_cndmask_b32_e64 v24, v24, v11, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_eq_u32_e64 s44, v7, v25
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, v17, v10, s44
	v_cndmask_b32_e64 v25, v25, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v26
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v18, v18, v10, s44
	v_cndmask_b32_e64 v26, v26, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v27
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, v19, v10, s44
	v_cndmask_b32_e64 v27, v27, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v28
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, v20, v10, s44
	v_cndmask_b32_e64 v28, v28, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v29
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v21, v21, v10, s44
	v_cndmask_b32_e64 v29, v29, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v30
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v22, v22, v10, s44
	v_cndmask_b32_e64 v30, v30, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v31
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v23, v23, v10, s44
	v_cndmask_b32_e64 v31, v31, v11, s44
	v_mov_b32_e32 v48, v6
	v_add_nc_u32_e32 v12, s27, v7
	v_add_nc_u32_e32 v13, s28, v7
	v_cmp_gt_i32_e64 s44, s26, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v56, v13, v12, s44
	v_cmp_gt_i32_e64 s46, s16, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v56, v56, v7, s46
	v_mov_b32_e32 v6, v10
	v_mov_b32_e32 v7, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v24, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v16, s44
	v_cndmask_b32_e64 v7, v7, v24, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v25, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v17, s44
	v_cndmask_b32_e64 v7, v7, v25, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v26, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v18, s44
	v_cndmask_b32_e64 v7, v7, v26, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v27, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v19, s44
	v_cndmask_b32_e64 v7, v7, v27, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v28, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v20, s44
	v_cndmask_b32_e64 v7, v7, v28, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v29, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v21, s44
	v_cndmask_b32_e64 v7, v7, v29, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v30, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v22, s44
	v_cndmask_b32_e64 v7, v7, v30, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v31, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v23, s44
	v_cndmask_b32_e64 v7, v7, v31, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v8, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v9, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v4, v8, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v5, v9, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	ds_store_b64 v2, v[8:9]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[32:35], v3
	ds_load_b128 v[36:39], v3 offset:16
	ds_load_b128 v[40:43], v3 offset:32
	ds_load_b128 v[44:47], v3 offset:48
	s_wait_dscnt 0x3
	v_mov_b32_e32 v6, v32
	v_mov_b32_e32 v7, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v35, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v34, s44
	v_cndmask_b32_e64 v7, v7, v35, s44
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v37, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v36, s44
	v_cndmask_b32_e64 v7, v7, v37, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v39, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v38, s44
	v_cndmask_b32_e64 v7, v7, v39, s44
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v41, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v40, s44
	v_cndmask_b32_e64 v7, v7, v41, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v43, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v42, s44
	v_cndmask_b32_e64 v7, v7, v43, s44
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v45, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v44, s44
	v_cndmask_b32_e64 v7, v7, v45, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v47, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v46, s44
	v_cndmask_b32_e64 v7, v7, v47, s44
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e64 s44, v7, v24
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v16, v16, v10, s44
	v_cndmask_b32_e64 v24, v24, v11, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_eq_u32_e64 s44, v7, v25
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, v17, v10, s44
	v_cndmask_b32_e64 v25, v25, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v26
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v18, v18, v10, s44
	v_cndmask_b32_e64 v26, v26, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v27
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, v19, v10, s44
	v_cndmask_b32_e64 v27, v27, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v28
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, v20, v10, s44
	v_cndmask_b32_e64 v28, v28, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v29
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v21, v21, v10, s44
	v_cndmask_b32_e64 v29, v29, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v30
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v22, v22, v10, s44
	v_cndmask_b32_e64 v30, v30, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v31
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v23, v23, v10, s44
	v_cndmask_b32_e64 v31, v31, v11, s44
	v_mov_b32_e32 v49, v6
	v_add_nc_u32_e32 v12, s27, v7
	v_add_nc_u32_e32 v13, s28, v7
	v_cmp_gt_i32_e64 s44, s26, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v57, v13, v12, s44
	v_cmp_gt_i32_e64 s46, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v57, v57, v7, s46
	v_mov_b32_e32 v6, v10
	v_mov_b32_e32 v7, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v24, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v16, s44
	v_cndmask_b32_e64 v7, v7, v24, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v25, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v17, s44
	v_cndmask_b32_e64 v7, v7, v25, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v26, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v18, s44
	v_cndmask_b32_e64 v7, v7, v26, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v27, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v19, s44
	v_cndmask_b32_e64 v7, v7, v27, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v28, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v20, s44
	v_cndmask_b32_e64 v7, v7, v28, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v29, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v21, s44
	v_cndmask_b32_e64 v7, v7, v29, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v30, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v22, s44
	v_cndmask_b32_e64 v7, v7, v30, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v31, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v23, s44
	v_cndmask_b32_e64 v7, v7, v31, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v8, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v9, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v4, v8, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v5, v9, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	ds_store_b64 v2, v[8:9]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[32:35], v3
	ds_load_b128 v[36:39], v3 offset:16
	ds_load_b128 v[40:43], v3 offset:32
	ds_load_b128 v[44:47], v3 offset:48
	s_wait_dscnt 0x3
	v_mov_b32_e32 v6, v32
	v_mov_b32_e32 v7, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v35, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v34, s44
	v_cndmask_b32_e64 v7, v7, v35, s44
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v37, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v36, s44
	v_cndmask_b32_e64 v7, v7, v37, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v39, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v38, s44
	v_cndmask_b32_e64 v7, v7, v39, s44
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v41, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v40, s44
	v_cndmask_b32_e64 v7, v7, v41, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v43, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v42, s44
	v_cndmask_b32_e64 v7, v7, v43, s44
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v45, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v44, s44
	v_cndmask_b32_e64 v7, v7, v45, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v47, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v46, s44
	v_cndmask_b32_e64 v7, v7, v47, s44
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e64 s44, v7, v24
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v16, v16, v10, s44
	v_cndmask_b32_e64 v24, v24, v11, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_eq_u32_e64 s44, v7, v25
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, v17, v10, s44
	v_cndmask_b32_e64 v25, v25, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v26
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v18, v18, v10, s44
	v_cndmask_b32_e64 v26, v26, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v27
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, v19, v10, s44
	v_cndmask_b32_e64 v27, v27, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v28
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, v20, v10, s44
	v_cndmask_b32_e64 v28, v28, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v29
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v21, v21, v10, s44
	v_cndmask_b32_e64 v29, v29, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v30
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v22, v22, v10, s44
	v_cndmask_b32_e64 v30, v30, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v31
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v23, v23, v10, s44
	v_cndmask_b32_e64 v31, v31, v11, s44
	v_mov_b32_e32 v50, v6
	v_add_nc_u32_e32 v12, s27, v7
	v_add_nc_u32_e32 v13, s28, v7
	v_cmp_gt_i32_e64 s44, s26, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v58, v13, v12, s44
	v_cmp_gt_i32_e64 s46, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v58, v58, v7, s46
	v_mov_b32_e32 v6, v10
	v_mov_b32_e32 v7, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v24, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v16, s44
	v_cndmask_b32_e64 v7, v7, v24, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v25, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v17, s44
	v_cndmask_b32_e64 v7, v7, v25, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v26, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v18, s44
	v_cndmask_b32_e64 v7, v7, v26, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v27, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v19, s44
	v_cndmask_b32_e64 v7, v7, v27, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v28, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v20, s44
	v_cndmask_b32_e64 v7, v7, v28, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v29, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v21, s44
	v_cndmask_b32_e64 v7, v7, v29, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v30, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v22, s44
	v_cndmask_b32_e64 v7, v7, v30, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v31, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v23, s44
	v_cndmask_b32_e64 v7, v7, v31, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v8, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v9, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v4, v8, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v5, v9, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	ds_store_b64 v2, v[8:9]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[32:35], v3
	ds_load_b128 v[36:39], v3 offset:16
	ds_load_b128 v[40:43], v3 offset:32
	ds_load_b128 v[44:47], v3 offset:48
	s_wait_dscnt 0x3
	v_mov_b32_e32 v6, v32
	v_mov_b32_e32 v7, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v35, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v34, s44
	v_cndmask_b32_e64 v7, v7, v35, s44
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v37, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v36, s44
	v_cndmask_b32_e64 v7, v7, v37, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v39, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v38, s44
	v_cndmask_b32_e64 v7, v7, v39, s44
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v41, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v40, s44
	v_cndmask_b32_e64 v7, v7, v41, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v43, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v42, s44
	v_cndmask_b32_e64 v7, v7, v43, s44
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v45, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v44, s44
	v_cndmask_b32_e64 v7, v7, v45, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v47, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v46, s44
	v_cndmask_b32_e64 v7, v7, v47, s44
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e64 s44, v7, v24
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v16, v16, v10, s44
	v_cndmask_b32_e64 v24, v24, v11, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_eq_u32_e64 s44, v7, v25
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, v17, v10, s44
	v_cndmask_b32_e64 v25, v25, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v26
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v18, v18, v10, s44
	v_cndmask_b32_e64 v26, v26, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v27
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, v19, v10, s44
	v_cndmask_b32_e64 v27, v27, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v28
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, v20, v10, s44
	v_cndmask_b32_e64 v28, v28, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v29
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v21, v21, v10, s44
	v_cndmask_b32_e64 v29, v29, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v30
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v22, v22, v10, s44
	v_cndmask_b32_e64 v30, v30, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v31
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v23, v23, v10, s44
	v_cndmask_b32_e64 v31, v31, v11, s44
	v_mov_b32_e32 v51, v6
	v_add_nc_u32_e32 v12, s27, v7
	v_add_nc_u32_e32 v13, s28, v7
	v_cmp_gt_i32_e64 s44, s26, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v59, v13, v12, s44
	v_cmp_gt_i32_e64 s46, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v59, v59, v7, s46
	v_mov_b32_e32 v6, v10
	v_mov_b32_e32 v7, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v24, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v16, s44
	v_cndmask_b32_e64 v7, v7, v24, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v25, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v17, s44
	v_cndmask_b32_e64 v7, v7, v25, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v26, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v18, s44
	v_cndmask_b32_e64 v7, v7, v26, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v27, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v19, s44
	v_cndmask_b32_e64 v7, v7, v27, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v28, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v20, s44
	v_cndmask_b32_e64 v7, v7, v28, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v29, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v21, s44
	v_cndmask_b32_e64 v7, v7, v29, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v30, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v22, s44
	v_cndmask_b32_e64 v7, v7, v30, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v31, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v23, s44
	v_cndmask_b32_e64 v7, v7, v31, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v8, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v9, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v4, v8, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v5, v9, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	ds_store_b64 v2, v[8:9]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[32:35], v3
	ds_load_b128 v[36:39], v3 offset:16
	ds_load_b128 v[40:43], v3 offset:32
	ds_load_b128 v[44:47], v3 offset:48
	s_wait_dscnt 0x3
	v_mov_b32_e32 v6, v32
	v_mov_b32_e32 v7, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v35, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v34, s44
	v_cndmask_b32_e64 v7, v7, v35, s44
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v37, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v36, s44
	v_cndmask_b32_e64 v7, v7, v37, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v39, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v38, s44
	v_cndmask_b32_e64 v7, v7, v39, s44
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v41, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v40, s44
	v_cndmask_b32_e64 v7, v7, v41, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v43, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v42, s44
	v_cndmask_b32_e64 v7, v7, v43, s44
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v45, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v44, s44
	v_cndmask_b32_e64 v7, v7, v45, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v47, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v46, s44
	v_cndmask_b32_e64 v7, v7, v47, s44
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e64 s44, v7, v24
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v16, v16, v10, s44
	v_cndmask_b32_e64 v24, v24, v11, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_eq_u32_e64 s44, v7, v25
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, v17, v10, s44
	v_cndmask_b32_e64 v25, v25, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v26
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v18, v18, v10, s44
	v_cndmask_b32_e64 v26, v26, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v27
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, v19, v10, s44
	v_cndmask_b32_e64 v27, v27, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v28
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, v20, v10, s44
	v_cndmask_b32_e64 v28, v28, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v29
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v21, v21, v10, s44
	v_cndmask_b32_e64 v29, v29, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v30
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v22, v22, v10, s44
	v_cndmask_b32_e64 v30, v30, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v31
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v23, v23, v10, s44
	v_cndmask_b32_e64 v31, v31, v11, s44
	v_mov_b32_e32 v52, v6
	v_add_nc_u32_e32 v12, s27, v7
	v_add_nc_u32_e32 v13, s28, v7
	v_cmp_gt_i32_e64 s44, s26, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v60, v13, v12, s44
	v_cmp_gt_i32_e64 s46, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v60, v60, v7, s46
	v_mov_b32_e32 v6, v10
	v_mov_b32_e32 v7, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v24, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v16, s44
	v_cndmask_b32_e64 v7, v7, v24, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v25, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v17, s44
	v_cndmask_b32_e64 v7, v7, v25, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v26, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v18, s44
	v_cndmask_b32_e64 v7, v7, v26, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v27, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v19, s44
	v_cndmask_b32_e64 v7, v7, v27, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v28, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v20, s44
	v_cndmask_b32_e64 v7, v7, v28, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v29, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v21, s44
	v_cndmask_b32_e64 v7, v7, v29, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v30, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v22, s44
	v_cndmask_b32_e64 v7, v7, v30, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v31, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v23, s44
	v_cndmask_b32_e64 v7, v7, v31, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v8, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v9, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v4, v8, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v5, v9, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	ds_store_b64 v2, v[8:9]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[32:35], v3
	ds_load_b128 v[36:39], v3 offset:16
	ds_load_b128 v[40:43], v3 offset:32
	ds_load_b128 v[44:47], v3 offset:48
	s_wait_dscnt 0x3
	v_mov_b32_e32 v6, v32
	v_mov_b32_e32 v7, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v35, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v34, s44
	v_cndmask_b32_e64 v7, v7, v35, s44
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v37, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v36, s44
	v_cndmask_b32_e64 v7, v7, v37, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v39, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v38, s44
	v_cndmask_b32_e64 v7, v7, v39, s44
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v41, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v40, s44
	v_cndmask_b32_e64 v7, v7, v41, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v43, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v42, s44
	v_cndmask_b32_e64 v7, v7, v43, s44
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v45, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v44, s44
	v_cndmask_b32_e64 v7, v7, v45, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v47, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v46, s44
	v_cndmask_b32_e64 v7, v7, v47, s44
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e64 s44, v7, v24
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v16, v16, v10, s44
	v_cndmask_b32_e64 v24, v24, v11, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_eq_u32_e64 s44, v7, v25
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, v17, v10, s44
	v_cndmask_b32_e64 v25, v25, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v26
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v18, v18, v10, s44
	v_cndmask_b32_e64 v26, v26, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v27
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, v19, v10, s44
	v_cndmask_b32_e64 v27, v27, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v28
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, v20, v10, s44
	v_cndmask_b32_e64 v28, v28, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v29
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v21, v21, v10, s44
	v_cndmask_b32_e64 v29, v29, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v30
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v22, v22, v10, s44
	v_cndmask_b32_e64 v30, v30, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v31
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v23, v23, v10, s44
	v_cndmask_b32_e64 v31, v31, v11, s44
	v_mov_b32_e32 v53, v6
	v_add_nc_u32_e32 v12, s27, v7
	v_add_nc_u32_e32 v13, s28, v7
	v_cmp_gt_i32_e64 s44, s26, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v61, v13, v12, s44
	v_cmp_gt_i32_e64 s46, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v61, v61, v7, s46
	v_mov_b32_e32 v6, v10
	v_mov_b32_e32 v7, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v24, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v16, s44
	v_cndmask_b32_e64 v7, v7, v24, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v25, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v17, s44
	v_cndmask_b32_e64 v7, v7, v25, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v26, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v18, s44
	v_cndmask_b32_e64 v7, v7, v26, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v27, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v19, s44
	v_cndmask_b32_e64 v7, v7, v27, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v28, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v20, s44
	v_cndmask_b32_e64 v7, v7, v28, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v29, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v21, s44
	v_cndmask_b32_e64 v7, v7, v29, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v30, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v22, s44
	v_cndmask_b32_e64 v7, v7, v30, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v31, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v23, s44
	v_cndmask_b32_e64 v7, v7, v31, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v8, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v9, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v4, v8, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v5, v9, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	ds_store_b64 v2, v[8:9]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[32:35], v3
	ds_load_b128 v[36:39], v3 offset:16
	ds_load_b128 v[40:43], v3 offset:32
	ds_load_b128 v[44:47], v3 offset:48
	s_wait_dscnt 0x3
	v_mov_b32_e32 v6, v32
	v_mov_b32_e32 v7, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v35, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v34, s44
	v_cndmask_b32_e64 v7, v7, v35, s44
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v37, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v36, s44
	v_cndmask_b32_e64 v7, v7, v37, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v39, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v38, s44
	v_cndmask_b32_e64 v7, v7, v39, s44
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v41, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v40, s44
	v_cndmask_b32_e64 v7, v7, v41, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v43, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v42, s44
	v_cndmask_b32_e64 v7, v7, v43, s44
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v45, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v44, s44
	v_cndmask_b32_e64 v7, v7, v45, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v47, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v46, s44
	v_cndmask_b32_e64 v7, v7, v47, s44
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e64 s44, v7, v24
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v16, v16, v10, s44
	v_cndmask_b32_e64 v24, v24, v11, s44
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_eq_u32_e64 s44, v7, v25
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, v17, v10, s44
	v_cndmask_b32_e64 v25, v25, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v26
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v18, v18, v10, s44
	v_cndmask_b32_e64 v26, v26, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v27
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, v19, v10, s44
	v_cndmask_b32_e64 v27, v27, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v28
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, v20, v10, s44
	v_cndmask_b32_e64 v28, v28, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v29
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v21, v21, v10, s44
	v_cndmask_b32_e64 v29, v29, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v30
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v22, v22, v10, s44
	v_cndmask_b32_e64 v30, v30, v11, s44
	v_cmp_eq_u32_e64 s44, v7, v31
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v23, v23, v10, s44
	v_cndmask_b32_e64 v31, v31, v11, s44
	v_mov_b32_e32 v54, v6
	v_add_nc_u32_e32 v12, s27, v7
	v_add_nc_u32_e32 v13, s28, v7
	v_cmp_gt_i32_e64 s44, s26, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v62, v13, v12, s44
	v_cmp_gt_i32_e64 s46, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v62, v62, v7, s46
	v_mov_b32_e32 v6, v10
	v_mov_b32_e32 v7, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v24, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v16, s44
	v_cndmask_b32_e64 v7, v7, v24, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v25, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v17, s44
	v_cndmask_b32_e64 v7, v7, v25, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v18, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v26, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v18, s44
	v_cndmask_b32_e64 v7, v7, v26, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v19, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v27, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v19, s44
	v_cndmask_b32_e64 v7, v7, v27, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v20, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v28, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v20, s44
	v_cndmask_b32_e64 v7, v7, v28, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v21, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v29, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v21, s44
	v_cndmask_b32_e64 v7, v7, v29, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v22, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v30, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v22, s44
	v_cndmask_b32_e64 v7, v7, v30, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v23, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v31, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v23, s44
	v_cndmask_b32_e64 v7, v7, v31, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v8, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v9, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:1 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:2 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:4 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v4, v8 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_dpp v5, v9 row_xmask:8 row_mask:0xf bank_mask:0xf
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v4, v8, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_permlanex16_b32 v5, v9, s52, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v4, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v5, v9
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v8, v4, s44
	v_cndmask_b32_e64 v9, v9, v5, s44
	ds_store_b64 v2, v[8:9]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[32:35], v3
	ds_load_b128 v[36:39], v3 offset:16
	ds_load_b128 v[40:43], v3 offset:32
	ds_load_b128 v[44:47], v3 offset:48
	s_wait_dscnt 0x3
	v_mov_b32_e32 v6, v32
	v_mov_b32_e32 v7, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v34, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v35, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v34, s44
	v_cndmask_b32_e64 v7, v7, v35, s44
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v36, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v37, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v36, s44
	v_cndmask_b32_e64 v7, v7, v37, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v38, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v39, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v38, s44
	v_cndmask_b32_e64 v7, v7, v39, s44
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v40, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v41, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v40, s44
	v_cndmask_b32_e64 v7, v7, v41, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v42, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v43, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v42, s44
	v_cndmask_b32_e64 v7, v7, v43, s44
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v44, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v45, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v44, s44
	v_cndmask_b32_e64 v7, v7, v45, s44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_f32_e64 s44, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_f32_e64 s46, v46, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s48, v47, v7
	s_and_b32 s46, s46, s48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s44, s44, s46
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v6, v6, v46, s44
	v_cndmask_b32_e64 v7, v7, v47, s44
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v55, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v12, s27, v7
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v13, s28, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_gt_i32_e64 s44, s26, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v63, v13, v12, s44
	v_cmp_gt_i32_e64 s46, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v63, v63, v7, s46
	v_cmp_eq_u32_e64 s46, 0, v0
	s_and_b32 exec_lo, exec_lo, s46
	buffer_store_b32 v48, v3, s[36:39], null offen
	buffer_store_b32 v56, v3, s[40:43], null offen
	buffer_store_b32 v49, v3, s[36:39], null offen offset:4
	buffer_store_b32 v57, v3, s[40:43], null offen offset:4
	buffer_store_b32 v50, v3, s[36:39], null offen offset:8
	buffer_store_b32 v58, v3, s[40:43], null offen offset:8
	buffer_store_b32 v51, v3, s[36:39], null offen offset:12
	buffer_store_b32 v59, v3, s[40:43], null offen offset:12
	buffer_store_b32 v52, v3, s[36:39], null offen offset:16
	buffer_store_b32 v60, v3, s[40:43], null offen offset:16
	buffer_store_b32 v53, v3, s[36:39], null offen offset:20
	buffer_store_b32 v61, v3, s[40:43], null offen offset:20
	buffer_store_b32 v54, v3, s[36:39], null offen offset:24
	buffer_store_b32 v62, v3, s[40:43], null offen offset:24
	buffer_store_b32 v55, v3, s[36:39], null offen offset:28
	buffer_store_b32 v63, v3, s[40:43], null offen offset:28
	s_wait_storecnt 0x0
	.Ltopk8_end:
	s_endpgm
.Ltopk8_partial_f32_blocks_pm_gfx1201_end:
.size topk8_partial_f32_blocks_pm_gfx1201, .Ltopk8_partial_f32_blocks_pm_gfx1201_end-topk8_partial_f32_blocks_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel topk8_partial_f32_blocks_pm_gfx1201
	.amdhsa_group_segment_fixed_size 64
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 48
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
	.amdhsa_next_free_vgpr 64
	.amdhsa_next_free_sgpr 56
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Ltopk8_partial_f32_blocks_pm_gfx1201_end-topk8_partial_f32_blocks_pm_gfx1201)<<4)&4080)>>4
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
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: v
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - 
        .name: n
        .offset: 8
        .size: 4
        .value_kind: by_value
      - 
        .name: chunk
        .offset: 12
        .size: 4
        .value_kind: by_value
      - .actual_access: write_only
        .address_space: global
        .name: pv
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: pi
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - 
        .name: front
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: special
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: tail
        .offset: 40
        .size: 4
        .value_kind: by_value
      - 
        .name: block0
        .offset: 44
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 64
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 256
    .name: topk8_partial_f32_blocks_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 56
    .sgpr_spill_count: 0
    .symbol: topk8_partial_f32_blocks_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 64
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
