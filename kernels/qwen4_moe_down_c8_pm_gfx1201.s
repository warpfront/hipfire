.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected qwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201
.globl qwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201
.p2align 8
.type qwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201,@function
qwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x20
	s_load_b128 s[20:23], s[0:1], 0x28
	s_load_b64 s[24:25], s[0:1], 0x38
	s_load_b32 s74, s[0:1], 0x40
	s_mov_b32 s72, ttmp9
	s_wait_kmcnt 0x0
	s_cmp_le_i32 s74, 0
	s_cbranch_scc1 .Lf02p_done
	s_cmp_le_i32 s22, 0
	s_cbranch_scc1 .Lf02p_done
	s_cmp_lg_u32 s21, 0x280
	s_cbranch_scc1 .Lf02p_done
	s_cmp_gt_u32 s23, 0x4000
	s_cbranch_scc1 .Lf02p_done
	s_cmp_gt_u32 s25, 0x400000
	s_cbranch_scc1 .Lf02p_done
	v_mov_b32_e32 v91, 0
	v_lshrrev_b32_e32 v89, 5, v0
	v_readfirstlane_b32 s80, v89
	s_add_co_i32 s81, s23, 15
	s_lshr_b32 s81, s81, 4
	s_mov_b32 s76, s10
	s_mov_b32 s77, s11
	s_lshl_b32 s78, s81, 2
	s_mov_b32 s79, 0x31004000
	v_add_nc_u32_e32 v8, 0, v0
	v_lshlrev_b32_e32 v24, 2, v8
	buffer_load_b32 v16, v24, s[76:79], null offen
	v_add_nc_u32_e32 v9, 0x80, v0
	v_lshlrev_b32_e32 v25, 2, v9
	buffer_load_b32 v17, v25, s[76:79], null offen
	v_add_nc_u32_e32 v10, 0x100, v0
	v_lshlrev_b32_e32 v26, 2, v10
	buffer_load_b32 v18, v26, s[76:79], null offen
	v_add_nc_u32_e32 v11, 0x180, v0
	v_lshlrev_b32_e32 v27, 2, v11
	buffer_load_b32 v19, v27, s[76:79], null offen
	v_add_nc_u32_e32 v12, 0x200, v0
	v_lshlrev_b32_e32 v28, 2, v12
	buffer_load_b32 v20, v28, s[76:79], null offen
	v_add_nc_u32_e32 v13, 0x280, v0
	v_lshlrev_b32_e32 v29, 2, v13
	buffer_load_b32 v21, v29, s[76:79], null offen
	v_add_nc_u32_e32 v14, 0x300, v0
	v_lshlrev_b32_e32 v30, 2, v14
	buffer_load_b32 v22, v30, s[76:79], null offen
	v_add_nc_u32_e32 v15, 0x380, v0
	v_lshlrev_b32_e32 v31, 2, v15
	buffer_load_b32 v23, v31, s[76:79], null offen
	v_cmp_gt_u32_e64 s82, s81, v8
	s_wait_loadcnt 0x7
	v_cmp_le_i32_e64 s83, 0, v16
	v_cmp_gt_i32_e64 s84, s25, v16
	s_and_b32 s82, s82, s83
	s_and_b32 s82, s82, s84
	v_cndmask_b32_e64 v32, 0, 1, s82
	v_mbcnt_lo_u32_b32 v24, s82, 0
	s_bcnt1_i32_b32 s85, s82
	v_mov_b32_e32 v89, s85
	s_lshl_b32 s86, s80, 2
	v_mov_b32_e32 v90, s86
	ds_store_b32 v90, v89 offset:15056
	v_cmp_gt_u32_e64 s82, s81, v9
	s_wait_loadcnt 0x6
	v_cmp_le_i32_e64 s83, 0, v17
	v_cmp_gt_i32_e64 s84, s25, v17
	s_and_b32 s82, s82, s83
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s82, s82, s84
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v33, 0, 1, s82
	v_mbcnt_lo_u32_b32 v25, s82, 0
	s_bcnt1_i32_b32 s85, s82
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v89, s85
	s_lshl_b32 s86, s80, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v90, s86
	ds_store_b32 v90, v89 offset:15072
	v_cmp_gt_u32_e64 s82, s81, v10
	s_wait_loadcnt 0x5
	v_cmp_le_i32_e64 s83, 0, v18
	v_cmp_gt_i32_e64 s84, s25, v18
	s_and_b32 s82, s82, s83
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s82, s82, s84
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v34, 0, 1, s82
	v_mbcnt_lo_u32_b32 v26, s82, 0
	s_bcnt1_i32_b32 s85, s82
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v89, s85
	s_lshl_b32 s86, s80, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v90, s86
	ds_store_b32 v90, v89 offset:15088
	v_cmp_gt_u32_e64 s82, s81, v11
	s_wait_loadcnt 0x4
	v_cmp_le_i32_e64 s83, 0, v19
	v_cmp_gt_i32_e64 s84, s25, v19
	s_and_b32 s82, s82, s83
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s82, s82, s84
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v35, 0, 1, s82
	v_mbcnt_lo_u32_b32 v27, s82, 0
	s_bcnt1_i32_b32 s85, s82
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v89, s85
	s_lshl_b32 s86, s80, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v90, s86
	ds_store_b32 v90, v89 offset:15104
	v_cmp_gt_u32_e64 s82, s81, v12
	s_wait_loadcnt 0x3
	v_cmp_le_i32_e64 s83, 0, v20
	v_cmp_gt_i32_e64 s84, s25, v20
	s_and_b32 s82, s82, s83
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s82, s82, s84
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v36, 0, 1, s82
	v_mbcnt_lo_u32_b32 v28, s82, 0
	s_bcnt1_i32_b32 s85, s82
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v89, s85
	s_lshl_b32 s86, s80, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v90, s86
	ds_store_b32 v90, v89 offset:15120
	v_cmp_gt_u32_e64 s82, s81, v13
	s_wait_loadcnt 0x2
	v_cmp_le_i32_e64 s83, 0, v21
	v_cmp_gt_i32_e64 s84, s25, v21
	s_and_b32 s82, s82, s83
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s82, s82, s84
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v37, 0, 1, s82
	v_mbcnt_lo_u32_b32 v29, s82, 0
	s_bcnt1_i32_b32 s85, s82
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v89, s85
	s_lshl_b32 s86, s80, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v90, s86
	ds_store_b32 v90, v89 offset:15136
	v_cmp_gt_u32_e64 s82, s81, v14
	s_wait_loadcnt 0x1
	v_cmp_le_i32_e64 s83, 0, v22
	v_cmp_gt_i32_e64 s84, s25, v22
	s_and_b32 s82, s82, s83
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s82, s82, s84
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v38, 0, 1, s82
	v_mbcnt_lo_u32_b32 v30, s82, 0
	s_bcnt1_i32_b32 s85, s82
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v89, s85
	s_lshl_b32 s86, s80, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v90, s86
	ds_store_b32 v90, v89 offset:15152
	v_cmp_gt_u32_e64 s82, s81, v15
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s83, 0, v23
	v_cmp_gt_i32_e64 s84, s25, v23
	s_and_b32 s82, s82, s83
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s82, s82, s84
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v39, 0, 1, s82
	v_mbcnt_lo_u32_b32 v31, s82, 0
	s_bcnt1_i32_b32 s85, s82
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v89, s85
	s_lshl_b32 s86, s80, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v90, s86
	ds_store_b32 v90, v89 offset:15168
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[48:51], v91 offset:15056
	ds_load_b128 v[52:55], v91 offset:15072
	ds_load_b128 v[56:59], v91 offset:15088
	ds_load_b128 v[60:63], v91 offset:15104
	ds_load_b128 v[64:67], v91 offset:15120
	ds_load_b128 v[68:71], v91 offset:15136
	ds_load_b128 v[72:75], v91 offset:15152
	ds_load_b128 v[76:79], v91 offset:15168
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v40, v88
	s_cmp_gt_u32 s80, 0
	s_cselect_b32 s82, -1, 0
	s_wait_dscnt 0x7
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v48, s82
	v_add_nc_u32_e32 v40, v40, v89
	v_add_nc_u32_e32 v88, v88, v48
	s_cmp_gt_u32 s80, 1
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v49, s82
	v_add_nc_u32_e32 v40, v40, v89
	v_add_nc_u32_e32 v88, v88, v49
	s_cmp_gt_u32 s80, 2
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v50, s82
	v_add_nc_u32_e32 v40, v40, v89
	v_add_nc_u32_e32 v88, v88, v50
	s_cmp_gt_u32 s80, 3
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v51, s82
	v_add_nc_u32_e32 v40, v40, v89
	v_add_nc_u32_e32 v88, v88, v51
	v_mov_b32_e32 v41, v88
	s_cmp_gt_u32 s80, 0
	s_cselect_b32 s82, -1, 0
	s_wait_dscnt 0x6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v52, s82
	v_add_nc_u32_e32 v41, v41, v89
	v_add_nc_u32_e32 v88, v88, v52
	s_cmp_gt_u32 s80, 1
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v53, s82
	v_add_nc_u32_e32 v41, v41, v89
	v_add_nc_u32_e32 v88, v88, v53
	s_cmp_gt_u32 s80, 2
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v54, s82
	v_add_nc_u32_e32 v41, v41, v89
	v_add_nc_u32_e32 v88, v88, v54
	s_cmp_gt_u32 s80, 3
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v55, s82
	v_add_nc_u32_e32 v41, v41, v89
	v_add_nc_u32_e32 v88, v88, v55
	v_mov_b32_e32 v42, v88
	s_cmp_gt_u32 s80, 0
	s_cselect_b32 s82, -1, 0
	s_wait_dscnt 0x5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v56, s82
	v_add_nc_u32_e32 v42, v42, v89
	v_add_nc_u32_e32 v88, v88, v56
	s_cmp_gt_u32 s80, 1
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v57, s82
	v_add_nc_u32_e32 v42, v42, v89
	v_add_nc_u32_e32 v88, v88, v57
	s_cmp_gt_u32 s80, 2
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v58, s82
	v_add_nc_u32_e32 v42, v42, v89
	v_add_nc_u32_e32 v88, v88, v58
	s_cmp_gt_u32 s80, 3
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v59, s82
	v_add_nc_u32_e32 v42, v42, v89
	v_add_nc_u32_e32 v88, v88, v59
	v_mov_b32_e32 v43, v88
	s_cmp_gt_u32 s80, 0
	s_cselect_b32 s82, -1, 0
	s_wait_dscnt 0x4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v60, s82
	v_add_nc_u32_e32 v43, v43, v89
	v_add_nc_u32_e32 v88, v88, v60
	s_cmp_gt_u32 s80, 1
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v61, s82
	v_add_nc_u32_e32 v43, v43, v89
	v_add_nc_u32_e32 v88, v88, v61
	s_cmp_gt_u32 s80, 2
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v62, s82
	v_add_nc_u32_e32 v43, v43, v89
	v_add_nc_u32_e32 v88, v88, v62
	s_cmp_gt_u32 s80, 3
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v63, s82
	v_add_nc_u32_e32 v43, v43, v89
	v_add_nc_u32_e32 v88, v88, v63
	v_mov_b32_e32 v44, v88
	s_cmp_gt_u32 s80, 0
	s_cselect_b32 s82, -1, 0
	s_wait_dscnt 0x3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v64, s82
	v_add_nc_u32_e32 v44, v44, v89
	v_add_nc_u32_e32 v88, v88, v64
	s_cmp_gt_u32 s80, 1
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v65, s82
	v_add_nc_u32_e32 v44, v44, v89
	v_add_nc_u32_e32 v88, v88, v65
	s_cmp_gt_u32 s80, 2
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v66, s82
	v_add_nc_u32_e32 v44, v44, v89
	v_add_nc_u32_e32 v88, v88, v66
	s_cmp_gt_u32 s80, 3
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v67, s82
	v_add_nc_u32_e32 v44, v44, v89
	v_add_nc_u32_e32 v88, v88, v67
	v_mov_b32_e32 v45, v88
	s_cmp_gt_u32 s80, 0
	s_cselect_b32 s82, -1, 0
	s_wait_dscnt 0x2
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v68, s82
	v_add_nc_u32_e32 v45, v45, v89
	v_add_nc_u32_e32 v88, v88, v68
	s_cmp_gt_u32 s80, 1
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v69, s82
	v_add_nc_u32_e32 v45, v45, v89
	v_add_nc_u32_e32 v88, v88, v69
	s_cmp_gt_u32 s80, 2
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v70, s82
	v_add_nc_u32_e32 v45, v45, v89
	v_add_nc_u32_e32 v88, v88, v70
	s_cmp_gt_u32 s80, 3
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v71, s82
	v_add_nc_u32_e32 v45, v45, v89
	v_add_nc_u32_e32 v88, v88, v71
	v_mov_b32_e32 v46, v88
	s_cmp_gt_u32 s80, 0
	s_cselect_b32 s82, -1, 0
	s_wait_dscnt 0x1
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v72, s82
	v_add_nc_u32_e32 v46, v46, v89
	v_add_nc_u32_e32 v88, v88, v72
	s_cmp_gt_u32 s80, 1
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v73, s82
	v_add_nc_u32_e32 v46, v46, v89
	v_add_nc_u32_e32 v88, v88, v73
	s_cmp_gt_u32 s80, 2
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v74, s82
	v_add_nc_u32_e32 v46, v46, v89
	v_add_nc_u32_e32 v88, v88, v74
	s_cmp_gt_u32 s80, 3
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v75, s82
	v_add_nc_u32_e32 v46, v46, v89
	v_add_nc_u32_e32 v88, v88, v75
	v_mov_b32_e32 v47, v88
	s_cmp_gt_u32 s80, 0
	s_cselect_b32 s82, -1, 0
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v76, s82
	v_add_nc_u32_e32 v47, v47, v89
	v_add_nc_u32_e32 v88, v88, v76
	s_cmp_gt_u32 s80, 1
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v77, s82
	v_add_nc_u32_e32 v47, v47, v89
	v_add_nc_u32_e32 v88, v88, v77
	s_cmp_gt_u32 s80, 2
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v78, s82
	v_add_nc_u32_e32 v47, v47, v89
	v_add_nc_u32_e32 v88, v88, v78
	s_cmp_gt_u32 s80, 3
	s_cselect_b32 s82, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v89, 0, v79, s82
	v_add_nc_u32_e32 v47, v47, v89
	v_add_nc_u32_e32 v88, v88, v79
	v_readfirstlane_b32 s75, v88
	v_add_nc_u32_e32 v89, v40, v24
	v_lshlrev_b32_e32 v89, 2, v89
	v_lshl_or_b32 v24, v16, 10, v8
	v_cmp_ne_u32_e64 s82, 0, v32
	s_and_saveexec_b32 s76, s82
	ds_store_b32 v89, v24 offset:10944
	s_mov_b32 exec_lo, s76
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v89, v41, v25
	v_lshlrev_b32_e32 v89, 2, v89
	v_lshl_or_b32 v25, v17, 10, v9
	v_cmp_ne_u32_e64 s82, 0, v33
	s_and_saveexec_b32 s76, s82
	ds_store_b32 v89, v25 offset:10944
	s_mov_b32 exec_lo, s76
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v89, v42, v26
	v_lshlrev_b32_e32 v89, 2, v89
	v_lshl_or_b32 v26, v18, 10, v10
	v_cmp_ne_u32_e64 s82, 0, v34
	s_and_saveexec_b32 s76, s82
	ds_store_b32 v89, v26 offset:10944
	s_mov_b32 exec_lo, s76
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v89, v43, v27
	v_lshlrev_b32_e32 v89, 2, v89
	v_lshl_or_b32 v27, v19, 10, v11
	v_cmp_ne_u32_e64 s82, 0, v35
	s_and_saveexec_b32 s76, s82
	ds_store_b32 v89, v27 offset:10944
	s_mov_b32 exec_lo, s76
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v89, v44, v28
	v_lshlrev_b32_e32 v89, 2, v89
	v_lshl_or_b32 v28, v20, 10, v12
	v_cmp_ne_u32_e64 s82, 0, v36
	s_and_saveexec_b32 s76, s82
	ds_store_b32 v89, v28 offset:10944
	s_mov_b32 exec_lo, s76
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v89, v45, v29
	v_lshlrev_b32_e32 v89, 2, v89
	v_lshl_or_b32 v29, v21, 10, v13
	v_cmp_ne_u32_e64 s82, 0, v37
	s_and_saveexec_b32 s76, s82
	ds_store_b32 v89, v29 offset:10944
	s_mov_b32 exec_lo, s76
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v89, v46, v30
	v_lshlrev_b32_e32 v89, 2, v89
	v_lshl_or_b32 v30, v22, 10, v14
	v_cmp_ne_u32_e64 s82, 0, v38
	s_and_saveexec_b32 s76, s82
	ds_store_b32 v89, v30 offset:10944
	s_mov_b32 exec_lo, s76
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v89, v47, v31
	v_lshlrev_b32_e32 v89, 2, v89
	v_lshl_or_b32 v31, v23, 10, v15
	v_cmp_ne_u32_e64 s82, 0, v39
	s_and_saveexec_b32 s76, s82
	ds_store_b32 v89, v31 offset:10944
	s_mov_b32 exec_lo, s76
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_mul_i32 s73, s75, 40
	.Lf02p_walk:
	s_cmp_ge_u32 s72, s73
	s_cbranch_scc1 .Lf02p_done
	s_mul_hi_u32 s76, s72, 0xcccccccd
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s76, s76, 5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s77, s76, 40
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s26, s72, s77
	s_lshl_b32 s76, s76, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s76
	ds_load_b32 v9, v8 offset:10944
	s_wait_dscnt 0x0
	v_readfirstlane_b32 s78, v9
	s_and_b32 s27, s78, 0x3ff
	s_lshr_b32 s29, s78, 10
	.Ldown_a8_item_entry:
	s_lshl_b32 s28, s27, 4
	s_lshl_b32 s31, s26, 6
	s_cmp_ge_i32 s31, s20
	s_cbranch_scc1 .Lf02p_item_done
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s52, s29, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_u32 s50, s8, s52
	s_add_co_ci_u32 s51, s9, 0
	s_load_b64 s[48:49], s[50:51], 0x0
	s_mul_i32 s60, s24, 0x2a8
	s_mov_b32 s36, s14
	s_mov_b32 s37, s15
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s38, s60
	s_mov_b32 s39, 0x31004000
	s_mov_b32 s32, s12
	s_mov_b32 s33, s13
	s_mov_b32 s34, -1
	s_mov_b32 s35, 0x31004000
	s_mov_b32 s44, s16
	s_mov_b32 s45, s17
	s_mul_i32 s52, s23, s20
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s52, s52, 2
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s46, s52
	s_mov_b32 s47, 0x31004000
	s_mul_i32 s56, s20, 0x140
	s_lshl_b32 s57, s20, 2
	s_add_co_i32 s58, s20, -1
	v_and_b32_e32 v53, 31, v0
	v_lshrrev_b32_e32 v106, 5, v0
	v_readfirstlane_b32 s30, v106
	v_and_b32_e32 v1, 15, v53
	v_lshrrev_b32_e32 v2, 4, v53
	s_lshl_b32 s52, s30, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s61, s31, s52
	v_lshlrev_b32_e32 v54, 3, v53
	v_add_nc_u32_e32 v55, 64, v53
	v_min_u32_e32 v55, 0x54, v55
	v_lshlrev_b32_e32 v55, 3, v55
	s_add_co_i32 s54, s30, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s54, s54, s28
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s55, s54, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v61, s55
	buffer_load_b32 v109, v61, s[32:35], null offen
	s_add_co_i32 s54, s30, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s54, s54, s28
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s55, s54, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v61, s55
	buffer_load_b32 v110, v61, s[32:35], null offen
	s_add_co_i32 s54, s30, 8
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s54, s54, s28
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s55, s54, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v61, s55
	buffer_load_b32 v111, v61, s[32:35], null offen
	s_add_co_i32 s54, s30, 12
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s54, s54, s28
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s55, s54, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v61, s55
	buffer_load_b32 v112, v61, s[32:35], null offen
	s_wait_kmcnt 0x0
	s_mov_b32 s40, s48
	s_mov_b32 s41, s49
	s_mov_b32 s42, -1
	s_mov_b32 s43, 0x31004000
	s_add_co_i32 s54, s30, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s54, s54, s28
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v62, s54
	v_cmp_gt_i32_e64 s62, s23, v62
	s_wait_loadcnt 0x3
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v109, -1, v109, s62
	v_cvt_f32_u32_e32 v67, s22
	v_rcp_iflag_f32_e32 v67, v67
	v_cvt_f32_u32_e32 v64, v109
	v_mul_f32_e32 v64, v64, v67
	v_cvt_u32_f32_e32 v64, v64
	v_mul_lo_u32 v65, v64, s22
	v_sub_nc_u32_e32 v65, v109, v65
	v_cmp_gt_i32_e64 s66, 0, v65
	v_add_nc_u32_e32 v66, -1, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v64, v64, v66, s66
	v_add_nc_u32_e32 v66, s22, v65
	v_cndmask_b32_e64 v65, v65, v66, s66
	v_cmp_le_u32_e64 s66, s22, v65
	v_add_nc_u32_e32 v66, 1, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v64, v64, v66, s66
	v_cmp_le_i32_e64 s64, 0, v109
	v_mul_u32_u24_e32 v64, 0x2a8, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v68, 0x7fffff00, v64, s64
	v_cmp_gt_u32_e64 s68, s60, v68
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v72, 0, 1, s68
	v_add_nc_u32_e32 v106, v68, v54
	buffer_load_b64 v[76:77], v106, s[36:39], null offen
	buffer_load_b64 v[78:79], v106, s[36:39], null offen offset:256
	v_add_nc_u32_e32 v107, v68, v55
	buffer_load_b64 v[80:81], v107, s[36:39], null offen
	s_add_co_i32 s54, s30, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s54, s54, s28
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v62, s54
	v_cmp_gt_i32_e64 s62, s23, v62
	s_wait_loadcnt 0x5
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v110, -1, v110, s62
	v_cvt_f32_u32_e32 v67, s22
	v_rcp_iflag_f32_e32 v67, v67
	v_cvt_f32_u32_e32 v64, v110
	v_mul_f32_e32 v64, v64, v67
	v_cvt_u32_f32_e32 v64, v64
	v_mul_lo_u32 v65, v64, s22
	v_sub_nc_u32_e32 v65, v110, v65
	v_cmp_gt_i32_e64 s66, 0, v65
	v_add_nc_u32_e32 v66, -1, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v64, v64, v66, s66
	v_add_nc_u32_e32 v66, s22, v65
	v_cndmask_b32_e64 v65, v65, v66, s66
	v_cmp_le_u32_e64 s66, s22, v65
	v_add_nc_u32_e32 v66, 1, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v64, v64, v66, s66
	v_cmp_le_i32_e64 s64, 0, v110
	v_mul_u32_u24_e32 v64, 0x2a8, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v69, 0x7fffff00, v64, s64
	v_cmp_gt_u32_e64 s68, s60, v69
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v73, 0, 1, s68
	v_add_nc_u32_e32 v106, v69, v54
	buffer_load_b64 v[84:85], v106, s[36:39], null offen
	buffer_load_b64 v[86:87], v106, s[36:39], null offen offset:256
	v_add_nc_u32_e32 v107, v69, v55
	buffer_load_b64 v[88:89], v107, s[36:39], null offen
	s_add_co_i32 s54, s30, 8
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s54, s54, s28
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v62, s54
	v_cmp_gt_i32_e64 s62, s23, v62
	s_wait_loadcnt 0x7
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v111, -1, v111, s62
	v_cvt_f32_u32_e32 v67, s22
	v_rcp_iflag_f32_e32 v67, v67
	v_cvt_f32_u32_e32 v64, v111
	v_mul_f32_e32 v64, v64, v67
	v_cvt_u32_f32_e32 v64, v64
	v_mul_lo_u32 v65, v64, s22
	v_sub_nc_u32_e32 v65, v111, v65
	v_cmp_gt_i32_e64 s66, 0, v65
	v_add_nc_u32_e32 v66, -1, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v64, v64, v66, s66
	v_add_nc_u32_e32 v66, s22, v65
	v_cndmask_b32_e64 v65, v65, v66, s66
	v_cmp_le_u32_e64 s66, s22, v65
	v_add_nc_u32_e32 v66, 1, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v64, v64, v66, s66
	v_cmp_le_i32_e64 s64, 0, v111
	v_mul_u32_u24_e32 v64, 0x2a8, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v70, 0x7fffff00, v64, s64
	v_cmp_gt_u32_e64 s68, s60, v70
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v74, 0, 1, s68
	v_add_nc_u32_e32 v106, v70, v54
	buffer_load_b64 v[92:93], v106, s[36:39], null offen
	buffer_load_b64 v[94:95], v106, s[36:39], null offen offset:256
	v_add_nc_u32_e32 v107, v70, v55
	buffer_load_b64 v[96:97], v107, s[36:39], null offen
	s_add_co_i32 s54, s30, 12
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s54, s54, s28
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v62, s54
	v_cmp_gt_i32_e64 s62, s23, v62
	s_wait_loadcnt 0x9
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v112, -1, v112, s62
	v_cvt_f32_u32_e32 v67, s22
	v_rcp_iflag_f32_e32 v67, v67
	v_cvt_f32_u32_e32 v64, v112
	v_mul_f32_e32 v64, v64, v67
	v_cvt_u32_f32_e32 v64, v64
	v_mul_lo_u32 v65, v64, s22
	v_sub_nc_u32_e32 v65, v112, v65
	v_cmp_gt_i32_e64 s66, 0, v65
	v_add_nc_u32_e32 v66, -1, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v64, v64, v66, s66
	v_add_nc_u32_e32 v66, s22, v65
	v_cndmask_b32_e64 v65, v65, v66, s66
	v_cmp_le_u32_e64 s66, s22, v65
	v_add_nc_u32_e32 v66, 1, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v64, v64, v66, s66
	v_cmp_le_i32_e64 s64, 0, v112
	v_mul_u32_u24_e32 v64, 0x2a8, v64
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v71, 0x7fffff00, v64, s64
	v_cmp_gt_u32_e64 s68, s60, v71
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v75, 0, 1, s68
	v_add_nc_u32_e32 v106, v71, v54
	buffer_load_b64 v[100:101], v106, s[36:39], null offen
	buffer_load_b64 v[102:103], v106, s[36:39], null offen offset:256
	v_add_nc_u32_e32 v107, v71, v55
	buffer_load_b64 v[104:105], v107, s[36:39], null offen
	s_add_co_i32 s53, s61, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v140, s53, v1
	v_min_u32_e32 v140, s58, v140
	v_mul_u32_u24_e32 v141, 10, v140
	v_add_nc_u32_e32 v141, s56, v141
	v_mul_u32_u24_e32 v140, 0x140, v140
	v_lshl_add_u32 v140, v2, 5, v140
	buffer_load_u16 v48, v141, s[40:43], null offen
	buffer_load_u16 v49, v141, s[40:43], null offen offset:2
	buffer_load_u16 v50, v141, s[40:43], null offen offset:4
	buffer_load_u16 v51, v141, s[40:43], null offen offset:6
	buffer_load_u16 v52, v141, s[40:43], null offen offset:8
	buffer_load_b64 v[8:9], v140, s[40:43], null offen
	buffer_load_b64 v[10:11], v140, s[40:43], null offen offset:8
	buffer_load_b64 v[12:13], v140, s[40:43], null offen offset:16
	buffer_load_b64 v[14:15], v140, s[40:43], null offen offset:24
	buffer_load_b64 v[16:17], v140, s[40:43], null offen offset:64
	buffer_load_b64 v[18:19], v140, s[40:43], null offen offset:72
	buffer_load_b64 v[20:21], v140, s[40:43], null offen offset:80
	buffer_load_b64 v[22:23], v140, s[40:43], null offen offset:88
	buffer_load_b64 v[24:25], v140, s[40:43], null offen offset:128
	buffer_load_b64 v[26:27], v140, s[40:43], null offen offset:136
	buffer_load_b64 v[28:29], v140, s[40:43], null offen offset:144
	buffer_load_b64 v[30:31], v140, s[40:43], null offen offset:152
	buffer_load_b64 v[32:33], v140, s[40:43], null offen offset:192
	buffer_load_b64 v[34:35], v140, s[40:43], null offen offset:200
	buffer_load_b64 v[36:37], v140, s[40:43], null offen offset:208
	buffer_load_b64 v[38:39], v140, s[40:43], null offen offset:216
	buffer_load_b64 v[40:41], v140, s[40:43], null offen offset:256
	buffer_load_b64 v[42:43], v140, s[40:43], null offen offset:264
	buffer_load_b64 v[44:45], v140, s[40:43], null offen offset:272
	buffer_load_b64 v[46:47], v140, s[40:43], null offen offset:280
	s_wait_loadcnt 0x14
	v_mov_b32_e32 v114, v48
	v_or_b32_e32 v114, v114, v49
	v_or_b32_e32 v114, v114, v50
	v_or_b32_e32 v114, v114, v51
	v_or_b32_e32 v114, v114, v52
	v_cmp_ne_u32_e64 s62, 0, v114
	s_cmp_lg_u32 s62, 0
	s_cselect_b32 s87, 1, 0
	s_mov_b32 s90, 0
	v_and_b32_e32 v113, 0x7f800000, v76
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 0x20001
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v78
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 0x80004
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v80
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v84
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 0x20001
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v86
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 0x80004
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v88
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v92
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 0x20001
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v94
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 0x80004
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v96
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v100
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 0x20001
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v102
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 0x80004
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	v_and_b32_e32 v113, 0x7f800000, v104
	v_cmp_eq_u32_e64 s62, 0x7f800000, v113
	s_and_b32 s62, s62, 16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s90, s90, s62
	s_cmp_eq_u32 s87, 0
	s_cselect_b32 s88, 1, 0
	s_cmp_lg_u32 s90, 0
	s_cselect_b32 s91, 2, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s88, s88, s91
	s_mul_i32 s59, s30, 0x2a8
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v106, s59, v54
	v_add_nc_u32_e32 v107, s59, v55
	s_lshl_b32 s70, s30, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v108, s70
	ds_store_b64 v106, v[76:77]
	ds_store_b64 v106, v[78:79] offset:256
	ds_store_b64 v107, v[80:81]
	ds_store_b32 v108, v72 offset:10880
	ds_store_b64 v106, v[84:85] offset:2720
	ds_store_b64 v106, v[86:87] offset:2976
	ds_store_b64 v107, v[88:89] offset:2720
	ds_store_b32 v108, v73 offset:10896
	ds_store_b64 v106, v[92:93] offset:5440
	ds_store_b64 v106, v[94:95] offset:5696
	ds_store_b64 v107, v[96:97] offset:5440
	ds_store_b32 v108, v74 offset:10912
	ds_store_b64 v106, v[100:101] offset:8160
	ds_store_b64 v106, v[102:103] offset:8416
	ds_store_b64 v107, v[104:105] offset:8160
	ds_store_b32 v108, v75 offset:10928
	v_mov_b32_e32 v114, s88
	ds_store_b32 v108, v114 offset:15184
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mul_u32_u24_e32 v53, 0x2a8, v1
	v_lshl_add_u32 v53, v2, 6, v53
	v_mul_u32_u24_e32 v54, 0x1540, v2
	v_lshlrev_b32_e32 v55, 5, v2
	v_mov_b32_e32 v139, 0
	ds_load_b128 v[128:131], v139 offset:15184
	ds_load_b128 v[120:123], v55 offset:10880
	ds_load_b128 v[124:127], v55 offset:10896
	s_wait_dscnt 0x2
	v_and_b32_e32 v137, v128, v129
	v_and_b32_e32 v138, v130, v131
	v_and_b32_e32 v137, v137, v138
	v_or_b32_e32 v136, v128, v129
	v_or_b32_e32 v138, v130, v131
	v_or_b32_e32 v136, v136, v138
	v_readfirstlane_b32 s89, v137
	v_readfirstlane_b32 s91, v136
	s_and_b32 s89, s89, 1
	s_lshr_b32 s91, s91, 1
	s_xor_b32 s91, s91, 1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s89, s89, s91
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s89, 1
	s_cbranch_scc1 .Lf02p_fast
	ds_load_b64 v[56:57], v53 offset:8
	ds_load_b64 v[72:73], v54
	ds_load_b64 v[58:59], v53 offset:16
	ds_load_b64 v[74:75], v54 offset:680
	ds_load_b64 v[60:61], v53 offset:24
	ds_load_b64 v[76:77], v54 offset:1360
	ds_load_b64 v[62:63], v53 offset:32
	ds_load_b64 v[78:79], v54 offset:2040
	ds_load_b64 v[64:65], v53 offset:40
	ds_load_b64 v[80:81], v54 offset:2720
	ds_load_b64 v[66:67], v53 offset:48
	ds_load_b64 v[82:83], v54 offset:3400
	ds_load_b64 v[68:69], v53 offset:56
	ds_load_b64 v[84:85], v54 offset:4080
	ds_load_b64 v[70:71], v53 offset:64
	ds_load_b64 v[86:87], v54 offset:4760
	s_wait_dscnt 0xe
	v_mul_lo_u32 v104, -8, v73
	s_wait_dscnt 0xc
	v_mul_lo_u32 v105, -8, v75
	s_wait_dscnt 0xa
	v_mul_lo_u32 v106, -8, v77
	s_wait_dscnt 0x8
	v_mul_lo_u32 v107, -8, v79
	s_wait_dscnt 0x6
	v_mul_lo_u32 v108, -8, v81
	s_wait_dscnt 0x4
	v_mul_lo_u32 v109, -8, v83
	s_wait_dscnt 0x2
	v_mul_lo_u32 v110, -8, v85
	s_wait_dscnt 0x0
	v_mul_lo_u32 v111, -8, v87
	s_wait_loadcnt 0x13
	v_and_b32_e32 v88, 0x0f0f0f0f, v8
	v_lshrrev_b32_e32 v89, 4, v8
	v_and_b32_e32 v89, 0x0f0f0f0f, v89
	v_and_b32_e32 v90, 0x0f0f0f0f, v9
	v_lshrrev_b32_e32 v91, 4, v9
	v_and_b32_e32 v91, 0x0f0f0f0f, v91
	s_wait_loadcnt 0x12
	v_and_b32_e32 v92, 0x0f0f0f0f, v10
	v_lshrrev_b32_e32 v93, 4, v10
	v_and_b32_e32 v93, 0x0f0f0f0f, v93
	v_and_b32_e32 v94, 0x0f0f0f0f, v11
	v_lshrrev_b32_e32 v95, 4, v11
	v_and_b32_e32 v95, 0x0f0f0f0f, v95
	s_wait_loadcnt 0x11
	v_and_b32_e32 v96, 0x0f0f0f0f, v12
	v_lshrrev_b32_e32 v97, 4, v12
	v_and_b32_e32 v97, 0x0f0f0f0f, v97
	v_and_b32_e32 v98, 0x0f0f0f0f, v13
	v_lshrrev_b32_e32 v99, 4, v13
	v_and_b32_e32 v99, 0x0f0f0f0f, v99
	s_wait_loadcnt 0x10
	v_and_b32_e32 v100, 0x0f0f0f0f, v14
	v_lshrrev_b32_e32 v101, 4, v14
	v_and_b32_e32 v101, 0x0f0f0f0f, v101
	v_and_b32_e32 v102, 0x0f0f0f0f, v15
	v_lshrrev_b32_e32 v103, 4, v15
	v_and_b32_e32 v103, 0x0f0f0f0f, v103
	v_wmma_i32_16x16x16_iu8 v[104:111], v[56:57], v[88:89], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[58:59], v[90:91], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[60:61], v[92:93], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[62:63], v[94:95], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[64:65], v[96:97], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[66:67], v[98:99], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[68:69], v[100:101], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[70:71], v[102:103], v[104:111] neg_lo:[1,0,0]
	v_cvt_f32_f16_e32 v48, v48.l
	v_mul_f32_e32 v128, v72, v48
	v_mul_f32_e32 v129, v74, v48
	v_mul_f32_e32 v130, v76, v48
	v_mul_f32_e32 v131, v78, v48
	v_mul_f32_e32 v132, v80, v48
	v_mul_f32_e32 v133, v82, v48
	v_mul_f32_e32 v134, v84, v48
	v_mul_f32_e32 v135, v86, v48
	v_cvt_f32_i32_e32 v104, v104
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v111, v111
	v_fma_f32 v112, v128, v104, v112
	v_fma_f32 v113, v129, v105, v113
	v_fma_f32 v114, v130, v106, v114
	v_fma_f32 v115, v131, v107, v115
	v_fma_f32 v116, v132, v108, v116
	v_fma_f32 v117, v133, v109, v117
	v_fma_f32 v118, v134, v110, v118
	v_fma_f32 v119, v135, v111, v119
	ds_load_b64 v[56:57], v53 offset:144
	ds_load_b64 v[72:73], v54 offset:136
	ds_load_b64 v[58:59], v53 offset:152
	ds_load_b64 v[74:75], v54 offset:816
	ds_load_b64 v[60:61], v53 offset:160
	ds_load_b64 v[76:77], v54 offset:1496
	ds_load_b64 v[62:63], v53 offset:168
	ds_load_b64 v[78:79], v54 offset:2176
	ds_load_b64 v[64:65], v53 offset:176
	ds_load_b64 v[80:81], v54 offset:2856
	ds_load_b64 v[66:67], v53 offset:184
	ds_load_b64 v[82:83], v54 offset:3536
	ds_load_b64 v[68:69], v53 offset:192
	ds_load_b64 v[84:85], v54 offset:4216
	ds_load_b64 v[70:71], v53 offset:200
	ds_load_b64 v[86:87], v54 offset:4896
	s_wait_dscnt 0xe
	v_mul_lo_u32 v104, -8, v73
	s_wait_dscnt 0xc
	v_mul_lo_u32 v105, -8, v75
	s_wait_dscnt 0xa
	v_mul_lo_u32 v106, -8, v77
	s_wait_dscnt 0x8
	v_mul_lo_u32 v107, -8, v79
	s_wait_dscnt 0x6
	v_mul_lo_u32 v108, -8, v81
	s_wait_dscnt 0x4
	v_mul_lo_u32 v109, -8, v83
	s_wait_dscnt 0x2
	v_mul_lo_u32 v110, -8, v85
	s_wait_dscnt 0x0
	v_mul_lo_u32 v111, -8, v87
	s_wait_loadcnt 0xf
	v_and_b32_e32 v88, 0x0f0f0f0f, v16
	v_lshrrev_b32_e32 v89, 4, v16
	v_and_b32_e32 v89, 0x0f0f0f0f, v89
	v_and_b32_e32 v90, 0x0f0f0f0f, v17
	v_lshrrev_b32_e32 v91, 4, v17
	v_and_b32_e32 v91, 0x0f0f0f0f, v91
	s_wait_loadcnt 0xe
	v_and_b32_e32 v92, 0x0f0f0f0f, v18
	v_lshrrev_b32_e32 v93, 4, v18
	v_and_b32_e32 v93, 0x0f0f0f0f, v93
	v_and_b32_e32 v94, 0x0f0f0f0f, v19
	v_lshrrev_b32_e32 v95, 4, v19
	v_and_b32_e32 v95, 0x0f0f0f0f, v95
	s_wait_loadcnt 0xd
	v_and_b32_e32 v96, 0x0f0f0f0f, v20
	v_lshrrev_b32_e32 v97, 4, v20
	v_and_b32_e32 v97, 0x0f0f0f0f, v97
	v_and_b32_e32 v98, 0x0f0f0f0f, v21
	v_lshrrev_b32_e32 v99, 4, v21
	v_and_b32_e32 v99, 0x0f0f0f0f, v99
	s_wait_loadcnt 0xc
	v_and_b32_e32 v100, 0x0f0f0f0f, v22
	v_lshrrev_b32_e32 v101, 4, v22
	v_and_b32_e32 v101, 0x0f0f0f0f, v101
	v_and_b32_e32 v102, 0x0f0f0f0f, v23
	v_lshrrev_b32_e32 v103, 4, v23
	v_and_b32_e32 v103, 0x0f0f0f0f, v103
	v_wmma_i32_16x16x16_iu8 v[104:111], v[56:57], v[88:89], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[58:59], v[90:91], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[60:61], v[92:93], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[62:63], v[94:95], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[64:65], v[96:97], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[66:67], v[98:99], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[68:69], v[100:101], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[70:71], v[102:103], v[104:111] neg_lo:[1,0,0]
	v_cvt_f32_f16_e32 v49, v49.l
	v_mul_f32_e32 v128, v72, v49
	v_mul_f32_e32 v129, v74, v49
	v_mul_f32_e32 v130, v76, v49
	v_mul_f32_e32 v131, v78, v49
	v_mul_f32_e32 v132, v80, v49
	v_mul_f32_e32 v133, v82, v49
	v_mul_f32_e32 v134, v84, v49
	v_mul_f32_e32 v135, v86, v49
	v_cvt_f32_i32_e32 v104, v104
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v111, v111
	v_fma_f32 v112, v128, v104, v112
	v_fma_f32 v113, v129, v105, v113
	v_fma_f32 v114, v130, v106, v114
	v_fma_f32 v115, v131, v107, v115
	v_fma_f32 v116, v132, v108, v116
	v_fma_f32 v117, v133, v109, v117
	v_fma_f32 v118, v134, v110, v118
	v_fma_f32 v119, v135, v111, v119
	ds_load_b64 v[56:57], v53 offset:280
	ds_load_b64 v[72:73], v54 offset:272
	ds_load_b64 v[58:59], v53 offset:288
	ds_load_b64 v[74:75], v54 offset:952
	ds_load_b64 v[60:61], v53 offset:296
	ds_load_b64 v[76:77], v54 offset:1632
	ds_load_b64 v[62:63], v53 offset:304
	ds_load_b64 v[78:79], v54 offset:2312
	ds_load_b64 v[64:65], v53 offset:312
	ds_load_b64 v[80:81], v54 offset:2992
	ds_load_b64 v[66:67], v53 offset:320
	ds_load_b64 v[82:83], v54 offset:3672
	ds_load_b64 v[68:69], v53 offset:328
	ds_load_b64 v[84:85], v54 offset:4352
	ds_load_b64 v[70:71], v53 offset:336
	ds_load_b64 v[86:87], v54 offset:5032
	s_wait_dscnt 0xe
	v_mul_lo_u32 v104, -8, v73
	s_wait_dscnt 0xc
	v_mul_lo_u32 v105, -8, v75
	s_wait_dscnt 0xa
	v_mul_lo_u32 v106, -8, v77
	s_wait_dscnt 0x8
	v_mul_lo_u32 v107, -8, v79
	s_wait_dscnt 0x6
	v_mul_lo_u32 v108, -8, v81
	s_wait_dscnt 0x4
	v_mul_lo_u32 v109, -8, v83
	s_wait_dscnt 0x2
	v_mul_lo_u32 v110, -8, v85
	s_wait_dscnt 0x0
	v_mul_lo_u32 v111, -8, v87
	s_wait_loadcnt 0xb
	v_and_b32_e32 v88, 0x0f0f0f0f, v24
	v_lshrrev_b32_e32 v89, 4, v24
	v_and_b32_e32 v89, 0x0f0f0f0f, v89
	v_and_b32_e32 v90, 0x0f0f0f0f, v25
	v_lshrrev_b32_e32 v91, 4, v25
	v_and_b32_e32 v91, 0x0f0f0f0f, v91
	s_wait_loadcnt 0xa
	v_and_b32_e32 v92, 0x0f0f0f0f, v26
	v_lshrrev_b32_e32 v93, 4, v26
	v_and_b32_e32 v93, 0x0f0f0f0f, v93
	v_and_b32_e32 v94, 0x0f0f0f0f, v27
	v_lshrrev_b32_e32 v95, 4, v27
	v_and_b32_e32 v95, 0x0f0f0f0f, v95
	s_wait_loadcnt 0x9
	v_and_b32_e32 v96, 0x0f0f0f0f, v28
	v_lshrrev_b32_e32 v97, 4, v28
	v_and_b32_e32 v97, 0x0f0f0f0f, v97
	v_and_b32_e32 v98, 0x0f0f0f0f, v29
	v_lshrrev_b32_e32 v99, 4, v29
	v_and_b32_e32 v99, 0x0f0f0f0f, v99
	s_wait_loadcnt 0x8
	v_and_b32_e32 v100, 0x0f0f0f0f, v30
	v_lshrrev_b32_e32 v101, 4, v30
	v_and_b32_e32 v101, 0x0f0f0f0f, v101
	v_and_b32_e32 v102, 0x0f0f0f0f, v31
	v_lshrrev_b32_e32 v103, 4, v31
	v_and_b32_e32 v103, 0x0f0f0f0f, v103
	v_wmma_i32_16x16x16_iu8 v[104:111], v[56:57], v[88:89], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[58:59], v[90:91], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[60:61], v[92:93], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[62:63], v[94:95], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[64:65], v[96:97], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[66:67], v[98:99], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[68:69], v[100:101], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[70:71], v[102:103], v[104:111] neg_lo:[1,0,0]
	v_cvt_f32_f16_e32 v50, v50.l
	v_mul_f32_e32 v128, v72, v50
	v_mul_f32_e32 v129, v74, v50
	v_mul_f32_e32 v130, v76, v50
	v_mul_f32_e32 v131, v78, v50
	v_mul_f32_e32 v132, v80, v50
	v_mul_f32_e32 v133, v82, v50
	v_mul_f32_e32 v134, v84, v50
	v_mul_f32_e32 v135, v86, v50
	v_cvt_f32_i32_e32 v104, v104
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v111, v111
	v_fma_f32 v112, v128, v104, v112
	v_fma_f32 v113, v129, v105, v113
	v_fma_f32 v114, v130, v106, v114
	v_fma_f32 v115, v131, v107, v115
	v_fma_f32 v116, v132, v108, v116
	v_fma_f32 v117, v133, v109, v117
	v_fma_f32 v118, v134, v110, v118
	v_fma_f32 v119, v135, v111, v119
	ds_load_b64 v[56:57], v53 offset:416
	ds_load_b64 v[72:73], v54 offset:408
	ds_load_b64 v[58:59], v53 offset:424
	ds_load_b64 v[74:75], v54 offset:1088
	ds_load_b64 v[60:61], v53 offset:432
	ds_load_b64 v[76:77], v54 offset:1768
	ds_load_b64 v[62:63], v53 offset:440
	ds_load_b64 v[78:79], v54 offset:2448
	ds_load_b64 v[64:65], v53 offset:448
	ds_load_b64 v[80:81], v54 offset:3128
	ds_load_b64 v[66:67], v53 offset:456
	ds_load_b64 v[82:83], v54 offset:3808
	ds_load_b64 v[68:69], v53 offset:464
	ds_load_b64 v[84:85], v54 offset:4488
	ds_load_b64 v[70:71], v53 offset:472
	ds_load_b64 v[86:87], v54 offset:5168
	s_wait_dscnt 0xe
	v_mul_lo_u32 v104, -8, v73
	s_wait_dscnt 0xc
	v_mul_lo_u32 v105, -8, v75
	s_wait_dscnt 0xa
	v_mul_lo_u32 v106, -8, v77
	s_wait_dscnt 0x8
	v_mul_lo_u32 v107, -8, v79
	s_wait_dscnt 0x6
	v_mul_lo_u32 v108, -8, v81
	s_wait_dscnt 0x4
	v_mul_lo_u32 v109, -8, v83
	s_wait_dscnt 0x2
	v_mul_lo_u32 v110, -8, v85
	s_wait_dscnt 0x0
	v_mul_lo_u32 v111, -8, v87
	s_wait_loadcnt 0x7
	v_and_b32_e32 v88, 0x0f0f0f0f, v32
	v_lshrrev_b32_e32 v89, 4, v32
	v_and_b32_e32 v89, 0x0f0f0f0f, v89
	v_and_b32_e32 v90, 0x0f0f0f0f, v33
	v_lshrrev_b32_e32 v91, 4, v33
	v_and_b32_e32 v91, 0x0f0f0f0f, v91
	s_wait_loadcnt 0x6
	v_and_b32_e32 v92, 0x0f0f0f0f, v34
	v_lshrrev_b32_e32 v93, 4, v34
	v_and_b32_e32 v93, 0x0f0f0f0f, v93
	v_and_b32_e32 v94, 0x0f0f0f0f, v35
	v_lshrrev_b32_e32 v95, 4, v35
	v_and_b32_e32 v95, 0x0f0f0f0f, v95
	s_wait_loadcnt 0x5
	v_and_b32_e32 v96, 0x0f0f0f0f, v36
	v_lshrrev_b32_e32 v97, 4, v36
	v_and_b32_e32 v97, 0x0f0f0f0f, v97
	v_and_b32_e32 v98, 0x0f0f0f0f, v37
	v_lshrrev_b32_e32 v99, 4, v37
	v_and_b32_e32 v99, 0x0f0f0f0f, v99
	s_wait_loadcnt 0x4
	v_and_b32_e32 v100, 0x0f0f0f0f, v38
	v_lshrrev_b32_e32 v101, 4, v38
	v_and_b32_e32 v101, 0x0f0f0f0f, v101
	v_and_b32_e32 v102, 0x0f0f0f0f, v39
	v_lshrrev_b32_e32 v103, 4, v39
	v_and_b32_e32 v103, 0x0f0f0f0f, v103
	v_wmma_i32_16x16x16_iu8 v[104:111], v[56:57], v[88:89], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[58:59], v[90:91], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[60:61], v[92:93], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[62:63], v[94:95], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[64:65], v[96:97], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[66:67], v[98:99], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[68:69], v[100:101], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[70:71], v[102:103], v[104:111] neg_lo:[1,0,0]
	v_cvt_f32_f16_e32 v51, v51.l
	v_mul_f32_e32 v128, v72, v51
	v_mul_f32_e32 v129, v74, v51
	v_mul_f32_e32 v130, v76, v51
	v_mul_f32_e32 v131, v78, v51
	v_mul_f32_e32 v132, v80, v51
	v_mul_f32_e32 v133, v82, v51
	v_mul_f32_e32 v134, v84, v51
	v_mul_f32_e32 v135, v86, v51
	v_cvt_f32_i32_e32 v104, v104
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v111, v111
	v_fma_f32 v112, v128, v104, v112
	v_fma_f32 v113, v129, v105, v113
	v_fma_f32 v114, v130, v106, v114
	v_fma_f32 v115, v131, v107, v115
	v_fma_f32 v116, v132, v108, v116
	v_fma_f32 v117, v133, v109, v117
	v_fma_f32 v118, v134, v110, v118
	v_fma_f32 v119, v135, v111, v119
	ds_load_b64 v[56:57], v53 offset:552
	ds_load_b64 v[72:73], v54 offset:544
	ds_load_b64 v[58:59], v53 offset:560
	ds_load_b64 v[74:75], v54 offset:1224
	ds_load_b64 v[60:61], v53 offset:568
	ds_load_b64 v[76:77], v54 offset:1904
	ds_load_b64 v[62:63], v53 offset:576
	ds_load_b64 v[78:79], v54 offset:2584
	ds_load_b64 v[64:65], v53 offset:584
	ds_load_b64 v[80:81], v54 offset:3264
	ds_load_b64 v[66:67], v53 offset:592
	ds_load_b64 v[82:83], v54 offset:3944
	ds_load_b64 v[68:69], v53 offset:600
	ds_load_b64 v[84:85], v54 offset:4624
	ds_load_b64 v[70:71], v53 offset:608
	ds_load_b64 v[86:87], v54 offset:5304
	s_wait_dscnt 0xe
	v_mul_lo_u32 v104, -8, v73
	s_wait_dscnt 0xc
	v_mul_lo_u32 v105, -8, v75
	s_wait_dscnt 0xa
	v_mul_lo_u32 v106, -8, v77
	s_wait_dscnt 0x8
	v_mul_lo_u32 v107, -8, v79
	s_wait_dscnt 0x6
	v_mul_lo_u32 v108, -8, v81
	s_wait_dscnt 0x4
	v_mul_lo_u32 v109, -8, v83
	s_wait_dscnt 0x2
	v_mul_lo_u32 v110, -8, v85
	s_wait_dscnt 0x0
	v_mul_lo_u32 v111, -8, v87
	s_wait_loadcnt 0x3
	v_and_b32_e32 v88, 0x0f0f0f0f, v40
	v_lshrrev_b32_e32 v89, 4, v40
	v_and_b32_e32 v89, 0x0f0f0f0f, v89
	v_and_b32_e32 v90, 0x0f0f0f0f, v41
	v_lshrrev_b32_e32 v91, 4, v41
	v_and_b32_e32 v91, 0x0f0f0f0f, v91
	s_wait_loadcnt 0x2
	v_and_b32_e32 v92, 0x0f0f0f0f, v42
	v_lshrrev_b32_e32 v93, 4, v42
	v_and_b32_e32 v93, 0x0f0f0f0f, v93
	v_and_b32_e32 v94, 0x0f0f0f0f, v43
	v_lshrrev_b32_e32 v95, 4, v43
	v_and_b32_e32 v95, 0x0f0f0f0f, v95
	s_wait_loadcnt 0x1
	v_and_b32_e32 v96, 0x0f0f0f0f, v44
	v_lshrrev_b32_e32 v97, 4, v44
	v_and_b32_e32 v97, 0x0f0f0f0f, v97
	v_and_b32_e32 v98, 0x0f0f0f0f, v45
	v_lshrrev_b32_e32 v99, 4, v45
	v_and_b32_e32 v99, 0x0f0f0f0f, v99
	s_wait_loadcnt 0x0
	v_and_b32_e32 v100, 0x0f0f0f0f, v46
	v_lshrrev_b32_e32 v101, 4, v46
	v_and_b32_e32 v101, 0x0f0f0f0f, v101
	v_and_b32_e32 v102, 0x0f0f0f0f, v47
	v_lshrrev_b32_e32 v103, 4, v47
	v_and_b32_e32 v103, 0x0f0f0f0f, v103
	v_wmma_i32_16x16x16_iu8 v[104:111], v[56:57], v[88:89], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[58:59], v[90:91], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[60:61], v[92:93], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[62:63], v[94:95], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[64:65], v[96:97], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[66:67], v[98:99], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[68:69], v[100:101], v[104:111] neg_lo:[1,0,0]
	v_wmma_i32_16x16x16_iu8 v[104:111], v[70:71], v[102:103], v[104:111] neg_lo:[1,0,0]
	v_cvt_f32_f16_e32 v52, v52.l
	v_mul_f32_e32 v128, v72, v52
	v_mul_f32_e32 v129, v74, v52
	v_mul_f32_e32 v130, v76, v52
	v_mul_f32_e32 v131, v78, v52
	v_mul_f32_e32 v132, v80, v52
	v_mul_f32_e32 v133, v82, v52
	v_mul_f32_e32 v134, v84, v52
	v_mul_f32_e32 v135, v86, v52
	v_cvt_f32_i32_e32 v104, v104
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v111, v111
	v_fma_f32 v112, v128, v104, v112
	v_fma_f32 v113, v129, v105, v113
	v_fma_f32 v114, v130, v106, v114
	v_fma_f32 v115, v131, v107, v115
	v_fma_f32 v116, v132, v108, v116
	v_fma_f32 v117, v133, v109, v117
	v_fma_f32 v118, v134, v110, v118
	v_fma_f32 v119, v135, v111, v119
	s_add_co_i32 s53, s61, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v138, s53, v1
	v_lshl_add_u32 v139, v2, 3, s28
	v_mul_lo_u32 v139, v139, s20
	v_add_nc_u32_e32 v139, v139, v138
	v_lshlrev_b32_e32 v139, 2, v139
	v_cmp_gt_i32_e64 s62, s20, v138
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v136, 0x7fffff00, v139, s62
	v_cmp_ne_u32_e64 s68, 0, v120
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v104, 0x7fffff00, v136, s68
	buffer_store_b32 v112, v104, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v121
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v105, 0x7fffff00, v136, s68
	buffer_store_b32 v113, v105, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v122
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v106, 0x7fffff00, v136, s68
	buffer_store_b32 v114, v106, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v123
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v107, 0x7fffff00, v136, s68
	buffer_store_b32 v115, v107, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v124
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v108, 0x7fffff00, v136, s68
	buffer_store_b32 v116, v108, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v125
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v109, 0x7fffff00, v136, s68
	buffer_store_b32 v117, v109, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v126
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v110, 0x7fffff00, v136, s68
	buffer_store_b32 v118, v110, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v127
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v111, 0x7fffff00, v136, s68
	buffer_store_b32 v119, v111, s[44:47], null offen
	s_wait_storecnt 0x0
	s_branch .Lf02p_end
	.Lf02p_fast:
	s_wait_loadcnt 0x0
	s_add_co_i32 s53, s61, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v138, s53, v1
	v_lshl_add_u32 v139, v2, 3, s28
	v_mul_lo_u32 v139, v139, s20
	v_add_nc_u32_e32 v139, v139, v138
	v_lshlrev_b32_e32 v139, 2, v139
	v_cmp_gt_i32_e64 s62, s20, v138
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v136, 0x7fffff00, v139, s62
	v_cmp_ne_u32_e64 s68, 0, v120
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v104, 0x7fffff00, v136, s68
	buffer_store_b32 v112, v104, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v121
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v105, 0x7fffff00, v136, s68
	buffer_store_b32 v113, v105, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v122
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v106, 0x7fffff00, v136, s68
	buffer_store_b32 v114, v106, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v123
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v107, 0x7fffff00, v136, s68
	buffer_store_b32 v115, v107, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v124
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v108, 0x7fffff00, v136, s68
	buffer_store_b32 v116, v108, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v125
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v109, 0x7fffff00, v136, s68
	buffer_store_b32 v117, v109, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v126
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v110, 0x7fffff00, v136, s68
	buffer_store_b32 v118, v110, s[44:47], null offen
	v_add_nc_u32_e32 v136, s57, v136
	v_cmp_ne_u32_e64 s68, 0, v127
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v111, 0x7fffff00, v136, s68
	buffer_store_b32 v119, v111, s[44:47], null offen
	.Lf02p_end:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lf02p_item_done:
	s_wait_storecnt 0x0
	.Ldown_a8_item_exit:
	s_add_co_i32 s72, s72, s74
	s_branch .Lf02p_walk
	.Lf02p_done:
	s_endpgm
.Lqwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201_end:
.size qwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201, .Lqwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201_end-qwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel qwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201
	.amdhsa_group_segment_fixed_size 15200
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 68
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
	.amdhsa_next_free_vgpr 142
	.amdhsa_next_free_sgpr 104
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lqwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201_end-qwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201)<<4)&4080)>>4
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
        .name: expert_ptrs
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: tile_ids
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: sorted_slots
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: xq
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 40
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 44
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_div
        .offset: 48
        .size: 4
        .value_kind: by_value
      - 
        .name: grouped_rows
        .offset: 52
        .size: 4
        .value_kind: by_value
      - 
        .name: x_src_rows
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: n_exp
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: n_blocks
        .offset: 64
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 15200
    .kernarg_segment_align: 8
    .kernarg_segment_size: 68
    .max_flat_workgroup_size: 128
    .name: qwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 104
    .sgpr_spill_count: 0
    .symbol: qwen4_moe_down_a8_grouped_planar_nodz_pm_persist_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 142
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
