.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
.amdhsa_code_object_version 6
.text
.protected gemm_mq6g256v2_residual_pm_gfx1100_w4
.globl gemm_mq6g256v2_residual_pm_gfx1100_w4
.p2align 8
.type gemm_mq6g256v2_residual_pm_gfx1100_w4,@function
gemm_mq6g256v2_residual_pm_gfx1100_w4:
	s_load_b256 s[16:23], s[0:1], null
	s_load_b32 s24, s[0:1], 0x20
	v_and_b32_e32 v146, 15, v0
	v_bfe_u32 v147, v0, 4, 1
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v6, 0x50, v146
	v_lshrrev_b32_e32 v149, 2, v0
	v_and_b32_e32 v150, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v150, 4, v150
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v5, v149, 0x50, v150
	v_cmp_ne_u32_e64 s54, 0, v147
	v_mov_b32_e32 v151, 0xc0c0100
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v151, 0xc0c0302, s54
	v_mov_b32_e32 v151, 0xc020201
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v2, v151, 0xc040403, s54
	v_mov_b32_e32 v151, 0xc0c0403
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v222, v151, 0xc0c0605, s54
	v_mov_b32_e32 v151, 0xc050504
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v223, v151, 0xc070706, s54
	s_mov_b32 s50, 0x76543210
	s_mov_b32 s80, 0x3f003f
	s_mov_b32 s81, 0xfc03f0
	s_mov_b32 s82, 0xe400e400
	s_mov_b32 s83, 0xdc00d400
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
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
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	s_waitcnt lgkmcnt(0)
	s_min_i32 s35, s22, s23
	s_min_i32 s35, s35, s24
	s_cmp_lt_i32 s35, 1
	s_cbranch_scc1 .Lmq6g11w4y_end
	s_mov_b32 s28, s14
	s_mov_b32 s29, s15
	s_lshr_b32 s25, s23, 8
	s_lshr_b32 s26, s23, 5
	s_add_i32 s26, s26, -1
	s_add_i32 s27, s25, -1
	s_lshl_b32 s43, s23, 1
	s_mul_i32 s36, s24, s43
	s_mov_b32 s4, s16
	s_and_b32 s5, s17, 0xffff
	s_brev_b32 s6, -2
	s_mov_b32 s7, 0x31004000
	s_mov_b32 s8, s18
	s_and_b32 s9, s19, 0xffff
	s_mov_b32 s10, s36
	s_mov_b32 s11, 0x31004000
	s_mov_b32 s56, s20
	s_and_b32 s57, s21, 0xffff
	s_brev_b32 s58, -2
	s_mov_b32 s59, 0x31004000
	v_lshrrev_b32_e32 v149, 1, v0
	s_mul_i32 s35, s28, 64
	s_delay_alu instid0(VALU_DEP_1)
	v_and_or_b32 v149, v149, 48, s35
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v9, v147, 3, v149
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v149, v149, v146
	s_add_i32 s35, s22, -1
	s_mul_i32 s36, s25, 0xc8
	s_delay_alu instid0(VALU_DEP_1)
	v_min_i32_e32 v149, s35, v149
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v4, v149, s36
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v3, v147, 2, v4
	s_lshl_b32 s35, s29, 8
	v_add_nc_u32_e32 v7, s35, v146
	v_lshrrev_b32_e32 v148, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v148, s35, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v10, 0, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v10, v10, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v10, v10, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v11, 32, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v11, v11, v150
	v_add_nc_u32_e32 v12, 64, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v12, v12, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v12, v12, v150
	v_add_nc_u32_e32 v13, 0x60, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v13, v13, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v13, v13, v150
	v_add_nc_u32_e32 v14, 0x80, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v14, v14, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v14, v14, v150
	v_add_nc_u32_e32 v15, 0xa0, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v15, v15, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v15, v15, v150
	v_add_nc_u32_e32 v16, 0xc0, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v16, v16, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v16, v16, v150
	v_add_nc_u32_e32 v17, 0xe0, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v17, v17, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v17, v17, v150
	v_mul_lo_u32 v8, v7, s22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 2, v8
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmq6g11w4y_epi
	s_mov_b32 s30, 0
	s_mov_b32 s31, 0
	s_mov_b32 s34, 0
	s_mov_b32 s33, 0
	buffer_load_b64 v[210:211], v4, s[4:7], s31 offen
	buffer_load_b64 v[194:195], v3, s[4:7], s31 offen offset:8
	buffer_load_b64 v[196:197], v3, s[4:7], s31 offen offset:20
	buffer_load_b128 v[224:227], v10, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v11, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v12, s[8:11], s33 offen
	buffer_load_b128 v[236:239], v13, s[8:11], s33 offen
	buffer_load_b128 v[240:243], v14, s[8:11], s33 offen
	buffer_load_b128 v[244:247], v15, s[8:11], s33 offen
	buffer_load_b128 v[248:251], v16, s[8:11], s33 offen
	buffer_load_b128 v[252:255], v17, s[8:11], s33 offen
	s_waitcnt vmcnt(7)
	ds_store_b128 v5, v[224:227]
	s_waitcnt vmcnt(6)
	ds_store_b128 v5, v[228:231] offset:2560
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[232:235] offset:5120
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[236:239] offset:7680
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[240:243] offset:10240
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[244:247] offset:12800
	s_waitcnt vmcnt(1)
	ds_store_b128 v5, v[248:251] offset:15360
	s_waitcnt vmcnt(0)
	ds_store_b128 v5, v[252:255] offset:17920
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b64 v[198:199], v3, s[4:7], s31 offen offset:32
	buffer_load_b64 v[200:201], v3, s[4:7], s31 offen offset:44
	s_min_u32 s33, s26, 1
	s_lshl_b32 s33, s33, 6
	buffer_load_b128 v[224:227], v10, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v11, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v12, s[8:11], s33 offen
	buffer_load_b128 v[236:239], v13, s[8:11], s33 offen
	buffer_load_b128 v[240:243], v14, s[8:11], s33 offen
	buffer_load_b128 v[244:247], v15, s[8:11], s33 offen
	buffer_load_b128 v[248:251], v16, s[8:11], s33 offen
	buffer_load_b128 v[252:255], v17, s[8:11], s33 offen
	v_perm_b32 v214, v195, v194, v1
	v_perm_b32 v215, v195, v194, v2
	v_perm_b32 v216, v195, v194, v222
	v_perm_b32 v217, v195, v194, v223
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_or_b32 v216, v216, 10, v216
	s_delay_alu instid0(VALU_DEP_2)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f16 v214, v214, s82
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f16 v215, v215, s83
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f16 v216, v216, s82
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f16 v217, v217, s83
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_fma_f16 v214, v210, v214, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_fma_f16 v215, v210, v215, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_fma_f16 v216, v210, v216, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_fma_f16 v217, v210, v217, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_delay_alu instid0(VALU_DEP_4)
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4)
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4)
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4)
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v178, v214, v218, s54
	v_cndmask_b32_e64 v182, v218, v214, s54
	v_cndmask_b32_e64 v179, v215, v219, s54
	v_cndmask_b32_e64 v183, v219, v215, s54
	v_cndmask_b32_e64 v180, v216, v220, s54
	v_cndmask_b32_e64 v184, v220, v216, s54
	v_cndmask_b32_e64 v181, v217, v221, s54
	v_cndmask_b32_e64 v185, v221, v217, s54
	ds_load_b128 v[146:149], v6
	ds_load_b128 v[150:153], v6 offset:16
	ds_load_b128 v[154:157], v6 offset:1280
	ds_load_b128 v[158:161], v6 offset:1296
	ds_load_b128 v[162:165], v6 offset:2560
	ds_load_b128 v[166:169], v6 offset:2576
	ds_load_b128 v[170:173], v6 offset:3840
	ds_load_b128 v[174:177], v6 offset:3856
	.Lmq6g11w4y_loop:
	s_add_i32 s32, s30, 1
	s_min_u32 s32, s32, s27
	s_mul_i32 s32, s32, 0xc8
	s_add_i32 s33, s34, 2
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[178:185], v[146:153], v[18:25]
	buffer_load_b64 v[202:203], v3, s[4:7], s31 offen offset:56
	buffer_load_b64 v[204:205], v3, s[4:7], s31 offen offset:68
	ds_load_b128 v[146:149], v6 offset:5120
	ds_load_b128 v[150:153], v6 offset:5136
	v_perm_b32 v214, v197, v196, v1
	v_perm_b32 v215, v197, v196, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[178:185], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:6400
	ds_load_b128 v[158:161], v6 offset:6416
	v_perm_b32 v216, v197, v196, v222
	v_perm_b32 v217, v197, v196, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[178:185], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:7680
	ds_load_b128 v[166:169], v6 offset:7696
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[42:49], v[178:185], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:8960
	ds_load_b128 v[174:177], v6 offset:8976
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[50:57], v[178:185], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:10240
	ds_load_b128 v[150:153], v6 offset:10256
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[58:65], v[178:185], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:11520
	ds_load_b128 v[158:161], v6 offset:11536
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[66:73], v[178:185], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:12800
	ds_load_b128 v[166:169], v6 offset:12816
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[74:81], v[178:185], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:14080
	ds_load_b128 v[174:177], v6 offset:14096
	v_pk_fma_f16 v214, v210, v214, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v210, v215, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[82:89], v[178:185], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:15360
	ds_load_b128 v[150:153], v6 offset:15376
	v_pk_fma_f16 v216, v210, v216, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v210, v217, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[90:97], v[178:185], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:16640
	ds_load_b128 v[158:161], v6 offset:16656
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[98:105], v[178:185], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:17920
	ds_load_b128 v[166:169], v6 offset:17936
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[106:113], v[178:185], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:19200
	ds_load_b128 v[174:177], v6 offset:19216
	v_cndmask_b32_e64 v186, v214, v218, s54
	v_cndmask_b32_e64 v190, v218, v214, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[114:121], v[178:185], v[146:153], v[114:121]
	ds_load_b128 v[146:149], v6 offset:32
	ds_load_b128 v[150:153], v6 offset:48
	v_cndmask_b32_e64 v187, v215, v219, s54
	v_cndmask_b32_e64 v191, v219, v215, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[122:129], v[178:185], v[154:161], v[122:129]
	ds_load_b128 v[154:157], v6 offset:1312
	ds_load_b128 v[158:161], v6 offset:1328
	v_cndmask_b32_e64 v188, v216, v220, s54
	v_cndmask_b32_e64 v192, v220, v216, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[130:137], v[178:185], v[162:169], v[130:137]
	ds_load_b128 v[162:165], v6 offset:2592
	ds_load_b128 v[166:169], v6 offset:2608
	v_cndmask_b32_e64 v189, v217, v221, s54
	v_cndmask_b32_e64 v193, v221, v217, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[138:145], v[178:185], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:3872
	ds_load_b128 v[174:177], v6 offset:3888
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[186:193], v[146:153], v[18:25]
	ds_load_b128 v[146:149], v6 offset:5152
	ds_load_b128 v[150:153], v6 offset:5168
	s_waitcnt vmcnt(11)
	v_perm_b32 v214, v199, v198, v1
	v_perm_b32 v215, v199, v198, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[186:193], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:6432
	ds_load_b128 v[158:161], v6 offset:6448
	v_perm_b32 v216, v199, v198, v222
	v_perm_b32 v217, v199, v198, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[186:193], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:7712
	ds_load_b128 v[166:169], v6 offset:7728
	s_waitcnt vmcnt(9)
	ds_store_b128 v5, v[224:227] offset:20480
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[42:49], v[186:193], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:8992
	ds_load_b128 v[174:177], v6 offset:9008
	s_waitcnt vmcnt(8)
	ds_store_b128 v5, v[228:231] offset:23040
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[50:57], v[186:193], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:10272
	ds_load_b128 v[150:153], v6 offset:10288
	s_waitcnt vmcnt(7)
	ds_store_b128 v5, v[232:235] offset:25600
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[58:65], v[186:193], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:11552
	ds_load_b128 v[158:161], v6 offset:11568
	s_waitcnt vmcnt(6)
	ds_store_b128 v5, v[236:239] offset:28160
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[66:73], v[186:193], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:12832
	ds_load_b128 v[166:169], v6 offset:12848
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[240:243] offset:30720
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[74:81], v[186:193], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:14112
	ds_load_b128 v[174:177], v6 offset:14128
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[244:247] offset:33280
	v_pk_fma_f16 v214, v210, v214, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v210, v215, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[82:89], v[186:193], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:15392
	ds_load_b128 v[150:153], v6 offset:15408
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[248:251] offset:35840
	v_pk_fma_f16 v216, v210, v216, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v210, v217, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[90:97], v[186:193], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:16672
	ds_load_b128 v[158:161], v6 offset:16688
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[252:255] offset:38400
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[98:105], v[186:193], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:17952
	ds_load_b128 v[166:169], v6 offset:17968
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[106:113], v[186:193], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:19232
	ds_load_b128 v[174:177], v6 offset:19248
	v_cndmask_b32_e64 v178, v214, v218, s54
	v_cndmask_b32_e64 v182, v218, v214, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[224:227], v10, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v11, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v12, s[8:11], s33 offen
	buffer_load_b128 v[236:239], v13, s[8:11], s33 offen
	buffer_load_b128 v[240:243], v14, s[8:11], s33 offen
	buffer_load_b128 v[244:247], v15, s[8:11], s33 offen
	buffer_load_b128 v[248:251], v16, s[8:11], s33 offen
	buffer_load_b128 v[252:255], v17, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[114:121], v[186:193], v[146:153], v[114:121]
	v_cndmask_b32_e64 v179, v215, v219, s54
	v_cndmask_b32_e64 v183, v219, v215, s54
	ds_load_b128 v[146:149], v6 offset:20480
	ds_load_b128 v[150:153], v6 offset:20496
	v_wmma_f32_16x16x16_f16 v[122:129], v[186:193], v[154:161], v[122:129]
	v_cndmask_b32_e64 v180, v216, v220, s54
	v_cndmask_b32_e64 v184, v220, v216, s54
	ds_load_b128 v[154:157], v6 offset:21760
	ds_load_b128 v[158:161], v6 offset:21776
	v_wmma_f32_16x16x16_f16 v[130:137], v[186:193], v[162:169], v[130:137]
	v_cndmask_b32_e64 v181, v217, v221, s54
	v_cndmask_b32_e64 v185, v221, v217, s54
	ds_load_b128 v[162:165], v6 offset:23040
	ds_load_b128 v[166:169], v6 offset:23056
	v_wmma_f32_16x16x16_f16 v[138:145], v[186:193], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:24320
	ds_load_b128 v[174:177], v6 offset:24336
	s_add_i32 s33, s34, 3
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[178:185], v[146:153], v[18:25]
	buffer_load_b64 v[206:207], v3, s[4:7], s31 offen offset:80
	buffer_load_b64 v[208:209], v3, s[4:7], s31 offen offset:92
	ds_load_b128 v[146:149], v6 offset:25600
	ds_load_b128 v[150:153], v6 offset:25616
	v_perm_b32 v214, v201, v200, v1
	v_perm_b32 v215, v201, v200, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[178:185], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:26880
	ds_load_b128 v[158:161], v6 offset:26896
	v_perm_b32 v216, v201, v200, v222
	v_perm_b32 v217, v201, v200, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[178:185], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:28160
	ds_load_b128 v[166:169], v6 offset:28176
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[42:49], v[178:185], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:29440
	ds_load_b128 v[174:177], v6 offset:29456
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[50:57], v[178:185], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:30720
	ds_load_b128 v[150:153], v6 offset:30736
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[58:65], v[178:185], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:32000
	ds_load_b128 v[158:161], v6 offset:32016
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[66:73], v[178:185], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:33280
	ds_load_b128 v[166:169], v6 offset:33296
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[74:81], v[178:185], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:34560
	ds_load_b128 v[174:177], v6 offset:34576
	v_pk_fma_f16 v214, v210, v214, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v210, v215, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[82:89], v[178:185], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:35840
	ds_load_b128 v[150:153], v6 offset:35856
	v_pk_fma_f16 v216, v210, v216, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v210, v217, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[90:97], v[178:185], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:37120
	ds_load_b128 v[158:161], v6 offset:37136
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[98:105], v[178:185], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:38400
	ds_load_b128 v[166:169], v6 offset:38416
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[106:113], v[178:185], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:39680
	ds_load_b128 v[174:177], v6 offset:39696
	v_cndmask_b32_e64 v186, v214, v218, s54
	v_cndmask_b32_e64 v190, v218, v214, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[114:121], v[178:185], v[146:153], v[114:121]
	ds_load_b128 v[146:149], v6 offset:20512
	ds_load_b128 v[150:153], v6 offset:20528
	v_cndmask_b32_e64 v187, v215, v219, s54
	v_cndmask_b32_e64 v191, v219, v215, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[122:129], v[178:185], v[154:161], v[122:129]
	ds_load_b128 v[154:157], v6 offset:21792
	ds_load_b128 v[158:161], v6 offset:21808
	v_cndmask_b32_e64 v188, v216, v220, s54
	v_cndmask_b32_e64 v192, v220, v216, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[130:137], v[178:185], v[162:169], v[130:137]
	ds_load_b128 v[162:165], v6 offset:23072
	ds_load_b128 v[166:169], v6 offset:23088
	v_cndmask_b32_e64 v189, v217, v221, s54
	v_cndmask_b32_e64 v193, v221, v217, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[138:145], v[178:185], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:24352
	ds_load_b128 v[174:177], v6 offset:24368
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[186:193], v[146:153], v[18:25]
	ds_load_b128 v[146:149], v6 offset:25632
	ds_load_b128 v[150:153], v6 offset:25648
	s_waitcnt vmcnt(11)
	v_perm_b32 v214, v203, v202, v1
	v_perm_b32 v215, v203, v202, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[186:193], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:26912
	ds_load_b128 v[158:161], v6 offset:26928
	v_perm_b32 v216, v203, v202, v222
	v_perm_b32 v217, v203, v202, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[186:193], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:28192
	ds_load_b128 v[166:169], v6 offset:28208
	s_waitcnt vmcnt(9)
	ds_store_b128 v5, v[224:227]
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[42:49], v[186:193], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:29472
	ds_load_b128 v[174:177], v6 offset:29488
	s_waitcnt vmcnt(8)
	ds_store_b128 v5, v[228:231] offset:2560
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[50:57], v[186:193], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:30752
	ds_load_b128 v[150:153], v6 offset:30768
	s_waitcnt vmcnt(7)
	ds_store_b128 v5, v[232:235] offset:5120
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[58:65], v[186:193], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:32032
	ds_load_b128 v[158:161], v6 offset:32048
	s_waitcnt vmcnt(6)
	ds_store_b128 v5, v[236:239] offset:7680
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[66:73], v[186:193], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:33312
	ds_load_b128 v[166:169], v6 offset:33328
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[240:243] offset:10240
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[74:81], v[186:193], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:34592
	ds_load_b128 v[174:177], v6 offset:34608
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[244:247] offset:12800
	v_pk_fma_f16 v214, v210, v214, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v210, v215, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[82:89], v[186:193], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:35872
	ds_load_b128 v[150:153], v6 offset:35888
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[248:251] offset:15360
	v_pk_fma_f16 v216, v210, v216, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v210, v217, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[90:97], v[186:193], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:37152
	ds_load_b128 v[158:161], v6 offset:37168
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[252:255] offset:17920
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[98:105], v[186:193], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:38432
	ds_load_b128 v[166:169], v6 offset:38448
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[106:113], v[186:193], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:39712
	ds_load_b128 v[174:177], v6 offset:39728
	v_cndmask_b32_e64 v178, v214, v218, s54
	v_cndmask_b32_e64 v182, v218, v214, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[224:227], v10, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v11, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v12, s[8:11], s33 offen
	buffer_load_b128 v[236:239], v13, s[8:11], s33 offen
	buffer_load_b128 v[240:243], v14, s[8:11], s33 offen
	buffer_load_b128 v[244:247], v15, s[8:11], s33 offen
	buffer_load_b128 v[248:251], v16, s[8:11], s33 offen
	buffer_load_b128 v[252:255], v17, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[114:121], v[186:193], v[146:153], v[114:121]
	v_cndmask_b32_e64 v179, v215, v219, s54
	v_cndmask_b32_e64 v183, v219, v215, s54
	ds_load_b128 v[146:149], v6
	ds_load_b128 v[150:153], v6 offset:16
	v_wmma_f32_16x16x16_f16 v[122:129], v[186:193], v[154:161], v[122:129]
	v_cndmask_b32_e64 v180, v216, v220, s54
	v_cndmask_b32_e64 v184, v220, v216, s54
	ds_load_b128 v[154:157], v6 offset:1280
	ds_load_b128 v[158:161], v6 offset:1296
	v_wmma_f32_16x16x16_f16 v[130:137], v[186:193], v[162:169], v[130:137]
	v_cndmask_b32_e64 v181, v217, v221, s54
	v_cndmask_b32_e64 v185, v221, v217, s54
	ds_load_b128 v[162:165], v6 offset:2560
	ds_load_b128 v[166:169], v6 offset:2576
	v_wmma_f32_16x16x16_f16 v[138:145], v[186:193], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:3840
	ds_load_b128 v[174:177], v6 offset:3856
	s_add_i32 s33, s34, 4
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[178:185], v[146:153], v[18:25]
	buffer_load_b64 v[194:195], v3, s[4:7], s31 offen offset:104
	buffer_load_b64 v[196:197], v3, s[4:7], s31 offen offset:116
	ds_load_b128 v[146:149], v6 offset:5120
	ds_load_b128 v[150:153], v6 offset:5136
	v_perm_b32 v214, v205, v204, v1
	v_perm_b32 v215, v205, v204, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[178:185], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:6400
	ds_load_b128 v[158:161], v6 offset:6416
	v_perm_b32 v216, v205, v204, v222
	v_perm_b32 v217, v205, v204, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[178:185], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:7680
	ds_load_b128 v[166:169], v6 offset:7696
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[42:49], v[178:185], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:8960
	ds_load_b128 v[174:177], v6 offset:8976
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[50:57], v[178:185], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:10240
	ds_load_b128 v[150:153], v6 offset:10256
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[58:65], v[178:185], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:11520
	ds_load_b128 v[158:161], v6 offset:11536
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[66:73], v[178:185], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:12800
	ds_load_b128 v[166:169], v6 offset:12816
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[74:81], v[178:185], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:14080
	ds_load_b128 v[174:177], v6 offset:14096
	v_pk_fma_f16 v214, v210, v214, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v210, v215, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[82:89], v[178:185], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:15360
	ds_load_b128 v[150:153], v6 offset:15376
	v_pk_fma_f16 v216, v210, v216, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v210, v217, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[90:97], v[178:185], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:16640
	ds_load_b128 v[158:161], v6 offset:16656
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[98:105], v[178:185], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:17920
	ds_load_b128 v[166:169], v6 offset:17936
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[106:113], v[178:185], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:19200
	ds_load_b128 v[174:177], v6 offset:19216
	v_cndmask_b32_e64 v186, v214, v218, s54
	v_cndmask_b32_e64 v190, v218, v214, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[114:121], v[178:185], v[146:153], v[114:121]
	ds_load_b128 v[146:149], v6 offset:32
	ds_load_b128 v[150:153], v6 offset:48
	v_cndmask_b32_e64 v187, v215, v219, s54
	v_cndmask_b32_e64 v191, v219, v215, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[122:129], v[178:185], v[154:161], v[122:129]
	ds_load_b128 v[154:157], v6 offset:1312
	ds_load_b128 v[158:161], v6 offset:1328
	v_cndmask_b32_e64 v188, v216, v220, s54
	v_cndmask_b32_e64 v192, v220, v216, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[130:137], v[178:185], v[162:169], v[130:137]
	ds_load_b128 v[162:165], v6 offset:2592
	ds_load_b128 v[166:169], v6 offset:2608
	v_cndmask_b32_e64 v189, v217, v221, s54
	v_cndmask_b32_e64 v193, v221, v217, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[138:145], v[178:185], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:3872
	ds_load_b128 v[174:177], v6 offset:3888
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[186:193], v[146:153], v[18:25]
	ds_load_b128 v[146:149], v6 offset:5152
	ds_load_b128 v[150:153], v6 offset:5168
	s_waitcnt vmcnt(11)
	v_perm_b32 v214, v207, v206, v1
	v_perm_b32 v215, v207, v206, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[186:193], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:6432
	ds_load_b128 v[158:161], v6 offset:6448
	v_perm_b32 v216, v207, v206, v222
	v_perm_b32 v217, v207, v206, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[186:193], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:7712
	ds_load_b128 v[166:169], v6 offset:7728
	s_waitcnt vmcnt(9)
	ds_store_b128 v5, v[224:227] offset:20480
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[42:49], v[186:193], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:8992
	ds_load_b128 v[174:177], v6 offset:9008
	s_waitcnt vmcnt(8)
	ds_store_b128 v5, v[228:231] offset:23040
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[50:57], v[186:193], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:10272
	ds_load_b128 v[150:153], v6 offset:10288
	s_waitcnt vmcnt(7)
	ds_store_b128 v5, v[232:235] offset:25600
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[58:65], v[186:193], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:11552
	ds_load_b128 v[158:161], v6 offset:11568
	s_waitcnt vmcnt(6)
	ds_store_b128 v5, v[236:239] offset:28160
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[66:73], v[186:193], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:12832
	ds_load_b128 v[166:169], v6 offset:12848
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[240:243] offset:30720
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[74:81], v[186:193], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:14112
	ds_load_b128 v[174:177], v6 offset:14128
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[244:247] offset:33280
	v_pk_fma_f16 v214, v210, v214, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v210, v215, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[82:89], v[186:193], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:15392
	ds_load_b128 v[150:153], v6 offset:15408
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[248:251] offset:35840
	v_pk_fma_f16 v216, v210, v216, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v210, v217, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[90:97], v[186:193], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:16672
	ds_load_b128 v[158:161], v6 offset:16688
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[252:255] offset:38400
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[98:105], v[186:193], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:17952
	ds_load_b128 v[166:169], v6 offset:17968
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[106:113], v[186:193], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:19232
	ds_load_b128 v[174:177], v6 offset:19248
	v_cndmask_b32_e64 v178, v214, v218, s54
	v_cndmask_b32_e64 v182, v218, v214, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[224:227], v10, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v11, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v12, s[8:11], s33 offen
	buffer_load_b128 v[236:239], v13, s[8:11], s33 offen
	buffer_load_b128 v[240:243], v14, s[8:11], s33 offen
	buffer_load_b128 v[244:247], v15, s[8:11], s33 offen
	buffer_load_b128 v[248:251], v16, s[8:11], s33 offen
	buffer_load_b128 v[252:255], v17, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[114:121], v[186:193], v[146:153], v[114:121]
	v_cndmask_b32_e64 v179, v215, v219, s54
	v_cndmask_b32_e64 v183, v219, v215, s54
	ds_load_b128 v[146:149], v6 offset:20480
	ds_load_b128 v[150:153], v6 offset:20496
	v_wmma_f32_16x16x16_f16 v[122:129], v[186:193], v[154:161], v[122:129]
	v_cndmask_b32_e64 v180, v216, v220, s54
	v_cndmask_b32_e64 v184, v220, v216, s54
	ds_load_b128 v[154:157], v6 offset:21760
	ds_load_b128 v[158:161], v6 offset:21776
	v_wmma_f32_16x16x16_f16 v[130:137], v[186:193], v[162:169], v[130:137]
	v_cndmask_b32_e64 v181, v217, v221, s54
	v_cndmask_b32_e64 v185, v221, v217, s54
	ds_load_b128 v[162:165], v6 offset:23040
	ds_load_b128 v[166:169], v6 offset:23056
	v_wmma_f32_16x16x16_f16 v[138:145], v[186:193], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:24320
	ds_load_b128 v[174:177], v6 offset:24336
	s_add_i32 s33, s34, 5
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[178:185], v[146:153], v[18:25]
	buffer_load_b64 v[198:199], v3, s[4:7], s31 offen offset:128
	buffer_load_b64 v[200:201], v3, s[4:7], s31 offen offset:140
	ds_load_b128 v[146:149], v6 offset:25600
	ds_load_b128 v[150:153], v6 offset:25616
	v_perm_b32 v214, v209, v208, v1
	v_perm_b32 v215, v209, v208, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[178:185], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:26880
	ds_load_b128 v[158:161], v6 offset:26896
	v_perm_b32 v216, v209, v208, v222
	v_perm_b32 v217, v209, v208, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[178:185], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:28160
	ds_load_b128 v[166:169], v6 offset:28176
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[42:49], v[178:185], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:29440
	ds_load_b128 v[174:177], v6 offset:29456
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[50:57], v[178:185], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:30720
	ds_load_b128 v[150:153], v6 offset:30736
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[58:65], v[178:185], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:32000
	ds_load_b128 v[158:161], v6 offset:32016
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[66:73], v[178:185], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:33280
	ds_load_b128 v[166:169], v6 offset:33296
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[74:81], v[178:185], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:34560
	ds_load_b128 v[174:177], v6 offset:34576
	v_pk_fma_f16 v214, v210, v214, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v210, v215, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[82:89], v[178:185], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:35840
	ds_load_b128 v[150:153], v6 offset:35856
	v_pk_fma_f16 v216, v210, v216, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v210, v217, v210 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[90:97], v[178:185], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:37120
	ds_load_b128 v[158:161], v6 offset:37136
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[98:105], v[178:185], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:38400
	ds_load_b128 v[166:169], v6 offset:38416
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[106:113], v[178:185], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:39680
	ds_load_b128 v[174:177], v6 offset:39696
	v_cndmask_b32_e64 v186, v214, v218, s54
	v_cndmask_b32_e64 v190, v218, v214, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[114:121], v[178:185], v[146:153], v[114:121]
	ds_load_b128 v[146:149], v6 offset:20512
	ds_load_b128 v[150:153], v6 offset:20528
	v_cndmask_b32_e64 v187, v215, v219, s54
	v_cndmask_b32_e64 v191, v219, v215, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[122:129], v[178:185], v[154:161], v[122:129]
	ds_load_b128 v[154:157], v6 offset:21792
	ds_load_b128 v[158:161], v6 offset:21808
	v_cndmask_b32_e64 v188, v216, v220, s54
	v_cndmask_b32_e64 v192, v220, v216, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[130:137], v[178:185], v[162:169], v[130:137]
	ds_load_b128 v[162:165], v6 offset:23072
	ds_load_b128 v[166:169], v6 offset:23088
	v_cndmask_b32_e64 v189, v217, v221, s54
	v_cndmask_b32_e64 v193, v221, v217, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[138:145], v[178:185], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:24352
	ds_load_b128 v[174:177], v6 offset:24368
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[186:193], v[146:153], v[18:25]
	ds_load_b128 v[146:149], v6 offset:25632
	ds_load_b128 v[150:153], v6 offset:25648
	s_waitcnt vmcnt(11)
	v_perm_b32 v214, v195, v194, v1
	v_perm_b32 v215, v195, v194, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[186:193], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:26912
	ds_load_b128 v[158:161], v6 offset:26928
	v_perm_b32 v216, v195, v194, v222
	v_perm_b32 v217, v195, v194, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[186:193], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:28192
	ds_load_b128 v[166:169], v6 offset:28208
	s_waitcnt vmcnt(9)
	ds_store_b128 v5, v[224:227]
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[42:49], v[186:193], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:29472
	ds_load_b128 v[174:177], v6 offset:29488
	s_waitcnt vmcnt(8)
	ds_store_b128 v5, v[228:231] offset:2560
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[50:57], v[186:193], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:30752
	ds_load_b128 v[150:153], v6 offset:30768
	s_waitcnt vmcnt(7)
	ds_store_b128 v5, v[232:235] offset:5120
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[58:65], v[186:193], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:32032
	ds_load_b128 v[158:161], v6 offset:32048
	s_waitcnt vmcnt(6)
	ds_store_b128 v5, v[236:239] offset:7680
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[66:73], v[186:193], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:33312
	ds_load_b128 v[166:169], v6 offset:33328
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[240:243] offset:10240
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[74:81], v[186:193], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:34592
	ds_load_b128 v[174:177], v6 offset:34608
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[244:247] offset:12800
	v_pk_fma_f16 v214, v211, v214, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v211, v215, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[82:89], v[186:193], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:35872
	ds_load_b128 v[150:153], v6 offset:35888
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[248:251] offset:15360
	v_pk_fma_f16 v216, v211, v216, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v211, v217, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[90:97], v[186:193], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:37152
	ds_load_b128 v[158:161], v6 offset:37168
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[252:255] offset:17920
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[98:105], v[186:193], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:38432
	ds_load_b128 v[166:169], v6 offset:38448
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[106:113], v[186:193], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:39712
	ds_load_b128 v[174:177], v6 offset:39728
	v_cndmask_b32_e64 v178, v214, v218, s54
	v_cndmask_b32_e64 v182, v218, v214, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[224:227], v10, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v11, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v12, s[8:11], s33 offen
	buffer_load_b128 v[236:239], v13, s[8:11], s33 offen
	buffer_load_b128 v[240:243], v14, s[8:11], s33 offen
	buffer_load_b128 v[244:247], v15, s[8:11], s33 offen
	buffer_load_b128 v[248:251], v16, s[8:11], s33 offen
	buffer_load_b128 v[252:255], v17, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[114:121], v[186:193], v[146:153], v[114:121]
	v_cndmask_b32_e64 v179, v215, v219, s54
	v_cndmask_b32_e64 v183, v219, v215, s54
	ds_load_b128 v[146:149], v6
	ds_load_b128 v[150:153], v6 offset:16
	v_wmma_f32_16x16x16_f16 v[122:129], v[186:193], v[154:161], v[122:129]
	v_cndmask_b32_e64 v180, v216, v220, s54
	v_cndmask_b32_e64 v184, v220, v216, s54
	ds_load_b128 v[154:157], v6 offset:1280
	ds_load_b128 v[158:161], v6 offset:1296
	v_wmma_f32_16x16x16_f16 v[130:137], v[186:193], v[162:169], v[130:137]
	v_cndmask_b32_e64 v181, v217, v221, s54
	v_cndmask_b32_e64 v185, v221, v217, s54
	ds_load_b128 v[162:165], v6 offset:2560
	ds_load_b128 v[166:169], v6 offset:2576
	v_wmma_f32_16x16x16_f16 v[138:145], v[186:193], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:3840
	ds_load_b128 v[174:177], v6 offset:3856
	s_add_i32 s33, s34, 6
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[178:185], v[146:153], v[18:25]
	buffer_load_b64 v[202:203], v3, s[4:7], s31 offen offset:152
	buffer_load_b64 v[204:205], v3, s[4:7], s31 offen offset:164
	ds_load_b128 v[146:149], v6 offset:5120
	ds_load_b128 v[150:153], v6 offset:5136
	v_perm_b32 v214, v197, v196, v1
	v_perm_b32 v215, v197, v196, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[178:185], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:6400
	ds_load_b128 v[158:161], v6 offset:6416
	v_perm_b32 v216, v197, v196, v222
	v_perm_b32 v217, v197, v196, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[178:185], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:7680
	ds_load_b128 v[166:169], v6 offset:7696
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[42:49], v[178:185], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:8960
	ds_load_b128 v[174:177], v6 offset:8976
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[50:57], v[178:185], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:10240
	ds_load_b128 v[150:153], v6 offset:10256
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[58:65], v[178:185], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:11520
	ds_load_b128 v[158:161], v6 offset:11536
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[66:73], v[178:185], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:12800
	ds_load_b128 v[166:169], v6 offset:12816
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[74:81], v[178:185], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:14080
	ds_load_b128 v[174:177], v6 offset:14096
	v_pk_fma_f16 v214, v211, v214, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v211, v215, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[82:89], v[178:185], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:15360
	ds_load_b128 v[150:153], v6 offset:15376
	v_pk_fma_f16 v216, v211, v216, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v211, v217, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[90:97], v[178:185], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:16640
	ds_load_b128 v[158:161], v6 offset:16656
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[98:105], v[178:185], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:17920
	ds_load_b128 v[166:169], v6 offset:17936
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[106:113], v[178:185], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:19200
	ds_load_b128 v[174:177], v6 offset:19216
	v_cndmask_b32_e64 v186, v214, v218, s54
	v_cndmask_b32_e64 v190, v218, v214, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[114:121], v[178:185], v[146:153], v[114:121]
	ds_load_b128 v[146:149], v6 offset:32
	ds_load_b128 v[150:153], v6 offset:48
	v_cndmask_b32_e64 v187, v215, v219, s54
	v_cndmask_b32_e64 v191, v219, v215, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[122:129], v[178:185], v[154:161], v[122:129]
	ds_load_b128 v[154:157], v6 offset:1312
	ds_load_b128 v[158:161], v6 offset:1328
	v_cndmask_b32_e64 v188, v216, v220, s54
	v_cndmask_b32_e64 v192, v220, v216, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[130:137], v[178:185], v[162:169], v[130:137]
	ds_load_b128 v[162:165], v6 offset:2592
	ds_load_b128 v[166:169], v6 offset:2608
	v_cndmask_b32_e64 v189, v217, v221, s54
	v_cndmask_b32_e64 v193, v221, v217, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[138:145], v[178:185], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:3872
	ds_load_b128 v[174:177], v6 offset:3888
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[186:193], v[146:153], v[18:25]
	ds_load_b128 v[146:149], v6 offset:5152
	ds_load_b128 v[150:153], v6 offset:5168
	s_waitcnt vmcnt(11)
	v_perm_b32 v214, v199, v198, v1
	v_perm_b32 v215, v199, v198, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[186:193], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:6432
	ds_load_b128 v[158:161], v6 offset:6448
	v_perm_b32 v216, v199, v198, v222
	v_perm_b32 v217, v199, v198, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[186:193], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:7712
	ds_load_b128 v[166:169], v6 offset:7728
	s_waitcnt vmcnt(9)
	ds_store_b128 v5, v[224:227] offset:20480
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[42:49], v[186:193], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:8992
	ds_load_b128 v[174:177], v6 offset:9008
	s_waitcnt vmcnt(8)
	ds_store_b128 v5, v[228:231] offset:23040
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[50:57], v[186:193], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:10272
	ds_load_b128 v[150:153], v6 offset:10288
	s_waitcnt vmcnt(7)
	ds_store_b128 v5, v[232:235] offset:25600
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[58:65], v[186:193], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:11552
	ds_load_b128 v[158:161], v6 offset:11568
	s_waitcnt vmcnt(6)
	ds_store_b128 v5, v[236:239] offset:28160
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[66:73], v[186:193], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:12832
	ds_load_b128 v[166:169], v6 offset:12848
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[240:243] offset:30720
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[74:81], v[186:193], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:14112
	ds_load_b128 v[174:177], v6 offset:14128
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[244:247] offset:33280
	v_pk_fma_f16 v214, v211, v214, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v211, v215, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[82:89], v[186:193], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:15392
	ds_load_b128 v[150:153], v6 offset:15408
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[248:251] offset:35840
	v_pk_fma_f16 v216, v211, v216, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v211, v217, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[90:97], v[186:193], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:16672
	ds_load_b128 v[158:161], v6 offset:16688
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[252:255] offset:38400
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[98:105], v[186:193], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:17952
	ds_load_b128 v[166:169], v6 offset:17968
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[106:113], v[186:193], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:19232
	ds_load_b128 v[174:177], v6 offset:19248
	v_cndmask_b32_e64 v178, v214, v218, s54
	v_cndmask_b32_e64 v182, v218, v214, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[224:227], v10, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v11, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v12, s[8:11], s33 offen
	buffer_load_b128 v[236:239], v13, s[8:11], s33 offen
	buffer_load_b128 v[240:243], v14, s[8:11], s33 offen
	buffer_load_b128 v[244:247], v15, s[8:11], s33 offen
	buffer_load_b128 v[248:251], v16, s[8:11], s33 offen
	buffer_load_b128 v[252:255], v17, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[114:121], v[186:193], v[146:153], v[114:121]
	v_cndmask_b32_e64 v179, v215, v219, s54
	v_cndmask_b32_e64 v183, v219, v215, s54
	ds_load_b128 v[146:149], v6 offset:20480
	ds_load_b128 v[150:153], v6 offset:20496
	v_wmma_f32_16x16x16_f16 v[122:129], v[186:193], v[154:161], v[122:129]
	v_cndmask_b32_e64 v180, v216, v220, s54
	v_cndmask_b32_e64 v184, v220, v216, s54
	ds_load_b128 v[154:157], v6 offset:21760
	ds_load_b128 v[158:161], v6 offset:21776
	v_wmma_f32_16x16x16_f16 v[130:137], v[186:193], v[162:169], v[130:137]
	v_cndmask_b32_e64 v181, v217, v221, s54
	v_cndmask_b32_e64 v185, v221, v217, s54
	ds_load_b128 v[162:165], v6 offset:23040
	ds_load_b128 v[166:169], v6 offset:23056
	v_wmma_f32_16x16x16_f16 v[138:145], v[186:193], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:24320
	ds_load_b128 v[174:177], v6 offset:24336
	s_add_i32 s33, s34, 7
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[178:185], v[146:153], v[18:25]
	buffer_load_b64 v[206:207], v3, s[4:7], s31 offen offset:176
	buffer_load_b64 v[208:209], v3, s[4:7], s31 offen offset:188
	ds_load_b128 v[146:149], v6 offset:25600
	ds_load_b128 v[150:153], v6 offset:25616
	v_perm_b32 v214, v201, v200, v1
	v_perm_b32 v215, v201, v200, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[178:185], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:26880
	ds_load_b128 v[158:161], v6 offset:26896
	v_perm_b32 v216, v201, v200, v222
	v_perm_b32 v217, v201, v200, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[178:185], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:28160
	ds_load_b128 v[166:169], v6 offset:28176
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[42:49], v[178:185], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:29440
	ds_load_b128 v[174:177], v6 offset:29456
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[50:57], v[178:185], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:30720
	ds_load_b128 v[150:153], v6 offset:30736
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[58:65], v[178:185], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:32000
	ds_load_b128 v[158:161], v6 offset:32016
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[66:73], v[178:185], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:33280
	ds_load_b128 v[166:169], v6 offset:33296
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[74:81], v[178:185], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:34560
	ds_load_b128 v[174:177], v6 offset:34576
	v_pk_fma_f16 v214, v211, v214, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v211, v215, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[82:89], v[178:185], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:35840
	ds_load_b128 v[150:153], v6 offset:35856
	v_pk_fma_f16 v216, v211, v216, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v211, v217, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[90:97], v[178:185], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:37120
	ds_load_b128 v[158:161], v6 offset:37136
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[98:105], v[178:185], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:38400
	ds_load_b128 v[166:169], v6 offset:38416
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[106:113], v[178:185], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:39680
	ds_load_b128 v[174:177], v6 offset:39696
	v_cndmask_b32_e64 v186, v214, v218, s54
	v_cndmask_b32_e64 v190, v218, v214, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[114:121], v[178:185], v[146:153], v[114:121]
	ds_load_b128 v[146:149], v6 offset:20512
	ds_load_b128 v[150:153], v6 offset:20528
	v_cndmask_b32_e64 v187, v215, v219, s54
	v_cndmask_b32_e64 v191, v219, v215, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[122:129], v[178:185], v[154:161], v[122:129]
	ds_load_b128 v[154:157], v6 offset:21792
	ds_load_b128 v[158:161], v6 offset:21808
	v_cndmask_b32_e64 v188, v216, v220, s54
	v_cndmask_b32_e64 v192, v220, v216, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[130:137], v[178:185], v[162:169], v[130:137]
	ds_load_b128 v[162:165], v6 offset:23072
	ds_load_b128 v[166:169], v6 offset:23088
	v_cndmask_b32_e64 v189, v217, v221, s54
	v_cndmask_b32_e64 v193, v221, v217, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[138:145], v[178:185], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:24352
	ds_load_b128 v[174:177], v6 offset:24368
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[186:193], v[146:153], v[18:25]
	ds_load_b128 v[146:149], v6 offset:25632
	ds_load_b128 v[150:153], v6 offset:25648
	s_waitcnt vmcnt(11)
	v_perm_b32 v214, v203, v202, v1
	v_perm_b32 v215, v203, v202, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[186:193], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:26912
	ds_load_b128 v[158:161], v6 offset:26928
	v_perm_b32 v216, v203, v202, v222
	v_perm_b32 v217, v203, v202, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[186:193], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:28192
	ds_load_b128 v[166:169], v6 offset:28208
	s_waitcnt vmcnt(9)
	ds_store_b128 v5, v[224:227]
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[42:49], v[186:193], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:29472
	ds_load_b128 v[174:177], v6 offset:29488
	s_waitcnt vmcnt(8)
	ds_store_b128 v5, v[228:231] offset:2560
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[50:57], v[186:193], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:30752
	ds_load_b128 v[150:153], v6 offset:30768
	s_waitcnt vmcnt(7)
	ds_store_b128 v5, v[232:235] offset:5120
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[58:65], v[186:193], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:32032
	ds_load_b128 v[158:161], v6 offset:32048
	s_waitcnt vmcnt(6)
	ds_store_b128 v5, v[236:239] offset:7680
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[66:73], v[186:193], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:33312
	ds_load_b128 v[166:169], v6 offset:33328
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[240:243] offset:10240
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[74:81], v[186:193], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:34592
	ds_load_b128 v[174:177], v6 offset:34608
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[244:247] offset:12800
	v_pk_fma_f16 v214, v211, v214, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v211, v215, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[82:89], v[186:193], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:35872
	ds_load_b128 v[150:153], v6 offset:35888
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[248:251] offset:15360
	v_pk_fma_f16 v216, v211, v216, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v211, v217, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[90:97], v[186:193], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:37152
	ds_load_b128 v[158:161], v6 offset:37168
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[252:255] offset:17920
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[98:105], v[186:193], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:38432
	ds_load_b128 v[166:169], v6 offset:38448
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[106:113], v[186:193], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:39712
	ds_load_b128 v[174:177], v6 offset:39728
	v_cndmask_b32_e64 v178, v214, v218, s54
	v_cndmask_b32_e64 v182, v218, v214, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[224:227], v10, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v11, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v12, s[8:11], s33 offen
	buffer_load_b128 v[236:239], v13, s[8:11], s33 offen
	buffer_load_b128 v[240:243], v14, s[8:11], s33 offen
	buffer_load_b128 v[244:247], v15, s[8:11], s33 offen
	buffer_load_b128 v[248:251], v16, s[8:11], s33 offen
	buffer_load_b128 v[252:255], v17, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[114:121], v[186:193], v[146:153], v[114:121]
	v_cndmask_b32_e64 v179, v215, v219, s54
	v_cndmask_b32_e64 v183, v219, v215, s54
	ds_load_b128 v[146:149], v6
	ds_load_b128 v[150:153], v6 offset:16
	v_wmma_f32_16x16x16_f16 v[122:129], v[186:193], v[154:161], v[122:129]
	v_cndmask_b32_e64 v180, v216, v220, s54
	v_cndmask_b32_e64 v184, v220, v216, s54
	ds_load_b128 v[154:157], v6 offset:1280
	ds_load_b128 v[158:161], v6 offset:1296
	v_wmma_f32_16x16x16_f16 v[130:137], v[186:193], v[162:169], v[130:137]
	v_cndmask_b32_e64 v181, v217, v221, s54
	v_cndmask_b32_e64 v185, v221, v217, s54
	ds_load_b128 v[162:165], v6 offset:2560
	ds_load_b128 v[166:169], v6 offset:2576
	v_wmma_f32_16x16x16_f16 v[138:145], v[186:193], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:3840
	ds_load_b128 v[174:177], v6 offset:3856
	s_add_i32 s33, s34, 8
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[178:185], v[146:153], v[18:25]
	buffer_load_b64 v[194:195], v3, s[4:7], s32 offen offset:8
	buffer_load_b64 v[196:197], v3, s[4:7], s32 offen offset:20
	buffer_load_b64 v[212:213], v4, s[4:7], s32 offen
	ds_load_b128 v[146:149], v6 offset:5120
	ds_load_b128 v[150:153], v6 offset:5136
	v_perm_b32 v214, v205, v204, v1
	v_perm_b32 v215, v205, v204, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[178:185], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:6400
	ds_load_b128 v[158:161], v6 offset:6416
	v_perm_b32 v216, v205, v204, v222
	v_perm_b32 v217, v205, v204, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[178:185], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:7680
	ds_load_b128 v[166:169], v6 offset:7696
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[42:49], v[178:185], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:8960
	ds_load_b128 v[174:177], v6 offset:8976
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[50:57], v[178:185], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:10240
	ds_load_b128 v[150:153], v6 offset:10256
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[58:65], v[178:185], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:11520
	ds_load_b128 v[158:161], v6 offset:11536
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[66:73], v[178:185], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:12800
	ds_load_b128 v[166:169], v6 offset:12816
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[74:81], v[178:185], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:14080
	ds_load_b128 v[174:177], v6 offset:14096
	v_pk_fma_f16 v214, v211, v214, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v211, v215, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[82:89], v[178:185], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:15360
	ds_load_b128 v[150:153], v6 offset:15376
	v_pk_fma_f16 v216, v211, v216, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v211, v217, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[90:97], v[178:185], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:16640
	ds_load_b128 v[158:161], v6 offset:16656
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[98:105], v[178:185], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:17920
	ds_load_b128 v[166:169], v6 offset:17936
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[106:113], v[178:185], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:19200
	ds_load_b128 v[174:177], v6 offset:19216
	v_cndmask_b32_e64 v186, v214, v218, s54
	v_cndmask_b32_e64 v190, v218, v214, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[114:121], v[178:185], v[146:153], v[114:121]
	ds_load_b128 v[146:149], v6 offset:32
	ds_load_b128 v[150:153], v6 offset:48
	v_cndmask_b32_e64 v187, v215, v219, s54
	v_cndmask_b32_e64 v191, v219, v215, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[122:129], v[178:185], v[154:161], v[122:129]
	ds_load_b128 v[154:157], v6 offset:1312
	ds_load_b128 v[158:161], v6 offset:1328
	v_cndmask_b32_e64 v188, v216, v220, s54
	v_cndmask_b32_e64 v192, v220, v216, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[130:137], v[178:185], v[162:169], v[130:137]
	ds_load_b128 v[162:165], v6 offset:2592
	ds_load_b128 v[166:169], v6 offset:2608
	v_cndmask_b32_e64 v189, v217, v221, s54
	v_cndmask_b32_e64 v193, v221, v217, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[138:145], v[178:185], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:3872
	ds_load_b128 v[174:177], v6 offset:3888
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[186:193], v[146:153], v[18:25]
	ds_load_b128 v[146:149], v6 offset:5152
	ds_load_b128 v[150:153], v6 offset:5168
	s_waitcnt vmcnt(12)
	v_perm_b32 v214, v207, v206, v1
	v_perm_b32 v215, v207, v206, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[186:193], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:6432
	ds_load_b128 v[158:161], v6 offset:6448
	v_perm_b32 v216, v207, v206, v222
	v_perm_b32 v217, v207, v206, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[186:193], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:7712
	ds_load_b128 v[166:169], v6 offset:7728
	s_waitcnt vmcnt(10)
	ds_store_b128 v5, v[224:227] offset:20480
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[42:49], v[186:193], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:8992
	ds_load_b128 v[174:177], v6 offset:9008
	s_waitcnt vmcnt(9)
	ds_store_b128 v5, v[228:231] offset:23040
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[50:57], v[186:193], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:10272
	ds_load_b128 v[150:153], v6 offset:10288
	s_waitcnt vmcnt(8)
	ds_store_b128 v5, v[232:235] offset:25600
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[58:65], v[186:193], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:11552
	ds_load_b128 v[158:161], v6 offset:11568
	s_waitcnt vmcnt(7)
	ds_store_b128 v5, v[236:239] offset:28160
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[66:73], v[186:193], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:12832
	ds_load_b128 v[166:169], v6 offset:12848
	s_waitcnt vmcnt(6)
	ds_store_b128 v5, v[240:243] offset:30720
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[74:81], v[186:193], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:14112
	ds_load_b128 v[174:177], v6 offset:14128
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[244:247] offset:33280
	v_pk_fma_f16 v214, v211, v214, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v211, v215, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[82:89], v[186:193], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:15392
	ds_load_b128 v[150:153], v6 offset:15408
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[248:251] offset:35840
	v_pk_fma_f16 v216, v211, v216, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v211, v217, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[90:97], v[186:193], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:16672
	ds_load_b128 v[158:161], v6 offset:16688
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[252:255] offset:38400
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[98:105], v[186:193], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:17952
	ds_load_b128 v[166:169], v6 offset:17968
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[106:113], v[186:193], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:19232
	ds_load_b128 v[174:177], v6 offset:19248
	v_cndmask_b32_e64 v178, v214, v218, s54
	v_cndmask_b32_e64 v182, v218, v214, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[224:227], v10, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v11, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v12, s[8:11], s33 offen
	buffer_load_b128 v[236:239], v13, s[8:11], s33 offen
	buffer_load_b128 v[240:243], v14, s[8:11], s33 offen
	buffer_load_b128 v[244:247], v15, s[8:11], s33 offen
	buffer_load_b128 v[248:251], v16, s[8:11], s33 offen
	buffer_load_b128 v[252:255], v17, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[114:121], v[186:193], v[146:153], v[114:121]
	v_cndmask_b32_e64 v179, v215, v219, s54
	v_cndmask_b32_e64 v183, v219, v215, s54
	ds_load_b128 v[146:149], v6 offset:20480
	ds_load_b128 v[150:153], v6 offset:20496
	v_wmma_f32_16x16x16_f16 v[122:129], v[186:193], v[154:161], v[122:129]
	v_cndmask_b32_e64 v180, v216, v220, s54
	v_cndmask_b32_e64 v184, v220, v216, s54
	ds_load_b128 v[154:157], v6 offset:21760
	ds_load_b128 v[158:161], v6 offset:21776
	v_wmma_f32_16x16x16_f16 v[130:137], v[186:193], v[162:169], v[130:137]
	v_cndmask_b32_e64 v181, v217, v221, s54
	v_cndmask_b32_e64 v185, v221, v217, s54
	ds_load_b128 v[162:165], v6 offset:23040
	ds_load_b128 v[166:169], v6 offset:23056
	v_wmma_f32_16x16x16_f16 v[138:145], v[186:193], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:24320
	ds_load_b128 v[174:177], v6 offset:24336
	s_add_i32 s33, s34, 9
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[178:185], v[146:153], v[18:25]
	buffer_load_b64 v[198:199], v3, s[4:7], s32 offen offset:32
	buffer_load_b64 v[200:201], v3, s[4:7], s32 offen offset:44
	ds_load_b128 v[146:149], v6 offset:25600
	ds_load_b128 v[150:153], v6 offset:25616
	v_perm_b32 v214, v209, v208, v1
	v_perm_b32 v215, v209, v208, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[178:185], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:26880
	ds_load_b128 v[158:161], v6 offset:26896
	v_perm_b32 v216, v209, v208, v222
	v_perm_b32 v217, v209, v208, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[178:185], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:28160
	ds_load_b128 v[166:169], v6 offset:28176
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[42:49], v[178:185], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:29440
	ds_load_b128 v[174:177], v6 offset:29456
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[50:57], v[178:185], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:30720
	ds_load_b128 v[150:153], v6 offset:30736
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[58:65], v[178:185], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:32000
	ds_load_b128 v[158:161], v6 offset:32016
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[66:73], v[178:185], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:33280
	ds_load_b128 v[166:169], v6 offset:33296
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[74:81], v[178:185], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:34560
	ds_load_b128 v[174:177], v6 offset:34576
	v_pk_fma_f16 v214, v211, v214, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v211, v215, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[82:89], v[178:185], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:35840
	ds_load_b128 v[150:153], v6 offset:35856
	v_pk_fma_f16 v216, v211, v216, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v211, v217, v211 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[90:97], v[178:185], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:37120
	ds_load_b128 v[158:161], v6 offset:37136
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[98:105], v[178:185], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:38400
	ds_load_b128 v[166:169], v6 offset:38416
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[106:113], v[178:185], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:39680
	ds_load_b128 v[174:177], v6 offset:39696
	v_cndmask_b32_e64 v186, v214, v218, s54
	v_cndmask_b32_e64 v190, v218, v214, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[114:121], v[178:185], v[146:153], v[114:121]
	ds_load_b128 v[146:149], v6 offset:20512
	ds_load_b128 v[150:153], v6 offset:20528
	v_cndmask_b32_e64 v187, v215, v219, s54
	v_cndmask_b32_e64 v191, v219, v215, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[122:129], v[178:185], v[154:161], v[122:129]
	ds_load_b128 v[154:157], v6 offset:21792
	ds_load_b128 v[158:161], v6 offset:21808
	v_cndmask_b32_e64 v188, v216, v220, s54
	v_cndmask_b32_e64 v192, v220, v216, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[130:137], v[178:185], v[162:169], v[130:137]
	ds_load_b128 v[162:165], v6 offset:23072
	ds_load_b128 v[166:169], v6 offset:23088
	v_cndmask_b32_e64 v189, v217, v221, s54
	v_cndmask_b32_e64 v193, v221, v217, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[138:145], v[178:185], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:24352
	ds_load_b128 v[174:177], v6 offset:24368
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[18:25], v[186:193], v[146:153], v[18:25]
	ds_load_b128 v[146:149], v6 offset:25632
	ds_load_b128 v[150:153], v6 offset:25648
	s_waitcnt vmcnt(12)
	v_perm_b32 v214, v195, v194, v1
	v_perm_b32 v215, v195, v194, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[26:33], v[186:193], v[154:161], v[26:33]
	ds_load_b128 v[154:157], v6 offset:26912
	ds_load_b128 v[158:161], v6 offset:26928
	v_perm_b32 v216, v195, v194, v222
	v_perm_b32 v217, v195, v194, v223
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[34:41], v[186:193], v[162:169], v[34:41]
	ds_load_b128 v[162:165], v6 offset:28192
	ds_load_b128 v[166:169], v6 offset:28208
	s_waitcnt vmcnt(9)
	ds_store_b128 v5, v[224:227]
	v_lshl_or_b32 v214, v214, 10, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v216, v216, 10, v216
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[42:49], v[186:193], v[170:177], v[42:49]
	ds_load_b128 v[170:173], v6 offset:29472
	ds_load_b128 v[174:177], v6 offset:29488
	s_waitcnt vmcnt(8)
	ds_store_b128 v5, v[228:231] offset:2560
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v214, v214, s80, 0x64006400
	v_and_or_b32 v215, v215, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[50:57], v[186:193], v[146:153], v[50:57]
	ds_load_b128 v[146:149], v6 offset:30752
	ds_load_b128 v[150:153], v6 offset:30768
	s_waitcnt vmcnt(7)
	ds_store_b128 v5, v[232:235] offset:5120
	v_and_or_b32 v216, v216, s80, 0x64006400
	v_and_or_b32 v217, v217, s81, 0x5c005400
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[58:65], v[186:193], v[154:161], v[58:65]
	ds_load_b128 v[154:157], v6 offset:32032
	ds_load_b128 v[158:161], v6 offset:32048
	s_waitcnt vmcnt(6)
	ds_store_b128 v5, v[236:239] offset:7680
	v_pk_add_f16 v214, v214, s82
	v_pk_add_f16 v215, v215, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[66:73], v[186:193], v[162:169], v[66:73]
	ds_load_b128 v[162:165], v6 offset:33312
	ds_load_b128 v[166:169], v6 offset:33328
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[240:243] offset:10240
	v_pk_add_f16 v216, v216, s82
	v_pk_add_f16 v217, v217, s83
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[74:81], v[186:193], v[170:177], v[74:81]
	ds_load_b128 v[170:173], v6 offset:34592
	ds_load_b128 v[174:177], v6 offset:34608
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[244:247] offset:12800
	v_pk_fma_f16 v214, v212, v214, v212 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v215, v212, v215, v212 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[82:89], v[186:193], v[146:153], v[82:89]
	ds_load_b128 v[146:149], v6 offset:35872
	ds_load_b128 v[150:153], v6 offset:35888
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[248:251] offset:15360
	v_pk_fma_f16 v216, v212, v216, v212 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v217, v212, v217, v212 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[90:97], v[186:193], v[154:161], v[90:97]
	ds_load_b128 v[154:157], v6 offset:37152
	ds_load_b128 v[158:161], v6 offset:37168
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[252:255] offset:17920
	v_permlanex16_b32 v218, v214, s50, 0xfedcba98
	v_permlanex16_b32 v219, v215, s50, 0xfedcba98
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[98:105], v[186:193], v[162:169], v[98:105]
	ds_load_b128 v[162:165], v6 offset:38432
	ds_load_b128 v[166:169], v6 offset:38448
	v_permlanex16_b32 v220, v216, s50, 0xfedcba98
	v_permlanex16_b32 v221, v217, s50, 0xfedcba98
	s_waitcnt lgkmcnt(9)
	v_wmma_f32_16x16x16_f16 v[106:113], v[186:193], v[170:177], v[106:113]
	ds_load_b128 v[170:173], v6 offset:39712
	ds_load_b128 v[174:177], v6 offset:39728
	v_cndmask_b32_e64 v178, v214, v218, s54
	v_cndmask_b32_e64 v182, v218, v214, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[224:227], v10, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v11, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v12, s[8:11], s33 offen
	buffer_load_b128 v[236:239], v13, s[8:11], s33 offen
	buffer_load_b128 v[240:243], v14, s[8:11], s33 offen
	buffer_load_b128 v[244:247], v15, s[8:11], s33 offen
	buffer_load_b128 v[248:251], v16, s[8:11], s33 offen
	buffer_load_b128 v[252:255], v17, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[114:121], v[186:193], v[146:153], v[114:121]
	v_cndmask_b32_e64 v179, v215, v219, s54
	v_cndmask_b32_e64 v183, v219, v215, s54
	ds_load_b128 v[146:149], v6
	ds_load_b128 v[150:153], v6 offset:16
	v_wmma_f32_16x16x16_f16 v[122:129], v[186:193], v[154:161], v[122:129]
	v_cndmask_b32_e64 v180, v216, v220, s54
	v_cndmask_b32_e64 v184, v220, v216, s54
	ds_load_b128 v[154:157], v6 offset:1280
	ds_load_b128 v[158:161], v6 offset:1296
	v_wmma_f32_16x16x16_f16 v[130:137], v[186:193], v[162:169], v[130:137]
	v_cndmask_b32_e64 v181, v217, v221, s54
	v_cndmask_b32_e64 v185, v221, v217, s54
	ds_load_b128 v[162:165], v6 offset:2560
	ds_load_b128 v[166:169], v6 offset:2576
	v_wmma_f32_16x16x16_f16 v[138:145], v[186:193], v[170:177], v[138:145]
	ds_load_b128 v[170:173], v6 offset:3840
	ds_load_b128 v[174:177], v6 offset:3856
	v_mov_b32_e32 v210, v212
	v_mov_b32_e32 v211, v213
	s_add_i32 s30, s30, 1
	s_add_i32 s31, s31, 0xc8
	s_add_i32 s34, s34, 8
	s_cmp_lt_u32 s30, s25
	s_cbranch_scc1 .Lmq6g11w4y_loop
	s_waitcnt vmcnt(0) lgkmcnt(0)
	.Lmq6g11w4y_epi:
	v_cndmask_b32_e64 v146, v22, v18, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v23, v19, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v24, v20, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v25, v21, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v19
	v_mov_b32_e32 v151, v21
	v_cndmask_b32_e64 v18, v18, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v25, v149, v25, s54
	v_cndmask_b32_e64 v19, v146, v22, s54
	v_cndmask_b32_e64 v22, v20, v148, s54
	v_cndmask_b32_e64 v20, v150, v147, s54
	v_cndmask_b32_e64 v21, v147, v23, s54
	v_cndmask_b32_e64 v23, v148, v24, s54
	v_cndmask_b32_e64 v24, v151, v149, s54
	v_cndmask_b32_e64 v146, v30, v26, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v31, v27, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v32, v28, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v33, v29, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v27
	v_mov_b32_e32 v151, v29
	v_cndmask_b32_e64 v26, v26, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v33, v149, v33, s54
	v_cndmask_b32_e64 v27, v146, v30, s54
	v_cndmask_b32_e64 v30, v28, v148, s54
	v_cndmask_b32_e64 v28, v150, v147, s54
	v_cndmask_b32_e64 v29, v147, v31, s54
	v_cndmask_b32_e64 v31, v148, v32, s54
	v_cndmask_b32_e64 v32, v151, v149, s54
	v_cndmask_b32_e64 v146, v38, v34, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v39, v35, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v40, v36, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v41, v37, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v35
	v_mov_b32_e32 v151, v37
	v_cndmask_b32_e64 v34, v34, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v41, v149, v41, s54
	v_cndmask_b32_e64 v35, v146, v38, s54
	v_cndmask_b32_e64 v38, v36, v148, s54
	v_cndmask_b32_e64 v36, v150, v147, s54
	v_cndmask_b32_e64 v37, v147, v39, s54
	v_cndmask_b32_e64 v39, v148, v40, s54
	v_cndmask_b32_e64 v40, v151, v149, s54
	v_cndmask_b32_e64 v146, v46, v42, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v47, v43, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v48, v44, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v49, v45, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v43
	v_mov_b32_e32 v151, v45
	v_cndmask_b32_e64 v42, v42, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v49, v149, v49, s54
	v_cndmask_b32_e64 v43, v146, v46, s54
	v_cndmask_b32_e64 v46, v44, v148, s54
	v_cndmask_b32_e64 v44, v150, v147, s54
	v_cndmask_b32_e64 v45, v147, v47, s54
	v_cndmask_b32_e64 v47, v148, v48, s54
	v_cndmask_b32_e64 v48, v151, v149, s54
	v_cndmask_b32_e64 v146, v54, v50, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v55, v51, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v56, v52, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v57, v53, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v51
	v_mov_b32_e32 v151, v53
	v_cndmask_b32_e64 v50, v50, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v57, v149, v57, s54
	v_cndmask_b32_e64 v51, v146, v54, s54
	v_cndmask_b32_e64 v54, v52, v148, s54
	v_cndmask_b32_e64 v52, v150, v147, s54
	v_cndmask_b32_e64 v53, v147, v55, s54
	v_cndmask_b32_e64 v55, v148, v56, s54
	v_cndmask_b32_e64 v56, v151, v149, s54
	v_cndmask_b32_e64 v146, v62, v58, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v63, v59, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v64, v60, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v65, v61, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v59
	v_mov_b32_e32 v151, v61
	v_cndmask_b32_e64 v58, v58, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v65, v149, v65, s54
	v_cndmask_b32_e64 v59, v146, v62, s54
	v_cndmask_b32_e64 v62, v60, v148, s54
	v_cndmask_b32_e64 v60, v150, v147, s54
	v_cndmask_b32_e64 v61, v147, v63, s54
	v_cndmask_b32_e64 v63, v148, v64, s54
	v_cndmask_b32_e64 v64, v151, v149, s54
	v_cndmask_b32_e64 v146, v70, v66, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v71, v67, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v72, v68, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v73, v69, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v67
	v_mov_b32_e32 v151, v69
	v_cndmask_b32_e64 v66, v66, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v73, v149, v73, s54
	v_cndmask_b32_e64 v67, v146, v70, s54
	v_cndmask_b32_e64 v70, v68, v148, s54
	v_cndmask_b32_e64 v68, v150, v147, s54
	v_cndmask_b32_e64 v69, v147, v71, s54
	v_cndmask_b32_e64 v71, v148, v72, s54
	v_cndmask_b32_e64 v72, v151, v149, s54
	v_cndmask_b32_e64 v146, v78, v74, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v79, v75, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v80, v76, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v81, v77, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v75
	v_mov_b32_e32 v151, v77
	v_cndmask_b32_e64 v74, v74, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v81, v149, v81, s54
	v_cndmask_b32_e64 v75, v146, v78, s54
	v_cndmask_b32_e64 v78, v76, v148, s54
	v_cndmask_b32_e64 v76, v150, v147, s54
	v_cndmask_b32_e64 v77, v147, v79, s54
	v_cndmask_b32_e64 v79, v148, v80, s54
	v_cndmask_b32_e64 v80, v151, v149, s54
	v_cndmask_b32_e64 v146, v86, v82, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v87, v83, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v88, v84, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v89, v85, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v83
	v_mov_b32_e32 v151, v85
	v_cndmask_b32_e64 v82, v82, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v89, v149, v89, s54
	v_cndmask_b32_e64 v83, v146, v86, s54
	v_cndmask_b32_e64 v86, v84, v148, s54
	v_cndmask_b32_e64 v84, v150, v147, s54
	v_cndmask_b32_e64 v85, v147, v87, s54
	v_cndmask_b32_e64 v87, v148, v88, s54
	v_cndmask_b32_e64 v88, v151, v149, s54
	v_cndmask_b32_e64 v146, v94, v90, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v95, v91, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v96, v92, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v97, v93, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v91
	v_mov_b32_e32 v151, v93
	v_cndmask_b32_e64 v90, v90, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v97, v149, v97, s54
	v_cndmask_b32_e64 v91, v146, v94, s54
	v_cndmask_b32_e64 v94, v92, v148, s54
	v_cndmask_b32_e64 v92, v150, v147, s54
	v_cndmask_b32_e64 v93, v147, v95, s54
	v_cndmask_b32_e64 v95, v148, v96, s54
	v_cndmask_b32_e64 v96, v151, v149, s54
	v_cndmask_b32_e64 v146, v102, v98, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v103, v99, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v104, v100, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v105, v101, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v99
	v_mov_b32_e32 v151, v101
	v_cndmask_b32_e64 v98, v98, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v105, v149, v105, s54
	v_cndmask_b32_e64 v99, v146, v102, s54
	v_cndmask_b32_e64 v102, v100, v148, s54
	v_cndmask_b32_e64 v100, v150, v147, s54
	v_cndmask_b32_e64 v101, v147, v103, s54
	v_cndmask_b32_e64 v103, v148, v104, s54
	v_cndmask_b32_e64 v104, v151, v149, s54
	v_cndmask_b32_e64 v146, v110, v106, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v111, v107, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v112, v108, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v113, v109, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v107
	v_mov_b32_e32 v151, v109
	v_cndmask_b32_e64 v106, v106, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v113, v149, v113, s54
	v_cndmask_b32_e64 v107, v146, v110, s54
	v_cndmask_b32_e64 v110, v108, v148, s54
	v_cndmask_b32_e64 v108, v150, v147, s54
	v_cndmask_b32_e64 v109, v147, v111, s54
	v_cndmask_b32_e64 v111, v148, v112, s54
	v_cndmask_b32_e64 v112, v151, v149, s54
	v_cndmask_b32_e64 v146, v118, v114, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v119, v115, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v120, v116, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v121, v117, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v115
	v_mov_b32_e32 v151, v117
	v_cndmask_b32_e64 v114, v114, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v121, v149, v121, s54
	v_cndmask_b32_e64 v115, v146, v118, s54
	v_cndmask_b32_e64 v118, v116, v148, s54
	v_cndmask_b32_e64 v116, v150, v147, s54
	v_cndmask_b32_e64 v117, v147, v119, s54
	v_cndmask_b32_e64 v119, v148, v120, s54
	v_cndmask_b32_e64 v120, v151, v149, s54
	v_cndmask_b32_e64 v146, v126, v122, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v127, v123, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v128, v124, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v129, v125, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v123
	v_mov_b32_e32 v151, v125
	v_cndmask_b32_e64 v122, v122, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v129, v149, v129, s54
	v_cndmask_b32_e64 v123, v146, v126, s54
	v_cndmask_b32_e64 v126, v124, v148, s54
	v_cndmask_b32_e64 v124, v150, v147, s54
	v_cndmask_b32_e64 v125, v147, v127, s54
	v_cndmask_b32_e64 v127, v148, v128, s54
	v_cndmask_b32_e64 v128, v151, v149, s54
	v_cndmask_b32_e64 v146, v134, v130, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v135, v131, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v136, v132, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v137, v133, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v131
	v_mov_b32_e32 v151, v133
	v_cndmask_b32_e64 v130, v130, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v137, v149, v137, s54
	v_cndmask_b32_e64 v131, v146, v134, s54
	v_cndmask_b32_e64 v134, v132, v148, s54
	v_cndmask_b32_e64 v132, v150, v147, s54
	v_cndmask_b32_e64 v133, v147, v135, s54
	v_cndmask_b32_e64 v135, v148, v136, s54
	v_cndmask_b32_e64 v136, v151, v149, s54
	v_cndmask_b32_e64 v146, v142, v138, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v146, v146, s50, 0xfedcba98
	v_cndmask_b32_e64 v147, v143, v139, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v147, v147, s50, 0xfedcba98
	v_cndmask_b32_e64 v148, v144, v140, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v148, v148, s50, 0xfedcba98
	v_cndmask_b32_e64 v149, v145, v141, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v149, v149, s50, 0xfedcba98
	v_mov_b32_e32 v150, v139
	v_mov_b32_e32 v151, v141
	v_cndmask_b32_e64 v138, v138, v146, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v145, v149, v145, s54
	v_cndmask_b32_e64 v139, v146, v142, s54
	v_cndmask_b32_e64 v142, v140, v148, s54
	v_cndmask_b32_e64 v140, v150, v147, s54
	v_cndmask_b32_e64 v141, v147, v143, s54
	v_cndmask_b32_e64 v143, v148, v144, s54
	v_cndmask_b32_e64 v144, v151, v149, s54
	s_lshl_b32 s40, s22, 6
	s_mov_b32 s37, 0
	v_mov_b32_e32 v152, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v153, s40, v152
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v154, s40, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v155, s40, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v156, s40, v155
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v157, s40, v156
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v158, s40, v157
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v159, s40, v158
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v160, s40, v159
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v161, s40, v160
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v162, s40, v161
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v163, s40, v162
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v164, s40, v163
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v165, s40, v164
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v166, s40, v165
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v167, s40, v166
	s_and_b32 s35, s22, 7
	s_cmp_lg_u32 s35, 0
	s_cbranch_scc1 .Lmq6g11w4y_scalar
	v_cmp_gt_i32_e64 s46, s22, v9
	s_sub_i32 s41, s24, 0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[168:171], v152, s[56:59], s37 offen
	buffer_load_b128 v[172:175], v152, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 16
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[176:179], v153, s[56:59], s37 offen
	buffer_load_b128 v[180:183], v153, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 32
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[184:187], v154, s[56:59], s37 offen
	buffer_load_b128 v[188:191], v154, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 48
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[192:195], v155, s[56:59], s37 offen
	buffer_load_b128 v[196:199], v155, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 64
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[200:203], v156, s[56:59], s37 offen
	buffer_load_b128 v[204:207], v156, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x50
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[208:211], v157, s[56:59], s37 offen
	buffer_load_b128 v[212:215], v157, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x60
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[216:219], v158, s[56:59], s37 offen
	buffer_load_b128 v[220:223], v158, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x70
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[224:227], v159, s[56:59], s37 offen
	buffer_load_b128 v[228:231], v159, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(15)
	v_add_f32_e32 v18, v18, v168
	v_add_f32_e32 v19, v19, v169
	v_add_f32_e32 v20, v20, v170
	v_add_f32_e32 v21, v21, v171
	s_waitcnt vmcnt(14)
	v_add_f32_e32 v22, v22, v172
	v_add_f32_e32 v23, v23, v173
	v_add_f32_e32 v24, v24, v174
	v_add_f32_e32 v25, v25, v175
	buffer_store_b128 v[18:21], v152, s[56:59], s37 offen
	buffer_store_b128 v[22:25], v152, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 16
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(13)
	v_add_f32_e32 v26, v26, v176
	v_add_f32_e32 v27, v27, v177
	v_add_f32_e32 v28, v28, v178
	v_add_f32_e32 v29, v29, v179
	s_waitcnt vmcnt(12)
	v_add_f32_e32 v30, v30, v180
	v_add_f32_e32 v31, v31, v181
	v_add_f32_e32 v32, v32, v182
	v_add_f32_e32 v33, v33, v183
	buffer_store_b128 v[26:29], v153, s[56:59], s37 offen
	buffer_store_b128 v[30:33], v153, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 32
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(11)
	v_add_f32_e32 v34, v34, v184
	v_add_f32_e32 v35, v35, v185
	v_add_f32_e32 v36, v36, v186
	v_add_f32_e32 v37, v37, v187
	s_waitcnt vmcnt(10)
	v_add_f32_e32 v38, v38, v188
	v_add_f32_e32 v39, v39, v189
	v_add_f32_e32 v40, v40, v190
	v_add_f32_e32 v41, v41, v191
	buffer_store_b128 v[34:37], v154, s[56:59], s37 offen
	buffer_store_b128 v[38:41], v154, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 48
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(9)
	v_add_f32_e32 v42, v42, v192
	v_add_f32_e32 v43, v43, v193
	v_add_f32_e32 v44, v44, v194
	v_add_f32_e32 v45, v45, v195
	s_waitcnt vmcnt(8)
	v_add_f32_e32 v46, v46, v196
	v_add_f32_e32 v47, v47, v197
	v_add_f32_e32 v48, v48, v198
	v_add_f32_e32 v49, v49, v199
	buffer_store_b128 v[42:45], v155, s[56:59], s37 offen
	buffer_store_b128 v[46:49], v155, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 64
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v50, v50, v200
	v_add_f32_e32 v51, v51, v201
	v_add_f32_e32 v52, v52, v202
	v_add_f32_e32 v53, v53, v203
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v54, v54, v204
	v_add_f32_e32 v55, v55, v205
	v_add_f32_e32 v56, v56, v206
	v_add_f32_e32 v57, v57, v207
	buffer_store_b128 v[50:53], v156, s[56:59], s37 offen
	buffer_store_b128 v[54:57], v156, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x50
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v58, v58, v208
	v_add_f32_e32 v59, v59, v209
	v_add_f32_e32 v60, v60, v210
	v_add_f32_e32 v61, v61, v211
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v62, v62, v212
	v_add_f32_e32 v63, v63, v213
	v_add_f32_e32 v64, v64, v214
	v_add_f32_e32 v65, v65, v215
	buffer_store_b128 v[58:61], v157, s[56:59], s37 offen
	buffer_store_b128 v[62:65], v157, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x60
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v66, v66, v216
	v_add_f32_e32 v67, v67, v217
	v_add_f32_e32 v68, v68, v218
	v_add_f32_e32 v69, v69, v219
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v70, v70, v220
	v_add_f32_e32 v71, v71, v221
	v_add_f32_e32 v72, v72, v222
	v_add_f32_e32 v73, v73, v223
	buffer_store_b128 v[66:69], v158, s[56:59], s37 offen
	buffer_store_b128 v[70:73], v158, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x70
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v74, v74, v224
	v_add_f32_e32 v75, v75, v225
	v_add_f32_e32 v76, v76, v226
	v_add_f32_e32 v77, v77, v227
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v78, v78, v228
	v_add_f32_e32 v79, v79, v229
	v_add_f32_e32 v80, v80, v230
	v_add_f32_e32 v81, v81, v231
	buffer_store_b128 v[74:77], v159, s[56:59], s37 offen
	buffer_store_b128 v[78:81], v159, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x80
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[168:171], v160, s[56:59], s37 offen
	buffer_load_b128 v[172:175], v160, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x90
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[176:179], v161, s[56:59], s37 offen
	buffer_load_b128 v[180:183], v161, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xa0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[184:187], v162, s[56:59], s37 offen
	buffer_load_b128 v[188:191], v162, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xb0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[192:195], v163, s[56:59], s37 offen
	buffer_load_b128 v[196:199], v163, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xc0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[200:203], v164, s[56:59], s37 offen
	buffer_load_b128 v[204:207], v164, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xd0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[208:211], v165, s[56:59], s37 offen
	buffer_load_b128 v[212:215], v165, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xe0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[216:219], v166, s[56:59], s37 offen
	buffer_load_b128 v[220:223], v166, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xf0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[224:227], v167, s[56:59], s37 offen
	buffer_load_b128 v[228:231], v167, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x80
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(15)
	v_add_f32_e32 v82, v82, v168
	v_add_f32_e32 v83, v83, v169
	v_add_f32_e32 v84, v84, v170
	v_add_f32_e32 v85, v85, v171
	s_waitcnt vmcnt(14)
	v_add_f32_e32 v86, v86, v172
	v_add_f32_e32 v87, v87, v173
	v_add_f32_e32 v88, v88, v174
	v_add_f32_e32 v89, v89, v175
	buffer_store_b128 v[82:85], v160, s[56:59], s37 offen
	buffer_store_b128 v[86:89], v160, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x90
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(13)
	v_add_f32_e32 v90, v90, v176
	v_add_f32_e32 v91, v91, v177
	v_add_f32_e32 v92, v92, v178
	v_add_f32_e32 v93, v93, v179
	s_waitcnt vmcnt(12)
	v_add_f32_e32 v94, v94, v180
	v_add_f32_e32 v95, v95, v181
	v_add_f32_e32 v96, v96, v182
	v_add_f32_e32 v97, v97, v183
	buffer_store_b128 v[90:93], v161, s[56:59], s37 offen
	buffer_store_b128 v[94:97], v161, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xa0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(11)
	v_add_f32_e32 v98, v98, v184
	v_add_f32_e32 v99, v99, v185
	v_add_f32_e32 v100, v100, v186
	v_add_f32_e32 v101, v101, v187
	s_waitcnt vmcnt(10)
	v_add_f32_e32 v102, v102, v188
	v_add_f32_e32 v103, v103, v189
	v_add_f32_e32 v104, v104, v190
	v_add_f32_e32 v105, v105, v191
	buffer_store_b128 v[98:101], v162, s[56:59], s37 offen
	buffer_store_b128 v[102:105], v162, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xb0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(9)
	v_add_f32_e32 v106, v106, v192
	v_add_f32_e32 v107, v107, v193
	v_add_f32_e32 v108, v108, v194
	v_add_f32_e32 v109, v109, v195
	s_waitcnt vmcnt(8)
	v_add_f32_e32 v110, v110, v196
	v_add_f32_e32 v111, v111, v197
	v_add_f32_e32 v112, v112, v198
	v_add_f32_e32 v113, v113, v199
	buffer_store_b128 v[106:109], v163, s[56:59], s37 offen
	buffer_store_b128 v[110:113], v163, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xc0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v114, v114, v200
	v_add_f32_e32 v115, v115, v201
	v_add_f32_e32 v116, v116, v202
	v_add_f32_e32 v117, v117, v203
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v118, v118, v204
	v_add_f32_e32 v119, v119, v205
	v_add_f32_e32 v120, v120, v206
	v_add_f32_e32 v121, v121, v207
	buffer_store_b128 v[114:117], v164, s[56:59], s37 offen
	buffer_store_b128 v[118:121], v164, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xd0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v122, v122, v208
	v_add_f32_e32 v123, v123, v209
	v_add_f32_e32 v124, v124, v210
	v_add_f32_e32 v125, v125, v211
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v126, v126, v212
	v_add_f32_e32 v127, v127, v213
	v_add_f32_e32 v128, v128, v214
	v_add_f32_e32 v129, v129, v215
	buffer_store_b128 v[122:125], v165, s[56:59], s37 offen
	buffer_store_b128 v[126:129], v165, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xe0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v130, v130, v216
	v_add_f32_e32 v131, v131, v217
	v_add_f32_e32 v132, v132, v218
	v_add_f32_e32 v133, v133, v219
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v134, v134, v220
	v_add_f32_e32 v135, v135, v221
	v_add_f32_e32 v136, v136, v222
	v_add_f32_e32 v137, v137, v223
	buffer_store_b128 v[130:133], v166, s[56:59], s37 offen
	buffer_store_b128 v[134:137], v166, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xf0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v138, v138, v224
	v_add_f32_e32 v139, v139, v225
	v_add_f32_e32 v140, v140, v226
	v_add_f32_e32 v141, v141, v227
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v142, v142, v228
	v_add_f32_e32 v143, v143, v229
	v_add_f32_e32 v144, v144, v230
	v_add_f32_e32 v145, v145, v231
	buffer_store_b128 v[138:141], v167, s[56:59], s37 offen
	buffer_store_b128 v[142:145], v167, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_waitcnt_vscnt null, 0x0
	s_branch .Lmq6g11w4y_end
	.Lmq6g11w4y_scalar:
	s_sub_i32 s41, s24, 0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v152, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v152, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v152, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v152, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v152, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v152, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v152, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v152, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v18, v18, v168
	buffer_store_b32 v18, v152, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v19, v19, v169
	buffer_store_b32 v19, v152, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v20, v20, v170
	buffer_store_b32 v20, v152, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v21, v21, v171
	buffer_store_b32 v21, v152, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v22, v22, v172
	buffer_store_b32 v22, v152, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v23, v23, v173
	buffer_store_b32 v23, v152, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v24, v24, v174
	buffer_store_b32 v24, v152, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v25, v25, v175
	buffer_store_b32 v25, v152, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 16
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v153, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v153, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v153, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v153, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v153, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v153, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v153, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v153, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v26, v26, v168
	buffer_store_b32 v26, v153, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v27, v27, v169
	buffer_store_b32 v27, v153, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v28, v28, v170
	buffer_store_b32 v28, v153, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v29, v29, v171
	buffer_store_b32 v29, v153, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v30, v30, v172
	buffer_store_b32 v30, v153, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v31, v31, v173
	buffer_store_b32 v31, v153, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v32, v32, v174
	buffer_store_b32 v32, v153, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v33, v33, v175
	buffer_store_b32 v33, v153, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 32
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v154, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v154, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v154, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v154, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v154, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v154, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v154, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v154, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v34, v34, v168
	buffer_store_b32 v34, v154, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v35, v35, v169
	buffer_store_b32 v35, v154, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v36, v36, v170
	buffer_store_b32 v36, v154, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v37, v37, v171
	buffer_store_b32 v37, v154, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v38, v38, v172
	buffer_store_b32 v38, v154, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v39, v39, v173
	buffer_store_b32 v39, v154, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v40, v40, v174
	buffer_store_b32 v40, v154, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v41, v41, v175
	buffer_store_b32 v41, v154, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 48
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v155, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v155, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v155, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v155, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v155, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v155, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v155, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v155, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v42, v42, v168
	buffer_store_b32 v42, v155, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v43, v43, v169
	buffer_store_b32 v43, v155, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v44, v44, v170
	buffer_store_b32 v44, v155, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v45, v45, v171
	buffer_store_b32 v45, v155, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v46, v46, v172
	buffer_store_b32 v46, v155, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v47, v47, v173
	buffer_store_b32 v47, v155, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v48, v48, v174
	buffer_store_b32 v48, v155, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v49, v49, v175
	buffer_store_b32 v49, v155, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 64
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v156, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v156, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v156, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v156, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v156, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v156, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v156, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v156, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v50, v50, v168
	buffer_store_b32 v50, v156, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v51, v51, v169
	buffer_store_b32 v51, v156, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v52, v52, v170
	buffer_store_b32 v52, v156, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v53, v53, v171
	buffer_store_b32 v53, v156, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v54, v54, v172
	buffer_store_b32 v54, v156, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v55, v55, v173
	buffer_store_b32 v55, v156, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v56, v56, v174
	buffer_store_b32 v56, v156, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v57, v57, v175
	buffer_store_b32 v57, v156, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0x50
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v157, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v157, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v157, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v157, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v157, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v157, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v157, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v157, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v58, v58, v168
	buffer_store_b32 v58, v157, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v59, v59, v169
	buffer_store_b32 v59, v157, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v60, v60, v170
	buffer_store_b32 v60, v157, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v61, v61, v171
	buffer_store_b32 v61, v157, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v62, v62, v172
	buffer_store_b32 v62, v157, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v63, v63, v173
	buffer_store_b32 v63, v157, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v64, v64, v174
	buffer_store_b32 v64, v157, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v65, v65, v175
	buffer_store_b32 v65, v157, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0x60
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v158, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v158, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v158, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v158, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v158, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v158, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v158, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v158, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v66, v66, v168
	buffer_store_b32 v66, v158, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v67, v67, v169
	buffer_store_b32 v67, v158, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v68, v68, v170
	buffer_store_b32 v68, v158, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v69, v69, v171
	buffer_store_b32 v69, v158, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v70, v70, v172
	buffer_store_b32 v70, v158, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v71, v71, v173
	buffer_store_b32 v71, v158, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v72, v72, v174
	buffer_store_b32 v72, v158, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v73, v73, v175
	buffer_store_b32 v73, v158, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0x70
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v159, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v159, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v159, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v159, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v159, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v159, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v159, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v159, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v74, v74, v168
	buffer_store_b32 v74, v159, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v75, v75, v169
	buffer_store_b32 v75, v159, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v76, v76, v170
	buffer_store_b32 v76, v159, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v77, v77, v171
	buffer_store_b32 v77, v159, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v78, v78, v172
	buffer_store_b32 v78, v159, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v79, v79, v173
	buffer_store_b32 v79, v159, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v80, v80, v174
	buffer_store_b32 v80, v159, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v81, v81, v175
	buffer_store_b32 v81, v159, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0x80
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v160, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v160, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v160, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v160, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v160, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v160, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v160, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v160, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v82, v82, v168
	buffer_store_b32 v82, v160, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v83, v83, v169
	buffer_store_b32 v83, v160, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v84, v84, v170
	buffer_store_b32 v84, v160, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v85, v85, v171
	buffer_store_b32 v85, v160, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v86, v86, v172
	buffer_store_b32 v86, v160, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v87, v87, v173
	buffer_store_b32 v87, v160, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v88, v88, v174
	buffer_store_b32 v88, v160, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v89, v89, v175
	buffer_store_b32 v89, v160, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0x90
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v161, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v161, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v161, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v161, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v161, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v161, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v161, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v161, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v90, v90, v168
	buffer_store_b32 v90, v161, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v91, v91, v169
	buffer_store_b32 v91, v161, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v92, v92, v170
	buffer_store_b32 v92, v161, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v93, v93, v171
	buffer_store_b32 v93, v161, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v94, v94, v172
	buffer_store_b32 v94, v161, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v95, v95, v173
	buffer_store_b32 v95, v161, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v96, v96, v174
	buffer_store_b32 v96, v161, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v97, v97, v175
	buffer_store_b32 v97, v161, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xa0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v162, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v162, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v162, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v162, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v162, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v162, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v162, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v162, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v98, v98, v168
	buffer_store_b32 v98, v162, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v99, v99, v169
	buffer_store_b32 v99, v162, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v100, v100, v170
	buffer_store_b32 v100, v162, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v101, v101, v171
	buffer_store_b32 v101, v162, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v102, v102, v172
	buffer_store_b32 v102, v162, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v103, v103, v173
	buffer_store_b32 v103, v162, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v104, v104, v174
	buffer_store_b32 v104, v162, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v105, v105, v175
	buffer_store_b32 v105, v162, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xb0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v163, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v163, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v163, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v163, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v163, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v163, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v163, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v163, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v106, v106, v168
	buffer_store_b32 v106, v163, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v107, v107, v169
	buffer_store_b32 v107, v163, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v108, v108, v170
	buffer_store_b32 v108, v163, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v109, v109, v171
	buffer_store_b32 v109, v163, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v110, v110, v172
	buffer_store_b32 v110, v163, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v111, v111, v173
	buffer_store_b32 v111, v163, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v112, v112, v174
	buffer_store_b32 v112, v163, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v113, v113, v175
	buffer_store_b32 v113, v163, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xc0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v164, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v164, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v164, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v164, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v164, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v164, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v164, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v164, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v114, v114, v168
	buffer_store_b32 v114, v164, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v115, v115, v169
	buffer_store_b32 v115, v164, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v116, v116, v170
	buffer_store_b32 v116, v164, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v117, v117, v171
	buffer_store_b32 v117, v164, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v118, v118, v172
	buffer_store_b32 v118, v164, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v119, v119, v173
	buffer_store_b32 v119, v164, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v120, v120, v174
	buffer_store_b32 v120, v164, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v121, v121, v175
	buffer_store_b32 v121, v164, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xd0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v165, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v165, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v165, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v165, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v165, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v165, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v165, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v165, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v122, v122, v168
	buffer_store_b32 v122, v165, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v123, v123, v169
	buffer_store_b32 v123, v165, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v124, v124, v170
	buffer_store_b32 v124, v165, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v125, v125, v171
	buffer_store_b32 v125, v165, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v126, v126, v172
	buffer_store_b32 v126, v165, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v127, v127, v173
	buffer_store_b32 v127, v165, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v128, v128, v174
	buffer_store_b32 v128, v165, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v129, v129, v175
	buffer_store_b32 v129, v165, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xe0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v166, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v166, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v166, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v166, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v166, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v166, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v166, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v166, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v130, v130, v168
	buffer_store_b32 v130, v166, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v131, v131, v169
	buffer_store_b32 v131, v166, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v132, v132, v170
	buffer_store_b32 v132, v166, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v133, v133, v171
	buffer_store_b32 v133, v166, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v134, v134, v172
	buffer_store_b32 v134, v166, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v135, v135, v173
	buffer_store_b32 v135, v166, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v136, v136, v174
	buffer_store_b32 v136, v166, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v137, v137, v175
	buffer_store_b32 v137, v166, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xf0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v167, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v167, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v167, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v167, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v172, v167, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v173, v167, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v174, v167, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v175, v167, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v138, v138, v168
	buffer_store_b32 v138, v167, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v139, v139, v169
	buffer_store_b32 v139, v167, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v140, v140, v170
	buffer_store_b32 v140, v167, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v141, v141, v171
	buffer_store_b32 v141, v167, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v142, v142, v172
	buffer_store_b32 v142, v167, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v143, v143, v173
	buffer_store_b32 v143, v167, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v144, v144, v174
	buffer_store_b32 v144, v167, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v145, v145, v175
	buffer_store_b32 v145, v167, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	.Lmq6g11w4y_end:
	s_endpgm
.Lgemm_mq6g256v2_residual_pm_gfx1100_w4_end:
.size gemm_mq6g256v2_residual_pm_gfx1100_w4, .Lgemm_mq6g256v2_residual_pm_gfx1100_w4_end-gemm_mq6g256v2_residual_pm_gfx1100_w4
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemm_mq6g256v2_residual_pm_gfx1100_w4
	.amdhsa_group_segment_fixed_size 40960
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
	.amdhsa_next_free_vgpr 256
	.amdhsa_next_free_sgpr 84
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemm_mq6g256v2_residual_pm_gfx1100_w4_end-gemm_mq6g256v2_residual_pm_gfx1100_w4)<<4)&1008)>>4
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
.protected gemm_mq6g256v2_residual_pm_gfx1100_w8
.globl gemm_mq6g256v2_residual_pm_gfx1100_w8
.p2align 8
.type gemm_mq6g256v2_residual_pm_gfx1100_w8,@function
gemm_mq6g256v2_residual_pm_gfx1100_w8:
	s_load_b256 s[16:23], s[0:1], null
	s_load_b32 s24, s[0:1], 0x20
	v_and_b32_e32 v142, 15, v0
	v_bfe_u32 v143, v0, 4, 1
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v6, 0x50, v142
	v_lshrrev_b32_e32 v145, 2, v0
	v_and_b32_e32 v146, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v146, 4, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v5, v145, 0x50, v146
	v_cmp_ne_u32_e64 s54, 0, v143
	v_mov_b32_e32 v147, 0xc0c0100
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v147, 0xc0c0302, s54
	v_mov_b32_e32 v147, 0xc020201
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v2, v147, 0xc040403, s54
	v_mov_b32_e32 v147, 0xc0c0403
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v218, v147, 0xc0c0605, s54
	v_mov_b32_e32 v147, 0xc050504
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v219, v147, 0xc070706, s54
	s_mov_b32 s50, 0x76543210
	s_mov_b32 s80, 0x3f003f
	s_mov_b32 s81, 0xfc03f0
	s_mov_b32 s82, 0xe400e400
	s_mov_b32 s83, 0xdc00d400
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
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
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	s_waitcnt lgkmcnt(0)
	s_min_i32 s35, s22, s23
	s_min_i32 s35, s35, s24
	s_cmp_lt_i32 s35, 1
	s_cbranch_scc1 .Lmq6g11w8y_end
	s_mov_b32 s28, s14
	s_mov_b32 s29, s15
	s_lshr_b32 s25, s23, 8
	s_lshr_b32 s26, s23, 5
	s_add_i32 s26, s26, -1
	s_add_i32 s27, s25, -1
	s_lshl_b32 s43, s23, 1
	s_mul_i32 s36, s24, s43
	s_mov_b32 s4, s16
	s_and_b32 s5, s17, 0xffff
	s_brev_b32 s6, -2
	s_mov_b32 s7, 0x31004000
	s_mov_b32 s8, s18
	s_and_b32 s9, s19, 0xffff
	s_mov_b32 s10, s36
	s_mov_b32 s11, 0x31004000
	s_mov_b32 s56, s20
	s_and_b32 s57, s21, 0xffff
	s_brev_b32 s58, -2
	s_mov_b32 s59, 0x31004000
	v_lshrrev_b32_e32 v145, 1, v0
	s_mul_i32 s35, s28, 0x80
	s_delay_alu instid0(VALU_DEP_1)
	v_and_or_b32 v145, v145, 0x70, s35
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v9, v143, 3, v145
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v145, v145, v142
	s_add_i32 s35, s22, -1
	s_mul_i32 s36, s25, 0xc8
	s_delay_alu instid0(VALU_DEP_1)
	v_min_i32_e32 v145, s35, v145
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v4, v145, s36
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v3, v143, 2, v4
	s_lshl_b32 s35, s29, 8
	v_add_nc_u32_e32 v7, s35, v142
	v_lshrrev_b32_e32 v144, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v144, s35, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v10, 0, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v10, v10, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v10, v10, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v11, 64, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v11, v11, v146
	v_add_nc_u32_e32 v12, 0x80, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v12, v12, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v12, v12, v146
	v_add_nc_u32_e32 v13, 0xc0, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v13, v13, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v13, v13, v146
	v_mul_lo_u32 v8, v7, s22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 2, v8
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmq6g11w8y_epi
	s_mov_b32 s30, 0
	s_mov_b32 s31, 0
	s_mov_b32 s34, 0
	s_mov_b32 s33, 0
	buffer_load_b64 v[206:207], v4, s[4:7], s31 offen
	buffer_load_b64 v[190:191], v3, s[4:7], s31 offen offset:8
	buffer_load_b64 v[192:193], v3, s[4:7], s31 offen offset:20
	buffer_load_b128 v[220:223], v10, s[8:11], s33 offen
	buffer_load_b128 v[224:227], v11, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v12, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v13, s[8:11], s33 offen
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[220:223]
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[224:227] offset:5120
	s_waitcnt vmcnt(1)
	ds_store_b128 v5, v[228:231] offset:10240
	s_waitcnt vmcnt(0)
	ds_store_b128 v5, v[232:235] offset:15360
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b64 v[194:195], v3, s[4:7], s31 offen offset:32
	buffer_load_b64 v[196:197], v3, s[4:7], s31 offen offset:44
	s_min_u32 s33, s26, 1
	s_lshl_b32 s33, s33, 6
	buffer_load_b128 v[220:223], v10, s[8:11], s33 offen
	buffer_load_b128 v[224:227], v11, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v12, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v13, s[8:11], s33 offen
	v_perm_b32 v210, v191, v190, v1
	v_perm_b32 v211, v191, v190, v2
	v_perm_b32 v212, v191, v190, v218
	v_perm_b32 v213, v191, v190, v219
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_or_b32 v212, v212, 10, v212
	s_delay_alu instid0(VALU_DEP_2)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f16 v210, v210, s82
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f16 v211, v211, s83
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f16 v212, v212, s82
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f16 v213, v213, s83
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_fma_f16 v210, v206, v210, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_fma_f16 v211, v206, v211, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_fma_f16 v212, v206, v212, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_fma_f16 v213, v206, v213, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_delay_alu instid0(VALU_DEP_4)
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4)
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4)
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4)
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v174, v210, v214, s54
	v_cndmask_b32_e64 v178, v214, v210, s54
	v_cndmask_b32_e64 v175, v211, v215, s54
	v_cndmask_b32_e64 v179, v215, v211, s54
	v_cndmask_b32_e64 v176, v212, v216, s54
	v_cndmask_b32_e64 v180, v216, v212, s54
	v_cndmask_b32_e64 v177, v213, v217, s54
	v_cndmask_b32_e64 v181, v217, v213, s54
	ds_load_b128 v[142:145], v6
	ds_load_b128 v[146:149], v6 offset:16
	ds_load_b128 v[150:153], v6 offset:1280
	ds_load_b128 v[154:157], v6 offset:1296
	ds_load_b128 v[158:161], v6 offset:2560
	ds_load_b128 v[162:165], v6 offset:2576
	ds_load_b128 v[166:169], v6 offset:3840
	ds_load_b128 v[170:173], v6 offset:3856
	.Lmq6g11w8y_loop:
	s_add_i32 s32, s30, 1
	s_min_u32 s32, s32, s27
	s_mul_i32 s32, s32, 0xc8
	s_add_i32 s33, s34, 2
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[174:181], v[142:149], v[14:21]
	buffer_load_b64 v[198:199], v3, s[4:7], s31 offen offset:56
	buffer_load_b64 v[200:201], v3, s[4:7], s31 offen offset:68
	ds_load_b128 v[142:145], v6 offset:5120
	ds_load_b128 v[146:149], v6 offset:5136
	v_perm_b32 v210, v193, v192, v1
	v_perm_b32 v211, v193, v192, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[174:181], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:6400
	ds_load_b128 v[154:157], v6 offset:6416
	v_perm_b32 v212, v193, v192, v218
	v_perm_b32 v213, v193, v192, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[174:181], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:7680
	ds_load_b128 v[162:165], v6 offset:7696
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[38:45], v[174:181], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:8960
	ds_load_b128 v[170:173], v6 offset:8976
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[46:53], v[174:181], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:10240
	ds_load_b128 v[146:149], v6 offset:10256
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[54:61], v[174:181], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:11520
	ds_load_b128 v[154:157], v6 offset:11536
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[62:69], v[174:181], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:12800
	ds_load_b128 v[162:165], v6 offset:12816
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[70:77], v[174:181], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:14080
	ds_load_b128 v[170:173], v6 offset:14096
	v_pk_fma_f16 v210, v206, v210, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v206, v211, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[78:85], v[174:181], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:15360
	ds_load_b128 v[146:149], v6 offset:15376
	v_pk_fma_f16 v212, v206, v212, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v206, v213, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[86:93], v[174:181], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:16640
	ds_load_b128 v[154:157], v6 offset:16656
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[94:101], v[174:181], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:17920
	ds_load_b128 v[162:165], v6 offset:17936
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[102:109], v[174:181], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:19200
	ds_load_b128 v[170:173], v6 offset:19216
	v_cndmask_b32_e64 v182, v210, v214, s54
	v_cndmask_b32_e64 v186, v214, v210, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[110:117], v[174:181], v[142:149], v[110:117]
	ds_load_b128 v[142:145], v6 offset:32
	ds_load_b128 v[146:149], v6 offset:48
	v_cndmask_b32_e64 v183, v211, v215, s54
	v_cndmask_b32_e64 v187, v215, v211, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[118:125], v[174:181], v[150:157], v[118:125]
	ds_load_b128 v[150:153], v6 offset:1312
	ds_load_b128 v[154:157], v6 offset:1328
	v_cndmask_b32_e64 v184, v212, v216, s54
	v_cndmask_b32_e64 v188, v216, v212, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[126:133], v[174:181], v[158:165], v[126:133]
	ds_load_b128 v[158:161], v6 offset:2592
	ds_load_b128 v[162:165], v6 offset:2608
	v_cndmask_b32_e64 v185, v213, v217, s54
	v_cndmask_b32_e64 v189, v217, v213, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[134:141], v[174:181], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:3872
	ds_load_b128 v[170:173], v6 offset:3888
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[182:189], v[142:149], v[14:21]
	ds_load_b128 v[142:145], v6 offset:5152
	ds_load_b128 v[146:149], v6 offset:5168
	s_waitcnt vmcnt(7)
	v_perm_b32 v210, v195, v194, v1
	v_perm_b32 v211, v195, v194, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[182:189], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:6432
	ds_load_b128 v[154:157], v6 offset:6448
	v_perm_b32 v212, v195, v194, v218
	v_perm_b32 v213, v195, v194, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[182:189], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:7712
	ds_load_b128 v[162:165], v6 offset:7728
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[220:223] offset:20480
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[38:45], v[182:189], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:8992
	ds_load_b128 v[170:173], v6 offset:9008
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[46:53], v[182:189], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:10272
	ds_load_b128 v[146:149], v6 offset:10288
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[224:227] offset:25600
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[54:61], v[182:189], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:11552
	ds_load_b128 v[154:157], v6 offset:11568
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[62:69], v[182:189], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:12832
	ds_load_b128 v[162:165], v6 offset:12848
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[228:231] offset:30720
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[70:77], v[182:189], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:14112
	ds_load_b128 v[170:173], v6 offset:14128
	v_pk_fma_f16 v210, v206, v210, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v206, v211, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[78:85], v[182:189], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:15392
	ds_load_b128 v[146:149], v6 offset:15408
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[232:235] offset:35840
	v_pk_fma_f16 v212, v206, v212, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v206, v213, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[86:93], v[182:189], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:16672
	ds_load_b128 v[154:157], v6 offset:16688
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[94:101], v[182:189], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:17952
	ds_load_b128 v[162:165], v6 offset:17968
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[102:109], v[182:189], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:19232
	ds_load_b128 v[170:173], v6 offset:19248
	v_cndmask_b32_e64 v174, v210, v214, s54
	v_cndmask_b32_e64 v178, v214, v210, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[220:223], v10, s[8:11], s33 offen
	buffer_load_b128 v[224:227], v11, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v12, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v13, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[110:117], v[182:189], v[142:149], v[110:117]
	v_cndmask_b32_e64 v175, v211, v215, s54
	v_cndmask_b32_e64 v179, v215, v211, s54
	ds_load_b128 v[142:145], v6 offset:20480
	ds_load_b128 v[146:149], v6 offset:20496
	v_wmma_f32_16x16x16_f16 v[118:125], v[182:189], v[150:157], v[118:125]
	v_cndmask_b32_e64 v176, v212, v216, s54
	v_cndmask_b32_e64 v180, v216, v212, s54
	ds_load_b128 v[150:153], v6 offset:21760
	ds_load_b128 v[154:157], v6 offset:21776
	v_wmma_f32_16x16x16_f16 v[126:133], v[182:189], v[158:165], v[126:133]
	v_cndmask_b32_e64 v177, v213, v217, s54
	v_cndmask_b32_e64 v181, v217, v213, s54
	ds_load_b128 v[158:161], v6 offset:23040
	ds_load_b128 v[162:165], v6 offset:23056
	v_wmma_f32_16x16x16_f16 v[134:141], v[182:189], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:24320
	ds_load_b128 v[170:173], v6 offset:24336
	s_add_i32 s33, s34, 3
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[174:181], v[142:149], v[14:21]
	buffer_load_b64 v[202:203], v3, s[4:7], s31 offen offset:80
	buffer_load_b64 v[204:205], v3, s[4:7], s31 offen offset:92
	ds_load_b128 v[142:145], v6 offset:25600
	ds_load_b128 v[146:149], v6 offset:25616
	v_perm_b32 v210, v197, v196, v1
	v_perm_b32 v211, v197, v196, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[174:181], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:26880
	ds_load_b128 v[154:157], v6 offset:26896
	v_perm_b32 v212, v197, v196, v218
	v_perm_b32 v213, v197, v196, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[174:181], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:28160
	ds_load_b128 v[162:165], v6 offset:28176
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[38:45], v[174:181], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:29440
	ds_load_b128 v[170:173], v6 offset:29456
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[46:53], v[174:181], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:30720
	ds_load_b128 v[146:149], v6 offset:30736
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[54:61], v[174:181], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:32000
	ds_load_b128 v[154:157], v6 offset:32016
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[62:69], v[174:181], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:33280
	ds_load_b128 v[162:165], v6 offset:33296
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[70:77], v[174:181], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:34560
	ds_load_b128 v[170:173], v6 offset:34576
	v_pk_fma_f16 v210, v206, v210, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v206, v211, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[78:85], v[174:181], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:35840
	ds_load_b128 v[146:149], v6 offset:35856
	v_pk_fma_f16 v212, v206, v212, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v206, v213, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[86:93], v[174:181], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:37120
	ds_load_b128 v[154:157], v6 offset:37136
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[94:101], v[174:181], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:38400
	ds_load_b128 v[162:165], v6 offset:38416
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[102:109], v[174:181], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:39680
	ds_load_b128 v[170:173], v6 offset:39696
	v_cndmask_b32_e64 v182, v210, v214, s54
	v_cndmask_b32_e64 v186, v214, v210, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[110:117], v[174:181], v[142:149], v[110:117]
	ds_load_b128 v[142:145], v6 offset:20512
	ds_load_b128 v[146:149], v6 offset:20528
	v_cndmask_b32_e64 v183, v211, v215, s54
	v_cndmask_b32_e64 v187, v215, v211, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[118:125], v[174:181], v[150:157], v[118:125]
	ds_load_b128 v[150:153], v6 offset:21792
	ds_load_b128 v[154:157], v6 offset:21808
	v_cndmask_b32_e64 v184, v212, v216, s54
	v_cndmask_b32_e64 v188, v216, v212, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[126:133], v[174:181], v[158:165], v[126:133]
	ds_load_b128 v[158:161], v6 offset:23072
	ds_load_b128 v[162:165], v6 offset:23088
	v_cndmask_b32_e64 v185, v213, v217, s54
	v_cndmask_b32_e64 v189, v217, v213, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[134:141], v[174:181], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:24352
	ds_load_b128 v[170:173], v6 offset:24368
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[182:189], v[142:149], v[14:21]
	ds_load_b128 v[142:145], v6 offset:25632
	ds_load_b128 v[146:149], v6 offset:25648
	s_waitcnt vmcnt(7)
	v_perm_b32 v210, v199, v198, v1
	v_perm_b32 v211, v199, v198, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[182:189], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:26912
	ds_load_b128 v[154:157], v6 offset:26928
	v_perm_b32 v212, v199, v198, v218
	v_perm_b32 v213, v199, v198, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[182:189], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:28192
	ds_load_b128 v[162:165], v6 offset:28208
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[220:223]
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[38:45], v[182:189], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:29472
	ds_load_b128 v[170:173], v6 offset:29488
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[46:53], v[182:189], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:30752
	ds_load_b128 v[146:149], v6 offset:30768
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[224:227] offset:5120
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[54:61], v[182:189], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:32032
	ds_load_b128 v[154:157], v6 offset:32048
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[62:69], v[182:189], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:33312
	ds_load_b128 v[162:165], v6 offset:33328
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[228:231] offset:10240
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[70:77], v[182:189], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:34592
	ds_load_b128 v[170:173], v6 offset:34608
	v_pk_fma_f16 v210, v206, v210, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v206, v211, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[78:85], v[182:189], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:35872
	ds_load_b128 v[146:149], v6 offset:35888
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[232:235] offset:15360
	v_pk_fma_f16 v212, v206, v212, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v206, v213, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[86:93], v[182:189], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:37152
	ds_load_b128 v[154:157], v6 offset:37168
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[94:101], v[182:189], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:38432
	ds_load_b128 v[162:165], v6 offset:38448
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[102:109], v[182:189], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:39712
	ds_load_b128 v[170:173], v6 offset:39728
	v_cndmask_b32_e64 v174, v210, v214, s54
	v_cndmask_b32_e64 v178, v214, v210, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[220:223], v10, s[8:11], s33 offen
	buffer_load_b128 v[224:227], v11, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v12, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v13, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[110:117], v[182:189], v[142:149], v[110:117]
	v_cndmask_b32_e64 v175, v211, v215, s54
	v_cndmask_b32_e64 v179, v215, v211, s54
	ds_load_b128 v[142:145], v6
	ds_load_b128 v[146:149], v6 offset:16
	v_wmma_f32_16x16x16_f16 v[118:125], v[182:189], v[150:157], v[118:125]
	v_cndmask_b32_e64 v176, v212, v216, s54
	v_cndmask_b32_e64 v180, v216, v212, s54
	ds_load_b128 v[150:153], v6 offset:1280
	ds_load_b128 v[154:157], v6 offset:1296
	v_wmma_f32_16x16x16_f16 v[126:133], v[182:189], v[158:165], v[126:133]
	v_cndmask_b32_e64 v177, v213, v217, s54
	v_cndmask_b32_e64 v181, v217, v213, s54
	ds_load_b128 v[158:161], v6 offset:2560
	ds_load_b128 v[162:165], v6 offset:2576
	v_wmma_f32_16x16x16_f16 v[134:141], v[182:189], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:3840
	ds_load_b128 v[170:173], v6 offset:3856
	s_add_i32 s33, s34, 4
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[174:181], v[142:149], v[14:21]
	buffer_load_b64 v[190:191], v3, s[4:7], s31 offen offset:104
	buffer_load_b64 v[192:193], v3, s[4:7], s31 offen offset:116
	ds_load_b128 v[142:145], v6 offset:5120
	ds_load_b128 v[146:149], v6 offset:5136
	v_perm_b32 v210, v201, v200, v1
	v_perm_b32 v211, v201, v200, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[174:181], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:6400
	ds_load_b128 v[154:157], v6 offset:6416
	v_perm_b32 v212, v201, v200, v218
	v_perm_b32 v213, v201, v200, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[174:181], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:7680
	ds_load_b128 v[162:165], v6 offset:7696
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[38:45], v[174:181], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:8960
	ds_load_b128 v[170:173], v6 offset:8976
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[46:53], v[174:181], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:10240
	ds_load_b128 v[146:149], v6 offset:10256
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[54:61], v[174:181], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:11520
	ds_load_b128 v[154:157], v6 offset:11536
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[62:69], v[174:181], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:12800
	ds_load_b128 v[162:165], v6 offset:12816
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[70:77], v[174:181], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:14080
	ds_load_b128 v[170:173], v6 offset:14096
	v_pk_fma_f16 v210, v206, v210, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v206, v211, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[78:85], v[174:181], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:15360
	ds_load_b128 v[146:149], v6 offset:15376
	v_pk_fma_f16 v212, v206, v212, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v206, v213, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[86:93], v[174:181], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:16640
	ds_load_b128 v[154:157], v6 offset:16656
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[94:101], v[174:181], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:17920
	ds_load_b128 v[162:165], v6 offset:17936
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[102:109], v[174:181], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:19200
	ds_load_b128 v[170:173], v6 offset:19216
	v_cndmask_b32_e64 v182, v210, v214, s54
	v_cndmask_b32_e64 v186, v214, v210, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[110:117], v[174:181], v[142:149], v[110:117]
	ds_load_b128 v[142:145], v6 offset:32
	ds_load_b128 v[146:149], v6 offset:48
	v_cndmask_b32_e64 v183, v211, v215, s54
	v_cndmask_b32_e64 v187, v215, v211, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[118:125], v[174:181], v[150:157], v[118:125]
	ds_load_b128 v[150:153], v6 offset:1312
	ds_load_b128 v[154:157], v6 offset:1328
	v_cndmask_b32_e64 v184, v212, v216, s54
	v_cndmask_b32_e64 v188, v216, v212, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[126:133], v[174:181], v[158:165], v[126:133]
	ds_load_b128 v[158:161], v6 offset:2592
	ds_load_b128 v[162:165], v6 offset:2608
	v_cndmask_b32_e64 v185, v213, v217, s54
	v_cndmask_b32_e64 v189, v217, v213, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[134:141], v[174:181], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:3872
	ds_load_b128 v[170:173], v6 offset:3888
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[182:189], v[142:149], v[14:21]
	ds_load_b128 v[142:145], v6 offset:5152
	ds_load_b128 v[146:149], v6 offset:5168
	s_waitcnt vmcnt(7)
	v_perm_b32 v210, v203, v202, v1
	v_perm_b32 v211, v203, v202, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[182:189], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:6432
	ds_load_b128 v[154:157], v6 offset:6448
	v_perm_b32 v212, v203, v202, v218
	v_perm_b32 v213, v203, v202, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[182:189], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:7712
	ds_load_b128 v[162:165], v6 offset:7728
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[220:223] offset:20480
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[38:45], v[182:189], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:8992
	ds_load_b128 v[170:173], v6 offset:9008
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[46:53], v[182:189], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:10272
	ds_load_b128 v[146:149], v6 offset:10288
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[224:227] offset:25600
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[54:61], v[182:189], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:11552
	ds_load_b128 v[154:157], v6 offset:11568
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[62:69], v[182:189], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:12832
	ds_load_b128 v[162:165], v6 offset:12848
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[228:231] offset:30720
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[70:77], v[182:189], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:14112
	ds_load_b128 v[170:173], v6 offset:14128
	v_pk_fma_f16 v210, v206, v210, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v206, v211, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[78:85], v[182:189], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:15392
	ds_load_b128 v[146:149], v6 offset:15408
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[232:235] offset:35840
	v_pk_fma_f16 v212, v206, v212, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v206, v213, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[86:93], v[182:189], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:16672
	ds_load_b128 v[154:157], v6 offset:16688
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[94:101], v[182:189], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:17952
	ds_load_b128 v[162:165], v6 offset:17968
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[102:109], v[182:189], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:19232
	ds_load_b128 v[170:173], v6 offset:19248
	v_cndmask_b32_e64 v174, v210, v214, s54
	v_cndmask_b32_e64 v178, v214, v210, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[220:223], v10, s[8:11], s33 offen
	buffer_load_b128 v[224:227], v11, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v12, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v13, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[110:117], v[182:189], v[142:149], v[110:117]
	v_cndmask_b32_e64 v175, v211, v215, s54
	v_cndmask_b32_e64 v179, v215, v211, s54
	ds_load_b128 v[142:145], v6 offset:20480
	ds_load_b128 v[146:149], v6 offset:20496
	v_wmma_f32_16x16x16_f16 v[118:125], v[182:189], v[150:157], v[118:125]
	v_cndmask_b32_e64 v176, v212, v216, s54
	v_cndmask_b32_e64 v180, v216, v212, s54
	ds_load_b128 v[150:153], v6 offset:21760
	ds_load_b128 v[154:157], v6 offset:21776
	v_wmma_f32_16x16x16_f16 v[126:133], v[182:189], v[158:165], v[126:133]
	v_cndmask_b32_e64 v177, v213, v217, s54
	v_cndmask_b32_e64 v181, v217, v213, s54
	ds_load_b128 v[158:161], v6 offset:23040
	ds_load_b128 v[162:165], v6 offset:23056
	v_wmma_f32_16x16x16_f16 v[134:141], v[182:189], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:24320
	ds_load_b128 v[170:173], v6 offset:24336
	s_add_i32 s33, s34, 5
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[174:181], v[142:149], v[14:21]
	buffer_load_b64 v[194:195], v3, s[4:7], s31 offen offset:128
	buffer_load_b64 v[196:197], v3, s[4:7], s31 offen offset:140
	ds_load_b128 v[142:145], v6 offset:25600
	ds_load_b128 v[146:149], v6 offset:25616
	v_perm_b32 v210, v205, v204, v1
	v_perm_b32 v211, v205, v204, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[174:181], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:26880
	ds_load_b128 v[154:157], v6 offset:26896
	v_perm_b32 v212, v205, v204, v218
	v_perm_b32 v213, v205, v204, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[174:181], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:28160
	ds_load_b128 v[162:165], v6 offset:28176
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[38:45], v[174:181], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:29440
	ds_load_b128 v[170:173], v6 offset:29456
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[46:53], v[174:181], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:30720
	ds_load_b128 v[146:149], v6 offset:30736
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[54:61], v[174:181], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:32000
	ds_load_b128 v[154:157], v6 offset:32016
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[62:69], v[174:181], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:33280
	ds_load_b128 v[162:165], v6 offset:33296
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[70:77], v[174:181], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:34560
	ds_load_b128 v[170:173], v6 offset:34576
	v_pk_fma_f16 v210, v206, v210, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v206, v211, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[78:85], v[174:181], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:35840
	ds_load_b128 v[146:149], v6 offset:35856
	v_pk_fma_f16 v212, v206, v212, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v206, v213, v206 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[86:93], v[174:181], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:37120
	ds_load_b128 v[154:157], v6 offset:37136
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[94:101], v[174:181], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:38400
	ds_load_b128 v[162:165], v6 offset:38416
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[102:109], v[174:181], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:39680
	ds_load_b128 v[170:173], v6 offset:39696
	v_cndmask_b32_e64 v182, v210, v214, s54
	v_cndmask_b32_e64 v186, v214, v210, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[110:117], v[174:181], v[142:149], v[110:117]
	ds_load_b128 v[142:145], v6 offset:20512
	ds_load_b128 v[146:149], v6 offset:20528
	v_cndmask_b32_e64 v183, v211, v215, s54
	v_cndmask_b32_e64 v187, v215, v211, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[118:125], v[174:181], v[150:157], v[118:125]
	ds_load_b128 v[150:153], v6 offset:21792
	ds_load_b128 v[154:157], v6 offset:21808
	v_cndmask_b32_e64 v184, v212, v216, s54
	v_cndmask_b32_e64 v188, v216, v212, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[126:133], v[174:181], v[158:165], v[126:133]
	ds_load_b128 v[158:161], v6 offset:23072
	ds_load_b128 v[162:165], v6 offset:23088
	v_cndmask_b32_e64 v185, v213, v217, s54
	v_cndmask_b32_e64 v189, v217, v213, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[134:141], v[174:181], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:24352
	ds_load_b128 v[170:173], v6 offset:24368
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[182:189], v[142:149], v[14:21]
	ds_load_b128 v[142:145], v6 offset:25632
	ds_load_b128 v[146:149], v6 offset:25648
	s_waitcnt vmcnt(7)
	v_perm_b32 v210, v191, v190, v1
	v_perm_b32 v211, v191, v190, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[182:189], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:26912
	ds_load_b128 v[154:157], v6 offset:26928
	v_perm_b32 v212, v191, v190, v218
	v_perm_b32 v213, v191, v190, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[182:189], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:28192
	ds_load_b128 v[162:165], v6 offset:28208
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[220:223]
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[38:45], v[182:189], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:29472
	ds_load_b128 v[170:173], v6 offset:29488
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[46:53], v[182:189], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:30752
	ds_load_b128 v[146:149], v6 offset:30768
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[224:227] offset:5120
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[54:61], v[182:189], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:32032
	ds_load_b128 v[154:157], v6 offset:32048
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[62:69], v[182:189], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:33312
	ds_load_b128 v[162:165], v6 offset:33328
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[228:231] offset:10240
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[70:77], v[182:189], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:34592
	ds_load_b128 v[170:173], v6 offset:34608
	v_pk_fma_f16 v210, v207, v210, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v207, v211, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[78:85], v[182:189], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:35872
	ds_load_b128 v[146:149], v6 offset:35888
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[232:235] offset:15360
	v_pk_fma_f16 v212, v207, v212, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v207, v213, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[86:93], v[182:189], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:37152
	ds_load_b128 v[154:157], v6 offset:37168
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[94:101], v[182:189], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:38432
	ds_load_b128 v[162:165], v6 offset:38448
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[102:109], v[182:189], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:39712
	ds_load_b128 v[170:173], v6 offset:39728
	v_cndmask_b32_e64 v174, v210, v214, s54
	v_cndmask_b32_e64 v178, v214, v210, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[220:223], v10, s[8:11], s33 offen
	buffer_load_b128 v[224:227], v11, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v12, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v13, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[110:117], v[182:189], v[142:149], v[110:117]
	v_cndmask_b32_e64 v175, v211, v215, s54
	v_cndmask_b32_e64 v179, v215, v211, s54
	ds_load_b128 v[142:145], v6
	ds_load_b128 v[146:149], v6 offset:16
	v_wmma_f32_16x16x16_f16 v[118:125], v[182:189], v[150:157], v[118:125]
	v_cndmask_b32_e64 v176, v212, v216, s54
	v_cndmask_b32_e64 v180, v216, v212, s54
	ds_load_b128 v[150:153], v6 offset:1280
	ds_load_b128 v[154:157], v6 offset:1296
	v_wmma_f32_16x16x16_f16 v[126:133], v[182:189], v[158:165], v[126:133]
	v_cndmask_b32_e64 v177, v213, v217, s54
	v_cndmask_b32_e64 v181, v217, v213, s54
	ds_load_b128 v[158:161], v6 offset:2560
	ds_load_b128 v[162:165], v6 offset:2576
	v_wmma_f32_16x16x16_f16 v[134:141], v[182:189], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:3840
	ds_load_b128 v[170:173], v6 offset:3856
	s_add_i32 s33, s34, 6
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[174:181], v[142:149], v[14:21]
	buffer_load_b64 v[198:199], v3, s[4:7], s31 offen offset:152
	buffer_load_b64 v[200:201], v3, s[4:7], s31 offen offset:164
	ds_load_b128 v[142:145], v6 offset:5120
	ds_load_b128 v[146:149], v6 offset:5136
	v_perm_b32 v210, v193, v192, v1
	v_perm_b32 v211, v193, v192, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[174:181], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:6400
	ds_load_b128 v[154:157], v6 offset:6416
	v_perm_b32 v212, v193, v192, v218
	v_perm_b32 v213, v193, v192, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[174:181], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:7680
	ds_load_b128 v[162:165], v6 offset:7696
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[38:45], v[174:181], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:8960
	ds_load_b128 v[170:173], v6 offset:8976
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[46:53], v[174:181], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:10240
	ds_load_b128 v[146:149], v6 offset:10256
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[54:61], v[174:181], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:11520
	ds_load_b128 v[154:157], v6 offset:11536
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[62:69], v[174:181], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:12800
	ds_load_b128 v[162:165], v6 offset:12816
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[70:77], v[174:181], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:14080
	ds_load_b128 v[170:173], v6 offset:14096
	v_pk_fma_f16 v210, v207, v210, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v207, v211, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[78:85], v[174:181], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:15360
	ds_load_b128 v[146:149], v6 offset:15376
	v_pk_fma_f16 v212, v207, v212, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v207, v213, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[86:93], v[174:181], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:16640
	ds_load_b128 v[154:157], v6 offset:16656
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[94:101], v[174:181], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:17920
	ds_load_b128 v[162:165], v6 offset:17936
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[102:109], v[174:181], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:19200
	ds_load_b128 v[170:173], v6 offset:19216
	v_cndmask_b32_e64 v182, v210, v214, s54
	v_cndmask_b32_e64 v186, v214, v210, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[110:117], v[174:181], v[142:149], v[110:117]
	ds_load_b128 v[142:145], v6 offset:32
	ds_load_b128 v[146:149], v6 offset:48
	v_cndmask_b32_e64 v183, v211, v215, s54
	v_cndmask_b32_e64 v187, v215, v211, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[118:125], v[174:181], v[150:157], v[118:125]
	ds_load_b128 v[150:153], v6 offset:1312
	ds_load_b128 v[154:157], v6 offset:1328
	v_cndmask_b32_e64 v184, v212, v216, s54
	v_cndmask_b32_e64 v188, v216, v212, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[126:133], v[174:181], v[158:165], v[126:133]
	ds_load_b128 v[158:161], v6 offset:2592
	ds_load_b128 v[162:165], v6 offset:2608
	v_cndmask_b32_e64 v185, v213, v217, s54
	v_cndmask_b32_e64 v189, v217, v213, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[134:141], v[174:181], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:3872
	ds_load_b128 v[170:173], v6 offset:3888
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[182:189], v[142:149], v[14:21]
	ds_load_b128 v[142:145], v6 offset:5152
	ds_load_b128 v[146:149], v6 offset:5168
	s_waitcnt vmcnt(7)
	v_perm_b32 v210, v195, v194, v1
	v_perm_b32 v211, v195, v194, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[182:189], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:6432
	ds_load_b128 v[154:157], v6 offset:6448
	v_perm_b32 v212, v195, v194, v218
	v_perm_b32 v213, v195, v194, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[182:189], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:7712
	ds_load_b128 v[162:165], v6 offset:7728
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[220:223] offset:20480
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[38:45], v[182:189], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:8992
	ds_load_b128 v[170:173], v6 offset:9008
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[46:53], v[182:189], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:10272
	ds_load_b128 v[146:149], v6 offset:10288
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[224:227] offset:25600
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[54:61], v[182:189], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:11552
	ds_load_b128 v[154:157], v6 offset:11568
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[62:69], v[182:189], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:12832
	ds_load_b128 v[162:165], v6 offset:12848
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[228:231] offset:30720
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[70:77], v[182:189], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:14112
	ds_load_b128 v[170:173], v6 offset:14128
	v_pk_fma_f16 v210, v207, v210, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v207, v211, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[78:85], v[182:189], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:15392
	ds_load_b128 v[146:149], v6 offset:15408
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[232:235] offset:35840
	v_pk_fma_f16 v212, v207, v212, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v207, v213, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[86:93], v[182:189], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:16672
	ds_load_b128 v[154:157], v6 offset:16688
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[94:101], v[182:189], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:17952
	ds_load_b128 v[162:165], v6 offset:17968
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[102:109], v[182:189], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:19232
	ds_load_b128 v[170:173], v6 offset:19248
	v_cndmask_b32_e64 v174, v210, v214, s54
	v_cndmask_b32_e64 v178, v214, v210, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[220:223], v10, s[8:11], s33 offen
	buffer_load_b128 v[224:227], v11, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v12, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v13, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[110:117], v[182:189], v[142:149], v[110:117]
	v_cndmask_b32_e64 v175, v211, v215, s54
	v_cndmask_b32_e64 v179, v215, v211, s54
	ds_load_b128 v[142:145], v6 offset:20480
	ds_load_b128 v[146:149], v6 offset:20496
	v_wmma_f32_16x16x16_f16 v[118:125], v[182:189], v[150:157], v[118:125]
	v_cndmask_b32_e64 v176, v212, v216, s54
	v_cndmask_b32_e64 v180, v216, v212, s54
	ds_load_b128 v[150:153], v6 offset:21760
	ds_load_b128 v[154:157], v6 offset:21776
	v_wmma_f32_16x16x16_f16 v[126:133], v[182:189], v[158:165], v[126:133]
	v_cndmask_b32_e64 v177, v213, v217, s54
	v_cndmask_b32_e64 v181, v217, v213, s54
	ds_load_b128 v[158:161], v6 offset:23040
	ds_load_b128 v[162:165], v6 offset:23056
	v_wmma_f32_16x16x16_f16 v[134:141], v[182:189], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:24320
	ds_load_b128 v[170:173], v6 offset:24336
	s_add_i32 s33, s34, 7
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[174:181], v[142:149], v[14:21]
	buffer_load_b64 v[202:203], v3, s[4:7], s31 offen offset:176
	buffer_load_b64 v[204:205], v3, s[4:7], s31 offen offset:188
	ds_load_b128 v[142:145], v6 offset:25600
	ds_load_b128 v[146:149], v6 offset:25616
	v_perm_b32 v210, v197, v196, v1
	v_perm_b32 v211, v197, v196, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[174:181], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:26880
	ds_load_b128 v[154:157], v6 offset:26896
	v_perm_b32 v212, v197, v196, v218
	v_perm_b32 v213, v197, v196, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[174:181], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:28160
	ds_load_b128 v[162:165], v6 offset:28176
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[38:45], v[174:181], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:29440
	ds_load_b128 v[170:173], v6 offset:29456
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[46:53], v[174:181], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:30720
	ds_load_b128 v[146:149], v6 offset:30736
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[54:61], v[174:181], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:32000
	ds_load_b128 v[154:157], v6 offset:32016
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[62:69], v[174:181], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:33280
	ds_load_b128 v[162:165], v6 offset:33296
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[70:77], v[174:181], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:34560
	ds_load_b128 v[170:173], v6 offset:34576
	v_pk_fma_f16 v210, v207, v210, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v207, v211, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[78:85], v[174:181], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:35840
	ds_load_b128 v[146:149], v6 offset:35856
	v_pk_fma_f16 v212, v207, v212, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v207, v213, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[86:93], v[174:181], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:37120
	ds_load_b128 v[154:157], v6 offset:37136
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[94:101], v[174:181], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:38400
	ds_load_b128 v[162:165], v6 offset:38416
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[102:109], v[174:181], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:39680
	ds_load_b128 v[170:173], v6 offset:39696
	v_cndmask_b32_e64 v182, v210, v214, s54
	v_cndmask_b32_e64 v186, v214, v210, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[110:117], v[174:181], v[142:149], v[110:117]
	ds_load_b128 v[142:145], v6 offset:20512
	ds_load_b128 v[146:149], v6 offset:20528
	v_cndmask_b32_e64 v183, v211, v215, s54
	v_cndmask_b32_e64 v187, v215, v211, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[118:125], v[174:181], v[150:157], v[118:125]
	ds_load_b128 v[150:153], v6 offset:21792
	ds_load_b128 v[154:157], v6 offset:21808
	v_cndmask_b32_e64 v184, v212, v216, s54
	v_cndmask_b32_e64 v188, v216, v212, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[126:133], v[174:181], v[158:165], v[126:133]
	ds_load_b128 v[158:161], v6 offset:23072
	ds_load_b128 v[162:165], v6 offset:23088
	v_cndmask_b32_e64 v185, v213, v217, s54
	v_cndmask_b32_e64 v189, v217, v213, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[134:141], v[174:181], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:24352
	ds_load_b128 v[170:173], v6 offset:24368
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[182:189], v[142:149], v[14:21]
	ds_load_b128 v[142:145], v6 offset:25632
	ds_load_b128 v[146:149], v6 offset:25648
	s_waitcnt vmcnt(7)
	v_perm_b32 v210, v199, v198, v1
	v_perm_b32 v211, v199, v198, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[182:189], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:26912
	ds_load_b128 v[154:157], v6 offset:26928
	v_perm_b32 v212, v199, v198, v218
	v_perm_b32 v213, v199, v198, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[182:189], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:28192
	ds_load_b128 v[162:165], v6 offset:28208
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[220:223]
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[38:45], v[182:189], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:29472
	ds_load_b128 v[170:173], v6 offset:29488
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[46:53], v[182:189], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:30752
	ds_load_b128 v[146:149], v6 offset:30768
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[224:227] offset:5120
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[54:61], v[182:189], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:32032
	ds_load_b128 v[154:157], v6 offset:32048
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[62:69], v[182:189], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:33312
	ds_load_b128 v[162:165], v6 offset:33328
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[228:231] offset:10240
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[70:77], v[182:189], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:34592
	ds_load_b128 v[170:173], v6 offset:34608
	v_pk_fma_f16 v210, v207, v210, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v207, v211, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[78:85], v[182:189], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:35872
	ds_load_b128 v[146:149], v6 offset:35888
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[232:235] offset:15360
	v_pk_fma_f16 v212, v207, v212, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v207, v213, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[86:93], v[182:189], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:37152
	ds_load_b128 v[154:157], v6 offset:37168
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[94:101], v[182:189], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:38432
	ds_load_b128 v[162:165], v6 offset:38448
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[102:109], v[182:189], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:39712
	ds_load_b128 v[170:173], v6 offset:39728
	v_cndmask_b32_e64 v174, v210, v214, s54
	v_cndmask_b32_e64 v178, v214, v210, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[220:223], v10, s[8:11], s33 offen
	buffer_load_b128 v[224:227], v11, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v12, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v13, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[110:117], v[182:189], v[142:149], v[110:117]
	v_cndmask_b32_e64 v175, v211, v215, s54
	v_cndmask_b32_e64 v179, v215, v211, s54
	ds_load_b128 v[142:145], v6
	ds_load_b128 v[146:149], v6 offset:16
	v_wmma_f32_16x16x16_f16 v[118:125], v[182:189], v[150:157], v[118:125]
	v_cndmask_b32_e64 v176, v212, v216, s54
	v_cndmask_b32_e64 v180, v216, v212, s54
	ds_load_b128 v[150:153], v6 offset:1280
	ds_load_b128 v[154:157], v6 offset:1296
	v_wmma_f32_16x16x16_f16 v[126:133], v[182:189], v[158:165], v[126:133]
	v_cndmask_b32_e64 v177, v213, v217, s54
	v_cndmask_b32_e64 v181, v217, v213, s54
	ds_load_b128 v[158:161], v6 offset:2560
	ds_load_b128 v[162:165], v6 offset:2576
	v_wmma_f32_16x16x16_f16 v[134:141], v[182:189], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:3840
	ds_load_b128 v[170:173], v6 offset:3856
	s_add_i32 s33, s34, 8
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[174:181], v[142:149], v[14:21]
	buffer_load_b64 v[190:191], v3, s[4:7], s32 offen offset:8
	buffer_load_b64 v[192:193], v3, s[4:7], s32 offen offset:20
	buffer_load_b64 v[208:209], v4, s[4:7], s32 offen
	ds_load_b128 v[142:145], v6 offset:5120
	ds_load_b128 v[146:149], v6 offset:5136
	v_perm_b32 v210, v201, v200, v1
	v_perm_b32 v211, v201, v200, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[174:181], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:6400
	ds_load_b128 v[154:157], v6 offset:6416
	v_perm_b32 v212, v201, v200, v218
	v_perm_b32 v213, v201, v200, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[174:181], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:7680
	ds_load_b128 v[162:165], v6 offset:7696
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[38:45], v[174:181], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:8960
	ds_load_b128 v[170:173], v6 offset:8976
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[46:53], v[174:181], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:10240
	ds_load_b128 v[146:149], v6 offset:10256
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[54:61], v[174:181], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:11520
	ds_load_b128 v[154:157], v6 offset:11536
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[62:69], v[174:181], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:12800
	ds_load_b128 v[162:165], v6 offset:12816
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[70:77], v[174:181], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:14080
	ds_load_b128 v[170:173], v6 offset:14096
	v_pk_fma_f16 v210, v207, v210, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v207, v211, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[78:85], v[174:181], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:15360
	ds_load_b128 v[146:149], v6 offset:15376
	v_pk_fma_f16 v212, v207, v212, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v207, v213, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[86:93], v[174:181], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:16640
	ds_load_b128 v[154:157], v6 offset:16656
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[94:101], v[174:181], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:17920
	ds_load_b128 v[162:165], v6 offset:17936
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[102:109], v[174:181], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:19200
	ds_load_b128 v[170:173], v6 offset:19216
	v_cndmask_b32_e64 v182, v210, v214, s54
	v_cndmask_b32_e64 v186, v214, v210, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[110:117], v[174:181], v[142:149], v[110:117]
	ds_load_b128 v[142:145], v6 offset:32
	ds_load_b128 v[146:149], v6 offset:48
	v_cndmask_b32_e64 v183, v211, v215, s54
	v_cndmask_b32_e64 v187, v215, v211, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[118:125], v[174:181], v[150:157], v[118:125]
	ds_load_b128 v[150:153], v6 offset:1312
	ds_load_b128 v[154:157], v6 offset:1328
	v_cndmask_b32_e64 v184, v212, v216, s54
	v_cndmask_b32_e64 v188, v216, v212, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[126:133], v[174:181], v[158:165], v[126:133]
	ds_load_b128 v[158:161], v6 offset:2592
	ds_load_b128 v[162:165], v6 offset:2608
	v_cndmask_b32_e64 v185, v213, v217, s54
	v_cndmask_b32_e64 v189, v217, v213, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[134:141], v[174:181], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:3872
	ds_load_b128 v[170:173], v6 offset:3888
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[182:189], v[142:149], v[14:21]
	ds_load_b128 v[142:145], v6 offset:5152
	ds_load_b128 v[146:149], v6 offset:5168
	s_waitcnt vmcnt(8)
	v_perm_b32 v210, v203, v202, v1
	v_perm_b32 v211, v203, v202, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[182:189], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:6432
	ds_load_b128 v[154:157], v6 offset:6448
	v_perm_b32 v212, v203, v202, v218
	v_perm_b32 v213, v203, v202, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[182:189], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:7712
	ds_load_b128 v[162:165], v6 offset:7728
	s_waitcnt vmcnt(6)
	ds_store_b128 v5, v[220:223] offset:20480
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[38:45], v[182:189], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:8992
	ds_load_b128 v[170:173], v6 offset:9008
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[46:53], v[182:189], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:10272
	ds_load_b128 v[146:149], v6 offset:10288
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[224:227] offset:25600
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[54:61], v[182:189], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:11552
	ds_load_b128 v[154:157], v6 offset:11568
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[62:69], v[182:189], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:12832
	ds_load_b128 v[162:165], v6 offset:12848
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[228:231] offset:30720
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[70:77], v[182:189], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:14112
	ds_load_b128 v[170:173], v6 offset:14128
	v_pk_fma_f16 v210, v207, v210, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v207, v211, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[78:85], v[182:189], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:15392
	ds_load_b128 v[146:149], v6 offset:15408
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[232:235] offset:35840
	v_pk_fma_f16 v212, v207, v212, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v207, v213, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[86:93], v[182:189], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:16672
	ds_load_b128 v[154:157], v6 offset:16688
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[94:101], v[182:189], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:17952
	ds_load_b128 v[162:165], v6 offset:17968
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[102:109], v[182:189], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:19232
	ds_load_b128 v[170:173], v6 offset:19248
	v_cndmask_b32_e64 v174, v210, v214, s54
	v_cndmask_b32_e64 v178, v214, v210, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[220:223], v10, s[8:11], s33 offen
	buffer_load_b128 v[224:227], v11, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v12, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v13, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[110:117], v[182:189], v[142:149], v[110:117]
	v_cndmask_b32_e64 v175, v211, v215, s54
	v_cndmask_b32_e64 v179, v215, v211, s54
	ds_load_b128 v[142:145], v6 offset:20480
	ds_load_b128 v[146:149], v6 offset:20496
	v_wmma_f32_16x16x16_f16 v[118:125], v[182:189], v[150:157], v[118:125]
	v_cndmask_b32_e64 v176, v212, v216, s54
	v_cndmask_b32_e64 v180, v216, v212, s54
	ds_load_b128 v[150:153], v6 offset:21760
	ds_load_b128 v[154:157], v6 offset:21776
	v_wmma_f32_16x16x16_f16 v[126:133], v[182:189], v[158:165], v[126:133]
	v_cndmask_b32_e64 v177, v213, v217, s54
	v_cndmask_b32_e64 v181, v217, v213, s54
	ds_load_b128 v[158:161], v6 offset:23040
	ds_load_b128 v[162:165], v6 offset:23056
	v_wmma_f32_16x16x16_f16 v[134:141], v[182:189], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:24320
	ds_load_b128 v[170:173], v6 offset:24336
	s_add_i32 s33, s34, 9
	s_min_u32 s33, s33, s26
	s_lshl_b32 s33, s33, 6
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[174:181], v[142:149], v[14:21]
	buffer_load_b64 v[194:195], v3, s[4:7], s32 offen offset:32
	buffer_load_b64 v[196:197], v3, s[4:7], s32 offen offset:44
	ds_load_b128 v[142:145], v6 offset:25600
	ds_load_b128 v[146:149], v6 offset:25616
	v_perm_b32 v210, v205, v204, v1
	v_perm_b32 v211, v205, v204, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[174:181], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:26880
	ds_load_b128 v[154:157], v6 offset:26896
	v_perm_b32 v212, v205, v204, v218
	v_perm_b32 v213, v205, v204, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[174:181], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:28160
	ds_load_b128 v[162:165], v6 offset:28176
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[38:45], v[174:181], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:29440
	ds_load_b128 v[170:173], v6 offset:29456
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[46:53], v[174:181], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:30720
	ds_load_b128 v[146:149], v6 offset:30736
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[54:61], v[174:181], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:32000
	ds_load_b128 v[154:157], v6 offset:32016
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[62:69], v[174:181], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:33280
	ds_load_b128 v[162:165], v6 offset:33296
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[70:77], v[174:181], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:34560
	ds_load_b128 v[170:173], v6 offset:34576
	v_pk_fma_f16 v210, v207, v210, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v207, v211, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[78:85], v[174:181], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:35840
	ds_load_b128 v[146:149], v6 offset:35856
	v_pk_fma_f16 v212, v207, v212, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v207, v213, v207 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[86:93], v[174:181], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:37120
	ds_load_b128 v[154:157], v6 offset:37136
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[94:101], v[174:181], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:38400
	ds_load_b128 v[162:165], v6 offset:38416
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[102:109], v[174:181], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:39680
	ds_load_b128 v[170:173], v6 offset:39696
	v_cndmask_b32_e64 v182, v210, v214, s54
	v_cndmask_b32_e64 v186, v214, v210, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[110:117], v[174:181], v[142:149], v[110:117]
	ds_load_b128 v[142:145], v6 offset:20512
	ds_load_b128 v[146:149], v6 offset:20528
	v_cndmask_b32_e64 v183, v211, v215, s54
	v_cndmask_b32_e64 v187, v215, v211, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[118:125], v[174:181], v[150:157], v[118:125]
	ds_load_b128 v[150:153], v6 offset:21792
	ds_load_b128 v[154:157], v6 offset:21808
	v_cndmask_b32_e64 v184, v212, v216, s54
	v_cndmask_b32_e64 v188, v216, v212, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[126:133], v[174:181], v[158:165], v[126:133]
	ds_load_b128 v[158:161], v6 offset:23072
	ds_load_b128 v[162:165], v6 offset:23088
	v_cndmask_b32_e64 v185, v213, v217, s54
	v_cndmask_b32_e64 v189, v217, v213, s54
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[134:141], v[174:181], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:24352
	ds_load_b128 v[170:173], v6 offset:24368
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[14:21], v[182:189], v[142:149], v[14:21]
	ds_load_b128 v[142:145], v6 offset:25632
	ds_load_b128 v[146:149], v6 offset:25648
	s_waitcnt vmcnt(8)
	v_perm_b32 v210, v191, v190, v1
	v_perm_b32 v211, v191, v190, v2
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[22:29], v[182:189], v[150:157], v[22:29]
	ds_load_b128 v[150:153], v6 offset:26912
	ds_load_b128 v[154:157], v6 offset:26928
	v_perm_b32 v212, v191, v190, v218
	v_perm_b32 v213, v191, v190, v219
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[30:37], v[182:189], v[158:165], v[30:37]
	ds_load_b128 v[158:161], v6 offset:28192
	ds_load_b128 v[162:165], v6 offset:28208
	s_waitcnt vmcnt(5)
	ds_store_b128 v5, v[220:223]
	v_lshl_or_b32 v210, v210, 10, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v212, v212, 10, v212
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[38:45], v[182:189], v[166:173], v[38:45]
	ds_load_b128 v[166:169], v6 offset:29472
	ds_load_b128 v[170:173], v6 offset:29488
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v210, v210, s80, 0x64006400
	v_and_or_b32 v211, v211, s81, 0x5c005400
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[46:53], v[182:189], v[142:149], v[46:53]
	ds_load_b128 v[142:145], v6 offset:30752
	ds_load_b128 v[146:149], v6 offset:30768
	s_waitcnt vmcnt(4)
	ds_store_b128 v5, v[224:227] offset:5120
	v_and_or_b32 v212, v212, s80, 0x64006400
	v_and_or_b32 v213, v213, s81, 0x5c005400
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[54:61], v[182:189], v[150:157], v[54:61]
	ds_load_b128 v[150:153], v6 offset:32032
	ds_load_b128 v[154:157], v6 offset:32048
	v_pk_add_f16 v210, v210, s82
	v_pk_add_f16 v211, v211, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[62:69], v[182:189], v[158:165], v[62:69]
	ds_load_b128 v[158:161], v6 offset:33312
	ds_load_b128 v[162:165], v6 offset:33328
	s_waitcnt vmcnt(3)
	ds_store_b128 v5, v[228:231] offset:10240
	v_pk_add_f16 v212, v212, s82
	v_pk_add_f16 v213, v213, s83
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[70:77], v[182:189], v[166:173], v[70:77]
	ds_load_b128 v[166:169], v6 offset:34592
	ds_load_b128 v[170:173], v6 offset:34608
	v_pk_fma_f16 v210, v208, v210, v208 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v211, v208, v211, v208 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[78:85], v[182:189], v[142:149], v[78:85]
	ds_load_b128 v[142:145], v6 offset:35872
	ds_load_b128 v[146:149], v6 offset:35888
	s_waitcnt vmcnt(2)
	ds_store_b128 v5, v[232:235] offset:15360
	v_pk_fma_f16 v212, v208, v212, v208 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	v_pk_fma_f16 v213, v208, v213, v208 op_sel:[0,0,1] op_sel_hi:[0,1,1]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[86:93], v[182:189], v[150:157], v[86:93]
	ds_load_b128 v[150:153], v6 offset:37152
	ds_load_b128 v[154:157], v6 offset:37168
	v_permlanex16_b32 v214, v210, s50, 0xfedcba98
	v_permlanex16_b32 v215, v211, s50, 0xfedcba98
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[94:101], v[182:189], v[158:165], v[94:101]
	ds_load_b128 v[158:161], v6 offset:38432
	ds_load_b128 v[162:165], v6 offset:38448
	v_permlanex16_b32 v216, v212, s50, 0xfedcba98
	v_permlanex16_b32 v217, v213, s50, 0xfedcba98
	s_waitcnt lgkmcnt(7)
	v_wmma_f32_16x16x16_f16 v[102:109], v[182:189], v[166:173], v[102:109]
	ds_load_b128 v[166:169], v6 offset:39712
	ds_load_b128 v[170:173], v6 offset:39728
	v_cndmask_b32_e64 v174, v210, v214, s54
	v_cndmask_b32_e64 v178, v214, v210, s54
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_b128 v[220:223], v10, s[8:11], s33 offen
	buffer_load_b128 v[224:227], v11, s[8:11], s33 offen
	buffer_load_b128 v[228:231], v12, s[8:11], s33 offen
	buffer_load_b128 v[232:235], v13, s[8:11], s33 offen
	v_wmma_f32_16x16x16_f16 v[110:117], v[182:189], v[142:149], v[110:117]
	v_cndmask_b32_e64 v175, v211, v215, s54
	v_cndmask_b32_e64 v179, v215, v211, s54
	ds_load_b128 v[142:145], v6
	ds_load_b128 v[146:149], v6 offset:16
	v_wmma_f32_16x16x16_f16 v[118:125], v[182:189], v[150:157], v[118:125]
	v_cndmask_b32_e64 v176, v212, v216, s54
	v_cndmask_b32_e64 v180, v216, v212, s54
	ds_load_b128 v[150:153], v6 offset:1280
	ds_load_b128 v[154:157], v6 offset:1296
	v_wmma_f32_16x16x16_f16 v[126:133], v[182:189], v[158:165], v[126:133]
	v_cndmask_b32_e64 v177, v213, v217, s54
	v_cndmask_b32_e64 v181, v217, v213, s54
	ds_load_b128 v[158:161], v6 offset:2560
	ds_load_b128 v[162:165], v6 offset:2576
	v_wmma_f32_16x16x16_f16 v[134:141], v[182:189], v[166:173], v[134:141]
	ds_load_b128 v[166:169], v6 offset:3840
	ds_load_b128 v[170:173], v6 offset:3856
	v_mov_b32_e32 v206, v208
	v_mov_b32_e32 v207, v209
	s_add_i32 s30, s30, 1
	s_add_i32 s31, s31, 0xc8
	s_add_i32 s34, s34, 8
	s_cmp_lt_u32 s30, s25
	s_cbranch_scc1 .Lmq6g11w8y_loop
	s_waitcnt vmcnt(0) lgkmcnt(0)
	.Lmq6g11w8y_epi:
	v_cndmask_b32_e64 v142, v18, v14, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v19, v15, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v20, v16, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v21, v17, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v15
	v_mov_b32_e32 v147, v17
	v_cndmask_b32_e64 v14, v14, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v21, v145, v21, s54
	v_cndmask_b32_e64 v15, v142, v18, s54
	v_cndmask_b32_e64 v18, v16, v144, s54
	v_cndmask_b32_e64 v16, v146, v143, s54
	v_cndmask_b32_e64 v17, v143, v19, s54
	v_cndmask_b32_e64 v19, v144, v20, s54
	v_cndmask_b32_e64 v20, v147, v145, s54
	v_cndmask_b32_e64 v142, v26, v22, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v27, v23, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v28, v24, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v29, v25, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v23
	v_mov_b32_e32 v147, v25
	v_cndmask_b32_e64 v22, v22, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v29, v145, v29, s54
	v_cndmask_b32_e64 v23, v142, v26, s54
	v_cndmask_b32_e64 v26, v24, v144, s54
	v_cndmask_b32_e64 v24, v146, v143, s54
	v_cndmask_b32_e64 v25, v143, v27, s54
	v_cndmask_b32_e64 v27, v144, v28, s54
	v_cndmask_b32_e64 v28, v147, v145, s54
	v_cndmask_b32_e64 v142, v34, v30, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v35, v31, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v36, v32, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v37, v33, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v31
	v_mov_b32_e32 v147, v33
	v_cndmask_b32_e64 v30, v30, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v37, v145, v37, s54
	v_cndmask_b32_e64 v31, v142, v34, s54
	v_cndmask_b32_e64 v34, v32, v144, s54
	v_cndmask_b32_e64 v32, v146, v143, s54
	v_cndmask_b32_e64 v33, v143, v35, s54
	v_cndmask_b32_e64 v35, v144, v36, s54
	v_cndmask_b32_e64 v36, v147, v145, s54
	v_cndmask_b32_e64 v142, v42, v38, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v43, v39, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v44, v40, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v45, v41, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v39
	v_mov_b32_e32 v147, v41
	v_cndmask_b32_e64 v38, v38, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v45, v145, v45, s54
	v_cndmask_b32_e64 v39, v142, v42, s54
	v_cndmask_b32_e64 v42, v40, v144, s54
	v_cndmask_b32_e64 v40, v146, v143, s54
	v_cndmask_b32_e64 v41, v143, v43, s54
	v_cndmask_b32_e64 v43, v144, v44, s54
	v_cndmask_b32_e64 v44, v147, v145, s54
	v_cndmask_b32_e64 v142, v50, v46, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v51, v47, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v52, v48, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v53, v49, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v47
	v_mov_b32_e32 v147, v49
	v_cndmask_b32_e64 v46, v46, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v53, v145, v53, s54
	v_cndmask_b32_e64 v47, v142, v50, s54
	v_cndmask_b32_e64 v50, v48, v144, s54
	v_cndmask_b32_e64 v48, v146, v143, s54
	v_cndmask_b32_e64 v49, v143, v51, s54
	v_cndmask_b32_e64 v51, v144, v52, s54
	v_cndmask_b32_e64 v52, v147, v145, s54
	v_cndmask_b32_e64 v142, v58, v54, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v59, v55, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v60, v56, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v61, v57, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v55
	v_mov_b32_e32 v147, v57
	v_cndmask_b32_e64 v54, v54, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v61, v145, v61, s54
	v_cndmask_b32_e64 v55, v142, v58, s54
	v_cndmask_b32_e64 v58, v56, v144, s54
	v_cndmask_b32_e64 v56, v146, v143, s54
	v_cndmask_b32_e64 v57, v143, v59, s54
	v_cndmask_b32_e64 v59, v144, v60, s54
	v_cndmask_b32_e64 v60, v147, v145, s54
	v_cndmask_b32_e64 v142, v66, v62, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v67, v63, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v68, v64, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v69, v65, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v63
	v_mov_b32_e32 v147, v65
	v_cndmask_b32_e64 v62, v62, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v69, v145, v69, s54
	v_cndmask_b32_e64 v63, v142, v66, s54
	v_cndmask_b32_e64 v66, v64, v144, s54
	v_cndmask_b32_e64 v64, v146, v143, s54
	v_cndmask_b32_e64 v65, v143, v67, s54
	v_cndmask_b32_e64 v67, v144, v68, s54
	v_cndmask_b32_e64 v68, v147, v145, s54
	v_cndmask_b32_e64 v142, v74, v70, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v75, v71, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v76, v72, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v77, v73, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v71
	v_mov_b32_e32 v147, v73
	v_cndmask_b32_e64 v70, v70, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v77, v145, v77, s54
	v_cndmask_b32_e64 v71, v142, v74, s54
	v_cndmask_b32_e64 v74, v72, v144, s54
	v_cndmask_b32_e64 v72, v146, v143, s54
	v_cndmask_b32_e64 v73, v143, v75, s54
	v_cndmask_b32_e64 v75, v144, v76, s54
	v_cndmask_b32_e64 v76, v147, v145, s54
	v_cndmask_b32_e64 v142, v82, v78, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v83, v79, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v84, v80, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v85, v81, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v79
	v_mov_b32_e32 v147, v81
	v_cndmask_b32_e64 v78, v78, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v85, v145, v85, s54
	v_cndmask_b32_e64 v79, v142, v82, s54
	v_cndmask_b32_e64 v82, v80, v144, s54
	v_cndmask_b32_e64 v80, v146, v143, s54
	v_cndmask_b32_e64 v81, v143, v83, s54
	v_cndmask_b32_e64 v83, v144, v84, s54
	v_cndmask_b32_e64 v84, v147, v145, s54
	v_cndmask_b32_e64 v142, v90, v86, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v91, v87, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v92, v88, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v93, v89, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v87
	v_mov_b32_e32 v147, v89
	v_cndmask_b32_e64 v86, v86, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v93, v145, v93, s54
	v_cndmask_b32_e64 v87, v142, v90, s54
	v_cndmask_b32_e64 v90, v88, v144, s54
	v_cndmask_b32_e64 v88, v146, v143, s54
	v_cndmask_b32_e64 v89, v143, v91, s54
	v_cndmask_b32_e64 v91, v144, v92, s54
	v_cndmask_b32_e64 v92, v147, v145, s54
	v_cndmask_b32_e64 v142, v98, v94, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v99, v95, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v100, v96, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v101, v97, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v95
	v_mov_b32_e32 v147, v97
	v_cndmask_b32_e64 v94, v94, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v101, v145, v101, s54
	v_cndmask_b32_e64 v95, v142, v98, s54
	v_cndmask_b32_e64 v98, v96, v144, s54
	v_cndmask_b32_e64 v96, v146, v143, s54
	v_cndmask_b32_e64 v97, v143, v99, s54
	v_cndmask_b32_e64 v99, v144, v100, s54
	v_cndmask_b32_e64 v100, v147, v145, s54
	v_cndmask_b32_e64 v142, v106, v102, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v107, v103, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v108, v104, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v109, v105, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v103
	v_mov_b32_e32 v147, v105
	v_cndmask_b32_e64 v102, v102, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v109, v145, v109, s54
	v_cndmask_b32_e64 v103, v142, v106, s54
	v_cndmask_b32_e64 v106, v104, v144, s54
	v_cndmask_b32_e64 v104, v146, v143, s54
	v_cndmask_b32_e64 v105, v143, v107, s54
	v_cndmask_b32_e64 v107, v144, v108, s54
	v_cndmask_b32_e64 v108, v147, v145, s54
	v_cndmask_b32_e64 v142, v114, v110, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v115, v111, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v116, v112, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v117, v113, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v111
	v_mov_b32_e32 v147, v113
	v_cndmask_b32_e64 v110, v110, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v117, v145, v117, s54
	v_cndmask_b32_e64 v111, v142, v114, s54
	v_cndmask_b32_e64 v114, v112, v144, s54
	v_cndmask_b32_e64 v112, v146, v143, s54
	v_cndmask_b32_e64 v113, v143, v115, s54
	v_cndmask_b32_e64 v115, v144, v116, s54
	v_cndmask_b32_e64 v116, v147, v145, s54
	v_cndmask_b32_e64 v142, v122, v118, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v123, v119, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v124, v120, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v125, v121, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v119
	v_mov_b32_e32 v147, v121
	v_cndmask_b32_e64 v118, v118, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v125, v145, v125, s54
	v_cndmask_b32_e64 v119, v142, v122, s54
	v_cndmask_b32_e64 v122, v120, v144, s54
	v_cndmask_b32_e64 v120, v146, v143, s54
	v_cndmask_b32_e64 v121, v143, v123, s54
	v_cndmask_b32_e64 v123, v144, v124, s54
	v_cndmask_b32_e64 v124, v147, v145, s54
	v_cndmask_b32_e64 v142, v130, v126, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v131, v127, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v132, v128, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v133, v129, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v127
	v_mov_b32_e32 v147, v129
	v_cndmask_b32_e64 v126, v126, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v133, v145, v133, s54
	v_cndmask_b32_e64 v127, v142, v130, s54
	v_cndmask_b32_e64 v130, v128, v144, s54
	v_cndmask_b32_e64 v128, v146, v143, s54
	v_cndmask_b32_e64 v129, v143, v131, s54
	v_cndmask_b32_e64 v131, v144, v132, s54
	v_cndmask_b32_e64 v132, v147, v145, s54
	v_cndmask_b32_e64 v142, v138, v134, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v142, v142, s50, 0xfedcba98
	v_cndmask_b32_e64 v143, v139, v135, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v143, v143, s50, 0xfedcba98
	v_cndmask_b32_e64 v144, v140, v136, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v144, v144, s50, 0xfedcba98
	v_cndmask_b32_e64 v145, v141, v137, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v145, v145, s50, 0xfedcba98
	v_mov_b32_e32 v146, v135
	v_mov_b32_e32 v147, v137
	v_cndmask_b32_e64 v134, v134, v142, s54
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v141, v145, v141, s54
	v_cndmask_b32_e64 v135, v142, v138, s54
	v_cndmask_b32_e64 v138, v136, v144, s54
	v_cndmask_b32_e64 v136, v146, v143, s54
	v_cndmask_b32_e64 v137, v143, v139, s54
	v_cndmask_b32_e64 v139, v144, v140, s54
	v_cndmask_b32_e64 v140, v147, v145, s54
	s_lshl_b32 s40, s22, 6
	s_mov_b32 s37, 0
	v_mov_b32_e32 v148, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v149, s40, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v150, s40, v149
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v151, s40, v150
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v152, s40, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v153, s40, v152
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v154, s40, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v155, s40, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v156, s40, v155
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v157, s40, v156
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v158, s40, v157
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v159, s40, v158
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v160, s40, v159
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v161, s40, v160
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v162, s40, v161
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v163, s40, v162
	s_and_b32 s35, s22, 7
	s_cmp_lg_u32 s35, 0
	s_cbranch_scc1 .Lmq6g11w8y_scalar
	v_cmp_gt_i32_e64 s46, s22, v9
	s_sub_i32 s41, s24, 0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[164:167], v148, s[56:59], s37 offen
	buffer_load_b128 v[168:171], v148, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 16
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[172:175], v149, s[56:59], s37 offen
	buffer_load_b128 v[176:179], v149, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 32
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[180:183], v150, s[56:59], s37 offen
	buffer_load_b128 v[184:187], v150, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 48
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[188:191], v151, s[56:59], s37 offen
	buffer_load_b128 v[192:195], v151, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 64
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[196:199], v152, s[56:59], s37 offen
	buffer_load_b128 v[200:203], v152, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x50
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[204:207], v153, s[56:59], s37 offen
	buffer_load_b128 v[208:211], v153, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x60
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[212:215], v154, s[56:59], s37 offen
	buffer_load_b128 v[216:219], v154, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x70
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[220:223], v155, s[56:59], s37 offen
	buffer_load_b128 v[224:227], v155, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(15)
	v_add_f32_e32 v14, v14, v164
	v_add_f32_e32 v15, v15, v165
	v_add_f32_e32 v16, v16, v166
	v_add_f32_e32 v17, v17, v167
	s_waitcnt vmcnt(14)
	v_add_f32_e32 v18, v18, v168
	v_add_f32_e32 v19, v19, v169
	v_add_f32_e32 v20, v20, v170
	v_add_f32_e32 v21, v21, v171
	buffer_store_b128 v[14:17], v148, s[56:59], s37 offen
	buffer_store_b128 v[18:21], v148, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 16
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(13)
	v_add_f32_e32 v22, v22, v172
	v_add_f32_e32 v23, v23, v173
	v_add_f32_e32 v24, v24, v174
	v_add_f32_e32 v25, v25, v175
	s_waitcnt vmcnt(12)
	v_add_f32_e32 v26, v26, v176
	v_add_f32_e32 v27, v27, v177
	v_add_f32_e32 v28, v28, v178
	v_add_f32_e32 v29, v29, v179
	buffer_store_b128 v[22:25], v149, s[56:59], s37 offen
	buffer_store_b128 v[26:29], v149, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 32
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(11)
	v_add_f32_e32 v30, v30, v180
	v_add_f32_e32 v31, v31, v181
	v_add_f32_e32 v32, v32, v182
	v_add_f32_e32 v33, v33, v183
	s_waitcnt vmcnt(10)
	v_add_f32_e32 v34, v34, v184
	v_add_f32_e32 v35, v35, v185
	v_add_f32_e32 v36, v36, v186
	v_add_f32_e32 v37, v37, v187
	buffer_store_b128 v[30:33], v150, s[56:59], s37 offen
	buffer_store_b128 v[34:37], v150, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 48
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(9)
	v_add_f32_e32 v38, v38, v188
	v_add_f32_e32 v39, v39, v189
	v_add_f32_e32 v40, v40, v190
	v_add_f32_e32 v41, v41, v191
	s_waitcnt vmcnt(8)
	v_add_f32_e32 v42, v42, v192
	v_add_f32_e32 v43, v43, v193
	v_add_f32_e32 v44, v44, v194
	v_add_f32_e32 v45, v45, v195
	buffer_store_b128 v[38:41], v151, s[56:59], s37 offen
	buffer_store_b128 v[42:45], v151, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 64
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v46, v46, v196
	v_add_f32_e32 v47, v47, v197
	v_add_f32_e32 v48, v48, v198
	v_add_f32_e32 v49, v49, v199
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v50, v50, v200
	v_add_f32_e32 v51, v51, v201
	v_add_f32_e32 v52, v52, v202
	v_add_f32_e32 v53, v53, v203
	buffer_store_b128 v[46:49], v152, s[56:59], s37 offen
	buffer_store_b128 v[50:53], v152, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x50
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v54, v54, v204
	v_add_f32_e32 v55, v55, v205
	v_add_f32_e32 v56, v56, v206
	v_add_f32_e32 v57, v57, v207
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v58, v58, v208
	v_add_f32_e32 v59, v59, v209
	v_add_f32_e32 v60, v60, v210
	v_add_f32_e32 v61, v61, v211
	buffer_store_b128 v[54:57], v153, s[56:59], s37 offen
	buffer_store_b128 v[58:61], v153, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x60
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v62, v62, v212
	v_add_f32_e32 v63, v63, v213
	v_add_f32_e32 v64, v64, v214
	v_add_f32_e32 v65, v65, v215
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v66, v66, v216
	v_add_f32_e32 v67, v67, v217
	v_add_f32_e32 v68, v68, v218
	v_add_f32_e32 v69, v69, v219
	buffer_store_b128 v[62:65], v154, s[56:59], s37 offen
	buffer_store_b128 v[66:69], v154, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x70
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v70, v70, v220
	v_add_f32_e32 v71, v71, v221
	v_add_f32_e32 v72, v72, v222
	v_add_f32_e32 v73, v73, v223
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v74, v74, v224
	v_add_f32_e32 v75, v75, v225
	v_add_f32_e32 v76, v76, v226
	v_add_f32_e32 v77, v77, v227
	buffer_store_b128 v[70:73], v155, s[56:59], s37 offen
	buffer_store_b128 v[74:77], v155, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x80
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[164:167], v156, s[56:59], s37 offen
	buffer_load_b128 v[168:171], v156, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x90
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[172:175], v157, s[56:59], s37 offen
	buffer_load_b128 v[176:179], v157, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xa0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[180:183], v158, s[56:59], s37 offen
	buffer_load_b128 v[184:187], v158, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xb0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[188:191], v159, s[56:59], s37 offen
	buffer_load_b128 v[192:195], v159, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xc0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[196:199], v160, s[56:59], s37 offen
	buffer_load_b128 v[200:203], v160, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xd0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[204:207], v161, s[56:59], s37 offen
	buffer_load_b128 v[208:211], v161, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xe0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[212:215], v162, s[56:59], s37 offen
	buffer_load_b128 v[216:219], v162, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xf0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	buffer_load_b128 v[220:223], v163, s[56:59], s37 offen
	buffer_load_b128 v[224:227], v163, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x80
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(15)
	v_add_f32_e32 v78, v78, v164
	v_add_f32_e32 v79, v79, v165
	v_add_f32_e32 v80, v80, v166
	v_add_f32_e32 v81, v81, v167
	s_waitcnt vmcnt(14)
	v_add_f32_e32 v82, v82, v168
	v_add_f32_e32 v83, v83, v169
	v_add_f32_e32 v84, v84, v170
	v_add_f32_e32 v85, v85, v171
	buffer_store_b128 v[78:81], v156, s[56:59], s37 offen
	buffer_store_b128 v[82:85], v156, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0x90
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(13)
	v_add_f32_e32 v86, v86, v172
	v_add_f32_e32 v87, v87, v173
	v_add_f32_e32 v88, v88, v174
	v_add_f32_e32 v89, v89, v175
	s_waitcnt vmcnt(12)
	v_add_f32_e32 v90, v90, v176
	v_add_f32_e32 v91, v91, v177
	v_add_f32_e32 v92, v92, v178
	v_add_f32_e32 v93, v93, v179
	buffer_store_b128 v[86:89], v157, s[56:59], s37 offen
	buffer_store_b128 v[90:93], v157, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xa0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(11)
	v_add_f32_e32 v94, v94, v180
	v_add_f32_e32 v95, v95, v181
	v_add_f32_e32 v96, v96, v182
	v_add_f32_e32 v97, v97, v183
	s_waitcnt vmcnt(10)
	v_add_f32_e32 v98, v98, v184
	v_add_f32_e32 v99, v99, v185
	v_add_f32_e32 v100, v100, v186
	v_add_f32_e32 v101, v101, v187
	buffer_store_b128 v[94:97], v158, s[56:59], s37 offen
	buffer_store_b128 v[98:101], v158, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xb0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(9)
	v_add_f32_e32 v102, v102, v188
	v_add_f32_e32 v103, v103, v189
	v_add_f32_e32 v104, v104, v190
	v_add_f32_e32 v105, v105, v191
	s_waitcnt vmcnt(8)
	v_add_f32_e32 v106, v106, v192
	v_add_f32_e32 v107, v107, v193
	v_add_f32_e32 v108, v108, v194
	v_add_f32_e32 v109, v109, v195
	buffer_store_b128 v[102:105], v159, s[56:59], s37 offen
	buffer_store_b128 v[106:109], v159, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xc0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v110, v110, v196
	v_add_f32_e32 v111, v111, v197
	v_add_f32_e32 v112, v112, v198
	v_add_f32_e32 v113, v113, v199
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v114, v114, v200
	v_add_f32_e32 v115, v115, v201
	v_add_f32_e32 v116, v116, v202
	v_add_f32_e32 v117, v117, v203
	buffer_store_b128 v[110:113], v160, s[56:59], s37 offen
	buffer_store_b128 v[114:117], v160, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xd0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v118, v118, v204
	v_add_f32_e32 v119, v119, v205
	v_add_f32_e32 v120, v120, v206
	v_add_f32_e32 v121, v121, v207
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v122, v122, v208
	v_add_f32_e32 v123, v123, v209
	v_add_f32_e32 v124, v124, v210
	v_add_f32_e32 v125, v125, v211
	buffer_store_b128 v[118:121], v161, s[56:59], s37 offen
	buffer_store_b128 v[122:125], v161, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xe0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v126, v126, v212
	v_add_f32_e32 v127, v127, v213
	v_add_f32_e32 v128, v128, v214
	v_add_f32_e32 v129, v129, v215
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v130, v130, v216
	v_add_f32_e32 v131, v131, v217
	v_add_f32_e32 v132, v132, v218
	v_add_f32_e32 v133, v133, v219
	buffer_store_b128 v[126:129], v162, s[56:59], s37 offen
	buffer_store_b128 v[130:133], v162, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_sub_i32 s41, s24, 0xf0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v134, v134, v220
	v_add_f32_e32 v135, v135, v221
	v_add_f32_e32 v136, v136, v222
	v_add_f32_e32 v137, v137, v223
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v138, v138, v224
	v_add_f32_e32 v139, v139, v225
	v_add_f32_e32 v140, v140, v226
	v_add_f32_e32 v141, v141, v227
	buffer_store_b128 v[134:137], v163, s[56:59], s37 offen
	buffer_store_b128 v[138:141], v163, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_waitcnt_vscnt null, 0x0
	s_branch .Lmq6g11w8y_end
	.Lmq6g11w8y_scalar:
	s_sub_i32 s41, s24, 0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v148, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v148, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v148, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v148, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v148, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v148, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v148, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v148, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v14, v14, v164
	buffer_store_b32 v14, v148, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v15, v15, v165
	buffer_store_b32 v15, v148, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v16, v16, v166
	buffer_store_b32 v16, v148, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v17, v17, v167
	buffer_store_b32 v17, v148, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v18, v18, v168
	buffer_store_b32 v18, v148, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v19, v19, v169
	buffer_store_b32 v19, v148, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v20, v20, v170
	buffer_store_b32 v20, v148, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v21, v21, v171
	buffer_store_b32 v21, v148, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 16
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v149, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v149, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v149, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v149, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v149, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v149, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v149, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v149, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v22, v22, v164
	buffer_store_b32 v22, v149, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v23, v23, v165
	buffer_store_b32 v23, v149, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v24, v24, v166
	buffer_store_b32 v24, v149, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v25, v25, v167
	buffer_store_b32 v25, v149, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v26, v26, v168
	buffer_store_b32 v26, v149, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v27, v27, v169
	buffer_store_b32 v27, v149, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v28, v28, v170
	buffer_store_b32 v28, v149, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v29, v29, v171
	buffer_store_b32 v29, v149, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 32
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v150, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v150, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v150, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v150, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v150, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v150, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v150, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v150, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v30, v30, v164
	buffer_store_b32 v30, v150, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v31, v31, v165
	buffer_store_b32 v31, v150, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v32, v32, v166
	buffer_store_b32 v32, v150, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v33, v33, v167
	buffer_store_b32 v33, v150, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v34, v34, v168
	buffer_store_b32 v34, v150, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v35, v35, v169
	buffer_store_b32 v35, v150, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v36, v36, v170
	buffer_store_b32 v36, v150, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v37, v37, v171
	buffer_store_b32 v37, v150, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 48
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v151, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v151, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v151, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v151, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v151, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v151, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v151, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v151, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v38, v38, v164
	buffer_store_b32 v38, v151, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v39, v39, v165
	buffer_store_b32 v39, v151, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v40, v40, v166
	buffer_store_b32 v40, v151, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v41, v41, v167
	buffer_store_b32 v41, v151, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v42, v42, v168
	buffer_store_b32 v42, v151, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v43, v43, v169
	buffer_store_b32 v43, v151, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v44, v44, v170
	buffer_store_b32 v44, v151, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v45, v45, v171
	buffer_store_b32 v45, v151, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 64
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v152, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v152, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v152, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v152, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v152, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v152, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v152, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v152, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v46, v46, v164
	buffer_store_b32 v46, v152, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v47, v47, v165
	buffer_store_b32 v47, v152, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v48, v48, v166
	buffer_store_b32 v48, v152, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v49, v49, v167
	buffer_store_b32 v49, v152, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v50, v50, v168
	buffer_store_b32 v50, v152, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v51, v51, v169
	buffer_store_b32 v51, v152, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v52, v52, v170
	buffer_store_b32 v52, v152, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v53, v53, v171
	buffer_store_b32 v53, v152, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0x50
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v153, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v153, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v153, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v153, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v153, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v153, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v153, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v153, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v54, v54, v164
	buffer_store_b32 v54, v153, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v55, v55, v165
	buffer_store_b32 v55, v153, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v56, v56, v166
	buffer_store_b32 v56, v153, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v57, v57, v167
	buffer_store_b32 v57, v153, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v58, v58, v168
	buffer_store_b32 v58, v153, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v59, v59, v169
	buffer_store_b32 v59, v153, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v60, v60, v170
	buffer_store_b32 v60, v153, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v61, v61, v171
	buffer_store_b32 v61, v153, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0x60
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v154, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v154, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v154, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v154, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v154, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v154, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v154, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v154, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v62, v62, v164
	buffer_store_b32 v62, v154, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v63, v63, v165
	buffer_store_b32 v63, v154, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v64, v64, v166
	buffer_store_b32 v64, v154, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v65, v65, v167
	buffer_store_b32 v65, v154, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v66, v66, v168
	buffer_store_b32 v66, v154, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v67, v67, v169
	buffer_store_b32 v67, v154, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v68, v68, v170
	buffer_store_b32 v68, v154, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v69, v69, v171
	buffer_store_b32 v69, v154, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0x70
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v155, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v155, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v155, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v155, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v155, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v155, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v155, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v155, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v70, v70, v164
	buffer_store_b32 v70, v155, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v71, v71, v165
	buffer_store_b32 v71, v155, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v72, v72, v166
	buffer_store_b32 v72, v155, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v73, v73, v167
	buffer_store_b32 v73, v155, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v74, v74, v168
	buffer_store_b32 v74, v155, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v75, v75, v169
	buffer_store_b32 v75, v155, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v76, v76, v170
	buffer_store_b32 v76, v155, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v77, v77, v171
	buffer_store_b32 v77, v155, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0x80
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v156, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v156, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v156, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v156, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v156, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v156, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v156, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v156, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v78, v78, v164
	buffer_store_b32 v78, v156, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v79, v79, v165
	buffer_store_b32 v79, v156, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v80, v80, v166
	buffer_store_b32 v80, v156, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v81, v81, v167
	buffer_store_b32 v81, v156, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v82, v82, v168
	buffer_store_b32 v82, v156, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v83, v83, v169
	buffer_store_b32 v83, v156, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v84, v84, v170
	buffer_store_b32 v84, v156, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v85, v85, v171
	buffer_store_b32 v85, v156, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0x90
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v157, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v157, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v157, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v157, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v157, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v157, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v157, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v157, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v86, v86, v164
	buffer_store_b32 v86, v157, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v87, v87, v165
	buffer_store_b32 v87, v157, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v88, v88, v166
	buffer_store_b32 v88, v157, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v89, v89, v167
	buffer_store_b32 v89, v157, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v90, v90, v168
	buffer_store_b32 v90, v157, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v91, v91, v169
	buffer_store_b32 v91, v157, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v92, v92, v170
	buffer_store_b32 v92, v157, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v93, v93, v171
	buffer_store_b32 v93, v157, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xa0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v158, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v158, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v158, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v158, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v158, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v158, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v158, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v158, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v94, v94, v164
	buffer_store_b32 v94, v158, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v95, v95, v165
	buffer_store_b32 v95, v158, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v96, v96, v166
	buffer_store_b32 v96, v158, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v97, v97, v167
	buffer_store_b32 v97, v158, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v98, v98, v168
	buffer_store_b32 v98, v158, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v99, v99, v169
	buffer_store_b32 v99, v158, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v100, v100, v170
	buffer_store_b32 v100, v158, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v101, v101, v171
	buffer_store_b32 v101, v158, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xb0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v159, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v159, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v159, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v159, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v159, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v159, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v159, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v159, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v102, v102, v164
	buffer_store_b32 v102, v159, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v103, v103, v165
	buffer_store_b32 v103, v159, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v104, v104, v166
	buffer_store_b32 v104, v159, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v105, v105, v167
	buffer_store_b32 v105, v159, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v106, v106, v168
	buffer_store_b32 v106, v159, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v107, v107, v169
	buffer_store_b32 v107, v159, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v108, v108, v170
	buffer_store_b32 v108, v159, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v109, v109, v171
	buffer_store_b32 v109, v159, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xc0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v160, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v160, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v160, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v160, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v160, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v160, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v160, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v160, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v110, v110, v164
	buffer_store_b32 v110, v160, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v111, v111, v165
	buffer_store_b32 v111, v160, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v112, v112, v166
	buffer_store_b32 v112, v160, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v113, v113, v167
	buffer_store_b32 v113, v160, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v114, v114, v168
	buffer_store_b32 v114, v160, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v115, v115, v169
	buffer_store_b32 v115, v160, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v116, v116, v170
	buffer_store_b32 v116, v160, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v117, v117, v171
	buffer_store_b32 v117, v160, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xd0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v161, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v161, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v161, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v161, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v161, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v161, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v161, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v161, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v118, v118, v164
	buffer_store_b32 v118, v161, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v119, v119, v165
	buffer_store_b32 v119, v161, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v120, v120, v166
	buffer_store_b32 v120, v161, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v121, v121, v167
	buffer_store_b32 v121, v161, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v122, v122, v168
	buffer_store_b32 v122, v161, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v123, v123, v169
	buffer_store_b32 v123, v161, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v124, v124, v170
	buffer_store_b32 v124, v161, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v125, v125, v171
	buffer_store_b32 v125, v161, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xe0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v162, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v162, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v162, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v162, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v162, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v162, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v162, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v162, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v126, v126, v164
	buffer_store_b32 v126, v162, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v127, v127, v165
	buffer_store_b32 v127, v162, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v128, v128, v166
	buffer_store_b32 v128, v162, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v129, v129, v167
	buffer_store_b32 v129, v162, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v130, v130, v168
	buffer_store_b32 v130, v162, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v131, v131, v169
	buffer_store_b32 v131, v162, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v132, v132, v170
	buffer_store_b32 v132, v162, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v133, v133, v171
	buffer_store_b32 v133, v162, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_sub_i32 s41, s24, 0xf0
	v_cmp_gt_i32_e64 s44, s41, v7
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v164, v163, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v165, v163, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v166, v163, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v167, v163, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v168, v163, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v169, v163, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v170, v163, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	buffer_load_b32 v171, v163, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	s_mov_b32 s42, s22
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(7)
	v_add_f32_e32 v134, v134, v164
	buffer_store_b32 v134, v163, s[56:59], s37 offen
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(6)
	v_add_f32_e32 v135, v135, v165
	buffer_store_b32 v135, v163, s[56:59], s37 offen offset:4
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(5)
	v_add_f32_e32 v136, v136, v166
	buffer_store_b32 v136, v163, s[56:59], s37 offen offset:8
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(4)
	v_add_f32_e32 v137, v137, v167
	buffer_store_b32 v137, v163, s[56:59], s37 offen offset:12
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(3)
	v_add_f32_e32 v138, v138, v168
	buffer_store_b32 v138, v163, s[56:59], s37 offen offset:16
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(2)
	v_add_f32_e32 v139, v139, v169
	buffer_store_b32 v139, v163, s[56:59], s37 offen offset:20
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v140, v140, v170
	buffer_store_b32 v140, v163, s[56:59], s37 offen offset:24
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	v_cmp_gt_i32_e64 s46, s42, v9
	s_and_b32 exec_lo, s44, s46
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v141, v141, v171
	buffer_store_b32 v141, v163, s[56:59], s37 offen offset:28
	s_mov_b32 exec_lo, -1
	s_add_i32 s42, s42, -1
	.Lmq6g11w8y_end:
	s_endpgm
.Lgemm_mq6g256v2_residual_pm_gfx1100_w8_end:
.size gemm_mq6g256v2_residual_pm_gfx1100_w8, .Lgemm_mq6g256v2_residual_pm_gfx1100_w8_end-gemm_mq6g256v2_residual_pm_gfx1100_w8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemm_mq6g256v2_residual_pm_gfx1100_w8
	.amdhsa_group_segment_fixed_size 40960
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
	.amdhsa_next_free_vgpr 240
	.amdhsa_next_free_sgpr 84
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemm_mq6g256v2_residual_pm_gfx1100_w8_end-gemm_mq6g256v2_residual_pm_gfx1100_w8)<<4)&1008)>>4
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
      - .address_space: global
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: Y
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
        .name: N
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 40960
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 128
    .name: gemm_mq6g256v2_residual_pm_gfx1100_w4
    .private_segment_fixed_size: 0
    .sgpr_count: 84
    .sgpr_spill_count: 0
    .symbol: gemm_mq6g256v2_residual_pm_gfx1100_w4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 256
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: Y
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
        .name: N
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 40960
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 256
    .name: gemm_mq6g256v2_residual_pm_gfx1100_w8
    .private_segment_fixed_size: 0
    .sgpr_count: 84
    .sgpr_spill_count: 0
    .symbol: gemm_mq6g256v2_residual_pm_gfx1100_w8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 240
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
