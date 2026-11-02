.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151
.globl gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151
.p2align 8
.type gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151,@function
gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b32 s16, s[0:1], 0x20
	v_lshrrev_b32_e32 v1, 5, v0
	v_and_b32_e32 v2, 31, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_readfirstlane_b32 s21, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v3, 15, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_lshrrev_b32_e32 v4, 4, v2
	s_lshr_b32 s23, s21, 2
	s_and_b32 s24, s21, 3
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s20, s15, 8
	s_add_i32 s22, s20, -1
	s_mulk_i32 s20, 0x88
	s_lshl_b32 s4, s3, 8
	s_lshl_b32 s5, s21, 4
	s_add_u32 s5, s4, s5
	s_mul_i32 s42, s5, s20
	s_mul_hi_u32 s43, s5, s20
	s_add_u32 s28, s8, s42
	s_addc_u32 s29, s9, s43
	s_lshl_b32 s5, s23, 6
	s_add_u32 s5, s4, s5
	s_mul_i32 s42, s5, s20
	s_mul_hi_u32 s43, s5, s20
	s_add_u32 s30, s8, s42
	s_addc_u32 s31, s9, s43
	s_lshl_b32 s5, s20, 4
	s_add_u32 s32, s30, s5
	s_addc_u32 s33, s31, 0
	s_mul_i32 s5, s2, 0x4800
	s_add_u32 s34, s10, s5
	s_addc_u32 s35, s11, 0
	s_mul_i32 s25, s16, 0x48
	s_lshl_b32 s26, s14, 6
	s_lshl_b32 s5, s2, 8
	s_mul_hi_u32 s43, s5, s14
	s_mul_i32 s42, s5, s14
	s_add_u32 s42, s42, s4
	s_addc_u32 s43, s43, 0
	s_lshl_b64 s[42:43], s[42:43], 2
	s_add_u32 s36, s12, s42
	s_addc_u32 s37, s13, s43
	s_add_u32 s48, s36, s26
	s_addc_u32 s49, s37, 0
	s_add_u32 s50, s48, s26
	s_addc_u32 s51, s49, 0
	s_add_u32 s52, s50, s26
	s_addc_u32 s53, s51, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v6, v4, 3, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v240, v3, s20, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v5, s21, 4, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v241, v5, 0x48, v6
	v_lshlrev_b32_e32 v7, 3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v16, v4, 7, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v244, s21, 10, v16
	v_and_b32_e32 v8, 1, v3
	v_lshrrev_b32_e32 v9, 1, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 3, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 6, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v245, s23, 12, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v246, 0x800, v245
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v247, 0x8000, v245
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v248, 0x8800, v245
	v_lshl_add_u32 v10, s24, 12, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v249, 0x4000, v10
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v250, 0x4800, v10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v251, 0xc000, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v252, 0xc800, v10
	v_lshrrev_b32_e32 v12, 3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v12, 5, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v12, v4, 3, v12
	v_and_b32_e32 v13, 7, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_u32_u24_e32 v242, s20, v12
	v_lshl_add_u32 v14, s24, 6, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_u32_u24_e32 v243, 0x48, v14
	v_lshlrev_b32_e32 v15, 3, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v15, s23, 6, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v15, v14, s14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v223, 2, v15
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29]
	global_load_b64 v[226:227], v240, s[28:29] offset:16
	global_load_b64 v[228:229], v240, s[28:29] offset:32
	global_load_b64 v[230:231], v240, s[28:29] offset:48
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	v_mov_b32_e32 v160, 0x4b400000
	v_mov_b32_e32 v161, 0x4b400000
	v_mov_b32_e32 v162, 0x4b400000
	v_mov_b32_e32 v163, 0x4b400000
	v_mov_b32_e32 v164, 0x4b400000
	v_mov_b32_e32 v165, 0x4b400000
	v_mov_b32_e32 v166, 0x4b400000
	v_mov_b32_e32 v167, 0x4b400000
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
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225]
	ds_store_b64 v244, v[226:227] offset:256
	ds_store_b64 v244, v[228:229] offset:512
	ds_store_b64 v244, v[230:231] offset:768
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:16384
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:16640
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:16896
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:17152
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_load_2addr_b64 v[168:171], v245 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	global_load_u16 v221, v242, s[30:31]
	global_load_u16 v222, v242, s[32:33]
	global_load_b32 v208, v243, s[34:35]
	global_load_b32 v209, v243, s[34:35] offset:1152
	global_load_b32 v210, v243, s[34:35] offset:2304
	global_load_b32 v211, v243, s[34:35] offset:3456
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29] offset:64
	global_load_b64 v[226:227], v240, s[28:29] offset:80
	global_load_b64 v[228:229], v240, s[28:29] offset:96
	global_load_b64 v[230:231], v240, s[28:29] offset:112
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	.Lv2b_set_k_begin:
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .Lv2b_set_k_loop_end
	.Lv2b_set_k_loop:
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v245 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11)
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225] offset:32768
	ds_store_b64 v244, v[226:227] offset:33024
	ds_store_b64 v244, v[228:229] offset:33280
	ds_store_b64 v244, v[230:231] offset:33536
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:49152
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:49408
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:49664
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:49920
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v247 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	global_load_u16 v221, v242, s[30:31] offset:4
	global_load_u16 v222, v242, s[32:33] offset:4
	global_load_b32 v212, v243, s[34:35]
	global_load_b32 v213, v243, s[34:35] offset:1152
	global_load_b32 v214, v243, s[34:35] offset:2304
	global_load_b32 v215, v243, s[34:35] offset:3456
	s_add_u32 s28, s28, 0x88
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 0x88
	s_addc_u32 s31, s31, 0
	s_add_u32 s32, s32, 0x88
	s_addc_u32 s33, s33, 0
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29]
	global_load_b64 v[226:227], v240, s[28:29] offset:16
	global_load_b64 v[228:229], v240, s[28:29] offset:32
	global_load_b64 v[230:231], v240, s[28:29] offset:48
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v247 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11)
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225]
	ds_store_b64 v244, v[226:227] offset:256
	ds_store_b64 v244, v[228:229] offset:512
	ds_store_b64 v244, v[230:231] offset:768
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:16384
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:16640
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:16896
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:17152
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v245 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	global_load_u16 v221, v242, s[30:31]
	global_load_u16 v222, v242, s[32:33]
	global_load_b32 v208, v243, s[34:35]
	global_load_b32 v209, v243, s[34:35] offset:1152
	global_load_b32 v210, v243, s[34:35] offset:2304
	global_load_b32 v211, v243, s[34:35] offset:3456
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29] offset:64
	global_load_b64 v[226:227], v240, s[28:29] offset:80
	global_load_b64 v[228:229], v240, s[28:29] offset:96
	global_load_b64 v[230:231], v240, s[28:29] offset:112
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	s_add_i32 s22, s22, -1
	s_cmp_lg_u32 s22, 0
	s_cbranch_scc1 .Lv2b_set_k_loop
	.Lv2b_set_k_loop_end:
	.Lv2b_set_tail:
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v245 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11)
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225] offset:32768
	ds_store_b64 v244, v[226:227] offset:33024
	ds_store_b64 v244, v[228:229] offset:33280
	ds_store_b64 v244, v[230:231] offset:33536
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:49152
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:49408
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:49664
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:49920
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v247 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	global_load_u16 v221, v242, s[30:31] offset:4
	global_load_u16 v222, v242, s[32:33] offset:4
	global_load_b32 v212, v243, s[34:35]
	global_load_b32 v213, v243, s[34:35] offset:1152
	global_load_b32 v214, v243, s[34:35] offset:2304
	global_load_b32 v215, v243, s[34:35] offset:3456
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v247 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(3)
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(2)
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(1)
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(0)
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	global_store_b128 v223, v[32:35], s[36:37]
	ds_load_2addr_b64 v[172:175], v247 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[36:39], s[36:37] offset:16
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[40:43], s[48:49]
	ds_load_2addr_b64 v[168:171], v247 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[44:47], s[48:49] offset:16
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[48:51], s[50:51]
	ds_load_2addr_b64 v[172:175], v247 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[52:55], s[50:51] offset:16
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[56:59], s[52:53]
	ds_load_2addr_b64 v[168:171], v248 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[60:63], s[52:53] offset:16
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	global_store_b128 v223, v[64:67], s[36:37] offset:64
	ds_load_2addr_b64 v[172:175], v248 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[68:71], s[36:37] offset:80
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[72:75], s[48:49] offset:64
	ds_load_2addr_b64 v[168:171], v248 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[76:79], s[48:49] offset:80
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[80:83], s[50:51] offset:64
	ds_load_2addr_b64 v[172:175], v248 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[84:87], s[50:51] offset:80
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[88:91], s[52:53] offset:64
	ds_load_2addr_b64 v[168:171], v248 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[92:95], s[52:53] offset:80
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	global_store_b128 v223, v[96:99], s[36:37] offset:128
	ds_load_2addr_b64 v[172:175], v248 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[100:103], s[36:37] offset:144
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[104:107], s[48:49] offset:128
	ds_load_2addr_b64 v[168:171], v248 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[108:111], s[48:49] offset:144
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[112:115], s[50:51] offset:128
	ds_load_2addr_b64 v[172:175], v248 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[116:119], s[50:51] offset:144
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[120:123], s[52:53] offset:128
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	global_store_b128 v223, v[124:127], s[52:53] offset:144
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	.Lv2b_set_epilogue:
	.Lv2b_set_epi_body:
	global_store_b128 v223, v[128:131], s[36:37] offset:192
	global_store_b128 v223, v[132:135], s[36:37] offset:208
	global_store_b128 v223, v[136:139], s[48:49] offset:192
	global_store_b128 v223, v[140:143], s[48:49] offset:208
	global_store_b128 v223, v[144:147], s[50:51] offset:192
	global_store_b128 v223, v[148:151], s[50:51] offset:208
	global_store_b128 v223, v[152:155], s[52:53] offset:192
	global_store_b128 v223, v[156:159], s[52:53] offset:208
	.Lv2b_set_end:
	s_mov_b32 m0, 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lgemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151_end:
.size gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151, .Lgemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151_end-gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 36
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
	.amdhsa_next_free_vgpr 256
	.amdhsa_next_free_sgpr 54
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151_end-gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151)<<4)&1008)>>4
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
.protected gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151
.globl gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151
.p2align 8
.type gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151,@function
gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b64 s[16:17], s[0:1], 0x20
	v_lshrrev_b32_e32 v1, 5, v0
	v_and_b32_e32 v2, 31, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_readfirstlane_b32 s21, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v3, 15, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_lshrrev_b32_e32 v4, 4, v2
	s_lshr_b32 s23, s21, 2
	s_and_b32 s24, s21, 3
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s20, s15, 8
	s_add_i32 s22, s20, -8
	s_mulk_i32 s20, 0x88
	s_lshl_b32 s4, 1, s17
	s_add_i32 s4, s4, -1
	s_and_b32 s4, s2, s4
	s_lshl_b32 s27, s3, s17
	s_add_i32 s27, s27, s4
	s_lshr_b32 s40, s2, s17
	s_lshl_b32 s4, s27, 8
	s_lshl_b32 s5, s21, 4
	s_add_u32 s5, s4, s5
	s_mul_i32 s42, s5, s20
	s_mul_hi_u32 s43, s5, s20
	s_add_u32 s28, s8, s42
	s_addc_u32 s29, s9, s43
	s_lshl_b32 s5, s23, 6
	s_add_u32 s5, s4, s5
	s_mul_i32 s42, s5, s20
	s_mul_hi_u32 s43, s5, s20
	s_add_u32 s30, s8, s42
	s_addc_u32 s31, s9, s43
	s_lshl_b32 s5, s20, 4
	s_add_u32 s32, s30, s5
	s_addc_u32 s33, s31, 0
	s_mul_i32 s5, s40, 0x4800
	s_add_u32 s34, s10, s5
	s_addc_u32 s35, s11, 0
	s_mul_i32 s25, s16, 0x48
	s_lshl_b32 s26, s14, 6
	s_lshl_b32 s5, s40, 8
	s_mul_hi_u32 s43, s5, s14
	s_mul_i32 s42, s5, s14
	s_add_u32 s42, s42, s4
	s_addc_u32 s43, s43, 0
	s_lshl_b64 s[42:43], s[42:43], 2
	s_add_u32 s36, s12, s42
	s_addc_u32 s37, s13, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v6, v4, 3, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v240, v3, s20, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v5, s21, 4, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v241, v5, 0x48, v6
	v_lshlrev_b32_e32 v7, 3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v16, v4, 7, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v244, s21, 10, v16
	v_and_b32_e32 v8, 1, v3
	v_lshrrev_b32_e32 v9, 1, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 3, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 6, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v245, s23, 12, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v246, 0x800, v245
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v247, 0x8000, v245
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v248, 0x8800, v245
	v_lshl_add_u32 v10, s24, 12, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v249, 0x4000, v10
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v250, 0x4800, v10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v251, 0xc000, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v252, 0xc800, v10
	v_lshrrev_b32_e32 v12, 3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v12, 5, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v12, v4, 3, v12
	v_and_b32_e32 v13, 7, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_u32_u24_e32 v242, s20, v12
	v_lshl_add_u32 v14, s24, 6, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_u32_u24_e32 v243, 0x48, v14
	v_lshlrev_b32_e32 v15, 3, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v15, s23, 6, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v15, v14, s14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v223, 2, v15
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29]
	global_load_b64 v[226:227], v240, s[28:29] offset:16
	global_load_b64 v[228:229], v240, s[28:29] offset:32
	global_load_b64 v[230:231], v240, s[28:29] offset:48
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	v_mov_b32_e32 v160, 0x4b400000
	v_mov_b32_e32 v161, 0x4b400000
	v_mov_b32_e32 v162, 0x4b400000
	v_mov_b32_e32 v163, 0x4b400000
	v_mov_b32_e32 v164, 0x4b400000
	v_mov_b32_e32 v165, 0x4b400000
	v_mov_b32_e32 v166, 0x4b400000
	v_mov_b32_e32 v167, 0x4b400000
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
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225]
	ds_store_b64 v244, v[226:227] offset:256
	ds_store_b64 v244, v[228:229] offset:512
	ds_store_b64 v244, v[230:231] offset:768
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:16384
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:16640
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:16896
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:17152
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_load_2addr_b64 v[168:171], v245 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	.Lv2b_add_k_begin:
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .Lv2b_add_k_loop_end
	.Lv2b_add_k_loop:
	global_load_u16 v221, v242, s[30:31]
	global_load_u16 v222, v242, s[32:33]
	global_load_b32 v208, v243, s[34:35]
	global_load_b32 v209, v243, s[34:35] offset:1152
	global_load_b32 v210, v243, s[34:35] offset:2304
	global_load_b32 v211, v243, s[34:35] offset:3456
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29] offset:64
	global_load_b64 v[226:227], v240, s[28:29] offset:80
	global_load_b64 v[228:229], v240, s[28:29] offset:96
	global_load_b64 v[230:231], v240, s[28:29] offset:112
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v245 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11) lgkmcnt(10)
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	s_waitcnt lgkmcnt(9)
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	s_waitcnt lgkmcnt(8)
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	s_waitcnt lgkmcnt(7)
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	s_waitcnt lgkmcnt(6)
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	s_waitcnt lgkmcnt(5)
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	s_waitcnt lgkmcnt(4)
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	s_waitcnt lgkmcnt(3)
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225] offset:32768
	ds_store_b64 v244, v[226:227] offset:33024
	ds_store_b64 v244, v[228:229] offset:33280
	ds_store_b64 v244, v[230:231] offset:33536
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:49152
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:49408
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:49664
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:49920
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v247 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	global_load_u16 v221, v242, s[30:31] offset:4
	global_load_u16 v222, v242, s[32:33] offset:4
	global_load_b32 v212, v243, s[34:35]
	global_load_b32 v213, v243, s[34:35] offset:1152
	global_load_b32 v214, v243, s[34:35] offset:2304
	global_load_b32 v215, v243, s[34:35] offset:3456
	s_add_u32 s28, s28, 0x88
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 0x88
	s_addc_u32 s31, s31, 0
	s_add_u32 s32, s32, 0x88
	s_addc_u32 s33, s33, 0
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29]
	global_load_b64 v[226:227], v240, s[28:29] offset:16
	global_load_b64 v[228:229], v240, s[28:29] offset:32
	global_load_b64 v[230:231], v240, s[28:29] offset:48
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v247 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11) lgkmcnt(10)
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	s_waitcnt lgkmcnt(9)
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	s_waitcnt lgkmcnt(8)
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	s_waitcnt lgkmcnt(7)
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	s_waitcnt lgkmcnt(6)
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	s_waitcnt lgkmcnt(5)
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	s_waitcnt lgkmcnt(4)
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	s_waitcnt lgkmcnt(3)
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225]
	ds_store_b64 v244, v[226:227] offset:256
	ds_store_b64 v244, v[228:229] offset:512
	ds_store_b64 v244, v[230:231] offset:768
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:16384
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:16640
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:16896
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:17152
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v245 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	s_add_i32 s22, s22, -1
	s_cmp_lg_u32 s22, 0
	s_cbranch_scc1 .Lv2b_add_k_loop
	.Lv2b_add_k_loop_end:
	s_mov_b32 s22, 4
	s_mov_b64 s[38:39], s[36:37]
	.Lv2b_add_touch_loop:
	global_load_u16 v221, v242, s[30:31]
	global_load_u16 v222, v242, s[32:33]
	global_load_b32 v208, v243, s[34:35]
	global_load_b32 v209, v243, s[34:35] offset:1152
	global_load_b32 v210, v243, s[34:35] offset:2304
	global_load_b32 v211, v243, s[34:35] offset:3456
	global_load_b32 v216, v223, s[38:39]
	global_load_b32 v217, v223, s[38:39] offset:64
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29] offset:64
	global_load_b64 v[226:227], v240, s[28:29] offset:80
	global_load_b64 v[228:229], v240, s[28:29] offset:96
	global_load_b64 v[230:231], v240, s[28:29] offset:112
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(15)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v245 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13) lgkmcnt(10)
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	s_waitcnt lgkmcnt(9)
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	s_waitcnt lgkmcnt(8)
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	s_waitcnt lgkmcnt(7)
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	s_waitcnt lgkmcnt(6)
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	s_waitcnt lgkmcnt(5)
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	s_waitcnt lgkmcnt(4)
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	s_waitcnt lgkmcnt(3)
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(12)
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(11)
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225] offset:32768
	ds_store_b64 v244, v[226:227] offset:33024
	ds_store_b64 v244, v[228:229] offset:33280
	ds_store_b64 v244, v[230:231] offset:33536
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:49152
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:49408
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:49664
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:49920
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v247 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	global_load_u16 v221, v242, s[30:31] offset:4
	global_load_u16 v222, v242, s[32:33] offset:4
	global_load_b32 v212, v243, s[34:35]
	global_load_b32 v213, v243, s[34:35] offset:1152
	global_load_b32 v214, v243, s[34:35] offset:2304
	global_load_b32 v215, v243, s[34:35] offset:3456
	global_load_b32 v216, v223, s[38:39] offset:128
	global_load_b32 v217, v223, s[38:39] offset:192
	s_add_u32 s28, s28, 0x88
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 0x88
	s_addc_u32 s31, s31, 0
	s_add_u32 s32, s32, 0x88
	s_addc_u32 s33, s33, 0
	s_add_u32 s38, s38, s26
	s_addc_u32 s39, s39, 0
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29]
	global_load_b64 v[226:227], v240, s[28:29] offset:16
	global_load_b64 v[228:229], v240, s[28:29] offset:32
	global_load_b64 v[230:231], v240, s[28:29] offset:48
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(15)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v247 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13) lgkmcnt(10)
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	s_waitcnt lgkmcnt(9)
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	s_waitcnt lgkmcnt(8)
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	s_waitcnt lgkmcnt(7)
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	s_waitcnt lgkmcnt(6)
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	s_waitcnt lgkmcnt(5)
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	s_waitcnt lgkmcnt(4)
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	s_waitcnt lgkmcnt(3)
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(12)
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(11)
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225]
	ds_store_b64 v244, v[226:227] offset:256
	ds_store_b64 v244, v[228:229] offset:512
	ds_store_b64 v244, v[230:231] offset:768
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:16384
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:16640
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:16896
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:17152
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v245 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	s_add_i32 s22, s22, -1
	s_cmp_lg_u32 s22, 0
	s_cbranch_scc1 .Lv2b_add_touch_loop
	.Lv2b_add_touch_loop_end:
	s_mov_b32 s22, 3
	.Lv2b_add_late_loop:
	global_load_u16 v221, v242, s[30:31]
	global_load_u16 v222, v242, s[32:33]
	global_load_b32 v208, v243, s[34:35]
	global_load_b32 v209, v243, s[34:35] offset:1152
	global_load_b32 v210, v243, s[34:35] offset:2304
	global_load_b32 v211, v243, s[34:35] offset:3456
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29] offset:64
	global_load_b64 v[226:227], v240, s[28:29] offset:80
	global_load_b64 v[228:229], v240, s[28:29] offset:96
	global_load_b64 v[230:231], v240, s[28:29] offset:112
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v245 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11) lgkmcnt(10)
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	s_waitcnt lgkmcnt(9)
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	s_waitcnt lgkmcnt(8)
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	s_waitcnt lgkmcnt(7)
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	s_waitcnt lgkmcnt(6)
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	s_waitcnt lgkmcnt(5)
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	s_waitcnt lgkmcnt(4)
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	s_waitcnt lgkmcnt(3)
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225] offset:32768
	ds_store_b64 v244, v[226:227] offset:33024
	ds_store_b64 v244, v[228:229] offset:33280
	ds_store_b64 v244, v[230:231] offset:33536
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:49152
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:49408
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:49664
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:49920
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v247 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	global_load_u16 v221, v242, s[30:31] offset:4
	global_load_u16 v222, v242, s[32:33] offset:4
	global_load_b32 v212, v243, s[34:35]
	global_load_b32 v213, v243, s[34:35] offset:1152
	global_load_b32 v214, v243, s[34:35] offset:2304
	global_load_b32 v215, v243, s[34:35] offset:3456
	s_add_u32 s28, s28, 0x88
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 0x88
	s_addc_u32 s31, s31, 0
	s_add_u32 s32, s32, 0x88
	s_addc_u32 s33, s33, 0
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29]
	global_load_b64 v[226:227], v240, s[28:29] offset:16
	global_load_b64 v[228:229], v240, s[28:29] offset:32
	global_load_b64 v[230:231], v240, s[28:29] offset:48
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v247 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11) lgkmcnt(10)
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	s_waitcnt lgkmcnt(9)
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	s_waitcnt lgkmcnt(8)
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	s_waitcnt lgkmcnt(7)
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	s_waitcnt lgkmcnt(6)
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	s_waitcnt lgkmcnt(5)
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	s_waitcnt lgkmcnt(4)
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	s_waitcnt lgkmcnt(3)
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225]
	ds_store_b64 v244, v[226:227] offset:256
	ds_store_b64 v244, v[228:229] offset:512
	ds_store_b64 v244, v[230:231] offset:768
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:16384
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:16640
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:16896
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:17152
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v245 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	s_add_i32 s22, s22, -1
	s_cmp_lg_u32 s22, 0
	s_cbranch_scc1 .Lv2b_add_late_loop
	.Lv2b_add_late_loop_end:
	.Lv2b_add_tail:
	global_load_u16 v221, v242, s[30:31]
	global_load_u16 v222, v242, s[32:33]
	global_load_b32 v208, v243, s[34:35]
	global_load_b32 v209, v243, s[34:35] offset:1152
	global_load_b32 v210, v243, s[34:35] offset:2304
	global_load_b32 v211, v243, s[34:35] offset:3456
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29] offset:64
	global_load_b64 v[226:227], v240, s[28:29] offset:80
	global_load_b64 v[228:229], v240, s[28:29] offset:96
	global_load_b64 v[230:231], v240, s[28:29] offset:112
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v245 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11) lgkmcnt(10)
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	s_waitcnt lgkmcnt(9)
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	s_waitcnt lgkmcnt(8)
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	s_waitcnt lgkmcnt(7)
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	s_waitcnt lgkmcnt(6)
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	s_waitcnt lgkmcnt(5)
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	s_waitcnt lgkmcnt(4)
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	s_waitcnt lgkmcnt(3)
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225] offset:32768
	ds_store_b64 v244, v[226:227] offset:33024
	ds_store_b64 v244, v[228:229] offset:33280
	ds_store_b64 v244, v[230:231] offset:33536
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:49152
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:49408
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:49664
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:49920
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v247 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	global_load_u16 v221, v242, s[30:31] offset:4
	global_load_u16 v222, v242, s[32:33] offset:4
	global_load_b32 v212, v243, s[34:35]
	global_load_b32 v213, v243, s[34:35] offset:1152
	global_load_b32 v214, v243, s[34:35] offset:2304
	global_load_b32 v215, v243, s[34:35] offset:3456
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v247 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(3) lgkmcnt(10)
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	s_waitcnt lgkmcnt(9)
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	s_waitcnt lgkmcnt(8)
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	s_waitcnt lgkmcnt(7)
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	s_waitcnt lgkmcnt(6)
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	s_waitcnt lgkmcnt(5)
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	s_waitcnt lgkmcnt(4)
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	s_waitcnt lgkmcnt(3)
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(2)
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(1)
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(0)
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	.Lv2b_add_epilogue:
	v_mov_b32_e32 v0, v223
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s26, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s26, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s26, v2
	.Lv2b_add_epi_body:
	global_load_b128 v[160:163], v0, s[36:37]
	global_load_b128 v[164:167], v0, s[36:37] offset:16
	global_load_b128 v[168:171], v1, s[36:37]
	global_load_b128 v[172:175], v1, s[36:37] offset:16
	global_load_b128 v[176:179], v2, s[36:37]
	global_load_b128 v[180:183], v2, s[36:37] offset:16
	global_load_b128 v[184:187], v3, s[36:37]
	global_load_b128 v[188:191], v3, s[36:37] offset:16
	global_load_b128 v[192:195], v0, s[36:37] offset:64
	global_load_b128 v[196:199], v0, s[36:37] offset:80
	global_load_b128 v[200:203], v1, s[36:37] offset:64
	global_load_b128 v[204:207], v1, s[36:37] offset:80
	global_load_b128 v[208:211], v2, s[36:37] offset:64
	global_load_b128 v[212:215], v2, s[36:37] offset:80
	global_load_b128 v[216:219], v3, s[36:37] offset:64
	global_load_b128 v[220:223], v3, s[36:37] offset:80
	global_load_b128 v[224:227], v0, s[36:37] offset:128
	global_load_b128 v[228:231], v0, s[36:37] offset:144
	global_load_b128 v[232:235], v1, s[36:37] offset:128
	global_load_b128 v[236:239], v1, s[36:37] offset:144
	global_load_b128 v[240:243], v2, s[36:37] offset:128
	global_load_b128 v[244:247], v2, s[36:37] offset:144
	global_load_b128 v[248:251], v3, s[36:37] offset:128
	global_load_b128 v[252:255], v3, s[36:37] offset:144
	s_waitcnt vmcnt(23)
	v_dual_add_f32 v32, v160, v32 :: v_dual_add_f32 v33, v161, v33
	v_dual_add_f32 v34, v162, v34 :: v_dual_add_f32 v35, v163, v35
	s_waitcnt vmcnt(22)
	v_dual_add_f32 v36, v164, v36 :: v_dual_add_f32 v37, v165, v37
	v_dual_add_f32 v38, v166, v38 :: v_dual_add_f32 v39, v167, v39
	s_waitcnt vmcnt(21)
	v_dual_add_f32 v40, v168, v40 :: v_dual_add_f32 v41, v169, v41
	v_dual_add_f32 v42, v170, v42 :: v_dual_add_f32 v43, v171, v43
	s_waitcnt vmcnt(20)
	v_dual_add_f32 v44, v172, v44 :: v_dual_add_f32 v45, v173, v45
	v_dual_add_f32 v46, v174, v46 :: v_dual_add_f32 v47, v175, v47
	s_waitcnt vmcnt(19)
	v_dual_add_f32 v48, v176, v48 :: v_dual_add_f32 v49, v177, v49
	v_dual_add_f32 v50, v178, v50 :: v_dual_add_f32 v51, v179, v51
	s_waitcnt vmcnt(18)
	v_dual_add_f32 v52, v180, v52 :: v_dual_add_f32 v53, v181, v53
	v_dual_add_f32 v54, v182, v54 :: v_dual_add_f32 v55, v183, v55
	s_waitcnt vmcnt(17)
	v_dual_add_f32 v56, v184, v56 :: v_dual_add_f32 v57, v185, v57
	v_dual_add_f32 v58, v186, v58 :: v_dual_add_f32 v59, v187, v59
	s_waitcnt vmcnt(16)
	v_dual_add_f32 v60, v188, v60 :: v_dual_add_f32 v61, v189, v61
	v_dual_add_f32 v62, v190, v62 :: v_dual_add_f32 v63, v191, v63
	global_load_b128 v[160:163], v0, s[36:37] offset:192
	global_load_b128 v[164:167], v0, s[36:37] offset:208
	global_load_b128 v[168:171], v1, s[36:37] offset:192
	global_load_b128 v[172:175], v1, s[36:37] offset:208
	global_load_b128 v[176:179], v2, s[36:37] offset:192
	global_load_b128 v[180:183], v2, s[36:37] offset:208
	global_load_b128 v[184:187], v3, s[36:37] offset:192
	global_load_b128 v[188:191], v3, s[36:37] offset:208
	global_store_b128 v0, v[32:35], s[36:37]
	global_store_b128 v0, v[36:39], s[36:37] offset:16
	global_store_b128 v1, v[40:43], s[36:37]
	global_store_b128 v1, v[44:47], s[36:37] offset:16
	global_store_b128 v2, v[48:51], s[36:37]
	global_store_b128 v2, v[52:55], s[36:37] offset:16
	global_store_b128 v3, v[56:59], s[36:37]
	global_store_b128 v3, v[60:63], s[36:37] offset:16
	s_waitcnt vmcnt(23)
	v_dual_add_f32 v64, v192, v64 :: v_dual_add_f32 v65, v193, v65
	v_dual_add_f32 v66, v194, v66 :: v_dual_add_f32 v67, v195, v67
	s_waitcnt vmcnt(22)
	v_dual_add_f32 v68, v196, v68 :: v_dual_add_f32 v69, v197, v69
	v_dual_add_f32 v70, v198, v70 :: v_dual_add_f32 v71, v199, v71
	s_waitcnt vmcnt(21)
	v_dual_add_f32 v72, v200, v72 :: v_dual_add_f32 v73, v201, v73
	v_dual_add_f32 v74, v202, v74 :: v_dual_add_f32 v75, v203, v75
	s_waitcnt vmcnt(20)
	v_dual_add_f32 v76, v204, v76 :: v_dual_add_f32 v77, v205, v77
	v_dual_add_f32 v78, v206, v78 :: v_dual_add_f32 v79, v207, v79
	s_waitcnt vmcnt(19)
	v_dual_add_f32 v80, v208, v80 :: v_dual_add_f32 v81, v209, v81
	v_dual_add_f32 v82, v210, v82 :: v_dual_add_f32 v83, v211, v83
	s_waitcnt vmcnt(18)
	v_dual_add_f32 v84, v212, v84 :: v_dual_add_f32 v85, v213, v85
	v_dual_add_f32 v86, v214, v86 :: v_dual_add_f32 v87, v215, v87
	s_waitcnt vmcnt(17)
	v_dual_add_f32 v88, v216, v88 :: v_dual_add_f32 v89, v217, v89
	v_dual_add_f32 v90, v218, v90 :: v_dual_add_f32 v91, v219, v91
	s_waitcnt vmcnt(16)
	v_dual_add_f32 v92, v220, v92 :: v_dual_add_f32 v93, v221, v93
	v_dual_add_f32 v94, v222, v94 :: v_dual_add_f32 v95, v223, v95
	global_store_b128 v0, v[64:67], s[36:37] offset:64
	global_store_b128 v0, v[68:71], s[36:37] offset:80
	global_store_b128 v1, v[72:75], s[36:37] offset:64
	global_store_b128 v1, v[76:79], s[36:37] offset:80
	global_store_b128 v2, v[80:83], s[36:37] offset:64
	global_store_b128 v2, v[84:87], s[36:37] offset:80
	global_store_b128 v3, v[88:91], s[36:37] offset:64
	global_store_b128 v3, v[92:95], s[36:37] offset:80
	s_waitcnt vmcnt(15)
	v_dual_add_f32 v96, v224, v96 :: v_dual_add_f32 v97, v225, v97
	v_dual_add_f32 v98, v226, v98 :: v_dual_add_f32 v99, v227, v99
	s_waitcnt vmcnt(14)
	v_dual_add_f32 v100, v228, v100 :: v_dual_add_f32 v101, v229, v101
	v_dual_add_f32 v102, v230, v102 :: v_dual_add_f32 v103, v231, v103
	s_waitcnt vmcnt(13)
	v_dual_add_f32 v104, v232, v104 :: v_dual_add_f32 v105, v233, v105
	v_dual_add_f32 v106, v234, v106 :: v_dual_add_f32 v107, v235, v107
	s_waitcnt vmcnt(12)
	v_dual_add_f32 v108, v236, v108 :: v_dual_add_f32 v109, v237, v109
	v_dual_add_f32 v110, v238, v110 :: v_dual_add_f32 v111, v239, v111
	s_waitcnt vmcnt(11)
	v_dual_add_f32 v112, v240, v112 :: v_dual_add_f32 v113, v241, v113
	v_dual_add_f32 v114, v242, v114 :: v_dual_add_f32 v115, v243, v115
	s_waitcnt vmcnt(10)
	v_dual_add_f32 v116, v244, v116 :: v_dual_add_f32 v117, v245, v117
	v_dual_add_f32 v118, v246, v118 :: v_dual_add_f32 v119, v247, v119
	s_waitcnt vmcnt(9)
	v_dual_add_f32 v120, v248, v120 :: v_dual_add_f32 v121, v249, v121
	v_dual_add_f32 v122, v250, v122 :: v_dual_add_f32 v123, v251, v123
	s_waitcnt vmcnt(8)
	v_dual_add_f32 v124, v252, v124 :: v_dual_add_f32 v125, v253, v125
	v_dual_add_f32 v126, v254, v126 :: v_dual_add_f32 v127, v255, v127
	global_store_b128 v0, v[96:99], s[36:37] offset:128
	global_store_b128 v0, v[100:103], s[36:37] offset:144
	global_store_b128 v1, v[104:107], s[36:37] offset:128
	global_store_b128 v1, v[108:111], s[36:37] offset:144
	global_store_b128 v2, v[112:115], s[36:37] offset:128
	global_store_b128 v2, v[116:119], s[36:37] offset:144
	global_store_b128 v3, v[120:123], s[36:37] offset:128
	global_store_b128 v3, v[124:127], s[36:37] offset:144
	s_waitcnt vmcnt(7)
	v_dual_add_f32 v128, v160, v128 :: v_dual_add_f32 v129, v161, v129
	v_dual_add_f32 v130, v162, v130 :: v_dual_add_f32 v131, v163, v131
	s_waitcnt vmcnt(6)
	v_dual_add_f32 v132, v164, v132 :: v_dual_add_f32 v133, v165, v133
	v_dual_add_f32 v134, v166, v134 :: v_dual_add_f32 v135, v167, v135
	s_waitcnt vmcnt(5)
	v_dual_add_f32 v136, v168, v136 :: v_dual_add_f32 v137, v169, v137
	v_dual_add_f32 v138, v170, v138 :: v_dual_add_f32 v139, v171, v139
	s_waitcnt vmcnt(4)
	v_dual_add_f32 v140, v172, v140 :: v_dual_add_f32 v141, v173, v141
	v_dual_add_f32 v142, v174, v142 :: v_dual_add_f32 v143, v175, v143
	s_waitcnt vmcnt(3)
	v_dual_add_f32 v144, v176, v144 :: v_dual_add_f32 v145, v177, v145
	v_dual_add_f32 v146, v178, v146 :: v_dual_add_f32 v147, v179, v147
	s_waitcnt vmcnt(2)
	v_dual_add_f32 v148, v180, v148 :: v_dual_add_f32 v149, v181, v149
	v_dual_add_f32 v150, v182, v150 :: v_dual_add_f32 v151, v183, v151
	s_waitcnt vmcnt(1)
	v_dual_add_f32 v152, v184, v152 :: v_dual_add_f32 v153, v185, v153
	v_dual_add_f32 v154, v186, v154 :: v_dual_add_f32 v155, v187, v155
	s_waitcnt vmcnt(0)
	v_dual_add_f32 v156, v188, v156 :: v_dual_add_f32 v157, v189, v157
	v_dual_add_f32 v158, v190, v158 :: v_dual_add_f32 v159, v191, v159
	global_store_b128 v0, v[128:131], s[36:37] offset:192
	global_store_b128 v0, v[132:135], s[36:37] offset:208
	global_store_b128 v1, v[136:139], s[36:37] offset:192
	global_store_b128 v1, v[140:143], s[36:37] offset:208
	global_store_b128 v2, v[144:147], s[36:37] offset:192
	global_store_b128 v2, v[148:151], s[36:37] offset:208
	global_store_b128 v3, v[152:155], s[36:37] offset:192
	global_store_b128 v3, v[156:159], s[36:37] offset:208
	.Lv2b_add_end:
	s_mov_b32 m0, 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lgemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151_end:
.size gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151, .Lgemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151_end-gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151
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
	.amdhsa_system_sgpr_workgroup_id_y 1
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 256
	.amdhsa_next_free_sgpr 44
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151_end-gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151)<<4)&1008)>>4
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
.protected gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151
.globl gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151
.p2align 8
.type gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151,@function
gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b64 s[16:17], s[0:1], 0x20
	s_load_b32 s18, s[0:1], 0x28
	v_lshrrev_b32_e32 v1, 5, v0
	v_and_b32_e32 v2, 31, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_readfirstlane_b32 s21, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v3, 15, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_lshrrev_b32_e32 v4, 4, v2
	s_lshr_b32 s23, s21, 2
	s_and_b32 s24, s21, 3
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s20, s17, 8
	s_add_i32 s22, s20, -1
	s_mulk_i32 s20, 0x88
	s_lshl_b32 s4, s3, 7
	s_lshr_b32 s5, s21, 1
	s_lshl_b32 s5, s5, 4
	s_add_u32 s5, s4, s5
	s_bitcmp1_b32 s21, 0
	s_cselect_b64 s[6:7], s[10:11], s[8:9]
	s_mul_i32 s42, s5, s20
	s_mul_hi_u32 s43, s5, s20
	s_add_u32 s28, s6, s42
	s_addc_u32 s29, s7, s43
	s_lshl_b32 s5, s23, 5
	s_add_u32 s5, s4, s5
	s_mul_i32 s42, s5, s20
	s_mul_hi_u32 s43, s5, s20
	s_add_u32 s30, s8, s42
	s_addc_u32 s31, s9, s43
	s_mul_i32 s42, s5, s20
	s_mul_hi_u32 s43, s5, s20
	s_add_u32 s32, s10, s42
	s_addc_u32 s33, s11, s43
	s_mul_i32 s5, s2, 0x4800
	s_add_u32 s34, s12, s5
	s_addc_u32 s35, s13, 0
	s_mul_i32 s25, s18, 0x48
	s_lshl_b32 s26, s16, 6
	s_lshl_b32 s5, s2, 8
	s_mul_hi_u32 s43, s5, s16
	s_mul_i32 s42, s5, s16
	s_add_u32 s42, s42, s4
	s_addc_u32 s43, s43, 0
	s_lshl_b64 s[42:43], s[42:43], 2
	s_add_u32 s36, s14, s42
	s_addc_u32 s37, s15, s43
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v6, v4, 3, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v240, v3, s20, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v5, s21, 4, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v241, v5, 0x48, v6
	v_lshlrev_b32_e32 v7, 3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v16, v4, 7, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v244, s21, 10, v16
	v_and_b32_e32 v8, 1, v3
	v_lshrrev_b32_e32 v9, 1, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 3, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 6, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v245, s23, 12, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v246, 0x800, v245
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v247, 0x8000, v245
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v248, 0x8800, v245
	v_lshl_add_u32 v10, s24, 12, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v249, 0x4000, v10
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v250, 0x4800, v10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v251, 0xc000, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v252, 0xc800, v10
	v_lshrrev_b32_e32 v12, 3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v12, 4, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v12, v4, 3, v12
	v_and_b32_e32 v13, 7, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_u32_u24_e32 v242, s20, v12
	v_lshl_add_u32 v14, s24, 6, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_u32_u24_e32 v243, 0x48, v14
	v_lshlrev_b32_e32 v15, 3, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v15, s23, 5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32_u24 v15, v14, s16, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v223, 2, v15
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29]
	global_load_b64 v[226:227], v240, s[28:29] offset:16
	global_load_b64 v[228:229], v240, s[28:29] offset:32
	global_load_b64 v[230:231], v240, s[28:29] offset:48
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	v_mov_b32_e32 v160, 0x4b400000
	v_mov_b32_e32 v161, 0x4b400000
	v_mov_b32_e32 v162, 0x4b400000
	v_mov_b32_e32 v163, 0x4b400000
	v_mov_b32_e32 v164, 0x4b400000
	v_mov_b32_e32 v165, 0x4b400000
	v_mov_b32_e32 v166, 0x4b400000
	v_mov_b32_e32 v167, 0x4b400000
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
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225]
	ds_store_b64 v244, v[226:227] offset:256
	ds_store_b64 v244, v[228:229] offset:512
	ds_store_b64 v244, v[230:231] offset:768
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:16384
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:16640
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:16896
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:17152
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_load_2addr_b64 v[168:171], v245 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	global_load_u16 v221, v242, s[30:31]
	global_load_u16 v222, v242, s[32:33]
	global_load_b32 v208, v243, s[34:35]
	global_load_b32 v209, v243, s[34:35] offset:1152
	global_load_b32 v210, v243, s[34:35] offset:2304
	global_load_b32 v211, v243, s[34:35] offset:3456
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29] offset:64
	global_load_b64 v[226:227], v240, s[28:29] offset:80
	global_load_b64 v[228:229], v240, s[28:29] offset:96
	global_load_b64 v[230:231], v240, s[28:29] offset:112
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	.Lv2b_silu_k_begin:
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .Lv2b_silu_k_loop_end
	.Lv2b_silu_k_loop:
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v245 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11)
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225] offset:32768
	ds_store_b64 v244, v[226:227] offset:33024
	ds_store_b64 v244, v[228:229] offset:33280
	ds_store_b64 v244, v[230:231] offset:33536
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:49152
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:49408
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:49664
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:49920
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v247 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	global_load_u16 v221, v242, s[30:31] offset:4
	global_load_u16 v222, v242, s[32:33] offset:4
	global_load_b32 v212, v243, s[34:35]
	global_load_b32 v213, v243, s[34:35] offset:1152
	global_load_b32 v214, v243, s[34:35] offset:2304
	global_load_b32 v215, v243, s[34:35] offset:3456
	s_add_u32 s28, s28, 0x88
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 0x88
	s_addc_u32 s31, s31, 0
	s_add_u32 s32, s32, 0x88
	s_addc_u32 s33, s33, 0
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29]
	global_load_b64 v[226:227], v240, s[28:29] offset:16
	global_load_b64 v[228:229], v240, s[28:29] offset:32
	global_load_b64 v[230:231], v240, s[28:29] offset:48
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v247 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11)
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225]
	ds_store_b64 v244, v[226:227] offset:256
	ds_store_b64 v244, v[228:229] offset:512
	ds_store_b64 v244, v[230:231] offset:768
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:16384
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:16640
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:16896
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:17152
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v245 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	global_load_u16 v221, v242, s[30:31]
	global_load_u16 v222, v242, s[32:33]
	global_load_b32 v208, v243, s[34:35]
	global_load_b32 v209, v243, s[34:35] offset:1152
	global_load_b32 v210, v243, s[34:35] offset:2304
	global_load_b32 v211, v243, s[34:35] offset:3456
	s_add_u32 s34, s34, s25
	s_addc_u32 s35, s35, 0
	s_clause 0x3
	global_load_b64 v[224:225], v240, s[28:29] offset:64
	global_load_b64 v[226:227], v240, s[28:29] offset:80
	global_load_b64 v[228:229], v240, s[28:29] offset:96
	global_load_b64 v[230:231], v240, s[28:29] offset:112
	s_clause 0x3
	global_load_b64 v[232:233], v241, s[34:35]
	global_load_b64 v[234:235], v241, s[34:35] offset:16
	global_load_b64 v[236:237], v241, s[34:35] offset:32
	global_load_b64 v[238:239], v241, s[34:35] offset:48
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	s_add_i32 s22, s22, -1
	s_cmp_lg_u32 s22, 0
	s_cbranch_scc1 .Lv2b_silu_k_loop
	.Lv2b_silu_k_loop_end:
	.Lv2b_silu_tail:
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v245 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(11)
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(10)
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(9)
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(8)
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v245 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v245 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset1:16
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v249 offset1:128
	ds_load_2addr_b64 v[180:183], v250 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	s_waitcnt vmcnt(7)
	v_xor_b32_e32 v224, 0x88888888, v224
	v_xor_b32_e32 v225, 0x88888888, v225
	s_waitcnt vmcnt(6)
	v_xor_b32_e32 v226, 0x88888888, v226
	v_xor_b32_e32 v227, 0x88888888, v227
	s_waitcnt vmcnt(5)
	v_xor_b32_e32 v228, 0x88888888, v228
	v_xor_b32_e32 v229, 0x88888888, v229
	s_waitcnt vmcnt(4)
	v_xor_b32_e32 v230, 0x88888888, v230
	v_xor_b32_e32 v231, 0x88888888, v231
	ds_store_b64 v244, v[224:225] offset:32768
	ds_store_b64 v244, v[226:227] offset:33024
	ds_store_b64 v244, v[228:229] offset:33280
	ds_store_b64 v244, v[230:231] offset:33536
	s_waitcnt vmcnt(3)
	ds_store_b64 v244, v[232:233] offset:49152
	s_waitcnt vmcnt(2)
	ds_store_b64 v244, v[234:235] offset:49408
	s_waitcnt vmcnt(1)
	ds_store_b64 v244, v[236:237] offset:49664
	s_waitcnt vmcnt(0)
	ds_store_b64 v244, v[238:239] offset:49920
	ds_load_2addr_b64 v[184:187], v249 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v250 offset0:16 offset1:144
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v249 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v250 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v250 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v246 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v249 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v250 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v250 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v246 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v249 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v250 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v249 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v250 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_barrier
	ds_load_2addr_b64 v[168:171], v247 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	global_load_u16 v221, v242, s[30:31] offset:4
	global_load_u16 v222, v242, s[32:33] offset:4
	global_load_b32 v212, v243, s[34:35]
	global_load_b32 v213, v243, s[34:35] offset:1152
	global_load_b32 v214, v243, s[34:35] offset:2304
	global_load_b32 v215, v243, s[34:35] offset:3456
	v_dual_mul_f32 v202, v208, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v208, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v208, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v208, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v208, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v208, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v208, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v208, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v209, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v209, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v209, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v209, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v209, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v209, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v209, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v209, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v210, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v210, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v210, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v210, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v210, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v210, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v210, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v210, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v211, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v211, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v211, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v211, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v211, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v211, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v211, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v211, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(5)
	v_cvt_f32_f16_e64 v220, v221.l
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[168:171], v247 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	s_waitcnt vmcnt(3)
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v32, v202, v0 :: v_dual_fmac_f32 v33, v203, v1
	v_dual_fmac_f32 v34, v200, v2 :: v_dual_fmac_f32 v35, v201, v3
	v_dual_fmac_f32 v36, v206, v4 :: v_dual_fmac_f32 v37, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v204, v6 :: v_dual_fmac_f32 v39, v205, v7
	s_waitcnt vmcnt(2)
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v40, v202, v8 :: v_dual_fmac_f32 v41, v203, v9
	v_dual_fmac_f32 v42, v200, v10 :: v_dual_fmac_f32 v43, v201, v11
	v_dual_fmac_f32 v44, v206, v12 :: v_dual_fmac_f32 v45, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v46, v204, v14 :: v_dual_fmac_f32 v47, v205, v15
	s_waitcnt vmcnt(1)
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v48, v202, v16 :: v_dual_fmac_f32 v49, v203, v17
	v_dual_fmac_f32 v50, v200, v18 :: v_dual_fmac_f32 v51, v201, v19
	v_dual_fmac_f32 v52, v206, v20 :: v_dual_fmac_f32 v53, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v204, v22 :: v_dual_fmac_f32 v55, v205, v23
	s_waitcnt vmcnt(0)
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v56, v202, v24 :: v_dual_fmac_f32 v57, v203, v25
	v_dual_fmac_f32 v58, v200, v26 :: v_dual_fmac_f32 v59, v201, v27
	v_dual_fmac_f32 v60, v206, v28 :: v_dual_fmac_f32 v61, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v204, v30 :: v_dual_fmac_f32 v63, v205, v31
	v_cvt_f32_f16_e64 v255, v222.l
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,0)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,1)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,2)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,3)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,4)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,5)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,6)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,7)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v247 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v247 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset1:16
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v64, v202, v0 :: v_dual_fmac_f32 v65, v203, v1
	v_dual_fmac_f32 v66, v200, v2 :: v_dual_fmac_f32 v67, v201, v3
	v_dual_fmac_f32 v68, v206, v4 :: v_dual_fmac_f32 v69, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v70, v204, v6 :: v_dual_fmac_f32 v71, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v72, v202, v8 :: v_dual_fmac_f32 v73, v203, v9
	v_dual_fmac_f32 v74, v200, v10 :: v_dual_fmac_f32 v75, v201, v11
	v_dual_fmac_f32 v76, v206, v12 :: v_dual_fmac_f32 v77, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v78, v204, v14 :: v_dual_fmac_f32 v79, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v80, v202, v16 :: v_dual_fmac_f32 v81, v203, v17
	v_dual_fmac_f32 v82, v200, v18 :: v_dual_fmac_f32 v83, v201, v19
	v_dual_fmac_f32 v84, v206, v20 :: v_dual_fmac_f32 v85, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v86, v204, v22 :: v_dual_fmac_f32 v87, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v88, v202, v24 :: v_dual_fmac_f32 v89, v203, v25
	v_dual_fmac_f32 v90, v200, v26 :: v_dual_fmac_f32 v91, v201, v27
	v_dual_fmac_f32 v92, v206, v28 :: v_dual_fmac_f32 v93, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v94, v204, v30 :: v_dual_fmac_f32 v95, v205, v31
	ds_swizzle_b32 v192, v220 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v220 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v220 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v220 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v220 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v220 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v220 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v220 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:32 offset1:48
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:64 offset1:80
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:96 offset1:112
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:128 offset1:144
	ds_load_2addr_b64 v[176:179], v251 offset1:128
	ds_load_2addr_b64 v[180:183], v252 offset1:128
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v96, v202, v0 :: v_dual_fmac_f32 v97, v203, v1
	v_dual_fmac_f32 v98, v200, v2 :: v_dual_fmac_f32 v99, v201, v3
	v_dual_fmac_f32 v100, v206, v4 :: v_dual_fmac_f32 v101, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v102, v204, v6 :: v_dual_fmac_f32 v103, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v104, v202, v8 :: v_dual_fmac_f32 v105, v203, v9
	v_dual_fmac_f32 v106, v200, v10 :: v_dual_fmac_f32 v107, v201, v11
	v_dual_fmac_f32 v108, v206, v12 :: v_dual_fmac_f32 v109, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v110, v204, v14 :: v_dual_fmac_f32 v111, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v112, v202, v16 :: v_dual_fmac_f32 v113, v203, v17
	v_dual_fmac_f32 v114, v200, v18 :: v_dual_fmac_f32 v115, v201, v19
	v_dual_fmac_f32 v116, v206, v20 :: v_dual_fmac_f32 v117, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v118, v204, v22 :: v_dual_fmac_f32 v119, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v120, v202, v24 :: v_dual_fmac_f32 v121, v203, v25
	v_dual_fmac_f32 v122, v200, v26 :: v_dual_fmac_f32 v123, v201, v27
	v_dual_fmac_f32 v124, v206, v28 :: v_dual_fmac_f32 v125, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v126, v204, v30 :: v_dual_fmac_f32 v127, v205, v31
	ds_swizzle_b32 v192, v255 offset:swizzle(BROADCAST,16,8)
	ds_swizzle_b32 v193, v255 offset:swizzle(BROADCAST,16,9)
	ds_swizzle_b32 v194, v255 offset:swizzle(BROADCAST,16,10)
	ds_swizzle_b32 v195, v255 offset:swizzle(BROADCAST,16,11)
	ds_swizzle_b32 v196, v255 offset:swizzle(BROADCAST,16,12)
	ds_swizzle_b32 v197, v255 offset:swizzle(BROADCAST,16,13)
	ds_swizzle_b32 v198, v255 offset:swizzle(BROADCAST,16,14)
	ds_swizzle_b32 v199, v255 offset:swizzle(BROADCAST,16,15)
	ds_load_2addr_b64 v[184:187], v251 offset0:16 offset1:144
	ds_load_2addr_b64 v[188:191], v252 offset0:16 offset1:144
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[160:167] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[160:167] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[160:167] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:160 offset1:176
	ds_load_2addr_b64 v[176:179], v251 offset0:32 offset1:160
	ds_load_2addr_b64 v[180:183], v252 offset0:32 offset1:160
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:48 offset1:176
	ds_load_2addr_b64 v[188:191], v252 offset0:48 offset1:176
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[168:171], v248 offset0:192 offset1:208
	ds_load_2addr_b64 v[176:179], v251 offset0:64 offset1:192
	ds_load_2addr_b64 v[180:183], v252 offset0:64 offset1:192
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:80 offset1:208
	ds_load_2addr_b64 v[188:191], v252 offset0:80 offset1:208
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[168:169], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[168:169], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[168:169], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[168:169], v[182:183], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[172:175], v248 offset0:224 offset1:240
	ds_load_2addr_b64 v[176:179], v251 offset0:96 offset1:224
	ds_load_2addr_b64 v[180:183], v252 offset0:96 offset1:224
	s_waitcnt lgkmcnt(4)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[170:171], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[170:171], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[170:171], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[170:171], v[190:191], v[24:31] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[184:187], v251 offset0:112 offset1:240
	ds_load_2addr_b64 v[188:191], v252 offset0:112 offset1:240
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[172:173], v[176:177], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[172:173], v[178:179], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[172:173], v[180:181], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[172:173], v[182:183], v[24:31] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu4 v[0:7], v[174:175], v[184:185], v[0:7] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[8:15], v[174:175], v[186:187], v[8:15] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu4 v[16:23], v[174:175], v[188:189], v[16:23] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu4 v[24:31], v[174:175], v[190:191], v[24:31] neg_lo:[1,1,0]
	v_dual_mul_f32 v202, v212, v192 :: v_dual_add_f32 v1, 0xcb400000, v1
	v_dual_mul_f32 v203, v212, v193 :: v_dual_add_f32 v0, 0xcb400000, v0
	v_dual_mul_f32 v200, v212, v194 :: v_dual_add_f32 v3, 0xcb400000, v3
	v_dual_mul_f32 v201, v212, v195 :: v_dual_add_f32 v2, 0xcb400000, v2
	v_dual_mul_f32 v206, v212, v196 :: v_dual_add_f32 v5, 0xcb400000, v5
	v_dual_mul_f32 v207, v212, v197 :: v_dual_add_f32 v4, 0xcb400000, v4
	v_dual_mul_f32 v204, v212, v198 :: v_dual_add_f32 v7, 0xcb400000, v7
	v_dual_mul_f32 v205, v212, v199 :: v_dual_add_f32 v6, 0xcb400000, v6
	v_dual_fmac_f32 v128, v202, v0 :: v_dual_fmac_f32 v129, v203, v1
	v_dual_fmac_f32 v130, v200, v2 :: v_dual_fmac_f32 v131, v201, v3
	v_dual_fmac_f32 v132, v206, v4 :: v_dual_fmac_f32 v133, v207, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v134, v204, v6 :: v_dual_fmac_f32 v135, v205, v7
	v_dual_mul_f32 v202, v213, v192 :: v_dual_add_f32 v9, 0xcb400000, v9
	v_dual_mul_f32 v203, v213, v193 :: v_dual_add_f32 v8, 0xcb400000, v8
	v_dual_mul_f32 v200, v213, v194 :: v_dual_add_f32 v11, 0xcb400000, v11
	v_dual_mul_f32 v201, v213, v195 :: v_dual_add_f32 v10, 0xcb400000, v10
	v_dual_mul_f32 v206, v213, v196 :: v_dual_add_f32 v13, 0xcb400000, v13
	v_dual_mul_f32 v207, v213, v197 :: v_dual_add_f32 v12, 0xcb400000, v12
	v_dual_mul_f32 v204, v213, v198 :: v_dual_add_f32 v15, 0xcb400000, v15
	v_dual_mul_f32 v205, v213, v199 :: v_dual_add_f32 v14, 0xcb400000, v14
	v_dual_fmac_f32 v136, v202, v8 :: v_dual_fmac_f32 v137, v203, v9
	v_dual_fmac_f32 v138, v200, v10 :: v_dual_fmac_f32 v139, v201, v11
	v_dual_fmac_f32 v140, v206, v12 :: v_dual_fmac_f32 v141, v207, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v142, v204, v14 :: v_dual_fmac_f32 v143, v205, v15
	v_dual_mul_f32 v202, v214, v192 :: v_dual_add_f32 v17, 0xcb400000, v17
	v_dual_mul_f32 v203, v214, v193 :: v_dual_add_f32 v16, 0xcb400000, v16
	v_dual_mul_f32 v200, v214, v194 :: v_dual_add_f32 v19, 0xcb400000, v19
	v_dual_mul_f32 v201, v214, v195 :: v_dual_add_f32 v18, 0xcb400000, v18
	v_dual_mul_f32 v206, v214, v196 :: v_dual_add_f32 v21, 0xcb400000, v21
	v_dual_mul_f32 v207, v214, v197 :: v_dual_add_f32 v20, 0xcb400000, v20
	v_dual_mul_f32 v204, v214, v198 :: v_dual_add_f32 v23, 0xcb400000, v23
	v_dual_mul_f32 v205, v214, v199 :: v_dual_add_f32 v22, 0xcb400000, v22
	v_dual_fmac_f32 v144, v202, v16 :: v_dual_fmac_f32 v145, v203, v17
	v_dual_fmac_f32 v146, v200, v18 :: v_dual_fmac_f32 v147, v201, v19
	v_dual_fmac_f32 v148, v206, v20 :: v_dual_fmac_f32 v149, v207, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v150, v204, v22 :: v_dual_fmac_f32 v151, v205, v23
	v_dual_mul_f32 v202, v215, v192 :: v_dual_add_f32 v25, 0xcb400000, v25
	v_dual_mul_f32 v203, v215, v193 :: v_dual_add_f32 v24, 0xcb400000, v24
	v_dual_mul_f32 v200, v215, v194 :: v_dual_add_f32 v27, 0xcb400000, v27
	v_dual_mul_f32 v201, v215, v195 :: v_dual_add_f32 v26, 0xcb400000, v26
	v_dual_mul_f32 v206, v215, v196 :: v_dual_add_f32 v29, 0xcb400000, v29
	v_dual_mul_f32 v207, v215, v197 :: v_dual_add_f32 v28, 0xcb400000, v28
	v_dual_mul_f32 v204, v215, v198 :: v_dual_add_f32 v31, 0xcb400000, v31
	v_dual_mul_f32 v205, v215, v199 :: v_dual_add_f32 v30, 0xcb400000, v30
	v_dual_fmac_f32 v152, v202, v24 :: v_dual_fmac_f32 v153, v203, v25
	v_dual_fmac_f32 v154, v200, v26 :: v_dual_fmac_f32 v155, v201, v27
	v_dual_fmac_f32 v156, v206, v28 :: v_dual_fmac_f32 v157, v207, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v158, v204, v30 :: v_dual_fmac_f32 v159, v205, v31
	.Lv2b_silu_epilogue:
	v_mov_b32_e32 v0, v223
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s26, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s26, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s26, v2
	.Lv2b_silu_epi_body:
	v_mul_f32_e32 v160, 0xbfb8aa3b, v32
	v_mul_f32_e32 v166, 0xbfb8aa3b, v33
	v_mul_f32_e32 v172, 0xbfb8aa3b, v34
	v_mul_f32_e32 v178, 0xbfb8aa3b, v35
	v_mul_f32_e32 v184, 0xbfb8aa3b, v36
	v_mul_f32_e32 v190, 0xbfb8aa3b, v37
	v_mul_f32_e32 v196, 0xbfb8aa3b, v38
	v_mul_f32_e32 v202, 0xbfb8aa3b, v39
	v_fma_f32 v161, 0xbfb8aa3b, v32, -v160
	v_fma_f32 v167, 0xbfb8aa3b, v33, -v166
	v_fma_f32 v173, 0xbfb8aa3b, v34, -v172
	v_fma_f32 v179, 0xbfb8aa3b, v35, -v178
	v_fma_f32 v185, 0xbfb8aa3b, v36, -v184
	v_fma_f32 v191, 0xbfb8aa3b, v37, -v190
	v_fma_f32 v197, 0xbfb8aa3b, v38, -v196
	v_fma_f32 v203, 0xbfb8aa3b, v39, -v202
	v_fmac_f32_e32 v161, 0xb2a5705f, v32
	v_fmac_f32_e32 v167, 0xb2a5705f, v33
	v_fmac_f32_e32 v173, 0xb2a5705f, v34
	v_fmac_f32_e32 v179, 0xb2a5705f, v35
	v_fmac_f32_e32 v185, 0xb2a5705f, v36
	v_fmac_f32_e32 v191, 0xb2a5705f, v37
	v_fmac_f32_e32 v197, 0xb2a5705f, v38
	v_fmac_f32_e32 v203, 0xb2a5705f, v39
	v_rndne_f32_e32 v162, v160
	v_rndne_f32_e32 v168, v166
	v_rndne_f32_e32 v174, v172
	v_rndne_f32_e32 v180, v178
	v_rndne_f32_e32 v186, v184
	v_rndne_f32_e32 v192, v190
	v_rndne_f32_e32 v198, v196
	v_rndne_f32_e32 v204, v202
	v_sub_f32_e32 v160, v160, v162
	v_sub_f32_e32 v166, v166, v168
	v_sub_f32_e32 v172, v172, v174
	v_sub_f32_e32 v178, v178, v180
	v_sub_f32_e32 v184, v184, v186
	v_sub_f32_e32 v190, v190, v192
	v_sub_f32_e32 v196, v196, v198
	v_sub_f32_e32 v202, v202, v204
	v_add_f32_e32 v160, v160, v161
	v_add_f32_e32 v166, v166, v167
	v_add_f32_e32 v172, v172, v173
	v_add_f32_e32 v178, v178, v179
	v_add_f32_e32 v184, v184, v185
	v_add_f32_e32 v190, v190, v191
	v_add_f32_e32 v196, v196, v197
	v_add_f32_e32 v202, v202, v203
	v_exp_f32_e32 v160, v160
	v_exp_f32_e32 v166, v166
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v190, v190
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v202, v202
	v_cvt_i32_f32_e32 v162, v162
	v_cvt_i32_f32_e32 v168, v168
	v_cvt_i32_f32_e32 v174, v174
	v_cvt_i32_f32_e32 v180, v180
	v_cvt_i32_f32_e32 v186, v186
	v_cvt_i32_f32_e32 v192, v192
	v_cvt_i32_f32_e32 v198, v198
	v_cvt_i32_f32_e32 v204, v204
	v_ldexp_f32 v160, v160, v162
	v_ldexp_f32 v166, v166, v168
	v_ldexp_f32 v172, v172, v174
	v_ldexp_f32 v178, v178, v180
	v_ldexp_f32 v184, v184, v186
	v_ldexp_f32 v190, v190, v192
	v_ldexp_f32 v196, v196, v198
	v_ldexp_f32 v202, v202, v204
	v_cmp_nlt_f32_e64 s48, 0x42ce8ed0, v32
	v_cmp_nlt_f32_e64 s50, 0x42ce8ed0, v33
	v_cmp_nlt_f32_e64 s52, 0x42ce8ed0, v34
	v_cmp_nlt_f32_e64 s54, 0x42ce8ed0, v35
	v_cmp_nlt_f32_e64 s56, 0x42ce8ed0, v36
	v_cmp_nlt_f32_e64 s58, 0x42ce8ed0, v37
	v_cmp_nlt_f32_e64 s60, 0x42ce8ed0, v38
	v_cmp_nlt_f32_e64 s62, 0x42ce8ed0, v39
	v_cndmask_b32_e64 v160, 0, v160, s48
	v_cndmask_b32_e64 v166, 0, v166, s50
	v_cndmask_b32_e64 v172, 0, v172, s52
	v_cndmask_b32_e64 v178, 0, v178, s54
	v_cndmask_b32_e64 v184, 0, v184, s56
	v_cndmask_b32_e64 v190, 0, v190, s58
	v_cndmask_b32_e64 v196, 0, v196, s60
	v_cndmask_b32_e64 v202, 0, v202, s62
	v_cmp_ngt_f32_e64 s64, 0xc2b17218, v32
	v_cmp_ngt_f32_e64 s66, 0xc2b17218, v33
	v_cmp_ngt_f32_e64 s68, 0xc2b17218, v34
	v_cmp_ngt_f32_e64 s70, 0xc2b17218, v35
	v_cmp_ngt_f32_e64 s72, 0xc2b17218, v36
	v_cmp_ngt_f32_e64 s74, 0xc2b17218, v37
	v_cmp_ngt_f32_e64 s76, 0xc2b17218, v38
	v_cmp_ngt_f32_e64 s78, 0xc2b17218, v39
	v_cndmask_b32_e64 v160, 0x7f800000, v160, s64
	v_cndmask_b32_e64 v166, 0x7f800000, v166, s66
	v_cndmask_b32_e64 v172, 0x7f800000, v172, s68
	v_cndmask_b32_e64 v178, 0x7f800000, v178, s70
	v_cndmask_b32_e64 v184, 0x7f800000, v184, s72
	v_cndmask_b32_e64 v190, 0x7f800000, v190, s74
	v_cndmask_b32_e64 v196, 0x7f800000, v196, s76
	v_cndmask_b32_e64 v202, 0x7f800000, v202, s78
	v_add_f32_e32 v160, 1.0, v160
	v_add_f32_e32 v166, 1.0, v166
	v_add_f32_e32 v172, 1.0, v172
	v_add_f32_e32 v178, 1.0, v178
	v_add_f32_e32 v184, 1.0, v184
	v_add_f32_e32 v190, 1.0, v190
	v_add_f32_e32 v196, 1.0, v196
	v_add_f32_e32 v202, 1.0, v202
	v_div_scale_f32 v161, null, v160, v160, v32
	v_div_scale_f32 v167, null, v166, v166, v33
	v_div_scale_f32 v173, null, v172, v172, v34
	v_div_scale_f32 v179, null, v178, v178, v35
	v_div_scale_f32 v185, null, v184, v184, v36
	v_div_scale_f32 v191, null, v190, v190, v37
	v_div_scale_f32 v197, null, v196, v196, v38
	v_div_scale_f32 v203, null, v202, v202, v39
	v_div_scale_f32 v162, s80, v32, v160, v32
	v_div_scale_f32 v168, s82, v33, v166, v33
	v_div_scale_f32 v174, s84, v34, v172, v34
	v_div_scale_f32 v180, s86, v35, v178, v35
	v_div_scale_f32 v186, s88, v36, v184, v36
	v_div_scale_f32 v192, s90, v37, v190, v37
	v_div_scale_f32 v198, s92, v38, v196, v38
	v_div_scale_f32 v204, s94, v39, v202, v39
	v_rcp_f32_e32 v163, v161
	v_rcp_f32_e32 v169, v167
	v_rcp_f32_e32 v175, v173
	v_rcp_f32_e32 v181, v179
	v_rcp_f32_e32 v187, v185
	v_rcp_f32_e32 v193, v191
	v_rcp_f32_e32 v199, v197
	v_rcp_f32_e32 v205, v203
	v_fma_f32 v164, -v161, v163, 1.0
	v_fma_f32 v170, -v167, v169, 1.0
	v_fma_f32 v176, -v173, v175, 1.0
	v_fma_f32 v182, -v179, v181, 1.0
	v_fma_f32 v188, -v185, v187, 1.0
	v_fma_f32 v194, -v191, v193, 1.0
	v_fma_f32 v200, -v197, v199, 1.0
	v_fma_f32 v206, -v203, v205, 1.0
	v_fmac_f32_e32 v163, v164, v163
	v_fmac_f32_e32 v169, v170, v169
	v_fmac_f32_e32 v175, v176, v175
	v_fmac_f32_e32 v181, v182, v181
	v_fmac_f32_e32 v187, v188, v187
	v_fmac_f32_e32 v193, v194, v193
	v_fmac_f32_e32 v199, v200, v199
	v_fmac_f32_e32 v205, v206, v205
	v_mul_f32_e32 v164, v162, v163
	v_mul_f32_e32 v170, v168, v169
	v_mul_f32_e32 v176, v174, v175
	v_mul_f32_e32 v182, v180, v181
	v_mul_f32_e32 v188, v186, v187
	v_mul_f32_e32 v194, v192, v193
	v_mul_f32_e32 v200, v198, v199
	v_mul_f32_e32 v206, v204, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	v_fmac_f32_e32 v164, v165, v163
	v_fmac_f32_e32 v170, v171, v169
	v_fmac_f32_e32 v176, v177, v175
	v_fmac_f32_e32 v182, v183, v181
	v_fmac_f32_e32 v188, v189, v187
	v_fmac_f32_e32 v194, v195, v193
	v_fmac_f32_e32 v200, v201, v199
	v_fmac_f32_e32 v206, v207, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	s_mov_b32 vcc_lo, s80
	v_div_fmas_f32 v165, v165, v163, v164
	s_mov_b32 vcc_lo, s82
	v_div_fmas_f32 v171, v171, v169, v170
	s_mov_b32 vcc_lo, s84
	v_div_fmas_f32 v177, v177, v175, v176
	s_mov_b32 vcc_lo, s86
	v_div_fmas_f32 v183, v183, v181, v182
	s_mov_b32 vcc_lo, s88
	v_div_fmas_f32 v189, v189, v187, v188
	s_mov_b32 vcc_lo, s90
	v_div_fmas_f32 v195, v195, v193, v194
	s_mov_b32 vcc_lo, s92
	v_div_fmas_f32 v201, v201, v199, v200
	s_mov_b32 vcc_lo, s94
	v_div_fmas_f32 v207, v207, v205, v206
	v_div_fixup_f32 v165, v165, v160, v32
	v_div_fixup_f32 v171, v171, v166, v33
	v_div_fixup_f32 v177, v177, v172, v34
	v_div_fixup_f32 v183, v183, v178, v35
	v_div_fixup_f32 v189, v189, v184, v36
	v_div_fixup_f32 v195, v195, v190, v37
	v_div_fixup_f32 v201, v201, v196, v38
	v_div_fixup_f32 v207, v207, v202, v39
	v_mul_f32_e32 v32, v165, v64
	v_mul_f32_e32 v33, v171, v65
	v_mul_f32_e32 v34, v177, v66
	v_mul_f32_e32 v35, v183, v67
	v_mul_f32_e32 v36, v189, v68
	v_mul_f32_e32 v37, v195, v69
	v_mul_f32_e32 v38, v201, v70
	v_mul_f32_e32 v39, v207, v71
	global_store_b128 v0, v[32:35], s[36:37]
	global_store_b128 v0, v[36:39], s[36:37] offset:16
	v_mul_f32_e32 v160, 0xbfb8aa3b, v40
	v_mul_f32_e32 v166, 0xbfb8aa3b, v41
	v_mul_f32_e32 v172, 0xbfb8aa3b, v42
	v_mul_f32_e32 v178, 0xbfb8aa3b, v43
	v_mul_f32_e32 v184, 0xbfb8aa3b, v44
	v_mul_f32_e32 v190, 0xbfb8aa3b, v45
	v_mul_f32_e32 v196, 0xbfb8aa3b, v46
	v_mul_f32_e32 v202, 0xbfb8aa3b, v47
	v_fma_f32 v161, 0xbfb8aa3b, v40, -v160
	v_fma_f32 v167, 0xbfb8aa3b, v41, -v166
	v_fma_f32 v173, 0xbfb8aa3b, v42, -v172
	v_fma_f32 v179, 0xbfb8aa3b, v43, -v178
	v_fma_f32 v185, 0xbfb8aa3b, v44, -v184
	v_fma_f32 v191, 0xbfb8aa3b, v45, -v190
	v_fma_f32 v197, 0xbfb8aa3b, v46, -v196
	v_fma_f32 v203, 0xbfb8aa3b, v47, -v202
	v_fmac_f32_e32 v161, 0xb2a5705f, v40
	v_fmac_f32_e32 v167, 0xb2a5705f, v41
	v_fmac_f32_e32 v173, 0xb2a5705f, v42
	v_fmac_f32_e32 v179, 0xb2a5705f, v43
	v_fmac_f32_e32 v185, 0xb2a5705f, v44
	v_fmac_f32_e32 v191, 0xb2a5705f, v45
	v_fmac_f32_e32 v197, 0xb2a5705f, v46
	v_fmac_f32_e32 v203, 0xb2a5705f, v47
	v_rndne_f32_e32 v162, v160
	v_rndne_f32_e32 v168, v166
	v_rndne_f32_e32 v174, v172
	v_rndne_f32_e32 v180, v178
	v_rndne_f32_e32 v186, v184
	v_rndne_f32_e32 v192, v190
	v_rndne_f32_e32 v198, v196
	v_rndne_f32_e32 v204, v202
	v_sub_f32_e32 v160, v160, v162
	v_sub_f32_e32 v166, v166, v168
	v_sub_f32_e32 v172, v172, v174
	v_sub_f32_e32 v178, v178, v180
	v_sub_f32_e32 v184, v184, v186
	v_sub_f32_e32 v190, v190, v192
	v_sub_f32_e32 v196, v196, v198
	v_sub_f32_e32 v202, v202, v204
	v_add_f32_e32 v160, v160, v161
	v_add_f32_e32 v166, v166, v167
	v_add_f32_e32 v172, v172, v173
	v_add_f32_e32 v178, v178, v179
	v_add_f32_e32 v184, v184, v185
	v_add_f32_e32 v190, v190, v191
	v_add_f32_e32 v196, v196, v197
	v_add_f32_e32 v202, v202, v203
	v_exp_f32_e32 v160, v160
	v_exp_f32_e32 v166, v166
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v190, v190
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v202, v202
	v_cvt_i32_f32_e32 v162, v162
	v_cvt_i32_f32_e32 v168, v168
	v_cvt_i32_f32_e32 v174, v174
	v_cvt_i32_f32_e32 v180, v180
	v_cvt_i32_f32_e32 v186, v186
	v_cvt_i32_f32_e32 v192, v192
	v_cvt_i32_f32_e32 v198, v198
	v_cvt_i32_f32_e32 v204, v204
	v_ldexp_f32 v160, v160, v162
	v_ldexp_f32 v166, v166, v168
	v_ldexp_f32 v172, v172, v174
	v_ldexp_f32 v178, v178, v180
	v_ldexp_f32 v184, v184, v186
	v_ldexp_f32 v190, v190, v192
	v_ldexp_f32 v196, v196, v198
	v_ldexp_f32 v202, v202, v204
	v_cmp_nlt_f32_e64 s48, 0x42ce8ed0, v40
	v_cmp_nlt_f32_e64 s50, 0x42ce8ed0, v41
	v_cmp_nlt_f32_e64 s52, 0x42ce8ed0, v42
	v_cmp_nlt_f32_e64 s54, 0x42ce8ed0, v43
	v_cmp_nlt_f32_e64 s56, 0x42ce8ed0, v44
	v_cmp_nlt_f32_e64 s58, 0x42ce8ed0, v45
	v_cmp_nlt_f32_e64 s60, 0x42ce8ed0, v46
	v_cmp_nlt_f32_e64 s62, 0x42ce8ed0, v47
	v_cndmask_b32_e64 v160, 0, v160, s48
	v_cndmask_b32_e64 v166, 0, v166, s50
	v_cndmask_b32_e64 v172, 0, v172, s52
	v_cndmask_b32_e64 v178, 0, v178, s54
	v_cndmask_b32_e64 v184, 0, v184, s56
	v_cndmask_b32_e64 v190, 0, v190, s58
	v_cndmask_b32_e64 v196, 0, v196, s60
	v_cndmask_b32_e64 v202, 0, v202, s62
	v_cmp_ngt_f32_e64 s64, 0xc2b17218, v40
	v_cmp_ngt_f32_e64 s66, 0xc2b17218, v41
	v_cmp_ngt_f32_e64 s68, 0xc2b17218, v42
	v_cmp_ngt_f32_e64 s70, 0xc2b17218, v43
	v_cmp_ngt_f32_e64 s72, 0xc2b17218, v44
	v_cmp_ngt_f32_e64 s74, 0xc2b17218, v45
	v_cmp_ngt_f32_e64 s76, 0xc2b17218, v46
	v_cmp_ngt_f32_e64 s78, 0xc2b17218, v47
	v_cndmask_b32_e64 v160, 0x7f800000, v160, s64
	v_cndmask_b32_e64 v166, 0x7f800000, v166, s66
	v_cndmask_b32_e64 v172, 0x7f800000, v172, s68
	v_cndmask_b32_e64 v178, 0x7f800000, v178, s70
	v_cndmask_b32_e64 v184, 0x7f800000, v184, s72
	v_cndmask_b32_e64 v190, 0x7f800000, v190, s74
	v_cndmask_b32_e64 v196, 0x7f800000, v196, s76
	v_cndmask_b32_e64 v202, 0x7f800000, v202, s78
	v_add_f32_e32 v160, 1.0, v160
	v_add_f32_e32 v166, 1.0, v166
	v_add_f32_e32 v172, 1.0, v172
	v_add_f32_e32 v178, 1.0, v178
	v_add_f32_e32 v184, 1.0, v184
	v_add_f32_e32 v190, 1.0, v190
	v_add_f32_e32 v196, 1.0, v196
	v_add_f32_e32 v202, 1.0, v202
	v_div_scale_f32 v161, null, v160, v160, v40
	v_div_scale_f32 v167, null, v166, v166, v41
	v_div_scale_f32 v173, null, v172, v172, v42
	v_div_scale_f32 v179, null, v178, v178, v43
	v_div_scale_f32 v185, null, v184, v184, v44
	v_div_scale_f32 v191, null, v190, v190, v45
	v_div_scale_f32 v197, null, v196, v196, v46
	v_div_scale_f32 v203, null, v202, v202, v47
	v_div_scale_f32 v162, s80, v40, v160, v40
	v_div_scale_f32 v168, s82, v41, v166, v41
	v_div_scale_f32 v174, s84, v42, v172, v42
	v_div_scale_f32 v180, s86, v43, v178, v43
	v_div_scale_f32 v186, s88, v44, v184, v44
	v_div_scale_f32 v192, s90, v45, v190, v45
	v_div_scale_f32 v198, s92, v46, v196, v46
	v_div_scale_f32 v204, s94, v47, v202, v47
	v_rcp_f32_e32 v163, v161
	v_rcp_f32_e32 v169, v167
	v_rcp_f32_e32 v175, v173
	v_rcp_f32_e32 v181, v179
	v_rcp_f32_e32 v187, v185
	v_rcp_f32_e32 v193, v191
	v_rcp_f32_e32 v199, v197
	v_rcp_f32_e32 v205, v203
	v_fma_f32 v164, -v161, v163, 1.0
	v_fma_f32 v170, -v167, v169, 1.0
	v_fma_f32 v176, -v173, v175, 1.0
	v_fma_f32 v182, -v179, v181, 1.0
	v_fma_f32 v188, -v185, v187, 1.0
	v_fma_f32 v194, -v191, v193, 1.0
	v_fma_f32 v200, -v197, v199, 1.0
	v_fma_f32 v206, -v203, v205, 1.0
	v_fmac_f32_e32 v163, v164, v163
	v_fmac_f32_e32 v169, v170, v169
	v_fmac_f32_e32 v175, v176, v175
	v_fmac_f32_e32 v181, v182, v181
	v_fmac_f32_e32 v187, v188, v187
	v_fmac_f32_e32 v193, v194, v193
	v_fmac_f32_e32 v199, v200, v199
	v_fmac_f32_e32 v205, v206, v205
	v_mul_f32_e32 v164, v162, v163
	v_mul_f32_e32 v170, v168, v169
	v_mul_f32_e32 v176, v174, v175
	v_mul_f32_e32 v182, v180, v181
	v_mul_f32_e32 v188, v186, v187
	v_mul_f32_e32 v194, v192, v193
	v_mul_f32_e32 v200, v198, v199
	v_mul_f32_e32 v206, v204, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	v_fmac_f32_e32 v164, v165, v163
	v_fmac_f32_e32 v170, v171, v169
	v_fmac_f32_e32 v176, v177, v175
	v_fmac_f32_e32 v182, v183, v181
	v_fmac_f32_e32 v188, v189, v187
	v_fmac_f32_e32 v194, v195, v193
	v_fmac_f32_e32 v200, v201, v199
	v_fmac_f32_e32 v206, v207, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	s_mov_b32 vcc_lo, s80
	v_div_fmas_f32 v165, v165, v163, v164
	s_mov_b32 vcc_lo, s82
	v_div_fmas_f32 v171, v171, v169, v170
	s_mov_b32 vcc_lo, s84
	v_div_fmas_f32 v177, v177, v175, v176
	s_mov_b32 vcc_lo, s86
	v_div_fmas_f32 v183, v183, v181, v182
	s_mov_b32 vcc_lo, s88
	v_div_fmas_f32 v189, v189, v187, v188
	s_mov_b32 vcc_lo, s90
	v_div_fmas_f32 v195, v195, v193, v194
	s_mov_b32 vcc_lo, s92
	v_div_fmas_f32 v201, v201, v199, v200
	s_mov_b32 vcc_lo, s94
	v_div_fmas_f32 v207, v207, v205, v206
	v_div_fixup_f32 v165, v165, v160, v40
	v_div_fixup_f32 v171, v171, v166, v41
	v_div_fixup_f32 v177, v177, v172, v42
	v_div_fixup_f32 v183, v183, v178, v43
	v_div_fixup_f32 v189, v189, v184, v44
	v_div_fixup_f32 v195, v195, v190, v45
	v_div_fixup_f32 v201, v201, v196, v46
	v_div_fixup_f32 v207, v207, v202, v47
	v_mul_f32_e32 v40, v165, v72
	v_mul_f32_e32 v41, v171, v73
	v_mul_f32_e32 v42, v177, v74
	v_mul_f32_e32 v43, v183, v75
	v_mul_f32_e32 v44, v189, v76
	v_mul_f32_e32 v45, v195, v77
	v_mul_f32_e32 v46, v201, v78
	v_mul_f32_e32 v47, v207, v79
	global_store_b128 v1, v[40:43], s[36:37]
	global_store_b128 v1, v[44:47], s[36:37] offset:16
	v_mul_f32_e32 v160, 0xbfb8aa3b, v48
	v_mul_f32_e32 v166, 0xbfb8aa3b, v49
	v_mul_f32_e32 v172, 0xbfb8aa3b, v50
	v_mul_f32_e32 v178, 0xbfb8aa3b, v51
	v_mul_f32_e32 v184, 0xbfb8aa3b, v52
	v_mul_f32_e32 v190, 0xbfb8aa3b, v53
	v_mul_f32_e32 v196, 0xbfb8aa3b, v54
	v_mul_f32_e32 v202, 0xbfb8aa3b, v55
	v_fma_f32 v161, 0xbfb8aa3b, v48, -v160
	v_fma_f32 v167, 0xbfb8aa3b, v49, -v166
	v_fma_f32 v173, 0xbfb8aa3b, v50, -v172
	v_fma_f32 v179, 0xbfb8aa3b, v51, -v178
	v_fma_f32 v185, 0xbfb8aa3b, v52, -v184
	v_fma_f32 v191, 0xbfb8aa3b, v53, -v190
	v_fma_f32 v197, 0xbfb8aa3b, v54, -v196
	v_fma_f32 v203, 0xbfb8aa3b, v55, -v202
	v_fmac_f32_e32 v161, 0xb2a5705f, v48
	v_fmac_f32_e32 v167, 0xb2a5705f, v49
	v_fmac_f32_e32 v173, 0xb2a5705f, v50
	v_fmac_f32_e32 v179, 0xb2a5705f, v51
	v_fmac_f32_e32 v185, 0xb2a5705f, v52
	v_fmac_f32_e32 v191, 0xb2a5705f, v53
	v_fmac_f32_e32 v197, 0xb2a5705f, v54
	v_fmac_f32_e32 v203, 0xb2a5705f, v55
	v_rndne_f32_e32 v162, v160
	v_rndne_f32_e32 v168, v166
	v_rndne_f32_e32 v174, v172
	v_rndne_f32_e32 v180, v178
	v_rndne_f32_e32 v186, v184
	v_rndne_f32_e32 v192, v190
	v_rndne_f32_e32 v198, v196
	v_rndne_f32_e32 v204, v202
	v_sub_f32_e32 v160, v160, v162
	v_sub_f32_e32 v166, v166, v168
	v_sub_f32_e32 v172, v172, v174
	v_sub_f32_e32 v178, v178, v180
	v_sub_f32_e32 v184, v184, v186
	v_sub_f32_e32 v190, v190, v192
	v_sub_f32_e32 v196, v196, v198
	v_sub_f32_e32 v202, v202, v204
	v_add_f32_e32 v160, v160, v161
	v_add_f32_e32 v166, v166, v167
	v_add_f32_e32 v172, v172, v173
	v_add_f32_e32 v178, v178, v179
	v_add_f32_e32 v184, v184, v185
	v_add_f32_e32 v190, v190, v191
	v_add_f32_e32 v196, v196, v197
	v_add_f32_e32 v202, v202, v203
	v_exp_f32_e32 v160, v160
	v_exp_f32_e32 v166, v166
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v190, v190
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v202, v202
	v_cvt_i32_f32_e32 v162, v162
	v_cvt_i32_f32_e32 v168, v168
	v_cvt_i32_f32_e32 v174, v174
	v_cvt_i32_f32_e32 v180, v180
	v_cvt_i32_f32_e32 v186, v186
	v_cvt_i32_f32_e32 v192, v192
	v_cvt_i32_f32_e32 v198, v198
	v_cvt_i32_f32_e32 v204, v204
	v_ldexp_f32 v160, v160, v162
	v_ldexp_f32 v166, v166, v168
	v_ldexp_f32 v172, v172, v174
	v_ldexp_f32 v178, v178, v180
	v_ldexp_f32 v184, v184, v186
	v_ldexp_f32 v190, v190, v192
	v_ldexp_f32 v196, v196, v198
	v_ldexp_f32 v202, v202, v204
	v_cmp_nlt_f32_e64 s48, 0x42ce8ed0, v48
	v_cmp_nlt_f32_e64 s50, 0x42ce8ed0, v49
	v_cmp_nlt_f32_e64 s52, 0x42ce8ed0, v50
	v_cmp_nlt_f32_e64 s54, 0x42ce8ed0, v51
	v_cmp_nlt_f32_e64 s56, 0x42ce8ed0, v52
	v_cmp_nlt_f32_e64 s58, 0x42ce8ed0, v53
	v_cmp_nlt_f32_e64 s60, 0x42ce8ed0, v54
	v_cmp_nlt_f32_e64 s62, 0x42ce8ed0, v55
	v_cndmask_b32_e64 v160, 0, v160, s48
	v_cndmask_b32_e64 v166, 0, v166, s50
	v_cndmask_b32_e64 v172, 0, v172, s52
	v_cndmask_b32_e64 v178, 0, v178, s54
	v_cndmask_b32_e64 v184, 0, v184, s56
	v_cndmask_b32_e64 v190, 0, v190, s58
	v_cndmask_b32_e64 v196, 0, v196, s60
	v_cndmask_b32_e64 v202, 0, v202, s62
	v_cmp_ngt_f32_e64 s64, 0xc2b17218, v48
	v_cmp_ngt_f32_e64 s66, 0xc2b17218, v49
	v_cmp_ngt_f32_e64 s68, 0xc2b17218, v50
	v_cmp_ngt_f32_e64 s70, 0xc2b17218, v51
	v_cmp_ngt_f32_e64 s72, 0xc2b17218, v52
	v_cmp_ngt_f32_e64 s74, 0xc2b17218, v53
	v_cmp_ngt_f32_e64 s76, 0xc2b17218, v54
	v_cmp_ngt_f32_e64 s78, 0xc2b17218, v55
	v_cndmask_b32_e64 v160, 0x7f800000, v160, s64
	v_cndmask_b32_e64 v166, 0x7f800000, v166, s66
	v_cndmask_b32_e64 v172, 0x7f800000, v172, s68
	v_cndmask_b32_e64 v178, 0x7f800000, v178, s70
	v_cndmask_b32_e64 v184, 0x7f800000, v184, s72
	v_cndmask_b32_e64 v190, 0x7f800000, v190, s74
	v_cndmask_b32_e64 v196, 0x7f800000, v196, s76
	v_cndmask_b32_e64 v202, 0x7f800000, v202, s78
	v_add_f32_e32 v160, 1.0, v160
	v_add_f32_e32 v166, 1.0, v166
	v_add_f32_e32 v172, 1.0, v172
	v_add_f32_e32 v178, 1.0, v178
	v_add_f32_e32 v184, 1.0, v184
	v_add_f32_e32 v190, 1.0, v190
	v_add_f32_e32 v196, 1.0, v196
	v_add_f32_e32 v202, 1.0, v202
	v_div_scale_f32 v161, null, v160, v160, v48
	v_div_scale_f32 v167, null, v166, v166, v49
	v_div_scale_f32 v173, null, v172, v172, v50
	v_div_scale_f32 v179, null, v178, v178, v51
	v_div_scale_f32 v185, null, v184, v184, v52
	v_div_scale_f32 v191, null, v190, v190, v53
	v_div_scale_f32 v197, null, v196, v196, v54
	v_div_scale_f32 v203, null, v202, v202, v55
	v_div_scale_f32 v162, s80, v48, v160, v48
	v_div_scale_f32 v168, s82, v49, v166, v49
	v_div_scale_f32 v174, s84, v50, v172, v50
	v_div_scale_f32 v180, s86, v51, v178, v51
	v_div_scale_f32 v186, s88, v52, v184, v52
	v_div_scale_f32 v192, s90, v53, v190, v53
	v_div_scale_f32 v198, s92, v54, v196, v54
	v_div_scale_f32 v204, s94, v55, v202, v55
	v_rcp_f32_e32 v163, v161
	v_rcp_f32_e32 v169, v167
	v_rcp_f32_e32 v175, v173
	v_rcp_f32_e32 v181, v179
	v_rcp_f32_e32 v187, v185
	v_rcp_f32_e32 v193, v191
	v_rcp_f32_e32 v199, v197
	v_rcp_f32_e32 v205, v203
	v_fma_f32 v164, -v161, v163, 1.0
	v_fma_f32 v170, -v167, v169, 1.0
	v_fma_f32 v176, -v173, v175, 1.0
	v_fma_f32 v182, -v179, v181, 1.0
	v_fma_f32 v188, -v185, v187, 1.0
	v_fma_f32 v194, -v191, v193, 1.0
	v_fma_f32 v200, -v197, v199, 1.0
	v_fma_f32 v206, -v203, v205, 1.0
	v_fmac_f32_e32 v163, v164, v163
	v_fmac_f32_e32 v169, v170, v169
	v_fmac_f32_e32 v175, v176, v175
	v_fmac_f32_e32 v181, v182, v181
	v_fmac_f32_e32 v187, v188, v187
	v_fmac_f32_e32 v193, v194, v193
	v_fmac_f32_e32 v199, v200, v199
	v_fmac_f32_e32 v205, v206, v205
	v_mul_f32_e32 v164, v162, v163
	v_mul_f32_e32 v170, v168, v169
	v_mul_f32_e32 v176, v174, v175
	v_mul_f32_e32 v182, v180, v181
	v_mul_f32_e32 v188, v186, v187
	v_mul_f32_e32 v194, v192, v193
	v_mul_f32_e32 v200, v198, v199
	v_mul_f32_e32 v206, v204, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	v_fmac_f32_e32 v164, v165, v163
	v_fmac_f32_e32 v170, v171, v169
	v_fmac_f32_e32 v176, v177, v175
	v_fmac_f32_e32 v182, v183, v181
	v_fmac_f32_e32 v188, v189, v187
	v_fmac_f32_e32 v194, v195, v193
	v_fmac_f32_e32 v200, v201, v199
	v_fmac_f32_e32 v206, v207, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	s_mov_b32 vcc_lo, s80
	v_div_fmas_f32 v165, v165, v163, v164
	s_mov_b32 vcc_lo, s82
	v_div_fmas_f32 v171, v171, v169, v170
	s_mov_b32 vcc_lo, s84
	v_div_fmas_f32 v177, v177, v175, v176
	s_mov_b32 vcc_lo, s86
	v_div_fmas_f32 v183, v183, v181, v182
	s_mov_b32 vcc_lo, s88
	v_div_fmas_f32 v189, v189, v187, v188
	s_mov_b32 vcc_lo, s90
	v_div_fmas_f32 v195, v195, v193, v194
	s_mov_b32 vcc_lo, s92
	v_div_fmas_f32 v201, v201, v199, v200
	s_mov_b32 vcc_lo, s94
	v_div_fmas_f32 v207, v207, v205, v206
	v_div_fixup_f32 v165, v165, v160, v48
	v_div_fixup_f32 v171, v171, v166, v49
	v_div_fixup_f32 v177, v177, v172, v50
	v_div_fixup_f32 v183, v183, v178, v51
	v_div_fixup_f32 v189, v189, v184, v52
	v_div_fixup_f32 v195, v195, v190, v53
	v_div_fixup_f32 v201, v201, v196, v54
	v_div_fixup_f32 v207, v207, v202, v55
	v_mul_f32_e32 v48, v165, v80
	v_mul_f32_e32 v49, v171, v81
	v_mul_f32_e32 v50, v177, v82
	v_mul_f32_e32 v51, v183, v83
	v_mul_f32_e32 v52, v189, v84
	v_mul_f32_e32 v53, v195, v85
	v_mul_f32_e32 v54, v201, v86
	v_mul_f32_e32 v55, v207, v87
	global_store_b128 v2, v[48:51], s[36:37]
	global_store_b128 v2, v[52:55], s[36:37] offset:16
	v_mul_f32_e32 v160, 0xbfb8aa3b, v56
	v_mul_f32_e32 v166, 0xbfb8aa3b, v57
	v_mul_f32_e32 v172, 0xbfb8aa3b, v58
	v_mul_f32_e32 v178, 0xbfb8aa3b, v59
	v_mul_f32_e32 v184, 0xbfb8aa3b, v60
	v_mul_f32_e32 v190, 0xbfb8aa3b, v61
	v_mul_f32_e32 v196, 0xbfb8aa3b, v62
	v_mul_f32_e32 v202, 0xbfb8aa3b, v63
	v_fma_f32 v161, 0xbfb8aa3b, v56, -v160
	v_fma_f32 v167, 0xbfb8aa3b, v57, -v166
	v_fma_f32 v173, 0xbfb8aa3b, v58, -v172
	v_fma_f32 v179, 0xbfb8aa3b, v59, -v178
	v_fma_f32 v185, 0xbfb8aa3b, v60, -v184
	v_fma_f32 v191, 0xbfb8aa3b, v61, -v190
	v_fma_f32 v197, 0xbfb8aa3b, v62, -v196
	v_fma_f32 v203, 0xbfb8aa3b, v63, -v202
	v_fmac_f32_e32 v161, 0xb2a5705f, v56
	v_fmac_f32_e32 v167, 0xb2a5705f, v57
	v_fmac_f32_e32 v173, 0xb2a5705f, v58
	v_fmac_f32_e32 v179, 0xb2a5705f, v59
	v_fmac_f32_e32 v185, 0xb2a5705f, v60
	v_fmac_f32_e32 v191, 0xb2a5705f, v61
	v_fmac_f32_e32 v197, 0xb2a5705f, v62
	v_fmac_f32_e32 v203, 0xb2a5705f, v63
	v_rndne_f32_e32 v162, v160
	v_rndne_f32_e32 v168, v166
	v_rndne_f32_e32 v174, v172
	v_rndne_f32_e32 v180, v178
	v_rndne_f32_e32 v186, v184
	v_rndne_f32_e32 v192, v190
	v_rndne_f32_e32 v198, v196
	v_rndne_f32_e32 v204, v202
	v_sub_f32_e32 v160, v160, v162
	v_sub_f32_e32 v166, v166, v168
	v_sub_f32_e32 v172, v172, v174
	v_sub_f32_e32 v178, v178, v180
	v_sub_f32_e32 v184, v184, v186
	v_sub_f32_e32 v190, v190, v192
	v_sub_f32_e32 v196, v196, v198
	v_sub_f32_e32 v202, v202, v204
	v_add_f32_e32 v160, v160, v161
	v_add_f32_e32 v166, v166, v167
	v_add_f32_e32 v172, v172, v173
	v_add_f32_e32 v178, v178, v179
	v_add_f32_e32 v184, v184, v185
	v_add_f32_e32 v190, v190, v191
	v_add_f32_e32 v196, v196, v197
	v_add_f32_e32 v202, v202, v203
	v_exp_f32_e32 v160, v160
	v_exp_f32_e32 v166, v166
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v190, v190
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v202, v202
	v_cvt_i32_f32_e32 v162, v162
	v_cvt_i32_f32_e32 v168, v168
	v_cvt_i32_f32_e32 v174, v174
	v_cvt_i32_f32_e32 v180, v180
	v_cvt_i32_f32_e32 v186, v186
	v_cvt_i32_f32_e32 v192, v192
	v_cvt_i32_f32_e32 v198, v198
	v_cvt_i32_f32_e32 v204, v204
	v_ldexp_f32 v160, v160, v162
	v_ldexp_f32 v166, v166, v168
	v_ldexp_f32 v172, v172, v174
	v_ldexp_f32 v178, v178, v180
	v_ldexp_f32 v184, v184, v186
	v_ldexp_f32 v190, v190, v192
	v_ldexp_f32 v196, v196, v198
	v_ldexp_f32 v202, v202, v204
	v_cmp_nlt_f32_e64 s48, 0x42ce8ed0, v56
	v_cmp_nlt_f32_e64 s50, 0x42ce8ed0, v57
	v_cmp_nlt_f32_e64 s52, 0x42ce8ed0, v58
	v_cmp_nlt_f32_e64 s54, 0x42ce8ed0, v59
	v_cmp_nlt_f32_e64 s56, 0x42ce8ed0, v60
	v_cmp_nlt_f32_e64 s58, 0x42ce8ed0, v61
	v_cmp_nlt_f32_e64 s60, 0x42ce8ed0, v62
	v_cmp_nlt_f32_e64 s62, 0x42ce8ed0, v63
	v_cndmask_b32_e64 v160, 0, v160, s48
	v_cndmask_b32_e64 v166, 0, v166, s50
	v_cndmask_b32_e64 v172, 0, v172, s52
	v_cndmask_b32_e64 v178, 0, v178, s54
	v_cndmask_b32_e64 v184, 0, v184, s56
	v_cndmask_b32_e64 v190, 0, v190, s58
	v_cndmask_b32_e64 v196, 0, v196, s60
	v_cndmask_b32_e64 v202, 0, v202, s62
	v_cmp_ngt_f32_e64 s64, 0xc2b17218, v56
	v_cmp_ngt_f32_e64 s66, 0xc2b17218, v57
	v_cmp_ngt_f32_e64 s68, 0xc2b17218, v58
	v_cmp_ngt_f32_e64 s70, 0xc2b17218, v59
	v_cmp_ngt_f32_e64 s72, 0xc2b17218, v60
	v_cmp_ngt_f32_e64 s74, 0xc2b17218, v61
	v_cmp_ngt_f32_e64 s76, 0xc2b17218, v62
	v_cmp_ngt_f32_e64 s78, 0xc2b17218, v63
	v_cndmask_b32_e64 v160, 0x7f800000, v160, s64
	v_cndmask_b32_e64 v166, 0x7f800000, v166, s66
	v_cndmask_b32_e64 v172, 0x7f800000, v172, s68
	v_cndmask_b32_e64 v178, 0x7f800000, v178, s70
	v_cndmask_b32_e64 v184, 0x7f800000, v184, s72
	v_cndmask_b32_e64 v190, 0x7f800000, v190, s74
	v_cndmask_b32_e64 v196, 0x7f800000, v196, s76
	v_cndmask_b32_e64 v202, 0x7f800000, v202, s78
	v_add_f32_e32 v160, 1.0, v160
	v_add_f32_e32 v166, 1.0, v166
	v_add_f32_e32 v172, 1.0, v172
	v_add_f32_e32 v178, 1.0, v178
	v_add_f32_e32 v184, 1.0, v184
	v_add_f32_e32 v190, 1.0, v190
	v_add_f32_e32 v196, 1.0, v196
	v_add_f32_e32 v202, 1.0, v202
	v_div_scale_f32 v161, null, v160, v160, v56
	v_div_scale_f32 v167, null, v166, v166, v57
	v_div_scale_f32 v173, null, v172, v172, v58
	v_div_scale_f32 v179, null, v178, v178, v59
	v_div_scale_f32 v185, null, v184, v184, v60
	v_div_scale_f32 v191, null, v190, v190, v61
	v_div_scale_f32 v197, null, v196, v196, v62
	v_div_scale_f32 v203, null, v202, v202, v63
	v_div_scale_f32 v162, s80, v56, v160, v56
	v_div_scale_f32 v168, s82, v57, v166, v57
	v_div_scale_f32 v174, s84, v58, v172, v58
	v_div_scale_f32 v180, s86, v59, v178, v59
	v_div_scale_f32 v186, s88, v60, v184, v60
	v_div_scale_f32 v192, s90, v61, v190, v61
	v_div_scale_f32 v198, s92, v62, v196, v62
	v_div_scale_f32 v204, s94, v63, v202, v63
	v_rcp_f32_e32 v163, v161
	v_rcp_f32_e32 v169, v167
	v_rcp_f32_e32 v175, v173
	v_rcp_f32_e32 v181, v179
	v_rcp_f32_e32 v187, v185
	v_rcp_f32_e32 v193, v191
	v_rcp_f32_e32 v199, v197
	v_rcp_f32_e32 v205, v203
	v_fma_f32 v164, -v161, v163, 1.0
	v_fma_f32 v170, -v167, v169, 1.0
	v_fma_f32 v176, -v173, v175, 1.0
	v_fma_f32 v182, -v179, v181, 1.0
	v_fma_f32 v188, -v185, v187, 1.0
	v_fma_f32 v194, -v191, v193, 1.0
	v_fma_f32 v200, -v197, v199, 1.0
	v_fma_f32 v206, -v203, v205, 1.0
	v_fmac_f32_e32 v163, v164, v163
	v_fmac_f32_e32 v169, v170, v169
	v_fmac_f32_e32 v175, v176, v175
	v_fmac_f32_e32 v181, v182, v181
	v_fmac_f32_e32 v187, v188, v187
	v_fmac_f32_e32 v193, v194, v193
	v_fmac_f32_e32 v199, v200, v199
	v_fmac_f32_e32 v205, v206, v205
	v_mul_f32_e32 v164, v162, v163
	v_mul_f32_e32 v170, v168, v169
	v_mul_f32_e32 v176, v174, v175
	v_mul_f32_e32 v182, v180, v181
	v_mul_f32_e32 v188, v186, v187
	v_mul_f32_e32 v194, v192, v193
	v_mul_f32_e32 v200, v198, v199
	v_mul_f32_e32 v206, v204, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	v_fmac_f32_e32 v164, v165, v163
	v_fmac_f32_e32 v170, v171, v169
	v_fmac_f32_e32 v176, v177, v175
	v_fmac_f32_e32 v182, v183, v181
	v_fmac_f32_e32 v188, v189, v187
	v_fmac_f32_e32 v194, v195, v193
	v_fmac_f32_e32 v200, v201, v199
	v_fmac_f32_e32 v206, v207, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	s_mov_b32 vcc_lo, s80
	v_div_fmas_f32 v165, v165, v163, v164
	s_mov_b32 vcc_lo, s82
	v_div_fmas_f32 v171, v171, v169, v170
	s_mov_b32 vcc_lo, s84
	v_div_fmas_f32 v177, v177, v175, v176
	s_mov_b32 vcc_lo, s86
	v_div_fmas_f32 v183, v183, v181, v182
	s_mov_b32 vcc_lo, s88
	v_div_fmas_f32 v189, v189, v187, v188
	s_mov_b32 vcc_lo, s90
	v_div_fmas_f32 v195, v195, v193, v194
	s_mov_b32 vcc_lo, s92
	v_div_fmas_f32 v201, v201, v199, v200
	s_mov_b32 vcc_lo, s94
	v_div_fmas_f32 v207, v207, v205, v206
	v_div_fixup_f32 v165, v165, v160, v56
	v_div_fixup_f32 v171, v171, v166, v57
	v_div_fixup_f32 v177, v177, v172, v58
	v_div_fixup_f32 v183, v183, v178, v59
	v_div_fixup_f32 v189, v189, v184, v60
	v_div_fixup_f32 v195, v195, v190, v61
	v_div_fixup_f32 v201, v201, v196, v62
	v_div_fixup_f32 v207, v207, v202, v63
	v_mul_f32_e32 v56, v165, v88
	v_mul_f32_e32 v57, v171, v89
	v_mul_f32_e32 v58, v177, v90
	v_mul_f32_e32 v59, v183, v91
	v_mul_f32_e32 v60, v189, v92
	v_mul_f32_e32 v61, v195, v93
	v_mul_f32_e32 v62, v201, v94
	v_mul_f32_e32 v63, v207, v95
	global_store_b128 v3, v[56:59], s[36:37]
	global_store_b128 v3, v[60:63], s[36:37] offset:16
	v_mul_f32_e32 v160, 0xbfb8aa3b, v96
	v_mul_f32_e32 v166, 0xbfb8aa3b, v97
	v_mul_f32_e32 v172, 0xbfb8aa3b, v98
	v_mul_f32_e32 v178, 0xbfb8aa3b, v99
	v_mul_f32_e32 v184, 0xbfb8aa3b, v100
	v_mul_f32_e32 v190, 0xbfb8aa3b, v101
	v_mul_f32_e32 v196, 0xbfb8aa3b, v102
	v_mul_f32_e32 v202, 0xbfb8aa3b, v103
	v_fma_f32 v161, 0xbfb8aa3b, v96, -v160
	v_fma_f32 v167, 0xbfb8aa3b, v97, -v166
	v_fma_f32 v173, 0xbfb8aa3b, v98, -v172
	v_fma_f32 v179, 0xbfb8aa3b, v99, -v178
	v_fma_f32 v185, 0xbfb8aa3b, v100, -v184
	v_fma_f32 v191, 0xbfb8aa3b, v101, -v190
	v_fma_f32 v197, 0xbfb8aa3b, v102, -v196
	v_fma_f32 v203, 0xbfb8aa3b, v103, -v202
	v_fmac_f32_e32 v161, 0xb2a5705f, v96
	v_fmac_f32_e32 v167, 0xb2a5705f, v97
	v_fmac_f32_e32 v173, 0xb2a5705f, v98
	v_fmac_f32_e32 v179, 0xb2a5705f, v99
	v_fmac_f32_e32 v185, 0xb2a5705f, v100
	v_fmac_f32_e32 v191, 0xb2a5705f, v101
	v_fmac_f32_e32 v197, 0xb2a5705f, v102
	v_fmac_f32_e32 v203, 0xb2a5705f, v103
	v_rndne_f32_e32 v162, v160
	v_rndne_f32_e32 v168, v166
	v_rndne_f32_e32 v174, v172
	v_rndne_f32_e32 v180, v178
	v_rndne_f32_e32 v186, v184
	v_rndne_f32_e32 v192, v190
	v_rndne_f32_e32 v198, v196
	v_rndne_f32_e32 v204, v202
	v_sub_f32_e32 v160, v160, v162
	v_sub_f32_e32 v166, v166, v168
	v_sub_f32_e32 v172, v172, v174
	v_sub_f32_e32 v178, v178, v180
	v_sub_f32_e32 v184, v184, v186
	v_sub_f32_e32 v190, v190, v192
	v_sub_f32_e32 v196, v196, v198
	v_sub_f32_e32 v202, v202, v204
	v_add_f32_e32 v160, v160, v161
	v_add_f32_e32 v166, v166, v167
	v_add_f32_e32 v172, v172, v173
	v_add_f32_e32 v178, v178, v179
	v_add_f32_e32 v184, v184, v185
	v_add_f32_e32 v190, v190, v191
	v_add_f32_e32 v196, v196, v197
	v_add_f32_e32 v202, v202, v203
	v_exp_f32_e32 v160, v160
	v_exp_f32_e32 v166, v166
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v190, v190
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v202, v202
	v_cvt_i32_f32_e32 v162, v162
	v_cvt_i32_f32_e32 v168, v168
	v_cvt_i32_f32_e32 v174, v174
	v_cvt_i32_f32_e32 v180, v180
	v_cvt_i32_f32_e32 v186, v186
	v_cvt_i32_f32_e32 v192, v192
	v_cvt_i32_f32_e32 v198, v198
	v_cvt_i32_f32_e32 v204, v204
	v_ldexp_f32 v160, v160, v162
	v_ldexp_f32 v166, v166, v168
	v_ldexp_f32 v172, v172, v174
	v_ldexp_f32 v178, v178, v180
	v_ldexp_f32 v184, v184, v186
	v_ldexp_f32 v190, v190, v192
	v_ldexp_f32 v196, v196, v198
	v_ldexp_f32 v202, v202, v204
	v_cmp_nlt_f32_e64 s48, 0x42ce8ed0, v96
	v_cmp_nlt_f32_e64 s50, 0x42ce8ed0, v97
	v_cmp_nlt_f32_e64 s52, 0x42ce8ed0, v98
	v_cmp_nlt_f32_e64 s54, 0x42ce8ed0, v99
	v_cmp_nlt_f32_e64 s56, 0x42ce8ed0, v100
	v_cmp_nlt_f32_e64 s58, 0x42ce8ed0, v101
	v_cmp_nlt_f32_e64 s60, 0x42ce8ed0, v102
	v_cmp_nlt_f32_e64 s62, 0x42ce8ed0, v103
	v_cndmask_b32_e64 v160, 0, v160, s48
	v_cndmask_b32_e64 v166, 0, v166, s50
	v_cndmask_b32_e64 v172, 0, v172, s52
	v_cndmask_b32_e64 v178, 0, v178, s54
	v_cndmask_b32_e64 v184, 0, v184, s56
	v_cndmask_b32_e64 v190, 0, v190, s58
	v_cndmask_b32_e64 v196, 0, v196, s60
	v_cndmask_b32_e64 v202, 0, v202, s62
	v_cmp_ngt_f32_e64 s64, 0xc2b17218, v96
	v_cmp_ngt_f32_e64 s66, 0xc2b17218, v97
	v_cmp_ngt_f32_e64 s68, 0xc2b17218, v98
	v_cmp_ngt_f32_e64 s70, 0xc2b17218, v99
	v_cmp_ngt_f32_e64 s72, 0xc2b17218, v100
	v_cmp_ngt_f32_e64 s74, 0xc2b17218, v101
	v_cmp_ngt_f32_e64 s76, 0xc2b17218, v102
	v_cmp_ngt_f32_e64 s78, 0xc2b17218, v103
	v_cndmask_b32_e64 v160, 0x7f800000, v160, s64
	v_cndmask_b32_e64 v166, 0x7f800000, v166, s66
	v_cndmask_b32_e64 v172, 0x7f800000, v172, s68
	v_cndmask_b32_e64 v178, 0x7f800000, v178, s70
	v_cndmask_b32_e64 v184, 0x7f800000, v184, s72
	v_cndmask_b32_e64 v190, 0x7f800000, v190, s74
	v_cndmask_b32_e64 v196, 0x7f800000, v196, s76
	v_cndmask_b32_e64 v202, 0x7f800000, v202, s78
	v_add_f32_e32 v160, 1.0, v160
	v_add_f32_e32 v166, 1.0, v166
	v_add_f32_e32 v172, 1.0, v172
	v_add_f32_e32 v178, 1.0, v178
	v_add_f32_e32 v184, 1.0, v184
	v_add_f32_e32 v190, 1.0, v190
	v_add_f32_e32 v196, 1.0, v196
	v_add_f32_e32 v202, 1.0, v202
	v_div_scale_f32 v161, null, v160, v160, v96
	v_div_scale_f32 v167, null, v166, v166, v97
	v_div_scale_f32 v173, null, v172, v172, v98
	v_div_scale_f32 v179, null, v178, v178, v99
	v_div_scale_f32 v185, null, v184, v184, v100
	v_div_scale_f32 v191, null, v190, v190, v101
	v_div_scale_f32 v197, null, v196, v196, v102
	v_div_scale_f32 v203, null, v202, v202, v103
	v_div_scale_f32 v162, s80, v96, v160, v96
	v_div_scale_f32 v168, s82, v97, v166, v97
	v_div_scale_f32 v174, s84, v98, v172, v98
	v_div_scale_f32 v180, s86, v99, v178, v99
	v_div_scale_f32 v186, s88, v100, v184, v100
	v_div_scale_f32 v192, s90, v101, v190, v101
	v_div_scale_f32 v198, s92, v102, v196, v102
	v_div_scale_f32 v204, s94, v103, v202, v103
	v_rcp_f32_e32 v163, v161
	v_rcp_f32_e32 v169, v167
	v_rcp_f32_e32 v175, v173
	v_rcp_f32_e32 v181, v179
	v_rcp_f32_e32 v187, v185
	v_rcp_f32_e32 v193, v191
	v_rcp_f32_e32 v199, v197
	v_rcp_f32_e32 v205, v203
	v_fma_f32 v164, -v161, v163, 1.0
	v_fma_f32 v170, -v167, v169, 1.0
	v_fma_f32 v176, -v173, v175, 1.0
	v_fma_f32 v182, -v179, v181, 1.0
	v_fma_f32 v188, -v185, v187, 1.0
	v_fma_f32 v194, -v191, v193, 1.0
	v_fma_f32 v200, -v197, v199, 1.0
	v_fma_f32 v206, -v203, v205, 1.0
	v_fmac_f32_e32 v163, v164, v163
	v_fmac_f32_e32 v169, v170, v169
	v_fmac_f32_e32 v175, v176, v175
	v_fmac_f32_e32 v181, v182, v181
	v_fmac_f32_e32 v187, v188, v187
	v_fmac_f32_e32 v193, v194, v193
	v_fmac_f32_e32 v199, v200, v199
	v_fmac_f32_e32 v205, v206, v205
	v_mul_f32_e32 v164, v162, v163
	v_mul_f32_e32 v170, v168, v169
	v_mul_f32_e32 v176, v174, v175
	v_mul_f32_e32 v182, v180, v181
	v_mul_f32_e32 v188, v186, v187
	v_mul_f32_e32 v194, v192, v193
	v_mul_f32_e32 v200, v198, v199
	v_mul_f32_e32 v206, v204, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	v_fmac_f32_e32 v164, v165, v163
	v_fmac_f32_e32 v170, v171, v169
	v_fmac_f32_e32 v176, v177, v175
	v_fmac_f32_e32 v182, v183, v181
	v_fmac_f32_e32 v188, v189, v187
	v_fmac_f32_e32 v194, v195, v193
	v_fmac_f32_e32 v200, v201, v199
	v_fmac_f32_e32 v206, v207, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	s_mov_b32 vcc_lo, s80
	v_div_fmas_f32 v165, v165, v163, v164
	s_mov_b32 vcc_lo, s82
	v_div_fmas_f32 v171, v171, v169, v170
	s_mov_b32 vcc_lo, s84
	v_div_fmas_f32 v177, v177, v175, v176
	s_mov_b32 vcc_lo, s86
	v_div_fmas_f32 v183, v183, v181, v182
	s_mov_b32 vcc_lo, s88
	v_div_fmas_f32 v189, v189, v187, v188
	s_mov_b32 vcc_lo, s90
	v_div_fmas_f32 v195, v195, v193, v194
	s_mov_b32 vcc_lo, s92
	v_div_fmas_f32 v201, v201, v199, v200
	s_mov_b32 vcc_lo, s94
	v_div_fmas_f32 v207, v207, v205, v206
	v_div_fixup_f32 v165, v165, v160, v96
	v_div_fixup_f32 v171, v171, v166, v97
	v_div_fixup_f32 v177, v177, v172, v98
	v_div_fixup_f32 v183, v183, v178, v99
	v_div_fixup_f32 v189, v189, v184, v100
	v_div_fixup_f32 v195, v195, v190, v101
	v_div_fixup_f32 v201, v201, v196, v102
	v_div_fixup_f32 v207, v207, v202, v103
	v_mul_f32_e32 v96, v165, v128
	v_mul_f32_e32 v97, v171, v129
	v_mul_f32_e32 v98, v177, v130
	v_mul_f32_e32 v99, v183, v131
	v_mul_f32_e32 v100, v189, v132
	v_mul_f32_e32 v101, v195, v133
	v_mul_f32_e32 v102, v201, v134
	v_mul_f32_e32 v103, v207, v135
	global_store_b128 v0, v[96:99], s[36:37] offset:64
	global_store_b128 v0, v[100:103], s[36:37] offset:80
	v_mul_f32_e32 v160, 0xbfb8aa3b, v104
	v_mul_f32_e32 v166, 0xbfb8aa3b, v105
	v_mul_f32_e32 v172, 0xbfb8aa3b, v106
	v_mul_f32_e32 v178, 0xbfb8aa3b, v107
	v_mul_f32_e32 v184, 0xbfb8aa3b, v108
	v_mul_f32_e32 v190, 0xbfb8aa3b, v109
	v_mul_f32_e32 v196, 0xbfb8aa3b, v110
	v_mul_f32_e32 v202, 0xbfb8aa3b, v111
	v_fma_f32 v161, 0xbfb8aa3b, v104, -v160
	v_fma_f32 v167, 0xbfb8aa3b, v105, -v166
	v_fma_f32 v173, 0xbfb8aa3b, v106, -v172
	v_fma_f32 v179, 0xbfb8aa3b, v107, -v178
	v_fma_f32 v185, 0xbfb8aa3b, v108, -v184
	v_fma_f32 v191, 0xbfb8aa3b, v109, -v190
	v_fma_f32 v197, 0xbfb8aa3b, v110, -v196
	v_fma_f32 v203, 0xbfb8aa3b, v111, -v202
	v_fmac_f32_e32 v161, 0xb2a5705f, v104
	v_fmac_f32_e32 v167, 0xb2a5705f, v105
	v_fmac_f32_e32 v173, 0xb2a5705f, v106
	v_fmac_f32_e32 v179, 0xb2a5705f, v107
	v_fmac_f32_e32 v185, 0xb2a5705f, v108
	v_fmac_f32_e32 v191, 0xb2a5705f, v109
	v_fmac_f32_e32 v197, 0xb2a5705f, v110
	v_fmac_f32_e32 v203, 0xb2a5705f, v111
	v_rndne_f32_e32 v162, v160
	v_rndne_f32_e32 v168, v166
	v_rndne_f32_e32 v174, v172
	v_rndne_f32_e32 v180, v178
	v_rndne_f32_e32 v186, v184
	v_rndne_f32_e32 v192, v190
	v_rndne_f32_e32 v198, v196
	v_rndne_f32_e32 v204, v202
	v_sub_f32_e32 v160, v160, v162
	v_sub_f32_e32 v166, v166, v168
	v_sub_f32_e32 v172, v172, v174
	v_sub_f32_e32 v178, v178, v180
	v_sub_f32_e32 v184, v184, v186
	v_sub_f32_e32 v190, v190, v192
	v_sub_f32_e32 v196, v196, v198
	v_sub_f32_e32 v202, v202, v204
	v_add_f32_e32 v160, v160, v161
	v_add_f32_e32 v166, v166, v167
	v_add_f32_e32 v172, v172, v173
	v_add_f32_e32 v178, v178, v179
	v_add_f32_e32 v184, v184, v185
	v_add_f32_e32 v190, v190, v191
	v_add_f32_e32 v196, v196, v197
	v_add_f32_e32 v202, v202, v203
	v_exp_f32_e32 v160, v160
	v_exp_f32_e32 v166, v166
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v190, v190
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v202, v202
	v_cvt_i32_f32_e32 v162, v162
	v_cvt_i32_f32_e32 v168, v168
	v_cvt_i32_f32_e32 v174, v174
	v_cvt_i32_f32_e32 v180, v180
	v_cvt_i32_f32_e32 v186, v186
	v_cvt_i32_f32_e32 v192, v192
	v_cvt_i32_f32_e32 v198, v198
	v_cvt_i32_f32_e32 v204, v204
	v_ldexp_f32 v160, v160, v162
	v_ldexp_f32 v166, v166, v168
	v_ldexp_f32 v172, v172, v174
	v_ldexp_f32 v178, v178, v180
	v_ldexp_f32 v184, v184, v186
	v_ldexp_f32 v190, v190, v192
	v_ldexp_f32 v196, v196, v198
	v_ldexp_f32 v202, v202, v204
	v_cmp_nlt_f32_e64 s48, 0x42ce8ed0, v104
	v_cmp_nlt_f32_e64 s50, 0x42ce8ed0, v105
	v_cmp_nlt_f32_e64 s52, 0x42ce8ed0, v106
	v_cmp_nlt_f32_e64 s54, 0x42ce8ed0, v107
	v_cmp_nlt_f32_e64 s56, 0x42ce8ed0, v108
	v_cmp_nlt_f32_e64 s58, 0x42ce8ed0, v109
	v_cmp_nlt_f32_e64 s60, 0x42ce8ed0, v110
	v_cmp_nlt_f32_e64 s62, 0x42ce8ed0, v111
	v_cndmask_b32_e64 v160, 0, v160, s48
	v_cndmask_b32_e64 v166, 0, v166, s50
	v_cndmask_b32_e64 v172, 0, v172, s52
	v_cndmask_b32_e64 v178, 0, v178, s54
	v_cndmask_b32_e64 v184, 0, v184, s56
	v_cndmask_b32_e64 v190, 0, v190, s58
	v_cndmask_b32_e64 v196, 0, v196, s60
	v_cndmask_b32_e64 v202, 0, v202, s62
	v_cmp_ngt_f32_e64 s64, 0xc2b17218, v104
	v_cmp_ngt_f32_e64 s66, 0xc2b17218, v105
	v_cmp_ngt_f32_e64 s68, 0xc2b17218, v106
	v_cmp_ngt_f32_e64 s70, 0xc2b17218, v107
	v_cmp_ngt_f32_e64 s72, 0xc2b17218, v108
	v_cmp_ngt_f32_e64 s74, 0xc2b17218, v109
	v_cmp_ngt_f32_e64 s76, 0xc2b17218, v110
	v_cmp_ngt_f32_e64 s78, 0xc2b17218, v111
	v_cndmask_b32_e64 v160, 0x7f800000, v160, s64
	v_cndmask_b32_e64 v166, 0x7f800000, v166, s66
	v_cndmask_b32_e64 v172, 0x7f800000, v172, s68
	v_cndmask_b32_e64 v178, 0x7f800000, v178, s70
	v_cndmask_b32_e64 v184, 0x7f800000, v184, s72
	v_cndmask_b32_e64 v190, 0x7f800000, v190, s74
	v_cndmask_b32_e64 v196, 0x7f800000, v196, s76
	v_cndmask_b32_e64 v202, 0x7f800000, v202, s78
	v_add_f32_e32 v160, 1.0, v160
	v_add_f32_e32 v166, 1.0, v166
	v_add_f32_e32 v172, 1.0, v172
	v_add_f32_e32 v178, 1.0, v178
	v_add_f32_e32 v184, 1.0, v184
	v_add_f32_e32 v190, 1.0, v190
	v_add_f32_e32 v196, 1.0, v196
	v_add_f32_e32 v202, 1.0, v202
	v_div_scale_f32 v161, null, v160, v160, v104
	v_div_scale_f32 v167, null, v166, v166, v105
	v_div_scale_f32 v173, null, v172, v172, v106
	v_div_scale_f32 v179, null, v178, v178, v107
	v_div_scale_f32 v185, null, v184, v184, v108
	v_div_scale_f32 v191, null, v190, v190, v109
	v_div_scale_f32 v197, null, v196, v196, v110
	v_div_scale_f32 v203, null, v202, v202, v111
	v_div_scale_f32 v162, s80, v104, v160, v104
	v_div_scale_f32 v168, s82, v105, v166, v105
	v_div_scale_f32 v174, s84, v106, v172, v106
	v_div_scale_f32 v180, s86, v107, v178, v107
	v_div_scale_f32 v186, s88, v108, v184, v108
	v_div_scale_f32 v192, s90, v109, v190, v109
	v_div_scale_f32 v198, s92, v110, v196, v110
	v_div_scale_f32 v204, s94, v111, v202, v111
	v_rcp_f32_e32 v163, v161
	v_rcp_f32_e32 v169, v167
	v_rcp_f32_e32 v175, v173
	v_rcp_f32_e32 v181, v179
	v_rcp_f32_e32 v187, v185
	v_rcp_f32_e32 v193, v191
	v_rcp_f32_e32 v199, v197
	v_rcp_f32_e32 v205, v203
	v_fma_f32 v164, -v161, v163, 1.0
	v_fma_f32 v170, -v167, v169, 1.0
	v_fma_f32 v176, -v173, v175, 1.0
	v_fma_f32 v182, -v179, v181, 1.0
	v_fma_f32 v188, -v185, v187, 1.0
	v_fma_f32 v194, -v191, v193, 1.0
	v_fma_f32 v200, -v197, v199, 1.0
	v_fma_f32 v206, -v203, v205, 1.0
	v_fmac_f32_e32 v163, v164, v163
	v_fmac_f32_e32 v169, v170, v169
	v_fmac_f32_e32 v175, v176, v175
	v_fmac_f32_e32 v181, v182, v181
	v_fmac_f32_e32 v187, v188, v187
	v_fmac_f32_e32 v193, v194, v193
	v_fmac_f32_e32 v199, v200, v199
	v_fmac_f32_e32 v205, v206, v205
	v_mul_f32_e32 v164, v162, v163
	v_mul_f32_e32 v170, v168, v169
	v_mul_f32_e32 v176, v174, v175
	v_mul_f32_e32 v182, v180, v181
	v_mul_f32_e32 v188, v186, v187
	v_mul_f32_e32 v194, v192, v193
	v_mul_f32_e32 v200, v198, v199
	v_mul_f32_e32 v206, v204, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	v_fmac_f32_e32 v164, v165, v163
	v_fmac_f32_e32 v170, v171, v169
	v_fmac_f32_e32 v176, v177, v175
	v_fmac_f32_e32 v182, v183, v181
	v_fmac_f32_e32 v188, v189, v187
	v_fmac_f32_e32 v194, v195, v193
	v_fmac_f32_e32 v200, v201, v199
	v_fmac_f32_e32 v206, v207, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	s_mov_b32 vcc_lo, s80
	v_div_fmas_f32 v165, v165, v163, v164
	s_mov_b32 vcc_lo, s82
	v_div_fmas_f32 v171, v171, v169, v170
	s_mov_b32 vcc_lo, s84
	v_div_fmas_f32 v177, v177, v175, v176
	s_mov_b32 vcc_lo, s86
	v_div_fmas_f32 v183, v183, v181, v182
	s_mov_b32 vcc_lo, s88
	v_div_fmas_f32 v189, v189, v187, v188
	s_mov_b32 vcc_lo, s90
	v_div_fmas_f32 v195, v195, v193, v194
	s_mov_b32 vcc_lo, s92
	v_div_fmas_f32 v201, v201, v199, v200
	s_mov_b32 vcc_lo, s94
	v_div_fmas_f32 v207, v207, v205, v206
	v_div_fixup_f32 v165, v165, v160, v104
	v_div_fixup_f32 v171, v171, v166, v105
	v_div_fixup_f32 v177, v177, v172, v106
	v_div_fixup_f32 v183, v183, v178, v107
	v_div_fixup_f32 v189, v189, v184, v108
	v_div_fixup_f32 v195, v195, v190, v109
	v_div_fixup_f32 v201, v201, v196, v110
	v_div_fixup_f32 v207, v207, v202, v111
	v_mul_f32_e32 v104, v165, v136
	v_mul_f32_e32 v105, v171, v137
	v_mul_f32_e32 v106, v177, v138
	v_mul_f32_e32 v107, v183, v139
	v_mul_f32_e32 v108, v189, v140
	v_mul_f32_e32 v109, v195, v141
	v_mul_f32_e32 v110, v201, v142
	v_mul_f32_e32 v111, v207, v143
	global_store_b128 v1, v[104:107], s[36:37] offset:64
	global_store_b128 v1, v[108:111], s[36:37] offset:80
	v_mul_f32_e32 v160, 0xbfb8aa3b, v112
	v_mul_f32_e32 v166, 0xbfb8aa3b, v113
	v_mul_f32_e32 v172, 0xbfb8aa3b, v114
	v_mul_f32_e32 v178, 0xbfb8aa3b, v115
	v_mul_f32_e32 v184, 0xbfb8aa3b, v116
	v_mul_f32_e32 v190, 0xbfb8aa3b, v117
	v_mul_f32_e32 v196, 0xbfb8aa3b, v118
	v_mul_f32_e32 v202, 0xbfb8aa3b, v119
	v_fma_f32 v161, 0xbfb8aa3b, v112, -v160
	v_fma_f32 v167, 0xbfb8aa3b, v113, -v166
	v_fma_f32 v173, 0xbfb8aa3b, v114, -v172
	v_fma_f32 v179, 0xbfb8aa3b, v115, -v178
	v_fma_f32 v185, 0xbfb8aa3b, v116, -v184
	v_fma_f32 v191, 0xbfb8aa3b, v117, -v190
	v_fma_f32 v197, 0xbfb8aa3b, v118, -v196
	v_fma_f32 v203, 0xbfb8aa3b, v119, -v202
	v_fmac_f32_e32 v161, 0xb2a5705f, v112
	v_fmac_f32_e32 v167, 0xb2a5705f, v113
	v_fmac_f32_e32 v173, 0xb2a5705f, v114
	v_fmac_f32_e32 v179, 0xb2a5705f, v115
	v_fmac_f32_e32 v185, 0xb2a5705f, v116
	v_fmac_f32_e32 v191, 0xb2a5705f, v117
	v_fmac_f32_e32 v197, 0xb2a5705f, v118
	v_fmac_f32_e32 v203, 0xb2a5705f, v119
	v_rndne_f32_e32 v162, v160
	v_rndne_f32_e32 v168, v166
	v_rndne_f32_e32 v174, v172
	v_rndne_f32_e32 v180, v178
	v_rndne_f32_e32 v186, v184
	v_rndne_f32_e32 v192, v190
	v_rndne_f32_e32 v198, v196
	v_rndne_f32_e32 v204, v202
	v_sub_f32_e32 v160, v160, v162
	v_sub_f32_e32 v166, v166, v168
	v_sub_f32_e32 v172, v172, v174
	v_sub_f32_e32 v178, v178, v180
	v_sub_f32_e32 v184, v184, v186
	v_sub_f32_e32 v190, v190, v192
	v_sub_f32_e32 v196, v196, v198
	v_sub_f32_e32 v202, v202, v204
	v_add_f32_e32 v160, v160, v161
	v_add_f32_e32 v166, v166, v167
	v_add_f32_e32 v172, v172, v173
	v_add_f32_e32 v178, v178, v179
	v_add_f32_e32 v184, v184, v185
	v_add_f32_e32 v190, v190, v191
	v_add_f32_e32 v196, v196, v197
	v_add_f32_e32 v202, v202, v203
	v_exp_f32_e32 v160, v160
	v_exp_f32_e32 v166, v166
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v190, v190
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v202, v202
	v_cvt_i32_f32_e32 v162, v162
	v_cvt_i32_f32_e32 v168, v168
	v_cvt_i32_f32_e32 v174, v174
	v_cvt_i32_f32_e32 v180, v180
	v_cvt_i32_f32_e32 v186, v186
	v_cvt_i32_f32_e32 v192, v192
	v_cvt_i32_f32_e32 v198, v198
	v_cvt_i32_f32_e32 v204, v204
	v_ldexp_f32 v160, v160, v162
	v_ldexp_f32 v166, v166, v168
	v_ldexp_f32 v172, v172, v174
	v_ldexp_f32 v178, v178, v180
	v_ldexp_f32 v184, v184, v186
	v_ldexp_f32 v190, v190, v192
	v_ldexp_f32 v196, v196, v198
	v_ldexp_f32 v202, v202, v204
	v_cmp_nlt_f32_e64 s48, 0x42ce8ed0, v112
	v_cmp_nlt_f32_e64 s50, 0x42ce8ed0, v113
	v_cmp_nlt_f32_e64 s52, 0x42ce8ed0, v114
	v_cmp_nlt_f32_e64 s54, 0x42ce8ed0, v115
	v_cmp_nlt_f32_e64 s56, 0x42ce8ed0, v116
	v_cmp_nlt_f32_e64 s58, 0x42ce8ed0, v117
	v_cmp_nlt_f32_e64 s60, 0x42ce8ed0, v118
	v_cmp_nlt_f32_e64 s62, 0x42ce8ed0, v119
	v_cndmask_b32_e64 v160, 0, v160, s48
	v_cndmask_b32_e64 v166, 0, v166, s50
	v_cndmask_b32_e64 v172, 0, v172, s52
	v_cndmask_b32_e64 v178, 0, v178, s54
	v_cndmask_b32_e64 v184, 0, v184, s56
	v_cndmask_b32_e64 v190, 0, v190, s58
	v_cndmask_b32_e64 v196, 0, v196, s60
	v_cndmask_b32_e64 v202, 0, v202, s62
	v_cmp_ngt_f32_e64 s64, 0xc2b17218, v112
	v_cmp_ngt_f32_e64 s66, 0xc2b17218, v113
	v_cmp_ngt_f32_e64 s68, 0xc2b17218, v114
	v_cmp_ngt_f32_e64 s70, 0xc2b17218, v115
	v_cmp_ngt_f32_e64 s72, 0xc2b17218, v116
	v_cmp_ngt_f32_e64 s74, 0xc2b17218, v117
	v_cmp_ngt_f32_e64 s76, 0xc2b17218, v118
	v_cmp_ngt_f32_e64 s78, 0xc2b17218, v119
	v_cndmask_b32_e64 v160, 0x7f800000, v160, s64
	v_cndmask_b32_e64 v166, 0x7f800000, v166, s66
	v_cndmask_b32_e64 v172, 0x7f800000, v172, s68
	v_cndmask_b32_e64 v178, 0x7f800000, v178, s70
	v_cndmask_b32_e64 v184, 0x7f800000, v184, s72
	v_cndmask_b32_e64 v190, 0x7f800000, v190, s74
	v_cndmask_b32_e64 v196, 0x7f800000, v196, s76
	v_cndmask_b32_e64 v202, 0x7f800000, v202, s78
	v_add_f32_e32 v160, 1.0, v160
	v_add_f32_e32 v166, 1.0, v166
	v_add_f32_e32 v172, 1.0, v172
	v_add_f32_e32 v178, 1.0, v178
	v_add_f32_e32 v184, 1.0, v184
	v_add_f32_e32 v190, 1.0, v190
	v_add_f32_e32 v196, 1.0, v196
	v_add_f32_e32 v202, 1.0, v202
	v_div_scale_f32 v161, null, v160, v160, v112
	v_div_scale_f32 v167, null, v166, v166, v113
	v_div_scale_f32 v173, null, v172, v172, v114
	v_div_scale_f32 v179, null, v178, v178, v115
	v_div_scale_f32 v185, null, v184, v184, v116
	v_div_scale_f32 v191, null, v190, v190, v117
	v_div_scale_f32 v197, null, v196, v196, v118
	v_div_scale_f32 v203, null, v202, v202, v119
	v_div_scale_f32 v162, s80, v112, v160, v112
	v_div_scale_f32 v168, s82, v113, v166, v113
	v_div_scale_f32 v174, s84, v114, v172, v114
	v_div_scale_f32 v180, s86, v115, v178, v115
	v_div_scale_f32 v186, s88, v116, v184, v116
	v_div_scale_f32 v192, s90, v117, v190, v117
	v_div_scale_f32 v198, s92, v118, v196, v118
	v_div_scale_f32 v204, s94, v119, v202, v119
	v_rcp_f32_e32 v163, v161
	v_rcp_f32_e32 v169, v167
	v_rcp_f32_e32 v175, v173
	v_rcp_f32_e32 v181, v179
	v_rcp_f32_e32 v187, v185
	v_rcp_f32_e32 v193, v191
	v_rcp_f32_e32 v199, v197
	v_rcp_f32_e32 v205, v203
	v_fma_f32 v164, -v161, v163, 1.0
	v_fma_f32 v170, -v167, v169, 1.0
	v_fma_f32 v176, -v173, v175, 1.0
	v_fma_f32 v182, -v179, v181, 1.0
	v_fma_f32 v188, -v185, v187, 1.0
	v_fma_f32 v194, -v191, v193, 1.0
	v_fma_f32 v200, -v197, v199, 1.0
	v_fma_f32 v206, -v203, v205, 1.0
	v_fmac_f32_e32 v163, v164, v163
	v_fmac_f32_e32 v169, v170, v169
	v_fmac_f32_e32 v175, v176, v175
	v_fmac_f32_e32 v181, v182, v181
	v_fmac_f32_e32 v187, v188, v187
	v_fmac_f32_e32 v193, v194, v193
	v_fmac_f32_e32 v199, v200, v199
	v_fmac_f32_e32 v205, v206, v205
	v_mul_f32_e32 v164, v162, v163
	v_mul_f32_e32 v170, v168, v169
	v_mul_f32_e32 v176, v174, v175
	v_mul_f32_e32 v182, v180, v181
	v_mul_f32_e32 v188, v186, v187
	v_mul_f32_e32 v194, v192, v193
	v_mul_f32_e32 v200, v198, v199
	v_mul_f32_e32 v206, v204, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	v_fmac_f32_e32 v164, v165, v163
	v_fmac_f32_e32 v170, v171, v169
	v_fmac_f32_e32 v176, v177, v175
	v_fmac_f32_e32 v182, v183, v181
	v_fmac_f32_e32 v188, v189, v187
	v_fmac_f32_e32 v194, v195, v193
	v_fmac_f32_e32 v200, v201, v199
	v_fmac_f32_e32 v206, v207, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	s_mov_b32 vcc_lo, s80
	v_div_fmas_f32 v165, v165, v163, v164
	s_mov_b32 vcc_lo, s82
	v_div_fmas_f32 v171, v171, v169, v170
	s_mov_b32 vcc_lo, s84
	v_div_fmas_f32 v177, v177, v175, v176
	s_mov_b32 vcc_lo, s86
	v_div_fmas_f32 v183, v183, v181, v182
	s_mov_b32 vcc_lo, s88
	v_div_fmas_f32 v189, v189, v187, v188
	s_mov_b32 vcc_lo, s90
	v_div_fmas_f32 v195, v195, v193, v194
	s_mov_b32 vcc_lo, s92
	v_div_fmas_f32 v201, v201, v199, v200
	s_mov_b32 vcc_lo, s94
	v_div_fmas_f32 v207, v207, v205, v206
	v_div_fixup_f32 v165, v165, v160, v112
	v_div_fixup_f32 v171, v171, v166, v113
	v_div_fixup_f32 v177, v177, v172, v114
	v_div_fixup_f32 v183, v183, v178, v115
	v_div_fixup_f32 v189, v189, v184, v116
	v_div_fixup_f32 v195, v195, v190, v117
	v_div_fixup_f32 v201, v201, v196, v118
	v_div_fixup_f32 v207, v207, v202, v119
	v_mul_f32_e32 v112, v165, v144
	v_mul_f32_e32 v113, v171, v145
	v_mul_f32_e32 v114, v177, v146
	v_mul_f32_e32 v115, v183, v147
	v_mul_f32_e32 v116, v189, v148
	v_mul_f32_e32 v117, v195, v149
	v_mul_f32_e32 v118, v201, v150
	v_mul_f32_e32 v119, v207, v151
	global_store_b128 v2, v[112:115], s[36:37] offset:64
	global_store_b128 v2, v[116:119], s[36:37] offset:80
	v_mul_f32_e32 v160, 0xbfb8aa3b, v120
	v_mul_f32_e32 v166, 0xbfb8aa3b, v121
	v_mul_f32_e32 v172, 0xbfb8aa3b, v122
	v_mul_f32_e32 v178, 0xbfb8aa3b, v123
	v_mul_f32_e32 v184, 0xbfb8aa3b, v124
	v_mul_f32_e32 v190, 0xbfb8aa3b, v125
	v_mul_f32_e32 v196, 0xbfb8aa3b, v126
	v_mul_f32_e32 v202, 0xbfb8aa3b, v127
	v_fma_f32 v161, 0xbfb8aa3b, v120, -v160
	v_fma_f32 v167, 0xbfb8aa3b, v121, -v166
	v_fma_f32 v173, 0xbfb8aa3b, v122, -v172
	v_fma_f32 v179, 0xbfb8aa3b, v123, -v178
	v_fma_f32 v185, 0xbfb8aa3b, v124, -v184
	v_fma_f32 v191, 0xbfb8aa3b, v125, -v190
	v_fma_f32 v197, 0xbfb8aa3b, v126, -v196
	v_fma_f32 v203, 0xbfb8aa3b, v127, -v202
	v_fmac_f32_e32 v161, 0xb2a5705f, v120
	v_fmac_f32_e32 v167, 0xb2a5705f, v121
	v_fmac_f32_e32 v173, 0xb2a5705f, v122
	v_fmac_f32_e32 v179, 0xb2a5705f, v123
	v_fmac_f32_e32 v185, 0xb2a5705f, v124
	v_fmac_f32_e32 v191, 0xb2a5705f, v125
	v_fmac_f32_e32 v197, 0xb2a5705f, v126
	v_fmac_f32_e32 v203, 0xb2a5705f, v127
	v_rndne_f32_e32 v162, v160
	v_rndne_f32_e32 v168, v166
	v_rndne_f32_e32 v174, v172
	v_rndne_f32_e32 v180, v178
	v_rndne_f32_e32 v186, v184
	v_rndne_f32_e32 v192, v190
	v_rndne_f32_e32 v198, v196
	v_rndne_f32_e32 v204, v202
	v_sub_f32_e32 v160, v160, v162
	v_sub_f32_e32 v166, v166, v168
	v_sub_f32_e32 v172, v172, v174
	v_sub_f32_e32 v178, v178, v180
	v_sub_f32_e32 v184, v184, v186
	v_sub_f32_e32 v190, v190, v192
	v_sub_f32_e32 v196, v196, v198
	v_sub_f32_e32 v202, v202, v204
	v_add_f32_e32 v160, v160, v161
	v_add_f32_e32 v166, v166, v167
	v_add_f32_e32 v172, v172, v173
	v_add_f32_e32 v178, v178, v179
	v_add_f32_e32 v184, v184, v185
	v_add_f32_e32 v190, v190, v191
	v_add_f32_e32 v196, v196, v197
	v_add_f32_e32 v202, v202, v203
	v_exp_f32_e32 v160, v160
	v_exp_f32_e32 v166, v166
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v190, v190
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v202, v202
	v_cvt_i32_f32_e32 v162, v162
	v_cvt_i32_f32_e32 v168, v168
	v_cvt_i32_f32_e32 v174, v174
	v_cvt_i32_f32_e32 v180, v180
	v_cvt_i32_f32_e32 v186, v186
	v_cvt_i32_f32_e32 v192, v192
	v_cvt_i32_f32_e32 v198, v198
	v_cvt_i32_f32_e32 v204, v204
	v_ldexp_f32 v160, v160, v162
	v_ldexp_f32 v166, v166, v168
	v_ldexp_f32 v172, v172, v174
	v_ldexp_f32 v178, v178, v180
	v_ldexp_f32 v184, v184, v186
	v_ldexp_f32 v190, v190, v192
	v_ldexp_f32 v196, v196, v198
	v_ldexp_f32 v202, v202, v204
	v_cmp_nlt_f32_e64 s48, 0x42ce8ed0, v120
	v_cmp_nlt_f32_e64 s50, 0x42ce8ed0, v121
	v_cmp_nlt_f32_e64 s52, 0x42ce8ed0, v122
	v_cmp_nlt_f32_e64 s54, 0x42ce8ed0, v123
	v_cmp_nlt_f32_e64 s56, 0x42ce8ed0, v124
	v_cmp_nlt_f32_e64 s58, 0x42ce8ed0, v125
	v_cmp_nlt_f32_e64 s60, 0x42ce8ed0, v126
	v_cmp_nlt_f32_e64 s62, 0x42ce8ed0, v127
	v_cndmask_b32_e64 v160, 0, v160, s48
	v_cndmask_b32_e64 v166, 0, v166, s50
	v_cndmask_b32_e64 v172, 0, v172, s52
	v_cndmask_b32_e64 v178, 0, v178, s54
	v_cndmask_b32_e64 v184, 0, v184, s56
	v_cndmask_b32_e64 v190, 0, v190, s58
	v_cndmask_b32_e64 v196, 0, v196, s60
	v_cndmask_b32_e64 v202, 0, v202, s62
	v_cmp_ngt_f32_e64 s64, 0xc2b17218, v120
	v_cmp_ngt_f32_e64 s66, 0xc2b17218, v121
	v_cmp_ngt_f32_e64 s68, 0xc2b17218, v122
	v_cmp_ngt_f32_e64 s70, 0xc2b17218, v123
	v_cmp_ngt_f32_e64 s72, 0xc2b17218, v124
	v_cmp_ngt_f32_e64 s74, 0xc2b17218, v125
	v_cmp_ngt_f32_e64 s76, 0xc2b17218, v126
	v_cmp_ngt_f32_e64 s78, 0xc2b17218, v127
	v_cndmask_b32_e64 v160, 0x7f800000, v160, s64
	v_cndmask_b32_e64 v166, 0x7f800000, v166, s66
	v_cndmask_b32_e64 v172, 0x7f800000, v172, s68
	v_cndmask_b32_e64 v178, 0x7f800000, v178, s70
	v_cndmask_b32_e64 v184, 0x7f800000, v184, s72
	v_cndmask_b32_e64 v190, 0x7f800000, v190, s74
	v_cndmask_b32_e64 v196, 0x7f800000, v196, s76
	v_cndmask_b32_e64 v202, 0x7f800000, v202, s78
	v_add_f32_e32 v160, 1.0, v160
	v_add_f32_e32 v166, 1.0, v166
	v_add_f32_e32 v172, 1.0, v172
	v_add_f32_e32 v178, 1.0, v178
	v_add_f32_e32 v184, 1.0, v184
	v_add_f32_e32 v190, 1.0, v190
	v_add_f32_e32 v196, 1.0, v196
	v_add_f32_e32 v202, 1.0, v202
	v_div_scale_f32 v161, null, v160, v160, v120
	v_div_scale_f32 v167, null, v166, v166, v121
	v_div_scale_f32 v173, null, v172, v172, v122
	v_div_scale_f32 v179, null, v178, v178, v123
	v_div_scale_f32 v185, null, v184, v184, v124
	v_div_scale_f32 v191, null, v190, v190, v125
	v_div_scale_f32 v197, null, v196, v196, v126
	v_div_scale_f32 v203, null, v202, v202, v127
	v_div_scale_f32 v162, s80, v120, v160, v120
	v_div_scale_f32 v168, s82, v121, v166, v121
	v_div_scale_f32 v174, s84, v122, v172, v122
	v_div_scale_f32 v180, s86, v123, v178, v123
	v_div_scale_f32 v186, s88, v124, v184, v124
	v_div_scale_f32 v192, s90, v125, v190, v125
	v_div_scale_f32 v198, s92, v126, v196, v126
	v_div_scale_f32 v204, s94, v127, v202, v127
	v_rcp_f32_e32 v163, v161
	v_rcp_f32_e32 v169, v167
	v_rcp_f32_e32 v175, v173
	v_rcp_f32_e32 v181, v179
	v_rcp_f32_e32 v187, v185
	v_rcp_f32_e32 v193, v191
	v_rcp_f32_e32 v199, v197
	v_rcp_f32_e32 v205, v203
	v_fma_f32 v164, -v161, v163, 1.0
	v_fma_f32 v170, -v167, v169, 1.0
	v_fma_f32 v176, -v173, v175, 1.0
	v_fma_f32 v182, -v179, v181, 1.0
	v_fma_f32 v188, -v185, v187, 1.0
	v_fma_f32 v194, -v191, v193, 1.0
	v_fma_f32 v200, -v197, v199, 1.0
	v_fma_f32 v206, -v203, v205, 1.0
	v_fmac_f32_e32 v163, v164, v163
	v_fmac_f32_e32 v169, v170, v169
	v_fmac_f32_e32 v175, v176, v175
	v_fmac_f32_e32 v181, v182, v181
	v_fmac_f32_e32 v187, v188, v187
	v_fmac_f32_e32 v193, v194, v193
	v_fmac_f32_e32 v199, v200, v199
	v_fmac_f32_e32 v205, v206, v205
	v_mul_f32_e32 v164, v162, v163
	v_mul_f32_e32 v170, v168, v169
	v_mul_f32_e32 v176, v174, v175
	v_mul_f32_e32 v182, v180, v181
	v_mul_f32_e32 v188, v186, v187
	v_mul_f32_e32 v194, v192, v193
	v_mul_f32_e32 v200, v198, v199
	v_mul_f32_e32 v206, v204, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	v_fmac_f32_e32 v164, v165, v163
	v_fmac_f32_e32 v170, v171, v169
	v_fmac_f32_e32 v176, v177, v175
	v_fmac_f32_e32 v182, v183, v181
	v_fmac_f32_e32 v188, v189, v187
	v_fmac_f32_e32 v194, v195, v193
	v_fmac_f32_e32 v200, v201, v199
	v_fmac_f32_e32 v206, v207, v205
	v_fma_f32 v165, -v161, v164, v162
	v_fma_f32 v171, -v167, v170, v168
	v_fma_f32 v177, -v173, v176, v174
	v_fma_f32 v183, -v179, v182, v180
	v_fma_f32 v189, -v185, v188, v186
	v_fma_f32 v195, -v191, v194, v192
	v_fma_f32 v201, -v197, v200, v198
	v_fma_f32 v207, -v203, v206, v204
	s_mov_b32 vcc_lo, s80
	v_div_fmas_f32 v165, v165, v163, v164
	s_mov_b32 vcc_lo, s82
	v_div_fmas_f32 v171, v171, v169, v170
	s_mov_b32 vcc_lo, s84
	v_div_fmas_f32 v177, v177, v175, v176
	s_mov_b32 vcc_lo, s86
	v_div_fmas_f32 v183, v183, v181, v182
	s_mov_b32 vcc_lo, s88
	v_div_fmas_f32 v189, v189, v187, v188
	s_mov_b32 vcc_lo, s90
	v_div_fmas_f32 v195, v195, v193, v194
	s_mov_b32 vcc_lo, s92
	v_div_fmas_f32 v201, v201, v199, v200
	s_mov_b32 vcc_lo, s94
	v_div_fmas_f32 v207, v207, v205, v206
	v_div_fixup_f32 v165, v165, v160, v120
	v_div_fixup_f32 v171, v171, v166, v121
	v_div_fixup_f32 v177, v177, v172, v122
	v_div_fixup_f32 v183, v183, v178, v123
	v_div_fixup_f32 v189, v189, v184, v124
	v_div_fixup_f32 v195, v195, v190, v125
	v_div_fixup_f32 v201, v201, v196, v126
	v_div_fixup_f32 v207, v207, v202, v127
	v_mul_f32_e32 v120, v165, v152
	v_mul_f32_e32 v121, v171, v153
	v_mul_f32_e32 v122, v177, v154
	v_mul_f32_e32 v123, v183, v155
	v_mul_f32_e32 v124, v189, v156
	v_mul_f32_e32 v125, v195, v157
	v_mul_f32_e32 v126, v201, v158
	v_mul_f32_e32 v127, v207, v159
	global_store_b128 v3, v[120:123], s[36:37] offset:64
	global_store_b128 v3, v[124:127], s[36:37] offset:80
	.Lv2b_silu_end:
	s_mov_b32 m0, 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lgemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151_end:
.size gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151, .Lgemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151_end-gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 44
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
	.amdhsa_next_free_vgpr 256
	.amdhsa_next_free_sgpr 96
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151_end-gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151)<<4)&1008)>>4
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
        .name: Xq
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 512
    .name: gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151
    .private_segment_fixed_size: 0
    .sgpr_count: 54
    .sgpr_spill_count: 0
    .symbol: gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151.kd
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
        .name: Xq
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
      - 
        .name: GSHIFT
        .offset: 36
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 512
    .name: gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151
    .private_segment_fixed_size: 0
    .sgpr_count: 44
    .sgpr_spill_count: 0
    .symbol: gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 256
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: G
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: U
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: Xq
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: Y
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: N
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 44
    .max_flat_workgroup_size: 512
    .name: gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151
    .private_segment_fixed_size: 0
    .sgpr_count: 98
    .sgpr_spill_count: 0
    .symbol: gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 256
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1151
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
