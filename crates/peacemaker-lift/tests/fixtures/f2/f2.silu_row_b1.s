// Excerpt of perf/builder-delay/f2/f2.s (SHA-256 eaf7178fa023a3e5ddea4ed8c75811646b6e5e49467bfb8b6a3896f17b6f113f), lines 6066-11133: the gemm_mq4g256v2_fp8_silu_row_b1 kernel.
gemm_mq4g256v2_fp8_silu_row_b1:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b256 s[24:31], s[0:1], 0x40
	s_wait_kmcnt 0x0
	s_add_co_i32 s80, s26, s27
	s_add_co_i32 s80, s80, 0xff
	s_lshr_b32 s80, s80, 8
	s_add_co_i32 s81, s31, 0x7f
	s_lshr_b32 s81, s81, 7
	s_and_b32 s95, ttmp7, 0xffff
	s_mul_i32 s95, s95, s80
	s_add_co_i32 s95, s95, ttmp9
	s_lshl_b32 s96, s80, 3
	s_cvt_f32_u32 s99, s96
	v_s_rcp_f32 s99, s99
	s_mul_f32 s99, s99, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s99, s99
	s_sub_co_i32 s100, 0, s96
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s100, s100, s99
	s_mul_hi_u32 s100, s99, s100
	s_add_co_i32 s99, s99, s100
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s97, s95, s99
	s_mul_i32 s100, s97, s96
	s_sub_co_i32 s98, s95, s100
	s_add_co_i32 s99, s97, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s100, s98, s96
	s_cmp_ge_u32 s98, s96
	s_cselect_b32 s97, s99, s97
	s_cselect_b32 s98, s100, s98
	s_add_co_i32 s99, s97, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s100, s98, s96
	s_cmp_ge_u32 s98, s96
	s_cselect_b32 s97, s99, s97
	s_cselect_b32 s98, s100, s98
	s_lshl_b32 s99, s97, 3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s100, s81, s99
	s_min_u32 s100, s100, 8
	s_cvt_f32_u32 s103, s100
	v_s_rcp_f32 s103, s103
	s_mul_f32 s103, s103, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s103, s103
	s_sub_co_i32 s72, 0, s100
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s72, s72, s103
	s_mul_hi_u32 s72, s103, s72
	s_add_co_i32 s103, s103, s72
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s101, s98, s103
	s_mul_i32 s72, s101, s100
	s_sub_co_i32 s102, s98, s72
	s_add_co_i32 s103, s101, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s72, s102, s100
	s_cmp_ge_u32 s102, s100
	s_cselect_b32 s101, s103, s101
	s_cselect_b32 s102, s72, s102
	s_add_co_i32 s103, s101, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s72, s102, s100
	s_cmp_ge_u32 s102, s100
	s_cselect_b32 s101, s103, s101
	s_cselect_b32 s102, s72, s102
	s_mov_b32 s83, s101
	s_lshl_b32 s82, s97, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s82, s82, s102
	s_lshl_b32 s82, s82, 7
	s_lshr_b32 s84, s30, 7
	s_mov_b32 s85, 0
	s_cmp_ge_u32 s82, s31
	s_cbranch_scc1 .Lfp8_ratio_256x128x8_row_silu_end
	s_mov_b32 s94, s83
	s_mul_i32 s94, s94, s84
	v_readfirstlane_b32 s87, v0
	s_lshr_b32 s87, s87, 5
	s_and_b32 s88, s87, 3
	s_lshr_b32 s89, s87, 2
	s_mul_i32 s73, s31, s30
	s_lshr_b32 s74, s73, 5
	s_mov_b32 s32, s8
	s_and_b32 s33, s9, 0xffff
	s_mov_b32 s34, -1
	s_mov_b32 s35, 0x31004000
	s_mov_b32 s36, s14
	s_and_b32 s37, s15, 0xffff
	s_mov_b32 s38, s73
	s_mov_b32 s39, 0x31004000
	s_mov_b32 s40, s10
	s_and_b32 s41, s11, 0xffff
	s_mov_b32 s42, -1
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s44, s12
	s_and_b32 s45, s13, 0xffff
	s_mov_b32 s46, -1
	s_mov_b32 s47, 0x31004000
	s_mov_b32 s48, s16
	s_and_b32 s49, s17, 0xffff
	s_mov_b32 s50, s74
	s_mov_b32 s51, 0x31004000
	s_mov_b32 s52, s18
	s_and_b32 s53, s19, 0xffff
	s_mov_b32 s54, -1
	s_mov_b32 s55, 0x31004000
	s_mov_b32 s56, s20
	s_and_b32 s57, s21, 0xffff
	s_mov_b32 s58, -1
	s_mov_b32 s59, 0x31004000
	s_mov_b32 s60, s22
	s_and_b32 s61, s23, 0xffff
	s_mov_b32 s62, -1
	s_mov_b32 s63, 0x31004000
	s_mov_b32 s64, s24
	s_and_b32 s65, s25, 0xffff
	s_mov_b32 s66, -1
	s_mov_b32 s67, 0x31004000
	v_mov_b32_e32 v187, v0
	v_and_b32_e32 v176, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v176, 4, v176
	v_lshrrev_b32_e32 v177, 1, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v177, v177, s30
	v_and_b32_e32 v170, 1, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v170, 5, v170
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v177, v177, v170
	s_mul_i32 s72, s82, s30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v177, s72, v177
	v_lshrrev_b32_e32 v188, 1, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v178, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v178, 8, v178
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v171, 15, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v171, 3, v171
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v178, v178, v171
	v_and_b32_e32 v171, 1, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v171, 12, v171
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v178, v178, v171
	v_and_b32_e32 v179, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v179, 3, v179
	s_lshl_b32 s72, s89, 10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v179, s72, v179
	v_lshlrev_b32_e32 v182, 2, v0
	v_and_b32_e32 v183, 0x7f, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v183, s82, v183
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v183, v183, s84
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v183, 2, v183
	v_and_b32_e32 v180, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v180, 4, v180
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v180, 5, v180
	s_lshl_b32 s72, s88, 8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v180, s72, v180
	v_and_b32_e32 v181, 15, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v181, 2, v181
	s_lshl_b32 s72, s89, 8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v181, s72, v181
	s_lshl_b32 s95, s88, 13
	s_lshl_b32 s91, s85, 7
	s_clause 0x3
	buffer_load_b64 v[168:169], v177, s[36:39], s91 offen
	buffer_load_b64 v[170:171], v177, s[36:39], s91 offen offset:8
	buffer_load_b64 v[172:173], v177, s[36:39], s91 offen offset:16
	buffer_load_b64 v[174:175], v177, s[36:39], s91 offen offset:24
	s_wait_loadcnt 0x3
	ds_store_b64 v178, v[168:169]
	s_wait_loadcnt 0x2
	ds_store_b64 v178, v[170:171] offset:128
	s_wait_loadcnt 0x1
	ds_store_b64 v178, v[172:173] offset:2048
	s_wait_loadcnt 0x0
	ds_store_b64 v178, v[174:175] offset:2176
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lfp8_ratio_256x128x8_row_silu_begin:
	s_add_co_i32 s90, s94, s85
	s_lshl_b32 s90, s90, 15
	s_add_co_i32 s90, s90, s95
	s_clause 0x3
	buffer_load_b128 v[128:131], v176, s[32:35], s90 offen
	buffer_load_b128 v[132:135], v176, s[32:35], s90 offen offset:512
	buffer_load_b128 v[136:139], v176, s[32:35], s90 offen offset:1024
	buffer_load_b128 v[140:143], v176, s[32:35], s90 offen offset:1536
	s_clause 0x3
	buffer_load_b128 v[144:147], v176, s[32:35], s90 offen offset:2048
	buffer_load_b128 v[148:151], v176, s[32:35], s90 offen offset:2560
	buffer_load_b128 v[152:155], v176, s[32:35], s90 offen offset:3072
	buffer_load_b128 v[156:159], v176, s[32:35], s90 offen offset:3584
	s_lshl_b32 s91, s85, 7
	s_add_co_i32 s91, s91, 64
	s_clause 0x3
	buffer_load_b64 v[168:169], v177, s[36:39], s91 offen
	buffer_load_b64 v[170:171], v177, s[36:39], s91 offen offset:8
	buffer_load_b64 v[172:173], v177, s[36:39], s91 offen offset:16
	buffer_load_b64 v[174:175], v177, s[36:39], s91 offen offset:24
	ds_load_b64 v[160:161], v179
	ds_load_b64 v[162:163], v179 offset:256
	ds_load_b64 v[164:165], v179 offset:512
	ds_load_b64 v[166:167], v179 offset:768
	s_wait_loadcnt_dscnt 0xb03
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[128:129], v[160:161], 0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[128:129], v[162:163], 0
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[128:129], v[164:165], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[128:129], v[166:167], 0
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[132:133], v[160:161], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[132:133], v[162:163], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[132:133], v[164:165], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[132:133], v[166:167], 0
	s_wait_loadcnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[136:137], v[160:161], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[136:137], v[162:163], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[136:137], v[164:165], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[136:137], v[166:167], 0
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[140:141], v[160:161], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[140:141], v[162:163], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[140:141], v[164:165], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[140:141], v[166:167], 0
	ds_load_b64 v[160:161], v179 offset:2048
	ds_load_b64 v[162:163], v179 offset:2304
	ds_load_b64 v[164:165], v179 offset:2560
	ds_load_b64 v[166:167], v179 offset:2816
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[130:131], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[130:131], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[130:131], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[130:131], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[134:135], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[134:135], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[134:135], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[134:135], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[138:139], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[138:139], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[138:139], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[138:139], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[142:143], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[142:143], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[142:143], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[142:143], v[166:167], v[120:127]
	s_clause 0x3
	buffer_load_b128 v[128:131], v176, s[32:35], s90 offen offset:4096
	buffer_load_b128 v[132:135], v176, s[32:35], s90 offen offset:4608
	buffer_load_b128 v[136:139], v176, s[32:35], s90 offen offset:5120
	buffer_load_b128 v[140:143], v176, s[32:35], s90 offen offset:5632
	ds_load_b64 v[160:161], v179 offset:4096
	ds_load_b64 v[162:163], v179 offset:4352
	ds_load_b64 v[164:165], v179 offset:4608
	ds_load_b64 v[166:167], v179 offset:4864
	s_wait_loadcnt_dscnt 0xb03
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[144:145], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[144:145], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[144:145], v[166:167], v[24:31]
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[148:149], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[148:149], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[148:149], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[148:149], v[166:167], v[56:63]
	s_wait_loadcnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[152:153], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[152:153], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[152:153], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[152:153], v[166:167], v[88:95]
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[156:157], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[156:157], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[156:157], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[156:157], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:6144
	ds_load_b64 v[162:163], v179 offset:6400
	ds_load_b64 v[164:165], v179 offset:6656
	ds_load_b64 v[166:167], v179 offset:6912
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[146:147], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[146:147], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[146:147], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[146:147], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[150:151], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[150:151], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[150:151], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[150:151], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[154:155], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[154:155], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[154:155], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[154:155], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[158:159], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[158:159], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[158:159], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[158:159], v[166:167], v[120:127]
	s_clause 0x3
	buffer_load_b128 v[144:147], v176, s[32:35], s90 offen offset:6144
	buffer_load_b128 v[148:151], v176, s[32:35], s90 offen offset:6656
	buffer_load_b128 v[152:155], v176, s[32:35], s90 offen offset:7168
	buffer_load_b128 v[156:159], v176, s[32:35], s90 offen offset:7680
	s_wait_loadcnt 0xb
	ds_store_b64 v178, v[168:169] offset:8192
	s_wait_loadcnt 0xa
	ds_store_b64 v178, v[170:171] offset:8320
	s_wait_loadcnt 0x9
	ds_store_b64 v178, v[172:173] offset:10240
	s_wait_loadcnt 0x8
	ds_store_b64 v178, v[174:175] offset:10368
	s_add_co_i32 s92, s94, s85
	s_add_co_i32 s92, s92, 1
	s_lshl_b32 s92, s92, 10
	s_wait_dscnt 0x3
	buffer_load_b32 v168, v182, s[40:43], s92 offen
	s_wait_loadcnt 0x0
	ds_store_b32 v182, v168 offset:16384
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_lshl_b32 s91, s85, 7
	s_add_co_i32 s91, s91, 0x80
	s_clause 0x3
	buffer_load_b64 v[168:169], v177, s[36:39], s91 offen
	buffer_load_b64 v[170:171], v177, s[36:39], s91 offen offset:8
	buffer_load_b64 v[172:173], v177, s[36:39], s91 offen offset:16
	buffer_load_b64 v[174:175], v177, s[36:39], s91 offen offset:24
	ds_load_b64 v[160:161], v179 offset:8192
	ds_load_b64 v[162:163], v179 offset:8448
	ds_load_b64 v[164:165], v179 offset:8704
	ds_load_b64 v[166:167], v179 offset:8960
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[128:129], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[128:129], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[128:129], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[128:129], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[132:133], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[132:133], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[132:133], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[132:133], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[136:137], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[136:137], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[136:137], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[136:137], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[140:141], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[140:141], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[140:141], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[140:141], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:10240
	ds_load_b64 v[162:163], v179 offset:10496
	ds_load_b64 v[164:165], v179 offset:10752
	ds_load_b64 v[166:167], v179 offset:11008
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[130:131], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[130:131], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[130:131], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[130:131], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[134:135], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[134:135], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[134:135], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[134:135], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[138:139], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[138:139], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[138:139], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[138:139], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[142:143], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[142:143], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[142:143], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[142:143], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:12288
	ds_load_b64 v[162:163], v179 offset:12544
	ds_load_b64 v[164:165], v179 offset:12800
	ds_load_b64 v[166:167], v179 offset:13056
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[144:145], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[144:145], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[144:145], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[148:149], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[148:149], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[148:149], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[148:149], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[152:153], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[152:153], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[152:153], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[152:153], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[156:157], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[156:157], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[156:157], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[156:157], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:14336
	ds_load_b64 v[162:163], v179 offset:14592
	ds_load_b64 v[164:165], v179 offset:14848
	ds_load_b64 v[166:167], v179 offset:15104
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[146:147], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[146:147], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[146:147], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[146:147], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[150:151], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[150:151], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[150:151], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[150:151], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[154:155], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[154:155], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[154:155], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[154:155], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[158:159], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[158:159], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[158:159], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[158:159], v[166:167], v[120:127]
	s_wait_loadcnt 0x3
	ds_store_b64 v178, v[168:169]
	s_wait_loadcnt 0x2
	ds_store_b64 v178, v[170:171] offset:128
	s_wait_loadcnt 0x1
	ds_store_b64 v178, v[172:173] offset:2048
	s_wait_loadcnt 0x0
	ds_store_b64 v178, v[174:175] offset:2176
	s_wait_dscnt 0x3
	ds_load_b32 v168, v180 offset:16384
	ds_load_b32 v169, v180 offset:16388
	s_wait_dscnt 0x0
	v_dual_mul_f32 v0, v0, v168 :: v_dual_mul_f32 v1, v1, v169
	v_dual_mul_f32 v8, v8, v168 :: v_dual_mul_f32 v9, v9, v169
	v_dual_mul_f32 v16, v16, v168 :: v_dual_mul_f32 v17, v17, v169
	v_dual_mul_f32 v24, v24, v168 :: v_dual_mul_f32 v25, v25, v169
	ds_load_b32 v168, v180 offset:16392
	ds_load_b32 v169, v180 offset:16396
	s_wait_dscnt 0x0
	v_dual_mul_f32 v2, v2, v168 :: v_dual_mul_f32 v3, v3, v169
	v_dual_mul_f32 v10, v10, v168 :: v_dual_mul_f32 v11, v11, v169
	v_dual_mul_f32 v18, v18, v168 :: v_dual_mul_f32 v19, v19, v169
	v_dual_mul_f32 v26, v26, v168 :: v_dual_mul_f32 v27, v27, v169
	ds_load_b32 v168, v180 offset:16400
	ds_load_b32 v169, v180 offset:16404
	s_wait_dscnt 0x0
	v_dual_mul_f32 v4, v4, v168 :: v_dual_mul_f32 v5, v5, v169
	v_dual_mul_f32 v12, v12, v168 :: v_dual_mul_f32 v13, v13, v169
	v_dual_mul_f32 v20, v20, v168 :: v_dual_mul_f32 v21, v21, v169
	v_dual_mul_f32 v28, v28, v168 :: v_dual_mul_f32 v29, v29, v169
	ds_load_b32 v168, v180 offset:16408
	ds_load_b32 v169, v180 offset:16412
	s_wait_dscnt 0x0
	v_dual_mul_f32 v6, v6, v168 :: v_dual_mul_f32 v7, v7, v169
	v_dual_mul_f32 v14, v14, v168 :: v_dual_mul_f32 v15, v15, v169
	v_dual_mul_f32 v22, v22, v168 :: v_dual_mul_f32 v23, v23, v169
	v_dual_mul_f32 v30, v30, v168 :: v_dual_mul_f32 v31, v31, v169
	ds_load_b32 v168, v180 offset:16448
	ds_load_b32 v169, v180 offset:16452
	s_wait_dscnt 0x0
	v_dual_mul_f32 v32, v32, v168 :: v_dual_mul_f32 v33, v33, v169
	v_dual_mul_f32 v40, v40, v168 :: v_dual_mul_f32 v41, v41, v169
	v_dual_mul_f32 v48, v48, v168 :: v_dual_mul_f32 v49, v49, v169
	v_dual_mul_f32 v56, v56, v168 :: v_dual_mul_f32 v57, v57, v169
	ds_load_b32 v168, v180 offset:16456
	ds_load_b32 v169, v180 offset:16460
	s_wait_dscnt 0x0
	v_dual_mul_f32 v34, v34, v168 :: v_dual_mul_f32 v35, v35, v169
	v_dual_mul_f32 v42, v42, v168 :: v_dual_mul_f32 v43, v43, v169
	v_dual_mul_f32 v50, v50, v168 :: v_dual_mul_f32 v51, v51, v169
	v_dual_mul_f32 v58, v58, v168 :: v_dual_mul_f32 v59, v59, v169
	ds_load_b32 v168, v180 offset:16464
	ds_load_b32 v169, v180 offset:16468
	s_wait_dscnt 0x0
	v_dual_mul_f32 v36, v36, v168 :: v_dual_mul_f32 v37, v37, v169
	v_dual_mul_f32 v44, v44, v168 :: v_dual_mul_f32 v45, v45, v169
	v_dual_mul_f32 v52, v52, v168 :: v_dual_mul_f32 v53, v53, v169
	v_dual_mul_f32 v60, v60, v168 :: v_dual_mul_f32 v61, v61, v169
	ds_load_b32 v168, v180 offset:16472
	ds_load_b32 v169, v180 offset:16476
	s_wait_dscnt 0x0
	v_dual_mul_f32 v38, v38, v168 :: v_dual_mul_f32 v39, v39, v169
	v_dual_mul_f32 v46, v46, v168 :: v_dual_mul_f32 v47, v47, v169
	v_dual_mul_f32 v54, v54, v168 :: v_dual_mul_f32 v55, v55, v169
	v_dual_mul_f32 v62, v62, v168 :: v_dual_mul_f32 v63, v63, v169
	ds_load_b32 v168, v180 offset:16512
	ds_load_b32 v169, v180 offset:16516
	s_wait_dscnt 0x0
	v_dual_mul_f32 v64, v64, v168 :: v_dual_mul_f32 v65, v65, v169
	v_dual_mul_f32 v72, v72, v168 :: v_dual_mul_f32 v73, v73, v169
	v_dual_mul_f32 v80, v80, v168 :: v_dual_mul_f32 v81, v81, v169
	v_dual_mul_f32 v88, v88, v168 :: v_dual_mul_f32 v89, v89, v169
	ds_load_b32 v168, v180 offset:16520
	ds_load_b32 v169, v180 offset:16524
	s_wait_dscnt 0x0
	v_dual_mul_f32 v66, v66, v168 :: v_dual_mul_f32 v67, v67, v169
	v_dual_mul_f32 v74, v74, v168 :: v_dual_mul_f32 v75, v75, v169
	v_dual_mul_f32 v82, v82, v168 :: v_dual_mul_f32 v83, v83, v169
	v_dual_mul_f32 v90, v90, v168 :: v_dual_mul_f32 v91, v91, v169
	ds_load_b32 v168, v180 offset:16528
	ds_load_b32 v169, v180 offset:16532
	s_wait_dscnt 0x0
	v_dual_mul_f32 v68, v68, v168 :: v_dual_mul_f32 v69, v69, v169
	v_dual_mul_f32 v76, v76, v168 :: v_dual_mul_f32 v77, v77, v169
	v_dual_mul_f32 v84, v84, v168 :: v_dual_mul_f32 v85, v85, v169
	v_dual_mul_f32 v92, v92, v168 :: v_dual_mul_f32 v93, v93, v169
	ds_load_b32 v168, v180 offset:16536
	ds_load_b32 v169, v180 offset:16540
	s_wait_dscnt 0x0
	v_dual_mul_f32 v70, v70, v168 :: v_dual_mul_f32 v71, v71, v169
	v_dual_mul_f32 v78, v78, v168 :: v_dual_mul_f32 v79, v79, v169
	v_dual_mul_f32 v86, v86, v168 :: v_dual_mul_f32 v87, v87, v169
	v_dual_mul_f32 v94, v94, v168 :: v_dual_mul_f32 v95, v95, v169
	ds_load_b32 v168, v180 offset:16576
	ds_load_b32 v169, v180 offset:16580
	s_wait_dscnt 0x0
	v_dual_mul_f32 v96, v96, v168 :: v_dual_mul_f32 v97, v97, v169
	v_dual_mul_f32 v104, v104, v168 :: v_dual_mul_f32 v105, v105, v169
	v_dual_mul_f32 v112, v112, v168 :: v_dual_mul_f32 v113, v113, v169
	v_dual_mul_f32 v120, v120, v168 :: v_dual_mul_f32 v121, v121, v169
	ds_load_b32 v168, v180 offset:16584
	ds_load_b32 v169, v180 offset:16588
	s_wait_dscnt 0x0
	v_dual_mul_f32 v98, v98, v168 :: v_dual_mul_f32 v99, v99, v169
	v_dual_mul_f32 v106, v106, v168 :: v_dual_mul_f32 v107, v107, v169
	v_dual_mul_f32 v114, v114, v168 :: v_dual_mul_f32 v115, v115, v169
	v_dual_mul_f32 v122, v122, v168 :: v_dual_mul_f32 v123, v123, v169
	ds_load_b32 v168, v180 offset:16592
	ds_load_b32 v169, v180 offset:16596
	s_wait_dscnt 0x0
	v_dual_mul_f32 v100, v100, v168 :: v_dual_mul_f32 v101, v101, v169
	v_dual_mul_f32 v108, v108, v168 :: v_dual_mul_f32 v109, v109, v169
	v_dual_mul_f32 v116, v116, v168 :: v_dual_mul_f32 v117, v117, v169
	v_dual_mul_f32 v124, v124, v168 :: v_dual_mul_f32 v125, v125, v169
	ds_load_b32 v168, v180 offset:16600
	ds_load_b32 v169, v180 offset:16604
	s_wait_dscnt 0x0
	v_dual_mul_f32 v102, v102, v168 :: v_dual_mul_f32 v103, v103, v169
	v_dual_mul_f32 v110, v110, v168 :: v_dual_mul_f32 v111, v111, v169
	v_dual_mul_f32 v118, v118, v168 :: v_dual_mul_f32 v119, v119, v169
	v_dual_mul_f32 v126, v126, v168 :: v_dual_mul_f32 v127, v127, v169
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s85, s85, 1
	s_add_co_i32 s86, s84, -2
	s_lshr_b32 s86, s86, 1
	s_cmp_eq_u32 s86, 0
	s_cbranch_scc1 .Lfp8_ratio_256x128x8_row_silu_loop_end
	.Lfp8_ratio_256x128x8_row_silu_loop:
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s90, s94, s85
	s_lshl_b32 s90, s90, 15
	s_add_co_i32 s90, s90, s95
	s_clause 0x3
	buffer_load_b128 v[128:131], v176, s[32:35], s90 offen
	buffer_load_b128 v[132:135], v176, s[32:35], s90 offen offset:512
	buffer_load_b128 v[136:139], v176, s[32:35], s90 offen offset:1024
	buffer_load_b128 v[140:143], v176, s[32:35], s90 offen offset:1536
	s_clause 0x3
	buffer_load_b128 v[144:147], v176, s[32:35], s90 offen offset:2048
	buffer_load_b128 v[148:151], v176, s[32:35], s90 offen offset:2560
	buffer_load_b128 v[152:155], v176, s[32:35], s90 offen offset:3072
	buffer_load_b128 v[156:159], v176, s[32:35], s90 offen offset:3584
	s_lshl_b32 s91, s85, 7
	s_add_co_i32 s91, s91, 64
	s_clause 0x3
	buffer_load_b64 v[168:169], v177, s[36:39], s91 offen
	buffer_load_b64 v[170:171], v177, s[36:39], s91 offen offset:8
	buffer_load_b64 v[172:173], v177, s[36:39], s91 offen offset:16
	buffer_load_b64 v[174:175], v177, s[36:39], s91 offen offset:24
	ds_load_b64 v[160:161], v179
	ds_load_b64 v[162:163], v179 offset:256
	ds_load_b64 v[164:165], v179 offset:512
	ds_load_b64 v[166:167], v179 offset:768
	s_wait_loadcnt_dscnt 0xb03
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[128:129], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[128:129], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[128:129], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[128:129], v[166:167], v[24:31]
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[132:133], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[132:133], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[132:133], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[132:133], v[166:167], v[56:63]
	s_wait_loadcnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[136:137], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[136:137], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[136:137], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[136:137], v[166:167], v[88:95]
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[140:141], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[140:141], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[140:141], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[140:141], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:2048
	ds_load_b64 v[162:163], v179 offset:2304
	ds_load_b64 v[164:165], v179 offset:2560
	ds_load_b64 v[166:167], v179 offset:2816
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[130:131], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[130:131], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[130:131], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[130:131], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[134:135], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[134:135], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[134:135], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[134:135], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[138:139], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[138:139], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[138:139], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[138:139], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[142:143], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[142:143], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[142:143], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[142:143], v[166:167], v[120:127]
	s_clause 0x3
	buffer_load_b128 v[128:131], v176, s[32:35], s90 offen offset:4096
	buffer_load_b128 v[132:135], v176, s[32:35], s90 offen offset:4608
	buffer_load_b128 v[136:139], v176, s[32:35], s90 offen offset:5120
	buffer_load_b128 v[140:143], v176, s[32:35], s90 offen offset:5632
	ds_load_b64 v[160:161], v179 offset:4096
	ds_load_b64 v[162:163], v179 offset:4352
	ds_load_b64 v[164:165], v179 offset:4608
	ds_load_b64 v[166:167], v179 offset:4864
	s_wait_loadcnt_dscnt 0xb03
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[144:145], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[144:145], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[144:145], v[166:167], v[24:31]
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[148:149], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[148:149], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[148:149], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[148:149], v[166:167], v[56:63]
	s_wait_loadcnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[152:153], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[152:153], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[152:153], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[152:153], v[166:167], v[88:95]
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[156:157], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[156:157], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[156:157], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[156:157], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:6144
	ds_load_b64 v[162:163], v179 offset:6400
	ds_load_b64 v[164:165], v179 offset:6656
	ds_load_b64 v[166:167], v179 offset:6912
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[146:147], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[146:147], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[146:147], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[146:147], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[150:151], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[150:151], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[150:151], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[150:151], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[154:155], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[154:155], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[154:155], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[154:155], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[158:159], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[158:159], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[158:159], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[158:159], v[166:167], v[120:127]
	s_clause 0x3
	buffer_load_b128 v[144:147], v176, s[32:35], s90 offen offset:6144
	buffer_load_b128 v[148:151], v176, s[32:35], s90 offen offset:6656
	buffer_load_b128 v[152:155], v176, s[32:35], s90 offen offset:7168
	buffer_load_b128 v[156:159], v176, s[32:35], s90 offen offset:7680
	s_wait_loadcnt 0xb
	ds_store_b64 v178, v[168:169] offset:8192
	s_wait_loadcnt 0xa
	ds_store_b64 v178, v[170:171] offset:8320
	s_wait_loadcnt 0x9
	ds_store_b64 v178, v[172:173] offset:10240
	s_wait_loadcnt 0x8
	ds_store_b64 v178, v[174:175] offset:10368
	s_add_co_i32 s92, s94, s85
	s_add_co_i32 s92, s92, 1
	s_lshl_b32 s92, s92, 10
	s_wait_dscnt 0x3
	buffer_load_b32 v168, v182, s[40:43], s92 offen
	s_wait_loadcnt 0x0
	ds_store_b32 v182, v168 offset:17408
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_lshl_b32 s91, s85, 7
	s_add_co_i32 s91, s91, 0x80
	s_clause 0x3
	buffer_load_b64 v[168:169], v177, s[36:39], s91 offen
	buffer_load_b64 v[170:171], v177, s[36:39], s91 offen offset:8
	buffer_load_b64 v[172:173], v177, s[36:39], s91 offen offset:16
	buffer_load_b64 v[174:175], v177, s[36:39], s91 offen offset:24
	ds_load_b64 v[160:161], v179 offset:8192
	ds_load_b64 v[162:163], v179 offset:8448
	ds_load_b64 v[164:165], v179 offset:8704
	ds_load_b64 v[166:167], v179 offset:8960
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[128:129], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[128:129], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[128:129], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[128:129], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[132:133], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[132:133], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[132:133], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[132:133], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[136:137], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[136:137], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[136:137], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[136:137], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[140:141], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[140:141], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[140:141], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[140:141], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:10240
	ds_load_b64 v[162:163], v179 offset:10496
	ds_load_b64 v[164:165], v179 offset:10752
	ds_load_b64 v[166:167], v179 offset:11008
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[130:131], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[130:131], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[130:131], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[130:131], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[134:135], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[134:135], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[134:135], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[134:135], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[138:139], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[138:139], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[138:139], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[138:139], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[142:143], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[142:143], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[142:143], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[142:143], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:12288
	ds_load_b64 v[162:163], v179 offset:12544
	ds_load_b64 v[164:165], v179 offset:12800
	ds_load_b64 v[166:167], v179 offset:13056
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[144:145], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[144:145], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[144:145], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[148:149], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[148:149], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[148:149], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[148:149], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[152:153], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[152:153], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[152:153], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[152:153], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[156:157], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[156:157], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[156:157], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[156:157], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:14336
	ds_load_b64 v[162:163], v179 offset:14592
	ds_load_b64 v[164:165], v179 offset:14848
	ds_load_b64 v[166:167], v179 offset:15104
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[146:147], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[146:147], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[146:147], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[146:147], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[150:151], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[150:151], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[150:151], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[150:151], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[154:155], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[154:155], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[154:155], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[154:155], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[158:159], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[158:159], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[158:159], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[158:159], v[166:167], v[120:127]
	s_wait_loadcnt 0x3
	ds_store_b64 v178, v[168:169]
	s_wait_loadcnt 0x2
	ds_store_b64 v178, v[170:171] offset:128
	s_wait_loadcnt 0x1
	ds_store_b64 v178, v[172:173] offset:2048
	s_wait_loadcnt 0x0
	ds_store_b64 v178, v[174:175] offset:2176
	s_wait_dscnt 0x3
	ds_load_b32 v168, v180 offset:17408
	ds_load_b32 v169, v180 offset:17412
	s_wait_dscnt 0x0
	v_dual_mul_f32 v0, v0, v168 :: v_dual_mul_f32 v1, v1, v169
	v_dual_mul_f32 v8, v8, v168 :: v_dual_mul_f32 v9, v9, v169
	v_dual_mul_f32 v16, v16, v168 :: v_dual_mul_f32 v17, v17, v169
	v_dual_mul_f32 v24, v24, v168 :: v_dual_mul_f32 v25, v25, v169
	ds_load_b32 v168, v180 offset:17416
	ds_load_b32 v169, v180 offset:17420
	s_wait_dscnt 0x0
	v_dual_mul_f32 v2, v2, v168 :: v_dual_mul_f32 v3, v3, v169
	v_dual_mul_f32 v10, v10, v168 :: v_dual_mul_f32 v11, v11, v169
	v_dual_mul_f32 v18, v18, v168 :: v_dual_mul_f32 v19, v19, v169
	v_dual_mul_f32 v26, v26, v168 :: v_dual_mul_f32 v27, v27, v169
	ds_load_b32 v168, v180 offset:17424
	ds_load_b32 v169, v180 offset:17428
	s_wait_dscnt 0x0
	v_dual_mul_f32 v4, v4, v168 :: v_dual_mul_f32 v5, v5, v169
	v_dual_mul_f32 v12, v12, v168 :: v_dual_mul_f32 v13, v13, v169
	v_dual_mul_f32 v20, v20, v168 :: v_dual_mul_f32 v21, v21, v169
	v_dual_mul_f32 v28, v28, v168 :: v_dual_mul_f32 v29, v29, v169
	ds_load_b32 v168, v180 offset:17432
	ds_load_b32 v169, v180 offset:17436
	s_wait_dscnt 0x0
	v_dual_mul_f32 v6, v6, v168 :: v_dual_mul_f32 v7, v7, v169
	v_dual_mul_f32 v14, v14, v168 :: v_dual_mul_f32 v15, v15, v169
	v_dual_mul_f32 v22, v22, v168 :: v_dual_mul_f32 v23, v23, v169
	v_dual_mul_f32 v30, v30, v168 :: v_dual_mul_f32 v31, v31, v169
	ds_load_b32 v168, v180 offset:17472
	ds_load_b32 v169, v180 offset:17476
	s_wait_dscnt 0x0
	v_dual_mul_f32 v32, v32, v168 :: v_dual_mul_f32 v33, v33, v169
	v_dual_mul_f32 v40, v40, v168 :: v_dual_mul_f32 v41, v41, v169
	v_dual_mul_f32 v48, v48, v168 :: v_dual_mul_f32 v49, v49, v169
	v_dual_mul_f32 v56, v56, v168 :: v_dual_mul_f32 v57, v57, v169
	ds_load_b32 v168, v180 offset:17480
	ds_load_b32 v169, v180 offset:17484
	s_wait_dscnt 0x0
	v_dual_mul_f32 v34, v34, v168 :: v_dual_mul_f32 v35, v35, v169
	v_dual_mul_f32 v42, v42, v168 :: v_dual_mul_f32 v43, v43, v169
	v_dual_mul_f32 v50, v50, v168 :: v_dual_mul_f32 v51, v51, v169
	v_dual_mul_f32 v58, v58, v168 :: v_dual_mul_f32 v59, v59, v169
	ds_load_b32 v168, v180 offset:17488
	ds_load_b32 v169, v180 offset:17492
	s_wait_dscnt 0x0
	v_dual_mul_f32 v36, v36, v168 :: v_dual_mul_f32 v37, v37, v169
	v_dual_mul_f32 v44, v44, v168 :: v_dual_mul_f32 v45, v45, v169
	v_dual_mul_f32 v52, v52, v168 :: v_dual_mul_f32 v53, v53, v169
	v_dual_mul_f32 v60, v60, v168 :: v_dual_mul_f32 v61, v61, v169
	ds_load_b32 v168, v180 offset:17496
	ds_load_b32 v169, v180 offset:17500
	s_wait_dscnt 0x0
	v_dual_mul_f32 v38, v38, v168 :: v_dual_mul_f32 v39, v39, v169
	v_dual_mul_f32 v46, v46, v168 :: v_dual_mul_f32 v47, v47, v169
	v_dual_mul_f32 v54, v54, v168 :: v_dual_mul_f32 v55, v55, v169
	v_dual_mul_f32 v62, v62, v168 :: v_dual_mul_f32 v63, v63, v169
	ds_load_b32 v168, v180 offset:17536
	ds_load_b32 v169, v180 offset:17540
	s_wait_dscnt 0x0
	v_dual_mul_f32 v64, v64, v168 :: v_dual_mul_f32 v65, v65, v169
	v_dual_mul_f32 v72, v72, v168 :: v_dual_mul_f32 v73, v73, v169
	v_dual_mul_f32 v80, v80, v168 :: v_dual_mul_f32 v81, v81, v169
	v_dual_mul_f32 v88, v88, v168 :: v_dual_mul_f32 v89, v89, v169
	ds_load_b32 v168, v180 offset:17544
	ds_load_b32 v169, v180 offset:17548
	s_wait_dscnt 0x0
	v_dual_mul_f32 v66, v66, v168 :: v_dual_mul_f32 v67, v67, v169
	v_dual_mul_f32 v74, v74, v168 :: v_dual_mul_f32 v75, v75, v169
	v_dual_mul_f32 v82, v82, v168 :: v_dual_mul_f32 v83, v83, v169
	v_dual_mul_f32 v90, v90, v168 :: v_dual_mul_f32 v91, v91, v169
	ds_load_b32 v168, v180 offset:17552
	ds_load_b32 v169, v180 offset:17556
	s_wait_dscnt 0x0
	v_dual_mul_f32 v68, v68, v168 :: v_dual_mul_f32 v69, v69, v169
	v_dual_mul_f32 v76, v76, v168 :: v_dual_mul_f32 v77, v77, v169
	v_dual_mul_f32 v84, v84, v168 :: v_dual_mul_f32 v85, v85, v169
	v_dual_mul_f32 v92, v92, v168 :: v_dual_mul_f32 v93, v93, v169
	ds_load_b32 v168, v180 offset:17560
	ds_load_b32 v169, v180 offset:17564
	s_wait_dscnt 0x0
	v_dual_mul_f32 v70, v70, v168 :: v_dual_mul_f32 v71, v71, v169
	v_dual_mul_f32 v78, v78, v168 :: v_dual_mul_f32 v79, v79, v169
	v_dual_mul_f32 v86, v86, v168 :: v_dual_mul_f32 v87, v87, v169
	v_dual_mul_f32 v94, v94, v168 :: v_dual_mul_f32 v95, v95, v169
	ds_load_b32 v168, v180 offset:17600
	ds_load_b32 v169, v180 offset:17604
	s_wait_dscnt 0x0
	v_dual_mul_f32 v96, v96, v168 :: v_dual_mul_f32 v97, v97, v169
	v_dual_mul_f32 v104, v104, v168 :: v_dual_mul_f32 v105, v105, v169
	v_dual_mul_f32 v112, v112, v168 :: v_dual_mul_f32 v113, v113, v169
	v_dual_mul_f32 v120, v120, v168 :: v_dual_mul_f32 v121, v121, v169
	ds_load_b32 v168, v180 offset:17608
	ds_load_b32 v169, v180 offset:17612
	s_wait_dscnt 0x0
	v_dual_mul_f32 v98, v98, v168 :: v_dual_mul_f32 v99, v99, v169
	v_dual_mul_f32 v106, v106, v168 :: v_dual_mul_f32 v107, v107, v169
	v_dual_mul_f32 v114, v114, v168 :: v_dual_mul_f32 v115, v115, v169
	v_dual_mul_f32 v122, v122, v168 :: v_dual_mul_f32 v123, v123, v169
	ds_load_b32 v168, v180 offset:17616
	ds_load_b32 v169, v180 offset:17620
	s_wait_dscnt 0x0
	v_dual_mul_f32 v100, v100, v168 :: v_dual_mul_f32 v101, v101, v169
	v_dual_mul_f32 v108, v108, v168 :: v_dual_mul_f32 v109, v109, v169
	v_dual_mul_f32 v116, v116, v168 :: v_dual_mul_f32 v117, v117, v169
	v_dual_mul_f32 v124, v124, v168 :: v_dual_mul_f32 v125, v125, v169
	ds_load_b32 v168, v180 offset:17624
	ds_load_b32 v169, v180 offset:17628
	s_wait_dscnt 0x0
	v_dual_mul_f32 v102, v102, v168 :: v_dual_mul_f32 v103, v103, v169
	v_dual_mul_f32 v110, v110, v168 :: v_dual_mul_f32 v111, v111, v169
	v_dual_mul_f32 v118, v118, v168 :: v_dual_mul_f32 v119, v119, v169
	v_dual_mul_f32 v126, v126, v168 :: v_dual_mul_f32 v127, v127, v169
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s85, s85, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s90, s94, s85
	s_lshl_b32 s90, s90, 15
	s_add_co_i32 s90, s90, s95
	s_clause 0x3
	buffer_load_b128 v[128:131], v176, s[32:35], s90 offen
	buffer_load_b128 v[132:135], v176, s[32:35], s90 offen offset:512
	buffer_load_b128 v[136:139], v176, s[32:35], s90 offen offset:1024
	buffer_load_b128 v[140:143], v176, s[32:35], s90 offen offset:1536
	s_clause 0x3
	buffer_load_b128 v[144:147], v176, s[32:35], s90 offen offset:2048
	buffer_load_b128 v[148:151], v176, s[32:35], s90 offen offset:2560
	buffer_load_b128 v[152:155], v176, s[32:35], s90 offen offset:3072
	buffer_load_b128 v[156:159], v176, s[32:35], s90 offen offset:3584
	s_lshl_b32 s91, s85, 7
	s_add_co_i32 s91, s91, 64
	s_clause 0x3
	buffer_load_b64 v[168:169], v177, s[36:39], s91 offen
	buffer_load_b64 v[170:171], v177, s[36:39], s91 offen offset:8
	buffer_load_b64 v[172:173], v177, s[36:39], s91 offen offset:16
	buffer_load_b64 v[174:175], v177, s[36:39], s91 offen offset:24
	ds_load_b64 v[160:161], v179
	ds_load_b64 v[162:163], v179 offset:256
	ds_load_b64 v[164:165], v179 offset:512
	ds_load_b64 v[166:167], v179 offset:768
	s_wait_loadcnt_dscnt 0xb03
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[128:129], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[128:129], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[128:129], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[128:129], v[166:167], v[24:31]
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[132:133], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[132:133], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[132:133], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[132:133], v[166:167], v[56:63]
	s_wait_loadcnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[136:137], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[136:137], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[136:137], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[136:137], v[166:167], v[88:95]
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[140:141], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[140:141], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[140:141], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[140:141], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:2048
	ds_load_b64 v[162:163], v179 offset:2304
	ds_load_b64 v[164:165], v179 offset:2560
	ds_load_b64 v[166:167], v179 offset:2816
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[130:131], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[130:131], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[130:131], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[130:131], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[134:135], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[134:135], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[134:135], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[134:135], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[138:139], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[138:139], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[138:139], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[138:139], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[142:143], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[142:143], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[142:143], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[142:143], v[166:167], v[120:127]
	s_clause 0x3
	buffer_load_b128 v[128:131], v176, s[32:35], s90 offen offset:4096
	buffer_load_b128 v[132:135], v176, s[32:35], s90 offen offset:4608
	buffer_load_b128 v[136:139], v176, s[32:35], s90 offen offset:5120
	buffer_load_b128 v[140:143], v176, s[32:35], s90 offen offset:5632
	ds_load_b64 v[160:161], v179 offset:4096
	ds_load_b64 v[162:163], v179 offset:4352
	ds_load_b64 v[164:165], v179 offset:4608
	ds_load_b64 v[166:167], v179 offset:4864
	s_wait_loadcnt_dscnt 0xb03
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[144:145], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[144:145], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[144:145], v[166:167], v[24:31]
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[148:149], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[148:149], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[148:149], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[148:149], v[166:167], v[56:63]
	s_wait_loadcnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[152:153], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[152:153], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[152:153], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[152:153], v[166:167], v[88:95]
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[156:157], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[156:157], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[156:157], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[156:157], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:6144
	ds_load_b64 v[162:163], v179 offset:6400
	ds_load_b64 v[164:165], v179 offset:6656
	ds_load_b64 v[166:167], v179 offset:6912
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[146:147], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[146:147], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[146:147], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[146:147], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[150:151], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[150:151], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[150:151], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[150:151], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[154:155], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[154:155], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[154:155], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[154:155], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[158:159], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[158:159], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[158:159], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[158:159], v[166:167], v[120:127]
	s_clause 0x3
	buffer_load_b128 v[144:147], v176, s[32:35], s90 offen offset:6144
	buffer_load_b128 v[148:151], v176, s[32:35], s90 offen offset:6656
	buffer_load_b128 v[152:155], v176, s[32:35], s90 offen offset:7168
	buffer_load_b128 v[156:159], v176, s[32:35], s90 offen offset:7680
	s_wait_loadcnt 0xb
	ds_store_b64 v178, v[168:169] offset:8192
	s_wait_loadcnt 0xa
	ds_store_b64 v178, v[170:171] offset:8320
	s_wait_loadcnt 0x9
	ds_store_b64 v178, v[172:173] offset:10240
	s_wait_loadcnt 0x8
	ds_store_b64 v178, v[174:175] offset:10368
	s_add_co_i32 s92, s94, s85
	s_add_co_i32 s92, s92, 1
	s_lshl_b32 s92, s92, 10
	s_wait_dscnt 0x3
	buffer_load_b32 v168, v182, s[40:43], s92 offen
	s_wait_loadcnt 0x0
	ds_store_b32 v182, v168 offset:17408
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_lshl_b32 s91, s85, 7
	s_add_co_i32 s91, s91, 0x80
	s_clause 0x3
	buffer_load_b64 v[168:169], v177, s[36:39], s91 offen
	buffer_load_b64 v[170:171], v177, s[36:39], s91 offen offset:8
	buffer_load_b64 v[172:173], v177, s[36:39], s91 offen offset:16
	buffer_load_b64 v[174:175], v177, s[36:39], s91 offen offset:24
	ds_load_b64 v[160:161], v179 offset:8192
	ds_load_b64 v[162:163], v179 offset:8448
	ds_load_b64 v[164:165], v179 offset:8704
	ds_load_b64 v[166:167], v179 offset:8960
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[128:129], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[128:129], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[128:129], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[128:129], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[132:133], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[132:133], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[132:133], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[132:133], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[136:137], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[136:137], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[136:137], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[136:137], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[140:141], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[140:141], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[140:141], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[140:141], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:10240
	ds_load_b64 v[162:163], v179 offset:10496
	ds_load_b64 v[164:165], v179 offset:10752
	ds_load_b64 v[166:167], v179 offset:11008
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[130:131], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[130:131], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[130:131], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[130:131], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[134:135], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[134:135], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[134:135], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[134:135], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[138:139], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[138:139], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[138:139], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[138:139], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[142:143], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[142:143], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[142:143], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[142:143], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:12288
	ds_load_b64 v[162:163], v179 offset:12544
	ds_load_b64 v[164:165], v179 offset:12800
	ds_load_b64 v[166:167], v179 offset:13056
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[144:145], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[144:145], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[144:145], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[148:149], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[148:149], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[148:149], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[148:149], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[152:153], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[152:153], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[152:153], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[152:153], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[156:157], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[156:157], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[156:157], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[156:157], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:14336
	ds_load_b64 v[162:163], v179 offset:14592
	ds_load_b64 v[164:165], v179 offset:14848
	ds_load_b64 v[166:167], v179 offset:15104
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[146:147], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[146:147], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[146:147], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[146:147], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[150:151], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[150:151], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[150:151], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[150:151], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[154:155], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[154:155], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[154:155], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[154:155], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[158:159], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[158:159], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[158:159], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[158:159], v[166:167], v[120:127]
	s_wait_loadcnt 0x3
	ds_store_b64 v178, v[168:169]
	s_wait_loadcnt 0x2
	ds_store_b64 v178, v[170:171] offset:128
	s_wait_loadcnt 0x1
	ds_store_b64 v178, v[172:173] offset:2048
	s_wait_loadcnt 0x0
	ds_store_b64 v178, v[174:175] offset:2176
	s_wait_dscnt 0x3
	ds_load_b32 v168, v180 offset:17408
	ds_load_b32 v169, v180 offset:17412
	s_wait_dscnt 0x0
	v_dual_mul_f32 v0, v0, v168 :: v_dual_mul_f32 v1, v1, v169
	v_dual_mul_f32 v8, v8, v168 :: v_dual_mul_f32 v9, v9, v169
	v_dual_mul_f32 v16, v16, v168 :: v_dual_mul_f32 v17, v17, v169
	v_dual_mul_f32 v24, v24, v168 :: v_dual_mul_f32 v25, v25, v169
	ds_load_b32 v168, v180 offset:17416
	ds_load_b32 v169, v180 offset:17420
	s_wait_dscnt 0x0
	v_dual_mul_f32 v2, v2, v168 :: v_dual_mul_f32 v3, v3, v169
	v_dual_mul_f32 v10, v10, v168 :: v_dual_mul_f32 v11, v11, v169
	v_dual_mul_f32 v18, v18, v168 :: v_dual_mul_f32 v19, v19, v169
	v_dual_mul_f32 v26, v26, v168 :: v_dual_mul_f32 v27, v27, v169
	ds_load_b32 v168, v180 offset:17424
	ds_load_b32 v169, v180 offset:17428
	s_wait_dscnt 0x0
	v_dual_mul_f32 v4, v4, v168 :: v_dual_mul_f32 v5, v5, v169
	v_dual_mul_f32 v12, v12, v168 :: v_dual_mul_f32 v13, v13, v169
	v_dual_mul_f32 v20, v20, v168 :: v_dual_mul_f32 v21, v21, v169
	v_dual_mul_f32 v28, v28, v168 :: v_dual_mul_f32 v29, v29, v169
	ds_load_b32 v168, v180 offset:17432
	ds_load_b32 v169, v180 offset:17436
	s_wait_dscnt 0x0
	v_dual_mul_f32 v6, v6, v168 :: v_dual_mul_f32 v7, v7, v169
	v_dual_mul_f32 v14, v14, v168 :: v_dual_mul_f32 v15, v15, v169
	v_dual_mul_f32 v22, v22, v168 :: v_dual_mul_f32 v23, v23, v169
	v_dual_mul_f32 v30, v30, v168 :: v_dual_mul_f32 v31, v31, v169
	ds_load_b32 v168, v180 offset:17472
	ds_load_b32 v169, v180 offset:17476
	s_wait_dscnt 0x0
	v_dual_mul_f32 v32, v32, v168 :: v_dual_mul_f32 v33, v33, v169
	v_dual_mul_f32 v40, v40, v168 :: v_dual_mul_f32 v41, v41, v169
	v_dual_mul_f32 v48, v48, v168 :: v_dual_mul_f32 v49, v49, v169
	v_dual_mul_f32 v56, v56, v168 :: v_dual_mul_f32 v57, v57, v169
	ds_load_b32 v168, v180 offset:17480
	ds_load_b32 v169, v180 offset:17484
	s_wait_dscnt 0x0
	v_dual_mul_f32 v34, v34, v168 :: v_dual_mul_f32 v35, v35, v169
	v_dual_mul_f32 v42, v42, v168 :: v_dual_mul_f32 v43, v43, v169
	v_dual_mul_f32 v50, v50, v168 :: v_dual_mul_f32 v51, v51, v169
	v_dual_mul_f32 v58, v58, v168 :: v_dual_mul_f32 v59, v59, v169
	ds_load_b32 v168, v180 offset:17488
	ds_load_b32 v169, v180 offset:17492
	s_wait_dscnt 0x0
	v_dual_mul_f32 v36, v36, v168 :: v_dual_mul_f32 v37, v37, v169
	v_dual_mul_f32 v44, v44, v168 :: v_dual_mul_f32 v45, v45, v169
	v_dual_mul_f32 v52, v52, v168 :: v_dual_mul_f32 v53, v53, v169
	v_dual_mul_f32 v60, v60, v168 :: v_dual_mul_f32 v61, v61, v169
	ds_load_b32 v168, v180 offset:17496
	ds_load_b32 v169, v180 offset:17500
	s_wait_dscnt 0x0
	v_dual_mul_f32 v38, v38, v168 :: v_dual_mul_f32 v39, v39, v169
	v_dual_mul_f32 v46, v46, v168 :: v_dual_mul_f32 v47, v47, v169
	v_dual_mul_f32 v54, v54, v168 :: v_dual_mul_f32 v55, v55, v169
	v_dual_mul_f32 v62, v62, v168 :: v_dual_mul_f32 v63, v63, v169
	ds_load_b32 v168, v180 offset:17536
	ds_load_b32 v169, v180 offset:17540
	s_wait_dscnt 0x0
	v_dual_mul_f32 v64, v64, v168 :: v_dual_mul_f32 v65, v65, v169
	v_dual_mul_f32 v72, v72, v168 :: v_dual_mul_f32 v73, v73, v169
	v_dual_mul_f32 v80, v80, v168 :: v_dual_mul_f32 v81, v81, v169
	v_dual_mul_f32 v88, v88, v168 :: v_dual_mul_f32 v89, v89, v169
	ds_load_b32 v168, v180 offset:17544
	ds_load_b32 v169, v180 offset:17548
	s_wait_dscnt 0x0
	v_dual_mul_f32 v66, v66, v168 :: v_dual_mul_f32 v67, v67, v169
	v_dual_mul_f32 v74, v74, v168 :: v_dual_mul_f32 v75, v75, v169
	v_dual_mul_f32 v82, v82, v168 :: v_dual_mul_f32 v83, v83, v169
	v_dual_mul_f32 v90, v90, v168 :: v_dual_mul_f32 v91, v91, v169
	ds_load_b32 v168, v180 offset:17552
	ds_load_b32 v169, v180 offset:17556
	s_wait_dscnt 0x0
	v_dual_mul_f32 v68, v68, v168 :: v_dual_mul_f32 v69, v69, v169
	v_dual_mul_f32 v76, v76, v168 :: v_dual_mul_f32 v77, v77, v169
	v_dual_mul_f32 v84, v84, v168 :: v_dual_mul_f32 v85, v85, v169
	v_dual_mul_f32 v92, v92, v168 :: v_dual_mul_f32 v93, v93, v169
	ds_load_b32 v168, v180 offset:17560
	ds_load_b32 v169, v180 offset:17564
	s_wait_dscnt 0x0
	v_dual_mul_f32 v70, v70, v168 :: v_dual_mul_f32 v71, v71, v169
	v_dual_mul_f32 v78, v78, v168 :: v_dual_mul_f32 v79, v79, v169
	v_dual_mul_f32 v86, v86, v168 :: v_dual_mul_f32 v87, v87, v169
	v_dual_mul_f32 v94, v94, v168 :: v_dual_mul_f32 v95, v95, v169
	ds_load_b32 v168, v180 offset:17600
	ds_load_b32 v169, v180 offset:17604
	s_wait_dscnt 0x0
	v_dual_mul_f32 v96, v96, v168 :: v_dual_mul_f32 v97, v97, v169
	v_dual_mul_f32 v104, v104, v168 :: v_dual_mul_f32 v105, v105, v169
	v_dual_mul_f32 v112, v112, v168 :: v_dual_mul_f32 v113, v113, v169
	v_dual_mul_f32 v120, v120, v168 :: v_dual_mul_f32 v121, v121, v169
	ds_load_b32 v168, v180 offset:17608
	ds_load_b32 v169, v180 offset:17612
	s_wait_dscnt 0x0
	v_dual_mul_f32 v98, v98, v168 :: v_dual_mul_f32 v99, v99, v169
	v_dual_mul_f32 v106, v106, v168 :: v_dual_mul_f32 v107, v107, v169
	v_dual_mul_f32 v114, v114, v168 :: v_dual_mul_f32 v115, v115, v169
	v_dual_mul_f32 v122, v122, v168 :: v_dual_mul_f32 v123, v123, v169
	ds_load_b32 v168, v180 offset:17616
	ds_load_b32 v169, v180 offset:17620
	s_wait_dscnt 0x0
	v_dual_mul_f32 v100, v100, v168 :: v_dual_mul_f32 v101, v101, v169
	v_dual_mul_f32 v108, v108, v168 :: v_dual_mul_f32 v109, v109, v169
	v_dual_mul_f32 v116, v116, v168 :: v_dual_mul_f32 v117, v117, v169
	v_dual_mul_f32 v124, v124, v168 :: v_dual_mul_f32 v125, v125, v169
	ds_load_b32 v168, v180 offset:17624
	ds_load_b32 v169, v180 offset:17628
	s_wait_dscnt 0x0
	v_dual_mul_f32 v102, v102, v168 :: v_dual_mul_f32 v103, v103, v169
	v_dual_mul_f32 v110, v110, v168 :: v_dual_mul_f32 v111, v111, v169
	v_dual_mul_f32 v118, v118, v168 :: v_dual_mul_f32 v119, v119, v169
	v_dual_mul_f32 v126, v126, v168 :: v_dual_mul_f32 v127, v127, v169
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s85, s85, 1
	s_add_co_i32 s86, s86, -1
	s_cmp_lg_u32 s86, 0
	s_cbranch_scc1 .Lfp8_ratio_256x128x8_row_silu_loop
	.Lfp8_ratio_256x128x8_row_silu_loop_end:
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s90, s94, s85
	s_lshl_b32 s90, s90, 15
	s_add_co_i32 s90, s90, s95
	s_clause 0x3
	buffer_load_b128 v[128:131], v176, s[32:35], s90 offen
	buffer_load_b128 v[132:135], v176, s[32:35], s90 offen offset:512
	buffer_load_b128 v[136:139], v176, s[32:35], s90 offen offset:1024
	buffer_load_b128 v[140:143], v176, s[32:35], s90 offen offset:1536
	s_clause 0x3
	buffer_load_b128 v[144:147], v176, s[32:35], s90 offen offset:2048
	buffer_load_b128 v[148:151], v176, s[32:35], s90 offen offset:2560
	buffer_load_b128 v[152:155], v176, s[32:35], s90 offen offset:3072
	buffer_load_b128 v[156:159], v176, s[32:35], s90 offen offset:3584
	s_lshl_b32 s91, s85, 7
	s_add_co_i32 s91, s91, 64
	s_clause 0x3
	buffer_load_b64 v[168:169], v177, s[36:39], s91 offen
	buffer_load_b64 v[170:171], v177, s[36:39], s91 offen offset:8
	buffer_load_b64 v[172:173], v177, s[36:39], s91 offen offset:16
	buffer_load_b64 v[174:175], v177, s[36:39], s91 offen offset:24
	ds_load_b64 v[160:161], v179
	ds_load_b64 v[162:163], v179 offset:256
	ds_load_b64 v[164:165], v179 offset:512
	ds_load_b64 v[166:167], v179 offset:768
	s_wait_loadcnt_dscnt 0xb03
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[128:129], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[128:129], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[128:129], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[128:129], v[166:167], v[24:31]
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[132:133], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[132:133], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[132:133], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[132:133], v[166:167], v[56:63]
	s_wait_loadcnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[136:137], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[136:137], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[136:137], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[136:137], v[166:167], v[88:95]
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[140:141], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[140:141], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[140:141], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[140:141], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:2048
	ds_load_b64 v[162:163], v179 offset:2304
	ds_load_b64 v[164:165], v179 offset:2560
	ds_load_b64 v[166:167], v179 offset:2816
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[130:131], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[130:131], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[130:131], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[130:131], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[134:135], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[134:135], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[134:135], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[134:135], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[138:139], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[138:139], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[138:139], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[138:139], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[142:143], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[142:143], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[142:143], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[142:143], v[166:167], v[120:127]
	s_clause 0x3
	buffer_load_b128 v[128:131], v176, s[32:35], s90 offen offset:4096
	buffer_load_b128 v[132:135], v176, s[32:35], s90 offen offset:4608
	buffer_load_b128 v[136:139], v176, s[32:35], s90 offen offset:5120
	buffer_load_b128 v[140:143], v176, s[32:35], s90 offen offset:5632
	ds_load_b64 v[160:161], v179 offset:4096
	ds_load_b64 v[162:163], v179 offset:4352
	ds_load_b64 v[164:165], v179 offset:4608
	ds_load_b64 v[166:167], v179 offset:4864
	s_wait_loadcnt_dscnt 0xb03
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[144:145], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[144:145], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[144:145], v[166:167], v[24:31]
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[148:149], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[148:149], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[148:149], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[148:149], v[166:167], v[56:63]
	s_wait_loadcnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[152:153], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[152:153], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[152:153], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[152:153], v[166:167], v[88:95]
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[156:157], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[156:157], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[156:157], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[156:157], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:6144
	ds_load_b64 v[162:163], v179 offset:6400
	ds_load_b64 v[164:165], v179 offset:6656
	ds_load_b64 v[166:167], v179 offset:6912
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[146:147], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[146:147], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[146:147], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[146:147], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[150:151], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[150:151], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[150:151], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[150:151], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[154:155], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[154:155], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[154:155], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[154:155], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[158:159], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[158:159], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[158:159], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[158:159], v[166:167], v[120:127]
	s_clause 0x3
	buffer_load_b128 v[144:147], v176, s[32:35], s90 offen offset:6144
	buffer_load_b128 v[148:151], v176, s[32:35], s90 offen offset:6656
	buffer_load_b128 v[152:155], v176, s[32:35], s90 offen offset:7168
	buffer_load_b128 v[156:159], v176, s[32:35], s90 offen offset:7680
	s_wait_loadcnt 0xb
	ds_store_b64 v178, v[168:169] offset:8192
	s_wait_loadcnt 0xa
	ds_store_b64 v178, v[170:171] offset:8320
	s_wait_loadcnt 0x9
	ds_store_b64 v178, v[172:173] offset:10240
	s_wait_loadcnt 0x8
	ds_store_b64 v178, v[174:175] offset:10368
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b64 v[160:161], v179 offset:8192
	ds_load_b64 v[162:163], v179 offset:8448
	ds_load_b64 v[164:165], v179 offset:8704
	ds_load_b64 v[166:167], v179 offset:8960
	s_wait_loadcnt_dscnt 0x703
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[128:129], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[128:129], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[128:129], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[128:129], v[166:167], v[24:31]
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[132:133], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[132:133], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[132:133], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[132:133], v[166:167], v[56:63]
	s_wait_loadcnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[136:137], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[136:137], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[136:137], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[136:137], v[166:167], v[88:95]
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[140:141], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[140:141], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[140:141], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[140:141], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:10240
	ds_load_b64 v[162:163], v179 offset:10496
	ds_load_b64 v[164:165], v179 offset:10752
	ds_load_b64 v[166:167], v179 offset:11008
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[130:131], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[130:131], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[130:131], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[130:131], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[134:135], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[134:135], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[134:135], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[134:135], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[138:139], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[138:139], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[138:139], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[138:139], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[142:143], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[142:143], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[142:143], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[142:143], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:12288
	ds_load_b64 v[162:163], v179 offset:12544
	ds_load_b64 v[164:165], v179 offset:12800
	ds_load_b64 v[166:167], v179 offset:13056
	s_wait_loadcnt_dscnt 0x303
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[144:145], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[144:145], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[144:145], v[166:167], v[24:31]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[148:149], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[148:149], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[148:149], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[148:149], v[166:167], v[56:63]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[152:153], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[152:153], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[152:153], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[152:153], v[166:167], v[88:95]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[156:157], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[156:157], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[156:157], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[156:157], v[166:167], v[120:127]
	ds_load_b64 v[160:161], v179 offset:14336
	ds_load_b64 v[162:163], v179 offset:14592
	ds_load_b64 v[164:165], v179 offset:14848
	ds_load_b64 v[166:167], v179 offset:15104
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[146:147], v[160:161], v[0:7]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[146:147], v[162:163], v[8:15]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[146:147], v[164:165], v[16:23]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[146:147], v[166:167], v[24:31]
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[150:151], v[160:161], v[32:39]
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[150:151], v[162:163], v[40:47]
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[150:151], v[164:165], v[48:55]
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[150:151], v[166:167], v[56:63]
	v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[154:155], v[160:161], v[64:71]
	v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[154:155], v[162:163], v[72:79]
	v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[154:155], v[164:165], v[80:87]
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[154:155], v[166:167], v[88:95]
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[158:159], v[160:161], v[96:103]
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[158:159], v[162:163], v[104:111]
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[158:159], v[164:165], v[112:119]
	v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[158:159], v[166:167], v[120:127]
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s85, s85, 1
	s_branch .Lfp8_ratio_256x128x8_row_silu_epilogue
	.Lfp8_ratio_256x128x8_row_silu_epilogue:
	s_lshl_b32 s72, s83, 10
	buffer_load_b32 v128, v180, s[44:47], s72 offen
	buffer_load_b32 v129, v180, s[44:47], s72 offen offset:4
	buffer_load_b32 v130, v180, s[44:47], s72 offen offset:8
	buffer_load_b32 v131, v180, s[44:47], s72 offen offset:12
	buffer_load_b32 v132, v180, s[44:47], s72 offen offset:16
	buffer_load_b32 v133, v180, s[44:47], s72 offen offset:20
	buffer_load_b32 v134, v180, s[44:47], s72 offen offset:24
	buffer_load_b32 v135, v180, s[44:47], s72 offen offset:28
	buffer_load_b32 v136, v180, s[44:47], s72 offen offset:64
	buffer_load_b32 v137, v180, s[44:47], s72 offen offset:68
	buffer_load_b32 v138, v180, s[44:47], s72 offen offset:72
	buffer_load_b32 v139, v180, s[44:47], s72 offen offset:76
	buffer_load_b32 v140, v180, s[44:47], s72 offen offset:80
	buffer_load_b32 v141, v180, s[44:47], s72 offen offset:84
	buffer_load_b32 v142, v180, s[44:47], s72 offen offset:88
	buffer_load_b32 v143, v180, s[44:47], s72 offen offset:92
	buffer_load_b32 v144, v180, s[44:47], s72 offen offset:128
	buffer_load_b32 v145, v180, s[44:47], s72 offen offset:132
	buffer_load_b32 v146, v180, s[44:47], s72 offen offset:136
	buffer_load_b32 v147, v180, s[44:47], s72 offen offset:140
	buffer_load_b32 v148, v180, s[44:47], s72 offen offset:144
	buffer_load_b32 v149, v180, s[44:47], s72 offen offset:148
	buffer_load_b32 v150, v180, s[44:47], s72 offen offset:152
	buffer_load_b32 v151, v180, s[44:47], s72 offen offset:156
	buffer_load_b32 v152, v180, s[44:47], s72 offen offset:192
	buffer_load_b32 v153, v180, s[44:47], s72 offen offset:196
	buffer_load_b32 v154, v180, s[44:47], s72 offen offset:200
	buffer_load_b32 v155, v180, s[44:47], s72 offen offset:204
	buffer_load_b32 v156, v180, s[44:47], s72 offen offset:208
	buffer_load_b32 v157, v180, s[44:47], s72 offen offset:212
	buffer_load_b32 v158, v180, s[44:47], s72 offen offset:216
	buffer_load_b32 v159, v180, s[44:47], s72 offen offset:220
	s_lshl_b32 s72, s82, 2
	v_mov_b32_e32 v190, v181
	buffer_load_b32 v160, v190, s[48:51], s72 offen
	v_add_nc_u32_e32 v190, 64, v181
	buffer_load_b32 v161, v190, s[48:51], s72 offen
	v_add_nc_u32_e32 v190, 0x80, v181
	buffer_load_b32 v162, v190, s[48:51], s72 offen
	v_add_nc_u32_e32 v190, 0xc0, v181
	buffer_load_b32 v163, v190, s[48:51], s72 offen
	s_wait_loadcnt 0x23
	v_mul_f32_e32 v0, v0, v128
	s_wait_loadcnt 0x3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v0, v0, v160
	v_mul_f32_e32 v1, v1, v129
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v1, v1, v160
	v_mul_f32_e32 v2, v2, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v2, v2, v160
	v_mul_f32_e32 v3, v3, v131
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v3, v3, v160
	v_mul_f32_e32 v4, v4, v132
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, v4, v160
	v_mul_f32_e32 v5, v5, v133
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v5, v5, v160
	v_mul_f32_e32 v6, v6, v134
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v6, v6, v160
	v_mul_f32_e32 v7, v7, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v7, v7, v160
	v_mul_f32_e32 v8, v8, v128
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, v8, v161
	v_mul_f32_e32 v9, v9, v129
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, v9, v161
	v_mul_f32_e32 v10, v10, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v10, v10, v161
	v_mul_f32_e32 v11, v11, v131
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v11, v11, v161
	v_mul_f32_e32 v12, v12, v132
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v12, v12, v161
	v_mul_f32_e32 v13, v13, v133
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v13, v13, v161
	v_mul_f32_e32 v14, v14, v134
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v14, v14, v161
	v_mul_f32_e32 v15, v15, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v15, v15, v161
	v_mul_f32_e32 v16, v16, v128
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, v16, v162
	v_mul_f32_e32 v17, v17, v129
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, v17, v162
	v_mul_f32_e32 v18, v18, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v18, v18, v162
	v_mul_f32_e32 v19, v19, v131
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v19, v19, v162
	v_mul_f32_e32 v20, v20, v132
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v20, v20, v162
	v_mul_f32_e32 v21, v21, v133
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, v21, v162
	v_mul_f32_e32 v22, v22, v134
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, v22, v162
	v_mul_f32_e32 v23, v23, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, v23, v162
	v_mul_f32_e32 v24, v24, v128
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, v24, v163
	v_mul_f32_e32 v25, v25, v129
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, v25, v163
	v_mul_f32_e32 v26, v26, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, v26, v163
	v_mul_f32_e32 v27, v27, v131
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v27, v27, v163
	v_mul_f32_e32 v28, v28, v132
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v28, v163
	v_mul_f32_e32 v29, v29, v133
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v29, v163
	v_mul_f32_e32 v30, v30, v134
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v30, v163
	v_mul_f32_e32 v31, v31, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v31, v163
	v_mul_f32_e32 v32, v32, v136
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v32, v32, v160
	v_mul_f32_e32 v33, v33, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v33, v33, v160
	v_mul_f32_e32 v34, v34, v138
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v34, v34, v160
	v_mul_f32_e32 v35, v35, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v35, v35, v160
	v_mul_f32_e32 v36, v36, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v36, v36, v160
	v_mul_f32_e32 v37, v37, v141
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v37, v37, v160
	v_mul_f32_e32 v38, v38, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v38, v38, v160
	v_mul_f32_e32 v39, v39, v143
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v39, v39, v160
	v_mul_f32_e32 v40, v40, v136
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v40, v40, v161
	v_mul_f32_e32 v41, v41, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v41, v41, v161
	v_mul_f32_e32 v42, v42, v138
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v42, v42, v161
	v_mul_f32_e32 v43, v43, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v43, v43, v161
	v_mul_f32_e32 v44, v44, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v44, v44, v161
	v_mul_f32_e32 v45, v45, v141
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v45, v45, v161
	v_mul_f32_e32 v46, v46, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v46, v46, v161
	v_mul_f32_e32 v47, v47, v143
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v47, v47, v161
	v_mul_f32_e32 v48, v48, v136
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v48, v48, v162
	v_mul_f32_e32 v49, v49, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v49, v49, v162
	v_mul_f32_e32 v50, v50, v138
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v50, v50, v162
	v_mul_f32_e32 v51, v51, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v51, v51, v162
	v_mul_f32_e32 v52, v52, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v52, v52, v162
	v_mul_f32_e32 v53, v53, v141
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v53, v53, v162
	v_mul_f32_e32 v54, v54, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v54, v54, v162
	v_mul_f32_e32 v55, v55, v143
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v55, v55, v162
	v_mul_f32_e32 v56, v56, v136
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v56, v56, v163
	v_mul_f32_e32 v57, v57, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v57, v57, v163
	v_mul_f32_e32 v58, v58, v138
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v58, v58, v163
	v_mul_f32_e32 v59, v59, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v59, v59, v163
	v_mul_f32_e32 v60, v60, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v60, v60, v163
	v_mul_f32_e32 v61, v61, v141
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v61, v61, v163
	v_mul_f32_e32 v62, v62, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v62, v62, v163
	v_mul_f32_e32 v63, v63, v143
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v63, v63, v163
	v_mul_f32_e32 v64, v64, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v64, v64, v160
	v_mul_f32_e32 v65, v65, v145
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v65, v65, v160
	v_mul_f32_e32 v66, v66, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v66, v66, v160
	v_mul_f32_e32 v67, v67, v147
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v67, v67, v160
	v_mul_f32_e32 v68, v68, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v68, v68, v160
	v_mul_f32_e32 v69, v69, v149
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v69, v69, v160
	v_mul_f32_e32 v70, v70, v150
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v70, v70, v160
	v_mul_f32_e32 v71, v71, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v71, v71, v160
	v_mul_f32_e32 v72, v72, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v72, v72, v161
	v_mul_f32_e32 v73, v73, v145
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v73, v73, v161
	v_mul_f32_e32 v74, v74, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v74, v74, v161
	v_mul_f32_e32 v75, v75, v147
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v75, v75, v161
	v_mul_f32_e32 v76, v76, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v76, v76, v161
	v_mul_f32_e32 v77, v77, v149
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v77, v77, v161
	v_mul_f32_e32 v78, v78, v150
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v78, v78, v161
	v_mul_f32_e32 v79, v79, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v79, v79, v161
	v_mul_f32_e32 v80, v80, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v80, v80, v162
	v_mul_f32_e32 v81, v81, v145
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v81, v81, v162
	v_mul_f32_e32 v82, v82, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v82, v82, v162
	v_mul_f32_e32 v83, v83, v147
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v83, v83, v162
	v_mul_f32_e32 v84, v84, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v84, v84, v162
	v_mul_f32_e32 v85, v85, v149
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v85, v85, v162
	v_mul_f32_e32 v86, v86, v150
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v86, v86, v162
	v_mul_f32_e32 v87, v87, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v87, v87, v162
	v_mul_f32_e32 v88, v88, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v88, v88, v163
	v_mul_f32_e32 v89, v89, v145
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v89, v89, v163
	v_mul_f32_e32 v90, v90, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v90, v90, v163
	v_mul_f32_e32 v91, v91, v147
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v91, v91, v163
	v_mul_f32_e32 v92, v92, v148
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v92, v92, v163
	v_mul_f32_e32 v93, v93, v149
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v93, v93, v163
	v_mul_f32_e32 v94, v94, v150
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v94, v94, v163
	v_mul_f32_e32 v95, v95, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v95, v95, v163
	v_mul_f32_e32 v96, v96, v152
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v96, v96, v160
	v_mul_f32_e32 v97, v97, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v97, v97, v160
	v_mul_f32_e32 v98, v98, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v98, v98, v160
	v_mul_f32_e32 v99, v99, v155
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v99, v99, v160
	v_mul_f32_e32 v100, v100, v156
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v100, v100, v160
	v_mul_f32_e32 v101, v101, v157
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v101, v101, v160
	v_mul_f32_e32 v102, v102, v158
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v102, v102, v160
	v_mul_f32_e32 v103, v103, v159
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v103, v103, v160
	v_mul_f32_e32 v104, v104, v152
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v104, v104, v161
	v_mul_f32_e32 v105, v105, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v105, v105, v161
	v_mul_f32_e32 v106, v106, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v106, v106, v161
	v_mul_f32_e32 v107, v107, v155
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v107, v107, v161
	v_mul_f32_e32 v108, v108, v156
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v108, v108, v161
	v_mul_f32_e32 v109, v109, v157
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v109, v109, v161
	v_mul_f32_e32 v110, v110, v158
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v110, v110, v161
	v_mul_f32_e32 v111, v111, v159
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v111, v111, v161
	v_mul_f32_e32 v112, v112, v152
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v112, v112, v162
	v_mul_f32_e32 v113, v113, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v113, v113, v162
	v_mul_f32_e32 v114, v114, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v114, v114, v162
	v_mul_f32_e32 v115, v115, v155
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v115, v115, v162
	v_mul_f32_e32 v116, v116, v156
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v116, v116, v162
	v_mul_f32_e32 v117, v117, v157
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v117, v117, v162
	v_mul_f32_e32 v118, v118, v158
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v118, v118, v162
	v_mul_f32_e32 v119, v119, v159
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v119, v119, v162
	v_mul_f32_e32 v120, v120, v152
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v120, v120, v163
	v_mul_f32_e32 v121, v121, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v121, v121, v163
	v_mul_f32_e32 v122, v122, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v122, v122, v163
	v_mul_f32_e32 v123, v123, v155
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v123, v123, v163
	v_mul_f32_e32 v124, v124, v156
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v124, v124, v163
	v_mul_f32_e32 v125, v125, v157
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v125, v125, v163
	v_mul_f32_e32 v126, v126, v158
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v126, v126, v163
	v_mul_f32_e32 v127, v127, v159
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v127, v127, v163
	s_lshl_b32 s72, s83, 8
	s_lshl_b32 s73, s88, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v184, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v184, 4, v184
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v184, 3, v184
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v184, s72, v184
	v_and_b32_e32 v183, 15, v187
	s_lshl_b32 s73, s89, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s73, s82
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v183, s73, v183
	v_dual_mul_f32 v128, 0xbfb8aa3b, v0 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v1
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v0
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v0
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v1
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v1
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v0, -v128
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v137, 0xbfb8aa3b, v1, -v135
	v_dual_mul_f32 v142, 0xbfb8aa3b, v2 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v3
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v2
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v2
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	v_dual_fmac_f32 v130, 0xb2a5705f, v0 :: v_dual_fmac_f32 v137, 0xb2a5705f, v1
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v2, -v142
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v3
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v3
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v3, -v149
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v2 :: v_dual_fmac_f32 v151, 0xb2a5705f, v3
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_ldexp_f32 v129, v131, v129
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	v_ldexp_f32 v143, v145, v143
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v131, null, v129, v129, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v138, null, v136, v136, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v130, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v132, -v131, v130, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v0, v129, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v133, v132, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v134, -v131, v133, v132
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v1, v136, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v129, v131, v129, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v2, v143, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v0, v32, v129 :: v_dual_mul_f32 v1, v33, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	v_dual_mul_f32 v128, 0xbfb8aa3b, v4 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v5
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v4
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v4
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v5
	v_fmac_f32_e32 v147, v148, v144
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v4, -v128
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v5
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	v_fma_f32 v137, 0xbfb8aa3b, v5, -v135
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v3, v150, v3
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v130, 0xb2a5705f, v4 :: v_dual_fmac_f32 v137, 0xb2a5705f, v5
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_div_fixup_f32 v143, v145, v143, v2
	v_mul_f32_e32 v154, v153, v151
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v131, v131
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2)
	v_ldexp_f32 v129, v131, v129
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v152, v152, v151, v154
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v150, v152, v150, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v2, v34, v143 :: v_dual_mul_f32 v3, v35, v150
	v_dual_mul_f32 v142, 0xbfb8aa3b, v6 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v7
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v6
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v7
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v6, -v142
	v_div_scale_f32 v131, null, v129, v129, v4
	v_div_scale_f32 v138, null, v136, v136, v5
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v7
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v7, -v149
	v_rcp_f32_e32 v130, v131
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v6 :: v_dual_fmac_f32 v151, 0xb2a5705f, v7
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_fma_f32 v132, -v131, v130, 1.0
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v4, v129, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v133, v132, v130
	v_ldexp_f32 v143, v145, v143
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v134, -v131, v133, v132
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v5, v136, v5
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v129, v131, v129, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v6, v143, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v5
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v4, v36, v129 :: v_dual_mul_f32 v5, v37, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v7, v150, v7
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v143, v145, v143, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v154, v153, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v152, v152, v151, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v150, v152, v150, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v6, v38, v143 :: v_dual_mul_f32 v7, v39, v150
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 0, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 0, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[0:3], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 4, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 0, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[4:7], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	v_dual_mul_f32 v128, 0xbfb8aa3b, v8 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v9
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v8
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v8
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v9
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v9
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v8, -v128
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v137, 0xbfb8aa3b, v9, -v135
	v_dual_mul_f32 v142, 0xbfb8aa3b, v10 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v11
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v10
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v10
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	v_dual_fmac_f32 v130, 0xb2a5705f, v8 :: v_dual_fmac_f32 v137, 0xb2a5705f, v9
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v10, -v142
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v11
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v11
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v11, -v149
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v10 :: v_dual_fmac_f32 v151, 0xb2a5705f, v11
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_ldexp_f32 v129, v131, v129
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	v_ldexp_f32 v143, v145, v143
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v131, null, v129, v129, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v138, null, v136, v136, v9
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v130, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v132, -v131, v130, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v11
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v8, v129, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v133, v132, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v134, -v131, v133, v132
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v9, v136, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v129, v131, v129, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v10, v143, v10
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v8, v40, v129 :: v_dual_mul_f32 v9, v41, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	v_dual_mul_f32 v128, 0xbfb8aa3b, v12 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v13
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v12
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v12
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v13
	v_fmac_f32_e32 v147, v148, v144
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v12, -v128
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v13
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	v_fma_f32 v137, 0xbfb8aa3b, v13, -v135
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v11, v150, v11
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v130, 0xb2a5705f, v12 :: v_dual_fmac_f32 v137, 0xb2a5705f, v13
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_div_fixup_f32 v143, v145, v143, v10
	v_mul_f32_e32 v154, v153, v151
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v131, v131
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2)
	v_ldexp_f32 v129, v131, v129
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v152, v152, v151, v154
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v150, v152, v150, v11
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v10, v42, v143 :: v_dual_mul_f32 v11, v43, v150
	v_dual_mul_f32 v142, 0xbfb8aa3b, v14 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v15
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v14
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v15
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v14, -v142
	v_div_scale_f32 v131, null, v129, v129, v12
	v_div_scale_f32 v138, null, v136, v136, v13
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v15
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v15, -v149
	v_rcp_f32_e32 v130, v131
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v14 :: v_dual_fmac_f32 v151, 0xb2a5705f, v15
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_fma_f32 v132, -v131, v130, 1.0
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v12, v129, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v133, v132, v130
	v_ldexp_f32 v143, v145, v143
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v134, -v131, v133, v132
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v13, v136, v13
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v14
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v15
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v129, v131, v129, v12
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v14, v143, v14
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v13
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v12, v44, v129 :: v_dual_mul_f32 v13, v45, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v15, v150, v15
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v143, v145, v143, v14
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v154, v153, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v152, v152, v151, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v150, v152, v150, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v14, v46, v143 :: v_dual_mul_f32 v15, v47, v150
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 0, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 16, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[8:11], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 4, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 16, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[12:15], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	v_dual_mul_f32 v128, 0xbfb8aa3b, v16 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v17
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v16
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v16
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v17
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v17
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v16, -v128
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v137, 0xbfb8aa3b, v17, -v135
	v_dual_mul_f32 v142, 0xbfb8aa3b, v18 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v19
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v18
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v18
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	v_dual_fmac_f32 v130, 0xb2a5705f, v16 :: v_dual_fmac_f32 v137, 0xb2a5705f, v17
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v18, -v142
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v19
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v19
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v19, -v149
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v18 :: v_dual_fmac_f32 v151, 0xb2a5705f, v19
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_ldexp_f32 v129, v131, v129
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	v_ldexp_f32 v143, v145, v143
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v131, null, v129, v129, v16
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v138, null, v136, v136, v17
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v130, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v132, -v131, v130, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v18
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v19
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v16, v129, v16
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v133, v132, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v134, -v131, v133, v132
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v17, v136, v17
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v129, v131, v129, v16
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v18, v143, v18
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v17
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v16, v48, v129 :: v_dual_mul_f32 v17, v49, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	v_dual_mul_f32 v128, 0xbfb8aa3b, v20 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v21
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v20
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v20
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v21
	v_fmac_f32_e32 v147, v148, v144
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v20, -v128
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v21
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	v_fma_f32 v137, 0xbfb8aa3b, v21, -v135
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v19, v150, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v130, 0xb2a5705f, v20 :: v_dual_fmac_f32 v137, 0xb2a5705f, v21
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_div_fixup_f32 v143, v145, v143, v18
	v_mul_f32_e32 v154, v153, v151
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v131, v131
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2)
	v_ldexp_f32 v129, v131, v129
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v152, v152, v151, v154
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v150, v152, v150, v19
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v18, v50, v143 :: v_dual_mul_f32 v19, v51, v150
	v_dual_mul_f32 v142, 0xbfb8aa3b, v22 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v23
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v22
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v23
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v22, -v142
	v_div_scale_f32 v131, null, v129, v129, v20
	v_div_scale_f32 v138, null, v136, v136, v21
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v23
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v23, -v149
	v_rcp_f32_e32 v130, v131
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v22 :: v_dual_fmac_f32 v151, 0xb2a5705f, v23
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_fma_f32 v132, -v131, v130, 1.0
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v20, v129, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v133, v132, v130
	v_ldexp_f32 v143, v145, v143
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v134, -v131, v133, v132
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v21, v136, v21
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v23
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v129, v131, v129, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v22, v143, v22
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v21
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v20, v52, v129 :: v_dual_mul_f32 v21, v53, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v23, v150, v23
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v143, v145, v143, v22
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v154, v153, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v152, v152, v151, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v150, v152, v150, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v22, v54, v143 :: v_dual_mul_f32 v23, v55, v150
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 0, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 32, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[16:19], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 4, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 32, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[20:23], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	v_dual_mul_f32 v128, 0xbfb8aa3b, v24 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v25
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v24
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v24
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v25
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v25
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v24, -v128
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v137, 0xbfb8aa3b, v25, -v135
	v_dual_mul_f32 v142, 0xbfb8aa3b, v26 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v27
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v26
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v26
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	v_dual_fmac_f32 v130, 0xb2a5705f, v24 :: v_dual_fmac_f32 v137, 0xb2a5705f, v25
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v26, -v142
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v27
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v27
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v27, -v149
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v26 :: v_dual_fmac_f32 v151, 0xb2a5705f, v27
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_ldexp_f32 v129, v131, v129
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	v_ldexp_f32 v143, v145, v143
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v131, null, v129, v129, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v138, null, v136, v136, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v130, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v132, -v131, v130, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v27
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v24, v129, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v133, v132, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v134, -v131, v133, v132
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v25, v136, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v129, v131, v129, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v26, v143, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v24, v56, v129 :: v_dual_mul_f32 v25, v57, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	v_dual_mul_f32 v128, 0xbfb8aa3b, v28 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v29
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v28
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v28
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v29
	v_fmac_f32_e32 v147, v148, v144
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v28, -v128
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v29
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	v_fma_f32 v137, 0xbfb8aa3b, v29, -v135
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v27, v150, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v130, 0xb2a5705f, v28 :: v_dual_fmac_f32 v137, 0xb2a5705f, v29
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_div_fixup_f32 v143, v145, v143, v26
	v_mul_f32_e32 v154, v153, v151
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v131, v131
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2)
	v_ldexp_f32 v129, v131, v129
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v152, v152, v151, v154
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v150, v152, v150, v27
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v26, v58, v143 :: v_dual_mul_f32 v27, v59, v150
	v_dual_mul_f32 v142, 0xbfb8aa3b, v30 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v31
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v30
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v31
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v30, -v142
	v_div_scale_f32 v131, null, v129, v129, v28
	v_div_scale_f32 v138, null, v136, v136, v29
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v31
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v31, -v149
	v_rcp_f32_e32 v130, v131
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v30 :: v_dual_fmac_f32 v151, 0xb2a5705f, v31
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_fma_f32 v132, -v131, v130, 1.0
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v28, v129, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v133, v132, v130
	v_ldexp_f32 v143, v145, v143
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v134, -v131, v133, v132
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v29, v136, v29
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v129, v131, v129, v28
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v30, v143, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v29
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v28, v60, v129 :: v_dual_mul_f32 v29, v61, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v31, v150, v31
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v143, v145, v143, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v154, v153, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v152, v152, v151, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v150, v152, v150, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v30, v62, v143 :: v_dual_mul_f32 v31, v63, v150
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 0, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 48, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[24:27], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 4, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 48, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[28:31], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	v_dual_mul_f32 v128, 0xbfb8aa3b, v64 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v65
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v64
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v64
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v65
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v65
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v64, -v128
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v137, 0xbfb8aa3b, v65, -v135
	v_dual_mul_f32 v142, 0xbfb8aa3b, v66 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v67
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v66
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v66
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	v_dual_fmac_f32 v130, 0xb2a5705f, v64 :: v_dual_fmac_f32 v137, 0xb2a5705f, v65
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v66, -v142
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v67
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v67
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v67, -v149
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v66 :: v_dual_fmac_f32 v151, 0xb2a5705f, v67
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_ldexp_f32 v129, v131, v129
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	v_ldexp_f32 v143, v145, v143
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v131, null, v129, v129, v64
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v138, null, v136, v136, v65
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v130, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v132, -v131, v130, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v66
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v67
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v64, v129, v64
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v133, v132, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v134, -v131, v133, v132
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v65, v136, v65
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v129, v131, v129, v64
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v66, v143, v66
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v65
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v64, v96, v129 :: v_dual_mul_f32 v65, v97, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	v_dual_mul_f32 v128, 0xbfb8aa3b, v68 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v69
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v68
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v68
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v69
	v_fmac_f32_e32 v147, v148, v144
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v68, -v128
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v69
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	v_fma_f32 v137, 0xbfb8aa3b, v69, -v135
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v67, v150, v67
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v130, 0xb2a5705f, v68 :: v_dual_fmac_f32 v137, 0xb2a5705f, v69
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_div_fixup_f32 v143, v145, v143, v66
	v_mul_f32_e32 v154, v153, v151
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v131, v131
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2)
	v_ldexp_f32 v129, v131, v129
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v152, v152, v151, v154
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v150, v152, v150, v67
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v66, v98, v143 :: v_dual_mul_f32 v67, v99, v150
	v_dual_mul_f32 v142, 0xbfb8aa3b, v70 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v71
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v70
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v70
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v71
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v70, -v142
	v_div_scale_f32 v131, null, v129, v129, v68
	v_div_scale_f32 v138, null, v136, v136, v69
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v71
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v71, -v149
	v_rcp_f32_e32 v130, v131
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v70 :: v_dual_fmac_f32 v151, 0xb2a5705f, v71
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_fma_f32 v132, -v131, v130, 1.0
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v68, v129, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v133, v132, v130
	v_ldexp_f32 v143, v145, v143
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v134, -v131, v133, v132
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v69, v136, v69
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v70
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v71
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v129, v131, v129, v68
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v70, v143, v70
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v69
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v68, v100, v129 :: v_dual_mul_f32 v69, v101, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v71, v150, v71
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v143, v145, v143, v70
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v154, v153, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v152, v152, v151, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v150, v152, v150, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v70, v102, v143 :: v_dual_mul_f32 v71, v103, v150
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 16, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 0, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[64:67], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 20, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 0, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[68:71], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	v_dual_mul_f32 v128, 0xbfb8aa3b, v72 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v73
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v72
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v72
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v73
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v73
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v72, -v128
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v137, 0xbfb8aa3b, v73, -v135
	v_dual_mul_f32 v142, 0xbfb8aa3b, v74 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v75
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v74
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v74
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	v_dual_fmac_f32 v130, 0xb2a5705f, v72 :: v_dual_fmac_f32 v137, 0xb2a5705f, v73
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v74, -v142
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v75
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v75
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v75, -v149
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v74 :: v_dual_fmac_f32 v151, 0xb2a5705f, v75
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_ldexp_f32 v129, v131, v129
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	v_ldexp_f32 v143, v145, v143
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v131, null, v129, v129, v72
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v138, null, v136, v136, v73
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v130, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v132, -v131, v130, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v74
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v75
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v72, v129, v72
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v133, v132, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v134, -v131, v133, v132
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v73, v136, v73
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v129, v131, v129, v72
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v74, v143, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v73
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v72, v104, v129 :: v_dual_mul_f32 v73, v105, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	v_dual_mul_f32 v128, 0xbfb8aa3b, v76 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v77
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v76
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v76
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v77
	v_fmac_f32_e32 v147, v148, v144
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v76, -v128
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v77
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	v_fma_f32 v137, 0xbfb8aa3b, v77, -v135
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v75, v150, v75
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v130, 0xb2a5705f, v76 :: v_dual_fmac_f32 v137, 0xb2a5705f, v77
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_div_fixup_f32 v143, v145, v143, v74
	v_mul_f32_e32 v154, v153, v151
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v131, v131
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2)
	v_ldexp_f32 v129, v131, v129
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v152, v152, v151, v154
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v150, v152, v150, v75
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v74, v106, v143 :: v_dual_mul_f32 v75, v107, v150
	v_dual_mul_f32 v142, 0xbfb8aa3b, v78 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v79
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v78
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v78
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v79
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v78, -v142
	v_div_scale_f32 v131, null, v129, v129, v76
	v_div_scale_f32 v138, null, v136, v136, v77
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v79
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v79, -v149
	v_rcp_f32_e32 v130, v131
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v78 :: v_dual_fmac_f32 v151, 0xb2a5705f, v79
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_fma_f32 v132, -v131, v130, 1.0
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v76, v129, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v133, v132, v130
	v_ldexp_f32 v143, v145, v143
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v134, -v131, v133, v132
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v77, v136, v77
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v78
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v79
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v129, v131, v129, v76
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v78, v143, v78
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v77
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v76, v108, v129 :: v_dual_mul_f32 v77, v109, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v79, v150, v79
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v143, v145, v143, v78
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v154, v153, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v152, v152, v151, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v150, v152, v150, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v78, v110, v143 :: v_dual_mul_f32 v79, v111, v150
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 16, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 16, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[72:75], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 20, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 16, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[76:79], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	v_dual_mul_f32 v128, 0xbfb8aa3b, v80 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v81
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v80
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v80
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v81
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v81
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v80, -v128
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v137, 0xbfb8aa3b, v81, -v135
	v_dual_mul_f32 v142, 0xbfb8aa3b, v82 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v83
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v82
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v82
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	v_dual_fmac_f32 v130, 0xb2a5705f, v80 :: v_dual_fmac_f32 v137, 0xb2a5705f, v81
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v82, -v142
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v83
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v83
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v83, -v149
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v82 :: v_dual_fmac_f32 v151, 0xb2a5705f, v83
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_ldexp_f32 v129, v131, v129
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	v_ldexp_f32 v143, v145, v143
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v131, null, v129, v129, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v138, null, v136, v136, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v130, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v132, -v131, v130, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v83
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v80, v129, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v133, v132, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v134, -v131, v133, v132
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v81, v136, v81
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v129, v131, v129, v80
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v82, v143, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v81
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v80, v112, v129 :: v_dual_mul_f32 v81, v113, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	v_dual_mul_f32 v128, 0xbfb8aa3b, v84 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v85
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v84
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v84
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v85
	v_fmac_f32_e32 v147, v148, v144
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v84, -v128
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v85
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	v_fma_f32 v137, 0xbfb8aa3b, v85, -v135
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v83, v150, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v130, 0xb2a5705f, v84 :: v_dual_fmac_f32 v137, 0xb2a5705f, v85
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_div_fixup_f32 v143, v145, v143, v82
	v_mul_f32_e32 v154, v153, v151
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v131, v131
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2)
	v_ldexp_f32 v129, v131, v129
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v152, v152, v151, v154
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v150, v152, v150, v83
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v82, v114, v143 :: v_dual_mul_f32 v83, v115, v150
	v_dual_mul_f32 v142, 0xbfb8aa3b, v86 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v87
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v86
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v86
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v87
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v86, -v142
	v_div_scale_f32 v131, null, v129, v129, v84
	v_div_scale_f32 v138, null, v136, v136, v85
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v87
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v87, -v149
	v_rcp_f32_e32 v130, v131
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v86 :: v_dual_fmac_f32 v151, 0xb2a5705f, v87
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_fma_f32 v132, -v131, v130, 1.0
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v84, v129, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v133, v132, v130
	v_ldexp_f32 v143, v145, v143
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v134, -v131, v133, v132
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v85, v136, v85
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v86
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v87
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v129, v131, v129, v84
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v86, v143, v86
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v85
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v84, v116, v129 :: v_dual_mul_f32 v85, v117, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v87, v150, v87
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v143, v145, v143, v86
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v154, v153, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v152, v152, v151, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v150, v152, v150, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v86, v118, v143 :: v_dual_mul_f32 v87, v119, v150
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 16, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 32, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[80:83], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 20, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 32, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[84:87], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	v_dual_mul_f32 v128, 0xbfb8aa3b, v88 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v89
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v88
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v88
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v89
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v89
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v88, -v128
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v137, 0xbfb8aa3b, v89, -v135
	v_dual_mul_f32 v142, 0xbfb8aa3b, v90 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v91
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v90
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v90
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	v_dual_fmac_f32 v130, 0xb2a5705f, v88 :: v_dual_fmac_f32 v137, 0xb2a5705f, v89
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v90, -v142
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v91
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v91
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v91, -v149
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v90 :: v_dual_fmac_f32 v151, 0xb2a5705f, v91
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_ldexp_f32 v129, v131, v129
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	v_ldexp_f32 v143, v145, v143
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v131, null, v129, v129, v88
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v138, null, v136, v136, v89
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v130, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v132, -v131, v130, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v90
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v91
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v88, v129, v88
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v133, v132, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v134, -v131, v133, v132
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v89, v136, v89
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v129, v131, v129, v88
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v90, v143, v90
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v89
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v88, v120, v129 :: v_dual_mul_f32 v89, v121, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	v_dual_mul_f32 v128, 0xbfb8aa3b, v92 :: v_dual_mul_f32 v135, 0xbfb8aa3b, v93
	v_cmp_nlt_f32_e64 s80, 0x42ce8ed0, v92
	v_cmp_ngt_f32_e64 s81, 0xc2b17218, v92
	v_cmp_nlt_f32_e64 s84, 0x42ce8ed0, v93
	v_fmac_f32_e32 v147, v148, v144
	v_rndne_f32_e32 v129, v128
	v_fma_f32 v130, 0xbfb8aa3b, v92, -v128
	v_cmp_ngt_f32_e64 s85, 0xc2b17218, v93
	v_rndne_f32_e32 v136, v135
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	v_fma_f32 v137, 0xbfb8aa3b, v93, -v135
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_sub_f32 v131, v128, v129 :: v_dual_sub_f32 v138, v135, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v91, v150, v91
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v130, 0xb2a5705f, v92 :: v_dual_fmac_f32 v137, 0xb2a5705f, v93
	v_cvt_i32_f32_e32 v129, v129
	v_cvt_i32_f32_e32 v136, v136
	v_div_fixup_f32 v143, v145, v143, v90
	v_mul_f32_e32 v154, v153, v151
	v_dual_add_f32 v131, v131, v130 :: v_dual_add_f32 v138, v138, v137
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v131, v131
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2)
	v_ldexp_f32 v129, v131, v129
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v136, v138, v136
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fmas_f32 v152, v152, v151, v154
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0, v129, s80
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0, v136, s84
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v150, v152, v150, v91
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v129, 0x7f800000, v129, s81
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136, 0x7f800000, v136, s85
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v90, v122, v143 :: v_dual_mul_f32 v91, v123, v150
	v_dual_mul_f32 v142, 0xbfb8aa3b, v94 :: v_dual_mul_f32 v149, 0xbfb8aa3b, v95
	v_cmp_nlt_f32_e64 s90, 0x42ce8ed0, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v129, 1.0, v129 :: v_dual_add_f32 v136, 1.0, v136
	v_cmp_ngt_f32_e64 s91, 0xc2b17218, v94
	v_cmp_nlt_f32_e64 s92, 0x42ce8ed0, v95
	v_rndne_f32_e32 v143, v142
	v_fma_f32 v144, 0xbfb8aa3b, v94, -v142
	v_div_scale_f32 v131, null, v129, v129, v92
	v_div_scale_f32 v138, null, v136, v136, v93
	v_cmp_ngt_f32_e64 s93, 0xc2b17218, v95
	v_rndne_f32_e32 v150, v149
	v_fma_f32 v151, 0xbfb8aa3b, v95, -v149
	v_rcp_f32_e32 v130, v131
	v_rcp_f32_e32 v137, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v145, v142, v143 :: v_dual_sub_f32 v152, v149, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v144, 0xb2a5705f, v94 :: v_dual_fmac_f32 v151, 0xb2a5705f, v95
	v_cvt_i32_f32_e32 v143, v143
	v_cvt_i32_f32_e32 v150, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v144 :: v_dual_add_f32 v152, v152, v151
	v_fma_f32 v132, -v131, v130, 1.0
	v_fma_f32 v139, -v138, v137, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v152, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v130, v132, v130 :: v_dual_fmac_f32 v137, v139, v137
	v_div_scale_f32 v132, vcc_lo, v92, v129, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v133, v132, v130
	v_ldexp_f32 v143, v145, v143
	v_ldexp_f32 v150, v152, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v134, -v131, v133, v132
	s_wait_alu depctr_va_sdst(0)
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0, v143, s90
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0, v150, s92
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v133, v134, v130
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v143, 0x7f800000, v143, s91
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v150, 0x7f800000, v150, s93
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v131, -v131, v133, v132
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v143, 1.0, v143 :: v_dual_add_f32 v150, 1.0, v150
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v131, v131, v130, v133
	v_div_scale_f32 v139, vcc_lo, v93, v136, v93
	s_delay_alu instid0(VALU_DEP_3)
	v_div_scale_f32 v145, null, v143, v143, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_div_scale_f32 v152, null, v150, v150, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v129, v131, v129, v92
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v140, v139, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v144, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_rcp_f32_e32 v151, v152
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v141, -v138, v140, v139
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v137
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v146, -v145, v144, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v153, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v138, -v138, v140, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v144, v146, v144 :: v_dual_fmac_f32 v151, v153, v151
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v138, v138, v137, v140
	v_div_scale_f32 v146, vcc_lo, v94, v143, v94
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v136, v138, v136, v93
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v147, v146, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v92, v124, v129 :: v_dual_mul_f32 v93, v125, v136
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v148, -v145, v147, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v145, -v145, v147, v146
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v145, v145, v144, v147
	v_div_scale_f32 v153, vcc_lo, v95, v150, v95
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v143, v145, v143, v94
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v154, v153, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v155, -v152, v154, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v152, -v152, v154, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v152, v152, v151, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v150, v152, v150, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v94, v126, v143 :: v_dual_mul_f32 v95, v127, v150
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 16, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 48, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[88:91], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	s_lshl_b32 s72, s83, 7
	s_lshl_b32 s73, s88, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s72, s72, s73
	v_and_b32_e32 v188, 31, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v188, 4, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v188, 3, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, 20, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v188, s72, v188
	v_add_nc_u32_e32 v189, 48, v183
	s_mov_b32 s72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s73, s72, s26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s74, s72, v188
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s76, s73, v188
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e64 s78, s31, v189
	s_and_b32 s74, s74, s76
	s_and_b32 s74, s74, s78
	s_wait_storecnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v185, v189, s26
	v_subrev_nc_u32_e32 v186, s72, v188
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v185, v185, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v185, 2, v185
	s_mov_b32 s79, exec_lo
	s_mov_b32 exec_lo, s74
	buffer_store_b128 v[92:95], v185, s[52:55], null offen
	s_mov_b32 exec_lo, s79
	s_mov_b32 exec_lo, -1
	s_wait_storecnt 0x0
	.Lfp8_ratio_256x128x8_row_silu_end:
	s_endpgm
.Lgemm_mq4g256v2_fp8_silu_row_b1_end:
.size gemm_mq4g256v2_fp8_silu_row_b1, .Lgemm_mq4g256v2_fp8_silu_row_b1_end-gemm_mq4g256v2_fp8_silu_row_b1
