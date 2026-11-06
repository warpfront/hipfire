
kernels/gemv_mq2_xbatch_pm_gfx1201.co:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000001700 <gemv_mq2g256v2_xbatch_pm>:
	s_load_b64 s[10:11], s[0:1], 0x18                          // 000000001700: F4002280 F8000018
	s_lshl_b32 s12, ttmp9, 2                                   // 000000001708: 840C8275
	s_wait_kmcnt 0x0                                           // 00000000170C: BFC70000
	s_cmp_ge_i32 s12, s10                                      // 000000001710: BF030A0C
	s_cbranch_scc1 9810                                        // 000000001714: BFA22652 <gemv_mq2g256v2_xbatch_pm+0x9960>
	s_load_b128 s[4:7], s[0:1], 0x0                            // 000000001718: F4004100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 000000001720: F4002200 F8000010
	s_load_b32 s3, s[0:1], 0x20                                // 000000001728: F40000C0 F8000020
	s_sub_co_i32 s13, s10, s12                                 // 000000001730: 818D0C0A
	s_min_i32 s13, s13, 4                                      // 000000001734: 890D840D
	s_add_co_i32 s40, s10, -1                                  // 000000001738: 8128C10A
	s_lshr_b32 s14, s11, 8                                     // 00000000173C: 850E880B
	s_mul_i32 s15, s14, 0x48                                   // 000000001740: 960FFF0E 00000048
	s_add_co_i32 s41, s12, 0                                   // 000000001748: 8129800C
	s_min_i32 s41, s41, s40                                    // 00000000174C: 89292829
	s_mul_i32 s16, s41, s15                                    // 000000001750: 96100F29
	s_add_co_i32 s41, s12, 1                                   // 000000001754: 8129810C
	s_min_i32 s41, s41, s40                                    // 000000001758: 89292829
	s_mul_i32 s17, s41, s15                                    // 00000000175C: 96110F29
	s_add_co_i32 s41, s12, 2                                   // 000000001760: 8129820C
	s_min_i32 s41, s41, s40                                    // 000000001764: 89292829
	s_mul_i32 s18, s41, s15                                    // 000000001768: 96120F29
	s_add_co_i32 s41, s12, 3                                   // 00000000176C: 8129830C
	s_min_i32 s41, s41, s40                                    // 000000001770: 89292829
	s_mul_i32 s19, s41, s15                                    // 000000001774: 96130F29
	s_lshl_b32 s38, s11, 2                                     // 000000001778: 8426820B
	s_lshl_b32 s39, s10, 2                                     // 00000000177C: 8427820A
	s_lshl_b32 s42, s12, 2                                     // 000000001780: 842A820C
	s_mul_i32 s28, s38, 0                                      // 000000001784: 961C8026
	s_mul_i32 s29, s38, 1                                      // 000000001788: 961D8126
	s_mul_i32 s30, s38, 2                                      // 00000000178C: 961E8226
	s_mul_i32 s31, s38, 3                                      // 000000001790: 961F8326
	v_lshrrev_b32_e32 v1, 4, v0                                // 000000001794: 32020084
	v_lshlrev_b32_e32 v1, 2, v1                                // 000000001798: 30020282
	v_lshlrev_b32_e32 v2, 1, v0                                // 00000000179C: 30040081
	v_lshlrev_b32_e32 v3, 5, v0                                // 0000000017A0: 30060085
	s_wait_kmcnt 0x0                                           // 0000000017A4: BFC70000
	s_mov_b32 s20, s4                                          // 0000000017A8: BE940004
	s_and_b32 s21, s5, 0xffff                                  // 0000000017AC: 8B15FF05 0000FFFF
	s_mov_b32 s22, -1                                          // 0000000017B4: BE9600C1
	s_mov_b32 s23, 0x31004000                                  // 0000000017B8: BE9700FF 31004000
	s_mov_b32 s24, s6                                          // 0000000017C0: BE980006
	s_and_b32 s25, s7, 0xffff                                  // 0000000017C4: 8B19FF07 0000FFFF
	s_mov_b32 s26, -1                                          // 0000000017CC: BE9A00C1
	s_mov_b32 s27, 0x31004000                                  // 0000000017D0: BE9B00FF 31004000
	s_min_i32 s3, s3, 4                                        // 0000000017D8: 89038403
	s_cmp_eq_u32 s3, 1                                         // 0000000017DC: BF068103
	s_cbranch_scc1 7                                           // 0000000017E0: BFA20007 <gemv_mq2g256v2_xbatch_pm+0x100>
	s_cmp_eq_u32 s3, 2                                         // 0000000017E4: BF068203
	s_cbranch_scc1 1798                                        // 0000000017E8: BFA20706 <gemv_mq2g256v2_xbatch_pm+0x1d04>
	s_cmp_eq_u32 s3, 3                                         // 0000000017EC: BF068303
	s_cbranch_scc1 4015                                        // 0000000017F0: BFA20FAF <gemv_mq2g256v2_xbatch_pm+0x3fb0>
	s_cmp_eq_u32 s3, 4                                         // 0000000017F4: BF068403
	s_cbranch_scc1 6669                                        // 0000000017F8: BFA21A0D <gemv_mq2g256v2_xbatch_pm+0x6930>
	s_branch 9752                                              // 0000000017FC: BFA02618 <gemv_mq2g256v2_xbatch_pm+0x9960>
	v_mov_b32_e32 v112, 0                                      // 000000001800: 7EE00280
	v_mov_b32_e32 v113, 0                                      // 000000001804: 7EE20280
	v_mov_b32_e32 v114, 0                                      // 000000001808: 7EE40280
	v_mov_b32_e32 v115, 0                                      // 00000000180C: 7EE60280
	v_mov_b32_e32 v116, 0                                      // 000000001810: 7EE80280
	v_mov_b32_e32 v117, 0                                      // 000000001814: 7EEA0280
	v_mov_b32_e32 v118, 0                                      // 000000001818: 7EEC0280
	v_mov_b32_e32 v119, 0                                      // 00000000181C: 7EEE0280
	v_mov_b32_e32 v120, 0                                      // 000000001820: 7EF00280
	v_mov_b32_e32 v121, 0                                      // 000000001824: 7EF20280
	v_mov_b32_e32 v122, 0                                      // 000000001828: 7EF40280
	v_mov_b32_e32 v123, 0                                      // 00000000182C: 7EF60280
	v_mov_b32_e32 v124, 0                                      // 000000001830: 7EF80280
	v_mov_b32_e32 v125, 0                                      // 000000001834: 7EFA0280
	v_mov_b32_e32 v126, 0                                      // 000000001838: 7EFC0280
	v_mov_b32_e32 v127, 0                                      // 00000000183C: 7EFE0280
	s_lshr_b32 s36, s14, 2                                     // 000000001840: 8524820E
	s_and_b32 s37, s14, 3                                      // 000000001844: 8B25830E
	s_cmp_eq_u32 s36, 0                                        // 000000001848: BF068024
	s_cbranch_scc1 933                                         // 00000000184C: BFA203A5 <gemv_mq2g256v2_xbatch_pm+0xfe4>
	buffer_load_b32 v8, v1, s[20:23], s16 offen                // 000000001850: C4050010 40802808 00000001
	buffer_load_b32 v9, v1, s[20:23], s17 offen                // 00000000185C: C4050011 40802809 00000001
	buffer_load_b32 v10, v1, s[20:23], s18 offen               // 000000001868: C4050012 4080280A 00000001
	buffer_load_b32 v11, v1, s[20:23], s19 offen               // 000000001874: C4050013 4080280B 00000001
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8      // 000000001880: C4050010 40802818 00000802
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8      // 00000000188C: C4050011 40802819 00000802
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8      // 000000001898: C4050012 4080281A 00000802
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8      // 0000000018A4: C4050013 4080281B 00000802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen         // 0000000018B0: C405C01C 40803050 00000003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16// 0000000018BC: C405C01C 40803054 00001003
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:72     // 0000000018C8: C4050010 4080280C 00004801
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:72     // 0000000018D4: C4050011 4080280D 00004801
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:72     // 0000000018E0: C4050012 4080280E 00004801
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:72     // 0000000018EC: C4050013 4080280F 00004801
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:80     // 0000000018F8: C4050010 4080281C 00005002
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:80     // 000000001904: C4050011 4080281D 00005002
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:80     // 000000001910: C4050012 4080281E 00005002
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:80     // 00000000191C: C4050013 4080281F 00005002
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:144    // 000000001928: C4050010 40802810 00009001
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:144    // 000000001934: C4050011 40802811 00009001
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:144    // 000000001940: C4050012 40802812 00009001
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:144    // 00000000194C: C4050013 40802813 00009001
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:152    // 000000001958: C4050010 40802820 00009802
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:152    // 000000001964: C4050011 40802821 00009802
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:152    // 000000001970: C4050012 40802822 00009802
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:152    // 00000000197C: C4050013 40802823 00009802
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:216    // 000000001988: C4050010 40802814 0000D801
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:216    // 000000001994: C4050011 40802815 0000D801
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:216    // 0000000019A0: C4050012 40802816 0000D801
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:216    // 0000000019AC: C4050013 40802817 0000D801
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:224    // 0000000019B8: C4050010 40802824 0000E002
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:224    // 0000000019C4: C4050011 40802825 0000E002
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:224    // 0000000019D0: C4050012 40802826 0000E002
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:224    // 0000000019DC: C4050013 40802827 0000E002
	s_wait_loadcnt 0x1d                                        // 0000000019E8: BFC0001D
	v_bfe_u32 v40, v24, 0, 2                                   // 0000000019EC: D6100028 02090118
	s_wait_loadcnt 0x1c                                        // 0000000019F4: BFC0001C
	v_bfe_u32 v50, v25, 0, 2                                   // 0000000019F8: D6100032 02090119
	s_wait_loadcnt 0x1b                                        // 000000001A00: BFC0001B
	v_bfe_u32 v60, v26, 0, 2                                   // 000000001A04: D610003C 0209011A
	s_wait_loadcnt 0x1a                                        // 000000001A0C: BFC0001A
	v_bfe_u32 v70, v27, 0, 2                                   // 000000001A10: D6100046 0209011B
	v_bfe_u32 v41, v24, 2, 2                                   // 000000001A18: D6100029 02090518
	v_bfe_u32 v51, v25, 2, 2                                   // 000000001A20: D6100033 02090519
	v_bfe_u32 v61, v26, 2, 2                                   // 000000001A28: D610003D 0209051A
	v_bfe_u32 v71, v27, 2, 2                                   // 000000001A30: D6100047 0209051B
	v_bfe_u32 v42, v24, 4, 2                                   // 000000001A38: D610002A 02090918
	v_bfe_u32 v52, v25, 4, 2                                   // 000000001A40: D6100034 02090919
	v_bfe_u32 v62, v26, 4, 2                                   // 000000001A48: D610003E 0209091A
	v_bfe_u32 v72, v27, 4, 2                                   // 000000001A50: D6100048 0209091B
	v_bfe_u32 v43, v24, 6, 2                                   // 000000001A58: D610002B 02090D18
	v_bfe_u32 v53, v25, 6, 2                                   // 000000001A60: D6100035 02090D19
	v_bfe_u32 v63, v26, 6, 2                                   // 000000001A68: D610003F 02090D1A
	v_bfe_u32 v73, v27, 6, 2                                   // 000000001A70: D6100049 02090D1B
	v_bfe_u32 v44, v24, 8, 2                                   // 000000001A78: D610002C 02091118
	v_bfe_u32 v54, v25, 8, 2                                   // 000000001A80: D6100036 02091119
	v_bfe_u32 v64, v26, 8, 2                                   // 000000001A88: D6100040 0209111A
	v_bfe_u32 v74, v27, 8, 2                                   // 000000001A90: D610004A 0209111B
	v_bfe_u32 v45, v24, 10, 2                                  // 000000001A98: D610002D 02091518
	v_bfe_u32 v55, v25, 10, 2                                  // 000000001AA0: D6100037 02091519
	v_bfe_u32 v65, v26, 10, 2                                  // 000000001AA8: D6100041 0209151A
	v_bfe_u32 v75, v27, 10, 2                                  // 000000001AB0: D610004B 0209151B
	v_bfe_u32 v46, v24, 12, 2                                  // 000000001AB8: D610002E 02091918
	v_bfe_u32 v56, v25, 12, 2                                  // 000000001AC0: D6100038 02091919
	v_bfe_u32 v66, v26, 12, 2                                  // 000000001AC8: D6100042 0209191A
	v_bfe_u32 v76, v27, 12, 2                                  // 000000001AD0: D610004C 0209191B
	v_bfe_u32 v47, v24, 14, 2                                  // 000000001AD8: D610002F 02091D18
	v_bfe_u32 v57, v25, 14, 2                                  // 000000001AE0: D6100039 02091D19
	v_bfe_u32 v67, v26, 14, 2                                  // 000000001AE8: D6100043 02091D1A
	v_bfe_u32 v77, v27, 14, 2                                  // 000000001AF0: D610004D 02091D1B
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000001AF8: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000001AFC: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000001B00: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000001B04: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000001B08: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000001B0C: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000001B10: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000001B14: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000001B18: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000001B1C: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000001B20: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000001B24: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000001B28: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000001B2C: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000001B30: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000001B34: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000001B38: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000001B3C: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000001B40: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000001B44: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000001B48: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000001B4C: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000001B50: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000001B54: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000001B58: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000001B5C: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000001B60: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000001B64: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000001B68: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000001B6C: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000001B70: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000001B74: 7E9A234D
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001B78: CC206028 0C225108
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001B80: CC206032 0C266509
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001B88: CC20603C 0C2A790A
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001B90: CC206046 0C2E8D0B
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001B98: CC206029 0C225308
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BA0: CC206033 0C266709
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BA8: CC20603D 0C2A7B0A
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BB0: CC206047 0C2E8F0B
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BB8: CC20602A 0C225508
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BC0: CC206034 0C266909
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BC8: CC20603E 0C2A7D0A
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BD0: CC206048 0C2E910B
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BD8: CC20602B 0C225708
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BE0: CC206035 0C266B09
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BE8: CC20603F 0C2A7F0A
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BF0: CC206049 0C2E930B
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001BF8: CC20602C 0C225908
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C00: CC206036 0C266D09
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C08: CC206040 0C2A810A
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C10: CC20604A 0C2E950B
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C18: CC20602D 0C225B08
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C20: CC206037 0C266F09
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C28: CC206041 0C2A830A
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C30: CC20604B 0C2E970B
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C38: CC20602E 0C225D08
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C40: CC206038 0C267109
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C48: CC206042 0C2A850A
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C50: CC20604C 0C2E990B
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C58: CC20602F 0C225F08
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C60: CC206039 0C267309
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C68: CC206043 0C2A870A
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001C70: CC20604D 0C2E9B0B
	buffer_load_b128 v[88:91], v3, s[24:27], s28 offen offset:1024// 000000001C78: C405C01C 40803058 00040003
	buffer_load_b128 v[92:95], v3, s[24:27], s28 offen offset:1040// 000000001C84: C405C01C 4080305C 00041003
	buffer_load_b128 v[96:99], v3, s[24:27], s28 offen offset:2048// 000000001C90: C405C01C 40803060 00080003
	buffer_load_b128 v[100:103], v3, s[24:27], s28 offen offset:2064// 000000001C9C: C405C01C 40803064 00081003
	buffer_load_b128 v[104:107], v3, s[24:27], s28 offen offset:3072// 000000001CA8: C405C01C 40803068 000C0003
	buffer_load_b128 v[108:111], v3, s[24:27], s28 offen offset:3088// 000000001CB4: C405C01C 4080306C 000C1003
	s_wait_loadcnt 0x1f                                        // 000000001CC0: BFC0001F
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 000000001CC4: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 000000001CCC: C8C6A33D 3A3AA347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000001CD4: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000001CDC: C800A13C 3A3AA146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000001CE4: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000001CEC: C800A53E 3A3AA548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000001CF4: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000001CFC: C800A73F 3A3AA749
	s_wait_loadcnt 0x1e                                        // 000000001D04: BFC0001E
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000001D08: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000001D10: C800A940 3A3AA94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000001D18: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000001D20: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 000000001D28: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000001D30: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 000000001D38: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 000000001D40: C800AF43 3A3AAF4D
	v_dual_add_f32 v112, v112, v48 :: v_dual_add_f32 v113, v113, v49// 000000001D48: C9086170 70706371
	v_dual_add_f32 v114, v114, v58 :: v_dual_add_f32 v115, v115, v59// 000000001D50: C9087572 72727773
	s_wait_loadcnt 0x19                                        // 000000001D58: BFC00019
	v_bfe_u32 v40, v28, 0, 2                                   // 000000001D5C: D6100028 0209011C
	s_wait_loadcnt 0x18                                        // 000000001D64: BFC00018
	v_bfe_u32 v50, v29, 0, 2                                   // 000000001D68: D6100032 0209011D
	s_wait_loadcnt 0x17                                        // 000000001D70: BFC00017
	v_bfe_u32 v60, v30, 0, 2                                   // 000000001D74: D610003C 0209011E
	s_wait_loadcnt 0x16                                        // 000000001D7C: BFC00016
	v_bfe_u32 v70, v31, 0, 2                                   // 000000001D80: D6100046 0209011F
	v_bfe_u32 v41, v28, 2, 2                                   // 000000001D88: D6100029 0209051C
	v_bfe_u32 v51, v29, 2, 2                                   // 000000001D90: D6100033 0209051D
	v_bfe_u32 v61, v30, 2, 2                                   // 000000001D98: D610003D 0209051E
	v_bfe_u32 v71, v31, 2, 2                                   // 000000001DA0: D6100047 0209051F
	v_bfe_u32 v42, v28, 4, 2                                   // 000000001DA8: D610002A 0209091C
	v_bfe_u32 v52, v29, 4, 2                                   // 000000001DB0: D6100034 0209091D
	v_bfe_u32 v62, v30, 4, 2                                   // 000000001DB8: D610003E 0209091E
	v_bfe_u32 v72, v31, 4, 2                                   // 000000001DC0: D6100048 0209091F
	v_bfe_u32 v43, v28, 6, 2                                   // 000000001DC8: D610002B 02090D1C
	v_bfe_u32 v53, v29, 6, 2                                   // 000000001DD0: D6100035 02090D1D
	v_bfe_u32 v63, v30, 6, 2                                   // 000000001DD8: D610003F 02090D1E
	v_bfe_u32 v73, v31, 6, 2                                   // 000000001DE0: D6100049 02090D1F
	v_bfe_u32 v44, v28, 8, 2                                   // 000000001DE8: D610002C 0209111C
	v_bfe_u32 v54, v29, 8, 2                                   // 000000001DF0: D6100036 0209111D
	v_bfe_u32 v64, v30, 8, 2                                   // 000000001DF8: D6100040 0209111E
	v_bfe_u32 v74, v31, 8, 2                                   // 000000001E00: D610004A 0209111F
	v_bfe_u32 v45, v28, 10, 2                                  // 000000001E08: D610002D 0209151C
	v_bfe_u32 v55, v29, 10, 2                                  // 000000001E10: D6100037 0209151D
	v_bfe_u32 v65, v30, 10, 2                                  // 000000001E18: D6100041 0209151E
	v_bfe_u32 v75, v31, 10, 2                                  // 000000001E20: D610004B 0209151F
	v_bfe_u32 v46, v28, 12, 2                                  // 000000001E28: D610002E 0209191C
	v_bfe_u32 v56, v29, 12, 2                                  // 000000001E30: D6100038 0209191D
	v_bfe_u32 v66, v30, 12, 2                                  // 000000001E38: D6100042 0209191E
	v_bfe_u32 v76, v31, 12, 2                                  // 000000001E40: D610004C 0209191F
	v_bfe_u32 v47, v28, 14, 2                                  // 000000001E48: D610002F 02091D1C
	v_bfe_u32 v57, v29, 14, 2                                  // 000000001E50: D6100039 02091D1D
	v_bfe_u32 v67, v30, 14, 2                                  // 000000001E58: D6100043 02091D1E
	v_bfe_u32 v77, v31, 14, 2                                  // 000000001E60: D610004D 02091D1F
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000001E68: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000001E6C: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000001E70: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000001E74: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000001E78: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000001E7C: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000001E80: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000001E84: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000001E88: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000001E8C: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000001E90: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000001E94: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000001E98: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000001E9C: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000001EA0: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000001EA4: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000001EA8: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000001EAC: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000001EB0: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000001EB4: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000001EB8: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000001EBC: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000001EC0: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000001EC4: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000001EC8: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000001ECC: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000001ED0: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000001ED4: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000001ED8: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000001EDC: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000001EE0: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000001EE4: 7E9A234D
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001EE8: CC206028 0C32510C
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001EF0: CC206032 0C36650D
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001EF8: CC20603C 0C3A790E
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F00: CC206046 0C3E8D0F
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F08: CC206029 0C32530C
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F10: CC206033 0C36670D
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F18: CC20603D 0C3A7B0E
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F20: CC206047 0C3E8F0F
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F28: CC20602A 0C32550C
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F30: CC206034 0C36690D
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F38: CC20603E 0C3A7D0E
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F40: CC206048 0C3E910F
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F48: CC20602B 0C32570C
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F50: CC206035 0C366B0D
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F58: CC20603F 0C3A7F0E
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F60: CC206049 0C3E930F
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F68: CC20602C 0C32590C
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F70: CC206036 0C366D0D
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F78: CC206040 0C3A810E
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F80: CC20604A 0C3E950F
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F88: CC20602D 0C325B0C
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F90: CC206037 0C366F0D
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001F98: CC206041 0C3A830E
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001FA0: CC20604B 0C3E970F
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001FA8: CC20602E 0C325D0C
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001FB0: CC206038 0C36710D
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001FB8: CC206042 0C3A850E
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001FC0: CC20604C 0C3E990F
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001FC8: CC20602F 0C325F0C
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001FD0: CC206039 0C36730D
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001FD8: CC206043 0C3A870E
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000001FE0: CC20604D 0C3E9B0F
	s_wait_loadcnt 0x5                                         // 000000001FE8: BFC00005
	v_dual_mul_f32 v48, v41, v89 :: v_dual_mul_f32 v49, v51, v89// 000000001FEC: C8C6B329 3030B333
	v_dual_mul_f32 v58, v61, v89 :: v_dual_mul_f32 v59, v71, v89// 000000001FF4: C8C6B33D 3A3AB347
	v_dual_fmac_f32 v48, v40, v88 :: v_dual_fmac_f32 v49, v50, v88// 000000001FFC: C800B128 3030B132
	v_dual_fmac_f32 v58, v60, v88 :: v_dual_fmac_f32 v59, v70, v88// 000000002004: C800B13C 3A3AB146
	v_dual_fmac_f32 v48, v42, v90 :: v_dual_fmac_f32 v49, v52, v90// 00000000200C: C800B52A 3030B534
	v_dual_fmac_f32 v58, v62, v90 :: v_dual_fmac_f32 v59, v72, v90// 000000002014: C800B53E 3A3AB548
	v_dual_fmac_f32 v48, v43, v91 :: v_dual_fmac_f32 v49, v53, v91// 00000000201C: C800B72B 3030B735
	v_dual_fmac_f32 v58, v63, v91 :: v_dual_fmac_f32 v59, v73, v91// 000000002024: C800B73F 3A3AB749
	s_wait_loadcnt 0x4                                         // 00000000202C: BFC00004
	v_dual_fmac_f32 v48, v44, v92 :: v_dual_fmac_f32 v49, v54, v92// 000000002030: C800B92C 3030B936
	v_dual_fmac_f32 v58, v64, v92 :: v_dual_fmac_f32 v59, v74, v92// 000000002038: C800B940 3A3AB94A
	v_dual_fmac_f32 v48, v45, v93 :: v_dual_fmac_f32 v49, v55, v93// 000000002040: C800BB2D 3030BB37
	v_dual_fmac_f32 v58, v65, v93 :: v_dual_fmac_f32 v59, v75, v93// 000000002048: C800BB41 3A3ABB4B
	v_dual_fmac_f32 v48, v46, v94 :: v_dual_fmac_f32 v49, v56, v94// 000000002050: C800BD2E 3030BD38
	v_dual_fmac_f32 v58, v66, v94 :: v_dual_fmac_f32 v59, v76, v94// 000000002058: C800BD42 3A3ABD4C
	v_dual_fmac_f32 v48, v47, v95 :: v_dual_fmac_f32 v49, v57, v95// 000000002060: C800BF2F 3030BF39
	v_dual_fmac_f32 v58, v67, v95 :: v_dual_fmac_f32 v59, v77, v95// 000000002068: C800BF43 3A3ABF4D
	v_dual_add_f32 v116, v116, v48 :: v_dual_add_f32 v117, v117, v49// 000000002070: C9086174 74746375
	v_dual_add_f32 v118, v118, v58 :: v_dual_add_f32 v119, v119, v59// 000000002078: C9087576 76767777
	v_bfe_u32 v40, v32, 0, 2                                   // 000000002080: D6100028 02090120
	v_bfe_u32 v50, v33, 0, 2                                   // 000000002088: D6100032 02090121
	v_bfe_u32 v60, v34, 0, 2                                   // 000000002090: D610003C 02090122
	v_bfe_u32 v70, v35, 0, 2                                   // 000000002098: D6100046 02090123
	v_bfe_u32 v41, v32, 2, 2                                   // 0000000020A0: D6100029 02090520
	v_bfe_u32 v51, v33, 2, 2                                   // 0000000020A8: D6100033 02090521
	v_bfe_u32 v61, v34, 2, 2                                   // 0000000020B0: D610003D 02090522
	v_bfe_u32 v71, v35, 2, 2                                   // 0000000020B8: D6100047 02090523
	v_bfe_u32 v42, v32, 4, 2                                   // 0000000020C0: D610002A 02090920
	v_bfe_u32 v52, v33, 4, 2                                   // 0000000020C8: D6100034 02090921
	v_bfe_u32 v62, v34, 4, 2                                   // 0000000020D0: D610003E 02090922
	v_bfe_u32 v72, v35, 4, 2                                   // 0000000020D8: D6100048 02090923
	v_bfe_u32 v43, v32, 6, 2                                   // 0000000020E0: D610002B 02090D20
	v_bfe_u32 v53, v33, 6, 2                                   // 0000000020E8: D6100035 02090D21
	v_bfe_u32 v63, v34, 6, 2                                   // 0000000020F0: D610003F 02090D22
	v_bfe_u32 v73, v35, 6, 2                                   // 0000000020F8: D6100049 02090D23
	v_bfe_u32 v44, v32, 8, 2                                   // 000000002100: D610002C 02091120
	v_bfe_u32 v54, v33, 8, 2                                   // 000000002108: D6100036 02091121
	v_bfe_u32 v64, v34, 8, 2                                   // 000000002110: D6100040 02091122
	v_bfe_u32 v74, v35, 8, 2                                   // 000000002118: D610004A 02091123
	v_bfe_u32 v45, v32, 10, 2                                  // 000000002120: D610002D 02091520
	v_bfe_u32 v55, v33, 10, 2                                  // 000000002128: D6100037 02091521
	v_bfe_u32 v65, v34, 10, 2                                  // 000000002130: D6100041 02091522
	v_bfe_u32 v75, v35, 10, 2                                  // 000000002138: D610004B 02091523
	v_bfe_u32 v46, v32, 12, 2                                  // 000000002140: D610002E 02091920
	v_bfe_u32 v56, v33, 12, 2                                  // 000000002148: D6100038 02091921
	v_bfe_u32 v66, v34, 12, 2                                  // 000000002150: D6100042 02091922
	v_bfe_u32 v76, v35, 12, 2                                  // 000000002158: D610004C 02091923
	v_bfe_u32 v47, v32, 14, 2                                  // 000000002160: D610002F 02091D20
	v_bfe_u32 v57, v33, 14, 2                                  // 000000002168: D6100039 02091D21
	v_bfe_u32 v67, v34, 14, 2                                  // 000000002170: D6100043 02091D22
	v_bfe_u32 v77, v35, 14, 2                                  // 000000002178: D610004D 02091D23
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000002180: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000002184: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000002188: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 00000000218C: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000002190: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000002194: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000002198: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 00000000219C: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 0000000021A0: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 0000000021A4: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 0000000021A8: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 0000000021AC: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 0000000021B0: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 0000000021B4: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 0000000021B8: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 0000000021BC: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 0000000021C0: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 0000000021C4: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 0000000021C8: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 0000000021CC: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 0000000021D0: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 0000000021D4: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 0000000021D8: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 0000000021DC: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 0000000021E0: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 0000000021E4: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 0000000021E8: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 0000000021EC: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 0000000021F0: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 0000000021F4: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 0000000021F8: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 0000000021FC: 7E9A234D
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002200: CC206028 0C425110
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002208: CC206032 0C466511
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002210: CC20603C 0C4A7912
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002218: CC206046 0C4E8D13
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002220: CC206029 0C425310
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002228: CC206033 0C466711
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002230: CC20603D 0C4A7B12
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002238: CC206047 0C4E8F13
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002240: CC20602A 0C425510
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002248: CC206034 0C466911
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002250: CC20603E 0C4A7D12
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002258: CC206048 0C4E9113
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002260: CC20602B 0C425710
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002268: CC206035 0C466B11
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002270: CC20603F 0C4A7F12
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002278: CC206049 0C4E9313
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002280: CC20602C 0C425910
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002288: CC206036 0C466D11
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002290: CC206040 0C4A8112
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002298: CC20604A 0C4E9513
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022A0: CC20602D 0C425B10
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022A8: CC206037 0C466F11
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022B0: CC206041 0C4A8312
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022B8: CC20604B 0C4E9713
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022C0: CC20602E 0C425D10
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022C8: CC206038 0C467111
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022D0: CC206042 0C4A8512
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022D8: CC20604C 0C4E9913
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022E0: CC20602F 0C425F10
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022E8: CC206039 0C467311
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022F0: CC206043 0C4A8712
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000022F8: CC20604D 0C4E9B13
	s_wait_loadcnt 0x3                                         // 000000002300: BFC00003
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97// 000000002304: C8C6C329 3030C333
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97// 00000000230C: C8C6C33D 3A3AC347
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96// 000000002314: C800C128 3030C132
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96// 00000000231C: C800C13C 3A3AC146
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98// 000000002324: C800C52A 3030C534
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98// 00000000232C: C800C53E 3A3AC548
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99// 000000002334: C800C72B 3030C735
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99// 00000000233C: C800C73F 3A3AC749
	s_wait_loadcnt 0x2                                         // 000000002344: BFC00002
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100// 000000002348: C800C92C 3030C936
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100// 000000002350: C800C940 3A3AC94A
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101// 000000002358: C800CB2D 3030CB37
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101// 000000002360: C800CB41 3A3ACB4B
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102// 000000002368: C800CD2E 3030CD38
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102// 000000002370: C800CD42 3A3ACD4C
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103// 000000002378: C800CF2F 3030CF39
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103// 000000002380: C800CF43 3A3ACF4D
	v_dual_add_f32 v120, v120, v48 :: v_dual_add_f32 v121, v121, v49// 000000002388: C9086178 78786379
	v_dual_add_f32 v122, v122, v58 :: v_dual_add_f32 v123, v123, v59// 000000002390: C908757A 7A7A777B
	v_bfe_u32 v40, v36, 0, 2                                   // 000000002398: D6100028 02090124
	v_bfe_u32 v50, v37, 0, 2                                   // 0000000023A0: D6100032 02090125
	v_bfe_u32 v60, v38, 0, 2                                   // 0000000023A8: D610003C 02090126
	v_bfe_u32 v70, v39, 0, 2                                   // 0000000023B0: D6100046 02090127
	v_bfe_u32 v41, v36, 2, 2                                   // 0000000023B8: D6100029 02090524
	v_bfe_u32 v51, v37, 2, 2                                   // 0000000023C0: D6100033 02090525
	v_bfe_u32 v61, v38, 2, 2                                   // 0000000023C8: D610003D 02090526
	v_bfe_u32 v71, v39, 2, 2                                   // 0000000023D0: D6100047 02090527
	v_bfe_u32 v42, v36, 4, 2                                   // 0000000023D8: D610002A 02090924
	v_bfe_u32 v52, v37, 4, 2                                   // 0000000023E0: D6100034 02090925
	v_bfe_u32 v62, v38, 4, 2                                   // 0000000023E8: D610003E 02090926
	v_bfe_u32 v72, v39, 4, 2                                   // 0000000023F0: D6100048 02090927
	v_bfe_u32 v43, v36, 6, 2                                   // 0000000023F8: D610002B 02090D24
	v_bfe_u32 v53, v37, 6, 2                                   // 000000002400: D6100035 02090D25
	v_bfe_u32 v63, v38, 6, 2                                   // 000000002408: D610003F 02090D26
	v_bfe_u32 v73, v39, 6, 2                                   // 000000002410: D6100049 02090D27
	v_bfe_u32 v44, v36, 8, 2                                   // 000000002418: D610002C 02091124
	v_bfe_u32 v54, v37, 8, 2                                   // 000000002420: D6100036 02091125
	v_bfe_u32 v64, v38, 8, 2                                   // 000000002428: D6100040 02091126
	v_bfe_u32 v74, v39, 8, 2                                   // 000000002430: D610004A 02091127
	v_bfe_u32 v45, v36, 10, 2                                  // 000000002438: D610002D 02091524
	v_bfe_u32 v55, v37, 10, 2                                  // 000000002440: D6100037 02091525
	v_bfe_u32 v65, v38, 10, 2                                  // 000000002448: D6100041 02091526
	v_bfe_u32 v75, v39, 10, 2                                  // 000000002450: D610004B 02091527
	v_bfe_u32 v46, v36, 12, 2                                  // 000000002458: D610002E 02091924
	v_bfe_u32 v56, v37, 12, 2                                  // 000000002460: D6100038 02091925
	v_bfe_u32 v66, v38, 12, 2                                  // 000000002468: D6100042 02091926
	v_bfe_u32 v76, v39, 12, 2                                  // 000000002470: D610004C 02091927
	v_bfe_u32 v47, v36, 14, 2                                  // 000000002478: D610002F 02091D24
	v_bfe_u32 v57, v37, 14, 2                                  // 000000002480: D6100039 02091D25
	v_bfe_u32 v67, v38, 14, 2                                  // 000000002488: D6100043 02091D26
	v_bfe_u32 v77, v39, 14, 2                                  // 000000002490: D610004D 02091D27
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000002498: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 00000000249C: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 0000000024A0: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 0000000024A4: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 0000000024A8: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 0000000024AC: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 0000000024B0: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 0000000024B4: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 0000000024B8: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 0000000024BC: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 0000000024C0: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 0000000024C4: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 0000000024C8: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 0000000024CC: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 0000000024D0: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 0000000024D4: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 0000000024D8: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 0000000024DC: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 0000000024E0: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 0000000024E4: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 0000000024E8: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 0000000024EC: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 0000000024F0: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 0000000024F4: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 0000000024F8: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 0000000024FC: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000002500: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000002504: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000002508: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 00000000250C: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000002510: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000002514: 7E9A234D
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002518: CC206028 0C525114
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002520: CC206032 0C566515
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002528: CC20603C 0C5A7916
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002530: CC206046 0C5E8D17
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002538: CC206029 0C525314
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002540: CC206033 0C566715
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002548: CC20603D 0C5A7B16
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002550: CC206047 0C5E8F17
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002558: CC20602A 0C525514
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002560: CC206034 0C566915
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002568: CC20603E 0C5A7D16
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002570: CC206048 0C5E9117
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002578: CC20602B 0C525714
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002580: CC206035 0C566B15
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002588: CC20603F 0C5A7F16
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002590: CC206049 0C5E9317
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002598: CC20602C 0C525914
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025A0: CC206036 0C566D15
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025A8: CC206040 0C5A8116
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025B0: CC20604A 0C5E9517
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025B8: CC20602D 0C525B14
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025C0: CC206037 0C566F15
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025C8: CC206041 0C5A8316
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025D0: CC20604B 0C5E9717
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025D8: CC20602E 0C525D14
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025E0: CC206038 0C567115
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025E8: CC206042 0C5A8516
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025F0: CC20604C 0C5E9917
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000025F8: CC20602F 0C525F14
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002600: CC206039 0C567315
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002608: CC206043 0C5A8716
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002610: CC20604D 0C5E9B17
	s_wait_loadcnt 0x1                                         // 000000002618: BFC00001
	v_dual_mul_f32 v48, v41, v105 :: v_dual_mul_f32 v49, v51, v105// 00000000261C: C8C6D329 3030D333
	v_dual_mul_f32 v58, v61, v105 :: v_dual_mul_f32 v59, v71, v105// 000000002624: C8C6D33D 3A3AD347
	v_dual_fmac_f32 v48, v40, v104 :: v_dual_fmac_f32 v49, v50, v104// 00000000262C: C800D128 3030D132
	v_dual_fmac_f32 v58, v60, v104 :: v_dual_fmac_f32 v59, v70, v104// 000000002634: C800D13C 3A3AD146
	v_dual_fmac_f32 v48, v42, v106 :: v_dual_fmac_f32 v49, v52, v106// 00000000263C: C800D52A 3030D534
	v_dual_fmac_f32 v58, v62, v106 :: v_dual_fmac_f32 v59, v72, v106// 000000002644: C800D53E 3A3AD548
	v_dual_fmac_f32 v48, v43, v107 :: v_dual_fmac_f32 v49, v53, v107// 00000000264C: C800D72B 3030D735
	v_dual_fmac_f32 v58, v63, v107 :: v_dual_fmac_f32 v59, v73, v107// 000000002654: C800D73F 3A3AD749
	s_wait_loadcnt 0x0                                         // 00000000265C: BFC00000
	v_dual_fmac_f32 v48, v44, v108 :: v_dual_fmac_f32 v49, v54, v108// 000000002660: C800D92C 3030D936
	v_dual_fmac_f32 v58, v64, v108 :: v_dual_fmac_f32 v59, v74, v108// 000000002668: C800D940 3A3AD94A
	v_dual_fmac_f32 v48, v45, v109 :: v_dual_fmac_f32 v49, v55, v109// 000000002670: C800DB2D 3030DB37
	v_dual_fmac_f32 v58, v65, v109 :: v_dual_fmac_f32 v59, v75, v109// 000000002678: C800DB41 3A3ADB4B
	v_dual_fmac_f32 v48, v46, v110 :: v_dual_fmac_f32 v49, v56, v110// 000000002680: C800DD2E 3030DD38
	v_dual_fmac_f32 v58, v66, v110 :: v_dual_fmac_f32 v59, v76, v110// 000000002688: C800DD42 3A3ADD4C
	v_dual_fmac_f32 v48, v47, v111 :: v_dual_fmac_f32 v49, v57, v111// 000000002690: C800DF2F 3030DF39
	v_dual_fmac_f32 v58, v67, v111 :: v_dual_fmac_f32 v59, v77, v111// 000000002698: C800DF43 3A3ADF4D
	v_dual_add_f32 v124, v124, v48 :: v_dual_add_f32 v125, v125, v49// 0000000026A0: C908617C 7C7C637D
	v_dual_add_f32 v126, v126, v58 :: v_dual_add_f32 v127, v127, v59// 0000000026A8: C908757E 7E7E777F
	s_add_co_i32 s16, s16, 0x120                               // 0000000026B0: 8110FF10 00000120
	s_add_co_i32 s17, s17, 0x120                               // 0000000026B8: 8111FF11 00000120
	s_add_co_i32 s18, s18, 0x120                               // 0000000026C0: 8112FF12 00000120
	s_add_co_i32 s19, s19, 0x120                               // 0000000026C8: 8113FF13 00000120
	s_add_co_i32 s28, s28, 0x1000                              // 0000000026D0: 811CFF1C 00001000
	s_add_co_i32 s36, s36, -1                                  // 0000000026D8: 8124C124
	s_cmp_lg_u32 s36, 0                                        // 0000000026DC: BF078024
	s_cbranch_scc1 64603                                       // 0000000026E0: BFA2FC5B <gemv_mq2g256v2_xbatch_pm+0x150>
	s_cmp_gt_u32 s37, 0                                        // 0000000026E4: BF088025
	s_cbranch_scc0 700                                         // 0000000026E8: BFA102BC <gemv_mq2g256v2_xbatch_pm+0x1adc>
	buffer_load_b32 v8, v1, s[20:23], s16 offen                // 0000000026EC: C4050010 40802808 00000001
	buffer_load_b32 v9, v1, s[20:23], s17 offen                // 0000000026F8: C4050011 40802809 00000001
	buffer_load_b32 v10, v1, s[20:23], s18 offen               // 000000002704: C4050012 4080280A 00000001
	buffer_load_b32 v11, v1, s[20:23], s19 offen               // 000000002710: C4050013 4080280B 00000001
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8      // 00000000271C: C4050010 40802818 00000802
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8      // 000000002728: C4050011 40802819 00000802
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8      // 000000002734: C4050012 4080281A 00000802
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8      // 000000002740: C4050013 4080281B 00000802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen         // 00000000274C: C405C01C 40803050 00000003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16// 000000002758: C405C01C 40803054 00001003
	s_wait_loadcnt 0x5                                         // 000000002764: BFC00005
	v_bfe_u32 v40, v24, 0, 2                                   // 000000002768: D6100028 02090118
	s_wait_loadcnt 0x4                                         // 000000002770: BFC00004
	v_bfe_u32 v50, v25, 0, 2                                   // 000000002774: D6100032 02090119
	s_wait_loadcnt 0x3                                         // 00000000277C: BFC00003
	v_bfe_u32 v60, v26, 0, 2                                   // 000000002780: D610003C 0209011A
	s_wait_loadcnt 0x2                                         // 000000002788: BFC00002
	v_bfe_u32 v70, v27, 0, 2                                   // 00000000278C: D6100046 0209011B
	v_bfe_u32 v41, v24, 2, 2                                   // 000000002794: D6100029 02090518
	v_bfe_u32 v51, v25, 2, 2                                   // 00000000279C: D6100033 02090519
	v_bfe_u32 v61, v26, 2, 2                                   // 0000000027A4: D610003D 0209051A
	v_bfe_u32 v71, v27, 2, 2                                   // 0000000027AC: D6100047 0209051B
	v_bfe_u32 v42, v24, 4, 2                                   // 0000000027B4: D610002A 02090918
	v_bfe_u32 v52, v25, 4, 2                                   // 0000000027BC: D6100034 02090919
	v_bfe_u32 v62, v26, 4, 2                                   // 0000000027C4: D610003E 0209091A
	v_bfe_u32 v72, v27, 4, 2                                   // 0000000027CC: D6100048 0209091B
	v_bfe_u32 v43, v24, 6, 2                                   // 0000000027D4: D610002B 02090D18
	v_bfe_u32 v53, v25, 6, 2                                   // 0000000027DC: D6100035 02090D19
	v_bfe_u32 v63, v26, 6, 2                                   // 0000000027E4: D610003F 02090D1A
	v_bfe_u32 v73, v27, 6, 2                                   // 0000000027EC: D6100049 02090D1B
	v_bfe_u32 v44, v24, 8, 2                                   // 0000000027F4: D610002C 02091118
	v_bfe_u32 v54, v25, 8, 2                                   // 0000000027FC: D6100036 02091119
	v_bfe_u32 v64, v26, 8, 2                                   // 000000002804: D6100040 0209111A
	v_bfe_u32 v74, v27, 8, 2                                   // 00000000280C: D610004A 0209111B
	v_bfe_u32 v45, v24, 10, 2                                  // 000000002814: D610002D 02091518
	v_bfe_u32 v55, v25, 10, 2                                  // 00000000281C: D6100037 02091519
	v_bfe_u32 v65, v26, 10, 2                                  // 000000002824: D6100041 0209151A
	v_bfe_u32 v75, v27, 10, 2                                  // 00000000282C: D610004B 0209151B
	v_bfe_u32 v46, v24, 12, 2                                  // 000000002834: D610002E 02091918
	v_bfe_u32 v56, v25, 12, 2                                  // 00000000283C: D6100038 02091919
	v_bfe_u32 v66, v26, 12, 2                                  // 000000002844: D6100042 0209191A
	v_bfe_u32 v76, v27, 12, 2                                  // 00000000284C: D610004C 0209191B
	v_bfe_u32 v47, v24, 14, 2                                  // 000000002854: D610002F 02091D18
	v_bfe_u32 v57, v25, 14, 2                                  // 00000000285C: D6100039 02091D19
	v_bfe_u32 v67, v26, 14, 2                                  // 000000002864: D6100043 02091D1A
	v_bfe_u32 v77, v27, 14, 2                                  // 00000000286C: D610004D 02091D1B
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000002874: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000002878: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 00000000287C: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000002880: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000002884: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000002888: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 00000000288C: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000002890: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000002894: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000002898: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 00000000289C: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 0000000028A0: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 0000000028A4: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 0000000028A8: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 0000000028AC: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 0000000028B0: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 0000000028B4: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 0000000028B8: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 0000000028BC: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 0000000028C0: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 0000000028C4: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 0000000028C8: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 0000000028CC: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 0000000028D0: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 0000000028D4: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 0000000028D8: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 0000000028DC: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 0000000028E0: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 0000000028E4: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 0000000028E8: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 0000000028EC: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 0000000028F0: 7E9A234D
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000028F4: CC206028 0C225108
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000028FC: CC206032 0C266509
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002904: CC20603C 0C2A790A
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000290C: CC206046 0C2E8D0B
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002914: CC206029 0C225308
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000291C: CC206033 0C266709
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002924: CC20603D 0C2A7B0A
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000292C: CC206047 0C2E8F0B
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002934: CC20602A 0C225508
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000293C: CC206034 0C266909
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002944: CC20603E 0C2A7D0A
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000294C: CC206048 0C2E910B
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002954: CC20602B 0C225708
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000295C: CC206035 0C266B09
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002964: CC20603F 0C2A7F0A
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000296C: CC206049 0C2E930B
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002974: CC20602C 0C225908
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000297C: CC206036 0C266D09
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002984: CC206040 0C2A810A
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000298C: CC20604A 0C2E950B
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002994: CC20602D 0C225B08
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000299C: CC206037 0C266F09
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000029A4: CC206041 0C2A830A
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000029AC: CC20604B 0C2E970B
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000029B4: CC20602E 0C225D08
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000029BC: CC206038 0C267109
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000029C4: CC206042 0C2A850A
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000029CC: CC20604C 0C2E990B
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000029D4: CC20602F 0C225F08
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000029DC: CC206039 0C267309
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000029E4: CC206043 0C2A870A
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000029EC: CC20604D 0C2E9B0B
	s_wait_loadcnt 0x1                                         // 0000000029F4: BFC00001
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 0000000029F8: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 000000002A00: C8C6A33D 3A3AA347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000002A08: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000002A10: C800A13C 3A3AA146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000002A18: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000002A20: C800A53E 3A3AA548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000002A28: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000002A30: C800A73F 3A3AA749
	s_wait_loadcnt 0x0                                         // 000000002A38: BFC00000
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000002A3C: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000002A44: C800A940 3A3AA94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000002A4C: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000002A54: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 000000002A5C: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000002A64: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 000000002A6C: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 000000002A74: C800AF43 3A3AAF4D
	v_dual_add_f32 v112, v112, v48 :: v_dual_add_f32 v113, v113, v49// 000000002A7C: C9086170 70706371
	v_dual_add_f32 v114, v114, v58 :: v_dual_add_f32 v115, v115, v59// 000000002A84: C9087572 72727773
	s_cmp_gt_u32 s37, 1                                        // 000000002A8C: BF088125
	s_cbranch_scc0 466                                         // 000000002A90: BFA101D2 <gemv_mq2g256v2_xbatch_pm+0x1adc>
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:72     // 000000002A94: C4050010 4080280C 00004801
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:72     // 000000002AA0: C4050011 4080280D 00004801
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:72     // 000000002AAC: C4050012 4080280E 00004801
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:72     // 000000002AB8: C4050013 4080280F 00004801
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:80     // 000000002AC4: C4050010 4080281C 00005002
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:80     // 000000002AD0: C4050011 4080281D 00005002
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:80     // 000000002ADC: C4050012 4080281E 00005002
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:80     // 000000002AE8: C4050013 4080281F 00005002
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024// 000000002AF4: C405C01C 40803050 00040003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040// 000000002B00: C405C01C 40803054 00041003
	s_wait_loadcnt 0x5                                         // 000000002B0C: BFC00005
	v_bfe_u32 v40, v28, 0, 2                                   // 000000002B10: D6100028 0209011C
	s_wait_loadcnt 0x4                                         // 000000002B18: BFC00004
	v_bfe_u32 v50, v29, 0, 2                                   // 000000002B1C: D6100032 0209011D
	s_wait_loadcnt 0x3                                         // 000000002B24: BFC00003
	v_bfe_u32 v60, v30, 0, 2                                   // 000000002B28: D610003C 0209011E
	s_wait_loadcnt 0x2                                         // 000000002B30: BFC00002
	v_bfe_u32 v70, v31, 0, 2                                   // 000000002B34: D6100046 0209011F
	v_bfe_u32 v41, v28, 2, 2                                   // 000000002B3C: D6100029 0209051C
	v_bfe_u32 v51, v29, 2, 2                                   // 000000002B44: D6100033 0209051D
	v_bfe_u32 v61, v30, 2, 2                                   // 000000002B4C: D610003D 0209051E
	v_bfe_u32 v71, v31, 2, 2                                   // 000000002B54: D6100047 0209051F
	v_bfe_u32 v42, v28, 4, 2                                   // 000000002B5C: D610002A 0209091C
	v_bfe_u32 v52, v29, 4, 2                                   // 000000002B64: D6100034 0209091D
	v_bfe_u32 v62, v30, 4, 2                                   // 000000002B6C: D610003E 0209091E
	v_bfe_u32 v72, v31, 4, 2                                   // 000000002B74: D6100048 0209091F
	v_bfe_u32 v43, v28, 6, 2                                   // 000000002B7C: D610002B 02090D1C
	v_bfe_u32 v53, v29, 6, 2                                   // 000000002B84: D6100035 02090D1D
	v_bfe_u32 v63, v30, 6, 2                                   // 000000002B8C: D610003F 02090D1E
	v_bfe_u32 v73, v31, 6, 2                                   // 000000002B94: D6100049 02090D1F
	v_bfe_u32 v44, v28, 8, 2                                   // 000000002B9C: D610002C 0209111C
	v_bfe_u32 v54, v29, 8, 2                                   // 000000002BA4: D6100036 0209111D
	v_bfe_u32 v64, v30, 8, 2                                   // 000000002BAC: D6100040 0209111E
	v_bfe_u32 v74, v31, 8, 2                                   // 000000002BB4: D610004A 0209111F
	v_bfe_u32 v45, v28, 10, 2                                  // 000000002BBC: D610002D 0209151C
	v_bfe_u32 v55, v29, 10, 2                                  // 000000002BC4: D6100037 0209151D
	v_bfe_u32 v65, v30, 10, 2                                  // 000000002BCC: D6100041 0209151E
	v_bfe_u32 v75, v31, 10, 2                                  // 000000002BD4: D610004B 0209151F
	v_bfe_u32 v46, v28, 12, 2                                  // 000000002BDC: D610002E 0209191C
	v_bfe_u32 v56, v29, 12, 2                                  // 000000002BE4: D6100038 0209191D
	v_bfe_u32 v66, v30, 12, 2                                  // 000000002BEC: D6100042 0209191E
	v_bfe_u32 v76, v31, 12, 2                                  // 000000002BF4: D610004C 0209191F
	v_bfe_u32 v47, v28, 14, 2                                  // 000000002BFC: D610002F 02091D1C
	v_bfe_u32 v57, v29, 14, 2                                  // 000000002C04: D6100039 02091D1D
	v_bfe_u32 v67, v30, 14, 2                                  // 000000002C0C: D6100043 02091D1E
	v_bfe_u32 v77, v31, 14, 2                                  // 000000002C14: D610004D 02091D1F
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000002C1C: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000002C20: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000002C24: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000002C28: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000002C2C: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000002C30: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000002C34: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000002C38: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000002C3C: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000002C40: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000002C44: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000002C48: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000002C4C: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000002C50: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000002C54: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000002C58: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000002C5C: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000002C60: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000002C64: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000002C68: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000002C6C: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000002C70: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000002C74: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000002C78: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000002C7C: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000002C80: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000002C84: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000002C88: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000002C8C: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000002C90: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000002C94: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000002C98: 7E9A234D
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002C9C: CC206028 0C32510C
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CA4: CC206032 0C36650D
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CAC: CC20603C 0C3A790E
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CB4: CC206046 0C3E8D0F
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CBC: CC206029 0C32530C
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CC4: CC206033 0C36670D
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CCC: CC20603D 0C3A7B0E
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CD4: CC206047 0C3E8F0F
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CDC: CC20602A 0C32550C
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CE4: CC206034 0C36690D
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CEC: CC20603E 0C3A7D0E
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CF4: CC206048 0C3E910F
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002CFC: CC20602B 0C32570C
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D04: CC206035 0C366B0D
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D0C: CC20603F 0C3A7F0E
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D14: CC206049 0C3E930F
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D1C: CC20602C 0C32590C
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D24: CC206036 0C366D0D
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D2C: CC206040 0C3A810E
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D34: CC20604A 0C3E950F
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D3C: CC20602D 0C325B0C
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D44: CC206037 0C366F0D
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D4C: CC206041 0C3A830E
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D54: CC20604B 0C3E970F
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D5C: CC20602E 0C325D0C
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D64: CC206038 0C36710D
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D6C: CC206042 0C3A850E
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D74: CC20604C 0C3E990F
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D7C: CC20602F 0C325F0C
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D84: CC206039 0C36730D
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D8C: CC206043 0C3A870E
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000002D94: CC20604D 0C3E9B0F
	s_wait_loadcnt 0x1                                         // 000000002D9C: BFC00001
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 000000002DA0: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 000000002DA8: C8C6A33D 3A3AA347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000002DB0: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000002DB8: C800A13C 3A3AA146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000002DC0: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000002DC8: C800A53E 3A3AA548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000002DD0: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000002DD8: C800A73F 3A3AA749
	s_wait_loadcnt 0x0                                         // 000000002DE0: BFC00000
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000002DE4: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000002DEC: C800A940 3A3AA94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000002DF4: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000002DFC: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 000000002E04: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000002E0C: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 000000002E14: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 000000002E1C: C800AF43 3A3AAF4D
	v_dual_add_f32 v116, v116, v48 :: v_dual_add_f32 v117, v117, v49// 000000002E24: C9086174 74746375
	v_dual_add_f32 v118, v118, v58 :: v_dual_add_f32 v119, v119, v59// 000000002E2C: C9087576 76767777
	s_cmp_gt_u32 s37, 2                                        // 000000002E34: BF088225
	s_cbranch_scc0 232                                         // 000000002E38: BFA100E8 <gemv_mq2g256v2_xbatch_pm+0x1adc>
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:144    // 000000002E3C: C4050010 40802810 00009001
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:144    // 000000002E48: C4050011 40802811 00009001
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:144    // 000000002E54: C4050012 40802812 00009001
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:144    // 000000002E60: C4050013 40802813 00009001
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:152    // 000000002E6C: C4050010 40802820 00009802
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:152    // 000000002E78: C4050011 40802821 00009802
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:152    // 000000002E84: C4050012 40802822 00009802
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:152    // 000000002E90: C4050013 40802823 00009802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048// 000000002E9C: C405C01C 40803050 00080003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064// 000000002EA8: C405C01C 40803054 00081003
	s_wait_loadcnt 0x5                                         // 000000002EB4: BFC00005
	v_bfe_u32 v40, v32, 0, 2                                   // 000000002EB8: D6100028 02090120
	s_wait_loadcnt 0x4                                         // 000000002EC0: BFC00004
	v_bfe_u32 v50, v33, 0, 2                                   // 000000002EC4: D6100032 02090121
	s_wait_loadcnt 0x3                                         // 000000002ECC: BFC00003
	v_bfe_u32 v60, v34, 0, 2                                   // 000000002ED0: D610003C 02090122
	s_wait_loadcnt 0x2                                         // 000000002ED8: BFC00002
	v_bfe_u32 v70, v35, 0, 2                                   // 000000002EDC: D6100046 02090123
	v_bfe_u32 v41, v32, 2, 2                                   // 000000002EE4: D6100029 02090520
	v_bfe_u32 v51, v33, 2, 2                                   // 000000002EEC: D6100033 02090521
	v_bfe_u32 v61, v34, 2, 2                                   // 000000002EF4: D610003D 02090522
	v_bfe_u32 v71, v35, 2, 2                                   // 000000002EFC: D6100047 02090523
	v_bfe_u32 v42, v32, 4, 2                                   // 000000002F04: D610002A 02090920
	v_bfe_u32 v52, v33, 4, 2                                   // 000000002F0C: D6100034 02090921
	v_bfe_u32 v62, v34, 4, 2                                   // 000000002F14: D610003E 02090922
	v_bfe_u32 v72, v35, 4, 2                                   // 000000002F1C: D6100048 02090923
	v_bfe_u32 v43, v32, 6, 2                                   // 000000002F24: D610002B 02090D20
	v_bfe_u32 v53, v33, 6, 2                                   // 000000002F2C: D6100035 02090D21
	v_bfe_u32 v63, v34, 6, 2                                   // 000000002F34: D610003F 02090D22
	v_bfe_u32 v73, v35, 6, 2                                   // 000000002F3C: D6100049 02090D23
	v_bfe_u32 v44, v32, 8, 2                                   // 000000002F44: D610002C 02091120
	v_bfe_u32 v54, v33, 8, 2                                   // 000000002F4C: D6100036 02091121
	v_bfe_u32 v64, v34, 8, 2                                   // 000000002F54: D6100040 02091122
	v_bfe_u32 v74, v35, 8, 2                                   // 000000002F5C: D610004A 02091123
	v_bfe_u32 v45, v32, 10, 2                                  // 000000002F64: D610002D 02091520
	v_bfe_u32 v55, v33, 10, 2                                  // 000000002F6C: D6100037 02091521
	v_bfe_u32 v65, v34, 10, 2                                  // 000000002F74: D6100041 02091522
	v_bfe_u32 v75, v35, 10, 2                                  // 000000002F7C: D610004B 02091523
	v_bfe_u32 v46, v32, 12, 2                                  // 000000002F84: D610002E 02091920
	v_bfe_u32 v56, v33, 12, 2                                  // 000000002F8C: D6100038 02091921
	v_bfe_u32 v66, v34, 12, 2                                  // 000000002F94: D6100042 02091922
	v_bfe_u32 v76, v35, 12, 2                                  // 000000002F9C: D610004C 02091923
	v_bfe_u32 v47, v32, 14, 2                                  // 000000002FA4: D610002F 02091D20
	v_bfe_u32 v57, v33, 14, 2                                  // 000000002FAC: D6100039 02091D21
	v_bfe_u32 v67, v34, 14, 2                                  // 000000002FB4: D6100043 02091D22
	v_bfe_u32 v77, v35, 14, 2                                  // 000000002FBC: D610004D 02091D23
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000002FC4: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000002FC8: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000002FCC: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000002FD0: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000002FD4: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000002FD8: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000002FDC: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000002FE0: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000002FE4: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000002FE8: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000002FEC: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000002FF0: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000002FF4: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000002FF8: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000002FFC: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000003000: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000003004: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000003008: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 00000000300C: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000003010: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000003014: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000003018: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 00000000301C: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000003020: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000003024: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000003028: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 00000000302C: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000003030: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000003034: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000003038: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 00000000303C: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000003040: 7E9A234D
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003044: CC206028 0C425110
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000304C: CC206032 0C466511
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003054: CC20603C 0C4A7912
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000305C: CC206046 0C4E8D13
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003064: CC206029 0C425310
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000306C: CC206033 0C466711
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003074: CC20603D 0C4A7B12
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000307C: CC206047 0C4E8F13
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003084: CC20602A 0C425510
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000308C: CC206034 0C466911
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003094: CC20603E 0C4A7D12
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000309C: CC206048 0C4E9113
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030A4: CC20602B 0C425710
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030AC: CC206035 0C466B11
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030B4: CC20603F 0C4A7F12
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030BC: CC206049 0C4E9313
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030C4: CC20602C 0C425910
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030CC: CC206036 0C466D11
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030D4: CC206040 0C4A8112
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030DC: CC20604A 0C4E9513
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030E4: CC20602D 0C425B10
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030EC: CC206037 0C466F11
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030F4: CC206041 0C4A8312
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000030FC: CC20604B 0C4E9713
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003104: CC20602E 0C425D10
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000310C: CC206038 0C467111
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003114: CC206042 0C4A8512
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000311C: CC20604C 0C4E9913
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003124: CC20602F 0C425F10
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000312C: CC206039 0C467311
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003134: CC206043 0C4A8712
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000313C: CC20604D 0C4E9B13
	s_wait_loadcnt 0x1                                         // 000000003144: BFC00001
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 000000003148: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 000000003150: C8C6A33D 3A3AA347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000003158: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000003160: C800A13C 3A3AA146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000003168: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000003170: C800A53E 3A3AA548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000003178: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000003180: C800A73F 3A3AA749
	s_wait_loadcnt 0x0                                         // 000000003188: BFC00000
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 00000000318C: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000003194: C800A940 3A3AA94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 00000000319C: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 0000000031A4: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 0000000031AC: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 0000000031B4: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 0000000031BC: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 0000000031C4: C800AF43 3A3AAF4D
	v_dual_add_f32 v120, v120, v48 :: v_dual_add_f32 v121, v121, v49// 0000000031CC: C9086178 78786379
	v_dual_add_f32 v122, v122, v58 :: v_dual_add_f32 v123, v123, v59// 0000000031D4: C908757A 7A7A777B
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117// 0000000031DC: C908E970 7070EB71
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119// 0000000031E4: C908ED72 7272EF73
	v_dual_add_f32 v120, v120, v124 :: v_dual_add_f32 v121, v121, v125// 0000000031EC: C908F978 7878FB79
	v_dual_add_f32 v122, v122, v126 :: v_dual_add_f32 v123, v123, v127// 0000000031F4: C908FD7A 7A7AFF7B
	v_dual_add_f32 v112, v112, v120 :: v_dual_add_f32 v113, v113, v121// 0000000031FC: C908F170 7070F371
	v_dual_add_f32 v114, v114, v122 :: v_dual_add_f32 v115, v115, v123// 000000003204: C908F572 7272F773
	ds_swizzle_b32 v8, v112 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000320C: D8D4020F 08000070
	ds_swizzle_b32 v9, v113 offset:swizzle(BITMASK_PERM,"1pppp")// 000000003214: D8D4020F 09000071
	ds_swizzle_b32 v10, v114 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000321C: D8D4020F 0A000072
	ds_swizzle_b32 v11, v115 offset:swizzle(BITMASK_PERM,"1pppp")// 000000003224: D8D4020F 0B000073
	s_wait_dscnt 0x3                                           // 00000000322C: BFC60003
	v_add_f32_e32 v112, v112, v8                               // 000000003230: 06E01170
	s_wait_dscnt 0x2                                           // 000000003234: BFC60002
	v_add_f32_e32 v113, v113, v9                               // 000000003238: 06E21371
	s_wait_dscnt 0x1                                           // 00000000323C: BFC60001
	v_add_f32_e32 v114, v114, v10                              // 000000003240: 06E41572
	s_wait_dscnt 0x0                                           // 000000003244: BFC60000
	v_add_f32_e32 v115, v115, v11                              // 000000003248: 06E61773
	v_cmp_gt_u32_e32 vcc_lo, 24, v0                            // 00000000324C: 7C980098
	s_wait_alu depctr_va_vcc(0)                                // 000000003250: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo                         // 000000003254: D5010004 01A91080
	v_add_lshl_u32 v4, v4, v0, 2                               // 00000000325C: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v112                               // 000000003264: DACC0000 08007004
	ds_bpermute_b32 v9, v4, v113                               // 00000000326C: DACC0000 09007104
	ds_bpermute_b32 v10, v4, v114                              // 000000003274: DACC0000 0A007204
	ds_bpermute_b32 v11, v4, v115                              // 00000000327C: DACC0000 0B007304
	s_wait_dscnt 0x3                                           // 000000003284: BFC60003
	v_add_f32_e32 v112, v112, v8                               // 000000003288: 06E01170
	s_wait_dscnt 0x2                                           // 00000000328C: BFC60002
	v_add_f32_e32 v113, v113, v9                               // 000000003290: 06E21371
	s_wait_dscnt 0x1                                           // 000000003294: BFC60001
	v_add_f32_e32 v114, v114, v10                              // 000000003298: 06E41572
	s_wait_dscnt 0x0                                           // 00000000329C: BFC60000
	v_add_f32_e32 v115, v115, v11                              // 0000000032A0: 06E61773
	v_cmp_gt_u32_e32 vcc_lo, 28, v0                            // 0000000032A4: 7C98009C
	s_wait_alu depctr_va_vcc(0)                                // 0000000032A8: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo                         // 0000000032AC: D5010004 01A90880
	v_add_lshl_u32 v4, v4, v0, 2                               // 0000000032B4: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v112                               // 0000000032BC: DACC0000 08007004
	ds_bpermute_b32 v9, v4, v113                               // 0000000032C4: DACC0000 09007104
	ds_bpermute_b32 v10, v4, v114                              // 0000000032CC: DACC0000 0A007204
	ds_bpermute_b32 v11, v4, v115                              // 0000000032D4: DACC0000 0B007304
	s_wait_dscnt 0x3                                           // 0000000032DC: BFC60003
	v_add_f32_e32 v112, v112, v8                               // 0000000032E0: 06E01170
	s_wait_dscnt 0x2                                           // 0000000032E4: BFC60002
	v_add_f32_e32 v113, v113, v9                               // 0000000032E8: 06E21371
	s_wait_dscnt 0x1                                           // 0000000032EC: BFC60001
	v_add_f32_e32 v114, v114, v10                              // 0000000032F0: 06E41572
	s_wait_dscnt 0x0                                           // 0000000032F4: BFC60000
	v_add_f32_e32 v115, v115, v11                              // 0000000032F8: 06E61773
	v_cmp_gt_u32_e32 vcc_lo, 30, v0                            // 0000000032FC: 7C98009E
	s_wait_alu depctr_va_vcc(0)                                // 000000003300: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo                         // 000000003304: D5010004 01A90480
	v_add_lshl_u32 v4, v4, v0, 2                               // 00000000330C: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v112                               // 000000003314: DACC0000 08007004
	ds_bpermute_b32 v9, v4, v113                               // 00000000331C: DACC0000 09007104
	ds_bpermute_b32 v10, v4, v114                              // 000000003324: DACC0000 0A007204
	ds_bpermute_b32 v11, v4, v115                              // 00000000332C: DACC0000 0B007304
	s_wait_dscnt 0x3                                           // 000000003334: BFC60003
	v_add_f32_e32 v112, v112, v8                               // 000000003338: 06E01170
	s_wait_dscnt 0x2                                           // 00000000333C: BFC60002
	v_add_f32_e32 v113, v113, v9                               // 000000003340: 06E21371
	s_wait_dscnt 0x1                                           // 000000003344: BFC60001
	v_add_f32_e32 v114, v114, v10                              // 000000003348: 06E41572
	s_wait_dscnt 0x0                                           // 00000000334C: BFC60000
	v_add_f32_e32 v115, v115, v11                              // 000000003350: 06E61773
	v_cmp_gt_u32_e32 vcc_lo, 31, v0                            // 000000003354: 7C98009F
	s_wait_alu depctr_va_vcc(0)                                // 000000003358: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo                         // 00000000335C: D5010004 01A90280
	v_add_lshl_u32 v4, v4, v0, 2                               // 000000003364: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v112                               // 00000000336C: DACC0000 08007004
	ds_bpermute_b32 v9, v4, v113                               // 000000003374: DACC0000 09007104
	ds_bpermute_b32 v10, v4, v114                              // 00000000337C: DACC0000 0A007204
	ds_bpermute_b32 v11, v4, v115                              // 000000003384: DACC0000 0B007304
	s_wait_dscnt 0x3                                           // 00000000338C: BFC60003
	v_add_f32_e32 v112, v112, v8                               // 000000003390: 06E01170
	s_wait_dscnt 0x2                                           // 000000003394: BFC60002
	v_add_f32_e32 v113, v113, v9                               // 000000003398: 06E21371
	s_wait_dscnt 0x1                                           // 00000000339C: BFC60001
	v_add_f32_e32 v114, v114, v10                              // 0000000033A0: 06E41572
	s_wait_dscnt 0x0                                           // 0000000033A4: BFC60000
	v_add_f32_e32 v115, v115, v11                              // 0000000033A8: 06E61773
	v_cmpx_eq_u32_e32 0, v0                                    // 0000000033AC: 7D940080
	v_mov_b32_e32 v40, s42                                     // 0000000033B0: 7E50022A
	global_store_b32 v40, v112, s[8:9]                         // 0000000033B4: EE068008 38000000 00000028
	s_cmp_le_i32 s13, 1                                        // 0000000033C0: BF05810D
	s_cbranch_scc1 13                                          // 0000000033C4: BFA2000D <gemv_mq2g256v2_xbatch_pm+0x1cfc>
	global_store_b32 v40, v113, s[8:9] offset:4                // 0000000033C8: EE068008 38800000 00000428
	s_cmp_le_i32 s13, 2                                        // 0000000033D4: BF05820D
	s_cbranch_scc1 8                                           // 0000000033D8: BFA20008 <gemv_mq2g256v2_xbatch_pm+0x1cfc>
	global_store_b32 v40, v114, s[8:9] offset:8                // 0000000033DC: EE068008 39000000 00000828
	s_cmp_le_i32 s13, 3                                        // 0000000033E8: BF05830D
	s_cbranch_scc1 3                                           // 0000000033EC: BFA20003 <gemv_mq2g256v2_xbatch_pm+0x1cfc>
	global_store_b32 v40, v115, s[8:9] offset:12               // 0000000033F0: EE068008 39800000 00000C28
	s_wait_storecnt 0x0                                        // 0000000033FC: BFC10000
	s_branch 7959                                              // 000000003400: BFA01F17 <gemv_mq2g256v2_xbatch_pm+0x9960>
	v_mov_b32_e32 v144, 0                                      // 000000003404: 7F200280
	v_mov_b32_e32 v145, 0                                      // 000000003408: 7F220280
	v_mov_b32_e32 v146, 0                                      // 00000000340C: 7F240280
	v_mov_b32_e32 v147, 0                                      // 000000003410: 7F260280
	v_mov_b32_e32 v148, 0                                      // 000000003414: 7F280280
	v_mov_b32_e32 v149, 0                                      // 000000003418: 7F2A0280
	v_mov_b32_e32 v150, 0                                      // 00000000341C: 7F2C0280
	v_mov_b32_e32 v151, 0                                      // 000000003420: 7F2E0280
	v_mov_b32_e32 v152, 0                                      // 000000003424: 7F300280
	v_mov_b32_e32 v153, 0                                      // 000000003428: 7F320280
	v_mov_b32_e32 v154, 0                                      // 00000000342C: 7F340280
	v_mov_b32_e32 v155, 0                                      // 000000003430: 7F360280
	v_mov_b32_e32 v156, 0                                      // 000000003434: 7F380280
	v_mov_b32_e32 v157, 0                                      // 000000003438: 7F3A0280
	v_mov_b32_e32 v158, 0                                      // 00000000343C: 7F3C0280
	v_mov_b32_e32 v159, 0                                      // 000000003440: 7F3E0280
	v_mov_b32_e32 v160, 0                                      // 000000003444: 7F400280
	v_mov_b32_e32 v161, 0                                      // 000000003448: 7F420280
	v_mov_b32_e32 v162, 0                                      // 00000000344C: 7F440280
	v_mov_b32_e32 v163, 0                                      // 000000003450: 7F460280
	v_mov_b32_e32 v164, 0                                      // 000000003454: 7F480280
	v_mov_b32_e32 v165, 0                                      // 000000003458: 7F4A0280
	v_mov_b32_e32 v166, 0                                      // 00000000345C: 7F4C0280
	v_mov_b32_e32 v167, 0                                      // 000000003460: 7F4E0280
	v_mov_b32_e32 v168, 0                                      // 000000003464: 7F500280
	v_mov_b32_e32 v169, 0                                      // 000000003468: 7F520280
	v_mov_b32_e32 v170, 0                                      // 00000000346C: 7F540280
	v_mov_b32_e32 v171, 0                                      // 000000003470: 7F560280
	v_mov_b32_e32 v172, 0                                      // 000000003474: 7F580280
	v_mov_b32_e32 v173, 0                                      // 000000003478: 7F5A0280
	v_mov_b32_e32 v174, 0                                      // 00000000347C: 7F5C0280
	v_mov_b32_e32 v175, 0                                      // 000000003480: 7F5E0280
	s_lshr_b32 s36, s14, 2                                     // 000000003484: 8524820E
	s_and_b32 s37, s14, 3                                      // 000000003488: 8B25830E
	s_cmp_eq_u32 s36, 0                                        // 00000000348C: BF068024
	s_cbranch_scc1 1107                                        // 000000003490: BFA20453 <gemv_mq2g256v2_xbatch_pm+0x2ee0>
	buffer_load_b32 v8, v1, s[20:23], s16 offen                // 000000003494: C4050010 40802808 00000001
	buffer_load_b32 v9, v1, s[20:23], s17 offen                // 0000000034A0: C4050011 40802809 00000001
	buffer_load_b32 v10, v1, s[20:23], s18 offen               // 0000000034AC: C4050012 4080280A 00000001
	buffer_load_b32 v11, v1, s[20:23], s19 offen               // 0000000034B8: C4050013 4080280B 00000001
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8      // 0000000034C4: C4050010 40802818 00000802
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8      // 0000000034D0: C4050011 40802819 00000802
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8      // 0000000034DC: C4050012 4080281A 00000802
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8      // 0000000034E8: C4050013 4080281B 00000802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen         // 0000000034F4: C405C01C 40803050 00000003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16// 000000003500: C405C01C 40803054 00001003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen         // 00000000350C: C405C01D 40803058 00000003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16// 000000003518: C405C01D 4080305C 00001003
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:72     // 000000003524: C4050010 4080280C 00004801
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:72     // 000000003530: C4050011 4080280D 00004801
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:72     // 00000000353C: C4050012 4080280E 00004801
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:72     // 000000003548: C4050013 4080280F 00004801
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:80     // 000000003554: C4050010 4080281C 00005002
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:80     // 000000003560: C4050011 4080281D 00005002
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:80     // 00000000356C: C4050012 4080281E 00005002
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:80     // 000000003578: C4050013 4080281F 00005002
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:144    // 000000003584: C4050010 40802810 00009001
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:144    // 000000003590: C4050011 40802811 00009001
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:144    // 00000000359C: C4050012 40802812 00009001
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:144    // 0000000035A8: C4050013 40802813 00009001
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:152    // 0000000035B4: C4050010 40802820 00009802
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:152    // 0000000035C0: C4050011 40802821 00009802
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:152    // 0000000035CC: C4050012 40802822 00009802
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:152    // 0000000035D8: C4050013 40802823 00009802
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:216    // 0000000035E4: C4050010 40802814 0000D801
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:216    // 0000000035F0: C4050011 40802815 0000D801
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:216    // 0000000035FC: C4050012 40802816 0000D801
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:216    // 000000003608: C4050013 40802817 0000D801
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:224    // 000000003614: C4050010 40802824 0000E002
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:224    // 000000003620: C4050011 40802825 0000E002
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:224    // 00000000362C: C4050012 40802826 0000E002
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:224    // 000000003638: C4050013 40802827 0000E002
	s_wait_loadcnt 0x1f                                        // 000000003644: BFC0001F
	v_bfe_u32 v40, v24, 0, 2                                   // 000000003648: D6100028 02090118
	s_wait_loadcnt 0x1e                                        // 000000003650: BFC0001E
	v_bfe_u32 v50, v25, 0, 2                                   // 000000003654: D6100032 02090119
	s_wait_loadcnt 0x1d                                        // 00000000365C: BFC0001D
	v_bfe_u32 v60, v26, 0, 2                                   // 000000003660: D610003C 0209011A
	s_wait_loadcnt 0x1c                                        // 000000003668: BFC0001C
	v_bfe_u32 v70, v27, 0, 2                                   // 00000000366C: D6100046 0209011B
	v_bfe_u32 v41, v24, 2, 2                                   // 000000003674: D6100029 02090518
	v_bfe_u32 v51, v25, 2, 2                                   // 00000000367C: D6100033 02090519
	v_bfe_u32 v61, v26, 2, 2                                   // 000000003684: D610003D 0209051A
	v_bfe_u32 v71, v27, 2, 2                                   // 00000000368C: D6100047 0209051B
	v_bfe_u32 v42, v24, 4, 2                                   // 000000003694: D610002A 02090918
	v_bfe_u32 v52, v25, 4, 2                                   // 00000000369C: D6100034 02090919
	v_bfe_u32 v62, v26, 4, 2                                   // 0000000036A4: D610003E 0209091A
	v_bfe_u32 v72, v27, 4, 2                                   // 0000000036AC: D6100048 0209091B
	v_bfe_u32 v43, v24, 6, 2                                   // 0000000036B4: D610002B 02090D18
	v_bfe_u32 v53, v25, 6, 2                                   // 0000000036BC: D6100035 02090D19
	v_bfe_u32 v63, v26, 6, 2                                   // 0000000036C4: D610003F 02090D1A
	v_bfe_u32 v73, v27, 6, 2                                   // 0000000036CC: D6100049 02090D1B
	v_bfe_u32 v44, v24, 8, 2                                   // 0000000036D4: D610002C 02091118
	v_bfe_u32 v54, v25, 8, 2                                   // 0000000036DC: D6100036 02091119
	v_bfe_u32 v64, v26, 8, 2                                   // 0000000036E4: D6100040 0209111A
	v_bfe_u32 v74, v27, 8, 2                                   // 0000000036EC: D610004A 0209111B
	v_bfe_u32 v45, v24, 10, 2                                  // 0000000036F4: D610002D 02091518
	v_bfe_u32 v55, v25, 10, 2                                  // 0000000036FC: D6100037 02091519
	v_bfe_u32 v65, v26, 10, 2                                  // 000000003704: D6100041 0209151A
	v_bfe_u32 v75, v27, 10, 2                                  // 00000000370C: D610004B 0209151B
	v_bfe_u32 v46, v24, 12, 2                                  // 000000003714: D610002E 02091918
	v_bfe_u32 v56, v25, 12, 2                                  // 00000000371C: D6100038 02091919
	v_bfe_u32 v66, v26, 12, 2                                  // 000000003724: D6100042 0209191A
	v_bfe_u32 v76, v27, 12, 2                                  // 00000000372C: D610004C 0209191B
	v_bfe_u32 v47, v24, 14, 2                                  // 000000003734: D610002F 02091D18
	v_bfe_u32 v57, v25, 14, 2                                  // 00000000373C: D6100039 02091D19
	v_bfe_u32 v67, v26, 14, 2                                  // 000000003744: D6100043 02091D1A
	v_bfe_u32 v77, v27, 14, 2                                  // 00000000374C: D610004D 02091D1B
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000003754: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000003758: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 00000000375C: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000003760: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000003764: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000003768: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 00000000376C: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000003770: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000003774: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000003778: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 00000000377C: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000003780: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000003784: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000003788: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 00000000378C: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000003790: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000003794: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000003798: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 00000000379C: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 0000000037A0: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 0000000037A4: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 0000000037A8: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 0000000037AC: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 0000000037B0: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 0000000037B4: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 0000000037B8: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 0000000037BC: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 0000000037C0: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 0000000037C4: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 0000000037C8: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 0000000037CC: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 0000000037D0: 7E9A234D
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000037D4: CC206028 0C225108
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000037DC: CC206032 0C266509
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000037E4: CC20603C 0C2A790A
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000037EC: CC206046 0C2E8D0B
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000037F4: CC206029 0C225308
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000037FC: CC206033 0C266709
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003804: CC20603D 0C2A7B0A
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000380C: CC206047 0C2E8F0B
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003814: CC20602A 0C225508
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000381C: CC206034 0C266909
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003824: CC20603E 0C2A7D0A
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000382C: CC206048 0C2E910B
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003834: CC20602B 0C225708
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000383C: CC206035 0C266B09
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003844: CC20603F 0C2A7F0A
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000384C: CC206049 0C2E930B
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003854: CC20602C 0C225908
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000385C: CC206036 0C266D09
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003864: CC206040 0C2A810A
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000386C: CC20604A 0C2E950B
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003874: CC20602D 0C225B08
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000387C: CC206037 0C266F09
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003884: CC206041 0C2A830A
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000388C: CC20604B 0C2E970B
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003894: CC20602E 0C225D08
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000389C: CC206038 0C267109
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000038A4: CC206042 0C2A850A
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000038AC: CC20604C 0C2E990B
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000038B4: CC20602F 0C225F08
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000038BC: CC206039 0C267309
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000038C4: CC206043 0C2A870A
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000038CC: CC20604D 0C2E9B0B
	buffer_load_b128 v[96:99], v3, s[24:27], s28 offen offset:1024// 0000000038D4: C405C01C 40803060 00040003
	buffer_load_b128 v[100:103], v3, s[24:27], s28 offen offset:1040// 0000000038E0: C405C01C 40803064 00041003
	buffer_load_b128 v[104:107], v3, s[24:27], s29 offen offset:1024// 0000000038EC: C405C01D 40803068 00040003
	buffer_load_b128 v[108:111], v3, s[24:27], s29 offen offset:1040// 0000000038F8: C405C01D 4080306C 00041003
	buffer_load_b128 v[112:115], v3, s[24:27], s28 offen offset:2048// 000000003904: C405C01C 40803070 00080003
	buffer_load_b128 v[116:119], v3, s[24:27], s28 offen offset:2064// 000000003910: C405C01C 40803074 00081003
	buffer_load_b128 v[120:123], v3, s[24:27], s29 offen offset:2048// 00000000391C: C405C01D 40803078 00080003
	buffer_load_b128 v[124:127], v3, s[24:27], s29 offen offset:2064// 000000003928: C405C01D 4080307C 00081003
	buffer_load_b128 v[128:131], v3, s[24:27], s28 offen offset:3072// 000000003934: C405C01C 40803080 000C0003
	buffer_load_b128 v[132:135], v3, s[24:27], s28 offen offset:3088// 000000003940: C405C01C 40803084 000C1003
	buffer_load_b128 v[136:139], v3, s[24:27], s29 offen offset:3072// 00000000394C: C405C01D 40803088 000C0003
	buffer_load_b128 v[140:143], v3, s[24:27], s29 offen offset:3088// 000000003958: C405C01D 4080308C 000C1003
	s_wait_loadcnt 0x27                                        // 000000003964: BFC00027
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 000000003968: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 000000003970: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x25                                        // 000000003978: BFC00025
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 00000000397C: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 000000003984: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 00000000398C: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000003994: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 00000000399C: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 0000000039A4: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 0000000039AC: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 0000000039B4: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 0000000039BC: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 0000000039C4: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 0000000039CC: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 0000000039D4: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 0000000039DC: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 0000000039E4: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 0000000039EC: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 0000000039F4: C800A940 3A3AA94A
	s_wait_loadcnt 0x24                                        // 0000000039FC: BFC00024
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 000000003A00: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 000000003A08: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000003A10: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000003A18: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 000000003A20: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 000000003A28: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 000000003A30: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000003A38: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 000000003A40: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 000000003A48: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 000000003A50: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 000000003A58: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 000000003A60: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 000000003A68: C800BF43 4E4EBF4D
	v_dual_add_f32 v144, v144, v48 :: v_dual_add_f32 v145, v145, v49// 000000003A70: C9086190 90906391
	v_dual_add_f32 v146, v146, v58 :: v_dual_add_f32 v147, v147, v59// 000000003A78: C9087592 92927793
	v_dual_add_f32 v160, v160, v68 :: v_dual_add_f32 v161, v161, v69// 000000003A80: C90889A0 A0A08BA1
	v_dual_add_f32 v162, v162, v78 :: v_dual_add_f32 v163, v163, v79// 000000003A88: C9089DA2 A2A29FA3
	s_wait_loadcnt 0x1f                                        // 000000003A90: BFC0001F
	v_bfe_u32 v40, v28, 0, 2                                   // 000000003A94: D6100028 0209011C
	s_wait_loadcnt 0x1e                                        // 000000003A9C: BFC0001E
	v_bfe_u32 v50, v29, 0, 2                                   // 000000003AA0: D6100032 0209011D
	s_wait_loadcnt 0x1d                                        // 000000003AA8: BFC0001D
	v_bfe_u32 v60, v30, 0, 2                                   // 000000003AAC: D610003C 0209011E
	s_wait_loadcnt 0x1c                                        // 000000003AB4: BFC0001C
	v_bfe_u32 v70, v31, 0, 2                                   // 000000003AB8: D6100046 0209011F
	v_bfe_u32 v41, v28, 2, 2                                   // 000000003AC0: D6100029 0209051C
	v_bfe_u32 v51, v29, 2, 2                                   // 000000003AC8: D6100033 0209051D
	v_bfe_u32 v61, v30, 2, 2                                   // 000000003AD0: D610003D 0209051E
	v_bfe_u32 v71, v31, 2, 2                                   // 000000003AD8: D6100047 0209051F
	v_bfe_u32 v42, v28, 4, 2                                   // 000000003AE0: D610002A 0209091C
	v_bfe_u32 v52, v29, 4, 2                                   // 000000003AE8: D6100034 0209091D
	v_bfe_u32 v62, v30, 4, 2                                   // 000000003AF0: D610003E 0209091E
	v_bfe_u32 v72, v31, 4, 2                                   // 000000003AF8: D6100048 0209091F
	v_bfe_u32 v43, v28, 6, 2                                   // 000000003B00: D610002B 02090D1C
	v_bfe_u32 v53, v29, 6, 2                                   // 000000003B08: D6100035 02090D1D
	v_bfe_u32 v63, v30, 6, 2                                   // 000000003B10: D610003F 02090D1E
	v_bfe_u32 v73, v31, 6, 2                                   // 000000003B18: D6100049 02090D1F
	v_bfe_u32 v44, v28, 8, 2                                   // 000000003B20: D610002C 0209111C
	v_bfe_u32 v54, v29, 8, 2                                   // 000000003B28: D6100036 0209111D
	v_bfe_u32 v64, v30, 8, 2                                   // 000000003B30: D6100040 0209111E
	v_bfe_u32 v74, v31, 8, 2                                   // 000000003B38: D610004A 0209111F
	v_bfe_u32 v45, v28, 10, 2                                  // 000000003B40: D610002D 0209151C
	v_bfe_u32 v55, v29, 10, 2                                  // 000000003B48: D6100037 0209151D
	v_bfe_u32 v65, v30, 10, 2                                  // 000000003B50: D6100041 0209151E
	v_bfe_u32 v75, v31, 10, 2                                  // 000000003B58: D610004B 0209151F
	v_bfe_u32 v46, v28, 12, 2                                  // 000000003B60: D610002E 0209191C
	v_bfe_u32 v56, v29, 12, 2                                  // 000000003B68: D6100038 0209191D
	v_bfe_u32 v66, v30, 12, 2                                  // 000000003B70: D6100042 0209191E
	v_bfe_u32 v76, v31, 12, 2                                  // 000000003B78: D610004C 0209191F
	v_bfe_u32 v47, v28, 14, 2                                  // 000000003B80: D610002F 02091D1C
	v_bfe_u32 v57, v29, 14, 2                                  // 000000003B88: D6100039 02091D1D
	v_bfe_u32 v67, v30, 14, 2                                  // 000000003B90: D6100043 02091D1E
	v_bfe_u32 v77, v31, 14, 2                                  // 000000003B98: D610004D 02091D1F
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000003BA0: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000003BA4: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000003BA8: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000003BAC: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000003BB0: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000003BB4: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000003BB8: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000003BBC: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000003BC0: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000003BC4: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000003BC8: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000003BCC: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000003BD0: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000003BD4: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000003BD8: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000003BDC: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000003BE0: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000003BE4: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000003BE8: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000003BEC: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000003BF0: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000003BF4: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000003BF8: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000003BFC: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000003C00: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000003C04: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000003C08: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000003C0C: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000003C10: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000003C14: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000003C18: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000003C1C: 7E9A234D
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C20: CC206028 0C32510C
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C28: CC206032 0C36650D
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C30: CC20603C 0C3A790E
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C38: CC206046 0C3E8D0F
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C40: CC206029 0C32530C
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C48: CC206033 0C36670D
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C50: CC20603D 0C3A7B0E
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C58: CC206047 0C3E8F0F
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C60: CC20602A 0C32550C
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C68: CC206034 0C36690D
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C70: CC20603E 0C3A7D0E
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C78: CC206048 0C3E910F
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C80: CC20602B 0C32570C
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C88: CC206035 0C366B0D
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C90: CC20603F 0C3A7F0E
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003C98: CC206049 0C3E930F
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CA0: CC20602C 0C32590C
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CA8: CC206036 0C366D0D
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CB0: CC206040 0C3A810E
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CB8: CC20604A 0C3E950F
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CC0: CC20602D 0C325B0C
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CC8: CC206037 0C366F0D
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CD0: CC206041 0C3A830E
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CD8: CC20604B 0C3E970F
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CE0: CC20602E 0C325D0C
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CE8: CC206038 0C36710D
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CF0: CC206042 0C3A850E
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003CF8: CC20604C 0C3E990F
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003D00: CC20602F 0C325F0C
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003D08: CC206039 0C36730D
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003D10: CC206043 0C3A870E
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003D18: CC20604D 0C3E9B0F
	s_wait_loadcnt 0xb                                         // 000000003D20: BFC0000B
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97// 000000003D24: C8C6C329 3030C333
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97// 000000003D2C: C8C6C33D 3A3AC347
	s_wait_loadcnt 0x9                                         // 000000003D34: BFC00009
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105// 000000003D38: C8C6D329 4444D333
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105// 000000003D40: C8C6D33D 4E4ED347
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96// 000000003D48: C800C128 3030C132
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96// 000000003D50: C800C13C 3A3AC146
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104// 000000003D58: C800D128 4444D132
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104// 000000003D60: C800D13C 4E4ED146
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98// 000000003D68: C800C52A 3030C534
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98// 000000003D70: C800C53E 3A3AC548
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106// 000000003D78: C800D52A 4444D534
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106// 000000003D80: C800D53E 4E4ED548
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99// 000000003D88: C800C72B 3030C735
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99// 000000003D90: C800C73F 3A3AC749
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107// 000000003D98: C800D72B 4444D735
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107// 000000003DA0: C800D73F 4E4ED749
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100// 000000003DA8: C800C92C 3030C936
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100// 000000003DB0: C800C940 3A3AC94A
	s_wait_loadcnt 0x8                                         // 000000003DB8: BFC00008
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108// 000000003DBC: C800D92C 4444D936
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108// 000000003DC4: C800D940 4E4ED94A
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101// 000000003DCC: C800CB2D 3030CB37
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101// 000000003DD4: C800CB41 3A3ACB4B
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109// 000000003DDC: C800DB2D 4444DB37
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109// 000000003DE4: C800DB41 4E4EDB4B
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102// 000000003DEC: C800CD2E 3030CD38
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102// 000000003DF4: C800CD42 3A3ACD4C
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110// 000000003DFC: C800DD2E 4444DD38
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110// 000000003E04: C800DD42 4E4EDD4C
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103// 000000003E0C: C800CF2F 3030CF39
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103// 000000003E14: C800CF43 3A3ACF4D
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111// 000000003E1C: C800DF2F 4444DF39
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111// 000000003E24: C800DF43 4E4EDF4D
	v_dual_add_f32 v148, v148, v48 :: v_dual_add_f32 v149, v149, v49// 000000003E2C: C9086194 94946395
	v_dual_add_f32 v150, v150, v58 :: v_dual_add_f32 v151, v151, v59// 000000003E34: C9087596 96967797
	v_dual_add_f32 v164, v164, v68 :: v_dual_add_f32 v165, v165, v69// 000000003E3C: C90889A4 A4A48BA5
	v_dual_add_f32 v166, v166, v78 :: v_dual_add_f32 v167, v167, v79// 000000003E44: C9089DA6 A6A69FA7
	v_bfe_u32 v40, v32, 0, 2                                   // 000000003E4C: D6100028 02090120
	v_bfe_u32 v50, v33, 0, 2                                   // 000000003E54: D6100032 02090121
	v_bfe_u32 v60, v34, 0, 2                                   // 000000003E5C: D610003C 02090122
	v_bfe_u32 v70, v35, 0, 2                                   // 000000003E64: D6100046 02090123
	v_bfe_u32 v41, v32, 2, 2                                   // 000000003E6C: D6100029 02090520
	v_bfe_u32 v51, v33, 2, 2                                   // 000000003E74: D6100033 02090521
	v_bfe_u32 v61, v34, 2, 2                                   // 000000003E7C: D610003D 02090522
	v_bfe_u32 v71, v35, 2, 2                                   // 000000003E84: D6100047 02090523
	v_bfe_u32 v42, v32, 4, 2                                   // 000000003E8C: D610002A 02090920
	v_bfe_u32 v52, v33, 4, 2                                   // 000000003E94: D6100034 02090921
	v_bfe_u32 v62, v34, 4, 2                                   // 000000003E9C: D610003E 02090922
	v_bfe_u32 v72, v35, 4, 2                                   // 000000003EA4: D6100048 02090923
	v_bfe_u32 v43, v32, 6, 2                                   // 000000003EAC: D610002B 02090D20
	v_bfe_u32 v53, v33, 6, 2                                   // 000000003EB4: D6100035 02090D21
	v_bfe_u32 v63, v34, 6, 2                                   // 000000003EBC: D610003F 02090D22
	v_bfe_u32 v73, v35, 6, 2                                   // 000000003EC4: D6100049 02090D23
	v_bfe_u32 v44, v32, 8, 2                                   // 000000003ECC: D610002C 02091120
	v_bfe_u32 v54, v33, 8, 2                                   // 000000003ED4: D6100036 02091121
	v_bfe_u32 v64, v34, 8, 2                                   // 000000003EDC: D6100040 02091122
	v_bfe_u32 v74, v35, 8, 2                                   // 000000003EE4: D610004A 02091123
	v_bfe_u32 v45, v32, 10, 2                                  // 000000003EEC: D610002D 02091520
	v_bfe_u32 v55, v33, 10, 2                                  // 000000003EF4: D6100037 02091521
	v_bfe_u32 v65, v34, 10, 2                                  // 000000003EFC: D6100041 02091522
	v_bfe_u32 v75, v35, 10, 2                                  // 000000003F04: D610004B 02091523
	v_bfe_u32 v46, v32, 12, 2                                  // 000000003F0C: D610002E 02091920
	v_bfe_u32 v56, v33, 12, 2                                  // 000000003F14: D6100038 02091921
	v_bfe_u32 v66, v34, 12, 2                                  // 000000003F1C: D6100042 02091922
	v_bfe_u32 v76, v35, 12, 2                                  // 000000003F24: D610004C 02091923
	v_bfe_u32 v47, v32, 14, 2                                  // 000000003F2C: D610002F 02091D20
	v_bfe_u32 v57, v33, 14, 2                                  // 000000003F34: D6100039 02091D21
	v_bfe_u32 v67, v34, 14, 2                                  // 000000003F3C: D6100043 02091D22
	v_bfe_u32 v77, v35, 14, 2                                  // 000000003F44: D610004D 02091D23
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000003F4C: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000003F50: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000003F54: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000003F58: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000003F5C: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000003F60: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000003F64: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000003F68: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000003F6C: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000003F70: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000003F74: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000003F78: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000003F7C: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000003F80: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000003F84: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000003F88: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000003F8C: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000003F90: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000003F94: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000003F98: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000003F9C: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000003FA0: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000003FA4: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000003FA8: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000003FAC: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000003FB0: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000003FB4: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000003FB8: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000003FBC: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000003FC0: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000003FC4: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000003FC8: 7E9A234D
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003FCC: CC206028 0C425110
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003FD4: CC206032 0C466511
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003FDC: CC20603C 0C4A7912
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003FE4: CC206046 0C4E8D13
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003FEC: CC206029 0C425310
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003FF4: CC206033 0C466711
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000003FFC: CC20603D 0C4A7B12
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004004: CC206047 0C4E8F13
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000400C: CC20602A 0C425510
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004014: CC206034 0C466911
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000401C: CC20603E 0C4A7D12
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004024: CC206048 0C4E9113
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000402C: CC20602B 0C425710
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004034: CC206035 0C466B11
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000403C: CC20603F 0C4A7F12
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004044: CC206049 0C4E9313
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000404C: CC20602C 0C425910
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004054: CC206036 0C466D11
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000405C: CC206040 0C4A8112
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004064: CC20604A 0C4E9513
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000406C: CC20602D 0C425B10
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004074: CC206037 0C466F11
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000407C: CC206041 0C4A8312
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004084: CC20604B 0C4E9713
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000408C: CC20602E 0C425D10
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004094: CC206038 0C467111
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000409C: CC206042 0C4A8512
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000040A4: CC20604C 0C4E9913
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000040AC: CC20602F 0C425F10
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000040B4: CC206039 0C467311
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000040BC: CC206043 0C4A8712
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000040C4: CC20604D 0C4E9B13
	s_wait_loadcnt 0x7                                         // 0000000040CC: BFC00007
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113// 0000000040D0: C8C6E329 3030E333
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113// 0000000040D8: C8C6E33D 3A3AE347
	s_wait_loadcnt 0x5                                         // 0000000040E0: BFC00005
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121// 0000000040E4: C8C6F329 4444F333
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121// 0000000040EC: C8C6F33D 4E4EF347
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112// 0000000040F4: C800E128 3030E132
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112// 0000000040FC: C800E13C 3A3AE146
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120// 000000004104: C800F128 4444F132
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120// 00000000410C: C800F13C 4E4EF146
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114// 000000004114: C800E52A 3030E534
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114// 00000000411C: C800E53E 3A3AE548
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122// 000000004124: C800F52A 4444F534
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122// 00000000412C: C800F53E 4E4EF548
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115// 000000004134: C800E72B 3030E735
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115// 00000000413C: C800E73F 3A3AE749
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123// 000000004144: C800F72B 4444F735
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123// 00000000414C: C800F73F 4E4EF749
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116// 000000004154: C800E92C 3030E936
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116// 00000000415C: C800E940 3A3AE94A
	s_wait_loadcnt 0x4                                         // 000000004164: BFC00004
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124// 000000004168: C800F92C 4444F936
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124// 000000004170: C800F940 4E4EF94A
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117// 000000004178: C800EB2D 3030EB37
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117// 000000004180: C800EB41 3A3AEB4B
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125// 000000004188: C800FB2D 4444FB37
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125// 000000004190: C800FB41 4E4EFB4B
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118// 000000004198: C800ED2E 3030ED38
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118// 0000000041A0: C800ED42 3A3AED4C
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126// 0000000041A8: C800FD2E 4444FD38
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126// 0000000041B0: C800FD42 4E4EFD4C
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119// 0000000041B8: C800EF2F 3030EF39
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119// 0000000041C0: C800EF43 3A3AEF4D
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127// 0000000041C8: C800FF2F 4444FF39
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127// 0000000041D0: C800FF43 4E4EFF4D
	v_dual_add_f32 v152, v152, v48 :: v_dual_add_f32 v153, v153, v49// 0000000041D8: C9086198 98986399
	v_dual_add_f32 v154, v154, v58 :: v_dual_add_f32 v155, v155, v59// 0000000041E0: C908759A 9A9A779B
	v_dual_add_f32 v168, v168, v68 :: v_dual_add_f32 v169, v169, v69// 0000000041E8: C90889A8 A8A88BA9
	v_dual_add_f32 v170, v170, v78 :: v_dual_add_f32 v171, v171, v79// 0000000041F0: C9089DAA AAAA9FAB
	v_bfe_u32 v40, v36, 0, 2                                   // 0000000041F8: D6100028 02090124
	v_bfe_u32 v50, v37, 0, 2                                   // 000000004200: D6100032 02090125
	v_bfe_u32 v60, v38, 0, 2                                   // 000000004208: D610003C 02090126
	v_bfe_u32 v70, v39, 0, 2                                   // 000000004210: D6100046 02090127
	v_bfe_u32 v41, v36, 2, 2                                   // 000000004218: D6100029 02090524
	v_bfe_u32 v51, v37, 2, 2                                   // 000000004220: D6100033 02090525
	v_bfe_u32 v61, v38, 2, 2                                   // 000000004228: D610003D 02090526
	v_bfe_u32 v71, v39, 2, 2                                   // 000000004230: D6100047 02090527
	v_bfe_u32 v42, v36, 4, 2                                   // 000000004238: D610002A 02090924
	v_bfe_u32 v52, v37, 4, 2                                   // 000000004240: D6100034 02090925
	v_bfe_u32 v62, v38, 4, 2                                   // 000000004248: D610003E 02090926
	v_bfe_u32 v72, v39, 4, 2                                   // 000000004250: D6100048 02090927
	v_bfe_u32 v43, v36, 6, 2                                   // 000000004258: D610002B 02090D24
	v_bfe_u32 v53, v37, 6, 2                                   // 000000004260: D6100035 02090D25
	v_bfe_u32 v63, v38, 6, 2                                   // 000000004268: D610003F 02090D26
	v_bfe_u32 v73, v39, 6, 2                                   // 000000004270: D6100049 02090D27
	v_bfe_u32 v44, v36, 8, 2                                   // 000000004278: D610002C 02091124
	v_bfe_u32 v54, v37, 8, 2                                   // 000000004280: D6100036 02091125
	v_bfe_u32 v64, v38, 8, 2                                   // 000000004288: D6100040 02091126
	v_bfe_u32 v74, v39, 8, 2                                   // 000000004290: D610004A 02091127
	v_bfe_u32 v45, v36, 10, 2                                  // 000000004298: D610002D 02091524
	v_bfe_u32 v55, v37, 10, 2                                  // 0000000042A0: D6100037 02091525
	v_bfe_u32 v65, v38, 10, 2                                  // 0000000042A8: D6100041 02091526
	v_bfe_u32 v75, v39, 10, 2                                  // 0000000042B0: D610004B 02091527
	v_bfe_u32 v46, v36, 12, 2                                  // 0000000042B8: D610002E 02091924
	v_bfe_u32 v56, v37, 12, 2                                  // 0000000042C0: D6100038 02091925
	v_bfe_u32 v66, v38, 12, 2                                  // 0000000042C8: D6100042 02091926
	v_bfe_u32 v76, v39, 12, 2                                  // 0000000042D0: D610004C 02091927
	v_bfe_u32 v47, v36, 14, 2                                  // 0000000042D8: D610002F 02091D24
	v_bfe_u32 v57, v37, 14, 2                                  // 0000000042E0: D6100039 02091D25
	v_bfe_u32 v67, v38, 14, 2                                  // 0000000042E8: D6100043 02091D26
	v_bfe_u32 v77, v39, 14, 2                                  // 0000000042F0: D610004D 02091D27
	v_cvt_f32_ubyte0_e32 v40, v40                              // 0000000042F8: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 0000000042FC: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000004300: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000004304: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000004308: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 00000000430C: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000004310: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000004314: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000004318: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 00000000431C: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000004320: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000004324: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000004328: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 00000000432C: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000004330: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000004334: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000004338: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 00000000433C: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000004340: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000004344: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000004348: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 00000000434C: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000004350: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000004354: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000004358: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 00000000435C: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000004360: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000004364: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000004368: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 00000000436C: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000004370: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000004374: 7E9A234D
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004378: CC206028 0C525114
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004380: CC206032 0C566515
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004388: CC20603C 0C5A7916
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004390: CC206046 0C5E8D17
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004398: CC206029 0C525314
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043A0: CC206033 0C566715
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043A8: CC20603D 0C5A7B16
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043B0: CC206047 0C5E8F17
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043B8: CC20602A 0C525514
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043C0: CC206034 0C566915
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043C8: CC20603E 0C5A7D16
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043D0: CC206048 0C5E9117
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043D8: CC20602B 0C525714
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043E0: CC206035 0C566B15
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043E8: CC20603F 0C5A7F16
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043F0: CC206049 0C5E9317
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000043F8: CC20602C 0C525914
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004400: CC206036 0C566D15
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004408: CC206040 0C5A8116
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004410: CC20604A 0C5E9517
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004418: CC20602D 0C525B14
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004420: CC206037 0C566F15
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004428: CC206041 0C5A8316
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004430: CC20604B 0C5E9717
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004438: CC20602E 0C525D14
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004440: CC206038 0C567115
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004448: CC206042 0C5A8516
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004450: CC20604C 0C5E9917
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004458: CC20602F 0C525F14
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004460: CC206039 0C567315
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004468: CC206043 0C5A8716
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004470: CC20604D 0C5E9B17
	s_wait_loadcnt 0x3                                         // 000000004478: BFC00003
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129// 00000000447C: C8C70329 30310333
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129// 000000004484: C8C7033D 3A3B0347
	s_wait_loadcnt 0x1                                         // 00000000448C: BFC00001
	v_dual_mul_f32 v68, v41, v137 :: v_dual_mul_f32 v69, v51, v137// 000000004490: C8C71329 44451333
	v_dual_mul_f32 v78, v61, v137 :: v_dual_mul_f32 v79, v71, v137// 000000004498: C8C7133D 4E4F1347
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128// 0000000044A0: C8010128 30310132
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128// 0000000044A8: C801013C 3A3B0146
	v_dual_fmac_f32 v68, v40, v136 :: v_dual_fmac_f32 v69, v50, v136// 0000000044B0: C8011128 44451132
	v_dual_fmac_f32 v78, v60, v136 :: v_dual_fmac_f32 v79, v70, v136// 0000000044B8: C801113C 4E4F1146
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130// 0000000044C0: C801052A 30310534
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130// 0000000044C8: C801053E 3A3B0548
	v_dual_fmac_f32 v68, v42, v138 :: v_dual_fmac_f32 v69, v52, v138// 0000000044D0: C801152A 44451534
	v_dual_fmac_f32 v78, v62, v138 :: v_dual_fmac_f32 v79, v72, v138// 0000000044D8: C801153E 4E4F1548
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131// 0000000044E0: C801072B 30310735
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131// 0000000044E8: C801073F 3A3B0749
	v_dual_fmac_f32 v68, v43, v139 :: v_dual_fmac_f32 v69, v53, v139// 0000000044F0: C801172B 44451735
	v_dual_fmac_f32 v78, v63, v139 :: v_dual_fmac_f32 v79, v73, v139// 0000000044F8: C801173F 4E4F1749
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132// 000000004500: C801092C 30310936
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132// 000000004508: C8010940 3A3B094A
	s_wait_loadcnt 0x0                                         // 000000004510: BFC00000
	v_dual_fmac_f32 v68, v44, v140 :: v_dual_fmac_f32 v69, v54, v140// 000000004514: C801192C 44451936
	v_dual_fmac_f32 v78, v64, v140 :: v_dual_fmac_f32 v79, v74, v140// 00000000451C: C8011940 4E4F194A
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133// 000000004524: C8010B2D 30310B37
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133// 00000000452C: C8010B41 3A3B0B4B
	v_dual_fmac_f32 v68, v45, v141 :: v_dual_fmac_f32 v69, v55, v141// 000000004534: C8011B2D 44451B37
	v_dual_fmac_f32 v78, v65, v141 :: v_dual_fmac_f32 v79, v75, v141// 00000000453C: C8011B41 4E4F1B4B
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134// 000000004544: C8010D2E 30310D38
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134// 00000000454C: C8010D42 3A3B0D4C
	v_dual_fmac_f32 v68, v46, v142 :: v_dual_fmac_f32 v69, v56, v142// 000000004554: C8011D2E 44451D38
	v_dual_fmac_f32 v78, v66, v142 :: v_dual_fmac_f32 v79, v76, v142// 00000000455C: C8011D42 4E4F1D4C
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135// 000000004564: C8010F2F 30310F39
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135// 00000000456C: C8010F43 3A3B0F4D
	v_dual_fmac_f32 v68, v47, v143 :: v_dual_fmac_f32 v69, v57, v143// 000000004574: C8011F2F 44451F39
	v_dual_fmac_f32 v78, v67, v143 :: v_dual_fmac_f32 v79, v77, v143// 00000000457C: C8011F43 4E4F1F4D
	v_dual_add_f32 v156, v156, v48 :: v_dual_add_f32 v157, v157, v49// 000000004584: C908619C 9C9C639D
	v_dual_add_f32 v158, v158, v58 :: v_dual_add_f32 v159, v159, v59// 00000000458C: C908759E 9E9E779F
	v_dual_add_f32 v172, v172, v68 :: v_dual_add_f32 v173, v173, v69// 000000004594: C90889AC ACAC8BAD
	v_dual_add_f32 v174, v174, v78 :: v_dual_add_f32 v175, v175, v79// 00000000459C: C9089DAE AEAE9FAF
	s_add_co_i32 s16, s16, 0x120                               // 0000000045A4: 8110FF10 00000120
	s_add_co_i32 s17, s17, 0x120                               // 0000000045AC: 8111FF11 00000120
	s_add_co_i32 s18, s18, 0x120                               // 0000000045B4: 8112FF12 00000120
	s_add_co_i32 s19, s19, 0x120                               // 0000000045BC: 8113FF13 00000120
	s_add_co_i32 s28, s28, 0x1000                              // 0000000045C4: 811CFF1C 00001000
	s_add_co_i32 s29, s29, 0x1000                              // 0000000045CC: 811DFF1D 00001000
	s_add_co_i32 s36, s36, -1                                  // 0000000045D4: 8124C124
	s_cmp_lg_u32 s36, 0                                        // 0000000045D8: BF078024
	s_cbranch_scc1 64429                                       // 0000000045DC: BFA2FBAD <gemv_mq2g256v2_xbatch_pm+0x1d94>
	s_cmp_gt_u32 s37, 0                                        // 0000000045E0: BF088025
	s_cbranch_scc0 829                                         // 0000000045E4: BFA1033D <gemv_mq2g256v2_xbatch_pm+0x3bdc>
	buffer_load_b32 v8, v1, s[20:23], s16 offen                // 0000000045E8: C4050010 40802808 00000001
	buffer_load_b32 v9, v1, s[20:23], s17 offen                // 0000000045F4: C4050011 40802809 00000001
	buffer_load_b32 v10, v1, s[20:23], s18 offen               // 000000004600: C4050012 4080280A 00000001
	buffer_load_b32 v11, v1, s[20:23], s19 offen               // 00000000460C: C4050013 4080280B 00000001
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8      // 000000004618: C4050010 40802818 00000802
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8      // 000000004624: C4050011 40802819 00000802
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8      // 000000004630: C4050012 4080281A 00000802
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8      // 00000000463C: C4050013 4080281B 00000802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen         // 000000004648: C405C01C 40803050 00000003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16// 000000004654: C405C01C 40803054 00001003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen         // 000000004660: C405C01D 40803058 00000003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16// 00000000466C: C405C01D 4080305C 00001003
	s_wait_loadcnt 0x7                                         // 000000004678: BFC00007
	v_bfe_u32 v40, v24, 0, 2                                   // 00000000467C: D6100028 02090118
	s_wait_loadcnt 0x6                                         // 000000004684: BFC00006
	v_bfe_u32 v50, v25, 0, 2                                   // 000000004688: D6100032 02090119
	s_wait_loadcnt 0x5                                         // 000000004690: BFC00005
	v_bfe_u32 v60, v26, 0, 2                                   // 000000004694: D610003C 0209011A
	s_wait_loadcnt 0x4                                         // 00000000469C: BFC00004
	v_bfe_u32 v70, v27, 0, 2                                   // 0000000046A0: D6100046 0209011B
	v_bfe_u32 v41, v24, 2, 2                                   // 0000000046A8: D6100029 02090518
	v_bfe_u32 v51, v25, 2, 2                                   // 0000000046B0: D6100033 02090519
	v_bfe_u32 v61, v26, 2, 2                                   // 0000000046B8: D610003D 0209051A
	v_bfe_u32 v71, v27, 2, 2                                   // 0000000046C0: D6100047 0209051B
	v_bfe_u32 v42, v24, 4, 2                                   // 0000000046C8: D610002A 02090918
	v_bfe_u32 v52, v25, 4, 2                                   // 0000000046D0: D6100034 02090919
	v_bfe_u32 v62, v26, 4, 2                                   // 0000000046D8: D610003E 0209091A
	v_bfe_u32 v72, v27, 4, 2                                   // 0000000046E0: D6100048 0209091B
	v_bfe_u32 v43, v24, 6, 2                                   // 0000000046E8: D610002B 02090D18
	v_bfe_u32 v53, v25, 6, 2                                   // 0000000046F0: D6100035 02090D19
	v_bfe_u32 v63, v26, 6, 2                                   // 0000000046F8: D610003F 02090D1A
	v_bfe_u32 v73, v27, 6, 2                                   // 000000004700: D6100049 02090D1B
	v_bfe_u32 v44, v24, 8, 2                                   // 000000004708: D610002C 02091118
	v_bfe_u32 v54, v25, 8, 2                                   // 000000004710: D6100036 02091119
	v_bfe_u32 v64, v26, 8, 2                                   // 000000004718: D6100040 0209111A
	v_bfe_u32 v74, v27, 8, 2                                   // 000000004720: D610004A 0209111B
	v_bfe_u32 v45, v24, 10, 2                                  // 000000004728: D610002D 02091518
	v_bfe_u32 v55, v25, 10, 2                                  // 000000004730: D6100037 02091519
	v_bfe_u32 v65, v26, 10, 2                                  // 000000004738: D6100041 0209151A
	v_bfe_u32 v75, v27, 10, 2                                  // 000000004740: D610004B 0209151B
	v_bfe_u32 v46, v24, 12, 2                                  // 000000004748: D610002E 02091918
	v_bfe_u32 v56, v25, 12, 2                                  // 000000004750: D6100038 02091919
	v_bfe_u32 v66, v26, 12, 2                                  // 000000004758: D6100042 0209191A
	v_bfe_u32 v76, v27, 12, 2                                  // 000000004760: D610004C 0209191B
	v_bfe_u32 v47, v24, 14, 2                                  // 000000004768: D610002F 02091D18
	v_bfe_u32 v57, v25, 14, 2                                  // 000000004770: D6100039 02091D19
	v_bfe_u32 v67, v26, 14, 2                                  // 000000004778: D6100043 02091D1A
	v_bfe_u32 v77, v27, 14, 2                                  // 000000004780: D610004D 02091D1B
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000004788: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 00000000478C: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000004790: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000004794: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000004798: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 00000000479C: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 0000000047A0: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 0000000047A4: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 0000000047A8: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 0000000047AC: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 0000000047B0: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 0000000047B4: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 0000000047B8: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 0000000047BC: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 0000000047C0: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 0000000047C4: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 0000000047C8: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 0000000047CC: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 0000000047D0: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 0000000047D4: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 0000000047D8: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 0000000047DC: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 0000000047E0: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 0000000047E4: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 0000000047E8: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 0000000047EC: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 0000000047F0: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 0000000047F4: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 0000000047F8: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 0000000047FC: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000004800: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000004804: 7E9A234D
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004808: CC206028 0C225108
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004810: CC206032 0C266509
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004818: CC20603C 0C2A790A
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004820: CC206046 0C2E8D0B
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004828: CC206029 0C225308
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004830: CC206033 0C266709
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004838: CC20603D 0C2A7B0A
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004840: CC206047 0C2E8F0B
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004848: CC20602A 0C225508
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004850: CC206034 0C266909
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004858: CC20603E 0C2A7D0A
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004860: CC206048 0C2E910B
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004868: CC20602B 0C225708
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004870: CC206035 0C266B09
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004878: CC20603F 0C2A7F0A
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004880: CC206049 0C2E930B
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004888: CC20602C 0C225908
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004890: CC206036 0C266D09
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004898: CC206040 0C2A810A
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048A0: CC20604A 0C2E950B
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048A8: CC20602D 0C225B08
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048B0: CC206037 0C266F09
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048B8: CC206041 0C2A830A
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048C0: CC20604B 0C2E970B
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048C8: CC20602E 0C225D08
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048D0: CC206038 0C267109
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048D8: CC206042 0C2A850A
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048E0: CC20604C 0C2E990B
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048E8: CC20602F 0C225F08
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048F0: CC206039 0C267309
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000048F8: CC206043 0C2A870A
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004900: CC20604D 0C2E9B0B
	s_wait_loadcnt 0x3                                         // 000000004908: BFC00003
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 00000000490C: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 000000004914: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x1                                         // 00000000491C: BFC00001
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 000000004920: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 000000004928: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000004930: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000004938: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 000000004940: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 000000004948: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000004950: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000004958: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 000000004960: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 000000004968: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000004970: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000004978: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 000000004980: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 000000004988: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000004990: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000004998: C800A940 3A3AA94A
	s_wait_loadcnt 0x0                                         // 0000000049A0: BFC00000
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 0000000049A4: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 0000000049AC: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 0000000049B4: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 0000000049BC: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 0000000049C4: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 0000000049CC: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 0000000049D4: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 0000000049DC: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 0000000049E4: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 0000000049EC: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 0000000049F4: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 0000000049FC: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 000000004A04: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 000000004A0C: C800BF43 4E4EBF4D
	v_dual_add_f32 v144, v144, v48 :: v_dual_add_f32 v145, v145, v49// 000000004A14: C9086190 90906391
	v_dual_add_f32 v146, v146, v58 :: v_dual_add_f32 v147, v147, v59// 000000004A1C: C9087592 92927793
	v_dual_add_f32 v160, v160, v68 :: v_dual_add_f32 v161, v161, v69// 000000004A24: C90889A0 A0A08BA1
	v_dual_add_f32 v162, v162, v78 :: v_dual_add_f32 v163, v163, v79// 000000004A2C: C9089DA2 A2A29FA3
	s_cmp_gt_u32 s37, 1                                        // 000000004A34: BF088125
	s_cbranch_scc0 552                                         // 000000004A38: BFA10228 <gemv_mq2g256v2_xbatch_pm+0x3bdc>
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:72     // 000000004A3C: C4050010 4080280C 00004801
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:72     // 000000004A48: C4050011 4080280D 00004801
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:72     // 000000004A54: C4050012 4080280E 00004801
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:72     // 000000004A60: C4050013 4080280F 00004801
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:80     // 000000004A6C: C4050010 4080281C 00005002
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:80     // 000000004A78: C4050011 4080281D 00005002
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:80     // 000000004A84: C4050012 4080281E 00005002
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:80     // 000000004A90: C4050013 4080281F 00005002
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024// 000000004A9C: C405C01C 40803050 00040003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040// 000000004AA8: C405C01C 40803054 00041003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:1024// 000000004AB4: C405C01D 40803058 00040003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:1040// 000000004AC0: C405C01D 4080305C 00041003
	s_wait_loadcnt 0x7                                         // 000000004ACC: BFC00007
	v_bfe_u32 v40, v28, 0, 2                                   // 000000004AD0: D6100028 0209011C
	s_wait_loadcnt 0x6                                         // 000000004AD8: BFC00006
	v_bfe_u32 v50, v29, 0, 2                                   // 000000004ADC: D6100032 0209011D
	s_wait_loadcnt 0x5                                         // 000000004AE4: BFC00005
	v_bfe_u32 v60, v30, 0, 2                                   // 000000004AE8: D610003C 0209011E
	s_wait_loadcnt 0x4                                         // 000000004AF0: BFC00004
	v_bfe_u32 v70, v31, 0, 2                                   // 000000004AF4: D6100046 0209011F
	v_bfe_u32 v41, v28, 2, 2                                   // 000000004AFC: D6100029 0209051C
	v_bfe_u32 v51, v29, 2, 2                                   // 000000004B04: D6100033 0209051D
	v_bfe_u32 v61, v30, 2, 2                                   // 000000004B0C: D610003D 0209051E
	v_bfe_u32 v71, v31, 2, 2                                   // 000000004B14: D6100047 0209051F
	v_bfe_u32 v42, v28, 4, 2                                   // 000000004B1C: D610002A 0209091C
	v_bfe_u32 v52, v29, 4, 2                                   // 000000004B24: D6100034 0209091D
	v_bfe_u32 v62, v30, 4, 2                                   // 000000004B2C: D610003E 0209091E
	v_bfe_u32 v72, v31, 4, 2                                   // 000000004B34: D6100048 0209091F
	v_bfe_u32 v43, v28, 6, 2                                   // 000000004B3C: D610002B 02090D1C
	v_bfe_u32 v53, v29, 6, 2                                   // 000000004B44: D6100035 02090D1D
	v_bfe_u32 v63, v30, 6, 2                                   // 000000004B4C: D610003F 02090D1E
	v_bfe_u32 v73, v31, 6, 2                                   // 000000004B54: D6100049 02090D1F
	v_bfe_u32 v44, v28, 8, 2                                   // 000000004B5C: D610002C 0209111C
	v_bfe_u32 v54, v29, 8, 2                                   // 000000004B64: D6100036 0209111D
	v_bfe_u32 v64, v30, 8, 2                                   // 000000004B6C: D6100040 0209111E
	v_bfe_u32 v74, v31, 8, 2                                   // 000000004B74: D610004A 0209111F
	v_bfe_u32 v45, v28, 10, 2                                  // 000000004B7C: D610002D 0209151C
	v_bfe_u32 v55, v29, 10, 2                                  // 000000004B84: D6100037 0209151D
	v_bfe_u32 v65, v30, 10, 2                                  // 000000004B8C: D6100041 0209151E
	v_bfe_u32 v75, v31, 10, 2                                  // 000000004B94: D610004B 0209151F
	v_bfe_u32 v46, v28, 12, 2                                  // 000000004B9C: D610002E 0209191C
	v_bfe_u32 v56, v29, 12, 2                                  // 000000004BA4: D6100038 0209191D
	v_bfe_u32 v66, v30, 12, 2                                  // 000000004BAC: D6100042 0209191E
	v_bfe_u32 v76, v31, 12, 2                                  // 000000004BB4: D610004C 0209191F
	v_bfe_u32 v47, v28, 14, 2                                  // 000000004BBC: D610002F 02091D1C
	v_bfe_u32 v57, v29, 14, 2                                  // 000000004BC4: D6100039 02091D1D
	v_bfe_u32 v67, v30, 14, 2                                  // 000000004BCC: D6100043 02091D1E
	v_bfe_u32 v77, v31, 14, 2                                  // 000000004BD4: D610004D 02091D1F
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000004BDC: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000004BE0: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000004BE4: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000004BE8: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000004BEC: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000004BF0: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000004BF4: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000004BF8: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000004BFC: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000004C00: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000004C04: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000004C08: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000004C0C: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000004C10: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000004C14: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000004C18: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000004C1C: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000004C20: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000004C24: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000004C28: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000004C2C: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000004C30: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000004C34: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000004C38: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000004C3C: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000004C40: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000004C44: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000004C48: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000004C4C: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000004C50: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000004C54: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000004C58: 7E9A234D
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004C5C: CC206028 0C32510C
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004C64: CC206032 0C36650D
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004C6C: CC20603C 0C3A790E
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004C74: CC206046 0C3E8D0F
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004C7C: CC206029 0C32530C
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004C84: CC206033 0C36670D
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004C8C: CC20603D 0C3A7B0E
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004C94: CC206047 0C3E8F0F
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004C9C: CC20602A 0C32550C
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CA4: CC206034 0C36690D
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CAC: CC20603E 0C3A7D0E
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CB4: CC206048 0C3E910F
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CBC: CC20602B 0C32570C
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CC4: CC206035 0C366B0D
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CCC: CC20603F 0C3A7F0E
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CD4: CC206049 0C3E930F
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CDC: CC20602C 0C32590C
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CE4: CC206036 0C366D0D
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CEC: CC206040 0C3A810E
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CF4: CC20604A 0C3E950F
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004CFC: CC20602D 0C325B0C
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D04: CC206037 0C366F0D
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D0C: CC206041 0C3A830E
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D14: CC20604B 0C3E970F
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D1C: CC20602E 0C325D0C
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D24: CC206038 0C36710D
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D2C: CC206042 0C3A850E
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D34: CC20604C 0C3E990F
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D3C: CC20602F 0C325F0C
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D44: CC206039 0C36730D
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D4C: CC206043 0C3A870E
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000004D54: CC20604D 0C3E9B0F
	s_wait_loadcnt 0x3                                         // 000000004D5C: BFC00003
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 000000004D60: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 000000004D68: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x1                                         // 000000004D70: BFC00001
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 000000004D74: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 000000004D7C: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000004D84: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000004D8C: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 000000004D94: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 000000004D9C: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000004DA4: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000004DAC: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 000000004DB4: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 000000004DBC: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000004DC4: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000004DCC: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 000000004DD4: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 000000004DDC: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000004DE4: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000004DEC: C800A940 3A3AA94A
	s_wait_loadcnt 0x0                                         // 000000004DF4: BFC00000
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 000000004DF8: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 000000004E00: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000004E08: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000004E10: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 000000004E18: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 000000004E20: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 000000004E28: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000004E30: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 000000004E38: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 000000004E40: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 000000004E48: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 000000004E50: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 000000004E58: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 000000004E60: C800BF43 4E4EBF4D
	v_dual_add_f32 v148, v148, v48 :: v_dual_add_f32 v149, v149, v49// 000000004E68: C9086194 94946395
	v_dual_add_f32 v150, v150, v58 :: v_dual_add_f32 v151, v151, v59// 000000004E70: C9087596 96967797
	v_dual_add_f32 v164, v164, v68 :: v_dual_add_f32 v165, v165, v69// 000000004E78: C90889A4 A4A48BA5
	v_dual_add_f32 v166, v166, v78 :: v_dual_add_f32 v167, v167, v79// 000000004E80: C9089DA6 A6A69FA7
	s_cmp_gt_u32 s37, 2                                        // 000000004E88: BF088225
	s_cbranch_scc0 275                                         // 000000004E8C: BFA10113 <gemv_mq2g256v2_xbatch_pm+0x3bdc>
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:144    // 000000004E90: C4050010 40802810 00009001
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:144    // 000000004E9C: C4050011 40802811 00009001
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:144    // 000000004EA8: C4050012 40802812 00009001
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:144    // 000000004EB4: C4050013 40802813 00009001
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:152    // 000000004EC0: C4050010 40802820 00009802
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:152    // 000000004ECC: C4050011 40802821 00009802
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:152    // 000000004ED8: C4050012 40802822 00009802
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:152    // 000000004EE4: C4050013 40802823 00009802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048// 000000004EF0: C405C01C 40803050 00080003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064// 000000004EFC: C405C01C 40803054 00081003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:2048// 000000004F08: C405C01D 40803058 00080003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:2064// 000000004F14: C405C01D 4080305C 00081003
	s_wait_loadcnt 0x7                                         // 000000004F20: BFC00007
	v_bfe_u32 v40, v32, 0, 2                                   // 000000004F24: D6100028 02090120
	s_wait_loadcnt 0x6                                         // 000000004F2C: BFC00006
	v_bfe_u32 v50, v33, 0, 2                                   // 000000004F30: D6100032 02090121
	s_wait_loadcnt 0x5                                         // 000000004F38: BFC00005
	v_bfe_u32 v60, v34, 0, 2                                   // 000000004F3C: D610003C 02090122
	s_wait_loadcnt 0x4                                         // 000000004F44: BFC00004
	v_bfe_u32 v70, v35, 0, 2                                   // 000000004F48: D6100046 02090123
	v_bfe_u32 v41, v32, 2, 2                                   // 000000004F50: D6100029 02090520
	v_bfe_u32 v51, v33, 2, 2                                   // 000000004F58: D6100033 02090521
	v_bfe_u32 v61, v34, 2, 2                                   // 000000004F60: D610003D 02090522
	v_bfe_u32 v71, v35, 2, 2                                   // 000000004F68: D6100047 02090523
	v_bfe_u32 v42, v32, 4, 2                                   // 000000004F70: D610002A 02090920
	v_bfe_u32 v52, v33, 4, 2                                   // 000000004F78: D6100034 02090921
	v_bfe_u32 v62, v34, 4, 2                                   // 000000004F80: D610003E 02090922
	v_bfe_u32 v72, v35, 4, 2                                   // 000000004F88: D6100048 02090923
	v_bfe_u32 v43, v32, 6, 2                                   // 000000004F90: D610002B 02090D20
	v_bfe_u32 v53, v33, 6, 2                                   // 000000004F98: D6100035 02090D21
	v_bfe_u32 v63, v34, 6, 2                                   // 000000004FA0: D610003F 02090D22
	v_bfe_u32 v73, v35, 6, 2                                   // 000000004FA8: D6100049 02090D23
	v_bfe_u32 v44, v32, 8, 2                                   // 000000004FB0: D610002C 02091120
	v_bfe_u32 v54, v33, 8, 2                                   // 000000004FB8: D6100036 02091121
	v_bfe_u32 v64, v34, 8, 2                                   // 000000004FC0: D6100040 02091122
	v_bfe_u32 v74, v35, 8, 2                                   // 000000004FC8: D610004A 02091123
	v_bfe_u32 v45, v32, 10, 2                                  // 000000004FD0: D610002D 02091520
	v_bfe_u32 v55, v33, 10, 2                                  // 000000004FD8: D6100037 02091521
	v_bfe_u32 v65, v34, 10, 2                                  // 000000004FE0: D6100041 02091522
	v_bfe_u32 v75, v35, 10, 2                                  // 000000004FE8: D610004B 02091523
	v_bfe_u32 v46, v32, 12, 2                                  // 000000004FF0: D610002E 02091920
	v_bfe_u32 v56, v33, 12, 2                                  // 000000004FF8: D6100038 02091921
	v_bfe_u32 v66, v34, 12, 2                                  // 000000005000: D6100042 02091922
	v_bfe_u32 v76, v35, 12, 2                                  // 000000005008: D610004C 02091923
	v_bfe_u32 v47, v32, 14, 2                                  // 000000005010: D610002F 02091D20
	v_bfe_u32 v57, v33, 14, 2                                  // 000000005018: D6100039 02091D21
	v_bfe_u32 v67, v34, 14, 2                                  // 000000005020: D6100043 02091D22
	v_bfe_u32 v77, v35, 14, 2                                  // 000000005028: D610004D 02091D23
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000005030: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000005034: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000005038: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 00000000503C: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000005040: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000005044: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000005048: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 00000000504C: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000005050: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000005054: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000005058: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 00000000505C: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000005060: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000005064: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000005068: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 00000000506C: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000005070: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000005074: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000005078: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 00000000507C: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000005080: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000005084: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000005088: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 00000000508C: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000005090: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000005094: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000005098: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 00000000509C: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 0000000050A0: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 0000000050A4: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 0000000050A8: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 0000000050AC: 7E9A234D
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000050B0: CC206028 0C425110
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000050B8: CC206032 0C466511
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000050C0: CC20603C 0C4A7912
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000050C8: CC206046 0C4E8D13
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000050D0: CC206029 0C425310
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000050D8: CC206033 0C466711
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000050E0: CC20603D 0C4A7B12
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000050E8: CC206047 0C4E8F13
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000050F0: CC20602A 0C425510
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000050F8: CC206034 0C466911
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005100: CC20603E 0C4A7D12
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005108: CC206048 0C4E9113
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005110: CC20602B 0C425710
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005118: CC206035 0C466B11
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005120: CC20603F 0C4A7F12
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005128: CC206049 0C4E9313
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005130: CC20602C 0C425910
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005138: CC206036 0C466D11
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005140: CC206040 0C4A8112
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005148: CC20604A 0C4E9513
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005150: CC20602D 0C425B10
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005158: CC206037 0C466F11
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005160: CC206041 0C4A8312
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005168: CC20604B 0C4E9713
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005170: CC20602E 0C425D10
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005178: CC206038 0C467111
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005180: CC206042 0C4A8512
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005188: CC20604C 0C4E9913
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005190: CC20602F 0C425F10
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005198: CC206039 0C467311
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000051A0: CC206043 0C4A8712
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000051A8: CC20604D 0C4E9B13
	s_wait_loadcnt 0x3                                         // 0000000051B0: BFC00003
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 0000000051B4: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 0000000051BC: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x1                                         // 0000000051C4: BFC00001
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 0000000051C8: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 0000000051D0: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 0000000051D8: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 0000000051E0: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 0000000051E8: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 0000000051F0: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 0000000051F8: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000005200: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 000000005208: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 000000005210: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000005218: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000005220: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 000000005228: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 000000005230: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000005238: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000005240: C800A940 3A3AA94A
	s_wait_loadcnt 0x0                                         // 000000005248: BFC00000
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 00000000524C: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 000000005254: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 00000000525C: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000005264: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 00000000526C: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 000000005274: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 00000000527C: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000005284: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 00000000528C: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 000000005294: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 00000000529C: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 0000000052A4: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 0000000052AC: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 0000000052B4: C800BF43 4E4EBF4D
	v_dual_add_f32 v152, v152, v48 :: v_dual_add_f32 v153, v153, v49// 0000000052BC: C9086198 98986399
	v_dual_add_f32 v154, v154, v58 :: v_dual_add_f32 v155, v155, v59// 0000000052C4: C908759A 9A9A779B
	v_dual_add_f32 v168, v168, v68 :: v_dual_add_f32 v169, v169, v69// 0000000052CC: C90889A8 A8A88BA9
	v_dual_add_f32 v170, v170, v78 :: v_dual_add_f32 v171, v171, v79// 0000000052D4: C9089DAA AAAA9FAB
	v_dual_add_f32 v144, v144, v148 :: v_dual_add_f32 v145, v145, v149// 0000000052DC: C9092990 90912B91
	v_dual_add_f32 v146, v146, v150 :: v_dual_add_f32 v147, v147, v151// 0000000052E4: C9092D92 92932F93
	v_dual_add_f32 v152, v152, v156 :: v_dual_add_f32 v153, v153, v157// 0000000052EC: C9093998 98993B99
	v_dual_add_f32 v154, v154, v158 :: v_dual_add_f32 v155, v155, v159// 0000000052F4: C9093D9A 9A9B3F9B
	v_dual_add_f32 v144, v144, v152 :: v_dual_add_f32 v145, v145, v153// 0000000052FC: C9093190 90913391
	v_dual_add_f32 v146, v146, v154 :: v_dual_add_f32 v147, v147, v155// 000000005304: C9093592 92933793
	v_dual_add_f32 v160, v160, v164 :: v_dual_add_f32 v161, v161, v165// 00000000530C: C90949A0 A0A14BA1
	v_dual_add_f32 v162, v162, v166 :: v_dual_add_f32 v163, v163, v167// 000000005314: C9094DA2 A2A34FA3
	v_dual_add_f32 v168, v168, v172 :: v_dual_add_f32 v169, v169, v173// 00000000531C: C90959A8 A8A95BA9
	v_dual_add_f32 v170, v170, v174 :: v_dual_add_f32 v171, v171, v175// 000000005324: C9095DAA AAAB5FAB
	v_dual_add_f32 v160, v160, v168 :: v_dual_add_f32 v161, v161, v169// 00000000532C: C90951A0 A0A153A1
	v_dual_add_f32 v162, v162, v170 :: v_dual_add_f32 v163, v163, v171// 000000005334: C90955A2 A2A357A3
	ds_swizzle_b32 v8, v144 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000533C: D8D4020F 08000090
	ds_swizzle_b32 v9, v145 offset:swizzle(BITMASK_PERM,"1pppp")// 000000005344: D8D4020F 09000091
	ds_swizzle_b32 v10, v146 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000534C: D8D4020F 0A000092
	ds_swizzle_b32 v11, v147 offset:swizzle(BITMASK_PERM,"1pppp")// 000000005354: D8D4020F 0B000093
	ds_swizzle_b32 v12, v160 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000535C: D8D4020F 0C0000A0
	ds_swizzle_b32 v13, v161 offset:swizzle(BITMASK_PERM,"1pppp")// 000000005364: D8D4020F 0D0000A1
	ds_swizzle_b32 v14, v162 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000536C: D8D4020F 0E0000A2
	ds_swizzle_b32 v15, v163 offset:swizzle(BITMASK_PERM,"1pppp")// 000000005374: D8D4020F 0F0000A3
	s_wait_dscnt 0x7                                           // 00000000537C: BFC60007
	v_add_f32_e32 v144, v144, v8                               // 000000005380: 07201190
	s_wait_dscnt 0x6                                           // 000000005384: BFC60006
	v_add_f32_e32 v145, v145, v9                               // 000000005388: 07221391
	s_wait_dscnt 0x5                                           // 00000000538C: BFC60005
	v_add_f32_e32 v146, v146, v10                              // 000000005390: 07241592
	s_wait_dscnt 0x4                                           // 000000005394: BFC60004
	v_add_f32_e32 v147, v147, v11                              // 000000005398: 07261793
	s_wait_dscnt 0x3                                           // 00000000539C: BFC60003
	v_add_f32_e32 v160, v160, v12                              // 0000000053A0: 074019A0
	s_wait_dscnt 0x2                                           // 0000000053A4: BFC60002
	v_add_f32_e32 v161, v161, v13                              // 0000000053A8: 07421BA1
	s_wait_dscnt 0x1                                           // 0000000053AC: BFC60001
	v_add_f32_e32 v162, v162, v14                              // 0000000053B0: 07441DA2
	s_wait_dscnt 0x0                                           // 0000000053B4: BFC60000
	v_add_f32_e32 v163, v163, v15                              // 0000000053B8: 07461FA3
	v_cmp_gt_u32_e32 vcc_lo, 24, v0                            // 0000000053BC: 7C980098
	s_wait_alu depctr_va_vcc(0)                                // 0000000053C0: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo                         // 0000000053C4: D5010004 01A91080
	v_add_lshl_u32 v4, v4, v0, 2                               // 0000000053CC: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v144                               // 0000000053D4: DACC0000 08009004
	ds_bpermute_b32 v9, v4, v145                               // 0000000053DC: DACC0000 09009104
	ds_bpermute_b32 v10, v4, v146                              // 0000000053E4: DACC0000 0A009204
	ds_bpermute_b32 v11, v4, v147                              // 0000000053EC: DACC0000 0B009304
	ds_bpermute_b32 v12, v4, v160                              // 0000000053F4: DACC0000 0C00A004
	ds_bpermute_b32 v13, v4, v161                              // 0000000053FC: DACC0000 0D00A104
	ds_bpermute_b32 v14, v4, v162                              // 000000005404: DACC0000 0E00A204
	ds_bpermute_b32 v15, v4, v163                              // 00000000540C: DACC0000 0F00A304
	s_wait_dscnt 0x7                                           // 000000005414: BFC60007
	v_add_f32_e32 v144, v144, v8                               // 000000005418: 07201190
	s_wait_dscnt 0x6                                           // 00000000541C: BFC60006
	v_add_f32_e32 v145, v145, v9                               // 000000005420: 07221391
	s_wait_dscnt 0x5                                           // 000000005424: BFC60005
	v_add_f32_e32 v146, v146, v10                              // 000000005428: 07241592
	s_wait_dscnt 0x4                                           // 00000000542C: BFC60004
	v_add_f32_e32 v147, v147, v11                              // 000000005430: 07261793
	s_wait_dscnt 0x3                                           // 000000005434: BFC60003
	v_add_f32_e32 v160, v160, v12                              // 000000005438: 074019A0
	s_wait_dscnt 0x2                                           // 00000000543C: BFC60002
	v_add_f32_e32 v161, v161, v13                              // 000000005440: 07421BA1
	s_wait_dscnt 0x1                                           // 000000005444: BFC60001
	v_add_f32_e32 v162, v162, v14                              // 000000005448: 07441DA2
	s_wait_dscnt 0x0                                           // 00000000544C: BFC60000
	v_add_f32_e32 v163, v163, v15                              // 000000005450: 07461FA3
	v_cmp_gt_u32_e32 vcc_lo, 28, v0                            // 000000005454: 7C98009C
	s_wait_alu depctr_va_vcc(0)                                // 000000005458: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo                         // 00000000545C: D5010004 01A90880
	v_add_lshl_u32 v4, v4, v0, 2                               // 000000005464: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v144                               // 00000000546C: DACC0000 08009004
	ds_bpermute_b32 v9, v4, v145                               // 000000005474: DACC0000 09009104
	ds_bpermute_b32 v10, v4, v146                              // 00000000547C: DACC0000 0A009204
	ds_bpermute_b32 v11, v4, v147                              // 000000005484: DACC0000 0B009304
	ds_bpermute_b32 v12, v4, v160                              // 00000000548C: DACC0000 0C00A004
	ds_bpermute_b32 v13, v4, v161                              // 000000005494: DACC0000 0D00A104
	ds_bpermute_b32 v14, v4, v162                              // 00000000549C: DACC0000 0E00A204
	ds_bpermute_b32 v15, v4, v163                              // 0000000054A4: DACC0000 0F00A304
	s_wait_dscnt 0x7                                           // 0000000054AC: BFC60007
	v_add_f32_e32 v144, v144, v8                               // 0000000054B0: 07201190
	s_wait_dscnt 0x6                                           // 0000000054B4: BFC60006
	v_add_f32_e32 v145, v145, v9                               // 0000000054B8: 07221391
	s_wait_dscnt 0x5                                           // 0000000054BC: BFC60005
	v_add_f32_e32 v146, v146, v10                              // 0000000054C0: 07241592
	s_wait_dscnt 0x4                                           // 0000000054C4: BFC60004
	v_add_f32_e32 v147, v147, v11                              // 0000000054C8: 07261793
	s_wait_dscnt 0x3                                           // 0000000054CC: BFC60003
	v_add_f32_e32 v160, v160, v12                              // 0000000054D0: 074019A0
	s_wait_dscnt 0x2                                           // 0000000054D4: BFC60002
	v_add_f32_e32 v161, v161, v13                              // 0000000054D8: 07421BA1
	s_wait_dscnt 0x1                                           // 0000000054DC: BFC60001
	v_add_f32_e32 v162, v162, v14                              // 0000000054E0: 07441DA2
	s_wait_dscnt 0x0                                           // 0000000054E4: BFC60000
	v_add_f32_e32 v163, v163, v15                              // 0000000054E8: 07461FA3
	v_cmp_gt_u32_e32 vcc_lo, 30, v0                            // 0000000054EC: 7C98009E
	s_wait_alu depctr_va_vcc(0)                                // 0000000054F0: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo                         // 0000000054F4: D5010004 01A90480
	v_add_lshl_u32 v4, v4, v0, 2                               // 0000000054FC: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v144                               // 000000005504: DACC0000 08009004
	ds_bpermute_b32 v9, v4, v145                               // 00000000550C: DACC0000 09009104
	ds_bpermute_b32 v10, v4, v146                              // 000000005514: DACC0000 0A009204
	ds_bpermute_b32 v11, v4, v147                              // 00000000551C: DACC0000 0B009304
	ds_bpermute_b32 v12, v4, v160                              // 000000005524: DACC0000 0C00A004
	ds_bpermute_b32 v13, v4, v161                              // 00000000552C: DACC0000 0D00A104
	ds_bpermute_b32 v14, v4, v162                              // 000000005534: DACC0000 0E00A204
	ds_bpermute_b32 v15, v4, v163                              // 00000000553C: DACC0000 0F00A304
	s_wait_dscnt 0x7                                           // 000000005544: BFC60007
	v_add_f32_e32 v144, v144, v8                               // 000000005548: 07201190
	s_wait_dscnt 0x6                                           // 00000000554C: BFC60006
	v_add_f32_e32 v145, v145, v9                               // 000000005550: 07221391
	s_wait_dscnt 0x5                                           // 000000005554: BFC60005
	v_add_f32_e32 v146, v146, v10                              // 000000005558: 07241592
	s_wait_dscnt 0x4                                           // 00000000555C: BFC60004
	v_add_f32_e32 v147, v147, v11                              // 000000005560: 07261793
	s_wait_dscnt 0x3                                           // 000000005564: BFC60003
	v_add_f32_e32 v160, v160, v12                              // 000000005568: 074019A0
	s_wait_dscnt 0x2                                           // 00000000556C: BFC60002
	v_add_f32_e32 v161, v161, v13                              // 000000005570: 07421BA1
	s_wait_dscnt 0x1                                           // 000000005574: BFC60001
	v_add_f32_e32 v162, v162, v14                              // 000000005578: 07441DA2
	s_wait_dscnt 0x0                                           // 00000000557C: BFC60000
	v_add_f32_e32 v163, v163, v15                              // 000000005580: 07461FA3
	v_cmp_gt_u32_e32 vcc_lo, 31, v0                            // 000000005584: 7C98009F
	s_wait_alu depctr_va_vcc(0)                                // 000000005588: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo                         // 00000000558C: D5010004 01A90280
	v_add_lshl_u32 v4, v4, v0, 2                               // 000000005594: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v144                               // 00000000559C: DACC0000 08009004
	ds_bpermute_b32 v9, v4, v145                               // 0000000055A4: DACC0000 09009104
	ds_bpermute_b32 v10, v4, v146                              // 0000000055AC: DACC0000 0A009204
	ds_bpermute_b32 v11, v4, v147                              // 0000000055B4: DACC0000 0B009304
	ds_bpermute_b32 v12, v4, v160                              // 0000000055BC: DACC0000 0C00A004
	ds_bpermute_b32 v13, v4, v161                              // 0000000055C4: DACC0000 0D00A104
	ds_bpermute_b32 v14, v4, v162                              // 0000000055CC: DACC0000 0E00A204
	ds_bpermute_b32 v15, v4, v163                              // 0000000055D4: DACC0000 0F00A304
	s_wait_dscnt 0x7                                           // 0000000055DC: BFC60007
	v_add_f32_e32 v144, v144, v8                               // 0000000055E0: 07201190
	s_wait_dscnt 0x6                                           // 0000000055E4: BFC60006
	v_add_f32_e32 v145, v145, v9                               // 0000000055E8: 07221391
	s_wait_dscnt 0x5                                           // 0000000055EC: BFC60005
	v_add_f32_e32 v146, v146, v10                              // 0000000055F0: 07241592
	s_wait_dscnt 0x4                                           // 0000000055F4: BFC60004
	v_add_f32_e32 v147, v147, v11                              // 0000000055F8: 07261793
	s_wait_dscnt 0x3                                           // 0000000055FC: BFC60003
	v_add_f32_e32 v160, v160, v12                              // 000000005600: 074019A0
	s_wait_dscnt 0x2                                           // 000000005604: BFC60002
	v_add_f32_e32 v161, v161, v13                              // 000000005608: 07421BA1
	s_wait_dscnt 0x1                                           // 00000000560C: BFC60001
	v_add_f32_e32 v162, v162, v14                              // 000000005610: 07441DA2
	s_wait_dscnt 0x0                                           // 000000005614: BFC60000
	v_add_f32_e32 v163, v163, v15                              // 000000005618: 07461FA3
	v_cmpx_eq_u32_e32 0, v0                                    // 00000000561C: 7D940080
	v_mov_b32_e32 v40, s42                                     // 000000005620: 7E50022A
	s_mul_i32 s40, s39, 1                                      // 000000005624: 96288127
	s_add_co_i32 s40, s40, s42                                 // 000000005628: 81282A28
	v_mov_b32_e32 v41, s40                                     // 00000000562C: 7E520228
	global_store_b32 v40, v144, s[8:9]                         // 000000005630: EE068008 48000000 00000028
	global_store_b32 v41, v160, s[8:9]                         // 00000000563C: EE068008 50000000 00000029
	s_cmp_le_i32 s13, 1                                        // 000000005648: BF05810D
	s_cbranch_scc1 22                                          // 00000000564C: BFA20016 <gemv_mq2g256v2_xbatch_pm+0x3fa8>
	global_store_b32 v40, v145, s[8:9] offset:4                // 000000005650: EE068008 48800000 00000428
	global_store_b32 v41, v161, s[8:9] offset:4                // 00000000565C: EE068008 50800000 00000429
	s_cmp_le_i32 s13, 2                                        // 000000005668: BF05820D
	s_cbranch_scc1 14                                          // 00000000566C: BFA2000E <gemv_mq2g256v2_xbatch_pm+0x3fa8>
	global_store_b32 v40, v146, s[8:9] offset:8                // 000000005670: EE068008 49000000 00000828
	global_store_b32 v41, v162, s[8:9] offset:8                // 00000000567C: EE068008 51000000 00000829
	s_cmp_le_i32 s13, 3                                        // 000000005688: BF05830D
	s_cbranch_scc1 6                                           // 00000000568C: BFA20006 <gemv_mq2g256v2_xbatch_pm+0x3fa8>
	global_store_b32 v40, v147, s[8:9] offset:12               // 000000005690: EE068008 49800000 00000C28
	global_store_b32 v41, v163, s[8:9] offset:12               // 00000000569C: EE068008 51800000 00000C29
	s_wait_storecnt 0x0                                        // 0000000056A8: BFC10000
	s_branch 5740                                              // 0000000056AC: BFA0166C <gemv_mq2g256v2_xbatch_pm+0x9960>
	v_mov_b32_e32 v176, 0                                      // 0000000056B0: 7F600280
	v_mov_b32_e32 v177, 0                                      // 0000000056B4: 7F620280
	v_mov_b32_e32 v178, 0                                      // 0000000056B8: 7F640280
	v_mov_b32_e32 v179, 0                                      // 0000000056BC: 7F660280
	v_mov_b32_e32 v180, 0                                      // 0000000056C0: 7F680280
	v_mov_b32_e32 v181, 0                                      // 0000000056C4: 7F6A0280
	v_mov_b32_e32 v182, 0                                      // 0000000056C8: 7F6C0280
	v_mov_b32_e32 v183, 0                                      // 0000000056CC: 7F6E0280
	v_mov_b32_e32 v184, 0                                      // 0000000056D0: 7F700280
	v_mov_b32_e32 v185, 0                                      // 0000000056D4: 7F720280
	v_mov_b32_e32 v186, 0                                      // 0000000056D8: 7F740280
	v_mov_b32_e32 v187, 0                                      // 0000000056DC: 7F760280
	v_mov_b32_e32 v188, 0                                      // 0000000056E0: 7F780280
	v_mov_b32_e32 v189, 0                                      // 0000000056E4: 7F7A0280
	v_mov_b32_e32 v190, 0                                      // 0000000056E8: 7F7C0280
	v_mov_b32_e32 v191, 0                                      // 0000000056EC: 7F7E0280
	v_mov_b32_e32 v192, 0                                      // 0000000056F0: 7F800280
	v_mov_b32_e32 v193, 0                                      // 0000000056F4: 7F820280
	v_mov_b32_e32 v194, 0                                      // 0000000056F8: 7F840280
	v_mov_b32_e32 v195, 0                                      // 0000000056FC: 7F860280
	v_mov_b32_e32 v196, 0                                      // 000000005700: 7F880280
	v_mov_b32_e32 v197, 0                                      // 000000005704: 7F8A0280
	v_mov_b32_e32 v198, 0                                      // 000000005708: 7F8C0280
	v_mov_b32_e32 v199, 0                                      // 00000000570C: 7F8E0280
	v_mov_b32_e32 v200, 0                                      // 000000005710: 7F900280
	v_mov_b32_e32 v201, 0                                      // 000000005714: 7F920280
	v_mov_b32_e32 v202, 0                                      // 000000005718: 7F940280
	v_mov_b32_e32 v203, 0                                      // 00000000571C: 7F960280
	v_mov_b32_e32 v204, 0                                      // 000000005720: 7F980280
	v_mov_b32_e32 v205, 0                                      // 000000005724: 7F9A0280
	v_mov_b32_e32 v206, 0                                      // 000000005728: 7F9C0280
	v_mov_b32_e32 v207, 0                                      // 00000000572C: 7F9E0280
	v_mov_b32_e32 v208, 0                                      // 000000005730: 7FA00280
	v_mov_b32_e32 v209, 0                                      // 000000005734: 7FA20280
	v_mov_b32_e32 v210, 0                                      // 000000005738: 7FA40280
	v_mov_b32_e32 v211, 0                                      // 00000000573C: 7FA60280
	v_mov_b32_e32 v212, 0                                      // 000000005740: 7FA80280
	v_mov_b32_e32 v213, 0                                      // 000000005744: 7FAA0280
	v_mov_b32_e32 v214, 0                                      // 000000005748: 7FAC0280
	v_mov_b32_e32 v215, 0                                      // 00000000574C: 7FAE0280
	v_mov_b32_e32 v216, 0                                      // 000000005750: 7FB00280
	v_mov_b32_e32 v217, 0                                      // 000000005754: 7FB20280
	v_mov_b32_e32 v218, 0                                      // 000000005758: 7FB40280
	v_mov_b32_e32 v219, 0                                      // 00000000575C: 7FB60280
	v_mov_b32_e32 v220, 0                                      // 000000005760: 7FB80280
	v_mov_b32_e32 v221, 0                                      // 000000005764: 7FBA0280
	v_mov_b32_e32 v222, 0                                      // 000000005768: 7FBC0280
	v_mov_b32_e32 v223, 0                                      // 00000000576C: 7FBE0280
	s_lshr_b32 s36, s14, 2                                     // 000000005770: 8524820E
	s_and_b32 s37, s14, 3                                      // 000000005774: 8B25830E
	s_cmp_eq_u32 s36, 0                                        // 000000005778: BF068024
	s_cbranch_scc1 1285                                        // 00000000577C: BFA20505 <gemv_mq2g256v2_xbatch_pm+0x5494>
	buffer_load_b32 v8, v1, s[20:23], s16 offen                // 000000005780: C4050010 40802808 00000001
	buffer_load_b32 v9, v1, s[20:23], s17 offen                // 00000000578C: C4050011 40802809 00000001
	buffer_load_b32 v10, v1, s[20:23], s18 offen               // 000000005798: C4050012 4080280A 00000001
	buffer_load_b32 v11, v1, s[20:23], s19 offen               // 0000000057A4: C4050013 4080280B 00000001
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8      // 0000000057B0: C4050010 40802818 00000802
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8      // 0000000057BC: C4050011 40802819 00000802
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8      // 0000000057C8: C4050012 4080281A 00000802
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8      // 0000000057D4: C4050013 4080281B 00000802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen         // 0000000057E0: C405C01C 40803050 00000003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16// 0000000057EC: C405C01C 40803054 00001003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen         // 0000000057F8: C405C01D 40803058 00000003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16// 000000005804: C405C01D 4080305C 00001003
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen         // 000000005810: C405C01E 40803060 00000003
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16// 00000000581C: C405C01E 40803064 00001003
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:72     // 000000005828: C4050010 4080280C 00004801
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:72     // 000000005834: C4050011 4080280D 00004801
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:72     // 000000005840: C4050012 4080280E 00004801
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:72     // 00000000584C: C4050013 4080280F 00004801
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:80     // 000000005858: C4050010 4080281C 00005002
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:80     // 000000005864: C4050011 4080281D 00005002
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:80     // 000000005870: C4050012 4080281E 00005002
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:80     // 00000000587C: C4050013 4080281F 00005002
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:144    // 000000005888: C4050010 40802810 00009001
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:144    // 000000005894: C4050011 40802811 00009001
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:144    // 0000000058A0: C4050012 40802812 00009001
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:144    // 0000000058AC: C4050013 40802813 00009001
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:152    // 0000000058B8: C4050010 40802820 00009802
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:152    // 0000000058C4: C4050011 40802821 00009802
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:152    // 0000000058D0: C4050012 40802822 00009802
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:152    // 0000000058DC: C4050013 40802823 00009802
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:216    // 0000000058E8: C4050010 40802814 0000D801
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:216    // 0000000058F4: C4050011 40802815 0000D801
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:216    // 000000005900: C4050012 40802816 0000D801
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:216    // 00000000590C: C4050013 40802817 0000D801
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:224    // 000000005918: C4050010 40802824 0000E002
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:224    // 000000005924: C4050011 40802825 0000E002
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:224    // 000000005930: C4050012 40802826 0000E002
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:224    // 00000000593C: C4050013 40802827 0000E002
	s_wait_loadcnt 0x21                                        // 000000005948: BFC00021
	v_bfe_u32 v40, v24, 0, 2                                   // 00000000594C: D6100028 02090118
	s_wait_loadcnt 0x20                                        // 000000005954: BFC00020
	v_bfe_u32 v50, v25, 0, 2                                   // 000000005958: D6100032 02090119
	s_wait_loadcnt 0x1f                                        // 000000005960: BFC0001F
	v_bfe_u32 v60, v26, 0, 2                                   // 000000005964: D610003C 0209011A
	s_wait_loadcnt 0x1e                                        // 00000000596C: BFC0001E
	v_bfe_u32 v70, v27, 0, 2                                   // 000000005970: D6100046 0209011B
	v_bfe_u32 v41, v24, 2, 2                                   // 000000005978: D6100029 02090518
	v_bfe_u32 v51, v25, 2, 2                                   // 000000005980: D6100033 02090519
	v_bfe_u32 v61, v26, 2, 2                                   // 000000005988: D610003D 0209051A
	v_bfe_u32 v71, v27, 2, 2                                   // 000000005990: D6100047 0209051B
	v_bfe_u32 v42, v24, 4, 2                                   // 000000005998: D610002A 02090918
	v_bfe_u32 v52, v25, 4, 2                                   // 0000000059A0: D6100034 02090919
	v_bfe_u32 v62, v26, 4, 2                                   // 0000000059A8: D610003E 0209091A
	v_bfe_u32 v72, v27, 4, 2                                   // 0000000059B0: D6100048 0209091B
	v_bfe_u32 v43, v24, 6, 2                                   // 0000000059B8: D610002B 02090D18
	v_bfe_u32 v53, v25, 6, 2                                   // 0000000059C0: D6100035 02090D19
	v_bfe_u32 v63, v26, 6, 2                                   // 0000000059C8: D610003F 02090D1A
	v_bfe_u32 v73, v27, 6, 2                                   // 0000000059D0: D6100049 02090D1B
	v_bfe_u32 v44, v24, 8, 2                                   // 0000000059D8: D610002C 02091118
	v_bfe_u32 v54, v25, 8, 2                                   // 0000000059E0: D6100036 02091119
	v_bfe_u32 v64, v26, 8, 2                                   // 0000000059E8: D6100040 0209111A
	v_bfe_u32 v74, v27, 8, 2                                   // 0000000059F0: D610004A 0209111B
	v_bfe_u32 v45, v24, 10, 2                                  // 0000000059F8: D610002D 02091518
	v_bfe_u32 v55, v25, 10, 2                                  // 000000005A00: D6100037 02091519
	v_bfe_u32 v65, v26, 10, 2                                  // 000000005A08: D6100041 0209151A
	v_bfe_u32 v75, v27, 10, 2                                  // 000000005A10: D610004B 0209151B
	v_bfe_u32 v46, v24, 12, 2                                  // 000000005A18: D610002E 02091918
	v_bfe_u32 v56, v25, 12, 2                                  // 000000005A20: D6100038 02091919
	v_bfe_u32 v66, v26, 12, 2                                  // 000000005A28: D6100042 0209191A
	v_bfe_u32 v76, v27, 12, 2                                  // 000000005A30: D610004C 0209191B
	v_bfe_u32 v47, v24, 14, 2                                  // 000000005A38: D610002F 02091D18
	v_bfe_u32 v57, v25, 14, 2                                  // 000000005A40: D6100039 02091D19
	v_bfe_u32 v67, v26, 14, 2                                  // 000000005A48: D6100043 02091D1A
	v_bfe_u32 v77, v27, 14, 2                                  // 000000005A50: D610004D 02091D1B
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000005A58: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000005A5C: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000005A60: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000005A64: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000005A68: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000005A6C: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000005A70: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000005A74: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000005A78: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000005A7C: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000005A80: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000005A84: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000005A88: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000005A8C: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000005A90: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000005A94: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000005A98: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000005A9C: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000005AA0: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000005AA4: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000005AA8: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000005AAC: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000005AB0: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000005AB4: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000005AB8: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000005ABC: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000005AC0: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000005AC4: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000005AC8: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000005ACC: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000005AD0: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000005AD4: 7E9A234D
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005AD8: CC206028 0C225108
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005AE0: CC206032 0C266509
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005AE8: CC20603C 0C2A790A
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005AF0: CC206046 0C2E8D0B
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005AF8: CC206029 0C225308
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B00: CC206033 0C266709
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B08: CC20603D 0C2A7B0A
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B10: CC206047 0C2E8F0B
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B18: CC20602A 0C225508
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B20: CC206034 0C266909
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B28: CC20603E 0C2A7D0A
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B30: CC206048 0C2E910B
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B38: CC20602B 0C225708
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B40: CC206035 0C266B09
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B48: CC20603F 0C2A7F0A
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B50: CC206049 0C2E930B
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B58: CC20602C 0C225908
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B60: CC206036 0C266D09
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B68: CC206040 0C2A810A
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B70: CC20604A 0C2E950B
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B78: CC20602D 0C225B08
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B80: CC206037 0C266F09
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B88: CC206041 0C2A830A
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B90: CC20604B 0C2E970B
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005B98: CC20602E 0C225D08
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005BA0: CC206038 0C267109
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005BA8: CC206042 0C2A850A
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005BB0: CC20604C 0C2E990B
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005BB8: CC20602F 0C225F08
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005BC0: CC206039 0C267309
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005BC8: CC206043 0C2A870A
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000005BD0: CC20604D 0C2E9B0B
	buffer_load_b128 v[104:107], v3, s[24:27], s28 offen offset:1024// 000000005BD8: C405C01C 40803068 00040003
	buffer_load_b128 v[108:111], v3, s[24:27], s28 offen offset:1040// 000000005BE4: C405C01C 4080306C 00041003
	buffer_load_b128 v[112:115], v3, s[24:27], s29 offen offset:1024// 000000005BF0: C405C01D 40803070 00040003
	buffer_load_b128 v[116:119], v3, s[24:27], s29 offen offset:1040// 000000005BFC: C405C01D 40803074 00041003
	buffer_load_b128 v[120:123], v3, s[24:27], s30 offen offset:1024// 000000005C08: C405C01E 40803078 00040003
	buffer_load_b128 v[124:127], v3, s[24:27], s30 offen offset:1040// 000000005C14: C405C01E 4080307C 00041003
	buffer_load_b128 v[128:131], v3, s[24:27], s28 offen offset:2048// 000000005C20: C405C01C 40803080 00080003
	buffer_load_b128 v[132:135], v3, s[24:27], s28 offen offset:2064// 000000005C2C: C405C01C 40803084 00081003
	buffer_load_b128 v[136:139], v3, s[24:27], s29 offen offset:2048// 000000005C38: C405C01D 40803088 00080003
	buffer_load_b128 v[140:143], v3, s[24:27], s29 offen offset:2064// 000000005C44: C405C01D 4080308C 00081003
	buffer_load_b128 v[144:147], v3, s[24:27], s30 offen offset:2048// 000000005C50: C405C01E 40803090 00080003
	buffer_load_b128 v[148:151], v3, s[24:27], s30 offen offset:2064// 000000005C5C: C405C01E 40803094 00081003
	buffer_load_b128 v[152:155], v3, s[24:27], s28 offen offset:3072// 000000005C68: C405C01C 40803098 000C0003
	buffer_load_b128 v[156:159], v3, s[24:27], s28 offen offset:3088// 000000005C74: C405C01C 4080309C 000C1003
	buffer_load_b128 v[160:163], v3, s[24:27], s29 offen offset:3072// 000000005C80: C405C01D 408030A0 000C0003
	buffer_load_b128 v[164:167], v3, s[24:27], s29 offen offset:3088// 000000005C8C: C405C01D 408030A4 000C1003
	buffer_load_b128 v[168:171], v3, s[24:27], s30 offen offset:3072// 000000005C98: C405C01E 408030A8 000C0003
	buffer_load_b128 v[172:175], v3, s[24:27], s30 offen offset:3088// 000000005CA4: C405C01E 408030AC 000C1003
	s_wait_loadcnt 0x2f                                        // 000000005CB0: BFC0002F
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 000000005CB4: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 000000005CBC: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x2d                                        // 000000005CC4: BFC0002D
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 000000005CC8: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 000000005CD0: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000005CD8: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000005CE0: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 000000005CE8: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 000000005CF0: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000005CF8: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000005D00: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 000000005D08: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 000000005D10: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000005D18: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000005D20: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 000000005D28: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 000000005D30: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000005D38: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000005D40: C800A940 3A3AA94A
	s_wait_loadcnt 0x2c                                        // 000000005D48: BFC0002C
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 000000005D4C: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 000000005D54: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000005D5C: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000005D64: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 000000005D6C: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 000000005D74: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 000000005D7C: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000005D84: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 000000005D8C: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 000000005D94: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 000000005D9C: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 000000005DA4: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 000000005DAC: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 000000005DB4: C800BF43 4E4EBF4D
	v_dual_add_f32 v176, v176, v48 :: v_dual_add_f32 v177, v177, v49// 000000005DBC: C90861B0 B0B063B1
	v_dual_add_f32 v178, v178, v58 :: v_dual_add_f32 v179, v179, v59// 000000005DC4: C90875B2 B2B277B3
	v_dual_add_f32 v192, v192, v68 :: v_dual_add_f32 v193, v193, v69// 000000005DCC: C90889C0 C0C08BC1
	v_dual_add_f32 v194, v194, v78 :: v_dual_add_f32 v195, v195, v79// 000000005DD4: C9089DC2 C2C29FC3
	s_wait_loadcnt 0x2b                                        // 000000005DDC: BFC0002B
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97// 000000005DE0: C8C6C329 3030C333
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97// 000000005DE8: C8C6C33D 3A3AC347
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96// 000000005DF0: C800C128 3030C132
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96// 000000005DF8: C800C13C 3A3AC146
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98// 000000005E00: C800C52A 3030C534
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98// 000000005E08: C800C53E 3A3AC548
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99// 000000005E10: C800C72B 3030C735
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99// 000000005E18: C800C73F 3A3AC749
	s_wait_loadcnt 0x2a                                        // 000000005E20: BFC0002A
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100// 000000005E24: C800C92C 3030C936
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100// 000000005E2C: C800C940 3A3AC94A
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101// 000000005E34: C800CB2D 3030CB37
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101// 000000005E3C: C800CB41 3A3ACB4B
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102// 000000005E44: C800CD2E 3030CD38
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102// 000000005E4C: C800CD42 3A3ACD4C
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103// 000000005E54: C800CF2F 3030CF39
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103// 000000005E5C: C800CF43 3A3ACF4D
	v_dual_add_f32 v208, v208, v48 :: v_dual_add_f32 v209, v209, v49// 000000005E64: C90861D0 D0D063D1
	v_dual_add_f32 v210, v210, v58 :: v_dual_add_f32 v211, v211, v59// 000000005E6C: C90875D2 D2D277D3
	s_wait_loadcnt 0x25                                        // 000000005E74: BFC00025
	v_bfe_u32 v40, v28, 0, 2                                   // 000000005E78: D6100028 0209011C
	s_wait_loadcnt 0x24                                        // 000000005E80: BFC00024
	v_bfe_u32 v50, v29, 0, 2                                   // 000000005E84: D6100032 0209011D
	s_wait_loadcnt 0x23                                        // 000000005E8C: BFC00023
	v_bfe_u32 v60, v30, 0, 2                                   // 000000005E90: D610003C 0209011E
	s_wait_loadcnt 0x22                                        // 000000005E98: BFC00022
	v_bfe_u32 v70, v31, 0, 2                                   // 000000005E9C: D6100046 0209011F
	v_bfe_u32 v41, v28, 2, 2                                   // 000000005EA4: D6100029 0209051C
	v_bfe_u32 v51, v29, 2, 2                                   // 000000005EAC: D6100033 0209051D
	v_bfe_u32 v61, v30, 2, 2                                   // 000000005EB4: D610003D 0209051E
	v_bfe_u32 v71, v31, 2, 2                                   // 000000005EBC: D6100047 0209051F
	v_bfe_u32 v42, v28, 4, 2                                   // 000000005EC4: D610002A 0209091C
	v_bfe_u32 v52, v29, 4, 2                                   // 000000005ECC: D6100034 0209091D
	v_bfe_u32 v62, v30, 4, 2                                   // 000000005ED4: D610003E 0209091E
	v_bfe_u32 v72, v31, 4, 2                                   // 000000005EDC: D6100048 0209091F
	v_bfe_u32 v43, v28, 6, 2                                   // 000000005EE4: D610002B 02090D1C
	v_bfe_u32 v53, v29, 6, 2                                   // 000000005EEC: D6100035 02090D1D
	v_bfe_u32 v63, v30, 6, 2                                   // 000000005EF4: D610003F 02090D1E
	v_bfe_u32 v73, v31, 6, 2                                   // 000000005EFC: D6100049 02090D1F
	v_bfe_u32 v44, v28, 8, 2                                   // 000000005F04: D610002C 0209111C
	v_bfe_u32 v54, v29, 8, 2                                   // 000000005F0C: D6100036 0209111D
	v_bfe_u32 v64, v30, 8, 2                                   // 000000005F14: D6100040 0209111E
	v_bfe_u32 v74, v31, 8, 2                                   // 000000005F1C: D610004A 0209111F
	v_bfe_u32 v45, v28, 10, 2                                  // 000000005F24: D610002D 0209151C
	v_bfe_u32 v55, v29, 10, 2                                  // 000000005F2C: D6100037 0209151D
	v_bfe_u32 v65, v30, 10, 2                                  // 000000005F34: D6100041 0209151E
	v_bfe_u32 v75, v31, 10, 2                                  // 000000005F3C: D610004B 0209151F
	v_bfe_u32 v46, v28, 12, 2                                  // 000000005F44: D610002E 0209191C
	v_bfe_u32 v56, v29, 12, 2                                  // 000000005F4C: D6100038 0209191D
	v_bfe_u32 v66, v30, 12, 2                                  // 000000005F54: D6100042 0209191E
	v_bfe_u32 v76, v31, 12, 2                                  // 000000005F5C: D610004C 0209191F
	v_bfe_u32 v47, v28, 14, 2                                  // 000000005F64: D610002F 02091D1C
	v_bfe_u32 v57, v29, 14, 2                                  // 000000005F6C: D6100039 02091D1D
	v_bfe_u32 v67, v30, 14, 2                                  // 000000005F74: D6100043 02091D1E
	v_bfe_u32 v77, v31, 14, 2                                  // 000000005F7C: D610004D 02091D1F
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000005F84: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000005F88: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000005F8C: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000005F90: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000005F94: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000005F98: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000005F9C: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000005FA0: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000005FA4: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000005FA8: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000005FAC: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000005FB0: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000005FB4: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000005FB8: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000005FBC: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000005FC0: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000005FC4: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000005FC8: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000005FCC: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000005FD0: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000005FD4: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000005FD8: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000005FDC: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000005FE0: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000005FE4: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000005FE8: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000005FEC: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000005FF0: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000005FF4: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000005FF8: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000005FFC: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000006000: 7E9A234D
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006004: CC206028 0C32510C
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000600C: CC206032 0C36650D
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006014: CC20603C 0C3A790E
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000601C: CC206046 0C3E8D0F
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006024: CC206029 0C32530C
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000602C: CC206033 0C36670D
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006034: CC20603D 0C3A7B0E
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000603C: CC206047 0C3E8F0F
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006044: CC20602A 0C32550C
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000604C: CC206034 0C36690D
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006054: CC20603E 0C3A7D0E
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000605C: CC206048 0C3E910F
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006064: CC20602B 0C32570C
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000606C: CC206035 0C366B0D
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006074: CC20603F 0C3A7F0E
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000607C: CC206049 0C3E930F
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006084: CC20602C 0C32590C
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000608C: CC206036 0C366D0D
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006094: CC206040 0C3A810E
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000609C: CC20604A 0C3E950F
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060A4: CC20602D 0C325B0C
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060AC: CC206037 0C366F0D
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060B4: CC206041 0C3A830E
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060BC: CC20604B 0C3E970F
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060C4: CC20602E 0C325D0C
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060CC: CC206038 0C36710D
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060D4: CC206042 0C3A850E
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060DC: CC20604C 0C3E990F
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060E4: CC20602F 0C325F0C
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060EC: CC206039 0C36730D
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060F4: CC206043 0C3A870E
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000060FC: CC20604D 0C3E9B0F
	s_wait_loadcnt 0x11                                        // 000000006104: BFC00011
	v_dual_mul_f32 v48, v41, v105 :: v_dual_mul_f32 v49, v51, v105// 000000006108: C8C6D329 3030D333
	v_dual_mul_f32 v58, v61, v105 :: v_dual_mul_f32 v59, v71, v105// 000000006110: C8C6D33D 3A3AD347
	s_wait_loadcnt 0xf                                         // 000000006118: BFC0000F
	v_dual_mul_f32 v68, v41, v113 :: v_dual_mul_f32 v69, v51, v113// 00000000611C: C8C6E329 4444E333
	v_dual_mul_f32 v78, v61, v113 :: v_dual_mul_f32 v79, v71, v113// 000000006124: C8C6E33D 4E4EE347
	v_dual_fmac_f32 v48, v40, v104 :: v_dual_fmac_f32 v49, v50, v104// 00000000612C: C800D128 3030D132
	v_dual_fmac_f32 v58, v60, v104 :: v_dual_fmac_f32 v59, v70, v104// 000000006134: C800D13C 3A3AD146
	v_dual_fmac_f32 v68, v40, v112 :: v_dual_fmac_f32 v69, v50, v112// 00000000613C: C800E128 4444E132
	v_dual_fmac_f32 v78, v60, v112 :: v_dual_fmac_f32 v79, v70, v112// 000000006144: C800E13C 4E4EE146
	v_dual_fmac_f32 v48, v42, v106 :: v_dual_fmac_f32 v49, v52, v106// 00000000614C: C800D52A 3030D534
	v_dual_fmac_f32 v58, v62, v106 :: v_dual_fmac_f32 v59, v72, v106// 000000006154: C800D53E 3A3AD548
	v_dual_fmac_f32 v68, v42, v114 :: v_dual_fmac_f32 v69, v52, v114// 00000000615C: C800E52A 4444E534
	v_dual_fmac_f32 v78, v62, v114 :: v_dual_fmac_f32 v79, v72, v114// 000000006164: C800E53E 4E4EE548
	v_dual_fmac_f32 v48, v43, v107 :: v_dual_fmac_f32 v49, v53, v107// 00000000616C: C800D72B 3030D735
	v_dual_fmac_f32 v58, v63, v107 :: v_dual_fmac_f32 v59, v73, v107// 000000006174: C800D73F 3A3AD749
	v_dual_fmac_f32 v68, v43, v115 :: v_dual_fmac_f32 v69, v53, v115// 00000000617C: C800E72B 4444E735
	v_dual_fmac_f32 v78, v63, v115 :: v_dual_fmac_f32 v79, v73, v115// 000000006184: C800E73F 4E4EE749
	v_dual_fmac_f32 v48, v44, v108 :: v_dual_fmac_f32 v49, v54, v108// 00000000618C: C800D92C 3030D936
	v_dual_fmac_f32 v58, v64, v108 :: v_dual_fmac_f32 v59, v74, v108// 000000006194: C800D940 3A3AD94A
	s_wait_loadcnt 0xe                                         // 00000000619C: BFC0000E
	v_dual_fmac_f32 v68, v44, v116 :: v_dual_fmac_f32 v69, v54, v116// 0000000061A0: C800E92C 4444E936
	v_dual_fmac_f32 v78, v64, v116 :: v_dual_fmac_f32 v79, v74, v116// 0000000061A8: C800E940 4E4EE94A
	v_dual_fmac_f32 v48, v45, v109 :: v_dual_fmac_f32 v49, v55, v109// 0000000061B0: C800DB2D 3030DB37
	v_dual_fmac_f32 v58, v65, v109 :: v_dual_fmac_f32 v59, v75, v109// 0000000061B8: C800DB41 3A3ADB4B
	v_dual_fmac_f32 v68, v45, v117 :: v_dual_fmac_f32 v69, v55, v117// 0000000061C0: C800EB2D 4444EB37
	v_dual_fmac_f32 v78, v65, v117 :: v_dual_fmac_f32 v79, v75, v117// 0000000061C8: C800EB41 4E4EEB4B
	v_dual_fmac_f32 v48, v46, v110 :: v_dual_fmac_f32 v49, v56, v110// 0000000061D0: C800DD2E 3030DD38
	v_dual_fmac_f32 v58, v66, v110 :: v_dual_fmac_f32 v59, v76, v110// 0000000061D8: C800DD42 3A3ADD4C
	v_dual_fmac_f32 v68, v46, v118 :: v_dual_fmac_f32 v69, v56, v118// 0000000061E0: C800ED2E 4444ED38
	v_dual_fmac_f32 v78, v66, v118 :: v_dual_fmac_f32 v79, v76, v118// 0000000061E8: C800ED42 4E4EED4C
	v_dual_fmac_f32 v48, v47, v111 :: v_dual_fmac_f32 v49, v57, v111// 0000000061F0: C800DF2F 3030DF39
	v_dual_fmac_f32 v58, v67, v111 :: v_dual_fmac_f32 v59, v77, v111// 0000000061F8: C800DF43 3A3ADF4D
	v_dual_fmac_f32 v68, v47, v119 :: v_dual_fmac_f32 v69, v57, v119// 000000006200: C800EF2F 4444EF39
	v_dual_fmac_f32 v78, v67, v119 :: v_dual_fmac_f32 v79, v77, v119// 000000006208: C800EF43 4E4EEF4D
	v_dual_add_f32 v180, v180, v48 :: v_dual_add_f32 v181, v181, v49// 000000006210: C90861B4 B4B463B5
	v_dual_add_f32 v182, v182, v58 :: v_dual_add_f32 v183, v183, v59// 000000006218: C90875B6 B6B677B7
	v_dual_add_f32 v196, v196, v68 :: v_dual_add_f32 v197, v197, v69// 000000006220: C90889C4 C4C48BC5
	v_dual_add_f32 v198, v198, v78 :: v_dual_add_f32 v199, v199, v79// 000000006228: C9089DC6 C6C69FC7
	s_wait_loadcnt 0xd                                         // 000000006230: BFC0000D
	v_dual_mul_f32 v48, v41, v121 :: v_dual_mul_f32 v49, v51, v121// 000000006234: C8C6F329 3030F333
	v_dual_mul_f32 v58, v61, v121 :: v_dual_mul_f32 v59, v71, v121// 00000000623C: C8C6F33D 3A3AF347
	v_dual_fmac_f32 v48, v40, v120 :: v_dual_fmac_f32 v49, v50, v120// 000000006244: C800F128 3030F132
	v_dual_fmac_f32 v58, v60, v120 :: v_dual_fmac_f32 v59, v70, v120// 00000000624C: C800F13C 3A3AF146
	v_dual_fmac_f32 v48, v42, v122 :: v_dual_fmac_f32 v49, v52, v122// 000000006254: C800F52A 3030F534
	v_dual_fmac_f32 v58, v62, v122 :: v_dual_fmac_f32 v59, v72, v122// 00000000625C: C800F53E 3A3AF548
	v_dual_fmac_f32 v48, v43, v123 :: v_dual_fmac_f32 v49, v53, v123// 000000006264: C800F72B 3030F735
	v_dual_fmac_f32 v58, v63, v123 :: v_dual_fmac_f32 v59, v73, v123// 00000000626C: C800F73F 3A3AF749
	s_wait_loadcnt 0xc                                         // 000000006274: BFC0000C
	v_dual_fmac_f32 v48, v44, v124 :: v_dual_fmac_f32 v49, v54, v124// 000000006278: C800F92C 3030F936
	v_dual_fmac_f32 v58, v64, v124 :: v_dual_fmac_f32 v59, v74, v124// 000000006280: C800F940 3A3AF94A
	v_dual_fmac_f32 v48, v45, v125 :: v_dual_fmac_f32 v49, v55, v125// 000000006288: C800FB2D 3030FB37
	v_dual_fmac_f32 v58, v65, v125 :: v_dual_fmac_f32 v59, v75, v125// 000000006290: C800FB41 3A3AFB4B
	v_dual_fmac_f32 v48, v46, v126 :: v_dual_fmac_f32 v49, v56, v126// 000000006298: C800FD2E 3030FD38
	v_dual_fmac_f32 v58, v66, v126 :: v_dual_fmac_f32 v59, v76, v126// 0000000062A0: C800FD42 3A3AFD4C
	v_dual_fmac_f32 v48, v47, v127 :: v_dual_fmac_f32 v49, v57, v127// 0000000062A8: C800FF2F 3030FF39
	v_dual_fmac_f32 v58, v67, v127 :: v_dual_fmac_f32 v59, v77, v127// 0000000062B0: C800FF43 3A3AFF4D
	v_dual_add_f32 v212, v212, v48 :: v_dual_add_f32 v213, v213, v49// 0000000062B8: C90861D4 D4D463D5
	v_dual_add_f32 v214, v214, v58 :: v_dual_add_f32 v215, v215, v59// 0000000062C0: C90875D6 D6D677D7
	v_bfe_u32 v40, v32, 0, 2                                   // 0000000062C8: D6100028 02090120
	v_bfe_u32 v50, v33, 0, 2                                   // 0000000062D0: D6100032 02090121
	v_bfe_u32 v60, v34, 0, 2                                   // 0000000062D8: D610003C 02090122
	v_bfe_u32 v70, v35, 0, 2                                   // 0000000062E0: D6100046 02090123
	v_bfe_u32 v41, v32, 2, 2                                   // 0000000062E8: D6100029 02090520
	v_bfe_u32 v51, v33, 2, 2                                   // 0000000062F0: D6100033 02090521
	v_bfe_u32 v61, v34, 2, 2                                   // 0000000062F8: D610003D 02090522
	v_bfe_u32 v71, v35, 2, 2                                   // 000000006300: D6100047 02090523
	v_bfe_u32 v42, v32, 4, 2                                   // 000000006308: D610002A 02090920
	v_bfe_u32 v52, v33, 4, 2                                   // 000000006310: D6100034 02090921
	v_bfe_u32 v62, v34, 4, 2                                   // 000000006318: D610003E 02090922
	v_bfe_u32 v72, v35, 4, 2                                   // 000000006320: D6100048 02090923
	v_bfe_u32 v43, v32, 6, 2                                   // 000000006328: D610002B 02090D20
	v_bfe_u32 v53, v33, 6, 2                                   // 000000006330: D6100035 02090D21
	v_bfe_u32 v63, v34, 6, 2                                   // 000000006338: D610003F 02090D22
	v_bfe_u32 v73, v35, 6, 2                                   // 000000006340: D6100049 02090D23
	v_bfe_u32 v44, v32, 8, 2                                   // 000000006348: D610002C 02091120
	v_bfe_u32 v54, v33, 8, 2                                   // 000000006350: D6100036 02091121
	v_bfe_u32 v64, v34, 8, 2                                   // 000000006358: D6100040 02091122
	v_bfe_u32 v74, v35, 8, 2                                   // 000000006360: D610004A 02091123
	v_bfe_u32 v45, v32, 10, 2                                  // 000000006368: D610002D 02091520
	v_bfe_u32 v55, v33, 10, 2                                  // 000000006370: D6100037 02091521
	v_bfe_u32 v65, v34, 10, 2                                  // 000000006378: D6100041 02091522
	v_bfe_u32 v75, v35, 10, 2                                  // 000000006380: D610004B 02091523
	v_bfe_u32 v46, v32, 12, 2                                  // 000000006388: D610002E 02091920
	v_bfe_u32 v56, v33, 12, 2                                  // 000000006390: D6100038 02091921
	v_bfe_u32 v66, v34, 12, 2                                  // 000000006398: D6100042 02091922
	v_bfe_u32 v76, v35, 12, 2                                  // 0000000063A0: D610004C 02091923
	v_bfe_u32 v47, v32, 14, 2                                  // 0000000063A8: D610002F 02091D20
	v_bfe_u32 v57, v33, 14, 2                                  // 0000000063B0: D6100039 02091D21
	v_bfe_u32 v67, v34, 14, 2                                  // 0000000063B8: D6100043 02091D22
	v_bfe_u32 v77, v35, 14, 2                                  // 0000000063C0: D610004D 02091D23
	v_cvt_f32_ubyte0_e32 v40, v40                              // 0000000063C8: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 0000000063CC: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 0000000063D0: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 0000000063D4: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 0000000063D8: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 0000000063DC: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 0000000063E0: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 0000000063E4: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 0000000063E8: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 0000000063EC: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 0000000063F0: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 0000000063F4: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 0000000063F8: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 0000000063FC: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000006400: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000006404: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000006408: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 00000000640C: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000006410: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000006414: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000006418: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 00000000641C: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000006420: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000006424: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000006428: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 00000000642C: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000006430: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000006434: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000006438: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 00000000643C: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000006440: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000006444: 7E9A234D
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006448: CC206028 0C425110
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006450: CC206032 0C466511
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006458: CC20603C 0C4A7912
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006460: CC206046 0C4E8D13
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006468: CC206029 0C425310
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006470: CC206033 0C466711
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006478: CC20603D 0C4A7B12
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006480: CC206047 0C4E8F13
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006488: CC20602A 0C425510
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006490: CC206034 0C466911
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006498: CC20603E 0C4A7D12
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064A0: CC206048 0C4E9113
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064A8: CC20602B 0C425710
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064B0: CC206035 0C466B11
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064B8: CC20603F 0C4A7F12
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064C0: CC206049 0C4E9313
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064C8: CC20602C 0C425910
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064D0: CC206036 0C466D11
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064D8: CC206040 0C4A8112
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064E0: CC20604A 0C4E9513
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064E8: CC20602D 0C425B10
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064F0: CC206037 0C466F11
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000064F8: CC206041 0C4A8312
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006500: CC20604B 0C4E9713
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006508: CC20602E 0C425D10
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006510: CC206038 0C467111
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006518: CC206042 0C4A8512
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006520: CC20604C 0C4E9913
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006528: CC20602F 0C425F10
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006530: CC206039 0C467311
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006538: CC206043 0C4A8712
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006540: CC20604D 0C4E9B13
	s_wait_loadcnt 0xb                                         // 000000006548: BFC0000B
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129// 00000000654C: C8C70329 30310333
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129// 000000006554: C8C7033D 3A3B0347
	s_wait_loadcnt 0x9                                         // 00000000655C: BFC00009
	v_dual_mul_f32 v68, v41, v137 :: v_dual_mul_f32 v69, v51, v137// 000000006560: C8C71329 44451333
	v_dual_mul_f32 v78, v61, v137 :: v_dual_mul_f32 v79, v71, v137// 000000006568: C8C7133D 4E4F1347
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128// 000000006570: C8010128 30310132
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128// 000000006578: C801013C 3A3B0146
	v_dual_fmac_f32 v68, v40, v136 :: v_dual_fmac_f32 v69, v50, v136// 000000006580: C8011128 44451132
	v_dual_fmac_f32 v78, v60, v136 :: v_dual_fmac_f32 v79, v70, v136// 000000006588: C801113C 4E4F1146
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130// 000000006590: C801052A 30310534
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130// 000000006598: C801053E 3A3B0548
	v_dual_fmac_f32 v68, v42, v138 :: v_dual_fmac_f32 v69, v52, v138// 0000000065A0: C801152A 44451534
	v_dual_fmac_f32 v78, v62, v138 :: v_dual_fmac_f32 v79, v72, v138// 0000000065A8: C801153E 4E4F1548
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131// 0000000065B0: C801072B 30310735
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131// 0000000065B8: C801073F 3A3B0749
	v_dual_fmac_f32 v68, v43, v139 :: v_dual_fmac_f32 v69, v53, v139// 0000000065C0: C801172B 44451735
	v_dual_fmac_f32 v78, v63, v139 :: v_dual_fmac_f32 v79, v73, v139// 0000000065C8: C801173F 4E4F1749
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132// 0000000065D0: C801092C 30310936
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132// 0000000065D8: C8010940 3A3B094A
	s_wait_loadcnt 0x8                                         // 0000000065E0: BFC00008
	v_dual_fmac_f32 v68, v44, v140 :: v_dual_fmac_f32 v69, v54, v140// 0000000065E4: C801192C 44451936
	v_dual_fmac_f32 v78, v64, v140 :: v_dual_fmac_f32 v79, v74, v140// 0000000065EC: C8011940 4E4F194A
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133// 0000000065F4: C8010B2D 30310B37
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133// 0000000065FC: C8010B41 3A3B0B4B
	v_dual_fmac_f32 v68, v45, v141 :: v_dual_fmac_f32 v69, v55, v141// 000000006604: C8011B2D 44451B37
	v_dual_fmac_f32 v78, v65, v141 :: v_dual_fmac_f32 v79, v75, v141// 00000000660C: C8011B41 4E4F1B4B
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134// 000000006614: C8010D2E 30310D38
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134// 00000000661C: C8010D42 3A3B0D4C
	v_dual_fmac_f32 v68, v46, v142 :: v_dual_fmac_f32 v69, v56, v142// 000000006624: C8011D2E 44451D38
	v_dual_fmac_f32 v78, v66, v142 :: v_dual_fmac_f32 v79, v76, v142// 00000000662C: C8011D42 4E4F1D4C
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135// 000000006634: C8010F2F 30310F39
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135// 00000000663C: C8010F43 3A3B0F4D
	v_dual_fmac_f32 v68, v47, v143 :: v_dual_fmac_f32 v69, v57, v143// 000000006644: C8011F2F 44451F39
	v_dual_fmac_f32 v78, v67, v143 :: v_dual_fmac_f32 v79, v77, v143// 00000000664C: C8011F43 4E4F1F4D
	v_dual_add_f32 v184, v184, v48 :: v_dual_add_f32 v185, v185, v49// 000000006654: C90861B8 B8B863B9
	v_dual_add_f32 v186, v186, v58 :: v_dual_add_f32 v187, v187, v59// 00000000665C: C90875BA BABA77BB
	v_dual_add_f32 v200, v200, v68 :: v_dual_add_f32 v201, v201, v69// 000000006664: C90889C8 C8C88BC9
	v_dual_add_f32 v202, v202, v78 :: v_dual_add_f32 v203, v203, v79// 00000000666C: C9089DCA CACA9FCB
	s_wait_loadcnt 0x7                                         // 000000006674: BFC00007
	v_dual_mul_f32 v48, v41, v145 :: v_dual_mul_f32 v49, v51, v145// 000000006678: C8C72329 30312333
	v_dual_mul_f32 v58, v61, v145 :: v_dual_mul_f32 v59, v71, v145// 000000006680: C8C7233D 3A3B2347
	v_dual_fmac_f32 v48, v40, v144 :: v_dual_fmac_f32 v49, v50, v144// 000000006688: C8012128 30312132
	v_dual_fmac_f32 v58, v60, v144 :: v_dual_fmac_f32 v59, v70, v144// 000000006690: C801213C 3A3B2146
	v_dual_fmac_f32 v48, v42, v146 :: v_dual_fmac_f32 v49, v52, v146// 000000006698: C801252A 30312534
	v_dual_fmac_f32 v58, v62, v146 :: v_dual_fmac_f32 v59, v72, v146// 0000000066A0: C801253E 3A3B2548
	v_dual_fmac_f32 v48, v43, v147 :: v_dual_fmac_f32 v49, v53, v147// 0000000066A8: C801272B 30312735
	v_dual_fmac_f32 v58, v63, v147 :: v_dual_fmac_f32 v59, v73, v147// 0000000066B0: C801273F 3A3B2749
	s_wait_loadcnt 0x6                                         // 0000000066B8: BFC00006
	v_dual_fmac_f32 v48, v44, v148 :: v_dual_fmac_f32 v49, v54, v148// 0000000066BC: C801292C 30312936
	v_dual_fmac_f32 v58, v64, v148 :: v_dual_fmac_f32 v59, v74, v148// 0000000066C4: C8012940 3A3B294A
	v_dual_fmac_f32 v48, v45, v149 :: v_dual_fmac_f32 v49, v55, v149// 0000000066CC: C8012B2D 30312B37
	v_dual_fmac_f32 v58, v65, v149 :: v_dual_fmac_f32 v59, v75, v149// 0000000066D4: C8012B41 3A3B2B4B
	v_dual_fmac_f32 v48, v46, v150 :: v_dual_fmac_f32 v49, v56, v150// 0000000066DC: C8012D2E 30312D38
	v_dual_fmac_f32 v58, v66, v150 :: v_dual_fmac_f32 v59, v76, v150// 0000000066E4: C8012D42 3A3B2D4C
	v_dual_fmac_f32 v48, v47, v151 :: v_dual_fmac_f32 v49, v57, v151// 0000000066EC: C8012F2F 30312F39
	v_dual_fmac_f32 v58, v67, v151 :: v_dual_fmac_f32 v59, v77, v151// 0000000066F4: C8012F43 3A3B2F4D
	v_dual_add_f32 v216, v216, v48 :: v_dual_add_f32 v217, v217, v49// 0000000066FC: C90861D8 D8D863D9
	v_dual_add_f32 v218, v218, v58 :: v_dual_add_f32 v219, v219, v59// 000000006704: C90875DA DADA77DB
	v_bfe_u32 v40, v36, 0, 2                                   // 00000000670C: D6100028 02090124
	v_bfe_u32 v50, v37, 0, 2                                   // 000000006714: D6100032 02090125
	v_bfe_u32 v60, v38, 0, 2                                   // 00000000671C: D610003C 02090126
	v_bfe_u32 v70, v39, 0, 2                                   // 000000006724: D6100046 02090127
	v_bfe_u32 v41, v36, 2, 2                                   // 00000000672C: D6100029 02090524
	v_bfe_u32 v51, v37, 2, 2                                   // 000000006734: D6100033 02090525
	v_bfe_u32 v61, v38, 2, 2                                   // 00000000673C: D610003D 02090526
	v_bfe_u32 v71, v39, 2, 2                                   // 000000006744: D6100047 02090527
	v_bfe_u32 v42, v36, 4, 2                                   // 00000000674C: D610002A 02090924
	v_bfe_u32 v52, v37, 4, 2                                   // 000000006754: D6100034 02090925
	v_bfe_u32 v62, v38, 4, 2                                   // 00000000675C: D610003E 02090926
	v_bfe_u32 v72, v39, 4, 2                                   // 000000006764: D6100048 02090927
	v_bfe_u32 v43, v36, 6, 2                                   // 00000000676C: D610002B 02090D24
	v_bfe_u32 v53, v37, 6, 2                                   // 000000006774: D6100035 02090D25
	v_bfe_u32 v63, v38, 6, 2                                   // 00000000677C: D610003F 02090D26
	v_bfe_u32 v73, v39, 6, 2                                   // 000000006784: D6100049 02090D27
	v_bfe_u32 v44, v36, 8, 2                                   // 00000000678C: D610002C 02091124
	v_bfe_u32 v54, v37, 8, 2                                   // 000000006794: D6100036 02091125
	v_bfe_u32 v64, v38, 8, 2                                   // 00000000679C: D6100040 02091126
	v_bfe_u32 v74, v39, 8, 2                                   // 0000000067A4: D610004A 02091127
	v_bfe_u32 v45, v36, 10, 2                                  // 0000000067AC: D610002D 02091524
	v_bfe_u32 v55, v37, 10, 2                                  // 0000000067B4: D6100037 02091525
	v_bfe_u32 v65, v38, 10, 2                                  // 0000000067BC: D6100041 02091526
	v_bfe_u32 v75, v39, 10, 2                                  // 0000000067C4: D610004B 02091527
	v_bfe_u32 v46, v36, 12, 2                                  // 0000000067CC: D610002E 02091924
	v_bfe_u32 v56, v37, 12, 2                                  // 0000000067D4: D6100038 02091925
	v_bfe_u32 v66, v38, 12, 2                                  // 0000000067DC: D6100042 02091926
	v_bfe_u32 v76, v39, 12, 2                                  // 0000000067E4: D610004C 02091927
	v_bfe_u32 v47, v36, 14, 2                                  // 0000000067EC: D610002F 02091D24
	v_bfe_u32 v57, v37, 14, 2                                  // 0000000067F4: D6100039 02091D25
	v_bfe_u32 v67, v38, 14, 2                                  // 0000000067FC: D6100043 02091D26
	v_bfe_u32 v77, v39, 14, 2                                  // 000000006804: D610004D 02091D27
	v_cvt_f32_ubyte0_e32 v40, v40                              // 00000000680C: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000006810: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000006814: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000006818: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 00000000681C: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000006820: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000006824: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000006828: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 00000000682C: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000006830: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000006834: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000006838: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 00000000683C: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000006840: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000006844: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000006848: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 00000000684C: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000006850: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000006854: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000006858: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 00000000685C: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000006860: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000006864: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000006868: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 00000000686C: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000006870: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000006874: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000006878: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 00000000687C: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000006880: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000006884: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000006888: 7E9A234D
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000688C: CC206028 0C525114
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006894: CC206032 0C566515
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000689C: CC20603C 0C5A7916
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068A4: CC206046 0C5E8D17
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068AC: CC206029 0C525314
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068B4: CC206033 0C566715
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068BC: CC20603D 0C5A7B16
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068C4: CC206047 0C5E8F17
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068CC: CC20602A 0C525514
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068D4: CC206034 0C566915
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068DC: CC20603E 0C5A7D16
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068E4: CC206048 0C5E9117
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068EC: CC20602B 0C525714
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068F4: CC206035 0C566B15
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000068FC: CC20603F 0C5A7F16
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006904: CC206049 0C5E9317
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000690C: CC20602C 0C525914
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006914: CC206036 0C566D15
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000691C: CC206040 0C5A8116
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006924: CC20604A 0C5E9517
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000692C: CC20602D 0C525B14
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006934: CC206037 0C566F15
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000693C: CC206041 0C5A8316
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006944: CC20604B 0C5E9717
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000694C: CC20602E 0C525D14
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006954: CC206038 0C567115
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000695C: CC206042 0C5A8516
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006964: CC20604C 0C5E9917
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000696C: CC20602F 0C525F14
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006974: CC206039 0C567315
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000697C: CC206043 0C5A8716
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006984: CC20604D 0C5E9B17
	s_wait_loadcnt 0x5                                         // 00000000698C: BFC00005
	v_dual_mul_f32 v48, v41, v153 :: v_dual_mul_f32 v49, v51, v153// 000000006990: C8C73329 30313333
	v_dual_mul_f32 v58, v61, v153 :: v_dual_mul_f32 v59, v71, v153// 000000006998: C8C7333D 3A3B3347
	s_wait_loadcnt 0x3                                         // 0000000069A0: BFC00003
	v_dual_mul_f32 v68, v41, v161 :: v_dual_mul_f32 v69, v51, v161// 0000000069A4: C8C74329 44454333
	v_dual_mul_f32 v78, v61, v161 :: v_dual_mul_f32 v79, v71, v161// 0000000069AC: C8C7433D 4E4F4347
	v_dual_fmac_f32 v48, v40, v152 :: v_dual_fmac_f32 v49, v50, v152// 0000000069B4: C8013128 30313132
	v_dual_fmac_f32 v58, v60, v152 :: v_dual_fmac_f32 v59, v70, v152// 0000000069BC: C801313C 3A3B3146
	v_dual_fmac_f32 v68, v40, v160 :: v_dual_fmac_f32 v69, v50, v160// 0000000069C4: C8014128 44454132
	v_dual_fmac_f32 v78, v60, v160 :: v_dual_fmac_f32 v79, v70, v160// 0000000069CC: C801413C 4E4F4146
	v_dual_fmac_f32 v48, v42, v154 :: v_dual_fmac_f32 v49, v52, v154// 0000000069D4: C801352A 30313534
	v_dual_fmac_f32 v58, v62, v154 :: v_dual_fmac_f32 v59, v72, v154// 0000000069DC: C801353E 3A3B3548
	v_dual_fmac_f32 v68, v42, v162 :: v_dual_fmac_f32 v69, v52, v162// 0000000069E4: C801452A 44454534
	v_dual_fmac_f32 v78, v62, v162 :: v_dual_fmac_f32 v79, v72, v162// 0000000069EC: C801453E 4E4F4548
	v_dual_fmac_f32 v48, v43, v155 :: v_dual_fmac_f32 v49, v53, v155// 0000000069F4: C801372B 30313735
	v_dual_fmac_f32 v58, v63, v155 :: v_dual_fmac_f32 v59, v73, v155// 0000000069FC: C801373F 3A3B3749
	v_dual_fmac_f32 v68, v43, v163 :: v_dual_fmac_f32 v69, v53, v163// 000000006A04: C801472B 44454735
	v_dual_fmac_f32 v78, v63, v163 :: v_dual_fmac_f32 v79, v73, v163// 000000006A0C: C801473F 4E4F4749
	v_dual_fmac_f32 v48, v44, v156 :: v_dual_fmac_f32 v49, v54, v156// 000000006A14: C801392C 30313936
	v_dual_fmac_f32 v58, v64, v156 :: v_dual_fmac_f32 v59, v74, v156// 000000006A1C: C8013940 3A3B394A
	s_wait_loadcnt 0x2                                         // 000000006A24: BFC00002
	v_dual_fmac_f32 v68, v44, v164 :: v_dual_fmac_f32 v69, v54, v164// 000000006A28: C801492C 44454936
	v_dual_fmac_f32 v78, v64, v164 :: v_dual_fmac_f32 v79, v74, v164// 000000006A30: C8014940 4E4F494A
	v_dual_fmac_f32 v48, v45, v157 :: v_dual_fmac_f32 v49, v55, v157// 000000006A38: C8013B2D 30313B37
	v_dual_fmac_f32 v58, v65, v157 :: v_dual_fmac_f32 v59, v75, v157// 000000006A40: C8013B41 3A3B3B4B
	v_dual_fmac_f32 v68, v45, v165 :: v_dual_fmac_f32 v69, v55, v165// 000000006A48: C8014B2D 44454B37
	v_dual_fmac_f32 v78, v65, v165 :: v_dual_fmac_f32 v79, v75, v165// 000000006A50: C8014B41 4E4F4B4B
	v_dual_fmac_f32 v48, v46, v158 :: v_dual_fmac_f32 v49, v56, v158// 000000006A58: C8013D2E 30313D38
	v_dual_fmac_f32 v58, v66, v158 :: v_dual_fmac_f32 v59, v76, v158// 000000006A60: C8013D42 3A3B3D4C
	v_dual_fmac_f32 v68, v46, v166 :: v_dual_fmac_f32 v69, v56, v166// 000000006A68: C8014D2E 44454D38
	v_dual_fmac_f32 v78, v66, v166 :: v_dual_fmac_f32 v79, v76, v166// 000000006A70: C8014D42 4E4F4D4C
	v_dual_fmac_f32 v48, v47, v159 :: v_dual_fmac_f32 v49, v57, v159// 000000006A78: C8013F2F 30313F39
	v_dual_fmac_f32 v58, v67, v159 :: v_dual_fmac_f32 v59, v77, v159// 000000006A80: C8013F43 3A3B3F4D
	v_dual_fmac_f32 v68, v47, v167 :: v_dual_fmac_f32 v69, v57, v167// 000000006A88: C8014F2F 44454F39
	v_dual_fmac_f32 v78, v67, v167 :: v_dual_fmac_f32 v79, v77, v167// 000000006A90: C8014F43 4E4F4F4D
	v_dual_add_f32 v188, v188, v48 :: v_dual_add_f32 v189, v189, v49// 000000006A98: C90861BC BCBC63BD
	v_dual_add_f32 v190, v190, v58 :: v_dual_add_f32 v191, v191, v59// 000000006AA0: C90875BE BEBE77BF
	v_dual_add_f32 v204, v204, v68 :: v_dual_add_f32 v205, v205, v69// 000000006AA8: C90889CC CCCC8BCD
	v_dual_add_f32 v206, v206, v78 :: v_dual_add_f32 v207, v207, v79// 000000006AB0: C9089DCE CECE9FCF
	s_wait_loadcnt 0x1                                         // 000000006AB8: BFC00001
	v_dual_mul_f32 v48, v41, v169 :: v_dual_mul_f32 v49, v51, v169// 000000006ABC: C8C75329 30315333
	v_dual_mul_f32 v58, v61, v169 :: v_dual_mul_f32 v59, v71, v169// 000000006AC4: C8C7533D 3A3B5347
	v_dual_fmac_f32 v48, v40, v168 :: v_dual_fmac_f32 v49, v50, v168// 000000006ACC: C8015128 30315132
	v_dual_fmac_f32 v58, v60, v168 :: v_dual_fmac_f32 v59, v70, v168// 000000006AD4: C801513C 3A3B5146
	v_dual_fmac_f32 v48, v42, v170 :: v_dual_fmac_f32 v49, v52, v170// 000000006ADC: C801552A 30315534
	v_dual_fmac_f32 v58, v62, v170 :: v_dual_fmac_f32 v59, v72, v170// 000000006AE4: C801553E 3A3B5548
	v_dual_fmac_f32 v48, v43, v171 :: v_dual_fmac_f32 v49, v53, v171// 000000006AEC: C801572B 30315735
	v_dual_fmac_f32 v58, v63, v171 :: v_dual_fmac_f32 v59, v73, v171// 000000006AF4: C801573F 3A3B5749
	s_wait_loadcnt 0x0                                         // 000000006AFC: BFC00000
	v_dual_fmac_f32 v48, v44, v172 :: v_dual_fmac_f32 v49, v54, v172// 000000006B00: C801592C 30315936
	v_dual_fmac_f32 v58, v64, v172 :: v_dual_fmac_f32 v59, v74, v172// 000000006B08: C8015940 3A3B594A
	v_dual_fmac_f32 v48, v45, v173 :: v_dual_fmac_f32 v49, v55, v173// 000000006B10: C8015B2D 30315B37
	v_dual_fmac_f32 v58, v65, v173 :: v_dual_fmac_f32 v59, v75, v173// 000000006B18: C8015B41 3A3B5B4B
	v_dual_fmac_f32 v48, v46, v174 :: v_dual_fmac_f32 v49, v56, v174// 000000006B20: C8015D2E 30315D38
	v_dual_fmac_f32 v58, v66, v174 :: v_dual_fmac_f32 v59, v76, v174// 000000006B28: C8015D42 3A3B5D4C
	v_dual_fmac_f32 v48, v47, v175 :: v_dual_fmac_f32 v49, v57, v175// 000000006B30: C8015F2F 30315F39
	v_dual_fmac_f32 v58, v67, v175 :: v_dual_fmac_f32 v59, v77, v175// 000000006B38: C8015F43 3A3B5F4D
	v_dual_add_f32 v220, v220, v48 :: v_dual_add_f32 v221, v221, v49// 000000006B40: C90861DC DCDC63DD
	v_dual_add_f32 v222, v222, v58 :: v_dual_add_f32 v223, v223, v59// 000000006B48: C90875DE DEDE77DF
	s_add_co_i32 s16, s16, 0x120                               // 000000006B50: 8110FF10 00000120
	s_add_co_i32 s17, s17, 0x120                               // 000000006B58: 8111FF11 00000120
	s_add_co_i32 s18, s18, 0x120                               // 000000006B60: 8112FF12 00000120
	s_add_co_i32 s19, s19, 0x120                               // 000000006B68: 8113FF13 00000120
	s_add_co_i32 s28, s28, 0x1000                              // 000000006B70: 811CFF1C 00001000
	s_add_co_i32 s29, s29, 0x1000                              // 000000006B78: 811DFF1D 00001000
	s_add_co_i32 s30, s30, 0x1000                              // 000000006B80: 811EFF1E 00001000
	s_add_co_i32 s36, s36, -1                                  // 000000006B88: 8124C124
	s_cmp_lg_u32 s36, 0                                        // 000000006B8C: BF078024
	s_cbranch_scc1 64251                                       // 000000006B90: BFA2FAFB <gemv_mq2g256v2_xbatch_pm+0x4080>
	s_cmp_gt_u32 s37, 0                                        // 000000006B94: BF088025
	s_cbranch_scc0 961                                         // 000000006B98: BFA103C1 <gemv_mq2g256v2_xbatch_pm+0x63a0>
	buffer_load_b32 v8, v1, s[20:23], s16 offen                // 000000006B9C: C4050010 40802808 00000001
	buffer_load_b32 v9, v1, s[20:23], s17 offen                // 000000006BA8: C4050011 40802809 00000001
	buffer_load_b32 v10, v1, s[20:23], s18 offen               // 000000006BB4: C4050012 4080280A 00000001
	buffer_load_b32 v11, v1, s[20:23], s19 offen               // 000000006BC0: C4050013 4080280B 00000001
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8      // 000000006BCC: C4050010 40802818 00000802
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8      // 000000006BD8: C4050011 40802819 00000802
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8      // 000000006BE4: C4050012 4080281A 00000802
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8      // 000000006BF0: C4050013 4080281B 00000802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen         // 000000006BFC: C405C01C 40803050 00000003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16// 000000006C08: C405C01C 40803054 00001003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen         // 000000006C14: C405C01D 40803058 00000003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16// 000000006C20: C405C01D 4080305C 00001003
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen         // 000000006C2C: C405C01E 40803060 00000003
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16// 000000006C38: C405C01E 40803064 00001003
	s_wait_loadcnt 0x9                                         // 000000006C44: BFC00009
	v_bfe_u32 v40, v24, 0, 2                                   // 000000006C48: D6100028 02090118
	s_wait_loadcnt 0x8                                         // 000000006C50: BFC00008
	v_bfe_u32 v50, v25, 0, 2                                   // 000000006C54: D6100032 02090119
	s_wait_loadcnt 0x7                                         // 000000006C5C: BFC00007
	v_bfe_u32 v60, v26, 0, 2                                   // 000000006C60: D610003C 0209011A
	s_wait_loadcnt 0x6                                         // 000000006C68: BFC00006
	v_bfe_u32 v70, v27, 0, 2                                   // 000000006C6C: D6100046 0209011B
	v_bfe_u32 v41, v24, 2, 2                                   // 000000006C74: D6100029 02090518
	v_bfe_u32 v51, v25, 2, 2                                   // 000000006C7C: D6100033 02090519
	v_bfe_u32 v61, v26, 2, 2                                   // 000000006C84: D610003D 0209051A
	v_bfe_u32 v71, v27, 2, 2                                   // 000000006C8C: D6100047 0209051B
	v_bfe_u32 v42, v24, 4, 2                                   // 000000006C94: D610002A 02090918
	v_bfe_u32 v52, v25, 4, 2                                   // 000000006C9C: D6100034 02090919
	v_bfe_u32 v62, v26, 4, 2                                   // 000000006CA4: D610003E 0209091A
	v_bfe_u32 v72, v27, 4, 2                                   // 000000006CAC: D6100048 0209091B
	v_bfe_u32 v43, v24, 6, 2                                   // 000000006CB4: D610002B 02090D18
	v_bfe_u32 v53, v25, 6, 2                                   // 000000006CBC: D6100035 02090D19
	v_bfe_u32 v63, v26, 6, 2                                   // 000000006CC4: D610003F 02090D1A
	v_bfe_u32 v73, v27, 6, 2                                   // 000000006CCC: D6100049 02090D1B
	v_bfe_u32 v44, v24, 8, 2                                   // 000000006CD4: D610002C 02091118
	v_bfe_u32 v54, v25, 8, 2                                   // 000000006CDC: D6100036 02091119
	v_bfe_u32 v64, v26, 8, 2                                   // 000000006CE4: D6100040 0209111A
	v_bfe_u32 v74, v27, 8, 2                                   // 000000006CEC: D610004A 0209111B
	v_bfe_u32 v45, v24, 10, 2                                  // 000000006CF4: D610002D 02091518
	v_bfe_u32 v55, v25, 10, 2                                  // 000000006CFC: D6100037 02091519
	v_bfe_u32 v65, v26, 10, 2                                  // 000000006D04: D6100041 0209151A
	v_bfe_u32 v75, v27, 10, 2                                  // 000000006D0C: D610004B 0209151B
	v_bfe_u32 v46, v24, 12, 2                                  // 000000006D14: D610002E 02091918
	v_bfe_u32 v56, v25, 12, 2                                  // 000000006D1C: D6100038 02091919
	v_bfe_u32 v66, v26, 12, 2                                  // 000000006D24: D6100042 0209191A
	v_bfe_u32 v76, v27, 12, 2                                  // 000000006D2C: D610004C 0209191B
	v_bfe_u32 v47, v24, 14, 2                                  // 000000006D34: D610002F 02091D18
	v_bfe_u32 v57, v25, 14, 2                                  // 000000006D3C: D6100039 02091D19
	v_bfe_u32 v67, v26, 14, 2                                  // 000000006D44: D6100043 02091D1A
	v_bfe_u32 v77, v27, 14, 2                                  // 000000006D4C: D610004D 02091D1B
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000006D54: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000006D58: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000006D5C: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000006D60: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000006D64: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000006D68: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000006D6C: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000006D70: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000006D74: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000006D78: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000006D7C: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000006D80: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000006D84: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000006D88: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000006D8C: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000006D90: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000006D94: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000006D98: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000006D9C: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000006DA0: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000006DA4: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000006DA8: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000006DAC: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000006DB0: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000006DB4: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000006DB8: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000006DBC: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000006DC0: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000006DC4: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000006DC8: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000006DCC: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000006DD0: 7E9A234D
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006DD4: CC206028 0C225108
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006DDC: CC206032 0C266509
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006DE4: CC20603C 0C2A790A
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006DEC: CC206046 0C2E8D0B
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006DF4: CC206029 0C225308
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006DFC: CC206033 0C266709
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E04: CC20603D 0C2A7B0A
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E0C: CC206047 0C2E8F0B
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E14: CC20602A 0C225508
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E1C: CC206034 0C266909
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E24: CC20603E 0C2A7D0A
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E2C: CC206048 0C2E910B
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E34: CC20602B 0C225708
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E3C: CC206035 0C266B09
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E44: CC20603F 0C2A7F0A
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E4C: CC206049 0C2E930B
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E54: CC20602C 0C225908
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E5C: CC206036 0C266D09
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E64: CC206040 0C2A810A
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E6C: CC20604A 0C2E950B
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E74: CC20602D 0C225B08
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E7C: CC206037 0C266F09
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E84: CC206041 0C2A830A
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E8C: CC20604B 0C2E970B
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E94: CC20602E 0C225D08
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006E9C: CC206038 0C267109
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006EA4: CC206042 0C2A850A
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006EAC: CC20604C 0C2E990B
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006EB4: CC20602F 0C225F08
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006EBC: CC206039 0C267309
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006EC4: CC206043 0C2A870A
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000006ECC: CC20604D 0C2E9B0B
	s_wait_loadcnt 0x5                                         // 000000006ED4: BFC00005
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 000000006ED8: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 000000006EE0: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x3                                         // 000000006EE8: BFC00003
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 000000006EEC: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 000000006EF4: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000006EFC: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000006F04: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 000000006F0C: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 000000006F14: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000006F1C: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000006F24: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 000000006F2C: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 000000006F34: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000006F3C: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000006F44: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 000000006F4C: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 000000006F54: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000006F5C: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000006F64: C800A940 3A3AA94A
	s_wait_loadcnt 0x2                                         // 000000006F6C: BFC00002
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 000000006F70: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 000000006F78: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000006F80: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000006F88: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 000000006F90: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 000000006F98: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 000000006FA0: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000006FA8: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 000000006FB0: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 000000006FB8: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 000000006FC0: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 000000006FC8: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 000000006FD0: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 000000006FD8: C800BF43 4E4EBF4D
	v_dual_add_f32 v176, v176, v48 :: v_dual_add_f32 v177, v177, v49// 000000006FE0: C90861B0 B0B063B1
	v_dual_add_f32 v178, v178, v58 :: v_dual_add_f32 v179, v179, v59// 000000006FE8: C90875B2 B2B277B3
	v_dual_add_f32 v192, v192, v68 :: v_dual_add_f32 v193, v193, v69// 000000006FF0: C90889C0 C0C08BC1
	v_dual_add_f32 v194, v194, v78 :: v_dual_add_f32 v195, v195, v79// 000000006FF8: C9089DC2 C2C29FC3
	s_wait_loadcnt 0x1                                         // 000000007000: BFC00001
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97// 000000007004: C8C6C329 3030C333
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97// 00000000700C: C8C6C33D 3A3AC347
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96// 000000007014: C800C128 3030C132
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96// 00000000701C: C800C13C 3A3AC146
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98// 000000007024: C800C52A 3030C534
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98// 00000000702C: C800C53E 3A3AC548
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99// 000000007034: C800C72B 3030C735
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99// 00000000703C: C800C73F 3A3AC749
	s_wait_loadcnt 0x0                                         // 000000007044: BFC00000
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100// 000000007048: C800C92C 3030C936
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100// 000000007050: C800C940 3A3AC94A
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101// 000000007058: C800CB2D 3030CB37
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101// 000000007060: C800CB41 3A3ACB4B
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102// 000000007068: C800CD2E 3030CD38
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102// 000000007070: C800CD42 3A3ACD4C
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103// 000000007078: C800CF2F 3030CF39
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103// 000000007080: C800CF43 3A3ACF4D
	v_dual_add_f32 v208, v208, v48 :: v_dual_add_f32 v209, v209, v49// 000000007088: C90861D0 D0D063D1
	v_dual_add_f32 v210, v210, v58 :: v_dual_add_f32 v211, v211, v59// 000000007090: C90875D2 D2D277D3
	s_cmp_gt_u32 s37, 1                                        // 000000007098: BF088125
	s_cbranch_scc0 640                                         // 00000000709C: BFA10280 <gemv_mq2g256v2_xbatch_pm+0x63a0>
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:72     // 0000000070A0: C4050010 4080280C 00004801
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:72     // 0000000070AC: C4050011 4080280D 00004801
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:72     // 0000000070B8: C4050012 4080280E 00004801
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:72     // 0000000070C4: C4050013 4080280F 00004801
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:80     // 0000000070D0: C4050010 4080281C 00005002
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:80     // 0000000070DC: C4050011 4080281D 00005002
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:80     // 0000000070E8: C4050012 4080281E 00005002
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:80     // 0000000070F4: C4050013 4080281F 00005002
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024// 000000007100: C405C01C 40803050 00040003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040// 00000000710C: C405C01C 40803054 00041003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:1024// 000000007118: C405C01D 40803058 00040003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:1040// 000000007124: C405C01D 4080305C 00041003
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:1024// 000000007130: C405C01E 40803060 00040003
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:1040// 00000000713C: C405C01E 40803064 00041003
	s_wait_loadcnt 0x9                                         // 000000007148: BFC00009
	v_bfe_u32 v40, v28, 0, 2                                   // 00000000714C: D6100028 0209011C
	s_wait_loadcnt 0x8                                         // 000000007154: BFC00008
	v_bfe_u32 v50, v29, 0, 2                                   // 000000007158: D6100032 0209011D
	s_wait_loadcnt 0x7                                         // 000000007160: BFC00007
	v_bfe_u32 v60, v30, 0, 2                                   // 000000007164: D610003C 0209011E
	s_wait_loadcnt 0x6                                         // 00000000716C: BFC00006
	v_bfe_u32 v70, v31, 0, 2                                   // 000000007170: D6100046 0209011F
	v_bfe_u32 v41, v28, 2, 2                                   // 000000007178: D6100029 0209051C
	v_bfe_u32 v51, v29, 2, 2                                   // 000000007180: D6100033 0209051D
	v_bfe_u32 v61, v30, 2, 2                                   // 000000007188: D610003D 0209051E
	v_bfe_u32 v71, v31, 2, 2                                   // 000000007190: D6100047 0209051F
	v_bfe_u32 v42, v28, 4, 2                                   // 000000007198: D610002A 0209091C
	v_bfe_u32 v52, v29, 4, 2                                   // 0000000071A0: D6100034 0209091D
	v_bfe_u32 v62, v30, 4, 2                                   // 0000000071A8: D610003E 0209091E
	v_bfe_u32 v72, v31, 4, 2                                   // 0000000071B0: D6100048 0209091F
	v_bfe_u32 v43, v28, 6, 2                                   // 0000000071B8: D610002B 02090D1C
	v_bfe_u32 v53, v29, 6, 2                                   // 0000000071C0: D6100035 02090D1D
	v_bfe_u32 v63, v30, 6, 2                                   // 0000000071C8: D610003F 02090D1E
	v_bfe_u32 v73, v31, 6, 2                                   // 0000000071D0: D6100049 02090D1F
	v_bfe_u32 v44, v28, 8, 2                                   // 0000000071D8: D610002C 0209111C
	v_bfe_u32 v54, v29, 8, 2                                   // 0000000071E0: D6100036 0209111D
	v_bfe_u32 v64, v30, 8, 2                                   // 0000000071E8: D6100040 0209111E
	v_bfe_u32 v74, v31, 8, 2                                   // 0000000071F0: D610004A 0209111F
	v_bfe_u32 v45, v28, 10, 2                                  // 0000000071F8: D610002D 0209151C
	v_bfe_u32 v55, v29, 10, 2                                  // 000000007200: D6100037 0209151D
	v_bfe_u32 v65, v30, 10, 2                                  // 000000007208: D6100041 0209151E
	v_bfe_u32 v75, v31, 10, 2                                  // 000000007210: D610004B 0209151F
	v_bfe_u32 v46, v28, 12, 2                                  // 000000007218: D610002E 0209191C
	v_bfe_u32 v56, v29, 12, 2                                  // 000000007220: D6100038 0209191D
	v_bfe_u32 v66, v30, 12, 2                                  // 000000007228: D6100042 0209191E
	v_bfe_u32 v76, v31, 12, 2                                  // 000000007230: D610004C 0209191F
	v_bfe_u32 v47, v28, 14, 2                                  // 000000007238: D610002F 02091D1C
	v_bfe_u32 v57, v29, 14, 2                                  // 000000007240: D6100039 02091D1D
	v_bfe_u32 v67, v30, 14, 2                                  // 000000007248: D6100043 02091D1E
	v_bfe_u32 v77, v31, 14, 2                                  // 000000007250: D610004D 02091D1F
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000007258: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 00000000725C: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000007260: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000007264: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000007268: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 00000000726C: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000007270: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000007274: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000007278: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 00000000727C: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000007280: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000007284: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000007288: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 00000000728C: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000007290: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000007294: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000007298: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 00000000729C: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 0000000072A0: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 0000000072A4: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 0000000072A8: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 0000000072AC: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 0000000072B0: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 0000000072B4: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 0000000072B8: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 0000000072BC: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 0000000072C0: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 0000000072C4: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 0000000072C8: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 0000000072CC: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 0000000072D0: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 0000000072D4: 7E9A234D
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000072D8: CC206028 0C32510C
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000072E0: CC206032 0C36650D
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000072E8: CC20603C 0C3A790E
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000072F0: CC206046 0C3E8D0F
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000072F8: CC206029 0C32530C
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007300: CC206033 0C36670D
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007308: CC20603D 0C3A7B0E
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007310: CC206047 0C3E8F0F
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007318: CC20602A 0C32550C
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007320: CC206034 0C36690D
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007328: CC20603E 0C3A7D0E
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007330: CC206048 0C3E910F
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007338: CC20602B 0C32570C
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007340: CC206035 0C366B0D
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007348: CC20603F 0C3A7F0E
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007350: CC206049 0C3E930F
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007358: CC20602C 0C32590C
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007360: CC206036 0C366D0D
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007368: CC206040 0C3A810E
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007370: CC20604A 0C3E950F
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007378: CC20602D 0C325B0C
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007380: CC206037 0C366F0D
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007388: CC206041 0C3A830E
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007390: CC20604B 0C3E970F
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007398: CC20602E 0C325D0C
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000073A0: CC206038 0C36710D
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000073A8: CC206042 0C3A850E
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000073B0: CC20604C 0C3E990F
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000073B8: CC20602F 0C325F0C
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000073C0: CC206039 0C36730D
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000073C8: CC206043 0C3A870E
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000073D0: CC20604D 0C3E9B0F
	s_wait_loadcnt 0x5                                         // 0000000073D8: BFC00005
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 0000000073DC: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 0000000073E4: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x3                                         // 0000000073EC: BFC00003
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 0000000073F0: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 0000000073F8: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000007400: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000007408: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 000000007410: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 000000007418: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000007420: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000007428: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 000000007430: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 000000007438: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000007440: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000007448: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 000000007450: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 000000007458: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000007460: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000007468: C800A940 3A3AA94A
	s_wait_loadcnt 0x2                                         // 000000007470: BFC00002
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 000000007474: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 00000000747C: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000007484: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 00000000748C: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 000000007494: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 00000000749C: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 0000000074A4: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 0000000074AC: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 0000000074B4: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 0000000074BC: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 0000000074C4: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 0000000074CC: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 0000000074D4: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 0000000074DC: C800BF43 4E4EBF4D
	v_dual_add_f32 v180, v180, v48 :: v_dual_add_f32 v181, v181, v49// 0000000074E4: C90861B4 B4B463B5
	v_dual_add_f32 v182, v182, v58 :: v_dual_add_f32 v183, v183, v59// 0000000074EC: C90875B6 B6B677B7
	v_dual_add_f32 v196, v196, v68 :: v_dual_add_f32 v197, v197, v69// 0000000074F4: C90889C4 C4C48BC5
	v_dual_add_f32 v198, v198, v78 :: v_dual_add_f32 v199, v199, v79// 0000000074FC: C9089DC6 C6C69FC7
	s_wait_loadcnt 0x1                                         // 000000007504: BFC00001
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97// 000000007508: C8C6C329 3030C333
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97// 000000007510: C8C6C33D 3A3AC347
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96// 000000007518: C800C128 3030C132
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96// 000000007520: C800C13C 3A3AC146
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98// 000000007528: C800C52A 3030C534
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98// 000000007530: C800C53E 3A3AC548
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99// 000000007538: C800C72B 3030C735
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99// 000000007540: C800C73F 3A3AC749
	s_wait_loadcnt 0x0                                         // 000000007548: BFC00000
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100// 00000000754C: C800C92C 3030C936
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100// 000000007554: C800C940 3A3AC94A
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101// 00000000755C: C800CB2D 3030CB37
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101// 000000007564: C800CB41 3A3ACB4B
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102// 00000000756C: C800CD2E 3030CD38
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102// 000000007574: C800CD42 3A3ACD4C
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103// 00000000757C: C800CF2F 3030CF39
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103// 000000007584: C800CF43 3A3ACF4D
	v_dual_add_f32 v212, v212, v48 :: v_dual_add_f32 v213, v213, v49// 00000000758C: C90861D4 D4D463D5
	v_dual_add_f32 v214, v214, v58 :: v_dual_add_f32 v215, v215, v59// 000000007594: C90875D6 D6D677D7
	s_cmp_gt_u32 s37, 2                                        // 00000000759C: BF088225
	s_cbranch_scc0 319                                         // 0000000075A0: BFA1013F <gemv_mq2g256v2_xbatch_pm+0x63a0>
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:144    // 0000000075A4: C4050010 40802810 00009001
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:144    // 0000000075B0: C4050011 40802811 00009001
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:144    // 0000000075BC: C4050012 40802812 00009001
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:144    // 0000000075C8: C4050013 40802813 00009001
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:152    // 0000000075D4: C4050010 40802820 00009802
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:152    // 0000000075E0: C4050011 40802821 00009802
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:152    // 0000000075EC: C4050012 40802822 00009802
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:152    // 0000000075F8: C4050013 40802823 00009802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048// 000000007604: C405C01C 40803050 00080003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064// 000000007610: C405C01C 40803054 00081003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:2048// 00000000761C: C405C01D 40803058 00080003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:2064// 000000007628: C405C01D 4080305C 00081003
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:2048// 000000007634: C405C01E 40803060 00080003
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:2064// 000000007640: C405C01E 40803064 00081003
	s_wait_loadcnt 0x9                                         // 00000000764C: BFC00009
	v_bfe_u32 v40, v32, 0, 2                                   // 000000007650: D6100028 02090120
	s_wait_loadcnt 0x8                                         // 000000007658: BFC00008
	v_bfe_u32 v50, v33, 0, 2                                   // 00000000765C: D6100032 02090121
	s_wait_loadcnt 0x7                                         // 000000007664: BFC00007
	v_bfe_u32 v60, v34, 0, 2                                   // 000000007668: D610003C 02090122
	s_wait_loadcnt 0x6                                         // 000000007670: BFC00006
	v_bfe_u32 v70, v35, 0, 2                                   // 000000007674: D6100046 02090123
	v_bfe_u32 v41, v32, 2, 2                                   // 00000000767C: D6100029 02090520
	v_bfe_u32 v51, v33, 2, 2                                   // 000000007684: D6100033 02090521
	v_bfe_u32 v61, v34, 2, 2                                   // 00000000768C: D610003D 02090522
	v_bfe_u32 v71, v35, 2, 2                                   // 000000007694: D6100047 02090523
	v_bfe_u32 v42, v32, 4, 2                                   // 00000000769C: D610002A 02090920
	v_bfe_u32 v52, v33, 4, 2                                   // 0000000076A4: D6100034 02090921
	v_bfe_u32 v62, v34, 4, 2                                   // 0000000076AC: D610003E 02090922
	v_bfe_u32 v72, v35, 4, 2                                   // 0000000076B4: D6100048 02090923
	v_bfe_u32 v43, v32, 6, 2                                   // 0000000076BC: D610002B 02090D20
	v_bfe_u32 v53, v33, 6, 2                                   // 0000000076C4: D6100035 02090D21
	v_bfe_u32 v63, v34, 6, 2                                   // 0000000076CC: D610003F 02090D22
	v_bfe_u32 v73, v35, 6, 2                                   // 0000000076D4: D6100049 02090D23
	v_bfe_u32 v44, v32, 8, 2                                   // 0000000076DC: D610002C 02091120
	v_bfe_u32 v54, v33, 8, 2                                   // 0000000076E4: D6100036 02091121
	v_bfe_u32 v64, v34, 8, 2                                   // 0000000076EC: D6100040 02091122
	v_bfe_u32 v74, v35, 8, 2                                   // 0000000076F4: D610004A 02091123
	v_bfe_u32 v45, v32, 10, 2                                  // 0000000076FC: D610002D 02091520
	v_bfe_u32 v55, v33, 10, 2                                  // 000000007704: D6100037 02091521
	v_bfe_u32 v65, v34, 10, 2                                  // 00000000770C: D6100041 02091522
	v_bfe_u32 v75, v35, 10, 2                                  // 000000007714: D610004B 02091523
	v_bfe_u32 v46, v32, 12, 2                                  // 00000000771C: D610002E 02091920
	v_bfe_u32 v56, v33, 12, 2                                  // 000000007724: D6100038 02091921
	v_bfe_u32 v66, v34, 12, 2                                  // 00000000772C: D6100042 02091922
	v_bfe_u32 v76, v35, 12, 2                                  // 000000007734: D610004C 02091923
	v_bfe_u32 v47, v32, 14, 2                                  // 00000000773C: D610002F 02091D20
	v_bfe_u32 v57, v33, 14, 2                                  // 000000007744: D6100039 02091D21
	v_bfe_u32 v67, v34, 14, 2                                  // 00000000774C: D6100043 02091D22
	v_bfe_u32 v77, v35, 14, 2                                  // 000000007754: D610004D 02091D23
	v_cvt_f32_ubyte0_e32 v40, v40                              // 00000000775C: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000007760: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000007764: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000007768: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 00000000776C: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000007770: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000007774: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000007778: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 00000000777C: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000007780: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000007784: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000007788: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 00000000778C: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000007790: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000007794: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000007798: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 00000000779C: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 0000000077A0: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 0000000077A4: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 0000000077A8: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 0000000077AC: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 0000000077B0: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 0000000077B4: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 0000000077B8: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 0000000077BC: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 0000000077C0: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 0000000077C4: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 0000000077C8: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 0000000077CC: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 0000000077D0: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 0000000077D4: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 0000000077D8: 7E9A234D
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000077DC: CC206028 0C425110
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000077E4: CC206032 0C466511
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000077EC: CC20603C 0C4A7912
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000077F4: CC206046 0C4E8D13
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000077FC: CC206029 0C425310
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007804: CC206033 0C466711
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000780C: CC20603D 0C4A7B12
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007814: CC206047 0C4E8F13
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000781C: CC20602A 0C425510
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007824: CC206034 0C466911
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000782C: CC20603E 0C4A7D12
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007834: CC206048 0C4E9113
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000783C: CC20602B 0C425710
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007844: CC206035 0C466B11
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000784C: CC20603F 0C4A7F12
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007854: CC206049 0C4E9313
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000785C: CC20602C 0C425910
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007864: CC206036 0C466D11
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000786C: CC206040 0C4A8112
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007874: CC20604A 0C4E9513
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000787C: CC20602D 0C425B10
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007884: CC206037 0C466F11
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000788C: CC206041 0C4A8312
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000007894: CC20604B 0C4E9713
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000789C: CC20602E 0C425D10
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000078A4: CC206038 0C467111
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000078AC: CC206042 0C4A8512
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000078B4: CC20604C 0C4E9913
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000078BC: CC20602F 0C425F10
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000078C4: CC206039 0C467311
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000078CC: CC206043 0C4A8712
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000078D4: CC20604D 0C4E9B13
	s_wait_loadcnt 0x5                                         // 0000000078DC: BFC00005
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 0000000078E0: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 0000000078E8: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x3                                         // 0000000078F0: BFC00003
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 0000000078F4: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 0000000078FC: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000007904: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 00000000790C: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 000000007914: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 00000000791C: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000007924: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 00000000792C: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 000000007934: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 00000000793C: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000007944: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 00000000794C: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 000000007954: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 00000000795C: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000007964: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 00000000796C: C800A940 3A3AA94A
	s_wait_loadcnt 0x2                                         // 000000007974: BFC00002
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 000000007978: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 000000007980: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000007988: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000007990: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 000000007998: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 0000000079A0: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 0000000079A8: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 0000000079B0: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 0000000079B8: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 0000000079C0: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 0000000079C8: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 0000000079D0: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 0000000079D8: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 0000000079E0: C800BF43 4E4EBF4D
	v_dual_add_f32 v184, v184, v48 :: v_dual_add_f32 v185, v185, v49// 0000000079E8: C90861B8 B8B863B9
	v_dual_add_f32 v186, v186, v58 :: v_dual_add_f32 v187, v187, v59// 0000000079F0: C90875BA BABA77BB
	v_dual_add_f32 v200, v200, v68 :: v_dual_add_f32 v201, v201, v69// 0000000079F8: C90889C8 C8C88BC9
	v_dual_add_f32 v202, v202, v78 :: v_dual_add_f32 v203, v203, v79// 000000007A00: C9089DCA CACA9FCB
	s_wait_loadcnt 0x1                                         // 000000007A08: BFC00001
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97// 000000007A0C: C8C6C329 3030C333
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97// 000000007A14: C8C6C33D 3A3AC347
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96// 000000007A1C: C800C128 3030C132
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96// 000000007A24: C800C13C 3A3AC146
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98// 000000007A2C: C800C52A 3030C534
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98// 000000007A34: C800C53E 3A3AC548
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99// 000000007A3C: C800C72B 3030C735
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99// 000000007A44: C800C73F 3A3AC749
	s_wait_loadcnt 0x0                                         // 000000007A4C: BFC00000
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100// 000000007A50: C800C92C 3030C936
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100// 000000007A58: C800C940 3A3AC94A
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101// 000000007A60: C800CB2D 3030CB37
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101// 000000007A68: C800CB41 3A3ACB4B
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102// 000000007A70: C800CD2E 3030CD38
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102// 000000007A78: C800CD42 3A3ACD4C
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103// 000000007A80: C800CF2F 3030CF39
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103// 000000007A88: C800CF43 3A3ACF4D
	v_dual_add_f32 v216, v216, v48 :: v_dual_add_f32 v217, v217, v49// 000000007A90: C90861D8 D8D863D9
	v_dual_add_f32 v218, v218, v58 :: v_dual_add_f32 v219, v219, v59// 000000007A98: C90875DA DADA77DB
	v_dual_add_f32 v176, v176, v180 :: v_dual_add_f32 v177, v177, v181// 000000007AA0: C90969B0 B0B16BB1
	v_dual_add_f32 v178, v178, v182 :: v_dual_add_f32 v179, v179, v183// 000000007AA8: C9096DB2 B2B36FB3
	v_dual_add_f32 v184, v184, v188 :: v_dual_add_f32 v185, v185, v189// 000000007AB0: C90979B8 B8B97BB9
	v_dual_add_f32 v186, v186, v190 :: v_dual_add_f32 v187, v187, v191// 000000007AB8: C9097DBA BABB7FBB
	v_dual_add_f32 v176, v176, v184 :: v_dual_add_f32 v177, v177, v185// 000000007AC0: C90971B0 B0B173B1
	v_dual_add_f32 v178, v178, v186 :: v_dual_add_f32 v179, v179, v187// 000000007AC8: C90975B2 B2B377B3
	v_dual_add_f32 v192, v192, v196 :: v_dual_add_f32 v193, v193, v197// 000000007AD0: C90989C0 C0C18BC1
	v_dual_add_f32 v194, v194, v198 :: v_dual_add_f32 v195, v195, v199// 000000007AD8: C9098DC2 C2C38FC3
	v_dual_add_f32 v200, v200, v204 :: v_dual_add_f32 v201, v201, v205// 000000007AE0: C90999C8 C8C99BC9
	v_dual_add_f32 v202, v202, v206 :: v_dual_add_f32 v203, v203, v207// 000000007AE8: C9099DCA CACB9FCB
	v_dual_add_f32 v192, v192, v200 :: v_dual_add_f32 v193, v193, v201// 000000007AF0: C90991C0 C0C193C1
	v_dual_add_f32 v194, v194, v202 :: v_dual_add_f32 v195, v195, v203// 000000007AF8: C90995C2 C2C397C3
	v_dual_add_f32 v208, v208, v212 :: v_dual_add_f32 v209, v209, v213// 000000007B00: C909A9D0 D0D1ABD1
	v_dual_add_f32 v210, v210, v214 :: v_dual_add_f32 v211, v211, v215// 000000007B08: C909ADD2 D2D3AFD3
	v_dual_add_f32 v216, v216, v220 :: v_dual_add_f32 v217, v217, v221// 000000007B10: C909B9D8 D8D9BBD9
	v_dual_add_f32 v218, v218, v222 :: v_dual_add_f32 v219, v219, v223// 000000007B18: C909BDDA DADBBFDB
	v_dual_add_f32 v208, v208, v216 :: v_dual_add_f32 v209, v209, v217// 000000007B20: C909B1D0 D0D1B3D1
	v_dual_add_f32 v210, v210, v218 :: v_dual_add_f32 v211, v211, v219// 000000007B28: C909B5D2 D2D3B7D3
	ds_swizzle_b32 v8, v176 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B30: D8D4020F 080000B0
	ds_swizzle_b32 v9, v177 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B38: D8D4020F 090000B1
	ds_swizzle_b32 v10, v178 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B40: D8D4020F 0A0000B2
	ds_swizzle_b32 v11, v179 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B48: D8D4020F 0B0000B3
	ds_swizzle_b32 v12, v192 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B50: D8D4020F 0C0000C0
	ds_swizzle_b32 v13, v193 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B58: D8D4020F 0D0000C1
	ds_swizzle_b32 v14, v194 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B60: D8D4020F 0E0000C2
	ds_swizzle_b32 v15, v195 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B68: D8D4020F 0F0000C3
	ds_swizzle_b32 v16, v208 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B70: D8D4020F 100000D0
	ds_swizzle_b32 v17, v209 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B78: D8D4020F 110000D1
	ds_swizzle_b32 v18, v210 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B80: D8D4020F 120000D2
	ds_swizzle_b32 v19, v211 offset:swizzle(BITMASK_PERM,"1pppp")// 000000007B88: D8D4020F 130000D3
	s_wait_dscnt 0xb                                           // 000000007B90: BFC6000B
	v_add_f32_e32 v176, v176, v8                               // 000000007B94: 076011B0
	s_wait_dscnt 0xa                                           // 000000007B98: BFC6000A
	v_add_f32_e32 v177, v177, v9                               // 000000007B9C: 076213B1
	s_wait_dscnt 0x9                                           // 000000007BA0: BFC60009
	v_add_f32_e32 v178, v178, v10                              // 000000007BA4: 076415B2
	s_wait_dscnt 0x8                                           // 000000007BA8: BFC60008
	v_add_f32_e32 v179, v179, v11                              // 000000007BAC: 076617B3
	s_wait_dscnt 0x7                                           // 000000007BB0: BFC60007
	v_add_f32_e32 v192, v192, v12                              // 000000007BB4: 078019C0
	s_wait_dscnt 0x6                                           // 000000007BB8: BFC60006
	v_add_f32_e32 v193, v193, v13                              // 000000007BBC: 07821BC1
	s_wait_dscnt 0x5                                           // 000000007BC0: BFC60005
	v_add_f32_e32 v194, v194, v14                              // 000000007BC4: 07841DC2
	s_wait_dscnt 0x4                                           // 000000007BC8: BFC60004
	v_add_f32_e32 v195, v195, v15                              // 000000007BCC: 07861FC3
	s_wait_dscnt 0x3                                           // 000000007BD0: BFC60003
	v_add_f32_e32 v208, v208, v16                              // 000000007BD4: 07A021D0
	s_wait_dscnt 0x2                                           // 000000007BD8: BFC60002
	v_add_f32_e32 v209, v209, v17                              // 000000007BDC: 07A223D1
	s_wait_dscnt 0x1                                           // 000000007BE0: BFC60001
	v_add_f32_e32 v210, v210, v18                              // 000000007BE4: 07A425D2
	s_wait_dscnt 0x0                                           // 000000007BE8: BFC60000
	v_add_f32_e32 v211, v211, v19                              // 000000007BEC: 07A627D3
	v_cmp_gt_u32_e32 vcc_lo, 24, v0                            // 000000007BF0: 7C980098
	s_wait_alu depctr_va_vcc(0)                                // 000000007BF4: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo                         // 000000007BF8: D5010004 01A91080
	v_add_lshl_u32 v4, v4, v0, 2                               // 000000007C00: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v176                               // 000000007C08: DACC0000 0800B004
	ds_bpermute_b32 v9, v4, v177                               // 000000007C10: DACC0000 0900B104
	ds_bpermute_b32 v10, v4, v178                              // 000000007C18: DACC0000 0A00B204
	ds_bpermute_b32 v11, v4, v179                              // 000000007C20: DACC0000 0B00B304
	ds_bpermute_b32 v12, v4, v192                              // 000000007C28: DACC0000 0C00C004
	ds_bpermute_b32 v13, v4, v193                              // 000000007C30: DACC0000 0D00C104
	ds_bpermute_b32 v14, v4, v194                              // 000000007C38: DACC0000 0E00C204
	ds_bpermute_b32 v15, v4, v195                              // 000000007C40: DACC0000 0F00C304
	ds_bpermute_b32 v16, v4, v208                              // 000000007C48: DACC0000 1000D004
	ds_bpermute_b32 v17, v4, v209                              // 000000007C50: DACC0000 1100D104
	ds_bpermute_b32 v18, v4, v210                              // 000000007C58: DACC0000 1200D204
	ds_bpermute_b32 v19, v4, v211                              // 000000007C60: DACC0000 1300D304
	s_wait_dscnt 0xb                                           // 000000007C68: BFC6000B
	v_add_f32_e32 v176, v176, v8                               // 000000007C6C: 076011B0
	s_wait_dscnt 0xa                                           // 000000007C70: BFC6000A
	v_add_f32_e32 v177, v177, v9                               // 000000007C74: 076213B1
	s_wait_dscnt 0x9                                           // 000000007C78: BFC60009
	v_add_f32_e32 v178, v178, v10                              // 000000007C7C: 076415B2
	s_wait_dscnt 0x8                                           // 000000007C80: BFC60008
	v_add_f32_e32 v179, v179, v11                              // 000000007C84: 076617B3
	s_wait_dscnt 0x7                                           // 000000007C88: BFC60007
	v_add_f32_e32 v192, v192, v12                              // 000000007C8C: 078019C0
	s_wait_dscnt 0x6                                           // 000000007C90: BFC60006
	v_add_f32_e32 v193, v193, v13                              // 000000007C94: 07821BC1
	s_wait_dscnt 0x5                                           // 000000007C98: BFC60005
	v_add_f32_e32 v194, v194, v14                              // 000000007C9C: 07841DC2
	s_wait_dscnt 0x4                                           // 000000007CA0: BFC60004
	v_add_f32_e32 v195, v195, v15                              // 000000007CA4: 07861FC3
	s_wait_dscnt 0x3                                           // 000000007CA8: BFC60003
	v_add_f32_e32 v208, v208, v16                              // 000000007CAC: 07A021D0
	s_wait_dscnt 0x2                                           // 000000007CB0: BFC60002
	v_add_f32_e32 v209, v209, v17                              // 000000007CB4: 07A223D1
	s_wait_dscnt 0x1                                           // 000000007CB8: BFC60001
	v_add_f32_e32 v210, v210, v18                              // 000000007CBC: 07A425D2
	s_wait_dscnt 0x0                                           // 000000007CC0: BFC60000
	v_add_f32_e32 v211, v211, v19                              // 000000007CC4: 07A627D3
	v_cmp_gt_u32_e32 vcc_lo, 28, v0                            // 000000007CC8: 7C98009C
	s_wait_alu depctr_va_vcc(0)                                // 000000007CCC: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo                         // 000000007CD0: D5010004 01A90880
	v_add_lshl_u32 v4, v4, v0, 2                               // 000000007CD8: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v176                               // 000000007CE0: DACC0000 0800B004
	ds_bpermute_b32 v9, v4, v177                               // 000000007CE8: DACC0000 0900B104
	ds_bpermute_b32 v10, v4, v178                              // 000000007CF0: DACC0000 0A00B204
	ds_bpermute_b32 v11, v4, v179                              // 000000007CF8: DACC0000 0B00B304
	ds_bpermute_b32 v12, v4, v192                              // 000000007D00: DACC0000 0C00C004
	ds_bpermute_b32 v13, v4, v193                              // 000000007D08: DACC0000 0D00C104
	ds_bpermute_b32 v14, v4, v194                              // 000000007D10: DACC0000 0E00C204
	ds_bpermute_b32 v15, v4, v195                              // 000000007D18: DACC0000 0F00C304
	ds_bpermute_b32 v16, v4, v208                              // 000000007D20: DACC0000 1000D004
	ds_bpermute_b32 v17, v4, v209                              // 000000007D28: DACC0000 1100D104
	ds_bpermute_b32 v18, v4, v210                              // 000000007D30: DACC0000 1200D204
	ds_bpermute_b32 v19, v4, v211                              // 000000007D38: DACC0000 1300D304
	s_wait_dscnt 0xb                                           // 000000007D40: BFC6000B
	v_add_f32_e32 v176, v176, v8                               // 000000007D44: 076011B0
	s_wait_dscnt 0xa                                           // 000000007D48: BFC6000A
	v_add_f32_e32 v177, v177, v9                               // 000000007D4C: 076213B1
	s_wait_dscnt 0x9                                           // 000000007D50: BFC60009
	v_add_f32_e32 v178, v178, v10                              // 000000007D54: 076415B2
	s_wait_dscnt 0x8                                           // 000000007D58: BFC60008
	v_add_f32_e32 v179, v179, v11                              // 000000007D5C: 076617B3
	s_wait_dscnt 0x7                                           // 000000007D60: BFC60007
	v_add_f32_e32 v192, v192, v12                              // 000000007D64: 078019C0
	s_wait_dscnt 0x6                                           // 000000007D68: BFC60006
	v_add_f32_e32 v193, v193, v13                              // 000000007D6C: 07821BC1
	s_wait_dscnt 0x5                                           // 000000007D70: BFC60005
	v_add_f32_e32 v194, v194, v14                              // 000000007D74: 07841DC2
	s_wait_dscnt 0x4                                           // 000000007D78: BFC60004
	v_add_f32_e32 v195, v195, v15                              // 000000007D7C: 07861FC3
	s_wait_dscnt 0x3                                           // 000000007D80: BFC60003
	v_add_f32_e32 v208, v208, v16                              // 000000007D84: 07A021D0
	s_wait_dscnt 0x2                                           // 000000007D88: BFC60002
	v_add_f32_e32 v209, v209, v17                              // 000000007D8C: 07A223D1
	s_wait_dscnt 0x1                                           // 000000007D90: BFC60001
	v_add_f32_e32 v210, v210, v18                              // 000000007D94: 07A425D2
	s_wait_dscnt 0x0                                           // 000000007D98: BFC60000
	v_add_f32_e32 v211, v211, v19                              // 000000007D9C: 07A627D3
	v_cmp_gt_u32_e32 vcc_lo, 30, v0                            // 000000007DA0: 7C98009E
	s_wait_alu depctr_va_vcc(0)                                // 000000007DA4: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo                         // 000000007DA8: D5010004 01A90480
	v_add_lshl_u32 v4, v4, v0, 2                               // 000000007DB0: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v176                               // 000000007DB8: DACC0000 0800B004
	ds_bpermute_b32 v9, v4, v177                               // 000000007DC0: DACC0000 0900B104
	ds_bpermute_b32 v10, v4, v178                              // 000000007DC8: DACC0000 0A00B204
	ds_bpermute_b32 v11, v4, v179                              // 000000007DD0: DACC0000 0B00B304
	ds_bpermute_b32 v12, v4, v192                              // 000000007DD8: DACC0000 0C00C004
	ds_bpermute_b32 v13, v4, v193                              // 000000007DE0: DACC0000 0D00C104
	ds_bpermute_b32 v14, v4, v194                              // 000000007DE8: DACC0000 0E00C204
	ds_bpermute_b32 v15, v4, v195                              // 000000007DF0: DACC0000 0F00C304
	ds_bpermute_b32 v16, v4, v208                              // 000000007DF8: DACC0000 1000D004
	ds_bpermute_b32 v17, v4, v209                              // 000000007E00: DACC0000 1100D104
	ds_bpermute_b32 v18, v4, v210                              // 000000007E08: DACC0000 1200D204
	ds_bpermute_b32 v19, v4, v211                              // 000000007E10: DACC0000 1300D304
	s_wait_dscnt 0xb                                           // 000000007E18: BFC6000B
	v_add_f32_e32 v176, v176, v8                               // 000000007E1C: 076011B0
	s_wait_dscnt 0xa                                           // 000000007E20: BFC6000A
	v_add_f32_e32 v177, v177, v9                               // 000000007E24: 076213B1
	s_wait_dscnt 0x9                                           // 000000007E28: BFC60009
	v_add_f32_e32 v178, v178, v10                              // 000000007E2C: 076415B2
	s_wait_dscnt 0x8                                           // 000000007E30: BFC60008
	v_add_f32_e32 v179, v179, v11                              // 000000007E34: 076617B3
	s_wait_dscnt 0x7                                           // 000000007E38: BFC60007
	v_add_f32_e32 v192, v192, v12                              // 000000007E3C: 078019C0
	s_wait_dscnt 0x6                                           // 000000007E40: BFC60006
	v_add_f32_e32 v193, v193, v13                              // 000000007E44: 07821BC1
	s_wait_dscnt 0x5                                           // 000000007E48: BFC60005
	v_add_f32_e32 v194, v194, v14                              // 000000007E4C: 07841DC2
	s_wait_dscnt 0x4                                           // 000000007E50: BFC60004
	v_add_f32_e32 v195, v195, v15                              // 000000007E54: 07861FC3
	s_wait_dscnt 0x3                                           // 000000007E58: BFC60003
	v_add_f32_e32 v208, v208, v16                              // 000000007E5C: 07A021D0
	s_wait_dscnt 0x2                                           // 000000007E60: BFC60002
	v_add_f32_e32 v209, v209, v17                              // 000000007E64: 07A223D1
	s_wait_dscnt 0x1                                           // 000000007E68: BFC60001
	v_add_f32_e32 v210, v210, v18                              // 000000007E6C: 07A425D2
	s_wait_dscnt 0x0                                           // 000000007E70: BFC60000
	v_add_f32_e32 v211, v211, v19                              // 000000007E74: 07A627D3
	v_cmp_gt_u32_e32 vcc_lo, 31, v0                            // 000000007E78: 7C98009F
	s_wait_alu depctr_va_vcc(0)                                // 000000007E7C: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo                         // 000000007E80: D5010004 01A90280
	v_add_lshl_u32 v4, v4, v0, 2                               // 000000007E88: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v176                               // 000000007E90: DACC0000 0800B004
	ds_bpermute_b32 v9, v4, v177                               // 000000007E98: DACC0000 0900B104
	ds_bpermute_b32 v10, v4, v178                              // 000000007EA0: DACC0000 0A00B204
	ds_bpermute_b32 v11, v4, v179                              // 000000007EA8: DACC0000 0B00B304
	ds_bpermute_b32 v12, v4, v192                              // 000000007EB0: DACC0000 0C00C004
	ds_bpermute_b32 v13, v4, v193                              // 000000007EB8: DACC0000 0D00C104
	ds_bpermute_b32 v14, v4, v194                              // 000000007EC0: DACC0000 0E00C204
	ds_bpermute_b32 v15, v4, v195                              // 000000007EC8: DACC0000 0F00C304
	ds_bpermute_b32 v16, v4, v208                              // 000000007ED0: DACC0000 1000D004
	ds_bpermute_b32 v17, v4, v209                              // 000000007ED8: DACC0000 1100D104
	ds_bpermute_b32 v18, v4, v210                              // 000000007EE0: DACC0000 1200D204
	ds_bpermute_b32 v19, v4, v211                              // 000000007EE8: DACC0000 1300D304
	s_wait_dscnt 0xb                                           // 000000007EF0: BFC6000B
	v_add_f32_e32 v176, v176, v8                               // 000000007EF4: 076011B0
	s_wait_dscnt 0xa                                           // 000000007EF8: BFC6000A
	v_add_f32_e32 v177, v177, v9                               // 000000007EFC: 076213B1
	s_wait_dscnt 0x9                                           // 000000007F00: BFC60009
	v_add_f32_e32 v178, v178, v10                              // 000000007F04: 076415B2
	s_wait_dscnt 0x8                                           // 000000007F08: BFC60008
	v_add_f32_e32 v179, v179, v11                              // 000000007F0C: 076617B3
	s_wait_dscnt 0x7                                           // 000000007F10: BFC60007
	v_add_f32_e32 v192, v192, v12                              // 000000007F14: 078019C0
	s_wait_dscnt 0x6                                           // 000000007F18: BFC60006
	v_add_f32_e32 v193, v193, v13                              // 000000007F1C: 07821BC1
	s_wait_dscnt 0x5                                           // 000000007F20: BFC60005
	v_add_f32_e32 v194, v194, v14                              // 000000007F24: 07841DC2
	s_wait_dscnt 0x4                                           // 000000007F28: BFC60004
	v_add_f32_e32 v195, v195, v15                              // 000000007F2C: 07861FC3
	s_wait_dscnt 0x3                                           // 000000007F30: BFC60003
	v_add_f32_e32 v208, v208, v16                              // 000000007F34: 07A021D0
	s_wait_dscnt 0x2                                           // 000000007F38: BFC60002
	v_add_f32_e32 v209, v209, v17                              // 000000007F3C: 07A223D1
	s_wait_dscnt 0x1                                           // 000000007F40: BFC60001
	v_add_f32_e32 v210, v210, v18                              // 000000007F44: 07A425D2
	s_wait_dscnt 0x0                                           // 000000007F48: BFC60000
	v_add_f32_e32 v211, v211, v19                              // 000000007F4C: 07A627D3
	v_cmpx_eq_u32_e32 0, v0                                    // 000000007F50: 7D940080
	v_mov_b32_e32 v40, s42                                     // 000000007F54: 7E50022A
	s_mul_i32 s40, s39, 1                                      // 000000007F58: 96288127
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F5C: BF88FF9E
	s_add_co_i32 s40, s40, s42                                 // 000000007F60: 81282A28
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F64: BF88FF9E
	v_mov_b32_e32 v41, s40                                     // 000000007F68: 7E520228
	s_mul_i32 s40, s39, 2                                      // 000000007F6C: 96288227
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F70: BF88FF9E
	s_add_co_i32 s40, s40, s42                                 // 000000007F74: 81282A28
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F78: BF88FF9E
	v_mov_b32_e32 v42, s40                                     // 000000007F7C: 7E540228
	global_store_b32 v40, v176, s[8:9]                         // 000000007F80: EE068008 58000000 00000028
	global_store_b32 v41, v192, s[8:9]                         // 000000007F8C: EE068008 60000000 00000029
	global_store_b32 v42, v208, s[8:9]                         // 000000007F98: EE068008 68000000 0000002A
	s_cmp_le_i32 s13, 1                                        // 000000007FA4: BF05810D
	s_cbranch_scc1 31                                          // 000000007FA8: BFA2001F <gemv_mq2g256v2_xbatch_pm+0x6928>
	global_store_b32 v40, v177, s[8:9] offset:4                // 000000007FAC: EE068008 58800000 00000428
	global_store_b32 v41, v193, s[8:9] offset:4                // 000000007FB8: EE068008 60800000 00000429
	global_store_b32 v42, v209, s[8:9] offset:4                // 000000007FC4: EE068008 68800000 0000042A
	s_cmp_le_i32 s13, 2                                        // 000000007FD0: BF05820D
	s_cbranch_scc1 20                                          // 000000007FD4: BFA20014 <gemv_mq2g256v2_xbatch_pm+0x6928>
	global_store_b32 v40, v178, s[8:9] offset:8                // 000000007FD8: EE068008 59000000 00000828
	global_store_b32 v41, v194, s[8:9] offset:8                // 000000007FE4: EE068008 61000000 00000829
	global_store_b32 v42, v210, s[8:9] offset:8                // 000000007FF0: EE068008 69000000 0000082A
	s_cmp_le_i32 s13, 3                                        // 000000007FFC: BF05830D
	s_cbranch_scc1 9                                           // 000000008000: BFA20009 <gemv_mq2g256v2_xbatch_pm+0x6928>
	global_store_b32 v40, v179, s[8:9] offset:12               // 000000008004: EE068008 59800000 00000C28
	global_store_b32 v41, v195, s[8:9] offset:12               // 000000008010: EE068008 61800000 00000C29
	global_store_b32 v42, v211, s[8:9] offset:12               // 00000000801C: EE068008 69800000 00000C2A
	s_wait_storecnt 0x0                                        // 000000008028: BFC10000
	s_branch 3084                                              // 00000000802C: BFA00C0C <gemv_mq2g256v2_xbatch_pm+0x9960>
	v_mov_b32_e32 v192, 0                                      // 000000008030: 7F800280
	v_mov_b32_e32 v193, 0                                      // 000000008034: 7F820280
	v_mov_b32_e32 v194, 0                                      // 000000008038: 7F840280
	v_mov_b32_e32 v195, 0                                      // 00000000803C: 7F860280
	v_mov_b32_e32 v196, 0                                      // 000000008040: 7F880280
	v_mov_b32_e32 v197, 0                                      // 000000008044: 7F8A0280
	v_mov_b32_e32 v198, 0                                      // 000000008048: 7F8C0280
	v_mov_b32_e32 v199, 0                                      // 00000000804C: 7F8E0280
	v_mov_b32_e32 v200, 0                                      // 000000008050: 7F900280
	v_mov_b32_e32 v201, 0                                      // 000000008054: 7F920280
	v_mov_b32_e32 v202, 0                                      // 000000008058: 7F940280
	v_mov_b32_e32 v203, 0                                      // 00000000805C: 7F960280
	v_mov_b32_e32 v204, 0                                      // 000000008060: 7F980280
	v_mov_b32_e32 v205, 0                                      // 000000008064: 7F9A0280
	v_mov_b32_e32 v206, 0                                      // 000000008068: 7F9C0280
	v_mov_b32_e32 v207, 0                                      // 00000000806C: 7F9E0280
	v_mov_b32_e32 v208, 0                                      // 000000008070: 7FA00280
	v_mov_b32_e32 v209, 0                                      // 000000008074: 7FA20280
	v_mov_b32_e32 v210, 0                                      // 000000008078: 7FA40280
	v_mov_b32_e32 v211, 0                                      // 00000000807C: 7FA60280
	v_mov_b32_e32 v212, 0                                      // 000000008080: 7FA80280
	v_mov_b32_e32 v213, 0                                      // 000000008084: 7FAA0280
	v_mov_b32_e32 v214, 0                                      // 000000008088: 7FAC0280
	v_mov_b32_e32 v215, 0                                      // 00000000808C: 7FAE0280
	v_mov_b32_e32 v216, 0                                      // 000000008090: 7FB00280
	v_mov_b32_e32 v217, 0                                      // 000000008094: 7FB20280
	v_mov_b32_e32 v218, 0                                      // 000000008098: 7FB40280
	v_mov_b32_e32 v219, 0                                      // 00000000809C: 7FB60280
	v_mov_b32_e32 v220, 0                                      // 0000000080A0: 7FB80280
	v_mov_b32_e32 v221, 0                                      // 0000000080A4: 7FBA0280
	v_mov_b32_e32 v222, 0                                      // 0000000080A8: 7FBC0280
	v_mov_b32_e32 v223, 0                                      // 0000000080AC: 7FBE0280
	v_mov_b32_e32 v224, 0                                      // 0000000080B0: 7FC00280
	v_mov_b32_e32 v225, 0                                      // 0000000080B4: 7FC20280
	v_mov_b32_e32 v226, 0                                      // 0000000080B8: 7FC40280
	v_mov_b32_e32 v227, 0                                      // 0000000080BC: 7FC60280
	v_mov_b32_e32 v228, 0                                      // 0000000080C0: 7FC80280
	v_mov_b32_e32 v229, 0                                      // 0000000080C4: 7FCA0280
	v_mov_b32_e32 v230, 0                                      // 0000000080C8: 7FCC0280
	v_mov_b32_e32 v231, 0                                      // 0000000080CC: 7FCE0280
	v_mov_b32_e32 v232, 0                                      // 0000000080D0: 7FD00280
	v_mov_b32_e32 v233, 0                                      // 0000000080D4: 7FD20280
	v_mov_b32_e32 v234, 0                                      // 0000000080D8: 7FD40280
	v_mov_b32_e32 v235, 0                                      // 0000000080DC: 7FD60280
	v_mov_b32_e32 v236, 0                                      // 0000000080E0: 7FD80280
	v_mov_b32_e32 v237, 0                                      // 0000000080E4: 7FDA0280
	v_mov_b32_e32 v238, 0                                      // 0000000080E8: 7FDC0280
	v_mov_b32_e32 v239, 0                                      // 0000000080EC: 7FDE0280
	v_mov_b32_e32 v240, 0                                      // 0000000080F0: 7FE00280
	v_mov_b32_e32 v241, 0                                      // 0000000080F4: 7FE20280
	v_mov_b32_e32 v242, 0                                      // 0000000080F8: 7FE40280
	v_mov_b32_e32 v243, 0                                      // 0000000080FC: 7FE60280
	v_mov_b32_e32 v244, 0                                      // 000000008100: 7FE80280
	v_mov_b32_e32 v245, 0                                      // 000000008104: 7FEA0280
	v_mov_b32_e32 v246, 0                                      // 000000008108: 7FEC0280
	v_mov_b32_e32 v247, 0                                      // 00000000810C: 7FEE0280
	v_mov_b32_e32 v248, 0                                      // 000000008110: 7FF00280
	v_mov_b32_e32 v249, 0                                      // 000000008114: 7FF20280
	v_mov_b32_e32 v250, 0                                      // 000000008118: 7FF40280
	v_mov_b32_e32 v251, 0                                      // 00000000811C: 7FF60280
	v_mov_b32_e32 v252, 0                                      // 000000008120: 7FF80280
	v_mov_b32_e32 v253, 0                                      // 000000008124: 7FFA0280
	v_mov_b32_e32 v254, 0                                      // 000000008128: 7FFC0280
	v_mov_b32_e32 v255, 0                                      // 00000000812C: 7FFE0280
	s_lshr_b32 s36, s14, 2                                     // 000000008130: 8524820E
	s_and_b32 s37, s14, 3                                      // 000000008134: 8B25830E
	s_cmp_eq_u32 s36, 0                                        // 000000008138: BF068024
	s_cbranch_scc1 1459                                        // 00000000813C: BFA205B3 <gemv_mq2g256v2_xbatch_pm+0x810c>
	buffer_load_b32 v8, v1, s[20:23], s16 offen                // 000000008140: C4050010 40802808 00000001
	buffer_load_b32 v9, v1, s[20:23], s17 offen                // 00000000814C: C4050011 40802809 00000001
	buffer_load_b32 v10, v1, s[20:23], s18 offen               // 000000008158: C4050012 4080280A 00000001
	buffer_load_b32 v11, v1, s[20:23], s19 offen               // 000000008164: C4050013 4080280B 00000001
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8      // 000000008170: C4050010 40802818 00000802
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8      // 00000000817C: C4050011 40802819 00000802
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8      // 000000008188: C4050012 4080281A 00000802
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8      // 000000008194: C4050013 4080281B 00000802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen         // 0000000081A0: C405C01C 40803050 00000003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16// 0000000081AC: C405C01C 40803054 00001003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen         // 0000000081B8: C405C01D 40803058 00000003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16// 0000000081C4: C405C01D 4080305C 00001003
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen         // 0000000081D0: C405C01E 40803060 00000003
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16// 0000000081DC: C405C01E 40803064 00001003
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen       // 0000000081E8: C405C01F 40803068 00000003
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16// 0000000081F4: C405C01F 4080306C 00001003
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:72     // 000000008200: C4050010 4080280C 00004801
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:72     // 00000000820C: C4050011 4080280D 00004801
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:72     // 000000008218: C4050012 4080280E 00004801
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:72     // 000000008224: C4050013 4080280F 00004801
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:80     // 000000008230: C4050010 4080281C 00005002
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:80     // 00000000823C: C4050011 4080281D 00005002
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:80     // 000000008248: C4050012 4080281E 00005002
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:80     // 000000008254: C4050013 4080281F 00005002
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:144    // 000000008260: C4050010 40802810 00009001
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:144    // 00000000826C: C4050011 40802811 00009001
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:144    // 000000008278: C4050012 40802812 00009001
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:144    // 000000008284: C4050013 40802813 00009001
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:152    // 000000008290: C4050010 40802820 00009802
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:152    // 00000000829C: C4050011 40802821 00009802
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:152    // 0000000082A8: C4050012 40802822 00009802
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:152    // 0000000082B4: C4050013 40802823 00009802
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:216    // 0000000082C0: C4050010 40802814 0000D801
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:216    // 0000000082CC: C4050011 40802815 0000D801
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:216    // 0000000082D8: C4050012 40802816 0000D801
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:216    // 0000000082E4: C4050013 40802817 0000D801
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:224    // 0000000082F0: C4050010 40802824 0000E002
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:224    // 0000000082FC: C4050011 40802825 0000E002
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:224    // 000000008308: C4050012 40802826 0000E002
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:224    // 000000008314: C4050013 40802827 0000E002
	s_wait_loadcnt 0x23                                        // 000000008320: BFC00023
	v_bfe_u32 v40, v24, 0, 2                                   // 000000008324: D6100028 02090118
	s_wait_loadcnt 0x22                                        // 00000000832C: BFC00022
	v_bfe_u32 v50, v25, 0, 2                                   // 000000008330: D6100032 02090119
	s_wait_loadcnt 0x21                                        // 000000008338: BFC00021
	v_bfe_u32 v60, v26, 0, 2                                   // 00000000833C: D610003C 0209011A
	s_wait_loadcnt 0x20                                        // 000000008344: BFC00020
	v_bfe_u32 v70, v27, 0, 2                                   // 000000008348: D6100046 0209011B
	v_bfe_u32 v41, v24, 2, 2                                   // 000000008350: D6100029 02090518
	v_bfe_u32 v51, v25, 2, 2                                   // 000000008358: D6100033 02090519
	v_bfe_u32 v61, v26, 2, 2                                   // 000000008360: D610003D 0209051A
	v_bfe_u32 v71, v27, 2, 2                                   // 000000008368: D6100047 0209051B
	v_bfe_u32 v42, v24, 4, 2                                   // 000000008370: D610002A 02090918
	v_bfe_u32 v52, v25, 4, 2                                   // 000000008378: D6100034 02090919
	v_bfe_u32 v62, v26, 4, 2                                   // 000000008380: D610003E 0209091A
	v_bfe_u32 v72, v27, 4, 2                                   // 000000008388: D6100048 0209091B
	v_bfe_u32 v43, v24, 6, 2                                   // 000000008390: D610002B 02090D18
	v_bfe_u32 v53, v25, 6, 2                                   // 000000008398: D6100035 02090D19
	v_bfe_u32 v63, v26, 6, 2                                   // 0000000083A0: D610003F 02090D1A
	v_bfe_u32 v73, v27, 6, 2                                   // 0000000083A8: D6100049 02090D1B
	v_bfe_u32 v44, v24, 8, 2                                   // 0000000083B0: D610002C 02091118
	v_bfe_u32 v54, v25, 8, 2                                   // 0000000083B8: D6100036 02091119
	v_bfe_u32 v64, v26, 8, 2                                   // 0000000083C0: D6100040 0209111A
	v_bfe_u32 v74, v27, 8, 2                                   // 0000000083C8: D610004A 0209111B
	v_bfe_u32 v45, v24, 10, 2                                  // 0000000083D0: D610002D 02091518
	v_bfe_u32 v55, v25, 10, 2                                  // 0000000083D8: D6100037 02091519
	v_bfe_u32 v65, v26, 10, 2                                  // 0000000083E0: D6100041 0209151A
	v_bfe_u32 v75, v27, 10, 2                                  // 0000000083E8: D610004B 0209151B
	v_bfe_u32 v46, v24, 12, 2                                  // 0000000083F0: D610002E 02091918
	v_bfe_u32 v56, v25, 12, 2                                  // 0000000083F8: D6100038 02091919
	v_bfe_u32 v66, v26, 12, 2                                  // 000000008400: D6100042 0209191A
	v_bfe_u32 v76, v27, 12, 2                                  // 000000008408: D610004C 0209191B
	v_bfe_u32 v47, v24, 14, 2                                  // 000000008410: D610002F 02091D18
	v_bfe_u32 v57, v25, 14, 2                                  // 000000008418: D6100039 02091D19
	v_bfe_u32 v67, v26, 14, 2                                  // 000000008420: D6100043 02091D1A
	v_bfe_u32 v77, v27, 14, 2                                  // 000000008428: D610004D 02091D1B
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000008430: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000008434: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000008438: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 00000000843C: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000008440: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000008444: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000008448: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 00000000844C: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000008450: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000008454: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000008458: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 00000000845C: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000008460: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000008464: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000008468: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 00000000846C: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000008470: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000008474: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000008478: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 00000000847C: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000008480: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000008484: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000008488: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 00000000848C: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000008490: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000008494: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000008498: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 00000000849C: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 0000000084A0: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 0000000084A4: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 0000000084A8: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 0000000084AC: 7E9A234D
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000084B0: CC206028 0C225108
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000084B8: CC206032 0C266509
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000084C0: CC20603C 0C2A790A
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000084C8: CC206046 0C2E8D0B
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000084D0: CC206029 0C225308
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000084D8: CC206033 0C266709
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000084E0: CC20603D 0C2A7B0A
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000084E8: CC206047 0C2E8F0B
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000084F0: CC20602A 0C225508
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000084F8: CC206034 0C266909
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008500: CC20603E 0C2A7D0A
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008508: CC206048 0C2E910B
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008510: CC20602B 0C225708
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008518: CC206035 0C266B09
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008520: CC20603F 0C2A7F0A
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008528: CC206049 0C2E930B
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008530: CC20602C 0C225908
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008538: CC206036 0C266D09
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008540: CC206040 0C2A810A
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008548: CC20604A 0C2E950B
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008550: CC20602D 0C225B08
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008558: CC206037 0C266F09
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008560: CC206041 0C2A830A
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008568: CC20604B 0C2E970B
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008570: CC20602E 0C225D08
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008578: CC206038 0C267109
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008580: CC206042 0C2A850A
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008588: CC20604C 0C2E990B
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008590: CC20602F 0C225F08
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008598: CC206039 0C267309
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000085A0: CC206043 0C2A870A
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000085A8: CC20604D 0C2E9B0B
	buffer_load_b128 v[112:115], v3, s[24:27], s28 offen offset:1024// 0000000085B0: C405C01C 40803070 00040003
	buffer_load_b128 v[116:119], v3, s[24:27], s28 offen offset:1040// 0000000085BC: C405C01C 40803074 00041003
	buffer_load_b128 v[120:123], v3, s[24:27], s29 offen offset:1024// 0000000085C8: C405C01D 40803078 00040003
	buffer_load_b128 v[124:127], v3, s[24:27], s29 offen offset:1040// 0000000085D4: C405C01D 4080307C 00041003
	buffer_load_b128 v[128:131], v3, s[24:27], s30 offen offset:1024// 0000000085E0: C405C01E 40803080 00040003
	buffer_load_b128 v[132:135], v3, s[24:27], s30 offen offset:1040// 0000000085EC: C405C01E 40803084 00041003
	buffer_load_b128 v[136:139], v3, s[24:27], s31 offen offset:1024// 0000000085F8: C405C01F 40803088 00040003
	buffer_load_b128 v[140:143], v3, s[24:27], s31 offen offset:1040// 000000008604: C405C01F 4080308C 00041003
	buffer_load_b128 v[144:147], v3, s[24:27], s28 offen offset:2048// 000000008610: C405C01C 40803090 00080003
	buffer_load_b128 v[148:151], v3, s[24:27], s28 offen offset:2064// 00000000861C: C405C01C 40803094 00081003
	buffer_load_b128 v[152:155], v3, s[24:27], s29 offen offset:2048// 000000008628: C405C01D 40803098 00080003
	buffer_load_b128 v[156:159], v3, s[24:27], s29 offen offset:2064// 000000008634: C405C01D 4080309C 00081003
	buffer_load_b128 v[160:163], v3, s[24:27], s30 offen offset:2048// 000000008640: C405C01E 408030A0 00080003
	buffer_load_b128 v[164:167], v3, s[24:27], s30 offen offset:2064// 00000000864C: C405C01E 408030A4 00081003
	buffer_load_b128 v[168:171], v3, s[24:27], s31 offen offset:2048// 000000008658: C405C01F 408030A8 00080003
	buffer_load_b128 v[172:175], v3, s[24:27], s31 offen offset:2064// 000000008664: C405C01F 408030AC 00081003
	buffer_load_b128 v[176:179], v3, s[24:27], s28 offen offset:3072// 000000008670: C405C01C 408030B0 000C0003
	buffer_load_b128 v[180:183], v3, s[24:27], s28 offen offset:3088// 00000000867C: C405C01C 408030B4 000C1003
	buffer_load_b128 v[184:187], v3, s[24:27], s29 offen offset:3072// 000000008688: C405C01D 408030B8 000C0003
	buffer_load_b128 v[188:191], v3, s[24:27], s29 offen offset:3088// 000000008694: C405C01D 408030BC 000C1003
	s_wait_loadcnt 0x33                                        // 0000000086A0: BFC00033
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 0000000086A4: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 0000000086AC: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x31                                        // 0000000086B4: BFC00031
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 0000000086B8: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 0000000086C0: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 0000000086C8: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 0000000086D0: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 0000000086D8: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 0000000086E0: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 0000000086E8: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 0000000086F0: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 0000000086F8: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 000000008700: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000008708: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000008710: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 000000008718: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 000000008720: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000008728: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000008730: C800A940 3A3AA94A
	s_wait_loadcnt 0x30                                        // 000000008738: BFC00030
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 00000000873C: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 000000008744: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 00000000874C: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000008754: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 00000000875C: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 000000008764: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 00000000876C: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000008774: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 00000000877C: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 000000008784: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 00000000878C: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 000000008794: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 00000000879C: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 0000000087A4: C800BF43 4E4EBF4D
	v_dual_add_f32 v192, v192, v48 :: v_dual_add_f32 v193, v193, v49// 0000000087AC: C90861C0 C0C063C1
	v_dual_add_f32 v194, v194, v58 :: v_dual_add_f32 v195, v195, v59// 0000000087B4: C90875C2 C2C277C3
	v_dual_add_f32 v208, v208, v68 :: v_dual_add_f32 v209, v209, v69// 0000000087BC: C90889D0 D0D08BD1
	v_dual_add_f32 v210, v210, v78 :: v_dual_add_f32 v211, v211, v79// 0000000087C4: C9089DD2 D2D29FD3
	buffer_load_b128 v[80:83], v3, s[24:27], s30 offen offset:3072// 0000000087CC: C405C01E 40803050 000C0003
	buffer_load_b128 v[84:87], v3, s[24:27], s30 offen offset:3088// 0000000087D8: C405C01E 40803054 000C1003
	buffer_load_b128 v[88:91], v3, s[24:27], s31 offen offset:3072// 0000000087E4: C405C01F 40803058 000C0003
	buffer_load_b128 v[92:95], v3, s[24:27], s31 offen offset:3088// 0000000087F0: C405C01F 4080305C 000C1003
	s_wait_loadcnt 0x33                                        // 0000000087FC: BFC00033
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97// 000000008800: C8C6C329 3030C333
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97// 000000008808: C8C6C33D 3A3AC347
	s_wait_loadcnt 0x31                                        // 000000008810: BFC00031
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105// 000000008814: C8C6D329 4444D333
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105// 00000000881C: C8C6D33D 4E4ED347
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96// 000000008824: C800C128 3030C132
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96// 00000000882C: C800C13C 3A3AC146
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104// 000000008834: C800D128 4444D132
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104// 00000000883C: C800D13C 4E4ED146
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98// 000000008844: C800C52A 3030C534
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98// 00000000884C: C800C53E 3A3AC548
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106// 000000008854: C800D52A 4444D534
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106// 00000000885C: C800D53E 4E4ED548
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99// 000000008864: C800C72B 3030C735
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99// 00000000886C: C800C73F 3A3AC749
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107// 000000008874: C800D72B 4444D735
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107// 00000000887C: C800D73F 4E4ED749
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100// 000000008884: C800C92C 3030C936
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100// 00000000888C: C800C940 3A3AC94A
	s_wait_loadcnt 0x30                                        // 000000008894: BFC00030
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108// 000000008898: C800D92C 4444D936
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108// 0000000088A0: C800D940 4E4ED94A
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101// 0000000088A8: C800CB2D 3030CB37
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101// 0000000088B0: C800CB41 3A3ACB4B
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109// 0000000088B8: C800DB2D 4444DB37
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109// 0000000088C0: C800DB41 4E4EDB4B
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102// 0000000088C8: C800CD2E 3030CD38
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102// 0000000088D0: C800CD42 3A3ACD4C
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110// 0000000088D8: C800DD2E 4444DD38
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110// 0000000088E0: C800DD42 4E4EDD4C
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103// 0000000088E8: C800CF2F 3030CF39
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103// 0000000088F0: C800CF43 3A3ACF4D
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111// 0000000088F8: C800DF2F 4444DF39
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111// 000000008900: C800DF43 4E4EDF4D
	v_dual_add_f32 v224, v224, v48 :: v_dual_add_f32 v225, v225, v49// 000000008908: C90861E0 E0E063E1
	v_dual_add_f32 v226, v226, v58 :: v_dual_add_f32 v227, v227, v59// 000000008910: C90875E2 E2E277E3
	v_dual_add_f32 v240, v240, v68 :: v_dual_add_f32 v241, v241, v69// 000000008918: C90889F0 F0F08BF1
	v_dual_add_f32 v242, v242, v78 :: v_dual_add_f32 v243, v243, v79// 000000008920: C9089DF2 F2F29FF3
	s_wait_loadcnt 0x2b                                        // 000000008928: BFC0002B
	v_bfe_u32 v40, v28, 0, 2                                   // 00000000892C: D6100028 0209011C
	s_wait_loadcnt 0x2a                                        // 000000008934: BFC0002A
	v_bfe_u32 v50, v29, 0, 2                                   // 000000008938: D6100032 0209011D
	s_wait_loadcnt 0x29                                        // 000000008940: BFC00029
	v_bfe_u32 v60, v30, 0, 2                                   // 000000008944: D610003C 0209011E
	s_wait_loadcnt 0x28                                        // 00000000894C: BFC00028
	v_bfe_u32 v70, v31, 0, 2                                   // 000000008950: D6100046 0209011F
	v_bfe_u32 v41, v28, 2, 2                                   // 000000008958: D6100029 0209051C
	v_bfe_u32 v51, v29, 2, 2                                   // 000000008960: D6100033 0209051D
	v_bfe_u32 v61, v30, 2, 2                                   // 000000008968: D610003D 0209051E
	v_bfe_u32 v71, v31, 2, 2                                   // 000000008970: D6100047 0209051F
	v_bfe_u32 v42, v28, 4, 2                                   // 000000008978: D610002A 0209091C
	v_bfe_u32 v52, v29, 4, 2                                   // 000000008980: D6100034 0209091D
	v_bfe_u32 v62, v30, 4, 2                                   // 000000008988: D610003E 0209091E
	v_bfe_u32 v72, v31, 4, 2                                   // 000000008990: D6100048 0209091F
	v_bfe_u32 v43, v28, 6, 2                                   // 000000008998: D610002B 02090D1C
	v_bfe_u32 v53, v29, 6, 2                                   // 0000000089A0: D6100035 02090D1D
	v_bfe_u32 v63, v30, 6, 2                                   // 0000000089A8: D610003F 02090D1E
	v_bfe_u32 v73, v31, 6, 2                                   // 0000000089B0: D6100049 02090D1F
	v_bfe_u32 v44, v28, 8, 2                                   // 0000000089B8: D610002C 0209111C
	v_bfe_u32 v54, v29, 8, 2                                   // 0000000089C0: D6100036 0209111D
	v_bfe_u32 v64, v30, 8, 2                                   // 0000000089C8: D6100040 0209111E
	v_bfe_u32 v74, v31, 8, 2                                   // 0000000089D0: D610004A 0209111F
	v_bfe_u32 v45, v28, 10, 2                                  // 0000000089D8: D610002D 0209151C
	v_bfe_u32 v55, v29, 10, 2                                  // 0000000089E0: D6100037 0209151D
	v_bfe_u32 v65, v30, 10, 2                                  // 0000000089E8: D6100041 0209151E
	v_bfe_u32 v75, v31, 10, 2                                  // 0000000089F0: D610004B 0209151F
	v_bfe_u32 v46, v28, 12, 2                                  // 0000000089F8: D610002E 0209191C
	v_bfe_u32 v56, v29, 12, 2                                  // 000000008A00: D6100038 0209191D
	v_bfe_u32 v66, v30, 12, 2                                  // 000000008A08: D6100042 0209191E
	v_bfe_u32 v76, v31, 12, 2                                  // 000000008A10: D610004C 0209191F
	v_bfe_u32 v47, v28, 14, 2                                  // 000000008A18: D610002F 02091D1C
	v_bfe_u32 v57, v29, 14, 2                                  // 000000008A20: D6100039 02091D1D
	v_bfe_u32 v67, v30, 14, 2                                  // 000000008A28: D6100043 02091D1E
	v_bfe_u32 v77, v31, 14, 2                                  // 000000008A30: D610004D 02091D1F
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000008A38: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000008A3C: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000008A40: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000008A44: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000008A48: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000008A4C: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000008A50: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000008A54: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000008A58: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000008A5C: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000008A60: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000008A64: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000008A68: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000008A6C: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000008A70: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000008A74: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000008A78: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000008A7C: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000008A80: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000008A84: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000008A88: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000008A8C: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000008A90: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000008A94: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000008A98: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000008A9C: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000008AA0: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000008AA4: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000008AA8: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000008AAC: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000008AB0: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000008AB4: 7E9A234D
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008AB8: CC206028 0C32510C
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008AC0: CC206032 0C36650D
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008AC8: CC20603C 0C3A790E
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008AD0: CC206046 0C3E8D0F
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008AD8: CC206029 0C32530C
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008AE0: CC206033 0C36670D
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008AE8: CC20603D 0C3A7B0E
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008AF0: CC206047 0C3E8F0F
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008AF8: CC20602A 0C32550C
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B00: CC206034 0C36690D
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B08: CC20603E 0C3A7D0E
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B10: CC206048 0C3E910F
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B18: CC20602B 0C32570C
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B20: CC206035 0C366B0D
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B28: CC20603F 0C3A7F0E
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B30: CC206049 0C3E930F
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B38: CC20602C 0C32590C
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B40: CC206036 0C366D0D
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B48: CC206040 0C3A810E
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B50: CC20604A 0C3E950F
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B58: CC20602D 0C325B0C
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B60: CC206037 0C366F0D
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B68: CC206041 0C3A830E
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B70: CC20604B 0C3E970F
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B78: CC20602E 0C325D0C
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B80: CC206038 0C36710D
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B88: CC206042 0C3A850E
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B90: CC20604C 0C3E990F
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008B98: CC20602F 0C325F0C
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008BA0: CC206039 0C36730D
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008BA8: CC206043 0C3A870E
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008BB0: CC20604D 0C3E9B0F
	s_wait_loadcnt 0x17                                        // 000000008BB8: BFC00017
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113// 000000008BBC: C8C6E329 3030E333
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113// 000000008BC4: C8C6E33D 3A3AE347
	s_wait_loadcnt 0x15                                        // 000000008BCC: BFC00015
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121// 000000008BD0: C8C6F329 4444F333
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121// 000000008BD8: C8C6F33D 4E4EF347
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112// 000000008BE0: C800E128 3030E132
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112// 000000008BE8: C800E13C 3A3AE146
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120// 000000008BF0: C800F128 4444F132
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120// 000000008BF8: C800F13C 4E4EF146
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114// 000000008C00: C800E52A 3030E534
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114// 000000008C08: C800E53E 3A3AE548
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122// 000000008C10: C800F52A 4444F534
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122// 000000008C18: C800F53E 4E4EF548
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115// 000000008C20: C800E72B 3030E735
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115// 000000008C28: C800E73F 3A3AE749
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123// 000000008C30: C800F72B 4444F735
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123// 000000008C38: C800F73F 4E4EF749
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116// 000000008C40: C800E92C 3030E936
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116// 000000008C48: C800E940 3A3AE94A
	s_wait_loadcnt 0x14                                        // 000000008C50: BFC00014
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124// 000000008C54: C800F92C 4444F936
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124// 000000008C5C: C800F940 4E4EF94A
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117// 000000008C64: C800EB2D 3030EB37
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117// 000000008C6C: C800EB41 3A3AEB4B
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125// 000000008C74: C800FB2D 4444FB37
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125// 000000008C7C: C800FB41 4E4EFB4B
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118// 000000008C84: C800ED2E 3030ED38
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118// 000000008C8C: C800ED42 3A3AED4C
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126// 000000008C94: C800FD2E 4444FD38
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126// 000000008C9C: C800FD42 4E4EFD4C
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119// 000000008CA4: C800EF2F 3030EF39
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119// 000000008CAC: C800EF43 3A3AEF4D
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127// 000000008CB4: C800FF2F 4444FF39
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127// 000000008CBC: C800FF43 4E4EFF4D
	v_dual_add_f32 v196, v196, v48 :: v_dual_add_f32 v197, v197, v49// 000000008CC4: C90861C4 C4C463C5
	v_dual_add_f32 v198, v198, v58 :: v_dual_add_f32 v199, v199, v59// 000000008CCC: C90875C6 C6C677C7
	v_dual_add_f32 v212, v212, v68 :: v_dual_add_f32 v213, v213, v69// 000000008CD4: C90889D4 D4D48BD5
	v_dual_add_f32 v214, v214, v78 :: v_dual_add_f32 v215, v215, v79// 000000008CDC: C9089DD6 D6D69FD7
	s_wait_loadcnt 0x13                                        // 000000008CE4: BFC00013
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129// 000000008CE8: C8C70329 30310333
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129// 000000008CF0: C8C7033D 3A3B0347
	s_wait_loadcnt 0x11                                        // 000000008CF8: BFC00011
	v_dual_mul_f32 v68, v41, v137 :: v_dual_mul_f32 v69, v51, v137// 000000008CFC: C8C71329 44451333
	v_dual_mul_f32 v78, v61, v137 :: v_dual_mul_f32 v79, v71, v137// 000000008D04: C8C7133D 4E4F1347
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128// 000000008D0C: C8010128 30310132
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128// 000000008D14: C801013C 3A3B0146
	v_dual_fmac_f32 v68, v40, v136 :: v_dual_fmac_f32 v69, v50, v136// 000000008D1C: C8011128 44451132
	v_dual_fmac_f32 v78, v60, v136 :: v_dual_fmac_f32 v79, v70, v136// 000000008D24: C801113C 4E4F1146
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130// 000000008D2C: C801052A 30310534
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130// 000000008D34: C801053E 3A3B0548
	v_dual_fmac_f32 v68, v42, v138 :: v_dual_fmac_f32 v69, v52, v138// 000000008D3C: C801152A 44451534
	v_dual_fmac_f32 v78, v62, v138 :: v_dual_fmac_f32 v79, v72, v138// 000000008D44: C801153E 4E4F1548
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131// 000000008D4C: C801072B 30310735
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131// 000000008D54: C801073F 3A3B0749
	v_dual_fmac_f32 v68, v43, v139 :: v_dual_fmac_f32 v69, v53, v139// 000000008D5C: C801172B 44451735
	v_dual_fmac_f32 v78, v63, v139 :: v_dual_fmac_f32 v79, v73, v139// 000000008D64: C801173F 4E4F1749
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132// 000000008D6C: C801092C 30310936
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132// 000000008D74: C8010940 3A3B094A
	s_wait_loadcnt 0x10                                        // 000000008D7C: BFC00010
	v_dual_fmac_f32 v68, v44, v140 :: v_dual_fmac_f32 v69, v54, v140// 000000008D80: C801192C 44451936
	v_dual_fmac_f32 v78, v64, v140 :: v_dual_fmac_f32 v79, v74, v140// 000000008D88: C8011940 4E4F194A
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133// 000000008D90: C8010B2D 30310B37
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133// 000000008D98: C8010B41 3A3B0B4B
	v_dual_fmac_f32 v68, v45, v141 :: v_dual_fmac_f32 v69, v55, v141// 000000008DA0: C8011B2D 44451B37
	v_dual_fmac_f32 v78, v65, v141 :: v_dual_fmac_f32 v79, v75, v141// 000000008DA8: C8011B41 4E4F1B4B
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134// 000000008DB0: C8010D2E 30310D38
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134// 000000008DB8: C8010D42 3A3B0D4C
	v_dual_fmac_f32 v68, v46, v142 :: v_dual_fmac_f32 v69, v56, v142// 000000008DC0: C8011D2E 44451D38
	v_dual_fmac_f32 v78, v66, v142 :: v_dual_fmac_f32 v79, v76, v142// 000000008DC8: C8011D42 4E4F1D4C
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135// 000000008DD0: C8010F2F 30310F39
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135// 000000008DD8: C8010F43 3A3B0F4D
	v_dual_fmac_f32 v68, v47, v143 :: v_dual_fmac_f32 v69, v57, v143// 000000008DE0: C8011F2F 44451F39
	v_dual_fmac_f32 v78, v67, v143 :: v_dual_fmac_f32 v79, v77, v143// 000000008DE8: C8011F43 4E4F1F4D
	v_dual_add_f32 v228, v228, v48 :: v_dual_add_f32 v229, v229, v49// 000000008DF0: C90861E4 E4E463E5
	v_dual_add_f32 v230, v230, v58 :: v_dual_add_f32 v231, v231, v59// 000000008DF8: C90875E6 E6E677E7
	v_dual_add_f32 v244, v244, v68 :: v_dual_add_f32 v245, v245, v69// 000000008E00: C90889F4 F4F48BF5
	v_dual_add_f32 v246, v246, v78 :: v_dual_add_f32 v247, v247, v79// 000000008E08: C9089DF6 F6F69FF7
	v_bfe_u32 v40, v32, 0, 2                                   // 000000008E10: D6100028 02090120
	v_bfe_u32 v50, v33, 0, 2                                   // 000000008E18: D6100032 02090121
	v_bfe_u32 v60, v34, 0, 2                                   // 000000008E20: D610003C 02090122
	v_bfe_u32 v70, v35, 0, 2                                   // 000000008E28: D6100046 02090123
	v_bfe_u32 v41, v32, 2, 2                                   // 000000008E30: D6100029 02090520
	v_bfe_u32 v51, v33, 2, 2                                   // 000000008E38: D6100033 02090521
	v_bfe_u32 v61, v34, 2, 2                                   // 000000008E40: D610003D 02090522
	v_bfe_u32 v71, v35, 2, 2                                   // 000000008E48: D6100047 02090523
	v_bfe_u32 v42, v32, 4, 2                                   // 000000008E50: D610002A 02090920
	v_bfe_u32 v52, v33, 4, 2                                   // 000000008E58: D6100034 02090921
	v_bfe_u32 v62, v34, 4, 2                                   // 000000008E60: D610003E 02090922
	v_bfe_u32 v72, v35, 4, 2                                   // 000000008E68: D6100048 02090923
	v_bfe_u32 v43, v32, 6, 2                                   // 000000008E70: D610002B 02090D20
	v_bfe_u32 v53, v33, 6, 2                                   // 000000008E78: D6100035 02090D21
	v_bfe_u32 v63, v34, 6, 2                                   // 000000008E80: D610003F 02090D22
	v_bfe_u32 v73, v35, 6, 2                                   // 000000008E88: D6100049 02090D23
	v_bfe_u32 v44, v32, 8, 2                                   // 000000008E90: D610002C 02091120
	v_bfe_u32 v54, v33, 8, 2                                   // 000000008E98: D6100036 02091121
	v_bfe_u32 v64, v34, 8, 2                                   // 000000008EA0: D6100040 02091122
	v_bfe_u32 v74, v35, 8, 2                                   // 000000008EA8: D610004A 02091123
	v_bfe_u32 v45, v32, 10, 2                                  // 000000008EB0: D610002D 02091520
	v_bfe_u32 v55, v33, 10, 2                                  // 000000008EB8: D6100037 02091521
	v_bfe_u32 v65, v34, 10, 2                                  // 000000008EC0: D6100041 02091522
	v_bfe_u32 v75, v35, 10, 2                                  // 000000008EC8: D610004B 02091523
	v_bfe_u32 v46, v32, 12, 2                                  // 000000008ED0: D610002E 02091920
	v_bfe_u32 v56, v33, 12, 2                                  // 000000008ED8: D6100038 02091921
	v_bfe_u32 v66, v34, 12, 2                                  // 000000008EE0: D6100042 02091922
	v_bfe_u32 v76, v35, 12, 2                                  // 000000008EE8: D610004C 02091923
	v_bfe_u32 v47, v32, 14, 2                                  // 000000008EF0: D610002F 02091D20
	v_bfe_u32 v57, v33, 14, 2                                  // 000000008EF8: D6100039 02091D21
	v_bfe_u32 v67, v34, 14, 2                                  // 000000008F00: D6100043 02091D22
	v_bfe_u32 v77, v35, 14, 2                                  // 000000008F08: D610004D 02091D23
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000008F10: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000008F14: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000008F18: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000008F1C: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000008F20: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000008F24: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000008F28: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000008F2C: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000008F30: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000008F34: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000008F38: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000008F3C: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000008F40: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000008F44: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000008F48: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000008F4C: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000008F50: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000008F54: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000008F58: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000008F5C: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000008F60: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000008F64: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000008F68: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000008F6C: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000008F70: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000008F74: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000008F78: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000008F7C: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000008F80: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000008F84: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000008F88: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000008F8C: 7E9A234D
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008F90: CC206028 0C425110
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008F98: CC206032 0C466511
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FA0: CC20603C 0C4A7912
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FA8: CC206046 0C4E8D13
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FB0: CC206029 0C425310
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FB8: CC206033 0C466711
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FC0: CC20603D 0C4A7B12
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FC8: CC206047 0C4E8F13
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FD0: CC20602A 0C425510
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FD8: CC206034 0C466911
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FE0: CC20603E 0C4A7D12
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FE8: CC206048 0C4E9113
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FF0: CC20602B 0C425710
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000008FF8: CC206035 0C466B11
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009000: CC20603F 0C4A7F12
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009008: CC206049 0C4E9313
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009010: CC20602C 0C425910
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009018: CC206036 0C466D11
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009020: CC206040 0C4A8112
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009028: CC20604A 0C4E9513
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009030: CC20602D 0C425B10
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009038: CC206037 0C466F11
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009040: CC206041 0C4A8312
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009048: CC20604B 0C4E9713
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009050: CC20602E 0C425D10
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009058: CC206038 0C467111
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009060: CC206042 0C4A8512
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009068: CC20604C 0C4E9913
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009070: CC20602F 0C425F10
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009078: CC206039 0C467311
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009080: CC206043 0C4A8712
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009088: CC20604D 0C4E9B13
	s_wait_loadcnt 0xf                                         // 000000009090: BFC0000F
	v_dual_mul_f32 v48, v41, v145 :: v_dual_mul_f32 v49, v51, v145// 000000009094: C8C72329 30312333
	v_dual_mul_f32 v58, v61, v145 :: v_dual_mul_f32 v59, v71, v145// 00000000909C: C8C7233D 3A3B2347
	s_wait_loadcnt 0xd                                         // 0000000090A4: BFC0000D
	v_dual_mul_f32 v68, v41, v153 :: v_dual_mul_f32 v69, v51, v153// 0000000090A8: C8C73329 44453333
	v_dual_mul_f32 v78, v61, v153 :: v_dual_mul_f32 v79, v71, v153// 0000000090B0: C8C7333D 4E4F3347
	v_dual_fmac_f32 v48, v40, v144 :: v_dual_fmac_f32 v49, v50, v144// 0000000090B8: C8012128 30312132
	v_dual_fmac_f32 v58, v60, v144 :: v_dual_fmac_f32 v59, v70, v144// 0000000090C0: C801213C 3A3B2146
	v_dual_fmac_f32 v68, v40, v152 :: v_dual_fmac_f32 v69, v50, v152// 0000000090C8: C8013128 44453132
	v_dual_fmac_f32 v78, v60, v152 :: v_dual_fmac_f32 v79, v70, v152// 0000000090D0: C801313C 4E4F3146
	v_dual_fmac_f32 v48, v42, v146 :: v_dual_fmac_f32 v49, v52, v146// 0000000090D8: C801252A 30312534
	v_dual_fmac_f32 v58, v62, v146 :: v_dual_fmac_f32 v59, v72, v146// 0000000090E0: C801253E 3A3B2548
	v_dual_fmac_f32 v68, v42, v154 :: v_dual_fmac_f32 v69, v52, v154// 0000000090E8: C801352A 44453534
	v_dual_fmac_f32 v78, v62, v154 :: v_dual_fmac_f32 v79, v72, v154// 0000000090F0: C801353E 4E4F3548
	v_dual_fmac_f32 v48, v43, v147 :: v_dual_fmac_f32 v49, v53, v147// 0000000090F8: C801272B 30312735
	v_dual_fmac_f32 v58, v63, v147 :: v_dual_fmac_f32 v59, v73, v147// 000000009100: C801273F 3A3B2749
	v_dual_fmac_f32 v68, v43, v155 :: v_dual_fmac_f32 v69, v53, v155// 000000009108: C801372B 44453735
	v_dual_fmac_f32 v78, v63, v155 :: v_dual_fmac_f32 v79, v73, v155// 000000009110: C801373F 4E4F3749
	v_dual_fmac_f32 v48, v44, v148 :: v_dual_fmac_f32 v49, v54, v148// 000000009118: C801292C 30312936
	v_dual_fmac_f32 v58, v64, v148 :: v_dual_fmac_f32 v59, v74, v148// 000000009120: C8012940 3A3B294A
	s_wait_loadcnt 0xc                                         // 000000009128: BFC0000C
	v_dual_fmac_f32 v68, v44, v156 :: v_dual_fmac_f32 v69, v54, v156// 00000000912C: C801392C 44453936
	v_dual_fmac_f32 v78, v64, v156 :: v_dual_fmac_f32 v79, v74, v156// 000000009134: C8013940 4E4F394A
	v_dual_fmac_f32 v48, v45, v149 :: v_dual_fmac_f32 v49, v55, v149// 00000000913C: C8012B2D 30312B37
	v_dual_fmac_f32 v58, v65, v149 :: v_dual_fmac_f32 v59, v75, v149// 000000009144: C8012B41 3A3B2B4B
	v_dual_fmac_f32 v68, v45, v157 :: v_dual_fmac_f32 v69, v55, v157// 00000000914C: C8013B2D 44453B37
	v_dual_fmac_f32 v78, v65, v157 :: v_dual_fmac_f32 v79, v75, v157// 000000009154: C8013B41 4E4F3B4B
	v_dual_fmac_f32 v48, v46, v150 :: v_dual_fmac_f32 v49, v56, v150// 00000000915C: C8012D2E 30312D38
	v_dual_fmac_f32 v58, v66, v150 :: v_dual_fmac_f32 v59, v76, v150// 000000009164: C8012D42 3A3B2D4C
	v_dual_fmac_f32 v68, v46, v158 :: v_dual_fmac_f32 v69, v56, v158// 00000000916C: C8013D2E 44453D38
	v_dual_fmac_f32 v78, v66, v158 :: v_dual_fmac_f32 v79, v76, v158// 000000009174: C8013D42 4E4F3D4C
	v_dual_fmac_f32 v48, v47, v151 :: v_dual_fmac_f32 v49, v57, v151// 00000000917C: C8012F2F 30312F39
	v_dual_fmac_f32 v58, v67, v151 :: v_dual_fmac_f32 v59, v77, v151// 000000009184: C8012F43 3A3B2F4D
	v_dual_fmac_f32 v68, v47, v159 :: v_dual_fmac_f32 v69, v57, v159// 00000000918C: C8013F2F 44453F39
	v_dual_fmac_f32 v78, v67, v159 :: v_dual_fmac_f32 v79, v77, v159// 000000009194: C8013F43 4E4F3F4D
	v_dual_add_f32 v200, v200, v48 :: v_dual_add_f32 v201, v201, v49// 00000000919C: C90861C8 C8C863C9
	v_dual_add_f32 v202, v202, v58 :: v_dual_add_f32 v203, v203, v59// 0000000091A4: C90875CA CACA77CB
	v_dual_add_f32 v216, v216, v68 :: v_dual_add_f32 v217, v217, v69// 0000000091AC: C90889D8 D8D88BD9
	v_dual_add_f32 v218, v218, v78 :: v_dual_add_f32 v219, v219, v79// 0000000091B4: C9089DDA DADA9FDB
	s_wait_loadcnt 0xb                                         // 0000000091BC: BFC0000B
	v_dual_mul_f32 v48, v41, v161 :: v_dual_mul_f32 v49, v51, v161// 0000000091C0: C8C74329 30314333
	v_dual_mul_f32 v58, v61, v161 :: v_dual_mul_f32 v59, v71, v161// 0000000091C8: C8C7433D 3A3B4347
	s_wait_loadcnt 0x9                                         // 0000000091D0: BFC00009
	v_dual_mul_f32 v68, v41, v169 :: v_dual_mul_f32 v69, v51, v169// 0000000091D4: C8C75329 44455333
	v_dual_mul_f32 v78, v61, v169 :: v_dual_mul_f32 v79, v71, v169// 0000000091DC: C8C7533D 4E4F5347
	v_dual_fmac_f32 v48, v40, v160 :: v_dual_fmac_f32 v49, v50, v160// 0000000091E4: C8014128 30314132
	v_dual_fmac_f32 v58, v60, v160 :: v_dual_fmac_f32 v59, v70, v160// 0000000091EC: C801413C 3A3B4146
	v_dual_fmac_f32 v68, v40, v168 :: v_dual_fmac_f32 v69, v50, v168// 0000000091F4: C8015128 44455132
	v_dual_fmac_f32 v78, v60, v168 :: v_dual_fmac_f32 v79, v70, v168// 0000000091FC: C801513C 4E4F5146
	v_dual_fmac_f32 v48, v42, v162 :: v_dual_fmac_f32 v49, v52, v162// 000000009204: C801452A 30314534
	v_dual_fmac_f32 v58, v62, v162 :: v_dual_fmac_f32 v59, v72, v162// 00000000920C: C801453E 3A3B4548
	v_dual_fmac_f32 v68, v42, v170 :: v_dual_fmac_f32 v69, v52, v170// 000000009214: C801552A 44455534
	v_dual_fmac_f32 v78, v62, v170 :: v_dual_fmac_f32 v79, v72, v170// 00000000921C: C801553E 4E4F5548
	v_dual_fmac_f32 v48, v43, v163 :: v_dual_fmac_f32 v49, v53, v163// 000000009224: C801472B 30314735
	v_dual_fmac_f32 v58, v63, v163 :: v_dual_fmac_f32 v59, v73, v163// 00000000922C: C801473F 3A3B4749
	v_dual_fmac_f32 v68, v43, v171 :: v_dual_fmac_f32 v69, v53, v171// 000000009234: C801572B 44455735
	v_dual_fmac_f32 v78, v63, v171 :: v_dual_fmac_f32 v79, v73, v171// 00000000923C: C801573F 4E4F5749
	v_dual_fmac_f32 v48, v44, v164 :: v_dual_fmac_f32 v49, v54, v164// 000000009244: C801492C 30314936
	v_dual_fmac_f32 v58, v64, v164 :: v_dual_fmac_f32 v59, v74, v164// 00000000924C: C8014940 3A3B494A
	s_wait_loadcnt 0x8                                         // 000000009254: BFC00008
	v_dual_fmac_f32 v68, v44, v172 :: v_dual_fmac_f32 v69, v54, v172// 000000009258: C801592C 44455936
	v_dual_fmac_f32 v78, v64, v172 :: v_dual_fmac_f32 v79, v74, v172// 000000009260: C8015940 4E4F594A
	v_dual_fmac_f32 v48, v45, v165 :: v_dual_fmac_f32 v49, v55, v165// 000000009268: C8014B2D 30314B37
	v_dual_fmac_f32 v58, v65, v165 :: v_dual_fmac_f32 v59, v75, v165// 000000009270: C8014B41 3A3B4B4B
	v_dual_fmac_f32 v68, v45, v173 :: v_dual_fmac_f32 v69, v55, v173// 000000009278: C8015B2D 44455B37
	v_dual_fmac_f32 v78, v65, v173 :: v_dual_fmac_f32 v79, v75, v173// 000000009280: C8015B41 4E4F5B4B
	v_dual_fmac_f32 v48, v46, v166 :: v_dual_fmac_f32 v49, v56, v166// 000000009288: C8014D2E 30314D38
	v_dual_fmac_f32 v58, v66, v166 :: v_dual_fmac_f32 v59, v76, v166// 000000009290: C8014D42 3A3B4D4C
	v_dual_fmac_f32 v68, v46, v174 :: v_dual_fmac_f32 v69, v56, v174// 000000009298: C8015D2E 44455D38
	v_dual_fmac_f32 v78, v66, v174 :: v_dual_fmac_f32 v79, v76, v174// 0000000092A0: C8015D42 4E4F5D4C
	v_dual_fmac_f32 v48, v47, v167 :: v_dual_fmac_f32 v49, v57, v167// 0000000092A8: C8014F2F 30314F39
	v_dual_fmac_f32 v58, v67, v167 :: v_dual_fmac_f32 v59, v77, v167// 0000000092B0: C8014F43 3A3B4F4D
	v_dual_fmac_f32 v68, v47, v175 :: v_dual_fmac_f32 v69, v57, v175// 0000000092B8: C8015F2F 44455F39
	v_dual_fmac_f32 v78, v67, v175 :: v_dual_fmac_f32 v79, v77, v175// 0000000092C0: C8015F43 4E4F5F4D
	v_dual_add_f32 v232, v232, v48 :: v_dual_add_f32 v233, v233, v49// 0000000092C8: C90861E8 E8E863E9
	v_dual_add_f32 v234, v234, v58 :: v_dual_add_f32 v235, v235, v59// 0000000092D0: C90875EA EAEA77EB
	v_dual_add_f32 v248, v248, v68 :: v_dual_add_f32 v249, v249, v69// 0000000092D8: C90889F8 F8F88BF9
	v_dual_add_f32 v250, v250, v78 :: v_dual_add_f32 v251, v251, v79// 0000000092E0: C9089DFA FAFA9FFB
	v_bfe_u32 v40, v36, 0, 2                                   // 0000000092E8: D6100028 02090124
	v_bfe_u32 v50, v37, 0, 2                                   // 0000000092F0: D6100032 02090125
	v_bfe_u32 v60, v38, 0, 2                                   // 0000000092F8: D610003C 02090126
	v_bfe_u32 v70, v39, 0, 2                                   // 000000009300: D6100046 02090127
	v_bfe_u32 v41, v36, 2, 2                                   // 000000009308: D6100029 02090524
	v_bfe_u32 v51, v37, 2, 2                                   // 000000009310: D6100033 02090525
	v_bfe_u32 v61, v38, 2, 2                                   // 000000009318: D610003D 02090526
	v_bfe_u32 v71, v39, 2, 2                                   // 000000009320: D6100047 02090527
	v_bfe_u32 v42, v36, 4, 2                                   // 000000009328: D610002A 02090924
	v_bfe_u32 v52, v37, 4, 2                                   // 000000009330: D6100034 02090925
	v_bfe_u32 v62, v38, 4, 2                                   // 000000009338: D610003E 02090926
	v_bfe_u32 v72, v39, 4, 2                                   // 000000009340: D6100048 02090927
	v_bfe_u32 v43, v36, 6, 2                                   // 000000009348: D610002B 02090D24
	v_bfe_u32 v53, v37, 6, 2                                   // 000000009350: D6100035 02090D25
	v_bfe_u32 v63, v38, 6, 2                                   // 000000009358: D610003F 02090D26
	v_bfe_u32 v73, v39, 6, 2                                   // 000000009360: D6100049 02090D27
	v_bfe_u32 v44, v36, 8, 2                                   // 000000009368: D610002C 02091124
	v_bfe_u32 v54, v37, 8, 2                                   // 000000009370: D6100036 02091125
	v_bfe_u32 v64, v38, 8, 2                                   // 000000009378: D6100040 02091126
	v_bfe_u32 v74, v39, 8, 2                                   // 000000009380: D610004A 02091127
	v_bfe_u32 v45, v36, 10, 2                                  // 000000009388: D610002D 02091524
	v_bfe_u32 v55, v37, 10, 2                                  // 000000009390: D6100037 02091525
	v_bfe_u32 v65, v38, 10, 2                                  // 000000009398: D6100041 02091526
	v_bfe_u32 v75, v39, 10, 2                                  // 0000000093A0: D610004B 02091527
	v_bfe_u32 v46, v36, 12, 2                                  // 0000000093A8: D610002E 02091924
	v_bfe_u32 v56, v37, 12, 2                                  // 0000000093B0: D6100038 02091925
	v_bfe_u32 v66, v38, 12, 2                                  // 0000000093B8: D6100042 02091926
	v_bfe_u32 v76, v39, 12, 2                                  // 0000000093C0: D610004C 02091927
	v_bfe_u32 v47, v36, 14, 2                                  // 0000000093C8: D610002F 02091D24
	v_bfe_u32 v57, v37, 14, 2                                  // 0000000093D0: D6100039 02091D25
	v_bfe_u32 v67, v38, 14, 2                                  // 0000000093D8: D6100043 02091D26
	v_bfe_u32 v77, v39, 14, 2                                  // 0000000093E0: D610004D 02091D27
	v_cvt_f32_ubyte0_e32 v40, v40                              // 0000000093E8: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 0000000093EC: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 0000000093F0: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 0000000093F4: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 0000000093F8: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 0000000093FC: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000009400: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000009404: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000009408: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 00000000940C: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000009410: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000009414: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000009418: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 00000000941C: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000009420: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000009424: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000009428: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 00000000942C: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000009430: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000009434: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000009438: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 00000000943C: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000009440: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000009444: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000009448: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 00000000944C: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000009450: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000009454: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000009458: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 00000000945C: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000009460: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000009464: 7E9A234D
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009468: CC206028 0C525114
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009470: CC206032 0C566515
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009478: CC20603C 0C5A7916
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009480: CC206046 0C5E8D17
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009488: CC206029 0C525314
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009490: CC206033 0C566715
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009498: CC20603D 0C5A7B16
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094A0: CC206047 0C5E8F17
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094A8: CC20602A 0C525514
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094B0: CC206034 0C566915
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094B8: CC20603E 0C5A7D16
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094C0: CC206048 0C5E9117
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094C8: CC20602B 0C525714
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094D0: CC206035 0C566B15
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094D8: CC20603F 0C5A7F16
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094E0: CC206049 0C5E9317
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094E8: CC20602C 0C525914
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094F0: CC206036 0C566D15
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 0000000094F8: CC206040 0C5A8116
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009500: CC20604A 0C5E9517
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009508: CC20602D 0C525B14
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009510: CC206037 0C566F15
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009518: CC206041 0C5A8316
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009520: CC20604B 0C5E9717
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009528: CC20602E 0C525D14
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009530: CC206038 0C567115
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009538: CC206042 0C5A8516
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009540: CC20604C 0C5E9917
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009548: CC20602F 0C525F14
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009550: CC206039 0C567315
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009558: CC206043 0C5A8716
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009560: CC20604D 0C5E9B17
	s_wait_loadcnt 0x7                                         // 000000009568: BFC00007
	v_dual_mul_f32 v48, v41, v177 :: v_dual_mul_f32 v49, v51, v177// 00000000956C: C8C76329 30316333
	v_dual_mul_f32 v58, v61, v177 :: v_dual_mul_f32 v59, v71, v177// 000000009574: C8C7633D 3A3B6347
	s_wait_loadcnt 0x5                                         // 00000000957C: BFC00005
	v_dual_mul_f32 v68, v41, v185 :: v_dual_mul_f32 v69, v51, v185// 000000009580: C8C77329 44457333
	v_dual_mul_f32 v78, v61, v185 :: v_dual_mul_f32 v79, v71, v185// 000000009588: C8C7733D 4E4F7347
	v_dual_fmac_f32 v48, v40, v176 :: v_dual_fmac_f32 v49, v50, v176// 000000009590: C8016128 30316132
	v_dual_fmac_f32 v58, v60, v176 :: v_dual_fmac_f32 v59, v70, v176// 000000009598: C801613C 3A3B6146
	v_dual_fmac_f32 v68, v40, v184 :: v_dual_fmac_f32 v69, v50, v184// 0000000095A0: C8017128 44457132
	v_dual_fmac_f32 v78, v60, v184 :: v_dual_fmac_f32 v79, v70, v184// 0000000095A8: C801713C 4E4F7146
	v_dual_fmac_f32 v48, v42, v178 :: v_dual_fmac_f32 v49, v52, v178// 0000000095B0: C801652A 30316534
	v_dual_fmac_f32 v58, v62, v178 :: v_dual_fmac_f32 v59, v72, v178// 0000000095B8: C801653E 3A3B6548
	v_dual_fmac_f32 v68, v42, v186 :: v_dual_fmac_f32 v69, v52, v186// 0000000095C0: C801752A 44457534
	v_dual_fmac_f32 v78, v62, v186 :: v_dual_fmac_f32 v79, v72, v186// 0000000095C8: C801753E 4E4F7548
	v_dual_fmac_f32 v48, v43, v179 :: v_dual_fmac_f32 v49, v53, v179// 0000000095D0: C801672B 30316735
	v_dual_fmac_f32 v58, v63, v179 :: v_dual_fmac_f32 v59, v73, v179// 0000000095D8: C801673F 3A3B6749
	v_dual_fmac_f32 v68, v43, v187 :: v_dual_fmac_f32 v69, v53, v187// 0000000095E0: C801772B 44457735
	v_dual_fmac_f32 v78, v63, v187 :: v_dual_fmac_f32 v79, v73, v187// 0000000095E8: C801773F 4E4F7749
	v_dual_fmac_f32 v48, v44, v180 :: v_dual_fmac_f32 v49, v54, v180// 0000000095F0: C801692C 30316936
	v_dual_fmac_f32 v58, v64, v180 :: v_dual_fmac_f32 v59, v74, v180// 0000000095F8: C8016940 3A3B694A
	s_wait_loadcnt 0x4                                         // 000000009600: BFC00004
	v_dual_fmac_f32 v68, v44, v188 :: v_dual_fmac_f32 v69, v54, v188// 000000009604: C801792C 44457936
	v_dual_fmac_f32 v78, v64, v188 :: v_dual_fmac_f32 v79, v74, v188// 00000000960C: C8017940 4E4F794A
	v_dual_fmac_f32 v48, v45, v181 :: v_dual_fmac_f32 v49, v55, v181// 000000009614: C8016B2D 30316B37
	v_dual_fmac_f32 v58, v65, v181 :: v_dual_fmac_f32 v59, v75, v181// 00000000961C: C8016B41 3A3B6B4B
	v_dual_fmac_f32 v68, v45, v189 :: v_dual_fmac_f32 v69, v55, v189// 000000009624: C8017B2D 44457B37
	v_dual_fmac_f32 v78, v65, v189 :: v_dual_fmac_f32 v79, v75, v189// 00000000962C: C8017B41 4E4F7B4B
	v_dual_fmac_f32 v48, v46, v182 :: v_dual_fmac_f32 v49, v56, v182// 000000009634: C8016D2E 30316D38
	v_dual_fmac_f32 v58, v66, v182 :: v_dual_fmac_f32 v59, v76, v182// 00000000963C: C8016D42 3A3B6D4C
	v_dual_fmac_f32 v68, v46, v190 :: v_dual_fmac_f32 v69, v56, v190// 000000009644: C8017D2E 44457D38
	v_dual_fmac_f32 v78, v66, v190 :: v_dual_fmac_f32 v79, v76, v190// 00000000964C: C8017D42 4E4F7D4C
	v_dual_fmac_f32 v48, v47, v183 :: v_dual_fmac_f32 v49, v57, v183// 000000009654: C8016F2F 30316F39
	v_dual_fmac_f32 v58, v67, v183 :: v_dual_fmac_f32 v59, v77, v183// 00000000965C: C8016F43 3A3B6F4D
	v_dual_fmac_f32 v68, v47, v191 :: v_dual_fmac_f32 v69, v57, v191// 000000009664: C8017F2F 44457F39
	v_dual_fmac_f32 v78, v67, v191 :: v_dual_fmac_f32 v79, v77, v191// 00000000966C: C8017F43 4E4F7F4D
	v_dual_add_f32 v204, v204, v48 :: v_dual_add_f32 v205, v205, v49// 000000009674: C90861CC CCCC63CD
	v_dual_add_f32 v206, v206, v58 :: v_dual_add_f32 v207, v207, v59// 00000000967C: C90875CE CECE77CF
	v_dual_add_f32 v220, v220, v68 :: v_dual_add_f32 v221, v221, v69// 000000009684: C90889DC DCDC8BDD
	v_dual_add_f32 v222, v222, v78 :: v_dual_add_f32 v223, v223, v79// 00000000968C: C9089DDE DEDE9FDF
	s_wait_loadcnt 0x3                                         // 000000009694: BFC00003
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 000000009698: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 0000000096A0: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x1                                         // 0000000096A8: BFC00001
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 0000000096AC: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 0000000096B4: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 0000000096BC: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 0000000096C4: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 0000000096CC: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 0000000096D4: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 0000000096DC: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 0000000096E4: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 0000000096EC: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 0000000096F4: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 0000000096FC: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000009704: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 00000000970C: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 000000009714: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 00000000971C: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000009724: C800A940 3A3AA94A
	s_wait_loadcnt 0x0                                         // 00000000972C: BFC00000
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 000000009730: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 000000009738: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000009740: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000009748: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 000000009750: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 000000009758: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 000000009760: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000009768: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 000000009770: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 000000009778: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 000000009780: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 000000009788: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 000000009790: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 000000009798: C800BF43 4E4EBF4D
	v_dual_add_f32 v236, v236, v48 :: v_dual_add_f32 v237, v237, v49// 0000000097A0: C90861EC ECEC63ED
	v_dual_add_f32 v238, v238, v58 :: v_dual_add_f32 v239, v239, v59// 0000000097A8: C90875EE EEEE77EF
	v_dual_add_f32 v252, v252, v68 :: v_dual_add_f32 v253, v253, v69// 0000000097B0: C90889FC FCFC8BFD
	v_dual_add_f32 v254, v254, v78 :: v_dual_add_f32 v255, v255, v79// 0000000097B8: C9089DFE FEFE9FFF
	s_add_co_i32 s16, s16, 0x120                               // 0000000097C0: 8110FF10 00000120
	s_add_co_i32 s17, s17, 0x120                               // 0000000097C8: 8111FF11 00000120
	s_add_co_i32 s18, s18, 0x120                               // 0000000097D0: 8112FF12 00000120
	s_add_co_i32 s19, s19, 0x120                               // 0000000097D8: 8113FF13 00000120
	s_add_co_i32 s28, s28, 0x1000                              // 0000000097E0: 811CFF1C 00001000
	s_add_co_i32 s29, s29, 0x1000                              // 0000000097E8: 811DFF1D 00001000
	s_add_co_i32 s30, s30, 0x1000                              // 0000000097F0: 811EFF1E 00001000
	s_add_co_i32 s31, s31, 0x1000                              // 0000000097F8: 811FFF1F 00001000
	s_add_co_i32 s36, s36, -1                                  // 000000009800: 8124C124
	s_cmp_lg_u32 s36, 0                                        // 000000009804: BF078024
	s_cbranch_scc1 64077                                       // 000000009808: BFA2FA4D <gemv_mq2g256v2_xbatch_pm+0x6a40>
	s_cmp_gt_u32 s37, 0                                        // 00000000980C: BF088025
	s_cbranch_scc0 1090                                        // 000000009810: BFA10442 <gemv_mq2g256v2_xbatch_pm+0x921c>
	buffer_load_b32 v8, v1, s[20:23], s16 offen                // 000000009814: C4050010 40802808 00000001
	buffer_load_b32 v9, v1, s[20:23], s17 offen                // 000000009820: C4050011 40802809 00000001
	buffer_load_b32 v10, v1, s[20:23], s18 offen               // 00000000982C: C4050012 4080280A 00000001
	buffer_load_b32 v11, v1, s[20:23], s19 offen               // 000000009838: C4050013 4080280B 00000001
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8      // 000000009844: C4050010 40802818 00000802
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8      // 000000009850: C4050011 40802819 00000802
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8      // 00000000985C: C4050012 4080281A 00000802
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8      // 000000009868: C4050013 4080281B 00000802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen         // 000000009874: C405C01C 40803050 00000003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16// 000000009880: C405C01C 40803054 00001003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen         // 00000000988C: C405C01D 40803058 00000003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16// 000000009898: C405C01D 4080305C 00001003
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen         // 0000000098A4: C405C01E 40803060 00000003
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16// 0000000098B0: C405C01E 40803064 00001003
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen       // 0000000098BC: C405C01F 40803068 00000003
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16// 0000000098C8: C405C01F 4080306C 00001003
	s_wait_loadcnt 0xb                                         // 0000000098D4: BFC0000B
	v_bfe_u32 v40, v24, 0, 2                                   // 0000000098D8: D6100028 02090118
	s_wait_loadcnt 0xa                                         // 0000000098E0: BFC0000A
	v_bfe_u32 v50, v25, 0, 2                                   // 0000000098E4: D6100032 02090119
	s_wait_loadcnt 0x9                                         // 0000000098EC: BFC00009
	v_bfe_u32 v60, v26, 0, 2                                   // 0000000098F0: D610003C 0209011A
	s_wait_loadcnt 0x8                                         // 0000000098F8: BFC00008
	v_bfe_u32 v70, v27, 0, 2                                   // 0000000098FC: D6100046 0209011B
	v_bfe_u32 v41, v24, 2, 2                                   // 000000009904: D6100029 02090518
	v_bfe_u32 v51, v25, 2, 2                                   // 00000000990C: D6100033 02090519
	v_bfe_u32 v61, v26, 2, 2                                   // 000000009914: D610003D 0209051A
	v_bfe_u32 v71, v27, 2, 2                                   // 00000000991C: D6100047 0209051B
	v_bfe_u32 v42, v24, 4, 2                                   // 000000009924: D610002A 02090918
	v_bfe_u32 v52, v25, 4, 2                                   // 00000000992C: D6100034 02090919
	v_bfe_u32 v62, v26, 4, 2                                   // 000000009934: D610003E 0209091A
	v_bfe_u32 v72, v27, 4, 2                                   // 00000000993C: D6100048 0209091B
	v_bfe_u32 v43, v24, 6, 2                                   // 000000009944: D610002B 02090D18
	v_bfe_u32 v53, v25, 6, 2                                   // 00000000994C: D6100035 02090D19
	v_bfe_u32 v63, v26, 6, 2                                   // 000000009954: D610003F 02090D1A
	v_bfe_u32 v73, v27, 6, 2                                   // 00000000995C: D6100049 02090D1B
	v_bfe_u32 v44, v24, 8, 2                                   // 000000009964: D610002C 02091118
	v_bfe_u32 v54, v25, 8, 2                                   // 00000000996C: D6100036 02091119
	v_bfe_u32 v64, v26, 8, 2                                   // 000000009974: D6100040 0209111A
	v_bfe_u32 v74, v27, 8, 2                                   // 00000000997C: D610004A 0209111B
	v_bfe_u32 v45, v24, 10, 2                                  // 000000009984: D610002D 02091518
	v_bfe_u32 v55, v25, 10, 2                                  // 00000000998C: D6100037 02091519
	v_bfe_u32 v65, v26, 10, 2                                  // 000000009994: D6100041 0209151A
	v_bfe_u32 v75, v27, 10, 2                                  // 00000000999C: D610004B 0209151B
	v_bfe_u32 v46, v24, 12, 2                                  // 0000000099A4: D610002E 02091918
	v_bfe_u32 v56, v25, 12, 2                                  // 0000000099AC: D6100038 02091919
	v_bfe_u32 v66, v26, 12, 2                                  // 0000000099B4: D6100042 0209191A
	v_bfe_u32 v76, v27, 12, 2                                  // 0000000099BC: D610004C 0209191B
	v_bfe_u32 v47, v24, 14, 2                                  // 0000000099C4: D610002F 02091D18
	v_bfe_u32 v57, v25, 14, 2                                  // 0000000099CC: D6100039 02091D19
	v_bfe_u32 v67, v26, 14, 2                                  // 0000000099D4: D6100043 02091D1A
	v_bfe_u32 v77, v27, 14, 2                                  // 0000000099DC: D610004D 02091D1B
	v_cvt_f32_ubyte0_e32 v40, v40                              // 0000000099E4: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 0000000099E8: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 0000000099EC: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 0000000099F0: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 0000000099F4: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 0000000099F8: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 0000000099FC: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000009A00: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000009A04: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000009A08: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000009A0C: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000009A10: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000009A14: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000009A18: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000009A1C: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000009A20: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000009A24: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000009A28: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000009A2C: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000009A30: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000009A34: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000009A38: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000009A3C: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000009A40: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000009A44: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000009A48: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000009A4C: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 000000009A50: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 000000009A54: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 000000009A58: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 000000009A5C: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 000000009A60: 7E9A234D
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009A64: CC206028 0C225108
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009A6C: CC206032 0C266509
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009A74: CC20603C 0C2A790A
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009A7C: CC206046 0C2E8D0B
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009A84: CC206029 0C225308
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009A8C: CC206033 0C266709
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009A94: CC20603D 0C2A7B0A
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009A9C: CC206047 0C2E8F0B
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009AA4: CC20602A 0C225508
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009AAC: CC206034 0C266909
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009AB4: CC20603E 0C2A7D0A
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009ABC: CC206048 0C2E910B
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009AC4: CC20602B 0C225708
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009ACC: CC206035 0C266B09
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009AD4: CC20603F 0C2A7F0A
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009ADC: CC206049 0C2E930B
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009AE4: CC20602C 0C225908
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009AEC: CC206036 0C266D09
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009AF4: CC206040 0C2A810A
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009AFC: CC20604A 0C2E950B
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B04: CC20602D 0C225B08
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B0C: CC206037 0C266F09
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B14: CC206041 0C2A830A
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B1C: CC20604B 0C2E970B
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B24: CC20602E 0C225D08
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B2C: CC206038 0C267109
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B34: CC206042 0C2A850A
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B3C: CC20604C 0C2E990B
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B44: CC20602F 0C225F08
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B4C: CC206039 0C267309
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B54: CC206043 0C2A870A
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 000000009B5C: CC20604D 0C2E9B0B
	s_wait_loadcnt 0x7                                         // 000000009B64: BFC00007
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 000000009B68: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 000000009B70: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x5                                         // 000000009B78: BFC00005
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 000000009B7C: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 000000009B84: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 000000009B8C: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 000000009B94: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 000000009B9C: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 000000009BA4: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 000000009BAC: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 000000009BB4: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 000000009BBC: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 000000009BC4: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 000000009BCC: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 000000009BD4: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 000000009BDC: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 000000009BE4: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 000000009BEC: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 000000009BF4: C800A940 3A3AA94A
	s_wait_loadcnt 0x4                                         // 000000009BFC: BFC00004
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 000000009C00: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 000000009C08: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 000000009C10: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 000000009C18: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 000000009C20: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 000000009C28: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 000000009C30: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 000000009C38: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 000000009C40: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 000000009C48: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 000000009C50: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 000000009C58: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 000000009C60: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 000000009C68: C800BF43 4E4EBF4D
	v_dual_add_f32 v192, v192, v48 :: v_dual_add_f32 v193, v193, v49// 000000009C70: C90861C0 C0C063C1
	v_dual_add_f32 v194, v194, v58 :: v_dual_add_f32 v195, v195, v59// 000000009C78: C90875C2 C2C277C3
	v_dual_add_f32 v208, v208, v68 :: v_dual_add_f32 v209, v209, v69// 000000009C80: C90889D0 D0D08BD1
	v_dual_add_f32 v210, v210, v78 :: v_dual_add_f32 v211, v211, v79// 000000009C88: C9089DD2 D2D29FD3
	s_wait_loadcnt 0x3                                         // 000000009C90: BFC00003
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97// 000000009C94: C8C6C329 3030C333
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97// 000000009C9C: C8C6C33D 3A3AC347
	s_wait_loadcnt 0x1                                         // 000000009CA4: BFC00001
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105// 000000009CA8: C8C6D329 4444D333
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105// 000000009CB0: C8C6D33D 4E4ED347
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96// 000000009CB8: C800C128 3030C132
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96// 000000009CC0: C800C13C 3A3AC146
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104// 000000009CC8: C800D128 4444D132
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104// 000000009CD0: C800D13C 4E4ED146
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98// 000000009CD8: C800C52A 3030C534
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98// 000000009CE0: C800C53E 3A3AC548
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106// 000000009CE8: C800D52A 4444D534
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106// 000000009CF0: C800D53E 4E4ED548
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99// 000000009CF8: C800C72B 3030C735
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99// 000000009D00: C800C73F 3A3AC749
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107// 000000009D08: C800D72B 4444D735
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107// 000000009D10: C800D73F 4E4ED749
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100// 000000009D18: C800C92C 3030C936
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100// 000000009D20: C800C940 3A3AC94A
	s_wait_loadcnt 0x0                                         // 000000009D28: BFC00000
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108// 000000009D2C: C800D92C 4444D936
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108// 000000009D34: C800D940 4E4ED94A
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101// 000000009D3C: C800CB2D 3030CB37
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101// 000000009D44: C800CB41 3A3ACB4B
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109// 000000009D4C: C800DB2D 4444DB37
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109// 000000009D54: C800DB41 4E4EDB4B
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102// 000000009D5C: C800CD2E 3030CD38
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102// 000000009D64: C800CD42 3A3ACD4C
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110// 000000009D6C: C800DD2E 4444DD38
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110// 000000009D74: C800DD42 4E4EDD4C
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103// 000000009D7C: C800CF2F 3030CF39
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103// 000000009D84: C800CF43 3A3ACF4D
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111// 000000009D8C: C800DF2F 4444DF39
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111// 000000009D94: C800DF43 4E4EDF4D
	v_dual_add_f32 v224, v224, v48 :: v_dual_add_f32 v225, v225, v49// 000000009D9C: C90861E0 E0E063E1
	v_dual_add_f32 v226, v226, v58 :: v_dual_add_f32 v227, v227, v59// 000000009DA4: C90875E2 E2E277E3
	v_dual_add_f32 v240, v240, v68 :: v_dual_add_f32 v241, v241, v69// 000000009DAC: C90889F0 F0F08BF1
	v_dual_add_f32 v242, v242, v78 :: v_dual_add_f32 v243, v243, v79// 000000009DB4: C9089DF2 F2F29FF3
	s_cmp_gt_u32 s37, 1                                        // 000000009DBC: BF088125
	s_cbranch_scc0 726                                         // 000000009DC0: BFA102D6 <gemv_mq2g256v2_xbatch_pm+0x921c>
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:72     // 000000009DC4: C4050010 4080280C 00004801
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:72     // 000000009DD0: C4050011 4080280D 00004801
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:72     // 000000009DDC: C4050012 4080280E 00004801
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:72     // 000000009DE8: C4050013 4080280F 00004801
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:80     // 000000009DF4: C4050010 4080281C 00005002
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:80     // 000000009E00: C4050011 4080281D 00005002
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:80     // 000000009E0C: C4050012 4080281E 00005002
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:80     // 000000009E18: C4050013 4080281F 00005002
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024// 000000009E24: C405C01C 40803050 00040003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040// 000000009E30: C405C01C 40803054 00041003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:1024// 000000009E3C: C405C01D 40803058 00040003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:1040// 000000009E48: C405C01D 4080305C 00041003
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:1024// 000000009E54: C405C01E 40803060 00040003
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:1040// 000000009E60: C405C01E 40803064 00041003
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:1024// 000000009E6C: C405C01F 40803068 00040003
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:1040// 000000009E78: C405C01F 4080306C 00041003
	s_wait_loadcnt 0xb                                         // 000000009E84: BFC0000B
	v_bfe_u32 v40, v28, 0, 2                                   // 000000009E88: D6100028 0209011C
	s_wait_loadcnt 0xa                                         // 000000009E90: BFC0000A
	v_bfe_u32 v50, v29, 0, 2                                   // 000000009E94: D6100032 0209011D
	s_wait_loadcnt 0x9                                         // 000000009E9C: BFC00009
	v_bfe_u32 v60, v30, 0, 2                                   // 000000009EA0: D610003C 0209011E
	s_wait_loadcnt 0x8                                         // 000000009EA8: BFC00008
	v_bfe_u32 v70, v31, 0, 2                                   // 000000009EAC: D6100046 0209011F
	v_bfe_u32 v41, v28, 2, 2                                   // 000000009EB4: D6100029 0209051C
	v_bfe_u32 v51, v29, 2, 2                                   // 000000009EBC: D6100033 0209051D
	v_bfe_u32 v61, v30, 2, 2                                   // 000000009EC4: D610003D 0209051E
	v_bfe_u32 v71, v31, 2, 2                                   // 000000009ECC: D6100047 0209051F
	v_bfe_u32 v42, v28, 4, 2                                   // 000000009ED4: D610002A 0209091C
	v_bfe_u32 v52, v29, 4, 2                                   // 000000009EDC: D6100034 0209091D
	v_bfe_u32 v62, v30, 4, 2                                   // 000000009EE4: D610003E 0209091E
	v_bfe_u32 v72, v31, 4, 2                                   // 000000009EEC: D6100048 0209091F
	v_bfe_u32 v43, v28, 6, 2                                   // 000000009EF4: D610002B 02090D1C
	v_bfe_u32 v53, v29, 6, 2                                   // 000000009EFC: D6100035 02090D1D
	v_bfe_u32 v63, v30, 6, 2                                   // 000000009F04: D610003F 02090D1E
	v_bfe_u32 v73, v31, 6, 2                                   // 000000009F0C: D6100049 02090D1F
	v_bfe_u32 v44, v28, 8, 2                                   // 000000009F14: D610002C 0209111C
	v_bfe_u32 v54, v29, 8, 2                                   // 000000009F1C: D6100036 0209111D
	v_bfe_u32 v64, v30, 8, 2                                   // 000000009F24: D6100040 0209111E
	v_bfe_u32 v74, v31, 8, 2                                   // 000000009F2C: D610004A 0209111F
	v_bfe_u32 v45, v28, 10, 2                                  // 000000009F34: D610002D 0209151C
	v_bfe_u32 v55, v29, 10, 2                                  // 000000009F3C: D6100037 0209151D
	v_bfe_u32 v65, v30, 10, 2                                  // 000000009F44: D6100041 0209151E
	v_bfe_u32 v75, v31, 10, 2                                  // 000000009F4C: D610004B 0209151F
	v_bfe_u32 v46, v28, 12, 2                                  // 000000009F54: D610002E 0209191C
	v_bfe_u32 v56, v29, 12, 2                                  // 000000009F5C: D6100038 0209191D
	v_bfe_u32 v66, v30, 12, 2                                  // 000000009F64: D6100042 0209191E
	v_bfe_u32 v76, v31, 12, 2                                  // 000000009F6C: D610004C 0209191F
	v_bfe_u32 v47, v28, 14, 2                                  // 000000009F74: D610002F 02091D1C
	v_bfe_u32 v57, v29, 14, 2                                  // 000000009F7C: D6100039 02091D1D
	v_bfe_u32 v67, v30, 14, 2                                  // 000000009F84: D6100043 02091D1E
	v_bfe_u32 v77, v31, 14, 2                                  // 000000009F8C: D610004D 02091D1F
	v_cvt_f32_ubyte0_e32 v40, v40                              // 000000009F94: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 000000009F98: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 000000009F9C: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 000000009FA0: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 000000009FA4: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 000000009FA8: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 000000009FAC: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 000000009FB0: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 000000009FB4: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 000000009FB8: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 000000009FBC: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 000000009FC0: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 000000009FC4: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 000000009FC8: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 000000009FCC: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 000000009FD0: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 000000009FD4: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 000000009FD8: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 000000009FDC: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 000000009FE0: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 000000009FE4: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 000000009FE8: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 000000009FEC: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 000000009FF0: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 000000009FF4: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 000000009FF8: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 000000009FFC: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 00000000A000: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 00000000A004: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 00000000A008: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 00000000A00C: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 00000000A010: 7E9A234D
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A014: CC206028 0C32510C
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A01C: CC206032 0C36650D
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A024: CC20603C 0C3A790E
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A02C: CC206046 0C3E8D0F
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A034: CC206029 0C32530C
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A03C: CC206033 0C36670D
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A044: CC20603D 0C3A7B0E
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A04C: CC206047 0C3E8F0F
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A054: CC20602A 0C32550C
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A05C: CC206034 0C36690D
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A064: CC20603E 0C3A7D0E
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A06C: CC206048 0C3E910F
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A074: CC20602B 0C32570C
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A07C: CC206035 0C366B0D
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A084: CC20603F 0C3A7F0E
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A08C: CC206049 0C3E930F
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A094: CC20602C 0C32590C
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A09C: CC206036 0C366D0D
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0A4: CC206040 0C3A810E
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0AC: CC20604A 0C3E950F
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0B4: CC20602D 0C325B0C
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0BC: CC206037 0C366F0D
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0C4: CC206041 0C3A830E
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0CC: CC20604B 0C3E970F
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0D4: CC20602E 0C325D0C
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0DC: CC206038 0C36710D
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0E4: CC206042 0C3A850E
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0EC: CC20604C 0C3E990F
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0F4: CC20602F 0C325F0C
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A0FC: CC206039 0C36730D
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A104: CC206043 0C3A870E
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A10C: CC20604D 0C3E9B0F
	s_wait_loadcnt 0x7                                         // 00000000A114: BFC00007
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 00000000A118: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 00000000A120: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x5                                         // 00000000A128: BFC00005
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 00000000A12C: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 00000000A134: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 00000000A13C: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 00000000A144: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 00000000A14C: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 00000000A154: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 00000000A15C: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 00000000A164: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 00000000A16C: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 00000000A174: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 00000000A17C: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 00000000A184: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 00000000A18C: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 00000000A194: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 00000000A19C: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 00000000A1A4: C800A940 3A3AA94A
	s_wait_loadcnt 0x4                                         // 00000000A1AC: BFC00004
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 00000000A1B0: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 00000000A1B8: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 00000000A1C0: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 00000000A1C8: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 00000000A1D0: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 00000000A1D8: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 00000000A1E0: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 00000000A1E8: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 00000000A1F0: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 00000000A1F8: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 00000000A200: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 00000000A208: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 00000000A210: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 00000000A218: C800BF43 4E4EBF4D
	v_dual_add_f32 v196, v196, v48 :: v_dual_add_f32 v197, v197, v49// 00000000A220: C90861C4 C4C463C5
	v_dual_add_f32 v198, v198, v58 :: v_dual_add_f32 v199, v199, v59// 00000000A228: C90875C6 C6C677C7
	v_dual_add_f32 v212, v212, v68 :: v_dual_add_f32 v213, v213, v69// 00000000A230: C90889D4 D4D48BD5
	v_dual_add_f32 v214, v214, v78 :: v_dual_add_f32 v215, v215, v79// 00000000A238: C9089DD6 D6D69FD7
	s_wait_loadcnt 0x3                                         // 00000000A240: BFC00003
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97// 00000000A244: C8C6C329 3030C333
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97// 00000000A24C: C8C6C33D 3A3AC347
	s_wait_loadcnt 0x1                                         // 00000000A254: BFC00001
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105// 00000000A258: C8C6D329 4444D333
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105// 00000000A260: C8C6D33D 4E4ED347
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96// 00000000A268: C800C128 3030C132
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96// 00000000A270: C800C13C 3A3AC146
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104// 00000000A278: C800D128 4444D132
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104// 00000000A280: C800D13C 4E4ED146
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98// 00000000A288: C800C52A 3030C534
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98// 00000000A290: C800C53E 3A3AC548
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106// 00000000A298: C800D52A 4444D534
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106// 00000000A2A0: C800D53E 4E4ED548
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99// 00000000A2A8: C800C72B 3030C735
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99// 00000000A2B0: C800C73F 3A3AC749
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107// 00000000A2B8: C800D72B 4444D735
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107// 00000000A2C0: C800D73F 4E4ED749
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100// 00000000A2C8: C800C92C 3030C936
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100// 00000000A2D0: C800C940 3A3AC94A
	s_wait_loadcnt 0x0                                         // 00000000A2D8: BFC00000
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108// 00000000A2DC: C800D92C 4444D936
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108// 00000000A2E4: C800D940 4E4ED94A
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101// 00000000A2EC: C800CB2D 3030CB37
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101// 00000000A2F4: C800CB41 3A3ACB4B
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109// 00000000A2FC: C800DB2D 4444DB37
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109// 00000000A304: C800DB41 4E4EDB4B
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102// 00000000A30C: C800CD2E 3030CD38
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102// 00000000A314: C800CD42 3A3ACD4C
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110// 00000000A31C: C800DD2E 4444DD38
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110// 00000000A324: C800DD42 4E4EDD4C
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103// 00000000A32C: C800CF2F 3030CF39
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103// 00000000A334: C800CF43 3A3ACF4D
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111// 00000000A33C: C800DF2F 4444DF39
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111// 00000000A344: C800DF43 4E4EDF4D
	v_dual_add_f32 v228, v228, v48 :: v_dual_add_f32 v229, v229, v49// 00000000A34C: C90861E4 E4E463E5
	v_dual_add_f32 v230, v230, v58 :: v_dual_add_f32 v231, v231, v59// 00000000A354: C90875E6 E6E677E7
	v_dual_add_f32 v244, v244, v68 :: v_dual_add_f32 v245, v245, v69// 00000000A35C: C90889F4 F4F48BF5
	v_dual_add_f32 v246, v246, v78 :: v_dual_add_f32 v247, v247, v79// 00000000A364: C9089DF6 F6F69FF7
	s_cmp_gt_u32 s37, 2                                        // 00000000A36C: BF088225
	s_cbranch_scc0 362                                         // 00000000A370: BFA1016A <gemv_mq2g256v2_xbatch_pm+0x921c>
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:144    // 00000000A374: C4050010 40802810 00009001
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:144    // 00000000A380: C4050011 40802811 00009001
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:144    // 00000000A38C: C4050012 40802812 00009001
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:144    // 00000000A398: C4050013 40802813 00009001
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:152    // 00000000A3A4: C4050010 40802820 00009802
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:152    // 00000000A3B0: C4050011 40802821 00009802
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:152    // 00000000A3BC: C4050012 40802822 00009802
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:152    // 00000000A3C8: C4050013 40802823 00009802
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048// 00000000A3D4: C405C01C 40803050 00080003
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064// 00000000A3E0: C405C01C 40803054 00081003
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:2048// 00000000A3EC: C405C01D 40803058 00080003
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:2064// 00000000A3F8: C405C01D 4080305C 00081003
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:2048// 00000000A404: C405C01E 40803060 00080003
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:2064// 00000000A410: C405C01E 40803064 00081003
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:2048// 00000000A41C: C405C01F 40803068 00080003
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:2064// 00000000A428: C405C01F 4080306C 00081003
	s_wait_loadcnt 0xb                                         // 00000000A434: BFC0000B
	v_bfe_u32 v40, v32, 0, 2                                   // 00000000A438: D6100028 02090120
	s_wait_loadcnt 0xa                                         // 00000000A440: BFC0000A
	v_bfe_u32 v50, v33, 0, 2                                   // 00000000A444: D6100032 02090121
	s_wait_loadcnt 0x9                                         // 00000000A44C: BFC00009
	v_bfe_u32 v60, v34, 0, 2                                   // 00000000A450: D610003C 02090122
	s_wait_loadcnt 0x8                                         // 00000000A458: BFC00008
	v_bfe_u32 v70, v35, 0, 2                                   // 00000000A45C: D6100046 02090123
	v_bfe_u32 v41, v32, 2, 2                                   // 00000000A464: D6100029 02090520
	v_bfe_u32 v51, v33, 2, 2                                   // 00000000A46C: D6100033 02090521
	v_bfe_u32 v61, v34, 2, 2                                   // 00000000A474: D610003D 02090522
	v_bfe_u32 v71, v35, 2, 2                                   // 00000000A47C: D6100047 02090523
	v_bfe_u32 v42, v32, 4, 2                                   // 00000000A484: D610002A 02090920
	v_bfe_u32 v52, v33, 4, 2                                   // 00000000A48C: D6100034 02090921
	v_bfe_u32 v62, v34, 4, 2                                   // 00000000A494: D610003E 02090922
	v_bfe_u32 v72, v35, 4, 2                                   // 00000000A49C: D6100048 02090923
	v_bfe_u32 v43, v32, 6, 2                                   // 00000000A4A4: D610002B 02090D20
	v_bfe_u32 v53, v33, 6, 2                                   // 00000000A4AC: D6100035 02090D21
	v_bfe_u32 v63, v34, 6, 2                                   // 00000000A4B4: D610003F 02090D22
	v_bfe_u32 v73, v35, 6, 2                                   // 00000000A4BC: D6100049 02090D23
	v_bfe_u32 v44, v32, 8, 2                                   // 00000000A4C4: D610002C 02091120
	v_bfe_u32 v54, v33, 8, 2                                   // 00000000A4CC: D6100036 02091121
	v_bfe_u32 v64, v34, 8, 2                                   // 00000000A4D4: D6100040 02091122
	v_bfe_u32 v74, v35, 8, 2                                   // 00000000A4DC: D610004A 02091123
	v_bfe_u32 v45, v32, 10, 2                                  // 00000000A4E4: D610002D 02091520
	v_bfe_u32 v55, v33, 10, 2                                  // 00000000A4EC: D6100037 02091521
	v_bfe_u32 v65, v34, 10, 2                                  // 00000000A4F4: D6100041 02091522
	v_bfe_u32 v75, v35, 10, 2                                  // 00000000A4FC: D610004B 02091523
	v_bfe_u32 v46, v32, 12, 2                                  // 00000000A504: D610002E 02091920
	v_bfe_u32 v56, v33, 12, 2                                  // 00000000A50C: D6100038 02091921
	v_bfe_u32 v66, v34, 12, 2                                  // 00000000A514: D6100042 02091922
	v_bfe_u32 v76, v35, 12, 2                                  // 00000000A51C: D610004C 02091923
	v_bfe_u32 v47, v32, 14, 2                                  // 00000000A524: D610002F 02091D20
	v_bfe_u32 v57, v33, 14, 2                                  // 00000000A52C: D6100039 02091D21
	v_bfe_u32 v67, v34, 14, 2                                  // 00000000A534: D6100043 02091D22
	v_bfe_u32 v77, v35, 14, 2                                  // 00000000A53C: D610004D 02091D23
	v_cvt_f32_ubyte0_e32 v40, v40                              // 00000000A544: 7E502328
	v_cvt_f32_ubyte0_e32 v50, v50                              // 00000000A548: 7E642332
	v_cvt_f32_ubyte0_e32 v60, v60                              // 00000000A54C: 7E78233C
	v_cvt_f32_ubyte0_e32 v70, v70                              // 00000000A550: 7E8C2346
	v_cvt_f32_ubyte0_e32 v41, v41                              // 00000000A554: 7E522329
	v_cvt_f32_ubyte0_e32 v51, v51                              // 00000000A558: 7E662333
	v_cvt_f32_ubyte0_e32 v61, v61                              // 00000000A55C: 7E7A233D
	v_cvt_f32_ubyte0_e32 v71, v71                              // 00000000A560: 7E8E2347
	v_cvt_f32_ubyte0_e32 v42, v42                              // 00000000A564: 7E54232A
	v_cvt_f32_ubyte0_e32 v52, v52                              // 00000000A568: 7E682334
	v_cvt_f32_ubyte0_e32 v62, v62                              // 00000000A56C: 7E7C233E
	v_cvt_f32_ubyte0_e32 v72, v72                              // 00000000A570: 7E902348
	v_cvt_f32_ubyte0_e32 v43, v43                              // 00000000A574: 7E56232B
	v_cvt_f32_ubyte0_e32 v53, v53                              // 00000000A578: 7E6A2335
	v_cvt_f32_ubyte0_e32 v63, v63                              // 00000000A57C: 7E7E233F
	v_cvt_f32_ubyte0_e32 v73, v73                              // 00000000A580: 7E922349
	v_cvt_f32_ubyte0_e32 v44, v44                              // 00000000A584: 7E58232C
	v_cvt_f32_ubyte0_e32 v54, v54                              // 00000000A588: 7E6C2336
	v_cvt_f32_ubyte0_e32 v64, v64                              // 00000000A58C: 7E802340
	v_cvt_f32_ubyte0_e32 v74, v74                              // 00000000A590: 7E94234A
	v_cvt_f32_ubyte0_e32 v45, v45                              // 00000000A594: 7E5A232D
	v_cvt_f32_ubyte0_e32 v55, v55                              // 00000000A598: 7E6E2337
	v_cvt_f32_ubyte0_e32 v65, v65                              // 00000000A59C: 7E822341
	v_cvt_f32_ubyte0_e32 v75, v75                              // 00000000A5A0: 7E96234B
	v_cvt_f32_ubyte0_e32 v46, v46                              // 00000000A5A4: 7E5C232E
	v_cvt_f32_ubyte0_e32 v56, v56                              // 00000000A5A8: 7E702338
	v_cvt_f32_ubyte0_e32 v66, v66                              // 00000000A5AC: 7E842342
	v_cvt_f32_ubyte0_e32 v76, v76                              // 00000000A5B0: 7E98234C
	v_cvt_f32_ubyte0_e32 v47, v47                              // 00000000A5B4: 7E5E232F
	v_cvt_f32_ubyte0_e32 v57, v57                              // 00000000A5B8: 7E722339
	v_cvt_f32_ubyte0_e32 v67, v67                              // 00000000A5BC: 7E862343
	v_cvt_f32_ubyte0_e32 v77, v77                              // 00000000A5C0: 7E9A234D
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A5C4: CC206028 0C425110
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A5CC: CC206032 0C466511
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A5D4: CC20603C 0C4A7912
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A5DC: CC206046 0C4E8D13
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A5E4: CC206029 0C425310
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A5EC: CC206033 0C466711
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A5F4: CC20603D 0C4A7B12
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A5FC: CC206047 0C4E8F13
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A604: CC20602A 0C425510
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A60C: CC206034 0C466911
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A614: CC20603E 0C4A7D12
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A61C: CC206048 0C4E9113
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A624: CC20602B 0C425710
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A62C: CC206035 0C466B11
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A634: CC20603F 0C4A7F12
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A63C: CC206049 0C4E9313
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A644: CC20602C 0C425910
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A64C: CC206036 0C466D11
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A654: CC206040 0C4A8112
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A65C: CC20604A 0C4E9513
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A664: CC20602D 0C425B10
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A66C: CC206037 0C466F11
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A674: CC206041 0C4A8312
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A67C: CC20604B 0C4E9713
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A684: CC20602E 0C425D10
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A68C: CC206038 0C467111
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A694: CC206042 0C4A8512
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A69C: CC20604C 0C4E9913
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A6A4: CC20602F 0C425F10
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A6AC: CC206039 0C467311
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A6B4: CC206043 0C4A8712
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]// 00000000A6BC: CC20604D 0C4E9B13
	s_wait_loadcnt 0x7                                         // 00000000A6C4: BFC00007
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81// 00000000A6C8: C8C6A329 3030A333
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81// 00000000A6D0: C8C6A33D 3A3AA347
	s_wait_loadcnt 0x5                                         // 00000000A6D8: BFC00005
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89// 00000000A6DC: C8C6B329 4444B333
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89// 00000000A6E4: C8C6B33D 4E4EB347
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80// 00000000A6EC: C800A128 3030A132
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80// 00000000A6F4: C800A13C 3A3AA146
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88// 00000000A6FC: C800B128 4444B132
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88// 00000000A704: C800B13C 4E4EB146
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82// 00000000A70C: C800A52A 3030A534
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82// 00000000A714: C800A53E 3A3AA548
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90// 00000000A71C: C800B52A 4444B534
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90// 00000000A724: C800B53E 4E4EB548
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83// 00000000A72C: C800A72B 3030A735
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83// 00000000A734: C800A73F 3A3AA749
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91// 00000000A73C: C800B72B 4444B735
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91// 00000000A744: C800B73F 4E4EB749
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84// 00000000A74C: C800A92C 3030A936
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84// 00000000A754: C800A940 3A3AA94A
	s_wait_loadcnt 0x4                                         // 00000000A75C: BFC00004
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92// 00000000A760: C800B92C 4444B936
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92// 00000000A768: C800B940 4E4EB94A
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85// 00000000A770: C800AB2D 3030AB37
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85// 00000000A778: C800AB41 3A3AAB4B
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93// 00000000A780: C800BB2D 4444BB37
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93// 00000000A788: C800BB41 4E4EBB4B
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86// 00000000A790: C800AD2E 3030AD38
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86// 00000000A798: C800AD42 3A3AAD4C
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94// 00000000A7A0: C800BD2E 4444BD38
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94// 00000000A7A8: C800BD42 4E4EBD4C
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87// 00000000A7B0: C800AF2F 3030AF39
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87// 00000000A7B8: C800AF43 3A3AAF4D
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95// 00000000A7C0: C800BF2F 4444BF39
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95// 00000000A7C8: C800BF43 4E4EBF4D
	v_dual_add_f32 v200, v200, v48 :: v_dual_add_f32 v201, v201, v49// 00000000A7D0: C90861C8 C8C863C9
	v_dual_add_f32 v202, v202, v58 :: v_dual_add_f32 v203, v203, v59// 00000000A7D8: C90875CA CACA77CB
	v_dual_add_f32 v216, v216, v68 :: v_dual_add_f32 v217, v217, v69// 00000000A7E0: C90889D8 D8D88BD9
	v_dual_add_f32 v218, v218, v78 :: v_dual_add_f32 v219, v219, v79// 00000000A7E8: C9089DDA DADA9FDB
	s_wait_loadcnt 0x3                                         // 00000000A7F0: BFC00003
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97// 00000000A7F4: C8C6C329 3030C333
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97// 00000000A7FC: C8C6C33D 3A3AC347
	s_wait_loadcnt 0x1                                         // 00000000A804: BFC00001
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105// 00000000A808: C8C6D329 4444D333
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105// 00000000A810: C8C6D33D 4E4ED347
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96// 00000000A818: C800C128 3030C132
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96// 00000000A820: C800C13C 3A3AC146
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104// 00000000A828: C800D128 4444D132
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104// 00000000A830: C800D13C 4E4ED146
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98// 00000000A838: C800C52A 3030C534
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98// 00000000A840: C800C53E 3A3AC548
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106// 00000000A848: C800D52A 4444D534
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106// 00000000A850: C800D53E 4E4ED548
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99// 00000000A858: C800C72B 3030C735
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99// 00000000A860: C800C73F 3A3AC749
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107// 00000000A868: C800D72B 4444D735
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107// 00000000A870: C800D73F 4E4ED749
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100// 00000000A878: C800C92C 3030C936
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100// 00000000A880: C800C940 3A3AC94A
	s_wait_loadcnt 0x0                                         // 00000000A888: BFC00000
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108// 00000000A88C: C800D92C 4444D936
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108// 00000000A894: C800D940 4E4ED94A
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101// 00000000A89C: C800CB2D 3030CB37
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101// 00000000A8A4: C800CB41 3A3ACB4B
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109// 00000000A8AC: C800DB2D 4444DB37
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109// 00000000A8B4: C800DB41 4E4EDB4B
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102// 00000000A8BC: C800CD2E 3030CD38
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102// 00000000A8C4: C800CD42 3A3ACD4C
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110// 00000000A8CC: C800DD2E 4444DD38
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110// 00000000A8D4: C800DD42 4E4EDD4C
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103// 00000000A8DC: C800CF2F 3030CF39
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103// 00000000A8E4: C800CF43 3A3ACF4D
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111// 00000000A8EC: C800DF2F 4444DF39
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111// 00000000A8F4: C800DF43 4E4EDF4D
	v_dual_add_f32 v232, v232, v48 :: v_dual_add_f32 v233, v233, v49// 00000000A8FC: C90861E8 E8E863E9
	v_dual_add_f32 v234, v234, v58 :: v_dual_add_f32 v235, v235, v59// 00000000A904: C90875EA EAEA77EB
	v_dual_add_f32 v248, v248, v68 :: v_dual_add_f32 v249, v249, v69// 00000000A90C: C90889F8 F8F88BF9
	v_dual_add_f32 v250, v250, v78 :: v_dual_add_f32 v251, v251, v79// 00000000A914: C9089DFA FAFA9FFB
	v_dual_add_f32 v192, v192, v196 :: v_dual_add_f32 v193, v193, v197// 00000000A91C: C90989C0 C0C18BC1
	v_dual_add_f32 v194, v194, v198 :: v_dual_add_f32 v195, v195, v199// 00000000A924: C9098DC2 C2C38FC3
	v_dual_add_f32 v200, v200, v204 :: v_dual_add_f32 v201, v201, v205// 00000000A92C: C90999C8 C8C99BC9
	v_dual_add_f32 v202, v202, v206 :: v_dual_add_f32 v203, v203, v207// 00000000A934: C9099DCA CACB9FCB
	v_dual_add_f32 v192, v192, v200 :: v_dual_add_f32 v193, v193, v201// 00000000A93C: C90991C0 C0C193C1
	v_dual_add_f32 v194, v194, v202 :: v_dual_add_f32 v195, v195, v203// 00000000A944: C90995C2 C2C397C3
	v_dual_add_f32 v208, v208, v212 :: v_dual_add_f32 v209, v209, v213// 00000000A94C: C909A9D0 D0D1ABD1
	v_dual_add_f32 v210, v210, v214 :: v_dual_add_f32 v211, v211, v215// 00000000A954: C909ADD2 D2D3AFD3
	v_dual_add_f32 v216, v216, v220 :: v_dual_add_f32 v217, v217, v221// 00000000A95C: C909B9D8 D8D9BBD9
	v_dual_add_f32 v218, v218, v222 :: v_dual_add_f32 v219, v219, v223// 00000000A964: C909BDDA DADBBFDB
	v_dual_add_f32 v208, v208, v216 :: v_dual_add_f32 v209, v209, v217// 00000000A96C: C909B1D0 D0D1B3D1
	v_dual_add_f32 v210, v210, v218 :: v_dual_add_f32 v211, v211, v219// 00000000A974: C909B5D2 D2D3B7D3
	v_dual_add_f32 v224, v224, v228 :: v_dual_add_f32 v225, v225, v229// 00000000A97C: C909C9E0 E0E1CBE1
	v_dual_add_f32 v226, v226, v230 :: v_dual_add_f32 v227, v227, v231// 00000000A984: C909CDE2 E2E3CFE3
	v_dual_add_f32 v232, v232, v236 :: v_dual_add_f32 v233, v233, v237// 00000000A98C: C909D9E8 E8E9DBE9
	v_dual_add_f32 v234, v234, v238 :: v_dual_add_f32 v235, v235, v239// 00000000A994: C909DDEA EAEBDFEB
	v_dual_add_f32 v224, v224, v232 :: v_dual_add_f32 v225, v225, v233// 00000000A99C: C909D1E0 E0E1D3E1
	v_dual_add_f32 v226, v226, v234 :: v_dual_add_f32 v227, v227, v235// 00000000A9A4: C909D5E2 E2E3D7E3
	v_dual_add_f32 v240, v240, v244 :: v_dual_add_f32 v241, v241, v245// 00000000A9AC: C909E9F0 F0F1EBF1
	v_dual_add_f32 v242, v242, v246 :: v_dual_add_f32 v243, v243, v247// 00000000A9B4: C909EDF2 F2F3EFF3
	v_dual_add_f32 v248, v248, v252 :: v_dual_add_f32 v249, v249, v253// 00000000A9BC: C909F9F8 F8F9FBF9
	v_dual_add_f32 v250, v250, v254 :: v_dual_add_f32 v251, v251, v255// 00000000A9C4: C909FDFA FAFBFFFB
	v_dual_add_f32 v240, v240, v248 :: v_dual_add_f32 v241, v241, v249// 00000000A9CC: C909F1F0 F0F1F3F1
	v_dual_add_f32 v242, v242, v250 :: v_dual_add_f32 v243, v243, v251// 00000000A9D4: C909F5F2 F2F3F7F3
	ds_swizzle_b32 v8, v192 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000A9DC: D8D4020F 080000C0
	ds_swizzle_b32 v9, v193 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000A9E4: D8D4020F 090000C1
	ds_swizzle_b32 v10, v194 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000A9EC: D8D4020F 0A0000C2
	ds_swizzle_b32 v11, v195 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000A9F4: D8D4020F 0B0000C3
	ds_swizzle_b32 v12, v208 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000A9FC: D8D4020F 0C0000D0
	ds_swizzle_b32 v13, v209 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA04: D8D4020F 0D0000D1
	ds_swizzle_b32 v14, v210 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA0C: D8D4020F 0E0000D2
	ds_swizzle_b32 v15, v211 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA14: D8D4020F 0F0000D3
	ds_swizzle_b32 v16, v224 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA1C: D8D4020F 100000E0
	ds_swizzle_b32 v17, v225 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA24: D8D4020F 110000E1
	ds_swizzle_b32 v18, v226 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA2C: D8D4020F 120000E2
	ds_swizzle_b32 v19, v227 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA34: D8D4020F 130000E3
	ds_swizzle_b32 v20, v240 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA3C: D8D4020F 140000F0
	ds_swizzle_b32 v21, v241 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA44: D8D4020F 150000F1
	ds_swizzle_b32 v22, v242 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA4C: D8D4020F 160000F2
	ds_swizzle_b32 v23, v243 offset:swizzle(BITMASK_PERM,"1pppp")// 00000000AA54: D8D4020F 170000F3
	s_wait_dscnt 0xf                                           // 00000000AA5C: BFC6000F
	v_add_f32_e32 v192, v192, v8                               // 00000000AA60: 078011C0
	s_wait_dscnt 0xe                                           // 00000000AA64: BFC6000E
	v_add_f32_e32 v193, v193, v9                               // 00000000AA68: 078213C1
	s_wait_dscnt 0xd                                           // 00000000AA6C: BFC6000D
	v_add_f32_e32 v194, v194, v10                              // 00000000AA70: 078415C2
	s_wait_dscnt 0xc                                           // 00000000AA74: BFC6000C
	v_add_f32_e32 v195, v195, v11                              // 00000000AA78: 078617C3
	s_wait_dscnt 0xb                                           // 00000000AA7C: BFC6000B
	v_add_f32_e32 v208, v208, v12                              // 00000000AA80: 07A019D0
	s_wait_dscnt 0xa                                           // 00000000AA84: BFC6000A
	v_add_f32_e32 v209, v209, v13                              // 00000000AA88: 07A21BD1
	s_wait_dscnt 0x9                                           // 00000000AA8C: BFC60009
	v_add_f32_e32 v210, v210, v14                              // 00000000AA90: 07A41DD2
	s_wait_dscnt 0x8                                           // 00000000AA94: BFC60008
	v_add_f32_e32 v211, v211, v15                              // 00000000AA98: 07A61FD3
	s_wait_dscnt 0x7                                           // 00000000AA9C: BFC60007
	v_add_f32_e32 v224, v224, v16                              // 00000000AAA0: 07C021E0
	s_wait_dscnt 0x6                                           // 00000000AAA4: BFC60006
	v_add_f32_e32 v225, v225, v17                              // 00000000AAA8: 07C223E1
	s_wait_dscnt 0x5                                           // 00000000AAAC: BFC60005
	v_add_f32_e32 v226, v226, v18                              // 00000000AAB0: 07C425E2
	s_wait_dscnt 0x4                                           // 00000000AAB4: BFC60004
	v_add_f32_e32 v227, v227, v19                              // 00000000AAB8: 07C627E3
	s_wait_dscnt 0x3                                           // 00000000AABC: BFC60003
	v_add_f32_e32 v240, v240, v20                              // 00000000AAC0: 07E029F0
	s_wait_dscnt 0x2                                           // 00000000AAC4: BFC60002
	v_add_f32_e32 v241, v241, v21                              // 00000000AAC8: 07E22BF1
	s_wait_dscnt 0x1                                           // 00000000AACC: BFC60001
	v_add_f32_e32 v242, v242, v22                              // 00000000AAD0: 07E42DF2
	s_wait_dscnt 0x0                                           // 00000000AAD4: BFC60000
	v_add_f32_e32 v243, v243, v23                              // 00000000AAD8: 07E62FF3
	v_cmp_gt_u32_e32 vcc_lo, 24, v0                            // 00000000AADC: 7C980098
	s_wait_alu depctr_va_vcc(0)                                // 00000000AAE0: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo                         // 00000000AAE4: D5010004 01A91080
	v_add_lshl_u32 v4, v4, v0, 2                               // 00000000AAEC: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v192                               // 00000000AAF4: DACC0000 0800C004
	ds_bpermute_b32 v9, v4, v193                               // 00000000AAFC: DACC0000 0900C104
	ds_bpermute_b32 v10, v4, v194                              // 00000000AB04: DACC0000 0A00C204
	ds_bpermute_b32 v11, v4, v195                              // 00000000AB0C: DACC0000 0B00C304
	ds_bpermute_b32 v12, v4, v208                              // 00000000AB14: DACC0000 0C00D004
	ds_bpermute_b32 v13, v4, v209                              // 00000000AB1C: DACC0000 0D00D104
	ds_bpermute_b32 v14, v4, v210                              // 00000000AB24: DACC0000 0E00D204
	ds_bpermute_b32 v15, v4, v211                              // 00000000AB2C: DACC0000 0F00D304
	ds_bpermute_b32 v16, v4, v224                              // 00000000AB34: DACC0000 1000E004
	ds_bpermute_b32 v17, v4, v225                              // 00000000AB3C: DACC0000 1100E104
	ds_bpermute_b32 v18, v4, v226                              // 00000000AB44: DACC0000 1200E204
	ds_bpermute_b32 v19, v4, v227                              // 00000000AB4C: DACC0000 1300E304
	ds_bpermute_b32 v20, v4, v240                              // 00000000AB54: DACC0000 1400F004
	ds_bpermute_b32 v21, v4, v241                              // 00000000AB5C: DACC0000 1500F104
	ds_bpermute_b32 v22, v4, v242                              // 00000000AB64: DACC0000 1600F204
	ds_bpermute_b32 v23, v4, v243                              // 00000000AB6C: DACC0000 1700F304
	s_wait_dscnt 0xf                                           // 00000000AB74: BFC6000F
	v_add_f32_e32 v192, v192, v8                               // 00000000AB78: 078011C0
	s_wait_dscnt 0xe                                           // 00000000AB7C: BFC6000E
	v_add_f32_e32 v193, v193, v9                               // 00000000AB80: 078213C1
	s_wait_dscnt 0xd                                           // 00000000AB84: BFC6000D
	v_add_f32_e32 v194, v194, v10                              // 00000000AB88: 078415C2
	s_wait_dscnt 0xc                                           // 00000000AB8C: BFC6000C
	v_add_f32_e32 v195, v195, v11                              // 00000000AB90: 078617C3
	s_wait_dscnt 0xb                                           // 00000000AB94: BFC6000B
	v_add_f32_e32 v208, v208, v12                              // 00000000AB98: 07A019D0
	s_wait_dscnt 0xa                                           // 00000000AB9C: BFC6000A
	v_add_f32_e32 v209, v209, v13                              // 00000000ABA0: 07A21BD1
	s_wait_dscnt 0x9                                           // 00000000ABA4: BFC60009
	v_add_f32_e32 v210, v210, v14                              // 00000000ABA8: 07A41DD2
	s_wait_dscnt 0x8                                           // 00000000ABAC: BFC60008
	v_add_f32_e32 v211, v211, v15                              // 00000000ABB0: 07A61FD3
	s_wait_dscnt 0x7                                           // 00000000ABB4: BFC60007
	v_add_f32_e32 v224, v224, v16                              // 00000000ABB8: 07C021E0
	s_wait_dscnt 0x6                                           // 00000000ABBC: BFC60006
	v_add_f32_e32 v225, v225, v17                              // 00000000ABC0: 07C223E1
	s_wait_dscnt 0x5                                           // 00000000ABC4: BFC60005
	v_add_f32_e32 v226, v226, v18                              // 00000000ABC8: 07C425E2
	s_wait_dscnt 0x4                                           // 00000000ABCC: BFC60004
	v_add_f32_e32 v227, v227, v19                              // 00000000ABD0: 07C627E3
	s_wait_dscnt 0x3                                           // 00000000ABD4: BFC60003
	v_add_f32_e32 v240, v240, v20                              // 00000000ABD8: 07E029F0
	s_wait_dscnt 0x2                                           // 00000000ABDC: BFC60002
	v_add_f32_e32 v241, v241, v21                              // 00000000ABE0: 07E22BF1
	s_wait_dscnt 0x1                                           // 00000000ABE4: BFC60001
	v_add_f32_e32 v242, v242, v22                              // 00000000ABE8: 07E42DF2
	s_wait_dscnt 0x0                                           // 00000000ABEC: BFC60000
	v_add_f32_e32 v243, v243, v23                              // 00000000ABF0: 07E62FF3
	v_cmp_gt_u32_e32 vcc_lo, 28, v0                            // 00000000ABF4: 7C98009C
	s_wait_alu depctr_va_vcc(0)                                // 00000000ABF8: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo                         // 00000000ABFC: D5010004 01A90880
	v_add_lshl_u32 v4, v4, v0, 2                               // 00000000AC04: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v192                               // 00000000AC0C: DACC0000 0800C004
	ds_bpermute_b32 v9, v4, v193                               // 00000000AC14: DACC0000 0900C104
	ds_bpermute_b32 v10, v4, v194                              // 00000000AC1C: DACC0000 0A00C204
	ds_bpermute_b32 v11, v4, v195                              // 00000000AC24: DACC0000 0B00C304
	ds_bpermute_b32 v12, v4, v208                              // 00000000AC2C: DACC0000 0C00D004
	ds_bpermute_b32 v13, v4, v209                              // 00000000AC34: DACC0000 0D00D104
	ds_bpermute_b32 v14, v4, v210                              // 00000000AC3C: DACC0000 0E00D204
	ds_bpermute_b32 v15, v4, v211                              // 00000000AC44: DACC0000 0F00D304
	ds_bpermute_b32 v16, v4, v224                              // 00000000AC4C: DACC0000 1000E004
	ds_bpermute_b32 v17, v4, v225                              // 00000000AC54: DACC0000 1100E104
	ds_bpermute_b32 v18, v4, v226                              // 00000000AC5C: DACC0000 1200E204
	ds_bpermute_b32 v19, v4, v227                              // 00000000AC64: DACC0000 1300E304
	ds_bpermute_b32 v20, v4, v240                              // 00000000AC6C: DACC0000 1400F004
	ds_bpermute_b32 v21, v4, v241                              // 00000000AC74: DACC0000 1500F104
	ds_bpermute_b32 v22, v4, v242                              // 00000000AC7C: DACC0000 1600F204
	ds_bpermute_b32 v23, v4, v243                              // 00000000AC84: DACC0000 1700F304
	s_wait_dscnt 0xf                                           // 00000000AC8C: BFC6000F
	v_add_f32_e32 v192, v192, v8                               // 00000000AC90: 078011C0
	s_wait_dscnt 0xe                                           // 00000000AC94: BFC6000E
	v_add_f32_e32 v193, v193, v9                               // 00000000AC98: 078213C1
	s_wait_dscnt 0xd                                           // 00000000AC9C: BFC6000D
	v_add_f32_e32 v194, v194, v10                              // 00000000ACA0: 078415C2
	s_wait_dscnt 0xc                                           // 00000000ACA4: BFC6000C
	v_add_f32_e32 v195, v195, v11                              // 00000000ACA8: 078617C3
	s_wait_dscnt 0xb                                           // 00000000ACAC: BFC6000B
	v_add_f32_e32 v208, v208, v12                              // 00000000ACB0: 07A019D0
	s_wait_dscnt 0xa                                           // 00000000ACB4: BFC6000A
	v_add_f32_e32 v209, v209, v13                              // 00000000ACB8: 07A21BD1
	s_wait_dscnt 0x9                                           // 00000000ACBC: BFC60009
	v_add_f32_e32 v210, v210, v14                              // 00000000ACC0: 07A41DD2
	s_wait_dscnt 0x8                                           // 00000000ACC4: BFC60008
	v_add_f32_e32 v211, v211, v15                              // 00000000ACC8: 07A61FD3
	s_wait_dscnt 0x7                                           // 00000000ACCC: BFC60007
	v_add_f32_e32 v224, v224, v16                              // 00000000ACD0: 07C021E0
	s_wait_dscnt 0x6                                           // 00000000ACD4: BFC60006
	v_add_f32_e32 v225, v225, v17                              // 00000000ACD8: 07C223E1
	s_wait_dscnt 0x5                                           // 00000000ACDC: BFC60005
	v_add_f32_e32 v226, v226, v18                              // 00000000ACE0: 07C425E2
	s_wait_dscnt 0x4                                           // 00000000ACE4: BFC60004
	v_add_f32_e32 v227, v227, v19                              // 00000000ACE8: 07C627E3
	s_wait_dscnt 0x3                                           // 00000000ACEC: BFC60003
	v_add_f32_e32 v240, v240, v20                              // 00000000ACF0: 07E029F0
	s_wait_dscnt 0x2                                           // 00000000ACF4: BFC60002
	v_add_f32_e32 v241, v241, v21                              // 00000000ACF8: 07E22BF1
	s_wait_dscnt 0x1                                           // 00000000ACFC: BFC60001
	v_add_f32_e32 v242, v242, v22                              // 00000000AD00: 07E42DF2
	s_wait_dscnt 0x0                                           // 00000000AD04: BFC60000
	v_add_f32_e32 v243, v243, v23                              // 00000000AD08: 07E62FF3
	v_cmp_gt_u32_e32 vcc_lo, 30, v0                            // 00000000AD0C: 7C98009E
	s_wait_alu depctr_va_vcc(0)                                // 00000000AD10: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo                         // 00000000AD14: D5010004 01A90480
	v_add_lshl_u32 v4, v4, v0, 2                               // 00000000AD1C: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v192                               // 00000000AD24: DACC0000 0800C004
	ds_bpermute_b32 v9, v4, v193                               // 00000000AD2C: DACC0000 0900C104
	ds_bpermute_b32 v10, v4, v194                              // 00000000AD34: DACC0000 0A00C204
	ds_bpermute_b32 v11, v4, v195                              // 00000000AD3C: DACC0000 0B00C304
	ds_bpermute_b32 v12, v4, v208                              // 00000000AD44: DACC0000 0C00D004
	ds_bpermute_b32 v13, v4, v209                              // 00000000AD4C: DACC0000 0D00D104
	ds_bpermute_b32 v14, v4, v210                              // 00000000AD54: DACC0000 0E00D204
	ds_bpermute_b32 v15, v4, v211                              // 00000000AD5C: DACC0000 0F00D304
	ds_bpermute_b32 v16, v4, v224                              // 00000000AD64: DACC0000 1000E004
	ds_bpermute_b32 v17, v4, v225                              // 00000000AD6C: DACC0000 1100E104
	ds_bpermute_b32 v18, v4, v226                              // 00000000AD74: DACC0000 1200E204
	ds_bpermute_b32 v19, v4, v227                              // 00000000AD7C: DACC0000 1300E304
	ds_bpermute_b32 v20, v4, v240                              // 00000000AD84: DACC0000 1400F004
	ds_bpermute_b32 v21, v4, v241                              // 00000000AD8C: DACC0000 1500F104
	ds_bpermute_b32 v22, v4, v242                              // 00000000AD94: DACC0000 1600F204
	ds_bpermute_b32 v23, v4, v243                              // 00000000AD9C: DACC0000 1700F304
	s_wait_dscnt 0xf                                           // 00000000ADA4: BFC6000F
	v_add_f32_e32 v192, v192, v8                               // 00000000ADA8: 078011C0
	s_wait_dscnt 0xe                                           // 00000000ADAC: BFC6000E
	v_add_f32_e32 v193, v193, v9                               // 00000000ADB0: 078213C1
	s_wait_dscnt 0xd                                           // 00000000ADB4: BFC6000D
	v_add_f32_e32 v194, v194, v10                              // 00000000ADB8: 078415C2
	s_wait_dscnt 0xc                                           // 00000000ADBC: BFC6000C
	v_add_f32_e32 v195, v195, v11                              // 00000000ADC0: 078617C3
	s_wait_dscnt 0xb                                           // 00000000ADC4: BFC6000B
	v_add_f32_e32 v208, v208, v12                              // 00000000ADC8: 07A019D0
	s_wait_dscnt 0xa                                           // 00000000ADCC: BFC6000A
	v_add_f32_e32 v209, v209, v13                              // 00000000ADD0: 07A21BD1
	s_wait_dscnt 0x9                                           // 00000000ADD4: BFC60009
	v_add_f32_e32 v210, v210, v14                              // 00000000ADD8: 07A41DD2
	s_wait_dscnt 0x8                                           // 00000000ADDC: BFC60008
	v_add_f32_e32 v211, v211, v15                              // 00000000ADE0: 07A61FD3
	s_wait_dscnt 0x7                                           // 00000000ADE4: BFC60007
	v_add_f32_e32 v224, v224, v16                              // 00000000ADE8: 07C021E0
	s_wait_dscnt 0x6                                           // 00000000ADEC: BFC60006
	v_add_f32_e32 v225, v225, v17                              // 00000000ADF0: 07C223E1
	s_wait_dscnt 0x5                                           // 00000000ADF4: BFC60005
	v_add_f32_e32 v226, v226, v18                              // 00000000ADF8: 07C425E2
	s_wait_dscnt 0x4                                           // 00000000ADFC: BFC60004
	v_add_f32_e32 v227, v227, v19                              // 00000000AE00: 07C627E3
	s_wait_dscnt 0x3                                           // 00000000AE04: BFC60003
	v_add_f32_e32 v240, v240, v20                              // 00000000AE08: 07E029F0
	s_wait_dscnt 0x2                                           // 00000000AE0C: BFC60002
	v_add_f32_e32 v241, v241, v21                              // 00000000AE10: 07E22BF1
	s_wait_dscnt 0x1                                           // 00000000AE14: BFC60001
	v_add_f32_e32 v242, v242, v22                              // 00000000AE18: 07E42DF2
	s_wait_dscnt 0x0                                           // 00000000AE1C: BFC60000
	v_add_f32_e32 v243, v243, v23                              // 00000000AE20: 07E62FF3
	v_cmp_gt_u32_e32 vcc_lo, 31, v0                            // 00000000AE24: 7C98009F
	s_wait_alu depctr_va_vcc(0)                                // 00000000AE28: BF88FF9D
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo                         // 00000000AE2C: D5010004 01A90280
	v_add_lshl_u32 v4, v4, v0, 2                               // 00000000AE34: D6470004 020A0104
	ds_bpermute_b32 v8, v4, v192                               // 00000000AE3C: DACC0000 0800C004
	ds_bpermute_b32 v9, v4, v193                               // 00000000AE44: DACC0000 0900C104
	ds_bpermute_b32 v10, v4, v194                              // 00000000AE4C: DACC0000 0A00C204
	ds_bpermute_b32 v11, v4, v195                              // 00000000AE54: DACC0000 0B00C304
	ds_bpermute_b32 v12, v4, v208                              // 00000000AE5C: DACC0000 0C00D004
	ds_bpermute_b32 v13, v4, v209                              // 00000000AE64: DACC0000 0D00D104
	ds_bpermute_b32 v14, v4, v210                              // 00000000AE6C: DACC0000 0E00D204
	ds_bpermute_b32 v15, v4, v211                              // 00000000AE74: DACC0000 0F00D304
	ds_bpermute_b32 v16, v4, v224                              // 00000000AE7C: DACC0000 1000E004
	ds_bpermute_b32 v17, v4, v225                              // 00000000AE84: DACC0000 1100E104
	ds_bpermute_b32 v18, v4, v226                              // 00000000AE8C: DACC0000 1200E204
	ds_bpermute_b32 v19, v4, v227                              // 00000000AE94: DACC0000 1300E304
	ds_bpermute_b32 v20, v4, v240                              // 00000000AE9C: DACC0000 1400F004
	ds_bpermute_b32 v21, v4, v241                              // 00000000AEA4: DACC0000 1500F104
	ds_bpermute_b32 v22, v4, v242                              // 00000000AEAC: DACC0000 1600F204
	ds_bpermute_b32 v23, v4, v243                              // 00000000AEB4: DACC0000 1700F304
	s_wait_dscnt 0xf                                           // 00000000AEBC: BFC6000F
	v_add_f32_e32 v192, v192, v8                               // 00000000AEC0: 078011C0
	s_wait_dscnt 0xe                                           // 00000000AEC4: BFC6000E
	v_add_f32_e32 v193, v193, v9                               // 00000000AEC8: 078213C1
	s_wait_dscnt 0xd                                           // 00000000AECC: BFC6000D
	v_add_f32_e32 v194, v194, v10                              // 00000000AED0: 078415C2
	s_wait_dscnt 0xc                                           // 00000000AED4: BFC6000C
	v_add_f32_e32 v195, v195, v11                              // 00000000AED8: 078617C3
	s_wait_dscnt 0xb                                           // 00000000AEDC: BFC6000B
	v_add_f32_e32 v208, v208, v12                              // 00000000AEE0: 07A019D0
	s_wait_dscnt 0xa                                           // 00000000AEE4: BFC6000A
	v_add_f32_e32 v209, v209, v13                              // 00000000AEE8: 07A21BD1
	s_wait_dscnt 0x9                                           // 00000000AEEC: BFC60009
	v_add_f32_e32 v210, v210, v14                              // 00000000AEF0: 07A41DD2
	s_wait_dscnt 0x8                                           // 00000000AEF4: BFC60008
	v_add_f32_e32 v211, v211, v15                              // 00000000AEF8: 07A61FD3
	s_wait_dscnt 0x7                                           // 00000000AEFC: BFC60007
	v_add_f32_e32 v224, v224, v16                              // 00000000AF00: 07C021E0
	s_wait_dscnt 0x6                                           // 00000000AF04: BFC60006
	v_add_f32_e32 v225, v225, v17                              // 00000000AF08: 07C223E1
	s_wait_dscnt 0x5                                           // 00000000AF0C: BFC60005
	v_add_f32_e32 v226, v226, v18                              // 00000000AF10: 07C425E2
	s_wait_dscnt 0x4                                           // 00000000AF14: BFC60004
	v_add_f32_e32 v227, v227, v19                              // 00000000AF18: 07C627E3
	s_wait_dscnt 0x3                                           // 00000000AF1C: BFC60003
	v_add_f32_e32 v240, v240, v20                              // 00000000AF20: 07E029F0
	s_wait_dscnt 0x2                                           // 00000000AF24: BFC60002
	v_add_f32_e32 v241, v241, v21                              // 00000000AF28: 07E22BF1
	s_wait_dscnt 0x1                                           // 00000000AF2C: BFC60001
	v_add_f32_e32 v242, v242, v22                              // 00000000AF30: 07E42DF2
	s_wait_dscnt 0x0                                           // 00000000AF34: BFC60000
	v_add_f32_e32 v243, v243, v23                              // 00000000AF38: 07E62FF3
	v_cmpx_eq_u32_e32 0, v0                                    // 00000000AF3C: 7D940080
	v_mov_b32_e32 v40, s42                                     // 00000000AF40: 7E50022A
	s_mul_i32 s40, s39, 1                                      // 00000000AF44: 96288127
	s_wait_alu depctr_sa_sdst(0)                               // 00000000AF48: BF88FF9E
	s_add_co_i32 s40, s40, s42                                 // 00000000AF4C: 81282A28
	s_wait_alu depctr_sa_sdst(0)                               // 00000000AF50: BF88FF9E
	v_mov_b32_e32 v41, s40                                     // 00000000AF54: 7E520228
	s_mul_i32 s40, s39, 2                                      // 00000000AF58: 96288227
	s_wait_alu depctr_sa_sdst(0)                               // 00000000AF5C: BF88FF9E
	s_add_co_i32 s40, s40, s42                                 // 00000000AF60: 81282A28
	s_wait_alu depctr_sa_sdst(0)                               // 00000000AF64: BF88FF9E
	v_mov_b32_e32 v42, s40                                     // 00000000AF68: 7E540228
	s_mul_i32 s40, s39, 3                                      // 00000000AF6C: 96288327
	s_wait_alu depctr_sa_sdst(0)                               // 00000000AF70: BF88FF9E
	s_add_co_i32 s40, s40, s42                                 // 00000000AF74: 81282A28
	s_wait_alu depctr_sa_sdst(0)                               // 00000000AF78: BF88FF9E
	v_mov_b32_e32 v43, s40                                     // 00000000AF7C: 7E560228
	global_store_b32 v40, v192, s[8:9]                         // 00000000AF80: EE068008 60000000 00000028
	global_store_b32 v41, v208, s[8:9]                         // 00000000AF8C: EE068008 68000000 00000029
	global_store_b32 v42, v224, s[8:9]                         // 00000000AF98: EE068008 70000000 0000002A
	global_store_b32 v43, v240, s[8:9]                         // 00000000AFA4: EE068008 78000000 0000002B
	s_cmp_le_i32 s13, 1                                        // 00000000AFB0: BF05810D
	s_cbranch_scc1 40                                          // 00000000AFB4: BFA20028 <gemv_mq2g256v2_xbatch_pm+0x9958>
	global_store_b32 v40, v193, s[8:9] offset:4                // 00000000AFB8: EE068008 60800000 00000428
	global_store_b32 v41, v209, s[8:9] offset:4                // 00000000AFC4: EE068008 68800000 00000429
	global_store_b32 v42, v225, s[8:9] offset:4                // 00000000AFD0: EE068008 70800000 0000042A
	global_store_b32 v43, v241, s[8:9] offset:4                // 00000000AFDC: EE068008 78800000 0000042B
	s_cmp_le_i32 s13, 2                                        // 00000000AFE8: BF05820D
	s_cbranch_scc1 26                                          // 00000000AFEC: BFA2001A <gemv_mq2g256v2_xbatch_pm+0x9958>
	global_store_b32 v40, v194, s[8:9] offset:8                // 00000000AFF0: EE068008 61000000 00000828
	global_store_b32 v41, v210, s[8:9] offset:8                // 00000000AFFC: EE068008 69000000 00000829
	global_store_b32 v42, v226, s[8:9] offset:8                // 00000000B008: EE068008 71000000 0000082A
	global_store_b32 v43, v242, s[8:9] offset:8                // 00000000B014: EE068008 79000000 0000082B
	s_cmp_le_i32 s13, 3                                        // 00000000B020: BF05830D
	s_cbranch_scc1 12                                          // 00000000B024: BFA2000C <gemv_mq2g256v2_xbatch_pm+0x9958>
	global_store_b32 v40, v195, s[8:9] offset:12               // 00000000B028: EE068008 61800000 00000C28
	global_store_b32 v41, v211, s[8:9] offset:12               // 00000000B034: EE068008 69800000 00000C29
	global_store_b32 v42, v227, s[8:9] offset:12               // 00000000B040: EE068008 71800000 00000C2A
	global_store_b32 v43, v243, s[8:9] offset:12               // 00000000B04C: EE068008 79800000 00000C2B
	s_wait_storecnt 0x0                                        // 00000000B058: BFC10000
	s_branch 0                                                 // 00000000B05C: BFA00000 <gemv_mq2g256v2_xbatch_pm+0x9960>
	s_endpgm                                                   // 00000000B060: BFB00000
	s_code_end                                                 // 00000000B064: BF9F0000
	s_code_end                                                 // 00000000B068: BF9F0000
	s_code_end                                                 // 00000000B06C: BF9F0000
	s_code_end                                                 // 00000000B070: BF9F0000
	s_code_end                                                 // 00000000B074: BF9F0000
	s_code_end                                                 // 00000000B078: BF9F0000
	s_code_end                                                 // 00000000B07C: BF9F0000
	s_code_end                                                 // 00000000B080: BF9F0000
	s_code_end                                                 // 00000000B084: BF9F0000
	s_code_end                                                 // 00000000B088: BF9F0000
	s_code_end                                                 // 00000000B08C: BF9F0000
	s_code_end                                                 // 00000000B090: BF9F0000
	s_code_end                                                 // 00000000B094: BF9F0000
	s_code_end                                                 // 00000000B098: BF9F0000
	s_code_end                                                 // 00000000B09C: BF9F0000
	s_code_end                                                 // 00000000B0A0: BF9F0000
	s_code_end                                                 // 00000000B0A4: BF9F0000
	s_code_end                                                 // 00000000B0A8: BF9F0000
	s_code_end                                                 // 00000000B0AC: BF9F0000
	s_code_end                                                 // 00000000B0B0: BF9F0000
	s_code_end                                                 // 00000000B0B4: BF9F0000
	s_code_end                                                 // 00000000B0B8: BF9F0000
	s_code_end                                                 // 00000000B0BC: BF9F0000
	s_code_end                                                 // 00000000B0C0: BF9F0000
	s_code_end                                                 // 00000000B0C4: BF9F0000
	s_code_end                                                 // 00000000B0C8: BF9F0000
	s_code_end                                                 // 00000000B0CC: BF9F0000
	s_code_end                                                 // 00000000B0D0: BF9F0000
	s_code_end                                                 // 00000000B0D4: BF9F0000
	s_code_end                                                 // 00000000B0D8: BF9F0000
	s_code_end                                                 // 00000000B0DC: BF9F0000
	s_code_end                                                 // 00000000B0E0: BF9F0000
	s_code_end                                                 // 00000000B0E4: BF9F0000
	s_code_end                                                 // 00000000B0E8: BF9F0000
	s_code_end                                                 // 00000000B0EC: BF9F0000
	s_code_end                                                 // 00000000B0F0: BF9F0000
	s_code_end                                                 // 00000000B0F4: BF9F0000
	s_code_end                                                 // 00000000B0F8: BF9F0000
	s_code_end                                                 // 00000000B0FC: BF9F0000
	s_code_end                                                 // 00000000B100: BF9F0000
	s_code_end                                                 // 00000000B104: BF9F0000
	s_code_end                                                 // 00000000B108: BF9F0000
	s_code_end                                                 // 00000000B10C: BF9F0000
	s_code_end                                                 // 00000000B110: BF9F0000
	s_code_end                                                 // 00000000B114: BF9F0000
	s_code_end                                                 // 00000000B118: BF9F0000
	s_code_end                                                 // 00000000B11C: BF9F0000
	s_code_end                                                 // 00000000B120: BF9F0000
	s_code_end                                                 // 00000000B124: BF9F0000
	s_code_end                                                 // 00000000B128: BF9F0000
	s_code_end                                                 // 00000000B12C: BF9F0000
	s_code_end                                                 // 00000000B130: BF9F0000
	s_code_end                                                 // 00000000B134: BF9F0000
	s_code_end                                                 // 00000000B138: BF9F0000
	s_code_end                                                 // 00000000B13C: BF9F0000
	s_code_end                                                 // 00000000B140: BF9F0000
	s_code_end                                                 // 00000000B144: BF9F0000
	s_code_end                                                 // 00000000B148: BF9F0000
	s_code_end                                                 // 00000000B14C: BF9F0000
	s_code_end                                                 // 00000000B150: BF9F0000
	s_code_end                                                 // 00000000B154: BF9F0000
	s_code_end                                                 // 00000000B158: BF9F0000
	s_code_end                                                 // 00000000B15C: BF9F0000
	s_code_end                                                 // 00000000B160: BF9F0000
	s_code_end                                                 // 00000000B164: BF9F0000
	s_code_end                                                 // 00000000B168: BF9F0000
	s_code_end                                                 // 00000000B16C: BF9F0000
	s_code_end                                                 // 00000000B170: BF9F0000
	s_code_end                                                 // 00000000B174: BF9F0000
	s_code_end                                                 // 00000000B178: BF9F0000
	s_code_end                                                 // 00000000B17C: BF9F0000
	s_code_end                                                 // 00000000B180: BF9F0000
	s_code_end                                                 // 00000000B184: BF9F0000
	s_code_end                                                 // 00000000B188: BF9F0000
	s_code_end                                                 // 00000000B18C: BF9F0000
	s_code_end                                                 // 00000000B190: BF9F0000
	s_code_end                                                 // 00000000B194: BF9F0000
	s_code_end                                                 // 00000000B198: BF9F0000
	s_code_end                                                 // 00000000B19C: BF9F0000
	s_code_end                                                 // 00000000B1A0: BF9F0000
	s_code_end                                                 // 00000000B1A4: BF9F0000
	s_code_end                                                 // 00000000B1A8: BF9F0000
	s_code_end                                                 // 00000000B1AC: BF9F0000
	s_code_end                                                 // 00000000B1B0: BF9F0000
	s_code_end                                                 // 00000000B1B4: BF9F0000
	s_code_end                                                 // 00000000B1B8: BF9F0000
	s_code_end                                                 // 00000000B1BC: BF9F0000
	s_code_end                                                 // 00000000B1C0: BF9F0000
	s_code_end                                                 // 00000000B1C4: BF9F0000
	s_code_end                                                 // 00000000B1C8: BF9F0000
	s_code_end                                                 // 00000000B1CC: BF9F0000
	s_code_end                                                 // 00000000B1D0: BF9F0000
	s_code_end                                                 // 00000000B1D4: BF9F0000
	s_code_end                                                 // 00000000B1D8: BF9F0000
	s_code_end                                                 // 00000000B1DC: BF9F0000
	s_code_end                                                 // 00000000B1E0: BF9F0000
	s_code_end                                                 // 00000000B1E4: BF9F0000
	s_code_end                                                 // 00000000B1E8: BF9F0000
	s_code_end                                                 // 00000000B1EC: BF9F0000
	s_code_end                                                 // 00000000B1F0: BF9F0000
	s_code_end                                                 // 00000000B1F4: BF9F0000
	s_code_end                                                 // 00000000B1F8: BF9F0000
	s_code_end                                                 // 00000000B1FC: BF9F0000
