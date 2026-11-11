
kernels/qwen4_moe_gu_few_pm_gfx1201.co:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000001900 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch>:
	s_load_b256 s[8:15], s[0:1], 0x0                           // 000000001900: F4006200 F8000000
	s_load_b64 s[16:17], s[0:1], 0x20                          // 000000001908: F4002400 F8000020
	s_load_b128 s[20:23], s[0:1], 0x28                         // 000000001910: F4004500 F8000028
	s_load_b32 s24, s[0:1], 0x38                               // 000000001918: F4000600 F8000038
	s_mov_b32 s34, 0                                           // 000000001920: BEA20080
	s_mov_b32 s35, 0                                           // 000000001924: BEA30080
	s_mov_b32 s27, ttmp9                                       // 000000001928: BE9B0075
	s_mul_i32 s28, ttmp7, 4                                    // 00000000192C: 961C8473
	s_wait_kmcnt 0x0                                           // 000000001930: BFC70000
	s_cmp_ge_i32 s28, s23                                      // 000000001934: BF03171C
	s_cbranch_scc1 5784                                        // 000000001938: BFA21698 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x5a9c>
	s_add_co_i32 s31, s23, -1                                  // 00000000193C: 811FC117
	s_add_co_i32 s88, s28, 0                                   // 000000001940: 8158801C
	s_min_u32 s88, s88, s31                                    // 000000001944: 89D81F58
	s_lshl_b32 s34, s88, 2                                     // 000000001948: 84228258
	s_add_nc_u64 s[90:91], s[12:13], s[34:35]                  // 00000000194C: A9DA220C
	s_load_b32 s40, s[90:91], 0x0                              // 000000001950: F4000A2D F8000000
	s_add_co_i32 s88, s28, 1                                   // 000000001958: 8158811C
	s_min_u32 s88, s88, s31                                    // 00000000195C: 89D81F58
	s_lshl_b32 s34, s88, 2                                     // 000000001960: 84228258
	s_add_nc_u64 s[90:91], s[12:13], s[34:35]                  // 000000001964: A9DA220C
	s_load_b32 s41, s[90:91], 0x0                              // 000000001968: F4000A6D F8000000
	s_add_co_i32 s88, s28, 2                                   // 000000001970: 8158821C
	s_min_u32 s88, s88, s31                                    // 000000001974: 89D81F58
	s_lshl_b32 s34, s88, 2                                     // 000000001978: 84228258
	s_add_nc_u64 s[90:91], s[12:13], s[34:35]                  // 00000000197C: A9DA220C
	s_load_b32 s42, s[90:91], 0x0                              // 000000001980: F4000AAD F8000000
	s_add_co_i32 s88, s28, 3                                   // 000000001988: 8158831C
	s_min_u32 s88, s88, s31                                    // 00000000198C: 89D81F58
	s_lshl_b32 s34, s88, 2                                     // 000000001990: 84228258
	s_add_nc_u64 s[90:91], s[12:13], s[34:35]                  // 000000001994: A9DA220C
	s_load_b32 s43, s[90:91], 0x0                              // 000000001998: F4000AED F8000000
	s_lshr_b32 s88, s28, 4                                     // 0000000019A0: 8558841C
	s_lshl_b32 s34, s88, 2                                     // 0000000019A4: 84228258
	s_add_nc_u64 s[90:91], s[10:11], s[34:35]                  // 0000000019A8: A9DA220A
	s_load_b32 s32, s[90:91], 0x0                              // 0000000019AC: F400082D F8000000
	s_wait_kmcnt 0x0                                           // 0000000019B4: BFC70000
	s_cmp_lt_i32 s32, 0                                        // 0000000019B8: BF048020
	s_cbranch_scc1 5751                                        // 0000000019BC: BFA21677 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x5a9c>
	s_cmp_ge_i32 s32, s24                                      // 0000000019C0: BF031820
	s_cbranch_scc1 5749                                        // 0000000019C4: BFA21675 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x5a9c>
	s_mov_b32 s29, 0                                           // 0000000019C8: BE9D0080
	s_mov_b32 s30, 0                                           // 0000000019CC: BE9E0080
	s_add_co_i32 s88, s28, 0                                   // 0000000019D0: 8158801C
	s_cmp_lt_i32 s88, s23                                      // 0000000019D4: BF041758
	s_cselect_b32 s89, -1, 0                                   // 0000000019D8: 985980C1
	s_cmp_ge_i32 s40, 0                                        // 0000000019DC: BF038028
	s_cselect_b32 s90, -1, 0                                   // 0000000019E0: 985A80C1
	s_and_b32 s90, s90, s89                                    // 0000000019E4: 8B5A595A
	s_and_b32 s91, s90, 1                                      // 0000000019E8: 8B5B815A
	s_or_b32 s29, s29, s91                                     // 0000000019EC: 8C1D5B1D
	s_add_co_i32 s88, s28, 1                                   // 0000000019F0: 8158811C
	s_cmp_lt_i32 s88, s23                                      // 0000000019F4: BF041758
	s_cselect_b32 s89, -1, 0                                   // 0000000019F8: 985980C1
	s_cmp_ge_i32 s41, 0                                        // 0000000019FC: BF038029
	s_cselect_b32 s90, -1, 0                                   // 000000001A00: 985A80C1
	s_and_b32 s90, s90, s89                                    // 000000001A04: 8B5A595A
	s_and_b32 s91, s90, 2                                      // 000000001A08: 8B5B825A
	s_or_b32 s29, s29, s91                                     // 000000001A0C: 8C1D5B1D
	s_add_co_i32 s88, s28, 2                                   // 000000001A10: 8158821C
	s_cmp_lt_i32 s88, s23                                      // 000000001A14: BF041758
	s_cselect_b32 s89, -1, 0                                   // 000000001A18: 985980C1
	s_cmp_ge_i32 s42, 0                                        // 000000001A1C: BF03802A
	s_cselect_b32 s90, -1, 0                                   // 000000001A20: 985A80C1
	s_and_b32 s90, s90, s89                                    // 000000001A24: 8B5A595A
	s_and_b32 s91, s90, 4                                      // 000000001A28: 8B5B845A
	s_or_b32 s29, s29, s91                                     // 000000001A2C: 8C1D5B1D
	s_add_co_i32 s88, s28, 3                                   // 000000001A30: 8158831C
	s_cmp_lt_i32 s88, s23                                      // 000000001A34: BF041758
	s_cselect_b32 s89, -1, 0                                   // 000000001A38: 985980C1
	s_cmp_ge_i32 s43, 0                                        // 000000001A3C: BF03802B
	s_cselect_b32 s90, -1, 0                                   // 000000001A40: 985A80C1
	s_and_b32 s90, s90, s89                                    // 000000001A44: 8B5A595A
	s_and_b32 s91, s90, 8                                      // 000000001A48: 8B5B885A
	s_or_b32 s29, s29, s91                                     // 000000001A4C: 8C1D5B1D
	s_and_b32 s88, s29, 3                                      // 000000001A50: 8B58831D
	s_cmp_lg_u32 s88, 0                                        // 000000001A54: BF078058
	s_cselect_b32 s88, 1, 0                                    // 000000001A58: 98588081
	s_add_co_i32 s89, s28, 0                                   // 000000001A5C: 8159801C
	s_cmp_lt_i32 s89, s23                                      // 000000001A60: BF041759
	s_cselect_b32 s88, s88, 0                                  // 000000001A64: 98588058
	s_or_b32 s30, s30, s88                                     // 000000001A68: 8C1E581E
	s_and_b32 s88, s29, 3                                      // 000000001A6C: 8B58831D
	s_cmp_lg_u32 s88, 0                                        // 000000001A70: BF078058
	s_cselect_b32 s88, 2, 0                                    // 000000001A74: 98588082
	s_add_co_i32 s89, s28, 1                                   // 000000001A78: 8159811C
	s_cmp_lt_i32 s89, s23                                      // 000000001A7C: BF041759
	s_cselect_b32 s88, s88, 0                                  // 000000001A80: 98588058
	s_or_b32 s30, s30, s88                                     // 000000001A84: 8C1E581E
	s_and_b32 s88, s29, 12                                     // 000000001A88: 8B588C1D
	s_cmp_lg_u32 s88, 0                                        // 000000001A8C: BF078058
	s_cselect_b32 s88, 4, 0                                    // 000000001A90: 98588084
	s_add_co_i32 s89, s28, 2                                   // 000000001A94: 8159821C
	s_cmp_lt_i32 s89, s23                                      // 000000001A98: BF041759
	s_cselect_b32 s88, s88, 0                                  // 000000001A9C: 98588058
	s_or_b32 s30, s30, s88                                     // 000000001AA0: 8C1E581E
	s_and_b32 s88, s29, 12                                     // 000000001AA4: 8B588C1D
	s_cmp_lg_u32 s88, 0                                        // 000000001AA8: BF078058
	s_cselect_b32 s88, 8, 0                                    // 000000001AAC: 98588088
	s_add_co_i32 s89, s28, 3                                   // 000000001AB0: 8159831C
	s_cmp_lt_i32 s89, s23                                      // 000000001AB4: BF041759
	s_cselect_b32 s88, s88, 0                                  // 000000001AB8: 98588058
	s_or_b32 s30, s30, s88                                     // 000000001ABC: 8C1E581E
	s_mov_b32 s63, 0xc1000000                                  // 000000001AC0: BEBF00FF C1000000
	s_mov_b32 s64, 0x76543210                                  // 000000001AC8: BEC000FF 76543210
	v_and_b32_e32 v1, 31, v0                                   // 000000001AD0: 3602009F
	v_and_b32_e32 v2, 16, v0                                   // 000000001AD4: 36040090
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001AD8: BF870002
	v_lshlrev_b32_e32 v3, 2, v1                                // 000000001ADC: 30060282
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001AE0: BF870003
	v_lshlrev_b32_e32 v4, 5, v1                                // 000000001AE4: 30080285
	v_mov_b32_e32 v8, 0x80000000                               // 000000001AE8: 7E1002FF 80000000
	v_mov_b32_e32 v12, 0                                       // 000000001AF0: 7E180280
	v_mov_b32_e32 v13, 0                                       // 000000001AF4: 7E1A0280
	v_mov_b32_e32 v14, 0                                       // 000000001AF8: 7E1C0280
	v_mov_b32_e32 v15, 0                                       // 000000001AFC: 7E1E0280
	v_cmp_eq_u32_e64 s65, 0, v2                                // 000000001B00: D44A0041 02020480
	v_lshrrev_b32_e32 v128, 5, v0                              // 000000001B08: 33000085
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B0C: BF870001
	v_readfirstlane_b32 s26, v128                              // 000000001B10: 7E340580
	s_lshr_b32 s66, s26, 1                                     // 000000001B14: 8542811A
	s_and_b32 s67, s26, 1                                      // 000000001B18: 8B43811A
	s_lshl_b32 s67, s67, 5                                     // 000000001B1C: 84438543
	v_add_nc_u32_e32 v6, s67, v1                               // 000000001B20: 4A0C0243
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B24: BF870001
	v_lshlrev_b32_e32 v6, 4, v6                                // 000000001B28: 300C0C84
	s_mul_i32 s88, s66, 0x410                                  // 000000001B2C: 9658FF42 00000410
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B34: BF870001
	v_add_nc_u32_e32 v5, s88, v6                               // 000000001B38: 4A0A0C58
	v_cmp_eq_u32_e64 s88, 0, v1                                // 000000001B3C: D44A0058 02020280
	s_wait_alu depctr_va_sdst(0)                               // 000000001B44: BF88F19F
	v_cndmask_b32_e64 v7, v8, v12, s88                         // 000000001B48: D5010007 01621908
	s_cvt_f32_u32 s88, s22                                     // 000000001B50: BED86516
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B54: BF88FF9E
	v_s_rcp_f32 s88, s88                                       // 000000001B58: D6840058 02010058
	s_mul_f32 s88, s88, 0x4f7ffffe                             // 000000001B60: A258FF58 4F7FFFFE
	s_sub_co_i32 s89, 0, s22                                   // 000000001B68: 81D91680
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B6C: BF88FF9E
	s_cvt_u32_f32 s88, s88                                     // 000000001B70: BED86758
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B74: BF88FF9E
	s_mul_i32 s89, s89, s88                                    // 000000001B78: 96595859
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B7C: BF88FF9E
	s_mul_hi_u32 s90, s88, s89                                 // 000000001B80: 96DA5958
	s_add_co_i32 s62, s88, s90                                 // 000000001B84: 813E5A58
	s_lshl_b32 s88, s32, 3                                     // 000000001B88: 84588320
	s_mov_b32 s89, 0                                           // 000000001B8C: BED90080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B90: BF88FF9E
	s_add_nc_u64 s[90:91], s[8:9], s[88:89]                    // 000000001B94: A9DA5808
	s_load_b64 s[60:61], s[90:91], 0x0                         // 000000001B98: F4002F2D F8000000
	s_mov_b32 s52, s14                                         // 000000001BA0: BEB4000E
	s_and_b32 s53, s15, 0xffff                                 // 000000001BA4: 8B35FF0F 0000FFFF
	s_mov_b32 s54, 0x7fffffff                                  // 000000001BAC: BEB600FF 7FFFFFFF
	s_mov_b32 s55, 0x31004000                                  // 000000001BB4: BEB700FF 31004000
	s_mov_b32 s56, s16                                         // 000000001BBC: BEB80010
	s_and_b32 s57, s17, 0xffff                                 // 000000001BC0: 8B39FF11 0000FFFF
	s_mov_b32 s58, 0x7fffffff                                  // 000000001BC8: BEBA00FF 7FFFFFFF
	s_mov_b32 s59, 0x31004000                                  // 000000001BD0: BEBB00FF 31004000
	s_lshl_b32 s88, s27, 2                                     // 000000001BD8: 8458821B
	s_wait_alu depctr_sa_sdst(0)                               // 000000001BDC: BF88FF9E
	s_add_co_i32 s88, s88, s26                                 // 000000001BE0: 81581A58
	s_wait_alu depctr_sa_sdst(0)                               // 000000001BE4: BF88FF9E
	s_lshl_b32 s68, s88, 2                                     // 000000001BE8: 84448258
	s_add_co_i32 s88, s20, -1                                  // 000000001BEC: 8158C114
	s_mul_i32 s89, s20, 0x500                                  // 000000001BF0: 9659FF14 00000500
	s_add_co_i32 s90, s68, 0                                   // 000000001BF8: 815A8044
	s_cmp_lt_i32 s90, s20                                      // 000000001BFC: BF04145A
	s_cselect_b32 s77, 0, 0x80000000                           // 000000001C00: 984DFF80 80000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C08: BF88FF9E
	s_min_i32 s91, s90, s88                                    // 000000001C0C: 895B585A
	s_mul_i32 s69, s91, 0x500                                  // 000000001C10: 9645FF5B 00000500
	s_mul_i32 s73, s91, 40                                     // 000000001C18: 9649A85B
	s_add_co_i32 s73, s73, s89                                 // 000000001C1C: 81495949
	s_add_co_i32 s90, s68, 1                                   // 000000001C20: 815A8144
	s_cmp_lt_i32 s90, s20                                      // 000000001C24: BF04145A
	s_cselect_b32 s78, 0, 0x80000000                           // 000000001C28: 984EFF80 80000000
	s_min_i32 s91, s90, s88                                    // 000000001C30: 895B585A
	s_mul_i32 s70, s91, 0x500                                  // 000000001C34: 9646FF5B 00000500
	s_mul_i32 s74, s91, 40                                     // 000000001C3C: 964AA85B
	s_add_co_i32 s74, s74, s89                                 // 000000001C40: 814A594A
	s_add_co_i32 s90, s68, 2                                   // 000000001C44: 815A8244
	s_cmp_lt_i32 s90, s20                                      // 000000001C48: BF04145A
	s_cselect_b32 s79, 0, 0x80000000                           // 000000001C4C: 984FFF80 80000000
	s_min_i32 s91, s90, s88                                    // 000000001C54: 895B585A
	s_mul_i32 s71, s91, 0x500                                  // 000000001C58: 9647FF5B 00000500
	s_mul_i32 s75, s91, 40                                     // 000000001C60: 964BA85B
	s_add_co_i32 s75, s75, s89                                 // 000000001C64: 814B594B
	s_add_co_i32 s90, s68, 3                                   // 000000001C68: 815A8344
	s_cmp_lt_i32 s90, s20                                      // 000000001C6C: BF04145A
	s_cselect_b32 s80, 0, 0x80000000                           // 000000001C70: 9850FF80 80000000
	s_min_i32 s91, s90, s88                                    // 000000001C78: 895B585A
	s_mul_i32 s72, s91, 0x500                                  // 000000001C7C: 9648FF5B 00000500
	s_mul_i32 s76, s91, 40                                     // 000000001C84: 964CA85B
	s_add_co_i32 s76, s76, s89                                 // 000000001C88: 814C594C
	v_mov_b32_e32 v32, 0                                       // 000000001C8C: 7E400280
	v_mov_b32_e32 v33, 0                                       // 000000001C90: 7E420280
	v_mov_b32_e32 v34, 0                                       // 000000001C94: 7E440280
	v_mov_b32_e32 v35, 0                                       // 000000001C98: 7E460280
	v_mov_b32_e32 v36, 0                                       // 000000001C9C: 7E480280
	v_mov_b32_e32 v37, 0                                       // 000000001CA0: 7E4A0280
	v_mov_b32_e32 v38, 0                                       // 000000001CA4: 7E4C0280
	v_mov_b32_e32 v39, 0                                       // 000000001CA8: 7E4E0280
	v_mov_b32_e32 v40, 0                                       // 000000001CAC: 7E500280
	v_mov_b32_e32 v41, 0                                       // 000000001CB0: 7E520280
	v_mov_b32_e32 v42, 0                                       // 000000001CB4: 7E540280
	v_mov_b32_e32 v43, 0                                       // 000000001CB8: 7E560280
	v_mov_b32_e32 v44, 0                                       // 000000001CBC: 7E580280
	v_mov_b32_e32 v45, 0                                       // 000000001CC0: 7E5A0280
	v_mov_b32_e32 v46, 0                                       // 000000001CC4: 7E5C0280
	v_mov_b32_e32 v47, 0                                       // 000000001CC8: 7E5E0280
	v_mov_b32_e32 v48, 0                                       // 000000001CCC: 7E600280
	v_mov_b32_e32 v49, 0                                       // 000000001CD0: 7E620280
	v_mov_b32_e32 v50, 0                                       // 000000001CD4: 7E640280
	v_mov_b32_e32 v51, 0                                       // 000000001CD8: 7E660280
	v_mov_b32_e32 v52, 0                                       // 000000001CDC: 7E680280
	v_mov_b32_e32 v53, 0                                       // 000000001CE0: 7E6A0280
	v_mov_b32_e32 v54, 0                                       // 000000001CE4: 7E6C0280
	v_mov_b32_e32 v55, 0                                       // 000000001CE8: 7E6E0280
	v_mov_b32_e32 v56, 0                                       // 000000001CEC: 7E700280
	v_mov_b32_e32 v57, 0                                       // 000000001CF0: 7E720280
	v_mov_b32_e32 v58, 0                                       // 000000001CF4: 7E740280
	v_mov_b32_e32 v59, 0                                       // 000000001CF8: 7E760280
	v_mov_b32_e32 v60, 0                                       // 000000001CFC: 7E780280
	v_mov_b32_e32 v61, 0                                       // 000000001D00: 7E7A0280
	v_mov_b32_e32 v62, 0                                       // 000000001D04: 7E7C0280
	v_mov_b32_e32 v63, 0                                       // 000000001D08: 7E7E0280
	v_mov_b32_e32 v64, 0                                       // 000000001D0C: 7E800280
	v_mov_b32_e32 v65, 0                                       // 000000001D10: 7E820280
	v_mov_b32_e32 v66, 0                                       // 000000001D14: 7E840280
	v_mov_b32_e32 v67, 0                                       // 000000001D18: 7E860280
	v_mov_b32_e32 v68, 0                                       // 000000001D1C: 7E880280
	v_mov_b32_e32 v69, 0                                       // 000000001D20: 7E8A0280
	v_mov_b32_e32 v70, 0                                       // 000000001D24: 7E8C0280
	v_mov_b32_e32 v71, 0                                       // 000000001D28: 7E8E0280
	v_mov_b32_e32 v72, 0                                       // 000000001D2C: 7E900280
	v_mov_b32_e32 v73, 0                                       // 000000001D30: 7E920280
	v_mov_b32_e32 v74, 0                                       // 000000001D34: 7E940280
	v_mov_b32_e32 v75, 0                                       // 000000001D38: 7E960280
	v_mov_b32_e32 v76, 0                                       // 000000001D3C: 7E980280
	v_mov_b32_e32 v77, 0                                       // 000000001D40: 7E9A0280
	v_mov_b32_e32 v78, 0                                       // 000000001D44: 7E9C0280
	v_mov_b32_e32 v79, 0                                       // 000000001D48: 7E9E0280
	v_mov_b32_e32 v80, 0                                       // 000000001D4C: 7EA00280
	v_mov_b32_e32 v81, 0                                       // 000000001D50: 7EA20280
	v_mov_b32_e32 v82, 0                                       // 000000001D54: 7EA40280
	v_mov_b32_e32 v83, 0                                       // 000000001D58: 7EA60280
	v_mov_b32_e32 v84, 0                                       // 000000001D5C: 7EA80280
	v_mov_b32_e32 v85, 0                                       // 000000001D60: 7EAA0280
	v_mov_b32_e32 v86, 0                                       // 000000001D64: 7EAC0280
	v_mov_b32_e32 v87, 0                                       // 000000001D68: 7EAE0280
	v_mov_b32_e32 v88, 0                                       // 000000001D6C: 7EB00280
	v_mov_b32_e32 v89, 0                                       // 000000001D70: 7EB20280
	v_mov_b32_e32 v90, 0                                       // 000000001D74: 7EB40280
	v_mov_b32_e32 v91, 0                                       // 000000001D78: 7EB60280
	v_mov_b32_e32 v92, 0                                       // 000000001D7C: 7EB80280
	v_mov_b32_e32 v93, 0                                       // 000000001D80: 7EBA0280
	v_mov_b32_e32 v94, 0                                       // 000000001D84: 7EBC0280
	v_mov_b32_e32 v95, 0                                       // 000000001D88: 7EBE0280
	v_mov_b32_e32 v128, 0                                      // 000000001D8C: 7F000280
	v_mov_b32_e32 v129, 0                                      // 000000001D90: 7F020280
	v_mov_b32_e32 v130, 0                                      // 000000001D94: 7F040280
	v_mov_b32_e32 v131, 0                                      // 000000001D98: 7F060280
	s_wait_kmcnt 0x0                                           // 000000001D9C: BFC70000
	s_mov_b32 s48, s60                                         // 000000001DA0: BEB0003C
	s_and_b32 s49, s61, 0xffff                                 // 000000001DA4: 8B31FF3D 0000FFFF
	s_mov_b32 s50, 0x7fffffff                                  // 000000001DAC: BEB200FF 7FFFFFFF
	s_mov_b32 s51, 0x31004000                                  // 000000001DB4: BEB300FF 31004000
	s_cmp_eq_u32 s29, 0                                        // 000000001DBC: BF06801D
	s_cbranch_scc1 5494                                        // 000000001DC0: BFA21576 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x5a9c>
	s_cmp_eq_u32 s66, 0                                        // 000000001DC4: BF068042
	s_cselect_b32 s89, s40, s41                                // 000000001DC8: 98592928
	s_add_co_i32 s88, s66, 0                                   // 000000001DCC: 81588042
	s_wait_alu depctr_sa_sdst(0)                               // 000000001DD0: BF88FF9E
	s_lshr_b32 s90, s29, s88                                   // 000000001DD4: 855A581D
	s_and_b32 s90, s90, 1                                      // 000000001DD8: 8B5A815A
	s_mul_hi_u32 s91, s89, s62                                 // 000000001DDC: 96DB3E59
	s_mul_i32 s92, s91, s22                                    // 000000001DE0: 965C165B
	s_sub_co_i32 s92, s89, s92                                 // 000000001DE4: 81DC5C59
	s_add_co_i32 s93, s91, 1                                   // 000000001DE8: 815D815B
	s_sub_co_i32 s94, s92, s22                                 // 000000001DEC: 81DE165C
	s_cmp_ge_u32 s92, s22                                      // 000000001DF0: BF09165C
	s_cselect_b32 s91, s93, s91                                // 000000001DF4: 985B5B5D
	s_cselect_b32 s92, s94, s92                                // 000000001DF8: 985C5C5E
	s_add_co_i32 s93, s91, 1                                   // 000000001DFC: 815D815B
	s_cmp_ge_u32 s92, s22                                      // 000000001E00: BF09165C
	s_cselect_b32 s91, s93, s91                                // 000000001E04: 985B5B5D
	s_cmp_gt_i32 s22, 1                                        // 000000001E08: BF028116
	s_cselect_b32 s91, s91, s89                                // 000000001E0C: 985B595B
	s_cmp_eq_u32 s90, 0                                        // 000000001E10: BF06805A
	s_cselect_b32 s91, 0, s91                                  // 000000001E14: 985B5B80
	s_mul_i32 s81, s91, 0x2800                                 // 000000001E18: 9651FF5B 00002800
	s_cmp_eq_u32 s66, 0                                        // 000000001E20: BF068042
	s_cselect_b32 s89, s42, s43                                // 000000001E24: 98592B2A
	s_add_co_i32 s88, s66, 2                                   // 000000001E28: 81588242
	s_wait_alu depctr_sa_sdst(0)                               // 000000001E2C: BF88FF9E
	s_lshr_b32 s90, s29, s88                                   // 000000001E30: 855A581D
	s_and_b32 s90, s90, 1                                      // 000000001E34: 8B5A815A
	s_mul_hi_u32 s91, s89, s62                                 // 000000001E38: 96DB3E59
	s_mul_i32 s92, s91, s22                                    // 000000001E3C: 965C165B
	s_sub_co_i32 s92, s89, s92                                 // 000000001E40: 81DC5C59
	s_add_co_i32 s93, s91, 1                                   // 000000001E44: 815D815B
	s_sub_co_i32 s94, s92, s22                                 // 000000001E48: 81DE165C
	s_cmp_ge_u32 s92, s22                                      // 000000001E4C: BF09165C
	s_cselect_b32 s91, s93, s91                                // 000000001E50: 985B5B5D
	s_cselect_b32 s92, s94, s92                                // 000000001E54: 985C5C5E
	s_add_co_i32 s93, s91, 1                                   // 000000001E58: 815D815B
	s_cmp_ge_u32 s92, s22                                      // 000000001E5C: BF09165C
	s_cselect_b32 s91, s93, s91                                // 000000001E60: 985B5B5D
	s_cmp_gt_i32 s22, 1                                        // 000000001E64: BF028116
	s_cselect_b32 s91, s91, s89                                // 000000001E68: 985B595B
	s_cmp_eq_u32 s90, 0                                        // 000000001E6C: BF06805A
	s_cselect_b32 s91, 0, s91                                  // 000000001E70: 985B5B80
	s_mul_i32 s82, s91, 0x2800                                 // 000000001E74: 9652FF5B 00002800
	buffer_load_b128 v[104:107], v6, s[52:55], s81 offen       // 000000001E7C: C405C051 40806868 00000006
	buffer_load_b128 v[108:111], v6, s[52:55], s82 offen       // 000000001E88: C405C052 4080686C 00000006
	buffer_load_b32 v96, off, s[48:51], s73                    // 000000001E94: C4050049 00806060 00000000
	buffer_load_b32 v100, v3, s[48:51], s69 offen              // 000000001EA0: C4050045 40806064 00000003
	buffer_load_b32 v97, off, s[48:51], s74                    // 000000001EAC: C405004A 00806061 00000000
	buffer_load_b32 v101, v3, s[48:51], s70 offen              // 000000001EB8: C4050046 40806065 00000003
	buffer_load_b32 v98, off, s[48:51], s75                    // 000000001EC4: C405004B 00806062 00000000
	buffer_load_b32 v102, v3, s[48:51], s71 offen              // 000000001ED0: C4050047 40806066 00000003
	buffer_load_b32 v99, off, s[48:51], s76                    // 000000001EDC: C405004C 00806063 00000000
	buffer_load_b32 v103, v3, s[48:51], s72 offen              // 000000001EE8: C4050048 40806067 00000003
	s_wait_loadcnt 0x9                                         // 000000001EF4: BFC00009
	ds_store_b128 v5, v[104:107]                               // 000000001EF8: DB7C0000 00006805
	s_wait_loadcnt 0x8                                         // 000000001F00: BFC00008
	ds_store_b128 v5, v[108:111] offset:2080                   // 000000001F04: DB7C0820 00006C05
	s_wait_dscnt 0x0                                           // 000000001F0C: BFC60000
	s_barrier_signal -1                                        // 000000001F10: BE804EC1
	buffer_load_b128 v[120:123], v6, s[52:55], s81 offen offset:1024// 000000001F14: C405C051 40806878 00040006
	buffer_load_b128 v[124:127], v6, s[52:55], s82 offen offset:1024// 000000001F20: C405C052 4080687C 00040006
	buffer_load_b32 v112, off, s[48:51], s73 offset:4          // 000000001F2C: C4050049 00806070 00000400
	buffer_load_b32 v116, v3, s[48:51], s69 offen offset:128   // 000000001F38: C4050045 40806074 00008003
	buffer_load_b32 v113, off, s[48:51], s74 offset:4          // 000000001F44: C405004A 00806071 00000400
	buffer_load_b32 v117, v3, s[48:51], s70 offen offset:128   // 000000001F50: C4050046 40806075 00008003
	buffer_load_b32 v114, off, s[48:51], s75 offset:4          // 000000001F5C: C405004B 00806072 00000400
	buffer_load_b32 v118, v3, s[48:51], s71 offen offset:128   // 000000001F68: C4050047 40806076 00008003
	buffer_load_b32 v115, off, s[48:51], s76 offset:4          // 000000001F74: C405004C 00806073 00000400
	buffer_load_b32 v119, v3, s[48:51], s72 offen offset:128   // 000000001F80: C4050048 40806077 00008003
	s_wait_loadcnt 0x11                                        // 000000001F8C: BFC00011
	v_lshrrev_b32_e32 v28, v2, v96                             // 000000001F90: 3238C102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001F94: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000001F98: CC20001D 8A007F1C
	s_wait_loadcnt 0x10                                        // 000000001FA0: BFC00010
	v_and_b32_e32 v30, 0xf0f0f0f, v100                         // 000000001FA4: 363CC8FF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v100                             // 000000001FAC: 323EC884
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001FB0: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000001FB4: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001FBC: BF870003
	v_cvt_f32_ubyte0_e32 v128, v30                             // 000000001FC0: 7F00231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001FC4: BF870002
	v_cvt_f32_ubyte0_e32 v129, v31                             // 000000001FC8: 7F02231F
	v_cvt_f32_ubyte1_e32 v130, v30                             // 000000001FCC: 7F04251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001FD0: BF870004
	v_cvt_f32_ubyte1_e32 v131, v31                             // 000000001FD4: 7F06251F
	v_cvt_f32_ubyte2_e32 v132, v30                             // 000000001FD8: 7F08271E
	v_cvt_f32_ubyte2_e32 v133, v31                             // 000000001FDC: 7F0A271F
	v_cvt_f32_ubyte3_e32 v134, v30                             // 000000001FE0: 7F0C291E
	v_cvt_f32_ubyte3_e32 v135, v31                             // 000000001FE4: 7F0E291F
	v_fma_mix_f32 v128, v28, v128, v29 op_sel_hi:[1,0,0]       // 000000001FE8: CC200080 0C77011C
	v_fma_mix_f32 v129, v28, v129, v29 op_sel_hi:[1,0,0]       // 000000001FF0: CC200081 0C77031C
	v_fma_mix_f32 v130, v28, v130, v29 op_sel_hi:[1,0,0]       // 000000001FF8: CC200082 0C77051C
	v_fma_mix_f32 v131, v28, v131, v29 op_sel_hi:[1,0,0]       // 000000002000: CC200083 0C77071C
	v_fma_mix_f32 v132, v28, v132, v29 op_sel_hi:[1,0,0]       // 000000002008: CC200084 0C77091C
	v_fma_mix_f32 v133, v28, v133, v29 op_sel_hi:[1,0,0]       // 000000002010: CC200085 0C770B1C
	v_fma_mix_f32 v134, v28, v134, v29 op_sel_hi:[1,0,0]       // 000000002018: CC200086 0C770D1C
	v_fma_mix_f32 v135, v28, v135, v29 op_sel_hi:[1,0,0]       // 000000002020: CC200087 0C770F1C
	s_wait_loadcnt 0xf                                         // 000000002028: BFC0000F
	v_lshrrev_b32_e32 v28, v2, v97                             // 00000000202C: 3238C302
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002030: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000002034: CC20001D 8A007F1C
	s_wait_loadcnt 0xe                                         // 00000000203C: BFC0000E
	v_and_b32_e32 v30, 0xf0f0f0f, v101                         // 000000002040: 363CCAFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v101                             // 000000002048: 323ECA84
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000204C: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000002050: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000002058: BF870003
	v_cvt_f32_ubyte0_e32 v136, v30                             // 00000000205C: 7F10231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002060: BF870002
	v_cvt_f32_ubyte0_e32 v137, v31                             // 000000002064: 7F12231F
	v_cvt_f32_ubyte1_e32 v138, v30                             // 000000002068: 7F14251E
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000206C: BF870004
	v_cvt_f32_ubyte1_e32 v139, v31                             // 000000002070: 7F16251F
	v_cvt_f32_ubyte2_e32 v140, v30                             // 000000002074: 7F18271E
	v_cvt_f32_ubyte2_e32 v141, v31                             // 000000002078: 7F1A271F
	v_cvt_f32_ubyte3_e32 v142, v30                             // 00000000207C: 7F1C291E
	v_cvt_f32_ubyte3_e32 v143, v31                             // 000000002080: 7F1E291F
	v_fma_mix_f32 v136, v28, v136, v29 op_sel_hi:[1,0,0]       // 000000002084: CC200088 0C77111C
	v_fma_mix_f32 v137, v28, v137, v29 op_sel_hi:[1,0,0]       // 00000000208C: CC200089 0C77131C
	v_fma_mix_f32 v138, v28, v138, v29 op_sel_hi:[1,0,0]       // 000000002094: CC20008A 0C77151C
	v_fma_mix_f32 v139, v28, v139, v29 op_sel_hi:[1,0,0]       // 00000000209C: CC20008B 0C77171C
	v_fma_mix_f32 v140, v28, v140, v29 op_sel_hi:[1,0,0]       // 0000000020A4: CC20008C 0C77191C
	v_fma_mix_f32 v141, v28, v141, v29 op_sel_hi:[1,0,0]       // 0000000020AC: CC20008D 0C771B1C
	v_fma_mix_f32 v142, v28, v142, v29 op_sel_hi:[1,0,0]       // 0000000020B4: CC20008E 0C771D1C
	v_fma_mix_f32 v143, v28, v143, v29 op_sel_hi:[1,0,0]       // 0000000020BC: CC20008F 0C771F1C
	s_wait_loadcnt 0xd                                         // 0000000020C4: BFC0000D
	v_lshrrev_b32_e32 v28, v2, v98                             // 0000000020C8: 3238C502
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000020CC: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 0000000020D0: CC20001D 8A007F1C
	s_wait_loadcnt 0xc                                         // 0000000020D8: BFC0000C
	v_and_b32_e32 v30, 0xf0f0f0f, v102                         // 0000000020DC: 363CCCFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v102                             // 0000000020E4: 323ECC84
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000020E8: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 0000000020EC: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000020F4: BF870003
	v_cvt_f32_ubyte0_e32 v144, v30                             // 0000000020F8: 7F20231E
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000020FC: BF870002
	v_cvt_f32_ubyte0_e32 v145, v31                             // 000000002100: 7F22231F
	v_cvt_f32_ubyte1_e32 v146, v30                             // 000000002104: 7F24251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002108: BF870004
	v_cvt_f32_ubyte1_e32 v147, v31                             // 00000000210C: 7F26251F
	v_cvt_f32_ubyte2_e32 v148, v30                             // 000000002110: 7F28271E
	v_cvt_f32_ubyte2_e32 v149, v31                             // 000000002114: 7F2A271F
	v_cvt_f32_ubyte3_e32 v150, v30                             // 000000002118: 7F2C291E
	v_cvt_f32_ubyte3_e32 v151, v31                             // 00000000211C: 7F2E291F
	v_fma_mix_f32 v144, v28, v144, v29 op_sel_hi:[1,0,0]       // 000000002120: CC200090 0C77211C
	v_fma_mix_f32 v145, v28, v145, v29 op_sel_hi:[1,0,0]       // 000000002128: CC200091 0C77231C
	v_fma_mix_f32 v146, v28, v146, v29 op_sel_hi:[1,0,0]       // 000000002130: CC200092 0C77251C
	v_fma_mix_f32 v147, v28, v147, v29 op_sel_hi:[1,0,0]       // 000000002138: CC200093 0C77271C
	v_fma_mix_f32 v148, v28, v148, v29 op_sel_hi:[1,0,0]       // 000000002140: CC200094 0C77291C
	v_fma_mix_f32 v149, v28, v149, v29 op_sel_hi:[1,0,0]       // 000000002148: CC200095 0C772B1C
	v_fma_mix_f32 v150, v28, v150, v29 op_sel_hi:[1,0,0]       // 000000002150: CC200096 0C772D1C
	v_fma_mix_f32 v151, v28, v151, v29 op_sel_hi:[1,0,0]       // 000000002158: CC200097 0C772F1C
	s_wait_loadcnt 0xb                                         // 000000002160: BFC0000B
	v_lshrrev_b32_e32 v28, v2, v99                             // 000000002164: 3238C702
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002168: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 00000000216C: CC20001D 8A007F1C
	s_wait_loadcnt 0xa                                         // 000000002174: BFC0000A
	v_and_b32_e32 v30, 0xf0f0f0f, v103                         // 000000002178: 363CCEFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v103                             // 000000002180: 323ECE84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002184: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000002188: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000002190: BF870003
	v_cvt_f32_ubyte0_e32 v152, v30                             // 000000002194: 7F30231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002198: BF870002
	v_cvt_f32_ubyte0_e32 v153, v31                             // 00000000219C: 7F32231F
	v_cvt_f32_ubyte1_e32 v154, v30                             // 0000000021A0: 7F34251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000021A4: BF870004
	v_cvt_f32_ubyte1_e32 v155, v31                             // 0000000021A8: 7F36251F
	v_cvt_f32_ubyte2_e32 v156, v30                             // 0000000021AC: 7F38271E
	v_cvt_f32_ubyte2_e32 v157, v31                             // 0000000021B0: 7F3A271F
	v_cvt_f32_ubyte3_e32 v158, v30                             // 0000000021B4: 7F3C291E
	v_cvt_f32_ubyte3_e32 v159, v31                             // 0000000021B8: 7F3E291F
	v_fma_mix_f32 v152, v28, v152, v29 op_sel_hi:[1,0,0]       // 0000000021BC: CC200098 0C77311C
	v_fma_mix_f32 v153, v28, v153, v29 op_sel_hi:[1,0,0]       // 0000000021C4: CC200099 0C77331C
	v_fma_mix_f32 v154, v28, v154, v29 op_sel_hi:[1,0,0]       // 0000000021CC: CC20009A 0C77351C
	v_fma_mix_f32 v155, v28, v155, v29 op_sel_hi:[1,0,0]       // 0000000021D4: CC20009B 0C77371C
	v_fma_mix_f32 v156, v28, v156, v29 op_sel_hi:[1,0,0]       // 0000000021DC: CC20009C 0C77391C
	v_fma_mix_f32 v157, v28, v157, v29 op_sel_hi:[1,0,0]       // 0000000021E4: CC20009D 0C773B1C
	v_fma_mix_f32 v158, v28, v158, v29 op_sel_hi:[1,0,0]       // 0000000021EC: CC20009E 0C773D1C
	v_fma_mix_f32 v159, v28, v159, v29 op_sel_hi:[1,0,0]       // 0000000021F4: CC20009F 0C773F1C
	s_barrier_wait 0xffff                                      // 0000000021FC: BF94FFFF
	s_bitcmp0_b32 s29, 0                                       // 000000002200: BF0C801D
	s_cbranch_scc1 74                                          // 000000002204: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0xa30>
	ds_load_b128 v[16:19], v4                                  // 000000002208: DBFC0000 10000004
	ds_load_b128 v[20:23], v4 offset:16                        // 000000002210: DBFC0010 14000004
	s_wait_dscnt 0x1                                           // 000000002218: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 00000000221C: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000002220: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000002224: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000002228: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000222C: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000002230: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002234: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000002238: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000223C: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000002240: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002244: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000002248: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000224C: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000002250: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002254: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000002258: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000225C: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000002260: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002264: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000002268: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000226C: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000002270: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002274: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000002278: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000227C: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000002280: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002284: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000002288: 5636279B
	s_wait_dscnt 0x0                                           // 00000000228C: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002290: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000002294: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002298: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 00000000229C: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022A0: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000022A4: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022A8: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000022AC: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022B0: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000022B4: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022B8: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000022BC: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022C0: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000022C4: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022C8: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000022CC: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022D0: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000022D4: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022D8: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 0000000022DC: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022E0: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 0000000022E4: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022E8: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 0000000022EC: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022F0: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 0000000022F4: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000022F8: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 0000000022FC: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002300: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000002304: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002308: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 00000000230C: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002310: BF870004
	v_add_f32_e32 v32, 0, v24                                  // 000000002314: 06403080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002318: BF870004
	v_add_f32_e32 v36, 0, v25                                  // 00000000231C: 06483280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002320: BF870004
	v_add_f32_e32 v40, 0, v26                                  // 000000002324: 06503480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002328: BF870004
	v_add_f32_e32 v44, 0, v27                                  // 00000000232C: 06583680
	s_bitcmp0_b32 s29, 1                                       // 000000002330: BF0C811D
	s_cbranch_scc1 74                                          // 000000002334: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0xb60>
	ds_load_b128 v[16:19], v4 offset:1040                      // 000000002338: DBFC0410 10000004
	ds_load_b128 v[20:23], v4 offset:1056                      // 000000002340: DBFC0420 14000004
	s_wait_dscnt 0x1                                           // 000000002348: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 00000000234C: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000002350: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000002354: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000002358: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000235C: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000002360: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002364: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000002368: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000236C: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000002370: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002374: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000002378: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000237C: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000002380: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002384: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000002388: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000238C: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000002390: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002394: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000002398: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000239C: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 0000000023A0: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023A4: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 0000000023A8: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023AC: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 0000000023B0: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023B4: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 0000000023B8: 5636279B
	s_wait_dscnt 0x0                                           // 0000000023BC: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023C0: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 0000000023C4: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023C8: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 0000000023CC: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023D0: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000023D4: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023D8: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000023DC: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023E0: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000023E4: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023E8: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000023EC: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023F0: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000023F4: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000023F8: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000023FC: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002400: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000002404: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002408: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 00000000240C: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002410: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000002414: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002418: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 00000000241C: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002420: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000002424: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002428: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 00000000242C: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002430: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000002434: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002438: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 00000000243C: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002440: BF870004
	v_add_f32_e32 v33, 0, v24                                  // 000000002444: 06423080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002448: BF870004
	v_add_f32_e32 v37, 0, v25                                  // 00000000244C: 064A3280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002450: BF870004
	v_add_f32_e32 v41, 0, v26                                  // 000000002454: 06523480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002458: BF870004
	v_add_f32_e32 v45, 0, v27                                  // 00000000245C: 065A3680
	s_bitcmp0_b32 s29, 2                                       // 000000002460: BF0C821D
	s_cbranch_scc1 74                                          // 000000002464: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0xc90>
	ds_load_b128 v[16:19], v4 offset:2080                      // 000000002468: DBFC0820 10000004
	ds_load_b128 v[20:23], v4 offset:2096                      // 000000002470: DBFC0830 14000004
	s_wait_dscnt 0x1                                           // 000000002478: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 00000000247C: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000002480: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000002484: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000002488: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000248C: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000002490: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002494: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000002498: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000249C: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 0000000024A0: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024A4: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 0000000024A8: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024AC: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 0000000024B0: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024B4: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 0000000024B8: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024BC: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 0000000024C0: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024C4: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 0000000024C8: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024CC: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 0000000024D0: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024D4: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 0000000024D8: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024DC: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 0000000024E0: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024E4: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 0000000024E8: 5636279B
	s_wait_dscnt 0x0                                           // 0000000024EC: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024F0: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 0000000024F4: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000024F8: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 0000000024FC: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002500: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000002504: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002508: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 00000000250C: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002510: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000002514: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002518: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 00000000251C: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002520: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000002524: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002528: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 00000000252C: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002530: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000002534: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002538: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 00000000253C: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002540: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000002544: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002548: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 00000000254C: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002550: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000002554: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002558: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 00000000255C: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002560: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000002564: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002568: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 00000000256C: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002570: BF870004
	v_add_f32_e32 v34, 0, v24                                  // 000000002574: 06443080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002578: BF870004
	v_add_f32_e32 v38, 0, v25                                  // 00000000257C: 064C3280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002580: BF870004
	v_add_f32_e32 v42, 0, v26                                  // 000000002584: 06543480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002588: BF870004
	v_add_f32_e32 v46, 0, v27                                  // 00000000258C: 065C3680
	s_bitcmp0_b32 s29, 3                                       // 000000002590: BF0C831D
	s_cbranch_scc1 74                                          // 000000002594: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0xdc0>
	ds_load_b128 v[16:19], v4 offset:3120                      // 000000002598: DBFC0C30 10000004
	ds_load_b128 v[20:23], v4 offset:3136                      // 0000000025A0: DBFC0C40 14000004
	s_wait_dscnt 0x1                                           // 0000000025A8: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 0000000025AC: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 0000000025B0: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000025B4: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 0000000025B8: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000025BC: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 0000000025C0: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000025C4: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 0000000025C8: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000025CC: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 0000000025D0: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000025D4: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 0000000025D8: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000025DC: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 0000000025E0: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000025E4: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 0000000025E8: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000025EC: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 0000000025F0: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000025F4: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 0000000025F8: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000025FC: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000002600: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002604: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000002608: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000260C: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000002610: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002614: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000002618: 5636279B
	s_wait_dscnt 0x0                                           // 00000000261C: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002620: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000002624: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002628: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 00000000262C: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002630: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000002634: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002638: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 00000000263C: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002640: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000002644: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002648: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 00000000264C: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002650: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000002654: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002658: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 00000000265C: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002660: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000002664: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002668: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 00000000266C: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002670: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000002674: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002678: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 00000000267C: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002680: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000002684: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002688: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 00000000268C: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002690: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000002694: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002698: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 00000000269C: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000026A0: BF870004
	v_add_f32_e32 v35, 0, v24                                  // 0000000026A4: 06463080
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000026A8: BF870004
	v_add_f32_e32 v39, 0, v25                                  // 0000000026AC: 064E3280
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000026B0: BF870004
	v_add_f32_e32 v43, 0, v26                                  // 0000000026B4: 06563480
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000026B8: BF870004
	v_add_f32_e32 v47, 0, v27                                  // 0000000026BC: 065E3680
	s_barrier_signal -1                                        // 0000000026C0: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000026C4: BF94FFFF
	s_wait_loadcnt 0x9                                         // 0000000026C8: BFC00009
	ds_store_b128 v5, v[120:123]                               // 0000000026CC: DB7C0000 00007805
	s_wait_loadcnt 0x8                                         // 0000000026D4: BFC00008
	ds_store_b128 v5, v[124:127] offset:2080                   // 0000000026D8: DB7C0820 00007C05
	s_wait_dscnt 0x0                                           // 0000000026E0: BFC60000
	s_barrier_signal -1                                        // 0000000026E4: BE804EC1
	buffer_load_b128 v[104:107], v6, s[52:55], s81 offen offset:2048// 0000000026E8: C405C051 40806868 00080006
	buffer_load_b128 v[108:111], v6, s[52:55], s82 offen offset:2048// 0000000026F4: C405C052 4080686C 00080006
	buffer_load_b32 v96, off, s[48:51], s73 offset:8           // 000000002700: C4050049 00806060 00000800
	buffer_load_b32 v100, v3, s[48:51], s69 offen offset:256   // 00000000270C: C4050045 40806064 00010003
	buffer_load_b32 v97, off, s[48:51], s74 offset:8           // 000000002718: C405004A 00806061 00000800
	buffer_load_b32 v101, v3, s[48:51], s70 offen offset:256   // 000000002724: C4050046 40806065 00010003
	buffer_load_b32 v98, off, s[48:51], s75 offset:8           // 000000002730: C405004B 00806062 00000800
	buffer_load_b32 v102, v3, s[48:51], s71 offen offset:256   // 00000000273C: C4050047 40806066 00010003
	buffer_load_b32 v99, off, s[48:51], s76 offset:8           // 000000002748: C405004C 00806063 00000800
	buffer_load_b32 v103, v3, s[48:51], s72 offen offset:256   // 000000002754: C4050048 40806067 00010003
	s_wait_loadcnt 0x11                                        // 000000002760: BFC00011
	v_lshrrev_b32_e32 v28, v2, v112                            // 000000002764: 3238E102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002768: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 00000000276C: CC20001D 8A007F1C
	s_wait_loadcnt 0x10                                        // 000000002774: BFC00010
	v_and_b32_e32 v30, 0xf0f0f0f, v116                         // 000000002778: 363CE8FF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v116                             // 000000002780: 323EE884
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002784: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000002788: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000002790: BF870003
	v_cvt_f32_ubyte0_e32 v128, v30                             // 000000002794: 7F00231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002798: BF870002
	v_cvt_f32_ubyte0_e32 v129, v31                             // 00000000279C: 7F02231F
	v_cvt_f32_ubyte1_e32 v130, v30                             // 0000000027A0: 7F04251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000027A4: BF870004
	v_cvt_f32_ubyte1_e32 v131, v31                             // 0000000027A8: 7F06251F
	v_cvt_f32_ubyte2_e32 v132, v30                             // 0000000027AC: 7F08271E
	v_cvt_f32_ubyte2_e32 v133, v31                             // 0000000027B0: 7F0A271F
	v_cvt_f32_ubyte3_e32 v134, v30                             // 0000000027B4: 7F0C291E
	v_cvt_f32_ubyte3_e32 v135, v31                             // 0000000027B8: 7F0E291F
	v_fma_mix_f32 v128, v28, v128, v29 op_sel_hi:[1,0,0]       // 0000000027BC: CC200080 0C77011C
	v_fma_mix_f32 v129, v28, v129, v29 op_sel_hi:[1,0,0]       // 0000000027C4: CC200081 0C77031C
	v_fma_mix_f32 v130, v28, v130, v29 op_sel_hi:[1,0,0]       // 0000000027CC: CC200082 0C77051C
	v_fma_mix_f32 v131, v28, v131, v29 op_sel_hi:[1,0,0]       // 0000000027D4: CC200083 0C77071C
	v_fma_mix_f32 v132, v28, v132, v29 op_sel_hi:[1,0,0]       // 0000000027DC: CC200084 0C77091C
	v_fma_mix_f32 v133, v28, v133, v29 op_sel_hi:[1,0,0]       // 0000000027E4: CC200085 0C770B1C
	v_fma_mix_f32 v134, v28, v134, v29 op_sel_hi:[1,0,0]       // 0000000027EC: CC200086 0C770D1C
	v_fma_mix_f32 v135, v28, v135, v29 op_sel_hi:[1,0,0]       // 0000000027F4: CC200087 0C770F1C
	s_wait_loadcnt 0xf                                         // 0000000027FC: BFC0000F
	v_lshrrev_b32_e32 v28, v2, v113                            // 000000002800: 3238E302
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002804: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000002808: CC20001D 8A007F1C
	s_wait_loadcnt 0xe                                         // 000000002810: BFC0000E
	v_and_b32_e32 v30, 0xf0f0f0f, v117                         // 000000002814: 363CEAFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v117                             // 00000000281C: 323EEA84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002820: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000002824: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000282C: BF870003
	v_cvt_f32_ubyte0_e32 v136, v30                             // 000000002830: 7F10231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002834: BF870002
	v_cvt_f32_ubyte0_e32 v137, v31                             // 000000002838: 7F12231F
	v_cvt_f32_ubyte1_e32 v138, v30                             // 00000000283C: 7F14251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002840: BF870004
	v_cvt_f32_ubyte1_e32 v139, v31                             // 000000002844: 7F16251F
	v_cvt_f32_ubyte2_e32 v140, v30                             // 000000002848: 7F18271E
	v_cvt_f32_ubyte2_e32 v141, v31                             // 00000000284C: 7F1A271F
	v_cvt_f32_ubyte3_e32 v142, v30                             // 000000002850: 7F1C291E
	v_cvt_f32_ubyte3_e32 v143, v31                             // 000000002854: 7F1E291F
	v_fma_mix_f32 v136, v28, v136, v29 op_sel_hi:[1,0,0]       // 000000002858: CC200088 0C77111C
	v_fma_mix_f32 v137, v28, v137, v29 op_sel_hi:[1,0,0]       // 000000002860: CC200089 0C77131C
	v_fma_mix_f32 v138, v28, v138, v29 op_sel_hi:[1,0,0]       // 000000002868: CC20008A 0C77151C
	v_fma_mix_f32 v139, v28, v139, v29 op_sel_hi:[1,0,0]       // 000000002870: CC20008B 0C77171C
	v_fma_mix_f32 v140, v28, v140, v29 op_sel_hi:[1,0,0]       // 000000002878: CC20008C 0C77191C
	v_fma_mix_f32 v141, v28, v141, v29 op_sel_hi:[1,0,0]       // 000000002880: CC20008D 0C771B1C
	v_fma_mix_f32 v142, v28, v142, v29 op_sel_hi:[1,0,0]       // 000000002888: CC20008E 0C771D1C
	v_fma_mix_f32 v143, v28, v143, v29 op_sel_hi:[1,0,0]       // 000000002890: CC20008F 0C771F1C
	s_wait_loadcnt 0xd                                         // 000000002898: BFC0000D
	v_lshrrev_b32_e32 v28, v2, v114                            // 00000000289C: 3238E502
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000028A0: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 0000000028A4: CC20001D 8A007F1C
	s_wait_loadcnt 0xc                                         // 0000000028AC: BFC0000C
	v_and_b32_e32 v30, 0xf0f0f0f, v118                         // 0000000028B0: 363CECFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v118                             // 0000000028B8: 323EEC84
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000028BC: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 0000000028C0: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000028C8: BF870003
	v_cvt_f32_ubyte0_e32 v144, v30                             // 0000000028CC: 7F20231E
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000028D0: BF870002
	v_cvt_f32_ubyte0_e32 v145, v31                             // 0000000028D4: 7F22231F
	v_cvt_f32_ubyte1_e32 v146, v30                             // 0000000028D8: 7F24251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000028DC: BF870004
	v_cvt_f32_ubyte1_e32 v147, v31                             // 0000000028E0: 7F26251F
	v_cvt_f32_ubyte2_e32 v148, v30                             // 0000000028E4: 7F28271E
	v_cvt_f32_ubyte2_e32 v149, v31                             // 0000000028E8: 7F2A271F
	v_cvt_f32_ubyte3_e32 v150, v30                             // 0000000028EC: 7F2C291E
	v_cvt_f32_ubyte3_e32 v151, v31                             // 0000000028F0: 7F2E291F
	v_fma_mix_f32 v144, v28, v144, v29 op_sel_hi:[1,0,0]       // 0000000028F4: CC200090 0C77211C
	v_fma_mix_f32 v145, v28, v145, v29 op_sel_hi:[1,0,0]       // 0000000028FC: CC200091 0C77231C
	v_fma_mix_f32 v146, v28, v146, v29 op_sel_hi:[1,0,0]       // 000000002904: CC200092 0C77251C
	v_fma_mix_f32 v147, v28, v147, v29 op_sel_hi:[1,0,0]       // 00000000290C: CC200093 0C77271C
	v_fma_mix_f32 v148, v28, v148, v29 op_sel_hi:[1,0,0]       // 000000002914: CC200094 0C77291C
	v_fma_mix_f32 v149, v28, v149, v29 op_sel_hi:[1,0,0]       // 00000000291C: CC200095 0C772B1C
	v_fma_mix_f32 v150, v28, v150, v29 op_sel_hi:[1,0,0]       // 000000002924: CC200096 0C772D1C
	v_fma_mix_f32 v151, v28, v151, v29 op_sel_hi:[1,0,0]       // 00000000292C: CC200097 0C772F1C
	s_wait_loadcnt 0xb                                         // 000000002934: BFC0000B
	v_lshrrev_b32_e32 v28, v2, v115                            // 000000002938: 3238E702
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000293C: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000002940: CC20001D 8A007F1C
	s_wait_loadcnt 0xa                                         // 000000002948: BFC0000A
	v_and_b32_e32 v30, 0xf0f0f0f, v119                         // 00000000294C: 363CEEFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v119                             // 000000002954: 323EEE84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002958: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 00000000295C: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000002964: BF870003
	v_cvt_f32_ubyte0_e32 v152, v30                             // 000000002968: 7F30231E
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000296C: BF870002
	v_cvt_f32_ubyte0_e32 v153, v31                             // 000000002970: 7F32231F
	v_cvt_f32_ubyte1_e32 v154, v30                             // 000000002974: 7F34251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002978: BF870004
	v_cvt_f32_ubyte1_e32 v155, v31                             // 00000000297C: 7F36251F
	v_cvt_f32_ubyte2_e32 v156, v30                             // 000000002980: 7F38271E
	v_cvt_f32_ubyte2_e32 v157, v31                             // 000000002984: 7F3A271F
	v_cvt_f32_ubyte3_e32 v158, v30                             // 000000002988: 7F3C291E
	v_cvt_f32_ubyte3_e32 v159, v31                             // 00000000298C: 7F3E291F
	v_fma_mix_f32 v152, v28, v152, v29 op_sel_hi:[1,0,0]       // 000000002990: CC200098 0C77311C
	v_fma_mix_f32 v153, v28, v153, v29 op_sel_hi:[1,0,0]       // 000000002998: CC200099 0C77331C
	v_fma_mix_f32 v154, v28, v154, v29 op_sel_hi:[1,0,0]       // 0000000029A0: CC20009A 0C77351C
	v_fma_mix_f32 v155, v28, v155, v29 op_sel_hi:[1,0,0]       // 0000000029A8: CC20009B 0C77371C
	v_fma_mix_f32 v156, v28, v156, v29 op_sel_hi:[1,0,0]       // 0000000029B0: CC20009C 0C77391C
	v_fma_mix_f32 v157, v28, v157, v29 op_sel_hi:[1,0,0]       // 0000000029B8: CC20009D 0C773B1C
	v_fma_mix_f32 v158, v28, v158, v29 op_sel_hi:[1,0,0]       // 0000000029C0: CC20009E 0C773D1C
	v_fma_mix_f32 v159, v28, v159, v29 op_sel_hi:[1,0,0]       // 0000000029C8: CC20009F 0C773F1C
	s_barrier_wait 0xffff                                      // 0000000029D0: BF94FFFF
	s_bitcmp0_b32 s29, 0                                       // 0000000029D4: BF0C801D
	s_cbranch_scc1 74                                          // 0000000029D8: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x1204>
	ds_load_b128 v[16:19], v4                                  // 0000000029DC: DBFC0000 10000004
	ds_load_b128 v[20:23], v4 offset:16                        // 0000000029E4: DBFC0010 14000004
	s_wait_dscnt 0x1                                           // 0000000029EC: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 0000000029F0: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 0000000029F4: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000029F8: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 0000000029FC: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A00: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000002A04: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A08: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000002A0C: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A10: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000002A14: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A18: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000002A1C: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A20: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000002A24: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A28: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000002A2C: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A30: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000002A34: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A38: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000002A3C: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A40: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000002A44: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A48: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000002A4C: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A50: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000002A54: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A58: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000002A5C: 5636279B
	s_wait_dscnt 0x0                                           // 000000002A60: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A64: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000002A68: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A6C: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000002A70: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A74: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000002A78: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A7C: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000002A80: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A84: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000002A88: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A8C: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000002A90: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A94: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000002A98: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A9C: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000002AA0: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002AA4: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000002AA8: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002AAC: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000002AB0: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002AB4: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000002AB8: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002ABC: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000002AC0: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002AC4: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000002AC8: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002ACC: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000002AD0: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002AD4: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000002AD8: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002ADC: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000002AE0: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002AE4: BF870004
	v_add_f32_e32 v48, 0, v24                                  // 000000002AE8: 06603080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002AEC: BF870004
	v_add_f32_e32 v52, 0, v25                                  // 000000002AF0: 06683280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002AF4: BF870004
	v_add_f32_e32 v56, 0, v26                                  // 000000002AF8: 06703480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002AFC: BF870004
	v_add_f32_e32 v60, 0, v27                                  // 000000002B00: 06783680
	s_bitcmp0_b32 s29, 1                                       // 000000002B04: BF0C811D
	s_cbranch_scc1 74                                          // 000000002B08: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x1334>
	ds_load_b128 v[16:19], v4 offset:1040                      // 000000002B0C: DBFC0410 10000004
	ds_load_b128 v[20:23], v4 offset:1056                      // 000000002B14: DBFC0420 14000004
	s_wait_dscnt 0x1                                           // 000000002B1C: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000002B20: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000002B24: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000002B28: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000002B2C: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B30: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000002B34: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B38: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000002B3C: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B40: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000002B44: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B48: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000002B4C: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B50: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000002B54: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B58: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000002B5C: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B60: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000002B64: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B68: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000002B6C: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B70: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000002B74: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B78: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000002B7C: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B80: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000002B84: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B88: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000002B8C: 5636279B
	s_wait_dscnt 0x0                                           // 000000002B90: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B94: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000002B98: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002B9C: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000002BA0: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BA4: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000002BA8: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BAC: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000002BB0: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BB4: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000002BB8: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BBC: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000002BC0: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BC4: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000002BC8: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BCC: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000002BD0: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BD4: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000002BD8: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BDC: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000002BE0: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BE4: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000002BE8: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BEC: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000002BF0: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BF4: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000002BF8: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BFC: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000002C00: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C04: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000002C08: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C0C: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000002C10: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C14: BF870004
	v_add_f32_e32 v49, 0, v24                                  // 000000002C18: 06623080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C1C: BF870004
	v_add_f32_e32 v53, 0, v25                                  // 000000002C20: 066A3280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C24: BF870004
	v_add_f32_e32 v57, 0, v26                                  // 000000002C28: 06723480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C2C: BF870004
	v_add_f32_e32 v61, 0, v27                                  // 000000002C30: 067A3680
	s_bitcmp0_b32 s29, 2                                       // 000000002C34: BF0C821D
	s_cbranch_scc1 74                                          // 000000002C38: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x1464>
	ds_load_b128 v[16:19], v4 offset:2080                      // 000000002C3C: DBFC0820 10000004
	ds_load_b128 v[20:23], v4 offset:2096                      // 000000002C44: DBFC0830 14000004
	s_wait_dscnt 0x1                                           // 000000002C4C: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000002C50: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000002C54: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000002C58: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000002C5C: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C60: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000002C64: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C68: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000002C6C: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C70: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000002C74: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C78: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000002C7C: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C80: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000002C84: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C88: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000002C8C: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C90: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000002C94: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002C98: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000002C9C: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CA0: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000002CA4: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CA8: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000002CAC: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CB0: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000002CB4: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CB8: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000002CBC: 5636279B
	s_wait_dscnt 0x0                                           // 000000002CC0: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CC4: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000002CC8: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CCC: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000002CD0: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CD4: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000002CD8: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CDC: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000002CE0: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CE4: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000002CE8: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CEC: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000002CF0: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CF4: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000002CF8: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002CFC: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000002D00: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D04: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000002D08: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D0C: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000002D10: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D14: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000002D18: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D1C: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000002D20: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D24: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000002D28: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D2C: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000002D30: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D34: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000002D38: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D3C: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000002D40: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D44: BF870004
	v_add_f32_e32 v50, 0, v24                                  // 000000002D48: 06643080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D4C: BF870004
	v_add_f32_e32 v54, 0, v25                                  // 000000002D50: 066C3280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D54: BF870004
	v_add_f32_e32 v58, 0, v26                                  // 000000002D58: 06743480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D5C: BF870004
	v_add_f32_e32 v62, 0, v27                                  // 000000002D60: 067C3680
	s_bitcmp0_b32 s29, 3                                       // 000000002D64: BF0C831D
	s_cbranch_scc1 74                                          // 000000002D68: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x1594>
	ds_load_b128 v[16:19], v4 offset:3120                      // 000000002D6C: DBFC0C30 10000004
	ds_load_b128 v[20:23], v4 offset:3136                      // 000000002D74: DBFC0C40 14000004
	s_wait_dscnt 0x1                                           // 000000002D7C: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000002D80: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000002D84: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000002D88: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000002D8C: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D90: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000002D94: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D98: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000002D9C: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DA0: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000002DA4: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DA8: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000002DAC: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DB0: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000002DB4: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DB8: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000002DBC: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DC0: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000002DC4: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DC8: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000002DCC: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DD0: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000002DD4: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DD8: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000002DDC: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DE0: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000002DE4: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DE8: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000002DEC: 5636279B
	s_wait_dscnt 0x0                                           // 000000002DF0: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DF4: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000002DF8: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002DFC: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000002E00: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E04: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000002E08: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E0C: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000002E10: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E14: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000002E18: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E1C: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000002E20: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E24: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000002E28: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E2C: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000002E30: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E34: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000002E38: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E3C: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000002E40: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E44: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000002E48: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E4C: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000002E50: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E54: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000002E58: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E5C: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000002E60: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E64: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000002E68: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E6C: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000002E70: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E74: BF870004
	v_add_f32_e32 v51, 0, v24                                  // 000000002E78: 06663080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E7C: BF870004
	v_add_f32_e32 v55, 0, v25                                  // 000000002E80: 066E3280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E84: BF870004
	v_add_f32_e32 v59, 0, v26                                  // 000000002E88: 06763480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E8C: BF870004
	v_add_f32_e32 v63, 0, v27                                  // 000000002E90: 067E3680
	s_barrier_signal -1                                        // 000000002E94: BE804EC1
	s_barrier_wait 0xffff                                      // 000000002E98: BF94FFFF
	s_wait_loadcnt 0x9                                         // 000000002E9C: BFC00009
	ds_store_b128 v5, v[104:107]                               // 000000002EA0: DB7C0000 00006805
	s_wait_loadcnt 0x8                                         // 000000002EA8: BFC00008
	ds_store_b128 v5, v[108:111] offset:2080                   // 000000002EAC: DB7C0820 00006C05
	s_wait_dscnt 0x0                                           // 000000002EB4: BFC60000
	s_barrier_signal -1                                        // 000000002EB8: BE804EC1
	buffer_load_b128 v[120:123], v6, s[52:55], s81 offen offset:3072// 000000002EBC: C405C051 40806878 000C0006
	buffer_load_b128 v[124:127], v6, s[52:55], s82 offen offset:3072// 000000002EC8: C405C052 4080687C 000C0006
	buffer_load_b32 v112, off, s[48:51], s73 offset:12         // 000000002ED4: C4050049 00806070 00000C00
	buffer_load_b32 v116, v3, s[48:51], s69 offen offset:384   // 000000002EE0: C4050045 40806074 00018003
	buffer_load_b32 v113, off, s[48:51], s74 offset:12         // 000000002EEC: C405004A 00806071 00000C00
	buffer_load_b32 v117, v3, s[48:51], s70 offen offset:384   // 000000002EF8: C4050046 40806075 00018003
	buffer_load_b32 v114, off, s[48:51], s75 offset:12         // 000000002F04: C405004B 00806072 00000C00
	buffer_load_b32 v118, v3, s[48:51], s71 offen offset:384   // 000000002F10: C4050047 40806076 00018003
	buffer_load_b32 v115, off, s[48:51], s76 offset:12         // 000000002F1C: C405004C 00806073 00000C00
	buffer_load_b32 v119, v3, s[48:51], s72 offen offset:384   // 000000002F28: C4050048 40806077 00018003
	s_wait_loadcnt 0x11                                        // 000000002F34: BFC00011
	v_lshrrev_b32_e32 v28, v2, v96                             // 000000002F38: 3238C102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002F3C: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000002F40: CC20001D 8A007F1C
	s_wait_loadcnt 0x10                                        // 000000002F48: BFC00010
	v_and_b32_e32 v30, 0xf0f0f0f, v100                         // 000000002F4C: 363CC8FF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v100                             // 000000002F54: 323EC884
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002F58: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000002F5C: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000002F64: BF870003
	v_cvt_f32_ubyte0_e32 v128, v30                             // 000000002F68: 7F00231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002F6C: BF870002
	v_cvt_f32_ubyte0_e32 v129, v31                             // 000000002F70: 7F02231F
	v_cvt_f32_ubyte1_e32 v130, v30                             // 000000002F74: 7F04251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002F78: BF870004
	v_cvt_f32_ubyte1_e32 v131, v31                             // 000000002F7C: 7F06251F
	v_cvt_f32_ubyte2_e32 v132, v30                             // 000000002F80: 7F08271E
	v_cvt_f32_ubyte2_e32 v133, v31                             // 000000002F84: 7F0A271F
	v_cvt_f32_ubyte3_e32 v134, v30                             // 000000002F88: 7F0C291E
	v_cvt_f32_ubyte3_e32 v135, v31                             // 000000002F8C: 7F0E291F
	v_fma_mix_f32 v128, v28, v128, v29 op_sel_hi:[1,0,0]       // 000000002F90: CC200080 0C77011C
	v_fma_mix_f32 v129, v28, v129, v29 op_sel_hi:[1,0,0]       // 000000002F98: CC200081 0C77031C
	v_fma_mix_f32 v130, v28, v130, v29 op_sel_hi:[1,0,0]       // 000000002FA0: CC200082 0C77051C
	v_fma_mix_f32 v131, v28, v131, v29 op_sel_hi:[1,0,0]       // 000000002FA8: CC200083 0C77071C
	v_fma_mix_f32 v132, v28, v132, v29 op_sel_hi:[1,0,0]       // 000000002FB0: CC200084 0C77091C
	v_fma_mix_f32 v133, v28, v133, v29 op_sel_hi:[1,0,0]       // 000000002FB8: CC200085 0C770B1C
	v_fma_mix_f32 v134, v28, v134, v29 op_sel_hi:[1,0,0]       // 000000002FC0: CC200086 0C770D1C
	v_fma_mix_f32 v135, v28, v135, v29 op_sel_hi:[1,0,0]       // 000000002FC8: CC200087 0C770F1C
	s_wait_loadcnt 0xf                                         // 000000002FD0: BFC0000F
	v_lshrrev_b32_e32 v28, v2, v97                             // 000000002FD4: 3238C302
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002FD8: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000002FDC: CC20001D 8A007F1C
	s_wait_loadcnt 0xe                                         // 000000002FE4: BFC0000E
	v_and_b32_e32 v30, 0xf0f0f0f, v101                         // 000000002FE8: 363CCAFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v101                             // 000000002FF0: 323ECA84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002FF4: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000002FF8: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003000: BF870003
	v_cvt_f32_ubyte0_e32 v136, v30                             // 000000003004: 7F10231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003008: BF870002
	v_cvt_f32_ubyte0_e32 v137, v31                             // 00000000300C: 7F12231F
	v_cvt_f32_ubyte1_e32 v138, v30                             // 000000003010: 7F14251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003014: BF870004
	v_cvt_f32_ubyte1_e32 v139, v31                             // 000000003018: 7F16251F
	v_cvt_f32_ubyte2_e32 v140, v30                             // 00000000301C: 7F18271E
	v_cvt_f32_ubyte2_e32 v141, v31                             // 000000003020: 7F1A271F
	v_cvt_f32_ubyte3_e32 v142, v30                             // 000000003024: 7F1C291E
	v_cvt_f32_ubyte3_e32 v143, v31                             // 000000003028: 7F1E291F
	v_fma_mix_f32 v136, v28, v136, v29 op_sel_hi:[1,0,0]       // 00000000302C: CC200088 0C77111C
	v_fma_mix_f32 v137, v28, v137, v29 op_sel_hi:[1,0,0]       // 000000003034: CC200089 0C77131C
	v_fma_mix_f32 v138, v28, v138, v29 op_sel_hi:[1,0,0]       // 00000000303C: CC20008A 0C77151C
	v_fma_mix_f32 v139, v28, v139, v29 op_sel_hi:[1,0,0]       // 000000003044: CC20008B 0C77171C
	v_fma_mix_f32 v140, v28, v140, v29 op_sel_hi:[1,0,0]       // 00000000304C: CC20008C 0C77191C
	v_fma_mix_f32 v141, v28, v141, v29 op_sel_hi:[1,0,0]       // 000000003054: CC20008D 0C771B1C
	v_fma_mix_f32 v142, v28, v142, v29 op_sel_hi:[1,0,0]       // 00000000305C: CC20008E 0C771D1C
	v_fma_mix_f32 v143, v28, v143, v29 op_sel_hi:[1,0,0]       // 000000003064: CC20008F 0C771F1C
	s_wait_loadcnt 0xd                                         // 00000000306C: BFC0000D
	v_lshrrev_b32_e32 v28, v2, v98                             // 000000003070: 3238C502
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003074: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000003078: CC20001D 8A007F1C
	s_wait_loadcnt 0xc                                         // 000000003080: BFC0000C
	v_and_b32_e32 v30, 0xf0f0f0f, v102                         // 000000003084: 363CCCFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v102                             // 00000000308C: 323ECC84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003090: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000003094: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000309C: BF870003
	v_cvt_f32_ubyte0_e32 v144, v30                             // 0000000030A0: 7F20231E
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000030A4: BF870002
	v_cvt_f32_ubyte0_e32 v145, v31                             // 0000000030A8: 7F22231F
	v_cvt_f32_ubyte1_e32 v146, v30                             // 0000000030AC: 7F24251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000030B0: BF870004
	v_cvt_f32_ubyte1_e32 v147, v31                             // 0000000030B4: 7F26251F
	v_cvt_f32_ubyte2_e32 v148, v30                             // 0000000030B8: 7F28271E
	v_cvt_f32_ubyte2_e32 v149, v31                             // 0000000030BC: 7F2A271F
	v_cvt_f32_ubyte3_e32 v150, v30                             // 0000000030C0: 7F2C291E
	v_cvt_f32_ubyte3_e32 v151, v31                             // 0000000030C4: 7F2E291F
	v_fma_mix_f32 v144, v28, v144, v29 op_sel_hi:[1,0,0]       // 0000000030C8: CC200090 0C77211C
	v_fma_mix_f32 v145, v28, v145, v29 op_sel_hi:[1,0,0]       // 0000000030D0: CC200091 0C77231C
	v_fma_mix_f32 v146, v28, v146, v29 op_sel_hi:[1,0,0]       // 0000000030D8: CC200092 0C77251C
	v_fma_mix_f32 v147, v28, v147, v29 op_sel_hi:[1,0,0]       // 0000000030E0: CC200093 0C77271C
	v_fma_mix_f32 v148, v28, v148, v29 op_sel_hi:[1,0,0]       // 0000000030E8: CC200094 0C77291C
	v_fma_mix_f32 v149, v28, v149, v29 op_sel_hi:[1,0,0]       // 0000000030F0: CC200095 0C772B1C
	v_fma_mix_f32 v150, v28, v150, v29 op_sel_hi:[1,0,0]       // 0000000030F8: CC200096 0C772D1C
	v_fma_mix_f32 v151, v28, v151, v29 op_sel_hi:[1,0,0]       // 000000003100: CC200097 0C772F1C
	s_wait_loadcnt 0xb                                         // 000000003108: BFC0000B
	v_lshrrev_b32_e32 v28, v2, v99                             // 00000000310C: 3238C702
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003110: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000003114: CC20001D 8A007F1C
	s_wait_loadcnt 0xa                                         // 00000000311C: BFC0000A
	v_and_b32_e32 v30, 0xf0f0f0f, v103                         // 000000003120: 363CCEFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v103                             // 000000003128: 323ECE84
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000312C: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000003130: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003138: BF870003
	v_cvt_f32_ubyte0_e32 v152, v30                             // 00000000313C: 7F30231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003140: BF870002
	v_cvt_f32_ubyte0_e32 v153, v31                             // 000000003144: 7F32231F
	v_cvt_f32_ubyte1_e32 v154, v30                             // 000000003148: 7F34251E
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000314C: BF870004
	v_cvt_f32_ubyte1_e32 v155, v31                             // 000000003150: 7F36251F
	v_cvt_f32_ubyte2_e32 v156, v30                             // 000000003154: 7F38271E
	v_cvt_f32_ubyte2_e32 v157, v31                             // 000000003158: 7F3A271F
	v_cvt_f32_ubyte3_e32 v158, v30                             // 00000000315C: 7F3C291E
	v_cvt_f32_ubyte3_e32 v159, v31                             // 000000003160: 7F3E291F
	v_fma_mix_f32 v152, v28, v152, v29 op_sel_hi:[1,0,0]       // 000000003164: CC200098 0C77311C
	v_fma_mix_f32 v153, v28, v153, v29 op_sel_hi:[1,0,0]       // 00000000316C: CC200099 0C77331C
	v_fma_mix_f32 v154, v28, v154, v29 op_sel_hi:[1,0,0]       // 000000003174: CC20009A 0C77351C
	v_fma_mix_f32 v155, v28, v155, v29 op_sel_hi:[1,0,0]       // 00000000317C: CC20009B 0C77371C
	v_fma_mix_f32 v156, v28, v156, v29 op_sel_hi:[1,0,0]       // 000000003184: CC20009C 0C77391C
	v_fma_mix_f32 v157, v28, v157, v29 op_sel_hi:[1,0,0]       // 00000000318C: CC20009D 0C773B1C
	v_fma_mix_f32 v158, v28, v158, v29 op_sel_hi:[1,0,0]       // 000000003194: CC20009E 0C773D1C
	v_fma_mix_f32 v159, v28, v159, v29 op_sel_hi:[1,0,0]       // 00000000319C: CC20009F 0C773F1C
	s_barrier_wait 0xffff                                      // 0000000031A4: BF94FFFF
	s_bitcmp0_b32 s29, 0                                       // 0000000031A8: BF0C801D
	s_cbranch_scc1 74                                          // 0000000031AC: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x19d8>
	ds_load_b128 v[16:19], v4                                  // 0000000031B0: DBFC0000 10000004
	ds_load_b128 v[20:23], v4 offset:16                        // 0000000031B8: DBFC0010 14000004
	s_wait_dscnt 0x1                                           // 0000000031C0: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 0000000031C4: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 0000000031C8: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000031CC: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 0000000031D0: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000031D4: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 0000000031D8: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000031DC: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 0000000031E0: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000031E4: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 0000000031E8: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000031EC: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 0000000031F0: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000031F4: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 0000000031F8: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000031FC: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000003200: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003204: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000003208: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000320C: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000003210: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003214: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000003218: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000321C: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000003220: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003224: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000003228: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000322C: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000003230: 5636279B
	s_wait_dscnt 0x0                                           // 000000003234: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003238: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 00000000323C: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003240: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000003244: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003248: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 00000000324C: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003250: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000003254: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003258: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 00000000325C: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003260: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000003264: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003268: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 00000000326C: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003270: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000003274: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003278: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 00000000327C: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003280: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000003284: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003288: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 00000000328C: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003290: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000003294: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003298: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 00000000329C: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000032A0: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 0000000032A4: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000032A8: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 0000000032AC: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000032B0: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 0000000032B4: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000032B8: BF870004
	v_add_f32_e32 v64, 0, v24                                  // 0000000032BC: 06803080
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000032C0: BF870004
	v_add_f32_e32 v68, 0, v25                                  // 0000000032C4: 06883280
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000032C8: BF870004
	v_add_f32_e32 v72, 0, v26                                  // 0000000032CC: 06903480
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000032D0: BF870004
	v_add_f32_e32 v76, 0, v27                                  // 0000000032D4: 06983680
	s_bitcmp0_b32 s29, 1                                       // 0000000032D8: BF0C811D
	s_cbranch_scc1 74                                          // 0000000032DC: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x1b08>
	ds_load_b128 v[16:19], v4 offset:1040                      // 0000000032E0: DBFC0410 10000004
	ds_load_b128 v[20:23], v4 offset:1056                      // 0000000032E8: DBFC0420 14000004
	s_wait_dscnt 0x1                                           // 0000000032F0: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 0000000032F4: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 0000000032F8: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000032FC: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000003300: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003304: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000003308: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000330C: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000003310: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003314: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000003318: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000331C: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000003320: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003324: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000003328: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000332C: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000003330: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003334: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000003338: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000333C: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000003340: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003344: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000003348: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000334C: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000003350: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003354: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000003358: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000335C: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000003360: 5636279B
	s_wait_dscnt 0x0                                           // 000000003364: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003368: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 00000000336C: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003370: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000003374: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003378: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 00000000337C: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003380: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000003384: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003388: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 00000000338C: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003390: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000003394: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003398: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 00000000339C: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033A0: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000033A4: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033A8: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000033AC: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033B0: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 0000000033B4: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033B8: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 0000000033BC: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033C0: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 0000000033C4: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033C8: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 0000000033CC: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033D0: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 0000000033D4: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033D8: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 0000000033DC: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033E0: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 0000000033E4: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033E8: BF870004
	v_add_f32_e32 v65, 0, v24                                  // 0000000033EC: 06823080
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033F0: BF870004
	v_add_f32_e32 v69, 0, v25                                  // 0000000033F4: 068A3280
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000033F8: BF870004
	v_add_f32_e32 v73, 0, v26                                  // 0000000033FC: 06923480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003400: BF870004
	v_add_f32_e32 v77, 0, v27                                  // 000000003404: 069A3680
	s_bitcmp0_b32 s29, 2                                       // 000000003408: BF0C821D
	s_cbranch_scc1 74                                          // 00000000340C: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x1c38>
	ds_load_b128 v[16:19], v4 offset:2080                      // 000000003410: DBFC0820 10000004
	ds_load_b128 v[20:23], v4 offset:2096                      // 000000003418: DBFC0830 14000004
	s_wait_dscnt 0x1                                           // 000000003420: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000003424: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000003428: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 00000000342C: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000003430: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003434: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000003438: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000343C: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000003440: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003444: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000003448: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000344C: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000003450: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003454: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000003458: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000345C: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000003460: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003464: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000003468: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000346C: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000003470: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003474: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000003478: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000347C: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000003480: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003484: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000003488: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000348C: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000003490: 5636279B
	s_wait_dscnt 0x0                                           // 000000003494: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003498: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 00000000349C: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034A0: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 0000000034A4: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034A8: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000034AC: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034B0: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000034B4: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034B8: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000034BC: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034C0: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000034C4: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034C8: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000034CC: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034D0: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000034D4: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034D8: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000034DC: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034E0: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 0000000034E4: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034E8: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 0000000034EC: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034F0: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 0000000034F4: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000034F8: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 0000000034FC: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003500: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000003504: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003508: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 00000000350C: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003510: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000003514: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003518: BF870004
	v_add_f32_e32 v66, 0, v24                                  // 00000000351C: 06843080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003520: BF870004
	v_add_f32_e32 v70, 0, v25                                  // 000000003524: 068C3280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003528: BF870004
	v_add_f32_e32 v74, 0, v26                                  // 00000000352C: 06943480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003530: BF870004
	v_add_f32_e32 v78, 0, v27                                  // 000000003534: 069C3680
	s_bitcmp0_b32 s29, 3                                       // 000000003538: BF0C831D
	s_cbranch_scc1 74                                          // 00000000353C: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x1d68>
	ds_load_b128 v[16:19], v4 offset:3120                      // 000000003540: DBFC0C30 10000004
	ds_load_b128 v[20:23], v4 offset:3136                      // 000000003548: DBFC0C40 14000004
	s_wait_dscnt 0x1                                           // 000000003550: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000003554: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000003558: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 00000000355C: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000003560: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003564: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000003568: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000356C: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000003570: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003574: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000003578: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000357C: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000003580: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003584: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000003588: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000358C: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000003590: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003594: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000003598: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000359C: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 0000000035A0: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035A4: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 0000000035A8: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035AC: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 0000000035B0: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035B4: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 0000000035B8: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035BC: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 0000000035C0: 5636279B
	s_wait_dscnt 0x0                                           // 0000000035C4: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035C8: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 0000000035CC: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035D0: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 0000000035D4: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035D8: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000035DC: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035E0: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000035E4: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035E8: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000035EC: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035F0: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000035F4: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000035F8: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000035FC: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003600: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000003604: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003608: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 00000000360C: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003610: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000003614: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003618: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 00000000361C: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003620: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000003624: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003628: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 00000000362C: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003630: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000003634: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003638: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 00000000363C: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003640: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000003644: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003648: BF870004
	v_add_f32_e32 v67, 0, v24                                  // 00000000364C: 06863080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003650: BF870004
	v_add_f32_e32 v71, 0, v25                                  // 000000003654: 068E3280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003658: BF870004
	v_add_f32_e32 v75, 0, v26                                  // 00000000365C: 06963480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003660: BF870004
	v_add_f32_e32 v79, 0, v27                                  // 000000003664: 069E3680
	s_barrier_signal -1                                        // 000000003668: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000366C: BF94FFFF
	s_wait_loadcnt 0x9                                         // 000000003670: BFC00009
	ds_store_b128 v5, v[120:123]                               // 000000003674: DB7C0000 00007805
	s_wait_loadcnt 0x8                                         // 00000000367C: BFC00008
	ds_store_b128 v5, v[124:127] offset:2080                   // 000000003680: DB7C0820 00007C05
	s_wait_dscnt 0x0                                           // 000000003688: BFC60000
	s_barrier_signal -1                                        // 00000000368C: BE804EC1
	buffer_load_b128 v[104:107], v6, s[52:55], s81 offen offset:4096// 000000003690: C405C051 40806868 00100006
	buffer_load_b128 v[108:111], v6, s[52:55], s82 offen offset:4096// 00000000369C: C405C052 4080686C 00100006
	buffer_load_b32 v96, off, s[48:51], s73 offset:16          // 0000000036A8: C4050049 00806060 00001000
	buffer_load_b32 v100, v3, s[48:51], s69 offen offset:512   // 0000000036B4: C4050045 40806064 00020003
	buffer_load_b32 v97, off, s[48:51], s74 offset:16          // 0000000036C0: C405004A 00806061 00001000
	buffer_load_b32 v101, v3, s[48:51], s70 offen offset:512   // 0000000036CC: C4050046 40806065 00020003
	buffer_load_b32 v98, off, s[48:51], s75 offset:16          // 0000000036D8: C405004B 00806062 00001000
	buffer_load_b32 v102, v3, s[48:51], s71 offen offset:512   // 0000000036E4: C4050047 40806066 00020003
	buffer_load_b32 v99, off, s[48:51], s76 offset:16          // 0000000036F0: C405004C 00806063 00001000
	buffer_load_b32 v103, v3, s[48:51], s72 offen offset:512   // 0000000036FC: C4050048 40806067 00020003
	s_wait_loadcnt 0x11                                        // 000000003708: BFC00011
	v_lshrrev_b32_e32 v28, v2, v112                            // 00000000370C: 3238E102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003710: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000003714: CC20001D 8A007F1C
	s_wait_loadcnt 0x10                                        // 00000000371C: BFC00010
	v_and_b32_e32 v30, 0xf0f0f0f, v116                         // 000000003720: 363CE8FF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v116                             // 000000003728: 323EE884
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000372C: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000003730: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003738: BF870003
	v_cvt_f32_ubyte0_e32 v128, v30                             // 00000000373C: 7F00231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003740: BF870002
	v_cvt_f32_ubyte0_e32 v129, v31                             // 000000003744: 7F02231F
	v_cvt_f32_ubyte1_e32 v130, v30                             // 000000003748: 7F04251E
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000374C: BF870004
	v_cvt_f32_ubyte1_e32 v131, v31                             // 000000003750: 7F06251F
	v_cvt_f32_ubyte2_e32 v132, v30                             // 000000003754: 7F08271E
	v_cvt_f32_ubyte2_e32 v133, v31                             // 000000003758: 7F0A271F
	v_cvt_f32_ubyte3_e32 v134, v30                             // 00000000375C: 7F0C291E
	v_cvt_f32_ubyte3_e32 v135, v31                             // 000000003760: 7F0E291F
	v_fma_mix_f32 v128, v28, v128, v29 op_sel_hi:[1,0,0]       // 000000003764: CC200080 0C77011C
	v_fma_mix_f32 v129, v28, v129, v29 op_sel_hi:[1,0,0]       // 00000000376C: CC200081 0C77031C
	v_fma_mix_f32 v130, v28, v130, v29 op_sel_hi:[1,0,0]       // 000000003774: CC200082 0C77051C
	v_fma_mix_f32 v131, v28, v131, v29 op_sel_hi:[1,0,0]       // 00000000377C: CC200083 0C77071C
	v_fma_mix_f32 v132, v28, v132, v29 op_sel_hi:[1,0,0]       // 000000003784: CC200084 0C77091C
	v_fma_mix_f32 v133, v28, v133, v29 op_sel_hi:[1,0,0]       // 00000000378C: CC200085 0C770B1C
	v_fma_mix_f32 v134, v28, v134, v29 op_sel_hi:[1,0,0]       // 000000003794: CC200086 0C770D1C
	v_fma_mix_f32 v135, v28, v135, v29 op_sel_hi:[1,0,0]       // 00000000379C: CC200087 0C770F1C
	s_wait_loadcnt 0xf                                         // 0000000037A4: BFC0000F
	v_lshrrev_b32_e32 v28, v2, v113                            // 0000000037A8: 3238E302
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000037AC: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 0000000037B0: CC20001D 8A007F1C
	s_wait_loadcnt 0xe                                         // 0000000037B8: BFC0000E
	v_and_b32_e32 v30, 0xf0f0f0f, v117                         // 0000000037BC: 363CEAFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v117                             // 0000000037C4: 323EEA84
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000037C8: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 0000000037CC: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000037D4: BF870003
	v_cvt_f32_ubyte0_e32 v136, v30                             // 0000000037D8: 7F10231E
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000037DC: BF870002
	v_cvt_f32_ubyte0_e32 v137, v31                             // 0000000037E0: 7F12231F
	v_cvt_f32_ubyte1_e32 v138, v30                             // 0000000037E4: 7F14251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000037E8: BF870004
	v_cvt_f32_ubyte1_e32 v139, v31                             // 0000000037EC: 7F16251F
	v_cvt_f32_ubyte2_e32 v140, v30                             // 0000000037F0: 7F18271E
	v_cvt_f32_ubyte2_e32 v141, v31                             // 0000000037F4: 7F1A271F
	v_cvt_f32_ubyte3_e32 v142, v30                             // 0000000037F8: 7F1C291E
	v_cvt_f32_ubyte3_e32 v143, v31                             // 0000000037FC: 7F1E291F
	v_fma_mix_f32 v136, v28, v136, v29 op_sel_hi:[1,0,0]       // 000000003800: CC200088 0C77111C
	v_fma_mix_f32 v137, v28, v137, v29 op_sel_hi:[1,0,0]       // 000000003808: CC200089 0C77131C
	v_fma_mix_f32 v138, v28, v138, v29 op_sel_hi:[1,0,0]       // 000000003810: CC20008A 0C77151C
	v_fma_mix_f32 v139, v28, v139, v29 op_sel_hi:[1,0,0]       // 000000003818: CC20008B 0C77171C
	v_fma_mix_f32 v140, v28, v140, v29 op_sel_hi:[1,0,0]       // 000000003820: CC20008C 0C77191C
	v_fma_mix_f32 v141, v28, v141, v29 op_sel_hi:[1,0,0]       // 000000003828: CC20008D 0C771B1C
	v_fma_mix_f32 v142, v28, v142, v29 op_sel_hi:[1,0,0]       // 000000003830: CC20008E 0C771D1C
	v_fma_mix_f32 v143, v28, v143, v29 op_sel_hi:[1,0,0]       // 000000003838: CC20008F 0C771F1C
	s_wait_loadcnt 0xd                                         // 000000003840: BFC0000D
	v_lshrrev_b32_e32 v28, v2, v114                            // 000000003844: 3238E502
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003848: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 00000000384C: CC20001D 8A007F1C
	s_wait_loadcnt 0xc                                         // 000000003854: BFC0000C
	v_and_b32_e32 v30, 0xf0f0f0f, v118                         // 000000003858: 363CECFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v118                             // 000000003860: 323EEC84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003864: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000003868: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003870: BF870003
	v_cvt_f32_ubyte0_e32 v144, v30                             // 000000003874: 7F20231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003878: BF870002
	v_cvt_f32_ubyte0_e32 v145, v31                             // 00000000387C: 7F22231F
	v_cvt_f32_ubyte1_e32 v146, v30                             // 000000003880: 7F24251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003884: BF870004
	v_cvt_f32_ubyte1_e32 v147, v31                             // 000000003888: 7F26251F
	v_cvt_f32_ubyte2_e32 v148, v30                             // 00000000388C: 7F28271E
	v_cvt_f32_ubyte2_e32 v149, v31                             // 000000003890: 7F2A271F
	v_cvt_f32_ubyte3_e32 v150, v30                             // 000000003894: 7F2C291E
	v_cvt_f32_ubyte3_e32 v151, v31                             // 000000003898: 7F2E291F
	v_fma_mix_f32 v144, v28, v144, v29 op_sel_hi:[1,0,0]       // 00000000389C: CC200090 0C77211C
	v_fma_mix_f32 v145, v28, v145, v29 op_sel_hi:[1,0,0]       // 0000000038A4: CC200091 0C77231C
	v_fma_mix_f32 v146, v28, v146, v29 op_sel_hi:[1,0,0]       // 0000000038AC: CC200092 0C77251C
	v_fma_mix_f32 v147, v28, v147, v29 op_sel_hi:[1,0,0]       // 0000000038B4: CC200093 0C77271C
	v_fma_mix_f32 v148, v28, v148, v29 op_sel_hi:[1,0,0]       // 0000000038BC: CC200094 0C77291C
	v_fma_mix_f32 v149, v28, v149, v29 op_sel_hi:[1,0,0]       // 0000000038C4: CC200095 0C772B1C
	v_fma_mix_f32 v150, v28, v150, v29 op_sel_hi:[1,0,0]       // 0000000038CC: CC200096 0C772D1C
	v_fma_mix_f32 v151, v28, v151, v29 op_sel_hi:[1,0,0]       // 0000000038D4: CC200097 0C772F1C
	s_wait_loadcnt 0xb                                         // 0000000038DC: BFC0000B
	v_lshrrev_b32_e32 v28, v2, v115                            // 0000000038E0: 3238E702
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000038E4: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 0000000038E8: CC20001D 8A007F1C
	s_wait_loadcnt 0xa                                         // 0000000038F0: BFC0000A
	v_and_b32_e32 v30, 0xf0f0f0f, v119                         // 0000000038F4: 363CEEFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v119                             // 0000000038FC: 323EEE84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003900: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000003904: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000390C: BF870003
	v_cvt_f32_ubyte0_e32 v152, v30                             // 000000003910: 7F30231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003914: BF870002
	v_cvt_f32_ubyte0_e32 v153, v31                             // 000000003918: 7F32231F
	v_cvt_f32_ubyte1_e32 v154, v30                             // 00000000391C: 7F34251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003920: BF870004
	v_cvt_f32_ubyte1_e32 v155, v31                             // 000000003924: 7F36251F
	v_cvt_f32_ubyte2_e32 v156, v30                             // 000000003928: 7F38271E
	v_cvt_f32_ubyte2_e32 v157, v31                             // 00000000392C: 7F3A271F
	v_cvt_f32_ubyte3_e32 v158, v30                             // 000000003930: 7F3C291E
	v_cvt_f32_ubyte3_e32 v159, v31                             // 000000003934: 7F3E291F
	v_fma_mix_f32 v152, v28, v152, v29 op_sel_hi:[1,0,0]       // 000000003938: CC200098 0C77311C
	v_fma_mix_f32 v153, v28, v153, v29 op_sel_hi:[1,0,0]       // 000000003940: CC200099 0C77331C
	v_fma_mix_f32 v154, v28, v154, v29 op_sel_hi:[1,0,0]       // 000000003948: CC20009A 0C77351C
	v_fma_mix_f32 v155, v28, v155, v29 op_sel_hi:[1,0,0]       // 000000003950: CC20009B 0C77371C
	v_fma_mix_f32 v156, v28, v156, v29 op_sel_hi:[1,0,0]       // 000000003958: CC20009C 0C77391C
	v_fma_mix_f32 v157, v28, v157, v29 op_sel_hi:[1,0,0]       // 000000003960: CC20009D 0C773B1C
	v_fma_mix_f32 v158, v28, v158, v29 op_sel_hi:[1,0,0]       // 000000003968: CC20009E 0C773D1C
	v_fma_mix_f32 v159, v28, v159, v29 op_sel_hi:[1,0,0]       // 000000003970: CC20009F 0C773F1C
	s_barrier_wait 0xffff                                      // 000000003978: BF94FFFF
	s_bitcmp0_b32 s29, 0                                       // 00000000397C: BF0C801D
	s_cbranch_scc1 74                                          // 000000003980: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x21ac>
	ds_load_b128 v[16:19], v4                                  // 000000003984: DBFC0000 10000004
	ds_load_b128 v[20:23], v4 offset:16                        // 00000000398C: DBFC0010 14000004
	s_wait_dscnt 0x1                                           // 000000003994: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000003998: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 00000000399C: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000039A0: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 0000000039A4: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039A8: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 0000000039AC: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039B0: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 0000000039B4: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039B8: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 0000000039BC: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039C0: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 0000000039C4: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039C8: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 0000000039CC: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039D0: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 0000000039D4: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039D8: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 0000000039DC: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039E0: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 0000000039E4: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039E8: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 0000000039EC: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039F0: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 0000000039F4: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000039F8: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 0000000039FC: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A00: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000003A04: 5636279B
	s_wait_dscnt 0x0                                           // 000000003A08: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A0C: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000003A10: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A14: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000003A18: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A1C: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000003A20: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A24: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000003A28: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A2C: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000003A30: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A34: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000003A38: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A3C: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000003A40: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A44: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000003A48: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A4C: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000003A50: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A54: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000003A58: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A5C: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000003A60: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A64: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000003A68: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A6C: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000003A70: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A74: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000003A78: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A7C: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000003A80: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A84: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000003A88: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A8C: BF870004
	v_add_f32_e32 v80, 0, v24                                  // 000000003A90: 06A03080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A94: BF870004
	v_add_f32_e32 v84, 0, v25                                  // 000000003A98: 06A83280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A9C: BF870004
	v_add_f32_e32 v88, 0, v26                                  // 000000003AA0: 06B03480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003AA4: BF870004
	v_add_f32_e32 v92, 0, v27                                  // 000000003AA8: 06B83680
	s_bitcmp0_b32 s29, 1                                       // 000000003AAC: BF0C811D
	s_cbranch_scc1 74                                          // 000000003AB0: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x22dc>
	ds_load_b128 v[16:19], v4 offset:1040                      // 000000003AB4: DBFC0410 10000004
	ds_load_b128 v[20:23], v4 offset:1056                      // 000000003ABC: DBFC0420 14000004
	s_wait_dscnt 0x1                                           // 000000003AC4: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000003AC8: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000003ACC: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000003AD0: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000003AD4: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003AD8: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000003ADC: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003AE0: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000003AE4: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003AE8: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000003AEC: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003AF0: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000003AF4: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003AF8: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000003AFC: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B00: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000003B04: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B08: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000003B0C: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B10: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000003B14: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B18: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000003B1C: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B20: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000003B24: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B28: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000003B2C: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B30: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000003B34: 5636279B
	s_wait_dscnt 0x0                                           // 000000003B38: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B3C: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000003B40: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B44: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000003B48: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B4C: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000003B50: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B54: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000003B58: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B5C: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000003B60: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B64: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000003B68: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B6C: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000003B70: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B74: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000003B78: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B7C: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000003B80: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B84: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000003B88: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B8C: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000003B90: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B94: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000003B98: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B9C: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000003BA0: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003BA4: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000003BA8: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003BAC: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000003BB0: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003BB4: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000003BB8: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003BBC: BF870004
	v_add_f32_e32 v81, 0, v24                                  // 000000003BC0: 06A23080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003BC4: BF870004
	v_add_f32_e32 v85, 0, v25                                  // 000000003BC8: 06AA3280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003BCC: BF870004
	v_add_f32_e32 v89, 0, v26                                  // 000000003BD0: 06B23480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003BD4: BF870004
	v_add_f32_e32 v93, 0, v27                                  // 000000003BD8: 06BA3680
	s_bitcmp0_b32 s29, 2                                       // 000000003BDC: BF0C821D
	s_cbranch_scc1 74                                          // 000000003BE0: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x240c>
	ds_load_b128 v[16:19], v4 offset:2080                      // 000000003BE4: DBFC0820 10000004
	ds_load_b128 v[20:23], v4 offset:2096                      // 000000003BEC: DBFC0830 14000004
	s_wait_dscnt 0x1                                           // 000000003BF4: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000003BF8: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000003BFC: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000003C00: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000003C04: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C08: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000003C0C: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C10: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000003C14: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C18: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000003C1C: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C20: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000003C24: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C28: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000003C2C: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C30: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000003C34: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C38: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000003C3C: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C40: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000003C44: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C48: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000003C4C: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C50: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000003C54: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C58: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000003C5C: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C60: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000003C64: 5636279B
	s_wait_dscnt 0x0                                           // 000000003C68: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C6C: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000003C70: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C74: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000003C78: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C7C: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000003C80: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C84: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000003C88: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C8C: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000003C90: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C94: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000003C98: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003C9C: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000003CA0: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CA4: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000003CA8: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CAC: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000003CB0: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CB4: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000003CB8: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CBC: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000003CC0: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CC4: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000003CC8: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CCC: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000003CD0: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CD4: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000003CD8: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CDC: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000003CE0: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CE4: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000003CE8: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CEC: BF870004
	v_add_f32_e32 v82, 0, v24                                  // 000000003CF0: 06A43080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CF4: BF870004
	v_add_f32_e32 v86, 0, v25                                  // 000000003CF8: 06AC3280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003CFC: BF870004
	v_add_f32_e32 v90, 0, v26                                  // 000000003D00: 06B43480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D04: BF870004
	v_add_f32_e32 v94, 0, v27                                  // 000000003D08: 06BC3680
	s_bitcmp0_b32 s29, 3                                       // 000000003D0C: BF0C831D
	s_cbranch_scc1 74                                          // 000000003D10: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x253c>
	ds_load_b128 v[16:19], v4 offset:3120                      // 000000003D14: DBFC0C30 10000004
	ds_load_b128 v[20:23], v4 offset:3136                      // 000000003D1C: DBFC0C40 14000004
	s_wait_dscnt 0x1                                           // 000000003D24: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000003D28: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000003D2C: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000003D30: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000003D34: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D38: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000003D3C: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D40: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000003D44: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D48: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000003D4C: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D50: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000003D54: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D58: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000003D5C: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D60: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000003D64: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D68: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000003D6C: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D70: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000003D74: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D78: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000003D7C: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D80: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000003D84: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D88: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000003D8C: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D90: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000003D94: 5636279B
	s_wait_dscnt 0x0                                           // 000000003D98: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D9C: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000003DA0: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DA4: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000003DA8: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DAC: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000003DB0: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DB4: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000003DB8: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DBC: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000003DC0: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DC4: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000003DC8: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DCC: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000003DD0: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DD4: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000003DD8: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DDC: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000003DE0: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DE4: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000003DE8: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DEC: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000003DF0: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DF4: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000003DF8: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003DFC: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000003E00: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003E04: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000003E08: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003E0C: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000003E10: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003E14: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000003E18: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003E1C: BF870004
	v_add_f32_e32 v83, 0, v24                                  // 000000003E20: 06A63080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003E24: BF870004
	v_add_f32_e32 v87, 0, v25                                  // 000000003E28: 06AE3280
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003E2C: BF870004
	v_add_f32_e32 v91, 0, v26                                  // 000000003E30: 06B63480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003E34: BF870004
	v_add_f32_e32 v95, 0, v27                                  // 000000003E38: 06BE3680
	s_barrier_signal -1                                        // 000000003E3C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000003E40: BF94FFFF
	s_wait_loadcnt 0x9                                         // 000000003E44: BFC00009
	ds_store_b128 v5, v[104:107]                               // 000000003E48: DB7C0000 00006805
	s_wait_loadcnt 0x8                                         // 000000003E50: BFC00008
	ds_store_b128 v5, v[108:111] offset:2080                   // 000000003E54: DB7C0820 00006C05
	s_wait_dscnt 0x0                                           // 000000003E5C: BFC60000
	s_barrier_signal -1                                        // 000000003E60: BE804EC1
	buffer_load_b128 v[120:123], v6, s[52:55], s81 offen offset:5120// 000000003E64: C405C051 40806878 00140006
	buffer_load_b128 v[124:127], v6, s[52:55], s82 offen offset:5120// 000000003E70: C405C052 4080687C 00140006
	buffer_load_b32 v112, off, s[48:51], s73 offset:20         // 000000003E7C: C4050049 00806070 00001400
	buffer_load_b32 v116, v3, s[48:51], s69 offen offset:640   // 000000003E88: C4050045 40806074 00028003
	buffer_load_b32 v113, off, s[48:51], s74 offset:20         // 000000003E94: C405004A 00806071 00001400
	buffer_load_b32 v117, v3, s[48:51], s70 offen offset:640   // 000000003EA0: C4050046 40806075 00028003
	buffer_load_b32 v114, off, s[48:51], s75 offset:20         // 000000003EAC: C405004B 00806072 00001400
	buffer_load_b32 v118, v3, s[48:51], s71 offen offset:640   // 000000003EB8: C4050047 40806076 00028003
	buffer_load_b32 v115, off, s[48:51], s76 offset:20         // 000000003EC4: C405004C 00806073 00001400
	buffer_load_b32 v119, v3, s[48:51], s72 offen offset:640   // 000000003ED0: C4050048 40806077 00028003
	s_wait_loadcnt 0x11                                        // 000000003EDC: BFC00011
	v_lshrrev_b32_e32 v28, v2, v96                             // 000000003EE0: 3238C102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003EE4: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000003EE8: CC20001D 8A007F1C
	s_wait_loadcnt 0x10                                        // 000000003EF0: BFC00010
	v_and_b32_e32 v30, 0xf0f0f0f, v100                         // 000000003EF4: 363CC8FF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v100                             // 000000003EFC: 323EC884
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003F00: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000003F04: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003F0C: BF870003
	v_cvt_f32_ubyte0_e32 v128, v30                             // 000000003F10: 7F00231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003F14: BF870002
	v_cvt_f32_ubyte0_e32 v129, v31                             // 000000003F18: 7F02231F
	v_cvt_f32_ubyte1_e32 v130, v30                             // 000000003F1C: 7F04251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003F20: BF870004
	v_cvt_f32_ubyte1_e32 v131, v31                             // 000000003F24: 7F06251F
	v_cvt_f32_ubyte2_e32 v132, v30                             // 000000003F28: 7F08271E
	v_cvt_f32_ubyte2_e32 v133, v31                             // 000000003F2C: 7F0A271F
	v_cvt_f32_ubyte3_e32 v134, v30                             // 000000003F30: 7F0C291E
	v_cvt_f32_ubyte3_e32 v135, v31                             // 000000003F34: 7F0E291F
	v_fma_mix_f32 v128, v28, v128, v29 op_sel_hi:[1,0,0]       // 000000003F38: CC200080 0C77011C
	v_fma_mix_f32 v129, v28, v129, v29 op_sel_hi:[1,0,0]       // 000000003F40: CC200081 0C77031C
	v_fma_mix_f32 v130, v28, v130, v29 op_sel_hi:[1,0,0]       // 000000003F48: CC200082 0C77051C
	v_fma_mix_f32 v131, v28, v131, v29 op_sel_hi:[1,0,0]       // 000000003F50: CC200083 0C77071C
	v_fma_mix_f32 v132, v28, v132, v29 op_sel_hi:[1,0,0]       // 000000003F58: CC200084 0C77091C
	v_fma_mix_f32 v133, v28, v133, v29 op_sel_hi:[1,0,0]       // 000000003F60: CC200085 0C770B1C
	v_fma_mix_f32 v134, v28, v134, v29 op_sel_hi:[1,0,0]       // 000000003F68: CC200086 0C770D1C
	v_fma_mix_f32 v135, v28, v135, v29 op_sel_hi:[1,0,0]       // 000000003F70: CC200087 0C770F1C
	s_wait_loadcnt 0xf                                         // 000000003F78: BFC0000F
	v_lshrrev_b32_e32 v28, v2, v97                             // 000000003F7C: 3238C302
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003F80: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000003F84: CC20001D 8A007F1C
	s_wait_loadcnt 0xe                                         // 000000003F8C: BFC0000E
	v_and_b32_e32 v30, 0xf0f0f0f, v101                         // 000000003F90: 363CCAFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v101                             // 000000003F98: 323ECA84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003F9C: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000003FA0: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003FA8: BF870003
	v_cvt_f32_ubyte0_e32 v136, v30                             // 000000003FAC: 7F10231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003FB0: BF870002
	v_cvt_f32_ubyte0_e32 v137, v31                             // 000000003FB4: 7F12231F
	v_cvt_f32_ubyte1_e32 v138, v30                             // 000000003FB8: 7F14251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003FBC: BF870004
	v_cvt_f32_ubyte1_e32 v139, v31                             // 000000003FC0: 7F16251F
	v_cvt_f32_ubyte2_e32 v140, v30                             // 000000003FC4: 7F18271E
	v_cvt_f32_ubyte2_e32 v141, v31                             // 000000003FC8: 7F1A271F
	v_cvt_f32_ubyte3_e32 v142, v30                             // 000000003FCC: 7F1C291E
	v_cvt_f32_ubyte3_e32 v143, v31                             // 000000003FD0: 7F1E291F
	v_fma_mix_f32 v136, v28, v136, v29 op_sel_hi:[1,0,0]       // 000000003FD4: CC200088 0C77111C
	v_fma_mix_f32 v137, v28, v137, v29 op_sel_hi:[1,0,0]       // 000000003FDC: CC200089 0C77131C
	v_fma_mix_f32 v138, v28, v138, v29 op_sel_hi:[1,0,0]       // 000000003FE4: CC20008A 0C77151C
	v_fma_mix_f32 v139, v28, v139, v29 op_sel_hi:[1,0,0]       // 000000003FEC: CC20008B 0C77171C
	v_fma_mix_f32 v140, v28, v140, v29 op_sel_hi:[1,0,0]       // 000000003FF4: CC20008C 0C77191C
	v_fma_mix_f32 v141, v28, v141, v29 op_sel_hi:[1,0,0]       // 000000003FFC: CC20008D 0C771B1C
	v_fma_mix_f32 v142, v28, v142, v29 op_sel_hi:[1,0,0]       // 000000004004: CC20008E 0C771D1C
	v_fma_mix_f32 v143, v28, v143, v29 op_sel_hi:[1,0,0]       // 00000000400C: CC20008F 0C771F1C
	s_wait_loadcnt 0xd                                         // 000000004014: BFC0000D
	v_lshrrev_b32_e32 v28, v2, v98                             // 000000004018: 3238C502
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000401C: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000004020: CC20001D 8A007F1C
	s_wait_loadcnt 0xc                                         // 000000004028: BFC0000C
	v_and_b32_e32 v30, 0xf0f0f0f, v102                         // 00000000402C: 363CCCFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v102                             // 000000004034: 323ECC84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004038: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 00000000403C: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004044: BF870003
	v_cvt_f32_ubyte0_e32 v144, v30                             // 000000004048: 7F20231E
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000404C: BF870002
	v_cvt_f32_ubyte0_e32 v145, v31                             // 000000004050: 7F22231F
	v_cvt_f32_ubyte1_e32 v146, v30                             // 000000004054: 7F24251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004058: BF870004
	v_cvt_f32_ubyte1_e32 v147, v31                             // 00000000405C: 7F26251F
	v_cvt_f32_ubyte2_e32 v148, v30                             // 000000004060: 7F28271E
	v_cvt_f32_ubyte2_e32 v149, v31                             // 000000004064: 7F2A271F
	v_cvt_f32_ubyte3_e32 v150, v30                             // 000000004068: 7F2C291E
	v_cvt_f32_ubyte3_e32 v151, v31                             // 00000000406C: 7F2E291F
	v_fma_mix_f32 v144, v28, v144, v29 op_sel_hi:[1,0,0]       // 000000004070: CC200090 0C77211C
	v_fma_mix_f32 v145, v28, v145, v29 op_sel_hi:[1,0,0]       // 000000004078: CC200091 0C77231C
	v_fma_mix_f32 v146, v28, v146, v29 op_sel_hi:[1,0,0]       // 000000004080: CC200092 0C77251C
	v_fma_mix_f32 v147, v28, v147, v29 op_sel_hi:[1,0,0]       // 000000004088: CC200093 0C77271C
	v_fma_mix_f32 v148, v28, v148, v29 op_sel_hi:[1,0,0]       // 000000004090: CC200094 0C77291C
	v_fma_mix_f32 v149, v28, v149, v29 op_sel_hi:[1,0,0]       // 000000004098: CC200095 0C772B1C
	v_fma_mix_f32 v150, v28, v150, v29 op_sel_hi:[1,0,0]       // 0000000040A0: CC200096 0C772D1C
	v_fma_mix_f32 v151, v28, v151, v29 op_sel_hi:[1,0,0]       // 0000000040A8: CC200097 0C772F1C
	s_wait_loadcnt 0xb                                         // 0000000040B0: BFC0000B
	v_lshrrev_b32_e32 v28, v2, v99                             // 0000000040B4: 3238C702
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000040B8: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 0000000040BC: CC20001D 8A007F1C
	s_wait_loadcnt 0xa                                         // 0000000040C4: BFC0000A
	v_and_b32_e32 v30, 0xf0f0f0f, v103                         // 0000000040C8: 363CCEFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v103                             // 0000000040D0: 323ECE84
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000040D4: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 0000000040D8: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000040E0: BF870003
	v_cvt_f32_ubyte0_e32 v152, v30                             // 0000000040E4: 7F30231E
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000040E8: BF870002
	v_cvt_f32_ubyte0_e32 v153, v31                             // 0000000040EC: 7F32231F
	v_cvt_f32_ubyte1_e32 v154, v30                             // 0000000040F0: 7F34251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000040F4: BF870004
	v_cvt_f32_ubyte1_e32 v155, v31                             // 0000000040F8: 7F36251F
	v_cvt_f32_ubyte2_e32 v156, v30                             // 0000000040FC: 7F38271E
	v_cvt_f32_ubyte2_e32 v157, v31                             // 000000004100: 7F3A271F
	v_cvt_f32_ubyte3_e32 v158, v30                             // 000000004104: 7F3C291E
	v_cvt_f32_ubyte3_e32 v159, v31                             // 000000004108: 7F3E291F
	v_fma_mix_f32 v152, v28, v152, v29 op_sel_hi:[1,0,0]       // 00000000410C: CC200098 0C77311C
	v_fma_mix_f32 v153, v28, v153, v29 op_sel_hi:[1,0,0]       // 000000004114: CC200099 0C77331C
	v_fma_mix_f32 v154, v28, v154, v29 op_sel_hi:[1,0,0]       // 00000000411C: CC20009A 0C77351C
	v_fma_mix_f32 v155, v28, v155, v29 op_sel_hi:[1,0,0]       // 000000004124: CC20009B 0C77371C
	v_fma_mix_f32 v156, v28, v156, v29 op_sel_hi:[1,0,0]       // 00000000412C: CC20009C 0C77391C
	v_fma_mix_f32 v157, v28, v157, v29 op_sel_hi:[1,0,0]       // 000000004134: CC20009D 0C773B1C
	v_fma_mix_f32 v158, v28, v158, v29 op_sel_hi:[1,0,0]       // 00000000413C: CC20009E 0C773D1C
	v_fma_mix_f32 v159, v28, v159, v29 op_sel_hi:[1,0,0]       // 000000004144: CC20009F 0C773F1C
	s_barrier_wait 0xffff                                      // 00000000414C: BF94FFFF
	s_bitcmp0_b32 s29, 0                                       // 000000004150: BF0C801D
	s_cbranch_scc1 74                                          // 000000004154: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x2980>
	ds_load_b128 v[16:19], v4                                  // 000000004158: DBFC0000 10000004
	ds_load_b128 v[20:23], v4 offset:16                        // 000000004160: DBFC0010 14000004
	s_wait_dscnt 0x1                                           // 000000004168: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 00000000416C: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000004170: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000004174: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000004178: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000417C: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000004180: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004184: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000004188: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000418C: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000004190: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004194: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000004198: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000419C: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 0000000041A0: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041A4: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 0000000041A8: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041AC: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 0000000041B0: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041B4: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 0000000041B8: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041BC: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 0000000041C0: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041C4: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 0000000041C8: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041CC: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 0000000041D0: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041D4: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 0000000041D8: 5636279B
	s_wait_dscnt 0x0                                           // 0000000041DC: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041E0: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 0000000041E4: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041E8: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 0000000041EC: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041F0: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000041F4: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041F8: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000041FC: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004200: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000004204: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004208: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 00000000420C: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004210: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000004214: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004218: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 00000000421C: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004220: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000004224: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004228: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 00000000422C: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004230: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000004234: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004238: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 00000000423C: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004240: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000004244: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004248: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 00000000424C: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004250: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000004254: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004258: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 00000000425C: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004260: BF870004
	v_add_f32_e32 v32, v32, v24                                // 000000004264: 06403120
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004268: BF870004
	v_add_f32_e32 v36, v36, v25                                // 00000000426C: 06483324
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004270: BF870004
	v_add_f32_e32 v40, v40, v26                                // 000000004274: 06503528
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004278: BF870004
	v_add_f32_e32 v44, v44, v27                                // 00000000427C: 0658372C
	s_bitcmp0_b32 s29, 1                                       // 000000004280: BF0C811D
	s_cbranch_scc1 74                                          // 000000004284: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x2ab0>
	ds_load_b128 v[16:19], v4 offset:1040                      // 000000004288: DBFC0410 10000004
	ds_load_b128 v[20:23], v4 offset:1056                      // 000000004290: DBFC0420 14000004
	s_wait_dscnt 0x1                                           // 000000004298: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 00000000429C: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 0000000042A0: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000042A4: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 0000000042A8: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042AC: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 0000000042B0: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042B4: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 0000000042B8: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042BC: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 0000000042C0: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042C4: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 0000000042C8: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042CC: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 0000000042D0: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042D4: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 0000000042D8: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042DC: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 0000000042E0: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042E4: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 0000000042E8: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042EC: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 0000000042F0: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042F4: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 0000000042F8: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000042FC: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000004300: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004304: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000004308: 5636279B
	s_wait_dscnt 0x0                                           // 00000000430C: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004310: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000004314: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004318: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 00000000431C: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004320: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000004324: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004328: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 00000000432C: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004330: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000004334: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004338: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 00000000433C: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004340: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000004344: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004348: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 00000000434C: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004350: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000004354: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004358: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 00000000435C: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004360: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000004364: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004368: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 00000000436C: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004370: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000004374: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004378: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 00000000437C: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004380: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000004384: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004388: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 00000000438C: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004390: BF870004
	v_add_f32_e32 v33, v33, v24                                // 000000004394: 06423121
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004398: BF870004
	v_add_f32_e32 v37, v37, v25                                // 00000000439C: 064A3325
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000043A0: BF870004
	v_add_f32_e32 v41, v41, v26                                // 0000000043A4: 06523529
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000043A8: BF870004
	v_add_f32_e32 v45, v45, v27                                // 0000000043AC: 065A372D
	s_bitcmp0_b32 s29, 2                                       // 0000000043B0: BF0C821D
	s_cbranch_scc1 74                                          // 0000000043B4: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x2be0>
	ds_load_b128 v[16:19], v4 offset:2080                      // 0000000043B8: DBFC0820 10000004
	ds_load_b128 v[20:23], v4 offset:2096                      // 0000000043C0: DBFC0830 14000004
	s_wait_dscnt 0x1                                           // 0000000043C8: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 0000000043CC: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 0000000043D0: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000043D4: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 0000000043D8: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000043DC: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 0000000043E0: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000043E4: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 0000000043E8: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000043EC: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 0000000043F0: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000043F4: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 0000000043F8: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000043FC: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000004400: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004404: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000004408: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000440C: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000004410: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004414: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000004418: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000441C: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000004420: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004424: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000004428: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000442C: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000004430: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004434: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000004438: 5636279B
	s_wait_dscnt 0x0                                           // 00000000443C: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004440: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000004444: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004448: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 00000000444C: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004450: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000004454: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004458: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 00000000445C: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004460: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000004464: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004468: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 00000000446C: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004470: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000004474: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004478: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 00000000447C: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004480: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000004484: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004488: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 00000000448C: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004490: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000004494: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004498: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 00000000449C: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000044A0: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 0000000044A4: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000044A8: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 0000000044AC: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000044B0: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 0000000044B4: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000044B8: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 0000000044BC: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000044C0: BF870004
	v_add_f32_e32 v34, v34, v24                                // 0000000044C4: 06443122
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000044C8: BF870004
	v_add_f32_e32 v38, v38, v25                                // 0000000044CC: 064C3326
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000044D0: BF870004
	v_add_f32_e32 v42, v42, v26                                // 0000000044D4: 0654352A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000044D8: BF870004
	v_add_f32_e32 v46, v46, v27                                // 0000000044DC: 065C372E
	s_bitcmp0_b32 s29, 3                                       // 0000000044E0: BF0C831D
	s_cbranch_scc1 74                                          // 0000000044E4: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x2d10>
	ds_load_b128 v[16:19], v4 offset:3120                      // 0000000044E8: DBFC0C30 10000004
	ds_load_b128 v[20:23], v4 offset:3136                      // 0000000044F0: DBFC0C40 14000004
	s_wait_dscnt 0x1                                           // 0000000044F8: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 0000000044FC: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000004500: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000004504: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000004508: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000450C: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000004510: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004514: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000004518: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000451C: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000004520: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004524: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000004528: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000452C: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000004530: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004534: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000004538: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000453C: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000004540: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004544: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000004548: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000454C: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000004550: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004554: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000004558: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000455C: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000004560: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004564: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000004568: 5636279B
	s_wait_dscnt 0x0                                           // 00000000456C: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004570: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000004574: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004578: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 00000000457C: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004580: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000004584: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004588: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 00000000458C: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004590: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000004594: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004598: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 00000000459C: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045A0: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000045A4: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045A8: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000045AC: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045B0: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000045B4: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045B8: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 0000000045BC: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045C0: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 0000000045C4: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045C8: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 0000000045CC: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045D0: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 0000000045D4: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045D8: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 0000000045DC: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045E0: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 0000000045E4: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045E8: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 0000000045EC: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045F0: BF870004
	v_add_f32_e32 v35, v35, v24                                // 0000000045F4: 06463123
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045F8: BF870004
	v_add_f32_e32 v39, v39, v25                                // 0000000045FC: 064E3327
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004600: BF870004
	v_add_f32_e32 v43, v43, v26                                // 000000004604: 0656352B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004608: BF870004
	v_add_f32_e32 v47, v47, v27                                // 00000000460C: 065E372F
	s_barrier_signal -1                                        // 000000004610: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004614: BF94FFFF
	s_wait_loadcnt 0x9                                         // 000000004618: BFC00009
	ds_store_b128 v5, v[120:123]                               // 00000000461C: DB7C0000 00007805
	s_wait_loadcnt 0x8                                         // 000000004624: BFC00008
	ds_store_b128 v5, v[124:127] offset:2080                   // 000000004628: DB7C0820 00007C05
	s_wait_dscnt 0x0                                           // 000000004630: BFC60000
	s_barrier_signal -1                                        // 000000004634: BE804EC1
	buffer_load_b128 v[104:107], v6, s[52:55], s81 offen offset:6144// 000000004638: C405C051 40806868 00180006
	buffer_load_b128 v[108:111], v6, s[52:55], s82 offen offset:6144// 000000004644: C405C052 4080686C 00180006
	buffer_load_b32 v96, off, s[48:51], s73 offset:24          // 000000004650: C4050049 00806060 00001800
	buffer_load_b32 v100, v3, s[48:51], s69 offen offset:768   // 00000000465C: C4050045 40806064 00030003
	buffer_load_b32 v97, off, s[48:51], s74 offset:24          // 000000004668: C405004A 00806061 00001800
	buffer_load_b32 v101, v3, s[48:51], s70 offen offset:768   // 000000004674: C4050046 40806065 00030003
	buffer_load_b32 v98, off, s[48:51], s75 offset:24          // 000000004680: C405004B 00806062 00001800
	buffer_load_b32 v102, v3, s[48:51], s71 offen offset:768   // 00000000468C: C4050047 40806066 00030003
	buffer_load_b32 v99, off, s[48:51], s76 offset:24          // 000000004698: C405004C 00806063 00001800
	buffer_load_b32 v103, v3, s[48:51], s72 offen offset:768   // 0000000046A4: C4050048 40806067 00030003
	s_wait_loadcnt 0x11                                        // 0000000046B0: BFC00011
	v_lshrrev_b32_e32 v28, v2, v112                            // 0000000046B4: 3238E102
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000046B8: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 0000000046BC: CC20001D 8A007F1C
	s_wait_loadcnt 0x10                                        // 0000000046C4: BFC00010
	v_and_b32_e32 v30, 0xf0f0f0f, v116                         // 0000000046C8: 363CE8FF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v116                             // 0000000046D0: 323EE884
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000046D4: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 0000000046D8: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000046E0: BF870003
	v_cvt_f32_ubyte0_e32 v128, v30                             // 0000000046E4: 7F00231E
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000046E8: BF870002
	v_cvt_f32_ubyte0_e32 v129, v31                             // 0000000046EC: 7F02231F
	v_cvt_f32_ubyte1_e32 v130, v30                             // 0000000046F0: 7F04251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000046F4: BF870004
	v_cvt_f32_ubyte1_e32 v131, v31                             // 0000000046F8: 7F06251F
	v_cvt_f32_ubyte2_e32 v132, v30                             // 0000000046FC: 7F08271E
	v_cvt_f32_ubyte2_e32 v133, v31                             // 000000004700: 7F0A271F
	v_cvt_f32_ubyte3_e32 v134, v30                             // 000000004704: 7F0C291E
	v_cvt_f32_ubyte3_e32 v135, v31                             // 000000004708: 7F0E291F
	v_fma_mix_f32 v128, v28, v128, v29 op_sel_hi:[1,0,0]       // 00000000470C: CC200080 0C77011C
	v_fma_mix_f32 v129, v28, v129, v29 op_sel_hi:[1,0,0]       // 000000004714: CC200081 0C77031C
	v_fma_mix_f32 v130, v28, v130, v29 op_sel_hi:[1,0,0]       // 00000000471C: CC200082 0C77051C
	v_fma_mix_f32 v131, v28, v131, v29 op_sel_hi:[1,0,0]       // 000000004724: CC200083 0C77071C
	v_fma_mix_f32 v132, v28, v132, v29 op_sel_hi:[1,0,0]       // 00000000472C: CC200084 0C77091C
	v_fma_mix_f32 v133, v28, v133, v29 op_sel_hi:[1,0,0]       // 000000004734: CC200085 0C770B1C
	v_fma_mix_f32 v134, v28, v134, v29 op_sel_hi:[1,0,0]       // 00000000473C: CC200086 0C770D1C
	v_fma_mix_f32 v135, v28, v135, v29 op_sel_hi:[1,0,0]       // 000000004744: CC200087 0C770F1C
	s_wait_loadcnt 0xf                                         // 00000000474C: BFC0000F
	v_lshrrev_b32_e32 v28, v2, v113                            // 000000004750: 3238E302
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004754: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000004758: CC20001D 8A007F1C
	s_wait_loadcnt 0xe                                         // 000000004760: BFC0000E
	v_and_b32_e32 v30, 0xf0f0f0f, v117                         // 000000004764: 363CEAFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v117                             // 00000000476C: 323EEA84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004770: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000004774: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000477C: BF870003
	v_cvt_f32_ubyte0_e32 v136, v30                             // 000000004780: 7F10231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004784: BF870002
	v_cvt_f32_ubyte0_e32 v137, v31                             // 000000004788: 7F12231F
	v_cvt_f32_ubyte1_e32 v138, v30                             // 00000000478C: 7F14251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004790: BF870004
	v_cvt_f32_ubyte1_e32 v139, v31                             // 000000004794: 7F16251F
	v_cvt_f32_ubyte2_e32 v140, v30                             // 000000004798: 7F18271E
	v_cvt_f32_ubyte2_e32 v141, v31                             // 00000000479C: 7F1A271F
	v_cvt_f32_ubyte3_e32 v142, v30                             // 0000000047A0: 7F1C291E
	v_cvt_f32_ubyte3_e32 v143, v31                             // 0000000047A4: 7F1E291F
	v_fma_mix_f32 v136, v28, v136, v29 op_sel_hi:[1,0,0]       // 0000000047A8: CC200088 0C77111C
	v_fma_mix_f32 v137, v28, v137, v29 op_sel_hi:[1,0,0]       // 0000000047B0: CC200089 0C77131C
	v_fma_mix_f32 v138, v28, v138, v29 op_sel_hi:[1,0,0]       // 0000000047B8: CC20008A 0C77151C
	v_fma_mix_f32 v139, v28, v139, v29 op_sel_hi:[1,0,0]       // 0000000047C0: CC20008B 0C77171C
	v_fma_mix_f32 v140, v28, v140, v29 op_sel_hi:[1,0,0]       // 0000000047C8: CC20008C 0C77191C
	v_fma_mix_f32 v141, v28, v141, v29 op_sel_hi:[1,0,0]       // 0000000047D0: CC20008D 0C771B1C
	v_fma_mix_f32 v142, v28, v142, v29 op_sel_hi:[1,0,0]       // 0000000047D8: CC20008E 0C771D1C
	v_fma_mix_f32 v143, v28, v143, v29 op_sel_hi:[1,0,0]       // 0000000047E0: CC20008F 0C771F1C
	s_wait_loadcnt 0xd                                         // 0000000047E8: BFC0000D
	v_lshrrev_b32_e32 v28, v2, v114                            // 0000000047EC: 3238E502
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000047F0: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 0000000047F4: CC20001D 8A007F1C
	s_wait_loadcnt 0xc                                         // 0000000047FC: BFC0000C
	v_and_b32_e32 v30, 0xf0f0f0f, v118                         // 000000004800: 363CECFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v118                             // 000000004808: 323EEC84
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000480C: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000004810: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004818: BF870003
	v_cvt_f32_ubyte0_e32 v144, v30                             // 00000000481C: 7F20231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004820: BF870002
	v_cvt_f32_ubyte0_e32 v145, v31                             // 000000004824: 7F22231F
	v_cvt_f32_ubyte1_e32 v146, v30                             // 000000004828: 7F24251E
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000482C: BF870004
	v_cvt_f32_ubyte1_e32 v147, v31                             // 000000004830: 7F26251F
	v_cvt_f32_ubyte2_e32 v148, v30                             // 000000004834: 7F28271E
	v_cvt_f32_ubyte2_e32 v149, v31                             // 000000004838: 7F2A271F
	v_cvt_f32_ubyte3_e32 v150, v30                             // 00000000483C: 7F2C291E
	v_cvt_f32_ubyte3_e32 v151, v31                             // 000000004840: 7F2E291F
	v_fma_mix_f32 v144, v28, v144, v29 op_sel_hi:[1,0,0]       // 000000004844: CC200090 0C77211C
	v_fma_mix_f32 v145, v28, v145, v29 op_sel_hi:[1,0,0]       // 00000000484C: CC200091 0C77231C
	v_fma_mix_f32 v146, v28, v146, v29 op_sel_hi:[1,0,0]       // 000000004854: CC200092 0C77251C
	v_fma_mix_f32 v147, v28, v147, v29 op_sel_hi:[1,0,0]       // 00000000485C: CC200093 0C77271C
	v_fma_mix_f32 v148, v28, v148, v29 op_sel_hi:[1,0,0]       // 000000004864: CC200094 0C77291C
	v_fma_mix_f32 v149, v28, v149, v29 op_sel_hi:[1,0,0]       // 00000000486C: CC200095 0C772B1C
	v_fma_mix_f32 v150, v28, v150, v29 op_sel_hi:[1,0,0]       // 000000004874: CC200096 0C772D1C
	v_fma_mix_f32 v151, v28, v151, v29 op_sel_hi:[1,0,0]       // 00000000487C: CC200097 0C772F1C
	s_wait_loadcnt 0xb                                         // 000000004884: BFC0000B
	v_lshrrev_b32_e32 v28, v2, v115                            // 000000004888: 3238E702
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000488C: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000004890: CC20001D 8A007F1C
	s_wait_loadcnt 0xa                                         // 000000004898: BFC0000A
	v_and_b32_e32 v30, 0xf0f0f0f, v119                         // 00000000489C: 363CEEFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v119                             // 0000000048A4: 323EEE84
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000048A8: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 0000000048AC: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000048B4: BF870003
	v_cvt_f32_ubyte0_e32 v152, v30                             // 0000000048B8: 7F30231E
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000048BC: BF870002
	v_cvt_f32_ubyte0_e32 v153, v31                             // 0000000048C0: 7F32231F
	v_cvt_f32_ubyte1_e32 v154, v30                             // 0000000048C4: 7F34251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000048C8: BF870004
	v_cvt_f32_ubyte1_e32 v155, v31                             // 0000000048CC: 7F36251F
	v_cvt_f32_ubyte2_e32 v156, v30                             // 0000000048D0: 7F38271E
	v_cvt_f32_ubyte2_e32 v157, v31                             // 0000000048D4: 7F3A271F
	v_cvt_f32_ubyte3_e32 v158, v30                             // 0000000048D8: 7F3C291E
	v_cvt_f32_ubyte3_e32 v159, v31                             // 0000000048DC: 7F3E291F
	v_fma_mix_f32 v152, v28, v152, v29 op_sel_hi:[1,0,0]       // 0000000048E0: CC200098 0C77311C
	v_fma_mix_f32 v153, v28, v153, v29 op_sel_hi:[1,0,0]       // 0000000048E8: CC200099 0C77331C
	v_fma_mix_f32 v154, v28, v154, v29 op_sel_hi:[1,0,0]       // 0000000048F0: CC20009A 0C77351C
	v_fma_mix_f32 v155, v28, v155, v29 op_sel_hi:[1,0,0]       // 0000000048F8: CC20009B 0C77371C
	v_fma_mix_f32 v156, v28, v156, v29 op_sel_hi:[1,0,0]       // 000000004900: CC20009C 0C77391C
	v_fma_mix_f32 v157, v28, v157, v29 op_sel_hi:[1,0,0]       // 000000004908: CC20009D 0C773B1C
	v_fma_mix_f32 v158, v28, v158, v29 op_sel_hi:[1,0,0]       // 000000004910: CC20009E 0C773D1C
	v_fma_mix_f32 v159, v28, v159, v29 op_sel_hi:[1,0,0]       // 000000004918: CC20009F 0C773F1C
	s_barrier_wait 0xffff                                      // 000000004920: BF94FFFF
	s_bitcmp0_b32 s29, 0                                       // 000000004924: BF0C801D
	s_cbranch_scc1 74                                          // 000000004928: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x3154>
	ds_load_b128 v[16:19], v4                                  // 00000000492C: DBFC0000 10000004
	ds_load_b128 v[20:23], v4 offset:16                        // 000000004934: DBFC0010 14000004
	s_wait_dscnt 0x1                                           // 00000000493C: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000004940: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000004944: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000004948: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 00000000494C: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004950: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000004954: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004958: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 00000000495C: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004960: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000004964: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004968: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 00000000496C: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004970: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000004974: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004978: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 00000000497C: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004980: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000004984: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004988: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 00000000498C: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004990: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000004994: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004998: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 00000000499C: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049A0: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 0000000049A4: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049A8: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 0000000049AC: 5636279B
	s_wait_dscnt 0x0                                           // 0000000049B0: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049B4: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 0000000049B8: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049BC: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 0000000049C0: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049C4: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000049C8: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049CC: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000049D0: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049D4: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000049D8: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049DC: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000049E0: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049E4: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000049E8: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049EC: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000049F0: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049F4: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000049F8: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049FC: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000004A00: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A04: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000004A08: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A0C: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000004A10: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A14: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000004A18: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A1C: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000004A20: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A24: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000004A28: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A2C: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000004A30: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A34: BF870004
	v_add_f32_e32 v48, v48, v24                                // 000000004A38: 06603130
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A3C: BF870004
	v_add_f32_e32 v52, v52, v25                                // 000000004A40: 06683334
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A44: BF870004
	v_add_f32_e32 v56, v56, v26                                // 000000004A48: 06703538
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A4C: BF870004
	v_add_f32_e32 v60, v60, v27                                // 000000004A50: 0678373C
	s_bitcmp0_b32 s29, 1                                       // 000000004A54: BF0C811D
	s_cbranch_scc1 74                                          // 000000004A58: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x3284>
	ds_load_b128 v[16:19], v4 offset:1040                      // 000000004A5C: DBFC0410 10000004
	ds_load_b128 v[20:23], v4 offset:1056                      // 000000004A64: DBFC0420 14000004
	s_wait_dscnt 0x1                                           // 000000004A6C: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000004A70: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000004A74: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000004A78: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000004A7C: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A80: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000004A84: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A88: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000004A8C: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A90: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000004A94: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004A98: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000004A9C: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AA0: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000004AA4: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AA8: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000004AAC: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AB0: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000004AB4: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AB8: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000004ABC: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AC0: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000004AC4: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AC8: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000004ACC: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AD0: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000004AD4: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AD8: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000004ADC: 5636279B
	s_wait_dscnt 0x0                                           // 000000004AE0: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AE4: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000004AE8: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AEC: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000004AF0: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AF4: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000004AF8: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004AFC: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000004B00: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B04: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000004B08: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B0C: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000004B10: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B14: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000004B18: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B1C: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000004B20: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B24: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000004B28: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B2C: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000004B30: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B34: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000004B38: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B3C: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000004B40: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B44: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000004B48: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B4C: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000004B50: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B54: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000004B58: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B5C: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000004B60: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B64: BF870004
	v_add_f32_e32 v49, v49, v24                                // 000000004B68: 06623131
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B6C: BF870004
	v_add_f32_e32 v53, v53, v25                                // 000000004B70: 066A3335
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B74: BF870004
	v_add_f32_e32 v57, v57, v26                                // 000000004B78: 06723539
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B7C: BF870004
	v_add_f32_e32 v61, v61, v27                                // 000000004B80: 067A373D
	s_bitcmp0_b32 s29, 2                                       // 000000004B84: BF0C821D
	s_cbranch_scc1 74                                          // 000000004B88: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x33b4>
	ds_load_b128 v[16:19], v4 offset:2080                      // 000000004B8C: DBFC0820 10000004
	ds_load_b128 v[20:23], v4 offset:2096                      // 000000004B94: DBFC0830 14000004
	s_wait_dscnt 0x1                                           // 000000004B9C: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000004BA0: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000004BA4: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000004BA8: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000004BAC: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004BB0: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000004BB4: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004BB8: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000004BBC: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004BC0: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000004BC4: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004BC8: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000004BCC: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004BD0: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000004BD4: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004BD8: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000004BDC: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004BE0: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000004BE4: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004BE8: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000004BEC: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004BF0: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000004BF4: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004BF8: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000004BFC: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C00: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000004C04: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C08: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000004C0C: 5636279B
	s_wait_dscnt 0x0                                           // 000000004C10: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C14: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000004C18: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C1C: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000004C20: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C24: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000004C28: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C2C: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000004C30: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C34: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000004C38: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C3C: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000004C40: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C44: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000004C48: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C4C: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000004C50: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C54: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000004C58: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C5C: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000004C60: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C64: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000004C68: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C6C: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000004C70: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C74: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000004C78: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C7C: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000004C80: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C84: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000004C88: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C8C: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000004C90: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C94: BF870004
	v_add_f32_e32 v50, v50, v24                                // 000000004C98: 06643132
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004C9C: BF870004
	v_add_f32_e32 v54, v54, v25                                // 000000004CA0: 066C3336
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004CA4: BF870004
	v_add_f32_e32 v58, v58, v26                                // 000000004CA8: 0674353A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004CAC: BF870004
	v_add_f32_e32 v62, v62, v27                                // 000000004CB0: 067C373E
	s_bitcmp0_b32 s29, 3                                       // 000000004CB4: BF0C831D
	s_cbranch_scc1 74                                          // 000000004CB8: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x34e4>
	ds_load_b128 v[16:19], v4 offset:3120                      // 000000004CBC: DBFC0C30 10000004
	ds_load_b128 v[20:23], v4 offset:3136                      // 000000004CC4: DBFC0C40 14000004
	s_wait_dscnt 0x1                                           // 000000004CCC: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000004CD0: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000004CD4: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000004CD8: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000004CDC: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004CE0: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000004CE4: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004CE8: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000004CEC: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004CF0: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000004CF4: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004CF8: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000004CFC: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D00: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000004D04: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D08: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000004D0C: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D10: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000004D14: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D18: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000004D1C: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D20: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000004D24: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D28: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000004D2C: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D30: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000004D34: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D38: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000004D3C: 5636279B
	s_wait_dscnt 0x0                                           // 000000004D40: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D44: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000004D48: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D4C: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000004D50: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D54: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000004D58: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D5C: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000004D60: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D64: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000004D68: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D6C: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000004D70: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D74: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000004D78: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D7C: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000004D80: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D84: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000004D88: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D8C: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000004D90: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D94: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000004D98: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D9C: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000004DA0: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004DA4: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000004DA8: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004DAC: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000004DB0: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004DB4: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000004DB8: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004DBC: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000004DC0: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004DC4: BF870004
	v_add_f32_e32 v51, v51, v24                                // 000000004DC8: 06663133
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004DCC: BF870004
	v_add_f32_e32 v55, v55, v25                                // 000000004DD0: 066E3337
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004DD4: BF870004
	v_add_f32_e32 v59, v59, v26                                // 000000004DD8: 0676353B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004DDC: BF870004
	v_add_f32_e32 v63, v63, v27                                // 000000004DE0: 067E373F
	s_barrier_signal -1                                        // 000000004DE4: BE804EC1
	s_barrier_wait 0xffff                                      // 000000004DE8: BF94FFFF
	s_wait_loadcnt 0x9                                         // 000000004DEC: BFC00009
	ds_store_b128 v5, v[104:107]                               // 000000004DF0: DB7C0000 00006805
	s_wait_loadcnt 0x8                                         // 000000004DF8: BFC00008
	ds_store_b128 v5, v[108:111] offset:2080                   // 000000004DFC: DB7C0820 00006C05
	s_wait_dscnt 0x0                                           // 000000004E04: BFC60000
	s_barrier_signal -1                                        // 000000004E08: BE804EC1
	buffer_load_b128 v[120:123], v6, s[52:55], s81 offen offset:7168// 000000004E0C: C405C051 40806878 001C0006
	buffer_load_b128 v[124:127], v6, s[52:55], s82 offen offset:7168// 000000004E18: C405C052 4080687C 001C0006
	buffer_load_b32 v112, off, s[48:51], s73 offset:28         // 000000004E24: C4050049 00806070 00001C00
	buffer_load_b32 v116, v3, s[48:51], s69 offen offset:896   // 000000004E30: C4050045 40806074 00038003
	buffer_load_b32 v113, off, s[48:51], s74 offset:28         // 000000004E3C: C405004A 00806071 00001C00
	buffer_load_b32 v117, v3, s[48:51], s70 offen offset:896   // 000000004E48: C4050046 40806075 00038003
	buffer_load_b32 v114, off, s[48:51], s75 offset:28         // 000000004E54: C405004B 00806072 00001C00
	buffer_load_b32 v118, v3, s[48:51], s71 offen offset:896   // 000000004E60: C4050047 40806076 00038003
	buffer_load_b32 v115, off, s[48:51], s76 offset:28         // 000000004E6C: C405004C 00806073 00001C00
	buffer_load_b32 v119, v3, s[48:51], s72 offen offset:896   // 000000004E78: C4050048 40806077 00038003
	s_wait_loadcnt 0x11                                        // 000000004E84: BFC00011
	v_lshrrev_b32_e32 v28, v2, v96                             // 000000004E88: 3238C102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004E8C: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000004E90: CC20001D 8A007F1C
	s_wait_loadcnt 0x10                                        // 000000004E98: BFC00010
	v_and_b32_e32 v30, 0xf0f0f0f, v100                         // 000000004E9C: 363CC8FF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v100                             // 000000004EA4: 323EC884
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004EA8: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000004EAC: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004EB4: BF870003
	v_cvt_f32_ubyte0_e32 v128, v30                             // 000000004EB8: 7F00231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004EBC: BF870002
	v_cvt_f32_ubyte0_e32 v129, v31                             // 000000004EC0: 7F02231F
	v_cvt_f32_ubyte1_e32 v130, v30                             // 000000004EC4: 7F04251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004EC8: BF870004
	v_cvt_f32_ubyte1_e32 v131, v31                             // 000000004ECC: 7F06251F
	v_cvt_f32_ubyte2_e32 v132, v30                             // 000000004ED0: 7F08271E
	v_cvt_f32_ubyte2_e32 v133, v31                             // 000000004ED4: 7F0A271F
	v_cvt_f32_ubyte3_e32 v134, v30                             // 000000004ED8: 7F0C291E
	v_cvt_f32_ubyte3_e32 v135, v31                             // 000000004EDC: 7F0E291F
	v_fma_mix_f32 v128, v28, v128, v29 op_sel_hi:[1,0,0]       // 000000004EE0: CC200080 0C77011C
	v_fma_mix_f32 v129, v28, v129, v29 op_sel_hi:[1,0,0]       // 000000004EE8: CC200081 0C77031C
	v_fma_mix_f32 v130, v28, v130, v29 op_sel_hi:[1,0,0]       // 000000004EF0: CC200082 0C77051C
	v_fma_mix_f32 v131, v28, v131, v29 op_sel_hi:[1,0,0]       // 000000004EF8: CC200083 0C77071C
	v_fma_mix_f32 v132, v28, v132, v29 op_sel_hi:[1,0,0]       // 000000004F00: CC200084 0C77091C
	v_fma_mix_f32 v133, v28, v133, v29 op_sel_hi:[1,0,0]       // 000000004F08: CC200085 0C770B1C
	v_fma_mix_f32 v134, v28, v134, v29 op_sel_hi:[1,0,0]       // 000000004F10: CC200086 0C770D1C
	v_fma_mix_f32 v135, v28, v135, v29 op_sel_hi:[1,0,0]       // 000000004F18: CC200087 0C770F1C
	s_wait_loadcnt 0xf                                         // 000000004F20: BFC0000F
	v_lshrrev_b32_e32 v28, v2, v97                             // 000000004F24: 3238C302
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004F28: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000004F2C: CC20001D 8A007F1C
	s_wait_loadcnt 0xe                                         // 000000004F34: BFC0000E
	v_and_b32_e32 v30, 0xf0f0f0f, v101                         // 000000004F38: 363CCAFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v101                             // 000000004F40: 323ECA84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004F44: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000004F48: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004F50: BF870003
	v_cvt_f32_ubyte0_e32 v136, v30                             // 000000004F54: 7F10231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004F58: BF870002
	v_cvt_f32_ubyte0_e32 v137, v31                             // 000000004F5C: 7F12231F
	v_cvt_f32_ubyte1_e32 v138, v30                             // 000000004F60: 7F14251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004F64: BF870004
	v_cvt_f32_ubyte1_e32 v139, v31                             // 000000004F68: 7F16251F
	v_cvt_f32_ubyte2_e32 v140, v30                             // 000000004F6C: 7F18271E
	v_cvt_f32_ubyte2_e32 v141, v31                             // 000000004F70: 7F1A271F
	v_cvt_f32_ubyte3_e32 v142, v30                             // 000000004F74: 7F1C291E
	v_cvt_f32_ubyte3_e32 v143, v31                             // 000000004F78: 7F1E291F
	v_fma_mix_f32 v136, v28, v136, v29 op_sel_hi:[1,0,0]       // 000000004F7C: CC200088 0C77111C
	v_fma_mix_f32 v137, v28, v137, v29 op_sel_hi:[1,0,0]       // 000000004F84: CC200089 0C77131C
	v_fma_mix_f32 v138, v28, v138, v29 op_sel_hi:[1,0,0]       // 000000004F8C: CC20008A 0C77151C
	v_fma_mix_f32 v139, v28, v139, v29 op_sel_hi:[1,0,0]       // 000000004F94: CC20008B 0C77171C
	v_fma_mix_f32 v140, v28, v140, v29 op_sel_hi:[1,0,0]       // 000000004F9C: CC20008C 0C77191C
	v_fma_mix_f32 v141, v28, v141, v29 op_sel_hi:[1,0,0]       // 000000004FA4: CC20008D 0C771B1C
	v_fma_mix_f32 v142, v28, v142, v29 op_sel_hi:[1,0,0]       // 000000004FAC: CC20008E 0C771D1C
	v_fma_mix_f32 v143, v28, v143, v29 op_sel_hi:[1,0,0]       // 000000004FB4: CC20008F 0C771F1C
	s_wait_loadcnt 0xd                                         // 000000004FBC: BFC0000D
	v_lshrrev_b32_e32 v28, v2, v98                             // 000000004FC0: 3238C502
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004FC4: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000004FC8: CC20001D 8A007F1C
	s_wait_loadcnt 0xc                                         // 000000004FD0: BFC0000C
	v_and_b32_e32 v30, 0xf0f0f0f, v102                         // 000000004FD4: 363CCCFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v102                             // 000000004FDC: 323ECC84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004FE0: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000004FE4: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004FEC: BF870003
	v_cvt_f32_ubyte0_e32 v144, v30                             // 000000004FF0: 7F20231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004FF4: BF870002
	v_cvt_f32_ubyte0_e32 v145, v31                             // 000000004FF8: 7F22231F
	v_cvt_f32_ubyte1_e32 v146, v30                             // 000000004FFC: 7F24251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005000: BF870004
	v_cvt_f32_ubyte1_e32 v147, v31                             // 000000005004: 7F26251F
	v_cvt_f32_ubyte2_e32 v148, v30                             // 000000005008: 7F28271E
	v_cvt_f32_ubyte2_e32 v149, v31                             // 00000000500C: 7F2A271F
	v_cvt_f32_ubyte3_e32 v150, v30                             // 000000005010: 7F2C291E
	v_cvt_f32_ubyte3_e32 v151, v31                             // 000000005014: 7F2E291F
	v_fma_mix_f32 v144, v28, v144, v29 op_sel_hi:[1,0,0]       // 000000005018: CC200090 0C77211C
	v_fma_mix_f32 v145, v28, v145, v29 op_sel_hi:[1,0,0]       // 000000005020: CC200091 0C77231C
	v_fma_mix_f32 v146, v28, v146, v29 op_sel_hi:[1,0,0]       // 000000005028: CC200092 0C77251C
	v_fma_mix_f32 v147, v28, v147, v29 op_sel_hi:[1,0,0]       // 000000005030: CC200093 0C77271C
	v_fma_mix_f32 v148, v28, v148, v29 op_sel_hi:[1,0,0]       // 000000005038: CC200094 0C77291C
	v_fma_mix_f32 v149, v28, v149, v29 op_sel_hi:[1,0,0]       // 000000005040: CC200095 0C772B1C
	v_fma_mix_f32 v150, v28, v150, v29 op_sel_hi:[1,0,0]       // 000000005048: CC200096 0C772D1C
	v_fma_mix_f32 v151, v28, v151, v29 op_sel_hi:[1,0,0]       // 000000005050: CC200097 0C772F1C
	s_wait_loadcnt 0xb                                         // 000000005058: BFC0000B
	v_lshrrev_b32_e32 v28, v2, v99                             // 00000000505C: 3238C702
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005060: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000005064: CC20001D 8A007F1C
	s_wait_loadcnt 0xa                                         // 00000000506C: BFC0000A
	v_and_b32_e32 v30, 0xf0f0f0f, v103                         // 000000005070: 363CCEFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v103                             // 000000005078: 323ECE84
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000507C: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000005080: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005088: BF870003
	v_cvt_f32_ubyte0_e32 v152, v30                             // 00000000508C: 7F30231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000005090: BF870002
	v_cvt_f32_ubyte0_e32 v153, v31                             // 000000005094: 7F32231F
	v_cvt_f32_ubyte1_e32 v154, v30                             // 000000005098: 7F34251E
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000509C: BF870004
	v_cvt_f32_ubyte1_e32 v155, v31                             // 0000000050A0: 7F36251F
	v_cvt_f32_ubyte2_e32 v156, v30                             // 0000000050A4: 7F38271E
	v_cvt_f32_ubyte2_e32 v157, v31                             // 0000000050A8: 7F3A271F
	v_cvt_f32_ubyte3_e32 v158, v30                             // 0000000050AC: 7F3C291E
	v_cvt_f32_ubyte3_e32 v159, v31                             // 0000000050B0: 7F3E291F
	v_fma_mix_f32 v152, v28, v152, v29 op_sel_hi:[1,0,0]       // 0000000050B4: CC200098 0C77311C
	v_fma_mix_f32 v153, v28, v153, v29 op_sel_hi:[1,0,0]       // 0000000050BC: CC200099 0C77331C
	v_fma_mix_f32 v154, v28, v154, v29 op_sel_hi:[1,0,0]       // 0000000050C4: CC20009A 0C77351C
	v_fma_mix_f32 v155, v28, v155, v29 op_sel_hi:[1,0,0]       // 0000000050CC: CC20009B 0C77371C
	v_fma_mix_f32 v156, v28, v156, v29 op_sel_hi:[1,0,0]       // 0000000050D4: CC20009C 0C77391C
	v_fma_mix_f32 v157, v28, v157, v29 op_sel_hi:[1,0,0]       // 0000000050DC: CC20009D 0C773B1C
	v_fma_mix_f32 v158, v28, v158, v29 op_sel_hi:[1,0,0]       // 0000000050E4: CC20009E 0C773D1C
	v_fma_mix_f32 v159, v28, v159, v29 op_sel_hi:[1,0,0]       // 0000000050EC: CC20009F 0C773F1C
	s_barrier_wait 0xffff                                      // 0000000050F4: BF94FFFF
	s_bitcmp0_b32 s29, 0                                       // 0000000050F8: BF0C801D
	s_cbranch_scc1 74                                          // 0000000050FC: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x3928>
	ds_load_b128 v[16:19], v4                                  // 000000005100: DBFC0000 10000004
	ds_load_b128 v[20:23], v4 offset:16                        // 000000005108: DBFC0010 14000004
	s_wait_dscnt 0x1                                           // 000000005110: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000005114: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000005118: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 00000000511C: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000005120: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005124: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000005128: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000512C: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000005130: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005134: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000005138: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000513C: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000005140: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005144: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000005148: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000514C: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000005150: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005154: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000005158: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000515C: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000005160: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005164: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000005168: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000516C: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000005170: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005174: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000005178: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000517C: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000005180: 5636279B
	s_wait_dscnt 0x0                                           // 000000005184: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005188: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 00000000518C: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005190: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000005194: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005198: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 00000000519C: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051A0: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000051A4: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051A8: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000051AC: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051B0: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000051B4: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051B8: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000051BC: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051C0: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000051C4: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051C8: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000051CC: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051D0: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 0000000051D4: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051D8: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 0000000051DC: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051E0: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 0000000051E4: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051E8: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 0000000051EC: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051F0: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 0000000051F4: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000051F8: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 0000000051FC: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005200: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000005204: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005208: BF870004
	v_add_f32_e32 v64, v64, v24                                // 00000000520C: 06803140
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005210: BF870004
	v_add_f32_e32 v68, v68, v25                                // 000000005214: 06883344
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005218: BF870004
	v_add_f32_e32 v72, v72, v26                                // 00000000521C: 06903548
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005220: BF870004
	v_add_f32_e32 v76, v76, v27                                // 000000005224: 0698374C
	s_bitcmp0_b32 s29, 1                                       // 000000005228: BF0C811D
	s_cbranch_scc1 74                                          // 00000000522C: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x3a58>
	ds_load_b128 v[16:19], v4 offset:1040                      // 000000005230: DBFC0410 10000004
	ds_load_b128 v[20:23], v4 offset:1056                      // 000000005238: DBFC0420 14000004
	s_wait_dscnt 0x1                                           // 000000005240: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000005244: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000005248: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 00000000524C: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000005250: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005254: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000005258: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000525C: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000005260: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005264: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000005268: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000526C: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000005270: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005274: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000005278: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000527C: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000005280: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005284: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000005288: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000528C: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000005290: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005294: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000005298: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000529C: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 0000000052A0: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052A4: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 0000000052A8: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052AC: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 0000000052B0: 5636279B
	s_wait_dscnt 0x0                                           // 0000000052B4: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052B8: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 0000000052BC: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052C0: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 0000000052C4: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052C8: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000052CC: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052D0: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000052D4: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052D8: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000052DC: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052E0: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000052E4: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052E8: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000052EC: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052F0: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000052F4: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000052F8: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000052FC: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005300: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000005304: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005308: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 00000000530C: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005310: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000005314: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005318: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 00000000531C: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005320: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000005324: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005328: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 00000000532C: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005330: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000005334: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005338: BF870004
	v_add_f32_e32 v65, v65, v24                                // 00000000533C: 06823141
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005340: BF870004
	v_add_f32_e32 v69, v69, v25                                // 000000005344: 068A3345
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005348: BF870004
	v_add_f32_e32 v73, v73, v26                                // 00000000534C: 06923549
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005350: BF870004
	v_add_f32_e32 v77, v77, v27                                // 000000005354: 069A374D
	s_bitcmp0_b32 s29, 2                                       // 000000005358: BF0C821D
	s_cbranch_scc1 74                                          // 00000000535C: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x3b88>
	ds_load_b128 v[16:19], v4 offset:2080                      // 000000005360: DBFC0820 10000004
	ds_load_b128 v[20:23], v4 offset:2096                      // 000000005368: DBFC0830 14000004
	s_wait_dscnt 0x1                                           // 000000005370: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000005374: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000005378: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 00000000537C: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000005380: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005384: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000005388: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000538C: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000005390: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005394: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000005398: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000539C: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 0000000053A0: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053A4: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 0000000053A8: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053AC: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 0000000053B0: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053B4: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 0000000053B8: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053BC: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 0000000053C0: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053C4: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 0000000053C8: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053CC: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 0000000053D0: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053D4: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 0000000053D8: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053DC: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 0000000053E0: 5636279B
	s_wait_dscnt 0x0                                           // 0000000053E4: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053E8: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 0000000053EC: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053F0: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 0000000053F4: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000053F8: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000053FC: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005400: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000005404: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005408: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 00000000540C: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005410: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000005414: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005418: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 00000000541C: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005420: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000005424: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005428: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 00000000542C: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005430: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000005434: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005438: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 00000000543C: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005440: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000005444: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005448: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 00000000544C: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005450: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000005454: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005458: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 00000000545C: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005460: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000005464: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005468: BF870004
	v_add_f32_e32 v66, v66, v24                                // 00000000546C: 06843142
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005470: BF870004
	v_add_f32_e32 v70, v70, v25                                // 000000005474: 068C3346
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005478: BF870004
	v_add_f32_e32 v74, v74, v26                                // 00000000547C: 0694354A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005480: BF870004
	v_add_f32_e32 v78, v78, v27                                // 000000005484: 069C374E
	s_bitcmp0_b32 s29, 3                                       // 000000005488: BF0C831D
	s_cbranch_scc1 74                                          // 00000000548C: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x3cb8>
	ds_load_b128 v[16:19], v4 offset:3120                      // 000000005490: DBFC0C30 10000004
	ds_load_b128 v[20:23], v4 offset:3136                      // 000000005498: DBFC0C40 14000004
	s_wait_dscnt 0x1                                           // 0000000054A0: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 0000000054A4: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 0000000054A8: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000054AC: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 0000000054B0: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054B4: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 0000000054B8: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054BC: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 0000000054C0: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054C4: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 0000000054C8: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054CC: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 0000000054D0: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054D4: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 0000000054D8: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054DC: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 0000000054E0: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054E4: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 0000000054E8: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054EC: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 0000000054F0: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054F4: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 0000000054F8: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000054FC: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000005500: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005504: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000005508: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000550C: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000005510: 5636279B
	s_wait_dscnt 0x0                                           // 000000005514: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005518: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 00000000551C: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005520: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000005524: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005528: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 00000000552C: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005530: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000005534: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005538: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 00000000553C: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005540: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000005544: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005548: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 00000000554C: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005550: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000005554: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005558: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 00000000555C: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005560: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000005564: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005568: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 00000000556C: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005570: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000005574: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005578: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 00000000557C: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005580: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000005584: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005588: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 00000000558C: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005590: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000005594: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005598: BF870004
	v_add_f32_e32 v67, v67, v24                                // 00000000559C: 06863143
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000055A0: BF870004
	v_add_f32_e32 v71, v71, v25                                // 0000000055A4: 068E3347
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000055A8: BF870004
	v_add_f32_e32 v75, v75, v26                                // 0000000055AC: 0696354B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000055B0: BF870004
	v_add_f32_e32 v79, v79, v27                                // 0000000055B4: 069E374F
	s_barrier_signal -1                                        // 0000000055B8: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000055BC: BF94FFFF
	s_wait_loadcnt 0x9                                         // 0000000055C0: BFC00009
	ds_store_b128 v5, v[120:123]                               // 0000000055C4: DB7C0000 00007805
	s_wait_loadcnt 0x8                                         // 0000000055CC: BFC00008
	ds_store_b128 v5, v[124:127] offset:2080                   // 0000000055D0: DB7C0820 00007C05
	s_wait_dscnt 0x0                                           // 0000000055D8: BFC60000
	s_barrier_signal -1                                        // 0000000055DC: BE804EC1
	buffer_load_b128 v[104:107], v6, s[52:55], s81 offen offset:8192// 0000000055E0: C405C051 40806868 00200006
	buffer_load_b128 v[108:111], v6, s[52:55], s82 offen offset:8192// 0000000055EC: C405C052 4080686C 00200006
	buffer_load_b32 v96, off, s[48:51], s73 offset:32          // 0000000055F8: C4050049 00806060 00002000
	buffer_load_b32 v100, v3, s[48:51], s69 offen offset:1024  // 000000005604: C4050045 40806064 00040003
	buffer_load_b32 v97, off, s[48:51], s74 offset:32          // 000000005610: C405004A 00806061 00002000
	buffer_load_b32 v101, v3, s[48:51], s70 offen offset:1024  // 00000000561C: C4050046 40806065 00040003
	buffer_load_b32 v98, off, s[48:51], s75 offset:32          // 000000005628: C405004B 00806062 00002000
	buffer_load_b32 v102, v3, s[48:51], s71 offen offset:1024  // 000000005634: C4050047 40806066 00040003
	buffer_load_b32 v99, off, s[48:51], s76 offset:32          // 000000005640: C405004C 00806063 00002000
	buffer_load_b32 v103, v3, s[48:51], s72 offen offset:1024  // 00000000564C: C4050048 40806067 00040003
	s_wait_loadcnt 0x11                                        // 000000005658: BFC00011
	v_lshrrev_b32_e32 v28, v2, v112                            // 00000000565C: 3238E102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005660: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000005664: CC20001D 8A007F1C
	s_wait_loadcnt 0x10                                        // 00000000566C: BFC00010
	v_and_b32_e32 v30, 0xf0f0f0f, v116                         // 000000005670: 363CE8FF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v116                             // 000000005678: 323EE884
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000567C: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000005680: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005688: BF870003
	v_cvt_f32_ubyte0_e32 v128, v30                             // 00000000568C: 7F00231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000005690: BF870002
	v_cvt_f32_ubyte0_e32 v129, v31                             // 000000005694: 7F02231F
	v_cvt_f32_ubyte1_e32 v130, v30                             // 000000005698: 7F04251E
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000569C: BF870004
	v_cvt_f32_ubyte1_e32 v131, v31                             // 0000000056A0: 7F06251F
	v_cvt_f32_ubyte2_e32 v132, v30                             // 0000000056A4: 7F08271E
	v_cvt_f32_ubyte2_e32 v133, v31                             // 0000000056A8: 7F0A271F
	v_cvt_f32_ubyte3_e32 v134, v30                             // 0000000056AC: 7F0C291E
	v_cvt_f32_ubyte3_e32 v135, v31                             // 0000000056B0: 7F0E291F
	v_fma_mix_f32 v128, v28, v128, v29 op_sel_hi:[1,0,0]       // 0000000056B4: CC200080 0C77011C
	v_fma_mix_f32 v129, v28, v129, v29 op_sel_hi:[1,0,0]       // 0000000056BC: CC200081 0C77031C
	v_fma_mix_f32 v130, v28, v130, v29 op_sel_hi:[1,0,0]       // 0000000056C4: CC200082 0C77051C
	v_fma_mix_f32 v131, v28, v131, v29 op_sel_hi:[1,0,0]       // 0000000056CC: CC200083 0C77071C
	v_fma_mix_f32 v132, v28, v132, v29 op_sel_hi:[1,0,0]       // 0000000056D4: CC200084 0C77091C
	v_fma_mix_f32 v133, v28, v133, v29 op_sel_hi:[1,0,0]       // 0000000056DC: CC200085 0C770B1C
	v_fma_mix_f32 v134, v28, v134, v29 op_sel_hi:[1,0,0]       // 0000000056E4: CC200086 0C770D1C
	v_fma_mix_f32 v135, v28, v135, v29 op_sel_hi:[1,0,0]       // 0000000056EC: CC200087 0C770F1C
	s_wait_loadcnt 0xf                                         // 0000000056F4: BFC0000F
	v_lshrrev_b32_e32 v28, v2, v113                            // 0000000056F8: 3238E302
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000056FC: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000005700: CC20001D 8A007F1C
	s_wait_loadcnt 0xe                                         // 000000005708: BFC0000E
	v_and_b32_e32 v30, 0xf0f0f0f, v117                         // 00000000570C: 363CEAFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v117                             // 000000005714: 323EEA84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005718: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 00000000571C: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005724: BF870003
	v_cvt_f32_ubyte0_e32 v136, v30                             // 000000005728: 7F10231E
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000572C: BF870002
	v_cvt_f32_ubyte0_e32 v137, v31                             // 000000005730: 7F12231F
	v_cvt_f32_ubyte1_e32 v138, v30                             // 000000005734: 7F14251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005738: BF870004
	v_cvt_f32_ubyte1_e32 v139, v31                             // 00000000573C: 7F16251F
	v_cvt_f32_ubyte2_e32 v140, v30                             // 000000005740: 7F18271E
	v_cvt_f32_ubyte2_e32 v141, v31                             // 000000005744: 7F1A271F
	v_cvt_f32_ubyte3_e32 v142, v30                             // 000000005748: 7F1C291E
	v_cvt_f32_ubyte3_e32 v143, v31                             // 00000000574C: 7F1E291F
	v_fma_mix_f32 v136, v28, v136, v29 op_sel_hi:[1,0,0]       // 000000005750: CC200088 0C77111C
	v_fma_mix_f32 v137, v28, v137, v29 op_sel_hi:[1,0,0]       // 000000005758: CC200089 0C77131C
	v_fma_mix_f32 v138, v28, v138, v29 op_sel_hi:[1,0,0]       // 000000005760: CC20008A 0C77151C
	v_fma_mix_f32 v139, v28, v139, v29 op_sel_hi:[1,0,0]       // 000000005768: CC20008B 0C77171C
	v_fma_mix_f32 v140, v28, v140, v29 op_sel_hi:[1,0,0]       // 000000005770: CC20008C 0C77191C
	v_fma_mix_f32 v141, v28, v141, v29 op_sel_hi:[1,0,0]       // 000000005778: CC20008D 0C771B1C
	v_fma_mix_f32 v142, v28, v142, v29 op_sel_hi:[1,0,0]       // 000000005780: CC20008E 0C771D1C
	v_fma_mix_f32 v143, v28, v143, v29 op_sel_hi:[1,0,0]       // 000000005788: CC20008F 0C771F1C
	s_wait_loadcnt 0xd                                         // 000000005790: BFC0000D
	v_lshrrev_b32_e32 v28, v2, v114                            // 000000005794: 3238E502
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005798: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 00000000579C: CC20001D 8A007F1C
	s_wait_loadcnt 0xc                                         // 0000000057A4: BFC0000C
	v_and_b32_e32 v30, 0xf0f0f0f, v118                         // 0000000057A8: 363CECFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v118                             // 0000000057B0: 323EEC84
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000057B4: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 0000000057B8: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000057C0: BF870003
	v_cvt_f32_ubyte0_e32 v144, v30                             // 0000000057C4: 7F20231E
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000057C8: BF870002
	v_cvt_f32_ubyte0_e32 v145, v31                             // 0000000057CC: 7F22231F
	v_cvt_f32_ubyte1_e32 v146, v30                             // 0000000057D0: 7F24251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000057D4: BF870004
	v_cvt_f32_ubyte1_e32 v147, v31                             // 0000000057D8: 7F26251F
	v_cvt_f32_ubyte2_e32 v148, v30                             // 0000000057DC: 7F28271E
	v_cvt_f32_ubyte2_e32 v149, v31                             // 0000000057E0: 7F2A271F
	v_cvt_f32_ubyte3_e32 v150, v30                             // 0000000057E4: 7F2C291E
	v_cvt_f32_ubyte3_e32 v151, v31                             // 0000000057E8: 7F2E291F
	v_fma_mix_f32 v144, v28, v144, v29 op_sel_hi:[1,0,0]       // 0000000057EC: CC200090 0C77211C
	v_fma_mix_f32 v145, v28, v145, v29 op_sel_hi:[1,0,0]       // 0000000057F4: CC200091 0C77231C
	v_fma_mix_f32 v146, v28, v146, v29 op_sel_hi:[1,0,0]       // 0000000057FC: CC200092 0C77251C
	v_fma_mix_f32 v147, v28, v147, v29 op_sel_hi:[1,0,0]       // 000000005804: CC200093 0C77271C
	v_fma_mix_f32 v148, v28, v148, v29 op_sel_hi:[1,0,0]       // 00000000580C: CC200094 0C77291C
	v_fma_mix_f32 v149, v28, v149, v29 op_sel_hi:[1,0,0]       // 000000005814: CC200095 0C772B1C
	v_fma_mix_f32 v150, v28, v150, v29 op_sel_hi:[1,0,0]       // 00000000581C: CC200096 0C772D1C
	v_fma_mix_f32 v151, v28, v151, v29 op_sel_hi:[1,0,0]       // 000000005824: CC200097 0C772F1C
	s_wait_loadcnt 0xb                                         // 00000000582C: BFC0000B
	v_lshrrev_b32_e32 v28, v2, v115                            // 000000005830: 3238E702
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005834: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000005838: CC20001D 8A007F1C
	s_wait_loadcnt 0xa                                         // 000000005840: BFC0000A
	v_and_b32_e32 v30, 0xf0f0f0f, v119                         // 000000005844: 363CEEFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v119                             // 00000000584C: 323EEE84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005850: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000005854: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000585C: BF870003
	v_cvt_f32_ubyte0_e32 v152, v30                             // 000000005860: 7F30231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000005864: BF870002
	v_cvt_f32_ubyte0_e32 v153, v31                             // 000000005868: 7F32231F
	v_cvt_f32_ubyte1_e32 v154, v30                             // 00000000586C: 7F34251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005870: BF870004
	v_cvt_f32_ubyte1_e32 v155, v31                             // 000000005874: 7F36251F
	v_cvt_f32_ubyte2_e32 v156, v30                             // 000000005878: 7F38271E
	v_cvt_f32_ubyte2_e32 v157, v31                             // 00000000587C: 7F3A271F
	v_cvt_f32_ubyte3_e32 v158, v30                             // 000000005880: 7F3C291E
	v_cvt_f32_ubyte3_e32 v159, v31                             // 000000005884: 7F3E291F
	v_fma_mix_f32 v152, v28, v152, v29 op_sel_hi:[1,0,0]       // 000000005888: CC200098 0C77311C
	v_fma_mix_f32 v153, v28, v153, v29 op_sel_hi:[1,0,0]       // 000000005890: CC200099 0C77331C
	v_fma_mix_f32 v154, v28, v154, v29 op_sel_hi:[1,0,0]       // 000000005898: CC20009A 0C77351C
	v_fma_mix_f32 v155, v28, v155, v29 op_sel_hi:[1,0,0]       // 0000000058A0: CC20009B 0C77371C
	v_fma_mix_f32 v156, v28, v156, v29 op_sel_hi:[1,0,0]       // 0000000058A8: CC20009C 0C77391C
	v_fma_mix_f32 v157, v28, v157, v29 op_sel_hi:[1,0,0]       // 0000000058B0: CC20009D 0C773B1C
	v_fma_mix_f32 v158, v28, v158, v29 op_sel_hi:[1,0,0]       // 0000000058B8: CC20009E 0C773D1C
	v_fma_mix_f32 v159, v28, v159, v29 op_sel_hi:[1,0,0]       // 0000000058C0: CC20009F 0C773F1C
	s_barrier_wait 0xffff                                      // 0000000058C8: BF94FFFF
	s_bitcmp0_b32 s29, 0                                       // 0000000058CC: BF0C801D
	s_cbranch_scc1 74                                          // 0000000058D0: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x40fc>
	ds_load_b128 v[16:19], v4                                  // 0000000058D4: DBFC0000 10000004
	ds_load_b128 v[20:23], v4 offset:16                        // 0000000058DC: DBFC0010 14000004
	s_wait_dscnt 0x1                                           // 0000000058E4: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 0000000058E8: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 0000000058EC: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000058F0: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 0000000058F4: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000058F8: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 0000000058FC: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005900: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000005904: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005908: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 00000000590C: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005910: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000005914: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005918: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 00000000591C: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005920: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000005924: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005928: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 00000000592C: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005930: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000005934: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005938: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 00000000593C: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005940: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000005944: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005948: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 00000000594C: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005950: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000005954: 5636279B
	s_wait_dscnt 0x0                                           // 000000005958: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000595C: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000005960: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005964: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000005968: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000596C: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000005970: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005974: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000005978: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000597C: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000005980: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005984: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000005988: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000598C: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000005990: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005994: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000005998: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000599C: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000059A0: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059A4: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 0000000059A8: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059AC: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 0000000059B0: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059B4: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 0000000059B8: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059BC: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 0000000059C0: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059C4: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 0000000059C8: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059CC: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 0000000059D0: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059D4: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 0000000059D8: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059DC: BF870004
	v_add_f32_e32 v80, v80, v24                                // 0000000059E0: 06A03150
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059E4: BF870004
	v_add_f32_e32 v84, v84, v25                                // 0000000059E8: 06A83354
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059EC: BF870004
	v_add_f32_e32 v88, v88, v26                                // 0000000059F0: 06B03558
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000059F4: BF870004
	v_add_f32_e32 v92, v92, v27                                // 0000000059F8: 06B8375C
	s_bitcmp0_b32 s29, 1                                       // 0000000059FC: BF0C811D
	s_cbranch_scc1 74                                          // 000000005A00: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x422c>
	ds_load_b128 v[16:19], v4 offset:1040                      // 000000005A04: DBFC0410 10000004
	ds_load_b128 v[20:23], v4 offset:1056                      // 000000005A0C: DBFC0420 14000004
	s_wait_dscnt 0x1                                           // 000000005A14: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000005A18: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000005A1C: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000005A20: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000005A24: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A28: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000005A2C: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A30: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000005A34: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A38: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000005A3C: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A40: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000005A44: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A48: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000005A4C: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A50: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000005A54: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A58: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000005A5C: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A60: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000005A64: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A68: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000005A6C: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A70: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000005A74: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A78: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000005A7C: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A80: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000005A84: 5636279B
	s_wait_dscnt 0x0                                           // 000000005A88: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A8C: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000005A90: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A94: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000005A98: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005A9C: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000005AA0: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005AA4: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000005AA8: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005AAC: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000005AB0: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005AB4: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000005AB8: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005ABC: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000005AC0: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005AC4: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000005AC8: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005ACC: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000005AD0: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005AD4: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000005AD8: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005ADC: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000005AE0: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005AE4: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000005AE8: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005AEC: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000005AF0: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005AF4: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000005AF8: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005AFC: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000005B00: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B04: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000005B08: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B0C: BF870004
	v_add_f32_e32 v81, v81, v24                                // 000000005B10: 06A23151
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B14: BF870004
	v_add_f32_e32 v85, v85, v25                                // 000000005B18: 06AA3355
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B1C: BF870004
	v_add_f32_e32 v89, v89, v26                                // 000000005B20: 06B23559
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B24: BF870004
	v_add_f32_e32 v93, v93, v27                                // 000000005B28: 06BA375D
	s_bitcmp0_b32 s29, 2                                       // 000000005B2C: BF0C821D
	s_cbranch_scc1 74                                          // 000000005B30: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x435c>
	ds_load_b128 v[16:19], v4 offset:2080                      // 000000005B34: DBFC0820 10000004
	ds_load_b128 v[20:23], v4 offset:2096                      // 000000005B3C: DBFC0830 14000004
	s_wait_dscnt 0x1                                           // 000000005B44: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000005B48: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000005B4C: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000005B50: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000005B54: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B58: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000005B5C: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B60: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000005B64: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B68: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000005B6C: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B70: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000005B74: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B78: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000005B7C: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B80: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000005B84: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B88: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000005B8C: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B90: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000005B94: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005B98: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000005B9C: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BA0: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000005BA4: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BA8: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000005BAC: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BB0: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000005BB4: 5636279B
	s_wait_dscnt 0x0                                           // 000000005BB8: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BBC: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000005BC0: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BC4: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000005BC8: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BCC: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000005BD0: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BD4: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000005BD8: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BDC: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000005BE0: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BE4: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000005BE8: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BEC: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000005BF0: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BF4: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000005BF8: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005BFC: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000005C00: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C04: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000005C08: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C0C: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000005C10: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C14: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000005C18: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C1C: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000005C20: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C24: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000005C28: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C2C: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000005C30: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C34: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000005C38: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C3C: BF870004
	v_add_f32_e32 v82, v82, v24                                // 000000005C40: 06A43152
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C44: BF870004
	v_add_f32_e32 v86, v86, v25                                // 000000005C48: 06AC3356
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C4C: BF870004
	v_add_f32_e32 v90, v90, v26                                // 000000005C50: 06B4355A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C54: BF870004
	v_add_f32_e32 v94, v94, v27                                // 000000005C58: 06BC375E
	s_bitcmp0_b32 s29, 3                                       // 000000005C5C: BF0C831D
	s_cbranch_scc1 74                                          // 000000005C60: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x448c>
	ds_load_b128 v[16:19], v4 offset:3120                      // 000000005C64: DBFC0C30 10000004
	ds_load_b128 v[20:23], v4 offset:3136                      // 000000005C6C: DBFC0C40 14000004
	s_wait_dscnt 0x1                                           // 000000005C74: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000005C78: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000005C7C: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000005C80: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000005C84: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C88: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000005C8C: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C90: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000005C94: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C98: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000005C9C: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CA0: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000005CA4: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CA8: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000005CAC: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CB0: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000005CB4: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CB8: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000005CBC: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CC0: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000005CC4: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CC8: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000005CCC: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CD0: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000005CD4: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CD8: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000005CDC: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CE0: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000005CE4: 5636279B
	s_wait_dscnt 0x0                                           // 000000005CE8: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CEC: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000005CF0: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CF4: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000005CF8: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005CFC: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000005D00: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D04: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000005D08: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D0C: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000005D10: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D14: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000005D18: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D1C: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000005D20: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D24: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000005D28: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D2C: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000005D30: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D34: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000005D38: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D3C: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000005D40: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D44: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000005D48: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D4C: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000005D50: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D54: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000005D58: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D5C: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000005D60: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D64: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000005D68: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D6C: BF870004
	v_add_f32_e32 v83, v83, v24                                // 000000005D70: 06A63153
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D74: BF870004
	v_add_f32_e32 v87, v87, v25                                // 000000005D78: 06AE3357
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D7C: BF870004
	v_add_f32_e32 v91, v91, v26                                // 000000005D80: 06B6355B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005D84: BF870004
	v_add_f32_e32 v95, v95, v27                                // 000000005D88: 06BE375F
	s_barrier_signal -1                                        // 000000005D8C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000005D90: BF94FFFF
	s_wait_loadcnt 0x9                                         // 000000005D94: BFC00009
	ds_store_b128 v5, v[104:107]                               // 000000005D98: DB7C0000 00006805
	s_wait_loadcnt 0x8                                         // 000000005DA0: BFC00008
	ds_store_b128 v5, v[108:111] offset:2080                   // 000000005DA4: DB7C0820 00006C05
	s_wait_dscnt 0x0                                           // 000000005DAC: BFC60000
	s_barrier_signal -1                                        // 000000005DB0: BE804EC1
	buffer_load_b128 v[120:123], v6, s[52:55], s81 offen offset:9216// 000000005DB4: C405C051 40806878 00240006
	buffer_load_b128 v[124:127], v6, s[52:55], s82 offen offset:9216// 000000005DC0: C405C052 4080687C 00240006
	buffer_load_b32 v112, off, s[48:51], s73 offset:36         // 000000005DCC: C4050049 00806070 00002400
	buffer_load_b32 v116, v3, s[48:51], s69 offen offset:1152  // 000000005DD8: C4050045 40806074 00048003
	buffer_load_b32 v113, off, s[48:51], s74 offset:36         // 000000005DE4: C405004A 00806071 00002400
	buffer_load_b32 v117, v3, s[48:51], s70 offen offset:1152  // 000000005DF0: C4050046 40806075 00048003
	buffer_load_b32 v114, off, s[48:51], s75 offset:36         // 000000005DFC: C405004B 00806072 00002400
	buffer_load_b32 v118, v3, s[48:51], s71 offen offset:1152  // 000000005E08: C4050047 40806076 00048003
	buffer_load_b32 v115, off, s[48:51], s76 offset:36         // 000000005E14: C405004C 00806073 00002400
	buffer_load_b32 v119, v3, s[48:51], s72 offen offset:1152  // 000000005E20: C4050048 40806077 00048003
	s_wait_loadcnt 0x11                                        // 000000005E2C: BFC00011
	v_lshrrev_b32_e32 v28, v2, v96                             // 000000005E30: 3238C102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005E34: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000005E38: CC20001D 8A007F1C
	s_wait_loadcnt 0x10                                        // 000000005E40: BFC00010
	v_and_b32_e32 v30, 0xf0f0f0f, v100                         // 000000005E44: 363CC8FF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v100                             // 000000005E4C: 323EC884
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005E50: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000005E54: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005E5C: BF870003
	v_cvt_f32_ubyte0_e32 v128, v30                             // 000000005E60: 7F00231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000005E64: BF870002
	v_cvt_f32_ubyte0_e32 v129, v31                             // 000000005E68: 7F02231F
	v_cvt_f32_ubyte1_e32 v130, v30                             // 000000005E6C: 7F04251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005E70: BF870004
	v_cvt_f32_ubyte1_e32 v131, v31                             // 000000005E74: 7F06251F
	v_cvt_f32_ubyte2_e32 v132, v30                             // 000000005E78: 7F08271E
	v_cvt_f32_ubyte2_e32 v133, v31                             // 000000005E7C: 7F0A271F
	v_cvt_f32_ubyte3_e32 v134, v30                             // 000000005E80: 7F0C291E
	v_cvt_f32_ubyte3_e32 v135, v31                             // 000000005E84: 7F0E291F
	v_fma_mix_f32 v128, v28, v128, v29 op_sel_hi:[1,0,0]       // 000000005E88: CC200080 0C77011C
	v_fma_mix_f32 v129, v28, v129, v29 op_sel_hi:[1,0,0]       // 000000005E90: CC200081 0C77031C
	v_fma_mix_f32 v130, v28, v130, v29 op_sel_hi:[1,0,0]       // 000000005E98: CC200082 0C77051C
	v_fma_mix_f32 v131, v28, v131, v29 op_sel_hi:[1,0,0]       // 000000005EA0: CC200083 0C77071C
	v_fma_mix_f32 v132, v28, v132, v29 op_sel_hi:[1,0,0]       // 000000005EA8: CC200084 0C77091C
	v_fma_mix_f32 v133, v28, v133, v29 op_sel_hi:[1,0,0]       // 000000005EB0: CC200085 0C770B1C
	v_fma_mix_f32 v134, v28, v134, v29 op_sel_hi:[1,0,0]       // 000000005EB8: CC200086 0C770D1C
	v_fma_mix_f32 v135, v28, v135, v29 op_sel_hi:[1,0,0]       // 000000005EC0: CC200087 0C770F1C
	s_wait_loadcnt 0xf                                         // 000000005EC8: BFC0000F
	v_lshrrev_b32_e32 v28, v2, v97                             // 000000005ECC: 3238C302
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005ED0: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000005ED4: CC20001D 8A007F1C
	s_wait_loadcnt 0xe                                         // 000000005EDC: BFC0000E
	v_and_b32_e32 v30, 0xf0f0f0f, v101                         // 000000005EE0: 363CCAFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v101                             // 000000005EE8: 323ECA84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005EEC: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000005EF0: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005EF8: BF870003
	v_cvt_f32_ubyte0_e32 v136, v30                             // 000000005EFC: 7F10231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000005F00: BF870002
	v_cvt_f32_ubyte0_e32 v137, v31                             // 000000005F04: 7F12231F
	v_cvt_f32_ubyte1_e32 v138, v30                             // 000000005F08: 7F14251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005F0C: BF870004
	v_cvt_f32_ubyte1_e32 v139, v31                             // 000000005F10: 7F16251F
	v_cvt_f32_ubyte2_e32 v140, v30                             // 000000005F14: 7F18271E
	v_cvt_f32_ubyte2_e32 v141, v31                             // 000000005F18: 7F1A271F
	v_cvt_f32_ubyte3_e32 v142, v30                             // 000000005F1C: 7F1C291E
	v_cvt_f32_ubyte3_e32 v143, v31                             // 000000005F20: 7F1E291F
	v_fma_mix_f32 v136, v28, v136, v29 op_sel_hi:[1,0,0]       // 000000005F24: CC200088 0C77111C
	v_fma_mix_f32 v137, v28, v137, v29 op_sel_hi:[1,0,0]       // 000000005F2C: CC200089 0C77131C
	v_fma_mix_f32 v138, v28, v138, v29 op_sel_hi:[1,0,0]       // 000000005F34: CC20008A 0C77151C
	v_fma_mix_f32 v139, v28, v139, v29 op_sel_hi:[1,0,0]       // 000000005F3C: CC20008B 0C77171C
	v_fma_mix_f32 v140, v28, v140, v29 op_sel_hi:[1,0,0]       // 000000005F44: CC20008C 0C77191C
	v_fma_mix_f32 v141, v28, v141, v29 op_sel_hi:[1,0,0]       // 000000005F4C: CC20008D 0C771B1C
	v_fma_mix_f32 v142, v28, v142, v29 op_sel_hi:[1,0,0]       // 000000005F54: CC20008E 0C771D1C
	v_fma_mix_f32 v143, v28, v143, v29 op_sel_hi:[1,0,0]       // 000000005F5C: CC20008F 0C771F1C
	s_wait_loadcnt 0xd                                         // 000000005F64: BFC0000D
	v_lshrrev_b32_e32 v28, v2, v98                             // 000000005F68: 3238C502
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005F6C: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000005F70: CC20001D 8A007F1C
	s_wait_loadcnt 0xc                                         // 000000005F78: BFC0000C
	v_and_b32_e32 v30, 0xf0f0f0f, v102                         // 000000005F7C: 363CCCFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v102                             // 000000005F84: 323ECC84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005F88: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000005F8C: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005F94: BF870003
	v_cvt_f32_ubyte0_e32 v144, v30                             // 000000005F98: 7F20231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000005F9C: BF870002
	v_cvt_f32_ubyte0_e32 v145, v31                             // 000000005FA0: 7F22231F
	v_cvt_f32_ubyte1_e32 v146, v30                             // 000000005FA4: 7F24251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005FA8: BF870004
	v_cvt_f32_ubyte1_e32 v147, v31                             // 000000005FAC: 7F26251F
	v_cvt_f32_ubyte2_e32 v148, v30                             // 000000005FB0: 7F28271E
	v_cvt_f32_ubyte2_e32 v149, v31                             // 000000005FB4: 7F2A271F
	v_cvt_f32_ubyte3_e32 v150, v30                             // 000000005FB8: 7F2C291E
	v_cvt_f32_ubyte3_e32 v151, v31                             // 000000005FBC: 7F2E291F
	v_fma_mix_f32 v144, v28, v144, v29 op_sel_hi:[1,0,0]       // 000000005FC0: CC200090 0C77211C
	v_fma_mix_f32 v145, v28, v145, v29 op_sel_hi:[1,0,0]       // 000000005FC8: CC200091 0C77231C
	v_fma_mix_f32 v146, v28, v146, v29 op_sel_hi:[1,0,0]       // 000000005FD0: CC200092 0C77251C
	v_fma_mix_f32 v147, v28, v147, v29 op_sel_hi:[1,0,0]       // 000000005FD8: CC200093 0C77271C
	v_fma_mix_f32 v148, v28, v148, v29 op_sel_hi:[1,0,0]       // 000000005FE0: CC200094 0C77291C
	v_fma_mix_f32 v149, v28, v149, v29 op_sel_hi:[1,0,0]       // 000000005FE8: CC200095 0C772B1C
	v_fma_mix_f32 v150, v28, v150, v29 op_sel_hi:[1,0,0]       // 000000005FF0: CC200096 0C772D1C
	v_fma_mix_f32 v151, v28, v151, v29 op_sel_hi:[1,0,0]       // 000000005FF8: CC200097 0C772F1C
	s_wait_loadcnt 0xb                                         // 000000006000: BFC0000B
	v_lshrrev_b32_e32 v28, v2, v99                             // 000000006004: 3238C702
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006008: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 00000000600C: CC20001D 8A007F1C
	s_wait_loadcnt 0xa                                         // 000000006014: BFC0000A
	v_and_b32_e32 v30, 0xf0f0f0f, v103                         // 000000006018: 363CCEFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v103                             // 000000006020: 323ECE84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006024: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000006028: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000006030: BF870003
	v_cvt_f32_ubyte0_e32 v152, v30                             // 000000006034: 7F30231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000006038: BF870002
	v_cvt_f32_ubyte0_e32 v153, v31                             // 00000000603C: 7F32231F
	v_cvt_f32_ubyte1_e32 v154, v30                             // 000000006040: 7F34251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006044: BF870004
	v_cvt_f32_ubyte1_e32 v155, v31                             // 000000006048: 7F36251F
	v_cvt_f32_ubyte2_e32 v156, v30                             // 00000000604C: 7F38271E
	v_cvt_f32_ubyte2_e32 v157, v31                             // 000000006050: 7F3A271F
	v_cvt_f32_ubyte3_e32 v158, v30                             // 000000006054: 7F3C291E
	v_cvt_f32_ubyte3_e32 v159, v31                             // 000000006058: 7F3E291F
	v_fma_mix_f32 v152, v28, v152, v29 op_sel_hi:[1,0,0]       // 00000000605C: CC200098 0C77311C
	v_fma_mix_f32 v153, v28, v153, v29 op_sel_hi:[1,0,0]       // 000000006064: CC200099 0C77331C
	v_fma_mix_f32 v154, v28, v154, v29 op_sel_hi:[1,0,0]       // 00000000606C: CC20009A 0C77351C
	v_fma_mix_f32 v155, v28, v155, v29 op_sel_hi:[1,0,0]       // 000000006074: CC20009B 0C77371C
	v_fma_mix_f32 v156, v28, v156, v29 op_sel_hi:[1,0,0]       // 00000000607C: CC20009C 0C77391C
	v_fma_mix_f32 v157, v28, v157, v29 op_sel_hi:[1,0,0]       // 000000006084: CC20009D 0C773B1C
	v_fma_mix_f32 v158, v28, v158, v29 op_sel_hi:[1,0,0]       // 00000000608C: CC20009E 0C773D1C
	v_fma_mix_f32 v159, v28, v159, v29 op_sel_hi:[1,0,0]       // 000000006094: CC20009F 0C773F1C
	s_barrier_wait 0xffff                                      // 00000000609C: BF94FFFF
	s_bitcmp0_b32 s29, 0                                       // 0000000060A0: BF0C801D
	s_cbranch_scc1 74                                          // 0000000060A4: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x48d0>
	ds_load_b128 v[16:19], v4                                  // 0000000060A8: DBFC0000 10000004
	ds_load_b128 v[20:23], v4 offset:16                        // 0000000060B0: DBFC0010 14000004
	s_wait_dscnt 0x1                                           // 0000000060B8: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 0000000060BC: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 0000000060C0: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000060C4: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 0000000060C8: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000060CC: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 0000000060D0: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000060D4: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 0000000060D8: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000060DC: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 0000000060E0: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000060E4: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 0000000060E8: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000060EC: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 0000000060F0: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000060F4: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 0000000060F8: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000060FC: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000006100: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006104: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000006108: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000610C: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000006110: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006114: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000006118: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000611C: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000006120: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006124: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000006128: 5636279B
	s_wait_dscnt 0x0                                           // 00000000612C: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006130: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000006134: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006138: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 00000000613C: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006140: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000006144: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006148: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 00000000614C: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006150: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000006154: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006158: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 00000000615C: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006160: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000006164: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006168: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 00000000616C: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006170: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000006174: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006178: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 00000000617C: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006180: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000006184: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006188: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 00000000618C: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006190: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000006194: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006198: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 00000000619C: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000061A0: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 0000000061A4: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000061A8: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 0000000061AC: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000061B0: BF870004
	v_add_f32_e32 v32, v32, v24                                // 0000000061B4: 06403120
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000061B8: BF870004
	v_add_f32_e32 v36, v36, v25                                // 0000000061BC: 06483324
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000061C0: BF870004
	v_add_f32_e32 v40, v40, v26                                // 0000000061C4: 06503528
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000061C8: BF870004
	v_add_f32_e32 v44, v44, v27                                // 0000000061CC: 0658372C
	s_bitcmp0_b32 s29, 1                                       // 0000000061D0: BF0C811D
	s_cbranch_scc1 74                                          // 0000000061D4: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x4a00>
	ds_load_b128 v[16:19], v4 offset:1040                      // 0000000061D8: DBFC0410 10000004
	ds_load_b128 v[20:23], v4 offset:1056                      // 0000000061E0: DBFC0420 14000004
	s_wait_dscnt 0x1                                           // 0000000061E8: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 0000000061EC: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 0000000061F0: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 0000000061F4: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 0000000061F8: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000061FC: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000006200: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006204: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000006208: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000620C: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000006210: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006214: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000006218: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000621C: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000006220: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006224: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000006228: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000622C: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000006230: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006234: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000006238: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000623C: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000006240: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006244: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000006248: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000624C: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000006250: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006254: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000006258: 5636279B
	s_wait_dscnt 0x0                                           // 00000000625C: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006260: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000006264: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006268: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 00000000626C: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006270: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000006274: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006278: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 00000000627C: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006280: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000006284: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006288: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 00000000628C: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006290: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000006294: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006298: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 00000000629C: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062A0: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000062A4: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062A8: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 0000000062AC: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062B0: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 0000000062B4: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062B8: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 0000000062BC: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062C0: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 0000000062C4: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062C8: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 0000000062CC: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062D0: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 0000000062D4: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062D8: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 0000000062DC: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062E0: BF870004
	v_add_f32_e32 v33, v33, v24                                // 0000000062E4: 06423121
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062E8: BF870004
	v_add_f32_e32 v37, v37, v25                                // 0000000062EC: 064A3325
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062F0: BF870004
	v_add_f32_e32 v41, v41, v26                                // 0000000062F4: 06523529
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000062F8: BF870004
	v_add_f32_e32 v45, v45, v27                                // 0000000062FC: 065A372D
	s_bitcmp0_b32 s29, 2                                       // 000000006300: BF0C821D
	s_cbranch_scc1 74                                          // 000000006304: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x4b30>
	ds_load_b128 v[16:19], v4 offset:2080                      // 000000006308: DBFC0820 10000004
	ds_load_b128 v[20:23], v4 offset:2096                      // 000000006310: DBFC0830 14000004
	s_wait_dscnt 0x1                                           // 000000006318: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 00000000631C: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000006320: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000006324: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000006328: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000632C: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000006330: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006334: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000006338: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000633C: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000006340: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006344: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000006348: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000634C: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000006350: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006354: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000006358: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000635C: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000006360: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006364: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000006368: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000636C: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000006370: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006374: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000006378: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000637C: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000006380: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006384: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000006388: 5636279B
	s_wait_dscnt 0x0                                           // 00000000638C: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006390: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000006394: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006398: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 00000000639C: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063A0: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000063A4: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063A8: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000063AC: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063B0: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000063B4: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063B8: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000063BC: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063C0: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000063C4: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063C8: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000063CC: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063D0: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000063D4: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063D8: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 0000000063DC: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063E0: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 0000000063E4: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063E8: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 0000000063EC: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063F0: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 0000000063F4: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000063F8: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 0000000063FC: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006400: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000006404: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006408: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 00000000640C: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006410: BF870004
	v_add_f32_e32 v34, v34, v24                                // 000000006414: 06443122
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006418: BF870004
	v_add_f32_e32 v38, v38, v25                                // 00000000641C: 064C3326
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006420: BF870004
	v_add_f32_e32 v42, v42, v26                                // 000000006424: 0654352A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006428: BF870004
	v_add_f32_e32 v46, v46, v27                                // 00000000642C: 065C372E
	s_bitcmp0_b32 s29, 3                                       // 000000006430: BF0C831D
	s_cbranch_scc1 74                                          // 000000006434: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x4c60>
	ds_load_b128 v[16:19], v4 offset:3120                      // 000000006438: DBFC0C30 10000004
	ds_load_b128 v[20:23], v4 offset:3136                      // 000000006440: DBFC0C40 14000004
	s_wait_dscnt 0x1                                           // 000000006448: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 00000000644C: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000006450: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000006454: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000006458: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000645C: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000006460: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006464: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000006468: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000646C: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000006470: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006474: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000006478: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000647C: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000006480: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006484: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000006488: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000648C: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000006490: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006494: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000006498: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000649C: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 0000000064A0: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064A4: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 0000000064A8: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064AC: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 0000000064B0: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064B4: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 0000000064B8: 5636279B
	s_wait_dscnt 0x0                                           // 0000000064BC: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064C0: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 0000000064C4: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064C8: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 0000000064CC: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064D0: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000064D4: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064D8: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000064DC: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064E0: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000064E4: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064E8: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000064EC: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064F0: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000064F4: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000064F8: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000064FC: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006500: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000006504: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006508: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 00000000650C: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006510: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000006514: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006518: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 00000000651C: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006520: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000006524: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006528: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 00000000652C: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006530: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000006534: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006538: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 00000000653C: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006540: BF870004
	v_add_f32_e32 v35, v35, v24                                // 000000006544: 06463123
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006548: BF870004
	v_add_f32_e32 v39, v39, v25                                // 00000000654C: 064E3327
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006550: BF870004
	v_add_f32_e32 v43, v43, v26                                // 000000006554: 0656352B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006558: BF870004
	v_add_f32_e32 v47, v47, v27                                // 00000000655C: 065E372F
	s_barrier_signal -1                                        // 000000006560: BE804EC1
	s_barrier_wait 0xffff                                      // 000000006564: BF94FFFF
	s_wait_loadcnt 0x9                                         // 000000006568: BFC00009
	ds_store_b128 v5, v[120:123]                               // 00000000656C: DB7C0000 00007805
	s_wait_loadcnt 0x8                                         // 000000006574: BFC00008
	ds_store_b128 v5, v[124:127] offset:2080                   // 000000006578: DB7C0820 00007C05
	s_wait_dscnt 0x0                                           // 000000006580: BFC60000
	s_barrier_signal -1                                        // 000000006584: BE804EC1
	s_wait_loadcnt 0x7                                         // 000000006588: BFC00007
	v_lshrrev_b32_e32 v28, v2, v112                            // 00000000658C: 3238E102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006590: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000006594: CC20001D 8A007F1C
	s_wait_loadcnt 0x6                                         // 00000000659C: BFC00006
	v_and_b32_e32 v30, 0xf0f0f0f, v116                         // 0000000065A0: 363CE8FF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v116                             // 0000000065A8: 323EE884
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000065AC: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 0000000065B0: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000065B8: BF870003
	v_cvt_f32_ubyte0_e32 v128, v30                             // 0000000065BC: 7F00231E
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000065C0: BF870002
	v_cvt_f32_ubyte0_e32 v129, v31                             // 0000000065C4: 7F02231F
	v_cvt_f32_ubyte1_e32 v130, v30                             // 0000000065C8: 7F04251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000065CC: BF870004
	v_cvt_f32_ubyte1_e32 v131, v31                             // 0000000065D0: 7F06251F
	v_cvt_f32_ubyte2_e32 v132, v30                             // 0000000065D4: 7F08271E
	v_cvt_f32_ubyte2_e32 v133, v31                             // 0000000065D8: 7F0A271F
	v_cvt_f32_ubyte3_e32 v134, v30                             // 0000000065DC: 7F0C291E
	v_cvt_f32_ubyte3_e32 v135, v31                             // 0000000065E0: 7F0E291F
	v_fma_mix_f32 v128, v28, v128, v29 op_sel_hi:[1,0,0]       // 0000000065E4: CC200080 0C77011C
	v_fma_mix_f32 v129, v28, v129, v29 op_sel_hi:[1,0,0]       // 0000000065EC: CC200081 0C77031C
	v_fma_mix_f32 v130, v28, v130, v29 op_sel_hi:[1,0,0]       // 0000000065F4: CC200082 0C77051C
	v_fma_mix_f32 v131, v28, v131, v29 op_sel_hi:[1,0,0]       // 0000000065FC: CC200083 0C77071C
	v_fma_mix_f32 v132, v28, v132, v29 op_sel_hi:[1,0,0]       // 000000006604: CC200084 0C77091C
	v_fma_mix_f32 v133, v28, v133, v29 op_sel_hi:[1,0,0]       // 00000000660C: CC200085 0C770B1C
	v_fma_mix_f32 v134, v28, v134, v29 op_sel_hi:[1,0,0]       // 000000006614: CC200086 0C770D1C
	v_fma_mix_f32 v135, v28, v135, v29 op_sel_hi:[1,0,0]       // 00000000661C: CC200087 0C770F1C
	s_wait_loadcnt 0x5                                         // 000000006624: BFC00005
	v_lshrrev_b32_e32 v28, v2, v113                            // 000000006628: 3238E302
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000662C: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000006630: CC20001D 8A007F1C
	s_wait_loadcnt 0x4                                         // 000000006638: BFC00004
	v_and_b32_e32 v30, 0xf0f0f0f, v117                         // 00000000663C: 363CEAFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v117                             // 000000006644: 323EEA84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006648: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 00000000664C: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000006654: BF870003
	v_cvt_f32_ubyte0_e32 v136, v30                             // 000000006658: 7F10231E
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000665C: BF870002
	v_cvt_f32_ubyte0_e32 v137, v31                             // 000000006660: 7F12231F
	v_cvt_f32_ubyte1_e32 v138, v30                             // 000000006664: 7F14251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006668: BF870004
	v_cvt_f32_ubyte1_e32 v139, v31                             // 00000000666C: 7F16251F
	v_cvt_f32_ubyte2_e32 v140, v30                             // 000000006670: 7F18271E
	v_cvt_f32_ubyte2_e32 v141, v31                             // 000000006674: 7F1A271F
	v_cvt_f32_ubyte3_e32 v142, v30                             // 000000006678: 7F1C291E
	v_cvt_f32_ubyte3_e32 v143, v31                             // 00000000667C: 7F1E291F
	v_fma_mix_f32 v136, v28, v136, v29 op_sel_hi:[1,0,0]       // 000000006680: CC200088 0C77111C
	v_fma_mix_f32 v137, v28, v137, v29 op_sel_hi:[1,0,0]       // 000000006688: CC200089 0C77131C
	v_fma_mix_f32 v138, v28, v138, v29 op_sel_hi:[1,0,0]       // 000000006690: CC20008A 0C77151C
	v_fma_mix_f32 v139, v28, v139, v29 op_sel_hi:[1,0,0]       // 000000006698: CC20008B 0C77171C
	v_fma_mix_f32 v140, v28, v140, v29 op_sel_hi:[1,0,0]       // 0000000066A0: CC20008C 0C77191C
	v_fma_mix_f32 v141, v28, v141, v29 op_sel_hi:[1,0,0]       // 0000000066A8: CC20008D 0C771B1C
	v_fma_mix_f32 v142, v28, v142, v29 op_sel_hi:[1,0,0]       // 0000000066B0: CC20008E 0C771D1C
	v_fma_mix_f32 v143, v28, v143, v29 op_sel_hi:[1,0,0]       // 0000000066B8: CC20008F 0C771F1C
	s_wait_loadcnt 0x3                                         // 0000000066C0: BFC00003
	v_lshrrev_b32_e32 v28, v2, v114                            // 0000000066C4: 3238E502
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000066C8: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 0000000066CC: CC20001D 8A007F1C
	s_wait_loadcnt 0x2                                         // 0000000066D4: BFC00002
	v_and_b32_e32 v30, 0xf0f0f0f, v118                         // 0000000066D8: 363CECFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v118                             // 0000000066E0: 323EEC84
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000066E4: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 0000000066E8: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000066F0: BF870003
	v_cvt_f32_ubyte0_e32 v144, v30                             // 0000000066F4: 7F20231E
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000066F8: BF870002
	v_cvt_f32_ubyte0_e32 v145, v31                             // 0000000066FC: 7F22231F
	v_cvt_f32_ubyte1_e32 v146, v30                             // 000000006700: 7F24251E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006704: BF870004
	v_cvt_f32_ubyte1_e32 v147, v31                             // 000000006708: 7F26251F
	v_cvt_f32_ubyte2_e32 v148, v30                             // 00000000670C: 7F28271E
	v_cvt_f32_ubyte2_e32 v149, v31                             // 000000006710: 7F2A271F
	v_cvt_f32_ubyte3_e32 v150, v30                             // 000000006714: 7F2C291E
	v_cvt_f32_ubyte3_e32 v151, v31                             // 000000006718: 7F2E291F
	v_fma_mix_f32 v144, v28, v144, v29 op_sel_hi:[1,0,0]       // 00000000671C: CC200090 0C77211C
	v_fma_mix_f32 v145, v28, v145, v29 op_sel_hi:[1,0,0]       // 000000006724: CC200091 0C77231C
	v_fma_mix_f32 v146, v28, v146, v29 op_sel_hi:[1,0,0]       // 00000000672C: CC200092 0C77251C
	v_fma_mix_f32 v147, v28, v147, v29 op_sel_hi:[1,0,0]       // 000000006734: CC200093 0C77271C
	v_fma_mix_f32 v148, v28, v148, v29 op_sel_hi:[1,0,0]       // 00000000673C: CC200094 0C77291C
	v_fma_mix_f32 v149, v28, v149, v29 op_sel_hi:[1,0,0]       // 000000006744: CC200095 0C772B1C
	v_fma_mix_f32 v150, v28, v150, v29 op_sel_hi:[1,0,0]       // 00000000674C: CC200096 0C772D1C
	v_fma_mix_f32 v151, v28, v151, v29 op_sel_hi:[1,0,0]       // 000000006754: CC200097 0C772F1C
	s_wait_loadcnt 0x1                                         // 00000000675C: BFC00001
	v_lshrrev_b32_e32 v28, v2, v115                            // 000000006760: 3238E702
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006764: BF870001
	v_fma_mix_f32 v29, v28, s63, neg(0) op_sel_hi:[1,0,0]      // 000000006768: CC20001D 8A007F1C
	s_wait_loadcnt 0x0                                         // 000000006770: BFC00000
	v_and_b32_e32 v30, 0xf0f0f0f, v119                         // 000000006774: 363CEEFF 0F0F0F0F
	v_lshrrev_b32_e32 v31, 4, v119                             // 00000000677C: 323EEE84
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006780: BF870001
	v_and_b32_e32 v31, 0xf0f0f0f, v31                          // 000000006784: 363E3EFF 0F0F0F0F
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000678C: BF870003
	v_cvt_f32_ubyte0_e32 v152, v30                             // 000000006790: 7F30231E
	s_delay_alu instid0(VALU_DEP_2)                            // 000000006794: BF870002
	v_cvt_f32_ubyte0_e32 v153, v31                             // 000000006798: 7F32231F
	v_cvt_f32_ubyte1_e32 v154, v30                             // 00000000679C: 7F34251E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000067A0: BF870004
	v_cvt_f32_ubyte1_e32 v155, v31                             // 0000000067A4: 7F36251F
	v_cvt_f32_ubyte2_e32 v156, v30                             // 0000000067A8: 7F38271E
	v_cvt_f32_ubyte2_e32 v157, v31                             // 0000000067AC: 7F3A271F
	v_cvt_f32_ubyte3_e32 v158, v30                             // 0000000067B0: 7F3C291E
	v_cvt_f32_ubyte3_e32 v159, v31                             // 0000000067B4: 7F3E291F
	v_fma_mix_f32 v152, v28, v152, v29 op_sel_hi:[1,0,0]       // 0000000067B8: CC200098 0C77311C
	v_fma_mix_f32 v153, v28, v153, v29 op_sel_hi:[1,0,0]       // 0000000067C0: CC200099 0C77331C
	v_fma_mix_f32 v154, v28, v154, v29 op_sel_hi:[1,0,0]       // 0000000067C8: CC20009A 0C77351C
	v_fma_mix_f32 v155, v28, v155, v29 op_sel_hi:[1,0,0]       // 0000000067D0: CC20009B 0C77371C
	v_fma_mix_f32 v156, v28, v156, v29 op_sel_hi:[1,0,0]       // 0000000067D8: CC20009C 0C77391C
	v_fma_mix_f32 v157, v28, v157, v29 op_sel_hi:[1,0,0]       // 0000000067E0: CC20009D 0C773B1C
	v_fma_mix_f32 v158, v28, v158, v29 op_sel_hi:[1,0,0]       // 0000000067E8: CC20009E 0C773D1C
	v_fma_mix_f32 v159, v28, v159, v29 op_sel_hi:[1,0,0]       // 0000000067F0: CC20009F 0C773F1C
	s_barrier_wait 0xffff                                      // 0000000067F8: BF94FFFF
	s_bitcmp0_b32 s29, 0                                       // 0000000067FC: BF0C801D
	s_cbranch_scc1 74                                          // 000000006800: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x502c>
	ds_load_b128 v[16:19], v4                                  // 000000006804: DBFC0000 10000004
	ds_load_b128 v[20:23], v4 offset:16                        // 00000000680C: DBFC0010 14000004
	s_wait_dscnt 0x1                                           // 000000006814: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000006818: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 00000000681C: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000006820: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000006824: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006828: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 00000000682C: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006830: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000006834: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006838: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 00000000683C: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006840: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000006844: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006848: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 00000000684C: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006850: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000006854: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006858: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 00000000685C: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006860: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000006864: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006868: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 00000000686C: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006870: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000006874: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006878: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 00000000687C: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006880: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000006884: 5636279B
	s_wait_dscnt 0x0                                           // 000000006888: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000688C: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000006890: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006894: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000006898: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000689C: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000068A0: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068A4: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000068A8: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068AC: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000068B0: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068B4: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000068B8: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068BC: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000068C0: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068C4: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000068C8: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068CC: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 0000000068D0: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068D4: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 0000000068D8: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068DC: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 0000000068E0: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068E4: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 0000000068E8: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068EC: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 0000000068F0: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068F4: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 0000000068F8: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000068FC: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000006900: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006904: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000006908: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000690C: BF870004
	v_add_f32_e32 v48, v48, v24                                // 000000006910: 06603130
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006914: BF870004
	v_add_f32_e32 v52, v52, v25                                // 000000006918: 06683334
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000691C: BF870004
	v_add_f32_e32 v56, v56, v26                                // 000000006920: 06703538
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006924: BF870004
	v_add_f32_e32 v60, v60, v27                                // 000000006928: 0678373C
	s_bitcmp0_b32 s29, 1                                       // 00000000692C: BF0C811D
	s_cbranch_scc1 74                                          // 000000006930: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x515c>
	ds_load_b128 v[16:19], v4 offset:1040                      // 000000006934: DBFC0410 10000004
	ds_load_b128 v[20:23], v4 offset:1056                      // 00000000693C: DBFC0420 14000004
	s_wait_dscnt 0x1                                           // 000000006944: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000006948: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 00000000694C: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000006950: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000006954: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006958: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 00000000695C: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006960: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000006964: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006968: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 00000000696C: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006970: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000006974: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006978: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 00000000697C: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006980: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000006984: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006988: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 00000000698C: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006990: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000006994: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006998: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 00000000699C: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069A0: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 0000000069A4: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069A8: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 0000000069AC: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069B0: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 0000000069B4: 5636279B
	s_wait_dscnt 0x0                                           // 0000000069B8: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069BC: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 0000000069C0: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069C4: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 0000000069C8: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069CC: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 0000000069D0: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069D4: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 0000000069D8: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069DC: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 0000000069E0: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069E4: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 0000000069E8: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069EC: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 0000000069F0: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069F4: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 0000000069F8: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000069FC: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000006A00: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A04: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000006A08: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A0C: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000006A10: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A14: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000006A18: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A1C: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000006A20: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A24: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000006A28: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A2C: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000006A30: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A34: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000006A38: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A3C: BF870004
	v_add_f32_e32 v49, v49, v24                                // 000000006A40: 06623131
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A44: BF870004
	v_add_f32_e32 v53, v53, v25                                // 000000006A48: 066A3335
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A4C: BF870004
	v_add_f32_e32 v57, v57, v26                                // 000000006A50: 06723539
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A54: BF870004
	v_add_f32_e32 v61, v61, v27                                // 000000006A58: 067A373D
	s_bitcmp0_b32 s29, 2                                       // 000000006A5C: BF0C821D
	s_cbranch_scc1 74                                          // 000000006A60: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x528c>
	ds_load_b128 v[16:19], v4 offset:2080                      // 000000006A64: DBFC0820 10000004
	ds_load_b128 v[20:23], v4 offset:2096                      // 000000006A6C: DBFC0830 14000004
	s_wait_dscnt 0x1                                           // 000000006A74: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000006A78: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000006A7C: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000006A80: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000006A84: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A88: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000006A8C: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A90: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000006A94: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006A98: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000006A9C: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AA0: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000006AA4: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AA8: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000006AAC: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AB0: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000006AB4: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AB8: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000006ABC: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AC0: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000006AC4: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AC8: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000006ACC: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AD0: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000006AD4: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AD8: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000006ADC: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AE0: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000006AE4: 5636279B
	s_wait_dscnt 0x0                                           // 000000006AE8: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AEC: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000006AF0: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AF4: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000006AF8: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006AFC: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000006B00: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B04: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000006B08: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B0C: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000006B10: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B14: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000006B18: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B1C: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000006B20: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B24: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000006B28: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B2C: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000006B30: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B34: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000006B38: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B3C: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000006B40: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B44: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000006B48: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B4C: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000006B50: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B54: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000006B58: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B5C: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000006B60: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B64: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000006B68: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B6C: BF870004
	v_add_f32_e32 v50, v50, v24                                // 000000006B70: 06643132
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B74: BF870004
	v_add_f32_e32 v54, v54, v25                                // 000000006B78: 066C3336
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B7C: BF870004
	v_add_f32_e32 v58, v58, v26                                // 000000006B80: 0674353A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006B84: BF870004
	v_add_f32_e32 v62, v62, v27                                // 000000006B88: 067C373E
	s_bitcmp0_b32 s29, 3                                       // 000000006B8C: BF0C831D
	s_cbranch_scc1 74                                          // 000000006B90: BFA2004A <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x53bc>
	ds_load_b128 v[16:19], v4 offset:3120                      // 000000006B94: DBFC0C30 10000004
	ds_load_b128 v[20:23], v4 offset:3136                      // 000000006B9C: DBFC0C40 14000004
	s_wait_dscnt 0x1                                           // 000000006BA4: BFC60001
	v_mul_f32_e32 v24, v129, v17                               // 000000006BA8: 10302381
	v_mul_f32_e32 v25, v137, v17                               // 000000006BAC: 10322389
	v_mul_f32_e32 v26, v145, v17                               // 000000006BB0: 10342391
	v_mul_f32_e32 v27, v153, v17                               // 000000006BB4: 10362399
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006BB8: BF870004
	v_fmac_f32_e32 v24, v128, v16                              // 000000006BBC: 56302180
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006BC0: BF870004
	v_fmac_f32_e32 v25, v136, v16                              // 000000006BC4: 56322188
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006BC8: BF870004
	v_fmac_f32_e32 v26, v144, v16                              // 000000006BCC: 56342190
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006BD0: BF870004
	v_fmac_f32_e32 v27, v152, v16                              // 000000006BD4: 56362198
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006BD8: BF870004
	v_fmac_f32_e32 v24, v130, v18                              // 000000006BDC: 56302582
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006BE0: BF870004
	v_fmac_f32_e32 v25, v138, v18                              // 000000006BE4: 5632258A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006BE8: BF870004
	v_fmac_f32_e32 v26, v146, v18                              // 000000006BEC: 56342592
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006BF0: BF870004
	v_fmac_f32_e32 v27, v154, v18                              // 000000006BF4: 5636259A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006BF8: BF870004
	v_fmac_f32_e32 v24, v131, v19                              // 000000006BFC: 56302783
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C00: BF870004
	v_fmac_f32_e32 v25, v139, v19                              // 000000006C04: 5632278B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C08: BF870004
	v_fmac_f32_e32 v26, v147, v19                              // 000000006C0C: 56342793
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C10: BF870004
	v_fmac_f32_e32 v27, v155, v19                              // 000000006C14: 5636279B
	s_wait_dscnt 0x0                                           // 000000006C18: BFC60000
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C1C: BF870004
	v_fmac_f32_e32 v24, v132, v20                              // 000000006C20: 56302984
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C24: BF870004
	v_fmac_f32_e32 v25, v140, v20                              // 000000006C28: 5632298C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C2C: BF870004
	v_fmac_f32_e32 v26, v148, v20                              // 000000006C30: 56342994
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C34: BF870004
	v_fmac_f32_e32 v27, v156, v20                              // 000000006C38: 5636299C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C3C: BF870004
	v_fmac_f32_e32 v24, v133, v21                              // 000000006C40: 56302B85
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C44: BF870004
	v_fmac_f32_e32 v25, v141, v21                              // 000000006C48: 56322B8D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C4C: BF870004
	v_fmac_f32_e32 v26, v149, v21                              // 000000006C50: 56342B95
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C54: BF870004
	v_fmac_f32_e32 v27, v157, v21                              // 000000006C58: 56362B9D
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C5C: BF870004
	v_fmac_f32_e32 v24, v134, v22                              // 000000006C60: 56302D86
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C64: BF870004
	v_fmac_f32_e32 v25, v142, v22                              // 000000006C68: 56322D8E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C6C: BF870004
	v_fmac_f32_e32 v26, v150, v22                              // 000000006C70: 56342D96
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C74: BF870004
	v_fmac_f32_e32 v27, v158, v22                              // 000000006C78: 56362D9E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C7C: BF870004
	v_fmac_f32_e32 v24, v135, v23                              // 000000006C80: 56302F87
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C84: BF870004
	v_fmac_f32_e32 v25, v143, v23                              // 000000006C88: 56322F8F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C8C: BF870004
	v_fmac_f32_e32 v26, v151, v23                              // 000000006C90: 56342F97
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C94: BF870004
	v_fmac_f32_e32 v27, v159, v23                              // 000000006C98: 56362F9F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006C9C: BF870004
	v_add_f32_e32 v51, v51, v24                                // 000000006CA0: 06663133
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006CA4: BF870004
	v_add_f32_e32 v55, v55, v25                                // 000000006CA8: 066E3337
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006CAC: BF870004
	v_add_f32_e32 v59, v59, v26                                // 000000006CB0: 0676353B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000006CB4: BF870004
	v_add_f32_e32 v63, v63, v27                                // 000000006CB8: 067E373F
	s_wait_alu depctr_vm_vsrc(0)                               // 000000006CBC: BF88FF83
	s_bitcmp0_b32 s29, 0                                       // 000000006CC0: BF0C801D
	s_cbranch_scc1 70                                          // 000000006CC4: BFA20046 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x54e0>
	v_add_f32_e32 v32, v32, v48                                // 000000006CC8: 06406120
	v_add_f32_e32 v64, v64, v80                                // 000000006CCC: 0680A140
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006CD0: BF870001
	v_add_f32_e32 v32, v32, v64                                // 000000006CD4: 06408120
	v_add_f32_e32 v36, v36, v52                                // 000000006CD8: 06486924
	v_add_f32_e32 v68, v68, v84                                // 000000006CDC: 0688A944
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006CE0: BF870001
	v_add_f32_e32 v36, v36, v68                                // 000000006CE4: 06488924
	v_add_f32_e32 v40, v40, v56                                // 000000006CE8: 06507128
	v_add_f32_e32 v72, v72, v88                                // 000000006CEC: 0690B148
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006CF0: BF870001
	v_add_f32_e32 v40, v40, v72                                // 000000006CF4: 06509128
	v_add_f32_e32 v44, v44, v60                                // 000000006CF8: 0658792C
	v_add_f32_e32 v76, v76, v92                                // 000000006CFC: 0698B94C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D00: BF870001
	v_add_f32_e32 v44, v44, v76                                // 000000006D04: 0658992C
	v_cndmask_b32_e64 v128, v36, v32, s65                      // 000000006D08: D5010080 01064124
	v_cndmask_b32_e64 v136, v32, v36, s65                      // 000000006D10: D5010088 01064920
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D18: BF870001
	v_permlanex16_b32 v136, v136, s64, 0xfedcba98              // 000000006D1C: D65C0088 03FC8188 FEDCBA98
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D28: BF870001
	v_add_f32_e32 v128, v128, v136                             // 000000006D2C: 07011180
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D30: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006D34: 070100FA FF090880
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D3C: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006D40: 070100FA FF090480
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D48: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006D4C: 070100FA FF090280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D54: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006D58: 070100FA FF090180
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D60: BF870001
	v_readlane_b32 s88, v128, 16                               // 000000006D64: D7600058 02012180
	s_wait_alu depctr_va_sdst(0)                               // 000000006D6C: BF88F19F
	v_mov_b32_e32 v129, s88                                    // 000000006D70: 7F020258
	v_cndmask_b32_e64 v130, v44, v40, s65                      // 000000006D74: D5010082 0106512C
	v_cndmask_b32_e64 v136, v40, v44, s65                      // 000000006D7C: D5010088 01065928
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D84: BF870001
	v_permlanex16_b32 v136, v136, s64, 0xfedcba98              // 000000006D88: D65C0088 03FC8188 FEDCBA98
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D94: BF870001
	v_add_f32_e32 v130, v130, v136                             // 000000006D98: 07051182
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006D9C: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006DA0: 070504FA FF090882
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006DA8: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006DAC: 070504FA FF090482
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006DB4: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006DB8: 070504FA FF090282
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006DC0: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006DC4: 070504FA FF090182
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006DCC: BF870001
	v_readlane_b32 s89, v130, 16                               // 000000006DD0: D7600059 02012182
	s_wait_alu depctr_va_sdst(0)                               // 000000006DD8: BF88F19F
	v_mov_b32_e32 v131, s89                                    // 000000006DDC: 7F060259
	s_bitcmp1_b32 s29, 0                                       // 000000006DE0: BF0D801D
	s_cbranch_scc1 4                                           // 000000006DE4: BFA20004 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x54f8>
	v_mov_b32_e32 v128, 0                                      // 000000006DE8: 7F000280
	v_mov_b32_e32 v129, 0                                      // 000000006DEC: 7F020280
	v_mov_b32_e32 v130, 0                                      // 000000006DF0: 7F040280
	v_mov_b32_e32 v131, 0                                      // 000000006DF4: 7F060280
	s_add_co_i32 s92, s28, 0                                   // 000000006DF8: 815C801C
	s_mul_i32 s92, s92, s20                                    // 000000006DFC: 965C145C
	s_add_co_i32 s92, s92, s68                                 // 000000006E00: 815C445C
	s_lshl_b32 s92, s92, 2                                     // 000000006E04: 845C825C
	s_bitcmp0_b32 s30, 0                                       // 000000006E08: BF0C801E
	s_cselect_b32 s93, 0x80000000, 0                           // 000000006E0C: 985D80FF 80000000
	s_or_b32 s94, s93, s77                                     // 000000006E14: 8C5E4D5D
	v_or_b32_e32 v137, s94, v7                                 // 000000006E18: 39120E5E
	buffer_store_b32 v128, v137, s[56:59], s92 offen           // 000000006E1C: C406805C 40807080 00000089
	s_or_b32 s94, s93, s78                                     // 000000006E28: 8C5E4E5D
	s_wait_alu depctr_sa_sdst(0)                               // 000000006E2C: BF88FF9E
	v_or_b32_e32 v138, s94, v7                                 // 000000006E30: 39140E5E
	buffer_store_b32 v129, v138, s[56:59], s92 offen offset:4  // 000000006E34: C406805C 40807081 0000048A
	s_or_b32 s94, s93, s79                                     // 000000006E40: 8C5E4F5D
	s_wait_alu depctr_sa_sdst(0)                               // 000000006E44: BF88FF9E
	v_or_b32_e32 v139, s94, v7                                 // 000000006E48: 39160E5E
	buffer_store_b32 v130, v139, s[56:59], s92 offen offset:8  // 000000006E4C: C406805C 40807082 0000088B
	s_or_b32 s94, s93, s80                                     // 000000006E58: 8C5E505D
	s_wait_alu depctr_sa_sdst(0)                               // 000000006E5C: BF88FF9E
	v_or_b32_e32 v140, s94, v7                                 // 000000006E60: 39180E5E
	buffer_store_b32 v131, v140, s[56:59], s92 offen offset:12 // 000000006E64: C406805C 40807083 00000C8C
	s_wait_alu depctr_vm_vsrc(0)                               // 000000006E70: BF88FF83
	s_bitcmp0_b32 s29, 1                                       // 000000006E74: BF0C811D
	s_cbranch_scc1 70                                          // 000000006E78: BFA20046 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x5694>
	v_add_f32_e32 v33, v33, v49                                // 000000006E7C: 06426321
	v_add_f32_e32 v65, v65, v81                                // 000000006E80: 0682A341
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006E84: BF870001
	v_add_f32_e32 v33, v33, v65                                // 000000006E88: 06428321
	v_add_f32_e32 v37, v37, v53                                // 000000006E8C: 064A6B25
	v_add_f32_e32 v69, v69, v85                                // 000000006E90: 068AAB45
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006E94: BF870001
	v_add_f32_e32 v37, v37, v69                                // 000000006E98: 064A8B25
	v_add_f32_e32 v41, v41, v57                                // 000000006E9C: 06527329
	v_add_f32_e32 v73, v73, v89                                // 000000006EA0: 0692B349
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006EA4: BF870001
	v_add_f32_e32 v41, v41, v73                                // 000000006EA8: 06529329
	v_add_f32_e32 v45, v45, v61                                // 000000006EAC: 065A7B2D
	v_add_f32_e32 v77, v77, v93                                // 000000006EB0: 069ABB4D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006EB4: BF870001
	v_add_f32_e32 v45, v45, v77                                // 000000006EB8: 065A9B2D
	v_cndmask_b32_e64 v128, v37, v33, s65                      // 000000006EBC: D5010080 01064325
	v_cndmask_b32_e64 v136, v33, v37, s65                      // 000000006EC4: D5010088 01064B21
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006ECC: BF870001
	v_permlanex16_b32 v136, v136, s64, 0xfedcba98              // 000000006ED0: D65C0088 03FC8188 FEDCBA98
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006EDC: BF870001
	v_add_f32_e32 v128, v128, v136                             // 000000006EE0: 07011180
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006EE4: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006EE8: 070100FA FF090880
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006EF0: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006EF4: 070100FA FF090480
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006EFC: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006F00: 070100FA FF090280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006F08: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006F0C: 070100FA FF090180
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006F14: BF870001
	v_readlane_b32 s88, v128, 16                               // 000000006F18: D7600058 02012180
	s_wait_alu depctr_va_sdst(0)                               // 000000006F20: BF88F19F
	v_mov_b32_e32 v129, s88                                    // 000000006F24: 7F020258
	v_cndmask_b32_e64 v130, v45, v41, s65                      // 000000006F28: D5010082 0106532D
	v_cndmask_b32_e64 v136, v41, v45, s65                      // 000000006F30: D5010088 01065B29
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006F38: BF870001
	v_permlanex16_b32 v136, v136, s64, 0xfedcba98              // 000000006F3C: D65C0088 03FC8188 FEDCBA98
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006F48: BF870001
	v_add_f32_e32 v130, v130, v136                             // 000000006F4C: 07051182
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006F50: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006F54: 070504FA FF090882
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006F5C: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006F60: 070504FA FF090482
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006F68: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006F6C: 070504FA FF090282
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006F74: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000006F78: 070504FA FF090182
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006F80: BF870001
	v_readlane_b32 s89, v130, 16                               // 000000006F84: D7600059 02012182
	s_wait_alu depctr_va_sdst(0)                               // 000000006F8C: BF88F19F
	v_mov_b32_e32 v131, s89                                    // 000000006F90: 7F060259
	s_bitcmp1_b32 s29, 1                                       // 000000006F94: BF0D811D
	s_cbranch_scc1 4                                           // 000000006F98: BFA20004 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x56ac>
	v_mov_b32_e32 v128, 0                                      // 000000006F9C: 7F000280
	v_mov_b32_e32 v129, 0                                      // 000000006FA0: 7F020280
	v_mov_b32_e32 v130, 0                                      // 000000006FA4: 7F040280
	v_mov_b32_e32 v131, 0                                      // 000000006FA8: 7F060280
	s_add_co_i32 s92, s28, 1                                   // 000000006FAC: 815C811C
	s_mul_i32 s92, s92, s20                                    // 000000006FB0: 965C145C
	s_add_co_i32 s92, s92, s68                                 // 000000006FB4: 815C445C
	s_lshl_b32 s92, s92, 2                                     // 000000006FB8: 845C825C
	s_bitcmp0_b32 s30, 1                                       // 000000006FBC: BF0C811E
	s_cselect_b32 s93, 0x80000000, 0                           // 000000006FC0: 985D80FF 80000000
	s_or_b32 s94, s93, s77                                     // 000000006FC8: 8C5E4D5D
	s_wait_alu depctr_sa_sdst(0)                               // 000000006FCC: BF88FF9E
	v_or_b32_e32 v137, s94, v7                                 // 000000006FD0: 39120E5E
	buffer_store_b32 v128, v137, s[56:59], s92 offen           // 000000006FD4: C406805C 40807080 00000089
	s_or_b32 s94, s93, s78                                     // 000000006FE0: 8C5E4E5D
	s_wait_alu depctr_sa_sdst(0)                               // 000000006FE4: BF88FF9E
	v_or_b32_e32 v138, s94, v7                                 // 000000006FE8: 39140E5E
	buffer_store_b32 v129, v138, s[56:59], s92 offen offset:4  // 000000006FEC: C406805C 40807081 0000048A
	s_or_b32 s94, s93, s79                                     // 000000006FF8: 8C5E4F5D
	s_wait_alu depctr_sa_sdst(0)                               // 000000006FFC: BF88FF9E
	v_or_b32_e32 v139, s94, v7                                 // 000000007000: 39160E5E
	buffer_store_b32 v130, v139, s[56:59], s92 offen offset:8  // 000000007004: C406805C 40807082 0000088B
	s_or_b32 s94, s93, s80                                     // 000000007010: 8C5E505D
	s_wait_alu depctr_sa_sdst(0)                               // 000000007014: BF88FF9E
	v_or_b32_e32 v140, s94, v7                                 // 000000007018: 39180E5E
	buffer_store_b32 v131, v140, s[56:59], s92 offen offset:12 // 00000000701C: C406805C 40807083 00000C8C
	s_wait_alu depctr_vm_vsrc(0)                               // 000000007028: BF88FF83
	s_bitcmp0_b32 s29, 2                                       // 00000000702C: BF0C821D
	s_cbranch_scc1 70                                          // 000000007030: BFA20046 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x584c>
	v_add_f32_e32 v34, v34, v50                                // 000000007034: 06446522
	v_add_f32_e32 v66, v66, v82                                // 000000007038: 0684A542
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000703C: BF870001
	v_add_f32_e32 v34, v34, v66                                // 000000007040: 06448522
	v_add_f32_e32 v38, v38, v54                                // 000000007044: 064C6D26
	v_add_f32_e32 v70, v70, v86                                // 000000007048: 068CAD46
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000704C: BF870001
	v_add_f32_e32 v38, v38, v70                                // 000000007050: 064C8D26
	v_add_f32_e32 v42, v42, v58                                // 000000007054: 0654752A
	v_add_f32_e32 v74, v74, v90                                // 000000007058: 0694B54A
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000705C: BF870001
	v_add_f32_e32 v42, v42, v74                                // 000000007060: 0654952A
	v_add_f32_e32 v46, v46, v62                                // 000000007064: 065C7D2E
	v_add_f32_e32 v78, v78, v94                                // 000000007068: 069CBD4E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000706C: BF870001
	v_add_f32_e32 v46, v46, v78                                // 000000007070: 065C9D2E
	v_cndmask_b32_e64 v128, v38, v34, s65                      // 000000007074: D5010080 01064526
	v_cndmask_b32_e64 v136, v34, v38, s65                      // 00000000707C: D5010088 01064D22
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007084: BF870001
	v_permlanex16_b32 v136, v136, s64, 0xfedcba98              // 000000007088: D65C0088 03FC8188 FEDCBA98
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007094: BF870001
	v_add_f32_e32 v128, v128, v136                             // 000000007098: 07011180
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000709C: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000070A0: 070100FA FF090880
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000070A8: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000070AC: 070100FA FF090480
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000070B4: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000070B8: 070100FA FF090280
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000070C0: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000070C4: 070100FA FF090180
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000070CC: BF870001
	v_readlane_b32 s88, v128, 16                               // 0000000070D0: D7600058 02012180
	s_wait_alu depctr_va_sdst(0)                               // 0000000070D8: BF88F19F
	v_mov_b32_e32 v129, s88                                    // 0000000070DC: 7F020258
	v_cndmask_b32_e64 v130, v46, v42, s65                      // 0000000070E0: D5010082 0106552E
	v_cndmask_b32_e64 v136, v42, v46, s65                      // 0000000070E8: D5010088 01065D2A
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000070F0: BF870001
	v_permlanex16_b32 v136, v136, s64, 0xfedcba98              // 0000000070F4: D65C0088 03FC8188 FEDCBA98
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007100: BF870001
	v_add_f32_e32 v130, v130, v136                             // 000000007104: 07051182
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007108: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000710C: 070504FA FF090882
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007114: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000007118: 070504FA FF090482
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007120: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000007124: 070504FA FF090282
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000712C: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000007130: 070504FA FF090182
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007138: BF870001
	v_readlane_b32 s89, v130, 16                               // 00000000713C: D7600059 02012182
	s_wait_alu depctr_va_sdst(0)                               // 000000007144: BF88F19F
	v_mov_b32_e32 v131, s89                                    // 000000007148: 7F060259
	s_bitcmp1_b32 s29, 2                                       // 00000000714C: BF0D821D
	s_cbranch_scc1 4                                           // 000000007150: BFA20004 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x5864>
	v_mov_b32_e32 v128, 0                                      // 000000007154: 7F000280
	v_mov_b32_e32 v129, 0                                      // 000000007158: 7F020280
	v_mov_b32_e32 v130, 0                                      // 00000000715C: 7F040280
	v_mov_b32_e32 v131, 0                                      // 000000007160: 7F060280
	s_add_co_i32 s92, s28, 2                                   // 000000007164: 815C821C
	s_mul_i32 s92, s92, s20                                    // 000000007168: 965C145C
	s_add_co_i32 s92, s92, s68                                 // 00000000716C: 815C445C
	s_lshl_b32 s92, s92, 2                                     // 000000007170: 845C825C
	s_bitcmp0_b32 s30, 2                                       // 000000007174: BF0C821E
	s_cselect_b32 s93, 0x80000000, 0                           // 000000007178: 985D80FF 80000000
	s_or_b32 s94, s93, s77                                     // 000000007180: 8C5E4D5D
	s_wait_alu depctr_sa_sdst(0)                               // 000000007184: BF88FF9E
	v_or_b32_e32 v137, s94, v7                                 // 000000007188: 39120E5E
	buffer_store_b32 v128, v137, s[56:59], s92 offen           // 00000000718C: C406805C 40807080 00000089
	s_or_b32 s94, s93, s78                                     // 000000007198: 8C5E4E5D
	s_wait_alu depctr_sa_sdst(0)                               // 00000000719C: BF88FF9E
	v_or_b32_e32 v138, s94, v7                                 // 0000000071A0: 39140E5E
	buffer_store_b32 v129, v138, s[56:59], s92 offen offset:4  // 0000000071A4: C406805C 40807081 0000048A
	s_or_b32 s94, s93, s79                                     // 0000000071B0: 8C5E4F5D
	s_wait_alu depctr_sa_sdst(0)                               // 0000000071B4: BF88FF9E
	v_or_b32_e32 v139, s94, v7                                 // 0000000071B8: 39160E5E
	buffer_store_b32 v130, v139, s[56:59], s92 offen offset:8  // 0000000071BC: C406805C 40807082 0000088B
	s_or_b32 s94, s93, s80                                     // 0000000071C8: 8C5E505D
	s_wait_alu depctr_sa_sdst(0)                               // 0000000071CC: BF88FF9E
	v_or_b32_e32 v140, s94, v7                                 // 0000000071D0: 39180E5E
	buffer_store_b32 v131, v140, s[56:59], s92 offen offset:12 // 0000000071D4: C406805C 40807083 00000C8C
	s_wait_alu depctr_vm_vsrc(0)                               // 0000000071E0: BF88FF83
	s_bitcmp0_b32 s29, 3                                       // 0000000071E4: BF0C831D
	s_cbranch_scc1 70                                          // 0000000071E8: BFA20046 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x5a04>
	v_add_f32_e32 v35, v35, v51                                // 0000000071EC: 06466723
	v_add_f32_e32 v67, v67, v83                                // 0000000071F0: 0686A743
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000071F4: BF870001
	v_add_f32_e32 v35, v35, v67                                // 0000000071F8: 06468723
	v_add_f32_e32 v39, v39, v55                                // 0000000071FC: 064E6F27
	v_add_f32_e32 v71, v71, v87                                // 000000007200: 068EAF47
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007204: BF870001
	v_add_f32_e32 v39, v39, v71                                // 000000007208: 064E8F27
	v_add_f32_e32 v43, v43, v59                                // 00000000720C: 0656772B
	v_add_f32_e32 v75, v75, v91                                // 000000007210: 0696B74B
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007214: BF870001
	v_add_f32_e32 v43, v43, v75                                // 000000007218: 0656972B
	v_add_f32_e32 v47, v47, v63                                // 00000000721C: 065E7F2F
	v_add_f32_e32 v79, v79, v95                                // 000000007220: 069EBF4F
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007224: BF870001
	v_add_f32_e32 v47, v47, v79                                // 000000007228: 065E9F2F
	v_cndmask_b32_e64 v128, v39, v35, s65                      // 00000000722C: D5010080 01064727
	v_cndmask_b32_e64 v136, v35, v39, s65                      // 000000007234: D5010088 01064F23
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000723C: BF870001
	v_permlanex16_b32 v136, v136, s64, 0xfedcba98              // 000000007240: D65C0088 03FC8188 FEDCBA98
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000724C: BF870001
	v_add_f32_e32 v128, v128, v136                             // 000000007250: 07011180
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007254: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000007258: 070100FA FF090880
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007260: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000007264: 070100FA FF090480
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000726C: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000007270: 070100FA FF090280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007278: BF870001
	v_add_f32_dpp v128, v128, v128 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000727C: 070100FA FF090180
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007284: BF870001
	v_readlane_b32 s88, v128, 16                               // 000000007288: D7600058 02012180
	s_wait_alu depctr_va_sdst(0)                               // 000000007290: BF88F19F
	v_mov_b32_e32 v129, s88                                    // 000000007294: 7F020258
	v_cndmask_b32_e64 v130, v47, v43, s65                      // 000000007298: D5010082 0106572F
	v_cndmask_b32_e64 v136, v43, v47, s65                      // 0000000072A0: D5010088 01065F2B
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000072A8: BF870001
	v_permlanex16_b32 v136, v136, s64, 0xfedcba98              // 0000000072AC: D65C0088 03FC8188 FEDCBA98
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000072B8: BF870001
	v_add_f32_e32 v130, v130, v136                             // 0000000072BC: 07051182
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000072C0: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000072C4: 070504FA FF090882
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000072CC: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000072D0: 070504FA FF090482
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000072D8: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000072DC: 070504FA FF090282
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000072E4: BF870001
	v_add_f32_dpp v130, v130, v130 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000072E8: 070504FA FF090182
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000072F0: BF870001
	v_readlane_b32 s89, v130, 16                               // 0000000072F4: D7600059 02012182
	s_wait_alu depctr_va_sdst(0)                               // 0000000072FC: BF88F19F
	v_mov_b32_e32 v131, s89                                    // 000000007300: 7F060259
	s_bitcmp1_b32 s29, 3                                       // 000000007304: BF0D831D
	s_cbranch_scc1 4                                           // 000000007308: BFA20004 <gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_pm_r4_pad16_b128_altch+0x5a1c>
	v_mov_b32_e32 v128, 0                                      // 00000000730C: 7F000280
	v_mov_b32_e32 v129, 0                                      // 000000007310: 7F020280
	v_mov_b32_e32 v130, 0                                      // 000000007314: 7F040280
	v_mov_b32_e32 v131, 0                                      // 000000007318: 7F060280
	s_add_co_i32 s92, s28, 3                                   // 00000000731C: 815C831C
	s_mul_i32 s92, s92, s20                                    // 000000007320: 965C145C
	s_add_co_i32 s92, s92, s68                                 // 000000007324: 815C445C
	s_lshl_b32 s92, s92, 2                                     // 000000007328: 845C825C
	s_bitcmp0_b32 s30, 3                                       // 00000000732C: BF0C831E
	s_cselect_b32 s93, 0x80000000, 0                           // 000000007330: 985D80FF 80000000
	s_or_b32 s94, s93, s77                                     // 000000007338: 8C5E4D5D
	s_wait_alu depctr_sa_sdst(0)                               // 00000000733C: BF88FF9E
	v_or_b32_e32 v137, s94, v7                                 // 000000007340: 39120E5E
	buffer_store_b32 v128, v137, s[56:59], s92 offen           // 000000007344: C406805C 40807080 00000089
	s_or_b32 s94, s93, s78                                     // 000000007350: 8C5E4E5D
	s_wait_alu depctr_sa_sdst(0)                               // 000000007354: BF88FF9E
	v_or_b32_e32 v138, s94, v7                                 // 000000007358: 39140E5E
	buffer_store_b32 v129, v138, s[56:59], s92 offen offset:4  // 00000000735C: C406805C 40807081 0000048A
	s_or_b32 s94, s93, s79                                     // 000000007368: 8C5E4F5D
	s_wait_alu depctr_sa_sdst(0)                               // 00000000736C: BF88FF9E
	v_or_b32_e32 v139, s94, v7                                 // 000000007370: 39160E5E
	buffer_store_b32 v130, v139, s[56:59], s92 offen offset:8  // 000000007374: C406805C 40807082 0000088B
	s_or_b32 s94, s93, s80                                     // 000000007380: 8C5E505D
	s_wait_alu depctr_sa_sdst(0)                               // 000000007384: BF88FF9E
	v_or_b32_e32 v140, s94, v7                                 // 000000007388: 39180E5E
	buffer_store_b32 v131, v140, s[56:59], s92 offen offset:12 // 00000000738C: C406805C 40807083 00000C8C
	s_wait_storecnt 0x0                                        // 000000007398: BFC10000
	s_endpgm                                                   // 00000000739C: BFB00000
	s_code_end                                                 // 0000000073A0: BF9F0000
	s_code_end                                                 // 0000000073A4: BF9F0000
	s_code_end                                                 // 0000000073A8: BF9F0000
	s_code_end                                                 // 0000000073AC: BF9F0000
	s_code_end                                                 // 0000000073B0: BF9F0000
	s_code_end                                                 // 0000000073B4: BF9F0000
	s_code_end                                                 // 0000000073B8: BF9F0000
	s_code_end                                                 // 0000000073BC: BF9F0000
	s_code_end                                                 // 0000000073C0: BF9F0000
	s_code_end                                                 // 0000000073C4: BF9F0000
	s_code_end                                                 // 0000000073C8: BF9F0000
	s_code_end                                                 // 0000000073CC: BF9F0000
	s_code_end                                                 // 0000000073D0: BF9F0000
	s_code_end                                                 // 0000000073D4: BF9F0000
	s_code_end                                                 // 0000000073D8: BF9F0000
	s_code_end                                                 // 0000000073DC: BF9F0000
	s_code_end                                                 // 0000000073E0: BF9F0000
	s_code_end                                                 // 0000000073E4: BF9F0000
	s_code_end                                                 // 0000000073E8: BF9F0000
	s_code_end                                                 // 0000000073EC: BF9F0000
	s_code_end                                                 // 0000000073F0: BF9F0000
	s_code_end                                                 // 0000000073F4: BF9F0000
	s_code_end                                                 // 0000000073F8: BF9F0000
	s_code_end                                                 // 0000000073FC: BF9F0000
	s_code_end                                                 // 000000007400: BF9F0000
	s_code_end                                                 // 000000007404: BF9F0000
	s_code_end                                                 // 000000007408: BF9F0000
	s_code_end                                                 // 00000000740C: BF9F0000
	s_code_end                                                 // 000000007410: BF9F0000
	s_code_end                                                 // 000000007414: BF9F0000
	s_code_end                                                 // 000000007418: BF9F0000
	s_code_end                                                 // 00000000741C: BF9F0000
	s_code_end                                                 // 000000007420: BF9F0000
	s_code_end                                                 // 000000007424: BF9F0000
	s_code_end                                                 // 000000007428: BF9F0000
	s_code_end                                                 // 00000000742C: BF9F0000
	s_code_end                                                 // 000000007430: BF9F0000
	s_code_end                                                 // 000000007434: BF9F0000
	s_code_end                                                 // 000000007438: BF9F0000
	s_code_end                                                 // 00000000743C: BF9F0000
	s_code_end                                                 // 000000007440: BF9F0000
	s_code_end                                                 // 000000007444: BF9F0000
	s_code_end                                                 // 000000007448: BF9F0000
	s_code_end                                                 // 00000000744C: BF9F0000
	s_code_end                                                 // 000000007450: BF9F0000
	s_code_end                                                 // 000000007454: BF9F0000
	s_code_end                                                 // 000000007458: BF9F0000
	s_code_end                                                 // 00000000745C: BF9F0000
	s_code_end                                                 // 000000007460: BF9F0000
	s_code_end                                                 // 000000007464: BF9F0000
	s_code_end                                                 // 000000007468: BF9F0000
	s_code_end                                                 // 00000000746C: BF9F0000
	s_code_end                                                 // 000000007470: BF9F0000
	s_code_end                                                 // 000000007474: BF9F0000
	s_code_end                                                 // 000000007478: BF9F0000
	s_code_end                                                 // 00000000747C: BF9F0000
	s_code_end                                                 // 000000007480: BF9F0000
	s_code_end                                                 // 000000007484: BF9F0000
	s_code_end                                                 // 000000007488: BF9F0000
	s_code_end                                                 // 00000000748C: BF9F0000
	s_code_end                                                 // 000000007490: BF9F0000
	s_code_end                                                 // 000000007494: BF9F0000
	s_code_end                                                 // 000000007498: BF9F0000
	s_code_end                                                 // 00000000749C: BF9F0000
	s_code_end                                                 // 0000000074A0: BF9F0000
	s_code_end                                                 // 0000000074A4: BF9F0000
	s_code_end                                                 // 0000000074A8: BF9F0000
	s_code_end                                                 // 0000000074AC: BF9F0000
	s_code_end                                                 // 0000000074B0: BF9F0000
	s_code_end                                                 // 0000000074B4: BF9F0000
	s_code_end                                                 // 0000000074B8: BF9F0000
	s_code_end                                                 // 0000000074BC: BF9F0000
	s_code_end                                                 // 0000000074C0: BF9F0000
	s_code_end                                                 // 0000000074C4: BF9F0000
	s_code_end                                                 // 0000000074C8: BF9F0000
	s_code_end                                                 // 0000000074CC: BF9F0000
	s_code_end                                                 // 0000000074D0: BF9F0000
	s_code_end                                                 // 0000000074D4: BF9F0000
	s_code_end                                                 // 0000000074D8: BF9F0000
	s_code_end                                                 // 0000000074DC: BF9F0000
	s_code_end                                                 // 0000000074E0: BF9F0000
	s_code_end                                                 // 0000000074E4: BF9F0000
	s_code_end                                                 // 0000000074E8: BF9F0000
	s_code_end                                                 // 0000000074EC: BF9F0000
	s_code_end                                                 // 0000000074F0: BF9F0000
	s_code_end                                                 // 0000000074F4: BF9F0000
	s_code_end                                                 // 0000000074F8: BF9F0000
	s_code_end                                                 // 0000000074FC: BF9F0000
	s_code_end                                                 // 000000007500: BF9F0000
	s_code_end                                                 // 000000007504: BF9F0000
	s_code_end                                                 // 000000007508: BF9F0000
	s_code_end                                                 // 00000000750C: BF9F0000
	s_code_end                                                 // 000000007510: BF9F0000
	s_code_end                                                 // 000000007514: BF9F0000
	s_code_end                                                 // 000000007518: BF9F0000
	s_code_end                                                 // 00000000751C: BF9F0000
	s_code_end                                                 // 000000007520: BF9F0000
	s_code_end                                                 // 000000007524: BF9F0000
	s_code_end                                                 // 000000007528: BF9F0000
	s_code_end                                                 // 00000000752C: BF9F0000
	s_code_end                                                 // 000000007530: BF9F0000
	s_code_end                                                 // 000000007534: BF9F0000
	s_code_end                                                 // 000000007538: BF9F0000
	s_code_end                                                 // 00000000753C: BF9F0000
	s_code_end                                                 // 000000007540: BF9F0000
	s_code_end                                                 // 000000007544: BF9F0000
	s_code_end                                                 // 000000007548: BF9F0000
	s_code_end                                                 // 00000000754C: BF9F0000
	s_code_end                                                 // 000000007550: BF9F0000
	s_code_end                                                 // 000000007554: BF9F0000
	s_code_end                                                 // 000000007558: BF9F0000
	s_code_end                                                 // 00000000755C: BF9F0000
	s_code_end                                                 // 000000007560: BF9F0000
	s_code_end                                                 // 000000007564: BF9F0000
	s_code_end                                                 // 000000007568: BF9F0000
	s_code_end                                                 // 00000000756C: BF9F0000
	s_code_end                                                 // 000000007570: BF9F0000
	s_code_end                                                 // 000000007574: BF9F0000
	s_code_end                                                 // 000000007578: BF9F0000
	s_code_end                                                 // 00000000757C: BF9F0000
