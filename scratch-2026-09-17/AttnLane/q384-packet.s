	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_gfx1201:     ; @attention_fp8_e4m3_fa2_gqa_gfx1201
.Lfunc_begin0:
	.file	0 "/home/kaden/ClaudeCode/warpfront/wt-attnlane" "kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip" md5 0x6542cd8a2237d64c20ace509433bd3eb
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.loc	0 1334 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1334:17
	s_load_b128 s[4:7], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	.loc	0 1334 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1334:23
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_89
; %bb.1:
	.loc	0 1337 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1337:14
	s_cmp_gt_i32 ttmp7, 3
	s_cbranch_scc1 .LBB0_89
; %bb.2:
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0
	s_load_b32 s22, s[0:1], 0x38
	.loc	0 1339 35 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1339:35
	s_lshl_b32 s3, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 1340 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1340:16
	s_cmp_ge_i32 s3, s7
	s_cbranch_scc1 .LBB0_89
; %bb.3:
.Ltmp0:
	.loc	0 957 31                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:957:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v3, 0
	v_lshrrev_b32_e32 v156, 4, v0
	v_lshrrev_b32_e32 v157, 3, v0
	v_mov_b32_e32 v147, 0
	s_mov_b32 s23, 0
	.loc	0 974 9                         ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:974:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_5
; %bb.4:
	.loc	0 975 33                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshrrev_b32_e32 v1, 4, v0
	.loc	0 976 24                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:976:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_or_b32 v145, v0, 7, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 975 33                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_b32_e32 v1, 6, v1
	.loc	0 978 31                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:978:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_lo_u32 v3, v145, 24
	.loc	0 977 24                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:977:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmp_gt_i32_e32 vcc_lo, s7, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	.loc	0 975 25                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[1:2], null, ttmp7, 6, v[1:2]
	.loc	0 975 45 is_stmt 0              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshrrev_b32_e32 v2, 3, v0
	s_and_b32 s23, vcc_lo, exec_lo
	.loc	0 975 39                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_or_b32 v147, v2, 1, v1
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 978 55 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:978:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_lshl_u32 v2, v147, v3, 8
	.loc	0 1150 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1150:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b32_e32 v3, 0
.Ltmp1:
.LBB0_5:
	.loc	0 0 38 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:38
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1334 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1334:17
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x20
	v_dual_mov_b32 v5, -1 :: v_dual_and_b32 v6, 31, v0
	v_bfrev_b32_e32 v7, -2
.Ltmp2:
	.loc	0 991 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:991:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_u32_e32 8, v6
	s_cbranch_execz .LBB0_9
; %bb.6:
	.loc	0 992 31                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:992:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v4, s3, v6
	v_bfrev_b32_e32 v7, -2
	v_mov_b32_e32 v5, -1
	.loc	0 993 16                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:993:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s7, v4
	s_cbranch_execz .LBB0_8
; %bb.7:
	.loc	0 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_ashrrev_i32_e32 v5, 31, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_add_co_u32 v4, vcc_lo, s0, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s1, v5, vcc_lo
	.loc	0 994 27 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:994:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_b32 v5, v[4:5], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v7, v5
.LBB0_8:
	.loc	0 0 27 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:27
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_9:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp3:
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_cndmask_b32_e64 v14, 0, v3, s23
	v_dual_mov_b32 v166, 1.0 :: v_dual_and_b32 v159, 15, v0
.Ltmp4:
	.loc	0 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshrrev_b32_e32 v161, 5, v0
.Ltmp5:
	.file	1 "/opt/rocm/core/include/hip/amd_detail" "amd_warp_functions.h" md5 0x78b3d571ca85d7d16c60115ed6cb4554
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_xor_b32_e32 v1, 16, v11
.Ltmp6:
	.loc	1 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_xor_b32_e32 v3, 4, v11
	v_cndmask_b32_e64 v15, 0, v2, s23
	v_ashrrev_i32_e32 v146, 31, v145
	v_lshrrev_b32_e32 v2, 1, v6
.Ltmp7:
	.loc	1 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
.Ltmp8:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_xor_b32_e32 v17, 1, v11
	v_lshl_add_u32 v160, v6, 3, 0
	v_lshlrev_b64_e32 v[129:130], 2, v[145:146]
	s_mov_b32 s17, 0
.Ltmp9:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v11, v1, vcc_lo
	s_mul_i32 s16, s7, 0x1800
	s_mov_b32 s4, ttmp7
	s_ashr_i32 s5, ttmp7, 31
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[18:19], s[8:9], s[16:17]
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_lshlrev_b32_e32 v158, 2, v1
.Ltmp10:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_xor_b32_e32 v1, 8, v11
	s_lshl_b64 s[4:5], s[4:5], 8
	s_lshl_b32 s6, ttmp7, 1
	s_add_nc_u64 s[20:21], s[10:11], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s7, s6, 31
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
	s_mov_b32 s16, s17
	s_wait_alu depctr_va_vcc(0)
	v_dual_mov_b32 v168, 0xff800000 :: v_dual_cndmask_b32 v9, v11, v1
.Ltmp11:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v4, v158, v5
.Ltmp12:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1004:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v8, v158, v7
.Ltmp13:
	.loc	1 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	v_mov_b32_e32 v1, 0
.Ltmp14:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_lshlrev_b32_e32 v9, 2, v9
	v_and_b32_e32 v146, 8, v2
	v_lshlrev_b32_e32 v162, 7, v159
	v_lshlrev_b32_e32 v18, 4, v159
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v2, v1
.Ltmp15:
	.file	2 "/opt/rocm/core-10.0/lib/llvm/lib/clang/23/include" "__clang_hip_math.h" md5 0x036a896c80a276b79275b247e57f4652
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v5, v4
.Ltmp16:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1004:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v7, v8
	v_lshrrev_b32_e32 v7, 1, v0
	v_lshlrev_b32_e32 v8, 3, v0
.Ltmp17:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v11, v3, vcc_lo
.Ltmp18:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v12, v9, v4
.Ltmp19:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1005:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v13, v9, v5
	v_mov_b32_e32 v167, 0
	v_and_b32_e32 v19, 0xf8, v8
.Ltmp20:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_lshlrev_b32_e32 v0, 2, v0
	v_and_b32_e32 v163, 8, v7
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v7, v1
.Ltmp21:
	.loc	0 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshl_or_b32 v133, v161, 8, v19
	v_mad_co_u64_u32 v[9:10], null, v145, 24, v[147:148]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v10, v1 :: v_dual_add_nc_u32 v165, 0, v133
	v_lshlrev_b64_e32 v[131:132], 2, v[9:10]
.Ltmp22:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v12, v4, v12
.Ltmp23:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1005:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v13, v5, v13
.Ltmp24:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_xor_b32_e32 v4, 2, v11
	v_mov_b32_e32 v5, v1
	v_add_co_u32 v148, s3, s20, v18
.Ltmp25:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v16, v0, v12
.Ltmp26:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1006:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v0, v0, v13
.Ltmp27:
	.loc	1 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v149, null, s21, 0, s3
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[20:21], s[10:11], s[6:7]
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v11, v4, vcc_lo
.Ltmp28:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v17
	v_mov_b32_e32 v4, v1
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v66, v11, v17, vcc_lo
.Ltmp29:
	.loc	0 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_co_u32 v134, vcc_lo, v15, v146
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v135, null, 0, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp30:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_lshlrev_b32_e32 v89, 2, v66
.Ltmp31:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_lshlrev_b32_e32 v20, 2, v8
	v_mov_b32_e32 v8, v1
.Ltmp32:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v65, v12, v16
.Ltmp33:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1006:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v13, v0
	v_add_co_u32 v150, vcc_lo, s0, v129
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v11, v3
	v_mov_b32_e32 v13, v5
.Ltmp34:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v67, v20, v65
.Ltmp35:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1007:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v68, v20, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v151, null, s1, v130, vcc_lo
.Ltmp36:
	.loc	0 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_co_u32 v129, vcc_lo, s8, v134
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, s9, v135, vcc_lo
	v_add_co_u32 v152, vcc_lo, s18, v131
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, s19, v132, vcc_lo
	v_add_co_u32 v154, vcc_lo, v129, 48
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v33, v1
.Ltmp37:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v136, v65, v67
.Ltmp38:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1007:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v68
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v41, v1
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v49, v1
.Ltmp39:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v137, v89, v136
.Ltmp40:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1008:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v138, v89, v0
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v57, v1
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v65, v1
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v73, v1
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v81, v1
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v89, v1
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v97, v1
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v105, v1
	v_dual_mov_b32 v120, v8 :: v_dual_mov_b32 v113, v1
	v_dual_mov_b32 v128, v8 :: v_dual_mov_b32 v121, v1
.Ltmp41:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v131, v136, v137
.Ltmp42:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1008:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v138
.Ltmp43:
	.loc	0 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v155, null, 0, v130, vcc_lo
	v_dual_mov_b32 v15, v7 :: v_dual_lshlrev_b32 v164, 4, v161
	.loc	0 1009 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1009:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_readfirstlane_b32 s24, v131
	v_dual_mov_b32 v136, v8 :: v_dual_mov_b32 v129, v1
	v_dual_mov_b32 v14, v6 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v12, v4 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v10, v2 :: v_dual_mov_b32 v19, v3
	v_dual_mov_b32 v9, v1 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v39, v7 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v35, v3 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v71, v7 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v79, v7 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v77, v5 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v87, v7 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v85, v5 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v95, v7 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v93, v5 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v103, v7 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v101, v5 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v99, v3 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v111, v7 :: v_dual_mov_b32 v100, v4
	v_dual_mov_b32 v109, v5 :: v_dual_mov_b32 v98, v2
	v_dual_mov_b32 v107, v3 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v119, v7 :: v_dual_mov_b32 v108, v4
	v_dual_mov_b32 v117, v5 :: v_dual_mov_b32 v106, v2
	v_dual_mov_b32 v115, v3 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v127, v7 :: v_dual_mov_b32 v116, v4
	v_dual_mov_b32 v125, v5 :: v_dual_mov_b32 v114, v2
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v135, v7 :: v_dual_mov_b32 v124, v4
	v_dual_mov_b32 v133, v5 :: v_dual_mov_b32 v122, v2
	v_mov_b32_e32 v131, v3
	.loc	0 1010 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1010:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_readfirstlane_b32 s25, v0
	v_mov_b32_e32 v134, v6
	v_mov_b32_e32 v132, v4
	v_mov_b32_e32 v130, v2
	s_add_nc_u64 s[18:19], s[12:13], s[4:5]
	s_add_nc_u64 s[12:13], s[12:13], s[6:7]
	s_branch .LBB0_13
.LBB0_10:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	0 0 22 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:22
	v_mov_b32_e32 v168, v2
.LBB0_11:                               ;   in Loop: Header=BB0_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s27
.Ltmp44:
	.file	3 "/opt/rocm/core/include/hip/amd_detail" "amd_device_functions.h" md5 0xa1a79f59a27f4196ae31454cacd6cd14
	.loc	3 701 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp45:
.LBB0_12:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	3 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
	s_add_co_i32 s16, s16, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s16, 0x3fffffff
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s17, s17, 64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s11, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_87
.LBB0_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_16 Depth 2
                                        ;     Child Loop BB0_22 Depth 2
                                        ;     Child Loop BB0_44 Depth 2
                                        ;       Child Loop BB0_49 Depth 3
	.loc	0 1015 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1015:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_lshl_b32 s26, s16, 6
	.loc	0 1016 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1016:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s26, s24
	s_cselect_b32 s11, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s11
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_12
; %bb.14:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_mov_b32_e32 v0, v156
	v_mov_b32_e32 v6, v157
	s_mov_b32 s0, 8
	s_branch .LBB0_16
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 1032 49 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_b32_e32 v7, 0xe0, v6
	.loc	0 1024 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v6, 16, v6
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	0 1032 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v7, v7, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1032 54 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:54 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_or_b32 v7, v0, 15, v7
	.loc	0 1024 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v0, 8, v0
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshl_add_u32 v7, v7, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v7, v[2:3], v[4:5] offset1:16
	.loc	0 1024 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cbranch_scc1 .LBB0_18
.LBB0_16:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	0 0 9                           ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v7, s17, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v5, 0
	.loc	0 1033 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1033:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s1, exec_lo
	v_cmpx_ge_i32_e64 s24, v7
	s_cbranch_execz .LBB0_15
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=2
	.loc	0 1037 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1037:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v7, v[148:149]
	.loc	0 1038 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1038:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_b128 v[2:5], v[2:3], off
	s_branch .LBB0_15
.LBB0_18:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	0 1049 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v6, s26, v163
	v_dual_mov_b32 v7, v164 :: v_dual_mov_b32 v8, v161
	s_movk_i32 s0, 0xc000
	s_branch .LBB0_22
.LBB0_19:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 0 9 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB0_20:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_21:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 1088 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1088:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v0, s0, v165
	.loc	0 1049 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v8, 4, v8
	v_add_nc_u32_e32 v7, 64, v7
	s_addk_co_i32 s0, 0x400
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	0 1088 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1088:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_store_b64 v0, v[2:3] offset:32768
	.loc	0 1049 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cbranch_scc1 .LBB0_38
.LBB0_22:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	0 0 9 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1058 43 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1058:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_b32_e32 v0, 0x70, v8
	v_add_nc_u32_e32 v137, v0, v6
	.loc	0 1057 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1057:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_or_b32 v0, 0xf0, v7, v159
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1062 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1062:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v2, 7, v137
	.loc	0 1062 27 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1062:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmp_ge_i32_e32 vcc_lo, s24, v2
                                        ; implicit-def: $vgpr2_vgpr3
	s_and_saveexec_b32 s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 1065 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1065:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v137, s[18:19]
	.loc	0 1067 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1067:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e32 v5.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 1073 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1073:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e64 v139.l, v5.h
	.loc	0 1066 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1066:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_co_u32 v2, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	.loc	0 1068 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x7
	global_load_u8 v0, v[2:3], off offset:1032
	global_load_u8 v4, v[2:3], off offset:2064
	global_load_d16_u8 v5, v[2:3], off
	global_load_u8 v137, v[2:3], off offset:3096
	global_load_u8 v138, v[2:3], off offset:5160
	global_load_u8 v140, v[2:3], off offset:4128
	global_load_u8 v141, v[2:3], off offset:7224
	global_load_d16_hi_u8 v139, v[2:3], off offset:6192
	.loc	0 1068 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v0, 8, v0
	.loc	0 1069 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1069:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	.loc	0 1070 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1070:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v137
	.loc	0 1072 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v138
	.loc	0 1068 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, v0, v5
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	.loc	0 1070 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1070:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or3_b32 v0, v0, v2, v3
	.loc	0 1072 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v140, v4
	.loc	0 1074 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1074:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v141
	.loc	0 1072 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or3_b32 v0, v0, 0, 0
	.loc	0 1074 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1074:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v3, v2, v139, v3
	v_or3_b32 v2, v0, 0, 0
                                        ; implicit-def: $vgpr0
.LBB0_24:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_21
; %bb.25:                               ;   in Loop: Header=BB0_22 Depth=2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	.loc	0 1075 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1075:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s3, exec_lo
	v_cmpx_ge_i32_e64 s24, v137
	s_cbranch_execz .LBB0_20
; %bb.26:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 1077 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1077:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_co_u32 v4, s4, s18, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s19, 0, s4
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e32 v0.h, 0
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v137, v[4:5]
	global_load_d16_u8 v0, v[2:3], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmpx_gt_i32_e64 s24, v137
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 1078 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v2, 1, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v2, v[4:5]
	global_load_u8 v2, v[2:3], off
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v0, v2, 8, v0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
.LBB0_28:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, 2, v137
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 16, v2
.LBB0_30:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, 3, v137
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	.loc	0 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_u8 v0, v[138:139], off
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 24, v2
.LBB0_32:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, 4, v137
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
.LBB0_34:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, 5, v137
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	0 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 8, v0
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v3, v0, v3
.LBB0_36:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, 6, v137
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_19
; %bb.37:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[4:5], null, 0x408, v0, v[4:5]
	.loc	0 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e32 v0.l, 0
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_d16_hi_u8 v0, v[4:5], off
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
	s_branch .LBB0_19
.LBB0_38:                               ;   in Loop: Header=BB0_13 Depth=1
.Ltmp46:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp47:
	.loc	0 1096 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1096:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_and_saveexec_b32 s27, s2
	s_cbranch_execz .LBB0_11
; %bb.39:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b32_e32 v0, s22
	.loc	0 1111 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1111:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_and_saveexec_b32 s0, s23
	s_cbranch_execz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	0 1112 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1112:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_b32 v0, v[152:153], off
	.loc	0 1113 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1113:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v0, s22, v0
.LBB0_41:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	0 0 31 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:31
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	0 1098 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1098:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_or_b32 s0, s26, 16
	v_mov_b32_e32 v4, v160
	.loc	0 1098 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1098:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s24
	s_mov_b32 s28, 0
	s_cselect_b32 s29, -1, 0
	s_branch .LBB0_44
.LBB0_42:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_sub_f32_e32 v143, v168, v2
	.loc	0 1189 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1189:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v168
	.loc	0 1209 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1209:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v172, null, v3, v3, v140
	.loc	0 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v174, null, v3, v3, v139
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	.loc	0 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v177, null, v3, v3, v137
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v176, null, v3, v3, v8
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v173, v172
	.loc	0 1251 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1251:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshl_add_u32 v188, s28, 12, v160
	v_exp_f32_e32 v143, v143
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v180, v177
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v179, v176
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v175, -v172, v173, 1.0
	.loc	0 1189 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1189:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v143, 0, v143, vcc_lo
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_fmac_f32 v173, v175, v173 :: v_dual_mul_f32 v144, v166, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1226 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1226:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v166, null, v3, v3, v144
	v_div_scale_f32 v170, vcc_lo, v144, v3, v144
	v_rcp_f32_e32 v168, v166
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v169, -v166, v168, 1.0
	v_fmac_f32_e32 v168, v169, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v169, v170, v168
	v_fma_f32 v171, -v166, v169, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v169, v171, v168
	v_fma_f32 v166, -v166, v169, v170
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v166, v166, v168, v169
	v_div_fixup_f32 v6, v166, v3, v144
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v166, null, v3, v3, v141
	s_delay_alu instid0(VALU_DEP_2)
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v134, v134, v6
	v_dual_mul_f32 v130, v130, v6 :: v_dual_fmac_f32 v5, v167, v143
	v_mul_f32_e32 v128, v128, v6
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v143, null, v3, v3, v142
	.loc	0 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v168, v166
	.loc	0 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v136, v136, v6 :: v_dual_mul_f32 v135, v135, v6
	v_mul_f32_e32 v126, v126, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v144, v143
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v133, v133, v6 :: v_dual_mul_f32 v124, v124, v6
	v_dual_mul_f32 v132, v132, v6 :: v_dual_mul_f32 v131, v131, v6
	v_mul_f32_e32 v122, v122, v6
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v170, -v166, v168, 1.0
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v25, v25, v6
	v_dual_mul_f32 v129, v129, v6 :: v_dual_mul_f32 v120, v120, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v167, -v143, v144, 1.0
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v33, v33, v6 :: v_dual_fmac_f32 v168, v170, v168
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v170, s0, v141, v3, v141
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v127, v127, v6 :: v_dual_mul_f32 v118, v118, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v144, v167, v144
	v_div_scale_f32 v167, vcc_lo, v142, v3, v142
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v16, v16, v6 :: v_dual_mul_f32 v125, v125, v6
	v_dual_mul_f32 v116, v116, v6 :: v_dual_mul_f32 v123, v123, v6
	v_dual_mul_f32 v114, v114, v6 :: v_dual_mul_f32 v169, v167, v144
	v_dual_mul_f32 v121, v121, v6 :: v_dual_mul_f32 v112, v112, v6
	v_dual_mul_f32 v119, v119, v6 :: v_dual_mul_f32 v110, v110, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v171, -v143, v169, v167
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v117, v117, v6 :: v_dual_mul_f32 v108, v108, v6
	v_dual_mul_f32 v115, v115, v6 :: v_dual_mul_f32 v106, v106, v6
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v169, v171, v144
	.loc	0 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v171, v170, v168
	.loc	0 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v113, v113, v6 :: v_dual_mul_f32 v104, v104, v6
	v_dual_mul_f32 v111, v111, v6 :: v_dual_mul_f32 v102, v102, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v143, -v143, v169, v167
	.loc	0 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v167, -v166, v171, v170
	.loc	0 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v109, v109, v6 :: v_dual_mul_f32 v100, v100, v6
	v_dual_mul_f32 v107, v107, v6 :: v_dual_mul_f32 v98, v98, v6
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v169
	.loc	0 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v171, v167, v168
	.loc	0 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v167, s1, v140, v3, v140
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v144, v174
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v143, v143, v3, v142
	.loc	0 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v142, -v166, v171, v170
	.loc	0 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v166, v167, v173
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v170, null, v3, v3, v7
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v105, v105, v6 :: v_dual_mul_f32 v96, v96, v6
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v168, v142, v168, v171
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v171, -v172, v166, v167
	s_mov_b32 vcc_lo, s1
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v175, v170
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v169, -v174, v144, 1.0
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v19, v19, v6
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v166, v171, v173
	.loc	0 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e64 v142.h, 0
	.loc	0 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e64 v142.l, v1.l
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, s0, v139, v3, v139
	.loc	0 1242 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v167, -v172, v166, v167
	.loc	0 1245 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v178, -v170, v175, 1.0
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v168, v168, v3, v141
	.loc	0 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e64 v141.h, v142.h
	.loc	0 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e64 v141.l, v142.l
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v166, v167, v173, v166
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v167, null, v3, v3, v138
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v173, -v176, v179, 1.0
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	0 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cvt_pk_fp8_f32 v141.l, v143, v168
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v140, v166, v3, v140
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v166, -v177, v180, 1.0
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v15, v15, v6
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v179, v173, v179
	v_div_scale_f32 v173, s3, v8, v3, v8
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v103, v103, v6 :: v_dual_mul_f32 v94, v94, v6
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v180, v166, v180
	v_div_scale_f32 v166, s4, v137, v3, v137
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v10, v10, v6 :: v_dual_mul_f32 v101, v101, v6
	v_dual_mul_f32 v92, v92, v6 :: v_dual_mul_f32 v99, v99, v6
	v_mul_f32_e32 v90, v90, v6
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v182, v166, v180
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v175, v178, v175
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v171, v169, v144
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v178, v167
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v97, v97, v6 :: v_dual_mul_f32 v88, v88, v6
	v_dual_mul_f32 v95, v95, v6 :: v_dual_mul_f32 v86, v86, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v172, -v174, v171, v169
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v93, v93, v6 :: v_dual_mul_f32 v84, v84, v6
	v_dual_mul_f32 v91, v91, v6 :: v_dual_mul_f32 v82, v82, v6
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v171, v172, v144
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v172, s1, v7, v3, v7
	s_delay_alu instid0(TRANS32_DEP_1)
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v181, -v167, v178, 1.0
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v17, v17, v6
	v_dual_mul_f32 v89, v89, v6 :: v_dual_mul_f32 v80, v80, v6
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v169, -v174, v171, v169
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v174, v172, v175
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v178, v181, v178
	v_div_scale_f32 v181, s0, v138, v3, v138
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v87, v87, v6 :: v_dual_mul_f32 v78, v78, v6
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v169, v144, v171
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v171, v173, v179
	.loc	0 1245 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v169, -v170, v174, v172
	.loc	0 1247 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v183, v181, v178
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s1
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v139, v144, v3, v139
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v144, -v176, v171, v173
	.loc	0 1245 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v174, v169, v175
	.loc	0 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v169, -v177, v182, v166
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v85, v85, v6 :: v_dual_mul_f32 v76, v76, v6
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v171, v144, v179
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v144, -v167, v183, v181
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v170, -v170, v174, v172
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v182, v169, v180
	.loc	0 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cvt_pk_fp8_f32 v141.h, v140, v139
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v83, v83, v6 :: v_dual_mul_f32 v74, v74, v6
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v183, v144, v178
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v169, v170, v175, v174
	.loc	0 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v170, -v176, v171, v173
	.loc	0 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v144, -v177, v182, v166
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v167, -v167, v183, v181
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v7, v169, v3, v7
	.loc	0 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v166, v170, v179, v171
	.loc	0 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v81, v81, v6 :: v_dual_mul_f32 v72, v72, v6
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v180, v182
	.loc	0 1247 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	0 1245 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v8, v166, v3, v8
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v167, v167, v178, v183
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v79, v79, v6 :: v_dual_mul_f32 v70, v70, v6
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v137, v144, v3, v137
	.loc	0 1244 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1244:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cvt_pk_fp8_f32 v142.l, v7, v8
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v138, v167, v3, v138
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[7:8], v188 offset:16384
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v77, v77, v6 :: v_dual_mul_f32 v68, v68, v6
	.loc	0 1246 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1246:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cvt_pk_fp8_f32 v142.h, v137, v138
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[137:138], v188 offset:16640
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[139:140], v188 offset:16896
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[143:144], v188 offset:17152
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[166:167], v188 offset:17408
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[168:169], v188 offset:17664
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[170:171], v188 offset:17920
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[172:173], v188 offset:18176
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[174:175], v188 offset:18432
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[176:177], v188 offset:18688
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[178:179], v188 offset:18944
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[180:181], v188 offset:19200
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[182:183], v188 offset:19456
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[184:185], v188 offset:19712
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[186:187], v188 offset:19968
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[188:189], v188 offset:20224
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v75, v75, v6 :: v_dual_mul_f32 v66, v66, v6
	v_dual_mul_f32 v73, v73, v6 :: v_dual_mul_f32 v64, v64, v6
	v_dual_mul_f32 v71, v71, v6 :: v_dual_mul_f32 v62, v62, v6
	v_dual_mul_f32 v69, v69, v6 :: v_dual_mul_f32 v60, v60, v6
	v_dual_mul_f32 v67, v67, v6 :: v_dual_mul_f32 v58, v58, v6
	v_dual_mul_f32 v65, v65, v6 :: v_dual_mul_f32 v56, v56, v6
	v_dual_mul_f32 v63, v63, v6 :: v_dual_mul_f32 v54, v54, v6
	v_dual_mul_f32 v61, v61, v6 :: v_dual_mul_f32 v52, v52, v6
	v_dual_mul_f32 v59, v59, v6 :: v_dual_mul_f32 v50, v50, v6
	v_dual_mul_f32 v57, v57, v6 :: v_dual_mul_f32 v48, v48, v6
	v_dual_mul_f32 v55, v55, v6 :: v_dual_mul_f32 v46, v46, v6
	v_dual_mul_f32 v53, v53, v6 :: v_dual_mul_f32 v44, v44, v6
	v_dual_mul_f32 v51, v51, v6 :: v_dual_mul_f32 v42, v42, v6
	v_dual_mul_f32 v49, v49, v6 :: v_dual_mul_f32 v40, v40, v6
	v_dual_mul_f32 v47, v47, v6 :: v_dual_mul_f32 v38, v38, v6
	v_dual_mul_f32 v45, v45, v6 :: v_dual_mul_f32 v36, v36, v6
	v_dual_mul_f32 v43, v43, v6 :: v_dual_mul_f32 v34, v34, v6
	v_dual_mul_f32 v41, v41, v6 :: v_dual_mul_f32 v32, v32, v6
	v_dual_mul_f32 v39, v39, v6 :: v_dual_mul_f32 v30, v30, v6
	v_dual_mul_f32 v37, v37, v6 :: v_dual_mul_f32 v28, v28, v6
	v_dual_mul_f32 v35, v35, v6 :: v_dual_mul_f32 v26, v26, v6
	v_dual_mul_f32 v31, v31, v6 :: v_dual_mul_f32 v24, v24, v6
	v_dual_mul_f32 v29, v29, v6 :: v_dual_mul_f32 v22, v22, v6
	v_dual_mul_f32 v27, v27, v6 :: v_dual_mul_f32 v20, v20, v6
	v_dual_mul_f32 v23, v23, v6 :: v_dual_mul_f32 v18, v18, v6
	v_dual_mul_f32 v21, v21, v6 :: v_dual_mul_f32 v14, v14, v6
	v_dual_mul_f32 v13, v13, v6 :: v_dual_mul_f32 v12, v12, v6
	v_mul_f32_e32 v11, v11, v6
	v_mul_f32_e32 v9, v9, v6
	.loc	0 1253 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1253:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[7:8], v[141:142], v[129:136]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[137:138], v[141:142], v[121:128]
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[139:140], v[141:142], v[113:120]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[143:144], v[141:142], v[105:112]
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[166:167], v[141:142], v[97:104]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[168:169], v[141:142], v[89:96]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[170:171], v[141:142], v[81:88]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[172:173], v[141:142], v[73:80]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[174:175], v[141:142], v[65:72]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[176:177], v[141:142], v[57:64]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[178:179], v[141:142], v[49:56]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[180:181], v[141:142], v[41:48]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[182:183], v[141:142], v[33:40]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[184:185], v[141:142], v[25:32]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[186:187], v[141:142], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[188:189], v[141:142], v[9:16]
	v_dual_mov_b32 v167, v5 :: v_dual_mov_b32 v166, v3
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
.LBB0_43:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 1116 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v4, 0x100, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v168, v2
	.loc	0 1116 40 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_add_co_i32 s28, s28, 1
	.loc	0 1116 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s28, 4
	.loc	0 1116 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cbranch_scc0 .LBB0_10
.LBB0_44:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_49 Depth 3
	.loc	0 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s28, 1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB0_47
; %bb.45:                               ;   in Loop: Header=BB0_44 Depth=2
	s_cmp_eq_u32 s28, 1
	s_mov_b32 s0, s29
	s_cbranch_scc1 .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 1118 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1118:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cmp_eq_u32 s28, 2
	.loc	0 1118 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1118:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cselect_b32 s0, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s24, s0
	s_cselect_b32 s0, -1, 0
.LBB0_47:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_84
; %bb.48:                               ;   in Loop: Header=BB0_44 Depth=2
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v2, v154
	v_mov_b32_e32 v3, v155
	s_movk_i32 s0, 0xc000
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_mov_b32_e32 v144, v137
.LBB0_49:                               ;   Parent Loop BB0_13 Depth=1
                                        ;     Parent Loop BB0_44 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	.loc	0 1151 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1151:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x3
	global_load_b64 v[173:174], v[2:3], off offset:-48
	global_load_b64 v[175:176], v[2:3], off offset:-32
	global_load_b64 v[177:178], v[2:3], off offset:-16
	global_load_b64 v[179:180], v[2:3], off
	.loc	0 1154 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1154:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v169, s0, v4
	.loc	0 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_co_u32 v2, vcc_lo, v2, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	.loc	0 1154 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1154:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_2addr_stride64_b64 v[5:8], v169 offset0:32 offset1:34
	ds_load_2addr_stride64_b64 v[169:172], v169 offset0:36 offset1:38
	.loc	0 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_addk_co_i32 s0, 0x1000
	.loc	0 1159 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1159:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	.loc	0 1155 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1155:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[5:6], v[173:174], v[137:144]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[7:8], v[175:176], v[137:144]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[169:170], v[177:178], v[137:144]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[171:172], v[179:180], v[137:144]
	.loc	0 1139 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cbranch_scc1 .LBB0_49
; %bb.50:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 1161 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1161:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_lshl4_add_u32 s0, s28, s26
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v170, 0
	.loc	0 1162 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1162:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 15
	.loc	0 1162 49 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1162:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s25
	s_cselect_b32 s30, -1, 0
	s_cmp_gt_i32 s1, s25
	s_cselect_b32 s1, -1, 0
	.loc	0 1164 28 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1164:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s1, s23
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s3
	s_cbranch_execz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 1164 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1164:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_b32 v170, v[150:151], off
.LBB0_52:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, s0, v146
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp48:
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e32 vcc_lo, s24, v2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[5:6], null, 0x408, v2, s[20:21]
	global_load_d16_b16 v3, v[5:6], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v173, v3.l
.Ltmp49:
.LBB0_54:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v5, 1, v2
.Ltmp50:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_i32_e64 s0, s24, v2
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[6:7], null, 0x408, v5, s[20:21]
	global_load_d16_b16 v3, v[6:7], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v175, v3.l
.Ltmp51:
.LBB0_56:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v6, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp52:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s1, s24, v6
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[7:8], null, 0x408, v6, s[20:21]
	global_load_d16_b16 v3, v[7:8], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v3.l
.Ltmp53:
.LBB0_58:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v7, 3, v2
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v178, 0
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp54:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s3, s24, v7
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[171:172], null, 0x408, v7, s[20:21]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v178, v3.l
.Ltmp55:
.LBB0_60:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v8, 4, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp56:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s4, s24, v8
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[171:172], null, 0x408, v8, s[20:21]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v176, v3.l
.Ltmp57:
.LBB0_62:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v169, 5, v2
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp58:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s5, s24, v169
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[171:172], null, 0x408, v169, s[20:21]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v179, v3.l
.Ltmp59:
.LBB0_64:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v171, 6, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp60:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s6, s24, v171
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[180:181], null, 0x408, v171, s[20:21]
	global_load_d16_b16 v3, v[180:181], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v177, v3.l
.Ltmp61:
.LBB0_66:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v172, 7, v2
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v180, 0
	s_delay_alu instid0(VALU_DEP_2)
.Ltmp62:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s7, s24, v172
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[180:181], null, 0x408, v172, s[20:21]
	global_load_d16_b16 v180, v[180:181], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v180, v180.l
.Ltmp63:
.LBB0_68:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mul_f32 v137, v0, v137 :: v_dual_mul_f32 v138, v0, v138
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s8, v2, v170
	v_cmp_lt_i32_e64 s9, v2, v170
	v_dual_mul_f32 v139, v0, v139 :: v_dual_mul_f32 v140, v0, v140
	v_cmp_le_i32_e64 s10, v6, v170
	v_dual_mul_f32 v137, v137, v173 :: v_dual_mul_f32 v138, v138, v175
	s_or_b32 s8, s30, s8
	s_or_b32 s9, s30, s9
	.loc	0 1178 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s23, s8
	v_dual_mul_f32 v173, v139, v174 :: v_dual_mul_f32 v144, v0, v144
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v137, 0xff800000, v137, s8
	s_or_b32 s10, s30, s10
	v_cmp_le_i32_e64 s8, v7, v170
	s_and_b32 s9, s23, s9
	v_mul_f32_e32 v142, v0, v142
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v139, 0xff800000, v138, s9
	s_and_b32 s9, s23, s10
	v_mul_f32_e32 v140, v140, v178
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, 0xff800000, v173, s9
	v_mul_f32_e32 v173, v0, v141
	s_or_b32 s10, s30, s8
	v_cmp_le_i32_e64 s8, v8, v170
	v_cmp_le_i32_e64 s9, v169, v170
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s10, s23, s10
	v_mul_f32_e32 v143, v0, v143
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v141, 0xff800000, v140, s10
	v_mul_f32_e32 v140, v173, v176
	s_or_b32 s8, s30, s8
	v_mul_f32_e32 v173, v142, v179
	s_or_b32 s9, s30, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s23, s8
	v_dual_mul_f32 v143, v143, v177 :: v_dual_mul_f32 v144, v144, v180
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v142, 0xff800000, v140, s8
	s_and_b32 s8, s23, s9
	v_cmp_le_i32_e64 s9, v172, v170
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v140, 0xff800000, v173, s8
	v_cmp_le_i32_e64 s8, v171, v170
.Ltmp64:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v170, v137, 0xff800000, v139
	s_or_b32 s9, s30, s9
	s_or_b32 s8, s30, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v170, v170, v138, v141
.Ltmp65:
	.loc	0 1178 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s23, s8
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v173, 0xff800000, v143, s8
	s_and_b32 s8, s23, s9
.Ltmp66:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v143, v170, v142, v140
.Ltmp67:
	.loc	0 1178 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v144, 0xff800000, v144, s8
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp68:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v175, v143, v173, v144
.Ltmp69:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1186:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ]
	ds_bpermute_b32 v176, v158, v175
.Ltmp70:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v2, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v3, v2.l
.LBB0_70:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v170, 0
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[177:178], null, 0x408, v5, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v170, v2.l
.LBB0_72:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[5:6], null, 0x408, v6, s[12:13]
	global_load_d16_b16 v2, v[5:6], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v143, v2.l
.LBB0_74:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[6:7], null, 0x408, v7, s[12:13]
	global_load_d16_b16 v2, v[6:7], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v6, v2.l
.LBB0_76:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[7:8], null, 0x408, v8, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v5, v2.l
.LBB0_78:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v8, 0
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB0_85
; %bb.79:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 14 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s6
	s_cbranch_execnz .LBB0_86
.LBB0_80:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 14                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB0_82
.LBB0_81:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[171:172], null, 0x408, v172, s[12:13]
	global_load_d16_b16 v2, v[171:172], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v169, v2.l
.Ltmp71:
.LBB0_82:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
.Ltmp72:
	.loc	2 454 44 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1187:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_max3_num_f32 v2, v168, v175, v176
.Ltmp73:
	.loc	0 1222 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1222:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v7, v137, v2 :: v_dual_sub_f32 v138, v138, v2
	v_dual_sub_f32 v137, v139, v2 :: v_dual_sub_f32 v140, v140, v2
	v_dual_sub_f32 v139, v141, v2 :: v_dual_sub_f32 v144, v144, v2
	v_dual_mul_f32 v7, 0x3fb8aa3b, v7 :: v_dual_mul_f32 v138, 0x3fb8aa3b, v138
	v_dual_sub_f32 v141, v142, v2 :: v_dual_sub_f32 v142, v173, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_mul_f32 v140, 0x3fb8aa3b, v140
	v_exp_f32_e32 v7, v7
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v2
	v_exp_f32_e32 v138, v138
	v_dual_mul_f32 v139, 0x3fb8aa3b, v139 :: v_dual_mul_f32 v144, 0x3fb8aa3b, v144
	v_exp_f32_e32 v137, v137
	v_exp_f32_e32 v140, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v139, v139
	.loc	0 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, v7, 0, vcc_lo
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_mul_f32_e32 v171, 0x3fb8aa3b, v142
	v_exp_f32_e32 v144, v144
	.loc	0 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v142, v7, v3
	v_exp_f32_e32 v172, v141
	.loc	0 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cndmask_b32_e64 v137, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v3, v7, v137
	.loc	0 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v141, v137, v170
	.loc	0 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cndmask_b32_e64 v170, v139, 0, vcc_lo
	v_exp_f32_e32 v137, v171
	v_cndmask_b32_e64 v171, v172, 0, vcc_lo
	v_cndmask_b32_e64 v172, v140, 0, vcc_lo
	.loc	0 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v140, v138, v143
	.loc	0 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cndmask_b32_e64 v143, v144, 0, vcc_lo
	.loc	0 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v139, v170, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v7, v171, v5 :: v_dual_mul_f32 v8, v172, v8
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_add_f32 v3, v138, v3 :: v_dual_mul_f32 v138, v143, v169
.Ltmp74:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v6, v142, 0, v141
.Ltmp75:
	.loc	0 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cndmask_b32_e64 v5, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v3, v170, v3
.Ltmp76:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v6, v6, v140, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
.Ltmp77:
	.loc	0 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v137, v5, v174
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v3, v171, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp78:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v6, v6, v7, v8
.Ltmp79:
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v3, v172, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp80:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v144, v6, v137, v138
.Ltmp81:
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v3, v5, v3
.Ltmp82:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1220:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ]
	ds_bpermute_b32 v169, v158, v144
.Ltmp83:
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v5, v143, v3
.Ltmp84:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1209:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ]
	ds_bpermute_b32 v6, v158, v5
.Ltmp85:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1220:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v3, v169, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v143, v144, v3
	v_mov_b32_e32 v3, v166
.Ltmp86:
	.loc	0 1222 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1222:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmpx_lt_f32_e32 0, v143
	s_cbranch_execz .LBB0_42
; %bb.83:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 1223 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1223:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v3, null, 0x43e00000, 0x43e00000, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v144, v3
	v_fma_f32 v169, -v3, v144, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, vcc_lo, v143, 0x43e00000, v143
	v_mul_f32_e32 v170, v169, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v171, -v3, v170, v169
	v_fmac_f32_e32 v170, v171, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v3, v170, v169
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v144, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v3, 0x43e00000, v143
.Ltmp87:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1223:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max_num_f32_e32 v3, 0x1f800000, v3
	s_branch .LBB0_42
.Ltmp88:
.LBB0_84:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	2 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v2, v168
	s_branch .LBB0_43
.LBB0_85:                               ;   in Loop: Header=BB0_44 Depth=2
.Ltmp89:
	.loc	0 760 23 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[7:8], null, 0x408, v169, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v8, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s6
	s_cbranch_execz .LBB0_80
.LBB0_86:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[177:178], null, 0x408, v171, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s7
	s_cbranch_execnz .LBB0_81
	s_branch .LBB0_82
.Ltmp90:
.LBB0_87:
	.loc	0 1267 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1267:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_and_saveexec_b32 s0, s23
	s_cbranch_execz .LBB0_89
; %bb.88:
	.loc	0 1270 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1270:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v0, null, v167, v167, 1.0
	v_div_scale_f32 v3, vcc_lo, 1.0, v167, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v1, v0
	v_fma_f32 v2, -v0, v1, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v2, v1
	v_mul_f32_e32 v2, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v0, v2, v3
	v_fmac_f32_e32 v2, v4, v1
	.loc	0 1272 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1272:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_lo_u32 v4, 0x1800, v145
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	0 1270 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1270:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v0, -v0, v2, v3
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v0, v0, v1, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	.loc	0 1272 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1272:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshl_add_u32 v2, v147, 8, v4
	.loc	0 1270 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1270:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmp_lt_f32_e32 vcc_lo, 0, v167
	.loc	0 1278 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b32_e32 v1, 0
	.loc	0 1270 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1270:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v3, v0, v167, 1.0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1274 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1274:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, v2, v146
	.loc	0 1270 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1270:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 1278 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_mul_f32_e32 v139, v166, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v137, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, s15, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	.loc	0 1278 60 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v0, v129, v139 :: v_dual_mul_f32 v3, v132, v139
	v_dual_mul_f32 v1, v130, v139 :: v_dual_mul_f32 v2, v131, v139
	v_dual_mul_f32 v5, v134, v139 :: v_dual_mul_f32 v4, v133, v139
	v_dual_mul_f32 v7, v136, v139 :: v_dual_mul_f32 v6, v135, v139
	v_dual_mul_f32 v121, v121, v139 :: v_dual_mul_f32 v122, v122, v139
	v_dual_mul_f32 v123, v123, v139 :: v_dual_mul_f32 v124, v124, v139
	v_dual_mul_f32 v125, v125, v139 :: v_dual_mul_f32 v126, v126, v139
	v_dual_mul_f32 v127, v127, v139 :: v_dual_mul_f32 v128, v128, v139
	v_dual_mul_f32 v113, v113, v139 :: v_dual_mul_f32 v112, v112, v139
	v_dual_mul_f32 v97, v97, v139 :: v_dual_mul_f32 v98, v98, v139
	v_dual_mul_f32 v99, v99, v139 :: v_dual_mul_f32 v100, v100, v139
	v_dual_mul_f32 v101, v101, v139 :: v_dual_mul_f32 v114, v114, v139
	v_dual_mul_f32 v115, v115, v139 :: v_dual_mul_f32 v116, v116, v139
	v_dual_mul_f32 v117, v117, v139 :: v_dual_mul_f32 v102, v102, v139
	v_dual_mul_f32 v103, v103, v139 :: v_dual_mul_f32 v104, v104, v139
	v_dual_mul_f32 v118, v118, v139 :: v_dual_mul_f32 v119, v119, v139
	v_dual_mul_f32 v120, v120, v139 :: v_dual_mul_f32 v105, v105, v139
	v_dual_mul_f32 v106, v106, v139 :: v_dual_mul_f32 v107, v107, v139
	v_dual_mul_f32 v108, v108, v139 :: v_dual_mul_f32 v109, v109, v139
	v_dual_mul_f32 v110, v110, v139 :: v_dual_mul_f32 v111, v111, v139
	.loc	0 1278 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x7
	global_store_b128 v[137:138], v[0:3], off
	global_store_b128 v[137:138], v[4:7], off offset:16
	global_store_b128 v[137:138], v[121:124], off offset:64
	global_store_b128 v[137:138], v[125:128], off offset:80
	global_store_b128 v[137:138], v[113:116], off offset:128
	global_store_b128 v[137:138], v[117:120], off offset:144
	global_store_b128 v[137:138], v[105:108], off offset:192
	global_store_b128 v[137:138], v[109:112], off offset:208
	.loc	0 1278 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v0, v89, v139 :: v_dual_mul_f32 v3, v92, v139
	v_dual_mul_f32 v1, v90, v139 :: v_dual_mul_f32 v2, v91, v139
	v_dual_mul_f32 v5, v94, v139 :: v_dual_mul_f32 v4, v93, v139
	v_dual_mul_f32 v7, v96, v139 :: v_dual_mul_f32 v6, v95, v139
	v_dual_mul_f32 v81, v81, v139 :: v_dual_mul_f32 v82, v82, v139
	v_dual_mul_f32 v83, v83, v139 :: v_dual_mul_f32 v84, v84, v139
	v_dual_mul_f32 v85, v85, v139 :: v_dual_mul_f32 v86, v86, v139
	v_dual_mul_f32 v87, v87, v139 :: v_dual_mul_f32 v88, v88, v139
	.loc	0 1278 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[97:100], off offset:256
	global_store_b128 v[137:138], v[101:104], off offset:272
	global_store_b128 v[137:138], v[0:3], off offset:320
	global_store_b128 v[137:138], v[4:7], off offset:336
	global_store_b128 v[137:138], v[81:84], off offset:384
	global_store_b128 v[137:138], v[85:88], off offset:400
	.loc	0 1278 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v0, v73, v139 :: v_dual_mul_f32 v3, v76, v139
	v_dual_mul_f32 v1, v74, v139 :: v_dual_mul_f32 v2, v75, v139
	v_dual_mul_f32 v5, v78, v139 :: v_dual_mul_f32 v4, v77, v139
	v_dual_mul_f32 v7, v80, v139 :: v_dual_mul_f32 v6, v79, v139
	v_dual_mul_f32 v65, v65, v139 :: v_dual_mul_f32 v66, v66, v139
	v_dual_mul_f32 v67, v67, v139 :: v_dual_mul_f32 v68, v68, v139
	v_dual_mul_f32 v69, v69, v139 :: v_dual_mul_f32 v70, v70, v139
	v_dual_mul_f32 v71, v71, v139 :: v_dual_mul_f32 v72, v72, v139
	v_dual_mul_f32 v57, v57, v139 :: v_dual_mul_f32 v58, v58, v139
	v_dual_mul_f32 v59, v59, v139 :: v_dual_mul_f32 v60, v60, v139
	v_dual_mul_f32 v61, v61, v139 :: v_dual_mul_f32 v62, v62, v139
	v_dual_mul_f32 v63, v63, v139 :: v_dual_mul_f32 v64, v64, v139
	.loc	0 1278 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:448
	global_store_b128 v[137:138], v[4:7], off offset:464
	global_store_b128 v[137:138], v[65:68], off offset:512
	global_store_b128 v[137:138], v[69:72], off offset:528
	global_store_b128 v[137:138], v[57:60], off offset:576
	global_store_b128 v[137:138], v[61:64], off offset:592
	.loc	0 1278 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v0, v49, v139 :: v_dual_mul_f32 v3, v52, v139
	v_dual_mul_f32 v1, v50, v139 :: v_dual_mul_f32 v2, v51, v139
	v_dual_mul_f32 v5, v54, v139 :: v_dual_mul_f32 v4, v53, v139
	v_dual_mul_f32 v7, v56, v139 :: v_dual_mul_f32 v6, v55, v139
	v_dual_mul_f32 v41, v41, v139 :: v_dual_mul_f32 v42, v42, v139
	v_dual_mul_f32 v43, v43, v139 :: v_dual_mul_f32 v44, v44, v139
	v_dual_mul_f32 v45, v45, v139 :: v_dual_mul_f32 v46, v46, v139
	v_dual_mul_f32 v47, v47, v139 :: v_dual_mul_f32 v48, v48, v139
	v_dual_mul_f32 v33, v33, v139 :: v_dual_mul_f32 v34, v34, v139
	v_dual_mul_f32 v35, v35, v139 :: v_dual_mul_f32 v36, v36, v139
	v_dual_mul_f32 v37, v37, v139 :: v_dual_mul_f32 v38, v38, v139
	v_dual_mul_f32 v39, v39, v139 :: v_dual_mul_f32 v40, v40, v139
	.loc	0 1278 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:640
	global_store_b128 v[137:138], v[4:7], off offset:656
	global_store_b128 v[137:138], v[41:44], off offset:704
	global_store_b128 v[137:138], v[45:48], off offset:720
	global_store_b128 v[137:138], v[33:36], off offset:768
	global_store_b128 v[137:138], v[37:40], off offset:784
	.loc	0 1278 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v0, v25, v139 :: v_dual_mul_f32 v3, v28, v139
	v_dual_mul_f32 v1, v26, v139 :: v_dual_mul_f32 v2, v27, v139
	v_dual_mul_f32 v5, v30, v139 :: v_dual_mul_f32 v4, v29, v139
	v_dual_mul_f32 v7, v32, v139 :: v_dual_mul_f32 v6, v31, v139
	v_dual_mul_f32 v17, v17, v139 :: v_dual_mul_f32 v18, v18, v139
	v_dual_mul_f32 v19, v19, v139 :: v_dual_mul_f32 v20, v20, v139
	v_dual_mul_f32 v21, v21, v139 :: v_dual_mul_f32 v22, v22, v139
	v_dual_mul_f32 v23, v23, v139 :: v_dual_mul_f32 v24, v24, v139
	v_dual_mul_f32 v8, v9, v139 :: v_dual_mul_f32 v9, v10, v139
	v_dual_mul_f32 v10, v11, v139 :: v_dual_mul_f32 v11, v12, v139
	v_dual_mul_f32 v12, v13, v139 :: v_dual_mul_f32 v13, v14, v139
	v_dual_mul_f32 v14, v15, v139 :: v_dual_mul_f32 v15, v16, v139
	.loc	0 1278 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:832
	global_store_b128 v[137:138], v[4:7], off offset:848
	global_store_b128 v[137:138], v[17:20], off offset:896
	global_store_b128 v[137:138], v[21:24], off offset:912
	global_store_b128 v[137:138], v[8:11], off offset:960
	global_store_b128 v[137:138], v[12:15], off offset:976
.Ltmp91:
.LBB0_89:
	.loc	0 1345 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1345:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp92:
.Lfunc_end0:
	.size	attention_fp8_e4m3_fa2_gqa_gfx1201, .Lfunc_end0-attention_fp8_e4m3_fa2_gqa_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 60
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
		.amdhsa_next_free_vgpr 190
		.amdhsa_next_free_sgpr 31
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-attention_fp8_e4m3_fa2_gqa_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.num_vgpr, 190
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.numbered_sgpr, 31
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7996
; TotalNumSgprs: 33
; NumVgprs: 190
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 33
; NumVGPRsForWavesPerEU: 190
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
	.globl	attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201,@function
attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201: ; @attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
.Lfunc_begin1:
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.file	4 "/opt/rocm/core/include/hip/amd_detail" "amd_hip_runtime.h" md5 0xe1e2844c29b46b290fe1912868fc4e78
	.loc	4 248 59 prologue_end           ; /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:248:59 @[ /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:299:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1423:26 ] ]
	s_load_b32 s2, s[0:1], 0x20
.Ltmp93:
	.loc	0 1424 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1424:26
	v_lshrrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1426 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1426:39
	v_lshl_or_b32 v1, ttmp9, 2, v1
	.loc	0 1427 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1427:25
	s_wait_kmcnt 0x0
	s_mul_i32 s2, s2, 24
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	.loc	0 1427 11 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1427:11
	v_cmp_gt_i32_e32 vcc_lo, s2, v1
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB1_2
; %bb.1:
	.loc	0 1429 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1429:22
	v_mul_hi_i32 v2, 0x2aaaaaab, v1
.Ltmp94:
	.loc	4 248 59                        ; /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:248:59 @[ /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:299:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1423:26 ] ]
	s_load_b128 s[0:3], s[0:1], 0x0
.Ltmp95:
	.loc	0 1431 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:25
	v_mov_b32_e32 v9, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 1431 48 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:48
	v_mov_b32_e32 v11, v9
	.loc	0 1429 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1429:22
	v_lshrrev_b32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v2, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v2, v2, v3
	.loc	0 1430 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1430:21
	v_mul_lo_u32 v3, v2, 24
	.loc	0 1431 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:40
	v_mul_lo_u32 v8, 0x1800, v2
	.loc	0 1433 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1433:24
	v_lshlrev_b32_e32 v2, 3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v12, 0xf8, v2
	.loc	0 1430 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1430:21
	v_sub_nc_u32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 1433 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1433:16
	v_lshlrev_b32_e32 v4, 2, v12
	.loc	0 1431 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:62
	v_lshlrev_b32_e32 v10, 8, v1
	.loc	0 1431 25 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:25
	v_lshlrev_b64_e32 v[0:1], 2, v[8:9]
	.loc	0 1432 24 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1432:24
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]
	.loc	0 1451 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1451:5
	v_lshlrev_b32_e32 v12, 1, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	.loc	0 1431 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:48
	v_lshlrev_b64_e32 v[2:3], 2, v[10:11]
	.loc	0 1432 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1432:47
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]
	.loc	0 1431 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:25
	s_wait_kmcnt 0x0
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1431 48 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:48
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1433 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1433:16
	v_add_co_u32 v4, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v1, vcc_lo
	.loc	0 1432 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1432:24
	v_add_co_u32 v8, vcc_lo, s2, v8
	.loc	0 1433 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1433:16
	s_clause 0x1
	global_load_b128 v[0:3], v[4:5], off
	global_load_b128 v[4:7], v[4:5], off offset:16
	.loc	0 1432 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1432:24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s3, v9, vcc_lo
	.loc	0 1432 47 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1432:47
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	.loc	0 1447 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1447:22
	s_wait_loadcnt 0x1
	v_cvt_f16_f32_e32 v0.l, v0
	.loc	0 1447 43 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1447:43
	v_cvt_f16_f32_e32 v0.h, v1
	.loc	0 1448 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1448:22
	v_cvt_f16_f32_e32 v1.l, v2
	.loc	0 1449 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1449:22
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v2.l, v4
	.loc	0 1451 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1451:5
	v_add_co_u32 v4, vcc_lo, v8, v12
	.loc	0 1448 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1448:43
	v_cvt_f16_f32_e32 v1.h, v3
	.loc	0 1449 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1449:43
	v_cvt_f16_f32_e32 v2.h, v5
	.loc	0 1450 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1450:22
	v_cvt_f16_f32_e32 v3.l, v6
	.loc	0 1450 43 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1450:43
	v_cvt_f16_f32_e32 v3.h, v7
	.loc	0 1451 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1451:5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v9, vcc_lo
	.loc	0 1451 32 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1451:32
	global_store_b128 v[4:5], v[0:3], off
.LBB1_2:
	.loc	0 1452 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1452:1
	s_endpgm
.Ltmp96:
.Lfunc_end1:
	.size	attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201, .Lfunc_end1-attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 13
		.amdhsa_next_free_sgpr 4
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.num_vgpr, 13
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.numbered_sgpr, 4
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 388
; TotalNumSgprs: 6
; NumVgprs: 13
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 6
; NumVGPRsForWavesPerEU: 13
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
	.protected	attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
	.globl	attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201,@function
attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201: ; @attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
.Lfunc_begin2:
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.loc	4 248 59 prologue_end           ; /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:248:59 @[ /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:299:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1471:26 ] ]
	s_load_b32 s2, s[0:1], 0x18
.Ltmp97:
	.loc	0 1472 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1472:26
	v_lshrrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1474 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1474:39
	v_lshl_or_b32 v1, ttmp9, 2, v1
	.loc	0 1475 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1475:25
	s_wait_kmcnt 0x0
	s_mul_i32 s2, s2, 24
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	.loc	0 1475 11 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1475:11
	v_cmp_gt_i32_e32 vcc_lo, s2, v1
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB2_3
; %bb.1:
	.loc	0 1477 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1477:22
	v_mul_hi_i32 v2, 0x2aaaaaab, v1
.Ltmp98:
	.loc	4 248 59                        ; /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:248:59 @[ /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:299:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1471:26 ] ]
	s_load_b128 s[8:11], s[0:1], 0x0
.Ltmp99:
	.loc	1 0 0 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:0 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:522:14 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:18 ] ] ]
	v_mbcnt_lo_u32_b32 v12, -1, 0
.Ltmp100:
	.loc	0 1473 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1473:26
	v_and_b32_e32 v14, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
.Ltmp101:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:18 ] ]
	v_xor_b32_e32 v13, 8, v12
.Ltmp102:
	.loc	0 1477 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1477:22
	v_lshrrev_b32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v2, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1479 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:25
	v_dual_mov_b32 v3, 0 :: v_dual_add_nc_u32 v2, v2, v3
	.loc	0 1478 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1478:21
	v_mul_lo_u32 v4, v2, 24
	.loc	0 1479 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:40
	v_mul_lo_u32 v2, 0x1800, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 1478 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1478:21
	v_sub_nc_u32_e32 v6, v1, v4
	.loc	0 1479 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:25
	v_lshlrev_b64_e32 v[4:5], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1479 62 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:62
	v_lshlrev_b32_e32 v2, 8, v6
	.loc	0 1479 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:25
	s_wait_kmcnt 0x0
	v_add_co_u32 v0, vcc_lo, s8, v4
	.loc	0 1480 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1480:22
	v_lshlrev_b32_e32 v4, 5, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1479 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:48
	v_lshlrev_b64_e32 v[6:7], 2, v[2:3]
	.loc	0 1479 25 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:25
	v_add_co_ci_u32_e64 v2, null, s9, v5, vcc_lo
	.loc	0 1479 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:48
	v_add_co_u32 v0, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v7, vcc_lo
	.loc	0 1480 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1480:22
	v_add_co_u32 v8, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, 0, v2, vcc_lo
	s_clause 0x1
	global_load_b128 v[4:7], v[8:9], off
	global_load_b128 v[8:11], v[8:9], off offset:16
.Ltmp103:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1485:9 ]
	s_wait_loadcnt 0x1
	v_max_num_f32_e64 v0, |v5|, |v5|
	v_max_num_f32_e64 v2, |v4|, |v4|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v2, v0
.Ltmp104:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:18 ] ]
	v_xor_b32_e32 v2, 16, v12
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:18 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:18 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v12, v2, vcc_lo
.Ltmp105:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:18 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:18 ] ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v13, v12, v13 :: v_dual_lshlrev_b32 v2, 2, v2
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:18 ] ]
	v_lshlrev_b32_e32 v13, 2, v13
.Ltmp106:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1487:9 ]
	v_max3_num_f32 v0, v0, |v6|, |v7|
.Ltmp107:
	.loc	2 454 44 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1489:9 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v0, v0, |v8|, |v9|
.Ltmp108:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1491:9 ]
	v_max3_num_f32 v0, v0, |v10|, |v11|
.Ltmp109:
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:18 ] ]
	ds_bpermute_b32 v2, v2, v0
.Ltmp110:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:9 ]
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v2
.Ltmp111:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:18 ] ]
	ds_bpermute_b32 v2, v13, v0
.Ltmp112:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1494:18 ] ]
	v_xor_b32_e32 v13, 4, v12
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1494:18 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1494:18 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v12, v13, vcc_lo
.Ltmp113:
	.loc	2 454 44 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:9 ]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v2, v2, v2 :: v_dual_lshlrev_b32 v13, 2, v13
	v_max_num_f32_e32 v0, v0, v2
.Ltmp114:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1494:18 ] ]
	ds_bpermute_b32 v2, v13, v0
.Ltmp115:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1495:18 ] ]
	v_xor_b32_e32 v13, 2, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1495:18 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1495:18 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v12, v13, vcc_lo
.Ltmp116:
	.loc	2 454 44 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1494:9 ]
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v2, v2, v2 :: v_dual_lshlrev_b32 v13, 2, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v2
.Ltmp117:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1495:18 ] ]
	ds_bpermute_b32 v2, v13, v0
.Ltmp118:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:18 ] ]
	v_xor_b32_e32 v13, 1, v12
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:18 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:18 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:18 ] ]
	v_lshlrev_b32_e32 v12, 2, v12
.Ltmp119:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1495:9 ]
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v2, v2, v2
	v_max_num_f32_e32 v0, v0, v2
.Ltmp120:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:18 ] ]
	ds_bpermute_b32 v2, v12, v0
.Ltmp121:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:9 ]
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v2
.Ltmp122:
	.loc	0 1497 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1497:21
	v_div_scale_f32 v2, null, 0x43e00000, 0x43e00000, v0
	v_div_scale_f32 v15, vcc_lo, v0, 0x43e00000, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v12, v2
	v_fma_f32 v13, -v2, v12, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v12, v13, v12
	v_mul_f32_e32 v13, v15, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v2, v13, v15
	v_fmac_f32_e32 v13, v16, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f32 v2, -v2, v13, v15
	.loc	0 1480 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1480:30
	v_lshlrev_b32_e32 v15, 3, v14
	.loc	0 1497 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1497:21
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v2, v2, v12, v13
	v_cmp_neq_f32_e32 vcc_lo, 0, v0
	.loc	0 1499 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:10
	v_mov_b16_e32 v13.l, v3.l
	.loc	0 1500 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:10
	v_mov_b16_e32 v13.h, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 1497 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1497:21
	v_div_fixup_f32 v2, v2, 0x43e00000, v0
	.loc	0 1499 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:10
	v_mov_b16_e32 v12.l, v13.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	.loc	0 1500 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:10
	v_mov_b16_e32 v12.h, v13.h
	.loc	0 1497 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1497:21
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 1.0, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_div_scale_f32 v30, null, v0, v0, v11
	.loc	0 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_div_scale_f32 v16, null, v0, v0, v4
	.loc	0 1499 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_div_scale_f32 v18, null, v0, v0, v5
	.loc	0 1500 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_div_scale_f32 v20, null, v0, v0, v6
	.loc	0 1500 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_div_scale_f32 v22, null, v0, v0, v7
	.loc	0 1503 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_rcp_f32_e32 v38, v30
	.loc	0 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_rcp_f32_e32 v31, v16
	.loc	0 1499 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_rcp_f32_e32 v32, v18
	.loc	0 1502 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_div_scale_f32 v24, null, v0, v0, v8
	.loc	0 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_rcp_f32_e32 v33, v20
	.loc	0 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_div_scale_f32 v26, null, v0, v0, v9
	.loc	0 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_rcp_f32_e32 v34, v22
	.loc	0 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_div_scale_f32 v39, s8, v11, v0, v11
	.loc	0 1503 45 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_div_scale_f32 v28, null, v0, v0, v10
	.loc	0 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_fma_f32 v47, -v30, v38, 1.0
	.loc	0 1502 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_rcp_f32_e32 v35, v24
	.loc	0 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_fma_f32 v40, -v16, v31, 1.0
	.loc	0 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_rcp_f32_e32 v36, v26
	.loc	0 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_fma_f32 v41, -v18, v32, 1.0
	.loc	0 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_fmac_f32_e32 v38, v47, v38
	.loc	0 1503 45 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_rcp_f32_e32 v37, v28
	.loc	0 1500 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_fma_f32 v42, -v20, v33, 1.0
	.loc	0 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_div_scale_f32 v17, vcc_lo, v4, v0, v4
	v_fmac_f32_e32 v31, v40, v31
	.loc	0 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_mul_f32_e32 v47, v39, v38
	.loc	0 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_fma_f32 v43, -v22, v34, 1.0
	.loc	0 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_div_scale_f32 v19, s2, v5, v0, v5
	.loc	0 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_fma_f32 v44, -v24, v35, 1.0
	.loc	0 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_div_scale_f32 v21, s3, v6, v0, v6
	.loc	0 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_fma_f32 v55, -v30, v47, v39
	.loc	0 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_dual_fmac_f32 v32, v41, v32 :: v_dual_fmac_f32 v33, v42, v33
	.loc	0 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_fma_f32 v45, -v26, v36, 1.0
	.loc	0 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_mul_f32_e32 v40, v17, v31
	.loc	0 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_div_scale_f32 v23, s4, v7, v0, v7
	v_fmac_f32_e32 v34, v43, v34
	.loc	0 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_fma_f32 v46, -v28, v37, 1.0
	.loc	0 1503 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_fmac_f32_e32 v47, v55, v38
	.loc	0 1499 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_mul_f32_e32 v41, v19, v32
	.loc	0 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_div_scale_f32 v25, s5, v8, v0, v8
	v_dual_fmac_f32 v35, v44, v35 :: v_dual_mul_f32 v42, v21, v33
	.loc	0 1502 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_div_scale_f32 v27, s6, v9, v0, v9
	v_fmac_f32_e32 v36, v45, v36
	.loc	0 1499 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_fma_f32 v48, -v16, v40, v17
	.loc	0 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_mul_f32_e32 v43, v23, v34
	.loc	0 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_div_scale_f32 v29, s7, v10, v0, v10
	v_fmac_f32_e32 v37, v46, v37
	.loc	0 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_fma_f32 v49, -v18, v41, v19
	.loc	0 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_mul_f32_e32 v44, v25, v35
	.loc	0 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_fma_f32 v50, -v20, v42, v21
	.loc	0 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_dual_mul_f32 v45, v27, v36 :: v_dual_fmac_f32 v40, v48, v31
	.loc	0 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_fma_f32 v51, -v22, v43, v23
	.loc	0 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_mul_f32_e32 v46, v29, v37
	.loc	0 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_fmac_f32_e32 v41, v49, v32
	.loc	0 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_fma_f32 v52, -v24, v44, v25
	.loc	0 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_fmac_f32_e32 v42, v50, v33
	.loc	0 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_fma_f32 v53, -v26, v45, v27
	.loc	0 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_fma_f32 v16, -v16, v40, v17
	.loc	0 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_dual_fmac_f32 v43, v51, v34 :: v_dual_lshlrev_b32 v2, 8, v1
	.loc	0 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_fma_f32 v54, -v28, v46, v29
	.loc	0 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_fma_f32 v17, -v18, v41, v19
	.loc	0 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_fmac_f32_e32 v44, v52, v35
	.loc	0 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_fma_f32 v18, -v20, v42, v21
	.loc	0 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_fmac_f32_e32 v45, v53, v36
	.loc	0 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v31, v40
	.loc	0 1499 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	s_mov_b32 vcc_lo, s2
	.loc	0 1500 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_fma_f32 v19, -v22, v43, v23
	.loc	0 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_fmac_f32_e32 v46, v54, v37
	.loc	0 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v32, v41
	.loc	0 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	s_mov_b32 vcc_lo, s3
	.loc	0 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_fma_f32 v20, -v24, v44, v25
	.loc	0 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v33, v42
	.loc	0 1500 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	s_mov_b32 vcc_lo, s4
	.loc	0 1502 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_fma_f32 v21, -v26, v45, v27
	.loc	0 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_div_fixup_f32 v4, v16, v0, v4
	.loc	0 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v16, v19, v34, v43
	.loc	0 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	s_mov_b32 vcc_lo, s5
	.loc	0 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_fma_f32 v22, -v28, v46, v29
	.loc	0 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_div_fixup_f32 v5, v17, v0, v5
	.loc	0 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v20, v35, v44
	.loc	0 1502 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	s_mov_b32 vcc_lo, s6
	.loc	0 1503 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_fma_f32 v23, -v30, v47, v39
	.loc	0 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_div_fixup_f32 v6, v18, v0, v6
	.loc	0 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v21, v36, v45
	.loc	0 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	s_mov_b32 vcc_lo, s7
	.loc	0 1504 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1504:31
	v_add_co_u32 v2, s2, s10, v2
	.loc	0 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v19, v22, v37, v46
	.loc	0 1503 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	s_mov_b32 vcc_lo, s8
	.loc	0 1500 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_div_fixup_f32 v7, v16, v0, v7
	.loc	0 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v20, v23, v38, v47
	.loc	0 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_div_fixup_f32 v8, v17, v0, v8
	.loc	0 1502 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_div_fixup_f32 v9, v18, v0, v9
	.loc	0 1503 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_div_fixup_f32 v10, v19, v0, v10
	.loc	0 1504 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1504:31
	v_add_co_ci_u32_e64 v16, null, s11, 0, s2
	.loc	0 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_div_fixup_f32 v11, v20, v0, v11
	.loc	0 1499 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:10
	v_cvt_pk_fp8_f32 v12.l, v4, v5
	.loc	0 1505 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1505:9
	v_add_co_u32 v4, vcc_lo, v2, v15
	.loc	0 1500 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:10
	v_cvt_pk_fp8_f32 v12.h, v6, v7
	.loc	0 1502 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:10
	v_cvt_pk_fp8_f32 v13.l, v8, v9
	.loc	0 1503 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:10
	v_cvt_pk_fp8_f32 v13.h, v10, v11
	.loc	0 1505 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1505:9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v16, vcc_lo
	.loc	0 1508 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1508:14
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	.loc	0 1507 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1507:30
	global_store_b64 v[4:5], v[12:13], off
	.loc	0 1508 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1508:14
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB2_3
; %bb.2:
.Ltmp123:
	.loc	4 248 59                        ; /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:248:59 @[ /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:299:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1471:26 ] ]
	s_load_b64 s[0:1], s[0:1], 0x10
	v_mov_b32_e32 v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
.Ltmp124:
	.loc	0 1509 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1509:46
	global_store_b32 v[1:2], v0, off
.LBB2_3:
	.loc	0 1510 1                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1510:1
	s_endpgm
.Ltmp125:
.Lfunc_end2:
	.size	attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201, .Lfunc_end2-attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
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
		.amdhsa_next_free_vgpr 56
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.num_vgpr, 56
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.numbered_sgpr, 12
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1484
; TotalNumSgprs: 14
; NumVgprs: 56
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 56
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
	.protected	attention_fp8_e4m3_fa2_gqa_partial_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_partial_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_partial_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_partial_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_partial_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_partial_gfx1201
.Lfunc_begin3:
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.loc	0 1569 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1569:17
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x28
	s_load_b64 s[16:17], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 1569 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1569:23
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s4, s17, -9
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s4, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB3_93
; %bb.1:
	.loc	0 0 23                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 1574 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1574:14
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB3_93
; %bb.2:
	.loc	0 1576 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1576:35
	s_lshl_b32 s4, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 1577 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1577:16
	s_cmp_ge_i32 s4, s7
	s_cbranch_scc1 .LBB3_93
; %bb.3:
	.loc	0 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	s_lshr_b32 s18, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 1580 15 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1580:15
	s_cmp_ge_i32 s18, s17
	s_cbranch_scc1 .LBB3_93
; %bb.4:
.Ltmp126:
	.loc	0 957 31                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:957:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_lshrrev_b32_e32 v156, 4, v0
	v_lshrrev_b32_e32 v157, 3, v0
	s_mov_b32 s19, 0
	.loc	0 974 9                         ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:974:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB3_6
; %bb.5:
	.loc	0 975 33                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshrrev_b32_e32 v1, 4, v0
	.loc	0 976 24                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:976:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_or_b32 v2, v0, 7, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 975 33                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_b32_e32 v1, 6, v1
	.loc	0 977 24                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:977:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_gt_i32_e32 vcc_lo, s7, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	.loc	0 975 25                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_u64_u32 v[3:4], null, s3, 6, v[1:2]
	.loc	0 975 45 is_stmt 0              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshrrev_b32_e32 v1, 3, v0
	.loc	0 978 31 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:978:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_lo_u32 v4, v2, 24
	s_and_b32 s19, vcc_lo, exec_lo
	.loc	0 975 39                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_or_b32 v5, v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 978 55                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:978:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_lshl_u32 v3, v5, v4, 8
	.loc	0 1150 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1150:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b32_e32 v4, 0
.Ltmp127:
.LBB3_6:
	.loc	0 0 38 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:38
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1569 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1569:17
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x20
.Ltmp128:
	.loc	0 954 26                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:954:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_b32_e32 v9, 31, v0
	v_mov_b32_e32 v7, -1
	v_bfrev_b32_e32 v8, -2
	.loc	0 991 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:991:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_u32_e32 8, v9
	s_cbranch_execz .LBB3_10
; %bb.7:
	.loc	0 992 31                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:992:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v6, s4, v9
	v_bfrev_b32_e32 v8, -2
	v_mov_b32_e32 v7, -1
	.loc	0 993 16                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:993:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s7, v6
	s_cbranch_execz .LBB3_9
; %bb.8:
	.loc	0 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_ashrrev_i32_e32 v7, 31, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_wait_kmcnt 0x0
	v_add_co_u32 v6, vcc_lo, s0, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	.loc	0 994 27 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:994:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_b32 v7, v[6:7], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v8, v7
.LBB3_9:
	.loc	0 0 27 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:27
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB3_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
.Ltmp129:
	v_mbcnt_lo_u32_b32 v10, -1, 0
.Ltmp130:
	.loc	0 956 25 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:956:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshrrev_b32_e32 v159, 4, v9
	v_lshrrev_b32_e32 v13, 1, v0
	v_dual_mov_b32 v167, 1.0 :: v_dual_lshlrev_b32 v14, 3, v0
	s_delay_alu instid0(VALU_DEP_4)
.Ltmp131:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v1, 16, v10
.Ltmp132:
	.loc	1 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v12, 8, v10
	v_lshl_add_u32 v161, v9, 3, 0
	v_cndmask_b32_e64 v9, 0, v4, s19
.Ltmp133:
	.loc	0 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshrrev_b32_e32 v162, 5, v0
.Ltmp134:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
.Ltmp135:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v15, 1, v10
.Ltmp136:
	.loc	0 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_cvt_f32_u32 s4, s17
	s_add_co_i32 s6, s17, 0x1ff
	s_mov_b32 s5, 0
.Ltmp137:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, v10, v1 :: v_dual_and_b32 v160, 15, v0
.Ltmp138:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v12
.Ltmp139:
	.loc	0 1582 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s20, s4
	s_and_b32 s6, s6, 0xffff
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v158, 2, v1
	v_and_b32_e32 v165, 8, v13
	v_and_b32_e32 v13, 0xf8, v14
	v_lshlrev_b32_e32 v163, 3, v159
.Ltmp140:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v6, v158, v7
.Ltmp141:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1004:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v11, v158, v8
.Ltmp142:
	.loc	0 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b32_e32 v166, 4, v162
	v_lshlrev_b32_e32 v133, 4, v160
.Ltmp143:
	.loc	0 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s6, s6
	v_dual_mov_b32 v154, 0xff800000 :: v_dual_mov_b32 v155, 0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s20, s6, s20
	s_trunc_f32 s20, s20
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[145:146], null, v2, 24, v[5:6]
.Ltmp144:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v10, v12, vcc_lo
.Ltmp145:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max_i32_e32 v4, v7, v6
.Ltmp146:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v7, 4, v10
	v_cndmask_b32_e64 v12, 0, v3, s19
	v_ashrrev_i32_e32 v3, 31, v2
.Ltmp147:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_lshlrev_b32_e32 v0, 2, v5
.Ltmp148:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1004:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v8, v11
.Ltmp149:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	v_lshlrev_b32_e32 v164, 7, v160
.Ltmp150:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v11, 2, v10
.Ltmp151:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v6, v0, v4
.Ltmp152:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1005:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v0, v0, v5
.Ltmp153:
	.loc	1 524 11 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v10, v7, vcc_lo
	v_lshlrev_b64_e32 v[129:130], 2, v[2:3]
	v_mov_b32_e32 v8, v1
.Ltmp154:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, v1 :: v_dual_lshlrev_b32 v7, 2, v7
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v146, v1
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, v10, v11, vcc_lo
.Ltmp155:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v15
	v_lshlrev_b64_e32 v[131:132], 2, v[145:146]
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v134, v10, v15, vcc_lo
.Ltmp156:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v41, v4, v6
.Ltmp157:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1005:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v5, v0
	v_dual_mov_b32 v5, v1 :: v_dual_mov_b32 v4, v1
	v_mov_b32_e32 v6, v1
.Ltmp158:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v42, v7, v41
.Ltmp159:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1006:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v43, v7, v0
	v_mov_b32_e32 v7, v1
.Ltmp160:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_lshlrev_b32_e32 v65, 2, v44
.Ltmp161:
	.loc	0 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v136, vcc_lo, v12, v163
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v137, null, 0, v9, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v146, vcc_lo, s0, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v147, null, s1, v130, vcc_lo
.Ltmp162:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_lshlrev_b32_e32 v129, 2, v134
.Ltmp163:
	.loc	0 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_xor_b32 s0, s20, 0x80000000
.Ltmp164:
	.loc	0 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshl_or_b32 v135, v162, 8, v13
.Ltmp165:
	.loc	0 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s6, s0, s4
	s_cvt_u32_f32 s0, s20
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v7
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s6, 31
.Ltmp166:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v138, v41, v42
.Ltmp167:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1006:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v43
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v41, v1
	v_mov_b32_e32 v56, v8
.Ltmp168:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v139, v65, v138
.Ltmp169:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1007:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v140, v65, v0
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v64, v8
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v72, v8
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v80, v8
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v88, v8
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v96, v8
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v104, v8
	v_dual_mov_b32 v97, v1 :: v_dual_mov_b32 v112, v8
.Ltmp170:
	.loc	0 1582 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s6, s4
	v_dual_mov_b32 v105, v1 :: v_dual_mov_b32 v120, v8
	v_dual_mov_b32 v113, v1 :: v_dual_mov_b32 v128, v8
.Ltmp171:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v130, v138, v139
.Ltmp172:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1007:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v140
	v_dual_mov_b32 v121, v1 :: v_dual_add_nc_u32 v168, 0, v135
.Ltmp173:
	.loc	0 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v135, vcc_lo, s8, v136
.Ltmp174:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v134, v129, v130
.Ltmp175:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1008:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v129, v129, v0
.Ltmp176:
	.loc	0 1582 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_add_co_ci_u32 s6, s0, 0
	s_lshl_b32 s4, s3, 8
.Ltmp177:
	.loc	0 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v136, null, s9, v137, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[4:5]
	s_add_nc_u64 s[20:21], s[12:13], s[4:5]
	s_mul_i32 s4, s7, 0x1800
	v_add_co_u32 v148, vcc_lo, v135, 48
.Ltmp178:
	.loc	0 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_and_b32 s22, s6, 0xffff
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[8:9], s[4:5]
.Ltmp179:
	.loc	0 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v149, null, 0, v136, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v152, vcc_lo, s6, v131
	v_add_co_u32 v150, s0, s0, v133
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
.Ltmp180:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v130, v130, v134
.Ltmp181:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1008:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v129
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, s7, v132, vcc_lo
.Ltmp182:
	.loc	0 1009 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1009:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_readfirstlane_b32 s26, v130
	v_mov_b32_e32 v136, v8
	v_dual_mov_b32 v14, v6 :: v_dual_mov_b32 v13, v5
	v_dual_mov_b32 v12, v4 :: v_dual_mov_b32 v11, v3
	v_dual_mov_b32 v10, v2 :: v_dual_mov_b32 v9, v1
	v_dual_mov_b32 v22, v6 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v20, v4 :: v_dual_mov_b32 v19, v3
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v30, v6 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v28, v4 :: v_dual_mov_b32 v27, v3
	v_dual_mov_b32 v26, v2 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v36, v4 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v33, v1
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v71, v7 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v79, v7 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v77, v5 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v87, v7 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v85, v5 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v95, v7 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v93, v5 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v103, v7 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v101, v5 :: v_dual_mov_b32 v100, v4
	v_dual_mov_b32 v99, v3 :: v_dual_mov_b32 v98, v2
	v_dual_mov_b32 v111, v7 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v109, v5 :: v_dual_mov_b32 v108, v4
	v_dual_mov_b32 v107, v3 :: v_dual_mov_b32 v106, v2
	v_dual_mov_b32 v119, v7 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v117, v5 :: v_dual_mov_b32 v116, v4
	v_dual_mov_b32 v115, v3 :: v_dual_mov_b32 v114, v2
	v_dual_mov_b32 v127, v7 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v125, v5 :: v_dual_mov_b32 v124, v4
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v122, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v151, null, s1, 0, s0
	.loc	0 1010 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1010:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_readfirstlane_b32 s27, v0
	v_dual_mov_b32 v135, v7 :: v_dual_mov_b32 v134, v6
	v_dual_mov_b32 v133, v5 :: v_dual_mov_b32 v132, v4
	v_dual_mov_b32 v131, v3 :: v_dual_mov_b32 v130, v2
	v_mov_b32_e32 v129, v1
	.loc	0 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mul_i32 s24, s18, s22
	s_lshl_b32 s4, s3, 1
.Ltmp183:
	.loc	0 1584 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1584:23
	s_add_co_i32 s25, s24, s22
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[10:11], s[4:5]
	s_add_nc_u64 s[12:13], s[12:13], s[4:5]
.Ltmp184:
	.loc	0 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_lshl_b32 s11, s24, 6
	s_branch .LBB3_14
.LBB3_11:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	0 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_mov_b32_e32 v154, v2
.LBB3_12:                               ;   in Loop: Header=BB3_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s30
.Ltmp185:
	.loc	3 701 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp186:
.LBB3_13:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	3 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
	s_add_co_i32 s24, s24, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s24, s25
	s_cselect_b32 s0, -1, 0
	s_xor_b32 s1, s28, -1
	s_add_co_i32 s11, s11, 64
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_88
.LBB3_14:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_17 Depth 2
                                        ;     Child Loop BB3_23 Depth 2
                                        ;     Child Loop BB3_45 Depth 2
                                        ;       Child Loop BB3_50 Depth 3
	.loc	0 1015 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1015:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_lshl_b32 s29, s24, 6
	.loc	0 1016 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1016:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s29, s26
	s_cselect_b32 s28, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_13
; %bb.15:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_mov_b32_e32 v0, v156
	v_mov_b32_e32 v6, v157
	s_mov_b32 s0, 8
	s_branch .LBB3_17
.LBB3_16:                               ;   in Loop: Header=BB3_17 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 1032 49 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_b32_e32 v7, 0xe0, v6
	.loc	0 1024 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v6, 16, v6
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	0 1032 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v7, v7, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1032 54 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:54 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_or_b32 v7, v0, 15, v7
	.loc	0 1024 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 8, v0
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshl_add_u32 v7, v7, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v7, v[2:3], v[4:5] offset1:16
	.loc	0 1024 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cbranch_scc1 .LBB3_19
.LBB3_17:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	0 0 9                           ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v7, s11, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v5, 0
	.loc	0 1033 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1033:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s1, exec_lo
	v_cmpx_ge_i32_e64 s26, v7
	s_cbranch_execz .LBB3_16
; %bb.18:                               ;   in Loop: Header=BB3_17 Depth=2
	.loc	0 1037 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1037:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v7, v[150:151]
	.loc	0 1038 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1038:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_b128 v[2:5], v[2:3], off
	s_branch .LBB3_16
.LBB3_19:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	0 1049 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v6, s29, v165
	v_dual_mov_b32 v7, v166 :: v_dual_mov_b32 v8, v162
	s_movk_i32 s0, 0xc000
	s_branch .LBB3_23
.LBB3_20:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 0 9 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB3_21:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB3_22:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 1088 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1088:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, s0, v168
	.loc	0 1049 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v8, 4, v8
	v_add_nc_u32_e32 v7, 64, v7
	s_addk_co_i32 s0, 0x400
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	0 1088 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1088:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_store_b64 v0, v[2:3] offset:32768
	.loc	0 1049 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cbranch_scc1 .LBB3_39
.LBB3_23:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	0 0 9 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1058 43 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1058:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_b32_e32 v0, 0x70, v8
	v_add_nc_u32_e32 v137, v0, v6
	.loc	0 1057 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1057:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_or_b32 v0, 0xf0, v7, v160
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1062 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1062:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v2, 7, v137
	.loc	0 1062 27 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1062:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_ge_i32_e32 vcc_lo, s26, v2
                                        ; implicit-def: $vgpr2_vgpr3
	s_and_saveexec_b32 s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB3_25
; %bb.24:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 1065 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1065:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v137, s[20:21]
	.loc	0 1067 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1067:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e32 v5.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 1073 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1073:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v139.l, v5.h
	.loc	0 1066 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1066:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v2, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	.loc	0 1068 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x7
	global_load_u8 v0, v[2:3], off offset:1032
	global_load_u8 v4, v[2:3], off offset:2064
	global_load_d16_u8 v5, v[2:3], off
	global_load_u8 v137, v[2:3], off offset:3096
	global_load_u8 v138, v[2:3], off offset:5160
	global_load_u8 v140, v[2:3], off offset:4128
	global_load_u8 v141, v[2:3], off offset:7224
	global_load_d16_hi_u8 v139, v[2:3], off offset:6192
	.loc	0 1068 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v0, 8, v0
	.loc	0 1069 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1069:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	.loc	0 1070 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1070:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v137
	.loc	0 1072 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v138
	.loc	0 1068 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, v0, v5
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	.loc	0 1070 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1070:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or3_b32 v0, v0, v2, v3
	.loc	0 1072 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v140, v4
	.loc	0 1074 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1074:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v141
	.loc	0 1072 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or3_b32 v0, v0, 0, 0
	.loc	0 1074 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1074:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v3, v2, v139, v3
	v_or3_b32 v2, v0, 0, 0
                                        ; implicit-def: $vgpr0
.LBB3_25:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB3_22
; %bb.26:                               ;   in Loop: Header=BB3_23 Depth=2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	.loc	0 1075 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1075:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s3, exec_lo
	v_cmpx_ge_i32_e64 s26, v137
	s_cbranch_execz .LBB3_21
; %bb.27:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 1077 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1077:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v4, s4, s20, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s21, 0, s4
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e32 v0.h, 0
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v137, v[4:5]
	global_load_d16_u8 v0, v[2:3], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmpx_gt_i32_e64 s26, v137
	s_cbranch_execz .LBB3_29
; %bb.28:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 1078 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v2, 1, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, v[4:5]
	global_load_u8 v2, v[2:3], off
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v0, v2, 8, v0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
.LBB3_29:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, 2, v137
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_31
; %bb.30:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 16, v2
.LBB3_31:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, 3, v137
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_33
; %bb.32:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	.loc	0 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_u8 v0, v[138:139], off
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 24, v2
.LBB3_33:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, 4, v137
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_35
; %bb.34:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
.LBB3_35:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, 5, v137
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_37
; %bb.36:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	0 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 8, v0
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v3, v0, v3
.LBB3_37:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, 6, v137
	.loc	0 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_20
; %bb.38:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[4:5], null, 0x408, v0, v[4:5]
	.loc	0 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e32 v0.l, 0
	.loc	0 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_d16_hi_u8 v0, v[4:5], off
	.loc	0 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
	s_branch .LBB3_20
.LBB3_39:                               ;   in Loop: Header=BB3_14 Depth=1
.Ltmp187:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp188:
	.loc	0 1096 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1096:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB3_12
; %bb.40:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b32_e32 v0, s16
	.loc	0 1111 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1111:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB3_42
; %bb.41:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	0 1112 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1112:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_b32 v0, v[152:153], off
	.loc	0 1113 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1113:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v0, s16, v0
.LBB3_42:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	0 0 31 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:31
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	0 1098 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1098:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_or_b32 s0, s29, 16
	v_mov_b32_e32 v4, v161
	.loc	0 1098 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1098:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s26
	s_mov_b32 s31, 0
	s_cselect_b32 s33, -1, 0
	s_branch .LBB3_45
.LBB3_43:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_sub_f32_e32 v143, v154, v2
	.loc	0 1189 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1189:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v154
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v172, null, v3, v3, v140
	.loc	0 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v174, null, v3, v3, v139
	.loc	0 1245 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v176, null, v3, v3, v8
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v177, null, v3, v3, v137
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v173, v172
	.loc	0 1251 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1251:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshl_add_u32 v189, s31, 12, v161
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v179, v176
	v_exp_f32_e32 v143, v143
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v180, v177
	.loc	0 1209 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1209:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v175, -v172, v173, 1.0
	.loc	0 1189 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1189:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v143, 0, v143, vcc_lo
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v173, v175, v173
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1226 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1226:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v144, v167, v143
	.loc	0 1226 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1226:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v154, null, v3, v3, v144
	v_div_scale_f32 v170, vcc_lo, v144, v3, v144
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v167, v154
	v_fma_f32 v169, -v154, v167, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v167, v169, v167
	v_mul_f32_e32 v169, v170, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v171, -v154, v169, v170
	v_fmac_f32_e32 v169, v171, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v154, v169, v170
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v154, v154, v167, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v6, v154, v3, v144
	.loc	0 1240 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v154, null, v3, v3, v141
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v134, v134, v6 :: v_dual_fmac_f32 v5, v155, v143
	v_mul_f32_e32 v126, v126, v6
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v143, null, v3, v3, v142
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v167, v154
	.loc	0 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v136, v136, v6 :: v_dual_mul_f32 v135, v135, v6
	v_mul_f32_e32 v124, v124, v6
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v144, v143
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v133, v133, v6 :: v_dual_mul_f32 v122, v122, v6
	v_dual_mul_f32 v132, v132, v6 :: v_dual_mul_f32 v131, v131, v6
	v_mul_f32_e32 v120, v120, v6
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v170, -v154, v167, 1.0
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v130, v130, v6 :: v_dual_mul_f32 v129, v129, v6
	v_mul_f32_e32 v118, v118, v6
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v143, v144, 1.0
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v167, v170, v167
	v_div_scale_f32 v170, s0, v141, v3, v141
	.loc	0 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v128, v128, v6 :: v_dual_mul_f32 v127, v127, v6
	v_mul_f32_e32 v116, v116, v6
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v144, v155, v144
	v_div_scale_f32 v155, vcc_lo, v142, v3, v142
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v125, v125, v6 :: v_dual_mul_f32 v114, v114, v6
	v_dual_mul_f32 v123, v123, v6 :: v_dual_mul_f32 v112, v112, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v169, v155, v144
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v121, v121, v6 :: v_dual_mul_f32 v110, v110, v6
	v_dual_mul_f32 v119, v119, v6 :: v_dual_mul_f32 v108, v108, v6
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v171, -v143, v169, v155
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v12, v12, v6 :: v_dual_mul_f32 v117, v117, v6
	v_dual_mul_f32 v106, v106, v6 :: v_dual_mul_f32 v115, v115, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v104, v104, v6 :: v_dual_fmac_f32 v169, v171, v144
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v171, v170, v167
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v113, v113, v6 :: v_dual_mul_f32 v102, v102, v6
	v_mul_f32_e32 v10, v10, v6
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v143, -v143, v169, v155
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v154, v171, v170
	.loc	0 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v111, v111, v6 :: v_dual_mul_f32 v100, v100, v6
	v_dual_mul_f32 v109, v109, v6 :: v_dual_mul_f32 v98, v98, v6
	.loc	0 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v169
	.loc	0 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v171, v155, v167
	.loc	0 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v155, s1, v140, v3, v140
	.loc	0 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v144, v174
	.loc	0 1240 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	0 1240 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v143, v143, v3, v142
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v142, -v154, v171, v170
	.loc	0 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v154, v155, v173
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v170, null, v3, v3, v7
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v107, v107, v6 :: v_dual_mul_f32 v96, v96, v6
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v167, v142, v167, v171
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v171, -v172, v154, v155
	.loc	0 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v174, v144, 1.0
	.loc	0 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v21, v21, v6
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v175, v170
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v23, v23, v6
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v154, v171, v173
	.loc	0 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, s0, v139, v3, v139
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s1
	.loc	0 1241 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v142.h, 0
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v172, v154, v155
	.loc	0 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v142.l, v1.l
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v171, v169, v144
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v178, -v170, v175, 1.0
	.loc	0 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v167, v167, v3, v141
	.loc	0 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v154, v155, v173, v154
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v155, null, v3, v3, v138
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v172, -v174, v171, v169
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v175, v178, v175
	.loc	0 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v173, -v176, v179, 1.0
	.loc	0 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v140, v154, v3, v140
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v178, v155
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v171, v172, v144
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v172, s1, v7, v3, v7
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v154, -v177, v180, 1.0
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v179, v173, v179
	v_div_scale_f32 v173, s3, v8, v3, v8
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v174, v171, v169
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v19, v19, v6 :: v_dual_mul_f32 v174, v172, v175
	s_delay_alu instid0(TRANS32_DEP_1)
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v181, -v155, v178, 1.0
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v17, v17, v6 :: v_dual_fmac_f32 v180, v154, v180
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v154, s4, v137, v3, v137
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v178, v181, v178
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v169, v144, v171
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v171, v173, v179
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v181, s0, v138, v3, v138
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v170, v174, v172
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v15, v15, v6 :: v_dual_mul_f32 v182, v154, v180
	.loc	0 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v139, v144, v3, v139
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v144, -v176, v171, v173
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v183, v181, v178
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v174, v169, v175
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v177, v182, v154
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s1
	.loc	0 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v171, v144, v179
	.loc	0 1247 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v144, -v155, v183, v181
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v170, -v170, v174, v172
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v182, v169, v180
	.loc	0 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v141.h, v142.h
	.loc	0 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v141.l, v142.l
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v183, v144, v178
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v169, v170, v175, v174
	.loc	0 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v170, -v176, v171, v173
	.loc	0 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v144, -v177, v182, v154
	.loc	0 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v155, v183, v181
	.loc	0 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v7, v169, v3, v7
	.loc	0 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v154, v170, v179, v171
	.loc	0 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	0 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v141.h, v140, v139
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v180, v182
	.loc	0 1247 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	0 1245 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v8, v154, v3, v8
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v155, v155, v178, v183
	.loc	0 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v141.l, v143, v167
	.loc	0 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v137, v144, v3, v137
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v105, v105, v6 :: v_dual_mul_f32 v94, v94, v6
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v138, v155, v3, v138
	.loc	0 1244 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1244:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v142.l, v7, v8
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[7:8], v189 offset:16384
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v103, v103, v6 :: v_dual_mul_f32 v92, v92, v6
	.loc	0 1246 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1246:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v142.h, v137, v138
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[137:138], v189 offset:16640
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[139:140], v189 offset:16896
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[143:144], v189 offset:17152
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[154:155], v189 offset:17408
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[169:170], v189 offset:17664
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[171:172], v189 offset:17920
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[173:174], v189 offset:18176
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[175:176], v189 offset:18432
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[177:178], v189 offset:18688
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[179:180], v189 offset:18944
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[181:182], v189 offset:19200
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[183:184], v189 offset:19456
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[185:186], v189 offset:19712
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[187:188], v189 offset:19968
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[189:190], v189 offset:20224
	.loc	0 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v101, v101, v6 :: v_dual_mul_f32 v90, v90, v6
	v_dual_mul_f32 v99, v99, v6 :: v_dual_mul_f32 v88, v88, v6
	v_dual_mul_f32 v97, v97, v6 :: v_dual_mul_f32 v86, v86, v6
	v_dual_mul_f32 v95, v95, v6 :: v_dual_mul_f32 v84, v84, v6
	v_dual_mul_f32 v93, v93, v6 :: v_dual_mul_f32 v82, v82, v6
	v_dual_mul_f32 v91, v91, v6 :: v_dual_mul_f32 v80, v80, v6
	v_dual_mul_f32 v89, v89, v6 :: v_dual_mul_f32 v78, v78, v6
	v_dual_mul_f32 v87, v87, v6 :: v_dual_mul_f32 v76, v76, v6
	v_dual_mul_f32 v85, v85, v6 :: v_dual_mul_f32 v74, v74, v6
	v_dual_mul_f32 v83, v83, v6 :: v_dual_mul_f32 v72, v72, v6
	v_dual_mul_f32 v81, v81, v6 :: v_dual_mul_f32 v70, v70, v6
	v_dual_mul_f32 v79, v79, v6 :: v_dual_mul_f32 v68, v68, v6
	v_dual_mul_f32 v77, v77, v6 :: v_dual_mul_f32 v66, v66, v6
	v_dual_mul_f32 v75, v75, v6 :: v_dual_mul_f32 v64, v64, v6
	v_dual_mul_f32 v73, v73, v6 :: v_dual_mul_f32 v62, v62, v6
	v_dual_mul_f32 v71, v71, v6 :: v_dual_mul_f32 v60, v60, v6
	v_dual_mul_f32 v69, v69, v6 :: v_dual_mul_f32 v58, v58, v6
	v_dual_mul_f32 v67, v67, v6 :: v_dual_mul_f32 v56, v56, v6
	v_dual_mul_f32 v65, v65, v6 :: v_dual_mul_f32 v54, v54, v6
	v_dual_mul_f32 v63, v63, v6 :: v_dual_mul_f32 v52, v52, v6
	v_dual_mul_f32 v61, v61, v6 :: v_dual_mul_f32 v50, v50, v6
	v_dual_mul_f32 v59, v59, v6 :: v_dual_mul_f32 v48, v48, v6
	v_dual_mul_f32 v57, v57, v6 :: v_dual_mul_f32 v46, v46, v6
	v_dual_mul_f32 v55, v55, v6 :: v_dual_mul_f32 v44, v44, v6
	v_dual_mul_f32 v53, v53, v6 :: v_dual_mul_f32 v42, v42, v6
	v_dual_mul_f32 v51, v51, v6 :: v_dual_mul_f32 v40, v40, v6
	v_dual_mul_f32 v49, v49, v6 :: v_dual_mul_f32 v38, v38, v6
	v_dual_mul_f32 v47, v47, v6 :: v_dual_mul_f32 v36, v36, v6
	v_dual_mul_f32 v45, v45, v6 :: v_dual_mul_f32 v34, v34, v6
	v_dual_mul_f32 v43, v43, v6 :: v_dual_mul_f32 v32, v32, v6
	v_dual_mul_f32 v41, v41, v6 :: v_dual_mul_f32 v30, v30, v6
	v_dual_mul_f32 v39, v39, v6 :: v_dual_mul_f32 v28, v28, v6
	v_dual_mul_f32 v37, v37, v6 :: v_dual_mul_f32 v26, v26, v6
	v_dual_mul_f32 v35, v35, v6 :: v_dual_mul_f32 v24, v24, v6
	v_dual_mul_f32 v33, v33, v6 :: v_dual_mul_f32 v22, v22, v6
	v_dual_mul_f32 v31, v31, v6 :: v_dual_mul_f32 v20, v20, v6
	v_dual_mul_f32 v29, v29, v6 :: v_dual_mul_f32 v18, v18, v6
	v_dual_mul_f32 v27, v27, v6 :: v_dual_mul_f32 v16, v16, v6
	v_dual_mul_f32 v25, v25, v6 :: v_dual_mul_f32 v14, v14, v6
	v_mul_f32_e32 v13, v13, v6
	v_mul_f32_e32 v11, v11, v6
	v_mul_f32_e32 v9, v9, v6
	.loc	0 1253 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1253:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[7:8], v[141:142], v[129:136]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[137:138], v[141:142], v[121:128]
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[139:140], v[141:142], v[113:120]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[143:144], v[141:142], v[105:112]
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[154:155], v[141:142], v[97:104]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[169:170], v[141:142], v[89:96]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[171:172], v[141:142], v[81:88]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[173:174], v[141:142], v[73:80]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[175:176], v[141:142], v[65:72]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[177:178], v[141:142], v[57:64]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[179:180], v[141:142], v[49:56]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[181:182], v[141:142], v[41:48]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[183:184], v[141:142], v[33:40]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[185:186], v[141:142], v[25:32]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[187:188], v[141:142], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[189:190], v[141:142], v[9:16]
	v_mov_b32_e32 v155, v5
	v_mov_b32_e32 v167, v3
	.loc	0 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
.LBB3_44:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 1116 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v4, 0x100, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v154, v2
	.loc	0 1116 40 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_add_co_i32 s31, s31, 1
	.loc	0 1116 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s31, 4
	.loc	0 1116 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cbranch_scc0 .LBB3_11
.LBB3_45:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB3_50 Depth 3
	.loc	0 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s31, 1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB3_48
; %bb.46:                               ;   in Loop: Header=BB3_45 Depth=2
	s_cmp_eq_u32 s31, 1
	s_mov_b32 s0, s33
	s_cbranch_scc1 .LBB3_48
; %bb.47:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 1118 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1118:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cmp_eq_u32 s31, 2
	.loc	0 1118 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1118:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cselect_b32 s0, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, s29
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s26, s0
	s_cselect_b32 s0, -1, 0
.LBB3_48:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_85
; %bb.49:                               ;   in Loop: Header=BB3_45 Depth=2
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v2, v148
	v_mov_b32_e32 v3, v149
	s_movk_i32 s0, 0xc000
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_mov_b32_e32 v144, v137
.LBB3_50:                               ;   Parent Loop BB3_14 Depth=1
                                        ;     Parent Loop BB3_45 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	.loc	0 1151 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1151:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x3
	global_load_b64 v[173:174], v[2:3], off offset:-48
	global_load_b64 v[175:176], v[2:3], off offset:-32
	global_load_b64 v[177:178], v[2:3], off offset:-16
	global_load_b64 v[179:180], v[2:3], off
	.loc	0 1154 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1154:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v169, s0, v4
	.loc	0 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v2, vcc_lo, v2, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	.loc	0 1154 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1154:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_2addr_stride64_b64 v[5:8], v169 offset0:32 offset1:34
	ds_load_2addr_stride64_b64 v[169:172], v169 offset0:36 offset1:38
	.loc	0 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_addk_co_i32 s0, 0x1000
	.loc	0 1159 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1159:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	.loc	0 1155 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1155:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[5:6], v[173:174], v[137:144]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[7:8], v[175:176], v[137:144]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[169:170], v[177:178], v[137:144]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[171:172], v[179:180], v[137:144]
	.loc	0 1139 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cbranch_scc1 .LBB3_50
; %bb.51:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 1161 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1161:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_lshl4_add_u32 s0, s31, s29
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v170, 0
	.loc	0 1162 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1162:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 15
	.loc	0 1162 49 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1162:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s27
	s_cselect_b32 s34, -1, 0
	s_cmp_gt_i32 s1, s27
	s_cselect_b32 s1, -1, 0
	.loc	0 1164 28 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1164:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s1, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s3
	s_cbranch_execz .LBB3_53
; %bb.52:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 1164 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1164:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_b32 v170, v[146:147], off
.LBB3_53:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, s0, v163
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp189:
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e32 vcc_lo, s26, v2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB3_55
; %bb.54:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[5:6], null, 0x408, v2, s[22:23]
	global_load_d16_b16 v3, v[5:6], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v173, v3.l
.Ltmp190:
.LBB3_55:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v5, 1, v2
.Ltmp191:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_i32_e64 s0, s26, v2
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB3_57
; %bb.56:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[6:7], null, 0x408, v5, s[22:23]
	global_load_d16_b16 v3, v[6:7], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v175, v3.l
.Ltmp192:
.LBB3_57:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v6, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp193:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s1, s26, v6
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB3_59
; %bb.58:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[7:8], null, 0x408, v6, s[22:23]
	global_load_d16_b16 v3, v[7:8], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v3.l
.Ltmp194:
.LBB3_59:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v7, 3, v2
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v178, 0
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp195:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s3, s26, v7
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_61
; %bb.60:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[171:172], null, 0x408, v7, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v178, v3.l
.Ltmp196:
.LBB3_61:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v8, 4, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp197:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s4, s26, v8
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB3_63
; %bb.62:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[171:172], null, 0x408, v8, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v176, v3.l
.Ltmp198:
.LBB3_63:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v169, 5, v2
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp199:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s5, s26, v169
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_65
; %bb.64:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[171:172], null, 0x408, v169, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v179, v3.l
.Ltmp200:
.LBB3_65:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v171, 6, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp201:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s6, s26, v171
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB3_67
; %bb.66:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[180:181], null, 0x408, v171, s[22:23]
	global_load_d16_b16 v3, v[180:181], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v177, v3.l
.Ltmp202:
.LBB3_67:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	.loc	0 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v172, 7, v2
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v180, 0
	s_delay_alu instid0(VALU_DEP_2)
.Ltmp203:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s7, s26, v172
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_69
; %bb.68:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[180:181], null, 0x408, v172, s[22:23]
	global_load_d16_b16 v180, v[180:181], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v180, v180.l
.Ltmp204:
.LBB3_69:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mul_f32 v137, v0, v137 :: v_dual_mul_f32 v138, v0, v138
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s8, v2, v170
	v_cmp_lt_i32_e64 s9, v2, v170
	v_dual_mul_f32 v139, v0, v139 :: v_dual_mul_f32 v140, v0, v140
	v_cmp_le_i32_e64 s10, v6, v170
	v_dual_mul_f32 v137, v137, v173 :: v_dual_mul_f32 v138, v138, v175
	s_or_b32 s8, s34, s8
	s_or_b32 s9, s34, s9
	.loc	0 1178 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	v_dual_mul_f32 v173, v139, v174 :: v_dual_mul_f32 v144, v0, v144
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v137, 0xff800000, v137, s8
	s_or_b32 s10, s34, s10
	v_cmp_le_i32_e64 s8, v7, v170
	s_and_b32 s9, s19, s9
	v_mul_f32_e32 v142, v0, v142
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v139, 0xff800000, v138, s9
	s_and_b32 s9, s19, s10
	v_mul_f32_e32 v140, v140, v178
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, 0xff800000, v173, s9
	v_mul_f32_e32 v173, v0, v141
	s_or_b32 s10, s34, s8
	v_cmp_le_i32_e64 s8, v8, v170
	v_cmp_le_i32_e64 s9, v169, v170
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s10, s19, s10
	v_mul_f32_e32 v143, v0, v143
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v141, 0xff800000, v140, s10
	v_mul_f32_e32 v140, v173, v176
	s_or_b32 s8, s34, s8
	v_mul_f32_e32 v173, v142, v179
	s_or_b32 s9, s34, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	v_dual_mul_f32 v143, v143, v177 :: v_dual_mul_f32 v144, v144, v180
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v142, 0xff800000, v140, s8
	s_and_b32 s8, s19, s9
	v_cmp_le_i32_e64 s9, v172, v170
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v140, 0xff800000, v173, s8
	v_cmp_le_i32_e64 s8, v171, v170
.Ltmp205:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v170, v137, 0xff800000, v139
	s_or_b32 s9, s34, s9
	s_or_b32 s8, s34, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v170, v170, v138, v141
.Ltmp206:
	.loc	0 1178 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v173, 0xff800000, v143, s8
	s_and_b32 s8, s19, s9
.Ltmp207:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v143, v170, v142, v140
.Ltmp208:
	.loc	0 1178 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v144, 0xff800000, v144, s8
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp209:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v175, v143, v173, v144
.Ltmp210:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1186:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ]
	ds_bpermute_b32 v176, v158, v175
.Ltmp211:
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB3_71
; %bb.70:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v3, v2.l
.LBB3_71:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v170, 0
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB3_73
; %bb.72:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[177:178], null, 0x408, v5, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v170, v2.l
.LBB3_73:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_75
; %bb.74:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[5:6], null, 0x408, v6, s[12:13]
	global_load_d16_b16 v2, v[5:6], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v143, v2.l
.LBB3_75:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB3_77
; %bb.76:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[6:7], null, 0x408, v7, s[12:13]
	global_load_d16_b16 v2, v[6:7], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v6, v2.l
.LBB3_77:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB3_79
; %bb.78:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[7:8], null, 0x408, v8, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v5, v2.l
.LBB3_79:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v8, 0
	.loc	0 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB3_86
; %bb.80:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 14 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s6
	s_cbranch_execnz .LBB3_87
.LBB3_81:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 14                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB3_83
.LBB3_82:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[171:172], null, 0x408, v172, s[12:13]
	global_load_d16_b16 v2, v[171:172], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v169, v2.l
.Ltmp212:
.LBB3_83:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
.Ltmp213:
	.loc	2 454 44 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1187:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_max3_num_f32 v2, v154, v175, v176
.Ltmp214:
	.loc	0 1222 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1222:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v7, v137, v2 :: v_dual_sub_f32 v138, v138, v2
	v_dual_sub_f32 v137, v139, v2 :: v_dual_sub_f32 v140, v140, v2
	v_dual_sub_f32 v139, v141, v2 :: v_dual_sub_f32 v144, v144, v2
	v_dual_mul_f32 v7, 0x3fb8aa3b, v7 :: v_dual_mul_f32 v138, 0x3fb8aa3b, v138
	v_dual_sub_f32 v141, v142, v2 :: v_dual_sub_f32 v142, v173, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_mul_f32 v140, 0x3fb8aa3b, v140
	v_exp_f32_e32 v7, v7
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v2
	v_exp_f32_e32 v138, v138
	v_dual_mul_f32 v139, 0x3fb8aa3b, v139 :: v_dual_mul_f32 v144, 0x3fb8aa3b, v144
	v_exp_f32_e32 v137, v137
	v_exp_f32_e32 v140, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v139, v139
	.loc	0 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, v7, 0, vcc_lo
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_mul_f32_e32 v171, 0x3fb8aa3b, v142
	v_exp_f32_e32 v144, v144
	.loc	0 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v142, v7, v3
	v_exp_f32_e32 v172, v141
	.loc	0 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cndmask_b32_e64 v137, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v7, v137
	.loc	0 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v141, v137, v170
	.loc	0 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cndmask_b32_e64 v170, v139, 0, vcc_lo
	v_exp_f32_e32 v137, v171
	v_cndmask_b32_e64 v171, v172, 0, vcc_lo
	v_cndmask_b32_e64 v172, v140, 0, vcc_lo
	.loc	0 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v140, v138, v143
	.loc	0 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cndmask_b32_e64 v143, v144, 0, vcc_lo
	.loc	0 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v139, v170, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v7, v171, v5 :: v_dual_mul_f32 v8, v172, v8
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_add_f32 v3, v138, v3 :: v_dual_mul_f32 v138, v143, v169
.Ltmp215:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v6, v142, 0, v141
.Ltmp216:
	.loc	0 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cndmask_b32_e64 v5, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v170, v3
.Ltmp217:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v6, v6, v140, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
.Ltmp218:
	.loc	0 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v137, v5, v174
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v171, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp219:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v6, v6, v7, v8
.Ltmp220:
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v172, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp221:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v144, v6, v137, v138
.Ltmp222:
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v5, v3
.Ltmp223:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1220:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ]
	ds_bpermute_b32 v169, v158, v144
.Ltmp224:
	.loc	0 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v5, v143, v3
.Ltmp225:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1209:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ]
	ds_bpermute_b32 v6, v158, v5
.Ltmp226:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1220:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v3, v169, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v143, v144, v3
	v_mov_b32_e32 v3, v167
.Ltmp227:
	.loc	0 1222 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1222:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmpx_lt_f32_e32 0, v143
	s_cbranch_execz .LBB3_43
; %bb.84:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 1223 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1223:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v3, null, 0x43e00000, 0x43e00000, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v144, v3
	v_fma_f32 v169, -v3, v144, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, vcc_lo, v143, 0x43e00000, v143
	v_mul_f32_e32 v170, v169, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v171, -v3, v170, v169
	v_fmac_f32_e32 v170, v171, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v3, v170, v169
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v144, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v3, 0x43e00000, v143
.Ltmp228:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1223:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max_num_f32_e32 v3, 0x1f800000, v3
	s_branch .LBB3_43
.Ltmp229:
.LBB3_85:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	2 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v2, v154
	s_branch .LBB3_44
.LBB3_86:                               ;   in Loop: Header=BB3_45 Depth=2
.Ltmp230:
	.loc	0 760 23 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[7:8], null, 0x408, v169, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v8, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s6
	s_cbranch_execz .LBB3_81
.LBB3_87:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	0 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[177:178], null, 0x408, v171, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	.loc	0 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	.loc	0 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s7
	s_cbranch_execnz .LBB3_82
	s_branch .LBB3_83
.Ltmp231:
.LBB3_88:
	.loc	0 1267 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1267:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB3_93
; %bb.89:
	.loc	0 1282 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1282:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_eq_u32_e32 vcc_lo, 0, v159
	s_and_b32 s1, vcc_lo, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_91
; %bb.90:
	.loc	0 1286 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1286:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_u64_u32 v[0:1], null, v145, s17, s[18:19]
	.loc	0 1288 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1288:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1287 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1287:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_lo_u32 v0, 0x102, v0
	.loc	0 1288 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1288:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	.loc	0 1290 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1290:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_store_b64 v[0:1], v[154:155], off
.LBB3_91:
	.loc	0 0 28 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:28
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 1293 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1293:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_b32 exec_lo, exec_lo, s19
	s_cbranch_execz .LBB3_93
; %bb.92:
	.loc	0 1296 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1296:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_u64_u32 v[1:2], null, v145, s17, s[18:19]
	.loc	0 1298 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1298:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v5, 2, v163
	.loc	0 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v0, v167, v129 :: v_dual_mul_f32 v105, v105, v167
	v_mul_f32_e32 v4, v121, v167
	v_mul_f32_e32 v8, v113, v167
	v_mul_f32_e32 v81, v81, v167
	.loc	0 1297 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1297:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_lo_u32 v1, 0x102, v1
	.loc	0 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v7, v124, v167 :: v_dual_mul_f32 v66, v66, v167
	v_dual_mul_f32 v57, v57, v167 :: v_dual_mul_f32 v106, v106, v167
	v_dual_mul_f32 v113, v9, v167 :: v_dual_mul_f32 v34, v34, v167
	v_mul_f32_e32 v9, v114, v167
	.loc	0 1298 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1298:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[2:3], 2, v[1:2]
	.loc	0 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v1, v167, v130 :: v_dual_mul_f32 v26, v26, v167
	v_dual_mul_f32 v107, v107, v167 :: v_dual_mul_f32 v114, v10, v167
	v_mul_f32_e32 v91, v91, v167
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	.loc	0 1298 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1298:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v6, vcc_lo, s14, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s15, v3, vcc_lo
	.loc	0 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v2, v167, v131 :: v_dual_mul_f32 v129, v167, v133
	.loc	0 1304 25 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v137, vcc_lo, v6, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v138, null, 0, v3, vcc_lo
	.loc	0 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v3, v167, v132
	v_dual_mul_f32 v5, v122, v167 :: v_dual_mul_f32 v10, v115, v167
	v_mul_f32_e32 v6, v123, v167
	v_dual_mul_f32 v83, v83, v167 :: v_dual_mul_f32 v108, v108, v167
	v_mul_f32_e32 v75, v75, v167
	v_dual_mul_f32 v115, v11, v167 :: v_dual_mul_f32 v44, v44, v167
	.loc	0 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x1
	global_store_b128 v[137:138], v[0:3], off offset:8
	global_store_b128 v[137:138], v[4:7], off offset:72
	.loc	0 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v11, v116, v167 :: v_dual_mul_f32 v2, v119, v167
	v_dual_mul_f32 v1, v118, v167 :: v_dual_mul_f32 v6, v111, v167
	v_dual_mul_f32 v4, v109, v167 :: v_dual_mul_f32 v5, v110, v167
	v_mul_f32_e32 v36, v36, v167
	v_dual_mul_f32 v7, v112, v167 :: v_dual_mul_f32 v0, v117, v167
	v_mul_f32_e32 v3, v120, v167
	v_dual_mul_f32 v97, v97, v167 :: v_dual_mul_f32 v130, v167, v134
	v_dual_mul_f32 v89, v89, v167 :: v_dual_mul_f32 v132, v167, v136
	v_dual_mul_f32 v73, v73, v167 :: v_dual_mul_f32 v122, v126, v167
	v_dual_mul_f32 v65, v65, v167 :: v_dual_mul_f32 v124, v128, v167
	v_dual_mul_f32 v49, v49, v167 :: v_dual_mul_f32 v98, v98, v167
	v_dual_mul_f32 v41, v41, v167 :: v_dual_mul_f32 v90, v90, v167
	v_dual_mul_f32 v33, v33, v167 :: v_dual_mul_f32 v82, v82, v167
	v_mul_f32_e32 v131, v167, v135
	v_dual_mul_f32 v25, v25, v167 :: v_dual_mul_f32 v74, v74, v167
	v_dual_mul_f32 v121, v125, v167 :: v_dual_mul_f32 v58, v58, v167
	v_dual_mul_f32 v123, v127, v167 :: v_dual_mul_f32 v50, v50, v167
	v_dual_mul_f32 v17, v17, v167 :: v_dual_mul_f32 v42, v42, v167
	v_dual_mul_f32 v18, v18, v167 :: v_dual_mul_f32 v99, v99, v167
	v_dual_mul_f32 v67, v67, v167 :: v_dual_mul_f32 v100, v100, v167
	v_dual_mul_f32 v59, v59, v167 :: v_dual_mul_f32 v92, v92, v167
	v_dual_mul_f32 v51, v51, v167 :: v_dual_mul_f32 v84, v84, v167
	v_dual_mul_f32 v43, v43, v167 :: v_dual_mul_f32 v76, v76, v167
	v_dual_mul_f32 v35, v35, v167 :: v_dual_mul_f32 v68, v68, v167
	v_dual_mul_f32 v27, v27, v167 :: v_dual_mul_f32 v60, v60, v167
	v_dual_mul_f32 v19, v19, v167 :: v_dual_mul_f32 v52, v52, v167
	v_dual_mul_f32 v28, v28, v167 :: v_dual_mul_f32 v85, v85, v167
	.loc	0 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_store_b128 v[137:138], v[8:11], off offset:136
	.loc	0 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v20, v20, v167 :: v_dual_mul_f32 v77, v77, v167
	v_dual_mul_f32 v116, v12, v167 :: v_dual_mul_f32 v69, v69, v167
	v_mul_f32_e32 v8, v101, v167
	v_mul_f32_e32 v12, v93, v167
	.loc	0 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x1
	global_store_b128 v[137:138], v[105:108], off offset:200
	global_store_b128 v[137:138], v[4:7], off offset:216
	.loc	0 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v61, v61, v167 :: v_dual_mul_f32 v10, v103, v167
	v_mul_f32_e32 v9, v102, v167
	v_dual_mul_f32 v11, v104, v167 :: v_dual_mul_f32 v4, v13, v167
	v_dual_mul_f32 v53, v53, v167 :: v_dual_mul_f32 v86, v86, v167
	v_dual_mul_f32 v45, v45, v167 :: v_dual_mul_f32 v78, v78, v167
	v_dual_mul_f32 v37, v37, v167 :: v_dual_mul_f32 v70, v70, v167
	v_dual_mul_f32 v29, v29, v167 :: v_dual_mul_f32 v62, v62, v167
	v_mul_f32_e32 v13, v94, v167
	v_dual_mul_f32 v54, v54, v167 :: v_dual_mul_f32 v87, v87, v167
	v_dual_mul_f32 v46, v46, v167 :: v_dual_mul_f32 v79, v79, v167
	v_dual_mul_f32 v38, v38, v167 :: v_dual_mul_f32 v71, v71, v167
	v_dual_mul_f32 v30, v30, v167 :: v_dual_mul_f32 v63, v63, v167
	v_dual_mul_f32 v5, v14, v167 :: v_dual_mul_f32 v6, v15, v167
	v_mul_f32_e32 v14, v95, v167
	v_dual_mul_f32 v55, v55, v167 :: v_dual_mul_f32 v88, v88, v167
	v_dual_mul_f32 v47, v47, v167 :: v_dual_mul_f32 v80, v80, v167
	v_dual_mul_f32 v39, v39, v167 :: v_dual_mul_f32 v72, v72, v167
	v_dual_mul_f32 v31, v31, v167 :: v_dual_mul_f32 v64, v64, v167
	v_mul_f32_e32 v15, v96, v167
	v_mul_f32_e32 v56, v56, v167
	v_mul_f32_e32 v48, v48, v167
	v_mul_f32_e32 v40, v40, v167
	v_mul_f32_e32 v32, v32, v167
	.loc	0 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_store_b128 v[137:138], v[0:3], off offset:152
	.loc	0 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v0, v21, v167 :: v_dual_mul_f32 v1, v22, v167
	v_dual_mul_f32 v2, v23, v167 :: v_dual_mul_f32 v3, v24, v167
	.loc	0 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x13
	global_store_b128 v[137:138], v[129:132], off offset:24
	global_store_b128 v[137:138], v[121:124], off offset:88
	global_store_b128 v[137:138], v[97:100], off offset:264
	global_store_b128 v[137:138], v[8:11], off offset:280
	global_store_b128 v[137:138], v[89:92], off offset:328
	global_store_b128 v[137:138], v[12:15], off offset:344
	global_store_b128 v[137:138], v[81:84], off offset:392
	global_store_b128 v[137:138], v[85:88], off offset:408
	global_store_b128 v[137:138], v[73:76], off offset:456
	global_store_b128 v[137:138], v[77:80], off offset:472
	global_store_b128 v[137:138], v[65:68], off offset:520
	global_store_b128 v[137:138], v[69:72], off offset:536
	global_store_b128 v[137:138], v[57:60], off offset:584
	global_store_b128 v[137:138], v[61:64], off offset:600
	global_store_b128 v[137:138], v[49:52], off offset:648
	global_store_b128 v[137:138], v[53:56], off offset:664
	global_store_b128 v[137:138], v[41:44], off offset:712
	global_store_b128 v[137:138], v[45:48], off offset:728
	global_store_b128 v[137:138], v[33:36], off offset:776
	global_store_b128 v[137:138], v[37:40], off offset:792
	.loc	0 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v7, v16, v167
	.loc	0 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[25:28], off offset:840
	global_store_b128 v[137:138], v[29:32], off offset:856
	global_store_b128 v[137:138], v[17:20], off offset:904
	global_store_b128 v[137:138], v[0:3], off offset:920
	global_store_b128 v[137:138], v[113:116], off offset:968
	global_store_b128 v[137:138], v[4:7], off offset:984
.Ltmp232:
.LBB3_93:
	.loc	0 1588 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1588:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp233:
.Lfunc_end3:
	.size	attention_fp8_e4m3_fa2_gqa_partial_gfx1201, .Lfunc_end3-attention_fp8_e4m3_fa2_gqa_partial_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_partial_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
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
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 191
		.amdhsa_next_free_sgpr 35
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-attention_fp8_e4m3_fa2_gqa_partial_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.num_vgpr, 191
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.numbered_sgpr, 35
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8148
; TotalNumSgprs: 37
; NumVgprs: 191
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 37
; NumVGPRsForWavesPerEU: 191
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_merge_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_merge_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_merge_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_merge_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_merge_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_merge_gfx1201
.Lfunc_begin4:
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.loc	0 1648 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1648:17
	s_load_b128 s[4:7], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s5, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s5, s7, -9
	.loc	0 1648 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1648:23
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s5, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB4_10
; %bb.1:
	.loc	0 1653 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1653:26
	v_lshrrev_b32_e32 v1, 5, v0
	.loc	0 1656 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1656:27
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1655 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1655:36
	v_lshl_or_b32 v4, ttmp9, 3, v1
	.loc	0 1656 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1656:13
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB4_10
; %bb.2:
	.loc	0 1648 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1648:17
	s_load_b128 s[0:3], s[0:1], 0x0
	.loc	0 1659 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1659:5
	v_mad_co_u64_u32 v[1:2], null, v4, s7, 0
	v_ashrrev_i32_e32 v5, 31, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[6:7], null, v5, s7, v[2:3]
	s_wait_kmcnt 0x0
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, s[0:1]
	s_mov_b32 s0, s7
	v_mad_co_u64_u32 v[2:3], null, 0x408, v6, v[2:3]
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v8, 0xff800000 :: v_dual_mov_b32 v7, v2
	v_mov_b32_e32 v6, v1
.LBB4_3:                                ; =>This Inner Loop Header: Depth=1
	.loc	0 1662 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1662:22
	global_load_b32 v3, v[6:7], off
.Ltmp234:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1662:13 ]
	v_max_num_f32_e32 v8, v8, v8
.Ltmp235:
	.loc	0 1659 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1659:23
	v_add_co_u32 v6, vcc_lo, 0x408, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
.Ltmp236:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1662:13 ]
	s_wait_loadcnt 0x0
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v8, v8, v3
.Ltmp237:
	.loc	0 1659 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1659:5
	s_cbranch_scc0 .LBB4_3
; %bb.4:
	.loc	0 1654 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1654:26
	v_and_b32_e32 v3, 31, v0
	v_lshlrev_b64_e32 v[5:6], 10, v[4:5]
	.loc	0 1664 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:5
	v_mov_b32_e32 v4, 0
	s_mov_b32 s1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v9, 2, v3
	v_add_co_u32 v0, vcc_lo, s2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s3, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v9, s0, v9, 8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, 0, s0
	s_branch .LBB4_6
.LBB4_5:                                ;   in Loop: Header=BB4_6 Depth=1
	.loc	0 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1676 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1676:13
	v_div_scale_f32 v5, null, v11, v11, v12
	v_div_scale_f32 v14, vcc_lo, v12, v11, v12
	v_rcp_f32_e32 v6, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v13, -v5, v6, 1.0
	v_fmac_f32_e32 v6, v13, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v13, v14, v6
	v_fma_f32 v15, -v5, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v13, v15, v6
	v_fma_f32 v5, -v5, v13, v14
	.loc	0 1664 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:40
	v_add_nc_u32_e32 v14, 32, v3
	.loc	0 1676 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1676:13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v13, v5, v6, v13
	.loc	0 1675 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1675:9
	v_lshlrev_b64_e32 v[5:6], 2, v[3:4]
	.loc	0 1664 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:26
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v3
	.loc	0 1676 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1676:13
	v_div_fixup_f32 v3, v13, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	0 1675 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1675:9
	v_add_co_u32 v5, s0, v0, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v7, v6, s0
	.loc	0 1676 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1676:18
	v_cmp_lt_f32_e64 s0, 0, v11
	.loc	0 1664 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:5
	s_or_b32 s1, vcc_lo, s1
	.loc	0 1676 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1676:13
	v_cndmask_b32_e64 v11, 0, v3, s0
	.loc	0 1664 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:5
	v_add_co_u32 v9, s0, 0x80, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0
	v_mov_b32_e32 v3, v14
	.loc	0 1675 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1675:44
	global_store_b32 v[5:6], v11, off
	.loc	0 1664 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:5
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB4_10
.LBB4_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_8 Depth 2
	.loc	0 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_dual_mov_b32 v6, v2 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	s_mov_b32 s0, s7
	s_branch .LBB4_8
.LBB4_7:                                ;   in Loop: Header=BB4_8 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	.loc	0 1673 24 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1673:24
	v_add_co_u32 v15, vcc_lo, v5, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v6, v10, vcc_lo
	.loc	0 1667 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1667:27
	v_add_co_u32 v5, vcc_lo, 0x408, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	.loc	0 1673 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1673:24
	global_load_b32 v15, v[15:16], off
	.loc	0 1672 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1672:17
	v_fmac_f32_e32 v11, v13, v14
	.loc	0 1667 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1667:27
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	0 1673 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1673:17
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v12, v14, v15
	.loc	0 1667 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1667:9
	s_cbranch_scc1 .LBB4_5
.LBB4_8:                                ;   Parent Loop BB4_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	0 1671 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:18
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	.loc	0 1671 17 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:17
	s_mov_b32 s2, exec_lo
	.loc	0 1671 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:24
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	.loc	0 1671 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:17
	s_cbranch_execz .LBB4_7
; %bb.9:                                ;   in Loop: Header=BB4_8 Depth=2
	.loc	0 1671 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:41
	global_load_b32 v14, v[5:6], off
	.loc	0 1671 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:47
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp238:
	.loc	2 202 42 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:34 ]
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	.loc	2 202 10 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:34 ]
	v_exp_f32_e32 v14, v14
	s_branch .LBB4_7
.Ltmp239:
.LBB4_10:
	.loc	0 1678 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1678:1
	s_endpgm
.Ltmp240:
.Lfunc_end4:
	.size	attention_fp8_e4m3_fa2_gqa_merge_gfx1201, .Lfunc_end4-attention_fp8_e4m3_fa2_gqa_merge_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_merge_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
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
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-attention_fp8_e4m3_fa2_gqa_merge_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.num_vgpr, 17
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.numbered_sgpr, 8
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_merge_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 728
; TotalNumSgprs: 10
; NumVgprs: 17
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 10
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
	.protected	attention_fp8_e4m3_fa2_gqa_packet_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_packet_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_packet_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_packet_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_packet_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_packet_gfx1201
.Lfunc_begin5:
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.loc	0 2175 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2175:17
	s_load_b128 s[4:7], s[0:1], 0x30
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	.loc	0 2175 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2175:23
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB5_56
; %bb.1:
	.loc	0 0 23                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_and_b32 s22, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2178 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2178:14
	s_cmp_gt_i32 s22, 3
	s_cbranch_scc1 .LBB5_56
; %bb.2:
	.loc	0 0 14 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_load_b32 s24, s[0:1], 0x40
	s_lshr_b32 s2, ttmp7, 7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, 0x1fffe00
	.loc	0 2180 41 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2180:41
	s_cmp_gt_i32 s7, 0x200
	.loc	0 2180 30 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2180:30
	s_cselect_b32 s23, s2, 0
	.loc	0 2181 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2181:22
	s_cmp_le_i32 s7, s23
	s_cbranch_scc1 .LBB5_56
; %bb.3:
	.loc	0 2183 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:46
	s_sub_co_i32 s2, s7, s23
	.loc	0 2184 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2184:30
	s_mul_i32 s21, ttmp9, 0x180
.Ltmp241:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:26 ]
	s_min_i32 s25, s2, 0x200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
.Ltmp242:
	.loc	0 2185 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2185:23
	s_mul_i32 s20, s25, 6
	.loc	0 2185 18 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2185:18
	s_cmp_ge_i32 s21, s20
	s_cbranch_scc1 .LBB5_56
; %bb.4:
	.loc	0 0 18                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:18
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v151, s24 :: v_dual_and_b32 v8, 15, v0
.Ltmp243:
	.loc	0 1767 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1767:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshrrev_b32_e32 v9, 5, v0
.Ltmp244:
	.loc	0 2175 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2175:17
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b128 s[16:19], s[0:1], 0x20
.Ltmp245:
	.loc	0 1778 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v1, s21, v8
	.loc	0 1770 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1770:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_bfe_u32 v150, v0, 4, 1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1778 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshl_add_u32 v3, v9, 4, v1
	.loc	0 1779 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mul_hi_i32 v1, 0x2aaaaaab, v3
	.loc	0 1782 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1782:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmp_le_i32_e64 s1, s20, v3
	v_cmp_gt_i32_e64 s0, s20, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1779 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshrrev_b32_e32 v2, 31, v1
	v_add_nc_u32_e32 v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1781 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mul_lo_u32 v1, v2, 6
	v_sub_nc_u32_e32 v1, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	0 1781 31 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[141:142], null, s22, 6, v[1:2]
	.loc	0 1780 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1780:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_nc_u32_e32 v142, s23, v2
	v_mad_co_u64_u32 v[1:2], null, v142, 24, v[141:142]
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 8, v1
	.loc	0 1800 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1800:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_and_saveexec_b32 s26, s0
	s_cbranch_execz .LBB5_8
; %bb.5:
	.loc	0 1801 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1801:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mov_b32_e32 v2, 0
.Ltmp246:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_mov_b32 s2, 0x76543210
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
.Ltmp247:
	.loc	0 1801 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1801:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshlrev_b64_e32 v[134:135], 10, v[1:2]
	.loc	0 1805 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshlrev_b32_e32 v1, 5, v150
	.loc	0 1801 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1801:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, vcc_lo, s8, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v135, vcc_lo
	.loc	0 1805 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_co_u32 v130, vcc_lo, v3, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v131, null, 0, v4, vcc_lo
	s_clause 0x1f
	global_load_b128 v[3:6], v[130:131], off
	global_load_b128 v[10:13], v[130:131], off offset:16
	global_load_b128 v[14:17], v[130:131], off offset:64
	global_load_b128 v[18:21], v[130:131], off offset:80
	global_load_b128 v[22:25], v[130:131], off offset:128
	global_load_b128 v[26:29], v[130:131], off offset:144
	global_load_b128 v[30:33], v[130:131], off offset:192
	global_load_b128 v[34:37], v[130:131], off offset:208
	global_load_b128 v[38:41], v[130:131], off offset:256
	global_load_b128 v[42:45], v[130:131], off offset:272
	global_load_b128 v[46:49], v[130:131], off offset:320
	global_load_b128 v[50:53], v[130:131], off offset:336
	global_load_b128 v[54:57], v[130:131], off offset:384
	global_load_b128 v[58:61], v[130:131], off offset:400
	global_load_b128 v[62:65], v[130:131], off offset:448
	global_load_b128 v[66:69], v[130:131], off offset:464
	global_load_b128 v[70:73], v[130:131], off offset:512
	global_load_b128 v[74:77], v[130:131], off offset:528
	global_load_b128 v[78:81], v[130:131], off offset:576
	global_load_b128 v[82:85], v[130:131], off offset:592
	global_load_b128 v[86:89], v[130:131], off offset:640
	global_load_b128 v[90:93], v[130:131], off offset:656
	global_load_b128 v[94:97], v[130:131], off offset:704
	global_load_b128 v[98:101], v[130:131], off offset:720
	global_load_b128 v[102:105], v[130:131], off offset:768
	global_load_b128 v[106:109], v[130:131], off offset:784
	global_load_b128 v[110:113], v[130:131], off offset:832
	global_load_b128 v[114:117], v[130:131], off offset:848
	global_load_b128 v[118:121], v[130:131], off offset:896
	global_load_b128 v[122:125], v[130:131], off offset:912
	global_load_b128 v[126:129], v[130:131], off offset:960
	global_load_b128 v[130:133], v[130:131], off offset:976
.Ltmp248:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x1f
	v_max3_num_f32 v1, |v3|, 0, |v4|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp249:
	.loc	2 454 44 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v5|, |v6|
.Ltmp250:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x1e
	v_max3_num_f32 v1, v1, |v10|, |v11|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp251:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v12|, |v13|
.Ltmp252:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x1d
	v_max3_num_f32 v1, v1, |v14|, |v15|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp253:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v16|, |v17|
.Ltmp254:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x1c
	v_max3_num_f32 v1, v1, |v18|, |v19|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp255:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v20|, |v21|
.Ltmp256:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x1b
	v_max3_num_f32 v1, v1, |v22|, |v23|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp257:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v24|, |v25|
.Ltmp258:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x1a
	v_max3_num_f32 v1, v1, |v26|, |v27|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp259:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v28|, |v29|
.Ltmp260:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x19
	v_max3_num_f32 v1, v1, |v30|, |v31|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp261:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v32|, |v33|
.Ltmp262:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x18
	v_max3_num_f32 v1, v1, |v34|, |v35|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp263:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v36|, |v37|
.Ltmp264:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x17
	v_max3_num_f32 v1, v1, |v38|, |v39|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp265:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v40|, |v41|
.Ltmp266:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x16
	v_max3_num_f32 v1, v1, |v42|, |v43|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp267:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v44|, |v45|
.Ltmp268:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x15
	v_max3_num_f32 v1, v1, |v46|, |v47|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp269:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v48|, |v49|
.Ltmp270:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x14
	v_max3_num_f32 v1, v1, |v50|, |v51|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp271:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v52|, |v53|
.Ltmp272:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x13
	v_max3_num_f32 v1, v1, |v54|, |v55|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp273:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v56|, |v57|
.Ltmp274:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x12
	v_max3_num_f32 v1, v1, |v58|, |v59|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp275:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v60|, |v61|
.Ltmp276:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x11
	v_max3_num_f32 v1, v1, |v62|, |v63|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp277:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v64|, |v65|
.Ltmp278:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x10
	v_max3_num_f32 v1, v1, |v66|, |v67|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp279:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v68|, |v69|
.Ltmp280:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0xf
	v_max3_num_f32 v1, v1, |v70|, |v71|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp281:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v72|, |v73|
.Ltmp282:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0xe
	v_max3_num_f32 v1, v1, |v74|, |v75|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp283:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v76|, |v77|
.Ltmp284:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0xd
	v_max3_num_f32 v1, v1, |v78|, |v79|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp285:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v80|, |v81|
.Ltmp286:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0xc
	v_max3_num_f32 v1, v1, |v82|, |v83|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp287:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v84|, |v85|
.Ltmp288:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0xb
	v_max3_num_f32 v1, v1, |v86|, |v87|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp289:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v88|, |v89|
.Ltmp290:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0xa
	v_max3_num_f32 v1, v1, |v90|, |v91|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp291:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v92|, |v93|
.Ltmp292:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x9
	v_max3_num_f32 v1, v1, |v94|, |v95|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp293:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v96|, |v97|
.Ltmp294:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x8
	v_max3_num_f32 v1, v1, |v98|, |v99|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp295:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v100|, |v101|
.Ltmp296:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x7
	v_max3_num_f32 v1, v1, |v102|, |v103|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp297:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v104|, |v105|
.Ltmp298:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x6
	v_max3_num_f32 v1, v1, |v106|, |v107|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp299:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v108|, |v109|
.Ltmp300:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x5
	v_max3_num_f32 v1, v1, |v110|, |v111|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp301:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v112|, |v113|
.Ltmp302:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x4
	v_max3_num_f32 v1, v1, |v114|, |v115|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp303:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v116|, |v117|
.Ltmp304:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x3
	v_max3_num_f32 v1, v1, |v118|, |v119|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp305:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v120|, |v121|
.Ltmp306:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x2
	v_max3_num_f32 v1, v1, |v122|, |v123|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp307:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v124|, |v125|
.Ltmp308:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x1
	v_max3_num_f32 v1, v1, |v126|, |v127|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp309:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v128|, |v129|
.Ltmp310:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x0
	v_max3_num_f32 v1, v1, |v130|, |v131|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp311:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v1, v1, |v132|, |v133|
.Ltmp312:
	.loc	0 1740 12 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_mov_b32_e32 v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v3, v3, s2, 0xfedcba98
.Ltmp313:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v1, v1, v3
.Ltmp314:
	.loc	0 1817 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v3, null, 0x43e00000, 0x43e00000, v1
	v_div_scale_f32 v6, vcc_lo, v1, 0x43e00000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v4, v3
	v_fma_f32 v5, -v3, v4, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v5, v4
	v_mul_f32_e32 v5, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v10, -v3, v5, v6
	v_fmac_f32_e32 v5, v10, v4
	.loc	0 1819 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshl_or_b32 v10, v150, 5, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	.loc	0 1817 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v3, -v3, v5, v6
	.loc	0 1819 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshl_or_b32 v6, v150, 3, v7
	.loc	0 1817 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 1819 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_co_u32 v5, s2, s10, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s11, 0, s2
	.loc	0 1817 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_fixup_f32 v11, v3, 0x43e00000, v1
	.loc	0 1819 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_co_u32 v3, vcc_lo, s8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s9, v135, vcc_lo
	.loc	0 1817 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0, v1
	s_mov_b32 s9, 16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 1.0, v11, vcc_lo
	.loc	0 1819 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_co_u32 v5, vcc_lo, v5, 4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
.LBB5_6:                                ; =>This Inner Loop Header: Depth=1
	.loc	0 1820 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1820:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_clause 0x1
	global_load_b128 v[10:13], v[3:4], off
	global_load_b128 v[14:17], v[3:4], off offset:16
	.loc	0 1819 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_co_u32 v3, vcc_lo, v3, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	.loc	0 1823 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1823:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mov_b16_e32 v19.l, v2.l
	.loc	0 1825 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mov_b16_e32 v19.h, 0
	.loc	0 1819 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s9, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s9, 0
	.loc	0 1823 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1823:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mov_b16_e32 v18.l, v19.l
	.loc	0 1825 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mov_b16_e32 v18.h, v19.h
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x1
	v_div_scale_f32 v20, null, v1, v1, v10
	.loc	0 1824 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v22, null, v1, v1, v11
	.loc	0 1826 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v24, null, v1, v1, v12
	.loc	0 1826 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v26, null, v1, v1, v13
	.loc	0 1829 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	v_div_scale_f32 v28, null, v1, v1, v14
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_rcp_f32_e32 v36, v20
	.loc	0 1824 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_rcp_f32_e32 v37, v22
	.loc	0 1826 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_rcp_f32_e32 v38, v24
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v30, null, v1, v1, v15
	.loc	0 1826 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_rcp_f32_e32 v39, v26
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v32, null, v1, v1, v16
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_rcp_f32_e32 v40, v28
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v34, null, v1, v1, v17
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_rcp_f32_e32 v41, v30
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v44, -v20, v36, 1.0
	.loc	0 1824 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v45, -v22, v37, 1.0
	.loc	0 1831 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_rcp_f32_e32 v42, v32
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v46, -v24, v38, 1.0
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v21, vcc_lo, v10, v1, v10
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_rcp_f32_e32 v43, v34
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_fmac_f32 v36, v44, v36 :: v_dual_fmac_f32 v37, v45, v37
	.loc	0 1826 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v47, -v26, v39, 1.0
	.loc	0 1824 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v23, s2, v11, v1, v11
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v48, -v28, v40, 1.0
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v25, s3, v12, v1, v12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v38, v46, v38 :: v_dual_fmac_f32 v39, v47, v39
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v49, -v30, v41, 1.0
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mul_f32 v44, v21, v36 :: v_dual_mul_f32 v45, v23, v37
	.loc	0 1826 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v27, s4, v13, v1, v13
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v50, -v32, v42, 1.0
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v29, s5, v14, v1, v14
	v_dual_fmac_f32 v40, v48, v40 :: v_dual_fmac_f32 v41, v49, v41
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v51, -v34, v43, 1.0
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mul_f32 v46, v25, v38 :: v_dual_mul_f32 v47, v27, v39
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v31, s6, v15, v1, v15
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v52, -v20, v44, v21
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v33, s7, v16, v1, v16
	v_dual_fmac_f32 v42, v50, v42 :: v_dual_fmac_f32 v43, v51, v43
	.loc	0 1824 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v53, -v22, v45, v23
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mul_f32 v48, v29, v40 :: v_dual_mul_f32 v49, v31, v41
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v35, s8, v17, v1, v17
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v54, -v24, v46, v25
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_fmac_f32 v44, v52, v36 :: v_dual_fmac_f32 v45, v53, v37
	.loc	0 1826 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v55, -v26, v47, v27
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mul_f32 v50, v33, v42 :: v_dual_mul_f32 v51, v35, v43
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v56, -v28, v48, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_fmac_f32 v46, v54, v38 :: v_dual_fmac_f32 v47, v55, v39
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v57, -v30, v49, v31
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v20, -v20, v44, v21
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v58, -v32, v50, v33
	.loc	0 1824 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v21, -v22, v45, v23
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_fmac_f32 v48, v56, v40 :: v_dual_fmac_f32 v49, v57, v41
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v59, -v34, v51, v35
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v22, -v24, v46, v25
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v20, v20, v36, v44
	.loc	0 1824 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 vcc_lo, s2
	.loc	0 1826 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v23, -v26, v47, v27
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_fmac_f32 v50, v58, v42 :: v_dual_fmac_f32 v51, v59, v43
	.loc	0 1824 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v21, v21, v37, v45
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v24, -v28, v48, v29
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v22, v22, v38, v46
	.loc	0 1826 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	0 1829 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v25, -v30, v49, v31
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_fixup_f32 v10, v20, v1, v10
	.loc	0 1826 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v20, v23, v39, v47
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 vcc_lo, s5
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v26, -v32, v50, v33
	.loc	0 1824 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_fixup_f32 v11, v21, v1, v11
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v21, v24, v40, v48
	.loc	0 1829 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 vcc_lo, s6
	.loc	0 1831 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v27, -v34, v51, v35
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_fixup_f32 v12, v22, v1, v12
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v22, v25, v41, v49
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 vcc_lo, s7
	.loc	0 1823 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1823:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cvt_pk_fp8_f32 v18.l, v10, v11
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v10, v26, v42, v50
	.loc	0 1831 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 vcc_lo, s8
	.loc	0 1826 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_fixup_f32 v13, v20, v1, v13
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_fixup_f32 v11, v21, v1, v14
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v14, v27, v43, v51
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_fixup_f32 v15, v22, v1, v15
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_fixup_f32 v10, v10, v1, v16
	.loc	0 1825 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cvt_pk_fp8_f32 v18.h, v12, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_fixup_f32 v12, v14, v1, v17
	.loc	0 1828 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1828:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cvt_pk_fp8_f32 v19.l, v11, v15
	s_delay_alu instid0(VALU_DEP_2)
	.loc	0 1830 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1830:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cvt_pk_fp8_f32 v19.h, v10, v12
	.loc	0 1835 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1835:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	global_store_b64 v[5:6], v[18:19], off offset:-4
	.loc	0 1819 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_co_u32 v5, vcc_lo, v5, 16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	.loc	0 1819 13 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_scc0 .LBB5_6
; %bb.7:
	.loc	0 1983 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1983:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mul_f32_e32 v151, s24, v1
.LBB5_8:
	.loc	0 0 47 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:47
	s_or_b32 exec_lo, exec_lo, s26
	v_and_b32_e32 v3, 31, v0
	v_mov_b32_e32 v5, -1
	v_bfrev_b32_e32 v4, -2
	.loc	0 1849 28 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1849:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_add_co_i32 s25, s25, s23
	.loc	0 1849 13 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1849:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s2, exec_lo
	.loc	0 1847 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1847:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v1, s23, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1847 49 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1847:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshl_add_u32 v1, ttmp9, 6, v1
	v_ashrrev_i32_e32 v2, 31, v1
	.loc	0 1849 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1849:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cmpx_gt_i32_e64 s25, v1
	s_cbranch_execz .LBB5_10
; %bb.9:
	.loc	0 0 13 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v4, vcc_lo, s18, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s19, v5, vcc_lo
	.loc	0 1850 23 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	global_load_b32 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v5, v4
.LBB5_10:
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_or_b32 exec_lo, exec_lo, s2
	.loc	0 1848 25 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1848:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v6, 32, v1
	.loc	0 1854 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s25, v6
	s_cbranch_execz .LBB5_12
; %bb.11:
	.loc	0 0 13 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s18, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s19, v2, vcc_lo
	.loc	0 1855 23 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	global_load_b32 v1, v[1:2], off offset:128
.Ltmp315:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:15 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_loadcnt 0x0
	v_max_i32_e32 v5, v5, v1
.Ltmp316:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:15 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_min_i32_e32 v4, v4, v1
.Ltmp317:
.LBB5_12:
	.loc	2 0 10 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:10
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
.Ltmp318:
	v_mbcnt_lo_u32_b32 v1, -1, 0
	s_mov_b32 s7, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
.Ltmp319:
	.loc	1 523 20 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_xor_b32_e32 v2, 16, v1
.Ltmp320:
	.loc	1 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_xor_b32_e32 v10, 8, v1
.Ltmp321:
	.loc	1 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v1, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp322:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
.Ltmp323:
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_lshlrev_b32_e32 v2, 2, v2
.Ltmp324:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v1, v10, vcc_lo
.Ltmp325:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v6, v2, v5
.Ltmp326:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v2, v2, v4
.Ltmp327:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_lshlrev_b32_e32 v10, 2, v10
.Ltmp328:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v5, v5, v6
.Ltmp329:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v4, v2
.Ltmp330:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v4, v10, v5
.Ltmp331:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v6, v10, v2
.Ltmp332:
	.loc	1 523 20 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_xor_b32_e32 v10, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v1, v10, vcc_lo
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_lshlrev_b32_e32 v10, 2, v10
.Ltmp333:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v5, v4
.Ltmp334:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v2, v6
.Ltmp335:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v5, v10, v4
.Ltmp336:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1866:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v6, v10, v2
.Ltmp337:
	.loc	1 523 20 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_xor_b32_e32 v10, 2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v1, v10, vcc_lo
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_lshlrev_b32_e32 v152, 2, v10
.Ltmp338:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_xor_b32_e32 v10, 1, v1
.Ltmp339:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v5
.Ltmp340:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1866:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v2, v6
.Ltmp341:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v10, vcc_lo
.Ltmp342:
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v5, v152, v4
.Ltmp343:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1867:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v6, v152, v2
.Ltmp344:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_lshlrev_b32_e32 v153, 2, v1
.Ltmp345:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v4, v5
.Ltmp346:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1867:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v1, v2, v6
.Ltmp347:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v2, v153, v4
.Ltmp348:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1868:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v5, v153, v1
.Ltmp349:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v4, v2
.Ltmp350:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1868:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp351:
	.loc	0 1869 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_readfirstlane_b32 s5, v2
	.loc	0 1870 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1870:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_readfirstlane_b32 s3, v1
	.loc	0 1874 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cmp_lt_i32 s5, 0
	.loc	0 1874 5 is_stmt 0              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_scc1 .LBB5_53
; %bb.13:
	.loc	0 0 5                           ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_mul_lo_u16 v1.l, 0x56, v9.l
	v_mul_u32_u24_e32 v2, 0x2ab, v0
	v_mov_b16_e32 v1.h, 0
	.loc	0 1874 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mov_b32 v155, 0 :: v_dual_lshlrev_b32 v154, 3, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_lshrrev_b16 v1.l, 8, v1.l
	.loc	0 1872 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1872:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_addk_co_i32 s21, 0x17f
	.loc	0 1774 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1774:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshl_add_u32 v4, v9, 11, 0
	.loc	0 1872 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1872:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s21, s20
	v_lshlrev_b32_e32 v156, 3, v9
	v_mul_lo_u16 v1.l, v1.l, 3
	s_cselect_b32 s20, -1, 0
	s_lshl_b32 s6, s22, 8
	v_and_b32_e32 v5, 3, v0
	.loc	0 1874 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[8:9], s[12:13], s[6:7]
	v_sub_nc_u16 v2.l, v9.l, v1.l
	v_mov_b16_e32 v1.l, v2.h
	.loc	0 1874 10 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_min_i32 s21, s5, s3
	v_ashrrev_i32_e32 v143, 31, v142
	v_bfe_u32 v11, v0, 5, 1
	v_and_b32_e32 v2, 0xff, v2
	.loc	0 1874 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshl_or_b32 v6, v1, 5, v154
	.loc	0 1791 10 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1791:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cndmask_b32_e64 v7, v7, 0, s1
	v_cmp_gt_u32_e64 s2, 0xc0, v0
	s_add_nc_u64 s[24:25], s[14:15], s[6:7]
	v_mad_u32_u24 v1, v1, 3, v2
	.loc	0 1874 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v144, s3, s8, v6
	v_lshl_or_b32 v157, v2, 4, v8
	v_lshrrev_b32_e32 v2, 2, v8
	v_lshl_add_u32 v9, v1, 9, 0
	v_dual_mov_b32 v8, 0x5040100 :: v_dual_and_b32 v1, 1, v0
	v_mov_b32_e32 v6, 0x6020400
	v_lshlrev_b32_e32 v12, 3, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v145, null, s9, 0, s3
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_lshl_b32 s6, s22, 1
	v_mov_b32_e32 v205, 1.0
	s_add_nc_u64 s[8:9], s[12:13], s[6:7]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v158, 0x3070105, v6, vcc_lo
	v_lshl_or_b32 v6, v5, 6, v2
	v_lshlrev_b64_e32 v[1:2], 2, v[142:143]
	v_cmp_gt_u32_e32 vcc_lo, 2, v5
	v_lshlrev_b32_e32 v5, 5, v5
	v_lshlrev_b32_e32 v10, 4, v3
	v_lshl_add_u32 v143, v0, 1, 0
	v_add_co_u32 v148, s4, s18, v1
	v_and_b32_e32 v1, 16, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v159, 0x3020706, v8, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, v150, v11
	v_add_co_ci_u32_e64 v149, null, s19, v2, s4
	v_add_nc_u32_e32 v161, 0, v1
	v_lshlrev_b32_e32 v1, 2, v6
	v_add_co_u32 v2, s4, s10, v7
	v_or_b32_e32 v8, 12, v6
	s_and_b32 s18, s2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_add3_u32 v175, v4, v1, v5
	v_lshlrev_b32_e32 v1, 6, v0
	v_or_b32_e32 v5, 0x108, v6
	v_add_co_u32 v200, vcc_lo, v2, v154
	v_or_b32_e32 v11, 0x10c, v6
	v_and_b32_e32 v13, 0xf000, v1
	v_and_b32_e32 v14, 0x3000, v1
	v_or_b32_e32 v1, 16, v6
	v_xor_b32_e32 v5, v5, v12
	v_add_co_ci_u32_e64 v7, null, s11, 0, s4
	v_dual_mov_b32 v207, 0xff800000 :: v_dual_add_nc_u32 v202, v9, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v1, v1, v12
	v_lshl_add_u32 v177, v5, 2, v4
	v_or_b32_e32 v5, 20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v201, null, 0, v7, vcc_lo
	v_lshl_add_u32 v180, v1, 2, v4
	v_or_b32_e32 v1, 0x118, v6
	v_xor_b32_e32 v5, v5, v12
	s_add_nc_u64 s[10:11], s[14:15], s[6:7]
	s_mov_b32 s6, 0x76543210
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v1, v1, v12
	v_lshl_add_u32 v182, v5, 2, v4
	v_or_b32_e32 v5, 0x11c, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v185, v1, 2, v4
	v_or_b32_e32 v1, 44, v6
	v_xor_b32_e32 v5, v5, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v1, v1, v12
	v_lshl_add_u32 v187, v5, 2, v4
	v_or_b32_e32 v5, 48, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v190, v1, 2, v4
	v_or_b32_e32 v1, 0x134, v6
	v_xor_b32_e32 v5, v5, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v1, v1, v12
	v_lshl_add_u32 v192, v5, 2, v4
	v_or_b32_e32 v5, 0x138, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v195, v1, 2, v4
	v_mov_b32_e32 v1, 0
	v_lshlrev_b32_e32 v3, 3, v3
	v_xor_b32_e32 v5, v5, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v7, v1 :: v_dual_add_nc_u32 v160, 0, v10
	v_mov_b32_e32 v2, v1
	v_add_co_u32 v146, s3, s24, v3
	v_add_nc_u32_e32 v162, v4, v3
	v_xad_u32 v163, 0x120, v3, v4
	v_xad_u32 v164, 0x124, v3, v4
	v_xad_u32 v165, 0x240, v3, v4
	v_xad_u32 v166, 0x244, v3, v4
	v_xad_u32 v167, 0x360, v3, v4
	v_xad_u32 v168, 0x364, v3, v4
	v_xad_u32 v169, 0x520, v3, v4
	v_xad_u32 v170, 0x524, v3, v4
	v_xad_u32 v171, 0x640, v3, v4
	v_xad_u32 v172, 0x644, v3, v4
	v_xad_u32 v173, 0x760, v3, v4
	v_xad_u32 v174, 0x764, v3, v4
	v_or_b32_e32 v3, 8, v6
	v_lshl_add_u32 v197, v5, 2, v4
	v_mov_b32_e32 v5, v1
	v_xor_b32_e32 v8, v8, v12
	v_xor_b32_e32 v11, v11, v12
	v_xor_b32_e32 v3, v3, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v147, null, s25, 0, s3
	v_lshl_add_u32 v178, v8, 2, v4
	v_or_b32_e32 v8, 0x114, v6
	v_lshl_add_u32 v176, v3, 2, v4
	v_or_b32_e32 v3, 0x110, v6
	v_lshl_add_u32 v179, v11, 2, v4
	v_or_b32_e32 v11, 24, v6
	v_xor_b32_e32 v8, v8, v12
	v_cmp_gt_u32_e64 s3, 48, v0
	v_xor_b32_e32 v3, v3, v12
	v_mov_b32_e32 v129, v1
	v_xor_b32_e32 v11, v11, v12
	v_lshl_add_u32 v183, v8, 2, v4
	v_or_b32_e32 v8, 40, v6
	v_lshl_add_u32 v181, v3, 2, v4
	v_or_b32_e32 v3, 28, v6
	v_lshl_add_u32 v184, v11, 2, v4
	v_or_b32_e32 v11, 0x128, v6
	v_xor_b32_e32 v8, v8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, v3, v12
	v_xor_b32_e32 v11, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v188, v8, 2, v4
	v_or_b32_e32 v8, 0x130, v6
	v_lshl_add_u32 v186, v3, 2, v4
	v_or_b32_e32 v3, 0x12c, v6
	v_lshl_add_u32 v189, v11, 2, v4
	v_or_b32_e32 v11, 52, v6
	v_xor_b32_e32 v8, v8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, v3, v12
	v_xor_b32_e32 v11, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v193, v8, 2, v4
	v_or_b32_e32 v8, 60, v6
	v_lshl_add_u32 v191, v3, 2, v4
	v_or_b32_e32 v3, 56, v6
	v_or_b32_e32 v6, 0x13c, v6
	v_lshl_add_u32 v194, v11, 2, v4
	v_xor_b32_e32 v8, v8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v3, v3, v12
	v_xor_b32_e32 v6, v6, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v198, v8, 2, v4
	v_mov_b32_e32 v8, v1
	v_lshl_add_u32 v196, v3, 2, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v199, v6, 2, v4
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v4, v1
	v_dual_mov_b32 v6, v1 :: v_dual_add_nc_u32 v203, v160, v14
	v_add_nc_u32_e32 v204, v160, v13
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v11, v3
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v45, v5
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v53, v5
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v61, v5
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v69, v5
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v77, v5
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v85, v5
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v93, v5
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v101, v5
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v109, v5
	v_dual_mov_b32 v120, v8 :: v_dual_mov_b32 v117, v5
	v_dual_mov_b32 v128, v8 :: v_dual_mov_b32 v125, v5
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v9, v1 :: v_dual_mov_b32 v10, v2
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v39, v7 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v35, v3 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v71, v7 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v79, v7 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v87, v7 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v95, v7 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v103, v7 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v99, v3 :: v_dual_mov_b32 v100, v4
	v_dual_mov_b32 v97, v1 :: v_dual_mov_b32 v98, v2
	v_dual_mov_b32 v111, v7 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v107, v3 :: v_dual_mov_b32 v108, v4
	v_dual_mov_b32 v105, v1 :: v_dual_mov_b32 v106, v2
	v_dual_mov_b32 v119, v7 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v115, v3 :: v_dual_mov_b32 v116, v4
	v_dual_mov_b32 v113, v1 :: v_dual_mov_b32 v114, v2
	v_dual_mov_b32 v127, v7 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v124, v4
	v_dual_mov_b32 v121, v1 :: v_dual_mov_b32 v122, v2
	s_branch .LBB5_15
.LBB5_14:                               ;   in Loop: Header=BB5_15 Depth=1
.Ltmp352:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2112:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2112:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ] ] ]
	s_barrier_signal -1
	v_mov_b32_e32 v207, v130
.Ltmp353:
	.loc	0 1874 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_add_co_i32 s7, s7, 48
	.loc	0 1874 39 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s7, s5
.Ltmp354:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2112:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ] ] ]
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2112:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp355:
	.loc	0 1874 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_scc1 .LBB5_54
.LBB5_15:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_45 Depth 2
                                        ;       Child Loop BB5_47 Depth 3
	.loc	0 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_dual_mov_b32 v133, 0 :: v_dual_add_nc_u32 v134, s7, v157
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v131, 0
	v_mov_b32_e32 v130, 0
	.loc	0 1887 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1887:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_ge_i32_e64 s5, v134
	s_cbranch_execz .LBB5_17
; %bb.16:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1894 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1894:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[132:133], null, 0x408, v134, v[144:145]
	s_clause 0x1
	global_load_b64 v[130:131], v[132:133], off
	global_load_b64 v[132:133], v[132:133], off offset:16
.LBB5_17:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1902 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v202, v[130:133]
	.loc	0 1905 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1905:18 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB5_35
; %bb.18:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 18 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:18
	v_dual_mov_b32 v133, 0 :: v_dual_add_nc_u32 v136, s7, v156
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v131, 0
	v_mov_b32_e32 v130, 0
	.loc	0 1910 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s12, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmpx_ge_i32_e64 s5, v136
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_execz .LBB5_20
; %bb.19:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[130:131], null, 0x408, v136, v[146:147]
	global_load_b64 v[130:131], v[130:131], off
.LBB5_20:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_add_nc_u32_e32 v134, 0x6000, v162
	.loc	0 1910 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s12, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v134, v130, v131 offset1:1
	.loc	0 1908 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmpx_gt_i32_e64 s5, v136
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_execz .LBB5_22
; %bb.21:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1907 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v130, 1, v136
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[130:131], null, 0x408, v130, v[146:147]
	global_load_b64 v[132:133], v[130:131], off
.LBB5_22:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v137, 2, v136
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s12, exec_lo
	.loc	0 1916 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v163, v132 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b32 v164, v133 offset:24576
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmpx_ge_i32_e64 s5, v137
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_execz .LBB5_24
; %bb.23:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[132:133], null, 0x408, v137, v[146:147]
	global_load_b64 v[134:135], v[132:133], off
.LBB5_24:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v132, 3, v136
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s12, exec_lo
	.loc	0 1916 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v165, v134 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b32 v166, v135 offset:24576
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmpx_ge_i32_e64 s5, v132
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_execz .LBB5_26
; %bb.25:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[130:131], null, 0x408, v132, v[146:147]
	global_load_b64 v[130:131], v[130:131], off
.LBB5_26:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v137, 4, v136
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s12, exec_lo
	.loc	0 1916 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v167, v130 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b32 v168, v131 offset:24576
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmpx_ge_i32_e64 s5, v137
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_execz .LBB5_28
; %bb.27:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[130:131], null, 0x408, v137, v[146:147]
	global_load_b64 v[134:135], v[130:131], off
.LBB5_28:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v130, 5, v136
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_nc_u32_e32 v131, 0x6400, v162
	.loc	0 1910 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s12, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v131, v134, v135 offset1:1
	.loc	0 1908 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmpx_ge_i32_e64 s5, v130
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_execz .LBB5_30
; %bb.29:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[130:131], null, 0x408, v130, v[146:147]
	global_load_b64 v[132:133], v[130:131], off
.LBB5_30:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v137, 6, v136
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s12, exec_lo
	.loc	0 1916 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v169, v132 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b32 v170, v133 offset:24576
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmpx_ge_i32_e64 s5, v137
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_execz .LBB5_32
; %bb.31:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[132:133], null, 0x408, v137, v[146:147]
	global_load_b64 v[134:135], v[132:133], off
.LBB5_32:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v132, 7, v136
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s12, exec_lo
	.loc	0 1916 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v171, v134 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b32 v172, v135 offset:24576
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmpx_ge_i32_e64 s5, v132
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_execz .LBB5_34
; %bb.33:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[130:131], null, 0x408, v132, v[146:147]
	global_load_b64 v[130:131], v[130:131], off
.LBB5_34:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	.loc	0 1916 62 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v173, v130 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b32 v174, v131 offset:24576
.LBB5_35:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 66 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:66
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp356:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1922:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ] ] ]
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1922:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp357:
	.loc	0 1924 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1924:18 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_and_saveexec_b32 s4, s18
	s_cbranch_execz .LBB5_37
; %bb.36:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_nc_u32_e32 v134, 0x6000, v175
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_nc_u32_e32 v135, 0x6400, v175
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_2addr_b32 v[130:131], v134 offset1:4
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_2addr_b32 v[132:133], v135 offset1:4
.Ltmp358:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v136, v153, v130
.Ltmp359:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v137, v153, v132
.Ltmp360:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v138, v153, v131
.Ltmp361:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v139, v153, v133
.Ltmp362:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v136, v130, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v137, v132, v158
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v138, v131, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v139, v133, v158
.Ltmp363:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v131, v152, v130
.Ltmp364:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v137, v152, v132
.Ltmp365:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v138, v152, v136
.Ltmp366:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v139, v152, v133
.Ltmp367:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v131, v130, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v137, v132, v159
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v138, v136, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v139, v133, v159
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b128 v203, v[130:133] offset:12288
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v130, v176 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v131, v177 offset:24576
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v132, v178 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v133, v179 offset:24576
	ds_load_b32 v136, v181 offset:24576
	ds_load_b32 v137, v183 offset:24576
.Ltmp368:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x5
	ds_bpermute_b32 v138, v153, v130
.Ltmp369:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x5
	ds_bpermute_b32 v139, v153, v131
.Ltmp370:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x5
	ds_bpermute_b32 v140, v153, v132
.Ltmp371:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x5
	ds_bpermute_b32 v206, v153, v133
.Ltmp372:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v138, v130, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v139, v131, v158
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v140, v132, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v206, v133, v158
.Ltmp373:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v138, v152, v130
.Ltmp374:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v139, v152, v131
.Ltmp375:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v140, v152, v132
.Ltmp376:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v206, v152, v133
.Ltmp377:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v138, v130, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v139, v131, v159
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v140, v132, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v206, v133, v159
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b128 v203, v[130:133] offset:12800
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v130, v180 offset:24576
	ds_load_b32 v131, v182 offset:24576
.Ltmp378:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v132, v153, v136
	ds_bpermute_b32 v133, v153, v137
.Ltmp379:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v153, v130
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v153, v131
.Ltmp380:
	.loc	0 1947 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v132, v132, v136, v158
	s_wait_dscnt 0x2
	v_perm_b32 v133, v133, v137, v158
.Ltmp381:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v137, v152, v133
.Ltmp382:
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v130, v138, v130, v158
	s_wait_dscnt 0x1
	v_perm_b32 v136, v139, v131, v158
.Ltmp383:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v131, v152, v132
.Ltmp384:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v138, v152, v130
	ds_bpermute_b32 v139, v152, v136
.Ltmp385:
	.loc	0 1952 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v133, v137, v133, v159
	s_wait_dscnt 0x2
	v_perm_b32 v131, v131, v132, v159
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v130, v138, v130, v159
	s_wait_dscnt 0x0
	v_perm_b32 v132, v139, v136, v159
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b128 v203, v[130:133] offset:13312
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v130, v184 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v131, v185 offset:24576
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v132, v186 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v133, v187 offset:24576
.Ltmp386:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v153, v130
.Ltmp387:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v153, v131
.Ltmp388:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v153, v132
.Ltmp389:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v153, v133
.Ltmp390:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v136, v130, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v137, v131, v158
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v138, v132, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v139, v133, v158
.Ltmp391:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v136, v152, v130
.Ltmp392:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v137, v152, v131
.Ltmp393:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v138, v152, v132
.Ltmp394:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v139, v152, v133
.Ltmp395:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v136, v130, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v137, v131, v159
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v138, v132, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v139, v133, v159
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b128 v203, v[130:133] offset:13824
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_2addr_b32 v[130:131], v134 offset0:32 offset1:36
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_2addr_b32 v[132:133], v135 offset0:32 offset1:36
.Ltmp396:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v134, v153, v130
.Ltmp397:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v135, v153, v132
.Ltmp398:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v136, v153, v131
.Ltmp399:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v137, v153, v133
.Ltmp400:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v134, v130, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v135, v132, v158
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v134, v136, v131, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v137, v133, v158
.Ltmp401:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v131, v152, v130
.Ltmp402:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v135, v152, v132
.Ltmp403:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v136, v152, v134
.Ltmp404:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v137, v152, v133
.Ltmp405:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v131, v130, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v135, v132, v159
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v136, v134, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v137, v133, v159
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b128 v203, v[130:133] offset:14336
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v130, v188 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v131, v189 offset:24576
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v132, v190 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v133, v191 offset:24576
.Ltmp406:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v153, v130
.Ltmp407:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v153, v131
.Ltmp408:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v153, v132
.Ltmp409:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v153, v133
.Ltmp410:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v134, v130, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v135, v131, v158
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v136, v132, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v137, v133, v158
.Ltmp411:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v134, v152, v130
.Ltmp412:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v135, v152, v131
.Ltmp413:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v136, v152, v132
.Ltmp414:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v137, v152, v133
.Ltmp415:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v134, v130, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v135, v131, v159
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v136, v132, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v137, v133, v159
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b128 v203, v[130:133] offset:14848
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v130, v192 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v131, v193 offset:24576
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v132, v194 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v133, v195 offset:24576
.Ltmp416:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v153, v130
.Ltmp417:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v153, v131
.Ltmp418:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v153, v132
.Ltmp419:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v153, v133
.Ltmp420:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v134, v130, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v135, v131, v158
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v136, v132, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v137, v133, v158
.Ltmp421:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v134, v152, v130
.Ltmp422:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v135, v152, v131
.Ltmp423:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v136, v152, v132
.Ltmp424:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v137, v152, v133
.Ltmp425:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v134, v130, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v135, v131, v159
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v136, v132, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v137, v133, v159
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b128 v203, v[130:133] offset:15360
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v130, v196 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v131, v197 offset:24576
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v132, v198 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b32 v133, v199 offset:24576
.Ltmp426:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v153, v130
.Ltmp427:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v153, v131
.Ltmp428:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v153, v132
.Ltmp429:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v153, v133
.Ltmp430:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v134, v130, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v135, v131, v158
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v136, v132, v158
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v137, v133, v158
.Ltmp431:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v134, v152, v130
.Ltmp432:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v135, v152, v131
.Ltmp433:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v136, v152, v132
.Ltmp434:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	ds_bpermute_b32 v137, v152, v133
.Ltmp435:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v130, v134, v130, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v135, v131, v159
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v132, v136, v132, v159
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v133, v137, v133, v159
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b128 v204, v[130:133] offset:15872
.LBB5_37:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 77 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:77
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1961 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_41
; %bb.38:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 17 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:17
	v_dual_mov_b32 v130, 0 :: v_dual_add_nc_u32 v131, s7, v0
	.loc	0 1964 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s12, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s5, v131
	s_cbranch_execz .LBB5_40
; %bb.39:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1969 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[132:133], null, 0x408, v131, s[8:9]
	.loc	0 1971 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mad_co_u64_u32 v[134:135], null, 0x408, v131, s[10:11]
	.loc	0 1969 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	global_load_d16_b16 v130, v[132:133], off offset:1024
	.loc	0 1971 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	global_load_d16_hi_b16 v130, v[134:135], off offset:1024
	.loc	0 1969 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v131.h, 8, v130.l
	.loc	0 1971 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshrrev_b16 v131.l, 8, v130.h
	.loc	0 1971 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_and_b16 v132.h, 0xff, v130.l
	v_and_b16 v132.l, 0xff, v130.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1972 56 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_pk_lshlrev_b16 v130, 8, v131 op_sel_hi:[0,1]
	.loc	0 1971 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_or_b32_e32 v130, v130, v132
.LBB5_40:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 53 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:53
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	.loc	0 1974 61 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b16_d16_hi v143, v130 offset:36864
	.loc	0 1975 61                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1975:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_store_b16 v143, v130 offset:36960
.LBB5_41:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 61 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:61
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp436:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
.Ltmp437:
	.loc	0 1878 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1878:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_add_co_i32 s4, s7, 47
	v_mov_b32_e32 v206, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s21
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s12, s20, s4
	s_wait_alu depctr_sa_sdst(0)
	s_nor_b32 s13, s1, s12
.Ltmp438:
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ] ] ]
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp439:
	.loc	0 1984 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1984:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s13
	s_cbranch_execz .LBB5_43
; %bb.42:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1984 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1984:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	global_load_b32 v206, v[148:149], off
.LBB5_43:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 46                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:46
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_mov_b32 s13, 0
	s_branch .LBB5_45
.LBB5_44:                               ;   in Loop: Header=BB5_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_sub_f32 v209, v207, v130 :: v_dual_add_f32 v132, v132, v208
	.loc	0 2051 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v207
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v209, 0x3fb8aa3b, v209
	v_exp_f32_e32 v209, v209
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v207, 0, v209, vcc_lo
	.loc	0 2066 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fmac_f32_e32 v132, v129, v207
	.loc	0 2075 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2075:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mul_f32_e32 v129, v205, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2075 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2075:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v205, null, v131, v131, v129
	v_rcp_f32_e32 v207, v205
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v208, -v205, v207, 1.0
	v_fmac_f32_e32 v207, v208, v207
	v_div_scale_f32 v208, vcc_lo, v129, v131, v129
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v209, v208, v207
	v_fma_f32 v210, -v205, v209, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v209, v210, v207
	v_fma_f32 v205, -v205, v209, v208
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v205, v205, v207, v209
	v_div_fixup_f32 v129, v205, v131, v129
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mul_f32 v128, v128, v129 :: v_dual_mul_f32 v127, v127, v129
	v_mul_f32_e32 v124, v124, v129
	v_dual_mul_f32 v126, v126, v129 :: v_dual_mul_f32 v125, v125, v129
	v_dual_mul_f32 v120, v120, v129 :: v_dual_mul_f32 v123, v123, v129
	v_mul_f32_e32 v118, v118, v129
	v_dual_mul_f32 v122, v122, v129 :: v_dual_mul_f32 v121, v121, v129
	v_dual_mul_f32 v116, v116, v129 :: v_dual_mul_f32 v119, v119, v129
	v_dual_mul_f32 v114, v114, v129 :: v_dual_mul_f32 v117, v117, v129
	v_dual_mul_f32 v112, v112, v129 :: v_dual_mul_f32 v115, v115, v129
	v_dual_mul_f32 v110, v110, v129 :: v_dual_mul_f32 v113, v113, v129
	v_dual_mul_f32 v108, v108, v129 :: v_dual_mul_f32 v111, v111, v129
	v_dual_mul_f32 v106, v106, v129 :: v_dual_mul_f32 v109, v109, v129
	v_dual_mul_f32 v104, v104, v129 :: v_dual_mul_f32 v107, v107, v129
	v_dual_mul_f32 v102, v102, v129 :: v_dual_mul_f32 v105, v105, v129
	v_dual_mul_f32 v100, v100, v129 :: v_dual_mul_f32 v103, v103, v129
	v_dual_mul_f32 v98, v98, v129 :: v_dual_mul_f32 v101, v101, v129
	v_dual_mul_f32 v96, v96, v129 :: v_dual_mul_f32 v99, v99, v129
	v_dual_mul_f32 v94, v94, v129 :: v_dual_mul_f32 v97, v97, v129
	v_dual_mul_f32 v92, v92, v129 :: v_dual_mul_f32 v95, v95, v129
	v_dual_mul_f32 v90, v90, v129 :: v_dual_mul_f32 v93, v93, v129
	v_dual_mul_f32 v88, v88, v129 :: v_dual_mul_f32 v91, v91, v129
	v_dual_mul_f32 v86, v86, v129 :: v_dual_mul_f32 v89, v89, v129
	v_dual_mul_f32 v84, v84, v129 :: v_dual_mul_f32 v87, v87, v129
	v_dual_mul_f32 v82, v82, v129 :: v_dual_mul_f32 v85, v85, v129
	v_dual_mul_f32 v80, v80, v129 :: v_dual_mul_f32 v83, v83, v129
	v_dual_mul_f32 v78, v78, v129 :: v_dual_mul_f32 v81, v81, v129
	v_dual_mul_f32 v76, v76, v129 :: v_dual_mul_f32 v79, v79, v129
	v_dual_mul_f32 v74, v74, v129 :: v_dual_mul_f32 v77, v77, v129
	v_dual_mul_f32 v72, v72, v129 :: v_dual_mul_f32 v75, v75, v129
	v_dual_mul_f32 v70, v70, v129 :: v_dual_mul_f32 v73, v73, v129
	v_dual_mul_f32 v68, v68, v129 :: v_dual_mul_f32 v71, v71, v129
	v_dual_mul_f32 v66, v66, v129 :: v_dual_mul_f32 v69, v69, v129
	v_dual_mul_f32 v64, v64, v129 :: v_dual_mul_f32 v67, v67, v129
	v_dual_mul_f32 v62, v62, v129 :: v_dual_mul_f32 v65, v65, v129
	v_dual_mul_f32 v60, v60, v129 :: v_dual_mul_f32 v63, v63, v129
	v_dual_mul_f32 v58, v58, v129 :: v_dual_mul_f32 v61, v61, v129
	v_dual_mul_f32 v56, v56, v129 :: v_dual_mul_f32 v59, v59, v129
	v_dual_mul_f32 v54, v54, v129 :: v_dual_mul_f32 v57, v57, v129
	v_dual_mul_f32 v52, v52, v129 :: v_dual_mul_f32 v55, v55, v129
	v_dual_mul_f32 v50, v50, v129 :: v_dual_mul_f32 v53, v53, v129
	v_dual_mul_f32 v48, v48, v129 :: v_dual_mul_f32 v51, v51, v129
	v_dual_mul_f32 v46, v46, v129 :: v_dual_mul_f32 v49, v49, v129
	v_dual_mul_f32 v44, v44, v129 :: v_dual_mul_f32 v47, v47, v129
	v_dual_mul_f32 v42, v42, v129 :: v_dual_mul_f32 v45, v45, v129
	v_dual_mul_f32 v40, v40, v129 :: v_dual_mul_f32 v43, v43, v129
	v_dual_mul_f32 v38, v38, v129 :: v_dual_mul_f32 v41, v41, v129
	v_dual_mul_f32 v36, v36, v129 :: v_dual_mul_f32 v39, v39, v129
	v_dual_mul_f32 v34, v34, v129 :: v_dual_mul_f32 v37, v37, v129
	v_dual_mul_f32 v32, v32, v129 :: v_dual_mul_f32 v35, v35, v129
	v_dual_mul_f32 v30, v30, v129 :: v_dual_mul_f32 v33, v33, v129
	v_dual_mul_f32 v28, v28, v129 :: v_dual_mul_f32 v31, v31, v129
	v_dual_mul_f32 v26, v26, v129 :: v_dual_mul_f32 v29, v29, v129
	v_dual_mul_f32 v24, v24, v129 :: v_dual_mul_f32 v27, v27, v129
	v_dual_mul_f32 v22, v22, v129 :: v_dual_mul_f32 v25, v25, v129
	v_dual_mul_f32 v20, v20, v129 :: v_dual_mul_f32 v23, v23, v129
	v_dual_mul_f32 v18, v18, v129 :: v_dual_mul_f32 v21, v21, v129
	v_dual_mul_f32 v16, v16, v129 :: v_dual_mul_f32 v19, v19, v129
	v_dual_mul_f32 v14, v14, v129 :: v_dual_mul_f32 v17, v17, v129
	v_dual_mul_f32 v12, v12, v129 :: v_dual_mul_f32 v15, v15, v129
	v_dual_mul_f32 v10, v10, v129 :: v_dual_mul_f32 v13, v13, v129
	v_dual_mul_f32 v8, v8, v129 :: v_dual_mul_f32 v11, v11, v129
	v_dual_mul_f32 v6, v6, v129 :: v_dual_mul_f32 v9, v9, v129
	v_dual_mul_f32 v4, v4, v129 :: v_dual_mul_f32 v7, v7, v129
	v_dual_mul_f32 v2, v2, v129 :: v_dual_mul_f32 v5, v5, v129
	v_mul_f32_e32 v3, v3, v129
	v_mul_f32_e32 v1, v1, v129
	.loc	0 2081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v129, null, v131, v131, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v205, v129
	v_fma_f32 v207, -v129, v205, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v205, v207, v205
	v_div_scale_f32 v207, vcc_lo, v140, v131, v140
	v_mul_f32_e32 v208, v207, v205
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v209, -v129, v208, v207
	v_fmac_f32_e32 v208, v209, v205
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v129, -v129, v208, v207
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v129, v129, v205, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v129, v129, v131, v140
	.loc	0 2081 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v140, null, v131, v131, v139
	v_rcp_f32_e32 v205, v140
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v207, -v140, v205, 1.0
	v_fmac_f32_e32 v205, v207, v205
	v_div_scale_f32 v207, vcc_lo, v139, v131, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v208, v207, v205
	v_fma_f32 v209, -v140, v208, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v208, v209, v205
	v_fma_f32 v140, -v140, v208, v207
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v140, v140, v205, v208
	v_div_fixup_f32 v205, v140, v131, v139
	.loc	0 2080 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2080:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mov_b16_e64 v140.l, v155.l
	.loc	0 2082 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mov_b16_e64 v140.h, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 2080 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2080:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mov_b16_e64 v139.l, v140.l
	.loc	0 2082 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mov_b16_e64 v139.h, v140.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	0 2080 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2080:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cvt_pk_fp8_f32 v139.l, v129, v205
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v129, null, v131, v131, v138
	v_rcp_f32_e32 v205, v129
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v207, -v129, v205, 1.0
	v_fmac_f32_e32 v205, v207, v205
	v_div_scale_f32 v207, vcc_lo, v138, v131, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v208, v207, v205
	v_fma_f32 v209, -v129, v208, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v208, v209, v205
	v_fma_f32 v129, -v129, v208, v207
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v129, v129, v205, v208
	v_div_fixup_f32 v129, v129, v131, v138
	.loc	0 2083 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v138, null, v131, v131, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v205, v138
	v_fma_f32 v207, -v138, v205, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v205, v207, v205
	v_div_scale_f32 v207, vcc_lo, v137, v131, v137
	v_mul_f32_e32 v208, v207, v205
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v209, -v138, v208, v207
	v_fmac_f32_e32 v208, v209, v205
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v138, -v138, v208, v207
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v138, v138, v205, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v137, v138, v131, v137
	.loc	0 2082 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cvt_pk_fp8_f32 v139.h, v129, v137
	.loc	0 2086 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v129, null, v131, v131, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v137, v129
	v_fma_f32 v138, -v129, v137, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v137, v138, v137
	v_div_scale_f32 v138, vcc_lo, v136, v131, v136
	v_mul_f32_e32 v205, v138, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v207, -v129, v205, v138
	v_fmac_f32_e32 v205, v207, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v129, -v129, v205, v138
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v129, v129, v137, v205
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v129, v129, v131, v136
	.loc	0 2086 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v136, null, v131, v131, v135
	v_rcp_f32_e32 v137, v136
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v138, -v136, v137, 1.0
	v_fmac_f32_e32 v137, v138, v137
	v_div_scale_f32 v138, vcc_lo, v135, v131, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v205, v138, v137
	v_fma_f32 v207, -v136, v205, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v205, v207, v137
	v_fma_f32 v136, -v136, v205, v138
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v136, v136, v137, v205
	v_mov_b32_e32 v205, v131
	v_div_fixup_f32 v135, v136, v131, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	0 2085 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cvt_pk_fp8_f32 v140.l, v129, v135
	.loc	0 2088 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v129, null, v131, v131, v134
	v_rcp_f32_e32 v135, v129
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v136, -v129, v135, 1.0
	v_fmac_f32_e32 v135, v136, v135
	v_div_scale_f32 v136, vcc_lo, v134, v131, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v137, v136, v135
	v_fma_f32 v138, -v129, v137, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v137, v138, v135
	v_fma_f32 v129, -v129, v137, v136
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v129, v129, v135, v137
	v_div_fixup_f32 v129, v129, v131, v134
	.loc	0 2088 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v134, null, v131, v131, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v135, v134
	v_fma_f32 v136, -v134, v135, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v135, v136, v135
	v_div_scale_f32 v136, vcc_lo, v133, v131, v133
	v_mul_f32_e32 v137, v136, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v138, -v134, v137, v136
	v_fmac_f32_e32 v137, v138, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v134, -v134, v137, v136
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v134, v134, v135, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v133, v134, v131, v133
	.loc	0 2087 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cvt_pk_fp8_f32 v140.h, v129, v133
	.loc	0 2093 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshl_add_u32 v129, s13, 12, v160
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b128 v[133:136], v129 offset:12288
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[133:134], v[139:140], v[121:128]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[135:136], v[139:140], v[113:120]
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b128 v[133:136], v129 offset:12800
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[133:134], v[139:140], v[105:112]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[135:136], v[139:140], v[97:104]
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b128 v[133:136], v129 offset:13312
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[133:134], v[139:140], v[89:96]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[135:136], v[139:140], v[81:88]
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b128 v[133:136], v129 offset:13824
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[133:134], v[139:140], v[73:80]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[135:136], v[139:140], v[65:72]
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b128 v[133:136], v129 offset:14336
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[133:134], v[139:140], v[57:64]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[135:136], v[139:140], v[49:56]
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b128 v[133:136], v129 offset:14848
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[133:134], v[139:140], v[41:48]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[135:136], v[139:140], v[33:40]
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b128 v[133:136], v129 offset:15360
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[133:134], v[139:140], v[25:32]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[135:136], v[139:140], v[17:24]
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b128 v[133:136], v129 offset:15872
	v_mov_b32_e32 v129, v132
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[133:134], v[139:140], v[9:16]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[135:136], v[139:140], v[1:8]
	v_mov_b32_e32 v207, v130
	.loc	0 1990 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_add_co_i32 s13, s13, 1
	.loc	0 1990 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s13, 3
	.loc	0 1990 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_scc1 .LBB5_14
.LBB5_45:                               ;   Parent Loop BB5_15 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_47 Depth 3
	.loc	0 1991 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1991:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s13, 1
	s_cselect_b32 s4, 16, 32
	s_cmp_lg_u32 s13, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s4, s4, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s4, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s5, s4
	.loc	0 1992 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1992:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_scc1 .LBB5_52
; %bb.46:                               ;   in Loop: Header=BB5_45 Depth=2
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	v_mov_b32_e32 v130, 0
	s_mov_b32 s4, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v131, v130 :: v_dual_mov_b32 v132, v130
	v_dual_mov_b32 v133, v130 :: v_dual_mov_b32 v134, v130
	v_dual_mov_b32 v135, v130 :: v_dual_mov_b32 v136, v130
	v_mov_b32_e32 v137, v130
.LBB5_47:                               ;   Parent Loop BB5_15 Depth=1
                                        ;     Parent Loop BB5_45 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	.loc	0 2007 48 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2007:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s14, s4, 6
	.loc	0 2007 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2007:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s14, s14, s13
	.loc	0 2006 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_add_u32 v140, s14, 9, v160
	.loc	0 2013 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_lshl_b32 s14, s4, 6
	.loc	0 2001 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_add_co_i32 s4, s4, 1
	.loc	0 2016 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v138, vcc_lo, v200, s14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v139, null, 0, v201, vcc_lo
	.loc	0 2008 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2008:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b128 v[208:211], v140
	.loc	0 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cmp_lg_u32 s4, 4
	.loc	0 2016 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_clause 0x1
	global_load_b64 v[212:213], v[138:139], off
	global_load_b64 v[214:215], v[138:139], off offset:16
	.loc	0 2019 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2019:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[130:137], v[208:209], v[212:213], v[130:137]
	.loc	0 2016 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_clause 0x1
	global_load_b64 v[212:213], v[138:139], off offset:32
	global_load_b64 v[138:139], v[138:139], off offset:48
	.loc	0 2022 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[130:137], v[210:211], v[214:215], v[130:137]
	.loc	0 2008 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2008:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b128 v[208:211], v140 offset:1536
	.loc	0 2025 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2019 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2019:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[130:137], v[208:209], v[212:213], v[130:137]
	.loc	0 2022 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[130:137], v[210:211], v[138:139], v[130:137]
	.loc	0 2001 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_scc1 .LBB5_47
; %bb.48:                               ;   in Loop: Header=BB5_45 Depth=2
	.loc	0 0 17 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:17
	v_lshl_add_u32 v208, s13, 5, v161
	.loc	0 2028 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_lshl4_add_u32 s4, s13, s7
	v_mov_b32_e32 v209, 0xff800000
	.loc	0 2030 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s14, s4, 15
	v_or_b32_e32 v211, s4, v154
	.loc	0 2034 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2034:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_b96 v[138:140], v208 offset:36866
	ds_load_u16_d16 v210, v208 offset:36878
	.loc	0 2030 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s14, s21
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s12, s4
	.loc	0 2039 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2039:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_and_saveexec_b32 s14, s0
	s_cbranch_execz .LBB5_50
; %bb.49:                               ;   in Loop: Header=BB5_45 Depth=2
	.loc	0 2034 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2034:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_u16_d16 v209, v208 offset:36864
	v_mul_f32_e32 v130, v151, v130
	v_cmp_le_i32_e32 vcc_lo, v211, v206
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s4, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v130, v209, v130, neg(0) op_sel_hi:[1,0,0]
	.loc	0 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v209, 0xff800000, v130, vcc_lo
.LBB5_50:                               ;   in Loop: Header=BB5_45 Depth=2
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	v_mul_f32_e32 v130, v151, v131
	v_or_b32_e32 v131, 2, v211
	v_cmp_ge_i32_e32 vcc_lo, v211, v206
	s_xor_b32 s14, s4, -1
	v_mul_f32_e32 v132, v151, v132
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s4, v131, v206
	v_or_b32_e32 v131, 3, v211
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s15, s14, vcc_lo
	.loc	0 2039 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2039:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s15, s1, s15
	s_and_b32 s4, s14, s4
	v_cmp_gt_i32_e32 vcc_lo, v131, v206
	v_mul_f32_e32 v131, v151, v133
	s_wait_dscnt 0x1
	v_fma_mix_f32 v130, v138, v130, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v212, v130, 0xff800000, s15
	v_fma_mix_f32 v130, v138, v132, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_or_b32_e32 v132, 4, v211
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v133, v130, 0xff800000, s4
	v_fma_mix_f32 v130, v139, v131, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v131, v151, v134
	s_and_b32 s4, s14, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v132, v206
	v_or_b32_e32 v132, 5, v211
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v134, v130, 0xff800000, s4
	v_fma_mix_f32 v130, v139, v131, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v131, v151, v135
	s_and_b32 s4, s14, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v132, v206
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	v_or_b32_e32 v132, 7, v211
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v135, v130, 0xff800000, s4
	v_fma_mix_f32 v130, v140, v131, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v131, 6, v211
	s_and_b32 s4, s14, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, v130, 0xff800000, s4
	v_cmp_gt_i32_e32 vcc_lo, v131, v206
	v_mul_f32_e32 v130, v151, v136
	v_cmp_gt_i32_e64 s4, v132, v206
	v_mul_f32_e32 v131, v151, v137
.Ltmp440:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2047:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v132, v209, 0xff800000, v212
	s_and_b32 s15, s14, vcc_lo
	v_fma_mix_f32 v130, v140, v130, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_and_b32 s4, s14, s4
	s_wait_dscnt 0x0
	v_fma_mix_f32 v131, v210, v131, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v132, v132, v133, v134
.Ltmp441:
	.loc	0 2039 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2039:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s14, s1, s15
	s_or_b32 s4, s1, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v136, v130, 0xff800000, s14
	v_cndmask_b32_e64 v131, v131, 0xff800000, s4
.Ltmp442:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2047:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v130, v132, v135, v138
.Ltmp443:
	.loc	0 2073 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2073:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp444:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2047:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v130, v130, v136, v131
.Ltmp445:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2048:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_mov_b32_e32 v132, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v132, v132, s6, 0xfedcba98
.Ltmp446:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v130, v207, v130, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v136, v136, v130 :: v_dual_add_nc_u32 v139, 0x9060, v208
	v_dual_sub_f32 v131, v131, v130 :: v_dual_sub_f32 v132, v209, v130
	v_dual_sub_f32 v137, v212, v130 :: v_dual_sub_f32 v138, v138, v130
	v_dual_mul_f32 v136, 0x3fb8aa3b, v136 :: v_dual_mul_f32 v131, 0x3fb8aa3b, v131
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v132, 0x3fb8aa3b, v132
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v130
	v_sub_f32_e32 v135, v135, v130
	v_exp_f32_e32 v136, v136
	v_exp_f32_e32 v140, v131
	v_sub_f32_e32 v131, v133, v130
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_mul_f32 v138, 0x3fb8aa3b, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v210, 0x3fb8aa3b, v131
	v_exp_f32_e32 v137, v137
.Ltmp447:
	.loc	0 2059 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v213, v136, 0, vcc_lo
	v_exp_f32_e32 v138, v138
	v_cndmask_b32_e64 v137, v137, 0, vcc_lo
	v_sub_f32_e32 v133, v134, v130
	v_exp_f32_e32 v134, v132
	.loc	0 2055 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_2addr_b32 v[131:132], v139 offset1:1
	v_add_nc_u32_e32 v139, 0x9068, v208
	.loc	0 2059 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cndmask_b32_e64 v211, v138, 0, vcc_lo
	.loc	0 2055 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	ds_load_2addr_b32 v[208:209], v139 offset1:1
	v_exp_f32_e32 v139, v210
	.loc	0 2059 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cndmask_b32_e64 v134, v134, 0, vcc_lo
	v_cndmask_b32_e64 v210, v140, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v212, v139, 0, vcc_lo
	.loc	0 2063 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x1
	v_fma_mix_f32 v140, v131, v134, neg(0) op_sel_hi:[1,0,0]
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_f32_e32 v134, v134, v137
	v_mul_f32_e32 v133, 0x3fb8aa3b, v133
	.loc	0 2063 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_mix_f32 v139, v131, v137, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v138, v132, v212, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_f32_e32 v134, v212, v134
	v_exp_f32_e32 v133, v133
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 2059 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cndmask_b32_e64 v133, v133, 0, vcc_lo
	v_mul_f32_e32 v135, 0x3fb8aa3b, v135
	.loc	0 2063 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_mix_f32 v137, v132, v133, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v135, v135
.Ltmp448:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v132, v140, 0, v139
	v_max3_num_f32 v132, v132, v138, v137
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
.Ltmp449:
	.loc	0 2059 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cndmask_b32_e64 v131, v135, 0, vcc_lo
	.loc	0 2063 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v135, v208, v211, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v136, v208, v131, neg(0) op_sel_hi:[1,0,0]
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_f32_e32 v208, v133, v134
	.loc	0 2063 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_mix_f32 v134, v209, v213, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v133, v209, v210, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
.Ltmp450:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v132, v132, v136, v135
.Ltmp451:
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_add_f32_e32 v131, v131, v208
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp452:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max3_num_f32 v209, v132, v134, v133
.Ltmp453:
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_add_f32 v131, v211, v131 :: v_dual_mov_b32 v208, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v131, v213, v131
.Ltmp454:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2071:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_permlanex16_b32 v208, v208, s6, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp455:
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_add_f32 v132, v210, v131 :: v_dual_max_num_f32 v131, v208, v208
.Ltmp456:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2065:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_dual_mov_b32 v208, v132 :: v_dual_max_num_f32 v209, v209, v131
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_permlanex16_b32 v208, v208, s6, 0xfedcba98
	v_mov_b32_e32 v131, v205
.Ltmp457:
	.loc	0 2073 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2073:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmpx_lt_f32_e32 0, v209
	s_cbranch_execz .LBB5_44
; %bb.51:                               ;   in Loop: Header=BB5_45 Depth=2
	.loc	0 2074 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v131, null, 0x43e00000, 0x43e00000, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v210, v131
	v_fma_f32 v211, -v131, v210, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v210, v211, v210
	v_div_scale_f32 v211, vcc_lo, v209, 0x43e00000, v209
	v_mul_f32_e32 v212, v211, v210
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v213, -v131, v212, v211
	v_fmac_f32_e32 v212, v213, v210
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v131, -v131, v212, v211
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v131, v131, v210, v212
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v131, v131, 0x43e00000, v209
.Ltmp458:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ] ]
	v_max_num_f32_e32 v131, 0x1f800000, v131
	s_branch .LBB5_44
.Ltmp459:
.LBB5_52:                               ;   in Loop: Header=BB5_45 Depth=2
	.loc	2 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v130, v207
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v207, v130
	.loc	0 1990 40 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_add_co_i32 s13, s13, 1
	.loc	0 1990 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s13, 3
	.loc	0 1990 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_cbranch_scc0 .LBB5_45
	s_branch .LBB5_14
.LBB5_53:
	.loc	0 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	v_mov_b32_e32 v129, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v205, 1.0 :: v_dual_mov_b32 v136, v129
	v_dual_mov_b32 v130, v129 :: v_dual_mov_b32 v131, v129
	v_dual_mov_b32 v132, v129 :: v_dual_mov_b32 v133, v129
	v_dual_mov_b32 v134, v129 :: v_dual_mov_b32 v135, v129
	v_dual_mov_b32 v121, v129 :: v_dual_mov_b32 v122, v130
	v_dual_mov_b32 v113, v129 :: v_dual_mov_b32 v114, v130
	v_dual_mov_b32 v105, v129 :: v_dual_mov_b32 v106, v130
	v_dual_mov_b32 v97, v129 :: v_dual_mov_b32 v98, v130
	v_dual_mov_b32 v89, v129 :: v_dual_mov_b32 v90, v130
	v_dual_mov_b32 v81, v129 :: v_dual_mov_b32 v82, v130
	v_dual_mov_b32 v73, v129 :: v_dual_mov_b32 v74, v130
	v_dual_mov_b32 v65, v129 :: v_dual_mov_b32 v66, v130
	v_dual_mov_b32 v57, v129 :: v_dual_mov_b32 v58, v130
	v_dual_mov_b32 v49, v129 :: v_dual_mov_b32 v50, v130
	v_dual_mov_b32 v41, v129 :: v_dual_mov_b32 v42, v130
	v_dual_mov_b32 v33, v129 :: v_dual_mov_b32 v34, v130
	v_dual_mov_b32 v25, v129 :: v_dual_mov_b32 v26, v130
	v_dual_mov_b32 v17, v129 :: v_dual_mov_b32 v18, v130
	v_dual_mov_b32 v9, v129 :: v_dual_mov_b32 v10, v130
	v_dual_mov_b32 v1, v129 :: v_dual_mov_b32 v2, v130
	v_dual_mov_b32 v123, v131 :: v_dual_mov_b32 v124, v132
	v_dual_mov_b32 v125, v133 :: v_dual_mov_b32 v126, v134
	v_dual_mov_b32 v127, v135 :: v_dual_mov_b32 v128, v136
	v_dual_mov_b32 v115, v131 :: v_dual_mov_b32 v116, v132
	v_dual_mov_b32 v117, v133 :: v_dual_mov_b32 v118, v134
	v_dual_mov_b32 v119, v135 :: v_dual_mov_b32 v120, v136
	v_dual_mov_b32 v107, v131 :: v_dual_mov_b32 v108, v132
	v_dual_mov_b32 v109, v133 :: v_dual_mov_b32 v110, v134
	v_dual_mov_b32 v111, v135 :: v_dual_mov_b32 v112, v136
	v_dual_mov_b32 v99, v131 :: v_dual_mov_b32 v100, v132
	v_dual_mov_b32 v101, v133 :: v_dual_mov_b32 v102, v134
	v_dual_mov_b32 v103, v135 :: v_dual_mov_b32 v104, v136
	v_dual_mov_b32 v91, v131 :: v_dual_mov_b32 v92, v132
	v_dual_mov_b32 v93, v133 :: v_dual_mov_b32 v94, v134
	v_dual_mov_b32 v95, v135 :: v_dual_mov_b32 v96, v136
	v_dual_mov_b32 v83, v131 :: v_dual_mov_b32 v84, v132
	v_dual_mov_b32 v85, v133 :: v_dual_mov_b32 v86, v134
	v_dual_mov_b32 v87, v135 :: v_dual_mov_b32 v88, v136
	v_dual_mov_b32 v75, v131 :: v_dual_mov_b32 v76, v132
	v_dual_mov_b32 v77, v133 :: v_dual_mov_b32 v78, v134
	v_dual_mov_b32 v79, v135 :: v_dual_mov_b32 v80, v136
	v_dual_mov_b32 v67, v131 :: v_dual_mov_b32 v68, v132
	v_dual_mov_b32 v69, v133 :: v_dual_mov_b32 v70, v134
	v_dual_mov_b32 v71, v135 :: v_dual_mov_b32 v72, v136
	v_dual_mov_b32 v59, v131 :: v_dual_mov_b32 v60, v132
	v_dual_mov_b32 v61, v133 :: v_dual_mov_b32 v62, v134
	v_dual_mov_b32 v63, v135 :: v_dual_mov_b32 v64, v136
	v_dual_mov_b32 v51, v131 :: v_dual_mov_b32 v52, v132
	v_dual_mov_b32 v53, v133 :: v_dual_mov_b32 v54, v134
	v_dual_mov_b32 v55, v135 :: v_dual_mov_b32 v56, v136
	v_dual_mov_b32 v43, v131 :: v_dual_mov_b32 v44, v132
	v_dual_mov_b32 v45, v133 :: v_dual_mov_b32 v46, v134
	v_dual_mov_b32 v47, v135 :: v_dual_mov_b32 v48, v136
	v_dual_mov_b32 v35, v131 :: v_dual_mov_b32 v36, v132
	v_dual_mov_b32 v37, v133 :: v_dual_mov_b32 v38, v134
	v_dual_mov_b32 v39, v135 :: v_dual_mov_b32 v40, v136
	v_dual_mov_b32 v27, v131 :: v_dual_mov_b32 v28, v132
	v_dual_mov_b32 v29, v133 :: v_dual_mov_b32 v30, v134
	v_dual_mov_b32 v31, v135 :: v_dual_mov_b32 v32, v136
	v_dual_mov_b32 v19, v131 :: v_dual_mov_b32 v20, v132
	v_dual_mov_b32 v21, v133 :: v_dual_mov_b32 v22, v134
	v_dual_mov_b32 v23, v135 :: v_dual_mov_b32 v24, v136
	v_dual_mov_b32 v11, v131 :: v_dual_mov_b32 v12, v132
	v_dual_mov_b32 v13, v133 :: v_dual_mov_b32 v14, v134
	v_dual_mov_b32 v15, v135 :: v_dual_mov_b32 v16, v136
	v_dual_mov_b32 v3, v131 :: v_dual_mov_b32 v4, v132
	v_dual_mov_b32 v5, v133 :: v_dual_mov_b32 v6, v134
	v_dual_mov_b32 v7, v135 :: v_dual_mov_b32 v8, v136
.LBB5_54:
	.loc	0 2116 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2116:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_56
; %bb.55:
	.loc	0 2117 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_scale_f32 v0, null, v129, v129, 1.0
	v_div_scale_f32 v132, vcc_lo, 1.0, v129, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v130, v0
	v_fma_f32 v131, -v0, v130, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v130, v131, v130
	v_mul_f32_e32 v131, v132, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v133, -v0, v131, v132
	v_fmac_f32_e32 v131, v133, v130
	.loc	0 2119 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2119:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mul_lo_u32 v133, 0x1800, v142
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	0 2117 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_fma_f32 v0, -v0, v131, v132
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v0, v0, v130, v131
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	.loc	0 2119 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2119:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshl_add_u32 v130, v141, 8, v133
	.loc	0 2117 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_cmp_lt_f32_e32 vcc_lo, 0, v129
	.loc	0 2125 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_mov_b32_e32 v131, 0
	.loc	0 2117 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_div_fixup_f32 v0, v0, v129, 1.0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 2121 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:18 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshl_or_b32 v130, v150, 3, v130
	.loc	0 2117 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 2125 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_lshlrev_b64_e32 v[129:130], 2, v[130:131]
	v_mul_f32_e32 v131, v205, v0
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v129, vcc_lo, s16, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, s17, v130, vcc_lo
	.loc	0 2125 56 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mul_f32 v121, v121, v131 :: v_dual_mul_f32 v122, v122, v131
	v_dual_mul_f32 v123, v123, v131 :: v_dual_mul_f32 v124, v124, v131
	v_dual_mul_f32 v89, v89, v131 :: v_dual_mul_f32 v90, v90, v131
	v_dual_mul_f32 v91, v91, v131 :: v_dual_mul_f32 v92, v92, v131
	v_dual_mul_f32 v65, v65, v131 :: v_dual_mul_f32 v66, v66, v131
	v_dual_mul_f32 v67, v67, v131 :: v_dual_mul_f32 v68, v68, v131
	v_dual_mul_f32 v41, v41, v131 :: v_dual_mul_f32 v42, v42, v131
	v_dual_mul_f32 v43, v43, v131 :: v_dual_mul_f32 v44, v44, v131
	v_dual_mul_f32 v17, v17, v131 :: v_dual_mul_f32 v18, v18, v131
	v_dual_mul_f32 v19, v19, v131 :: v_dual_mul_f32 v20, v20, v131
	v_dual_mul_f32 v125, v125, v131 :: v_dual_mul_f32 v126, v126, v131
	v_dual_mul_f32 v127, v127, v131 :: v_dual_mul_f32 v128, v128, v131
	v_dual_mul_f32 v93, v93, v131 :: v_dual_mul_f32 v94, v94, v131
	v_dual_mul_f32 v95, v95, v131 :: v_dual_mul_f32 v96, v96, v131
	v_dual_mul_f32 v69, v69, v131 :: v_dual_mul_f32 v70, v70, v131
	v_dual_mul_f32 v71, v71, v131 :: v_dual_mul_f32 v72, v72, v131
	v_dual_mul_f32 v45, v45, v131 :: v_dual_mul_f32 v46, v46, v131
	v_dual_mul_f32 v47, v47, v131 :: v_dual_mul_f32 v48, v48, v131
	v_dual_mul_f32 v21, v21, v131 :: v_dual_mul_f32 v22, v22, v131
	v_dual_mul_f32 v23, v23, v131 :: v_dual_mul_f32 v24, v24, v131
	v_dual_mul_f32 v113, v113, v131 :: v_dual_mul_f32 v114, v114, v131
	v_dual_mul_f32 v115, v115, v131 :: v_dual_mul_f32 v116, v116, v131
	v_dual_mul_f32 v81, v81, v131 :: v_dual_mul_f32 v82, v82, v131
	v_dual_mul_f32 v83, v83, v131 :: v_dual_mul_f32 v84, v84, v131
	v_dual_mul_f32 v57, v57, v131 :: v_dual_mul_f32 v58, v58, v131
	v_dual_mul_f32 v59, v59, v131 :: v_dual_mul_f32 v60, v60, v131
	v_dual_mul_f32 v33, v33, v131 :: v_dual_mul_f32 v34, v34, v131
	v_dual_mul_f32 v35, v35, v131 :: v_dual_mul_f32 v36, v36, v131
	v_dual_mul_f32 v9, v9, v131 :: v_dual_mul_f32 v10, v10, v131
	v_dual_mul_f32 v11, v11, v131 :: v_dual_mul_f32 v12, v12, v131
	v_dual_mul_f32 v117, v117, v131 :: v_dual_mul_f32 v118, v118, v131
	v_dual_mul_f32 v119, v119, v131 :: v_dual_mul_f32 v120, v120, v131
	v_dual_mul_f32 v85, v85, v131 :: v_dual_mul_f32 v86, v86, v131
	v_dual_mul_f32 v87, v87, v131 :: v_dual_mul_f32 v88, v88, v131
	v_dual_mul_f32 v61, v61, v131 :: v_dual_mul_f32 v62, v62, v131
	v_dual_mul_f32 v63, v63, v131 :: v_dual_mul_f32 v64, v64, v131
	v_dual_mul_f32 v37, v37, v131 :: v_dual_mul_f32 v38, v38, v131
	v_dual_mul_f32 v39, v39, v131 :: v_dual_mul_f32 v40, v40, v131
	v_dual_mul_f32 v13, v13, v131 :: v_dual_mul_f32 v14, v14, v131
	v_dual_mul_f32 v15, v15, v131 :: v_dual_mul_f32 v16, v16, v131
	v_dual_mul_f32 v105, v105, v131 :: v_dual_mul_f32 v106, v106, v131
	v_dual_mul_f32 v107, v107, v131 :: v_dual_mul_f32 v108, v108, v131
	v_dual_mul_f32 v73, v73, v131 :: v_dual_mul_f32 v74, v74, v131
	v_dual_mul_f32 v75, v75, v131 :: v_dual_mul_f32 v76, v76, v131
	v_dual_mul_f32 v49, v49, v131 :: v_dual_mul_f32 v50, v50, v131
	v_dual_mul_f32 v51, v51, v131 :: v_dual_mul_f32 v52, v52, v131
	v_dual_mul_f32 v25, v25, v131 :: v_dual_mul_f32 v26, v26, v131
	v_dual_mul_f32 v27, v27, v131 :: v_dual_mul_f32 v28, v28, v131
	v_dual_mul_f32 v0, v1, v131 :: v_dual_mul_f32 v1, v2, v131
	v_dual_mul_f32 v2, v3, v131 :: v_dual_mul_f32 v3, v4, v131
	v_dual_mul_f32 v109, v109, v131 :: v_dual_mul_f32 v110, v110, v131
	v_dual_mul_f32 v111, v111, v131 :: v_dual_mul_f32 v112, v112, v131
	v_dual_mul_f32 v97, v97, v131 :: v_dual_mul_f32 v98, v98, v131
	v_dual_mul_f32 v99, v99, v131 :: v_dual_mul_f32 v100, v100, v131
	v_dual_mul_f32 v101, v101, v131 :: v_dual_mul_f32 v102, v102, v131
	v_dual_mul_f32 v103, v103, v131 :: v_dual_mul_f32 v104, v104, v131
	.loc	0 2125 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_clause 0x7
	global_store_b128 v[129:130], v[121:124], off
	global_store_b128 v[129:130], v[125:128], off offset:16
	global_store_b128 v[129:130], v[113:116], off offset:64
	global_store_b128 v[129:130], v[117:120], off offset:80
	global_store_b128 v[129:130], v[105:108], off offset:128
	global_store_b128 v[129:130], v[109:112], off offset:144
	global_store_b128 v[129:130], v[97:100], off offset:192
	global_store_b128 v[129:130], v[101:104], off offset:208
	.loc	0 2125 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mul_f32 v77, v77, v131 :: v_dual_mul_f32 v78, v78, v131
	v_dual_mul_f32 v79, v79, v131 :: v_dual_mul_f32 v80, v80, v131
	.loc	0 2125 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_clause 0x5
	global_store_b128 v[129:130], v[89:92], off offset:256
	global_store_b128 v[129:130], v[93:96], off offset:272
	global_store_b128 v[129:130], v[81:84], off offset:320
	global_store_b128 v[129:130], v[85:88], off offset:336
	global_store_b128 v[129:130], v[73:76], off offset:384
	global_store_b128 v[129:130], v[77:80], off offset:400
	.loc	0 2125 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mul_f32 v53, v53, v131 :: v_dual_mul_f32 v54, v54, v131
	v_dual_mul_f32 v55, v55, v131 :: v_dual_mul_f32 v56, v56, v131
	.loc	0 2125 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_clause 0x5
	global_store_b128 v[129:130], v[65:68], off offset:448
	global_store_b128 v[129:130], v[69:72], off offset:464
	global_store_b128 v[129:130], v[57:60], off offset:512
	global_store_b128 v[129:130], v[61:64], off offset:528
	global_store_b128 v[129:130], v[49:52], off offset:576
	global_store_b128 v[129:130], v[53:56], off offset:592
	.loc	0 2125 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mul_f32 v29, v29, v131 :: v_dual_mul_f32 v30, v30, v131
	v_dual_mul_f32 v31, v31, v131 :: v_dual_mul_f32 v32, v32, v131
	.loc	0 2125 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_clause 0x5
	global_store_b128 v[129:130], v[41:44], off offset:640
	global_store_b128 v[129:130], v[45:48], off offset:656
	global_store_b128 v[129:130], v[33:36], off offset:704
	global_store_b128 v[129:130], v[37:40], off offset:720
	global_store_b128 v[129:130], v[25:28], off offset:768
	global_store_b128 v[129:130], v[29:32], off offset:784
	.loc	0 2125 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	v_dual_mul_f32 v4, v5, v131 :: v_dual_mul_f32 v5, v6, v131
	v_dual_mul_f32 v6, v7, v131 :: v_dual_mul_f32 v7, v8, v131
	.loc	0 2125 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5 ]
	s_clause 0x5
	global_store_b128 v[129:130], v[17:20], off offset:832
	global_store_b128 v[129:130], v[21:24], off offset:848
	global_store_b128 v[129:130], v[9:12], off offset:896
	global_store_b128 v[129:130], v[13:16], off offset:912
	global_store_b128 v[129:130], v[0:3], off offset:960
	global_store_b128 v[129:130], v[4:7], off offset:976
.Ltmp460:
.LBB5_56:
	.loc	0 2190 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2190:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp461:
.Lfunc_end5:
	.size	attention_fp8_e4m3_fa2_gqa_packet_gfx1201, .Lfunc_end5-attention_fp8_e4m3_fa2_gqa_packet_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_gfx1201
		.amdhsa_group_segment_fixed_size 0
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 216
		.amdhsa_next_free_sgpr 27
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-attention_fp8_e4m3_fa2_gqa_packet_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_vgpr, 216
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.numbered_sgpr, 27
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 12456
; TotalNumSgprs: 29
; NumVgprs: 216
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 26
; NumSGPRsForWavesPerEU: 29
; NumVGPRsForWavesPerEU: 216
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
.Lfunc_begin6:
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.loc	0 2206 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2206:17
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x28
	s_load_b64 s[16:17], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s12, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s13, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2206 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2206:23
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s14, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s4, s17, -9
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s4, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB6_56
; %bb.1:
	.loc	0 0 23                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2211 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2211:14
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB6_56
; %bb.2:
	.loc	0 2213 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2213:30
	s_mul_i32 s14, ttmp9, 0x180
	.loc	0 2214 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:23
	s_mul_i32 s2, s15, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2214 18 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:18
	s_cmp_ge_i32 s14, s2
	s_cbranch_scc1 .LBB6_56
; %bb.3:
	.loc	0 0 18                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:18
	s_lshr_b32 s12, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2217 15 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:15
	s_cmp_ge_i32 s12, s17
	s_cbranch_scc1 .LBB6_56
; %bb.4:
.Ltmp462:
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mov_b32 v149, s16 :: v_dual_and_b32 v4, 15, v0
	.loc	0 1767 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1767:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshrrev_b32_e32 v5, 5, v0
.Ltmp463:
	.loc	0 2206 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2206:17
	s_clause 0x1
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b64 s[18:19], s[0:1], 0x20
.Ltmp464:
	.loc	0 1781 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mul_i32 s1, s3, 6
	.loc	0 1778 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_or_b32_e32 v1, s14, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1778 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshl_add_u32 v2, v5, 4, v1
	.loc	0 1779 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_hi_i32 v1, 0x2aaaaaab, v2
	.loc	0 1782 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1782:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmp_gt_i32_e64 s0, s2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1779 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshrrev_b32_e32 v3, 31, v1
	v_add_nc_u32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1781 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_lo_u32 v3, v1, 6
	v_mul_lo_u32 v6, v1, 24
	v_sub_nc_u32_e32 v3, v2, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v142, v3, s1, v6
	.loc	0 1838 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1838:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB6_6
; %bb.5:
	.loc	0 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_mov_b32_e32 v143, 0
	s_mul_i32 s20, s15, 0x1800
	s_mov_b32 s21, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[20:21], s[4:5], s[20:21]
	v_lshlrev_b64_e32 v[2:3], 2, v[142:143]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s20, v2
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	.loc	0 1841 23 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1841:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	global_load_b32 v2, v[2:3], off
	.loc	0 1983 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1983:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v149, s16, v2
.LBB6_6:
	.loc	0 0 47 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:47
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 1768 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1768:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_and_b32_e32 v6, 31, v0
	v_mov_b32_e32 v8, -1
	v_bfrev_b32_e32 v7, -2
	.loc	0 1849 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1849:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1847 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1847:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshl_or_b32 v2, ttmp9, 6, v6
	v_ashrrev_i32_e32 v3, 31, v2
	.loc	0 1849 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1849:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmpx_gt_i32_e64 s15, v2
	s_cbranch_execz .LBB6_8
; %bb.7:
	.loc	0 0 13 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[2:3]
	s_wait_kmcnt 0x0
	v_add_co_u32 v7, vcc_lo, s18, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s19, v8, vcc_lo
	.loc	0 1850 23 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	global_load_b32 v7, v[7:8], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v8, v7
.LBB6_8:
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 1848 25 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1848:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_or_b32_e32 v9, 32, v2
	.loc	0 1854 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v9
	s_cbranch_execz .LBB6_10
; %bb.9:
	.loc	0 0 13 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s18, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s19, v3, vcc_lo
	.loc	0 1855 23 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	global_load_b32 v2, v[2:3], off offset:128
.Ltmp465:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:15 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_loadcnt 0x0
	v_max_i32_e32 v8, v8, v2
.Ltmp466:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:15 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_min_i32_e32 v7, v7, v2
.Ltmp467:
.LBB6_10:
	.loc	2 0 10 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:10
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
.Ltmp468:
	v_mbcnt_lo_u32_b32 v2, -1, 0
.Ltmp469:
	.loc	0 2219 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2219:42
	s_cvt_f32_u32 s1, s17
	s_add_co_i32 s13, s17, 0x1ff
.Ltmp470:
	.loc	0 1770 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1770:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshrrev_b32_e32 v150, 4, v6
.Ltmp471:
	.loc	0 2219 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2219:42
	s_and_b32 s13, s13, 0xffff
.Ltmp472:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_xor_b32_e32 v3, 16, v2
.Ltmp473:
	.loc	1 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_xor_b32_e32 v10, 8, v2
.Ltmp474:
	.loc	0 2219 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2219:42
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s15, s1
	s_cvt_f32_u32 s13, s13
	s_mov_b32 s21, 0
.Ltmp475:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_1)
.Ltmp476:
	.loc	0 2219 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2219:42
	s_mul_f32 s15, s13, s15
.Ltmp477:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v2, v3, vcc_lo
.Ltmp478:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
.Ltmp479:
	.loc	0 2219 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2219:42
	s_wait_alu depctr_sa_sdst(0)
	s_trunc_f32 s15, s15
.Ltmp480:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v10, v2, v10 :: v_dual_lshlrev_b32 v3, 2, v3
.Ltmp481:
	.loc	0 2219 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2219:42
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s16, s15, 0x80000000
	s_cvt_u32_f32 s15, s15
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s13, s16, s1
.Ltmp482:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_lshlrev_b32_e32 v10, 2, v10
.Ltmp483:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v9, v3, v8
.Ltmp484:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v3, v3, v7
.Ltmp485:
	.loc	0 2219 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2219:42
	s_bitset0_b32 s13, 31
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_f32 s13, s1
	s_add_co_ci_u32 s1, s15, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	.loc	0 2220 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2220:33
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s12, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2221 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2221:38
	s_add_co_i32 s1, s13, s1
	.loc	0 2220 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2220:39
	s_lshl_b32 s13, s13, 6
	.loc	0 2221 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2221:45
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s1, s1, 6
.Ltmp486:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v8, v8, v9
.Ltmp487:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v7, v3
.Ltmp488:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v9, v10, v8
.Ltmp489:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v7, v10, v3
.Ltmp490:
	.loc	1 523 20 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_xor_b32_e32 v10, 4, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v2, v10, vcc_lo
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_lshlrev_b32_e32 v10, 2, v10
.Ltmp491:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v8, v8, v9
.Ltmp492:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v7
.Ltmp493:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v9, v10, v8
.Ltmp494:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1866:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v7, v10, v3
.Ltmp495:
	.loc	1 523 20 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_xor_b32_e32 v10, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v2, v10, vcc_lo
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_lshlrev_b32_e32 v151, 2, v10
.Ltmp496:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_xor_b32_e32 v10, 1, v2
.Ltmp497:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v8, v8, v9
.Ltmp498:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1866:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v7
.Ltmp499:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v2, v10, vcc_lo
.Ltmp500:
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v9, v151, v8
.Ltmp501:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1867:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v7, v151, v3
.Ltmp502:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_lshlrev_b32_e32 v152, 2, v2
.Ltmp503:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v8, v8, v9
.Ltmp504:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1867:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v7
.Ltmp505:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v2, v152, v8
.Ltmp506:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1868:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v7, v152, v3
.Ltmp507:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v8, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
.Ltmp508:
	.loc	0 1869 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_readfirstlane_b32 s15, v2
.Ltmp509:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1868:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v2, v3, v7
.Ltmp510:
	.loc	0 1871 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_add_co_i32 s15, s15, 1
	.loc	0 1870 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1870:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_readfirstlane_b32 s16, v2
.Ltmp511:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_alu depctr_sa_sdst(0)
	s_min_i32 s22, s15, s1
	s_delay_alu instid0(SALU_CYCLE_1)
.Ltmp512:
	.loc	0 1874 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cmp_ge_i32 s13, s22
	.loc	0 1874 5 is_stmt 0              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_scc1 .LBB6_51
; %bb.11:
	.loc	0 0 5                           ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_mul_lo_u16 v2.l, 0x56, v5.l
	.loc	0 1774 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1774:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshl_add_u32 v3, v5, 11, 0
	v_mul_u32_u24_e32 v7, 0x2ab, v0
	v_lshlrev_b32_e32 v155, 3, v5
	v_mov_b16_e32 v2.h, 0
	v_lshrrev_b16 v2.l, 8, v2.l
	.loc	0 1874 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mov_b32 v154, 0 :: v_dual_lshlrev_b32 v153, 3, v150
	.loc	0 1872 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1872:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_addk_co_i32 s14, 0x17f
	v_bfe_u32 v10, v0, 5, 1
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_lo_u16 v2.l, v2.l, 3
	.loc	0 1872 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1872:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s14, s2
	v_cmp_gt_u32_e64 s1, 0xc0, v0
	s_cselect_b32 s23, -1, 0
	s_lshl_b32 s20, s3, 8
	v_sub_nc_u16 v5.l, v5.l, v2.l
	v_mov_b16_e32 v2.l, v7.h
	v_and_b32_e32 v7, 3, v0
	.loc	0 1874 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[14:15], s[6:7], s[20:21]
	s_add_nc_u64 s[24:25], s[8:9], s[20:21]
	v_and_b32_e32 v5, 0xff, v5
	v_lshl_or_b32 v8, v2, 5, v153
	s_lshl_b32 s20, s3, 1
	v_lshl_add_u32 v160, v0, 1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[8:9], s[20:21]
	v_mad_u32_u24 v2, v2, 3, v5
	v_lshl_or_b32 v156, v5, 4, v4
	v_lshrrev_b32_e32 v4, 2, v4
	v_add_co_u32 v143, s2, s14, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v12, v2, 9, 0
	v_dual_mov_b32 v5, 0x6020400 :: v_dual_and_b32 v2, 1, v0
	v_lshl_or_b32 v4, v7, 6, v4
	v_dual_mov_b32 v9, 0x5040100 :: v_dual_lshlrev_b32 v8, 3, v7
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	v_ashrrev_i32_e32 v2, 31, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v144, null, s15, 0, s2
	v_or_b32_e32 v14, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v157, 0x3070105, v5, vcc_lo
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_cmp_gt_u32_e32 vcc_lo, 2, v7
	v_lshlrev_b32_e32 v11, 4, v6
	v_lshlrev_b32_e32 v6, 3, v6
	v_lshlrev_b32_e32 v7, 5, v7
	v_xor_b32_e32 v14, v14, v8
	v_add_co_u32 v147, s3, s18, v1
	v_and_b32_e32 v1, 16, v0
	v_add_co_u32 v145, s2, s24, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v158, 0x3020706, v9, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, v150, v10
	v_add_nc_u32_e32 v161, 0, v1
	v_lshlrev_b32_e32 v1, 2, v4
	v_add_nc_u32_e32 v162, v3, v6
	v_xad_u32 v163, 0x120, v6, v3
	v_xad_u32 v164, 0x124, v6, v3
	v_xad_u32 v165, 0x240, v6, v3
	v_xad_u32 v166, 0x244, v6, v3
	v_xad_u32 v167, 0x360, v6, v3
	v_xad_u32 v168, 0x364, v6, v3
	v_xad_u32 v169, 0x520, v6, v3
	v_xad_u32 v170, 0x524, v6, v3
	v_xad_u32 v171, 0x640, v6, v3
	v_xad_u32 v172, 0x644, v6, v3
	v_xad_u32 v173, 0x760, v6, v3
	v_xad_u32 v174, 0x764, v6, v3
	v_add3_u32 v175, v3, v1, v7
	v_or_b32_e32 v6, 8, v4
	v_or_b32_e32 v7, 0x108, v4
	v_or_b32_e32 v9, 12, v4
	v_or_b32_e32 v10, 0x10c, v4
	v_lshl_add_u32 v184, v14, 2, v3
	v_xor_b32_e32 v6, v6, v8
	v_xor_b32_e32 v7, v7, v8
	v_xor_b32_e32 v9, v9, v8
	v_xor_b32_e32 v10, v10, v8
	v_or_b32_e32 v14, 0x128, v4
	v_lshl_add_u32 v176, v6, 2, v3
	v_lshl_add_u32 v177, v7, 2, v3
	v_lshl_add_u32 v178, v9, 2, v3
	v_lshl_add_u32 v179, v10, 2, v3
	v_or_b32_e32 v6, 16, v4
	v_or_b32_e32 v7, 0x110, v4
	v_or_b32_e32 v9, 20, v4
	v_or_b32_e32 v10, 0x114, v4
	v_xor_b32_e32 v14, v14, v8
	v_xor_b32_e32 v6, v6, v8
	v_xor_b32_e32 v7, v7, v8
	v_xor_b32_e32 v9, v9, v8
	v_xor_b32_e32 v10, v10, v8
	v_lshl_add_u32 v189, v14, 2, v3
	v_lshl_add_u32 v180, v6, 2, v3
	v_lshl_add_u32 v181, v7, 2, v3
	v_lshl_add_u32 v182, v9, 2, v3
	v_lshl_add_u32 v183, v10, 2, v3
	v_or_b32_e32 v6, 0x118, v4
	v_or_b32_e32 v7, 28, v4
	v_or_b32_e32 v9, 0x11c, v4
	v_or_b32_e32 v10, 40, v4
	v_or_b32_e32 v14, 52, v4
	v_xor_b32_e32 v6, v6, v8
	v_xor_b32_e32 v7, v7, v8
	v_xor_b32_e32 v9, v9, v8
	v_xor_b32_e32 v10, v10, v8
	v_xor_b32_e32 v14, v14, v8
	v_lshl_add_u32 v185, v6, 2, v3
	v_lshl_add_u32 v186, v7, 2, v3
	v_lshl_add_u32 v187, v9, 2, v3
	v_lshl_add_u32 v188, v10, 2, v3
	v_or_b32_e32 v6, 44, v4
	v_or_b32_e32 v7, 0x12c, v4
	v_or_b32_e32 v9, 48, v4
	v_or_b32_e32 v10, 0x130, v4
	v_lshl_add_u32 v194, v14, 2, v3
	v_xor_b32_e32 v6, v6, v8
	v_xor_b32_e32 v7, v7, v8
	v_xor_b32_e32 v9, v9, v8
	v_xor_b32_e32 v10, v10, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v148, null, s19, v2, s3
	v_lshl_add_u32 v190, v6, 2, v3
	v_lshl_add_u32 v191, v7, 2, v3
	v_lshl_add_u32 v192, v9, 2, v3
	v_lshl_add_u32 v193, v10, 2, v3
	v_or_b32_e32 v6, 0x134, v4
	v_or_b32_e32 v7, 56, v4
	v_or_b32_e32 v9, 0x138, v4
	v_or_b32_e32 v10, 60, v4
	v_or_b32_e32 v4, 0x13c, v4
	v_xor_b32_e32 v6, v6, v8
	v_xor_b32_e32 v7, v7, v8
	v_xor_b32_e32 v9, v9, v8
	v_xor_b32_e32 v10, v10, v8
	v_xor_b32_e32 v4, v4, v8
	v_lshl_add_u32 v195, v6, 2, v3
	v_lshl_add_u32 v196, v7, 2, v3
	v_lshl_add_u32 v197, v9, 2, v3
	v_lshl_add_u32 v198, v10, 2, v3
	v_lshl_add_u32 v199, v4, 2, v3
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v5, 8, v142
	s_and_b32 s18, s1, vcc_lo
	v_add_nc_u32_e32 v159, 0, v11
	v_add_nc_u32_e32 v202, v12, v11
	v_mov_b32_e32 v4, v3
	.loc	0 1791 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1791:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cndmask_b32_e64 v5, 0, v5, s0
	v_dual_mov_b32 v6, v3 :: v_dual_mov_b32 v7, v3
	v_dual_mov_b32 v8, v3 :: v_dual_mov_b32 v9, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, s3, s4, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s5, 0, s3
	v_mov_b32_e32 v10, v3
	v_add_co_u32 v200, vcc_lo, v2, v153
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v201, null, 0, v5, vcc_lo
	v_mov_b32_e32 v5, v3
	v_dual_mov_b32 v26, v10 :: v_dual_lshlrev_b32 v1, 6, v0
	v_dual_mov_b32 v34, v10 :: v_dual_mov_b32 v25, v9
	v_dual_mov_b32 v42, v10 :: v_dual_and_b32 v13, 0xf000, v1
	v_mov_b32_e32 v205, 1.0
	v_dual_mov_b32 v33, v9 :: v_dual_mov_b32 v50, v10
	v_dual_mov_b32 v41, v9 :: v_dual_mov_b32 v58, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v204, v159, v13
	v_dual_mov_b32 v18, v10 :: v_dual_mov_b32 v17, v9
	v_dual_mov_b32 v12, v4 :: v_dual_and_b32 v1, 0x3000, v1
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v49, v9
	v_dual_mov_b32 v66, v10 :: v_dual_mov_b32 v57, v9
	v_dual_mov_b32 v74, v10 :: v_dual_mov_b32 v65, v9
	v_dual_mov_b32 v82, v10 :: v_dual_mov_b32 v73, v9
	v_dual_mov_b32 v90, v10 :: v_dual_mov_b32 v81, v9
	v_dual_mov_b32 v98, v10 :: v_dual_mov_b32 v89, v9
	v_dual_mov_b32 v106, v10 :: v_dual_mov_b32 v97, v9
	v_dual_mov_b32 v114, v10 :: v_dual_mov_b32 v105, v9
	v_dual_mov_b32 v122, v10 :: v_dual_mov_b32 v113, v9
	v_mov_b32_e32 v130, v10
	v_add_co_ci_u32_e64 v146, null, s25, 0, s2
	v_cmp_gt_u32_e64 s2, 48, v0
	v_dual_mov_b32 v207, 0xff800000 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v20, v4 :: v_dual_add_nc_u32 v203, v159, v1
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v7
	v_mov_b32_e32 v32, v8
	v_dual_mov_b32 v14, v6 :: v_dual_mov_b32 v13, v5
	v_dual_mov_b32 v30, v6 :: v_dual_mov_b32 v11, v3
	v_dual_mov_b32 v28, v4 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v19, v3
	v_dual_mov_b32 v36, v4 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v46, v6 :: v_dual_mov_b32 v27, v3
	v_dual_mov_b32 v44, v4 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v54, v6 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v52, v4 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v45, v5
	v_dual_mov_b32 v62, v6 :: v_dual_mov_b32 v43, v3
	v_dual_mov_b32 v60, v4 :: v_dual_mov_b32 v55, v7
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v53, v5
	v_dual_mov_b32 v70, v6 :: v_dual_mov_b32 v51, v3
	v_dual_mov_b32 v68, v4 :: v_dual_mov_b32 v63, v7
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v61, v5
	v_dual_mov_b32 v78, v6 :: v_dual_mov_b32 v59, v3
	v_dual_mov_b32 v76, v4 :: v_dual_mov_b32 v71, v7
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v69, v5
	v_dual_mov_b32 v86, v6 :: v_dual_mov_b32 v67, v3
	v_dual_mov_b32 v84, v4 :: v_dual_mov_b32 v79, v7
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v77, v5
	v_dual_mov_b32 v94, v6 :: v_dual_mov_b32 v75, v3
	v_dual_mov_b32 v92, v4 :: v_dual_mov_b32 v87, v7
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v85, v5
	v_dual_mov_b32 v102, v6 :: v_dual_mov_b32 v83, v3
	v_dual_mov_b32 v100, v4 :: v_dual_mov_b32 v95, v7
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v93, v5
	v_dual_mov_b32 v110, v6 :: v_dual_mov_b32 v91, v3
	v_dual_mov_b32 v108, v4 :: v_dual_mov_b32 v103, v7
	v_dual_mov_b32 v120, v8 :: v_dual_mov_b32 v101, v5
	v_dual_mov_b32 v118, v6 :: v_dual_mov_b32 v99, v3
	v_dual_mov_b32 v116, v4 :: v_dual_mov_b32 v111, v7
	v_dual_mov_b32 v128, v8 :: v_dual_mov_b32 v109, v5
	v_dual_mov_b32 v126, v6 :: v_dual_mov_b32 v107, v3
	v_dual_mov_b32 v124, v4 :: v_dual_mov_b32 v121, v9
	v_mov_b32_e32 v119, v7
	v_mov_b32_e32 v117, v5
	v_mov_b32_e32 v115, v3
	v_mov_b32_e32 v129, v9
	v_mov_b32_e32 v127, v7
	v_mov_b32_e32 v125, v5
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v2, v3
	.loc	0 1874 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_add_nc_u64 s[14:15], s[6:7], s[20:21]
	s_mov_b32 s7, 0x76543210
	s_branch .LBB6_13
.LBB6_12:                               ;   in Loop: Header=BB6_13 Depth=1
.Ltmp513:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2112:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2112:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ] ] ]
	s_barrier_signal -1
	v_mov_b32_e32 v207, v1
.Ltmp514:
	.loc	0 1874 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_add_co_i32 s13, s13, 48
	.loc	0 1874 39 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s13, s22
.Ltmp515:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2112:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ] ] ]
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2112:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp516:
	.loc	0 1874 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1874:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_scc1 .LBB6_52
.LBB6_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_43 Depth 2
                                        ;       Child Loop BB6_45 Depth 3
	.loc	0 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_dual_mov_b32 v134, 0 :: v_dual_add_nc_u32 v1, s13, v156
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v132, 0
	v_mov_b32_e32 v131, 0
	.loc	0 1887 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1887:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s22, v1
	s_cbranch_execz .LBB6_15
; %bb.14:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1894 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1894:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[133:134], null, 0x408, v1, v[143:144]
	s_clause 0x1
	global_load_b64 v[131:132], v[133:134], off
	global_load_b64 v[133:134], v[133:134], off offset:16
.LBB6_15:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1902 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v202, v[131:134]
	.loc	0 1905 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1905:18 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB6_33
; %bb.16:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 18 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:18
	v_dual_mov_b32 v134, 0 :: v_dual_add_nc_u32 v1, s13, v155
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v132, 0
	v_mov_b32_e32 v131, 0
	.loc	0 1910 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmpx_gt_i32_e64 s22, v1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_execz .LBB6_18
; %bb.17:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[131:132], null, 0x408, v1, v[145:146]
	global_load_b64 v[131:132], v[131:132], off
.LBB6_18:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_or_b32_e32 v135, 1, v1
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_nc_u32_e32 v136, 0x6000, v162
	.loc	0 1910 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v136, v131, v132 offset1:1
	.loc	0 1908 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmpx_gt_i32_e64 s22, v135
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_execz .LBB6_20
; %bb.19:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[131:132], null, 0x408, v135, v[145:146]
	global_load_b64 v[133:134], v[131:132], off
.LBB6_20:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_or_b32_e32 v137, 2, v1
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v136, 0
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1916 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v163, v133 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b32 v164, v134 offset:24576
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmpx_gt_i32_e64 s22, v137
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_execz .LBB6_22
; %bb.21:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[133:134], null, 0x408, v137, v[145:146]
	global_load_b64 v[135:136], v[133:134], off
.LBB6_22:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_or_b32_e32 v133, 3, v1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1916 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v165, v135 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b32 v166, v136 offset:24576
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmpx_gt_i32_e64 s22, v133
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_execz .LBB6_24
; %bb.23:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[131:132], null, 0x408, v133, v[145:146]
	global_load_b64 v[131:132], v[131:132], off
.LBB6_24:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_or_b32_e32 v137, 4, v1
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v136, 0
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1916 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v167, v131 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b32 v168, v132 offset:24576
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmpx_gt_i32_e64 s22, v137
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_execz .LBB6_26
; %bb.25:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[131:132], null, 0x408, v137, v[145:146]
	global_load_b64 v[135:136], v[131:132], off
.LBB6_26:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_or_b32_e32 v131, 5, v1
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_nc_u32_e32 v132, 0x6400, v162
	.loc	0 1910 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v132, v135, v136 offset1:1
	.loc	0 1908 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmpx_gt_i32_e64 s22, v131
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_execz .LBB6_28
; %bb.27:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[131:132], null, 0x408, v131, v[145:146]
	global_load_b64 v[133:134], v[131:132], off
.LBB6_28:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_or_b32_e32 v137, 6, v1
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v136, 0
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1916 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v169, v133 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b32 v170, v134 offset:24576
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmpx_gt_i32_e64 s22, v137
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_execz .LBB6_30
; %bb.29:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[133:134], null, 0x408, v137, v[145:146]
	global_load_b64 v[135:136], v[133:134], off
.LBB6_30:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1907 51 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1907:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_or_b32_e32 v1, 7, v1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1916 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v171, v135 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b32 v172, v136 offset:24576
	.loc	0 1908 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1908:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmpx_gt_i32_e64 s22, v1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_execz .LBB6_32
; %bb.31:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1914 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[131:132], null, 0x408, v1, v[145:146]
	global_load_b64 v[131:132], v[131:132], off
.LBB6_32:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1916 62 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v173, v131 offset:24576
	.loc	0 1918 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b32 v174, v132 offset:24576
.LBB6_33:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 66 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:66
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp517:
	.loc	3 701 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1922:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1922:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1922:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp518:
	.loc	0 1924 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1924:18 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_and_saveexec_b32 s3, s18
	s_cbranch_execz .LBB6_35
; %bb.34:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_nc_u32_e32 v1, 0x6000, v175
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_nc_u32_e32 v135, 0x6400, v175
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_2addr_b32 v[131:132], v1 offset1:4
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_2addr_b32 v[133:134], v135 offset1:4
.Ltmp519:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v136, v152, v131
.Ltmp520:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v137, v152, v133
.Ltmp521:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v138, v152, v132
.Ltmp522:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v139, v152, v134
.Ltmp523:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v131, v136, v131, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v133, v137, v133, v157
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v138, v132, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v139, v134, v157
.Ltmp524:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v132, v151, v131
.Ltmp525:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v137, v151, v133
.Ltmp526:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v138, v151, v136
.Ltmp527:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v139, v151, v134
.Ltmp528:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v131, v132, v131, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v137, v133, v158
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v133, v138, v136, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v139, v134, v158
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b128 v203, v[131:134] offset:12288
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v131, v176 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v132, v177 offset:24576
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v133, v178 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v134, v179 offset:24576
	ds_load_b32 v136, v181 offset:24576
	ds_load_b32 v137, v183 offset:24576
.Ltmp529:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x5
	ds_bpermute_b32 v138, v152, v131
.Ltmp530:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x5
	ds_bpermute_b32 v139, v152, v132
.Ltmp531:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x5
	ds_bpermute_b32 v140, v152, v133
.Ltmp532:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x5
	ds_bpermute_b32 v141, v152, v134
.Ltmp533:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v131, v138, v131, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v139, v132, v157
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v133, v140, v133, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v141, v134, v157
.Ltmp534:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v138, v151, v131
.Ltmp535:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v139, v151, v132
.Ltmp536:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v140, v151, v133
.Ltmp537:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v141, v151, v134
.Ltmp538:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v131, v138, v131, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v139, v132, v158
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v133, v140, v133, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v141, v134, v158
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b128 v203, v[131:134] offset:12800
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v131, v180 offset:24576
	ds_load_b32 v132, v182 offset:24576
.Ltmp539:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v133, v152, v136
	ds_bpermute_b32 v134, v152, v137
.Ltmp540:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v152, v131
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v152, v132
.Ltmp541:
	.loc	0 1947 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v133, v133, v136, v157
	s_wait_dscnt 0x2
	v_perm_b32 v134, v134, v137, v157
.Ltmp542:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v137, v151, v134
.Ltmp543:
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v131, v138, v131, v157
	s_wait_dscnt 0x1
	v_perm_b32 v136, v139, v132, v157
.Ltmp544:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v132, v151, v133
.Ltmp545:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v138, v151, v131
	ds_bpermute_b32 v139, v151, v136
.Ltmp546:
	.loc	0 1952 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v134, v137, v134, v158
	s_wait_dscnt 0x2
	v_perm_b32 v132, v132, v133, v158
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v131, v138, v131, v158
	s_wait_dscnt 0x0
	v_perm_b32 v133, v139, v136, v158
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b128 v203, v[131:134] offset:13312
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v131, v184 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v132, v185 offset:24576
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v133, v186 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v134, v187 offset:24576
.Ltmp547:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v152, v131
.Ltmp548:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v152, v132
.Ltmp549:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v152, v133
.Ltmp550:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v152, v134
.Ltmp551:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v131, v136, v131, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v137, v132, v157
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v133, v138, v133, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v139, v134, v157
.Ltmp552:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v136, v151, v131
.Ltmp553:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v137, v151, v132
.Ltmp554:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v138, v151, v133
.Ltmp555:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v139, v151, v134
.Ltmp556:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v131, v136, v131, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v137, v132, v158
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v133, v138, v133, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v139, v134, v158
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b128 v203, v[131:134] offset:13824
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_2addr_b32 v[131:132], v1 offset0:32 offset1:36
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_2addr_b32 v[133:134], v135 offset0:32 offset1:36
.Ltmp557:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v1, v152, v131
.Ltmp558:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v135, v152, v133
.Ltmp559:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v136, v152, v132
.Ltmp560:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v137, v152, v134
.Ltmp561:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v1, v1, v131, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v133, v135, v133, v157
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v135, v136, v132, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v137, v134, v157
.Ltmp562:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v131, v151, v1
.Ltmp563:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v132, v151, v133
.Ltmp564:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v136, v151, v135
.Ltmp565:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v137, v151, v134
.Ltmp566:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v131, v131, v1, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v132, v133, v158
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v133, v136, v135, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v137, v134, v158
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b128 v203, v[131:134] offset:14336
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v1, v188 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v131, v189 offset:24576
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v132, v190 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v133, v191 offset:24576
.Ltmp567:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v152, v1
.Ltmp568:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v152, v131
.Ltmp569:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v152, v132
.Ltmp570:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v152, v133
.Ltmp571:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v1, v134, v1, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v134, v135, v131, v157
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v135, v136, v132, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v137, v133, v157
.Ltmp572:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v131, v151, v1
.Ltmp573:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v132, v151, v134
.Ltmp574:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v133, v151, v135
.Ltmp575:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v137, v151, v136
.Ltmp576:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v131, v131, v1, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v132, v134, v158
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v133, v133, v135, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v137, v136, v158
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b128 v203, v[131:134] offset:14848
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v1, v192 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v131, v193 offset:24576
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v132, v194 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v133, v195 offset:24576
.Ltmp577:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v152, v1
.Ltmp578:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v152, v131
.Ltmp579:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v152, v132
.Ltmp580:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v152, v133
.Ltmp581:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v1, v134, v1, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v134, v135, v131, v157
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v135, v136, v132, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v137, v133, v157
.Ltmp582:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v131, v151, v1
.Ltmp583:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v132, v151, v134
.Ltmp584:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v133, v151, v135
.Ltmp585:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v137, v151, v136
.Ltmp586:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v131, v131, v1, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v132, v134, v158
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v133, v133, v135, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v137, v136, v158
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b128 v203, v[131:134] offset:15360
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v1, v196 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v131, v197 offset:24576
	.loc	0 1942 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v132, v198 offset:24576
	.loc	0 1943 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b32 v133, v199 offset:24576
.Ltmp587:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v134, v152, v1
.Ltmp588:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v152, v131
.Ltmp589:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v152, v132
.Ltmp590:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v152, v133
.Ltmp591:
	.loc	0 1945 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v1, v134, v1, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v134, v135, v131, v157
	.loc	0 1945 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v135, v136, v132, v157
	.loc	0 1947 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v137, v133, v157
.Ltmp592:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v131, v151, v1
.Ltmp593:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v132, v151, v134
.Ltmp594:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v133, v151, v135
.Ltmp595:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	ds_bpermute_b32 v137, v151, v136
.Ltmp596:
	.loc	0 1949 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v131, v131, v1, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v132, v132, v134, v158
	.loc	0 1949 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v133, v133, v135, v158
	.loc	0 1952 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1952:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v137, v136, v158
	.loc	0 1955 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1955:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b128 v204, v[131:134] offset:15872
.LBB6_35:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 77 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:77
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1961 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB6_39
; %bb.36:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1962 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1962:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_nc_u32_e32 v131, s13, v0
	v_mov_b32_e32 v1, 0
	.loc	0 1964 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s22, v131
	s_cbranch_execz .LBB6_38
; %bb.37:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1969 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[132:133], null, 0x408, v131, s[14:15]
	.loc	0 1971 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_i64_i32 v[134:135], null, 0x408, v131, s[8:9]
	.loc	0 1969 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	global_load_d16_b16 v1, v[132:133], off offset:1024
	.loc	0 1971 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	global_load_d16_hi_b16 v1, v[134:135], off offset:1024
	.loc	0 1969 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v131.h, 8, v1.l
	.loc	0 1971 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshrrev_b16 v131.l, 8, v1.h
	.loc	0 1971 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_and_b16 v132.h, 0xff, v1.l
	v_and_b16 v132.l, 0xff, v1.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1972 56 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_pk_lshlrev_b16 v1, 8, v131 op_sel_hi:[0,1]
	.loc	0 1971 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_or_b32_e32 v1, v1, v132
.LBB6_38:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 53 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:53
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1974 61 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b16_d16_hi v160, v1 offset:36864
	.loc	0 1975 61                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1975:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_store_b16 v160, v1 offset:36960
.LBB6_39:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 61 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:61
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1878 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1878:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_add_co_i32 s3, s13, 47
.Ltmp597:
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
.Ltmp598:
	.loc	0 1878 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1878:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s3, s22
	v_mov_b32_e32 v206, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_le_i32 s3, s16
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s4, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s19, s23, s3
	.loc	0 1984 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1984:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s19, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s0, s3
.Ltmp599:
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ] ] ]
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp600:
	.loc	0 1984 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1984:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB6_41
; %bb.40:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1984 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1984:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	global_load_b32 v206, v[147:148], off
.LBB6_41:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 46                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:46
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_mov_b32 s20, 0
	s_branch .LBB6_43
.LBB6_42:                               ;   in Loop: Header=BB6_43 Depth=2
	s_or_b32 exec_lo, exec_lo, s3
	v_sub_f32_e32 v208, v207, v1
	.loc	0 2051 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v207
	.loc	0 2065 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2065:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_f32_e32 v132, v132, v133
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v213, null, v131, v131, v139
	.loc	0 2083 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v215, null, v131, v131, v138
	v_mul_f32_e32 v208, 0x3fb8aa3b, v208
	.loc	0 2080 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2080:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mov_b16_e64 v232.l, v154.l
	.loc	0 2082 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mov_b16_e64 v232.h, 0
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_rcp_f32_e32 v214, v213
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v208, v208
	.loc	0 2080 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2080:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mov_b16_e64 v231.l, v232.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	.loc	0 2082 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mov_b16_e64 v231.h, v232.h
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v216, -v213, v214, 1.0
	.loc	0 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v207, 0, v208 :: v_dual_fmac_f32 v214, v216, v214
	.loc	0 2088 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v216, null, v131, v131, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2075 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2075:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v205, v205, v207
	.loc	0 2075 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2075:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v208, null, v131, v131, v205
	v_div_scale_f32 v211, vcc_lo, v205, v131, v205
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v209, v208
	v_fma_f32 v210, -v208, v209, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v209, v210, v209
	v_mul_f32_e32 v210, v211, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v212, -v208, v210, v211
	v_fmac_f32_e32 v210, v212, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v208, -v208, v210, v211
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v208, v208, v209, v210
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v133, v208, v131, v205
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v129, v129, v133 :: v_dual_fmac_f32 v132, v2, v207
	v_mul_f32_e32 v127, v127, v133
	.loc	0 2081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v2, null, v131, v131, v141
	.loc	0 2081 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v207, null, v131, v131, v140
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v130, v130, v133 :: v_dual_mul_f32 v125, v125, v133
	v_dual_mul_f32 v128, v128, v133 :: v_dual_mul_f32 v123, v123, v133
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 2081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_rcp_f32_e32 v205, v2
	.loc	0 2081 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_rcp_f32_e32 v209, v207
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v126, v126, v133 :: v_dual_mul_f32 v121, v121, v133
	v_dual_mul_f32 v124, v124, v133 :: v_dual_mul_f32 v119, v119, v133
	v_dual_mul_f32 v122, v122, v133 :: v_dual_mul_f32 v117, v117, v133
	v_dual_mul_f32 v120, v120, v133 :: v_dual_mul_f32 v115, v115, v133
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	.loc	0 2081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v208, -v2, v205, 1.0
	.loc	0 2081 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v211, -v207, v209, 1.0
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v118, v118, v133 :: v_dual_mul_f32 v113, v113, v133
	v_dual_mul_f32 v116, v116, v133 :: v_dual_mul_f32 v111, v111, v133
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	.loc	0 2081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v205, v208, v205
	v_div_scale_f32 v208, vcc_lo, v141, v131, v141
	.loc	0 2081 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v209, v211, v209
	v_div_scale_f32 v211, s3, v140, v131, v140
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v114, v114, v133 :: v_dual_mul_f32 v109, v109, v133
	.loc	0 2081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v210, v208, v205
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v112, v112, v133 :: v_dual_mul_f32 v107, v107, v133
	v_dual_mul_f32 v110, v110, v133 :: v_dual_mul_f32 v105, v105, v133
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 2081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v212, -v2, v210, v208
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v108, v108, v133 :: v_dual_mul_f32 v103, v103, v133
	v_dual_mul_f32 v106, v106, v133 :: v_dual_mul_f32 v101, v101, v133
	.loc	0 2081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v210, v212, v205
	.loc	0 2081 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v212, v211, v209
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v104, v104, v133 :: v_dual_mul_f32 v99, v99, v133
	v_dual_mul_f32 v102, v102, v133 :: v_dual_mul_f32 v97, v97, v133
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	0 2081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v2, -v2, v210, v208
	.loc	0 2081 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v208, -v207, v212, v211
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v100, v100, v133 :: v_dual_mul_f32 v95, v95, v133
	v_dual_mul_f32 v98, v98, v133 :: v_dual_mul_f32 v93, v93, v133
	.loc	0 2081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v2, v2, v205, v210
	.loc	0 2083 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_rcp_f32_e32 v205, v215
	.loc	0 2081 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v212, v208, v209
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v208, s4, v139, v131, v139
	.loc	0 2081 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	0 2081 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_fixup_f32 v2, v2, v131, v141
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v96, v96, v133 :: v_dual_mul_f32 v91, v91, v133
	.loc	0 2081 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v141, -v207, v212, v211
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v94, v94, v133 :: v_dual_mul_f32 v89, v89, v133
	s_delay_alu instid0(TRANS32_DEP_1)
	.loc	0 2083 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v210, -v215, v205, 1.0
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v92, v92, v133 :: v_dual_mul_f32 v87, v87, v133
	.loc	0 2081 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v141, v141, v209, v212
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	0 2083 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v205, v210, v205
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v207, v208, v214
	.loc	0 2086 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v209, null, v131, v131, v137
	.loc	0 2081 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_fixup_f32 v140, v141, v131, v140
	.loc	0 2083 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v210, s3, v138, v131, v138
	.loc	0 2086 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v212, null, v131, v131, v136
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v211, -v213, v207, v208
	.loc	0 2086 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_rcp_f32_e32 v141, v209
	.loc	0 2080 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2080:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cvt_pk_fp8_f32 v231.l, v2, v140
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v90, v90, v133 :: v_dual_mul_f32 v85, v85, v133
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_fmac_f32 v207, v211, v214 :: v_dual_mul_f32 v88, v88, v133
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v83, v83, v133 :: v_dual_mul_f32 v86, v86, v133
	v_mul_f32_e32 v81, v81, v133
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v2, -v213, v207, v208
	.loc	0 2086 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_rcp_f32_e32 v213, v212
	s_delay_alu instid0(TRANS32_DEP_2)
	.loc	0 2086 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v208, -v209, v141, 1.0
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v84, v84, v133 :: v_dual_mul_f32 v79, v79, v133
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v2, v2, v214, v207
	.loc	0 2088 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_rcp_f32_e32 v207, v216
	.loc	0 2086 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v141, v208, v141
	.loc	0 2083 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v82, v82, v133 :: v_dual_mul_f32 v77, v77, v133
	.loc	0 2083 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_fixup_f32 v2, v2, v131, v139
	.loc	0 2088 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v139, null, v131, v131, v135
	.loc	0 2086 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v208, -v212, v213, 1.0
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v80, v80, v133 :: v_dual_mul_f32 v75, v75, v133
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	0 2088 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v217, -v216, v207, 1.0
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v78, v78, v133 :: v_dual_mul_f32 v73, v73, v133
	.loc	0 2086 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v213, v208, v213
	v_div_scale_f32 v208, s5, v136, v131, v136
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	.loc	0 2088 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v207, v217, v207
	.loc	0 2083 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v211, v210, v205
	.loc	0 2088 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v217, s3, v134, v131, v134
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v76, v76, v133 :: v_dual_mul_f32 v71, v71, v133
	v_dual_mul_f32 v74, v74, v133 :: v_dual_mul_f32 v69, v69, v133
	.loc	0 2083 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v140, -v215, v211, v210
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v72, v72, v133 :: v_dual_mul_f32 v67, v67, v133
	v_dual_mul_f32 v70, v70, v133 :: v_dual_mul_f32 v65, v65, v133
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	.loc	0 2083 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v211, v140, v205
	.loc	0 2086 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v140, s4, v137, v131, v137
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v68, v68, v133 :: v_dual_mul_f32 v63, v63, v133
	v_dual_mul_f32 v66, v66, v133 :: v_dual_mul_f32 v61, v61, v133
	.loc	0 2083 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v210, -v215, v211, v210
	.loc	0 2088 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_rcp_f32_e32 v215, v139
	.loc	0 2086 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v214, v140, v141
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v64, v64, v133 :: v_dual_mul_f32 v59, v59, v133
	.loc	0 2083 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v205, v210, v205, v211
	.loc	0 2086 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v211, v208, v213
	.loc	0 2086 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v210, -v209, v214, v140
	s_mov_b32 vcc_lo, s4
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v62, v62, v133 :: v_dual_mul_f32 v57, v57, v133
	.loc	0 2088 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v218, -v139, v215, 1.0
	.loc	0 2083 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_fixup_f32 v138, v205, v131, v138
	.loc	0 2086 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v214, v210, v141
	.loc	0 2086 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v205, -v212, v211, v208
	.loc	0 2088 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v210, v217, v207
	.loc	0 2088 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v215, v218, v215
	v_div_scale_f32 v218, s6, v135, v131, v135
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v8, v8, v133
	.loc	0 2086 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v140, -v209, v214, v140
	.loc	0 2086 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v211, v205, v213
	.loc	0 2088 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v205, -v216, v210, v217
	.loc	0 2088 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v209, v218, v215
	.loc	0 2082 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cvt_pk_fp8_f32 v231.h, v2, v138
	.loc	0 2086 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v140, v140, v141, v214
	.loc	0 2086 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v141, -v212, v211, v208
	.loc	0 2088 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v210, v205, v207
	.loc	0 2088 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v208, -v139, v209, v218
	.loc	0 2078 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v10, v10, v133
	.loc	0 2086 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 vcc_lo, s5
	.loc	0 2086 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_fixup_f32 v137, v140, v131, v137
	.loc	0 2088 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v140, -v216, v210, v217
	.loc	0 2088 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fmac_f32_e32 v209, v208, v215
	.loc	0 2086 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v141, v141, v213, v211
	.loc	0 2088 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	0 2093 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshl_add_u32 v205, s20, 12, v159
	.loc	0 2088 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v140, v140, v207, v210
	.loc	0 2088 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_f32 v139, -v139, v209, v218
	s_mov_b32 vcc_lo, s6
	.loc	0 2086 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_fixup_f32 v136, v141, v131, v136
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v60, v60, v133 :: v_dual_mul_f32 v55, v55, v133
	.loc	0 2088 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v139, v139, v215, v209
	.loc	0 2088 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_fixup_f32 v2, v140, v131, v134
	.loc	0 2085 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cvt_pk_fp8_f32 v232.l, v137, v136
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v58, v58, v133 :: v_dual_mul_f32 v53, v53, v133
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 2088 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2088:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_fixup_f32 v233, v139, v131, v135
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b128 v[134:137], v205 offset:12288
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b128 v[138:141], v205 offset:12800
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b128 v[207:210], v205 offset:13312
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b128 v[211:214], v205 offset:13824
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b128 v[215:218], v205 offset:14336
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b128 v[219:222], v205 offset:14848
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b128 v[223:226], v205 offset:15360
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2095 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b128 v[227:230], v205 offset:15872
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v56, v56, v133 :: v_dual_mul_f32 v51, v51, v133
	v_dual_mul_f32 v54, v54, v133 :: v_dual_mul_f32 v49, v49, v133
	v_dual_mul_f32 v52, v52, v133 :: v_dual_mul_f32 v47, v47, v133
	v_dual_mul_f32 v50, v50, v133 :: v_dual_mul_f32 v45, v45, v133
	v_dual_mul_f32 v48, v48, v133 :: v_dual_mul_f32 v43, v43, v133
	v_dual_mul_f32 v46, v46, v133 :: v_dual_mul_f32 v41, v41, v133
	v_dual_mul_f32 v44, v44, v133 :: v_dual_mul_f32 v39, v39, v133
	v_dual_mul_f32 v42, v42, v133 :: v_dual_mul_f32 v37, v37, v133
	v_dual_mul_f32 v40, v40, v133 :: v_dual_mul_f32 v35, v35, v133
	v_dual_mul_f32 v38, v38, v133 :: v_dual_mul_f32 v33, v33, v133
	v_dual_mul_f32 v36, v36, v133 :: v_dual_mul_f32 v31, v31, v133
	v_dual_mul_f32 v34, v34, v133 :: v_dual_mul_f32 v29, v29, v133
	v_dual_mul_f32 v32, v32, v133 :: v_dual_mul_f32 v27, v27, v133
	v_dual_mul_f32 v30, v30, v133 :: v_dual_mul_f32 v25, v25, v133
	v_dual_mul_f32 v28, v28, v133 :: v_dual_mul_f32 v23, v23, v133
	v_dual_mul_f32 v26, v26, v133 :: v_dual_mul_f32 v21, v21, v133
	v_dual_mul_f32 v24, v24, v133 :: v_dual_mul_f32 v19, v19, v133
	v_dual_mul_f32 v22, v22, v133 :: v_dual_mul_f32 v17, v17, v133
	v_dual_mul_f32 v20, v20, v133 :: v_dual_mul_f32 v15, v15, v133
	v_dual_mul_f32 v18, v18, v133 :: v_dual_mul_f32 v13, v13, v133
	v_dual_mul_f32 v16, v16, v133 :: v_dual_mul_f32 v11, v11, v133
	v_dual_mul_f32 v14, v14, v133 :: v_dual_mul_f32 v9, v9, v133
	v_dual_mul_f32 v12, v12, v133 :: v_dual_mul_f32 v7, v7, v133
	.loc	0 2087 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cvt_pk_fp8_f32 v232.h, v2, v233
	.loc	0 2078 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v6, v6, v133 :: v_dual_mul_f32 v5, v5, v133
	v_dual_mul_f32 v4, v4, v133 :: v_dual_mul_f32 v3, v3, v133
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x7
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_f32_16x16x16_fp8_fp8 v[123:130], v[134:135], v[231:232], v[123:130]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[115:122], v[136:137], v[231:232], v[115:122]
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[107:114], v[138:139], v[231:232], v[107:114]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[99:106], v[140:141], v[231:232], v[99:106]
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[91:98], v[207:208], v[231:232], v[91:98]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[83:90], v[209:210], v[231:232], v[83:90]
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[75:82], v[211:212], v[231:232], v[75:82]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[67:74], v[213:214], v[231:232], v[67:74]
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[59:66], v[215:216], v[231:232], v[59:66]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[51:58], v[217:218], v[231:232], v[51:58]
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[43:50], v[219:220], v[231:232], v[43:50]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[35:42], v[221:222], v[231:232], v[35:42]
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[27:34], v[223:224], v[231:232], v[27:34]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[19:26], v[225:226], v[231:232], v[19:26]
	.loc	0 2099 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2099:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[11:18], v[227:228], v[231:232], v[11:18]
	.loc	0 2102 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2102:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[3:10], v[229:230], v[231:232], v[3:10]
	v_dual_mov_b32 v2, v132 :: v_dual_mov_b32 v205, v131
	.loc	0 2104 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	;;#ASMSTART
	;;#ASMEND
	v_mov_b32_e32 v207, v1
	.loc	0 1990 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_add_co_i32 s20, s20, 1
	.loc	0 1990 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s20, 3
	.loc	0 1990 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_scc1 .LBB6_12
.LBB6_43:                               ;   Parent Loop BB6_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB6_45 Depth 3
	.loc	0 1991 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1991:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s20, 0
	s_cselect_b32 s3, -1, 0
	.loc	0 1991 34 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1991:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cmp_eq_u32 s20, 1
	s_cselect_b32 s4, 16, 32
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s4, s13
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s22, s4
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s3, s4
	.loc	0 1992 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1992:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_50
; %bb.44:                               ;   in Loop: Header=BB6_43 Depth=2
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	v_mov_b32_e32 v131, 0
	s_mov_b32 s3, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v132, v131 :: v_dual_mov_b32 v133, v131
	v_dual_mov_b32 v134, v131 :: v_dual_mov_b32 v135, v131
	v_dual_mov_b32 v136, v131 :: v_dual_mov_b32 v137, v131
	v_mov_b32_e32 v138, v131
.LBB6_45:                               ;   Parent Loop BB6_13 Depth=1
                                        ;     Parent Loop BB6_43 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	.loc	0 2013 58 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, s3, 6
	.loc	0 2016 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v139, vcc_lo, v200, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v140, null, 0, v201, vcc_lo
	.loc	0 2007 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2007:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mul_i32 s4, s3, 6
	.loc	0 2001 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_add_co_i32 s3, s3, 1
	.loc	0 2007 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2007:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s4, s20
	.loc	0 2016 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_clause 0x3
	global_load_b64 v[216:217], v[139:140], off
	global_load_b64 v[218:219], v[139:140], off offset:16
	global_load_b64 v[220:221], v[139:140], off offset:32
	global_load_b64 v[139:140], v[139:140], off offset:48
	.loc	0 2006 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_add_u32 v1, s4, 9, v159
	.loc	0 2008 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2008:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b128 v[208:211], v1
	ds_load_b128 v[212:215], v1 offset:1536
	.loc	0 2025 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cmp_lg_u32 s3, 4
	.loc	0 2019 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2019:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[131:138], v[208:209], v[216:217], v[131:138]
	.loc	0 2022 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[131:138], v[210:211], v[218:219], v[131:138]
	.loc	0 2019 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2019:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[131:138], v[212:213], v[220:221], v[131:138]
	.loc	0 2022 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[131:138], v[214:215], v[139:140], v[131:138]
	.loc	0 2001 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_scc1 .LBB6_45
; %bb.46:                               ;   in Loop: Header=BB6_43 Depth=2
	.loc	0 0 17 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:17
	v_lshl_add_u32 v208, s20, 5, v161
	.loc	0 2028 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_lshl4_add_u32 s3, s20, s13
	v_mov_b32_e32 v209, 0xff800000
	.loc	0 2030 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s3, 15
	v_or_b32_e32 v210, s3, v153
	.loc	0 2034 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2034:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_b96 v[139:141], v208 offset:36866
	ds_load_u16_d16 v1, v208 offset:36878
	.loc	0 2030 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s4, s22
	s_cselect_b32 s5, -1, 0
	s_cmp_le_i32 s4, s16
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s5, s4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s19, s3
	.loc	0 2039 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2039:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB6_48
; %bb.47:                               ;   in Loop: Header=BB6_43 Depth=2
	.loc	0 2034 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2034:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_u16_d16 v209, v208 offset:36864
	v_mul_f32_e32 v131, v149, v131
	v_cmp_le_i32_e32 vcc_lo, v210, v206
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s4, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v131, v209, v131, neg(0) op_sel_hi:[1,0,0]
	.loc	0 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v209, 0xff800000, v131, vcc_lo
.LBB6_48:                               ;   in Loop: Header=BB6_43 Depth=2
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mul_f32_e32 v131, v149, v132
	v_or_b32_e32 v132, 2, v210
	v_cmp_lt_i32_e32 vcc_lo, v210, v206
	v_dual_mul_f32 v133, v149, v133 :: v_dual_mul_f32 v136, v149, v136
	s_wait_dscnt 0x1
	v_fma_mix_f32 v131, v139, v131, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e64 s3, v132, v206
	v_or_b32_e32 v132, 3, v210
	s_or_b32 s5, s4, vcc_lo
	v_fma_mix_f32 v133, v139, v133, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	0 2039 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2039:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s5
	s_or_b32 s3, s4, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v131, 0xff800000, v131, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v132, v206
	v_mul_f32_e32 v132, v149, v134
	v_or_b32_e32 v134, 4, v210
	s_and_b32 s3, s0, s3
	v_mul_f32_e32 v135, v149, v135
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v133, 0xff800000, v133, s3
	s_or_b32 s3, s4, vcc_lo
	v_fma_mix_f32 v132, v140, v132, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e32 vcc_lo, v134, v206
	v_or_b32_e32 v134, 5, v210
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s0, s3
	v_fma_mix_f32 v135, v140, v135, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v132, 0xff800000, v132, s3
	s_or_b32 s3, s4, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v134, v206
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s0, s3
	v_or_b32_e32 v139, 7, v210
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v134, 0xff800000, v135, s3
	v_fma_mix_f32 v135, v141, v136, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v136, 6, v210
	s_or_b32 s3, s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s3
	v_cmp_le_i32_e64 s3, v139, v206
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v135, 0xff800000, v135, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v136, v206
	v_dual_mul_f32 v136, v149, v137 :: v_dual_mul_f32 v137, v149, v138
.Ltmp601:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2047:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_max3_num_f32 v138, v209, 0xff800000, v131
	s_or_b32 s3, s4, s3
	s_or_b32 s5, s4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_mix_f32 v136, v141, v136, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v1, v1, v137, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v137, v138, v133, v132
.Ltmp602:
	.loc	0 2039 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2039:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v136, 0xff800000, v136, vcc_lo
	s_and_b32 vcc_lo, s0, s3
	.loc	0 2073 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2073:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_mov_b32 s3, exec_lo
	.loc	0 2039 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2039:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v138, 0xff800000, v1, vcc_lo
.Ltmp603:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2047:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_max3_num_f32 v1, v137, v134, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v1, v1, v136, v138
.Ltmp604:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2048:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_mov_b32_e32 v137, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v137, v137, s7, 0xfedcba98
.Ltmp605:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_max3_num_f32 v1, v207, v1, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v136, v136, v1 :: v_dual_add_nc_u32 v139, 0x9060, v208
	v_dual_sub_f32 v137, v138, v1 :: v_dual_sub_f32 v138, v209, v1
	v_dual_sub_f32 v131, v131, v1 :: v_dual_sub_f32 v134, v134, v1
	v_sub_f32_e32 v132, v132, v1
	v_mul_f32_e32 v137, 0x3fb8aa3b, v137
	v_sub_f32_e32 v135, v135, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v131, 0x3fb8aa3b, v131 :: v_dual_mul_f32 v136, 0x3fb8aa3b, v136
	v_mul_f32_e32 v141, 0x3fb8aa3b, v132
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v135, 0x3fb8aa3b, v135
	v_exp_f32_e32 v140, v131
.Ltmp606:
	.loc	0 2055 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	ds_load_2addr_b32 v[131:132], v139 offset1:1
	v_add_nc_u32_e32 v139, 0x9068, v208
	v_mul_f32_e32 v208, 0x3fb8aa3b, v134
	v_exp_f32_e32 v137, v137
	v_mul_f32_e32 v138, 0x3fb8aa3b, v138
	v_exp_f32_e32 v209, v135
	ds_load_2addr_b32 v[134:135], v139 offset1:1
	v_exp_f32_e32 v139, v208
	v_exp_f32_e32 v141, v141
	v_exp_f32_e32 v138, v138
	v_exp_f32_e32 v136, v136
	.loc	0 2059 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v208, v137, 0, vcc_lo
	v_sub_f32_e32 v133, v133, v1
	v_cndmask_b32_e64 v209, v209, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v210, v141, 0, vcc_lo
	v_mul_f32_e32 v133, 0x3fb8aa3b, v133
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v137, v138, 0, vcc_lo
	v_cndmask_b32_e64 v138, v140, 0, vcc_lo
	v_cndmask_b32_e64 v211, v136, 0, vcc_lo
	v_exp_f32_e32 v133, v133
	.loc	0 2063 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_dscnt 0x1
	v_fma_mix_f32 v141, v131, v137, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v140, v131, v138, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_f32_e32 v137, v137, v138
	.loc	0 2059 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cndmask_b32_e64 v131, v139, 0, vcc_lo
	.loc	0 2063 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_mix_f32 v138, v132, v210, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v136, v134, v209, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	0 2059 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cndmask_b32_e64 v133, v133, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 2063 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_mix_f32 v139, v132, v133, neg(0) op_sel_hi:[1,0,0]
.Ltmp607:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_max3_num_f32 v132, v141, 0, v140
.Ltmp608:
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_f32_e32 v133, v133, v137
	.loc	0 2063 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_fma_mix_f32 v137, v134, v131, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v134, v135, v211, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v135, v135, v208, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
.Ltmp609:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_max3_num_f32 v132, v132, v139, v138
.Ltmp610:
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_f32_e32 v133, v210, v133
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp611:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_max3_num_f32 v132, v132, v137, v136
.Ltmp612:
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_f32_e32 v131, v131, v133
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp613:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_max3_num_f32 v210, v132, v134, v135
.Ltmp614:
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_f32_e32 v131, v209, v131
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp615:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2071:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_mov_b32_e32 v133, v210
.Ltmp616:
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_f32_e32 v131, v211, v131
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp617:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2071:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_permlanex16_b32 v133, v133, s7, 0xfedcba98
.Ltmp618:
	.loc	0 2062 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_add_f32 v132, v208, v131 :: v_dual_max_num_f32 v131, v133, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp619:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2065:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_dual_mov_b32 v133, v132 :: v_dual_max_num_f32 v208, v210, v131
	v_permlanex16_b32 v133, v133, s7, 0xfedcba98
	v_mov_b32_e32 v131, v205
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp620:
	.loc	0 2073 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2073:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmpx_lt_f32_e32 0, v208
	s_cbranch_execz .LBB6_42
; %bb.49:                               ;   in Loop: Header=BB6_43 Depth=2
	.loc	0 2074 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_div_scale_f32 v131, null, 0x43e00000, 0x43e00000, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v209, v131
	v_fma_f32 v210, -v131, v209, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v209, v210, v209
	v_div_scale_f32 v210, vcc_lo, v208, 0x43e00000, v208
	v_mul_f32_e32 v211, v210, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v212, -v131, v211, v210
	v_fmac_f32_e32 v211, v212, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v131, -v131, v211, v210
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v131, v131, v209, v211
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v131, v131, 0x43e00000, v208
.Ltmp621:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ] ]
	v_max_num_f32_e32 v131, 0x1f800000, v131
	s_branch .LBB6_42
.Ltmp622:
.LBB6_50:                               ;   in Loop: Header=BB6_43 Depth=2
	.loc	2 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v1, v207
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v207, v1
	.loc	0 1990 40 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_add_co_i32 s20, s20, 1
	.loc	0 1990 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s20, 3
	.loc	0 1990 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_cbranch_scc0 .LBB6_43
	s_branch .LBB6_12
.LBB6_51:
	.loc	0 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v205, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_mov_b32_e32 v9, v2
	v_dual_mov_b32 v3, v2 :: v_dual_mov_b32 v4, v2
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v6, v2
	v_dual_mov_b32 v7, v2 :: v_dual_mov_b32 v8, v2
	v_dual_mov_b32 v1, 0xff800000 :: v_dual_mov_b32 v130, v9
	v_dual_mov_b32 v123, v2 :: v_dual_mov_b32 v122, v9
	v_dual_mov_b32 v115, v2 :: v_dual_mov_b32 v114, v9
	v_dual_mov_b32 v107, v2 :: v_dual_mov_b32 v106, v9
	v_dual_mov_b32 v99, v2 :: v_dual_mov_b32 v98, v9
	v_dual_mov_b32 v91, v2 :: v_dual_mov_b32 v90, v9
	v_dual_mov_b32 v83, v2 :: v_dual_mov_b32 v82, v9
	v_dual_mov_b32 v75, v2 :: v_dual_mov_b32 v74, v9
	v_dual_mov_b32 v67, v2 :: v_dual_mov_b32 v66, v9
	v_dual_mov_b32 v59, v2 :: v_dual_mov_b32 v58, v9
	v_dual_mov_b32 v51, v2 :: v_dual_mov_b32 v50, v9
	v_dual_mov_b32 v43, v2 :: v_dual_mov_b32 v42, v9
	v_dual_mov_b32 v35, v2 :: v_dual_mov_b32 v34, v9
	v_dual_mov_b32 v27, v2 :: v_dual_mov_b32 v26, v9
	v_dual_mov_b32 v19, v2 :: v_dual_mov_b32 v18, v9
	v_dual_mov_b32 v129, v8 :: v_dual_mov_b32 v128, v7
	v_dual_mov_b32 v127, v6 :: v_dual_mov_b32 v126, v5
	v_dual_mov_b32 v125, v4 :: v_dual_mov_b32 v124, v3
	v_dual_mov_b32 v121, v8 :: v_dual_mov_b32 v120, v7
	v_dual_mov_b32 v119, v6 :: v_dual_mov_b32 v118, v5
	v_dual_mov_b32 v117, v4 :: v_dual_mov_b32 v116, v3
	v_dual_mov_b32 v113, v8 :: v_dual_mov_b32 v112, v7
	v_dual_mov_b32 v111, v6 :: v_dual_mov_b32 v110, v5
	v_dual_mov_b32 v109, v4 :: v_dual_mov_b32 v108, v3
	v_dual_mov_b32 v105, v8 :: v_dual_mov_b32 v104, v7
	v_dual_mov_b32 v103, v6 :: v_dual_mov_b32 v102, v5
	v_dual_mov_b32 v101, v4 :: v_dual_mov_b32 v100, v3
	v_dual_mov_b32 v97, v8 :: v_dual_mov_b32 v96, v7
	v_dual_mov_b32 v95, v6 :: v_dual_mov_b32 v94, v5
	v_dual_mov_b32 v93, v4 :: v_dual_mov_b32 v92, v3
	v_dual_mov_b32 v89, v8 :: v_dual_mov_b32 v88, v7
	v_dual_mov_b32 v87, v6 :: v_dual_mov_b32 v86, v5
	v_dual_mov_b32 v85, v4 :: v_dual_mov_b32 v84, v3
	v_dual_mov_b32 v81, v8 :: v_dual_mov_b32 v80, v7
	v_dual_mov_b32 v79, v6 :: v_dual_mov_b32 v78, v5
	v_dual_mov_b32 v77, v4 :: v_dual_mov_b32 v76, v3
	v_dual_mov_b32 v73, v8 :: v_dual_mov_b32 v72, v7
	v_dual_mov_b32 v71, v6 :: v_dual_mov_b32 v70, v5
	v_dual_mov_b32 v69, v4 :: v_dual_mov_b32 v68, v3
	v_dual_mov_b32 v65, v8 :: v_dual_mov_b32 v64, v7
	v_dual_mov_b32 v63, v6 :: v_dual_mov_b32 v62, v5
	v_dual_mov_b32 v61, v4 :: v_dual_mov_b32 v60, v3
	v_dual_mov_b32 v57, v8 :: v_dual_mov_b32 v56, v7
	v_dual_mov_b32 v55, v6 :: v_dual_mov_b32 v54, v5
	v_dual_mov_b32 v53, v4 :: v_dual_mov_b32 v52, v3
	v_dual_mov_b32 v49, v8 :: v_dual_mov_b32 v48, v7
	v_dual_mov_b32 v47, v6 :: v_dual_mov_b32 v46, v5
	v_dual_mov_b32 v45, v4 :: v_dual_mov_b32 v44, v3
	v_dual_mov_b32 v41, v8 :: v_dual_mov_b32 v40, v7
	v_dual_mov_b32 v39, v6 :: v_dual_mov_b32 v38, v5
	v_dual_mov_b32 v37, v4 :: v_dual_mov_b32 v36, v3
	v_dual_mov_b32 v33, v8 :: v_dual_mov_b32 v32, v7
	v_dual_mov_b32 v31, v6 :: v_dual_mov_b32 v30, v5
	v_dual_mov_b32 v29, v4 :: v_dual_mov_b32 v28, v3
	v_dual_mov_b32 v25, v8 :: v_dual_mov_b32 v24, v7
	v_dual_mov_b32 v23, v6 :: v_dual_mov_b32 v22, v5
	v_dual_mov_b32 v21, v4 :: v_dual_mov_b32 v20, v3
	v_dual_mov_b32 v17, v8 :: v_dual_mov_b32 v16, v7
	v_dual_mov_b32 v15, v6 :: v_dual_mov_b32 v14, v5
	v_dual_mov_b32 v13, v4 :: v_dual_mov_b32 v12, v3
	v_dual_mov_b32 v11, v2 :: v_dual_mov_b32 v10, v9
	v_mov_b32_e32 v9, v8
	v_mov_b32_e32 v8, v7
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
.LBB6_52:
	.loc	0 2129 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2129:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_cmp_eq_u32_e32 vcc_lo, 0, v150
	s_and_b32 s2, vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB6_54
; %bb.53:
	.loc	0 2133 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2133:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_u64_u32 v[131:132], null, s17, v142, s[12:13]
	.loc	0 2135 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2135:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mov_b32_e32 v132, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2134 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2134:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_lo_u32 v131, 0x102, v131
	.loc	0 2135 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2135:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshlrev_b64_e32 v[131:132], 2, v[131:132]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v131, vcc_lo, s10, v131
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v132, null, s11, v132, vcc_lo
	.loc	0 2137 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2137:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	global_store_b64 v[131:132], v[1:2], off
.LBB6_54:
	.loc	0 0 24 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 2140 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2140:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB6_56
; %bb.55:
	.loc	0 2143 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2143:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mad_co_u64_u32 v[1:2], null, s17, v142, s[12:13]
	.loc	0 2145 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2145:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mov_b32_e32 v2, 0
	.loc	0 2151 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v0, v205, v123 :: v_dual_mul_f32 v115, v115, v205
	.loc	0 2151 21 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshlrev_b32_e32 v123, 5, v150
	.loc	0 2151 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v107, v107, v205 :: v_dual_mul_f32 v116, v116, v205
	v_dual_mul_f32 v99, v99, v205 :: v_dual_mul_f32 v108, v108, v205
	.loc	0 2144 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2144:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_lo_u32 v1, 0x102, v1
	.loc	0 2151 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v91, v91, v205 :: v_dual_mul_f32 v100, v100, v205
	v_dual_mul_f32 v83, v83, v205 :: v_dual_mul_f32 v92, v92, v205
	v_dual_mul_f32 v75, v75, v205 :: v_dual_mul_f32 v84, v84, v205
	v_dual_mul_f32 v67, v67, v205 :: v_dual_mul_f32 v76, v76, v205
	.loc	0 2145 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2145:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	.loc	0 2151 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v59, v59, v205 :: v_dual_mul_f32 v68, v68, v205
	v_dual_mul_f32 v51, v51, v205 :: v_dual_mul_f32 v60, v60, v205
	v_dual_mul_f32 v43, v43, v205 :: v_dual_mul_f32 v52, v52, v205
	.loc	0 2145 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2145:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, s10, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s11, v2, vcc_lo
	.loc	0 2151 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v35, v35, v205 :: v_dual_mul_f32 v44, v44, v205
	.loc	0 2151 21 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_add_co_u32 v131, vcc_lo, v1, v123
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v132, null, 0, v2, vcc_lo
	.loc	0 2151 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v123, v3, v205 :: v_dual_mul_f32 v12, v12, v205
	v_dual_mul_f32 v1, v205, v124 :: v_dual_mul_f32 v124, v4, v205
	v_dual_mul_f32 v2, v205, v125 :: v_dual_mul_f32 v3, v205, v126
	v_dual_mul_f32 v27, v27, v205 :: v_dual_mul_f32 v36, v36, v205
	v_dual_mul_f32 v19, v19, v205 :: v_dual_mul_f32 v28, v28, v205
	v_dual_mul_f32 v11, v11, v205 :: v_dual_mul_f32 v20, v20, v205
	v_dual_mul_f32 v117, v117, v205 :: v_dual_mul_f32 v118, v118, v205
	v_dual_mul_f32 v109, v109, v205 :: v_dual_mul_f32 v110, v110, v205
	v_dual_mul_f32 v101, v101, v205 :: v_dual_mul_f32 v102, v102, v205
	v_dual_mul_f32 v93, v93, v205 :: v_dual_mul_f32 v94, v94, v205
	v_dual_mul_f32 v85, v85, v205 :: v_dual_mul_f32 v86, v86, v205
	v_dual_mul_f32 v77, v77, v205 :: v_dual_mul_f32 v78, v78, v205
	v_dual_mul_f32 v69, v69, v205 :: v_dual_mul_f32 v70, v70, v205
	v_dual_mul_f32 v61, v61, v205 :: v_dual_mul_f32 v62, v62, v205
	v_dual_mul_f32 v53, v53, v205 :: v_dual_mul_f32 v54, v54, v205
	v_dual_mul_f32 v45, v45, v205 :: v_dual_mul_f32 v46, v46, v205
	v_dual_mul_f32 v37, v37, v205 :: v_dual_mul_f32 v38, v38, v205
	v_dual_mul_f32 v29, v29, v205 :: v_dual_mul_f32 v30, v30, v205
	v_dual_mul_f32 v21, v21, v205 :: v_dual_mul_f32 v22, v22, v205
	v_dual_mul_f32 v127, v205, v127 :: v_dual_mul_f32 v128, v205, v128
	v_dual_mul_f32 v129, v205, v129 :: v_dual_mul_f32 v130, v205, v130
	v_dual_mul_f32 v125, v5, v205 :: v_dual_mul_f32 v126, v6, v205
	.loc	0 2151 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	global_store_b128 v[131:132], v[0:3], off offset:8
	.loc	0 2151 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_dual_mul_f32 v0, v119, v205 :: v_dual_mul_f32 v1, v120, v205
	v_dual_mul_f32 v4, v111, v205 :: v_dual_mul_f32 v3, v122, v205
	v_dual_mul_f32 v103, v103, v205 :: v_dual_mul_f32 v2, v121, v205
	v_dual_mul_f32 v95, v95, v205 :: v_dual_mul_f32 v104, v104, v205
	v_dual_mul_f32 v87, v87, v205 :: v_dual_mul_f32 v96, v96, v205
	v_dual_mul_f32 v79, v79, v205 :: v_dual_mul_f32 v88, v88, v205
	v_dual_mul_f32 v71, v71, v205 :: v_dual_mul_f32 v80, v80, v205
	v_dual_mul_f32 v63, v63, v205 :: v_dual_mul_f32 v72, v72, v205
	v_dual_mul_f32 v55, v55, v205 :: v_dual_mul_f32 v64, v64, v205
	v_dual_mul_f32 v47, v47, v205 :: v_dual_mul_f32 v56, v56, v205
	v_dual_mul_f32 v39, v39, v205 :: v_dual_mul_f32 v48, v48, v205
	v_dual_mul_f32 v31, v31, v205 :: v_dual_mul_f32 v40, v40, v205
	v_dual_mul_f32 v23, v23, v205 :: v_dual_mul_f32 v32, v32, v205
	v_dual_mul_f32 v15, v15, v205 :: v_dual_mul_f32 v24, v24, v205
	v_dual_mul_f32 v111, v7, v205 :: v_dual_mul_f32 v16, v16, v205
	v_dual_mul_f32 v5, v112, v205 :: v_dual_mul_f32 v6, v113, v205
	v_dual_mul_f32 v112, v8, v205 :: v_dual_mul_f32 v105, v105, v205
	v_dual_mul_f32 v97, v97, v205 :: v_dual_mul_f32 v106, v106, v205
	v_dual_mul_f32 v89, v89, v205 :: v_dual_mul_f32 v98, v98, v205
	v_dual_mul_f32 v81, v81, v205 :: v_dual_mul_f32 v90, v90, v205
	v_dual_mul_f32 v73, v73, v205 :: v_dual_mul_f32 v82, v82, v205
	v_dual_mul_f32 v65, v65, v205 :: v_dual_mul_f32 v74, v74, v205
	v_dual_mul_f32 v57, v57, v205 :: v_dual_mul_f32 v66, v66, v205
	v_dual_mul_f32 v49, v49, v205 :: v_dual_mul_f32 v58, v58, v205
	v_dual_mul_f32 v41, v41, v205 :: v_dual_mul_f32 v50, v50, v205
	v_dual_mul_f32 v33, v33, v205 :: v_dual_mul_f32 v42, v42, v205
	v_dual_mul_f32 v25, v25, v205 :: v_dual_mul_f32 v34, v34, v205
	v_dual_mul_f32 v17, v17, v205 :: v_dual_mul_f32 v26, v26, v205
	v_mul_f32_e32 v7, v114, v205
	v_dual_mul_f32 v13, v13, v205 :: v_dual_mul_f32 v14, v14, v205
	v_dual_mul_f32 v113, v9, v205 :: v_dual_mul_f32 v18, v18, v205
	.loc	0 2151 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_clause 0x18
	global_store_b128 v[131:132], v[127:130], off offset:24
	global_store_b128 v[131:132], v[115:118], off offset:72
	global_store_b128 v[131:132], v[0:3], off offset:88
	global_store_b128 v[131:132], v[107:110], off offset:136
	global_store_b128 v[131:132], v[4:7], off offset:152
	global_store_b128 v[131:132], v[99:102], off offset:200
	global_store_b128 v[131:132], v[103:106], off offset:216
	global_store_b128 v[131:132], v[91:94], off offset:264
	global_store_b128 v[131:132], v[95:98], off offset:280
	global_store_b128 v[131:132], v[83:86], off offset:328
	global_store_b128 v[131:132], v[87:90], off offset:344
	global_store_b128 v[131:132], v[75:78], off offset:392
	global_store_b128 v[131:132], v[79:82], off offset:408
	global_store_b128 v[131:132], v[67:70], off offset:456
	global_store_b128 v[131:132], v[71:74], off offset:472
	global_store_b128 v[131:132], v[59:62], off offset:520
	global_store_b128 v[131:132], v[63:66], off offset:536
	global_store_b128 v[131:132], v[51:54], off offset:584
	global_store_b128 v[131:132], v[55:58], off offset:600
	global_store_b128 v[131:132], v[43:46], off offset:648
	global_store_b128 v[131:132], v[47:50], off offset:664
	global_store_b128 v[131:132], v[35:38], off offset:712
	global_store_b128 v[131:132], v[39:42], off offset:728
	global_store_b128 v[131:132], v[27:30], off offset:776
	global_store_b128 v[131:132], v[31:34], off offset:792
	.loc	0 2151 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	v_mul_f32_e32 v114, v10, v205
	.loc	0 2151 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2222:5 ]
	s_clause 0x5
	global_store_b128 v[131:132], v[19:22], off offset:840
	global_store_b128 v[131:132], v[23:26], off offset:856
	global_store_b128 v[131:132], v[11:14], off offset:904
	global_store_b128 v[131:132], v[15:18], off offset:920
	global_store_b128 v[131:132], v[123:126], off offset:968
	global_store_b128 v[131:132], v[111:114], off offset:984
.Ltmp623:
.LBB6_56:
	.loc	0 2226 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2226:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp624:
.Lfunc_end6:
	.size	attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201, .Lfunc_end6-attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
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
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 234
		.amdhsa_next_free_sgpr 26
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_vgpr, 234
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.numbered_sgpr, 26
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10260
; TotalNumSgprs: 28
; NumVgprs: 234
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 28
; NumVGPRsForWavesPerEU: 234
; Occupancy: 6
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
.Lfunc_begin7:
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.loc	0 2237 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2237:17
	s_load_b128 s[4:7], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s5, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s5, s7, -9
	.loc	0 2237 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2237:23
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s5, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB7_10
; %bb.1:
	.loc	0 2242 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2242:26
	v_lshrrev_b32_e32 v1, 5, v0
	.loc	0 2245 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2245:27
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2244 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2244:36
	v_lshl_or_b32 v4, ttmp9, 3, v1
	.loc	0 2245 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2245:13
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB7_10
; %bb.2:
	.loc	0 2237 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2237:17
	s_load_b128 s[0:3], s[0:1], 0x0
	.loc	0 2248 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2248:5
	v_mad_co_u64_u32 v[1:2], null, v4, s7, 0
	v_ashrrev_i32_e32 v5, 31, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[6:7], null, v5, s7, v[2:3]
	s_wait_kmcnt 0x0
	v_mad_co_u64_u32 v[1:2], null, 0x408, v1, s[0:1]
	s_mov_b32 s0, s7
	v_mad_co_u64_u32 v[2:3], null, 0x408, v6, v[2:3]
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v8, 0xff800000 :: v_dual_mov_b32 v7, v2
	v_mov_b32_e32 v6, v1
.LBB7_3:                                ; =>This Inner Loop Header: Depth=1
	.loc	0 2251 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2251:22
	global_load_b32 v3, v[6:7], off
.Ltmp625:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2251:13 ]
	v_max_num_f32_e32 v8, v8, v8
.Ltmp626:
	.loc	0 2248 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2248:23
	v_add_co_u32 v6, vcc_lo, 0x408, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
.Ltmp627:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2251:13 ]
	s_wait_loadcnt 0x0
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v8, v8, v3
.Ltmp628:
	.loc	0 2248 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2248:5
	s_cbranch_scc0 .LBB7_3
; %bb.4:
	.loc	0 2243 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2243:26
	v_and_b32_e32 v3, 31, v0
	v_lshlrev_b64_e32 v[5:6], 10, v[4:5]
	.loc	0 2253 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2253:5
	v_mov_b32_e32 v4, 0
	s_mov_b32 s1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v9, 2, v3
	v_add_co_u32 v0, vcc_lo, s2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s3, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v9, s0, v9, 8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, 0, s0
	s_branch .LBB7_6
.LBB7_5:                                ;   in Loop: Header=BB7_6 Depth=1
	.loc	0 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 2265 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2265:13
	v_div_scale_f32 v5, null, v11, v11, v12
	v_div_scale_f32 v14, vcc_lo, v12, v11, v12
	v_rcp_f32_e32 v6, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v13, -v5, v6, 1.0
	v_fmac_f32_e32 v6, v13, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v13, v14, v6
	v_fma_f32 v15, -v5, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v13, v15, v6
	v_fma_f32 v5, -v5, v13, v14
	.loc	0 2253 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2253:40
	v_add_nc_u32_e32 v14, 32, v3
	.loc	0 2265 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2265:13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v13, v5, v6, v13
	.loc	0 2264 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2264:9
	v_lshlrev_b64_e32 v[5:6], 2, v[3:4]
	.loc	0 2253 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2253:26
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v3
	.loc	0 2265 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2265:13
	v_div_fixup_f32 v3, v13, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	0 2264 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2264:9
	v_add_co_u32 v5, s0, v0, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v7, v6, s0
	.loc	0 2265 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2265:18
	v_cmp_lt_f32_e64 s0, 0, v11
	.loc	0 2253 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2253:5
	s_or_b32 s1, vcc_lo, s1
	.loc	0 2265 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2265:13
	v_cndmask_b32_e64 v11, 0, v3, s0
	.loc	0 2253 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2253:5
	v_add_co_u32 v9, s0, 0x80, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0
	v_mov_b32_e32 v3, v14
	.loc	0 2264 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2264:44
	global_store_b32 v[5:6], v11, off
	.loc	0 2253 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2253:5
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB7_10
.LBB7_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB7_8 Depth 2
	.loc	0 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_dual_mov_b32 v6, v2 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	s_mov_b32 s0, s7
	s_branch .LBB7_8
.LBB7_7:                                ;   in Loop: Header=BB7_8 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	.loc	0 2262 24 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2262:24
	v_add_co_u32 v15, vcc_lo, v5, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v6, v10, vcc_lo
	.loc	0 2256 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2256:27
	v_add_co_u32 v5, vcc_lo, 0x408, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	.loc	0 2262 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2262:24
	global_load_b32 v15, v[15:16], off
	.loc	0 2261 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2261:17
	v_fmac_f32_e32 v11, v13, v14
	.loc	0 2256 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2256:27
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	0 2262 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2262:17
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v12, v14, v15
	.loc	0 2256 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2256:9
	s_cbranch_scc1 .LBB7_5
.LBB7_8:                                ;   Parent Loop BB7_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	0 2260 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2260:18
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	.loc	0 2260 17 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2260:17
	s_mov_b32 s2, exec_lo
	.loc	0 2260 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2260:24
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	.loc	0 2260 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2260:17
	s_cbranch_execz .LBB7_7
; %bb.9:                                ;   in Loop: Header=BB7_8 Depth=2
	.loc	0 2260 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2260:41
	global_load_b32 v14, v[5:6], off
	.loc	0 2260 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2260:47
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp629:
	.loc	2 202 42 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2260:34 ]
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	.loc	2 202 10 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2260:34 ]
	v_exp_f32_e32 v14, v14
	s_branch .LBB7_7
.Ltmp630:
.LBB7_10:
	.loc	0 2267 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2267:1
	s_endpgm
.Ltmp631:
.Lfunc_end7:
	.size	attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201, .Lfunc_end7-attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
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
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end7-attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201)<<4)&4080)>>4
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.num_vgpr, 17
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.numbered_sgpr, 8
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 728
; TotalNumSgprs: 10
; NumVgprs: 17
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 10
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
	.type	__hip_cuid_5122e9c22abf15a3,@object ; @__hip_cuid_5122e9c22abf15a3
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5122e9c22abf15a3
__hip_cuid_5122e9c22abf15a3:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5122e9c22abf15a3, 1

	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	37                              ; DW_FORM_strx1
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	114                             ; DW_AT_str_offsets_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	37                              ; DW_FORM_strx1
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	115                             ; DW_AT_addr_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	116                             ; DW_AT_rnglists_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	32                              ; DW_AT_inline
	.byte	33                              ; DW_FORM_implicit_const
	.byte	1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	122                             ; DW_AT_call_all_calls
	.byte	25                              ; DW_FORM_flag_present
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	1                               ; Abbrev [1] 0xc:0xab6 DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	48                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.long	.Lstr_offsets_base0             ; DW_AT_str_offsets_base
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.byte	2                               ; DW_AT_comp_dir
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin0       ; DW_AT_high_pc
	.long	.Laddr_table_base0              ; DW_AT_addr_base
	.long	.Lrnglists_table_base0          ; DW_AT_rnglists_base
	.byte	2                               ; Abbrev [2] 0x27:0x2 DW_TAG_subprogram
	.byte	3                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x29:0x2 DW_TAG_subprogram
	.byte	4                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x2b:0x2 DW_TAG_subprogram
	.byte	5                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x2d:0x2 DW_TAG_subprogram
	.byte	6                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x2f:0x2 DW_TAG_subprogram
	.byte	7                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x31:0x2 DW_TAG_subprogram
	.byte	8                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x33:0x2 DW_TAG_subprogram
	.byte	9                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x35:0x2 DW_TAG_subprogram
	.byte	10                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x37:0x2 DW_TAG_subprogram
	.byte	11                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x39:0x2 DW_TAG_subprogram
	.byte	12                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x3b:0x2 DW_TAG_subprogram
	.byte	5                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x3d:0x227 DW_TAG_subprogram
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	20                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x44:0x21f DW_TAG_inlined_subroutine
	.long	39                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1342                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x4e:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	999                             ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x58:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_low_pc
	.long	.Ltmp4-.Ltmp3                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x67:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1001                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x71:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	3                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1003                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7b:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	4                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1000                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x85:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_low_pc
	.long	.Ltmp13-.Ltmp12                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1004                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x93:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	3                               ; DW_AT_low_pc
	.long	.Ltmp16-.Ltmp15                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	999                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa1:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	4                               ; DW_AT_low_pc
	.long	.Ltmp17-.Ltmp16                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1004                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xaf:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	5                               ; DW_AT_low_pc
	.long	.Ltmp20-.Ltmp19                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1005                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xbd:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	6                               ; DW_AT_low_pc
	.long	.Ltmp23-.Ltmp22                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1000                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xcb:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	7                               ; DW_AT_low_pc
	.long	.Ltmp24-.Ltmp23                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1005                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xd9:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	5                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1002                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xe3:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	8                               ; DW_AT_low_pc
	.long	.Ltmp27-.Ltmp26                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1006                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xf1:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	9                               ; DW_AT_low_pc
	.long	.Ltmp33-.Ltmp32                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1001                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xff:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	10                              ; DW_AT_low_pc
	.long	.Ltmp34-.Ltmp33                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1006                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x10d:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	11                              ; DW_AT_low_pc
	.long	.Ltmp36-.Ltmp35                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1007                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x11b:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	12                              ; DW_AT_low_pc
	.long	.Ltmp38-.Ltmp37                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1002                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x129:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	13                              ; DW_AT_low_pc
	.long	.Ltmp39-.Ltmp38                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1007                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x137:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	14                              ; DW_AT_low_pc
	.long	.Ltmp41-.Ltmp40                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x145:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	15                              ; DW_AT_low_pc
	.long	.Ltmp42-.Ltmp41                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1003                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x153:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	16                              ; DW_AT_low_pc
	.long	.Ltmp43-.Ltmp42                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x161:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	17                              ; DW_AT_low_pc
	.long	.Ltmp45-.Ltmp44                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1263                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x16f:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	17                              ; DW_AT_low_pc
	.long	.Ltmp45-.Ltmp44                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x17d:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	17                              ; DW_AT_low_pc
	.long	.Ltmp45-.Ltmp44                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x18d:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	18                              ; DW_AT_low_pc
	.long	.Ltmp47-.Ltmp46                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1090                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x19b:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	18                              ; DW_AT_low_pc
	.long	.Ltmp47-.Ltmp46                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x1a9:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	18                              ; DW_AT_low_pc
	.long	.Ltmp47-.Ltmp46                 ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x1b9:0xa DW_TAG_inlined_subroutine
	.long	55                              ; DW_AT_abstract_origin
	.byte	6                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1172                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x1c3:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	7                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1185                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x1cd:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_AT_low_pc
	.long	.Ltmp70-.Ltmp69                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1186                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x1db:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_AT_low_pc
	.long	.Ltmp70-.Ltmp69                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x1ea:0xa DW_TAG_inlined_subroutine
	.long	55                              ; DW_AT_abstract_origin
	.byte	8                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1198                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x1f4:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	20                              ; DW_AT_low_pc
	.long	.Ltmp73-.Ltmp72                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1187                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x202:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	9                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1219                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x20c:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	21                              ; DW_AT_low_pc
	.long	.Ltmp83-.Ltmp82                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1220                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x21a:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	21                              ; DW_AT_low_pc
	.long	.Ltmp83-.Ltmp82                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x229:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	22                              ; DW_AT_low_pc
	.long	.Ltmp85-.Ltmp84                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1209                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x237:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	22                              ; DW_AT_low_pc
	.long	.Ltmp85-.Ltmp84                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x246:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	23                              ; DW_AT_low_pc
	.long	.Ltmp86-.Ltmp85                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1220                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x254:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	24                              ; DW_AT_low_pc
	.long	.Ltmp88-.Ltmp87                 ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1223                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x264:0x2 DW_TAG_subprogram
	.byte	13                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x266:0x2 DW_TAG_subprogram
	.byte	14                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x268:0x1d DW_TAG_subprogram
	.byte	25                              ; DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	21                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x26f:0x15 DW_TAG_inlined_subroutine
	.long	614                             ; DW_AT_abstract_origin
	.byte	10                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1423                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x279:0xa DW_TAG_inlined_subroutine
	.long	612                             ; DW_AT_abstract_origin
	.byte	10                              ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	299                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	3                               ; Abbrev [3] 0x285:0x113 DW_TAG_subprogram
	.byte	26                              ; DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	22                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x28c:0x15 DW_TAG_inlined_subroutine
	.long	614                             ; DW_AT_abstract_origin
	.byte	11                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1471                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x296:0xa DW_TAG_inlined_subroutine
	.long	612                             ; DW_AT_abstract_origin
	.byte	11                              ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	299                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x2a1:0x24 DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	12                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1492                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x2ab:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	12                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x2b5:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	27                              ; DW_AT_low_pc
	.long	.Ltmp100-.Ltmp99                ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x2c5:0x15 DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	13                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1493                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x2cf:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	13                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x2da:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	28                              ; DW_AT_low_pc
	.long	.Ltmp104-.Ltmp103               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1485                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x2e8:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	29                              ; DW_AT_low_pc
	.long	.Ltmp107-.Ltmp106               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1487                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x2f6:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	30                              ; DW_AT_low_pc
	.long	.Ltmp108-.Ltmp107               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1489                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x304:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	31                              ; DW_AT_low_pc
	.long	.Ltmp109-.Ltmp108               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1491                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x312:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	32                              ; DW_AT_low_pc
	.long	.Ltmp111-.Ltmp110               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1492                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x320:0x15 DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	14                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1494                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x32a:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	14                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x335:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	33                              ; DW_AT_low_pc
	.long	.Ltmp114-.Ltmp113               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1493                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x343:0x15 DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	15                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1495                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x34d:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	15                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x358:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_low_pc
	.long	.Ltmp117-.Ltmp116               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1494                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x366:0x15 DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	16                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1496                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x370:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	16                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x37b:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	35                              ; DW_AT_low_pc
	.long	.Ltmp120-.Ltmp119               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1495                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x389:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	36                              ; DW_AT_low_pc
	.long	.Ltmp122-.Ltmp121               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1496                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x398:0x2 DW_TAG_subprogram
	.byte	15                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x39a:0x227 DW_TAG_subprogram
	.byte	37                              ; DW_AT_low_pc
	.long	.Lfunc_end3-.Lfunc_begin3       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	23                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x3a1:0x21f DW_TAG_inlined_subroutine
	.long	920                             ; DW_AT_abstract_origin
	.byte	17                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1585                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x3ab:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	18                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	999                             ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x3b5:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	38                              ; DW_AT_low_pc
	.long	.Ltmp130-.Ltmp129               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x3c4:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1000                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x3ce:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	20                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1003                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x3d8:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_low_pc
	.long	.Ltmp142-.Ltmp141               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1004                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x3e6:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	40                              ; DW_AT_low_pc
	.long	.Ltmp146-.Ltmp145               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	999                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x3f4:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	21                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1001                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x3fe:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	41                              ; DW_AT_low_pc
	.long	.Ltmp149-.Ltmp148               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1004                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x40c:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	22                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1002                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x416:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	42                              ; DW_AT_low_pc
	.long	.Ltmp153-.Ltmp152               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1005                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x424:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_low_pc
	.long	.Ltmp157-.Ltmp156               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1000                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x432:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	44                              ; DW_AT_low_pc
	.long	.Ltmp158-.Ltmp157               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1005                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x440:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	45                              ; DW_AT_low_pc
	.long	.Ltmp160-.Ltmp159               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1006                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x44e:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	46                              ; DW_AT_low_pc
	.long	.Ltmp167-.Ltmp166               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1001                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x45c:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	47                              ; DW_AT_low_pc
	.long	.Ltmp168-.Ltmp167               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1006                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x46a:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	48                              ; DW_AT_low_pc
	.long	.Ltmp170-.Ltmp169               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1007                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x478:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	49                              ; DW_AT_low_pc
	.long	.Ltmp172-.Ltmp171               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1002                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x486:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	50                              ; DW_AT_low_pc
	.long	.Ltmp173-.Ltmp172               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1007                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x494:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	51                              ; DW_AT_low_pc
	.long	.Ltmp176-.Ltmp175               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4a2:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	52                              ; DW_AT_low_pc
	.long	.Ltmp181-.Ltmp180               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1003                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4b0:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	53                              ; DW_AT_low_pc
	.long	.Ltmp182-.Ltmp181               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x4be:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_low_pc
	.long	.Ltmp186-.Ltmp185               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1263                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x4cc:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_low_pc
	.long	.Ltmp186-.Ltmp185               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4da:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_low_pc
	.long	.Ltmp186-.Ltmp185               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x4ea:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_low_pc
	.long	.Ltmp188-.Ltmp187               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1090                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x4f8:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_low_pc
	.long	.Ltmp188-.Ltmp187               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x506:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_low_pc
	.long	.Ltmp188-.Ltmp187               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x516:0xa DW_TAG_inlined_subroutine
	.long	55                              ; DW_AT_abstract_origin
	.byte	23                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1172                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x520:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	24                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1185                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x52a:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_low_pc
	.long	.Ltmp211-.Ltmp210               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1186                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x538:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_low_pc
	.long	.Ltmp211-.Ltmp210               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x547:0xa DW_TAG_inlined_subroutine
	.long	55                              ; DW_AT_abstract_origin
	.byte	25                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1198                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x551:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	57                              ; DW_AT_low_pc
	.long	.Ltmp214-.Ltmp213               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1187                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x55f:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	26                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1219                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x569:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	58                              ; DW_AT_low_pc
	.long	.Ltmp224-.Ltmp223               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1220                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x577:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	58                              ; DW_AT_low_pc
	.long	.Ltmp224-.Ltmp223               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x586:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	59                              ; DW_AT_low_pc
	.long	.Ltmp226-.Ltmp225               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1209                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x594:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	59                              ; DW_AT_low_pc
	.long	.Ltmp226-.Ltmp225               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x5a3:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	60                              ; DW_AT_low_pc
	.long	.Ltmp227-.Ltmp226               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1220                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x5b1:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	61                              ; DW_AT_low_pc
	.long	.Ltmp229-.Ltmp228               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1223                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x5c1:0x2 DW_TAG_subprogram
	.byte	16                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x5c3:0x20 DW_TAG_subprogram
	.byte	62                              ; DW_AT_low_pc
	.long	.Lfunc_end4-.Lfunc_begin4       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	24                              ; DW_AT_name
	.byte	6                               ; Abbrev [6] 0x5ca:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	27                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1662                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x5d4:0xe DW_TAG_inlined_subroutine
	.long	1473                            ; DW_AT_abstract_origin
	.byte	63                              ; DW_AT_low_pc
	.long	.Ltmp239-.Ltmp238               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1671                            ; DW_AT_call_line
	.byte	34                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x5e3:0x2 DW_TAG_subprogram
	.byte	17                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x5e5:0x2 DW_TAG_subprogram
	.byte	18                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x5e7:0x27e DW_TAG_subprogram
	.byte	64                              ; DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	25                              ; DW_AT_name
	.byte	5                               ; Abbrev [5] 0x5ee:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	65                              ; DW_AT_low_pc
	.long	.Ltmp242-.Ltmp241               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2183                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x5fc:0x268 DW_TAG_inlined_subroutine
	.long	1507                            ; DW_AT_abstract_origin
	.byte	28                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2187                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x606:0xa DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	29                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1816                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x610:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	30                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1808                            ; DW_AT_call_line
	.byte	21                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x61a:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	31                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1810                            ; DW_AT_call_line
	.byte	21                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x624:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	32                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1812                            ; DW_AT_call_line
	.byte	21                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x62e:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	33                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1814                            ; DW_AT_call_line
	.byte	21                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x638:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	66                              ; DW_AT_low_pc
	.long	.Ltmp314-.Ltmp313               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1816                            ; DW_AT_call_line
	.byte	17                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x646:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	67                              ; DW_AT_low_pc
	.long	.Ltmp316-.Ltmp315               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1856                            ; DW_AT_call_line
	.byte	15                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x654:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	68                              ; DW_AT_low_pc
	.long	.Ltmp317-.Ltmp316               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1857                            ; DW_AT_call_line
	.byte	15                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x662:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x66c:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	69                              ; DW_AT_low_pc
	.long	.Ltmp319-.Ltmp318               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x67b:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	35                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x685:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	70                              ; DW_AT_low_pc
	.long	.Ltmp327-.Ltmp326               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1864                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x693:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	71                              ; DW_AT_low_pc
	.long	.Ltmp329-.Ltmp328               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6a1:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	72                              ; DW_AT_low_pc
	.long	.Ltmp330-.Ltmp329               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1864                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6af:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	73                              ; DW_AT_low_pc
	.long	.Ltmp332-.Ltmp331               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1865                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x6bd:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	36                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1861                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6c7:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	74                              ; DW_AT_low_pc
	.long	.Ltmp334-.Ltmp333               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6d5:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	75                              ; DW_AT_low_pc
	.long	.Ltmp335-.Ltmp334               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1865                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6e3:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	76                              ; DW_AT_low_pc
	.long	.Ltmp337-.Ltmp336               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1866                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x6f1:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	37                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1862                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x6fb:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	38                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1863                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x705:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	77                              ; DW_AT_low_pc
	.long	.Ltmp340-.Ltmp339               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1861                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x713:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	78                              ; DW_AT_low_pc
	.long	.Ltmp341-.Ltmp340               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1866                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x721:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	79                              ; DW_AT_low_pc
	.long	.Ltmp344-.Ltmp343               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1867                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x72f:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	80                              ; DW_AT_low_pc
	.long	.Ltmp346-.Ltmp345               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1862                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x73d:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	81                              ; DW_AT_low_pc
	.long	.Ltmp347-.Ltmp346               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1867                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x74b:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	82                              ; DW_AT_low_pc
	.long	.Ltmp349-.Ltmp348               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1868                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x759:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	83                              ; DW_AT_low_pc
	.long	.Ltmp350-.Ltmp349               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1863                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x767:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	84                              ; DW_AT_low_pc
	.long	.Ltmp351-.Ltmp350               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1868                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x775:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2112                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x77f:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x789:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x795:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	85                              ; DW_AT_low_pc
	.long	.Ltmp357-.Ltmp356               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1922                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x7a3:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	85                              ; DW_AT_low_pc
	.long	.Ltmp357-.Ltmp356               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7b1:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	85                              ; DW_AT_low_pc
	.long	.Ltmp357-.Ltmp356               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x7c1:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	40                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1944                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7cb:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	41                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1946                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7d5:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	42                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1948                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7df:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1950                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x7e9:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	44                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1977                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x7f3:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	44                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7fd:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	44                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x809:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	45                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2047                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x813:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	86                              ; DW_AT_low_pc
	.long	.Ltmp446-.Ltmp445               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2048                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x821:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	87                              ; DW_AT_low_pc
	.long	.Ltmp447-.Ltmp446               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2049                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x82f:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	46                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2070                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x839:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	88                              ; DW_AT_low_pc
	.long	.Ltmp455-.Ltmp454               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2071                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x847:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	89                              ; DW_AT_low_pc
	.long	.Ltmp457-.Ltmp456               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2065                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x855:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	90                              ; DW_AT_low_pc
	.long	.Ltmp459-.Ltmp458               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2074                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x865:0x2 DW_TAG_subprogram
	.byte	19                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x867:0x23a DW_TAG_subprogram
	.byte	91                              ; DW_AT_low_pc
	.long	.Lfunc_end6-.Lfunc_begin6       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	26                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x86e:0x232 DW_TAG_inlined_subroutine
	.long	2149                            ; DW_AT_abstract_origin
	.byte	47                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2222                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x878:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	92                              ; DW_AT_low_pc
	.long	.Ltmp466-.Ltmp465               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1856                            ; DW_AT_call_line
	.byte	15                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x886:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	93                              ; DW_AT_low_pc
	.long	.Ltmp467-.Ltmp466               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1857                            ; DW_AT_call_line
	.byte	15                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x894:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	48                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x89e:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	94                              ; DW_AT_low_pc
	.long	.Ltmp469-.Ltmp468               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x8ad:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	49                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8b7:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	95                              ; DW_AT_low_pc
	.long	.Ltmp485-.Ltmp484               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1864                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8c5:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	96                              ; DW_AT_low_pc
	.long	.Ltmp487-.Ltmp486               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8d3:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	97                              ; DW_AT_low_pc
	.long	.Ltmp488-.Ltmp487               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1864                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8e1:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	98                              ; DW_AT_low_pc
	.long	.Ltmp490-.Ltmp489               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1865                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x8ef:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	50                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1861                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8f9:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	99                              ; DW_AT_low_pc
	.long	.Ltmp492-.Ltmp491               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x907:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	100                             ; DW_AT_low_pc
	.long	.Ltmp493-.Ltmp492               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1865                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x915:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	101                             ; DW_AT_low_pc
	.long	.Ltmp495-.Ltmp494               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1866                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x923:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	51                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1862                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x92d:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	52                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1863                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x937:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	102                             ; DW_AT_low_pc
	.long	.Ltmp498-.Ltmp497               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1861                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x945:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	103                             ; DW_AT_low_pc
	.long	.Ltmp499-.Ltmp498               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1866                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x953:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	104                             ; DW_AT_low_pc
	.long	.Ltmp502-.Ltmp501               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1867                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x961:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	105                             ; DW_AT_low_pc
	.long	.Ltmp504-.Ltmp503               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1862                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x96f:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	106                             ; DW_AT_low_pc
	.long	.Ltmp505-.Ltmp504               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1867                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x97d:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	107                             ; DW_AT_low_pc
	.long	.Ltmp507-.Ltmp506               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1868                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x98b:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	108                             ; DW_AT_low_pc
	.long	.Ltmp508-.Ltmp507               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1863                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x999:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	109                             ; DW_AT_low_pc
	.long	.Ltmp510-.Ltmp509               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1868                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x9a7:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	110                             ; DW_AT_low_pc
	.long	.Ltmp512-.Ltmp511               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1873                            ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x9b5:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	53                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2112                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x9bf:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	53                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9c9:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	53                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x9d5:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	111                             ; DW_AT_low_pc
	.long	.Ltmp518-.Ltmp517               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1922                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x9e3:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	111                             ; DW_AT_low_pc
	.long	.Ltmp518-.Ltmp517               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x9f1:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	111                             ; DW_AT_low_pc
	.long	.Ltmp518-.Ltmp517               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0xa01:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1944                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa0b:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1946                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa15:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1948                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa1f:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	57                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1950                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xa29:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	58                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1977                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xa33:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	58                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa3d:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	58                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0xa49:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	59                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2047                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa53:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	112                             ; DW_AT_low_pc
	.long	.Ltmp605-.Ltmp604               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2048                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa61:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	113                             ; DW_AT_low_pc
	.long	.Ltmp606-.Ltmp605               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2049                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa6f:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	60                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2070                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa79:0xa DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	61                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2071                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa83:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	114                             ; DW_AT_low_pc
	.long	.Ltmp620-.Ltmp619               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2065                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa91:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	115                             ; DW_AT_low_pc
	.long	.Ltmp622-.Ltmp621               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2074                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	3                               ; Abbrev [3] 0xaa1:0x20 DW_TAG_subprogram
	.byte	116                             ; DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin7       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	27                              ; DW_AT_name
	.byte	6                               ; Abbrev [6] 0xaa8:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	62                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2251                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xab2:0xe DW_TAG_inlined_subroutine
	.long	1473                            ; DW_AT_abstract_origin
	.byte	117                             ; DW_AT_low_pc
	.long	.Ltmp630-.Ltmp629               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2260                            ; DW_AT_call_line
	.byte	34                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 ; Length
.Ldebug_list_header_start0:
	.short	5                               ; Version
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
	.long	63                              ; Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
	.long	.Ldebug_ranges1-.Lrnglists_table_base0
	.long	.Ldebug_ranges2-.Lrnglists_table_base0
	.long	.Ldebug_ranges3-.Lrnglists_table_base0
	.long	.Ldebug_ranges4-.Lrnglists_table_base0
	.long	.Ldebug_ranges5-.Lrnglists_table_base0
	.long	.Ldebug_ranges6-.Lrnglists_table_base0
	.long	.Ldebug_ranges7-.Lrnglists_table_base0
	.long	.Ldebug_ranges8-.Lrnglists_table_base0
	.long	.Ldebug_ranges9-.Lrnglists_table_base0
	.long	.Ldebug_ranges10-.Lrnglists_table_base0
	.long	.Ldebug_ranges11-.Lrnglists_table_base0
	.long	.Ldebug_ranges12-.Lrnglists_table_base0
	.long	.Ldebug_ranges13-.Lrnglists_table_base0
	.long	.Ldebug_ranges14-.Lrnglists_table_base0
	.long	.Ldebug_ranges15-.Lrnglists_table_base0
	.long	.Ldebug_ranges16-.Lrnglists_table_base0
	.long	.Ldebug_ranges17-.Lrnglists_table_base0
	.long	.Ldebug_ranges18-.Lrnglists_table_base0
	.long	.Ldebug_ranges19-.Lrnglists_table_base0
	.long	.Ldebug_ranges20-.Lrnglists_table_base0
	.long	.Ldebug_ranges21-.Lrnglists_table_base0
	.long	.Ldebug_ranges22-.Lrnglists_table_base0
	.long	.Ldebug_ranges23-.Lrnglists_table_base0
	.long	.Ldebug_ranges24-.Lrnglists_table_base0
	.long	.Ldebug_ranges25-.Lrnglists_table_base0
	.long	.Ldebug_ranges26-.Lrnglists_table_base0
	.long	.Ldebug_ranges27-.Lrnglists_table_base0
	.long	.Ldebug_ranges28-.Lrnglists_table_base0
	.long	.Ldebug_ranges29-.Lrnglists_table_base0
	.long	.Ldebug_ranges30-.Lrnglists_table_base0
	.long	.Ldebug_ranges31-.Lrnglists_table_base0
	.long	.Ldebug_ranges32-.Lrnglists_table_base0
	.long	.Ldebug_ranges33-.Lrnglists_table_base0
	.long	.Ldebug_ranges34-.Lrnglists_table_base0
	.long	.Ldebug_ranges35-.Lrnglists_table_base0
	.long	.Ldebug_ranges36-.Lrnglists_table_base0
	.long	.Ldebug_ranges37-.Lrnglists_table_base0
	.long	.Ldebug_ranges38-.Lrnglists_table_base0
	.long	.Ldebug_ranges39-.Lrnglists_table_base0
	.long	.Ldebug_ranges40-.Lrnglists_table_base0
	.long	.Ldebug_ranges41-.Lrnglists_table_base0
	.long	.Ldebug_ranges42-.Lrnglists_table_base0
	.long	.Ldebug_ranges43-.Lrnglists_table_base0
	.long	.Ldebug_ranges44-.Lrnglists_table_base0
	.long	.Ldebug_ranges45-.Lrnglists_table_base0
	.long	.Ldebug_ranges46-.Lrnglists_table_base0
	.long	.Ldebug_ranges47-.Lrnglists_table_base0
	.long	.Ldebug_ranges48-.Lrnglists_table_base0
	.long	.Ldebug_ranges49-.Lrnglists_table_base0
	.long	.Ldebug_ranges50-.Lrnglists_table_base0
	.long	.Ldebug_ranges51-.Lrnglists_table_base0
	.long	.Ldebug_ranges52-.Lrnglists_table_base0
	.long	.Ldebug_ranges53-.Lrnglists_table_base0
	.long	.Ldebug_ranges54-.Lrnglists_table_base0
	.long	.Ldebug_ranges55-.Lrnglists_table_base0
	.long	.Ldebug_ranges56-.Lrnglists_table_base0
	.long	.Ldebug_ranges57-.Lrnglists_table_base0
	.long	.Ldebug_ranges58-.Lrnglists_table_base0
	.long	.Ldebug_ranges59-.Lrnglists_table_base0
	.long	.Ldebug_ranges60-.Lrnglists_table_base0
	.long	.Ldebug_ranges61-.Lrnglists_table_base0
	.long	.Ldebug_ranges62-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp0-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp1-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp2-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp91-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges1:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp3-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp4-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp5-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp6-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp7-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp8-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp9-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp10-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp11-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp12-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges2:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp6-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp7-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp13-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp14-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp17-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp18-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp20-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp21-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp25-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp26-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges3:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp8-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp9-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp28-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp29-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp30-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp31-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp39-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp40-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges4:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp10-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp11-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp14-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp15-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp18-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp19-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges5:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp24-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp25-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp27-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp28-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp31-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp32-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp34-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp35-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges6:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp48-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp49-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp50-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp51-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp52-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp53-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp54-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp55-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp56-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp57-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp58-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp59-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp60-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp61-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp62-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp63-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges7:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp64-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp65-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp66-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp67-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp68-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp69-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges8:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp70-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp71-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp89-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp90-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges9:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp74-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp75-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp76-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp77-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp78-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp79-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp80-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp81-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges10:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Lfunc_begin1-.Lfunc_begin0    ;   starting offset
	.uleb128 .Ltmp93-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp94-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp95-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges11:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Lfunc_begin2-.Lfunc_begin0    ;   starting offset
	.uleb128 .Ltmp97-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp98-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp99-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp123-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp124-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges12:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp99-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp100-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp104-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp105-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp109-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp110-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges13:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp101-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp102-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp105-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp106-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp111-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp112-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges14:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp112-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp113-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp114-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp115-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges15:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp115-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp116-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp117-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp118-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges16:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp118-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp119-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp120-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp121-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges17:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp126-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp127-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp128-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp136-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp137-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp139-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp140-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp143-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp144-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp163-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp164-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp165-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp166-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp170-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp171-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp176-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp177-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp178-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp179-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp183-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp184-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp232-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges18:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp129-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp130-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp131-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp132-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp134-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp135-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp137-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp138-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp140-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp141-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges19:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp132-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp133-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp138-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp139-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp144-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp145-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp147-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp148-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp151-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp152-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges20:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp135-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp136-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp155-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp156-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp162-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp163-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp174-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp175-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges21:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp146-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp147-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp149-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp150-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp153-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp154-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp158-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp159-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges22:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp150-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp151-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp154-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp155-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp160-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp161-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp168-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp169-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges23:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp189-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp190-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp191-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp192-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp193-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp194-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp195-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp196-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp197-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp198-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp199-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp200-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp201-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp202-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp203-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp204-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges24:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp205-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp206-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp207-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp208-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp209-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp210-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges25:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp211-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp212-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp230-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp231-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges26:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp215-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp216-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp217-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp218-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp219-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp220-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp221-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp222-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges27:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp234-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp235-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp236-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp237-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges28:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp243-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp244-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp245-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp460-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges29:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp246-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp247-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp312-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp313-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges30:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp248-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp249-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp252-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp253-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp256-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp257-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp260-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp261-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp264-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp265-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp268-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp269-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp272-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp273-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp276-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp277-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp280-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp281-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp284-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp285-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp288-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp289-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp292-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp293-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp296-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp297-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp300-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp301-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp304-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp305-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp308-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp309-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges31:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp249-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp250-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp253-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp254-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp257-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp258-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp261-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp262-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp265-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp266-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp269-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp270-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp273-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp274-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp277-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp278-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp281-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp282-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp285-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp286-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp289-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp290-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp293-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp294-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp297-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp298-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp301-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp302-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp305-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp306-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp309-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp310-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges32:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp250-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp251-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp254-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp255-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp258-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp259-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp262-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp263-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp266-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp267-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp270-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp271-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp274-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp275-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp278-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp279-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp282-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp283-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp286-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp287-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp290-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp291-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp294-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp295-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp298-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp299-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp302-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp303-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp306-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp307-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp310-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp311-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges33:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp251-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp252-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp255-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp256-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp259-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp260-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp263-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp264-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp267-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp268-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp271-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp272-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp275-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp276-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp279-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp280-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp283-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp284-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp287-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp288-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp291-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp292-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp295-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp296-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp299-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp300-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp303-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp304-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp307-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp308-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp311-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp312-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges34:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp318-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp320-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp321-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp322-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp323-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp324-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp325-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp326-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges35:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp320-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp321-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp322-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp323-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp324-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp325-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp327-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp328-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp330-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp331-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges36:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp332-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp333-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp335-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp336-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges37:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp337-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp338-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp342-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp343-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges38:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp338-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp339-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp341-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp342-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp344-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp345-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp347-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp348-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges39:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp352-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp353-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp354-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp355-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges40:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp358-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp359-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp360-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp361-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp368-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp369-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp370-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp371-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp379-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp380-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp386-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp387-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp388-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp389-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp396-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp397-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp398-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp399-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp406-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp407-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp408-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp409-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp416-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp417-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp418-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp419-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp426-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp427-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp428-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp429-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges41:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp359-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp360-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp361-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp362-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp369-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp370-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp371-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp372-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp378-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp379-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp387-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp388-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp389-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp390-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp397-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp398-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp399-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp400-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp407-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp408-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp409-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp410-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp417-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp418-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp419-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp420-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp427-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp428-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp429-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp430-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges42:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp363-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp364-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp365-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp366-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp373-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp374-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp375-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp376-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp384-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp385-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp391-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp392-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp393-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp394-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp401-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp402-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp403-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp404-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp411-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp412-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp413-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp414-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp421-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp422-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp423-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp424-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp431-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp432-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp433-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp434-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges43:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp364-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp365-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp366-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp367-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp374-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp375-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp376-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp377-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp381-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp382-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp383-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp384-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp392-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp393-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp394-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp395-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp402-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp403-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp404-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp405-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp412-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp413-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp414-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp415-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp422-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp423-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp424-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp425-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp432-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp433-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp434-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp435-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges44:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp436-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp437-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp438-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp439-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges45:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp440-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp441-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp442-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp443-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp444-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp445-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges46:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp448-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp449-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp450-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp451-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp452-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp453-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges47:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp462-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp463-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp464-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp469-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp470-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp471-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp472-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp474-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp475-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp476-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp477-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp479-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp480-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp481-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp482-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp485-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp486-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp623-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges48:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp468-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp469-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp472-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp473-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp475-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp476-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp477-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp478-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp483-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp484-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges49:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp473-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp474-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp478-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp479-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp480-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp481-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp482-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp483-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp488-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp489-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges50:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp490-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp491-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp493-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp494-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges51:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp495-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp496-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp500-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp501-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges52:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp496-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp497-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp499-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp500-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp502-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp503-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp505-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp506-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges53:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp513-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp514-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp515-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp516-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges54:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp519-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp520-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp521-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp522-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp529-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp530-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp531-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp532-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp540-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp541-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp547-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp548-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp549-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp550-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp557-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp558-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp559-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp560-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp567-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp568-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp569-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp570-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp577-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp578-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp579-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp580-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp587-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp588-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp589-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp590-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges55:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp520-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp521-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp522-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp523-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp530-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp531-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp532-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp533-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp539-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp540-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp548-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp549-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp550-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp551-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp558-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp559-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp560-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp561-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp568-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp569-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp570-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp571-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp578-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp579-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp580-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp581-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp588-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp589-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp590-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp591-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges56:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp524-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp525-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp526-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp527-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp534-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp535-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp536-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp537-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp545-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp546-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp552-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp553-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp554-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp555-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp562-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp563-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp564-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp565-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp572-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp573-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp574-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp575-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp582-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp583-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp584-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp585-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp592-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp593-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp594-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp595-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges57:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp525-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp526-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp527-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp528-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp535-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp536-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp537-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp538-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp542-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp543-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp544-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp545-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp553-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp554-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp555-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp556-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp563-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp564-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp565-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp566-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp573-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp574-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp575-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp576-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp583-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp584-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp585-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp586-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp593-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp594-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp595-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp596-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges58:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp597-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp598-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp599-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp600-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges59:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp601-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp602-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp603-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp604-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges60:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp607-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp608-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp609-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp610-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp611-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp612-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp613-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp614-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges61:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp615-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp616-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp617-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp618-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges62:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp625-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp626-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp627-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp628-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	116                             ; Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)" ; string offset=0 ; AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)
.Linfo_string1:
	.asciz	"kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip" ; string offset=112 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip
.Linfo_string2:
	.asciz	"/home/kaden/ClaudeCode/warpfront/wt-attnlane" ; string offset=159 ; /home/kaden/ClaudeCode/warpfront/wt-attnlane
.Linfo_string3:
	.asciz	"fa2_stageb_nbody<false>"       ; string offset=204 ; fa2_stageb_nbody<false>
.Linfo_string4:
	.asciz	"__lane_id"                     ; string offset=228 ; __lane_id
.Linfo_string5:
	.asciz	"__shfl_xor"                    ; string offset=238 ; __shfl_xor
.Linfo_string6:
	.asciz	"max"                           ; string offset=249 ; max
.Linfo_string7:
	.asciz	"min"                           ; string offset=253 ; min
.Linfo_string8:
	.asciz	"__work_group_barrier"          ; string offset=257 ; __work_group_barrier
.Linfo_string9:
	.asciz	"__barrier"                     ; string offset=278 ; __barrier
.Linfo_string10:
	.asciz	"__syncthreads"                 ; string offset=288 ; __syncthreads
.Linfo_string11:
	.asciz	"fa2_scale_n"                   ; string offset=302 ; fa2_scale_n
.Linfo_string12:
	.asciz	"fmaxf"                         ; string offset=314 ; fmaxf
.Linfo_string13:
	.asciz	"__hip_get_thread_idx_x"        ; string offset=320 ; __hip_get_thread_idx_x
.Linfo_string14:
	.asciz	"__get_x"                       ; string offset=343 ; __get_x
.Linfo_string15:
	.asciz	"fa2_stageb_nbody<true>"        ; string offset=351 ; fa2_stageb_nbody<true>
.Linfo_string16:
	.asciz	"__expf"                        ; string offset=374 ; __expf
.Linfo_string17:
	.asciz	"fa2_stageb_packet_body<false, true>" ; string offset=381 ; fa2_stageb_packet_body<false, true>
.Linfo_string18:
	.asciz	"fa2_pkt_xor16"                 ; string offset=417 ; fa2_pkt_xor16
.Linfo_string19:
	.asciz	"fa2_stageb_packet_body<true, false>" ; string offset=431 ; fa2_stageb_packet_body<true, false>
.Linfo_string20:
	.asciz	"attention_fp8_e4m3_fa2_gqa_gfx1201" ; string offset=467 ; attention_fp8_e4m3_fa2_gqa_gfx1201
.Linfo_string21:
	.asciz	"attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201" ; string offset=502 ; attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
.Linfo_string22:
	.asciz	"attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201" ; string offset=550 ; attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
.Linfo_string23:
	.asciz	"attention_fp8_e4m3_fa2_gqa_partial_gfx1201" ; string offset=598 ; attention_fp8_e4m3_fa2_gqa_partial_gfx1201
.Linfo_string24:
	.asciz	"attention_fp8_e4m3_fa2_gqa_merge_gfx1201" ; string offset=641 ; attention_fp8_e4m3_fa2_gqa_merge_gfx1201
.Linfo_string25:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_gfx1201" ; string offset=682 ; attention_fp8_e4m3_fa2_gqa_packet_gfx1201
.Linfo_string26:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201" ; string offset=724 ; attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
.Linfo_string27:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201" ; string offset=774 ; attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string13
	.long	.Linfo_string14
	.long	.Linfo_string15
	.long	.Linfo_string16
	.long	.Linfo_string17
	.long	.Linfo_string18
	.long	.Linfo_string19
	.long	.Linfo_string20
	.long	.Linfo_string21
	.long	.Linfo_string22
	.long	.Linfo_string23
	.long	.Linfo_string24
	.long	.Linfo_string25
	.long	.Linfo_string26
	.long	.Linfo_string27
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 ; Length of contribution
.Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp3
	.quad	.Ltmp12
	.quad	.Ltmp15
	.quad	.Ltmp16
	.quad	.Ltmp19
	.quad	.Ltmp22
	.quad	.Ltmp23
	.quad	.Ltmp26
	.quad	.Ltmp32
	.quad	.Ltmp33
	.quad	.Ltmp35
	.quad	.Ltmp37
	.quad	.Ltmp38
	.quad	.Ltmp40
	.quad	.Ltmp41
	.quad	.Ltmp42
	.quad	.Ltmp44
	.quad	.Ltmp46
	.quad	.Ltmp69
	.quad	.Ltmp72
	.quad	.Ltmp82
	.quad	.Ltmp84
	.quad	.Ltmp85
	.quad	.Ltmp87
	.quad	.Lfunc_begin1
	.quad	.Lfunc_begin2
	.quad	.Ltmp99
	.quad	.Ltmp103
	.quad	.Ltmp106
	.quad	.Ltmp107
	.quad	.Ltmp108
	.quad	.Ltmp110
	.quad	.Ltmp113
	.quad	.Ltmp116
	.quad	.Ltmp119
	.quad	.Ltmp121
	.quad	.Lfunc_begin3
	.quad	.Ltmp129
	.quad	.Ltmp141
	.quad	.Ltmp145
	.quad	.Ltmp148
	.quad	.Ltmp152
	.quad	.Ltmp156
	.quad	.Ltmp157
	.quad	.Ltmp159
	.quad	.Ltmp166
	.quad	.Ltmp167
	.quad	.Ltmp169
	.quad	.Ltmp171
	.quad	.Ltmp172
	.quad	.Ltmp175
	.quad	.Ltmp180
	.quad	.Ltmp181
	.quad	.Ltmp185
	.quad	.Ltmp187
	.quad	.Ltmp210
	.quad	.Ltmp213
	.quad	.Ltmp223
	.quad	.Ltmp225
	.quad	.Ltmp226
	.quad	.Ltmp228
	.quad	.Lfunc_begin4
	.quad	.Ltmp238
	.quad	.Lfunc_begin5
	.quad	.Ltmp241
	.quad	.Ltmp313
	.quad	.Ltmp315
	.quad	.Ltmp316
	.quad	.Ltmp318
	.quad	.Ltmp326
	.quad	.Ltmp328
	.quad	.Ltmp329
	.quad	.Ltmp331
	.quad	.Ltmp333
	.quad	.Ltmp334
	.quad	.Ltmp336
	.quad	.Ltmp339
	.quad	.Ltmp340
	.quad	.Ltmp343
	.quad	.Ltmp345
	.quad	.Ltmp346
	.quad	.Ltmp348
	.quad	.Ltmp349
	.quad	.Ltmp350
	.quad	.Ltmp356
	.quad	.Ltmp445
	.quad	.Ltmp446
	.quad	.Ltmp454
	.quad	.Ltmp456
	.quad	.Ltmp458
	.quad	.Lfunc_begin6
	.quad	.Ltmp465
	.quad	.Ltmp466
	.quad	.Ltmp468
	.quad	.Ltmp484
	.quad	.Ltmp486
	.quad	.Ltmp487
	.quad	.Ltmp489
	.quad	.Ltmp491
	.quad	.Ltmp492
	.quad	.Ltmp494
	.quad	.Ltmp497
	.quad	.Ltmp498
	.quad	.Ltmp501
	.quad	.Ltmp503
	.quad	.Ltmp504
	.quad	.Ltmp506
	.quad	.Ltmp507
	.quad	.Ltmp509
	.quad	.Ltmp511
	.quad	.Ltmp517
	.quad	.Ltmp604
	.quad	.Ltmp605
	.quad	.Ltmp619
	.quad	.Ltmp621
	.quad	.Lfunc_begin7
	.quad	.Ltmp629
.Ldebug_addr_end0:
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5122e9c22abf15a3
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 60
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           attention_fp8_e4m3_fa2_gqa_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     190
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     13
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     56
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           attention_fp8_e4m3_fa2_gqa_partial_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     37
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_partial_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     191
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           attention_fp8_e4m3_fa2_gqa_merge_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_merge_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     17
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 68
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 768
    .name:           attention_fp8_e4m3_fa2_gqa_packet_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     29
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     216
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 768
    .name:           attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     28
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     234
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201.kd
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
	.section	.debug_line,"",@progbits
.Lline_table_start0:
